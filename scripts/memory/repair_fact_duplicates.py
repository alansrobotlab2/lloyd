#!/usr/bin/env python3
"""Remove the exact-duplicate fact rows the write-time guard never cleared (#1144).

Why this script exists
----------------------
#499 (landed ``282a6280``, settled 2026-09-14 09:14Z) refuses a verbatim repeat
at the write, keyed ``(entity, text_hash)`` across EVERY category — but only in
``_fact_add``. Nightly extraction writes fact files through
``scripts/memory/next-gen-memory/fact_extractor.py``, whose ``_merge_facts``
folds by ``(text_hash, expired, invalid)`` *within one file* and never consults
the sibling categories of the same entity. Measured on the live store at
2026-09-21 07:34Z: 5,658 same-entity duplicate groups covering 11,499 active
rows (5,841 redundant), 5,114 of those groups spanning two different files, and
**95 groups whose both copies were created on or after #499 settled** — the
proof that the guard did not stop the leak. The extractor half is closed in the
same round that adds this script; this script is the stock.

Why a repair and not an expiry
------------------------------
The ruling is explicit: remove the losing copies from the markdown, never
expire them. ``facts_idx.exact_duplicate_stats()`` counts expired rows, so
expiring would leave the number this item is tracked on exactly where it was,
and ``_merge_facts`` deliberately keeps an expired copy and an active copy of
one text apart — a retired copy is the record that a claim was retired. #499
also records what deciding a pair by confidence alone did to retrieval:
``fact_entity_recall`` 0.35 → 0.30. So this script never writes
``expired_at``/``invalid_at``, never touches a copy that already carries one,
and refuses to apply against a real store without a before/after retrieval
comparison attached (``--recall-compare``).

What "reversible" means here
----------------------------
Removals are line surgery on the bytes already on disk — never a re-dump of the
parsed YAML, because a re-serialised file could not be put back byte for byte,
and a repair that cannot be undone exactly is not reversible. Every removed
entry is written to a JSONL manifest with its exact source lines, and
``--restore`` replays that manifest into an already-repaired tree to reproduce
the pre-repair files exactly.

    python scripts/memory/repair_fact_duplicates.py                      # dry run, live tree
    python scripts/memory/repair_fact_duplicates.py --apply --manifest /tmp/m.jsonl
    python scripts/memory/repair_fact_duplicates.py --restore /tmp/m.jsonl --apply
    python scripts/memory/repair_fact_duplicates.py --facts-dir COPY --db COPY.sqlite \
        --apply --recall-compare --eval-limit 20                          # the eval gate
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD = HERE.parent.parent
sys.path.insert(0, str(LLOYD))

import yaml  # noqa: E402

from app.atomic_io import atomic_write_text, locked_file  # noqa: E402
from app.kg_store import KGStore, parse_fact_file, text_hash  # noqa: E402
from app.paths import (  # noqa: E402
    PIPELINE_DIR, VAULT_FACTS_ROOT, VAULT_FACTS_ROOT_DEFAULT, VAULT_KG_DB, VAULT_KG_DB_DEFAULT,
)

FENCE = "---"
FACT_COUNT_MARK = "**Fact Count:** "
MANIFEST_DIR = PIPELINE_DIR / "fact-repair"
SCHEMA = "lloyd.fact-duplicate-repair/1"

# The child process the retrieval gate runs. `eval/run_eval.py` is loaded by path
# from the checkout named in argv, so this script never keeps a second copy of the
# scoring code that can fall out of step with it — and so the checkout it loads is
# decided by one argument rather than by whatever `import eval` happens to find
# first. A wrong checkout would produce a plausible score, which is the one failure
# this gate must not be able to hide, so the child also re-reads what it bound and
# refuses if it is not the file it was given.
_RECALL_SNIPPET = r'''
import importlib.util, json, sys
from pathlib import Path
import yaml
root, limit = Path(sys.argv[1]), int(sys.argv[2])
harness = (root / "eval" / "run_eval.py").resolve()
spec = importlib.util.spec_from_file_location("lloyd_run_eval", harness)
if spec is None or spec.loader is None:
    print(json.dumps({"error": f"no importable eval harness at {harness}"}))
    raise SystemExit(1)
m = importlib.util.module_from_spec(spec)
sys.modules["lloyd_run_eval"] = m
spec.loader.exec_module(m)
if Path(str(getattr(m, "__file__", ""))).resolve() != harness:
    print(json.dumps({"error": f"bound run_eval at {getattr(m, '__file__', None)}, "
                               f"not the harness it was given ({harness})"}))
    raise SystemExit(1)
queries = (yaml.safe_load((root / "eval" / "vault_recall_queries.yaml").read_text()) or {}).get("queries") or []
recs = m.run_eval(queries[:limit], limit=limit, expand_graph=True, counterfactual=False)
overall = m.summarize(recs)["overall"]
try:
    corpus = m._corpus_provenance()
except Exception as exc:
    corpus = {"error": str(exc)}
# `harness` is what was asked for; `bound` is what the module object says it is.
# Reported separately so a reader of the run record can see the child's own answer
# rather than an echo of the argument.
corpus = {**corpus, "harness": str(harness), "bound": str(getattr(m, "__file__", ""))}
print(json.dumps({
    "fact_entity_recall_avg": overall.get("fact_entity_recall_avg"),
    "errors": overall.get("errors"),
    "corpus": corpus,
    "records": [{"query": r["query"], "category": r.get("category"),
                 "fact_entities_matched": r["scoring"].get("fact_entities_matched") or []}
                for r in recs],
}))
'''


# ── reading a file as lines, not as re-serialisable objects ─────────────────


def _frontmatter_span(lines: list[str]) -> tuple[int, int] | None:
    """Indices of the opening and closing ``---`` fences, or None."""
    if not lines or lines[0].rstrip("\n") != FENCE:
        return None
    for i in range(1, len(lines)):
        if lines[i].rstrip("\n") == FENCE:
            return 0, i
    return None


def _fact_block_ranges(lines: list[str], span: tuple[int, int]) -> tuple[int, list[tuple[int, int]]]:
    """Locate each entry of the ``facts:`` sequence by line geometry.

    Returns ``(indent, [(start, end_exclusive), ...])`` in document order. The
    sequence ends at the next top-level key (``last_extracted:``,
    ``relationships:``) or at the closing fence.
    """
    open_i, close_i = span
    key_i = None
    for i in range(open_i + 1, close_i):
        if lines[i].rstrip("\n") == "facts:":
            key_i = i
            break
    if key_i is None:
        return 0, []
    indent: int | None = None
    starts: list[int] = []
    stops: list[int] = []
    for i in range(key_i + 1, close_i):
        line = lines[i].rstrip("\n")
        if not line.strip():
            continue
        stripped = line.lstrip(" ")
        is_item = stripped == "-" or stripped.startswith("- ")
        row_indent = len(line) - len(stripped)
        if indent is None:
            if not is_item:
                return 0, []                 # `facts:` is empty, or not a sequence
            indent = row_indent
            starts.append(i)
            continue
        if is_item and row_indent == indent:
            stops.append(i)
            starts.append(i)
            continue
        if row_indent <= indent:
            # A sibling key at the items' own column, or shallower — in the real
            # writer that is `last_extracted:`, the line immediately after the
            # last fact — ends the sequence. The comparison is `<=` and not `<`
            # because `write_fact_file` dumps with no `indent=`, so the dashes of
            # a normal fact file sit at column zero and nothing is ever shallower
            # than them: `<` would never fire, the last block would run to the
            # closing fence, and it would swallow the mapping keys that follow.
            stops.append(i)
            return indent, list(zip(starts, stops))
    if indent is None or not starts:
        return 0, []
    stops.append(close_i)
    return (indent if indent is not None else 2), list(zip(starts, stops))


def _body_section_ranges(lines: list[str], close_i: int) -> list[tuple[str, int, int]]:
    """``(fact_id, start, end_exclusive)`` for each ``### <id>`` section of the body.

    The body is regenerated by the writer from the facts, so a file whose body
    predates its frontmatter simply has no matching heading: recorded as "no
    body section" and left alone rather than half-edited.
    """
    heads = [(i, lines[i][4:].strip()) for i in range(close_i + 1, len(lines))
             if lines[i].startswith("### ")]
    out = []
    for n, (i, fid) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        out.append((fid, i, end))
    return out


def _block_to_entry(block: str, indent: int) -> dict | None:
    """Parse one sequence entry out on its own.

    The ``- `` marker becomes two spaces and every line is dedented by the
    sequence indent, so the entry's own keys all sit at column 2. Folding a long
    fact across lines leaves its continuation at column 2 as well; leaving it at
    column 0 would read as a new key and fail the parse.
    """
    lines = block.splitlines(keepends=True)
    if not lines:
        return None
    out = ["  " + lines[0][indent + 2:]]
    for ln in lines[1:]:
        out.append(ln[indent:] if ln[:indent].strip() == "" else ln)
    try:
        doc = yaml.safe_load("".join(out))
    except yaml.YAMLError:
        return None
    return doc if isinstance(doc, dict) else None


def _dump(entry) -> str:
    """A comparable rendering of one fact entry, tolerant of what YAML yields.

    `yaml.safe_load` hands back ``datetime`` objects for an unquoted timestamp —
    real in the legacy `fact-*.md` and profile-written files — and
    ``json.dumps`` refuses those outright. ``default=str`` keeps the comparison
    total, and both sides of every comparison here go through this function, so
    the two can only differ where the entries genuinely differ.
    """
    return json.dumps(entry, sort_keys=True, default=str)


def _facts_key_line(lines: list[str], span: tuple[int, int]) -> int | None:
    """The line index of the top-level ``facts:`` key, or None.

    Matches the flow-empty forms too (`facts: []`): this is how the repair leaves
    a file whose last copy went, and restore has to recognise the shape it wrote
    in order to reopen it.
    """
    for i in range(span[0] + 1, span[1]):
        if re.fullmatch(r"facts:\s*(\[\])?", lines[i].rstrip("\n")):
            return i
    return None


def _fact_count_line_index(lines: list[str]) -> int | None:
    """The body's ``**Fact Count:** N`` line, or None unless there is exactly one.

    Two of them would mean the body carries more than one rendered copy of this
    file's own header, which is not something a repair should guess about.
    """
    hits = [i for i, ln in enumerate(lines) if ln.startswith(FACT_COUNT_MARK)]
    return hits[0] if len(hits) == 1 else None


def _parse_frontmatter_text(text: str) -> tuple[dict, list | None]:
    """``parse_fact_file`` over a string, to verify bytes that are not on disk yet."""
    if not text.startswith(FENCE):
        return {}, None
    end = text.find(FENCE, 3)
    if end == -1:
        return {}, None
    try:
        fm = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        return {}, None
    facts = fm.get("facts")
    if not isinstance(facts, list):
        return fm, None
    return fm, [f for f in facts if isinstance(f, dict)]


def file_plan(path: Path, root: Path) -> tuple[dict | None, str]:
    """Read one fact file into a repair plan, or say why it cannot be repaired.

    The plan is refused for any file whose line geometry cannot be tied back to
    the parsed frontmatter: an entry parsed from its own block must carry the
    same ``id`` and the same ``fact`` as the entry the loader found. That check
    is what makes the geometry safe to cut on — where it does not hold, the file
    is left completely untouched rather than half-repaired.

    This function does not raise. One file the rest of the tree cannot parse is
    a `skipped` line in the report, not an aborted repair pass: the live tree
    carries blocks a per-block re-parse legitimately fails on, and a run that
    died on the first of them would fix nothing and report nothing.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"unreadable ({exc})"
    fm, facts = parse_fact_file(path)
    facts = [f for f in (facts or []) if isinstance(f, dict)]
    if not facts:
        return None, "no facts"
    lines = text.splitlines(keepends=True)
    try:
        span = _frontmatter_span(lines)
    except Exception as exc:
        return None, f"frontmatter could not be scanned ({type(exc).__name__})"
    if span is None:
        return None, "no frontmatter fence"
    indent, ranges = _fact_block_ranges(lines, span)
    entries = [_block_to_entry("".join(lines[s:e]), indent) for s, e in ranges]
    if [(e.get("id"), e.get("fact")) if e else None for e in entries] != \
            [(f.get("id"), f.get("fact")) for f in facts]:
        return None, "block geometry does not match the parsed facts"

    sections = _body_section_ranges(lines, span[1])
    section_no: dict[str, list[int]] = defaultdict(list)
    for n, (fid, _s, _e) in enumerate(sections):
        section_no[fid].append(n)
    return {
        "path": path,
        "rel": str(path.relative_to(root)),
        "folder": path.parent.name,
        "entity": fm.get("entity") or path.parent.name,
        "category": fm.get("category"),
        "text": text,
        "lines": lines,
        "indent": indent,
        "ranges": ranges,
        "sections": sections,
        # A fact whose id names exactly one body section can have that section
        # removed with it; an ambiguous or missing heading is left in place.
        "section_index": [
            section_no[f.get("id")][0] if len(section_no.get(f.get("id") or "", [])) == 1
            else None for f in facts],
        "facts_key_line": _facts_key_line(lines, span),
        "fact_count_line": _fact_count_line_index(lines),
        "facts": facts,
    }, ""


# ── choosing the survivor ───────────────────────────────────────────────────


def _survivor_key(occ: dict) -> tuple:
    """Ascending sort key; the first row wins. The order the ruling names:

    highest confidence → the copy carrying ``source_doc``/``created_at`` → the
    earliest ``created_at``, then folder/file/entry position as a final
    tiebreak, so one input tree always yields one plan instead of one plan per
    directory order.
    """
    e = occ["entry"]
    conf = e.get("confidence")
    has_conf = isinstance(conf, (int, float)) and not isinstance(conf, bool)
    created = str(e.get("created_at") or "")
    # `source_doc` and `created_at` are one clause's two named fields, but they
    # are not one tier: a copy carrying BOTH holds strictly more than one
    # carrying only a date, and keeping the richer copy is what makes a later
    # removal auditable. Ranking them together would let a dated copy win and
    # retire the one that says where the claim came from.
    meta = (2 if e.get("source_doc") else 0) + (1 if created else 0)
    return (0 if has_conf else 1, -(float(conf) if has_conf else 0.0),
            -meta, 0 if created else 1, created, occ["rel"], occ["seq"])


def _retired(entry: dict) -> bool:
    return bool(entry.get("expired_at")) or bool(entry.get("invalid_at"))


def plan_removals(root: Path, *, folders: set[str] | None = None) -> dict:
    """Every removal this repair would make, plus what it refused to consider.

    Grouping is ``(entity folder, text_hash)`` and deliberately never crosses
    folders: alias folders — ``AgentSkills/`` beside ``Agent Skills/``, 44 groups
    / 45 rows in the 2026-09-14 measurement — are entity resolution's to fuse,
    and deleting one half of a split pair makes the split harder to see.
    """
    plans: dict[str, dict] = {}
    skipped: dict[str, int] = defaultdict(int)
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    scanned = 0

    for folder in sorted(p for p in (root.iterdir() if root.is_dir() else [])
                         if p.is_dir() and not p.name.startswith((".", "_"))):
        if folders and folder.name not in folders:
            continue
        for path in sorted(folder.glob("*.md")):
            scanned += 1
            plan, reason = file_plan(path, root)
            if plan is None:
                # Keyed by file, not by reason: "3 files were skipped" is not
                # actionable, "which 3 and why" is. The report still prints the
                # reason counts.
                skipped[path.relative_to(root).as_posix()] = reason
                continue
            plans[plan["rel"]] = plan
            for seq, entry in enumerate(plan["facts"]):
                by_group[(plan["folder"], text_hash(str(entry.get("fact") or "")))].append({
                    "rel": plan["rel"], "folder": plan["folder"], "seq": seq,
                    "entry": entry, "body_section": plan["section_index"][seq],
                })

    groups = {k: v for k, v in by_group.items() if len(v) > 1}
    mixed = mixed_rows = retired_all = retired_rows = 0
    cross_file_groups = same_file_groups = 0
    removals: list[dict] = []
    for _key, occs in sorted(groups.items()):
        statuses = {_retired(o["entry"]) for o in occs}
        if statuses != {False}:
            # Clause 4: touches no expired or invalid row. A group holding BOTH
            # is left for a person — which copy a superseded claim belongs to is
            # a judgement about the claim, not about the text.
            if len(statuses) > 1:
                mixed += 1
                mixed_rows += len(occs)
            else:
                retired_all += 1
                retired_rows += len(occs)
            continue
        ordered = sorted(occs, key=_survivor_key)
        if len({o["rel"] for o in occs}) > 1:
            cross_file_groups += 1
        else:
            same_file_groups += 1
        removals.extend(ordered[1:])

    removals_by_file: dict[str, list[dict]] = defaultdict(list)
    for occ in removals:
        removals_by_file[occ["rel"]].append(occ)
    for occs in removals_by_file.values():
        occs.sort(key=lambda o: o["seq"])

    return {
        "root": root, "plans": plans, "scanned": scanned, "skipped": dict(skipped),
        "duplicate_groups": len(groups), "cross_file_groups": cross_file_groups,
        "same_file_groups": same_file_groups, "mixed_status_groups": mixed,
        "mixed_status_rows": mixed_rows, "retired_groups": retired_all,
        "retired_rows": retired_rows, "removals": removals,
        "removals_by_file": dict(removals_by_file),
    }


# ── the edit itself ─────────────────────────────────────────────────────────


def _shift(index: int, dropped: set[int]) -> int:
    """Where line `index` lands once `dropped` lines are gone."""
    return index - sum(1 for d in dropped if d < index)


def build_edit(plan: dict, removals: list[dict]) -> tuple[str, list[dict]]:
    """The repaired text plus one manifest record per removal. Pure: nothing is written.

    A record carries where the entry came from, its full entry, its exact source
    lines, and the position to put them back at. ``insert_index`` and
    ``body_insert_index`` count the copies that SURVIVED ahead of the removed
    one, so replaying a file's records in descending ``seq`` and inserting before
    the survivor now standing at that index rebuilds the original order — even
    when two adjacent copies went. The frontmatter blocks and the body sections
    are separate lists with separate indices, which is why both are recorded.
    """
    lines = plan["lines"]
    removed_seqs = {o["seq"] for o in removals}
    removed_sections = {o["body_section"] for o in removals if o["body_section"] is not None}

    dropped: set[int] = set()
    for occ in removals:
        dropped.update(range(*plan["ranges"][occ["seq"]]))
        if occ["body_section"] is not None:
            _fid, bs, be = plan["sections"][occ["body_section"]]
            dropped.update(range(bs, be))

    new_lines = [ln for i, ln in enumerate(lines) if i not in dropped]
    # A file whose last copy went keeps its `facts:` key, emptied to `facts: []`
    # rather than left dangling: `_read_fact_file` in `entity_merge_disposition`
    # does `frontmatter.get("facts", [])`, which hands back the null a bare
    # `facts:` parses to and then iterates it. The repair must not leave a file
    # that turns a later merge into a TypeError.
    emptied = len(removed_seqs) == len(plan["facts"]) and plan["facts_key_line"] is not None
    if emptied:
        pos = _shift(plan["facts_key_line"], dropped)
        had_newline = new_lines[pos].endswith("\n")
        new_lines[pos] = "facts: []" + ("\n" if had_newline else "")
    if plan["fact_count_line"] is not None:
        pos = _shift(plan["fact_count_line"], dropped)
        had_newline = new_lines[pos].endswith("\n")
        n_after = len(plan["facts"]) - len(removals)
        new_lines[pos] = f"{FACT_COUNT_MARK}{n_after}" + ("\n" if had_newline else "")
    new_text = "".join(new_lines)

    _verify_edit(plan, removed_seqs, new_text)

    records = []
    for occ in removals:
        start, end = plan["ranges"][occ["seq"]]
        text = str(occ["entry"].get("fact") or "")
        body_lines = None
        body_insert = None
        if occ["body_section"] is not None:
            _fid, bs, be = plan["sections"][occ["body_section"]]
            body_lines = lines[bs:be]
            body_insert = sum(1 for n in range(occ["body_section"]) if n not in removed_sections)
        records.append({
            "schema": SCHEMA,
            "folder": plan["folder"],
            "entity": plan["entity"],
            "category": plan["category"],
            "file_path": plan["rel"],
            "seq": occ["seq"],
            "insert_index": sum(1 for i in range(occ["seq"]) if i not in removed_seqs),
            "entry_indent": plan["indent"],
            "fact_id": occ["entry"].get("id"),
            "text_hash": text_hash(text),
            "entry": occ["entry"],
            "block_lines": lines[start:end],
            "body_section_index": occ["body_section"],
            "body_insert_index": body_insert,
            "body_section_lines": body_lines,
            "fact_count_line": None,
        })
    if plan["fact_count_line"] is not None:
        payload = {"index": plan["fact_count_line"],
                   "text": lines[plan["fact_count_line"]]}
        for rec in records:
            rec["fact_count_line"] = payload
    return new_text, records


def _verify_edit(plan: dict, removed_seqs: set[int], new_text: str) -> None:
    """Prove the edit before writing it. Raises instead of writing.

    The repaired frontmatter has to parse, and its facts must be exactly the
    original list minus the removed entries, in the original order. That single
    comparison carries the rest: this repair never re-serialises, so a survivor
    whose bytes moved at all shows up here as a mismatch. A repair that silently
    reordered or re-dated a fact it was not asked to touch would be a worse
    defect than the duplicates it is clearing.
    """
    _fm, facts_after = _parse_frontmatter_text(new_text)
    if facts_after is None:
        raise ValueError("repaired frontmatter does not parse")
    want = [f for i, f in enumerate(plan["facts"]) if i not in removed_seqs]
    if [json.dumps(f, sort_keys=True) for f in facts_after] != \
            [json.dumps(f, sort_keys=True) for f in want]:
        raise ValueError("repaired facts are not the survivors in their original order")


def _duplicate_pairs(db: Path) -> dict:
    """The numbers this item is tracked on, read the way the item reads them.

    Both denominators are reported because they answer different questions: the
    active-only count is what a repair can move, while
    ``exact_duplicate_stats()`` is what the health report trends, and includes
    expired and invalid rows by design.
    """
    st = KGStore(db)
    try:
        active = st._query(
            "SELECT COUNT(*) g, COALESCE(SUM(n - 1), 0) r FROM (SELECT entity, text_hash, "
            "COUNT(*) n FROM facts_idx WHERE expired_at IS NULL AND invalid_at IS NULL "
            "GROUP BY entity, text_hash HAVING COUNT(*) > 1)")[0]
        stats = st.facts_idx.exact_duplicate_stats()
        distinct = st._query(
            "SELECT COUNT(*) n FROM (SELECT entity, text_hash FROM facts_idx "
            "GROUP BY entity, text_hash)")[0]
        return {"active_groups": int(active["g"]),
                "active_redundant_rows": int(active["r"]),
                "same_entity_groups": stats["same_entity_groups"],
                "same_entity_redundant_rows": stats["same_entity_redundant_rows"],
                "distinct_entity_text_pairs": int(distinct["n"])}
    finally:
        st.close()


# ── the retrieval gate ──────────────────────────────────────────────────────


def _child_env() -> dict:
    keep = {k: v for k, v in os.environ.items()
            if k.startswith(("CLAUDE", "ANTHROPIC", "CUDA", "PATH", "HOME", "USER",
                             "TMPDIR", "QMD", "XDG"))}
    keep.setdefault("PYTHONUNBUFFERED", "1")
    return keep


def run_recall(root: Path, facts_root: Path, db: Path, limit: int) -> dict:
    """Score the real retrieval eval against one fact tree and store.

    ``root`` is the CHECKOUT holding ``eval/run_eval.py`` — the harness — and is
    never the corpus: the tree and store being scored arrive through the
    environment. The distinction is load-bearing and the failure was silent, so it
    is checked here rather than discovered in the child's traceback.

    Run as a child process with ``LLOYD_FACTS_ROOT``/``LLOYD_KG_DB`` pointed at
    the copy, which is the override ``app.paths`` exists for. Not in-process: a
    second ``kg_store.configure()`` would retarget every other reader in this
    process, including the caller's own measurement, and a gate that reads the
    tree it is repairing cannot tell a repair from a reindex.
    """
    harness = root / "eval" / "run_eval.py"
    if not harness.is_file():
        raise FileNotFoundError(
            f"no eval harness at {harness}: the retrieval gate's child loads "
            f"eval/run_eval.py from the checkout named in its argv, so the harness "
            f"root must be a Lloyd checkout — the fact tree being scored is a "
            f"separate argument and holds no eval/ directory")
    proc = subprocess.run(
        [sys.executable, "-c", _RECALL_SNIPPET, str(root), str(limit)],
        capture_output=True, text=True, timeout=3600, cwd=str(root),
        env={**_child_env(), "LLOYD_FACTS_ROOT": str(facts_root), "LLOYD_KG_DB": str(db)},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"recall comparison failed (exit {proc.returncode}): "
                           f"{(proc.stderr or proc.stdout)[-800:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["facts_root"] = str(facts_root)
    payload["kg_db"] = str(db)
    return payload


def _entity_losses(before: dict, after: dict) -> list[dict]:
    """Queries whose expected-entity set shrank, from the eval's own comparison.

    ``count_overreach_regressions`` is imported, not restated: it is the
    function the nightly chain already uses to tell a retirement that was a
    correction from one that was an over-reach (#1250 — two copies of a guard is
    how two sides come to disagree about whether an arm measured anything).
    """
    try:
        from eval import run_eval as m
    except Exception:                                       # pragma: no cover - fallback below
        m = None
    if m is not None and hasattr(m, "count_overreach_regressions"):
        def wrap(recs):
            return [{"query": r["query"],
                     "scoring": {"fact_entities_matched": r.get("fact_entities_matched") or []}}
                    for r in recs]
        return m.count_overreach_regressions(wrap(before.get("records") or []),
                                             wrap(after.get("records") or []))
    after_by = {r["query"]: set(r.get("fact_entities_matched") or [])
                for r in after.get("records") or []}
    out = []
    for r in before.get("records") or []:
        had = set(r.get("fact_entities_matched") or [])
        gone = sorted(had - after_by.get(r["query"], set()))
        if gone:
            out.append({"query": r["query"], "category": r.get("category"), "entities": gone})
    return out


def gate_verdict(before: dict, after: dict) -> tuple[bool, list[str]]:
    """May this repair touch a real store? Reasons why not, when it may not.

    ``fact_entity_recall_avg`` must not fall and no query may lose an expected
    entity. An arm that never measured is NOT a pass: a null average is what an
    unreadable fact leg looks like, and certifying that would certify anything.
    """
    reasons: list[str] = []
    b, a = before.get("fact_entity_recall_avg"), after.get("fact_entity_recall_avg")
    if b is None or a is None:
        reasons.append(f"fact_entity_recall_avg was not measured (before={b}, after={a})")
    elif a < b:
        reasons.append(f"fact_entity_recall_avg fell {b} → {a}")
    for label, arm in (("before", before), ("after", after)):
        if arm.get("errors"):
            reasons.append(f"{label} run reported {arm['errors']} query errors")
    losses = _entity_losses(before, after)
    if losses:
        reasons.append("queries lost an expected entity: "
                       + "; ".join(f"{l['query']} ({', '.join(l['entities'])})" for l in losses[:5]))
    return (not reasons, reasons)


# ── apply / restore ─────────────────────────────────────────────────────────


def apply_removals(plans: dict, removals_by_file: dict, manifest: Path, *,
                   apply: bool) -> dict:
    """Write the removals. Default is a dry run; ``--apply`` writes.

    The order inside one file is deliberate: verify the new bytes, append the
    manifest and fsync it, and only then move the file into place. A crash
    between the two leaves a manifest line with no edit, which is harmless; the
    reverse would leave an edit with no manifest, the one state that makes a
    removal unrecoverable.

    A dry run does not open the manifest at all — not even to truncate it. The
    manifest is the only thing that can put removed rows back, so a survey that
    opened the path in write mode would destroy the record of a real repair just by
    pointing at it, and a survey that leaves an empty file behind hands out a
    document describing nothing, whose ``--restore`` reads as a clean no-op.
    """
    handle = None
    if apply:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        # Appended, never truncated: passes may share one manifest, and an apply
        # that clobbered its own earlier lines would orphan those rows for good.
        handle = manifest.open("a", encoding="utf-8")
    done: list[dict] = []
    refused: list[dict] = []
    try:
        for rel in sorted(removals_by_file):
            occs = removals_by_file[rel]
            plan = plans[rel]
            try:
                new_text, records = build_edit(plan, occs)
            except Exception as exc:                         # noqa: BLE001 - refuse the file
                refused.append({"file": rel, "refused": f"edit verification failed: {exc}"})
                continue
            if not apply:
                done.extend({"file": rel, "removed": occ["entry"].get("id")} for occ in occs)
                continue
            outcome = _write_one(plan["path"], plan["text"], new_text, records, handle)
            if isinstance(outcome, dict):
                refused.append({"file": rel, **outcome})
            else:
                done.extend(outcome)
    finally:
        if handle is not None:                                # a dry run opens nothing
            handle.close()
    return {"applied_files": len({d["file"] for d in done}),
            "removed_rows": len(done), "refused": refused, "manifest": str(manifest)}


def _write_one(path: Path, expected_text: str, new_text: str, records: list[dict],
               handle) -> list[dict] | dict:
    """One file, under its lock: re-read, prove nothing moved, manifest, write."""
    with locked_file(path):
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return {"refused": f"unreadable under lock ({exc})"}
        if current != expected_text:
            # The plan was surveyed unlocked, exactly like `repair_fact_ids.py`
            # does: if the file changed since, the line numbers in the plan may
            # now name a different fact. Refusing is the only honest answer.
            return {"refused": "file changed since the plan was built; re-run"}
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        atomic_write_text(path, new_text, fsync=True)
    return [{"file": str(path), "removed": rec["fact_id"]} for rec in records]


def restore(manifest_path: Path, *, facts_root: Path, apply: bool) -> dict:
    """Replay a manifest into a tree, putting every removed entry back.

    Byte-for-byte is the requirement, so the payload is the file's own source
    lines rather than a re-dump of the entry — a re-dump could not reproduce a
    folded scalar or a key order, and a repair that cannot be undone exactly is
    not reversible.
    """
    recs = [json.loads(ln) for ln in manifest_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    by_file: dict[str, list[dict]] = defaultdict(list)
    for rec in recs:
        by_file[rec["file_path"]].append(rec)

    restored: list[str] = []
    missing: list[str] = []
    refused: list[dict] = []
    for rel, file_recs in sorted(by_file.items()):
        path = facts_root / rel
        if not path.exists():
            missing.append(rel)
            continue
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            refused.append({"file": rel, "refused": f"unreadable ({exc})"})
            continue
        expected = _restore_text(current, file_recs)
        if isinstance(expected, tuple):
            refused.append({"file": rel, "refused": expected[0]})
            continue
        if apply:
            with locked_file(path):
                atomic_write_text(path, expected, fsync=True)
        restored.append(rel)
    return {"manifest": str(manifest_path), "files": len(by_file),
            "restored": len(restored), "missing": missing, "refused": refused,
            "mode": "apply" if apply else "dry run"}


def _restore_text(text: str, recs: list[dict]) -> str | tuple[str]:
    """Put one file back, or say why this tree is not the repaired output.

    A manifest replayed against a tree it does not belong to would splice a fact
    into the wrong file and report success, so the rebuilt list is checked
    entry by entry: every restored record must land at its own ``seq`` carrying
    the entry the manifest says it is.
    """
    span = _frontmatter_span(text.splitlines(keepends=True))
    if span is None:
        return ("no frontmatter fence",)
    fm, facts_before = _parse_frontmatter_text(text)
    if facts_before is None:
        return ("frontmatter does not parse",)
    lines = text.splitlines(keepends=True)

    for rec in sorted(recs, key=lambda r: -r["seq"]):
        want = rec.get("entry") or {}
        for field, value in (("entity", fm.get("entity")), ("category", fm.get("category"))):
            if rec.get(field) is not None and value is not None and rec[field] != value:
                return (f"this file's {field} is {value!r}, the manifest's record "
                        f"says {rec[field]!r}",)
        plan_indent, ranges = _fact_block_ranges(lines, span)
        recorded = rec.get("entry_indent")
        # The indent the block was written at, from the manifest — not read back
        # off the file. `indent or 2` is the trap: PyYAML puts a top-level
        # sequence's dashes at column 0, so a real 0 becomes 2 and every restored
        # line is silently mis-dedented.
        indent = plan_indent if recorded is None else recorded
        if not ranges:
            # Every copy in this file went, so the repair left the key standing
            # empty (`facts: []`, see build_edit) rather than dangling it. Only
            # that exact shape is reopened: any other empty value is something
            # this script did not write and will not overwrite.
            key = _facts_key_line(lines, span)
            if key is None or lines[key].rstrip("\n") not in ("facts: []", "facts:"):
                return ("no facts: sequence to restore into",)
            lines[key] = "facts:" + ("\n" if lines[key].endswith("\n") else "")
            span = _frontmatter_span(lines) or span
            pos = key + 1
        else:
            pos = (ranges[-1][1] if rec["insert_index"] >= len(ranges)
                   else ranges[rec["insert_index"]][0])
        block = rec["block_lines"]
        entry = _block_to_entry("".join(block), indent)
        if entry is None:
            return ("restored block does not parse",)
        if _dump(entry) != _dump(want):
            return (f"restored block is {_dump(entry)[:120]!r}, the manifest recorded "
                    f"{_dump(want)[:120]!r}",)
        lines[pos:pos] = block
        span = _frontmatter_span(lines) or span

    _fm, facts_after = _parse_frontmatter_text("".join(lines))
    if facts_after is None:
        return ("restored frontmatter does not parse",)
    for rec in recs:
        got = facts_after[rec["seq"]] if rec["seq"] < len(facts_after) else None
        if _dump(got) != _dump(rec["entry"]):
            return (f"entry at seq {rec['seq']} is not the manifest's copy",)

    body = _restore_body(lines, span[1], recs)
    if isinstance(body, tuple):
        return body
    lines = body

    # The header's own count is the one line the repair rewrote rather than
    # deleted, so no removed block carries it back; it comes from the record.
    count_recs = [r for r in recs if r.get("fact_count_line")]
    if count_recs:
        payload = max(count_recs, key=lambda r: r["seq"])["fact_count_line"]
        hits = [i for i, ln in enumerate(lines) if ln.startswith(FACT_COUNT_MARK)]
        if len(hits) != 1:
            return ("no single **Fact Count:** line to restore",)
        lines[hits[0]] = payload["text"]
    return "".join(lines)


def _restore_body(lines: list[str], close_i: int, recs: list[dict]) -> list[str] | tuple[str]:
    for rec in sorted([r for r in recs if r.get("body_section_lines")],
                      key=lambda r: -r["seq"]):
        idx = rec["body_insert_index"]
        if idx is None:
            continue
        sections = _body_section_ranges(lines, close_i)
        pos = len(lines) if idx >= len(sections) else sections[idx][1]
        lines[pos:pos] = rec["body_section_lines"]
        span = _frontmatter_span(lines)
        if span is None:
            return ("restored body lost the frontmatter fence",)
        close_i = span[1]
    return lines


# ── CLI ─────────────────────────────────────────────────────────────────────


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Remove exact-duplicate fact rows the write-time guard never cleared")
    ap.add_argument("--facts-dir", type=Path, default=VAULT_FACTS_ROOT)
    ap.add_argument("--db", type=Path, default=VAULT_KG_DB)
    ap.add_argument("--apply", action="store_true",
                    help="write; default is a dry run. Refused against a real store "
                         "without --recall-compare.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="JSONL manifest; defaults under _pipeline/fact-repair/")
    ap.add_argument("--restore", type=Path, default=None, metavar="MANIFEST",
                    help="replay a manifest back into --facts-dir (--apply to write)")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--limit-entities", type=int, default=0,
                    help="survey only the first N duplicated entities (a copy of the tree)")
    ap.add_argument("--recall-compare", action="store_true",
                    help="run the retrieval eval before and after, against this tree/store")
    ap.add_argument("--eval-limit", type=int, default=20)
    ap.add_argument("--allow-legacy-store", action="store_true",
                    help=f"apply without the gate against a store predating #499 "
                         f"({VAULT_KG_DB_DEFAULT}), which has no duplicate group to repair")
    ap.add_argument("--skip-reindex", action="store_true",
                    help="leave facts_idx describing rows that no longer exist")
    args = ap.parse_args(argv)

    root = Path(args.facts_dir)
    if not root.is_dir():
        print(f"no fact tree at {root}", file=sys.stderr)
        return 2
    is_live = root.resolve() == Path(VAULT_FACTS_ROOT_DEFAULT).resolve()
    manifest = args.manifest or (MANIFEST_DIR / f"fact-dupes-{_stamp()}.jsonl")

    # Named, not just printed: `--facts-dir` decides what this run writes, and #700
    # is the record of what an artifact that could not say which tree it described
    # cost. `VAULT_FACTS_ROOT_DEFAULT` is that tree with the env override removed —
    # a copy certifying itself as production is the failure that note is about.
    print(f"tree: {root} ({'live default tree' if is_live else 'a copy, not the default'})"
          f"\ndb:   {args.db}\nmanifest: {manifest}")

    if args.restore:
        if not args.restore.exists():
            print(f"no manifest at {args.restore}", file=sys.stderr)
            return 2
        res = restore(args.restore, facts_root=root, apply=args.apply)
        print(f"{'restored' if args.apply else 'would restore'} {res['restored']} of "
              f"{res['files']} files ({len(res['missing'])} missing, "
              f"{len(res['refused'])} refused)")
        if res["missing"] or res["refused"]:
            print("restore was not complete — the tree does not match this manifest",
                  file=sys.stderr)
            for r in res["refused"][:5]:
                print(f"  {r['file']}: {r['refused']}", file=sys.stderr)
            return 1
        if not args.apply:
            print("(dry run — pass --apply to write)")
        return 0

    before_store = _duplicate_pairs(args.db) if args.db.exists() else None
    if before_store:
        print(f"before: {before_store['active_groups']:,} active duplicate groups, "
              f"{before_store['active_redundant_rows']:,} redundant active rows "
              f"(index says {before_store['same_entity_redundant_rows']:,} including retired)")

    planned = plan_removals(root)
    if args.limit_entities:
        keep = sorted({o["folder"] for o in planned["removals"]})[:args.limit_entities]
        planned = plan_removals(root, folders=set(keep))
    entities = {o["folder"] for o in planned["removals"]}
    print(f"scanned {planned['scanned']:,} files; {len(entities):,} entities hold a duplicate")
    print(f"  duplicate groups                       {planned['duplicate_groups']:,}")
    print(f"    across files (cross-category shape)  {planned['cross_file_groups']:,}")
    print(f"    inside one file                      {planned['same_file_groups']:,}")
    print(f"  rows to remove                         {len(planned['removals']):,}")
    print(f"  left alone: mixed live+retired "
          f"{planned['mixed_status_groups']:,} groups ({planned['mixed_status_rows']:,} rows), "
          f"all-retired {planned['retired_groups']:,} groups ({planned['retired_rows']:,} rows)")
    reasons = defaultdict(int)
    for reason in planned["skipped"].values():
        reasons[reason] += 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  skipped file ({reason}): {n:,}")

    if not args.apply:
        print("\n(dry run — pass --apply to write)")
        _report(args, planned, {"would_remove_rows": len(planned["removals"])})
        return 0

    # Clause 5: `--apply` without the retrieval comparison is refused, always,
    # and loudly — a silent fallback to "no gate" is how a guard dies.
    if not args.recall_compare:
        print("\nREFUSED: --apply requires --recall-compare. Removing thousands of fact "
              "rows is only admissible with a before/after vault_recall comparison in "
              "which fact_entity_recall_avg does not fall and no query loses an expected "
              "entity (#499 records 0.35 → 0.30 from retiring the wrong half of a pair). "
              "Point --facts-dir/--db at a copy of the tree and store, or pass "
              "--allow-legacy-store for a store that predates #499.", file=sys.stderr)
        return 1
    if not args.allow_legacy_store and not (before_store and before_store["active_groups"]):
        print("\nREFUSED: the store holds no same-entity duplicate group at all, so a "
              "before/after comparison could not move and would certify anything. Pass "
              "--allow-legacy-store if this tree knowingly predates #499.", file=sys.stderr)
        return 1

    # The harness and the corpus are separate arguments and coincide only by
    # accident. `eval/run_eval.py` lives in the checkout running this script; the
    # corpus is the tree and store the child is handed through the environment.
    # Deriving the harness root from the corpus argument — which this line did —
    # pointed the child at `<facts-tree>/eval/run_eval.py` for a live-tree run,
    # where the derived facts tree holds no harness, so the gate died on its first
    # call on the one run it exists to serve (#1144 review, seam finding).
    recall_root = LLOYD
    print(f"\nretrieval gate: {args.eval_limit} queries, harness {recall_root}, "
          f"corpus {root} + {args.db}")
    before_recall = run_recall(recall_root, root, args.db, args.eval_limit)
    print(f"  before: fact_entity_recall_avg={before_recall['fact_entity_recall_avg']}")

    result = apply_removals(planned["plans"], planned["removals_by_file"], manifest, apply=True)
    print(f"applied: {result['removed_rows']:,} rows removed from "
          f"{result['applied_files']:,} files")
    for r in result["refused"][:5]:
        print(f"  refused {r['file']}: {r['refused']}")

    after_store = dict(before_store or {})
    if not args.skip_reindex and args.db.exists():
        st = KGStore(args.db)
        try:
            st.facts_idx.reindex(root=root)
        finally:
            st.close()
        after_store = _duplicate_pairs(args.db)
        print(f"after:  {after_store['active_groups']:,} active duplicate groups, "
              f"{after_store['active_redundant_rows']:,} redundant active rows")
        print(f"        distinct (entity, text_hash) pairs "
              f"{before_store['distinct_entity_text_pairs']:,} → "
              f"{after_store['distinct_entity_text_pairs']:,}")
    else:
        print("after:  facts_idx NOT reindexed (--skip-reindex or no db)", file=sys.stderr)

    after_recall = run_recall(recall_root, root, args.db, args.eval_limit)
    print(f"  after:  fact_entity_recall_avg={after_recall['fact_entity_recall_avg']}")
    ok, reasons = gate_verdict(before_recall, after_recall)
    if not ok:
        print("\nREGRESSION: retrieval fell after the repair. Put it back with:", file=sys.stderr)
        print(f"  {sys.executable} {Path(__file__)} --facts-dir {root} --db {args.db} "
              f"--restore {manifest} --apply", file=sys.stderr)
        for why in reasons:
            print(f"  - {why}", file=sys.stderr)
        return 1
    print("gate: fact_entity_recall_avg did not fall and no query lost an expected entity")

    _report(args, planned, {**result, "before": before_store, "after": after_store,
                            "recall_before": _brief(before_recall),
                            "recall_after": _brief(after_recall)})
    return 0


def _brief(recall: dict) -> dict:
    return {k: recall.get(k) for k in
            ("fact_entity_recall_avg", "errors", "facts_root", "kg_db")}


def _report(args, planned: dict, extra: dict) -> None:
    if not args.report:
        return
    payload = {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(),
        "facts_dir": str(args.facts_dir), "db": str(args.db),
        "scanned_files": planned["scanned"],
        "duplicate_groups": planned["duplicate_groups"],
        "cross_file_groups": planned["cross_file_groups"],
        "same_file_groups": planned["same_file_groups"],
        "mixed_status_groups": planned["mixed_status_groups"],
        "retired_groups": planned["retired_groups"],
        "removed_rows": len(planned["removals"]),
        "removed": [{"file_path": o["rel"], "fact_id": o["entry"].get("id"),
                     "fact": o["entry"].get("fact")} for o in planned["removals"]],
        **extra,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"report: {args.report}")


if __name__ == "__main__":
    sys.exit(main())
