#!/usr/bin/env python3
"""Move the `general` god entity's fact tree out of the facts tree (#2443, after #2303).

`facts/general/` holds 365 facts in 8 `.md` files (7 category files + `general-overview.md`)
plus 7 zero-byte `.lock` files, 248K, and `knowledge-health-2026-10-08.md:43` still ranks it
9th: `| general | 365 | decision, event, general, goal, preference, relationship, state |`.
Nothing files there any more — #1999 closed that route at `70c2d002` ("facts with no usable
entity are held back and counted, never filed under general") — so the residue is static, and
it is still a god entity: 56 ACTIVE edges name `general`, `entities.get("general")` answers a
row (`kind: concept`, created 2026-09-23T04:24:54Z), and `facts_idx.count(entity="general")`
is 365 with every row active. A name every category appears under answers no query about
anything, so retrieval pays a god node's cost for no lookups.

#2303 quarantined the rows by hand and left the tree; this is the mover it never wrote. What it
does is the tree move plus the two graph consequences, and nothing else:

  * `facts/general/` is RELOCATED, whole, to `<vault-derived-root>/facts-quarantine-<run-tag>/general/`
    — the root `kg_rebuild.py:911` already names and `facts-quarantine-20260923T070837Z` already
    occupies beside the facts tree. Not deleted: the fact text is the only copy, and a move that
    can be read back the next day is the difference between a quarantine and a data loss.
  * every active edge naming it is expired through `kg.edges.expire(edge_id, reason)`
    (`app/kg_store.py:729`), its indexed rows leave the table with the files through
    `facts_idx.reindex(paths, root=…)`, and its `entities` row goes with the existing
    `kg.entities.remove(name)` (`app/kg_store.py:1067`) — no new store API, since this one has
    existed all along. So the entity stops being a node rather than merely stopping being a
    directory.

It does NOT touch `content-hashes.json` and does NOT re-extract the 32 source documents. That
half of #2303's original contract is dropped: 4/32 documents are hashed at all, 31/32 sit
outside the mtime gate, and 803 rows from those same documents are already filed under real
entities, so a re-extract would re-file facts that are already filed and rewrite a hash file to
trigger it (#2443's owed clause 4 asks a person whether it is ever wanted).

**Counting is by the TWO-space prefix.** A fact line in these files is a YAML mapping key
inside a list item, so it is indented two spaces — `- entity: ''` then `  fact: …`. On the live
tree 2026-10-08 `grep -hc '^  fact:' general/*.md` is 365 while `'^ fact:'` and `'^fact:'` are
both 0. #2443's clause 1 as written named the one-space spelling, which matches nothing; a
counter built on it prints 0, and then clause 3's "post-move total equals the pre-move total"
passes at 0 == 0 on a run that dropped every fact. So `FACT_LINE_RE` below pins `^ {2}fact:`,
`tests/test_quarantine_general_facts.py` seeds at least two fact lines, and `--apply` refuses a
tree whose fact count reads zero (`--allow-zero-facts` overrides) rather than performing a move
whose only losslessness check is vacuous.

Dry-run by default; `--apply` moves and writes. Every path is resolved through `app.paths`
(`VAULT_FACTS_ROOT` is the `LLOYD_FACTS_ROOT` override, `app/paths.py:236`), and all graph state
goes through `app.kg_store`, which is the only writer of `kg.sqlite` — this script opens no
database connection of its own.

    quarantine_general_facts.py                    # read-only plan
    quarantine_general_facts.py --apply            # the move + the graph writes
    quarantine_general_facts.py --apply --json     # for a job to parse

Running it against the live root is the OPERATOR's step, not this round's: the move lands under
`~/lloyd-data/_pipeline/`, outside both the repo and the vault, so no automod round may perform
it. What lands here is the script and its tests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD = HERE.parent.parent
for _p in (str(LLOYD), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app import kg_store, paths  # noqa: E402
from app.paths import VAULT_KG_DB  # noqa: E402

#: The god entity this mover exists for (#2303). One entity per run, so the report
#: names one directory and one entity throughout; nothing here takes a list.
ENTITY = "general"

#: The quarantine-root naming the KG maintenance scripts already use
#: (`kg_rebuild.py:911`, documented at `expire_numeric_entity_facts.py:323` and
#: `link_stranded_entities.py:409`). A second, differently-spelled quarantine
#: root would be a second thing every later reader of `vault-derived/` has to
#: learn — and it is the root the 09-23 precedent already occupies.
QUARANTINE_PREFIX = "facts-quarantine-"

#: A fact line, as the fact files actually write one: the `fact:` key of a mapping
#: inside a `- ` list item, indented TWO spaces. See the module docstring — the
#: one-space spelling matches 0 of the 365 live lines, and counting with it makes
#: the pre/post equality check vacuous at 0 == 0.
FACT_LINE_RE = re.compile(r"^ {2}fact:", re.M)

#: Names the item that decided this and the item landing the mover. A reason that
#: says "quarantined" without a number is unreadable in six months.
DEFAULT_REASON = ("#2303: facts misfiled under the `general` god entity, quarantined out of "
                  "the facts tree and its graph edges expired (#2443)")


# ── reading the tree ─────────────────────────────────────────────────────────

def resolve_facts_root() -> Path:
    """The tree to act on: `$LLOYD_FACTS_ROOT` if set, else `app.paths.VAULT_FACTS_ROOT`.

    `app/paths.py:236` applies that override, so the fallback is the canonical answer and is
    what an unset variable must produce — the test asserts the two agree when the variable is
    unset, which is what stops this function's own read from silently diverging from the
    canonical resolver if the variable is ever renamed there.

    The variable is re-read here instead of taken from an import-time binding on purpose:
    `app.paths` resolves it at import, and a mover holding a value frozen at import cannot be
    pointed at a tree by anything that sets the variable after the interpreter started — a
    wrapper script, an autonomy job's environment, or a test. Both spellings name the same
    directory whenever the variable is set before the run, which is every real invocation.
    """
    override = os.environ.get("LLOYD_FACTS_ROOT")
    return Path(override) if override else Path(paths.VAULT_FACTS_ROOT)


def fact_line_count(entity_dir: Path) -> int:
    """Total two-space `fact:` lines across every `.md` under `entity_dir`.

    Read with `errors="replace"` like the other markdown readers here
    (`_invocation.py`, `kg_store.parse_fact_file`): one unreadable byte must not turn a
    count of 365 into an exception, and must not silently drop a file from the total
    either — a file that fails to read has to contribute zero lines AND be visible in
    the report, which the md-file count beside it is what makes legible.
    """
    return sum(len(FACT_LINE_RE.findall(p.read_text(encoding="utf-8", errors="replace")))
               for p in sorted(Path(entity_dir).glob("*.md")))


def md_file_count(entity_dir: Path) -> int:
    return len(sorted(Path(entity_dir).glob("*.md")))


def lock_file_count(entity_dir: Path) -> int:
    return len(sorted(Path(entity_dir).glob("*.lock")))


def tree_digest(root: Path) -> tuple[int, str]:
    """`(file count, sha256 over the whole tree)` — the measurement behind "changed nothing".

    A dry run claiming it touched the tree is a claim about bytes, and "no error was raised"
    is not that measurement. This walks `root`, folds each file's relative path and contents
    into one hash, so a rewrite, a rename, an added file and a deleted file all move the
    number. The dry run reports it twice — before and after its own work — and `tests` compare
    the two, so "no quarantine directory and every file byte-identical" is read off the tree
    rather than inferred from the absence of a traceback.
    """
    root = Path(root)
    if not root.exists():
        return (0, "absent-tree")
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        n += 1
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(p.read_bytes())
    return (n, h.hexdigest()[:16])


# ── the plan ─────────────────────────────────────────────────────────────────

def entity_dir_for(facts_root: Path, entity: str = ENTITY) -> Path:
    return Path(facts_root) / entity


def quarantine_root_for(facts_root: Path, run_tag: str) -> Path:
    """`<facts_root>/../facts-quarantine-<run-tag>`, whose parent is the vault-derived root.

    The quarantine root is the SIBLING of the facts tree, which is where the live precedent
    sits (`facts-quarantine-20260923T070837Z` beside `facts/`) and what `kg_rebuild.py:911`
    names. It is derived from the facts root rather than from `app.paths.VAULT_DERIVED_ROOT`
    on purpose: when `LLOYD_FACTS_ROOT` points the mover at another tree, the quarantine has
    to land beside THAT tree. A root imported from config would move a temp tree's contents
    into the live `~/lloyd-data/_pipeline/` — the one failure mode this script must not have.
    For the default facts root the two spellings name the same directory.
    """
    return Path(facts_root).resolve().parent / f"{QUARANTINE_PREFIX}{run_tag}"


def run_tag(now: dt.datetime | None = None) -> str:
    """The run tag: `%Y%m%dT%H%M%SZ` in UTC, the shape `kg_rebuild.py:910` writes.

    UTC because the tag carries a `Z`, and a naive local stamp inside a name that reads as
    UTC is the #1525 class of bug: every later reader takes it as the same instant it is not.
    """
    return (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def files_to_clear(kg, facts_root: Path, entity: str = ENTITY
                   ) -> tuple[list[Path], list[str]]:
    """`(every file the quarantine takes, stored paths that escape the facts root)`.

    The index's own rows for the entity, unioned with the `.md` files physically in the
    entity directory: a row whose file was already removed still has to leave the table, and
    a file the index never read (or read as another entity, since
    `kg_store._rows_for_file` lets a fact override its entity from inside a neighbour's
    file) still has to be moved. Both have to be captured BEFORE the move — after it the
    directory is gone and the on-disk half of the union is silently empty, which is how a
    quarantine would leave live rows pointing into the quarantine root.

    A stored `file_path` that resolves above the facts root — a `..`, or an absolute path a
    rebuild left behind — is reported instead of being fed to `reindex`, which would delete
    the rows of a file outside the tree this run was pointed at.
    """
    root = Path(facts_root).resolve()
    files: list[Path] = []
    seen: set[str] = set()
    escaped: list[str] = []

    def add(candidate: Path) -> None:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            files.append(candidate)

    for row in kg.facts_idx.for_entity(entity, include_expired=True):
        rel = (row.get("file_path") or "").strip()
        if not rel:
            continue
        candidate = root / rel
        try:
            candidate.resolve().relative_to(root)
        except ValueError:
            escaped.append(rel)
            continue
        add(candidate)
    for p in sorted(entity_dir_for(root, entity).glob("*.md")):
        add(p)
    return files, escaped


# ── the move and the writes ──────────────────────────────────────────────────

def quarantine(kg, facts_root: Path, *, apply: bool = False, ts: str | None = None,
               reason: str = DEFAULT_REASON, entity: str = ENTITY,
               allow_zero_facts: bool = False) -> dict:
    """Report the quarantine, and perform it when `apply`. Returns the report dict.

    The report always carries the pre-move fact count; `fact_lines_after_move` exists only
    when a move happened, because a dry run has no after-figure to print.
    """
    facts_root = Path(facts_root)
    entity_dir = entity_dir_for(facts_root, entity)
    tag = ts or run_tag()
    dest = quarantine_root_for(facts_root, tag) / entity

    exists = entity_dir.is_dir()
    before_lines = fact_line_count(entity_dir) if exists else 0
    files, escaped = files_to_clear(kg, facts_root, entity)

    rep: dict = {
        "entity": entity,
        "entity_dir": str(entity_dir),
        "entity_dir_exists": exists,
        "md_files": md_file_count(entity_dir) if exists else 0,
        "lock_files": lock_file_count(entity_dir) if exists else 0,
        "fact_lines_before_move": before_lines,
        "run_tag": tag,
        "quarantine_dest": str(dest),
        "apply": bool(apply),
        "moved": False,
        "moved_files": 0,
        "fact_lines_after_move": None,
        "entity_dir_still_exists": False,
        "files_to_clear": len(files),
        "reindexed_paths": 0,
        "escaped_paths": escaped,
        "edges_active_before": 0,
        "edges_expired": 0,
        "entities_row_before": False,
        "entities_row_removed": False,
        "facts_idx_rows_before": kg.facts_idx.count(entity=entity, active_only=False),
        "facts_idx_rows_after": None,
        "refused": None,
        "tree_files_before": None,
        "tree_digest_before": None,
        "tree_files_after": None,
        "tree_digest_after": None,
    }
    if facts_root.exists():
        rep["tree_files_before"], rep["tree_digest_before"] = tree_digest(facts_root)

    # Graph state is read whether or not the directory is there, because the two halves of a
    # quarantine can disagree: the tree can be gone with the edges still live, and that is
    # exactly the state a report must not print as clean.
    active = kg.edges.active(either=entity)
    rep["edges_active_before"] = len(active)
    rep["entities_row_before"] = kg.entities.get(entity) is not None

    if not apply:
        rep["nothing_written"] = True
        if facts_root.exists():
            rep["tree_files_after"], rep["tree_digest_after"] = tree_digest(facts_root)
        return rep

    if exists and before_lines == 0 and not allow_zero_facts:
        # The vacuous-pass guard. `general-overview.md` legitimately holds 0 fact lines, so a
        # zero total is possible for a real tree — but only for a tree the counter cannot
        # read, which is precisely the one-space-spelling mistake #2443's clause 1 contained.
        # Moving on a 0 would run the losslessness check as 0 == 0, so the run refuses and
        # says which of the two it is looking at.
        rep["refused"] = ("zero-fact-count: the two-space prefix matched none of the "
                          f"{rep['md_files']} .md files, so the pre/post comparison would "
                          f"pass at 0 == 0. Check the fact-line spelling, or pass "
                          f"--allow-zero-facts to move anyway.")
        return rep

    if exists:
        if dest.exists():
            # Never merge into a quarantine that already holds a copy. `shutil.move` onto an
            # existing directory nests it as `general/general`, and the operator finds the
            # tree one level deeper than the report said.
            rep["refused"] = f"quarantine destination already exists: {dest}"
            return rep
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entity_dir), str(dest))
        rep["moved"] = True
        rep["moved_files"] = sum(1 for p in dest.rglob("*") if p.is_file())
        rep["fact_lines_after_move"] = fact_line_count(dest)
        rep["entity_dir_still_exists"] = entity_dir.exists()

    if files:
        # Every row the index attributes to those files leaves the table with them:
        # `reindex(paths, root=…)` deletes by `file_path` and re-reads only a file that
        # still exists, and after the move none of these do. Scoped to these paths rather
        # than a whole-tree `reindex()`, which re-walks every entity directory in the tree
        # and re-registers each one it sees — including, if a move ever failed half-way, the
        # one being quarantined. `register_entities=False` for the same reason: a mover must
        # never re-register the entity it is about to deregister.
        kg.facts_idx.reindex(files, root=facts_root, register_entities=False)
        rep["reindexed_paths"] = len(files)
    rep["facts_idx_rows_after"] = kg.facts_idx.count(entity=entity, active_only=False)

    for edge in active:
        if kg.edges.expire(edge["id"], reason):
            rep["edges_expired"] += 1

    # After the reindex, for the reason the clause gives it: `reindex` registers every
    # entity directory it sees, so deregistering first would have the index put the row
    # straight back.
    rep["entities_row_removed"] = bool(kg.entities.remove(entity))
    if facts_root.exists():
        rep["tree_files_after"], rep["tree_digest_after"] = tree_digest(facts_root)
    return rep


def writes_disabled_by_rebuild() -> bool:
    """True while a knowledge-graph rebuild holds writes off the tree (#1833's guard).

    `kg_rebuild.py` flips `knowledge_graph.write_enabled` false and extracts into a parallel
    tree it can rename `facts-quarantine-<ts>` into at any moment — and this mover renames the
    facts tree itself, into a root of exactly that name. Two components renaming the same
    directory in one window is how a rebuild's output and a quarantine end up inside each
    other. Read, and failed open, the way every other writer here does: a missing config is
    not a rebuild, `--apply` is explicit, and the dry run is always safe.
    """
    from app.config import CONFIG
    try:
        return not bool(CONFIG.get("knowledge_graph", {}).get("write_enabled", True))
    except Exception:
        return False


# ── reporting ────────────────────────────────────────────────────────────────

def print_report(rep: dict) -> None:
    mode = "APPLY" if rep["apply"] else "dry run"
    print(f"quarantine `{rep['entity']}` — {mode}")
    print(f"  entity dir            {rep['entity_dir']}  "
          f"({'present' if rep['entity_dir_exists'] else 'ABSENT — nothing to move'})")
    print(f"  .md / .lock files     {rep['md_files']} / {rep['lock_files']}")
    print(f"  fact lines (2-space)  {rep['fact_lines_before_move']}")
    print(f"  would relocate to     {rep['quarantine_dest']}")
    print(f"  active edges naming   {rep['edges_active_before']}")
    print(f"  entities row          {'present' if rep['entities_row_before'] else 'absent'}")
    print(f"  indexed fact rows     {rep['facts_idx_rows_before']}")
    if rep["tree_digest_before"] is not None:
        print(f"  facts tree            {rep['tree_files_before']} files  "
              f"sha256:{rep['tree_digest_before']}")
        if rep["tree_digest_after"] is not None:
            same = (rep["tree_digest_before"] == rep["tree_digest_after"]
                    and rep["tree_files_before"] == rep["tree_files_after"])
            print(f"  facts tree after      {rep['tree_files_after']} files  "
                  f"sha256:{rep['tree_digest_after']}  "
                  f"{'UNCHANGED' if same else 'CHANGED'}")
    if rep["moved"]:
        print(f"  MOVED                 {rep['moved_files']} files -> "
              f"{rep['quarantine_dest']}")
        print(f"  fact lines before move {rep['fact_lines_before_move']}")
        print(f"  fact lines after move  {rep['fact_lines_after_move']}")
        if rep["fact_lines_before_move"] != rep["fact_lines_after_move"]:
            print(f"  MISMATCH              the relocated tree holds "
                  f"{rep['fact_lines_after_move']} of {rep['fact_lines_before_move']} "
                  f"fact lines — the move was not lossless")
        if rep["entity_dir_still_exists"]:
            print("  STILL PRESENT         facts/" + rep["entity"] + "/ survived the move")
    elif rep["apply"]:
        print("  nothing moved         the entity directory was already absent")
    if rep["facts_idx_rows_after"] is not None:
        print(f"  indexed rows after    {rep['facts_idx_rows_after']}")
    if rep["apply"]:
        print(f"  edges expired         {rep['edges_expired']}  "
              f"(reason names #2303)")
        print(f"  entities row          "
              f"{'removed' if rep['entities_row_removed'] else 'not removed'}")
    if not rep["apply"] and (rep["entity_dir_exists"] or rep["edges_active_before"]
                             or rep["entities_row_before"]):
        print(f"dry run: {rep['fact_lines_before_move']} fact lines, "
              f"{rep['edges_active_before']} active edges and "
              f"{'an' if rep['entities_row_before'] else 'no'} entities row would go. "
              f"Pass --apply.")
    if rep["refused"]:
        print(f"REFUSED: {rep['refused']}")
    for rel in rep["escaped_paths"]:
        print(f"  stored path escapes the facts root, NOT de-indexed: {rel}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="move the tree and write the graph. Without this the run "
                         "only reports.")
    ap.add_argument("--facts-root", default=None,
                    help=f"fact tree to quarantine from (default {resolve_facts_root()}, "
                         f"which is $LLOYD_FACTS_ROOT when that is set)")
    ap.add_argument("--kg-db", default=None,
                    help=f"store to expire edges in (default {VAULT_KG_DB})")
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--reason", default=DEFAULT_REASON)
    ap.add_argument("--ts", default=None, help="run tag (default: now, UTC)")
    ap.add_argument("--allow-zero-facts", action="store_true",
                    help="move a tree whose two-space fact-line count is 0. Without "
                         "this --apply refuses one, because its own losslessness check "
                         "would be 0 == 0.")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    if args.apply and writes_disabled_by_rebuild():
        print("writes are disabled (knowledge_graph.write_enabled = false), which "
              "kg_rebuild.py sets while a rebuild runs and while its output can be "
              "renamed into a facts-quarantine-<ts> root. Re-run --apply after it "
              "finishes; the dry run is always safe.")
        return 2

    facts_root = Path(args.facts_root) if args.facts_root else resolve_facts_root()
    if not facts_root.is_dir():
        # A facts root that is not there answers "0 files, 0 facts" to every question,
        # which is the reading of an already-clean tree. A typo'd --facts-root or an
        # LLOYD_FACTS_ROOT pointing nowhere must not report that.
        print(f"no facts tree at {facts_root}; refusing to report a count from a "
              f"directory that is not there")
        return 2

    if args.kg_db:
        if not Path(args.kg_db).is_file():
            # `kg_store.configure()` provisions an absent path, so a typo would hand this
            # run an empty store whose report — 0 edges, no entities row — is the exact
            # shape of a quarantine already done.
            print(f"no knowledge-graph database at {args.kg_db}; refusing to report "
                  f"a count from a store that is not there")
            return 2
        kg = kg_store.configure(Path(args.kg_db))
    else:
        kg = kg_store.store()

    rep = quarantine(kg, facts_root, apply=args.apply, ts=args.ts, reason=args.reason,
                     entity=args.entity, allow_zero_facts=args.allow_zero_facts)
    if args.as_json:
        print(json.dumps(rep, default=str))
    else:
        print_report(rep)
    return 1 if rep["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
