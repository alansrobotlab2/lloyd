#!/usr/bin/env python3
"""#2258: does K=3 theory fan-out diagnose better than one linear trace?

Cleric's grounding technique (Pienaar, AI Engineer 2026-10-05) says an agent
investigating an incident should generate several disjoint candidate root causes
BEFORE reading any log, investigate each independently, and pick among them by
RELATIVE ranking — because agents rank a candidate set well but score one
candidate's absolute confidence badly. Lloyd's diagnosis turns do the opposite:
`systematic-debugging/SKILL.md:211` is literally "Form a Single Hypothesis".
This module is the instrument that measures the difference on Lloyd's own
record. It decides nothing itself: it builds the corpus, records runs, and
re-grades them. Whether arm B wins is the study's output, owed to the owed-check
job after this lands.

    python eval/diagnosis_fanout_replay.py corpus    [--ledger …] [--learnings …] [--out …]
    python eval/diagnosis_fanout_replay.py run       [--only KEY] [--arms A,A-retry,B] [--ceiling N]
    python eval/diagnosis_fanout_replay.py baseline  [--rows …] [--out …]
    python eval/diagnosis_fanout_replay.py report    [--cases …] [--rows …] [--baseline …] [--root …]
    python eval/diagnosis_fanout_replay.py witness   [--path …]

`witness` re-derives the ledger figures this item quotes from the COMMITTED bytes
(`~/obsidian/backlog/data/2026-10-06.2258-ledger-witness.jsonl`, vault commit
`5c00fdb5`), because the live ledger under `~/.local/state` has no history and
grows with every gate — it measured 23,388, then 23,389, then 23,413 across one
working session, while the triage's 23,329 was already stale when written. The
witness reports `wc -l` 2621, 585 reason-bearing rows, and the event census
round_aborted 283 / round_abandoned 224 / red_tree_filed 94 / red_tree_closed 26
/ alert 55 / vault_revert 8 / rollback_succeeded 2, and rebuilding `corpus` over
it yields the same 98 cases. A row-count mismatch exits non-zero, because those
are not the bytes the study was built on.

The witness is an EXTRACT, not the whole ledger: every row of the seven quoted
event classes (so those counts are complete) plus every row of each round the
case builder cites. It excludes the rest, so its 2,621 rows are never quotable as
a row count of the live ledger. It is a dated sibling because the vault's own
rails retire the wholesale-copy path in `backlog/data/`:
`tests/test_failure_ledger_witness.py:82` asserts that path stays absent and
`tests/test_automod_vault_round.py::test_no_reader_under_tests_or_scripts_opens_the_retired_mirror`
refuses even a new file that names it (#2054 clause 5; the retention sweep retired
it as a 33,113,707-byte copy, and #2243 settled the dated-sibling route).

`corpus` reads two sources and nowhere else: the automod promotion ledger
(`promotions.jsonl`, structured fields only — a claim and its refutation both
already sit in the ledger's own fields) and the memory learnings tree (a
markdown table row whose cells name a retracted claim and its correction). A
case is complete only when it carries `input`, `claimed_cause`, `actual_cause`,
`source_ref` and at least one `actual_cause_marker`; **below 10 complete cases
`corpus` prints why and writes no artifact**, because a smaller corpus is a
different study and padding a case from prose is how a fake result gets made.

`run` is the only subcommand that needs the engine, and it is the owed study,
not this round: arm A is today's single trace, `A-retry` the compute-matched
retry baseline (#627's rule, so a win is not "more tokens"), arm B three
theories generated before any evidence row, one sandboxed `bench_` turn per
theory returning claims with citations, then `djev_rank` over them — argmax
read-out only, never a score threshold, per #1452.

`report` needs no engine: it re-grades recorded rows. Each row is appended to
`rows.jsonl` under the data root (`<data>/eval/2258/`), and the retry baseline
is a cached file keyed by the corpus fingerprint, its N, the model and the
token ceiling — written by `baseline` from the retry arm's own recorded rows, so
the study's order is `run --arms A-retry`, `baseline`, `report`. Missing or
mismatched, the headline prints `cannot evaluate` and the command exits 3, so a
study that never measured its own baseline can never read as a pass.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

LLOYD_HOME = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LLOYD_HOME))

ITEM_ID = 2258
OUT_DIR = Path.home() / "lloyd-data" / "eval" / str(ITEM_ID)
CASES_PATH = OUT_DIR / "cases.jsonl"
ROWS_PATH = OUT_DIR / "rows.jsonl"
BASELINE_PATH = OUT_DIR / "retry_baseline.json"

#: The automod promotion ledger (`scripts/automod/state.py:67`). Its rows are
#: machine-written records, so a claim and its refutation can be joined on
#: fields rather than read out of prose.
LEDGER_PATH = Path.home() / ".local" / "state" / "lloyd-automod" / "promotions.jsonl"
#: The committed extract of that ledger (`automod_vault_land`, vault commit
#: `5c00fdb5`). The live file has no history and grows by the hour, so every
#: figure #2258 quotes is re-derived from these bytes: `wc -l
#: ~/obsidian/backlog/data/2026-10-06.2258-ledger-witness.jsonl` is 2621, and
#: `witness` below recomputes the census from it. Dated, because the vault
#: retires the wholesale-copy path (see the module docstring) — do not "fix"
#: this by pointing it at `backlog/data/promotions.jsonl`.
VAULT_WITNESS = (Path.home() / "obsidian" / "backlog" / "data"
                 / "2026-10-06.2258-ledger-witness.jsonl")
#: `wc -l` of the witness as committed. Pinned by a test so a corpus rebuilt
#: over different bytes cannot quietly inherit the quoted figures.
WITNESS_ROWS = 2_621
#: Vault commit holding those bytes. It arrived in another job's commit through
#: the shared vault tree (`5c00fdb5`, whose message carries the #2175
#: "unattributed dirty state" list naming this path), verified byte-identical by
#: sha256 against the working file rather than by authorship.
WITNESS_SHA = "5c00fdb5"
#: Cases the witness rebuilds. The study's corpus fingerprint is stable only
#: because this number is: over the live ledger the count moves as rounds accrue.
WITNESS_CASES = 98
#: Daily-note tree holding the retraction tables (`| n | area | 1 (retracted) |
#: "<claim>" — correction |`). The only prose source, and only its table cells.
LEARNINGS_ROOT = Path.home() / "obsidian" / "memory" / "learnings"

MIN_CASES = 10
THEORIES_PER_CASE = 3
#: The item's own acceptance bar, carried here so the printed headline states the
#: pre-registered numbers rather than a fresh judgement.
TOP1_MARGIN = 2
CITATION_BAR = 0.90
ARMS = ("A", "A-retry", "B")
ARM_A, ARM_RETRY, ARM_B = "A", "A-retry", "B"
#: Matched token ceiling per case-arm, shared by every theory in arm B, so the
#: fan-out has to buy its extra dispatches out of the same budget.
DEFAULT_TOKEN_CEILING = 120_000


class RecordError(Exception):
    """A row that would not prove what the clause says it proves."""


# ── corpus ──────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row.setdefault("_line", lineno)
            # The file, not just the line: a `source_ref` must name the file the
            # number resolves in, and the default ledger is now the committed
            # witness, whose name is not `promotions.jsonl`.
            row.setdefault("_name", path.name)
            out.append(row)
    return out


def _short(sha: Any) -> str:
    return str(sha or "")[:8]


def _case_tests_blamed(round_id: str, rows: list[dict]) -> dict | None:
    """Ledger family 1: the suite went red, the round was blamed, the base probe
    then reproduced the same failures with the round's diff absent.

    Both halves are ledger fields — `red_tree_filed.node_ids` is the blame,
    the later `tests` rung's `pre_existing_failures` is the refutation — so
    nothing here is inferred from prose.
    """
    filed = [r for r in rows if r.get("event") == "red_tree_filed" and r.get("node_ids")]
    refuted = [r for r in rows
               if r.get("event") == "gate" and r.get("rung") == "tests"
               and r.get("pre_existing_failures")]
    if not filed or not refuted:
        return None
    blame = min(filed, key=lambda r: r.get("ts") or 0)
    blamed = set(blame["node_ids"])
    after = [r for r in refuted
             if (r.get("ts") or 0) > (blame.get("ts") or 0)
             and blamed & set(r.get("pre_existing_failures") or [])]
    if not after:
        return None
    ref = min(after, key=lambda r: r.get("ts") or 0)
    overlap = sorted(blamed & set(ref.get("pre_existing_failures") or []))
    base = _short(ref.get("base") or blame.get("base"))
    tracked = ref.get("red_tree_item")
    return {
        "key": f"preexisting-{round_id}",
        "family": "tests-blamed-then-refuted",
        "source": "ledger",
        "subsystem": "single-service",
        "input": (f"automod round {round_id} (base {base}): the suite came back red on "
                  f"{', '.join(overlap)}. Is this round's diff the cause?"),
        "claimed_cause": (f"{round_id}'s own diff caused {len(overlap)} test failure(s): "
                          f"{', '.join(overlap)}"),
        "actual_cause": (f"pre-existing at base {base}: {', '.join(overlap)} reproduce with "
                         f"this round's diff absent"
                         + (f" (tracked by item #{tracked})" if tracked else "")),
        "actual_cause_markers": ["pre-existing", base],
        "source_ref": (f"{ref.get('_name') or LEDGER_PATH.name}:{blame['_line']}+{ref['_line']} "
                       f"(round {round_id})"),
    }


def _case_flaky(round_id: str, rows: list[dict]) -> dict | None:
    """Ledger family 2: failures that only appear under the parallel run.

    `flaky_node_ids` is written by the rung after it re-ran the file serially,
    so the refutation is a field, not a guess.
    """
    hits = [r for r in rows
            if r.get("event") == "gate" and r.get("rung") == "tests"
            and r.get("flaky_node_ids")]
    if not hits:
        return None
    ref = min(hits, key=lambda r: r.get("ts") or 0)
    names = sorted(set(ref["flaky_node_ids"]))
    base = _short(ref.get("base") or "")
    return {
        "key": f"flaky-{round_id}",
        "family": "parallel-load-flake",
        "source": "ledger",
        "subsystem": "test-harness",
        "input": (f"automod round {round_id} (base {base}): {len(names)} test(s) failed "
                  f"under the parallel run — {', '.join(names[:3])}"
                  f"{' and others' if len(names) > 3 else ''}. Is the diff broken?"),
        "claimed_cause": f"{len(names)} test(s) fail on this round's diff: {', '.join(names)}",
        "actual_cause": (f"the same {len(names)} node(s) passed when their files were "
                         f"re-run serially: parallel-load flakiness, not the diff"),
        "actual_cause_markers": ["serial"],
        "source_ref": f"{ref.get('_name') or LEDGER_PATH.name}:{ref['_line']} (round {round_id})",
    }


def _case_landing_killed(round_id: str, rows: list[dict]) -> dict | None:
    """Ledger family 3: a landing that did not promote because its own process
    was killed, which reads identically to a change that cannot be promoted."""
    hits = [r for r in rows
            if r.get("event") == "land_failed" and r.get("killed_by_signal")]
    if not hits:
        return None
    ref = min(hits, key=lambda r: r.get("ts") or 0)
    sig = ref["killed_by_signal"]
    return {
        "key": f"landkill-{round_id}",
        "family": "landing-killed-externally",
        "source": "ledger",
        "subsystem": "single-service",
        "input": (f"automod round {round_id}: the promotion wrote nothing and the tree "
                  f"is unchanged. Did the change fail to promote?"),
        "claimed_cause": f"{round_id}'s change could not be promoted",
        "actual_cause": (f"the landing process was killed by signal {sig} before it "
                         f"promoted anything — an external block on the process, not a "
                         f"verdict on the change"),
        "actual_cause_markers": [f"signal {sig}"],
        "source_ref": f"{ref.get('_name') or LEDGER_PATH.name}:{ref['_line']} (round {round_id})",
    }


#: One case per round, from the first family whose fields support it: two
#: families over the same round would be one incident counted twice.
_LEDGER_FAMILIES = (_case_tests_blamed, _case_flaky, _case_landing_killed)

_TABLE_ROW = re.compile(r"^\|(?:[^|]*\|){3}\s*(?P<cell>[^|]*?)\s*\|\s*$")
_RETRACTION = re.compile(r"(retracted|self-caught)", re.I)
_QUOTED = re.compile(r"[\u201c\"]([^\u201d\"]{20,})[\u201d\"]")
_BACKTICK = re.compile(r"`([^`]{3,60})`")


def learnings_cases(root: Path) -> list[dict]:
    """The learnings-tree source: a retraction table row naming a claim and its
    correction.

    Only markdown table rows whose last cell carries a quoted claim, a
    retraction marker and at least one backticked identifier qualify — the
    backticked token is the named mechanism, and a row without one states a
    reversal with no mechanism, which cannot be scored. Keyed by file and line
    so a re-run of the builder is reproducible.

    It has yielded **0 cases every time it has been run** (measured 2026-10-06
    over the whole tree: the retraction-bearing lines that exist do not also
    carry a quoted claim in the same cell). That is a fact about the corpus, not
    a defect to chase here, and `cmd_corpus` prints this source's count on every
    run so the shortfall is on the record rather than inferred from a total.
    """
    out: list[dict] = []
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.md")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            matched = _TABLE_ROW.match(line.strip())
            if not matched:
                continue
            cell = matched.group("cell")
            if not cell or not _RETRACTION.search(cell):
                continue
            claim = _QUOTED.search(cell)
            markers = _BACKTICK.findall(cell)
            if not claim or not markers:
                continue
            actual = cell[cell.lower().index("retract" if "retract" in cell.lower()
                                           else "self-caught"):].split(":", 1)
            actual_text = actual[1].strip() if len(actual) > 1 else cell
            rel = path.relative_to(root).as_posix()
            out.append({
                "key": f"learnings-{rel.replace('/', '-').replace('.md', '')}-{lineno}",
                "family": "retracted-recorded-cause",
                "source": "learnings",
                "subsystem": "multi-subsystem",
                "input": f"A prior Lloyd run reported: {claim.group(1)}",
                "claimed_cause": claim.group(1),
                "actual_cause": actual_text,
                "actual_cause_markers": markers[:3],
                "source_ref": f"memory/learnings/{rel}:{lineno}",
            })
    return out


def case_complete(case: dict) -> bool:
    """The four clause-1 fields plus at least one marker to score against."""
    return all(str(case.get(f) or "").strip()
               for f in ("input", "claimed_cause", "actual_cause", "source_ref")) \
        and bool([m for m in case.get("actual_cause_markers") or [] if str(m).strip()])


def build_cases(ledger_path: Path = LEDGER_PATH,
                learnings_root: Path = LEARNINGS_ROOT) -> tuple[list[dict], list[str]]:
    """Assemble the corpus. Returns (complete cases, drop notes).

    Cases are keyed, and a key seen twice is one incident recorded twice — the
    second copy is dropped rather than inflating N.
    """
    by_round: dict[str, list[dict]] = {}
    for row in _read_jsonl(ledger_path):
        rid = str(row.get("round_id") or "")
        if rid:
            by_round.setdefault(rid, []).append(row)

    notes: list[str] = []
    seen: set[str] = set()
    cases: list[dict] = []
    for round_id, rows in sorted(by_round.items()):
        for family in _LEDGER_FAMILIES:
            case = family(round_id, rows)
            if case is None:
                continue
            if not case_complete(case):
                notes.append(f"{case['key']}: incomplete")
                break
            if case["key"] in seen:
                notes.append(f"{case['key']}: duplicate key")
                break
            seen.add(case["key"])
            cases.append(case)
            break

    for case in learnings_cases(learnings_root):
        if not case_complete(case):
            notes.append(f"{case['key']}: incomplete")
        elif case["key"] in seen:
            notes.append(f"{case['key']}: duplicate key")
        else:
            seen.add(case["key"])
            cases.append(case)
    return cases, notes


def write_cases(cases: list[dict], out_path: Path,
                *, min_cases: int = MIN_CASES) -> tuple[int, str]:
    """Write the corpus, or refuse and write nothing. Returns (exit, message).

    The refusal is the whole point of the `min_cases` floor: below it the study
    is a different study, so an artifact must not exist to be read.
    """
    complete = [c for c in cases if case_complete(c)]
    if len(complete) < min_cases:
        return 1, (f"cannot evaluate: only {len(complete)} complete case(s) "
                   f"(floor {min_cases}) — no artifact written. The committed "
                   f"witness rebuilds {WITNESS_CASES}; a ledger that yields fewer "
                   f"is not the corpus the study was scoped on")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for case in complete:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return 0, f"wrote {len(complete)} case(s) to {out_path}"


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return [c for c in _read_jsonl(path) if "key" in c]


# ── citation verification ───────────────────────────────────────────────────

def _line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def verify_citation(citation: dict, *, root: Path | None = None) -> dict:
    """Resolve one citation against the filesystem; never raise on a bad one.

    A citation is `{path, line?, value?, timestamp?}`; `path` may also be given
    as `"some/file.py:123"`, which is the form a transcript actually contains.
    The verdict is per citation: `ok` plus a `reason` naming what was checked.
    A path that does not exist fails `no-such-path`; an existing file whose line
    123 does not exist fails `line-out-of-range`. `root` relocates the whole
    check, which is how a fixture tree proves both failures.
    """
    raw = str(citation.get("path") or citation.get("file") or "").strip()
    line = citation.get("line")
    if line is None and ":" in raw:
        # `"app/loop.py:123"` is the form a transcript actually contains. The
        # cost of accepting it is a directory or file whose last `:`-suffix is
        # all digits being read as a line — which fails loudly here
        # (`line-out-of-range` or `no-such-path`), never as a pass.
        head, _, tail = raw.rpartition(":")
        if tail.isdigit():
            raw, line = head, int(tail)
    base = Path(raw).expanduser()
    resolved = (root / base) if (root is not None and not base.is_absolute()) else base
    out = {"path": str(base), "line": line, "ok": False, "reason": ""}
    if not str(base):
        out["reason"] = "no-path"
        return out
    if not resolved.exists():
        out["reason"] = "no-such-path"
        return out
    if line is None:
        out["ok"] = True
        out["reason"] = "path-exists"
        return out
    try:
        line_no = int(line)
    except (TypeError, ValueError):
        out["reason"] = "line-not-a-number"
        return out
    if not resolved.is_file():
        out["reason"] = "not-a-file"
        return out
    if line_no < 1 or line_no > _line_count(resolved):
        out["reason"] = "line-out-of-range"
        return out
    out["ok"] = True
    out["reason"] = "line-resolves"
    return out


def citation_rows(rows: Sequence[dict]) -> list[dict]:
    """Every citation carried by any recorded investigation row."""
    out = []
    for row in rows:
        for claim in row.get("claims") or []:
            for cite in claim.get("citations") or []:
                if isinstance(cite, dict):
                    out.append(cite)
                else:
                    out.append({"path": str(cite)})
    return out


def verify_citations(citations: Sequence[dict], *,
                     root: Path | None = None) -> dict:
    """Per-citation verdicts plus the rate the item's bar is measured against."""
    verdicts = [verify_citation(c, root=root) for c in citations]
    passed = sum(1 for v in verdicts if v["ok"])
    return {"n": len(verdicts), "passed": passed,
            "rate": (passed / len(verdicts)) if verdicts else 0.0,
            "verdicts": verdicts}


# ── arm records ─────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_theories(theories: Sequence[Any]) -> list[str]:
    """Exactly `THEORIES_PER_CASE` disjoint candidate causes, or no record."""
    texts = [str(t or "").strip() for t in theories]
    if len(texts) != THEORIES_PER_CASE:
        raise RecordError(
            f"arm B records exactly {THEORIES_PER_CASE} candidate causes, got {len(texts)}")
    if len({t.lower() for t in texts}) != len(texts):
        raise RecordError("the candidate causes are not disjoint; a duplicate theory is "
                          "pure fan-out cost")
    return texts


def validate_investigation(row: dict) -> list[dict]:
    """A per-theory row must carry claims with citations, not a conclusion.

    Pienaar's fourth failure class is a sub-agent handing back a prose summary;
    the ranking is only over evidence if the record cannot hold anything else.
    """
    claims = row.get("claims")
    if not isinstance(claims, list) or not claims:
        raise RecordError(f"theory {row.get('slot')!r} handed back no claims — "
                          "a prose conclusion is not evidence to rank")
    out = []
    for claim in claims:
        text = str((claim or {}).get("claim") or "").strip()
        cites = (claim or {}).get("citations")
        if not text or not isinstance(cites, list) or not cites:
            raise RecordError(f"theory {row.get('slot')!r} has a claim with no citation: "
                              f"{str(text)[:80]!r}")
        out.append({"claim": text,
                    "citations": [c if isinstance(c, dict) else {"path": str(c)}
                                  for c in cites]})
    return out


class RowWriter:
    """Append-only recorder that owns the sequence numbers per (case, arm)."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, row: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


# ── the ranking step (clause 4) ─────────────────────────────────────────────

def slot_labels(case_key: str, n: int = THEORIES_PER_CASE) -> list[str]:
    """Neutral, non-semantic option names, rotated per case.

    #1452 measured that djev's option-name prior grows with the number of
    options, so a slot must never be named for its theory ("the-cert-expired")
    and the same cause must not keep the same slot across cases — `slot 0` of a
    corpus that is always the right answer is a prior djev can learn. The
    rotation is derived from the case key, so a re-run rotates identically.
    """
    digest = hashlib.sha256(case_key.encode()).hexdigest()
    offset = int(digest[:8], 16) % n
    return [f"c{(offset + i) % n}" for i in range(n)]


def rank_candidate_text(slot: str, claims: Sequence[dict]) -> str:
    """The text djev ranks for one slot: its evidence only, under a neutral name.

    The theory's own wording is deliberately absent — it is the thing whose
    label would carry the prior — and the citations are what make the ranking
    a ranking over evidence.
    """
    lines = [f"[{slot}]"]
    for claim in claims:
        cites = ", ".join(f"{c.get('path')}"
                          + (f":{c['line']}" if c.get("line") else "")
                          for c in claim.get("citations") or [])
        lines.append(f"- {claim.get('claim')} ({cites})")
    return "\n".join(lines)


def rank_theories(case: dict, evidence: dict[str, list[dict]], *,
                 slots: Sequence[str],
                 rank_fn: Callable[[str, Sequence[str]], Any]) -> dict:
    """Rank the fan-out and persist the argmax order, never a score.

    `rank_fn` is the production `app.djev.rank` in a live run and a stub in a
    test. Only its ORDER is read: the first row it returns is the choice, so a
    score of 0.01 and a score of 0.99 do the same thing, and nothing here
    compares a score to a cutoff (#1452: argmax only, the scores are not
    calibrated probabilities).

    Candidates go to the engine sorted by slot name, not in generation order.
    djev scores candidates by POSITION, so leaving the first theory generated
    in position 0 for every case would hand it a standing advantage — the
    rotation in `slot_labels` only moves the prior if the position moves with
    the label.
    """
    by_slot = {s: (evidence.get(s) or []) for s in slots}
    pairs = sorted(by_slot.items(), key=lambda item: item[0])
    candidates = [rank_candidate_text(slot, claims) for slot, claims in pairs]
    query = (f"Which candidate root cause does the collected evidence support for: "
             f"{case.get('input') or ''}")
    rows = rank_fn(query, candidates)
    if rows is None or (rows is not None and not isinstance(rows, list)):
        return {"kind": "rank", "ok": False,
                "reason": "engine-unusable: rank returned no ordering "
                          "(a null or non-list answer is not a tie)",
                "order": [], "chosen_slot": None, "chosen_cause": None}
    if not rows:
        return {"kind": "rank", "ok": False,
                "reason": "engine-unusable: rank returned an empty ordering",
                "order": [], "chosen_slot": None, "chosen_cause": None}
    order = []
    for ranked in rows:
        idx = ranked.get("index")
        if isinstance(idx, int) and 0 <= idx < len(pairs):
            order.append(pairs[idx][0])
    chosen = order[0] if order else None
    return {"kind": "rank", "ok": True, "order": order, "chosen_slot": chosen,
            "chosen_cause": None}


def theory_for(slot: str, theories: Sequence[dict]) -> str:
    for t in theories:
        if t.get("slot") == slot:
            return str(t.get("theory") or "")
    return ""


# ── the two arms ────────────────────────────────────────────────────────────

async def run_case_b(case: dict, *, generate: Callable[[dict], Awaitable[Sequence[str]]],
                     investigate: Callable[[dict, str], Awaitable[dict]],
                     rank_fn: Callable[[str, Sequence[str]], Any],
                     writer: RowWriter, run_id: str,
                     token_ceiling: int = DEFAULT_TOKEN_CEILING,
                     slots: Sequence[str] | None = None) -> dict:
    """Arm B for one case: three theories first, evidence second, ranking last.

    The order is the clause, not an implementation detail: `generate` runs and
    its row is written before `investigate` is ever awaited, so the theories
    cannot have read a log. Spend is tracked against `token_ceiling` — the same
    ceiling arm A gets — and the fan-out stops there. A theory's cost is only
    known once it returns, so the arm can refuse the NEXT dispatch and not the
    one in flight: the bound is "stop at the ceiling, overshoot by at most one
    theory", and the refusal is recorded as a row so the study can see the fan-
    out was cut short rather than reading three theories that never ran.
    """
    case_key = case["key"]
    seq = 0

    def next_seq() -> int:
        nonlocal seq
        seq += 1
        return seq

    theories = validate_theories(await generate(case))
    labels = list(slots or slot_labels(case_key))
    theory_rows = [{"slot": s, "theory": t} for s, t in zip(labels, theories)]
    writer.append({"seq": next_seq(), "run_id": run_id, "case_key": case_key,
                   "arm": ARM_B, "kind": "theories", "ts": _now(),
                   "theories": theory_rows, "generated_before_evidence": True})

    evidence: dict[str, list[dict]] = {s: [] for s in labels}
    spent = 0
    stopped = None
    for entry in theory_rows:
        if spent >= token_ceiling:
            stopped = "token-ceiling"
            break
        result = await investigate(case, entry["theory"])
        claims = validate_investigation({**result, "slot": entry["slot"]})
        tokens = int(result.get("tokens") or 0)
        spent += tokens
        evidence[entry["slot"]] = claims
        writer.append({"seq": next_seq(), "run_id": run_id, "case_key": case_key,
                       "arm": ARM_B, "kind": "tool", "ts": _now(),
                       "slot": entry["slot"], "tool": result.get("tool") or "Task",
                       "claims": claims, "tokens": tokens})
    if stopped:
        writer.append({"seq": next_seq(), "run_id": run_id, "case_key": case_key,
                       "arm": ARM_B, "kind": "ceiling", "ts": _now(),
                       "reason": stopped, "tokens_spent": spent,
                       "token_ceiling": token_ceiling})

    ranking = rank_theories(case, evidence, slots=labels, rank_fn=rank_fn)
    if ranking["ok"]:
        ranking["chosen_cause"] = theory_for(ranking["chosen_slot"], theory_rows)
    writer.append({"seq": next_seq(), "run_id": run_id, "case_key": case_key,
                   "arm": ARM_B, "ts": _now(), **ranking})
    verdict = {"seq": next_seq(), "run_id": run_id, "case_key": case_key,
               "arm": ARM_B, "kind": "verdict", "ts": _now(),
               "chosen_cause": ranking.get("chosen_cause") or "",
               "order": ranking.get("order") or [], "tokens": spent}
    writer.append(verdict)
    return verdict


async def run_case_a(case: dict, *, investigate: Callable[[dict, str], Awaitable[dict]],
                     writer: RowWriter, run_id: str, arm: str = ARM_A,
                     token_ceiling: int = DEFAULT_TOKEN_CEILING) -> dict:
    """Arm A: today's single linear trace, one hypothesis under the same ceiling.

    Exactly one dispatch, because that is what the arm is measuring — the
    present behaviour of `systematic-debugging/SKILL.md:211`, "Form a Single
    Hypothesis". `A-retry`, the compute-matched second attempt, is the same
    function with `arm=ARM_RETRY`; #627 requires it be recorded separately
    rather than assumed, and `report` refuses to judge without it.
    """
    case_key = case["key"]
    result = await investigate(case, case.get("claimed_cause") or "")
    tokens = int(result.get("tokens") or 0)
    conclusion = str(result.get("conclusion") or "")
    claims = [{"claim": conclusion, "citations": result.get("citations") or []}] \
        if conclusion else []
    writer.append({"seq": 1, "run_id": run_id, "case_key": case_key, "arm": arm,
                   "kind": "tool", "ts": _now(), "slot": None,
                   "tool": result.get("tool") or "linear-trace",
                   "claims": claims, "tokens": tokens})
    if tokens > token_ceiling:
        writer.append({"seq": 2, "run_id": run_id, "case_key": case_key, "arm": arm,
                       "kind": "ceiling", "ts": _now(), "reason": "over-ceiling",
                       "tokens_spent": tokens, "token_ceiling": token_ceiling})
    verdict = {"seq": 3, "run_id": run_id, "case_key": case_key, "arm": arm,
               "kind": "verdict", "ts": _now(), "chosen_cause": conclusion,
               "tokens": tokens}
    writer.append(verdict)
    return verdict


# ── the live wiring (the owed study uses these; tests inject their own) ─────

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "object", "properties": {
                "path": {"type": "string"}, "line": {"type": "integer"},
                "value": {"type": "string"}, "timestamp": {"type": "string"}}}}}}}},
    "required": ["claims"],
}
THEORIES_SCHEMA = {
    "type": "object",
    "properties": {"theories": {"type": "array", "items": {"type": "string"},
                                 "minItems": THEORIES_PER_CASE,
                                 "maxItems": THEORIES_PER_CASE}},
    "required": ["theories"],
}


def _new_bench_session(case_key: str, tag: str) -> str:
    return f"bench_2258_{tag}_{re.sub(r'[^a-zA-Z0-9]', '_', case_key)[:48]}"


async def _engine_turn(prompt: str, *, schema: dict, session_id: str,
                       max_turns: int) -> dict:
    """One sandboxed engine turn, with the usage the ceiling is measured in.

    Same primitive `eval/decision_replay_588.py` uses: a `bench_` session id,
    which the aggregator sandboxes, so an investigation can read but never
    write, never dispatch a task and never file anything.
    """
    import yaml as _yaml
    from app.harness import HookRegistry, RunOptions, install_default_safety_hook
    from app.harness.loop import run_query
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
    from app.paths import VAULT_ROOT
    from app.prompt_builder import build_system_prompt

    from agent_mcp._tool_sandbox import is_sandboxed_session
    if not is_sandboxed_session(session_id):
        raise RuntimeError(f"refusing to run an unsandboxed session {session_id!r}")

    config = _yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
    alias = (config.get("model") or {}).get("default", "primary")
    model_env = ((config.get("models") or {}).get(alias) or {}).get("env") or {}
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    options = RunOptions(
        model=alias,
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=build_system_prompt(platform="worker",
                                          overlay_dir=VAULT_ROOT / "lloyd"),
        max_turns=max_turns, mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
        disallowed_tools=set(_get_disallowed_tools()) | {"Task"},
        session_id=session_id, priority=3, hooks=hooks,
        final_schema=schema, final_schema_prompt=(
            "Restate the answer above as a JSON object matching the schema."),
        **_get_harness_kwargs())
    options.surface = "worker"
    structured: dict = {}
    usage: dict = {}
    async for evt in run_query([{"role": "user", "content": prompt}], options):
        if evt["type"] == "result":
            structured = evt.get("structured") or {}
            usage = evt.get("usage") or {}
    return {"structured": structured,
            "tokens": int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0),
            "model": alias}


async def generate_theories_live(case: dict, *, max_turns: int = 6) -> list[str]:
    """Three disjoint candidate causes, from the symptom alone.

    The prompt carries the incident and nothing else — no log path, no ledger
    line — because a theory generated after reading a log is anchored to it,
    which is the failure class the whole technique exists to fight.
    """
    prompt = (
        f"Incident: {case.get('input')}\n\n"
        f"Name exactly {THEORIES_PER_CASE} MUTUALLY EXCLUSIVE candidate root causes "
        "for this incident, from the symptom alone. Do not read any file, log or "
        "database: you have none, and an idea formed after reading a log is "
        "anchored to it. Each theory must be a mechanism that could produce the "
        "symptom and be DISPROVABLE by evidence someone could gather. A second "
        "theory that is the first in other words is a failure, not a second "
        "opinion. Return them as the schema's `theories` array.")
    out = await _engine_turn(prompt, schema=THEORIES_SCHEMA,
                             session_id=_new_bench_session(case["key"], "gen"),
                             max_turns=max_turns)
    return [str(t) for t in (out["structured"].get("theories") or [])]


async def investigate_live(case: dict, theory: str, *,
                           max_turns: int = 20) -> dict:
    """One scoped per-theory investigation: evidence for THIS theory only."""
    prompt = (
        f"Incident: {case.get('input')}\n\n"
        f"Candidate cause under investigation, and the only one: {theory}\n\n"
        "Gather evidence FOR or AGAINST this one cause and no other: read the "
        "files, logs and stores that would distinguish it. Do not compare it to "
        "a rival, do not rank anything, do not name a root cause. Hand back the "
        "evidence case, never a summary: for each claim you found, give the "
        "citation that lets someone else check it — a path (a `path:line` form is "
        "fine), and where you have one the metric value and its timestamp. A "
        "claim you cannot cite does not go in the answer.")
    out = await _engine_turn(prompt, schema=CLAIMS_SCHEMA,
                             session_id=_new_bench_session(case["key"], "inv"),
                             max_turns=max_turns)
    return {"tool": "Task", "claims": out["structured"].get("claims") or [],
            "tokens": out["tokens"], "model": out["model"]}


def rank_live(query: str, candidates: Sequence[str]):
    """The production rank path. Argmax is read by the caller; no threshold."""
    from app import djev
    return djev.rank(query, candidates, seam="diagnosis_fanout_2258")


# ── the retry baseline (clause 5's #627 rule) ───────────────────────────────

def corpus_fingerprint(cases_path: Path) -> str:
    return hashlib.sha256(cases_path.read_bytes()).hexdigest()[:16]


def write_baseline(path: Path, *, cases_path: Path, n: int, model: str,
                   token_ceiling: int, token_spend: int, top1_hits: int,
                   measured_at: str | None = None) -> dict:
    record = {"kind": "retry_baseline", "corpus": corpus_fingerprint(cases_path),
              "n": n, "model": model, "token_ceiling": token_ceiling,
              "token_spend": token_spend, "top1_hits": top1_hits,
              "measured_at": measured_at or _now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return record


def baseline_problem(baseline: dict | None, *, corpus: str, n: int, model: str,
                     token_ceiling: int) -> str | None:
    """Why the cached baseline cannot stand in, or None when it can.

    Every mismatch is its own sentence, because "the corpus grew since the
    baseline was measured" and "the baseline used a different model" need
    different repairs, and none of them may be read as a pass.
    """
    if not baseline:
        return "no cached retry baseline recorded for this corpus"
    problems = []
    if baseline.get("corpus") != corpus:
        problems.append(f"corpus fingerprint {baseline.get('corpus')!r} != {corpus!r}")
    if int(baseline.get("n") or -1) != n:
        problems.append(f"baseline N {baseline.get('n')!r} != corpus N {n}")
    if baseline.get("model") != model:
        problems.append(f"baseline model {baseline.get('model')!r} != {model!r}")
    if int(baseline.get("token_ceiling") or -1) != token_ceiling:
        problems.append(f"baseline ceiling {baseline.get('token_ceiling')!r} "
                        f"!= {token_ceiling}")
    return "; ".join(problems) or None


# ── re-grading and report (clause 5) ────────────────────────────────────────

def agrees(case: dict, chosen_cause: str) -> bool:
    """Top-1 agreement: EVERY recorded marker names the same mechanism.

    Deterministic on purpose — this is an offline re-grade and the engine is
    not in the room. Markers come from the ledger fields or the named mechanism
    in the recorded correction, never from a model's judgement.
    """
    text = (chosen_cause or "").lower()
    markers = [str(m).lower() for m in (case.get("actual_cause_markers") or [])
               if str(m).strip()]
    return bool(markers) and all(m in text for m in markers)


def latest_verdicts(rows: Sequence[dict]) -> dict[tuple[str, str], dict]:
    """The last verdict row per (case, arm); a later run supersedes an earlier."""
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("kind") == "verdict":
            out[(str(row.get("case_key")), str(row.get("arm")))] = row
    return out


def score(rows: Sequence[dict], cases: Sequence[dict], *,
          citation_root: Path | None = None) -> dict:
    """Per-arm top-1, citation rate and spend, from recorded rows alone.

    `citation_root` relocates citation resolution, which is how the citation
    rate of a fixture run is checked and how a run recorded against a worktree
    can be re-graded after that worktree is gone. Live rows cite absolute paths,
    so the default resolves them where they are.
    """
    by_key = {c["key"]: c for c in cases}
    arms: dict[str, dict] = {}
    for arm in (ARM_A, ARM_B):
        verdicts = {k: v for k, v in latest_verdicts(rows).items() if k[1] == arm}
        hits = sum(1 for (key, _), row in verdicts.items()
                   if key in by_key and agrees(by_key[key], row.get("chosen_cause") or ""))
        arm_rows = [r for r in rows if r.get("arm") == arm]
        cites = verify_citations(citation_rows(arm_rows), root=citation_root)
        arms[arm] = {
            "n": len(verdicts),
            "top1_hits": hits,
            "citations": cites,
            "token_spend": sum(int(r.get("tokens") or 0) for r in arm_rows
                               if r.get("kind") != "verdict"),
        }
    return {"arms": arms,
            "unscored": sorted({k[0] for k in latest_verdicts(rows)} - set(by_key))}


def format_report(rows: Sequence[dict], cases: Sequence[dict], *,
                  baseline: dict | None, corpus: str, model: str,
                  token_ceiling: int,
                  citation_root: Path | None = None) -> tuple[list[str], bool]:
    """The printed lines, and whether the headline states an evaluation.

    `cannot evaluate` is the only thing a missing or mismatched retry baseline
    may produce — a study that has not measured its compute-matched baseline
    cannot say the fan-out won, because it cannot say the arms were matched.
    """
    scored = score(rows, cases, citation_root=citation_root)
    a, b = scored["arms"][ARM_A], scored["arms"][ARM_B]
    lines = [f"[2258] cases: {len(cases)}  recorded verdicts: "
             f"A={a['n']} B={b['n']}"]
    if scored["unscored"]:
        lines.append(f"  unscored case keys (no case row): {scored['unscored']}")
    lines.append(f"  top-1 agreement   A: {a['top1_hits']}/{a['n']}   "
                 f"B: {b['top1_hits']}/{b['n']}   "
                 f"(bar: B ahead by >= {TOP1_MARGIN})")
    lines.append(f"  citations verified A: {a['citations']['passed']}/"
                 f"{a['citations']['n']}   B: {b['citations']['passed']}/"
                 f"{b['citations']['n']}   (bar: B >= {CITATION_BAR:.0%})")
    lines.append(f"  token spend       A: {a['token_spend']}   B: {b['token_spend']}")

    problem = baseline_problem(baseline, corpus=corpus, n=len(cases), model=model,
                              token_ceiling=token_ceiling)
    if problem:
        lines.append(f"  token spend  A-retry: not recorded ({problem})")
        lines.append(f"HEADLINE: cannot evaluate — {problem}")
        return lines, False

    retry = baseline or {}
    lines.append(f"  token spend  A-retry: {retry.get('token_spend')} "
                 f"(cached baseline, N={retry.get('n')}, model={retry.get('model')}, "
                 f"ceiling={retry.get('token_ceiling')})")
    lines.append(f"  top-1 agreement A-retry: {retry.get('top1_hits')}/{retry.get('n')}")
    margin = b["top1_hits"] - max(a["top1_hits"], int(retry.get("top1_hits") or 0))
    rate = b["citations"]["rate"]
    won = margin >= TOP1_MARGIN and b["citations"]["n"] > 0 and rate >= CITATION_BAR
    lines.append(f"  margin over the better single-trace arm: {margin} case(s); "
                 f"B citation rate {rate:.1%}")
    lines.append(f"HEADLINE: {'arm B wins' if won else 'arm B does not win'} "
                 f"(margin {margin} >= {TOP1_MARGIN}: {margin >= TOP1_MARGIN}; "
                 f"citations {rate:.1%} >= {CITATION_BAR:.0%}: "
                 f"{b['citations']['n'] > 0 and rate >= CITATION_BAR})")
    return lines, True


# ── CLI ─────────────────────────────────────────────────────────────────────

# ── the witness (clause 6) ──────────────────────────────────────────────────

#: The events whose counts the item quotes, so the printed census is the one a
#: reader can compare against the item text rather than a fresh selection.
WITNESS_EVENTS = ("round_aborted", "round_abandoned", "red_tree_filed",
                  "red_tree_closed", "alert", "vault_revert", "rollback_succeeded")


def witness_census(path: Path = VAULT_WITNESS) -> dict:
    """Re-derive the quoted ledger figures from the committed bytes.

    The live ledger under `~/.local/state` is appended to by every gate and has
    no history, so a number read off it is true for one second. The committed
    extract is the witness, and this counts it: row count (the `wc -l` the item
    quotes), reason-bearing rows, and the event census the triage reported. The
    seven counted event classes are complete in the extract by construction; its
    total row count is the extract's own and says nothing about the ledger's.
    """
    rows = 0
    reason_rows = 0
    events: dict[str, int] = {}
    unparsable = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                unparsable += 1
                continue
            if not isinstance(row, dict):
                unparsable += 1
                continue
            if row.get("reason"):
                reason_rows += 1
            event = str(row.get("event") or "")
            events[event] = events.get(event, 0) + 1
    return {"path": str(path), "rows": rows, "reason_rows": reason_rows,
            "unparsable": unparsable,
            "events": {e: events.get(e, 0) for e in WITNESS_EVENTS}}


def cmd_witness(args) -> int:
    """Print the census from the witness bytes, and refuse if it is not the file.

    A row-count mismatch means the committed witness is not the corpus the
    study was built on — either it was replaced or it is being rewritten — so
    the command exits non-zero rather than printing a census of unknown bytes.
    """
    path = Path(args.path)
    if not path.exists():
        print(f"[2258] cannot evaluate: no witness at {path}")
        return 1
    census = witness_census(path)
    print(f"[2258] witness: {census['path']}")
    print(f"[2258] rows: {census['rows']}  (committed: {WITNESS_ROWS})")
    print(f"[2258] reason-bearing rows: {census['reason_rows']}")
    print(f"[2258] unparsable rows: {census['unparsable']}")
    for event in WITNESS_EVENTS:
        print(f"[2258]   event {event}: {census['events'][event]}")
    if census["rows"] != WITNESS_ROWS:
        print(f"[2258] cannot evaluate: witness rows {census['rows']} != committed "
              f"{WITNESS_ROWS} — these are not the bytes the item quoted")
        return 1
    return 0


def cmd_corpus(args) -> int:
    ledger = Path(args.ledger)
    if not ledger.exists() and ledger != Path(LEDGER_PATH):
        print(f"[2258] cannot evaluate: no ledger at {ledger}")
        return 1
    # The default is the COMMITTED witness, not the live ledger, and the reason
    # is the study's fingerprint: `run` writes the corpus identity into every row
    # and `report` refuses a baseline whose corpus id differs. Pointed at the live
    # file, that identity moves every time a round closes, so a re-run hours later
    # reports "cannot evaluate" for a reason that has nothing to do with the arms.
    print(f"[2258] ledger: {ledger} ({sum(1 for _ in ledger.open())} row(s))")
    if ledger == Path(LEDGER_PATH):
        print(f"[2258] note: the live ledger accrues; the committed witness is "
              f"{VAULT_WITNESS} (WITNESS_CASES={WITNESS_CASES})")
    cases, notes = build_cases(ledger, Path(args.learnings))
    for note in notes:
        print(f"[2258] dropped: {note}")
    # Both sources print their own count, always, including when one is zero.
    # The learnings tree has yielded 0 every time it has been run: a summary
    # that only says "wrote 97 case(s)" lets a reader conclude the second source
    # contributed, which is the zero-denominator class this repo already has a
    # rule about. The count comes from the `source` field each builder stamps,
    # never from a prefix of `source_ref`: learnings refs read
    # `memory/learnings/<file>:<line>`, which no `learnings/` prefix test
    # matches, so the old classification filed every learnings case under
    # `ledger` and printed `learnings=0` for a corpus that contained them.
    by_source = {"ledger": 0, "learnings": 0}
    for case in cases:
        by_source[str(case.get("source") or "ledger")] += 1
    code, message = write_cases(cases, Path(args.out), min_cases=args.min_cases)
    print(f"[2258] sources: ledger={by_source['ledger']} "
          f"learnings={by_source['learnings']}")
    print(f"[2258] {message}")
    return code


async def cmd_run(args) -> int:
    cases = load_cases(Path(args.cases))
    if not cases:
        print(f"[2258] no cases at {args.cases}; run `corpus` first")
        return 1
    arms = [a.strip() for a in (args.arms or f"{ARM_A},{ARM_B}").split(",") if a.strip()]
    unknown = sorted(set(arms) - {ARM_A, ARM_B, ARM_RETRY})
    if unknown:
        print(f"[2258] unknown arm(s) {unknown}; the arms are {ARM_A}, {ARM_B} and "
              f"{ARM_RETRY} (the retry arm is the compute-matched second single trace)")
        return 1
    writer = RowWriter(Path(args.rows))
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    ceiling = args.ceiling
    for case in cases:
        if args.only and case["key"] not in args.only:
            continue
        if ARM_A in arms:
            await run_case_a(case, investigate=investigate_live, writer=writer,
                             run_id=run_id, token_ceiling=ceiling)
        if ARM_RETRY in arms:
            await run_case_a(case, investigate=investigate_live, writer=writer,
                             run_id=run_id, arm=ARM_RETRY, token_ceiling=ceiling)
        if ARM_B in arms:
            verdict = await run_case_b(case, generate=generate_theories_live,
                                       investigate=investigate_live,
                                       rank_fn=rank_live, writer=writer,
                                       run_id=run_id, token_ceiling=ceiling)
            print(f"[2258] {case['key']} B chose: {str(verdict['chosen_cause'])[:90]}")
    print(f"[2258] rows appended to {args.rows} (run {run_id})")
    return 0


def cmd_baseline(args) -> int:
    """Cache what the retry arm measured, from its recorded rows — no engine.

    The order the owed study runs is `run --arms A-retry`, then this, then
    `report`. Recording the baseline as a file rather than re-deriving it from
    whatever rows exist at report time is what makes the match auditable: the
    file names the corpus, the N, the model and the ceiling it was measured
    under, and `report` refuses it when any of those no longer hold.
    """
    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    rows = [r for r in _read_jsonl(Path(args.rows)) if r.get("kind")]
    retry_rows = [r for r in rows if r.get("arm") == ARM_RETRY]
    if not retry_rows or not cases:
        print("[2258] cannot evaluate: no retry-arm rows recorded for this corpus; "
              "run `run --arms A-retry` first")
        return 1
    verdicts = {k: v for k, v in latest_verdicts(rows).items() if k[1] == ARM_RETRY}
    by_key = {c["key"]: c for c in cases}
    hits = sum(1 for (key, _), row in verdicts.items()
               if key in by_key and agrees(by_key[key], row.get("chosen_cause") or ""))
    record = write_baseline(Path(args.out), cases_path=cases_path, n=len(verdicts),
                            model=args.model, token_ceiling=args.ceiling,
                            token_spend=sum(int(r.get("tokens") or 0)
                                            for r in retry_rows
                                            if r.get("kind") != "verdict"),
                            top1_hits=hits)
    print(f"[2258] retry baseline: {json.dumps(record)}")
    return 0


def cmd_report(args) -> int:
    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    rows = [r for r in _read_jsonl(Path(args.rows)) if r.get("kind")]
    baseline = None
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            baseline = None
    corpus = (corpus_fingerprint(cases_path) if cases_path.exists() else "no-cases-file")
    lines, evaluated = format_report(
        rows, cases, baseline=baseline, corpus=corpus, model=args.model,
        token_ceiling=args.ceiling,
        citation_root=(Path(args.root) if args.root else None))
    for line in lines:
        print(line)
    return 0 if evaluated else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay instrument for #2258 (theory fan-out vs single trace)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    corpus = sub.add_parser("corpus", help="build the cases file or refuse")
    corpus.add_argument("--ledger", default=str(VAULT_WITNESS),
                        help="ledger rows to build over: the committed witness "
                             "by default, so the corpus fingerprint is stable")
    corpus.add_argument("--learnings", default=str(LEARNINGS_ROOT))
    corpus.add_argument("--out", default=str(CASES_PATH))
    corpus.add_argument("--min-cases", type=int, default=MIN_CASES)
    corpus.set_defaults(fn=cmd_corpus)

    run = sub.add_parser("run", help="run the arms over the corpus (needs the engine)")
    run.add_argument("--cases", default=str(CASES_PATH))
    run.add_argument("--rows", default=str(ROWS_PATH))
    run.add_argument("--arms", default=None)
    run.add_argument("--only", nargs="*", default=None)
    run.add_argument("--ceiling", type=int, default=DEFAULT_TOKEN_CEILING)
    run.set_defaults(fn=lambda a: asyncio.run(cmd_run(a)))

    base = sub.add_parser("baseline",
                          help="cache what the retry arm measured, from its rows")
    base.add_argument("--cases", default=str(CASES_PATH))
    base.add_argument("--rows", default=str(ROWS_PATH))
    base.add_argument("--out", default=str(BASELINE_PATH))
    base.add_argument("--model", default="primary")
    base.add_argument("--ceiling", type=int, default=DEFAULT_TOKEN_CEILING)
    base.set_defaults(fn=cmd_baseline)

    report = sub.add_parser("report", help="re-grade recorded rows, no engine")
    report.add_argument("--cases", default=str(CASES_PATH))
    report.add_argument("--rows", default=str(ROWS_PATH))
    report.add_argument("--baseline", default=str(BASELINE_PATH))
    report.add_argument("--model", default="primary")
    report.add_argument("--ceiling", type=int, default=DEFAULT_TOKEN_CEILING)
    report.add_argument("--root", default=None,
                        help="resolve citations under this tree (a fixture or a "
                             "worktree the run recorded against)")
    report.set_defaults(fn=cmd_report)

    witness = sub.add_parser("witness",
                            help="re-derive the quoted ledger figures from the "
                                 "committed witness bytes")
    witness.add_argument("--path", default=str(VAULT_WITNESS))
    witness.set_defaults(fn=cmd_witness)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
