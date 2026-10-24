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
    python eval/diagnosis_fanout_replay.py run       [--only KEY…] [--arms A,A-retry,B] [--ceiling N]
    python eval/diagnosis_fanout_replay.py baseline  [--rows …] [--out …]
    python eval/diagnosis_fanout_replay.py report    [--cases …] [--rows …] [--baseline …] [--root …] [--only KEY…]

`run --only` and `report --only` take the same list of case keys, and #2466 is why both
exist: the study runs over the 21 non-leaky cases and leaves the 77 leaky `preexisting`
ones in the file, so the subset has to survive from the run into the retry baseline match
and into the artifact without anybody cutting the corpus. A key that names no case is an
error on both, because a silently-shorter list is a silently-different study.

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
#:
#: The ceiling is CUMULATIVE and read BEFORE each dispatch (`spent >= token_ceiling`
#: in `run_case_b`), so it budgets the whole fan-out, not one theory: a theory's cost
#: is known only when it returns, so the arm can refuse the NEXT dispatch and never
#: the one in flight. 200,000 was therefore below arithmetic, not merely below one
#: measurement — one measured theory costs 161,011 tokens, so three of them cost
#: 483,033 and could never fit. The 2026-10-13 pass shows what that does to the
#: record: every one of its 13 `kind: ceiling` rows is `seq: 4` with `tokens_spent`
#: from 280,279 to 362,808 against this 200,000, so the third theory was cut off
#: after the arm had already spent past the ceiling, and `rank_theories` then ranked
#: the case with a slot that has no evidence under it still in the candidate list.
#:
#: 600,000, up from 200,000, is the smallest round number that holds three of the
#: WORST dispatch on record: the most ever charged to one theory is 199,283 tokens
#: (`~/lloyd-data/eval/2258/rows.jsonl`, line 126 — a `kind: refusal` row for slot
#: `c1`, and a refused dispatch still pays its tokens into `spent`), and three of
#: those are 597,849 — 2,151 tokens clear, which is arithmetic rather than comfort.
#: The trio arm B's own test reproduces, 161,011 + 184,976 + 161,011 = 506,998, sits
#: 93,002 under it. The arm-A anchor still holds beside the new figure: the one
#: investigation ever followed end-to-end spent 179,186 tokens over 27 turns
#: (`~/lloyd-data/eval/2258/probe-maxturns-40.log`, line
#: `CAP=40 claims_n=1 tokens=179186`), and smoke-b3's truncation is the newer
#: measurement — 161,011 tokens, `stop_reason: max_turns` at `num_turns: 41`
#: (`~/lloyd-data/eval/2258/smoke-b3.jsonl`, line 2, `run-20261018T220131Z`).
#:
#: Two things this number does not claim. (a) A case that really does spend past it
#: still records `kind: ceiling`, which is the true answer; what the figure may not do
#: is make an ordinary three-theory case look over budget. (b) The per-theory cost at
#: the RAISED turn cap below is unmeasured — the 199,283 census was taken under a
#: 40-turn cap — so the ceiling is sized against the recorded worst case, and if a
#: 60-turn dispatch costs more than 199,283 the next live pass says so with a ceiling
#: row instead of this comment having guessed. `run --ceiling` is the per-pass knob,
#: and the live launcher `~/lloyd-data/eval/2258/launch-pass.sh` passes `--ceiling`
#: explicitly, so raising this default does not by itself raise the next real pass.
#: Its figure is read from that file, never quoted here: `grep -n -- '--ceiling'`.
DEFAULT_TOKEN_CEILING = 600_000

#: Per-theory dispatch turn budget — the other half of the ONE arithmetic property
#: above, not a second knob: a longer dispatch costs more tokens, so the ceiling has
#: to absorb three of whatever this buys. 60, up from 40, because the cap that was in
#: force is itself the truncation on record: the arm-B theory smoke-b3 came back
#: `stop_reason: max_turns` at `num_turns: 41` for 161,011 tokens
#: (`~/lloyd-data/eval/2258/smoke-b3.jsonl`, line 2, `run-20261018T220131Z`) — the
#: harness reported 41 turns against the 40 this module shipped. A cap that produced
#: the study's own truncation cannot be the cap the study ships.
#:
#: The older anchor stays beside it, because the two bracket the cost rather than
#: averaging out: the same dispatch shape reached `stop_reason: stop` at 27 turns on
#: 179,186 tokens under that same cap of 40
#: (`~/lloyd-data/eval/2258/probe-maxturns-40.log`, line
#: `META: {"stop_reason": "stop", "num_turns": 27, ...}`), and the 20 before that
#: ended the case at 21 turns with `max_turns`. 60 is a round number above the only
#: truncation on record, not a figure taken from a 60-turn run — no such run exists
#: yet, which is what the owed live pass measures. Raising it is not free and is not
#: optional either: an exhausted dispatch is NO answer rather than a short one, since
#: the finalizer restates a `final_schema` only for `stop`/`end_turn`
#: (`FINALIZABLE_STOP_REASONS`, `app/harness/finalizer.py:78`), so the turn cap and
#: the token ceiling are the same decision read from two sides.
#: `generate_theories_live` deliberately keeps its own much smaller cap: writing
#: three candidate sentences is not an investigation.
DEFAULT_INVESTIGATION_TURNS = 60

#: Why a per-theory row was refused (#2635). Deliberately NOT the ceiling's word:
#: a `kind: ceiling` row with `reason: "token-ceiling"` means the fan-out was
#: shortened because the money ran out, and `report` prints that spend story. A
#: theory that ran, cost its tokens and answered in prose is the other failure
#: class entirely — Pienaar's fourth — and folding it into the ceiling row would
#: let a study read "budget too small" from a run whose model simply refused to
#: cite. `no-claims` is the raise's own complaint, kept as a recorded fact.
REFUSAL_NO_CLAIMS = "no-claims"
#: The harness's own word for a run its turn budget cut off: the `result` event's
#: `stop_reason` (`app/harness/events.py:324`). The P6b wrap-up leaves it at
#: `max_turns` rather than rewriting it, and the finalizer restates a
#: `final_schema` only for `stop`/`end_turn`
#: (`FINALIZABLE_STOP_REASONS`, `app/harness/finalizer.py:78`), so this is exactly
#: the stop that comes back with no structured answer and therefore no claims.
STOP_MAX_TURNS = "max_turns"
#: A per-theory refusal the turn budget caused (#2807). A third word, and not
#: either of the other two: `token-ceiling` says a theory was never dispatched
#: because the fan-out's money ran out, `no-claims` says a theory ran to its stop
#: and answered in prose, and this says the theory never got to answer at all.
#: Folding this into `no-claims` is what made the truncation-vs-honest-empty
#: question unanswerable from the 2026-10-13 pass's rows, and folding it into
#: `token-ceiling` would book the wrong follow-up (a bigger fan-out budget for a
#: single dispatch that was already over its own turn cap).
REFUSAL_TURN_CAP = "turn-cap"

#: #3241, by #2985's owed-check ruling 4 ("an uncitable claim becomes a recorded refusal
#: row (kind: refusal, reason 'uncitable-claim'), the same route"). #2635 drew the line
#: at "no claims at all" and left this neighbouring condition raising, and the comment
#: that defended it argued a citation-less claim is a broken instrument rather than a
#: declined theory — a distinction the live pass priced as a cancelled case, not as a
#: signal: six `FAILED <case> B: RecordError: theory 'cN' has a claim with no citation`
#: lines under `~/lloyd-data/eval/2258/` (`run.log:32,36,62,86,88`, `smoke-b3.log:1`),
#: every one of them a case that had already paid for its other two dispatches and then
#: disappeared from `report`'s denominator. Same route as `no-claims`; a distinct word
#: for the same reason the others are distinct — an uncitable claim is a prompting or
#: schema question, a turn cap is a budget question, a ceiling is a money question.
REFUSAL_UNCITABLE_CLAIM = "uncitable-claim"


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


def select_cases(cases: Sequence[dict], only: Sequence[str] | None
                 ) -> tuple[list[dict], list[str]]:
    """The cases a `--only` selector names, plus the keys it named that the corpus lacks.

    Shared by `run` and `report` on purpose. The study is pointed at a named subset by
    hand — today the 21 `flaky-*` and `landkill-*` keys, because the 77 `preexisting`
    ones leak the answer into the input (#2258) — and a hand-written list of 21 keys is
    where a typo goes to hide. A key that matches no case used to be ignored: `run`
    printed its normal closing line over however many cases it happened to match, and
    `report` had no selector at all. Both now say which names matched nothing.

    Selection is in the caller's order, which changes nothing about a tally: `score`
    counts per case.
    """
    if not only:
        return list(cases), []
    wanted = list(dict.fromkeys(only))
    by_key = {c["key"]: c for c in cases}
    return ([by_key[k] for k in wanted if k in by_key],
            [k for k in wanted if k not in by_key])


def excluded_note(cases: Sequence[dict], selected: Sequence[dict]) -> str:
    """What the graded subset leaves out, in words — because the artifact must own it.

    The subject of this study is narrower than its corpus, and the narrower set is not
    a random slice of it: the dropped cases are the family whose input quotes the
    refuted claim, which is the one family where a top-1 rate would be an
    answer-leakage measurement rather than a diagnosis one. A report that printed
    `cases: 21` and a HEADLINE over them, without saying what the other 77 were or why
    they are absent, would read as though the instrument had 21 cases in the world. It
    is one line, and it is the line that tells a reader of the artifact alone whether
    the number they are looking at covers the class they care about.
    """
    left_out = [c for c in cases
                if c["key"] not in {s["key"] for s in selected}]
    if not left_out:
        return (f"excluded from the graded set: none — all {len(cases)} corpus cases "
                f"are graded, so the headline covers the whole corpus")
    families = sorted({str(c.get("family") or "?") for c in left_out})
    detail = " ".join(
        f"{f}={sum(1 for c in left_out if (c.get('family') or '?') == f)}"
        for f in families)
    return (f"excluded from the graded set: {len(left_out)} of {len(cases)} corpus "
            f"cases, in {len(families)} famil"
            f"{'y' if len(families) == 1 else 'ies'}: {detail}. The headline below is "
            f"over the {len(selected)} graded cases only.")


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
    123 does not exist fails `line-out-of-range`; a string that cannot be stat'd
    at all — prose in the `path` field, one component past NAME_MAX — fails
    `not-a-path`. `root` relocates the whole check, which is how a fixture tree
    proves both failures.

    The `never raise` half is load-bearing and was a lie until #2799: `path` is
    model output, and `Path.exists()` answers a name longer than NAME_MAX with
    `OSError: [Errno 36] File name too long` rather than "no". 45 of the 2258
    artifact's citations are prose excerpts of that shape (the longest, 281
    characters), and every one of them aborted `cmd_report` before it printed a
    line, because `cmd_report` prints what `format_report` collected only after
    it returns. So the whole resolve-and-stat body is guarded — `exists()`,
    `is_file()` and the line count's `open` alike — and an unstatable citation is
    a FAILED citation, counted in the denominator exactly like a fabricated path,
    never an exception and never skipped.
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
    # Every filesystem call below is guarded, because `path` is model output and
    # prose is not always statable: one component past NAME_MAX makes `exists()`
    # RAISE `OSError: [Errno 36] File name too long` where a missing file merely
    # answers "no" (#2799). An unstatable string fails the citation as
    # `not-a-path` — a verdict in the denominator, never an exception.
    try:
        exists = resolved.exists()
    except OSError:
        out["reason"] = "not-a-path"
        return out
    if not exists:
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
    # The other two filesystem operations, guarded as one: a path that survives
    # `exists()` can still raise on `is_file()` or on the line count's `open`
    # (the long-name case arrives here too when the citation carries a line),
    # and a report that dies one line lower dies just as completely.
    try:
        is_file = resolved.is_file()
        n_lines = _line_count(resolved) if is_file else 0
    except OSError:
        out["reason"] = "not-a-path"
        return out
    if not is_file:
        out["reason"] = "not-a-file"
        return out
    if line_no < 1 or line_no > n_lines:
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


def empty_claims(row: dict) -> bool:
    """Whether a per-theory investigation row carries no claims at all.

    ONE definition, read twice (#2635): `validate_investigation` raises on it, and
    arm B's loop records it as a refusal instead of raising. Two spellings of "did
    it hand back nothing" would drift, and the drift is the bad kind — a row that
    fails one check and passes the other gets ranked with no evidence under it, or
    gets a refusal label the record cannot support.

    This is deliberately narrower than "the theory produced nothing to rank", which is
    two conditions and now two classifiers: this one, and `uncitable_claims` for a claim
    that cannot be cited. #3241 moved the second one off the raise path; this function
    used to argue a citation-less claim is a broken instrument rather than a declined
    theory and that the raise was what the next reader needed, and the traffic said
    otherwise: four cases of it in
    the Oct 13 pass and two more on Oct 18, six cancelled cases in total, none of them a
    broken instrument anyone acted on and all of them absent from the denominator that
    would have shown the arm losing cases it had already paid for. Both refusals are read
    from these two classifiers BEFORE the validator runs, which is why the validator's
    remaining raise cannot be reached by a row either classifier describes.
    """
    claims = row.get("claims")
    return not isinstance(claims, list) or not claims


def stop_pair(result: dict) -> tuple[Any, int | None]:
    """The harness's `stop_reason` / `num_turns` as #2807 requires them recorded.

    One read for both refusal branches (#3241), because #2807's whole point is that the
    pair means something precise: `stop_reason` is `None` when the investigator reported
    no stop (absence is not evidence of a cap), and `num_turns` is `None` unless the
    harness reported a number — a string turn count coerced to an int here would put a
    number on the row that the harness never sent.
    """
    stop_reason = result.get("stop_reason") or None
    reported_turns = result.get("num_turns")
    return (stop_reason,
            int(reported_turns) if isinstance(reported_turns, (int, float)) else None)


def uncitable_claims(row: dict) -> list[dict]:
    """The claims in a theory's row that cannot be pointed at anything, with why.

    #3241 (#2985's owed ruling 4). This is the condition `validate_investigation` used to
    raise on — `not text or not isinstance(cites, list) or not cites` — lifted out of the
    validator so the caller can refuse the slot as data instead of cancelling the case.
    It classifies each claim separately rather than answering yes/no because ONE
    condition covers two shapes, and on the recorded traffic they are not even the same
    majority: 4 of the 6 live failures carry an empty claim (`no citation: ''`) and 2 a
    worded claim with `citations: []` (`run.log:36,88`). A row that could not tell them
    apart would print `''` for most real cases and read as "the model declined to cite"
    when the model handed back nothing worded — the opposite follow-up. Which shape
    deserves its own `reason` is a ruling owed-check cannot make until rows exist that
    distinguish them; naming the shape here is what lets that ruling be measured.
    """
    claims = row.get("claims")
    if not isinstance(claims, list):
        return []
    findings: list[dict] = []
    for claim in claims:
        text = str((claim or {}).get("claim") or "").strip()
        cites = (claim or {}).get("citations")
        if text and isinstance(cites, list) and cites:
            continue
        findings.append({"claim": text[:80],
                         "why": "no claim text" if not text else "no citations"})
    return findings


def uncitable_message(slot: Any, findings: Sequence[dict]) -> str:
    """The refused theory's own words, and which half of the condition it fell through.

    Every refusal word in this file has meant a different follow-up since #2635, so the
    detail has to be actionable by itself: the slot, how many claims were affected, what
    is missing from each, and the first 80 characters of the claim so a reader can tell a
    truncated restatement from a confident one that simply forgot its citations.
    """
    subjects = "; ".join(f"{f['why']} ({f['claim']!r})" for f in findings) or \
        "no claims classified"
    return (f"theory {slot!r} returned {len(findings)} claim(s) that cannot be cited"
            f" — {subjects}. A claim that points at nothing is not evidence to rank, so "
            "the slot is refused and the rest of the fan-out continues (#3241).")


def no_claims_message(slot: Any) -> str:
    """The complaint, in the one wording both the raise and the refusal row use."""
    return (f"theory {slot!r} handed back no claims — "
            "a prose conclusion is not evidence to rank")


def turn_cap_message(slot: Any, num_turns: Any = None, *,
                     subject: str | None = None) -> str:
    """The complaint for a run the turn budget stopped, in its own wording.

    A separate wording is the point (#2807): `no_claims_message` says the model
    answered in prose, and reusing it on a truncated dispatch would put that
    sentence on a row whose model never got to answer. `num_turns` is the count
    the harness reported, so the row also says how far it got.

    `subject` is who the sentence is about — ONE definition read twice (#2984).
    Arm B's refusal row leaves it unset and keeps saying `theory 'c2'`; arm A's
    no-claims raise passes its own investigator, because that arm has one
    dispatch and no slots to name. A second truncation wording for the same
    mechanism is the drift #2807 was written against: two sentences for one cause
    is how a reader books a budget follow-up against the wrong arm.
    """
    reached = (f" at its {int(num_turns)}-turn budget"
               if isinstance(num_turns, (int, float)) else " at its turn budget")
    who = f"theory {slot!r}" if subject is None else subject
    return (f"{who} ran out of turns{reached} — stop_reason "
            f"{STOP_MAX_TURNS!r}: the harness cut the run off before any structured "
            "answer was restated, so this is truncation, not a theory that "
            "declined to cite")


def arm_a_no_claims_complaint(arm: Any, case_key: Any, result: dict) -> str:
    """Arm A's complaint for an answer with no cause, now naming its mechanism.

    The base sentence is the one this arm has always raised: an empty
    `chosen_cause` is worse than a wrong one, so the arm refuses the row rather
    than fill one in from the case's own answer key. What it never said is WHY
    there was nothing to record, and the two answers need opposite follow-ups — a
    run the harness cut off at its turn budget wants a dispatch budget, a model
    that finished and produced nothing wants a prompt. Both values were already in
    `result`: `investigate_live` re-emits the pair (#2807) and arm B has read it
    since, while arm A's message listed only the KEYS, printing the words
    `stop_reason` and `num_turns` and telling the reader nothing — worse than
    silence, because it shows the pair came back.

    The pair is read exactly as arm B reads it: a falsy `stop_reason` is ABSENCE,
    not a stop, and `num_turns` is a count only when the harness reported a number
    ("absence is not evidence of a cap"). An investigator that reports neither —
    the older single-shape fakes, or any wrapper that rebuilds the dict — gets the
    complaint byte for byte as it did before, with nothing invented beside it.
    """
    base = (f"the investigator for case {case_key} returned no non-empty claim, so "
            f"there is no cause to record and inventing one is the one thing this "
            f"instrument must not do.")
    stop_reason = result.get("stop_reason") or None
    reported_turns = result.get("num_turns")
    num_turns = (int(reported_turns)
                 if isinstance(reported_turns, (int, float)) else None)
    how = ""
    if stop_reason == STOP_MAX_TURNS:
        how = " " + turn_cap_message(
            None, num_turns, subject=f"The investigator on arm {arm}") + "."
    elif stop_reason:
        turns = (f" and num_turns {num_turns}" if num_turns is not None
                 else " with no turn count reported")
        how = (f" It reported stop_reason {stop_reason!r}{turns}, so the run reached "
               f"its stop and the model itself handed back nothing to record.")
    return f"{base}{how} Keys it did return: {sorted(result)}."


def validate_investigation(row: dict) -> list[dict]:
    """A per-theory row must carry claims with citations, not a conclusion.

    Pienaar's fourth failure class is a sub-agent handing back a prose summary;
    the ranking is only over evidence if the record cannot hold anything else.
    Since #2635 arm B turns the no-claims raise into a recorded refusal row before it
    gets here, and since #3241 it does the same for a claim that cannot be cited — by
    #2985's owed ruling 4 ("an uncitable claim becomes a recorded refusal row (kind:
    refusal, reason 'uncitable-claim'), the same route"), on the evidence of six cases
    the raise cancelled outright: `~/lloyd-data/eval/2258/run.log:32,36,62,86,88` and
    `smoke-b3.log:1`, each one a case that had already paid for its other two
    dispatches and then vanished from `report`'s denominator.

    So the return is the CITABLE SUBSET, not the row: a mixed row gives back the claims
    that do point somewhere, and unevidenced text cannot enter the evidence map even if a
    future caller forgets to consult `uncitable_claims` first. Whether the SLOT is refused
    over it is `run_case_b`'s decision — a theory with one citable claim and one uncitable
    one answered, and this function has no business deciding otherwise. The empty-row
    raise stands as arm B's backstop: `cmd_run` is this function's only caller, arm A
    validates separately (#2984), and #2635's refusal route reads `empty_claims` itself.
    """
    if empty_claims(row):
        raise RecordError(no_claims_message(row.get("slot")))
    out = []
    for claim in row["claims"]:
        text = str((claim or {}).get("claim") or "").strip()
        cites = (claim or {}).get("citations")
        if not text or not isinstance(cites, list) or not cites:
            continue  # #3241: classified by `uncitable_claims`, refused by `run_case_b`
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
                 rank_fn: Callable[[str, Sequence[str]], Any],
                 refused: Sequence[str] = ()) -> dict:
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

    `refused` (#2635, widened by #3241) names the slots the caller recorded a refusal for:
    a theory that handed back prose instead of claims at all (`no-claims` / `turn-cap`),
    and one whose claims could not be cited (`uncitable-claim`). Both classes reach here
    with nothing in the evidence map for them, so the only thing to decide is who is handed
    to the engine at all — and it is not nothing: a candidate with no evidence under it is a
    slot whose only content is its own label, which is exactly the ranking #2250 measured
    and this file exists to stop. The exclusion is of REFUSED slots only. A slot the token
    ceiling cut off before it was dispatched stays in the list, because the ceiling row
    already says what happened to it and #2258 pinned that fan-out shape; the refused
    classes are about a theory that ran and could not be pointed at anything.
    """
    by_slot = {s: (evidence.get(s) or []) for s in slots}
    pairs = sorted((item for item in by_slot.items() if item[0] not in set(refused)),
                   key=lambda item: item[0])
    candidates = [rank_candidate_text(slot, claims) for slot, claims in pairs]
    query = (f"Which candidate root cause does the collected evidence support for: "
             f"{case.get('input') or ''}")
    if not candidates:
        # Nothing was evidenced, so there is nothing to order and calling the
        # engine would ask it to choose among bare labels. Recorded rather than
        # raised (#2635): the case must still reach its verdict row so it enters
        # `report`'s denominator as the measured zero it is, not as an absence.
        # Since #2807 the WHY is not named here: a case can arrive with a
        # truncated theory and a prose-answering one, and only the per-slot
        # refusal rows know which was which.
        return {"kind": "rank", "ok": False,
                "reason": "no-evidence: every dispatched theory refused (see the "
                          "refusal rows for each reason); nothing was ranked",
                "order": [], "chosen_slot": None, "chosen_cause": None,
                "refused": sorted(set(refused))}
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

    A theory that came back and handed over prose is cut short by a different
    cause, and since #2635 it is recorded the same way instead of raising: one
    `kind: refusal` row, its own `tokens`, and the theories that DID produce
    evidence going on to the ranking and the verdict row. Since #2807 the row also
    carries the harness's `stop_reason` and `num_turns` beside those tokens, and
    the reason is read from them: `turn-cap` for a run the turn budget stopped,
    `no-claims` for a run that reached its stop and answered with nothing
    citable. Before that the raise propagated out of `run` — `cmd_run` guarded
    nothing — so case 1 of a 21-key pass killed the remaining 20 and wrote no
    `verdict` row for itself, which made it invisible to `report`'s denominator
    rather than merely unscored.

    #3241 brought a fourth refusal word into that route, in the same shape and for the
    same reason: `uncitable-claim`, for a theory whose claims name no file and line, which
    until then was the one answer that still cancelled a case. A max_turns stop still
    outranks it — #2807's precedence, since a run that never reached a finalisable answer
    is a budget question and not a citation question — so the words together are
    `token-ceiling` (never dispatched), `turn-cap` (stopped by turns), `no-claims` (ran,
    answered nothing) and `uncitable-claim` (answered, pointed at nothing).
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
    refused: list[str] = []
    for entry in theory_rows:
        if spent >= token_ceiling:
            stopped = "token-ceiling"
            break
        result = await investigate(case, entry["theory"])
        tokens = int(result.get("tokens") or 0)
        spent += tokens
        probed = {**result, "slot": entry["slot"]}
        if empty_claims(probed):
            # Recorded, not raised (#2635). The raise's purpose survives — the slot
            # is added to `refused`, so `rank_theories` never hands it to the engine
            # and nothing unevidenced is ranked — but a theory's answer is no longer
            # allowed to be a case's, and a case's is not allowed to be the pass's.
            # The tokens it did cost go on the row, so the spend the study reports
            # includes the dispatch that produced nothing.
            #
            # The pair the harness reported goes on the row too (#2807), and the
            # reason is chosen from it. A run the turn budget stopped has no
            # structured answer at all (`FINALIZABLE_STOP_REASONS`,
            # `app/harness/finalizer.py:78`), so its claim list is empty for a
            # reason with a completely different follow-up from a model that
            # finished and answered in prose; recording both as `no-claims` is what
            # made the truncation question unanswerable from the Oct 13 pass. An
            # investigator that reports no stop leaves both cells null and keeps
            # `no-claims` — absence is not evidence of a cap.
            stop_reason, num_turns = stop_pair(result)
            turn_capped = stop_reason == STOP_MAX_TURNS
            refused.append(entry["slot"])
            writer.append({"seq": next_seq(), "run_id": run_id, "case_key": case_key,
                           "arm": ARM_B, "kind": "refusal", "ts": _now(),
                           "slot": entry["slot"], "tokens": tokens,
                           "stop_reason": stop_reason, "num_turns": num_turns,
                           "reason": REFUSAL_TURN_CAP if turn_capped
                           else REFUSAL_NO_CLAIMS,
                           "detail": turn_cap_message(entry["slot"], num_turns)
                           if turn_capped else no_claims_message(entry["slot"])})
            continue
        uncitable = uncitable_claims(probed)
        if uncitable:
            # #3241, by #2985's owed ruling 4 — the same route as above, decided the same
            # way: the slot joins `refused` so nothing unevidenced is ranked, its tokens go
            # on the row, and the other two theories finish the case. What the raise bought
            # in exchange was a `FAILED <case> B` line in a log and no verdict row, four
            # times on Oct 13 and twice more on Oct 18.
            #
            # A max_turns stop still outranks this, and for the reason #2807 filed: the
            # harness's own word that the run never reached a finalisable answer has a
            # different follow-up (a bigger turn budget) from a model that finished and
            # forgot to cite, and booking the wrong one is what made the Oct 13 pass's
            # truncation question unanswerable. Such a row is the harness contract's
            # contradiction — `FINALIZABLE_STOP_REASONS` means a capped run has no claims
            # at all, which the branch above already caught — so it is labelled for the
            # stop and keeps its claim text in the detail either way.
            stop_reason, num_turns = stop_pair(result)
            turn_capped = stop_reason == STOP_MAX_TURNS
            refused.append(entry["slot"])
            writer.append({"seq": next_seq(), "run_id": run_id, "case_key": case_key,
                           "arm": ARM_B, "kind": "refusal", "ts": _now(),
                           "slot": entry["slot"], "tokens": tokens,
                           "stop_reason": stop_reason, "num_turns": num_turns,
                           "reason": REFUSAL_TURN_CAP if turn_capped
                           else REFUSAL_UNCITABLE_CLAIM,
                           "detail": turn_cap_message(entry["slot"], num_turns)
                           if turn_capped
                           else uncitable_message(entry["slot"], uncitable)})
            continue
        # What still raises is what no classifier describes: a non-disjoint theory set
        # (`validate_theories`) and an empty claims list reaching a caller outside the
        # fan-out's refusal route. `cmd_run` contains those at the arm rather than
        # letting one take the pass, which is #2635's own containment argument.
        claims = validate_investigation(probed)
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

    ranking = rank_theories(case, evidence, slots=labels, rank_fn=rank_fn,
                            refused=refused)
    if ranking["ok"]:
        ranking["chosen_cause"] = theory_for(ranking["chosen_slot"], theory_rows)
    if refused:
        ranking["refused"] = sorted(refused)
    writer.append({"seq": next_seq(), "run_id": run_id, "case_key": case_key,
                   "arm": ARM_B, "ts": _now(), **ranking})
    verdict = {"seq": next_seq(), "run_id": run_id, "case_key": case_key,
               "arm": ARM_B, "kind": "verdict", "ts": _now(),
               "chosen_cause": ranking.get("chosen_cause") or "",
               "order": ranking.get("order") or [], "tokens": spent}
    if refused:
        # Which of the three the top-1 was actually chosen among. `report` scores
        # this verdict like any other, so the denominator stays honest by counting
        # the case; this is how a reader sees the case was graded on two theories
        # rather than three without going back through the refusal rows.
        verdict["refused"] = sorted(refused)
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
    # The investigator is a tool-reading trace, not an essayist. `investigate_live`, the
    # one the owed study runs, returns `{"tool", "claims", "tokens", "model"}` and has
    # never returned a `conclusion` key or a top-level `citations` key — #2258's review
    # advisory of 2026-10-06 named this line and was never fixed, so the arm recorded a
    # cause of `""` for every case it ran. The cause therefore has to be read out of the
    # claims the trace actually produced, in the order it produced them.
    #
    # And it must never be recorded empty, which is why an answer with no cause raises
    # rather than writing the row. An empty `chosen_cause` is worse than a wrong one:
    # `agrees()` asks whether every marker of the expected cause occurs in the text, and
    # empty text answers NO to all of them, so both single-trace arms would carry a top-1
    # of 0/N that is a property of this wiring rather than of the engine — and
    # `format_report` subtracts that zero from arm B's measured rate before it prints a
    # HEADLINE. `validate_investigation`, which gates arm B's per-theory rows, already
    # refuses a record with no claims for the same reason; the two arms treat it alike now.
    claims = [c for c in (result.get("claims") or []) if isinstance(c, dict)]
    if not claims and str(result.get("conclusion") or "").strip():
        # The older shape an investigator may still answer in: one conclusion, with its
        # citations beside it rather than under it. Kept from before this item, because
        # dropping it would turn a reply this arm can grade into a refused run — and it
        # keeps the citations, which is the column the study compares against arm B's.
        claims = [{"claim": str(result["conclusion"]).strip(),
                   "citations": list(result.get("citations") or [])}]
    chosen = next((str(c.get("claim") or "").strip() for c in claims
                   if str(c.get("claim") or "").strip()), "")
    if not chosen:
        # Still a raise, and still BEFORE the writer is touched: the refused cell
        # must leave no row, or `report` counts an unmeasured arm as a miss. What
        # the message now does is say which of the two nothings happened, from the
        # pair `investigate_live` already handed back (#2984) — `cmd_run`'s per-arm
        # guard prints this text into the pass's `run.log`, which is the only place
        # anyone ever reads it, and until now that log could not tell a truncated
        # dispatch from an honest empty answer.
        raise RuntimeError(f"[2258] {arm}: "
                           f"{arm_a_no_claims_complaint(arm, case_key, result)}")
    writer.append({"seq": 1, "run_id": run_id, "case_key": case_key, "arm": arm,
                   "kind": "tool", "ts": _now(), "slot": None,
                   "tool": result.get("tool") or "linear-trace",
                   # Every claim the trace produced, each with the citations attached to
                   # IT. Not only the chosen one: a trace that reported three
                   # hypotheses put three citations on the record, and keeping just the
                   # first would understate this arm's evidence in the very column the
                   # study compares against arm B's.
                   "claims": [{"claim": str(c.get("claim") or ""),
                               "citations": c.get("citations") or []}
                              for c in claims],
                   "tokens": tokens})
    if tokens > token_ceiling:
        writer.append({"seq": 2, "run_id": run_id, "case_key": case_key, "arm": arm,
                       "kind": "ceiling", "ts": _now(), "reason": "over-ceiling",
                       "tokens_spent": tokens, "token_ceiling": token_ceiling})
    verdict = {"seq": 3, "run_id": run_id, "case_key": case_key, "arm": arm,
               "kind": "verdict", "ts": _now(), "chosen_cause": chosen,
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
    # `stop_reason` and `num_turns` come along because they are the only pair that
    # says WHY an answer is empty (#2807). The harness emits both on the same
    # `result` event as `structured` and `usage` (`app/harness/events.py:324,326`),
    # so reading only two of the four fields turned a run the turn budget stopped
    # into the same record as one that answered in prose.
    structured: dict = {}
    usage: dict = {}
    stop_reason = ""
    num_turns = 0
    async for evt in run_query([{"role": "user", "content": prompt}], options):
        if evt["type"] == "result":
            structured = evt.get("structured") or {}
            usage = evt.get("usage") or {}
            stop_reason = str(evt.get("stop_reason") or "")
            num_turns = int(evt.get("num_turns") or 0)
    return {"structured": structured,
            "tokens": int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0),
            "model": alias,
            "stop_reason": stop_reason,
            "num_turns": num_turns}


async def generate_theories_live(case: dict, *, max_turns: int = 6) -> list[str]:
    """Three disjoint candidate causes, from the symptom alone.

    The prompt carries the incident and nothing else — no log path, no ledger
    line — because a theory generated after reading a log is anchored to it,
    which is the failure class the whole technique exists to fight.

    The 6-turn budget is measured, and it is left exactly as shipped because the
    measurement says it holds. Both #2258 arm-B smoke runs reached this step
    under this default (`run_case_b` calls `generate(case)` with no override):
    `smoke-b2.jsonl` run-20261013T124552Z and `smoke-b3.jsonl`
    run-20261018T220131Z each hold a `kind: theories` row carrying 3 disjoint
    non-empty theories with `generated_before_evidence: true` — b2 names
    `T1_suite_interference`, `T2_diff_broke_the_deferral_contract` and
    `T3_wrong_thing_was_measured`; b3 fills slots c1/c2/c0 — so the step produced
    a full answer where the old note claimed no run had ever got here. Generation
    took 47–49 s: `run_id` is stamped at run start, so the span is the row's own
    `ts` minus its run id — b2 12:45:52Z → 12:46:41Z = 49 s, b3 22:01:31Z →
    22:02:18Z = 47 s. Reading nothing is why that is cheap here and why
    `investigate_live` needed the 60 turns it now carries: this step has no
    evidence to gather, only a symptom to reason from. The rows record no
    `num_turns`, so 6 stays a shipped default that returned a usable answer
    twice, not a measured ceiling.
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
                           max_turns: int = DEFAULT_INVESTIGATION_TURNS) -> dict:
    """One scoped per-theory investigation: evidence for THIS theory only.

    The budget is `DEFAULT_INVESTIGATION_TURNS`, 60 turns, up from the 40 that used
    to be spelled out here — and that 40 is not a number the study can keep shipping,
    because it is the cap the study's own truncation was measured under: the arm-B
    theory in smoke-b3 came back `stop_reason: max_turns` at `num_turns: 41` having
    spent 161,011 tokens
    (`~/lloyd-data/eval/2258/smoke-b3.jsonl`, line 2, `run-20261018T220131Z`), the
    harness reporting 41 turns against a cap of 40.

    Both figures are measured rather than chosen, and they bracket the cost rather
    than averaging out. Under that same cap of 40, a real investigation of the case
    `flaky-SM_20260928_080412` on this engine took 27 turns to reach
    `stop_reason: stop`
    (`~/lloyd-data/eval/2258/probe-maxturns-40.log`, line
    `META: {"stop_reason": "stop", "num_turns": 27, ...}`); the same case at 20 ended
    at 21 turns with `stop_reason: max_turns`. 60 is a round number above the only
    truncation on record, not a figure read off a 60-turn run — no such run exists
    yet, and the per-theory token cost at it is what the ceiling above is sized
    against rather than measured at (#2985's owed live pass measures it).

    Running out of turns is not a short answer, it is NO answer. A run that ends
    `max_turns` gets no `final_schema` restatement — the finalizer's
    `FINALIZABLE_STOP_REASONS` is `{stop, end_turn}` with "no verdict to restate"
    as the stated reason (`app/harness/finalizer.py:66`), and the P6b wrap-up
    leaves `stop_reason` at `max_turns` rather than changing that
    (`app/harness/options.py:33-41`). So `out["structured"]` came back `{}`, the
    claim list came back empty, and `run_case_a` raised — which read as "the model
    found no evidence" when the instrument had simply stopped it early. Cost of
    the raised budget, measured on the same probe: 601,744 ms and 179,186 tokens
    for ONE arm-A investigation.
    """
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
    # `stop_reason` and `num_turns` are re-emitted rather than dropped (#2807).
    # This rebuild is the second hop that used to discard them, and patching only
    # `_engine_turn` would have fixed nothing the arm could see: `run_case_b`
    # writes its refusal row from THIS dict.
    return {"tool": "Task", "claims": out["structured"].get("claims") or [],
            "tokens": out["tokens"], "model": out["model"],
            "stop_reason": out.get("stop_reason") or "",
            "num_turns": int(out.get("num_turns") or 0)}


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
        # "graded", not "corpus", and #2466 is why the word matters: `n` here is the set
        # being reported on, which a `--only` subset makes smaller than the file the
        # baseline's fingerprint was taken from. Calling 2 a corpus count would print an
        # N that no other line in the artifact agrees with.
        problems.append(f"baseline N {baseline.get('n')!r} != graded N {n}")
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
                  citation_root: Path | None = None,
                  all_cases: Sequence[dict] | None = None) -> tuple[list[str], bool]:
    """The printed lines, and whether the headline states an evaluation.

    `cannot evaluate` is the only thing a missing or mismatched retry baseline
    may produce — a study that has not measured its compute-matched baseline
    cannot say the fan-out won, because it cannot say the arms were matched.

    `cases` is what gets GRADED, which since #2466 need not be the whole corpus: the
    study runs over the 21 non-leaky cases and leaves the 77 `preexisting` ones out.
    `all_cases` is the corpus those 21 were cut from, and it exists so the artifact can
    say so; pass it nothing and the report states that it graded everything it was given.
    """
    scored = score(rows, cases, citation_root=citation_root)
    a, b = scored["arms"][ARM_A], scored["arms"][ARM_B]
    # The ceiling is on the artifact because the whole comparison is conditional on it:
    # the arms are matched by token budget and by nothing else, and a reader who cannot
    # see the number cannot check that the spend line below it is a like-for-like.
    lines = [f"[2258] cases: {len(cases)}  recorded verdicts: "
             f"A={a['n']} B={b['n']}  ceiling: {token_ceiling}"]
    if scored["unscored"]:
        lines.append(f"  unscored case keys (no case row): {scored['unscored']}")
    # Which cases this artifact is NOT about, before any rate and on the `cannot
    # evaluate` branch too — a reader who stops at a refused headline still needs the
    # study's scope, and one who reads a headline needs to know it does not cover the
    # 77 leaky cases (#2258 excludes them because their input quotes the refuted claim).
    lines.append("  " + excluded_note(all_cases if all_cases is not None else cases,
                                      cases))
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
    # `select_cases`, not a per-case membership test, so a key that names no case in the
    # file stops the run before the first engine call (#2466). The old form skipped
    # unmatched keys silently, so one typo in a 21-key list bought a study of 20 — or, if
    # the whole list typo'd, 98 cases including the 77 leaky ones the study excludes.
    cases, bad = select_cases(cases, args.only)
    if bad:
        print(f"[2258] --only names {len(bad)} key(s) with no case in {args.cases}: "
              f"{', '.join(bad)} — nothing was run")
        return 2
    writer = RowWriter(Path(args.rows))
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    ceiling = args.ceiling
    # #2635: one case's outcome may not cancel the pass. Both arms raise on
    # purpose — arm A refuses to record an empty cause, and arm B's validator
    # still refuses anything that is not citable evidence — and until now nothing
    # here guarded either, so the traceback left in `2258/smoke-b.log` propagated
    # out of `run` at theory 2 of case 1: the remaining 20 graded keys never ran,
    # and the dead case wrote no `verdict` row, which makes it ABSENT from
    # `report`'s denominator rather than counted as the miss it is. Guarded per
    # ARM, not just per case, because arm A dying also cancelled that same case's
    # arm B — the half of the comparison the study exists to make.
    failed: list[str] = []

    async def attempt(label: str, make):
        try:
            return await make()
        except Exception as exc:  # noqa: BLE001 — #2635: the pass outlives one arm
            failed.append(label)
            print(f"[2258] FAILED {label}: {type(exc).__name__}: {exc}")
            print(f"[2258] {label} recorded no verdict row, so `report` will not "
                  f"count it; the remaining cases continue")

    for case in cases:
        key = case["key"]
        if ARM_A in arms:
            await attempt(f"{key} {ARM_A}",
                          lambda c=case: run_case_a(c, investigate=investigate_live,
                                                    writer=writer, run_id=run_id,
                                                    token_ceiling=ceiling))
        if ARM_RETRY in arms:
            await attempt(f"{key} {ARM_RETRY}",
                          lambda c=case: run_case_a(c, investigate=investigate_live,
                                                    writer=writer, run_id=run_id,
                                                    arm=ARM_RETRY,
                                                    token_ceiling=ceiling))
        if ARM_B in arms:
            verdict = await attempt(
                f"{key} {ARM_B}",
                lambda c=case: run_case_b(c, generate=generate_theories_live,
                                          investigate=investigate_live,
                                          rank_fn=rank_live, writer=writer,
                                          run_id=run_id, token_ceiling=ceiling))
            if verdict is not None:
                print(f"[2258] {key} B chose: {str(verdict['chosen_cause'])[:90]}")
    print(f"[2258] rows appended to {args.rows} (run {run_id})")
    if failed:
        # Non-zero, and `launch-pass.sh` logs it as `PASS_EXIT=n`. `set -u` is the
        # only shell option there, so the number is recorded rather than fatal —
        # which is the point: a pass that lost an arm must not look like a clean
        # one to whoever reads the log in the morning.
        print(f"[2258] {len(failed)} arm-run(s) failed: {', '.join(failed)}")
        return 1
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
    all_cases = load_cases(cases_path)
    # #2466. Without this the study could not report itself: `run --only` recorded 21
    # verdicts, `cmd_baseline` wrote n=21 from them, and `report` graded all 98 rows of
    # the same file against that baseline, so the only headline the sequence could print
    # was `baseline N 21 != corpus N 98`. The file stays whole — the fingerprint below is
    # still of the file, not of the cut, and the exclusions are printed.
    cases, bad = select_cases(all_cases, args.only)
    if bad:
        print(f"[2258] --only names {len(bad)} key(s) with no case in {args.cases}: "
              f"{', '.join(bad)}")
        return 2
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
        citation_root=(Path(args.root) if args.root else None),
        all_cases=all_cases)
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
    # The same shape as `run --only`, deliberately: the subset that was run is the subset
    # that gets graded, so the two lists are one list and a subset can be re-graded
    # without cutting the corpus file (#2466). A key naming no case is refused rather than
    # quietly shrinking the study.
    report.add_argument("--only", nargs="*", default=None,
                        help="case KEYS to grade — a subset of the same names "
                             "`run --only` takes. The corpus file stays whole, the "
                             "retry baseline must match THIS subset's N, and the "
                             "report states how many cases were excluded")
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
