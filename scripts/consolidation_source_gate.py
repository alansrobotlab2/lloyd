#!/usr/bin/env python3
"""consolidation_source_gate.py — Phase 1.3's source gate (backlog #1287).

`scripts/mine-trajectories.py::is_emittable()` (landed under #1181) refuses to write
a candidate file for a `sequence` pattern flagged `has_error_recovery: false`. What
that flag means changed under #1327 and the gate deliberately did not: a sequence is
flagged true only when some *observed instance* of the n-gram recovered — a later
step in the same window re-attempted the failed call (same tool, same Bash command
category, so the same label without `:ERR`) or named the object the failure named
(an argument value equal to one of the failed call's object-naming values: a path,
URL, filename or dotted id). An `:ERR` step followed by any old next step sets
nothing, which is what makes a kept key evidence rather than a call order that
happened to sit next to a failure; authoring rule 5 still forbids the latter. That refusal is an *emit-time* gate. #1181 said so in
its own acceptance notes — "What the fix does NOT cure. It is an emit-time gate: the
~2,900 historical candidate files with `has_error_recovery: false` … the pool decays
only as they age out" — and the consequence is still on disk: `nightly-skill-consolidation`
Phase 1.3 had no matching rule, so every historical file keeps passing the
`sessions >= 3` and 2-snapshot evidence gates and gets hand-adjudicated five keys a
night for a class the code already decided. Measured 2026-09-20T08:0xZ over all 3,930
candidate files: 761 keys clear Phase 0's verdict drop, `sessions >= 3` and the
2-snapshot persistence gate; **761 of 761 are `type: sequence`**, 653 of them flagged
`has_error_recovery: false`, and all 761 carry a newest snapshot on or before
2026-09-13. At five patterns a night that is ~152 nights of hand-adjudication to reach
a verdict `is_emittable()` computes in one comparison. `run_58_20260920_104452` spent
all five slots on exactly that class (84/83/83/81/79 sessions each) and verdicted all
five `rejected_artifact_class`.

So this is Phase 1.3's copy of Phase 0's mechanism. `skill_verdicts.py check` already
does the identical shape for the verdict class — one `SKIP <pattern> :: <verdict> ::
<reason>` line per key plus a `skipped_by_verdict: N` total that must be printed even
when N is the whole run (0.4) — and this tool does it for the emitter class, printing
`DROP <pattern> :: not_emittable :: …` plus `dropped_by_source_gate: N`.

**The decision is imported, never re-typed.** `is_emittable()` lives in a
hyphen-named file, so it is loaded by path (the same way
`tests/test_trajectory_extraction.py` loads it) and `tests/test_consolidation_source_gate.py`
asserts the gate holds *that function object*, not a copy of its body. A hand-copied
rule is what the item forbids: the two would drift and Phase 1.3 would silently drop
a class the miner emits again. The pinning test checks the safe direction for every
pattern shape both tools recognise — a key the emitter keeps is never counted in
`dropped_by_source_gate` — so a future widening of `is_emittable()` moves the gate,
and a future narrowing can only ever under-drop, which is hand-adjudication, not a
lost lesson.

Two deliberate asymmetries, both toward keeping a key:

* **The flag is matched as a flag, not as a string.** A candidate's front matter is
  text, so a naive read hands `is_emittable()` the string `"false"`, which is not
  `False`, and the gate would drop zero keys forever while reporting success. Scalars
  are coerced here (`"false"` → `False`) and the coercion is pinned by a test.
* **A key is dropped only when its own front matter proves the refusal.**
  `is_emittable()` reads `type` and then one more field: `has_error_recovery` for a
  sequence/error pattern, `params_signature` for a `success` one. Candidate front
  matter has never carried `params_signature` (census 2026-09-20: 0 of 102 `success`
  keys — `write_candidate_file` does not write it), so passing an absent one would
  assert a signature the file never recorded and drop all 102, which is exactly the
  false drop clause 5 of #1287 forbids ("keys with no `has_error_recovery` field are
  never counted"). So a `success` key is only adjudicated when its file recorded the
  signature; every other key is, because an absent flag is a keep under the emitter's
  own `is False` test. Under-dropping is a slower run; over-dropping is a lost lesson.

Age is never a reason, and neither is a count. The gate reads the key's
`has_error_recovery` flag and nothing else, so every eligible key whose own front
matter proves a recovery reaches the evidence gate no matter how old its newest
snapshot is, and every key whose front matter proves it does not is dropped here —
the decision is `is_emittable()`'s, on whatever key set is on disk tonight. How many
keys land on either side moves with the corpus and with the flag's own semantics, so
this docstring deliberately carries no number for it: the counts it used to attest
("672 of 780 actionable keys read false, 108 carried a real recovery") were a
snapshot of the adjacency rule that #1327 retired, and one of that 108's
successors was a `date -u` after a failed grep.

A candidate file's flag is the value the miner wrote when it emitted that snapshot,
so historical files keep the meaning they were written with: a key flagged true
under the adjacency rule stays kept by this gate, because the gate drops a key only
on its own front matter and under-dropping is hand-adjudication rather than a lost
lesson. Newly mined candidates carry the new meaning, so the pool this gate keeps
shrinks towards the keys that really recovered.

**A verdict that has lapsed stops being a filter.** The skip list above is
`terminal_statuses()`, and it includes the status the miner writes when the ledger
blocks a key: `superseded_by_verdict`. That status is a *copy* of a ledger row, and the
row expires — `skill_verdicts.py` reopens a terminal verdict 60 days after
`decided_at`, so on the night the leg crosses the ledger says "propose freely" while
the file still says "already dispositioned", and `evidence_eligible()` filtered on the
file before `is_emittable()` was ever asked. Measured 2026-11-09T11:5xZ over the live
corpus: `skill_verdicts.py check` printed 43 `REOPEN` rows over 6 keys while this
tool's last line read `scanned: 12  eligible: 0  dropped_by_source_gate: 0
eligible_after_source_gate: 0` — two of those keys clear `sessions >= 3` and the
2-snapshot gate and landed in neither bucket. So for that one status the gate now asks
the ledger through `terminal_verdict()`, the same function `check` weighs a `REOPEN`
on, and releases the key when it answers None; the ledger is read once per scan and
never at all for any other status. Dropping `SUPERSEDED_STATUS` from the set wholesale
is the 09-09 re-adjudication loop #530 exists to prevent, and it stays in: the guard is
that the ledger is asked, not that the status is ignored. A release is named on the
`released_by_lapsed_leg: N :: a, b` line (``0 :: none`` on a quiet night), in
`ledger_reopened_keys`' grammar, so the counts above can never again be published over
a pool emptied upstream of the emitter with nothing to show for it.

Usage:
  consolidation_source_gate.py check --candidates ~/lloyd-data/_pipeline/skills/candidates/
  consolidation_source_gate.py drop-reason --frontmatter '<yaml-ish dict>'   # debug
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
MINER_PATH = REPO_ROOT / "scripts" / "mine-trajectories.py"
VERDICTS_PATH = REPO_ROOT / "scripts" / "skill_verdicts.py"
sys.path.insert(0, str(REPO_ROOT))
from app.paths import PIPELINE_DIR  # noqa: E402

DEFAULT_CANDIDATES = PIPELINE_DIR / "skills" / "candidates"

# The front-matter scalars a reader may meet for a boolean field. Anything else is
# not a flag and must not be guessed at: an unrecognised value keeps the key.
TRUE_WORDS = ("true", "yes")
FALSE_WORDS = ("false", "no")

_DATE_SUFFIX_RE = re.compile(r"^(candidate-?.*?)-(\d{8})\.md$")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n(?:---|\.\.\.)", re.S)
_FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


# ── loading the two rules this tool is not allowed to restate ────────────────

def _load_by_path(path: Path, module_name: str) -> ModuleType:
    """Import a hyphen-named (or package-less) module by path.

    `scripts/` is not a package, so `import mine_trajectories` cannot work; this is
    the loader `tests/test_trajectory_extraction.py` already uses for the same file,
    so both the suite and this tool reach the emitter the same way.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_MINER: ModuleType | None = None
_VERDICTS: ModuleType | None = None


def miner() -> ModuleType:
    """The trajectory miner module, loaded once. `is_emittable` is read off it."""
    global _MINER
    if _MINER is None:
        _MINER = _load_by_path(MINER_PATH, "lloyd_mine_trajectories")
    return _MINER


def verdicts() -> ModuleType:
    """`skill_verdicts.py`, loaded once — owner of the terminal-verdict vocabulary."""
    global _VERDICTS
    if _VERDICTS is None:
        _VERDICTS = _load_by_path(VERDICTS_PATH, "lloyd_skill_verdicts")
    return _VERDICTS


def is_emittable(pattern: dict) -> bool:
    """The miner's own predicate. Delegating, never reimplementing (clause 1)."""
    return bool(miner().is_emittable(pattern))


def terminal_statuses() -> frozenset[str]:
    """Statuses that mean "already dispositioned", re-derived, never re-typed.

    Phase 1.3's skip list is the ledger's terminal-verdict vocabulary plus the two
    statuses the candidate front matter uses for the same idea and the status the
    miner writes when the ledger blocks a key. Importing it means #830's pending
    `reviewed_authored` addition lands here for free instead of drifting.
    """
    v = verdicts()
    return frozenset(set(v.TERMINAL_VERDICTS) | {"consolidated", "noise",
                                                 v.SUPERSEDED_STATUS})


# ── the gate ────────────────────────────────────────────────────────────────

def coerce_flag(value) -> bool | None:
    """A front-matter `has_error_recovery` value as a real bool, or None.

    None means "not a flag": the field was absent, or its text is neither truth word
    nor falsehood word. The caller keeps the key in that case, which is
    `is_emittable()`'s `is False` test seen from the side of a text file — a falsy or
    string comparison is the trap that would have suppressed every mined `error`
    pattern (#1181 clause 3).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().strip("'\"").lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
    return None


def pattern_from_frontmatter(frontmatter: dict) -> dict:
    """Rebuild the pattern dict `is_emittable()` should judge, from the file's fields.

    Only fields the front matter actually carries go in. A missing field is not a
    false field: `params_signature` has never been written to a candidate file, so
    inventing an empty one would convert "this file cannot answer" into "the emitter
    refuses this", dropping all 102 `success` keys.
    """
    pattern: dict = {}
    ptype = frontmatter.get("type")
    if ptype is not None:
        pattern["type"] = str(ptype).strip().strip("'\"")
    flag = coerce_flag(frontmatter.get("has_error_recovery"))
    if "has_error_recovery" in frontmatter and flag is not None:
        pattern["has_error_recovery"] = flag
    sig = frontmatter.get("params_signature")
    if sig is not None:
        pattern["params_signature"] = str(sig)
    return pattern


def decidable(frontmatter: dict) -> bool:
    """Whether this front matter can answer the emitter's question at all.

    `is_emittable()` reads exactly two fields: `type`, and then `has_error_recovery`
    for anything that is not a `success` pattern or `params_signature` for a `success`
    one. So a non-success key is always decidable — an absent flag is a keep under the
    emitter's own `is False` test, not an unknown. A `success` key is decidable only if
    its file recorded `params_signature`, which candidate front matter has never done
    (census 2026-09-20: 0 of 102 `success` keys); inventing an empty one would convert
    "this file cannot answer" into "the emitter refuses this" and drop all 102, the
    false drop clause 5 of #1287 forbids.
    """
    ptype = _field_text(frontmatter.get("type"))
    if ptype == "success":
        return "params_signature" in frontmatter
    if "has_error_recovery" in frontmatter:
        return coerce_flag(frontmatter.get("has_error_recovery")) is not None
    return True


def _field_text(value) -> str:
    return "" if value is None else str(value).strip().strip("'\"")


def drop_reason(frontmatter: dict) -> str | None:
    """Why this key's whole class the emitter refuses, or None to keep it.

    Takes one candidate key's *newest-snapshot* front matter as a dict and decides
    through `is_emittable()`. The reason string renders the key's own fields; no part
    of the decision is computed from them.
    """
    if not decidable(frontmatter):
        return None
    pattern = pattern_from_frontmatter(frontmatter)
    if is_emittable(pattern):
        return None
    return (
        "not_emittable :: type="
        f"{frontmatter.get('type')!r} has_error_recovery="
        f"{coerce_flag(frontmatter.get('has_error_recovery'))}"
    )


# ── reading the corpus ──────────────────────────────────────────────────────

def parse_frontmatter(text: str) -> dict:
    """The YAML-ish front matter block as a flat dict of raw strings."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).split("\n"):
        field = _FIELD_RE.match(line)
        if field:
            fields[field.group(1)] = field.group(2).strip()
    return fields


def key_of(filename: str, frontmatter: dict) -> str:
    """The pattern key: the file's own `pattern:` field, never a filename glob.

    A key that is a short stem is a strict prefix of other keys' filenames, so
    deriving a key by stripping a suffix off a globbed name collides (Phase 1.2
    step 3, Phase 5.1). The fallback for a file with no `pattern:` field strips one
    trailing hyphen plus exactly eight characters, the same parameter-suffix rule
    the runbook prescribes.
    """
    field = frontmatter.get("pattern")
    if field:
        return field.strip().strip("'\"")
    stripped = re.sub(r"-.{8}$", "", filename[len("candidate-"):] if filename.startswith("candidate-") else filename)
    return stripped[:-3] if stripped.endswith(".md") else stripped


def newest_status_is_terminal(status: str, terminal: frozenset[str]) -> bool:
    """Prefix match, the way Phase 1.3 words it ("`status:` begins with one of")."""
    return any(str(status).startswith(word) for word in terminal)


def group_candidates(candidates_dir: Path) -> dict[str, list[tuple[str, dict]]]:
    """Every candidate file, grouped by key, oldest snapshot first."""
    groups: dict[str, list[tuple[str, dict]]] = {}
    for file in sorted(Path(candidates_dir).glob("candidate-*.md")):
        if _DATE_SUFFIX_RE.match(file.name) is None:
            continue  # INDEX.md and any non-dated artifact are not snapshots
        text = file.read_text(encoding="utf-8", errors="replace")
        frontmatter = parse_frontmatter(text)
        groups.setdefault(key_of(file.name, frontmatter), []).append((file.name, frontmatter))
    for rows in groups.values():
        rows.sort(key=lambda row: row[0])  # YYYYMMDD sorts chronologically
    return groups


def _sessions(frontmatter: dict) -> int:
    try:
        return int(str(frontmatter.get("sessions", "")).strip())
    except ValueError:
        return -1


# ── the ledger leg: a superseded snapshot whose verdict has lapsed (#4003) ────

RELEASED_LABEL = "released_by_lapsed_leg"


def _occurrences(frontmatter: dict) -> int:
    """The snapshot's own `occurrences:` count; 0 when the file never recorded one.

    The growth leg of the reopen rules weighs live occurrences against the
    `occurrences_at_decision` baseline the verdict was written with. A file that
    recorded no count cannot assert growth, and 0 is also what the miner passes when
    `--occurrences` is omitted (#736), so gate and emitter ask the same question.
    """
    try:
        return int(_field_text(frontmatter.get("occurrences")) or 0)
    except ValueError:
        return 0


def ledger_verdict_table() -> dict | None:
    """One ledger read for a whole scan, or None when there is nothing to ask.

    None is deliberately not an empty table: `terminal_verdict` answers None for every
    key when no ledger survives in either tree — and for every key when the file is
    present but emptied, which `check` treats as a different event from a deleted one
    and refuses to paper over. Releasing on either reading is the 09-09 re-adjudication
    loop #530 exists to prevent, so an unreadable or empty ledger keeps filtering
    exactly as it did before this leg existed.
    """
    v = verdicts()
    if not v.store_path(None).exists():
        return None
    source, _fell_back = v.resolve_verdict_source(None)
    table = v.load_verdicts(source)
    return table or None


def terminal_verdict_binds(key: str, occurrences: int, table: dict) -> bool:
    """The ledger's own answer, delegated to `terminal_verdict`, never recomputed.

    That is the same function `skill_verdicts.py check` weighs a `REOPEN` on, so "the
    leg has lapsed" means one thing here and there: it returns None on a non-terminal
    line or on a line the reopen rules have lifted.
    """
    return verdicts().terminal_verdict(key, occurrences=occurrences,
                                       verdicts=table) is not None


def lapsed_leg_released(key: str, newest: dict, table: dict | None,
                        binds=terminal_verdict_binds) -> bool:
    """Whether this key's `superseded_by_verdict` front matter has outlived its ledger.

    Scoped to the one status the *miner mints from the ledger*. `consolidated`, `noise`
    and every `TERMINAL_VERDICTS` value were written by a consolidator or a hand
    adjudication, so the file is the record of that decision and stays filtered with the
    ledger never consulted. Only `superseded_by_verdict` is a *copy* of a ledger row,
    and for that copy the row — not the file — decides. `SUPERSEDED_STATUS` stays in
    `terminal_statuses()` for the same reason: the guard is that the ledger is asked,
    not that the status is ignored.

    The status test is equality, not the prefix match the skip list uses, so a status
    that merely begins with the word `superseded` cannot borrow this leg.
    """
    if not table or not key:
        return False
    if _field_text(newest.get("status")) != verdicts().SUPERSEDED_STATUS:
        return False
    return not binds(key, _occurrences(newest), table)


def released_line(keys: list[str]) -> str:
    """`released_by_lapsed_leg: 2 :: a, b`, and `released_by_lapsed_leg: 0 :: none`.

    The same grammar and unit (keys, not files, every name printed with no cap) as
    `skill_verdicts.py`'s `ledger_reopened_keys`, because the two lines describe one lap
    of one clock from two ends and a nightly reads them as a pair. It prints the zero
    case for the reason #2644 recorded: a line that appears only when something happened
    cannot tell a quiet night from a comparison nobody ran — which is the exact gap that
    let `dropped_by_source_gate: 0` be published beside a pool emptied upstream of it.
    """
    return (f"{RELEASED_LABEL}: {len(keys)} :: "
            + (", ".join(keys) if keys else "none"))


def evidence_eligible(rows: list[tuple[str, dict]], terminal: frozenset[str],
                      released_by_lapsed_leg: bool = False) -> bool:
    """Phase 1.2/1.3's evidence gates, on the newest snapshot: not already
    dispositioned, `sessions >= 3`, and 2+ dated snapshots.

    `released_by_lapsed_leg` is `scan`'s reading of the ledger for this key
    (`lapsed_leg_released`): a superseded snapshot whose verdict the reopen rules have
    lifted is no longer a disposition, so the key falls through to the evidence gates
    like any other. Every other terminal status reaches this call with it False.
    """
    newest = rows[-1][1]
    if not released_by_lapsed_leg and newest_status_is_terminal(
            newest.get("status", ""), terminal):
        return False
    return _sessions(newest) >= 3 and len(rows) >= 2


def scan(candidates_dir: Path, ledger: dict | None = None) -> dict:
    """Apply the Phase 1.3 evidence gates, then the source gate, to a corpus.

    `ledger` is the verdict table the lapsed-leg check asks; None reads the live one,
    and only when some key on disk actually carries the status the ledger mints — a
    corpus of `pending_review` files costs no ledger read at all, and a read is one
    parse for the whole scan rather than one per superseded key. Pass a table (or `{}`
    for "no ledger behind any of these") to adjudicate a corpus against a known ledger.
    """
    terminal = terminal_statuses()
    groups = group_candidates(Path(candidates_dir))
    superseded = [k for k in sorted(groups)
                  if _field_text(groups[k][-1][1].get("status"))
                  == verdicts().SUPERSEDED_STATUS]
    table = ledger if ledger is not None else (
        ledger_verdict_table() if superseded else None)
    released = [k for k in superseded
                if lapsed_leg_released(k, groups[k][-1][1], table)]
    released_set = set(released)
    eligible: list[tuple[str, dict, str | None]] = []
    for key in sorted(groups):
        rows = groups[key]
        if not evidence_eligible(rows, terminal,
                                 released_by_lapsed_leg=key in released_set):
            continue
        eligible.append((key, rows[-1][1], drop_reason(rows[-1][1])))
    dropped = [row for row in eligible if row[2] is not None]
    return {
        "scanned": len(groups),
        "eligible": len(eligible),
        "dropped": dropped,
        "kept": [row for row in eligible if row[2] is None],
        "dropped_by_source_gate": len(dropped),
        "eligible_after_source_gate": len(eligible) - len(dropped),
        "released_by_lapsed_leg": len(released),
        "released_lapsed_leg_keys": released,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def cmd_check(args: argparse.Namespace) -> int:
    result = scan(Path(args.candidates).expanduser())
    for key, newest, reason in result["dropped"]:
        print(f"DROP {key} :: {reason} (newest snapshot, sessions={newest.get('sessions')})")
    for key, _newest in [(k, n) for k, n, r in result["kept"]]:
        print(f"KEEP {key}")
    # The release is named before the totals, so a `dropped_by_source_gate: 0` can never
    # again sit above a pool that was emptied upstream of the emitter with no evidence
    # that anything was ever behind it.
    print(released_line(result["released_lapsed_leg_keys"]))
    # Totals stay the last line: the runbook and the nightly greps read the counts
    # off `splitlines()[-1]`, the same contract `skill_verdicts.py check` keeps.
    print(
        f"scanned: {result['scanned']}  eligible: {result['eligible']}  "
        f"dropped_by_source_gate: {result['dropped_by_source_gate']}  "
        f"eligible_after_source_gate: {result['eligible_after_source_gate']}"
    )
    return 0


def cmd_drop_reason(args: argparse.Namespace) -> int:
    """Adjudicate one front-matter block, for a run that wants to test a single key."""
    frontmatter = parse_frontmatter(f"---\n{args.frontmatter}\n---\n")
    reason = drop_reason(frontmatter)
    key = frontmatter.get("pattern", "(no pattern field)")
    print(f"DROP {key} :: {reason}" if reason else f"KEEP {key}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="which eligible candidate keys the emitter refuses")
    p_check.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    p_check.set_defaults(func=cmd_check)

    p_one = sub.add_parser("drop-reason", help="adjudicate one key's front matter")
    p_one.add_argument("--frontmatter", required=True, help="front-matter body, without the --- fences")
    p_one.set_defaults(func=cmd_drop_reason)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
