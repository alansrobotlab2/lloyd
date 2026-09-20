#!/usr/bin/env python3
"""consolidation_source_gate.py — Phase 1.3's source gate (backlog #1287).

`scripts/mine-trajectories.py::is_emittable()` (landed under #1181) refuses to write
a candidate file for a `sequence` pattern flagged `has_error_recovery: false`: an
n-gram with no `:ERR` step in it is a call order, not a lesson, and authoring rule 5
forbids it as skill content. That refusal is an *emit-time* gate. #1181 said so in
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

Age is never a reason. The gate reads `has_error_recovery` and nothing else, so the
108 eligible keys flagged `true` still reach the evidence gate no matter how old their
newest snapshot is — which is what #1181's own measurement demands: 672 of 780
actionable keys read false, 108 carried a real recovery.

Usage:
  consolidation_source_gate.py check --candidates ~/lloyd/_pipeline/skills/candidates/
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
DEFAULT_CANDIDATES = REPO_ROOT / "_pipeline" / "skills" / "candidates"

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


def evidence_eligible(rows: list[tuple[str, dict]], terminal: frozenset[str]) -> bool:
    """Phase 1.2/1.3's evidence gates, on the newest snapshot: not already
    dispositioned, `sessions >= 3`, and 2+ dated snapshots."""
    newest = rows[-1][1]
    if newest_status_is_terminal(newest.get("status", ""), terminal):
        return False
    return _sessions(newest) >= 3 and len(rows) >= 2


def scan(candidates_dir: Path) -> dict:
    """Apply the Phase 1.3 evidence gates, then the source gate, to a corpus."""
    terminal = terminal_statuses()
    groups = group_candidates(Path(candidates_dir))
    eligible: list[tuple[str, dict, str | None]] = []
    for key in sorted(groups):
        rows = groups[key]
        if not evidence_eligible(rows, terminal):
            continue
        newest = rows[-1][1]
        eligible.append((key, newest, drop_reason(newest)))
    dropped = [row for row in eligible if row[2] is not None]
    return {
        "scanned": len(groups),
        "eligible": len(eligible),
        "dropped": dropped,
        "kept": [row for row in eligible if row[2] is None],
        "dropped_by_source_gate": len(dropped),
        "eligible_after_source_gate": len(eligible) - len(dropped),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def cmd_check(args: argparse.Namespace) -> int:
    result = scan(Path(args.candidates).expanduser())
    for key, newest, reason in result["dropped"]:
        print(f"DROP {key} :: {reason} (newest snapshot, sessions={newest.get('sessions')})")
    for key, _newest in [(k, n) for k, n, r in result["kept"]]:
        print(f"KEEP {key}")
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
