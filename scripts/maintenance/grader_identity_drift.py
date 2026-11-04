#!/usr/bin/env python3
"""Which calibrated judge artifacts were produced by an engine this box no longer serves?

    python scripts/maintenance/grader_identity_drift.py

WHY THIS EXISTS (#3753)
-----------------------
A threshold, a ceiling and a judge-agreement figure are all calibrations OF a judge.
When the thing that produced them is replaced, the number keeps its meaning in the file
and loses it in the world — and nothing says so. The talk this came out of (AI Engineer,
"Building the AI Agent Is Easy. Proving It's Safe Isn't", Jonathan Osborne, NatWest,
2026-11-04) measured the size of the hole: changing ONLY the judge model, with inputs
and outputs frozen, moved a faithfulness reading from roughly 0.8 to roughly 0.2, from
comfortably above its threshold to below it. Their conclusion is the one this script
acts on: the judge model is part of the threshold configuration, so a judge swap is a
breaking change, and the first honest response is to say which calibrations outlive it.

Lloyd has the same shape on disk and no way of seeing it. The live nightly retrieval run
`nightly-20261103` scores `entity_hit_rate` 0.652 against a ceiling of 0.5385 recorded by
`label-agreement-20261014-103853.json`, whose `labeler` block names the model as `djev`
and nothing else — while :8010 answers `/v1/models` with `id: "djev"` as a STATIC
`--served-model-name`. So `recorded == live` for that role is the tautology `djev ==
djev`: it passes forever however the checkpoint behind it changes. The comparable
identity is the checkpoint (`root`), which `eval/label_agreement_ceiling.py` now records
as `labeler.served_root` on every fresh artifact (#3753 clause 5), and which this script
compares for artifacts that have it. Artifacts written before that change — and the
autoresearch judge-compare baseline, which records no judge identity AT ALL — are
reported as `unrecorded`, which is a fact about the record and not a clean bill of
health.

WHAT IT DOES
------------
For every calibrated judge artifact it can find: print its path, how old it is, the judge
identity the artifact recorded, what the engine serving that role says it is serving
right now, and a verdict. Then a count. Nothing else.

Three artifact classes, found by the paths their own writers use:

  * `label-agreement-*.json` under `eval/label_agreement_ceiling.default_out_dir()` —
    the ceiling the retrieval-eval nightly is read against. Role from `labeler.model`,
    recorded identity from `labeler.served_root`.
  * `autoresearch-judge-compare-*.json` under `eval/measurements/` — the judge-agreement
    baseline. Its driver grades with `--model` whose default is `primary`
    (`eval/autoresearch_judge_compare.py`), and the artifact records no identity, so
    every row it produces is `unrecorded`. That IS the finding, printed rather than
    asserted.
  * `*-direct-n*.json` / `*-runtime-harness-n*.json` coverage arms under
    `research_root/coverage/` — the pass@N figures an autoresearch promotion is judged
    on. These already carry a served `model_id` from
    `scripts/autoresearch/coverage_leg.py::measurement_context`, so they are the class
    that can print a real `same` today, and the first to print `different` when the
    primary slot is swapped under a committed arm.

WHAT IT NEVER DOES
------------------
It writes nothing, gates nothing, and refuses nothing: it always exits 0 and the automod
gate never imports it. Acting on a mismatch is a separate, named call — a review rung
that refused whenever the engine moved could refuse a good build because the grader got
stricter, which is why #3749/#1549 established report-and-flag for this class (#3753
step 5). Compare the differing convention of `scripts/maintenance/engine_build_drift.py`,
which owns a build-property check and therefore has a state worth encoding in an exit
code; this one has none.

The live identity comes from the resolvers the repo already has, not a new probe: the
alias→served door is `scripts/autoresearch/coverage_leg.py::resolve_served_model`, and
the djev checkpoint is read through
`eval/label_agreement_ceiling.py::served_checkpoint_root` — the same one
`GET {base}/v1/models` the labeler itself is named with. A role whose engine will not
answer is printed as `unreachable`, never as a match: an unseen checkpoint is exactly
the state this script exists to report.

MATCHING RULE
-------------
Case-insensitive containment, the convention `eval/instruction_compliance_eval.py`
already uses to decide whether a served string is the expected model: an artifact's
recorded identity counts as the same engine when it appears anywhere inside what the
engine now reports (an artifact recorded by checkpoint and an engine reporting
`"<root> <served-name>"` are the same model, and requiring equality would print a
mismatch on every vLLM that appends its served name).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The five states one artifact can be in. `unrecorded` and `unreachable` are NOT
#: mismatches — they are the two ways the question cannot be answered, and both are
#: printed and counted so a reader never mistakes them for the zero.
SAME = "same"
DIFFERENT = "different"
UNRECORDED = "unrecorded"
UNREACHABLE = "unreachable"
NO_IDENTITY = "<none>"

#: `coverage_leg.measurement_context` writes this when its own probe did not answer.
#: Treating it as an identity would let `unresolved == unresolved` read as a match.
COVERAGE_UNRESOLVED = "unresolved"

#: The role the judge-compare driver grades with when no `--model` is passed. Read from
#: its `--model` default, not guessed: that driver is the only writer of the artifact.
JUDGE_COMPARE_ROLE = "primary"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_days(path: Path, ran_at: str | None) -> int:
    """Days between the artifact's own stamp (or its mtime, for one that stamps nothing)
    and now.

    `ran_at`/`generated_at` wins because it is what the instrument wrote about itself.
    mtime is the fallback, and it is the age of the FILE rather than of the run: a copy
    or a fresh checkout moves it, which is why the judge-compare baseline — the one
    class here with no timestamp of its own — is printed with an age a git operation
    could have changed. An unparseable stamp falls back the same way rather than
    printing a blank.
    """
    stamp: datetime | None = None
    if ran_at:
        try:
            stamp = datetime.fromisoformat(str(ran_at).replace("Z", "+00:00"))
        except ValueError:
            stamp = None
    if stamp is None:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return max(0, int((_now() - stamp).total_seconds() // 86400))


def _identity_from_label_agreement(art: dict[str, Any]) -> tuple[str, str]:
    """(role, recorded identity) for a label-agreement ceiling artifact."""
    labeler = art.get("labeler") or {}
    role = str(labeler.get("model") or "") or NO_IDENTITY
    recorded = str(labeler.get("served_root") or "")
    return role, recorded


def _identity_from_judge_compare(art: dict[str, Any]) -> tuple[str, str]:
    """(role, recorded identity) for the autoresearch judge-compare baseline.

    The empty identity is not a parsing gap to fill in here: the artifact records
    `n_traces`, `judges`, `coverage` and `judge_agreement` and no model name, so there
    is nothing on disk to compare. Reporting `unrecorded` is the whole point — naming
    `primary` from the driver's default would substitute an alias for a missing
    measurement, which is the move #3753 exists to stop.
    """
    return JUDGE_COMPARE_ROLE, ""


def _identity_from_coverage(art: dict[str, Any]) -> tuple[str, str]:
    """(role, recorded identity) for a pass@N coverage arm."""
    role = str(art.get("model_alias") or "") or NO_IDENTITY
    recorded = str(art.get("model_id") or "")
    if recorded == COVERAGE_UNRESOLVED:
        recorded = ""
    return role, recorded


#: (class name, artifact glob, identity reader) — the three writers whose numbers are
#: read as calibration, and the one reader each has here. The glob is the writer's own:
#: `label-agreement-*.json` and the `ARTIFACT_GLOB` it matches, the judge-compare
#: driver's `RESULT` name, and the `…-n<N>.json` shape
#: `coverage_leg.artifact_path` builds — so this table finds artifacts where they are
#: written and nothing else in the directory.
CLASSES: tuple[tuple[str, str, Callable[[dict[str, Any]], tuple[str, str]]], ...] = (
    ("label-agreement", "label-agreement-*.json", _identity_from_label_agreement),
    ("judge-compare", "autoresearch-judge-compare-*.json", _identity_from_judge_compare),
    ("coverage-arm", "*-n*.json", _identity_from_coverage),
)


def artifact_rows(kind: str, directory: Path, pattern: str,
                  reader: Callable) -> list[dict[str, Any]]:
    """One row per readable artifact of one class in `directory`.

    A file that will not parse is skipped rather than guessed at, and an absent
    directory contributes nothing — both are reported by the inventory line, so an
    empty run reads as "found none" and not as "found none, all matched".
    """
    rows: list[dict[str, Any]] = []
    if not directory or not Path(directory).is_dir():
        return rows
    for path in sorted(Path(directory).glob(pattern)):
        try:
            art = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(art, dict):
            continue
        role, recorded = reader(art)
        if recorded.strip().lower() == COVERAGE_UNRESOLVED:
            recorded = ""
        rows.append({"kind": kind, "path": str(path), "role": role,
                     "recorded": recorded.strip(),
                     "age_days": _age_days(path, art.get("ran_at")
                                           or art.get("generated_at"))})
    return rows


def compare(recorded: str, live: str) -> str:
    """`same` / `different` / `unrecorded` / `unreachable` for one artifact.

    Order matters and is the whole safety of the verdict: a missing recorded identity
    is answered BEFORE a missing live one, because `"" in "anything"` is True in
    Python — checking containment first would turn every silent artifact into a match.
    """
    rec, liv = (recorded or "").strip(), (live or "").strip()
    if not rec or rec.lower() == COVERAGE_UNRESOLVED:
        return UNRECORDED
    if not liv or liv.lower() == UNREACHABLE:
        return UNREACHABLE
    return SAME if rec.lower() in liv.lower() else DIFFERENT


def judged(rows: list[dict[str, Any]],
           live_for: Callable[[str], str]) -> list[dict[str, Any]]:
    """Attach the live served identity for each row's role and its verdict."""
    cache: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        role = row["role"]
        if role not in cache:
            cache[role] = (live_for(role) or "").strip() if role != NO_IDENTITY else ""
        live = cache[role]
        merged = dict(row)
        merged["live"] = live
        merged["verdict"] = compare(row["recorded"], live)
        out.append(merged)
    return out


def render(rows: list[dict[str, Any]]) -> list[str]:
    """One line per artifact: verdict, path, age, role, recorded, live."""
    return [f"{row['verdict'].upper():<11} {row['kind']:<15} {row['path']}"
            f"  age={row['age_days']}d  role={row['role']}"
            f"  recorded={row['recorded'] or NO_IDENTITY}"
            f"  live={row['live'] or NO_IDENTITY}"
            for row in rows]


def summary_line(rows: list[dict[str, Any]]) -> str:
    """The count clause 4 asks for, with the two unanswerable states counted beside it.

    `mismatched` counts ONLY `different`. A run over artifacts that record no identity
    prints `0 mismatched` alongside `unrecorded=N`, and that pairing is what keeps the
    zero from being read as an all-clear — the lesson recorded as "absence of captured
    corrections is a coverage gap, not a clean bill of health".
    """
    n = lambda v: sum(1 for r in rows if r["verdict"] == v)          # noqa: E731
    return (f"grader identity drift: {len(rows)} artifact(s) — "
            f"{n(DIFFERENT)} mismatched, {n(SAME)} same, "
            f"{n(UNRECORDED)} unrecorded, {n(UNREACHABLE)} unreachable "
            f"[report-only: gates nothing]")


def live_served_identity(role: str) -> str:
    """What the engine behind `role` says it is serving, via the resolvers it has.

    `djev` is not a `models:` slot — it is deliberately absent from that table so no
    turn ever routes there — so its endpoint is the `djev.base_url` block and the
    comparable identity is its checkpoint, read by the function that names the labeler.
    Every other role is an alias, resolved by the one alias→served door in the repo.
    """
    if role == "djev":
        from eval.label_agreement_ceiling import served_checkpoint_root
        try:
            from app.config import CONFIG
            base = str(((CONFIG or {}).get("djev") or {}).get("base_url") or "")
        except Exception:                                              # noqa: BLE001
            return UNREACHABLE
        return served_checkpoint_root(base) if base else UNREACHABLE
    try:
        from scripts.autoresearch.coverage_leg import resolve_served_model
        return resolve_served_model(role) or UNREACHABLE
    except Exception:                                                  # noqa: BLE001
        return UNREACHABLE


def inventory(overrides: dict[str, Path] | None = None) -> dict[str, Path | None]:
    """Where each class lives, at the directories its own writer uses.

    Each default is the writer's own accessor rather than a re-spelled path: the
    ceiling artifacts are found through `label_agreement_ceiling.default_out_dir()`
    (which honours its `OUT_DIR_ENV`), and the coverage arms through the autoresearch
    config's `research_root`, so a moved data root moves this read with it instead of
    leaving it silently pointed at nothing.
    """
    from eval.label_agreement_ceiling import default_out_dir
    overrides = overrides or {}
    coverage_dir = overrides.get("coverage-arm")
    if coverage_dir is None:
        try:
            from scripts.autoresearch.common import load_config
            coverage_dir = load_config().paths.research_root / "coverage"
        except Exception:                                              # noqa: BLE001
            coverage_dir = None
    return {"label-agreement": overrides.get("label-agreement") or default_out_dir(),
            "judge-compare": overrides.get("judge-compare") or (ROOT / "eval" / "measurements"),
            "coverage-arm": coverage_dir}


def main(argv: list[str] | None = None,
         *, live_for: Callable[[str], str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n")[0] + " — report-only; prints and gates nothing.")
    ap.add_argument("--label-agreement-dir", default=None,
                    help="override the ceiling artifact directory (default: the "
                         "writer's own default_out_dir)")
    ap.add_argument("--judge-compare-dir", default=None,
                    help="override the judge-compare directory (default: eval/measurements)")
    ap.add_argument("--coverage-dir", default=None,
                    help="override the coverage-arm directory (default: the "
                         "autoresearch research_root/coverage)")
    args = ap.parse_args(argv)
    probe = live_for or live_served_identity

    overrides = {k: Path(v) for k, v in (
        ("label-agreement", args.label_agreement_dir),
        ("judge_compare", args.judge_compare_dir),
        ("coverage_arm", args.coverage_dir)) if v}
    overrides = {"judge-compare" if k == "judge_compare" else
                 "coverage-arm" if k == "coverage_arm" else k: v
                 for k, v in overrides.items()}

    dirs = inventory(overrides)
    rows: list[dict[str, Any]] = []
    for kind, pattern, reader in CLASSES:
        directory = dirs.get(kind)
        found = artifact_rows(kind, directory, pattern, reader) if directory else []
        print(f"[{kind}] {directory or 'no directory resolvable'}: "
              f"{len(found)} artifact(s)")
        rows.extend(found)
    rows.sort(key=lambda r: (-r["age_days"], r["path"]))

    print()
    judged_rows = judged(rows, probe)
    for line in render(judged_rows):
        print(line)
    print()
    print(summary_line(judged_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
