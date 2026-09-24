#!/usr/bin/env python3
"""
eval/run_skill_activation_eval.py — does each covered skill fire when it should,
and only then (#711).

For every skill in `eval/skill_activation_cases.yaml` it runs the corpus
derived from `eval/skill_match_queries.yaml` through the production matcher
(`scripts/skill_activation.py`: what prefetch would inject, ranked by
`prefetch._search_skills`) and prints per skill and for the corpus: triggers,
misses, false triggers, recall and false-trigger rate. A rate over an empty
population is never printed as a number; its denominator is.

Exits 1 when a skill has positives but no negatives (its false-trigger rate
would read clean by construction) or fewer than five of either, and 2 when a
covered or labelled skill is not an active skill any more.

Deterministic: no model, no engine. `--write-baseline` records one row per
(skill, case) — id, label, winning skill, triggered, pass — plus the counts,
under `app.paths.EVAL_BASELINES_DIR/skill_activation_baseline.json`; two runs
over the same corpus and skills produce identical counts.

    .venvs/lloyd/bin/python eval/run_skill_activation_eval.py
    .venvs/lloyd/bin/python eval/run_skill_activation_eval.py --write-baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import skill_activation as A  # noqa: E402

BASELINE_NAME = "skill_activation_baseline.json"
MIN_CASES = 5


def baseline_path() -> Path:
    from app.paths import EVAL_BASELINES_DIR
    return EVAL_BASELINES_DIR / BASELINE_NAME


def fmt_rate(n: int, d: int, rate) -> str:
    return f"{rate:.4f}" if d else f"n/a (denominator {d})"


def render(result: dict) -> list[str]:
    lines = [f"{'skill':32} {'pos':>4} {'trig':>4} {'miss':>4} {'neg':>4} {'false':>5}  "
             f"recall / false-trigger rate"]
    rows = list(result["skills"].values()) + [dict(result["corpus"], skill="CORPUS")]
    for p in rows:
        lines.append(
            f"{p['skill']:32} {p['positives']:>4} {p['triggers']:>4} {p['misses']:>4} "
            f"{p['negatives']:>4} {p['false_triggers']:>5}  "
            f"{fmt_rate(p['triggers'], p['positives'], p['recall'])} / "
            f"{fmt_rate(p['false_triggers'], p['negatives'], p['false_trigger_rate'])}")
    return lines


def build_report(result: dict) -> dict:
    return {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # tests/test_eval_scorer.py asks every baseline whether it measured
        # production and for the vault-leg knobs; this one scores the skill
        # matcher with its own untouched thresholds, so those do not apply.
        "matches_production_defaults": True,
        "graph_rerank": "not_applicable: scores skill activation, not the vault leg",
        "rerank_alpha": "not_applicable: scores skill activation, not the vault leg",
        "graph_top_k": "not_applicable: scores skill activation, not the vault leg",
        "graph_hops": "not_applicable: scores skill activation, not the vault leg",
        "set": "eval/skill_match_queries.yaml + eval/skill_activation_cases.yaml",
        "corpus": result["corpus"],
        "skills": {s: {k: v for k, v in p.items() if k != "rows"}
                   for s, p in result["skills"].items()},
        "rows": [r for p in result["skills"].values() for r in p["rows"]],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--write-baseline", action="store_true",
                    help=f"record the run as EVAL_BASELINES_DIR/{BASELINE_NAME}")
    ap.add_argument("--json", default="", help="also write the report here")
    args = ap.parse_args(argv)

    spec = A.load_spec()
    records = A.load_records(spec)
    active = {s["name"] for s in A.base_skills()}
    stale = sorted({s for s in spec["skills"] if s not in active}
                   | {s for r in records for s in (r.get("expected_skills") or [])
                      if s not in active})
    if stale:
        print(f"STALE: not active skills any more: {stale}", file=sys.stderr)
        return 2
    cases = A.build_cases(records, spec["skills"])
    errs = A.contract_errors(cases, MIN_CASES)
    result = A.evaluate(cases)
    print("\n".join(render(result)))
    report = build_report(result)
    if args.write_baseline:
        out = baseline_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    if errs:
        for e in errs:
            print("CORPUS: " + e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
