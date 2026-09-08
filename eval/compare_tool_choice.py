#!/usr/bin/env python3
"""Compare two tool-choice eval runs and say whether anything regressed.

Why this exists
---------------
`run_tool_choice_eval.py` writes a JSON baseline per run and prints nothing a
reader can compare. On 2026-09-08 the round that rewrote 66% of the operating
contract ran it, then spent four iterations guessing the file's shape —
`summary` came back `{}` on the first three probes because the script looked
for the wrong key — and landed without ever comparing against the 0.90
`correct_rate` recorded on 09-04. A number with no prior is not a regression
guard; it is a number.

So this does the comparison, names the prior it used, and exits non-zero when
something moved the wrong way. No model, no vault, no network: it reads two
JSON files.

Usage:
    python eval/compare_tool_choice.py                     # newest vs the one before
    python eval/compare_tool_choice.py --label soul377-post
    python eval/compare_tool_choice.py --current a.json --baseline b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE / "baselines" / "tool-choice"

# Metric -> whether higher is better. `shelled_to_web_rate` and
# `no_tool_call_rate` are failures being counted, so lower wins; the rest are
# successes. Getting this backwards would report a regression as an
# improvement, which is worse than not comparing at all.
METRICS: dict[str, bool] = {
    "correct_rate": True,
    "http_tool_first_rate": True,
    "control_correct_rate": True,
    "shelled_to_web_rate": False,
    "no_tool_call_rate": False,
}

# Below this, a move is sampling noise on a 20-query set: one query is 0.05.
# A single query flipping must not read as a regression, and two must.
TOLERANCE = 0.051


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def overall(run: dict[str, Any]) -> dict[str, Any]:
    """The summary block, tolerating the shape drift between versions."""
    summary = run.get("summary") or {}
    if isinstance(summary, dict) and summary.get("overall"):
        return summary["overall"]
    return summary if isinstance(summary, dict) else {}


def runs_by_recency() -> list[Path]:
    if not BASELINE_DIR.is_dir():
        return []
    return sorted(BASELINE_DIR.glob("*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(regressions, lines). `lines` is the full table, printed either way."""
    cur, base = overall(current), overall(baseline)
    regressions: list[str] = []
    lines: list[str] = []
    for metric, higher_is_better in METRICS.items():
        if metric not in cur and metric not in base:
            continue
        c = cur.get(metric)
        b = base.get(metric)
        if not isinstance(c, (int, float)) or not isinstance(b, (int, float)):
            lines.append(f"  {metric:<24} {b!r} -> {c!r}  (not comparable)")
            continue
        delta = c - b
        moved_wrong = (delta < -TOLERANCE) if higher_is_better else (delta > TOLERANCE)
        mark = "REGRESSED" if moved_wrong else "ok"
        lines.append(f"  {metric:<24} {b:.3f} -> {c:.3f}  ({delta:+.3f})  {mark}")
        if moved_wrong:
            regressions.append(f"{metric} {b:.3f} -> {c:.3f} ({delta:+.3f})")

    # An eval that errored measured nothing, whatever its rates say.
    errors = cur.get("errors")
    if isinstance(errors, int) and errors:
        regressions.append(f"{errors} quer(y/ies) errored — the run did not measure cleanly")
    return regressions, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", help="compare the newest run with this label "
                                    "against the newest run without it")
    ap.add_argument("--current", help="explicit path to the run under test")
    ap.add_argument("--baseline", help="explicit path to the prior run")
    args = ap.parse_args(argv)

    if args.current and args.baseline:
        cur_path, base_path = Path(args.current), Path(args.baseline)
    else:
        runs = runs_by_recency()
        if len(runs) < 2:
            print(f"[skip] need two runs under {BASELINE_DIR}, found {len(runs)}. "
                  "Nothing to compare — this is not a pass.")
            return 2
        if args.label:
            match = [p for p in runs if p.name.startswith(args.label)]
            if not match:
                print(f"[error] no run under {BASELINE_DIR} labelled {args.label!r}")
                return 2
            cur_path = match[0]
            prior = [p for p in runs if not p.name.startswith(args.label)]
            if not prior:
                print(f"[skip] every run is labelled {args.label!r}; no prior to compare against.")
                return 2
            base_path = prior[0]
        else:
            cur_path, base_path = runs[0], runs[1]

    current, baseline = load(cur_path), load(base_path)
    regressions, lines = compare(current, baseline)

    print(f"current : {cur_path.name}  ({current.get('n_queries', '?')} queries)")
    print(f"baseline: {base_path.name}  ({baseline.get('n_queries', '?')} queries)")
    if current.get("n_queries") != baseline.get("n_queries"):
        print("  note: query counts differ — the two runs are not the same test")
    print("\n".join(lines))
    if regressions:
        print("\nREGRESSED: " + "; ".join(regressions))
        return 1
    print(f"\nno regression beyond +/-{TOLERANCE:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
