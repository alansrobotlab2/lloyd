#!/usr/bin/env python3
"""Fact-quality improvement pass — the #376 feedback loop, runnable on its own.

Reads a real feedback signal (the user's corrections log and recent-write
drift), pairs contradictions, and — with --apply — expires the claims an
independent reason condemns. Dry-run by default: without --apply nothing is
written and the run says what it *would* have done.

    scripts/memory/fact-improvement.py                     # plan only
    scripts/memory/fact-improvement.py --apply             # act, capped
    scripts/memory/fact-improvement.py --apply --max-actions 10
    scripts/memory/fact-improvement.py --report-eval       # score the metric too

Exits 2 when the improvement pass could not complete, and 3 when it completed
but claims to have changed more facts than its own budget allowed. A consumer
that reports success when it could not see the graph is worse than none — same
rule knowledge-health-report.py runs on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import fact_improvement as fi  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually expire/invalidate facts (default: plan only)")
    ap.add_argument("--sources", default="corrections,drift",
                    help="comma-separated signals to read: corrections, drift")
    ap.add_argument("--entity", action="append", default=[],
                    help="improve only this entity (repeatable); bypasses signal sources")
    ap.add_argument("--days", type=int, default=fi.DRIFT_WINDOW_DAYS,
                    help=f"drift window in days (default {fi.DRIFT_WINDOW_DAYS})")
    ap.add_argument("--limit", type=int, default=40,
                    help="max entities to examine (default 40)")
    ap.add_argument("--max-actions", type=int, default=fi.MAX_ACTIONS_PER_RUN,
                    help=f"max facts this run may change (default {fi.MAX_ACTIONS_PER_RUN})")
    ap.add_argument("--report-eval", action="store_true",
                    help="also score fact_entity_recall before and after (slow: ~1s/query)")
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    args = ap.parse_args()

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    unknown = set(sources) - {"corrections", "drift"}
    if unknown:
        print(f"[error] unknown source(s): {', '.join(sorted(unknown))}"
              " (known: corrections, drift)", file=sys.stderr)
        return 2

    before_eval = fi._fact_entity_recall() if args.report_eval else None

    try:
        rec = fi.run_improvement(
            apply=args.apply, sources=sources, entities=args.entity or None,
            days=args.days, limit=args.limit, max_actions=args.max_actions,
            report_eval=args.report_eval,
        )
    except Exception as exc:  # noqa: BLE001 - a consumer must not exit 0 on a failed pass
        print(f"[error] improvement pass failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(rec, indent=2, default=str))
    else:
        mode = "APPLY" if rec["apply"] else "DRY RUN"
        near_dupes = sum(e.get("near_duplicates", 0) for e in rec["per_entity"])
        print(f"[{mode}] signals={rec['signals']} entities scanned={len(rec['entities'])}"
              f" actions planned={rec['actions_planned']} taken={rec['actions_taken']}"
              f" near-duplicate pairs reported-not-deleted={near_dupes}")
        print(f"[facts] active {rec['before_active']} -> {rec['after_active']}"
              f" (delta {rec['delta_active']})")
        if before_eval is not None or rec.get("fact_entity_recall") is not None:
            print(f"[metric] fact_entity_recall {before_eval} -> {rec.get('fact_entity_recall')}")
        if not rec["apply"]:
            print("[note] dry run: nothing was written. --apply to act on the list.")
        for entry in rec["per_entity"]:
            if entry.get("error"):
                print(f"  ! {entry['entity']}: {entry['error']}")
                continue
            for action in entry.get("actions", []):
                flag = ("expired" if action.get("applied") else
                        "would expire" if action.get("planned") else
                        "skipped" if action.get("skipped") else "no-op")
                print(f"  [{flag}] {entry['entity']}: {action['loser_fact'][:70]}")
                print(f"           {action['reason'][:120]}")
            if entry.get("stopped"):
                print(f"  = {entry['entity']}: stopped — {entry['stopped']}")
        print(f"[record] {rec.get('record_path') or rec.get('record_error')}")

    if rec["actions_taken"] > args.max_actions:
        print(f"[error] changed {rec['actions_taken']} facts, over the {args.max_actions}"
              " allowed by this run — inspect the record before trusting it",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
