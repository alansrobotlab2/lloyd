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

Exits 2 when the improvement pass could not complete — including when the
knowledge-graph store could not be read (#1383): the run then prints
`[warn] knowledge-graph store unreadable: …` beside its counts and exits 2
even when the markdown-driven half (drift, contradiction pairs) succeeded,
because partial success must be reportable as partial, not as clean. Exits 3
when it completed but claims to have changed more facts than its own budget
allowed — a distinct code, and an overrun keeps it even when the store was
also unreadable. A consumer that reports success when it could not see the
graph is worse than none — same rule knowledge-health-report.py runs on (it
prints `STORE UNAVAILABLE: …` to stderr; this pass keeps its line beside the
counts on stdout, where a counts-reader cannot miss it).
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
    ap.add_argument("--corrections-window-days", type=int,
                    default=fi.CORRECTIONS_WINDOW_DAYS,
                    help=f"corrections window in days (default "
                         f"{fi.CORRECTIONS_WINDOW_DAYS}). An entry older than "
                         "this names no entity; the run then says the log is "
                         "stale since <date> instead of reporting no "
                         "corrections (#802)")
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
            corrections_days=args.corrections_window_days,
        )
    except Exception as exc:  # noqa: BLE001 - a consumer must not exit 0 on a failed pass
        print(f"[error] improvement pass failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(rec, indent=2, default=str))
    else:
        mode = "APPLY" if rec["apply"] else "DRY RUN"
        near_dupes = sum(e.get("near_duplicates") or 0 for e in rec["per_entity"])
        # `of N` is the point (#699): a bare `planned=0` was read as a claim
        # about the knowledge graph when it is a claim about the slice this run
        # scanned. None = drift was not consulted, so there is no pool to name;
        # print the count rather than a fraction over a total nobody measured.
        # A refused entity is not a scanned one (#702): `scanned=40` with seven
        # god-nodes refused was a coverage claim about facts nobody compared,
        # so the refusals come off the count and are named beside it.
        refused = [e["entity"] for e in rec["per_entity"] if e.get("refused")]
        total = rec.get("drift_candidates_total")
        n_scanned = len(rec["entities"]) - len(refused)
        scanned = (f"entities scanned={n_scanned} of {total} drift candidates"
                   if total is not None else
                   f"entities scanned={n_scanned}")
        if refused:
            scanned += f" refused={len(refused)} ({', '.join(refused)})"
        print(f"[{mode}] signals={rec['signals']} {scanned}"
              f" actions planned={rec['actions_planned']} taken={rec['actions_taken']}"
              f" near-duplicate pairs reported-not-deleted={near_dupes}")
        print(f"[facts] active {rec['before_active']} -> {rec['after_active']}"
              f" (delta {rec['delta_active']})")
        # The store verdict beside the counts (#1383): `-1 -> -1` is the
        # sentinel shape for "the store could not answer", and on its own it
        # read as a zero. It must sit on the SAME output as the counts — a
        # warning on stderr next to a clean stdout is the false green again.
        # A stale corrections log beside the counts, not a bare `signals=0`
        # (#802): `memory/corrections.md` went unwritten on 2026-05-08 and every
        # pass still printed a quiet zero, because a zero from a dead channel and
        # a zero from a quiet fortnight printed identically. Same rule as the
        # store verdict above — the caveat goes on stdout, where the
        # counts-reader is, never only in the JSON nobody opens.
        if rec.get("corrections_stale_since"):
            print(f"[corrections] 0 signals in window: log stale since "
                  f"{rec['corrections_stale_since']} (window "
                  f"{rec.get('corrections_window_days')} days)")
        if rec.get("store_ok") is False:
            print(f"[warn] knowledge-graph store unreadable: {rec.get('store_error')}")
        elif rec.get("store_ok") is True:
            # `is True`, not `else`: a record that carries no verdict at all
            # must not print the ok line either, or the printed half of this
            # fix is once again a default rather than a measurement.
            print(f"[store] ok ({rec.get('kg_db')})")
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
    if rec.get("store_ok") is False:
        # #1383: the markdown half may have planned and reported everything
        # correctly and the graph still was never seen. Checked after the
        # budget rung so an overrun keeps its distinct 3 — but the pass that
        # could not read the store can no longer exit 0.
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
