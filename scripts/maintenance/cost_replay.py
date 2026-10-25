#!/usr/bin/env python3
"""Replay the cost-outlier trigger over the runs this box actually holds (#2510).

Read-only over `workers.db` + `usage.db`. Prints, per worker source: how many
successful runs had an attributable token cost in the window, that population's
P90 in tokens, and how many runs would have armed a diagnostic at the configured
margin. Nothing is dispatched and no watermark moves — this is the kill switch the
item ships with. Run it before trusting the trigger, and again after changing the
margin.

    .venvs/lloyd/bin/python scripts/maintenance/cost_replay.py
    .venvs/lloyd/bin/python scripts/maintenance/cost_replay.py --margin 2.0

**The window line is the point of the output, not decoration.** The item asked for
a 30-day replay; this box's `workers.db` has never held 30 days (earliest
`completed_at` was 2026-09-22 when #2510 was triaged). A table that printed a
30-day label over an 18-day store would report a coverage the data does not have,
which is the same class of error `app/routers/workers.py:174` already documents
for the dashboard's per-source stats. So the header says what was held and what
was asked, in that order, and the shortfall is loud in the table rather than in the
exit code: a short window is a normal state of a young store, so this command exits 0
on it, and a non-zero code here means the replay could not measure at all (1 = no
attributable run, 2 = no store; see the comment above that `return`). #3280's clause 2
asked a sibling command to fail loudly on a short window, copying a promise this
sentence used to make and this code never kept. A false exit code is not cosmetic: the
next reader writes a gate around the number a docstring promised.

Tokens, not dollars: `cost_usd` is 0.0 on every usage row, so a dollar P90 here
would be a column of zeros dressed up as a cost. See `app/cost_outlier.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import cost_outlier as CO  # noqa: E402
from app import paths  # noqa: E402


def _int(x) -> str:
    return "—" if x is None else f"{int(x):,}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay the token cost-outlier trigger without dispatching anything.")
    ap.add_argument("--workers-db", default=None,
                    help="defaults to $LLOYD_DATA/workers.db")
    ap.add_argument("--usage-db", default=None,
                    help="defaults to $LLOYD_DATA/usage.db")
    ap.add_argument("--days", type=float, default=CO.DEFAULT_WINDOW_DAYS,
                    help=f"window to request (default {CO.DEFAULT_WINDOW_DAYS})")
    ap.add_argument("--margin", type=float, default=CO.DEFAULT_MARGIN,
                    help=f"multiple of the source's own P90 (default {CO.DEFAULT_MARGIN})")
    ap.add_argument("--min-samples", type=int, default=CO.DEFAULT_MIN_SAMPLES,
                    help=f"floor below which a source cannot arm (default "
                         f"{CO.DEFAULT_MIN_SAMPLES})")
    ap.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    args = ap.parse_args(argv)

    wdb = Path(args.workers_db) if args.workers_db else paths.WORKERS_DB
    udb = Path(args.usage_db) if args.usage_db else paths.USAGE_DB
    for p, what in ((wdb, "workers"), (udb, "usage")):
        if not p.exists():
            print(f"cost_replay: no {what} store at {p}", file=sys.stderr)
            return 2

    rep = CO.replay(wdb, udb, window_days=args.days, min_samples=args.min_samples,
                    margin=args.margin)

    if args.json:
        import json
        print(json.dumps(rep, indent=2))
        return 0 if rep["sources"] else 1

    print("cost-outlier replay — tokens, never dollars "
          "(cost_usd is 0.0 on every usage row)")
    print(f"  workers db : {wdb}")
    print(f"  usage db   : {udb}")
    print(f"  window     : held {rep['window_days_held']:.1f}d of "
          f"{rep['window_days_requested']:.1f}d requested "
          f"(first_completed={rep['first_completed_at']})")
    print(f"  rule       : arm above {rep['margin']}x the source's own window P90, "
          f"needs >={rep['min_samples']} samples")
    print(f"  excluded   : {', '.join(rep['excluded_sources'])} "
          f"(the investigator is never in the population that arms it)")
    print(f"  runs       : {rep['runs_in_window']} successful in window, "
          f"{rep['unattributed']} with no attributable cost "
          f"(held OUT of the distribution — not scored as zero)")
    print()
    if not rep["sources"]:
        print("  no source has a single attributable run in the window held: this "
              "replay measured nothing, so it neither supports nor kills the trigger")
        return 1

    print(f"  {'source':<22} {'samples':>8} {'p90 tokens':>12} {'fires':>7} "
          f"{'armed':>7}  {'median':>10} {'max':>10}")
    total_fires = 0
    for source, row in rep["sources"].items():
        total_fires += row["fires"]
        print(f"  {source:<22} {row['samples']:>8} {_int(row['p90_tokens']):>12} "
              f"{row['fires']:>7} {('yes' if row['armed'] else 'no'):>7}  "
              f"{_int(row['median_tokens']):>10} {_int(row['max_tokens']):>10}")
        for rid in row["fire_run_ids"]:
            print(f"      would-fire {source} {rid}")

    print()
    print(f"  would-be fires at {args.margin}x: {total_fires} over "
          f"{rep['window_days_held']:.1f} days")
    if total_fires == 0:
        print("  zero fires fleet-wide: on this window the trigger is dead weight "
              "— the item's own kill switch says stop here rather than ship it")
    # Exit 0 even when the store is shorter than the window asked for. A young
    # store is the normal state of this box, and a caller that reads `$?` as
    # success-or-failure would treat "we have 18 days, not 30" as a broken run.
    # Non-zero here means the replay could not measure: no store (2) or no
    # attributable run at all (1). Shortfall is printed, loudly, and stays a 0.
    if rep["window_days_held"] < rep["window_days_requested"]:
        print("  note: the store is shorter than the window asked for, so every "
              "count above rests on the held window, not the requested one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
