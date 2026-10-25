#!/usr/bin/env python3
"""Publish the two-week cost-outlier report #2510 owed and never wrote (#3280).

`cost_replay.py` answers "what WOULD have fired". This answers the three questions
#2510's owed entry asked over a fortnight of live fire and never got answered:
how much did the loop actually fire, how much of what it said was worth saying, and
what did it cost to say. One dated row, printed and appended to
`_pipeline/cost-outliers/report-<YYYY-MM-DD>.md` beside the per-fire records it
counts. Read-only over `workers.db` + `usage.db`; dispatches nothing.

    .venvs/lloyd/bin/python scripts/maintenance/cost_outlier_report.py
    .venvs/lloyd/bin/python scripts/maintenance/cost_outlier_report.py --days 30 --json

**Why the fire count is not a run count.** One arming is one `queue` row with
`source='cost-investigate'`; `runs` holds a row per ATTEMPT. The live store holds 5
arms and 7 run rows for 2026-10-13→10-20, and the difference is one fire that failed
three times before quarantining. Counting `runs` would have printed 7 fires and
reported retries as discoveries — which is why clause 2 of this item exists.

**Why the false-positive share is the row's point.** A candidate rule that labels
most of the fleet describes the fleet, not the run it fired on, and the record's
stated value ("where did THIS run's tokens go") is void. Measured over the trailing
14 days on this box, `prefix_cache` labelled 4,564 of the window's 4,564 sessions
(max per-session cache_pct 48.6 against a constant of 50.0), and 3 of the 4 records
the loop ever wrote name it — three restatements and one signal. The share is
computed by running the shipped `analyse()` over the window's sessions, so the
number that condemns a fire comes from the same code that wrote it.

**Exit codes, and what is NOT one of them.** No store (2) or a window holding no
measurable session at all (1). A store shorter than the window asked for is NOT an
error: it prints `held X of Y requested` and exits 0, because a young store is the
normal state of this box. `cost_replay.py`'s docstring promises an exit 3 for that
case; its code returns 0 at its own line 120 and says so in the comment above it —
this script follows the code, not the stale prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import cost_outlier as CO  # noqa: E402
from app import paths  # noqa: E402


def _pct(x) -> str:
    return "unmeasured" if x is None else f"{float(x):g}%"


def _int(x) -> str:
    return "—" if x is None else f"{int(x):,}"


def lost_to_2738(error) -> bool:
    """Does this quarantine error name the #2738 compaction `int()` cast?

    Matched on the error's own text rather than assumed from the date, because the
    report's claim is that the fire is unrecoverable *and* will not recur — both of
    which are true of that bug and not of a quarantine in general.
    """
    text = str(error or "")
    return "invalid literal for int()" in text and "turn_start" in text


def build_row(rep: dict) -> tuple[list[str], list[str], str]:
    """(the dated row, the lines under it, the date) — one report's markdown.

    Split into the row and its detail because the row is the artefact the owed entry
    asked for and the detail is what makes the row checkable: a caller that wants
    one line (a journal, a future scheduled job) takes `lines`, and the file gets
    both.
    """
    win, fired, cost = rep["window"], rep["fires"], rep["cost"]
    rates, scored = rep["base_rates"], rep["scored"]
    day = CO._now().strftime("%Y-%m-%d")
    prefix = rep["cache_pct"]

    share = ("—" if scored["share"] is None
             else f"{scored['share']}% ({scored['false_positives']} of "
                  f"{scored['scored']} scored fires)")
    lines = [
        f"| {day} | held {win['window_days_held']}d of "
        f"{win['window_days_requested']:g}d asked | {fired['arms']} arms / "
        f"{fired['records']} records / {fired['quarantined']} quarantined "
        f"({fired['attempts']} run rows = attempts, not fires) | "
        f"false-positive share {share} | own cost {_int(cost['tokens'])} tokens, "
        f"{cost['model_turns']} model turns, {cost['gpu_minutes']} gpu-min "
        f"({float(fired['run_seconds_total']):.3f} s) |",
    ]
    detail = [
        "",
        f"Window: {win['window_start']} → now, over {win['runs_in_window']} "
        f"successful runs and {len(win['session_ids'])} sessions "
        f"(first_completed={win['first_completed_at']}).",
        f"Fire count reads `queue` rows with `source='{CO.SOURCE}'` as arms and "
        f"`runs` as attempts/records; a quarantined arm's retries are not extra "
        f"fires.",
    ]
    if fired["orphan_run_rows"]:
        detail.append(
            f"{fired['orphan_run_rows']} `cost-investigate` run row(s) carry no "
            f"arm in `queue` (their seconds are still in the cost above).")

    detail += ["", "Base rate per candidate — the share of the window's "
                   f"{rates['sessions_with_rows']} measured sessions the shipped "
                   "`analyse()` labels:",
               ""]
    for rule, row in rates["rules"].items():
        if rule == "prefix_cache":
            detail.append(
                f"  - `prefix_cache`: {_pct(prefix['labelled_at_legacy_constant'].get('share'))} "
                f"of sessions at the retired constant (< {CO.LEGACY_PREFIX_CUT_OFF:g}%), "
                f"{_pct(prefix['labelled_at_window_p25'].get('share'))} at this "
                f"window's own P25 ({_pct(prefix['p25'])}) — the rule arms below the "
                f"P25 now (#3280), so it labels the worst quartile, not the fleet.")
        else:
            detail.append(f"  - `{rule}`: {_pct(row['share'])} "
                          f"({row['labelled']}/{rates['sessions_with_rows']})")

    detail += ["", "Each fire, scored at the cut-off it was armed with:", ""]
    for row in scored["per_fire"]:
        if row["candidate"] is None:
            detail.append(f"  - `{row['subject_run_id']}` ({row['subject_source']}): "
                          f"no candidate named — see the lost record below")
            continue
        at = (f" at cut-off {_pct(row['cut_off'])}" if row["cut_off"] is not None
              else "")
        detail.append(
            f"  - `{row['subject_run_id']}` ({row['subject_source']}): candidate "
            f"`{row['candidate']}`, base rate {_pct(row['base_rate'])}{at} → "
            f"{'FALSE POSITIVE' if row['false_positive'] else 'informative'} "
            f"(bar: >= {CO.FALSE_POSITIVE_BASE_RATE:g}%)")

    for f in fired["fires"]:
        if not f["quarantined"]:
            continue
        cause = ("lost to #2738 — `int()` cast on the `compaction` TEXT column, "
                 "fixed by 9356625c" if lost_to_2738(f["error"])
                 else f"lost after {f['attempt_rows']} attempt(s): {f['error']}")
        detail += [
            "",
            f"**Record lost, not re-fired:** `{f['subject_run_id']}` "
            f"({_int(f['tokens'])} tokens vs p90 {_int(f['p90_tokens'])}) is "
            f"{cause}. Its {f['attempt_rows']} run rows wrote no `artifact_path`, "
            "so the diagnostic ran and nothing survived it. This report names it "
            "rather than re-firing the job: the cause is gone from the tree, so a "
            "re-fire would either produce the record #2738 destroyed (not the same "
            "evidence, since the window has moved) or nothing at all.",
        ]

    detail += [
        "",
        f"The loop's own cost: {_int(cost['tokens'])} tokens and "
        f"{cost['model_turns']} model turns across "
        f"{cost['attributed_sessions']} attributed session(s) — every "
        f"`{CO.SOURCE}` run on this box carries `session_ids: []`, because the "
        f"diagnostic spends no model turn by design. gpu-minutes is "
        f"`SUM(duration_seconds)/60`, the seconds rule of `run_rollup_by_source()` "
        f"in `workers/queue.py`; if a future version spends a turn, its run's "
        f"`meta_json` carries the session and these two figures stop being zero "
        f"with no change to this line.",
        "*Written by `scripts/maintenance/cost_outlier_report.py` (#3280) from "
        "read-only copies of `workers.db` and `usage.db`. Base rates come from the "
        "shipped `analyse()`; no model turn was spent to produce this row.*",
    ]
    return lines, detail, day


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Publish the dated cost-outlier loop report (#3280).")
    ap.add_argument("--workers-db", default=None,
                    help="defaults to $LLOYD_DATA/workers.db")
    ap.add_argument("--usage-db", default=None,
                    help="defaults to $LLOYD_DATA/usage.db")
    ap.add_argument("--days", type=float, default=CO.REPORT_WINDOW_DAYS,
                    help=f"window to request (default {CO.REPORT_WINDOW_DAYS})")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument("--out-dir", default=None,
                    help="where report-<date>.md goes (default: "
                         "$LLOYD_DATA/_pipeline/cost-outliers). For tests.")
    ap.add_argument("--no-write", action="store_true",
                    help="print the row without publishing it")
    args = ap.parse_args(argv)

    wdb = Path(args.workers_db) if args.workers_db else paths.WORKERS_DB
    udb = Path(args.usage_db) if args.usage_db else paths.USAGE_DB
    for p, what in ((wdb, "workers"), (udb, "usage")):
        if not p.exists():
            print(f"cost_outlier_report: no {what} store at {p}", file=sys.stderr)
            return 2

    rep = CO.report(wdb, udb, window_days=args.days)

    if args.json:
        print(json.dumps({**rep,
                          "row": build_row(rep)[0][0],
                          "out_dir": str(args.out_dir or
                                         (paths.PIPELINE_DIR / "cost-outliers"))},
                         indent=2, default=str))
    else:
        lines, detail, day = build_row(rep)
        print(f"cost-outlier loop report — {day} (#3280)")
        print(f"  workers db : {wdb}")
        print(f"  usage db   : {udb}")
        print()
        print("| date | window | fire count | false-positive share | own cost |")
        print("|---|---|---|---|---|")
        for line in lines:
            print(line)
        for line in detail:
            print(line)

    # On stderr, so `--json` stays machine-readable while the note stays visible in
    # the mode a human runs: the held-of-asked figures are already in the JSON, and a
    # line printed after the object makes stdout unparseable for the next job.
    if rep["window"]["window_days_held"] < rep["window"]["window_days_requested"]:
        print("  note: the store is shorter than the window asked for, so every "
              "count above rests on the held window, not the requested one",
              file=sys.stderr)

    if not args.no_write:
        lines, detail, day = build_row(rep)
        dest = CO.report_path(day, out_dir=(Path(args.out_dir)
                                           if args.out_dir else None))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join([
            f"# Cost-outlier loop report — {day}", "",
            "The row #2510 owed: what the trigger fired, how much of it was worth "
            "saying, and what it cost.", "",
            "| date | window | fire count | false-positive share | own cost |",
            "|---|---|---|---|---|"] + lines + detail) + "\n", encoding="utf-8")
        print(f"  wrote       : {dest}")

    if rep["base_rates"]["sessions_with_rows"] == 0:
        print("  the window holds no session with a usage row: the false-positive "
              "share could not be measured, so this report measured nothing",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
