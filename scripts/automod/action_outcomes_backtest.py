"""#4048 backtest: what the action-class arm would have withheld, read-only.

Replays the automod ledger — the live `promotions.jsonl` AND every
`promotions-archive-*.jsonl.gz` beside it (a reader that misses the archives
reads ~5 weeks and reports "no bad outcomes": all 11 `rollback_succeeded` rows
are in the September archive) — through `app/harness/action_outcomes.py`'s own
grading and resolver, and for every unattended terminal action inside the window
asks: had the arm been armed, would this action have been withheld, and was it
later shown bad?

  automod_land  action = a `promoted` row; later bad = a bad graded row
                (rollback / confirmed regression) naming that commit
  vault_land    action = a `vault_land` row that was not a no-op; later bad =
                its own graded outcome is bad (the validator refused and
                reverted the bytes the worker had already written)

The standing at each action is resolved from the rows strictly before it, using
the whole history (as a populated store would), and the window bounds which
actions are counted.

Writes nothing: no database, no state file, no ledger row. Run:

  ~/lloyd/.venvs/lloyd/bin/python -m scripts.automod.action_outcomes_backtest
  ~/lloyd/.venvs/lloyd/bin/python -m scripts.automod.action_outcomes_backtest --json
  ... --review-refusal-grades    # ruling (b) = yes, for comparison
  ... --count-promote-failed     # ruling (a) = yes, for comparison
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from app.harness import action_outcomes as AO


def ledger_paths(state_dir: Path | None = None) -> list[Path]:
    from scripts.automod import state as S
    live = (Path(state_dir) / "promotions.jsonl") if state_dir else Path(S.LEDGER_PATH)
    return [*S.ledger_archives(live), live]


def _actions(lines: list[bytes]) -> list[dict]:
    """The terminal actions the arm would answer, in ledger order."""
    out: list[dict] = []
    seen: set[str] = set()
    for line in lines:
        sid = AO.line_id(line)
        if sid in seen:
            continue
        seen.add(sid)
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        name = e.get("event")
        ts = AO._event_ts(e)
        if not ts:
            continue
        if name == "promoted":
            out.append({"class": "automod_land", "ts": ts, "sid": sid,
                        "subject": str(e.get("commit") or "")[:12],
                        "ref": f"promoted {e.get('round_id') or '-'} "
                               f"{str(e.get('commit') or '')[:12]}"})
        elif name == "vault_land":
            g = AO.grade_event(e, sid)
            if g is None or g.outcome == AO.NEUTRAL:
                continue   # a no-op landing wrote nothing; there was no action
            out.append({"class": "vault_land", "ts": ts, "sid": sid,
                        "subject": g.subject, "own_outcome": g.outcome,
                        "ref": g.evidence_ref})
    return out


def backtest(paths: list[Path], *, now: dt.datetime, days: int = 90) -> dict[str, Any]:
    lines = list(AO.iter_ledger_lines(paths))
    graded = AO.graded_from_lines(lines)
    actions = _actions(lines)
    start = AO._iso(now - dt.timedelta(days=days))
    end = AO._iso(now)

    per_class_rows: dict[str, list[dict]] = {}
    for g in sorted(graded, key=lambda g: g.ts):
        per_class_rows.setdefault(g.action_class, []).append(
            {"ts": g.ts, "outcome": g.outcome, "evidence_ref": g.evidence_ref,
             "subject": g.subject})
    keys = {c: [r["ts"] for r in rows] for c, rows in per_class_rows.items()}

    report: dict[str, Any] = {
        "window": {"start": start, "end": end, "days": days},
        "ledger_files": [str(p) for p in paths],
        "ruling_a_promote_failed_counts": "promote_failed" in AO.COUNTED_ROLLBACK_TRIGGERS,
        "ruling_b_review_refusal_grades": AO.REVIEW_REFUSAL_GRADES,
        "K": AO.CLEARANCE_STREAK_K, "window_days": AO.WINDOW_DAYS,
        "excluded_classes": dict(AO.EXCLUDED_CLASSES),
        "classes": {},
    }
    for cls in AO.ACTION_CLASSES:
        rows = per_class_rows.get(cls, [])
        in_window = [r for r in rows if start <= r["ts"] < end]
        bad_subjects: dict[str, list[str]] = {}
        for r in rows:
            if r["outcome"] == AO.BAD and r["subject"]:
                bad_subjects.setdefault(r["subject"], []).append(r["ts"])
        stats = {"actions": 0, "withheld": 0, "withheld_later_bad": 0,
                 "allowed_later_bad": 0, "later_bad": 0,
                 "withheld_by_reason": {"no_track_record": 0, "bad_in_window": 0},
                 "graded_in_window": {o: sum(1 for r in in_window if r["outcome"] == o)
                                      for o in AO.OUTCOMES},
                 "examples_withheld_later_bad": []}
        for a in actions:
            if a["class"] != cls or not (start <= a["ts"] < end):
                continue
            stats["actions"] += 1
            prefix = rows[:bisect.bisect_left(keys.get(cls, []), a["ts"])]
            standing = AO.resolve_rows(cls, prefix, now=AO._parse(a["ts"]))
            if cls == "vault_land":
                later_bad = a.get("own_outcome") == AO.BAD
            else:
                later_bad = any(ts >= a["ts"] for ts in bad_subjects.get(a["subject"], []))
            stats["later_bad"] += later_bad
            if standing.status == AO.PROBATION:
                stats["withheld"] += 1
                reason = ("no_track_record" if "fewer than K" in standing.why
                          else "bad_in_window")
                stats["withheld_by_reason"][reason] += 1
                if later_bad:
                    stats["withheld_later_bad"] += 1
                    if len(stats["examples_withheld_later_bad"]) < 5:
                        stats["examples_withheld_later_bad"].append(a["ref"])
            elif later_bad:
                stats["allowed_later_bad"] += 1
        report["classes"][cls] = stats
    return report


def _print(report: dict[str, Any]) -> None:
    w = report["window"]
    print(f"#4048 action-class backtest — {w['days']} d, {w['start']} .. {w['end']}")
    print(f"K={report['K']}  window={report['window_days']} d  "
          f"ruling(a) promote_failed counts={report['ruling_a_promote_failed_counts']}  "
          f"ruling(b) review refusal grades={report['ruling_b_review_refusal_grades']}")
    print("ledger:", ", ".join(Path(p).name for p in report["ledger_files"]))
    for cls, s in report["classes"].items():
        g = s["graded_in_window"]
        print(f"\n{cls}: {s['actions']} actions; graded rows in window clean={g['clean']} "
              f"bad={g['bad']} neutral={g['neutral']}")
        print(f"  would withhold {s['withheld']} "
              f"(no track record {s['withheld_by_reason']['no_track_record']}, "
              f"bad in window {s['withheld_by_reason']['bad_in_window']}); "
              f"of those later shown bad: {s['withheld_later_bad']}")
        print(f"  later shown bad overall {s['later_bad']}; allowed and later bad "
              f"{s['allowed_later_bad']}")
        for ex in s["examples_withheld_later_bad"]:
            print(f"    e.g. {ex}")
        if not s["withheld"]:
            print("  → nothing withheld: this class stays disarmed")
    print("\nexcluded:", ", ".join(report["excluded_classes"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--state-dir", type=Path, default=None,
                    help="automod state dir (default: the live one, read-only)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--review-refusal-grades", action="store_true",
                    help="grade review refusals bad (ruling (b) = yes)")
    ap.add_argument("--count-promote-failed", action="store_true",
                    help="grade promote_failed rollbacks bad (ruling (a) = yes)")
    args = ap.parse_args(argv)
    if args.review_refusal_grades:
        AO.REVIEW_REFUSAL_GRADES = True
    if args.count_promote_failed:
        AO.COUNTED_ROLLBACK_TRIGGERS = AO.COUNTED_ROLLBACK_TRIGGERS | {"promote_failed"}
    report = backtest(ledger_paths(args.state_dir), now=dt.datetime.now(dt.timezone.utc),
                      days=args.days)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
