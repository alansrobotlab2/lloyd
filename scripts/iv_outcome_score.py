#!/usr/bin/env python3
"""Inner Voice — score interventions against turn outcomes (backlog #833).

`iv_grade.py` reports two proxies and says so: whether the loop kept going
after an inject, and whether the user's next message read like a correction.
Neither has an outcome in it. This script joins `inner_voice_observations`
(usage.db) to the event logs on (session_id, turn_id) — `related_tool` is a tool
NAME, never a call id, so the turn is the only key both stores share — and
labels each observed turn from the EVENT LOG alone:

  stop_reason  (default) — the turn's `brain1.result_message.data.stop_reason`;
               `stop`/`end_turn` is good, anything else (`max_turns`,
               `cancelled`, …) is bad. A turn with no result_message is
               unlabelled (in flight, or killed with the backend) and not scored.
  tool_error   — bad when any `brain1.tool_result_received.data.result` in the
               turn starts with `Error` or `Traceback`. Prose, at a base rate
               measured under 1 % on 2026-09-18: precision against it is
               dominated by class imbalance, not by the observer.

No column of `inner_voice_observations` other than the keys, the action and
the rule that decided (`safeguard`, #770) is read for scoring: an observer that
wrote "the task succeeded" into `reason` is scored against what the log says.
The worker-run ledger (`runs.status`) is not joined — its `meta_json` names a
session for about a third of runs — so a worker turn is labelled like a chat.

Per class, over the window:
  n      scored intervention rows (rows whose turn has a label)
  TP/FP  rows whose turn ended bad / good
  precision  TP / n
  recall     bad turns the class fired in / all bad observed turns
  FN pool    bad observed turns in which NO intervention of any class fired
Deterministic guard injects are their own class (`inject[guard]`), split by
`iv_grade._is_deterministic`, which reads `safeguard` and falls back to the
prose only for rows written before the column existed. A class under
MIN_RATED scored rows prints counts only: cancel and clarify have a handful of
rows all-time, and a rate over three rows is noise with a decimal point.

Counterfactual limit: a `cancel` usually CAUSES `stop_reason: cancelled`, so
its TP count is partly self-inflicted, and there is no observed world where the
observer stayed quiet. Its FP rate must come from matched near-miss turns, not
from this ratio. Whether a class predicts a bad outcome or only flags elevated
risk — and the keep / record-only / change-the-prompt decision that follows —
is a human call; the header states only the lift over the base rate.

Read-only, like iv_grade: usage.db is opened with `mode=ro`, blobs are read,
nothing is written.

    python scripts/iv_outcome_score.py --since 2026-09-17T00:00:00
    python scripts/iv_outcome_score.py --label tool_error --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LLOYD_HOME))

from app.paths import EVENT_LOGS_DIR, USAGE_DB  # noqa: E402
from scripts.iv_grade import (  # noqa: E402
    INTERVENTIONS, WINDOW_CLAUSE, _SINCE_HELP, _is_deterministic, _rows,
)

MIN_RATED = 30
GOOD_STOPS = frozenset({"stop", "end_turn"})
LABELS = ("stop_reason", "tool_error")
_ERROR_PREFIXES = ("Error", "Traceback")
# `iv_grade.WINDOW_CLAUSE` is the lower bound; the upper one normalises the
# separator the same way, for the reason spelled out there (#835).
UNTIL_CLAUSE = "replace(created_at, 'T', ' ') < replace(?, 'T', ' ')"


def class_of(row: dict) -> str | None:
    action = row["action"]
    if action not in INTERVENTIONS:
        return None
    if action == "inject":
        return "inject[guard]" if _is_deterministic(row) else "inject[model]"
    return action


def _result_head(result: Any, blobs: Path) -> str:
    """The first bytes of a tool result, following a `$blob` reference."""
    if isinstance(result, str):
        return result[:64]
    if isinstance(result, dict) and isinstance(result.get("$blob"), str):
        try:
            with (blobs / f"{result['$blob']}.txt").open(encoding="utf-8") as fh:
                return fh.read(64)
        except OSError:
            return ""
    return ""


def turn_outcomes(event_logs: Path, keys: set[tuple[str, str]]) -> dict:
    """(session_id, turn_id) -> {"stop_reason": str|None, "tool_error": bool}."""
    wanted: dict[str, set[str]] = defaultdict(set)
    for sid, tid in keys:
        wanted[sid].add(tid)
    blobs = event_logs / "blobs"
    out: dict[tuple[str, str], dict] = {}
    for sid, turns in wanted.items():
        path = event_logs / f"{sid}.events.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                # Cheap pre-filter: a session log is mostly other events.
                if ("brain1.result_message" not in line
                        and "brain1.tool_result_received" not in line):
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                tid = ev.get("turn_id")
                if tid not in turns:
                    continue
                rec = out.setdefault((sid, tid), {"stop_reason": None, "tool_error": False})
                data = ev.get("data") or {}
                if ev.get("event") == "brain1.result_message":
                    rec["stop_reason"] = data.get("stop_reason") or "unknown"
                elif ev.get("event") == "brain1.tool_result_received":
                    head = _result_head(data.get("result"), blobs).lstrip()
                    if head.startswith(_ERROR_PREFIXES):
                        rec["tool_error"] = True
    return out


def is_bad(outcome: dict | None, label: str) -> bool | None:
    """True/False, or None when the turn cannot be labelled."""
    if outcome is None or outcome["stop_reason"] is None:
        return None
    if label == "tool_error":
        return outcome["tool_error"]
    return outcome["stop_reason"] not in GOOD_STOPS


def score(rows: list[dict], outcomes: dict, label: str) -> dict[str, Any]:
    turn_bad: dict[tuple[str, str], bool] = {}
    unlabelled_turns = set()
    fired: dict[tuple[str, str], set[str]] = defaultdict(set)
    per_class: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        key = (r["session_id"], r["turn_id"])
        bad = is_bad(outcomes.get(key), label)
        if bad is None:
            unlabelled_turns.add(key)
        else:
            turn_bad[key] = bad
        cls = class_of(r)
        if cls is None:
            continue
        c = per_class[cls]
        if bad is None:
            c["unlabelled"] += 1
            continue
        fired[key].add(cls)
        c["n"] += 1
        c["tp" if bad else "fp"] += 1

    bad_turns = {k for k, b in turn_bad.items() if b}
    classes = {}
    for cls, c in sorted(per_class.items()):
        caught = sum(1 for k in bad_turns if cls in fired.get(k, ()))
        rated = c["n"] >= MIN_RATED
        classes[cls] = {
            "n": c["n"], "tp": c["tp"], "fp": c["fp"], "unlabelled": c["unlabelled"],
            "bad_turns_caught": caught,
            "precision": round(c["tp"] / c["n"], 3) if rated else None,
            "recall": round(caught / len(bad_turns), 3) if rated and bad_turns else None,
        }
    base = round(len(bad_turns) / len(turn_bad), 4) if turn_bad else None
    for v in classes.values():
        v["lift"] = (round(v["precision"] / base, 2)
                     if v["precision"] is not None and base else None)
    return {
        "label": label,
        "turns_labelled": len(turn_bad),
        "turns_unlabelled": len(unlabelled_turns),
        "bad_turns": len(bad_turns),
        "base_rate": base,
        "fn_pool": sum(1 for k in bad_turns if not fired.get(k)),
        "classes": classes,
    }


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help=_SINCE_HELP)
    ap.add_argument("--until", help="Upper bound on created_at (exclusive), same clock as --since.")
    ap.add_argument("--label", choices=LABELS, default="stop_reason",
                    help="which event-log field decides a bad turn (default stop_reason)")
    ap.add_argument("--db", type=Path, default=USAGE_DB, help="usage.db (opened read-only)")
    ap.add_argument("--event-logs", type=Path, default=EVENT_LOGS_DIR)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"no usage db at {args.db}", file=sys.stderr)
        return 1
    where, params = [], []
    if args.since:
        where.append(WINDOW_CLAUSE)
        params.append(args.since)
    if args.until:
        where.append(UNTIL_CLAUSE)
        params.append(args.until)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = _rows(conn, clause, params)
    finally:
        conn.close()

    keys = {(r["session_id"], r["turn_id"]) for r in rows}
    report = score(rows, turn_outcomes(args.event_logs, keys), args.label)
    report["window"] = {"since": args.since or "all time", "until": args.until or "now",
                        "observations": len(rows)}
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    w = report["window"]
    print(f"\nInner Voice outcome score — {w['since']} to {w['until']}, "
          f"{w['observations']} observations")
    print(f"label: {report['label']} (event log only; no IV column decides an outcome)")
    print(f"turns labelled {report['turns_labelled']}, unlabelled {report['turns_unlabelled']}, "
          f"bad {report['bad_turns']}, base rate {report['base_rate']}")
    print("=" * 78)
    print(f"  {'class':<16}{'n':>6}{'TP':>6}{'FP':>6}{'precision':>11}{'recall':>9}"
          f"{'lift':>7}{'unlab':>7}")
    for cls, v in report["classes"].items():
        lift = "—" if v["lift"] is None else f"{v['lift']:.2f}"
        print(f"  {cls:<16}{v['n']:>6}{v['tp']:>6}{v['fp']:>6}{_fmt(v['precision']):>11}"
              f"{_fmt(v['recall']):>9}{lift:>7}{v['unlabelled']:>7}")
    print(f"\n  FN pool: {report['fn_pool']} bad turns with no intervention at all")
    print(f"\n  '—' = under {MIN_RATED} scored rows: counts only, no rate.")
    print("  lift = precision / base rate. Above 1 says the class fires more often on"
          " turns that end badly;\n  it does not say the intervention caused or"
          " prevented anything (see the docstring's\n  counterfactual limit)."
          " The authority decision is a human call.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
