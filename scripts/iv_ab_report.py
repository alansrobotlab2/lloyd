#!/usr/bin/env python3
"""Inner Voice on/off A/B — the reading half (IV plan R3).

`app/inner_voice/ab.py` assigns each new chat session inside the window to an
arm and stamps it on the session file (`inner_voice_ab`). This joins those
sessions to their event logs and compares the arms on what the event log says
happened, never on what the observer wrote about itself:

  bad_stop     the turn's `brain1.result_message.stop_reason` is not
               `stop`/`end_turn` (`max_turns`, `cancelled`, `error`, …)
  tool_error   any tool result in the turn opens with `Error`/`Traceback`
               (the same rule as `iv_outcome_score.py`)
  iterations   `num_turns` on the result message (agent-loop iterations)
  duration_s   `duration_ms` on the result message
  correction   the user's NEXT message reads as a correction
               (`iv_grade._CORRECTION_RE` — narrow on purpose)

Per arm: sessions, turns, the two rates and the correction rate with a
two-proportion z statistic against the other arm, and the medians. A
difference under |z| 2 is reported as "no detectable difference", which with
a few hundred turns is what most weeks will say — that is the honest reading,
not a failure of the report.

A session whose flags were changed by hand after assignment (the arm says
`on` but `inner_voice` is now false, or the reverse) is counted as
`crossed_over` and left out: it no longer measures its arm.

Read-only: session files and event logs are read, nothing is written.

    python scripts/iv_ab_report.py
    python scripts/iv_ab_report.py --experiment iv-chat-1 --since 2026-09-25 --json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LLOYD_HOME))

from app.paths import EVENT_LOGS_DIR, SESSIONS_DIR  # noqa: E402
from scripts.iv_grade import _CORRECTION_RE  # noqa: E402
from scripts.iv_outcome_score import GOOD_STOPS, _ERROR_PREFIXES, _result_head  # noqa: E402


def _text(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    return "".join(ch.get("text") or "" for ch in (c or [])
                   if isinstance(ch, dict) and ch.get("type") == "text")


def load_sessions(sessions_dir: Path, experiment: str) -> list[dict]:
    out = []
    for p in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 — an unreadable file is not in the experiment
            continue
        ab = data.get("inner_voice_ab") or {}
        if ab.get("experiment") != experiment or ab.get("arm") not in ("on", "off"):
            continue
        out.append(data)
    return out


def turn_rows(event_logs: Path, session_id: str) -> list[dict]:
    """One dict per finished turn, in log order."""
    path = event_logs / f"{session_id}.events.jsonl"
    if not path.exists():
        return []
    blobs = event_logs / "blobs"
    turns: dict[str, dict] = {}
    order: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if ("brain1.result_message" not in line
                    and "brain1.tool_result_received" not in line):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            tid = ev.get("turn_id")
            if not tid:
                continue
            if tid not in turns:
                turns[tid] = {"turn_id": tid, "stop_reason": None, "tool_error": False,
                              "iterations": None, "duration_s": None}
                order.append(tid)
            rec, data = turns[tid], ev.get("data") or {}
            if ev.get("event") == "brain1.result_message":
                rec["stop_reason"] = data.get("stop_reason") or "unknown"
                rec["iterations"] = data.get("num_turns")
                if isinstance(data.get("duration_ms"), (int, float)):
                    rec["duration_s"] = data["duration_ms"] / 1000.0
            else:
                head = _result_head(data.get("result"), blobs).lstrip()
                if head.startswith(_ERROR_PREFIXES):
                    rec["tool_error"] = True
    return [turns[t] for t in order if turns[t]["stop_reason"] is not None]


def corrections(session: dict) -> list[bool]:
    """For each real user message after the first: does it read as a correction?"""
    users = [m for m in session.get("messages") or []
             if m.get("role") == "user" and (m.get("source") or "user") == "user"]
    return [bool(_CORRECTION_RE.search(_text(m))) for m in users[1:]]


def _z(p1: float, n1: int, p2: float, n2: int) -> float | None:
    if not n1 or not n2:
        return None
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return None if se == 0 else round((p1 - p2) / se, 2)


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def report(sessions: list[dict], event_logs: Path, *, since: str = "",
           until: str = "") -> dict[str, Any]:
    arms: dict[str, dict] = defaultdict(lambda: {
        "sessions": 0, "turns": 0, "bad_stop": 0, "tool_error": 0,
        "iterations": [], "duration_s": [], "corrections": 0, "followups": 0})
    crossed = 0
    for s in sessions:
        sid = str(s.get("session_id") or "")
        created = str(s.get("created_at") or "")
        if since and created[:len(since)] < since:
            continue
        if until and created[:len(until)] >= until:
            continue
        arm = s["inner_voice_ab"]["arm"]
        if bool(s.get("inner_voice")) != (arm == "on"):
            crossed += 1
            continue
        a = arms[arm]
        a["sessions"] += 1
        for t in turn_rows(event_logs, sid):
            a["turns"] += 1
            a["bad_stop"] += t["stop_reason"] not in GOOD_STOPS
            a["tool_error"] += t["tool_error"]
            a["iterations"].append(t["iterations"])
            a["duration_s"].append(t["duration_s"])
        cs = corrections(s)
        a["followups"] += len(cs)
        a["corrections"] += sum(cs)
    out: dict[str, Any] = {"crossed_over": crossed, "arms": {}}
    for arm in ("on", "off"):
        a = arms[arm]
        n, f = a["turns"], a["followups"]
        out["arms"][arm] = {
            "sessions": a["sessions"], "turns": n,
            "bad_stop_rate": round(a["bad_stop"] / n, 4) if n else None,
            "tool_error_rate": round(a["tool_error"] / n, 4) if n else None,
            "correction_rate": round(a["corrections"] / f, 4) if f else None,
            "followups": f,
            "median_iterations": _median(a["iterations"]),
            "median_duration_s": _median(a["duration_s"]),
        }
    on, off = out["arms"]["on"], out["arms"]["off"]
    out["z"] = {}
    for key, denom in (("bad_stop_rate", "turns"), ("tool_error_rate", "turns"),
                       ("correction_rate", "followups")):
        if on[key] is None or off[key] is None:
            out["z"][key] = None
            continue
        out["z"][key] = _z(on[key], on[denom], off[key], off[denom])
    return out


def _verdict(z: float | None) -> str:
    if z is None:
        return "not enough data"
    if abs(z) < 2:
        return "no detectable difference"
    return "IV-on lower" if z < 0 else "IV-on higher"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=None,
                    help="experiment name (default: inner_voice.ab.name)")
    ap.add_argument("--since", default="", help="session created_at lower bound (local)")
    ap.add_argument("--until", default="", help="session created_at upper bound (local)")
    ap.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    ap.add_argument("--event-logs", type=Path, default=EVENT_LOGS_DIR)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    name = args.experiment
    if not name:
        from app.inner_voice.ab import ab_config
        name = ab_config()["name"]
    rep = report(load_sessions(args.sessions_dir, name), args.event_logs,
                 since=args.since, until=args.until)
    rep["experiment"] = name
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    print(f"\nInner Voice A/B — experiment {name}"
          f"{' since ' + args.since if args.since else ''}"
          f"{' until ' + args.until if args.until else ''}")
    print("=" * 72)
    print(f"  {'':<20}{'IV on':>14}{'IV off':>14}   z, verdict")
    rows = [("sessions", "sessions", None), ("turns", "turns", None),
            ("bad stop rate", "bad_stop_rate", "bad_stop_rate"),
            ("tool error rate", "tool_error_rate", "tool_error_rate"),
            ("correction rate", "correction_rate", "correction_rate"),
            ("median iterations", "median_iterations", None),
            ("median duration s", "median_duration_s", None)]
    for label, key, zkey in rows:
        on, off = rep["arms"]["on"][key], rep["arms"]["off"][key]
        tail = ""
        if zkey:
            z = rep["z"][zkey]
            tail = f"   {'—' if z is None else z}, {_verdict(z)}"
        print(f"  {label:<20}{str(on if on is not None else '—'):>14}"
              f"{str(off if off is not None else '—'):>14}{tail}")
    print(f"\n  crossed over (flags changed by hand after assignment): {rep['crossed_over']}")
    print("  Outcomes come from the event log; nothing the observer wrote decides one.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
