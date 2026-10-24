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


#: The one platform the stamping branch admits, and the two writer sites a reader needs
#: to fix a window that enrolled nobody. `app/sessions_io.py:1361` gates assignment on
#: `data["platform"] == "mission-control"` and `:1363` is the sole call, inside
#: `_save_session_meta` and reached only on that function's create branch;
#: `create_session` (`app/sessions_io.py:398`) is the writer every chat actually goes
#: through (`POST /api/sessions/create` mints through it at `app/routers/sessions.py:988`)
#: and it never reaches that branch. The asymmetry is why iv-chat-1 read 0/0 across a
#: window that held 29 mission-control sessions, none of them stamped (#3220).
ENROLLED_PLATFORM = "mission-control"
STAMPING_SITE = "_save_session_meta (app/sessions_io.py:1363)"
CREATE_WRITER = "create_session (app/sessions_io.py:398)"


def _configured_window(experiment: str) -> tuple[str, str]:
    """The experiment's own `inner_voice.ab` start/end, or ("", "") when this is not it.

    `ab_config()` reads the live `config.yaml`, so the window counted against is the one
    the engine actually armed and not one this script re-derives. Local import and a
    swallow, both load-bearing: the test execs this file standalone with no `app`
    importable, and a diagnostic must never be the reason a report cannot run.
    """
    try:
        from app.inner_voice.ab import ab_config
        cfg = ab_config()
    except Exception:
        return "", ""
    if cfg.get("name") != experiment:
        return "", ""
    return str(cfg.get("start") or ""), str(cfg.get("end") or "")


def enrollment_gap(sessions_dir: Path, experiment: str, since: str = "",
                   until: str = "") -> dict:
    """Count the window's sessions of the enrolled platform by whether they carry a stamp.

    `load_sessions` selects solely on the stored stamp, so an unstamped chat is invisible
    to it and a window that enrolled nobody prints the same 0/0 with null rates that a
    window where the treatment did nothing would. This walks the same population that
    function globs — top-level `*.json`, turn logs excluded — and reports what it could
    not read.

    Window: `--since`/`--until` when both bound it, else the experiment's own
    `inner_voice.ab` start/end, else all time. End-exclusive, the convention of
    `window_open` (`app/inner_voice/ab.py:69-76`), so a `--until 2026-10-09` does not
    count 2026-10-09.
    """
    lo = since[:10] if since else ""
    hi = until[:10] if until else ""
    source = "argv --since/--until" if (lo and hi) else "all time (no window given)"
    if not (lo and hi):
        cfg_lo, cfg_hi = _configured_window(experiment)
        if cfg_lo and cfg_hi:
            lo, hi, source = cfg_lo, cfg_hi, "config inner_voice.ab"

    in_window = stamped = undated = unreadable = 0
    for path in sorted(sessions_dir.glob("*.json")):
        if path.name.endswith("-turnlog.json"):
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            unreadable += 1
            continue
        if not isinstance(rec, dict) or rec.get("platform") != ENROLLED_PLATFORM:
            continue
        day = str(rec.get("created_at") or "")[:10]
        if not day:
            undated += 1
            continue
        if (lo and day < lo) or (hi and day >= hi):
            continue
        in_window += 1
        ab = rec.get("inner_voice_ab")
        if isinstance(ab, dict) and ab.get("experiment") == experiment:
            stamped += 1
    return {"experiment": experiment, "platform": ENROLLED_PLATFORM,
            "window_start": lo, "window_end": hi, "window_source": source,
            "platform_sessions": in_window, "stamped": stamped,
            "unstamped": in_window - stamped, "undated": undated,
            "unreadable": unreadable}


def gap_line(gap: dict) -> str:
    """The one line that tells a reader of an empty window why the window is empty."""
    span = (f"{gap['window_start'] or '..'}..{gap['window_end'] or '..'} "
            f"({gap['window_source']})")
    extra = ""
    if gap["undated"] or gap["unreadable"]:
        extra = (f" ({gap['undated']} of the platform's files carried no created_at and "
                 f"{gap['unreadable']} did not parse, so neither is counted here)")
    return (f"ENROLLMENT-GAP: {gap['platform_sessions']} {gap['platform']} sessions in "
            f"window {span}, {gap['stamped']} stamped for {gap['experiment']}, "
            f"{gap['unstamped']} unstamped{extra} — an arm is written only by "
            f"{STAMPING_SITE}; {CREATE_WRITER}, which mints every chat the API creates, "
            f"writes no stamp, so this window cannot be read as an experiment.")


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
    # Counted over the same store and window `load_sessions` just filtered, and before
    # either output branch, so the plain run and --json can never disagree about it.
    gap = enrollment_gap(args.sessions_dir, name, args.since, args.until)
    if args.json:
        print(json.dumps({**rep, "enrollment_gap": gap,
                          "enrollment_gap_line": (gap_line(gap)
                                                  if gap["unstamped"] else None)},
                         indent=2))
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
    # Only when a session that could have been enrolled was not. A window whose every
    # platform session carries the stamp has no gap to report, and printing a line there
    # would teach the reader to ignore the line.
    if gap["unstamped"]:
        print(f"\n  {gap_line(gap)}\n")
    print("  Outcomes come from the event log; nothing the observer wrote decides one.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
