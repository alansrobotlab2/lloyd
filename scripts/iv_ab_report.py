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
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LLOYD_HOME))

from app.paths import EVENT_LOGS_DIR, SESSIONS_DIR, describe_tree  # noqa: E402
from scripts.iv_grade import _CORRECTION_RE  # noqa: E402
from scripts.iv_outcome_score import GOOD_STOPS, _ERROR_PREFIXES, _result_head  # noqa: E402


def _text(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    return "".join(ch.get("text") or "" for ch in (c or [])
                   if isinstance(ch, dict) and ch.get("type") == "text")


#: The archived form of a transcript, named as `app/uptake.py:622` names it. The
#: retention sweep (`scripts/groundskeeper/retention-sweep.py::sweep_sessions`) gzips
#: `<id>.json` to `<id>.json.gz` once `last_active` passes `SESSION_ARCHIVE_AGE_DAYS`
#: (90) and unlinks the original, on a policy that is *gzip and never delete* — so past
#: that horizon an archive is the ONLY copy a session will ever have. Both readers here
#: globbed `*.json`, which made the weekly sweep an eraser of this report's own
#: evidence: a window older than 90 days reported `0 vs 0` arms with every rate null AND
#: `enrollment_gap_line: null` — the exact silent reading #3220 exists to make
#: impossible — because `main` prints the line only `if gap["unstamped"]` and an
#: archived window has no unstamped file it can see. Measured 2026-11-17, the store held
#: 10187 `.json` against 13891 `.json.gz`: two thirds of the corpus invisible. The
#: precedent is uptake's own (#1848, the same store): `app/uptake.py:626` enumerates
#: both forms and `:655` reads them gzip-transparently (#4214).
SESSION_ARCHIVE_SUFFIX = ".json.gz"
SESSION_SUFFIXES = (".json", SESSION_ARCHIVE_SUFFIX)
#: A turn-log sibling in either form: `[{turn}, ...]`, not a session.
#: `enrollment_gap` has always skipped the `.json` spelling, and the skip has to be
#: archive-aware or widening the glob re-admits every turn log as a chat.
TURNLOG_SUFFIXES = ("-turnlog.json", "-turnlog.json.gz")


class UnreadableTranscript(Exception):
    """A transcript whose bytes cannot be read or parsed at all.

    The shapes are the ones uptake measured on the same store (`app/uptake.py:655-672`):
    a live file whose JSON is corrupt, an archive cut mid-write (`EOFError`), bytes
    behind a `.json.gz` name that were never a gzip stream (`BadGzipFile`, an `OSError`),
    and a valid header over a broken deflate stream, which leaves the decompressor as
    `zlib.error` and is the only one of those that is not an `OSError`. Named as a class
    rather than caught inline because both readers have to distinguish "I could not read
    this" from "this is not the experiment's session" — collapsing them is what made a
    wiped store and an unreadable store print the same line.
    """


def session_id_of(path: Path) -> str:
    """The session id a transcript names, with either suffix.

    NOT `path.stem`: for an archive that is `<id>.json`, which matches no session id
    anywhere in the store.
    """
    name = path.name
    for suffix in (SESSION_ARCHIVE_SUFFIX, ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def session_transcripts(sessions_dir: Path) -> list[Path]:
    """Every session transcript in the store, live or archived, ordered by name, turn
    logs excluded — the ONE population both readers of this report walk.

    A session held in BOTH forms is the state a sweep interrupted between its
    `gzip.open()` and its `path.unlink()`; it resolves to the live file, which is the
    current prose, exactly as `app/uptake.py:626-639` resolves it. Counting the pair
    twice would inflate a window's population out of one half-finished write.
    """
    by_id: dict[str, Path] = {}
    for suffix in SESSION_SUFFIXES:                 # ".json" first, and it wins
        for path in sessions_dir.glob(f"*{suffix}"):
            if path.name.endswith(TURNLOG_SUFFIXES):
                continue
            by_id.setdefault(session_id_of(path), path)
    return sorted(by_id.values())


def read_transcript(path: Path):
    """One parsed transcript, gzip-transparent.

    Raises `UnreadableTranscript` when the file cannot be read or parsed; a document
    that parses but is not a session object comes back as it stands, which is what this
    script reported before archives existed.
    """
    try:
        if path.name.endswith(SESSION_ARCHIVE_SUFFIX):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                return json.load(fh)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — one rotten file must never end the walk,
                              # and both readers counted "could not read" long before
                              # any of them could name the exception
        raise UnreadableTranscript(f"{path.name}: {type(exc).__name__}: {exc}") from exc


def load_sessions(sessions_dir: Path, experiment: str) -> list[dict]:
    """Every session stamped for `experiment`, live or archived.

    Walks `session_transcripts`, the same population `enrollment_gap` counts, so an
    aged window can never empty the arms while still naming a gap in the line below
    them (#4214).
    """
    out = []
    for path in session_transcripts(sessions_dir):
        try:
            data = read_transcript(path)
        except UnreadableTranscript:  # an unreadable file is not in the experiment
            continue
        if not isinstance(data, dict):
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
#: to fix a window that enrolled nobody. `app/sessions_io.py:1449` gates assignment on
#: `data["platform"] == "mission-control"` and `:1418` is the sole call, inside
#: `_save_session_meta` and reached only on that function's create branch;
#: `create_session` (`app/sessions_io.py:398`) is the writer every chat actually goes
#: through (`POST /api/sessions/create` mints through it at `app/routers/sessions.py:988`)
#: and it never reaches that branch. The asymmetry is why iv-chat-1 read 0/0 across a
#: window that held 29 mission-control sessions, none of them stamped (#3220).
ENROLLED_PLATFORM = "mission-control"
STAMPING_SITE = "_save_session_meta (app/sessions_io.py:1451)"
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
    function walks — `session_transcripts`, every transcript in the store live or
    archived (`<id>.json` and `<id>.json.gz`), turn logs excluded — and reports what it
    could not read. Reading only the live half would have this counter fall silent at
    exactly the horizon the report exists to be read at: the sweep archives a window at
    90 days inactive, `main` prints the line only `if gap["unstamped"]`, and an all
    archived window then renders as the clean 0/0 that #3220 was built to make
    impossible (#4214).

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
    for path in session_transcripts(sessions_dir):
        try:
            rec = read_transcript(path)
        except UnreadableTranscript:
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


#: What a void run exits with (#4215). Distinct from argparse's usage error (2) and from
#: the 1 a traceback leaves, so a caller can branch on "the store named nothing" without
#: parsing prose, and a wrapper that ignores the difference cannot mistake the two.
VOID_EXIT = 3


def void_reason(sessions_dir: Path) -> str | None:
    """Why this sessions store measures NOTHING, or None when it holds sessions.

    `app.paths` anchors `SESSIONS_DIR` to the tree it is imported from (#733), so run
    this script from a linked worktree — an automod round, an arch-review checkout, an
    agent scratch tree — and the default store is a path inside that tree which was
    never created. The report then printed both arms at 0 sessions / 0 turns with every
    rate, z and median null, `enrollment_gap_line: null` and exit 0, over a store that
    holds ~10,000 sessions in production: a clean no-gap reading of a store that is not
    there. Nothing caught it because `enrollment_gap` counts only files it finds, and a
    walk of a missing directory finds none — an absent store and a healthy one took the
    same path, and `main` suppresses the gap line whenever `unstamped == 0`.

    Two shapes, one verdict: the directory is not there, or nothing in it parses. The
    walk SHORT-CIRCUITS at the first transcript that parses, because the question is
    "is the count zero" and the production store holds ~10,000 transcripts; a full
    re-parse to answer it on a healthy store would cost the report its own runtime.
    """
    if not sessions_dir.is_dir():
        return f"the resolved sessions dir does not exist: {sessions_dir}"
    for path in session_transcripts(sessions_dir):
        try:
            read_transcript(path)
        except UnreadableTranscript:
            continue
        return None
    return f"no session transcript in the resolved sessions dir parses: {sessions_dir}"


def _void_payload(sessions_dir: Path, event_logs: Path, experiment: str,
                  reason: str) -> dict:
    """The JSON of a void run: the marker, the reason, the tree — and no measurement.

    Deliberately NOT the report shape. `arms`, `z`, `enrollment_gap` and
    `enrollment_gap_line` are ABSENT rather than null, because those nulls are precisely
    the payload a doc or a later run cites; a void artifact that carried them would
    still be quotable as "the experiment did nothing" (#3220's shape, #4215's route).
    """
    return {"void": True,
            "void_reason": reason,
            "sessions_dir": str(sessions_dir),
            "event_logs": str(event_logs),
            "event_logs_exist": event_logs.is_dir(),
            "experiment": experiment,
            "tree": describe_tree()}


def _print_void(reason: str, sessions_dir: Path, event_logs: Path) -> None:
    """The plain-mode void report: which directory, which kind of tree, why it voids.

    `describe_tree()` is #733's line and it is the only thing that distinguishes an
    absent worktree store from an empty real one, so it travels here as well as in the
    JSON rather than only on stderr as an import warning a caller may never grep.
    """
    print("VOID — the Inner Voice A/B reading measured nothing.")
    print(f"  reason:      {reason}")
    print(f"  sessions:    {sessions_dir}")
    print(f"  tree:        {describe_tree()}")
    if not event_logs.is_dir():
        print(f"  event logs:  {event_logs} (does not exist)")
    print("\n  No session in that store could be read, so every count this report would\n"
          "  print is a zero of an empty walk, not a measurement. Run it from the main\n"
          "  checkout, or pass --sessions-dir naming the real store.\n")


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
    # #4215, before anything is computed: a store that cannot be read is not a window
    # with nothing in it. Keyed on the SESSIONS STORE ALONE — the event-log root is named
    # in both views below but never voids a run, because #3220's clause 4 runs a real
    # store against an absent log root and requires the verdict and the exit code to
    # stay put.
    reason = void_reason(args.sessions_dir)
    if reason:
        if args.json:
            print(json.dumps(_void_payload(args.sessions_dir, args.event_logs,
                                           name, reason), indent=2))
        else:
            _print_void(reason, args.sessions_dir, args.event_logs)
        return VOID_EXIT
    rep = report(load_sessions(args.sessions_dir, name), args.event_logs,
                 since=args.since, until=args.until)
    rep["experiment"] = name
    # Counted over the same store and window `load_sessions` just filtered, and before
    # either output branch, so the plain run and --json can never disagree about it.
    gap = enrollment_gap(args.sessions_dir, name, args.since, args.until)
    if args.json:
        print(json.dumps({**rep, "enrollment_gap": gap,
                          "enrollment_gap_line": (gap_line(gap)
                                                  if gap["unstamped"] else None),
                          # The three corpus fields, so an artifact says what it was
                          # measured over: `void` is the machine-readable answer to
                          # #4215, and the tree is #733's distinction.
                          "void": False,
                          "sessions_dir": str(args.sessions_dir),
                          "event_logs": str(args.event_logs),
                          "event_logs_exist": args.event_logs.is_dir(),
                          "tree": describe_tree()},
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
    # Named, never fatal (#4215). An absent log root leaves every outcome unverifiable,
    # which the reader is entitled to know, and the nulls above are then a null of that
    # and not of the arms. A populated store read against one is still a run whose exit
    # code belongs to the store alone (#3220 clause 4).
    if not args.event_logs.is_dir():
        print(f"\n  Event logs: {args.event_logs} does not exist, so no turn's outcome "
              f"was readable and every rate above is a null of that, not of the arms.\n")
    print("  Outcomes come from the event log; nothing the observer wrote decides one.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
