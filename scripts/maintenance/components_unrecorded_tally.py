#!/usr/bin/env python3
"""Per-send-site unrecorded tally of the request manifest, for #1879's residual.

#1879's acceptance is a claim about traffic over a day, and the check the item names
(`cm_scan5.py`, `cm_1879_check.py`) lived in /tmp — a directory that is wiped, so the
next pass had to rewrite the measurement and could quietly write a different one. This
is that measurement, in the tree, with the two halves the item needs joined into one
run:

  * the per-`send_site` unrecorded tally, with the DENOMINATOR printed beside every
    verdict (a percentage over zero lines is not a check);
  * the `turn_start` cross-check that attributes the residual into buckets that are
    distinguished by DIFFERENT evidence, because the manifest alone cannot tell them
    apart. Each residual session lands in the FIRST bucket whose evidence it has:

      1. **registry_eviction** — it emitted `turn_start` earlier in the day and reads
         `unrecorded` later, which `86aef7e5` made possible when it raised the
         registry bound and counted its drops.
      2. **trial_traffic** — its manifest lines carry `source` `bench`/`benchmine`
         (or, failing that field, its session-id slug is one of those). Trials run
         from copies of the runner — a round worktree, a detached grid — that may
         predate the threading fix, so they are reported as what they are, with
         their session count and burst window (first and last `ts`), and they are
         tested BEFORE the session store: whether a trial has a session file says
         nothing about why its lines are unrecorded.
      3. **unthreaded_send_site** — no `turn_start` line and a session file in
         `~/lloyd-data/sessions`: the thing #1879 fixes.
      4. **restart_boundary** — no session file AND a process-death discriminant: a
         restart time passed with `--restart-at` that falls after the session was
         minted (the `YYYYMMDD_HHMMSS` its id starts with) and at or before its
         first unrecorded line. Without `--restart-at` this bucket is empty by
         construction, and the report says so.
      5. **session_file_absent** — no session file and nothing else. This is the raw
         observation and it is NOT a cause.

Session-file absence used to be called a restart boundary on its own (#1984). It is
not evidence of one: on 2026-10-01 fifteen `bench` trials read `restart_boundary: 15`
on a day the claim was never checked against any restart. What makes a trial's
session file go missing is not established. One CANDIDATE, unconfirmed:
`scripts/autoresearch/bench_runner_sdk.py:691-696` calls `create_session` in a `try`
that only logs a warning, so a trial can send under an id the session store never
received — but no such warning was found in any trajectory for those fifteen, and 56
sibling trials of the same night do have files, so do not read it as the cause.

The file test is the session store and not the manifest: every unrecorded line
carries its own `session_id`, so asking the manifest whether a session "appears in the
day" answers yes by construction. #1879's own triage ran the check that way — all 29
unrecorded ids on 2026-09-30 exist as session files, which is what let it say the
residual is unthreaded-site traffic. The store is `*.json` only; a retention sweep
that gzips old worker sessions would read as file-absent, which is one more reason
trial traffic is bucketed before the store is asked.

Every bucket line prints its count over the day's unrecorded-session denominator.

Usage:
    python3 scripts/maintenance/components_unrecorded_tally.py [YYYY-MM-DD ...]

Exit status: 0 when the window has lines at the stream_chat site, 2 when the store or a
day is missing or holds no line at that site — never a clean-looking non-answer.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import datetime, timezone

DEFAULT_STORE = os.path.expanduser(
    "~/.local/state/lloyd-request-manifests/manifests")
STREAM_CHAT = "app/harness/client.py::stream_chat"
#: In evidence order — `tally` puts a residual in the first one that applies.
KINDS = ("registry_eviction", "trial_traffic", "unthreaded_send_site",
         "restart_boundary", "session_file_absent")
#: Manifest `source` values, and session-id slugs, that mark background trial runs.
TRIAL_SOURCES = frozenset({"bench", "benchmine"})


def _epoch(value) -> float | None:
    """A manifest `ts` (epoch seconds, as a number or a string, or ISO-8601) → epoch."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _slug(sid: str) -> str:
    parts = sid.split("_")
    return parts[2] if len(parts) >= 4 else "chat"


def _minted_at(sid: str) -> float | None:
    """When a `YYYYMMDD_HHMMSS_<slug>_<hex>` id was minted. The stamp is local time
    (`app.sessions_io`), so it is read as local time."""
    parts = sid.split("_")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None


def day_rows(store: str, day: str) -> list[dict]:
    path = os.path.join(store, f"{day}.ndjson")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def known_sessions(sessions_dir: str) -> set[str]:
    """Every session id the session store knows about, on any day.

    `~/lloyd-data/sessions/<id>.json`, the file `app.sessions_io.create_session`
    writes when a run is minted. This is the second corpus the attribution needs:
    a line's own `session_id` cannot tell you whether the session is real, because
    every line carries one, so whether it ever had a session file has to be found
    by asking the store. A missing file is an observation (`session_file_absent`),
    not a cause.
    """
    try:
        return {p.name[:-5] for p in os.scandir(sessions_dir)
                if p.name.endswith(".json")}
    except OSError:
        return set()


def tally(rows: list[dict], *, session_ids: set[str],
          restarts: tuple[float, ...] = ()) -> dict:
    """The report. `denominator == 0` means there is nothing to rule on.

    `session_ids` is `known_sessions()` of the session store; pass an empty set to
    say "the store could not be read", which puts every non-trial residual in
    `session_file_absent` only after the `turn_start` and trial tests. `restarts`
    are epoch times of known process restarts — the only thing that can put a
    residual in `restart_boundary`.
    """
    by_site_total: collections.Counter = collections.Counter()
    by_site_unrec: collections.Counter = collections.Counter()
    turn_start_sessions: set[str] = set()
    unrec_lines: collections.Counter = collections.Counter()
    unrec_sources: dict[str, set[str]] = collections.defaultdict(set)
    unrec_ts: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        site = str(row.get("send_site") or "?")
        captured = str(row.get("components_captured") or "")
        sid = str(row.get("session_id") or "")
        by_site_total[site] += 1
        if captured == "unrecorded":
            by_site_unrec[site] += 1
            if sid:
                unrec_lines[sid] += 1
                if row.get("source"):
                    unrec_sources[sid].add(str(row["source"]))
                ts = _epoch(row.get("ts"))
                if ts is not None:
                    unrec_ts[sid].append(ts)
        elif captured == "turn_start":
            turn_start_sessions.add(sid)
    sc_total = by_site_total.get(STREAM_CHAT, 0)
    sc_unrec = by_site_unrec.get(STREAM_CHAT, 0)
    attribution: dict[str, list[str]] = {k: [] for k in KINDS}
    for sid in unrec_lines:
        if sid in turn_start_sessions:
            attribution["registry_eviction"].append(sid)
        elif (unrec_sources[sid] & TRIAL_SOURCES) or (
                not unrec_sources[sid] and _slug(sid) in TRIAL_SOURCES):
            # Before the session store, on purpose: see the module docstring.
            attribution["trial_traffic"].append(sid)
        elif sid in session_ids:
            attribution["unthreaded_send_site"].append(sid)
        elif _spans_a_restart(sid, unrec_ts[sid], restarts):
            attribution["restart_boundary"].append(sid)
        else:
            attribution["session_file_absent"].append(sid)
    trial = attribution["trial_traffic"]
    trial_ts = [t for sid in trial for t in unrec_ts[sid]]
    return {
        "lines": len(rows),
        "send_sites": {
            site: {"lines": total,
                   "unrecorded": by_site_unrec.get(site, 0),
                   "pct": (100.0 * by_site_unrec.get(site, 0) / total) if total else None}
            for site, total in sorted(by_site_total.items())},
        "stream_chat": {"lines": sc_total, "unrecorded": sc_unrec,
                        "pct": (100.0 * sc_unrec / sc_total) if sc_total else None},
        "unrecorded_sessions": len(unrec_lines),
        "attribution": {k: sorted(v) for k, v in attribution.items()},
        "trial_burst": {
            "sessions": len(trial),
            "lines": sum(unrec_lines[sid] for sid in trial),
            "first_ts": min(trial_ts) if trial_ts else None,
            "last_ts": max(trial_ts) if trial_ts else None},
        "restarts_supplied": len(restarts),
        "source_mix": dict(collections.Counter(_slug(sid) for sid in unrec_lines)),
    }


def _spans_a_restart(sid: str, line_ts: list[float], restarts: tuple[float, ...]) -> bool:
    """The process-death discriminant: some supplied restart falls after the session
    was minted and at or before its first unrecorded line. Any missing half — no
    restart supplied, an id with no stamp, lines with no `ts` — is "not shown"."""
    minted = _minted_at(sid)
    if not restarts or minted is None or not line_ts:
        return False
    first = min(line_ts)
    return any(minted < r <= first for r in restarts)


def _stamp(ts: float | None) -> str:
    if ts is None:
        return "no ts"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_sessions_dir(store: str) -> str:
    """`~/lloyd-data/sessions`, located from the store root's home directory.

    Derived, not hardcoded, so a check run against a scratch store can be pointed at
    a scratch session store — and so a moved root cannot quietly turn every residual
    session into `session_file_absent`, which is what an unreadable session dir
    would mean if the two paths were assumed to move together.
    """
    marker = os.sep + ".local" + os.sep + "state" + os.sep
    root = os.path.abspath(store)
    home = root.split(marker)[0] if marker in root else os.path.expanduser("~")
    return os.path.join(home, "lloyd-data", "sessions")


def report(store: str, days: list[str], *, json_out: bool,
           sessions_dir: str = "", restarts: tuple[float, ...] = ()) -> int:
    rc = 0
    out = []
    sessions_dir = sessions_dir or default_sessions_dir(store)
    session_ids = known_sessions(sessions_dir)
    for day in days:
        rows = day_rows(store, day)
        rep = tally(rows, session_ids=session_ids, restarts=restarts)
        rep["day"] = day
        rep["store"] = store
        rep["sessions_dir"] = sessions_dir
        rep["sessions_known"] = len(session_ids)
        out.append(rep)
        if not rows or rep["stream_chat"]["lines"] == 0:
            rc = 2
    if json_out:
        print(json.dumps(out, indent=1, sort_keys=True))
        return rc
    for rep in out:
        print(f"== {rep['day']}  store={rep['store']}")
        print(f"   lines={rep['lines']}  unrecorded_sessions={rep['unrecorded_sessions']}  "
              f"sessions_known={rep['sessions_known']} ({rep['sessions_dir']})")
        for site, s in rep["send_sites"].items():
            pct = "n/a (denominator 0)" if s["pct"] is None else f"{s['pct']:.1f}%"
            print(f"   {site:64s} lines={s['lines']:6d} unrecorded={s['unrecorded']:5d} {pct}")
        sc = rep["stream_chat"]
        if sc["lines"] == 0:
            print("   NO stream_chat LINE IN THIS WINDOW — no denominator, no verdict")
        else:
            print(f"   TARGET {STREAM_CHAT}: {sc['unrecorded']}/{sc['lines']} = {sc['pct']:.1f}%")
        if rep["sessions_known"] == 0:
            print("   SESSION STORE UNREADABLE at " + rep["sessions_dir"] + " — the "
                  "unthreaded/absent split below is meaningless; every non-trial "
                  "residual reads as session_file_absent because nothing was known "
                  "to be a session")
        denom = rep["unrecorded_sessions"]
        for kind, sids in rep["attribution"].items():
            line = f"   {kind}: {len(sids)}/{denom} unrecorded sessions"
            if kind == "trial_traffic" and sids:
                burst = rep["trial_burst"]
                line += (f" ({burst['lines']} lines, burst {_stamp(burst['first_ts'])}"
                         f" -> {_stamp(burst['last_ts'])})")
            if kind == "restart_boundary" and not rep["restarts_supplied"]:
                line += " (no --restart-at given, so nothing can be shown to be one)"
            print(line)
        print(f"   source_mix: {rep['source_mix']}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("days", nargs="*",
                    default=[datetime.now(timezone.utc).strftime("%Y-%m-%d")])
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--sessions", default="",
                    help="session store holding <id>.json, for the session-file "
                         "half of the attribution (default: ~/lloyd-data/sessions)")
    ap.add_argument("--restart-at", action="append", default=[], metavar="TIME",
                    help="a known process restart (epoch seconds or ISO-8601; "
                         "repeatable) — e.g. the `restart` rows of the automod ledger. "
                         "The only evidence that can fill restart_boundary.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    restarts = []
    for raw in args.restart_at:
        ts = _epoch(raw)
        if ts is None:
            ap.error(f"--restart-at {raw!r} is neither epoch seconds nor ISO-8601")
        restarts.append(ts)
    return report(args.store, args.days or [datetime.now(timezone.utc).strftime("%Y-%m-%d")],
                  json_out=args.json, sessions_dir=args.sessions,
                  restarts=tuple(restarts))


if __name__ == "__main__":
    raise SystemExit(main())
