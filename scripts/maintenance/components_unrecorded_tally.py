#!/usr/bin/env python3
"""Per-send-site unrecorded tally of the request manifest, for #1879's residual.

#1879's acceptance is a claim about traffic over a day, and the check the item names
(`cm_scan5.py`, `cm_1879_check.py`) lived in /tmp — a directory that is wiped, so the
next pass had to rewrite the measurement and could quietly write a different one. This
is that measurement, in the tree, with the two halves the item needs joined into one
run:

  * the per-`send_site` unrecorded tally, with the DENOMINATOR printed beside every
    verdict (a percentage over zero lines is not a check);
  * the `turn_start` cross-check that attributes the residual into three buckets that
    are distinguished by DIFFERENT evidence, because the manifest alone cannot tell
    them apart: a session with unrecorded lines, no `turn_start` line, and a session
    file in `~/lloyd-data/sessions` is an **unthreaded send site** — the thing #1879
    fixes; one whose id has no session file at all is a **restart boundary**, a prompt
    built in a process whose registry is gone; and one that emitted `turn_start`
    earlier in the day and reads `unrecorded` later is a **registry eviction**, which
    `86aef7e5` made possible when it raised the registry bound and counted its drops.

The boundary test is the session store and not the manifest: every unrecorded line
carries its own `session_id`, so asking the manifest whether a session "appears in the
day" answers yes by construction and the bucket could never fill. #1879's own triage ran
the check that way — all 29 unrecorded ids on 2026-09-30 exist as session files, which
is what let it say the residual is unthreaded-site traffic and not restart noise.

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
KINDS = ("unthreaded_send_site", "restart_boundary", "registry_eviction")


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
    every line carries one, so a restart boundary — a send whose registry died with
    the process that built its prompt — has to be found by asking the store.
    """
    try:
        return {p.name[:-5] for p in os.scandir(sessions_dir)
                if p.name.endswith(".json")}
    except OSError:
        return set()


def tally(rows: list[dict], *, session_ids: set[str]) -> dict:
    """The report. `denominator == 0` means there is nothing to rule on.

    `session_ids` is `known_sessions()` of the session store; pass an empty set to
    say "the store could not be read", which puts every residual session in the
    `restart_boundary` bucket only after the `turn_start` test, never silently.
    """
    by_site_total: collections.Counter = collections.Counter()
    by_site_unrec: collections.Counter = collections.Counter()
    turn_start_sessions: set[str] = set()
    unrec_lines: collections.Counter = collections.Counter()
    for row in rows:
        site = str(row.get("send_site") or "?")
        captured = str(row.get("components_captured") or "")
        sid = str(row.get("session_id") or "")
        by_site_total[site] += 1
        if captured == "unrecorded":
            by_site_unrec[site] += 1
            if sid:
                unrec_lines[sid] += 1
        elif captured == "turn_start":
            turn_start_sessions.add(sid)
    sc_total = by_site_total.get(STREAM_CHAT, 0)
    sc_unrec = by_site_unrec.get(STREAM_CHAT, 0)
    attribution: dict[str, list[str]] = {k: [] for k in KINDS}
    for sid in unrec_lines:
        if sid in turn_start_sessions:
            attribution["registry_eviction"].append(sid)
        elif sid in session_ids:
            attribution["unthreaded_send_site"].append(sid)
        else:
            attribution["restart_boundary"].append(sid)
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
        "source_mix": dict(collections.Counter(
            (sid.split("_")[2] if len(sid.split("_")) >= 4 else "chat")
            for sid in unrec_lines)),
    }


def default_sessions_dir(store: str) -> str:
    """`~/lloyd-data/sessions`, located from the store root's home directory.

    Derived, not hardcoded, so a check run against a scratch store can be pointed at
    a scratch session store — and so a moved root cannot quietly turn every residual
    session into a `restart_boundary`, which is what an unreadable session dir would
    mean if the two paths were assumed to move together.
    """
    marker = os.sep + ".local" + os.sep + "state" + os.sep
    root = os.path.abspath(store)
    home = root.split(marker)[0] if marker in root else os.path.expanduser("~")
    return os.path.join(home, "lloyd-data", "sessions")


def report(store: str, days: list[str], *, json_out: bool,
           sessions_dir: str = "") -> int:
    rc = 0
    out = []
    sessions_dir = sessions_dir or default_sessions_dir(store)
    session_ids = known_sessions(sessions_dir)
    for day in days:
        rows = day_rows(store, day)
        rep = tally(rows, session_ids=session_ids)
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
                  "unthreaded/restart split below is meaningless; every residual reads "
                  "as a restart boundary because nothing was known to be a session")
        for kind, sids in rep["attribution"].items():
            print(f"   {kind}: {len(sids)}")
        print(f"   source_mix: {rep['source_mix']}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("days", nargs="*",
                    default=[datetime.now(timezone.utc).strftime("%Y-%m-%d")])
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--sessions", default="",
                    help="session store holding <id>.json, for the restart-boundary "
                         "half of the attribution (default: ~/lloyd-data/sessions)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return report(args.store, args.days or [datetime.now(timezone.utc).strftime("%Y-%m-%d")],
                  json_out=args.json, sessions_dir=args.sessions)


if __name__ == "__main__":
    raise SystemExit(main())
