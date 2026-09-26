#!/usr/bin/env python3
"""How much the eval pin costs production qmd on GPU 0 (#1495): read it off the logs.

Production's daemon (:8181) logs every `/query` with its phases
(`[fts=.. embed=.. vec=.. chunk=.. rerank=..]`, UTC wall clock, no date). The
regression runner logs when each pinned-corpus check starts ("measuring X against
Y") and ends ("X: success|..."), in local time. Joining the two answers the
question #1495 is waiting on without booting anything: is a production request
slower while a pin is serving an eval on the same card, and how often does that
happen?

Only the GPU phases can be contended by a second daemon on GPU 0 — `embed` (the
query embedding) and `rerank` (the cross-encoder, which production only reaches on
the recall's fallback path, `vault_search`, and backlog dedupe since djev ranks
the recall, #1336). `vec` is the in-memory VecIndex on the CPU. The TTS server
on the same card leaves no request log to join, so it is not separated here.

Reads two files, writes nothing. Stdlib only.

    .venvs/lloyd/bin/python eval/qmd_contention.py [--json out.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

DAEMON_LOG = Path.home() / "lloyd-data/logs/services/agent-qmd-daemon.err"
REGRESSION_LOG = Path.home() / ".local/state/lloyd-automod/regression.log"

_TIME = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{3}) ")
_QUERY = re.compile(r"^\S+ POST /query (\d+) quer\w* \((\d+)ms\) \[([^\]]*)\]")
_REG = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} \[\w+\] [\w.-]+: (.*)")


def parse_daemon(text: str, first_day: dt.date, *, tz=dt.timezone.utc) -> list[dict]:
    """Rows with an absolute timestamp. The log has no date, so a clock that runs
    backwards by more than an hour is the next day (a restart does not reset it)."""
    day = dt.datetime.combine(first_day, dt.time(), tzinfo=tz)
    prev = None
    rows = []
    for line in text.splitlines():
        m = _TIME.match(line)
        if not m:
            continue
        t = dt.timedelta(hours=int(m[1]), minutes=int(m[2]), seconds=int(m[3]),
                         milliseconds=int(m[4]))
        if prev is not None and t < prev - dt.timedelta(hours=1):
            day += dt.timedelta(days=1)
        prev = t
        q = _QUERY.match(line)
        if not q:
            continue
        phases = {}
        for kv in q[3].split():
            k, _, v = kv.partition("=")
            if v.isdigit():
                phases[k] = int(v)
        rows.append({"ts": day + t, "n": int(q[1]), "ms": int(q[2]), **phases})
    return rows


def parse_pin_windows(text: str, *, local_offset_h: int = -7) -> list[tuple]:
    """(start, end) of every regression check, from its 'measuring' line to its verdict."""
    tz = dt.timezone(dt.timedelta(hours=local_offset_h))
    wins, cur = [], None
    for line in text.splitlines():
        m = _REG.match(line)
        if not m:
            continue
        ts = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
        msg = m[2]
        if msg.startswith("measuring "):
            cur = ts
        elif cur is not None and re.match(r"^[0-9a-f]{8}: ", msg):
            wins.append((cur, ts))
            cur = None
    return wins


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def summ(xs) -> dict:
    xs = list(xs)
    return {"n": len(xs), "p50": _q(xs, .5), "p95": _q(xs, .95),
            "mean": round(statistics.fmean(xs), 1) if xs else None}


def analyse(rows: list[dict], wins: list[tuple]) -> dict:
    if not rows:
        return {"error": "no /query rows"}
    first, last = rows[0]["ts"], rows[-1]["ts"]
    wins = [(a, b) for a, b in wins if b >= first and a <= last]
    starts = [a for a, _ in wins]

    def in_pin(ts):
        # windows are sorted; a linear scan is fine at a few hundred windows
        return any(a <= ts <= b for a, b in wins)

    span = (last - first).total_seconds()
    pin_s = sum((min(b, last) - max(a, first)).total_seconds() for a, b in wins)
    out = {"log_span_utc": [first.isoformat(), last.isoformat()],
           "span_h": round(span / 3600, 1), "pin_windows": len(wins),
           "pin_window_s": summ([(b - a).total_seconds() for a, b in wins]),
           "pin_fraction_of_time": round(pin_s / span, 4) if span else None,
           "first_pin": starts[0].isoformat() if starts else None}
    classes = {
        "recall_doc_leg": lambda r: "embed" in r and "rerank" not in r,
        "rerank": lambda r: "rerank" in r,
        "lex_only": lambda r: "embed" not in r and "rerank" not in r,
    }
    for name, pred in classes.items():
        sel = [r for r in rows if pred(r)]
        for inside in (False, True):
            s = [r for r in sel if in_pin(r["ts"]) == inside]
            block = {"total_ms": summ(r["ms"] for r in s)}
            for ph in ("embed", "vec", "rerank"):
                vals = [r[ph] for r in s if ph in r]
                if vals:
                    block[f"{ph}_ms"] = summ(vals)
            out[f"{name}__{'pin' if inside else 'no_pin'}"] = block
    per_min = Counter(r["ts"].replace(second=0, microsecond=0) for r in rows)
    out["minutes_with_any_request"] = len(per_min)
    out["minutes_with_10plus_requests"] = sum(1 for c in per_min.values() if c >= 10)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--daemon-log", default=str(DAEMON_LOG))
    ap.add_argument("--regression-log", default=str(REGRESSION_LOG))
    ap.add_argument("--first-day", default=None,
                    help="UTC date of the daemon log's first line (default: its mtime minus "
                         "the days the clock wrapped is unknowable, so pass it)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    first_day = dt.date.fromisoformat(a.first_day) if a.first_day else dt.date(2026, 9, 22)
    rows = parse_daemon(Path(a.daemon_log).read_text(errors="replace"), first_day)
    wins = parse_pin_windows(Path(a.regression_log).read_text(errors="replace"))
    out = analyse(rows, wins)
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
