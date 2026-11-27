#!/usr/bin/env python3
"""How much the eval pin costs production qmd on GPU 4 (#1495): read it off the logs.

Production's daemon (:8181) logs every `/query` with its phases
(`[fts=.. embed=.. vec=.. chunk=.. rerank=..]`, UTC wall clock, no date). The
regression runner logs when each pinned-corpus check starts ("measuring X against
Y") and ends ("X: success|..."), in local time. Joining the two answers the
question #1495 is waiting on without booting anything: is a production request
slower while a pin is serving an eval on the same card, and how often does that
happen?

Only the GPU phases can be contended by a second daemon on GPU 4 — `embed` (the
query embedding) and `rerank` (the cross-encoder, which production only reaches on
the recall's fallback path, `vault_search`, and backlog dedupe since djev ranks
the recall, #1336). `vec` is the in-memory VecIndex on the CPU. Two things are
co-resident with production qmd on that card: this script's own eval pin, and
`agent-qmd-watcher`, the inotify writer that re-embeds the vault and pins the same
card the daemon's conf does (see its own supervisord conf), so the join prices the
eval pin and the watcher's embedding writes are the part of the card's load no
`/query` line ever shows. TTS pins a different card, and the vendored TTS server
that used to live on qmd's was retired in the move that brought qmd here (a372c876,
2026-11-10).

GPU 4 is the card `agent-services/supervisor/conf.d/agent-qmd-daemon.conf` sets, not
a value this script reads: it opens only the logs below. That is why the number
above is a claim about the conf rather than a constant.
`tests/test_qmd_contention.py::test_the_docstring_names_the_card_the_daemon_conf_pins`
is what keeps that claim true: it parses `CUDA_VISIBLE_DEVICES` out of that conf and
goes red if this docstring ever names another card, so the next move is a failing
test and not a stale premise (#4474).

Neither log carries what it needs to be joined on its own, so both halves date
themselves rather than being told a date (#4631). The runner's stamps are `logging`'s
asctime, so they are the machine's local wall clock, and any window long enough to
hold a quiet stretch crosses the DST boundary: one fixed offset put every November
stamp an hour early against the UTC daemon log, so an offset is now read per stamp
from the zone (`RUNNER_TZ`), never from a number. The daemon log has no date at all,
and supervisord rotates it, so each generation starts at its own creation instant:
`--daemon-log-glob`/repeated `--daemon-log` read the set, and each file is dated from
its own creation time in UTC and given back any whole day its stamps hide.

Reads those logs, writes only what `--json` names. Stdlib plus `stat(1)`.

    .venvs/lloyd/bin/python eval/qmd_contention.py --json out.json \\
          --daemon-log-glob ~/lloyd-data/logs/services/agent-qmd-daemon.err\\*
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

DAEMON_LOG = Path.home() / "lloyd-data/logs/services/agent-qmd-daemon.err"
REGRESSION_LOG = Path.home() / ".local/state/lloyd-automod/regression.log"

#: The zone the regression runner's `logging` asctime stamps are in. It is a zone,
#: not an offset, because the runner's clock springs forward: a stamp of
#: 2026-10-15 01:00 is 08:00Z and one of 2026-11-15 01:00 is 09:00Z, and the same
#: file can hold both. `tests/test_qmd_contention.py` goes red if this name and the
#: machine's own clock ever disagree (#4631).
RUNNER_TZ = ZoneInfo("America/Los_Angeles")

_TIME = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{3}) ")
_QUERY = re.compile(r"^\S+ POST /query (\d+) quer\w* \((\d+)ms\) \[([^\]]*)\]")
_REG = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} \[\w+\] [\w.-]+: (.*)")

#: A backwards jump past this is the log's day rolling over, not the clock being put
#: back: a restart does not reset a wall clock, and nothing legitimate steps it back.
_WRAP = dt.timedelta(hours=1)


def _stamp(line: str) -> dt.timedelta | None:
    m = _TIME.match(line)
    if not m:
        return None
    return dt.timedelta(hours=int(m[1]), minutes=int(m[2]), seconds=int(m[3]),
                        milliseconds=int(m[4]))


def _advances(stamps: list[dt.timedelta]) -> list[dt.timedelta]:
    """Elapsed time from each stamp to the next, as the undated log reports it: a
    jump backwards by more than an hour is read as the next day."""
    out = []
    for prev, nxt in zip(stamps, stamps[1:]):
        d = nxt - prev
        out.append(d + dt.timedelta(days=1) if d < -_WRAP else d)
    return out


def missing_days(advances: list[dt.timedelta], observed_span_s: float | None) -> Counter:
    """Whole days the stamp sequence is short of the file's own duration, and where
    they go.

    A gap in the log is only ever two things: the clock rolled over, which the caller
    has already counted, or it did not move, which is invisible. So the shortfall
    against the observed duration is whole days, and they belong to the longest
    stretches of unchanged clock — the only place a silence longer than a day can be
    (#4631).
    """
    if observed_span_s is None or not advances:
        return Counter()
    short = dt.timedelta(seconds=observed_span_s) - sum(advances, dt.timedelta())
    k = int((short + dt.timedelta(hours=12)) // dt.timedelta(days=1))
    if k < 1:
        return Counter()
    order = sorted(range(len(advances)), key=lambda i: (-advances[i], i))
    return Counter(order[j % len(order)] for j in range(k))


def parse_daemon(text: str, first_day: dt.date, *, tz=dt.timezone.utc,
                 observed_span_s: float | None = None) -> list[dict]:
    """Rows with an absolute timestamp, from one generation's text.

    The log has no date, so `first_day` is the UTC date of its first line and a clock
    that runs backwards by more than an hour is the next day (a restart does not reset
    it). That rule cannot see a quiet stretch: 30 h of no requests reads as the 6 h
    the stamp moved, which is how a 90 h generation came out 51 h and one calendar day
    short. `observed_span_s` — the file's own creation-to-last-write duration — is
    what says how long it really ran, and `missing_days` puts the difference on the
    stretches long enough to hide it.
    """
    stamps, queries = [], []
    for line in text.splitlines():
        s = _stamp(line)
        if s is None:
            continue
        stamps.append(s)
        queries.append(_QUERY.match(line))
    day0 = dt.datetime.combine(first_day, dt.time(), tzinfo=tz)
    adv = _advances(stamps)
    extra = missing_days(adv, observed_span_s)
    rows, elapsed = [], dt.timedelta()
    origin = day0 + stamps[0] if stamps else day0
    for i, (t, q) in enumerate(zip(stamps, queries)):
        if i:
            elapsed += adv[i - 1] + dt.timedelta(days=extra.get(i - 1, 0))
        if not q:
            continue
        phases = {}
        for kv in q[3].split():
            k, _, v = kv.partition("=")
            if v.isdigit():
                phases[k] = int(v)
        rows.append({"ts": origin + elapsed, "n": int(q[1]), "ms": int(q[2]), **phases})
    return rows


def birth_and_mtime(path: Path) -> tuple:
    """(creation epoch seconds, last-write epoch seconds) for one log generation.

    `os.stat()` has no `st_birthtime` on this interpreter even where the volume
    records a creation time, so that half comes from `stat(1)`'s `%W`, which prints
    0 on a filesystem that does not carry one — and 0 is reported as ``None`` so a
    caller cannot mistake the epoch for a date (#4631)."""
    r = subprocess.run(["stat", "-c", "%W %Y", str(path)], capture_output=True,
                       text=True)
    born, written = (r.stdout or "").split()[:2]
    return (int(born) or None), int(written)


def generation_span(path: Path) -> dict:
    """When one generation started and how long it ran, both read in UTC.

    The creation instant is the rotation that made the file, so the UTC date of that
    instant is the date of its first line. A *local* date will not do: rotations land
    whenever the log fills, and one at 00:34Z is 16:34 the previous day here — which
    is the 86,400 s every generation used to read at under the single `--first-day`.
    A creation time of 0 means the filesystem does not record one, and then only a
    hand-passed date can place the file, so `first_day_utc` comes back ``None``.
    """
    born, written = birth_and_mtime(path)
    utc = dt.timezone.utc
    end = dt.datetime.fromtimestamp(written, tz=utc)
    if born is None:
        return {"path": path, "first_day_utc": None, "observed_span_s": None,
                "last_utc": end}
    start = dt.datetime.fromtimestamp(born, tz=utc)
    return {"path": path, "first_day_utc": start.date(),
            "observed_span_s": max(0.0, written - born), "last_utc": end}


def daemon_generations(patterns: list[str], *, first_day: dt.date | None = None) -> list[dict]:
    """Every daemon-log generation the patterns name, oldest first, each with its own
    start date — so no generation is dated by another one's `--first-day`.

    Rotations chain (a generation's last write is the next one's creation), so the
    set's ordering is its creation order. `first_day` covers a generation whose
    filesystem records no creation time; it is a guess about one file, never about a
    whole set, so a generation with no birth time and no `first_day` is refused here
    rather than silently dated to someone else's day.
    """
    seen, paths = set(), []
    for pat in patterns:
        p = Path(pat).expanduser()
        hits = sorted(p.parent.glob(p.name)) if any(c in p.name for c in "*?[") else [p]
        for h in hits:
            if h.is_file() and h not in seen:
                seen.add(h)
                paths.append(h)
    if not paths:
        raise SystemExit(f"no daemon log matches {patterns}")
    gens = [generation_span(p) for p in paths]
    gens.sort(key=lambda g: (g["first_day_utc"] or dt.date.max, g["last_utc"]))
    unknown = [g["path"] for g in gens if g["first_day_utc"] is None]
    if unknown:
        if first_day is None:
            raise SystemExit(
                f"{', '.join(str(p) for p in unknown)} records no creation time, so "
                "nothing in the file says what day it starts: pass --first-day")
        for g in gens:
            if g["first_day_utc"] is None:
                g["first_day_utc"] = first_day
    return gens


def read_daemon_rows(patterns: list[str], *, first_day: dt.date | None = None) -> list[dict]:
    """The `/query` rows of every generation, oldest first and in clock order.

    Each generation is parsed against its own start date and its own observed
    duration, then the union is sorted: concatenated files are not one chronology,
    because a rotation can land mid-day on a later one's first day.
    """
    rows = []
    for g in daemon_generations(patterns, first_day=first_day):
        text = g["path"].read_text(errors="replace")
        rows += parse_daemon(text, g["first_day_utc"],
                             observed_span_s=g["observed_span_s"])
    rows.sort(key=lambda r: r["ts"])
    return rows


def parse_pin_windows(text: str) -> list[tuple]:
    """(start, end) of every regression check, from its 'measuring' line to its verdict.

    Each stamp is placed in `RUNNER_TZ` by its own date, so one file can hold a PDT
    October window and a PST November window and both land on the right hour of the
    UTC daemon log (#4631).
    """
    wins, cur = [], None
    for line in text.splitlines():
        m = _REG.match(line)
        if not m:
            continue
        ts = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=RUNNER_TZ)
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


def merge(wins: list[tuple]) -> list[tuple]:
    """Disjoint ascending `(start, end)` intervals covering the same instants.

    A pin window per regression check is a few hundred of them over a real window, and
    testing every request against every one of them is what made the full rotated join
    run for more than nine minutes; merged intervals and a binary search make the same
    answer a function of the rows, not of rows times windows (#4631).
    """
    out = []
    for a, b in sorted(wins):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def in_window(merged: list[tuple], ts) -> bool:
    """Whether `ts` falls in any interval of a `merge`d set — the same answer a scan of
    the raw windows gives, in log time rather than linearly."""
    lo, hi = 0, len(merged)
    while lo < hi:
        mid = (lo + hi) // 2
        if merged[mid][1] < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(merged) and merged[lo][0] <= ts


def analyse(rows: list[dict], wins: list[tuple]) -> dict:
    if not rows:
        return {"error": "no /query rows"}
    first, last = rows[0]["ts"], rows[-1]["ts"]
    wins = [(a, b) for a, b in wins if b >= first and a <= last]
    starts = [a for a, _ in wins]
    covered = merge(list(w) for w in wins)

    span = (last - first).total_seconds()
    pin_s = sum((min(b, last) - max(a, first)).total_seconds() for a, b in covered)
    out = {"query_rows": len(rows),
           "log_span_utc": [first.isoformat(), last.isoformat()],
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
            s = [r for r in sel if in_window(covered, r["ts"]) == inside]
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
    ap.add_argument("--daemon-log", action="append", default=None,
                    help="one daemon log generation; repeat it for each rotated one. "
                         f"Default: {DAEMON_LOG}")
    ap.add_argument("--daemon-log-glob", action="append", default=None,
                    help="a pattern naming the rotated generations together, e.g. "
                         "'~/lloyd-data/logs/services/agent-qmd-daemon.err*'")
    ap.add_argument("--regression-log", default=str(REGRESSION_LOG))
    ap.add_argument("--first-day", default=None,
                    help="UTC date for a generation whose filesystem records no creation "
                         "time (stat's %%W is 0 there); generations that do record one "
                         "are dated from it and ignore this")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    patterns = a.daemon_log_glob or []
    patterns += a.daemon_log or ([str(DAEMON_LOG)] if not a.daemon_log_glob else [])
    first_day = dt.date.fromisoformat(a.first_day) if a.first_day else None
    rows = read_daemon_rows(patterns, first_day=first_day)
    wins = parse_pin_windows(Path(a.regression_log).read_text(errors="replace"))
    out = analyse(rows, wins)
    out["daemon_logs"] = [str(p) for p in patterns]
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
