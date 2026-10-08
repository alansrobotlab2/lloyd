"""The mitigation drill's readings per surface, on disk (#703, history #2153, #2333).

`scripts/mitigation_drill.py` measures what each stop control actually stops;
this file keeps those measurements so `GET /api/workers/status` can say which
controls are in-flight stops, which are dispatch-only, and how long a stop took
— without running anything. A surface is merged in, not replaced wholesale, so a
drill that measured one surface keeps the other's readings.

Each reading is also APPENDED to that surface's history, capped at
`HISTORY_CAP` entries with the oldest dropped, and `read()` publishes the median
of that history (`median_seconds`, beside its count `n`) next to the latest
reading (`classification`, `seconds`, `at`). Before #2153 the file held only the
newest reading, so the status page quoted one drill's stop-time as the control's
stop-time and no aggregate over a series of drills was computable from anything
on disk. The cap is what makes keeping a series affordable: `HISTORY_CAP`
readings per surface is the whole file, and the oldest one is the least
interesting. What the number is sized *for* has moved since it was written — the
comment on the constant carries the window it now means.

A count of readings is not a count of drillings, and #2431 is the incident that
says so: the file found on 2026-10-08 held `n: 20` per surface with every
reading stamped inside ten seconds — one process looping the drill, against two
logged seat spawns — and no published field could tell a burst from three days
of hourly firings. So two things are recorded and published that were not
before. Every reading carries the pid of the process that wrote it, and the
command line when that process is the CLI, which makes "20 readings, one
process" readable off the bytes themselves rather than off log archaeology. And
`read()` publishes `spaced_firings` beside `n`: how many separate firings the
stored readings could have come from at `SPACING_GAP_S` apart, which is 1 for a
ten-second burst and rises only as readings land further and further apart.

The reader never raises: a status route is most useful when something is wrong,
so a missing or unreadable file is reported as a never-run marker.
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import paths
from app.atomic_io import atomic_write_text

NEVER_RUN = {"state": "never-run"}

#: Readings kept per surface — which is to say, the window `median_seconds` is
#: the median OF. The number is unchanged at 20; what moved is what 20 means.
#:
#: #703 sized it for a drill that fired once a day, and the comment said so: 20
#: readings was roughly three weeks of medians. #2333 moved the trigger onto the
#: pool's maintenance seat, which spawns one at most once an hour
#: (`workers.maintenance.MITIGATION_DRILL_DEFAULTS["interval_seconds"]`), because
#: the once-a-day caller — a scheduled task — was refused by the round hold on
#: every day it tried and wrote no reading at all. So 20 readings is now about
#: 20 HOURS of firings: `median_seconds` answers "what did this control do in its
#: last day of firings", which is the question a stop-time is asked, and the
#: three-weeks window is gone with the trigger that produced it.
#:
#: The number itself must not be raised without the prose that quotes it: the
#: mitigation-drill skill tells its runner to "keep the most recent 20 readings",
#: and `tests/fixtures/mitigation_drill/SKILL.md` plus
#: `tests/test_mitigation_drill_task.py` are the shipped instruction and its
#: enforcement — a constant moved out from under a sentence that quotes it is how
#: a documented number stops being true (#1217). Raise the cap there too, or not
#: here.
HISTORY_CAP = 20

#: Two readings closer together than this count as ONE firing, not two.
#:
#: `n` counts readings, and a reading is one call to `record()` — which is not
#: the same unit as a drill firing. The file found on 2026-10-08 held 20 readings
#: per surface stamped 17:17:34→17:17:44Z: ten seconds of a single process
#: looping the drill, against exactly two logged seat spawns in that hour, and
#: `GET /api/workers/status` reported `n: 20` for both, which reads to its owed
#: check as twenty hourly firings (#2431). `spaced_firings` is the published
#: answer to "how many firings could these readings have come from", and this is
#: the unit it measures in.
#:
#: Half the seat's hourly interval
#: (`workers.maintenance.MITIGATION_DRILL_DEFAULTS["interval_seconds"]`, 3600 s,
#: the same cross-file arithmetic `tests/test_workers_maintenance.py` already
#: does to `HISTORY_CAP`): the count has to stay well UNDER the real gap or it
#: merges genuine hourly firings, and a burst has to run for half an hour to be
#: miscounted as two. One real case does merge, and it is the safe one: a drill
#: that spent nearly its whole `--wait-free-window` waiting records seconds after
#: the previous firing's reading, and the pair counts as one. That error says
#: "these might be one firing" — it never inflates the series — and the pid every
#: reading now carries is what splits it again off the bytes.
SPACING_GAP_S = 1800.0


def _path(path: Optional[Path]) -> Path:
    # Read at call time, so a test that points `paths.MITIGATION_DRILL_STATE`
    # elsewhere is honoured.
    return Path(path) if path is not None else paths.MITIGATION_DRILL_STATE


def _readings(entry: dict) -> list[dict]:
    """The readings stored for one surface entry, oldest first.

    An entry written before #2153 has no `history` key and IS a single reading;
    counting it as zero would report `n: 0` and a null median for a control that
    was measured, which is the one regression a reader could not tell from a
    drill that never ran.
    """
    history = entry.get("history")
    if isinstance(history, list):
        kept = [r for r in history if isinstance(r, dict)]
        if kept:
            return kept
    return [{k: entry.get(k)
             for k in ("classification", "seconds", "ok", "at", "pid",
                       "invocation")}]


def _stamp(value: Any) -> Optional[datetime]:
    """A reading's `at` as an aware datetime, or None when it will not parse.

    `record()` writes whole seconds with an offset; the pre-#2153 file and the
    test fixtures above both hold bare words ("later", "old"), which are
    readings with no usable time rather than a reason to raise — a status route
    that throws because one stamp is odd is a route that goes silent on the
    control it was asked about.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _spaced_firings(readings: list[dict]) -> int:
    """How many separate drillings the stored readings could have come from.

    Sort the stamps and count the clusters `SPACING_GAP_S` apart: the 20 readings
    inside ten seconds that started #2431 count as 1, five readings an hour apart
    count as 5, and the number only rises as readings land further apart. It
    never exceeds `n`, and it is not a substitute for it — `n` is what was
    measured, this is what may have caused it.

    Readings whose `at` will not parse contribute nothing, so a series of those
    reports the one firing the file at least evidences rather than 0: `n: 5,
    spaced_firings: 0` would be read as "measured five times, caused by nothing",
    when the honest reading is "five readings, no usable clock".
    """
    stamps = sorted(s for s in (_stamp(r.get("at")) for r in readings) if s)
    if not stamps:
        return 1 if readings else 0
    firings = 1
    for prev, nxt in zip(stamps, stamps[1:]):
        if (nxt - prev).total_seconds() >= SPACING_GAP_S:
            firings += 1
    return firings


def newest_reading_at(*, path: Optional[Path] = None) -> Optional[datetime]:
    """The newest reading time in the file across every surface; None if it holds
    none the reader can use — no file, an unreadable one, or stamps that parse.

    The maintenance seat asks this to tell a drill that recorded something from
    one that fired and wrote nothing, which is the difference between an hour
    with no measurement in it and an hour that left no trace of having tried
    (#2431 clause 4). Deliberately not a `read()`: that publishes a per-surface
    aggregate for a dashboard, and the seat wants the raw maximum over the whole
    file, including a surface it has never heard of.
    """
    try:
        surfaces = json.loads(_path(path).read_text(encoding="utf-8")).get("surfaces") or {}
        stamps = [s for s in (_stamp(entry.get("at"))
                              for entry in surfaces.values()
                              if isinstance(entry, dict)) if s]
    except Exception:
        return None  # unreadable answers "no reading", exactly as read() answers never-run
    return max(stamps) if stamps else None


def _median_seconds(readings: list[dict]) -> Optional[float]:
    """Median of the readings that timed a stop; None if none of them did.

    A dispatch-only control (`pool_pause`) records `seconds: None` by design — a
    pause stops claims, not the run in flight — so its median is null while `n`
    still counts its readings. Null is the honest answer there; zero would be a
    measurement nobody took.
    """
    timed = [r["seconds"] for r in readings
             if isinstance(r.get("seconds"), (int, float))
             and not isinstance(r.get("seconds"), bool)]
    return statistics.median(timed) if timed else None


def record(surfaces: list[dict], *, path: Optional[Path] = None,
           at: Optional[str] = None,
           invocation: Optional[str] = None) -> dict[str, Any]:
    """Merge each drilled surface's result into the state file, atomically.

    The merge has two halves: the surface's top-level keys keep holding THIS
    reading, which is what every existing reader of `classification`/`seconds`/
    `at` was already reading, and the reading is appended to the surface's
    bounded history. An unreadable file starts the surface again — the readings
    it could not be read from are not recoverable, and a partial history is
    still a history.

    Each reading is stamped with the writing process's pid, and with
    `invocation` — the command line — when the caller is the CLI. Attribution is
    the whole reason: `scripts/mitigation_drill.py::main` is one
    `asyncio.run(run(...))` per process, so twenty readings spread over ten
    seconds cannot be twenty interpreter boots, and on 2026-10-08 the only way to
    learn that was to say so in prose and cite log lines. Now the bytes carry it:
    one pid repeated twenty times, or a `null` invocation beside the seat's
    `… scripts.mitigation_drill.py --wait-free-window 3600`, is the finding. An
    in-process caller that passes nothing gets `invocation: null`, which is also
    an answer — it says a reading was written by something that was not the CLI.
    """
    target = _path(path)
    at = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        merged = dict(json.loads(target.read_text(encoding="utf-8")).get("surfaces") or {})
    except Exception:
        merged = {}  # absent or unreadable: this drill's results start it again
    for s in surfaces:
        prior = merged.get(s["surface"])
        reading = {"classification": s.get("classification"),
                   "seconds": s.get("seconds"),
                   "ok": s.get("ok"), "at": at,
                   "pid": os.getpid(), "invocation": invocation}
        history = (_readings(prior) if isinstance(prior, dict) else []) + [reading]
        merged[s["surface"]] = {**reading, "history": history[-HISTORY_CAP:]}
    state = {"surfaces": merged}
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def read(*, path: Optional[Path] = None) -> dict[str, Any]:
    """`{surface: {classification, seconds, at, median_seconds, n,
    spaced_firings}}`, or `NEVER_RUN` (with an `error` when the file exists but
    cannot be read). Never raises.

    `seconds`/`at` are the newest reading; `median_seconds`/`n` are the aggregate
    over the surface's stored history. Both are published because they answer
    different questions: the newest reading is "what did the last drill see",
    the median is "what does this control usually take", and one slow drill
    moves only the first. The history itself stays in the file — 20 readings per
    surface on a route that is read as fast as a dashboard polls is not a
    payload, and `tests/test_mitigation_drill.py` pins that it is not sent.

    `spaced_firings` is what keeps `n` honest (#2431): the count of readings a
    burst of them can satisfy, so a caller asking "has this control been drilled
    five times" reads the firings it can actually evidence and not the 20 rows a
    single looping process left behind. It is a count of clusters at
    `SPACING_GAP_S`, so it costs one pass over stamps already in hand — no second
    read of the file, and no new key beside the surface entries the route already
    sends."""
    try:
        surfaces = json.loads(_path(path).read_text(encoding="utf-8")).get("surfaces")
        out: dict[str, Any] = {}
        for name, entry in (surfaces or {}).items():
            if not isinstance(entry, dict):
                continue
            readings = _readings(entry)
            out[str(name)] = {"classification": entry.get("classification"),
                              "seconds": entry.get("seconds"),
                              "at": entry.get("at"),
                              "median_seconds": _median_seconds(readings),
                              "n": len(readings),
                              "spaced_firings": _spaced_firings(readings)}
        return out or dict(NEVER_RUN)
    except FileNotFoundError:
        return dict(NEVER_RUN)
    except Exception as e:  # unreadable is reported, never raised
        return {**NEVER_RUN, "error": f"{type(e).__name__}: {e}"}
