"""The mitigation drill's readings per surface, on disk (#703, history #2153).

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
stop-time and no aggregate over the series a daily drill produces was computable
from anything on disk. The cap is what makes keeping the readings affordable for
a job that runs daily: 20 readings per surface is the whole file, and the oldest
one is the least interesting.

The reader never raises: a status route is most useful when something is wrong,
so a missing or unreadable file is reported as a never-run marker.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import paths
from app.atomic_io import atomic_write_text

NEVER_RUN = {"state": "never-run"}

#: Readings kept per surface. The drill is scheduled daily, so 20 is roughly
#: three weeks of medians and the file stays that size forever.
HISTORY_CAP = 20


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
             for k in ("classification", "seconds", "ok", "at")}]


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
           at: Optional[str] = None) -> dict[str, Any]:
    """Merge each drilled surface's result into the state file, atomically.

    The merge has two halves: the surface's top-level keys keep holding THIS
    reading, which is what every existing reader of `classification`/`seconds`/
    `at` was already reading, and the reading is appended to the surface's
    bounded history. An unreadable file starts the surface again — the readings
    it could not be read from are not recoverable, and a partial history is
    still a history."""
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
                   "ok": s.get("ok"), "at": at}
        history = (_readings(prior) if isinstance(prior, dict) else []) + [reading]
        merged[s["surface"]] = {**reading, "history": history[-HISTORY_CAP:]}
    state = {"surfaces": merged}
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def read(*, path: Optional[Path] = None) -> dict[str, Any]:
    """`{surface: {classification, seconds, at, median_seconds, n}}`, or
    `NEVER_RUN` (with an `error` when the file exists but cannot be read). Never
    raises.

    `seconds`/`at` are the newest reading; `median_seconds`/`n` are the aggregate
    over the surface's stored history. Both are published because they answer
    different questions: the newest reading is "what did the last drill see",
    the median is "what does this control usually take", and one slow drill
    moves only the first. The history itself stays in the file — 20 readings per
    surface on a route that is read as fast as a dashboard polls is not a
    payload, and `tests/test_mitigation_drill.py` pins that it is not sent."""
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
                              "n": len(readings)}
        return out or dict(NEVER_RUN)
    except FileNotFoundError:
        return dict(NEVER_RUN)
    except Exception as e:  # unreadable is reported, never raised
        return {**NEVER_RUN, "error": f"{type(e).__name__}: {e}"}
