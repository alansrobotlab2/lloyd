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

Spacing alone still cannot see the burst found on this box on 2026-10-11: 20
readings per surface stamped inside 22 seconds, and they were 20 processes, each a
drill spawned by the drill that started it — `drill_pool_pause` runs a real
`WorkerPool`, whose own maintenance seat spawns the drill, against a scratch queue
with no watermark to hold it off. So every reading now carries `ppid` beside `pid`,
and `read()` publishes `distinct_pids`: how many different writer pids a surface's
history holds. 20 readings on 20 pids all parented to one pid is a drill that
drilled itself; 20 on 20 pids over 20 hours is the series the drill exists to
build; a pre-#2551 reading with no `pid` counts 0 and says so rather than inventing
one. `drill_slot()` is the other half — the machine-wide bound that makes exactly
one of those processes able to record at a time.

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

from contextlib import contextmanager

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


#: The keys one reading is made of. `ppid` arrived with #2551, `pid` and
#: `invocation` with #2431, and a stored reading predating either has simply fewer
#: keys than this — which is why `_readings` fills the missing ones rather than
#: letting a reader ask a question of a dict that cannot answer it.
_READ_KEYS = ("classification", "seconds", "ok", "at", "pid", "ppid", "invocation")


def _readings(entry: dict) -> list[dict]:
    """The readings stored for one surface entry, oldest first.

    An entry written before #2153 has no `history` key and IS a single reading;
    counting it as zero would report `n: 0` and a null median for a control that
    was measured, which is the one regression a reader could not tell from a
    drill that never ran.

    Every reading comes back carrying all of `_READ_KEYS`, with the ones the writer
    did not store set to None. A file written before #2431 has no `pid`, and one
    written before #2551 has no `ppid`; both have to read as *unattributed* rather
    than raising KeyError in a status route, and `distinct_pids` has to be able to
    count them as nulls instead of deciding the question per missing key. Copies,
    so a caller that appends to a history never mutates the dict it read.
    """
    history = entry.get("history")
    kept = ([dict(r) for r in history if isinstance(r, dict)]
            if isinstance(history, list) else [])
    if not kept:
        # Pre-#2153: the entry IS the reading, and its other keys — a `history`
        # that held nothing, a stray note — are not part of a reading.
        kept = [{k: entry.get(k) for k in _READ_KEYS}]
    for r in kept:
        for k in _READ_KEYS:
            r.setdefault(k, None)
    return kept


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


def _distinct_pids(readings: list[dict]) -> int:
    """How many different processes wrote these readings.

    The clock cannot see this burst: `spaced_firings` counts clusters of stamps, and
    20 readings from 20 processes inside 22 seconds is one cluster whether the cause
    was one process looping or twenty drilling each other. The writer's pid can, and
    #2431's follow-up on 2026-10-11 needed it — the file held 20 readings per surface
    from 20 distinct pids, with `ps` showing drills orphaned onto `systemd --user`,
    which is what a drill that spawns a drill leaves behind.

    A reading written before #2431 has no pid and contributes nothing, so the count
    can legitimately come back lower than `n` — 0 beside `n: 1` says "this one is
    from before attribution", not "no one measured it". That is why it is published
    beside `n` and `spaced_firings` rather than replacing either: three numbers, three
    different questions, and a reader needs all three to tell a series from a burst
    from an unattributed legacy row.
    """
    return len({r.get("pid") for r in readings if r.get("pid")})


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


#: Env marker, set for the duration of a drill run in the process running it.
#: `workers.maintenance.maybe_run_mitigation_drill` refuses while it is present.
DRILL_RUNNING_ENV = "LLOYD_MITIGATION_DRILL_RUNNING"

#: How long a drill waits for the machine-wide slot. Zero, deliberately: a drill
#: that queued behind another would fire seconds later into the SAME round-hold gap
#: and record a second reading for one firing, which is the inflation #2431 exists to
#: catch. Standing aside is the useful answer; measuring twice is not.
DRILL_SLOT_TIMEOUT_S = 0.0


def drill_recursion_blocked() -> bool:
    """True when THIS process is a drill run in progress.

    The recursion this closes is not hypothetical: `scripts/mitigation_drill.py`'s
    `drill_pool_pause` starts a real `WorkerPool`, `WorkerPool.start()` creates the
    scheduler task whose first action is the maintenance seat, and that seat is the
    thing that spawns the drill — against a scratch temp-dir queue with no
    `last_drill_at` watermark and this box's `config.yaml` saying `enabled: true`. The
    seat's own liveness gate is `_drill_proc`, correctly process-local because
    `poll()` is the only honest answer about a child the seat itself owns, and it
    cannot see a child of a child. Chain depth is therefore unbounded without this
    marker, and on 2026-10-11 the file it produced held 20 readings per surface from
    20 distinct pids inside 22 seconds with drills orphaned onto `systemd --user`.
    """
    return os.environ.get(DRILL_RUNNING_ENV) == "1"


@contextmanager
def drill_process():
    """Mark this process as a drill, for the duration of one run (#2551).

    Restores the previous value rather than unsetting, because a drill can be started
    inside a process that is already marked (the CLI under a test harness, a candidate
    venv re-invoked by a parent) and the outer run's marker must survive the inner
    one's exit.
    """
    prior = os.environ.get(DRILL_RUNNING_ENV)
    os.environ[DRILL_RUNNING_ENV] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(DRILL_RUNNING_ENV, None)
        else:
            os.environ[DRILL_RUNNING_ENV] = prior


def _slot_file(path: Optional[Path]) -> Path:
    """The sidecar naming the pid holding the drill slot, beside the state file.

    Beside it and not in the lock directory: the slot is a question a person asks
    after the fact — `cat ~/lloyd-data/mitigation_drill.json.slot` while the box is
    mid-burst — and a dotted sibling of the file it describes is where they look. The
    flock itself lives in `atomic_io`'s lock directory; this file only carries the
    holder's pid, because an flock leaves nothing to read.
    """
    target = _path(path)
    return target.with_name(target.name + ".slot")


def drill_slot_pid(*, path: Optional[Path] = None) -> Optional[int]:
    """The pid currently holding the drill slot; None when nothing holds it."""
    try:
        raw = _slot_file(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None  # absent or half-written answers "no one is drilling"
    return int(raw) if raw.isdigit() else None


def _write_slot_pid(path: Optional[Path], pid: Optional[int]) -> None:
    slot = _slot_file(path)
    try:
        if pid is None:
            slot.unlink(missing_ok=True)
        else:
            slot.parent.mkdir(parents=True, exist_ok=True)
            slot.write_text(f"{pid}\n", encoding="utf-8")
    except Exception:
        pass  # attribution of the slot is never worth losing a reading over


@contextmanager
def drill_slot(*, path: Optional[Path] = None):
    """Yield True holding the one-live-drill slot for this machine, False without it.

    `atomic_io.commit_lock` is the machine-wide primitive — an flock on a lock file
    named off the state file's real path, so a candidate venv drilling a worktree and
    the production backend contend for the SAME slot. Nothing else on this box was
    that: the `logs/locks/mitigation_drill.py-*.lock` files present throughout the
    20-pid burst are `atomic_write_text`'s per-write flocks, which serialise one file
    replacement and cannot span the ~3 s a drill measures for.

    The holder's pid goes in a sidecar for the duration (`drill_slot_pid`), cleared on
    the way out. The lock would be released by a crash anyway; the sidecar would
    not — hence `_log_drill_exit` in the seat, which is the reader that says
    "the file has no newer reading" out loud rather than leaving a stale pid to be
    read as a live drill.
    """
    from app.atomic_io import commit_lock

    target = _path(path)
    try:
        with commit_lock(target, timeout=DRILL_SLOT_TIMEOUT_S):
            _write_slot_pid(target, os.getpid())
            try:
                yield True
            finally:
                _write_slot_pid(target, None)
    except TimeoutError:
        yield False


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
                   "pid": os.getpid(), "ppid": os.getppid(),
                   "invocation": invocation}
        history = (_readings(prior) if isinstance(prior, dict) else []) + [reading]
        merged[s["surface"]] = {**reading, "history": history[-HISTORY_CAP:]}
    state = {"surfaces": merged}
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def read(*, path: Optional[Path] = None) -> dict[str, Any]:
    """`{surface: {classification, seconds, at, median_seconds, n,
    spaced_firings, distinct_pids}}`, or `NEVER_RUN` (with an `error` when the file
    exists but cannot be read). Never raises.

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
    sends.

    `distinct_pids` is the same honesty against the burst spacing cannot see
    (#2551): 20 readings inside 22 seconds count as one firing whether they came
    from one process looping or twenty drills recursing, and only the writer's pid
    splits those. So the route now sends three counts per surface and they mean
    different things — `n` readings, `spaced_firings` firings the clock can
    evidence, `distinct_pids` processes that could have caused them, 0 for a file
    written before attribution existed."""
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
                              "spaced_firings": _spaced_firings(readings),
                              "distinct_pids": _distinct_pids(readings)}
        return out or dict(NEVER_RUN)
    except FileNotFoundError:
        return dict(NEVER_RUN)
    except Exception as e:  # unreadable is reported, never raised
        return {**NEVER_RUN, "error": f"{type(e).__name__}: {e}"}
