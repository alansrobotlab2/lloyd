"""Fire the stop controls at a synthetic run and measure what each one stops (#703).

A control that is wired and never fired reads as working until the day it is
needed. This drill starts a synthetic run, fires a control at it, and records
what happened, classified by what was MEASURED rather than by what the control
is called:

- ``in-flight``     — the running run reached a terminal state; ``seconds`` is
                      fire → terminal.
- ``dispatch-only`` — nothing new was claimed, but the running run carried on.
- ``no-op``         — the control changed nothing observable. A failure.

Two surfaces, both in-process and offline — no engine, no live pool, no live
session is touched:

``session_cancel``
    The real per-session turn queue (`app.sessions_io.enqueue_turn`) runs a
    synthetic turn that polls the queue's ``cancel_event`` at each decision
    point, the way the harness loop does between iterations, and the control
    fired is the real route handler behind ``POST /api/sessions/{id}/cancel``.
    What this does NOT measure is how long the real agent loop takes to reach
    its next decision point — an engine stream in progress is not one.

``pool_pause``
    A real `WorkerPool` over a scratch `WorkQueue` runs a synthetic job; the
    pool is paused and a second item enqueued. Pause is **dispatch-only by
    design**: the landing path pauses the pool and then *waits for* the jobs in
    flight (`promote.wait_idle`, architecture/workers.md §7), so a pause that
    cancelled them would kill the very work the drain exists to protect. The
    drill pins that it stops claims and does not stop the run, so nobody reads
    a pause as a kill switch. The in-flight stop is ``session_cancel``.

Not drilled here: the Inner Voice cancel sets the same kind of turn
``cancel_event`` from inside the backend, and the guardian rollback already has
its canary drill (`scripts/automod/rehearse.py`, gate rung 7).

The drill refuses while a self-mod round holds the live pool (read from
``/api/workers/status``'s ``round_hold``, the pool's own state), so it never
adds load while a round or the landing behind it wants the box. On this box that
hold is engaged most of the day — 84%, 94% and 73% of the minutes on
2026-10-04/05/06 — so a caller that fires once a day is refused most days it
tries, and the reading series never accumulates. ``--wait-free-window SECONDS``
is for that caller: re-read the status every ``FREE_WINDOW_POLL_S`` seconds and
fire the moment the hold releases, instead of exiting 2 on the first look (#2333).

The hold is consulted ONCE, before any surface runs, and never re-checked after
the drill has started. That is not an oversight: of the 61 free gaps the hold
left across 2026-10-03..06, 31 were under five minutes, and two of the three that
opened after a scheduled 13:02Z fire were 0.7 and 0.6 minutes — shorter than the
drill itself takes. A caller that confirmed the hold was still released after the
surfaces began would throw away most of the windows it waited for, and a hold
that re-engages mid-drill is harmless here because both surfaces are in-process
and offline and the state-file merge is additive.

    .venvs/lloyd/bin/python -m scripts.mitigation_drill        # JSON, exit 1 on a failed surface
    .venvs/lloyd/bin/python -m scripts.mitigation_drill --wait-free-window 3600

The pool's maintenance seat launches the second form at most once an hour
(``workers.maintenance.mitigation_drill``, ``workers/maintenance.py``), which is
what lets the series grow without spending engine tokens on the attempt.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

SYNTHETIC_SOURCE = "mitigation-drill"
DECISION_POINT_S = 0.02
STATUS_URL = "http://127.0.0.1:8080/api/workers/status"

#: How often `--wait-free-window` re-reads the pool status while it waits.
#: Sixty-one free gaps across 2026-10-03..06, 31 of them under five minutes: a
#: coarser poll sits through the short ones, a finer one is a tighter loop on a
#: route the dashboard is already polling.
FREE_WINDOW_POLL_S = 15.0

#: Test seams. `main` reaches the nap and the clock through these two names, so
#: a test can run a whole wait window — an hour of them — without waiting.
_sleep = time.sleep
_monotonic = time.monotonic


def refusal(status: Optional[dict]) -> Optional[str]:
    """Why the drill must not run now, from a `/api/workers/status` payload.

    The round hold is the pool's own "a self-mod round is in flight" state; it
    is read, not re-derived. None means the backend did not answer, which for
    an in-process drill that touches nothing live is not a reason to refuse.
    """
    if not status:
        return None
    hold = ((status.get("pool") or {}).get("round_hold") or {})
    if hold.get("engaged"):
        since = hold.get("engaged_since") or "?"
        return f"a self-mod round holds the worker pool (round_hold engaged since {since})"
    return None


def live_status(url: str = STATUS_URL, timeout: float = 3.0) -> Optional[dict]:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def wait_for_free_window(url: str = STATUS_URL, window_s: float = 0.0, *,
                         poll_s: float = FREE_WINDOW_POLL_S,
                         status_fn: Optional[Callable[[str], Optional[dict]]] = None,
                         sleep: Optional[Callable[[float], Any]] = None,
                         clock: Optional[Callable[[], float]] = None,
                         ) -> tuple[Optional[dict], Optional[str]]:
    """Wait up to `window_s` for the round hold to release. (#2333)

    Returns `(status, None)` at the first read `refusal()` calls clean, and
    `(last_status, reason)` when the window expires with the hold still engaged —
    `reason` being the very sentence `refusal()` returned, so the caller's exit-2
    text does not change shape.

    The status payload returned on the clean read is the one the caller hands to
    `run()`: the hold is judged off that one read and never re-read, which is how
    a window shorter than the drill's own runtime still produces a reading.

    `status_fn`, `sleep` and `clock` default to the module's own readers — looked
    up at call time, so patching `live_status`, `_sleep` or `_monotonic` on this
    module is what a test does rather than an alternative to it.
    """
    read = status_fn or live_status
    nap = sleep or _sleep
    now = clock or _monotonic

    deadline = now() + window_s
    while True:
        status = read(url)
        reason = refusal(status)
        if reason is None:
            return status, None
        remaining = deadline - now()
        if remaining <= 0:
            return status, reason
        nap(min(poll_s, remaining))


def _result(surface: str, classification: str, seconds: Optional[float],
            detail: str) -> dict[str, Any]:
    return {"surface": surface, "classification": classification,
            "seconds": None if seconds is None else round(seconds, 3),
            "ok": classification != "no-op", "detail": detail}


# ── session_cancel ──────────────────────────────────────────────────────────

async def _route_cancel(session_id: str) -> None:
    """The handler behind POST /api/sessions/{id}/cancel, called directly."""
    from app.routers.sessions import cancel_session
    await cancel_session(session_id, SimpleNamespace(query_params={}))


async def drill_session_cancel(
        fire: Optional[Callable[[str], Awaitable[None]]] = None,
        *, timeout_s: float = 5.0) -> dict[str, Any]:
    from app.sessions_io import (SessionTurn, _session_queues, enqueue_turn,
                                 get_cancel_event)

    fire = fire or _route_cancel
    session_id = f"mitigation-drill-{uuid.uuid4().hex[:8]}"
    started = asyncio.Event()
    stopped: dict[str, float] = {}

    async def consumer() -> None:
        # A minimal stand-in for `messages._session_consumer`: pop the turn,
        # install a fresh cancel_event, run it, clear the slot.
        q = _session_queues[session_id]
        async with q.lock:
            turn = q.pending_user.popleft()
            q.current = turn
            q.cancel_event = asyncio.Event()
        started.set()
        deadline = time.monotonic() + timeout_s + 1.0
        try:
            while time.monotonic() < deadline:
                if q.cancel_event.is_set():
                    stopped["at"] = time.monotonic()
                    break
                await asyncio.sleep(DECISION_POINT_S)
        finally:
            async with q.lock:
                q.current = None
                q.consumer_task = None
            turn.done.set()

    turn = SessionTurn(turn_id=uuid.uuid4().hex[:8], source="user",
                       payload={"drill": True}, enqueued_at=datetime.now())
    try:
        await enqueue_turn(session_id, turn, consumer_factory=consumer)
        await asyncio.wait_for(started.wait(), timeout=2.0)
        if get_cancel_event(session_id) is None:
            return _result("session_cancel", "no-op", None,
                           "the running turn exposed no cancel_event to fire")
        fired_at = time.monotonic()
        await fire(session_id)
        try:
            await asyncio.wait_for(turn.done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass
    finally:
        q = _session_queues.pop(session_id, None)
        if q is not None and q.consumer_task is not None:
            q.consumer_task.cancel()
    if "at" in stopped:
        return _result("session_cancel", "in-flight", stopped["at"] - fired_at,
                       "the running turn saw cancel_event at its next decision point")
    return _result("session_cancel", "no-op", None,
                   f"the running turn did not stop within {timeout_s}s of the cancel")


# ── pool_pause ──────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _synthetic_registry(source: Any):
    """Point the pool at the synthetic source only, for the drill's duration."""
    import workers.sources as sources
    saved = (sources.SOURCE_REGISTRY, sources.get_sources_config)
    sources.SOURCE_REGISTRY = {SYNTHETIC_SOURCE: source}
    sources.get_sources_config = lambda: {SYNTHETIC_SOURCE: {"max_duration_seconds": 60}}
    try:
        yield
    finally:
        sources.SOURCE_REGISTRY, sources.get_sources_config = saved


async def drill_pool_pause(pause: Optional[Callable[[Any], None]] = None,
                           *, probe_s: float = 1.0) -> dict[str, Any]:
    from workers.pool import WorkerPool
    from workers.queue import WorkQueue

    pause = pause or (lambda pool: pool.pause(True))
    release = asyncio.Event()
    ran: list[int] = []
    finished: dict[int, float] = {}

    async def execute(item):
        ran.append(item.id)
        while not release.is_set():
            await asyncio.sleep(DECISION_POINT_S)
        finished[item.id] = time.monotonic()
        return {"summary": "synthetic drill job"}

    source = SimpleNamespace(NAME=SYNTHETIC_SOURCE, execute=execute)
    with tempfile.TemporaryDirectory() as tmp, _synthetic_registry(source):
        q = WorkQueue(Path(tmp) / "drill.db")
        pool = WorkerPool(q, slots=2, poll_idle_seconds=DECISION_POINT_S)
        first = q.enqueue(SYNTHETIC_SOURCE, "run")
        await pool.start()
        try:
            for _ in range(250):
                if ran:
                    break
                await asyncio.sleep(DECISION_POINT_S)
            if not ran:
                return _result("pool_pause", "no-op", None,
                               "the synthetic job never started; nothing was drilled")
            pause(pool)
            q.enqueue(SYNTHETIC_SOURCE, "run")
            await asyncio.sleep(probe_s)
            claimed_after = [i for i in ran if i != first]
            in_flight_stopped = first in finished
        finally:
            release.set()
            await pool.stop()
    if claimed_after:
        return _result("pool_pause", "no-op", None,
                       f"an item enqueued after the pause was claimed within {probe_s}s")
    if in_flight_stopped:
        return _result("pool_pause", "in-flight", None,
                       "the pause stopped the running job")
    return _result("pool_pause", "dispatch-only", None,
                   f"no new claim in {probe_s}s, and the running job carried on — "
                   "a pause is not a stop; use session_cancel for a run in flight")


# ── the drill ───────────────────────────────────────────────────────────────

async def run(status: Optional[dict] = None, *, state_path: Optional[Path] = None,
              invocation: Optional[str] = None,
              **overrides: Any) -> dict[str, Any]:
    """Run every surface and return the report. `overrides` replace a surface's
    control (`session_cancel=`, `pool_pause=`), which is how the tests prove a
    no-op control fails the drill.

    Every measured surface is merged into `app.paths.MITIGATION_DRILL_STATE`
    (or `state_path`), which `GET /api/workers/status` reports as `mitigation`.
    A refused drill measured nothing and writes nothing.

    `invocation` is written onto each reading as the command line responsible
    for it, and only `main()` passes it: an in-process caller of `run()` — a
    test, a REPL, whatever loop left the 20 readings #2431 is about — leaves it
    null, and that is what makes the file self-explaining. `pid` is stamped by
    `mitigation_state.record()` itself, so attribution does not depend on any
    caller remembering to pass it."""
    reason = refusal(status)
    if reason:
        return {"refused": reason, "surfaces": [], "ok": False}
    surfaces = [
        await drill_session_cancel(overrides.get("session_cancel")),
        await drill_pool_pause(overrides.get("pool_pause")),
    ]
    from app import mitigation_state
    try:
        mitigation_state.record(surfaces, path=state_path, invocation=invocation)
    except Exception as e:  # the measurement is the report; a lost write is a note
        print(f"mitigation drill: could not record state: {e}", file=sys.stderr)
    return {
        "refused": None,
        "surfaces": surfaces,
        # Only a measured in-flight stop is a stop. A dispatch-only control is
        # reported beside it, never inside it.
        "in_flight": [s for s in surfaces if s["classification"] == "in-flight"],
        "dispatch_only": [s["surface"] for s in surfaces
                          if s["classification"] == "dispatch-only"],
        "ok": all(s["ok"] for s in surfaces),
    }


def _invocation() -> str:
    """This process's own command line, as it goes onto every reading it writes.

    The seat spawns `[python, "-m", "scripts.mitigation_drill",
    "--wait-free-window", "3600"]`, and python rewrites `argv[0]` to the module's
    FILE path for a `-m` launch — so the recorded string names
    `scripts/mitigation_drill.py` rather than the literal `-m
    scripts.mitigation_drill`. The path is the better half anyway: it says WHICH
    checkout drilled, and `sys.executable` leads it because a candidate venv
    booted from a worktree and the production backend both write to the same
    state file. Neither fact is recoverable afterwards from the reading alone,
    which is exactly the hole #2431 fell into.
    """
    return " ".join([sys.executable, *sys.argv])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--status-url", default=STATUS_URL)
    ap.add_argument("--wait-free-window", dest="wait_free_window", type=int,
                    default=0, metavar="SECONDS",
                    help=f"wait up to SECONDS for the round hold to release, polling "
                         f"every {FREE_WINDOW_POLL_S:g}s, instead of refusing on the "
                         f"first look (0 = one status read and exit 2, the default)")
    args = ap.parse_args(argv)

    if args.wait_free_window > 0:
        # The hold is judged once here, and the payload the wait returned on its
        # clean read is the one `run()` is handed: nothing re-reads the status
        # after the surfaces start, so a gap narrower than the drill still yields
        # a reading instead of a second refusal (#2333).
        status, reason = wait_for_free_window(args.status_url,
                                              args.wait_free_window)
        if reason:
            print(json.dumps({"refused": reason, "surfaces": [], "ok": False,
                              "waited_seconds": args.wait_free_window}, indent=2))
            print(f"refused: {reason}", file=sys.stderr)
            return 2
    else:
        status = live_status(args.status_url)
    report = asyncio.run(run(status, invocation=_invocation()))
    print(json.dumps(report, indent=2))
    if report["refused"]:
        print(f"refused: {report['refused']}", file=sys.stderr)
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
