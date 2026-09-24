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
adds load while a round or the landing behind it wants the box.

    .venvs/lloyd/bin/python -m scripts.mitigation_drill        # JSON, exit 1 on a failed surface
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

async def run(status: Optional[dict] = None, **overrides: Any) -> dict[str, Any]:
    """Run every surface and return the report. `overrides` replace a surface's
    control (`session_cancel=`, `pool_pause=`), which is how the tests prove a
    no-op control fails the drill."""
    reason = refusal(status)
    if reason:
        return {"refused": reason, "surfaces": [], "ok": False}
    surfaces = [
        await drill_session_cancel(overrides.get("session_cancel")),
        await drill_pool_pause(overrides.get("pool_pause")),
    ]
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


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--status-url", default=STATUS_URL)
    args = ap.parse_args(argv)
    report = asyncio.run(run(live_status(args.status_url)))
    print(json.dumps(report, indent=2))
    if report["refused"]:
        print(f"refused: {report['refused']}", file=sys.stderr)
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
