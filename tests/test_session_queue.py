"""Task #296 Phase 1: verify per-session turn queue serializes concurrent enqueues.

Run: .venvs/lloyd/bin/python -m tests.test_session_queue
"""
import asyncio
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sessions_io import (  # noqa: E402
    SessionTurn,
    _get_or_create_queue,
    _session_queues,
    is_session_active,
    get_cancel_event,
)
from app.routers import messages as msg_mod  # noqa: E402


async def _fake_run_turn(session_id: str, turn: SessionTurn, q):
    """Stub — records start/end timestamps on turn.payload."""
    turn.payload["start"] = asyncio.get_event_loop().time()
    await asyncio.sleep(0.3)
    turn.payload["end"] = asyncio.get_event_loop().time()
    # Minimal event flow so the SSE subscriber would work if attached.
    await turn.events.put({"event": "done", "data": {"ok": True}})


async def _enqueue(session_id: str, label: str) -> SessionTurn:
    q = _get_or_create_queue(session_id)
    turn = SessionTurn(
        turn_id=uuid.uuid4().hex[:8],
        source="user",
        payload={"label": label},
        enqueued_at=datetime.now(),
    )
    async with q.lock:
        q.pending.append(turn)
        if q.consumer_task is None or q.consumer_task.done():
            q.consumer_task = asyncio.create_task(msg_mod._session_consumer(session_id))
    return turn


async def test_serial_execution():
    # Monkey-patch _run_turn inside messages module
    msg_mod._run_turn = _fake_run_turn
    session_id = "test-session-serial"
    _session_queues.pop(session_id, None)

    # Kick off two near-simultaneous enqueues
    turn_a, turn_b = await asyncio.gather(
        _enqueue(session_id, "A"),
        _enqueue(session_id, "B"),
    )

    # Wait for both to finish
    await asyncio.wait_for(turn_a.done.wait(), timeout=5)
    await asyncio.wait_for(turn_b.done.wait(), timeout=5)

    a_start = turn_a.payload["start"]
    a_end = turn_a.payload["end"]
    b_start = turn_b.payload["start"]
    b_end = turn_b.payload["end"]

    assert a_end <= b_start + 0.01, (
        f"Turn B started before turn A finished: a_end={a_end:.3f} b_start={b_start:.3f}"
    )
    assert b_end > b_start
    print(f"OK serial: A=[{a_start:.3f}..{a_end:.3f}] B=[{b_start:.3f}..{b_end:.3f}]")


async def test_cancel_current():
    """Cancel-event targets the currently running turn only."""
    # Slow run_turn that respects cancel_event
    async def slow_run(session_id: str, turn: SessionTurn, q):
        for _ in range(50):
            if q.cancel_event.is_set():
                turn.payload["cancelled"] = True
                break
            await asyncio.sleep(0.05)
        await turn.events.put({"event": "done", "data": {}})

    msg_mod._run_turn = slow_run
    session_id = "test-session-cancel"
    _session_queues.pop(session_id, None)

    turn = await _enqueue(session_id, "slow")
    # Give consumer a moment to pick up
    await asyncio.sleep(0.1)
    assert is_session_active(session_id)
    ev = get_cancel_event(session_id)
    assert ev is not None
    ev.set()
    await asyncio.wait_for(turn.done.wait(), timeout=3)
    assert turn.payload.get("cancelled") is True
    print("OK cancel: slow turn honored cancel_event")


async def test_idle_returns_none_event():
    session_id = "test-session-idle"
    _session_queues.pop(session_id, None)
    assert not is_session_active(session_id)
    assert get_cancel_event(session_id) is None
    print("OK idle: no cancel_event for idle session")


async def main():
    await test_serial_execution()
    await test_cancel_current()
    await test_idle_returns_none_event()
    print("\nAll tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
