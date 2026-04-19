"""Task #296: verify per-session turn queue behavior.

Covers Phases 1 and 2:
- Phase 1: concurrent user POSTs serialize; cancel targets current turn
- Phase 2: user preempts running ambient; user tier popped first;
           drain_pending clears ambient only; ambient cancel marker

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
    _session_queues,
    is_session_active,
    get_cancel_event,
    get_queue_state,
    enqueue_turn,
    drain_pending,
)
from app.routers import messages as msg_mod  # noqa: E402


async def _fake_run_turn(session_id: str, turn: SessionTurn, q):
    """Stub — records start/end timestamps on turn.payload."""
    turn.payload["start"] = asyncio.get_event_loop().time()
    await asyncio.sleep(0.3)
    turn.payload["end"] = asyncio.get_event_loop().time()
    await turn.events.put({"event": "done", "data": {"ok": True}})


async def _slow_cancel_aware_run(session_id: str, turn: SessionTurn, q):
    """Run that respects cancel_event — for preempt/cancel tests.

    Total runtime ~0.4s if never cancelled; exits immediately on cancel.
    """
    for _ in range(20):
        if q.cancel_event.is_set():
            turn.payload["cancelled"] = True
            break
        await asyncio.sleep(0.02)
    await turn.events.put({"event": "done", "data": {}})


def _make_turn(label: str, source="user") -> SessionTurn:
    return SessionTurn(
        turn_id=uuid.uuid4().hex[:8],
        source=source,
        payload={"label": label},
        enqueued_at=datetime.now(),
    )


async def _enqueue(session_id: str, turn: SessionTurn):
    return await enqueue_turn(
        session_id,
        turn,
        consumer_factory=lambda: msg_mod._session_consumer(session_id),
    )


async def test_serial_execution():
    msg_mod._run_turn = _fake_run_turn
    session_id = "test-session-serial"
    _session_queues.pop(session_id, None)

    turn_a = _make_turn("A")
    turn_b = _make_turn("B")
    await asyncio.gather(_enqueue(session_id, turn_a), _enqueue(session_id, turn_b))

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
    msg_mod._run_turn = _slow_cancel_aware_run
    session_id = "test-session-cancel"
    _session_queues.pop(session_id, None)

    turn = _make_turn("slow")
    await _enqueue(session_id, turn)
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
    state = get_queue_state(session_id)
    assert state["depth"] == 0
    assert state["current"] is None
    print("OK idle: no cancel_event or queue state for idle session")


async def test_user_preempts_ambient():
    """Enqueueing a user turn while an ambient is running must set the
    cancel_event so the ambient exits and the user turn is next to run.
    """
    msg_mod._run_turn = _slow_cancel_aware_run
    session_id = "test-session-preempt"
    _session_queues.pop(session_id, None)

    ambient = _make_turn("A", source="ambient")
    user = _make_turn("U", source="user")

    # Enqueue ambient first, let it start running
    await _enqueue(session_id, ambient)
    await asyncio.sleep(0.1)
    state = get_queue_state(session_id)
    assert state["current"]["source"] == "ambient", f"expected ambient running, got {state}"

    # Now enqueue user — preempt should fire
    result = await _enqueue(session_id, user)
    assert result["preempted"] is True, f"expected preempted=True, got {result}"
    assert ambient.preempted is True

    await asyncio.wait_for(ambient.done.wait(), timeout=3)
    await asyncio.wait_for(user.done.wait(), timeout=3)

    assert ambient.payload.get("cancelled") is True
    assert user.payload.get("cancelled") is not True
    print("OK preempt: user turn preempted running ambient")


async def test_user_tier_popped_first():
    """User turns in the queue run before ambient turns, regardless of
    enqueue order.
    """
    session_id = "test-session-tier-order"
    _session_queues.pop(session_id, None)

    gate_turn = _make_turn("gate", source="user")

    async def gated_run(session_id, turn, q):
        if turn.turn_id == gate_turn.turn_id:
            await asyncio.sleep(0.2)
        turn.payload["end"] = asyncio.get_event_loop().time()
        await turn.events.put({"event": "done", "data": {}})

    msg_mod._run_turn = gated_run

    await _enqueue(session_id, gate_turn)
    await asyncio.sleep(0.05)  # gate starts running

    ambient = _make_turn("ambient-later", source="ambient")
    user_late = _make_turn("user-late", source="user")
    await _enqueue(session_id, ambient)
    await _enqueue(session_id, user_late)

    await asyncio.wait_for(gate_turn.done.wait(), timeout=3)
    await asyncio.wait_for(user_late.done.wait(), timeout=3)
    await asyncio.wait_for(ambient.done.wait(), timeout=3)

    assert user_late.payload["end"] < ambient.payload["end"], (
        f"user_late ran after ambient: user_late={user_late.payload['end']:.3f} "
        f"ambient={ambient.payload['end']:.3f}"
    )
    print("OK tier order: user-tier popped before ambient-tier")


async def test_drain_pending_ambient_only():
    """drain_pending(source='ambient') clears queued ambients; user queue
    untouched.
    """
    msg_mod._run_turn = _slow_cancel_aware_run
    session_id = "test-session-drain"
    _session_queues.pop(session_id, None)

    running = _make_turn("running", source="ambient")
    a1 = _make_turn("a1", source="ambient")
    a2 = _make_turn("a2", source="ambient")
    u1 = _make_turn("u1", source="user")

    await _enqueue(session_id, running)
    await asyncio.sleep(0.05)
    await _enqueue(session_id, a1)
    await _enqueue(session_id, a2)

    drained = await drain_pending(session_id, source="ambient")
    assert drained == 2, f"expected 2 ambients drained, got {drained}"

    state = get_queue_state(session_id)
    assert state["pending_ambient"] == 0
    assert state["pending_user"] == 0

    await _enqueue(session_id, u1)
    await asyncio.wait_for(running.done.wait(), timeout=3)
    await asyncio.wait_for(u1.done.wait(), timeout=3)
    assert a1.done.is_set() and a2.done.is_set()
    print("OK drain: 2 queued ambients dropped, running turn preempted, user ran")


async def main():
    await test_serial_execution()
    await test_cancel_current()
    await test_idle_returns_none_event()
    await test_user_preempts_ambient()
    await test_user_tier_popped_first()
    await test_drain_pending_ambient_only()
    print("\nAll tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
