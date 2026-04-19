"""Task #296: verify per-session turn queue behavior.

Covers Phases 1–3:
- Phase 1: concurrent user POSTs serialize; cancel targets current turn
- Phase 2: user preempts running ambient; user tier popped first;
           drain_pending clears ambient only; ambient cancel marker
- Phase 3: ambient queue cap drops oldest; dedup_key collapses duplicates;
           queue_state event emitted on enqueue/drain

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
    AMBIENT_QUEUE_CAP,
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


async def test_ambient_queue_cap():
    """Queued ambient turns beyond AMBIENT_QUEUE_CAP get dropped (oldest
    first). Running turn is untouched.
    """
    msg_mod._run_turn = _slow_cancel_aware_run
    session_id = "test-session-cap"
    _session_queues.pop(session_id, None)

    running = _make_turn("running", source="ambient")
    await _enqueue(session_id, running)
    await asyncio.sleep(0.05)

    # Fill the queue up to cap, then add one more.
    queued = [_make_turn(f"a{i}", source="ambient") for i in range(AMBIENT_QUEUE_CAP + 1)]
    results = []
    for t in queued:
        results.append(await _enqueue(session_id, t))

    # The final enqueue should have reported the oldest queued turn as dropped.
    assert results[-1]["dropped"], f"expected non-empty dropped list, got {results[-1]}"
    assert queued[0].turn_id in results[-1]["dropped"], (
        f"expected oldest {queued[0].turn_id} in dropped, got {results[-1]['dropped']}"
    )
    assert queued[0].done.is_set(), "dropped turn's done event should be set"

    state = get_queue_state(session_id)
    assert state["pending_ambient"] == AMBIENT_QUEUE_CAP

    # Drain so the test exits cleanly.
    await drain_pending(session_id, source="ambient")
    ev = get_cancel_event(session_id)
    if ev: ev.set()
    await asyncio.wait_for(running.done.wait(), timeout=3)
    print(f"OK cap: AMBIENT_QUEUE_CAP={AMBIENT_QUEUE_CAP} enforced, oldest dropped")


async def test_ambient_dedup_collapse():
    """Enqueueing an ambient with the same dedup_key as a queued one
    drops the old one; newest wins.
    """
    msg_mod._run_turn = _slow_cancel_aware_run
    session_id = "test-session-dedup"
    _session_queues.pop(session_id, None)

    running = _make_turn("running", source="ambient")
    await _enqueue(session_id, running)
    await asyncio.sleep(0.05)

    # Two ambients with distinct keys.
    a1 = SessionTurn(
        turn_id=uuid.uuid4().hex[:8], source="ambient",
        payload={"dedup_key": "K1", "label": "v1"}, enqueued_at=datetime.now(),
    )
    a2 = SessionTurn(
        turn_id=uuid.uuid4().hex[:8], source="ambient",
        payload={"dedup_key": "K2", "label": "v1"}, enqueued_at=datetime.now(),
    )
    # Third one with same key as a1 — should collapse a1.
    a1_new = SessionTurn(
        turn_id=uuid.uuid4().hex[:8], source="ambient",
        payload={"dedup_key": "K1", "label": "v2"}, enqueued_at=datetime.now(),
    )
    await _enqueue(session_id, a1)
    await _enqueue(session_id, a2)
    result = await _enqueue(session_id, a1_new)

    assert result["deduped"] is True, f"expected deduped=True, got {result}"
    assert a1.turn_id in result["dropped"], f"expected a1 ({a1.turn_id}) dropped, got {result}"
    assert a1.done.is_set()

    state = get_queue_state(session_id)
    # a2 (K2) and a1_new (K1) remain, a1 collapsed.
    assert state["pending_ambient"] == 2, f"expected 2 pending, got {state}"

    # Drain + cancel so the running turn exits.
    await drain_pending(session_id, source="ambient")
    ev = get_cancel_event(session_id)
    if ev: ev.set()
    await asyncio.wait_for(running.done.wait(), timeout=3)
    print("OK dedup: same dedup_key collapsed, newest kept")


async def test_queue_state_event_emitted():
    """Subscribers of the running turn receive a queue_state event when
    another turn is enqueued.
    """
    msg_mod._run_turn = _slow_cancel_aware_run
    session_id = "test-session-qs-event"
    _session_queues.pop(session_id, None)

    running = _make_turn("running", source="ambient")
    await _enqueue(session_id, running)
    await asyncio.sleep(0.05)

    # Drain current events so we have a clean slate.
    while not running.events.empty():
        running.events.get_nowait()

    # Enqueueing another ambient should broadcast queue_state into running.events.
    extra = _make_turn("extra", source="ambient")
    await _enqueue(session_id, extra)

    # First event on running's queue should be queue_state.
    evt = await asyncio.wait_for(running.events.get(), timeout=1.0)
    assert evt["event"] == "queue_state", f"expected queue_state, got {evt}"
    assert evt["data"]["pending_ambient"] >= 1, f"expected pending_ambient>=1, got {evt}"

    # Cleanup.
    await drain_pending(session_id, source="ambient")
    ev = get_cancel_event(session_id)
    if ev: ev.set()
    await asyncio.wait_for(running.done.wait(), timeout=3)
    print("OK queue_state: event broadcast to running turn on enqueue")


async def main():
    await test_serial_execution()
    await test_cancel_current()
    await test_idle_returns_none_event()
    await test_user_preempts_ambient()
    await test_user_tier_popped_first()
    await test_drain_pending_ambient_only()
    await test_ambient_queue_cap()
    await test_ambient_dedup_collapse()
    await test_queue_state_event_emitted()
    print("\nAll tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
