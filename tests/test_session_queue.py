"""Task #296: verify per-session turn queue behavior.

Covers Phases 1–3:
- Phase 1: concurrent user POSTs serialize; cancel targets current turn
- Phase 2: user preempts running ambient; user tier popped first;
           drain_pending clears ambient only; ambient cancel marker
- Phase 3: ambient queue cap drops oldest; dedup_key collapses duplicates;
           queue_state event emitted on enqueue/drain
- D11: `/compact` is a queued turn — it waits behind a running turn, and a
        row that turn appends survives the compaction
- #909: the two-tier drain — `DELETE /api/sessions/{id}` empties both
        queues and a queued user turn never reaches `_run_turn`, while
        `/cancel?drain_pending=true` stays ambient-only

Run: .venvs/lloyd/bin/python -m tests.test_session_queue
"""
import asyncio
import json
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import sessions_io  # noqa: E402
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
from app.routers import sessions as sess_mod  # noqa: E402

import pytest  # noqa: E402

# Captured at import, before any test swaps in a stub: the tests below assign
# `msg_mod._run_turn` directly, and the D11 test needs the real dispatch.
_REAL_RUN_TURN = msg_mod._run_turn


@pytest.fixture(autouse=True)
def _restore_run_turn():
    """Put the real `_run_turn` back after every test, so a stub assigned here
    does not leak into a later test module in the same process."""
    yield
    msg_mod._run_turn = _REAL_RUN_TURN


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


# ── #909: the delete route has to drain BOTH tiers ───────────────────────────
#
# `drain_pending` pops `q.pending_user` only under `source == "user"`, and
# nothing in production ever passed that, so `DELETE /api/sessions/{id}` —
# the caller whose comment claimed "drain all queued turns" — left a queued
# user turn sitting in the queue. The consumer then ran it to completion for
# a session whose JSON was already unlinked, and its writes no-op'd, because
# `_append_messages` goes through `mutate_session`, which returns False once
# the file is gone. A turn spent on a transcript nobody will read again.
#
# None of the tests above could have caught that: every drain call there
# passes `source="ambient"` explicitly, so nothing pinned the `source=None`
# default at all, and nothing drove the two HTTP handlers. These do.


class _FakeRequest:
    """Stand-in for the `Request` the cancel route only reads query params from."""

    def __init__(self, query_params: dict):
        self.query_params = query_params


def _blocking_run_stub():
    """A `_run_turn` stub that records what reaches it and stays in flight.

    Returns (stub, started, release). `started` is the id of every turn that
    actually reached `_run_turn`, which is the only assertion that matters for
    a drain: a dropped turn is one the consumer never popped. The stub ignores
    `cancel_event` on purpose — the running turn has to still be in flight
    while the handler runs, which is what a real turn does between checks.
    """
    started: list[str] = []
    release = asyncio.Event()

    async def stub(session_id: str, turn: SessionTurn, q):
        started.append(turn.turn_id)
        await release.wait()
        await turn.events.put({"event": "done", "data": {}})

    return stub, started, release


async def _queue_both_tiers(session_id: str):
    """Put the queue in the state the clauses name: running ambient, one
    queued ambient, one queued user turn.

    Returns (running, queued_ambient, queued_user, started, release). The
    setup itself is asserted, so a test cannot pass against an empty queue.
    """
    stub, started, release = _blocking_run_stub()
    msg_mod._run_turn = stub
    _session_queues.pop(session_id, None)

    running = _make_turn("running", source="ambient")
    queued_ambient = _make_turn("q-ambient", source="ambient")
    queued_user = _make_turn("q-user", source="user")
    await _enqueue(session_id, running)
    await asyncio.sleep(0.05)  # it becomes q.current
    await _enqueue(session_id, queued_ambient)
    await _enqueue(session_id, queued_user)

    state = get_queue_state(session_id)
    assert state["current"] and state["current"]["source"] == "ambient", f"setup: {state}"
    assert state["pending_ambient"] == 1, f"setup: {state}"
    assert state["pending_user"] == 1, f"setup: {state}"
    return running, queued_ambient, queued_user, started, release


async def _run_delete_handler(session_id: str):
    """Drive the real delete route with `SESSIONS_DIR` aimed at a temp dir.

    Both modules get patched: the handler resolves `SESSIONS_DIR` through
    `app.routers.sessions`, while `mutate_session` — the write path a
    surviving queued turn would take — reads `app.sessions_io.SESSIONS_DIR`.
    Returns (parsed body, file still on disk after the call).
    """
    tmp = Path(tempfile.mkdtemp(prefix="lloyd-909-"))
    (tmp / f"{session_id}.json").write_text(
        json.dumps({"session_id": session_id, "messages": []})
    )
    try:
        with patch.object(sessions_io, "SESSIONS_DIR", tmp), patch.object(
            sess_mod, "SESSIONS_DIR", tmp
        ):
            response = await sess_mod.delete_session(session_id)
        await asyncio.sleep(0)  # one pass for the consumer to react
        return json.loads(response.body), (tmp / f"{session_id}.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def _release_and_join(release, *turns):
    """Let the in-flight turns finish so no consumer task outlives the test."""
    release.set()
    for turn in turns:
        await asyncio.wait_for(turn.done.wait(), timeout=3)


async def test_delete_session_leaves_nothing_queued_in_either_tier():
    """#909 clause 1: after DELETE, the queue reports zero in both tiers."""
    sid = "test-session-delete-both"
    running, _qa, _qu, _started, release = await _queue_both_tiers(sid)
    try:
        body, still_on_disk = await _run_delete_handler(sid)
        assert body["deleted"] is True, f"handler did not remove the JSON: {body}"
        assert not still_on_disk, "the session file survived the handler it targets"
        state = get_queue_state(sid)
        assert state["pending_ambient"] == 0, f"queued ambient survived: {state}"
        assert state["pending_user"] == 0, f"queued user turn survived: {state}"
        assert state["depth"] == 0, f"something is still queued: {state}"
    finally:
        await _release_and_join(release, running)
    print("OK delete: both tiers emptied, session JSON removed")


async def test_a_queued_user_turn_never_reaches_run_turn_after_delete():
    """#909 clause 2: the drained user turn is never executed.

    Three observations, and only two of them can fail, so the file is explicit
    about which is which. The stub **is** the step the consumer awaits, so while
    the running turn is in flight the consumer is suspended inside
    `_run_turn(running)` and cannot reach a second pop — `started` naming only
    the running turn is an invariant of that seam, not evidence about the
    handler, and it is checked only as the ordering it must never break. What
    the handler does change, and what pre-fix code left wrong, is the queue:
    `pending_user` is 0 by the time it returns instead of 1. The observation
    that proves the pop mattered is the one made *after* the running turn is
    released, because a survivor surfaces exactly then, as a second
    `_run_turn` call.
    """
    sid = "test-session-delete-no-run"
    running, queued_ambient, queued_user, started, release = await _queue_both_tiers(sid)
    body, _still = await _run_delete_handler(sid)
    assert body["deleted"] is True, f"setup: {body}"
    state = get_queue_state(sid)
    assert state["pending_user"] == 0, (
        f"the handler returned with the queued user turn still queued: {state}"
    )
    assert started == [running.turn_id], (
        f"the consumer reached a pop outside the running turn: {started}"
    )
    await _release_and_join(release, running)
    assert started == [running.turn_id], (
        f"turns reached _run_turn beyond the running one: {started} "
        f"(queued_user={queued_user.turn_id} queued_ambient={queued_ambient.turn_id})"
    )
    print("OK delete: queued user turn never reached _run_turn")


async def test_drain_pending_all_clears_both_tiers_in_one_call():
    """#909 clause 3: one call with source="all" pops both tiers and returns
    the summed count — the call the delete route makes, and what gives the
    `pending_user` pop a production caller.
    """
    stub, started, release = _blocking_run_stub()
    msg_mod._run_turn = stub
    sid = "test-session-drain-all"
    _session_queues.pop(sid, None)

    running = _make_turn("running", source="ambient")
    await _enqueue(sid, running)
    await asyncio.sleep(0.05)
    ambients = [_make_turn(f"a{i}", source="ambient") for i in range(2)]
    users = [_make_turn(f"u{i}", source="user") for i in range(2)]
    for turn in ambients + users:
        await _enqueue(sid, turn)

    before = get_queue_state(sid)
    assert (before["pending_ambient"], before["pending_user"]) == (2, 2), f"setup: {before}"

    drained = await drain_pending(sid, source="all")
    after = get_queue_state(sid)
    assert drained == 4, f"expected the 2+2 queued turns summed, got {drained}"
    assert (after["pending_ambient"], after["pending_user"]) == (0, 0), f"after: {after}"
    # Same drop signature as the ambient tier: marked preempted and released,
    # so nothing waits on a turn that will never run.
    assert all(t.preempted for t in ambients + users), "a drained turn was not marked preempted"
    assert all(t.done.is_set() for t in ambients + users), "a drained turn was left awaited"

    await _release_and_join(release, running)
    assert started == [running.turn_id], f"drained turns still ran: {started}"
    print(f"OK drain all: {drained} turns across both tiers, none ran")


async def test_drain_pending_without_a_source_is_ambient_only():
    """The `source=None` default — what `/cancel?drain_pending=true` relies
    on and what the delete route used to *believe* it had.

    No test pinned this before #909, which is how a comment saying "drain all
    queued turns" survived next to a one-tier call. Clause 5 requires the
    default stay exactly as documented; this is the pin.
    """
    sid = "test-session-drain-default"
    running, _qa, queued_user, started, release = await _queue_both_tiers(sid)
    try:
        drained = await drain_pending(sid)
        assert drained == 1, f"expected the one queued ambient, got {drained}"
        state = get_queue_state(sid)
        assert state["pending_ambient"] == 0, f"ambient tier not drained: {state}"
        assert state["pending_user"] == 1, f"default touched the user tier: {state}"
        assert not queued_user.done.is_set(), "default dropped a user turn"
    finally:
        await _release_and_join(release, running, queued_user)
    assert queued_user.turn_id in started, "the surviving user turn never ran"
    print("OK default: drain_pending() is ambient-only, user turn kept")


async def test_drain_pending_user_only_clears_the_user_tier():
    """`source="user"` exercises the user-tier arm on its own.

    `"all"` is what reaches that arm in production, but a one-tier request is
    the call that shows which arm did the work: the ambient count has to be
    untouched here, and the turns cleared have to be exactly the user ones.
    """
    sid = "test-session-drain-user-only"
    running, queued_ambient, queued_user, started, release = await _queue_both_tiers(sid)
    try:
        drained = await drain_pending(sid, source="user")
        assert drained == 1, f"expected the one queued user turn, got {drained}"
        state = get_queue_state(sid)
        assert state["pending_user"] == 0, f"user tier survived: {state}"
        assert state["pending_ambient"] == 1, f"ambient tier was cleared too: {state}"
        assert queued_user.preempted is True, "the dropped user turn was not marked preempted"
        assert queued_user.done.is_set(), "the dropped user turn was left awaited"
        assert not queued_ambient.done.is_set(), "the queued ambient was released by a user drain"
    finally:
        await _release_and_join(release, running, queued_ambient)
    assert queued_user.turn_id not in started, "the drained user turn ran anyway"
    assert queued_ambient.turn_id in started, "the untouched ambient never ran"
    print("OK drain user: user tier cleared alone, ambient kept and ran")


async def test_cancel_without_drain_pending_clears_nothing():
    """The route's other branch: `drain_pending` absent or false.

    `/cancel` with no query flag cancels the running turn and leaves **both**
    queues exactly as they were — nothing drained, no `done` set, every queued
    turn still waiting on its own turn with the consumer. The ambient-only test
    above drives only the draining arm, so without this one the route could
    start dropping the user tier on the flag-less path, the behaviour voice and
    the dashboard actually use, and every test here would stay green.
    """
    sid = "test-session-cancel-no-drain"
    running, queued_ambient, queued_user, started, release = await _queue_both_tiers(sid)

    response = await sess_mod.cancel_session(sid, _FakeRequest({}))
    body = json.loads(response.body)
    assert body["cancelled"] is True, f"running turn not cancelled: {body}"
    assert body["drained"] == 0, f"flag-less cancel drained something: {body}"

    state = get_queue_state(sid)
    assert (state["pending_ambient"], state["pending_user"]) == (1, 1), (
        f"a queue changed on the flag-less path: {state}"
    )
    assert not queued_ambient.done.is_set(), "flag-less cancel released the queued ambient"
    assert not queued_user.done.is_set(), "flag-less cancel released the queued user turn"

    await _release_and_join(release, running, queued_ambient, queued_user)
    assert started == [running.turn_id, queued_user.turn_id, queued_ambient.turn_id], (
        f"queue order changed: {started}"
    )
    print("OK cancel (no flag): nothing drained, both queued turns ran in order")


async def test_cancel_with_drain_pending_leaves_the_queued_user_turn():
    """#909 clause 4: `/cancel?drain_pending=true` stays ambient-only.

    The queued user turn keeps its place in `pending_user`, its `done` stays
    unset, and it still runs once the in-flight turn finishes. Dropping it
    here would defeat the whole point of the ambient-only default: user turns
    are never silently dropped by cancel.
    """
    sid = "test-session-cancel-keeps-user"
    running, queued_ambient, queued_user, started, release = await _queue_both_tiers(sid)

    response = await sess_mod.cancel_session(sid, _FakeRequest({"drain_pending": "true"}))
    body = json.loads(response.body)
    assert body["cancelled"] is True, f"running turn not cancelled: {body}"
    assert body["drained"] == 1, f"expected the one queued ambient drained, got {body}"

    state = get_queue_state(sid)
    assert state["pending_user"] == 1, f"cancel dropped the user turn: {state}"
    q = _session_queues[sid]
    assert q.pending_user[0].turn_id == queued_user.turn_id, "cancel reordered the user tier"
    assert not queued_user.done.is_set(), "cancel released a waiter on a turn it did not run"
    assert queued_user.preempted is False, "cancel marked the queued user turn preempted"

    await _release_and_join(release, running, queued_user)
    assert queued_user.turn_id in started, "the surviving user turn never ran"
    assert queued_ambient.turn_id not in started, "the drained ambient ran anyway"
    print("OK cancel: drain_pending=true cleared ambient, user turn survived and ran")


async def test_a_compact_request_waits_behind_the_running_turn_and_a_row_appended_meanwhile_survives(
    tmp_path, monkeypatch,
):
    """D11: `/compact` holds the session's slot like any turn. It used to run
    beside the queue, read a snapshot, await the summariser and then replace
    `data["messages"]` — deleting whatever the running turn appended during
    that wait. Queued, it starts only after the running turn ends, and it never
    rewrites the messages, so the late row is there afterwards."""
    from app import compaction_llm
    from app import event_log
    from app.config import CONFIG
    from agent_mcp import _change_ledger as ledger

    sid = "20260924_120000_d11q"
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(msg_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(event_log, "EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr(ledger, "CHANGES_ROOT", tmp_path / "changes")
    monkeypatch.setitem(CONFIG, "compaction", {
        "mode": "summarize", "keep_recent_turns": 1, "persist_summary": False,
        "microcompact": {"enabled": False}, "restore": {"enabled": False},
    })
    summarised: list[list[str]] = []

    async def _summarise(prior, delta, **kw):
        summarised.append([r.get("id") for r in delta])
        return "## Goal\nG"
    monkeypatch.setattr(compaction_llm, "summarize_incremental", _summarise)

    rows = []
    for i in range(3):
        rows += [{"id": f"u{i}", "role": "user", "content": [{"type": "text", "text": f"q{i}"}]},
                 {"id": f"k{i}", "role": "thinking", "content": [], "reasoning": "r"},
                 {"id": f"a{i}", "role": "assistant", "content": [{"type": "text", "text": f"a{i}"}]}]
    (tmp_path / f"{sid}.json").write_text(json.dumps({"session_id": sid, "messages": rows}))

    release = asyncio.Event()
    order: list[str] = []

    async def running_turn(session_id, turn, q):
        if turn.payload.get("kind") == "compact":
            order.append("compact")
            return await _REAL_RUN_TURN(session_id, turn, q)
        order.append("running")
        await release.wait()
        await sessions_io._append_messages(session_id, [
            {"id": "late", "role": "assistant",
             "content": [{"type": "text", "text": "written while /compact waited"}]}])
        order.append("appended")

    msg_mod._run_turn = running_turn
    _session_queues.pop(sid, None)
    running = _make_turn("running")
    await _enqueue(sid, running)
    await asyncio.sleep(0.05)
    compact = msg_mod._compact_turn(sid, "/compact", "qwen-unknown")
    await _enqueue(sid, compact)
    await asyncio.sleep(0.05)
    assert order == ["running"], "the compact turn started under a running turn"
    assert get_queue_state(sid)["pending_user"] == 1

    release.set()
    await asyncio.wait_for(compact.done.wait(), timeout=5)
    assert order == ["running", "appended", "compact"]

    data = json.loads((tmp_path / f"{sid}.json").read_text())
    ids = [m["id"] for m in data["messages"]]
    assert "late" in ids, "a row appended while /compact waited was deleted"
    assert ids[:len(rows)] == [r["id"] for r in rows]
    assert data["compaction"]["source"] == "manual"
    assert summarised and "late" not in summarised[0]
    events = []
    while not compact.events.empty():
        evt = compact.events.get_nowait()
        if evt is not None:
            events.append(evt["event"])
    assert "compact_done" in events


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
    await test_delete_session_leaves_nothing_queued_in_either_tier()
    await test_a_queued_user_turn_never_reaches_run_turn_after_delete()
    await test_drain_pending_all_clears_both_tiers_in_one_call()
    await test_drain_pending_without_a_source_is_ambient_only()
    await test_drain_pending_user_only_clears_the_user_tier()
    await test_cancel_without_drain_pending_clears_nothing()
    await test_cancel_with_drain_pending_leaves_the_queued_user_turn()
    print("\nAll tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
