"""#2599: an ambient `surface=false` silences the turn that wrote it, and no other.

`ambient_decide(surface=false)` answers a question about one signal: "read this,
is it worth interrupting the user?" The answer was stored against the *session*
(`_ambient_decisions[session_id]`) and popped by session alone, while the same
handler already returned the `turn_id` it never recorded. So the answer was never
attributed to the turn that gave it, and two paths spent it on somebody else:

  * A decide call can land up to 0.5 s after its own turn stopped being current,
    because `surface=false` schedules its cancel through `_deferred_ambient_cancel`
    precisely so the in-flight dispatch is not cut off. If the queue promoted a
    successor ambient turn in that window, the late call passed the tier-only
    gate and filled the slot the successor was about to read.
  * The pop is not in a `finally` clause: it sits inside the harness `result`
    branch of `app/routers/messages.py`, and the same file's Inner-Voice comment
    spells out the endings that never reach it — "the user hit Stop, the harness
    raised, the consumer was cancelled". One writer, one reader, no cleanup at
    turn end: an ambient turn that died that way left its `surface=false` in the
    dict for the next ambient turn of that session to inherit, no timing needed.

Either way the successor's reply was replaced by
`(ambient: Lloyd reviewed and chose not to surface — <the dead turn's reason>)`
plus an `ambient_silent` event, and the answer the agent meant to give the user
was gone. Traffic is thin but live (19 real `ambient_decide` calls across the
session JSONs on this box at 2026-10-12) and no crossed silencing has left a
fingerprint yet — a coverage gap, not a clean bill.

What is pinned here: the slot carries the deciding turn's id, the read answers
only that id, a mismatch is removed and logged naming both ids, and the prose
stops promising a `finally`-clause pop (that half is
`tests/test_session_doc_claims.py`). The write half is driven through
`post_ambient_decide` itself — over ASGI for the route, and from inside a driven
turn for the in-process tool call — and the read half through the real
`_run_turn`, on a private copy because `tests/test_session_queue.py` rebinds the
live module's `_run_turn` to its own stubs. The two halves are joined by the one
dict the code shares, not by a mock of it.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import sessions_io as sio
from app.routers import sessions as sess_mod
from tests._messages_copy import load_messages_copy, patch_on_copy

SID = "20261012_064015_webchat"
REASON = "the morning brief already carried this and nothing has changed since"
REPLY = "Your 10:00 moved to the lab; the courier is still coming at 11."
# The breadcrumb `_run_turn` writes for a silent turn. Its text and the
# `ambient_silent` event are the visible symptom of a crossed silencing, so a
# test that only looked at the store would miss what the user actually loses.
SILENT_PREFIX = "(ambient: Lloyd reviewed and chose not to surface"
DECIDE = {"surface": False, "message": "", "reasoning": REASON}


@pytest.fixture(autouse=True)
def isolated_ambient_state(monkeypatch):
    """The decision store and the queues are module globals: nothing is inherited."""
    monkeypatch.setattr(sio, "_ambient_decisions", {})
    monkeypatch.setattr(sio, "_session_queues", {})


@pytest.fixture(autouse=True)
def no_deferred_cancel(monkeypatch):
    """`surface=false` also schedules `_deferred_ambient_cancel` on the turn's
    cancel event, 0.5 s out. That delay is the queue-preemption half of the
    design — the arch page's "The cancel is deferred on purpose" section — and
    the persistence rewrite under test reads the decision, not the event, so the
    event is stubbed off rather than left to fire after `asyncio.run` closed the
    loop its task belongs to.
    """
    monkeypatch.setattr(sess_mod, "get_cancel_event", lambda session_id: None)


class _Request:
    """The one thing `post_ambient_decide` asks a `Request` for."""

    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return dict(self._body)


def _turn(turn_id: str, source: str = "ambient") -> sio.SessionTurn:
    return sio.SessionTurn(
        turn_id=turn_id, source=source,
        payload={"text": "a background signal",
                 "prefetched_text": "a background signal", "model": "primary"},
        enqueued_at=None,
    )


def _running(turn: sio.SessionTurn) -> sio.SessionQueue:
    """A queue holding `turn` as its current turn, the way `_session_consumer`
    does (`q.current = turn` immediately before it calls `_run_turn`). That is
    what the decide route's gate reads, so the gate runs against real state.
    """
    q = sio.SessionQueue()
    q.cancel_event = asyncio.Event()
    q.current = turn
    turn.session_id = SID
    sio._session_queues[SID] = q
    return q


# --- clause 1: the route records the id it already had in hand --------------

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(sess_mod.router)
    with TestClient(app) as c:
        yield c


def test_the_route_records_the_deciding_turn_id_and_answers_with_it(client):
    """The stored decision and the response name one and the same turn."""
    _running(_turn("turn-a-2599"))
    r = client.post(f"/api/sessions/{SID}/ambient-decide", json=DECIDE)
    assert r.status_code == 200, r.text
    stored = sio._ambient_decisions[SID]
    assert stored.get("turn_id") == "turn-a-2599", (
        f"the slot holds {stored!r}: with no id in it the consumer cannot tell "
        "whose decision it is reading")
    assert r.json()["turn_id"] == stored["turn_id"], (
        "the response names a different turn than the one recorded, so the "
        f"caller cannot check it either: {r.json()!r} vs {stored!r}")
    assert stored["surface"] is False and stored["reasoning"] == REASON, stored


def test_a_user_turn_current_still_cannot_record_a_decision(client):
    """The tier gate is untouched by this change, and this is the control that
    the write above came from the route and not from a slot left full.
    """
    _running(_turn("turn-user-2599", source="user"))
    r = client.post(f"/api/sessions/{SID}/ambient-decide", json=DECIDE)
    assert r.status_code == 400, r.text
    assert SID not in sio._ambient_decisions, sio._ambient_decisions


# --- clauses 2 and 4: the read answers only its own turn, and says so -------

def _record(turn_id: str, *, surface: bool = False, reasoning: str = REASON) -> None:
    """The dict the route writes — clause 1 pins that it is this one — placed
    directly, for the store-level tests.
    """
    sio.set_ambient_decision(SID, {
        "surface": surface, "message": "", "reasoning": reasoning,
        "turn_id": turn_id,
    })


def test_the_recording_turn_is_handed_its_own_decision():
    _record("turn-a-2599")
    got = sio.take_ambient_decision(SID, "turn-a-2599")
    assert got is not None and got["surface"] is False, got
    assert got["reasoning"] == REASON, got
    assert sio.take_ambient_decision(SID, "turn-a-2599") is None, (
        "a decision is read once; a second ask must not silence anything again")


def test_a_decision_recorded_by_another_turn_is_denied_and_cleared():
    """A mismatch is a discard, not a hold-back: leaving it in the slot would
    hand it to a third ambient turn, which is the bug being fixed.
    """
    _record("turn-a-2599")
    assert sio.take_ambient_decision(SID, "turn-b-2599") is None
    assert SID not in sio._ambient_decisions, (
        f"a mismatched decision stayed in the slot for the next reader: "
        f"{sio._ambient_decisions!r}")
    assert sio.take_ambient_decision(SID, "turn-a-2599") is None, (
        "the turn that recorded a discarded decision gets no second attempt at it")


def test_a_discarded_mismatch_is_logged_naming_both_turn_ids(caplog):
    """The discard has to be readable in the log rather than inferred after the
    fact — the reason #910's drops were named in theirs.
    """
    _record("recorded-by-aaaa1")
    with caplog.at_level(logging.WARNING, logger="lloyd-server"):
        assert sio.take_ambient_decision(SID, "asking-bbbb2") is None
    lines = [r.getMessage() for r in caplog.records
             if r.name == "lloyd-server" and r.levelno >= logging.WARNING]
    assert len(lines) == 1, f"expected exactly one warning for one discard: {lines}"
    assert "recorded-by-aaaa1" in lines[0], lines[0]
    assert "asking-bbbb2" in lines[0], lines[0]
    assert SID in lines[0], f"the warning does not name the session it is about: {lines[0]}"


# --- clause 3: one turn's silence stays that turn's -------------------------

async def _fake_compact(*a, **k):
    return {"history": [], "truncated": False, "tokens_before": 0,
            "tokens_after": 0, "context_window": 262144, "threshold": 0}


async def _fake_prepare(history, model=None, **k):
    return list(history)


async def _fake_prefetch(*a, **k):
    return "prefetched"


async def _no_observer(*a, **k):
    return None


@contextmanager
def _ambient_turns(monkeypatch, tmp_path):
    """A private copy of the chat router, plus `drive(...)` for one ambient turn.

    Faked: the harness, the transcript writer, and the side effects a turn would
    otherwise send to other services. Deliberately NOT faked: the ambient
    decision store in `app.sessions_io` and the sessions route that writes it.
    The bug lives across that seam, so the test crosses it with the real code on
    both sides — a mocked store would pin nothing.

    `drive` returns that turn's `(persisted rows, emitted frames)`, so two turns
    of one session can be told apart without bookkeeping in every caller.
    """
    msg = load_messages_copy(monkeypatch, name="messages_under_ambient_decision_test")
    persisted: list[dict] = []
    frames: list[dict] = []
    script: dict = {}
    meta = tmp_path / f"{SID}.json"
    meta.write_text('{"messages": [], "model": "primary"}')

    async def _append(_sid, entries):
        persisted.extend(entries)

    def _run_query(_harness_messages, _options, **_kw):
        async def _gen():
            if script["decide_body"] is not None:
                # The tool call, arriving mid-turn: the real route, its real
                # gate, its real store write, against the queue that holds this
                # very turn as current.
                await sess_mod.post_ambient_decide(SID, _Request(script["decide_body"]))
            yield {"type": "text_delta", "text": REPLY}
            yield {"type": "assistant_message", "iteration": 1, "text": REPLY,
                   "tool_calls": [],
                   "usage": {"input_tokens": 900, "output_tokens": 20}}
            if script["end_without_result"]:
                # One of the endings the Inner-Voice comment names: no `result`
                # event, so the pop inside that branch never runs.
                raise RuntimeError("engine died before its result event")
            yield {"type": "result", "stop_reason": script["stop_reason"],
                   "num_turns": 1, "duration_ms": 5, "response_text": REPLY,
                   "usage": {"input_tokens": 900, "output_tokens": 20}}
        return _gen()

    async def drive(turn_id: str, *, decide_body: dict | None = None,
                    end_without_result: bool = False, stop_reason: str = "stop"):
        from app.harness.options import RunOptions

        q = sio.SessionQueue()
        q.cancel_event = asyncio.Event()
        turn = sio.SessionTurn(
            turn_id=turn_id, source="ambient",
            payload={"text": "a background signal",
                     "prefetched_text": "a background signal",
                     "model": "primary",
                     "options": RunOptions(model="primary", max_turns=8),
                     "meta_path": meta, "deadline_seconds": 0.0},
            enqueued_at=None,
        )
        # The consumer's own bookkeeping, because the route's gate reads it.
        q.current = turn
        turn.session_id = SID
        sio._session_queues[SID] = q
        first_row, first_frame = len(persisted), len(frames)
        script.update(decide_body=decide_body,
                      end_without_result=end_without_result,
                      stop_reason=stop_reason)
        try:
            await msg._run_turn(SID, turn, q)
        finally:
            while not turn.events.empty():
                frames.append(turn.events.get_nowait())
        return persisted[first_row:], frames[first_frame:]

    with ExitStack() as stack:
        patch_on_copy(stack, msg, [
            ("run_query", _run_query),
            ("_append_messages", _append),
            ("_post_session_capture", AsyncMock()),
            ("_maybe_extract_focus", AsyncMock()),
            ("_fetch_files_changed", AsyncMock(return_value=None)),
            ("load_and_compact_session", _fake_compact),
            ("_prepare_messages_for_harness", _fake_prepare),
            ("prefetch_context_async", _fake_prefetch),
            ("attach_observer_for_turn", _no_observer),
            ("_build_state_anchor", lambda *a, **k: ""),
            ("_event_log", SimpleNamespace(log_event=lambda *a, **k: None)),
            ("set_last_user_session", lambda *a, **k: None),
        ])
        yield drive


def _silent_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("silent")]


def _assistant_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("role") == "assistant"]


def _texts(rows: list[dict]) -> str:
    return " | ".join("".join(p.get("text", "") for p in r.get("content", []))
                      for r in rows)


def _events(frames: list[dict]) -> list[str]:
    return [f["event"] for f in frames]


def test_a_silent_decision_silences_only_the_turn_that_made_it(monkeypatch, tmp_path):
    """Turn A stays quiet because A asked to; turn B, which never asked, speaks.

    A is driven twice, once ending in `result` with `stop_reason: stop` and once
    with `cancelled` — the deferred cancel means a silent turn may end either
    way and the breadcrumb is wanted in both, which is why the consumer checks
    the decision before the cancelled path. B then runs on the same session,
    which is the slot the store is keyed by.

    The turn-id match makes B safe here only because A consumed its own entry.
    The case that actually crossed — a decision the recording turn never
    consumed — is the sibling node below, and that one fails on the shipped
    code. This node is the other half of the clause: the fix must not cost a
    turn that *did* decide its breadcrumb.
    """
    with _ambient_turns(monkeypatch, tmp_path) as drive:
        async def _main():
            stop = await drive("turn-a-stop", decide_body=DECIDE, stop_reason="stop")
            cancelled = await drive("turn-a-cancelled", decide_body=DECIDE,
                                    stop_reason="cancelled")
            b = await drive("turn-b-2599")
            return stop, cancelled, b

        rows_a_stop, rows_a_cancelled, rows_b = asyncio.run(_main())

    for label, (rows, frames) in (("stop", rows_a_stop),
                                  ("cancelled", rows_a_cancelled)):
        silent = _silent_rows(rows)
        assert len(silent) == 1, (
            f"turn A ({label}) was not silenced: its rows are {rows!r}")
        text = _texts(silent)
        assert text.startswith(SILENT_PREFIX) and REASON in text, text
        assert not _assistant_rows(rows), (
            f"turn A ({label}) persisted the reply it decided not to surface")
        assert "ambient_silent" in _events(frames), _events(frames)

    b_rows, b_frames = rows_b
    assert not _silent_rows(b_rows), (
        "turn B was muted by a decision it never made: "
        f"{_texts(_silent_rows(b_rows))!r}")
    assert REPLY in _texts(_assistant_rows(b_rows)), (
        f"the successor's reply did not reach the transcript: {b_rows!r}")
    assert "ambient_silent" not in _events(b_frames), _events(b_frames)


def test_an_ambient_turn_that_never_reached_its_result_event_leaves_its_successor_alone(
        monkeypatch, tmp_path, caplog):
    """The leak that needs no race: the pop sits in the `result` branch.

    A records a silent decision and then dies before any `result` event, so
    nothing pops it — the state the module comment described as a `finally`
    clause. B, the next ambient turn of that session, is the one that finds the
    slot occupied, and after #2599 it is handed nothing: it answers the user,
    the orphan is cleared rather than left for a third turn, and the log names
    both turn ids so a future crossed silencing is readable rather than
    inferred.
    """
    with _ambient_turns(monkeypatch, tmp_path) as drive:
        async def _main():
            a = await drive("turn-a-2599", decide_body=DECIDE, end_without_result=True)
            # A never reached the pop, so the orphan is still there for B to hit.
            # Keyed by session and nothing else: this assertion is the premise,
            # and if a later change ever clears the slot at turn end the test
            # says so instead of quietly testing nothing.
            assert SID in sio._ambient_decisions, (
                "the premise of this test is gone: A's decision no longer "
                f"survives its own turn ({sio._ambient_decisions!r})")
            b = await drive("turn-b-2599")
            return a, b

        with caplog.at_level(logging.WARNING, logger="lloyd-server"):
            (a_rows, a_frames), (b_rows, b_frames) = asyncio.run(_main())

    assert not _silent_rows(a_rows), (
        "A ended without its result event, so it was never muted before this fix "
        "either: the half under test is what its successor does with what A left")
    assert not _silent_rows(b_rows), (
        f"B inherited A's silence: {_texts(_silent_rows(b_rows))!r}")
    assert REPLY in _texts(_assistant_rows(b_rows)), f"B lost its reply: {b_rows!r}"
    assert "ambient_silent" not in _events(b_frames), _events(b_frames)
    assert SID not in sio._ambient_decisions, (
        f"the orphan stayed in the store for a third turn: {sio._ambient_decisions!r}")
    discard = [r.getMessage() for r in caplog.records
               if r.name == "lloyd-server" and r.levelno >= logging.WARNING
               and "ambient" in r.getMessage()]
    assert len(discard) == 1, f"one discard, one warning: {discard}"
    assert "turn-a-2599" in discard[0] and "turn-b-2599" in discard[0], discard[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
