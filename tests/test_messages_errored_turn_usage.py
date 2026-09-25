"""D12: a turn that ends without its `result` event still books its tokens.

Only `_run_turn`'s `result` handler filled `stream_stats`, and the error arm
booked a row only when `stream_stats` held tokens — so a turn that raised after
two iterations, or whose task was cancelled, spent real prompt tokens and wrote
nothing to `usage.db`. The error arm now falls back to the running totals
`TurnTelemetry` folds from every `assistant_message` (peak prompt, summed
output), says `stop_reason="error"` on the row and on the `done` frame, and a
cancelled task saves its partial text the way the `cancel_event` path does.

Driven through a private copy of the router (`tests/_messages_copy.py`), because
`tests/test_session_queue.py` rebinds the live module's `_run_turn`.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import usage_store


def _copy(monkeypatch):
    from tests._messages_copy import load_messages_copy

    return load_messages_copy(monkeypatch, name="messages_under_errored_turn_test")


def _assistant(iteration: int, input_tokens: int, output_tokens: int) -> dict:
    return {"type": "assistant_message", "iteration": iteration, "text": "",
            "tool_calls": [], "usage": {"input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "cache_read": input_tokens // 2}}


async def _drive(monkeypatch, tmp_path, *, events, raise_with: BaseException,
                 session_id: str = "20260924_120000_webchat",
                 persisted: list | None = None, frames: list | None = None):
    """Run the real `_run_turn` over a harness that yields `events` and then
    raises `raise_with`. Returns (persisted rows, emitted frames); both are
    also appended to the caller's lists, so a test whose turn re-raises a
    cancel can still read them."""
    from app.harness.options import RunOptions
    from app.sessions_io import SessionQueue, SessionTurn

    msg = _copy(monkeypatch)
    monkeypatch.setattr(msg._event_log, "log_event", lambda *a, **k: None)

    async def _no_observer(*a, **k):
        return None

    async def _compact(*a, **k):
        return {"history": [], "truncated": False, "tokens_before": 0,
                "tokens_after": 0, "context_window": 262144, "threshold": 0}

    async def _prepare(history, model=None, **k):
        return list(history)

    persisted = [] if persisted is None else persisted
    frames = [] if frames is None else frames

    async def _append(sid, entries):
        persisted.extend(entries)

    async def _run_query(harness_messages, options):
        for evt in events:
            yield evt
        raise raise_with

    monkeypatch.setattr(msg, "attach_observer_for_turn", _no_observer)
    monkeypatch.setattr(msg, "_build_state_anchor", lambda *a, **k: "")
    monkeypatch.setattr(msg, "load_and_compact_session", _compact)
    monkeypatch.setattr(msg, "_prepare_messages_for_harness", _prepare)
    monkeypatch.setattr(msg, "_append_messages", _append)
    monkeypatch.setattr(msg, "run_query", _run_query)

    turn = SessionTurn(
        turn_id="d12", source="user",
        payload={"text": "go", "prefetched_text": "go", "model": "primary",
                 "options": RunOptions(model="primary", max_turns=60),
                 "meta_path": tmp_path / f"{session_id}.json",
                 "deadline_seconds": 0.0},
        enqueued_at=None,
    )
    q = SessionQueue()
    q.cancel_event = asyncio.Event()
    try:
        await msg._run_turn(session_id, turn, q)
    finally:
        while not turn.events.empty():
            frames.append(turn.events.get_nowait())
    return persisted, frames


def _rows(session_id: str = "20260924_120000_webchat"):
    return usage_store._conn().execute(
        "SELECT input_tokens, output_tokens, cache_read, num_turns, duration_ms, "
        "stop_reason FROM usage WHERE session_id = ? ORDER BY id",
        (session_id,)).fetchall()


def _done(frames):
    done = [f for f in frames if f and f.get("event") == "done"]
    assert len(done) == 1, f"expected one done frame, got {frames}"
    return done[0]["data"]


def test_a_turn_that_raises_after_two_iterations_books_peak_input_and_summed_output(
        monkeypatch, tmp_path):
    asyncio.run(_drive(monkeypatch, tmp_path, raise_with=RuntimeError("engine gone"),
                       events=[{"type": "text_delta", "text": "half"},
                               _assistant(1, 1000, 10),
                               _assistant(2, 3000, 20)]))
    rows = _rows()
    assert len(rows) == 1, f"one errored turn, one row: {[dict(r) for r in rows]}"
    row = rows[0]
    # Peak prompt, not a sum: the same units a finished turn's row carries.
    assert row["input_tokens"] == 3000
    assert row["output_tokens"] == 30
    assert row["cache_read"] == 1500
    assert row["num_turns"] == 2
    assert row["duration_ms"] is not None and row["duration_ms"] >= 0
    assert row["stop_reason"] == "error"


def test_an_errored_turn_with_no_text_still_books_and_says_error(monkeypatch, tmp_path):
    """The no-content branch emits an `error` frame, not `done`, and books too."""
    _, frames = asyncio.run(_drive(
        monkeypatch, tmp_path, raise_with=RuntimeError("boom"),
        events=[_assistant(1, 2000, 5)]))
    assert [f["event"] for f in frames if f and f.get("event") in ("done", "error")] == ["error"]
    rows = _rows()
    assert len(rows) == 1 and rows[0]["stop_reason"] == "error"
    assert rows[0]["input_tokens"] == 2000


def test_the_done_payload_of_an_errored_turn_says_error(monkeypatch, tmp_path):
    persisted, frames = asyncio.run(_drive(
        monkeypatch, tmp_path, raise_with=RuntimeError("x" * 1000),
        events=[{"type": "text_delta", "text": "partial answer"},
                _assistant(1, 1000, 10)]))
    data = _done(frames)
    assert data["stop_reason"] == "error"
    assert data["error"] == "x" * 300, "the error string is clipped to 300 chars"
    assert data["response"] == "partial answer"
    assert data["stats"]["input_tokens"] == 1000, (
        "the stats persisted and sent must carry the running totals, not zeros")
    texts = [e for e in persisted if e.get("role") == "assistant"]
    assert texts and not texts[-1].get("cancelled")


def test_a_cancelled_turn_saves_its_partial_text_and_books_usage(monkeypatch, tmp_path):
    """The consumer task cancelled mid-turn (not `cancel_event`, which ends in
    a `result`): the partial text is saved marked cancelled, the tokens are
    booked `cancelled`, a `done(cancelled)` frame goes out, and the cancel
    still propagates."""
    persisted: list = []
    frames: list = []
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_drive(
            monkeypatch, tmp_path, raise_with=asyncio.CancelledError(),
            persisted=persisted, frames=frames,
            events=[{"type": "text_delta", "text": "cut short"},
                    _assistant(1, 5000, 40)]))
    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["stop_reason"] == "cancelled"
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (5000, 40)

    texts = [e for e in persisted if e.get("role") == "assistant"]
    assert texts, f"the partial text was not saved: {json.dumps(persisted)[:400]}"
    assert texts[-1]["cancelled"] is True
    assert texts[-1]["content"][0]["text"] == "cut short"
    data = _done(frames)
    assert data["cancelled"] is True and data["stop_reason"] == "cancelled"


def test_a_turn_is_booked_once_when_the_result_handler_raises_after_booking(
        monkeypatch, tmp_path):
    """`_fetch_files_changed` raising inside the `result` handler lands the
    turn in the error arm with `stream_stats` already full; the row the
    handler wrote is the only one."""
    msg = _copy(monkeypatch)

    async def _boom(*a, **k):
        raise OSError("aggregator unreachable")

    monkeypatch.setattr(msg, "_fetch_files_changed", _boom)

    async def _go():
        return await _drive(
            monkeypatch, tmp_path, raise_with=RuntimeError("unreached"),
            events=[{"type": "text_delta", "text": "answer"},
                    _assistant(1, 1000, 10),
                    {"type": "result", "stop_reason": "stop", "num_turns": 1,
                     "duration_ms": 5, "response_text": "answer",
                     "usage": {"input_tokens": 1000, "output_tokens": 10}}])

    _, frames = asyncio.run(_go())
    rows = _rows()
    assert len(rows) == 1, f"booked twice: {[dict(r) for r in rows]}"
    assert rows[0]["stop_reason"] == "stop"
    assert _done(frames)["stop_reason"] == "error"


def test_the_background_recorder_books_an_interrupted_run_once(monkeypatch):
    """`run_recorder`'s error path: running totals, stop_reason by how it ended,
    and never a second row after `result`."""
    from app import run_recorder as rr

    monkeypatch.setattr(rr, "_append_messages", _async_noop)
    monkeypatch.setattr(rr._event_log, "log_event", lambda *a, **k: None)

    async def _events(raise_with):
        yield _assistant(1, 1000, 10)
        yield _assistant(2, 2500, 15)
        raise raise_with

    async def _consume(sid, raise_with):
        async for _ in rr.record_events(_events(raise_with), session_id=sid,
                                        turn_id="t", model="primary"):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(_consume("20260924_120001_worker_aaaa", RuntimeError("x")))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_consume("20260924_120002_worker_bbbb", asyncio.CancelledError()))

    err = _rows("20260924_120001_worker_aaaa")
    assert len(err) == 1
    assert (err[0]["input_tokens"], err[0]["output_tokens"], err[0]["num_turns"],
            err[0]["stop_reason"]) == (2500, 25, 2, "error")
    cancelled = _rows("20260924_120002_worker_bbbb")
    assert len(cancelled) == 1 and cancelled[0]["stop_reason"] == "cancelled"

    async def _finished():
        yield _assistant(1, 1000, 10)
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1,
               "duration_ms": 5, "usage": {"input_tokens": 1000, "output_tokens": 10}}

    async def _consume_ok():
        async for _ in rr.record_events(_finished(), session_id="20260924_120003_worker_cccc",
                                        turn_id="t", model="primary"):
            pass

    asyncio.run(_consume_ok())
    assert len(_rows("20260924_120003_worker_cccc")) == 1


async def _async_noop(*a, **k):
    return None


def test_a_worker_reads_an_errored_done_as_no_completion_and_keeps_the_error():
    """`run_prompt_in_session`'s callers read `stop_reason is None` as "the
    harness never completed" (autocode's infra_failed); an errored turn has
    always read that way there, and the D12 `error` string now reaches
    `errors`. Pinned on the source, since driving it needs a live backend."""
    import inspect

    from workers.sources import _common

    src = inspect.getsource(_common.run_prompt_in_session)
    assert 'if out["stop_reason"] == "error":' in src
    assert 'out["stop_reason"] = None' in src
    assert 'out["errors"].append(str(data["error"])' in src
