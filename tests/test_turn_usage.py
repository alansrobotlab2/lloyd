"""One fold of the turn's harness telemetry, used by both usage writers (P11).

`app/turn_usage.py::TurnTelemetry` turns a turn's events into the telemetry
columns of its `usage.db` row. The chat router and the background recorder
both use it; the test here drives each writer over the SAME event stream and
requires the two rows to agree column for column, which is the property the
class exists for. The router half runs through a private copy of
`app/routers/messages.py` (`tests/_messages_copy.py` says why a copy).
"""

from __future__ import annotations

import asyncio
import json

import usage_store
from app.harness import events as E
from app.harness import telemetry
from app.turn_usage import TurnTelemetry

TEL_COLS = [c for c, _ in usage_store._TELEMETRY_COLUMNS]
SESSION_ID = "20260924_120000_webchat"


def _events() -> list[dict]:
    return [
        E.assistant_message(
            text="", tool_calls=[{"id": "c1"}, {"id": "c2"}], iteration=1,
            usage={"input_tokens": 1000, "output_tokens": 50,
                   "cache_read": 800, "reasoning_tokens": 20},
            finish_reason="tool_calls", ttft_ms=400, request_ms=900,
            cache_ratio=0.8),
        E.tool_call(call_id="c1", name="Read", args_json="{}", args_dict={}),
        E.tool_result(call_id="c1", name="Read", content="ok", duration_ms=30),
        E.tool_call(call_id="c2", name="Bash", args_json="{}", args_dict={}),
        E.tool_result(call_id="c2", name="Bash", content="Tool call denied: x",
                      is_error=True, error_class="denied"),
        E.assistant_message(
            text="done", tool_calls=[], iteration=2,
            usage={"input_tokens": 1100, "output_tokens": 10,
                   "cache_read": 1000, "reasoning_tokens": 5},
            ttft_ms=120, request_ms=300, cache_ratio=0.91),
        E.text_delta("done"),
        E.result(stop_reason="stop", num_turns=2, duration_ms=1500,
                 response_text="done",
                 usage={"input_tokens": 1100, "output_tokens": 60,
                        "cache_read": 1000, "reasoning_tokens": 25}),
    ]


async def _stream(evts):
    for evt in evts:
        if evt["type"] == "result":
            # What D7's retry and D5's raised gate do from inside the loop:
            # write a harness event, which the writer's tally must see.
            telemetry.log_harness_event("", "harness.stream_retried", {})
            telemetry.log_harness_event("", "harness.hook_raised", {})
        yield evt


def _rows() -> list[dict]:
    conn = usage_store._conn()
    return [dict(r) for r in conn.execute(
        f"SELECT session_id, {', '.join(TEL_COLS)} FROM usage ORDER BY id")]


async def _drive_router(tmp_path, monkeypatch):
    from app.harness.options import RunOptions
    from app.sessions_io import SessionQueue, SessionTurn
    from tests._messages_copy import load_messages_copy

    msg = load_messages_copy(monkeypatch, name="messages_under_telemetry_test")
    monkeypatch.setattr(msg, "SESSIONS_DIR", tmp_path)
    import app.sessions_io as sio
    monkeypatch.setattr(sio, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(msg._event_log, "log_event", lambda *a, **k: None)

    async def _no_observer(*a, **k):
        return None

    monkeypatch.setattr(msg, "attach_observer_for_turn", _no_observer)
    monkeypatch.setattr(msg, "_build_state_anchor", lambda *a, **k: "")
    monkeypatch.setattr(msg, "run_query", lambda _m, _o: _stream(_events()))

    meta_path = tmp_path / f"{SESSION_ID}.json"
    meta_path.write_text(json.dumps({"messages": [], "model": "primary"}))
    turn = SessionTurn(
        turn_id="turnT", source="user",
        payload={"text": "go", "prefetched_text": "go", "model": "primary",
                 "options": RunOptions(model="primary", max_turns=60),
                 "meta_path": meta_path, "deadline_seconds": 0.0},
        enqueued_at=None,
    )
    q = SessionQueue()
    q.cancel_event = asyncio.Event()
    await msg._run_turn(SESSION_ID, turn, q)


async def _drive_recorder(tmp_path, monkeypatch):
    import app.sessions_io as sio
    from app import run_recorder

    monkeypatch.setattr(sio, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(run_recorder._event_log, "log_event", lambda *a, **k: None)
    async for _ in run_recorder.record_events(
            _stream(_events()), session_id="20260924_120001_worker_ab12",
            turn_id="run1", prompt="go", model="primary", source="worker"):
        pass


def test_turn_telemetry_row_is_the_same_from_both_writers(tmp_path, monkeypatch):
    asyncio.run(_drive_router(tmp_path, monkeypatch))
    asyncio.run(_drive_recorder(tmp_path, monkeypatch))
    rows = _rows()
    assert len(rows) == 2, rows
    chat, background = ({k: r[k] for k in TEL_COLS} for r in rows)
    assert chat == background
    assert chat == {
        "stop_reason": "stop",
        "reasoning_tokens": 25,
        "ttft_ms_first": 400,
        "ttft_ms_max": 400,
        "tool_calls": 2,
        "tool_errors": 1,
        "tool_errors_by_class": json.dumps({"denied": 1}),
        "tool_ms_total": 30,
        "stream_retries": 1,
        "overflow_recoveries": 0,
        "hook_raised": 1,
        "wrapped_up": 0,
    }


def test_unmeasured_is_null_not_zero():
    tel = TurnTelemetry()
    tel.note(E.assistant_message(text="x", tool_calls=[], usage={}))
    tel.note(E.tool_result(call_id="c", name="Bash", content="no",
                           is_error=True, error_class="denied"))
    row = tel.row()
    assert row["ttft_ms_first"] is None and row["ttft_ms_max"] is None
    assert row["reasoning_tokens"] is None
    assert row["tool_ms_total"] is None      # the deny never reached MCP
    assert row["stop_reason"] is None and row["wrapped_up"] is None
    assert row["tool_errors_by_class"] == {"denied": 1}


def test_partial_row_is_peak_prompt_and_summed_output():
    """What D12 books for a turn that raised before its `result` event."""
    tel = TurnTelemetry()
    tel.note(E.assistant_message(text="", tool_calls=[], usage={
        "input_tokens": 500, "output_tokens": 40, "cache_read": 400}))
    tel.note(E.assistant_message(text="", tool_calls=[], usage={
        "input_tokens": 900, "output_tokens": 60, "cache_read": 950,
        "cache_create": 7}))
    assert tel.partial_row() == {
        "input_tokens": 900, "output_tokens": 100, "cache_read": 900,
        "cache_create": 7, "num_turns": 2}


def test_a_new_turn_counts_from_zero():
    first = TurnTelemetry()
    telemetry.log_harness_event("", "harness.overflow_recovered", {})
    second = TurnTelemetry()
    telemetry.log_harness_event("", "harness.overflow_recovered", {})
    assert first.row()["overflow_recoveries"] == 1
    assert second.row()["overflow_recoveries"] == 1
