"""A discarded iteration is taken back off what the writers accumulated (X2).

Nothing emits `iteration_retry` yet — the stream retry (D7) and the echo
guard's `tool_choice` mode (P6a) will. What lands first is the contract they
rely on: the event carries how many text and thinking characters of the
thrown-away attempt were already yielded as deltas, and every consumer that
builds a transcript row out of deltas trims exactly that many. Without it a
retried iteration persists as the dead attempt's prefix glued onto the live
one's answer.

The router half is driven through a private copy of `app/routers/messages.py`
(`tests/_messages_copy.py` says why a copy), over a stubbed `run_query`, so the
branch under test is the one a live turn executes.
"""

from __future__ import annotations

import asyncio
import json

from app.harness import events as E


# ── The event and the trim ─────────────────────────────────────────────

def test_the_event_carries_its_counts_and_clamps_nonsense_to_zero():
    evt = E.iteration_retry(reason="stream_stalled", attempt=1,
                            discarded_text_chars=5,
                            discarded_thinking_chars=-3)
    assert evt["type"] == "iteration_retry"
    assert evt["reason"] == "stream_stalled" and evt["attempt"] == 1
    assert evt["discarded_text_chars"] == 5
    assert evt["discarded_thinking_chars"] == 0


def test_trim_takes_only_the_discarded_tail():
    evt = E.iteration_retry(reason="r", attempt=1, discarded_text_chars=5,
                            discarded_thinking_chars=3)
    assert E.trim_discarded("keep Wrong", "ok hmm", evt) == ("keep ", "ok ")


def test_trim_never_goes_below_empty():
    # A thinking phase that reached `thinking_done` was flushed and cleared
    # already, so the buffer can hold less than the attempt yielded.
    evt = E.iteration_retry(reason="r", attempt=1, discarded_text_chars=50,
                            discarded_thinking_chars=50)
    assert E.trim_discarded("abc", "", evt) == ("", "")


def test_zero_counts_leave_the_buffers_alone():
    evt = E.iteration_retry(reason="r", attempt=1)
    assert E.trim_discarded("abc", "def", evt) == ("abc", "def")


# ── The chat router ────────────────────────────────────────────────────

SESSION_ID = "20260924_120000_webchat"


async def _drive(tmp_path, monkeypatch, events):
    from app.harness.options import RunOptions
    from app.sessions_io import SessionQueue, SessionTurn
    from tests._messages_copy import load_messages_copy

    msg = load_messages_copy(monkeypatch, name="messages_under_retry_test")
    monkeypatch.setattr(msg, "SESSIONS_DIR", tmp_path)
    # The rows are written by `sessions_io._append_messages`, which resolves
    # its own module global.
    import app.sessions_io as sio
    monkeypatch.setattr(sio, "SESSIONS_DIR", tmp_path)
    logged: list[tuple[str, dict]] = []
    monkeypatch.setattr(msg._event_log, "log_event",
                        lambda sid, name, data, **k: logged.append((name, data)))

    async def _no_observer(*a, **k):
        return None

    monkeypatch.setattr(msg, "attach_observer_for_turn", _no_observer)
    monkeypatch.setattr(msg, "_build_state_anchor", lambda *a, **k: "")

    async def _fake_run_query(harness_messages, options):
        for evt in events:
            yield evt

    monkeypatch.setattr(msg, "run_query", _fake_run_query)

    meta_path = tmp_path / f"{SESSION_ID}.json"
    meta_path.write_text(json.dumps({"messages": [], "model": "primary"}))
    turn = SessionTurn(
        turn_id="turnR",
        source="user",
        payload={
            "text": "answer me",
            "prefetched_text": "answer me",
            "model": "primary",
            "options": RunOptions(model="primary", max_turns=60),
            "meta_path": meta_path,
            "deadline_seconds": 0.0,
        },
        enqueued_at=None,
    )
    q = SessionQueue()
    q.cancel_event = asyncio.Event()
    await msg._run_turn(SESSION_ID, turn, q)
    frames = []
    while not turn.events.empty():
        frames.append(turn.events.get_nowait())
    data = json.loads(meta_path.read_text())
    return data["messages"], frames, logged


def _result(text: str) -> dict:
    return {"type": "result", "stop_reason": "stop", "num_turns": 1,
            "duration_ms": 10, "response_text": text,
            "usage": {"input_tokens": 100, "output_tokens": 5}}


def test_the_router_persists_only_the_retried_attempt(tmp_path, monkeypatch):
    """The dead attempt's text and thinking are trimmed before any row is
    built from them, the browser is told with a `retry` frame, and the event
    log records the retry."""
    events = [
        {"type": "thinking_delta", "text": "hmm"},
        {"type": "text_delta", "text": "Wrong"},
        E.iteration_retry(reason="stream_stalled", attempt=1,
                          discarded_text_chars=5, discarded_thinking_chars=3),
        {"type": "text_delta", "text": "Right"},
        # `response_text` empty so the row is built from the router's own
        # buffer — the thing the trim acts on.
        _result(""),
    ]
    rows, frames, logged = asyncio.run(_drive(tmp_path, monkeypatch, events))

    answers = [r for r in rows if r["role"] == "assistant"]
    assert answers, rows
    assert answers[-1]["content"][0]["text"] == "Right"
    assert not answers[-1].get("reasoning"), answers[-1]

    retry = [f for f in frames if f["event"] == "retry"]
    assert len(retry) == 1
    assert retry[0]["data"]["discarded_text_chars"] == 5
    assert retry[0]["data"]["reason"] == "stream_stalled"
    assert retry[0]["data"]["turn_id"] == "turnR"
    assert [d for n, d in logged if n == "harness.iteration_retry"]


def test_the_user_row_names_its_turn(tmp_path, monkeypatch):
    """X3: the prompt row carries the turn id every other row of the turn does."""
    rows, _frames, _logged = asyncio.run(
        _drive(tmp_path, monkeypatch,
               [{"type": "text_delta", "text": "ok"}, _result("ok")]))
    users = [r for r in rows if r["role"] == "user"]
    assert users and users[0]["turn_id"] == "turnR"
