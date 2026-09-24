"""One transcript shape, two writers.

`app/routers/messages.py` (the chat path) and `app/run_recorder.py` (every
background run) both persist the same harness events as session messages. The
shapes used to be built inline in the router, so the recorder would have had to
carry a second private copy — and a second private definition of "what a tool
result row looks like" is how the two would come to disagree about the same
turn while both looking correct.

These tests pin the property that makes the split safe: fed the same events,
both writers produce identical entries.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app import transcript_entries as te
from app import turn_usage


# ── Shapes ─────────────────────────────────────────────────────────────

def test_a_tool_call_omits_an_empty_summary_rather_than_storing_one():
    # Every session on disk predates the field and must read the same way.
    assert "summary" not in te.build_tool_call("c1", "Bash", "{}")
    assert te.build_tool_call("c1", "Bash", "{}", "Checking disk")["summary"] \
        == "Checking disk"


def test_a_tool_result_omits_is_error_when_the_writer_does_not_know():
    # The eager path has the event and passes it; the two paths that
    # reconstruct from the call log do not, and inventing False there would
    # turn "we never saw the result" into "the result was fine".
    reconstructed = te.build_tool_result_entry("c1", "out", timestamp="T")
    assert "is_error" not in reconstructed["stats"]
    observed = te.build_tool_result_entry("c1", "out", timestamp="T",
                                          is_error=False)
    assert observed["stats"]["is_error"] is False


def test_a_thinking_entry_keeps_its_content_empty_and_its_role_distinct():
    entry = te.build_thinking_entry("turn1", "thought", 1200, 0, 3, "T")
    assert entry["role"] == "thinking"
    assert entry["content"] == []
    assert entry["reasoning"] == "thought"


def test_an_assistant_entry_drops_a_user_source_and_keeps_any_other():
    assert "source" not in te.build_assistant_text_entry(
        "hi", timestamp="T", source="user")
    assert te.build_assistant_text_entry(
        "hi", timestamp="T", source="ambient")["source"] == "ambient"


def test_a_long_tool_result_is_truncated_with_a_marker():
    out = te.truncate_tool_result("x" * 5000)
    assert len(out) == te.TOOL_RESULT_MAX_CHARS + len("...(truncated)")
    assert out.endswith("...(truncated)")
    assert te.truncate_tool_result("short") == "short"


# ── The two writers agree ──────────────────────────────────────────────

def _events():
    """One iteration: think, call a tool, get a result, answer."""
    return [
        {"type": "system", "session_id": "s", "model": "primary"},
        {"type": "thinking_delta", "text": "let me look"},
        {"type": "thinking_done", "text": "let me look", "duration_ms": 900},
        {"type": "tool_call", "call_id": "c1", "name": "Bash",
         "args_json": '{"command": "ls"}', "summary": "Listing files"},
        {"type": "assistant_message", "text": "", "iteration": 1,
         "tool_calls": [{"call_id": "c1"}],
         "usage": {"input_tokens": 10, "output_tokens": 4},
         "duration_ms": 120},
        # `raw_chars` deliberately differs from the length of `content`: a
        # writer that dropped it and recomputed from the content would still
        # agree with the other one, so agreeing would prove nothing (#1052).
        {"type": "tool_result", "call_id": "c1", "name": "Bash",
         "content": "a\nb", "is_error": False, "raw_chars": 81_234},
        {"type": "text_delta", "text": "Two files."},
        {"type": "result", "stop_reason": "stop", "num_turns": 2,
         "duration_ms": 400, "response_text": "Two files.",
         "usage": {"input_tokens": 20, "output_tokens": 6}},
    ]


def _normalise(entries: list[dict]) -> list[dict]:
    """Strip what is legitimately per-run: wall-clock stamps, uuid ids and the
    per-iteration timings. What is left is the shape under test."""
    out = []
    for e in entries:
        e = json.loads(json.dumps(e))
        e.pop("timestamp", None)
        if not str(e.get("id", "")).startswith(("msg_", "think_")):
            e.pop("id", None)
        stats = e.get("stats")
        if isinstance(stats, dict):
            stats.pop("duration_ms", None)
        out.append(e)
    return out


def _chat_entries(events: list[dict]) -> list[dict]:
    """What `messages.py` writes, calling the same builders it calls inline."""
    written: list[dict] = []
    text = ""
    thinking = ""
    thinking_ms = 0
    thinking_seq = 0
    iteration_stats: dict = {}
    calls: list[dict] = []
    persisted: set[str] = set()
    for evt in events:
        t = evt["type"]
        if t == "text_delta":
            text += evt["text"]
        elif t == "thinking_delta":
            thinking += evt["text"]
        elif t == "thinking_done":
            thinking, thinking_ms = evt["text"], int(evt["duration_ms"])
            written.append(te.build_thinking_entry(
                "turn1", thinking, thinking_ms, thinking_seq, 1, "T"))
            thinking_seq += 1
            thinking, thinking_ms = "", 0
        elif t == "assistant_message":
            usage = evt["usage"]
            iteration_stats = {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cache_read": 0, "cache_create": 0,
                "duration_ms": evt["duration_ms"],
                "iteration": evt["iteration"], "model": "primary",
            }
            if evt.get("tool_calls") and text.strip():
                written.append(te.build_assistant_text_entry(
                    text, timestamp="T", stats=dict(iteration_stats), turn_id="turn1"))
                text = ""
        elif t == "tool_call":
            calls.append(te.build_tool_call(
                evt["call_id"], evt["name"], evt["args_json"], evt["summary"]))
        elif t == "tool_result":
            # The chat half of the parity claim goes through the router's own
            # `_tool_pair`, not a hand-copy of it: a field one writer starts
            # emitting and the other does not fails here, which is the drift
            # this file exists to catch. The router's reconstruct-from-log
            # branch is the `evt=None` call below, in the error path.
            from app.routers.messages import _tool_pair
            result = te.truncate_tool_result(evt["content"])
            tc = next(c for c in calls if c["call_id"] == evt["call_id"])
            persisted.add(evt["call_id"])
            written.extend(_tool_pair(tc, result_str=result, timestamp="T",
                                      iteration_stats=iteration_stats,
                                      evt=evt))
        elif t == "result":
            usage = evt["usage"]
            # The same mapper `messages.py` applies inline (#859), so this
            # mirror cannot drift from the writer it mirrors again: the turn
            # row now carries the summed pair alongside the peak pair, and
            # hand-copying the key list is how it would have missed it.
            usage_row = turn_usage.turn_usage_row(usage)
            stats = {
                "input_tokens": usage_row["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cache_read": usage_row["cache_read"], "cache_create": 0,
                "prompt_tokens_sum": usage_row["prompt_tokens_sum"],
                "cache_read_sum": usage_row["cache_read_sum"],
                "duration_ms": evt["duration_ms"],
                "num_turns": evt["num_turns"], "model": "primary",
                # Prefix-miss accounting (app/prefix_miss.py): one
                # iteration counts nothing, so the turn measures zero.
                "reprefill_tokens": 0, "prefix_misses": 0,
            }
            written.append(te.build_assistant_text_entry(
                text, timestamp="T", stats=stats, source="background", turn_id="turn1"))
    return written


@pytest.fixture
def recorder_sessions(tmp_path, monkeypatch):
    """Point the recorder's session store at a scratch directory."""
    import app.sessions_io as sio
    monkeypatch.setattr(sio, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    return tmp_path


def _recorder_entries(events, sessions_dir) -> list[dict]:
    from app.run_recorder import record_events
    from app.sessions_io import create_session

    async def _feed():
        for evt in events:
            yield evt

    create_session("20260910_120000_test_aaaa", platform="autonomy",
                   model="primary", title="t", source="autonomy")

    async def _drive():
        async for _ in record_events(
                _feed(), session_id="20260910_120000_test_aaaa",
                turn_id="turn1", model="primary", source="background"):
            pass

    asyncio.run(_drive())
    data = json.loads((sessions_dir / "20260910_120000_test_aaaa.json").read_text())
    return data["messages"]


def test_both_writers_produce_identical_entries(recorder_sessions):
    """The property the split rests on. If this fails, the chat transcript and
    the background transcript have started describing the same turn
    differently — which is the whole failure mode `transcript_entries.py`
    exists to make impossible."""
    events = _events()
    recorded = _recorder_entries(events, recorder_sessions)
    chat = _chat_entries(events)
    assert _normalise(recorded) == _normalise(chat)


def test_an_assistant_entry_names_the_turn_that_wrote_it():
    """A turn persists one text row per segment between tool calls. The voice
    worker speaks a turn as it streams and must recognise every one of those
    rows afterwards, or the poller that covers typed turns says the reply a
    second time. An empty turn id stays off the row, so a caller with none
    writes exactly what it wrote before."""
    named = te.build_assistant_text_entry("hi", timestamp="T", turn_id="abc123")
    assert named["turn_id"] == "abc123"
    assert "turn_id" not in te.build_assistant_text_entry("hi", timestamp="T")
