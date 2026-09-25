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


# ── D1: a long result is a pointer, not a 2 KB cut ─────────────────────

SID = "20260910_120000_test_aaaa"


@pytest.fixture
def spill_dir(tmp_path, monkeypatch):
    """Point the spill module at a scratch sessions dir; return the session's
    `<sid>.tool-results/` directory there."""
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
    return tmp_path / f"{SID}.tool-results"


def _body(n: int) -> str:
    return "".join(f"line {i:05d} of the file\n" for i in range(n))


def test_a_result_over_the_cap_is_spilled_and_the_row_points_at_the_file(spill_dir):
    from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG
    full = _body(800)                      # ~18 kB: over 2k, under the 50k live spill
    out = te.shape_tool_result_for_transcript(
        full, call_id="c1", session_id=SID, tool_name="Read")
    assert out.startswith(PERSISTED_OUTPUT_TAG)
    path = spill_dir / "c1.txt"
    assert path.read_text() == full, "the file must hold the whole result"
    assert f"Full output saved to: {path}" in out
    assert "Read the full file with the Read tool" in out
    # The row names the file, and `result_chars` measures the pointer.
    row = te.build_tool_result_entry("c1", out, timestamp="T")
    assert row["stats"]["persisted_path"] == str(path)
    assert row["stats"]["result_chars"] == len(out) < len(full)
    assert len(out) < te.TOOL_RESULT_MAX_CHARS + 600


def test_a_result_under_the_cap_is_stored_verbatim(spill_dir):
    short = "x" * te.TOOL_RESULT_MAX_CHARS
    assert te.shape_tool_result_for_transcript(
        short, call_id="c1", session_id=SID) == short
    assert not spill_dir.exists(), "nothing to point at, so nothing written"
    assert "persisted_path" not in te.build_tool_result_entry(
        "c1", short, timestamp="T")["stats"]


def test_an_already_spilled_block_is_stored_whole_not_cut_at_2k(spill_dir):
    from app.harness.tool_result_spill import maybe_spill
    block = maybe_spill("y" * 60_000, tool_name="Grep", tool_use_id="c1",
                        session_id=SID)
    assert len(block) > te.TOOL_RESULT_MAX_CHARS + 14   # the old cut would bite
    out = te.shape_tool_result_for_transcript(
        block, call_id="c1", session_id=SID, tool_name="Grep")
    assert out == block
    assert out.rstrip().endswith("</persisted-output>")
    assert te.persisted_path_of(out) == str(spill_dir / "c1.txt")


def test_a_failed_spill_falls_back_to_the_old_truncation(tmp_path, monkeypatch):
    # A sessions "dir" that is a file: mkdir under it fails, maybe_spill
    # hands the original back, and the row gets the old cut — never 18 kB.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", blocker)
    full = _body(800)
    out = te.shape_tool_result_for_transcript(full, call_id="c1", session_id=SID)
    assert out == te.truncate_tool_result(full)
    assert "persisted_path" not in te.build_tool_result_entry(
        "c1", out, timestamp="T")["stats"]


def test_the_switch_off_and_no_session_both_mean_the_old_cut(spill_dir, monkeypatch):
    full = _body(800)
    assert te.shape_tool_result_for_transcript(
        full, call_id="c1", session_id="") == te.truncate_tool_result(full)
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "compaction", {
        **(CONFIG.get("compaction") or {}),
        "transcript_spill": {"enabled": False}})
    assert te.shape_tool_result_for_transcript(
        full, call_id="c1", session_id=SID) == te.truncate_tool_result(full)
    assert not spill_dir.exists()


def test_the_vault_excerpt_of_a_pointer_is_its_preview_not_its_path(spill_dir):
    out = te.shape_tool_result_for_transcript(
        _body(800), call_id="c1", session_id=SID)
    excerpt = te.tool_result_preview(out)
    assert excerpt.startswith("[full result on disk] line 00000 of the file")
    assert str(spill_dir) not in excerpt[:300]
    assert te.tool_result_preview("plain") == "plain"


def test_deleting_a_session_removes_its_spill_dir(spill_dir, tmp_path, monkeypatch):
    import app.routers.sessions as sess_mod
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path)
    (tmp_path / f"{SID}.json").write_text('{"messages": []}')
    te.shape_tool_result_for_transcript(_body(800), call_id="c1", session_id=SID)
    assert spill_dir.is_dir()
    body = json.loads(asyncio.run(sess_mod.delete_session(SID)).body)
    assert body["deleted"] is True and body["spills_removed"] is True
    assert not spill_dir.exists()
    # A name that is not a plain file name never reaches rmtree.
    body = json.loads(asyncio.run(sess_mod.delete_session("..")).body)
    assert body["spills_removed"] is False


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
        # D1: a second call whose 10k result both writers must turn into the
        # same pointer at the same file.
        {"type": "tool_call", "call_id": "c2", "name": "Read",
         "args_json": '{"file_path": "/f.py"}', "summary": "Reading f.py"},
        {"type": "tool_result", "call_id": "c2", "name": "Read",
         "content": "z\n" * 5_000, "is_error": False, "raw_chars": 10_000},
        # A discarded attempt (X2): both writers must take it back off the
        # final answer, or the row reads "ThreeTwo files.".
        {"type": "thinking_delta", "text": "hmm"},
        {"type": "text_delta", "text": "Three"},
        {"type": "iteration_retry", "reason": "stream_stalled", "attempt": 1,
         "discarded_text_chars": 5, "discarded_thinking_chars": 3},
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
        elif t == "iteration_retry":
            from app.harness.events import trim_discarded
            text, thinking = trim_discarded(text, thinking, evt)
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
            result = te.shape_tool_result_for_transcript(
                evt["content"], call_id=evt["call_id"], session_id=SID,
                tool_name=evt["name"])
            tc = next(c for c in calls if c["call_id"] == evt["call_id"])
            persisted.add(evt["call_id"])
            written.extend(_tool_pair(tc, result_str=result, timestamp="T",
                                      iteration_stats=iteration_stats,
                                      evt=evt, turn_id="turn1"))
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
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
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
    # X3: every row the turn wrote names it, the tool pair included.
    assert all(e.get("turn_id") == "turn1"
               or e.get("thinking", {}).get("turn_id") == "turn1"
               for e in recorded), recorded
    assert recorded[-1]["content"][0]["text"] == "Two files."
    # D1: the 10k result is a pointer on both sides, at the one file.
    pointer = next(e for e in recorded if e.get("tool_call_id") == "c2")
    path = recorder_sessions / "20260910_120000_test_aaaa.tool-results" / "c2.txt"
    assert pointer["stats"]["persisted_path"] == str(path)
    assert path.read_text() == "z\n" * 5_000


def test_rows_carry_the_turn_that_wrote_them_and_omit_it_when_unknown():
    """X3. A reader grouping a turn's rows reads the id off the row instead of
    inferring the boundary from roles. A writer with no turn id writes exactly
    what it wrote before, so every session on disk reads the same."""
    tc = te.build_tool_call("c1", "Bash", "{}")
    assert te.build_tool_call_entry(tc, timestamp="T", turn_id="t9")["turn_id"] == "t9"
    assert te.build_tool_result_entry("c1", "out", timestamp="T",
                                      turn_id="t9")["turn_id"] == "t9"
    assert te.build_user_entry("hi", timestamp="T", turn_id="t9")["turn_id"] == "t9"
    assert "turn_id" not in te.build_tool_call_entry(tc, timestamp="T")
    assert "turn_id" not in te.build_tool_result_entry("c1", "out", timestamp="T")
    assert "turn_id" not in te.build_user_entry("hi", timestamp="T")

    from app.routers.messages import _tool_pair
    pair = _tool_pair(tc, result_str="out", timestamp="T", iteration_stats={},
                      turn_id="t9")
    assert [r["turn_id"] for r in pair] == ["t9", "t9"]
    assert all("turn_id" not in r for r in _tool_pair(
        tc, result_str="out", timestamp="T", iteration_stats={}))


def test_an_assistant_entry_names_the_turn_that_wrote_it():
    """A turn persists one text row per segment between tool calls. The voice
    worker speaks a turn as it streams and must recognise every one of those
    rows afterwards, or the poller that covers typed turns says the reply a
    second time. An empty turn id stays off the row, so a caller with none
    writes exactly what it wrote before."""
    named = te.build_assistant_text_entry("hi", timestamp="T", turn_id="abc123")
    assert named["turn_id"] == "abc123"
    assert "turn_id" not in te.build_assistant_text_entry("hi", timestamp="T")
