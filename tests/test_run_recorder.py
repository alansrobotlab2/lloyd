"""Every background run leaves a record.

Three paths run an agent loop here and until 2026-09-10 only one of them wrote
anything down. `autonomy.run_task` and `_common.run_prompt_on_primary` called
`run_query` directly and kept the text; the tool calls they made existed
nowhere afterwards. That is why, when an autonomy task became the prime
suspect in a full vault wipe, it could be neither confirmed nor cleared.

What is pinned here:

  * an autonomy run writes a session, a transcript and an event log, and the
    run record names the session so the two can be joined;
  * `run_task` sets `session_id` AND `turn_id` on its `RunOptions`, which is
    what switches on the per-turn change ledger — an unattended run is the one
    that most needs an undo and it was the only path without one;
  * a run killed mid-stream still leaves the transcript up to the moment it
    died, which is the case the module exists for;
  * reasoning is persisted as `role="thinking"`, the role that keeps it out of
    every transcript producer in the tree;
  * the kill switch really is one.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import autonomy


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Sessions and event logs under tmp_path."""
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    return tmp_path


def _session(store_dir):
    files = sorted((store_dir / "sessions").glob("*.json"))
    assert len(files) == 1, f"expected one session, got {[f.name for f in files]}"
    return json.loads(files[0].read_text())


def _events(store_dir):
    files = sorted((store_dir / "events").glob("*.events.jsonl"))
    assert files, "no event log written"
    return [json.loads(line) for line in
            files[0].read_text().splitlines() if line.strip()]


# ── The recorder itself ────────────────────────────────────────────────

def _drive(events, *, session_id="20260910_120000_worker_aaaa",
           prompt="do the thing", **kw):
    from app.run_recorder import record_events
    from app.sessions_io import create_session

    async def _feed():
        for evt in events:
            yield evt

    create_session(session_id, platform="worker", model="primary",
                   title="t", source="gap-fill")

    async def _run():
        seen = []
        async for evt in record_events(_feed(), session_id=session_id,
                                       turn_id="turn1", prompt=prompt,
                                       model="primary", source="worker", **kw):
            seen.append(evt)
        return seen

    return asyncio.run(_run())


def test_the_recorder_re_yields_every_event_unchanged(store):
    """It is a passthrough. The caller's loop must see exactly what
    `run_query` produced — the recorder is a second reader of the stream, not
    a new owner of it."""
    events = [
        {"type": "text_delta", "text": "hello"},
        {"type": "result", "stop_reason": "stop", "num_turns": 1,
         "usage": {"input_tokens": 3, "output_tokens": 1}},
    ]
    seen = _drive(events)
    assert seen == events


def test_a_worker_run_writes_a_readable_transcript(store):
    events = [
        {"type": "thinking_delta", "text": "hm"},
        {"type": "thinking_done", "text": "hm, check disk", "duration_ms": 800},
        {"type": "tool_call", "call_id": "c1", "name": "Bash",
         "args_json": '{"command": "df -h"}', "summary": "Checking disk"},
        {"type": "assistant_message", "iteration": 1,
         "tool_calls": [{"call_id": "c1"}], "usage": {}, "duration_ms": 5},
        {"type": "tool_result", "call_id": "c1", "content": "80% used",
         "is_error": False},
        {"type": "text_delta", "text": "Disk is at 80%."},
        {"type": "result", "stop_reason": "stop", "num_turns": 2,
         "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]
    _drive(events)
    msgs = _session(store)["messages"]
    roles = [m["role"] for m in msgs]
    # prompt, reasoning, the tool pair, the answer — in the order they happened.
    assert roles == ["user", "thinking", "assistant", "tool", "assistant"]
    assert msgs[0]["content"][0]["text"] == "do the thing"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "Bash"
    assert msgs[2]["tool_calls"][0]["summary"] == "Checking disk"
    assert msgs[3]["content"][0]["text"] == "80% used"
    assert msgs[4]["content"][0]["text"] == "Disk is at 80%."

    kinds = [e["event"] for e in _events(store)]
    assert "brain1.query_started" in kinds
    assert "brain1.tool_call_proposed" in kinds
    assert "brain1.tool_result_received" in kinds
    assert "brain1.result_message" in kinds


def test_reasoning_is_persisted_under_the_role_that_hides_it(store):
    """`role="thinking"` is what keeps reasoning out of the vault exporter,
    the titler, session recall and the trajectory extractor — every one of
    them branches on role and none has a case for this one. See
    `tests/test_thinking_trace_transcripts.py` for the measurement."""
    _drive([
        {"type": "thinking_done", "text": "a private thought", "duration_ms": 10},
        {"type": "text_delta", "text": "answer"},
        {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}},
    ])
    thinking = [m for m in _session(store)["messages"] if m["role"] == "thinking"]
    assert len(thinking) == 1
    assert thinking[0]["reasoning"] == "a private thought"
    # Second layer: nothing in `content`, against a future producer that walks
    # content blocks without checking role.
    assert thinking[0]["content"] == []


def test_a_run_killed_mid_stream_still_leaves_what_it_had(store):
    """The #60 case, and the reason nothing is buffered until the `result`
    event: a run that ends badly is the run someone will want to read."""
    from app.run_recorder import record_events
    from app.sessions_io import create_session

    sid = "20260910_120000_autonomy_bbbb"
    create_session(sid, platform="autonomy", model="primary", title="t",
                   source="autonomy-task:80")

    async def _feed():
        yield {"type": "tool_call", "call_id": "c1", "name": "Bash",
               "args_json": '{"command": "sleep 900"}', "summary": "Waiting"}
        yield {"type": "tool_result", "call_id": "c1", "content": "ok"}
        yield {"type": "text_delta", "text": "partial finding: "}
        await asyncio.sleep(60)      # the run's deadline lands here
        yield {"type": "result", "stop_reason": "stop", "num_turns": 9}

    async def _run():
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                async for _ in record_events(_feed(), session_id=sid,
                                             turn_id="run_80_x",
                                             prompt="check the vault",
                                             source="autonomy"):
                    pass
        # The shielded flush completes after we stop waiting on it.
        await asyncio.sleep(0.05)

    asyncio.run(_run())
    msgs = json.loads((store / "sessions" / f"{sid}.json").read_text())["messages"]
    texts = [m["content"][0]["text"] for m in msgs if m["content"]]
    assert "check the vault" in texts          # the prompt
    assert "ok" in texts                       # the tool result
    assert "partial finding: " in texts        # the text it had reached
    partial = [m for m in msgs if m.get("cancelled")]
    assert partial and partial[0]["content"][0]["text"] == "partial finding: "
    assert any(e["event"] == "background.run_interrupted" for e in _events(store))


def test_the_kill_switch_stops_recording_without_stopping_the_run(store,
                                                                 monkeypatch):
    monkeypatch.setattr("app.config.CONFIG",
                        {"harness": {"background_recording": {"enabled": False}}})
    events = [{"type": "text_delta", "text": "hi"},
              {"type": "result", "stop_reason": "stop", "num_turns": 1}]
    seen = _drive(events)
    assert seen == events                       # the run is unaffected
    assert _session(store)["messages"] == []    # and nothing was written


# ── The other direct path ──────────────────────────────────────────────

def test_run_prompt_on_primary_leaves_a_session_and_a_transcript(store,
                                                                 monkeypatch):
    """The second path that recorded nothing. A `gap-fill` or
    `session-distill` run collected its text and discarded everything else."""
    from workers.sources import _common as C

    seen: dict = {}

    async def _fake_run_query(messages, options):
        seen["options"] = options
        yield {"type": "tool_call", "call_id": "c1", "name": "Read",
               "args_json": '{"file_path": "x"}', "summary": "Reading x"}
        yield {"type": "tool_result", "call_id": "c1", "content": "contents"}
        yield {"type": "text_delta", "text": "resolved"}
        yield {"type": "result", "stop_reason": "stop", "num_turns": 2,
               "usage": {"input_tokens": 4, "output_tokens": 2}}

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _fake_run_query)
    monkeypatch.setattr(C, "_worker_run_options",
                        lambda *a, **k: type("O", (), {"session_id": "",
                                                       "turn_id": ""})())

    turn = asyncio.run(C.run_prompt_on_primary(
        "resolve the gap", max_turns=5, source="gap-fill",
        title="gap-fill Anthropic"))

    assert turn.text == "resolved"
    assert turn.session_id
    data = _session(store)
    assert data["platform"] == "worker" and data["source"] == "gap-fill"
    # Recorded, and still not observed — the observer is wired in the chat
    # endpoint and this path does not go through it.
    assert data["inner_voice"] is False
    assert data["title"] == "gap-fill Anthropic"
    assert [m["role"] for m in data["messages"]] == [
        "user", "assistant", "tool", "assistant"]
    # And both ids reach the harness, which is what arms the change ledger.
    assert seen["options"].session_id == data["session_id"]
    assert seen["options"].turn_id
    assert any(e["event"] == "brain1.result_message" for e in _events(store))


# ── The autonomy path ──────────────────────────────────────────────────

def _stub_autonomy(monkeypatch, tmp_path, task, stream):
    captured: dict = {}

    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "x.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
    monkeypatch.setattr(autonomy, "_write_run_record",
                        lambda **kw: captured.setdefault("record", kw))

    async def _run_query(messages, options):
        captured["options"] = options
        captured["prompt"] = messages[0]["content"]
        for evt in stream:
            yield evt

    class Opts:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    import app.harness as harness
    import app.harness.mcp_pool as mcp_pool
    monkeypatch.setattr(harness, "run_query", _run_query)
    monkeypatch.setattr(harness, "RunOptions", Opts)
    monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")
    return captured


def test_an_autonomy_run_is_recorded_and_joined_to_its_run_record(store,
                                                                 monkeypatch):
    task = {"id": 80, "name": "OKF Conformance Check",
            "skill_name": "okf-check", "status": "up_next",
            "timeout_seconds": 300}
    captured = _stub_autonomy(monkeypatch, store, task, [
        {"type": "tool_call", "call_id": "c1", "name": "Bash",
         "args_json": '{"command": "validate_okf.py"}', "summary": "Validating"},
        {"type": "tool_result", "call_id": "c1", "content": "0 violations"},
        {"type": "text_delta", "text": "Conformance holds."},
        {"type": "result", "stop_reason": "stop", "num_turns": 2, "usage": {}},
    ])

    out = asyncio.run(autonomy.run_task(80))
    assert out["success"] is True

    data = _session(store)
    assert data["platform"] == "autonomy"
    assert data["source"] == "autonomy-task:80"
    # Titled at creation — a background session never queues behind the
    # single-tenant secondary for a label nobody asked for.
    assert data["title"].startswith("#80 ")
    assert data["inner_voice"] is False
    tool_rows = [m for m in data["messages"] if m["role"] == "tool"]
    assert tool_rows and tool_rows[0]["content"][0]["text"] == "0 violations"

    # The join. Two records of the same run with nothing connecting them is
    # what made a suspect run unreviewable.
    assert captured["record"]["extra"]["session_id"] == data["session_id"]


def test_run_task_sets_both_ids_so_the_change_ledger_arms(store, monkeypatch):
    """`turn_id` is what `agent_mcp/_change_ledger.py` keys on. Without it a
    scheduled task's file writes leave no pre-images and cannot be reverted —
    and an unattended run is the one that most needs an undo."""
    task = {"id": 42, "name": "Nightly", "skill_name": "s", "status": "up_next",
            "timeout_seconds": 300}
    captured = _stub_autonomy(monkeypatch, store, task, [
        {"type": "text_delta", "text": "done"},
        {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}},
    ])
    asyncio.run(autonomy.run_task(42))
    options = captured["options"]
    assert options.session_id == _session(store)["session_id"]
    assert options.turn_id.startswith("run_42_")
    assert captured["record"]["run_id"] == options.turn_id


def test_a_timed_out_autonomy_run_still_names_its_transcript(store, monkeypatch):
    """The run whose record reads "(no output before timeout)" is exactly the
    one somebody needs to open — and the one that had nothing to open."""
    task = {"id": 80, "name": "OKF", "skill_name": "s", "status": "up_next",
            # A one-second budget: the point is the deadline firing, not how
            # long it takes to fire.
            "timeout_seconds": 1}
    stream = [
        {"type": "tool_call", "call_id": "c1", "name": "Bash",
         "args_json": "{}", "summary": "Checking"},
        {"type": "tool_result", "call_id": "c1", "content": "partial"},
    ]
    captured = _stub_autonomy(monkeypatch, store, task, stream)

    async def _slow(messages, options):
        captured["options"] = options
        for evt in stream:
            yield evt
        await asyncio.sleep(30)

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _slow)

    async def _record(task, task_id, run_id, started_at, started_dt, **kw):
        captured["failure"] = kw
        return {"success": False, "status": "failed", "task_id": task_id}
    monkeypatch.setattr(autonomy, "_record_failure", _record)

    async def _go():
        result = await autonomy.run_task(80)
        await asyncio.sleep(0.05)     # let the shielded flush land
        return result

    out = asyncio.run(_go())
    assert out["status"] == "failed"
    assert captured["failure"]["extra"]["timeout"] is True
    assert captured["failure"]["extra"]["session_id"] == _session(store)["session_id"]
    rows = _session(store)["messages"]
    assert any(m["role"] == "tool" and m["content"][0]["text"] == "partial"
               for m in rows), [m["role"] for m in rows]
