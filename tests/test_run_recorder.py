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
  * a failed or timed-out run's record names the pre-images it left — the
    ledger scope and a file count read from that turn's index, zero included —
    because pre-images nobody can find are not an undo (#963);
  * reasoning is persisted as `role="thinking"`, the role that keeps it out of
    every transcript producer in the tree;
  * the kill switch really is one.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

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
# ── Which context policy fired, on the run's OWN usage row (#1078) ─────
#
# The item's own finding: the sessions that actually burn the relief ladder are
# the unattended ones — `20260918_175255_deepresearch_*`, `*_autotriage_*`,
# `*_boardsteward_*` — and they reach `usage_store` through this module, not
# through `messages.py`. A field wired only on the chat path would have recorded
# the mechanism's traffic everywhere except where it fires. These tests drive the
# real relief path and read the real row back.


@pytest.fixture
def usage_db(tmp_path, monkeypatch):
    """An isolated `usage.db`. The recorder writes one on every run, and this
    file's existing tests never asserted on it, so nothing isolated it before."""
    import usage_store

    monkeypatch.setattr(usage_store, "DB_PATH", tmp_path / "usage.db")
    return tmp_path / "usage.db"


def _usage_row(db, session_id: str) -> dict:
    import usage_store

    usage_store._conn().commit()
    rows = list(sqlite3.connect(db).execute(
        "SELECT compaction FROM usage WHERE session_id = ?", (session_id,)))
    assert len(rows) == 1, f"expected one usage row, got {len(rows)}"
    return json.loads(rows[0][0]) if rows[0][0] is not None else None


def _relieve_midrun(session_id: str, *, turn_id: str = "run-threaded",
                    reason: str = "intra_turn") -> dict:
    """Relieve one pressured history the way the loop does, mid-stream.

    Not a hand-built report: `_relieve_context` is the emitter, so the record
    under test was produced by the code that produces relief in production — and
    it is called from *inside* the event feed, so it happens while the run is
    open, which is when a pass actually happens.

    `turn_id` is set on the options the way both real background callers set it —
    `autonomy.py:2886` passes `turn_id=run_id` into `RunOptions`, and
    `workers/sources/_common.py:482` assigns `options.turn_id = run_id` — because
    the emit site reads the turn off `options`, not off the recorder. An earlier
    copy of this helper left it unset and then asserted the event carried no turn,
    which pinned a shape production never produces.
    """
    from app.harness.context_meter import ContextMeter
    from app.harness.loop import _relieve_context
    from app.harness.options import RunOptions

    msgs = [
        {"role": "assistant", "content": f"m{i}", "reasoning": "R" * 40_000,
         "reasoning_content": "R" * 40_000}
        for i in range(12)
    ]
    meter = ContextMeter(262_144)
    meter.observe_usage({"input_tokens": int(262_144 * 0.95)}, len(msgs))
    meter.observe_append(msgs)
    opts = RunOptions(model="primary", session_id=session_id,
                      tool_search_enabled=False)
    opts.turn_id = turn_id
    return _relieve_context(msgs, options=opts, meter=meter, reason=reason)


def test_an_unattended_run_that_relieves_context_records_it_on_its_own_row(
    store, usage_db,
):
    """Clause 4. The relief happens inside the stream, exactly as it does in a
    live worker turn, and the run's usage row carries the rung, the tokens freed
    and the reason — written by `run_recorder`, without `messages.py` anywhere in
    the path.
    """
    sid = "20260924_080000_worker_c0mp"
    captured: dict = {}

    async def _feed():
        yield {"type": "thinking_delta", "text": "working"}
        captured.update(_relieve_midrun(sid))
        yield {"type": "text_delta", "text": "done"}
        yield {"type": "result", "stop_reason": "stop", "num_turns": 2,
               "usage": {"input_tokens": 400_000, "output_tokens": 20}}

    from app.run_recorder import record_events
    from app.sessions_io import create_session

    create_session(sid, platform="worker", model="primary", title="t",
                   source="deep-research")

    async def _go():
        async for _ in record_events(_feed(), session_id=sid, turn_id="run1",
                                     prompt="p", model="primary",
                                     source="deep-research"):
            pass

    asyncio.run(_go())

    record = _usage_row(usage_db, sid)
    assert record is not None, (
        "the run relieved context, so its row is non-NULL; NULL here means the "
        "loop's pass never reached this writer, which is the defect #1078 names"
    )
    assert record["relief_passes"] == 1, record
    assert record["relief_tokens_freed"] == captured["freed_tokens"]
    assert record["relief"][0]["rungs"] == captured["rungs"]
    assert record["relief"][0]["reason"] == "intra_turn"
    assert record["mechanisms"] == ["relief:intra_turn"]


def test_the_relief_event_for_an_unattended_run_names_its_session(
    store, usage_db,
):
    """Both stores, same run.

    The event log is the record that survives a run killed before its `result`
    event — the interesting way an autonomy task ends — and it is the only place
    a killed run's relief passes exist at all. A row and an event that disagree
    would make the two numbers that should corroborate each other a bug report.
    """
    sid = "20260924_080000_worker_c1nt"
    run_id = "run2"

    async def _feed():
        yield {"type": "text_delta", "text": "x"}
        # Same id on both sides, which is what production does: the caller that
        # hands `turn_id` to `record_events` is the one that put it on the
        # RunOptions the relief pass then reads (`autonomy.py:2886`,
        # `workers/sources/_common.py:482`).
        _relieve_midrun(sid, turn_id=run_id, reason="overflow")
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1,
               "usage": {"input_tokens": 400_000}}

    from app.run_recorder import record_events
    from app.sessions_io import create_session

    create_session(sid, platform="worker", model="primary", title="t",
                   source="autotriage")

    async def _go():
        async for _ in record_events(_feed(), session_id=sid, turn_id=run_id,
                                     prompt="p", model="primary",
                                     source="autotriage"):
            pass

    asyncio.run(_go())

    reliefs = [e for e in _events(store)
               if e["event"] == "harness.context_relief"]
    assert len(reliefs) == 1, reliefs
    assert reliefs[0]["session_id"] == sid, (
        "the item's narrower defect: 7,253 relief lines with no session on any of "
        "them, so a firing could not be attributed to anything"
    )
    assert reliefs[0]["turn_id"] == run_id, (
        "the event names the session but not the run, so a worker session with "
        "several runs cannot tell which one burned the ladder; the recorder "
        "doesn't invent this — it comes from the caller's RunOptions, which both "
        "production callers set"
    )

    record = _usage_row(usage_db, sid)
    assert record["relief_passes"] == len(reliefs), (
        "one firing, two stores, one number"
    )


def test_an_unattended_run_that_never_relieves_stores_no_record(
    store, usage_db,
):
    """NULL on this writer means what it means on the chat one: nothing was
    measured, not nothing fired. An autonomy run that never touches the ladder
    must not read as a run whose ladder was disabled — and must not be dropped
    from the row count that answers "how many turns did the stack see".
    """
    sid = "20260924_080000_worker_c2lm"
    _drive([
        {"type": "text_delta", "text": "quiet"},
        {"type": "result", "stop_reason": "stop", "num_turns": 1,
         "usage": {"input_tokens": 1_000}},
    ], session_id=sid)

    assert _usage_row(usage_db, sid) is None
    reliefs = [e for e in _events(store)
               if e["event"] == "harness.context_relief"]
    assert reliefs == []


def test_the_record_survives_a_killed_run_through_the_event_log(
    store, usage_db,
):
    """The half the usage column structurally cannot cover.

    A run killed at its deadline never emits `result`, so `_record_usage` never
    runs and no row exists to populate — the recorder's own docstring names this
    as why it writes incrementally. The event was already written when the pass
    ran, so the firing is still attributable. This is the reason the change is
    two stores and not one.
    """
    sid = "20260924_080000_worker_c3ll"

    async def _feed():
        yield {"type": "text_delta", "text": "x"}
        _relieve_midrun(sid, reason="intra_turn")
        # No `result` event: the run died at its timeout mid-stream.
        return

    from app.run_recorder import record_events
    from app.sessions_io import create_session

    create_session(sid, platform="worker", model="primary", title="t",
                   source="deep-research")

    async def _go():
        async for _ in record_events(_feed(), session_id=sid, turn_id="run3",
                                     prompt="p", model="primary",
                                     source="deep-research"):
            pass

    asyncio.run(_go())

    reliefs = [e for e in _events(store)
               if e["event"] == "harness.context_relief"]
    assert len(reliefs) == 1, (
        "the pass fired and is recorded even though the run never wrote a row"
    )
    assert reliefs[0]["data"]["freed_tokens"] > 0


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


def test_the_paired_tool_row_carries_the_size_from_before_the_shaping(store):
    """`result_chars` is measured on the truncated text, so it saturates at
    2,014 and a spilled 80 KB answer is indistinguishable from a 2 KB one
    (#1052). `raw_chars` is what the harness knew before it spilled, and it
    rides on the event; the recorder's job is to persist it unchanged and
    leave `result_chars` meaning what it always meant. The dispatch half —
    where that number is captured — is pinned in
    `tests/test_tool_result_raw_chars.py`."""
    from app.transcript_entries import TOOL_RESULT_MAX_CHARS

    preview = "<persisted-output> Output too large " + "z" * 3_000
    _drive([
        {"type": "tool_call", "call_id": "c1", "name": "Grep",
         "args_json": '{"pattern": "x"}', "summary": "Searching"},
        {"type": "tool_result", "call_id": "c1", "content": preview,
         "is_error": False, "raw_chars": 81_600},
        {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}},
    ])
    row = next(m for m in _session(store)["messages"] if m["role"] == "tool")
    assert row["stats"]["raw_chars"] == 81_600
    assert row["stats"]["result_chars"] == \
        TOOL_RESULT_MAX_CHARS + len("...(truncated)")
    assert row["stats"]["result_chars"] < row["stats"]["raw_chars"]
    assert row["content"][0]["text"] == preview[:TOOL_RESULT_MAX_CHARS] \
        + "...(truncated)"


def test_an_unpaired_tool_row_omits_raw_chars_rather_than_guessing(store):
    """The result event never arrived, so the pair is rebuilt from the call
    log at the end of the run. That log holds only the truncated string, so
    the true size is unknown here — and `0` would read as "the tool answered
    nothing" and the cap as "it answered exactly 2,014 characters", neither
    of which was ever measured. Absence is the answer, which is also why
    `_unpersisted_pairs` invents no `is_error` either."""
    _drive([
        {"type": "tool_call", "call_id": "c1", "name": "Bash",
         "args_json": '{"command": "sleep 900"}', "summary": "Waiting"},
        {"type": "text_delta", "text": "answer without the result"},
        {"type": "result", "stop_reason": "stop", "num_turns": 2, "usage": {}},
    ])
    row = next(m for m in _session(store)["messages"] if m["role"] == "tool")
    assert "raw_chars" not in row["stats"]
    assert "is_error" not in row["stats"]
    assert row["stats"]["result_chars"] == 0


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

def _stub_autonomy(monkeypatch, tmp_path, task, stream, *, write_records=False):
    """`write_records=True` lets the real `_write_run_record` run against a runs
    directory under `tmp_path`, so a test can assert on the bytes of the record
    somebody opens at 03:00 rather than on the kwargs it would have been handed.
    The default keeps the existing tests reading those kwargs.
    """
    captured: dict = {}

    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "x.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
    if write_records:
        monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "autonomy-runs")
    else:
        monkeypatch.setattr(autonomy, "_write_run_record",
                            lambda **kw: captured.setdefault("record", kw))

    async def _run_query(messages, options):
        captured["options"] = options
        captured["prompt"] = messages[0]["content"]
        events = stream(options) if callable(stream) else stream
        for evt in events:
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


# ── The failure record names its change ledger (#963) ──────────────────
#
# Pre-images have existed for every scheduled run since `ef294bf` (2026-09-10),
# which armed the ledger with `turn_id=run_id` — the test above pins that half.
# What never got built was anything telling a reader they were there. Restoring
# one took the session id, the turn id and the knowledge that a 7-day retention
# window is running, and the run record — the document somebody actually opens
# when a 03:00 run dies mid-write — named none of them. At triage, across 359
# turn indexes and 1,051 entries, `reverted_at` was set on zero.

def _ledger_root(store, monkeypatch):
    """Point the ledger at the tree the sessions live in, as production does."""
    from agent_mcp import _change_ledger as CL
    root = store / "sessions"
    monkeypatch.setattr(CL, "CHANGES_ROOT", root)
    CL.reset()
    return root


def _write_files_under_ledger(options, root, names):
    """Record `names` as this turn's writes, then make the process forget.

    Drives what `agent_mcp/builtin_fs.py` does around a real Edit — `begin`,
    `snapshot_pre` with the bytes read BEFORE the mutation, `commit` with what is
    on disk after — so the count the record reports is the count the index on
    disk holds, not a number this test typed in. Each file is left holding its
    post-image, which is what makes a revert observable as bytes going back.

    The closing `CL.reset()` is the load-bearing part. In production the writes
    are recorded in the aggregator process and read back by the backend, which
    holds nothing but the index on disk; here one process does both, so dropping
    the mirror is what makes the read cross-process instead of a cache hit.
    """
    import os
    from agent_mcp import _change_ledger as CL

    scope = CL.scope(options.session_id, options.turn_id)
    assert scope == (options.session_id, options.turn_id), (
        "if the run does not arm the ledger, nothing below means anything")
    files = []
    for i, name in enumerate(names):
        p = root / "written" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        pre = f"pre-{i} content for {name}\n"
        p.write_text(pre)
        entry = CL.begin(scope, real=os.path.realpath(p), path=os.fspath(p),
                         op="edit", call_id=f"call-{i}")
        CL.snapshot_pre(scope, entry, pre.encode())
        p.write_text(pre + f"appended by run {options.turn_id}\n")
        CL.commit(scope, entry, CL.sha256_bytes(p.read_bytes()))
        files.append(p)
    CL.reset()
    return files


def _plant_an_unreported_write(options, root, name="planted.md"):
    """Append an entry to the turn's index that the run itself never saw.

    The control that proves the reported number is READ rather than asserted. An
    account built from the run's own tally of its tool calls could not contain
    this path, so a record that names it, and counts it, can only have come from
    the index — which is the copy the aggregator flushed and the dying run never
    controlled.
    """
    import os
    p = root / "written" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("recorded by the aggregator, not reported by the run\n")
    index = (root / f"{options.session_id}.changes" / options.turn_id
             / "index.json")
    raw = json.loads(index.read_text())
    raw["entries"].append({"path": os.fspath(p), "real": os.path.realpath(p),
                           "op": "write", "call_id": "call-unreported",
                           "via_session": "", "ts": 0.0, "pre_sha256": "",
                           "post_sha256": "", "writes": 1, "snapshot": "ok",
                           "reverted_at": None})
    index.write_text(json.dumps(raw))
    return p


def _run_record_file(store, task_id, run_id):
    path = store / "autonomy-runs" / str(task_id) / f"{run_id}.md"
    assert path.exists(), f"no run record at {path}"
    return path


def _split_record(text):
    """(front_matter_dict, body) for a run record's bytes."""
    import yaml
    _, fm_text, body = text.split("---\n", 2)
    return yaml.safe_load(fm_text), body


def _index_entry_count(store, session_id, run_id):
    """Entries in that turn's index, read straight off disk — the control."""
    path = (store / "sessions" / f"{session_id}.changes" / run_id
            / "index.json")
    assert path.exists(), f"no change-ledger index at {path}"
    return len(json.loads(path.read_text())["entries"])


def test_a_failed_run_names_its_ledger_scope_and_the_count_read_from_index(
        store, monkeypatch):
    """Clause 1: the scope and the count, and the count READ from the index.

    The distinction only bites on a mismatch, so the run records three writes and
    the index is then given a fourth it never reported. A record assembled from
    the run's own account could only ever say three; the one that says four and
    names the fourth file got its number from disk.
    """
    task = {"id": 42, "name": "Nightly", "skill_name": "s", "status": "up_next",
            "timeout_seconds": 300}
    captured = _stub_autonomy(monkeypatch, store, task, [], write_records=True)
    root = _ledger_root(store, monkeypatch)
    names = ["note-a.md", "note-b.md", "lloyd/MEMORY.md"]

    async def _crash(messages, options):
        captured["options"] = options
        _write_files_under_ledger(options, root, names)
        _plant_an_unreported_write(options, root)
        yield {"type": "text_delta", "text": "writing the notes…"}
        raise RuntimeError("boom mid-write")

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _crash)
    out = asyncio.run(autonomy.run_task(42))
    assert out["success"] is False

    options = captured["options"]
    run_id, session_id = options.turn_id, options.session_id
    fm, body = _split_record(_run_record_file(store, 42, run_id).read_text())
    scope = f"sessions/{session_id}.changes/{run_id}/"

    assert fm["status"] == "failed"
    assert _index_entry_count(store, session_id, run_id) == 4
    assert fm["changes"] == f"{scope} (4 files)"
    assert f"{scope} (4 files)" in body, "the body states it too, not just the yaml"
    # The write the run never reported is counted AND named: the list the record
    # carries is the index's, not the dead run's.
    assert "planted.md" in body
    for name in names:
        assert name in body, f"{name} missing from the record's file list"


def test_a_failed_run_that_recorded_no_writes_states_zero(store, monkeypatch):
    """Clause 2: the zero is written down, not inferred from an absent line.

    "This run touched no files" and "nobody looked" render identically as
    silence, and the difference decides whether a person has anything to undo.
    There is not even an index file on disk for this turn, so the line can only
    come from a reader that treats an absent index and an empty one as the same
    answer: zero.
    """
    task = {"id": 43, "name": "Quiet", "skill_name": "s", "status": "up_next",
            "timeout_seconds": 300}
    captured = _stub_autonomy(monkeypatch, store, task, [], write_records=True)
    root = _ledger_root(store, monkeypatch)

    async def _crash(messages, options):
        captured["options"] = options
        yield {"type": "text_delta", "text": "read everything, wrote nothing"}
        raise RuntimeError("died before writing anything")

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _crash)
    asyncio.run(autonomy.run_task(43))

    options = captured["options"]
    run_id, session_id = options.turn_id, options.session_id
    assert not (root / f"{session_id}.changes" / run_id / "index.json").exists()

    fm, body = _split_record(_run_record_file(store, 43, run_id).read_text())
    scope = f"sessions/{session_id}.changes/{run_id}/"
    assert fm["changes"] == f"{scope} (0 files)"
    assert f"{scope} (0 files)" in body
    assert "no file writes" in body, "and says so in words, not just as a zero"


def test_a_timed_out_run_names_its_ledger_and_nobody_reverted_it(
        store, monkeypatch):
    """Clause 1's timeout half, and the contract that nothing reverts itself.

    A timeout is the scenario the item is written around: notes half-written,
    the deadline fires, and the record reads "(no output before timeout)". The
    record must say what is revertable, and the failure path must leave the
    partial notes alone — a nightly that finished 6 of 9 notes has 6 notes of
    real progress, and an automatic revert would destroy the progress to save an
    invariant. Whether to put them back is a decision, so this reports and stops.

    The absence of a revert is invisible from the record, hence the tripwire:
    the ledger's own revert is watched, and the files are checked on disk.
    """
    from agent_mcp import _change_ledger as CL

    task = {"id": 44, "name": "Slow", "skill_name": "s", "status": "up_next",
            # A one-second budget: the deadline firing is the subject, not how
            # long it takes.
            "timeout_seconds": 1}
    captured = _stub_autonomy(monkeypatch, store, task, [], write_records=True)
    root = _ledger_root(store, monkeypatch)

    revert_calls: list = []
    monkeypatch.setattr(CL, "revert",
                        lambda *a, **k: (revert_calls.append(a), [])[1])

    async def _hang(messages, options):
        captured["options"] = options
        _write_files_under_ledger(options, root, ["kept-1.md", "kept-2.md"])
        yield {"type": "text_delta", "text": "still writing"}
        await asyncio.sleep(30)

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _hang)
    out = asyncio.run(autonomy.run_task(44))
    assert out["status"] == "failed"

    options = captured["options"]
    run_id, session_id = options.turn_id, options.session_id
    fm, body = _split_record(_run_record_file(store, 44, run_id).read_text())
    scope = f"sessions/{session_id}.changes/{run_id}/"
    assert fm["changes"] == f"{scope} (2 files)"
    assert f"{scope} (2 files)" in body
    assert "kept-1.md" in body and "kept-2.md" in body
    assert "revert_run_writes" in body, "the record has to say how to put them back"

    assert revert_calls == [], "a dead run's writes must not be reverted unattended"
    for name in ("kept-1.md", "kept-2.md"):
        assert "appended by run" in (root / "written" / name).read_text(), (
            f"{name} was rolled back; a failed run's partial progress stays")
