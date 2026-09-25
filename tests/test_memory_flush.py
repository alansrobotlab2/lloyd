"""P3 (review 2026-09-24): memory flush before compaction. Ships off.

Pins the five properties the plan names: one flush per compaction cycle once a
turn's peak prompt passes `trigger_fraction` of the threshold; the flush turn
advertises and dispatches only the memory tools (`RunOptions.allowed_tools`);
its rows stay in the transcript and never re-enter history or the persisted
summary's index; the Inner Voice observer does not fire on it; and with the
switch off nothing is enqueued or written. The router half is driven through a
private copy of `app/routers/messages.py` (`tests/_messages_copy.py`).
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from app import compaction_state as CS
from app import memory_flush as MF
from app.compaction import FLUSH_TURN_PREFIX, is_history_row, load_and_compact_session
from app.config import CONFIG
from app.harness.options import RunOptions

SESSION_ID = "20260924_130000_webchat"
THRESHOLD = 200_000
ON = {"enabled": True, "trigger_fraction": 0.85, "max_turns": 6,
      "tools": list(MF.DEFAULT_TOOLS), "once_per_cycle": True}


@pytest.fixture
def flush_on(monkeypatch):
    comp = dict(CONFIG.get("compaction") or {})
    comp["memory_flush"] = dict(ON)
    monkeypatch.setitem(CONFIG, "compaction", comp)


@pytest.fixture
def flush_off(monkeypatch):
    comp = dict(CONFIG.get("compaction") or {})
    comp["memory_flush"] = {**ON, "enabled": False}
    monkeypatch.setitem(CONFIG, "compaction", comp)


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    import app.sessions_io as sio
    monkeypatch.setattr(sio, "SESSIONS_DIR", tmp_path)
    path = tmp_path / f"{SESSION_ID}.json"
    path.write_text(json.dumps({"session_id": SESSION_ID, "messages": [],
                                "last_active": "x"}))
    return path


def _row(i: int, role: str, turn_id: str, text: str = "") -> dict:
    return {"id": f"r{i}", "role": role, "turn_id": turn_id,
            "content": [{"type": "text", "text": text or f"{role} {i}"}]}


# ── one flush per cycle ────────────────────────────────────────────────

def test_one_flush_per_cycle_past_the_pre_threshold(flush_on, sessions):
    enqueued: list[str] = []

    async def _enqueue(tid):
        enqueued.append(tid)

    async def _after(peak, turn_start=None):
        return await MF.after_turn(SESSION_ID, turn_id="t1", turn_start=turn_start,
                                   peak_tokens=peak, threshold=THRESHOLD,
                                   enqueue=_enqueue)

    async def run():
        # Below 0.85 x threshold: nothing.
        assert await _after(int(0.84 * THRESHOLD)) is False
        # Past it: one flush, bookkept under data["compaction"]["flush"].
        assert await _after(int(0.86 * THRESHOLD)) is True
        entry = MF.flush_entry(json.loads(sessions.read_text()))
        assert entry["status"] == "queued" and entry["turn_id"] == enqueued[0]
        assert entry["trigger_tokens"] == int(0.86 * THRESHOLD)
        # Still past it, same cycle: no second flush, queued or done.
        assert await _after(int(0.95 * THRESHOLD)) is False
        await MF.record_done(SESSION_ID, turn_id=enqueued[0],
                             tool_names=["memory_read", "memory_add", "memory_add", "fact_add"],
                             duration_ms=1200, stop_reason="stop")
        data = json.loads(sessions.read_text())
        assert MF.flush_entry(data)["memory_adds"] == 2
        assert MF.flush_entry(data)["fact_adds"] == 1
        assert MF.flushed_this_cycle(data)
        assert await _after(int(0.95 * THRESHOLD)) is False
        # A turn start that summarised closes the cycle; the next one may flush.
        assert await _after(int(0.9 * THRESHOLD), {"summarized": True}) is True

    asyncio.run(run())
    assert len(enqueued) == 2 and all(t.startswith(FLUSH_TURN_PREFIX) for t in enqueued)
    # The key stays after `messages` (X6): the retention sweep reads a prefix.
    keys = list(json.loads(sessions.read_text()))
    assert keys.index("compaction") > keys.index("messages")


def test_a_fold_after_the_flush_opens_a_new_cycle():
    now = time.time()
    data = {"compaction": {"flush": {"at": now - 60, "status": "done"}}}
    assert not MF.should_flush(data, peak_tokens=THRESHOLD, threshold=THRESHOLD, cfg=ON)
    # D2: the summary record updated after the flush.
    data["compaction"]["updated_at"] = now
    assert MF.should_flush(data, peak_tokens=THRESHOLD, threshold=THRESHOLD, cfg=ON)
    # A cancelled (preempted) flush does not hold the cycle closed.
    data = {"compaction": {"flush": {"at": now, "status": "cancelled"}}}
    assert MF.should_flush(data, peak_tokens=THRESHOLD, threshold=THRESHOLD, cfg=ON)
    # Nor does one queued long ago that never reported back.
    data = {"compaction": {"flush": {"at": now - 2 * MF.STALE_QUEUED_SECONDS,
                                     "status": "queued"}}}
    assert MF.should_flush(data, peak_tokens=THRESHOLD, threshold=THRESHOLD, cfg=ON)


# ── the allow-list ─────────────────────────────────────────────────────

def test_flush_turn_advertises_only_the_memory_tools(monkeypatch):
    from app.harness import tool_search_cache
    from app.harness.tests import _replay as R

    engine = R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Bash", command="rm -rf /"),
                           R.tool_call("c2", "ToolSearch", query="bash"),
                           R.tool_call("c3", "vault_write", path="x")]),
        R.Step(tool_calls=[R.tool_call("c4", "memory_add", entry="likes tea",
                                       type="user")]),
        R.Step(text="Saved one entry."),
    ])
    tools = {n: False for n in ("Bash", "Edit", "Write", "Task", "vault_write",
                                "memory_add", "fact_add")}
    tools.update({n: True for n in ("Read", "memory_read", "fact_get")})
    pool = R.ReplayPool(tools)
    R.install(monkeypatch, engine, pool)
    sid = "20260924_130001_flushcache"
    options = RunOptions(model="primary", session_id=sid, max_turns=6,
                         allowed_tools=list(MF.DEFAULT_TOOLS),
                         # Tool search on and a threshold the catalog is past:
                         # the allow-list must still switch it off.
                         tool_search_enabled=True, tool_search_threshold_tools=1)
    evts = asyncio.run(R.drive(options))

    advertised = {t["function"]["name"] for t in engine.requests[0].tools}
    assert advertised == set(MF.DEFAULT_TOOLS)
    for req in engine.requests:
        assert {t["function"]["name"] for t in req.tools} == advertised
    results = {e["call_id"]: e for e in R.of_type(evts, "tool_result")}
    for cid in ("c1", "c2", "c3"):
        assert results[cid]["is_error"], results[cid]
    assert not results["c4"]["is_error"]
    assert [c["name"] for c in pool.calls] == ["memory_add"]
    # The chat's cached loaded set is not replaced by the flush's catalog.
    assert sid not in tool_search_cache._CACHE


def test_no_allow_list_is_the_default_so_task_children_are_unaffected():
    # A Task child builds its own RunOptions (agent_mcp/builtin_task.py); the
    # default is no allow-list at all, not an empty one.
    assert RunOptions(model="primary").allowed_tools is None


# ── transcript yes, history no ─────────────────────────────────────────

def test_flush_rows_kept_in_transcript_dropped_from_history(tmp_path):
    flush = f"{FLUSH_TURN_PREFIX}abc123"
    rows = [
        _row(0, "user", "t1"), _row(1, "assistant", "t1"),
        _row(2, "user", flush, MF.FLUSH_PROMPT),
        {**_row(3, "assistant", flush), "tool_calls": [{"id": "c", "call_id": "c",
         "type": "function", "function": {"name": "memory_add", "arguments": "{}"}}]},
        {"id": "r4", "role": "tool", "turn_id": flush, "tool_call_id": "c",
         "content": [{"type": "text", "text": "ok"}]},
        _row(5, "assistant", flush, "Saved one entry."),
        _row(6, "user", "t2"), _row(7, "assistant", "t2"),
    ]
    path = tmp_path / f"{SESSION_ID}.json"
    path.write_text(json.dumps({"messages": rows}))
    comp = asyncio.run(load_and_compact_session(path, model="primary"))
    history_ids = [m["id"] for m in comp["history"]]
    assert history_ids == ["r0", "r1", "r6", "r7"]
    # Still in the transcript.
    assert len(json.loads(path.read_text())["messages"]) == len(rows)
    # One filter: the persisted summary's index agrees with the history.
    assert [m["id"] for m in CS.conversation_rows(rows)] == history_ids
    assert not any(is_history_row(r) for r in rows[2:6])


def test_the_flush_entry_survives_the_summary_record(tmp_path):
    rows = [_row(i, "user" if i % 2 == 0 else "assistant", "t1") for i in range(4)]
    path = tmp_path / f"{SESSION_ID}.json"
    flush = {"turn_id": f"{FLUSH_TURN_PREFIX}x", "at": 1.0, "status": "done"}
    path.write_text(json.dumps({"messages": rows, "compaction": {"flush": flush}}))
    convo = CS.conversation_rows(rows)
    rec = CS.build_record(summary="s", convo=convo, boundary=2, files_touched=[],
                          covered_turn_ids=["t1"], model="primary")
    assert asyncio.run(CS.save_record(SESSION_ID, rec, path=path))
    data = json.loads(path.read_text())
    assert data["compaction"]["flush"] == flush
    assert "flush" not in rec  # the caller's record is untouched
    loaded = CS.load_record(data)
    assert loaded is not None and CS.validate(loaded, convo) == 2
    # And the fold that closed the cycle reads as one.
    assert not MF.flushed_this_cycle(data)


def test_flushed_before_summary_reaches_the_turn_start_record():
    from app.compaction_record import turn_start_record
    assert turn_start_record({"flushed_before_summary": True})["flushed_before_summary"]
    assert turn_start_record({"tokens_before": 1})["flushed_before_summary"] is False


# ── the observer ───────────────────────────────────────────────────────

def test_observer_does_not_fire(monkeypatch):
    from app.routers import _messages_inner_voice as IV
    monkeypatch.setattr(IV, "_session_iv_flags", lambda sid: (True, True))
    assert IV._iv_should_fire_on_turn("s", "ambient", MF.PRODUCER) is False
    assert IV._iv_should_fire_on_turn("s", "ambient", "producer") is True


# ── the kill switch ────────────────────────────────────────────────────

def test_kill_switch_enqueues_nothing(flush_off, sessions):
    enqueued: list[str] = []

    async def _enqueue(tid):
        enqueued.append(tid)

    before = sessions.read_text()
    ok = asyncio.run(MF.after_turn(SESSION_ID, turn_id="t1",
                                   turn_start={"summarized": True},
                                   peak_tokens=THRESHOLD, threshold=THRESHOLD,
                                   enqueue=_enqueue))
    assert ok is False and enqueued == []
    assert sessions.read_text() == before


def test_config_ships_off():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    block = cfg["compaction"]["memory_flush"]
    assert block["enabled"] is False
    assert set(block["tools"]) == set(MF.DEFAULT_TOOLS)


# ── the router ─────────────────────────────────────────────────────────

async def _drive(tmp_path, monkeypatch, *, turn_id, producer, usage, calls=()):
    from app.sessions_io import SessionQueue, SessionTurn
    from tests._messages_copy import load_messages_copy

    msg = load_messages_copy(monkeypatch, name="messages_under_flush_test")
    monkeypatch.setattr(msg, "SESSIONS_DIR", tmp_path)
    import app.sessions_io as sio
    monkeypatch.setattr(sio, "SESSIONS_DIR", tmp_path)
    logged: list[tuple[str, dict]] = []
    monkeypatch.setattr(msg._event_log, "log_event",
                        lambda sid, name, data, **k: logged.append((name, data)))

    async def _no_observer(*a, **k):
        return None

    monkeypatch.setattr(msg, "attach_observer_for_turn", _no_observer)
    monkeypatch.setattr(msg, "_build_state_anchor", lambda *a, **k: "")
    enqueued: list[tuple[str, str]] = []

    async def _build(session_id, flush_turn_id=""):
        return ("flush-turn", flush_turn_id)

    async def _enqueue(session_id, turn):
        enqueued.append(turn)
        return {}

    monkeypatch.setattr(msg, "build_flush_turn", _build)
    monkeypatch.setattr(msg, "enqueue_ambient", _enqueue)

    async def _fake_run_query(harness_messages, options):
        for i, name in enumerate(calls):
            yield {"type": "tool_call", "call_id": f"c{i}", "name": name,
                   "args_json": "{}", "args_dict": {}, "summary": "saving"}
            yield {"type": "tool_result", "call_id": f"c{i}", "name": name,
                   "content": "ok", "is_error": False}
        yield {"type": "text_delta", "text": "Saved."}
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1,
               "duration_ms": 10, "response_text": "Saved.", "usage": usage}

    monkeypatch.setattr(msg, "run_query", _fake_run_query)

    meta_path = tmp_path / f"{SESSION_ID}.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({"messages": [], "model": "primary"}))
    turn = SessionTurn(
        turn_id=turn_id, source="ambient" if producer else "user",
        payload={"text": "go", "prefetched_text": "go", "model": "primary",
                 "options": RunOptions(model="primary", max_turns=6),
                 "meta_path": meta_path, "deadline_seconds": 0.0,
                 "producer_source": producer},
        enqueued_at=None,
    )
    q = SessionQueue()
    q.cancel_event = asyncio.Event()
    await msg._run_turn(SESSION_ID, turn, q)
    return json.loads(meta_path.read_text()), enqueued, logged


def test_the_result_branch_queues_then_books_the_flush(tmp_path, monkeypatch, flush_on):
    from app.compaction import get_context_window, truncation_threshold
    thr = truncation_threshold(get_context_window("primary"))
    data, enqueued, _ = asyncio.run(_drive(
        tmp_path, monkeypatch, turn_id="turnA", producer="",
        usage={"input_tokens": int(0.9 * thr), "output_tokens": 5}))
    assert len(enqueued) == 1
    flush_id = enqueued[0][1]
    assert flush_id.startswith(FLUSH_TURN_PREFIX)
    assert MF.flush_entry(data)["status"] == "queued"

    data, enqueued, logged = asyncio.run(_drive(
        tmp_path, monkeypatch, turn_id=flush_id, producer=MF.PRODUCER,
        usage={"input_tokens": int(0.9 * thr), "output_tokens": 5},
        calls=("memory_read", "memory_add")))
    assert enqueued == []  # a flush never queues a flush
    entry = MF.flush_entry(data)
    assert entry["status"] == "done" and entry["memory_adds"] == 1
    events = [d for n, d in logged if n == "compaction.memory_flush"]
    assert len(events) == 1
    assert events[0]["turn_id"] == flush_id and events[0]["memory_adds"] == 1
    assert events[0]["trigger_tokens"] == int(0.9 * thr)
    # The flush turn's rows are in the transcript, every one with its id.
    flush_rows = [m for m in data["messages"] if m.get("turn_id") == flush_id]
    assert {m["role"] for m in flush_rows} >= {"user", "assistant", "tool"}


def test_the_result_branch_is_inert_when_off(tmp_path, monkeypatch, flush_off):
    data, enqueued, logged = asyncio.run(_drive(
        tmp_path, monkeypatch, turn_id="turnB", producer="",
        usage={"input_tokens": 10**6, "output_tokens": 5}))
    assert enqueued == [] and "compaction" not in data
    assert not [n for n, _ in logged if n == "compaction.memory_flush"]


def test_build_flush_turn(tmp_path, monkeypatch, flush_on):
    from tests._messages_copy import load_messages_copy

    msg = load_messages_copy(monkeypatch, name="messages_under_flush_build")
    monkeypatch.setattr(msg, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(msg, "build_system_prompt", lambda **k: "SYS")
    (tmp_path / f"{SESSION_ID}.json").write_text(
        json.dumps({"messages": [], "model": "primary"}))
    tid = MF.new_turn_id()
    turn = asyncio.run(msg.build_flush_turn(SESSION_ID, tid))
    opts = turn.payload["options"]
    assert turn.turn_id == tid and turn.source == "ambient"
    assert turn.payload["producer_source"] == MF.PRODUCER
    assert turn.payload["dedup_key"] == MF.PRODUCER
    assert opts.allowed_tools == list(MF.DEFAULT_TOOLS)
    assert opts.tool_search_enabled is False
    assert opts.priority == 1 and opts.max_turns == 6
    assert "memory_add" in turn.payload["text"] and "feedback" in turn.payload["text"]
