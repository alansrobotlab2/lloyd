"""Resumable Task subagents.

`Task` started from nothing every call. That is right for a fire-and-forget
fan-out and wrong for the case that keeps recurring: a subagent burns its
budget mid-investigation, the caller reads the partial answer, and the only
way to ask a follow-up is to pay for the whole investigation again — a fresh
prompt, a cold KV cache, and no memory of the forty tool results it just
collected.

The two assertions worth reading first are
`test_a_resume_appends_to_the_stored_list_and_passes_it_as_the_handle` (the
loop ignores `messages` when a handle is non-empty, so getting this wrong
silently drops the follow-up) and
`test_an_unanswered_tool_call_is_dropped_before_storing` (the only invalid
shape the loop can leave behind).
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from agent_mcp import _subagent_registry as SR, builtin_task as T


@pytest.fixture(autouse=True)
def _clean():
    SR.reset()
    yield
    SR.reset()


def _store(task_id="sub-t1", **kw):
    base = dict(task_id=task_id, subagent_type="general-purpose",
                profile={"max_turns": 20, "system_prompt": "", "model": "",
                         "base_url": "", "disallowed_tools": []},
                model="primary", base_url="http://x:8096",
                session_id="task:general-purpose:abcd1234",
                description="d",
                chat_messages=[{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "hello"}],
                run_id="sub-1", runs=1)
    base.update(kw)
    SR.store_history(**base)
    return base


# ── the store ───────────────────────────────────────────────────────────────

def test_a_stored_history_can_be_claimed_once():
    _store()
    h = SR.claim_history("sub-t1")
    assert h.task_id == "sub-t1" and h.active is True
    with pytest.raises(SR.HistoryUnavailable) as exc:
        SR.claim_history("sub-t1")
    assert "still running" in str(exc.value)


def test_an_unknown_id_says_so():
    with pytest.raises(SR.HistoryUnavailable) as exc:
        SR.claim_history("nope")
    assert "unknown or evicted" in str(exc.value)


def test_an_expired_history_says_expired_and_is_dropped():
    _store()
    SR._history["sub-t1"].finished_at = time.time() - SR._HISTORY_TTL_S - 1
    with pytest.raises(SR.HistoryUnavailable) as exc:
        SR.claim_history("sub-t1")
    assert "expired" in str(exc.value)
    assert "sub-t1" not in SR._history


def test_release_puts_a_claimed_history_back():
    _store()
    SR.claim_history("sub-t1")
    SR.release_history("sub-t1")
    assert SR.claim_history("sub-t1").task_id == "sub-t1"


def test_the_store_is_bounded_by_count(monkeypatch):
    monkeypatch.setattr(SR, "_HISTORY_KEEP", 2)
    for i in range(4):
        _store(task_id=f"t{i}")
    assert list(SR._history) == ["t2", "t3"]


def test_the_store_is_bounded_by_size(monkeypatch):
    monkeypatch.setattr(SR, "_HISTORY_MAX_CHARS", 50)
    for i in range(4):
        _store(task_id=f"t{i}",
               chat_messages=[{"role": "user", "content": "x" * 30}])
    assert len(SR._history) <= 2
    assert "t3" in SR._history, "the newest must survive"


def test_expired_entries_are_swept_on_the_next_store():
    _store(task_id="old")
    SR._history["old"].finished_at = time.time() - SR._HISTORY_TTL_S - 1
    _store(task_id="new")
    assert "old" not in SR._history and "new" in SR._history


def test_an_unanswered_tool_call_is_dropped_before_storing():
    """The one invalid shape the loop can leave: cancelled or raised between
    the stream ending and the dispatch completing."""
    _store(chat_messages=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
    ])
    assert [m["role"] for m in SR._history["sub-t1"].chat_messages] == ["user"]


def test_an_answered_tool_call_is_kept():
    _store(chat_messages=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ])
    assert len(SR._history["sub-t1"].chat_messages) == 4


def test_a_conversation_that_sanitises_to_nothing_is_not_stored():
    _store(chat_messages=[{"role": "assistant", "content": "",
                           "tool_calls": [{"id": "c1"}]}])
    assert SR._history == {}


def test_history_never_reaches_the_dashboard_row():
    rec = SR.register(subagent_type="general-purpose", description="d", prompt="p",
                      parent_session_id="s", parent_turn_id="t",
                      session_id="task:gp:1", model="primary", max_turns=5)
    row = rec.to_dict()
    assert "chat_messages" not in row
    assert row["task_id"] and row["continuation_of"] == ""


def test_task_id_is_stable_across_runs_and_run_id_is_not():
    first = SR.register(subagent_type="gp", description="d", prompt="p",
                        parent_session_id="s", parent_turn_id="t",
                        session_id="task:gp:1", model="m", max_turns=5)
    second = SR.register(subagent_type="gp", description="d", prompt="p",
                         parent_session_id="s", parent_turn_id="t",
                         session_id="task:gp:1", model="m", max_turns=5,
                         task_id=first.task_id, continuation_of=first.run_id)
    assert second.task_id == first.task_id
    assert second.run_id != first.run_id
    assert second.continuation_of == first.run_id


# ── the tool ────────────────────────────────────────────────────────────────

class _FakeRun:
    """Stands in for run_query: records what it was handed, emits one answer."""

    seen: list = []
    answer = "the answer"

    def __init__(self, messages, options):
        _FakeRun.seen.append({"messages": list(messages),
                              "handle": options.chat_messages_handle,
                              "options": options})
        self._options = options

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        handle = self._options.chat_messages_handle
        if handle is not None:
            handle.append({"role": "assistant", "content": _FakeRun.answer})
        yield {"type": "assistant_message", "text": _FakeRun.answer,
               "thinking": "", "tool_calls": []}
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1}


@pytest.fixture
def fake_run(monkeypatch):
    import app.harness.loop as L
    _FakeRun.seen = []
    _FakeRun.answer = "the answer"
    monkeypatch.setattr(L, "run_query", lambda m, o: _FakeRun(m, o))
    return _FakeRun


async def test_a_fresh_task_returns_a_task_id(fake_run):
    out = json.loads(await T._task({"prompt": "investigate"}))
    assert out["response"] == "the answer"
    assert out["task_id"].startswith("sub-")
    assert out["task_id"] in SR._history


async def test_a_resume_appends_to_the_stored_list_and_passes_it_as_the_handle(fake_run):
    """`run_query` ignores `messages` when the handle is non-empty."""
    first = json.loads(await T._task({"prompt": "investigate"}))
    tid = first["task_id"]

    fake_run.answer = "the follow-up answer"
    second = json.loads(await T._task({"prompt": "and what about X?",
                                       "task_id": tid}))
    assert second["response"] == "the follow-up answer"
    assert second["task_id"] == tid

    call = fake_run.seen[-1]
    assert call["handle"] is not None and len(call["handle"]) > 2
    roles = [m["role"] for m in call["handle"]]
    assert roles[0] == "user" and "and what about X?" in \
        [m.get("content") for m in call["handle"]]
    assert call["messages"] == [], \
        "a resume must not pass `messages` — the loop would ignore them"


async def test_a_resume_keeps_the_session_id_model_and_endpoint(fake_run, monkeypatch):
    first = json.loads(await T._task({"prompt": "go"}))
    stored = SR._history[first["task_id"]]
    stored.model = "secondary"
    stored.base_url = "http://x:8091"

    # A different parent model must not move a half-finished conversation.
    tok = T.current_parent_model.set("primary")
    try:
        await T._task({"prompt": "more", "task_id": first["task_id"]})
    finally:
        T.current_parent_model.reset(tok)
    opts = fake_run.seen[-1]["options"]
    assert opts.model == "secondary" and opts.base_url == "http://x:8091"
    assert opts.session_id == stored.session_id


async def test_subagent_type_is_ignored_on_a_resume(fake_run, caplog):
    first = json.loads(await T._task({"prompt": "go",
                                      "subagent_type": "general-purpose"}))
    await T._task({"prompt": "more", "task_id": first["task_id"],
                   "subagent_type": "something-else"})
    row = SR.list_recent()[0]
    assert row["subagent_type"] == "general-purpose"


async def test_a_resume_opens_a_new_run_row_linked_to_the_old_one(fake_run):
    first = json.loads(await T._task({"prompt": "go"}))
    await T._task({"prompt": "more", "task_id": first["task_id"]})
    rows = SR.list_recent()
    assert len(rows) == 2
    assert rows[0]["task_id"] == rows[1]["task_id"] == first["task_id"]
    assert rows[0]["continuation_of"] == rows[1]["run_id"]


async def test_a_bogus_task_id_is_refused_without_running_anything(fake_run):
    out = json.loads(await T._task({"prompt": "more", "task_id": "sub-nope"}))
    assert "Cannot resume" in out["error"] and "unknown or evicted" in out["error"]
    assert out["task_id"] == "sub-nope"
    assert fake_run.seen == []


async def test_a_failed_run_still_returns_a_resumable_task_id(fake_run, monkeypatch):
    class _NoAnswer(_FakeRun):
        async def _gen(self):
            yield {"type": "tool_call", "name": "Read"}
            yield {"type": "assistant_message", "text": "",
                   "thinking": "", "tool_calls": [{"id": "c1"}]}
            yield {"type": "result", "stop_reason": "max_turns", "num_turns": 20}

    import app.harness.loop as L
    monkeypatch.setattr(L, "run_query", lambda m, o: _NoAnswer(m, o))
    out = json.loads(await T._task({"prompt": "go"}))
    assert "no final answer" in out["error"]
    assert out["task_id"] in SR._history, \
        "a task_id the model is told about must be resumable"


async def test_an_exception_closes_the_row_and_stores_what_there_was(monkeypatch):
    class _Boom:
        def __init__(self, messages, options):
            self._options = options

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            self._options.chat_messages_handle.append(
                {"role": "assistant", "content": "partial"})
            raise RuntimeError("engine gone")
            yield  # pragma: no cover

    import app.harness.loop as L
    monkeypatch.setattr(L, "run_query", lambda m, o: _Boom(m, o))
    out = json.loads(await T._task({"prompt": "go"}))
    assert "Subagent failed" in out["error"] and out["task_id"]
    assert SR.list_recent()[0]["status"] == "error"
    assert out["task_id"] in SR._history


async def test_a_cancel_stores_the_sanitised_list_and_re_raises(monkeypatch):
    class _Cancel:
        def __init__(self, messages, options):
            self._options = options

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            self._options.chat_messages_handle.append(
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]})
            raise asyncio.CancelledError
            yield  # pragma: no cover

    import app.harness.loop as L
    monkeypatch.setattr(L, "run_query", lambda m, o: _Cancel(m, o))
    with pytest.raises(asyncio.CancelledError):
        await T._task({"prompt": "go"})
    assert SR.list_recent()[0]["status"] == "cancelled"
    stored = list(SR._history.values())[0]
    assert [m["role"] for m in stored.chat_messages] == ["user"], \
        "the unanswered tool call must not be replayed on resume"


# ── the schema ──────────────────────────────────────────────────────────────

async def test_the_schema_advertises_task_id_and_states_the_retention():
    tool = (await T.list_tools())[0]
    props = tool.input_schema["properties"]
    assert "task_id" in props
    desc = props["task_id"]["description"]
    assert "30 minutes" in desc and "aggregator restart" in desc
    assert "task_id" in tool.description


async def test_task_still_advertises_no_second_caption_field():
    """The injected `summary` is the caption; a rival field splits the answer."""
    tool = (await T.list_tools())[0]
    assert "description" not in tool.input_schema["properties"]
