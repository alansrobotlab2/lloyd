"""Steering and killing a RUNNING `Task` child (#411).

A subagent used to be write-once: nothing could push new information into it
and nothing could stop it. The harness seam already exists —
`RunOptions.chat_messages_handle` is read by the loop *by reference*, and
`RunOptions.cancel_event` is checked by the loop at `loop.py:226` (iteration
boundary), `:405` (before tool dispatch) and `:1591-1650` (racing an
in-flight MCP call). What `Task` never did was hand either one to anyone: the
list is a local in `_task` whose only escaping reference is `store_history()`
inside `_close` — after the run — and `options.cancel_event` was never set.

Everything here that claims something about a *running* child runs the real
`app.harness.loop.run_query`:

  * the child's tool surface is faked (`_build_pool`), so a stop request is
    issued from outside the child's own code, mid tool dispatch, which is
    where killing a wedged subagent matters;
  * the vLLM engine is faked (`stream_chat`) and captures every request body,
    so "the child saw it" is asserted on the messages the model is actually
    sent — the way `app/harness/tests/test_loop_inject_ordering.py` asserts an
    Inner Voice inject;
  * nothing asserts a `stop_reason` a fake invented. The loop's own checks
    produce it, or the test fails.

The five clauses: the published handle reaching the next request body; the
refusal path; `options.cancel_event` stopping a real run while `_close` still
closes and stores; resume-after-cancel; and the framing that keeps an
injection from reading as the child's own turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

import pytest

import app.harness.loop as loop_mod
from agent_mcp import _subagent_registry as SR
from agent_mcp import _task_registry
from agent_mcp import builtin_task as T
from agent_mcp._subagent_registry import (
    STEERING_POLICY,
    CallerScope,
    SteeringRefused,
)

PARENT = "sess-parent-1"
OTHER = "sess-someone-else"

INJECT_TEXT = "the file is at /x, not /y"
INJECT_BODY = "[ORCHESTRATOR] the file is at /x, not /y"

_PROFILE = {"system_prompt": "sys", "max_turns": 20, "disallowed_tools": [],
            "model": "primary", "base_url": ""}


@pytest.fixture(autouse=True)
def _clean():
    SR.reset()
    yield
    SR.reset()


@pytest.fixture(autouse=True)
def _reset_cache():
    from app.harness import tool_search_cache
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


@contextlib.contextmanager
def bound_session(session_id: str = PARENT, turn_id: str = "turn-1"):
    """Act as the turn that spawned the child.

    These two contextvars are what `main.call_tool` binds from the request
    `_meta`; setting them is what a real dispatch does. (The two-dispatch test
    at the bottom goes through `main.call_tool` itself, so nothing is
    simulated there.)
    """
    stok = _task_registry.current_session_id.set(session_id)
    ttok = _task_registry.current_turn_id.set(turn_id)
    try:
        yield CallerScope(session_id=session_id, turn_id=turn_id)
    finally:
        _task_registry.current_session_id.reset(stok)
        _task_registry.current_turn_id.reset(ttok)


def _row(*, chat_messages=None, cancel_event="auto", parent=PARENT, **over):
    """An active registry row, published with the two handles by default."""
    kwargs = dict(
        subagent_type="general-purpose",
        description="probe",
        prompt="do the thing",
        parent_session_id=parent,
        parent_turn_id="turn-1",
        session_id="task:general-purpose:abcd",
        model="primary",
        max_turns=20,
        chat_messages=[{"role": "system", "content": "sys"},
                       {"role": "user", "content": "do the thing"}]
        if chat_messages is None else chat_messages,
        cancel_event=asyncio.Event() if cancel_event == "auto" else cancel_event,
    )
    kwargs.update(over)
    return SR.register(**kwargs)


def _caller(session_id: str = PARENT, turn_id: str = "turn-1") -> CallerScope:
    return CallerScope(session_id=session_id, turn_id=turn_id)


def _contents(messages: list[dict]) -> list[str]:
    return [str(m.get("content") or "") for m in messages]


def _injected(messages: list[dict]) -> list[dict]:
    return [m for m in messages
            if str(m.get("content", "")).startswith("[ORCHESTRATOR]")]


# ── 1. the running row publishes the live handles ───────────────────────────

def test_a_running_row_publishes_the_live_handle_and_the_cancel_event():
    chat, ev = [{"role": "user", "content": "go"}], asyncio.Event()
    rec = _row(chat_messages=chat, cancel_event=ev)

    assert SR.active_run(rec.task_id) is rec
    # THE list, not a copy — the loop reads this same object.
    assert SR.active_run(rec.task_id).chat_messages is chat
    assert SR.active_run(rec.task_id).cancel_event is ev


def test_the_dashboard_row_says_a_run_is_steerable_without_serialising_it():
    """`snapshot()` is a JSON surface (`main.py` → Mission Control).

    An `asyncio.Event` is not serialisable and a conversation is unbounded, so
    the row carries the capability, never the handle.
    """
    _row()
    row = SR.list_active()[0]
    assert row["steerable"] is True
    assert row["cancellable"] is True
    assert row["injected"] == 0
    assert "chat_messages" not in row and "cancel_event" not in row
    json.dumps(SR.snapshot())  # must not raise


def test_a_run_with_no_published_handle_is_not_steerable():
    rec = _row(cancel_event=None)
    rec.chat_messages = None
    with pytest.raises(SteeringRefused):
        SR.steer(rec.task_id, "new info", caller=_caller())


# ── 2. authorised append, and the refusal path ──────────────────────────────

def test_an_authorised_append_lands_in_the_childs_live_list_with_fixed_framing():
    chat = [{"role": "user", "content": "go"}]
    rec = _row(chat_messages=chat)

    msg = SR.steer(rec.task_id, "  the endpoint is :8091, not :8096  ",
                   caller=_caller())

    assert msg == {"role": "user",
                   "content": "[ORCHESTRATOR] the endpoint is :8091, not :8096"}
    assert chat[-1] is msg
    assert rec.injected == 1
    # Clause 5: an append must never read as the child's own assistant turn.
    assert msg["role"] != "assistant"


def test_an_append_from_another_session_adds_nothing_and_is_logged(caplog):
    chat = [{"role": "user", "content": "go"}]
    rec = _row(chat_messages=chat)

    with caplog.at_level(logging.WARNING, logger="lloyd-subagent-registry"):
        with pytest.raises(SteeringRefused) as exc:
            SR.steer(rec.task_id, "attack", caller=_caller(session_id=OTHER))

    assert chat == [{"role": "user", "content": "go"}], "refused means NOT appended"
    assert rec.injected == 0
    assert "refused" in caplog.text
    assert OTHER in caplog.text and PARENT in caplog.text
    # The refusal names the rule that fired, so it stays auditable against a
    # policy a person later changes.
    assert STEERING_POLICY in caplog.text and STEERING_POLICY in str(exc.value)


def test_an_unbound_caller_cannot_steer_fail_closed(caplog):
    """A caller with no session — a worker, a stray script — is refused.

    Authority is checked against the orchestrating session; with nothing to
    compare against, the only safe answer is no.
    """
    chat = [{"role": "user", "content": "go"}]
    _row(chat_messages=chat)
    with caplog.at_level(logging.WARNING, logger="lloyd-subagent-registry"):
        with pytest.raises(SteeringRefused):
            SR.steer(SR.list_active()[0]["task_id"], "hi",
                     caller=CallerScope(session_id=""))
    assert len(chat) == 1
    assert "refused" in caplog.text


def test_a_task_with_no_live_run_cannot_be_steered():
    with pytest.raises(SteeringRefused):
        SR.steer("sub-nope", "hi", caller=_caller())


def test_an_empty_injection_is_refused_rather_than_appended():
    chat = [{"role": "user", "content": "go"}]
    _row(chat_messages=chat)
    with pytest.raises(SteeringRefused):
        SR.steer(SR.list_active()[0]["task_id"], "   ", caller=_caller())
    assert len(chat) == 1


def test_cancelling_sets_the_childs_event():
    ev = asyncio.Event()
    rec = _row(cancel_event=ev)
    assert ev.is_set() is False
    SR.cancel_run(rec.task_id, caller=_caller(), reason="went wrong")
    assert ev.is_set() is True


def test_cancelling_from_another_session_is_refused_and_never_sets(caplog):
    ev = asyncio.Event()
    _row(cancel_event=ev)
    with caplog.at_level(logging.WARNING, logger="lloyd-subagent-registry"):
        with pytest.raises(SteeringRefused):
            SR.cancel_run(SR.list_active()[0]["task_id"],
                          caller=_caller(session_id=OTHER))
    assert ev.is_set() is False
    assert "refused" in caplog.text


def test_a_finished_run_is_not_steerable_or_cancellable():
    chat = [{"role": "user", "content": "go"}]
    rec = _row(chat_messages=chat)
    SR.finish(rec, status="completed", stop_reason="stop")
    with pytest.raises(SteeringRefused):
        SR.steer(rec.task_id, "late", caller=_caller())
    with pytest.raises(SteeringRefused):
        SR.cancel_run(rec.task_id, caller=_caller())


# ── the real loop ──────────────────────────────────────────────────────────

class _FakePool:
    """The child's tool surface.

    `during_call` runs *inside* a tool dispatch — the one moment outside the
    child's own code where a stop request is worth anything, because it is
    what a wedged subagent looks like from the outside. The loop checks
    `options.cancel_event` at the top of the next iteration, so a stop issued
    here is enforced by the loop, not by this pool.
    """

    def __init__(self, during_call=None) -> None:
        self._discovered = [("lloyd-mcp", [{
            "name": "Read",
            "description": "read a file",
            "inputSchema": {"type": "object", "properties": {}},
        }])]
        self._during = during_call

    @property
    def discovered(self):
        return self._discovered

    async def call_tool(self, name: str, args: dict, *, session_id: str = "", **_kw):
        if self._during is not None:
            await self._during(session_id)
        return {"content": f"FAKE_RESULT[{name}]", "is_error": False}


class _StreamScript:
    """Replays scripted vLLM responses and captures every request body.

    `on_request` runs at the moment the child asks for a completion — the
    instant a parent-side append is still in time to be seen, which is the
    race the feature lives in.
    """

    def __init__(self, turns, on_request=None):
        self.turns = turns
        self.captured: list[list[dict]] = []
        self._on_request = on_request

    def __call__(self, **kwargs):
        self.captured.append([dict(m) for m in (kwargs.get("messages") or [])])
        if self._on_request is not None:
            self._on_request(len(self.captured))
        text, tool_calls = self.turns[len(self.captured) - 1]
        return self._gen(text, tool_calls)

    async def _gen(self, text: str, tool_calls: list[dict]):
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments") or {}),
                },
            }]}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class _Harness:
    def __init__(self) -> None:
        # `RunOptions` of every child the loop was asked to run, captured at
        # `_build_pool` — the last place the options are seen before the loop
        # consumes them.
        self.options: list = []
        # async callable(sub_session_id), run inside a child's tool dispatch
        self.during_call = None


@pytest.fixture
def harness(monkeypatch):
    """Real `run_query`; fake pool and engine; deterministic profile.

    The child's handle and cancel event are read off the captured options, so
    the identity assertions are about what `Task` itself built, not a handle
    the test pre-populated.
    """
    h = _Harness()

    async def _build_pool(options):
        h.options.append(options)
        return _FakePool(during_call=h.during_call)

    monkeypatch.setattr(loop_mod, "_build_pool", _build_pool)
    monkeypatch.setattr(T, "_load_subagent_profile", lambda t: dict(_PROFILE))
    return h


async def test_an_append_through_the_registry_reaches_the_childs_next_request(harness, monkeypatch):
    """Clause 1 — mirrors test_loop_inject_ordering.py:135 for a Task child.

    Started by `Task` itself: the handle the append goes into is the one
    `_task` built and handed to the loop, and the claim is checked on the
    following iteration's real request body.
    """
    seen: dict = {}

    def on_request(n: int):
        if n != 1:
            return
        rec = SR.active_run(SR.list_active()[0]["task_id"])
        opts = harness.options[0]
        # The row publishes THE objects the child's options hold.
        seen["handle_is_childs"] = rec.chat_messages is opts.chat_messages_handle
        seen["cancel_is_childs"] = rec.cancel_event is opts.cancel_event
        SR.steer(rec.task_id, INJECT_TEXT, caller=_caller())

    script = _StreamScript(
        [("Reading it:", [{"id": "c1", "name": "Read", "arguments": {}}]),
         ("Done.", [])],
        on_request=on_request,
    )
    monkeypatch.setattr(loop_mod, "stream_chat", script)

    with bound_session():
        out = json.loads(await T._task({"prompt": "read /y"}))

    assert seen == {"handle_is_childs": True, "cancel_is_childs": True}
    assert len(script.captured) == 2, script.captured
    first, second = script.captured
    # Not in the prompt: the child saw it because it was appended mid-run.
    assert INJECT_BODY not in _contents(first), first
    assert INJECT_BODY in _contents(second), second
    assert [m["role"] for m in _injected(second)] == ["user"]
    # And the child kept working afterwards.
    assert out["response"] == "Done."


async def test_cancelling_mid_run_stops_the_real_loop_and_still_closes_and_stores(
    harness, monkeypatch,
):
    """Clause 3 — the stop comes from the loop's own check, not from a fake.

    The stop request is issued inside a tool dispatch, outside the child's
    code. `loop.py:226` is then the only thing that can end the run, so
    `stop_reason == "cancelled"` here is the loop's, and the child asking for
    no second completion is the proof it enforced it.
    """
    async def stop_mid_tool(sub_session_id: str):
        tid = next(r["task_id"] for r in SR.list_active())
        SR.steer(tid, INJECT_TEXT, caller=_caller())
        SR.cancel_run(tid, caller=_caller(), reason="wedged")

    harness.during_call = stop_mid_tool
    script = _StreamScript(
        [("Looking:", [{"id": "c1", "name": "Read", "arguments": {}}]),
         ("MUST NOT BE REACHED", [])],
    )
    monkeypatch.setattr(loop_mod, "stream_chat", script)

    with bound_session():
        out = json.loads(await asyncio.wait_for(T._task({"prompt": "go"}), 20))

    assert len(script.captured) == 1, \
        "the loop asked for a second completion, so cancellation did not stop it"
    assert out["stop_reason"] == "cancelled"
    assert "cancelled" in out["error"]
    assert out["task_id"]

    row = SR.list_recent()[0]
    assert row["status"] == "cancelled" and row["stop_reason"] == "cancelled"
    assert row["injected"] == 1, "the row records that it was steered"
    # `_close` closes the row AND stores history on every exit path — a
    # cancelled child has to be resumable, not lost.
    assert out["task_id"] in SR._history
    stored = SR._history[out["task_id"]].chat_messages
    assert INJECT_BODY in _contents(stored)
    # Clause 5, on a real run's stored conversation: the injection is framed
    # as an outside message, never as the child's own assistant turn.
    assert [m["role"] for m in _injected(stored)] == ["user"]


async def test_a_cancelled_child_resumes_with_the_injection_in_its_next_request(
    harness, monkeypatch,
):
    """Clause 4 — resume after cancel, checked on the resumed request body."""
    async def stop_mid_tool(sub_session_id: str):
        tid = next(r["task_id"] for r in SR.list_active())
        SR.steer(tid, INJECT_TEXT, caller=_caller())
        SR.cancel_run(tid, caller=_caller())

    harness.during_call = stop_mid_tool
    first = _StreamScript([("Looking:", [{"id": "c1", "name": "Read",
                                          "arguments": {}}])])
    monkeypatch.setattr(loop_mod, "stream_chat", first)

    with bound_session():
        cancelled = json.loads(await asyncio.wait_for(T._task({"prompt": "go"}), 20))
        assert cancelled["stop_reason"] == "cancelled"

        # Second run: same task_id, the loop's real resume path.
        harness.during_call = None
        resumed = _StreamScript([("finished at last", [])])
        monkeypatch.setattr(loop_mod, "stream_chat", resumed)
        out = json.loads(await asyncio.wait_for(
            T._task({"prompt": "finish it", "task_id": cancelled["task_id"]}), 20))

    assert out["response"] == "finished at last"
    assert out["task_id"] == cancelled["task_id"]

    body = resumed.captured[0]
    assert INJECT_BODY in _contents(body), \
        "the pre-cancel append did not survive into the resumed conversation"
    assert "finish it" in _contents(body)
    assert [m["role"] for m in _injected(body)] == ["user"], \
        "after one rebuild of history the injection is still not an " \
        "assistant turn of the child's own"
    assert harness.options[1].chat_messages_handle is not None


def test_the_caller_scope_helper_reads_the_meta_bound_identity():
    """`current_caller_scope()` is how a control surface names itself.

    The two contextvars it reads are bound by `agent_mcp.main.call_tool` from
    the request `_meta` (`lloyd/session_id`, `lloyd/turn_id`) — the only
    channel a caller inside the aggregator process has.
    """
    stok = _task_registry.current_session_id.set(PARENT)
    ttok = _task_registry.current_turn_id.set("turn-9")
    try:
        scope = T.current_caller_scope()
    finally:
        _task_registry.current_session_id.reset(stok)
        _task_registry.current_turn_id.reset(ttok)
    assert scope == CallerScope(session_id=PARENT, turn_id="turn-9")


async def test_authority_across_the_meta_boundary_one_session_cannot_steer(
    harness, monkeypatch, caplog,
):
    """Two real MCP dispatches, each with its own `_meta` session.

    A's child is steered from a caller inside A's own dispatch and accepts it;
    B's child — another session, both running concurrently, both real `Task`
    dispatches through `agent_mcp.main.call_tool` — tries to steer A's child
    and is refused. Authority is never a constructed `CallerScope` here: each
    side names itself with `current_caller_scope()`, which can only report the
    session its own request arrived on.
    """
    from agent_mcp import main as agg_main

    seen: dict = {}
    a_steer_attempted = asyncio.Event()
    b_finished = asyncio.Event()

    async def during(sub_session_id: str):
        rows = {r["session_id"]: r for r in SR.list_active()}
        mine = rows[sub_session_id]
        scope = T.current_caller_scope()
        if mine["parent_session_id"] == PARENT:
            SR.steer(mine["task_id"], INJECT_TEXT, caller=scope)
            seen["a_session"] = scope.session_id
            a_steer_attempted.set()
            await asyncio.wait_for(b_finished.wait(), 20)
        else:
            await asyncio.wait_for(a_steer_attempted.wait(), 20)
            target = next(r["task_id"] for r in SR.list_active()
                          if r["parent_session_id"] == PARENT)
            try:
                SR.steer(target, "is this mine to steer?", caller=scope)
                seen["b_attempt"] = "allowed"
            except SteeringRefused:
                seen["b_attempt"] = "refused"
            b_finished.set()

    harness.during_call = during

    scripts: dict = {}

    def router(**kwargs):
        msgs = kwargs.get("messages") or []
        which = "b" if any("do B" in str(m.get("content")) for m in msgs) else "a"
        script = scripts.get(which)
        if script is None:
            script = _StreamScript(
                [("probing:", [{"id": "c1", "name": "Read", "arguments": {}}]),
                 ("B done.", [])] if which == "b" else
                [("working:", [{"id": "c1", "name": "Read", "arguments": {}}]),
                 ("A done.", [])])
            scripts[which] = script
        return script(**kwargs)

    monkeypatch.setattr(loop_mod, "stream_chat", router)

    with caplog.at_level(logging.WARNING, logger="lloyd-subagent-registry"):
        res_a, res_b = await asyncio.wait_for(asyncio.gather(
            agg_main.call_tool("Task", {"prompt": "do A work"},
                               meta={agg_main.META_SESSION_ID: PARENT}),
            agg_main.call_tool("Task", {"prompt": "do B work"},
                               meta={agg_main.META_SESSION_ID: OTHER}),
        ), 30)

    a = json.loads(res_a.content[0].text)
    b = json.loads(res_b.content[0].text)
    assert a["response"] == "A done." and b["response"] == "B done."
    assert seen["a_session"] == PARENT, "A named itself from its own _meta"
    assert seen["b_attempt"] == "refused", \
        "a caller bound to another session must not be able to steer"
    assert "refused" in caplog.text
    assert OTHER in caplog.text and PARENT in caplog.text
    # The authorised append landed in A's child and is in the history that run
    # stored; exactly one — B's refused attempt added nothing.
    stored_a = SR._history[a["task_id"]].chat_messages
    assert [m["content"] for m in _injected(stored_a)] == [INJECT_BODY]
    assert [m["role"] for m in _injected(stored_a)] == ["user"]
