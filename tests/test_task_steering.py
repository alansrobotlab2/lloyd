"""Steering and killing a RUNNING `Task` child (#411).

A subagent used to be write-only once started: nothing could push new
information into it and nothing could stop it. The harness seam already
exists — `RunOptions.chat_messages_handle` is read by the loop *by reference*
(the loop uses the caller's list instead of copying it), and
`RunOptions.cancel_event` is checked between iterations, between SSE chunks
and before every tool dispatch. What `Task` never did was hand either one to
anyone: the list is a local in `_task` whose only escaping reference is
`store_history()` inside `_close` — i.e. AFTER the run — and
`options.cancel_event` was never set at all.

So these tests pin the four things that make the seam usable rather than
merely present:

1. the running row publishes the live list, and an append through the
   registry reaches the child's NEXT vLLM request body — asserted on the
   message list the model is actually sent, the way
   `app/harness/tests/test_loop_inject_ordering.py` asserts an Inner Voice
   inject (real `run_query`, scripted stream, nothing under test mocked);
2. an append from a caller outside the authorised set adds nothing and is
   logged;
3. `options.cancel_event` is set on the child, and setting it mid-run stops
   the child with `stop_reason == "cancelled"` while `_close` still closes
   the row and stores the history;
4. a cancelled child stays resumable, and the injected message survives into
   the resumed conversation.

The framing of an injected message is fixed in code (`STEER_ROLE`,
`STEER_PREFIX`) and asserted here, because a resumed history is rebuilt from
this same list: an append that read as an assistant turn would be the child
agreeing with itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

import pytest

from agent_mcp import _subagent_registry as SR
from agent_mcp import _task_registry
from agent_mcp import builtin_task as T
from agent_mcp._subagent_registry import CallerScope, SteeringRefused

PARENT = "sess-parent-1"
OTHER = "sess-someone-else"

INJECT_TEXT = "the file is at /x, not /y"
INJECT_BODY = "[ORCHESTRATOR] the file is at /x, not /y"


@pytest.fixture(autouse=True)
def _clean():
    SR.reset()
    yield
    SR.reset()


@contextlib.contextmanager
def bound_session(session_id: str = PARENT, turn_id: str = "turn-1"):
    """Act as the turn that spawned the child.

    These two contextvars are what `main.call_tool` binds from the request
    `_meta`; setting them is what a real dispatch does, and
    `current_caller_scope()` reads them back.
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

    An `asyncio.Event` is not serialisable and the conversation is unbounded,
    so the row carries the capability, never the handle.
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
    assert "refused" in str(exc.value)


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
    rec = _row(chat_messages=chat)
    with pytest.raises(SteeringRefused):
        SR.steer(rec.task_id, "   ", caller=_caller())
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


# ── the real loop: request bodies and cancellation ─────────────────────────
# Everything below drives the real `app.harness.loop.run_query` with a
# scripted vLLM stream, so "the child saw it" is asserted on the message list
# the model is actually sent, not on a mock of the loop.

class _FakePool:
    def __init__(self) -> None:
        self._discovered = [("lloyd-mcp", [{
            "name": "Read",
            "description": "read a file",
            "inputSchema": {"type": "object", "properties": {}},
        }])]

    @property
    def discovered(self):
        return self._discovered

    async def call_tool(self, name: str, args: dict, *, session_id: str = "", **_kw):
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


@pytest.fixture(autouse=True)
def _reset_cache():
    from app.harness import tool_search_cache
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


@pytest.fixture
def real_loop(monkeypatch):
    """Real `run_query`, fake pool, fake engine, deterministic profile."""
    from app.harness import loop as L

    async def _build_pool(_options):
        return _FakePool()

    monkeypatch.setattr(L, "_build_pool", _build_pool)
    monkeypatch.setattr(
        T, "_load_subagent_profile",
        lambda t: {"system_prompt": "sys", "max_turns": 20,
                   "disallowed_tools": [], "model": "primary", "base_url": ""},
    )
    return L


async def test_an_append_through_the_registry_reaches_the_childs_next_request(real_loop, monkeypatch):
    """Clause 1 — mirrors test_loop_inject_ordering.py for a Task child.

    The append happens while the child is mid-run and is asserted on the
    following iteration's request body.
    """
    def on_request(n: int):
        if n == 1:
            SR.steer(SR.list_active()[0]["task_id"], INJECT_TEXT, caller=_caller())

    script = _StreamScript(
        [("Reading it:", [{"id": "c1", "name": "Read", "arguments": {}}]),
         ("Done.", [])],
        on_request=on_request,
    )
    monkeypatch.setattr(real_loop, "stream_chat", script)

    with bound_session():
        out = json.loads(await T._task({"prompt": "read /y"}))

    assert len(script.captured) == 2, script.captured
    first, second = script.captured
    # Not in the prompt: the child saw it because it was appended mid-run.
    assert INJECT_BODY not in _contents(first), first
    assert INJECT_BODY in _contents(second), second
    injected = [m for m in second
                if str(m.get("content", "")).startswith("[ORCHESTRATOR]")]
    assert [m["role"] for m in injected] == ["user"]
    # And the child kept working afterwards.
    assert out["response"] == "Done."


class _Wedge:
    """A child that reads its own published `cancel_event` and stops on it.

    Not a mock of the loop's cancellation — the loop's own checks stand. What
    is under test here is that `_task` puts a real event on
    `options.cancel_event`, so this fake stops by doing exactly what the loop
    does with that field, and raises `AttributeError` if it is still unset.
    """

    instances: list = []

    def __init__(self, messages, options):
        self.options = options
        self.reached = asyncio.Event()
        self.stopped_via_event = False
        _Wedge.instances.append(self)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        handle = self.options.chat_messages_handle
        handle.append({"role": "assistant", "content": "half-way there"})
        yield {"type": "assistant_message", "text": "half-way there",
               "thinking": "", "tool_calls": []}
        self.reached.set()
        while not self.options.cancel_event.is_set():
            await asyncio.sleep(0.005)
        self.stopped_via_event = True
        yield {"type": "result", "stop_reason": "cancelled", "num_turns": 2}


@pytest.fixture
def wedged(monkeypatch):
    _Wedge.instances = []

    import app.harness.loop as L
    monkeypatch.setattr(L, "run_query", lambda m, o: _Wedge(m, o))
    monkeypatch.setattr(
        T, "_load_subagent_profile",
        lambda t: {"system_prompt": "sys", "max_turns": 20,
                   "disallowed_tools": [], "model": "primary", "base_url": ""},
    )
    return _Wedge


async def test_task_puts_a_cancel_event_and_the_live_handle_on_the_childs_options(wedged):
    """Clause 3's literal half: `_task` sets `options.cancel_event`."""
    async def drive(caller):
        run = asyncio.create_task(T._task({"prompt": "go"}))
        while not wedged.instances:
            await asyncio.sleep(0.005)
        child = wedged.instances[0]
        assert child.options.cancel_event is not None
        assert SR.active_run(SR.list_active()[0]["task_id"]).cancel_event is \
            child.options.cancel_event
        SR.cancel_run(SR.list_active()[0]["task_id"], caller=caller)
        return await run

    with bound_session() as caller:
        out = json.loads(await asyncio.wait_for(drive(caller), 10))
    assert out["stop_reason"] == "cancelled"


async def test_cancelling_mid_run_stops_the_child_and_still_closes_and_stores(wedged):
    """Clause 3 — `stop_reason` cancelled, row closed, history stored."""
    async def drive(caller):
        run = asyncio.create_task(T._task({"prompt": "go"}))
        while not SR.list_active():
            await asyncio.sleep(0.005)
        tid = SR.list_active()[0]["task_id"]
        SR.steer(tid, INJECT_TEXT, caller=caller)
        while not wedged.instances[0].reached.is_set():
            await asyncio.sleep(0.005)
        SR.cancel_run(tid, caller=caller, reason="wedged")
        return tid, await run

    with bound_session() as caller:
        tid, raw = await asyncio.wait_for(drive(caller), 10)
    out = json.loads(raw)

    assert out["stop_reason"] == "cancelled"
    assert "cancelled" in out["error"]
    assert out["task_id"] == tid
    assert wedged.instances[0].stopped_via_event is True

    row = SR.list_recent()[0]
    assert row["status"] == "cancelled" and row["stop_reason"] == "cancelled"
    assert row["injected"] == 1, "the row records that it was steered"
    # `_close` closes the row AND stores history on every exit path — a
    # cancelled child has to be resumable, not lost.
    assert tid in SR._history
    stored = SR._history[tid].chat_messages
    assert INJECT_BODY in _contents(stored)
    assert "half-way there" in _contents(stored)


async def test_a_cancelled_child_resumes_with_the_injection_still_in_it(wedged, monkeypatch):
    """Clause 4 — the pre-cancel append survives into the resumed run."""
    async def first(caller):
        run = asyncio.create_task(T._task({"prompt": "go"}))
        while not SR.list_active():
            await asyncio.sleep(0.005)
        tid = SR.list_active()[0]["task_id"]
        SR.steer(tid, INJECT_TEXT, caller=caller)
        while not wedged.instances[0].reached.is_set():
            await asyncio.sleep(0.005)
        SR.cancel_run(tid, caller=caller)
        return tid, json.loads(await run)

    with bound_session() as caller:
        tid, out = await asyncio.wait_for(first(caller), 10)
    assert out["stop_reason"] == "cancelled"

    seen: list = []

    class _Resume:
        def __init__(self, messages, options):
            seen.append({"messages": list(messages),
                         "handle": options.chat_messages_handle})
            self._options = options

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            self._options.chat_messages_handle.append(
                {"role": "assistant", "content": "finished at last"})
            yield {"type": "assistant_message", "text": "finished at last",
                   "thinking": "", "tool_calls": []}
            yield {"type": "result", "stop_reason": "stop", "num_turns": 1}

    import app.harness.loop as L
    monkeypatch.setattr(L, "run_query", lambda m, o: _Resume(m, o))

    with bound_session():
        resumed = json.loads(await T._task({"prompt": "finish it", "task_id": tid}))

    assert resumed["response"] == "finished at last"
    assert resumed["task_id"] == tid
    handle = seen[0]["handle"]
    assert INJECT_BODY in _contents(handle)
    assert "finish it" in _contents(handle)
    assert seen[0]["messages"] == [], \
        "a resume must not pass `messages` — the loop would ignore them"
    # Still framed as an outside message after one rebuild of history, never
    # as an assistant turn of the child's own.
    injected = [m for m in handle
                if str(m.get("content", "")).startswith("[ORCHESTRATOR]")]
    assert [m["role"] for m in injected] == ["user"]


# ── the `_meta` seam: caller identity arrives over the MCP boundary ─────────

def test_the_caller_scope_helper_reads_the_meta_bound_identity():
    """`current_caller_scope()` is how a control surface names itself.

    The two contextvars it reads are bound by `agent_mcp.main.call_tool` from
    the request `_meta` (`lloyd/session_id`, `lloyd/turn_id`), which is the
    only channel a caller inside the aggregator process has.
    """
    stok = _task_registry.current_session_id.set(PARENT)
    ttok = _task_registry.current_turn_id.set("turn-9")
    try:
        scope = T.current_caller_scope()
    finally:
        _task_registry.current_session_id.reset(stok)
        _task_registry.current_turn_id.reset(ttok)
    assert scope == CallerScope(session_id=PARENT, turn_id="turn-9")


async def test_authority_is_the_meta_bound_session_across_the_dispatch_seam(real_loop, monkeypatch):
    """One Task dispatched with `_meta`; the row's authority is what arrived.

    The child's body here acts as a *concurrent* caller inside the aggregator
    — the shape every real steering caller has, since `Task` blocks its own
    tool call. It steers with the identity the dispatch bound, and is refused
    when it claims someone else's.
    """
    from agent_mcp import main as agg_main

    seen: dict = {}

    async def _concurrent(messages, options):
        row = SR.list_active()[0]
        seen["row_parent"] = row["parent_session_id"]
        chat = SR.active_run(row["task_id"]).chat_messages
        seen["len_before"] = len(chat)
        msg = SR.steer(row["task_id"], INJECT_TEXT, caller=T.current_caller_scope())
        seen["injected"] = msg
        with pytest.raises(SteeringRefused):
            SR.steer(row["task_id"], "not mine", caller=CallerScope(session_id=OTHER))
        seen["len_after_refusal"] = len(chat)
        yield {"type": "assistant_message", "text": "ok",
               "thinking": "", "tool_calls": []}
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1}

    monkeypatch.setattr("app.harness.loop.run_query",
                        lambda m, o: _concurrent(m, o))

    res = await agg_main.call_tool("Task", {"prompt": "go"},
                                   meta={agg_main.META_SESSION_ID: PARENT})
    assert json.loads(res.content[0].text)["response"] == "ok"
    assert seen["row_parent"] == PARENT, "the row's authority comes from _meta"
    assert seen["injected"] == {"role": "user", "content": INJECT_BODY}
    assert seen["len_after_refusal"] == seen["len_before"] + 1, \
        "the refused append added nothing to the live list"
