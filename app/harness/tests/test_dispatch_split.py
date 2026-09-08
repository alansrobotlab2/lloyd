"""The dispatch split, the annotation passthrough, and the wire-order fix.

No behaviour change is claimed here except the last one. `_pre_dispatch` and
`_execute_tool_call` are the same code in two halves, split so a batch can
run the second half concurrently while the first stays ordered.

`_reorder_batch_messages` IS a behaviour change, and it fixes a latent bug
that predates any of this: Inner Voice's pretool inject appends a `user`
message to the shared buffer mid-batch, which leaves
`assistant(tool_calls) -> user -> tool` for every call after the first — a
shape no engine accepts.
"""

from __future__ import annotations

import inspect

from app.harness import loop as L
from app.harness.options import RunOptions
from app.harness.tool_search import LoadedToolSet


def _tc(name="Read", call_id="c1", args=None):
    return {
        "id": call_id,
        "function": {"name": name, "arguments": "{}"},
        "_args_dict": dict(args or {}),
        "_summary": "",
    }


def _loaded_set():
    return LoadedToolSet(enabled=False, catalog=[], loaded=set())


# ── wire order ──────────────────────────────────────────────────────────────

def test_an_inject_between_tool_results_is_lifted_past_them():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a"},
        {"role": "user", "content": "[INNER VOICE] stop reading and answer"},
        {"role": "tool", "tool_call_id": "c2", "content": "b"},
    ]
    L._reorder_batch_messages(msgs, 1)
    assert [m["role"] for m in msgs] == ["assistant", "tool", "tool", "user"]
    assert [m.get("tool_call_id") for m in msgs[1:3]] == ["c1", "c2"]


def test_an_inject_before_the_first_result_is_lifted_too():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "user", "content": "[INNER VOICE] hi"},
        {"role": "tool", "tool_call_id": "c1", "content": "a"},
    ]
    L._reorder_batch_messages(msgs, 1)
    assert [m["role"] for m in msgs] == ["assistant", "tool", "user"]


def test_several_injects_keep_their_relative_order():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": "a"},
        {"role": "user", "content": "first"},
        {"role": "tool", "tool_call_id": "c2", "content": "b"},
        {"role": "user", "content": "second"},
    ]
    L._reorder_batch_messages(msgs, 1)
    assert [m.get("content") for m in msgs[3:]] == ["first", "second"]


def test_a_batch_with_no_inject_is_left_exactly_alone():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": "a"},
        {"role": "tool", "tool_call_id": "c2", "content": "b"},
    ]
    before = list(msgs)
    L._reorder_batch_messages(msgs, 1)
    assert msgs == before


def test_a_batch_with_no_tool_results_is_left_alone():
    msgs = [{"role": "assistant", "content": "x"},
            {"role": "user", "content": "inject"}]
    before = list(msgs)
    L._reorder_batch_messages(msgs, 1)
    assert msgs == before


def test_it_mutates_in_place_so_the_observers_handle_stays_valid():
    """`chat_messages` may be the observer's own list; rebinding would leave
    it holding the old one."""
    msgs = [{"role": "assistant", "content": ""},
            {"role": "tool", "tool_call_id": "c1", "content": "a"},
            {"role": "user", "content": "inject"},
            {"role": "tool", "tool_call_id": "c2", "content": "b"}]
    handle = msgs
    L._reorder_batch_messages(msgs, 1)
    assert handle is msgs
    assert [m["role"] for m in handle] == ["assistant", "tool", "tool", "user"]


def test_the_loop_calls_it_after_every_batch():
    src = inspect.getsource(L.run_query)
    assert "batch_base = len(chat_messages)" in src
    assert "_reorder_batch_messages(chat_messages, batch_base)" in src


# ── the split ───────────────────────────────────────────────────────────────

async def test_pre_dispatch_returns_a_result_for_a_parse_error():
    tc = _tc(args={"__parse_error__": True, "error": "bad json"})
    evt = await L._pre_dispatch(tc=tc, options=RunOptions(model="m"),
                                session_id="s", loaded_set=_loaded_set())
    assert evt is not None and evt["is_error"]
    assert "could not be parsed" in evt["content"]


async def test_pre_dispatch_returns_a_result_for_a_disabled_tool():
    evt = await L._pre_dispatch(tc=_tc("Bash"), options=RunOptions(model="m"),
                                session_id="s", loaded_set=_loaded_set(),
                                runtime_disallowed={"Bash"})
    assert evt is not None and "disabled by configuration" in evt["content"]


async def test_pre_dispatch_returns_none_for_an_ordinary_call():
    evt = await L._pre_dispatch(tc=_tc(), options=RunOptions(model="m"),
                                session_id="s", loaded_set=_loaded_set())
    assert evt is None


async def test_pre_dispatch_honours_a_hook_deny():
    from app.harness.hooks import HookRegistry

    hooks = HookRegistry()

    async def deny(input_dict, tool_use_id, _ctx):
        return {"hookSpecificOutput": {"permissionDecision": "deny",
                                      "permissionDecisionReason": "nope"}}
    hooks.add_pre_tool_use(None, deny)

    evt = await L._pre_dispatch(tc=_tc(), options=RunOptions(model="m", hooks=hooks),
                                session_id="s", loaded_set=_loaded_set())
    assert evt is not None and "denied" in evt["content"] and "nope" in evt["content"]


async def test_execute_runs_the_call_and_post_processes(monkeypatch):
    class _Pool:
        async def call_tool(self, name, args, **kw):
            _Pool.kw = kw
            return {"content": "", "is_error": False}

    evt = await L._execute_tool_call(tc=_tc(), pool=_Pool(),
                                     options=RunOptions(model="m"), session_id="s")
    # The empty-result fallback still runs on this path.
    assert evt["content"] == "(Read completed with no output)"
    assert _Pool.kw["session_id"] == "s"


async def test_dispatch_one_still_composes_both_halves():
    calls = []

    async def fake_pre(**kw):
        calls.append("pre")
        return None

    async def fake_exec(**kw):
        calls.append("exec")
        return {"type": "tool_result", "call_id": "c1", "name": "Read",
                "content": "ok", "is_error": False}

    import app.harness.loop as M
    orig_pre, orig_exec = M._pre_dispatch, M._execute_tool_call
    M._pre_dispatch, M._execute_tool_call = fake_pre, fake_exec
    try:
        evt = await M._dispatch_one_tool_call(
            tc=_tc(), pool=None, options=RunOptions(model="m"),
            session_id="s", loaded_set=_loaded_set())
    finally:
        M._pre_dispatch, M._execute_tool_call = orig_pre, orig_exec
    assert calls == ["pre", "exec"] and evt["content"] == "ok"


async def test_an_early_result_short_circuits_the_dispatch():
    async def never(**kw):
        raise AssertionError("must not dispatch after an early result")

    import app.harness.loop as M
    orig = M._execute_tool_call
    M._execute_tool_call = never
    try:
        evt = await M._dispatch_one_tool_call(
            tc=_tc("Bash"), pool=None, options=RunOptions(model="m"),
            session_id="s", loaded_set=_loaded_set(),
            runtime_disallowed={"Bash"})
    finally:
        M._execute_tool_call = orig
    assert evt["is_error"]


# ── annotations reach the pool ──────────────────────────────────────────────

def test_list_tools_carries_annotations():
    from app.harness import mcp_pool as P
    src = inspect.getsource(P.MCPPool._list_tools)
    assert '"annotations": _annotations(t)' in src


def test_annotations_reads_both_sdk_naming_conventions():
    from app.harness.mcp_pool import _annotations

    class _Snake:
        read_only_hint = True
        destructive_hint = False

    class _Wire:
        readOnlyHint = True

    class _None:
        pass

    assert _annotations(type("T", (), {"annotations": _Snake()})())["readOnlyHint"] is True
    assert _annotations(type("T", (), {"annotations": _Wire()})())["readOnlyHint"] is True
    assert _annotations(type("T", (), {"annotations": None})()) == {}
    assert _annotations(_None()) == {}


def test_the_stdio_path_serialises_calls_on_its_shared_session():
    from app.harness import mcp_pool as P
    src = inspect.getsource(P.MCPPool._invoke)
    assert "self._stdio_locks" in src
    assert "async with lock:" in src
    # The HTTP path opens a session per call and must NOT be serialised.
    http_part = src.split("lock = self._stdio_locks")[0]
    assert "async with self._http_session(cfg) as session:" in http_part


def test_the_lock_map_survives_a_reopen():
    from app.harness import mcp_pool as P
    src = inspect.getsource(P.MCPPool._reopen)
    assert "_stdio_locks" not in src, \
        "a lock recreated under a waiter lets two frames onto one pipe"
