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

import pytest

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


@pytest.mark.parametrize("parallel", [False, True])
async def test_a_hook_inject_during_a_batch_lands_after_the_batch(monkeypatch, parallel):
    """Through the real loop, on both dispatch paths: a pretool hook that
    appends to the shared buffer while the batch runs ends up AFTER every tool
    message of that batch, and the next request carries that shape."""
    from app.harness.hooks import HookRegistry
    from app.harness.tests import _replay as R

    handle: list[dict] = []
    hooks = HookRegistry()

    async def inject(input_dict, tool_use_id, _ctx):
        if input_dict["tool_name"] == "Read":
            handle.append({"role": "user", "content": "[INNER VOICE] wrap up"})
        return {}
    hooks.add_pre_tool_use(None, inject)

    engine = R.ReplayEngine([R.Step(tool_calls=[
        R.tool_call("c1", "Read", "a"), R.tool_call("c2", "Grep", "b"),
        R.tool_call("c3", "Glob", "c")]), R.Step(text="done")])
    R.install(monkeypatch, engine, R.ReplayPool(delay_by_call_id={"c1": 0.02}))
    await R.drive(RunOptions(model="m", max_turns=4, tool_search_enabled=False,
                             parallel_tool_calls_enabled=parallel, hooks=hooks,
                             chat_messages_handle=handle))

    sent = [m["role"] for m in engine.requests[1].messages]
    assert sent == ["user", "assistant", "tool", "tool", "tool", "user"], sent
    assert [m["role"] for m in handle][:6] == sent


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

async def test_list_tools_carries_annotations():
    """Discovery keeps the server's hints, so a read-only batch can qualify
    from what the server said rather than a second list of names."""
    import mcp.types as T

    from app.harness import mcp_pool as P

    class _Session:
        async def list_tools(self):
            return T.ListToolsResult(tools=[
                T.Tool(name="Read", inputSchema={"type": "object"},
                       annotations=T.ToolAnnotations(readOnlyHint=True)),
                T.Tool(name="Bash", inputSchema={"type": "object"},
                       annotations=T.ToolAnnotations(readOnlyHint=False,
                                                     destructiveHint=True)),
                T.Tool(name="plain", inputSchema={"type": "object"}),
            ])

    pool = P.MCPPool({})
    tools = await pool._list_tools("srv", _Session())
    by_name = {t["name"]: t for t in tools}
    assert by_name["Read"]["annotations"]["readOnlyHint"] is True
    assert by_name["Bash"]["annotations"]["readOnlyHint"] is False
    assert by_name["plain"]["annotations"] == {}
    pool._register("srv", tools)
    assert L._read_only_names(pool) == {"Read"}


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


# ── the stdio lock, driven through MCPPool.call_tool ───────────────────────

class _FakeSession:
    """Stands in for a `ClientSession`. `shared` is one overlap counter across
    every session in a test, so a second session after a reopen still counts
    against the first."""

    def __init__(self, shared: dict, name: str = "s"):
        self.shared = shared
        self.name = name

    async def call_tool(self, bare, args, read_timeout_seconds=None, meta=None):
        import asyncio

        import mcp.types as T

        self.shared["live"] = self.shared.get("live", 0) + 1
        self.shared["peak"] = max(self.shared.get("peak", 0), self.shared["live"])
        try:
            await asyncio.sleep(self.shared.get("delay", 0.03))
        finally:
            self.shared["live"] -= 1
        return T.CallToolResult(content=[T.TextContent(type="text", text=bare)])


def _stdio_pool(servers=("s",), shared=None):
    from app.harness import mcp_pool as P

    shared = shared if shared is not None else {}
    pool = P.MCPPool({n: {"type": "stdio", "command": "true"} for n in servers})
    for n in servers:
        pool._sessions[n] = _FakeSession(shared, n)
        pool._tool_routes[f"t_{n}"] = n
    pool._opened = True
    return pool, shared


async def test_two_concurrent_stdio_calls_serialise_on_their_shared_session():
    import asyncio

    pool, shared = _stdio_pool()
    got = await asyncio.gather(pool.call_tool("t_s", {}), pool.call_tool("t_s", {}))
    assert [g["content"] for g in got] == ["t_s", "t_s"]
    assert shared["peak"] == 1, "two frames were on one pipe at once"


async def test_the_stdio_lock_is_per_server_not_global():
    import asyncio

    pool, shared = _stdio_pool(servers=("a", "b"))
    await asyncio.gather(pool.call_tool("t_a", {}), pool.call_tool("t_b", {}))
    assert shared["peak"] == 2, "different servers must not wait on each other"


async def test_the_http_path_is_not_serialised(monkeypatch):
    import asyncio
    from contextlib import asynccontextmanager

    from app.harness import mcp_pool as P

    shared: dict = {}
    # The real default config: an HTTP server, never contacted here because
    # `_http_session` is replaced below.
    pool = P.MCPPool(dict(P.DEFAULT_LLOYD_MCP_SERVERS))
    (server,) = P.DEFAULT_LLOYD_MCP_SERVERS
    pool._tool_routes["t_h"] = server
    pool._opened = True

    @asynccontextmanager
    async def _session(_cfg):
        yield _FakeSession(shared, "h")

    monkeypatch.setattr(pool, "_http_session", _session)
    await asyncio.gather(pool.call_tool("t_h", {}), pool.call_tool("t_h", {}))
    assert shared["peak"] == 2


async def test_the_lock_map_survives_a_reopen():
    """A reopen while a call holds the lock must not mint a fresh lock: the
    call issued after the reopen still waits for the one in flight."""
    import asyncio

    pool, shared = _stdio_pool()
    shared["delay"] = 0.05

    async def _open():
        pool._sessions["s"] = _FakeSession(shared, "s")
        pool._tool_routes["t_s"] = "s"
        pool._opened = True

    pool.open = _open   # the owner task and subprocess are not the subject here

    first = asyncio.create_task(pool.call_tool("t_s", {}))
    await asyncio.sleep(0.01)            # first now holds the lock
    await pool._reopen()
    second = asyncio.create_task(pool.call_tool("t_s", {}))
    await asyncio.gather(first, second)
    assert shared["peak"] == 1, \
        "a lock recreated under a waiter lets two frames onto one pipe"


async def test_cancelling_the_dispatcher_cancels_the_pool_call():
    """D9: `asyncio.wait` does not cancel what it waits on. When the task
    running `_execute_tool_call` is itself cancelled (a closed generator
    mid-batch, the pool's `wait_for`), the MCP call must go with it rather
    than run on as an orphan nobody awaits — and the cancel-watcher too."""
    import asyncio

    started = asyncio.Event()
    outcome: list[str] = []

    class _Pool:
        async def call_tool(self, name, args, **kw):
            started.set()
            try:
                await asyncio.sleep(30)
                outcome.append("finished")
            except asyncio.CancelledError:
                outcome.append("cancelled")
                raise
            return {"content": "late", "is_error": False}

    before = set(asyncio.all_tasks())
    cancel_event = asyncio.Event()
    dispatcher = asyncio.create_task(L._execute_tool_call(
        tc=_tc(), pool=_Pool(),
        options=RunOptions(model="m", cancel_event=cancel_event),
        session_id="s",
    ))
    await asyncio.wait_for(started.wait(), 2)
    dispatcher.cancel()
    try:
        await dispatcher
    except asyncio.CancelledError:
        pass
    for _ in range(5):
        await asyncio.sleep(0)
    assert outcome == ["cancelled"], outcome
    leftovers = [t for t in asyncio.all_tasks() - before
                 if t is not asyncio.current_task() and not t.done()]
    assert leftovers == [], leftovers
