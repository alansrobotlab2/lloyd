"""D4 (review 2026-09-24): a `Task` child inherits the grant gate and the deny list.

The #534 grant gate is a PreToolUse hook, so it exists only on a registry that
installs it — and `builtin_task` built its child's registry with the safety,
outbound-content and skill-dispatch hooks only. A worker turn that called
`Task` therefore handed its subagent every tier-2 tool with no gate, and none
of the parent's per-turn bans either: the child's deny list was config plus
profile plus a private five-name automod ban four names behind the shared one.

The seam is `_meta`, like the model, surface and effect scope before it: the
loop stamps `lloyd/grant_scope` and `lloyd/disallowed_tools`, the aggregator's
`call_tool` lifts them into contextvars around the dispatch, and `_task` reads
them when it builds the child. Each hop is pinned here, and the last test runs
all three on the aggregator side end to end.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import agent_mcp.main as M
from agent_mcp import _subagent_registry as reg
from agent_mcp import builtin_task
from app.harness import mcp_pool, policy
from app.harness.hooks import HookRegistry
from app.harness.policy import GrantStore
from app.tool_bans import WORKER_AUTOMOD_BAN

SESSION = "20260924_task_authority_test"


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    reg.reset()
    # An empty grant store of this test's own: nothing is granted, so every
    # tier-2 call a gated child makes must be denied.
    db = tmp_path / "grants.sqlite"
    GrantStore(db).ensure_schema()
    monkeypatch.setenv("LLOYD_GRANT_DB", str(db))
    policy._STORE_CACHE.clear()
    monkeypatch.setattr(
        builtin_task, "_load_subagent_profile",
        lambda t: {"system_prompt": "", "max_turns": 5,
                   "disallowed_tools": [], "model": "primary", "base_url": ""},
    )
    yield
    reg.reset()
    policy._STORE_CACHE.clear()


def _run_child(monkeypatch, probe=None) -> dict:
    """Run `_task` against a fake `run_query`; return what the child was built
    with, and whatever `probe(options)` answered from inside the child."""
    captured: dict = {}

    async def _fake_run_query(messages, options):
        captured["options"] = options
        if probe is not None:
            captured["probe"] = await probe(options)
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}}

    import app.harness.loop as loop_mod
    monkeypatch.setattr(loop_mod, "run_query", _fake_run_query)
    return captured


async def _email_send(options):
    return await options.hooks.fire_pre_tool_use(
        session_id=options.session_id, tool_name="email_send",
        tool_input={"to": "a@example.com", "subject": "s", "body": "b"})


def _denied(decision) -> str:
    out = (decision or {}).get("hookSpecificOutput") or {}
    return (out.get("permissionDecisionReason") or ""
            if out.get("permissionDecision") == "deny" else "")


def _policy_hooks(hooks: HookRegistry) -> int:
    return sum(1 for entry in hooks._pre if entry[1].__name__ == "_policy_pretool_cb")


def test_a_worker_parents_grant_scope_installs_the_gate_in_the_child(monkeypatch):
    captured = _run_child(monkeypatch, _email_send)
    tok = builtin_task.current_parent_grant_scope.set("worker:deep-research")
    try:
        json.loads(asyncio.run(builtin_task._task({"prompt": "p", "description": "d"})))
    finally:
        builtin_task.current_parent_grant_scope.reset(tok)
    options = captured["options"]
    assert _policy_hooks(options.hooks) == 1
    assert options.grant_scope == "worker:deep-research"
    reason = _denied(captured["probe"])
    assert reason.startswith("grant:"), captured["probe"]
    assert "worker:deep-research" in reason


def test_a_chat_parent_with_no_scope_installs_no_gate(monkeypatch):
    captured = _run_child(monkeypatch, _email_send)
    json.loads(asyncio.run(builtin_task._task({"prompt": "p", "description": "d"})))
    options = captured["options"]
    assert _policy_hooks(options.hooks) == 0
    assert options.grant_scope == ""
    assert not _denied(captured["probe"]).startswith("grant:")


def test_the_parents_disallowed_tools_reach_the_childs_options(monkeypatch):
    captured = _run_child(monkeypatch)
    tok = builtin_task.current_parent_disallowed.set(
        ("backlog_write_task", "grant_create", "Write"))
    try:
        json.loads(asyncio.run(builtin_task._task({"prompt": "p", "description": "d"})))
    finally:
        builtin_task.current_parent_disallowed.reset(tok)
    disallowed = captured["options"].disallowed_tools
    for name in ("backlog_write_task", "grant_create", "Write", "Task"):
        assert name in disallowed
    assert len(disallowed) == len(set(disallowed))


def test_the_automod_ban_is_the_shared_constant_in_both_spellings(monkeypatch):
    captured = _run_child(monkeypatch)
    json.loads(asyncio.run(builtin_task._task({"prompt": "p", "description": "d"})))
    disallowed = set(captured["options"].disallowed_tools)
    for name in WORKER_AUTOMOD_BAN:
        assert name in disallowed
        assert f"mcp__lloyd-mcp__{name}" in disallowed


def test_call_tool_puts_grant_scope_and_disallowed_in_meta():
    assert mcp_pool.META_GRANT_SCOPE == M.META_GRANT_SCOPE
    assert mcp_pool.META_DISALLOWED == M.META_DISALLOWED

    seen: list = []
    pool = mcp_pool.MCPPool({})
    pool._opened = True
    pool._tool_routes = {"Task": "lloyd-mcp"}
    pool._http_configs = {"lloyd-mcp": {}}

    async def _invoke(server, bare, args, budget, meta):
        seen.append(meta)
        return {"content": "ok", "is_error": False}

    pool._invoke = _invoke
    asyncio.run(pool.call_tool("Task", {"prompt": "p"}, session_id="s",
                               grant_scope="worker:x",
                               disallowed_tools=["Write", "grant_create"]))
    asyncio.run(pool.call_tool("Task", {"prompt": "p"}, session_id="s"))
    assert seen[0][M.META_GRANT_SCOPE] == "worker:x"
    assert seen[0][M.META_DISALLOWED] == ["Write", "grant_create"]
    assert M.META_GRANT_SCOPE not in seen[1]
    assert M.META_DISALLOWED not in seen[1]


def test_the_loop_stamps_scope_always_and_the_deny_list_on_task_only(monkeypatch):
    """The router's option wins; a pool-bound contextvar is the fallback; the
    unbound default ("worker") is never read, or every chat Task would be
    gated."""
    from app.harness import loop as L
    from app.harness.options import RunOptions

    kws: list[dict] = []

    class _Pool:
        async def call_tool(self, name, args, **kw):
            kws.append(kw)
            return {"content": "ok", "is_error": False}

    def _call(name, options, runtime=None):
        tc = {"id": "c1", "function": {"name": name, "arguments": "{}"},
              "_args_dict": {}}
        return asyncio.run(L._execute_tool_call(
            tc=tc, pool=_Pool(), options=options, session_id="s",
            runtime_disallowed=runtime))

    chat = RunOptions(model="m", disallowed_tools=["Write"])
    _call("Task", chat)
    assert kws[-1]["grant_scope"] == ""
    assert kws[-1]["disallowed_tools"] == ["Write"]
    _call("Task", chat, runtime={"Write", "Edit"})
    assert kws[-1]["disallowed_tools"] == ["Edit", "Write"]
    _call("Read", chat, runtime={"Write"})
    assert kws[-1]["disallowed_tools"] == ()

    _call("Task", RunOptions(model="m", grant_scope="worker:autotriage"))
    assert kws[-1]["grant_scope"] == "worker:autotriage"

    async def _bound():
        tok = policy.current_scope.set("autonomy-task:39")
        try:
            tc = {"id": "c2", "function": {"name": "Task", "arguments": "{}"},
                  "_args_dict": {}}
            await L._execute_tool_call(tc=tc, pool=_Pool(),
                                       options=RunOptions(model="m"), session_id="s")
        finally:
            policy.current_scope.reset(tok)
    asyncio.run(_bound())
    assert kws[-1]["grant_scope"] == "autonomy-task:39"


def test_a_task_called_with_a_grant_scope_in_meta_denies_a_tier_2_tool_inside_the_child(
        monkeypatch):
    """End to end on the aggregator side: `_meta` in, a gated child out."""
    captured = _run_child(monkeypatch, _email_send)
    base = dict(getattr(M, "_dispatch", None) or {})
    base["Task"] = builtin_task
    monkeypatch.setattr(M, "_dispatch", base)

    meta = {M.META_SESSION_ID: SESSION,
            M.META_GRANT_SCOPE: "worker:deep-research",
            M.META_DISALLOWED: ["backlog_write_task", "grant_create"]}
    result = asyncio.run(M.call_tool("Task", {"prompt": "p", "description": "d"}, meta))
    assert not getattr(result, "isError", False)

    options = captured["options"]
    assert _policy_hooks(options.hooks) == 1
    assert "backlog_write_task" in options.disallowed_tools
    reason = _denied(captured["probe"])
    assert reason.startswith("grant:"), captured["probe"]
    assert "worker:deep-research" in reason
    # The binds are scoped to the dispatch: nothing leaks to the caller.
    assert builtin_task.current_parent_grant_scope.get() == ""
    assert builtin_task.current_parent_disallowed.get() == ()
    assert policy.current_scope.get() == "worker"
    assert policy.current_scope not in __import__("contextvars").copy_context()

    # And a chat parent (no scope in `_meta`) still gets an ungated child.
    asyncio.run(M.call_tool("Task", {"prompt": "p", "description": "d"},
                            {M.META_SESSION_ID: SESSION}))
    assert _policy_hooks(captured["options"].hooks) == 0
