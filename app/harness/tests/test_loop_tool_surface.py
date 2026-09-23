"""The tool surface: a worker turn is not handed the chat-only tools, and a chat
turn is not handed the worker-only ones (2026-09-23).

`RunOptions.surface` feeds `agent_mcp.annotations.hidden_on_surface` into the
loop's existing disallowed set, so a hidden tool is neither advertised nor
dispatchable, including after a plan-mode refresher rebuilds the set.
"""

from __future__ import annotations

import asyncio

from agent_mcp.annotations import CHAT_ONLY, WORKER_ONLY
from app.harness.options import RunOptions
from app.harness.tests.test_loop_tool_search import (
    _drain, _FakePool, _mcp_tool, _patch_pool, _StreamScript)

CATALOG_NAMES = ["Bash", "mc_navigate", "ide_open_file", "grant_create",
                 "session_inject_context", "vault_search"]


class _KwPool(_FakePool):
    """Records the keyword arguments each dispatch carried."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.kw_log: list[dict] = []

    async def call_tool(self, name, args, *, session_id="", **kw):
        self.kw_log.append(dict(kw))
        return await super().call_tool(name, args, session_id=session_id)


def _run(monkeypatch, surface, turns, **opts):
    pool = _KwPool("lloyd-mcp", [_mcp_tool(n) for n in CATALOG_NAMES])
    _patch_pool(monkeypatch, pool)
    script = _StreamScript(turns)
    monkeypatch.setattr("app.harness.loop.stream_chat", script)
    options = RunOptions(model="primary", tool_search_enabled=False,
                         session_id="surface-test",
                         surface=surface, **opts)
    events = asyncio.run(_drain([{"role": "user", "content": "hi"}], options))
    advertised = {t["function"]["name"] for t in script.captured_tools[0]}
    return pool, advertised, events


def test_the_table_is_what_the_test_exercises():
    assert {"mc_navigate", "ide_open_file", "grant_create"} <= CHAT_ONLY
    assert "session_inject_context" in WORKER_ONLY


def test_a_worker_turn_is_not_shown_the_chat_only_tools(monkeypatch):
    _, advertised, _ = _run(monkeypatch, "worker", [("done", [])])
    assert advertised == {"Bash", "session_inject_context", "vault_search"}


def test_a_chat_turn_is_not_shown_the_worker_only_tools(monkeypatch):
    _, advertised, _ = _run(monkeypatch, "chat", [("done", [])])
    assert advertised == set(CATALOG_NAMES) - {"session_inject_context"}


def test_no_surface_hides_nothing(monkeypatch):
    """Every caller that never set one gets exactly what it got before."""
    _, advertised, _ = _run(monkeypatch, "", [("done", [])])
    assert advertised == set(CATALOG_NAMES)


def _refused(events, name):
    return [e for e in events if e.get("type") == "tool_result"
            and e.get("name") == name and e.get("is_error")]


def test_a_hidden_tool_named_anyway_is_refused_not_dispatched(monkeypatch):
    call = {"id": "c1", "name": "mc_navigate", "arguments": {"tab": "backlog"}}
    pool, _, events = _run(monkeypatch, "worker", [("", [call]), ("done", [])])
    assert _refused(events, "mc_navigate")
    assert [n for n, _a in pool.call_log] == []


def test_a_plan_mode_refresher_does_not_bring_a_hidden_tool_back(monkeypatch):
    """The refresher rebuilds the dispatch set every iteration from session
    state; the surface has to survive that, not only the turn-start build."""
    call = {"id": "c1", "name": "mc_navigate", "arguments": {"tab": "backlog"}}
    pool, _, events = _run(monkeypatch, "worker", [("", [call]), ("done", [])],
                           disallowed_tools_refresh=lambda: [])
    assert _refused(events, "mc_navigate")
    assert pool.call_log == []


def test_the_surface_rides_with_every_dispatch(monkeypatch):
    """So a Task subagent the call spawns runs on the same surface."""
    call = {"id": "c1", "name": "vault_search", "arguments": {"query": "x"}}
    pool, _, _ = _run(monkeypatch, "worker", [("", [call]), ("done", [])])
    assert pool.kw_log and pool.kw_log[0].get("surface") == "worker"
