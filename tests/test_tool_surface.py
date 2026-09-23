"""The tool surface's table, the platform that picks it, and its inheritance by
a Task subagent (2026-09-23). The loop half is app/harness/tests/
test_loop_tool_surface.py."""

from __future__ import annotations

import asyncio

from agent_mcp import annotations as A
from agent_mcp import builtin_task
from agent_mcp import main as M
from app.harness import mcp_pool
from app.routers import messages


def test_every_surface_entry_names_a_live_tool_and_the_sets_are_disjoint():
    names = {t.name for t in asyncio.run(M.list_tools())}
    assert sorted((A.CHAT_ONLY | A.WORKER_ONLY) - names) == []
    assert A.CHAT_ONLY & A.WORKER_ONLY == frozenset()


def test_an_unknown_or_empty_surface_hides_nothing():
    assert A.hidden_on_surface("") == frozenset()
    assert A.hidden_on_surface("subagent") == frozenset()
    assert A.hidden_on_surface("worker") == A.CHAT_ONLY
    assert A.hidden_on_surface("chat") == A.WORKER_ONLY


def test_the_worker_platforms_are_the_worker_surface():
    for platform in ("autonomy", "worker"):
        assert messages._tool_surface(platform) == "worker"
    # A new chat's file may not exist yet; unknown platforms are chats.
    for platform in ("mission-control", "discord", "", "e2e-harness"):
        assert messages._tool_surface(platform) == "chat"


def test_the_meta_key_agrees_on_both_sides_of_the_seam():
    assert M.META_SURFACE == mcp_pool.META_SURFACE == "lloyd/surface"


async def test_a_task_subagent_inherits_the_calling_turns_surface(monkeypatch):
    seen: list[str] = []

    class TaskLike:
        async def call_tool(self, name, arguments):
            seen.append(builtin_task.current_parent_surface.get())
            return [M.TextContent(type="text", text="done")]

    table = dict(M._dispatch)
    table["Task"] = TaskLike()
    monkeypatch.setattr(M, "_dispatch", table)
    meta = {M.META_SESSION_ID: "20260923_101010_worker_abcd",
            M.META_SURFACE: "worker"}
    await M.call_tool("Task", {"prompt": "go"}, meta)
    assert seen == ["worker"]
    assert builtin_task.current_parent_surface.get() == ""
