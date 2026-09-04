"""Contract tests for the agent_mcp aggregator and the harness MCP pool.

Covers the defects found in the 2026-09-04 architecture review:

  P0-2  `list_tools` rebuilding the dispatch table must never expose an
        empty or partial map to a concurrent `call_tool`.
  P0-3  Subagents must inherit config-level tool disables.
  P1-2  Tool failures must arrive as `isError=True`, not as successful
        results whose text happens to contain `{"error": ...}`.
  P1-3  One module failing discovery must not take the tool surface down.
  P1-4  Tool and parameter descriptions must stay useful.
  P2-1  No module may carry an orphan `mcp.server.Server`.
  P2-2  No top-level input schema may set `additionalProperties: false`
        while anything is injected alongside the model's arguments.
  P2-4  Duplicate tool names must be reported, not silently shadowed.

Plus the annotation table (`agent_mcp.annotations`) that the plan-mode
gate is derived from.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.types import CallToolResult, Tool  # noqa: E402

from agent_mcp import annotations as A  # noqa: E402
from agent_mcp import main as M  # noqa: E402


@pytest.fixture(scope="module")
def tools() -> list[Tool]:
    return asyncio.run(M.list_tools())


@pytest.fixture(scope="module")
def names(tools) -> set[str]:
    return {t.name for t in tools}


# ── Discovery ────────────────────────────────────────────────────────────────

def test_every_module_discovers(tools):
    assert len(tools) > 100
    degraded = {n: s for n, s in M._discovery_status.items() if not s["ok"]}
    assert degraded == {}, f"modules failed discovery: {degraded}"


def test_dispatch_covers_every_advertised_tool(tools, names):
    assert set(M._dispatch) == names


def test_no_duplicate_tool_names(tools):
    seen = [t.name for t in tools]
    assert len(seen) == len(set(seen))


async def test_one_failing_module_does_not_break_discovery(monkeypatch):
    """P1-3: a raising module is skipped and recorded, not fatal."""

    async def _boom():
        raise RuntimeError("bridge is down")

    async def _never(name, arguments):
        raise AssertionError("never dispatched")

    boom = types.SimpleNamespace(
        __name__="agent_mcp.boom", list_tools=_boom, call_tool=_never,
    )
    monkeypatch.setattr(M, "MODULES", [*M.MODULES, boom])
    tools = await M.list_tools()
    assert len(tools) > 100                       # the rest still came through
    assert M._discovery_status["boom"]["ok"] is False
    assert "bridge is down" in M._discovery_status["boom"]["error"]


async def test_duplicate_name_is_dropped_not_shadowed(monkeypatch, caplog):
    """P2-4: a second claimant on a name loses, loudly."""

    async def _dupe_list():
        return [Tool(name="Read", description="impostor",
                     inputSchema={"type": "object", "properties": {}})]

    async def _never(name, arguments):
        raise AssertionError("never dispatched")

    dupe = types.SimpleNamespace(
        __name__="agent_mcp.dupe", list_tools=_dupe_list, call_tool=_never,
    )
    monkeypatch.setattr(M, "MODULES", [*M.MODULES, dupe])
    with caplog.at_level("ERROR"):
        tools = await M.list_tools()
    assert sum(1 for t in tools if t.name == "Read") == 1
    assert M._dispatch["Read"].__name__.endswith("builtin_fs")
    assert any("duplicate tool" in r.getMessage() for r in caplog.records)


async def test_dispatch_never_observed_empty_during_rebuild():
    """P0-2: the old code cleared `_dispatch` then refilled it across 22
    awaits, so a concurrent call routed to nothing. Hammer both at once.
    """
    await M.list_tools()
    stop = False
    misses: list[str] = []

    async def caller():
        while not stop:
            result = await M.call_tool("Read", {"file_path": "/etc/hostname"})
            text = result.content[0].text
            if "Unknown tool" in text:
                misses.append(text)
            await asyncio.sleep(0)

    async def rebuilder():
        for _ in range(40):
            await M.list_tools()
            await asyncio.sleep(0)

    task = asyncio.create_task(caller())
    await rebuilder()
    stop = True
    await task
    assert misses == []


# ── Error signalling (P1-2) ──────────────────────────────────────────────────

@pytest.mark.parametrize("tool,args", [
    ("Read", {"file_path": "/definitely/not/here"}),
    ("Bash", {"command": "exit 7"}),
    ("Bash", {"command": "pwd", "cwd": "not-absolute"}),
    ("fact_get", {}),
    ("vault_read", {}),
])
async def test_failures_set_is_error(tool, args):
    result = await M.call_tool(tool, args)
    assert isinstance(result, CallToolResult)
    assert result.isError is True, f"{tool}{args} did not report isError"


@pytest.mark.parametrize("tool,args", [
    ("Read", {"file_path": "/etc/hostname"}),
    ("Bash", {"command": "echo ok"}),
    ("Glob", {"pattern": "*.py", "path": str(ROOT / "agent_mcp")}),
])
async def test_successes_do_not_set_is_error(tool, args):
    result = await M.call_tool(tool, args)
    assert isinstance(result, CallToolResult)
    assert result.isError is False


async def test_unknown_tool_is_an_error_result():
    result = await M.call_tool("NoSuchToolAtAll", {})
    assert isinstance(result, CallToolResult)
    assert result.isError is True


# ── Schema hygiene ───────────────────────────────────────────────────────────

def test_no_toplevel_additional_properties_false(tools):
    """P2-2: the aggregator strips a legacy `_session_id` argument, but the
    SDK validates arguments against inputSchema *before* the handler runs.
    A top-level `additionalProperties: false` would reject the call
    outright. Session id moved to `_meta`, and this keeps the door shut.
    """
    offenders = [
        t.name for t in tools
        if (t.inputSchema or {}).get("additionalProperties") is False
    ]
    assert offenders == []


def test_tool_names_within_openai_limit(tools):
    assert [t.name for t in tools if len(t.name) > M.TOOL_NAME_MAX] == []


def test_tool_descriptions_are_useful(tools):
    """P1-4: a description short enough to be a label can't disambiguate a
    tool from its 123 neighbours."""
    thin = {t.name: len(t.description or "") for t in tools
            if len(t.description or "") < 60}
    assert thin == {}, f"tools with thin descriptions: {thin}"


def test_tool_parameters_are_documented(tools):
    """P1-4: `Edit.old_string` with no description is the whole tool."""
    undocumented: list[str] = []
    for t in tools:
        props = (t.inputSchema or {}).get("properties") or {}
        for pname, spec in props.items():
            if not isinstance(spec, dict) or not (spec.get("description") or "").strip():
                undocumented.append(f"{t.name}.{pname}")
    assert undocumented == [], f"undocumented parameters: {undocumented}"


# ── Module contract (P2-1) ───────────────────────────────────────────────────

def test_modules_meet_the_contract():
    for mod in M.MODULES:
        M._check_module(mod)


def test_no_orphan_server_instances():
    """The SDK's decorators return the function unchanged, so a per-module
    `Server` is dead weight — and mcp 2.x removes the decorator API those
    objects use, so any survivor becomes migration cost for nothing."""
    import agent_mcp

    pkg_dir = Path(agent_mcp.__file__).parent
    offenders = []
    for path in sorted(pkg_dir.glob("*.py")):
        if path.name == "main.py":
            continue
        src = path.read_text()
        if "= Server(" in src or "@app.list_tools" in src or "@app.call_tool" in src:
            offenders.append(path.name)
    assert offenders == []


# ── Annotations + plan-mode gate ─────────────────────────────────────────────

def test_every_tool_is_annotated(tools):
    assert [t.name for t in tools if t.annotations is None] == []


def test_annotation_tables_have_no_stale_entries(names):
    for label, table in (("READ_ONLY", A.READ_ONLY),
                         ("DESTRUCTIVE", A.DESTRUCTIVE),
                         ("IDEMPOTENT", A.IDEMPOTENT),
                         ("PLAN_MODE_ALWAYS_ALLOWED", A.PLAN_MODE_ALWAYS_ALLOWED)):
        stale = sorted(table - names - {"ToolSearch"})  # ToolSearch is harness-side
        assert stale == [], f"{label} names tools that no longer exist: {stale}"


def test_destructive_tools_are_not_read_only():
    assert A.DESTRUCTIVE & A.READ_ONLY == frozenset()


def test_plan_mode_blocks_actuators_and_spares_controls(names):
    blocked = set(A.plan_mode_blocked_tools(names))
    for actuator in ("Bash", "Write", "Edit", "Task", "email_send", "vault_write",
                     "fact_add", "discord_send", "browser_click", "memory_add",
                     "autonomy_write_task", "http_request"):
        assert actuator in blocked, f"{actuator} should be blocked in plan mode"
    for control in ("ExitPlanMode", "EnterPlanMode", "TodoWrite", "Read", "Grep",
                    "vault_search", "fact_get", "skills_search", "mc_navigate"):
        assert control not in blocked, f"{control} must stay usable in plan mode"


def test_plan_mode_gate_falls_back_before_discovery(monkeypatch):
    import app.mcp_discovery as D

    monkeypatch.setattr(D, "_TOOL_UNIVERSE", set())
    assert set(D._plan_mode_blocked()) == set(D.PLAN_MODE_BLOCKED_TOOLS)


def test_plan_mode_gate_uses_annotations_after_discovery(monkeypatch, names):
    import app.mcp_discovery as D

    monkeypatch.setattr(D, "_TOOL_UNIVERSE", set(names))
    blocked = set(D._get_disallowed_tools(plan_mode=True))
    assert "email_send" in blocked and "vault_write" in blocked
    assert "ExitPlanMode" not in blocked
