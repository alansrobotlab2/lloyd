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

# Modules whose tool count depends on something outside the git tree. The
# Thunderbird bridge (agent-services/services/thunderbird-mcp/mcp-bridge.cjs)
# is gitignored, so it is absent from every worktree — and it contributes ~40
# of the ~124 live tools. A flat `> 100` therefore passed in the live checkout
# and failed in any worktree, which made the self-modification gate (it runs
# the whole suite inside a worktree) unable to pass at all. Count only what git
# actually carries.
EXTERNAL_TOOL_MODULES = {"thunderbird"}


def _internal(tools):
    """Tools excluding those from external-application bridges.

    Accepts both shapes in use here: `Tool` objects from `M.list_tools()` and
    plain dicts from `app.mcp_discovery._discover_mcp_tools`.
    """
    from agent_mcp import main as _M
    external_names = set()
    for mod_name in EXTERNAL_TOOL_MODULES:
        mod = getattr(_M, mod_name, None)
        if mod is None:
            continue
        for tool_name, owner in _M._dispatch.items():
            if owner is mod:
                external_names.add(tool_name)

    def _name(t):
        return t["name"] if isinstance(t, dict) else t.name

    return [t for t in tools if _name(t) not in external_names]


MIN_INTERNAL_TOOLS = 80


def test_every_module_discovers(tools):
    assert len(_internal(tools)) >= MIN_INTERNAL_TOOLS
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
    assert len(_internal(tools)) >= MIN_INTERNAL_TOOLS   # the rest still came through
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
    assert result.is_error is True, f"{tool}{args} did not report isError"


@pytest.mark.parametrize("tool,args", [
    ("Read", {"file_path": "/etc/hostname"}),
    ("Bash", {"command": "echo ok"}),
    ("Glob", {"pattern": "*.py", "path": str(ROOT / "agent_mcp")}),
])
async def test_successes_do_not_set_is_error(tool, args):
    result = await M.call_tool(tool, args)
    assert isinstance(result, CallToolResult)
    assert result.is_error is False


async def test_unknown_tool_is_an_error_result():
    result = await M.call_tool("NoSuchToolAtAll", {})
    assert isinstance(result, CallToolResult)
    assert result.is_error is True


# ── Schema hygiene ───────────────────────────────────────────────────────────

def test_no_toplevel_additional_properties_false(tools):
    """P2-2: the aggregator strips a legacy `_session_id` argument, but the
    SDK validates arguments against inputSchema *before* the handler runs.
    A top-level `additionalProperties: false` would reject the call
    outright. Session id moved to `_meta`, and this keeps the door shut.
    """
    offenders = [
        t.name for t in tools
        if (t.input_schema or {}).get("additionalProperties") is False
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
        props = (t.input_schema or {}).get("properties") or {}
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



def _unverifiable_names():
    """Tool names we cannot check right now because their module is degraded.

    The Thunderbird bridge is a gitignored build artifact, so `thunderbird`
    exports zero tools in any worktree — and the self-modification gate runs
    this suite inside one. Rather than silently ignoring those names, say
    explicitly that they are unverifiable in this environment: a stale entry
    for a module that IS loaded still fails.
    """
    degraded = {n for n, st in M._discovery_status.items() if not st.get("ok") or not st.get("tools")}
    if not degraded:
        return set()
    return {n for n in A.READ_ONLY | A.DESTRUCTIVE | A.IDEMPOTENT | A.PLAN_MODE_ALWAYS_ALLOWED
            if n.split("_")[0] in {"email", "calendar", "contacts", "tasks"}
            and "thunderbird" in degraded}


def test_annotation_tables_have_no_stale_entries(names):
    exempt = _unverifiable_names()
    for label, table in (("READ_ONLY", A.READ_ONLY),
                         ("DESTRUCTIVE", A.DESTRUCTIVE),
                         ("IDEMPOTENT", A.IDEMPOTENT),
                         ("PLAN_MODE_ALWAYS_ALLOWED", A.PLAN_MODE_ALWAYS_ALLOWED)):
        stale = sorted(table - names - {"ToolSearch"} - exempt)  # ToolSearch is harness-side
        assert stale == [], f"{label} names tools that no longer exist: {stale}"


def test_destructive_tools_are_not_read_only():
    assert A.DESTRUCTIVE & A.READ_ONLY == frozenset()


def test_plan_mode_blocks_actuators_and_spares_controls(names):
    blocked = set(A.plan_mode_blocked_tools(names))
    for actuator in ("Bash", "Write", "Edit", "Task", "email_send", "vault_write",
                     "fact_add", "discord_send", "browser_click", "memory_add",
                     "autonomy_write_task", "http_request"):
        if actuator not in names:
            continue  # its module is degraded here (see _unverifiable_names)
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
    assert "vault_write" in blocked
    if "email_send" in names:   # thunderbird is absent in a worktree
        assert "email_send" in blocked
    assert "ExitPlanMode" not in blocked


# ── SDK version compatibility ────────────────────────────────────────────────

def test_field_accessors_handle_both_sdk_naming_conventions():
    """mcp 2.x renames model fields to snake_case in Python.

    Construction stays compatible (the models set `populate_by_name`), but
    attribute reads do not — and `getattr(result, "isError", False)`
    returns False rather than raising on a 2.x result, which would
    silently mark every failed tool call a success. These accessors are
    what stands between that rename and a repeat of P1-2.
    """
    from app.harness.mcp_pool import _input_schema, _is_error

    class V2Result:            # snake_case, as mcp 2.x exposes it
        is_error = True

    class V1Result:            # camelCase, as mcp 1.x exposes it
        isError = True

    class NoFlag:
        pass

    assert _is_error(V2Result()) is True
    assert _is_error(V1Result()) is True
    assert _is_error(NoFlag()) is False

    class V2Tool:
        input_schema = {"type": "object", "properties": {"a": {}}}

    class V1Tool:
        inputSchema = {"type": "object", "properties": {"b": {}}}

    assert _input_schema(V2Tool())["properties"] == {"a": {}}
    assert _input_schema(V1Tool())["properties"] == {"b": {}}
    assert _input_schema(NoFlag()) == {"type": "object", "properties": {}}


def test_mcp_is_pinned_to_2x():
    """agent_mcp/main.py passes handlers as Server(...) constructor
    arguments and reads model fields by their snake_case names; neither
    works on mcp 1.x, so the floor is a hard requirement, not a
    preference."""
    req = (ROOT / "requirements.txt").read_text()
    assert "mcp>=2.1.0,<3" in req


def test_negotiates_the_stateless_protocol():
    """The point of the 2.x move is the 2026-07-28 stateless core."""
    from mcp_types.version import LATEST_PROTOCOL_VERSION, MODERN_PROTOCOL_VERSIONS

    assert "2026-07-28" in MODERN_PROTOCOL_VERSIONS
    assert LATEST_PROTOCOL_VERSION == "2026-07-28"


def test_aggregator_serves_streamable_http_not_sse():
    """The legacy HTTP+SSE transport is deprecated upstream; the
    aggregator must not still be mounting it."""
    paths = {getattr(r, "path", None) for r in M.starlette_app.routes}
    assert "/mcp" in paths
    assert "/health" in paths
    assert "/sse" not in paths and "/messages/" not in paths


# ── Server config resolution ─────────────────────────────────────────────────

def test_configured_transport_survives_into_the_pool_config():
    """`_get_mcp_servers` must not drop the transport it was given.

    Its transport whitelist used to be a hardcoded ("sse", "http"), and
    anything else fell through to the stdio branch. When lloyd-mcp moved to
    `type: streamable-http` that produced `{"command": "python", "args":
    []}` — a config that spawns a bare Python REPL and hangs `pool.open()`
    forever. Nothing caught it: the aggregator was healthy, `/health`
    returned 200, and only an actual agent turn would have failed.
    """
    from app.mcp_discovery import _get_mcp_servers

    servers = _get_mcp_servers()
    assert servers, "no MCP servers resolved from config"
    for name, cfg in servers.items():
        assert "type" in cfg, f"{name} lost its transport type"
        if cfg["type"] in ("http", "streamable-http", "streamable_http", "sse"):
            assert cfg.get("url"), f"{name} is an HTTP transport with no url"
        else:
            assert cfg.get("command"), f"{name} is stdio with no command"


def test_unknown_transport_type_raises():
    import app.mcp_discovery as D

    original = D.CONFIG.get("mcp_servers")
    D.CONFIG["mcp_servers"] = {"bogus": {"type": "carrier-pigeon", "url": "x"}}
    try:
        with pytest.raises(ValueError, match="unknown transport type"):
            D._get_mcp_servers()
    finally:
        D.CONFIG["mcp_servers"] = original


def test_http_transport_without_url_raises():
    import app.mcp_discovery as D

    original = D.CONFIG.get("mcp_servers")
    D.CONFIG["mcp_servers"] = {"bogus": {"type": "streamable-http"}}
    try:
        with pytest.raises(ValueError, match="no url"):
            D._get_mcp_servers()
    finally:
        D.CONFIG["mcp_servers"] = original


def test_every_transport_dispatcher_handles_every_transport():
    """Every place that branches on a transport must know all of them.

    Three separate functions dispatch on `cfg["type"]`, and each one that
    grew its own literal list has broken in turn: `_get_mcp_servers`
    emitted a stdio config for `streamable-http` and hung the pool, and
    `_discover_mcp_tools` did the same thing one function later and left
    the Tools page stuck on "Discovering tools..." forever. They share
    HTTP_TRANSPORTS/STDIO_TRANSPORTS now; this keeps them sharing it.
    """
    import inspect

    from app.harness import mcp_pool
    from app import mcp_discovery
    from app.mcp_discovery import HTTP_TRANSPORTS, STDIO_TRANSPORTS

    dispatchers = {
        "MCPPool._open_session": inspect.getsource(mcp_pool.MCPPool._open_session),
        "_get_mcp_servers": inspect.getsource(mcp_discovery._get_mcp_servers),
        "_discover_mcp_tools": inspect.getsource(mcp_discovery._discover_mcp_tools),
    }
    for name, src in dispatchers.items():
        uses_shared_list = "HTTP_TRANSPORTS" in src
        for transport in HTTP_TRANSPORTS + STDIO_TRANSPORTS:
            assert uses_shared_list or f'"{transport}"' in src, (
                f"{name} does not handle transport {transport!r} — it should "
                f"branch on HTTP_TRANSPORTS/STDIO_TRANSPORTS, not its own list"
            )


async def test_discovery_resolves_the_configured_transport():
    """`_discover_mcp_tools` must work against the transport config
    actually names — this is what the Tools page calls."""
    from app.mcp_discovery import _discover_mcp_tools, _get_mcp_servers

    for name, cfg in _get_mcp_servers().items():
        found, err = await _discover_mcp_tools(name, cfg)
        assert err is None, f"{name} ({cfg.get('type')}): {err}"
        assert len(_internal(found)) >= MIN_INTERNAL_TOOLS, \
            f"{name} returned only {len(_internal(found))} internal tools"


def test_no_inline_mcp_server_config_anywhere_in_the_repo():
    """Nobody may write an MCP server-config dict inline.

    This is the bug that keeps recurring. `DEFAULT_LLOYD_MCP_URL` moved
    from /sse to /mcp and the `"type": "sse"` literals beside it did not,
    so an SSE client GET'd the Streamable HTTP endpoint and hung. It was
    five callsites, in five different directories, found three separate
    times — because each grep was scoped to wherever the last one was
    found. This walks the whole tree instead.

    Use `DEFAULT_LLOYD_MCP_SERVERS` (or `_get_mcp_servers()`), never a
    literal.
    """
    import re

    pattern = re.compile(r'\{\s*["\']type["\']\s*:\s*["\'](?:sse|http|streamable[-_]http)["\']')
    skip_dirs = {".git", ".venvs", "node_modules", "__pycache__", "_pipeline",
                 "web", "logs", "sessions", "agent-services"}
    # The modules that legitimately define or validate the transports.
    allowed = {"app/harness/mcp_pool.py", "app/mcp_discovery.py",
               "tests/test_mcp_layer.py", "tests/test_mcp_transport.py"}

    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if any(part in skip_dirs for part in path.relative_to(ROOT).parts):
            continue
        if rel in allowed:
            continue
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{rel}:{i}")
    assert offenders == [], (
        "inline MCP server config found — use DEFAULT_LLOYD_MCP_SERVERS: "
        + ", ".join(offenders)
    )
