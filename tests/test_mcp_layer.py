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
import json
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

# A `_meta` block with a session id, which is what `app/harness/mcp_pool.py`
# stamps on every `tools/call` the harness makes. Needed here since #1053: a
# state-changing call that names no session is refused before the module handler
# runs, so `Bash` with no id would return the refusal — an error result, which
# would let the failure test below pass without ever reaching `builtin_bash`.
HARNESS_META = {"lloyd/session_id": "20260918_layer_test_session"}


@pytest.mark.parametrize("tool,args", [
    ("Read", {"file_path": "/definitely/not/here"}),
    ("Bash", {"command": "exit 7"}),
    ("Bash", {"command": "pwd", "cwd": "not-absolute"}),
    ("fact_get", {}),
    ("vault_read", {}),
    ("research_propose", {}),
    ("research_complete", {}),
])
async def test_failures_set_is_error(tool, args):
    result = await M.call_tool(tool, args, HARNESS_META)
    assert isinstance(result, CallToolResult)
    assert result.is_error is True, f"{tool}{args} did not report isError"


@pytest.mark.parametrize("tool,args", [
    ("Read", {"file_path": "/etc/hostname"}),
    ("Bash", {"command": "echo ok"}),
    ("Glob", {"pattern": "*.py", "path": str(ROOT / "agent_mcp")}),
])
async def test_successes_do_not_set_is_error(tool, args):
    result = await M.call_tool(tool, args, HARNESS_META)
    assert isinstance(result, CallToolResult)
    assert result.is_error is False


async def test_undeclared_argument_reaches_handler_and_is_ignored():
    """#1125: no layer validates `arguments` against the tool's inputSchema.
    A key no schema declares is not a dispatch error; it reaches the handler,
    which ignores it. Comments that claimed otherwise were corrected, and
    this pins the real contract so a future claim can be checked against it.
    """
    target = ROOT / "agent_mcp" / "annotations.py"
    result = await M.call_tool(
        "Read", {"file_path": str(target), "bogus_key_zzz": 123}, HARNESS_META)
    assert isinstance(result, CallToolResult)
    assert result.is_error is False
    first_line = target.read_text().splitlines()[0]
    assert first_line in result.content[0].text


async def test_unknown_tool_is_an_error_result():
    result = await M.call_tool("NoSuchToolAtAll", {})
    assert isinstance(result, CallToolResult)
    assert result.is_error is True


# The generic case above names a tool that never existed. These two name tools
# that used to be served and were deleted (#1077), which is a different failure
# to pin: `tools/list` is cached by the client for TOOLS_LIST_TTL_MS, so a
# session opened before the landing keeps sending the old name for up to that
# window, and the answer it gets comes from `main`'s dispatch map, not from
# `agent_mcp.facts` — the aggregator resolves the name against a map discovery
# rebuilt, and returns its own payload (no `code` field, unlike the module
# handler's `_err`). That is the boundary a real client crosses, so the
# assertion sits here rather than only in tests/test_result_shape.py.
@pytest.mark.parametrize("name", ("fact_neighbors", "fact_path"))
async def test_a_deleted_kg_graph_read_is_answered_by_the_aggregator(name):
    await M.list_tools()          # rebuild the map, as a live server did at discovery
    assert name not in M._dispatch, f"{name} still routes to a handler"
    result = await M.call_tool(name, {})
    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert json.loads(result.content[0].text) == {"error": f"Unknown tool: {name}"}


# ── Schema hygiene ───────────────────────────────────────────────────────────

def test_no_toplevel_additional_properties_false(tools):
    """P2-2: nothing validates `arguments` against inputSchema today — the
    low-level server checks the params model only and unknown keys reach
    the handler (see `test_undeclared_argument_reaches_handler_and_is_ignored`).
    So a top-level `additionalProperties: false` would be a promise nothing
    keeps, and it is the one schema shape that would start rejecting the
    harness-injected keys (`_session_id`, a leaked `summary`) the day anyone
    switches validation on. Session id moved to `_meta`; this keeps the door
    shut.
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
                     "autonomy_write_task", "http_request", "research_propose",
                     "research_complete"):
        if actuator not in names:
            continue  # its module is degraded here (see _unverifiable_names)
        assert actuator in blocked, f"{actuator} should be blocked in plan mode"
    for control in ("ExitPlanMode", "EnterPlanMode", "TodoWrite", "Read", "Grep",
                    "vault_search", "fact_get", "skills_search", "mc_navigate",
                    "research_list", "research_stats", "research_next"):
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


# ── The grant gate meets the tool surface (#724) ─────────────────────────────
# `autonomy_write_task` rewrites the fields the scheduler reads to decide
# whether and when a task runs. The gate lives in the harness hook; the write
# lives here, in the server process. Clause 2 of #724 is that the interactive
# half of that seam is untouched: chat and owner-Discord turns upsert those
# fields exactly as they did before the tier promotion. Pinned across the seam —
# hook decision, then the real handler's bytes on disk.

def _seam_hook(tmp_path, scope):
    """A hook registry carrying the same gate the live dispatch paths install,
    over a grant store in `tmp_path` (the real one is the fleet's)."""
    from app.harness import HookRegistry
    from app.harness import policy

    store = policy.GrantStore(tmp_path / "seam-grants.db")
    store.ensure_schema()
    hooks = HookRegistry()
    # store= explicitly: omitted, the hook installs the FLEET's real store, and
    # a test that denies nothing because a live grant happens to exist is the
    # false green this suite exists to catch.
    policy.install_policy_hook(hooks, store=store, scope=scope)
    return hooks, store


def _seam_decision(hooks, args: dict) -> dict:
    """The hook's answer, in the shape the SDK sees."""
    out = asyncio.run(hooks.fire_pre_tool_use(
        session_id="", tool_name="autonomy_write_task", tool_input=args)) or {}
    spec = out.get("hookSpecificOutput") or {}
    # An absent decision means the hook let it through: `install_policy_hook`
    # denies by returning a `deny` decision and passes by returning nothing but
    # `continue_`, the same shape `tests/unit/test_grant_policy._denied` relies on.
    return {"decision": spec.get("permissionDecision") or "allow",
            "reason": spec.get("permissionDecisionReason") or ""}


def _seam_dispatch(name: str, args: dict) -> dict:
    """Call the tool the way the aggregator does, so an error shape is the
    server's and not a test's."""
    import json

    import agent_mcp.autonomy as AZ

    return json.loads(AZ._handle_write(args))


def test_interactive_scope_still_upserts_the_gated_fields(tmp_path, monkeypatch):
    """Chat and owner-Discord are behaviourally unchanged: every field #724
    gates is written, and written to the file the scheduler reads."""
    import agent_mcp.autonomy as AZ

    d = tmp_path / "autonomy"
    d.mkdir()
    (d / "68-morning-brief-triage.md").write_text(
        "---\nid: 68\nname: Morning Brief\nstatus: draft\n"
        "depends_on: 39\nfrequency: daily\n---\n\nbody\n", encoding="utf-8")
    monkeypatch.setattr(AZ, "AUTONOMY_DIR", d)

    hooks, _store = _seam_hook(tmp_path, "interactive")
    dec = _seam_decision(hooks, {"id": 68, "status": "up_next", "depends_on": None,
                                 "frequency": "weekdays", "auto_advance": False,
                                 "skill_name": "morning-brief-triage"})
    assert dec["decision"] == "allow", dec

    out = _seam_dispatch("autonomy_write_task", {
        "id": 68, "status": "up_next", "frequency": "weekdays",
        "auto_advance": False, "skill_name": "morning-brief-triage"})
    assert "error" not in out, out
    text = (d / "68-morning-brief-triage.md").read_text()
    assert "status: up_next" in text
    assert "frequency: weekdays" in text
    assert "auto_advance: false" in text
    assert "skill_name: morning-brief-triage" in text


def test_unattended_scope_is_denied_before_the_handler_runs(tmp_path, monkeypatch):
    """The other side of the same seam, in the same file: under a
    `scheduled-task`-shaped scope with nothing declared, the hook denies and the
    task file is byte-for-byte what it was. The denial is the harness stopping
    the call — the server never sees it, which is why no handler-side check is
    needed and why the same code path serves every worker source."""
    import agent_mcp.autonomy as AZ

    d = tmp_path / "autonomy"
    d.mkdir()
    original = ("---\nid: 68\nname: Morning Brief\nstatus: draft\n"
                "frequency: daily\n---\n\nbody\n")
    (d / "68-morning-brief-triage.md").write_text(original, encoding="utf-8")
    monkeypatch.setattr(AZ, "AUTONOMY_DIR", d)

    hooks, _store = _seam_hook(tmp_path, "worker:scheduled-task")
    dec = _seam_decision(hooks, {"id": 68, "status": "up_next"})
    assert dec["decision"] == "deny", dec
    assert "worker:scheduled-task" in dec["reason"], dec
    assert "autonomy_write_task" in dec["reason"], dec
    assert "target #68" in dec["reason"], dec
    assert "status" in dec["reason"], dec
    assert (d / "68-morning-brief-triage.md").read_text() == original


def test_a_declared_grant_reopens_the_seam(tmp_path, monkeypatch):
    """The nightly route: the human wrote the authority into the task's
    frontmatter, the dispatcher materialised it, and the same call that was
    denied a moment ago now passes. Without this the fix is a stop sign, not a
    gate, and the #40 -> #68/#85 re-arm breaks on the first nightly.

    What this does NOT prove, and the two nodes after it exist to prove: that
    `run_task` reaches that sync call. The row here is synced by the test, with
    the task dict handed in explicitly, so the dispatcher's own reading of
    `grants` from the file — clause 3's parser output — is still unexercised,
    and a sync that ran after the hook was installed, or not at all, would leave
    this green. An ungranted `autonomy_write_task` was tier 1 before #724, so no
    test on this path ever needed to get a run that far.

    It is also the reason `sync_task_grants` runs BEFORE the `GrantError` guard:
    a malformed block must not leave the rows its last good self minted live in
    the store while its file now says something else. That ordering is load-
    bearing and a refactor that "tidies" the two blocks together silently
    un-fixes #534's read-back half."""
    import datetime as dt

    import agent_mcp.autonomy as AZ
    from app.harness import policy

    d = tmp_path / "autonomy"
    d.mkdir()
    (d / "40-nightly-reflection-config.md").write_text(
        "---\nid: 40\nname: Nightly Reflection Config\nstatus: up_next\n"
        "grants:\n- tool: autonomy_write_task\n"
        "  expires_at: '2099-01-01T00:00:00Z'\n  issued_by: alan\n---\n\nbody\n",
        encoding="utf-8")
    (d / "68-morning-brief-triage.md").write_text(
        "---\nid: 68\nname: Morning Brief\nstatus: draft\n---\n\nbody\n",
        encoding="utf-8")
    monkeypatch.setattr(AZ, "AUTONOMY_DIR", d)

    hooks, store = _seam_hook(tmp_path, "autonomy-task:40")
    policy.sync_task_grants(
        store, task_id=40, scope="autonomy-task:40",
        grants=[{"tool": "autonomy_write_task",
                 "expires_at": "2099-01-01T00:00:00Z", "issued_by": "alan"}],
        now=dt.datetime.now(dt.timezone.utc))

    dec = _seam_decision(hooks, {"id": 68, "status": "up_next"})
    assert dec["decision"] == "allow", dec
    out = _seam_dispatch("autonomy_write_task", {"id": 68, "status": "up_next"})
    assert "error" not in out, out
    assert "status: up_next" in (d / "68-morning-brief-triage.md").read_text()


# ── #724: the dispatcher half — run_task arms the turn it built ──────────────
#
# The node above this seam calls `sync_task_grants` with a grant list the test
# wrote out by hand, and `tests/unit/test_grant_policy.py::
# test_the_shipped_nightly_rearm_grant_materialises_and_reopens_the_write` reads
# the shipped #40 block and then syncs it itself, from the tuple it named. Both
# are a mock AT the seam clause 4 names — "sync_task_grants materialises it at
# dispatch". What neither touches is the dispatcher: `run_task` takes the block
# out of `_parse_task_file`'s output (`autonomy.py:1860` and `:1868`), syncs it
# into `default_store()`, and installs the hook three statements later
# (`:1881-1882`). A `grants` key lost between the parse and the sync, or an
# install that ran before the sync, left both green while the task whose only
# authority is its frontmatter was denied on its first nightly. These two nodes
# run the real `autonomy.run_task` and assert on the store rows it leaves.

_GRANT_BLOCK = (
    "---\n"
    "id: 40\n"
    "name: Nightly Reflection Config\n"
    "status: up_next\n"
    "frequency: daily\n"
    "skill_name: nightly-reflection-config\n"
    "grants:\n"
    "- tool: autonomy_write_task\n"
    "  expires_at: '2099-01-01T00:00:00Z'\n"
    "  issued_by: alan\n"
    "  note: nightly re-arm of #68 and #85\n"
    "---\n"
    "\n# Nightly Reflection Config\n"
)

_TASK_ONLY = _GRANT_BLOCK.replace(
    "grants:\n"
    "- tool: autonomy_write_task\n"
    "  expires_at: '2099-01-01T00:00:00Z'\n"
    "  issued_by: alan\n"
    "  note: nightly re-arm of #68 and #85\n", "")


def _grant_dispatch_env(monkeypatch, tmp_path, frontmatter: str):
    """Run the real `autonomy.run_task` over a task file, with only the engines
    replaced: the model, the brain, the prompt builder, the run-record writer and
    the grant DB path. Everything between the parse and the hook install is the
    production code, which is the point — the row a run leaves in the store is
    what the next nightly will be judged against, so it is what gets asserted.

    `LLOYD_GRANT_DB` is what makes the registry the turn is handed and the store
    the test reads one pair of objects: `install_policy_hook` is called with no
    store, so the hook resolves `default_store()` itself, and that resolver reads
    this variable. The cache is the resolver's own, so it is cleared the way a
    cold dispatcher process would find it."""
    import autonomy as AUT

    from app.harness import policy

    db = tmp_path / "grants.db"
    monkeypatch.setenv("LLOYD_GRANT_DB", str(db))
    policy._STORE_CACHE.clear()
    # A dispatcher that got this far is a booted one, and boot owns the schema
    # (`GrantStore.__init__` does not create it). Without this the run dies on
    # `no such table: authority_grants` and the node measures the fixture.
    store = policy.GrantStore(db)
    store.ensure_schema()

    # The file is real and the reader is real: `run_task` reaches `grants`
    # through `_parse_task_file`, so a block the parser cannot return — the exact
    # failure clause 3 exists for — fails this node rather than being hand-
    # patched around. The frontmatter carries the timeout and retry fields the
    # failure path reads, so nothing else has to be stubbed.
    task_file = tmp_path / "40-nightly-reflection-config.md"
    task_file.write_text(frontmatter, encoding="utf-8")

    monkeypatch.setattr(AUT, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(AUT, "AUTONOMY_DIR", tmp_path)
    monkeypatch.setattr(AUT, "_load_skill_content", lambda s: "SKILL BODY")
    monkeypatch.setattr(AUT, "_get_model_env", lambda m: {})
    monkeypatch.setattr(AUT, "_task_inner_voice", lambda t: False)
    monkeypatch.setattr(AUT, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(AUT, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(AUT, "_write_run_record", lambda *a, **k: tmp_path / "r.md")
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")

    captured: dict = {}

    async def _run_query(messages, options):
        # `options` here is the real `RunOptions` `run_task` built — the object
        # clause 4 names, not a stand-in for it — so the hook this node reads is
        # the registry the model's own loop would consult.
        captured["options"] = options
        captured["hooks"] = options.hooks
        # The two events `run_task` actually reads: it concatenates `text` over
        # `text_delta` into the response, and keeps the last `assistant_message`
        # as the run's terminal block. A stream with only the second is an empty
        # run, and `run_task` correctly says so — which is not what this node is
        # asserting about.
        block = "re-armed #68 and #85\n"
        yield {"type": "text_delta", "text": block}
        yield {"type": "assistant_message", "text": block}

    import app.harness as harness_mod
    monkeypatch.setattr(harness_mod, "run_query", _run_query)
    return captured, store


def test_run_task_materialises_the_declared_grant_before_arming_the_hook(
        tmp_path, monkeypatch):
    """Clause 4, at the seam it names: the block on the task file becomes a row
    in the dispatch store, and the hook that judges the task's next call is
    installed holding the scope that row is addressed to."""
    import asyncio

    import autonomy as AUT

    captured, store = _grant_dispatch_env(monkeypatch, tmp_path, _GRANT_BLOCK)
    out = asyncio.run(AUT.run_task(40))

    assert out.get("success") is True, out
    rows = store.live(scope="autonomy-task:40")
    assert [r["tool_pattern"] for r in rows] == ["autonomy_write_task"], rows
    assert rows[0]["minted_by"] == "frontmatter:40", (
        "the row exists but not from this file, so the audit trail does not lead "
        "back to the human who wrote the block")

    # The hook is armed, and armed against the same scope the row carries. An
    # install that never ran, or a scope spelled differently from the sync's,
    # would leave the row unread by the gate and the task denied.
    #
    # The denial side is checked FIRST, on the same registry, because an empty
    # registry answers every call with `{}` and an `!= "deny"` assertion would
    # then pass on a turn that had no gate at all. Same call, same registry,
    # revocation the only difference: the row is what moves the answer.
    hooks = captured["hooks"]
    assert hooks is not None, "the turn was dispatched with no gate at all"

    def _fire() -> dict:
        return asyncio.run(hooks.fire_pre_tool_use(
            session_id="s", tool_name="mcp__lloyd-mcp__autonomy_write_task",
            tool_input={"id": 68, "status": "up_next"}, tool_use_id="t1"))

    granted = _fire()
    assert granted.get("hookSpecificOutput", {}).get(
        "permissionDecision") != "deny", (
        "the dispatcher minted the grant and then asked a gate that could not "
        "see it: %r" % (granted,))

    store.revoke(rows[0]["id"])
    revoked = _fire()
    decision = revoked.get("hookSpecificOutput", {}).get("permissionDecision")
    assert decision == "deny", (
        "an empty registry answers the same way as an armed one that let the "
        "call through; the allow above measured nothing: %r" % (revoked,))
    assert "autonomy-task:40" in revoked.get(
        "hookSpecificOutput", {}).get("permissionDecisionReason", ""), (
        "denied, but not by the scope whose frontmatter is the authority")


def test_run_task_refuses_to_dispatch_a_task_whose_grants_block_is_malformed(
        tmp_path, monkeypatch):
    """The fail-closed half, and it needs its own node because `run_task` has two
    exits that both mean 'did not run'.

    A corrupt block must cost the run before the turn starts, and must leave no
    row — a task whose YAML is broken must not be able to widen its own authority
    by editing bytes. The store is empty either way, so the store cannot tell
    this from the case where the file declares nothing; only the refusal names
    which happened."""
    import asyncio

    import autonomy as AUT

    broken = _GRANT_BLOCK.replace("expires_at: '2099-01-01T00:00:00Z'",
                                  "expires_at: not-a-date")
    captured, store = _grant_dispatch_env(monkeypatch, tmp_path, broken)
    out = asyncio.run(AUT.run_task(40))

    assert out.get("success") is not True, out
    assert "grants" in str(out.get("error", "")).lower(), out
    assert store.live() == [], "a block nobody could read still put a grant on the books"
    assert "hooks" not in captured, (
        "the turn was dispatched anyway: the refusal is decoration and the task "
        "runs with whatever the accidental scope resolves to")


# ── Catalog size (2026-09-23) ────────────────────────────────────────────────
#
# `harness.tool_search` is off by decision (#456), so every chat and
# session-backed worker turn is handed every schema, and nothing measured what
# that cost: #639 found the only token test guarded the disabled gist path.
# At 156 tools the advertised JSON was ~35.2k real tokens a request. These pin
# the tree's own part of it (the Thunderbird bridge is gitignored, so its ~40
# tools are measured separately in test_thunderbird_discovery_cache).

# The twelve tools retired as duplicates or subsumed on 2026-09-23, plus the two
# KG graph reads deleted on 2026-09-24 (#1077) — advertised for their whole
# life, never called, while `fact_add` wrote the same store constantly. Two
# different reasons, one table: what the test below refuses is the same in both
# cases, a name on the advertised surface or in an annotation table. A name
# coming back is a decision, not drift, so it has to be taken out of here.
#
# The annotation half is what the deletion had to reach: `READ_ONLY`
# (agent_mcp/annotations.py) is the table the plan-mode blocked set is derived
# from at discovery, so an entry left behind after the tool goes is a claim
# about a tool nothing serves.
RETIRED_TOOLS = frozenset({
    "fact_check", "fact_profile",
    "remember", "recall", "forget", "improve",
    "browser_type",
    "autoresearch_round", "autoresearch_promote", "autoresearch_bench_add",
    "autoresearch_bench_list", "autoresearch_ledger_query",
    "fact_path", "fact_neighbors",
})

# chars/4 via `app.compaction.estimate_tokens`, over the OpenAI-shaped JSON the
# harness actually sends. Measured 20,524 at the 2026-09-23 trim (104 tools);
# the primary's own tokenizer read 20,582, so the estimate is within 1% on this
# JSON. The ceiling is ~10% over,
# so growth past it is a conscious raise with a reason, not a slow creep.
INTERNAL_CATALOG_TOKEN_CEILING = 22_500


def test_retired_tools_stay_retired(names):
    back = sorted(RETIRED_TOOLS & names)
    assert back == [], f"retired tools are advertised again: {back}"
    tables = (A.READ_ONLY | A.DESTRUCTIVE | A.IDEMPOTENT | A.REPEAT_EXPECTED
              | A.PLAN_MODE_ALWAYS_ALLOWED)
    assert sorted(RETIRED_TOOLS & tables) == []


def test_advertised_catalog_stays_under_its_token_ceiling(tools):
    import json

    from app.compaction import estimate_tokens
    from app.harness.tool_schema import mcp_tool_to_openai

    advertised = [t for t in _internal(tools) if not t.name.startswith("_")]
    payload = json.dumps([mcp_tool_to_openai(t.model_dump(by_alias=True))
                          for t in advertised])
    tokens = estimate_tokens(payload)
    assert tokens < INTERNAL_CATALOG_TOKEN_CEILING, (
        f"the advertised catalog is {tokens} estimated tokens over "
        f"{len(advertised)} tools, past the {INTERNAL_CATALOG_TOKEN_CEILING} "
        "ceiling; trim a description or raise the ceiling with a reason")
