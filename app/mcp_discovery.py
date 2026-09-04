"""MCP server configuration + tool discovery.

`_get_mcp_servers()` builds the live server-config dict for the harness
from `CONFIG`. We deliberately do not cache this at module scope —
callers invoke the function each request so config edits via
`/api/tool-toggle` take effect immediately without requiring any
module-level rebinding.

All tools (built-in + domain) are advertised to the model under their
bare MCP name. To disable any tool, add its bare name to
`mcp_servers.<server>.disabled_tools`.
"""

import asyncio

from app.config import CONFIG

_MCP_SERVER_META: dict[str, dict] = {
    "lloyd-mcp": {
        "label": "Lloyd MCP",
        "description": "Unified aggregator: built-in tools (Bash/Read/Write/Edit/Grep/Glob/Task) + domain modules (autonomy, backlog, browser, facts, vault, mission control, HTTP, Thunderbird, pipeline, ambient, autoresearch, skills, session)",
    },
}

_tools_cache: dict[str, dict] = {}  # {server_name: {tools, error, ts}}
_TOOLS_CACHE_TTL = 300.0  # 5 minutes


# Tool → category mapping. Categories are derived from the source agent_mcp
# module each tool lives in (e.g., facts.py → "Memory: Facts"). Built-in
# tools are bare-named; everything else uses a stable module prefix.
# Order matters for prefix rules — first match wins.
_TOOL_EXACT_CATEGORY: dict[str, str] = {
    "Bash": "Shell",
    "Read": "Filesystem",
    "Write": "Filesystem",
    "Edit": "Filesystem",
    "Grep": "Filesystem",
    "Glob": "Filesystem",
    "Task": "Agents",
    "session_recall": "Memory: Session",
    "session_inject_context": "Memory: Session",
    "ambient_decide": "Ambient",
}

_TOOL_PREFIX_CATEGORY: list[tuple[str, str]] = [
    ("fact_", "Memory: Facts"),
    ("vault_", "Memory: Vault"),
    ("memory_", "Memory: Session"),
    ("ambient_", "Ambient"),
    ("autonomy_", "Autonomy"),
    ("autoresearch_", "Autoresearch"),
    ("backlog_", "Backlog"),
    ("browser_", "Browser"),
    ("discord_", "Discord"),
    ("chat_", "Mission Control"),
    ("skills_", "Skills"),
    ("http_", "HTTP"),
    ("tb_", "Thunderbird"),
    ("email_", "Email"),
    ("calendar_", "Calendar"),
    ("contacts_", "Contacts"),
]


def _categorize_tool(name: str) -> str:
    """Return the user-facing category label for a tool name."""
    if name in _TOOL_EXACT_CATEGORY:
        return _TOOL_EXACT_CATEGORY[name]
    for prefix, label in _TOOL_PREFIX_CATEGORY:
        if name.startswith(prefix):
            return label
    return "Other"


def _get_mcp_servers() -> dict[str, dict]:
    """Build MCP server configs for SDK options, filtering out disabled servers."""
    servers = {}
    for name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        server_type = cfg.get("type", "stdio")
        if server_type in ("sse", "http"):
            servers[name] = {"type": server_type, "url": cfg["url"]}
        else:
            servers[name] = {
                "command": cfg.get("command", "python"),
                "args": cfg.get("args", []),
            }
    return servers


# Plan B — tools that primary cannot use while in plan mode. The plan
# ritual is research-only: read tools, ToolSearch, TodoWrite, and the
# plan-mode tools themselves stay allowed; the actuator tools are
# blocked until ExitPlanMode commits or cancel flips plan_mode off.
#
# This tuple is now only the FLOOR. The real list is derived from the
# `readOnlyHint` annotations in `agent_mcp.annotations` (see
# `_plan_mode_blocked`), which covers every actuator rather than the three
# that happened to be listed here — `email_send`, `vault_write`,
# `discord_send`, `fact_add` and ~55 others used to sail straight through
# a "read-only" plan-mode turn.
PLAN_MODE_BLOCKED_TOOLS = ("Write", "Edit", "Bash")

# Tool-name universe, recorded by the harness after MCP discovery. Plan
# mode needs to know which tools exist in order to block the mutating
# ones; discovery is the only place that knows, and it happens after the
# routers have already constructed their RunOptions.
_TOOL_UNIVERSE: set[str] = set()


def record_tool_universe(names) -> None:
    """Record the discovered tool names for annotation-derived gating.

    Called by the harness once per pool open. Idempotent and cheap; a
    superset across servers is fine because the gate only ever removes
    names that are actually advertised.
    """
    if names:
        _TOOL_UNIVERSE.update(names)


def _plan_mode_blocked() -> list[str]:
    """Actuator tools to block while plan mode is active.

    Falls back to the three-tool floor until discovery has run — better a
    narrow gate than an empty one, and the very next turn has the full
    universe recorded.
    """
    if not _TOOL_UNIVERSE:
        return list(PLAN_MODE_BLOCKED_TOOLS)
    from agent_mcp.annotations import plan_mode_blocked_tools

    return plan_mode_blocked_tools(_TOOL_UNIVERSE)


def _get_disallowed_tools(plan_mode: bool = False) -> list[str]:
    """Build disallowed_tools list from config for harness RunOptions.

    Tools are advertised by bare name, so the disallow list uses bare
    names too. The legacy ``mcp__server__tool`` form is also recognized
    by ``build_tool_list`` for any rolled-forward configs.

    `plan_mode` (Plan B) — when true, append the actuator tools
    (Write, Edit, Bash) so primary cannot mutate state while drafting
    a plan. Read tools, TodoWrite, ToolSearch, EnterPlanMode, and
    ExitPlanMode all remain allowed.
    """
    disallowed: list[str] = []
    for _server_name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        for tool_name in cfg.get("disabled_tools", []):
            disallowed.append(tool_name)
    if plan_mode:
        for name in _plan_mode_blocked():
            if name not in disallowed:
                disallowed.append(name)
    return disallowed


DISCOVERY_TIMEOUT_SECONDS = 30.0


def _get_tool_search_kwargs() -> dict:
    """Resolve harness.tool_search.* config into RunOptions kwargs.

    Splatted into RunOptions(**...) at every construction site (chat
    streaming, ambient, sync, voice). Defaults align with RunOptions's
    own dataclass defaults so missing config keys behave sanely.
    """
    cfg = (CONFIG.get("harness") or {}).get("tool_search") or {}
    out: dict = {}
    if "enabled" in cfg:
        out["tool_search_enabled"] = bool(cfg["enabled"])
    if "threshold_tools" in cfg:
        out["tool_search_threshold_tools"] = int(cfg["threshold_tools"])
    if cfg.get("baseline_tools"):
        out["tool_search_baseline"] = list(cfg["baseline_tools"])
    if "max_results_default" in cfg:
        out["tool_search_max_results_default"] = int(cfg["max_results_default"])
    if "max_results_cap" in cfg:
        out["tool_search_max_results_cap"] = int(cfg["max_results_cap"])
    return out


def _root_cause(exc: BaseException) -> str:
    """Innermost message from a (possibly nested) ExceptionGroup.

    anyio task groups repackage a failure as an ExceptionGroup whose str()
    is "unhandled errors in a TaskGroup (1 sub-exception)" — true, and
    useless in a UI. Unwrap to the part a human can act on.
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return str(exc) or exc.__class__.__name__


async def _discover_mcp_tools(server_name: str, cfg: dict) -> tuple[list[dict], str | None]:
    """Discover tools from an MCP server. Supports SSE/HTTP and stdio.

    Both transports go through the SDK client. The stdio path used to be a
    hand-rolled JSON-RPC exchange — write a framed initialize, read one
    line, write tools/list, read one line — which pinned the protocol
    version at 2024-11-05, never drained stderr, and matched no request
    ids. It was the same shape of code as the Thunderbird bridge client,
    with the same defects; there is no reason to keep a second copy.

    Returns (tools, error). Never raises: a server that is down should
    render as an empty, explained row in the Tools page, not a 500.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_type = cfg.get("type", "stdio")

    async def _query() -> list[dict]:
        if server_type in ("sse", "http"):
            ctx = sse_client(cfg.get("url", ""))
        else:
            ctx = stdio_client(StdioServerParameters(
                command=cfg.get("command", "python"),
                args=list(cfg.get("args") or []),
                env=dict(cfg["env"]) if cfg.get("env") else None,
                cwd=cfg.get("cwd") or None,
            ))
        async with ctx as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {"name": t.name, "description": t.description or ""}
                    for t in result.tools
                ]

    try:
        # Bound the whole exchange rather than the read: ClientSession only
        # accepts a per-request timeout on call_tool, and a server that
        # accepts the connection then never speaks would hang discovery —
        # which the Tools page blocks on.
        return await asyncio.wait_for(_query(), timeout=DISCOVERY_TIMEOUT_SECONDS), None
    except asyncio.TimeoutError:
        return [], f"Timeout querying {server_name}"
    except Exception as exc:
        return [], _root_cause(exc)
