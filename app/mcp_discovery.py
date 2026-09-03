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
import json

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
PLAN_MODE_BLOCKED_TOOLS = ("Write", "Edit", "Bash")


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
        disallowed.extend(PLAN_MODE_BLOCKED_TOOLS)
    return disallowed


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


async def _discover_mcp_tools(server_name: str, cfg: dict) -> tuple[list[dict], str | None]:
    """Discover tools from an MCP server. Supports SSE and stdio transports."""
    server_type = cfg.get("type", "stdio")

    if server_type in ("sse", "http"):
        from mcp.client.sse import sse_client
        from mcp import ClientSession
        url = cfg.get("url", "")
        try:
            async with sse_client(url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return [{"name": t.name, "description": t.description or ""} for t in result.tools], None
        except Exception as e:
            return [], str(e)

    # stdio fallback
    command = cfg.get("command", "python")
    args = cfg.get("args", [])
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            command, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        init_msg = (json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lloyd-inspector", "version": "1.0"},
            },
        }) + "\n").encode()
        proc.stdin.write(init_msg)
        await proc.stdin.drain()

        await asyncio.wait_for(proc.stdout.readline(), timeout=15.0)

        tools_msg = (json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        }) + "\n").encode()
        proc.stdin.write(tools_msg)
        await proc.stdin.drain()

        tools_line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
        resp = json.loads(tools_line)

        if "error" in resp:
            return [], resp["error"].get("message", "Unknown server error")

        raw = resp.get("result", {}).get("tools", [])
        return [{"name": t["name"], "description": t.get("description", "")} for t in raw], None

    except asyncio.TimeoutError:
        return [], f"Timeout querying {server_name}"
    except Exception as e:
        return [], str(e)
    finally:
        if proc:
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
