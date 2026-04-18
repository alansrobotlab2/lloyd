"""MCP server configuration + tool discovery.

`_get_mcp_servers()` builds the live server-config dict for the SDK from
`CONFIG`. We deliberately do not cache this at module scope — callers
invoke the function each request so config edits via `/api/tool-toggle`
take effect immediately without requiring any module-level rebinding.
"""

import asyncio
import json

from app.config import CONFIG

_BUILTIN_TOOLS = [
    {"name": "Bash",         "description": "Execute bash commands"},
    {"name": "Read",         "description": "Read file contents"},
    {"name": "Write",        "description": "Write files"},
    {"name": "Edit",         "description": "Precise string replacement in files"},
    {"name": "Glob",         "description": "Find files by glob pattern"},
    {"name": "Grep",         "description": "Search file content with regex"},
    {"name": "WebFetch",     "description": "Fetch web page content"},
    {"name": "WebSearch",    "description": "Search the web"},
    {"name": "TodoWrite",    "description": "Manage task list"},
    {"name": "NotebookEdit", "description": "Edit Jupyter notebooks"},
    {"name": "Agent",        "description": "Spawn sub-agents for complex tasks"},
]

_MCP_SERVER_META: dict[str, dict] = {
    "": {"label": "Tools", "description": "Autonomy, backlog, browser, memory, mission control, subliminal, HTTP tools, Thunderbird, pipeline"},
}

_tools_cache: dict[str, dict] = {}  # {server_name: {tools, error, ts}}
_TOOLS_CACHE_TTL = 300.0  # 5 minutes


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


def _get_disallowed_tools() -> list[str]:
    """Build disallowed_tools list from config for SDK options."""
    disallowed: list[str] = list(CONFIG.get("tools", {}).get("disabled_builtin", []))
    for server_name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        for tool_name in cfg.get("disabled_tools", []):
            disallowed.append(f"mcp__{server_name}__{tool_name}")
    return disallowed


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
