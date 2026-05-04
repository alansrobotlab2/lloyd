"""MCP server configuration + tool discovery.

`_get_mcp_servers()` builds the live server-config dict for the harness
from `CONFIG`. We deliberately do not cache this at module scope —
callers invoke the function each request so config edits via
`/api/tool-toggle` take effect immediately without requiring any
module-level rebinding.

Built-in tools (Bash, Read, Write, Edit, Grep, Glob, Task) live inside
the lloyd-mcp aggregator (agent_mcp/builtin_*.py). To disable any of
them, add the bare name to `mcp_servers.lloyd-mcp.disabled_tools` —
the harness's bare-name aliasing in `tool_schema.py` blocks both the
bare and namespaced forms at advertise + dispatch time.
"""

import asyncio
import json

from app.config import CONFIG

_MCP_SERVER_META: dict[str, dict] = {
    "lloyd-mcp": {
        "label": "Lloyd MCP",
        "description": "Unified aggregator: built-in tools (Bash/Read/Write/Edit/Grep/Glob/Task) + domain modules (autonomy, backlog, browser, facts, vault, mission control, subliminal, HTTP, Thunderbird, pipeline, ambient, autoresearch, skills, session)",
    },
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
    """Build disallowed_tools list from config for harness RunOptions."""
    disallowed: list[str] = []
    for server_name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        for tool_name in cfg.get("disabled_tools", []):
            disallowed.append(f"mcp__{server_name}__{tool_name}")
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
