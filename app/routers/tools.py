"""Tools tab endpoints — MCP server inspection and tool enable/disable.

Built-in tools (Bash/Read/Write/Edit/Grep/Glob/Task) live inside the
lloyd-mcp aggregator now, so this router has only two real concepts:
servers and the tools each server exposes. Disabling a built-in is the
same operation as disabling any other tool: write its bare name into
the parent server's `disabled_tools` list.
"""

import time

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import CONFIG
from app.paths import LLOYD_HOME
from app.mcp_discovery import (
    _MCP_SERVER_META,
    _tools_cache,
    _TOOLS_CACHE_TTL,
    _categorize_tool,
    _discover_mcp_tools,
)


router = APIRouter()


@router.get("/api/tools")
async def get_tools():
    """List every MCP server with its discovered tools."""
    now = time.time()
    servers = []
    for server_name, cfg in CONFIG.get("mcp_servers", {}).items():
        server_enabled = cfg.get("enabled", True)
        disabled_tools = set(cfg.get("disabled_tools", []))
        meta = _MCP_SERVER_META.get(server_name, {})
        label = meta.get("label", server_name.replace("-", " ").title())
        description = meta.get("description", "")

        cached = _tools_cache.get(server_name)
        if cached and (now - cached["ts"]) < _TOOLS_CACHE_TTL:
            raw_tools, error = cached["tools"], cached["error"]
        elif server_enabled:
            raw_tools, error = await _discover_mcp_tools(server_name, cfg)
            # Only cache successful discoveries. A failed discovery (timeout,
            # transient network blip during MCP SSE handshake) used to get
            # cached for the full 5-minute TTL, leaving the Tools page with
            # a "TaskGroup … sub-exception" error long after the server
            # recovered. Skipping the cache write on failure means the next
            # call retries immediately.
            if error is None:
                _tools_cache[server_name] = {"tools": raw_tools, "error": None, "ts": now}
        else:
            raw_tools, error = [], None

        servers.append({
            "name": server_name,
            "label": label,
            "description": description,
            "enabled": server_enabled,
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "enabled": t["name"] not in disabled_tools,
                    "category": _categorize_tool(t["name"]),
                }
                for t in raw_tools
            ],
            "error": error,
        })

    return JSONResponse({"servers": servers})


@router.post("/api/tool-toggle")
async def toggle_tool(request: Request):
    """Toggle a server or individual tool, persisting changes to config.yaml."""
    data = await request.json()
    toggle_type = data.get("type")  # "server" | "tool"
    enabled = bool(data.get("enabled", True))
    config_path = LLOYD_HOME / "config.yaml"

    if toggle_type == "server":
        server_name = data.get("server", "")
        if server_name not in CONFIG.get("mcp_servers", {}):
            raise HTTPException(status_code=404, detail=f"Server not found: {server_name}")
        CONFIG["mcp_servers"][server_name]["enabled"] = enabled
        _tools_cache.pop(server_name, None)

    elif toggle_type == "tool":
        server_name = data.get("server", "")
        tool_name = data.get("tool", "")
        if server_name not in CONFIG.get("mcp_servers", {}):
            raise HTTPException(status_code=404, detail=f"Server not found: {server_name}")
        cfg = CONFIG["mcp_servers"][server_name]
        disabled = cfg.get("disabled_tools", [])
        if not enabled and tool_name not in disabled:
            disabled.append(tool_name)
        elif enabled and tool_name in disabled:
            disabled.remove(tool_name)
        cfg["disabled_tools"] = disabled

    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {toggle_type}")

    with open(config_path, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    _tools_cache.clear()

    return JSONResponse({"servers_updated": True})
