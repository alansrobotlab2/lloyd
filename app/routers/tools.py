"""Tools tab endpoints — MCP server inspection and tool enable/disable."""

import time

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import CONFIG
from app.paths import LLOYD_HOME
from app.mcp_discovery import (
    _BUILTIN_TOOLS,
    _MCP_SERVER_META,
    _tools_cache,
    _TOOLS_CACHE_TTL,
    _discover_mcp_tools,
)


router = APIRouter()


@router.get("/api/tools")
async def get_tools():
    """List all available tools: builtin Claude tools + each MCP server with its tools."""
    disabled_builtin = set(CONFIG.get("tools", {}).get("disabled_builtin", []))
    builtin = [
        {
            "name": t["name"],
            "label": t["name"],
            "description": t["description"],
            "enabled": t["name"] not in disabled_builtin,
        }
        for t in _BUILTIN_TOOLS
    ]

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
            _tools_cache[server_name] = {"tools": raw_tools, "error": error, "ts": now}
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
                }
                for t in raw_tools
            ],
            "error": error,
        })

    return JSONResponse({"builtin": builtin, "servers": servers})


@router.post("/api/tool-toggle")
async def toggle_tool(request: Request):
    """Toggle a server or individual tool, persisting changes to config.yaml."""
    data = await request.json()
    toggle_type = data.get("type")  # "server" | "tool" | "builtin"
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

    elif toggle_type == "builtin":
        tool_name = data.get("tool", "")
        tools_cfg = CONFIG.setdefault("tools", {})
        disabled = tools_cfg.get("disabled_builtin", [])
        if not enabled and tool_name not in disabled:
            disabled.append(tool_name)
        elif enabled and tool_name in disabled:
            disabled.remove(tool_name)
        tools_cfg["disabled_builtin"] = disabled

    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {toggle_type}")

    with open(config_path, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    _tools_cache.clear()

    return JSONResponse({"success": True})
