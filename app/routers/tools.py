"""Tools tab endpoints — MCP server inspection and tool enable/disable.

Built-in tools (Bash/Read/Write/Edit/Grep/Glob/Task) live inside the
lloyd-mcp aggregator now, so this router has only two real concepts:
servers and the tools each server exposes. Disabling a built-in is the
same operation as disabling any other tool: write its bare name into
the parent server's `disabled_tools` list.
"""

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import CONFIG, save_tool_overrides
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
    """Toggle a server, individual tool, or baseline membership, persisting
    changes to data/tool_overrides.yaml (config.yaml is read-only at boot).

    Payloads:
      {type: "server",   server, enabled}
      {type: "tool",     server, tool, enabled}
      {type: "baseline", tool, enabled}   # progressive-discovery baseline
    """
    data = await request.json()
    toggle_type = data.get("type")
    enabled = bool(data.get("enabled", True))

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

    elif toggle_type == "baseline":
        tool_name = (data.get("tool") or "").strip()
        if not tool_name:
            raise HTTPException(status_code=400, detail="tool is required for baseline toggle")
        ts_cfg = CONFIG.setdefault("harness", {}).setdefault("tool_search", {})
        baseline = list(ts_cfg.get("baseline_tools") or [])
        if enabled and tool_name not in baseline:
            baseline.append(tool_name)
        elif not enabled and tool_name in baseline:
            baseline.remove(tool_name)
        ts_cfg["baseline_tools"] = baseline

    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {toggle_type}")

    save_tool_overrides()

    _tools_cache.clear()

    return JSONResponse({"servers_updated": True})


# ── Tool discovery (progressive disclosure) settings ────────────────────

@router.get("/api/tool-discovery")
async def get_tool_discovery():
    """Return current `harness.tool_search.*` config plus a few derived
    counts so the UI can render a "9 baseline / 30 threshold" summary
    without re-parsing the catalog.

    Shape:
      {
        "enabled": bool,
        "threshold_tools": int,
        "baseline_tools": list[str],
        "max_results_default": int,
        "max_results_cap": int,
        "total_tools": int,         # discovered count across all enabled servers
        "active": bool,             # progressive disclosure currently kicking in
      }
    """
    ts = (CONFIG.get("harness") or {}).get("tool_search") or {}
    enabled = bool(ts.get("enabled", True))
    threshold = int(ts.get("threshold_tools", 30))
    baseline = list(ts.get("baseline_tools") or [])
    max_default = int(ts.get("max_results_default", 5))
    max_cap = int(ts.get("max_results_cap", 20))

    # Derive total_tools from the cached discovery (no extra network round
    # trips). Cache may be empty before the first /api/tools call; that's
    # fine — UI just shows 0 until the user visits Tools once.
    total = sum(
        len(entry.get("tools") or [])
        for entry in _tools_cache.values()
    )
    active = enabled and total >= threshold

    return JSONResponse({
        "enabled": enabled,
        "threshold_tools": threshold,
        "baseline_tools": baseline,
        "max_results_default": max_default,
        "max_results_cap": max_cap,
        "total_tools": total,
        "active": active,
    })


@router.post("/api/tool-discovery")
async def set_tool_discovery(request: Request):
    """Update the `harness.tool_search.*` block. Body keys are optional —
    only provided keys are written; everything else stays at its current
    value (or the dataclass default in app/harness/options.py).

    Body:
      {
        "enabled"?: bool,
        "threshold_tools"?: int,        # clamped to [0, 1000]
        "max_results_default"?: int,    # clamped to [1, 50]
        "max_results_cap"?: int,        # clamped to [1, 100]
        "baseline_tools"?: list[str],   # full replace
      }
    """
    data = await request.json() if (await request.body()) else {}
    ts = CONFIG.setdefault("harness", {}).setdefault("tool_search", {})

    if "enabled" in data:
        ts["enabled"] = bool(data["enabled"])
    if "threshold_tools" in data:
        try:
            n = int(data["threshold_tools"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="threshold_tools must be int")
        ts["threshold_tools"] = max(0, min(1000, n))
    if "max_results_default" in data:
        try:
            n = int(data["max_results_default"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="max_results_default must be int")
        ts["max_results_default"] = max(1, min(50, n))
    if "max_results_cap" in data:
        try:
            n = int(data["max_results_cap"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="max_results_cap must be int")
        ts["max_results_cap"] = max(1, min(100, n))
    if "baseline_tools" in data:
        bt = data["baseline_tools"]
        if not isinstance(bt, list) or not all(isinstance(x, str) for x in bt):
            raise HTTPException(status_code=400, detail="baseline_tools must be list[str]")
        # de-dup while preserving order
        seen: set[str] = set()
        ts["baseline_tools"] = [
            x for x in (s.strip() for s in bt)
            if x and not (x in seen or seen.add(x))
        ]

    save_tool_overrides()

    return JSONResponse({"updated": True, "tool_search": ts})
