"""Lloyd MCP Server: Mission Control UI — observe + control the MC frontend.

Two tools:
  mc_get_state  — read the user's current MC tab and focused work item.
  mc_navigate   — switch the user's tab (and optionally set focus inside it),
                  returning a brief tab-specific summary.

Why these tools live here: the frontend mirrors its current tab + focus
into the FastAPI backend (POST /api/mc/state) and subscribes to a
backend SSE bus. These tools just call those HTTP endpoints. This is
the first MCP module that calls back into the local backend; we use a
loopback URL to keep the boundary explicit.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent

from agent_mcp._shared import _err, _wrap, ErrorCode

LLOYD_API = os.environ.get("LLOYD_API_URL", "http://127.0.0.1:8080")

_VALID_TABS = [
    "inner_voice", "chat", "backlog", "autonomy", "workers",
    "memory", "architecture", "skills", "tools", "services",
    "settings", "graph", "ide",
]

app = Server("lloyd-mission-control-ui")


async def _mc_get_state(_params: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{LLOYD_API}/api/mc/state")
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return _err(f"failed to read MC state: {e}", ErrorCode.INTERNAL)

    return {
        "tab": data.get("tab"),
        "focus": data.get("focus"),
        "focus_by_tab": data.get("focus_by_tab", {}),
        "last_updated": data.get("last_updated"),
    }


async def _mc_navigate(params: dict) -> dict:
    tab = (params.get("tab") or "").strip()
    if not tab:
        return _err("tab is required", ErrorCode.MISSING_PARAM)
    if tab not in _VALID_TABS:
        return _err(
            f"unknown tab: {tab!r}. Valid: {_VALID_TABS}",
            ErrorCode.INVALID_PARAM,
        )

    focus_id = params.get("focus_id")
    if focus_id is not None and not isinstance(focus_id, str):
        return _err("focus_id must be a string when provided", ErrorCode.INVALID_PARAM)

    body: dict[str, Any] = {"tab": tab}
    if focus_id:
        body["focus_id"] = focus_id

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{LLOYD_API}/api/mc/navigate", json=body)
            if r.status_code >= 400:
                # Surface the FastAPI `detail` message so the caller sees the
                # actionable hint (e.g. "use vault_search") rather than a bare
                # HTTP status string.
                try:
                    msg = r.json().get("detail") or r.text
                except Exception:
                    msg = r.text
                code = ErrorCode.NOT_FOUND if r.status_code == 404 else ErrorCode.INVALID_PARAM
                return _err(str(msg), code)
            data = r.json()
    except Exception as e:
        return _err(f"navigate failed: {e}", ErrorCode.INTERNAL)

    return {
        "tab": data.get("tab", tab),
        "focus_id": data.get("focus_id", focus_id),
        "focus_error": data.get("focus_error"),
        "detail": data.get("detail", {}),
    }


async def _mc_close_modal(params: dict) -> dict:
    tab = (params.get("tab") or "").strip()
    if not tab:
        return _err("tab is required", ErrorCode.MISSING_PARAM)
    if tab not in _VALID_TABS:
        return _err(
            f"unknown tab: {tab!r}. Valid: {_VALID_TABS}",
            ErrorCode.INVALID_PARAM,
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{LLOYD_API}/api/mc/close_modal", json={"tab": tab}
            )
            if r.status_code >= 400:
                try:
                    msg = r.json().get("detail") or r.text
                except Exception:
                    msg = r.text
                return _err(str(msg), ErrorCode.INVALID_PARAM)
    except Exception as e:
        return _err(f"close_modal failed: {e}", ErrorCode.INTERNAL)

    return {"tab": tab, "closed": True}


_handlers = {
    "mc_get_state": _mc_get_state,
    "mc_navigate": _mc_navigate,
    "mc_close_modal": _mc_close_modal,
}


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="mc_get_state",
            description=(
                "Report which Mission Control tab the user is currently viewing "
                "and the work item (if any) they have focused inside it. "
                "Returns {tab, focus, focus_by_tab, last_updated}. focus is "
                "{kind, id, label?} when set, null otherwise."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="mc_navigate",
            description=(
                "Switch the user's Mission Control tab and optionally focus a "
                "specific item within it. Pushes the change to the user's open "
                "browser tabs and returns a brief summary of what's now visible.\n\n"
                "USE THIS WHENEVER THE USER REFERENCES A TAB IN THEIR REQUEST. "
                "Phrases like \"pull up X in the memory tab\", \"show me X on "
                "the backlog\", \"open task 42\", \"switch to tools\", \"let's "
                "look at autonomy\", \"surface X for me\" are all UI-navigation "
                "requests, not content retrieval. The user wants you to move "
                "their view, not just fetch the data into chat. Reach for this "
                "tool BEFORE vault_read / backlog_get / etc. when the request "
                "names a tab.\n\n"
                "focus_id semantics per tab:\n"
                "  inner_voice / chat → session id (UUID)\n"
                "  backlog            → numeric task id (opens task) or board "
                "id (switches board)\n"
                "  autonomy           → numeric task id (opens edit modal)\n"
                "  workers            → source name (filters queue)\n"
                "  memory             → entity name, OR vault path like "
                "\"agents/lloyd/SOUL.md\" — paths open the file in the explorer\n"
                "  tools              → MCP server id (expands server)\n"
                "  skills             → skill name (selects + opens viewer)\n"
                "  services           → service unit name (expands)\n"
                "  ide                → absolute file path (opens it in a "
                "new editor tab; prefer ide_open_file for richer feedback)\n"
                "  architecture / settings / graph → no focus supported\n\n"
                "If focus_id is invalid (path doesn't exist, escapes vault), the "
                "tab still switches and the result returns `focus_error` describing "
                "the failure with `focus_id: null`. Inspect `focus_error` and retry "
                "focus separately if needed (e.g. via vault_search to find the "
                "right path)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tab": {
                        "type": "string",
                        "enum": _VALID_TABS,
                        "description": "Tab to switch to.",
                    },
                    "focus_id": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional opaque id of an item to focus within the "
                            "tab (e.g. session id for chat/inner_voice, task id "
                            "for backlog/autonomy, server name for tools, skill "
                            "name for skills, vault path or entity name for "
                            "memory). When omitted, just switches the tab."
                        ),
                    },
                },
                "required": ["tab"],
            },
        ),
        Tool(
            name="mc_close_modal",
            description=(
                "Dismiss any modal popup currently open in the given Mission "
                "Control tab. Counterpart to mc_navigate when its focus_id "
                "opens a modal (memory document viewer, autonomy/backlog task "
                "editor, create-task dialog). Tabs without modals (workers, "
                "settings, graph, etc.) silently no-op.\n\n"
                "Use when the user says \"close that\", \"dismiss it\", "
                "\"close the popup\", \"close the document\", or after you've "
                "shown them an item via mc_navigate and they're done with it. "
                "For closing IDE editor tabs use ide_close_tab instead."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tab": {
                        "type": "string",
                        "enum": _VALID_TABS,
                        "description": (
                            "Tab whose open modal should be dismissed. Use "
                            "mc_get_state first if unsure which tab the user "
                            "is on."
                        ),
                    },
                },
                "required": ["tab"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handler = _handlers.get(name)
    if not handler:
        return _wrap(_err(f"Unknown tool: {name}", ErrorCode.UNKNOWN_TOOL))
    result = await handler(arguments or {})
    return _wrap(result)
