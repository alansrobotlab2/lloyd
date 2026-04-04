#!/usr/bin/env python3
"""
Lloyd MCP Server: Thunderbird — email and calendar tools.

Spawns the Thunderbird MCP bridge (mcp-bridge.cjs) as a subprocess,
discovers tools via tools/list, and re-exports them as Lloyd MCP tools.

Requires Thunderbird running with the MCP extension (localhost:8765).
"""

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

BRIDGE_PATH = Path.home() / "agent-services" / "services" / "thunderbird-mcp" / "mcp-bridge.cjs"

TOOL_NAME_MAP = {
    "listAccounts": "email_accounts",
    "listFolders": "email_folders",
    "searchMessages": "email_search",
    "getMessage": "email_read",
    "getRecentMessages": "email_recent",
    "updateMessage": "email_update",
    "deleteMessages": "email_delete",
    "createFolder": "email_create_folder",
    "sendMail": "email_send",
    "replyToMessage": "email_reply",
    "forwardMessage": "email_forward",
    "listFilters": "email_list_filters",
    "createFilter": "email_create_filter",
    "updateFilter": "email_update_filter",
    "deleteFilter": "email_delete_filter",
    "reorderFilters": "email_reorder_filters",
    "applyFilters": "email_apply_filters",
    "searchContacts": "contacts_search",
    "getContact": "contacts_get",
    "listCalendars": "calendar_list",
    "createEvent": "calendar_create",
    "getEvents": "calendar_events",
}

# Reverse map: hermes_name -> mcp_name
REVERSE_MAP = {v: k for k, v in TOOL_NAME_MAP.items()}

logger = logging.getLogger(__name__)

app = Server("lloyd-thunderbird")

# Bridge process state
_bridge_proc: Optional[subprocess.Popen] = None
_discovered_tools: list[dict] = []


def _ensure_bridge() -> subprocess.Popen:
    global _bridge_proc
    if _bridge_proc and _bridge_proc.poll() is None:
        return _bridge_proc

    if not BRIDGE_PATH.exists():
        raise FileNotFoundError(f"MCP bridge not found at {BRIDGE_PATH}")

    _bridge_proc = subprocess.Popen(
        ["node", str(BRIDGE_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Initialize
    init_msg = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "lloyd-thunderbird", "version": "1.0.0"},
        },
    }
    _bridge_send(init_msg)
    # Consume the init response so it doesn't pollute subsequent _bridge_receive() calls
    try:
        _bridge_receive(timeout=5.0)
    except Exception:
        pass
    return _bridge_proc


def _bridge_send(msg: dict) -> None:
    if _bridge_proc and _bridge_proc.stdin:
        _bridge_proc.stdin.write(json.dumps(msg) + "\n")
        _bridge_proc.stdin.flush()


def _bridge_receive(timeout: float = 30.0) -> dict:
    import select
    if not _bridge_proc or not _bridge_proc.stdout:
        raise RuntimeError("Bridge not running")
    start = time.time()
    while time.time() - start < timeout:
        if _bridge_proc.poll() is not None:
            raise RuntimeError("Bridge subprocess exited")
        if select.select([_bridge_proc.stdout], [], [], 0.5)[0]:
            line = _bridge_proc.stdout.readline().strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise TimeoutError("Timeout waiting for bridge response")


def _call_bridge_tool(mcp_name: str, args: dict, timeout: float = 30.0) -> str:
    _ensure_bridge()
    request_id = int(time.time() * 1000) % 1000000
    request = {
        "jsonrpc": "2.0", "id": request_id,
        "method": "tools/call",
        "params": {"name": mcp_name, "arguments": args},
    }
    _bridge_send(request)

    import select
    start = time.time()
    while time.time() - start < timeout:
        if _bridge_proc.poll() is not None:
            raise RuntimeError("Bridge subprocess exited")
        if select.select([_bridge_proc.stdout], [], [], 0.5)[0]:
            line = _bridge_proc.stdout.readline().strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("id") == request_id:
                    if "error" in msg:
                        raise RuntimeError(f"Bridge error: {msg['error'].get('message', 'unknown')}")
                    content = msg.get("result", {}).get("content", [])
                    if content:
                        return "".join(c.get("text", "") for c in content)
                    return "(no result)"
            except json.JSONDecodeError:
                continue
    raise TimeoutError(f"Timeout calling tool {mcp_name}")


def _discover_tools() -> list[dict]:
    global _discovered_tools
    try:
        _ensure_bridge()
        request_id = int(time.time() * 1000) % 1000000
        _bridge_send({"jsonrpc": "2.0", "id": request_id, "method": "tools/list", "params": {}})
        response = _bridge_receive(timeout=10.0)
        if "error" in response:
            return []
        _discovered_tools = response.get("result", {}).get("tools", [])
        return _discovered_tools
    except Exception as e:
        logger.warning(f"Failed to discover Thunderbird tools: {e}")
        return []


@app.list_tools()
async def list_tools():
    tools = _discover_tools()
    result = []
    for tool in tools:
        mcp_name = tool["name"]
        lloyd_name = TOOL_NAME_MAP.get(mcp_name, f"tb_{mcp_name}")
        description = tool.get("description", f"Thunderbird: {mcp_name}")
        schema = tool.get("inputSchema", {"type": "object", "properties": {}})
        result.append(Tool(name=lloyd_name, description=description, inputSchema=schema))
    return result


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    # Resolve lloyd name -> MCP name
    mcp_name = REVERSE_MAP.get(name)
    if not mcp_name:
        # Try stripping tb_ prefix
        if name.startswith("tb_"):
            mcp_name = name[3:]
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    try:
        result = _call_bridge_tool(mcp_name, arguments or {})
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
