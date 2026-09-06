#!/usr/bin/env python3
"""
Lloyd MCP Server: Mission Control — session management and messaging.

In Lloyd, this server manages SDK-based sessions rather than Hermes CLI sessions.
Provides tools for listing sessions and reading session metadata.

Tools: chat_list_sessions, chat_get_session
"""

import json
from pathlib import Path

from mcp.types import Tool

from agent_mcp._shared import text_result

from app.paths import LLOYD_HOME  # the running code dir, not $HOME/lloyd
SESSIONS_DIR = LLOYD_HOME / "sessions"


def _list_sessions() -> str:
    """List all Lloyd sessions from session metadata files."""
    if not SESSIONS_DIR.exists():
        return json.dumps({"success": True, "sessions": [], "count": 0})

    sessions = []
    for sf in sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id", sf.stem),
                "model": data.get("model", ""),
                "created_at": data.get("created_at", ""),
                "last_active": data.get("last_active", ""),
                "preview": data.get("preview", ""),
                "message_count": data.get("message_count", 0),
            })
        except Exception:
            continue

    return json.dumps({"success": True, "sessions": sessions[:50], "count": len(sessions)})


def _get_session(session_id: str) -> str:
    """Get session metadata by ID."""
    if not session_id:
        return json.dumps({"success": False, "error": "session_id is required"})

    session_file = SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        return json.dumps({"success": False, "error": f"Session {session_id} not found"})

    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        return json.dumps({"success": True, "session": data})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


async def list_tools():
    return [
        Tool(name="chat_list_sessions", description="List the Lloyd chat sessions with their ids, titles and last-activity times. Use chat_get_session to read one, or session_recall to search their content.", inputSchema={
            "type": "object", "properties": {},
        }),
        Tool(name="chat_get_session", description="Get one chat session's metadata: title, model, message count and timestamps. Returns metadata only — use session_recall to search what was actually said.", inputSchema={
            "type": "object",
            "properties": {"session_id": {"type": "string", "description": "Session ID to retrieve"}},
            "required": ["session_id"],
        }),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "chat_list_sessions":
        return text_result(_list_sessions())
    elif name == "chat_get_session":
        return text_result(_get_session(arguments.get("session_id", "")))
    return text_result(json.dumps({"error": f"Unknown tool: {name}"}))

