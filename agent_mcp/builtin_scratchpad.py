"""`Scratchpad` — one session's own mid-turn notes (#1554).

Thin surface over `app.scratchpad`, which owns the storage rules. There is
deliberately no session argument in the schema: the session comes from the
aggregator's bound context (`get_bound_session`, which `main.py` sets per call from
the harness's own `_meta`), so a model cannot name — and so cannot read or write —
another session's file, even by mistake. That is the whole cross-run leak story
closed at the schema rather than with a check.

Append is the only write, and it is additive: the file is opened `ab` and no path
in `app.scratchpad` truncates or replaces it, so two notes in a file are two notes
in order and an earlier note cannot be lost to a later one.
"""
from __future__ import annotations

import json
from typing import Any

from mcp.types import Tool

from agent_mcp._shared import get_bound_session, text_result
from app import scratchpad


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["append", "read"],
                "description": (
                    "append adds one note to the end of this session's scratchpad; "
                    "read returns the whole file."),
            },
            "content": {
                "type": "string",
                "description": (
                    "The note to append — prose: what you tried, what you ruled "
                    "out, what you do next. Ignored by read."),
            },
        },
        "required": ["action"],
    }


async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="Scratchpad",
            description=(
                "Append your own notes for this session; read them before reopening "
                "a question you already settled. Additive — a write never replaces a "
                "note. Not the todo list, not memory: scoped to this session."),
            inputSchema=_schema(),
        ),
    ]


async def call_tool(name: str, arguments: dict) -> Any:
    if name != "Scratchpad":
        return text_result(json.dumps({"error": f"Unknown tool: {name}"}))

    session_id = get_bound_session()
    if not session_id:
        # No bound session means nothing addressable. Refusing beats inventing a
        # bucket name: an invented one would be shared by every anonymous caller,
        # which is exactly the cross-session leak the design exists to prevent.
        return text_result(json.dumps(
            {"error": "Scratchpad needs a session context, and this call has "
                      "none. Nothing was written or read."}))

    action = (arguments or {}).get("action")

    if action == "read":
        try:
            body = scratchpad.read(session_id)
        except scratchpad.ScratchpadError as exc:
            return text_result(json.dumps({"error": str(exc)}))
        if not body:
            return text_result(json.dumps(
                {"session_id": session_id, "writes": 0, "bytes": 0,
                 "content": "", "note": "nothing appended yet this session"}))
        return text_result(json.dumps(
            {"session_id": session_id, **scratchpad.stats(session_id),
             "content": body}, ensure_ascii=False))

    if action == "append":
        content = (arguments or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            return text_result(json.dumps(
                {"error": "action=append needs non-empty `content`."}))
        try:
            return text_result(json.dumps(
                scratchpad.append(session_id, content), ensure_ascii=False))
        except scratchpad.ScratchpadError as exc:
            return text_result(json.dumps({"error": str(exc)}))

    return text_result(json.dumps(
        {"error": f"action must be 'append' or 'read', got {action!r}."}))
