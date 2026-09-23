#!/usr/bin/env python3
"""Lloyd MCP Server: TodoWrite built-in.

Recreates Claude Code's v1 TodoWrite tool. The model maintains a
per-session checklist of tasks (pending / in_progress / completed) by
calling TodoWrite with the entire updated list each time. Storage is
the session JSON: a flat ``todos`` array alongside ``messages``.

Auto-clear: if every item in the submitted list is ``completed``, the
persisted list is reset to empty — mirrors Claude Code TodoWriteTool.ts:70.
The frontend reads ``GET /api/sessions/{id}/todos`` after every TodoWrite
tool result and re-renders.

Session correlation: the harness aggregator binds ``_session_id`` from
the call's args into the session contextvar (read via ``_shared.get_bound_session()``) before
dispatch (see ``agent_mcp/main.py:99-106``); we read it from there so
the per-tool schema doesn't need to advertise an internal field.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.types import Tool

from agent_mcp._shared import get_bound_session, text_result
from agent_mcp._todo_validation import VALID_STATUSES, validate_todos
from app.sessions_io import mutate_session

logger = logging.getLogger("lloyd-builtin-todo")


# Backwards-compatible aliases — code outside this module imported these
# names before B.4 extracted them; keep both pointing at the canonical
# implementation in `_todo_validation`.
_VALID_STATUSES = VALID_STATUSES
_validate_todos = validate_todos


async def _todo_write(args: dict[str, Any]) -> str:
    todos, err = validate_todos(args.get("todos"))
    if err is not None:
        return json.dumps({"error": err})

    session_id = get_bound_session()
    if not session_id:
        return json.dumps({
            "error": "TodoWrite called outside a session context — _session_id not bound",
        })

    all_done = bool(todos) and all(t["status"] == "completed" for t in todos)
    new_todos = [] if all_done else todos

    def _apply(data: dict[str, Any]) -> None:
        data["todos"] = new_todos

    ok = await mutate_session(session_id, _apply)
    if not ok:
        return json.dumps({"error": f"Session {session_id} not found"})

    # Echo the stored list back. The bare prose confirmation left the
    # list itself with exactly one appearance in the model's context —
    # the call it just made — after which it decayed with distance. On
    # 20260905_151355_iv5174 that was iteration 6 of 52, and the list was
    # never touched again. Echoing costs a few hundred tokens and puts
    # the current state in context at the moment it changes.
    if not new_todos:
        return (
            "Todos have been modified successfully. All items were completed, "
            "so the list has been cleared."
        )
    rendered = "\n".join(
        f"  - [{t['status']}] {t['content']}" for t in new_todos
    )
    open_count = sum(1 for t in new_todos if t["status"] != "completed")
    return (
        "Todos have been modified successfully. Current list "
        f"({open_count} of {len(new_todos)} still open):\n"
        f"{rendered}\n"
        "Keep this list current — call TodoWrite again with the full "
        "updated list as soon as an item's status changes."
    )


_TOOL_DESCRIPTION = """Use for a task of three or more steps, or when the user hands you a list of things to do; skip it for a single trivial task or a conversational answer. It keeps a task list for this session that the user can watch.

Each call replaces the whole list: send every item, not a delta. Each item carries `content` (imperative, "Run tests"), `activeForm` (present continuous, "Running tests") and a `status` of pending, in_progress or completed.

- While work is ongoing, keep exactly one item in_progress, and mark it before you start it.
- Mark an item completed as soon as it is fully done, and not before: failing tests, a partial implementation or an unresolved error keep it in_progress. When blocked, add an item for what has to be resolved.
- Remove items that no longer apply, and add follow-ups as you find them.
- A list whose items are all completed is cleared on the next call."""


async def list_tools():
    return [
        Tool(
            name="TodoWrite",
            description=_TOOL_DESCRIPTION,
            inputSchema={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The updated todo list (replaces the previous list).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "Imperative form, e.g. 'Run tests'.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": list(_VALID_STATUSES),
                                },
                                "activeForm": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": (
                                        "Present continuous form shown during "
                                        "execution, e.g. 'Running tests'."
                                    ),
                                },
                            },
                            "required": ["content", "status", "activeForm"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["todos"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "TodoWrite":
        text = await _todo_write(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return text_result(text)
