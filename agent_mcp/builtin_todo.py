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

from mcp.server import Server
from mcp.types import TextContent, Tool

from agent_mcp._shared import get_bound_session
from agent_mcp._todo_validation import VALID_STATUSES, validate_todos
from app.sessions_io import mutate_session

logger = logging.getLogger("lloyd-builtin-todo")

app = Server("lloyd-builtin-todo")


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

    return (
        "Todos have been modified successfully. Ensure that you continue "
        "to use the todo list to track your progress. Please proceed with "
        "the current tasks if applicable"
    )


_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user. It also helps the user understand the progress of the task and overall progress of their requests.

The list is replaced wholesale on every call — submit the entire updated list each time, not a delta.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Examples of When to Use the Todo List

<example>
User: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!
Assistant: *Creates todo list with the following items:*
1. Creating dark mode toggle component in Settings page
2. Adding dark mode state management (context/store)
3. Implementing CSS-in-JS styles for dark theme
4. Updating existing components to support theme switching
5. Running tests and build process, addressing any failures or errors that occur
*Begins working on the first task*
</example>

<example>
User: Help me rename the function getCwd to getCurrentWorkingDirectory across my project
Assistant: *Uses grep or search tools to locate all instances of getCwd in the codebase*
I've found 15 instances of 'getCwd' across 8 different files.
*Creates todo list with specific items for each file that needs updating*
</example>

## Examples of When NOT to Use the Todo List

<example>
User: How do I print 'Hello World' in Python?
Assistant: In Python, you can print "Hello World" with print("Hello World").
</example>

<example>
User: What does the git status command do?
Assistant: The git status command shows the current state of your working directory and staging area.
</example>

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully

   **IMPORTANT**: Task descriptions must have two forms:
   - content: The imperative form describing what needs to be done (e.g., "Run tests", "Build the project")
   - activeForm: The present continuous form shown during execution (e.g., "Running tests", "Building the project")

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Exactly ONE task must be in_progress at any time (not less, not more)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - Tests are failing
     - Implementation is partial
     - You encountered unresolved errors
     - You couldn't find necessary files or dependencies

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names
   - Always provide both forms:
     - content: "Fix authentication bug"
     - activeForm: "Fixing authentication bug"

When the entire list is marked completed, it is automatically cleared on the next call. Make sure that at least one task is in_progress at all times while work is ongoing. Always provide both content (imperative) and activeForm (present continuous) for each task."""


@app.list_tools()
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


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "TodoWrite":
        text = await _todo_write(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return [TextContent(type="text", text=text)]
