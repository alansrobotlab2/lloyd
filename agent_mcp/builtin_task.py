#!/usr/bin/env python3
"""Lloyd MCP Server: Task built-in — in-process subagent.

Runs a nested `app.harness.run_query` call inside the current process.
The subagent gets the same lloyd-mcp tool pool minus `Task` itself
(recursion cap enforced via contextvars; currently 1 level deep).

Subagent profiles are defined under `subagents:` in config.yaml. If the
requested profile is absent, a minimal default is used (all tools, 20
turns). The `general-purpose` profile is the default.

Mounted into Server("lloyd") via agent_mcp/main.py MODULES.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import uuid
from typing import Any

import yaml
from mcp.server import Server
from mcp.types import TextContent, Tool

logger = logging.getLogger("lloyd-builtin-task")

app = Server("lloyd-builtin-task")

# Recursion guard — depth of nested Task calls on the current call stack.
_task_depth: contextvars.ContextVar[int] = contextvars.ContextVar("_task_depth", default=0)
MAX_TASK_DEPTH = 1

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def _load_subagent_profile(subagent_type: str) -> dict[str, Any]:
    try:
        with open(_CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    profiles = cfg.get("subagents") or {}
    profile = profiles.get(subagent_type) or {}
    return {
        "system_prompt": profile.get("system_prompt", ""),
        "max_turns": int(profile.get("max_turns", 20)),
        "disallowed_tools": list(profile.get("disallowed_tools") or []),
        "model": profile.get("model", ""),
        "base_url": profile.get("base_url", ""),
    }


async def _task(args: dict[str, Any]) -> str:
    description = args.get("description", "")
    prompt = args.get("prompt", "")
    subagent_type = args.get("subagent_type", "general-purpose")

    if not prompt:
        return json.dumps({"error": "prompt is required"})

    depth = _task_depth.get()
    if depth >= MAX_TASK_DEPTH:
        return json.dumps({
            "error": f"Task recursion limit ({MAX_TASK_DEPTH}) reached — nested Task calls are not allowed."
        })

    profile = _load_subagent_profile(subagent_type)

    # Import here to avoid circular import at module load time.
    from app.harness.loop import run_query
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_URL
    from app.harness.options import RunOptions

    # Resolve model and base_url — fall back to primary defaults.
    model = profile["model"] or "primary"
    base_url = profile["base_url"] or "http://127.0.0.1:8096"

    # Subagent always disallows Task to prevent infinite recursion.
    disallowed = list(profile["disallowed_tools"])
    if "Task" not in disallowed:
        disallowed.append("Task")

    options = RunOptions(
        model=model,
        base_url=base_url,
        system_prompt=profile["system_prompt"],
        max_turns=profile["max_turns"],
        disallowed_tools=disallowed,
        mcp_servers={"lloyd-mcp": {"type": "sse", "url": DEFAULT_LLOYD_MCP_URL}},
        # Per-invocation session id so each subagent run gets its own
        # tool_search LoadedToolSet — different disallowed_tools profiles
        # would otherwise share one cache entry.
        session_id=f"task:{subagent_type}:{uuid.uuid4().hex[:8]}",
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    final_text = ""
    tool_calls_summary: list[str] = []

    token = _task_depth.set(depth + 1)
    try:
        async for evt in run_query(messages, options):
            if evt["type"] == "text_delta":
                final_text += evt["text"]
            elif evt["type"] == "tool_call":
                tool_calls_summary.append(evt["name"])
            elif evt["type"] == "result":
                stop_reason = evt.get("stop_reason", "stop")
                if stop_reason not in ("stop", "end_turn"):
                    final_text += f"\n[stopped: {stop_reason}]"
    except Exception as exc:
        logger.exception("Task subagent error for prompt=%r", prompt[:120])
        return json.dumps({"error": f"Subagent failed: {exc}"})
    finally:
        _task_depth.reset(token)

    result: dict[str, Any] = {"response": final_text}
    if tool_calls_summary:
        result["tools_used"] = tool_calls_summary
    if description:
        result["description"] = description
    return json.dumps(result)


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="Task",
            description=(
                "Spawn a subagent to complete a task. The subagent runs in the "
                "same process with the full lloyd-mcp tool pool (minus Task). "
                "Returns the subagent's final response and a list of tools used. "
                "Nested Task calls are not allowed (recursion cap = 1)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short label for the task (informational)",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt for the subagent",
                    },
                    "subagent_type": {
                        "type": "string",
                        "description": "Profile name from config.yaml subagents section (default: general-purpose)",
                        "default": "general-purpose",
                    },
                },
                "required": ["prompt"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "Task":
        text = await _task(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return [TextContent(type="text", text=text)]
