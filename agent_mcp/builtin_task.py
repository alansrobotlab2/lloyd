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
import uuid
from typing import Any

from app.config import default_model_base_url
from mcp.types import Tool

from agent_mcp._shared import text_result

logger = logging.getLogger("lloyd-builtin-task")

# Recursion guard — depth of nested Task calls on the current call stack.
_task_depth: contextvars.ContextVar[int] = contextvars.ContextVar("_task_depth", default=0)
MAX_TASK_DEPTH = 1

def _load_subagent_profile(subagent_type: str) -> dict[str, Any]:
    """Read one `subagents:` profile from the live config.

    Goes through `app.config.CONFIG` rather than opening config.yaml
    directly. The direct read bypassed both `${VAR}` expansion and the
    merge of `data/tool_overrides.yaml`, so a subagent saw a different
    configuration than every other caller.
    """
    from app.config import CONFIG

    profiles = CONFIG.get("subagents") or {}
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
    from app.harness.hooks import HookRegistry
    from app.harness.loop import run_query
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.harness.options import RunOptions
    from app.harness.safety import install_default_safety_hook

    # Resolve model and base_url — fall back to primary defaults.
    model = profile["model"] or "primary"
    base_url = profile["base_url"] or default_model_base_url()

    # Config-level tool disables apply to subagents too. They did not
    # before: `disallowed` came from the profile alone, so any tool
    # switched off in the Tools page or listed in
    # `mcp_servers.<server>.disabled_tools` stayed fully callable inside
    # a Task — the one execution context with no human watching the
    # stream. Same reasoning that puts the safety hook here.
    from app.mcp_discovery import _get_disallowed_tools

    disallowed = list(_get_disallowed_tools())
    for name in profile["disallowed_tools"]:
        if name not in disallowed:
            disallowed.append(name)
    # Subagent always disallows Task to prevent infinite recursion.
    if "Task" not in disallowed:
        disallowed.append("Task")

    # Subagents ran with `hooks=None`, which meant the harness's default
    # destructive-Bash gate never installed inside a Task — the one place
    # with no human watching the stream. `safety.py` is documented as
    # running on every primary turn; a subagent is a harness run like any
    # other and gets the same floor.
    #
    # The Inner Voice observer is deliberately NOT attached here: it is
    # scoped to a session turn (goal card, session todos, ambient and
    # clarify channels) and a subagent has none of those.
    task_hooks = HookRegistry()
    install_default_safety_hook(task_hooks)

    options = RunOptions(
        model=model,
        base_url=base_url,
        system_prompt=profile["system_prompt"],
        max_turns=profile["max_turns"],
        disallowed_tools=disallowed,
        hooks=task_hooks,
        mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
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


async def call_tool(name: str, arguments: dict):
    if name == "Task":
        text = await _task(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return text_result(text)
