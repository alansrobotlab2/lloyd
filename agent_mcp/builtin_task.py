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

from agent_mcp import _subagent_registry
from agent_mcp._shared import get_bound_session, text_result

logger = logging.getLogger("lloyd-builtin-task")

# Recursion guard — depth of nested Task calls on the current call stack.
_task_depth: contextvars.ContextVar[int] = contextvars.ContextVar("_task_depth", default=0)
MAX_TASK_DEPTH = 1

# The calling turn's model and endpoint, bound by `agent_mcp.main.call_tool`
# from the request's `_meta` (see META_MODEL / META_BASE_URL there). This is
# the only channel that exists: Task runs inside the aggregator process, so
# without it a subagent has no idea which model spawned it and every profile
# fell back to `primary`. A turn on the secondary 35B delegated its
# subagents to the primary, which made the 35B untestable for exactly the
# fan-out work Task exists for.
current_parent_model: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_parent_model", default=""
)
current_parent_base_url: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_parent_base_url", default=""
)

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


def _base_url_for(model: str) -> str:
    """Endpoint that serves `model`, preferring the caller's own.

    The parent's base_url is authoritative when the model matches: it is
    the endpoint the turn is actually streaming from, which survives a
    caller that overrode the config default. Otherwise resolve the alias
    through `models:` in config, and only then fall back to the default
    model's endpoint.
    """
    if model and model == current_parent_model.get(""):
        parent_url = current_parent_base_url.get("")
        if parent_url:
            return parent_url
    from app.config import _get_model_cfg

    cfg = _get_model_cfg(model) or {}
    url = cfg.get("base_url") or (cfg.get("env") or {}).get("ANTHROPIC_BASE_URL", "")
    return url or default_model_base_url()


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

    # Resolve model and base_url.
    #
    # Precedence: an explicit `subagents.<type>.model` pin wins; otherwise
    # the subagent inherits the calling turn's model, so delegating from a
    # turn on the secondary keeps the work on the secondary. Falls back to
    # `primary` only when there is no parent context at all (Task invoked
    # outside a harness turn).
    #
    # base_url is resolved FROM the chosen model rather than from
    # `default_model_base_url()`, which always returns the primary's
    # endpoint: a profile pinned to `model: secondary` with an empty
    # `base_url` used to send "secondary" to the primary's port, where the
    # engine does not serve that name.
    model = profile["model"] or current_parent_model.get("") or "primary"
    base_url = profile["base_url"] or _base_url_for(model)

    # Config-level tool disables apply to subagents too. They did not
    # before: `disallowed` came from the profile alone, so any tool
    # switched off in the Tools page or listed in
    # `mcp_servers.<server>.disabled_tools` stayed fully callable inside
    # a Task — the one execution context with no human watching the
    # stream. Same reasoning that puts the safety hook here.
    from app.config import CONFIG
    from app.mcp_discovery import _get_disallowed_tools

    disallowed = list(_get_disallowed_tools())
    for name in profile["disallowed_tools"]:
        if name not in disallowed:
            disallowed.append(name)
    # Subagent always disallows Task to prevent infinite recursion.
    if "Task" not in disallowed:
        disallowed.append("Task")

    # ...and never the self-modification loop. A subagent is spawned to do a
    # bounded piece of research or editing and reports back through a summary;
    # nothing about that shape suits opening a round, gating it, or landing
    # code on production. Landing from inside a Task would also restart the
    # aggregator the Task is running in. Same reasoning as the Task recursion
    # cap: the constraint belongs here, not in a prompt.
    for name in ("selfmod_start", "selfmod_gate", "selfmod_land",
                 "selfmod_abort", "selfmod_rollback"):
        if name not in disallowed:
            disallowed.append(name)
            disallowed.append(f"mcp__lloyd-mcp__{name}")

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

    # Per-invocation session id so each subagent run gets its own
    # tool_search LoadedToolSet — different disallowed_tools profiles
    # would otherwise share one cache entry. Bound here rather than
    # inline so the registry row and the harness agree on the id.
    sub_session_id = f"task:{subagent_type}:{uuid.uuid4().hex[:8]}"

    options = RunOptions(
        model=model,
        base_url=base_url,
        system_prompt=profile["system_prompt"],
        max_turns=profile["max_turns"],
        disallowed_tools=disallowed,
        hooks=task_hooks,
        mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
        session_id=sub_session_id,
        # A subagent is an agent loop like any other and has the same
        # redundant-reasoning problem the main loop does — more so, since
        # it runs its whole investigation inside one turn.
        preserve_thinking_iterations=int(
            (CONFIG.get("harness") or {}).get("preserve_thinking_iterations", 0)
        ),
        # A wedged engine inside a subagent is worse than one on the main
        # turn: the caller is blocked on `_task` with nothing streaming and
        # no human watching. Same deadline as the parent.
        stream_chunk_timeout_s=float(
            (CONFIG.get("harness") or {}).get("stream_chunk_timeout_seconds", 0)
        ),
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    # The subagent's answer is the text of its TERMINATING iteration —
    # the one that stopped without calling a tool. Accumulating every
    # `text_delta` instead (as this did until 2026-09-05) concatenates
    # each iteration's preamble, so a subagent that burns all its turns
    # mid-investigation returns its opening "I'll start by reading the
    # repo" as though that were the finished review. Session
    # 20260905_151355_iv5174 lost 231s to exactly that: 19 tool calls,
    # a preamble for an answer, and no signal to the caller that the
    # work never happened.
    final_text = ""            # terminal (tool-call-free) iteration's text
    last_iter_text = ""        # most recent iteration's text, terminal or not
    last_iter_thinking = ""    # ...and its reasoning, for diagnosing empty answers
    tool_calls_summary: list[str] = []
    stop_reason = "stop"
    num_turns = 0

    # Open the live row BEFORE the loop starts. A Task blocks its caller
    # for as long as it runs, so a row created on completion would only
    # ever describe runs that already finished — precisely the ones that
    # no longer need watching.
    record = _subagent_registry.register(
        subagent_type=subagent_type,
        description=description,
        prompt=prompt,
        parent_session_id=get_bound_session(),
        session_id=sub_session_id,
        model=model,
        max_turns=profile["max_turns"],
    )

    token = _task_depth.set(depth + 1)
    try:
        async for evt in run_query(messages, options):
            if evt["type"] == "assistant_message":
                last_iter_text = evt.get("text") or ""
                last_iter_thinking = evt.get("thinking") or ""
                record.note_turn()
                if not evt.get("tool_calls"):
                    final_text = last_iter_text
            elif evt["type"] == "tool_call":
                tool_calls_summary.append(evt["name"])
                record.note_tool(evt["name"])
            elif evt["type"] == "result":
                stop_reason = evt.get("stop_reason", "stop")
                num_turns = int(evt.get("num_turns", 0) or 0)
    except asyncio.CancelledError:
        # Closed here rather than in `finally`: a finally runs before the
        # success paths below, and `finish` is idempotent, so a blanket
        # close there would stamp every completed run "cancelled" and
        # make the real status a no-op.
        _subagent_registry.finish(record, status="cancelled", stop_reason="cancelled")
        raise
    except Exception as exc:
        logger.exception("Task subagent error for prompt=%r", prompt[:120])
        _subagent_registry.finish(record, status="error", error=str(exc))
        return json.dumps({"error": f"Subagent failed: {exc}"})
    finally:
        _task_depth.reset(token)

    # A subagent that dispatched tools but never produced a closing
    # message did NOT do the job. Return an error rather than a
    # plausible-looking empty string — the caller cannot otherwise tell
    # "reviewed it, found nothing" from "never got there".
    if not final_text.strip() and tool_calls_summary:
        reason = _diagnose_empty_answer(
            stop_reason=stop_reason,
            num_turns=num_turns,
            max_turns=profile["max_turns"],
            thinking=last_iter_thinking,
        )
        logger.warning(
            "Task subagent produced no final answer (%s): stop_reason=%s "
            "turns=%d/%d tools=%d",
            reason, stop_reason, num_turns, profile["max_turns"],
            len(tool_calls_summary),
        )
        err: dict[str, Any] = {
            "error": (
                f"Subagent ran {len(tool_calls_summary)} tool calls but returned "
                f"no final answer ({reason}). The task was NOT completed — do "
                f"not treat any text below as the result."
            ),
            "stop_reason": stop_reason,
            "turns_used": num_turns,
            "max_turns": profile["max_turns"],
            "tools_used": tool_calls_summary,
        }
        if last_iter_text.strip():
            err["partial_text"] = last_iter_text[:500]
        if description:
            err["description"] = description
        _subagent_registry.finish(
            record, status="failed", stop_reason=stop_reason, error=reason,
        )
        return json.dumps(err)

    result: dict[str, Any] = {"response": final_text}
    # Surface truncation even when there IS text — a max_turns run can
    # still end on a partial answer.
    if stop_reason not in ("stop", "end_turn"):
        result["stop_reason"] = stop_reason
        result["truncated"] = True
        result["turns_used"] = num_turns
        result["max_turns"] = profile["max_turns"]
    if tool_calls_summary:
        result["tools_used"] = tool_calls_summary
    if description:
        result["description"] = description
    _subagent_registry.finish(
        record,
        status="completed",
        stop_reason=stop_reason,
        response_chars=len(final_text),
    )
    return json.dumps(result)


def _diagnose_empty_answer(
    *, stop_reason: str, num_turns: int, max_turns: int, thinking: str,
) -> str:
    """Short human-readable cause for a subagent that returned no answer."""
    if stop_reason == "max_turns" or num_turns >= max_turns:
        return f"exhausted its {max_turns}-turn budget"
    if stop_reason == "cancelled":
        return "was cancelled"
    if thinking.strip():
        return "emitted reasoning but no final message"
    return f"stopped with an empty final message (stop_reason={stop_reason})"


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
