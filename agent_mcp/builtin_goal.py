#!/usr/bin/env python3
"""Lloyd MCP Server: Persistent Goal (/goal).

A session-level "north star" the inner-voice observer reads each turn.
Distinct from the per-turn `goal_card` (which is extracted from the user
request automatically): a persistent goal is set explicitly by the user
via `/goal <text>` and survives across turns until cleared or achieved.

Two tools:

* **SetGoal(text)** — stores `session.goal = {text, set_at,
  achieved_at: null, attempts: 0}`. Auto-enables `session.inner_voice`
  and `session.inner_voice_evaluate_user_turns` so the goal loop
  actually runs. Replaces any prior goal wholesale (one goal per
  session, matches Claude Code's `/goal`).
* **ClearGoal()** — drops `session.goal` entirely. Used by the
  `/clear-goal` slash command and by the observer when the user
  resolves a stalled-clarify.

The completion loop lives in `app/inner_voice/observer.py`: at each
turn's `result` event, when `session.goal.text` is set and
`achieved_at` is null, the observer evaluates whether the turn met the
goal. Unmet → ambient follow-up turn with the evaluator's reason.
Met → mark achieved, stop looping. `attempts >= max_attempts` →
escalate to clarify so the user can intervene.

Session correlation: same pattern as builtin_todo / builtin_plan — read
``_shared.get_bound_session()`` — the contextvar bound by the
aggregator at dispatch time.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

from mcp.types import Tool

from agent_mcp._shared import get_bound_session, text_result
from app.sessions_io import mutate_session

logger = logging.getLogger("lloyd-builtin-goal")


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


async def _set_goal(args: dict[str, Any]) -> str:
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        return json.dumps({"error": "text (non-empty string) is required"})
    text = text.strip()
    if len(text) > 4000:
        return json.dumps({"error": "text must be <= 4000 chars"})

    session_id = get_bound_session()
    if not session_id:
        return json.dumps({
            "error": "SetGoal called outside a session context — _session_id not bound",
        })

    def _apply(data: dict[str, Any]) -> None:
        data["goal"] = {
            "text": text,
            "set_at": _now_iso(),
            "achieved_at": None,
            "attempts": 0,
        }
        # Auto-enable inner voice + per-user-turn evaluation. The goal
        # loop runs through the observer's result-event check; without
        # IV opted in, the goal would be stored but never evaluated.
        data["inner_voice"] = True
        data["inner_voice_evaluate_user_turns"] = True

    ok = await mutate_session(session_id, _apply)
    if not ok:
        return json.dumps({"error": f"Session {session_id} not found"})
    logger.info("[goal] SetGoal session=%s text=%r", session_id, text[:120])
    return (
        f"Goal set: {text!r}. Inner voice enabled for this session. "
        "After each turn the observer will check whether the goal has "
        "been met; if not, it queues an ambient follow-up turn with a "
        "short reason. Call ClearGoal to abandon."
    )


async def _clear_goal(_args: dict[str, Any]) -> str:
    session_id = get_bound_session()
    if not session_id:
        return json.dumps({
            "error": "ClearGoal called outside a session context — _session_id not bound",
        })

    def _apply(data: dict[str, Any]) -> None:
        data.pop("goal", None)

    ok = await mutate_session(session_id, _apply)
    if not ok:
        return json.dumps({"error": f"Session {session_id} not found"})
    logger.info("[goal] ClearGoal session=%s", session_id)
    return "Goal cleared. Inner voice stays enabled (toggle separately if you want it off)."


_SET_DESC = """Set a session-level persistent goal (the /goal slash command).

The goal is a verifiable end condition (e.g. "all tests in test/auth pass and lint is clean", "the haiku about supervisord is saved to /tmp/h.txt"). After each turn the inner-voice observer evaluates the conversation against the goal; if unmet, it queues a follow-up turn with a short reason; if met, it marks the goal achieved and stops looping.

## When to use
Long-horizon tasks where the user can articulate "done" better than they can specify each step. Bug hunts, feature builds, refactors with a clear acceptance criterion.

## Args
- `text` (required, string, max 4000 chars): the goal text. Single goal per session — setting a new goal replaces the prior one.

## Side effects
- Stores `session.goal = {text, set_at, achieved_at, attempts}`.
- Auto-enables inner voice (`session.inner_voice=true`, `session.inner_voice_evaluate_user_turns=true`) so the loop runs.

Use `ClearGoal` to abandon."""


_CLEAR_DESC = """Clear the session's persistent goal (the /clear-goal slash command).

Drops `session.goal` entirely. Used to abandon a goal that's stalled or no longer relevant. Inner voice opt-in is left as-is so the user retains explicit control of whether IV runs."""


async def list_tools():
    return [
        Tool(
            name="SetGoal",
            description=_SET_DESC,
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Goal text (verifiable end condition). Max 4000 chars."
                        ),
                        "minLength": 1,
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="ClearGoal",
            description=_CLEAR_DESC,
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "SetGoal":
        text = await _set_goal(arguments)
    elif name == "ClearGoal":
        text = await _clear_goal(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return text_result(text)
