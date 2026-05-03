"""In-process hook registry — replaces `claude_agent_sdk.HookMatcher`.

The SDK marshalled hook events through its CLI subprocess; we now run
callbacks directly in the harness loop. The callback input/output dict
shape is preserved verbatim so existing callbacks
(`app.inner_voice.heuristics`, `app.inner_voice.intra_turn`) need only
their import line changed.

PreToolUse callback contract (preserved from SDK):

    Input dict:
        {
            "session_id": "<lloyd-session-id>",
            "tool_name": "Bash",
            "tool_input": {"command": "..."},
        }

    Output dict (deny):
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "...",
            }
        }

    Output dict (pass): {}

PostToolUse / PostToolUseFailure callbacks always return `{}` —
they're observers, not gates. They commonly spawn `asyncio.ensure_future`
work to fire Brain 2 personas without blocking the primary loop.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger("lloyd-harness-hooks")

HookCallback = Callable[[dict[str, Any], str | None, Any], Awaitable[dict[str, Any]]]


class HookRegistry:
    """Holds PreToolUse / PostToolUse / PostToolUseFailure callbacks for
    one harness invocation.

    Built fresh per turn (or per session, depending on the caller) and
    passed via `RunOptions.hooks`. The matcher format mirrors the SDK's:
    `matcher` is a tool name string ("Bash") that fires only on that
    tool, or `None` to fire on every tool.
    """

    def __init__(self) -> None:
        self._pre: list[tuple[str | None, HookCallback]] = []
        self._post: list[HookCallback] = []
        self._post_failure: list[HookCallback] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_pre_tool_use(self, matcher: str | None, cb: HookCallback) -> None:
        self._pre.append((matcher, cb))

    def add_post_tool_use(self, cb: HookCallback) -> None:
        self._post.append(cb)

    def add_post_tool_use_failure(self, cb: HookCallback) -> None:
        self._post_failure.append(cb)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def fire_pre_tool_use(
        self,
        *,
        session_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str | None = None,
    ) -> dict[str, Any]:
        """Walk matching PreToolUse callbacks. First deny wins.

        Returns the deny dict (with `hookSpecificOutput`) if any callback
        denies; returns `{}` if all pass. Callback exceptions are logged
        and treated as pass — denial must be explicit, never accidental.
        """
        input_dict: dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        for matcher, cb in self._pre:
            if matcher is not None and matcher != tool_name:
                continue
            try:
                out = await cb(input_dict, tool_use_id, None)
            except Exception as exc:
                logger.warning(
                    "PreToolUse callback raised on %s: %s", tool_name, exc, exc_info=True
                )
                continue
            if not out:
                continue
            hso = out.get("hookSpecificOutput") or {}
            if hso.get("permissionDecision") == "deny":
                return out
        return {}

    async def fire_post_tool_use(
        self,
        *,
        session_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_response: Any,
        tool_use_id: str | None = None,
    ) -> None:
        """Fire all PostToolUse observers. Return values are ignored;
        callbacks are expected to spawn their own background work."""
        input_dict: dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
        }
        for cb in self._post:
            try:
                await cb(input_dict, tool_use_id, None)
            except Exception as exc:
                logger.warning(
                    "PostToolUse callback raised on %s: %s", tool_name, exc, exc_info=True
                )

    async def fire_post_tool_use_failure(
        self,
        *,
        session_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        error: str,
        tool_use_id: str | None = None,
    ) -> None:
        """Fire all PostToolUseFailure observers (validation errors)."""
        input_dict: dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "error": error,
        }
        for cb in self._post_failure:
            try:
                await cb(input_dict, tool_use_id, None)
            except Exception as exc:
                logger.warning(
                    "PostToolUseFailure callback raised on %s: %s",
                    tool_name,
                    exc,
                    exc_info=True,
                )
