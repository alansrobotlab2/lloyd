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

    Output dict (deliver — #536/#738, the second outcome beyond deny):
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "skillDeliver": {
                    "skill": "<name>", "label": "<rule>", "content": "<text>",
                },
            }
        }

        The loop answers a deliver with `content` as the tool_result and
        `is_error=False`: the call did not run, and that is an ordinary
        outcome, not a tool error. Deny is the only outcome that reaches
        the model as `is_error=True`.

    Output dict (pass): {}

Precedence: a deny beats a deliver regardless of registration order, so a
catastrophic `Bash` is blocked rather than answered with a skill card by a
deliverer registered ahead of the safety hook.

A callback that raises (review 2026-09-24, D5): every raise is logged as a
`harness.hook_raised` event. What happens next is the registration's
`fail_closed` flag. Fail-open (the default — observers, the skill deliverer)
treats it as a pass. Fail-closed (the gates: safety, outbound content, the
grant policy) denies the call and names the error, because a gate that cannot
evaluate must not open; before this a lazy import failing inside the safety
hook let every Bash through.

PostToolUse / PostToolUseFailure callbacks always return `{}` —
they're observers, not gates. They commonly spawn `asyncio.ensure_future`
work to fire the critic personas without blocking the primary loop.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.harness import telemetry

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
        # Parallel to `_pre`, index for index: whether a raise from that
        # callback denies (a gate) or passes (an observer). Kept beside the
        # pairs rather than in them because callers and tests read `_pre` as
        # `(matcher, cb)` pairs.
        self._pre_fail_closed: list[bool] = []
        self._post: list[HookCallback] = []
        self._post_failure: list[HookCallback] = []
        # OnEvent callbacks fire for assistant_message, tool_call,
        # tool_result and result — not for every event the loop yields
        # (no text_delta, thinking_delta, system). Used by the Inner Voice
        # observer to tap the primary's stream.
        self._on_event: list[Callable[[dict[str, Any]], Awaitable[None]]] = []
        self._skill_dispatch_installed = False

    # ------------------------------------------------------------------
    # What is registered here (set by the installer, read by anything that
    # has to report the regime this registry produced — #779)
    # ------------------------------------------------------------------

    def mark_skill_dispatch_installed(self) -> None:
        """Say that the dispatch-time SKILL.md deliverer is on this registry.

        Called by `skill_dispatch.install_skill_dispatch_hook` and by nothing
        else. The registry cannot detect it: `add_pre_tool_use` takes an
        anonymous callback, so a deliverer and a safety gate are the same object
        shape from here. Without this, a paired skill experiment could not tell
        a without-arm that withheld the body from one that leaked it at the tool
        call (#536's second delivery route), and a collapsed Δ would read as an
        inert skill.
        """
        self._skill_dispatch_installed = True

    @property
    def skill_dispatch_installed(self) -> bool:
        """Whether the #536 deliverer is registered here. Reported, never inferred."""
        return self._skill_dispatch_installed

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_pre_tool_use(
        self, matcher: str | None, cb: HookCallback, *, fail_closed: bool = False,
    ) -> None:
        """Register a PreToolUse callback.

        `fail_closed=True` is for gates: if `cb` raises, the call is denied
        with the error named. The default is for observers and deliverers,
        whose failure must never block a tool the gates would allow.
        """
        self._pre.append((matcher, cb))
        self._pre_fail_closed.append(bool(fail_closed))

    def add_post_tool_use(self, cb: HookCallback) -> None:
        self._post.append(cb)

    def add_post_tool_use_failure(self, cb: HookCallback) -> None:
        self._post_failure.append(cb)

    def add_on_event(
        self, cb: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register a callback the harness loop fires for assistant_message,
        tool_call, tool_result and result events — not for text_delta,
        thinking_delta or system, so a consumer that wants the text reads it
        off assistant_message. Awaited inline by the loop: keep it cheap.
        """
        self._on_event.append(cb)

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
        tool_summary: str = "",
    ) -> dict[str, Any]:
        """Walk matching PreToolUse callbacks. First deny wins over any deliver.

        Returns the deny dict (with `hookSpecificOutput`) if any callback
        denies; returns the first callback's `skillDeliver` dict if any
        callback asks for a delivery and none denies; returns `{}` if all
        pass.

        A callback that raises writes one `harness.hook_raised` event. A
        fail-open callback's raise is then a pass; a fail-closed one's (a
        gate registered with `fail_closed=True`) is a deny naming the gate and
        the error, returned at once — so, like any deny, it beats a deliver
        held from earlier in the walk.

        A deliver is provisional until the walk finishes: holding the first one
        and continuing is what keeps registration order from deciding whether a
        `rm -rf ~` gets denied or gets a protocol card.

        `tool_summary` is the model's own caption for this call (see
        `tool_schema.SUMMARY_ARG`). It rides in `input_dict` rather than in
        `tool_input` — deliberately, because `tool_input` is what safety
        matching and the Inner Voice repetition guard read, and neither
        should ever see a free-text caption. It is carried as a separate
        key so a callback that wants the primary's stated intent can ask
        for it, and every callback that does not is unaffected.
        """
        input_dict: dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_summary": tool_summary,
        }
        deliver: dict[str, Any] | None = None
        for i, (matcher, cb) in enumerate(self._pre):
            if matcher is not None and matcher != tool_name:
                continue
            try:
                out = await cb(input_dict, tool_use_id, None)
            except Exception as exc:
                fail_closed = (i < len(self._pre_fail_closed)
                               and self._pre_fail_closed[i])
                name = getattr(cb, "__qualname__", None) or repr(cb)
                telemetry.log_harness_event(session_id, "harness.hook_raised", {
                    "hook": name,
                    "tool": tool_name,
                    "error": f"{exc.__class__.__name__}: {exc}"[:500],
                    "fail_closed": fail_closed,
                    "tool_use_id": tool_use_id,
                })
                if not fail_closed:
                    logger.warning(
                        "PreToolUse callback raised on %s: %s",
                        tool_name, exc, exc_info=True,
                    )
                    continue
                logger.error(
                    "PreToolUse gate %s raised on %s — denying: %s",
                    name, tool_name, exc, exc_info=True,
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"gate {name} raised {exc.__class__.__name__}: "
                            f"{exc} — denied because a gate that cannot "
                            "evaluate must not open"
                        ),
                    }
                }
            if not out:
                continue
            hso = out.get("hookSpecificOutput") or {}
            if hso.get("permissionDecision") == "deny":
                return out
            if deliver is None and hso.get("skillDeliver"):
                deliver = out
        return deliver or {}

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

    async def fire_on_event(self, evt: dict[str, Any]) -> None:
        """Fire all OnEvent callbacks. Errors are swallowed so an observer
        bug never breaks the primary stream."""
        for cb in self._on_event:
            try:
                await cb(evt)
            except Exception as exc:
                logger.warning(
                    "OnEvent callback raised on %s: %s",
                    evt.get("type"), exc, exc_info=True,
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
