"""Inner Voice integration glue for the chat router (thin observer model).

Replaces the old Stage 1-7 ensemble/grading/intra-turn machinery with a
single observer task per turn. Public surface that `messages.py` imports:

  * `_session_inner_voice_enabled`           — flag from session JSON
  * `_session_iv_evaluate_user_turns_enabled` — user-turn opt-in flag
  * `_iv_should_fire_on_turn`                — single-source-of-truth gate
  * `build_iv_hook_registry`                 — fresh HookRegistry per turn
  * `attach_observer_for_turn`               — install observer onto a turn

Everything judgment-related lives in `app.inner_voice.observer` and its
prompt module. Python here is plumbing only.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from app.harness import HookRegistry
from app.paths import SESSIONS_DIR
from app.inner_voice.observer import (
    ObserverState,
    close_observer,
    extract_goal_card,
    install_observer,
)


logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Session opt-in helpers
# ---------------------------------------------------------------------------

def _session_inner_voice_enabled(session_id: str) -> bool:
    """Read the `inner_voice` flag from the session JSON.

    Returns False on any miss (no session yet, malformed JSON, missing
    field). Inner Voice is opt-in only.
    """
    if not session_id:
        return False
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return False
    try:
        data = json.loads(meta_path.read_text())
        return bool(data.get("inner_voice", False))
    except Exception:
        return False


def _session_iv_evaluate_user_turns_enabled(session_id: str) -> bool:
    """Read the `inner_voice_evaluate_user_turns` flag from session JSON.

    Default off — chat sessions don't pay the observer cost on user-driven
    chat unless explicitly opted in. Ambient/autonomy turns always get the
    observer when IV is enabled.
    """
    if not _session_inner_voice_enabled(session_id):
        return False
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return False
    try:
        data = json.loads(meta_path.read_text())
        return bool(data.get("inner_voice_evaluate_user_turns", False))
    except Exception:
        return False


def _iv_should_fire_on_turn(session_id: str, turn_source: str) -> bool:
    """Single gate: should the observer fire for this turn?

    True iff the session opted into Inner Voice AND either the turn is
    ambient OR the session opted into user-turn evaluation.
    """
    if not _session_inner_voice_enabled(session_id):
        return False
    if turn_source == "ambient":
        return True
    return _session_iv_evaluate_user_turns_enabled(session_id)


# ---------------------------------------------------------------------------
# Hook registry (per-turn)
# ---------------------------------------------------------------------------

def build_iv_hook_registry(session_id: str) -> HookRegistry:
    """Build an empty HookRegistry for an IV-enabled session.

    No callbacks are added at construction time. The observer installs its
    own callbacks (PreToolUse for the deny-tool gate, OnEvent for the
    stream tap) inside `attach_observer_for_turn`. Callers that need a
    HookRegistry before knowing the turn ID can call this and wire it
    later.
    """
    return HookRegistry()


# Backward-compat alias used by the older RunOptions construction sites in
# messages.py — they call `_inner_voice_hooks_dict(session_id)`.
_inner_voice_hooks_dict = build_iv_hook_registry


# ---------------------------------------------------------------------------
# Observer attach (per-turn)
# ---------------------------------------------------------------------------

async def attach_observer_for_turn(
    *,
    session_id: str,
    turn_id: str,
    turn_source: str,
    user_request: str,
    options: Any,                  # RunOptions; avoid import cycle
    chat_messages_handle: list[dict[str, Any]],
    cancel_event: asyncio.Event,
    enqueue_ambient_callback: Callable[[str, str], Awaitable[None]] | None = None,
    clarify_callback: Callable[[str, str], Awaitable[None]] | None = None,
) -> ObserverState | None:
    """Install the observer onto `options.hooks` for one turn.

    This is async because it runs goal extraction (one LLM call) before
    the primary turn starts. Returns the ObserverState, or None if Inner
    Voice is disabled for this session/turn source.

    Side effects:
      * Sets `options.hooks` to a fresh HookRegistry if none exists.
      * Sets `options.chat_messages_handle = chat_messages_handle` so the
        harness reads from the same list the observer mutates.
    """
    if not _iv_should_fire_on_turn(session_id, turn_source):
        return None

    if options.hooks is None:
        options.hooks = HookRegistry()
    options.chat_messages_handle = chat_messages_handle

    # Goal extraction — one LLM call before the primary turn starts. Best-
    # effort; on failure observer runs in lighter-touch mode (no goal card).
    goal_card = await extract_goal_card(user_request)

    state = install_observer(
        hooks=options.hooks,
        session_id=session_id,
        turn_id=turn_id,
        user_request=user_request,
        chat_messages_handle=chat_messages_handle,
        cancel_event=cancel_event,
        primary_model=options.model,
        enqueue_ambient_callback=enqueue_ambient_callback,
        clarify_callback=clarify_callback,
        goal_card=goal_card,
    )
    logger.info(
        "[iv.observer] attached session=%s turn=%s source=%s budget=%d "
        "goal_card=%s",
        session_id, turn_id, turn_source, state.intervention_budget,
        "extracted" if goal_card else "none",
    )
    return state


__all__ = [
    "ObserverState",
    "_session_inner_voice_enabled",
    "_session_iv_evaluate_user_turns_enabled",
    "_iv_should_fire_on_turn",
    "_inner_voice_hooks_dict",
    "attach_observer_for_turn",
    "build_iv_hook_registry",
    "close_observer",
]
