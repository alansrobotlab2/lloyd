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

import usage_store
from app import event_log as _event_log
from app.harness import HookRegistry
from app.paths import SESSIONS_DIR
from app.inner_voice import guards as _guards
from app.inner_voice import observer_prompt as _prompt
from app.inner_voice.observer import (
    ObserverState,
    _observer_cfg,
    close_observer,
    extract_goal_card,
    install_observer,
)


logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Session opt-in helpers
# ---------------------------------------------------------------------------

def _session_iv_flags(session_id: str) -> tuple[bool, bool]:
    """Read both Inner Voice opt-in flags in ONE pass over the session JSON.

    `_iv_should_fire_on_turn` used to parse the file three times per call:
    once itself, then once more inside `_session_iv_evaluate_user_turns_enabled`,
    which re-invoked `_session_inner_voice_enabled` first. The file is read on
    the turn's critical path and can be several megabytes on a long session.

    Returns `(enabled, evaluate_user_turns)`, both False on any miss — no
    session yet, malformed JSON, missing field. Inner Voice is opt-in only.
    """
    if not session_id:
        return (False, False)
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return (False, False)
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return (False, False)
    enabled = bool(data.get("inner_voice", False))
    if not enabled:
        # The user-turn flag is meaningless without the master switch, and
        # reporting it True would let a caller act on it alone.
        return (False, False)
    return (enabled, bool(data.get("inner_voice_evaluate_user_turns", False)))


def _session_inner_voice_enabled(session_id: str) -> bool:
    """Read the `inner_voice` flag from the session JSON.

    Returns False on any miss (no session yet, malformed JSON, missing
    field). Inner Voice is opt-in only.
    """
    return _session_iv_flags(session_id)[0]


def _session_iv_evaluate_user_turns_enabled(session_id: str) -> bool:
    """Read the `inner_voice_evaluate_user_turns` flag from session JSON.

    Default off — chat sessions don't pay the observer cost on user-driven
    chat unless explicitly opted in. Ambient/autonomy turns always get the
    observer when IV is enabled.
    """
    return _session_iv_flags(session_id)[1]


# Ambient turns the observer produced for itself. Plain `inner_voice`
# ambients are discretionary follow-ups and must NOT be observed: the
# intervention budget resets every turn, so an observer that watches its
# own follow-ups can spawn and re-judge them indefinitely.
#
# `inner_voice_goal` is the exception, and it has to be. A /goal
# follow-up exists precisely so the goal can be re-evaluated; refusing to
# observe it means `evaluate_goal_completion` never runs a second time,
# `session.goal.attempts` never advances past 1, and `max_attempts` is
# unreachable. The runaway risk that justifies the blanket refusal is
# handled here by the attempt cap instead — attempts increments on every
# unmet evaluation and escalates to `clarify` at the ceiling.
_SELF_OBSERVED_PRODUCERS = frozenset({"inner_voice_goal"})


def _iv_should_fire_on_turn(
    session_id: str, turn_source: str, producer_source: str = "",
) -> bool:
    """Single gate: should the observer fire for this turn?

    True iff the session opted into Inner Voice AND either the turn is
    ambient OR the session opted into user-turn evaluation.

    Discretionary IV-produced ambient turns are not observed (see
    `_SELF_OBSERVED_PRODUCERS` above for the one exception and why it is
    bounded). The deaf spot on a plain IV ambient — a stall on the retry
    isn't caught — is the accepted tradeoff: the user has UI controls to
    drive follow-up, and the next user turn re-opens IV observation.
    """
    enabled, evaluate_user_turns = _session_iv_flags(session_id)
    if not enabled:
        return False
    if turn_source == "ambient":
        # P3: a memory flush is bookkeeping the model does for itself before
        # compaction; there is no request to hold it to and nobody reading.
        if producer_source == "memory_flush":
            return False
        if producer_source.startswith("inner_voice"):
            return producer_source in _SELF_OBSERVED_PRODUCERS
        return True
    return evaluate_user_turns


def _load_prior_turn_interventions(
    session_id: str, *, current_turn_id: str, limit: int = 6,
) -> list[dict[str, Any]]:
    """Load interventions from EARLIER turns of this session.

    Cross-turn memory. The observer attaches fresh every turn with
    `decisions_this_turn` empty, so before this it could raise the same
    concern on five consecutive turns without ever noticing the nudge
    wasn't landing.

    Interventions only — the hundreds of noop rows carry no lesson
    forward and would swamp the prompt. Best-effort: any failure returns
    [] and the observer runs as it did before.
    """
    if not session_id:
        return []
    try:
        rows = usage_store.list_inner_voice_observations(
            session_id=session_id, limit=200,
        )
    except Exception as e:  # noqa: BLE001 — non-fatal
        logger.warning("[iv.observer] prior-intervention load failed: %s", e)
        return []
    keep = {"inject", "cancel", "ambient", "clarify"}
    out: list[dict[str, Any]] = []
    for r in rows:  # newest first
        if r.get("turn_id") == current_turn_id:
            continue
        if (r.get("action") or "") not in keep:
            continue
        out.append({
            "trigger": r.get("trigger"),
            "action": r.get("action"),
            "reason": r.get("reason"),
            "turn_id": r.get("turn_id"),
        })
        if len(out) >= limit:
            break
    out.reverse()  # oldest first, so the prompt reads chronologically
    return out


def _recent_exchanges_for_goal_extraction(
    session_id: str, *, current_user_text: str, max_messages: int = 6,
) -> list[dict[str, str]]:
    """Return the last few user/assistant text exchanges from the session,
    excluding the just-appended current user message.

    Used to give the goal extractor context for follow-up messages like
    "yeah do it" or "still broken" so it can resolve them against the
    prior thread instead of producing an empty goal card.

    Returns [] on any miss (no session yet, parse error, no prior turns).
    """
    if not session_id:
        return []
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return []
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return []
    msgs = data.get("messages") or []
    out: list[dict[str, str]] = []
    # Walk backwards collecting user/assistant text. Stop after we drop
    # the trailing current-user-message and find max_messages priors.
    seen_current = False
    for m in reversed(msgs):
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        # Skip the observer's own breadcrumbs (inner_voice_inject,
        # inner_voice_cancel, ...). They wear role=user/assistant but are
        # IV-generated; feeding them back makes the next turn's goal card
        # anchor on IV's own hallucinated demands.
        src = m.get("source") or ""
        if isinstance(src, str) and src.startswith("inner_voice_"):
            continue
        text = ""
        for chunk in m.get("content") or []:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text += chunk.get("text") or ""
        text = text.strip()
        if not text:
            continue
        # Skip the just-appended copy of the current request.
        if not seen_current and role == "user" and text == current_user_text.strip():
            seen_current = True
            continue
        out.append({"role": role, "text": text})
        if len(out) >= max_messages:
            break
    out.reverse()
    return out


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

def _observer_priority(options: Any, platform: str) -> int | None:
    """The vLLM priority the observer's own calls run at.

    Normally the watched turn's, so an observer never preempts the primary
    it is watching. One step lower (numerically higher — priority ASC) on an
    unattended turn: a worker round's observer calling the primary at the
    round's own priority puts a second-opinion request in front of a human's
    first token, and there is no human waiting on the round.
    """
    base = getattr(options, "priority", None)
    if base is None:
        return None
    try:
        from app.sessions_io import NON_USER_PLATFORMS
    except Exception:  # noqa: BLE001
        return base
    if platform and platform in NON_USER_PLATFORMS:
        return int(base) + 1
    return base


def _goal_source_text(session_id: str, user_request: str, turn_source: str,
                      producer_source: str = "") -> str:
    """What the goal card is extracted from: the USER's words (IV plan R2).

    2026-08-30 (session 20260830_182633_ive386): the user asked for YouTube
    highlights, a notification landed in the same turn, the card was built
    from the notification, and the observer then called the user's own
    request out of scope three times and cancelled it. So injected blocks are
    stripped, and an ambient turn (a notification, a producer, a follow-up)
    takes its card from the last message the user actually typed — the
    ambient text is what arrived, not what was asked. An `inner_voice_goal`
    follow-up is the exception: its text IS the goal's next step.
    """
    text = _guards.strip_injected_blocks(user_request)
    if turn_source != "ambient" or producer_source == "inner_voice_goal":
        return text or user_request
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    try:
        msgs = json.loads(meta_path.read_text()).get("messages") or []
    except Exception:  # noqa: BLE001
        return text or user_request
    for m in reversed(msgs):
        if m.get("role") != "user" or (m.get("source") or "user") != "user":
            continue
        body = m.get("content")
        if isinstance(body, list):
            body = "".join(c.get("text") or "" for c in body
                           if isinstance(c, dict) and c.get("type") == "text")
        body = _guards.strip_injected_blocks(str(body or ""))
        if body:
            return body
    return text or user_request


def goal_card_anchor(state: ObserverState):
    """A `state_anchor` that shows the primary its goal card once it exists.

    The card used to be glued onto the turn's user message before the first
    request, which is what made extraction block the first token. Now it is
    appended at the first iteration boundary after extraction finishes —
    append-only, so nothing already prefilled changes. Compose it with the
    turn's other anchors (`compose_state_anchors`).
    """
    delivered = {"done": False}

    async def anchor(_iteration: int) -> list[dict[str, Any]]:
        if delivered["done"] or not state.goal_card:
            return []
        if not _observer_cfg().get("goal_card_to_primary", True):
            return []
        block = _prompt.build_goal_card_block_for_primary(state.goal_card)
        if not block:
            return []
        delivered["done"] = True
        from app.deadline_anchor import ANCHOR_TAG
        return [{"role": "user", "content": block,
                 ANCHOR_TAG: {"kind": "goal_card", "level": "first"}}]

    return anchor


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
    persist_intervention_callback: Callable[[str, str, str], Awaitable[None]] | None = None,
    subliminal_context: str = "",
    producer_source: str = "",
    todos: list[dict[str, Any]] | None = None,
    plan_artifact: dict[str, Any] | None = None,
    persistent_goal: dict[str, Any] | None = None,
    platform: str = "",
    source: str = "",
) -> ObserverState | None:
    """Install the observer onto `options.hooks` for one turn.

    Goal extraction is started here and runs beside the primary's first
    iteration (it used to be awaited before it). Returns the ObserverState,
    or None if Inner Voice is disabled for this session/turn source. The
    caller composes `goal_card_anchor(state)` into `options.state_anchor`
    AFTER setting its own anchors, or the card never reaches the primary.

    Side effects:
      * Sets `options.hooks` to a fresh HookRegistry if none exists.
      * Sets `options.chat_messages_handle = chat_messages_handle` so the
        harness reads from the same list the observer mutates.
    """
    if not _iv_should_fire_on_turn(session_id, turn_source, producer_source):
        return None

    if options.hooks is None:
        options.hooks = HookRegistry()
    options.chat_messages_handle = chat_messages_handle
    cfg = _observer_cfg()

    prior_interventions = (
        _load_prior_turn_interventions(
            session_id,
            current_turn_id=turn_id,
            limit=int(cfg.get("cross_turn_memory_limit", 6)),
        )
        if cfg.get("cross_turn_memory_enabled", True)
        else []
    )

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
        persist_intervention_callback=persist_intervention_callback,
        goal_card=None,
        subliminal_context=subliminal_context,
        todos=todos,
        plan_artifact=plan_artifact,
        persistent_goal=persistent_goal,
        prior_turn_interventions=prior_interventions,
        max_turns=int(getattr(options, "max_turns", 0) or 0),
        # The SAME meter the loop relieves against. Without this the
        # observer judges a turn at 241k tokens exactly as it judges one at
        # 40k, which is how round 875 was injected into with no room left to
        # answer in.
        context_meter=getattr(options, "context_meter", None),
        platform=platform,
        source=source,
        # Observer calls are scheduled like the turn they watch — except on
        # an unattended one, which yields to anything a human is waiting on.
        priority=_observer_priority(options, platform),
    )

    # Goal extraction runs BESIDE iteration 1, not in front of it (IV plan
    # R2): awaited, it added up to 8 s to the first token of every observed
    # turn. It is a pending task of the observer, so the terminal review —
    # the one that reads the card — drains it first. The primary is shown the
    # card by `goal_card_anchor` once it exists, appended at an iteration
    # boundary, because the system prompt and the turn's first message are
    # already prefilled by then.
    goal_text = _goal_source_text(session_id, user_request, turn_source,
                                  producer_source)

    async def _extract() -> None:
        recent = _recent_exchanges_for_goal_extraction(
            session_id, current_user_text=user_request,
        )
        card = await extract_goal_card(
            goal_text, recent_exchanges=recent,
            priority=getattr(options, "priority", None))
        state.goal_card = card
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.goal_card_extracted",
                {"user_request": goal_text, "goal_card": card or {},
                 "turn_source": turn_source},
                turn_id=turn_id,
            )
        except Exception as e:  # noqa: BLE001 — non-fatal
            logger.warning("[iv.observer] goal_card event log failed: %s", e)

    task = asyncio.ensure_future(_extract())
    state.pending_tasks.add(task)
    task.add_done_callback(state.pending_tasks.discard)

    logger.info(
        "[iv.observer] attached session=%s turn=%s source=%s budget=%d "
        "goal_card=extracting",
        session_id, turn_id, turn_source, state.intervention_budget,
    )
    return state


__all__ = [
    "ObserverState",
    "_session_iv_flags",
    "_session_inner_voice_enabled",
    "_session_iv_evaluate_user_turns_enabled",
    "_iv_should_fire_on_turn",
    "_inner_voice_hooks_dict",
    "attach_observer_for_turn",
    "build_iv_hook_registry",
    "goal_card_anchor",
    "close_observer",
]
