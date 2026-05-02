"""Inner Voice (#345) Stage 7 — intra-turn progress monitoring.

Stages 0–6 cover three checkpoints: PreToolUse regex deny (Stage 1),
mid-turn text drift (Stage 3), post-loop ensemble (Stages 2–6). What's
not covered is *tool-boundary observation* — the agent firing 11 failed
``Read`` calls in a row, the search-thrash that never resolves into a
write, the wrong-tool-for-the-job pattern that an end-of-turn ensemble
catches too late.

Stage 7 wires two new Brain 2 personas into the SDK's tool-execution
flow:

  1. ``tool_result_grader`` — fires from the ``PostToolUse`` /
     ``PostToolUseFailure`` hook on every tool result. Sees the call,
     the result (or validation error), and the last few results in the
     turn. Catches validation loops, hallucination-imminent patterns,
     wrong-tool choices, and tool thrash.

  2. ``progress_monitor`` — fires every K tool calls or every M
     seconds (whichever comes first). Sees the entire turn-so-far and
     synthesizes "are we making progress?" Catches scope drift,
     stalls, and stuck loops that span multiple tool kinds.

Both share the Stage 6 dispatch infrastructure: on a disagreeing
verdict at severity ≥ ``severity_floor_for_steer`` (default 0.7 for
tool_result_grader, 0.6 for progress_monitor), we fire
``_maybe_dispatch_steer`` (lazy-imported from messages.py) which
enqueues an ambient retry via the same shared nudge budget that
``consensus_termination`` and Stage 6's nudge_proposed dispatch use.
That budget is finite per session — if Stage 4 already burned it on a
veto, Stage 7 will skip silently.

Throughput guards:

  * ``post_tool_use.max_calls_per_turn`` (default 5) — caps how many
    times we fire ``tool_result_grader`` on a single turn. Above the
    cap, an ``inner_voice.intra_turn_skipped`` event lands with reason
    ``cap_reached`` and we no-op.

  * ``progress_monitor.max_calls_per_turn`` (default 4) — same idea
    for the periodic synthesis.

  * Shared ``max_critiques_per_session`` cap (from ensemble.py) — both
    personas count against the global session budget. A session that
    already burned 50 critiques won't fire Stage 7 either.

State is in-process, keyed by Lloyd session_id, reset on every new
turn. The hook callbacks are bound via closure to the Lloyd session_id
the same way Stage 1's PreToolUse callback is — the SDK's hook input
carries the Claude-CLI UUID, not Lloyd's id.

Failure mode: the module never raises into the SDK callback path. Any
exception in the dispatcher is caught, logged, and the turn proceeds.
Tool execution must not block on a Brain 2 call; the persona dispatch
is always ``asyncio.ensure_future``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import CONFIG
from app import event_log as _event_log
from app.inner_voice import critic as _critic
from app.inner_voice import ensemble as _ensemble
from app.inner_voice import consensus_termination as _ct
from app.inner_voice.critic import Critique

logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Per-turn state — keyed by lloyd_session_id, reset on each new turn.
# ---------------------------------------------------------------------------


@dataclass
class _ToolCallRecord:
    """One tool boundary in this turn. Used both for grader user-prompt
    assembly (last-N pairs) and for loop detection in progress_monitor.
    """
    tool_name: str
    input_summary: str         # ``name(arg_keys)`` shape — never raw values
    result_excerpt: str        # first 500 chars of result or error
    is_error: bool
    when: float                # monotonic timestamp


@dataclass
class _IntraTurnState:
    """All the state we need to keep across a single turn for Stage 7.

    Reset at ``start_intra_turn`` time. The hook callbacks read this
    via ``_session_intra_state[lloyd_session_id]``.
    """
    turn_id: str
    turn_source: str
    frozen_task_intent: str
    started_at: float                            # monotonic seconds at turn start
    tool_calls: list[_ToolCallRecord] = field(default_factory=list)
    post_tool_calls_fired: int = 0               # fires of tool_result_grader
    progress_monitor_fired: int = 0              # fires of progress_monitor
    last_progress_check_at: float = 0.0          # monotonic seconds of last fire
    last_progress_check_tool_count: int = 0      # tool_call_count at last fire


_session_intra_state: dict[str, _IntraTurnState] = {}


def start_intra_turn(
    session_id: str,
    turn_id: str,
    *,
    turn_source: str,
    frozen_task_intent: str,
) -> None:
    """Initialize per-turn state. Called from ``_run_turn`` BEFORE the
    SDK ``query()`` loop begins.

    Replaces any existing state for this session — a turn boundary is a
    hard reset. If a previous Brain 2 call is still in flight when this
    runs, that call's reference to the old state object stays valid
    (Python GC keeps it alive as long as the call holds it).
    """
    now = time.monotonic()
    _session_intra_state[session_id] = _IntraTurnState(
        turn_id=turn_id,
        turn_source=turn_source,
        frozen_task_intent=frozen_task_intent or "",
        started_at=now,
        last_progress_check_at=now,
    )


def end_intra_turn(session_id: str, turn_id: str) -> None:
    """Clear the per-turn state. Called from ``_run_turn``'s finally
    block. Idempotent — safe to call multiple times.

    The turn_id check guards against a stale end-call clobbering a
    fresh turn's state (the ``finally`` of an earlier turn racing the
    ``start`` of the next). If the active state's turn_id doesn't
    match, we leave it in place.
    """
    state = _session_intra_state.get(session_id)
    if state is None:
        return
    if state.turn_id != turn_id:
        return
    _session_intra_state.pop(session_id, None)


def get_state(session_id: str) -> _IntraTurnState | None:
    """Test/debug accessor."""
    return _session_intra_state.get(session_id)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _intra_cfg() -> dict[str, Any]:
    iv = CONFIG.get("inner_voice") or {}
    cfg = dict(iv.get("intra_turn") or {})
    cfg.setdefault("enabled", True)
    cfg.setdefault("post_tool_use", {})
    cfg["post_tool_use"].setdefault("enabled", True)
    cfg["post_tool_use"].setdefault("severity_floor_for_steer", 0.7)
    cfg["post_tool_use"].setdefault("severity_floor_for_cancel", 0.95)
    cfg["post_tool_use"].setdefault("max_calls_per_turn", 5)
    cfg.setdefault("progress_monitor", {})
    cfg["progress_monitor"].setdefault("enabled", True)
    cfg["progress_monitor"].setdefault("every_n_tool_calls", 5)
    cfg["progress_monitor"].setdefault("every_n_seconds", 60)
    cfg["progress_monitor"].setdefault("severity_floor_for_steer", 0.6)
    cfg["progress_monitor"].setdefault("max_calls_per_turn", 4)
    return cfg


def is_enabled() -> bool:
    """Master kill switch — ``inner_voice.intra_turn.enabled`` (default true).

    Independent of the post-loop critic gate so Stage 7 can be A/B'd
    in isolation. Per-session opt-in still applies via the existing
    ``inner_voice`` flag in session JSON.
    """
    iv = CONFIG.get("inner_voice")
    if not iv:
        return False
    return bool(_intra_cfg().get("enabled", True))


def is_post_tool_use_enabled() -> bool:
    return is_enabled() and bool(_intra_cfg()["post_tool_use"].get("enabled", True))


def is_progress_monitor_enabled() -> bool:
    return is_enabled() and bool(_intra_cfg()["progress_monitor"].get("enabled", True))


# ---------------------------------------------------------------------------
# Tool-input/result summarization (cheap, deterministic — runs on every fire)
# ---------------------------------------------------------------------------


def _input_summary(tool_name: str, tool_input: Any) -> str:
    """Render ``tool_name(arg_keys)`` style summary. Never includes raw
    values — those can be huge (a Bash script body, a long Edit content
    block) and we don't want them on every grader prompt.
    """
    short = tool_name.rsplit("__", 1)[-1] if "__" in tool_name else tool_name
    if not isinstance(tool_input, dict):
        return f"{short}(<non-dict-input>)"
    keys = sorted(tool_input.keys())
    if not keys:
        return f"{short}()"
    return f"{short}({','.join(keys[:6])})"


def _result_excerpt(tool_response: Any, *, cap: int = 500) -> str:
    """Return a string excerpt of the tool response, truncated to ``cap``
    chars. Handles SDK shape variations: list-of-blocks, str, dict, None.
    """
    if tool_response is None:
        return "(null)"
    if isinstance(tool_response, str):
        text = tool_response
    elif isinstance(tool_response, dict):
        # Common SDK shape: {"content": [{"type":"text","text":"..."}], ...}
        if isinstance(tool_response.get("content"), list):
            parts: list[str] = []
            for block in tool_response["content"]:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            text = "\n".join(parts) if parts else json.dumps(tool_response, default=str)
        else:
            text = json.dumps(tool_response, default=str)
    elif isinstance(tool_response, list):
        parts = []
        for block in tool_response:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    else:
        text = str(tool_response)
    if len(text) > cap:
        text = text[: cap - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# Same-shape error detection — helper for the grader's loop catch.
# ---------------------------------------------------------------------------


def _same_shape_error_count(state: _IntraTurnState, *, lookback: int = 6) -> int:
    """Return the count of consecutive same-shape errors in the most
    recent ``lookback`` tool calls.

    "Same shape" means: same tool_name AND error excerpt's first 80
    chars match. Catches the "Read called 4× with missing file_path"
    pattern from session ``20260502_044548_iv5f83`` without false-
    positiving on the same tool returning legitimately different
    errors over time.
    """
    recent = state.tool_calls[-lookback:]
    if not recent:
        return 0
    last = recent[-1]
    if not last.is_error:
        return 0
    sig = (last.tool_name, last.result_excerpt[:80])
    count = 0
    for rec in reversed(recent):
        if not rec.is_error:
            break
        if (rec.tool_name, rec.result_excerpt[:80]) == sig:
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# User-prompt assembly for the two personas
# ---------------------------------------------------------------------------


_TASK_INTENT_CAP = 1500
_RESULT_EXCERPT_CAP = 500
_RECENT_TOOLS_FOR_GRADER = 3


def _format_recent_tool_results(state: _IntraTurnState, *, n: int) -> str:
    """Render the ``<recent_tool_results>`` block. ``n`` is how many of
    the most-recent calls (excluding the current one) to surface.
    """
    if len(state.tool_calls) <= 1:
        return "(empty)"
    # The CURRENT tool call is the last entry; recent = the n before it.
    recent = state.tool_calls[-(n + 1) : -1]
    if not recent:
        return "(empty)"
    lines: list[str] = []
    for rec in recent:
        excerpt = rec.result_excerpt[:_RESULT_EXCERPT_CAP]
        marker = "(error: " if rec.is_error else ""
        suffix = ")" if rec.is_error else ""
        lines.append(f"- `{rec.input_summary}` → {marker}{excerpt}{suffix}")
    return "\n".join(lines)


def _build_grader_user_prompt(state: _IntraTurnState) -> str:
    """Build the ``tool_result_grader`` user prompt. Layout matches the
    persona file's ``Examples`` section so calibration shots stay
    valid.
    """
    if not state.tool_calls:
        return "<task>\n(empty)\n</task>"
    current = state.tool_calls[-1]
    excerpt = current.result_excerpt[:_RESULT_EXCERPT_CAP]
    if current.is_error:
        result_render = f"(error: {excerpt})"
    else:
        result_render = excerpt or "(empty result)"
    intent = state.frozen_task_intent or "(no task text)"
    if len(intent) > _TASK_INTENT_CAP:
        intent = intent[: _TASK_INTENT_CAP - 3] + "..."
    return (
        "<task>\n"
        f"{intent}\n"
        "</task>\n\n"
        "<recent_tool_results>\n"
        f"{_format_recent_tool_results(state, n=_RECENT_TOOLS_FOR_GRADER)}\n"
        "</recent_tool_results>\n\n"
        f"<this_tool_call>\n{current.input_summary}\n</this_tool_call>\n\n"
        f"<this_tool_result>\n{result_render}\n</this_tool_result>"
    )


_TURN_HISTORY_CAP_PER_LINE = 200


def _build_progress_user_prompt(state: _IntraTurnState) -> str:
    """Build the ``progress_monitor`` user prompt. Includes the full
    turn-history-so-far via the tool_calls list. Text deltas are
    omitted in v1; the persona synthesizes from tool patterns.
    """
    elapsed = time.monotonic() - state.started_at
    intent = state.frozen_task_intent or "(no task text)"
    if len(intent) > _TASK_INTENT_CAP:
        intent = intent[: _TASK_INTENT_CAP - 3] + "..."

    history_lines: list[str] = []
    for rec in state.tool_calls:
        excerpt = rec.result_excerpt[:_TURN_HISTORY_CAP_PER_LINE]
        if rec.is_error:
            history_lines.append(f"- [tool] {rec.input_summary} → [result] (error: {excerpt})")
        else:
            history_lines.append(f"- [tool] {rec.input_summary} → [result] {excerpt}")
    history = "\n".join(history_lines) if history_lines else "(empty)"

    return (
        "<task>\n"
        f"{intent}\n"
        "</task>\n\n"
        f"<elapsed_seconds>\n{elapsed:.0f}\n</elapsed_seconds>\n\n"
        f"<tool_call_count>\n{len(state.tool_calls)}\n</tool_call_count>\n\n"
        "<turn_history>\n"
        f"{history}\n"
        "</turn_history>"
    )


# ---------------------------------------------------------------------------
# Steer dispatch — uses lazy import to avoid the circular dep with messages.py
# ---------------------------------------------------------------------------


async def _dispatch_steer_via_messages(
    *,
    session_id: str,
    turn_id: str,
    turn_source: str,
    critique: Critique,
    persona_kind_label: str,
    response_text: str,
) -> bool:
    """Build a synthetic ``critiques + agg`` shape compatible with
    ``_maybe_dispatch_steer`` and fire it. Returns True iff the steer
    was actually dispatched (budget had room AND config gate open).

    We rebuild the agg dict because ``_maybe_dispatch_steer`` keys off
    ``action_chosen == 'nudge_proposed'`` AND the per-critique
    severity. Cleanest path is to construct the same shape ensemble.py
    would have produced for a single-persona "ensemble".
    """
    # Lazy import to break the circular dep: messages.py imports this
    # module to wire the hook callbacks; if we imported messages.py at
    # module-load time the import order would break.
    try:
        from app.routers.messages import _maybe_dispatch_steer
    except ImportError as e:
        logger.warning(
            "[inner_voice.intra_turn] could not import _maybe_dispatch_steer: %s",
            e,
        )
        return False

    severity = float(critique.severity)
    agg = {
        "action_chosen": "nudge_proposed",
        "severity_max": severity,
        "severity_mean": severity,
        "disagree_count": 1,
        "rationale": (
            f"intra_turn:{persona_kind_label} severity={severity:.2f} "
            f"persona={critique.persona}"
        ),
    }

    async def _emit_via_turn(event: str, data: dict[str, Any]) -> None:
        # Stage 7 dispatcher fires from the hook callback path, which
        # has no SessionTurn handle. SSE delivery on the active turn
        # would be racy anyway (the hook is between two text streams);
        # we drop emit-events here. The SQLite intervention row + event
        # log line are still written by the dispatcher.
        return None

    return await _maybe_dispatch_steer(
        session_id=session_id,
        turn_id=turn_id,
        turn_source=turn_source,
        response_text=response_text,
        critiques=[critique],
        agg=agg,
        ensemble_name=f"intra_turn_{persona_kind_label}",
        emit_via_turn=_emit_via_turn,
    )


# ---------------------------------------------------------------------------
# Persona dispatchers — both spawned via asyncio.ensure_future from the hooks
# ---------------------------------------------------------------------------


async def _run_tool_result_grader(
    *,
    session_id: str,
    state_snapshot_turn_id: str,
) -> None:
    """Fire ``tool_result_grader`` against the current state. Best-effort.

    Skips if:
      - turn_id no longer matches (turn ended before we ran)
      - session-level critique cap is exhausted
      - persona file is missing
    """
    state = _session_intra_state.get(session_id)
    if state is None or state.turn_id != state_snapshot_turn_id:
        return

    if not _ensemble._under_session_cap(session_id, n=1):
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.intra_turn_skipped",
                {
                    "persona": "tool_result_grader",
                    "reason": "session_critique_cap_reached",
                    "tool_call_index": len(state.tool_calls),
                },
                turn_id=state.turn_id,
            )
        except Exception:
            pass
        return

    user_prompt = _build_grader_user_prompt(state)
    sse_count = _same_shape_error_count(state)

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.intra_turn_check_started",
            {
                "persona": "tool_result_grader",
                "tool_call_index": len(state.tool_calls),
                "tool_name": state.tool_calls[-1].tool_name if state.tool_calls else "",
                "is_error": state.tool_calls[-1].is_error if state.tool_calls else False,
                "same_shape_error_count": sse_count,
            },
            turn_id=state.turn_id,
        )
    except Exception:
        pass

    critique = await _ensemble._run_one_persona(
        session_id=session_id,
        turn_id=state.turn_id,
        persona_name="tool_result_grader",
        user_prompt=user_prompt,
        response_excerpt=(state.tool_calls[-1].result_excerpt if state.tool_calls else ""),
    )
    if critique is None:
        return

    _ensemble._bump_session_count(session_id, n=1)

    # Decide action_taken locally — same logic ensemble._resolve_per_critique_action
    # uses, but the dispatch happens inline below.
    if critique.error is not None:
        critique.action_taken = "log_only"
    elif not critique.disagrees:
        critique.action_taken = "agreement"
    else:
        critique.action_taken = "log_only"  # may be promoted by dispatch below

    # Persist critique row regardless of dispatch outcome.
    try:
        _ensemble._persist_critique(
            session_id=session_id, turn_id=state.turn_id, critique=critique,
        )
    except Exception as e:
        logger.warning(
            "[inner_voice.intra_turn] tool_result_grader persist failed: %s", e,
        )

    # Dispatch decision — only if disagreed at severity floor and no error.
    intra_cfg = _intra_cfg()["post_tool_use"]
    floor = float(intra_cfg.get("severity_floor_for_steer", 0.7))
    if (
        critique.disagrees
        and critique.error is None
        and critique.severity >= floor
    ):
        # Stage 7 always fires steer dispatch (budget permitting). The
        # cancel-band (severity ≥ 0.95) is intentionally NOT cancelling
        # mid-turn for tool-result issues per the spec ("don't kill
        # work in progress; let the next turn pick up the steer
        # context"). Future stages may revisit.
        try:
            dispatched = await _dispatch_steer_via_messages(
                session_id=session_id,
                turn_id=state.turn_id,
                turn_source=state.turn_source,
                critique=critique,
                persona_kind_label="tool_result_grader",
                response_text="",
            )
            if dispatched:
                critique.action_taken = "steer"
                # Re-persist with action_taken = steer. Avoid double-counting
                # by writing a dedicated event rather than another row.
                try:
                    _event_log.log_event(
                        session_id,
                        "inner_voice.intra_turn_steer_promoted",
                        {
                            "persona": "tool_result_grader",
                            "severity": critique.severity,
                            "reason": critique.reason,
                        },
                        turn_id=state.turn_id,
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(
                "[inner_voice.intra_turn] tool_result_grader steer dispatch failed: %s",
                e,
            )


async def _run_progress_monitor(
    *,
    session_id: str,
    state_snapshot_turn_id: str,
    trigger_kind: str,
) -> None:
    """Fire ``progress_monitor`` against the turn-so-far. Best-effort.

    ``trigger_kind`` is one of ``tool_count`` or ``elapsed_time`` —
    surfaced in the event log so meta-review can see which trigger
    fired most often.
    """
    state = _session_intra_state.get(session_id)
    if state is None or state.turn_id != state_snapshot_turn_id:
        return

    if not _ensemble._under_session_cap(session_id, n=1):
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.intra_turn_skipped",
                {
                    "persona": "progress_monitor",
                    "reason": "session_critique_cap_reached",
                    "trigger_kind": trigger_kind,
                },
                turn_id=state.turn_id,
            )
        except Exception:
            pass
        return

    user_prompt = _build_progress_user_prompt(state)

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.intra_turn_check_started",
            {
                "persona": "progress_monitor",
                "trigger_kind": trigger_kind,
                "tool_call_count": len(state.tool_calls),
                "elapsed_seconds": int(time.monotonic() - state.started_at),
            },
            turn_id=state.turn_id,
        )
    except Exception:
        pass

    critique = await _ensemble._run_one_persona(
        session_id=session_id,
        turn_id=state.turn_id,
        persona_name="progress_monitor",
        user_prompt=user_prompt,
        response_excerpt="",
    )
    if critique is None:
        return

    _ensemble._bump_session_count(session_id, n=1)

    if critique.error is not None:
        critique.action_taken = "log_only"
    elif not critique.disagrees:
        critique.action_taken = "agreement"
    else:
        critique.action_taken = "log_only"

    try:
        _ensemble._persist_critique(
            session_id=session_id, turn_id=state.turn_id, critique=critique,
        )
    except Exception as e:
        logger.warning(
            "[inner_voice.intra_turn] progress_monitor persist failed: %s", e,
        )

    intra_cfg = _intra_cfg()["progress_monitor"]
    floor = float(intra_cfg.get("severity_floor_for_steer", 0.6))
    if (
        critique.disagrees
        and critique.error is None
        and critique.severity >= floor
    ):
        try:
            dispatched = await _dispatch_steer_via_messages(
                session_id=session_id,
                turn_id=state.turn_id,
                turn_source=state.turn_source,
                critique=critique,
                persona_kind_label="progress_monitor",
                response_text="",
            )
            if dispatched:
                critique.action_taken = "steer"
                try:
                    _event_log.log_event(
                        session_id,
                        "inner_voice.intra_turn_steer_promoted",
                        {
                            "persona": "progress_monitor",
                            "severity": critique.severity,
                            "reason": critique.reason,
                            "trigger_kind": trigger_kind,
                        },
                        turn_id=state.turn_id,
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(
                "[inner_voice.intra_turn] progress_monitor steer dispatch failed: %s",
                e,
            )


# ---------------------------------------------------------------------------
# Hook callback factories — closure-bound to lloyd_session_id, mirroring the
# Stage 1 pattern in heuristics.py.
# ---------------------------------------------------------------------------


async def _record_tool_result(
    *,
    session_id: str,
    tool_name: str,
    tool_input: Any,
    tool_response: Any,
    error: str | None,
) -> bool:
    """Append the tool boundary to per-turn state and decide whether to
    fire ``tool_result_grader``. Returns True iff a fire was scheduled.
    """
    state = _session_intra_state.get(session_id)
    if state is None:
        return False  # turn not started; nothing to record

    record = _ToolCallRecord(
        tool_name=tool_name,
        input_summary=_input_summary(tool_name, tool_input),
        result_excerpt=(error if error is not None else _result_excerpt(tool_response)),
        is_error=(error is not None),
        when=time.monotonic(),
    )
    state.tool_calls.append(record)

    # Always log the boundary — useful even when the grader is skipped.
    try:
        _event_log.log_event(
            session_id,
            "inner_voice.intra_turn_tool_boundary",
            {
                "tool_name": tool_name,
                "input_summary": record.input_summary,
                "is_error": record.is_error,
                "result_chars": len(record.result_excerpt),
                "tool_call_index": len(state.tool_calls),
            },
            turn_id=state.turn_id,
        )
    except Exception:
        pass

    # Decide whether to fire tool_result_grader.
    if not is_post_tool_use_enabled():
        return False
    cap = int(_intra_cfg()["post_tool_use"].get("max_calls_per_turn", 5))
    if state.post_tool_calls_fired >= cap:
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.intra_turn_skipped",
                {
                    "persona": "tool_result_grader",
                    "reason": "cap_reached",
                    "cap": cap,
                    "tool_call_index": len(state.tool_calls),
                },
                turn_id=state.turn_id,
            )
        except Exception:
            pass
        return False

    state.post_tool_calls_fired += 1
    asyncio.ensure_future(_run_tool_result_grader(
        session_id=session_id,
        state_snapshot_turn_id=state.turn_id,
    ))

    # Also check if progress_monitor should fire — same hook is the
    # natural pulse for both checks.
    _maybe_fire_progress_monitor(session_id, trigger_check=True)
    return True


def _maybe_fire_progress_monitor(session_id: str, *, trigger_check: bool) -> None:
    """Decide whether to fire ``progress_monitor`` based on tool count
    or elapsed time since the last fire. ``trigger_check=True`` means
    we're being called from a tool-boundary event; pass False from a
    timer-driven path.
    """
    state = _session_intra_state.get(session_id)
    if state is None:
        return
    if not is_progress_monitor_enabled():
        return

    pm_cfg = _intra_cfg()["progress_monitor"]
    cap = int(pm_cfg.get("max_calls_per_turn", 4))
    if state.progress_monitor_fired >= cap:
        if trigger_check:
            try:
                _event_log.log_event(
                    session_id,
                    "inner_voice.intra_turn_skipped",
                    {
                        "persona": "progress_monitor",
                        "reason": "cap_reached",
                        "cap": cap,
                    },
                    turn_id=state.turn_id,
                )
            except Exception:
                pass
        return

    every_n = max(1, int(pm_cfg.get("every_n_tool_calls", 5)))
    every_s = max(1, int(pm_cfg.get("every_n_seconds", 60)))
    now = time.monotonic()

    tool_due = (
        len(state.tool_calls) - state.last_progress_check_tool_count >= every_n
    )
    time_due = (now - state.last_progress_check_at) >= every_s
    if not (tool_due or time_due):
        return

    state.progress_monitor_fired += 1
    state.last_progress_check_at = now
    state.last_progress_check_tool_count = len(state.tool_calls)
    trigger_kind = "tool_count" if tool_due else "elapsed_time"

    asyncio.ensure_future(_run_progress_monitor(
        session_id=session_id,
        state_snapshot_turn_id=state.turn_id,
        trigger_kind=trigger_kind,
    ))


def make_post_tool_use_callback(lloyd_session_id: str):
    """Closure-bound async PostToolUse hook for ``ClaudeAgentOptions``.

    Fires on every successful tool result. Records the boundary into
    state, then optionally spawns ``tool_result_grader`` (and possibly
    ``progress_monitor``) via ``asyncio.ensure_future``. Returns
    immediately — the SDK's tool flow does NOT block on Brain 2 calls.
    """
    async def _bound(
        input_data: dict[str, Any],   # PostToolUseHookInput TypedDict
        tool_use_id: str | None,
        context: Any,                 # HookContext
    ) -> dict[str, Any]:
        try:
            tool_name = input_data.get("tool_name", "") or ""
            tool_input = input_data.get("tool_input", {}) or {}
            tool_response = input_data.get("tool_response", None)
            await _record_tool_result(
                session_id=lloyd_session_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_response=tool_response,
                error=None,
            )
        except Exception as e:
            logger.warning("[inner_voice.intra_turn] PostToolUse hook failed: %s", e)
        # Always pass through. We never deny via PostToolUse.
        return {}
    return _bound


def make_post_tool_use_failure_callback(lloyd_session_id: str):
    """Closure-bound async PostToolUseFailure hook.

    Fires when tool execution failed at the SDK layer (validation
    error, tool raise, schema mismatch). We still want to see these —
    in fact they're the highest-signal events for the validation-loop
    catch (acceptance gate 1).
    """
    async def _bound(
        input_data: dict[str, Any],   # PostToolUseFailureHookInput
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        try:
            tool_name = input_data.get("tool_name", "") or ""
            tool_input = input_data.get("tool_input", {}) or {}
            error = input_data.get("error", "") or "(unknown error)"
            await _record_tool_result(
                session_id=lloyd_session_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_response=None,
                error=error,
            )
        except Exception as e:
            logger.warning(
                "[inner_voice.intra_turn] PostToolUseFailure hook failed: %s", e,
            )
        return {}
    return _bound


# ---------------------------------------------------------------------------
# Public introspection — used by tests and /api/inner_voice/state
# ---------------------------------------------------------------------------


def get_intra_turn_summary(session_id: str) -> dict[str, Any]:
    """Snapshot of the current state for /api/inner_voice/state. Empty
    dict if no active turn.
    """
    state = _session_intra_state.get(session_id)
    if state is None:
        return {}
    return {
        "turn_id": state.turn_id,
        "turn_source": state.turn_source,
        "elapsed_seconds": int(time.monotonic() - state.started_at),
        "tool_call_count": len(state.tool_calls),
        "post_tool_calls_fired": state.post_tool_calls_fired,
        "progress_monitor_fired": state.progress_monitor_fired,
        "post_tool_use_cap": int(_intra_cfg()["post_tool_use"].get("max_calls_per_turn", 5)),
        "progress_monitor_cap": int(_intra_cfg()["progress_monitor"].get("max_calls_per_turn", 4)),
    }


__all__ = [
    "start_intra_turn",
    "end_intra_turn",
    "get_state",
    "get_intra_turn_summary",
    "is_enabled",
    "is_post_tool_use_enabled",
    "is_progress_monitor_enabled",
    "make_post_tool_use_callback",
    "make_post_tool_use_failure_callback",
    "_record_tool_result",
    "_maybe_fire_progress_monitor",
    "_build_grader_user_prompt",
    "_build_progress_user_prompt",
    "_same_shape_error_count",
]
