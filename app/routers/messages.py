"""Chat endpoints — SSE streaming and synchronous one-shot.

`POST /api/message/stream` enqueues a user turn onto the session's queue
and returns a StreamingResponse that subscribes to the turn's event
broker. `POST /api/sessions/{id}/inject` enqueues an ambient (background)
turn. The per-session consumer task (lazily spawned on first enqueue)
pops from the user tier before the ambient tier; a user turn arriving
while an ambient is running preempts it via the queue's cancel_event.

Client disconnect does not kill the SDK subprocess — the consumer runs
to completion, events pile up in `turn.events`, and persistence lands in
the session JSON regardless. (Task #296 Phases 1+2.)
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

from claude_agent_sdk import (
    query, ClaudeAgentOptions,
    SystemMessage, AssistantMessage, UserMessage, ResultMessage,
    HookMatcher,
)
from claude_agent_sdk import TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock
from claude_agent_sdk.types import StreamEvent
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import usage_store
from app.config import (
    CONFIG,
    _get_model_env,
    _resolve_effort,
    _resolve_thinking,
    _model_base_url,
    _resolve_model_name,
)
from app.paths import SESSIONS_DIR, PIPELINE_RUNS_DIR
from app.sessions_io import (
    SessionTurn,
    SessionQueue,
    _get_or_create_queue,
    _session_queues,
    _save_session_meta,
    _append_messages,
    mutate_session,
    _broadcast_queue_state,
    enqueue_turn,
    get_queue_state,
    set_last_user_session,
    take_ambient_decision,
    enqueue_ambient_prefetch,
    AmbientPrefetchEntry,
)
from app.mcp_discovery import _get_mcp_servers, _get_disallowed_tools
from app.post_capture import _post_session_capture, _maybe_extract_focus
from app.routers.voice import (
    extract_first_two_sentences,
    speak_text,
    tts_is_enabled,
)
from prompt_builder import build_system_prompt
from prefetch import prefetch_context
from app.compaction import (
    format_conversation_for_local,
    is_local_model,
    load_and_compact_session,
)
from app import event_log as _event_log  # Inner Voice (#345) — brain1.* capture
from app.inner_voice import heuristics as _iv_heuristics  # Inner Voice (#345) — Stage 1
from app.inner_voice import ensemble as _iv_ensemble  # Inner Voice (#345) — Stage 2
from app.inner_voice import consensus_termination as _iv_consensus  # Inner Voice (#345) — Stage 4
from app.inner_voice import grading as _iv_grading  # Inner Voice (#345) — Stage 5
from app.inner_voice import intra_turn as _iv_intra_turn  # Inner Voice (#345) — Stage 7
from usage_store import record_inner_voice_intervention


router = APIRouter()
logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Inner Voice (#345 Stage 1) — session opt-in helpers
# ---------------------------------------------------------------------------

def _session_inner_voice_enabled(session_id: str) -> bool:
    """Read the `inner_voice` flag from the session JSON.

    Returns False on any miss (no session yet, malformed JSON, missing
    field). Stage 1 is opt-in only — Brain 2 ensembles never fire on a
    session whose JSON doesn't explicitly say `inner_voice: true`.
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
    """Read the `inner_voice_evaluate_user_turns` flag from the session JSON.

    Stage 5+ — opt-in for firing the Brain 2 ensemble + grading + mid-turn
    drift on user-typed turns (i.e. chat from the Inner Voice tab), not
    just ambient/autonomy turns. Default off — chat sessions don't pay the
    Brain 2 cost unless the user explicitly turns it on.

    Consensus termination + Stage 1 completion-check stay ambient-only
    regardless: SIGNAL:TASK_COMPLETE veto and premature-stop logic exist
    to bound autonomous loops, not human-driven chat.
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
    """Single-source-of-truth gate for Brain 2 ensemble + grading + mid-turn
    drift. Returns True iff the session is opted into Inner Voice AND
    either the turn is ambient OR the session opted into user-turn
    evaluation.
    """
    if not _session_inner_voice_enabled(session_id):
        return False
    if turn_source == "ambient":
        return True
    return _session_iv_evaluate_user_turns_enabled(session_id)


def _inner_voice_hooks_dict(session_id: str) -> dict[str, list[HookMatcher]]:
    """Build the `hooks=` value for ClaudeAgentOptions on Inner Voice
    sessions.

    The PreToolUse callback is bound to Lloyd's `session_id` via closure —
    `PreToolUseHookInput.session_id` carries the Claude CLI's UUID, which
    is useless for finding the events.jsonl file. The closure ensures
    `inner_voice.pre_tool_use_evaluated` events land in the same file as
    the rest of this turn's `brain1.*` events.

    Stage 1 wires only PreToolUse (Bash matcher — the deny rules currently
    only target Bash). Stage 7 adds PostToolUse + PostToolUseFailure
    (no matcher — fires on every tool) for ``tool_result_grader`` /
    ``progress_monitor`` dispatch.
    """
    hooks: dict[str, list[HookMatcher]] = {
        "PreToolUse": [
            HookMatcher(
                matcher="Bash",
                hooks=[_iv_heuristics.make_pretooluse_callback(session_id)],
            ),
        ],
    }
    if _iv_intra_turn.is_enabled():
        # PostToolUse fires after every successful tool result. No matcher
        # = matches every tool. The closure-bound callback records the
        # tool boundary into per-turn state and possibly fires
        # `tool_result_grader` / `progress_monitor` via
        # `asyncio.ensure_future`. Tool flow is NOT blocked.
        hooks["PostToolUse"] = [
            HookMatcher(
                matcher=None,
                hooks=[_iv_intra_turn.make_post_tool_use_callback(session_id)],
            ),
        ]
        # PostToolUseFailure fires when SDK validation rejects the call
        # (missing required param, schema mismatch, tool raise). Same
        # closure pattern, but the input carries `error: str` instead of
        # `tool_response`. Critical for the validation-loop catch — the
        # failed Read calls from session 20260502_044548_iv5f83 only
        # surface here, not in PostToolUse.
        hooks["PostToolUseFailure"] = [
            HookMatcher(
                matcher=None,
                hooks=[_iv_intra_turn.make_post_tool_use_failure_callback(session_id)],
            ),
        ]
    return hooks


# ---------------------------------------------------------------------------
# Inner Voice (#345 Stage 1) — post-loop completion check
# ---------------------------------------------------------------------------

async def _inner_voice_completion_check(
    session_id: str,
    turn_id: str,
    response_text: str,
    tool_calls: list[dict],
) -> None:
    """Run the Stage 1 completion heuristic on a finished ambient turn.

    Spawned as `asyncio.ensure_future` from the ResultMessage branch in
    `_run_turn`, alongside `_post_session_capture`. Best-effort — never
    raises into the consumer. On premature termination, enqueues an
    ambient prefetch entry that surfaces in the next user turn's
    `<context>` block.
    """
    try:
        if not _iv_heuristics.is_completion_check_enabled():
            return
        verdict = _iv_heuristics.evaluate_completion(response_text, tool_calls)
        # Always log the evaluation — passes are forensic data for tuning
        # the heuristic and later A/B'ing against Brain 2 ensembles.
        _event_log.log_event(
            session_id,
            "inner_voice.completion_check_evaluated",
            {
                "premature": verdict["premature"],
                "reason": verdict["reason"],
                "signal_seen": verdict["signal_seen"],
                "terminal_tool": verdict["terminal_tool"],
                "has_content": verdict.get("has_content", False),
                "response_chars": len(response_text or ""),
                "tool_call_count": len(tool_calls or []),
            },
            turn_id=turn_id,
        )
        if not verdict["premature"]:
            return

        # Build the prefetch nudge and enqueue. The next user turn picks
        # it up via prefetch_context()'s ambient drain.
        nudge_kwargs = _iv_heuristics.make_completion_nudge_entry(
            turn_id=turn_id,
            response_excerpt=response_text or "",
        )
        import time as _time
        entry = AmbientPrefetchEntry(
            source=nudge_kwargs["source"],
            summary=nudge_kwargs["summary"],
            content=nudge_kwargs["content"],
            dedup_key=nudge_kwargs["dedup_key"],
            enqueued_at=_time.time(),
        )
        result = enqueue_ambient_prefetch(session_id, entry)
        _event_log.log_event(
            session_id,
            "inner_voice.completion_check_nudge_enqueued",
            {
                "source": entry.source,
                "summary": entry.summary,
                "queue_depth": result.get("queue_depth"),
                "deduped": result.get("deduped"),
            },
            turn_id=turn_id,
        )
        logger.info(
            "[inner_voice] completion check fired nudge on session=%s turn=%s "
            "(reason=%s)",
            session_id, turn_id, verdict["reason"],
        )
    except Exception as e:
        logger.warning(
            "[inner_voice] completion check failed (session=%s turn=%s): %s",
            session_id, turn_id, e,
        )


# ---------------------------------------------------------------------------
# Inner Voice (#345 Stage 2) — single-persona Brain 2 critique
# ---------------------------------------------------------------------------

async def _inner_voice_brain2_check(
    session_id: str,
    turn_id: str,
    turn_source: str,
    frozen_task_intent: str,
    response_text: str,
    tool_calls: list[dict],
    turn: SessionTurn,
) -> None:
    """Spawn Brain 2 against the just-finished ambient turn.

    Best-effort. Handles all errors internally so the chat path never sees
    a Brain 2 failure. Stage 3 fans out 3 personas concurrently
    (``completion_checker``, ``drift_detector``, ``continuation_drive``
    for the ``autonomy_default`` ensemble) and runs threshold-driven
    aggregation.

    Stage 4 chains a *consensus-termination* check onto the back of the
    ensemble: if the response carries ``SIGNAL:TASK_COMPLETE`` and the
    aggregated severity crosses the veto threshold, we inject a
    please-continue ambient. Four escape hatches keep the loop bounded
    (Brain 2 timeout, three-strike, max_nudges, hard_max_turns).

    SSE delivery is racy — the turn's events queue receives the events but
    the consumer may already have closed by the time the 5s critic call
    returns. The frontend treats SSE as a fast-path hint and SQLite
    (via `/api/inner_voice/critiques`) as the source of truth.
    """
    try:
        async def _emit_via_turn(event: str, data: dict[str, Any]) -> None:
            try:
                await _emit(turn, event, data)
            except Exception:
                # Queue may be GC'd or closed; we don't care for SSE delivery.
                pass

        critiques = await _iv_ensemble.run_post_loop_critique(
            session_id=session_id,
            turn_id=turn_id,
            turn_source=turn_source,
            frozen_task_intent=frozen_task_intent or "",
            response_text=response_text or "",
            tool_calls=list(tool_calls or []),
            emit_sse=_emit_via_turn,
        )

        # Re-derive ensemble_name + recompute aggregate locally. Both
        # functions are pure, deterministic, and cheap. We need them for
        # both the consensus_termination branch (TASK_COMPLETE veto) AND
        # the Stage 6 steer-dispatch branch (nudge_proposed retry).
        ensemble_name, _personas, _rationale = _iv_ensemble._select_ensemble_for_turn(
            turn_source, frozen_task_intent or "",
            tool_calls=list(tool_calls or []),
        )
        agg = _iv_ensemble._aggregate(list(critiques or []))

        # ── Path A: SIGNAL:TASK_COMPLETE → consensus_termination veto ──
        # Stage 4. Fires only on ambient turns (chat users stop themselves).
        consensus_handled = False
        if (
            _iv_consensus.has_task_complete_signal(response_text or "")
            and turn_source == "ambient"
        ):
            try:
                decision = await _iv_consensus.evaluate(
                    session_id=session_id,
                    turn_id=turn_id,
                    response_text=response_text or "",
                    critiques=list(critiques or []),
                    ensemble_name=ensemble_name,
                    hard_max_turns_hit=False,  # autonomy scheduler wires
                                               # this in a follow-up.
                )
                consensus_handled = True
            except Exception as e:
                logger.warning(
                    "[inner_voice] consensus_termination evaluate failed "
                    "(session=%s turn=%s): %s", session_id, turn_id, e,
                )
                return

        # ── Path B: nudge_proposed → Stage 6 steer dispatch ─────────────
        # Fires when the ensemble landed in [severity_threshold,
        # veto_severity_threshold) — flagged but below the consensus veto
        # floor. Without this branch, every chat critique with severity
        # 0.6-0.85 was log_only with no follow-up. Brain 2 had feedback,
        # Brain 1 never saw it.
        #
        # Skipped when consensus_termination already vetoed (it owns the
        # please-continue dispatch in that case — no double-fire).
        if not consensus_handled:
            steer_fired = await _maybe_dispatch_steer(
                session_id=session_id,
                turn_id=turn_id,
                turn_source=turn_source,
                response_text=response_text or "",
                critiques=list(critiques or []),
                agg=agg,
                ensemble_name=ensemble_name,
                emit_via_turn=_emit_via_turn,
            )
            if steer_fired:
                return  # we enqueued an ambient retry; don't fall through
            return  # nothing to do (agreement / log_only / no-op)

        # consensus_handled == True; fall through to the existing veto +
        # escalation handling below.
        decision = decision  # type: ignore[name-defined]  # set in branch A above

        # Surface the decision to the frontend immediately.
        try:
            await _emit_via_turn("inner_voice_consensus_decision", {
                "action": decision.action,
                "rationale": decision.rationale,
                "severity_max": decision.severity_max,
                "nudge_count": decision.nudge_count_after,
                "hatch_fired": decision.hatch_fired,
                "ensemble_name": ensemble_name,
            })
        except Exception:
            pass

        # Veto branch — enqueue please-continue ambient and record the
        # intervention against the strongest disagreeing critique.
        if decision.action == "vetoed" and decision.please_continue_kwargs:
            try:
                kw = decision.please_continue_kwargs
                entry = AmbientPrefetchEntry(
                    source=kw["source"],
                    summary=kw["summary"],
                    content=kw["content"],
                    dedup_key=kw["dedup_key"],
                    enqueued_at=time.time(),
                )
                result = enqueue_ambient_prefetch(session_id, entry)
                _event_log.log_event(
                    session_id,
                    "inner_voice.intervention_dispatched",
                    {
                        "kind": "continue",
                        "target_turn_id": turn_id,
                        "trigger": "consensus_termination_vetoed",
                        "severity": decision.severity_max,
                        "queue_depth": result.get("queue_depth"),
                        "deduped": result.get("deduped"),
                    },
                    turn_id=turn_id,
                )
                # Persist intervention row. action_taken='continue' on
                # the inner_voice_interventions table.
                try:
                    record_inner_voice_intervention(
                        session_id=session_id,
                        kind="continue",
                        target_turn_id=turn_id,
                        content=kw["content"],
                    )
                except Exception as e:
                    logger.warning(
                        "[inner_voice] consensus intervention persist "
                        "failed (session=%s turn=%s): %s",
                        session_id, turn_id, e,
                    )
                logger.info(
                    "[inner_voice] consensus VETO session=%s turn=%s "
                    "(severity=%.2f nudge=%d/%d)",
                    session_id, turn_id, decision.severity_max,
                    decision.nudge_count_after, _iv_consensus._max_nudges_per_session(),
                )
            except Exception as e:
                logger.warning(
                    "[inner_voice] consensus please-continue enqueue "
                    "failed (session=%s turn=%s): %s",
                    session_id, turn_id, e,
                )

        # Escalation branch — write the escalations.jsonl row.
        # Stage 4 keeps this a flat-file append; a follow-up task wires
        # it to the MCP backlog API (we don't have a direct in-process
        # call yet). The event log already carries the full payload.
        elif decision.action.startswith("escalated_") and decision.escalation_kwargs:
            try:
                from app.paths import LLOYD_HOME
                esc_dir = LLOYD_HOME / "_pipeline" / "inner_voice"
                esc_dir.mkdir(parents=True, exist_ok=True)
                esc_file = esc_dir / "escalations.jsonl"
                row = {
                    "ts": time.time(),
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "action": decision.action,
                    "hatch_fired": decision.hatch_fired,
                    "ensemble_name": ensemble_name,
                    "severity_max": decision.severity_max,
                    "nudge_count": decision.nudge_count_after,
                    "kwargs": decision.escalation_kwargs,
                }
                with esc_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                # Persist as intervention row too — kind='escalate'
                try:
                    record_inner_voice_intervention(
                        session_id=session_id,
                        kind="escalate",
                        target_turn_id=turn_id,
                        content=json.dumps(decision.escalation_kwargs),
                    )
                except Exception as e:
                    logger.warning(
                        "[inner_voice] escalation intervention persist "
                        "failed (session=%s turn=%s): %s",
                        session_id, turn_id, e,
                    )
                logger.warning(
                    "[inner_voice] consensus ESCALATION session=%s turn=%s "
                    "hatch=%s (nudge_count=%d)",
                    session_id, turn_id, decision.hatch_fired,
                    decision.nudge_count_after,
                )
            except Exception as e:
                logger.warning(
                    "[inner_voice] escalation write failed "
                    "(session=%s turn=%s): %s", session_id, turn_id, e,
                )
    except Exception as e:
        logger.warning(
            "[inner_voice] brain2 check failed (session=%s turn=%s): %s",
            session_id, turn_id, e,
        )


# ---------------------------------------------------------------------------
# Inner Voice (#345 Stage 6) — steer dispatch on nudge_proposed
# ---------------------------------------------------------------------------

async def _maybe_dispatch_steer(
    *,
    session_id: str,
    turn_id: str,
    turn_source: str,
    response_text: str,
    critiques: list,
    agg: dict[str, Any],
    ensemble_name: str,
    emit_via_turn,
) -> bool:
    """Dispatch a Stage 6 steer ambient when the post-loop ensemble lands
    on ``nudge_proposed`` (severity ≥ severity_threshold but <
    veto_severity_threshold).

    Returns True iff a steer was actually dispatched (so the caller can
    short-circuit other dispatch paths). Best-effort — every internal
    error is caught and logged.

    Behavior:
      1. Gate on `inner_voice.self_correct_on_nudge` (default true) AND
         the existing `inner_voice` opt-in flag for the session.
      2. Gate on the SHARED nudge budget (consensus_termination uses the
         same counter; cap is `max_nudges_per_session`). If the session
         has already burned its budget, log + return False.
      3. Build the steer ambient kwargs via `make_steer_ambient`.
      4. Enqueue an ambient prefetch (surfaces in next user turn's
         `<context>` if the ambient turn races a user message).
      5. Build + enqueue an actual ambient turn so Brain 1 picks up the
         retry immediately rather than waiting for the user.
      6. Record an `inner_voice_interventions` row (kind='steer') so the
         grading pass picks it up on the next outcome turn.
      7. Bump the shared nudge counter.
    """
    if agg.get("action_chosen") != "nudge_proposed":
        return False
    if not _iv_should_fire_on_turn(session_id, turn_source):
        return False
    if not _iv_consensus.is_self_correct_on_nudge_enabled():
        return False
    if not _iv_consensus.can_consume_nudge_budget(session_id):
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.steer_skipped",
                {
                    "reason": "nudge_budget_exhausted",
                    "ensemble_name": ensemble_name,
                    "agg": agg,
                },
                turn_id=turn_id,
            )
        except Exception:
            pass
        logger.info(
            "[inner_voice] steer skipped (nudge cap reached) session=%s turn=%s",
            session_id, turn_id,
        )
        return False

    severity_max = float(agg.get("severity_max", 0.0))
    reasons: list[tuple[str, str]] = []
    triggering_critique_id: int | None = None
    for c in critiques:
        if not getattr(c, "disagrees", False):
            continue
        if getattr(c, "error", None):
            continue
        reasons.append((c.persona, c.reason or ""))
        # Pick the highest-severity disagreer as the link target.
        if (
            triggering_critique_id is None
            or c.severity >= severity_max
        ):
            cid = getattr(c, "id", None) or getattr(c, "_db_id", None)
            if isinstance(cid, int):
                triggering_critique_id = cid
            severity_max = max(severity_max, c.severity)

    steer_kwargs = _iv_ensemble.make_steer_ambient(
        turn_id=turn_id,
        severity_max=severity_max,
        reasons=reasons,
        response_excerpt=response_text or "",
    )

    try:
        await emit_via_turn("inner_voice_steer_dispatched", {
            "ensemble_name": ensemble_name,
            "severity_max": severity_max,
            "reason_count": len(reasons),
            "agg_rationale": agg.get("rationale"),
            "turn_id": turn_id,
        })
    except Exception:
        pass

    # 1. Enqueue prefetch (so the next user message also sees the steer
    #    context even if the ambient retry races and gets preempted).
    try:
        prefetch_entry = AmbientPrefetchEntry(
            source=steer_kwargs["source"],
            summary=steer_kwargs["summary"],
            content=steer_kwargs["content"],
            dedup_key=steer_kwargs["dedup_key"],
            enqueued_at=time.time(),
        )
        enqueue_ambient_prefetch(session_id, prefetch_entry)
    except Exception as e:
        logger.warning(
            "[inner_voice] steer prefetch enqueue failed "
            "(session=%s turn=%s): %s", session_id, turn_id, e,
        )

    # 2. Enqueue an ambient turn so Brain 1 actually retries. The body is
    #    the steer content — `build_ambient_turn` wraps it in its standard
    #    `<ambient ...>` envelope which is fine; the inner-voice tag is
    #    explicit enough.
    try:
        ambient_turn = await build_ambient_turn(
            session_id=session_id,
            text=steer_kwargs["content"],
            dedup_key=steer_kwargs["dedup_key"],
            priority="notable",
            source=steer_kwargs["source"],
            summary=steer_kwargs["summary"],
        )
        result = await enqueue_ambient(session_id, ambient_turn)
    except Exception as e:
        logger.warning(
            "[inner_voice] steer ambient enqueue failed "
            "(session=%s turn=%s): %s", session_id, turn_id, e,
        )
        result = {}

    # 3. Persist as intervention row.
    try:
        record_inner_voice_intervention(
            session_id=session_id,
            kind="steer",
            target_turn_id=turn_id,
            content=steer_kwargs["content"],
            triggered_by_critique_id=triggering_critique_id,
        )
    except Exception as e:
        logger.warning(
            "[inner_voice] steer intervention persist failed "
            "(session=%s turn=%s): %s", session_id, turn_id, e,
        )

    # 4. Event log + bump counter.
    new_count = _iv_consensus.consume_nudge_budget(session_id)
    try:
        _event_log.log_event(
            session_id,
            "inner_voice.steer_dispatched",
            {
                "kind": "steer",
                "target_turn_id": turn_id,
                "ensemble_name": ensemble_name,
                "severity_max": severity_max,
                "reason_count": len(reasons),
                "nudge_count_after": new_count,
                "ambient_turn_id": ambient_turn.turn_id if isinstance(result, dict) and result else None,
                "queue_depth": result.get("queue_depth") if isinstance(result, dict) else None,
                "deduped": result.get("deduped") if isinstance(result, dict) else None,
                "agg_rationale": agg.get("rationale"),
            },
            turn_id=turn_id,
        )
    except Exception:
        pass

    logger.info(
        "[inner_voice] steer dispatched session=%s target_turn=%s "
        "(severity=%.2f reasons=%d nudge_count=%d)",
        session_id, turn_id, severity_max, len(reasons), new_count,
    )
    return True


# ---------------------------------------------------------------------------
# Inner Voice (#345 Stage 5) — grading pass
# ---------------------------------------------------------------------------

async def _inner_voice_grading_pass(
    session_id: str,
    outcome_turn_id: str,
    outcome_response_text: str,
    outcome_tool_calls: list[dict],
    frozen_task_intent: str,
) -> None:
    """Run the Stage 5 grading pass against the just-finished ambient turn.

    Spawned via `asyncio.ensure_future` from the ResultMessage branch in
    `_run_turn`, alongside the Stage 1 heuristic and Stage 2 Brain 2 check.
    Best-effort. The grading module catches every internal error; this
    wrapper exists only to bound logging.

    The pass looks up any interventions for `session_id` whose
    `outcome_turn_id IS NULL` and grades each via the `grader` persona
    against the just-finished outcome turn. Backfills `outcome_addressed`
    + `outcome_summary` on `inner_voice_interventions`.
    """
    try:
        summary = await _iv_grading.grade_outcome_turn(
            session_id=session_id,
            outcome_turn_id=outcome_turn_id,
            outcome_response_text=outcome_response_text or "",
            outcome_tool_calls=list(outcome_tool_calls or []),
            frozen_task_intent=frozen_task_intent or "",
        )
        if summary.get("graded", 0) > 0 or summary.get("errors", 0) > 0:
            logger.info(
                "[inner_voice] grading pass session=%s outcome_turn=%s "
                "graded=%d skipped=%d errors=%d candidates=%d",
                session_id, outcome_turn_id,
                summary.get("graded", 0),
                summary.get("skipped", 0),
                summary.get("errors", 0),
                summary.get("total_candidates", 0),
            )
    except Exception as e:
        logger.warning(
            "[inner_voice] grading pass failed (session=%s outcome_turn=%s): %s",
            session_id, outcome_turn_id, e,
        )


# ---------------------------------------------------------------------------
# Inner Voice (#345 Stage 3) — mid-turn drift detection
# ---------------------------------------------------------------------------

async def _inner_voice_mid_turn_drift_check(
    session_id: str,
    turn: SessionTurn,
    frozen_task_intent: str,
    partial_response: str,
    stream_position_chars: int,
    delta_index: int,
    cancel_event: asyncio.Event,
) -> None:
    """Fire `drift_detector` against a partial-response stream sample.

    Spawned as `asyncio.ensure_future` from inside the SDK message loop's
    `text_delta` handler. Best-effort; never raises into the consumer.

    On a veto-severity disagreement (`severity >= veto_severity_threshold`,
    `disagrees=True`, `error is None`):
      1. Log `inner_voice.cancel_event_fired` with stream position +
         partial-response delta count so meta-review can replay the
         interrupt sequence.
      2. Enqueue an ambient prefetch entry describing the drift verdict.
         The next user/ambient turn will surface it via `<context>`.
      3. Set `cancel_event` — the SDK loop's `if cancel_event.is_set()`
         check at the top of the message iterator breaks out on the next
         message event.

    On any other verdict (sub-veto disagreement, agreement, error): no
    intervention. The verdict is still persisted to SQLite + event log
    by `_iv_ensemble.run_mid_turn_drift_check` — Stage 3 wants the data
    even when it doesn't act.
    """
    try:
        # Skip if the turn was already cancelled by a previous mid-turn
        # check (or by the user via API). One cancel per turn is enough.
        if cancel_event.is_set():
            return

        async def _emit_via_turn(event: str, data: dict[str, Any]) -> None:
            try:
                await _emit(turn, event, data)
            except Exception:
                pass

        critique = await _iv_ensemble.run_mid_turn_drift_check(
            session_id=session_id,
            turn_id=turn.turn_id,
            frozen_task_intent=frozen_task_intent or "",
            partial_response=partial_response or "",
            stream_position_chars=stream_position_chars,
            delta_index=delta_index,
            emit_sse=_emit_via_turn,
        )
        if critique is None:
            return

        # Re-check cancel_event — another mid-turn fire may have set it
        # while this Brain 2 call was in flight. First-cancel-wins.
        if cancel_event.is_set():
            return

        if critique.action_taken != "interrupt":
            return

        # ── veto path: cancel + inject + log ─────────────────────────────
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.cancel_event_fired",
                {
                    "persona": critique.persona,
                    "severity": critique.severity,
                    "reason": critique.reason,
                    "stream_position_chars": stream_position_chars,
                    "delta_index": delta_index,
                    "partial_chars_at_cancel": len(partial_response or ""),
                    "trigger": "mid_turn_drift",
                },
                turn_id=turn.turn_id,
            )
        except Exception as e:
            logger.warning("cancel_event_fired log failed: %s", e)

        # Emit a chat-facing SSE event BEFORE setting cancel_event, so the
        # ChatPanel knows the cancel came from Inner Voice drift detection
        # and can render a banner with the reason. The main loop's cancel
        # handler emits `done(cancelled=true)` to close the stream; this
        # event provides the human-readable why.
        try:
            await _emit(turn, "inner_voice_drift_cancel", {
                "persona": critique.persona,
                "persona_version": critique.persona_version,
                "severity": critique.severity,
                "reason": critique.reason,
                "stream_position_chars": stream_position_chars,
                "partial_excerpt": (partial_response or "")[-300:],
                "turn_id": turn.turn_id,
            })
        except Exception:
            pass

        try:
            nudge_kwargs = _iv_ensemble.make_drift_cancel_ambient(
                turn_id=turn.turn_id,
                persona=critique.persona,
                severity=critique.severity,
                reason=critique.reason,
                partial_excerpt=partial_response or "",
            )
            entry = AmbientPrefetchEntry(
                source=nudge_kwargs["source"],
                summary=nudge_kwargs["summary"],
                content=nudge_kwargs["content"],
                dedup_key=nudge_kwargs["dedup_key"],
                enqueued_at=time.time(),
            )
            result = enqueue_ambient_prefetch(session_id, entry)
            _event_log.log_event(
                session_id,
                "inner_voice.intervention_dispatched",
                {
                    "kind": "interrupt",
                    "target_turn_id": turn.turn_id,
                    "persona": critique.persona,
                    "severity": critique.severity,
                    "queue_depth": result.get("queue_depth"),
                    "deduped": result.get("deduped"),
                    "trigger": "mid_turn_drift",
                },
                turn_id=turn.turn_id,
            )
            logger.info(
                "[inner_voice] mid-turn drift cancelled session=%s turn=%s "
                "(persona=%s severity=%.2f at %d chars)",
                session_id, turn.turn_id, critique.persona,
                critique.severity, stream_position_chars,
            )
        except Exception as e:
            logger.warning(
                "[inner_voice] mid-turn cancel-ambient enqueue failed "
                "(session=%s turn=%s): %s", session_id, turn.turn_id, e,
            )

        cancel_event.set()
    except Exception as e:
        logger.warning(
            "[inner_voice] mid-turn drift check failed (session=%s turn=%s): %s",
            session_id, turn.turn_id, e,
        )


# ---------------------------------------------------------------------------
# Subliminal-injection capture (#306)
# ---------------------------------------------------------------------------
# Three ephemeral injection sites send text to the SDK that the session JSON
# never sees:
#   1. prefetch_context() <context> block prepended to user message
#   2. build_ambient_turn() <ambient ...>…</ambient> envelope wrapping producer text
#   3. 20-turn <system-reminder> memory-preservation nudge
# These helpers extract the injected prefix from `prefetched_text` vs `text`
# so we can persist it as a role="subliminal" entry for UI visibility.

# Ordered classifiers: first match wins. `memory_nudge` check precedes
# `prefetch` so a nudge+prefetch combo surfaces as "memory_nudge" (the
# more notable framing) when it leads the prefix.
_SUBLIMINAL_KINDS = (
    ("memory_nudge",     "<system-reminder>"),
    ("ambient_envelope", "<ambient "),
    ("prefetch",         "<context>"),
)

# Tag → source-name map for the summary badge. Ordering matches rendering
# order in prefetch._format_context().
_SUBLIMINAL_SOURCE_TAGS = (
    ("ambient",  "<ambient-signals>"),
    ("skills",   "<skill "),
    ("backlog",  "<backlog-refs>"),
    ("facts",    "<facts>"),
    ("vault",    "<vault-context>"),
    ("sessions", "<recent-sessions>"),
    ("hint",     "<skill-hint>"),
)


def _extract_subliminal_prefix(prefetched_text: str, text: str) -> str:
    """Return the injected-only portion of `prefetched_text`, or "" if none.

    Two shapes are handled:
      - Prefetch/nudge path: `prefetched_text` ends with "\\n\\n" + text
        (see prefetch.prefetch_context and the memory-nudge branch).
      - Ambient envelope path: text is embedded inside an <ambient> wrapper
        (see build_ambient_turn). The whole prefetched_text is "injection".
    If `prefetched_text == text` no injection happened.
    """
    if prefetched_text == text:
        return ""
    suffix = "\n\n" + text
    if prefetched_text.endswith(suffix):
        prefix = prefetched_text[: -len(suffix)]
        return prefix if prefix.strip() else ""
    # Ambient envelope (or any other shape where text is not a clean suffix)
    return prefetched_text


def _classify_subliminal(prefix: str) -> str:
    """Return 'prefetch' | 'ambient_envelope' | 'memory_nudge' | 'other'."""
    lead = prefix.lstrip()
    for kind, marker in _SUBLIMINAL_KINDS:
        if lead.startswith(marker):
            return kind
    return "other"


def _detect_subliminal_sources(prefix: str) -> list[str]:
    """Return the list of detected source-sections in this injection."""
    return [name for name, marker in _SUBLIMINAL_SOURCE_TAGS if marker in prefix]


def _build_subliminal_entry(turn: SessionTurn, prefix: str, timestamp: str) -> dict:
    """Shape the subliminal message entry. Kept pure for testability."""
    return {
        "id": f"subl_{turn.turn_id}",
        "role": "subliminal",
        "content": [{"type": "text", "text": prefix}],
        "timestamp": timestamp,
        "subliminal": {
            "kind":     _classify_subliminal(prefix),
            "sources":  _detect_subliminal_sources(prefix),
            "chars":    len(prefix),
            "turn_id":  turn.turn_id,
        },
    }


# ---------------------------------------------------------------------------
# Turn execution — SDK loop + persistence. Pushes events into turn.events.
# ---------------------------------------------------------------------------

async def _emit(turn: SessionTurn, event: str, data: dict):
    """Push a named event onto the turn's broker queue.

    Auto-tags `source` (user/ambient/system) so SSE clients can style
    ambient output distinctly without needing to correlate with the
    session event.
    """
    data.setdefault("source", turn.source)
    data.setdefault("turn_id", turn.turn_id)
    await turn.events.put({"event": event, "data": data})


async def _run_turn(session_id: str, turn: SessionTurn, q: SessionQueue) -> None:
    """Run a single turn through the SDK, persisting as we go.

    Does not format SSE wire bytes — that's the subscriber's job. On
    return, the consumer pushes the sentinel `None` to close the stream.
    """
    payload = turn.payload
    text: str = payload["text"]
    prefetched_text: str = payload["prefetched_text"]
    model: str = payload["model"]
    options: ClaudeAgentOptions = payload["options"]
    meta_path = payload["meta_path"]
    resume_id = payload.get("resume_id")
    cancel_event = q.cancel_event

    t_query_start = time.perf_counter()

    # Surface a session event immediately so the client has the turn_id
    # and can correlate any later queue-state updates.
    await _emit(turn, "session", {"session_id": session_id, "turn_id": turn.turn_id})
    # Initial queue snapshot so the client doesn't need a separate poll.
    await turn.events.put({"event": "queue_state", "data": get_queue_state(session_id)})

    # Inner Voice (#345) — capture turn-entry events to the per-session
    # event log. Best-effort; event_log.log_event swallows exceptions.
    _event_log.log_event(session_id, "brain1.user_prompt_received", {
        "text": text,
        "source": turn.source,
        "model": model,
        "resume_id": resume_id,
        "has_prefetched_context": prefetched_text != text,
    }, turn_id=turn.turn_id)
    # `options_built` reaches here pre-baked by post_message_stream; we can
    # surface its salient fields without dumping callbacks/closures.
    _event_log.log_event(session_id, "brain1.options_built", {
        "model": model,
        "max_turns": getattr(options, "max_turns", None),
        "permission_mode": getattr(options, "permission_mode", None),
        "env_keys": sorted(list((options.env or {}).keys())) if hasattr(options, "env") else [],
        "mcp_server_keys": sorted(list((options.mcp_servers or {}).keys())) if hasattr(options, "mcp_servers") else [],
        "disallowed_tools": list(getattr(options, "disallowed_tools", []) or []),
    }, turn_id=turn.turn_id)

    full_response = ""
    accumulated_thinking = ""
    tool_calls_log: list[dict] = []
    tool_results_log: list[dict] = []
    persisted_tool_ids: set[str] = set()
    final_persisted = False
    first_event = True
    pending_pipeline_wires: dict[str, str] = {}
    stream_stats: dict[str, Any] = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_create": 0, "cache_read": 0,
        "cost_usd": None, "duration_ms": None, "num_turns": None,
        "model": model,
    }
    # stream_stats["input_tokens"] gets overwritten per turn, but ResultMessage later
    # replaces it with the cumulative total — we need both values.
    last_turn_input: int = 0

    # TTS-on-response: cumulative buffer across all text segments in this turn
    # (full_response resets on tool-use; this doesn't). When voice mode's TTS
    # toggle is on AND the source is "user", we fire one /v1/say as soon as
    # two sentence-terminators land. Ambient/autonomy turns are intentionally
    # silent — voice is a user-facing feature.
    tts_buffer = ""
    tts_spoken = False
    tts_should_speak = tts_is_enabled() and turn.source == "user"

    # Persist the user message up-front so transcripts stay coherent even
    # if the SDK crashes before emitting anything. Tag ambient-sourced
    # turns so the UI can render them differently from real user input.
    now_ts = datetime.now().isoformat()
    user_msg: dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "timestamp": now_ts,
    }
    if turn.source != "user":
        user_msg["source"] = turn.source
    await _append_messages(session_id, [user_msg])

    # #306: Capture ephemeral context injection (prefetch block, ambient
    # envelope, 20-turn memory nudge) as a role="subliminal" entry so the
    # chat UI can surface what the agent actually saw. Persisted right
    # after the user message to reflect "this is the extra context Lloyd
    # had at turn time." Scripts filter by role, so they skip it for free.
    subl_prefix = _extract_subliminal_prefix(prefetched_text, text)
    if subl_prefix:
        await _append_messages(
            session_id,
            [_build_subliminal_entry(turn, subl_prefix, now_ts)],
        )

    # ------------------------------------------------------------------
    # Local-model compaction: local endpoints (vLLM/llama-server) are
    # stateless, so we reconstruct the conversation from the persisted
    # session JSON each turn, truncate to fit the model's context window,
    # and render it in the model's native chat template. Cloud models
    # skip this block entirely — the Anthropic API manages context via
    # `resume`.
    # ------------------------------------------------------------------
    base_url = (options.env or {}).get("ANTHROPIC_BASE_URL", "")
    is_local = is_local_model(base_url)
    conv_text = ""

    if is_local:
        comp = load_and_compact_session(meta_path, model=model)
        conv_text = format_conversation_for_local(
            comp["history"],
            current_user_text=prefetched_text,
            model=model,
        )
        if comp["truncated"]:
            logger.info(
                "[compaction] %s: dropped %d tokens (%d → %d, window=%d, threshold=%d)",
                session_id,
                comp["tokens_before"] - comp["tokens_after"],
                comp["tokens_before"],
                comp["tokens_after"],
                comp["context_window"],
                comp["threshold"],
            )
        # Local endpoints don't support SDK resume; drive context purely
        # from our reconstructed prompt.
        options.resume = None

    # Inner Voice (#345) — sampled stream-event capture. K=50 deltas; each
    # firing captures position + delta length so we can reconstruct the
    # token-by-token cadence without filling the log with one event per
    # token. K stays as a constant here; promotes to config.yaml once
    # Stage 1 lands (`inner_voice.event_log.stream_sample_tokens`).
    _stream_sample_every = 50
    _stream_delta_count = 0

    # Inner Voice (#345 Stage 3) — mid-turn drift detection state. Fires
    # `drift_detector` every K accumulated chars on Inner-Voice-opted-in
    # sessions. Stage 5+ also fires on user turns when the session opted
    # into `inner_voice_evaluate_user_turns`. State is per-turn; trackers
    # reset implicitly on each new `_run_turn` invocation.
    _iv_mtd_enabled = (
        _iv_should_fire_on_turn(session_id, turn.source)
        and _iv_ensemble._is_mid_turn_drift_enabled()
    )
    _iv_mtd_cfg = _iv_ensemble.get_mid_turn_drift_config() if _iv_mtd_enabled else {}
    _iv_mtd_min_first = int(_iv_mtd_cfg.get("min_chars_before_first_check", 250))
    _iv_mtd_every = max(50, int(_iv_mtd_cfg.get("check_every_chars", 500)))
    _iv_mtd_max_checks = max(1, int(_iv_mtd_cfg.get("max_checks_per_turn", 4)))
    _iv_mtd_chars_at_last_check = 0
    _iv_mtd_checks_fired = 0

    # Inner Voice (#345 Stage 7) — intra-turn progress monitoring state.
    # Records tool-boundary events from the SDK PostToolUse hooks and
    # decides when to fire `tool_result_grader` / `progress_monitor`.
    # State is keyed by Lloyd session_id and reset every turn; the hook
    # callbacks read from it via the closure-bound session_id. Fires only
    # on opted-in sessions AND only when the master kill switch is on.
    _iv_intra_active = (
        _iv_should_fire_on_turn(session_id, turn.source)
        and _iv_intra_turn.is_enabled()
    )
    if _iv_intra_active:
        _iv_intra_turn.start_intra_turn(
            session_id,
            turn.turn_id,
            turn_source=turn.source,
            frozen_task_intent=text,
        )

    _event_log.log_event(session_id, "brain1.query_started", {
        "model": model,
        "is_local": is_local,
        "prompt_chars": len(conv_text if is_local else prefetched_text),
    }, turn_id=turn.turn_id)

    cancelled_mid_stream = False
    try:
        async for message in query(
            prompt=conv_text if is_local else prefetched_text,
            options=options,
        ):
            if cancel_event.is_set():
                logger.info(f"Session {session_id} turn {turn.turn_id} cancelled via API/IV")
                cancelled_mid_stream = True
                break

            if first_event:
                logger.info(
                    f"[TIMING] first SDK event after {time.perf_counter() - t_query_start:.3f}s "
                    f"(SDK+MCP startup)  resume={'yes' if resume_id else 'no'}"
                )
                first_event = False
            if isinstance(message, StreamEvent):
                evt = message.event
                etype = evt.get("type", "")
                if etype == "content_block_delta":
                    delta = evt.get("delta", {})
                    dtype = delta.get("type", "")
                    if dtype == "text_delta":
                        delta_text = delta.get("text", "")
                        if delta_text:
                            if not full_response:
                                logger.info(
                                    f"[TIMING] first text token after "
                                    f"{time.perf_counter() - t_query_start:.3f}s (model TTFT)"
                                )
                            full_response += delta_text
                            await _emit(turn, "text_delta", {"text": delta_text})
                            # Inner Voice — sampled stream event every K deltas.
                            _stream_delta_count += 1
                            if _stream_delta_count % _stream_sample_every == 0:
                                _event_log.log_event(session_id, "brain1.stream_event", {
                                    "kind": "text_delta",
                                    "delta_index": _stream_delta_count,
                                    "position_chars": len(full_response),
                                    "delta_chars": len(delta_text),
                                }, turn_id=turn.turn_id)
                            # Inner Voice (#345 Stage 3) — mid-turn drift
                            # detection. Fires `drift_detector` against the
                            # accumulated partial response on two surfaces:
                            #   1. First check at len >= min_chars_before_first_check
                            #   2. Subsequent checks every check_every_chars after
                            # The spawned task may set `cancel_event` and enqueue
                            # an ambient nudge if the verdict crosses the
                            # veto threshold. Fire-and-forget — text streaming
                            # must not block on a 5s Brain 2 call.
                            _iv_mtd_first_check_due = (
                                _iv_mtd_checks_fired == 0
                                and len(full_response) >= _iv_mtd_min_first
                            )
                            _iv_mtd_followup_due = (
                                _iv_mtd_checks_fired > 0
                                and (len(full_response) - _iv_mtd_chars_at_last_check) >= _iv_mtd_every
                            )
                            if (
                                _iv_mtd_enabled
                                and _iv_mtd_checks_fired < _iv_mtd_max_checks
                                and not cancel_event.is_set()
                                and (_iv_mtd_first_check_due or _iv_mtd_followup_due)
                            ):
                                _iv_mtd_chars_at_last_check = len(full_response)
                                _iv_mtd_checks_fired += 1
                                asyncio.ensure_future(_inner_voice_mid_turn_drift_check(
                                    session_id=session_id,
                                    turn=turn,
                                    frozen_task_intent=text,
                                    partial_response=full_response,
                                    stream_position_chars=len(full_response),
                                    delta_index=_stream_delta_count,
                                    cancel_event=cancel_event,
                                ))
                            # Inner Voice (#345 Stage 7) — also pulse the
                            # progress_monitor's time trigger from the
                            # text-delta path. Without this, a turn that
                            # stalls with NO tool calls (long thinking +
                            # long text emission) never fires the periodic
                            # synthesizer. Sample every K=50 deltas to
                            # avoid per-token check overhead — same cadence
                            # as the brain1.stream_event sampling above.
                            if (
                                _iv_intra_active
                                and not cancel_event.is_set()
                                and _stream_delta_count % _stream_sample_every == 0
                            ):
                                _iv_intra_turn._maybe_fire_progress_monitor(
                                    session_id, trigger_check=False,
                                )
                            # Fire TTS for the first two sentences as soon as
                            # they land. Re-check `tts_is_enabled()` per delta
                            # so toggling mid-turn takes effect.
                            if tts_should_speak and not tts_spoken and tts_is_enabled():
                                tts_buffer += delta_text
                                spoken_chunk = extract_first_two_sentences(tts_buffer)
                                if spoken_chunk:
                                    tts_spoken = True
                                    asyncio.create_task(speak_text(spoken_chunk))
                    elif dtype == "thinking_delta":
                        thinking_text = delta.get("thinking", "")
                        if thinking_text:
                            accumulated_thinking += thinking_text
                            await _emit(turn, "thinking_delta", {"text": thinking_text})
                elif etype == "message_start":
                    msg_usage = evt.get("message", {}).get("usage", {})
                    stream_stats["input_tokens"] = msg_usage.get("input_tokens", 0)
                    last_turn_input = stream_stats["input_tokens"]
                    stream_stats["cache_create"] = msg_usage.get("cache_creation_input_tokens", 0)
                    stream_stats["cache_read"] = msg_usage.get("cache_read_input_tokens", 0)
                elif etype == "message_delta":
                    stream_stats["output_tokens"] = evt.get("usage", {}).get("output_tokens", 0)
                continue

            if isinstance(message, SystemMessage):
                sdk_session = message.data.get("session_id")
                if sdk_session:
                    await mutate_session(session_id, lambda d: d.__setitem__("sdk_session_id", sdk_session))

            elif isinstance(message, AssistantMessage):
                has_tool_use = any(isinstance(b, ToolUseBlock) for b in message.content)
                if has_tool_use and full_response.strip():
                    seg_ts = datetime.now().isoformat()
                    seg_entry: dict = {
                        "id": uuid.uuid4().hex[:8],
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_response}],
                        "timestamp": seg_ts,
                    }
                    if accumulated_thinking:
                        seg_entry["reasoning"] = accumulated_thinking
                    await _append_messages(session_id, [seg_entry])
                    full_response = ""
                    accumulated_thinking = ""
                for block in message.content:
                    if isinstance(block, ThinkingBlock):
                        accumulated_thinking = block.thinking
                        await _emit(turn, "thinking_done", {"text": block.thinking})
                        # Inner Voice (#345) — full thinking text is the
                        # strongest anchoring vector for Brain 2; we capture
                        # it here for forensic replay but later stages must
                        # NOT include this in Brain 2's context (see
                        # `include_brain1_thinking: false` in config).
                        _event_log.log_event(session_id, "brain1.thinking_block_emitted", {
                            "thinking": block.thinking,
                            "chars": len(block.thinking),
                        }, turn_id=turn.turn_id)
                    elif isinstance(block, ToolUseBlock):
                        args_str = json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                        tool_calls_log.append({
                            "id": block.id, "call_id": block.id, "type": "function",
                            "function": {"name": block.name, "arguments": args_str},
                        })
                        await _emit(turn, "tool_start", {
                            "call_id": block.id, "name": block.name,
                            "args": args_str, "context_tokens": last_turn_input,
                        })
                        # Inner Voice (#345) — pre-PreToolUse capture. In
                        # Stage 1 the deny-rule evaluation will land between
                        # this event and the tool's actual execution.
                        _event_log.log_event(session_id, "brain1.tool_call_proposed", {
                            "tool_call_id": block.id,
                            "name": block.name,
                            "args": args_str,
                            "context_tokens": last_turn_input,
                        }, turn_id=turn.turn_id)
                        if block.name.endswith("pipeline_dispatch"):
                            pending_pipeline_wires[block.id] = session_id
                            logger.info(f"Tracking pipeline_dispatch call {block.id!r} for session {session_id}")

            elif isinstance(message, UserMessage):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        result_str = ""
                        if hasattr(block, "content"):
                            if isinstance(block.content, str):
                                result_str = block.content
                            elif isinstance(block.content, list):
                                result_str = " ".join(
                                    getattr(c, "text", str(c)) for c in block.content
                                )
                        if len(result_str) > 2000:
                            result_str = result_str[:2000] + "...(truncated)"
                        call_id = getattr(block, 'tool_use_id', '')
                        tool_results_log.append({"call_id": call_id, "result": result_str})
                        await _emit(turn, "tool_complete", {
                            "call_id": call_id, "name": "", "result": result_str,
                        })
                        # Inner Voice (#345) — tool result captured to the
                        # event log. Truncated to 2KB inline (same as the
                        # SSE wire path); full result lives in the chat
                        # transcript for replay.
                        _event_log.log_event(session_id, "brain1.tool_result_received", {
                            "tool_call_id": call_id,
                            "result": result_str,
                            "result_chars": len(result_str),
                        }, turn_id=turn.turn_id)
                        if call_id in pending_pipeline_wires:
                            req_session = pending_pipeline_wires.pop(call_id)
                            try:
                                # result_str may be JSON or str() of a dict (single-quote repr)
                                import ast as _ast
                                try:
                                    res_data = json.loads(result_str)
                                except json.JSONDecodeError:
                                    res_data = _ast.literal_eval(result_str)
                                # Unwrap if result_str was str({'type':'text','text':'{...}'})
                                if isinstance(res_data, dict) and "text" in res_data:
                                    inner = res_data["text"]
                                    if isinstance(inner, str):
                                        try:
                                            res_data = json.loads(inner)
                                        except Exception:
                                            pass
                                run_id = res_data.get("run_id")
                                if run_id:
                                    run_path = PIPELINE_RUNS_DIR / f"{run_id}.json"
                                    if run_path.exists():
                                        run_json = json.loads(run_path.read_text(encoding="utf-8"))
                                        run_json["requester_session_id"] = req_session
                                        run_path.write_text(json.dumps(run_json, indent=2), encoding="utf-8")
                                        logger.info(f"Linked pipeline run #{run_id} → session {req_session}")
                                    else:
                                        logger.warning(f"Pipeline run file {run_id}.json not found for wiring")
                                else:
                                    logger.warning(f"No run_id in pipeline_dispatch result: {result_str[:200]}")
                            except Exception as _we:
                                logger.warning(f"Failed to wire pipeline session: {_we} | result={result_str[:200]}")

                        # Eager per-pair persistence so a mid-stream error doesn't
                        # lose tool history.
                        tc = next((t for t in tool_calls_log if t["call_id"] == call_id), None)
                        if tc and call_id not in persisted_tool_ids:
                            persisted_tool_ids.add(call_id)
                            pair_ts = datetime.now().isoformat()
                            await _append_messages(session_id, [
                                {
                                    "id": f"msg_{call_id}_tc",
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": ""}],
                                    "tool_calls": [tc],
                                    "timestamp": pair_ts,
                                },
                                {
                                    "id": f"msg_{call_id}_result",
                                    "role": "tool",
                                    "content": [{"type": "text", "text": result_str}],
                                    "tool_call_id": call_id,
                                    "timestamp": pair_ts,
                                },
                            ])

            elif isinstance(message, ResultMessage):
                try:
                    usage = getattr(message, "usage", None) or {}
                    usage_store.record_usage(
                        session_id=session_id,
                        model=model,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cache_create=usage.get("cache_creation_input_tokens", 0),
                        cache_read=usage.get("cache_read_input_tokens", 0),
                        cost_usd=getattr(message, "total_cost_usd", None),
                        duration_ms=getattr(message, "duration_ms", None),
                        duration_api_ms=getattr(message, "duration_api_ms", None),
                        num_turns=getattr(message, "num_turns", None),
                    )
                except Exception as ue:
                    logger.warning(f"Failed to record usage: {ue}")

                # Inner Voice (#345) — final ResultMessage capture. Includes
                # usage + duration + stop_reason. Brain 2 ensemble dispatch
                # in Stage 2+ keys off this event (post-loop seam).
                _event_log.log_event(session_id, "brain1.result_message", {
                    "usage": dict(usage) if usage else {},
                    "stop_reason": getattr(message, "stop_reason", None),
                    "duration_ms": getattr(message, "duration_ms", None),
                    "duration_api_ms": getattr(message, "duration_api_ms", None),
                    "num_turns": getattr(message, "num_turns", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                    "response_chars": len(full_response),
                    "had_tool_calls": bool(tool_calls_log),
                }, turn_id=turn.turn_id)

                if message.session_id:
                    try:
                        await mutate_session(
                            session_id,
                            lambda d, sid=message.session_id: d.__setitem__("sdk_session_id", sid),
                        )
                    except Exception as se:
                        logger.warning(f"Failed to save sdk_session_id: {se}")

                # Persisted text reflects only the post-last-tool segment;
                # earlier segments were already flushed mid-stream. The done
                # payload still surfaces message.result for callers that want
                # the full concatenated turn.
                result_text = full_response
                done_text = full_response
                if hasattr(message, "result") and message.result:
                    done_text = message.result

                end_ts = datetime.now().isoformat()
                tail: list[dict] = []
                results_by_id = {r["call_id"]: r["result"] for r in tool_results_log}
                for tc in tool_calls_log:
                    cid = tc["call_id"]
                    if cid not in persisted_tool_ids:
                        persisted_tool_ids.add(cid)
                        tail.append({"id": f"msg_{cid}_tc", "role": "assistant",
                                     "content": [{"type": "text", "text": ""}],
                                     "tool_calls": [tc], "timestamp": end_ts})
                        tail.append({"id": f"msg_{cid}_result", "role": "tool",
                                     "content": [{"type": "text", "text": results_by_id.get(cid, "")}],
                                     "tool_call_id": cid, "timestamp": end_ts})
                stream_stats.update({
                    "input_tokens":  usage.get("input_tokens", 0) or stream_stats["input_tokens"],
                    "output_tokens": usage.get("output_tokens", 0) or stream_stats["output_tokens"],
                    "cache_create":  usage.get("cache_creation_input_tokens", 0),
                    "cache_read":    usage.get("cache_read_input_tokens", 0),
                    "cost_usd":      getattr(message, "total_cost_usd", None),
                    "duration_ms":   getattr(message, "duration_ms", None),
                    "num_turns":     getattr(message, "num_turns", None),
                    "peak_input_tokens": last_turn_input,
                })
                stats_dict = stream_stats

                # Ambient turns only: if the agent called `ambient_decide`
                # mid-turn to opt out of surfacing, replace the assistant
                # text with a muted breadcrumb so the transcript shows the
                # decision without user-visible noise. (#295 Slice 4)
                ambient_decision = None
                if turn.source == "ambient":
                    ambient_decision = take_ambient_decision(session_id)

                if ambient_decision and not ambient_decision.get("surface", True):
                    # Silent path: muted system breadcrumb, no assistant message.
                    silent_entry = {
                        "id": uuid.uuid4().hex[:8],
                        "role": "system",
                        "source": "ambient",
                        "silent": True,
                        "content": [{"type": "text",
                                     "text": f"(ambient: Lloyd reviewed and chose not to surface — {ambient_decision.get('reasoning', 'no reason given')[:200]})"}],
                        "timestamp": end_ts,
                        "stats": stats_dict,
                    }
                    tail.append(silent_entry)
                    if tail:
                        await _append_messages(session_id, tail)
                    final_persisted = True
                    asyncio.ensure_future(_post_session_capture(session_id))
                    await _emit(turn, "ambient_silent", {
                        "session_id": session_id,
                        "reasoning": ambient_decision.get("reasoning", ""),
                    })
                    await _emit(turn, "done", {
                        "response": "",
                        "session_id": session_id,
                        "stats": stats_dict,
                        "ambient_silent": True,
                    })
                else:
                    if result_text.strip():
                        msg_entry: dict = {"id": uuid.uuid4().hex[:8], "role": "assistant",
                                     "content": [{"type": "text", "text": result_text}],
                                     "timestamp": end_ts, "stats": stats_dict}
                        if accumulated_thinking:
                            msg_entry["reasoning"] = accumulated_thinking
                        if turn.source != "user":
                            msg_entry["source"] = turn.source
                        tail.append(msg_entry)
                    if tail:
                        await _append_messages(session_id, tail)
                    final_persisted = True

                    asyncio.ensure_future(_post_session_capture(session_id))
                    asyncio.ensure_future(_maybe_extract_focus(session_id))

                    # Inner Voice (#345 Stage 1–5) — post-loop checks.
                    #
                    # Stage 1's `_inner_voice_completion_check` (the regex
                    # heuristic for premature SIGNAL:TASK_COMPLETE) stays
                    # ambient-only — chat users stop themselves, so the
                    # check is irrelevant on user turns and would just
                    # noise the event log.
                    #
                    # Stage 2+5 — Brain 2 ensemble + grading pass — fire
                    # on EITHER ambient turns OR user turns when the
                    # session opted into `inner_voice_evaluate_user_turns`.
                    # That flag is what makes the Inner Voice tab's chat
                    # actually surface Brain 2 verdicts in real time.
                    iv_should_fire = _iv_should_fire_on_turn(session_id, turn.source)
                    if turn.source == "ambient" and _session_inner_voice_enabled(session_id):
                        asyncio.ensure_future(_inner_voice_completion_check(
                            session_id=session_id,
                            turn_id=turn.turn_id,
                            response_text=done_text,
                            tool_calls=list(tool_calls_log),
                        ))
                    if iv_should_fire:
                        asyncio.ensure_future(_inner_voice_brain2_check(
                            session_id=session_id,
                            turn_id=turn.turn_id,
                            turn_source=turn.source,
                            frozen_task_intent=text,
                            response_text=done_text,
                            tool_calls=list(tool_calls_log),
                            turn=turn,
                        ))
                        # Stage 5 — grading pass against any ungraded
                        # interventions from PRIOR turns. The brain2 check
                        # above may write a new intervention for THIS turn;
                        # the grading pass filters it out via
                        # `exclude_target_turn_id` so the new row stays in
                        # the queue for the next outcome turn.
                        asyncio.ensure_future(_inner_voice_grading_pass(
                            session_id=session_id,
                            outcome_turn_id=turn.turn_id,
                            outcome_response_text=done_text,
                            outcome_tool_calls=list(tool_calls_log),
                            frozen_task_intent=text,
                        ))

                    # End-of-turn TTS fallback: response was shorter than two
                    # sentences (e.g. "Done.") so the mid-stream trigger never
                    # fired. Speak whatever we have.
                    if tts_should_speak and not tts_spoken and tts_is_enabled() and tts_buffer.strip():
                        tts_spoken = True
                        asyncio.create_task(speak_text(tts_buffer))

                    done_payload: dict = {'response': done_text, 'session_id': session_id, 'stats': stats_dict}
                    if accumulated_thinking:
                        done_payload['reasoning'] = accumulated_thinking
                    await _emit(turn, "done", done_payload)

        # Clean cancel exit (no exception, just a `break` from the
        # `if cancel_event.is_set()` check at the top of the loop).
        # Without this block the SSE stream hangs forever — the chat UI
        # waits for a `done` event that never arrives, the partial
        # response Brain 1 had accumulated never gets persisted, and the
        # input field stays disabled.
        #
        # Fires for BOTH API cancels (sessions/cancel) AND Inner Voice
        # mid-turn drift cancels (the IV check sets cancel_event after
        # logging cancel_event_fired + intervention_dispatched).
        if cancelled_mid_stream and not final_persisted:
            cancel_ts = datetime.now().isoformat()
            tail = []
            results_by_id = {r["call_id"]: r["result"] for r in tool_results_log}
            for tc in tool_calls_log:
                cid = tc["call_id"]
                if cid not in persisted_tool_ids:
                    tail.append({
                        "id": f"msg_{cid}_tc", "role": "assistant",
                        "content": [{"type": "text", "text": ""}],
                        "tool_calls": [tc], "timestamp": cancel_ts,
                    })
                    tail.append({
                        "id": f"msg_{cid}_result", "role": "tool",
                        "content": [{"type": "text", "text": results_by_id.get(cid, "")}],
                        "tool_call_id": cid, "timestamp": cancel_ts,
                    })
            stream_stats["peak_input_tokens"] = last_turn_input
            if full_response.strip():
                cancel_msg: dict = {
                    "id": uuid.uuid4().hex[:8], "role": "assistant",
                    "content": [{"type": "text", "text": full_response}],
                    "timestamp": cancel_ts, "stats": stream_stats,
                    "cancelled": True,
                }
                if accumulated_thinking:
                    cancel_msg["reasoning"] = accumulated_thinking
                tail.append(cancel_msg)
            if tail:
                await _append_messages(session_id, tail)
            try:
                if stream_stats["input_tokens"] or stream_stats["output_tokens"]:
                    usage_store.record_usage(
                        session_id=session_id,
                        model=model,
                        input_tokens=stream_stats["input_tokens"],
                        output_tokens=stream_stats["output_tokens"],
                        cache_create=stream_stats["cache_create"],
                        cache_read=stream_stats["cache_read"],
                    )
            except Exception as ue:
                logger.warning(f"Failed to record usage on cancel: {ue}")
            final_persisted = True
            await _emit(turn, "done", {
                "response": full_response,
                "session_id": session_id,
                "stats": stream_stats,
                "cancelled": True,
            })

    except Exception as e:
        if not final_persisted:
            if full_response or tool_calls_log:
                # SDK exit-code-1 after completion, or mid-stream error with content.
                logger.warning(f"Turn {turn.turn_id} ended without ResultMessage (content delivered): {e}")
                err_ts = datetime.now().isoformat()
                tail = []
                results_by_id = {r["call_id"]: r["result"] for r in tool_results_log}
                for tc in tool_calls_log:
                    cid = tc["call_id"]
                    if cid not in persisted_tool_ids:
                        tail.append({"id": f"msg_{cid}_tc", "role": "assistant",
                                     "content": [{"type": "text", "text": ""}],
                                     "tool_calls": [tc], "timestamp": err_ts})
                        tail.append({"id": f"msg_{cid}_result", "role": "tool",
                                     "content": [{"type": "text", "text": results_by_id.get(cid, "")}],
                                     "tool_call_id": cid, "timestamp": err_ts})
                stream_stats["peak_input_tokens"] = last_turn_input
                if full_response.strip():
                    err_msg_entry: dict = {"id": uuid.uuid4().hex[:8], "role": "assistant",
                                 "content": [{"type": "text", "text": full_response}],
                                 "timestamp": err_ts, "stats": stream_stats}
                    if accumulated_thinking:
                        err_msg_entry["reasoning"] = accumulated_thinking
                    tail.append(err_msg_entry)
                # Same TTS fallback as the success path: SDK died mid-stream
                # with content but our two-sentence trigger never fired.
                if tts_should_speak and not tts_spoken and tts_is_enabled() and tts_buffer.strip():
                    tts_spoken = True
                    asyncio.create_task(speak_text(tts_buffer))
                if tail:
                    await _append_messages(session_id, tail)
                try:
                    if stream_stats["input_tokens"] or stream_stats["output_tokens"]:
                        usage_store.record_usage(
                            session_id=session_id,
                            model=model,
                            input_tokens=stream_stats["input_tokens"],
                            output_tokens=stream_stats["output_tokens"],
                            cache_create=stream_stats["cache_create"],
                            cache_read=stream_stats["cache_read"],
                        )
                except Exception as ue:
                    logger.warning(f"Failed to record usage (exception path): {ue}")
                await _emit(turn, "done", {
                    'response': full_response, 'session_id': session_id, 'stats': stream_stats,
                })
            else:
                logger.error(f"Turn {turn.turn_id} SDK error: {e}")
                await _emit(turn, "error", {"detail": str(e)})
        else:
            # Normal: SDK exited with code 1 after clean completion — ignore
            logger.debug(f"Post-completion SDK exit (ignored): {e}")



# ---------------------------------------------------------------------------
# Per-session consumer. Drains q.pending serially; lazily spawned.
# ---------------------------------------------------------------------------

async def _session_consumer(session_id: str) -> None:
    """Drain the queue for a single session. Exits when queue empties.

    Pop order: user tier first, then ambient. This plus the preempt set
    at enqueue time (see `enqueue_turn`) is how "user always wins" is
    enforced — the currently running ambient gets its cancel_event fired,
    breaks out of its SDK loop, and the consumer immediately pops the
    newly-enqueued user turn.
    """
    q = _session_queues.get(session_id)
    if q is None:
        return
    try:
        while True:
            async with q.lock:
                if q.pending_user:
                    turn = q.pending_user.popleft()
                elif q.pending_ambient:
                    turn = q.pending_ambient.popleft()
                else:
                    q.current = None
                    q.consumer_task = None
                    return
                q.current = turn
                q.cancel_event = asyncio.Event()
            turn.started_at = datetime.now()
            try:
                await _run_turn(session_id, turn, q)
            except asyncio.CancelledError:
                # Don't swallow cancel — propagate to drop the consumer.
                await turn.events.put(None)
                turn.done.set()
                raise
            except Exception:
                logger.exception(f"Turn {turn.turn_id} raised from _run_turn")
                try:
                    await turn.events.put({"event": "error", "data": {"detail": "internal error"}})
                except Exception:
                    pass
            finally:
                # If an ambient turn was preempted by an incoming user turn,
                # drop a breadcrumb in the session transcript so history
                # shows that context-injection was interrupted.
                if turn.source == "ambient" and turn.preempted:
                    try:
                        await _append_messages(session_id, [{
                            "id": uuid.uuid4().hex[:8],
                            "role": "system",
                            "source": "ambient",
                            "canceled": True,
                            "content": [{"type": "text",
                                         "text": "(ambient turn interrupted by user input)"}],
                            "timestamp": datetime.now().isoformat(),
                        }])
                    except Exception as ce:
                        logger.warning(f"Failed to write ambient-cancel marker: {ce}")
                # Inner Voice (#345 Stage 7) — clear intra-turn state.
                # Idempotent + turn_id-checked so a stale finally doesn't
                # clobber a fresh turn's state.
                try:
                    _iv_intra_turn.end_intra_turn(session_id, turn.turn_id)
                except Exception as ce:
                    logger.warning(f"end_intra_turn failed: {ce}")
                # Sentinel: tells the SSE subscriber to close cleanly.
                try:
                    await turn.events.put(None)
                except Exception:
                    pass
                turn.done.set()
    except asyncio.CancelledError:
        logger.warning(f"Consumer for session {session_id} cancelled")
        async with q.lock:
            q.current = None
            q.consumer_task = None
        raise
    except Exception:
        logger.exception(f"Consumer for session {session_id} crashed")
        async with q.lock:
            q.current = None
            q.consumer_task = None


# ---------------------------------------------------------------------------
# SSE subscriber. Forwards events from turn.events to the wire.
# ---------------------------------------------------------------------------

async def _turn_sse_generator(turn: SessionTurn):
    """Yield SSE-formatted bytes for a single turn.

    Client disconnect cancels this generator; the consumer keeps running
    and persistence still lands — events pile up in turn.events until the
    turn finishes. That buffer is bounded by turn length, not by client.
    """
    try:
        while True:
            evt = await turn.events.get()
            if evt is None:
                break
            yield f"event: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"
    except asyncio.CancelledError:
        logger.info(f"Client disconnected from turn {turn.turn_id} (consumer continues)")
        raise


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

@router.post("/api/message/stream")
async def post_message_stream(request: Request):
    """SSE endpoint. Enqueues a user turn; consumer streams events back."""
    data = await request.json()
    text = data.get("text", "").strip()
    session_id = data.get("session_id", "")
    model_override = data.get("model", "")
    think_level = data.get("think", "")  # off/low/medium/high or empty

    if not text:
        raise HTTPException(status_code=400, detail="Message text required")

    model = model_override or ""
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            session_data = json.loads(meta_path.read_text())
            if not model:
                model = session_data.get("model", "")

    if not model:
        model = CONFIG.get("model", {}).get("default", "")

    model = _resolve_model_name(model)
    model_env = _get_model_env(model)

    if not session_id:
        session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    t0 = time.perf_counter()

    system_prompt = build_system_prompt()
    t_prompt = time.perf_counter()

    prefetched_text = prefetch_context(text, session_id=session_id)
    t_prefetch = time.perf_counter()

    meta_path = SESSIONS_DIR / f"{session_id}.json"
    resume_id = None
    session_turn_count = 0
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            session_turn_count = sum(1 for m in existing.get("messages", []) if m.get("role") == "user")
            session_model = _resolve_model_name(existing.get("model", ""))
            # Thinking block signatures are tied to the originating endpoint,
            # so crossing local↔remote boundaries causes a 400.
            if _model_base_url(session_model) == _model_base_url(model):
                resume_id = existing.get("sdk_session_id")
        except Exception:
            pass

    # Memory preservation nudge: every 20 turns, remind agent to capture durable context
    if session_turn_count > 0 and session_turn_count % 20 == 0:
        nudge = (
            f"<system-reminder>This session has {session_turn_count} turns. "
            "If any important decisions, preferences, system changes, or facts from "
            "earlier in this conversation haven't been captured yet, consider calling "
            "memory_add or fact_add now before context compaction loses them."
            "</system-reminder>\n"
        )
        prefetched_text = nudge + prefetched_text

    extra_disallowed: list[str] = data.get("extra_disallowed", [])
    permission_mode: str = (
        data.get("permission_mode")
        or CONFIG.get("agent", {}).get("permission_mode", "bypassPermissions")
    )

    # Inner Voice (#345 Stage 1): wire PreToolUse hook only on opted-in
    # sessions. Existing Chat-tab sessions have `inner_voice: false` and
    # see no behavior change.
    iv_enabled = _session_inner_voice_enabled(session_id)
    iv_hooks = _inner_voice_hooks_dict(session_id) if iv_enabled else None

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=permission_mode,
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools() + extra_disallowed,
        env=model_env,
        effort=_resolve_effort(model, think_level),
        thinking=_resolve_thinking(model),
        resume=resume_id,
        include_partial_messages=True,
        hooks=iv_hooks,
    )

    await _save_session_meta(session_id, model, preview=text)

    logger.info(
        f"[TIMING] pre-enqueue overhead: prompt={t_prompt - t0:.3f}s  "
        f"prefetch={t_prefetch - t_prompt:.3f}s  "
        f"total={time.perf_counter() - t0:.3f}s  resume={'yes' if resume_id else 'no'}"
    )

    turn = SessionTurn(
        turn_id=uuid.uuid4().hex[:12],
        source="user",
        payload={
            "text": text,
            "prefetched_text": prefetched_text,
            "model": model,
            "options": options,
            "meta_path": meta_path,
            "resume_id": resume_id,
        },
        enqueued_at=datetime.now(),
    )

    await enqueue_turn(
        session_id,
        turn,
        consumer_factory=lambda: _session_consumer(session_id),
    )
    # Ambient producers (autonomy, cron, pipelines) target "the user's
    # active session" — record this one so they can resolve without
    # scanning disk. #295.
    set_last_user_session(session_id)

    return StreamingResponse(_turn_sse_generator(turn), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Ambient turn builder — used by /api/sessions/{id}/inject (see sessions.py)
# ---------------------------------------------------------------------------

async def build_ambient_turn(
    session_id: str,
    text: str,
    dedup_key: str | None = None,
    priority: str = "notable",
    source: str = "producer",
    summary: str = "",
) -> SessionTurn:
    """Assemble a `SessionTurn` suitable for ambient injection.

    Reuses the session's existing model + system_prompt and skips the
    /stream handler's prefetch (ambient producers are expected to send
    already-contextualized content). Raises HTTPException(404) if the
    session doesn't exist.

    Wraps `text` in an `<ambient ...>` envelope plus an explicit hint that
    the agent may call `ambient_decide(session_id=..., surface=False)` to
    stay silent. Producers set `priority` to `notable` (default) or
    `urgent` to change the urgency framing.

    `dedup_key` (optional): if provided and another queued ambient has
    the same key, the older one is dropped when this turn enqueues.
    """
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    existing = json.loads(meta_path.read_text())
    model = _resolve_model_name(existing.get("model", "") or CONFIG.get("model", {}).get("default", ""))
    model_env = _get_model_env(model)

    resume_id = None
    try:
        session_model = _resolve_model_name(existing.get("model", ""))
        if _model_base_url(session_model) == _model_base_url(model):
            resume_id = existing.get("sdk_session_id")
    except Exception:
        pass

    system_prompt = build_system_prompt()

    # Inner Voice (#345 Stage 1): same opt-in gate as the user-stream path.
    # Ambient turns on Inner Voice sessions are the highest-value targets
    # for safety hooks — autonomy fires there with no human watching.
    iv_enabled = _session_inner_voice_enabled(session_id)
    iv_hooks = _inner_voice_hooks_dict(session_id) if iv_enabled else None

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=CONFIG.get("agent", {}).get("permission_mode", "bypassPermissions"),
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools(),
        env=model_env,
        effort=_resolve_effort(model),
        thinking=_resolve_thinking(model),
        resume=resume_id,
        include_partial_messages=True,
        hooks=iv_hooks,
    )

    # Envelope the raw producer text so the agent sees framing + knows it
    # can opt out. The session_id is pre-filled so the agent can just copy
    # it into the ambient_decide call. Urgent nudges get a stronger verb.
    urge = "surface this now if the user should know" if priority == "urgent" \
           else "consider whether to mention this to the user"
    prefetched_text = (
        f'<ambient priority="{priority}" source="{source}" session_id="{session_id}">\n'
        f'{text}\n'
        f'</ambient>\n\n'
        f'This is a background signal from `{source}`. You were not asked a question — '
        f'{urge}. If it is not worth interrupting them, call '
        f'`ambient_decide(session_id="{session_id}", surface=false, reasoning="...")` '
        f'and stop. If it is worth surfacing, reply briefly and naturally — the user will '
        f'see your message as a normal assistant turn.'
    )

    payload: dict[str, Any] = {
        "text": text,
        "prefetched_text": prefetched_text,
        "model": model,
        "options": options,
        "meta_path": meta_path,
        "resume_id": resume_id,
        "priority": priority,
        "producer_source": source,
        "summary": summary,
    }
    if dedup_key:
        payload["dedup_key"] = dedup_key
    return SessionTurn(
        turn_id=uuid.uuid4().hex[:12],
        source="ambient",
        payload=payload,
        enqueued_at=datetime.now(),
    )


async def enqueue_ambient(session_id: str, turn: SessionTurn) -> dict[str, Any]:
    """Enqueue a pre-built ambient turn. Thin wrapper around enqueue_turn
    that binds the consumer factory so sessions.py doesn't need to import
    the consumer coroutine directly.
    """
    return await enqueue_turn(
        session_id,
        turn,
        consumer_factory=lambda: _session_consumer(session_id),
    )


@router.post("/api/message")
async def post_message(request: Request):
    """Synchronous message endpoint — collects full response then returns.

    Not routed through the session queue (task #296 Phase 1 covers the
    streaming path only). Concurrent sync POSTs to the same session
    still race the SDK subprocess; callers should prefer /stream.
    """
    data = await request.json()
    text = data.get("text", "").strip()
    session_id = data.get("session_id", "")
    model_override = data.get("model", "")

    if not text:
        raise HTTPException(status_code=400, detail="Message text required")

    model = model_override or ""
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            session_data = json.loads(meta_path.read_text())
            if not model:
                model = session_data.get("model", "")

    if not model:
        model = CONFIG.get("model", {}).get("default", "")

    model = _resolve_model_name(model)
    model_env = _get_model_env(model)

    if not session_id:
        session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    system_prompt = build_system_prompt()
    prefetched_text = prefetch_context(text, session_id=session_id)

    resume_id = None
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            session_model = _resolve_model_name(existing.get("model", ""))
            if _model_base_url(session_model) == _model_base_url(model):
                resume_id = existing.get("sdk_session_id")
        except Exception:
            pass

    sync_extra_disallowed: list[str] = data.get("extra_disallowed", [])
    sync_permission_mode: str = (
        data.get("permission_mode")
        or CONFIG.get("agent", {}).get("permission_mode", "bypassPermissions")
    )

    # Inner Voice (#345 Stage 1): same opt-in gate. The sync endpoint is
    # rarely used (clients prefer /stream) but symmetry matters — a
    # dangerous Bash via /api/message bypasses the safety net otherwise.
    iv_enabled = _session_inner_voice_enabled(session_id)
    iv_hooks = _inner_voice_hooks_dict(session_id) if iv_enabled else None

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=sync_permission_mode,
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools() + sync_extra_disallowed,
        env=model_env,
        effort=_resolve_effort(model),
        thinking=_resolve_thinking(model),
        resume=resume_id,
        hooks=iv_hooks,
    )

    # --- Sync endpoint: local-model compaction ---
    base_url = (options.env or {}).get("ANTHROPIC_BASE_URL", "")
    is_local = is_local_model(base_url)
    conv_text = ""

    if is_local:
        comp = load_and_compact_session(meta_path, model=model)
        conv_text = format_conversation_for_local(
            comp["history"],
            current_user_text=prefetched_text,
            model=model,
        )
        if comp["truncated"]:
            logger.info(
                "[compaction] %s: dropped %d tokens (%d → %d)",
                session_id,
                comp["tokens_before"] - comp["tokens_after"],
                comp["tokens_before"],
                comp["tokens_after"],
            )
        options.resume = None

    await _save_session_meta(session_id, model, preview=text)

    try:
        full_response = ""
        turn_stats: dict | None = None
        async for message in query(prompt=conv_text if is_local else prefetched_text, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        full_response += block.text
            elif isinstance(message, ResultMessage):
                if hasattr(message, "result") and message.result:
                    full_response = message.result
                usage = getattr(message, "usage", None) or {}
                turn_stats = {
                    "input_tokens":  usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_create":  usage.get("cache_creation_input_tokens", 0),
                    "cache_read":    usage.get("cache_read_input_tokens", 0),
                    "cost_usd":      getattr(message, "total_cost_usd", None),
                    "duration_ms":   getattr(message, "duration_ms", None),
                    "num_turns":     getattr(message, "num_turns", None),
                    "model":         model,
                }
                try:
                    usage_store.record_usage(
                        session_id=session_id,
                        model=model,
                        input_tokens=turn_stats["input_tokens"],
                        output_tokens=turn_stats["output_tokens"],
                        cache_create=turn_stats["cache_create"],
                        cache_read=turn_stats["cache_read"],
                        cost_usd=turn_stats["cost_usd"],
                        duration_ms=turn_stats["duration_ms"],
                        duration_api_ms=getattr(message, "duration_api_ms", None),
                        num_turns=turn_stats["num_turns"],
                    )
                except Exception as ue:
                    logger.warning(f"Failed to record usage: {ue}")

        # Persist assistant response for local models (no SDK resume)
        if is_local and full_response:
            await _append_messages(session_id, [{
                "id": uuid.uuid4().hex[:8],
                "role": "assistant",
                "content": [{"type": "text", "text": full_response}],
                "timestamp": datetime.now().isoformat(),
            }])

        return JSONResponse({"success": True, "response": full_response, "session_id": session_id, "stats": turn_stats})

    except Exception as e:
        logger.error(f"Message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
