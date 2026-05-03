"""Inner Voice (#345) integration glue for the chat router.

Pulled out of `messages.py` to keep the router slim. Public surface that
`messages.py` imports back:

  * `_session_inner_voice_enabled`           — gate flag from session JSON
  * `_session_iv_evaluate_user_turns_enabled` — Stage-5+ user-turn gate
  * `_iv_should_fire_on_turn`                — single-source-of-truth gate
  * `_inner_voice_hooks_dict`                — build the HookRegistry
  * `_inner_voice_completion_check`          — Stage-1 post-loop heuristic
  * `_inner_voice_brain2_check`              — Stage-2/4/6 critique + dispatch
  * `_inner_voice_grading_pass`              — Stage-5 outcome grading
  * `_inner_voice_mid_turn_drift_check`      — Stage-3 drift veto

`_maybe_dispatch_steer` is the Stage-6 helper invoked by the brain2 path.

`build_ambient_turn` and `enqueue_ambient` are imported lazily inside
`_maybe_dispatch_steer` because they live in `messages.py` and a top-level
import would close the cycle. Every other dependency is a stable
non-router module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from app.harness import HookRegistry
from app.paths import SESSIONS_DIR
from app.sessions_io import (
    AmbientPrefetchEntry,
    SessionTurn,
    enqueue_ambient_prefetch,
)
from app import event_log as _event_log
from app.inner_voice import heuristics as _iv_heuristics
from app.inner_voice import ensemble as _iv_ensemble
from app.inner_voice import consensus_termination as _iv_consensus
from app.inner_voice import grading as _iv_grading
from app.inner_voice import intra_turn as _iv_intra_turn
from app.routers._messages_harness_adapter import _emit
from usage_store import record_inner_voice_intervention


logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Session opt-in helpers (Stage 1)
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


def _inner_voice_hooks_dict(session_id: str) -> HookRegistry:
    """Build a HookRegistry for Inner Voice sessions.

    The harness fires `cb(input_data, tool_use_id, None)` — same shape as
    the old SDK contract — so heuristics.py / intra_turn.py callbacks plug
    in without translation.

    Stage 1 wires only PreToolUse (Bash matcher). Stage 7 adds
    PostToolUse + PostToolUseFailure (no matcher, every tool) for
    ``tool_result_grader`` / ``progress_monitor`` dispatch.
    """
    reg = HookRegistry()

    reg.add_pre_tool_use("Bash", _iv_heuristics.make_pretooluse_callback(session_id))

    if _iv_intra_turn.is_enabled():
        reg.add_post_tool_use(_iv_intra_turn.make_post_tool_use_callback(session_id))
        reg.add_post_tool_use_failure(_iv_intra_turn.make_post_tool_use_failure_callback(session_id))

    return reg


# ---------------------------------------------------------------------------
# Stage 1 — post-loop completion check (heuristic only, no Brain 2)
# ---------------------------------------------------------------------------

async def _inner_voice_completion_check(
    session_id: str,
    turn_id: str,
    response_text: str,
    tool_calls: list[dict],
) -> None:
    """Run the Stage 1 completion heuristic on a finished ambient turn.

    Spawned as `asyncio.ensure_future` from the result branch in
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
# Stage 2 / 4 / 6 — Brain 2 critique + consensus + steer dispatch
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
# Stage 6 — steer dispatch on nudge_proposed
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
    #
    # Lazy import: build_ambient_turn / enqueue_ambient live in
    # messages.py and a top-level import would close the cycle.
    from app.routers.messages import build_ambient_turn, enqueue_ambient

    ambient_turn = None
    result: dict[str, Any] = {}
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
                "ambient_turn_id": ambient_turn.turn_id if ambient_turn else None,
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
# Stage 5 — grading pass
# ---------------------------------------------------------------------------

async def _inner_voice_grading_pass(
    session_id: str,
    outcome_turn_id: str,
    outcome_response_text: str,
    outcome_tool_calls: list[dict],
    frozen_task_intent: str,
) -> None:
    """Run the Stage 5 grading pass against the just-finished ambient turn.

    Spawned via `asyncio.ensure_future` from the result branch in
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
# Stage 3 — mid-turn drift detection
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

    Spawned as `asyncio.ensure_future` from inside the harness message loop's
    `text_delta` handler. Best-effort; never raises into the consumer.

    On a veto-severity disagreement (`severity >= veto_severity_threshold`,
    `disagrees=True`, `error is None`):
      1. Log `inner_voice.cancel_event_fired` with stream position +
         partial-response delta count so meta-review can replay the
         interrupt sequence.
      2. Enqueue an ambient prefetch entry describing the drift verdict.
         The next user/ambient turn will surface it via `<context>`.
      3. Set `cancel_event` — the harness loop's cancel check breaks out
         on the next event boundary.

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
