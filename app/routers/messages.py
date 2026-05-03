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

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import usage_store
from app.config import (
    CONFIG,
    _get_model_env,
    _model_base_url,
    _resolve_model_name,
)
from app.harness import run_query, RunOptions, HookRegistry
from app.paths import SESSIONS_DIR
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
from app.compaction import load_and_compact_session
from app import event_log as _event_log  # Inner Voice — agent-side event capture
from app.inner_voice import heuristics as _iv_heuristics
from app.inner_voice import ensemble as _iv_ensemble
from app.inner_voice import consensus_termination as _iv_consensus
from app.inner_voice import grading as _iv_grading
from app.inner_voice import intra_turn as _iv_intra_turn
from usage_store import record_inner_voice_intervention


router = APIRouter()
logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Re-exports from the decomposed sibling modules (#decomposition).
# These functions used to live inline in this file; they were extracted
# to keep the chat router slim. Same names, same signatures.
# ---------------------------------------------------------------------------

from app.routers._messages_harness_adapter import (
    _emit,
    _content_to_string,
    _prepare_messages_for_harness,
)
from app.routers._messages_subliminal import (
    _extract_subliminal_prefix,
    _classify_subliminal,
    _detect_subliminal_sources,
    _build_subliminal_entry,
)
from app.routers._messages_inner_voice import (
    _session_inner_voice_enabled,
    _session_iv_evaluate_user_turns_enabled,
    _iv_should_fire_on_turn,
    _inner_voice_hooks_dict,
    _inner_voice_completion_check,
    _inner_voice_critic_check,
    _inner_voice_grading_pass,
    _inner_voice_mid_turn_drift_check,
)








async def _run_turn(session_id: str, turn: SessionTurn, q: SessionQueue) -> None:
    """Run a single turn through the harness, persisting as we go.

    Does not format SSE wire bytes — that's the subscriber's job. On
    return, the consumer pushes the sentinel `None` to close the stream.
    """
    payload = turn.payload
    text: str = payload["text"]
    prefetched_text: str = payload["prefetched_text"]
    model: str = payload["model"]
    options: RunOptions = payload["options"]
    meta_path = payload["meta_path"]
    cancel_event = q.cancel_event

    t_query_start = time.perf_counter()

    # Surface a session event immediately so the client has the turn_id
    # and can correlate any later queue-state updates.
    await _emit(turn, "session", {"session_id": session_id, "turn_id": turn.turn_id})
    # Initial queue snapshot so the client doesn't need a separate poll.
    await turn.events.put({"event": "queue_state", "data": get_queue_state(session_id)})

    # Inner Voice — capture turn-entry events to the per-session event
    # log. Best-effort; event_log.log_event swallows exceptions. Event
    # names use the historical `brain1.*` prefix (= "agent side") for
    # backward-compatibility with existing logs and dashboards; do not
    # rename without a migration.
    _event_log.log_event(session_id, "brain1.user_prompt_received", {
        "text": text,
        "source": turn.source,
        "model": model,
        "has_prefetched_context": prefetched_text != text,
    }, turn_id=turn.turn_id)
    _event_log.log_event(session_id, "brain1.options_built", {
        "model": model,
        "max_turns": options.max_turns,
        "permission_mode": options.permission_mode,
        "env_keys": sorted(options.env.keys()),
        "mcp_server_keys": sorted(options.mcp_servers.keys()),
        "disallowed_tools": list(options.disallowed_tools),
    }, turn_id=turn.turn_id)

    full_response = ""
    accumulated_thinking = ""
    tool_calls_log: list[dict] = []
    tool_results_log: list[dict] = []
    persisted_tool_ids: set[str] = set()
    final_persisted = False
    first_event = True
    stream_stats: dict[str, Any] = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_create": 0, "cache_read": 0,
        "cost_usd": None, "duration_ms": None, "num_turns": None,
        "model": model,
    }
    last_turn_input: int = 0

    # Per-iteration LLM usage from the harness's `assistant_message` event.
    # Persisted onto each tool-call/tool-result row so the UI can show
    # tokens-in/out/cached for every LLM-output row, not just the final one.
    current_iteration_stats: dict[str, Any] = {}

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

    # Build OpenAI-format messages from the compacted session history.
    # The harness is stateless per request — we reconstruct the full
    # conversation each turn, truncate to the model's context window,
    # and append the current user message.
    comp = load_and_compact_session(meta_path, model=model)
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
    harness_messages = await _prepare_messages_for_harness(comp["history"])
    # Strip trailing user message if present — we'll append the fresh
    # prefetched version (includes subliminal context the persisted copy lacks).
    if harness_messages and harness_messages[-1].get("role") == "user":
        harness_messages = harness_messages[:-1]
    harness_messages.append({"role": "user", "content": prefetched_text})

    # Wire cancel_event and session_id into options at run time
    # (they're not available when options is built in post_message_stream).
    options.cancel_event = cancel_event
    options.session_id = session_id

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
            cancel_event=cancel_event,
        )

    _event_log.log_event(session_id, "brain1.query_started", {
        "model": model,
        "prompt_chars": len(prefetched_text),
        "history_messages": len(harness_messages) - 1,
    }, turn_id=turn.turn_id)

    cancelled_mid_stream = False
    try:
        async for evt in run_query(harness_messages, options):
            etype = evt["type"]

            if first_event and etype not in ("system",):
                logger.info(
                    f"[TIMING] first harness event ({etype}) after "
                    f"{time.perf_counter() - t_query_start:.3f}s"
                )
                first_event = False

            if etype == "text_delta":
                delta_text = evt["text"]
                if delta_text:
                    if not full_response:
                        logger.info(
                            f"[TIMING] first text token after "
                            f"{time.perf_counter() - t_query_start:.3f}s (model TTFT)"
                        )
                    full_response += delta_text
                    await _emit(turn, "text_delta", {"text": delta_text})
                    _stream_delta_count += 1
                    if _stream_delta_count % _stream_sample_every == 0:
                        _event_log.log_event(session_id, "brain1.stream_event", {
                            "kind": "text_delta",
                            "delta_index": _stream_delta_count,
                            "position_chars": len(full_response),
                            "delta_chars": len(delta_text),
                        }, turn_id=turn.turn_id)
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
                    if (
                        _iv_intra_active
                        and not cancel_event.is_set()
                        and _stream_delta_count % _stream_sample_every == 0
                    ):
                        _iv_intra_turn._maybe_fire_progress_monitor(
                            session_id, trigger_check=False,
                        )
                    if tts_should_speak and not tts_spoken and tts_is_enabled():
                        tts_buffer += delta_text
                        spoken_chunk = extract_first_two_sentences(tts_buffer)
                        if spoken_chunk:
                            tts_spoken = True
                            asyncio.create_task(speak_text(spoken_chunk))

            elif etype == "thinking_delta":
                thinking_text = evt.get("text", "")
                if thinking_text:
                    accumulated_thinking += thinking_text
                    await _emit(turn, "thinking_delta", {"text": thinking_text})

            elif etype == "thinking_done":
                thinking_text = evt.get("text", "")
                accumulated_thinking = thinking_text
                await _emit(turn, "thinking_done", {"text": thinking_text})
                _event_log.log_event(session_id, "brain1.thinking_block_emitted", {
                    "thinking": thinking_text,
                    "chars": len(thinking_text),
                }, turn_id=turn.turn_id)

            elif etype == "assistant_message":
                # Capture per-iteration usage so subsequent tool_call /
                # tool_result rows can carry their own stats block.
                iter_usage = evt.get("usage") or {}
                iter_input = iter_usage.get("input_tokens") or iter_usage.get("prompt_tokens", 0)
                iter_output = iter_usage.get("output_tokens") or iter_usage.get("completion_tokens", 0)
                if iter_input or iter_output:
                    last_turn_input = iter_input or last_turn_input
                current_iteration_stats = {
                    "input_tokens": iter_input,
                    "output_tokens": iter_output,
                    "cache_read": iter_usage.get("cache_read", 0)
                        or iter_usage.get("prompt_tokens_cached", 0)
                        or 0,
                    "cache_create": iter_usage.get("cache_create", 0) or 0,
                    "duration_ms": evt.get("duration_ms", 0),
                    "iteration": evt.get("iteration", 0),
                    "model": model,
                }
                # Flush text segment to disk if tool calls follow it.
                if evt.get("tool_calls") and full_response.strip():
                    seg_ts = datetime.now().isoformat()
                    seg_entry: dict = {
                        "id": uuid.uuid4().hex[:8],
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_response}],
                        "timestamp": seg_ts,
                        "stats": dict(current_iteration_stats),
                    }
                    if accumulated_thinking:
                        seg_entry["reasoning"] = accumulated_thinking
                    await _append_messages(session_id, [seg_entry])
                    full_response = ""
                    accumulated_thinking = ""

            elif etype == "tool_call":
                call_id = evt["call_id"]
                name = evt["name"]
                args_json = evt.get("args_json", "{}")
                tc = {
                    "id": call_id, "call_id": call_id, "type": "function",
                    "function": {"name": name, "arguments": args_json},
                }
                tool_calls_log.append(tc)
                await _emit(turn, "tool_start", {
                    "call_id": call_id, "name": name,
                    "args": args_json, "context_tokens": last_turn_input,
                })
                _event_log.log_event(session_id, "brain1.tool_call_proposed", {
                    "tool_call_id": call_id,
                    "name": name,
                    "args": args_json,
                    "context_tokens": last_turn_input,
                }, turn_id=turn.turn_id)

            elif etype == "tool_result":
                call_id = evt["call_id"]
                result_str = evt.get("content", "")
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "...(truncated)"
                tool_results_log.append({"call_id": call_id, "result": result_str})
                await _emit(turn, "tool_complete", {
                    "call_id": call_id, "name": evt.get("name", ""), "result": result_str,
                })
                _event_log.log_event(session_id, "brain1.tool_result_received", {
                    "tool_call_id": call_id,
                    "result": result_str,
                    "result_chars": len(result_str),
                }, turn_id=turn.turn_id)
                # Eager per-pair persistence. Per-iteration LLM usage
                # rides on the assistant tool-call row (the LLM produced
                # the tool_call); the result row carries result_chars
                # only since MCP dispatch has no token cost.
                tc = next((t for t in tool_calls_log if t["call_id"] == call_id), None)
                if tc and call_id not in persisted_tool_ids:
                    persisted_tool_ids.add(call_id)
                    pair_ts = datetime.now().isoformat()
                    tc_msg: dict = {
                        "id": f"msg_{call_id}_tc",
                        "role": "assistant",
                        "content": [{"type": "text", "text": ""}],
                        "tool_calls": [tc],
                        "timestamp": pair_ts,
                    }
                    if current_iteration_stats:
                        tc_msg["stats"] = dict(current_iteration_stats)
                    result_msg: dict = {
                        "id": f"msg_{call_id}_result",
                        "role": "tool",
                        "content": [{"type": "text", "text": result_str}],
                        "tool_call_id": call_id,
                        "timestamp": pair_ts,
                        "stats": {
                            "result_chars": len(result_str),
                            "is_error": bool(evt.get("is_error", False)),
                        },
                    }
                    await _append_messages(session_id, [tc_msg, result_msg])

            elif etype == "result":
                usage = evt.get("usage") or {}
                # Map vLLM's OpenAI-style keys → legacy internal keys
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)
                stop_reason = evt.get("stop_reason", "stop")
                duration_ms = evt.get("duration_ms", 0)
                num_turns_val = evt.get("num_turns", 0)
                done_text = evt.get("response_text") or full_response
                cancelled_mid_stream = stop_reason == "cancelled"

                try:
                    usage_store.record_usage(
                        session_id=session_id,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_create=0,
                        cache_read=0,
                        cost_usd=0.0,
                        duration_ms=duration_ms,
                        duration_api_ms=None,
                        num_turns=num_turns_val,
                    )
                except Exception as ue:
                    logger.warning(f"Failed to record usage: {ue}")

                _event_log.log_event(session_id, "brain1.result_message", {
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    "stop_reason": stop_reason,
                    "duration_ms": duration_ms,
                    "num_turns": num_turns_val,
                    "response_chars": len(full_response),
                    "had_tool_calls": bool(tool_calls_log),
                }, turn_id=turn.turn_id)

                stream_stats.update({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "duration_ms": duration_ms,
                    "num_turns": num_turns_val,
                    "peak_input_tokens": last_turn_input,
                })
                stats_dict = stream_stats

                result_text = full_response

                end_ts = datetime.now().isoformat()
                tail: list[dict] = []
                results_by_id = {r["call_id"]: r["result"] for r in tool_results_log}
                for tc in tool_calls_log:
                    cid = tc["call_id"]
                    if cid not in persisted_tool_ids:
                        persisted_tool_ids.add(cid)
                        fallback_tc: dict = {"id": f"msg_{cid}_tc", "role": "assistant",
                                     "content": [{"type": "text", "text": ""}],
                                     "tool_calls": [tc], "timestamp": end_ts}
                        if current_iteration_stats:
                            fallback_tc["stats"] = dict(current_iteration_stats)
                        result_text_str = results_by_id.get(cid, "")
                        tail.append(fallback_tc)
                        tail.append({"id": f"msg_{cid}_result", "role": "tool",
                                     "content": [{"type": "text", "text": result_text_str}],
                                     "tool_call_id": cid, "timestamp": end_ts,
                                     "stats": {"result_chars": len(result_text_str)}})

                # Cancelled path: persist and emit done(cancelled=True).
                if cancelled_mid_stream:
                    stream_stats["peak_input_tokens"] = last_turn_input
                    if full_response.strip():
                        cancel_msg: dict = {
                            "id": uuid.uuid4().hex[:8], "role": "assistant",
                            "content": [{"type": "text", "text": full_response}],
                            "timestamp": end_ts, "stats": stream_stats, "cancelled": True,
                        }
                        if accumulated_thinking:
                            cancel_msg["reasoning"] = accumulated_thinking
                        tail.append(cancel_msg)
                    if tail:
                        await _append_messages(session_id, tail)
                    final_persisted = True
                    await _emit(turn, "done", {
                        "response": full_response,
                        "session_id": session_id,
                        "stats": stream_stats,
                        "cancelled": True,
                    })
                    continue  # skip the normal completion path below

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

                    # Inner Voice — post-loop checks.
                    #
                    # The completion-check regex heuristic catches turns
                    # that ended with non-empty text but neither a
                    # SIGNAL token nor a terminal tool call (i.e. the
                    # model said "Let me X" then stopped). Fires on
                    # ambient turns whenever the session has Inner Voice
                    # on, and on user turns additionally when the
                    # session opted into `inner_voice_evaluate_user_turns`.
                    #
                    # The critic ensemble + grading pass fires on the
                    # same set of turns — that flag is what makes the
                    # Inner Voice tab's chat surface critic verdicts in
                    # real time.
                    iv_should_fire = _iv_should_fire_on_turn(session_id, turn.source)
                    if iv_should_fire:
                        asyncio.ensure_future(_inner_voice_completion_check(
                            session_id=session_id,
                            turn_id=turn.turn_id,
                            response_text=done_text,
                            tool_calls=list(tool_calls_log),
                        ))
                    if iv_should_fire:
                        asyncio.ensure_future(_inner_voice_critic_check(
                            session_id=session_id,
                            turn_id=turn.turn_id,
                            turn_source=turn.source,
                            frozen_task_intent=text,
                            response_text=done_text,
                            tool_calls=list(tool_calls_log),
                            turn=turn,
                        ))
                        # Grading pass against any ungraded interventions
                        # from PRIOR turns. The critic check above may
                        # write a new intervention for THIS turn;
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

        # The harness always emits a `result` event (even on cancel), so
        # the post-loop cancel block from the old SDK path is not needed.
        # cancelled_mid_stream is set in the `result` handler and handled there.

    except Exception as e:
        if not final_persisted:
            if full_response or tool_calls_log:
                logger.warning(f"Turn {turn.turn_id} harness error with content: {e}")
                err_ts = datetime.now().isoformat()
                tail = []
                results_by_id = {r["call_id"]: r["result"] for r in tool_results_log}
                for tc in tool_calls_log:
                    cid = tc["call_id"]
                    if cid not in persisted_tool_ids:
                        err_tc: dict = {"id": f"msg_{cid}_tc", "role": "assistant",
                                     "content": [{"type": "text", "text": ""}],
                                     "tool_calls": [tc], "timestamp": err_ts}
                        if current_iteration_stats:
                            err_tc["stats"] = dict(current_iteration_stats)
                        result_text_str = results_by_id.get(cid, "")
                        tail.append(err_tc)
                        tail.append({"id": f"msg_{cid}_result", "role": "tool",
                                     "content": [{"type": "text", "text": result_text_str}],
                                     "tool_call_id": cid, "timestamp": err_ts,
                                     "stats": {"result_chars": len(result_text_str)}})
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
                logger.error(f"Turn {turn.turn_id} harness error: {e}")
                await _emit(turn, "error", {"detail": str(e)})



# ---------------------------------------------------------------------------
# Per-session consumer. Drains q.pending serially; lazily spawned.
# ---------------------------------------------------------------------------

async def _session_consumer(session_id: str) -> None:
    """Drain the queue for a single session. Exits when queue empties.

    Pop order: user tier first, then ambient. A user turn arriving while
    an ambient is running preempts it via the queue's cancel_event.
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
    session_turn_count = 0
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            session_turn_count = sum(1 for m in existing.get("messages", []) if m.get("role") == "user")
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

    iv_enabled = _session_inner_voice_enabled(session_id)
    iv_hooks = _inner_voice_hooks_dict(session_id) if iv_enabled else None

    options = RunOptions(
        model=model,
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=permission_mode,
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools() + extra_disallowed,
        env=model_env,
        hooks=iv_hooks,
        # cancel_event and session_id wired in _run_turn at run time
    )

    await _save_session_meta(session_id, model, preview=text)

    logger.info(
        f"[TIMING] pre-enqueue overhead: prompt={t_prompt - t0:.3f}s  "
        f"prefetch={t_prefetch - t_prompt:.3f}s  "
        f"total={time.perf_counter() - t0:.3f}s"
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

    system_prompt = build_system_prompt()

    iv_enabled = _session_inner_voice_enabled(session_id)
    iv_hooks = _inner_voice_hooks_dict(session_id) if iv_enabled else None

    options = RunOptions(
        model=model,
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=CONFIG.get("agent", {}).get("permission_mode", "bypassPermissions"),
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools(),
        env=model_env,
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

    meta_path = SESSIONS_DIR / f"{session_id}.json"

    sync_extra_disallowed: list[str] = data.get("extra_disallowed", [])
    sync_permission_mode: str = (
        data.get("permission_mode")
        or CONFIG.get("agent", {}).get("permission_mode", "bypassPermissions")
    )

    iv_enabled = _session_inner_voice_enabled(session_id)
    iv_hooks = _inner_voice_hooks_dict(session_id) if iv_enabled else None

    options = RunOptions(
        model=model,
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=sync_permission_mode,
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools() + sync_extra_disallowed,
        env=model_env,
        hooks=iv_hooks,
        session_id=session_id,
    )

    comp = load_and_compact_session(meta_path, model=model)
    if comp["truncated"]:
        logger.info(
            "[compaction] %s: dropped %d tokens (%d → %d)",
            session_id,
            comp["tokens_before"] - comp["tokens_after"],
            comp["tokens_before"],
            comp["tokens_after"],
        )
    messages = await _prepare_messages_for_harness(comp["history"])
    if messages and messages[-1].get("role") == "user":
        messages = messages[:-1]
    messages.append({"role": "user", "content": prefetched_text})

    await _save_session_meta(session_id, model, preview=text)

    try:
        full_response = ""
        turn_stats: dict | None = None
        async for evt in run_query(messages, options):
            if evt["type"] == "text_delta":
                full_response += evt["text"]
            elif evt["type"] == "result":
                usage = evt.get("usage") or {}
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)
                turn_stats = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_create": 0,
                    "cache_read": 0,
                    "cost_usd": 0.0,
                    "duration_ms": evt.get("duration_ms"),
                    "num_turns": evt.get("num_turns"),
                    "model": model,
                }
                try:
                    usage_store.record_usage(
                        session_id=session_id,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_create=0,
                        cache_read=0,
                        cost_usd=0.0,
                        duration_ms=evt.get("duration_ms"),
                        num_turns=evt.get("num_turns"),
                    )
                except Exception as ue:
                    logger.warning(f"Failed to record usage: {ue}")

        if full_response:
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
