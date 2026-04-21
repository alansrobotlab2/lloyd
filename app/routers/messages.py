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
)
from app.mcp_discovery import _get_mcp_servers, _get_disallowed_tools
from app.post_capture import _post_session_capture, _maybe_extract_focus
from prompt_builder import build_system_prompt
from prefetch import prefetch_context


router = APIRouter()
logger = logging.getLogger("lloyd-server")


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

    try:
        async for message in query(
            prompt=prefetched_text,
            options=options,
        ):
            if cancel_event.is_set():
                logger.info(f"Session {session_id} turn {turn.turn_id} cancelled via API")
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

                    done_payload: dict = {'response': done_text, 'session_id': session_id, 'stats': stats_dict}
                    if accumulated_thinking:
                        done_payload['reasoning'] = accumulated_thinking
                    await _emit(turn, "done", done_payload)

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
    )

    await _save_session_meta(session_id, model, preview=text)

    try:
        full_response = ""
        turn_stats: dict | None = None
        async for message in query(prompt=prefetched_text, options=options):
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

        return JSONResponse({"success": True, "response": full_response, "session_id": session_id, "stats": turn_stats})

    except Exception as e:
        logger.error(f"Message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
