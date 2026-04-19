"""Chat endpoints — SSE streaming and synchronous one-shot.

`POST /api/message/stream` is the hot path. The nested `event_generator`
translates SDK events into the frontend's SSE contract, persists tool
pairs eagerly (so disconnects don't lose history), auto-wires
pipeline_dispatch results back to the requester session, records usage,
and fires post-session capture + focus extraction on completion.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime

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
from app.sessions_io import _active_streams, _save_session_meta, _append_messages
from app.mcp_discovery import _get_mcp_servers, _get_disallowed_tools
from app.post_capture import _post_session_capture, _maybe_extract_focus
from prompt_builder import build_system_prompt
from prefetch import prefetch_context


router = APIRouter()
logger = logging.getLogger("lloyd-server")


@router.post("/api/message/stream")
async def post_message_stream(request: Request):
    """SSE endpoint: streams tool_start, tool_complete, text_delta, and done events."""
    data = await request.json()
    text = data.get("text", "").strip()
    session_id = data.get("session_id", "")
    model_override = data.get("model", "")
    think_level = data.get("think", "")  # off/low/medium/high or empty

    if not text:
        raise HTTPException(status_code=400, detail="Message text required")

    if session_id and session_id in _active_streams:
        raise HTTPException(status_code=409, detail="Session is already streaming")

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

    _save_session_meta(session_id, model, preview=text)

    cancel_event = asyncio.Event()
    _active_streams[session_id] = cancel_event

    async def event_generator():
        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"

        t_query_start = time.perf_counter()
        logger.info(
            f"[TIMING] pre-query overhead: prompt={t_prompt - t0:.3f}s  "
            f"prefetch={t_prefetch - t_prompt:.3f}s  "
            f"total={t_query_start - t0:.3f}s  resume={'yes' if resume_id else 'no'}"
        )

        full_response = ""
        accumulated_thinking = ""
        tool_calls_log = []
        tool_results_log = []
        persisted_tool_ids: set[str] = set()
        final_persisted = False
        first_event = True
        pending_pipeline_wires: dict[str, str] = {}
        stream_stats: dict = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_create": 0, "cache_read": 0,
            "cost_usd": None, "duration_ms": None, "num_turns": None,
            "model": model,
        }
        # stream_stats["input_tokens"] gets overwritten per turn, but ResultMessage later
        # replaces it with the cumulative total — we need both values.
        last_turn_input: int = 0

        now_ts = datetime.now().isoformat()
        _append_messages(session_id, [{
            "id": uuid.uuid4().hex[:8],
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "timestamp": now_ts,
        }])

        try:
            async for message in query(
                prompt=prefetched_text,
                options=options,
            ):
                if cancel_event.is_set():
                    logger.info(f"Session {session_id} cancelled via API")
                    break

                if first_event:
                    logger.info(f"[TIMING] first SDK event after {time.perf_counter() - t_query_start:.3f}s (SDK+MCP startup)")
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
                                    logger.info(f"[TIMING] first text token after {time.perf_counter() - t_query_start:.3f}s (model TTFT)")
                                full_response += delta_text
                                yield f"event: text_delta\ndata: {json.dumps({'text': delta_text})}\n\n"
                        elif dtype == "thinking_delta":
                            thinking_text = delta.get("thinking", "")
                            if thinking_text:
                                accumulated_thinking += thinking_text
                                yield f"event: thinking_delta\ndata: {json.dumps({'text': thinking_text})}\n\n"
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
                        if meta_path.exists():
                            meta = json.loads(meta_path.read_text())
                            meta["sdk_session_id"] = sdk_session
                            meta_path.write_text(json.dumps(meta, indent=2))

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
                        _append_messages(session_id, [seg_entry])
                        full_response = ""
                        accumulated_thinking = ""
                    for block in message.content:
                        if isinstance(block, ThinkingBlock):
                            accumulated_thinking = block.thinking
                            yield f"event: thinking_done\ndata: {json.dumps({'text': block.thinking})}\n\n"
                        elif isinstance(block, ToolUseBlock):
                            args_str = json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                            tool_calls_log.append({"id": block.id, "call_id": block.id, "type": "function", "function": {"name": block.name, "arguments": args_str}})
                            yield f"event: tool_start\ndata: {json.dumps({'call_id': block.id, 'name': block.name, 'args': args_str, 'context_tokens': last_turn_input})}\n\n"
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
                            yield f"event: tool_complete\ndata: {json.dumps({'call_id': call_id, 'name': '', 'result': result_str})}\n\n"
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

                            # Eager per-pair persistence so a mid-stream disconnect
                            # doesn't lose tool history.
                            tc = next((t for t in tool_calls_log if t["call_id"] == call_id), None)
                            if tc and call_id not in persisted_tool_ids:
                                persisted_tool_ids.add(call_id)
                                pair_ts = datetime.now().isoformat()
                                _append_messages(session_id, [
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
                            if meta_path.exists():
                                meta = json.loads(meta_path.read_text())
                                meta["sdk_session_id"] = message.session_id
                                meta_path.write_text(json.dumps(meta, indent=2))
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
                    if result_text.strip():
                        msg_entry: dict = {"id": uuid.uuid4().hex[:8], "role": "assistant",
                                     "content": [{"type": "text", "text": result_text}],
                                     "timestamp": end_ts, "stats": stats_dict}
                        if accumulated_thinking:
                            msg_entry["reasoning"] = accumulated_thinking
                        tail.append(msg_entry)
                    if tail:
                        _append_messages(session_id, tail)
                    final_persisted = True

                    asyncio.ensure_future(_post_session_capture(session_id))
                    asyncio.ensure_future(_maybe_extract_focus(session_id))

                    done_payload: dict = {'response': done_text, 'session_id': session_id, 'stats': stats_dict}
                    if accumulated_thinking:
                        done_payload['reasoning'] = accumulated_thinking
                    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            if not final_persisted:
                if full_response or tool_calls_log:
                    # SDK exit-code-1 after completion, or mid-stream error with content.
                    logger.warning(f"Stream ended without ResultMessage (content delivered): {e}")
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
                        _append_messages(session_id, tail)
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
                    yield f"event: done\ndata: {json.dumps({'response': full_response, 'session_id': session_id, 'stats': stream_stats})}\n\n"
                else:
                    logger.error(f"Stream error: {e}")
                    yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            else:
                # Normal: SDK exited with code 1 after clean completion — ignore
                logger.debug(f"Post-completion SDK exit (ignored): {e}")
        finally:
            _active_streams.pop(session_id, None)
            # Catch client disconnects (CancelledError is BaseException, not Exception).
            # Tool pairs were already persisted eagerly; only the response text may be missing.
            if not final_persisted and full_response.strip():
                logger.info(f"Client disconnected mid-stream — persisting partial response ({len(full_response)} chars)")
                _append_messages(session_id, [{
                    "id": uuid.uuid4().hex[:8],
                    "role": "assistant",
                    "content": [{"type": "text", "text": full_response}],
                    "timestamp": datetime.now().isoformat(),
                }])

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/message")
async def post_message(request: Request):
    """Synchronous message endpoint — collects full response then returns."""
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

    _save_session_meta(session_id, model, preview=text)

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
