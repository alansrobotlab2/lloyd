"""Agent loop — replaces `claude_agent_sdk.query()`.

`run_query(messages, options)` opens an MCP pool, advertises discovered
tools to vLLM, streams a chat completion, accumulates tool_call deltas,
fires hooks (PreToolUse → optional deny, PostToolUse, PostToolUseFailure),
dispatches surviving tool_calls through MCP, and loops until the model
emits `finish_reason="stop"` or `max_turns` is hit.

Yields normalized events suitable for direct consumption by the streaming
endpoint in `app.routers.messages._run_turn`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from app.harness import events
from app.harness.client import stream_chat
from app.harness.errors import ParseError, ToolDispatchError
from app.harness.events import NormalizedEvent
from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_URL, MCPPool
from app.harness.options import RunOptions
from app.harness.tool_schema import build_tool_list

logger = logging.getLogger("lloyd-harness-loop")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_query(
    messages: list[dict[str, Any]],
    options: RunOptions,
) -> AsyncIterator[NormalizedEvent]:
    """Run one agent loop and yield NormalizedEvents.

    `messages` is an OpenAI-style message list — typically built by
    `app.compaction.load_and_compact_session(...)` plus the current
    user turn. The harness does NOT prepend the system prompt; do that
    in the caller (so options.system_prompt can stay informational).

    Cancellation: if `options.cancel_event` is set during streaming, the
    httpx context exits cleanly and the loop emits a final `result`
    event with `stop_reason="cancelled"`.
    """
    started_at = time.perf_counter()

    # Prepend system prompt as a system message so vLLM sees it. We do
    # this here so callers don't have to worry about it; if they already
    # supplied a system message, theirs wins (we don't double-add).
    chat_messages = list(messages)
    if options.system_prompt and not _has_system(chat_messages):
        chat_messages.insert(0, {"role": "system", "content": options.system_prompt})

    pool = await _build_pool(options)
    try:
        tools = build_tool_list(pool.discovered, set(options.disallowed_tools))

        session_id = options.session_id or uuid.uuid4().hex
        yield events.system(session_id=session_id, model=options.model)

        accumulated_text = ""
        cumulative_usage: dict[str, int] = {}
        num_turns = 0
        stop_reason = "stop"

        while True:
            num_turns += 1
            if num_turns > options.max_turns:
                stop_reason = "max_turns"
                break

            if options.cancel_event is not None and options.cancel_event.is_set():
                stop_reason = "cancelled"
                break

            assistant_text = ""
            thinking_text = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            try:
                async for chunk in stream_chat(
                    base_url=options.base_url,
                    model=options.model,
                    messages=chat_messages,
                    tools=tools,
                    extra_body=options.extra_body,
                    cancel_event=options.cancel_event,
                    timeout_s=options.request_timeout_s,
                    api_key=options.api_key,
                ):
                    # Usage chunk arrives as the last event when
                    # stream_options.include_usage=True. vLLM emits it
                    # with choices=[].
                    if usage := chunk.get("usage"):
                        cumulative_usage = _merge_usage(cumulative_usage, usage)

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    if (txt := delta.get("content")) is not None:
                        if txt:
                            assistant_text += txt
                            yield events.text_delta(txt)

                    if (rc := delta.get("reasoning_content")) is not None:
                        if rc:
                            thinking_text += rc
                            yield events.thinking_delta(rc)

                    for tc_delta in delta.get("tool_calls") or []:
                        _accumulate_tool_call(tool_calls_acc, tc_delta)

                    if fr := choice.get("finish_reason"):
                        finish_reason = fr
            except ParseError as exc:
                logger.warning("loop: parse error mid-stream — %s", exc)
                yield events.stream_raw(exc.raw, error=str(exc))
                # Treat parse failure as an end-of-turn with whatever we
                # accumulated. The model can retry next turn.
                finish_reason = finish_reason or "stop"

            if options.cancel_event is not None and options.cancel_event.is_set():
                stop_reason = "cancelled"
                accumulated_text += assistant_text
                break

            if thinking_text:
                yield events.thinking_done(thinking_text)

            tool_calls_committed = _commit_tool_calls(tool_calls_acc)

            yield events.assistant_message(
                text=assistant_text,
                tool_calls=tool_calls_committed,
                thinking=thinking_text,
            )

            accumulated_text += assistant_text

            # Append assistant turn to history for the next loop pass
            # (or for the final result, harmless either way).
            chat_messages.append(_assistant_message_for_history(
                text=assistant_text, tool_calls=tool_calls_committed
            ))

            if not tool_calls_committed:
                stop_reason = finish_reason or "stop"
                break

            # Dispatch each tool call; accumulate results so we can
            # append them to history before looping back.
            for tc in tool_calls_committed:
                yield events.tool_call(
                    call_id=tc["id"],
                    name=tc["function"]["name"],
                    args_json=tc["function"]["arguments"],
                    args_dict=tc["_args_dict"],
                )

                result_evt = await _dispatch_one_tool_call(
                    tc=tc,
                    pool=pool,
                    options=options,
                    session_id=session_id,
                )
                yield result_evt

                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_evt["content"],
                })

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        yield events.result(
            stop_reason=stop_reason,
            usage=cumulative_usage,
            num_turns=num_turns,
            duration_ms=duration_ms,
            response_text=accumulated_text,
        )
    finally:
        await pool.aclose()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _build_pool(options: RunOptions) -> MCPPool:
    """Build an MCPPool from options.mcp_servers (or sensible default).

    If mcp_servers is empty, fall back to the unified lloyd-mcp
    aggregator on its default URL — almost every code path wants that
    anyway.
    """
    cfg = dict(options.mcp_servers)
    if not cfg:
        cfg = {"lloyd-mcp": {"type": "sse", "url": DEFAULT_LLOYD_MCP_URL}}
    pool = MCPPool(cfg)
    await pool.open()
    return pool


def _has_system(messages: list[dict[str, Any]]) -> bool:
    return any(m.get("role") == "system" for m in messages)


def _merge_usage(acc: dict[str, int], chunk: dict[str, Any]) -> dict[str, int]:
    """Merge a usage chunk into the running total. vLLM emits the final
    usage as cumulative for the request, so we replace rather than add.
    """
    out = dict(acc)
    for k, v in chunk.items():
        if isinstance(v, int):
            out[k] = v
    return out


def _accumulate_tool_call(
    acc: dict[int, dict[str, Any]], delta: dict[str, Any]
) -> None:
    """Accumulate a tool_call delta into the per-index buffer.

    OpenAI streams tool calls in pieces — the first delta carries
    `id`/`type`/`function.name`, subsequent deltas append to
    `function.arguments`. The `index` field is the disambiguator when
    the assistant emits multiple tool calls in one turn.
    """
    idx = delta.get("index", 0)
    cur = acc.setdefault(idx, {
        "id": "",
        "type": "function",
        "function": {"name": "", "arguments": ""},
    })
    if (cid := delta.get("id")):
        cur["id"] = cid
    if (ctype := delta.get("type")):
        cur["type"] = ctype
    if fn := delta.get("function"):
        if (name := fn.get("name")):
            cur["function"]["name"] += name
        if (args := fn.get("arguments")) is not None:
            cur["function"]["arguments"] += args


def _commit_tool_calls(acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Finalize accumulated tool calls.

    Parses `arguments` JSON; on failure, attaches an `_args_dict` of
    `{"__parse_error__": True, "raw": "..."}` so the dispatcher can
    surface a tool_result with `is_error=True` instead of crashing.
    """
    committed: list[dict[str, Any]] = []
    for idx in sorted(acc):
        tc = acc[idx]
        if not tc["function"]["name"]:
            # Empty placeholder — vLLM occasionally emits one when no
            # tools were called. Skip silently.
            continue
        if not tc["id"]:
            tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
        raw_args = tc["function"]["arguments"] or "{}"
        try:
            args_dict = json.loads(raw_args)
            if not isinstance(args_dict, dict):
                raise ValueError("tool arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "loop: tool_call args parse failed for %s: %s (raw=%r)",
                tc["function"]["name"], exc, raw_args,
            )
            args_dict = {"__parse_error__": True, "raw": raw_args, "error": str(exc)}
        tc["_args_dict"] = args_dict
        committed.append(tc)
    return committed


def _assistant_message_for_history(
    *, text: str, tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the OpenAI assistant message we append back to chat_messages
    for the next loop iteration. Strips the `_args_dict` helper so what
    we send to vLLM is spec-compliant.
    """
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": tc.get("type", "function"),
                "function": dict(tc["function"]),
            }
            for tc in tool_calls
        ]
        # OpenAI requires content=None when tool_calls present and no
        # text was emitted; some servers tolerate empty string.
        if not text:
            msg["content"] = None
    return msg


async def _dispatch_one_tool_call(
    *,
    tc: dict[str, Any],
    pool: MCPPool,
    options: RunOptions,
    session_id: str,
) -> NormalizedEvent:
    """Run hooks → MCP dispatch → hooks for a single tool call.

    Returns a `tool_result` event ready to yield. Encapsulates the
    deny / dispatch-error / success paths so the loop body stays flat.
    """
    name = tc["function"]["name"]
    args_dict = tc["_args_dict"]
    call_id = tc["id"]

    if args_dict.get("__parse_error__"):
        msg = f"Tool call arguments could not be parsed as JSON: {args_dict.get('error', 'unknown')}"
        return events.tool_result(call_id=call_id, name=name, content=msg, is_error=True)

    if options.disallowed_tools and (
        name in options.disallowed_tools
        or _namespaced_form(name) in options.disallowed_tools
    ):
        return events.tool_result(
            call_id=call_id, name=name,
            content=f"Tool {name!r} is disabled by configuration.",
            is_error=True,
        )

    # PreToolUse — first deny wins.
    if options.hooks is not None:
        deny = await options.hooks.fire_pre_tool_use(
            session_id=session_id,
            tool_name=name,
            tool_input=args_dict,
            tool_use_id=call_id,
        )
        if deny:
            hso = deny.get("hookSpecificOutput") or {}
            reason = hso.get("permissionDecisionReason") or "denied by hook"
            return events.tool_result(
                call_id=call_id, name=name,
                content=f"Tool call denied: {reason}", is_error=True,
            )

    # Dispatch.
    try:
        result = await pool.call_tool(name, args_dict)
    except ToolDispatchError as exc:
        if options.hooks is not None:
            await options.hooks.fire_post_tool_use_failure(
                session_id=session_id,
                tool_name=name,
                tool_input=args_dict,
                error=str(exc),
                tool_use_id=call_id,
            )
        return events.tool_result(
            call_id=call_id, name=name, content=str(exc), is_error=True,
        )
    except Exception as exc:
        logger.exception("loop: unexpected dispatch failure on %s", name)
        if options.hooks is not None:
            await options.hooks.fire_post_tool_use_failure(
                session_id=session_id,
                tool_name=name,
                tool_input=args_dict,
                error=str(exc),
                tool_use_id=call_id,
            )
        return events.tool_result(
            call_id=call_id, name=name,
            content=f"Tool dispatch failed: {exc}", is_error=True,
        )

    if options.hooks is not None:
        await options.hooks.fire_post_tool_use(
            session_id=session_id,
            tool_name=name,
            tool_input=args_dict,
            tool_response=result["content"],
            tool_use_id=call_id,
        )

    return events.tool_result(
        call_id=call_id,
        name=name,
        content=result["content"],
        is_error=result["is_error"],
    )


def _namespaced_form(bare_name: str) -> str:
    """Best-guess of the namespaced form for disallowed-tools matching.
    We don't actually know which server the tool came from at deny time;
    callers ought to use bare names in disallowed_tools for built-ins.
    """
    return f"mcp__lloyd-mcp__{bare_name}"
