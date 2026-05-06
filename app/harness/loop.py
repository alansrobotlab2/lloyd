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
from app.harness.errors import ContextOverflowError, ParseError, ToolDispatchError
from app.harness.microcompact import microcompact as _intra_microcompact
from app.harness.tool_result_spill import (
    fallback_for_empty_result,
    maybe_spill,
)
from app.harness.events import NormalizedEvent
from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_URL, MCPPool, get_or_open_pool
from app.harness.options import RunOptions
from app.harness.tool_schema import build_tool_list
from app.harness import tool_search_cache
from app.harness.tool_search import (
    LoadedToolSet,
    TOOLSEARCH_TOOL_NAME,
    format_catalog_reminder,
    search_tools,
)

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
    # When the caller supplies a shared chat_messages buffer (used by the
    # Inner Voice observer to inject system messages mid-turn), use it
    # directly. Otherwise copy `messages` into a private list as before.
    if options.chat_messages_handle is not None:
        chat_messages = options.chat_messages_handle
        if not chat_messages:
            chat_messages.extend(messages)
    else:
        chat_messages = list(messages)
    if options.system_prompt and not _has_system(chat_messages):
        chat_messages.insert(0, {"role": "system", "content": options.system_prompt})

    pool = await _build_pool(options)
    # Pool is process-shared (see mcp_pool.get_or_open_pool); do NOT
    # aclose() it here — lifecycle.shutdown_cleanup tears down all
    # pools at FastAPI shutdown.
    try:
        catalog = build_tool_list(list(pool.discovered), set(options.disallowed_tools))
        loaded_set = await _resolve_loaded_tool_set(options, catalog)
        if loaded_set.enabled:
            _inject_catalog_reminder(chat_messages, loaded_set)

        session_id = options.session_id or uuid.uuid4().hex
        yield events.system(session_id=session_id, model=options.model)

        accumulated_text = ""
        # Two distinct usage trackers:
        #   iteration_usage — populated freshly each loop pass; reflects
        #     ONE chat-completion's tokens. Emitted on assistant_message
        #     so the UI can attach per-row stats.
        #   total_usage — cross-iteration aggregate. input_tokens is max
        #     (peak), everything else is summed. Reported on the final
        #     `result` event so usage_store/UI see the whole turn.
        total_usage: dict[str, int] = {}
        last_iteration_usage: dict[str, int] = {}
        num_turns = 0
        stop_reason = "stop"
        context_overflow_recoveries = 0
        max_context_overflow_recoveries = 2

        while True:
            num_turns += 1
            if num_turns > options.max_turns:
                stop_reason = "max_turns"
                break

            if options.cancel_event is not None and options.cancel_event.is_set():
                stop_reason = "cancelled"
                break

            # Background-task completion drain. Splices any pending
            # <task_notification> messages into chat_messages so the
            # model sees them on this iteration. The callback also
            # persists them into the session JSON so reconstruction on
            # subsequent turns stays consistent.
            if options.notification_drain is not None:
                try:
                    drained = await options.notification_drain()
                except Exception as exc:
                    logger.warning("loop: notification_drain raised: %s", exc)
                    drained = []
                if drained:
                    chat_messages.extend(drained)
                    logger.info(
                        "loop: drained %d background-task notification(s)", len(drained),
                    )

            # Plan B — per-iteration disallowed-tools refresh. When the
            # caller wired a refresher (typically a closure over session
            # state), the harness re-evaluates the disallowed list on
            # every iteration. This is what lets ExitPlanMode flipping
            # plan_mode=false take effect mid-turn instead of waiting
            # for a fresh user turn to rebuild options.
            if options.disallowed_tools_refresh is not None:
                try:
                    current_disallowed: set[str] = set(
                        options.disallowed_tools_refresh() or []
                    )
                except Exception as exc:
                    logger.warning(
                        "loop: disallowed_tools_refresh raised: %s", exc,
                    )
                    current_disallowed = set(options.disallowed_tools or [])
            else:
                current_disallowed = set(options.disallowed_tools or [])

            iteration_started_at = time.perf_counter()
            iteration_usage: dict[str, int] = {}
            assistant_text = ""
            thinking_text = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            try:
                async for chunk in stream_chat(
                    base_url=options.base_url,
                    model=options.model,
                    messages=chat_messages,
                    tools=loaded_set.visible_tools(extra_disallowed=current_disallowed),
                    extra_body=options.extra_body,
                    cancel_event=options.cancel_event,
                    timeout_s=options.request_timeout_s,
                    api_key=options.api_key,
                    priority=options.priority,
                ):
                    # Usage chunk arrives as the last event when
                    # stream_options.include_usage=True. vLLM emits it
                    # with choices=[]. Fold into per-iteration only;
                    # cross-iteration total is computed once we know
                    # this iteration is final (after the stream loop).
                    if usage := chunk.get("usage"):
                        iteration_usage = _merge_usage(iteration_usage, usage)

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
            except ContextOverflowError as exc:
                # vLLM rejected the prompt for exceeding context. Recovery:
                # truncate the largest tool result(s) in chat_messages,
                # append a synthetic tool note explaining the truncation,
                # and let the loop retry the same turn. Bounded by
                # ``max_context_overflow_recoveries`` to avoid an infinite
                # loop if truncation can't free enough budget.
                if context_overflow_recoveries >= max_context_overflow_recoveries:
                    logger.error(
                        "loop: context overflow after %d recovery attempts — giving up",
                        context_overflow_recoveries,
                    )
                    raise
                context_overflow_recoveries += 1
                truncated_count, freed_chars = _truncate_largest_tool_results(
                    chat_messages, target_chars=80_000,
                )
                logger.warning(
                    "loop: context overflow (requested=%s tokens), recovery #%d: "
                    "truncated %d tool result(s), freed ~%d chars",
                    exc.requested_input_tokens,
                    context_overflow_recoveries,
                    truncated_count,
                    freed_chars,
                )
                yield events.stream_raw(
                    "",
                    error=(
                        f"context_overflow_recovery: attempt={context_overflow_recoveries}, "
                        f"truncated={truncated_count}, freed_chars={freed_chars}, "
                        f"requested_input_tokens={exc.requested_input_tokens}"
                    ),
                )
                num_turns -= 1   # don't count the recovered attempt against max_turns
                continue

            if options.cancel_event is not None and options.cancel_event.is_set():
                stop_reason = "cancelled"
                accumulated_text += assistant_text
                break

            if thinking_text:
                yield events.thinking_done(thinking_text)

            tool_calls_committed = _commit_tool_calls(tool_calls_acc)

            iteration_duration_ms = int((time.perf_counter() - iteration_started_at) * 1000)
            last_iteration_usage = iteration_usage
            total_usage = _accumulate_iteration_usage(total_usage, iteration_usage)
            asst_evt = events.assistant_message(
                text=assistant_text,
                tool_calls=tool_calls_committed,
                thinking=thinking_text,
                usage=iteration_usage,
                duration_ms=iteration_duration_ms,
                iteration=num_turns,
                finish_reason=finish_reason or "stop",
            )
            yield asst_evt
            # Snapshot chat_messages length before firing OnEvent. The
            # observer may append a system message ("inject" lever); if it
            # does AND the model is otherwise about to terminate this turn
            # (no tool calls), we continue the loop so the inject takes
            # effect on the next iteration instead of being lost.
            chat_msgs_len_before_hook = len(chat_messages)
            if options.hooks is not None:
                await options.hooks.fire_on_event(asst_evt)
            observer_injected = len(chat_messages) > chat_msgs_len_before_hook

            accumulated_text += assistant_text

            # Append assistant turn to history for the next loop pass
            # (or for the final result, harmless either way).
            chat_messages.append(_assistant_message_for_history(
                text=assistant_text, tool_calls=tool_calls_committed
            ))

            if not tool_calls_committed:
                if observer_injected:
                    # Observer injected a system message. Continue the loop
                    # so the model gets to read it and respond.
                    logger.info(
                        "loop: observer injected on terminal iteration — continuing loop",
                    )
                    continue
                stop_reason = finish_reason or "stop"
                break

            # Dispatch each tool call; accumulate results so we can
            # append them to history before looping back.
            for tc in tool_calls_committed:
                tc_evt = events.tool_call(
                    call_id=tc["id"],
                    name=tc["function"]["name"],
                    args_json=tc["function"]["arguments"],
                    args_dict=tc["_args_dict"],
                )
                yield tc_evt
                if options.hooks is not None:
                    await options.hooks.fire_on_event(tc_evt)

                result_evt = await _dispatch_one_tool_call(
                    tc=tc,
                    pool=pool,
                    options=options,
                    session_id=session_id,
                    loaded_set=loaded_set,
                    runtime_disallowed=current_disallowed,
                )
                yield result_evt
                if options.hooks is not None:
                    await options.hooks.fire_on_event(result_evt)

                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_evt["content"],
                })

            # Mid-turn microcompaction. After this iteration's tool calls
            # land, check whether tool results have piled up enough to
            # warrant clearing stale ones. Mutates `chat_messages` in
            # place so the observer's chat_messages_handle stays pointing
            # at the same list.
            if (
                tool_calls_committed
                and getattr(options, "intra_turn_microcompact_enabled", True)
            ):
                threshold = getattr(options, "intra_turn_microcompact_threshold", 15)
                keep = getattr(options, "intra_turn_microcompact_keep_recent", 5)
                tool_count = sum(1 for m in chat_messages if m.get("role") == "tool")
                if tool_count >= threshold:
                    compacted, cleared = _intra_microcompact(
                        chat_messages,
                        keep_recent_tools=keep,
                        count_threshold=threshold,
                    )
                    if cleared:
                        chat_messages[:] = compacted
                        logger.info(
                            "loop: intra-turn microcompact cleared %d/%d tool results "
                            "(kept last %d, iter=%d)",
                            cleared, tool_count, keep, num_turns,
                        )

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        result_done_evt = events.result(
            stop_reason=stop_reason,
            usage=total_usage,
            num_turns=num_turns,
            duration_ms=duration_ms,
            response_text=accumulated_text,
        )
        yield result_done_evt
        if options.hooks is not None:
            await options.hooks.fire_on_event(result_done_evt)
    finally:
        # Pool is shared across turns — see comment above _build_pool call.
        pass


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _build_pool(options: RunOptions) -> MCPPool:
    """Resolve options.mcp_servers (or sensible default) to a process-shared pool.

    Empty mcp_servers falls back to the unified lloyd-mcp aggregator on
    its default URL — almost every code path wants that anyway.
    """
    cfg = dict(options.mcp_servers)
    if not cfg:
        cfg = {"lloyd-mcp": {"type": "sse", "url": DEFAULT_LLOYD_MCP_URL}}
    return await get_or_open_pool(cfg)


def _has_system(messages: list[dict[str, Any]]) -> bool:
    return any(m.get("role") == "system" for m in messages)


_USAGE_KEY_REMAP = {
    "prompt_tokens": "input_tokens",
    "completion_tokens": "output_tokens",
}


def _merge_usage(acc: dict[str, int], chunk: dict[str, Any]) -> dict[str, int]:
    """Merge a usage chunk into the running total for ONE request.

    vLLM emits the final usage as cumulative for the request, so we
    replace rather than add. OpenAI-style keys (`prompt_tokens` /
    `completion_tokens`) are normalized to the Anthropic-style keys
    (`input_tokens` / `output_tokens`) every downstream consumer
    (usage_store, messages.py stats panel) expects.
    """
    out = dict(acc)
    for k, v in chunk.items():
        if not isinstance(v, int):
            continue
        out[_USAGE_KEY_REMAP.get(k, k)] = v
    return out


def _accumulate_iteration_usage(
    total: dict[str, int], iteration: dict[str, int],
) -> dict[str, int]:
    """Fold one iteration's usage into the cross-iteration total.

    Each chat-completion request reports its own input/output counts.
    Across the agent loop we want:
      - ``input_tokens``: the peak (last iteration is typically largest
        because tool results keep appending; max is conservative).
      - ``output_tokens``, ``cache_read``, ``cache_create``: SUM —
        every iteration writes new tokens and may hit cache.
    Without this, the final ``result`` event would show only the LAST
    iteration's usage (replace semantics from ``_merge_usage``).
    """
    out = dict(total)
    for k, v in iteration.items():
        if not isinstance(v, int):
            continue
        if k == "input_tokens":
            out[k] = max(out.get(k, 0), v)
        else:
            out[k] = out.get(k, 0) + v
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


_TOOL_ARGS_DECODER = json.JSONDecoder()


def _parse_tool_args_tolerant(raw: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Parse a tool-call `arguments` string, tolerating common qwen3_xml
    parser quirks (most often a trailing ``}}`` instead of ``}``).

    Returns ``(args_dict, repaired_raw, error)``:
      * On clean parse: ``(dict, None, None)``.
      * On repair: ``(dict, repaired_raw_string, None)`` — the caller
        should overwrite the `arguments` field with `repaired_raw_string`
        so the next-turn rebuild doesn't re-trigger the parse failure.
      * On total failure: ``(None, None, error_string)``.
    """
    if not raw:
        return ({}, None, None)
    # Happy path.
    try:
        v = json.loads(raw)
        if isinstance(v, dict):
            return (v, None, None)
        return (None, None, "tool arguments must be a JSON object")
    except json.JSONDecodeError as e:
        first_err = str(e)

    # Repair: parse the first JSON value and accept if the rest is just
    # noise (extra `}`, whitespace). Catches the qwen3_xml trailing-brace
    # pattern: `{"file_path": "..."}}`.
    try:
        v, end = _TOOL_ARGS_DECODER.raw_decode(raw)
    except json.JSONDecodeError:
        return (None, None, first_err)
    if not isinstance(v, dict):
        return (None, None, "tool arguments must be a JSON object")
    trailing = raw[end:].strip()
    # Allow any combination of stray `}`, `,`, whitespace as trailing junk.
    if trailing and any(c not in "}, \t\r\n" for c in trailing):
        return (None, None, first_err)
    repaired = raw[:end]
    return (v, repaired, None)


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
        args_dict, repaired, err = _parse_tool_args_tolerant(raw_args)
        if args_dict is None:
            logger.warning(
                "loop: tool_call args parse failed for %s: %s (raw=%r)",
                tc["function"]["name"], err, raw_args,
            )
            args_dict = {"__parse_error__": True, "raw": raw_args, "error": err}
            # vLLM ingests assistant tool_calls back as conversation
            # history on the next turn and re-parses `arguments`. If we
            # leave the malformed string, the next request 400s and the
            # loop dies. Replace with valid empty JSON — the tool_result
            # we'll emit already tells the model what went wrong.
            tc["function"]["arguments"] = "{}"
        elif repaired is not None:
            # Quirk-recovered. Log at INFO so we can track frequency, and
            # rewrite the stored arguments string so next-turn replay sees
            # clean JSON instead of re-tripping the parser.
            logger.info(
                "loop: tool_call args quirk-repaired for %s (was %d chars, now %d)",
                tc["function"]["name"], len(raw_args), len(repaired),
            )
            tc["function"]["arguments"] = repaired
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
    loaded_set: LoadedToolSet,
    runtime_disallowed: set[str] | None = None,
) -> NormalizedEvent:
    """Run hooks → MCP dispatch → hooks for a single tool call.

    Returns a `tool_result` event ready to yield. Encapsulates the
    deny / dispatch-error / success paths so the loop body stays flat.

    `runtime_disallowed` (Plan B) — the loop's per-iteration disallowed
    set, computed via `options.disallowed_tools_refresh` if set. When
    None, falls back to `options.disallowed_tools` (the static turn-start
    list). Either way, this is the gate that blocks dispatch.
    """
    name = tc["function"]["name"]
    args_dict = tc["_args_dict"]
    call_id = tc["id"]

    if args_dict.get("__parse_error__"):
        msg = f"Tool call arguments could not be parsed as JSON: {args_dict.get('error', 'unknown')}"
        return events.tool_result(call_id=call_id, name=name, content=msg, is_error=True)

    effective_disallowed = (
        runtime_disallowed
        if runtime_disallowed is not None
        else set(options.disallowed_tools or [])
    )
    if name in effective_disallowed:
        return events.tool_result(
            call_id=call_id, name=name,
            content=f"Tool {name!r} is disabled by configuration.",
            is_error=True,
        )

    # ToolSearch — intercept locally, no MCP round-trip. The matched tools
    # are added to the LoadedToolSet so subsequent turns see them in
    # ``visible_tools()`` (and thus in vLLM's ``tools=`` array).
    if name == TOOLSEARCH_TOOL_NAME and loaded_set.enabled:
        query = str(args_dict.get("query", "") or "")
        max_results = args_dict.get("max_results")
        if not isinstance(max_results, int) or max_results < 1:
            max_results = options.tool_search_max_results_default
        max_results = min(max_results, options.tool_search_max_results_cap)
        matched_names, content = search_tools(
            query, max_results=max_results, catalog=loaded_set.catalog,
        )
        loaded_set.mark_loaded(matched_names)
        logger.info(
            "loop: ToolSearch query=%r max_results=%d matched=%d (loaded set now %d)",
            query, max_results, len(matched_names), len(loaded_set.loaded),
        )
        return events.tool_result(
            call_id=call_id, name=name, content=content, is_error=False,
        )

    # Defensive intercept: vLLM should only dispatch tools that are in the
    # ``tools=`` array we sent, but the qwen3_xml parser is lenient and may
    # let an unadvertised name through. Catch it before MCP and guide the
    # model back to ToolSearch instead of failing opaquely.
    if loaded_set.enabled and not loaded_set.is_visible(name):
        catalog_names = {t["function"]["name"] for t in loaded_set.catalog}
        if name in catalog_names:
            return events.tool_result(
                call_id=call_id, name=name,
                content=(
                    f"Tool {name!r} requires schema loading before it can be "
                    f"called. Call ToolSearch(query=\"select:{name}\") first, "
                    f"then retry."
                ),
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

    # Dispatch. Inject session correlation as `_session_id` on a copy of
    # the args — the per-tool handler in agent_mcp/main.py pops it before
    # schema validation. Don't mutate args_dict itself: it's also the
    # source for the persisted tool_call event and any next-turn replay.
    dispatch_args = dict(args_dict)
    if session_id:
        dispatch_args["_session_id"] = session_id
    try:
        result = await pool.call_tool(name, dispatch_args)
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

    # Tool-result post-processing — applies in this order:
    #   1) empty-result fallback: some local models treat "" as a stop
    #      signal; replace with an explicit "(<tool> completed with no
    #      output)" marker.
    #   2) disk spill: results above SPILL_THRESHOLD_CHARS go to
    #      <SESSIONS_DIR>/<sid>.tool-results/<call_id>.<ext>, replaced
    #      in-prompt with a <persisted-output> preview block. Modeled on
    #      Claude Code's tool-result storage. The model can re-read the
    #      full file with the Read tool if it needs more than the preview.
    content = result["content"]
    is_error = result["is_error"]
    if not is_error and isinstance(content, str):
        content = fallback_for_empty_result(content, name)
        content = maybe_spill(
            content,
            tool_name=name,
            tool_use_id=call_id,
            session_id=session_id,
        )

    if options.hooks is not None:
        await options.hooks.fire_post_tool_use(
            session_id=session_id,
            tool_name=name,
            tool_input=args_dict,
            tool_response=content,
            tool_use_id=call_id,
        )

    return events.tool_result(
        call_id=call_id,
        name=name,
        content=content,
        is_error=is_error,
    )


# ---------------------------------------------------------------------------
# Tool search wiring
# ---------------------------------------------------------------------------

# Default always-visible tools when ToolSearch is on. The seven built-in
# file/shell tools are useful enough on every turn that lazy-loading them
# would just waste a ToolSearch round-trip.
_DEFAULT_BASELINE_TOOLS = ("Bash", "Read", "Write", "Edit", "Grep", "Glob", "Task")


async def _resolve_loaded_tool_set(
    options: RunOptions, catalog: list[dict[str, Any]],
) -> LoadedToolSet:
    """Build the per-session LoadedToolSet, honoring activation thresholds.

    Activation rule:
      enabled = options.tool_search_enabled
                AND ToolSearch is not in disallowed_tools
                AND len(catalog) >= options.tool_search_threshold_tools
    """
    catalog_names = {t["function"]["name"] for t in catalog}
    disallowed = set(options.disallowed_tools or [])

    if options.tool_search_baseline:
        baseline_candidates = options.tool_search_baseline
    else:
        baseline_candidates = list(_DEFAULT_BASELINE_TOOLS)
    baseline = {
        n for n in baseline_candidates
        if n in catalog_names and n not in disallowed
    }

    enabled = (
        options.tool_search_enabled
        and TOOLSEARCH_TOOL_NAME not in disallowed
        and len(catalog) >= options.tool_search_threshold_tools
    )
    if options.tool_search_enabled and not enabled and TOOLSEARCH_TOOL_NAME in disallowed:
        logger.warning(
            "loop: tool_search requested but ToolSearch is in disallowed_tools — "
            "falling back to full catalog (%d tools).",
            len(catalog),
        )

    return await tool_search_cache.get_or_create(
        session_id=options.session_id or "",
        catalog=catalog,
        baseline=baseline,
        enabled=enabled,
    )


_CATALOG_REMINDER_MARKER = "<!--lloyd-toolsearch-catalog-reminder-->"


def _inject_catalog_reminder(
    chat_messages: list[dict[str, Any]], loaded_set: LoadedToolSet,
) -> None:
    """Append the deferred-tools catalog to the leading system message.

    Idempotent: tagged with a marker the function greps for before
    appending, so re-running on the same chat_messages list is a no-op.

    Why append instead of insert a second system message: vLLM's chat
    templates (qwen3, gpt-oss, etc.) require exactly one system message
    at position 0. A second ``role: system`` anywhere — including
    immediately after the first — gets rejected with
    ``"System message must be at the beginning."``. Appending into the
    existing system message's content is the only safe shape.

    If no system message exists yet, one is inserted at position 0 with
    only the reminder body — same constraint, just no leading content
    to merge into.
    """
    body = format_catalog_reminder(loaded_set.catalog)
    if not body:
        return
    addendum = f"\n\n{_CATALOG_REMINDER_MARKER}\n{body}"

    for m in chat_messages:
        if m.get("role") == "system" and _CATALOG_REMINDER_MARKER in (m.get("content") or ""):
            return

    for i, m in enumerate(chat_messages):
        if m.get("role") == "system":
            existing = m.get("content") or ""
            m["content"] = existing + addendum
            return
        # First non-system message — no system to merge into; insert one.
        break

    chat_messages.insert(0, {"role": "system", "content": addendum.lstrip()})




def _truncate_largest_tool_results(
    chat_messages: list[dict[str, Any]],
    *,
    target_chars: int,
) -> tuple[int, int]:
    """Replace the largest tool-result message contents with a truncation
    notice until at least ``target_chars`` of content has been freed.

    Operates in-place on ``chat_messages`` (each dict's ``content`` field
    is overwritten). Returns ``(num_truncated, total_chars_freed)``.

    Strategy: rank tool messages by content length descending, walk from
    the largest down, replacing each with a short error string that
    surfaces what happened to the model. Stop as soon as we've freed
    ``target_chars`` chars cumulatively, OR after we've replaced every
    tool message that's > 4KB (smaller results aren't worth touching).
    """
    candidates: list[tuple[int, int]] = []   # (size, message_index)
    for i, msg in enumerate(chat_messages):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            size = len(content)
        elif isinstance(content, list):
            size = sum(
                len(b.get("text", "")) for b in content
                if isinstance(b, dict)
            )
        else:
            size = 0
        if size > 4096:
            candidates.append((size, i))
    candidates.sort(reverse=True)

    freed = 0
    truncated = 0
    for size, idx in candidates:
        original_size = size
        notice = (
            f"[harness: tool result truncated by context-overflow recovery — "
            f"original was {original_size} chars. The combined tool history "
            f"exceeded the model's context window. Re-run the call with a "
            f"narrower query (smaller hops, higher min_confidence, fewer "
            f"max_results) or use a more targeted tool.]"
        )
        msg = chat_messages[idx]
        if isinstance(msg.get("content"), list):
            msg["content"] = [{"type": "text", "text": notice}]
        else:
            msg["content"] = notice
        freed += original_size - len(notice)
        truncated += 1
        if freed >= target_chars:
            break
    return truncated, freed
