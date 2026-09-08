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
import re
import time
import uuid
from typing import Any, AsyncIterator

from app.harness import events
from app.harness.client import stream_chat
from app.harness.errors import (
    ContextOverflowError,
    ParseError,
    ToolDiscoveryError,
    ToolDispatchError,
)
from app.harness.microcompact import microcompact as _intra_microcompact
from app.harness.tool_result_spill import (
    fallback_for_empty_result,
    maybe_spill,
)
from app.harness.events import NormalizedEvent
from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS, MCPPool, get_or_open_pool
from app.harness.options import RunOptions
from app.harness.tool_schema import (
    add_summary_param,
    build_tool_list,
    pop_summary,
)
from app.harness import tool_search_cache
from app.harness.tool_search import (
    LoadedToolSet,
    TOOLSEARCH_TOOL_NAME,
    format_catalog_reminder,
    search_tools,
)

logger = logging.getLogger("lloyd-harness-loop")


# ---------------------------------------------------------------------------
# Echo guard
# ---------------------------------------------------------------------------
#
# Failure mode this defends against: the model emits a fenced *shell* block
# (```bash …```) as plain text and then stops, WITHOUT calling the Bash tool —
# i.e. it prints the command instead of running it. This shows up when a
# command-dense skill (an auto-generated bash runbook) lands in context and
# the fenced examples act as few-shot "print this" demonstrations. When that
# happens and Bash is actually available, nudge the model once to either call
# the tool or, if it was only showing the command for reference, finish
# normally. Bounded to one re-prompt per turn so a model that keeps echoing
# can't spin the loop.

# Shell-family fences only. We deliberately exclude ```python / ```json / etc.
# — those are far more often legitimately *shown* to the user, whereas a bare
# shell fence in an agent turn almost always means "I meant to run this".
_EXEC_FENCE_RE = re.compile(r"```[ \t]*(?:bash|sh|shell|zsh|console)\b", re.IGNORECASE)

_ECHO_GUARD_NUDGE = (
    "You wrote a shell command inside a code block but did not actually call a "
    "tool, so nothing ran. If you intended to execute it, call the Bash tool now "
    "with that exact command. If you were only showing the command to the user "
    "for reference, reply normally and do not include a fenced shell block."
)

_MAX_ECHO_GUARD_REPROMPTS = 1


def _looks_like_unexecuted_command(text: str) -> bool:
    """True if `text` contains a fenced shell block (a likely echoed command)."""
    if not text:
        return False
    return bool(_EXEC_FENCE_RE.search(text))


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
    _run_started()

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
        # Every advertised tool grows one extra string parameter the model
        # fills in with a phrase describing what the call is doing, which
        # the transcript renders beside the tool name. `summary_tools` is
        # the set that actually received it — a tool with a real `summary`
        # parameter of its own (session_inject_context) keeps it, and its value must
        # reach MCP untouched.
        if options.tool_call_summaries:
            summary_tools = add_summary_param(catalog)
            summary_tools.add(TOOLSEARCH_TOOL_NAME)
        else:
            summary_tools = set()
        # A turn with no tools is not a degraded turn, it is a broken one,
        # and it fails in the least legible way available: `stream_chat`
        # omits `tools` from the request when the list is empty, vLLM
        # therefore never engages its tool parser, and the model — which
        # can still read its whole toolbox in the system prompt — reasons
        # its way to "call Bash" and then has no channel to do it on. What
        # comes out is an empty message, or the tool call written as prose,
        # or invented tool *output*. Nothing in the stream says "no tools";
        # it reads as the model having forgotten how to use them.
        #
        # `pool.discovered` empty means discovery, not config, is at fault:
        # disabling all 130 tools via `disallowed_tools` would still leave
        # `discovered` populated. Fail loudly so the turn surfaces an error
        # instead of silently hallucinating for half an hour (2026-09-06).
        if not any(tools for _srv, tools in pool.discovered):
            raise ToolDiscoveryError(
                "MCP pool advertised no tools — refusing to run a toolless "
                "turn (the model would narrate tool calls instead of making "
                "them). Check the lloyd-mcp aggregator on :8500.",
                servers=sorted(options.mcp_servers or {}),
            )
        # Record the tool universe so plan mode can derive its gate from
        # tool annotations rather than a hardcoded name list. Uses the
        # unfiltered discovery, not `catalog` — a tool disabled in config
        # still exists and must stay gated if it is ever re-enabled.
        try:
            from app.mcp_discovery import record_tool_universe

            record_tool_universe(
                t["name"] for _srv, tools in pool.discovered for t in tools
            )
        except Exception as exc:  # never let bookkeeping break a turn
            logger.debug("loop: record_tool_universe skipped: %s", exc)
        loaded_set = await _resolve_loaded_tool_set(
            options, catalog, summaries=options.tool_call_summaries,
        )
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
        # Caption bookkeeping — see `_CAPTION_NUDGE` and events.result.
        caption_total = 0
        caption_present = 0
        caption_nudged = False
        num_turns = 0
        stop_reason = "stop"
        context_overflow_recoveries = 0
        max_context_overflow_recoveries = 2
        echo_guard_reprompts = 0
        # Initialised before the loop: a turn that breaks on its first check
        # (cancelled, max_turns=0) never assigns it, and the finalizer below
        # runs on every exit path.
        last_visible_tools: list[dict[str, Any]] = []

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

            # Per-iteration state re-anchor (todos / plan / goal). Appended,
            # never merged into the system prompt: position 0 must stay
            # byte-stable or every iteration re-prefills the whole context.
            # These are NOT persisted — see RunOptions.state_anchor.
            if options.state_anchor is not None:
                try:
                    anchors = await options.state_anchor(num_turns)
                except Exception as exc:
                    logger.warning("loop: state_anchor raised: %s", exc)
                    anchors = []
                if anchors:
                    chat_messages.extend(anchors)
                    logger.info(
                        "loop: state anchor re-injected %d message(s) (iter=%d)",
                        len(anchors), num_turns,
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
            # Wall time spent producing reasoning: first chunk to last.
            # Deliberately not the iteration's own clock — that also
            # covers prefill (which can be minutes on a cold 262k prompt)
            # and the answer generated after thinking ends.
            thinking_started_at: float | None = None
            thinking_last_at: float = 0.0
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            # Held so the finalizer can send the IDENTICAL array. Qwen's
            # template renders `tools` inside the system message, so a
            # different (or absent) list changes the rendered prompt from the
            # first token and vLLM re-prefills the whole conversation.
            last_visible_tools = loaded_set.visible_tools(
                extra_disallowed=current_disallowed)

            try:
                async for chunk in stream_chat(
                    base_url=options.base_url,
                    model=options.model,
                    messages=chat_messages,
                    tools=last_visible_tools,
                    extra_body=options.extra_body,
                    cancel_event=options.cancel_event,
                    timeout_s=options.request_timeout_s,
                    api_key=options.api_key,
                    priority=options.priority,
                    chunk_timeout_s=getattr(
                        options, "stream_chunk_timeout_s", 0.0
                    ),
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

                    # vLLM's qwen3 reasoning parser emitted reasoning under
                    # `reasoning_content` through ~0.22; 0.23+ renamed the
                    # streaming/message field to `reasoning`. Accept both so the
                    # thinking panel keeps working across vLLM versions.
                    rc = delta.get("reasoning_content")
                    if rc is None:
                        rc = delta.get("reasoning")
                    if rc:
                        now = time.perf_counter()
                        if thinking_started_at is None:
                            thinking_started_at = now
                        thinking_last_at = now
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
                thinking_ms = (
                    int((thinking_last_at - thinking_started_at) * 1000)
                    if thinking_started_at is not None
                    else 0
                )
                yield events.thinking_done(thinking_text, duration_ms=thinking_ms)

            tool_calls_committed = _commit_tool_calls(
                tool_calls_acc, summary_tools=summary_tools,
            )

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
            accumulated_text += assistant_text

            # Append this iteration's assistant turn to history BEFORE firing
            # the hook.
            #
            # The observer's `inject` lever appends to this same list, so
            # firing first put the nudge at index n and the assistant text it
            # was reacting to at n+1:
            #
            #   user:      "[INNER VOICE] You ended the turn by announcing…"
            #   assistant: "Let me check the logs:"   <- what the inject is about
            #
            # The nudge preceded its referent and the request the model then
            # generated from ended on its own assistant turn rather than on a
            # user message. That is the stall-rescue path — the dominant
            # failure the observer exists for — and the persisted session
            # kept the same shape. The echo-guard re-prompt below always
            # appended after the assistant message and got this right.
            keep_reasoning = int(
                getattr(options, "preserve_thinking_iterations", 0) or 0
            )
            chat_messages.append(_assistant_message_for_history(
                text=assistant_text, tool_calls=tool_calls_committed,
                reasoning=thinking_text if keep_reasoning > 0 else "",
            ))
            if keep_reasoning > 0:
                _prune_reasoning(chat_messages, keep=keep_reasoning)

            # Snapshot chat_messages length before firing OnEvent. The
            # observer may append a user message ("inject" lever); if it
            # does AND the model is otherwise about to terminate this turn
            # (no tool calls), we continue the loop so the inject takes
            # effect on the next iteration instead of being lost.
            chat_msgs_len_before_hook = len(chat_messages)
            if options.hooks is not None:
                await options.hooks.fire_on_event(asst_evt)
            observer_injected = len(chat_messages) > chat_msgs_len_before_hook

            if not tool_calls_committed:
                if observer_injected:
                    # Observer injected a system message. Continue the loop
                    # so the model gets to read it and respond.
                    logger.info(
                        "loop: observer injected on terminal iteration — continuing loop",
                    )
                    continue
                # Echo guard — the model printed a shell command in a fenced
                # block but called no tool. If Bash is available and we haven't
                # already nudged this turn, append a user-role nudge and loop
                # once more so it can actually call the tool (or confirm it was
                # only showing the command). A user message is used rather than
                # a second system message because vLLM chat templates reject a
                # non-leading system role.
                if (
                    getattr(options, "echo_guard_enabled", True)
                    and echo_guard_reprompts < _MAX_ECHO_GUARD_REPROMPTS
                    and _looks_like_unexecuted_command(assistant_text)
                    and "Bash" not in current_disallowed
                ):
                    echo_guard_reprompts += 1
                    chat_messages.append({"role": "user", "content": _ECHO_GUARD_NUDGE})
                    logger.info(
                        "loop: echo-guard re-prompt #%d — assistant emitted a shell "
                        "fence with no tool call (iter=%d)",
                        echo_guard_reprompts, num_turns,
                    )
                    continue
                stop_reason = finish_reason or "stop"
                break

            # Dispatch each tool call; accumulate results so we can
            # append them to history before looping back.
            #
            # Where this batch's messages begin. A hook can append to
            # `chat_messages` while the batch is still running — Inner
            # Voice's pretool inject does exactly that — and an append that
            # lands between two tool messages leaves
            # `assistant(tool_calls) → user → tool`, which is not a shape
            # any engine accepts. `_reorder_batch_messages` below puts the
            # slice back in wire order once the batch is done.
            batch_base = len(chat_messages)

            run_parallel = (
                getattr(options, "parallel_tool_calls_enabled", False)
                and len(tool_calls_committed) > 1
                and _batch_is_read_only(tool_calls_committed, _read_only_names(pool))
            )

            if run_parallel:
                # Phase 1, wire order: announce every call and run the gates
                # that must stay ordered (ToolSearch's mark_loaded mutates the
                # shared LoadedToolSet; a hook deny is a decision the model
                # must see in the order it made the calls).
                early: dict[str, NormalizedEvent] = {}
                for tc in tool_calls_committed:
                    tc_evt = events.tool_call(
                        call_id=tc["id"],
                        name=tc["function"]["name"],
                        args_json=tc["function"]["arguments"],
                        args_dict=tc["_args_dict"],
                        summary=tc.get("_summary", ""),
                    )
                    yield tc_evt
                    if options.hooks is not None:
                        await options.hooks.fire_on_event(tc_evt)
                    pre = await _pre_dispatch(
                        tc=tc, options=options, session_id=session_id,
                        loaded_set=loaded_set,
                        runtime_disallowed=current_disallowed,
                    )
                    if pre is not None:
                        early[tc["id"]] = pre

                caption_total, caption_present, caption_nudged, nudge_id = \
                    _account_captions(tool_calls_committed, summary_tools,
                                      caption_total, caption_present,
                                      caption_nudged, session_id, num_turns)

                pending = [tc for tc in tool_calls_committed
                           if tc["id"] not in early]
                results: dict[str, NormalizedEvent] = dict(early)

                # Phase 2: overlap the MCP calls. No TaskGroup — it cancels
                # siblings on the first exception, and one tool failing is a
                # tool_result, not a reason to abandon the batch.
                sem = asyncio.Semaphore(
                    max(1, int(getattr(
                        options, "parallel_tool_calls_max_concurrency", 4))))

                async def _run_one(call: dict[str, Any]) -> NormalizedEvent:
                    async with sem:
                        try:
                            return await _execute_tool_call(
                                tc=call, pool=pool, options=options,
                                session_id=session_id,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:      # pragma: no cover
                            logger.exception(
                                "loop: parallel dispatch failed on %s",
                                call["function"]["name"])
                            return events.tool_result(
                                call_id=call["id"],
                                name=call["function"]["name"],
                                content=f"Tool dispatch failed: {exc}",
                                is_error=True,
                            )

                tasks = [asyncio.create_task(_run_one(tc)) for tc in pending]
                try:
                    for evt_id in early:
                        out = early[evt_id]
                        if evt_id == nudge_id:
                            out = _with_caption_nudge(out)
                        yield out
                        if options.hooks is not None:
                            await options.hooks.fire_on_event(out)
                    # Yielded as they land: the frontend and messages.py key
                    # on call_id, so completion order costs nothing there and
                    # buys the user seeing the first answer sooner.
                    for fut in asyncio.as_completed(tasks):
                        result_evt = await fut
                        results[result_evt["call_id"]] = result_evt
                        if result_evt["call_id"] == nudge_id:
                            result_evt = _with_caption_nudge(result_evt)
                            results[result_evt["call_id"]] = result_evt
                        yield result_evt
                        if options.hooks is not None:
                            await options.hooks.fire_on_event(result_evt)
                finally:
                    # The consumer can close this generator mid-batch (a Stop
                    # click, a disconnect). Leaving these running would keep
                    # dispatching tools for a turn nobody is reading.
                    for t in tasks:
                        if not t.done():
                            t.cancel()

                # Phase 3: history in WIRE order regardless of who finished
                # first, so the replayed conversation matches the assistant
                # message's own tool_calls array.
                for tc in tool_calls_committed:
                    evt = results.get(tc["id"])
                    if evt is None:
                        continue
                    chat_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": evt["content"],
                    })
                _reorder_batch_messages(chat_messages, batch_base)
                tool_calls_committed_done = True
            else:
                tool_calls_committed_done = False

            for tc in ([] if tool_calls_committed_done else tool_calls_committed):
                tc_evt = events.tool_call(
                    call_id=tc["id"],
                    name=tc["function"]["name"],
                    args_json=tc["function"]["arguments"],
                    args_dict=tc["_args_dict"],
                    summary=tc.get("_summary", ""),
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

                # The caption ratchet. Only tools whose schema actually asked
                # for a `summary` are counted; `session_inject_context` has a
                # real one of its own and is not in `summary_tools`.
                if tc["function"]["name"] in summary_tools:
                    caption_total += 1
                    if tc.get("_summary"):
                        caption_present += 1
                    elif not caption_nudged:
                        # Once per turn, on the FIRST miss, because the first
                        # miss is what decides the session: `arguments` is
                        # replayed as history, so an uncaptioned call becomes
                        # the model's own most recent example of calling that
                        # tool. Session 20260907_184351_ivec8d shows the whole
                        # arc — 5/5 captioned on first use of each tool, 0/31
                        # after. Correcting it at call two is cheap; at call
                        # thirty the example has been reinforced thirty times.
                        caption_nudged = True
                        result_evt = _with_caption_nudge(result_evt)
                        logger.info(
                            "loop: %s dispatched with no summary — nudged once "
                            "(session=%s, iteration=%d)",
                            tc["function"]["name"], session_id, num_turns,
                        )
                yield result_evt
                if options.hooks is not None:
                    await options.hooks.fire_on_event(result_evt)

                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_evt["content"],
                })

            if not tool_calls_committed_done:
                _reorder_batch_messages(chat_messages, batch_base)

            # Mid-turn microcompaction. After this iteration's tool calls
            # land, clear stale tool results IF the prompt is actually
            # pressing on the context window. Mutates `chat_messages` in
            # place so the observer's chat_messages_handle stays pointing
            # at the same list.
            #
            # The tool-count threshold is a cheap pre-check only. Until
            # 2026-09-05 it was the entire trigger, so a turn that ran 70
            # tool calls at 40% of its context window was held to 5 inline
            # results the whole way, and everything it had read was gone.
            if (
                tool_calls_committed
                and getattr(options, "intra_turn_microcompact_enabled", True)
            ):
                threshold = getattr(options, "intra_turn_microcompact_threshold", 15)
                keep = getattr(options, "intra_turn_microcompact_keep_recent", 15)
                tool_count = sum(1 for m in chat_messages if m.get("role") == "tool")
                if tool_count >= threshold:
                    _intra_turn_microcompact(
                        chat_messages,
                        options=options,
                        total_usage=total_usage,
                        keep_recent=keep,
                        tool_count=tool_count,
                        iteration=num_turns,
                    )

        structured, structured_error = await _maybe_finalize(
            options=options,
            stop_reason=stop_reason,
            chat_messages=chat_messages,
            tools=last_visible_tools,
            total_usage=total_usage,
        )

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        if caption_total:
            logger.info(
                "loop: tool captions %d/%d (%d%%) session=%s",
                caption_present, caption_total,
                round(100 * caption_present / caption_total), session_id,
            )
        result_done_evt = events.result(
            stop_reason=stop_reason,
            usage=total_usage,
            num_turns=num_turns,
            duration_ms=duration_ms,
            response_text=accumulated_text,
            tool_calls_total=caption_total,
            tool_calls_captioned=caption_present,
            structured=structured,
            structured_error=structured_error,
        )
        yield result_done_evt
        if options.hooks is not None:
            await options.hooks.fire_on_event(result_done_evt)
    finally:
        # Pool is shared across turns — see comment above _build_pool call.
        _run_finished()


async def _maybe_finalize(
    *,
    options: RunOptions,
    stop_reason: str,
    chat_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    total_usage: dict[str, int],
) -> tuple[dict | None, str]:
    """Restate the finished turn under `options.final_schema`, if asked.

    Lives inside `run_query` rather than in the router because with Inner
    Voice off, `chat_messages` is private to this function — and the exact
    `tools` array that was last sent is too. Both are required: see
    `app/harness/finalizer.py` on why a different tools array re-prefills
    the whole conversation.

    Never raises. A caller asking for structure keeps whatever fallback it
    had; a turn is not worth failing over its transcription.
    """
    from app.harness.finalizer import run_finalizer, should_finalize

    run, skip_reason = should_finalize(stop_reason, options.final_schema)
    if not run:
        return None, skip_reason

    try:
        parsed, error, usage = await run_finalizer(
            base_url=options.base_url,
            model=options.model,
            chat_messages=chat_messages,
            tools=tools,
            schema=options.final_schema,
            prompt=options.final_schema_prompt,
            api_key=options.api_key,
            max_tokens=options.finalizer_max_tokens,
            timeout_s=options.finalizer_timeout_s,
            priority=options.priority,
            cancel_event=options.cancel_event,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("loop: finalizer raised: %s", exc)
        return None, f"finalizer failed: {type(exc).__name__}: {exc}"

    # The extra completion's tokens are real tokens; fold them in so usage
    # accounting does not quietly under-report every worker verdict.
    for key in ("output_tokens", "total_tokens"):
        if usage.get(key):
            total_usage[key] = total_usage.get(key, 0) + usage[key]
    return parsed, error


# ---------------------------------------------------------------------------
# In-flight accounting
# ---------------------------------------------------------------------------
#
# Process-wide count of agent loops currently running, for the ONE consumer
# that needs it: the self-modification promoter's idle gate.
#
# `sessions_io.active_turn_summary()` walks `_session_queues`, which is the
# right answer for chat turns and blind to every other caller of `run_query` —
# worker jobs (`workers/sources/_common.py::run_prompt_on_primary`), the IDE
# routes, post-session capture. Those turns can run for minutes, and a landing
# that restarts the backend underneath one kills it mid-flight; the connection
# errors it then logs land squarely in the window the error-rate detector is
# watching, so the promotion is blamed for the damage its own landing did.
# That is the exact signature of the 2026-09-06 20:14 false positive:
# `failed domain-research/research: ConnectError` x9.
#
# Deliberately a plain int, not a lock or a registry: it is read by a health
# endpoint polled every 2 seconds, and a torn read costs the promoter one more
# poll. asyncio runs these in one thread anyway.
_active_runs = 0


def _run_started() -> None:
    global _active_runs
    _active_runs += 1


def _run_finished() -> None:
    global _active_runs
    _active_runs = max(0, _active_runs - 1)


def active_run_count() -> int:
    """Agent loops in flight in this process, by any caller."""
    return _active_runs


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
        cfg = DEFAULT_LLOYD_MCP_SERVERS
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
    # Prefix-cache hits arrive NESTED, and the int-only loop above skipped the
    # dict that carries them — so `cache_read` was absent from every usage
    # block this harness ever produced, and `messages.py` recorded a hardcoded
    # zero on top of that. Every session on 2026-09-08 reported `cache_read: 0`
    # for every iteration, which is not evidence of a cold cache: nothing ever
    # read the number. It matters because the position-0 rule (system prompt
    # built once, loop only appends) exists precisely to keep the prefix
    # cached across a long turn, and this is the only signal that says whether
    # it is working.
    details = chunk.get("prompt_tokens_details")
    if isinstance(details, dict):
        if isinstance(cached := details.get("cached_tokens"), int):
            out["cache_read"] = cached
        if isinstance(created := details.get("created_cache_tokens"), int):
            out["cache_create"] = created
    return out


def _intra_turn_microcompact(
    chat_messages: list[dict],
    *,
    options: Any,
    total_usage: dict[str, int],
    keep_recent: int,
    tool_count: int,
    iteration: int,
) -> None:
    """Clear stale tool results in place, but only under real pressure.

    Uses the same threshold arithmetic as the turn-start pass in
    `app.compaction`, so the two cannot disagree about where the wall is.
    Imports it lazily: `app.compaction` already lazy-imports this package
    in the other direction, and a module-level edge here would close the
    cycle.

    Context size anchors on vLLM's reported `input_tokens` — the real size
    of the prompt the server just processed — while the estimator supplies
    the per-message deltas. The gap between them is the system prompt and
    tool schemas, which `chat_messages` does not contain; carrying it as an
    `offset` keeps both halves in the same units. Triggering on the real
    figure while budgeting against the estimate would mean triggering and
    then clearing nothing, since the estimate is always the smaller number.
    """
    try:
        from app.compaction import (
            estimate_conversation_tokens,
            get_context_window,
            truncation_threshold,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("loop: intra-turn microcompact unavailable: %s", e)
        return

    threshold = truncation_threshold(get_context_window(getattr(options, "model", "")))
    trigger = int(threshold * float(
        getattr(options, "intra_turn_microcompact_trigger_fraction", 0.8)
    ))
    target = int(threshold * float(
        getattr(options, "intra_turn_microcompact_target_fraction", 0.6)
    ))

    def _estimate(msgs: list[dict]) -> int:
        return estimate_conversation_tokens(msgs, "")

    estimated = _estimate(chat_messages)
    reported = int(total_usage.get("input_tokens", 0) or 0)
    # Everything in the prompt that isn't in `chat_messages`. Clamped at 0
    # so an over-reporting estimate can't invent headroom.
    offset = max(0, reported - estimated)
    current = estimated + offset
    if current <= trigger:
        return

    compacted, cleared = _intra_microcompact(
        chat_messages,
        keep_recent_tools=keep_recent,
        # Budget in estimator units: the offset is fixed cost this pass
        # cannot reduce. If it alone exceeds the target, clear down to the
        # floor and let context-overflow recovery handle the remainder.
        token_budget=max(1, target - offset),
        estimate_fn=_estimate,
        min_chars_to_clear=int(
            getattr(options, "intra_turn_microcompact_min_chars", 2_000)
        ),
        session_id=getattr(options, "session_id", "") or "",
        legacy_count_rule=False,
    )
    if cleared:
        chat_messages[:] = compacted
        logger.info(
            "loop: intra-turn microcompact cleared %d/%d tool results "
            "(kept last %d, %d -> %d tokens, target %d, iter=%d)",
            cleared, tool_count, keep_recent, current,
            _estimate(chat_messages) + offset, target, iteration,
        )


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


# One line, appended to the result of the first uncaptioned tool call in a
# turn. It is phrased as an instruction for the NEXT call rather than a
# complaint about this one, because the model cannot edit a call it already
# made — and the point is to reset the example before the ratchet sets.
_CAPTION_NUDGE = (
    "\n\n[harness] That call carried no `summary`. Every tool takes a short "
    "`summary` as its first argument — a phrase like \"Reading server.py\" or "
    "\"Restarting the backend\" — and it is what the transcript shows in place "
    "of a bare tool name. Include one on every call from here on."
)


def _with_caption_nudge(result_evt: dict[str, Any]) -> dict[str, Any]:
    """Append the caption reminder to a tool_result event's content.

    A copy, not a mutation: the same dict is handed to `fire_on_event` hooks
    and appended to history, and an in-place edit would have the nudge land in
    the observer's view of the tool result as though the tool had said it.
    """
    out = dict(result_evt)
    out["content"] = f"{out.get('content', '')}{_CAPTION_NUDGE}"
    return out


def _commit_tool_calls(
    acc: dict[int, dict[str, Any]],
    *,
    summary_tools: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Finalize accumulated tool calls.

    Parses `arguments` JSON; on failure, attaches an `_args_dict` of
    `{"__parse_error__": True, "raw": "..."}` so the dispatcher can
    surface a tool_result with `is_error=True` instead of crashing.

    `summary_tools` is the set of tools whose advertised schema carried the
    injected `summary` display parameter (see
    ``tool_schema.add_summary_param``). For those, the value is lifted onto
    `_summary` and removed from `_args_dict` — the aggregator validates
    arguments against the tool's real inputSchema, so an extra key there is
    a dispatch error rather than a spare field. It is deliberately NOT
    removed from the `arguments` string: that string is what gets replayed
    to the engine as history, and it is the only record of this call the
    model will see again. Tools outside the set are left completely alone;
    ``session_inject_context`` has its own required `summary` and popping it
    would delete a real argument.
    """
    summary_tools = summary_tools or set()
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
        if (
            tc["function"]["name"] in summary_tools
            and not args_dict.get("__parse_error__")
        ):
            # Popped from `_args_dict` (what reaches MCP) and NOT from
            # `arguments` (what is replayed back as history). The two
            # deliberately disagree, and the disagreement is the whole
            # mechanism: `arguments` is the only record of this call the
            # model will ever see again, so stripping the caption there
            # taught it — within one turn — that calls of this tool do
            # not carry one. Measured on session 20260907_184351_ivec8d:
            # the FIRST call of each tool name had a summary and every
            # repeat had none, 5/5 vs 0/31. The schema said required; the
            # model's own most recent example said otherwise, and the
            # example won. It costs ~10 tokens per historical tool call
            # to keep, which is the price of the field working at all.
            summary = pop_summary(args_dict)
            if summary:
                tc["_summary"] = summary
        tc["_args_dict"] = args_dict
        committed.append(tc)
    return committed


def _assistant_message_for_history(
    *, text: str, tool_calls: list[dict[str, Any]], reasoning: str = "",
) -> dict[str, Any]:
    """Build the OpenAI assistant message we append back to chat_messages
    for the next loop iteration. Strips the `_args_dict` helper so what
    we send to vLLM is spec-compliant.

    ``reasoning`` carries this iteration's thinking back into history.
    Qwen3.8-Flash-Next renders it into the `<think>` block of each prior
    assistant turn ("preserved thinking", on by default in the chat
    template) and its model card calls that out as reducing redundant
    reasoning in agent loops. Dropping it — which the harness did until
    2026-09-05 — showed the model 50+ prior turns in which it had
    apparently thought nothing, and made it re-derive its own conclusions
    every iteration.

    Both `reasoning` and `reasoning_content` are set, because the two
    engines behind this harness disagree about which one is real:

      * vLLM 0.28 accepts both on the wire but only populates the template
        from `reasoning` (entrypoints/chat_utils.py:2000). Sending
        `reasoning_content` alone is silently dropped and renders an empty
        `<think>` block.
      * llama.cpp applies the model's jinja template directly, and
        Qwen3.6's reads `message.reasoning_content` (chat_template.jinja:91)
        — it never looks at `reasoning`.

    So sending only one field breaks preserved thinking on whichever
    engine is not vLLM, in exactly the silent way this whole mechanism
    exists to prevent: the model sees prior turns in which it apparently
    thought nothing. The secondary slot became llama.cpp on 2026-09-06
    (Qwen3.6-35B-A3B GGUF), which is when this stopped being academic.
    """
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        msg["reasoning"] = reasoning
        msg["reasoning_content"] = reasoning
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


def _prune_reasoning(chat_messages: list[dict[str, Any]], *, keep: int) -> None:
    """Keep `reasoning` on the most recent `keep` assistant messages only.

    Preserved thinking is not free: the review turn on
    20260905_151355_iv5174 generated 66.6k reasoning tokens, so carrying
    all of it would have pushed peak input from 162k past the 209k
    microcompaction trigger toward the 262k ceiling. Microcompaction only
    knows how to clear tool results, so unbounded reasoning would have no
    relief valve. Bounding it to a recent window keeps the useful part —
    continuity with what the model was just doing — at a fixed cost.

    Mutates in place. Older assistant turns keep their text and tool
    calls and simply render an empty `<think>` block, exactly as every
    turn did before preserved thinking was wired up.

    Both spellings go together — `_assistant_message_for_history` writes
    the pair (vLLM reads one, llama.cpp's Qwen3.6 template reads the
    other), so dropping only `reasoning` would leave the full reasoning
    still on the wire for llama.cpp and defeat the bound entirely.
    """
    seen = 0
    for msg in reversed(chat_messages):
        if msg.get("role") != "assistant":
            continue
        if "reasoning" not in msg and "reasoning_content" not in msg:
            continue
        seen += 1
        if seen > keep:
            msg.pop("reasoning", None)
            msg.pop("reasoning_content", None)


def _account_captions(tool_calls: list[dict[str, Any]], summary_tools: set[str],
                      total: int, present: int, nudged: bool,
                      session_id: str, iteration: int
                      ) -> tuple[int, int, bool, str]:
    """Caption bookkeeping for a batch, in WIRE order.

    The ratchet is about the FIRST miss, not the first to come back:
    `arguments` is replayed as history, so an uncaptioned call becomes the
    model's own most recent example of calling that tool. In a concurrent
    batch the first *completed* call is arbitrary, so this walks the calls in
    the order the model made them and returns which id should carry the
    nudge. See `events.result`.
    """
    nudge_id = ""
    for tc in tool_calls:
        if tc["function"]["name"] not in summary_tools:
            continue
        total += 1
        if tc.get("_summary"):
            present += 1
        elif not nudged:
            nudged = True
            nudge_id = tc["id"]
            logger.info(
                "loop: %s dispatched with no summary — nudged once "
                "(session=%s, iteration=%d)",
                tc["function"]["name"], session_id, iteration,
            )
    return total, present, nudged, nudge_id


def _read_only_names(pool: MCPPool) -> set[str]:
    """Every discovered tool whose server declares `readOnlyHint`.

    Asked of the server, not of a list kept here. `agent_mcp/annotations.py`
    is the declared table for lloyd-mcp and a server that sets no hints
    contributes nothing — which is that file's stated contract, and the
    reason this is a safe default rather than a permissive one.
    """
    names: set[str] = set()
    for _server, tools in pool.discovered:
        for tool in tools:
            ann = tool.get("annotations") or {}
            if ann.get("readOnlyHint"):
                names.add(tool["name"])
    return names


def _batch_is_read_only(tool_calls: list[dict[str, Any]],
                        read_only: set[str]) -> bool:
    """True when every call in the batch is safe to overlap with the others.

    A parse error never dispatches, and ToolSearch is intercepted in-process
    and touches only the LoadedToolSet, which `_pre_dispatch` mutates in wire
    order anyway. Everything else must carry the hint. One writer makes the
    whole batch sequential: two Edits to one file, or an Edit racing the Read
    that justifies it, is not a reordering anybody asked for.
    """
    for tc in tool_calls:
        if tc["_args_dict"].get("__parse_error__"):
            continue
        name = tc["function"]["name"]
        if name == TOOLSEARCH_TOOL_NAME or name in read_only:
            continue
        return False
    return True


def _reorder_batch_messages(chat_messages: list[dict[str, Any]], base: int) -> None:
    """Put a batch's messages back in wire order, in place.

    An `assistant` message with tool calls must be followed by ONE `tool`
    message per call before anything else. A hook that appends mid-batch —
    Inner Voice's pretool inject appends a `user` message
    (`app/inner_voice/observer.py`) — breaks that for every call after the
    first, producing `assistant(tool_calls) → user → tool`.

    Nothing caught it because the inject is rare, the engine's failure is a
    400 the loop reports as a stream error, and the sequential path only
    produces it when an inject fires between two calls of a multi-call
    batch. Concurrent dispatch would make it routine.

    Tool messages keep their order and so do the injects; only the boundary
    moves. In-place because `chat_messages` may be the observer's own
    handle, and rebinding it would leave the observer holding the old list.
    """
    tail = chat_messages[base:]
    if len(tail) < 2:
        return
    tools = [m for m in tail if m.get("role") == "tool"]
    others = [m for m in tail if m.get("role") != "tool"]
    if not others or not tools:
        return
    if tail == tools + others:
        return
    del chat_messages[base:]
    chat_messages.extend(tools + others)


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

    Composed of two halves, and the split is the point: `_pre_dispatch`
    decides whether the call happens at all (parse error, disabled tool,
    ToolSearch intercept, hook deny) and must run in wire order for every
    call in a batch; `_execute_tool_call` is the part that can safely run
    concurrently with its siblings. Behaviour here is unchanged — this
    function still does both, in order, for one call.

    `runtime_disallowed` (Plan B) — the loop's per-iteration disallowed
    set, computed via `options.disallowed_tools_refresh` if set. When
    None, falls back to `options.disallowed_tools` (the static turn-start
    list). Either way, this is the gate that blocks dispatch.
    """
    early = await _pre_dispatch(
        tc=tc, options=options, session_id=session_id,
        loaded_set=loaded_set, runtime_disallowed=runtime_disallowed,
    )
    if early is not None:
        return early
    return await _execute_tool_call(
        tc=tc, pool=pool, options=options, session_id=session_id,
    )


async def _pre_dispatch(
    *,
    tc: dict[str, Any],
    options: RunOptions,
    session_id: str,
    loaded_set: LoadedToolSet,
    runtime_disallowed: set[str] | None = None,
) -> NormalizedEvent | None:
    """Everything before the MCP call. Returns an early result, or None.

    Kept ordered and sequential even in a parallel batch: `mark_loaded`
    mutates the shared LoadedToolSet, and a hook deny is a decision the
    model must see in the order it made the calls.
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

    # Soft gate: if the model calls a deferred tool that's in the catalog
    # but not yet loaded, treat that as an implicit ToolSearch and proceed.
    # Rationale: a strict reject made parallel batches pathological — N
    # parallel calls to the same unloaded tool would each independently
    # fail with the same "call ToolSearch first" error, wasting N-1
    # dispatches before the model could recover. ToolSearch is now a
    # context-optimization hint (load schemas in advance to keep prompts
    # slim), not a hard gate. Truly unknown names (not in the catalog at
    # all) still fall through to MCP, which returns its own clean
    # "tool not found" error.
    if loaded_set.enabled and not loaded_set.is_visible(name):
        catalog_names = {t["function"]["name"] for t in loaded_set.catalog}
        if name in catalog_names:
            loaded_set.mark_loaded([name])
            logger.info(
                "loop: auto-loaded schema for %r on direct call (no prior "
                "ToolSearch); loaded set now %d.",
                name, len(loaded_set.loaded),
            )

    # PreToolUse — first deny wins.
    if options.hooks is not None:
        deny = await options.hooks.fire_pre_tool_use(
            session_id=session_id,
            tool_name=name,
            tool_input=args_dict,
            tool_use_id=call_id,
            tool_summary=tc.get("_summary", ""),
        )
        if deny:
            hso = deny.get("hookSpecificOutput") or {}
            reason = hso.get("permissionDecisionReason") or "denied by hook"
            return events.tool_result(
                call_id=call_id, name=name,
                content=f"Tool call denied: {reason}", is_error=True,
            )

    return None


async def _execute_tool_call(
    *,
    tc: dict[str, Any],
    pool: MCPPool,
    options: RunOptions,
    session_id: str,
) -> NormalizedEvent:
    """The MCP call and everything after it. Safe to run concurrently.

    Session correlation rides in the request's `_meta` (see
    MCPPool.call_tool), not in the arguments — the MCP server validates
    arguments against the tool's inputSchema before its handler runs, so
    an injected argument is validated as if it were a real parameter.
    """
    name = tc["function"]["name"]
    args_dict = tc["_args_dict"]
    call_id = tc["id"]
    dispatch_args = dict(args_dict)
    try:
        # Race the MCP call against options.cancel_event so an in-flight
        # tool (long-running Bash, slow MCP server) doesn't block the
        # loop's reaction to a Stop click. On cancel we cancel the
        # call_tool task — the SSE client closes the request, the MCP
        # server-side coroutine receives CancelledError, and individual
        # tools (e.g. Bash) are responsible for tearing down child
        # processes in their own finally clauses.
        cancel_event = getattr(options, "cancel_event", None)
        # `model` / `base_url` let a Task subagent inherit the calling
        # turn's model instead of falling back to `primary`.
        #
        # `summary` is the model's own caption for this call, already
        # popped off `args_dict`. It rides in `_meta` so the two tools
        # that open a row someone later reads — background Bash and Task
        # — can label it without asking the model for the same sentence a
        # second time under a different key. It is deliberately absent
        # from `dispatch_args`: `tool_input` is what the repetition guard
        # hashes, and a reworded caption there would make two identical
        # calls compare as different.
        call_kw = {
            "session_id": session_id,
            "model": options.model,
            "base_url": options.base_url,
            "summary": tc.get("_summary", ""),
            # Which turn and which call this write belongs to, for the
            # change ledger on the far side. Empty for a worker or a direct
            # run_query caller, which is what turns the ledger off for them.
            "turn_id": getattr(options, "turn_id", "") or "",
            "call_id": tc.get("id", "") or "",
        }
        if cancel_event is None:
            result = await pool.call_tool(name, dispatch_args, **call_kw)
        else:
            tool_task = asyncio.create_task(
                pool.call_tool(name, dispatch_args, **call_kw)
            )
            cancel_task = asyncio.create_task(cancel_event.wait())
            try:
                done, _pending = await asyncio.wait(
                    {tool_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not cancel_task.done():
                    cancel_task.cancel()
            if tool_task in done:
                result = tool_task.result()
            else:
                tool_task.cancel()
                try:
                    await tool_task
                except (asyncio.CancelledError, Exception):
                    pass
                logger.info(
                    "loop: tool %s cancelled mid-dispatch (cancel_event set)",
                    name,
                )
                return events.tool_result(
                    call_id=call_id, name=name,
                    content=f"Tool {name!r} cancelled by user.",
                    is_error=True,
                )
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
    options: RunOptions,
    catalog: list[dict[str, Any]],
    *,
    summaries: bool = False,
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
        summaries=summaries,
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
    body = format_catalog_reminder(loaded_set.catalog, loaded=loaded_set.loaded)
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
