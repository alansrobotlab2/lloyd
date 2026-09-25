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
import collections
import contextvars
import json
import logging
import re
import time
import uuid
from contextlib import aclosing
from typing import Any, AsyncIterator

import httpx

from app.deadline_anchor import ANCHOR_TAG
from app.harness import events
from app.harness.client import stream_chat
from app.harness.context_meter import ContextMeter, context_window_for
from app.harness.errors import (
    ContextOverflowError,
    MultimodalRejectedError,
    ParseError,
    StreamStalledError,
    ToolDiscoveryError,
    ToolDispatchError,
)
from app.harness.microcompact import microcompact as _intra_microcompact
from app.harness.tool_result_spill import (
    READ_TOOL,
    fallback_for_empty_result,
    maybe_spill,
    tool_is_denied,
)
from app.harness.events import NormalizedEvent
from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS, MCPPool, get_or_open_pool
from app.harness.options import RunOptions
from app.harness.policy import (
    current_effect_scope, current_scope, normalize_tool_name,
)
from app.harness.telemetry import log_harness_event
from app.harness.turn_state import Iteration, TurnState
from app.harness.tool_schema import (
    add_summary_param,
    build_tool_list,
    pop_summary,
)
from app.harness import rpc_policy, tool_search_cache
from app.harness.tool_search import (
    LoadedToolSet,
    TOOLSEARCH_TOOL_NAME,
    format_catalog_reminder,
    search_tools,
)
from app.thinking_fidelity import matched_marker

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

# `harness.echo_guard.mode: tool_choice` (P6a) answers the same trip without
# the nudge: the attempt is discarded and the identical request goes out again
# with `tool_choice: "required"`. Ships as "nudge" until `"required"` is
# measured against the qwen3_xml parser — vLLM satisfies it with a JSON
# grammar, not the model's own XML tool format.


# ---------------------------------------------------------------------------
# Max-turns wrap-up (P6b)
# ---------------------------------------------------------------------------
#
# A run that reaches `max_turns` used to end on whatever its last iteration
# said — usually the preamble of a tool call it never got to make. When the
# engine is one verified to honour `tool_choice: "none"` with the tools array
# present (vLLM with qwen3_xml; `finalizer.py` sends the same request shape
# for the same reason), the loop asks once more with tools off. The tools
# array is identical, so the prompt is a cache hit plus one user message.

_MAX_TURNS_WRAPUP_PROMPT = (
    "You have used all {n} iterations and cannot call tools now. In a few "
    "sentences: what was done, what was not, where the work stands and the "
    "next step."
)


def _wrapup_applies(options: RunOptions) -> bool:
    """True when this run may spend one toolless request at its budget.

    Per call, by engine: a llama.cpp slot is not verified to honour
    `tool_choice: "none"`, and a Task child or a turn on another slot carries
    its own `base_url`.
    """
    if not getattr(options, "max_turns_wrapup", False):
        return False
    if int(options.max_turns or 0) <= 0:
        return False
    allowed = {str(u).rstrip("/") for u in
               (getattr(options, "max_turns_wrapup_base_urls", ()) or ())}
    return str(options.base_url or "").rstrip("/") in allowed


def _looks_like_unexecuted_command(text: str) -> bool:
    """True if `text` contains a fenced shell block (a likely echoed command)."""
    if not text:
        return False
    return bool(_EXEC_FENCE_RE.search(text))


# ---------------------------------------------------------------------------
# Broken streams (review 2026-09-24, D7)
# ---------------------------------------------------------------------------
#
# A stream that dies mid-generation used to take one of two wrong exits: a
# malformed SSE line was finalized as `finish_reason="stop"` with whatever
# half-parsed tool calls had accumulated (and those were dispatched), and a
# stall, dropped connection or 5xx raised out of the turn. Both are cheap to
# retry while nothing was dispatched — the prefix is still cached, and the only
# thing lost is the partial text the consumers take back off their buffers on
# `iteration_retry`.

_BROKEN_STREAM_ERRORS = (
    ParseError,
    StreamStalledError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.HTTPStatusError,
)


def _stream_error_reason(exc: BaseException) -> str:
    """The `iteration_retry` / `harness.stream_retried` reason for `exc`."""
    if isinstance(exc, ParseError):
        return "parse_error"
    if isinstance(exc, StreamStalledError):
        return "stream_stalled"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    return "transport"


def _is_client_error(exc: BaseException) -> bool:
    """A 4xx is the request's fault; sending it again gets the same answer."""
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code < 500
    )


async def _retry_backoff(seconds: float, cancel_event: asyncio.Event | None) -> None:
    """Sleep before a retry, waking at once on Stop."""
    if seconds <= 0:
        return
    if cancel_event is None:
        await asyncio.sleep(seconds)
        return
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _surface_hidden(options: RunOptions) -> set[str]:
    """The tools this turn's surface hides; empty when the table is unreadable.

    Failing open is deliberate: the surface is a catalog trim, not a security
    boundary (the refusals that are one live in the tools and the policy
    gate), so an import problem costs the trim and never the turn.
    """
    surface = getattr(options, "surface", "") or ""
    if not surface:
        return set()
    try:
        from agent_mcp.annotations import hidden_on_surface
    except Exception as exc:  # pragma: no cover - import failure path
        logger.warning("loop: tool surface table unavailable (%s)", exc)
        return set()
    return set(hidden_on_surface(surface))


def _allowed_names(options: RunOptions) -> set[str] | None:
    """The turn's allow-list, normalised, or None when it has none (P3)."""
    allowed = getattr(options, "allowed_tools", None)
    if allowed is None:
        return None
    return {normalize_tool_name(str(n)) for n in allowed}


def _allow_list_hidden(options: RunOptions, discovered) -> set[str]:
    """Every discovered tool an allow-list leaves out (P3).

    Joined to the surface's hidden set, so it reaches the catalog build AND
    every iteration's dispatch set, a plan-mode refresher's included. The
    explicit check in `_pre_dispatch` is the second half: a name nobody
    discovered would otherwise fall through to MCP.
    """
    allowed = _allowed_names(options)
    if allowed is None:
        return set()
    return {t["name"] for _srv, tools in discovered for t in tools
            if normalize_tool_name(t["name"]) not in allowed}


async def run_query(
    messages: list[dict[str, Any]],
    options: RunOptions,
) -> AsyncIterator[NormalizedEvent]:
    """Run one agent loop and yield NormalizedEvents.

    `messages` is an OpenAI-style message list — typically built by
    `app.compaction.load_and_compact_session(...)` plus the current
    user turn. When `options.chat_messages_handle` is a non-empty list the
    loop runs on that list and ignores `messages` (a resumed Task passes its
    stored history that way). The harness prepends `options.system_prompt`
    as the system message unless the list already carries one (position 0,
    inserted once).

    Cancellation: if `options.cancel_event` is set during streaming, the
    httpx context exits cleanly and the loop emits a final `result`
    event with `stop_reason="cancelled"`.

    The turn is a sequence of phases over one `TurnState` (P13.2):
    `_open_turn` once, then per iteration `_prelude` → `_stream_iteration`
    → (`_end_without_tools` | `_dispatch_batch` → `_after_batch`), and
    `_close_turn` once. Each phase that yields events is an async generator
    this function re-yields through `aclosing`, so a consumer that closes the
    turn mid-phase closes the phase too (the batch's outstanding tool calls
    are cancelled at once, not when the garbage collector gets to them).
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
    # Hooks that append (the turn guards' inject) must reach THIS list, which
    # is a private copy whenever the caller passed no handle.
    if options.hooks is not None and hasattr(options.hooks, "bind_run"):
        options.hooks.bind_run(chat_messages, options)

    # Live context-window position. Caller-owned when supplied — the router
    # hands the SAME object to the Inner Voice observer so both read one
    # figure — and private otherwise, which still gets a bare `run_query`
    # caller the relief ladder without an anchor it has no way to render.
    meter = options.context_meter
    if meter is None:
        meter = ContextMeter(context_window_for(options.model))

    pool = await _build_pool(options)
    # Pool is process-shared (see mcp_pool.get_or_open_pool); do NOT
    # aclose() it here — lifecycle.shutdown_cleanup tears down all
    # pools at FastAPI shutdown.
    #
    # The run counter goes up HERE, immediately inside the try whose
    # `finally` brings it down, and not at the top of the function. It used
    # to be incremented before `_build_pool`, so a pool that failed to open
    # raised past the try and leaked one count for the life of the process
    # — and `harness_runs` is what the promoter's idle wait reads. On
    # 2026-09-16 one such leak after the 05:02Z landing held the counter at
    # 1 over an idle backend, and nine gate-passed rounds in a row failed to
    # land with "backend never went idle within 900s" until the next restart.
    try:
        _run_started()
        st = await _open_turn(options, chat_messages, meter, pool, started_at)
        yield events.system(session_id=st.session_id, model=options.model)

        # Preserved-thinking window, enforced HERE at turn entry and nowhere
        # else (backlog #520). See `_cap_history_reasoning`.
        _cap_history_reasoning(chat_messages, keep=st.keep_reasoning)

        while True:
            it = await _prelude(st)
            if it is None:
                break
            async with aclosing(_stream_iteration(st, it)) as phase:
                async for evt in phase:
                    yield evt
            if it.flow == "continue":
                continue
            if it.flow == "break":
                break

            if not it.tool_calls_committed:
                async with aclosing(_end_without_tools(st, it)) as phase:
                    async for evt in phase:
                        yield evt
                if it.flow == "continue":
                    continue
                break

            async with aclosing(_dispatch_batch(st, it)) as phase:
                async for evt in phase:
                    yield evt
            _after_batch(st, it)

        async with aclosing(_close_turn(st)) as phase:
            async for evt in phase:
                yield evt
    finally:
        # Pool is shared across turns — see comment above _build_pool call.
        _run_finished()


# ---------------------------------------------------------------------------
# The phases of one turn (P13.2)
# ---------------------------------------------------------------------------
#
# Each phase reads and writes the turn's `TurnState` and the iteration's
# `Iteration` (app/harness/turn_state.py) — the locals `run_query` used to
# hold, moved and not re-derived. A phase that decides how the loop goes on
# says so in `it.flow`: "continue" is the old `continue`, "break" the old
# `break`, "" falls through to the next phase.


async def _open_turn(
    options: RunOptions,
    chat_messages: list[dict[str, Any]],
    meter: ContextMeter,
    pool: MCPPool,
    started_at: float,
) -> TurnState:
    """Build the catalog, refuse a toolless turn, and name the session."""
    # The surface's hidden tools join the disallowed set here, for what is
    # advertised, and in every iteration's dispatch set below, including a
    # plan-mode refresher's, which would otherwise rebuild it without them.
    surface_hidden = _surface_hidden(options)
    surface_hidden |= _allow_list_hidden(options, pool.discovered)
    catalog = build_tool_list(list(pool.discovered),
                              set(options.disallowed_tools) | surface_hidden)
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

    return TurnState(
        options=options,
        chat_messages=chat_messages,
        meter=meter,
        pool=pool,
        started_at=started_at,
        session_id=options.session_id or uuid.uuid4().hex,
        surface_hidden=surface_hidden,
        summary_tools=summary_tools,
        loaded_set=loaded_set,
        keep_reasoning=int(
            getattr(options, "preserve_thinking_iterations", 0) or 0
        ),
        # One relief pass per crossing of the target instead of one per
        # iteration (#800). Per turn: the turn owns the message list and
        # the engine's cached prefix for it, and a new turn invalidates
        # both anyway. This latch gates ONLY the per-iteration call —
        # while it is closed the prompt can still climb, and it does so
        # into the pre-request rung (headroom under
        # `context_relief_min_completion_tokens`), the terminal-inject
        # guard, and the context-overflow recovery, all of which run the
        # whole ladder with no latch and are what keep a latched turn
        # from dying at the wall.
        relief_latch=_ReliefLatch(),
    )


async def _prelude(st: TurnState) -> Iteration | None:
    """The head of an iteration, before its request. None ends the turn.

    Counts the iteration, applies the `max_turns` budget (and its one
    toolless wrap-up request), honours Stop, splices drained notifications
    and the state anchor (once per iteration number, D9), refreshes the
    disallowed set, and runs the pre-request relief floor.
    """
    options = st.options
    chat_messages = st.chat_messages
    meter = st.meter

    st.num_turns += 1
    if st.num_turns > options.max_turns:
        st.stop_reason = "max_turns"
        if (
            st.wrapup_state == ""
            and _wrapup_applies(options)
            and not (options.cancel_event is not None
                     and options.cancel_event.is_set())
        ):
            st.wrapup_state = "requested"
            chat_messages.append({
                "role": "user",
                "content": _MAX_TURNS_WRAPUP_PROMPT.format(
                    n=options.max_turns),
            })
            meter.observe_append(chat_messages)
            logger.info(
                "loop: max_turns=%d reached — one toolless wrap-up "
                "request (session=%s)", options.max_turns, st.session_id,
            )
        elif st.wrapup_state != "requested":
            return None
        # No drain, no anchor on the wrap-up request: the budget
        # anchor would repeat what the wrap-up message already says.
        st.prelude_done_for = st.num_turns

    if options.cancel_event is not None and options.cancel_event.is_set():
        st.stop_reason = "cancelled"
        return None

    # Background-task completion drain. Splices any pending
    # <task_notification> messages into chat_messages so the
    # model sees them on this iteration. The callback also
    # persists them into the session JSON so reconstruction on
    # subsequent turns stays consistent.
    prelude_due = st.prelude_done_for != st.num_turns
    st.prelude_done_for = st.num_turns
    if prelude_due and options.notification_drain is not None:
        try:
            drained = await options.notification_drain()
        except Exception as exc:
            logger.warning("loop: notification_drain raised: %s", exc)
            drained = []
        if drained:
            chat_messages.extend(drained)
            meter.observe_append(chat_messages)
            logger.info(
                "loop: drained %d background-task notification(s)", len(drained),
            )

    # Per-iteration state re-anchor (todos / plan / goal). Appended,
    # never merged into the system prompt: position 0 must stay
    # byte-stable or every iteration re-prefills the whole context.
    # These are NOT persisted — see RunOptions.state_anchor — so the
    # event each one writes here is the only record that it fired.
    if prelude_due and options.state_anchor is not None:
        try:
            anchors = await options.state_anchor(st.num_turns)
        except Exception as exc:
            logger.warning("loop: state_anchor raised: %s", exc)
            anchors = []
        if anchors:
            tags = [
                m.pop(ANCHOR_TAG, None) if isinstance(m, dict) else None
                for m in anchors
            ]
            _record_anchor_fires(
                tags, st.session_id, getattr(options, "turn_id", "") or "",
                st.num_turns,
            )
            chat_messages.extend(anchors)
            meter.observe_append(chat_messages)
            logger.info(
                "loop: state anchor re-injected %d message(s) (iter=%d)",
                len(anchors), st.num_turns,
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
    current_disallowed |= st.surface_hidden

    it = Iteration(started_at=time.perf_counter(),
                   current_disallowed=current_disallowed)

    # Held so the finalizer can send the IDENTICAL array. Qwen's
    # template renders `tools` inside the system message, so a
    # different (or absent) list changes the rendered prompt from the
    # first token and vLLM re-prefills the whole conversation.
    st.last_visible_tools = st.loaded_set.visible_tools(
        extra_disallowed=current_disallowed)
    # Same reason, one caller further out: a state-patch re-ask issued
    # after the segment ends is outside this function and cannot
    # rebuild the array (#529, RunOptions.visible_tools_capture).
    # Slice assignment keeps the caller's list the one it handed over.
    if options.visible_tools_capture is not None:
        options.visible_tools_capture[:] = st.last_visible_tools

    # Pre-request floor. A request sent with less room than the
    # completion needs comes back truncated mid-tool-call, and the
    # model's own next move is to re-send it — which is how 875's
    # completions shrank 3147 -> 1760 -> 1175 tokens against the
    # same heredoc. Relieve first, so the request has somewhere to
    # write.
    min_completion = int(
        getattr(options, "context_relief_min_completion_tokens", 6_000)
    )
    if (
        getattr(options, "context_relief_enabled", True)
        and meter.measured
        and meter.headroom < min_completion
    ):
        _relieve_context(
            chat_messages,
            options=options,
            meter=meter,
            reason="pre_request",
            keep_recent=int(getattr(
                options, "intra_turn_microcompact_keep_recent", 15)),
            iteration=st.num_turns,
        )
    return it


async def _stream_iteration(st: TurnState, it: Iteration):
    """One request: stream it, recover a broken or rejected one, and commit.

    Ends with the iteration's `assistant_message` yielded, appended to
    history and handed to the hooks. Sets `it.flow` to "continue" for a
    retry of the same iteration (D7 stream retry, context-overflow and
    multimodal recovery) and to "break" on Stop or once the wrap-up answer
    is in.
    """
    options = st.options
    chat_messages = st.chat_messages
    meter = st.meter
    session_id = st.session_id

    it.request_msgs_len = len(chat_messages)
    # P11: the request's own clock starts here, after the pre-request
    # relief above, so a relief pass never reads as a slow prefill.
    # `duration_ms` keeps covering the whole iteration.
    it.request_started_at = time.perf_counter()
    it.first_chunk_at = None
    # P6: a stream retry or an overflow recovery re-sends with the
    # same choice; it is consumed once the request completes, below.
    it.request_tool_choice = (
        "none" if st.wrapup_state == "requested"
        else (st.forced_tool_choice or "auto"))
    try:
        async for chunk in stream_chat(
            base_url=options.base_url,
            model=options.model,
            messages=chat_messages,
            tools=st.last_visible_tools,
            extra_body=options.extra_body,
            cancel_event=options.cancel_event,
            timeout_s=options.request_timeout_s,
            api_key=options.api_key,
            priority=options.priority,
            chunk_timeout_s=getattr(
                options, "stream_chunk_timeout_s", 0.0
            ),
            # #581: which turn and which iteration this line describes.
            # `num_turns` is the same value `_log` and
            # `prefix_miss.record_iteration` already use for the
            # iteration index, so a manifest line and that module's
            # per-iteration cache series name the same iteration.
            session_id=options.session_id,
            iteration=st.num_turns,
            tool_choice=it.request_tool_choice,
        ):
            # Usage chunk arrives as the last event when
            # stream_options.include_usage=True. vLLM emits it
            # with choices=[]. Fold into per-iteration only;
            # cross-iteration total is computed once we know
            # this iteration is final (after the stream loop).
            if it.first_chunk_at is None:
                it.first_chunk_at = time.perf_counter()
            if usage := chunk.get("usage"):
                it.iteration_usage = _merge_usage(it.iteration_usage, usage)

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            if (txt := delta.get("content")) is not None:
                if txt:
                    it.assistant_text += txt
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
                if it.thinking_started_at is None:
                    it.thinking_started_at = now
                it.thinking_last_at = now
                it.thinking_text += rc
                yield events.thinking_delta(rc)

            for tc_delta in delta.get("tool_calls") or []:
                _accumulate_tool_call(it.tool_calls_acc, tc_delta)

            if fr := choice.get("finish_reason"):
                it.finish_reason = fr
    except _BROKEN_STREAM_ERRORS as exc:
        # One handler for every way a stream breaks (D7). A 4xx is the
        # request's own fault and goes up as before.
        if _is_client_error(exc):
            raise
        if isinstance(exc, ParseError):
            logger.warning("loop: parse error mid-stream — %s", exc)
            yield events.stream_raw(exc.raw, error=str(exc))
        reason = _stream_error_reason(exc)
        cancelled = (options.cancel_event is not None
                     and options.cancel_event.is_set())
        if isinstance(exc, ParseError) and it.finish_reason and not cancelled:
            # The finish frame already arrived: only a trailing line
            # (the usage chunk) was lost, and the completion is whole.
            pass
        elif (not it.tool_calls_acc and not cancelled
                and st.stream_retries < int(getattr(
                    options, "stream_retry_max", 0) or 0)):
            # Nothing was dispatched and no tool call had begun, so
            # the same request is safe to send again. The deltas it
            # already streamed are taken back by every consumer.
            st.stream_retries += 1
            logger.warning(
                "loop: broken stream (%s: %s) at iter=%d — retry %d",
                reason, exc, st.num_turns, st.stream_retries,
            )
            yield events.iteration_retry(
                reason=reason, attempt=st.stream_retries,
                discarded_text_chars=len(it.assistant_text),
                discarded_thinking_chars=len(it.thinking_text),
            )
            log_harness_event(session_id, "harness.stream_retried", {
                "reason": reason, "error": str(exc)[:300],
                "attempt": st.stream_retries, "iteration": st.num_turns,
                "discarded_text_chars": len(it.assistant_text),
                "discarded_thinking_chars": len(it.thinking_text),
            }, turn_id=getattr(options, "turn_id", "") or None)
            await _retry_backoff(
                float(getattr(options, "stream_retry_backoff_s", 0) or 0),
                options.cancel_event,
            )
            st.num_turns -= 1   # the same iteration, requested again
            it.flow = "continue"
            return
        elif isinstance(exc, ParseError):
            # Not retryable: end the turn on what streamed as text, and
            # dispatch none of the half-parsed tool calls.
            st.broken_stream = True
            it.finish_reason = "stream_error"
        else:
            raise
    except ContextOverflowError as exc:
        # vLLM rejected the prompt for exceeding context. Recovery:
        # truncate the largest tool result(s) in chat_messages,
        # append a synthetic tool note explaining the truncation,
        # and let the loop retry the same turn. Bounded by
        # ``options.max_context_overflow_recoveries`` to avoid an infinite
        # loop if truncation can't free enough budget.
        bound = st.options.max_context_overflow_recoveries
        if st.context_overflow_recoveries >= bound:
            logger.error(
                "loop: context overflow after %d recovery attempts — giving up",
                st.context_overflow_recoveries,
            )
            raise
        st.context_overflow_recoveries += 1
        # The engine told us the real size of the prompt it just
        # rejected, which is a better anchor than anything the
        # meter has: adopt it, then aim below the compaction wall.
        requested = int(getattr(exc, "requested_input_tokens", 0) or 0)
        if requested > 0:
            meter.observe_usage(
                {"input_tokens": requested}, it.request_msgs_len,
            )
            meter.observe_append(chat_messages)
        overflow_target = _relief_target(options, meter)
        report = _relieve_context(
            chat_messages,
            options=options,
            meter=meter,
            reason="overflow",
            target=overflow_target,
            keep_recent=int(getattr(
                options, "intra_turn_microcompact_keep_recent", 15)),
            iteration=st.num_turns,
        )
        logger.warning(
            "loop: context overflow (requested=%s tokens), recovery #%d: "
            "freed ~%d tokens via %s",
            exc.requested_input_tokens,
            st.context_overflow_recoveries,
            report.get("freed_tokens", 0),
            ", ".join(report.get("rungs") or []) or "nothing",
        )
        # P11: one event per recovery, the countable twin of the
        # stream_raw below (which is for the transcript's forensics).
        _log_harness_event(
            session_id, "harness.overflow_recovered",
            {
                "attempt": st.context_overflow_recoveries,
                "iteration": st.num_turns,
                "requested_input_tokens": exc.requested_input_tokens,
                "freed_tokens": report.get("freed_tokens", 0),
                "rungs": list(report.get("rungs") or []),
            },
            turn_id=getattr(options, "turn_id", "") or None,
        )
        yield events.stream_raw(
            "",
            error=(
                f"context_overflow_recovery: attempt={st.context_overflow_recoveries}, "
                f"rungs={','.join(report.get('rungs') or []) or 'none'}, "
                f"freed_tokens={report.get('freed_tokens', 0)}, "
                f"requested_input_tokens={exc.requested_input_tokens}"
            ),
        )
        st.num_turns -= 1   # don't count the recovered attempt against max_turns
        it.flow = "continue"
        return
    except MultimodalRejectedError as exc:
        # The engine was sent a screenshot it cannot take — a slot
        # whose `supports_vision` claims more than it serves. Strip
        # every image from the turn and retry; the tool text
        # (element lists, paths) still carries the work. Stripping the
        # messages is only half of it: the next capture would attach
        # refs again through the very code that produced this one, so
        # the flip latches attachment off as well (#1419).
        if st.images_latched:
            # Attachment is already off and the engine still saw an
            # image, so this is a part the strip cannot reach — a
            # caller-supplied image, not a tool ref. Retrying would
            # answer a 400 with the same 400, so this stays the one
            # case that ends the turn.
            raise
        st.images_latched = True
        from app.harness.tool_images import strip_all_image_refs
        n = strip_all_image_refs(chat_messages)
        # `exc` carries the engine's 400 body, and it goes in the line
        # because `looks_like_multimodal_rejection` is a substring test:
        # any 400 mentioning "image" on a payload that carries refs is
        # classified as a vision refusal, and once the latch makes that
        # non-fatal the body is the only trace a misclassification left.
        logger.error(
            "loop: %s — stripped images from %d message(s) and retrying; "
            "image parts stay off for the rest of this turn; set "
            "models.%s.supports_vision: false",
            exc, n, options.model,
        )
        yield events.stream_raw("", error=f"multimodal_rejected: {exc}")
        st.num_turns -= 1
        it.flow = "continue"
        return

    if options.cancel_event is not None and options.cancel_event.is_set():
        st.stop_reason = "cancelled"
        st.accumulated_text += it.assistant_text
        it.flow = "break"
        return

    if it.thinking_text:
        thinking_ms = (
            int((it.thinking_last_at - it.thinking_started_at) * 1000)
            if it.thinking_started_at is not None
            else 0
        )
        yield events.thinking_done(it.thinking_text, duration_ms=thinking_ms)

    st.forced_tool_choice = None   # P6: the request that used it is in
    it.tool_calls_committed = [] if st.broken_stream else _commit_tool_calls(
        it.tool_calls_acc, summary_tools=st.summary_tools,
        finish_reason=it.finish_reason or "",
    )
    if st.wrapup_state == "requested" and it.tool_calls_committed:
        # The engine was told "none"; a call that arrives anyway is
        # never dispatched, and never reaches history or the events,
        # where it would be a tool call with no result.
        logger.warning(
            "loop: wrap-up request returned %d tool call(s) despite "
            "tool_choice=none — dropped (session=%s)",
            len(it.tool_calls_committed), session_id,
        )
        it.tool_calls_committed = []

    iteration_ended_at = time.perf_counter()
    iteration_duration_ms = int((iteration_ended_at - it.started_at) * 1000)
    ttft_ms, request_ms, cache_ratio = _request_timing(
        it.request_started_at, it.first_chunk_at, iteration_ended_at,
        it.iteration_usage,
    )
    st.last_iteration_usage = it.iteration_usage
    st.total_usage = _accumulate_iteration_usage(st.total_usage, it.iteration_usage)
    # The engine just reported the real size of the prompt it
    # processed. `request_msgs_len` is the list length as that
    # request went out, so everything appended from here is
    # attributed to the meter's estimate rather than double-counted.
    meter.observe_usage(it.iteration_usage, it.request_msgs_len)
    meter.observe_append(chat_messages)
    asst_evt = events.assistant_message(
        text=it.assistant_text,
        tool_calls=it.tool_calls_committed,
        thinking=it.thinking_text,
        usage=it.iteration_usage,
        duration_ms=iteration_duration_ms,
        iteration=st.num_turns,
        finish_reason=it.finish_reason or "stop",
        context=meter.snapshot() if meter.measured else None,
        ttft_ms=ttft_ms,
        request_ms=request_ms,
        cache_ratio=cache_ratio,
    )
    yield asst_evt
    st.accumulated_text += it.assistant_text

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
    # Append-only from here on: `keep_reasoning` decides whether THIS
    # message carries its reasoning, but nothing already appended —
    # and therefore already prefilled and cached by the engine — is
    # ever edited again. The window is applied at turn entry.
    chat_messages.append(_assistant_message_for_history(
        text=it.assistant_text, tool_calls=it.tool_calls_committed,
        reasoning=it.thinking_text if st.keep_reasoning > 0 else "",
        session_id=options.session_id, iteration=st.num_turns,
    ))

    # Snapshot chat_messages length before firing OnEvent. The
    # observer may append a user message ("inject" lever); if it
    # does AND the model is otherwise about to terminate this turn
    # (no tool calls), we continue the loop so the inject takes
    # effect on the next iteration instead of being lost.
    #
    # It is also where this iteration's batch begins. A hook can append
    # to `chat_messages` while the batch is still running — Inner
    # Voice's pretool inject does exactly that — and an append that
    # lands between two tool messages leaves
    # `assistant(tool_calls) → user → tool`, which is not a shape any
    # engine accepts. `_reorder_batch_messages` puts the slice back in
    # wire order once the batch is done. Taken from BEFORE the
    # assistant_message hook fired, not after it: an observer inject
    # appended during `fire_on_event(asst_evt)` sits between
    # `assistant(tool_calls)` and this batch's tool messages, and a base
    # taken after it would leave the inject outside the slice the
    # reorder may move (review 2026-09-24, D8).
    it.batch_base = len(chat_messages)
    if options.hooks is not None:
        await options.hooks.fire_on_event(asst_evt)
    it.observer_injected = len(chat_messages) > it.batch_base

    if st.wrapup_state == "requested":
        # The one toolless answer is in; nothing continues past it,
        # an observer inject included. `stop_reason` is still
        # "max_turns", so INCOMPLETE and the finalizer skip hold.
        st.wrapup_state = "done"
        it.flow = "break"


async def _end_without_tools(st: TurnState, it: Iteration):
    """An iteration that made no tool call: end the turn, or go once more.

    Goes on ("continue") for an observer inject there is room to answer,
    and for the echo guard's one re-prompt; otherwise records why the turn
    ended and sets "break".
    """
    options = st.options
    chat_messages = st.chat_messages
    meter = st.meter

    if st.broken_stream:
        st.stop_reason = "stream_error"
        it.flow = "break"
        return
    if it.observer_injected:
        # Observer injected a system message. Continue the loop
        # so the model gets to read it and respond — but only if
        # there is room to respond IN.
        #
        # On 2026-09-11 round 875 reached this branch with the
        # window already full: the loop continued, the next
        # completion was capped at `window - prompt`, the model
        # re-sent the same cut-off heredoc, and vLLM eventually
        # 400'd. An inject the model cannot answer is worse than
        # no inject — it spends the last iteration the turn had.
        meter.observe_append(chat_messages)
        floor = int(getattr(
            options, "context_relief_terminal_floor_tokens", 12_000))
        if meter.measured and meter.headroom < floor:
            _relieve_context(
                chat_messages,
                options=options,
                meter=meter,
                reason="terminal_inject",
                keep_recent=int(getattr(
                    options, "intra_turn_microcompact_keep_recent", 15)),
                iteration=st.num_turns,
            )
        if meter.measured and meter.headroom < floor:
            logger.warning(
                "loop: dropping terminal inject — headroom %d < floor %d "
                "(iter=%d); ending turn as context_exhausted",
                meter.headroom, floor, st.num_turns,
            )
            _log_harness_event(
                st.session_id,
                "harness.terminal_inject_dropped_for_context",
                {
                    "headroom": meter.headroom,
                    "floor": floor,
                    "iteration": st.num_turns,
                    "used": meter.used,
                    "context_window": meter.window,
                },
            )
            st.stop_reason = "context_exhausted"
            st.accumulated_text += ""
            it.flow = "break"
            return
        logger.info(
            "loop: observer injected on terminal iteration — continuing loop",
        )
        it.flow = "continue"
        return
    # Echo guard — the model printed a shell command in a fenced
    # block but called no tool. If Bash is available and we haven't
    # already nudged this turn, append a user-role nudge and loop
    # once more so it can actually call the tool (or confirm it was
    # only showing the command). A user message is used rather than
    # a second system message because vLLM chat templates reject a
    # non-leading system role.
    if (
        getattr(options, "echo_guard_enabled", True)
        and st.echo_guard_reprompts < _MAX_ECHO_GUARD_REPROMPTS
        and _looks_like_unexecuted_command(it.assistant_text)
        and "Bash" not in it.current_disallowed
    ):
        st.echo_guard_reprompts += 1
        if getattr(options, "echo_guard_mode", "nudge") == "tool_choice":
            # P6a: discard the attempt and re-send the request
            # byte-identical but for `tool_choice: "required"`.
            # `observer_injected` is False here (that branch
            # continued above), so the last message is this
            # iteration's assistant turn.
            chat_messages.pop()
            meter.observe_append(chat_messages)
            if it.assistant_text:
                st.accumulated_text = st.accumulated_text[
                    : len(st.accumulated_text) - len(it.assistant_text)]
            yield events.iteration_retry(
                reason="echo_guard",
                attempt=st.echo_guard_reprompts,
                discarded_text_chars=len(it.assistant_text),
                discarded_thinking_chars=len(it.thinking_text),
            )
            st.forced_tool_choice = "required"
            logger.info(
                "loop: echo-guard reissue #%d with tool_choice="
                "required (iter=%d)", st.echo_guard_reprompts, st.num_turns,
            )
            st.num_turns -= 1
            it.flow = "continue"
            return
        chat_messages.append({"role": "user", "content": _ECHO_GUARD_NUDGE})
        logger.info(
            "loop: echo-guard re-prompt #%d — assistant emitted a shell "
            "fence with no tool call (iter=%d)",
            st.echo_guard_reprompts, st.num_turns,
        )
        it.flow = "continue"
        return
    st.stop_reason = it.finish_reason or "stop"
    it.flow = "break"


async def _dispatch_batch(st: TurnState, it: Iteration):
    """Dispatch this iteration's tool calls and append their results (P13.3).

    One path for every batch; only its concurrency differs — the parallel
    maximum for a batch that qualifies (read-only, flag on or a P8 Task
    fan-out), 1 for everything else. At 1 it is the sequential dispatch
    exactly, which is what nearly every production turn runs:

      - **Admission is lazy and in wire order.** A call is announced
        (`tool_call` + OnEvent) and put through `_pre_dispatch` only when a
        slot is free, so at 1 the stream is `call₁, result₁, call₂,
        result₂` and `_pre_dispatch(call₂)` — a hook deny, an Inner Voice
        pretool inject — runs after `result₁` is in history, as it always
        has. An early result (parse error, disabled, ToolSearch, a deny)
        holds its slot until it is yielded. Within one admission round no
        call starts executing until every admitted call has been through
        `_pre_dispatch`: `mark_loaded` mutates the shared LoadedToolSet and
        a deny is a decision the model must see in the order it made the
        calls.
      - **At 1 the MCP call is awaited in this task**, not in a child task:
        an exception from `_execute_tool_call` ends the turn as it did, and
        nothing it binds in a contextvar is lost to a copied context. Above
        1 each call runs in its own task, an unexpected raise becomes a
        `dispatch_failed` result (one tool failing is not a reason to
        abandon its siblings — and no TaskGroup, which would cancel them),
        and results are yielded as they land: the frontend and messages.py
        key on call_id.
      - **Captions are a wire-order pass** before the batch starts
        (`_account_captions`): the ratchet is about the first miss, not the
        first result back.
      - **History is written in wire order**, a contiguous prefix at a time
        as results land — at 1 that is right after each result, which is
        what lets the next call's pre-dispatch see it. The batch's slice is
        then put back in shape (`_reorder_batch_messages`) and the image cap
        applied, as before.

    The one difference from the old two-path code is on formerly-parallel
    batches wider than the semaphore: a call is announced when it is
    admitted rather than all up front. Closing the generator mid-batch (a
    Stop click, a disconnect) cancels every call still running.
    """
    options = st.options
    chat_messages = st.chat_messages
    pool = st.pool
    session_id = st.session_id
    calls = it.tool_calls_committed

    # P8: a batch of fresh Task calls to parallel-safe profiles
    # overlaps even with general dispatch off. Each child still gets
    # its own `_meta` grant scope and deny list from `_execute_tool_call`
    # (D4), exactly as it would sequentially.
    _safe_tasks = frozenset(
        getattr(options, "parallel_safe_task_profiles", None) or ())
    run_parallel = (
        len(calls) > 1
        and _batch_is_read_only(calls, _read_only_names(pool), _safe_tasks)
        and (getattr(options, "parallel_tool_calls_enabled", False)
             or _is_task_fanout(calls, _safe_tasks))
    )
    concurrency = (
        max(1, int(getattr(options, "parallel_tool_calls_max_concurrency", 4)))
        if run_parallel else 1
    )

    (st.caption_total, st.caption_present, st.caption_nudged,
     nudge_id) = _account_captions(
        calls, st.summary_tools, st.caption_total, st.caption_present,
        st.caption_nudged, session_id, st.num_turns)
    # By position, not id: the nudge belongs to the first miss itself.
    nudge_at = next(
        (i for i, tc in enumerate(calls)
         if nudge_id and tc["id"] == nudge_id
         and tc["function"]["name"] in st.summary_tools
         and not tc.get("_summary")),
        None,
    )

    async def _run_contained(idx: int) -> tuple[int, NormalizedEvent]:
        call = calls[idx]
        try:
            return idx, await _execute_tool_call(
                tc=call, pool=pool, options=options, session_id=session_id,
                runtime_disallowed=it.current_disallowed,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:      # pragma: no cover
            logger.exception(
                "loop: parallel dispatch failed on %s", call["function"]["name"])
            return idx, events.tool_result(
                call_id=call["id"], name=call["function"]["name"],
                content=f"Tool dispatch failed: {exc}", is_error=True,
                error_class="dispatch_failed",
            )

    results: dict[int, NormalizedEvent] = {}
    # Results in hand, not yet yielded: early ones in wire order, then
    # executed ones in the order they landed.
    ready: collections.deque[tuple[int, NormalizedEvent]] = collections.deque()
    running: set[asyncio.Task] = set()
    landed: collections.deque[asyncio.Task] = collections.deque()
    next_call = 0      # the next call to admit, in wire order
    written = 0        # calls whose result is in history, a wire-order prefix
    try:
        while next_call < len(calls) or ready or running:
            # Admission: announce and pre-dispatch as many as the slots allow.
            admitted: list[int] = []
            while (next_call < len(calls)
                   and len(running) + len(ready) + len(admitted) < concurrency):
                idx, tc = next_call, calls[next_call]
                next_call += 1
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
                    loaded_set=st.loaded_set,
                    runtime_disallowed=it.current_disallowed,
                    meter=st.meter,
                )
                if pre is not None:
                    ready.append((idx, pre))
                else:
                    admitted.append(idx)
            for idx in admitted:
                if concurrency == 1:
                    ready.append((idx, await _execute_tool_call(
                        tc=calls[idx], pool=pool, options=options,
                        session_id=session_id,
                        runtime_disallowed=it.current_disallowed,
                    )))
                else:
                    task = asyncio.create_task(_run_contained(idx))
                    task.add_done_callback(landed.append)
                    running.add(task)

            # Hand out one result, then admit again.
            if ready:
                idx, result_evt = ready.popleft()
            else:
                if not landed:
                    await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                task = landed.popleft()
                running.discard(task)
                idx, result_evt = task.result()
            if idx == nudge_at:
                result_evt = _with_caption_nudge(result_evt)
            results[idx] = result_evt
            yield result_evt
            if options.hooks is not None:
                await options.hooks.fire_on_event(result_evt)

            while written < len(calls) and written in results:
                chat_messages.append(_tool_history_message(
                    calls[written]["id"], results[written],
                    allow_images=not st.images_latched))
                written += 1
    finally:
        # The consumer can close this generator mid-batch (a Stop click, a
        # disconnect). Leaving these running would keep dispatching tools
        # for a turn nobody is reading.
        for task in running:
            if not task.done():
                task.cancel()

    _reorder_batch_messages(chat_messages, it.batch_base)
    _cap_images(chat_messages, it.batch_base)


def _after_batch(st: TurnState, it: Iteration) -> None:
    """Mid-turn microcompaction, once this iteration's results have landed.

    Clears stale tool results IF the prompt is actually pressing on the
    context window. Mutates `chat_messages` in place so the observer's
    chat_messages_handle stays pointing at the same list.

    The tool-count threshold is a cheap pre-check only. Until 2026-09-05 it
    was the entire trigger, so a turn that ran 70 tool calls at 40% of its
    context window was held to 5 inline results the whole way, and
    everything it had read was gone.
    """
    options = st.options
    chat_messages = st.chat_messages
    st.meter.observe_append(chat_messages)
    if (
        it.tool_calls_committed
        and getattr(options, "intra_turn_microcompact_enabled", True)
    ):
        threshold = getattr(options, "intra_turn_microcompact_threshold", 15)
        keep = getattr(options, "intra_turn_microcompact_keep_recent", 15)
        tool_count = sum(1 for m in chat_messages if m.get("role") == "tool")
        if tool_count >= threshold:
            # The tool-count threshold stays a cheap pre-check; the
            # ladder's own rungs each decide whether they are needed.
            # `relief_latch` is what makes that "needed" mean a
            # crossing rather than a standing state: the ladder re-
            # enters only once the prompt has regrown to the
            # intra-turn trigger (see `_ReliefLatch`).
            _relieve_context(
                chat_messages,
                options=options,
                meter=st.meter,
                reason="intra_turn",
                keep_recent=keep,
                tool_count=tool_count,
                iteration=st.num_turns,
                latch=st.relief_latch,
            )


async def _close_turn(st: TurnState):
    """The finalizer, if one was asked for, then the turn's `result`."""
    options = st.options
    structured, structured_error = await _maybe_finalize(
        options=options,
        stop_reason=st.stop_reason,
        chat_messages=st.chat_messages,
        tools=st.last_visible_tools,
        total_usage=st.total_usage,
    )

    duration_ms = int((time.perf_counter() - st.started_at) * 1000)
    if st.caption_total:
        logger.info(
            "loop: tool captions %d/%d (%d%%) session=%s",
            st.caption_present, st.caption_total,
            round(100 * st.caption_present / st.caption_total), st.session_id,
        )
    result_done_evt = events.result(
        stop_reason=st.stop_reason,
        usage=st.total_usage,
        num_turns=st.num_turns,
        duration_ms=duration_ms,
        response_text=st.accumulated_text,
        tool_calls_total=st.caption_total,
        tool_calls_captioned=st.caption_present,
        structured=structured,
        structured_error=structured_error,
        wrapped_up=st.wrapup_state == "done",
    )
    yield result_done_evt
    if options.hooks is not None:
        await options.hooks.fire_on_event(result_done_evt)


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
            session_id=options.session_id,
            # No extra_body, and no no-thinking default (#1431): on the
            # primary either spelling rewrites the system message's first
            # sentence and re-prefills the turn. finalizer.py's docstring.
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("loop: finalizer raised: %s", exc)
        return None, f"finalizer failed: {type(exc).__name__}: {exc}"

    # The extra completion's tokens are real tokens; fold them in so usage
    # accounting does not quietly under-report every worker verdict.
    #
    # `input_tokens` deliberately does NOT go into the same accumulator: for
    # every other consumer in the tree it is the PEAK single prompt (see
    # _accumulate_iteration_usage), and adding this request's prompt to the peak
    # would make every number downstream mean something different depending on
    # whether a finalizer ran. It gets its own key because until now its prompt
    # was dropped entirely — a turn that asked for three structured verdicts
    # reported the same prompt cost as one that asked for none, which is exactly
    # the direction of error that makes a re-prefill look free. #529's replay
    # needs the sum, and needs it not to lie.
    for key in ("output_tokens", "total_tokens"):
        if usage.get(key):
            total_usage[key] = total_usage.get(key, 0) + usage[key]
    # Its own key as well as the fold: the ledger records it per verdict so
    # a budget that is too tight shows up as a number, not as a regex rate.
    if usage.get("output_tokens"):
        total_usage["finalizer_output_tokens"] = (
            total_usage.get("finalizer_output_tokens", 0)
            + int(usage["output_tokens"]))
    if usage.get("input_tokens"):
        total_usage["finalizer_input_tokens"] = (
            total_usage.get("finalizer_input_tokens", 0)
            + int(usage["input_tokens"]))
    # The reasoning share of that output (#1431). A subset of
    # finalizer_output_tokens, so it is reported beside it, never folded again.
    if usage.get("reasoning_tokens"):
        total_usage["finalizer_reasoning_tokens"] = (
            total_usage.get("finalizer_reasoning_tokens", 0)
            + int(usage["reasoning_tokens"]))
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
    pool = await get_or_open_pool(cfg)
    # P12: re-list tools once the catalog is older than
    # `harness.mcp_pool.discovery_ttl_s`. Here, at the turn boundary and only
    # here, so a turn's advertised tools never change under it. It never
    # empties the catalog and never raises into the turn.
    try:
        await pool.ensure_fresh()
    except Exception as exc:
        logger.warning("loop: mcp discovery refresh skipped: %s", exc)
    return pool


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
    # The reasoning share of the completion, nested the same way (P11). A
    # subset of `output_tokens`, never added to it; `_accumulate_iteration_
    # usage` sums it across iterations like any other non-peak count.
    cdetails = chunk.get("completion_tokens_details")
    if isinstance(cdetails, dict):
        if isinstance(rt := cdetails.get("reasoning_tokens"), int) \
                and not isinstance(rt, bool):
            out["reasoning_tokens"] = rt
    return out


def _request_timing(
    request_started_at: float, first_chunk_at: float | None,
    ended_at: float, usage: dict[str, Any],
) -> tuple[int | None, int | None, float | None]:
    """`(ttft_ms, request_ms, cache_ratio)` for one iteration (P11).

    TTFT is None when no chunk arrived at all — a stream that produced
    nothing has no first token, and a zero would read as an instant one.
    The cache ratio is None without a prompt size to divide by, and clamped
    at 1.0 for the reason `_accumulate_iteration_usage` clamps the turn row.
    """
    ttft_ms = (
        max(0, int((first_chunk_at - request_started_at) * 1000))
        if first_chunk_at is not None else None
    )
    request_ms = max(0, int((ended_at - request_started_at) * 1000))
    prompt = usage.get("input_tokens")
    cached = usage.get("cache_read")
    cache_ratio: float | None = None
    if isinstance(prompt, int) and prompt > 0 and isinstance(cached, int):
        cache_ratio = round(min(1.0, max(0, cached) / prompt), 4)
    return ttft_ms, request_ms, cache_ratio


# The sink moved to `app.harness.telemetry` (X1) so modules below the loop
# can emit events without importing it; the alias keeps every caller as is.
_log_harness_event = log_harness_event


def _record_anchor_fires(
    tags: list[Any], session_id: str, turn_id: str, iteration: int,
) -> None:
    """One `harness.anchor_fired` event per anchor message appended (#769).

    `tags` are the `ANCHOR_TAG` values the builders in `app.deadline_anchor`
    and `app/routers/messages.py` attach; a caller's own untagged anchor is
    recorded as `unnamed` rather than skipped, so a count of events is a count
    of appended anchor messages. Never raises: a record that cannot be written
    costs the record, not the warning or the turn.
    """
    if not session_id:
        return
    for tag in tags:
        try:
            tag = tag if isinstance(tag, dict) else {}
            from app import event_log

            event_log.log_event(
                session_id, "harness.anchor_fired",
                {
                    "anchor": str(tag.get("kind") or "unnamed"),
                    "level": tag.get("level"),
                    "iteration": iteration,
                    "session_id": session_id,
                    "turn_id": turn_id,
                },
                turn_id=turn_id or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("loop: could not record anchor fire: %s", exc)


def _relief_target(options: Any, meter: Any) -> int:
    """Tokens the relief ladder is trying to get back under.

    The same target the intra-turn pass already budgets against, so the two
    cannot disagree about where "relieved" is.
    """
    try:
        threshold = int(meter.threshold)
    except Exception:  # noqa: BLE001
        return 0
    frac = float(getattr(options, "intra_turn_microcompact_target_fraction", 0.6))
    return int(threshold * frac)


def _relief_rearm_level(options: Any, meter: Any) -> int:
    """Tokens the prompt must regrow to before the next intra-turn pass runs.

    Deliberately the same arithmetic rung 1 uses for its own trigger —
    `truncation_threshold(window) × intra_turn_microcompact_trigger_fraction`
    (`app/harness/microcompact.py`; `truncation_threshold` is the compaction
    wall, not the raw context window) — so the latch releases exactly where
    the ladder would have fired on its own. Inventing a second, independent
    number for the same wall is how a latch ends up either never releasing or
    firing on every iteration anyway.
    """
    try:
        threshold = int(meter.threshold)
    except Exception:  # noqa: BLE001
        return 0
    frac = float(getattr(options, "intra_turn_microcompact_trigger_fraction", 0.8))
    return int(threshold * frac)


class _ReliefLatch:
    """One relief pass per crossing of the target, not one per iteration.

    Why this exists: on a long turn the ladder can never reach its target.
    Rung 1's budget is `target - offset`, where `offset` is the system prompt
    plus the tool schemas (~44k tokens the message list cannot pay for), so
    once a turn settles above the target the ladder's only gate —
    `meter.used > target` — is true on *every* iteration. Measured on
    2026-09-20 over `logs/server.err.1` (relief events 10:04:29 → 18:51:02,
    8.78 h): **2,474** `intra_turn` relief events = 282 an hour, 87.1% of them
    firing *below* the level at which rung 1 would have triggered by itself and
    73.9% freeing under 2,000 tokens; the item's own triage reading one day
    earlier (2026-09-19) was 240 an hour, 97.0% below trigger. Each event
    rewrites cached history, which costs a re-prefill of everything after the
    edit point (#520).

    So after a pass the ladder stays closed until the prompt regrows to
    `_relief_rearm_level`. The state is per turn: a new turn has a new message
    list and a new cached prefix, so it gets a new latch. It applies to the
    per-iteration (`intra_turn`) caller only — the wall paths (context-overflow
    recovery, pre-request, terminal-inject) run the whole ladder unconditionally,
    because three autocode rounds died at that wall on 2026-09-11 (866-a, 869,
    875) and a latch that can kill a turn is worse than a drip.
    """

    __slots__ = ("passes", "rearm", "announced")

    def __init__(self) -> None:
        self.passes = 0          # relief passes already run this turn
        self.rearm = 0           # token level at which the next pass is allowed
        self.announced = False   # the latched turn has said so once in the log

    def closed(self, used: int) -> bool:
        """True once a pass has run and until `used` reaches the rearm level.

        Fails open: with no pass run, or with a stub meter that left `rearm`
        at 0, the ladder runs. Relief that cannot be armed must not also be
        blocked.
        """
        return self.passes > 0 and used < self.rearm


def _relieve_context(
    chat_messages: list[dict[str, Any]],
    *,
    options: Any,
    meter: Any,
    reason: str,
    target: int = 0,
    keep_recent: int = 15,
    tool_count: int = 0,
    iteration: int = 0,
    latch: "_ReliefLatch | None" = None,
) -> dict[str, Any]:
    """Free context, cheapest rung first, stopping as soon as it is enough.

    Four rungs, in cost order, each running only while the meter still says
    the turn is over `target`:

    1. **Clear stale tool results** (`_intra_turn_microcompact`). Spills to
       disk, so nothing is lost. Self-gating on its own arithmetic, and the
       only rung that runs when the meter is unmeasured.
    2. **Prune preserved reasoning**, first to the configured window and
       then to `reasoning_keep_under_pressure`. This rewrites messages the
       engine has already cached and costs a re-prefill (#520). Until
       2026-09-19 the comment here claimed that meant it did not run in the
       per-iteration path; it ran there 2,474 times in 8.78 h (282/hour, and in
       all 2,474 of those lines — every one names `reasoning:` in its `via`
       field), because the gate is `used > target`
       and a long turn is above target
       from iteration ~15 on. Under pressure the alternative is losing the
       turn; what is not defensible is paying it once per iteration, which
       is what `latch` now stops.
    3. **Spill `Write`/`Edit` bodies** out of assistant tool-call arguments
       once their results have landed. The residue nothing else can reach.
    4. **Truncate the largest tool results** — destructive-ish (it spills
       first now), and last for that reason.

    `latch` is the per-turn `_ReliefLatch` the per-iteration caller passes:
    once a pass has run, later iterations of the same turn return without
    running any rung until the prompt regrows to `_relief_rearm_level`. The
    wall callers (overflow recovery, pre-request, terminal-inject) pass no
    latch at all and always get the whole ladder.

    Returns a report dict for the log and the event; never raises, because a
    turn that dies inside its own relief path is strictly worse than one
    that runs a little over budget.
    """
    report: dict[str, Any] = {"reason": reason, "rungs": [], "freed_tokens": 0}
    if not getattr(options, "context_relief_enabled", True):
        return report

    before = int(getattr(meter, "used", 0) or 0)
    target = target or _relief_target(options, meter)
    rearm = _relief_rearm_level(options, meter)

    # -- the latch, for the per-iteration caller only --------------------
    # An unmeasured meter cannot say where the prompt sits, so it cannot
    # justify closing: the ladder runs exactly as it always has.
    if (
        latch is not None
        and getattr(meter, "measured", False)
        and latch.closed(before)
    ):
        report["latched"] = True
        report["passes"] = latch.passes
        report["rearm"] = latch.rearm
        if not latch.announced:
            # Said once per turn, and worded so that it does NOT match the
            # grep for `loop: context relief (intra_turn)` — the follow-up
            # measurement counts those lines to see whether the drip
            # stopped. A turn that is simply under target logs nothing at
            # all, so this line is the only thing that tells the two apart.
            latch.announced = True
            logger.info(
                "loop: context relief latched (%s): used %d below rearm %d "
                "after %d pass%s this turn (target %d) — next pass allowed "
                "at >=%d",
                reason, before, latch.rearm, latch.passes,
                "" if latch.passes == 1 else "es", target, latch.rearm,
            )
        return report

    def _over() -> bool:
        # Unmeasured never counts as over: an iteration-1 turn with no
        # usage report has not been shown to need anything.
        return bool(getattr(meter, "measured", False)) and meter.used > target

    # -- rung 0: old screenshots ----------------------------------------
    # ~1.3k tokens each and the cheapest thing in the prompt to lose: the
    # newest few still show the current screen, and the files stay on disk.
    # Under pressure only, like every rung after rung 1 (D13): dropping an
    # image rewrites a message the engine has cached, and a turn below
    # target has nothing to buy with that re-prefill.
    try:
        from app.harness.tool_images import images_cfg, keep_newest
        if _over() and any(
            m.get("_image_refs") for m in chat_messages if isinstance(m, dict)
        ):
            dropped = keep_newest(
                chat_messages, int(images_cfg().get("keep_on_compaction") or 3))
            if dropped:
                meter.resync(chat_messages)
                report["rungs"].append(f"images:{dropped}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("loop: relief rung 'images' failed: %s", exc)

    # -- rung 1: stale tool results -------------------------------------
    try:
        cleared = _intra_turn_microcompact(
            chat_messages,
            options=options,
            meter=meter,
            keep_recent=keep_recent,
            tool_count=tool_count or sum(
                1 for m in chat_messages if m.get("role") == "tool"
            ),
            iteration=iteration,
        )
        meter.resync(chat_messages)
        # Named only when it cleared something (D13): rung 1 self-gates, and
        # a report listing it on every pass said it acted when it had not.
        # A pass whose other rungs ran and freed ~0 still records.
        if cleared:
            report["rungs"].append(f"tool_results:{cleared}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("loop: relief rung 'tool_results' failed: %s", exc)

    # -- rung 2: preserved reasoning ------------------------------------
    if _over():
        keep_now = int(getattr(options, "preserve_thinking_iterations", 0) or 0)
        under_pressure = int(
            getattr(options, "context_relief_reasoning_keep_under_pressure", 2)
        )
        for keep in (keep_now, under_pressure):
            if not _over():
                break
            if keep <= 0:
                continue
            try:
                _prune_reasoning(chat_messages, keep=keep)
                meter.resync(chat_messages)
                report["rungs"].append(f"reasoning:{keep}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("loop: relief rung 'reasoning' failed: %s", exc)
                break

    # -- rung 3: assistant tool-call argument bodies --------------------
    if _over() and getattr(options, "context_relief_shrink_arguments", True):
        try:
            from app.harness.microcompact import shrink_assistant_arguments

            shrunk_msgs, shrunk, freed_chars = shrink_assistant_arguments(
                chat_messages,
                keep_recent_tools=keep_recent,
                min_chars=int(
                    getattr(options, "context_relief_shrink_arguments_min_chars", 2_000)
                ),
                session_id=getattr(options, "session_id", "") or "",
                tools=getattr(
                    options, "context_relief_shrink_arguments_tools", ("Write", "Edit")
                ),
            )
            if shrunk:
                # Slice-assign: `chat_messages` may be the observer's own
                # handle and rebinding would leave it holding the old list.
                chat_messages[:] = shrunk_msgs
                meter.resync(chat_messages)
                report["rungs"].append(f"arguments:{shrunk}")
                report["argument_chars_freed"] = freed_chars
        except Exception as exc:  # noqa: BLE001
            logger.warning("loop: relief rung 'arguments' failed: %s", exc)

    # -- rung 4: truncate the largest tool results ----------------------
    if _over():
        try:
            over_tokens = max(0, meter.used - target)
            truncated, freed_chars = _truncate_largest_tool_results(
                chat_messages,
                target_chars=max(20_000, over_tokens * 4),
                session_id=getattr(options, "session_id", "") or "",
                min_chars=int(
                    getattr(options, "intra_turn_microcompact_min_chars", 2_000)
                ),
                disallowed_tools=list(
                    getattr(options, "disallowed_tools", None) or []
                ),
            )
            if truncated:
                meter.resync(chat_messages)
                report["rungs"].append(f"truncate:{truncated}")
                report["truncated_chars_freed"] = freed_chars
        except Exception as exc:  # noqa: BLE001
            logger.warning("loop: relief rung 'truncate' failed: %s", exc)

    after = int(getattr(meter, "used", 0) or 0)
    report["freed_tokens"] = max(0, before - after)
    report["used_before"] = before
    report["used_after"] = after
    report["target"] = target
    if latch is not None and rearm <= 0:
        # `_relief_rearm_level` could not name a level, which happens when the
        # meter has no readable `threshold`. Do not arm on that.
        #
        # What a 0 level actually does, read off `closed()`
        # (`passes > 0 and used < rearm`): every reading is non-negative, so
        # `used < 0` is false and the latch would never close — the ladder
        # would keep running every iteration exactly as it did before this
        # change. That is not the dangerous direction; a level ABOVE the wall
        # is, because then `used < rearm` stays true for the whole turn and
        # relief stops while the turn walks into the wall. What a 0 level does
        # buy is a lie in the log: `pass 1 this turn, next pass allowed at
        # >=0` records a release level the turn has already passed, so the
        # field the whole point of this line is to expose — the level the next
        # pass waits for — reads as a latch that exists and does nothing.
        # Refusing to arm keeps that field honest, and a meter that cannot
        # answer for its window costs the drip it has always cost, never a
        # pass it was entitled to run.
        if not latch.announced:
            logger.warning(
                "loop: context relief (%s) NOT latched: this meter yields no re-arm "
                "level (threshold unreadable), so the ladder stays open on every "
                "iteration (target %d)",
                reason, target,
            )
            latch.announced = True
        report["unlatched"] = True
        tail = ""
    elif latch is not None:
        # Counted even when every rung freed nothing. That is precisely the
        # case that used to re-arm forever: a pass that cannot reach its
        # target (the ~44k of system prompt and tool schemas inside `used` is
        # not in the message list to clear) would otherwise run again next
        # iteration, and free nothing again.
        latch.passes += 1
        latch.rearm = rearm
        report["passes"] = latch.passes
        report["rearm"] = latch.rearm
        tail = (
            f"; pass {latch.passes} this turn, "
            f"next pass allowed at >={latch.rearm}"
        )
    else:
        tail = ""
    if report["rungs"] and report["freed_tokens"]:
        logger.info(
            "loop: context relief (%s) freed ~%d tokens: %d -> %d (target %d) via %s%s",
            reason, report["freed_tokens"], before, after, target,
            ", ".join(report["rungs"]), tail,
        )
    elif _over():
        logger.warning(
            "loop: context relief (%s) could not get under target: %d > %d "
            "(rungs tried: %s)%s",
            reason, after, target, ", ".join(report["rungs"]) or "none", tail,
        )
    _record_relief_pass(options, report, iteration=iteration)
    return report


def _record_relief_pass(
    options: Any,
    report: dict[str, Any],
    *,
    iteration: int = 0,
) -> None:
    """Publish one relief pass that ran at least one rung (#1078).

    Two places, and both live here rather than at the ladder's four call sites
    so a new caller cannot forget one: the overflow-recovery, pre-request and
    terminal-inject paths own no `report` variable of their own, and before this
    function none of them could be attributed at all. The line above this call
    named the rungs and the tokens and no session — 7,253 of them in a 5.6-day
    window, 0 with a session on them — which is why the count has to come from
    an event, not from the log.

    * **The event** is the record that carries the session id, and it is the only
      one that survives a turn which never reaches its usage row (killed by the
      deadline, or a `run_query` caller that books nothing).
    * **The note** hands the pass to the turn's compaction record, which whoever
      writes the usage row reads back. Missed here, the pass is absent from the
      column with no other trace — which is exactly what NULL is reserved for,
      so a silent skip would lie about a measured turn.
    """
    session_id = getattr(options, "session_id", "") or ""
    if not report.get("rungs") or not session_id:
        return

    data: dict[str, Any] = {
        "reason": report.get("reason", ""),
        "rungs": list(report.get("rungs") or ()),
        "freed_tokens": int(report.get("freed_tokens") or 0),
        "iteration": iteration,
    }
    for key in ("used_before", "used_after", "target", "passes", "rearm"):
        value = report.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            data[key] = value

    _log_harness_event(
        session_id, "harness.context_relief", data,
        turn_id=getattr(options, "turn_id", "") or None,
    )
    try:
        from app import compaction_record

        compaction_record.note_relief(session_id, report)
    except Exception as exc:  # noqa: BLE001 — never lose a turn over accounting
        logger.debug("loop: relief pass not booked on the usage record: %s", exc)


def _intra_turn_microcompact(
    chat_messages: list[dict],
    *,
    options: Any,
    meter: Any,
    keep_recent: int,
    tool_count: int,
    iteration: int,
) -> int:
    """Clear stale tool results in place, but only under real pressure.

    Returns how many results it cleared.

    Uses the same threshold arithmetic as the turn-start pass in
    `app.compaction`, so the two cannot disagree about where the wall is.
    Imports it lazily: `app.compaction` already lazy-imports this package
    in the other direction, and a module-level edge here would close the
    cycle.

    Context size is the turn's `ContextMeter` (D6, 2026-09-24): the engine's
    last reported prompt plus an estimate of what was appended since, with
    the fixed cost `chat_messages` does not contain (system prompt, tool
    schemas) carried as `meter.offset`. The budget is in estimator units,
    so it is `target - offset`. Until D6 this pass re-derived the offset
    from `total_usage["input_tokens"]` — a PEAK over the whole turn — so
    after one relief pass the offset silently absorbed every token that
    pass had freed and the next pass over-cleared, and on the overflow path
    (whose rejected size never reaches `total_usage`) it under-triggered.
    `meter.resync`, which every rung already calls, is what lets a second
    pass see the relieved size. An unmeasured meter still lets the rung run
    on the estimate alone, as it always could.
    """
    try:
        from app.compaction import (
            estimate_conversation_tokens,
            get_context_window,
            truncation_threshold,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("loop: intra-turn microcompact unavailable: %s", e)
        return 0

    threshold = truncation_threshold(get_context_window(getattr(options, "model", "")))
    trigger = int(threshold * float(
        getattr(options, "intra_turn_microcompact_trigger_fraction", 0.8)
    ))
    target = int(threshold * float(
        getattr(options, "intra_turn_microcompact_target_fraction", 0.6)
    ))

    def _estimate(msgs: list[dict]) -> int:
        return estimate_conversation_tokens(msgs, "")

    measured = bool(getattr(meter, "measured", False))
    if measured:
        # Idempotent: re-estimates the tail since the last report and
        # learns the offset if no append has been observed yet, so a meter
        # primed with a report alone still budgets in the right units.
        meter.observe_append(chat_messages)
        current = int(meter.used)
        # Everything in the prompt that isn't in `chat_messages`. The meter
        # clamps it at 0, so an over-reporting estimate can't invent headroom.
        offset = int(meter.offset)
    else:
        current = _estimate(chat_messages)
        offset = 0
    if current <= trigger:
        return 0

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
        # The marker ends by telling the model how to get the result back.
        # Same reason as `maybe_spill` above: this pass is what produces that
        # marker on a long turn, and the turn that clears most is the one
        # with `Read` denied (#1066).
        disallowed_tools=list(getattr(options, "disallowed_tools", None) or []),
        **_rung_one_tool_selection(options),
    )
    if cleared:
        chat_messages[:] = compacted
        logger.info(
            "loop: intra-turn microcompact cleared %d/%d tool results "
            "(kept last %d, %d -> %d tokens, target %d, iter=%d)",
            cleared, tool_count, keep_recent, current,
            _estimate(chat_messages) + offset, target, iteration,
        )
    return int(cleared or 0)


def _rung_one_tool_selection(options: Any) -> dict[str, Any]:
    """Which tool results rung 1 may clear (D10).

    `intra_turn_microcompact_non_compactable` set (a tuple, possibly empty)
    is deny-list mode: every result may be cleared except those tools'.
    Unset (None) is the historical allow-list — `Read/Bash/Grep/Glob/Edit/
    Write` only — under which an MCP domain result (a vault search, an
    http_fetch, a graph query) was never cleared and went straight to rung
    4's truncation.
    """
    deny = getattr(options, "intra_turn_microcompact_non_compactable", None)
    if deny is None:
        return {}
    return {"non_compactable_tools": tuple(deny)}


# The pair a consumer divides to get a cached fraction. They have to be the
# same KIND of figure — both peak or both sum — or the quotient is not a
# fraction at all. Until #859 they were not: `input_tokens` was the peak and
# `cache_read` the sum, and every reader from `usage_store.summary()` to the
# dashboard's usage panel divided them anyway. See `app/turn_usage.py`.
_PEAK_USAGE_KEYS = ("input_tokens", "cache_read")

# Folded by hand below, never by the generic SUM branch: a dict that already
# carried one of these would otherwise be counted twice.
_EXPLICIT_SUM_USAGE_KEYS = ("prompt_tokens_sum", "cache_read_sum")


def _accumulate_iteration_usage(
    total: dict[str, int], iteration: dict[str, int],
) -> dict[str, int]:
    """Fold one iteration's usage into the cross-iteration total.

    Each chat-completion request reports its own input/output counts. Across
    the agent loop the aggregate row carries TWO units, each named for its
    unit, because one unit cannot answer both questions anyone asks:

      - ``input_tokens`` / ``cache_read``: the PEAK across iterations.
        ``input_tokens`` has always meant the peak single prompt (the last
        iteration is typically largest because tool results only append, so max
        is conservative) and everything in the tree reads it that way —
        `_maybe_finalize` deliberately declines to fold the finalizer's prompt
        into it for exactly that reason. ``cache_read`` is a peak too, which is
        what makes the pair commensurable: since vLLM reports
        ``cached_tokens <= prompt_tokens`` for one request, the max of the
        cached column can never pass the max of the prompt column, so no
        reader can divide its way past 100%.
      - ``prompt_tokens_sum`` / ``cache_read_sum``: the SUM across iterations.
        This is what the turn actually cost and what it actually read from
        cache, and it is also bounded at 100% because a sum of cached tokens
        cannot exceed the sum of the prompts that carried them.
      - ``output_tokens``, ``cache_create``, ``total_tokens``: SUM as before.
        ``cache_create`` in particular can legitimately exceed the peak prompt
        — each iteration writes its own new suffix into the cache — and nothing
        divides by it, so it stays a sum by design.

    Without the fold at all the final ``result`` event would show only the LAST
    iteration's usage (replace semantics from ``_merge_usage``), which is the
    reason this function exists; mixing units across the fold is the reason it
    needed this docstring.

    The clamps at the end are not decoration. An engine cannot report more
    cached tokens than prompt tokens, so a row that does is a parser or engine
    defect, and publishing it would put a 900% hit rate into `usage.db` and
    read as excellent prefix-cache performance. Clamping bounds the published
    aggregate while the exact per-request numbers stay on the turn's
    per-iteration rows, where the defect is still visible.
    """
    out = dict(total)
    for k, v in iteration.items():
        if not isinstance(v, int) or isinstance(v, bool) or k in _EXPLICIT_SUM_USAGE_KEYS:
            continue
        if k in _PEAK_USAGE_KEYS:
            out[k] = max(out.get(k, 0), v)
        else:
            out[k] = out.get(k, 0) + v

    out["prompt_tokens_sum"] = out.get("prompt_tokens_sum", 0) + max(
        0, _int_usage(iteration, "input_tokens", "prompt_tokens"))
    out["cache_read_sum"] = out.get("cache_read_sum", 0) + max(
        0, _int_usage(iteration, "cache_read", "prompt_tokens_cached"))

    if out.get("cache_read", 0) > out.get("input_tokens", 0):
        logger.warning(
            "loop: iteration reported %d cached over a %d-token prompt; "
            "publishing the bounded pair (engine/parser defect — the raw "
            "reading is still on the per-iteration rows)",
            out["cache_read"], out.get("input_tokens", 0),
        )
        out["cache_read"] = out.get("input_tokens", 0)
    if out["cache_read_sum"] > out["prompt_tokens_sum"]:
        out["cache_read_sum"] = out["prompt_tokens_sum"]
    return out


def _int_usage(usage: dict[str, Any], *keys: str) -> int:
    """First int-valued key in ``usage``, or 0. Guards the sum fold above."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


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
    finish_reason: str = "",
) -> list[dict[str, Any]]:
    """Finalize accumulated tool calls.

    Parses `arguments` JSON; on failure, attaches an `_args_dict` of
    `{"__parse_error__": True, "raw": "..."}` so the dispatcher can
    surface a tool_result with `is_error=True` instead of crashing.

    `summary_tools` is the set of tools whose advertised schema carried the
    injected `summary` display parameter (see
    ``tool_schema.add_summary_param``). For those, the value is lifted onto
    `_summary` and removed from `_args_dict` — nothing validates arguments
    against the tool's inputSchema (top-level primitives are coerced only,
    unknown keys reach the handler), so a leaked caption would arrive in the
    handler's arguments unannounced. It is deliberately NOT
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
            # `finish_reason` and the raw length are what tell a malformed
            # call apart from one the context window cut in half. They are
            # the same shape on the wire and the right next move is the
            # opposite in each case — see `_pre_dispatch`.
            args_dict = {
                "__parse_error__": True,
                "raw": raw_args,
                "error": err,
                "finish_reason": finish_reason,
                "raw_len": len(raw_args),
            }
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
    session_id: str = "", iteration: "int | None" = None,
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

    One exception, and it is the reason this function is the choke point:
    a trace `app.thinking_fidelity.flag_fabricated_reasoning` flags does not
    go back (#1510). The primary sometimes emits reasoning about a request
    that was never made — a 1,223-character trace about "reproduce my
    complete previous thinking using the audit tool" on a turn that asked
    what `stream_chat` does — and preserved thinking is exactly the route by
    which such a trace becomes the model's own prior thought: it goes out on
    the next request, and `preserve_thinking_iterations: 6` keeps it visible
    for six more iterations. The item's triage measured a large minority of
    flagged blocks sitting behind an earlier flagged block in the same session,
    the shape reinforcement predicts; `scripts/thinking_fidelity_scan.py` is how
    to re-measure that, and this docstring deliberately does not carry its number.
    Withholding one trace costs the
    continuity of one iteration; replaying one tells the model it thought
    something it never thought, on a turn whose answer the model then has to
    fight. Dropped here, not deleted: the `role="thinking"` row written from
    the `thinking_done` event still records the trace for audit, and
    `app/thinking_fidelity.py` scans for it afterwards.

    The check sits in the builder and not at the one call site because every
    surface that puts reasoning on the wire must pass through it — a guard on
    one of two write surfaces is not a guard. There is today only the one
    surface (`run_query` appends the result of this call), and rebuilt
    history is a second route that `_prepare_messages_for_harness` already
    strips `reasoning` out of.

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
        marker = matched_marker(reasoning)
        if marker:
            logger.info(
                "loop: withheld a fabricated reasoning trace from preserved "
                "thinking (session=%s iteration=%s, %d chars, marker %r); "
                "the role=\"thinking\" row still records it",
                session_id or "-", iteration if iteration is not None else "-",
                len(reasoning), marker)
            reasoning = ""
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


def _cap_history_reasoning(chat_messages: list[dict[str, Any]], *, keep: int) -> None:
    """Apply the preserved-thinking window at turn entry, and under pressure.

    The bound itself is unchanged (`_prune_reasoning`, same `keep`). What
    changed is *where* it is applied, and that is the whole of backlog
    #520: this used to run after every iteration, inside the loop.

    Removing reasoning from a message the engine already prefilled is not
    a free edit. vLLM's Automatic Prefix Caching keys KV blocks on the
    serialized token prefix, so dropping tokens out of message *k*
    invalidates every block after *k* and the next request re-prefills
    them. With `keep=6` the message dropped at iteration *n* is the one
    from *n-7*, so the cliff lands on iteration 8 and then recurs on every
    iteration after it. Measured over the sessions created 2026-09-09:
    cached fraction 79.0% at iteration 7 -> **24.5% at iteration 8**, 41
    individual sessions falling ≥4 points at exactly that index, ≈10.2M
    tokens/day of avoidable re-prefill (`vllm:prefix_cache_hits_total` /
    `queries_total` = 61.75% since boot).

    Turn entry is the free place to pay for the bound. History arrives
    from `app.compaction.load_and_compact_session`, the prefix is cold on
    the first request of a turn regardless, and microcompaction is already
    rewriting messages there — so windowing costs nothing that is not
    already being spent. The intra-turn list is then a strict extension of
    the previous request's for the whole turn.

    The cost is honest and small: reasoning carried *forward* past `keep`
    iterations within one long turn is not trimmed any more, so a turn's
    peak input grows by whatever the older thinking costs. Session
    20260905_151355_iv5174 produced 66.6k reasoning tokens across 52
    iterations, which is what the window exists to bound; at `keep=6` the
    intra-turn overshoot is the thinking from iterations 1..n-6 that used
    to be dropped. `eval/run_preserve_thinking_eval.py` prices the
    keep-window trade-off if that ever needs re-tuning.

    Since 2026-09-12 there is a second caller: `_relieve_context` rung 2,
    which prunes to `context_relief.reasoning_keep_under_pressure` when the
    turn is about to hit the wall. That does pay the re-prefill #520 was
    filed against, deliberately — at that point the alternative is not a
    cheaper turn, it is no turn. The per-iteration call this docstring was
    written against is still gone; relief runs only when the meter says the
    prompt is over target.
    """
    if keep <= 0:
        return
    _prune_reasoning(chat_messages, keep=keep)


def _prune_reasoning(chat_messages: list[dict[str, Any]], *, keep: int) -> None:
    """Keep `reasoning` on the most recent `keep` assistant messages only.

    Callers: `_cap_history_reasoning` at turn entry, and `_relieve_context`
    rung 2 under measured context pressure. Calling this *per iteration* is
    the defect backlog #520 was filed against — it rewrites history the
    engine has already cached and collapses the prefix-cache hit rate from
    iteration `keep`+2 onward. The relief caller is not that: it fires only
    when the meter says the prompt is over target, which on a healthy turn
    is never.

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


_TASK_TOOL_NAMES = frozenset({"Task", "mcp__lloyd-mcp__Task"})


def _is_parallel_safe_task(tc: dict[str, Any],
                           parallel_safe_profiles: frozenset[str] | set[str]) -> bool:
    """A fresh `Task` call to a profile declared `parallel_safe` (P8).

    The profile is read off the call's own arguments, with the tool's default
    (`general-purpose`) when none is given — a default that is not parallel
    safe. A `task_id` resume never qualifies: the stored run's profile wins
    over the argument (`builtin_task._task`), so the argument says nothing
    about what would run, and two resumes of one id race `claim_history`.
    """
    if tc["function"]["name"] not in _TASK_TOOL_NAMES or not parallel_safe_profiles:
        return False
    args = tc["_args_dict"]
    if str(args.get("task_id") or "").strip():
        return False
    return str(args.get("subagent_type") or "general-purpose") in parallel_safe_profiles


def _batch_is_read_only(tool_calls: list[dict[str, Any]],
                        read_only: set[str],
                        parallel_safe_profiles: frozenset[str] | set[str] = frozenset(),
                        ) -> bool:
    """True when every call in the batch is safe to overlap with the others.

    A parse error never dispatches, and ToolSearch is intercepted in-process
    and touches only the LoadedToolSet, which `_pre_dispatch` mutates in wire
    order anyway. Everything else must carry the hint — or be a fresh `Task`
    to a parallel-safe profile, whose child is held to the read-only set. One
    writer makes the whole batch sequential: two Edits to one file, or an Edit
    racing the Read that justifies it, is not a reordering anybody asked for.
    """
    for tc in tool_calls:
        if tc["_args_dict"].get("__parse_error__"):
            continue
        name = tc["function"]["name"]
        if name == TOOLSEARCH_TOOL_NAME or name in read_only:
            continue
        if _is_parallel_safe_task(tc, parallel_safe_profiles):
            continue
        return False
    return True


def _is_task_fanout(tool_calls: list[dict[str, Any]],
                    parallel_safe_profiles: frozenset[str] | set[str]) -> bool:
    """A batch that is nothing but parallel-safe `Task` calls (P8).

    This is the one batch shape that overlaps while general parallel dispatch
    (`harness.parallel_tool_calls.enabled`) is still off: the fan-out case is
    where overlap pays most — each child is minutes of engine time — and it is
    the case whose safety does not rest on the soak that flag is waiting on,
    because the child's tool set is fixed read-only by its profile. A batch
    that mixes in a Read waits for the general flag like any other.
    """
    seen = False
    for tc in tool_calls:
        if tc["_args_dict"].get("__parse_error__"):
            continue
        if not _is_parallel_safe_task(tc, parallel_safe_profiles):
            return False
        seen = True
    return seen


def _tool_history_message(call_id: str, evt: dict[str, Any], *,
                          allow_images: bool = True) -> dict[str, Any]:
    """The ``role:"tool"`` history message for one result.

    Content stays a string. Images the turn's model can see ride on the
    private ``_image_refs`` key and become ``image_url`` parts only at send
    time (``tool_images.wire_messages``); the refs are small, so the history
    list, Inner Voice's handle on it and every estimate stay cheap.

    ``allow_images=False`` is the turn whose image input was already refused
    (#1419). Attaching refs there is what used to kill it: the route comes from
    static config, which a rejection never changes, so the next capture would
    put an image back on the wire and the next 400 would end the turn. The
    result's text still carries the element list; the note names the file the
    screenshot was saved to, which is the other thing the turn can act on.
    """
    msg: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": evt["content"],
    }
    refs = [
        r for r in (evt.get("images") or [])
        if isinstance(r, dict) and r.get("route") == "native"
        and not r.get("deduped_from") and not r.get("described")
    ]
    if not refs:
        return msg
    if not allow_images:
        where = refs[0].get("path") or "the tool-results directory"
        msg["content"] = (
            f"{msg['content']}\n[screenshot kept on disk at {where}; this turn's "
            "image input was refused, so it was not shown - drive by the "
            "element list and that file]")
        return msg
    msg["_image_refs"] = [
        {k: r[k] for k in ("path", "sha256", "mime", "width", "height", "bytes")
         if k in r}
        for r in refs
    ]
    return msg


def _cap_images(chat_messages: list[dict[str, Any]], batch_base: int) -> None:
    """Apply the outbound image cap, never touching the batch just appended."""
    if not any(isinstance(m, dict) and m.get("_image_refs") for m in chat_messages):
        return
    try:
        from app.harness.tool_images import enforce_outbound_cap
        enforce_outbound_cap(chat_messages, protect_from=batch_base)
    except Exception as exc:  # noqa: BLE001
        logger.warning("loop: image cap failed: %s", exc)


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


def _parse_failure_text(args_dict: dict[str, Any], *, meter: Any | None = None) -> str:
    """What to tell the model about a tool call whose arguments did not parse.

    Two failures arrive in the same shape and want opposite answers:

    - **The completion was cut off by the context window.** The tool did not
      run, the arguments are a prefix of what the model meant to send, and
      re-sending is the worst available move — the retry is generated with
      *less* room than the attempt that was just cut, so each one is shorter
      than the last. That is measured: round 875 re-sent the same heredoc
      three times at 3147, then 1760, then 1175 tokens. Both the old tool
      result and the vault skill told it to "retry verbatim".
    - **The model emitted malformed JSON.** Re-sending is exactly right.

    `finish_reason == "length"` is the direct signal. The fallback — low
    headroom against a large raw argument string — catches the case where
    the stream ended without one. With no meter (a bare `run_query` caller)
    the text is byte-identical to what it has always been, which is what
    keeps `tests/test_tool_dispatch.py` honest.
    """
    err = args_dict.get("error", "unknown")
    base = f"Tool call arguments could not be parsed as JSON: {err}"

    raw_len = int(args_dict.get("raw_len", 0) or 0)
    cut_by_wall = str(args_dict.get("finish_reason", "")) == "length"
    headroom = None
    if meter is not None and getattr(meter, "measured", False):
        headroom = int(meter.headroom)
        # A big argument string plus little room left is the same event
        # without the flag: the completion ran out of window mid-write.
        if headroom < int(getattr(meter, "window", 0)) * 0.10 and raw_len > 2_000:
            cut_by_wall = True

    if cut_by_wall:
        used_k = f"~{headroom // 1000}k" if headroom is not None else "very little"
        win_k = (
            f"{int(getattr(meter, 'window', 0)) // 1000}k"
            if meter is not None else "the window"
        )
        call_k = max(1, raw_len // 4000)
        return (
            f"{base}\n\n"
            f"The completion was cut off by the context window: {used_k} of "
            f"{win_k} remain, and this call was ~{call_k}k tokens. The tool "
            f"did NOT run.\n\n"
            f"Do not re-send this call. The retry is generated with less room "
            f"than this attempt had, so each one comes back shorter. Instead: "
            f"commit what is already on disk, gate, and land or abort. If "
            f"content still has to be written, write it in pieces well under "
            f"{max(1, call_k // 2)}k tokens each — prefer small Edits over "
            f"whole-file Writes, and do not Read whole files."
        )

    if headroom is not None and headroom < int(
        getattr(meter, "window", 0)
    ) * 0.15:
        return (
            f"{base}\n\nRe-send the call with valid JSON. Note the context "
            f"window is nearly full (~{headroom // 1000}k tokens of headroom "
            f"remain), so keep the arguments small."
        )
    return base


async def _pre_dispatch(
    *,
    tc: dict[str, Any],
    options: RunOptions,
    session_id: str,
    loaded_set: LoadedToolSet,
    runtime_disallowed: set[str] | None = None,
    meter: Any | None = None,
) -> NormalizedEvent | None:
    """Everything before the MCP call. Returns an early result, or None.

    Kept ordered and sequential even in a parallel batch: `mark_loaded`
    mutates the shared LoadedToolSet, and a hook deny is a decision the
    model must see in the order it made the calls.

    Every gate here reads the BARE tool name. The model is advertised bare
    names, but `MCPPool.call_tool` still dispatches the legacy
    `mcp__<server>__<tool>` spelling so old session JSON replays — which made
    that spelling a route past this function: `mcp__lloyd-mcp__Bash` matched
    nothing in a deny list carrying `Bash` and skipped every hook keyed on
    `"Bash"`, the safety deny included (#727). The deny list is normalised the
    same way, since config and plan mode write one spelling and rolled-forward
    overrides the other. The events keep `raw_name`: they record what the
    model emitted, and `call_id` is what joins them to the result.
    """
    raw_name = tc["function"]["name"]
    name = normalize_tool_name(raw_name)
    args_dict = tc["_args_dict"]
    call_id = tc["id"]

    if args_dict.get("__parse_error__"):
        msg = _parse_failure_text(args_dict, meter=meter)
        return events.tool_result(call_id=call_id, name=raw_name, content=msg, is_error=True,
                                  error_class="parse_error")

    effective_disallowed = (
        runtime_disallowed
        if runtime_disallowed is not None
        else set(options.disallowed_tools or [])
    )
    if name in {normalize_tool_name(d) for d in effective_disallowed}:
        return events.tool_result(
            call_id=call_id, name=raw_name,
            content=f"Tool {name!r} is disabled by configuration.",
            is_error=True, error_class="disabled",
        )
    # P3: an allow-list refuses everything it does not name, including a
    # name no server discovered and ToolSearch itself.
    allowed = _allowed_names(options)
    if allowed is not None and name not in allowed:
        return events.tool_result(
            call_id=call_id, name=raw_name,
            content=(f"Tool {name!r} is not available on this turn. "
                     f"Available: {', '.join(sorted(allowed))}."),
            is_error=True, error_class="disabled",
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
            call_id=call_id, name=raw_name, content=content, is_error=False,
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

    # PreToolUse — a deny beats a deliver; either one short-circuits dispatch.
    if options.hooks is not None:
        pre = await options.hooks.fire_pre_tool_use(
            session_id=session_id,
            tool_name=name,
            tool_input=args_dict,
            tool_use_id=call_id,
            tool_summary=tc.get("_summary", ""),
        )
        if pre:
            hso = pre.get("hookSpecificOutput") or {}
            deliver = hso.get("skillDeliver")
            if deliver:
                # Dispatch-time skill delivery (#536). The call was drafted and
                # is being held back so the matched SKILL.md reaches the model
                # before the action, not after it. `is_error=False` is the whole
                # point of this second outcome: the same intercept expressed as a
                # deny lands in `tool_errors` (autonomy.py:912/:927) and would
                # make the fleet look sicker precisely where it is being taught
                # something. Shape matches the synthetic ToolSearch result above.
                return events.tool_result(
                    call_id=call_id, name=raw_name,
                    content=str(deliver.get("content") or ""), is_error=False,
                )
            reason = hso.get("permissionDecisionReason") or "denied by hook"
            return events.tool_result(
                call_id=call_id, name=raw_name,
                content=f"Tool call denied: {reason}", is_error=True,
                error_class="denied",
            )

    return None


def _bound_grant_scope() -> str:
    """`policy.current_scope` if this task bound it, else "" (D4)."""
    try:
        if current_scope in contextvars.copy_context():
            return current_scope.get() or ""
    except Exception:          # pragma: no cover — never block a dispatch
        pass
    return ""


async def _execute_tool_call(
    *,
    tc: dict[str, Any],
    pool: MCPPool,
    options: RunOptions,
    session_id: str,
    runtime_disallowed: set[str] | None = None,
) -> NormalizedEvent:
    """The MCP call and everything after it. Safe to run concurrently.

    Session correlation rides in the request's `_meta` (see
    MCPPool.call_tool), not in the arguments — nothing validates arguments
    against the tool's inputSchema and unknown keys reach the handler, so
    an injected argument would arrive as though it were a real parameter.
    """
    name = tc["function"]["name"]
    # Post hooks are keyed on the bare name like the pre hooks (#727); the
    # pool resolves the legacy prefix itself, so `name` goes to it as emitted.
    hook_name = normalize_tool_name(name)
    args_dict = tc["_args_dict"]
    call_id = tc["id"]
    dispatch_args = dict(args_dict)
    # P11: the MCP call's wall time, cancel race included — what the model
    # waited. Taken here, not around the await, so every exit below can
    # report it with one subtraction.
    call_started_at = time.perf_counter()
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
            # change ledger on the far side. Set by every caller that has a
            # turn to name — the chat path, and since 2026-09-10 the two
            # background paths, which mint one per run. Empty for a bare
            # `run_query` caller, which is what turns the ledger off there.
            "turn_id": getattr(options, "turn_id", "") or "",
            "call_id": tc.get("id", "") or "",
            # #544: the queue item this turn is running, bound by the pool per
            # job. It is the aggregator's only handle on "this call is a retry
            # of one I already served", because a retried attempt gets a fresh
            # session id. On an interactive turn it is a `turn:` scope the
            # ledger records but never enforces (#767) — see
            # agent_mcp/_tool_effects.py. The option
            # wins over the contextvar: a session-backed worker turn runs in
            # the backend after a loopback POST, where the pool's contextvar
            # is empty and the router has set the option from the payload.
            "effect_scope": (getattr(options, "effect_scope", "")
                             or current_effect_scope.get()),
            # So a Task subagent this call spawns runs on the same surface.
            "surface": getattr(options, "surface", "") or "",
            # D4: the authority gate and the deny list a Task child inherits.
            # The option first (the router sets it beside its policy hook);
            # then the pool's contextvar, but only where something BOUND it —
            # its default is "worker", and reading that on a chat turn would
            # gate every chat Task child.
            "grant_scope": (getattr(options, "grant_scope", "")
                            or _bound_grant_scope()),
            # This iteration's set (plan mode, surface refresh), not the
            # turn-start list. Only Task reads it, so only Task carries it.
            "disallowed_tools": (
                sorted(runtime_disallowed if runtime_disallowed is not None
                       else (options.disallowed_tools or []))
                if hook_name == "Task" else ()),
        }
        # P9: what a `lloyd_rpc` call from inside THIS shell may not use —
        # this iteration's set plus the fixed deny list. Bash calls only, and
        # only while the feature is on, so every other call's `_meta` is
        # byte-for-byte what it was.
        if hook_name == "Bash" and rpc_policy.enabled():
            call_kw["rpc_deny"] = rpc_policy.bash_deny(
                runtime_disallowed if runtime_disallowed is not None
                else (options.disallowed_tools or []))
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
            except BaseException:
                # The dispatcher itself was cancelled (the generator closed
                # mid-batch, the pool's wait_for fired) or the wait raised.
                # `asyncio.wait` does not cancel what it waits on, so without
                # this the MCP call ran on as an orphan task nobody awaits —
                # a Bash still executing after its turn was gone (D9).
                tool_task.cancel()
                raise
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
                    is_error=True, error_class="cancelled",
                    duration_ms=_ms_since(call_started_at),
                )
    except ToolDispatchError as exc:
        if options.hooks is not None:
            await options.hooks.fire_post_tool_use_failure(
                session_id=session_id,
                tool_name=hook_name,
                tool_input=args_dict,
                error=str(exc),
                tool_use_id=call_id,
            )
        return events.tool_result(
            call_id=call_id, name=name, content=str(exc), is_error=True,
            error_class="transport", duration_ms=_ms_since(call_started_at),
        )
    except Exception as exc:
        logger.exception("loop: unexpected dispatch failure on %s", name)
        if options.hooks is not None:
            await options.hooks.fire_post_tool_use_failure(
                session_id=session_id,
                tool_name=hook_name,
                tool_input=args_dict,
                error=str(exc),
                tool_use_id=call_id,
            )
        return events.tool_result(
            call_id=call_id, name=name,
            content=f"Tool dispatch failed: {exc}", is_error=True,
            error_class="dispatch_failed",
            duration_ms=_ms_since(call_started_at),
        )
    call_ms = _ms_since(call_started_at)

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
    # Measured here, on the line above the only step that rewrites it: past
    # `maybe_spill` the text is a ~2 KB `<persisted-output>` preview of a
    # 250 KB Grep, and every length taken downstream — including the
    # `result_chars` in the transcript, which is itself truncated to 2014 —
    # describes the preview and not the answer (#1052).
    raw_chars = len(content) if isinstance(content, str) else None
    if isinstance(content, str):
        # Errors are spilled too (D10): a 200 KB traceback or a failing test
        # run is as large as any success and was the one result nothing
        # bounded. The empty-result marker stays success-only.
        if not is_error:
            content = fallback_for_empty_result(content, name)
        content = maybe_spill(
            content,
            tool_name=name,
            tool_use_id=call_id,
            session_id=session_id,
            # The block closes by telling the model how to get the rest of
            # this result back, and that has to be a call the turn can make.
            # `deep-research` spills constantly — every http_fetch of a page
            # — and has `Read` denied (#1066), so the old unconditional
            # sentence sent it reaching for a tool its own deny list refuses.
            disallowed_tools=list(options.disallowed_tools or []),
        )

    # Images the tool returned (screenshots). Persisted beside the text
    # spills, deduped, and routed per model — see app/harness/tool_images.py.
    # The event carries refs only; base64 never leaves the file and the wire.
    image_refs: list[dict[str, Any]] = []
    if result.get("images") and isinstance(content, str):
        from app.harness.tool_images import shape_tool_images
        content, image_refs, _ = await shape_tool_images(
            images=result["images"], content=content, session_id=session_id,
            call_id=call_id, tool_name=name, model=options.model,
        )

    if options.hooks is not None:
        await options.hooks.fire_post_tool_use(
            session_id=session_id,
            tool_name=hook_name,
            tool_input=args_dict,
            tool_response=content,
            tool_use_id=call_id,
        )

    return events.tool_result(
        call_id=call_id,
        name=name,
        content=content,
        is_error=is_error,
        raw_chars=raw_chars,
        images=image_refs or None,
        duration_ms=call_ms,
        handshake_ms=_handshake_ms(result),
        error_class=_result_error_class(result) if is_error else None,
    )


def _ms_since(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _handshake_ms(result: dict[str, Any]) -> int | None:
    """The pool's per-call session handshake, when it reports one (P12).

    Optional by construction: a pool that carries no `timing` — today's, and
    every test double — yields None, and the event omits the key.
    """
    timing = result.get("timing") if isinstance(result, dict) else None
    if isinstance(timing, dict):
        value = timing.get("handshake_ms")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _result_error_class(result: dict[str, Any]) -> str:
    """`mcp_error` for a protocol-level refusal, else `tool_error`.

    `MCPPool.call_tool` answers an `MCPError` from a live server with this
    exact prefix rather than raising, so the prefix is the only mark it
    leaves; anything else flagged is_error is the tool reporting a failure.
    """
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, str) and content.startswith("MCP error calling "):
        return "mcp_error"
    return "tool_error"


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

    # P3: an allow-list turn (a memory flush) gets its own set and never
    # touches the session's cached one. Its catalog is a handful of tools, so
    # caching it would invalidate the chat's loaded set by signature and the
    # next chat turn would have to re-discover every tool it had loaded.
    allowed = _allowed_names(options)
    if allowed is not None:
        return LoadedToolSet(
            catalog=catalog,
            baseline=baseline,
            enabled=enabled and TOOLSEARCH_TOOL_NAME in allowed,
            catalog_signature=tool_search_cache.catalog_signature(catalog),
            summaries=summaries,
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
    session_id: str = "",
    min_chars: int = 4096,
    disallowed_tools: list[str] | None = None,
) -> tuple[int, int]:
    """Replace the largest tool-result message contents with a truncation
    notice until at least ``target_chars`` of content has been freed.

    Operates in-place on ``chat_messages`` (each dict's ``content`` field
    is overwritten). Returns ``(num_truncated, total_chars_freed)``.

    Strategy: rank tool messages by content length descending, walk from
    the largest down, replacing each with a short error string that
    surfaces what happened to the model. Stop as soon as we've freed
    ``target_chars`` chars cumulatively, OR after we've replaced every
    tool message bigger than ``min_chars`` (smaller results aren't worth
    touching).

    **Spills before it overwrites.** This is the last relief rung and the
    only destructive one: until 2026-09-11 it replaced the content with a
    notice telling the model to "re-run the call with a narrower query",
    which at the wall is advice the turn has no room to take. When a
    ``session_id`` is available the content goes to the same per-session
    spill directory microcompaction uses and the notice names the path, so
    the evidence survives as a file. Whether the notice then says "Read that
    path" is a fact about the turn, not about the file: ``disallowed_tools``
    is that turn's deny list, and a turn with ``Read`` on it gets the only
    recovery it can take — re-run narrower — instead of an instruction its
    own policy refuses (#1066; deep-research takes this branch on every long
    turn). A failed spill still truncates — this rung runs when the
    alternative is the request being rejected outright — but says so.
    """
    from app.harness.tool_result_spill import persist_for_compaction

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
        if size > min_chars:
            candidates.append((size, i))
    candidates.sort(reverse=True)

    # Computed once for the turn, not per message: every notice this pass
    # writes answers the same question about the same tool menu.
    read_denied = tool_is_denied(READ_TOOL, disallowed_tools)

    freed = 0
    truncated = 0
    for size, idx in candidates:
        original_size = size
        msg = chat_messages[idx]
        path = None
        if session_id:
            cid = msg.get("tool_call_id") or msg.get("call_id") or ""
            content = msg.get("content")
            if isinstance(content, list):
                text = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            else:
                text = content if isinstance(content, str) else ""
            if cid and text:
                try:
                    path = persist_for_compaction(
                        text, tool_use_id=f"{cid}.truncated", session_id=session_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("loop: truncation spill failed: %s", exc)
                    path = None
        if path is not None:
            # Two endings, because the honest one depends on the turn's menu.
            # "Re-run the call with a narrower query" is what this notice said
            # before 2026-09-11, when there was no spill to point at — see the
            # docstring. It comes back on this branch not as a regression but
            # as the only recovery left to a turn that cannot open the file.
            recovery = (
                "the Read tool is not available on this turn, so that path "
                "cannot be opened from here — re-run the call with a narrower "
                "query (fewer hops, higher min_confidence, fewer max_results) "
                "for the part you need.]"
                if read_denied else
                "Read that path if you need it again.]"
            )
            notice = (
                f"[harness: tool result cleared under context pressure — "
                f"{original_size:,} chars, full content at {path}. "
                f"{recovery}"
            )
        else:
            notice = (
                f"[harness: tool result truncated by context-overflow recovery — "
                f"original was {original_size} chars. The combined tool history "
                f"exceeded the model's context window. Re-run the call with a "
                f"narrower query (smaller hops, higher min_confidence, fewer "
                f"max_results) or use a more targeted tool.]"
            )
        if isinstance(msg.get("content"), list):
            msg["content"] = [{"type": "text", "text": notice}]
        else:
            msg["content"] = notice
        msg.pop("_image_refs", None)
        freed += original_size - len(notice)
        truncated += 1
        if freed >= target_chars:
            break
    return truncated, freed
