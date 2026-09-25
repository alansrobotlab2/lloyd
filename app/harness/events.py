"""Normalized event stream emitted by the harness loop.

We use a single `NormalizedEvent` TypedDict with a `type` discriminator
rather than a class hierarchy. The SDK we replaced used isinstance checks
against AssistantMessage/UserMessage/etc.; downstream code now does
`evt["type"] == "..."` and reads the relevant payload fields.

Event types:

    system            — emitted once at start; carries session_id, model
    text_delta        — incremental assistant text chunk (analog of SDK
                        text_delta)
    thinking_delta    — incremental reasoning chunk (vLLM
                        delta.reasoning_content on <=0.22, renamed to
                        delta.reasoning on 0.23+, under --reasoning-parser
                        qwen3)
    thinking_done     — final accumulated reasoning text for this
                        assistant message, plus `duration_ms`: wall time
                        from the first reasoning chunk to the last one
    tool_call         — a single fully-accumulated tool call from the
                        assistant (after finish_reason="tool_calls" or
                        stream end)
    tool_result       — result of dispatching a tool_call through MCP
    assistant_message — assistant turn flush event (text + tool_calls +
                        thinking), emitted before tool dispatch begins
    result            — terminal event with stop_reason, usage, num_turns
    stream_raw        — escape hatch for chunks we couldn't normalize
                        (e.g. malformed qwen3_xml tool_calls); carries
                        the raw line so callers can persist for forensics
    iteration_retry   — the iteration just streamed is being discarded and
                        re-requested; carries how many text / thinking
                        characters of it were already yielded as deltas,
                        so a consumer can take them back off what it
                        accumulated (review 2026-09-24, X2)

Event constructors below are thin helpers — they exist so call sites
read clearly (`events.text_delta("hi")`) without the TypedDict noise.
"""

from typing import Any, Literal, TypedDict


class NormalizedEvent(TypedDict, total=False):
    type: Literal[
        "system",
        "text_delta",
        "thinking_delta",
        "thinking_done",
        "tool_call",
        "tool_result",
        "assistant_message",
        "result",
        "stream_raw",
        "iteration_retry",
    ]

    # system
    session_id: str
    model: str

    # text_delta / thinking_delta / thinking_done
    text: str

    # tool_call
    call_id: str
    name: str
    args_json: str
    args_dict: dict[str, Any]
    summary: str  # model-written one-liner for the transcript UI

    # tool_result
    content: str
    is_error: bool

    # thinking_done / assistant_message / result
    duration_ms: int

    # assistant_message
    tool_calls: list[dict[str, Any]]
    thinking: str
    iteration: int
    finish_reason: str  # vLLM's stop reason for THIS iteration: "stop" | "tool_calls" | "length" | ...

    # result
    stop_reason: Literal["stop", "tool_calls", "max_turns", "cancelled", "error"]
    usage: dict[str, int]
    num_turns: int
    response_text: str
    tool_calls_total: int
    tool_calls_captioned: int
    structured: dict[str, Any] | None
    structured_error: str

    # stream_raw
    raw: str
    error: str

    # iteration_retry
    reason: str
    attempt: int
    discarded_text_chars: int
    discarded_thinking_chars: int


def system(*, session_id: str, model: str) -> NormalizedEvent:
    return {"type": "system", "session_id": session_id, "model": model}


def text_delta(text: str) -> NormalizedEvent:
    return {"type": "text_delta", "text": text}


def thinking_delta(text: str) -> NormalizedEvent:
    return {"type": "thinking_delta", "text": text}


def thinking_done(text: str, duration_ms: int = 0) -> NormalizedEvent:
    """Reasoning phase complete.

    ``duration_ms`` spans the first reasoning chunk to the last one — the
    time actually spent generating thinking, not the iteration's wall
    time, which also covers prefill and the answer that follows.
    """
    return {
        "type": "thinking_done",
        "text": text,
        "duration_ms": duration_ms,
    }


def tool_call(
    *,
    call_id: str,
    name: str,
    args_json: str,
    args_dict: dict[str, Any],
    summary: str = "",
) -> NormalizedEvent:
    """One fully-accumulated tool call.

    ``summary`` is the model's own one-line description of what this call
    is doing, lifted off the arguments by ``_commit_tool_calls``. It is
    display metadata for the transcript UI and is absent from both
    ``args_json`` and ``args_dict`` — empty string when the model omitted
    it, or when the tool declares a real ``summary`` parameter of its own.
    """
    return {
        "type": "tool_call",
        "call_id": call_id,
        "name": name,
        "args_json": args_json,
        "args_dict": args_dict,
        "summary": summary,
    }


def tool_result(*, call_id: str, name: str, content: str, is_error: bool = False,
                raw_chars: int | None = None,
                images: list[dict] | None = None) -> NormalizedEvent:
    """One tool call's result, as the model is about to be shown it.

    ``raw_chars`` is how long ``content`` was BEFORE the caller's spill
    step replaced it with a ``<persisted-output>`` preview. It has to be
    carried, not recomputed: on a spilled result ``len(content)`` here is
    the preview's length, so the size of the tool's actual answer exists
    nowhere else (#1052). Every early-return site — parse error, disabled
    tool, hook deny, dispatch failure, in-flight cancel — builds its own
    short string and is what the default is for; ``None`` from a caller
    means "there was no shaping step", which is the same number. The key
    is absent when the size is genuinely unknown (a result that is not
    text), and the writers downstream omit it from the row rather than
    inventing a length for it.
    """
    evt: NormalizedEvent = {
        "type": "tool_result",
        "call_id": call_id,
        "name": name,
        "content": content,
        "is_error": is_error,
    }
    if raw_chars is None and isinstance(content, str):
        raw_chars = len(content)
    if raw_chars is not None:
        evt["raw_chars"] = int(raw_chars)
    # ``images``: ImageRefs (app/harness/tool_images.py) — paths and hashes,
    # never base64. Absent when the tool returned none.
    if images:
        evt["images"] = list(images)
    return evt


def assistant_message(
    *,
    text: str,
    tool_calls: list[dict[str, Any]],
    thinking: str = "",
    usage: dict[str, int] | None = None,
    duration_ms: int = 0,
    iteration: int = 0,
    finish_reason: str = "stop",
    context: dict[str, Any] | None = None,
) -> NormalizedEvent:
    """Emitted at end of each agent-loop iteration.

    ``usage`` carries the vLLM completion's token counts for THIS
    iteration only (input/output/cached). Per-iteration usage lets the
    UI surface stats on every persisted LLM-output row instead of only
    the final assistant message.

    ``duration_ms`` is the wall-clock duration of just this iteration's
    chat completion. ``iteration`` is the 1-based index inside the
    agent loop.

    ``context`` is the turn's live context-window position as of this
    iteration — ``{input_tokens, context_window, headroom, fraction}`` — or
    None before an engine has reported a prompt size. The Inner Voice
    observer reads it to stop nudging a turn with no room to act on a
    nudge; nothing else may assume it is present.

    ``finish_reason`` is vLLM's stop reason for the assistant turn:
    ``"stop"`` (model emitted EOS — harness will terminate iff there
    are no tool_calls), ``"tool_calls"`` (model wants to dispatch
    tools — harness will loop), ``"length"`` (max_tokens hit — also a
    terminal state). IV uses this to distinguish "primary done" from
    "primary mid-thought" on text-only iterations.
    """
    return {
        "type": "assistant_message",
        "text": text,
        "tool_calls": tool_calls,
        "thinking": thinking,
        "usage": usage or {},
        "duration_ms": duration_ms,
        "iteration": iteration,
        "finish_reason": finish_reason,
        "context": context,
    }


def result(
    *,
    stop_reason: str,
    usage: dict[str, int] | None = None,
    num_turns: int = 0,
    duration_ms: int = 0,
    response_text: str = "",
    tool_calls_total: int = 0,
    tool_calls_captioned: int = 0,
    structured: dict | None = None,
    structured_error: str = "",
) -> NormalizedEvent:
    """`tool_calls_*` count only tools whose schema carried the injected
    `summary` parameter — the caption rate for this turn. It is reported
    because the failure mode is a ratchet, not a constant: `arguments` is
    replayed to the engine as history, so the first call that omits the
    caption becomes the model's own most recent example of calling that tool
    and the session locks into omitting it. Across the 16 sessions after the
    feature landed, every one whose first Bash call carried a caption stayed
    above 95%; both that missed stayed below 26%. A rate is the only way to
    see which side a session fell on."""
    return {
        "type": "result",
        "stop_reason": stop_reason,  # type: ignore[typeddict-item]
        "usage": usage or {},
        "num_turns": num_turns,
        "duration_ms": duration_ms,
        "response_text": response_text,
        "tool_calls_total": tool_calls_total,
        "tool_calls_captioned": tool_calls_captioned,
        # Present only when RunOptions.final_schema was set. `structured` is
        # the parsed object; `structured_error` says why there is none —
        # including the deliberate skips, so a caller can tell "the model
        # refused" from "the turn died at its budget and has no verdict".
        "structured": structured,
        "structured_error": structured_error,
    }


def stream_raw(raw: str, error: str = "") -> NormalizedEvent:
    return {"type": "stream_raw", "raw": raw, "error": error}


def iteration_retry(
    *,
    reason: str,
    attempt: int,
    discarded_text_chars: int = 0,
    discarded_thinking_chars: int = 0,
) -> NormalizedEvent:
    """The iteration in flight is thrown away and requested again.

    The loop has already yielded that iteration's `text_delta` /
    `thinking_delta` events, and every consumer appends them to a running
    buffer. The counts are the characters of each kind that were yielded
    for the discarded attempt — only the tail of the current iteration, never
    earlier ones — so a consumer trims exactly that many off the end of its
    buffer and the retried attempt streams into the gap. `reason` names the
    trigger (`stream_stalled`, `echo_guard`, ...) and `attempt` counts from 1.
    """
    return {
        "type": "iteration_retry",
        "reason": reason,
        "attempt": attempt,
        "discarded_text_chars": max(0, int(discarded_text_chars or 0)),
        "discarded_thinking_chars": max(0, int(discarded_thinking_chars or 0)),
    }


def trim_discarded(text: str, thinking: str, evt: NormalizedEvent) -> tuple[str, str]:
    """Take an `iteration_retry`'s discarded deltas back off two buffers.

    The one definition every consumer that accumulates deltas uses, so the
    chat router and the background recorder cannot come to disagree about
    what a retried iteration leaves behind. Clamped at the buffer's length: a
    consumer that flushed part of the attempt already (a thinking phase that
    reached `thinking_done`) holds less than was discarded, and must end at
    empty, never raise.
    """
    n_text = max(0, int(evt.get("discarded_text_chars") or 0))
    n_think = max(0, int(evt.get("discarded_thinking_chars") or 0))
    if n_text:
        text = text[: max(0, len(text) - n_text)]
    if n_think:
        thinking = thinking[: max(0, len(thinking) - n_think)]
    return text, thinking
