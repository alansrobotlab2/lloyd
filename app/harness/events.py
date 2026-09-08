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


def tool_result(*, call_id: str, name: str, content: str, is_error: bool = False) -> NormalizedEvent:
    return {
        "type": "tool_result",
        "call_id": call_id,
        "name": name,
        "content": content,
        "is_error": is_error,
    }


def assistant_message(
    *,
    text: str,
    tool_calls: list[dict[str, Any]],
    thinking: str = "",
    usage: dict[str, int] | None = None,
    duration_ms: int = 0,
    iteration: int = 0,
    finish_reason: str = "stop",
) -> NormalizedEvent:
    """Emitted at end of each agent-loop iteration.

    ``usage`` carries the vLLM completion's token counts for THIS
    iteration only (input/output/cached). Per-iteration usage lets the
    UI surface stats on every persisted LLM-output row instead of only
    the final assistant message.

    ``duration_ms`` is the wall-clock duration of just this iteration's
    chat completion. ``iteration`` is the 1-based index inside the
    agent loop.

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
