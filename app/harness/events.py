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
                        delta.reasoning_content under
                        --reasoning-parser qwen3)
    thinking_done     — final accumulated reasoning text for this
                        assistant message
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

    # tool_result
    content: str
    is_error: bool

    # assistant_message
    tool_calls: list[dict[str, Any]]
    thinking: str
    iteration: int

    # result
    stop_reason: Literal["stop", "tool_calls", "max_turns", "cancelled", "error"]
    usage: dict[str, int]
    num_turns: int
    duration_ms: int
    response_text: str

    # stream_raw
    raw: str
    error: str


def system(*, session_id: str, model: str) -> NormalizedEvent:
    return {"type": "system", "session_id": session_id, "model": model}


def text_delta(text: str) -> NormalizedEvent:
    return {"type": "text_delta", "text": text}


def thinking_delta(text: str) -> NormalizedEvent:
    return {"type": "thinking_delta", "text": text}


def thinking_done(text: str) -> NormalizedEvent:
    return {"type": "thinking_done", "text": text}


def tool_call(*, call_id: str, name: str, args_json: str, args_dict: dict[str, Any]) -> NormalizedEvent:
    return {
        "type": "tool_call",
        "call_id": call_id,
        "name": name,
        "args_json": args_json,
        "args_dict": args_dict,
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
) -> NormalizedEvent:
    """Emitted at end of each agent-loop iteration.

    ``usage`` carries the vLLM completion's token counts for THIS
    iteration only (input/output/cached). Per-iteration usage lets the
    UI surface stats on every persisted LLM-output row instead of only
    the final assistant message.

    ``duration_ms`` is the wall-clock duration of just this iteration's
    chat completion. ``iteration`` is the 1-based index inside the
    agent loop.
    """
    return {
        "type": "assistant_message",
        "text": text,
        "tool_calls": tool_calls,
        "thinking": thinking,
        "usage": usage or {},
        "duration_ms": duration_ms,
        "iteration": iteration,
    }


def result(
    *,
    stop_reason: str,
    usage: dict[str, int] | None = None,
    num_turns: int = 0,
    duration_ms: int = 0,
    response_text: str = "",
) -> NormalizedEvent:
    return {
        "type": "result",
        "stop_reason": stop_reason,  # type: ignore[typeddict-item]
        "usage": usage or {},
        "num_turns": num_turns,
        "duration_ms": duration_ms,
        "response_text": response_text,
    }


def stream_raw(raw: str, error: str = "") -> NormalizedEvent:
    return {"type": "stream_raw", "raw": raw, "error": error}
