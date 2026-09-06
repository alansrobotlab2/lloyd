"""Harness exception types.

Callers should generally let `HarnessError` subclasses propagate — they
carry forensic detail that the streaming consumer in `messages.py`
catches and persists to the session event log.
"""


class HarnessError(Exception):
    """Base class for harness errors."""


class ParseError(HarnessError):
    """vLLM SSE chunk could not be parsed.

    Wraps a raw SSE line (or accumulated tool-call argument string) that
    failed `json.loads`. The harness emits a `stream_raw` event with the
    raw payload before raising; downstream code can usually keep going.
    """

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class ToolDispatchError(HarnessError):
    """MCP tool dispatch failed (server unreachable, invalid args, etc.).

    Caller surfaces this as a tool_result with `is_error=True` so the
    model can retry.
    """

    def __init__(self, tool_name: str, message: str):
        super().__init__(f"{tool_name}: {message}")
        self.tool_name = tool_name


class MaxTurnsExceeded(HarnessError):
    """Loop hit the configured `max_turns` ceiling."""

    def __init__(self, max_turns: int):
        super().__init__(f"max_turns={max_turns} exceeded")
        self.max_turns = max_turns


class ContextOverflowError(HarnessError):
    """vLLM returned 400 because the prompt exceeded the model's context window.

    The harness catches this in ``run_query`` and attempts recovery by
    truncating the largest tool result(s) in chat_messages and re-prompting.
    Distinct from generic ``httpx.HTTPStatusError`` so the loop only
    recovers from this specific failure mode (other 400s should still surface).
    """

    def __init__(self, message: str, *, requested_input_tokens: int | None = None):
        super().__init__(message)
        self.requested_input_tokens = requested_input_tokens


class StreamStalledError(HarnessError):
    """The SSE stream went quiet mid-generation.

    Raised only after the engine has already sent at least one line and
    then produced nothing for `harness.stream_chunk_timeout_seconds`.
    Time-to-first-line is deliberately NOT bounded by this: a cold prefill
    emits no bytes at all, and the secondary slot runs llama.cpp with
    `--parallel 1`, so a queued request legitimately sits silent for as
    long as the request ahead of it takes. Those are governed by
    `RunOptions.request_timeout_s`.

    Distinct from a plain read timeout because `client.stream_chat` sets
    httpx `read=None` — without this, a wedged engine mid-generation
    blocks the turn forever.
    """

    def __init__(self, timeout_s: float, *, lines_seen: int = 0):
        super().__init__(
            f"stream produced no data for {timeout_s:g}s after {lines_seen} line(s)"
        )
        self.timeout_s = timeout_s
        self.lines_seen = lines_seen
