"""One definition of a transcript entry's shape.

A session JSON's `messages` list is read by nine or ten different programs —
the chat UI, the Inner Voice timeline, the vault exporter, `session_titles`,
`session_recall`, the memory renderers, the trajectory extractor — and every
one of them dispatches on `role` and then reaches for keys by name. The shapes
were built inline in `app/routers/messages.py`, five assistant variants and
three tool-pair variants spread across the streaming loop, the cancel path and
the error path.

That was tolerable while `messages.py` was the only writer. It stops being
tolerable the moment a second writer exists: `app/run_recorder.py` persists
the same events for a background run, and a second private definition of "what
a tool result row looks like" is how the two would come to disagree about the
same turn. The codebase has paid for second definitions of "due", "healthy"
and "tell the human" already.

So the shaping lives here, both writers call it, and
`tests/test_transcript_entries.py` pins that the entries they produce for the
same events are identical.

Two shapes are deliberately conditional rather than always-present, because
every historical session on disk predates them and must keep reading the same:

  * `summary` on a tool call is omitted when empty, never stored as `""`.
  * `is_error` on a tool result's stats is omitted when the caller does not
    know — the eager per-pair path has the event and passes it; the two
    reconstruct-from-the-log paths do not, and inventing `False` there would
    turn "we never saw the result" into "the result was fine".
"""

from __future__ import annotations

import uuid
from typing import Any

#: A tool result longer than this is truncated before it reaches the
#: transcript. The full text went to the model; this is the human-facing
#: record and a 300 kB `Read` result in it helps nobody.
TOOL_RESULT_MAX_CHARS = 2000


def truncate_tool_result(text: str) -> str:
    if len(text) > TOOL_RESULT_MAX_CHARS:
        return text[:TOOL_RESULT_MAX_CHARS] + "...(truncated)"
    return text


def new_entry_id() -> str:
    return uuid.uuid4().hex[:8]


def build_assistant_text_entry(
    text: str,
    *,
    timestamp: str,
    stats: dict[str, Any] | None = None,
    reasoning: str = "",
    reasoning_ms: int = 0,
    source: str = "",
    structured: Any = None,
    cancelled: bool = False,
    synthetic_empty_terminal: bool = False,
    entry_id: str = "",
) -> dict:
    """One assistant text row.

    Covers the mid-turn segment flush, the final answer, the cancelled
    partial, the error-path partial and the empty-terminal placeholder — they
    differ only in which of the optional flags are set.

    `structured` goes to the top level as well as into `stats`: it is the
    machine answer for the turn, and a consumer reading the session JSON
    should not have to know it rides in a stats blob.
    """
    entry: dict = {
        "id": entry_id or new_entry_id(),
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "timestamp": timestamp,
    }
    if stats is not None:
        entry["stats"] = stats
    if structured:
        entry["structured"] = structured
    if reasoning:
        entry["reasoning"] = reasoning
        if reasoning_ms:
            entry["reasoning_ms"] = reasoning_ms
    if cancelled:
        entry["cancelled"] = True
    if synthetic_empty_terminal:
        entry["synthetic_empty_terminal"] = True
    if source and source != "user":
        entry["source"] = source
    return entry


def build_tool_call(call_id: str, name: str, args_json: str,
                    summary: str = "") -> dict:
    """The OpenAI-shaped tool call that rides on an assistant row.

    `summary` is the model's own one-liner, already lifted off the arguments
    by the harness. Display metadata: the transcript renders it on reload and
    the live SSE frame carries it so the bubble has it before the result
    lands.
    """
    tc: dict = {
        "id": call_id,
        "call_id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args_json},
    }
    if summary:
        tc["summary"] = summary
    return tc


def build_tool_call_entry(tool_call: dict, *, timestamp: str,
                          stats: dict[str, Any] | None = None) -> dict:
    """The assistant row that carries one tool call.

    Per-iteration LLM usage belongs here rather than on the result row — the
    model produced the call, and MCP dispatch has no token cost.
    """
    entry: dict = {
        "id": f"msg_{tool_call['call_id']}_tc",
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [tool_call],
        "timestamp": timestamp,
    }
    if stats:
        entry["stats"] = dict(stats)
    return entry


def build_tool_result_entry(call_id: str, result: str, *, timestamp: str,
                            is_error: bool | None = None) -> dict:
    stats: dict[str, Any] = {"result_chars": len(result)}
    if is_error is not None:
        stats["is_error"] = bool(is_error)
    return {
        "id": f"msg_{call_id}_result",
        "role": "tool",
        "content": [{"type": "text", "text": result}],
        "tool_call_id": call_id,
        "timestamp": timestamp,
        "stats": stats,
    }


def build_thinking_entry(turn_id: str, text: str, duration_ms: int, seq: int,
                         iteration: int, timestamp: str) -> dict:
    """Shape one reasoning phase as a message entry. Pure, for testability.

    `role="thinking"` is what keeps reasoning out of every transcript producer
    in the tree, and `content` stays empty as a second layer — see
    `app/routers/_messages_thinking.py` for the measurement behind both, and
    `tests/test_thinking_trace_transcripts.py` for what pins them.

    `seq` orders phases within a turn and keeps ids unique; `iteration` is the
    agent-loop iteration the phase belongs to, which is what a reader
    correlating a thought with the tool call it produced actually wants.
    """
    return {
        "id": f"think_{turn_id}_{seq}",
        "role": "thinking",
        "content": [],
        "timestamp": timestamp,
        "reasoning": text,
        "reasoning_ms": duration_ms,
        "thinking": {
            "chars": len(text),
            "iteration": iteration,
            "turn_id": turn_id,
        },
    }


def build_user_entry(text: str, *, timestamp: str, source: str = "",
                     entry_id: str = "") -> dict:
    """The prompt row a background run opens its transcript with.

    A transcript whose first row is the model answering is unreadable — the
    question is the half a human needs. `source` names the producer for a row
    that no human typed, matching the convention `messages.py` uses for
    ambient turns.
    """
    entry: dict = {
        "id": entry_id or new_entry_id(),
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "timestamp": timestamp,
    }
    if source and source != "user":
        entry["source"] = source
    return entry
