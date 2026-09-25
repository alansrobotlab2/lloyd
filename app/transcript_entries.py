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

Three shapes are deliberately conditional rather than always-present, because
every historical session on disk predates them and must keep reading the same:

  * `summary` on a tool call is omitted when empty, never stored as `""`.
  * `is_error` on a tool result's stats is omitted when the caller does not
    know — the eager per-pair path has the event and passes it; the two
    reconstruct-from-the-log paths do not, and inventing `False` there would
    turn "we never saw the result" into "the result was fine".
  * `raw_chars` on a tool result's stats is omitted for the same reason. It
    is the length of the tool's answer before the harness truncated or
    spilled it, which is the only number that says how big the answer was —
    `result_chars` is measured on the truncated text and saturates at
    `TOOL_RESULT_MAX_CHARS + len("...(truncated)")`. A path holding only the
    truncated string cannot recover it, so absence is the answer it gives,
    never `0` and never the cap.
  * `persisted_path` on a tool result's stats is present only when the row
    is a `<persisted-output>` pointer (D1, review 2026-09-24): the full
    result is in that file and `result_chars` measures the pointer.
  * `turn_id` on a user, tool-call or tool-result row is omitted when the
    writer does not know it. It names the turn that wrote the row, so a
    reader can group a turn's rows without inferring the boundary from
    roles (review 2026-09-24, X3); rows written before it read the same.
"""

from __future__ import annotations

import uuid
from typing import Any

#: A tool result longer than this does not go into the transcript inline.
#: The full text went to the model in the turn that made the call; the
#: transcript is also what the NEXT turn's history is rebuilt from, so since
#: review 2026-09-24 (D1) a longer result is written to the session's spill
#: directory and the row carries `maybe_spill`'s `<persisted-output>` pointer
#: (path + preview) instead of a bare 2 KB cut. `truncate_tool_result` is the
#: old cut, kept for the kill switch and for a spill that failed to write.
TOOL_RESULT_MAX_CHARS = 2000


def truncate_tool_result(text: str) -> str:
    if len(text) > TOOL_RESULT_MAX_CHARS:
        return text[:TOOL_RESULT_MAX_CHARS] + "...(truncated)"
    return text


def transcript_spill_enabled() -> bool:
    """`compaction.transcript_spill.enabled`, defaulting to on.

    Read per call, like `run_recorder.recording_enabled`, so flipping the
    switch reaches the next row written rather than the next boot.
    """
    try:
        from app.config import CONFIG
        block = ((CONFIG.get("compaction") or {}).get("transcript_spill")
                 or {})
        return bool(block.get("enabled", True))
    except Exception:
        return True


def shape_tool_result_for_transcript(
    content: Any,
    *,
    call_id: str,
    session_id: str,
    tool_name: str = "",
    disallowed_tools=None,
) -> str:
    """The text a tool-result row stores: a pointer when it is long (D1).

    Four cases, in order:

      * already a `<persisted-output>` block — the live turn spilled it at
        50k and the block names the file; kept whole, never cut at 2k, which
        used to slice the path's recovery sentence off the end.
      * no session, or the switch off — the old 2 KB cut.
      * over `TOOL_RESULT_MAX_CHARS` — written to
        `<sid>.tool-results/<call_id>.{txt,json}` by the same `maybe_spill`
        the loop uses, at this lower threshold, so the transcript pointer and
        the live-turn pointer are one vocabulary (X5) and microcompact already
        knows how to shrink it to its header without losing the path.
      * the write failed — `maybe_spill` hands back the original, and the
        row falls back to the old cut rather than storing 40 kB inline.

    Same file name as a live spill of the same call is harmless by
    construction: the live spill fires only at 50k, and then this function
    sees its block and writes nothing; below 50k nothing else has written the
    call's file except microcompact's `persist_for_compaction`, which writes
    the same content under the same name.

    `disallowed_tools` is the writing turn's deny list, so the block's
    recovery sentence offers only what that session can do (#1066).
    """
    from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG, maybe_spill

    text = content if isinstance(content, str) else str(content or "")
    if text.startswith(PERSISTED_OUTPUT_TAG):
        return text
    if not session_id or not call_id or not transcript_spill_enabled():
        return truncate_tool_result(text)
    try:
        out = maybe_spill(text, tool_name=tool_name, tool_use_id=call_id,
                          session_id=session_id,
                          threshold=TOOL_RESULT_MAX_CHARS,
                          disallowed_tools=list(disallowed_tools or []))
    except Exception:  # noqa: BLE001 — the row matters more than the file
        out = text
    return out if out is not text else truncate_tool_result(text)


def persisted_path_of(text: str) -> str:
    """The file a `<persisted-output>` block points at, or `""`.

    Read off the block rather than passed alongside it, so the reconstruct-
    from-log paths (which hold only the shaped string) record it too.
    """
    from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG
    if not isinstance(text, str) or not text.startswith(PERSISTED_OUTPUT_TAG):
        return ""
    head = text[:2048]
    marker = "Full output saved to: "
    i = head.find(marker)
    if i == -1:
        return ""
    return head[i + len(marker):].split("\n", 1)[0].strip()


def tool_result_preview(text: str) -> str:
    """A tool result's text for a human-facing excerpt.

    For a pointer block that is the preview, prefixed with a short
    `[full result on disk]` tag, not the tag line and the absolute path —
    the vault exporters keep 300 chars of a result, and a path eats half of
    them. Anything else comes back unchanged.
    """
    path = persisted_path_of(text)
    if not path:
        return text
    marker = "Preview ("
    i = text.find(marker)
    body = ""
    if i != -1:
        nl = text.find("\n", i)
        body = text[nl + 1:] if nl != -1 else ""
    return f"[full result on disk] {body}".rstrip()


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
    turn_id: str = "",
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
    # Which turn wrote the row. A turn persists several text rows (one per
    # segment between tool calls), and a reader that watched the turn live —
    # the voice worker speaking it as it streams — needs to recognise every
    # one of them afterwards, or it says the reply a second time.
    if turn_id:
        entry["turn_id"] = turn_id
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
                          stats: dict[str, Any] | None = None,
                          turn_id: str = "") -> dict:
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
    if turn_id:
        entry["turn_id"] = turn_id
    return entry


def build_tool_result_entry(call_id: str, result: str, *, timestamp: str,
                            is_error: bool | None = None,
                            raw_chars: int | None = None,
                            images: list[dict] | None = None,
                            turn_id: str = "") -> dict:
    """The `role="tool"` row. `result` is already shaped
    (`shape_tool_result_for_transcript`).

    `raw_chars` is the length the tool's answer had before the harness
    shaped it, and only the two eager paths have it — pass nothing on a
    row reconstructed from the call log, which holds only the truncated
    text (#1052).
    """
    stats: dict[str, Any] = {"result_chars": len(result)}
    # D1: where the full result lives when the row is a pointer. Omitted
    # otherwise, so a row that holds its whole result reads as before.
    persisted = persisted_path_of(result)
    if persisted:
        stats["persisted_path"] = persisted
    if raw_chars is not None:
        stats["raw_chars"] = int(raw_chars)
    if is_error is not None:
        stats["is_error"] = bool(is_error)
    row = {
        "id": f"msg_{call_id}_result",
        "role": "tool",
        "content": [{"type": "text", "text": result}],
        "tool_call_id": call_id,
        "timestamp": timestamp,
        "stats": stats,
    }
    if turn_id:
        row["turn_id"] = turn_id
    # Screenshots: refs to files under `<sid>.tool-results/`, never base64
    # (app/harness/tool_images.py). Omitted when there are none, so every
    # pre-existing row reads the same.
    if images:
        from app.harness.tool_images import row_refs
        refs = row_refs(images)
        if refs:
            row["images"] = refs
    return row


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
                     entry_id: str = "", turn_id: str = "") -> dict:
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
    if turn_id:
        entry["turn_id"] = turn_id
    return entry
