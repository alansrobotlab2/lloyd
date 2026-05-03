"""Adapter helpers between persisted session JSON and the harness API.

The chat router (`messages.py`) and the Inner Voice glue (`_messages_inner_voice.py`)
both share these primitives:

  * `_emit`         — push a named SSE event onto a turn's broker queue
  * `_content_to_string` — flatten content blocks to plain strings (vLLM
                          rejects list-shaped content on tool messages
                          and on assistant messages with tool_calls)
  * `_prepare_messages_for_harness` — strip UI-only fields and normalize
                                      a session-history slice for vLLM

Pulled out of `messages.py` so the IV module can reuse `_emit` without
re-importing the chat router (which would close the import cycle).
"""

from __future__ import annotations

import json
from typing import Any

from app.sessions_io import SessionTurn


async def _emit(turn: SessionTurn, event: str, data: dict) -> None:
    """Push a named event onto the turn's broker queue.

    Auto-tags `source` (user/ambient/system) and `turn_id` so SSE clients
    can style ambient output distinctly without having to correlate with
    the session event.
    """
    data.setdefault("source", turn.source)
    data.setdefault("turn_id", turn.turn_id)
    await turn.events.put({"event": event, "data": data})


def _content_to_string(content: Any) -> str:
    """Flatten persisted content (string OR list[{type:text,text:...}]) to a string.

    vLLM's OpenAI endpoint rejects list-shaped content on `role:"tool"`
    and is unreliable on `role:"assistant"` when content is an empty
    list alongside `tool_calls`. We normalize everywhere to a plain
    string so the chat-completions server doesn't 400 us mid-turn.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text") or block.get("content") or ""
                if t:
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


async def _prepare_messages_for_harness(history: list[dict]) -> list[dict]:
    """Normalize a compacted session history for vLLM.

    Strips UI-only fields (id, timestamp, stats, source, cancelled,
    subliminal, reasoning) and non-conversation roles (system, subliminal).
    Coerces all content blocks to plain strings — vLLM rejects list
    content on tool messages and on assistant messages that also carry
    tool_calls.
    """
    keep_roles = {"user", "assistant", "tool"}
    out = []
    for m in history:
        role = m.get("role", "")
        if role not in keep_roles:
            continue
        msg: dict[str, Any] = {"role": role}
        msg["content"] = _content_to_string(m.get("content", ""))
        if role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                clean_tcs = []
                for tc in tcs:
                    fn = dict(tc.get("function", {}))
                    args = fn.get("arguments", "")
                    if isinstance(args, (dict, list)):
                        fn["arguments"] = json.dumps(args)
                    elif not isinstance(args, str):
                        fn["arguments"] = str(args)
                    entry = {
                        "id": tc.get("id") or tc.get("call_id", ""),
                        "type": tc.get("type", "function"),
                        "function": fn,
                    }
                    clean_tcs.append(entry)
                msg["tool_calls"] = clean_tcs
                if not msg["content"]:
                    msg["content"] = ""
        elif role == "tool":
            tc_id = m.get("tool_call_id", "")
            if tc_id:
                msg["tool_call_id"] = tc_id
        out.append(msg)
    return out
