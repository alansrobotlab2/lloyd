"""Microcompaction — clear stale tool results inline before they reach the wall.

Modeled on Claude Code's `microCompact.ts`. Cheaper than full LLM
summarization: this is a pure structural pass that replaces older
compactable-tool results with a short marker, while keeping the most
recent N inline. The model can re-read the persisted file (via the
spill mechanism in :mod:`app.harness.tool_result_spill`) if it still
needs the content.

Two triggers, evaluated together:

  1. **Count-based.** When more than ``count_threshold`` compactable
     tool results are in history, keep only the most recent
     ``keep_recent_tools`` and clear the rest.

  2. **Spill-aware.** Any tool result already containing the
     ``<persisted-output>`` spill marker (see
     :mod:`app.harness.tool_result_spill`) AND older than the last
     ``keep_recent_tools`` results gets its inline preview dropped
     entirely — the file path stays in the marker, so the model can
     ``Read`` it on demand.

There is no time-based trigger. Lloyd has no prompt cache to align
with, so the cache-TTL heuristic Claude Code uses doesn't apply here.

Compactable tool list defaults to ``Read, Bash, Grep, Glob, Edit,
Write`` plus any namespaced ``mcp__*`` variants of those names. The
caller can override via the ``compactable_tools`` argument.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG

logger = logging.getLogger("lloyd-microcompact")


DEFAULT_COMPACTABLE_TOOLS: tuple[str, ...] = (
    "Read", "Bash", "Grep", "Glob", "Edit", "Write",
)

# Marker that replaces a cleared tool result. Kept terse — the model
# only needs to know the result was here and is now gone.
CLEARED_MARKER = "[Old tool result content cleared — retrieve via Read if needed]"


def _is_compactable_tool_call(name: str, allow: set[str]) -> bool:
    """Match either bare ``Read`` or namespaced ``mcp__lloyd-mcp__Read``."""
    if not name:
        return False
    if name in allow:
        return True
    # Namespaced form: mcp__<server>__<tool>
    if "__" in name:
        bare = name.rsplit("__", 1)[-1]
        return bare in allow
    return False


def _tool_result_text(message: dict) -> str:
    """Extract the textual portion of a tool message's content for marker
    detection. Handles both string and structured content.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _replace_tool_content(message: dict, new_text: str) -> dict:
    """Return a shallow copy of ``message`` with its ``content`` replaced
    by a single text-block payload. Preserves all other fields (role,
    tool_call_id, stats, etc.).
    """
    out = dict(message)
    # Match the original shape: if it was a string, keep a string; if it
    # was a structured list, keep a list. Mixed-content callers see the
    # text variant either way — a cleared result has no other blocks
    # worth preserving.
    if isinstance(message.get("content"), str):
        out["content"] = new_text
    else:
        out["content"] = [{"type": "text", "text": new_text}]
    return out


def microcompact(
    messages: list[dict],
    *,
    keep_recent_tools: int = 5,
    count_threshold: int = 20,
    compactable_tools: Iterable[str] = DEFAULT_COMPACTABLE_TOOLS,
) -> tuple[list[dict], int]:
    """Replace stale compactable tool results with a cleared marker.

    Returns ``(new_messages, cleared_count)``. ``new_messages`` is a
    fresh list — the input is not mutated. Tool messages whose paired
    tool_call wasn't compactable are passed through unchanged.

    Behavior:

      * Walks ``messages`` once to map ``tool_call_id`` →
        ``tool_name`` from assistant messages with ``tool_calls``.
      * Counts compactable tool results; if count ≤ ``count_threshold``,
        only the spill-aware path runs.
      * Otherwise, identifies the most recent ``keep_recent_tools``
        compactable results and clears every older one.
      * Spill-aware: any older result that already contains the
        ``<persisted-output>`` marker is cleared too (its content is
        already on disk — keeping the inline preview is pure waste).

    Why this is safe even though the model just sees "[cleared]":
    the persisted content is reachable via the file path in the spill
    marker, and the disk spill module always writes the full original.
    The model can ``Read`` it back. Without spill, this would be
    lossy — but Lloyd already spills oversized results.
    """
    if not messages:
        return list(messages), 0

    allow: set[str] = {t for t in compactable_tools}

    # Pass 1: build tool_call_id → tool_name map. Assistant messages
    # carry tool_calls; we trust that mapping over any name on the tool
    # message itself (which Lloyd doesn't always populate consistently).
    tc_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            cid = tc.get("id") or tc.get("call_id")
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            if cid:
                tc_id_to_name[cid] = name

    # Pass 2: collect indices of compactable tool-result messages, in
    # order. A "tool" role message paired to a compactable tool_call is
    # compactable.
    compactable_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        cid = msg.get("tool_call_id") or msg.get("call_id") or ""
        name = tc_id_to_name.get(cid, "")
        if _is_compactable_tool_call(name, allow):
            compactable_indices.append(i)

    if not compactable_indices:
        return list(messages), 0

    # Decide which indices to clear.
    to_clear: set[int] = set()

    # Count-based: if over threshold, clear all but the last
    # ``keep_recent_tools``.
    if len(compactable_indices) > count_threshold:
        keep_count = max(0, keep_recent_tools)
        cutoff = len(compactable_indices) - keep_count
        to_clear.update(compactable_indices[:cutoff])

    # Spill-aware: any compactable result older than the last
    # ``keep_recent_tools`` that already contains a persisted-output
    # marker is wasted inline space — clear it.
    if compactable_indices:
        keep_count = max(0, keep_recent_tools)
        spill_cutoff = len(compactable_indices) - keep_count
        for j, idx in enumerate(compactable_indices[:spill_cutoff]):
            if PERSISTED_OUTPUT_TAG in _tool_result_text(messages[idx]):
                to_clear.add(idx)

    if not to_clear:
        return list(messages), 0

    # Pass 3: build the output list with cleared messages replaced.
    out: list[dict] = []
    cleared = 0
    for i, msg in enumerate(messages):
        if i in to_clear:
            out.append(_replace_tool_content(msg, CLEARED_MARKER))
            cleared += 1
        else:
            out.append(msg)

    if cleared:
        logger.info(
            "microcompact: cleared %d/%d compactable tool results "
            "(keep_recent=%d, count_threshold=%d)",
            cleared, len(compactable_indices),
            keep_recent_tools, count_threshold,
        )

    return out, cleared


__all__ = [
    "DEFAULT_COMPACTABLE_TOOLS",
    "CLEARED_MARKER",
    "microcompact",
]
