"""Tool-result disk spill — persist oversized tool results to file.

Modeled on Claude Code's `toolResultStorage.ts`. When a tool returns a
result larger than ``SPILL_THRESHOLD_CHARS``, the full content is written
to ``<SESSIONS_DIR>/<session_id>.tool-results/<tool_use_id>.{txt,json}``
and the in-prompt content is replaced with a ``<persisted-output>``
block containing:

  * total size + filepath (so the model can read it back via the ``Read``
    tool if it needs more detail)
  * a short preview (first ``PREVIEW_CHARS`` chars, cut at a newline)
  * a "...(more)" marker

This solves two problems at once:

  1. **Context overflow.** A 250KB Grep result no longer crowds out the
     working set; the model sees ~2KB inline and can re-read on demand.
  2. **Information loss.** Inline truncation drops everything past the
     cut point. Spill keeps the full result on disk — the model can
     Grep/Read into it for the bits it actually wants.

Empty-result guard: a tool that returns ``""`` / whitespace can cause
some local models to emit a stop token and end the turn with no output.
We replace empty results with ``"({tool_name} completed with no output)"``
so the model always has SOMETHING to react to.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app.paths import SESSIONS_DIR

logger = logging.getLogger("lloyd-harness-spill")


SPILL_THRESHOLD_CHARS = 50_000
PREVIEW_CHARS = 2_000

PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"


def _spill_dir(session_id: str) -> Path:
    """Per-session directory for spilled tool results."""
    return SESSIONS_DIR / f"{session_id}.tool-results"


def _spill_path(session_id: str, tool_use_id: str, *, is_json: bool) -> Path:
    ext = "json" if is_json else "txt"
    # Sanitize tool_use_id to a safe filename (it's already constrained
    # by vLLM's id format, but defense-in-depth).
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_use_id)[:128] or "unknown"
    return _spill_dir(session_id) / f"{safe}.{ext}"


def _looks_like_json(content: str) -> bool:
    """Cheap heuristic — first non-whitespace char is `{` or `[`. Used
    only to choose a file extension for human-friendliness; the on-disk
    content is the original string either way.
    """
    stripped = content.lstrip()
    return bool(stripped) and stripped[0] in "{["


def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def _generate_preview(content: str, max_chars: int) -> tuple[str, bool]:
    """Truncate at the last newline within ``max_chars`` (if the cut is
    >50% of the way through) so previews don't end mid-line.
    """
    if len(content) <= max_chars:
        return content, False
    head = content[:max_chars]
    nl = head.rfind("\n")
    cut = nl if nl > max_chars // 2 else max_chars
    return content[:cut], True


def maybe_spill(
    content: str,
    *,
    tool_name: str,
    tool_use_id: str,
    session_id: str,
    threshold: int = SPILL_THRESHOLD_CHARS,
) -> str:
    """Persist ``content`` to disk if it exceeds ``threshold`` chars.

    Returns the in-prompt replacement string (a ``<persisted-output>``
    block) on spill, or the original ``content`` unchanged when under
    threshold. On filesystem error, logs and returns the original
    content (no truncation) — losing forensic data is worse than
    sending too much in this rare case.
    """
    if not isinstance(content, str):
        return content   # type: ignore[return-value]
    if len(content) <= threshold:
        return content

    is_json = _looks_like_json(content)
    path = _spill_path(session_id, tool_use_id, is_json=is_json)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 'x' would error on duplicate; use 'w' so a re-dispatch (rare)
        # overwrites cleanly.
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning(
            "spill: failed to persist %s for %s/%s: %s",
            path, session_id, tool_use_id, e,
        )
        return content

    preview, has_more = _generate_preview(content, PREVIEW_CHARS)
    msg = (
        f"{PERSISTED_OUTPUT_TAG}\n"
        f"Output too large ({_format_size(len(content))}, {len(content):,} chars). "
        f"Full output saved to: {path}\n\n"
        f"Preview (first {_format_size(PREVIEW_CHARS)}):\n"
        f"{preview}"
    )
    if has_more:
        msg += "\n...\n"
    else:
        msg += "\n"
    msg += (
        "Read the full file with the Read tool if you need more than the preview. "
        "If you don't actually need it all, narrow your next query "
        "(smaller hops, higher min_confidence, --glob, --type, head_limit, etc.).\n"
        f"{PERSISTED_OUTPUT_CLOSING_TAG}"
    )
    return msg


def persist_for_compaction(
    content: str, *, tool_use_id: str, session_id: str,
) -> Path | None:
    """Write ``content`` to this session's spill dir and return the path.

    Unlike :func:`maybe_spill` this has no size threshold and builds no
    preview block — the caller is about to remove the content from the
    prompt entirely, so a 2 KB preview would defeat the point. It exists
    so microcompaction can be lossless: content leaves the prompt, never
    the machine.

    Returns ``None`` on any filesystem error. The caller must treat that
    as "do not clear" — clearing content that failed to persist is the
    one outcome worth avoiding.
    """
    if not isinstance(content, str) or not content:
        return None
    if not session_id or not tool_use_id:
        return None
    path = _spill_path(session_id, tool_use_id, is_json=_looks_like_json(content))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "spill: failed to persist %s for %s/%s: %s",
            path, session_id, tool_use_id, e,
        )
        return None
    return path


def fallback_for_empty_result(content: str | None, tool_name: str) -> str:
    """Replace empty/whitespace-only tool results with an explicit
    "no output" marker. Some local models (notably qwen3-derived) treat
    an empty tool result as an end-of-turn signal and stop generating.
    """
    if content is None or not str(content).strip():
        short = tool_name.rsplit("__", 1)[-1] if "__" in tool_name else tool_name
        return f"({short} completed with no output)"
    return content


__all__ = [
    "SPILL_THRESHOLD_CHARS",
    "PREVIEW_CHARS",
    "PERSISTED_OUTPUT_TAG",
    "PERSISTED_OUTPUT_CLOSING_TAG",
    "maybe_spill",
    "persist_for_compaction",
    "fallback_for_empty_result",
]
