"""Tool-result disk spill — persist oversized tool results to file.

Modeled on Claude Code's `toolResultStorage.ts`. When a tool returns a
result larger than ``SPILL_THRESHOLD_CHARS``, the full content is written
to ``<SESSIONS_DIR>/<session_id>.tool-results/<tool_use_id>.{txt,json}``
and the in-prompt content is replaced with a ``<persisted-output>``
block containing:

  * total size + filepath, and the recovery the reading turn can actually
    take — the ``Read`` tool when it owns one, a narrower re-run when the
    turn's deny list has that tool (#1066)
  * a short preview (first ``PREVIEW_CHARS`` chars, cut at a newline)
  * a "...(more)" marker

Where a result that has been *cleared* is described — microcompaction's
marker and relief rung 4's notice, both built on this module's
``persist_for_compaction`` — the free route back is the session's own
record: ``sessions/<session_id>.json`` and ``sessions/<session_id>.tool-results/``
(:func:`session_record_route`, #1514). Those markers carry it when
``compaction.microcompact.name_session_record`` is on.

This solves two problems at once:

  1. **Context overflow.** A 250KB Grep result no longer crowds out the
     working set; the model sees ~2KB inline and can re-read on demand.
  2. **Information loss.** Inline truncation drops everything past the
     cut point. Spill keeps the full result on disk — a turn with Read or
     Grep can page into it for the bits it actually wants, and one that
     has neither still gets the preview and a path out of the file.

Empty-result guard: a tool that returns ``""`` / whitespace can cause
some local models to emit a stop token and end the turn with no output.
We replace empty results with ``"({tool_name} completed with no output)"``
so the model always has SOMETHING to react to.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
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


#: The one tool both spill notices assumed every turn owns. It does not:
#: ``workers/sources/deep_research.py`` denies ``Read`` on purpose, because a
#: research turn fetches arbitrary web pages and a page must not be able to
#: steer it into reading the local filesystem. That is also the source that
#: spills most, so its harness was the loudest one promising a call its own
#: policy refused (#1066).
READ_TOOL = "Read"

#: Two recoveries, chosen per turn, because the honest one depends on a fact
#: about the turn rather than about the file.
_RECOVERY_WITH_READ = (
    "Read the full file with the Read tool if you need more than the preview. "
    "If you don't actually need it all, narrow your next query "
    "(smaller hops, higher min_confidence, --glob, --type, head_limit, etc.).\n"
)
_RECOVERY_WITHOUT_READ = (
    "The Read tool is not available on this turn, so the path above cannot be "
    "opened from here — do not reach for it. Re-run the call with a narrower "
    "query (smaller hops, higher min_confidence, --glob, --type, head_limit, "
    "etc.); that is the only way to see more of it from here.\n"
)


def tool_is_denied(name: str, disallowed) -> bool:
    """Whether ``name`` is off the menu for the turn that is asking.

    Dispatch is stricter than this: `_pre_dispatch` refuses on an exact hit in
    ``options.disallowed_tools``. This asks the weaker question — is *any*
    spelling of the tool on the list — because the two directions of error are
    not equal here. A miss prints the false promise this function exists to
    withhold; a surplus sentence takes a real tool off the menu for one turn,
    and the turn still has the re-run this notice names. Deny lists carry both
    spellings anyway (bare names for builtins, `mcp__<server>__<tool>` for MCP
    ones), so on the paths that matter the two answers agree.
    """
    if not disallowed:
        return False
    deny = set(disallowed)
    if name in deny:
        return True
    return any(t.endswith(f"__{name}") for t in deny)


def recovery_notice(disallowed) -> str:
    """The sentence that closes a spilled-result notice for this turn.

    Lives apart from both renderers because two call sites need it — the
    ``<persisted-output>`` block and the context-pressure notice — and the
    pair that drifted is exactly how a model ends up being told to do
    something it cannot.
    """
    return _RECOVERY_WITHOUT_READ if tool_is_denied(READ_TOOL, disallowed) \
        else _RECOVERY_WITH_READ


def maybe_spill(
    content: str,
    *,
    tool_name: str,
    tool_use_id: str,
    session_id: str,
    threshold: int = SPILL_THRESHOLD_CHARS,
    disallowed_tools: Sequence[str] | None = None,
) -> str:
    """Persist ``content`` to disk if it exceeds ``threshold`` chars.

    Returns the in-prompt replacement string (a ``<persisted-output>``
    block) on spill, or the original ``content`` unchanged when under
    threshold. On filesystem error, logs and returns the original
    content (no truncation) — losing forensic data is worse than
    sending too much in this rare case.

    ``disallowed_tools`` is the turn's own deny list. The block tells the
    model how to get the rest of its result back, and until #1066 it said
    ``Read`` unconditionally while deep-research has that tool denied —
    ``Read`` and ``Bash`` denials on the ``<sid>.tool-results/`` path it was
    just pointed at, seven Bash and three Read of them in the 09-14→09-17
    window. Pass the turn's list and the block offers only what it can do.
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
    msg += recovery_notice(disallowed_tools) + PERSISTED_OUTPUT_CLOSING_TAG
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


#: The read-only tool that returns a cleared result by its observation id
#: (#1481, `agent_mcp/recall_observation.py`). Advertised only to a turn whose
#: relief writes observation stubs.
RECALL_OBSERVATION_TOOL = "recall_observation"

#: The tool that searches a directory tree for a phrase. Named here for the
#: same reason as ``READ_TOOL``: a route is only a route if the turn owns it.
GREP_TOOL = "Grep"


def session_record_paths(session_id: str) -> tuple[Path, Path]:
    """``(sessions/<session_id>.json, sessions/<session_id>.tool-results/)``.

    The session's own record, and the directory every spilled, cleared and
    truncated result of the session lands in (this module's ``_spill_dir``).
    """
    return SESSIONS_DIR / f"{session_id}.json", _spill_dir(session_id)


def session_record_route(session_id: str, disallowed=None) -> str:
    """The free route back to a cleared result (#1514): the session's own record.

    A cleared tool result has already left the prompt, but it has not left the
    machine. Two copies of it are on disk before any marker is written:

      * ``sessions/<session_id>.tool-results/`` — the file this module (or
        microcompaction's ``persist_for_compaction``, or relief rung 4) wrote
        it to, one file per call;
      * ``sessions/<session_id>.json`` — the transcript row written when the
        result landed (the whole result, or a ``<persisted-output>`` pointer
        into the directory above when it was over 2,000 chars, D1).

    A marker that names only the one per-call file says *where one result
    is*; it does not tell a model that has lost track of *which* result held a
    fact that the whole record is searchable. This sentence does, with no new
    tool and no new permission — it is the baseline #1481's
    ``recall_observation(id)`` has to beat. It names the transcript only when
    it exists (a ``task:*`` subagent writes none, only a spill directory), and
    it offers only a tool the reading turn owns: ``Grep`` over both, else
    ``Read`` of the transcript, else nothing — a route the turn's deny list
    refuses is the #1066 false promise again.

    Returned as one clause ending in a newline-free sentence, or ``""``.
    """
    if not session_id:
        return ""
    record, spill = session_record_paths(session_id)
    has_record = record.is_file()
    if not tool_is_denied(GREP_TOOL, disallowed):
        where = (f"{record} (the conversation) and {spill}/ (every spilled or "
                 f"cleared result)" if has_record else f"{spill}/")
        return (f"This session's own record is on disk: Grep {where} for a "
                f"phrase you no longer have in context.")
    if not tool_is_denied(READ_TOOL, disallowed) and has_record:
        return (f"This session's own record is on disk at {record}; Read it "
                f"for text you no longer have in context.")
    return ""


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
    "READ_TOOL",
    "GREP_TOOL",
    "recovery_notice",
    "session_record_paths",
    "session_record_route",
    "tool_is_denied",
    "persist_for_compaction",
    "fallback_for_empty_result",
]
