"""Conversation compaction — client-side context management.

The harness is stateless per request, so we reconstruct the conversation
from the persisted session JSON each turn, fit it to the model's
context window, and return ready-to-send OpenAI-format messages.

Three defensive layers, in order of cost (cheapest first):

  1. **Microcompaction** (:mod:`app.harness.microcompact`) — clears
     stale compactable-tool results inline. Pure structural pass, no
     LLM call. Runs every load.
  2. **LLM summarization** (:mod:`app.compaction_llm`) — when
     microcompact alone doesn't get under threshold, summarize the
     dropped block via a non-streaming POST to the local vLLM. Falls
     back to truncation on any failure so the user's turn always
     completes.
  3. **Truncation** — last resort. Drops oldest turns past
     ``TURNS_TO_KEEP``, prepends a ``[compaction: N tokens omitted]``
     marker. This is what the module did historically and remains the
     fallback when ``compaction.mode`` is ``truncate`` or when
     summarization fails.

Behind everything else, the harness loop has a reactive 413 recovery
path (``app.harness.loop._truncate_largest_tool_results``) that fires
if vLLM still rejects the prompt. That's the safety net under all of
the above.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

# Rough English-text heuristic: 1 token ≈ 4 chars. Integer arithmetic only.
TOKENS_PER_CHAR = 4

# Output headroom — reserve this many tokens for the model's response.
OUTPUT_TOKENS_RESERVED = 20_000

# Additional safety buffer so we start truncating before hitting the wall.
TRUNCATION_BUFFER_TOKENS = 32_000

# Default context window if a model has no configured `context_length`.
DEFAULT_CONTEXT_WINDOW = 128_000

# Number of most-recent turns (user+assistant pairs) to keep when
# truncation fires.
TURNS_TO_KEEP = 20


def estimate_tokens(text: str) -> int:
    """Rough token count for a string. Integer result."""
    if not text:
        return 0
    return max(1, len(text) // TOKENS_PER_CHAR)


def _message_text(message: dict) -> str:
    """Flatten a message's content to a single string for token counting.

    Handles both string content and structured content (list of blocks).
    Tool-use / tool-result / thinking blocks are serialized in full — no
    silent truncation, since they're often the informationally dense
    parts of a turn.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type", "")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "")
            inp = json.dumps(block.get("input", {}))
            parts.append(f"[tool_call: {name}({inp})]")
        elif btype == "tool_result":
            result = block.get("content", "")
            if isinstance(result, str):
                parts.append(f"[tool_result: {result}]")
            elif isinstance(result, list):
                for b in result:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(f"[tool_result: {b.get('text', '')}]")
                    elif isinstance(b, dict):
                        parts.append(f"[tool_result: {json.dumps(b)}]")
        elif btype == "thinking":
            parts.append(f"[thinking: {block.get('thinking', '')}]")
        elif btype == "image":
            parts.append("[image]")
        else:
            parts.append(json.dumps(block))
    return "\n".join(p for p in parts if p)


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for a single message dict."""
    if not message:
        return 0
    return estimate_tokens(_message_text(message))


def estimate_conversation_tokens(messages: list[dict], system_prompt: str = "") -> int:
    """Estimate total tokens for a conversation.

    Counts the literal bytes of `system_prompt` + each message. Does NOT
    add any opaque padding constants — the caller is responsible for
    passing the real system prompt if it wants that accounted for.
    """
    total = estimate_tokens(system_prompt)
    total += sum(estimate_message_tokens(m) for m in messages)
    return total


# ---------------------------------------------------------------------------
# Per-model context window
# ---------------------------------------------------------------------------


def get_context_window(model: str) -> int:
    """Return the configured context length for a model, or the default.

    Reads `models.<name>.context_length` from config.yaml. Imports lazily
    so this module stays importable without a full app bootstrap (tests
    can monkey-patch `_get_model_cfg` if needed).
    """
    try:
        from app.config import _get_model_cfg  # lazy — avoids circular deps at import
    except Exception:
        return DEFAULT_CONTEXT_WINDOW
    cfg = _get_model_cfg(model) if model else {}
    ctx = cfg.get("context_length")
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    return DEFAULT_CONTEXT_WINDOW


def truncation_threshold(context_window: int) -> int:
    """Token budget above which truncation should fire."""
    return max(1_000, context_window - OUTPUT_TOKENS_RESERVED - TRUNCATION_BUFFER_TOKENS)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def _group_into_turns(messages: list[dict]) -> list[list[dict]]:
    """Group a message list into turns.

    A turn = one user message optionally followed by assistant/tool
    messages, up to the next user message. The very first segment
    (messages before any user msg) is treated as its own "turn" so the
    system-initial content isn't lost.
    """
    turns: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "user" and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)
    return turns


def _flatten_turns(turns: list[list[dict]]) -> list[dict]:
    """Flatten a turn-grouped message list back into a flat list."""
    out: list[dict] = []
    for t in turns:
        out.extend(t)
    return out


def _split_for_summary(
    messages: list[dict],
    keep_recent_turns: int,
) -> tuple[list[dict], list[dict]]:
    """Partition ``messages`` into (older, recent) along turn boundaries.

    The ``recent`` block — last ``keep_recent_turns`` turns — is kept
    verbatim past the summary boundary. The ``older`` block is what
    gets summarized.
    """
    turns = _group_into_turns(messages)
    if len(turns) <= keep_recent_turns:
        return [], _flatten_turns(turns)
    cutoff = len(turns) - keep_recent_turns
    return _flatten_turns(turns[:cutoff]), _flatten_turns(turns[cutoff:])


def truncate_conversation(
    messages: list[dict],
    max_tokens: int,
    turns_to_keep: int = TURNS_TO_KEEP,
    system_prompt: str = "",
) -> tuple[list[dict], int]:
    """Truncate conversation to fit within a token budget.

    Returns (truncated_messages, tokens_dropped). If no truncation is
    needed, returns (messages, 0).

    Strategy: keep the last `turns_to_keep` turns. If that's still too
    big, drop turns from the front until we fit. Always keeps at least
    the final turn. When anything is dropped, prepends one synthetic
    user message describing what was removed, so the model isn't
    confused by mid-conversation jumps.
    """
    current_tokens = estimate_conversation_tokens(messages, system_prompt)
    if current_tokens <= max_tokens:
        return messages, 0

    turns = _group_into_turns(messages)
    if not turns:
        return messages, 0

    # Start by keeping the last N turns.
    kept = turns[-turns_to_keep:] if len(turns) > turns_to_keep else turns[:]

    # Drop oldest turns until under budget, but never drop the last turn.
    while len(kept) > 1 and estimate_conversation_tokens(_flatten_turns(kept), system_prompt) > max_tokens:
        kept.pop(0)

    truncated = _flatten_turns(kept)
    dropped_tokens = current_tokens - estimate_conversation_tokens(truncated, system_prompt)

    if dropped_tokens > 0:
        note = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"[compaction: {dropped_tokens} tokens of earlier conversation "
                        f"omitted to fit the context window. {len(turns) - len(kept)} "
                        f"turns were dropped.]"
                    ),
                }
            ],
        }
        truncated = [note] + truncated

    return truncated, dropped_tokens


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _compaction_cfg() -> dict[str, Any]:
    """Read the ``compaction:`` block from config.yaml with defaults.

    Called per invocation so config edits take effect on the next
    turn (no restart needed for tuning).
    """
    try:
        from app.config import CONFIG  # type: ignore
    except Exception:
        return {}
    cfg = dict(CONFIG.get("compaction") or {})
    cfg.setdefault("mode", "summarize")          # summarize | truncate
    cfg.setdefault("summary_model", None)        # None → falls back to default model
    cfg.setdefault("keep_recent_turns", 5)
    micro = dict(cfg.get("microcompact") or {})
    micro.setdefault("enabled", True)
    micro.setdefault("keep_recent_tools", 5)
    micro.setdefault("count_threshold", 20)
    micro.setdefault("compactable_tools", None)  # None → use module default
    cfg["microcompact"] = micro
    restore = dict(cfg.get("restore") or {})
    restore.setdefault("enabled", True)
    restore.setdefault("budget_tokens", 50_000)
    restore.setdefault("max_per_file", 5_000)
    restore.setdefault("max_files", 5)
    cfg["restore"] = restore
    manual = dict(cfg.get("manual") or {})
    manual.setdefault("buffer_tokens", 3_000)
    cfg["manual"] = manual
    return cfg


# ---------------------------------------------------------------------------
# Session-level helper — read JSON, truncate, return ready-to-send history
# ---------------------------------------------------------------------------


async def load_and_compact_session(
    session_path: Path | str,
    model: str = "",
    system_prompt: str = "",
    *,
    mode_override: str | None = None,
) -> dict[str, Any]:
    """Read a persisted session, apply the compaction stack, and return
    ready-to-send history plus metadata.

    Layer order (cheapest first):

      1. Load + filter to conversation roles.
      2. Microcompact pre-pass (clears stale tool results).
      3. If still over threshold and ``mode == "summarize"``:
         try LLM summarization; on success, replace dropped block with
         the summary and re-inject recent files (Layer C).
      4. If still over threshold (or summary failed, or mode is
         ``"truncate"``): fall through to the historical drop-oldest
         truncation path.

    Returns a dict with:
      - history:        list[dict], possibly compacted
      - tokens_before:  int, estimated tokens in the original history
      - tokens_after:   int, estimated tokens after compaction
      - truncated:      bool, True if any turns were dropped (compat-name)
      - summarized:     bool, True if LLM summarization replaced a block
      - microcompacted: int, count of tool results cleared by the pre-pass
      - restored_files: int, count of files re-injected post-summary
      - context_window: int, the model's configured window
      - threshold:      int, the compaction threshold in use

    On any error (missing file, malformed JSON), returns an empty result
    with ``history=[]`` and logs a warning. Summarization failures are
    handled transparently — the user's turn always completes.
    """
    cfg = _compaction_cfg()
    context_window = get_context_window(model)
    threshold = truncation_threshold(context_window)

    empty: dict[str, Any] = {
        "history": [],
        "tokens_before": 0,
        "tokens_after": 0,
        "truncated": False,
        "summarized": False,
        "microcompacted": 0,
        "restored_files": 0,
        "context_window": context_window,
        "threshold": threshold,
    }

    try:
        path = Path(session_path) if not isinstance(session_path, Path) else session_path
        if not path.exists():
            return empty
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning("compaction: failed to read session %s: %s", session_path, e)
        return empty

    messages = data.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return empty

    # Drop non-conversation entries (subliminal, tombstones, ambient markers)
    # since they're for UI display, not for the model.
    convo = [m for m in messages if m.get("role") in ("user", "assistant", "tool", "system")]

    tokens_before = estimate_conversation_tokens(convo, system_prompt)

    # ---- Layer B: microcompaction pre-pass ----------------------------
    micro_cleared = 0
    if cfg["microcompact"].get("enabled", True):
        # Lazy import — keeps app.compaction importable without the
        # harness module tree (e.g. for unit tests of truncation alone).
        try:
            from app.harness.microcompact import (
                DEFAULT_COMPACTABLE_TOOLS,
                microcompact,
            )
            mc_cfg = cfg["microcompact"]
            tools: Iterable[str] = (
                mc_cfg.get("compactable_tools") or DEFAULT_COMPACTABLE_TOOLS
            )
            convo, micro_cleared = microcompact(
                convo,
                keep_recent_tools=int(mc_cfg.get("keep_recent_tools", 5)),
                count_threshold=int(mc_cfg.get("count_threshold", 20)),
                compactable_tools=tools,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("microcompact pre-pass failed: %s", e)

    # If microcompact alone got us under threshold, we're done.
    cur_tokens = estimate_conversation_tokens(convo, system_prompt)
    summarized = False
    restored_count = 0

    # ---- Layer A: LLM summarization -----------------------------------
    mode = (mode_override or cfg.get("mode") or "summarize").lower()
    if cur_tokens > threshold and mode == "summarize":
        try:
            from app.compaction_llm import (
                restore_recent_files,
                summarize_history,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("compaction_llm import failed, falling back to truncate: %s", e)
            summarize_history = None  # type: ignore[assignment]
            restore_recent_files = None  # type: ignore[assignment]

        if summarize_history is not None:
            keep_recent_turns = int(cfg.get("keep_recent_turns", 5))
            older, recent = _split_for_summary(convo, keep_recent_turns)
            if older:
                summary = await summarize_history(
                    older,
                    model=cfg.get("summary_model") or model or None,
                )
                if summary:
                    summarized = True
                    summary_msg = {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[compaction summary — earlier conversation "
                                    "summarized to fit context window]\n\n"
                                    + summary
                                ),
                            }
                        ],
                    }
                    new_convo: list[dict] = [summary_msg]
                    # ---- Layer C: post-compact restore ----------------
                    if cfg["restore"].get("enabled", True) and restore_recent_files is not None:
                        try:
                            restored = restore_recent_files(
                                older,
                                budget_tokens=int(cfg["restore"].get("budget_tokens", 50_000)),
                                max_per_file=int(cfg["restore"].get("max_per_file", 5_000)),
                                max_files=int(cfg["restore"].get("max_files", 5)),
                            )
                            new_convo.extend(restored)
                            restored_count = len(restored)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("restore_recent_files failed: %s", e)
                    new_convo.extend(recent)
                    convo = new_convo
                    cur_tokens = estimate_conversation_tokens(convo, system_prompt)

    # ---- Layer (fallback): truncation ---------------------------------
    truncated_msgs, dropped = truncate_conversation(
        convo,
        max_tokens=threshold,
        turns_to_keep=TURNS_TO_KEEP,
        system_prompt=system_prompt,
    )
    tokens_after = estimate_conversation_tokens(truncated_msgs, system_prompt)

    return {
        "history": truncated_msgs,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "truncated": dropped > 0,
        "summarized": summarized,
        "microcompacted": micro_cleared,
        "restored_files": restored_count,
        "context_window": context_window,
        "threshold": threshold,
    }


__all__ = [
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_conversation_tokens",
    "get_context_window",
    "truncation_threshold",
    "truncate_conversation",
    "load_and_compact_session",
    "TOKENS_PER_CHAR",
    "OUTPUT_TOKENS_RESERVED",
    "TRUNCATION_BUFFER_TOKENS",
    "DEFAULT_CONTEXT_WINDOW",
    "TURNS_TO_KEEP",
]
