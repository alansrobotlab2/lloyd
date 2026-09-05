"""Microcompaction — clear stale tool results inline before they reach the wall.

Modeled on Claude Code's `microCompact.ts`. Cheaper than full LLM
summarization: this is a pure structural pass that replaces older
compactable-tool results with a short marker, while keeping the most
recent N inline. The model can re-read the persisted file (via the
spill mechanism in :mod:`app.harness.tool_result_spill`) if it still
needs the content.

**Budget-driven since 2026-09-05.** The pass clears the fewest results
needed to get under ``token_budget``, oldest first, and stops. Pass no
budget and it falls back to the legacy count-based rule.

Why this changed: the count rule ignored context pressure entirely.
Session ``20260905_024955_iv5f05`` — an architecture review — peaked at
106,802 tokens against a 210,144 threshold, 40% of the budget, and the
pre-pass still cleared **93 of 97** tool results, 84% of the
conversation's content, because ``len(compactable) > 20``. The turn then
hit ``max_turns`` with no output and the next turn started blind. Layer A
(LLM summarization) was correctly gated on ``cur_tokens > threshold``;
this layer, which is destructive and runs first, was gated on nothing.

The fixed-working-set policy came from Claude Code's ``microCompact.ts``,
which lives under a ~200k window with prompt-cache TTL pressure. Lloyd
has 262k, a 210k threshold, and no cache TTL to align with — the
mechanism was ported without the constraint that justified it.

Clearing is oldest-first by design. A relevance-ranked policy would
fragment the shared prefix across turns (costing vLLM prefix-cache hits)
and would need a model call to rank, defeating the "cheaper than
summarization" premise. Recency is a decent proxy, and clearing strictly
from the front keeps each turn's prompt a clean extension of the last.

Two clearing triggers, evaluated together:

  1. **Budget-based** (when ``token_budget`` is given). Clear oldest
     compactable results one at a time until the conversation fits.
     Never drops below ``keep_recent_tools``, and skips results under
     ``min_chars_to_clear`` — the marker names the tool and its
     arguments, so it costs ~40 tokens; putting it in place of a
     200-byte result makes the prompt larger AND loses the content.

     Legacy count rule (over ``count_threshold``, keep the most recent
     ``keep_recent_tools``, clear the rest regardless of pressure) now
     requires ``legacy_count_rule=True`` *and* no budget. Both automatic
     call sites disable it; only ``/compact`` still uses it, where the
     user has explicitly asked to shrink.

  2. **Spill-aware.** Any tool result already containing the
     ``<persisted-output>`` spill marker AND older than the last
     ``keep_recent_tools`` gets its inline *preview* dropped — but the
     marker keeps the file path, so the model can ``Read`` it on demand.

Nothing is cleared without being recoverable. When ``session_id`` is
supplied, content is written to the session's spill dir *before* it is
removed from the prompt, and the marker names the tool, its arguments and
the file. Previously the marker was a bare "[content cleared — retrieve
via Read if needed]" with no path, no tool name and no arguments, so the
instruction it gave could not be followed; and ``_replace_tool_content``
overwrote spilled results' ``<persisted-output>`` blocks too, destroying
the very path this module's docstring promised was preserved.

Three call sites, and the second is the one that matters most:

  * ``app.compaction.load_and_compact_session`` — turn start, rebuilding
    history from the session JSON.
  * ``app.harness.loop`` — **mid-turn**, after each iteration's tool
    calls land, mutating ``chat_messages`` in place so the Inner Voice
    observer's handle stays valid. This is where a long turn is actually
    shaped: on ``20260905_024955_iv5f05`` it held a 70-tool-call turn to
    5 inline results for its whole length, and it is easy to miss because
    it imports this function under an alias.
  * ``app.routers.messages`` — the ``/compact`` slash command.

There is no time-based trigger. Lloyd has no prompt cache to align
with, so the cache-TTL heuristic Claude Code uses doesn't apply here.

Compactable tool list defaults to ``Read, Bash, Grep, Glob, Edit,
Write`` plus any namespaced ``mcp__*`` variants of those names. The
caller can override via the ``compactable_tools`` argument.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable

from app.harness.tool_result_spill import (
    PERSISTED_OUTPUT_TAG,
    persist_for_compaction,
)

logger = logging.getLogger("lloyd-microcompact")


DEFAULT_COMPACTABLE_TOOLS: tuple[str, ...] = (
    "Read", "Bash", "Grep", "Glob", "Edit", "Write",
)

# Fallback marker, used only when the call that produced the result can't
# be identified and nothing was persisted. Every other path produces a
# marker naming the tool, its arguments and the spill file — see
# `_cleared_marker`. Kept as a module constant because callers and tests
# match on it.
CLEARED_MARKER = "[Old tool result content cleared — retrieve via Read if needed]"

# Results smaller than this are left alone. Clearing one saves fewer
# tokens than the marker replacing it costs, and loses the content.
DEFAULT_MIN_CHARS_TO_CLEAR = 2_000

# Floor on how many recent results stay inline. A code review's working
# set is 15-25 files; the old default of 5 was well under the point where
# the history stops being usable. Only binds when actually over budget.
DEFAULT_KEEP_RECENT_TOOLS = 15


def _args_digest(raw_args: Any, cap: int = 120) -> str:
    """Render tool arguments compactly for a cleared-result marker."""
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception:  # noqa: BLE001 — not JSON; use the string as-is
            text = " ".join(raw_args.split())
            return text if len(text) <= cap else text[: cap - 3] + "..."
    if isinstance(raw_args, dict):
        parts = [
            f"{k}={raw_args[k]!r}"
            for k in sorted(raw_args)
            if k != "description" and raw_args[k] not in (None, "")
        ]
        text = " ".join(parts)
    else:
        text = str(raw_args or "")
    text = " ".join(text.split())
    return text if len(text) <= cap else text[: cap - 3] + "..."


def _cleared_marker(
    tool_name: str, raw_args: Any, size: int, path: Any = None,
) -> str:
    """Build a marker that says what was cleared and where it went.

    The model's only route back to this content is what this string says,
    so it names the call and — when the content was persisted — the file.
    """
    if not tool_name:
        return CLEARED_MARKER
    digest = _args_digest(raw_args)
    head = f"{tool_name} {digest}".strip()
    if path is not None:
        return (
            f"[{head} — {size:,} chars cleared from context; "
            f"full content at {path}. Read that path if you need it again.]"
        )
    return (
        f"[{head} — {size:,} chars of output cleared from context. "
        f"Re-run the call if you need it again.]"
    )


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


def _persisted_block_only(text: str) -> str:
    """Reduce a spilled result to its header, dropping the inline preview.

    The ``<persisted-output>`` block carries the size and file path in its
    first lines and then up to ``PREVIEW_CHARS`` of content. Once the
    result is stale the preview is the waste; the path is the point.
    """
    start = text.find(PERSISTED_OUTPUT_TAG)
    if start == -1:
        return text
    head = text[start:]
    lines = head.splitlines()
    # Tag line + the "Full output saved to: <path>" line are what matter.
    keep = [ln for ln in lines[:3] if ln.strip()]
    return "\n".join(keep) + "\n[preview dropped — Read the path above for the full content]"


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
    keep_recent_tools: int = DEFAULT_KEEP_RECENT_TOOLS,
    count_threshold: int = 20,
    compactable_tools: Iterable[str] = DEFAULT_COMPACTABLE_TOOLS,
    token_budget: int | None = None,
    estimate_fn: Callable[[list[dict]], int] | None = None,
    min_chars_to_clear: int = DEFAULT_MIN_CHARS_TO_CLEAR,
    session_id: str = "",
    legacy_count_rule: bool = True,
) -> tuple[list[dict], int]:
    """Replace stale compactable tool results with a cleared marker.

    Returns ``(new_messages, cleared_count)``. ``new_messages`` is a
    fresh list — the input is not mutated. Tool messages whose paired
    tool_call wasn't compactable are passed through unchanged.

    Args:
      token_budget: target token count. Clear oldest-first only until
        the conversation fits, then stop. Requires ``estimate_fn``.
        ``None`` means "do no budget-driven clearing" — which, with
        ``legacy_count_rule=False``, means the pass only drops previews
        from results already on disk.
      legacy_count_rule: fall back to the pre-2026-09-05 rule (over
        ``count_threshold``, keep the most recent ``keep_recent_tools``,
        clear everything else regardless of context pressure) when no
        ``token_budget`` is given. True for back-compat with direct
        callers; ``app.compaction`` passes False. Do not enable it
        alongside a budget — they answer the same question differently.
      estimate_fn: ``messages -> tokens``. Injected rather than imported
        so this module keeps no dependency on ``app.compaction``.
      min_chars_to_clear: leave results smaller than this alone.
      session_id: enables spill-before-clear. Without it, content that
        is not already on disk is cleared irrecoverably, so callers that
        have a session id should always pass it.

    Recoverability: with ``session_id`` set, each result is written to
    the session's spill dir before its content leaves the prompt, and the
    marker carries the path. A result that fails to persist is left
    inline — clearing content that could not be saved is the one outcome
    worth refusing, even at the cost of staying over budget.
    """
    if not messages:
        return list(messages), 0

    allow: set[str] = {t for t in compactable_tools}

    # Pass 1: build tool_call_id → tool_name map. Assistant messages
    # carry tool_calls; we trust that mapping over any name on the tool
    # message itself (which Lloyd doesn't always populate consistently).
    tc_id_to_name: dict[str, str] = {}
    tc_id_to_args: dict[str, Any] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            cid = tc.get("id") or tc.get("call_id")
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            if cid:
                tc_id_to_name[cid] = name
                # Kept so a cleared result can name the call that made
                # it. "[content cleared — retrieve via Read if needed]"
                # is an instruction the model cannot act on without this.
                tc_id_to_args[cid] = fn.get("arguments", tc.get("arguments"))

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

    # Everything older than the most recent `keep_recent_tools` is a
    # candidate; the recent window is never touched by either rule.
    keep_count = max(0, keep_recent_tools)
    cutoff = max(0, len(compactable_indices) - keep_count)
    candidates = compactable_indices[:cutoff]

    # Too small to be worth clearing, under either rule. The marker names
    # the tool and its arguments and so costs ~40 tokens; replacing a
    # 200-byte result with it makes the prompt LARGER while losing the
    # content. The spill-aware pass below is exempt — a spilled result is
    # over the 50 KB spill threshold by definition.
    sizeable = [
        idx for idx in candidates
        if len(_tool_result_text(messages[idx])) >= min_chars_to_clear
    ]

    to_clear: set[int] = set()

    if token_budget is not None and estimate_fn is not None:
        # Budget-based: clear the fewest candidates that get us under
        # budget, oldest first, and stop. Bail out immediately when the
        # conversation already fits — the common case, and the one the
        # old count rule got wrong.
        #
        # Token accounting is additive per message, so each clear's saving
        # is measured on that message alone rather than by re-estimating
        # the whole conversation. Re-estimating was O(candidates x size):
        # ~100 full scans of an 800 KB history on a turn near the wall,
        # which is exactly when the harness can least afford the stall.
        current = estimate_fn(list(messages))
        if current > token_budget:
            for idx in sizeable:
                marker = _replace_tool_content(messages[idx], CLEARED_MARKER)
                saved = estimate_fn([messages[idx]]) - estimate_fn([marker])
                if saved <= 0:
                    continue
                to_clear.add(idx)
                current -= saved
                if current <= token_budget:
                    break
    elif legacy_count_rule and len(compactable_indices) > count_threshold:
        # Legacy count rule. Ignores context pressure entirely, which is
        # what cleared 93 of 97 results on a conversation using 17% of
        # its window. Retained for `/compact`, where the user has
        # explicitly asked to shrink, and for direct callers that pass no
        # budget; the turn-start and mid-turn paths both disable it.
        to_clear.update(sizeable)

    # Spill-aware: a candidate that already carries a persisted-output
    # marker has its content on disk, so the inline preview is pure
    # waste. Safe to drop regardless of budget — nothing is lost.
    for idx in candidates:
        if PERSISTED_OUTPUT_TAG in _tool_result_text(messages[idx]):
            to_clear.add(idx)

    if not to_clear:
        return list(messages), 0

    # Pass 3: build the output list, persisting each result before its
    # content leaves the prompt.
    out: list[dict] = []
    cleared = 0
    persisted = 0
    for i, msg in enumerate(messages):
        if i not in to_clear:
            out.append(msg)
            continue

        text = _tool_result_text(msg)

        # Already spilled: the block names the file, so keep it verbatim
        # rather than overwriting it with a path-free marker. This is the
        # case the old code claimed to handle and did not — it replaced
        # the <persisted-output> block, destroying the only route back to
        # the content.
        if PERSISTED_OUTPUT_TAG in text:
            out.append(_replace_tool_content(msg, _persisted_block_only(text)))
            cleared += 1
            continue

        cid = msg.get("tool_call_id") or msg.get("call_id") or ""
        tool_name = tc_id_to_name.get(cid, "")
        path = None
        if session_id and cid:
            path = persist_for_compaction(
                text, tool_use_id=cid, session_id=session_id,
            )
            if path is not None:
                persisted += 1
            else:
                # Refuse to clear what we could not save. Staying over
                # budget is recoverable; silently deleting the primary's
                # evidence is not. Layer A summarization still runs after
                # this and will handle the overflow.
                out.append(msg)
                continue

        out.append(_replace_tool_content(
            msg, _cleared_marker(tool_name, tc_id_to_args.get(cid), len(text), path),
        ))
        cleared += 1

    if cleared:
        logger.info(
            "microcompact: cleared %d/%d compactable tool results "
            "(persisted=%d, keep_recent=%d, budget=%s)",
            cleared, len(compactable_indices), persisted,
            keep_recent_tools,
            token_budget if token_budget is not None else "count-rule",
        )

    return out, cleared


__all__ = [
    "DEFAULT_COMPACTABLE_TOOLS",
    "CLEARED_MARKER",
    "microcompact",
]
