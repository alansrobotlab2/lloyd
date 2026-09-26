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
     requires ``legacy_count_rule=True`` *and* no budget. Every production
     call site disables it; ``/compact`` used it until D11 (2026-09-24),
     when it became a queued fold into the persisted summary record.

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

Two call sites, and the second is the one that matters most:

  * ``app.compaction.load_and_compact_session`` — turn start, rebuilding
    history from the session JSON.
  * ``app.harness.loop`` — **mid-turn**, after each iteration's tool
    calls land, mutating ``chat_messages`` in place so the Inner Voice
    observer's handle stays valid. This is where a long turn is actually
    shaped: on ``20260905_024955_iv5f05`` it held a 70-tool-call turn to
    5 inline results for its whole length, and it is easy to miss because
    it imports this function under an alias.

(``/compact`` was a third, with the count rule, until D11.)

There is no time-based trigger. Lloyd has no prompt cache to align
with, so the cache-TTL heuristic Claude Code uses doesn't apply here.

Compactable tool list defaults to ``Read, Bash, Grep, Glob, Edit,
Write`` plus any namespaced ``mcp__*`` variants of those names. The
caller can override via the ``compactable_tools`` argument.
Or, with ``non_compactable_tools``, every tool except a deny list
(``DEFAULT_NON_COMPACTABLE``) — config ``compaction.microcompact.
non_compactable_tools``, which wins over ``compactable_tools`` when set.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable

from app.harness.tool_result_spill import (
    PERSISTED_OUTPUT_TAG,
    READ_TOOL,
    _generate_preview,
    persist_for_compaction,
    session_record_paths,
    session_record_route,
    tool_is_denied,
)

logger = logging.getLogger("lloyd-microcompact")


DEFAULT_COMPACTABLE_TOOLS: tuple[str, ...] = (
    "Read", "Bash", "Grep", "Glob", "Edit", "Write",
)

# Deny-list mode (D10, 2026-09-24): with ``non_compactable_tools`` given,
# every tool result may be cleared EXCEPT these. The allow-list above never
# covered an MCP domain tool (vault search, http_fetch, graph queries...), so
# those results were never cleared and went straight to truncation. What is
# protected here is state the model steers by and cannot cheaply re-derive:
# its todo list, the tool catalogue ToolSearch loaded, plan mode, the goal.
# Deliberately NOT listed: ``Task`` — the newest result is protected by
# ``keep_recent_tools`` like any other, and an old subagent report is exactly
# the bulk this exists to clear. Skill-delivery results carry the name of the
# tool call the skill intercepted and are short (under ``min_chars_to_clear``),
# so they are left alone by size rather than by name.
DEFAULT_NON_COMPACTABLE: tuple[str, ...] = (
    "TodoWrite", "ToolSearch", "EnterPlanMode", "ExitPlanMode",
    "SetGoal", "ClearGoal",
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
    read_denied: bool = False, route: str = "",
) -> str:
    """Build a marker that says what was cleared and where it went.

    The model's only route back to this content is what this string says,
    so it names the call and — when the content was persisted — the file.
    Naming the file is only half a route: the turn has to be able to open
    it. ``read_denied`` comes from the caller's deny list, and a turn with
    ``Read`` on it gets the re-run it can perform instead of a path it
    cannot open (#1066).
    """
    if not tool_name:
        return CLEARED_MARKER
    digest = _args_digest(raw_args)
    head = f"{tool_name} {digest}".strip()
    # #1514: the session's own record, only when the caller asked for it
    # (`name_session_record`) and only where the content was saved.
    tail = f" {route}" if route and path is not None else ""
    if path is not None and read_denied:
        return (
            f"[{head} — {size:,} chars cleared from context; full content at "
            f"{path}. The Read tool is not available on this turn, so that "
            f"path cannot be opened from here — re-run the call with a "
            f"narrower query for the part you need.{tail}]"
        )
    if path is not None:
        return (
            f"[{head} — {size:,} chars cleared from context; "
            f"full content at {path}. Read that path if you need it again.{tail}]"
        )
    return (
        f"[{head} — {size:,} chars of output cleared from context. "
        f"Re-run the call if you need it again.]"
    )


def _is_compactable_tool_call(
    name: str, allow: set[str], deny: set[str] | None = None,
) -> bool:
    """Match either bare ``Read`` or namespaced ``mcp__lloyd-mcp__Read``.

    ``deny`` given (even empty) is deny-list mode: any named tool is
    compactable unless its bare name is in ``deny``, and ``allow`` is not
    consulted. An unnamed call is never compactable in either mode — the
    marker could not say what it cleared.
    """
    if not name:
        return False
    bare = name.rsplit("__", 1)[-1] if "__" in name else name
    if deny is not None:
        return name not in deny and bare not in deny
    if name in allow:
        return True
    # Namespaced form: mcp__<server>__<tool>
    return bare in allow


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


#: The line a reduced ``<persisted-output>`` block ends with. A block that
#: already carries it is a stub, and reducing it again is what made this pass
#: non-idempotent (#1481: 175 -> 236 -> 236 chars, a duplicated notice line,
#: and ``cleared += 1`` on every relief pass for a result cleared long ago —
#: each rewrite a message the engine had already cached).
PREVIEW_DROPPED = "[preview dropped — Read the path above for the full content]"

#: How every observation stub (#1481) begins. Selection skips a result whose
#: text starts with it, so a stub is never cleared twice.
OBSERVATION_STUB_PREFIX = "[observation "

#: Default bound on a stub's verbatim head, in chars (~100 tokens). The stub
#: stays in the prompt for the rest of the turn and every later one, so this
#: is bytes paid per clear (`compaction.microcompact.observation_head_chars`).
DEFAULT_OBSERVATION_HEAD_CHARS = 400

_SAVED_TO = "Full output saved to: "


def _is_cleared_stub(text: str) -> bool:
    """Whether a tool result is already what a clear leaves behind.

    Two shapes: a ``<persisted-output>`` block reduced to its header, and an
    observation stub. Neither is selected again — re-reducing one rewrites a
    cached message for nothing and books a clear that freed nothing (#1481).
    A plain cleared marker needs no rule: it is far under
    ``min_chars_to_clear`` and carries no tag, so no selection reaches it.
    """
    if text.startswith(OBSERVATION_STUB_PREFIX):
        return True
    return PERSISTED_OUTPUT_TAG in text and PREVIEW_DROPPED in text


def _persisted_block_only(text: str) -> str:
    """Reduce a spilled result to its header, dropping the inline preview.

    The ``<persisted-output>`` block carries the size and file path in its
    first lines and then up to ``PREVIEW_CHARS`` of content. Once the
    result is stale the preview is the waste; the path is the point.

    Idempotent: a block that is already reduced comes back unchanged.
    """
    start = text.find(PERSISTED_OUTPUT_TAG)
    if start == -1 or _is_cleared_stub(text):
        return text
    head = text[start:]
    lines = head.splitlines()
    # Tag line + the "Full output saved to: <path>" line are what matter.
    keep = [ln for ln in lines[:3] if ln.strip()]
    return "\n".join(keep) + "\n" + PREVIEW_DROPPED


def _pointer_path(text: str) -> str:
    """The path a ``<persisted-output>`` block names, or ``""``."""
    at = text.find(_SAVED_TO)
    if at == -1:
        return ""
    return text[at + len(_SAVED_TO):].split("\n", 1)[0].strip()


def _pointer_size(text: str, pointer: str) -> int:
    """The original size a pointer block states (``…, 12,345 chars)``), else
    the file's size, else 0."""
    import re

    m = re.search(r"([0-9][0-9,]*) chars\)", text)
    if m:
        return int(m.group(1).replace(",", ""))
    try:
        from pathlib import Path

        return Path(pointer).stat().st_size
    except OSError:
        return 0


def _pointer_preview(text: str) -> str:
    """The verbatim preview a ``<persisted-output>`` block carries, or ``""``.

    ``maybe_spill`` writes ``Preview (first N):\\n<preview>`` followed by
    ``\\n...\\n`` or ``\\n`` and the recovery sentence. The preview is a prefix
    of the spilled file, so a head cut from it is a head of the original.
    """
    at = text.find("Preview (first ")
    nl = text.find("\n", at) if at != -1 else -1
    if nl == -1:
        return ""
    body = text[nl + 1:]
    for end in ("\n...\n", "\nRead the full file", "\nThe Read tool is not"):
        cut = body.find(end)
        if cut != -1:
            return body[:cut]
    return body


def observation_id_for_path(path: Any, session_id: str) -> str:
    """The observation id of a file in ``session_id``'s spill directory.

    An id is the file's stem — the (sanitised) tool-call id it was written
    under — and only a file directly inside that session's own
    ``sessions/<session_id>.tool-results/`` has one; ``""`` otherwise.
    `agent_mcp/recall_observation.py` resolves an id back the same way.
    """
    if not path or not session_id:
        return ""
    from pathlib import Path

    try:
        p = Path(str(path))
        if p.parent.resolve() != session_record_paths(session_id)[1].resolve():
            return ""
    except (OSError, ValueError):
        return ""
    return p.stem


def _observation_stub(
    obs_id: str, tool_name: str, raw_args: Any, size: int, head_src: str,
    head_chars: int, route: str = "",
) -> str:
    """What a cleared result leaves behind with observation stubs on (#1481).

    An addressable id, the call that made it, and a bounded VERBATIM head of
    the original — never a paraphrase (CliffCompaction: truncate only). The
    plain marker says only *where* the content went, so the model has to
    re-read a result to learn whether it needs it; the head says *what* it
    was, and ``recall_observation(id=…)`` returns the rest.
    """
    head, _more = _generate_preview(head_src or "", max(0, int(head_chars)))
    call = f"{tool_name} {_args_digest(raw_args)}".strip() or "tool result"
    lines = [
        f"{OBSERVATION_STUB_PREFIX}{obs_id} — {call} — {size:,} chars cleared "
        f"from context. Its first {len(head):,} chars, verbatim:",
        head,
        f"… recall_observation(id=\"{obs_id}\") returns the full text.",
    ]
    if route:
        lines.append(route)
    return "\n".join(lines) + "]"


def _replace_tool_content(message: dict, new_text: str) -> dict:
    """Return a shallow copy of ``message`` with its ``content`` replaced
    by a single text-block payload. Preserves all other fields (role,
    tool_call_id, stats, etc.).
    """
    out = dict(message)
    # A cleared result keeps no screenshot either: the image is on disk
    # beside the text spill, and its KV is the point of clearing.
    out.pop("_image_refs", None)
    if isinstance(message.get("images"), list):
        # A session row: keep the refs for the UI, but never re-send them.
        out["images"] = [dict(r, evicted=True) if isinstance(r, dict) else r
                         for r in message["images"]]
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
    disallowed_tools: Iterable[str] | None = None,
    non_compactable_tools: Iterable[str] | None = None,
    observation_stubs: bool = False,
    observation_head_chars: int = DEFAULT_OBSERVATION_HEAD_CHARS,
    name_session_record: bool = False,
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
      disallowed_tools: the reading turn's deny list, so a marker can name
        a recovery that turn can actually take. ``app.compaction`` has no
        turn to answer for and leaves it unset, which reads as
        everything-allowed — the conservative direction, since the
        alternative is withholding a Read from a chat turn that has one.
      non_compactable_tools: deny-list mode. When given (even empty),
        ``compactable_tools`` is ignored and every tool's result may be
        cleared except these (bare or namespaced). ``None`` keeps the
        allow-list. See ``DEFAULT_NON_COMPACTABLE``.
      observation_stubs: #1481, off by default. A cleared result — fresh, or
        a spilled pointer block — becomes an observation stub (an id, the
        call, a verbatim head of at most ``observation_head_chars``) that
        ``recall_observation(id)`` resolves. Needs ``session_id``; without
        one, or when the id cannot be derived, the old marker is written.
      name_session_record: #1514, off by default. A marker for content that
        was saved also names the session's own record
        (``tool_result_spill.session_record_route``).

    Whatever the switches, a result a previous pass already cleared is never
    selected again (``_is_cleared_stub``): a second pass over its own output
    changes no byte and counts no clear.

    Recoverability: with ``session_id`` set, each result is written to
    the session's spill dir before its content leaves the prompt, and the
    marker carries the path. A result that fails to persist is left
    inline — clearing content that could not be saved is the one outcome
    worth refusing, even at the cost of staying over budget.
    """
    if not messages:
        return list(messages), 0

    allow: set[str] = {t for t in compactable_tools}
    deny: set[str] | None = (
        None if non_compactable_tools is None
        else {str(t) for t in non_compactable_tools}
    )
    # One lookup for the whole pass: every marker this pass writes answers
    # the same question about the same tool menu.
    read_denied = tool_is_denied(READ_TOOL, disallowed_tools)
    route = (session_record_route(session_id, disallowed_tools)
             if name_session_record and session_id else "")
    stubs = bool(observation_stubs and session_id)

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
        if _is_compactable_tool_call(name, allow, deny):
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
    # A result a previous pass already cleared is a candidate for neither
    # rule: re-clearing a stub frees nothing and rewrites a message the
    # engine has cached (#1481).
    candidates = [
        idx for idx in candidates
        if not _is_cleared_stub(_tool_result_text(messages[idx]))
    ]
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
            # A stub keeps its head, so it frees less than a bare marker;
            # budget with a stand-in of the stub's size or the pass stops
            # short of the target it believes it reached.
            stand_in = (CLEARED_MARKER + " " * (max(0, int(observation_head_chars)) + 200)
                        if stubs else CLEARED_MARKER)
            for idx in sizeable:
                marker = _replace_tool_content(messages[idx], stand_in)
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
        # its window. Retained for direct callers that pass no budget; no
        # production path uses it since `/compact` stopped (D11).
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
        cid = msg.get("tool_call_id") or msg.get("call_id") or ""
        tool_name = tc_id_to_name.get(cid, "")
        if PERSISTED_OUTPUT_TAG in text:
            new_text = ""
            if stubs:
                pointer = _pointer_path(text)
                obs_id = observation_id_for_path(pointer, session_id)
                if obs_id:
                    new_text = _observation_stub(
                        obs_id, tool_name, tc_id_to_args.get(cid),
                        _pointer_size(text, pointer), _pointer_preview(text),
                        observation_head_chars, route)
            if not new_text:
                new_text = _persisted_block_only(text)
                if route:
                    new_text = f"{new_text} {route}"
            if new_text == text:
                out.append(msg)
                continue
            out.append(_replace_tool_content(msg, new_text))
            cleared += 1
            continue

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

        obs_id = observation_id_for_path(path, session_id) if stubs else ""
        if obs_id:
            new_text = _observation_stub(
                obs_id, tool_name, tc_id_to_args.get(cid), len(text), text,
                observation_head_chars, route)
        else:
            new_text = _cleared_marker(
                tool_name, tc_id_to_args.get(cid), len(text), path,
                read_denied=read_denied, route=route)
        out.append(_replace_tool_content(msg, new_text))
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


SHRUNK_ARG_MARKER = "[argument cleared from context]"

# Which argument of which tool carries a body big enough to be worth
# spilling. `agent_mcp/builtin_fs.py` is the authority on the names.
SHRINKABLE_ARGS: dict[str, tuple[str, ...]] = {
    "Write": ("content",),
    "Edit": ("old_string", "new_string"),
}


def _shrinkable_fields(tool_name: str, allow: set[str]) -> tuple[str, ...]:
    """Field names to spill for a tool, bare or namespaced."""
    if not tool_name:
        return ()
    bare = tool_name.rsplit("__", 1)[-1]
    if bare not in allow:
        return ()
    return SHRINKABLE_ARGS.get(bare, ())


def shrink_assistant_arguments(
    messages: list[dict],
    *,
    keep_recent_tools: int = 5,
    min_chars: int = 2_000,
    session_id: str = "",
    tools: Iterable[str] = ("Write", "Edit"),
) -> tuple[list[dict], int, int]:
    """Spill big `Write`/`Edit` bodies out of *assistant* tool_call arguments.

    The third relief rung, and the only one that can reach this residue.
    Microcompaction clears tool *results*; the file body the model wrote
    lives in the assistant message's `tool_calls[].function.arguments` and
    rides in the prompt for the rest of the turn. Measured on the rounds
    that died at the wall on 2026-09-11: a single 16k-char `Write` costs
    ~4k tokens every iteration after it, and a round that writes eight
    files has spent 32k tokens on text already on disk.

    Two rules keep it safe:

    - **Only once the result has landed.** An argument spilled before its
      tool ran would change what the model is shown it asked for while the
      call is still in flight. `keep_recent_tools` additionally holds the
      most recent N calls intact, because the model is usually still
      working with those.
    - **Refuse when the spill fails**, exactly as `microcompact` does at
      the equivalent point: staying over budget is recoverable, destroying
      the only copy of what was written is not. A `Write` body is not
      re-derivable from the marker.

    The rewritten arguments must stay valid JSON — `loop._commit_tool_calls`
    replaces unparseable arguments with `{}` on the way in precisely because
    vLLM re-parses this field as history and 400s on malformed input.

    Returns `(messages, shrunk_count, freed_chars)`. `messages` is a new
    list; callers holding a shared handle must slice-assign.
    """
    allow = {str(t) for t in tools if t}
    if not allow or not messages:
        return messages, 0, 0

    # Which tool_call ids already have a result in this list. An id with no
    # result is still in flight.
    landed: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id") or msg.get("call_id") or ""
            if cid:
                landed.add(str(cid))

    # Hold the most recent `keep_recent_tools` landed calls intact, in the
    # order they appear.
    ordered: list[str] = []
    for msg in messages:
        for tc in (msg.get("tool_calls") or []) if msg.get("role") == "assistant" else []:
            cid = str(tc.get("id") or "")
            if cid and cid in landed:
                ordered.append(cid)
    recent = set(ordered[-keep_recent_tools:]) if keep_recent_tools > 0 else set()

    out: list[dict] = []
    shrunk = 0
    freed = 0
    for msg in messages:
        tool_calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None
        if not tool_calls:
            out.append(msg)
            continue

        new_calls: list[dict] = []
        touched = False
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            fn = tc.get("function") or {}
            fields = _shrinkable_fields(str(fn.get("name") or ""), allow)
            if not fields or not cid or cid not in landed or cid in recent:
                new_calls.append(tc)
                continue
            raw = fn.get("arguments")
            if not isinstance(raw, str) or len(raw) < min_chars:
                new_calls.append(tc)
                continue
            try:
                args = json.loads(raw)
            except (ValueError, TypeError):
                new_calls.append(tc)
                continue
            if not isinstance(args, dict):
                new_calls.append(tc)
                continue

            changed = False
            for field_name in fields:
                body = args.get(field_name)
                if not isinstance(body, str) or len(body) < min_chars:
                    continue
                path = None
                if session_id:
                    path = persist_for_compaction(
                        body,
                        tool_use_id=f"{cid}.args.{field_name}",
                        session_id=session_id,
                    )
                if path is None:
                    # Could not save it — leave it alone.
                    continue
                target = args.get("file_path") or args.get("path") or ""
                args[field_name] = (
                    f"{SHRUNK_ARG_MARKER} {len(body):,} chars"
                    + (f" written to {target}" if target else "")
                    + f"; the text is at {path}. Read that path if you need it again."
                )
                freed += len(body) - len(args[field_name])
                changed = True

            if not changed:
                new_calls.append(tc)
                continue

            new_fn = dict(fn)
            new_fn["arguments"] = json.dumps(args)
            new_tc = dict(tc)
            new_tc["function"] = new_fn
            new_calls.append(new_tc)
            shrunk += 1
            touched = True

        if touched:
            new_msg = dict(msg)
            new_msg["tool_calls"] = new_calls
            out.append(new_msg)
        else:
            out.append(msg)

    if shrunk:
        logger.info(
            "microcompact: shrank %d assistant tool-call argument(s), freed ~%d chars "
            "(keep_recent=%d, min_chars=%d)",
            shrunk, freed, keep_recent_tools, min_chars,
        )
    return out, shrunk, freed


__all__ = [
    "DEFAULT_COMPACTABLE_TOOLS",
    "DEFAULT_OBSERVATION_HEAD_CHARS",
    "OBSERVATION_STUB_PREFIX",
    "PREVIEW_DROPPED",
    "observation_id_for_path",
    "DEFAULT_NON_COMPACTABLE",
    "CLEARED_MARKER",
    "SHRUNK_ARG_MARKER",
    "microcompact",
    "shrink_assistant_arguments",
]
