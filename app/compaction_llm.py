"""LLM-based context summarization + post-compact file restore.

The ``app.compaction`` module owns truncation. This module owns the
two layers Claude Code calls "auto-compact" and "post-compact
cleanup": a structured LLM summary of dropped history, and a
re-injection of the most-recently-touched files so the model keeps
code fidelity after summarization.

Both are async — the summary is a non-streaming POST to the same
local vLLM endpoint that ``app.harness.client.stream_chat`` talks to.
The summarization prompt is a port of Claude Code's
``BASE_COMPACT_PROMPT`` (9-section structure), with the Anthropic-API
specifics removed and a Lloyd note added explaining that tool
arguments and full results are already persisted on disk via the
spill mechanism, so the summary doesn't need to reproduce them.

Failure mode: any unrecoverable error (HTTP, timeout, empty output)
returns ``None`` from ``summarize_history``. The caller is expected
to fall back to plain truncation so the user's turn still completes.
Same posture as the inner-voice critic — best-effort, never
load-bearing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from app.compaction import estimate_tokens, _message_text

logger = logging.getLogger("lloyd-compaction-llm")


# ---------------------------------------------------------------------------
# Summarization prompt (ported from Claude Code's BASE_COMPACT_PROMPT)
# ---------------------------------------------------------------------------

# The prompt asks for two sections:
#   <analysis> ... </analysis>   — drafting scratchpad, stripped before use
#   <summary>  ... </summary>    — the kept output, used as the assistant
#                                  message that replaces dropped history
#
# We add a Lloyd-specific note so the model doesn't waste budget
# reproducing tool arguments — those are persisted on disk under
# ``sessions/<id>.tool-results/`` via the spill module and can be
# re-read on demand.
SUMMARIZATION_SYSTEM_PROMPT = """\
You are summarizing a long conversation between a user and an AI assistant
so the assistant can continue the conversation under a smaller context
window. Your job is to capture EVERYTHING the assistant will need to
resume work seamlessly: the user's intent, the technical details of what
was done, what's still in flight, and any errors or feedback that shaped
the approach.

TEXT ONLY. Do NOT call any tools. Respond with the structure described
below — no preamble, no other commentary outside the tags.

First, draft your analysis inside <analysis>...</analysis>. Use this as
scratchpad: walk the conversation chronologically, surface user
requests, key decisions, code changes, errors and fixes. The analysis
will be stripped before the summary is shown to the assistant.

Then write the final summary inside <summary>...</summary>, using these
nine sections, in this order:

1. **Primary Request and Intent** — what the user asked for, in their
   own words where possible. Include any explicit constraints,
   non-goals, or acceptance criteria.
2. **Key Technical Concepts** — frameworks, libraries, services,
   architectural patterns, and named conventions that came up.
3. **Files and Code Sections** — file paths touched, with short
   descriptions of what was read, edited, or created. Quote critical
   snippets verbatim (function signatures, key invariants). Do NOT
   reproduce full file contents — those are persisted to disk and can
   be re-read on demand.
4. **Errors and Fixes** — every error encountered and how it was
   resolved. Especially capture user corrections ("no, not that",
   "stop doing X") since those carry strong preference signals.
5. **Problem Solving** — the line of reasoning the assistant followed,
   approaches considered, and why they were chosen or rejected.
6. **All User Messages** — list every user message verbatim (or near-
   verbatim) so the assistant can re-read user intent without losing
   nuance.
7. **Pending Tasks** — anything explicitly requested that hasn't been
   completed yet, in priority order.
8. **Current Work** — what the assistant was doing at the moment the
   summary was requested. Be specific: the exact file, function,
   command, or paragraph being worked on.
9. **Optional Next Step** — the single most logical next action.
   Reference the most recent user message or task to justify it. If
   nothing is obviously next, say so.

Note on tool calls: arguments and full results from prior tool calls
have been persisted to disk. You don't need to reproduce them. Refer
to them by tool name and intent ("read auth.py", "ran pytest, 3
failures") — the assistant can re-read the files if it needs more.

Reminder: respond with <analysis>...</analysis> followed immediately
by <summary>...</summary>. No other text.
"""


# ---------------------------------------------------------------------------
# vLLM endpoint resolution
# ---------------------------------------------------------------------------


def _summary_endpoint(model_alias: str | None) -> tuple[str, str]:
    """Resolve (base_url, canonical_model_name) for the summarization call.

    Reads ``compaction.summary_model`` first, falls back to the global
    default model. Mirrors the resolution helper in
    :mod:`app.inner_voice.critic` so both subsystems use the same
    config conventions.

    Returns ('', '') if no resolvable URL — caller treats that as a
    skip-summarization signal and falls back to truncation.
    """
    # Lazy imports — avoids circular deps at import time and keeps the
    # module testable without a full app bootstrap.
    try:
        from app.config import CONFIG, _get_model_cfg  # type: ignore
    except Exception:
        return ("", "")

    comp = CONFIG.get("compaction") or {}
    name = (
        model_alias
        or comp.get("summary_model")
        or CONFIG.get("model", {}).get("default", "")
    )
    cfg = _get_model_cfg(name) or {}
    base = cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
    return (base.rstrip("/") if base else "", name or "")


async def _post_chat_completion(
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    timeout_seconds: float,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Non-streaming POST to ``/v1/chat/completions``. Raises on HTTP
    error so the caller can fold the failure into a skip decision.
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Force-disable thinking. Summarization should spend its budget
        # on the structured output, not hidden reasoning. Same convention
        # the critic uses.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as cli:
        resp = await cli.post(
            url,
            headers={
                "Authorization": "Bearer no-key-required",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


_SUMMARY_TAG_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)


def _extract_summary(raw: str) -> str:
    """Pull the ``<summary>...</summary>`` body out of the model's reply.

    Falls back to the whole reply (with the analysis tag stripped if
    present) when the model didn't honor the tag convention. The
    summary is going into the conversation as an assistant message
    either way; better to keep something than crash.
    """
    if not raw:
        return ""
    m = _SUMMARY_TAG_RE.search(raw)
    if m:
        return m.group(1).strip()
    # No <summary> tag — strip a leading <analysis>...</analysis> block
    # if present, return the rest.
    raw = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    return raw


def _format_history_for_summary(messages: list[dict]) -> str:
    """Serialize the conversation block being summarized into a single
    text payload. Uses the same flattening that
    :func:`app.compaction.estimate_message_tokens` uses, so token
    accounting and prompt construction stay aligned.

    Each message gets a ``[role]`` prefix so the summarizer can tell
    user from assistant from tool. Image blocks are stripped — they
    don't fit in a text-only summary request, and including them
    bloats the request without helping the output.
    """
    lines: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "?").upper()
        text = _message_text(msg)
        if not text.strip():
            continue
        lines.append(f"[{role}]\n{text}")
    return "\n\n".join(lines)


async def summarize_history(
    messages_to_summarize: list[dict],
    *,
    model: str | None = None,
    instructions: str | None = None,
    max_output_tokens: int = 8000,
    timeout_seconds: float = 120.0,
) -> str | None:
    """Summarize a block of conversation history into a single string.

    Returns the body of the ``<summary>`` block (without the tags),
    or ``None`` on any unrecoverable failure. The caller is responsible
    for wrapping the result in a message dict and inserting it into
    the new conversation history.

    Parameters
    ----------
    messages_to_summarize
        The block being summarized — typically all messages older than
        the most recent ``keep_recent_turns``.
    model
        Optional model alias override. Defaults to
        ``compaction.summary_model`` from config.yaml, then to the
        global default model.
    instructions
        Optional user-supplied focus instruction (from the manual
        ``/compact`` command). Appended to the system prompt as a
        focus directive.
    max_output_tokens
        Cap on the model's reply. Generous default — the 9-section
        format can be long.
    timeout_seconds
        Hard outer ceiling. If the local vLLM is loaded with a long
        queue, summarization can take a while; default is set high.
    """
    if not messages_to_summarize:
        return None

    base_url, model_name = _summary_endpoint(model)
    if not base_url:
        logger.warning("summarize_history: no base_url for model %r — skipping", model)
        return None

    history_text = _format_history_for_summary(messages_to_summarize)
    if not history_text.strip():
        return None

    system_prompt = SUMMARIZATION_SYSTEM_PROMPT
    if instructions:
        # Append, not prepend — the section-structure spec must remain
        # the dominant signal. The user's focus is a refinement.
        system_prompt = (
            system_prompt
            + "\n\nUser's focus for this summarization:\n"
            + instructions.strip()
        )

    request_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Summarize the following conversation block per the "
                "spec in the system prompt:\n\n" + history_text
            ),
        },
    ]

    try:
        data = await _post_chat_completion(
            base_url=base_url,
            model_name=model_name,
            messages=request_messages,
            max_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
    except httpx.HTTPError as e:
        logger.warning("summarize_history: HTTP error: %s", e)
        return None
    except Exception as e:  # noqa: BLE001 — broad on purpose, this is best-effort
        logger.warning("summarize_history: unexpected %s: %s", type(e).__name__, e)
        return None

    raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    summary = _extract_summary(raw)
    if not summary.strip():
        logger.warning("summarize_history: empty summary returned")
        return None

    usage = data.get("usage") or {}
    logger.info(
        "summarize_history: %d input msgs → %d-char summary "
        "(prompt_tokens=%d, completion_tokens=%d)",
        len(messages_to_summarize),
        len(summary),
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
    )
    return summary


# ---------------------------------------------------------------------------
# Post-compact file restore (Layer C)
# ---------------------------------------------------------------------------

# Regex to pull a file path out of a tool-call argument JSON. Matches
# the standard ``"file_path": "..."`` shape used by Read/Edit/Write.
_FILE_PATH_RE = re.compile(r'"file_path"\s*:\s*"([^"]+)"')

# Tool names whose results are file content the model probably wants
# back inline after a compact. Only Read/Edit/Write — Bash/Grep/Glob
# return ad-hoc text that's already in the summary by reference.
_RESTORE_TOOL_NAMES: tuple[str, ...] = ("Read", "Edit", "Write")


def _extract_file_paths_from_dropped(
    dropped_messages: list[dict],
) -> list[str]:
    """Walk dropped messages newest → oldest, pull unique file paths
    from Read/Edit/Write tool calls. Most-recent occurrence wins; the
    same file mentioned twice is restored once.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    # Walk newest first so the first occurrence we see is the most
    # recent.
    for msg in reversed(dropped_messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            # Tolerate the legacy mcp__server__name form found in old
            # session JSON; current advertise is bare-named.
            bare = name.rsplit("__", 1)[-1] if "__" in name else name
            if bare not in _RESTORE_TOOL_NAMES:
                continue
            args_raw = fn.get("arguments") or "{}"
            # Try strict JSON parse; fall back to regex (some local
            # models emit slightly malformed JSON in arguments).
            path: str | None = None
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                if isinstance(args, dict):
                    path = args.get("file_path") or args.get("path")
            except (json.JSONDecodeError, TypeError):
                m = _FILE_PATH_RE.search(args_raw if isinstance(args_raw, str) else "")
                if m:
                    path = m.group(1)
            if path and path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def _truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate ``text`` to roughly ``max_tokens``. Returns (text, was_truncated).

    Uses :func:`app.compaction.estimate_tokens`'s 4-char-per-token
    heuristic so the budget here matches the budget used elsewhere.
    Cuts at the last newline within the budget when possible so the
    truncation point is on a line boundary.
    """
    if estimate_tokens(text) <= max_tokens:
        return text, False
    # 4 chars per token → max chars = max_tokens * 4
    max_chars = max_tokens * 4
    head = text[:max_chars]
    nl = head.rfind("\n")
    cut = nl if nl > max_chars // 2 else max_chars
    return text[:cut], True


def restore_recent_files(
    dropped_messages: list[dict],
    *,
    budget_tokens: int = 50_000,
    max_per_file: int = 5_000,
    max_files: int = 5,
) -> list[dict]:
    """Re-inject the most-recently-touched files as system messages.

    Returns a list of system-role message dicts ready to be inserted
    into the new history right after the summary message. Each entry
    is tagged with a ``[restored-context: file=<path>]`` header so the
    model and the UI can both tell what's restored vs. live tool
    output.

    Files are re-read from disk *fresh* (current state), not from the
    original tool result. This is intentional — files often change
    between when they were read and when the compaction fires; the
    model wants the current state, not a stale snapshot.

    Files that no longer exist or aren't readable are silently
    skipped. Each file is capped at ``max_per_file`` tokens; the
    cumulative budget is ``budget_tokens``; at most ``max_files``
    are restored.
    """
    paths = _extract_file_paths_from_dropped(dropped_messages)
    if not paths:
        return []

    out: list[dict] = []
    spent = 0
    for path in paths:
        if len(out) >= max_files:
            break
        if spent >= budget_tokens:
            break
        try:
            p = Path(path).expanduser()
            if not p.exists() or not p.is_file():
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("restore_recent_files: skip %s: %s", path, e)
            continue
        # Per-file cap, then check global budget afterwards.
        per_file_cap = min(max_per_file, max(0, budget_tokens - spent))
        if per_file_cap <= 0:
            break
        content, was_trunc = _truncate_to_tokens(content, per_file_cap)
        header = f"[restored-context: file={path}"
        if was_trunc:
            header += f" truncated_to={per_file_cap}t"
        header += "]\n"
        body = header + content
        out.append({
            "role": "system",
            "content": [{"type": "text", "text": body}],
        })
        spent += estimate_tokens(body)

    if out:
        logger.info(
            "restore_recent_files: re-injected %d/%d candidate files (~%d tokens)",
            len(out), len(paths), spent,
        )
    return out


__all__ = [
    "SUMMARIZATION_SYSTEM_PROMPT",
    "summarize_history",
    "restore_recent_files",
]
