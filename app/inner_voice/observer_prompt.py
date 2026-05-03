"""System prompts and intervention schema for the Inner Voice observer.

The two system prompts (per-event observer + goal extraction) live in the
vault as editable markdown:

  ~/obsidian/lloyd/inner_voice/system_prompt.md
  ~/obsidian/lloyd/inner_voice/goal_extraction_prompt.md

This module loads them at import time, strips frontmatter, and exposes
them as `SYSTEM_PROMPT` / `GOAL_EXTRACTION_SYSTEM_PROMPT`. If a vault
file is missing or unreadable, a minimal embedded fallback keeps the
observer functional. Restart the backend to pick up edits.

The schema (output format, prefill, lever names) is part of the prompt
contract, not config — the parser in `observer.py` and the prompts here
must agree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-iv-observer-prompt")


# ---------------------------------------------------------------------------
# Vault-backed prompt loading
# ---------------------------------------------------------------------------

_LLOYD_HOME = Path(__file__).resolve().parents[2]
_VAULT_INNER_VOICE_DIR = _LLOYD_HOME.parent / "obsidian" / "lloyd" / "inner_voice"

_SYSTEM_PROMPT_PATH = _VAULT_INNER_VOICE_DIR / "system_prompt.md"
_GOAL_EXTRACTION_PATH = _VAULT_INNER_VOICE_DIR / "goal_extraction_prompt.md"


def _strip_frontmatter(content: str) -> str:
    """Strip a leading YAML frontmatter block (between `---` markers)."""
    s = content.lstrip()
    if not s.startswith("---"):
        return content.strip()
    end = s.find("\n---", 3)
    if end == -1:
        return content.strip()
    # Skip past the closing `---` line.
    rest = s[end + 4:]
    if rest.startswith("\n"):
        rest = rest[1:]
    return rest.strip()


def _load_prompt_file(path: Path, fallback: str, *, label: str) -> str:
    """Load a prompt markdown file from the vault, stripping frontmatter.

    Returns `fallback` on any error (missing file, read error, empty body).
    """
    try:
        if not path.exists():
            logger.warning(
                "[iv.observer.prompt] %s missing at %s — using embedded fallback",
                label, path,
            )
            return fallback
        raw = path.read_text(encoding="utf-8")
        body = _strip_frontmatter(raw)
        if not body:
            logger.warning(
                "[iv.observer.prompt] %s at %s is empty after frontmatter — "
                "using embedded fallback", label, path,
            )
            return fallback
        return body
    except Exception as e:  # noqa: BLE001 — defensive
        logger.warning(
            "[iv.observer.prompt] failed to load %s from %s: %s — using fallback",
            label, path, e,
        )
        return fallback


# Embedded fallbacks. Intentionally minimal — enough to keep the observer
# functional on a fresh checkout or when the vault isn't mounted, but the
# vault copy is the source of truth. Edit the markdown in the vault, not
# these.
_FALLBACK_SYSTEM_PROMPT = """You are Lloyd's Inner Voice — watch the primary agent and intervene only when it's drifting, looping, or about to do something destructive.

Levers: noop (default), inject (chat history nudge), cancel (stop iteration; also use when goal complete), ambient (queue follow-up turn), clarify (ask user a question, pauses primary). For pretool: allow or deny_tool.

Output a single JSON object: {"action":"...","reason":"...","content":"..."}. Most events should be noop.
"""

_FALLBACK_GOAL_EXTRACTION_PROMPT = """Extract a goal card from the user's request. Output a single JSON object on one line:
{"success_criteria":["..."],"out_of_scope":["..."],"completion_signals":["..."]}
Empty lists if the request is conversational or has no actionable goal.
"""


SYSTEM_PROMPT = _load_prompt_file(
    _SYSTEM_PROMPT_PATH,
    _FALLBACK_SYSTEM_PROMPT,
    label="system_prompt",
)

GOAL_EXTRACTION_SYSTEM_PROMPT = _load_prompt_file(
    _GOAL_EXTRACTION_PATH,
    _FALLBACK_GOAL_EXTRACTION_PROMPT,
    label="goal_extraction_prompt",
)


# ---------------------------------------------------------------------------
# Goal extraction user-prompt builder (kept in code — small, format-coupled
# to the response parser in observer.py).
# ---------------------------------------------------------------------------

GOAL_EXTRACTION_PREFILL = '{"success_criteria":'


def build_goal_extraction_user_prompt(user_request: str) -> str:
    return (
        f"USER REQUEST:\n{user_request}\n\n"
        f"Extract the goal card. Return JSON only."
    )


# Cap on interventions per turn. After this, the observer can only `noop`
# or `allow`. Prevents runaway intervention loops.
DEFAULT_INTERVENTION_BUDGET = 3

# Response token cap. Observer answers are short JSON; no need for more.
DEFAULT_MAX_TOKENS = 400

# Per-call timeout. Pretool calls especially must be tight.
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_PRETOOL_TIMEOUT_SECONDS = 3.0

# Goal extraction call gets its own (slightly larger) budget.
DEFAULT_GOAL_EXTRACTION_TIMEOUT_SECONDS = 8.0
DEFAULT_GOAL_EXTRACTION_MAX_TOKENS = 600

# JSON prefill so the model returns structured output even when its first
# token would otherwise be commentary.
JSON_PREFILL = '{"action":'


# ---------------------------------------------------------------------------
# Per-event prompt assembly
# ---------------------------------------------------------------------------


def _format_goal_card(goal_card: dict[str, Any] | None) -> str:
    """Render the goal card as a compact prompt block."""
    if not goal_card:
        return "GOAL CARD: (no actionable goal extracted — request was conversational or empty)"
    sc = goal_card.get("success_criteria") or []
    oos = goal_card.get("out_of_scope") or []
    cs = goal_card.get("completion_signals") or []
    parts = ["GOAL CARD:"]
    if sc:
        parts.append("  Success criteria:")
        parts.extend(f"    - {s}" for s in sc)
    if oos:
        parts.append("  Out of scope:")
        parts.extend(f"    - {s}" for s in oos)
    if cs:
        parts.append("  Completion signals:")
        parts.extend(f"    - {s}" for s in cs)
    if len(parts) == 1:
        return "GOAL CARD: (extracted, but empty — request had no concrete actionable items)"
    return "\n".join(parts)


def _format_prior_decisions(decisions: list[dict[str, Any]] | None) -> str:
    """Render the observer's prior decisions this turn."""
    if not decisions:
        return ""
    lines = ["YOUR PRIOR DECISIONS THIS TURN:"]
    for d in decisions[-8:]:  # cap at last 8 to keep prompts bounded
        trig = d.get("trigger", "?")
        act = d.get("action", "?")
        rsn = (d.get("reason") or "")[:80]
        related = d.get("related_tool")
        related_str = f" [{related}]" if related else ""
        lines.append(f"  - {trig}{related_str}: {act} — {rsn}")
    return "\n".join(lines) + "\n"


def build_user_prompt_for_event(
    *,
    user_request: str,
    goal_card: dict[str, Any] | None,
    event_summary: str,
    primary_text_so_far: str,
    interventions_used: int,
    interventions_budget: int,
    prior_decisions: list[dict[str, Any]] | None = None,
) -> str:
    """Assemble the per-event user prompt the observer evaluates."""
    budget_line = (
        f"Interventions used this turn: {interventions_used}/{interventions_budget}."
    )
    if interventions_used >= interventions_budget:
        budget_line += (
            " You have exhausted your intervention budget — only `noop` or `allow` responses will take effect."
        )
    prior_block = _format_prior_decisions(prior_decisions)
    prior_section = f"\n{prior_block}" if prior_block else ""
    return (
        f"USER REQUEST:\n{user_request}\n\n"
        f"{_format_goal_card(goal_card)}\n\n"
        f"PRIMARY'S RESPONSE SO FAR (visible text):\n"
        f"{primary_text_so_far or '(none yet)'}\n"
        f"{prior_section}\n"
        f"EVENT UNDER REVIEW:\n{event_summary}\n\n"
        f"{budget_line}\n\n"
        f"Decide: noop, inject, cancel, ambient, clarify, or (for pretool) allow / deny_tool."
    )


def build_pretool_event_summary(tool_name: str, tool_args: dict) -> str:
    """One-line summary of a proposed tool call for the pretool gate."""
    args_preview = str(tool_args)
    if len(args_preview) > 400:
        args_preview = args_preview[:400] + "...(truncated)"
    return (
        f"PRETOOL: primary is about to call `{tool_name}` with args {args_preview}. "
        f"Decide allow or deny_tool."
    )


def build_assistant_message_summary(
    iteration: int,
    text: str,
    tool_calls: list[dict],
) -> str:
    """One-line summary of an assistant_message event for review."""
    text_preview = text[:300] + ("..." if len(text) > 300 else "")
    if tool_calls:
        names = [tc.get("function", {}).get("name") or tc.get("name") or "?" for tc in tool_calls]
        return (
            f"Iteration {iteration} finished. Text: {text_preview!r}. "
            f"Tool calls proposed: {names}."
        )
    if not text.strip():
        return (
            f"Iteration {iteration} finished with EMPTY text and NO tool calls. "
            f"Primary is about to terminate without answering."
        )
    return (
        f"Iteration {iteration} finished with text only (no tool calls). "
        f"Text: {text_preview!r}. Primary will terminate after this iteration."
    )


def build_tool_result_summary(
    tool_name: str,
    result_preview: str,
    is_error: bool,
) -> str:
    """One-line summary of a tool_result event for review."""
    label = "ERROR" if is_error else "result"
    preview = result_preview[:300] + ("..." if len(result_preview) > 300 else "")
    return (
        f"Tool {tool_name} returned {label}: {preview!r}. "
        f"Primary will see this and decide its next move."
    )


def build_result_summary(stop_reason: str, response_text: str) -> str:
    """One-line summary of the terminal `result` event."""
    txt = response_text[:300] + ("..." if len(response_text) > 300 else "")
    return (
        f"Turn ended (stop_reason={stop_reason}). Final visible response: {txt!r}. "
        f"This is your last chance to ambient a follow-up."
    )
