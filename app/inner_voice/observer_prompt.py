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
_FALLBACK_SYSTEM_PROMPT = """You are Lloyd's Inner Voice — watch the primary agent and intervene only when it's drifting, looping, or about to terminate without answering.

Each assistant_message event includes `finish_reason`. When finish_reason="stop" and the iteration has no tool_calls, the harness is about to terminate the turn — that is your last chance to redirect. The harness has an inject-extends-turn mechanism: an inject on the terminal iteration restarts the loop so primary reads your nudge. Use this when the final text is a stub announce ("Let me check X:") or otherwise fails to actually deliver an answer.

Respond by calling exactly one of the lever tools loaded into your context: noop (default), inject (chat-history nudge), cancel (stop iteration), ambient (queue follow-up turn), clarify (ask user a question, pauses primary). Most events should be noop. Hard safety on destructive Bash is enforced by the harness; you do not gate.
"""

_FALLBACK_GOAL_EXTRACTION_PROMPT = """Extract a goal card from the user's request and call the `record_goal_card` tool with three array fields: success_criteria, out_of_scope, completion_signals. Empty lists if the request is conversational or has no actionable goal.
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
# Goal extraction user-prompt builder
# ---------------------------------------------------------------------------


def _format_recent_exchanges(exchanges: list[dict[str, str]] | None) -> str:
    """Render the last few user/assistant messages so goal extraction can
    recognize follow-up turns ("yeah do it", "still broken") as continuing
    a prior request rather than treating them in isolation.
    """
    if not exchanges:
        return ""
    lines = ["PRIOR CONVERSATION (most recent last):"]
    for ex in exchanges[-6:]:
        role = ex.get("role", "?")
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        if len(text) > 400:
            text = text[:400] + "..."
        lines.append(f"  [{role}] {text}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n\n"


def build_goal_extraction_user_prompt(
    user_request: str,
    recent_exchanges: list[dict[str, str]] | None = None,
) -> str:
    prior = _format_recent_exchanges(recent_exchanges)
    continuation_hint = ""
    if prior:
        continuation_hint = (
            "If the latest USER REQUEST is a brief follow-up (\"yeah\", "
            "\"do it\", \"still broken\", \"clear those too\"), it continues "
            "the prior thread — extract the goal in light of what the user "
            "and assistant were just doing, not from the literal words alone.\n\n"
        )
    return (
        f"{prior}"
        f"USER REQUEST:\n{user_request}\n\n"
        f"{continuation_hint}"
        f"Extract the goal card by calling the `record_goal_card` tool."
    )


# Cap on interventions per turn. After this, the observer can only `noop`
# or `cancel`. Prevents runaway intervention loops.
DEFAULT_INTERVENTION_BUDGET = 3

# Response token cap — tool-call args are short.
DEFAULT_MAX_TOKENS = 400

# Per-call timeout. Pretool no longer has a separate sync deadline (v4
# pretool can't block dispatch), so the same default applies everywhere.
DEFAULT_TIMEOUT_SECONDS = 5.0

# Goal extraction call gets its own (slightly larger) budget.
DEFAULT_GOAL_EXTRACTION_TIMEOUT_SECONDS = 8.0
DEFAULT_GOAL_EXTRACTION_MAX_TOKENS = 600


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


# Cap the subliminal context surfaced to the observer per event. The full
# prefix can be 6-8 KB on skill-heavy turns; the observer doesn't need
# every byte to judge "is the primary following documented procedure."
_SUBLIMINAL_PROMPT_CHAR_CAP = 4000


def windowed_text(text: str, cap: int) -> str:
    """Trim long text to the head + tail with a marker between.

    Head-only truncation hides the response's conclusion, which is what
    the observer needs to judge completeness. For any text longer than
    `cap` chars, return the first half + "[N chars trimmed]" + the last
    half so the IV always sees how the response opens AND closes.
    """
    if not text:
        return text
    if len(text) <= cap:
        return text
    half = cap // 2
    trimmed = len(text) - cap
    return f"{text[:half]}\n... [{trimmed} chars trimmed] ...\n{text[-half:]}"


def _format_subliminal_context(subliminal: str | None) -> str:
    """Render the prefetched subliminal block (skills, vault, facts, ...)
    so the observer sees what context the primary actually had.

    The primary's request often makes sense only in light of a documented
    skill or recent vault hit. Without this, the observer judges actions
    in a vacuum and labels documented procedures as "destructive" or
    "ambiguous."
    """
    if not subliminal or not subliminal.strip():
        return ""
    body = subliminal.strip()
    if len(body) > _SUBLIMINAL_PROMPT_CHAR_CAP:
        body = body[:_SUBLIMINAL_PROMPT_CHAR_CAP] + "\n...(truncated)"
    return (
        "CONTEXT THE PRIMARY SAW (prefetched skills/vault/facts — same "
        "block was injected into the primary's user message):\n"
        f"{body}\n"
    )


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
    subliminal_context: str | None = None,
) -> str:
    """Assemble the per-event user prompt the observer evaluates."""
    budget_line = (
        f"Interventions used this turn: {interventions_used}/{interventions_budget}."
    )
    if interventions_used >= interventions_budget:
        budget_line += (
            " You have used your inject/ambient/clarify budget — only `noop`, `cancel`, or `allow` "
            "will take effect from here on. If the primary is still off-track or looping, "
            "this is the moment to use `cancel` to end the turn."
        )
    prior_block = _format_prior_decisions(prior_decisions)
    prior_section = f"\n{prior_block}" if prior_block else ""
    subliminal_block = _format_subliminal_context(subliminal_context)
    subliminal_section = f"\n{subliminal_block}" if subliminal_block else ""
    return (
        f"USER REQUEST:\n{user_request}\n\n"
        f"{_format_goal_card(goal_card)}\n"
        f"{subliminal_section}\n"
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


def _format_goal_eval_block(
    goal_card: dict[str, Any] | None,
    *,
    on_unmet: str,
) -> str:
    """Render the goal-card-driven completion check appended to terminal
    event summaries.

    The block restates the goal card's success_criteria and
    completion_signals item-by-item and asks the IV to evaluate each
    against the response under review. ``on_unmet`` is the lever name to
    fire when any item is unaddressed (``inject`` at the terminal
    assistant_message, ``ambient`` at the result event).

    Returns "" when there's no actionable goal card — caller falls back
    to its existing judgment-based language.
    """
    if not goal_card:
        return ""
    sc = goal_card.get("success_criteria") or []
    cs = goal_card.get("completion_signals") or []
    if not sc and not cs:
        return ""
    lines = ["", "GOAL-COMPLETION CHECK (REQUIRED — do not skip):"]
    if sc:
        lines.append("  Success criteria:")
        lines.extend(f"    - {s}" for s in sc)
    if cs:
        lines.append("  Completion signals:")
        lines.extend(f"    - {s}" for s in cs)
    lines.extend([
        "",
        "Walk each item above. For each, decide: did the response under review "
        "actually address it (yes), or is it still unaddressed (no)?",
        "",
        f"  - ALL items addressed → noop (the response delivered).",
        f"  - ANY item unaddressed → {on_unmet} — the work is not done.",
        "",
        "Do NOT noop on the basis that the primary 'announced intent' or "
        "'will dispatch tools next.' The harness has terminated this iteration; "
        "there is no next dispatch unless you act now. Treat textual promises "
        "of future action as evidence the work was NOT done.",
        "",
        f"Equally, do NOT {on_unmet} just because a success_criterion isn't "
        "echoed verbatim. A response that delivers the intent of a criterion "
        "(rephrased, synthesized, or covered alongside another item) is "
        f"sufficient — match meaning, not exact wording. Only {on_unmet} when "
        "an item is plainly unaddressed AND the response shows stub-announce "
        "or mid-cutoff symptoms. A response ending with terminal punctuation "
        "(`.`, `?`, `!`) and a recommendation, summary, list, or offer-to-help "
        "is the natural close of the turn and gets `noop`, not "
        f"`{on_unmet}` — even if you would have structured it differently.",
    ])
    return "\n".join(lines)


def build_assistant_message_summary(
    iteration: int,
    text: str,
    tool_calls: list[dict],
    finish_reason: str = "stop",
    goal_card: dict[str, Any] | None = None,
) -> str:
    """One-line summary of an assistant_message event for review.

    Includes vLLM's `finish_reason` so IV can distinguish "primary done"
    (`finish_reason=stop` + no tool_calls → harness terminates) from
    "primary mid-thought" (`finish_reason=tool_calls` → harness will
    loop). When finish_reason == "stop" and there are no tool calls,
    THIS is the last chance to inject; the harness uses
    inject-extends-turn (loop.py:236-256) so an inject here continues
    the loop instead of letting primary terminate.

    On the terminal branch (stop + no tool_calls + non-empty text), the
    goal card is restated inline and the IV is required to walk each
    success_criterion / completion_signal against the iteration text.
    Without this, the IV pattern-matches on textual promises of future
    action ("I'll fetch X") and noops, missing the stall.
    """
    # Head+tail windowing: head-only truncation hides the conclusion, which
    # is exactly the signal the IV needs to judge completeness. 1200 char
    # budget ≈ enough for an opening paragraph + the closing paragraph for
    # any response shape we expect.
    text_preview = windowed_text(text, 1200)
    if tool_calls:
        names = [tc.get("function", {}).get("name") or tc.get("name") or "?" for tc in tool_calls]
        return (
            f"Iteration {iteration} finished (finish_reason={finish_reason!r}). "
            f"Text: {text_preview!r}. Tool calls proposed: {names}. "
            f"Primary will continue after tool dispatch."
        )
    if finish_reason == "stop":
        if not text.strip():
            return (
                f"Iteration {iteration} finished with EMPTY text and NO tool calls "
                f"(finish_reason=stop). PRIMARY IS ABOUT TO TERMINATE WITHOUT "
                f"ANSWERING. This is your last chance to inject — the harness "
                f"will continue the loop if you do."
            )
        eval_block = _format_goal_eval_block(goal_card, on_unmet="inject")
        return (
            f"Iteration {iteration} TERMINAL (finish_reason=stop, NO tool calls). "
            f"The harness EXITS the turn at this event unless you inject. "
            f"There is no next iteration. Tools are not coming.\n"
            f"Iteration text: {text_preview!r}.{eval_block}"
        )
    # finish_reason in {length, tool_calls without committed tools, error, ...}
    return (
        f"Iteration {iteration} finished (finish_reason={finish_reason!r}). "
        f"Text: {text_preview!r}. No tool calls. Primary may or may not continue."
    )


_PERSISTED_OUTPUT_TAG = "<persisted-output>"


def _format_spilled_summary(tool_name: str, content: str) -> str | None:
    """If `content` is a spill envelope (large tool result persisted to
    disk), return a richer one-line summary that names the size + path
    explicitly. Returns None if not a spill envelope.
    """
    if _PERSISTED_OUTPUT_TAG not in content:
        return None
    # Extract size and file path from the envelope's first lines without
    # reading the spilled file. The envelope shape is:
    #   <persisted-output>
    #   Output too large (68.3 KB, 69,943 chars). Full output saved to: /path/to/file
    #   ...
    head_lines = content.splitlines()[:6]
    size_line = next((l for l in head_lines if "Output too large" in l), "")
    path_line = next((l for l in head_lines if "saved to:" in l), "")
    size = ""
    if size_line:
        # Pull "(68.3 KB, 69,943 chars)" tail
        try:
            size = size_line.split("(", 1)[1].split(")", 1)[0]
        except IndexError:
            pass
    path = ""
    if path_line:
        path = path_line.split("saved to:", 1)[-1].strip()
    # First few content chars (after the envelope header) for flavor
    first_data = ""
    in_preview = False
    for line in content.splitlines():
        if "Preview" in line and ":" in line:
            in_preview = True
            continue
        if in_preview and line.strip():
            first_data = line[:120]
            break
    return (
        f"Tool {tool_name} returned a SPILLED result ({size or 'large'}); "
        f"full content at {path or 'disk'}. Primary saw a small preview; "
        f"can re-Read on demand. Preview opener: {first_data!r}. "
        f"Judge whether the primary is making progress with the data — not "
        f"whether it 'should' read more."
    )


def build_tool_result_summary(
    tool_name: str,
    result_preview: str,
    is_error: bool,
) -> str:
    """One-line summary of a tool_result event for review."""
    if not is_error:
        spilled = _format_spilled_summary(tool_name, result_preview)
        if spilled is not None:
            return spilled
    label = "ERROR" if is_error else "result"
    preview = result_preview[:300] + ("..." if len(result_preview) > 300 else "")
    return (
        f"Tool {tool_name} returned {label}: {preview!r}. "
        f"Primary will see this and decide its next move."
    )


def build_result_summary(
    stop_reason: str,
    response_text: str,
    goal_card: dict[str, Any] | None = None,
) -> str:
    """Summary of the terminal `result` event with goal-completion check.

    The harness has already exited by the time this event fires — inject
    is a no-op here and only ``ambient`` (queue follow-up turn) takes
    effect. The summary restates the goal card and forces an explicit
    item-by-item completion check, so the IV can't carry over a stale
    "will dispatch tools next" judgment from the assistant_message event.
    """
    txt = windowed_text(response_text, 1200)
    eval_block = _format_goal_eval_block(goal_card, on_unmet="ambient")
    if eval_block:
        return (
            f"Turn ENDED (stop_reason={stop_reason}). The harness has already "
            f"terminated. There will be no more text or tool calls. "
            f"Final visible response: {txt!r}.{eval_block}"
        )
    # No goal card — fall back to lighter-touch judgment.
    return (
        f"Turn ENDED (stop_reason={stop_reason}). The harness has already "
        f"terminated. Final visible response: {txt!r}. "
        f"If the response delivers a substantive answer → noop. If it's a "
        f"stub announce ('I'll X', 'Let me Y') or otherwise fails to deliver, "
        f"ambient a follow-up turn that completes the work."
    )
