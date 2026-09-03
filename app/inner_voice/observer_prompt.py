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
_GOAL_COMPLETION_PATH = _VAULT_INNER_VOICE_DIR / "goal_completion_prompt.md"


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

_FALLBACK_GOAL_COMPLETION_PROMPT = """You are Lloyd's goal-completion evaluator. The user set a persistent goal via /goal. After each turn you judge — strictly — whether the goal is actually met.

Call `record_goal_completion` with:
- `achieved`: true only when the goal's verifiable end condition is plainly satisfied by what already happened in the conversation (files written, tools confirmed the action, the answer was delivered). Default to false when in doubt.
- `reason`: when achieved=false, write the next user-visible follow-up — be specific about what is still missing and what concrete step should come next. Treat textual promises of future action ("I'll do X next") as evidence the work was NOT done.
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

GOAL_COMPLETION_SYSTEM_PROMPT = _load_prompt_file(
    _GOAL_COMPLETION_PATH,
    _FALLBACK_GOAL_COMPLETION_PROMPT,
    label="goal_completion_prompt",
)


# ---------------------------------------------------------------------------
# Hot reload
# ---------------------------------------------------------------------------
#
# The prompts above are the import-time snapshot, kept as module constants
# for back-compat. Live code should call the accessors below, which re-read
# the vault file whenever its mtime changes.
#
# Editing a prompt used to require a backend restart, which made the tuning
# loop — the whole reason judgment lives in the vault rather than in Python —
# far slower than it needed to be. An mtime stat per observer call is
# negligible next to the LLM round-trip it precedes.

_PROMPT_CACHE: dict[str, tuple[float, str]] = {}


def _load_cached(path: Path, fallback: str, *, label: str) -> str:
    """Return the prompt body, re-reading only when the file's mtime moves.

    A missing file stats as mtime 0.0 and serves the fallback; dropping the
    file in later changes the mtime and picks it up without a restart.
    """
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    cached = _PROMPT_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    body = _load_prompt_file(path, fallback, label=label)
    _PROMPT_CACHE[key] = (mtime, body)
    if cached is not None:
        logger.info("[iv.observer.prompt] reloaded %s from %s", label, path)
    return body


def get_system_prompt() -> str:
    """The per-event observer system prompt (vault, hot-reloaded)."""
    return _load_cached(
        _SYSTEM_PROMPT_PATH, _FALLBACK_SYSTEM_PROMPT, label="system_prompt",
    )


def get_goal_extraction_prompt() -> str:
    """The turn-start goal-extraction system prompt (vault, hot-reloaded)."""
    return _load_cached(
        _GOAL_EXTRACTION_PATH, _FALLBACK_GOAL_EXTRACTION_PROMPT,
        label="goal_extraction_prompt",
    )


def get_goal_completion_prompt() -> str:
    """The `/goal` completion-evaluator system prompt (vault, hot-reloaded)."""
    return _load_cached(
        _GOAL_COMPLETION_PATH, _FALLBACK_GOAL_COMPLETION_PROMPT,
        label="goal_completion_prompt",
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


def _format_persistent_goal(persistent_goal: dict[str, Any] | None) -> str:
    """Render the session's persistent goal (the /goal target) for IV.

    This is the user's session-wide north star, distinct from the per-turn
    goal_card extracted from the immediate request. The observer sees both:
    goal_card is "what does THIS turn need to deliver?" and persistent_goal
    is "what is the whole session trying to reach?".
    """
    if not persistent_goal:
        return ""
    text = (persistent_goal.get("text") or "").strip()
    if not text:
        return ""
    if persistent_goal.get("achieved_at"):
        return ""
    attempts = int(persistent_goal.get("attempts") or 0)
    attempt_line = f" (attempt {attempts + 1})" if attempts else ""
    return (
        f"PERSISTENT GOAL (user's /goal){attempt_line}: {text}\n"
        "Every turn should advance this goal or surface a concrete "
        "blocker. If the primary stops with the goal still unmet, the "
        "post-turn evaluator will queue a follow-up; you can also "
        "inject/ambient mid-turn if the primary is clearly drifting "
        "from the goal."
    )


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


def _format_plan_artifact(plan: dict[str, Any] | None) -> str:
    """Plan B — render the committed plan or plan_mode state for IV.

    When `plan_mode=True`, emit a PLANNING-MODE banner so the IV knows
    to evaluate plan quality (Phase B2 will pair this with the
    iv-plan-review skill). When a committed plan exists, render a
    compact reference block with stages so IV can judge whether the
    primary's actions advance the plan or drift from it. Empty when
    no plan and not in planning mode — caller falls back to the
    todos-only flow.
    """
    if not plan:
        return ""
    if plan.get("plan_mode"):
        return (
            "PLANNING MODE ACTIVE — evaluate plan QUALITY, not execution drift. "
            "Primary is drafting a plan via the plan-mode-authoring playbook. "
            "Use this rubric (see iv-plan-review skill for the full version):\n"
            "  • No clarifying questions asked → inject 'Ask 3 questions or "
            "state 3 assumptions before drafting plan_md.'\n"
            "  • Vague stage (goal, not deliverable) → inject 'How will we "
            "know stage X is done?'\n"
            "  • Missing acceptance criteria → inject 'What proves this stage "
            "worked?'\n"
            "  • Premature ExitPlanMode (pretool with obvious gaps) → inject "
            "BEFORE the commit lands, naming the gap.\n"
            "  • TodoWrite called instead of ExitPlanMode → inject 'Roll the "
            "todos into ExitPlanMode along with plan_md; TodoWrite alone "
            "leaves the session stuck in plan_mode.'\n"
            "  • Plan is overkill for a trivial request → inject 'Consider "
            "ExitPlanMode(cancel=true) and just do the work.'\n"
            "Off-plan tool calls during research are NOT drift — Read/Grep/"
            "skills_search exploration is the playbook. Noop unless one of "
            "the rubric triggers is concretely present."
        )
    stages = plan.get("stages") or []
    if not stages:
        return ""
    lines = ["PLAN ARTIFACT (committed plan — primary's stable cross-turn anchor):"]
    for s in stages:
        if not isinstance(s, dict):
            continue
        n = s.get("n", "?")
        title = (s.get("title") or "").strip()
        if not title:
            continue
        lines.append(f"  Stage {n}: {title}")
    if len(lines) == 1:
        return ""
    lines.append(
        "Use the plan as an additional progress reference. The TODOS "
        "block (when present) is the live state; the plan is the "
        "frame. Off-plan tool calls aren't automatic drift, but if "
        "primary is clearly working outside the plan's scope, that's "
        "the signal to inject."
    )
    return "\n".join(lines)


def _format_todos_block(todos: list[dict[str, Any]] | None) -> str:
    """Plan A — render `session.todos` as a compact reference block.

    This is the ongoing-context block included on every IV per-event
    prompt when stewardship is on. Distinct from the gating block in
    `_format_pending_todos_block`, which fires only at terminal events
    and forces a completion check.
    """
    if not todos:
        return ""
    lines = ["TODOS (primary's committed plan — reference artifact):"]
    for t in todos:
        status = (t.get("status") or "?").strip()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"  - [{status}] {content}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _format_pending_todos_block(
    todos: list[dict[str, Any]] | None,
    *,
    on_unmet: str,
) -> str:
    """Plan A.4 — the completion-gate eval block appended at terminal events.

    Walks the todo list and asks IV to fire `on_unmet` (`inject` at
    assistant_message, `ambient` at result) when any todo remains pending
    or in_progress and the response under review doesn't actually finish
    them. Returns "" when there are no todos or all are completed —
    caller falls back to the goal-card eval block alone.
    """
    if not todos:
        return ""
    pending = [t for t in todos if (t.get("status") or "") in ("pending", "in_progress")]
    if not pending:
        return ""
    lines = ["", "PENDING TODOS (REQUIRED — primary is about to stop with these unresolved):"]
    for t in pending:
        status = (t.get("status") or "?").strip()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"    - [{status}] {content}")
    lines.extend([
        "",
        "Walk each pending/in_progress todo above. For each, decide: did the "
        "response under review actually accomplish it (yes), or is it still "
        "pending (no)?",
        "",
        f"  - ALL pending todos addressed → noop (the work is done).",
        f"  - ANY pending todo unaddressed → {on_unmet}, naming the specific "
        "todo and what's missing.",
        "",
        "Do NOT noop on the basis that the primary 'announced intent' or "
        "'will dispatch tools next.' The harness has terminated this iteration; "
        "there is no next dispatch unless you act now. A textual promise to "
        "do work later is evidence the work was NOT done.",
        "",
        f"Equally, do NOT {on_unmet} just because a todo isn't echoed verbatim. "
        "A response that delivers the substance of a todo (rephrased, "
        "synthesized, or bundled with another) is sufficient — match meaning, "
        f"not exact wording. Only {on_unmet} when a todo is plainly "
        "unaddressed AND the response shows stub-announce or mid-cutoff "
        "symptoms.",
    ])
    return "\n".join(lines)


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


def _format_prior_turn_interventions(
    rows: list[dict[str, Any]] | None,
) -> str:
    """Render what the observer did on EARLIER turns of this session.

    Cross-turn memory. Until now the observer attached with no knowledge
    of its own history: it could nudge the primary about the same drift
    on five consecutive turns and never notice the nudge wasn't working,
    because `decisions_this_turn` resets at every attach.

    Deliberately narrow — interventions only, not the hundreds of noops.
    An intervention is the observer having spent something; a noop is
    silence and carries no lesson forward.
    """
    if not rows:
        return ""
    lines = ["WHAT YOU DID ON EARLIER TURNS OF THIS SESSION:"]
    for r in rows[-6:]:
        action = r.get("action") or "?"
        reason = (r.get("reason") or "").strip()[:100]
        trig = r.get("trigger") or "?"
        lines.append(f"  - [{trig}] {action} — {reason}")
    lines.append(
        "If you are about to intervene on a theme you already raised in an "
        "earlier turn and the primary still hasn't changed, that's a sign the "
        "nudge isn't landing — escalate or surface it to the user rather than "
        "repeating yourself. If an earlier intervention clearly worked, don't "
        "re-litigate it."
    )
    return "\n".join(lines)


def build_iteration_pressure_note(
    iteration: int, max_turns: int, elapsed_s: float = 0.0,
) -> str:
    """Warn the observer that the turn is running out of iterations.

    A turn that hits `max_turns` is killed by the harness mid-work: there
    is no terminal `assistant_message`, so the last-chance inject never
    fires and the only repair left is an ambient after the fact. Telling
    the observer while iterations remain converts an after-the-fact
    apology into a mid-turn "wrap up and deliver what you have".
    """
    remaining = max(0, max_turns - iteration)
    elapsed_note = f" ({elapsed_s:.0f}s elapsed)" if elapsed_s >= 1 else ""
    return (
        f"ITERATION PRESSURE: this turn is at iteration {iteration} of a "
        f"{max_turns} maximum — {remaining} left{elapsed_note}. If the harness "
        f"hits the cap the turn dies mid-work with no final answer and no "
        f"chance for you to intervene. If the primary is still exploring "
        f"rather than converging, inject now telling it to stop gathering, "
        f"commit to what it has, and deliver the answer. If it is clearly on "
        f"its final steps, noop."
    )


def build_goal_card_block_for_primary(goal_card: dict[str, Any] | None) -> str:
    """Render the goal card as a block for the PRIMARY's user message.

    The card is extracted every IV turn and, until now, only the observer
    ever read it. Showing the primary the same contract it is being
    judged against is close to free — the extraction call already
    happened — and it removes a whole class of intervention where the
    observer nudges the primary toward a criterion the primary was never
    told about.

    Returns "" when the card has no actionable content, so conversational
    turns don't get a pointless block.
    """
    if not goal_card:
        return ""
    sc = goal_card.get("success_criteria") or []
    cs = goal_card.get("completion_signals") or []
    oos = goal_card.get("out_of_scope") or []
    if not sc and not cs and not oos:
        return ""
    lines = [
        "<goal_card>",
        "Your inner voice extracted this contract from the request and will "
        "check your work against it. Treat it as a reading of the ask, not a "
        "replacement for it — if it misreads what the user wants, follow the "
        "user and say so.",
    ]
    if sc:
        lines.append("Success criteria:")
        lines.extend(f"  - {s}" for s in sc)
    if cs:
        lines.append("Done when:")
        lines.extend(f"  - {s}" for s in cs)
    if oos:
        lines.append("Out of scope:")
        lines.extend(f"  - {s}" for s in oos)
    lines.append("</goal_card>")
    return "\n".join(lines)


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
    todos: list[dict[str, Any]] | None = None,
    plan_artifact: dict[str, Any] | None = None,
    persistent_goal: dict[str, Any] | None = None,
    prior_turn_interventions: list[dict[str, Any]] | None = None,
    iteration_pressure_note: str = "",
) -> str:
    """Assemble the per-event user prompt the observer evaluates."""
    budget_line = (
        f"Interventions used this turn: {interventions_used}/{interventions_budget}."
    )
    if interventions_used >= interventions_budget:
        budget_line += (
            " You have used your inject/ambient/clarify budget — only `noop` or `cancel` "
            "will take effect from here on. If the primary is still off-track or looping, "
            "this is the moment to use `cancel` to end the turn."
        )
    prior_block = _format_prior_decisions(prior_decisions)
    prior_section = f"\n{prior_block}" if prior_block else ""
    history_block = _format_prior_turn_interventions(prior_turn_interventions)
    history_section = f"\n{history_block}\n" if history_block else ""
    pressure_section = (
        f"\n{iteration_pressure_note}\n" if iteration_pressure_note else ""
    )
    subliminal_block = _format_subliminal_context(subliminal_context)
    subliminal_section = f"\n{subliminal_block}" if subliminal_block else ""
    plan_block = _format_plan_artifact(plan_artifact)
    plan_section = f"\n{plan_block}\n" if plan_block else ""
    todos_block = _format_todos_block(todos)
    todos_section = f"\n{todos_block}\n" if todos_block else ""
    persistent_goal_block = _format_persistent_goal(persistent_goal)
    persistent_goal_section = f"{persistent_goal_block}\n\n" if persistent_goal_block else ""
    return (
        f"USER REQUEST:\n{user_request}\n\n"
        f"{persistent_goal_section}"
        f"{_format_goal_card(goal_card)}\n"
        f"{plan_section}"
        f"{todos_section}"
        f"{subliminal_section}"
        f"{history_section}\n"
        f"PRIMARY'S RESPONSE SO FAR (visible text):\n"
        f"{primary_text_so_far or '(none yet)'}\n"
        f"{prior_section}"
        f"{pressure_section}\n"
        f"EVENT UNDER REVIEW:\n{event_summary}\n\n"
        f"{budget_line}\n\n"
        f"Call exactly one lever tool: noop, inject, cancel, ambient, or clarify."
    )


def build_goal_completion_user_prompt(
    *,
    goal_text: str,
    user_request: str,
    response_text: str,
    attempts: int,
    max_attempts: int,
    recent_tool_calls: list[str] | None = None,
) -> str:
    """User prompt for the persistent-goal post-turn evaluator.

    Called from `observer._evaluate_goal_completion` at the `result` event
    when the session has a /goal set. The model must call
    `record_goal_completion` with achieved=bool, reason=str. Reason on
    `achieved=false` becomes the next ambient follow-up's body.
    """
    recent_block = ""
    if recent_tool_calls:
        names = ", ".join(recent_tool_calls[-12:])
        recent_block = f"\nTools called this turn (most recent last): {names}\n"
    txt = windowed_text(response_text or "", 1500)
    attempt_line = (
        f"\nThis is attempt {attempts + 1}/{max_attempts} since the goal "
        "was set. If the goal is plainly impossible from the agent's side "
        "(needs external action, missing credentials, etc.), say so in "
        "`reason` so the user can intervene."
    )
    return (
        f"PERSISTENT GOAL: {goal_text}\n\n"
        f"This turn's user request: {user_request}\n"
        f"{recent_block}\n"
        f"Primary's final visible response: {txt!r}\n"
        f"{attempt_line}\n\n"
        "Decide strictly: has the persistent goal been achieved? Call "
        "`record_goal_completion`."
    )


def build_pretool_event_summary(tool_name: str, tool_args: dict) -> str:
    """One-line summary of a proposed tool call for the pretool trigger.

    Pretool is observation-only since v4 — the observer cannot block
    dispatch, and destructive Bash is hard-denied by
    `app/harness/safety.py` without any LLM in the path. The only useful
    levers here are `noop` and, on clear off-task work, `inject` (which
    lands as the next user message AFTER the tool runs).
    """
    args_preview = str(tool_args)
    if len(args_preview) > 400:
        args_preview = args_preview[:400] + "...(truncated)"
    return (
        f"PRETOOL: primary is about to call `{tool_name}` with args {args_preview}. "
        f"You cannot block this dispatch — the tool will run either way. "
        f"`noop` unless the call is clearly off-task, in which case `inject`."
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
    todos: list[dict[str, Any]] | None = None,
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
        pending_block = _format_pending_todos_block(todos, on_unmet="inject")
        if not text.strip():
            return (
                f"Iteration {iteration} finished with EMPTY text and NO tool calls "
                f"(finish_reason=stop). PRIMARY IS ABOUT TO TERMINATE WITHOUT "
                f"ANSWERING. This is your last chance to inject — the harness "
                f"will continue the loop if you do.{pending_block}"
            )
        eval_block = _format_goal_eval_block(goal_card, on_unmet="inject")
        return (
            f"Iteration {iteration} TERMINAL (finish_reason=stop, NO tool calls). "
            f"The harness EXITS the turn at this event unless you inject. "
            f"There is no next iteration. Tools are not coming.\n"
            f"Iteration text: {text_preview!r}.{eval_block}{pending_block}"
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


def _format_mark_without_evidence_block(
    flips: list[dict[str, str]] | None,
    recent_decisions: list[dict[str, Any]] | None,
) -> str:
    """Plan A.5 — challenge an in_progress→completed flip without evidence.

    Lists the flips and the last few tool calls (via the IV's own decisions
    log, which carries `related_tool` for each prior event). Asks the IV
    to judge whether the recent work plausibly accomplished each completed
    todo. Returns "" when there are no flips — caller falls back to the
    normal tool_result summary.
    """
    if not flips:
        return ""
    lines = ["", "MARK-WITHOUT-EVIDENCE CHECK (REQUIRED — do not skip):"]
    lines.append("Primary just flipped these todos to `completed`:")
    for f in flips:
        c = (f.get("content") or "").strip()
        if not c:
            continue
        lines.append(f"  - {c}")
    # Pull the last ~6 tool-related decisions so IV can judge "did real
    # work happen recently?" The decisions log carries one entry per IV
    # decision; we filter to those tied to a tool (related_tool != None)
    # and surface the tool names + IV's reasoning at the time.
    if recent_decisions:
        tool_decisions = [
            d for d in recent_decisions
            if (d.get("related_tool") or "").strip()
        ]
        if tool_decisions:
            lines.append("")
            lines.append("Recent tool activity this turn (most recent last):")
            for d in tool_decisions[-6:]:
                tool = d.get("related_tool", "?")
                trig = d.get("trigger", "?")
                rsn = (d.get("reason") or "")[:80]
                lines.append(f"  - [{trig}] {tool} — {rsn}")
    lines.extend([
        "",
        "Walk each completed todo against the recent tool activity. For each:",
        "",
        "  - The tool calls plausibly accomplished the todo → noop "
        "(advancement is real, primary's bookkeeping is honest).",
        "  - No plausible work behind the flip → inject naming the specific "
        "todo and what's missing (e.g. \"You marked X completed but no "
        "tool call shows that work happening — what evidence do you have?\").",
        "",
        "Be specific about the tool-call evidence you saw. A vague \"primary "
        "did some work\" reasoning is the failure mode this check exists to "
        "prevent. If the evidence is genuinely thin (fewer than 1-2 tool "
        "calls on the relevant subject since the last flip), that's the "
        "signal to inject. If the tool calls are unrelated to the todo "
        "content, that's also the signal to inject.",
        "",
        "TodoWrite itself does not count as evidence of the todo's work — "
        "it's just bookkeeping. Look for Read/Write/Bash/Edit/skill calls "
        "or the equivalent that actually performed the action.",
    ])
    return "\n".join(lines)


def _format_stalled_progress_block(
    threshold: int,
    active_todos: list[dict[str, Any]] | None,
    recent_decisions: list[dict[str, Any]] | None,
) -> str:
    """Plan A.6 — primary has made tool calls but no todo has advanced.

    Fires after `threshold` non-TodoWrite tool results since the last
    status change. The block lists the active (pending/in_progress)
    todos and the recent tool activity, then asks the IV to judge
    whether the primary is doing legitimate exploration or has lost
    track of its committed plan.
    """
    if threshold <= 0 or not active_todos:
        return ""
    lines = ["", "STALLED-PROGRESS CHECK (REQUIRED — do not skip):"]
    lines.append(
        f"Primary has made {threshold}+ non-TodoWrite tool calls since the "
        "last todo status change, but at least one todo is still active."
    )
    lines.append("")
    lines.append("Active todos:")
    for t in active_todos:
        status = (t.get("status") or "?").strip()
        c = (t.get("content") or "").strip()
        if not c:
            continue
        if status not in ("pending", "in_progress"):
            continue
        lines.append(f"  - [{status}] {c}")
    if recent_decisions:
        tool_decisions = [
            d for d in recent_decisions
            if (d.get("related_tool") or "").strip()
        ]
        if tool_decisions:
            lines.append("")
            lines.append("Recent tool activity (most recent last):")
            for d in tool_decisions[-6:]:
                tool = d.get("related_tool", "?")
                trig = d.get("trigger", "?")
                rsn = (d.get("reason") or "")[:80]
                lines.append(f"  - [{trig}] {tool} — {rsn}")
    lines.extend([
        "",
        "Decide:",
        "  - Tool calls are plausibly serving an active todo (research, "
        "loading context, executing the in-progress step) → noop.",
        "  - Tool calls are unrelated to any active todo (drifted onto a "
        "side quest, gone exploring beyond the plan) → inject naming the "
        "in_progress todo and asking the primary to either advance it or "
        "explain what's blocking.",
        "  - Tool calls are progress on the in_progress step but the "
        "primary is taking too long without bookkeeping → inject a "
        "lighter nudge: \"You've been working on X for a while — update "
        "the todo list when you make progress.\"",
        "",
        "A long Read/Grep/Glob sequence on files relevant to the "
        "in_progress todo is legitimate research, not a stall. Only "
        "inject when the tool calls clearly aren't advancing any active "
        "todo OR when the primary has done the work but forgot to update "
        "TodoWrite.",
    ])
    return "\n".join(lines)


def build_tool_result_summary(
    tool_name: str,
    result_preview: str,
    is_error: bool,
    todo_flips: list[dict[str, str]] | None = None,
    recent_decisions: list[dict[str, Any]] | None = None,
    stalled_progress: bool = False,
    tool_calls_since_flip_threshold: int = 0,
    active_todos: list[dict[str, Any]] | None = None,
) -> str:
    """One-line summary of a tool_result event for review.

    `todo_flips` (Plan A.5) — when TodoWrite caused an in_progress→completed
    flip, the summary appends a mark-without-evidence check.
    `stalled_progress` (Plan A.6) — when the observer's stall counter
    crossed the threshold, append a stalled-progress block asking the
    LLM to judge whether tool activity is on-plan or drifted.
    """
    extra = _format_mark_without_evidence_block(todo_flips, recent_decisions)
    if stalled_progress:
        extra += _format_stalled_progress_block(
            tool_calls_since_flip_threshold, active_todos, recent_decisions,
        )
    if not is_error:
        spilled = _format_spilled_summary(tool_name, result_preview)
        if spilled is not None:
            return spilled + extra
    label = "ERROR" if is_error else "result"
    preview = result_preview[:300] + ("..." if len(result_preview) > 300 else "")
    return (
        f"Tool {tool_name} returned {label}: {preview!r}. "
        f"Primary will see this and decide its next move."
        f"{extra}"
    )


def build_result_summary(
    stop_reason: str,
    response_text: str,
    goal_card: dict[str, Any] | None = None,
    todos: list[dict[str, Any]] | None = None,
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
    pending_block = _format_pending_todos_block(todos, on_unmet="ambient")
    if eval_block or pending_block:
        return (
            f"Turn ENDED (stop_reason={stop_reason}). The harness has already "
            f"terminated. There will be no more text or tool calls. "
            f"Final visible response: {txt!r}.{eval_block}{pending_block}"
        )
    # No goal card or todos — fall back to lighter-touch judgment.
    return (
        f"Turn ENDED (stop_reason={stop_reason}). The harness has already "
        f"terminated. Final visible response: {txt!r}. "
        f"If the response delivers a substantive answer → noop. If it's a "
        f"stub announce ('I'll X', 'Let me Y') or otherwise fails to deliver, "
        f"ambient a follow-up turn that completes the work."
    )
