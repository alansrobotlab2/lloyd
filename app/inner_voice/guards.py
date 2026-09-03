"""Inner Voice — deterministic guards, as pure functions.

Everything in this module was previously inline in `observer.py`'s
`install_observer` closure, which made it untestable without building a
whole ObserverState and firing hooks. The judgment itself has not
changed shape; it has been lifted out so each rule can be exercised
directly and so the closure reads as dispatch rather than policy.

The design rule stays what it always was: **new judgment starts in the
vault prompt.** A rule only earns a place here after the prompt has
demonstrably failed at it and the failure is cheap to detect and
expensive to miss. Every function below traces to an observed
production failure; see `architecture/inner-voice.md` for the history.

All functions are pure: they take a decision plus context and return a
new action label (or a verdict), never touching ObserverState, the
event log, or the database. The caller applies the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------

# A text-only iteration whose text only ANNOUNCES a next action without
# dispatching it — the classic "stop at a colon" / "Let me check …" stall.
# Anchored so the announce verb is the LAST thing in the message (the model
# promised an action and then stopped), or the whole thing ends on a colon.
_STUB_ANNOUNCE_RE = re.compile(
    r"(?:"
    r":\s*$"                                   # ends on a colon — "announce then nothing"
    r"|(?:^|\n)\s*(?:let me|let's|i'll|i will|i'm going to|i am going to|"
    r"now i'll|now let me|next,?\s+i'll|first,?\s+i'll|i need to|i should|"
    r"going to|let me go ahead and)\b[^\n]*[.:]?\s*$"   # last line is a bare announce
    r")",
    re.IGNORECASE,
)

# Phrases that LOOK like an announce under `_STUB_ANNOUNCE_RE` but are
# actually the closing courtesy of a DELIVERED answer, or a speech act
# completed within the sentence itself ("I need to note that X" — the
# noting happens right there; nothing is deferred).
#
# Without this exclusion the stall fast-path fires an inject that
# bypasses the intervention budget AND bypasses the consecutive-inject
# suppressor, so a primary that habitually signs off with "Let me know
# if you need anything else!" would be re-prompted every iteration until
# max_turns. Verified against the live regex: every phrase below matched
# `_STUB_ANNOUNCE_RE` before this list existed.
_ANNOUNCE_FALSE_POSITIVE_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    # Offers to keep helping — the turn is over, not stalled.
    r"(?:please\s+)?let (?:me|us) know\b"
    r"|i(?:'ll| will| would|'d)? be (?:happy|glad|available)\b"
    r"|let me be (?:clear|specific|precise)\b"
    # Speech acts completed in this very sentence. "I need to note that
    # the config is read-only" is the note; nothing is being deferred.
    r"|i (?:need to|should|will|'ll|must|want to|have to) "
    r"(?:note|mention|point out|flag|emphasi[sz]e|clarify|stress|add|say|"
    r"highlight|call out|correct|caveat|warn|reiterate|repeat|admit|"
    r"acknowledge|be clear|be honest|be up ?front)\b"
    r")",
    re.IGNORECASE,
)

# Content for the deterministic stall-rescue inject (fast-path + lever paths).
STALL_RESCUE_CONTENT = (
    "You ended the turn by announcing an action without doing it. Do not stop — "
    "execute the action you just described now, in this same turn, and keep going "
    "until the task is actually complete and you have delivered the result. If you "
    "are genuinely finished, state the result explicitly instead of announcing more work."
)


def is_terminal_stall(text: str) -> bool:
    """True iff `text` is a text-only iteration that announces work and stops.

    Two-stage: the announce regex proposes, the false-positive regex
    disposes. A sign-off ("Let me know if you need anything else!") and a
    within-sentence speech act ("I should mention one caveat: …") are
    both delivered answers, not stalls.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if not _STUB_ANNOUNCE_RE.search(stripped):
        return False
    # Check only the final non-empty line — that's what the announce
    # regex anchored on, and it's where a sign-off lives.
    last_line = stripped.rsplit("\n", 1)[-1].strip()
    if _ANNOUNCE_FALSE_POSITIVE_RE.search("\n" + last_line):
        return False
    return True


# ---------------------------------------------------------------------------
# Cancel-for-completion
# ---------------------------------------------------------------------------

# Reason-text patterns that look like a "task complete, stopping early"
# cancel — the IV deciding the primary is done and force-stopping to save
# effort. The harness terminates naturally on a text-only iteration, so
# this cancel only ever adds a red breadcrumb to a successful turn.
COMPLETION_REASON_PATTERN = re.compile(
    r"\b(complete|completed|done|criteria met|all met|success criteria"
    r"|stopping early|stop early|avoid padding|no more (?:work|tools)"
    r"|nothing more)\b",
    re.IGNORECASE,
)


def cancel_for_completion_verdict(
    *, action: str, reason: str, has_pending_tools: bool, interventions_used: int,
) -> str | None:
    """Return a downgrade label for a cancel-for-completion, else None.

    Cancel-for-completion is only allowed through once the observer has
    already intervened this turn — i.e. it is escalating from injects the
    primary ignored, which is the documented escalation path.

    * pending tool calls  → `noop_cancel_with_pending_tools` (work in flight)
    * no pending tools    → `acknowledge_complete` (a positive ack; the
      harness terminates on its own next iteration)
    """
    if action != "cancel":
        return None
    if interventions_used != 0:
        return None
    if not COMPLETION_REASON_PATTERN.search(reason or ""):
        return None
    return "noop_cancel_with_pending_tools" if has_pending_tools else "acknowledge_complete"


# ---------------------------------------------------------------------------
# Consecutive-inject suppression
# ---------------------------------------------------------------------------

# Triggers whose injects the primary actually gets an iteration to read
# before the next judgment on the same trigger. `pretool` and
# `tool_result` both fire inside a single dispatch batch, so several can
# land in well under a second with no model turn between them — those are
# counted together as one "mid-work" stream.
_MIDWORK_TRIGGERS = frozenset({"pretool", "tool_result", "assistant_message"})


def suppress_consecutive_inject(
    *, action: str, prior_decisions: list[dict[str, Any]], is_terminal: bool,
) -> bool:
    """True iff this inject should be downgraded to a guard noop.

    The rationale is "give the primary one iteration to act on the last
    nudge." That holds only mid-work. On a terminal text-only iteration
    the inject is the ONLY thing keeping the loop alive, so suppressing
    there guarantees the turn ends with work undone.

    Unlike the v4 rule, this looks across ALL mid-work triggers rather
    than only `assistant_message`. In production a drift was nudged at
    `pretool`, again at `tool_result`, and again at `pretool` inside a
    single dispatch batch — three injects in 20 seconds, none of which
    the primary had a chance to read — and then escalated to `cancel` on
    the strength of "three injects ignored." Counting only same-trigger
    pairs made those look like independent evidence.
    """
    if action != "inject":
        return False
    if is_terminal:
        return False
    for d in reversed(prior_decisions):
        trig = d.get("trigger")
        if trig not in _MIDWORK_TRIGGERS:
            continue
        # The most recent mid-work decision decides. An intervening noop
        # means the observer looked and chose silence — not a landed nudge.
        return d.get("action") == "inject"
    return False


def injects_primary_has_seen(prior_decisions: list[dict[str, Any]]) -> int:
    """Count injects the primary plausibly READ, for escalation decisions.

    An inject only reaches the primary once it completes another
    iteration, which is marked by an `assistant_message` decision landing
    after it. Injects fired inside one dispatch batch with no
    `assistant_message` between them collapse to one — the primary saw
    the batch as a single block of user messages.
    """
    seen = 0
    pending_inject = False
    for d in prior_decisions:
        action = d.get("action")
        if action == "inject":
            pending_inject = True
        elif d.get("trigger") == "assistant_message" and pending_inject:
            seen += 1
            pending_inject = False
    return seen


# ---------------------------------------------------------------------------
# Result-trigger downgrades
# ---------------------------------------------------------------------------


def result_trigger_downgrade(
    *, action: str, has_ambient_channel: bool, has_content: bool,
) -> tuple[str, str | None]:
    """Map a lever chosen at the `result` event to what can still happen.

    The harness has already emitted its terminal event by then: there is
    no further iteration to read an inject, nothing left to cancel, and
    nobody waiting on a clarify. Returns `(action, note)` where a
    non-None note explains the rewrite for the persisted row.
    """
    if action == "inject":
        if has_ambient_channel and has_content:
            return ("ambient", "inject on result → ambient (no iteration left to read it)")
        return ("noop_inject_on_result", None)
    if action == "cancel":
        return ("noop_cancel_on_result", None)
    if action == "clarify":
        return ("noop_clarify_on_result", None)
    return (action, None)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def budget_exhausted(
    *, action: str, bypass_budget: bool, interventions_used: int, budget: int,
) -> bool:
    """True iff this action must be downgraded for exceeding the budget.

    `cancel` is exempt: it ends the turn, so rationing it would block
    recovery from "the primary keeps ignoring my injects". Stall-rescue
    injects set `bypass_budget` — they prevent a premature turn end
    rather than nagging, and `max_turns` is the real backstop.
    """
    if action == "cancel":
        return False
    if bypass_budget:
        return False
    return interventions_used >= budget


# ---------------------------------------------------------------------------
# Iteration pressure (deterministic trigger)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationPressure:
    """How close the turn is to the harness's `max_turns` wall."""

    iteration: int
    max_turns: int
    fraction: float
    critical: bool


def iteration_pressure(
    iteration: int, max_turns: int, *, threshold: float = 0.8,
) -> IterationPressure:
    """Measure progress toward `max_turns`.

    A turn that hits `max_turns` dies mid-work with no terminal
    assistant_message, so the observer's usual last-chance inject never
    fires — the only recourse left is an ambient repair after the fact.
    Two of the six ambients in the first production window were exactly
    that ("turn hit max_turns mid-diagnosis"). Detecting the approach is
    free and lets the observer nudge the primary to converge while it
    still has iterations left.
    """
    if max_turns <= 0:
        return IterationPressure(iteration, max_turns, 0.0, False)
    frac = iteration / max_turns
    return IterationPressure(iteration, max_turns, frac, frac >= threshold)
