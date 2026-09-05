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
        # Fast-path rows are bookkeeping, not judgment: a deterministic noop
        # means nobody looked. Counting them as "the observer chose silence"
        # made this walk stop on the first `pretool` or `assistant_message`
        # row, and with pretool_llm_enabled false there is always one of each
        # between any two tool_result injects — so the rule below could never
        # reach a prior inject. Only LLM-judged rows clear the suppressor.
        if d.get("fast_path"):
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


# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------
#
# Turn 20260905_011748_iv84e4: the primary ran six reformulations of one
# search — "who consumes iv_inject_queue / iv_cancel_requested outside the IV
# package" — over ~3 minutes. The answer (nobody) was correct and stable from
# the first attempt; the primary read the empty result as a broken query and
# kept rewriting the filter, then escalated to two `find /` scans and an
# unbounded `grep -rn` that hit the 120s Bash timeout.
#
# The observer could not see any of it. `build_tool_result_summary` renders
# the tool name and 300 chars of the RESULT; the command never reaches the
# prompt, and a loop is only visible in the arguments. So this is done
# deterministically here instead of asked of the LLM.
#
# Shell reformulation defeats naive string similarity — the six commands
# shared little literal text. What they shared was the identifiers being
# searched for. Tokens carrying an underscore (or unusually long) are code
# identifiers; bare English words are echo labels and shell noise. Comparing
# identifier sets by containment, calibrated on that turn, first fires on the
# 4th near-duplicate and stays silent through the 15 healthy exploration calls
# that preceded it.

# Tokens that look like code identifiers rather than prose: `iv_inject_queue`,
# `_messages_subliminal`, `build_subliminal_context`. Length 12 catches
# camel/flat names with no underscore (`claudesdkclient`).
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")

REPETITION_WINDOW = 6
REPETITION_MIN_OVERLAP = 2
REPETITION_CONTAINMENT = 0.5
REPETITION_THRESHOLD = 2


@dataclass(frozen=True)
class ToolCallSignature:
    """What a tool call was 'about', for repetition comparison."""

    tool: str
    exact: str                    # normalized args — identical strings are exact repeats
    idents: frozenset[str]        # code identifiers mentioned
    preview: str                  # short human-readable form for the prompt


def _identifiers(text: str) -> frozenset[str]:
    return frozenset(
        w.lower()
        for w in _IDENT_RE.findall(text or "")
        if "_" in w or len(w) >= 12
    )


def tool_call_signature(tool_name: str, tool_args: Any) -> ToolCallSignature:
    """Reduce a proposed tool call to its comparable essence."""
    if isinstance(tool_args, dict):
        if tool_name == "Bash":
            raw = str(tool_args.get("command") or "")
        else:
            # Sorted so key order can't make two identical calls look different.
            raw = " ".join(
                f"{k}={tool_args[k]!r}" for k in sorted(tool_args) if k != "description"
            )
    else:
        raw = str(tool_args or "")
    normalized = " ".join(raw.split())
    preview = normalized if len(normalized) <= 160 else normalized[:157] + "..."
    return ToolCallSignature(
        tool=tool_name or "",
        exact=normalized,
        idents=_identifiers(normalized),
        preview=preview,
    )


@dataclass(frozen=True)
class RepetitionVerdict:
    """The current call restates work already done this turn."""

    repeats: int                      # how many of the recent calls it matches
    exact: bool                       # at least one match was byte-identical
    shared_terms: tuple[str, ...]     # identifiers common to the matches
    previews: tuple[str, ...]         # the matching earlier calls


def _containment(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def repetition_verdict(
    recent: list[ToolCallSignature],
    *,
    window: int = REPETITION_WINDOW,
    min_overlap: int = REPETITION_MIN_OVERLAP,
    containment: float = REPETITION_CONTAINMENT,
    threshold: int = REPETITION_THRESHOLD,
) -> RepetitionVerdict | None:
    """Judge whether the LAST entry in `recent` re-runs earlier work.

    `recent` is oldest-first and includes the call being judged. Returns None
    unless the call matches at least `threshold` of the preceding `window`
    calls, so a single follow-up refinement — normal, healthy narrowing —
    never trips it. Two signals, either sufficient for a match:

    * exact — same tool, byte-identical normalized args. Re-reading one file
      or re-running one command verbatim needs no similarity heuristic.
    * near  — same tool, >= `min_overlap` shared identifiers AND containment
      >= `containment`. Containment rather than Jaccard because a command
      wrapped in extra `echo` labels should still match the bare one.
    """
    if len(recent) < 2:
        return None
    current = recent[-1]
    prior = recent[-(window + 1):-1]
    matches: list[ToolCallSignature] = []
    exact = False
    shared: set[str] = set()
    for p in prior:
        if p.tool != current.tool:
            continue
        if current.exact and p.exact == current.exact:
            matches.append(p)
            exact = True
            shared |= current.idents
            continue
        overlap = current.idents & p.idents
        if len(overlap) >= min_overlap and _containment(current.idents, p.idents) >= containment:
            matches.append(p)
            shared |= overlap
    if len(matches) < threshold:
        return None
    # Rarest first. A term carried by every recent call is ambient — the `cd
    # /home/<user>/<repo>` prefix, the username in a path — and says nothing
    # about what is being repeated. Live turn 20260905_020747_ivfe5f named
    # "alansrobotlab" ahead of "zzq_phantom_handle_v3", which is exactly
    # backwards for a message whose job is to tell the primary what it keeps
    # chasing. Alphabetical within a tier, so the text is deterministic.
    frequency = {
        term: sum(1 for sig in prior + [current] if term in sig.idents)
        for term in shared
    }
    ordered = tuple(sorted(shared, key=lambda t: (frequency[t], t)))
    return RepetitionVerdict(
        repeats=len(matches),
        exact=exact,
        shared_terms=ordered,
        previews=tuple(m.preview for m in matches),
    )


def repetition_inject_content(v: RepetitionVerdict) -> str:
    """The nudge text for a detected repetition.

    Names the shared terms, because the primary's failure mode is not knowing
    that it is repeating — it believes each reformulation is a new query. The
    key instruction is that a stable empty result is an ANSWER; treating it as
    a broken query is what drives the loop.
    """
    terms = ", ".join(v.shared_terms[:6]) or "the same target"
    kind = "the same call" if v.exact else "variations of the same search"
    return (
        f"Stop: you have now run {v.repeats + 1} {kind} for {terms}, and the result "
        f"has not changed. A stable empty or unchanged result is the ANSWER, not a "
        f"failed query — do not rewrite the filter again. If you are looking for a "
        f"symbol that may simply not exist, one scoped check settles it; if it is "
        f"not there, say so and move on. State what you have established so far and "
        f"continue to the deliverable the user actually asked for."
    )


# ---------------------------------------------------------------------------
# Inject pacing
# ---------------------------------------------------------------------------


def iterations_since_last_inject(prior_decisions: list[dict[str, Any]]) -> int | None:
    """Primary iterations completed since the observer's last inject.

    An `assistant_message` decision marks one completed primary iteration.
    Returns None when no inject has happened yet this turn.
    """
    seen_inject = False
    iterations = 0
    for d in reversed(prior_decisions):
        if d.get("action") == "inject":
            seen_inject = True
            break
        if d.get("trigger") == "assistant_message":
            iterations += 1
    return iterations if seen_inject else None


def inject_on_cooldown(
    prior_decisions: list[dict[str, Any]], *, cooldown_iterations: int,
) -> bool:
    """True iff a discretionary inject is too soon after the previous one.

    `suppress_consecutive_inject` enforces a one-iteration gap and is defeated
    the moment any decision lands in between. On turn 20260905_011748_iv84e4
    the observer injected at 18:24:52 and again at 18:25:14 — 22 seconds and
    three iterations later — spending two thirds of its budget before the
    first nudge could be judged, and leaving only a cancel when it mattered.
    A budget of 3 is worth little if it can be spent in 90 seconds.
    """
    if cooldown_iterations <= 0:
        return False
    since = iterations_since_last_inject(prior_decisions)
    if since is None:
        return False
    return since < cooldown_iterations


# ---------------------------------------------------------------------------
# Failure payloads inside successful results
# ---------------------------------------------------------------------------

# A tool that returns normally but whose payload reports the work failed.
# `Task` is the one that matters: a subagent that exhausts max_turns returns
# `{"response": "\n[stopped: max_turns]", ...}` — 300 bytes, is_error False —
# so the 1-in-5 sampler skipped it and the observer never learned that four
# minutes and 28 tool calls had produced nothing.
_FAILURE_PAYLOAD_RE = re.compile(
    r"\[stopped:\s*(?:max_turns|max turns|error|cancelled|canceled|timeout)"
    r"|\bcommand timed out after\b"
    r"|\"response\"\s*:\s*\"\s*\\n?\s*\"",
    re.IGNORECASE,
)


def looks_like_failure_payload(content: str) -> bool:
    """True iff a non-error result body reports that the work did not happen.

    Only inspects the head: these markers appear in the returned envelope,
    and scanning a 20 KB payload for them on every tool result is waste.
    """
    return bool(_FAILURE_PAYLOAD_RE.search((content or "")[:600]))
