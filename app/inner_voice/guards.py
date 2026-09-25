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
from pathlib import Path
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
    # Speech acts and decisions completed in this very sentence. "I need to
    # note that the config is read-only" is the note; nothing is deferred.
    # `i\s*` rather than `i ` so the contracted "I'll note ..." matches —
    # with a hard space the `'ll` branch could only ever match "I 'll".
    # The optional adverb slot catches "I should ALSO mention ...".
    r"|i\s*(?:need to|should|will|'ll|must|want to|have to)"
    r"(?:\s+(?:just|also|simply|quickly|briefly|probably|likely|however|now|"
    r"first|finally|therefore))?\s+"
    r"(?:note|mention|point out|flag|emphasi[sz]e|clarify|stress|add|say|"
    r"highlight|call out|correct|caveat|warn|reiterate|repeat|admit|"
    r"acknowledge|be clear|be honest|be up ?front|"
    # Decisions and recommendations. These resolve the question rather than
    # deferring work: "I'll leave that to you", "I need to hear which you
    # prefer" and "Let's go with option B" are answers, not stalls.
    r"recommend|suggest|leave|hear|defer|assume|go with|stick with|"
    r"proceed with)\b"
    r"|let's (?:go with|stick with|proceed with|say|assume|call)\b"
    r")",
    re.IGNORECASE,
)


# An announce phrase followed by a determiner or pronoun heads a NOUN
# phrase, not a promised action. "Going to the source, the loop appends the
# assistant message after the hook" reads as an announce to the regex above
# and is in fact a delivered sentence.
_ANNOUNCE_NOUN_FOLLOWER_RE = re.compile(
    r"(?:^|\n)\s*(?:now\s+|next,?\s+|first,?\s+)?going to\s+"
    r"(?:the|a|an|this|that|these|those|my|your|our|their|its|his|her|"
    r"it|them|him|be|where|what|which)\b",
    re.IGNORECASE,
)


# A justification or negation clause means the sentence RESOLVED rather than
# deferred. "I'll leave the config as-is since it already works" is a
# decision the primary made and explained; "I will not change that file
# because it is generated" is a refusal. Neither promises work it then
# skipped, and both matched the raw announce regex.
_ANNOUNCE_RESOLVED_RE = re.compile(
    r"\b(?:because|since|as long as|so that|rather than|instead of|unless|"
    r"although|though|given that|which is why|that is why|as-is|as is)\b"
    r"|\b(?:will|am|is|are|do|does|did|can|could|would|should|must|going to)"
    r"\s+not\b"
    r"|\b(?:won't|can't|cannot|don't|doesn't|didn't|shouldn't|wouldn't|"
    r"isn't|aren't|needn't)\b",
    re.IGNORECASE,
)


# The last line ends without closing the thought — a colon or an ellipsis is
# the strongest stall signal there is, and it outranks the resolved/noun
# checks below.
_ANNOUNCE_OPEN_END_RE = re.compile(r"(?::|\.\.\.|\u2026)\s*$")

# Content for the deterministic stall-rescue inject (fast-path + lever paths).
STALL_RESCUE_CONTENT = (
    "You ended the turn by announcing an action without doing it. Do not stop — "
    "execute the action you just described now, in this same turn, and keep going "
    "until the task is actually complete and you have delivered the result. If you "
    "are genuinely finished, state the result explicitly instead of announcing more work."
)


# The same stall on a turn nobody reads. "Deliver the result" is the wrong
# instruction there — `run_prompt_in_session` takes the harness finalizer's
# structured outcome, not the prose — so it says do the work or stop.
UNATTENDED_STALL_RESCUE_CONTENT = (
    "You ended the turn by announcing an action without doing it. Do it now, in "
    "this same turn. No human reads this session and the harness asks for the "
    "outcome itself, so do not write a report: finish the work, or end the turn "
    "if there is none left."
)

# What an unattended turn is told when it stops with nothing left to say.
# Round 874 is why the words are fixed: the observer injected "deliver the
# final report now" on the invented premise "working tree clean", and a
# healthy round was abandoned at iteration 38 with 44 minutes left.
UNATTENDED_TERMINAL_RESCUE_CONTENT = (
    "You are stopping, and this session has no human reader — the harness "
    "asks for the report itself, so there is nothing to deliver here. If "
    "there is work left that you can still finish, do it. If there is not, "
    "end the turn."
)

# The one terminal stop on an unattended turn that is always wrong: a round
# is open. The reaper keeps the branch, but a round whose author stopped
# before gating is a re-offer and another hour of the loop.
UNATTENDED_ROUND_OPEN_CONTENT = (
    "A round is open. Do not write a report — the harness asks for one. "
    "Commit what is in the worktree and call automod_gate, then "
    "automod_land or automod_abort."
)


def stall_rescue_content(*, unattended: bool, round_open: bool) -> str:
    """The words for a stall rescue, chosen by who reads the turn."""
    if unattended and round_open:
        return UNATTENDED_ROUND_OPEN_CONTENT
    if unattended:
        return UNATTENDED_STALL_RESCUE_CONTENT
    return STALL_RESCUE_CONTENT


def todo_gate_content(open_items: list[str]) -> str:
    """The nudge for a turn ending with its own todo list still open.

    Worded so a turn that is really done pays one short iteration, not a
    restated answer: the case study that motivated the old LLM version was
    "delivered, but never marked the list", and the answer is already on
    screen by then.
    """
    shown = "; ".join(f"'{c[:80]}'" for c in open_items[:5])
    more = f" (and {len(open_items) - 5} more)" if len(open_items) > 5 else ""
    return (
        f"Your todo list still shows {len(open_items)} open item(s): {shown}{more}. "
        f"If they are done, mark them completed with TodoWrite and stop — do not "
        f"restate your answer. If they are not done, keep working on them now."
    )


def failure_payload_content(tool: str) -> str:
    """The nudge for a result that returned normally but says nothing ran."""
    return (
        f"The {tool} call returned without an error, but its payload says the "
        f"work did not complete (stopped, timed out, or an empty response). Do "
        f"not treat that output as a result. Retry it narrower, do the work "
        f"directly, or say plainly that it failed."
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
    probe = "\n" + last_line
    if _ANNOUNCE_FALSE_POSITIVE_RE.search(probe):
        return False
    # An unclosed ending (colon, ellipsis) is the one signal strong enough
    # to stand on its own: the primary stopped mid-thought. Everything
    # below is about sentences that DID close, where the announce verb can
    # belong to a delivered statement.
    if _ANNOUNCE_OPEN_END_RE.search(last_line):
        return True
    if _ANNOUNCE_NOUN_FOLLOWER_RE.search(probe):
        return False
    if _ANNOUNCE_RESOLVED_RE.search(last_line):
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


@dataclass(frozen=True)
class ContextPressure:
    """How close the turn is to the CONTEXT wall, the other way to die.

    `iteration_pressure` watches `max_turns`; this watches tokens. They are
    independent, and on 2026-09-11 it was this one that killed rounds — 875
    died at iteration 50-odd of 150 with 44 minutes left, purely out of
    window.
    """

    used: int
    window: int
    fraction: float          # against the compaction wall, not the window
    critical: bool           # nudging is still worth it, but only to converge
    exhausted: bool          # nothing the observer says can be acted on


def context_pressure(
    meter: Any, *, threshold: float = 0.8, floor_tokens: int = 12_000,
) -> ContextPressure:
    """Read the shared `ContextMeter`. Unmeasured is never critical.

    Fails open in the direction that preserves the observer: a turn whose
    context position is unknown is judged exactly as it always was. The
    failure being closed is the opposite one — spending the last iteration a
    turn has on a nudge it has no room to answer, which is what happened to
    875 when the observer injected "deliver the final report now" onto a
    silent terminal iteration at 241k of 262,144 tokens.

    `floor_tokens` is the SAME number the loop drops a terminal inject at
    (`harness.context_relief.terminal_floor_tokens`). Two different floors
    would mean the observer speaking into a turn the loop has already
    decided to end, which is the disagreement this shared object exists to
    prevent.
    """
    if meter is None:
        return ContextPressure(0, 0, 0.0, False, False)
    try:
        if not meter.measured:
            return ContextPressure(0, int(meter.window), 0.0, False, False)
        used = int(meter.used)
        window = int(meter.window)
        frac = float(meter.threshold_fraction)
        headroom = int(meter.headroom)
    except Exception:  # noqa: BLE001
        return ContextPressure(0, 0, 0.0, False, False)
    return ContextPressure(
        used=used,
        window=window,
        fraction=frac,
        critical=frac >= threshold,
        exhausted=headroom < floor_tokens,
    )


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

# Cap the argument text scanned for identifiers. A `Write` carries a whole
# file body and an `Edit` carries two; letting those dominate the identifier
# set makes containment meaningless against a short sibling call.
_IDENT_SCAN_CHARS = 2000

# Tools whose repetition is only meaningful as a BYTE-IDENTICAL repeat.
#
# Near-matching exists for shell reformulation, where the same search is
# reworded until it looks new. It does not transfer to tools addressed by
# path: three Reads of unrelated files under one repo share their directory
# components and nothing else, and two shared identifiers is all a near match
# needs. On 2026-09-04 that fired 19 deterministic injects in one evening,
# 15 of them naming a path segment or an argument key as the "term the
# primary keeps chasing". Re-reading ONE file verbatim is still caught here —
# that is an exact match and needs no similarity heuristic — and a chunked
# read of the same file carries a different offset, so it is correctly not a
# repeat.
_EXACT_ONLY_TOOLS = frozenset({
    "Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Glob", "TodoWrite",
})


def _ambient_path_prefixes() -> tuple[str, ...]:
    """Path fragments every call in a session shares, so they carry no signal.

    `/home/<user>/<repo>/app/inner_voice/observer.py` contributed the
    username to every signature it appeared in. The username is 13
    characters, so it cleared the identifier length filter, and it is
    present in literally every absolute path the agent ever touches.
    """
    try:
        home = str(Path.home()).rstrip("/")
    except Exception:  # noqa: BLE001 — no home is not an error here
        return ()
    if not home or home == "/":
        return ()
    user = home.rsplit("/", 1)[-1]
    out = [home + "/", home]
    if len(user) >= 4:
        out.append(user)
    return tuple(out)


_AMBIENT_PATH_PREFIXES = _ambient_path_prefixes()


# Fragments that are ambient for a whole *kind* of turn rather than for the
# machine. The prefix list above can only remove what it can name literally;
# these are shapes.
#
# The round id is the one that keeps costing rounds. `SM_20260908_165950`
# survives `_strip_ambient` (it has underscores), reaches 18 characters, and
# so `_is_distinctive` lets it carry a near match on its own — and every call
# in an implement round mentions it. `_strip_cd_prefix` removes it from Bash
# commands only, which left it live in every other idiom: on 2026-09-11 the
# repetition guard fired 16 times on autocode turns in one day, 7 of them on
# the round id alone, through `graph_affected(root=…SM_…)` and
# `automod_gate_wait(round_id=…)`. One turn hit the deterministic cap of 5 at
# the moment its gate started.
#
# The worktree path is the second: `/lloyd-work/<round>/home/lloyd` is the
# prefix of every path a round touches, and it is not under `Path.home()` in
# the shape the prefix list matches.
# Order matters: the worktree path CONTAINS the round id, so the broader
# pattern has to run first or the narrower one eats its middle and leaves
# `/lloyd-work/ /home/lloyd` behind.
_AMBIENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/lloyd-work/[^\s'\"]*?/home/lloyd\b"),
    re.compile(r"\bSM_\d{8}_\d{6}\b"),
)


def _strip_ambient(text: str) -> str:
    """Drop the session-wide fragments before extracting identifiers.

    Patterns run BEFORE the literal prefixes: the worktree pattern contains a
    path the prefix pass would otherwise punch a hole in, leaving a fragment
    that still carries the round id.
    """
    out = text
    for rx in _AMBIENT_PATTERNS:
        out = rx.sub(" ", out)
    for frag in _AMBIENT_PATH_PREFIXES:
        out = out.replace(frag, " ")
    return out


# `cd <somewhere> && ` — the working-directory preamble, not the subject of
# the call. Every Bash call in an automod round opens with
# `cd /home/<user>/lloyd-work/SM_<id>/home/lloyd &&`, and `SM_20260908_165950`
# survives `_strip_ambient` (it has underscores), reaches 18 characters, and so
# `_is_distinctive` lets it carry a near match on its own. Five injects fired in
# round SM_20260908_165950 on five unrelated calls for exactly that reason, and
# they exhausted the observer's deterministic budget at the moment the round
# reached its gate. Every implement round works in a worktree, so this was
# structural rather than bad luck.
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s*(?:&&|;)\s*")


def _strip_cd_prefix(command: str) -> str:
    """Remove leading `cd <path> &&` hops from a shell command.

    Only for identifier extraction. `exact` keeps the whole command, because
    two calls that differ only in their `cd` are not byte-identical and should
    not be reported as though they were.
    """
    out = command
    for _ in range(4):  # `cd a && cd b && ...` — bounded, not a loop on input
        stripped = _CD_PREFIX_RE.sub("", out, count=1)
        if stripped == out:
            break
        out = stripped
    return out


# A term only reads as turn-ambient once there is more history than the
# comparison window itself — see `_ubiquitous`. One more than the default
# window, so the judgment always rests on at least one call the comparison
# is not already looking at.
_AMBIENT_MIN_HISTORY = 8

REPETITION_WINDOW = 6
REPETITION_MIN_OVERLAP = 2
REPETITION_CONTAINMENT = 0.5
REPETITION_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Path and filter operands — where a call looked, not what it hunted (#523)
# ---------------------------------------------------------------------------
#
# `_identifiers` keeps every `_`-bearing token, and a scan root is full of
# them: `_pipeline`, `trajectories`, `node_modules`, `__pycache__`, plus the
# file a call is pointed at (`test_trajectory_extraction`). So the operands
# alone cleared `min_overlap`, and two probes aimed at the same directory read
# as the same hunt. Measured over the 2026-09-07/08 session digests: 75
# firings, and the terms the message named were `_pipeline` 25, `trajectories`
# 9, `agent_mcp` 7, `read_text` 4, `test_trajectory_extraction` 4 — where the
# primary looked, inside a sentence asserting the primary keeps chasing one
# target. `_EXACT_ONLY_TOOLS` narrows this for path-addressed *tools*, but the
# probes that fire are `Bash` `grep -r` / `find` calls, whose subject is a path
# and which are not in that set.
#
# The fix is positional, not lexical. A deny list of "boring directory names"
# cannot work because the same token is a hunt in one call and a location in
# another — `kg_store` is both the table being hunted and `app/kg_store.py`,
# and `_is_distinctive` already admits both. So: classify each operand by the
# position it occupies, and let a token that BOTH calls use only as a place to
# look (or a thing to skip) carry nothing. A token either call names outside a
# path position stays eligible, which is what keeps a real hunt alive when the
# hunter also happens to `cat` the module it is chasing.
_SEARCH_COMMANDS = frozenset({"grep", "egrep", "fgrep", "rg", "ag", "find", "fd"})
_GREP_LIKE_COMMANDS = frozenset({"grep", "egrep", "fgrep", "rg", "ag"})
# Flags whose operand says what to skip or scope, not what to find. `--include`
# is here too: its operand is a filename glob, and a glob is not a hunt.
_FILTER_VALUE_FLAGS = frozenset({
    "--exclude-dir", "--exclude", "--exclude-from", "--ignore", "--ignore-file",
    "--include", "--include-dir", "-g",
    # ripgrep spells the same thing `--glob GLOB` / `-g GLOB`. Its operand is a
    # filename pattern saying what to skip, and `rg SYM ROOT --glob '!node_modules'`
    # is the shape that appeared in live traffic.
    "--glob", "--iglob",
})
# Flags whose operand is the HUNT, so it must stay eligible to carry a match
# even though it arrives after a flag rather than as the leading operand:
# `find . -name '*iv_inject*'`, `grep -e PATTERN`. Skipping classification for
# these leaves the operand untouched, which is the direction that keeps a real
# loop alive.
# `-f` is deliberately absent: `grep -f patterns.txt` names a FILE, so its
# operand is a path, which is the default classification here.
_PATTERN_VALUE_FLAGS = frozenset({
    "-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename",
    "-lname", "-ilname", "-regex", "-iregex", "-e",
})
# `grep -v`, `-vn`, `-vc`: an inverted match, so its operand is boilerplate.
_V_FLAG_RE = re.compile(r"^-[A-Za-z]*v[A-Za-z]*$")
# Quote-aware-ish shell split, built for the classifier below and used by it
# alone. Two requirements the obvious patterns each miss:
#
# * a bare word must STOP at `|;&`. Generated shell glues the separator to the
#   path with no space (`head -20 app/routers/_messages_subliminal.py; echo …`),
#   and a `\S+` fallback returns the path plus its `;` as one token — which
#   defeats `_dir_part`'s suffix test, marking the whole path as a location and
#   losing the file's own stem. That stem is the sole carrier of the
#   calibration turn's second fire (message 76).
# * a bare word must ALLOW quotes, since `--include='*'` and `grep -e "sym"`
#   mix quoting into one word. A bare-word class that excludes quote characters
#   splits `--include='*'` into two tokens, the operand counter is then off by
#   one, and the real pattern reads as the scan target instead.
#
# So: quoted run | metacharacter run | run of anything that is not whitespace or
# a separator. Not a shell parser; it only has to get operand ORDER right.
_SHELL_TOKEN_RE = re.compile(r"'[^']*'|\"[^\"]*\"|[|;&]+|[^|;&\s]+")


_FILE_SUFFIX_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]*$")


def _dir_part(token: str) -> str:
    """The directory components of a path operand, minus any file it names.

    Used only for a NON-search command's operand (`_path_operand_text`): a scan
    root and a file the call opens are different claims. Three greps over one
    tree say nothing about the same hunt — that is #523, and their whole scope is
    marked without reaching here — while `head`/`cat`/`sed` of one FILE is that
    file being re-opened, which is one target revisited. The guard's calibration
    case depends on the difference: turn 20260905_011748_iv84e4's second cluster
    reaches `app/routers/_messages_subliminal.py` as `grep -rn … <file>` (call
    70) and as `head -20 <file>` (call 76), and that file's stem is the pair's
    only shared token (`test_iv_loop_guards.py
    ::test_repetition_fires_on_the_real_loop` pins the fire at 62 and 76, and
    `test_real_turn_replayed_through_the_pretool_hook` pins that the inject
    names it). So a read marks its file's directories and leaves the stem
    eligible; a search scope marks everything it has.
    """
    bare = token.rstrip("/")
    if bare in ("", ".", ".."):
        return token
    if _FILE_SUFFIX_RE.search(bare.rsplit("/", 1)[-1]):
        return bare.rsplit("/", 1)[0] if "/" in bare else ""
    return bare


def _path_operand_text(command: str) -> str:
    """The operands of a shell command that say WHERE to look or WHAT to skip.

    Text, not a token set: `_identifiers()` runs over the result, so the same
    lowercasing and length rules apply as everywhere else in this module.

    Per pipeline stage, the command word decides what a bare operand means:

    * `grep`/`rg`/`ag` family — the first operand is the PATTERN (the hunt) and
      every later one is SCOPE: a file or directory to search, marked in full,
      its filename's stem included. A search is not about the file it reads.
    * `find`/`fd` — a bare operand is a path. Their patterns arrive after
      `-name`/`-path`/`-regex`, which are in `_PATTERN_VALUE_FLAGS` and so keep
      the operand eligible: `find . -name '*iv_inject*'` three times over is one
      pattern chased three times, not a directory being looked at.
    * anything else (`cat`, `sed`, `wc`, `pytest`, …) — an operand is a path
      when it looks like one (carries a `/`, or is a `~` hop). A bare
      `prompt_builder.py` handed to pyflakes names a module the call is about,
      and stays eligible.
    * a filter flag's operand (`--exclude-dir=X`, `grep -v X`, `-g GLOB`) is
      boilerplate in every stage.

    A token enters the returned text only when a position decides it; anything
    else is left out and stays eligible to carry a match. So an unrecognised
    shape degrades to this guard's pre-#523 behaviour for that call — it can
    cost the guard a carrier, and never grant one it did not have.
    """
    try:
        tokens = _SHELL_TOKEN_RE.findall(command or "")
    except Exception:  # noqa: BLE001 — a command that will not split is not a match
        return ""
    out: list[str] = []
    stage = ""           # command word of the current pipeline stage
    positional = 0       # non-flag operands seen in this stage
    filtering = False    # the next operand filters, per `--exclude-dir`/`grep -v`
    hunting = False      # the next operand is a pattern, per `-name`/`-e`
    for tok in tokens:
        if not tok or tok[0] in "|;&()" or tok in ("{", "}"):
            stage = ""
            positional = 0
            filtering = False
            hunting = False
            continue
        bare = tok.strip("'\"")
        if not stage:
            if tok.startswith("-"):
                continue
            stage = bare.lower().rsplit("/", 1)[-1].lstrip("$(").strip()
            continue
        # A flag written `--exclude-dir=X` carries its operand INSIDE the token,
        # so no further token belongs to it. Getting that wrong is how
        # `grep -rn --include='*' "sym" dir` came to mark `sym` — the hunted
        # term, and the sole shared identifier in
        # `test_iv_v52_review.py::test_repetition_catches_a_single_distinctive_symbol`
        # — as a filter operand, defanging the guard on exactly the case it
        # exists for.
        flag = bare.lower().split("=", 1)[0]
        inline = "=" in bare
        if flag in _FILTER_VALUE_FLAGS:
            if inline:
                out.append(bare.split("=", 1)[1])   # mark X where it stands
            else:
                filtering = True                    # mark the token that follows
            continue
        if tok.startswith("-"):
            # `find . -name GLOB` / `grep -e PATTERN`: the operand after such a
            # flag is the hunt, so it must not be read as a place to look. With
            # the value inline (`-name='*.py'`) there is no operand after it, and
            # claiming one would shift the count onto the next real path.
            if not inline and flag in _PATTERN_VALUE_FLAGS:
                hunting = True
            if stage in _GREP_LIKE_COMMANDS and _V_FLAG_RE.match(tok):
                filtering = True
            continue
        if hunting:
            positional += 1
            hunting = False
            continue
        positional += 1
        if filtering:
            # An excluded thing is not a hunted thing: mark the whole operand,
            # file-shaped or not.
            out.append(bare)
            filtering = False
            continue
        if stage in ("find", "fd") or (
            stage in _SEARCH_COMMANDS and positional > 1
        ):
            # A search's remaining operands are its SCOPE — the tree, or the one
            # file it is pointed at. Both are locations, so the whole operand is
            # marked, filename included. Dropping only the directories would
            # leave the file's stem to carry the match, and that is the second
            # measured false-fire channel of #523: three greps of
            # `tests/integration/test_trajectory_extraction.py` for three
            # different test names fired on `('test_trajectory_extraction',)`
            # alone — the file being scanned, named as the target.
            out.append(bare)
        elif "/" in bare or bare.startswith("~"):
            # Not a search: a `head`/`cat`/`sed`/`wc` operand is the file the
            # call is ABOUT, and that is a target, so only its directories are
            # locations here. See `_dir_part`.
            out.append(_dir_part(bare))
    return " ".join(out)


@dataclass(frozen=True)
class ToolCallSignature:
    """What a tool call was 'about', for repetition comparison."""

    tool: str
    exact: str                    # normalized args — identical strings are exact repeats
    idents: frozenset[str]        # what the call is ABOUT — drives matching
    preview: str                  # short human-readable form for the prompt
    # Everything the call mentions, including the parts `idents` deliberately
    # drops (the `cd <path> &&` preamble). Matching must not see these; the
    # ambient detector must. A round id reaches the ring through a `cd` hop in
    # most calls and a `W=<path>;` assignment in others, and stripping it from
    # `idents` alone left it un-ambient — present in some calls, absent from
    # others — so it survived in exactly the minority that used the other
    # idiom. That is the second false fire of round SM_20260908_165950.
    all_idents: frozenset[str] = frozenset()
    # Identifiers that sit in a path or filter-operand POSITION IN THIS CALL
    # (`#523`, see `_path_operand_text`): a scan root, a directory component, a
    # search's file scope, a `--exclude-dir` operand. A subset of `idents` — they
    # stay in `idents` so the ambient detector keeps seeing them, and so an exact
    # repeat still compares the whole command — and they cannot carry a near
    # match on their own: a pair whose every shared term is in BOTH calls'
    # `path_idents` shared only locations. The rule needs the position, not the
    # token: `P` in one call and `Q` in another are both hunts even when `P` is a
    # directory name elsewhere, and one call's `path_idents` entry can still be
    # the other call's subject — see `repetition_verdict`.
    path_idents: frozenset[str] = frozenset()


def _identifiers(text: str) -> frozenset[str]:
    return frozenset(
        w.lower()
        for w in _IDENT_RE.findall(text or "")
        if "_" in w or len(w) >= 12
    )


def tool_call_signature(tool_name: str, tool_args: Any) -> ToolCallSignature:
    """Reduce a proposed tool call to its comparable essence.

    `exact` keeps the full `key=value` rendering, so two calls are byte-equal
    only when every argument matches. `idents` is built from the argument
    VALUES alone: key names are identical on every call of a given tool, so
    counting them as shared identifiers meant `file_path` and `old_string`
    alone could carry a near match between three unrelated Edits.
    """
    path_text = ""
    if isinstance(tool_args, dict):
        items = [(k, tool_args[k]) for k in sorted(tool_args) if k != "description"]
        if tool_name == "Bash":
            raw = str(tool_args.get("command") or "")
            full_text = raw
            value_text = _strip_cd_prefix(raw)
            path_text = _path_operand_text(value_text)
        else:
            # Sorted so key order can't make two identical calls look different.
            # `path_text` stays empty here on purpose: every path-addressed
            # tool that could need it is already in `_EXACT_ONLY_TOOLS`, and
            # the near-matching tools addressed by a path argument (the MCP
            # `Grep` tool's `path`) are a separate channel — see the finding
            # appended to #523. Widening it is not this change's to take.
            raw = " ".join(f"{k}={v!r}" for k, v in items)
            value_text = full_text = " ".join(str(v) for _, v in items)
    else:
        raw = str(tool_args or "")
        value_text = full_text = raw
    normalized = " ".join(raw.split())
    preview = normalized if len(normalized) <= 160 else normalized[:157] + "..."
    ident_src = " ".join(value_text.split())[:_IDENT_SCAN_CHARS]
    idents = _identifiers(_strip_ambient(ident_src))
    return ToolCallSignature(
        tool=tool_name or "",
        exact=normalized,
        idents=idents,
        preview=preview,
        all_idents=_identifiers(_strip_ambient(
            " ".join(full_text.split())[:_IDENT_SCAN_CHARS]
        )),
        # The AND is what keeps `path_idents ⊆ idents` true unconditionally —
        # a marked operand that the length or identifier rules reject is not in
        # either set, and a token cut off by the scan cap above must not count
        # as classified.
        path_idents=_identifiers(_strip_ambient(path_text)) & idents,
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


def ubiquitous_identifiers(sigs: list[ToolCallSignature]) -> frozenset[str]:
    """Identifiers carried by every call in the compared window.

    `_strip_ambient` removes the fragments that are ambient for the whole
    *machine* — the home directory, the username. It cannot remove the ones
    that are ambient for a single *turn*, because it does not know what the
    turn is doing. An automod round is the case that matters: every Bash call
    is prefixed `cd /home/<user>/lloyd-work/SM_<id>/home/lloyd &&`, and
    `SM_20260908_165950` survives stripping (it has underscores), is present
    in literally every command, and is 18 characters — so `_is_distinctive`
    reads it as a specific symbol somebody is chasing and lets it carry a
    match on its own.

    That is what fired five times in round SM_20260908_165950, on five
    unrelated calls, and exhausted the observer's deterministic budget at the
    moment the round reached its gate. Every implement round works in a
    worktree, so the failure was structural, not incidental.

    A term present in *all* of the compared calls discriminates between none
    of them, which is the same argument the frequency ordering below already
    makes for *naming* — this applies it to matching, where it decides
    whether the guard fires at all.

    Measured over the WHOLE ring rather than the comparison window, and only
    once the ring is longer than the window. Over the window alone this
    reasoning eats its own guard: a primary hunting `iv_inject_queue` six
    times running shares that symbol across every compared call, so the very
    repetition the guard exists to catch would read as ambient and be
    stripped. That is not hypothetical — it silenced both fires of the real
    28-call replay in `test_iv_loop_guards`. Across a longer history the two
    separate cleanly: a chased symbol appears in a burst, while the worktree
    id is in the healthy exploration calls too.

    Public because the caller has a longer memory than the guard does. After
    a fire the observer compares only the calls made SINCE it last spoke
    (`repetition_baseline`), so the slice handed to `repetition_verdict` can
    be three calls long while `state.recent_tool_calls` still holds sixteen.
    Judged on the slice alone the worktree id looks distinctive again, which
    is exactly how the second false fire of round SM_20260908_165950 survived
    the first cut of this fix. The observer passes the full ring here and the
    truncated slice to the verdict.
    """
    if len(sigs) < _AMBIENT_MIN_HISTORY:
        return frozenset()
    # A signature with NO identifiers at all empties the intersection and so
    # declares nothing ambient — which is backwards: one bare `git status` in
    # the ring restored the round id to distinctive and let it carry a match
    # again. A call that mentions nothing is evidence about nothing, so it
    # abstains rather than voting.
    contributing = [s for s in sigs if s.all_idents]
    if len(contributing) < _AMBIENT_MIN_HISTORY:
        return frozenset()
    common = set(contributing[0].all_idents)
    for s in contributing[1:]:
        common &= s.all_idents
        if not common:
            break
    return frozenset(common)


# Opaque identifiers — specific, and not a source symbol.
#
# Both of `_is_distinctive`'s branches admit them. A git short-SHA is
# `^[0-9a-f]{7,40}$` by construction, so a 16-char id clears `len >= 16` exactly;
# and `_IDENT_RE` cannot START inside digits, so a session key or a timestamped
# id splices into whatever underscore-rich fragment follows its digits —
# `20260912_043416_iv9aa2` yields `_043416_iv9aa2`, three underscores, therefore
# distinctive. Either one then carries a near match alone, and inspecting ONE
# commit, or one session digest, from three ordinary angles reads as a loop:
# `git show <sha> -- FILE`, `git show <sha> --stat`,
# `git log -1 --format=%B <sha>`. Live on 2026-09-12 that fired fourteen
# deterministic repetition injects, the last naming
# `_043416_iv9aa2, a9a5bdae37eff3d4` — neither term a search target (#1026).
#
# The rejection is by token CLASS, never by read shape. Three reads of one file
# must keep firing on that file's stem — `test_iv_loop_guards.py
# ::test_repetition_ignores_three_greps_sharing_only_a_filename` pins it, because
# that is the shape calibration message 76 has — and the pinned stem
# (`test_trajectory_extraction`) carries no digit run, so it is untouched here.
# What goes is an id's right to stand alone as a subject. A match with a second,
# genuine carrier still fires: two shared non-path terms clear `min_overlap`
# without consulting this function at all.
_HEX_ID_RE = re.compile(r"^[0-9a-f]{7,40}$")
_LONG_DIGIT_RUN_RE = re.compile(r"\d{6,}")


def _is_distinctive(term: str) -> bool:
    """True for an identifier specific enough to carry a match on its own.

    `min_overlap = 2` was calibrated when ambient path components padded every
    identifier set — the username alone was a free second term. With those
    stripped, a genuine hunt for one symbol (`zzq_phantom_handle_v3`,
    `iv_inject_queue`) shares exactly one identifier and would never reach the
    threshold.

    Segment count separates the two cases. A name with three or more
    underscore-joined parts is a specific symbol somebody is chasing; a
    two-part name (`inner_voice`, `file_path`, `observer_prompt`) is a module
    or an argument key that half the calls in a session mention in passing.

    Shape alone, though, is only evidence that a token is *specific*, and an
    opaque id is as specific as a symbol while naming no symbol at all. So a
    hex id (`a9a5bdae37eff3d4`) and anything carrying a run of 6+ digits
    (`_043416_iv9aa2`, `sm_20260908_165950`) are refused the solo-carry before
    the shape test runs — see the two regexes above. This narrows the
    solo-carry branch only; it does not touch `min_overlap`, so a pair sharing
    the same id PLUS a real symbol still matches and still names both.
    """
    if _HEX_ID_RE.match(term) or _LONG_DIGIT_RUN_RE.search(term):
        return False
    return term.count("_") >= 2 or len(term) >= 16


# Tools whose repetition IS the protocol. Polling one of these four times is
# not a stuck model, it is the documented way to use them: `automod_gate_wait`
# is the poll a detached gate is designed around, and a `browser_wait` or an
# `autoresearch_status` says nothing until the thing it watches has moved.
# Round 874's observer read four silent `automod_gate_wait` calls as a loop.
#
# Configurable so a new polling tool does not need a code change, but the
# default is the four that exist.
REPETITION_EXEMPT_TOOLS: frozenset[str] = frozenset({
    "automod_gate_wait", "automod_status", "browser_wait", "graph_status",
    "autoresearch_status",
})


def _is_exempt(tool: str, exempt: frozenset[str]) -> bool:
    """Bare or namespaced (`mcp__lloyd-mcp__automod_gate_wait`)."""
    if not tool:
        return False
    return tool.rsplit("__", 1)[-1] in exempt


def repetition_verdict(
    recent: list[ToolCallSignature],
    *,
    window: int = REPETITION_WINDOW,
    min_overlap: int = REPETITION_MIN_OVERLAP,
    containment: float = REPETITION_CONTAINMENT,
    threshold: int = REPETITION_THRESHOLD,
    ambient: frozenset[str] | None = None,
    exempt_tools: frozenset[str] | None = None,
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

    A near match additionally needs at least one shared identifier that is a
    SUBJECT and not a place to look: a term both calls used only as a scan root,
    a directory component, a search's file scope or a filter operand cannot carry
    it (#523). See `_path_operand_text`. The test is per pair, on the pair's own
    two signatures, because the claim being judged is about the two calls being
    compared — and it asks whether EITHER call aimed at the term, not whether
    both did, because a hunt (`grep -rn SYM …`) and a read (`head -20 SYM_FILE`)
    of the same thing are one target pursued two ways. Calibration turn
    20260905_011748_iv84e4 depends on exactly that asymmetry at message 76.
    """
    if len(recent) < 2:
        return None
    current = recent[-1]
    # Polling is the protocol for a handful of tools — see
    # REPETITION_EXEMPT_TOOLS. Checked before anything else, so neither the
    # exact nor the near signal can fire on them.
    if _is_exempt(current.tool, REPETITION_EXEMPT_TOOLS if exempt_tools is None
                  else exempt_tools):
        return None
    prior = recent[-(window + 1):-1]
    matches: list[ToolCallSignature] = []
    exact = False
    shared: set[str] = set()
    # Turn-ambient identifiers carry no signal and must not carry a match.
    # `ambient` from the caller when it has more history than `recent` holds.
    if ambient is None:
        ambient = ubiquitous_identifiers(recent)
    cur_idents = current.idents - ambient
    # What the current call AIMS at, as opposed to where it looked or what it
    # skipped. Pairwise eligibility against each prior is applied below; a
    # carrier is also the only thing this verdict is allowed to name.
    cur_carriers = cur_idents - current.path_idents
    # Path-addressed tools compare by exact repeat only — see _EXACT_ONLY_TOOLS.
    near_allowed = current.tool not in _EXACT_ONLY_TOOLS
    for p in prior:
        if p.tool != current.tool:
            continue
        p_carriers = (p.idents - ambient) - p.path_idents
        if current.exact and p.exact == current.exact:
            matches.append(p)
            exact = True
            # A verbatim repeat names the hunt, not the location it aimed at
            # (clause 2 of #523 admits no path operand into `shared_terms`,
            # exact included). When every identifier is a location there is
            # nothing to name, and the inject quotes the command instead.
            shared |= cur_carriers
            continue
        if not near_allowed:
            continue
        p_idents = p.idents - ambient
        overlap = cur_idents & p_idents
        if not overlap:
            continue
        # Eligible when EITHER call aimed at the term; only a term that BOTH
        # calls used as a location is disqualified. Requiring both to aim
        # (intersecting the two carrier sets) silences the calibration turn:
        # message 76 reaches `app/routers/_messages_subliminal.py` with
        # `head -20`, a read whose subject is that file, while calls 66/70 name
        # it as a grep scope, and its stem is the pair's only shared token —
        # `test_repetition_fires_on_the_real_loop` expects a fire at 76 and
        # `test_operand_classification_survives_two_shell_shapes` pins the same
        # shape. #523's false fires survive the widening because they are
        # symmetric: `_pipeline` in three greps is scope in all three, so it sits
        # in no carrier set and no pair finds it — while a real hunt is carried in
        # at least one of the pair by construction, since a hunt puts its pattern
        # in operand position 1. Empty carriers therefore means every shared
        # identifier was a location — a scan root, a directory component, a
        # search's file scope, or a filter operand — in both calls of the pair.
        carriers = overlap & (cur_carriers | p_carriers)
        if not carriers:
            continue
        # Either several shared identifiers, or one distinctive enough to
        # stand alone. See `_is_distinctive`.
        enough = len(carriers) >= min_overlap or any(
            _is_distinctive(t) for t in carriers
        )
        if enough and _containment(cur_idents, p_idents) >= containment:
            matches.append(p)
            shared |= carriers
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
    that it is repeating — it believes each reformulation is a new query.

    It says NOTHING about the results, because it cannot know. The guard sees
    `ToolCallSignature` only: tool name, normalized arguments, identifiers
    pulled out of argument VALUES. No field carries result content and result
    text is never compared. Until 2026-09-07 this sentence read "and the result
    has not changed" — an assertion of something never observed. Backlog #393
    measured the 2026-09-05 sessions: 5 fires, and a real result-level check
    (token-set Jaccard > 0.85 against every earlier result) found ZERO
    unchanged-result repeats in two of the three sessions — 0 near-duplicates in
    66 calls, 0 in 103. The guard was matching command shape (repeated
    `grep -n … <symbol>` probes over *different* symbols) and reporting it as
    repeated results. Two reasons that is worth code rather than prose: an
    inject that asserts the unobserved trains the primary to discount injects,
    and #83 Stage 2 promotes injected corrections into skills, so a false fire
    becomes a false skill.

    The loop-breaking instruction is the point of the message, so it survives —
    reframed as guidance the primary can check against output it can see and the
    guard cannot. Earning the "the result has not changed" sentence back means
    putting result identity on the signature (a content hash or a coarse token
    sketch) and requiring agreement before claiming it — #393 branch 1.
    `tests/integration/test_iv_repetition_wording.py`
    ::test_no_result_claim_without_a_result_field detects that landing and lifts
    the ban.
    """
    n = v.repeats + 1
    if v.shared_terms:
        target = ", ".join(v.shared_terms[:6])
    elif v.previews:
        target = repr(v.previews[0][:120])
    else:
        target = "the same target"
    what = (
        f"run the same call {n} times"
        if v.exact
        else f"issued {n} near-identical queries"
    )
    return (
        f"Stop: you have {what} for {target}. This guard compares the calls you "
        f"issue, not what they returned, so look at those outputs yourself: if "
        f"they came back empty or the same, that is the ANSWER, not a failed "
        f"query — do not rewrite the filter a further time. If you are looking "
        f"for a symbol that may simply not exist, one scoped check settles it; "
        f"if it is not there, say so and move on. State what you have "
        f"established so far and continue to the deliverable the user actually "
        f"asked for."
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


# ---------------------------------------------------------------------------
# A tool call written as prose — a capability fault, not a nudge (IV plan R2)
# ---------------------------------------------------------------------------
#
# When the tool pool is empty the model still knows its tools from the system
# prompt and writes the call as text: `{"name": "Bash", "input": {...}}` (an
# Anthropic shape that appears nowhere in this repo), a `<tool_call>` block, or
# `<function=...>`. On 2026-09-06 the observer saw exactly that three times and
# injected "run it for real" three times; more text cannot supply a missing
# capability. So this is detected, logged and raised to a person, never
# injected. CLAUDE.md, "An empty tool pool is the worst failure in the system".
_PROSE_TOOL_CALL_RE = re.compile(
    r"\{\s*\"name\"\s*:\s*\"[A-Za-z_][\w.-]*\"\s*,\s*\"(?:input|arguments|parameters)\"\s*:"
    r"|<tool_call>\s*\{"
    r"|<function=[A-Za-z_][\w.-]*>",
)


def looks_like_prose_tool_call(text: str) -> bool:
    """True iff a text-only iteration carries a tool call written as text."""
    return bool(_PROSE_TOOL_CALL_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Text the goal extractor must not read (IV plan R2)
# ---------------------------------------------------------------------------
#
# 2026-08-30, session 20260830_182633_ive386: the user asked for YouTube
# highlights, a notification landed in the same turn, the goal card was built
# from the notification, and the observer injected three times that the
# transcript was "out of scope" and then cancelled the turn the user asked
# for. The card is about the USER's request; system-injected blocks are not it.
_INJECTED_BLOCK_RE = re.compile(
    r"<(task_notification|diagnostics_notification|system-reminder|ide_state|"
    r"background_task|notification|context|goal_card)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def strip_injected_blocks(text: str) -> str:
    """The user's own words: harness- and system-injected blocks removed."""
    return _INJECTED_BLOCK_RE.sub("", text or "").strip()
