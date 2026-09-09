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


def _strip_ambient(text: str) -> str:
    """Drop the session-wide path fragments before extracting identifiers."""
    out = text
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
    if isinstance(tool_args, dict):
        items = [(k, tool_args[k]) for k in sorted(tool_args) if k != "description"]
        if tool_name == "Bash":
            raw = str(tool_args.get("command") or "")
            full_text = raw
            value_text = _strip_cd_prefix(raw)
        else:
            # Sorted so key order can't make two identical calls look different.
            raw = " ".join(f"{k}={v!r}" for k, v in items)
            value_text = full_text = " ".join(str(v) for _, v in items)
    else:
        raw = str(tool_args or "")
        value_text = full_text = raw
    normalized = " ".join(raw.split())
    preview = normalized if len(normalized) <= 160 else normalized[:157] + "..."
    ident_src = _strip_ambient(" ".join(value_text.split())[:_IDENT_SCAN_CHARS])
    all_src = _strip_ambient(" ".join(full_text.split())[:_IDENT_SCAN_CHARS])
    return ToolCallSignature(
        tool=tool_name or "",
        exact=normalized,
        idents=_identifiers(ident_src),
        preview=preview,
        all_idents=_identifiers(all_src),
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
    common = set(sigs[0].all_idents)
    for s in sigs[1:]:
        common &= s.all_idents
        if not common:
            break
    return frozenset(common)


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
    """
    return term.count("_") >= 2 or len(term) >= 16


def repetition_verdict(
    recent: list[ToolCallSignature],
    *,
    window: int = REPETITION_WINDOW,
    min_overlap: int = REPETITION_MIN_OVERLAP,
    containment: float = REPETITION_CONTAINMENT,
    threshold: int = REPETITION_THRESHOLD,
    ambient: frozenset[str] | None = None,
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
    # Turn-ambient identifiers carry no signal and must not carry a match.
    # `ambient` from the caller when it has more history than `recent` holds.
    if ambient is None:
        ambient = ubiquitous_identifiers(recent)
    cur_idents = current.idents - ambient
    # Path-addressed tools compare by exact repeat only — see _EXACT_ONLY_TOOLS.
    near_allowed = current.tool not in _EXACT_ONLY_TOOLS
    for p in prior:
        if p.tool != current.tool:
            continue
        if current.exact and p.exact == current.exact:
            matches.append(p)
            exact = True
            shared |= cur_idents
            continue
        if not near_allowed:
            continue
        p_idents = p.idents - ambient
        overlap = cur_idents & p_idents
        if not overlap:
            continue
        # Either several shared identifiers, or one distinctive enough to
        # stand alone. See `_is_distinctive`.
        enough = len(overlap) >= min_overlap or any(
            _is_distinctive(t) for t in overlap
        )
        if enough and _containment(cur_idents, p_idents) >= containment:
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
