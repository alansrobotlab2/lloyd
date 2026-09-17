"""Implement a backlog item that triage has confirmed, through one automod round.

This is the only automated path from a backlog item to landed code, and it is
deliberately the last link in a chain with two gates in front of it:

  1. `autotriage` must have reached `confirmed` on the item, WITH an
     acceptance check. An item confirmed without one is skipped here, not
     guessed at: "what must become true for this to be done" is the contract
     the round is held to, and a round with no contract cannot fail.
  2. The self-modification loop must be free: enabled, not halted, not BROKEN,
     no promotion under observation, no rollback pending, no round open. That
     is re-checked at run time, because a queue item can sit for a while.

Then one turn, in a real session, following `automod-change-own-code`. The
turn opens the round, does the work, gates it, and either lands it or aborts.
Landing runs detached and the turn ends, exactly as it does when a human
drives it — the three rounds that proved the loop worked were driven this way
by hand, and this is that procedure with the human replaced by a queue.

One attempt per item, unattended. Whatever the turn concludes — landed,
aborted, gate failed, decided the item was not worth it after all — is
recorded and the item is not picked again by this source. A second attempt is
a human's decision.

Why this goes through `/api/message/stream` rather than `run_query`: the
observer. `automod_start` refuses a turn with no Inner Voice attached, and the
chat path is the only one that attaches it. Going through it also puts the
round in the Inner Voice history, which is where anyone reviews what the agent
did afterwards.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.automod")

NAME = "autocode"
#: A long-lived re-admitter: 60-100 iterations, each re-submitting a
#: 150-200k context (the 09-09 203k round missed at iterations 44 and 76).
#: Held by the pool's KV gate while the primary is over budget.
LONG_LIVED = True
# The queue dequeues `priority ASC`: a lower number runs sooner. Research and
# distill jobs sit at 70 and arrive every few minutes, so at 80 the first
# unattended round (job 4476, 2026-09-07) sat queued behind four of them with
# no path to a slot. One round every four hours, gated on the loop being free,
# is the rarest and most valuable job in this pool; it goes first.
DEFAULT_PRIORITY = 40
DEDUP_KEY = "autocode:round"


def round_depth(src_cfg: dict | None = None) -> int:
    """How many rounds may be open at once: `workers.sources.autocode.
    max_inflight`, the same key the queue's claim cap reads, so one number in
    config.yaml sets both. 1 when absent or unreadable — the loop's shape
    from its first day, and what to go back to once the board is caught up."""
    cfg = src_cfg if src_cfg is not None else _source_cfg(NAME)
    try:
        return max(1, int(cfg.get("max_inflight", 1)))
    except (TypeError, ValueError):
        return 1


def _slot_key(slot: int) -> str:
    """Slot 0 keeps the bare key, so a row queued before slots existed still
    coalesces against the first slot across the restart that lands this."""
    return DEDUP_KEY if slot == 0 else f"{DEDUP_KEY}:{slot}"
# The pool back-dates this source's watermark when one of its runs ends
# (`WorkerPool._repoll_on_complete`), so the next round is asked for at the
# next scheduler pass rather than up to `interval_seconds` later.
REPOLL_ON_COMPLETE = True

# Blockers, not findings. Over the loop's first four days implement rounds
# filed 102 items against 7 closed, hit the then-cap of 3 in 21 of 43 rounds,
# and the three that aborted at the gate on 2026-09-08 filed 3, 6 and 5 while
# landing nothing. By 2026-09-11 it was 120 filed against 7 closed, and the
# re-runs of one item filed the same finding three times. A finding the
# round turns up now goes ONTO the item it came from (`## Findings`, counted
# off the file into `findings_appended`); the one thing that still becomes an
# item is a blocker — a finding that stops a clause of this round's contract
# from becoming true — and one of those per round is the shape.
SPAWN_CAP = 1
# 150, not 100. Round SM_20260908_165950 called `automod_land` at iteration 91
# of 100 — nine left for a landing that must be followed by an immediate turn
# end, and #278 died at 101 with a gated, ready change it never landed. The
# budget anchor fires at 75% and 90%, so a cap that is too tight spends its
# warnings during normal work and has nothing left for the real deadline. The
# cost of a larger cap is bounded by `max_duration_seconds` either way.
#
# This number only survives the whole chain: the source asks for it, the
# queue payload carries it, `/api/message/stream` reads it, and
# `messages._turn_budget` clamps it to a ceiling. Until 2026-09-12 that
# ceiling was `agent.max_turns_ceiling` = 120 for every caller and the
# clamp was silent, so this constant read 150 and the turn got 120 —
# which is what killed round 858, at the cap, before it could re-gate.
# The worker ceiling is `agent.max_turns_ceiling_worker` (200) and a clamp
# now logs. Raising this past that ceiling puts it back to being a wish.
DEFAULT_MAX_TURNS = 150

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

PROMPT = """\
Backlog item #{item_id} on your own board was triaged {triaged_ago} and \
**confirmed**: the premise still holds and there is real work here. Implement \
it through the self-modification loop, following the skill \
`automod-change-own-code` exactly — it holds the procedure for a `code`, \
`frontend`, `vault` or `mixed` surface, each gate rung, the `review` rung, and \
what to do when an existing test fails. Read it first; this message is the \
contract, not the procedure.

<item id="{item_id}" status="{status}" priority="{priority}">
# {name}

{body}
</item>

{members}<triage>
Verdict: confirmed
Surface: {surface}
Check that was run: {check}
Evidence: {evidence}
</triage>

**The acceptance check, recorded at triage, is your contract:**

    {acceptance}

As separately checkable clauses — the gate's review rung grades each one, and \
your finalizer reports each one:

{clauses}
{human_clauses}{surface_rules}
The round is done when every clause has become true and a test pins each. If \
you cannot make them true with one small, well-tested change, do not land a \
larger one — abort the round, say why, and the item goes back to a human. \
Open the round with `automod_start(goal, item_id={item_id})` — the `item_id` \
is what lets the review rung find these clauses.

**What the review grades.** Name each process boundary your change crosses \
and put a test across it. Every clause needs a test node in a file this diff \
changes, or a suite run cited as `tests/ -k <expr>` that was run. Test prose — \
docstrings, names, assert messages — is graded too: keep its numbers and \
claims exact. You get two review attempts per round; each refused commit \
spends one.

**Scope you discover is not scope you take — and it is not a new item \
either.** A second bug beside the first, a refactor the fix wants, a missing \
test: each goes **onto this item**, once, as a section: \
`backlog_write_task(task_id={item_id}, description_mode="append", \
description="## Findings (round <round id>)\\n\\n- <what is wrong, where \
(file:line), how to verify>", activity="findings appended by round <round id>")`. \
One bullet per finding; do not leave them only in your report: it is read \
once, the item until it is done. One change per round is what makes a \
rollback mean something.

**The one thing that becomes a new item is a blocker**: a finding that stops \
one of this round's clauses from becoming true. File it with \
`backlog_write_task` (board `lloyd`, no `task_id`, tags `spawned-by-autocode` \
and `blocker`, first line "Blocks #{item_id}"), as a handoff a fresh session \
can execute alone, and name its id as what the deferred clause waits on. If \
the tool answers `merged_into: N`, cite N instead. **File at most \
{spawn_cap}.** A second blocker means the item needs a human: stop and report. \
Nothing else becomes an item; everything else is a finding on this item.

{reoffer}**The triage evidence above was measured today, on this tree.** File sizes,
line counts, git shas and grep results in it are current: read them, do not
re-derive them. Re-measure exactly one thing — the acceptance check, which you
must confirm fails before you start and passes when you finish.

**Pacing.** You have {max_turns} iterations and a wall clock; running out of
either lands nothing. Triage read for you: `automod_start` by iteration 6
or minute 8; the failing test by iteration 25; the first `automod_gate` by
minute 30; after minute 40 start nothing you cannot gate. Commit before every
gate: re-gating the same commit is answered from the ledger without a review.
If a `<context>` or `<budget>` anchor fires, it is not advice — commit, gate,
and land or abort. A review refusal with under 25 iterations left is an abort:
`automod_abort` (branch kept; the re-offer resumes with the findings), never an
edit, a re-gate or a `land`. Never restart an engine or a service from a round
— refused at dispatch; note it on the item.

**Your outcome closes the item — or leaves it open.** The item is a proposal, \
not a promise: find out whether it improves Lloyd, and land it only if it does. \
When this turn ends you restate the result as one JSON object: whether the \
change landed, and **per clause** `met`, `not_met` or `deferred`, with the test \
node id or file:line. Once the promotion settles, an all-`met` item is closed \
automatically. `deferred` \
leaves it open and names the ids it waits on. A deferral that names no id is \
recorded as `not_met`. `not_met` re-offers it once for those clauses. Two verdicts close \
it with no landing: `unnecessary` means the work is not needed after all — the \
premise no longer holds, or the acceptance is already true; `rejected` means \
you built or measured it and the evidence says it does not improve things (an \
eval no better, cost above the gain) — put the measurement in `summary`. A rejection with evidence is a good outcome; a \
landing that improves nothing is the failure. A closed item is never \
re-triaged, so `met`, `unnecessary` or `rejected` on evidence you did not \
actually gather is the one claim this loop cannot recover from.

If your change needed a path the loop may never write — anything the gate's \
scope check denies — leave it out, land the rest, and report it under \
`human_paths` with one sentence saying what needed to change there. Never \
`git add -f`.

Report what you did, quoting the gate line rather than saying "it passed", \
and end with one line `SPAWNED: <ids of blocker items you filed or were merged \
into, or the word none>`. Work autonomously; do not ask for confirmation.
"""


# Rendered into `{surface_rules}` for a `vault` item only; a code item's
# prompt is byte-for-byte what it was. The template reads as a code round —
# `automod_start`, a test pinning each clause, the gate — and a vault item
# followed it: 8 of the first 15 vault turns cut a code round beside their
# vault commits. #575 landed its fix at 21:25Z (vault review: all five clauses
# met), then spent two rounds and ~300 iterations gating a test file for a
# script `~/lloyd` does not own, re-offered when the first died at its budget.
# The vault review already grades without tests (`require_tests=False`); this
# says so to the implementer. Kept out of PROMPT because the template is held
# under 5.5k chars (`test_the_template_stays_bounded`).
VAULT_SURFACE_RULES = """
**This is a `vault` item, and a vault change is not a round.** Edit the paths \
the clauses name under `~/obsidian`, check each clause by reading or running \
the result, then `automod_vault_land(paths, message, item_id={item_id})`. Do \
not call `automod_start`, and do not open a code round to add tests: the vault \
review grades the landed files by reading and running them and asks for no \
test in `~/lloyd`. Where the text below speaks of `automod_start`, a test \
pinning each clause, the gate or a promotion settling, read \
`automod_vault_land` and its review instead. A pinning test you think the \
change deserves is a finding on this item, not part of this turn. When the \
vault review passes with every clause met, stop and report `met` — and if the \
turn ends before it reports, that review's verdict is what closes the item. \
If a clause truly needs code in `~/lloyd`, land the vault half, file that code \
as the blocker described below, and report the clause `deferred` to it.
"""


def _surface_rules(surface: str, item_id: int) -> str:
    """The surface-specific block for the implement prompt; empty but for vault."""
    if str(surface or "").strip().lower() != "vault":
        return ""
    return VAULT_SURFACE_RULES.format(item_id=item_id)


ABANDON_GRACE_SECONDS = 20 * 60


def _reoffer_block(reason: str, *, prior: tuple[int, ...] | list[int] = (),
                   findings_appended: int = 0, clause_verdicts=()) -> str:
    """The banner an item gets when it is being offered again.

    Worded per verdict. "Never reached a verdict" was true of every re-offer
    until the review rung existed; a review refusal IS a verdict on the
    change, and the next round's first move is to read the findings and
    resume the branch, not to start over.

    `prior` is what earlier rounds of this item filed or merged into. A
    re-offered round told nothing re-derives the same peripheral findings and
    files them again — #549 ran four times in 110 minutes and filed ten
    children, three of them one finding — so the ids are put in front of it
    with the one instruction that stops that: append, do not re-file.
    """
    if not reason:
        return ""
    return (_reoffer_verdict(reason, clause_verdicts)
            + _reoffer_memory(prior, findings_appended))


def _last_review_clauses(item_id: int) -> list[dict]:
    """The per-clause verdicts of the item's most recent graded review, or []."""
    from scripts.automod import backlog as B, state as S
    try:
        for ev in reversed(B.review_events_for_item(S.LEDGER_PATH, item_id)):
            if ev.get("ok") and ev.get("clauses"):
                return [c for c in ev["clauses"] if isinstance(c, dict)]
    except Exception as exc:  # noqa: BLE001 — a banner line is not the round
        logger.warning("#%s: could not read the last review's clauses: %s", item_id, exc)
    return []


def _clause_verdicts_line(clauses) -> str:
    """`clause 1 met; clause 2 partial (downgraded: …): note` — what the
    grader said per clause. The findings prose names what refused; this is
    what did NOT, so a re-offered round does not redo a clause already met."""
    rows = []
    for c in clauses or []:
        tag = f" (downgraded: {'; '.join(map(str, c['downgraded']))[:160]})" if c.get("downgraded") else ""
        note = f": {str(c.get('note') or '')[:160]}" if c.get("verdict") != "met" and c.get("note") else ""
        rows.append(f"clause {c.get('clause')} {c.get('verdict')}{tag}{note}")
    return "; ".join(rows)


def _reoffer_memory(prior, findings_appended: int) -> str:
    if not prior and not findings_appended:
        return ""
    parts = []
    if prior:
        ids = " ".join(f"#{i}" for i in prior)
        parts.append(f"**Already filed by earlier rounds of this item: {ids} — do not "
                     f"file these again.** If you have more on one of them, "
                     f"`backlog_write_task(task_id=<id>, description_mode=\"append\")`.")
    if findings_appended:
        parts.append(f"Earlier rounds also appended {findings_appended} finding(s) under "
                     f"`## Findings` on this item; read them before you re-derive anything.")
    return " ".join(parts) + "\n\n"


def _human_clauses_block(clauses) -> str:
    """Conditions a person will satisfy after the code lands. Rendered so the
    round knows they exist and knows they are not its to do — or to fake."""
    clauses = [str(c) for c in (clauses or []) if str(c).strip()]
    if not clauses:
        return ""
    rows = "\n".join(f"    - {c}" for c in clauses)
    return (f"\nA person will do these after you land — they are not yours to do, "
            f"and not yours to simulate; the item stays open for them:\n\n{rows}\n")


def _reoffer_verdict(reason: str, clause_verdicts=()) -> str:
    verdict = reason.split(":", 1)[0].strip()
    if verdict == "review_retry":
        m = re.search(r"`automod/(SM_[0-9_]+)`", reason)
        branch = f"automod/{m.group(1)}" if m else "the branch named below"
        verdicts = _clause_verdicts_line(clause_verdicts)
        return (f"**This item is being offered again — the gate's review rung sent its "
                f"previous round back with findings ({reason}).** The work is on "
                f"`{branch}`. Pass `from_branch=\"{branch}\"` to `automod_start` so the new "
                f"worktree starts from it, rebased onto live main; then address each "
                f"finding by name before anything else, and say in your report which "
                f"finding each commit answers. The same grader reads the result, and "
                f"sees its earlier reviews of this item."
                + (f" Its last per-clause verdicts: {verdicts}." if verdicts else "")
                + "\n\n")
    if verdict == "partial":
        return (f"**This item is being offered again — its previous round landed, but "
                f"its own outcome reported clauses not met ({reason}).** The landed change "
                f"is in live main; your round is about the named clauses only.\n\n")
    return (f"**This item is being offered again — its previous round never reached a "
            f"verdict on the change ({reason}).** Read what is there before you start: "
            f"if a branch is named, `git log`/`git diff` it against your new base and "
            f"reuse what still applies rather than rewriting it. Say in your report "
            f"what you reused and what you redid.\n\n")


def _members_block(candidate, all_items: dict | None = None) -> str:
    """For an umbrella: the members it consolidates, so the round can read
    the original findings. Empty for an ordinary item."""
    if not getattr(candidate, "members", None):
        return ""
    from scripts.automod import backlog as B
    all_items = all_items if all_items is not None else {i.id: i for i in B.all_items(None)}
    n = len(candidate.members)
    cap = max(2000, 12_000 // max(1, n))
    blocks = []
    for mid in candidate.members:
        m = all_items.get(int(mid))
        if m is None:
            continue
        blocks.append(f'<member id="{m.id}" status="{m.status}">\n# {m.name}\n\n{m.body[:cap]}\n</member>')
    return ("This item is an **umbrella**: group triage consolidated the members below into "
            "one piece of work, and they close automatically when every clause is met. Read "
            "them for the original findings; do not file follow-ups that restate a member.\n\n"
            + "\n\n".join(blocks) + "\n\n")


def _review_note(events: list[dict], round_id: str | None) -> dict | None:
    """The review rung's refusal for this round, if that is how the gate
    last ended — read off the gate event, which outlives the round dir."""
    if not round_id:
        return None
    last = None
    for ev in events:
        if ev.get("event") == "gate" and ev.get("round_id") == round_id:
            last = ev
    if last and not last.get("ok") and last.get("rung") == "review":
        return last
    return None


def _abandon_grace_seconds() -> int:
    """How long a round left open waits for a rescue: `ABANDON_GRACE_SECONDS`
    while the Inner Voice observer watches this source, else none — the
    observer's ambient follow-up is the only rescue the grace ever waited
    for, and with `autocode.inner_voice: false` (2026-09-12) none can come."""
    from workers.sources import _common as C
    return ABANDON_GRACE_SECONDS if C.source_inner_voice(NAME) else 0


def reap_abandoned_rounds(now: float | None = None, *,
                          finished_session: str | None = None) -> list[dict]:
    """Close rounds this source opened that nobody finished — after a grace
    period when someone may still finish them, and only while nothing is
    happening in their session.

    The grace is #278's. Its implement turn died at the 100-iteration cap
    with the change written, tested and un-gated, and the ledger row said
    `max_turns`. Two minutes later the Inner Voice observer queued an ambient
    follow-up into that same session — "the round ended at the cap with no
    report; if the gate passed, land it" — and that second turn gated, landed,
    and the whole feature was live at 17:34. A reaper that aborted at turn end
    would have raced the rescue and thrown away 875 lines the gate then
    passed. So while the source is observed, the observer is the first
    responder and this is the backstop, `ABANDON_GRACE_SECONDS` after the turn
    ended. While it is not observed nothing can rescue the round, and the
    grace only held the loop closed: 15 rounds in the week to 2026-09-14
    waited a median 26 minutes each. Then `_run_and_record` calls this the
    moment the turn ends, passing the session that just ended so its own
    wind-down is not read as activity.

    Two protections the grace was silently providing are explicit instead: a
    round whose detached gate is still running (`S.gate_in_progress`) or whose
    landing is in flight (`S.land_in_progress` — waiting for idle, or
    re-gating after `main` moved) is never reaped. The branch is kept — it is
    the only record of what was attempted — and the item is told where it is.

    A second pass (2026-09-17) closes an **orphan**: a round a tool opened
    (`opened_by: tool` on its `round_start`) that no implement row names.
    SM_20260917_003459 was opened by a turn that had already written its
    `finished` row — a person typed `continue` into the autocode session
    after the reaper removed its worktree — so nothing above could key on
    it, `_loop_is_free` read it as open, and the loop stood still until a
    human aborted it. Reaped once its opener session is quiet and it is at
    least `ORPHAN_ROUND_MIN_AGE_SECONDS` old; a CLI round (`opened_by: cli`,
    or no key at all) is a person's and never touched.
    """
    from scripts.automod import state as S, worktree as W
    now = now or time.time()
    grace = _abandon_grace_seconds()
    events = S.read_events(limit=500)
    # `infra_failed` too: a turn can open a round and then lose its stream, and
    # a round nobody will ever close blocks `_loop_is_free` for every item
    # behind it.
    finished = [e for e in events if e.get("event") == "backlog_implement"
                and e.get("phase") in ("finished", "infra_failed") and e.get("round_id")]
    closed = {e.get("round_id") for e in events
              if e.get("event") in ("promoted", "round_aborted", "round_abandoned")}
    try:
        from app.sessions_io import active_sessions_snapshot
        busy = {s.get("session_id") for s in active_sessions_snapshot()}
    except Exception as exc:  # noqa: BLE001
        # Closed, not open: with no grace, "could not tell which sessions are
        # busy" would otherwise read as "none are" and abort a live round.
        logger.warning("reaper: session snapshot unavailable, reaping nothing: %s", exc)
        return []
    busy.discard(finished_session)
    current = S.read_current() or {}
    reaped: list[dict] = []

    def _reapable(rid: str) -> bool:
        if rid in closed or current.get("round_id") == rid:
            return False
        if not W.worktree_path(rid).exists():
            return False
        return not (S.gate_in_progress(rid) or S.land_in_progress(rid))

    for e in finished:
        rid = e["round_id"]
        age = now - float(e.get("ts") or 0)
        if age < grace or e.get("session_id") in busy or not _reapable(rid):
            continue
        review = _review_note(events, rid)
        why = (f"implement turn ended ({e.get('stop_reason')}) and the round "
               f"stayed open for {int(age // 60)} min with nothing running in "
               f"its session" if grace else
               f"implement turn ended ({e.get('stop_reason')}) with the round still "
               f"open, no gate or landing running, and no observer to rescue it")
        if review is not None:
            why = (f"the review rung sent the round back and the turn ended without "
                   f"abort or re-gate; {why}")
        _reap_round(rid, e.get("item_id"), why, reaped)
    # Orphans: opened by a tool, named by no implement row, opener quiet.
    named = {e.get("round_id") for e in events
             if e.get("event") == "backlog_implement" and e.get("round_id")}
    for e in events:
        if e.get("event") != "round_start" or e.get("opened_by") != "tool":
            continue
        rid = str(e.get("round_id") or "")
        if not rid or rid in named or rid in closed:
            continue
        age = now - float(e.get("ts") or 0)
        if age < ORPHAN_ROUND_MIN_AGE_SECONDS or e.get("session_id") in busy or not _reapable(rid):
            continue
        closed.add(rid)
        why = (f"opened by a tool from session {e.get('session_id') or '?'} {int(age // 60)} min "
               f"ago, no implement turn names it, and nothing is running in that session — "
               f"an orphan the loop would otherwise read as open forever")
        _reap_round(rid, e.get("item_id"), why, reaped)
    return reaped


# How old a tool-opened round with no implement row must be before the reaper
# treats it as an orphan. The opener's turn is normally still running (its
# session is busy, so the age never matters); the floor is for the seconds
# between `automod_start` returning and the turn's next tool call.
ORPHAN_ROUND_MIN_AGE_SECONDS = 600


def _reap_round(rid: str, item_id, why: str, reaped: list[dict]) -> None:
    from scripts.automod import backlog as B, round as R, state as S
    R.abort(rid, reason=why)
    rec = {"event": "round_abandoned", "round_id": rid, "item_id": item_id,
           "branch": f"automod/{rid}", "reason": why}
    S.append_event(rec)
    if item_id is not None:
        B.note_item(int(item_id),
                    f"automod round {rid} abandoned: {rec['reason']}. Its work is on "
                    f"branch `automod/{rid}` in ~/lloyd.")
        B.set_status(int(item_id), "up_next", "its round was abandoned; back in the pool")
    logger.warning("reaped abandoned round %s (%s)", rid, rec["reason"])
    reaped.append(rec)


def _backend_boot_ts() -> float | None:
    """When this backend process started — or None anywhere else.

    Only the backend runs the pool and the turns it starts, so only here does
    "started before this process" mean "died with its predecessor". From a
    CLI the same test is true of every live turn. The process, not the pool:
    `POST /api/workers/enable` stops and restarts the pool in place, while the
    chat path keeps a turn running after its client goes away.
    """
    from workers.pool import get_pool
    if get_pool() is None:
        return None
    try:
        import psutil
        return float(psutil.Process().create_time())
    except Exception:  # noqa: BLE001 — an unknown boot settles nothing
        return None


def _orphan_round(starts: list[dict], item_id: int, since_ts: float, boot_ts: float) -> str | None:
    """The round a dead turn opened: the last `round_start` between its
    `started` row and the boot, unless that round's spec binds another item.

    Bounded above by the boot on purpose. `_round_opened_since` takes the last
    round at or after a time, which after a restart can be the NEXT item's
    live round — and handing that id to the reaper aborts it.
    """
    from scripts.automod import state as S
    import yaml
    for ev in reversed(starts):
        ts = float(ev.get("ts") or 0)
        if not (since_ts <= ts < boot_ts) or not ev.get("round_id"):
            continue
        rid = str(ev["round_id"])
        try:
            spec = yaml.safe_load((S.ROUNDS_DIR / rid / "run_spec.yaml").read_text()) or {}
            bound = (spec.get("item") or {}).get("id")
        except (OSError, ValueError, yaml.YAMLError):
            bound = None
        if bound is not None and int(bound) != int(item_id):
            continue
        return rid
    return None


def settle_orphaned_turns(boot_ts: float | None = None) -> list[dict]:
    """Write the terminal row a turn killed with the backend never wrote.

    `execute` records `started`, then `finished` or `infra_failed` when the
    turn returns. A process that dies in between writes neither, and a
    `started` row with nothing after it is exactly what a live turn looks
    like: `items_with_unfinished_rounds` calls the item mid-round, and
    `reap_abandoned_rounds`, which reads only terminal rows, never sees its
    round. On 2026-09-15 systemd-oomd killed the whole unit at 04:48:34Z two
    minutes into #1131's round; supervisord had everything back in half a
    minute, and every autocode poll declined "a round is already open" for
    the next thirteen and a half hours.

    A turn runs inside the backend process, so one whose `started` row
    predates this process's boot cannot still be running. Its row is
    `infra_failed` — `implement_outcomes` re-offers that, capped, because a
    crash is not a judgment on the item — carrying the round it opened, which
    the reaper then closes under its usual guards (a detached gate or landing
    survives a backend restart, and still protects the round). Rows written
    after the boot are this process's own and are never touched.
    """
    from scripts.automod import backlog as B, state as S
    boot = _backend_boot_ts() if boot_ts is None else boot_ts
    if not boot:
        return []
    orphans = [(iid, rows[-1]) for iid, rows in B.implement_history(S.LEDGER_PATH).items()
               if str(rows[-1].get("phase") or "") == "started"
               and 0 < float(rows[-1].get("ts") or 0) < boot]
    if not orphans:
        return []
    starts = [e for e in S.read_events(limit=10**9) if e.get("event") == "round_start"]

    def stamp(t: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    settled: list[dict] = []
    for iid, row in orphans:
        ts = float(row["ts"])
        rid = _orphan_round(starts, iid, ts, boot)
        why = (f"the backend restarted at {stamp(boot)} under a turn started {stamp(ts)}; "
               f"no terminal row was written")
        rec = {"event": "backlog_implement", "item_id": iid, "phase": "infra_failed",
               "session_id": None, "round_id": rid, "num_turns": None,
               "stop_reason": "backend_restarted", "errors": [why]}
        S.append_event(rec)
        B.note_item(iid, f"implement turn lost: {why}; not counted as an attempt"
                         + (f" (round {rid})" if rid else ""))
        logger.warning("backlog #%s: %s (round %s)", iid, why, rid)
        settled.append(rec)
    return settled


# Once per process. Every row older than the boot is settled by the first pass
# that succeeds, and rows written afterwards are never this pass's business.
_boot_settled = {"done": False}


def _settle_boot_orphans() -> None:
    """`settle_orphaned_turns`, then the reaper for the rounds it named — at
    the first poll after a boot rather than at housekeeping's next 900 s tick."""
    try:
        if settle_orphaned_turns():
            for r in reap_abandoned_rounds():
                logger.info("reaped round %s after a backend restart: %s",
                            r["round_id"], r["reason"])
        _boot_settled["done"] = True
    except Exception as exc:  # noqa: BLE001 — retried at the next poll
        logger.warning("settle_orphaned_turns failed: %s", exc)


def _loop_is_free(depth: int | None = None) -> tuple[bool, str]:
    """Every gate the loop itself enforces, checked here first so a queued
    item does not spend a full agent turn discovering it cannot proceed.

    `depth` rounds may be open at once (`round_depth`). Everything else is
    unchanged by it: a `landing` promotion still holds every new round back,
    which is also what lets a landing wait out the other round's turn."""
    depth = round_depth() if depth is None else max(1, int(depth))
    from scripts.automod import state as S, worktree as W

    if not S.is_enabled(LIVE_ROOT):
        return False, "automod.enabled is false"
    if S.is_halted():
        return False, "promotions are halted"
    if S.is_broken():
        return False, "guardian is BROKEN"
    current = S.read_current()
    # The chamber (`automod.chamber`): only `land` needs the observation
    # window closed. The turn and the gate touch the round's worktree and the
    # canary ports, and triage and digest turns already run on the live
    # backend while a promotion is observed. So an `observing` promotion no
    # longer holds the next round back — its landing waits for the settle
    # (`promote.wait_for_settle`). A `landing` one still does: the backend is
    # about to restart underneath whatever starts.
    if current and not (current.get("state") == "observing" and S.chamber_enabled(LIVE_ROOT)):
        return False, (f"promotion {str(current.get('commit'))[:8]} is under "
                       f"observation ({current.get('state')})")
    if S.read_rollback_request():
        return False, "a rollback request is pending"
    # Only worktrees the loop itself owns count as an open round: the round
    # worktrees and the review/calibration checkouts, all under
    # `~/lloyd-work`. `git worktree list` also reports anything a model or a
    # human added elsewhere — on 2026-09-13 a round's own scratch checkout at
    # `/tmp/wt484` was left behind and blocked every implement poll for
    # eleven hours ("a round is already open (1 worktree(s))") with nothing
    # in flight. A stray registration is logged, never counted or removed.
    owned, stray = _loop_worktrees(W.prune_orphans(LIVE_ROOT))
    if stray:
        logger.info("autocode: ignoring %d worktree(s) outside %s: %s",
                    len(stray), _LOOP_WORKTREE_ROOT, ", ".join(stray[:3]))
    if len(owned) >= depth:
        return False, (f"a round is already open ({len(owned)} worktree(s))" if depth == 1 else
                       f"{len(owned)} round(s) open, depth {depth}")
    return True, "free"


_LOOP_WORKTREE_ROOT = Path.home() / "lloyd-work"


def _loop_worktrees(paths: list[str]) -> tuple[list[str], list[str]]:
    """`(owned, stray)`: registered worktrees under the loop's root, and not.

    The main checkout is neither — it is the repo. Resolved, so a symlinked
    home cannot make an owned worktree look stray."""
    root = str(_LOOP_WORKTREE_ROOT.resolve())
    live = str(Path(LIVE_ROOT).resolve())
    owned, stray = [], []
    for raw in paths:
        try:
            resolved = str(Path(raw).resolve())
        except OSError:
            resolved = raw
        if resolved == live:
            continue
        (owned if resolved.startswith(root + "/") or resolved == root else stray).append(raw)
    return owned, stray


def _age_phrase(ts: float | None) -> str:
    if not ts:
        return "recently"
    days = max(0, int((time.time() - float(ts)) // 86400))
    return "today" if days == 0 else f"{days} day{'s' if days != 1 else ''} ago"


HOUSEKEEPING_KEY = "last_housekeeping"
_last_decline: dict[str, str] = {"why": ""}


def _housekeeping_due(queue: WorkQueue, src_cfg: dict) -> bool:
    """Whether the board passes are due: once per `interval_seconds`.

    They used to ride the source's own watermark. Now that a declined round
    check is retried in `retry_seconds`, they need their own, or a 60-second
    retry would walk the whole board — reap, close, reconcile, expire — every
    minute a round is under observation.
    """
    wait = int(src_cfg.get("interval_seconds", 900))
    last = queue.wm_get(NAME, HOUSEKEEPING_KEY)
    if last:
        try:
            from datetime import datetime, timezone
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
            if elapsed < wait:
                return False
        except ValueError:
            pass
    return True


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> str | None:
    """Housekeeping on its own clock, then: is the loop free, and is there work?

    Returns `DECLINED` when the loop is not free, or when the enqueue
    coalesced against a round row still in flight, so the pool looks again in
    `retry_seconds` instead of a whole `interval_seconds` — `execute`
    re-checks the gates at run time, so looking early is safe. Anything else
    (nothing confirmed, enqueued) returns None and the watermark advances as
    it always has.
    """
    from datetime import datetime, timezone

    from scripts.automod import backlog as B, state as S
    from workers.sources import DECLINED

    # Both board walks off the event loop: each reads every item file and the
    # ledger several times (~2 s for `select_confirmed` alone), and this loop
    # serves every HTTP request and streams every chat turn.
    if not _boot_settled["done"]:
        await asyncio.to_thread(_settle_boot_orphans)
    if _housekeeping_due(queue, src_cfg):
        await asyncio.to_thread(_housekeeping, src_cfg)
        queue.wm_set(NAME, HOUSEKEEPING_KEY, datetime.now(timezone.utc).isoformat())
    free, why = _loop_is_free()
    if not free:
        # Once per reason at INFO: at a 60 s retry the same sentence would
        # otherwise fill the log for the forty minutes a round runs.
        (logger.info if why != _last_decline["why"] else logger.debug)(
            "autocode: not queueing — %s", why)
        _last_decline["why"] = why
        return DECLINED
    _last_decline["why"] = ""
    # The gear change (2026-09-15): while autotriage's sweep still has open
    # items to read, no round starts. The round hold would otherwise keep the
    # sweep to the gaps between rounds — a third of the wall clock — and a
    # day of no landings buys every item on the board a reading and a rank.
    # Rounds resume by themselves when `sweep_pending` reaches 0. Ships off
    # like the sweep itself; config.yaml turns both on for the sprint.
    if bool(src_cfg.get("yield_to_sweep", False)):
        pending = await asyncio.to_thread(B.sweep_pending, S.LEDGER_PATH)
        if pending:
            why = f"yielding to the backlog sweep ({pending} item(s) unread)"
            (logger.info if why != _last_decline["why"] else logger.debug)(
                "autocode: not queueing — %s", why)
            _last_decline["why"] = why
            return DECLINED
    # A round row still queued or running coalesces any enqueue below. Asked
    # first, because `select_confirmed` walks the whole board (~2 s) and a
    # decline is retried every `retry_seconds`.
    depth = round_depth(src_cfg)
    slot = None
    for i in range(depth):
        if not await asyncio.to_thread(queue.has_live, _slot_key(i)):
            slot = i
            break
    if slot is None:
        logger.debug("autocode: not queueing — the previous round's row is still in flight")
        return DECLINED
    if await asyncio.to_thread(B.select_confirmed, S.LEDGER_PATH) is None:
        return None
    new_id = queue.enqueue(
        source=NAME, kind="round",
        payload={"max_turns": int(src_cfg.get("max_turns", DEFAULT_MAX_TURNS)),
                 # Carried in the payload like the budget, so a queued item runs
                 # under the config that was live when it was enqueued.
                 "structured_outcome": bool(src_cfg.get("structured_outcome", True))},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=_slot_key(slot),
    )
    if new_id is None:
        # Coalesced: the previous round's queue row is still `running`. The
        # loop reads free the moment `automod_abort` removes the worktree, but
        # the turn then spends one to three minutes on its finalizer, and
        # returning None here stamped the watermark as a spent interval — the
        # next look came 900 s later. 56 such gaps in the week to 2026-09-14,
        # median 12.6 min, ~20 h of a free loop. It is a decline: look again
        # in `retry_seconds`.
        logger.debug("autocode: not queueing — the previous round's row is still in flight")
        return DECLINED
    logger.info("Enqueued backlog implement id=%d (slot %d of %d)", new_id, slot + 1, depth)
    # One row per look. With a slot still empty, the next look is a retry
    # away rather than an interval — a decline in everything but name.
    return DECLINED if slot + 1 < depth else None


def _housekeeping(src_cfg: dict) -> None:
    """The board passes. Each is guarded: none may take the scheduler down."""
    from scripts.automod import backlog as B, state as S

    try:
        reap_abandoned_rounds()
    except Exception as exc:  # the backstop must never take the scheduler down
        logger.warning("reap_abandoned_rounds failed: %s", exc)
    try:
        # Landed items used to stay open forever: nothing joined `promoted`,
        # `settled` and `finished` back to the item. Same rule as the reaper —
        # this must never take the scheduler down.
        for r in B.close_settled_items(S.LEDGER_PATH,
                                       enabled=bool(src_cfg.get("close_on_settle", True)),
                                       close_members=bool(src_cfg.get("close_members_on_settle", True))):
            logger.info("backlog #%s landed: %s (acceptance=%s)", r["item_id"],
                        "closed" if r["closed"] else "noted, left open", r["acceptance"])
    except Exception as exc:
        logger.warning("close_settled_items failed: %s", exc)
    tri = _source_cfg("autotriage")
    try:
        # Before the reconcile, so a released member is judged in the same
        # pass. The switch lives in triage's block, which owns umbrellas.
        for r in B.unfold_spent_umbrellas(S.LEDGER_PATH,
                                          enabled=bool(tri.get("unfold_spent_umbrellas", True))):
            logger.info("umbrella #%s unfolded, %d member(s) released: %s",
                        r["umbrella_id"], len(r["released"]), r["reason"])
    except Exception as exc:
        logger.warning("unfold_spent_umbrellas failed: %s", exc)
    try:
        # One automatic second life, through triage, before the reconcile
        # would park the item for a human. See `B.retriage_spent_items`.
        for r in B.retriage_spent_items(S.LEDGER_PATH,
                                        enabled=bool(src_cfg.get("retriage_spent", True))):
            logger.info("backlog #%s sent back through triage: %s", r["item_id"], r["reason"])
    except Exception as exc:
        logger.warning("retriage_spent_items failed: %s", exc)
    try:
        # Status is the pipeline's state machine and the ledger is its source
        # of truth: every poll, anything the two disagree on moves. This is
        # also what migrated the board on 2026-09-09.
        for r in B.reconcile_statuses(S.LEDGER_PATH,
                                      enabled=bool(src_cfg.get("status_pipeline", True))):
            logger.info("backlog #%s: %s → %s", r["item_id"], r["from"], r["to"])
    except Exception as exc:
        logger.warning("reconcile_statuses failed: %s", exc)
    try:
        # Held confirmations enter the pool as rounds drain it. Triage releases
        # at the start of its own runs too; this is the path that still works
        # when triage is switched off, so nothing it held can strand. Floor and
        # switch come from triage's config block, which owns the gate.
        for r in B.release_held_confirmations(
                S.LEDGER_PATH, floor=int(tri.get("implement_pool_floor", B.IMPLEMENT_POOL_FLOOR)),
                enabled=bool(tri.get("hold_confirmations", True))):
            logger.info("backlog #%s released from hold: %s", r["item_id"], r["reason"])
    except Exception as exc:
        logger.warning("release_held_confirmations failed: %s", exc)
    try:
        # The hard bound: a self-filed draft nothing picked up in
        # `expire_spawns_after_days` (B.spawn_expiry_days) is closed, tagged,
        # reopenable. Same rule as the two above — never takes
        # the scheduler down. Off: held items only accumulate, visible in the
        # triage skip summary and the scorecard gauge, never lost.
        for r in B.expire_stale_spawns(S.LEDGER_PATH,
                                       enabled=bool(src_cfg.get("expire_spawns", True))):
            logger.info("backlog #%s expired after %s d untouched", r["item_id"], r["age_days"])
    except Exception as exc:
        logger.warning("expire_stale_spawns failed: %s", exc)


def _escalate_review_disagreement(candidate, round_id: str | None) -> bool:
    """When the round just recorded is a review disagreement: note it, tag it,
    write `review_escalated`, and tell a person — unless the loop's own second
    life is still owed (`_second_life_owed`). True when it escalated."""
    from scripts.automod import backlog as B, state as S
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH).get(candidate.id, ("", ""))
    if not detail.startswith("review disagreement"):
        return False
    B.note_item(candidate.id, f"escalated: {detail}")
    B.tag_item(candidate.id, add=("review-disagreement",))
    S.append_event({"event": "review_escalated", "item_id": candidate.id,
                    "round_id": round_id, "reason": detail[:400]})
    if _second_life_owed(candidate):
        # Housekeeping re-triages it (or unfolds the umbrella) with this refusal
        # attached; nobody is needed yet, and a toast saying otherwise is one a
        # person learns to ignore.
        logger.info("#%s: review disagreement; the automatic second life is still owed, "
                    "not announcing", candidate.id)
        return True
    try:
        from scripts.automod.promote import announce
        announce(f"#{candidate.id} needs you",
                 f"review sent it back twice on the same clause: {detail[:160]}")
    except Exception as exc:  # noqa: BLE001 — an announcement never fails a round
        logger.warning("announce failed: %s", exc)
    return True


def _second_life_owed(candidate) -> bool:
    """Whether housekeeping, not a person, handles this spend next: an
    umbrella is unfolded while `autotriage.unfold_spent_umbrellas` is on; any
    other item is re-triaged while `retriage_spent` is on and it has not had
    its one re-triage."""
    from scripts.automod import backlog as B, state as S
    try:
        if B.is_umbrella(candidate):
            return bool(_source_cfg("autotriage").get("unfold_spent_umbrellas", True))
        if not bool(_source_cfg(NAME).get("retriage_spent", True)):
            return False
        return B.retriage_counts(S.LEDGER_PATH).get(int(candidate.id), 0) < B.RETRIAGE_CAP
    except Exception:  # noqa: BLE001 — when unsure, tell the human
        return False


def _source_cfg(name: str) -> dict:
    """Another worker source's config block, or `{}` when config is unreadable."""
    try:
        from app.config import CONFIG
        return dict(((CONFIG.get("workers") or {}).get("sources") or {}).get(name) or {})
    except Exception:  # noqa: BLE001
        return {}


def _round_opened_since(events: list[dict], since_ts: float, *,
                        item_id: int | None = None,
                        session_id: str | None = None) -> str | None:
    """The round this turn opened: the latest `round_start` since it began.

    By time alone that was exact while one round ran at a time. With two
    (`max_inflight` > 1) the latest row may be the OTHER turn's, and a
    `finished` row naming the wrong round hands the reaper a round that is
    still being worked on. `round_start` has carried the opener's
    `session_id` and the `item_id` since the orphan fix: a row that names
    either one and names someone else is not ours. A row naming neither (a
    ledger from before those fields) matches as it always did.
    """
    for ev in reversed(events):
        if ev.get("event") != "round_start" or float(ev.get("ts") or 0) < since_ts:
            continue
        theirs_s, theirs_i = ev.get("session_id"), ev.get("item_id")
        if session_id and theirs_s and theirs_s != session_id:
            continue
        if item_id is not None and theirs_i is not None and int(theirs_i) != int(item_id):
            continue
        return ev.get("round_id")
    return None


async def execute(item: QueueItem) -> dict[str, Any]:
    from scripts.automod import backlog as B, state as S

    free, why = _loop_is_free()
    if not free:
        return {"status": "skipped", "summary": why}
    pair = B.select_confirmed(S.LEDGER_PATH)
    if pair is None:
        return {"status": "skipped", "summary": "no confirmed item with an acceptance check"}
    candidate, triage = pair
    budget = int((item.payload or {}).get("max_turns") or DEFAULT_MAX_TURNS)

    # Read BEFORE the `started` row below, and this ordering is the whole
    # fix. `implement_outcomes` judges an item by its LATEST implement row,
    # and a `started` row with no round reads as `spent` — so a banner
    # computed after it was always empty. 0 of the 92 autocode sessions on
    # record to 2026-09-13 had ever been told they were a re-offer: no
    # `from_branch`, no findings, no clause verdicts, every one a fresh start.
    reoffer = _reoffer_for(candidate.id)

    # Recorded BEFORE the turn. `implemented_ids` counts any event for the
    # item, so this is what makes it one attempt per item: a turn that crashes
    # or times out must not put the item back on the pile to be retried
    # unattended.
    started = time.time()
    S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                    "phase": "started", "name": candidate.name[:200],
                    "budget": budget})
    B.set_status(candidate.id, "in_progress", "automod round starting")
    try:
        return await _run_and_record(item, candidate, triage, budget, started, reoffer)
    finally:
        # Every exit — landed, aborted, timed out, never ran, skipped for a
        # drain — hands the item back to the ledger's verdict. Without this the
        # early returns left an item `in_progress` that nothing would ever
        # pick up again, which is the old failure wearing a new status.
        #
        # The settle sweep first. It otherwise runs only in housekeeping, once
        # per `interval_seconds`, while `REPOLL_ON_COMPLETE` asks for the next
        # round at once — so a vault landing this turn made (which needs no
        # window) was reconciled back into `up_next` and re-picked before
        # anything closed it. #575: round one ended 22:06:27Z, its item was
        # `up_next` at :38 and in a second round at 22:08:44.
        #
        # Only after a turn that made a vault landing: the sweep walks the
        # whole board (~2.3 s measured, on the event loop), and a code landing
        # has the guardian's window to wait out, which housekeeping covers.
        # Synchronous like the reconcile beside it, so a cancelled `execute`
        # cannot run the two concurrently over the same item files.
        try:
            if _vault_landed_since(candidate.id, started):
                cfg = _source_cfg(NAME)
                for r in B.close_settled_items(
                        S.LEDGER_PATH, enabled=bool(cfg.get("close_on_settle", True)),
                        close_members=bool(cfg.get("close_members_on_settle", True))):
                    logger.info("backlog #%s landed: %s (acceptance=%s)", r["item_id"],
                                "closed" if r["closed"] else "noted, left open", r["acceptance"])
        except Exception as exc:
            logger.warning("close_settled_items after #%s failed: %s", candidate.id, exc)
        try:
            B.reconcile_statuses(S.LEDGER_PATH)
        except Exception as exc:
            logger.warning("reconcile_statuses after #%s failed: %s", candidate.id, exc)


def _vault_commits_since(events: list[dict], item_id: int, since_ts: float) -> list[str]:
    """Shas of this item's successful `vault_land`s at or after `since_ts`."""
    return [e.get("commit") for e in events
            if e.get("event") == "vault_land" and e.get("ok")
            and e.get("item_id") == item_id and float(e.get("ts") or 0) >= since_ts]


def _vault_landed_since(item_id: int, since_ts: float) -> bool:
    """Whether a `vault_land` for this item succeeded at or after `since_ts`."""
    from scripts.automod import state as S
    return bool(_vault_commits_since(S.read_events(limit=500), item_id, since_ts))


async def _reap_at_turn_end(session_id: str | None) -> None:
    """The reaper, the moment an implement turn ends, off the event loop.

    With no observer the round a turn left open cannot be rescued, and every
    minute it stays open is a minute `_loop_is_free` refuses the next item.
    With one, the grace inside `reap_abandoned_rounds` still holds it and
    this call finds nothing. A failure costs the early close, never the run:
    housekeeping's pass is the backstop.
    """
    try:
        for r in await asyncio.to_thread(reap_abandoned_rounds, finished_session=session_id):
            logger.info("reaped round %s at turn end: %s", r["round_id"], r["reason"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("reap at turn end failed: %s", exc)


def _reoffer_for(item_id: int) -> str:
    """The re-offer banner for an item, from the ledger as it stands NOW.

    Call it before this attempt's `started` row is written (see `execute`).
    """
    from scripts.automod import backlog as B, state as S
    return _reoffer_block(
        B.reoffer_reason(S.LEDGER_PATH, item_id),
        prior=B.prior_spawned(S.LEDGER_PATH, item_id),
        findings_appended=sum(r["findings_appended"]
                              for r in B.prior_rounds(S.LEDGER_PATH, item_id)),
        clause_verdicts=_last_review_clauses(item_id))


async def _run_and_record(item, candidate, triage, budget, started,
                          reoffer: str = "") -> dict[str, Any]:
    from scripts.automod import backlog as B, state as S
    from workers.sources._common import DrainActive, TurnTimeout, run_prompt_in_session
    logger.info("implementing backlog #%s (budget %d): %s",
                candidate.id, budget, candidate.name[:70])

    prompt = PROMPT.format(
        item_id=candidate.id, status=candidate.status, priority=candidate.priority,
        # 12k, not 30k. The template is ~10k chars and the item body is the
        # other half of what a round reads before it starts — and an umbrella
        # brought 30k of body plus 24k of members, ~13k tokens of a 262k
        # window spent before the first tool call. Three rounds died at the
        # wall on 2026-09-11. A body longer than this is a sign the item
        # needs splitting, which the triage pass does.
        name=candidate.name, body=candidate.body[:12_000],
        # The budget the model actually has, so the pacing block is about
        # this turn rather than about a number nobody passed in.
        max_turns=DEFAULT_MAX_TURNS,
        triaged_ago=_age_phrase(triage.get("ts")),
        surface=triage.get("surface") or "code",
        check=triage.get("check") or "(none recorded)",
        evidence=(triage.get("evidence") or "(none recorded)")[:2000],
        acceptance=triage.get("acceptance") or "",
        clauses="\n".join(f"    {i}. {c}" for i, c in enumerate(
            B.acceptance_clauses_of(triage), 1)) or "    (the contract above is one clause)",
        human_clauses=_human_clauses_block(
            B.human_clauses_for_item(getattr(candidate, "path", None), triage)),
        surface_rules=_surface_rules(triage.get("surface") or "code", candidate.id),
        spawn_cap=SPAWN_CAP,
        members=_members_block(candidate),
        # A re-offer is not a fresh start. The previous round's branch may
        # still hold the work, or a landing may have been reverted, and a
        # round told nothing re-derives it — or redoes it. Nor re-files it:
        # the ids earlier rounds filed ride along. Built by `execute` before
        # this attempt's `started` row, which would otherwise erase it.
        reoffer=reoffer,
    )
    want_outcome = bool((item.payload or {}).get("structured_outcome", True))
    # Taken BEFORE the turn: an id the turn claims that is at or below this
    # already existed, so it is a merge (or a citation), not a spawn.
    id_floor = B.max_item_id()
    body_before = candidate.body
    try:
        run = await run_prompt_in_session(
            prompt, title=f"autocode #{candidate.id}: {candidate.name[:48]}",
            source=NAME, max_turns=budget, priority=1,
            final_schema=B.IMPLEMENT_OUTCOME_SCHEMA if want_outcome else None,
            final_schema_prompt=(
                "Restate the result of this round as a single JSON object matching "
                "the schema: whether the change landed, and for EACH acceptance clause "
                "in order whether it is now met, not_met, or deferred (with the ids it "
                "waits on) and the test node id or file:line that shows it. Same filed "
                "ids as your SPAWNED line. This is a transcription of what you already "
                "reported, not a re-decision — `met` on every clause closes the item "
                "once the promotion settles, and a deferral that names no id is "
                "recorded as not_met, so say what is true."
            ))
    except DrainActive as exc:
        S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                        "phase": "skipped", "reason": f"landing in progress: {exc}"})
        return {"status": "skipped", "summary": f"landing in progress: {exc}"}
    except TurnTimeout as exc:
        # `finished` rather than a bare failure, so `reap_abandoned_rounds`
        # can still find and close a round this turn opened. In-band, so the
        # queue does not retry: the `started` event above already means one
        # attempt per item, and a retry would only re-discover that.
        # `vault_commits` and `surface` as on the normal path: a vault item
        # that landed a passing fix and then hit the wall clock is otherwise
        # invisible to `settled_landings` — #575's churn by another exit.
        timeout_events = S.read_events(limit=200)
        S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                        "phase": "finished", "reason": str(exc),
                        "round_id": _round_opened_since(timeout_events, started,
                                                        item_id=candidate.id),
                        "vault_commits": _vault_commits_since(timeout_events, candidate.id, started),
                        "surface": triage.get("surface") or "code",
                        "stop_reason": "turn_timeout"})
        logger.warning("backlog #%s: %s", candidate.id, exc)
        await _reap_at_turn_end(None)
        return {"status": "failed", "item_id": candidate.id,
                "summary": f"#{candidate.id}: {exc}"}

    events = S.read_events(limit=200)
    round_id = _round_opened_since(events, started, item_id=candidate.id,
                                   session_id=run.get("session_id"))

    # A turn that never reported completion is not an attempt. The stream
    # closes without a `done` frame when the backend is down or the connection
    # drops, leaving `stop_reason=None` — and #392 was recorded as its item's
    # one attempt ONE SECOND after starting, on a session holding a single user
    # message, while the guardian was alerting that supervisord was
    # unreachable. Recorded under its own phase so `implement_outcomes` can
    # tell it from a round that ran and failed, and so `reap_abandoned_rounds`
    # (which looks for `finished`) is not handed a round that was never opened.
    if run.get("stop_reason") is None:
        S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                        "phase": "infra_failed", "session_id": run["session_id"],
                        "round_id": round_id,
                        "errors": [str(e)[:300] for e in (run.get("errors") or [])[:5]],
                        "num_turns": run.get("num_turns")})
        logger.warning("backlog #%s: turn never completed (session %s); not an attempt",
                       candidate.id, run["session_id"])
        return {"status": "failed", "item_id": candidate.id,
                "summary": f"#{candidate.id}: turn never completed — not counted as an attempt"}
    vault_commits = _vault_commits_since(events, candidate.id, started)
    outcome = B.parse_outcome(run.get("structured")) if want_outcome else None
    outcome_error = str(run.get("structured_error") or "")
    # A path the round needed and the loop may never write. Recorded on the
    # item, which tags it `needs-human` and holds it open — reported rather
    # than hidden, which is what `git add -f` was.
    if outcome and outcome.get("human_paths"):
        try:
            recorded = B.record_human_paths(candidate.id, outcome["human_paths"],
                                            round_id=round_id)
            if recorded:
                S.append_event({"event": "human_paths", "item_id": candidate.id,
                                "round_id": round_id, "paths": recorded})
        except Exception as exc:  # noqa: BLE001 — a note is not the round
            logger.warning("#%s: could not record human_paths: %s",
                           candidate.id, exc)
    if outcome and outcome["acceptance"] in B.ITEM_VERDICT_OUTCOMES:
        # A verdict on the item with no landing to wait for. `unnecessary`:
        # the premise no longer holds, or the acceptance is already true.
        # `rejected`: the round built or measured it and the evidence says it
        # does not improve things — Alan's rule (2026-09-16) that every item
        # is a proposal and a negative result is a clean close, not a spent
        # attempt. Closed here rather than by the settle sweep, which only
        # sees landings.
        verdict = outcome["acceptance"]
        if getattr(candidate, "members", None):
            # A wrong verdict on six findings is the one claim the loop
            # should not make alone: the umbrella closes, the members stay
            # folded and a human decides (unfold_umbrella releases them).
            B.note_item(candidate.id,
                        f"closed as {verdict} with members {candidate.members} still folded; "
                        f"a human decides whether to release them (unfold_umbrella)")
            B.tag_item(candidate.id, add=(B.NEEDS_HUMAN_TAG,))
        why = ("the round found the work unnecessary" if verdict == "unnecessary"
               else "the round tried it and rejected it on the evidence")
        if verdict == "rejected":
            # Findable on the board: a rejected proposal is a result worth
            # reading, and the tag is how a later triage of the same idea
            # sees that it was already tried. Tagged BEFORE the close:
            # `tag_item` reaches items through `open_items`, and a done item
            # is not open.
            B.tag_item(candidate.id, add=(B.REJECTED_TAG,))
        B.set_status(candidate.id, "done",
                     why + (f": {outcome['summary']}" if outcome.get("summary") else ""))
        S.append_event({"event": "item_closed", "item_id": candidate.id, "by": "autocode",
                        "acceptance": verdict, "reason": outcome.get("summary", "")[:300]})
    # The review rung's word, onto the item. A refusal is on the gate event,
    # which outlives the round dir; what goes on the item is the part a
    # human reads — the grader's findings, the branch, and when the grader
    # judged the premise itself unsound, a tag that makes it findable in a
    # 380-deep draft pile.
    review = _review_note(events, round_id)
    if review is not None:
        if review.get("review_premise_unsound"):
            B.note_item(candidate.id,
                        f"automod review judged the premise unsound (round {round_id}): "
                        f"{str(review.get('review_summary') or review.get('detail') or '')[:600]}")
            B.tag_item(candidate.id, add=("review-premise",))
        else:
            B.note_item(candidate.id,
                        f"automod review sent round {round_id} back: "
                        f"{str(review.get('review_findings') or review.get('detail') or '')[:800]} "
                        f"— work is on branch `automod/{round_id}`")
    claimed = B.parse_spawned_line(run.get("text") or "")
    spawned, merged = B.split_claimed(claimed, id_floor=id_floor, self_id=candidate.id)
    # Findings are counted off the item file, not the report.
    after = B.load_item(candidate.path)
    findings_appended = B.count_findings(body_before, after.body if after else body_before)
    # Recorded, not enforced — the items are on disk before this line runs.
    # Worth a warning of its own when the round landed nothing: that is the
    # shape that converts one problem into six and solves none.
    over_cap = max(0, len(spawned) - SPAWN_CAP)
    if over_cap:
        logger.warning("backlog #%s filed %d item(s) over the cap of %d%s",
                       candidate.id, over_cap, SPAWN_CAP,
                       " while landing nothing" if not (round_id or vault_commits) else "")
    S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                    "phase": "finished", "session_id": run["session_id"],
                    "spawn_cap": SPAWN_CAP, "spawned_over_cap": over_cap,
                    "round_id": round_id, "vault_commits": vault_commits,
                    "surface": triage.get("surface") or "code",
                    "stop_reason": run.get("stop_reason"),
                    "num_turns": run.get("num_turns"),
                    "spawned": spawned, "merged": merged, "id_floor": id_floor,
                    "findings_appended": findings_appended,
                    "spawned_unverified": [i for i in claimed if i not in spawned
                                           and i not in merged and i != candidate.id],
                    # The finalizer's verdict on the acceptance check, and why
                    # there is none when there is none — a finalizer that
                    # quietly stopped working must not look like one working.
                    "outcome": outcome, "outcome_error": outcome_error,
                    "finalizer_tokens": run.get("finalizer_tokens"),
                    "response_tail": (run.get("text") or "")[-1500:]})
    outcome = (f"round {round_id}" if round_id else
               f"vault commit {vault_commits[-1][:8]}" if vault_commits else "no round opened")
    logger.info("backlog #%s: %s (session %s, %s)", candidate.id, outcome,
                run["session_id"], run.get("stop_reason"))
    # After `finished` too: `implement_outcomes` judges an item by its latest
    # row, and before that row is written the latest is `started`, whose
    # detail is empty — so this escalation, placed above it, never fired
    # (zero `review_escalated` rows on the ledger ever).
    try:
        await asyncio.to_thread(_escalate_review_disagreement, candidate, round_id)
    except Exception as exc:  # noqa: BLE001 — bookkeeping never fails a round
        logger.warning("#%s: review escalation failed: %s", candidate.id, exc)
    # After `finished`, which is the row the reaper reads. Not on the
    # `infra_failed` branch above: a stream that dropped says nothing about
    # whether the backend is still running the turn, so that round waits for
    # housekeeping's pass.
    await _reap_at_turn_end(run["session_id"])
    return {"status": "success", "item_id": candidate.id, "round_id": round_id,
            "session_id": run["session_id"], "stop_reason": run.get("stop_reason"),
            "summary": f"#{candidate.id}: {outcome}"}
