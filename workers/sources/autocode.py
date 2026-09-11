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

import logging
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

# Same cap and the same reason as autotriage.SPAWN_CAP. An implement
# round ran hotter than triage did — 17 items over 6 runs, and the three
# that aborted at the gate on 2026-09-08 filed 3, 6 and 5 while landing
# nothing. A round that cannot make its own change true is the last one
# that should be growing the board.
SPAWN_CAP = 3
# 150, not 100. Round SM_20260908_165950 called `automod_land` at iteration 91
# of 100 — nine left for a landing that must be followed by an immediate turn
# end, and #278 died at 101 with a gated, ready change it never landed. The
# budget anchor fires at 75% and 90%, so a cap that is too tight spends its
# warnings during normal work and has nothing left for the real deadline. The
# cost of a larger cap is bounded by `max_duration_seconds` either way.
DEFAULT_MAX_TURNS = 150

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

PROMPT = """\
Backlog item #{item_id} on your own board was triaged {triaged_ago} and \
**confirmed**: the premise still holds and there is real work here. Implement \
it through the self-modification loop, following `automod-change-own-code` \
exactly.

<item id="{item_id}" status="{status}" priority="{priority}">
# {name}

{body}
</item>

<triage>
Verdict: confirmed
Surface: {surface}
Check that was run: {check}
Evidence: {evidence}
</triage>

**The acceptance check, recorded at triage, is your contract:**

    {acceptance}

The round is done when that has become true and a test pins it. If you cannot \
make it true with one small, well-tested change, do not land a larger one — \
abort the round, say why, and the item goes back to a human.

**Scope you discover is not scope you take.** The work will show you things \
the acceptance check does not cover — a second bug beside the first, a \
refactor the fix wants, a test the area is missing, a premise in the item that \
turned out wider than its check. Each one becomes **its own backlog item, filed \
by you** with `backlog_write_task` (board `lloyd`, no `task_id`, tag \
`spawned-by-autocode`, first line "Found while implementing #{item_id}"), \
written as a handoff a fresh session can execute alone: what is wrong, where \
(file paths and line numbers), and how to verify. Then keep this round to the \
contract. One change per round is what makes a rollback mean something. Do not \
fold the discovery into this change, and do not leave it in your report — the \
report is read once; the backlog is read until the item is done.

**File at most {spawn_cap}.** If the round turned up more than that, file the \
{spawn_cap} that block or change the next piece of work, and put the rest in \
**one** item titled "Further findings from implementing #{item_id}" with the \
same per-finding detail. Nothing is dropped; the fan-out is. If you are about \
to file more than {spawn_cap} while *not* landing a change, that is the signal \
to stop and report instead — a round that aborts and files six items has \
converted one problem into six and solved none.

{reoffer}**The triage evidence above was measured today, on this tree.** File sizes,
line counts, git shas and grep results in it are current: read them, do not
re-derive them. Re-measure exactly one thing — the acceptance check, which you
must confirm fails before you start and passes when you finish. Round
SM_20260908_165950 spent 27 of its 92 iterations re-establishing facts the
triage had already stated before it opened its round.

Procedure when the surface is `code` or `frontend`:
1. Re-read the item and the triage evidence. If anything has changed since the \
triage and the premise no longer holds, say so and stop — that is a result.
2. `automod_start` with a goal naming item #{item_id}. Work only in the \
worktree it returns. The frontend is in scope: `web/src/**`, `web/index.html` \
and `web/public/**` are writable and the gate type-checks and builds them; \
`package.json`, the lockfile and the Vite/TS config are not.
3. **Map the blast radius before you edit.** `graph_refresh(root=<worktree>)`, \
then `graph_affected(symbol, root=<worktree>)` for each symbol you are about \
to change and `graph_explain` for its callers. Pass `root=` every time — the \
default is the live checkout, not your worktree. Record the depth-1 callers \
and the file list in your report; they are what tells you whether this is a \
one-file change or a five-file one, and a grep for the symbol's spelling will \
not tell you.
4. Write the test that fails today. Then the smallest change that makes it \
pass. One change per round.
5. `automod_gate`. If it fails twice on the same rung for the same reason, \
`automod_abort` and report — with one exception, below. A preflight that says \
it **rebased** is a pass, not a warning: something landed on `main` under you \
and the gate moved your branch onto it and retested. Your base has moved; do \
not re-cut. Only a rebase *conflict* stops you, and it names the files — \
resolve in the worktree, commit, gate again.

**An existing test that fails because it pins the behaviour you were asked to \
change is work, not a blocker.** The gate reports whether a failure is new in \
this round or predates it; a *new* failure in a test you did not write is the \
case to look at rather than abort on. Read the test. If it asserts the old \
behaviour and the item's acceptance says that behaviour is wrong, updating it \
is part of the change — and then your report must name the test, quote the \
assertion you changed, and say which line of the acceptance check makes it \
wrong. If you cannot write that sentence, the test is catching your bug and \
the correct move is to fix the code. Never delete a test to get to green (the \
gate refuses it), never add a skip, and never weaken an assertion you cannot \
justify in those terms.
6. `automod_land`. Then **end your turn immediately** — the landing needs the \
backend idle, and your own turn is what keeps it busy.

Procedure when the surface is `vault`:
1. Re-read the item and the triage evidence, as above.
2. Edit the files directly under `~/obsidian` — the vault is a live tree and \
has no worktree. Touch only the paths the acceptance check names. **Those \
paths are pre-authorised and you do not need to ask**: the confirmation SOUL.md \
requires for a protected path is what the item's triage verdict already \
recorded, and `automod_vault_land` supplies the rest of what confirmation is \
for — it validates through the real loaders and commits one revertable sha. \
Any vault path the acceptance check does NOT name is still protected, and \
wanting to touch one is a reason to stop and file an item, not to widen the \
round.
3. Verify the acceptance check yourself, then \
`automod_vault_land(paths, message, item_id={item_id})`. It validates exactly \
those paths (front matter, and for skills, tasks and identity files the real \
loaders), commits them on the vault's main and records the sha. If it refuses, \
it has already reverted your edits: fix the cause and retry once, or stop and \
report. Nothing restarts, so your turn continues.

If the surface is `mixed`, land the vault half first, then run the code \
procedure, and end your turn after `automod_land`.

**If the change touches the prompt surface** — `prompt_builder.py`, \
`prefetch.py`, or `lloyd/SOUL.md` / `lloyd/MEMORY.md` / `lloyd/USER.md` in the \
vault — a scored behavioural check is required, and these are the commands. \
Run them from `~/lloyd`, not from the worktree: the eval writes its baseline \
next to the script, and a baseline written inside a worktree is deleted with \
it (SM_20260908_165950's was).

    cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.autoresearch.bench_runner_sdk \
        --task bench_006_contradiction_check --task bench_008_adversarial_gap \
        --task bench_009_adversarial_probe --task bench_010_safety_destructive --judge
    cd ~/lloyd && .venvs/lloyd/bin/python eval/run_tool_choice_eval.py --label {round_label}
    cd ~/lloyd && .venvs/lloyd/bin/python eval/compare_tool_choice.py --label {round_label}

`--judge` is what produces a score; without it every composite comes back \
`null` and you are reading final text by eye. `compare_tool_choice.py` names \
the prior run it compared against and exits non-zero on a regression — quote \
its output. Exit 2 means it had nothing to compare against, which is not a \
pass.

**Your outcome closes the item — or leaves it open.** When this turn ends you \
will be asked to restate the result as one JSON object: whether the change \
landed, and whether the acceptance check above is now `met`, `not_met`, or \
`deferred`. Once the promotion settles, an item whose round said `met` is \
closed automatically. `deferred` leaves it open and names the ids it waits on \
— that is the honest answer when the check needs traffic, a nightly run, or \
another item to close first; file that item and name it. `not_met` leaves it \
open. `unnecessary` means the work is not needed after all — the premise no \
longer holds, or the acceptance is already true — and closes the item without a \
landing; say so in the summary. A closed item is never re-triaged, so `met` (or \
`unnecessary`) on a check you did not actually verify is the one claim this loop \
cannot recover from.

Report what you did, quoting the gate line rather than saying "it passed", \
and end with one line `SPAWNED: <ids of the items you filed, or the word none>`. \
Work autonomously; do not ask for confirmation.
"""


ABANDON_GRACE_SECONDS = 20 * 60


def _reoffer_block(reason: str) -> str:
    """The banner an item gets when it is being offered again."""
    if not reason:
        return ""
    return (f"**This item is being offered again — its previous round never reached a "
            f"verdict on the change ({reason}).** Read what is there before you start: "
            f"if a branch is named, `git log`/`git diff` it against your new base and "
            f"reuse what still applies rather than rewriting it. Say in your report "
            f"what you reused and what you redid.\n\n")


def reap_abandoned_rounds(now: float | None = None) -> list[dict]:
    """Close rounds this source opened that nobody finished — after a grace
    period, and only while nothing is happening in their session.

    Not at turn end, and the reason is #278. Its implement turn died at the
    100-iteration cap with the change written, tested and un-gated, and the
    ledger row said `max_turns`. Two minutes later the Inner Voice observer
    queued an ambient follow-up into that same session — "the round ended at
    the cap with no report; if the gate passed, land it" — and that second
    turn gated, landed, and the whole feature was live at 17:34. A reaper
    that aborted at turn end would have raced the rescue and thrown away
    875 lines the gate then passed. So the observer is the first responder
    and this is the backstop: a round the implementer opened, still open,
    nothing under observation, its session idle, `ABANDON_GRACE_SECONDS`
    after the turn ended. The branch is kept — it is the only record of
    what was attempted — and the item is told where it is.
    """
    from scripts.automod import backlog as B, round as R, state as S, worktree as W
    now = now or time.time()
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
    except Exception:
        busy = set()
    current = S.read_current() or {}
    reaped: list[dict] = []
    for e in finished:
        rid = e["round_id"]
        if rid in closed or current.get("round_id") == rid:
            continue
        age = now - float(e.get("ts") or 0)
        if age < ABANDON_GRACE_SECONDS or e.get("session_id") in busy:
            continue
        if not W.worktree_path(rid).exists():
            continue
        R.abort(rid)
        rec = {"event": "round_abandoned", "round_id": rid, "item_id": e.get("item_id"),
               "branch": f"automod/{rid}",
               "reason": (f"implement turn ended ({e.get('stop_reason')}) and the round "
                          f"stayed open for {int(age // 60)} min with nothing running in "
                          f"its session")}
        S.append_event(rec)
        if e.get("item_id") is not None:
            B.note_item(int(e["item_id"]),
                        f"automod round {rid} abandoned: {rec['reason']}. Its work is on "
                        f"branch `automod/{rid}` in ~/lloyd.")
            B.set_status(int(e["item_id"]), "up_next", "its round was abandoned; back in the pool")
        logger.warning("reaped abandoned round %s (%s)", rid, rec["reason"])
        reaped.append(rec)
    return reaped


def _loop_is_free() -> tuple[bool, str]:
    """Every gate the loop itself enforces, checked here first so a queued
    item does not spend a full agent turn discovering it cannot proceed."""
    from scripts.automod import state as S, worktree as W

    if not S.is_enabled(LIVE_ROOT):
        return False, "automod.enabled is false"
    if S.is_halted():
        return False, "promotions are halted"
    if S.is_broken():
        return False, "guardian is BROKEN"
    current = S.read_current()
    if current:
        return False, (f"promotion {str(current.get('commit'))[:8]} is under "
                       f"observation ({current.get('state')})")
    if S.read_rollback_request():
        return False, "a rollback request is pending"
    worktrees = W.prune_orphans(LIVE_ROOT)
    if len(worktrees) > 1:
        return False, f"a round is already open ({len(worktrees) - 1} worktree(s))"
    return True, "free"


def _age_phrase(ts: float | None) -> str:
    if not ts:
        return "recently"
    days = max(0, int((time.time() - float(ts)) // 86400))
    return "today" if days == 0 else f"{days} day{'s' if days != 1 else ''} ago"


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
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
                                       enabled=bool(src_cfg.get("close_on_settle", True))):
            logger.info("backlog #%s landed: %s (acceptance=%s)", r["item_id"],
                        "closed" if r["closed"] else "noted, left open", r["acceptance"])
    except Exception as exc:
        logger.warning("close_settled_items failed: %s", exc)
    try:
        # Status is the pipeline's state machine and the ledger is its source
        # of truth: every poll, anything the two disagree on moves. This is
        # also what migrated the board on 2026-09-09.
        for r in B.reconcile_statuses(S.LEDGER_PATH,
                                      enabled=bool(src_cfg.get("status_pipeline", True))):
            logger.info("backlog #%s: %s → %s", r["item_id"], r["from"], r["to"])
    except Exception as exc:
        logger.warning("reconcile_statuses failed: %s", exc)
    free, why = _loop_is_free()
    if not free:
        logger.info("autocode: not queueing — %s", why)
        return
    if B.select_confirmed(S.LEDGER_PATH) is None:
        return
    new_id = queue.enqueue(
        source=NAME, kind="round",
        payload={"max_turns": int(src_cfg.get("max_turns", DEFAULT_MAX_TURNS)),
                 # Carried in the payload like the budget, so a queued item runs
                 # under the config that was live when it was enqueued.
                 "structured_outcome": bool(src_cfg.get("structured_outcome", True))},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued backlog implement id=%d", new_id)


def _round_opened_since(events: list[dict], since_ts: float) -> str | None:
    for ev in reversed(events):
        if ev.get("event") == "round_start" and float(ev.get("ts") or 0) >= since_ts:
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
        return await _run_and_record(item, candidate, triage, budget, started)
    finally:
        # Every exit — landed, aborted, timed out, never ran, skipped for a
        # drain — hands the item back to the ledger's verdict. Without this the
        # early returns left an item `in_progress` that nothing would ever
        # pick up again, which is the old failure wearing a new status.
        try:
            B.reconcile_statuses(S.LEDGER_PATH)
        except Exception as exc:
            logger.warning("reconcile_statuses after #%s failed: %s", candidate.id, exc)


async def _run_and_record(item, candidate, triage, budget, started) -> dict[str, Any]:
    from scripts.automod import backlog as B, state as S
    from workers.sources._common import DrainActive, TurnTimeout, run_prompt_in_session
    logger.info("implementing backlog #%s (budget %d): %s",
                candidate.id, budget, candidate.name[:70])

    prompt = PROMPT.format(
        item_id=candidate.id, status=candidate.status, priority=candidate.priority,
        name=candidate.name, body=candidate.body[:30_000],
        triaged_ago=_age_phrase(triage.get("ts")),
        surface=triage.get("surface") or "code",
        check=triage.get("check") or "(none recorded)",
        evidence=(triage.get("evidence") or "(none recorded)")[:2000],
        acceptance=triage.get("acceptance") or "",
        spawn_cap=SPAWN_CAP,
        # A label the eval comparer can select on, unique per item and stable
        # across the turn's retries. `--label item377` reads back as the run
        # that judged item 377, months later, in a directory of bare
        # timestamps.
        round_label=f"item{candidate.id}",
        # A re-offer is not a fresh start. The previous round's branch may
        # still hold the work, or a landing may have been reverted, and a
        # round told nothing re-derives it — or redoes it.
        reoffer=_reoffer_block(B.reoffer_reason(S.LEDGER_PATH, candidate.id)),
    )
    want_outcome = bool((item.payload or {}).get("structured_outcome", True))
    try:
        run = await run_prompt_in_session(
            prompt, title=f"autocode #{candidate.id}: {candidate.name[:48]}",
            source=NAME, max_turns=budget, priority=1,
            final_schema=B.IMPLEMENT_OUTCOME_SCHEMA if want_outcome else None,
            final_schema_prompt=(
                "Restate the result of this round as a single JSON object matching "
                "the schema: whether the change landed, and whether the acceptance "
                "check recorded at triage is now met, not_met, or deferred (with the "
                "ids it waits on). Same filed ids as your SPAWNED line. This is a "
                "transcription of what you already reported, not a re-decision — and "
                "`met` closes the item once the promotion settles, so say what is true."
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
        S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                        "phase": "finished", "reason": str(exc),
                        "round_id": _round_opened_since(S.read_events(limit=200), started),
                        "stop_reason": "turn_timeout"})
        logger.warning("backlog #%s: %s", candidate.id, exc)
        return {"status": "failed", "item_id": candidate.id,
                "summary": f"#{candidate.id}: {exc}"}

    events = S.read_events(limit=200)
    round_id = _round_opened_since(events, started)

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
    vault_commits = [e.get("commit") for e in events
                     if e.get("event") == "vault_land" and e.get("ok")
                     and e.get("item_id") == candidate.id
                     and float(e.get("ts") or 0) >= started]
    outcome = B.parse_outcome(run.get("structured")) if want_outcome else None
    outcome_error = str(run.get("structured_error") or "")
    if outcome and outcome["acceptance"] == "unnecessary":
        # "Determined not to be necessary" is a verdict with no landing to wait
        # for: the premise no longer holds, or the acceptance is already true.
        # Closed here rather than by the settle sweep, which only sees landings.
        B.set_status(candidate.id, "done",
                     "the round found the work unnecessary" + (f": {outcome['summary']}" if outcome.get("summary") else ""))
        S.append_event({"event": "item_closed", "item_id": candidate.id, "by": "autocode",
                        "acceptance": "unnecessary", "reason": outcome.get("summary", "")[:300]})
    claimed = B.parse_spawned_line(run.get("text") or "")
    spawned = B.existing_ids(claimed)
    # Recorded, not enforced — the items are on disk before this line runs.
    # Worth a warning of its own when the round landed nothing: that is the
    # shape that converts one problem into six and solves none.
    over_cap = max(0, len(spawned) - (SPAWN_CAP + 1))
    if over_cap:
        logger.warning("backlog #%s filed %d item(s) over the cap of %d(+1)%s",
                       candidate.id, over_cap, SPAWN_CAP,
                       " while landing nothing" if not (round_id or vault_commits) else "")
    S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                    "phase": "finished", "session_id": run["session_id"],
                    "spawn_cap": SPAWN_CAP, "spawned_over_cap": over_cap,
                    "round_id": round_id, "vault_commits": vault_commits,
                    "surface": triage.get("surface") or "code",
                    "stop_reason": run.get("stop_reason"),
                    "num_turns": run.get("num_turns"),
                    "spawned": spawned,
                    "spawned_unverified": [i for i in claimed if i not in spawned],
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
    return {"status": "success", "item_id": candidate.id, "round_id": round_id,
            "session_id": run["session_id"], "stop_reason": run.get("stop_reason"),
            "summary": f"#{candidate.id}: {outcome}"}
