"""Implement a backlog item that triage has confirmed, through one selfmod round.

This is the only automated path from a backlog item to landed code, and it is
deliberately the last link in a chain with two gates in front of it:

  1. `backlog-selfmod` must have reached `confirmed` on the item, WITH an
     acceptance check. An item confirmed without one is skipped here, not
     guessed at: "what must become true for this to be done" is the contract
     the round is held to, and a round with no contract cannot fail.
  2. The self-modification loop must be free: enabled, not halted, not BROKEN,
     no promotion under observation, no rollback pending, no round open. That
     is re-checked at run time, because a queue item can sit for a while.

Then one turn, in a real session, following `selfmod-change-own-code`. The
turn opens the round, does the work, gates it, and either lands it or aborts.
Landing runs detached and the turn ends, exactly as it does when a human
drives it — the three rounds that proved the loop worked were driven this way
by hand, and this is that procedure with the human replaced by a queue.

One attempt per item, unattended. Whatever the turn concludes — landed,
aborted, gate failed, decided the item was not worth it after all — is
recorded and the item is not picked again by this source. A second attempt is
a human's decision.

Why this goes through `/api/message/stream` rather than `run_query`: the
observer. `selfmod_start` refuses a turn with no Inner Voice attached, and the
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

logger = logging.getLogger("lloyd-workers.backlog-implement")

NAME = "backlog-implement"
DEFAULT_PRIORITY = 80
DEDUP_KEY = "backlog-implement:round"
DEFAULT_MAX_TURNS = 100

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

PROMPT = """\
Backlog item #{item_id} on your own board was triaged {triaged_ago} and \
**confirmed**: the premise still holds and there is real work here. Implement \
it through the self-modification loop, following `selfmod-change-own-code` \
exactly.

<item id="{item_id}" status="{status}" priority="{priority}">
# {name}

{body}
</item>

<triage>
Verdict: confirmed
Check that was run: {check}
Evidence: {evidence}
</triage>

**The acceptance check, recorded at triage, is your contract:**

    {acceptance}

The round is done when that has become true and a test pins it. If you cannot \
make it true with one small, well-tested change, do not land a larger one — \
abort the round, say why, and the item goes back to a human.

Procedure:
1. Re-read the item and the triage evidence. If anything has changed since the \
triage and the premise no longer holds, say so and stop — that is a result.
2. `selfmod_start` with a goal naming item #{item_id}. Work only in the \
worktree it returns.
3. Write the test that fails today. Then the smallest change that makes it \
pass. One change per round.
4. `selfmod_gate`. If it fails twice on the same rung for the same reason, \
`selfmod_abort` and report.
5. `selfmod_land`. Then **end your turn immediately** — the landing needs the \
backend idle, and your own turn is what keeps it busy.

Report what you did, quoting the gate line rather than saying "it passed". \
Work autonomously; do not ask for confirmation.
"""


def _loop_is_free() -> tuple[bool, str]:
    """Every gate the loop itself enforces, checked here first so a queued
    item does not spend a full agent turn discovering it cannot proceed."""
    from scripts.selfmod import state as S, worktree as W

    if not S.is_enabled(LIVE_ROOT):
        return False, "selfmod.enabled is false"
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
    from scripts.selfmod import backlog as B, state as S

    free, why = _loop_is_free()
    if not free:
        logger.info("backlog-implement: not queueing — %s", why)
        return
    if B.select_confirmed(S.LEDGER_PATH) is None:
        return
    new_id = queue.enqueue(
        source=NAME, kind="round",
        payload={"max_turns": int(src_cfg.get("max_turns", DEFAULT_MAX_TURNS))},
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
    from scripts.selfmod import backlog as B, state as S
    from workers.sources._common import DrainActive, run_prompt_in_session

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
    logger.info("implementing backlog #%s (budget %d): %s",
                candidate.id, budget, candidate.name[:70])

    prompt = PROMPT.format(
        item_id=candidate.id, status=candidate.status, priority=candidate.priority,
        name=candidate.name, body=candidate.body[:30_000],
        triaged_ago=_age_phrase(triage.get("ts")),
        check=triage.get("check") or "(none recorded)",
        evidence=(triage.get("evidence") or "(none recorded)")[:2000],
        acceptance=triage.get("acceptance") or "",
    )
    try:
        run = await run_prompt_in_session(
            prompt, title=f"selfmod #{candidate.id}: {candidate.name[:48]}",
            source=NAME, max_turns=budget, priority=1)
    except DrainActive as exc:
        S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                        "phase": "skipped", "reason": f"landing in progress: {exc}"})
        return {"status": "skipped", "summary": f"landing in progress: {exc}"}

    round_id = _round_opened_since(S.read_events(limit=200), started)
    S.append_event({"event": "backlog_implement", "item_id": candidate.id,
                    "phase": "finished", "session_id": run["session_id"],
                    "round_id": round_id, "stop_reason": run.get("stop_reason"),
                    "num_turns": run.get("num_turns"),
                    "response_tail": (run.get("text") or "")[-1500:]})
    outcome = f"round {round_id}" if round_id else "no round opened"
    logger.info("backlog #%s: %s (session %s, %s)", candidate.id, outcome,
                run["session_id"], run.get("stop_reason"))
    return {"status": "success", "item_id": candidate.id, "round_id": round_id,
            "session_id": run["session_id"], "stop_reason": run.get("stop_reason"),
            "summary": f"#{candidate.id}: {outcome}"}
