"""Settle what an item still owes — Lloyd decides, nothing waits on Alan.

Every item with a due entry on its `owed` list (scripts/automod/owed.py) gets
one visible session: it reads the item, measures each owed entry against the
live system (run records, logs, the ledger, the code), and answers per entry
with one of `owed.OUTCOMES`. The apply step writes the answer: evidence onto
the item, a recheck date, a ruling, a draft follow-up item, another implement
attempt, or a close. `outside` is only for what no software can do — sudo on
the host, a secret Alan holds, hardware, money — and lands on the one list
Mission Control shows, never as a tag.

Why this exists: on 2026-09-27 a hand sweep of 257 closed `needs-human` items
found 95 already done and 30 moot (post-landing checks nobody re-ran), 20
waiting on a date, 68 policy calls Alan had delegated anyway, 63 pieces of
leftover work written as prose on closed items, and 10 that needed his hands.
Alan's ruling the same day: "i don't want any more needs-human … lloyd can
approve his own choices now." This job is that sweep, run continuously.

Bounds: one item per job, up to `batch` jobs per tick (`DEFAULT_BATCH`, and
one queue row per item); `max_turns`; the session may read and run commands
but not edit files or write the board (the apply step is the only writer);
at most `spawn_cap` follow-up items per item; a recheck is clamped to 30 days
and an entry rechecked `owed.MAX_RECHECKS` times is ruled on, not rescheduled.
With `apply: false` it records its answer in the ledger and writes nothing.

Why the tick takes a batch and not one item (#1909): on 2026-09-30 the board
held 191 due entries across 76 items, a tick that reached one item settled
maybe two entries, and 92 new entries were filed the same day — the job could
not drain faster than the loop filled it, so four hours of hand sweeping closed
in one morning what the job had not reached in a day. One item per *session*
stays: the bound that moved is items per tick, never entries per session.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from workers.queue import QueueItem, WorkQueue

logger = logging.getLogger("lloyd-workers.owed-check")

NAME = "owed-check"
# Behind the steward (68): settling owed work can always wait a pass.
DEFAULT_PRIORITY = 72
LONG_LIVED = True
DEDUP_KEY = "owed-check:item"
#: Items one tick offers to the queue. The bound that ever mattered was items per
#: SESSION — one session measures and rules on one item's due entries — and it
#: stays. What #1909 moved is items per TICK: on 2026-09-30 the board held 191 due
#: entries across 76 items, a tick that reached one item settled maybe two entries,
#: and 92 new entries were filed the same day, so the pool grew while the job ran
#: clean. A hand sweep cleared in two hours what the tick had not reached in a day.
#: Overridable per source config as `batch`; `batch: 1` is the old behaviour.
DEFAULT_BATCH = 3
DEFAULT_MODEL = "primary"
DEFAULT_MAX_TURNS = 40
DEFAULT_SPAWN_CAP = 3
BODY_CHARS = 14_000
#: parse_answer's ceiling on evidence/ruling, far above
#: `owed.SETTLED_TEXT_LIMIT` so the writer's marked cut is the only cut.
_FIELD_CEILING = 6000

OWED_FIELD_MAX = 6500          # slice [:6000] = `_FIELD_CEILING`, owed_check.py:63
OWED_ARTIFACT_MAX = 450        # slice [:300]; a repo-relative path, longest real one is 100 chars
OWED_RECHECK_MAX = 80          # slice [:40]; an ISO date
OWED_OUTSIDE_MAX = 600         # slice [:500]
OWED_FOLLOWUP_NAME_MAX = 200   # slice [:140]; a backlog item name
OWED_FOLLOWUP_BODY_MAX = 6500  # slice [:6000]; `BODY_CHARS` for a new item is 14,000
OWED_CLAUSE_TEXT_MAX = 2200    # slice [:2000]; one rewritten acceptance clause
OWED_SUMMARY_MAX = 600         # slice [:400]; on record 400, hit on 528 of 1176 rows

# Bounded grammar (#2444): every string leaf carries one of the caps above, each
# strictly above the `[:N]` `parse_answer` applies to the same field (summary 600 over
# [:400], evidence/ruling 6500 over the [:6000] `_FIELD_CEILING`, the rest over their
# own slices), so the reader stays the cut that decides what is stored and a
# degenerate owed pass lands on the bounded branch — "generation diverged at N tokens
# … not a budget" (`DIVERGENCE_MARKER`) with its one re-draw — rather than the
# unbounded branch's advice to raise `harness.finalizer.max_tokens` (8192), #1706's
# false lead. On record (measured 2026-10-08): 528 of the 1176 owed_check rows that
# carry a summary sit at its stored 400, and none stores a per-entry string, so the
# entry caps are sized from their slices, not a measurement; owed-check #2444/1
# re-sizes from the at-the-cap ratio.
OWED_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "outcome": {"type": "string", "enum": ["settled", "recheck", "ruling", "work",
                                                       "reopen", "close", "outside"]},
                "evidence": {"type": "string", "maxLength": OWED_FIELD_MAX},
                "ruling": {"type": "string", "maxLength": OWED_FIELD_MAX},
                "recheck_after": {"type": "string", "maxLength": OWED_RECHECK_MAX},
                "follow_up": {"type": "object", "properties": {
                    "name": {"type": "string", "maxLength": OWED_FOLLOWUP_NAME_MAX},
                    "body": {"type": "string", "maxLength": OWED_FOLLOWUP_BODY_MAX}}},
                "outside": {"type": "string", "maxLength": OWED_OUTSIDE_MAX},
                "artifact": {"type": "string", "maxLength": OWED_ARTIFACT_MAX},
                "amend_clause": {"type": "object", "properties": {
                    "clause": {"type": "integer"},
                    "text": {"type": "string", "maxLength": OWED_CLAUSE_TEXT_MAX}}},
            },
            "required": ["n", "outcome", "evidence"]}},
        "summary": {"type": "string", "maxLength": OWED_SUMMARY_MAX},
    },
    "required": ["entries", "summary"],
}

PROMPT = """\
You are settling what backlog item #{id} still owes. The loop finished its part \
of this item; the entries below are what was left. Alan has delegated every one \
of these calls to you: nothing is waiting for him, and nothing may be handed \
back to him. Decide each entry. His standing ruling (2026-10-05): the loop goes \
with its own recommendation. Every time an item stalled for him he approved the \
action it recommended, so when the item or a round's findings name a \
recommended action, that IS the decision unless your measurement contradicts \
it. Only something physical waits for him.

<item>
#{id} — {name}
status: {status}   board: {board}
{body}
</item>

<owed>
{entries}
</owed>

For each numbered entry, first MEASURE: look at the live system — run records \
under ~/lloyd-data/autonomy-runs/, logs in ~/lloyd-data/logs/ (server.err), the \
automod ledger ~/.local/state/lloyd-automod/promotions.jsonl, the code in \
~/lloyd (git log, grep), the vault, other backlog items that superseded this \
one. Cite what you actually saw. Then answer with exactly one outcome:

- `settled` — it already happened, or it no longer applies (superseded, the \
code is gone, a later decision made it moot). The evidence is required.
- `recheck` — it can only be observed after a future event (a nightly run, a \
week of traffic). Give `recheck_after` as an ISO date when it becomes \
observable. If it is observable now, observe it instead.{recheck_note}
- `ruling` — a policy or design choice. Make the call yourself and write it in \
`ruling`: Alan prefers bounded guards over removing a capability, treats every \
change as a proposal deployed only on a measured gain ("tried, didn't work \
out" is a clean close), keeps safety layers (vault protection, the guardian, \
protected paths) in place, and does not want bulk rewrites without a measured \
gain. If the ruling leaves engineering work, add `follow_up`.
- `work` — concrete engineering is still owed. Give `follow_up` with a short \
`name` and a `body` that says what to change, why, and how to check it (at \
most six checkable clauses). It is filed as a draft backlog item.
- `reopen` — (open items only) the item deserves another implement attempt; \
say why in `ruling`. Only an item a round has already attempted, or whose \
triage confirmed a contract a round can meet, can be reopened: for one triage \
ruled `not_code` or `unverifiable` the answer is refused and the entry stays \
owed. If what such an item still needs is a measurement or a run, do it now, \
in this session, and rule on the result; if it needs a change, answer `work`.
  A review disagreement ("clause N came back unmet on two consecutive \
reviews") is yours to decide, not a person's: read both reviews' findings on \
the item. If the grader faulted the implementation, `reopen` or `close`. If \
the grader said the work is right and the clause is wrong as written (it names \
a path another item owns, a number that has since moved, a check that cannot \
run under the gate), `reopen` AND give `amend_clause` {{"clause": N, "text": \
"<the clause as it should read, still one checkable thing ending with the file \
that pins it>"}} — a reopen that leaves the clause alone repeats the refusal. \
  A round refused twice by a suite-wide rail (a skip ceiling, a collection \
floor, a preflight overlap) rather than by a finding on its diff has not been \
judged: check whether the rail still binds (the newest rounds' gate.json), and \
`reopen` when it does not.
- `close` — (open items only) close the item: tried and not worth another \
attempt, or not worth doing; say why in `ruling`.
- `outside` — ONLY for something physical or his alone: hardware to plug in or \
move, sudo on the host, a secret, login or account only Alan holds, spending \
money, posting under his name to a third party. Say exactly what he would do \
in `outside`. A protected path is NOT outside — `scripts/automod/**`, the \
guardian and the service units land through a round, with the rollback drill — \
and neither is `config.yaml` (comment edits land; so do value changes outside \
the fenced keys in `spec.CONFIG_DENIED_KEYS`): answer `work` or `reopen`. \
Deleting or moving data is NOT outside: rule on it, and file the change as \
`work` so a gated round does it. A decision is never outside: make it.

Evidence and rulings are stored up to 1200 characters. When one would run \
longer it is cut with a visible marker, so put the decision and its bounds \
first and name in `artifact` a path that already holds the detail (a run \
record, a session file, a log).

Do not edit files, write to the backlog, or change the system yourself — \
measure and decide; the job applies your answer. Finish by restating every \
entry's answer as the structured object you are asked for.
"""


def _entries_block(entries: list[dict], due: list[int]) -> str:
    rows = []
    for i, e in enumerate(entries):
        if i not in due:
            continue
        extra = f", rechecked {e['rechecks']}x" if e.get("rechecks") else ""
        # #2055: a clause can reach this list by derivation rather than by a
        # landing-time write, and a ruling has to be able to tell the two apart —
        # a stranded one has never been seen by any pass, so "already handled" is
        # not an available answer for it.
        from scripts.automod import owed as O
        if e.get("origin") == O.STRANDED_ORIGIN:
            extra += ", derived from human_clauses: never recorded as owed"
        if e.get("note"):
            extra += f"; last answer not applied: {e['note']}"
        rows.append(f"{i + 1}. [{e['kind']}, owed since {e.get('since') or '?'}{extra}] {e['what']}")
    return "\n".join(rows)


def build_prompt(owing) -> str:
    from scripts.automod import owed as O
    item = owing.item
    body = (item.body or "").strip()
    if len(body) > BODY_CHARS:
        body = "…(earlier text trimmed)…\n" + body[-BODY_CHARS:]
    near_cap = any(owing.entries[i].get("rechecks", 0) >= O.MAX_RECHECKS - 1 for i in owing.due)
    return PROMPT.format(
        id=item.id, name=item.name, status=item.status, board=item.board or "lloyd",
        body=body, entries=_entries_block(owing.entries, owing.due),
        recheck_note=(f" An entry already rechecked {O.MAX_RECHECKS - 1} times gets no "
                      f"further recheck: rule on it." if near_cap else ""))


def parse_answer(structured: Any, due_numbers: set[int]) -> dict | None:
    """Validated answers for due entries only, or None."""
    from scripts.automod import owed as O
    if not isinstance(structured, dict):
        return None
    answers: list[dict] = []
    for raw in structured.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        try:
            n = int(raw.get("n"))
        except (TypeError, ValueError):
            continue
        outcome = str(raw.get("outcome") or "").strip()
        if n not in due_numbers or outcome not in O.OUTCOMES:
            continue
        follow = raw.get("follow_up") if isinstance(raw.get("follow_up"), dict) else {}
        amend = raw.get("amend_clause") if isinstance(raw.get("amend_clause"), dict) else {}
        try:
            amend_n = int(amend.get("clause"))
        except (TypeError, ValueError):
            amend_n = 0
        amend_text = " ".join(str(amend.get("text") or "").split())[:2000]
        answers.append({"n": n, "outcome": outcome,
                        "amend_clause": ({"clause": amend_n, "text": amend_text}
                                         if outcome == "reopen" and amend_n > 0 and amend_text
                                         else {}),
                        # Never cut below the writer's bound: `O._bounded`
                        # makes the one visible cut (#1955). This is only a
                        # ceiling on a runaway answer.
                        "evidence": " ".join(str(raw.get("evidence") or "").split())[:_FIELD_CEILING],
                        "ruling": " ".join(str(raw.get("ruling") or "").split())[:_FIELD_CEILING],
                        "artifact": " ".join(str(raw.get("artifact") or "").split())[:300],
                        "recheck_after": str(raw.get("recheck_after") or "")[:40],
                        "outside": " ".join(str(raw.get("outside") or "").split())[:500],
                        "follow_up": {"name": " ".join(str(follow.get("name") or "").split())[:140],
                                      "body": str(follow.get("body") or "")[:6000]}})
    if not answers:
        return None
    return {"entries": answers,
            "summary": " ".join(str(structured.get("summary") or "").split())[:400]}


def _next_owings(count: int, skip: set[int] = frozenset()) -> list:
    """The next `count` items owing something, in the order `owing_items`
    already puts them in: an open item owed a decision first, then oldest owed
    entry first."""
    from scripts.automod import owed as O
    out = []
    for owing in O.owing_items():
        if owing.item.id in skip:
            continue
        out.append(owing)
        if len(out) >= max(1, int(count)):
            break
    return out


def _next_owing(skip: set[int] = frozenset()):
    owing = _next_owings(1, skip)
    return owing[0] if owing else None


def _dry_answered() -> set[int]:
    """Items a dry run already answered: with `apply: false` nothing settles,
    so without this the same item would be offered every tick."""
    from scripts.automod import state as S
    return {int(e["item_id"]) for e in S.read_events(limit=20000)
            if e.get("event") == "owed_check" and e.get("apply") is False and e.get("ok")
            and str(e.get("item_id") or "").isdigit()}


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    """Offer up to `batch` items to the queue, in `owing_items` order.

    Each row carries the `item_id` it was made for: once a tick offers several, an
    executor that re-picked "the oldest owing item" at claim time would have every
    session in the batch settle the same item, and the second and third would find
    nothing owed. The dedup key is per item, which is also what the old board-wide
    key got wrong: it coalesced every retry into one row for whichever item was
    offered first (#1418).

    `max_inflight` still runs one session at a time; batching moves items per TICK,
    never entries per session — a session measures and rules on its one item's due
    entries, then the next row's turn comes.
    """
    apply = bool(src_cfg.get("apply", False))
    skip = set() if apply else await asyncio.to_thread(_dry_answered)
    batch = max(1, int(src_cfg.get("batch", DEFAULT_BATCH) or 1))
    offered = await asyncio.to_thread(_next_owings, batch, skip)
    for owing in offered:
        new_id = queue.enqueue(
            source=NAME, kind="item",
            payload={"item_id": owing.item.id,
                     "apply": apply,
                     "model": str(src_cfg.get("model", DEFAULT_MODEL)),
                     "max_turns": int(src_cfg.get("max_turns", DEFAULT_MAX_TURNS)),
                     "spawn_cap": int(src_cfg.get("spawn_cap", DEFAULT_SPAWN_CAP))},
            priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
            dedup_key=f"{DEDUP_KEY}:{owing.item.id}",
        )
        if new_id is not None:
            logger.info("Enqueued owed-check id=%d for #%d", new_id, owing.item.id)
    return None


async def execute(item: QueueItem) -> dict[str, Any]:
    from scripts.automod import owed as O, state as S
    from workers.sources._common import DrainActive, TurnTimeout, run_prompt_in_session

    p = item.payload or {}
    apply = bool(p.get("apply", False))
    target = p.get("item_id")
    owing = None
    if target:
        owing = next((o for o in await asyncio.to_thread(O.owing_items, None, due_only=False)
                      if o.item.id == int(target)), None)
    else:
        skip = set() if apply else await asyncio.to_thread(_dry_answered)
        owing = await asyncio.to_thread(_next_owing, skip)
    if owing is None:
        return {"status": "skipped", "summary": "nothing owed is due"}
    if not owing.due:
        owing.due = list(range(len(owing.entries)))

    prompt = build_prompt(owing)
    try:
        run = await run_prompt_in_session(
            prompt, title=f"owed-check #{owing.item.id} ({len(owing.due)} owed)", source=NAME,
            max_turns=int(p.get("max_turns", DEFAULT_MAX_TURNS)), priority=2,
            model=str(p.get("model", DEFAULT_MODEL)),
            # Measure and decide; the apply step is the only writer.
            extra_disallowed=["Edit", "Write", "Task", "backlog_write_task",
                              "automod_start", "automod_gate", "automod_land",
                              "automod_abort", "automod_vault_land"],
            final_schema=OWED_SCHEMA,
            final_schema_prompt=("Restate your answer for every numbered owed entry as one JSON "
                                 "object: entries (n, outcome, evidence, and ruling / "
                                 "recheck_after / follow_up / outside where they apply) and a "
                                 "one-sentence summary."),
        )
    except DrainActive as exc:
        return {"status": "skipped", "summary": str(exc)}
    except TurnTimeout as exc:
        return {"status": "failed", "summary": f"owed-check turn timed out: {exc}"}

    due_numbers = {i + 1 for i in owing.due}
    parsed = parse_answer(run.get("structured"), due_numbers)
    if parsed is None:
        # No answer: push the item back a day so one stubborn item cannot
        # hold the queue, and say so on the item.
        if apply:
            await asyncio.to_thread(O.defer, owing.item.path, owing.entries, owing.due,
                                    why="no structured answer this pass")
        S.append_event({"event": "owed_check", "item_id": owing.item.id, "apply": apply,
                        "ok": False, "error": str(run.get("structured_error") or "")[:300],
                        "session_id": run.get("session_id")})
        return {"status": "failed", "summary": f"#{owing.item.id}: no structured answer",
                "meta": {"session_id": run.get("session_id")}}

    result: dict = {}
    if apply:
        result = await asyncio.to_thread(
            O.apply_verdict, owing.item.path, owing.entries, parsed["entries"],
            item_id=owing.item.id, session_id=str(run.get("session_id") or ""),
            spawn_cap=int(p.get("spawn_cap", DEFAULT_SPAWN_CAP)))
        outside = [a for a in parsed["entries"] if a["outcome"] == "outside"]
        if outside:
            try:
                from scripts.automod import promote as P
                P.announce(f"#{owing.item.id} needs your hands",
                           "; ".join(a["outside"] or a["evidence"] for a in outside)[:400])
            except Exception:  # noqa: BLE001 — an announcement never fails the job
                pass
    S.append_event({"event": "owed_check", "item_id": owing.item.id, "apply": apply, "ok": True,
                    "outcomes": [{"n": a["n"], "outcome": a["outcome"]} for a in parsed["entries"]],
                    "result": {k: result.get(k) for k in ("remaining", "filed", "moved")},
                    "session_id": run.get("session_id"), "summary": parsed["summary"]})
    counts: dict[str, int] = {}
    for a in parsed["entries"]:
        counts[a["outcome"]] = counts.get(a["outcome"], 0) + 1
    verb = "applied" if apply else "proposed (dry run)"
    return {"status": "success",
            "summary": (f"#{owing.item.id}: {verb} "
                        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
                        + (f"; filed {result.get('filed')}" if result.get("filed") else "")
                        + (f"; item {result['moved']}" if result.get("moved") else "")
                        + f" — {parsed['summary']}"),
            "meta": {"session_id": run.get("session_id"), "item_id": owing.item.id}}
