"""The failure-issue ledger's schedule and handler (#2079).

`app/failure_ledger.py` is the library: signatures off the guardian's own
`normalize_message`, an append-only `failure_events` table, a `failure_issues` row
per family whose `first_seen` never moves, two detectors that emit finding ROWS,
and `dispatch_findings`, which enqueues at most one investigation per day. What it
deliberately does not have is a scheduler and a handler — a library cannot wake
itself up, and the previous round of this item shipped the library alone. That
left a seam with nothing across it: `dispatch_findings` writes a queue row with
`source='failure-ledger'`, `workers/pool.py` resolves `item.source` against
`SOURCE_REGISTRY`, finds no module by that name, and marks the row poisoned with
`unknown source failure-ledger` at `attempts = 0` — terminal, never retried. So
the ledger could file an investigation every day and none of them could ever run,
and the only thing that would have said so is a worker running against the live
queue, which is exactly what a unit test cannot do.

This module is that missing side of the seam:

- `NAME` is `app.failure_ledger.SOURCE`, imported rather than spelled again, so
  the two names cannot drift apart. `tests/test_failure_ledger_dispatch.py` says
  so and checks the consequence: a row enqueued by the library resolves in the
  registry to THIS module.
- `enqueue_if_due` is the scheduled pass: ingest the automod promotions ledger
  (rotated archives AND live — the live file alone holds 2 rows of a family whose
  16 earlier rows sit in `promotions-archive-202609.jsonl.gz`, so live-only would
  invent an onset weeks late), aggregate it, run both detectors, and dispatch.
  The dispatch IS the enqueue: the pool's scheduler does not poll "are there
  findings", it asks this source, which answers by returning None on a quiet day
  and by spending the day's one investigation on a loud one.
- `execute` runs the prompt the ledger wrote. The prompt names finding ids and a
  ±3-day onset window but cannot quote the rows, so the handler reads them out of
  the ledger and pastes them in before the turn starts. A dispatched run that had
  no way to read a sqlite file would otherwise report the weather, which is the
  failure `dispatch_findings` already refuses to spend the cap on.

Two things about how it runs are choices, not accidents:

- **`diagnosis` is derived, and derived from the ledger.** `dispatch_findings`
  rejects an empty one, so a caller has to answer "what is this run for" before
  anything is enqueued. Hard-coding a generic question would satisfy the check
  and defeat its purpose, so the sentence is built from the day's strongest
  finding: its kind, its occurrence count, and the window around its onset —
  which is what a person reading the run row a week later wants to see.
- **No session, and no Inner Voice.** The turn goes through
  `run_prompt_on_primary` like `session-distill` and `bench-mine`: a diagnosis
  that reads ledger rows and `git log` is arithmetic over a store, not a
  judgement about a person.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app import failure_ledger as fl
from app import paths
from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.failure-ledger")

#: The queue's source name, taken from the library that writes the row. If this
#: were a second literal, `dispatch_findings` could enqueue a name no one answers
#: and the poison would look like a typo rather than a missing registration.
NAME = fl.SOURCE
KIND = fl.KIND
#: Not a second number: `app/failure_ledger.DISPATCH_PRIORITY` is what the library
#: stamps on the row it writes, and this is what the scheduler considers before a
#: row exists. They are the same object, so the queue depth a person reads in Mission
#: Control and the roster line in `architecture/workers-jobs.md` §1 cannot drift
#: apart — which they had: the library omitted the argument, the queue applied its
#: own default of 50, and nine sources the docs said outrank this one did not.
#: Lower runs sooner: behind every stream that does work for someone (autocode 40,
#: research 70, owed-check 72), ahead of the instrument checks (bench-mine and the
#: frontend-probe canary, both 80).
DEFAULT_PRIORITY = fl.DISPATCH_PRIORITY
LONG_LIVED = False
#: One day's sweep is one queue row, so the queue's own dedup can never stack a
#: second investigation for a day the ledger already spent.
DEDUP_KEY = "failure-ledger:scheduled"

#: The ledger's own cap on how many records one prompt names. Read off the library
#: so the handler cannot widen the bound the dispatch test pinned.
MAX_RECORDS = fl.DISPATCH_MAX_RECORDS

#: An investigation reads ≤20 ledger rows and one window of `git log`, then writes
#: a few paragraphs. `session-distill` runs on 15 for less work; 12 is well clear.
DEFAULT_MAX_TURNS = 12
DEFAULT_TIMEOUT_SECONDS = 900

#: `failure_findings.kind` carries the literals the detectors write
#: (`app/failure_ledger.py:519,539`); a person reading the run row a week later
#: reads the diagnosis, so the sentence spells them out. An unknown kind falls
#: through as its own name rather than being labelled wrong.
KIND_LABELS = {"new_onset": "New onset", "growth": "Growth"}

WATERMARK_LAST_ENQUEUED = "last_enqueued"
#: One day is one pass. The pool may poll sooner; the watermark is what makes the
#: sweep idempotent *in time* rather than only in content (ingestion is
#: insert-only, but a re-read of a 27 MB ledger is still a re-read of 27 MB).
DEFAULT_INTERVAL_SECONDS = 24 * 3600


def default_promotions_paths() -> tuple[Path, list[Path]]:
    """The live promotions ledger and its rotated siblings, as the guardian writes them.

    Both halves matter and neither is optional: `ingest_promotions` takes
    `archives=` as a parameter precisely because a reader that opens only
    `promotions.jsonl` reports the guardian's `Service down, but no promotion to
    revert` family as 2 occurrences from 2026-09-24 when the archive puts its
    first at 2026-09-06T18:34Z. The glob is a glob because rotation is monthly —
    next month's `promotions-archive-202610.jsonl.gz` is ingested by the same code
    path with no edit here.
    """
    from scripts.automod import state as automod_state
    state_dir = Path(automod_state.STATE_DIR)
    live = Path(automod_state.LEDGER_PATH)
    archives = sorted(state_dir.glob("promotions-archive-*.jsonl.gz"))
    return live, archives


def derive_diagnosis(conn, day: str) -> str:
    """The one question the day's single run is sent to answer, from the ledger.

    Built from the undispatched finding with the most occurrences — the same one
    `dispatch_findings` puts first in its prompt — and phrased as a checkable
    claim about the onset window: occurrences, the day it started, and the
    ±3-day range the run is told to count commits in. Deliberately no signature
    text in here: the prompt names records by their integer finding ids, and a
    12-hex signature in a diagnosis would put hex runs that read as ids into the
    same prompt the dispatch test counts ids in.
    """
    row = conn.execute(
        "SELECT f.signature, f.kind, f.occurrences, i.first_seen "
        "FROM failure_findings f LEFT JOIN failure_issues i ON i.signature = f.signature "
        "WHERE f.detected_on = ? AND f.dispatched = 0 "
        "ORDER BY f.occurrences DESC, f.id LIMIT 1",
        (day,)).fetchone()
    if row is None:
        return ""
    kind_label = KIND_LABELS.get(str(row["kind"]), str(row["kind"]))
    first_seen = str(row["first_seen"] or "")
    window = fl.window_around(first_seen) if first_seen else None
    onset = (f" since {first_seen}" if first_seen else "")
    if window is None:
        return (f"{kind_label}: {int(row['occurrences'])} occurrence(s){onset}. "
                "Establish what the failures have in common.")
    return (
        f"{kind_label}: {int(row['occurrences'])} occurrence(s){onset}. "
        f"Establish what changed between {window['from']} and {window['to']} that "
        "made it start, and whether the same cause is still failing today.")


def sweep_and_dispatch(*, queue, conn=None, day: str | None = None,
                       now: datetime | None = None,
                       live: Path | str | None = None,
                       archives: list[Path] | None = None,
                       diagnosis: str | None = None) -> dict[str, Any]:
    """One scheduled pass: ingest → sweep → detect → dispatch. Returns the report.

    Sync on purpose — the sqlite work and the queue write are both blocking — so
    `enqueue_if_due` runs it on a thread. Every argument is overridable because
    the test that crosses the pool boundary has to seed a store and a queue of its
    own; production passes nothing and gets the live paths.

    `diagnosis=None` means "derive it from the day's strongest finding". Passing a
    string is the caller taking responsibility for the question, which is what
    `dispatch_findings`' empty-diagnosis refusal is there to make explicit.
    """
    own_conn = conn is None
    conn = fl.connect() if own_conn else conn
    try:
        when = now or fl.now_utc()
        day = day or fl.day_of(when)
        if live is None or archives is None:
            default_live, default_archives = default_promotions_paths()
            live = default_live if live is None else live
            archives = default_archives if archives is None else archives
        ingested = fl.ingest_promotions(conn, live=live, archives=archives)
        swept = fl.sweep(conn)
        new_onset = fl.detect_new_onset(conn, now=when, day=day)
        growth = fl.detect_growth(conn, day=day)
        findings = list(dict.fromkeys([*new_onset, *growth]))
        text = derive_diagnosis(conn, day) if diagnosis is None else diagnosis
        queue_id: int | None = None
        if findings and text:
            queue_id = fl.dispatch_findings(conn, day, queue=queue,
                                            diagnosis=text, now=when)
        elif findings:
            # Reachable only if a caller passes an empty diagnosis deliberately.
            # Loud, because silently skipping is how a cap gets spent on nothing
            # and nobody can tell afterwards.
            logger.warning("failure-ledger %s: %d finding(s) and no diagnosis — "
                           "nothing dispatched", day, len(findings))
        return {"day": day, "considered": ingested["considered"],
                "inserted": ingested["inserted"],
                # `sweep` reports the aggregate it wrote, not a mutation count:
                # the pass is an upsert over the whole store, so "how many
                # signatures, how many events behind them" is the honest pair.
                "signatures": swept["signatures"], "events": swept["events"],
                "new_onset": len(new_onset), "growth": len(growth),
                "findings": findings, "diagnosis": text, "queue_id": queue_id}
    finally:
        if own_conn:
            conn.close()


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    """Run today's sweep; a day with findings gets its one investigation enqueued.

    Returns None on every path, so the pool stamps the scheduler's full
    `interval_seconds` whether or not anything was dispatched — see the watermark
    note. Findings that the cap held back stay in `failure_findings` with
    `dispatched = 0`; the next pass sees them again, and `dispatch_findings` will
    not spend a second run on a day it already spent.
    """
    # One clock for the whole pass: the ledger's. `sweep_and_dispatch` already asks
    # `fl.now_utc()` for `when` and derives `day` from it, and the cap that decides
    # whether today gets an investigation is keyed on THAT day — so an interval
    # watermark stamped by a second, hand-rolled `datetime.now(...)` is the same
    # reading taken by a different clock. It has to be the same one, or the pass
    # can be "due" on the watermark's calendar while the cap has already spent a
    # different day, and a test that pins the sweep's day (`tests/
    # test_failure_ledger_dispatch.py`) cannot pin the interval with it.
    last = await asyncio.to_thread(queue.wm_get, NAME, WATERMARK_LAST_ENQUEUED)
    interval = float(src_cfg.get("min_interval_seconds", DEFAULT_INTERVAL_SECONDS))
    if last:
        try:
            age = (fl.now_utc()
                   - datetime.fromisoformat(last)).total_seconds()
        except ValueError:
            logger.warning("failure-ledger: unreadable watermark %r — sweeping anyway",
                           last)
            age = None
        if age is not None and age < interval:
            return None
    report = await asyncio.to_thread(
        sweep_and_dispatch, queue=queue,
        live=src_cfg.get("promotions_ledger") or None,
        archives=[Path(p) for p in src_cfg["promotions_archives"]]
        if src_cfg.get("promotions_archives") else None)
    await asyncio.to_thread(queue.wm_set, NAME, WATERMARK_LAST_ENQUEUED,
                            fl.now_utc().isoformat())
    logger.info("failure-ledger %s: %d finding(s), %d event(s) new, "
                "investigation queue_id=%s", report["day"], len(report["findings"]),
                report["inserted"], report["queue_id"])
    return None


def issue_rows_for_prompt(db_path: Path | str, signatures: list[str],
                          limit: int = MAX_RECORDS) -> list[str]:
    """The ledger rows behind `signatures`, rendered for the prompt.

    The dispatched prompt names finding ids; this is the substance those ids point
    at — first/last seen, occurrence count, and the first sample the ledger ever
    stored for the family. Bounded by `limit` for the same reason the prompt is: a
    run that is handed 200 rows answers about none of them.
    """
    conn = fl.connect(db_path)
    try:
        rows: list[str] = []
        for sig in list(signatures)[:limit]:
            issue = fl.get_issue(conn, sig)
            if issue is None:
                rows.append(f"- {sig}: no issue row (was the store rebuilt?)")
                continue
            rows.append(
                f"- {sig}: {issue['status']}, {int(issue['occurrences'])} "
                f"occurrence(s), first_seen {issue['first_seen']}, "
                f"last_seen {issue['last_seen']}; sample: {issue['sample_summary']}")
        return rows
    finally:
        conn.close()


async def execute(item: QueueItem) -> dict[str, Any]:
    """Run the day's investigation prompt on the primary and report what it said."""
    payload = item.payload or {}
    prompt = str(payload.get("prompt") or "").strip()
    day = str(payload.get("day") or "")
    if not prompt:
        # The library never writes an empty prompt; reaching here means a row was
        # edited by hand, and a turn with no instruction would return prose as if
        # it were a diagnosis.
        return {"status": "failed",
                "summary": f"{day or 'failure-ledger'}: payload carries no prompt",
                "meta": {"reason": "empty_prompt"}}

    db_path = Path(paths.FAILURE_LEDGER_DB)
    rows = await asyncio.to_thread(
        issue_rows_for_prompt, db_path,
        list(payload.get("signatures") or []),
        int(payload.get("max_records") or MAX_RECORDS))
    turn_prompt = prompt
    if rows:
        turn_prompt = (prompt
                       + "\n\nLedger rows for these findings (from "
                       + str(db_path) + "):\n" + "\n".join(rows))

    from workers.sources._common import run_prompt_on_primary
    max_turns = int(payload.get("max_turns")
                    or DEFAULT_MAX_TURNS)
    turn = await run_prompt_on_primary(turn_prompt, max_turns=max_turns, source=NAME,
                                       title=f"failure-ledger {day}")
    finding_ids = list(payload.get("finding_ids") or [])
    if not turn.ok:
        return {"status": "failed",
                "summary": (f"{day}: {turn.failure_summary()}"
                            f" ({len(finding_ids)} finding(s) were named)"),
                "meta": {"stop_reason": turn.stop_reason, "num_turns": turn.num_turns,
                         "finding_ids": finding_ids, "rows_included": len(rows)}}
    return {"status": "success",
            "summary": (f"{day}: {len(finding_ids)} finding(s) investigated over "
                        f"{turn.num_turns} turn(s), {len(rows)} ledger row(s) given"),
            "response": turn.text[:20000],
            "meta": {"finding_ids": finding_ids, "rows_included": len(rows),
                     "diagnosis": str(payload.get("diagnosis") or ""),
                     "window": payload.get("window") or {},
                     "num_turns": turn.num_turns}}
