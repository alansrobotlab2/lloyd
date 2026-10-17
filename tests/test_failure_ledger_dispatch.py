"""#2079 — the seam where the ledger's queue row meets the worker pool.

`app/failure_ledger.py` + `tests/test_failure_ledger.py` prove the ledger can
enqueue an investigation. They stop at the library's own edge, which is exactly
where the previous round's review rung put its finger: `dispatch_findings` writes
a row with `source='failure-ledger'`, and `workers/pool.py` answers a claim with
`SOURCE_REGISTRY.get(item.source)`; when that misses it logs
`Unknown source … — marking poisoned` and calls `mark_failed(item.id, "unknown
source failure-ledger", 0)`. `attempts = 0` makes it terminal, so nothing retried
and nothing investigated, and every test in the suite stayed green because no unit
test claims a row off a live queue. A module docstring warning about it is not a
fix — the fix is a registered source, and the proof is a test that resolves the
row through the same registry the pool reads.

Registered ≠ scheduled: `WorkerPool._scheduler_pass` skips a source whose config
block is absent or `enabled: false`, and `config.yaml` is outside what the
self-modification loop may write. So these nodes pin the half that is code (a
claimed row resolves and executes) and one node states the remaining half in its
failure message rather than in prose nobody reads.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app import failure_ledger as fl  # noqa: E402
from app import paths  # noqa: E402
from workers.queue import WorkQueue  # noqa: E402
from workers.sources import SOURCE_REGISTRY  # noqa: E402
from workers.sources import _common, failure_ledger  # noqa: E402

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
DAY = fl.day_of(NOW)

#: `record_failure` collapses digits into `<n>`, so families are named by WORDS,
#: not by numbers — three `probe alpha` messages are one signature, and a second
#: word is a second family that cannot merge into the first.
WORDS = ("alpha", "beta", "gamma", "delta")

#: The clock the real scheduled pass runs on (#2893). `interval_seconds: 3600` with
#: `min_interval_seconds: 86400` lets exactly one sweep through per day, and the
#: watermark says when the last one was: `workers.db`
#: `watermarks('failure-ledger','last_enqueued')` is
#: `2026-10-17T02:38:29.088606+00:00`. #2893's triage read the three logged pass
#: instants off the worker log as 01:51:42, 02:03:27 and 02:38:29 UTC, the last
#: matching that watermark to the second. So the pass that owns day D is awake for
#: about the first two hours of D and never again. `NOW` above is midday — the one
#: clock shape that cannot see the bug this section exists for.
EARLY_PASS = datetime(2026, 10, 3, 2, 10, tzinfo=timezone.utc)
#: The day that pass has to review. Named by arithmetic on the pass's own day, not
#: by a typed literal, so the node and the library can never disagree about it.
YESTERDAY = fl.day_of(EARLY_PASS - timedelta(days=1))


@pytest.fixture(autouse=True)
def frozen_ledger_clock(monkeypatch):
    """Pin the ledger's one clock to `NOW` for every node in this file.

    The cap that decides whether a day gets an investigation is keyed on the day the
    pass derives from `app.failure_ledger.now_utc`, and every node here seeds its
    findings dated `NOW`. Unpatched, the pass takes its day from the machine instead:
    seeded on 2026-10-03, `test_a_finding_that_arrives_after_the_days_run_was_sent_
    stays_queued` was really asking "did the cap refuse a second run TODAY", which is
    true only while the wall clock agrees with the fixture. It passed the day it
    landed and failed at the next UTC midnight — the cap had not moved, the calendar
    had. Freezing the single function the ledger reads turns each node back into a
    statement about the cap and the seam rather than about the date it runs on.
    """
    monkeypatch.setattr(fl, "now_utc", lambda: NOW)


def seed_fresh_failures(conn, families: int = 1, occurrences: int = 3,
                        offset: int = 0) -> list[int]:
    """Emit `families` x `occurrences` failures inside the new-onset window.

    Timestamped 5 minutes before `NOW`, which is what makes `detect_new_onset`
    fire without anyone moving a detector's threshold to make a test pass.
    `offset` walks the word list so a second call seeds a DIFFERENT signature —
    without it a repeat call re-seeds the same family, which `_emit_finding`
    correctly refuses to double-count, and a test asking "what arrives after the
    cap?" would silently be asking "what re-arrives".
    """
    ts = fl.iso_utc(NOW - timedelta(minutes=5))
    for i in range(families):
        for n in range(occurrences):
            fl.record_failure(
                conn, logger_name="lloyd-worker",
                message=f"probe {WORDS[(offset + i) % len(WORDS)]} blew up in {n} ms",
                ts=ts, run_id=f"run-{offset}-{i}-{n}")
    fl.sweep(conn)
    onset = fl.detect_new_onset(conn, now=NOW, day=DAY)
    growth = fl.detect_growth(conn, day=DAY)
    return list(dict.fromkeys([*onset, *growth]))


def fixture_cfg(tmp_path) -> dict:
    """Config pointing at BOTH ledger fixtures, as the clause-3 reader needs them."""
    fixtures = Path(__file__).resolve().parent / "fixtures" / "failure_ledger"
    return {"promotions_ledger": str(fixtures / "promotions_live_rows.jsonl"),
            "promotions_archives": [str(fixtures / "promotions_archive_rows.jsonl")]}


def make_queue(tmp_path: Path) -> WorkQueue:
    return WorkQueue(tmp_path / "workers.db")


# --------------------------------------------------------------------- registry --

def test_the_name_the_library_writes_is_a_registered_source():
    """The pool resolves `item.source` in `SOURCE_REGISTRY`; this is that lookup.

    `NAME` is imported from `app.failure_ledger` rather than spelled in either
    place, so the assertion below cannot be satisfied by two literals that happen
    to look alike today and drift tomorrow.
    """
    assert fl.SOURCE in SOURCE_REGISTRY, (
        f"pool would mark every {fl.SOURCE} row poisoned: "
        f"unknown source {fl.SOURCE}")
    assert SOURCE_REGISTRY[fl.SOURCE] is failure_ledger
    assert failure_ledger.NAME == fl.SOURCE
    assert failure_ledger.KIND == fl.KIND


def test_a_row_the_library_enqueued_resolves_through_the_pools_own_lookup(
        tmp_path, monkeypatch):
    """Enqueue with the LIBRARY, then do to the row what `_worker_loop` does.

    Ends with the negative control: with the registration pulled, the same lookup
    on the same row misses — which is the state the base commit was in, and the
    proof that the assert above is a check and not a tautology.
    """
    conn = fl.connect(tmp_path / "failure_issues.sqlite")
    findings = seed_fresh_failures(conn)
    assert findings, "the fixture must produce a finding, or this proves nothing"
    queue = make_queue(tmp_path)
    qid = fl.dispatch_findings(
        conn, DAY, queue=queue,
        diagnosis="Establish whether the probe alpha onset lines up with a deploy.")
    conn.close()
    assert qid is not None

    item = next(it for it in queue.list_items(source=fl.SOURCE) if it.id == qid)
    assert item.state == "queued"
    assert item.kind == fl.KIND
    # workers/pool.py: a claim with no registry entry is poisoned at attempts=0.
    assert SOURCE_REGISTRY.get(item.source) is failure_ledger
    # The row carries the source's declared priority, not `WorkQueue.enqueue`'s
    # own default of 50 — a module that says 75 and lands at 50 outranks every
    # interactive source while claiming not to.
    assert item.priority == failure_ledger.DEFAULT_PRIORITY

    monkeypatch.delitem(SOURCE_REGISTRY, fl.SOURCE)
    assert SOURCE_REGISTRY.get(item.source) is None, \
        "the poison branch is reachable, so the assert above means something"


# ------------------------------------------------------------------- scheduling --

def test_the_scheduled_pass_enqueues_exactly_one_investigation(tmp_path, monkeypatch):
    """`enqueue_if_due` IS the enqueue: findings exist → exactly one queue row.

    Both ledger halves go in as fixtures, so the scheduled path is shown reading
    the rotated archive and the live file, not just the live one — the failure
    mode clause 3 exists for.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures" / "failure_ledger"
    conn = fl.connect(tmp_path / "failure_issues.sqlite")
    seed_fresh_failures(conn)
    conn.close()
    # `enqueue_if_due` opens the store at `paths.FAILURE_LEDGER_DB` — the one
    # production path, which is what the seam needs — so this node has to move that
    # name the way the handler nodes do. Without it the scheduled pass reads an empty
    # store and "enqueues nothing" for a reason that has nothing to do with the code.
    monkeypatch.setattr(paths, "FAILURE_LEDGER_DB", tmp_path / "failure_issues.sqlite")
    queue = make_queue(tmp_path)
    cfg = {"enabled": True, "interval_seconds": 86400,
           "promotions_ledger": str(fixtures / "promotions_live_rows.jsonl"),
           "promotions_archives": [str(fixtures / "promotions_archive_rows.jsonl")]}

    asyncio.run(failure_ledger.enqueue_if_due(queue, cfg))
    rows = queue.list_items(source=fl.SOURCE)
    assert len(rows) == 1, "one day, one investigation row"
    row = rows[0]
    assert row.kind == fl.KIND
    assert str(row.payload.get("diagnosis", "")).strip(), \
        "the derived diagnosis must be non-empty: dispatch_findings rejects a " \
        "blank one, so an empty string here would mean the sweep skipped dispatch"
    assert "Findings to read" in row.payload["prompt"]
    assert row.payload["window"]["from"] and row.payload["window"]["to"]

    # A second pass on the same day adds nothing: the pool's own interval
    # watermark, then the ledger's cap row, then the queue's dedup key.
    asyncio.run(failure_ledger.enqueue_if_due(queue, cfg))
    assert len(queue.list_items(source=fl.SOURCE)) == 1


def test_the_scheduled_pass_reads_only_the_ledger_clock(tmp_path, monkeypatch):
    """ONE clock, so freezing the ledger's freezes the whole pass.

    The scheduled pass reads the time in two places: `sweep_and_dispatch` takes its
    `when` (and therefore the day the cap is keyed on) from `fl.now_utc`, and
    `enqueue_if_due` stamps the interval watermark that decides when it is next due.
    Both must answer to the clock this file sets. A hand-rolled `datetime.now(...)`
    left in either one shows up here as a reading that is not `NOW` — on every
    calendar day, not only the one the fixture was written for, which is the whole
    difference between this node and the test whose date-dependence it retires.
    """
    assert fl.now_utc() == NOW, "the autouse fixture is the clock this file runs on"
    conn = fl.connect(tmp_path / "failure_issues.sqlite")
    assert seed_fresh_failures(conn)
    conn.close()
    monkeypatch.setattr(paths, "FAILURE_LEDGER_DB", tmp_path / "failure_issues.sqlite")
    queue = make_queue(tmp_path)

    asyncio.run(failure_ledger.enqueue_if_due(
        queue, {"enabled": True, **fixture_cfg(tmp_path)}))

    rows = queue.list_items(source=fl.SOURCE)
    assert len(rows) == 1, "one day, one investigation row"
    assert rows[0].payload["day"] == DAY, (
        "the cap is keyed on the day the pass derived, so a row dated anything but "
        f"{DAY} means the sweep read a clock this file did not set")
    stamp = queue.wm_get(failure_ledger.NAME, failure_ledger.WATERMARK_LAST_ENQUEUED)
    assert stamp == NOW.isoformat(), (
        f"watermark stamped {stamp!r} instead of the frozen {NOW.isoformat()!r}: the "
        "interval gate would be measuring with a second clock, and a node that pins "
        "the sweep's day could not pin the interval with it")


def test_a_quiet_day_enqueues_nothing_and_says_so(tmp_path):
    """No findings → no queue row, and the report says zero rather than lying."""
    fixtures = Path(__file__).resolve().parent / "fixtures" / "failure_ledger"
    conn = fl.connect(tmp_path / "failure_issues.sqlite")
    conn.close()  # sweep_and_dispatch opens its own handle on the same store
    queue = make_queue(tmp_path)
    report = failure_ledger.sweep_and_dispatch(
        queue=queue, conn=fl.connect(tmp_path / "failure_issues.sqlite"),
        now=NOW, day=DAY,
        live=fixtures / "promotions_live_rows.jsonl",
        archives=[fixtures / "promotions_archive_rows.jsonl"])
    assert report["considered"] == 18, (
        "both fixtures are alert rows of one family: 16 in the archive, 2 in the "
        "live file. 2 would mean the reader opened the live ledger alone")
    assert report["findings"] == []
    assert report["queue_id"] is None
    assert queue.list_items(source=fl.SOURCE) == []


# ---------------------------------------------------------------------- handler --

class StubTurn:
    """Stands in for `TurnResult`; only `.ok`, `.text` and the counters are read."""

    def __init__(self, text="", stop_reason=None, num_turns=3):
        self.text = text
        self.stop_reason = stop_reason
        self.num_turns = num_turns
        self.usage = {}
        self.run_state = None
        self.session_id = "stub-session"

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    def failure_summary(self) -> str:
        return f"empty response (stop_reason={self.stop_reason})"


def dispatched_item(tmp_path, monkeypatch):
    """A real queue row for today, produced the way production produces one."""
    conn = fl.connect(tmp_path / "failure_issues.sqlite")
    assert seed_fresh_failures(conn)
    queue = make_queue(tmp_path)
    qid = fl.dispatch_findings(
        conn, DAY, queue=queue,
        diagnosis="Establish what changed near the probe alpha onset.")
    item = next(it for it in queue.list_items(source=fl.SOURCE) if it.id == qid)
    monkeypatch.setattr(paths, "FAILURE_LEDGER_DB", tmp_path / "failure_issues.sqlite")
    return conn, item


def test_execute_hands_the_run_the_ledger_rows_behind_its_own_ids(tmp_path,
                                                                  monkeypatch):
    """The prompt names finding ids; a run cannot open sqlite, so `execute` must.

    The stubbed turn records the prompt it was GIVEN — that string is the
    boundary. It must carry the ledger's own sample text, which appears nowhere in
    the queue payload, so a handler that forwarded `payload["prompt"]` unchanged
    fails here instead of returning a confident guess about rows it never read.
    """
    conn, item = dispatched_item(tmp_path, monkeypatch)
    seen: dict[str, str] = {}

    async def fake_run(prompt, max_turns=20, *, source="worker", title="",
                       extra_disallowed=()):
        seen["prompt"] = prompt
        seen["source"] = source
        return StubTurn(text="Diagnosis: the onset lines up with commit abc1234.")

    monkeypatch.setattr(_common, "run_prompt_on_primary", fake_run)
    result = asyncio.run(failure_ledger.execute(item))
    conn.close()

    assert result["status"] == "success"
    assert seen["source"] == fl.SOURCE
    assert "probe alpha blew up" in seen["prompt"], \
        "the ledger's sample_summary must reach the prompt"
    assert "first_seen" in seen["prompt"]
    for fid in item.payload["finding_ids"]:
        assert str(fid) in seen["prompt"]
    assert item.payload["prompt"] in seen["prompt"], \
        "the library's bounded prompt is kept, not replaced"


def test_a_turn_that_said_nothing_is_a_failed_run_not_an_empty_success(
        tmp_path, monkeypatch):
    """`session-distill` shipped 135 `(no response)` notes before this rule; the
    ledger's run row must not be a 136th."""
    conn, item = dispatched_item(tmp_path, monkeypatch)

    async def dead_turn(prompt, max_turns=20, *, source="worker", title="",
                        extra_disallowed=()):
        return StubTurn(text="", stop_reason="max_turns", num_turns=12)

    monkeypatch.setattr(_common, "run_prompt_on_primary", dead_turn)
    result = asyncio.run(failure_ledger.execute(item))
    conn.close()
    assert result["status"] == "failed"
    assert "max_turns" in result["summary"]


def test_a_payload_with_no_prompt_fails_without_waking_the_model(tmp_path,
                                                                 monkeypatch):
    """A hand-edited queue row must not spend a primary turn on no instruction."""
    calls: list[str] = []

    async def should_not_run(prompt, max_turns=20, *, source="worker", title="",
                             extra_disallowed=()):
        calls.append(prompt)
        return StubTurn(text="should not have run")

    monkeypatch.setattr(_common, "run_prompt_on_primary", should_not_run)
    item = type("I", (), {"payload": {"day": DAY, "finding_ids": [1]},
                          "id": 1, "source": fl.SOURCE, "kind": fl.KIND})()
    result = asyncio.run(failure_ledger.execute(item))
    assert result["status"] == "failed"
    assert calls == []


# -------------------------------------------------------------------- diagnosis --

def test_derived_diagnosis_names_the_finding_and_its_window(tmp_path):
    """A hard-coded question would satisfy the non-empty check and defeat it."""
    conn = fl.connect(tmp_path / "failure_issues.sqlite")
    assert seed_fresh_failures(conn)
    text = failure_ledger.derive_diagnosis(conn, DAY)
    conn.close()
    assert text.strip()
    assert "new onset" in text.lower(), text
    assert "3 occurrence" in text, text
    assert DAY in text, "the diagnosis must say which onset it is asking about"
    assert not any(tok in text for tok in ("TODO", "placeholder")), text


def undispatched(conn, day: str) -> list[tuple[int, str, int]]:
    """`(id, signature, occurrences)` for findings the cap has not sent anyone for.

    Plain SQL against `failure_findings.dispatched`, not a library accessor, so
    what this file calls "left uninvestigated" is the same column the cap writes
    and not a second definition of it.
    """
    return [(int(r["id"]), str(r["signature"]), int(r["occurrences"]))
            for r in conn.execute(
                "SELECT id, signature, occurrences FROM failure_findings "
                "WHERE detected_on = ? AND dispatched = 0 ORDER BY id", (day,))]


def dispatched(conn, day: str) -> list[tuple[int, str, int]]:
    """The same rows with `dispatched = 1`, for the difference between the two."""
    return [(int(r["id"]), str(r["signature"]), int(r["occurrences"]))
            for r in conn.execute(
                "SELECT id, signature, occurrences FROM failure_findings "
                "WHERE detected_on = ? AND dispatched = 1 ORDER BY id", (day,))]


def test_a_finding_that_arrives_after_the_days_run_was_sent_stays_queued(
        tmp_path, monkeypatch):
    """The cap is a store property, and the residue it leaves is on the record.

    The real sequence: the morning pass dispatches, and later the same day a second
    family crosses the new-onset bar. `enqueue_if_due` must then send NOTHING — the
    cap is one dispatch per day, not one per finding — and the new finding must
    still be sitting there with `dispatched = 0`, which is the number that says how
    much the ledger deliberately left uninvestigated.
    """
    conn = fl.connect(tmp_path / "failure_issues.sqlite")
    monkeypatch.setattr(paths, "FAILURE_LEDGER_DB", tmp_path / "failure_issues.sqlite")
    assert seed_fresh_failures(conn, families=1)
    queue = make_queue(tmp_path)
    assert fl.dispatch_findings(conn, DAY, queue=queue,
                                diagnosis="Establish what changed near the onset.")
    assert undispatched(conn, DAY) == [], "the morning pass took the first family"

    seed_fresh_failures(conn, families=1, occurrences=5, offset=1)
    sent = {sig for _, sig, _ in dispatched(conn, DAY)}
    held = {sig for _, sig, _ in undispatched(conn, DAY)}
    assert held - sent, (
        "the fresh family must be held as a DIFFERENT signature from the one already "
        "sent. Held ∩ sent is allowed and is not what this proves: a family the "
        "growth detector picks up later the same day reappears undispatched, and the "
        "cap is what stops a second run, not the finding table")

    queue.wm_set(failure_ledger.NAME, failure_ledger.WATERMARK_LAST_ENQUEUED, "")
    assert not asyncio.run(failure_ledger.enqueue_if_due(
        queue, {"enabled": True, **fixture_cfg(tmp_path)})), \
        "watermark cleared, so this is the CAP refusing, not the interval"
    assert len(queue.list_items(source=fl.SOURCE)) == 1, "still exactly one row today"
    assert len(undispatched(conn, DAY)) == 1, "and it is still on the ledger"
    conn.close()


def test_derived_diagnosis_is_empty_once_the_days_findings_are_dispatched(tmp_path):
    """Post-dispatch is the state that asks for nothing.

    Reached through `dispatch_findings` rather than by hand-inserting a
    `failure_dispatches` row: the cap row's real columns are
    `(day, queue_id, finding_ids)`, and a test that invented a `reason` column
    would be describing a schema the store does not have.
    """
    conn = fl.connect(tmp_path / "failure_issues.sqlite")
    assert seed_fresh_failures(conn)
    assert failure_ledger.derive_diagnosis(conn, DAY).strip()
    fl.dispatch_findings(conn, DAY, queue=make_queue(tmp_path),
                         diagnosis="Establish what changed near the onset.")
    assert failure_ledger.derive_diagnosis(conn, DAY) == "", \
        "nothing left undispatched, so the sweep must invent a question to dispatch"
    conn.close()


# ------------------------------------------------------------- growth's day --


def seed_yesterdays_burst(conn) -> str:
    """One OLD family, five events yesterday, nothing on the pass's own day.

    The shape #2893 measured over the live ledger: a burst in a family that
    already existed. Two events on 2026-09-27 and two on 2026-09-29 make the
    reviewed day's trailing-7-day active median 2.0, so the 2x bar is 4.0; the
    five on 2026-10-02 clear `GROWTH_MIN_COUNT` (3) and that bar.

    `first_seen` is 2026-09-27, more than 24 h before either test clock here, so
    `detect_new_onset` cannot fire: a spike in a family the ledger already knows
    has exactly one detector, and these nodes are about whether it can ever see
    one. Every event is written with the same message, so `signature`'s digit
    collapse and all of it is one signature.
    """
    stamps = [datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc),
              datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc),
              datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc),
              datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)]
    stamps += [datetime(2026, 10, 2, h, 0, tzinfo=timezone.utc)
               for h in (18, 19, 20, 21, 22)]
    sig = ""
    for stamp in stamps:
        sig = fl.record_failure(conn, logger_name="lloyd-worker",
                                message="voice worker dropped a turn: asr timeout",
                                ts=fl.iso_utc(stamp), run_id=stamp.isoformat())
    fl.sweep(conn)
    return sig


def early_morning_pass(tmp_path, monkeypatch):
    """Freeze the ledger's clock to the real pass's instant and hand back the pool
    queue and the store path that pass will open by default."""
    monkeypatch.setattr(fl, "now_utc", lambda: EARLY_PASS)
    db = tmp_path / "failure_issues.sqlite"
    conn = fl.connect(db)
    sig = seed_yesterdays_burst(conn)
    conn.close()
    monkeypatch.setattr(paths, "FAILURE_LEDGER_DB", db)
    return sig, db, make_queue(tmp_path)


def test_the_morning_pass_finds_a_burst_dated_yesterday_as_one_growth_finding(
        tmp_path, monkeypatch):
    """The clause #2893 exists for, driven through the pool's own entry point.

    A pass frozen a few hours into day D (02:10 UTC, the shape of every real
    sweep) over a store whose >=3-event same-signature burst is dated the previous
    UTC day must find it as exactly one `growth` finding. Before #2893 the same
    store produced zero findings on this clock — the detector counted `ts[:10] ==
    day_of(now)`, so the only part of any day it ever saw was the slice before
    ~02:40, and the control at the end of this node is that fact still holding for
    a review aimed at the pass's own day.
    """
    assert (YESTERDAY, DAY) == ("2026-10-02", "2026-10-03"), \
        "the burst's day must be the day before the pass's, or this proves nothing"
    sig, db, queue = early_morning_pass(tmp_path, monkeypatch)

    asyncio.run(failure_ledger.enqueue_if_due(
        queue, {"enabled": True, **fixture_cfg(tmp_path)}))

    store = fl.connect(db)
    rows = store.execute(
        "SELECT signature, kind, detected_on, occurrences FROM failure_findings"
    ).fetchall()
    assert len(rows) == 1, f"one burst is one finding, got {tuple(rows)!r}"
    assert tuple(rows[0]) == (sig, "growth", DAY, 5), \
        "the burst is growth, stamped the day the pass ran (dispatch reads that " \
        "day), with all five of yesterday's events counted"

    # new_onset is silent on this store, so growth is the only detector in play…
    assert fl.detect_new_onset(store, now=EARLY_PASS, day=DAY) == [], \
        "the family is 6 days old: an onset finding here would hide which detector worked"
    # …and a review aimed at the pass's own day still finds nothing, which is the
    # exact shape that made 6 of the live ledger's 8 burst-days invisible.
    own_day = store.execute(
        "SELECT COUNT(*) FROM failure_events WHERE ts LIKE ?", (f"{DAY}T%",)).fetchone()[0]
    assert own_day == 0 and fl.detect_growth(store, day=DAY) == []
    store.close()


def test_yesterdays_burst_reaches_the_queue_instead_of_sitting_undispatched(
        tmp_path, monkeypatch):
    """A finding nothing enqueued is a row, not an investigation.

    The trap #2893 names in CHECK (2): `_emit_finding` stamps `detected_on` and
    `dispatch_findings` selects `WHERE detected_on = ?` on the pass's day. Review
    an earlier day without stamping the finding for the day that is dispatching,
    and the finding is written, `dispatched` stays 0, `derive_diagnosis` goes empty,
    the pass logs `no diagnosis — nothing dispatched`, and nothing fails.
    """
    sig, db, queue = early_morning_pass(tmp_path, monkeypatch)

    asyncio.run(failure_ledger.enqueue_if_due(
        queue, {"enabled": True, **fixture_cfg(tmp_path)}))

    rows = queue.list_items(source=fl.SOURCE)
    assert len(rows) == 1, f"yesterday's burst is today's one investigation, got {len(rows)}"
    row = rows[0]
    assert row.dedup_key == f"failure-ledger:{DAY}", \
        "the dedup key stays the pass's own day — the cap is one run per day"
    assert row.kind == fl.KIND and row.priority == 75
    assert row.payload["day"] == DAY and row.payload["signatures"] == [sig], \
        "the row is this pass's day and carries the burst's own family"
    assert "Growth" in row.payload["diagnosis"] and "2026-09-27" in row.payload["diagnosis"], \
        "the derived diagnosis names the kind and the family's onset, so the run is " \
        "asked about a growth burst rather than a generic failure"

    store = fl.connect(db)
    assert undispatched(store, DAY) == [], \
        "the finding was recorded AND dispatched: dispatched=0 here is the silent " \
        "detected_on mismatch, not a fix"
    assert [s for _, s, _ in dispatched(store, DAY)] == [sig]
    assert store.execute(
        "SELECT COUNT(*) FROM failure_dispatches WHERE day = ?", (DAY,)).fetchone()[0] == 1
    store.close()


def test_two_morning_passes_on_one_day_still_spend_one_run_on_the_burst(
        tmp_path, monkeypatch):
    """The fix must not double-spend the day it reviews.

    Same store, same frozen clock (`enqueue_if_due` returns None on every path by
    design, so the queue and the ledger are the evidence, not the return value). The
    interval watermark is cleared between the sweeps so the second pass is not waved
    off by the 86400-second wait: it runs the whole sweep, re-detects the same burst,
    finds every finding already dispatched, so `derive_diagnosis` returns "" and it
    enqueues nothing. And `dispatch_findings` called directly WITH a question to ask
    still returns None, because its first statement is
    `SELECT queue_id FROM failure_dispatches WHERE day = ?` — one burst buys one run
    however many times the pool polls.
    """
    _, db, queue = early_morning_pass(tmp_path, monkeypatch)
    cfg = {"enabled": True, **fixture_cfg(tmp_path)}

    asyncio.run(failure_ledger.enqueue_if_due(queue, cfg))
    first = queue.list_items(source=fl.SOURCE)
    assert len(first) == 1, "the first pass enqueues the day's one investigation"

    # The interval watermark is the POOL's (`workers.db`), not the ledger's, so it is
    # cleared on the queue — which is also the proof that a second sweep, not a
    # skipped one, is what the assertions below are reading.
    queue.wm_set(failure_ledger.NAME, failure_ledger.WATERMARK_LAST_ENQUEUED, "")
    asyncio.run(failure_ledger.enqueue_if_due(queue, cfg))
    store = fl.connect(db)
    assert fl.dispatch_findings(store, DAY, queue=queue,
                                diagnosis="Establish what changed near the burst.") is None, \
        "the daily cap row, not the interval, is what refuses a second run"
    rows = queue.list_items(source=fl.SOURCE)
    assert len(rows) == 1 and rows[0].id == first[0].id, \
        f"two passes on one day must not spend two runs, got {len(rows)}"
    assert store.execute("SELECT COUNT(*) FROM failure_dispatches").fetchone()[0] == 1
    assert store.execute("SELECT COUNT(*) FROM failure_findings").fetchone()[0] == 1, \
        "re-reviewing the same burst day re-uses the one (signature, kind, " \
        "detected_on) finding rather than appending a second"
    store.close()
