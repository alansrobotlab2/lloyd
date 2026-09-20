"""`WorkQueue` — the guarantees the self-modification loop now rides on.

This file exists because the queue had no tests at all. Every property below
is one the loop depends on to be unattended: that a claim is exclusive, that a
saturated source cannot hide claimable work behind it, that a failure backs
off instead of spinning, and that a crashed pool releases what it was holding.

Two of them are regressions, not hypotheticals — see
`test_a_saturated_source_cannot_starve_the_queue` and
`test_recovery_releases_rows_claimed_by_slots_that_no_longer_exist`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from workers.queue import QueueItem, WorkQueue, new_run_id


@pytest.fixture
def q(tmp_path) -> WorkQueue:
    return WorkQueue(tmp_path / "workers.db")


def _ids(items: list[QueueItem]) -> list[int]:
    return [i.id for i in items]


# ---------------------------------------------------------------------------
# Ordering and claiming
# ---------------------------------------------------------------------------


def test_priority_ascending_then_oldest_first(q):
    """Lower priority number runs sooner; ties break on enqueue order."""
    late_urgent = q.enqueue("a", "k", priority=10)
    early_dull = q.enqueue("b", "k", priority=70)
    mid = q.enqueue("c", "k", priority=40)

    order = []
    while (item := q.claim_next("worker-0")) is not None:
        order.append(item.id)
        q.mark_completed(item.id)
    assert order == [late_urgent, mid, early_dull]


def test_a_claim_is_exclusive(q):
    q.enqueue("a", "k")
    first = q.claim_next("worker-0")
    second = q.claim_next("worker-1")
    assert first is not None
    assert second is None, "the same row was handed to two workers"
    assert first.claimed_by == "worker-0"
    assert first.attempts == 1


def test_claiming_records_the_worker_and_counts_the_attempt(q):
    q.enqueue("a", "k")
    item = q.claim_next("worker-3")
    assert item.state == "claimed" and item.claimed_by == "worker-3"
    assert q.get(item.id).attempts == 1


def test_an_empty_queue_claims_nothing(q):
    assert q.claim_next("worker-0") is None


# ---------------------------------------------------------------------------
# Per-source inflight quota
# ---------------------------------------------------------------------------


def test_a_quota_bounds_concurrent_work_from_one_source(q):
    for _ in range(3):
        q.enqueue("heavy", "k")
    quota = {"heavy": 1}
    assert q.claim_next("worker-0", quota) is not None
    assert q.claim_next("worker-1", quota) is None, "quota of 1 allowed a second claim"


def test_a_saturated_source_cannot_starve_the_queue(q):
    """The regression: quota applied to a *window*, not to the query.

    `claim_next` used to `SELECT ... LIMIT 50` and then skip over-quota rows in
    Python. Fifty queued rows from one saturated source, all at a lower
    priority number, filled that window completely — so the claimable row
    behind them was never seen and the pool read the queue as empty and slept.
    With three sources sharing two slots on this box, "one source has fifty
    queued items" is an ordinary Tuesday, not a pathological case.
    """
    for _ in range(60):
        q.enqueue("flood", "k", priority=10)
    wanted = q.enqueue("rare", "k", priority=50)

    # One flood item is already in flight and the source's quota is 1.
    assert q.claim_next("worker-0", {"flood": 1}) is not None

    got = q.claim_next("worker-1", {"flood": 1})
    assert got is not None, "claimable work was starved behind a saturated source"
    assert got.id == wanted


def test_a_source_with_no_quota_is_unbounded(q):
    for _ in range(3):
        q.enqueue("free", "k")
    assert q.claim_next("worker-0", {"other": 1}) is not None
    assert q.claim_next("worker-1", {"other": 1}) is not None


def test_running_counts_toward_the_quota_not_just_claimed(q):
    q.enqueue("s", "k")
    q.enqueue("s", "k")
    first = q.claim_next("worker-0", {"s": 1})
    q.mark_running(first.id)
    assert q.claim_next("worker-1", {"s": 1}) is None


def test_a_completed_item_frees_the_quota(q):
    q.enqueue("s", "k")
    q.enqueue("s", "k")
    first = q.claim_next("worker-0", {"s": 1})
    q.mark_completed(first.id)
    assert q.claim_next("worker-1", {"s": 1}) is not None


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_a_live_dedup_key_coalesces(q):
    first = q.enqueue("s", "k", dedup_key="only-one")
    assert q.enqueue("s", "k", dedup_key="only-one") is None
    item = q.claim_next("worker-0")
    assert item.id == first
    assert q.enqueue("s", "k", dedup_key="only-one") is None, "claimed did not coalesce"
    q.mark_running(item.id)
    assert q.enqueue("s", "k", dedup_key="only-one") is None, "running did not coalesce"


def test_completion_releases_the_key_so_the_work_can_recur(q):
    """Most sources want to run the same key again later.

    scheduled-task dedups on `scheduled-task:<id>` and must be able to run
    that task on its next interval, so completion has to release the key.
    Sources that want once-ever semantics keep their own record — see
    `session_distill`'s `done:` markers.
    """
    first = q.enqueue("s", "k", dedup_key="recurring")
    item = q.claim_next("worker-0")
    q.mark_completed(item.id)
    second = q.enqueue("s", "k", dedup_key="recurring")
    assert second is not None and second != first


def test_a_poisoned_row_yields_its_key_to_the_next_enqueue(q):
    first = q.enqueue("s", "k", dedup_key="doomed")
    item = q.claim_next("worker-0")
    assert q.mark_failed(item.id, "boom", max_attempts=1) == "poisoned"
    second = q.enqueue("s", "k", dedup_key="doomed")
    assert second is not None and second != first
    # The old row keeps its history but no longer owns the key.
    assert q.get(first).dedup_key is None


def test_no_dedup_key_means_no_coalescing(q):
    a = q.enqueue("s", "k")
    b = q.enqueue("s", "k")
    assert a != b


# ---------------------------------------------------------------------------
# Failure, retry and backoff
# ---------------------------------------------------------------------------


def test_a_failure_under_the_cap_requeues_with_a_backoff(q):
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    assert q.mark_failed(item.id, "transient", max_attempts=3) == "queued"

    row = q.get(item.id)
    assert row.state == "queued" and row.claimed_by is None
    assert row.not_before, "a requeued item must not be instantly re-claimable"
    assert q.claim_next("worker-0") is None, "backoff did not hold the item back"


def test_the_backoff_grows_and_is_capped(q):
    """30s, 60s, 120s … capped at 10 minutes."""
    q.enqueue("s", "k")
    delays = []
    for attempt in range(1, 6):
        # Simulate the claim's attempt bump without waiting out the backoff.
        with q._connect() as conn:
            conn.execute("UPDATE queue SET state='claimed', attempts=? WHERE id=1",
                         (attempt,))
        before = datetime.now(timezone.utc)
        q.mark_failed(1, "again", max_attempts=99)
        nb = datetime.fromisoformat(q.get(1).not_before)
        delays.append(round((nb - before).total_seconds()))
    assert delays == [30, 60, 120, 240, 480]
    assert all(d <= 600 for d in delays)


def test_backoff_never_exceeds_ten_minutes(q):
    q.enqueue("s", "k")
    with q._connect() as conn:
        conn.execute("UPDATE queue SET state='claimed', attempts=20 WHERE id=1")
    before = datetime.now(timezone.utc)
    q.mark_failed(1, "again", max_attempts=99)
    nb = datetime.fromisoformat(q.get(1).not_before)
    assert round((nb - before).total_seconds()) == 600


def test_an_item_becomes_claimable_once_its_backoff_expires(q):
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    q.mark_failed(item.id, "transient", max_attempts=3)
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with q._connect() as conn:
        conn.execute("UPDATE queue SET not_before=? WHERE id=?", (past, item.id))
    assert q.claim_next("worker-0") is not None


def test_exhausting_the_attempts_poisons_rather_than_loops(q):
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    assert q.mark_failed(item.id, "fatal", max_attempts=1) == "poisoned"
    assert q.get(item.id).state == "poisoned"
    assert q.claim_next("worker-0") is None


def test_failing_a_row_that_is_gone_is_reported_not_raised(q):
    assert q.mark_failed(9999, "boom") == "missing"


def test_success_clears_a_stale_error_but_keeps_the_attempt_count(q):
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    q.mark_failed(item.id, "first try", max_attempts=3)
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with q._connect() as conn:
        conn.execute("UPDATE queue SET not_before=? WHERE id=?", (past, item.id))
    again = q.claim_next("worker-0")
    q.mark_completed(again.id)

    row = q.get(item.id)
    assert row.state == "completed"
    assert row.error is None, "a prior failure survived a later success"
    assert row.attempts == 2, "'recovered after N attempts' must stay visible"


def test_an_error_message_is_bounded(q):
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    q.mark_failed(item.id, "x" * 9000, max_attempts=99)
    assert len(q.get(item.id).error) == 2000


# ---------------------------------------------------------------------------
# The retry ceiling: who owns it, and what stops a row passing it
#
# Backlog item #765. `attempts` is incremented by the claim, but until this
# section the only code that tested the cap lived inside `mark_failed`. Every
# path that returns a row to `queued` without failing it — crash recovery above
# all — therefore raised the counter with nothing checking it, and queue row
# 3946 reached `attempts=9` against a cap of 3 in 44 minutes that way. The
# tests below are that violation, driven to a terminal state.
# ---------------------------------------------------------------------------


def _park(q: WorkQueue, item_id: int, *, attempts: int, state: str = "queued",
          claimed_by: str | None = None) -> None:
    """Stand in for a row that reached `attempts` without ever being failed.

    That is the shape the ceiling has to survive: a process died mid-run, so no
    `mark_failed` call exists, and the row was swept back to `queued` with its
    count intact. Writing the row directly is how the pre-fix queue produced
    this state on its own; writing it here is what lets the assertion be about
    the claim rather than about waiting for a crash.
    """
    with q._connect() as conn:
        conn.execute("UPDATE queue SET state=?, attempts=?, claimed_at=NULL, "
                     "claimed_by=?, not_before=NULL WHERE id=?",
                     (state, attempts, claimed_by, item_id))
        conn.commit()


def _clear_backoff(q: WorkQueue, item_id: int) -> None:
    """Stand in for the wall-clock wait `mark_failed`'s retry backoff imposes."""
    with q._connect() as conn:
        conn.execute("UPDATE queue SET not_before=NULL WHERE id=?", (item_id,))
        conn.commit()


def test_claiming_an_item_at_the_cap_poisons_it_instead_of_handing_it_out(q):
    """Clause 1. The claim is where the ceiling belongs, because that UPDATE is
    what raises `attempts` — testing it there holds against a caller, or a
    crash, that `mark_failed` never sees."""
    item_id = q.enqueue("s", "k")
    _park(q, item_id, attempts=3)

    assert q.claim_next("worker-0") is None, "an item at the cap was handed to a worker"

    row = q.get(item_id)
    assert row.state == "poisoned"
    assert row.attempts == 3, "the claim persisted attempts past the cap"
    assert "max_attempts=3" in row.error, "the row must say which cap stopped it"
    assert row.completed_at, "a poisoned row needs an end time like any other terminal row"
    assert q.claim_next("worker-0") is None, "a poisoned row became claimable again"


def test_a_poisoned_row_at_the_cap_releases_its_dedup_key(q):
    """The sweep terminates rows, so it has to release the key exactly as
    `mark_failed` does or the source is blocked from ever re-enqueuing."""
    item_id = q.enqueue("s", "k", dedup_key="capped")
    _park(q, item_id, attempts=3)

    assert q.claim_next("worker-0") is None
    assert q.get(item_id).dedup_key is None, "a terminated row still owned the key"
    assert q.enqueue("s", "k", dedup_key="capped") is not None


def test_a_capped_row_does_not_starve_the_work_behind_it(q):
    """Refusing a row is not the same as hiding it. A row left `queued` but
    unclaimable is the inflight-quota starvation bug this file already pins:
    the pool would read the queue as empty and sleep past the real work."""
    capped = q.enqueue("doomed", "k", priority=10)
    wanted = q.enqueue("useful", "k", priority=90)

    # priority 10 sorts first, so the capped row is what the claim sees first.
    _park(q, capped, attempts=3)

    assert q.claim_next("worker-0").id == wanted, "a capped row blocked the queue"
    assert q.get(capped).state == "poisoned"


@pytest.mark.parametrize("cap", [2, 5], ids=["cap-2", "cap-5"])
def test_the_ceiling_is_the_callers_cap_and_not_a_second_literal(tmp_path, cap):
    """Clause 4. Each queue stops at the number it was given, on both enforcing
    paths, so the configured `workers.max_attempts` is what is enforced rather
    than a literal copied into the enforcement."""
    queue = WorkQueue(tmp_path / f"cap-{cap}.db", max_attempts=cap)

    at_cap = queue.enqueue("doomed", "k")
    _park(queue, at_cap, attempts=cap)
    assert queue.claim_next("worker-0") is None, f"a cap-{cap} queue handed out claim {cap}"
    assert f"max_attempts={cap}" in queue.get(at_cap).error

    item_id = queue.enqueue("crash-loop", "k")
    claims = 0
    while (item := queue.claim_next("worker-0")) is not None:
        claims += 1
        assert item.attempts <= cap, f"ran claim {item.attempts} against cap {cap}"
        queue.recover_claimed()  # the process died; the next boot sweeps it back

    assert claims == cap, f"a cap-{cap} queue ran {claims} claims"
    row = queue.get(item_id)
    assert row.state == "poisoned"
    assert row.attempts == cap
    assert f"max_attempts={cap}" in row.error


def test_mark_failed_defaults_to_this_queues_cap(tmp_path):
    """`mark_failed` used to carry its own literal 3, which is how the ceiling
    could drift from the configured value without either site looking wrong. The
    pool passes its cap explicitly; every other caller now inherits the queue's.
    """
    queue = WorkQueue(tmp_path / "failed-cap.db", max_attempts=5)
    item_id = queue.enqueue("s", "k")

    _park(queue, item_id, attempts=4)
    assert queue.mark_failed(item_id, "boom") == "queued", \
        "a cap-5 queue poisoned at 4: it tested a literal 3, not its own cap"
    _park(queue, item_id, attempts=5)
    assert queue.mark_failed(item_id, "boom") == "poisoned"


def test_the_pool_hands_its_cap_to_the_queue_it_enforces(tmp_path):
    """The plumbing half of clause 4: `app/routers/workers.py` reads
    `workers.max_attempts` and passes it to `start_pool`, so the pool's own
    number has to be the one the queue enforces — not the constructor default.
    The queue is built by `get_queue()` before any pool exists, which is why
    this is a setter call and not a constructor argument."""
    from workers.pool import WorkerPool

    queue = WorkQueue(tmp_path / "pool-cap.db")
    assert queue.max_attempts == 3, "default cap drifted"

    WorkerPool(queue, slots=1, max_attempts=7)
    assert queue.max_attempts == 7, "the pool kept its cap to itself"


def test_the_shared_queue_takes_its_cap_from_the_same_config_key(tmp_path, monkeypatch):
    """The cross-process half of clause 4. Only the backend runs `start_pool`,
    so the aggregator — which reaches the same `workers.db` through
    `get_queue(configured_db_path())` — never goes through the pool at all. If it
    built its queue with this module's default while the backend used the config,
    one file would have two ceilings, which is the #765 defect wearing a
    different hat. `configured_db_path` was written for the same split."""
    import workers.queue as WQ
    import app.config

    monkeypatch.setattr(WQ, "_queue_instance", None)
    monkeypatch.setattr(app.config, "CONFIG", {"workers": {"max_attempts": 6}})

    queue = WQ.get_queue(tmp_path / "shared.db")

    assert queue.max_attempts == 6, "the singleton ignored the configured cap"


def test_an_unreadable_config_leaves_the_default_ceiling(tmp_path, monkeypatch):
    """A missing key must not stop the shared queue being constructed — but it
    must also not widen anything: the fallback is the default, not no cap."""
    import workers.queue as WQ
    import app.config

    monkeypatch.setattr(app.config, "CONFIG", {"workers": {}})
    assert WQ.configured_max_attempts() == 3

    monkeypatch.setattr(app.config, "CONFIG", {})
    assert WQ.configured_max_attempts() == 3


def test_a_cap_below_one_disables_nothing(q):
    """A zero cap would mean no work is ever claimable. The one caller that
    wants a row terminal without running it says so at the call site
    (`mark_failed(item, err, 0)` for an unknown source), so a boot-time zero is
    clamped rather than obeyed."""
    queue = WorkQueue(q.db_path.parent / "zero.db", max_attempts=0)
    assert queue.max_attempts == 1
    queue.enqueue("s", "k")
    assert queue.claim_next("worker-0") is not None


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def test_recovery_enforces_the_ceiling_in_both_directions(q):
    """Clause 2, and the actual #765 mechanism: `recover_claimed` runs at every
    pool start and used to reset `claimed|running` to `queued` without ever
    reading `attempts`. Eight of queue row 3946's nine claims were recovered
    sweeps, not failures. A row at the cap has to stop there; a row under it
    must still come back, or a reboot becomes lost work."""
    doomed = q.enqueue("crash-loop", "k", priority=10)
    healthy = q.enqueue("fine", "k", priority=20)

    # Where a boot finds each row: one out of GPU three deep into a crash loop,
    # one claimed once by a worker that died.
    _park(q, doomed, attempts=3, state="running", claimed_by="worker-0")
    _park(q, healthy, attempts=1, state="claimed", claimed_by="worker-1")

    recovered = q.recover_claimed()

    assert recovered == 1, "an exhausted row was counted as recovered"
    dead = q.get(doomed)
    assert dead.state == "poisoned"
    assert dead.attempts == 3
    assert "max_attempts=3" in dead.error

    alive = q.get(healthy)
    assert alive.state == "queued", "a sub-cap row was poisoned for someone else's crash"
    assert alive.attempts == 1 and alive.claimed_by is None
    again = q.claim_next("worker-1")
    assert again is not None and again.id == healthy
    q.mark_completed(again.id)
    assert q.get(healthy).state == "completed"


def test_a_crash_loop_ends_at_the_cap(q):
    """Clause 3, reproducing #765's own shape: claim, run, die, be swept back,
    be re-claimed — `mark_failed` never runs, so the only thing that can stop
    this is the ceiling on the paths themselves. The `mark_running` is what
    makes it the shape the sweep actually meets: a worker marks the row running
    before it executes the job, so a process that dies mid-job leaves `running`
    behind, which is one of the two states `recover_claimed` sweeps."""
    item_id = q.enqueue("crash-loop", "k")
    cap = q.max_attempts

    claims = []
    for _cycle in range(cap + 1):
        item = q.claim_next("worker-0")
        if item is None:
            break                    # the ceiling refused it; that is the outcome
        claims.append(item.attempts)
        q.mark_running(item.id)      # the worker got as far as executing the job
        q.recover_claimed()          # then the process died; the next boot sweeps it back

    assert claims == [1, 2, 3], f"claims ran to {claims} against a cap of {cap}"
    row = q.get(item_id)
    assert row.state == "poisoned", "a crash loop outlived its cap"
    assert row.attempts == cap == 3, "the terminal row sat above the cap"
    assert "max_attempts=3" in row.error
    assert q.claim_next("worker-0") is None, "a terminal row went back out to a worker"
    assert q.recover_claimed() == 0


# ---------------------------------------------------------------------------
# Poison triage: a revive hands out one claim, and the ceiling still stops it
#
# `workers/maintenance.py` triages every poisoned row and revives the ones it
# judges transient with `attempts=max(0, max_attempts - 1)` — deliberately one
# short of the cap, so the item gets exactly one further claim rather than a
# fresh budget. That grant sits directly on the line the new ceiling enforces,
# so it is pinned here in both directions: the grant must still be granted, and
# it must still be the last one.
# ---------------------------------------------------------------------------


def test_a_revive_one_short_of_the_cap_still_gets_exactly_one_claim(q):
    """Clause 5. The sweep parks a revived row at `cap - 1`; a claim-time
    ceiling that read `>= cap` one row too early would refuse that claim, so
    the sweep would report a revive and hand out nothing."""
    item_id = q.enqueue("flaky", "k")
    cap = q.max_attempts

    # How the sweep meets a row: poisoned, triaged transient, revived one
    # short of the cap with its count preserved rather than reset.
    _park(q, item_id, attempts=cap)
    assert q.claim_next("worker-0") is None      # the ceiling terminates it
    assert q.get(item_id).state == "poisoned"

    assert q.revive(
        item_id, attempts=cap - 1, delay_seconds=0,
        triage={"action": "revived", "reason": "transient failure — one more claim"},
    ), "the revive lost its row to another writer"

    item = q.claim_next("worker-0")
    assert item is not None, "the ceiling refused the one claim a revive exists to grant"
    assert item.id == item_id and item.attempts == cap, "a revive handed out more than one claim"

    # And that claim is the last: failing it returns the row to its terminal
    # state at the cap, never above it and never claimable again.
    assert q.mark_failed(item.id, "TimeoutError: gpu vanished", max_attempts=cap) == "poisoned"
    row = q.get(item_id)
    assert row.state == "poisoned", "the revived row escaped its terminal state"
    assert row.attempts == cap, "the revived row's count went past the cap"
    assert q.claim_next("worker-0") is None, "a re-poisoned row was handed out again"
    assert q.recover_claimed() == 0


def test_a_revive_under_a_cap_of_one_still_grants_its_single_claim(tmp_path):
    """The degenerate end of the same grant. `max(0, max_attempts - 1)` is 0 at
    a cap of 1, so a revived row is *below* the ceiling and must be claimable
    once; the next failure ends it. Guards the sweep's arithmetic at the
    smallest cap the queue will honour."""
    queue = WorkQueue(tmp_path / "cap1.db", max_attempts=1)
    item_id = queue.enqueue("flaky", "k")

    first = queue.claim_next("worker-0")
    assert first is not None and first.attempts == 1
    assert queue.mark_failed(item_id, "boom") == "poisoned"

    assert queue.revive(item_id, attempts=0, delay_seconds=0)

    item = queue.claim_next("worker-0")
    assert item is not None, "a cap-1 revive handed out nothing"
    assert item.attempts == 1
    assert queue.mark_failed(item.id, "boom again") == "poisoned"
    assert queue.get(item_id).attempts == 1
    assert queue.claim_next("worker-0") is None


def test_recovery_returns_in_flight_rows_to_the_queue(q):
    q.enqueue("s", "k")
    q.enqueue("s", "k")
    a = q.claim_next("worker-0")
    b = q.claim_next("worker-1")
    q.mark_running(b.id)

    assert q.recover_claimed() == 2
    for row in (q.get(a.id), q.get(b.id)):
        assert row.state == "queued"
        assert row.claimed_by is None and row.claimed_at is None


def test_recovery_releases_rows_claimed_by_slots_that_no_longer_exist(q):
    """The regression: `pool.start` filtered recovery by its current slots.

    Worker ids are positional (`worker-0` … `worker-{slots-1}`). Lowering
    `workers.slots` from 4 to 2 therefore left anything `worker-2` or
    `worker-3` was holding stuck in `running` forever — and because
    `claim_next` counts `claimed|running` toward the quota, a source with
    `max_inflight: 1` was then switched off permanently and silently.
    """
    q.enqueue("s", "k")
    stranded = q.claim_next("worker-3")
    q.mark_running(stranded.id)

    assert q.recover_claimed(["worker-0", "worker-1"]) == 0, "fixture is wrong"
    assert q.get(stranded.id).state == "running"
    assert q.claim_next("worker-0", {"s": 1}) is None, "the row still holds the quota"

    assert q.recover_claimed() == 1
    assert q.get(stranded.id).state == "queued"


def test_recovery_leaves_finished_rows_alone(q):
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    q.mark_completed(item.id)
    assert q.recover_claimed() == 0
    assert q.get(item.id).state == "completed"


# ---------------------------------------------------------------------------
# Inspection, runs and watermarks
# ---------------------------------------------------------------------------


def test_depth_counts_by_source_and_state(q):
    q.enqueue("a", "k")
    q.enqueue("a", "k")
    q.enqueue("b", "k")
    q.mark_completed(q.claim_next("worker-0").id)

    depth = q.depth_by_source()
    assert depth["a"]["completed"] == 1
    assert depth["a"]["queued"] == 1
    assert depth["b"]["queued"] == 1


def test_listing_filters_on_state_and_source(q):
    q.enqueue("a", "k")
    q.enqueue("b", "k")
    q.mark_completed(q.claim_next("worker-0").id)
    assert len(q.list_items(source="a")) == 1
    assert _ids(q.list_items(state="queued")) == _ids(q.list_items(source="b"))


def test_a_payload_round_trips(q):
    payload = {"task_id": 42, "nested": {"a": [1, 2]}, "unicode": "café"}
    q.enqueue("s", "k", payload=payload)
    assert q.claim_next("worker-0").payload == payload


def test_a_run_record_is_written_and_readable(q):
    run_id = new_run_id("s")
    q.record_run(run_id=run_id, queue_id=None, source="s", status="success",
                 started_at="2026-01-01T00:00:00+00:00",
                 completed_at="2026-01-01T00:01:00+00:00",
                 duration_seconds=60.0, summary="done", task_id="7",
                 meta_json=json.dumps({"k": "v"}))
    runs = q.list_runs(source="s")
    assert len(runs) == 1 and runs[0]["run_id"] == run_id
    assert q.list_runs(task_id="7")[0]["summary"] == "done"
    assert q.list_runs(task_id="nobody") == []


def test_run_fields_are_bounded(q):
    q.record_run(run_id=new_run_id("s"), queue_id=None, source="s", status="success",
                 started_at="a", completed_at="b", duration_seconds=1.0,
                 summary="s" * 2000, response_json="r" * 99999,
                 meta_json="m" * 99999)
    row = q.list_runs(source="s")[0]
    assert len(row["summary"]) == 500
    assert len(row["response_json"]) == 50000
    assert len(row["meta_json"]) == 20000


def test_run_ids_are_unique_and_named_for_their_source(q):
    ids = {new_run_id("gap-fill") for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("run_gap-fill_") for i in ids)


def test_a_watermark_round_trips_and_overwrites(q):
    assert q.wm_get("s", "cursor") is None
    q.wm_set("s", "cursor", "1")
    q.wm_set("s", "cursor", "2")
    assert q.wm_get("s", "cursor") == "2"


def test_a_watermark_can_be_retired(q):
    """A source that outgrows a cursor has to be able to delete it, or the
    stale key keeps deciding what gets skipped."""
    q.wm_set("s", "last_mtime", "123")
    q.wm_delete("s", "last_mtime")
    assert q.wm_get("s", "last_mtime") is None
    q.wm_delete("s", "never-existed")  # idempotent


def test_watermark_keys_are_listed_per_source(q):
    q.wm_set("s", "done:a.json", "1")
    q.wm_set("s", "done:b.json", "1")
    q.wm_set("s", "cursor", "1")
    q.wm_set("other", "done:c.json", "1")
    assert sorted(q.wm_keys("s")) == ["cursor", "done:a.json", "done:b.json"]
    assert q.wm_keys("nobody") == []


def test_the_schema_migrates_onto_an_existing_database(tmp_path):
    """`not_before` and `meta_json` are additive migrations, so reopening a
    database written before them must not fail or lose rows."""
    path = tmp_path / "workers.db"
    first = WorkQueue(path)
    first.enqueue("s", "k", payload={"a": 1})
    reopened = WorkQueue(path)
    assert reopened.claim_next("worker-0").payload == {"a": 1}
