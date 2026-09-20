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
# The interrupted run record (#1137)
#
# A run killed by a restart or a pool cancel reached neither of
# `WorkerPool._run_item`'s two recording arms, so it wrote no `runs` row: of
# 1,552 runs the pool logged as started over 09-15→09-20, 17 had no row at all,
# and every timeout and wasted-hours metric — which reads `runs` — reported a
# clean fleet across all of them. These tests pin the sweep as the writer that
# survives every kind of death.
# ---------------------------------------------------------------------------


def _backdate_claimed_at(q: WorkQueue, item_id: int, *, seconds: float) -> str:
    """Move a claimed row's `claimed_at` into the past and hand it back.

    Stands in for the wall clock a real run spends before its process dies: a
    freshly claimed row has `claimed_at` == now, so a sweep measured against it
    would report a duration of 0.0 and the assertion about duration would be
    about the fixture, not about the code.
    """
    past = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with q._connect() as conn:
        conn.execute("UPDATE queue SET claimed_at=? WHERE id=?", (past, item_id))
        conn.commit()
    return past


def _force_claimed_at(q: WorkQueue, item_id: int, stamp: str) -> None:
    """Stamp an EXACT `claimed_at`, for a test that is about a stamp's FORMAT
    rather than about how long ago it was (`_backdate_claimed_at` covers that)."""
    with q._connect() as conn:
        conn.execute("UPDATE queue SET claimed_at=? WHERE id=?", (stamp, item_id))
        conn.commit()


def _interrupted_rows(q: WorkQueue) -> list[dict]:
    return [r for r in q.list_runs() if r["status"] == "interrupted"]


def test_the_recovery_sweep_writes_one_interrupted_run_per_swept_row(q):
    """Clause 1. The killed attempt becomes a queryable fact at the next boot,
    which is the only moment anything still knows it existed."""
    q.enqueue("autocode", "task", payload={"task_id": 83})
    item = q.claim_next("worker-0")
    q.mark_running(item.id)

    assert q.recover_claimed() == 1

    rows = _interrupted_rows(q)
    assert len(rows) == 1, "the swept row is not recorded exactly once"
    assert rows[0]["queue_id"] == item.id
    assert rows[0]["source"] == item.source
    assert q.list_runs() == rows, "the sweep wrote a row for something it did not sweep"


def test_the_filtered_sweep_records_only_what_it_swept(q):
    """Clause 1's other form. `recover_claimed(worker_ids)` leaves the other
    worker's row `running` — and must not report a run it did not reclaim."""
    q.enqueue("autocode", "task", payload={"task_id": 83})
    q.enqueue("autotriage", "task", payload={"task_id": 76})
    mine = q.claim_next("worker-0")
    theirs = q.claim_next("worker-1")
    assert (mine.source, theirs.source) == ("autocode", "autotriage"), "fixture is wrong"

    assert q.recover_claimed(["worker-0"]) == 1

    rows = _interrupted_rows(q)
    assert len(rows) == 1
    assert rows[0]["queue_id"] == mine.id, "the recorded row is not the reclaimed one"
    assert rows[0]["source"] == "autocode"
    # Still `claimed` to `worker-1`, with its owner and stamp intact: a row this
    # call does not own must not be reclaimed, recorded, or un-assigned.
    held = q.get(theirs.id)
    assert held.state == "claimed", "the other worker's row was swept too"
    assert held.claimed_by == "worker-1"


def test_a_second_sweep_of_the_same_row_writes_no_second_run(q):
    """The sweep runs at every pool start, so it must be idempotent about a row
    it already reclaimed: a second boot with nothing new in flight must not
    invent an interruption, because the count is the point of writing it."""
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    q.mark_running(item.id)
    assert q.recover_claimed() == 1
    assert q.recover_claimed() == 0
    assert len(_interrupted_rows(q)) == 1
    assert q.get(item.id).state == "queued"


def test_the_interrupted_run_row_carries_the_swept_row_times(q):
    """Clause 2. `started_at` is the attempt's own start and `completed_at` the
    sweep instant, because `list_runs_joined` and `run_rollup_by_source` both
    window on `completed_at` — a row with only a start is invisible to the
    health report this exists to correct."""
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    q.mark_running(item.id)
    claimed_at = _backdate_claimed_at(q, item.id, seconds=90)

    q.recover_claimed()

    row = _interrupted_rows(q)[0]
    assert row["source"] == item.source
    assert row["started_at"] == claimed_at
    assert row["completed_at"] > row["started_at"], "completed_at is not the sweep instant"
    assert row["duration_seconds"] > 0, "a killed run reports no wall clock"
    # The duration is measured from the two stamps on the row, so the assertion
    # is that it EQUALS that span — not that it lands in a window around the
    # 90 s the fixture asked for. A hard window would make a loaded runner that
    # takes more than a few seconds to reach `recover_claimed` fail a clause test
    # for a reason that has nothing to do with the code.
    stamps = [datetime.fromisoformat(row[k]) for k in ("started_at", "completed_at")]
    assert row["duration_seconds"] == pytest.approx(
        (stamps[1] - stamps[0]).total_seconds(), abs=1e-6
    ), "duration_seconds is not the span between the row's own stamps"
    assert row["duration_seconds"] >= 90.0, (
        f"the 90 s the attempt ran is not on the row: {row['duration_seconds']}")


def test_the_interrupted_run_is_attributed_to_its_task(q):
    """Attribution comes from the payload the queue row already holds, so a
    killed scheduled-task run lands in its own task's health row instead of in
    `unattributed`."""
    q.enqueue("scheduled-task", "autonomy_task", payload={"task_id": 39})
    item = q.claim_next("worker-0")
    assert item.source == "scheduled-task", "fixture is wrong"
    q.mark_running(item.id)

    q.recover_claimed()

    row = _interrupted_rows(q)[0]
    assert row["task_id"] == "39"
    joined = q.list_runs_joined("scheduled-task", "2000-01-01T00:00:00+00:00")
    assert [j["run_id"] for j in joined if j["status"] == "interrupted"] == [row["run_id"]]


def test_a_run_recorded_before_completion_is_not_recorded_twice(q):
    """Clause 3. `record_run` happens before `mark_completed` (`pool.py`: the
    insert, then the queue transition), so a death in that gap leaves the row
    `running` with its run already written — and an unconditional sweep row
    would bill that run's wall clock twice.

    The guarded row travels with a second stranded row that has no run of its
    own. The guard has to skip one and record the other: a test that only
    asserts the skip would pass at base, where the sweep records nothing at all,
    and would report the very hole this item is about as correct behaviour."""
    already, owed = q.enqueue("autocode", "k"), q.enqueue("autotriage", "k")
    first, second = q.claim_next("worker-0"), q.claim_next("worker-1")
    assert {first.id, second.id} == {already, owed}, "fixture is wrong: not both rows"
    assert (first.source, second.source) == ("autocode", "autotriage"), "fixture is wrong"
    q.mark_running(first.id)
    q.mark_running(second.id)
    _backdate_claimed_at(q, first.id, seconds=30)
    _backdate_claimed_at(q, second.id, seconds=30)
    # The pool stamps `started_at` just after the claim, so it is at or after
    # `claimed_at` — which is exactly what tells the sweep this attempt already
    # reported itself.
    recorded_at = datetime.now(timezone.utc).isoformat()
    q.record_run(run_id=new_run_id(first.source), queue_id=first.id,
                 source=first.source, status="failed",
                 started_at=recorded_at, completed_at=recorded_at,
                 duration_seconds=30.0, summary="boom")

    assert q.recover_claimed() == 2, "fixture is wrong: both rows were not swept"

    rows = _interrupted_rows(q)
    assert [r["queue_id"] for r in rows] == [second.id], (
        "the sweep must skip the row that already reported itself and record the "
        "one that never did")
    assert rows[0]["source"] == "autotriage"
    assert len(q.list_runs()) == 2, (
        "two killed rows, one recorded attempt and one interruption — anything "
        "else is a count nobody can reconcile")


def test_a_retry_from_an_earlier_boot_still_gets_its_own_run(q):
    """The double-count guard must not become a mute button. The run already on
    the table belongs to a PREVIOUS attempt — it started before this claim — so
    this attempt, killed as well, still has to be recorded. Suppressing it would
    trade a double count for a silent one."""
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    q.mark_running(item.id)
    _backdate_claimed_at(q, item.id, seconds=3600)  # first attempt ran an hour ago
    assert q.recover_claimed() == 1, "fixture is wrong: the row was not swept"
    prior = _interrupted_rows(q)[0]

    requeued = q.claim_next("worker-0")
    assert requeued.id == item.id, "fixture is wrong: the row was not re-claimed"
    assert requeued.attempts == 2, "the retry did not carry its first attempt"
    q.mark_running(requeued.id)

    assert q.recover_claimed() == 1

    rows = _interrupted_rows(q)
    assert len(rows) == 2, "the second killed attempt was swallowed by the guard"
    assert prior["run_id"] in {r["run_id"] for r in rows}
    assert max(r["duration_seconds"] for r in rows) > 3000.0, (
        "the first attempt's hour of wall clock is not on the table")


def test_an_interrupted_row_is_counted_as_a_failure_by_the_rollup(q):
    """Clause 5. Interrupted rows are billed as failures so no run in the window
    falls outside ok/failed/skipped and `fail_rate` cannot be depressed by a
    source whose runs keep being killed."""
    q.enqueue("autocode", "k")
    item = q.claim_next("worker-0")
    q.mark_running(item.id)
    q.recover_claimed()
    rollup = q.run_rollup_by_source("2000-01-01T00:00:00+00:00")[item.source]
    assert (rollup["total"], rollup["ok"], rollup["failed"], rollup["skipped"]) == (
        1, 0, 1, 0)
    assert rollup["fail_rate"] == 1.0


def test_the_rollup_still_separates_success_from_interruption(q):
    """The same window, mixed: one clean run, one killed run, one skipped. The
    interruption lands in `failed` and nothing is left outside the three
    buckets, which is what makes `total` mean anything."""
    won = q.enqueue("s", "k")
    claimed = q.claim_next("worker-0")
    assert claimed.id == won, "fixture is wrong: not the row we enqueued"
    q.mark_completed(won)
    q.record_run(run_id=new_run_id("s"), queue_id=won, source="s",
                 status="success", started_at="2000-01-01T00:00:00+00:00",
                 completed_at=datetime.now(timezone.utc).isoformat(),
                 duration_seconds=60.0)
    dead = q.enqueue("s", "k")
    doomed = q.claim_next("worker-0")
    assert doomed.id == dead, "fixture is wrong: not the row we enqueued"
    q.mark_running(dead)
    assert q.recover_claimed() == 1, "fixture is wrong: the killed row was not swept"
    q.record_run(run_id=new_run_id("s"), queue_id=None, source="s", status="skipped",
                 started_at="2000-01-01T00:00:00+00:00",
                 completed_at=datetime.now(timezone.utc).isoformat(),
                 duration_seconds=0.0)

    rollup = q.run_rollup_by_source("2000-01-01T00:00:00+00:00")["s"]
    assert (rollup["total"], rollup["ok"], rollup["failed"], rollup["skipped"]) == (
        3, 1, 1, 1)
    assert rollup["fail_rate"] == pytest.approx(1 / 3)


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


def test_the_interrupted_run_reuses_the_run_id_the_pool_minted(q):
    """The run_id is minted BEFORE the claim is stamped `running`, so it survives
    a cancel or a SIGKILL on the row itself and the recovery sweep writes its row
    under the SAME id. That is what closes the log-vs-`runs` cross-check for real:
    the `run <id> starting` line names a run the table can then be queried for,
    instead of the attempt being logged under one id and recorded under another
    that never appeared in any log."""
    item = q.enqueue("s", "k")
    claimed = q.claim_next("worker-0")
    assert claimed.id == item, "fixture is wrong: not the row we enqueued"
    q.mark_running(item, "run_s_20260920_000000_dead01")

    assert q.recover_claimed() == 1, "fixture is wrong: the row was not swept"

    rows = _interrupted_rows(q)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_s_20260920_000000_dead01", (
        "the sweep invented a run_id the log never named")


def test_a_row_stranded_before_mark_running_still_gets_a_run_id(q):
    """The fallback half of the same seam. A row killed between `claim_next` and
    `mark_running` — or claimed by a build that predates the column — carries no
    id, and the sweep must still write a row rather than skip it for want of one:
    'started' without a run_id is still a run nobody recorded."""
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    assert q.get(item.id).claimed_at, "fixture is wrong: nothing was claimed"

    assert q.recover_claimed() == 1

    rows = _interrupted_rows(q)
    assert len(rows) == 1, "a claimed row with no stamped id went unrecorded"
    assert rows[0]["run_id"].startswith("run_s_"), "no id was minted for it"
    assert rows[0]["duration_seconds"] > 0


def test_a_row_poisoned_at_the_cap_is_recorded_as_interrupted_too(q):
    """The ceiling interaction, which is where the sweep's two jobs meet. A row
    at `max_attempts` is poisoned rather than requeued AND recorded as
    interrupted: its attempt died in flight like any other. The return value is
    the number REQUEUED, so it legitimately comes back lower than the number of
    rows written — pinning the two together is what stops a later 'fix' from
    silently dropping one of them.

    `_park` NULLs `claimed_at`, which is the row a sweep finds when a death left
    no claim stamp at all, so this is also the row that exercises the
    `started_at` fallback to `enqueued_at`: the attempt's start is then the
    ENQUEUE, and its billed duration includes however long it sat in the queue.
    That is what makes the guard's lower bound the enqueue rather than the claim
    on this path, so the fallback is asserted here rather than left to prose."""
    enqueued = q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    assert item.id == enqueued, "fixture is wrong: not the row we enqueued"
    q.mark_running(item.id, "run_s_20260920_000000_dead02")
    _park(q, item.id, attempts=q.max_attempts, state="running")

    assert q.recover_claimed() == 0, (
        "the at-cap row was requeued: the ceiling did not fire")
    assert q.get(item.id).state == "poisoned", "the ceiling did not poison the row"

    rows = _interrupted_rows(q)
    assert len(rows) == 1, "the poisoned row's killed attempt went unrecorded"
    assert rows[0]["queue_id"] == item.id
    row = rows[0]
    with q._connect() as conn:
        enqueued_at = conn.execute("SELECT enqueued_at FROM queue WHERE id=?",
                                   (item.id,)).fetchone()[0]
    assert row["run_id"] == "run_s_20260920_000000_dead02", (
        "the stamped id was dropped with the claim stamp")
    assert row["started_at"] == enqueued_at, (
        "a swept row with no claim stamp did not fall back to its enqueue")
    assert row["duration_seconds"] > 0, "the killed at-cap attempt reports no wall clock"
    stamps = [datetime.fromisoformat(row[k]) for k in ("started_at", "completed_at")]
    assert row["duration_seconds"] == pytest.approx(
        (stamps[1] - stamps[0]).total_seconds(), abs=1e-6
    ), "duration_seconds is not the span between the row's own stamps"


def test_the_guard_compares_instants_and_not_the_spelling_of_a_stamp(q):
    """The double-count guard decides "already recorded" on parsed timestamps, so
    its answer cannot depend on how the writer that made the prior row chose to
    spell it. `runs.started_at` is a plain TEXT column with no format constraint,
    and byte order compares that spelling: SQLite's own `CURRENT_TIMESTAMP`
    renders `2026-09-20 09:00:06.123456`, whose space at position 10 (0x20) sorts
    below every `2026-09-20T…` (0x54) whatever time it names — so a run that
    genuinely recorded itself reads as predating the claim, and the guard that is
    supposed to stop a run being billed twice is the thing that bills it twice.
    (Nothing in this module writes that form today; the point is that the column
    permits it and the guard must not care.) The same-second pair from the two
    spellings `_now_iso()` itself produces is asserted too, and there TEXT and
    clock already agree: the fix must not WEAKEN the guard either way."""
    claimed_at = "2026-09-20T09:00:05+00:00"
    recorded_at = "2026-09-20 09:00:06.123456"          # CURRENT_TIMESTAMP form
    assert recorded_at < claimed_at, (
        "fixture is wrong: TEXT orders these the right way round, so nothing here "
        "tests the spelling")
    # The naive form is read as UTC by the code under test; the fixture says so
    # explicitly rather than letting an offset-naive comparison raise.
    assert datetime.fromisoformat(recorded_at).replace(
               tzinfo=timezone.utc) > datetime.fromisoformat(claimed_at), (
        "fixture is wrong: the run is not genuinely recorded AFTER the claim")

    guarded = q.enqueue("s", "guarded")
    item = q.claim_next("worker-0")
    assert item.id == guarded, "fixture is wrong: not the row we enqueued"
    q.mark_running(item.id)
    _force_claimed_at(q, item.id, claimed_at)
    q.record_run(run_id="run_s_20260920_090006_recd01", queue_id=item.id, source="s",
                 status="failed", started_at=recorded_at,
                 completed_at="2026-09-20 09:00:07.123456", duration_seconds=0.9,
                 summary="boom")
    # The positive control. Asserting only that nothing was written is true at
    # base, where the sweep writes no rows at all — the assertion would then be
    # grading the absence this item exists to fix, not the guard. A second
    # stranded row with no run of its own proves the sweep IS recording here, so
    # the empty answer for the guarded row is the guard skipping ONE row.
    control = q.enqueue("s", "owed")
    other = q.claim_next("worker-1")
    assert other.id == control, "fixture is wrong: the control was not claimed"
    q.mark_running(other.id)
    _backdate_claimed_at(q, other.id, seconds=30)

    assert q.recover_claimed() == 2, "fixture is wrong: both rows were not swept"

    rows = _interrupted_rows(q)
    assert [r["queue_id"] for r in rows] == [other.id], (
        "the guard must skip the row that recorded itself and record the one "
        "that never did — an empty list here is the pre-fix hole, not a passing "
        "guard, and a second row is a run billed twice")


def test_the_guard_still_fires_for_the_two_spellings_this_module_writes(q):
    """The other half: parsing must not LOOSEN the guard. `isoformat()` omits the
    fractional part when `microsecond == 0`, so a claim and the run recorded
    inside that same second can be `…:05+00:00` and `…:05.123456+00:00`. As bytes
    those are already in clock order — `.` (0x2E) outranks `+` (0x2B) — and as
    instants the run is at or after the claim, so both readings must skip it."""
    claimed_at = "2026-09-20T09:00:05+00:00"
    recorded_at = "2026-09-20T09:00:05.123456+00:00"
    assert recorded_at > claimed_at, "fixture is wrong: TEXT does not order them"
    assert datetime.fromisoformat(recorded_at) > datetime.fromisoformat(claimed_at), (
        "fixture is wrong: the run is not recorded after the claim")

    guarded = q.enqueue("s", "guarded")
    item = q.claim_next("worker-0")
    assert item.id == guarded, "fixture is wrong: not the row we enqueued"
    q.mark_running(item.id)
    _force_claimed_at(q, item.id, claimed_at)
    q.record_run(run_id="run_s_20260920_090005_recd02", queue_id=item.id, source="s",
                 status="failed", started_at=recorded_at,
                 completed_at="2026-09-20T09:00:06+00:00", duration_seconds=0.876544,
                 summary="boom")
    # Same positive control as the test above: an empty list is also what the
    # code writes when it records nothing at all, so the skip is only evidence
    # once a row that DID need recording proves it got recorded.
    control = q.enqueue("s", "owed")
    other = q.claim_next("worker-1")
    assert other.id == control, "fixture is wrong: the control was not claimed"
    q.mark_running(other.id)
    _backdate_claimed_at(q, other.id, seconds=30)

    assert q.recover_claimed() == 2, "fixture is wrong: both rows were not swept"

    rows = _interrupted_rows(q)
    assert [r["queue_id"] for r in rows] == [other.id], (
        "a run recorded in the claim's own second was billed twice, or the "
        "un-recorded row went unrecorded")


def test_the_next_boot_sweep_survives_a_dead_attempt_s_stamped_run_id(tmp_path):
    """The seam the acceptance check actually rides on, across its real process
    boundary: `current_run_id` is written by the pool's process and read by the
    NEXT process's boot sweep (`app.on_event("startup")` -> `start_worker_pool`
    -> `WorkerPool.start` -> `recover_claimed`), so the value it finds can belong
    to an earlier attempt that already recorded a run. `runs.run_id` is a PRIMARY
    KEY, so writing this death under that id raises IntegrityError inside the
    sweep's transaction — and the sweep is one transaction, so the boot loses
    EVERY stranded row, not just this one. The sweep mints a fresh id instead:
    the cross-check loses one id it cannot resolve, the fleet keeps the rest."""
    path = tmp_path / "workers.db"
    pool_process = WorkQueue(path)
    item = pool_process.enqueue("s", "k")
    claimed = pool_process.claim_next("worker-0")
    assert claimed.id == item, "fixture is wrong: not the row we enqueued"
    dead_run_id = "run_s_20260919_000000_stale1"
    pool_process.mark_running(item, dead_run_id)
    pool_process.mark_completed(item)
    pool_process.record_run(run_id=dead_run_id, queue_id=item, source="s",
                            status="success", started_at="2026-09-19T00:00:00+00:00",
                            completed_at="2026-09-19T00:05:00+00:00",
                            duration_seconds=300.0)

    # The row is being held by a SECOND attempt whose death no arm recorded, and
    # the column still carries the FIRST attempt's id. `claim_next` clears it now
    # (see `test_a_new_claim_clears_the_previous_attempt_s_run_id`), so this state
    # is reachable only from a row stamped before that clear existed — written
    # directly here rather than trusted to a path the fixed code no longer takes.
    with pool_process._connect() as conn:
        conn.execute("UPDATE queue SET state='running', claimed_at=?, claimed_by=?, "
                     "current_run_id=? WHERE id=?",
                     (datetime.now(timezone.utc).isoformat(), "worker-0",
                      dead_run_id, item))
        conn.commit()

    next_boot = WorkQueue(path)
    assert next_boot.recover_claimed() == 1, (
        "the boot sweep died on the stamped id instead of recording the run")

    rows = [r for r in next_boot.list_runs() if r["status"] == "interrupted"]
    assert len(rows) == 1, "the killed second attempt went unrecorded"
    assert rows[0]["queue_id"] == item
    assert rows[0]["run_id"] != dead_run_id, (
        "the second death was written under the first attempt's id")
    assert next_boot.get(item).state == "queued"


def test_a_new_claim_clears_the_previous_attempt_s_run_id(q):
    """`current_run_id` means "the attempt NOW holding this row", and the pool
    mints a fresh id per attempt, so a value carried over from a prior attempt is
    a lie the next sweep would write a death under — the state the test above
    hand-builds. `claim_next` clears it on the way out, and a `mark_running` with
    no id clears it too, so no gap between the two leaves a stale stamp."""
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    q.mark_running(item.id, "run_s_20260920_000000_first1")
    q.record_run(run_id="run_s_20260920_000000_first1", queue_id=item.id, source="s",
                 status="failed",
                 started_at=datetime.now(timezone.utc).isoformat(),
                 completed_at=datetime.now(timezone.utc).isoformat(),
                 duration_seconds=5.0, summary="boom")

    assert q.mark_failed(item.id, "boom") == "queued", (
        "fixture is wrong: the row was not requeued for a retry")
    _clear_backoff(q, item.id)
    retry = q.claim_next("worker-0")
    assert retry.id == item.id, "fixture is wrong: the retry was not the same row"

    def _stamped() -> "str | None":
        with q._connect() as conn:
            return conn.execute("SELECT current_run_id FROM queue WHERE id=?",
                                (item.id,)).fetchone()[0]

    assert _stamped() is None, "the retry carried the dead attempt's run id"
    q.mark_running(item.id, "run_s_20260920_000000_second1")
    assert _stamped() == "run_s_20260920_000000_second1"


def test_mark_running_with_no_id_clears_a_stamp_the_claim_left_behind(q):
    """The other half of the same invariant. `mark_running` only fires on the
    claimed→running edge, so a row a pre-fix build left claimed-but-stamped is
    cleared by the transition that consumes it, not by a second call on a row that
    is already running (which the state guard makes a no-op)."""
    q.enqueue("s", "k")
    item = q.claim_next("worker-0")
    with q._connect() as conn:
        conn.execute("UPDATE queue SET current_run_id=? WHERE id=?",
                     ("run_s_20260919_000000_stale1", item.id))
        conn.commit()

    q.mark_running(item.id)

    with q._connect() as conn:
        left = conn.execute("SELECT current_run_id FROM queue WHERE id=?",
                            (item.id,)).fetchone()[0]
    assert left is None, "a no-id transition kept a stamp the next sweep would use"
