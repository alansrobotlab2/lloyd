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
# Crash recovery
# ---------------------------------------------------------------------------


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
