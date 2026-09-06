"""Poison sweep — workers/maintenance.py.

The sweep is the only thing in the system that ever looks at a poisoned queue
row again, so its two failure modes are both silent: a sweep that retries
forever burns GPU on a broken cause, and a sweep that quarantines everything
throws away work a restart would have fixed. Both are pinned here.

Run: .venvs/lloyd/bin/python -m pytest tests/test_worker_poison_sweep.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers import maintenance  # noqa: E402
from workers.queue import WorkQueue  # noqa: E402


@pytest.fixture
def queue(tmp_path) -> WorkQueue:
    return WorkQueue(tmp_path / "workers.db")


def poison(q: WorkQueue, error: str, *, source="domain-research",
           kind="research", payload=None, max_attempts=3) -> int:
    """Drive one item through the real failure path to state='poisoned'.

    priority=0 so claim_next reaches for this item and not whatever else the
    test has queued — the claim order is incidental to what is being pinned.
    """
    item_id = q.enqueue(source=source, kind=kind, payload=payload or {"topic": "x"},
                        priority=0, dedup_key=f"k-{error[:20]}-{source}-{payload}")
    for _ in range(max_attempts):
        q.claim_next("worker-0")
        q.mark_failed(item_id, error, max_attempts=max_attempts)
        _elapse_backoff(q, item_id)
    assert q.get(item_id).state == "poisoned"
    return item_id


def _elapse_backoff(q: WorkQueue, item_id: int) -> None:
    """Stand in for the wall-clock wait mark_failed's retry backoff imposes."""
    with q._connect() as conn:
        conn.execute("UPDATE queue SET not_before=NULL WHERE id=?", (item_id,))
        conn.commit()


# ── Classification ────────────────────────────────────────────────────────


@pytest.mark.parametrize("error", [
    "TimeoutError: exceeded max_duration_seconds=600",
    "ConnectError: All connection attempts failed",
    "httpx.RemoteProtocolError: Server disconnected without sending a response",
    "APIStatusError: 503 Service Unavailable",
])
def test_transient_errors_classify_transient(error):
    assert maintenance.classify(error) == "transient"


@pytest.mark.parametrize("error", [
    "unknown source domain-reserch",
    "KeyError: 'topic'",
    "FileNotFoundError: [Errno 2] No such file or directory: '/x/y.md'",
])
def test_structural_errors_classify_structural(error):
    assert maintenance.classify(error) == "structural"


def test_unrecognised_error_defaults_to_structural():
    """Retrying an error nobody has classified spends GPU on a guess."""
    assert maintenance.classify("Something nobody has seen before") == "structural"


def test_structural_wins_when_both_appear():
    """`TypeError` raised while handling a timeout is still deterministic."""
    assert maintenance.classify(
        "TypeError: unsupported operand, raised during TimeoutError handling"
    ) == "structural"


def test_signature_groups_the_same_failure_across_tunings():
    """`=600` and `=1800` are one problem, not two one-offs."""
    a = maintenance.signature("TimeoutError: exceeded max_duration_seconds=600")
    b = maintenance.signature("TimeoutError: exceeded max_duration_seconds=1800")
    assert a == b


def test_signature_separates_genuinely_different_failures():
    assert (maintenance.signature("KeyError: 'topic'")
            != maintenance.signature("TimeoutError: exceeded max=600"))


def test_signature_survives_an_empty_error():
    assert maintenance.signature(None) == maintenance.signature("")


# ── Revive vs quarantine ──────────────────────────────────────────────────


def test_transient_item_is_revived_for_exactly_one_more_claim(queue):
    item_id = poison(queue, "TimeoutError: exceeded max_duration_seconds=600")

    report = maintenance.sweep(queue, max_attempts=3, revive_delay_seconds=0)

    assert report["revived"] == 1 and report["quarantined"] == 0
    item = queue.get(item_id)
    assert item.state == "queued"
    # claim_next increments attempts, so the next claim lands on max_attempts
    # and a second failure re-poisons immediately.
    assert item.attempts == 2
    queue.claim_next("worker-0")
    assert queue.mark_failed(item_id, "TimeoutError: again", max_attempts=3) == "poisoned"


def test_structural_item_is_quarantined_not_retried(queue):
    item_id = poison(queue, "KeyError: 'topic'")

    report = maintenance.sweep(queue, revive_delay_seconds=0)

    assert report["quarantined"] == 1 and report["revived"] == 0
    item = queue.get(item_id)
    assert item.state == "quarantined"
    assert item.triage["class"] == "structural"


def test_revive_budget_is_spent_after_one_use(queue):
    """The second poisoning of a revived item is quarantined.

    Task #76's own log: "a weekly reset that does not fix the cause just
    re-poisons". A revive is one bounded retry, not a loop.
    """
    error = "TimeoutError: exceeded max_duration_seconds=600"
    item_id = poison(queue, error)

    maintenance.sweep(queue, max_attempts=3, revive_delay_seconds=0)
    queue.claim_next("worker-0")
    queue.mark_failed(item_id, error, max_attempts=3)
    assert queue.get(item_id).state == "poisoned"

    report = maintenance.sweep(queue, max_attempts=3, revive_delay_seconds=0,
                               repeat_threshold=99)

    assert report["quarantined"] == 1
    assert queue.get(item_id).state == "quarantined"
    assert "already revived" in queue.get(item_id).triage["reason"]


def test_revive_sets_a_backoff_before_the_item_is_claimable(queue):
    poison(queue, "TimeoutError: exceeded max_duration_seconds=600")

    maintenance.sweep(queue, revive_delay_seconds=300)

    assert queue.claim_next("worker-0") is None


def test_revive_is_skipped_when_the_source_re_enqueued_the_same_work(queue):
    """A poisoned row has lost its dedup_key, so a revive cannot coalesce."""
    payload = {"topic": "quantisation"}
    item_id = poison(queue, "TimeoutError: exceeded max=600", payload=payload)
    queue.enqueue(source="domain-research", kind="research", payload=payload,
                  dedup_key="fresh")

    report = maintenance.sweep(queue, revive_delay_seconds=0)

    assert report["quarantined"] == 1
    assert "already open" in queue.get(item_id).triage["reason"]


# ── Repeat offenders ──────────────────────────────────────────────────────


def test_a_recurring_transient_signature_stops_being_revived(queue):
    """Three poisonings of one signature is a broken cause, not bad luck."""
    error = "TimeoutError: exceeded max_duration_seconds=600"
    for n in range(3):
        item_id = poison(queue, error, payload={"topic": f"t{n}"})
        report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)
        if n < 2:
            assert report["revived"] == 1, f"pass {n} should still revive"
            queue.mark_completed(item_id)  # settle it before the next round
        else:
            assert report["quarantined"] == 1
            assert report["escalations"], "third poisoning must escalate"


def test_a_recurring_structural_signature_also_escalates(queue):
    """A source producing broken items repeatedly needs a human, not silence."""
    error = "KeyError: 'topic'"
    for n in range(3):
        poison(queue, error, payload={"topic": f"t{n}"})
        report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)
        assert report["quarantined"] == 1
    assert report["escalations"][0]["class"] == "structural"


def test_an_escalating_sweep_still_records_a_successful_run(queue, tmp_path, monkeypatch):
    """The sweep did its job; the fleet has the problem. Marking the run
    failed would misreport the sweep and erode the runs table."""
    monkeypatch.setattr(maintenance, "report_dir", lambda: tmp_path / "reports")
    error = "KeyError: 'topic'"
    for n in range(3):
        poison(queue, error, payload={"topic": f"t{n}"})
        report = maintenance.run_sweep(queue, {"revive_delay_seconds": 0})

    assert report["escalations"]
    runs = queue.list_runs(source=maintenance.SOURCE)
    assert runs[0]["status"] == "success"
    assert "1 recurring" in runs[0]["summary"]


def test_tally_forgets_signatures_older_than_retention(queue):
    poison(queue, "TimeoutError: exceeded max=600")
    maintenance.sweep(queue, revive_delay_seconds=0)
    assert any(k.startswith("sig:") for k in queue.wm_all(maintenance.SOURCE))

    future = datetime.now(timezone.utc) + timedelta(days=30)
    maintenance.sweep(queue, tally_retention_days=14, now=future)

    assert not any(k.startswith("sig:") for k in queue.wm_all(maintenance.SOURCE))


# ── run_sweep bookkeeping ─────────────────────────────────────────────────


def test_quiet_sweep_records_no_run_row(queue, tmp_path, monkeypatch):
    """96 "nothing poisoned" rows a day would bury the ones that matter."""
    monkeypatch.setattr(maintenance, "report_dir", lambda: tmp_path / "reports")

    maintenance.run_sweep(queue, {})

    assert queue.list_runs(source=maintenance.SOURCE) == []
    assert queue.wm_get(maintenance.SOURCE, "last_sweep_at")


def test_acting_sweep_records_a_run_and_writes_a_report(queue, tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, "report_dir", lambda: tmp_path / "reports")
    poison(queue, "KeyError: 'topic'")

    report = maintenance.run_sweep(queue, {"revive_delay_seconds": 0})

    runs = queue.list_runs(source=maintenance.SOURCE)
    assert len(runs) == 1 and "1 quarantined" in runs[0]["summary"]
    assert Path(report["report_path"]).exists()
    assert maintenance.last_sweep(queue)["quarantined"] == 1


def test_report_failure_does_not_lose_the_repair(queue, monkeypatch):
    """The queue is fixed either way; a note is the less important half."""
    monkeypatch.setattr(maintenance, "report_dir",
                        lambda: (_ for _ in ()).throw(OSError("no disk")))
    item_id = poison(queue, "KeyError: 'topic'")

    report = maintenance.run_sweep(queue, {"revive_delay_seconds": 0})

    assert report["report_path"] is None
    assert queue.get(item_id).state == "quarantined"


def test_sweep_leaves_healthy_items_alone(queue):
    live = queue.enqueue(source="gap-fill", kind="fill", payload={"a": 1})
    poison(queue, "KeyError: 'topic'")

    maintenance.sweep(queue, revive_delay_seconds=0)

    assert queue.get(live).state == "queued"
