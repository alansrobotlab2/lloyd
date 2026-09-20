"""Poison sweep — workers/maintenance.py.

The sweep is the only thing in the system that ever looks at a poisoned queue
row again, so its two failure modes are both silent: a sweep that retries
forever burns GPU on a broken cause, and a sweep that quarantines everything
throws away work a restart would have fixed. Both are pinned here.

Run: .venvs/lloyd/bin/python -m pytest tests/test_worker_poison_sweep.py
"""
from __future__ import annotations

import json
import logging
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


def _seed_tally(q: WorkQueue, source: str, error: str, count: int) -> None:
    """Pre-load the cross-sweep (source, signature) tally with `count` poisonings.

    This is the live shape #1295 found rather than a synthetic one: the
    watermark `sig:93b180d28f99` — source `bench-mine`, signature `ConnectError:
    All connection attempts failed` — read `count: 5` against
    `repeat_threshold: 3`, while the automod ledger recorded 13 `promoted`
    landings between 00:35Z and 05:57Z that day and every claim caught inside a
    landing's backend restart writes that one string. So the *next* innocent
    item was condemned on its first poisoning.
    """
    now = datetime.now(timezone.utc)
    sig = maintenance.signature(error)
    q.wm_set(maintenance.SOURCE, maintenance._tally_key(source, sig), json.dumps({
        "source": source,
        "signature": sig,
        "count": count,
        "first_seen": now.isoformat(),
        "last_seen": now.isoformat(),
    }))


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


def test_a_first_poisoning_on_a_hot_transient_signature_is_revived(queue):
    """A tally of collisions cannot condemn an item the classifier cleared.

    Acceptance clause 1 (#1295). Before the fix this same sweep — a fresh
    `bench-mine` item, its tally seeded to 5 — was quarantined with
    `class: "transient"` written into the same triage JSON as the reason "6
    poisonings on this signature in 14d — the cause is not transient": the row's
    own field printed against itself. The counted poisonings are claims that died
    while the backend was restarting, not broken items.
    """
    error = "ConnectError: All connection attempts failed"
    _seed_tally(queue, "bench-mine", error, count=5)
    item_id = poison(queue, error, source="bench-mine", kind="mine",
                     payload={"topic": "t0"})

    report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)

    assert report["revived"] == 1 and report["quarantined"] == 0
    item = queue.get(item_id)
    assert item.state == "queued"
    assert item.attempts == 2                      # one more claim, not a fresh budget
    assert item.triage["class"] == "transient"
    assert item.triage["occurrences"] == 6         # 5 counted + this poisoning
    assert "not transient" not in item.triage["reason"]


def test_a_recurring_transient_signature_is_revived_and_still_escalates(queue):
    """Three transient poisonings of one signature: revive the item, raise the alert.

    Rewritten from `test_a_recurring_transient_signature_stops_being_revived`,
    which asserted `report["quarantined"] == 1` and
    `report["escalations"], "third poisoning must escalate"` on the third pass.
    The escalation half of that is kept here unchanged; the quarantine half is
    what acceptance clauses 1 and 3 (#1295) say is wrong — killing a `transient`
    item because the *cause* is common throws away the work while the cause
    (restarts) keeps happening anyway.
    """
    error = "TimeoutError: exceeded max_duration_seconds=600"
    for n in range(3):
        item_id = poison(queue, error, payload={"topic": f"t{n}"})
        report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)

        assert report["revived"] == 1, f"pass {n} must still revive"
        assert report["quarantined"] == 0, f"pass {n} must not quarantine"
        assert queue.get(item_id).state == "queued"
        # Clause 3: on the pass the tally reaches repeat_threshold, the churn
        # must reach a human even though nothing was quarantined.
        assert len(report["escalations"]) == (1 if n == 2 else 0), f"pass {n}"
        queue.mark_completed(item_id)  # settle it before the next round

    assert report["escalations"][0] == {
        "source": "domain-research",
        "signature": maintenance.signature(error),
        "count": 3,
        "class": "transient",
    }


def test_a_hot_transient_tally_does_not_outvote_the_revive_budget(queue):
    """The per-item budget still condemns a second collision — and names itself.

    Clause 1 keeps `max_revives` unchanged and clause 2 requires this reason to
    be about the budget. Rows 8292/8334 already had `revives: 1`, so a landing
    that catches the same item twice is still dropped; widening the budget for
    restart collisions is a decision this item did not make.
    """
    error = "ConnectError: All connection attempts failed"
    _seed_tally(queue, "bench-mine", error, count=5)
    item_id = poison(queue, error, source="bench-mine", kind="mine",
                     payload={"topic": "t0"})

    maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)
    queue.claim_next("worker-0")
    assert queue.mark_failed(item_id, error, max_attempts=3) == "poisoned"

    report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)

    assert report["quarantined"] == 1 and report["revived"] == 0
    triage = queue.get(item_id).triage
    assert triage["class"] == "transient"
    assert triage["reason"] == "transient, but already revived 1x"


def test_a_hot_transient_tally_does_not_outvote_the_duplicate_rule(queue):
    """An equivalent open item still wins over a revive (clause 1's exception)."""
    error = "ConnectError: All connection attempts failed"
    payload = {"topic": "quantisation"}
    _seed_tally(queue, "bench-mine", error, count=5)
    item_id = poison(queue, error, source="bench-mine", kind="mine", payload=payload)
    queue.enqueue(source="bench-mine", kind="mine", payload=payload, dedup_key="fresh")

    report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)

    assert report["quarantined"] == 1 and report["revived"] == 0
    triage = queue.get(item_id).triage
    assert triage["class"] == "transient"
    assert triage["reason"] == (
        "an equivalent item is already open — reviving would duplicate it"
    )


def test_a_quarantine_reason_never_denies_the_class_written_beside_it(queue):
    """Clause 2 (#1295): no reason string may contradict its own triage `class`.

    One sweep, two condemned `bench-mine` items, both tallies already past
    `repeat_threshold`: a transient one that has spent its revive budget, and a
    structural one. Only the structural row may be told its cause is not
    transient.
    """
    transient = "ConnectError: All connection attempts failed"
    structural = "KeyError: 'topic'"
    _seed_tally(queue, "bench-mine", transient, count=5)
    _seed_tally(queue, "bench-mine", structural, count=5)

    transient_id = poison(queue, transient, source="bench-mine", kind="mine",
                          payload={"topic": "t0"})
    maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)
    queue.claim_next("worker-0")
    assert queue.mark_failed(transient_id, transient, max_attempts=3) == "poisoned"
    structural_id = poison(queue, structural, source="bench-mine", kind="mine",
                           payload={"topic": "s0"})

    report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)

    assert report["revived"] == 0
    condemned = {a["item_id"]: a for a in report["actions"]
                 if a["action"] == "quarantined"}
    assert set(condemned) == {transient_id, structural_id}
    for item_id, action in condemned.items():
        triage = queue.get(item_id)
        assert triage.state == "quarantined"
        for reason in (action["reason"], triage.triage["reason"]):
            # One direction only, exactly as clause 2 is written: the phrase is
            # admissible for a structural row, never for any other class.
            if "not transient" in reason:
                assert triage.triage["class"] == "structural", (item_id, reason)
    assert queue.get(transient_id).triage["reason"] == "transient, but already revived 1x"
    assert "not retryable" in queue.get(structural_id).triage["reason"]


def test_two_items_on_one_hot_signature_escalate_once(queue):
    """One churning signature is one finding per sweep, not one per item.

    Both items are revived, so the count the entry reports is the tally when the
    signature was first detected in this sweep (5 seeded + this poisoning = 6);
    the second item still revives and still pushes the watermark to 7.
    """
    error = "ConnectError: All connection attempts failed"
    _seed_tally(queue, "bench-mine", error, count=5)
    first = poison(queue, error, source="bench-mine", kind="mine",
                   payload={"topic": "t0"})
    second = poison(queue, error, source="bench-mine", kind="mine",
                    payload={"topic": "t1"})

    report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)

    assert report["revived"] == 2 and report["quarantined"] == 0
    assert [queue.get(i).state for i in (first, second)] == ["queued", "queued"]
    assert len(report["escalations"]) == 1
    assert report["escalations"][0]["count"] == 6


def test_a_recurring_structural_signature_also_escalates(queue):
    """A source producing broken items repeatedly needs a human, not silence."""
    error = "KeyError: 'topic'"
    for n in range(3):
        poison(queue, error, payload={"topic": f"t{n}"})
        report = maintenance.sweep(queue, revive_delay_seconds=0, repeat_threshold=3)
        assert report["quarantined"] == 1
    assert report["escalations"][0]["class"] == "structural"


def test_a_revived_transient_churn_still_reaches_every_alert_surface(queue, tmp_path,
                                                                    monkeypatch, caplog):
    """Clause 3 (#1295): reviving the item must not silence the alert.

    `logger.error`, the `## Escalations` section of the sweep report and
    `runs.meta_json` are the only places a source that churns forever shows up,
    and once a transient item is revived it is no longer sitting in
    `quarantined` to draw the eye. The escalation entry carries source,
    signature, count and class; the log line names the recurrence as the
    evidence and claims no action it did not take.
    """
    monkeypatch.setattr(maintenance, "report_dir", lambda: tmp_path / "reports")
    error = "ConnectError: All connection attempts failed"
    _seed_tally(queue, "bench-mine", error, count=5)
    poison(queue, error, source="bench-mine", kind="mine", payload={"topic": "t0"})

    with caplog.at_level(logging.ERROR, logger="lloyd-workers.maintenance"):
        report = maintenance.run_sweep(
            queue, {"revive_delay_seconds": 0, "repeat_threshold": 3})

    assert report["revived"] == 1 and report["quarantined"] == 0
    assert report["escalations"] == [{
        "source": "bench-mine", "signature": maintenance.signature(error),
        "count": 6, "class": "transient",
    }]

    runs = queue.list_runs(source=maintenance.SOURCE)
    assert runs[0]["status"] == "success"
    assert "1 recurring" in runs[0]["summary"]
    assert json.loads(runs[0]["meta_json"])["escalations"] == report["escalations"]

    note = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "## Escalations" in note
    assert "**bench-mine** — 6x" in note

    assert "bench-mine has poisoned 6x" in caplog.text
    assert "transient" in caplog.text
    assert "quarantined rather than retried" not in caplog.text


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
