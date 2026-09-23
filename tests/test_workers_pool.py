"""`WorkerPool` — the run-record contract, and staying off the event loop.

The pool had two tests before this file, both in `test_autonomy_scheduler.py`
and both about scheduled-task's timeout. Everything here is about the parts no
test covered: what a source's return value means, what happens when it means
nothing, and the rule that keeps a worker from freezing the backend.
"""

from __future__ import annotations

import asyncio
import logging
import re
import inspect
from types import SimpleNamespace

import pytest

from workers.pool import WorkerPool, normalize_result
from workers.queue import QueueItem, WorkQueue


def _item(source: str = "s", payload: dict | None = None, **kw) -> QueueItem:
    base = dict(id=1, source=source, kind="k", priority=50, payload=payload or {},
                dedup_key=None, state="running", attempts=1,
                enqueued_at="", claimed_at=None, claimed_by=None,
                completed_at=None, error=None)
    base.update(kw)
    return QueueItem(**base)


@pytest.fixture
def q(tmp_path) -> WorkQueue:
    return WorkQueue(tmp_path / "workers.db")


# ---------------------------------------------------------------------------
# The run-record contract
# ---------------------------------------------------------------------------


def test_a_bare_result_is_a_success():
    """Most sources finish by returning their artifact and never mention it."""
    out = normalize_result(_item(), {"summary": "wrote a note", "artifact_path": "/x.md"})
    assert out["status"] == "success"
    assert out["summary"] == "wrote a note"


@pytest.mark.parametrize("status", ["success", "failed", "skipped"])
def test_a_declared_status_is_honoured(status):
    assert normalize_result(_item(), {"status": status, "summary": "s"})["status"] == status


def test_an_unknown_status_falls_back_to_success_and_is_logged(caplog):
    with caplog.at_level("WARNING"):
        out = normalize_result(_item(), {"status": "finished-ish", "summary": "s"})
    assert out["status"] == "success"
    assert "unknown status" in caplog.text


def test_a_bare_skipped_key_is_a_skip_not_a_success():
    """The regression: `automod-regression` reports "cannot evaluate" this way.

    It returns `{"skipped": "<reason>"}` — a key where a status belongs — and
    all 22 of its runs were therefore recorded as successes with an empty
    summary. For a detector whose own docstring insists that a missing noise
    floor means *cannot evaluate*, never *no regression*, a skip recorded as a
    success is the failure it exists to prevent, and it is invisible on the
    dashboard and in the runs table alike.
    """
    out = normalize_result(_item(), {"skipped": "no promotion to check"})
    assert out["status"] == "skipped"
    assert out["summary"] == "no promotion to check", "the reason must reach the record"


def test_a_result_with_nothing_readable_in_it_is_logged(caplog):
    with caplog.at_level("WARNING"):
        normalize_result(_item(), {})
    assert "unreadable" in caplog.text


def test_a_non_dict_result_does_not_crash_the_worker(caplog):
    with caplog.at_level("WARNING"):
        out = normalize_result(_item(), "I am not a dict")
    assert out["status"] == "success" and out["summary"] == ""
    assert "not a dict" in caplog.text


def test_none_is_tolerated_quietly():
    assert normalize_result(_item(), None)["status"] == "success"


def test_record_fields_are_bounded_before_they_reach_the_database():
    out = normalize_result(_item(), {"summary": "s" * 900, "response": "r" * 99999})
    assert len(out["summary"]) == 500
    assert len(out["response"]) == 50000


def test_meta_must_be_a_mapping():
    assert normalize_result(_item(), {"summary": "s", "meta": ["not", "a", "dict"]})["meta"] == {}
    assert normalize_result(_item(), {"summary": "s", "meta": {"a": 1}})["meta"] == {"a": 1}


def test_the_task_id_prefers_the_result_then_the_payload():
    """A per-task view can only find a run that carries the id.

    Omitting it on the timeout and exception branches is what left 237 runs
    and 73.6 GPU-hours unattributable in this table.
    """
    item = _item(payload={"task_id": 36})
    assert normalize_result(item, {"summary": "s"})["task_id"] == "36"
    assert normalize_result(item, {"summary": "s", "task_id": 99})["task_id"] == "99"
    assert normalize_result(_item(), {"summary": "s"})["task_id"] is None


# ---------------------------------------------------------------------------
# Draining
# ---------------------------------------------------------------------------


async def _drain_one(q, monkeypatch, source_module, *, source_name="s", cfg=None):
    """Run exactly one item through a real pool, then stop it."""
    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {source_name: source_module},
                        raising=False)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {source_name: cfg or {"max_duration_seconds": 30}},
                        raising=False)
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        for _ in range(400):
            if q.list_runs(limit=5):
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()
    return q.list_runs(limit=5)


async def test_a_successful_run_completes_its_queue_row(q, monkeypatch):
    async def execute(item):
        return {"summary": "did the thing", "artifact_path": "/tmp/x.md"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    assert runs[0]["status"] == "success" and runs[0]["summary"] == "did the thing"
    assert q.get(1).state == "completed"


async def test_an_in_band_failure_does_not_send_the_item_back_for_a_retry(q, monkeypatch):
    """A handler that reports failure by returning is not asking to be re-run.

    Raising sends the item through the queue's retry path, and for a
    scheduled task that means re-running a whole timed-out job up to
    `max_attempts` times before the scheduler's own cooldown is consulted —
    3 x 600s on task #36.
    """
    async def execute(item):
        return {"status": "failed", "summary": "the task itself failed"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    assert runs[0]["status"] == "failed"
    assert q.get(1).state == "completed", "an in-band failure was retried"


async def test_a_raised_exception_is_recorded_and_requeued(q, monkeypatch):
    async def execute(item):
        raise ValueError("boom")

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    assert runs[0]["status"] == "failed"
    assert "ValueError: boom" in runs[0]["summary"]
    assert q.get(1).state == "queued", "a raised failure should be retried"


async def test_a_run_that_overruns_its_cap_is_stopped_and_recorded(q, monkeypatch):
    async def execute(item):
        await asyncio.sleep(30)

    q.enqueue("s", "k", payload={"task_id": 7})
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute),
                            cfg={"max_duration_seconds": 1})
    assert runs[0]["status"] == "failed"
    assert "max_duration_seconds=1" in runs[0]["summary"]
    assert runs[0]["task_id"] == "7", "a timed-out run must still name its task"


@pytest.mark.parametrize("opted_in", [True, False])
async def test_a_source_that_opts_in_is_due_again_when_its_run_ends(q, monkeypatch, opted_in):
    """Nothing woke a source when its own job finished, so autocode's next
    round was queued up to `interval_seconds` after the last one ended — a
    run `skipped` in milliseconds idled a free loop for most of 15 minutes."""
    from datetime import datetime, timezone
    import workers.sources as sources

    async def execute(item):
        return {"status": "skipped", "summary": "loop busy"}

    src = SimpleNamespace(NAME="s", execute=execute)
    if opted_in:
        src.REPOLL_ON_COMPLETE = True
    now = datetime.now(timezone.utc).isoformat()
    q.wm_set("s", "last_enqueue_check", now)
    q.enqueue("s", "k")
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {"s": src}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"s": {"max_duration_seconds": 30}}, raising=False)
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        for _ in range(200):
            if q.get(1).state == "completed" and not pool.status()["in_flight_count"]:
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)
    finally:
        await pool.stop()
    stamp = q.wm_get("s", "last_enqueue_check")
    if opted_in:
        assert datetime.fromisoformat(stamp).year == 1970, "not re-armed for the next pass"
    else:
        assert stamp == now, "a source that did not opt in had its clock moved"


def test_the_implement_source_opts_in_to_the_early_repoll():
    from workers.sources import autocode
    assert autocode.REPOLL_ON_COMPLETE is True


async def test_an_unregistered_source_is_poisoned_rather_than_retried(q, monkeypatch):
    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {}, raising=False)

    q.enqueue("ghost", "k")
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        for _ in range(200):
            if q.get(1).state == "poisoned":
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()
    assert q.get(1).state == "poisoned"


async def test_a_paused_pool_claims_nothing(q, monkeypatch):
    claimed = []

    async def execute(item):
        claimed.append(item.id)
        return {"summary": "ran"}

    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY",
                        {"s": SimpleNamespace(NAME="s", execute=execute)}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {"s": {}}, raising=False)

    q.enqueue("s", "k")
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    pool.pause(True)
    await pool.start()
    try:
        await asyncio.sleep(0.2)
        assert claimed == [], "a paused pool ran an item"
        assert pool.paused is True
        pool.pause(False)
        for _ in range(200):
            if claimed:
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()
    assert claimed == [1], "resuming did not release the item"


# ---------------------------------------------------------------------------
# A pause outlives a restart when a person took it, and not when a landing did
# ---------------------------------------------------------------------------
#
# 2026-09-22: three restarts with the pool paused by hand each came back running
# and claimed 3-6 jobs (4 of them autocode rounds) before a re-pause could land.
# A "restart" here is what it is in production: a new WorkerPool on the same
# workers.db.


async def test_an_operator_pause_survives_a_restart_and_the_new_pool_claims_nothing(q, monkeypatch):
    claimed = []

    async def execute(item):
        claimed.append(item.id)
        return {"summary": "ran"}

    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY",
                        {"s": SimpleNamespace(NAME="s", execute=execute)}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {"s": {}}, raising=False)

    WorkerPool(q, slots=1).pause(True)
    q.enqueue("s", "k")
    reborn = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    assert reborn.paused is True and reborn.paused_by == ["operator"]
    await reborn.start()
    try:
        await asyncio.sleep(0.2)
        assert claimed == [], "the restarted pool claimed work under a persisted pause"
    finally:
        await reborn.stop()


def test_an_operator_resume_is_persisted_too(q):
    WorkerPool(q, slots=1).pause(True)
    WorkerPool(q, slots=1).pause(False)
    assert WorkerPool(q, slots=1).paused is False


def test_an_automod_pause_does_not_survive_a_restart(q):
    """The promoter leaves its pause for the landing's restart to clear
    (`promote._POOL_PAUSED_BY_US`); persisted, every landing would leave the
    pool paused for good."""
    pool = WorkerPool(q, slots=1)
    pool.pause(True, owner="automod")
    assert pool.paused is True and pool.paused_by == ["automod"]
    assert WorkerPool(q, slots=1).paused is False


def test_an_automod_resume_does_not_lift_a_persons_pause(q):
    pool = WorkerPool(q, slots=1)
    pool.pause(True, owner="automod")
    pool.pause(True)                        # a person pauses during the landing
    pool.pause(False, owner="automod")      # the landing gives up and releases its own
    assert pool.paused is True and pool.paused_by == ["operator"]
    assert WorkerPool(q, slots=1).paused is True


def test_an_operator_resume_lifts_both(q):
    pool = WorkerPool(q, slots=1)
    pool.pause(True, owner="automod")
    pool.pause(True)
    pool.pause(False)
    assert pool.paused is False and pool.paused_by == []


def test_the_promoter_pauses_as_automod(monkeypatch):
    from scripts.automod import promote
    sent = []
    monkeypatch.setattr(promote, "_post", lambda url, payload, timeout=5.0: sent.append(payload) or True)
    promote.set_pool_paused(True)
    assert sent == [{"paused": True, "owner": "automod"}]


async def test_in_flight_state_is_reported_while_a_job_runs_and_cleared_after(q, monkeypatch):
    """The dashboard's live worker panel reads exactly this."""
    running = asyncio.Event()
    release = asyncio.Event()

    async def execute(item):
        running.set()
        await release.wait()
        return {"summary": "ran"}

    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY",
                        {"s": SimpleNamespace(NAME="s", execute=execute)}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"s": {"max_duration_seconds": 30}}, raising=False)

    q.enqueue("s", "k")
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        await asyncio.wait_for(running.wait(), timeout=5)
        status = pool.status()
        assert status["in_flight_count"] == 1
        assert status["in_flight"]["1"]["source"] == "s"
        release.set()
        for _ in range(200):
            if pool.status()["in_flight_count"] == 0:
                break
            await asyncio.sleep(0.02)
    finally:
        release.set()
        await pool.stop()
    assert pool.status()["in_flight_count"] == 0


async def test_starting_recovers_everything_a_dead_pool_was_holding(q, monkeypatch):
    """Not just the slot names this pool is about to use — see
    `test_workers_queue.test_recovery_releases_rows_claimed_by_slots_that_no_longer_exist`."""
    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {}, raising=False)

    q.enqueue("s", "k")
    stranded = q.claim_next("worker-9")
    q.mark_running(stranded.id)

    pool = WorkerPool(q, slots=2, poll_idle_seconds=0.01)
    await pool.start()
    try:
        assert q.get(stranded.id).state == "queued"
    finally:
        await pool.stop()


async def test_the_scheduler_survives_a_broken_config(q, monkeypatch):
    """The regression: `interval` was assigned inside the try block.

    The `sleep` that uses it is outside, so the very first raise from
    `get_sources_config()` fell through to a `NameError` and killed the only
    task that enqueues work — silently, since the exception died with it.
    """
    calls = {"n": 0}

    def exploding_config():
        calls["n"] += 1
        raise RuntimeError("config is broken")

    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config", exploding_config, raising=False)

    pool = WorkerPool(q, slots=0, poll_idle_seconds=0.01)
    await pool.start()
    try:
        await asyncio.sleep(0.2)
        assert calls["n"] >= 1, "the scheduler never read the config"
        # It sleeps for `interval` between passes, so what matters is that it
        # reached the sleep at all rather than dying on the way there.
        assert not pool._scheduler_task.done(), (
            "the scheduler task exited on a bad config read: "
            f"{pool._scheduler_task.exception() if pool._scheduler_task.done() else ''}")
    finally:
        await pool.stop()


def _age_of_watermark(q, source: str) -> float:
    from datetime import datetime, timezone
    stamp = q.wm_get(source, "last_enqueue_check")
    return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()


async def _one_pass(q, monkeypatch, outcome, cfg):
    import workers.sources as sources
    calls = []

    async def enqueue_if_due(queue, src_cfg):
        calls.append(src_cfg)
        return outcome

    monkeypatch.setattr(sources, "SOURCE_REGISTRY",
                        {"s": SimpleNamespace(NAME="s", enqueue_if_due=enqueue_if_due)},
                        raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {"s": cfg}, raising=False)
    pool = WorkerPool(q, slots=0, poll_idle_seconds=0.01)
    await pool._scheduler_pass()
    return calls


async def test_a_declined_check_is_due_again_in_retry_seconds(q, monkeypatch):
    """The 900 s clock idled the loop: a poll that could not start a round
    (a promotion under observation) stamped the watermark like one that did,
    and the next look was a full interval later — ~12 h of a free loop over
    fifty rounds."""
    from datetime import datetime, timedelta
    import workers.sources as sources
    cfg = {"enabled": True, "interval_seconds": 900, "retry_seconds": 60}
    assert len(await _one_pass(q, monkeypatch, sources.DECLINED, cfg)) == 1
    age = _age_of_watermark(q, "s")
    assert 835 <= age <= 845, f"stamped {age:.0f}s ago; due again in {900 - age:.0f}s"
    assert await _one_pass(q, monkeypatch, sources.DECLINED, cfg) == [], "not inside the minute"
    # A minute later it is due, and the source is asked again.
    stamp = datetime.fromisoformat(q.wm_get("s", "last_enqueue_check"))
    q.wm_set("s", "last_enqueue_check", (stamp - timedelta(seconds=61)).isoformat())
    assert len(await _one_pass(q, monkeypatch, sources.DECLINED, cfg)) == 1


@pytest.mark.parametrize("outcome", [None, "enqueued", "declined-without-retry"])
async def test_every_other_outcome_advances_the_full_interval(q, monkeypatch, outcome):
    import workers.sources as sources
    cfg = {"enabled": True, "interval_seconds": 900, "retry_seconds": 60}
    if outcome == "declined-without-retry":
        outcome, cfg = sources.DECLINED, {"enabled": True, "interval_seconds": 900}
    await _one_pass(q, monkeypatch, outcome, cfg)
    assert _age_of_watermark(q, "s") < 5
    # ...and not due again yet.
    assert await _one_pass(q, monkeypatch, outcome, cfg) == []


async def test_a_broken_retry_seconds_costs_the_fast_retry_not_the_pass(q, monkeypatch):
    import workers.sources as sources
    cfg = {"enabled": True, "interval_seconds": 900, "retry_seconds": "soon"}
    await _one_pass(q, monkeypatch, sources.DECLINED, cfg)
    assert _age_of_watermark(q, "s") < 5


# ---------------------------------------------------------------------------
# Event-loop discipline
# ---------------------------------------------------------------------------


def test_no_source_blocks_the_event_loop_in_execute():
    """`execute` runs on the backend's one event loop.

    That loop serves every HTTP request and streams every chat turn, so a
    source that calls `subprocess.run` from it does not slow the pool down —
    it stops Lloyd answering. `automod_regression.execute` did exactly that,
    with two 900-second eval arms and a `git worktree add` between them, and
    the 109-second run in its history is 109 seconds of dead backend.

    The rule is mechanical: a blocking call inside an `async def execute` is a
    bug, and moving the body to a helper the coroutine hands to
    `asyncio.to_thread` is the fix.
    """
    import workers.sources as sources_pkg

    blocking = ("subprocess.run", "subprocess.check_output", "subprocess.call",
                "time.sleep(", "urlopen(")
    offenders = []
    for name, mod in sources_pkg.SOURCE_REGISTRY.items():
        execute = getattr(mod, "execute", None)
        if execute is None:
            continue
        src = inspect.getsource(execute)
        for pattern in blocking:
            if pattern in src:
                offenders.append(f"{name}.execute calls {pattern}")
    assert not offenders, "; ".join(offenders)


def test_every_registered_source_has_the_interface_the_pool_calls():
    import workers.sources as sources_pkg

    assert sources_pkg.SOURCE_REGISTRY, "no sources registered"
    for name, mod in sources_pkg.SOURCE_REGISTRY.items():
        assert getattr(mod, "NAME", None) == name
        assert inspect.iscoroutinefunction(mod.execute), f"{name}.execute is not async"
        assert inspect.iscoroutinefunction(mod.enqueue_if_due), \
            f"{name}.enqueue_if_due is not async"
        assert isinstance(getattr(mod, "DEFAULT_PRIORITY", None), int)


def test_the_pool_and_its_startup_hook_agree_on_the_default_slot_count():
    """They disagreed (4 in the class, 8 in the router), so "how many workers
    with no `workers.slots` set" had two answers depending on which you read."""
    import app.routers.workers as router

    class_default = inspect.signature(WorkerPool.__init__).parameters["slots"].default
    assert f'cfg.get("slots", {class_default})' in inspect.getsource(router.start_worker_pool)


# ---------------------------------------------------------------------------
# A run killed by a restart (#1137)
# ---------------------------------------------------------------------------


async def test_a_run_killed_by_a_pool_restart_is_recorded_under_its_logged_run_id(
        q, monkeypatch, caplog):
    """The death this item is about, through the real pool. `stop()` calls
    `task.cancel()` on the worker, `CancelledError` is a `BaseException`, and
    neither of `_run_item`'s two recording arms catches a `BaseException` — so
    the attempt ends with the queue row left `running` and no `runs` row at all.
    The next pool's startup sweep is the only thing that still knows it happened,
    and from that moment on the killed attempt is a queryable fact."""
    import workers.sources as sources

    started = asyncio.Event()

    async def execute(item):
        started.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(sources, "SOURCE_REGISTRY",
                        {"s": SimpleNamespace(NAME="s", execute=execute)}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"s": {"max_duration_seconds": 300}}, raising=False)
    q.enqueue("s", "k")

    caplog.set_level(logging.INFO, logger="lloyd-workers.pool")
    first = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await first.start()
    await asyncio.wait_for(started.wait(), timeout=10)
    await first.stop()

    assert q.list_runs() == [], "the killed attempt wrote a run row after all"
    assert q.get(1).state == "running", "the row was not left stranded"

    second = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await second.start()
    try:
        rows = [r for r in q.list_runs() if r["status"] == "interrupted"]
    finally:
        await second.stop()

    assert len(rows) == 1, "the startup sweep did not record the killed attempt"
    assert rows[0]["queue_id"] == 1
    assert rows[0]["duration_seconds"] > 0

    # The seam this item is actually about: the log named a run_id, and the row
    # is written under THAT id — so the logged-run_id-vs-`SELECT run_id FROM
    # runs` cross-check stops producing new logged-but-unrecorded runs. Before
    # the row carried its id, the sweep minted a fresh one and the run in the log
    # stayed missing from the table forever.
    logged = re.findall(r"run_id=(\S+)", caplog.text)
    assert logged, "the pool never logged the attempt starting"
    assert logged[0] == rows[0]["run_id"], (
        f"log named {logged[0]}, the table recorded {rows[0]['run_id']}")
