"""`WorkerPool` — the run-record contract, and staying off the event loop.

The pool had two tests before this file, both in `test_autonomy_scheduler.py`
and both about scheduled-task's timeout. Everything here is about the parts no
test covered: what a source's return value means, what happens when it means
nothing, and the rule that keeps a worker from freezing the backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import inspect
from types import SimpleNamespace

import pytest

from datetime import datetime, timezone

from workers import dispatch_watch
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


async def test_a_returned_deferral_holds_the_row_out_of_the_next_tick(q, monkeypatch):
    """`defer_seconds` is the third thing a source can ask for (#1714).

    An in-band `{"status": "failed"}` used to have exactly two spacings: the
    queue's backoff, reachable only by RAISING, or the source's own
    `interval_seconds`. `youtube-digest`'s infra branch needed the second to stop
    being the whole story — a primary outage re-offered the same videos every
    300 s with a fresh row and `attempts=1` each time — without taking the first,
    because `mark_failed` counts an attempt against work the engine never got. So
    the run stays recorded and terminal-in-effect while the ROW waits behind
    `not_before`. The control is `test_an_in_band_failure_does_not_send_the_item_back_for_a_retry`
    one test up: a `failed` with no deferral still completes.
    """
    async def execute(item):
        return {"status": "failed", "summary": "turn produced nothing",
                "defer_seconds": 900}

    q.enqueue("s", "k", dedup_key="yd:abc")
    before = datetime.now(timezone.utc)
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))

    assert runs[0]["status"] == "failed", "the deferral changed what the run records"
    row = q.get(1)
    assert row.state == "queued", "a deferred in-band failure completed its row"
    assert row.not_before, "the row was released with no spacing: it is queueable next tick"
    held = (datetime.fromisoformat(row.not_before) - before).total_seconds()
    assert 800 < held <= 905, f"the row waits {held:.0f}s, not the 900 the source asked for"
    assert q.claim_next("w0") is None, "the deferred row was claimable inside its window"
    assert q.enqueue("s", "k", dedup_key="yd:abc") is None, (
        "the deferred row released its dedup key, so the next tick enqueues a second "
        "row for the same work and the bound never applies to one row twice")


async def test_a_deferral_on_a_successful_run_is_ignored(q, monkeypatch, caplog):
    """Deferring a success would re-run finished work: a second note, a second send.

    The field is a source's request, not the pool's policy, so the policy lives
    here — including the junk cases, which must land on the ordinary completion
    path with a warning rather than raise inside a worker.
    """
    async def execute(item):
        return {"status": "success", "summary": "wrote the note", "defer_seconds": 900}

    q.enqueue("s", "k")
    with caplog.at_level("WARNING"):
        await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    assert q.get(1).state == "completed" and q.get(1).not_before is None
    assert "SUCCESSFUL" in caplog.text, (
        "the pool ignored a deferral on a success silently — the source believes it "
        "backed off when it did not")

    assert normalize_result(_item(), {"status": "failed", "summary": "x",
                                      "defer_seconds": "900"})["defer_seconds"] == 900.0
    for junk in ("soon", [], float("inf"), float("nan"), -5, 0):
        assert normalize_result(_item(), {"status": "failed", "summary": "x",
                                          "defer_seconds": junk})["defer_seconds"] is None, (
            f"defer_seconds={junk!r} was accepted; a non-finite or non-positive value "
            "would park the row for a time nobody stated")
    assert normalize_result(_item(), {"status": "failed", "summary": "x"})["defer_seconds"] is None


async def test_the_digest_infra_turn_end_to_end_leaves_its_row_waiting(q, monkeypatch, tmp_path):
    """One run across all three modules this change touches: source, pool, queue.

    Each layer has its own node, and the seam between them is where #1714 lived —
    the digest module returned a number, the pool had to carry it, the queue had to
    hold the row by it, and the claim query had to honour it. Driving the real
    `youtube_digest.execute` through a real `WorkerPool` against a real
    `WorkQueue` is the only way to see the three agree: any one of them could
    satisfy its own test while the chain still re-offered the video on the tick.
    """
    from workers.sources import youtube_digest as Y
    from tests.test_youtube_digest_source import _Script, _meta, _turn

    monkeypatch.setattr(Y, "BACKLOG_DIR", tmp_path / "backlog")
    monkeypatch.setattr(Y, "_vault_dirty_paths", lambda: set())
    monkeypatch.setattr(Y, "_script", _Script({"ok": True, "meta": _meta(tmp_path)}))
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", stop_reason=None))

    q.enqueue(Y.NAME, "video",
              payload={"channel": "ai-engineer", "video_id": "abc123"},
              dedup_key="youtube-digest:ai-engineer:abc123")
    before = datetime.now(timezone.utc)
    runs = await _drain_one(q, monkeypatch,
                            SimpleNamespace(NAME=Y.NAME, execute=Y.execute),
                            source_name=Y.NAME)

    assert runs[0]["status"] == "failed" and runs[0]["source"] == Y.NAME
    assert json.loads(runs[0]["meta_json"])["infra"] is True, (
        "the run record is the owed check's only evidence of an infra failure")
    row = q.get(1)
    assert row.state == "queued" and row.dedup_key == "youtube-digest:ai-engineer:abc123"
    held = (datetime.fromisoformat(row.not_before) - before).total_seconds()
    assert 800 < held <= 905, (
        f"the row waits {held:.0f}s — the digest asked for {Y.INFRA_DEFER_SECONDS}s "
        "and the pool or the queue dropped it")
    assert q.claim_next("w0") is None, (
        "the row was claimable inside its window: across the three modules the video "
        "would still be re-offered every 300 s tick, which is the bug")


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


# ── GET /api/workers/status carries the mitigation block (#703) ─────────────

async def _status_body(q, monkeypatch) -> dict:
    import json

    import app.routers.workers as W
    monkeypatch.setattr(W, "get_queue", lambda: q)
    monkeypatch.setattr(W, "get_pool", lambda: None)
    return json.loads((await W.workers_status()).body)


async def test_the_status_says_the_drill_never_ran_when_it_has_not(q, monkeypatch, tmp_path):
    import app.paths as paths
    monkeypatch.setattr(paths, "MITIGATION_DRILL_STATE", tmp_path / "absent.json")
    body = await _status_body(q, monkeypatch)
    assert body["mitigation"] == {"state": "never-run"}


async def test_an_unreadable_drill_state_is_reported_not_raised(q, monkeypatch, tmp_path):
    import app.paths as paths
    bad = tmp_path / "mitigation_drill.json"
    bad.write_text("{not json")
    monkeypatch.setattr(paths, "MITIGATION_DRILL_STATE", bad)
    body = await _status_body(q, monkeypatch)
    assert body["mitigation"]["state"] == "never-run"
    assert body["mitigation"]["error"]


async def test_the_status_carries_each_surfaces_last_measured_mitigation(q, monkeypatch, tmp_path):
    import app.paths as paths
    from app import mitigation_state
    state = tmp_path / "mitigation_drill.json"
    monkeypatch.setattr(paths, "MITIGATION_DRILL_STATE", state)
    mitigation_state.record(
        [{"surface": "session_cancel", "classification": "in-flight", "seconds": 0.021, "ok": True},
         {"surface": "pool_pause", "classification": "dispatch-only", "seconds": None, "ok": True}],
        at="2026-09-24T20:00:00+00:00")
    body = await _status_body(q, monkeypatch)
    # #2153: every surface entry now carries the aggregate over its history too.
    # This fixture records once, so the median IS that reading and n is 1 — the
    # exact-equality form is kept deliberately, because a fifth key here is a
    # fifth thing `/api/workers/status` sends on every call, and a null median
    # for a dispatch-only control is the honest answer rather than a missing
    # measurement (`seconds: None` already says a pause times no stop).
    assert body["mitigation"] == {
        "session_cancel": {"classification": "in-flight", "seconds": 0.021,
                           "at": "2026-09-24T20:00:00+00:00",
                           "median_seconds": 0.021, "n": 1},
        "pool_pause": {"classification": "dispatch-only", "seconds": None,
                       "at": "2026-09-24T20:00:00+00:00",
                       "median_seconds": None, "n": 1},
    }


async def test_the_status_median_is_the_history_and_not_the_last_drill(q, monkeypatch, tmp_path):
    """#2153 clause 4's other half: the route publishes the aggregate, so the
    number an operator reads is the control's typical stop-time and not whatever
    the newest drill happened to measure.

    Five readings at 0.010 … 0.050 s, then a sixth at 9 s. `seconds` — the latest
    reading — moves to 9.0; `median_seconds` does not, because six values median
    to the mean of their middle pair, (0.030 + 0.040) / 2 = 0.035, and one slow
    drill is not a regression in the control. Read the latest value instead of
    the median and this goes red on 0.035 against 9.0, which is the whole
    difference between the two numbers.
    """
    import app.paths as paths
    from app import mitigation_state

    state = tmp_path / "mitigation_drill.json"
    monkeypatch.setattr(paths, "MITIGATION_DRILL_STATE", state)
    for i, seconds in enumerate((0.010, 0.020, 0.030, 0.040, 0.050), start=1):
        mitigation_state.record(
            [{"surface": "session_cancel", "classification": "in-flight",
              "seconds": seconds, "ok": True}], at=f"2026-10-0{i}T00:00:00+00:00")
    mitigation_state.record(
        [{"surface": "session_cancel", "classification": "in-flight",
          "seconds": 9.0, "ok": True}], at="2026-10-06T00:00:00+00:00")

    entry = (await _status_body(q, monkeypatch))["mitigation"]["session_cancel"]
    assert entry["median_seconds"] == pytest.approx(0.035), entry
    assert entry["n"] == 6, "six drills recorded, six counted"
    assert entry["seconds"] == 9.0, "the latest reading is still published beside it"


# ---------------------------------------------------------------------------
# #1550 — the alarm's pause verdict is the database's, not this process's
#
# An operator pause is persisted (`29d88695`) precisely so a restart cannot
# resume a pool a person stopped. The same durability is what makes it invisible:
# a 16.5 h hold on 2026-09-24/25 produced eight `autonomy scheduler may be
# stalled` alerts and not one mention of the pause, because the alarm read
# nothing but queue ages and task files. These tests pin where the alert's answer
# comes from.
# ---------------------------------------------------------------------------


def _age_a_queued_row(q, *, age_min: int) -> int:
    """A queued scheduled-task row `age_min` minutes old, by hand.

    Raw SQL because the age IS the fixture: `enqueue` stamps `enqueued_at` with
    the wall clock. 990 min is the item's queue row 510, eleven times the
    starving threshold (3 x `max_duration_seconds` = 90 min at the 1800 s the
    tests below pass in).
    """
    import datetime as _dt
    import sqlite3
    row_id = q.enqueue(source="scheduled-task", kind="autonomy-task",
                       payload={"task_id": 510})
    aged = (_dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(minutes=age_min)).isoformat()
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE queue SET enqueued_at=? WHERE id=?", (aged, row_id))
        conn.commit()
    return row_id


def _hold_operator_pause(q, *, hours: float) -> str:
    """Pause as an operator, then wind the persisted row's `updated_at` back.

    Only the timestamp is wound back — `pause()` and its `wm_set` are the real
    writers, so the row is the production one in every other respect. That is
    what makes the assertion below a test of provenance: a fix that read the
    pause from process memory, or stamped its own start instant, could not report
    16.5 h for a pool that has existed for milliseconds.
    """
    import datetime as _dt
    import sqlite3
    from workers.pool import PAUSE_WM_KEY, PAUSE_WM_SOURCE
    WorkerPool(q, slots=1).pause(True)
    since = (_dt.datetime.now(_dt.timezone.utc)
             - _dt.timedelta(hours=hours)).isoformat()
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE watermarks SET updated_at=? "
                     "WHERE source=? AND key=?", (since, PAUSE_WM_SOURCE, PAUSE_WM_KEY))
        conn.commit()
    return since


async def _starving_alerts(q, monkeypatch, tmp_path):
    """One real scheduler-loop pass over `q`; returns the alerts it posted.

    The pass is the loop's own order — `fleet_watchdog.tick` (the alarms, seated on
    the loop itself by #1682) and then `enqueue_if_due` (the dispatch half) — because
    the starving clause is not in the dispatch call any more: a test that ran only
    `enqueue_if_due` would see no alert and would be measuring the split, not the
    alarm.

    The task dir is pointed at an EMPTY directory, so the alert under test can
    only be the starving clause: with no task files neither `overdue` nor the
    `next_run` scan has anything to name. That isolation is load-bearing here —
    the alternative is a tick that scans and re-arms the LIVE board.
    """
    from app import autonomy
    import workers.fleet_watchdog as fw
    import workers.sources.scheduled_task as st
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "tasks")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    (tmp_path / "tasks").mkdir(exist_ok=True)
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(fw, "_state", {**fw._state, "unparseable_scan_at": None,
                                       "stall_streak": fw._STALL_ALARM_TICKS,
                                       "stall_alerted_at": None,
                                       "nextrun_streak": 0,
                                       "nextrun_alerted_at": None})
    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(fw, "_alert", _capture)
    await fw.tick(q)
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    return alerts


async def test_a_pool_built_fresh_on_a_paused_database_alerts_by_the_pause(
        q, monkeypatch, tmp_path):
    """Clause 2: the restart analogue still produces the pause-named alert.

    The pause is taken by one pool and the alert is produced after a SECOND
    `WorkerPool` is constructed over the same database — the way a backend
    restart leaves it — with the first pool dropped. `_load_operator_pause` is
    the only thing that could carry the hold across that gap, and the alert must
    agree with it rather than with whatever the new process happened to remember.
    The control at the end is what makes the first assertion non-vacuous: the
    same fresh pool, resumed, must go back to the age-only sentence.
    """
    _age_a_queued_row(q, age_min=990)
    since = _hold_operator_pause(q, hours=16.5)

    fresh = WorkerPool(q, slots=1)
    assert fresh.paused_by == ["operator"], "the restart analogue lost the hold"
    assert fresh.paused_since == since, (
        f"status() does not report the persisted pause instant: {fresh.status()}")

    alerts = await _starving_alerts(q, monkeypatch, tmp_path)
    assert len(alerts) == 1, alerts
    msg = alerts[0]
    assert "operator" in msg, f"the alert does not name the pause: {msg}"
    assert "16.5 h" in msg, f"the alert does not state the held duration: {msg}"
    assert since in msg, f"the alert does not quote the pause instant: {msg}"
    assert "oldest claimable queue item is 990 min old" in msg, msg

    fresh.pause(False)
    alerts = await _starving_alerts(q, monkeypatch, tmp_path)
    assert alerts == ["autonomy scheduler may be stalled: "
                      "oldest claimable queue item is 990 min old"], (
        f"after the resume the alert must fall back to today's text: {alerts}")


#: ── #1554: the run record carries its scratchpad tally ───────────────────────
#:
#: Step 1 of #1554 asks whether scratchpad write-rate predicts an outcome. That is
#: only a query if the number is on the run row: `meta_json.scratchpad` joined to
#: `runs.status` over `duration_seconds` is the whole experiment, and no new
#: instrument is needed. A run that wrote nothing records zeros rather than an
#: absent key, because a missing key is a question a later reader has to guess at —
#: and "the scratchpad never caught on" and "nobody recorded" look identical in a
#: count of nulls.
#:
#: The source below writes the notes through the real `app.scratchpad.append` and
#: registers its session through the real `app.sessions_io.note_run_session`, so the
#: test crosses the seam the feature actually uses: the pool reads totals back from
#: disk at record time, which is the only thing that can be true when the append
#: happens in another process.


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    import app.scratchpad as sp
    root = tmp_path / "data"
    monkeypatch.setattr(sp, "DATA_ROOT", root)
    return root


async def test_a_finished_run_records_its_scratchpad_writes_beside_duration(
        q, monkeypatch, scratch_root):
    """Clause 5. The source writes three notes totalling a known byte count for the
    session it registers, and the run row has to say exactly that, next to the
    duration the rate is divided by."""
    import app.scratchpad as sp
    from app.sessions_io import note_run_session

    sid = "worker:s:aaa111"

    async def execute(item):
        note_run_session(sid)                    # what the worker driver does
        for text in ("ruled out the 4k window", "hypothesis: kv gate", "next: 8k"):
            sp.append(sid, text)
        return {"summary": "did the thing"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    meta = json.loads(runs[0]["meta_json"])
    assert meta["session_ids"] == [sid], meta
    sp_meta = meta["scratchpad"]
    assert sp_meta["writes"] == 3, sp_meta
    assert sp_meta["bytes"] == len(
        "ruled out the 4k window") + len("hypothesis: kv gate") + len("next: 8k"), sp_meta
    assert sp_meta["sessions"] == 1, sp_meta
    assert runs[0]["duration_seconds"] is not None, (
        "the tally is meant to be divided by this; a run row without it cannot "
        "answer writes-per-active-hour")


async def test_a_run_that_never_used_the_scratchpad_records_zeros(
        q, monkeypatch, scratch_root):
    """The control arm of the same query. Two thirds of the fleet will never touch
    the tool, and if their rows simply omit the key then "no correlation" and "no
    data" are the same shape — the ambiguity that made the earlier autonomy
    telemetry unreadable."""
    async def execute(item):
        return {"summary": "no notes taken"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    meta = json.loads(runs[0]["meta_json"])
    assert meta["scratchpad"] == {"writes": 0, "bytes": 0, "sessions": 0}, meta


async def test_a_timed_out_run_still_records_how_much_it_had_externalised(
        q, monkeypatch, scratch_root):
    """The population the item is about. A run killed at its cap is the row whose
    write-rate means something — how much had it written down by the moment it died
    is the comparison — and this branch is where that row is written. The 1 s window
    is the pool's real timeout path, the same one #1050's tests drive."""
    import app.scratchpad as sp
    from app.sessions_io import note_run_session

    sid = "worker:s:tim101"

    async def execute(item):
        note_run_session(sid)
        sp.append(sid, "halfway through the corpus, still on pass 2")
        await asyncio.sleep(5)                   # past the 1 s pool timeout
        return {"summary": "never reaches here"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute),
                            cfg={"max_duration_seconds": 1})
    meta = json.loads(runs[0]["meta_json"])
    assert meta["pool_timeout"] is True, meta
    assert meta["scratchpad"]["writes"] == 1, meta
    assert meta["scratchpad"]["bytes"] == len(
        "halfway through the corpus, still on pass 2"), meta


async def test_a_totals_failure_never_loses_the_run_record(q, monkeypatch, scratch_root):
    """The tally is a reader of someone else's file, on the way to writing this
    row. If it ever raises, the run's own record must still be written: a run that
    died is the run whose tally we most want and least want to lose, and losing the
    whole row to save a key is the wrong trade."""
    import app.scratchpad as sp

    def _boom(session_ids):
        raise RuntimeError("data root unreadable")

    monkeypatch.setattr(sp, "summarize", _boom, raising=False)
    # `summarize` is imported inside the helper, so patch the attribute it resolves.
    monkeypatch.setattr("app.scratchpad.summarize", _boom, raising=False)

    async def execute(item):
        return {"summary": "still recorded"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    assert runs[0]["status"] == "success", runs[0]
    assert json.loads(runs[0]["meta_json"])["scratchpad"] == {
        "writes": 0, "bytes": 0, "sessions": 0}


# ── dispatch watermark pair: attempt vs clean return (#1681) ───────────────
#
# `last_enqueue_check` is the interval clock and has to advance whatever the
# call does, or a source raising every tick is retried every tick. That is why
# it cannot also be the signal that dispatch works — which is what
# `workers/dispatch_watch.py` reads, and what the four tests below pin from
# either side of the swallow at `workers/pool.py`.


async def _one_pass_with(q, monkeypatch, *, raiser, cfg, repoll=False):
    """Run one scheduler pass over a single source named `s`.

    `raiser` is the point of these tests: `_one_pass` above can only return an
    outcome, so the raise path — the one the whole item is about — had no test
    and no harness. `repoll` puts REPOLL_ON_COMPLETE on the fake source, which
    is what makes `_repoll_on_complete` re-arm the attempt stamp to the epoch.
    """
    import workers.sources as sources

    async def enqueue_if_due(queue, src_cfg):
        if raiser:
            raise RuntimeError("ImportError inside the alarm code")
        return None

    monkeypatch.setattr(
        sources, "SOURCE_REGISTRY",
        {"s": SimpleNamespace(NAME="s", enqueue_if_due=enqueue_if_due,
                              REPOLL_ON_COMPLETE=repoll)},
        raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {"s": cfg},
                        raising=False)
    pool = WorkerPool(q, slots=0, poll_idle_seconds=0.01)
    return pool, await pool._scheduler_pass()


async def test_a_raising_enqueue_advances_the_attempt_stamp_only(q, monkeypatch):
    """Clause 1: the swallow stamps the interval clock and leaves the clean
    stamp alone, which is the only way the two can disagree — and a source
    that raises on every tick must read OLDER, never fresher, than one merely
    between intervals."""
    cfg = {"enabled": True, "interval_seconds": 60}
    pool, _ = await _one_pass_with(q, monkeypatch, raiser=True, cfg=cfg)
    assert pool._dispatch_watch is None, "the tick belongs to the loop, not the pass"

    await pool._watch_dispatch()
    row = pool._dispatch_watch.rows()["s"]
    # `pending`, not `unmeasured`: the attempt stamp proves the pool is ASKING and
    # the missing clean stamp proves the answer is not arriving. Before this pair
    # existed there was no state to distinguish that from a healthy source —
    # `unmeasured` is a source with neither stamp, which is what a fresh install
    # reports and what silence about a dead dispatcher used to look like.
    assert row["state"] == dispatch_watch.STATE_PENDING, row
    assert row["stalled"] is False, row
    assert q.wm_get("s", "last_enqueue_check") is not None, (
        "the interval clock stopped, so a raising source is retried every tick")
    assert q.wm_get("s", dispatch_watch.OK_WM_KEY) is None, (
        "a raise was recorded as a clean return")

    age = _age_of_watermark(q, "s")
    assert age < 5, f"attempt stamp should be ~now, is {age:.0f}s old"


async def test_a_clean_return_advances_the_success_stamp_in_all_three_forms(
        q, monkeypatch):
    """Clause 2: None, an enqueue result and DECLINED are three different
    returns and all three prove the call ran, so all three stamp. DECLINED
    matters most: it is the one outcome whose ATTEMPT stamp is back-dated, so
    it is the case where the two stamps deliberately differ."""
    import workers.sources as sources
    from datetime import datetime

    for label, outcome in (("none", None), ("declined", sources.DECLINED)):
        q.wm_set("s", dispatch_watch.OK_WM_KEY, "2020-01-01T00:00:00+00:00")
        # Both stamps back to an old day first: the first pass through the loop
        # stamps the attempt clock at `now`, and the interval arithmetic would
        # otherwise skip the second pass as not-yet-due and leave the success
        # stamp at 2020 — which is the right behaviour for a real source and the
        # wrong result for this assertion.
        q.wm_set("s", "last_enqueue_check", "2020-01-01T00:00:00+00:00")

        async def enqueue_if_due(queue, src_cfg):
            return outcome

        monkeypatch.setattr(
            sources, "SOURCE_REGISTRY",
            {"s": SimpleNamespace(NAME="s", enqueue_if_due=enqueue_if_due)},
            raising=False)
        monkeypatch.setattr(
            sources, "get_sources_config",
            lambda: {"s": {"enabled": True, "interval_seconds": 900,
                           "retry_seconds": 60}}, raising=False)
        pool = WorkerPool(q, slots=0, poll_idle_seconds=0.01)
        await pool._scheduler_pass()

        ok = datetime.fromisoformat(q.wm_get("s", dispatch_watch.OK_WM_KEY))
        age = (datetime.now(timezone.utc) - ok).total_seconds()
        assert age < 5, f"{label}: success stamp is {age:.0f}s old, should be now"
        if label == "declined":
            assert _age_of_watermark(q, "s") > 800, (
                "the DECLINED back-dating that clause 1 depends on has moved")
            assert age < _age_of_watermark(q, "s"), (
                "the clean stamp must not be back-dated too; it is the only "
                "stamp that says the call worked")


async def test_the_watch_runs_from_the_loop_seat_and_names_a_stalled_source(
        q, monkeypatch, caplog):
    """Clause 3: the reader sits beside _maybe_sweep_poisoned/_probe_services,
    outside the try that swallows a raising pass, and calls a source stalled
    once its clean stamp passes 3x its own interval."""
    src = inspect.getsource(WorkerPool._scheduler_loop)
    watch_at = src.index("await self._watch_dispatch()")
    pass_at = src.index("await self._scheduler_pass()")
    try_at = src.index("try:")
    assert watch_at < pass_at, "the reader runs after the pass it measures"
    assert watch_at < try_at or watch_at > src.index("except Exception"), \
        "the reader is inside the try that can skip the pass"
    loop_src = inspect.getsource(WorkerPool._watch_dispatch)
    assert "except Exception" in loop_src, "the seat's never-raises contract"

    cfg = {"enabled": True, "interval_seconds": 60}
    pool, _ = await _one_pass_with(q, monkeypatch, raiser=True, cfg=cfg)
    from datetime import datetime, timedelta
    old = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
    q.wm_set("s", dispatch_watch.OK_WM_KEY, old)

    pool._dispatch_watch = dispatch_watch.DispatchWatch(
        queue=q, sources=lambda: {"s": cfg}, announce=None,
        repoll=set())
    row = pool._dispatch_watch.rows()["s"]
    assert row["stalled"] is True, row
    assert row["state"] == dispatch_watch.STATE_STALLED, row
    assert row["threshold_seconds"] == 180.0, row
    assert row["age_seconds"] > 180, row

    # Fresh inside the window: the same source, one tick younger in effect.
    q.wm_set("s", dispatch_watch.OK_WM_KEY,
             datetime.now(timezone.utc).isoformat())
    row = pool._dispatch_watch.rows()["s"]
    assert row["stalled"] is False and row["state"] == dispatch_watch.STATE_OK, row


async def test_the_watch_is_silent_on_a_re_armed_or_disabled_source(q, monkeypatch):
    """Clause 4: two states that make the ATTEMPT stamp meaningless, and one of
    them is a trap the fix would have walked straight into.
    _repoll_on_complete back-dates `last_enqueue_check` to the epoch for a
    source with REPOLL_ON_COMPLETE, so a reader watching that stamp would call
    autocode stalled for the whole of its next interval after every run. A
    source disabled in config is skipped before either stamp is written, so its
    stamps go stale forever."""
    from datetime import datetime, timedelta
    cfg = {"enabled": True, "interval_seconds": 900}

    pool, _ = await _one_pass_with(q, monkeypatch, raiser=False, cfg=cfg,
                                   repoll=True)
    q.wm_delete("s", "last_enqueue_check")
    q.wm_set("s", "last_enqueue_check",
             datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat())
    q.wm_delete("s", dispatch_watch.OK_WM_KEY)
    watch = dispatch_watch.DispatchWatch(
        queue=q, sources=lambda: {"s": cfg}, announce=None, repoll={"s"})
    row = watch.rows()["s"]
    assert row["repoll_on_complete"] is True, row
    assert row["stalled"] is False, row
    assert row["state"] == dispatch_watch.STATE_UNMEASURED, row
    assert "re-arm" in row["detail"], row

    # The same two stamps, without the declaration: `never_succeeded`, and
    # stalled. So the exemption above is doing real work on the state the epoch
    # stamp would otherwise produce, and is not a blanket silence — a repoll
    # source with a clean stamp that has gone stale IS still called stalled, the
    # first branch of `verdict` running before this one.
    # The same two stamps on a source that does NOT re-arm: the epoch attempt
    # stamp reads as "raised for 57 years", so the exemption above is doing real
    # work on exactly the state that stamp produces, and is not a blanket
    # silence — `plain` is the false positive clause 4 exists to prevent.
    plain = dispatch_watch.DispatchWatch(
        queue=q, sources=lambda: {"s": cfg}, announce=None, repoll=set())
    assert plain.rows()["s"]["stalled"] is True, plain.rows()["s"]
    assert (plain.rows()["s"]["state"]
            == dispatch_watch.STATE_NEVER_SUCCEEDED), plain.rows()["s"]

    # The state the re-arm actually leaves a WORKING repoll source in: attempt
    # stamp at the epoch, clean stamp seconds old, because the re-arm follows a
    # run and a run means the enqueue returned cleanly.
    q.wm_set("s", dispatch_watch.OK_WM_KEY, datetime.now(timezone.utc).isoformat())
    row = watch.rows()["s"]
    assert row["stalled"] is False and row["state"] == dispatch_watch.STATE_OK, row

    # And a re-armed source whose clean stamp DOES go stale is still caught: the
    # exemption covers the meaningless stamp, never the meaningful one.
    q.wm_set("s", dispatch_watch.OK_WM_KEY,
             (datetime.now(timezone.utc) - timedelta(seconds=3000)).isoformat())
    assert watch.rows()["s"]["stalled"] is True, watch.rows()["s"]
    q.wm_set("s", dispatch_watch.OK_WM_KEY, datetime.now(timezone.utc).isoformat())

    # Disabled, with a stamp two days old: not judged at all.
    q.wm_set("s", dispatch_watch.OK_WM_KEY,
             (datetime.now(timezone.utc) - timedelta(days=2)).isoformat())
    off = dispatch_watch.DispatchWatch(
        queue=q, sources=lambda: {"s": {**cfg, "enabled": False}},
        announce=None, repoll=set())
    row = off.rows()["s"]
    assert (row["stalled"] is False
            and row["state"] == dispatch_watch.STATE_DISABLED), row

    # And a tick over a stall announces once, then stays quiet, then announces
    # the recovery: a steady fault must not become a recurring alarm. On the
    # 60-second source, which is `scheduled-task`'s real shape: 200 s old is
    # past its 180 s threshold, where the same stamp is inside the 900 s cfg
    # used by every block above.
    q.wm_set("s", dispatch_watch.OK_WM_KEY,
             (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat())
    said = []
    live = dispatch_watch.DispatchWatch(
        queue=q, sources=lambda: {"s": {"enabled": True, "interval_seconds": 60}},
        announce=lambda *a: said.append(a), repoll=set())
    assert len(live.tick()) == 1 and live.tick() == [] and live.tick() == []
    q.wm_set("s", dispatch_watch.OK_WM_KEY, datetime.now(timezone.utc).isoformat())
    assert len(live.tick()) == 1
    assert [s[2] for s in said] == ["warn", "info"], said


# ---------------------------------------------------------------------------
# #1606 — the summary is cut on a word, and says so
# ---------------------------------------------------------------------------
#
# `runs.summary` is 500 characters wide and every reader of a run — the
# dashboard cell, the fleet-watchdog alert, the `grep` over a day of rows — reads
# that string alone. The cap was a byte slice, so 77 of board-steward's 154 rows
# sit at exactly 500 characters and 63 of them stop inside a word — one ends
# "…remain under guardian observati". The reader cannot tell a truncated record
# from a complete one, and neither could the source. `"word "` is 5 characters,
# so a blind `[:497]` lands on the 'r' of a word, which is what the assertions
# below catch.


def test_a_summary_that_fits_is_stored_exactly_as_the_source_wrote_it():
    from workers.queue import SUMMARY_MARKER, SUMMARY_MAX_CHARS

    fits = "wrote the note; 3 claims checked"
    assert normalize_result(_item(), {"summary": fits})["summary"] == fits

    # Exactly 500 characters still "fits": the boundary is `<=`, not `<`. A cap
    # that fires at `>=` would append a marker to a record with nothing missing,
    # and the marker would stop meaning anything.
    exact = "word " * (SUMMARY_MAX_CHARS // 5)
    assert len(exact) == SUMMARY_MAX_CHARS
    stored = normalize_result(_item(), {"summary": exact})["summary"]
    assert stored == exact, "a full-width summary must not be told it overflowed"
    assert not stored.endswith(SUMMARY_MARKER)


def test_a_summary_over_the_cap_is_cut_at_a_word_boundary_and_marked():
    from workers.queue import SUMMARY_MARKER, SUMMARY_MAX_CHARS

    text = "word " * 180                       # 900 characters, all whole words
    stored = normalize_result(_item(), {"summary": text})["summary"]

    assert stored.endswith(SUMMARY_MARKER), "a truncated record must say so"
    kept = stored[:-len(SUMMARY_MARKER)]
    assert len(stored) <= SUMMARY_MAX_CHARS, "the marker counts against the cap"
    assert text.startswith(kept)
    assert kept and text[len(kept)].isspace(), \
        f"cut landed inside a word: ...{kept[-8:]!r}"
    # 5-character words, so a prefix ending on a whole word is 5n-1 characters:
    # an arithmetic statement of "word boundary" that a byte slice cannot satisfy
    # (497 and 500 are both ≡ 2 mod 5).
    assert len(kept) % 5 == 4
    # The two byte slices this replaces, named rather than implied: neither the
    # raw cap nor the cap minus the marker is what a word-boundary cut produces.
    assert kept != text[:SUMMARY_MAX_CHARS]
    assert kept != text[:SUMMARY_MAX_CHARS - len(SUMMARY_MARKER)]


def test_a_900_character_wordy_summary_records_complete_words_plus_the_marker():
    """Clause 2 of #1606, at the width the clause names.

    900 characters of space-separated words must come back as at most 500
    characters whose last real word is complete, marker included — the record a
    person can quote without inventing the rest of a word.
    """
    from workers.queue import SUMMARY_MARKER, SUMMARY_MAX_CHARS

    stored = normalize_result(_item(), {"summary": "word " * 180})["summary"]

    assert len(stored) <= SUMMARY_MAX_CHARS
    assert stored.endswith(SUMMARY_MARKER)
    kept = stored[:-len(SUMMARY_MARKER)]
    assert kept.endswith("word"), f"final token is not a complete word: {kept[-10:]!r}"
    # Truncation is a length rule, not an excuse to drop most of the record: a
    # cut that kept 60 characters would satisfy "≤ 500" just as well.
    assert len(kept) > SUMMARY_MAX_CHARS - 20, "threw away more than the marker needed"


# ---------------------------------------------------------------------------
# #1684 — the fleet's health gate probes the endpoint config names
#
# `WorkerPool._scheduler_loop` → `scheduled_task.enqueue_if_due` is the one pass
# that decides whether ANY autonomy task may be enqueued, and until now the URL it
# probed on that pass was a literal baked into the source file. Moving the
# primary's port under `models:` therefore stopped dispatch on an endpoint the
# engine never served, and nothing on this path could see the disagreement. The
# tick below is the pool's real call into the source, so what it records is what
# the fleet's verdict is actually made of.
# ---------------------------------------------------------------------------


async def test_one_tick_probes_the_endpoint_config_says_the_primary_lives_on(
        q, monkeypatch, tmp_path):
    """Clause 3: the recorded probe carries the config-derived URL, not the literal.

    The task dir is redirected to an EMPTY directory, as `_starving_alerts` does,
    so the tick scans nothing live; the probe is the only thing under test and it
    refuses. Pre-fix the recorded call is `()` — `_vllm_healthy` was invoked with
    no URL at all and fell through to the module constant — so this fails on the
    first assertion, not on a string comparison someone could retarget.
    """
    from app import autonomy, config as cfg
    import workers.sources.scheduled_task as st

    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "tasks")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    (tmp_path / "tasks").mkdir(exist_ok=True)
    # The outage accounting is process-global; snapshot it through monkeypatch so
    # this tick's bookkeeping cannot make a later test's first tick a "recovery".
    monkeypatch.setitem(st._state, "vllm_down_logged", False)
    monkeypatch.setitem(st._state, "vllm_down_since", None)
    monkeypatch.setitem(st._state, "vllm_down_alerted", False)
    monkeypatch.setitem(cfg.MODEL_CONFIGS["primary"], "base_url",
                        "http://127.0.0.1:9999")

    probes: list = []
    monkeypatch.setattr(st, "_vllm_healthy",
                        lambda *a, **k: probes.append(a) or False)

    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})

    assert len(probes) == 1, f"the fleet gate probed {len(probes)} times"
    assert probes[0] == (4.0, "http://127.0.0.1:9999/health"), (
        f"the fleet gate probed {probes[0]!r}, not the config-derived endpoint")
    assert probes[0][1] != st._VLLM_HEALTH_URL, (
        "the URL that gates every dispatch is still the module's literal")
    assert q.list_items(source=st.NAME) == [], (
        "dispatch enqueued while its own health probe was refusing")


# ---------------------------------------------------------------------------
# A death that never reached the task file (#2037)
#
# `scheduled_task.execute()` opens with three imports outside any try, and
# `run_task` runs ~270 lines before its first failure-recording try. A death in
# either stretch passes `WorkerPool._run_item`'s `except Exception` arm and
# nothing else: no `_record_failure`, so `failure_count` never moved, so the
# scheduler kept offering the task, so the row re-claimed every ~minute until the
# poison sweep took it. Task #74 measured 41 failed runs behind 14 queue rows in
# 27 minutes on 2026-09-29, every one of them with `meta.failure_kind` NULL —
# that NULL is the signature of a death the task file never saw, and it is what
# the nodes below replace.
# ---------------------------------------------------------------------------


@pytest.fixture
def task_board(tmp_path, monkeypatch):
    """One real task file and runs tree, both redirected off the live vault.

    The charge this section exercises lives in `app.autonomy` and resolves every
    path it writes through two module globals, so redirecting them is what keeps
    a death charged against task #74 here from landing in `~/obsidian/autonomy`
    or the live data root. `discord_notify.discord_alert` is patched at its
    source because the disable arm imports it inside its own try block.
    """
    import yaml
    from app import autonomy
    import app.discord_notify as discord_notify

    tasks = tmp_path / "autonomy"
    tasks.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tasks)
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "autonomy-runs")

    async def _no_alert(*_a, **_k):
        return None

    monkeypatch.setattr(discord_notify, "discord_alert", _no_alert)

    def write(task_id: int, **over) -> None:
        fm = {"id": task_id, "name": f"task{task_id}", "type": "autonomy",
              "status": "up_next", "frequency": "hourly", "priority": "low",
              "skill_name": "some-skill", "max_retries": 3, "failure_count": 0}
        fm.update(over)
        (tasks / f"{task_id}-task{task_id}.md").write_text(
            f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# task\n\n"
            "## Activity Log\n", encoding="utf-8")

    def read(task_id: int) -> dict:
        return autonomy._parse_task_file(autonomy._find_task_file(task_id))

    return SimpleNamespace(write=write, read=read, tasks=tasks, autonomy=autonomy)


def _death_source(exc, *, name: str = "scheduled-task", charge: bool = True,
                  verdict_recorded: bool = False):
    """A source that dies the way `scheduled_task.execute` dies on import.

    `charge` is the source's opt-in (`CHARGE_TASK_ON_DEATH`).
    `verdict_recorded` models the one death that must NOT be charged: `run_task`
    returned, wrote this attempt's verdict through the real `_record_failure`, and
    the adapter then died in its own post-processing. The verdict is recorded
    through production code rather than simulated, so the budget really has moved
    once by the time the death is handled — and a second unit from the pool is a
    number the caller can see.
    """
    async def execute(item):
        if verdict_recorded:
            from datetime import datetime, timezone
            from app import autonomy
            from workers.pool import mark_task_verdict

            path = autonomy._find_task_file(item.payload["task_id"])
            task = autonomy._parse_task_file(path)
            now = datetime.now(timezone.utc)
            await autonomy._record_failure(
                task, task["id"], "run_74_verdict_20261001", now.isoformat(), now,
                summary="RuntimeError: engine refused the prompt",
                body="## Prompt\n\n(test)\n", kind="task")
            mark_task_verdict()
        raise exc

    src = SimpleNamespace(NAME=name, execute=execute)
    if charge:
        src.CHARGE_TASK_ON_DEATH = True
    return src


def _meta_of(runs_row: dict) -> dict:
    return json.loads(runs_row["meta_json"] or "{}")


async def _drain_deaths(q, monkeypatch, src, *, count: int = 1, task_id: int = 74,
                        source_name: str = "scheduled-task", max_attempts: int = 1,
                        dedup_key: str | None = None) -> list[dict]:
    """`count` deaths, each its own queue row, through one real pool.

    One row, one attempt, by `max_attempts=1`: the queue re-claims a failed row up
    to its own cap, and every claim is a death that owes exactly one budget unit.
    Leaving that cap at its default of 3 would fold a row's retries into a count
    the assertions could not attribute to anything. Rows are enqueued and awaited
    one at a time, so `count` is exact — and a row poisoned at the cap NULLs its
    dedup key and hands it to the next row, which is the churn under test, not a
    race to avoid.
    """
    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {source_name: src}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {source_name: {"max_duration_seconds": 30}},
                        raising=False)
    pool = WorkerPool(q, slots=1, max_attempts=max_attempts,
                      poll_idle_seconds=0.01)
    await pool.start()
    try:
        for i in range(count):
            q.enqueue(source_name, "run", payload={"task_id": task_id},
                      dedup_key=dedup_key)
            for _ in range(400):
                if len(q.list_runs(limit=50)) >= i + 1:
                    break
                await asyncio.sleep(0.02)
    finally:
        await pool.stop()
    return q.list_runs(limit=50)


async def test_a_death_with_no_verdict_charges_the_task_retry_budget(
        q, monkeypatch, task_board):
    """Clause 1: one death the task file never saw costs that file exactly 1.

    Driven through a real `WorkerPool` and a real queue rather than by calling
    the charge directly, because the seam IS the pool's `except Exception` arm —
    the one place that ever saw task #74's `ModuleNotFoundError`.
    """
    task_board.write(74)
    runs = await _drain_deaths(
        q, monkeypatch,
        _death_source(ModuleNotFoundError("No module named 'app.discord_notify'")),
        dedup_key="scheduled-task:74")

    assert runs[0]["status"] == "failed" and runs[0]["task_id"] == "74"
    fm = task_board.read(74)
    assert fm["failure_count"] == 1, (
        f"failure_count is {fm['failure_count']!r}: the death spent no budget, so "
        "the scheduler offers the task again and the loop outlives the sweep")
    assert fm["status"] == "up_next", "one death of three already stopped the task"
    assert fm["next_run"], (
        "the charge wrote no cooldown, so `hourly` is due again on the next tick")


async def test_charged_deaths_reach_max_retries_and_the_task_reads_failed(
        q, monkeypatch, task_board):
    """Clause 2, first half: three charged deaths retire a `max_retries: 3` task.

    Three separate queue rows, because the queue bound is per ROW: `mark_failed`
    NULLs the dedup key when it poisons, and the source re-enqueues under
    `scheduled-task:74` every poll, which is how 14 rows carried 41 runs.
    """
    task_board.write(74, max_retries=3)
    await _drain_deaths(q, monkeypatch, _death_source(RuntimeError("boom")), count=3)

    fm = task_board.read(74)
    assert fm["failure_count"] == 3, (
        f"three deaths moved failure_count to {fm['failure_count']!r}")
    assert fm["status"] == "failed", (
        "the retry budget was never reached, so `get_due_tasks` still offers it")


async def test_the_failed_row_of_a_charged_death_names_its_failure_kind(
        q, monkeypatch, task_board):
    """Clause 4: the exception path writes no NULL-kind row for a charged death.

    `meta.failure_kind` is the key `app/autonomy.py` already writes on the
    verdict path; 42 failed `scheduled-task` rows carried none of it, which is
    how a week of a task dying every minute stayed invisible to every reader
    that groups runs by kind.
    """
    task_board.write(74)
    runs = await _drain_deaths(q, monkeypatch, _death_source(RuntimeError("boom")))

    meta = _meta_of(runs[0])
    assert meta.get("failure_kind") == "task", (
        f"the row carries {meta.get('failure_kind')!r}; a NULL is what made the "
        "09-29 loop uncountable")
    assert meta.get("task_budget_charged") is True, (
        "the row says a death happened but not that anything was charged for it")


async def test_three_pool_deaths_put_one_fast_failure_line_in_the_daily_note(
        q, monkeypatch, task_board, tmp_path):
    """The alert seam reached FROM THE POOL, so the chain is not wired by prose.

    `tests/test_autonomy_failure_alert.py::test_four_charged_pool_deaths_append_exactly_one_fast_failure_line`
    pins the seam itself — one line per streak, that writer's format — by calling
    the charge directly. This node pins the link the clause is actually about:
    three real deaths through this pool append exactly one line naming #74 to
    today's note. Delete the pool's charge and this goes red while the direct-call
    node stays green; that pairing is what makes "the silence is the defect"
    (#1085) checkable end to end rather than by two halves that never meet.

    `LLOYD_DAILY_NOTE_DIR` is pointed at this test's own directory: the conftest
    default is a dir the whole session shares, and a count of alert lines needs a
    note only this test wrote in.
    """
    from app import autonomy

    monkeypatch.setenv("LLOYD_DAILY_NOTE_DIR", str(tmp_path / "notes"))
    task_board.write(74, max_retries=9)
    await _drain_deaths(q, monkeypatch,
                        _death_source(ModuleNotFoundError("No module named 'x'")),
                        count=3)

    notes = sorted((tmp_path / "notes").glob("*.md"))
    assert len(notes) == 1, f"notes written: {[str(p.name) for p in notes]}"
    lines = [ln for ln in notes[0].read_text().splitlines() if "Autonomy #" in ln]
    assert len(lines) == 1, (
        f"the note carries {len(lines)} alert lines for a 3-death streak: {lines}")
    assert "#74" in lines[0], lines[0]
    assert autonomy._FAST_FAILURE_ALERT_STREAK == 3, (
        "the threshold this node counts one line at has moved; the assertion above "
        "is pinned to it, not to a number copied here")


@pytest.mark.parametrize("charge,verdict,expected",
                         [(False, False, 0), (True, True, 1)],
                         ids=["source-did-not-opt-in", "verdict-already-written"])
async def test_a_death_the_pool_must_not_charge_is_still_classified(
        q, monkeypatch, task_board, charge, verdict, expected):
    """The two deaths that must not spend a budget, and must still be countable.

    `source-did-not-opt-in` is the case `bench-mine` is in: its payload carries a
    `task_id` naming an autonomy task it is MINING, not a task this run is, so
    charging from the payload alone would retire someone else's schedule — the
    file has to read 0. `verdict-already-written` is a source that died AFTER
    `run_task` had already recorded this attempt's verdict through the real
    `_record_failure`, so the file has to read exactly the 1 the verdict booked:
    a second unit from the pool would retire a task at half its `max_retries`.

    Both cases still classify, which is the half that stops the change trading one
    silence for another: `meta.failure_kind` is written for every exception from
    every source, charged or not.
    """
    task_board.write(74, max_retries=9)
    runs = await _drain_deaths(
        q, monkeypatch,
        _death_source(ModuleNotFoundError("No module named 'x'"),
                      charge=charge, verdict_recorded=verdict))

    got = task_board.read(74)["failure_count"]
    assert got == expected, (
        f"failure_count is {got}, expected {expected} — a death that is not this "
        "pool's to charge still moved the task's budget")
    assert _meta_of(runs[0]).get("failure_kind") == "task", (
        "not charging is not the same as not classifying: the row still has to be "
        "countable, or the same silence is back under a different name")
    assert "task_budget_charged" not in _meta_of(runs[0]), (
        "the row claims a charge the task file never saw")


# ── #2087: the run row carries the step count of the turns it ran ─────────────
#
#: `runs.meta_json.num_turns` existed on 6 of the 15 sources and on NONE of the
#: four highest-volume ones — autotriage 1,702 runs, autocode 688, owed-check 591,
#: board-steward 366, zero between them — because each of those four reads the
#: count off the terminal event and then only writes it when it happens to build a
#: `meta` dict at all. The fix collects rather than reports: `workers/pool.py` binds
#: an empty bucket around the claimed job, every turn path in
#: `workers/sources/_common.py` appends what it read, and the pool reads it back at
#: all three `record_run` sites.
#:
#: So every test below drives the REAL `run_prompt_in_session`, and only its
#: transport is swapped onto a stubbed SSE `done` event — the same shape
#: `tests/conftest.py::worker_turn_post` uses and for the same reason: the number
#: must be produced by the parse the four sources actually read it from, not handed
#: to the pool by the test. A test that called `note_run_turns` itself would still
#: pass if the note were deleted from `_common.py`, which is precisely the drop this
#: item is about.


async def test_a_turn_that_reported_its_steps_puts_them_on_the_run_row(q, monkeypatch, worker_turn_post):
    """Clause 1, in the shape of the sources that were missing it.

    The handler returns no `meta` key at all — autotriage.py and autocode.py both
    return `{"summary": ..., "artifact_path": ...}` and nothing else — so the only
    way this number can reach the row is by being collected. The count itself comes
    from the real SSE parse inside the real `run_prompt_in_session`.
    """
    from workers.sources._common import run_prompt_in_session

    worker_turn_post.report(7)

    async def execute(item):
        run = await run_prompt_in_session("classify the queue",
                                          title="triage", source="s")
        assert run["num_turns"] == 7, (
            "the stub turn did not report 7 steps through the real parser, so "
            "this is no longer measuring the drop this item is about")
        return {"summary": "verdicts written"}      # no `meta` key, like autotriage

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    assert _meta_of(runs[0]).get("num_turns") == 7, (
        f"the row does not carry the 7 steps its one turn reported: {_meta_of(runs[0])}")
    # And the turn was a real loopback POST rather than a call stubbed at both
    # ends: its body carried the instruction, so the count on the row is
    # downstream of the same request a live autotriage tick sends.
    assert worker_turn_post[0]["text"] == "classify the queue", (
        "the turn was asked something else, so this step count did not arrive on "
        f"the payload a real run sends: {worker_turn_post[0].get('text')!r}")


async def test_a_two_turn_job_records_the_sum_and_not_the_last_turn(q, monkeypatch, worker_turn_post):
    """Clause 2: 3 then 5 is 8, because the field is the job's steps.

    #583 wants to predict how long a job will take, and a job whose turns are
    3 and 5 took eight steps. Reading the last turn only would have been the
    lazier implementation and would have passed clause 1, so this is the test that
    tells the two apart — owed_check.py runs a turn per owed item, and board_steward
    likewise, so multi-turn jobs are ordinary here, not a corner.
    """
    from workers.sources._common import run_prompt_in_session

    worker_turn_post.report(3, 5)

    async def execute(item):
        first = await run_prompt_in_session("first pass", title="p1", source="s")
        second = await run_prompt_in_session("second pass", title="p2", source="s")
        assert (first["num_turns"], second["num_turns"]) == (3, 5)
        return {"summary": "two passes"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    meta = _meta_of(runs[0])
    assert meta.get("num_turns") == 8, (
        f"a job whose turns reported 3 and 5 must record 8 steps, got {meta}")


async def test_a_run_killed_at_its_cap_keeps_the_steps_it_already_took(q, monkeypatch, worker_turn_post):
    """Clause 3, timeout arm: the count of the turns that FINISHED, on the same key.

    A run killed at `max_duration_seconds` is the population #583 and #1554 both
    want a denominator for, and it is the arm that has no handler return to read a
    number from. One real turn reports 6 steps, the job then sleeps past the 1 s cap
    the way `test_a_timed_out_run_still_records_how_much_it_had_externalised` drives
    the same path — so 6 is exactly how far the run got, and the turn still in
    flight when the cap fired contributes nothing rather than a zero.
    """
    from workers.sources._common import run_prompt_in_session

    worker_turn_post.report(6)

    async def execute(item):
        run = await run_prompt_in_session("long haul", title="slow", source="s")
        assert run["num_turns"] == 6
        await asyncio.sleep(5)                    # past the 1 s pool timeout
        return {"summary": "never reaches here"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute),
                            cfg={"max_duration_seconds": 1})
    meta = _meta_of(runs[0])
    assert meta.get("pool_timeout") is True, f"this is not the timeout arm: {meta}"
    assert meta.get("num_turns") == 6, (
        f"a run killed after one finished turn of 6 steps must record 6, got {meta}")


async def test_a_run_that_raised_keeps_the_steps_it_already_took(q, monkeypatch, worker_turn_post):
    """Clause 3, exception arm — the other death, and the one with no return value.

    Same 6 steps from one real finished turn, then the handler raises. This is the
    arm #2037 exists for: a death used to record the exception and nothing else, so
    how far the run got was not on the row at all.
    """
    from workers.sources._common import run_prompt_in_session

    worker_turn_post.report(6)

    async def execute(item):
        run = await run_prompt_in_session("doomed pass", title="die", source="s")
        assert run["num_turns"] == 6
        raise RuntimeError("engine refused the prompt")

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    meta = _meta_of(runs[0])
    assert meta.get("exception") == "RuntimeError", f"not the death arm: {meta}"
    assert meta.get("num_turns") == 6, (
        f"a run that died after one finished turn of 6 steps must record 6, got {meta}")


async def test_a_job_that_ran_no_turn_omits_the_count_rather_than_calling_it_zero(
        q, monkeypatch):
    """Clause 4: "not measured" must stay a different shape from "took no steps".

    A `skipped` run is most of what autotriage and board_steward produce on a quiet
    tick, and this is the query the item's own check runs:
    `SUM(CASE WHEN meta_json LIKE '%num_turns%' THEN 1 ELSE 0 END)`. Writing a
    `"num_turns": null` onto every one of those rows would make that count non-zero
    for the emptiest possible reason — it would count rows that measured nothing —
    which is the ambiguity the zero set was. So the key is absent here, and the
    test asserts absence rather than a particular null, because the difference is
    the whole content of this clause.
    """
    async def execute(item):
        return {"status": "skipped", "summary": "nothing owed"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    meta = _meta_of(runs[0])
    assert "num_turns" not in meta, (
        f"a job that ran no harness turn must not be stamped with a step count; "
        f"a null or a 0 here reads as a measurement: {meta}")
    assert runs[0]["status"] == "skipped", runs[0]


async def test_a_turn_that_reported_no_count_leaves_the_key_absent(q, monkeypatch, worker_turn_post):
    """Clause 4's other edge: the turn ran, and the harness never said how long.

    A `done` event with no `num_turns` is not a turn that took zero steps. This is
    the arm a wedged engine produces, and a `0` would be the one value a later
    reader could not tell apart from an idle job.
    """
    from workers.sources._common import run_prompt_in_session

    worker_turn_post.report(None)

    async def execute(item):
        run = await run_prompt_in_session("wedged", title="wedged", source="s")
        assert run["num_turns"] is None, "the stub turn reported a count after all"
        return {"summary": "turn ended without a count"}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    meta = _meta_of(runs[0])
    assert "num_turns" not in meta, (
        f"a turn that never reported a count must leave the key absent, not 0: {meta}")


async def test_a_source_that_reports_its_own_count_keeps_it(q, monkeypatch, worker_turn_post):
    """Clause 5: the collector fills gaps, it does not overrule a report.

    Six sources already write `num_turns` into the meta they return
    (`workers/sources/youtube_digest.py:901`, `bench_mine.py:920`,
    `session_distill.py:342`, `arch_review.py:1928`, `deep_research.py:639` and the
    autonomy path at `app/autonomy.py:4393`), and some report a subset of the job's
    turns — youtube_digest reports one turn's count, not a sum. Overwriting those
    with the collected aggregate would silently change the meaning of every row they
    already wrote — 661 of them in the committed witness, from its own
    `SUM(CASE WHEN meta_json LIKE '%num_turns%' THEN 1 ELSE 0 END)` — so the reported
    value wins. Here the turn really reports 4 and the handler says 11; the row has
    to read 11.
    """
    from workers.sources._common import run_prompt_in_session

    worker_turn_post.report(4)

    async def execute(item):
        run = await run_prompt_in_session("self-reported", title="own", source="s")
        assert run["num_turns"] == 4, (
            "the collected count would be 4, so this test needs the turn to "
            "actually report 4 before the handler's 11 can overrule it")
        return {"summary": "reported myself",
                "meta": {"num_turns": run["num_turns"] + 7}}

    q.enqueue("s", "k")
    runs = await _drain_one(q, monkeypatch, SimpleNamespace(NAME="s", execute=execute))
    meta = _meta_of(runs[0])
    assert meta.get("num_turns") == 11, (
        f"a source that reported 11 kept {meta.get('num_turns')!r} — the collected "
        "4 must fill a gap, not overwrite a report")


def test_the_committed_witness_bytes_reproduce_the_zero_set_this_item_is_about():
    """Clause 6: the report is re-derivable from committed bytes, read-only.

    `backlog/data/workers-runs-2087.db` is an extract of `runs` for
    `started_at >= '2026-09-03'` whose 13 columns are identical to the live store.
    The live store is retention-pruned, so without it the claim that autotriage,
    autocode, owed-check and board-steward carried `num_turns` on ZERO of their
    runs would be a sentence about a table nobody can open again — and this round's
    own fix starts making that query non-zero, so after landing it could never be
    re-checked at all.

    Read `mode=ro`, in place, for the same reason the companion witness test does:
    a test must not write the artifact it certifies. Two assertions, and the second
    one is the load-bearing half — a query whose `LIKE` pattern had silently broken
    would report zero for the four AND zero for everyone else, and read as a clean
    result. `scheduled-task` and `bench-mine` carrying counts in the same bytes is
    what proves the instrument distinguishes the two states.

    The exact per-source figures live in `workers-runs-2087.witness.md` and are not
    asserted here: the extract is frozen so those cannot drift, and pinning them
    again in Python would only mean a future re-extract trips a test that is not
    grading anything.
    """
    import sqlite3
    from pathlib import Path

    path = (Path.home() / "obsidian" / "backlog" / "data"
            / "workers-runs-2087.db")
    if not path.exists():
        pytest.skip(f"no vault witness at {path} on this machine")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = dict(conn.execute(
            "SELECT source, SUM(CASE WHEN meta_json LIKE '%num_turns%' "
            "THEN 1 ELSE 0 END) FROM runs WHERE started_at >= '2026-09-03' "
            "GROUP BY source").fetchall())
        totals = dict(conn.execute(
            "SELECT source, COUNT(*) FROM runs WHERE started_at >= '2026-09-03' "
            "GROUP BY source").fetchall())
    finally:
        conn.close()

    for source in ("autotriage", "autocode", "owed-check", "board-steward"):
        assert totals.get(source, 0) > 0, (
            f"{source} has no runs in the witness at all, so its zero below is "
            "an empty result dressed as a measurement")
        assert rows.get(source, 0) == 0, (
            f"{source} carries num_turns on {rows[source]} of {totals[source]} "
            "witnessed runs, so this is no longer the zero set the item filed")

    populated = {s: rows[s] for s in ("scheduled-task", "bench-mine")
                 if rows.get(s, 0) > 0}
    assert len(populated) == 2, (
        f"positive control failed: no `num_turns` in the bytes for scheduled-task "
        f"or bench-mine ({rows}), so the LIKE pattern is measuring nothing and the "
        "four zeros above prove nothing either")
