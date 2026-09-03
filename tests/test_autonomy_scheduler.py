"""Autonomy scheduler contract (autonomy.py + workers).

Pins the fixes from the 2026-09-03 fleet audit, which found ~73 GPU-hours a
week burned on failed runs. Each test names the failure mode it prevents.
"""
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autonomy  # noqa: E402


def _iso(**delta):
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(**delta)).isoformat()


@pytest.fixture
def aut(tmp_path, monkeypatch):
    """Isolated task dir + runs dir, with a resolvable skill file."""
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "autonomy")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    autonomy.AUTONOMY_DIR.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n")
    monkeypatch.setattr(autonomy, "_SKILL_FOR_TESTS", str(skill), raising=False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda: "sys", raising=False)
    return autonomy


def write_task(aut, task_id, **fm):
    base = {
        "id": task_id, "name": f"task{task_id}", "type": "autonomy",
        "status": "up_next", "frequency": "daily", "priority": "medium",
        "skill_name": aut._SKILL_FOR_TESTS, "timeout_seconds": 2,
        "max_retries": 3, "failure_count": 0,
    }
    base.update(fm)
    base = {k: v for k, v in base.items() if v is not None}
    path = aut.AUTONOMY_DIR / f"{task_id}-task{task_id}.md"
    path.write_text(f"---\n{yaml.dump(base)}---\n\nbody\n\n## Activity Log\n")
    return path


def read_task(aut, task_id):
    return aut._parse_task_file(aut._find_task_file(task_id))


def fake_run_query(events, delay=0.0):
    async def _rq(messages, options):
        for e in events:
            if delay:
                await asyncio.sleep(delay)
            yield e
    return _rq


RESULT = {"type": "result", "stop_reason": "stop",
          "usage": {"input_tokens": 10, "output_tokens": 5}, "num_turns": 1}
TEXT = {"type": "text_delta", "text": "did the thing"}
TOOLCALL = {"type": "tool_call", "id": "c1", "name": "Bash", "input": {}}


# ── Failure backoff (the retry storm) ────────────────────────────────────────

async def test_timeout_records_failure_and_stops_immediate_retry(aut, monkeypatch):
    """A timed-out task used to be due again on the next 60s tick, forever:
    12 consecutive 600s timeouts on #36 in one night."""
    write_task(aut, 1, timeout_seconds=1)
    monkeypatch.setattr("app.harness.run_query", fake_run_query([TEXT], delay=5))
    result = await aut.run_task(1)

    assert result["success"] is False
    assert result["failure_kind"] == "task"
    t = read_task(aut, 1)
    assert t["status"] == "up_next"
    assert int(t["failure_count"]) == 1
    assert t.get("last_attempt")          # attempt recorded
    assert not t.get("last_run")          # but NOT counted as a completion
    assert aut._in_failure_cooldown(t, dt.datetime.now(dt.timezone.utc))
    assert aut._is_task_due(t, [t]) is False
    runs = list((aut.AUTONOMY_RUNS_DIR / "1").glob("run_*.md"))
    assert len(runs) == 1 and "status: failed" in runs[0].read_text()


async def test_cooldown_grows_and_expires(aut):
    write_task(aut, 1, failure_count=1)
    t = read_task(aut, 1)
    assert aut._failure_cooldown_seconds(t) == 600
    t["failure_count"] = 3
    assert aut._failure_cooldown_seconds(t) == 2400
    t["failure_count"] = 99  # capped
    assert aut._failure_cooldown_seconds(t) == max(86400, 21600)

    # An old failure no longer holds the task back.
    write_task(aut, 2, failure_count=1, last_attempt=_iso(hours=3))
    t2 = read_task(aut, 2)
    assert aut._in_failure_cooldown(t2, dt.datetime.now(dt.timezone.utc)) is False


async def test_success_resets_failure_count_and_sets_both_timestamps(aut, monkeypatch):
    write_task(aut, 1, failure_count=2, last_attempt=_iso(hours=1))
    monkeypatch.setattr("app.harness.run_query", fake_run_query([TEXT, RESULT]))
    result = await aut.run_task(1)

    assert result["success"] is True
    t = read_task(aut, 1)
    assert int(t["failure_count"]) == 0
    assert t["last_run"] == t["last_attempt"]
    assert aut._in_failure_cooldown(t, dt.datetime.now(dt.timezone.utc)) is False


async def test_max_retries_disables_task_and_alerts_once(aut, monkeypatch):
    """Without a terminal state a broken task burns one timeout per cooldown
    forever. #69 did ~130 consecutive timeouts over two days."""
    calls = []

    async def fake_alert(msg, *a, **k):
        calls.append(msg)

    monkeypatch.setattr("app.discord_notify.discord_alert", fake_alert, raising=False)
    write_task(aut, 1, max_retries=2, failure_count=1, timeout_seconds=1)
    monkeypatch.setattr("app.harness.run_query", fake_run_query([TEXT], delay=5))

    result = await aut.run_task(1)
    assert result["disabled"] is True
    t = read_task(aut, 1)
    assert t["status"] == "failed"
    assert len(calls) == 1

    # A disabled task is not dispatched, but stays visible for dependency lookups.
    assert aut._is_task_due(t, [t]) is False
    assert 1 in [int(x["id"]) for x in aut._all_runnable_tasks()]


# ── Empty responses ──────────────────────────────────────────────────────────

async def test_empty_response_after_work_is_a_task_failure(aut, monkeypatch):
    """An empty response was relabelled '(No response)' and recorded as SUCCESS,
    advancing last_run and unblocking dependents. #79 went dark for a week."""
    write_task(aut, 1)
    monkeypatch.setattr("app.harness.run_query",
                        fake_run_query([TOOLCALL, RESULT]))
    result = await aut.run_task(1)

    assert result["success"] is False
    assert result["failure_kind"] == "task"
    t = read_task(aut, 1)
    assert int(t["failure_count"]) == 1
    assert not t.get("last_run")


async def test_fast_empty_response_is_infra_and_does_not_escalate(aut, monkeypatch):
    """On 2026-09-01 every task returned empty in ~1s for 11 hours. Counting
    those would have disabled the whole fleet and needed 30 manual re-enables."""
    write_task(aut, 1, failure_count=0)
    monkeypatch.setattr("app.harness.run_query", fake_run_query([RESULT]))
    result = await aut.run_task(1)

    assert result["failure_kind"] == "infra"
    t = read_task(aut, 1)
    assert int(t.get("failure_count") or 0) == 0     # budget untouched
    assert t["status"] == "up_next"                   # not disabled
    assert aut._in_failure_cooldown(t, dt.datetime.now(dt.timezone.utc))


async def test_connection_error_is_infra(aut, monkeypatch):
    write_task(aut, 1)

    def boom(messages, options):
        raise __import__("httpx").ConnectError("all connection attempts failed")

    monkeypatch.setattr("app.harness.run_query", boom)
    result = await aut.run_task(1)
    assert result["failure_kind"] == "infra"
    assert int(read_task(aut, 1).get("failure_count") or 0) == 0


# ── Cancellation and timeout interaction with the pool ───────────────────────

async def test_cancellation_writes_a_run_record_and_reraises(aut, monkeypatch):
    """The pool cancels via asyncio.wait_for; CancelledError is a BaseException,
    so the run used to vanish with no record and the task stuck in_progress."""
    write_task(aut, 1, timeout_seconds=30)
    monkeypatch.setattr("app.harness.run_query", fake_run_query([TEXT], delay=5))

    task = asyncio.create_task(aut.run_task(1))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    runs = list((aut.AUTONOMY_RUNS_DIR / "1").glob("run_*.md"))
    assert len(runs) == 1
    assert "cancelled" in runs[0].read_text()
    assert read_task(aut, 1)["status"] == "up_next"   # not left in_progress


async def test_effective_timeout_stays_under_the_pool_cap(aut, monkeypatch):
    """Equal caps let the pool win the race, producing no run record at all."""
    write_task(aut, 1, timeout_seconds=1800)
    seen = {}

    async def _rq(messages, options):
        seen["ran"] = True
        yield TEXT
        yield RESULT

    monkeypatch.setattr("app.harness.run_query", _rq)
    await aut.run_task(1, max_duration=1800)
    assert seen["ran"]
    run = next((aut.AUTONOMY_RUNS_DIR / "1").glob("run_*.md")).read_text()
    assert "status: success" in run


# ── Concurrency ──────────────────────────────────────────────────────────────

async def test_in_progress_task_is_not_dispatched_or_re_run(aut, monkeypatch):
    """Two runs of #38 once started 9 seconds apart and interleaved."""
    write_task(aut, 1, status="in_progress", timeout_seconds=600,
               updated=_iso(seconds=5))
    t = read_task(aut, 1)
    assert aut._is_task_due(t, [t]) is False

    def must_not_run(messages, options):
        raise AssertionError("run_query called for an in-progress task")

    monkeypatch.setattr("app.harness.run_query", must_not_run)
    result = await aut.run_task(1)
    assert result.get("skipped") is True
    assert not (aut.AUTONOMY_RUNS_DIR / "1").exists()


async def test_stale_in_progress_is_recovered(aut):
    write_task(aut, 1, status="in_progress", timeout_seconds=60,
               updated=_iso(hours=2))
    assert aut.recover_stuck_tasks() == [1]
    assert read_task(aut, 1)["status"] == "up_next"


# ── Dependencies ─────────────────────────────────────────────────────────────

async def test_failed_upstream_does_not_satisfy_a_dependent(aut):
    """A phantom success on an upstream used to unblock its dependents."""
    write_task(aut, 1, last_run=_iso(days=2), last_attempt=_iso(minutes=5),
               failure_count=1)
    write_task(aut, 2, depends_on=1, last_run=_iso(days=2))
    tasks = [read_task(aut, 1), read_task(aut, 2)]
    assert aut._is_dependency_met(tasks[1], tasks) is False


async def test_stale_bypass_hours_lets_a_dependent_run(aut):
    """Documented as the 'fail forward' principle; nothing read the field."""
    write_task(aut, 1, last_run=_iso(days=3))
    write_task(aut, 2, depends_on=1, last_run=_iso(days=2), stale_bypass_hours=36)
    tasks = [read_task(aut, 1), read_task(aut, 2)]
    assert aut._is_dependency_met(tasks[1], tasks) is True

    write_task(aut, 3, depends_on=1, last_run=_iso(days=2))  # no bypass
    tasks = [read_task(aut, 1), read_task(aut, 3)]
    assert aut._is_dependency_met(tasks[1], tasks) is False


async def test_stale_bypass_waits_for_a_running_upstream(aut):
    write_task(aut, 1, status="in_progress", last_run=_iso(days=3),
               updated=_iso(minutes=1))
    write_task(aut, 2, depends_on=1, last_run=_iso(days=2), stale_bypass_hours=36)
    tasks = [read_task(aut, 1), read_task(aut, 2)]
    assert aut._is_dependency_met(tasks[1], tasks) is False


# ── Schedule windows ─────────────────────────────────────────────────────────

async def test_scheduled_at_derives_the_preferred_hour(aut, monkeypatch):
    """#60 set scheduled_at 04:30 and #81 documented 05:00; both were ignored
    because preferred_hours was null, and #81 ran at 17:57."""
    write_task(aut, 1, scheduled_at="04:30:00")
    t = read_task(aut, 1)
    assert aut._effective_preferred_hours(t) == [4]

    monkeypatch.setattr(aut, "_local_hour", lambda: 4)
    assert aut._is_preferred_hour(t) is True
    monkeypatch.setattr(aut, "_local_hour", lambda: 17)
    assert aut._is_preferred_hour(t) is False

    # An explicit list still wins, and a cron string yields no window.
    write_task(aut, 2, scheduled_at="04:30:00", preferred_hours=[9])
    assert aut._effective_preferred_hours(read_task(aut, 2)) == [9]
    write_task(aut, 3, scheduled_at="0 4 * * 0")
    assert aut._effective_preferred_hours(read_task(aut, 3)) is None


async def test_window_slack_prevents_a_skipped_day(aut, monkeypatch):
    """last_run is a completion time, so due-time drifts later by the run's own
    duration each cycle until it steps outside a one-hour window."""
    monkeypatch.setattr(aut, "_local_hour", lambda: 4)
    write_task(aut, 1, preferred_hours=[4], last_run=_iso(hours=23, minutes=20))
    t = read_task(aut, 1)
    assert aut._is_task_due(t, [t]) is True

    write_task(aut, 2, last_run=_iso(hours=23, minutes=20))  # no window, no slack
    t2 = read_task(aut, 2)
    assert aut._is_task_due(t2, [t2]) is False


# ── Silent-failure detector ──────────────────────────────────────────────────

def test_expected_error_patterns_suppress_known_indicators():
    """#48's dry-run is REQUIRED to raise FileNotFoundError while the graph is
    missing: 33 false positives in a week taught everyone to ignore the flag."""
    text = "Dry-run: rc=1, FileNotFoundError: _relationships.json — expected."
    assert autonomy._detect_silent_failures(text)
    assert autonomy._detect_silent_failures(text, ["FileNotFoundError"]) == []
    # Unrelated indicators still fire.
    other = "Traceback (most recent call last): boom"
    assert autonomy._detect_silent_failures(other, ["FileNotFoundError"])


def test_detector_default_behaviour_unchanged():
    assert autonomy._detect_silent_failures("exit code 3") 
    assert autonomy._detect_silent_failures("all good") == []
    assert autonomy._detect_silent_failures("") == []


# ── Frontmatter preservation ─────────────────────────────────────────────────

async def test_update_preserves_unknown_keys(aut):
    """Both writers rebuilt frontmatter from a fixed key list, silently
    destroying tags, stale_bypass_hours, expected_error_patterns and segment."""
    write_task(aut, 1, tags=["a", "b"], segment="autonomy",
               stale_bypass_hours=36, expected_error_patterns=["Foo"])
    aut._update_task_field(1, status="paused")
    t = read_task(aut, 1)
    assert t["tags"] == ["a", "b"]
    assert t["segment"] == "autonomy"
    assert float(t["stale_bypass_hours"]) == 36
    assert t["expected_error_patterns"] == ["Foo"]
    assert t["status"] == "paused"


# ── Health aggregation ───────────────────────────────────────────────────────

def test_compute_health_reclassifies_historical_phantom_successes():
    """Empty runs were written as status=success; the health view must not
    inherit that lie, and must attribute rows whose task_id is NULL."""
    # Newest first, matching list_runs_joined's ORDER BY completed_at DESC.
    rows = [
        # pool timeout: no task_id column, only the queue payload
        {"task_id": None, "queue_payload_json": json.dumps({"task_id": 1}),
         "status": "failed", "duration_seconds": 1800.0,
         "summary": "TimeoutError: exceeded max_duration_seconds=1800",
         "response_json": "", "meta_json": json.dumps({"pool_timeout": True}),
         "completed_at": "2026-09-03T12:00:00+00:00"},
        {"task_id": "1", "status": "success", "duration_seconds": 0.6,
         "summary": "(No response)", "response_json": "(No response)",
         "meta_json": None, "completed_at": "2026-09-03T11:00:00+00:00"},
        {"task_id": "1", "status": "success", "duration_seconds": 100.0,
         "summary": "ok", "response_json": "did work", "meta_json": None,
         "completed_at": "2026-09-03T10:00:00+00:00"},
    ]
    tasks = [{"id": 1, "name": "t1", "status": "up_next", "failure_count": 2}]
    h = autonomy.compute_health(rows, tasks, 7)
    t = h["tasks"][0]

    assert t["task_id"] == "1"
    assert t["runs"] == 3
    assert t["successes"] == 1
    assert t["failures"] == 2          # the empty "success" counts as a failure
    assert t["empty"] == 1
    assert t["timeouts"] == 1
    # Both most-recent runs failed; the streak stops at the older success.
    assert t["consecutive_failures"] == 2
    assert t["wasted_hours"] == pytest.approx(0.5, abs=0.01)
    assert h["fleet"]["runs"] == 3 and h["fleet"]["failures"] == 2


def test_compute_health_counts_silent_and_lists_disabled_tasks():
    rows = [{"task_id": "5", "status": "success", "duration_seconds": 8.0,
             "summary": "[SILENT]", "response_json": "[SILENT]",
             "meta_json": json.dumps({"silent": True, "stop_reason": "stop"}),
             "completed_at": "2026-09-03T10:00:00+00:00"}]
    tasks = [{"id": 5, "name": "t5", "status": "up_next"},
             {"id": 9, "name": "t9", "status": "failed"},
             {"id": 8, "name": "t8", "status": "paused"}]
    h = autonomy.compute_health(rows, tasks, 7)
    assert h["tasks"][0]["silent"] == 1
    assert h["tasks"][0]["silent_rate"] == 1.0
    assert h["fleet"]["failed_tasks"] == ["9"]
    assert h["fleet"]["paused_tasks"] == ["8"]
    assert {t["task_id"] for t in h["idle_tasks"]} == {"9", "8"}


# ── Worker pool contract ─────────────────────────────────────────────────────

async def test_pool_timeout_records_the_task_id(tmp_path, monkeypatch):
    """pool.record_run omitted task_id on the timeout path, so 237 rows /
    73.6 GPU-hours were unattributable to any task."""
    from workers.queue import WorkQueue
    from workers.pool import WorkerPool
    import workers.sources as sources

    q = WorkQueue(tmp_path / "w.db")

    class FakeSource:
        NAME = "scheduled-task"

        @staticmethod
        async def enqueue_if_due(queue, cfg):
            return None

        @staticmethod
        async def execute(item):
            await asyncio.sleep(10)

    monkeypatch.setitem(sources.SOURCE_REGISTRY, "scheduled-task", FakeSource)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"scheduled-task": {"max_duration_seconds": 1}})
    q.enqueue(source="scheduled-task", kind="run", payload={"task_id": 7})

    pool = WorkerPool(q, slots=1)
    pool._running = True
    worker = asyncio.create_task(pool._worker_loop("worker-0"))
    await asyncio.sleep(2.5)
    pool._running = False
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    runs = q.list_runs(source="scheduled-task")
    assert runs and runs[0]["status"] == "failed"
    assert runs[0]["task_id"] == "7"
    assert json.loads(runs[0]["meta_json"])["pool_timeout"] is True


async def test_pool_honours_in_band_failure_without_queue_retry(tmp_path, monkeypatch):
    """Raising sent the item back through the queue's retry path, so one
    timeout became up to max_attempts full re-runs before the scheduler's own
    cooldown was consulted."""
    from workers.queue import WorkQueue
    from workers.pool import WorkerPool
    import workers.sources as sources

    q = WorkQueue(tmp_path / "w.db")

    class FakeSource:
        NAME = "scheduled-task"

        @staticmethod
        async def enqueue_if_due(queue, cfg):
            return None

        @staticmethod
        async def execute(item):
            return {"status": "failed", "summary": "timed out after 900s",
                    "task_id": "36", "meta": {"timeout": True}}

    monkeypatch.setitem(sources.SOURCE_REGISTRY, "scheduled-task", FakeSource)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"scheduled-task": {"max_duration_seconds": 60}})
    qid = q.enqueue(source="scheduled-task", kind="run", payload={"task_id": 36})

    pool = WorkerPool(q, slots=1)
    pool._running = True
    worker = asyncio.create_task(pool._worker_loop("worker-0"))
    await asyncio.sleep(1.0)
    pool._running = False
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    runs = q.list_runs(source="scheduled-task")
    assert runs and runs[0]["status"] == "failed" and runs[0]["task_id"] == "36"
    item = q.get(qid)
    assert item.state == "completed"   # not requeued
    assert item.attempts == 1
