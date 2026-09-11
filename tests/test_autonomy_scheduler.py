"""Autonomy scheduler contract (autonomy.py + workers).

Pins the fixes from the 2026-09-03 fleet audit, which found ~73 GPU-hours a
week burned on failed runs. Each test names the failure mode it prevents.
"""
import asyncio
import ast
import datetime as dt
import inspect
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


# ── secondary_enabled switch ─────────────────────────────────────────────────

def test_secondary_routes_to_primary_when_disabled(monkeypatch):
    """server.py stops the secondary process whenever secondary_enabled is
    false, so a task pinned to `secondary` must fall back rather than aim at a
    dead port. Everything else already used this helper; autonomy did not."""
    from app.config import CONFIG, resolve_model_alias
    from workers.sources.scheduled_task import _model_health_url

    monkeypatch.setitem(CONFIG, "secondary_enabled", False)
    assert resolve_model_alias("secondary") == "primary"
    assert _model_health_url("secondary").endswith("8096/health")

    monkeypatch.setitem(CONFIG, "secondary_enabled", True)
    assert resolve_model_alias("secondary") == "secondary"
    assert _model_health_url("secondary").endswith("8091/health")
    # primary is unaffected either way
    assert _model_health_url("primary").endswith("8096/health")


# ── #870: one reproducible dependency gate, one resolution input ─────────────
#
# The gate read `datetime.datetime.now` twice inside its own body and its two
# callers handed it two different task sets — dispatch the status-filtered
# runnable set, the Mission Control board every parsed task. So the gate was not
# a function of its inputs (nothing could ask "would this have been due at
# instant T?") and the two surfaces could print opposite answers for one task in
# one second. Every test below pins the instant through the ONE module-level
# clock helper, `autonomy._utcnow`, and never touches a stdlib datetime
# attribute — which is why `_pin` passes `raising=False`: the helper is new, and
# the patch must not be the thing that fails.
#
# Red-for-the-right-reason, measured by checking out the base commit's source and
# re-running this section: most of these tests exercise API that did not exist
# (`now=`, `dependency_resolution_set`), so pre-fix they die on a TypeError or
# AttributeError, which is the honest outcome for new-API coverage.
# `test_paused_upstream_never_both_holds_and_dispatches` is the one written
# against only the old surface — it builds the board the way the endpoint built
# it before #870 — so pre-fix it fails on the disagreement assertion itself.

PIN = dt.datetime(2026, 9, 11, 12, 0, 0, tzinfo=dt.timezone.utc)


def _pin(aut, monkeypatch, when=PIN):
    """Replace the single module-level clock helper. Clause 2: one helper, and
    no patching of a stdlib `datetime` attribute anywhere in this section."""
    monkeypatch.setattr(aut, "_utcnow", lambda: when, raising=False)
    return when


def _stale_pair(aut, up_status="paused", up_age_days=2, dep_age_days=3):
    """The reproduction from the item: upstream `paused` and 2 days stale,
    dependent `up_next` and 3 days stale, both `daily`."""
    write_task(aut, 1, status=up_status,
               last_run=(PIN - dt.timedelta(days=up_age_days)).isoformat())
    write_task(aut, 2, depends_on=1,
               last_run=(PIN - dt.timedelta(days=dep_age_days)).isoformat())


async def test_gate_is_a_function_of_its_inputs(aut, monkeypatch):
    """Clause 1. Two calls, the same two task dicts, the same pinned instant →
    the same value. Before #813 the gate built its own `now` twice, so this
    question could not even be asked, let alone answered twice the same way."""
    _pin(aut, monkeypatch)
    _stale_pair(aut)
    tasks = [read_task(aut, 1), read_task(aut, 2)]
    first = aut._is_dependency_met(tasks[1], tasks, now=PIN)
    second = aut._is_dependency_met(tasks[1], tasks, now=PIN)
    assert first is False and second is False
    # And with no explicit instant, the pinned helper is still the only clock,
    # so the answer is the same one — reproducible across runs, not just calls.
    assert aut._is_dependency_met(tasks[1], tasks) is first
    # The two in-body wall-clock reads are gone; one indirection replaced them.
    # Read the CALLS, not the text: `_utcnow`'s own docstring quotes the literal,
    # and a source-text assertion would fail on someone rewording a comment.
    tree = ast.parse(inspect.getsource(aut._is_dependency_met))
    calls = [ast.unparse(node.func) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and node.func is not None]
    assert "datetime.datetime.now" not in calls, calls
    assert "_utcnow" in calls, calls


async def test_freshness_bound_asserted_from_both_sides(aut, monkeypatch):
    """Clause 3. A daily dependent's bound is `interval / 2` = 43200 s. One
    second inside the bound the dependency is met; one second outside it is not.
    A real-clock test cannot make that claim — the case sat exactly on a moving
    line, which is the flake #813 names."""
    _pin(aut, monkeypatch)
    inside = PIN - dt.timedelta(seconds=43200 - 1)    # 43199 s: inside
    outside = PIN - dt.timedelta(seconds=43200 + 1)   # 43201 s: outside
    stale = (PIN - dt.timedelta(days=3)).isoformat()
    write_task(aut, 1, last_run=inside.isoformat())
    write_task(aut, 3, last_run=outside.isoformat())
    write_task(aut, 2, depends_on=1, last_run=stale)
    write_task(aut, 4, depends_on=3, last_run=stale)
    board = [read_task(aut, i) for i in (1, 2, 3, 4)]
    by = {int(t["id"]): t for t in board}
    assert aut._is_dependency_met(by[2], board, now=PIN) is True
    assert aut._is_dependency_met(by[4], board, now=PIN) is False


@pytest.mark.parametrize("up_status", [
    "up_next", "in_progress", "failed", "paused", "draft", None])
async def test_board_and_scheduler_agree_for_every_upstream_status(
        aut, monkeypatch, up_status):
    """Clause 4. Every upstream status class, and the case where the
    `depends_on` id has no file at all: the board's hold reason and the
    scheduler's due decision must give one verdict, at one instant.

    The expected column is NOT this item choosing an answer for the absent id:
    the no-file row keeps today's fail-open answer, which is #558's to move.
    What is pinned here is that both surfaces say the same thing."""
    _pin(aut, monkeypatch)
    if up_status is None:                       # depends_on points at nothing
        write_task(aut, 2, depends_on=1,
                   last_run=(PIN - dt.timedelta(days=3)).isoformat())
        expect_held = False
    else:
        _stale_pair(aut, up_status=up_status)
        # Upstream exists and ran 2 days ago; the dependent is daily, so the
        # half-interval freshness bound is 12 h and it is 40x past. Met → held,
        # and `stale_bypass_hours` is unset, so nothing forwards past it.
        expect_held = True
    board = list(aut.dependency_resolution_set())
    due = [int(t["id"]) for t in aut.get_due_tasks(now=PIN)]
    dep = next(t for t in board if int(t["id"]) == 2)
    held = aut.hold_reason(dep, board, now=PIN)
    assert (2 in due) is not (held is not None), (
        f"upstream {up_status!r}: board said {held!r}, scheduler due={2 in due}")
    if expect_held:
        assert held == "waiting on #1" and 2 not in due
    else:
        assert held is None and 2 in due


async def test_paused_upstream_never_both_holds_and_dispatches(aut, monkeypatch):
    """Clause 5 — the check that opens the item, and the one test in this
    section written against ONLY the pre-#870 surface so it can fail on the
    behaviour rather than on a missing function: the board's set is assembled the
    way the endpoint assembled it (`every parsed task file, whatever its status`)
    and passed in positionally. Pre-fix, run against the base commit's
    autonomy.py, the assertion below fires: the scheduler returned the dependent
    — a `paused` upstream was invisible to the set dispatch resolved against, so
    the lookup missed it and read as satisfied — while the same call stack
    returned `waiting on #1` for it. Verified that way, not asserted."""
    _pin(aut, monkeypatch)
    _stale_pair(aut)                            # upstream paused, 2 days stale
    board = [t for t in (aut._parse_task_file(p)
                         for p in sorted(aut.AUTONOMY_DIR.glob("*.md"))) if t]
    due = [int(t["id"]) for t in aut.get_due_tasks()]
    dep = next(t for t in board if int(t["id"]) == 2)
    held = aut.hold_reason(dep, board)
    assert not (held == "waiting on #1" and 2 in due), (
        f"board and scheduler disagreed at one instant: board={held!r}, "
        f"get_due_tasks={due}")


async def test_the_stall_alarm_shares_the_one_verdict(aut, monkeypatch, tmp_path):
    """Clause 4/5 across the worker seam. `_grossly_overdue` is the gate's third
    caller and the only one that runs in the worker pool process rather than the
    backend; it handed `_is_task_due` the runnable set to resolve `depends_on`
    with, so a `paused` upstream was invisible to the alarm exactly as it was to
    dispatch, and an unheld-dependent-forever read as a broken dispatch path.
    This calls the worker-side function itself, not a copy of its logic."""
    from workers.queue import WorkQueue
    from workers.sources.scheduled_task import _grossly_overdue

    _pin(aut, monkeypatch)
    _stale_pair(aut)                              # upstream paused, 2 days stale
    # A genuinely stalled task, so the alarm returning nothing cannot pass this
    # test by being broken: hourly, last ran 10 days ago, nothing holding it.
    write_task(aut, 3, frequency="hourly",
               last_run=(PIN - dt.timedelta(days=10)).isoformat())
    q = WorkQueue(tmp_path / "w.db")

    # A bound pair, either side of `_STALL_INTERVAL_MULT * interval` (2.5 x 3600
    # = 9000 s) by one second. Both tasks are due; only the magnitude differs.
    # This is what pins the POOL's clock rather than assuming it: the alarm used
    # to read `_dt.datetime.now` locally for this comparison, and the real clock
    # is hours past PIN here, so both tasks would report overdue and the
    # one-second-outside case could only pass if the instant is the pinned one.
    bound = 9000
    write_task(aut, 4, frequency="hourly",
               last_run=(PIN - dt.timedelta(seconds=bound + 1)).isoformat())
    write_task(aut, 5, frequency="hourly",
               last_run=(PIN - dt.timedelta(seconds=bound - 1)).isoformat())

    alarm = _grossly_overdue(q)
    due = [int(t["id"]) for t in aut.get_due_tasks()]
    assert 3 in alarm, "the alarm stopped reporting a genuinely stalled task"
    assert 4 in alarm and 5 not in alarm, (
        f"the alarm is not answering at the pinned instant: "
        f"9001 s past reports={4 in alarm}, 8999 s past reports={5 in alarm}")
    assert 2 not in due and 2 not in alarm, (
        f"dispatch says due={2 in due} while the stall alarm says "
        f"{2 in alarm} — one gate, two answers")


async def test_both_gate_branches_under_one_pinned_instant(aut, monkeypatch):
    """Clause 7. The never-ran branch and the ran-but-stale branch, each with
    and without the `stale_bypass_hours` fail-forward, evaluated at ONE instant.

    The bypass pairs matter most: they differ only in how far `now` is from the
    upstream's `last_run`, and `now` used to be recomputed inside
    `_is_dependency_met` before being handed to `_dependency_bypassed` — which
    already took it as a parameter. If the instant were not the pinned one, the
    30-hour case would read ~40 hours against the real clock and both halves of
    each pair would come out True."""
    _pin(aut, monkeypatch)
    stale = (PIN - dt.timedelta(days=3)).isoformat()
    write_task(aut, 1, last_run="")                                     # never ran
    write_task(aut, 3, last_run=(PIN - dt.timedelta(hours=40)).isoformat())  # stale
    write_task(aut, 5, last_run=(PIN - dt.timedelta(hours=30)).isoformat())  # stale
    write_task(aut, 2, depends_on=1, last_run=stale)                    # no bypass
    write_task(aut=aut, task_id=6, depends_on=1, last_run=stale, stale_bypass_hours=36)
    write_task(aut, 4, depends_on=3, last_run=stale)                    # no bypass
    write_task(aut=aut, task_id=7, depends_on=3, last_run=stale, stale_bypass_hours=36)
    write_task(aut=aut, task_id=8, depends_on=5, last_run=stale, stale_bypass_hours=36)
    board = list(aut.dependency_resolution_set())
    by = {int(t["id"]): t for t in board}
    met = {d: aut._is_dependency_met(by[d], board, now=PIN) for d in (2, 4, 6, 7, 8)}
    # Never ran, no bypass window: not met, and the dependent waits.
    assert met[2] is False
    # Never ran, bypass set: the documented fail-forward answer is met.
    assert met[6] is True
    # Ran but stale past the half-interval bound, no bypass: not met.
    assert met[4] is False
    # Ran 40 h ago against a 36 h window: stale enough to forward. Met.
    assert met[7] is True
    # Ran 30 h ago against the same 36 h window: inside it. Not met. This is the
    # half that only holds if the evaluation used the pinned instant.
    assert met[8] is False
