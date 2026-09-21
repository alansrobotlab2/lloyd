"""Autonomy scheduler contract (autonomy.py + workers).

Pins the fixes from the 2026-09-03 fleet audit, which found ~73 GPU-hours a
week burned on failed runs. Each test names the failure mode it prevents.
"""
import asyncio
import ast
import datetime as dt
import inspect
import json
import logging
import os
import re
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
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "sys", raising=False)
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
    """The ladder doubles per consecutive failure up to one declared period.

    #1086 reshaped the numbers: a step is a fraction of the task's own period,
    so an hourly ladder is 900/1800/3600 where it used to be 600/2400/capped-at
    86400. Written against the frequency-blind ladder this test certified a
    term (`max(interval, 21600)`) that no task could reach at max_retries 3.
    """
    write_task(aut, 1, frequency="hourly", failure_count=1)
    t = read_task(aut, 1)
    assert aut._failure_cooldown_seconds(t) == 900
    t["failure_count"] = 2
    assert aut._failure_cooldown_seconds(t) == 1800
    t["failure_count"] = 99  # capped at one declared period (hourly)
    assert aut._failure_cooldown_seconds(t) == 3600

    # An old failure no longer holds the task back. A daily step is 6h, so the
    # gap has to clear that — 3h would now still be inside the cooldown.
    write_task(aut, 2, failure_count=1, last_attempt=_iso(hours=8))
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


# ── Frequency-aware retry spacing, and the way out of `failed` (#1086) ───────
#
# Two claims held here:
#   * one retry step is a fraction of the task's OWN declared period, so a
#     weekly job's permitted retries do not all land inside one coffee break;
#   * `status: failed` is a resting state with a way out — one declared period
#     after the disable the scheduler returns the task to `up_next` by itself,
#     writes a `next_run`, and logs the rearm, so a bad half hour costs one
#     cycle instead of ending the schedule until a human edits the file.
#
# The pre-change behaviour is re-readable at commit a7bb8744 (`git show
# a7bb8744:autonomy.py`): `_failure_cooldown_seconds` returned 600.0 s at
# failure_count 1 for `weekly`, `daily` and `hourly` alike, and
# `_write_run_record` was never handed the failure count, so no record could
# say which attempt it was. Both claims name that commit rather than a corpus
# this file cannot measure.


def _step(aut, frequency, failure_count):
    return aut._failure_cooldown_seconds(
        {"frequency": frequency, "failure_count": failure_count})


async def test_one_queue_tick_rearms_a_retired_task(aut, monkeypatch, tmp_path):
    """Clause 2 across the process boundary the rearm actually lives behind.

    The rearm is only automatic if the queue source's own tick reaches the scan,
    so this calls the tick — `workers.sources.scheduled_task.enqueue_if_due`, the
    function the pool runs every `tick_interval` seconds — over an isolated task
    dir, and reads the answer off the task files on disk. Nothing here greps
    source text, nothing here calls `recover_stuck_tasks` by hand.

    Before #1086 this tick looked at a `failed` task and walked past it: the
    scan it already called handled only `in_progress`, so the retired schedule
    stayed retired until a human edited the file, and both stall alarms skipped
    it (once for the status, once for the `next_run` the disable clears).

    Task 2 is the non-vacuity half: retired, but its own period has not elapsed,
    so a tick that rearmed everything indiscrimately would fail here instead of
    passing on a rule that is really "always rearm".
    """
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    notices: list[tuple[str, str]] = []
    monkeypatch.setattr(aut, "_notify_alert_channel",
                        lambda title, message: notices.append((title, message)))

    # `recover_stuck_tasks` reads the wall clock, so the ages are relative to
    # now: one daily period plus, versus six hours of a daily period.
    write_task(aut, 1, frequency="daily", status="failed", failure_count=3,
               last_run=_iso(hours=30), last_attempt=_iso(hours=30))
    write_task(aut, 2, frequency="daily", status="failed", failure_count=3,
               last_run=_iso(hours=30), last_attempt=_iso(hours=6))

    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    # Both stall streaks are seeded AT the alerting threshold, not below it, so
    # the `alerts == []` assertion below has a firing condition: on this seed a
    # tick that finds any task grossly overdue or the queue starving alerts on
    # its first pass. Seeded at 0 the counter needed five ticks and the
    # assertion could not fail inside one tick at all — the #1086 review called
    # that documenting intent rather than discriminating, and it was right.
    # The seed was checked by mutation, not by reading: with the rearm writing
    # `next_run` one period in the PAST instead of the future, this assertion
    # fails on the real nextrun stall alarm ("#1 (task1) is 24.0h past its
    # next_run"), so it is a live check on the rearm's next_run and not a
    # tautology about an empty list.
    monkeypatch.setattr(st, "_state", {**st._state, "startup_checked": True,
                                       "stall_streak": st._STALL_ALARM_TICKS,
                                       "nextrun_streak": st._STALL_NEXTRUN_TICKS,
                                       "stall_alerted_at": None,
                                       "nextrun_alerted_at": None})
    alerts: list[str] = []

    async def _record_alert(msg):
        alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _record_alert)

    q = WorkQueue(tmp_path / "rearm-tick.db")
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})

    t1, t2 = read_task(aut, 1), read_task(aut, 2)
    assert t1["status"] == "up_next", (
        "the tick must return a retired task to the schedule by itself")
    assert int(t1["failure_count"]) == 0
    assert t1.get("next_run"), (
        "a rearm without next_run is invisible to both stall alarms")
    assert t2["status"] == "failed" and int(t2["failure_count"]) == 3, (
        "one declared period of idleness is the whole rule; a task still inside "
        "its own period must stay resting")

    enqueued = {int(i.payload["task_id"]) for i in
                q.list_items(source="scheduled-task", limit=500)
                if i.state == "queued" and i.payload.get("task_id") is not None}
    assert 1 in enqueued, "the rearmed task is dispatchable on the same tick"
    assert 2 not in enqueued

    assert alerts == [], f"the tick that rearms must not also alarm: {alerts}"
    assert any(aut._rearm_alert_id("1-task1") in m for _t, m in notices), (
        "the rearm reaches the alert channel from inside the tick, so a healed "
        "schedule is distinguishable from an edited one")
    text = (aut.AUTONOMY_DIR / "1-task1.md").read_text()
    assert "AUTO-REARMED" in text, "the tick's rearm is in the Activity Log"
    # Clause 3 asks for a DATED line, so the stamp is pinned and not just the
    # words: one line, written in the task's own Activity Log, carrying the ISO
    # instant `_append_activity_log` puts on every entry. A reworded note keeps
    # this passing; an undated or duplicated one does not.
    rearm_lines = [ln for ln in text.splitlines() if "AUTO-REARMED" in ln]
    assert len(rearm_lines) == 1, f"exactly one rearm line: {rearm_lines}"
    assert re.match(r"^- \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z: AUTO-REARMED",
                    rearm_lines[0]), rearm_lines[0]


def test_first_retry_step_is_a_fraction_of_the_declared_period(aut):
    """`frequency` reached the ladder only through a cap that could not bind
    below failure_count 11, so the period was dead code (#607's spacing half)."""
    weekly = _step(aut, "weekly", 1)
    assert 3600 <= weekly <= 21600, weekly
    hourly = _step(aut, "hourly", 1)
    assert hourly < 3600, hourly
    assert weekly > hourly, "the declared period must reach the ladder"

    # A step never exceeds the task's own period (a retry may not outlive the
    # schedule it is retrying) and never falls below the 10-minute floor. The 6h
    # ceiling bounds the FIRST step only: a later step may double past it (a
    # daily task's second is 12h) because capping every step at 6h would stop
    # the ladder doubling for any task with a period of a day or more, which is
    # the property clause 1 asks to keep.
    for frequency, period in (("hourly", 3600.0), ("every-15min", 900.0),
                              ("daily", 86400.0), ("weekly", 604800.0)):
        for n in (1, 2, 3):
            step = _step(aut, frequency, n)
            assert step <= period, (frequency, n, step)
            assert step >= 600.0, (frequency, n, step)
        assert _step(aut, frequency, 1) <= min(period, 21600.0)

    # Still exponential while there is room below the ceiling.
    assert _step(aut, "hourly", 2) == 2 * _step(aut, "hourly", 1)
    assert _step(aut, "hourly", 3) == 2 * _step(aut, "hourly", 2)
    for n in (1, 2, 3, 4):
        assert _step(aut, "weekly", n + 1) >= _step(aut, "weekly", n)

    # A frequency the parser cannot read yields no period, and with no period
    # the ladder keeps the PRE-#1086 shape (10m doubling under a 6h ceiling)
    # rather than inheriting a daily period it was never given: coercing unknown
    # into daily would hand a `6x-daily` task a 6h first retry, longer than its
    # own cadence. Two live spellings land here: `cron: 0 4 * * *` and
    # `6x-daily` (task #24). Widening the parser is a separate change; what this
    # pins is the two properties that must hold whatever the fallback is: the
    # step is bounded and spaced, and a task that just failed is still HELD
    # BACK. `_in_failure_cooldown` has no "no period ⇒ no cooldown" branch, and
    # that is the storm shape #607 killed — so this is asserted through the gate,
    # not just through the scalar.
    cron = {"frequency": "cron: 0 4 * * *", "failure_count": 1}
    assert aut._frequency_interval_seconds(cron) is None
    # No period ⇒ the pre-#1086 ladder (10m doubling under a 6h ceiling), NOT
    # daily's 6h: an unknown period is not the same fact as a daily one, and
    # `6x-daily` (task #24) plus the spaced `every 15 min` both land here.
    assert _step(aut, "cron: 0 4 * * *", 1) == 600.0
    assert _step(aut, "cron: 0 4 * * *", 2) == 1200.0
    assert _step(aut, "cron: 0 4 * * *", 99) == 21600.0
    assert _step(aut, "cron: 0 4 * * *", 1) < _step(aut, "daily", 1)
    assert _step(aut, "6x-daily", 1) == 600.0
    fired = dt.datetime.now(dt.timezone.utc)
    held = dict(cron, last_attempt=fired.isoformat(),
                last_run=(fired - dt.timedelta(hours=9)).isoformat())
    assert aut._in_failure_cooldown(
        held, fired + dt.timedelta(seconds=30)) is True
    assert aut._in_failure_cooldown(
        held, fired + dt.timedelta(hours=7)) is False

    # The one sub-hour spelling the parser DOES read must not land on the hourly
    # ladder: `every-15min` has a 900 s period, so its step is capped there.
    assert _step(aut, "every-15min", 1) == 600.0
    assert _step(aut, "every-15min", 3) == 900.0


def test_the_declared_period_survives_the_task_file_round_trip(aut):
    """The ladder is computed from a parsed task file, not a literal dict, so
    what a task declares on disk is what must reach it."""
    write_task(aut, 1, frequency="weekly")
    write_task(aut, 2, frequency="hourly")
    weekly = aut._failure_cooldown_seconds(read_task(aut, 1))
    hourly = aut._failure_cooldown_seconds(read_task(aut, 2))
    assert 3600 <= weekly <= 21600, weekly
    assert hourly < 3600, hourly
    assert weekly > hourly


async def test_disabled_task_rearms_itself_at_the_next_declared_period(aut):
    """#78 burned 3 attempts inside ~30 min on 2026-09-09 and stayed `failed`
    until a human edited the file; #85 lost ~21 h the same way on 2026-09-16."""
    write_task(aut, 1, frequency="weekly", status="failed", failure_count=3,
               last_attempt=_iso(hours=6))    # weekly: one period has not passed
    write_task(aut, 2, frequency="daily", status="failed", failure_count=3,
               last_attempt=_iso(hours=30))   # daily: 1.25 periods have passed

    assert sorted(aut.recover_stuck_tasks()) == [2]

    t1, t2 = read_task(aut, 1), read_task(aut, 2)
    assert t1["status"] == "failed"
    assert int(t1["failure_count"]) == 3
    assert t2["status"] == "up_next"
    assert int(t2["failure_count"]) == 0
    # The rearm must write a next_run: a disabled task with no next_run is
    # invisible to both stall alarms (`_next_run_stalled` skips a missing one).
    assert t2.get("next_run")
    # And the rearm is a real return to the schedule: `up_next` is the only
    # status the queue source enqueues, and this is the gate it then passes.
    assert aut._is_task_due(t2, [t2]) is True
    assert aut._is_task_due(t1, [t1]) is False
    assert aut.hold_reason(t1, [t1], now=aut._utcnow()) is not None


async def test_the_rearm_is_logged_and_the_disable_alerts_exactly_once(aut, monkeypatch):
    """The alert is the point of retiring a task; the rearm is the point of not
    retiring it forever. A rearmed task must not start alerting every period."""
    calls = []

    async def fake_alert(msg, *a, **k):
        calls.append(msg)

    notices = []
    # The first positional is the alert TITLE, not a subsystem — `autonomy
    # ._notify_alert_channel(title, message)` — and naming it anything else made
    # the recorded tuple read as if the rearm were routed by subsystem (#1086
    # review advisory).
    monkeypatch.setattr(autonomy, "_notify_alert_channel",
                        lambda title, message: notices.append((title, message)))
    monkeypatch.setattr("app.discord_notify.discord_alert", fake_alert, raising=False)
    write_task(aut, 1, frequency="daily", max_retries=2, failure_count=1,
               timeout_seconds=1)
    monkeypatch.setattr("app.harness.run_query", fake_run_query([TEXT], delay=5))

    result = await aut.run_task(1)
    assert result["disabled"] is True
    assert read_task(aut, 1)["status"] == "failed"
    assert len(calls) == 1, "the disable alerts once"
    notices.clear()

    # Age it one declared period and let the ticker do what the ticker does.
    aut._update_task_field(1, last_attempt=_iso(hours=30))
    assert aut.recover_stuck_tasks() == [1]

    text = (aut.AUTONOMY_DIR / "1-task1.md").read_text()
    # The exact marker the rearm writes, not a loose `REARMED` substring that a
    # reworded note could satisfy (#1086 review advisory).
    assert "AUTO-REARMED" in text, "the rearm is recorded in the task's Activity Log"
    # The pairing id is pinned through the message the rearm actually passed to
    # the channel, not by comparing the formatter against a literal: the same
    # string has to appear in the Activity Log line as well, which is what lets a
    # reader pair this rearm with the DISABLED alert that retired the task.
    assert any(aut._rearm_alert_id("1-task1") in m for _s, m in notices), (
        "the rearm reaches the alert channel, so a healed schedule is not "
        "confused with an edited one")
    assert aut._rearm_alert_id("1-task1") in text, (
        "the Activity Log names the same id the alert carries, so the rearm can "
        "be paired with the DISABLED alert that retired the task")
    assert len(calls) == 1, "the rearm sends no DISABLED alert"
    # A re-armed task alerts again only by being disabled again — and for that
    # it must spend a whole retry budget, because the count was reset.
    result2 = await aut.run_task(1)
    assert result2["disabled"] is False
    assert read_task(aut, 1)["status"] == "up_next"
    assert int(read_task(aut, 1)["failure_count"]) == 1
    assert len(calls) == 1, "one alert per retirement, not one per period"


async def test_the_rearm_alert_actually_reaches_the_async_post(aut, monkeypatch):
    """The seam the alert tests above step over, walked end to end.

    The recovery scan that rearms a task is synchronous and
    `app.discord_notify.discord_alert` is async, and the scan runs inside the
    worker pool's own event loop via `run_in_executor` — so it cannot call
    `asyncio.run` on that stack and posts from a detached daemon loop
    (`autonomy._notify_alert_channel`). Patching the notifier, as
    `test_the_rearm_is_logged_and_the_disable_alerts_exactly_once` does, replaces
    that whole boundary: a coroutine that was created and never awaited would post
    nothing and still satisfy it. Here only the transport at the far side is
    replaced, the notifier itself runs, and the started thread is joined — so this
    fails if the post is never awaited, if it is awaited on a loop that is already
    running, or if the title is dropped. The Discord webhook beyond
    `discord_alert` is out of scope here; the seam this pins is sync scan → thread
    → `asyncio.run` → `discord_alert`.
    """
    posted = []

    async def fake_discord_alert(message, title=""):
        posted.append((message, title))

    monkeypatch.setattr("app.discord_notify.discord_alert", fake_discord_alert)

    thread = aut._notify_alert_channel("TITLE", "the rearm message")
    assert thread is not None, "the notifier hands back the Thread it started"
    await asyncio.to_thread(thread.join, 5.0)
    assert not thread.is_alive(), "the detached loop never finished the post"
    assert posted == [("the rearm message", "TITLE")], (
        "the alert must be awaited in that loop with its title intact")


async def test_a_failed_run_record_carries_the_attempt_number(aut, monkeypatch):
    """A record used to say nothing about which attempt it was. `_record_failure`
    already had the count in hand but handed `_write_run_record` only
    `failure_kind` (`git show a7bb8744:autonomy.py`, the extra dict in that call),
    which is why #78's five runs on 2026-09-08/09 had to be reassembled by hand
    from five timestamps."""
    started = dt.datetime.now(dt.timezone.utc)
    for n in (1, 2, 3):
        await aut._record_failure(
            {"id": 1, "name": "task1", "frequency": "weekly",
             "max_retries": 5, "failure_count": n - 1},
            1, f"run_1_probe{n}", started.isoformat(), started,
            summary="boom", body="body")

    counts = []
    for n in (1, 2, 3):
        run = (aut.AUTONOMY_RUNS_DIR / "1" / f"run_1_probe{n}.md").read_text()
        fm = yaml.safe_load(run.split("---\n", 2)[1])
        counts.append(fm["failure_count"])
    assert counts == [1, 2, 3], counts

    # And through the real path: one timed-out run records itself as attempt 1.
    write_task(aut, 2, frequency="weekly", timeout_seconds=1)
    monkeypatch.setattr("app.harness.run_query", fake_run_query([TEXT], delay=5))
    await aut.run_task(2)
    run = next((aut.AUTONOMY_RUNS_DIR / "2").glob("run_*.md")).read_text()
    assert yaml.safe_load(run.split("---\n", 2)[1])["failure_count"] == 1


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


# ── The never-ran bypass (#814) ──────────────────────────────────────────────

def _never_ran_bypass_lines(caplog) -> list[str]:
    """Every WARNING on `lloyd-autonomy` reporting the never-ran bypass.

    Filtered on the phrase rather than counted off `caplog.records`, so a test
    cannot be greened by some unrelated warning the fixture happens to raise. The
    filter is also its own positive control: `_warn_fail_closed`'s text ("…which
    dispatch does not run") shares no substring with "never run", so a zero here
    means the bypass did not speak, not that the pattern was wrong.
    """
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-autonomy"
            and r.levelno >= logging.WARNING
            and "never run" in r.getMessage().lower()]


def test_a_bypass_past_an_upstream_that_never_run_still_dispatches_and_says_so(
        aut, monkeypatch, caplog):
    """Clause 1. The verdict is unchanged; the silence is what goes.

    Upstream #1 exists and has NEVER run (`last_run` empty — a fresh, renamed or
    never-succeeded task); dependent #2 carries `stale_bypass_hours: 36`. This
    branch compares the window against nothing, so it forwards and dispatch
    proceeds — the documented purpose of the field, and what #870 pinned with
    `assert met[6] is True`. What #814 adds is that the pass is now announced
    exactly once, naming the upstream and saying it has never run: the dependent
    was consuming an artifact nobody had ever produced and no surface said so —
    `hold_reason` answers None for this pair, which clause 2 pins.

    The gate is asked twice on purpose. Dispatch re-answers it every 60 s tick,
    and a warning that prints 1,440 times a day is read by nobody, so "exactly
    one" here is one per (dependent, upstream) episode, not one per call.
    """
    _pin(aut, monkeypatch)
    monkeypatch.setattr(autonomy, "_never_ran_bypass_warned", {})
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    write_task(aut, 1, last_run="")                             # never ran
    write_task(aut, 2, depends_on=1, stale_bypass_hours=36,
               last_run=(PIN - dt.timedelta(days=2)).isoformat())
    board = [read_task(aut, 1), read_task(aut, 2)]
    assert aut._is_dependency_met(board[1], board, now=PIN) is True, (
        "the never-ran fail-forward moved: #814 chose to keep the verdict and "
        "make it visible, not to change it")
    assert aut._is_task_due(board[1], board, now=PIN) is True, (
        "the gate says met but dispatch still will not run the dependent")
    lines = _never_ran_bypass_lines(caplog)
    assert len(lines) == 1, (
        f"expected exactly one never-ran bypass warning across two gate "
        f"evaluations of the same pair, got {lines}")
    assert "#1" in lines[0], f"the warning must name the upstream: {lines[0]}"
    assert "#2" in lines[0], f"the warning must name the dependent: {lines[0]}"


def test_a_never_run_upstream_without_a_bypass_holds_and_names_itself(
        aut, monkeypatch, caplog):
    """Clause 2. Same board minus the window: held, said out loud, and silent.

    This is the control for clause 1 — without it, `len(lines) == 1` would also
    pass on a fixture that warns for the bare "my upstream is new" reason. No
    `stale_bypass_hours` means no bypass and no bypass warning; the hold is
    visible where a bypass cannot be, because `hold_reason` is exactly for the
    case that IS held.
    """
    _pin(aut, monkeypatch)
    monkeypatch.setattr(autonomy, "_never_ran_bypass_warned", {})
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    write_task(aut, 1, last_run="")                             # never ran
    write_task(aut, 2, depends_on=1,
               last_run=(PIN - dt.timedelta(days=2)).isoformat())
    board = [read_task(aut, 1), read_task(aut, 2)]
    assert aut._is_dependency_met(board[1], board, now=PIN) is False
    assert aut.hold_reason(board[1], board, now=PIN) == "waiting on #1"
    assert _never_ran_bypass_lines(caplog) == [], (
        "a dependent that is held must not also be reported as bypassing")


def test_a_never_run_upstream_that_is_in_progress_still_holds_its_dependent(
        aut, monkeypatch):
    """Clause 3. The ORDER of two checks in `_dependency_bypassed` is load-bearing.

    `test_stale_bypass_waits_for_a_running_upstream` above gives its upstream a
    `last_run` three days back, so it exercises the elapsed-time branch. The
    combination pinned here — running AND never-ran, the state an upstream is in
    while it runs its FIRST cycle — was unpinned, and holds only because the
    `in_progress` check sits above the `dep_last_run is None` return. Swap those
    two lines and every dependent of a first-time-running upstream dispatches
    against an artifact being written right now, with a green suite.
    """
    _pin(aut, monkeypatch)
    write_task(aut, 1, status="in_progress", last_run="", updated=_iso(minutes=1))
    write_task(aut, 2, depends_on=1, stale_bypass_hours=36,
               last_run=(PIN - dt.timedelta(days=2)).isoformat())
    board = [read_task(aut, 1), read_task(aut, 2)]
    assert aut._is_dependency_met(board[1], board, now=PIN) is False, (
        "an upstream running its first-ever run is an artifact in progress, not "
        "a missing one; the in_progress guard has to stay above the never-ran "
        "bypass")


def test_the_bypass_helper_names_the_dependent_and_states_the_never_ran_rule():
    """Clause 4. Whose field is this — in the signature and in the prose.

    `stale_bypass_hours: 36` sat on #38, an upstream root whose own `depends_on`
    is null and which is therefore read by nothing, while the code read the
    DEPENDENT — because the first parameter was named `task` and this docstring
    named upstreams. The #38→#42 edge had no fail-forward path at all, and a
    completed 12,700-byte report blocked the nightly chain ~10 h on
    2026-09-10/11. Both halves of the ambiguity are pinned: the parameter is
    `dependent`, and the docstring says in words that the window is the
    dependent's field and that the never-ran branch has no elapsed-time
    condition — the sentence the next reader needs before "bypass" reads as
    "it waited its turn".
    """
    params = list(inspect.signature(autonomy._dependency_bypassed).parameters)
    assert params[0] == "dependent", (
        f"the first parameter reads as the upstream again ({params[0]!r}); the "
        "2026-09-10/11 stall was caused by exactly that ambiguity")
    doc = (autonomy._dependency_bypassed.__doc__ or "").lower()
    assert "read from the dependent" in doc, (
        "the docstring stopped saying whose field this is")
    assert "never from the upstream" in doc, doc
    assert "no elapsed-time condition" in doc, (
        "the docstring stopped saying the never-ran branch measures nothing")
    assert "has never run" in doc, doc


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


def test_compute_health_counts_an_interrupted_run_as_a_timeout():
    """#1137 clause 4. A run killed by a restart or a pool cancel is written by
    the boot-time recovery sweep with `status='interrupted'` and no timeout meta
    — it never reached a timeout, the process holding it died. Classifying
    timeouts only from `meta.timeout`/`pool_timeout`/summary text is what made
    every one of them invisible: the fleet reported zero timeouts and zero
    wasted hours while items were re-queued from scratch.

    The interrupted row is a failure as well as a timeout, so its wall clock
    lands in `wasted_hours`, which is the field that exists to catch it."""
    rows = [
        {"task_id": "39", "status": "interrupted", "duration_seconds": 900.0,
         "summary": "recovered: no live owner at pool start", "response_json": "",
         "meta_json": None, "completed_at": "2026-09-20T00:00:00+00:00"},
        {"task_id": "39", "status": "success", "duration_seconds": 100.0,
         "summary": "ok", "response_json": "did work", "meta_json": None,
         "completed_at": "2026-09-20T01:00:00+00:00"},
    ]
    tasks = [{"id": 39, "name": "t39", "status": "up_next"}]
    h = autonomy.compute_health(rows, tasks, 7)
    t = h["tasks"][0]

    assert t["timeouts"] == 1, "the killed run is not counted as a timeout"
    assert t["wasted_hours"] == pytest.approx(0.25, abs=0.01), (
        "the killed run's 900 s is not in wasted_hours")
    assert t["runs"] == 2 and t["successes"] == 1 and t["failures"] == 1
    assert t["fail_rate"] == pytest.approx(0.5)
    # The timeout is visible at fleet level too — that is the number the report
    # and the autonomy_health tool answer with.
    assert h["fleet"]["timeout_runs"] == 1
    assert h["fleet"]["wasted_hours"] == pytest.approx(0.25, abs=0.01)


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
# attribute.
#
# Red-for-the-right-reason, measured by checking out the base commit's source and
# re-running this section: most of these tests exercise API that did not exist
# (`now=`, `dependency_resolution_set`), so pre-fix they die on a TypeError or
# AttributeError, which is the honest outcome for new-API coverage.
# `test_paused_upstream_never_both_holds_and_dispatches` is the one written
# against only the old surface — it builds the board the way the endpoint built
# it before #870 — so pre-fix it fails on the disagreement assertion itself.

PIN = dt.datetime(2026, 9, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
# Six hours after the upstream of `_stale_pair` last ran, i.e. inside that
# dependent's 12 h half-interval bound, where `PIN` is two days outside it. Used
# only to give one test two instants that the gate must answer differently.
AUX = PIN - dt.timedelta(days=2) + dt.timedelta(hours=6)


def _pin(aut, monkeypatch, when=PIN):
    """Replace the single module-level clock helper. Clause 2: one helper, and
    no patching of a stdlib `datetime` attribute anywhere in this section.

    The patch is STRICT on purpose. It used to pass `raising=False` so that a
    missing helper could not be the thing that failed a behavioural test — which
    also meant renaming `_utcnow` would have silently turned every pin in this
    section into a no-op, leaving one AST call-check as the only witness. A pin
    that pins nothing is worse than a failure, so the helper's existence is
    asserted here and a rename fails loudly, in every test in the section,
    naming the helper.
    """
    assert callable(getattr(aut, "_utcnow", None)), (
        "autonomy._utcnow is gone or not callable: the dependency gate's clock "
        "has no single indirection to pin, so no instant can be pinned")
    monkeypatch.setattr(aut, "_utcnow", lambda: when)
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
    question could not even be asked, let alone answered twice the same way.

    The fixture straddles the freshness bound on purpose, because the first
    version of this test did not: all three of its calls landed on `False`, so a
    gate that ignored `now=` and read the module clock regardless would have
    passed it. Same two dicts, two instants, two answers here — `PIN` puts the
    upstream 2 days behind a 12 h half-interval bound (not met), while `AUX`, 42 h
    earlier, sits 6 h behind that same bound and after the dependent's own run
    (met). So the parameter is doing the work, and the helper is the only thing
    that supplies it when it is absent.

    The upstream is `up_next`, not the section default of `paused`: #558 made a
    parked upstream hold at EVERY instant, so with a parked one all three calls
    return False and the test would pass without the `now=` parameter doing
    anything. The straddle needs an upstream the gate is willing to consider."""
    _pin(aut, monkeypatch, when=AUX)
    _stale_pair(aut, up_status="up_next")
    tasks = [read_task(aut, 1), read_task(aut, 2)]
    first = aut._is_dependency_met(tasks[1], tasks, now=PIN)
    second = aut._is_dependency_met(tasks[1], tasks, now=PIN)
    assert first is False and second is False
    # Same dicts, the module clock's instant: a DIFFERENT answer, which is what
    # shows `now=` is honoured rather than decorative — and that when it is
    # absent the pinned helper, not the wall clock, is what the gate reads.
    assert aut._is_dependency_met(tasks[1], tasks) is True
    # Re-pin to the instant the explicit calls used and the helper reproduces
    # them exactly: reproducible across runs, not merely across calls.
    _pin(aut, monkeypatch, when=PIN)
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

    What is pinned is the AGREEMENT — every row holds or none of them is
    believed. The expected column for the absent id moved on 2026-09-19: this row
    used to record `expect_held = False` with a note that the fail-open answer
    was #558's to move, and #558 moved it. Alan's ruling on the reopened item is
    that a `depends_on` naming no task file HOLDS its dependent (clause 3), so
    the no-file row now expects the hold for the same reason the `paused` and
    `draft` rows do. `test_the_rung_that_decides_a_dependent_is_the_rung_that_
    decides_a_run` in tests/test_autonomy_dependency_fail_closed.py is where the
    verdict itself is pinned, status by status; this test only refuses to let the
    two surfaces disagree about it."""
    _pin(aut, monkeypatch)
    if up_status is None:                       # depends_on points at nothing
        write_task(aut, 2, depends_on=1,
                   last_run=(PIN - dt.timedelta(days=3)).isoformat())
    else:
        _stale_pair(aut, up_status=up_status)
    # In every row the upstream either cannot run (#558) or ran 2 days ago — and
    # the dependent is daily, so the half-interval freshness bound is 12 h and
    # that is 40x past. Not met either way, and `stale_bypass_hours` is unset, so
    # nothing forwards past it.
    board = list(aut.dependency_resolution_set())
    due = [int(t["id"]) for t in aut.get_due_tasks(now=PIN)]
    dep = next(t for t in board if int(t["id"]) == 2)
    held = aut.hold_reason(dep, board, now=PIN)
    assert (2 in due) is not (held is not None), (
        f"upstream {up_status!r}: board said {held!r}, scheduler due={2 in due}")
    assert held == "waiting on #1" and 2 not in due, (
        f"upstream {up_status!r}: the scheduler dispatched a dependent whose "
        f"upstream cannot produce its input (board said {held!r})")


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
    # Positive control, in the same test, because the assertion above is
    # negative and would equally pass on a fixture where nothing is EVER due —
    # the shape that makes a green reproduction test prove nothing. Repair the
    # upstream (back to `up_next`, and 1 h old, inside the 12 h bound) and the
    # same two calls must now agree the other way: dispatched, and held by
    # nothing. The dependent is daily and last ran 3 days ago, so its own
    # interval is long past and the interval is not what is being tested here.
    write_task(aut, 1, last_run=(PIN - dt.timedelta(hours=1)).isoformat())
    board = [t for t in (aut._parse_task_file(p)
                         for p in sorted(aut.AUTONOMY_DIR.glob("*.md"))) if t]
    due = [int(t["id"]) for t in aut.get_due_tasks()]
    dep = next(t for t in board if int(t["id"]) == 2)
    held = aut.hold_reason(dep, board)
    assert 2 in due, f"repaired upstream, still not dispatched: held={held!r}, due={due}"
    assert held is None, f"dispatched while the board still holds it: {held!r}"


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


async def test_one_pool_tick_enqueues_exactly_what_the_board_calls_unheld(
        aut, monkeypatch, tmp_path):
    """#870 across the process seam: the worker's real tick, not a copy of it.

    Dispatch has three surfaces. `get_due_tasks` and the board endpoint are
    asserted against each other above; the third is the worker pool, whose
    `enqueue_if_due` imports `autonomy` inside the worker module, runs
    `get_due_tasks()` on an executor thread (scheduled_task.py:235) and writes
    the queue rows. Asserting the shared verdict only of an in-process function
    leaves the production claim — the pool enqueues what the board calls unheld —
    untested, so this drives the REAL coroutine against a real WorkQueue at one
    pinned instant. Only its out-of-process probes are stubbed: the vLLM health
    socket, the startup file scan and the Discord alert. `_state` is module-global
    and is reset so the tick cannot inherit an earlier test's stall streak and
    alert its way into a false failure.

    Every task here is past its OWN interval on purpose: `hold_reason` answers "is
    anything holding this" and has no "not due yet" branch, so `blocked is None`
    and "due" coincide only once nothing is sitting inside its window (#3 is
    every-15min and 20 min old, past its 900 s interval and still inside #4's
    hourly half-interval bound of 1800 s; #4 is hourly and 3 days old).
    """
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    _pin(aut, monkeypatch)
    _stale_pair(aut)                              # #1 paused 2 d, #2 waits on it
    write_task(aut, 3, frequency="every-15min",
               last_run=(PIN - dt.timedelta(minutes=20)).isoformat())
    write_task(aut, 4, depends_on=3,
               last_run=(PIN - dt.timedelta(days=3)).isoformat())

    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(st, "_state", {**st._state, "startup_checked": True,
                                       "stall_streak": 0, "stall_alerted_at": None})

    async def _no_alert(msg):
        raise AssertionError(f"a clean tick alerted: {msg}")

    monkeypatch.setattr(st, "_alert", _no_alert)

    q = WorkQueue(tmp_path / "tick.db")
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})

    enqueued = {str(i.payload.get("task_id")) for i in
                q.list_items(source="scheduled-task", limit=500)
                if i.state == "queued" and i.payload.get("task_id") is not None}
    board = list(aut.dependency_resolution_set())
    held = {str(t["id"]): aut.hold_reason(t, board, now=PIN) for t in board}
    unheld = {tid for tid, why in held.items() if why is None}
    assert enqueued == unheld, (
        f"one tick enqueued {sorted(enqueued)} while the board at the same "
        f"instant calls unheld {sorted(unheld)} (hold reasons: {held})")
    # Non-vacuous in both directions: the reproduction sits on the held side and
    # a satisfied dependent really reached the queue, so neither an empty queue
    # nor an always-enqueueing one can pass this.
    assert held["2"] == "waiting on #1" and "2" not in enqueued
    assert held["4"] is None and "4" in enqueued


# ── #421: a next_run-keyed stall assertion beside the due-ness alarm ─────────
#
# `_grossly_overdue` is the fleet's only stall detector and it is keyed on
# DUE-ness and on `last_run`, so its own three filters exclude the three shapes
# that actually went silent for 41 h (#51) and 60 h (#40) on 2026-09-07/08: a
# task blocked by an upstream whose run was never recorded is never DUE, a task
# that has never run has no `last_run`, and a task whose queue row is
# re-created every cycle reads as "starved for capacity, not stalled". The
# assertion added beside it is keyed on the task's own `next_run` and consults
# none of those three inputs, so it sees all of them. Every test in this section
# drives a SYNTHESIZED fleet: the live board currently has zero tasks past
# `next_run`, so a production-shaped test could not fail on the defect.

def _next_run_fleet(aut, *, control: bool = True):
    """Six tasks, one per shape, all `daily`, all read at the pinned `PIN`.

    `control=False` leaves out task 902, the one shape the OLD alarm does
    return. Tick-level tests need that: they assert an exact alert count, and
    the noisy alarm would otherwise add its own message on the same tick and
    make the count about two alarms instead of one.

    Ids are in the 900s so nothing here can collide with a real task id.
    `next_run` is the only field the new assertion reads; `last_run`, `status`
    and `depends_on` are set to the values that make each shape invisible to
    `_grossly_overdue`, and the same fleet is asserted against that alarm in the
    same test — so clause 6 ('the exclusions remain') cannot pass by being
    unexercised.
    """
    day = dt.timedelta(days=1)
    # Shape 1 (clause 1): `depends_on` an upstream 5 days stale against a 12 h
    # half-interval bound, so `_is_task_due` says False and the noisy alarm's
    # due-ness filter drops it before it can ever be counted.
    write_task(aut, 950, status="paused",
               last_run=(PIN - 5 * day).isoformat(),
               next_run=(PIN + 2 * day).isoformat())
    write_task(aut, 900, depends_on=950,
               last_run=(PIN - 6 * day).isoformat(),
               next_run=(PIN - 2 * day).isoformat())
    # Shape 2 (clause 2): never ran, so `last_run` is null and the noisy
    # alarm's `if not last: continue` skips it outright.
    write_task(aut, 901, last_run=None,
               next_run=(PIN - 2 * day).isoformat())
    # Shape 3 (clause 3): a live queue row, added by the caller — the noisy
    # alarm's `if str(id) in active: continue` skips it.
    write_task(aut, 903, last_run=(PIN - 3 * day).isoformat(),
               next_run=(PIN - 2 * day).isoformat())
    # Control (clause 6): due, 5 days grossly overdue, dependency-free, no
    # queue row — the one shape the noisy alarm does return, and still must.
    if control:
        write_task(aut, 902, last_run=(PIN - 5 * day).isoformat(),
                   next_run=(PIN - 4 * day).isoformat())
    # Clause 4: the one-interval bound, one second either side, each with an
    # ordinary `last_run` one period before its `next_run`. Only a pinned clock
    # can put a case 1 s from a one-day line, so these two pin the instant too.
    write_task(aut, 904,
               last_run=(PIN - dt.timedelta(seconds=2 * 86400 - 1)).isoformat(),
               next_run=(PIN - dt.timedelta(seconds=86400 - 1)).isoformat())
    write_task(aut, 905,
               last_run=(PIN - dt.timedelta(seconds=2 * 86400 + 1)).isoformat(),
               next_run=(PIN - dt.timedelta(seconds=86400 + 1)).isoformat())


def test_the_next_run_assertion_sees_the_three_shapes_the_stall_alarm_excludes(
        aut, monkeypatch, tmp_path):
    """Clauses 1, 2, 3, 4 and 6 on one synthesized board.

    Clause 6 is asserted here rather than in its own test on purpose: the old
    alarm is graded on the SAME fleet as the new one, so the three exclusions
    and the control are all exercised in the same breath as the new keys."""
    from workers.queue import WorkQueue
    from workers.sources.scheduled_task import _grossly_overdue, _next_run_stalled

    _pin(aut, monkeypatch)
    _next_run_fleet(aut)
    q = WorkQueue(tmp_path / "nextrun.db")
    q.enqueue(source="scheduled-task", kind="run",
              payload={"task_id": 903, "name": "task903"},
              priority=30, dedup_key="scheduled-task:903")

    # Clause 6, first and unqualified: the noisy alarm is unchanged, down to
    # the three exclusions this item says to leave alone.
    assert _grossly_overdue(q) == [902], (
        "the existing alarm changed: it must still return the due, grossly "
        "overdue, dependency-free, queue-row-free control and nothing else")

    flagged = {e["id"]: e for e in _next_run_stalled(q)}
    for missing in (900, 901, 903):
        assert missing in flagged, f"shape {missing} is still invisible to the alarm"
    assert 902 in flagged, "the noisy alarm's own case disappeared from the new one"
    # Clause 4, and with it the pinned instant: 86399 s past `next_run` is
    # inside one period, 86401 s is past it.
    assert 904 not in flagged, "flagged a task less than one period past next_run"
    assert 905 in flagged, (
        "did not flag a task one second past one period — the bound is not "
        "answering at the pinned instant")
    # Clause 1's second half, stated as the clause states it: the task is
    # flagged EVEN THOUGH the scheduler's own due gate rejects it. If the
    # fixture ever stops being that case, this assertion says so instead of
    # letting the clause above pass vacuously.
    board = list(aut.dependency_resolution_set())
    t900 = next(t for t in board if int(t["id"]) == 900)
    assert aut._is_task_due(t900, board, now=PIN) is False, (
        "the fixture is no longer a dependency-blocked case, so the clause "
        "about _is_task_due returning False is not being tested")
    # Clause 3's second half: the queue-row shape is flagged with its queue
    # state carried, not silently merged with the un-queued ones.
    assert flagged[903]["queued"] is True and flagged[900]["queued"] is False


async def test_the_next_run_alert_carries_the_hold_reason_and_fires_low_frequency(
        aut, monkeypatch, tmp_path):
    """Clauses 1, 2 and 5 across the executor seam, plus the low-frequency half.

    Drives the real `enqueue_if_due` coroutine tick after tick the way the pool
    calls it, stubbing only what leaves the process: the vLLM health socket and
    the Discord alert. "Low-frequency" is part of the acceptance contract — the
    alarm this one must not disturb needs 5 ticks and a 6 h cooldown, so the
    quieter one is asserted to confirm across ticks and to respect its own
    cooldown, not assumed to."""
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    _pin(aut, monkeypatch)
    # No control task: task 902 is the one shape the noisy alarm returns, and
    # this test counts alerts, so leaving it in would make the expected count a
    # sum of two alarms. Its own alarm is pinned unchanged by
    # test_the_stall_alarm_shares_the_one_verdict and again in the test above.
    _next_run_fleet(aut, control=False)
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(st, "_state", {**st._state, "startup_checked": True,
                                       "stall_streak": 0, "stall_alerted_at": None,
                                       "nextrun_streak": 0,
                                       "nextrun_alerted_at": None})
    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _capture)
    q = WorkQueue(tmp_path / "nextrun-tick.db")
    ticks = st._STALL_NEXTRUN_TICKS

    for _ in range(ticks - 1):
        await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert not alerts, (
        f"alerted on tick {len(alerts)} of {ticks} — the new assertion does not "
        f"confirm across ticks the way the alarm beside it does")
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert len(alerts) == 1, f"expected exactly one alert on tick {ticks}, got {len(alerts)}"
    for _ in range(ticks):
        await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert len(alerts) == 1, (
        "re-alerted inside its own cooldown — that is the noisy alarm's failure")

    msg = alerts[0]
    # The count above is only about the NEW alarm if the old one stayed silent
    # on this fleet — assert that rather than trusting `control=False`.
    assert all("Autonomy scheduler may be stalled" not in m for m in alerts), alerts
    # Clause 5: the message carries the string `autonomy.hold_reason` returns,
    # not a paraphrase, so a change to that function surfaces here.
    board = list(aut.dependency_resolution_set())
    t900 = next(t for t in board if int(t["id"]) == 900)
    reason = aut.hold_reason(t900, board, now=PIN)
    assert reason == "waiting on #950", f"fixture drifted: hold reason is {reason!r}"
    assert "900" in msg and reason in msg, f"task 900 flagged without its reason: {msg}"
    # The never-ran task is held by nothing, so it must say so instead of
    # printing an empty reason: a stall message nobody can act on is the defect
    # this item is about.
    assert "901" in msg and "nothing holds it" in msg, (
        f"never-run task flagged without an actionable reason: {msg}")


async def test_the_pool_reaches_the_next_run_assertion_through_the_registry(
        aut, monkeypatch, tmp_path):
    """The seam the code graph cannot see, crossed for real.

    `graph_affected(enqueue_if_due)` returns ZERO dependents: the pool never
    names this function, it dispatches `source.enqueue_if_due(...)` by attribute
    off `SOURCE_REGISTRY` (workers/pool.py:391). Every assertion made by calling
    the coroutine directly is therefore an assertion about a caller production
    does not have. This registers the real module and lets the real scheduler
    loop find it.

    The streak is seeded one tick short because the pool's loop is 60 s a tick
    and a test may not wait five minutes — carrying state across ticks is
    exactly what the module-global `_state` does between real ticks. Nothing in
    this fleet is due, so the pool's worker loop has nothing to claim and
    `run_task` is never reached; the assertion that the queue stayed empty is
    what proves that, and it is also what makes driving a real pool safe here."""
    from workers.queue import WorkQueue
    import workers.sources as sources
    import workers.sources.scheduled_task as st
    from workers.pool import WorkerPool

    _pin(aut, monkeypatch)
    day = dt.timedelta(days=1)
    write_task(aut, 950, status="paused",
               last_run=(PIN - 5 * day).isoformat(),
               next_run=(PIN + 2 * day).isoformat())
    write_task(aut, 900, depends_on=950,
               last_run=(PIN - 6 * day).isoformat(),
               next_run=(PIN - 2 * day).isoformat())

    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(st, "_state", {**st._state, "startup_checked": True,
                                       "stall_streak": 0, "stall_alerted_at": None,
                                       "nextrun_streak": st._STALL_NEXTRUN_TICKS - 1,
                                       "nextrun_alerted_at": None})
    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _capture)
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {"scheduled-task": st},
                        raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {
        "scheduled-task": {"enabled": True, "interval_seconds": 0,
                           "max_duration_seconds": 60}}, raising=False)

    q = WorkQueue(tmp_path / "pool.db")
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        for _ in range(300):
            if alerts:
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()

    assert alerts, ("the pool's own scheduler loop never reached the next_run "
                    "assertion — it would be dead code in production")
    assert "900" in alerts[0] and "waiting on #950" in alerts[0], alerts[0]
    assert not q.list_items(source="scheduled-task", limit=50), (
        "a dependency-blocked task reached the queue; this test would then "
        "have executed it")


# ── #1121: the stall scan must see the statuses that stop a task ─────────────
#
# `_next_run_stalled` iterated the whole resolution set and then dropped every
# task whose status was not `up_next`, so `draft`, `paused` and `failed` — the
# statuses that STOP a task — were invisible to the one mechanism built to catch
# a stall. #68 `Email & Calendar Triage` (frequency `every-15min`, the fleet's
# highest-volume task) went dark on an unattributed `up_next -> draft` flip while
# that alarm reported zero stalls; measured 2026-09-18 on the live board it sits
# 42.1 h past its own `next_run`, a gap_ratio of 168 periods, and the shipped
# scan returns `[]`. `autonomy_health` had no elapsed-time field at all, so the
# same call scored #68 `runs 30 / successes 30 / fail_rate 0.0` over a 1-day
# window and listed it in no failure, idle or gap list.
#
# Both halves now read one predicate, `autonomy.next_run_gap`, so the alert and
# the fleet report cannot call the same board two different things.

def _status_fleet(aut):
    """Six tasks, one per shape, all read at the pinned `PIN`. Ids in the 900s.

    906/907 are the case the filter swallowed: a `draft` task shaped like #68
    and a `paused` task, each far past its own `next_run`. 908 is clause 5's
    control — `up_next`, on cadence, half a period past `next_run`, and flagged
    nowhere. 909/910 are the two never-run shapes clause 4 must tell apart: a
    daily job three periods overdue and a weekly job whose `next_run` is still
    five days off. 911 is a `draft` task that is NOT overdue, because a
    deliberate stop is not itself a stall and the widening must not say it is.
    902 is the noisy alarm's own control, reused from `_next_run_fleet` with the
    same stamps, so clause 5's "unchanged" is asserted against a case that is
    known to fire rather than against an empty list."""
    day = dt.timedelta(days=1)
    write_task(aut, 906, status="draft", frequency="every-15min",
               last_run=(PIN - dt.timedelta(hours=12, minutes=30)).isoformat(),
               next_run=(PIN - dt.timedelta(hours=12, minutes=15)).isoformat())
    write_task(aut, 907, status="paused",
               last_run=(PIN - 3 * day).isoformat(),
               next_run=(PIN - 2 * day).isoformat())
    write_task(aut, 908, frequency="hourly",
               last_run=(PIN - dt.timedelta(minutes=40)).isoformat(),
               next_run=(PIN - dt.timedelta(minutes=30)).isoformat())
    write_task(aut, 909, last_run=None, next_run=(PIN - 3 * day).isoformat())
    write_task(aut, 910, frequency="weekly", last_run=None,
               next_run=(PIN + 5 * day).isoformat())
    write_task(aut, 911, status="draft", frequency="weekly",
               last_run=(PIN - 2 * day).isoformat(),
               next_run=(PIN + 5 * day).isoformat())
    write_task(aut, 902, last_run=(PIN - 5 * day).isoformat(),
               next_run=(PIN - 4 * day).isoformat())


def test_the_stall_scan_flags_a_draft_task_and_names_its_status(
        aut, monkeypatch, tmp_path):
    """Clauses 1 and 5, plus the never-run shapes clause 4 has to separate.

    Pre-fix this fails on the first assertion: 906 and 907 were dropped by the
    status filter, so the flagged set was {902, 909} and a board containing only
    non-`up_next` tasks returned `[]` — which is exactly what the shipped alarm
    returned for the live board while #68 was ~168 periods late."""
    from workers.queue import WorkQueue
    from workers.sources.scheduled_task import _grossly_overdue, _next_run_stalled

    _pin(aut, monkeypatch)
    _status_fleet(aut)
    q = WorkQueue(tmp_path / "status-stall.db")

    flagged = {e["id"]: e for e in _next_run_stalled(q)}
    assert flagged[906]["status"] == "draft", (
        "a draft task more than one period past its next_run is still invisible "
        "to the stall scan — the filter this item exists to remove")
    assert flagged[906]["hold"] == "draft", (
        "the carried reason is not the string autonomy.hold_reason returns for a "
        f"draft task: {flagged[906]['hold']!r}")
    assert flagged[907]["status"] == flagged[907]["hold"] == "paused"
    assert {906, 907, 909, 902} <= set(flagged), (
        f"stall set is {sorted(flagged)} — every shape the scan should see must "
        "be in it")

    # Clause 4's two never-run shapes, separated at the row level: the daily job
    # that has never run is 3 periods past and IS flagged; the weekly job that
    # has never run is 0.0 and is not. `never_run` is True for both, so it is the
    # gap_ratio that tells an overdue job from one that was never due.
    assert flagged[909]["gap_ratio"] == pytest.approx(3.0, abs=0.01)
    assert 910 not in flagged, "flagged a weekly task whose next_run is ahead"
    assert 911 not in flagged, ("flagged a draft task that is not past its own "
                                "next_run — a deliberate stop is not a stall")

    # Clause 5, both directions in one line: widening the scan must not move the
    # due-ness alarm. It still returns its own control 902, and only 902 —
    # `draft`/`paused` are outside `_all_runnable_tasks`, the hourly control is
    # inside its interval, and neither never-run task has a `last_run`.
    assert _grossly_overdue(q) == [902], (
        "the widened scan leaked into the noisy alarm")

    # And the same predicate the scan used, read directly: the bound is still
    # STRICTLY one interval, so 908 (half a period) is nowhere near it.
    board = list(aut.dependency_resolution_set())
    t908 = next(t for t in board if int(t["id"]) == 908)
    assert aut.next_run_gap(t908, now=PIN)["past_next_run"] is False
    assert 908 not in flagged, "flagged an up_next task less than one period late"


async def test_the_widened_alert_names_the_draft_task_and_its_status(
        aut, monkeypatch, tmp_path):
    """Clause 2 across the executor seam, on a board nothing else can flag.

    Only the #68-shaped draft task on the board, so the flagged set is entirely
    non-`up_next`: pre-fix this test gets zero alerts on every tick and the
    message it does not produce obviously cannot name a status. The count phrase
    is asserted too, because the alert used to read `N up_next task(s)`
    unconditionally — an alert that misdescribes its own contents is the same
    defect one level up."""
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    _pin(aut, monkeypatch)
    write_task(aut, 906, status="draft", frequency="every-15min",
               last_run=(PIN - dt.timedelta(hours=12, minutes=30)).isoformat(),
               next_run=(PIN - dt.timedelta(hours=12, minutes=15)).isoformat())
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(st, "_state", {**st._state, "startup_checked": True,
                                       "stall_streak": 0, "stall_alerted_at": None,
                                       "nextrun_streak": 0,
                                       "nextrun_alerted_at": None})
    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _capture)
    q = WorkQueue(tmp_path / "draft-alert.db")
    for _ in range(st._STALL_NEXTRUN_TICKS):
        await st.enqueue_if_due(q, {"max_duration_seconds": 1800})

    assert len(alerts) == 1, f"expected one alert on the confirm tick, got {alerts}"
    msg = alerts[0]
    assert "#906" in msg, f"the draft task is not named in the alert: {msg}"
    assert "held: draft" in msg, (
        f"the alert does not carry the status as its reason: {msg}")
    assert msg.startswith("1 draft task(s)"), (
        f"the count line does not describe the statuses it flagged: {msg}")
    assert "up_next task(s)" not in msg, (
        f"the message still asserts the old scope while flagging a draft task: {msg}")
    assert "Autonomy scheduler may be stalled" not in msg, (
        "the due-ness alarm fired on a board with no runnable task")


def test_next_run_gap_is_the_single_predicate_both_surfaces_share(aut, monkeypatch):
    """The arithmetic both #1121 halves read, at its three bounds.

    `gap_ratio` is measured against `last_run` — the reference the item's clause
    3 names — and the stall bound against `next_run`, because a last_run-based
    ratio crosses 1.0 the moment a task becomes due: measured on the live board
    2026-09-18, the healthy nightly task #51 reads gap_ratio 1.01 on that
    reference while #68 reads 169.4, so `gap_ratio > 1` as an alarm would fire on
    every on-cadence job. This test pins that the two are NOT conflated."""
    _pin(aut, monkeypatch)
    day = dt.timedelta(days=1)
    write_task(aut, 920, frequency="daily",
               last_run=(PIN - dt.timedelta(hours=25)).isoformat(),
               next_run=(PIN - dt.timedelta(hours=1)).isoformat())
    write_task(aut, 921, frequency="weekly",
               last_run=(PIN - 2 * day).isoformat(),
               next_run=(PIN - 2 * day).isoformat())
    write_task(aut, 922, frequency="bogus",
               last_run=(PIN - 9 * day).isoformat())
    board = {int(t["id"]): t for t in aut.dependency_resolution_set()}

    g920 = aut.next_run_gap(board[920], now=PIN)
    assert g920["expected_interval_seconds"] == 86400
    assert g920["hours_since_last_run"] == pytest.approx(25.0, abs=0.01)
    assert g920["gap_ratio"] == pytest.approx(25 / 24, abs=0.01)
    # One hour past next_run against a one-day period: inside the bound, so a
    # task on cadence inside its own window is never a stall. That is the
    # difference between this alarm and the noisy one.
    assert g920["past_next_run"] is False, (
        "the stall bound moved off the strict one-interval next_run test")
    assert g920["never_run"] is False

    # Two days past `next_run` against a SEVEN-day period is still inside one
    # period, so the widened scan does not flag it. This is the bound clause 5
    # leans on: widening which statuses are scanned did not widen how late a task
    # has to be, and a weekly job cannot be made to look stalled by a bad week.
    g921 = aut.next_run_gap(board[921], now=PIN)
    assert g921["past_next_run"] is False, (
        "flagged a weekly task 2 days past next_run — the bound is one full "
        "interval, not one calendar day")
    assert g921["gap_ratio"] == pytest.approx(2 / 7, abs=0.01)

    # A frequency the parser does not know yields no verdict, not a healthy one:
    # a denominator of None must never become a 0.0 gap.
    g922 = aut.next_run_gap(board[922], now=PIN)
    assert g922["gap_ratio"] is None and g922["past_next_run"] is False
    assert g922["hours_since_last_run"] == pytest.approx(216.0, abs=0.01)


def test_health_reports_the_real_68_gap_alongside_a_passing_fail_rate(aut, monkeypatch):
    """Clauses 3 and 4 on #68's real row shape: 30 successes in the window, dark
    for half of it.

    `compute_health` is given `now=` so the elapsed arithmetic is askable at a
    chosen instant instead of at whatever wall clock the suite runs on. The
    window contains only successes — that is the whole point: `fail_rate 0.0`
    must not be able to cover a task that has stopped running, which is what the
    2026-09-17 `autonomy_health(days=1)` call did to #68."""
    _pin(aut, monkeypatch)
    minutes = dt.timedelta(minutes=15)
    write_task(aut, 968, status="draft", name="Email Calendar Triage",
               frequency="every-15min",
               last_run=(PIN - dt.timedelta(hours=12, minutes=30)).isoformat(),
               next_run=(PIN - dt.timedelta(hours=12, minutes=15)).isoformat())
    rows = [{"task_id": "968", "status": "success", "duration_seconds": 60.0,
             "summary": "ok", "response_json": "did work", "meta_json": None,
             # The 30 runs sit BEFORE the 12.5 h dark period, the way #68's did:
             # the window holds only successes, then the task stops, and the
             # window still cannot see the stopping.
             "completed_at": (PIN - dt.timedelta(hours=13) - i * minutes).isoformat()}
            for i in range(30)]
    tasks = list(aut.dependency_resolution_set())
    h = autonomy.compute_health(rows, tasks, 1, now=PIN)

    row = next(t for t in h["tasks"] if t["task_id"] == "968")
    assert row["runs"] == 30 and row["fail_rate"] == 0.0, (
        "the fixture stopped being the passing-rate case")
    assert row["expected_interval_seconds"] == 900
    assert row["hours_since_last_run"] == pytest.approx(12.5, abs=0.01)
    assert row["gap_ratio"] >= 40, (
        f"gap_ratio {row['gap_ratio']} is not the ≥ 40 the item's clause 3 "
        "requires for a 12.5 h gap at every-15min")
    assert row["never_run"] is False

    stalled = {s["task_id"]: s for s in h["stalled"]}
    assert "968" in stalled, (
        f"a task 50 periods past its next_run is absent from the stalled list: "
        f"{sorted(stalled)}")
    assert stalled["968"]["fail_rate"] == 0.0 and stalled["968"]["runs_in_window"] == 30, (
        "the stalled entry was filtered out by a passing rate rather than carried "
        "alongside it")
    assert stalled["968"]["hold"] == "draft"
    assert stalled["968"]["gap_ratio"] == row["gap_ratio"], (
        "the stalled entry and the per-task row computed the same task's gap "
        "differently — the two surfaces must read one predicate")

    # Window-independence, asserted rather than asserted-by-construction: an
    # identical task with NO rows in the window carries the same gap fields.
    write_task(aut, 969, status="draft", frequency="every-15min",
               last_run=(PIN - dt.timedelta(hours=12, minutes=30)).isoformat(),
               next_run=(PIN - dt.timedelta(hours=12, minutes=15)).isoformat())
    h2 = autonomy.compute_health(rows, list(aut.dependency_resolution_set()), 1, now=PIN)
    idle = next(t for t in h2["idle_tasks"] if t["task_id"] == "969")
    assert idle["runs"] == 0
    assert (idle["hours_since_last_run"], idle["gap_ratio"]) == (
        row["hours_since_last_run"], row["gap_ratio"]), (
        "the gap fields depend on how many rows fell in the window; they must "
        "come from the task's own stamps")


def test_health_distinguishes_a_never_run_overdue_task_from_a_weekly_one_not_due(
        aut, monkeypatch):
    """Clause 4's second half: `run_count: 0` alone cannot say which of the two
    shapes it is, so both fields have to be on the row."""
    _pin(aut, monkeypatch)
    day = dt.timedelta(days=1)
    write_task(aut, 909, last_run=None, next_run=(PIN - 3 * day).isoformat())
    write_task(aut, 910, frequency="weekly", last_run=None,
               next_run=(PIN + 5 * day).isoformat())
    h = autonomy.compute_health([], list(aut.dependency_resolution_set()), 1, now=PIN)

    rows = {t["task_id"]: t for t in h["idle_tasks"]}
    assert rows["909"]["never_run"] is True and rows["910"]["never_run"] is True
    assert rows["909"]["gap_ratio"] == pytest.approx(3.0, abs=0.01)
    assert rows["910"]["gap_ratio"] == 0.0, (
        "a weekly task whose next_run is still ahead read as late")
    assert rows["909"]["hours_since_last_run"] is None, (
        "never-ran reported an elapsed time it cannot know")
    stalled = {s["task_id"] for s in h["stalled"]}
    assert stalled == {"909"}, (
        f"stalled set {sorted(stalled)} does not separate an overdue never-run "
        "daily task from a weekly task that was never due")


# ── #832: a declared output artifact is evidence for run status ───────────────
#
# Five recorded runs (2026-09-11 #38, 09-14 #57, 09-16 #39, 09-17 #39 twice)
# share one shape: the agent loop ended on a tool call, so the
# terminal assistant block was empty; run status was derived from that block
# alone; and the deliverable the run had actually written disagreed. Advancing
# `last_run` on a recorded failure never happened, so `_is_dependency_met` —
# which consumes only `last_run` — saw a dead upstream, and the dependent waited
# for a duplicate GPU run.


def _query_ending_on_a_tool_call(artifact, nbytes):
    """Stand-in agent loop: one tool call, the artifact landing on disk, then a
    turn-capped result with NO text — the shape of all five instances:
    `stop_reason: max_turns`, `empty: true`, `saw_tool_call: true`, work done.

    The artifact is written *inside* the run, between the tool call and the
    result, because the property under test is "this run wrote it": a file the
    fixture stamped before the run started is the opposite property (clause 4's
    stale case) and must be refused.
    """
    async def _rq(messages, options):
        yield {"type": "tool_call", "id": "c1", "name": "Write", "input": {}}
        if artifact is not None and nbytes:
            # The run must not be instantaneous: with `started_at ==
            # completed_at`, the assertions that `last_run` equals the run's
            # COMPLETION and that it equals its START become one assertion, and
            # a recorder that stamps the start would pass the completion test.
            # 20 ms separates the two stamps by more than the clock's grain.
            await asyncio.sleep(0.02)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"x" * nbytes)
        yield {"type": "tool_result", "call_id": "c1", "content": "written"}
        yield {"type": "result", "stop_reason": "max_turns", "num_turns": 61,
               "usage": {"output_tokens": 52579}}
    return _rq


async def _run_empty_terminal(aut, monkeypatch, tmp_path, *, task_id=39,
                              declared=None, artifact=None, nbytes=4096,
                              failure_count=0, **fm):
    """Write task `task_id`, run it to an empty terminal text, return its result.

    `declared` is the task's `output_artifact` front-matter value (None = it
    declares nothing); `artifact` the file it actually leaves behind (None =
    nothing); `nbytes` that file's size; `failure_count` the retry state the task
    is in when the run starts — seeded non-zero wherever the assertion is about
    the counter being RESET, because seeding 0 makes "it went to 0" undecidable.
    """
    monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False,
                        raising=False)
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    write_task(aut, task_id, timeout_seconds=300, failure_count=failure_count,
               **({"output_artifact": str(declared)} if declared else {}), **fm)
    monkeypatch.setattr("app.harness.run_query",
                        _query_ending_on_a_tool_call(artifact, nbytes))
    return await aut.run_task(task_id)


def _one_record(aut, task_id):
    """(front matter, whole file) for the single run record written."""
    files = list((aut.AUTONOMY_RUNS_DIR / str(task_id)).glob("run_*.md"))
    assert len(files) == 1, [p.name for p in files]
    text = files[0].read_text(encoding="utf-8")
    head, fm, body = text.split("---\n", 2)
    assert head == ""
    return yaml.safe_load(fm), body


async def test_an_empty_terminal_run_with_a_fresh_artifact_advances_last_run(
        aut, monkeypatch, tmp_path):
    """Clause 1. The 2026-09-17 replay: #39 ran 615.7 s over 61 turns, wrote its
    complete knowledge write (8,918 B), and ended with no terminal text. It was
    recorded `failed`, `last_run` stayed a day old, and the chain waited on a
    retry that happened to sign off with prose.
    """
    art = tmp_path / "artifacts" / "knowledge-write-2026-09-17.md"
    # Seeded at 2 because `2 -> 0` is a fact about the recorder while `0 -> 0`
    # is a fact about the fixture: the artifact path never increments the
    # counter, so a task starting at 0 would still read 0 with the reset deleted.
    # 2 is also the state that matters — two prior failures is two wasted GPU
    # runs spent on the same phantom, which is what this clause stops spending.
    out = await _run_empty_terminal(aut, monkeypatch, tmp_path, task_id=39,
                                    declared=art, artifact=art, nbytes=8918,
                                    failure_count=2)

    assert out["success"] is True
    task = read_task(aut, 39)
    assert task["status"] == "up_next"
    assert int(task["failure_count"]) == 0, (
        "a run that produced its deliverable must not spend the retry budget; "
        "the task came in at failure_count 2 and must leave at 0")
    # `last_run` is the run's OWN completion stamp, byte for byte — not a
    # timestamp bracketed around the call, which a wall-clock check cannot tell
    # apart from a recorder that stamped the run's start instead.
    fm, _body = _one_record(aut, 39)
    assert task["last_run"] == fm["completed_at"], (
        "the task file must carry the same completion instant the run record "
        f"does: record completed_at={fm['completed_at']!r}, "
        f"task last_run={task['last_run']!r}")
    # Control for the line above: the fixture's run has a real duration, so
    # `last_run == completed_at` is NOT also satisfied by stamping started_at.
    assert fm["started_at"] != fm["completed_at"], (
        "fixture is a zero-length run; equality with completed_at would prove "
        "nothing about a recorder that stamped the start")
    assert task["last_run"] != fm["started_at"], (
        f"last_run carries the run's start {fm['started_at']!r}, not its end")
    # The artifact the status rests on landed INSIDE that span, which is the
    # property that makes it evidence for THIS run rather than for last night's.
    modified = dt.datetime.fromisoformat(fm["output_artifact"]["modified_at"])
    assert (dt.datetime.fromisoformat(fm["started_at"]) <= modified
            <= dt.datetime.fromisoformat(fm["completed_at"])), (
        f"artifact mtime {modified} is outside the run "
        f"{fm['started_at']} .. {fm['completed_at']}")
    # Cadence resumes from the completion, exactly as on a text-confirmed run.
    last_run = dt.datetime.fromisoformat(task["last_run"])
    nxt = dt.datetime.fromisoformat(task["next_run"])
    assert (nxt - last_run).total_seconds() == pytest.approx(86400, abs=2)


async def test_the_artifact_backed_record_names_its_evidence_and_the_text_path_does_not(
        aut, monkeypatch, tmp_path):
    """Clause 2. A status decided from an artifact has to say so — the artifact,
    its size, and that the terminal text was empty — and it must stay
    distinguishable from the ordinary text-confirmed success, which gains no
    such field.
    """
    art = tmp_path / "artifacts" / "signals-latest.md"
    out = await _run_empty_terminal(aut, monkeypatch, tmp_path, task_id=38,
                                   declared=art, artifact=art, nbytes=31229)
    fm, body = _one_record(aut, 38)

    assert fm["status"] == "success"
    assert fm["empty"] is True, "the empty terminal text stays on the record"
    assert fm["status_basis"] == "artifact"
    assert out["meta"]["status_basis"] == "artifact"
    assert fm["output_artifact"]["path"] == str(art)
    assert fm["output_artifact"]["bytes"] == 31229
    assert "31229" in fm["summary"] and "signals-latest.md" in fm["summary"], fm["summary"]
    assert "signals-latest.md" in body
    assert "failure_kind" not in fm, "an artifact-backed run is not a failure row"

    # The text-confirmed path is untouched, even for a task that declares an
    # artifact: a run that says what it did needs no second witness.
    art2 = tmp_path / "artifacts" / "knowledge-handoff-2026-09-18.md"
    monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False,
                        raising=False)
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    write_task(aut, 42, timeout_seconds=300, output_artifact=str(art2))
    monkeypatch.setattr("app.harness.run_query", fake_run_query([TEXT, RESULT]))
    ok = await aut.run_task(42)
    assert ok["success"] is True and ok["status"] == "success"
    fm2, _ = _one_record(aut, 42)
    assert "status_basis" not in fm2, fm2
    assert "output_artifact" not in fm2, fm2
    assert "empty" not in fm2 or fm2["empty"] is False


async def test_the_dependent_is_due_in_the_same_cycle_as_an_artifact_backed_upstream(
        aut, monkeypatch, tmp_path):
    """Clause 3: the gate consumes `last_run` and nothing else, so the fix has to
    reach it through the recorder. #40 depends on #39 and deliberately carries no
    `stale_bypass_hours` — the pre-fix shape of #42 — so the only thing that can
    make it due is an upstream stamp. The control below is the same run with a
    stub artifact: the dependent must NOT be let through.
    """
    good = tmp_path / "artifacts" / "knowledge-write-2026-09-17.md"
    out = await _run_empty_terminal(aut, monkeypatch, tmp_path, task_id=39,
                                    declared=good, artifact=good, nbytes=8918)
    assert out["status"] == "success"
    write_task(aut, 40, depends_on=39, timeout_seconds=300)

    board = list(aut.dependency_resolution_set())
    dep = next(t for t in board if str(t["id"]) == "40")
    assert aut._is_dependency_met(dep, board) is True, (
        "an artifact-backed upstream left the dependent gated — the recorder "
        "did not advance the field the gate reads")
    assert aut._is_task_due(dep, board) is True, (
        "the dependent would have waited for #39's duplicate retry, which is "
        "the cost #832 was filed for")

    # Control: an upstream whose declared artifact is the 69-byte 09-16 stub
    # still fails, and its dependent still waits.
    stub = tmp_path / "stub" / "knowledge-write-2026-09-16.md"
    await _run_empty_terminal(aut, monkeypatch, tmp_path, task_id=41,
                              declared=stub, artifact=stub, nbytes=69)
    write_task(aut, 43, depends_on=41, timeout_seconds=300)
    board = list(aut.dependency_resolution_set())
    dep2 = next(t for t in board if str(t["id"]) == "43")
    assert aut._is_dependency_met(dep2, board) is False
    assert aut._is_task_due(dep2, board) is False


@pytest.mark.parametrize("shape", ["none-declared", "missing", "stub", "stale"])
async def test_an_empty_terminal_run_with_no_qualifying_artifact_still_fails(
        aut, monkeypatch, tmp_path, shape):
    """Clause 4 — what the gate exists to catch: the ~180 phantom successes of
    2026-09-01 and #79's dark week, where an empty window was recorded a success
    and `last_run` advanced for work that never happened. An empty-terminal run
    with nothing declared, nothing on disk, a 69-byte stub
    (`knowledge-write-2026-09-16.md` is exactly that size), or a file older than
    the run's start is still `failure_kind: task` and still leaves `last_run`
    absent, so the dependent waits for a real upstream.
    """
    declared = tmp_path / "artifacts" / "knowledge-write-2026-09-17.md"
    artifact, nbytes = declared, 4096
    if shape == "none-declared":
        declared = None
    elif shape == "missing":
        artifact = None
    elif shape == "stub":
        nbytes = 69
    elif shape == "stale":
        # Yesterday's file, 4096 bytes and complete: the run wrote nothing, the
        # mtime predates its start. Size is not the disqualifier here — age is,
        # which is why the file has to exist and be full-size.
        artifact = None
        declared.parent.mkdir(parents=True, exist_ok=True)
        declared.write_bytes(b"x" * 4096)
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=9)
        os.utime(declared, (old.timestamp(), old.timestamp()))

    out = await _run_empty_terminal(aut, monkeypatch, tmp_path, task_id=39,
                                    declared=declared, artifact=artifact,
                                    nbytes=nbytes)

    assert out["success"] is False, shape
    assert out["failure_kind"] == "task", shape
    fm, _ = _one_record(aut, 39)
    assert fm["status"] == "failed", shape
    assert fm["empty"] is True, shape
    assert "status_basis" not in fm, shape
    task = read_task(aut, 39)
    assert not task.get("last_run"), (
        f"{shape}: last_run advanced on a run with no qualifying artifact")
    assert int(task["failure_count"]) == 1, shape


def test_the_declared_artifact_macro_resolves_to_the_local_or_utc_run_date(tmp_path):
    """#39/#40/#42 declare `output_artifact` with `{date}`. #42's handoff lands at
    ~22:15 local, which is already the NEXT day in UTC, so the resolver has to try
    both dates of the run's start instant and take whichever file qualifies —
    otherwise the mechanism exists and never fires on the chain it was built for.
    """
    started = dt.datetime(2026, 9, 17, 5, 14, tzinfo=dt.timezone.utc)  # 09-16 22:14 local
    root = tmp_path / "reflection"
    root.mkdir()
    template = str(root / "knowledge-handoff-{date}.md")
    task = {"output_artifact": template}

    local_day = started.astimezone().strftime("%Y-%m-%d")
    utc_day = started.strftime("%Y-%m-%d")

    local = root / f"knowledge-handoff-{local_day}.md"
    local.write_bytes(b"y" * 24027)
    # Written at the instant the run started: qualifying, and dated by LOCAL day.
    os.utime(local, (started.timestamp(), started.timestamp()))
    ev = autonomy._declared_artifact_evidence(task, started)
    assert ev is not None
    assert ev["path"] == str(local) and ev["bytes"] == 24027

    # And the UTC-dated form qualifies on its own, for a job that names that one.
    local.unlink()
    utc = root / f"knowledge-handoff-{utc_day}.md"
    utc.write_bytes(b"z" * 30000)
    os.utime(utc, (started.timestamp() + 60,) * 2)
    ev2 = autonomy._declared_artifact_evidence(task, started)
    assert ev2 is not None and ev2["path"] == str(utc), ev2

    # No qualifying file at either date: no evidence, not a guess.
    utc.unlink()
    assert autonomy._declared_artifact_evidence(task, started) is None
    # A task that declares nothing is never artifact-backed.
    assert autonomy._declared_artifact_evidence({}, started) is None


def test_an_artifact_backed_run_is_not_counted_as_a_failure_in_health(aut):
    """The two surfaces must agree: `compute_health` reclassified every empty
    response into a failure so the 2026-09-01 phantom successes could not read as
    successes, and it reads run rows, not the task file. A run the scheduler now
    counts as delivered must not be counted as a failure by the health report —
    and every OTHER empty row, including the pre-meta_json phantoms, still is.
    """
    def row(task_id, **meta):
        return {"task_id": task_id, "status": "success", "summary": "s",
                "response_json": "text", "duration_seconds": 600.0,
                "completed_at": _iso(hours=1),
                "meta_json": json.dumps(meta) if meta else ""}

    rows = [row("39", empty=True, status_basis="artifact",
                output_artifact={"path": "/x/knowledge-write-2026-09-17.md",
                                 "bytes": 8918}),
            row("41", empty=True, failure_kind="task"),
            {"task_id": "79", "status": "success", "summary": "(No response)",
             "response_json": "(No response)", "duration_seconds": 0.6,
             "completed_at": _iso(hours=2), "meta_json": ""}]
    h = autonomy.compute_health(rows, [], 1, now=dt.datetime.now(dt.timezone.utc))
    by = {t["task_id"]: t for t in h["tasks"]}

    assert by["39"]["successes"] == 1 and by["39"]["failures"] == 0, (
        "the scheduler recorded this run as delivered; the health report "
        "calling the same row a failure is two surfaces disagreeing")
    assert by["39"]["wasted_hours"] == 0.0, (
        "a run that wrote its artifact is not GPU burned for nothing")
    assert by["39"]["empty"] == 1, (
        "`empty` stays the SHAPE signal — the terminal block really was empty — "
        "only the verdict changed")
    assert by["79"]["failures"] == 1 and by["79"]["empty"] == 1, (
        "a pre-meta_json phantom success must stay classified as a failure")
    assert by["41"]["failures"] == 1, (
        "an empty row with no `status_basis` is untouched: 2026-09-18's "
        "empty-terminal failure still counts as one")


async def test_the_artifact_backed_result_survives_the_pool_status_boundary(
        aut, monkeypatch, tmp_path):
    """The status string is a process boundary, not a local label. `run_task`'s
    dict reaches `workers/pool.normalize_result` (`workers/sources/
    scheduled_task.py:413`, then the pool's own call), which validates `status`
    against the closed set `RUN_STATUSES` and, for anything outside it, logs
    "source scheduled-task returned unknown status" and rewrites the value
    (`pool.py:135-141`). An invented `"artifact_success"` would therefore have
    been renamed on its way into `workers/queue.runs`, leaving one run carrying
    one status in `autonomy-runs/` and a different one in the queue table, and
    logging a warning about a source that was reporting correctly. The status has
    to arrive already valid; the basis rides along in `meta`, which
    `normalize_result` copies verbatim. So the real pool function gets the real
    result, not my summary of it.
    """
    from types import SimpleNamespace
    from workers.pool import RUN_STATUSES, normalize_result

    art = tmp_path / "artifacts" / "signals-latest.md"
    out = await _run_empty_terminal(aut, monkeypatch, tmp_path, task_id=38,
                                    declared=art, artifact=art, nbytes=31229)
    assert out["meta"]["status_basis"] == "artifact"
    assert out["status"] in RUN_STATUSES, (
        f"status {out['status']!r} is outside the pool's closed set, so the pool "
        "would rewrite it and the two tables would disagree about this run")

    item = SimpleNamespace(source="scheduled-task", kind="task",
                           payload={"task_id": "38"})
    norm = normalize_result(item, out)
    assert norm["status"] == "success", norm["status"]
    assert norm["meta"]["status_basis"] == "artifact", (
        "normalize_result copies meta, so the basis has to survive into the row")


async def test_the_artifact_evidence_survives_the_queue_meta_json_round_trip(
        aut, monkeypatch, tmp_path):
    """The other boundary: `status_basis` and the NESTED `output_artifact` dict
    are the only things the health exemption keys on, and they reach the health
    report as a string. `pool.py:683` writes `json.dumps(meta, default=str)`,
    `queue.list_runs_joined` returns that column, and `compute_health`
    `json.loads` it (`autonomy.py:1904`). A nested dict serialising to a non-dict on
    the far side would silently drop the exemption and reclassify the run back to
    a failure, so the round trip is run for real with the bytes the recorder
    actually produced.
    """
    art = tmp_path / "artifacts" / "knowledge-write-2026-09-18.md"
    out = await _run_empty_terminal(aut, monkeypatch, tmp_path, task_id=39,
                                    declared=art, artifact=art, nbytes=9000)

    blob = json.dumps(out["meta"], default=str)        # pool.py:683
    meta = json.loads(blob)                            # autonomy.py:1904
    assert isinstance(meta.get("output_artifact"), dict), blob
    assert meta["output_artifact"]["bytes"] == 9000, meta
    assert meta["status_basis"] == "artifact"

    row = {"task_id": "39", "status": "success", "summary": "s",
           "response_json": "", "duration_seconds": 600.0,
           "completed_at": _iso(hours=1), "meta_json": blob}
    h = autonomy.compute_health([row], [], 1, now=dt.datetime.now(dt.timezone.utc))
    t = {x["task_id"]: x for x in h["tasks"]}["39"]
    assert t["failures"] == 0 and t["successes"] == 1, (
        "the exemption died in serialisation: the row reads as a failure again")


async def test_the_artifact_evidence_survives_the_real_execute_adapter(
        aut, monkeypatch, tmp_path):
    """Clause 2, across the boundary a hand-typed dict cannot reach.

    `workers/sources/scheduled_task.execute()` (line 377) builds the dict the
    queue stores from a HAND-WRITTEN whitelist — `status`, `summary`, `task_id`,
    `artifact_path`, `response`, `meta`, and `claims` only when the result has it
    — and that literal is exactly where the evidence pilot's `claims` was severed
    for 4,922 runs (#945, whose own test file says the same thing at
    tests/test_worker_evidence.py:456-460). `workers/pool.normalize_result` is the
    hop AFTER it, so a test that hands `normalize_result` a dict shaped like
    `execute()`'s output asserts on its own input and never touches the whitelist.

    So this drives the real `execute()` on a real `run_task` result — the queue
    row's `meta_json`, which is the surface a dashboard filtering on
    `status_basis` actually reads, is written from that dict and from nothing
    else. Stubbed: the model loop, the vllm probe, the Discord notifier, and the
    second `run_task` call (its own record is already on disk from this test's
    run, so re-running would add a file for no claim). Everything between
    `execute()` and `run_task` is the shipped code.
    """
    import app.discord_notify as discord_notify
    import workers.sources.scheduled_task as scheduled_task
    from workers.queue import QueueItem

    art = tmp_path / "artifacts" / "knowledge-handoff-2026-09-19.md"
    real = await _run_empty_terminal(aut, monkeypatch, tmp_path, task_id=39,
                                     declared=art, artifact=art, nbytes=30237)
    assert real["meta"]["status_basis"] == "artifact", real["meta"]

    async def _already_ran(task_id, **kw):
        return real
    monkeypatch.setattr(autonomy, "run_task", _already_ran)
    monkeypatch.setattr(scheduled_task, "_vllm_healthy", lambda *a, **k: True)

    async def _no_notify(*_a, **_k):
        return None
    monkeypatch.setattr(discord_notify, "_discord_notify_task_complete", _no_notify)

    item = QueueItem(id=1, source="scheduled-task", kind="autonomy", priority=50,
                     payload={"task_id": "39"}, dedup_key=None, state="running",
                     attempts=1, enqueued_at="", claimed_at=None, claimed_by=None,
                     completed_at=None, error=None)
    out = await scheduled_task.execute(item)

    assert out["status"] == "success", out
    # Clause 2's two distinguishing fields survive the whitelist, with the byte
    # count and not just the path: that number is half the evidence.
    assert out["meta"]["status_basis"] == "artifact"
    assert out["meta"]["output_artifact"] == real["meta"]["output_artifact"]
    assert out["meta"]["output_artifact"]["bytes"] == 30237
    assert out["meta"]["empty"] is True
    assert out["task_id"] == "39"
    assert out["artifact_path"].endswith(
        f"autonomy-runs/39/{real['run_id']}.md"), out["artifact_path"]
    # #39 is in EVIDENCE_PILOT_TASK_IDS and an empty-terminal run emits no
    # claims, so `execute()` must still COPY the key: its presence is the pool's
    # pilot-scope switch, and dropping it here would repeat #945 for precisely the
    # runs this change makes newly classifiable.
    assert "claims" in out, "execute() dropped the pilot's claims key again"
    assert out["claims"] == []
    # And through the NEXT hop as well, on the object the adapter really produced
    # rather than one typed to match it: `meta` is unwhitelisted there.
    from workers.pool import normalize_result
    norm = normalize_result(item, out)
    assert norm["status"] == "success"
    assert norm["meta"]["output_artifact"]["bytes"] == 30237


@pytest.mark.asyncio
async def test_a_parked_upstream_keeps_its_dependent_out_of_the_queue(
        aut, monkeypatch, tmp_path):
    """A parked upstream means the dependent is never ENQUEUED, not enqueued-then-skipped.

    #558's hold is asserted against `get_due_tasks` in
    tests/test_autonomy_dependency_fail_closed.py. This pins the trip THROUGH the
    worker pool, which is where a task is actually stopped: `enqueue_if_due`
    computes due-ness on the pool's default executor — a different thread, with
    `autonomy` imported inside the worker module rather than this test's — and
    then applies its own model-health and open-item filters before `enqueue`. A
    dependent of a parked upstream has to be absent from that loop entirely: a
    source that enqueued first and second-guessed later would still take a worker
    slot and still write a run record, which is the 2026-09-08 shape exactly
    (#39/#40 ran at 06:00Z while #42 sat paused; #42's handoff landed at 06:10Z).
    The queue mouth is where the fix has to be seen biting, not the board.

    The health probe is stubbed so the tick reaches the enqueue loop; the real
    `WorkQueue` is used so `enqueue` is really called. The control half at the end
    is what turns the empty first half into a verdict rather than an artifact of a
    tick that never got that far.
    """
    import workers.sources.scheduled_task as st
    from workers.queue import WorkQueue

    enqueued: list[str] = []

    def _capture(_self, **kw):
        # `enqueue_if_due` calls it all-keyword: source/kind/payload/priority/
        # dedup_key, with the task id in payload. Reading the dedup key rather
        # than a positional keeps this honest about the real call shape.
        enqueued.append(str(kw["dedup_key"]).rsplit(":", 1)[-1])
        return 1

    monkeypatch.setattr(st.WorkQueue, "enqueue", _capture)
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(autonomy, "_local_hour", lambda: 2)

    # The upstream's last run sits INSIDE the freshness bound (6 h of a daily
    # interval's 12 h half), so the only thing holding #2 is its status. A stale
    # upstream would hold the dependent through the freshness rule instead, and
    # the test would be pinning the wrong mechanism.
    write_task(aut, 1, status="paused", last_run=_iso(hours=6))
    write_task(aut, 2, depends_on=1, preferred_hours=[2], last_run=_iso(days=3))

    await st.enqueue_if_due(WorkQueue(tmp_path / "parked.db"),
                            {"max_duration_seconds": 1800})
    held = list(enqueued)
    assert "2" not in held, (
        f"queue was handed {held}: a dependent of a paused upstream reached the "
        "worker pool. The hold has to hold at the enqueue mouth and not only on "
        "the board — that is where a slot is spent and a run record written.")

    # Control, same harness, same tick: flip ONLY the upstream status, and #2 must
    # now be enqueued. Without it, the empty list above could equally mean the
    # tick never reached the enqueue loop (health gate, open-item cap, a fixture
    # that is simply not due) — the false negative this file exists to prevent.
    enqueued.clear()
    write_task(aut, 1, status="up_next", last_run=_iso(hours=6))
    await st.enqueue_if_due(WorkQueue(tmp_path / "unparked.db"),
                            {"max_duration_seconds": 1800})
    assert "2" in enqueued, (
        f"nothing enqueued even with a runnable upstream ({enqueued}): the fixture "
        "was never due, so the assertion above proves nothing")


# ── #938: a model-server outage pauses dispatch, it does not pause watching ───
#
# `enqueue_if_due` probed the primary model server first and `return`ed on a
# failure, which put the vLLM gate ABOVE both stall detectors. With the engine
# down, `_grossly_overdue`, `_queue_starving` and `_next_run_stalled` were each
# called ZERO times — measured over 8 consecutive ticks on 2026-09-20 — with 0
# `_alert` calls and exactly one `logger.warning` line total, deduped for the
# whole outage by `_state["vllm_down_logged"]`. No other watcher covered the
# gap: `agent-services/guardian/policy.py` WATCHED names only
# `lloyd-backend`/`lloyd-mcp`, and the vault tasks that do probe the model
# server are autonomy tasks dispatched through this very gate, so the monitor
# died with the thing it monitors. At `tick_interval: 60` a 3-day outage is
# ~4300 ticks and one log line.
#
# Every other stall test in this file forces the gate OPEN
# (`_vllm_healthy` → True), so the outage path had no coverage and the suite
# could not see the detectors going un-called. These tests are the inverse: the
# gate is CLOSED, and the assertion is that the detectors are still CALLED.

def _outage_fleet(aut):
    """Four `daily` tasks, read at the pinned `PIN`, that tell the four shapes
    apart with an outage in progress. Ids in the 900s so nothing collides.

    912 is the control and the only one worth an alert: `up_next`,
    dependency-free, 5 days since `last_run` against a 1-day interval — past
    the 2.5x bound — 5 days past its own `next_run`, and no queue row. Dispatch
    had a job to do here and did not do it, which is the shape the noisy alarm
    was built for and still must return with the model server down.
    913 is the healthy on-cadence task: `up_next`, 2 days since `last_run`
    against a 1-day interval, so it IS due (past the half-interval freshness
    bound) and yet inside the 2.5x multiple, and its `next_run` is exactly one
    period behind `last_run`, so it is exactly one period past it and the strict
    `> one interval` bound leaves it alone too. It is excluded by the multiple
    and by the period, by nothing else — an outage must not make a task that ran
    on schedule look stalled.
    914 is `draft` and 5 days stale — grossly overdue in every other respect,
    and excluded by status, because a deliberate stop is not a stall.
    915 is `up_next` and 5 days stale and gets a live queue row from the caller:
    starved for capacity, not stalled, and excluded by the queue-row rule.
    """
    day = dt.timedelta(days=1)
    write_task(aut, 912, last_run=(PIN - 5 * day).isoformat(),
               next_run=(PIN - 5 * day).isoformat())
    write_task(aut, 913,
               last_run=(PIN - 2 * day).isoformat(),
               next_run=(PIN - day).isoformat())
    write_task(aut, 914, status="draft",
               last_run=(PIN - 5 * day).isoformat(),
               next_run=(PIN - 5 * day).isoformat())
    write_task(aut, 915, last_run=(PIN - 5 * day).isoformat(),
               next_run=(PIN - 5 * day).isoformat())


def _seed_queue_row(q, task_id):
    """A live `queued` row for one task, fresh enough that `_queue_starving`
    reads 0.0 and cannot add its own clause to the alert text."""
    q.enqueue(source="scheduled-task", kind="run",
              payload={"task_id": task_id, "name": f"task{task_id}"},
              priority=30, dedup_key=f"scheduled-task:{task_id}")


def _clean_outage_state(monkeypatch, st, **over):
    """Swap in a `_state` with both stall streaks and all outage accounting at
    zero, so no tick here inherits another test's streak, cooldown or outage, and
    nothing here leaks into the module copy afterwards."""
    fresh = {**st._state, "startup_checked": True,
             "stall_streak": 0, "stall_alerted_at": None,
             "nextrun_streak": 0, "nextrun_alerted_at": None,
             "vllm_down_logged": False, "vllm_down_since": None,
             "vllm_down_alerted": False}
    fresh.update(over)
    monkeypatch.setattr(st, "_state", fresh)
    return fresh


async def test_a_closed_health_gate_still_evaluates_every_stall_detector(
        aut, monkeypatch, tmp_path):
    """Clause 1, across the executor seam: ONE tick with the model server down.

    The health probe is the only thing stubbed besides the alert, so the tick is
    the real coroutine, the real `WorkQueue`, the real detector bodies run on
    executor threads. Each detector is wrapped rather than replaced, so the call
    count is the assertion and the returned verdict is still the real one. Pre-fix
    this fails on the first assertion: the gate returned before all three.
    """
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    _pin(aut, monkeypatch)
    _outage_fleet(aut)

    q = WorkQueue(tmp_path / "closed-gate.db")
    _seed_queue_row(q, 915)

    calls = {"grossly": 0, "starving": 0, "nextrun": 0}
    real = (st._grossly_overdue, st._queue_starving, st._next_run_stalled)

    def _spy_grossly(queue):
        calls["grossly"] += 1
        return real[0](queue)

    def _spy_starving(queue, max_duration):
        calls["starving"] += 1
        return real[1](queue, max_duration)

    def _spy_nextrun(queue):
        calls["nextrun"] += 1
        return real[2](queue)

    monkeypatch.setattr(st, "_grossly_overdue", _spy_grossly)
    monkeypatch.setattr(st, "_queue_starving", _spy_starving)
    monkeypatch.setattr(st, "_next_run_stalled", _spy_nextrun)
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: False)
    _clean_outage_state(monkeypatch, st)
    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _capture)

    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})

    assert calls["grossly"] >= 1, (
        "the due-ness stall detector was never called on a tick with the model "
        "server down: the outage silenced the alarm built for the 2026-05-28 "
        "silent stall, which is the whole defect")
    assert calls["nextrun"] >= 1, (
        "the next_run stall detector was never called on a tick with the model "
        "server down — it is the quieter alarm built for the 41 h and 60 h "
        "stalls, so its blindness is the costly half")
    assert calls["starving"] >= 1, (
        "`_queue_starving` was never called, so a dead worker pool is invisible "
        "for the length of an outage")
    # The verdicts themselves still come from the real detectors, on the pinned
    # instant, with the gate shut: 912 grossly overdue, and nothing new reached
    # the queue.
    assert real[0](q) == [912], "the control fleet stopped being the noisy alarm's case"
    assert {str(i.payload.get("task_id")) for i in
            q.list_items(source="scheduled-task", limit=500)} == {"915"}, (
        "a closed health gate still enqueued work; clause 4 is about exactly this")
    # One tick is 0 s of outage, so nothing is due an alert yet: neither a stall
    # streak of 1 nor the transition itself may shout.
    assert alerts == [], f"the first down tick alerted: {alerts}"


async def test_the_stall_alarm_alerts_through_an_outage_and_keeps_its_exclusions(
        aut, monkeypatch, tmp_path):
    """Clauses 2 and 3: the noisy alarm fires with the engine DOWN, and the three
    exclusions survive the move.

    `_STALL_NEXTRUN_TICKS` equals `_STALL_ALARM_TICKS`, so on tick 5 the
    next_run alarm would otherwise alert on the same tick and the count would be
    about two alarms. Its cooldown is seeded as just-fired — its real clock, not
    the pinned one, since that alarm measures wall time — which is the same
    state production is in whenever it has alerted in the last day. What is
    measured here is the alarm #938 moved, and the assertion below that no
    message mentions `next_run` is what proves the count stayed about one.
    """
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    _pin(aut, monkeypatch)
    _outage_fleet(aut)

    q = WorkQueue(tmp_path / "outage-stall.db")
    _seed_queue_row(q, 915)

    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: False)
    _clean_outage_state(
        monkeypatch, st,
        nextrun_alerted_at=dt.datetime.now(dt.timezone.utc))
    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _capture)

    ticks = st._STALL_ALARM_TICKS
    for _ in range(ticks - 1):
        await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert alerts == [], (
        f"alerted on tick {len(alerts)} of {ticks}: the moved alarm stopped "
        "confirming across consecutive ticks")
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})

    stalled_alerts = [m for m in alerts if "autonomy scheduler may be stalled" in m]
    assert len(stalled_alerts) == 1, (
        f"no stall alert after {ticks} ticks with the model server down: {alerts}")
    assert len(alerts) == 1, f"a second alert came in on the same tick: {alerts}"
    assert all("next_run" not in m for m in alerts), (
        "the count above is not about one alarm: " + repr(alerts))

    msg = stalled_alerts[0]
    # Clause 2: the id dispatch skipped is named.
    assert "912" in msg, f"the grossly overdue task is not named: {msg}"
    # Clause 3, as the alarm's own three exclusions.
    assert "913" not in msg, (
        f"flagged a task inside {st._STALL_INTERVAL_MULT}x its interval: {msg}")
    assert "914" not in msg, f"flagged the deliberately stopped task: {msg}"
    assert "915" not in msg, f"flagged the task with a live queue row: {msg}"
    board = list(aut.dependency_resolution_set())
    t913 = next(t for t in board if int(t["id"]) == 913)
    assert aut._is_task_due(t913, board, now=PIN) is True, (
        "the fixture is no longer a due-but-within-2.5x case, so the exclusion "
        "above is not testing the multiple")
    assert "915" in {str(t) for t in st._active_task_ids(q)}, (
        "the queue-row fixture lost its live row, so that exclusion tested nothing")
    assert st._grossly_overdue(q) == [912], (
        "the detector's own exclusions drifted from the message's")

    # And it stays one alert: the 6 h cooldown is load-bearing, since this alarm
    # alerted 100 times in 6 days before it was added.
    for _ in range(ticks):
        await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert len(alerts) == 1, (
        f"re-alerted inside the cooldown: {len(alerts)} alerts — that noise is "
        "why the cooldown exists")


async def test_dispatch_stays_paused_while_the_outage_is_being_watched(
        aut, monkeypatch, tmp_path):
    """Clause 4: with the model server unhealthy, `enqueue_if_due` enqueues
    NOTHING for a due task, so the gate keeps its skip-rather-than-retry-loop
    purpose. The control on the next tick flips only the health probe, which is
    what turns the empty list into a verdict instead of an artifact of a tick
    that never reached the enqueue loop.
    """
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    _pin(aut, monkeypatch)
    day = dt.timedelta(days=1)
    write_task(aut, 912, last_run=(PIN - 3 * day).isoformat(),
               next_run=(PIN - 2 * day).isoformat())

    enqueued: list[str] = []

    def _capture(_self, **kw):
        enqueued.append(str(kw["payload"]["task_id"]))
        return 1

    monkeypatch.setattr(st.WorkQueue, "enqueue", _capture)
    health = {"ok": False}
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: health["ok"])
    _clean_outage_state(monkeypatch, st)
    alerts: list[str] = []

    async def _capture_alert(msg):
        alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _capture_alert)

    q = WorkQueue(tmp_path / "paused.db")
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert enqueued == [], (
        f"the queue was handed {enqueued} with the model server down: the gate "
        "moved below the detectors must still stop dispatch, or every due task "
        "becomes a ConnectError retry loop")

    health["ok"] = True
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert enqueued == ["912"], (
        f"control tick enqueued {enqueued} with the model server healthy: the "
        "fixture is not the shape this clause is about, so the empty list above "
        "says nothing about the gate. Nothing but the probe result differs "
        "between the two ticks, and the assertion below says so in dispatch's "
        "own words.")
    board = list(aut.dependency_resolution_set())
    t912 = next(t for t in board if int(t["id"]) == 912)
    assert aut.hold_reason(t912, board, now=PIN) is None, (
        "the control's meaning is inverted: `hold_reason` says something holds "
        "this task, so an empty queue on the first tick would be correct "
        "behaviour and clause 4 would be vacuous")
    assert alerts == [], (
        f"recovery alerted: {alerts} — only a sustained outage is an alert")


async def test_a_sustained_outage_alerts_once_and_a_later_outage_alerts_again(
        aut, monkeypatch, tmp_path):
    """Clause 5, on an empty board so the only possible alert is the outage's.

    The elapsed time is driven through `_state["vllm_down_since"]` rather than by
    sleeping: the quantity under test is "how long has this uninterrupted run of
    unhealthy ticks been going", and the tick that crosses the constant is the
    tick that must alert. The constant itself is asserted inside the contract
    band the clause states, so a threshold drift fails here and not in review.
    """
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    assert 30 * 60 <= st._VLLM_DOWN_ALERT_SECONDS <= 6 * 3600, (
        f"outage alert threshold is {st._VLLM_DOWN_ALERT_SECONDS}s, outside the "
        "30 min to 6 h the clause allows")

    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: False)
    _clean_outage_state(monkeypatch, st)
    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _capture)
    q = WorkQueue(tmp_path / "outage-alert.db")
    threshold = st._VLLM_DOWN_ALERT_SECONDS

    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert alerts == [], "the first unhealthy tick alerted, before any downtime elapsed"
    assert st._state["vllm_down_since"] is not None, (
        "the outage has no start instant, so its duration can never be measured")

    # Cross the threshold on the next tick: the outage began before it.
    st._state["vllm_down_since"] = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=threshold + 120))
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert len(alerts) == 1, f"no outage alert past {threshold}s: {alerts}"
    msg = alerts[0]
    assert "model server" in msg and st._VLLM_HEALTH_URL in msg, (
        f"the outage alert does not name the model server: {msg}")

    for _ in range(3):
        await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert len(alerts) == 1, (
        f"a 3-day outage at `tick_interval: 60` would produce {len(alerts)} "
        "alerts — one per outage is the contract")

    # Recovery clears the accounting, so the NEXT outage is its own incident.
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert len(alerts) == 1, f"recovery alerted: {alerts}"
    assert st._state["vllm_down_since"] is None and (
        st._state["vllm_down_alerted"] is False), (
        "recovery left the outage accounting behind, so a later outage inherits "
        "this one's silence")

    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: False)
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert len(alerts) == 1, "a fresh outage alerted immediately"
    st._state["vllm_down_since"] = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=threshold + 120))
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert len(alerts) == 2, (
        f"the second outage never alerted: {alerts} — a fleet that has been "
        "through one outage must not be deaf to the next")


# ── The run record vetoes a second dispatch of a period that already ran ──────
# #1296. An arch-review stray sweep reverted task #82's completion stamp 48 s
# after the run finished, so `last_run` stayed on 2026-09-19, elapsed read a full
# 24 h on the next tick, and the same nightly job dispatched again 27 s after it
# finished. Every gate that existed read the task file, and the task file is in
# the vault — the thing that got rewritten. These tests pin the one verdict that
# consults a second, gitignored account: the run record.

STALE_19 = "2026-09-19T13:07:31.589361+00:00"
RUN_20_DONE = "2026-09-20T13:07:04.862230+00:00"
REDISPATCH_20 = dt.datetime(2026, 9, 20, 13, 8, 17, 549634, tzinfo=dt.timezone.utc)


def _record(aut, task_id, run_id, *, status, completed_at, started_at=None):
    """A run record on disk, in the exact shape `run_task` writes: `---`
    delimited front matter then the body. Without the delimiters the loader
    under test never sees front matter, so the fixture would test nothing."""
    d = aut.AUTONOMY_RUNS_DIR / str(task_id)
    d.mkdir(parents=True, exist_ok=True)
    fm = {"run_id": run_id, "task_id": task_id, "status": status,
          "completed_at": completed_at}
    if started_at:
        fm["started_at"] = started_at
    (d / f"{run_id}.md").write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\nreport\n",
        encoding="utf-8")


def test_a_stale_last_run_cannot_re_dispatch_a_task_the_record_says_succeeded(
        aut, monkeypatch):
    """The 2026-09-20 window for task #82, with the timestamps as recorded.

    `last_run`/`next_run` are the values the vault held after the sweep reverted
    the stamp; the record's times are `run_82_20260920_130244.md`'s own; `now` is
    the instant queue row 8514 was enqueued. At that instant elapsed-since-
    `last_run` is 86,446 s against an 82,800 s interval minus the 3,600 s window
    slack, so the elapsed gate calls it due and only the record can refuse.

    The second half is the clause's other edge: 23 h 59 m later the same record
    must NOT hold it. A guard that kept refusing would be a worse failure than
    the duplicate it prevents.
    """
    _pin(aut, monkeypatch, REDISPATCH_20)
    monkeypatch.setattr(aut, "_local_hour", lambda: 6)  # hour 06 local, UTC-7
    write_task(aut, 82, name="Nightly Retrieval Eval", frequency="daily",
               skill_name="retrieval-eval", status="up_next",
               preferred_hours=[6], last_run=STALE_19, next_run=STALE_19)
    task = read_task(aut, 82)
    _record(aut, 82, "run_82_20260920_130244", status="success",
            completed_at=RUN_20_DONE, started_at="2026-09-20T13:02:44.476948+00:00")

    assert aut._already_ran_this_period(task, now=REDISPATCH_20) == "run_82_20260920_130244"
    assert aut._is_task_due(task, [task], now=REDISPATCH_20) is False
    # And the board says why, in the same vocabulary. A reverted stamp with the
    # window open is otherwise indistinguishable from a scheduler that silently
    # skipped a night.
    assert aut.hold_reason(task, [task], now=REDISPATCH_20) == aut.RUN_SUCCESS_HOLD

    # Expiry: past the guard's hold, the same never-repaired file must dispatch.
    later = REDISPATCH_20 + dt.timedelta(hours=23, minutes=59)
    assert aut._already_ran_this_period(task, now=later) == ""
    assert aut._is_task_due(task, [task], now=later) is True


def test_the_guard_holds_exactly_as_long_as_the_intact_stamp_would(
        aut, monkeypatch):
    """The guard must substitute for the lost stamp and never delay past it.

    A run that COMPLETED at S makes the elapsed gate grant due-ness at
    S + interval - slack (the slack the windowed task is given, because
    completion time drifts the cycle later every night otherwise). The guard
    refuses while `now < S + interval - slack` — the same instant, to the second.
    Two ticks either side of it: refused at 13:08:17Z on the 20th, due at
    12:08:17Z on the 21st — one second past S + (86,400 − 3,600) =
    2026-09-21T12:07:04.862230Z — and the healthy path would have dispatched at
    that same moment. That identity is what makes the guard safe to add to
    every dispatch path; a period anchored on `last_run` or `next_run` instead —
    both vault writes, both reverted by this very defect — would hold a task
    inside one stale period forever and silence it outright.
    """
    _pin(aut, monkeypatch, REDISPATCH_20)
    monkeypatch.setattr(aut, "_local_hour", lambda: 6)
    write_task(aut, 76, name="Queue Health Check", frequency="daily",
               skill_name="queue-health-check", status="up_next",
               preferred_hours=[6], last_run=STALE_19)
    task = read_task(aut, 76)
    _record(aut, 76, "run_76_20260920_130244", status="success",
            completed_at=RUN_20_DONE)

    assert aut._is_task_due(task, [task], now=REDISPATCH_20) is False

    # S + (interval - slack) = 2026-09-20T13:07:04.862230Z + 82,800 s =
    # 2026-09-21T12:07:04.862230Z. The first tick strictly after it dispatches.
    past_hold = dt.datetime(2026, 9, 21, 12, 8, 17, tzinfo=dt.timezone.utc)
    assert aut._is_task_due(task, [task], now=past_hold) is True


def test_a_failed_run_record_does_not_refuse_the_next_dispatch(aut, monkeypatch):
    """The guard reads `status`, not the existence of a record.

    Same instant, same stale `last_run`, one field different: a failed run stays
    eligible, or the guard would silently retire every task that had one bad
    night. `last_failure_at` is absent, so the failure-cooldown gate is not what
    keeps it eligible either.
    """
    _pin(aut, monkeypatch, REDISPATCH_20)
    monkeypatch.setattr(aut, "_local_hour", lambda: 6)
    write_task(aut, 82, frequency="daily", skill_name="retrieval-eval",
               status="up_next", preferred_hours=[6], last_run=STALE_19)
    task = read_task(aut, 82)
    _record(aut, 82, "run_82_20260920_130244", status="failed",
            completed_at=RUN_20_DONE)

    assert aut._already_ran_this_period(task, now=REDISPATCH_20) == ""
    assert aut._is_task_due(task, [task], now=REDISPATCH_20) is True
    assert aut.hold_reason(task, [task], now=REDISPATCH_20) is None


def test_a_task_with_no_declared_period_is_never_vetoed_by_a_record(aut, monkeypatch):
    """No `frequency`, no `runs_per_day` → no defensible window → no veto.

    A success of unknown age would otherwise silence the job forever, which is a
    worse failure than the duplicate. Such a task is not due either — the cadence
    gate refuses it first (`hold_reason` answers "no frequency") — so what this
    pins is that the guard is inert, not load-bearing, in that shape.
    """
    _pin(aut, monkeypatch, REDISPATCH_20)
    write_task(aut, 55, skill_name="some-skill", status="up_next",
               frequency=None)   # `write_task` drops keys whose value is None
    task = read_task(aut, 55)
    _record(aut, 55, "run_55_20260920_120000", status="success",
            completed_at="2026-09-20T12:00:10+00:00")

    assert aut._run_period_start(task, now=REDISPATCH_20) is None
    assert aut._already_ran_this_period(task, now=REDISPATCH_20) == ""


@pytest.mark.asyncio
async def test_a_succeeded_run_cannot_be_enqueued_twice_in_one_night(
        aut, tmp_path, monkeypatch):
    """The harm the item predicted, refused by the DISPATCHER (clause 4).

    Task #82 writes one `eval/baselines/nightly-<date>-*.json` per night.
    `scripts/eval_trend_stats.load_window` returns one `Night` per FILE — sorted by
    each file's own `ran_at`, never deduplicated by date — and the caller pairs
    consecutive rows, so a second file for one night is accepted by `join_ids`
    (same query ids) and becomes a night→itself transition in the series, while
    the row count that stands for "how many nights" overstates by one. On
    2026-09-20 only judgment stopped that: the second dispatch spent 259 s and
    2,254,592 tokens of a primary engine turn to conclude "this is a duplicate
    dispatch, I won't re-run", and the fleet default is that a run follows its
    skill.

    The instant is the real clock, not a pinned one, so this holds whatever `TZ`
    pytest runs under. The run itself is the real `autonomy.run_task` with
    `app.harness.run_query` patched to the canned `stop` stream — the same call
    `scheduled_task.execute` makes — so both accounts the guard reads are written
    by production code: the vault stamp AND the run record. Nothing here touches
    `eval/baselines/`; the only writer of a second baseline that has to be
    stopped is the dispatcher handing the job a second turn.
    """
    import workers.sources.scheduled_task as st
    from workers.queue import WorkQueue

    q = WorkQueue(tmp_path / "workers.db")
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    # The stall alarms live in this module and notify through autonomy; the
    # queue rows are what this test reads, and no Discord message is owed here.
    monkeypatch.setattr(aut, "_notify_alert_channel", lambda *a, **k: None)
    monkeypatch.setattr("app.harness.run_query", fake_run_query([TEXT, RESULT]))
    write_task(aut, 102, name="Nightly Retrieval Eval", frequency="daily",
               skill_name="retrieval-eval", status="up_next",
               timeout_seconds=120)

    async def tick():
        """One scheduler tick, then close what it enqueued.

        Draining matters: an item left in `queued` would dedup-collide on
        `scheduled-task:102` and the NEXT tick would enqueue nothing for a reason
        that has nothing to do with the guard — the test would pass on plumbing.
        """
        await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
        rows = [i for i in q.list_items(source="scheduled-task", limit=500)
                if i.state == "queued" and i.payload.get("task_id") == 102]
        for i in rows:
            q.mark_completed(i.id)
        return rows

    # Tick 1: no run record, no stamp — the first run must be dispatched.
    assert len(await tick()) == 1, (
        "a first-ever run must still be dispatched — the guard's subject is the "
        "second one, not the first")
    result = await aut.run_task(102)
    assert result["success"] is True
    assert read_task(aut, 102)["last_run"], "the success stamped the task file"
    assert len(list((aut.AUTONOMY_RUNS_DIR / "102").glob("run_*.md"))) == 1

    # Tick 2: the stamp survived, so the elapsed gate alone refuses.
    assert await tick() == []

    # Tick 3: the stamp is gone — what an arch-review stray sweep actually
    # leaves behind. The run record is now the only thing between this task and a
    # second `nightly-<date>-*.json` for the same night.
    stripped = dict(read_task(aut, 102))
    for key in ("last_run", "last_attempt", "next_run"):
        stripped.pop(key, None)
    (aut.AUTONOMY_DIR / "102-task102.md").write_text(
        f"---\n{yaml.dump(stripped, default_flow_style=False)}---\n"
        f"{stripped.get('body', '')}", encoding="utf-8")
    board = [read_task(aut, 102)]

    assert await tick() == [], (
        "a reverted `last_run` re-dispatched a task whose run record reports "
        "`status: success` — that second turn is what would have written the "
        "second same-night baseline")
    assert board[0].get("last_run") is None, "the file is still the reverted one"
    assert aut.hold_reason(board[0], board) == aut.RUN_SUCCESS_HOLD, (
        "the board must name the veto, or the skipped dispatch reads as a "
        "scheduler that silently dropped the job")
    assert len(list((aut.AUTONOMY_RUNS_DIR / "102").glob("run_*.md"))) == 1, (
        "one run record, one night")


# ── `model:` validated against the configured slots (#1209) ──────────────────
#
# Task #85 dispatched on 2026-09-16 carrying `model: eco`, an alias
# `config.yaml` never defined, and burned three sub-second vLLM 404s before the
# row disabled itself. Every one of those refusals was avoidable with one
# membership test against `models:` — and the three that were spent were spent
# because the 404 was booked as a task failure, so the check below refuses
# BEFORE any engine call and BEFORE `failure_count` moves.

_CONFIGURED_MODELS = {"primary": {"alias": "primary"},
                      "secondary": {"alias": "secondary"}}


def _stub_engine(monkeypatch):
    """Replace the engine with a recorder; return the list of attempted calls.

    `MODEL_CONFIGS` is pinned to two slots so the refusal text names
    `primary, secondary` as a property of this fixture and not of whatever the
    live config.yaml happens to define the day someone adds a third model.
    """
    calls = []

    async def _rq(messages, options):
        calls.append(options)
        yield TEXT
        yield RESULT

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _rq)
    monkeypatch.setattr("app.config.MODEL_CONFIGS", _CONFIGURED_MODELS, raising=False)
    return calls


def _stub_sessions(monkeypatch):
    created = []
    monkeypatch.setattr("app.sessions_io.create_session",
                        lambda sid, **_kw: created.append(sid), raising=False)
    return created


async def test_a_model_that_is_not_a_configured_slot_refuses_before_the_engine(aut, monkeypatch):
    """Clause 1: `model: eco` cannot dispatch, and the refusal says why.

    The proof of "before the engine" is that the engine was never reached — the
    recorder stays empty and no run record exists, so there is no vLLM 404
    artifact for the next reader to find, which is what the acceptance check
    asks must stop happening. `create_session` is asserted too: a dispatch that
    was refused must not leave a phantom session row behind either.
    """
    calls = _stub_engine(monkeypatch)
    sessions = _stub_sessions(monkeypatch)
    write_task(aut, 1, model="eco")

    result = await aut.run_task(1)

    assert result["success"] is False
    text = result["error"]
    assert "eco" in text, f"the refusal must name the offending value, got: {text}"
    assert "primary" in text and "secondary" in text, (
        f"the refusal must name the slots that DO exist, got: {text}")
    assert calls == [], "a refused dispatch must not reach the engine at all"
    assert sessions == [], "a refused dispatch must not create a session either"
    assert list(aut.AUTONOMY_RUNS_DIR.rglob("run_*.md")) == [], (
        "a refused dispatch writes no run record — the #1209 artifact was a "
        "404 in a run record nobody read")


async def test_a_configured_model_or_the_default_fallback_still_dispatches(
        aut, monkeypatch, tmp_path):
    """Clause 2: the validator refuses typos, never a working dispatch.

    Three shapes must all still run: a slot key (`primary`), a declared alias
    (`secondary`), and no `model:` at all, which falls back to the `model.default`
    of the config file `run_task` reads.

    The third case is why this test carries its own `config.yaml`. Pointing
    `LLOYD_HOME` at a fixture config whose `model.default` is `secondary` — a
    real slot, and NOT this box's default, which is `primary` — is what makes the
    assertion falsifiable: measured against the live default, a fallback that
    hardcoded the string `"primary"` would have passed the first version of this
    test, because the fixture and the box happened to agree. `secondary_enabled`
    is forced on for the same reason: `resolve_model_alias` rewrites
    `secondary`→`primary` when that switch is off, and with it off every case
    below collapses onto `primary` and neither the slot lookup nor the fallback
    has any observable content. That rewrite predates this check and is unchanged
    by it — pinned separately at
    `test_a_secondary_task_still_falls_back_when_the_secondary_slot_is_off`.
    """
    from app.config import CONFIG

    cfg = tmp_path / "config.yaml"
    cfg.write_text("agent:\n  max_turns: 60\nmodel:\n  default: secondary\n")
    monkeypatch.setattr(autonomy, "LLOYD_HOME", tmp_path)
    monkeypatch.setitem(CONFIG, "secondary_enabled", True)

    for model, expected in (("primary", "primary"),
                            ("secondary", "secondary"),
                            # Unset: must land on THIS file's default, not the
                            # box's, and not on a literal.
                            (None, "secondary")):
        calls = _stub_engine(monkeypatch)
        write_task(aut, 1, model=model)

        result = await aut.run_task(1)

        assert result["success"] is True, (
            f"model={model!r} was refused; only an unconfigured NAME may be")
        assert len(calls) == 1, f"model={model!r} must reach the engine once"
        handed = getattr(calls[0], "model", None)
        assert handed in _CONFIGURED_MODELS, (
            f"the engine was handed {handed!r}, which is not a configured slot")
        assert handed == expected, (
            f"model={model!r}: expected the engine to be handed {expected!r}, "
            f"got {handed!r}")


async def test_the_unset_model_lands_on_the_default_this_box_boots_with(
        aut, monkeypatch):
    """Clause 2's literal: `model.default: primary`, read off the tracked file.

    The test above proves the fallback is READ from a config file; this one proves
    what the shipped one says, because the clause is stated in terms of `primary`
    and a fixture cannot stand in for the file `run_task` opens when
    `LLOYD_HOME` is unpatched — which is the state production runs in. Both halves
    are checked: that `config.yaml` really carries `model.default: primary` (it is
    a tracked, human-editable file, so the premise is measured, not assumed), and
    that an unset `model:` therefore hands the engine `primary` rather than the
    empty string `resolve_model_alias` would otherwise pass through.

    If a person ever repoints the default, this fails and names the file — the
    clause's wording then needs the same edit, which is the point of pinning it.
    """
    import yaml as _yaml
    from app.config import CONFIG

    shipped = _yaml.safe_load(
        (autonomy.LLOYD_HOME / "config.yaml").read_text()) or {}
    default = (shipped.get("model") or {}).get("default")
    assert default == "primary", (
        f"clause 2 is written against `model.default: primary`; "
        f"{autonomy.LLOYD_HOME / 'config.yaml'} now says {default!r}")
    assert default in _CONFIGURED_MODELS, "and the default is itself a real slot"
    assert (CONFIG.get("model") or {}).get("default") == default, (
        "the file this process actually booted with says the same thing")

    calls = _stub_engine(monkeypatch)
    write_task(aut, 1, model=None)

    result = await aut.run_task(1)

    assert result["success"] is True, "an unset model is the default path, not a refusal"
    assert len(calls) == 1
    assert getattr(calls[0], "model", None) == "primary"


async def test_a_secondary_task_still_falls_back_when_the_secondary_slot_is_off(
        aut, monkeypatch):
    """The pre-existing `secondary_enabled` rewrite survives the new validator.

    `resolve_model_alias` turns `secondary` into `primary` when the switch is
    off, and it runs BEFORE the membership test. With both `primary` and
    `secondary` configured the rewritten value is a configured slot either way,
    so this dispatch must not be refused — a validator that ran first, or that
    checked the file's raw string instead of the resolved one, would refuse a
    task that has always worked.
    """
    from app.config import CONFIG

    monkeypatch.setitem(CONFIG, "secondary_enabled", False)
    calls = _stub_engine(monkeypatch)
    write_task(aut, 1, model="secondary")

    result = await aut.run_task(1)

    assert result["success"] is True, "a switched-off secondary falls back; it does not refuse"
    assert len(calls) == 1
    assert getattr(calls[0], "model", None) == "primary"


async def test_a_refusal_leaves_one_line_in_the_task_file_it_refuses(aut, monkeypatch):
    """The refusal is durable where a person reads, not only in a log line.

    `run_task` returns a `skipped` queue row and logs at error level; the review
    rung of the previous round named the gap — "the refusal is loud but not
    durable", and clause 3 forbids the obvious remedy of moving the row. The task
    file's own Activity Log is the durable surface that already works on this box
    (it is where #85's `DISABLED after 3 consecutive failures` line landed), so
    the refusal gets one line there naming the value and the real slots. Once
    per (task, model): a refusal does not consume the retry budget, so the same
    typo is refused again on the next tick and the next, and a curated file that
    grows a line per tick is a file nobody reads.
    """
    monkeypatch.setattr(autonomy, "_model_refusal_logged", set())
    _stub_engine(monkeypatch)
    _stub_sessions(monkeypatch)
    path = write_task(aut, 1, model="eco")

    await aut.run_task(1)
    lines = [ln for ln in path.read_text().splitlines()
             if "not a configured alias" in ln]
    assert len(lines) == 1, f"one durable line for the first refusal, got {lines}"
    assert "eco" in lines[0] and "primary" in lines[0] and "secondary" in lines[0]

    await aut.run_task(1)
    await aut.run_task(1)
    lines = [ln for ln in path.read_text().splitlines()
             if "not a configured alias" in ln]
    assert len(lines) == 1, (
        f"the second and third refusal must not repeat the note, got {lines}")
    # And clause 3 again from the other side: the note is a body append, so no
    # front-matter field moved to carry it.
    assert int(read_task(aut, 1)["failure_count"]) == 0


async def test_the_refusal_survives_the_real_execute_adapter(
        aut, monkeypatch, tmp_path):
    """The refusal reaches the board as a skipped row that says why — across the seam.

    `workers/sources/scheduled_task.execute()` builds the dict the queue stores
    from a HAND-WRITTEN whitelist (`status`, `summary`, `task_id`,
    `artifact_path`, `response`, `meta`, `claims`), and `workers/pool.
    normalize_result` is the hop after it. A test that hands `normalize_result` a
    dict shaped like `execute()`'s output asserts on its own input and never
    touches either (#945 severed the pilot's `claims` exactly there, for 4,922
    rows). So this drives the real `run_task` refusal through the real
    `execute()` into the real `normalize_result`.

    The outcome the pool's own docstring is written to prevent is the one being
    refused here: `{"skipped": reason}` with no status was recorded as a SUCCESS
    for all 22 runs of `automod-regression`, so "a check that never ran" read as
    "a check that ran and found nothing". For #1209 the same misread would be
    worse — a task that cannot dispatch at all would read as a green night.

    Stubbed only what is not this claim: the model loop, the vLLM health probe
    `execute()` waits on before it dispatches, and the Discord completion
    notifier. Everything between `run_task` and `normalize_result` is shipped
    code.
    """
    import workers.sources.scheduled_task as scheduled_task
    from workers.pool import normalize_result
    from workers.queue import QueueItem

    calls = _stub_engine(monkeypatch)
    _stub_sessions(monkeypatch)
    write_task(aut, 1, name="Nightly Secondary Routing Eval", model="eco")
    monkeypatch.setattr(scheduled_task, "_vllm_healthy", lambda *a, **k: True)

    async def _no_notify(*_a, **_k):
        return None
    monkeypatch.setattr("app.discord_notify._discord_notify_task_complete", _no_notify)

    item = QueueItem(id=1, source="scheduled-task", kind="run", priority=50,
                     payload={"task_id": "1"}, dedup_key=None, state="running",
                     attempts=1, enqueued_at="", claimed_at=None, claimed_by=None,
                     completed_at=None, error=None)
    out = await scheduled_task.execute(item)

    assert out["status"] == "skipped", out
    assert "eco" in out["summary"] and "primary" in out["summary"], out
    assert calls == [], "crossing the seam must not smuggle in an engine call"

    norm = normalize_result(item, out)
    assert norm["status"] == "skipped", (
        "a refused dispatch must not be the pool's default `success`")
    assert "eco" in norm["summary"] and "not a configured alias" in norm["summary"], (
        f"the row a human reads must carry the refusal, got: {norm['summary']!r}")
    assert norm["artifact_path"] == "", (
        "no run record exists, so the row must not point at one")


async def test_an_unknown_model_refusal_does_not_spend_the_retry_budget(aut, monkeypatch):
    """Clause 3: one typo in one field cannot disable a schedule.

    The 09-16 refusals were booked as `failure_kind: task`, so three of them
    hit `max_retries: 3` and nulled `next_run`. This task starts one failure
    short of that line with the SAME bad model: if the refusal were recorded as
    a failure, the file would read `status: failed` afterwards. Every field
    that a failure would move is compared against its own before-value, so the
    test fails on a refusal that writes any of them.
    """
    _stub_engine(monkeypatch)
    write_task(aut, 1, model="eco", failure_count=2, max_retries=3)
    before = read_task(aut, 1)

    result = await aut.run_task(1)

    assert result["success"] is False
    assert result.get("disabled") is not True
    after = read_task(aut, 1)
    assert int(after["failure_count"]) == 2, "a config refusal is not a retry"
    for field in ("status", "failure_count", "next_run", "last_attempt", "last_run"):
        assert after.get(field) == before.get(field), (
            f"`{field}` moved on a refused dispatch: {before.get(field)!r} -> "
            f"{after.get(field)!r}")
