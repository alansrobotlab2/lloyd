"""#1682 — the fleet watchdog is driven by the scheduler loop, not by a source tick.

The three stall alarms and the unparseable-task-file scan used to sit on the body of
`workers/sources/scheduled_task.py:enqueue_if_due`, which the pool reaches only for a
source whose config says `enabled: true`. One dispatch-side switch therefore decided
whether the fleet was watched at all, and switching it off produced no alert about the
disarmament. These tests drive the two halves from the seats they now have: the alarms
from `WorkerPool._scheduler_loop` (or `fleet_watchdog.tick` directly), and dispatch
from `scheduled_task.enqueue_if_due` — and pin that each half keeps doing its own job
while the other is switched off, held, or down.

What this file does NOT touch is the outage alarm's own threshold: `_note_vllm_outage`
and `_VLLM_DOWN_ALERT_SECONDS` stayed in the source, and #1683 owns them.
"""
from __future__ import annotations

import ast
import asyncio
import datetime as dt
import inspect
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import workers.fleet_watchdog as fw                     # noqa: E402
import workers.sources as sources                       # noqa: E402
import workers.sources.scheduled_task as st             # noqa: E402
from app import autonomy                                # noqa: E402
from workers.pool import WorkerPool                     # noqa: E402
from workers.queue import WorkQueue                     # noqa: E402


@pytest.fixture(autouse=True)
def _both_states_restored():
    """Restore the CONTENTS of both module-global state dicts around every test.

    The surveillance streaks moved to `fleet_watchdog._state` and the outage
    accounting stayed in `scheduled_task._state`, so a test here mutates both and
    `monkeypatch` cannot undo a mutation inside a dict a pre-existing binding
    points at. Without this, a tick that seeds a stall streak leaves it set for
    whichever test runs next, and "did this alarm confirm across ticks?" becomes
    a function of test order rather than of the state the test asked for. Same
    contract `tests/test_autonomy_scheduler.py` holds for its file (#939, #1682).
    """
    saved = {"fw": {k: (set(v) if isinstance(v, set) else v)
                    for k, v in fw._state.items()},
             "st": {k: (set(v) if isinstance(v, set) else v)
                    for k, v in st._state.items()}}
    yield
    # Read back through the module attribute, as `tests/test_autonomy_scheduler.py`
    # does: a test that swapped a dict in with `monkeypatch` has the module's own
    # object back by the time this runs, and that object is the one whose contents
    # have to go back.
    fw._state.clear()
    fw._state.update(saved["fw"])
    st._state.clear()
    st._state.update(saved["st"])


@pytest.fixture
def board(tmp_path, monkeypatch):
    """An isolated autonomy task dir with a resolvable skill: a board to point at.

    Empty by default. `_starving_alerts` in `tests/test_workers_pool.py` established
    why that matters: with no task files, neither stall alarm has anything to name,
    so an alert observed here can only be the clause under test.
    """
    d = tmp_path / "autonomy"
    d.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", d)
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n")
    monkeypatch.setattr(autonomy, "_SKILL_FOR_TESTS", str(skill), raising=False)
    return d


def write_task(board, task_id, **fm):
    """One task file, shaped exactly as `app.autonomy` writes them."""
    base = {"id": task_id, "name": f"task{task_id}", "type": "autonomy",
            "status": "up_next", "frequency": "daily", "priority": "medium",
            "skill_name": str(board.parent / "SKILL.md"), "timeout_seconds": 2,
            "max_retries": 3, "failure_count": 0}
    base.update(fm)
    base = {k: v for k, v in base.items() if v is not None}
    (board / f"{task_id}-task{task_id}.md").write_text(
        f"---\n{yaml.dump(base)}---\n\nbody\n\n## Activity Log\n")


def _ago(**delta) -> str:
    return (dt.datetime.now(dt.timezone.utc)
            - dt.timedelta(**delta)).isoformat()


def _aged_queued_row(db_path, *, minutes: int):
    """A `scheduled-task` queue row `minutes` old, with a payload that names no task.

    Raw SQL because the age IS the fixture: `WorkQueue.enqueue` stamps `enqueued_at`
    from the wall clock. `payload=None` on purpose — `_queue_starving` skips any row
    whose payload carries a `task_id`, so a row built like a real dispatch enqueue
    would be invisible to the detector this file's clause-5 test depends on.
    """
    q = WorkQueue(db_path)
    row_id = q.enqueue(source="scheduled-task", kind="heartbeat", payload=None)
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE queue SET enqueued_at=? WHERE id=?",
                     (_ago(minutes=minutes), row_id))
        conn.commit()
    return q.db_path


def _pacing_constants_unchanged():
    """The five numbers that ARE the pacing, at their pre-move values.

    Every pacing loop in this file counts in units of these constants, which is right
    for "does the alarm pace itself this way" and useless for "did the move change the
    pace": a tick count read from the constant follows a constant that was edited. So
    each loop starts here, with the values `scheduled_task.py` carried into the split
    (#1682 changes no alarm's pacing): two 5-tick streaks, a 6 h and a 24 h cooldown,
    a 30 min scan. These are the numbers `architecture/workers-jobs.md` and the alert
    messages quote, not numbers this file invented.
    """
    assert (fw._STALL_ALARM_TICKS, fw._STALL_NEXTRUN_TICKS) == (5, 5)
    assert (fw._STALL_ALERT_INTERVAL_SECONDS,
            fw._STALL_NEXTRUN_ALERT_INTERVAL_SECONDS) == (6 * 3600, 24 * 3600)
    assert fw._UNPARSEABLE_SCAN_SECONDS == 30 * 60


def _watch_alerts(monkeypatch) -> list[str]:
    """Capture every alert the watchdog posts, replacing the Discord hop."""
    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(fw, "_alert", _capture)
    return alerts


def _watch_detectors(monkeypatch) -> dict:
    """Count the three detectors and the scan by WRAPPING the real ones.

    Wrapping rather than replacing is the point: the call count is the assertion
    and the returned verdict stays the real one, so a green here cannot come from
    a stub that reported a stall the fleet does not have.
    """
    calls = {"scan": 0, "grossly": 0, "nextrun": 0}
    real_scan, real_grossly, real_nextrun = (
        fw._unparseable_task_files, fw._grossly_overdue, fw._next_run_stalled)

    def _scan():
        calls["scan"] += 1
        return real_scan()

    def _grossly(queue):
        calls["grossly"] += 1
        return real_grossly(queue)

    def _nextrun(queue):
        calls["nextrun"] += 1
        return real_nextrun(queue)

    monkeypatch.setattr(fw, "_unparseable_task_files", _scan)
    monkeypatch.setattr(fw, "_grossly_overdue", _grossly)
    monkeypatch.setattr(fw, "_next_run_stalled", _nextrun)
    return calls


def _streak_at_alerting_threshold(monkeypatch, **over):
    """Seed both streaks and clear both cooldowns, without touching other keys.

    Seeded AT the threshold, not below it, so an `alerts == []` or a single-tick
    alert assertion has a firing condition inside one tick: at 0 the counter needs
    five ticks and the assertion could not fail at all.
    """
    fresh = {**fw._state,
             "unparseable_scan_at": None,
             "stall_streak": fw._STALL_ALARM_TICKS, "stall_alerted_at": None,
             "nextrun_streak": 0, "nextrun_alerted_at": None}
    fresh.update(over)
    monkeypatch.setattr(fw, "_state", fresh)


class _NoEnabledKeyConfig(dict):
    """A source config block that refuses to be asked whether it is enabled.

    The negative half of clause 5, made unfalsifiable-proof: `fleet_watchdog` may
    read `max_duration_seconds` out of the block it is handed, and if it asks the
    one question this extraction exists to stop asking, the read raises through
    `tick()` instead of being quietly tolerated.
    """

    def _refuse(self, key):
        if key == "enabled":
            raise AssertionError(
                "the fleet watchdog read the `scheduled-task` source's `enabled` "
                "flag, which is the disarmament path #1682 exists to remove")
        raise KeyError(key)

    def get(self, key, default=None):
        if key == "enabled":
            self._refuse(key)
        return super().get(key, default)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            self._refuse(key)


# ── clause 1: the alarms run from the loop, source enabled or not ────────────


async def test_the_scheduler_loop_watches_the_fleet_with_the_source_disabled(
        board, monkeypatch, tmp_path):
    """Clause 1, across the real seam: a started pool, `enabled: false`, one alert.

    `WorkerPool.start()` runs its own scheduler loop, and the pass under that loop
    skips a disabled source before it reads any stamp or calls any `enqueue_if_due`
    (`workers/pool.py`, `if not src_cfg.get("enabled", False): continue`). So the
    alert below can only have come from the seat #1682 gave the alarms: the loop's
    own `_watch_fleet`. Pre-fix this is silent forever — the disabled source never
    reached the tick that carried the alarms, and nothing reported the silence.

    The sibling seats (`_maybe_sweep_poisoned`, `_probe_services`,
    `_watch_dispatch`) are stubbed because each has its own tests and
    `_probe_services` would probe live service ports from a unit test;
    `_watch_fleet` — the seat under test — is left real, as is `tick`, the
    detectors and the queue.
    """
    write_task(board, 701, frequency="daily",
               last_run=_ago(days=5), next_run=_ago(days=5))
    q = WorkQueue(tmp_path / "loop.db")

    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {"scheduled-task": st},
                        raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {
        "scheduled-task": {"enabled": False, "interval_seconds": 0,
                           "max_duration_seconds": 1800}}, raising=False)
    calls = _watch_detectors(monkeypatch)
    _streak_at_alerting_threshold(monkeypatch)
    alerts = _watch_alerts(monkeypatch)

    async def _noop(*_a, **_k):
        return None

    pool = WorkerPool(q, slots=0, poll_idle_seconds=0.01)
    monkeypatch.setattr(pool, "_maybe_sweep_poisoned", _noop)
    monkeypatch.setattr(pool, "_probe_services", _noop)
    monkeypatch.setattr(pool, "_watch_dispatch", _noop)
    await pool.start()
    try:
        for _ in range(300):
            if alerts:
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()

    assert calls["scan"] >= 1, "the task-file scan never ran on a scheduler tick"
    assert calls["grossly"] >= 1, "the due-ness stall detector never ran"
    assert calls["nextrun"] >= 1, "the next_run stall detector never ran"
    assert len(alerts) == 1, (
        f"expected the one stall alert, got {alerts}")
    assert "701" in alerts[0], (
        "the seeded overdue task was not the one alerted on: " + alerts[0])
    assert not q.list_items(source="scheduled-task", limit=50), (
        "a source disabled in config enqueued work, so the alert above could have "
        "come from dispatch rather than from the watchdog seat")


def test_the_watchdog_seat_is_outside_the_pass_it_watches():
    """Clause 1's other half: WHERE it is seated, and what that buys.

    The seat is the claim, so it is asserted structurally the way #1681 asserted
    `_watch_dispatch`: before the pass, outside the `try` whose `except` swallows a
    raising source, and beside the poison sweep and the service probe — the three
    readers the loop keeps because a source cannot be trusted to report on itself.
    """
    src = inspect.getsource(WorkerPool._scheduler_loop)
    fleet_at = src.index("await self._watch_fleet()")
    pass_at = src.index("await self._scheduler_pass()")
    sweep_at = src.index("await self._maybe_sweep_poisoned()")
    probe_at = src.index("await self._probe_services()")
    try_at = src.index("try:")
    assert fleet_at < pass_at, "the watchdog runs after the pass it watches"
    assert fleet_at < try_at, "the watchdog sits inside the try that can be skipped"
    assert sweep_at < fleet_at and probe_at < fleet_at, (
        "the watchdog is not beside the other two always-on readers")
    seat = inspect.getsource(WorkerPool._watch_fleet)
    assert "except Exception" in seat, "the seat's never-raises contract"


# ── clause 2: dispatch no longer runs the watchdog ──────────────────────────


async def test_the_dispatch_tick_runs_no_watchdog_and_still_enqueues(
        board, monkeypatch, tmp_path):
    """Clause 2: the source's tick dispatches and nothing else.

    The split has to be clean in both directions, or the alarms are simply called
    twice per minute and the 60 s dispatch tick is still what paces a 6 h alarm. So
    `tick` and every detector and the scan are counted while the REAL
    `enqueue_if_due` runs a due task into the queue: the enqueued row is the
    positive control that the tick was not merely emptied, and the zero counts are
    the relocation.
    """
    write_task(board, 702, frequency="hourly", last_run=_ago(minutes=90))
    q = WorkQueue(tmp_path / "dispatch.db")

    ticks: list[int] = []

    async def _counting_tick(queue):
        ticks.append(1)

    monkeypatch.setattr(fw, "tick", _counting_tick)
    calls = _watch_detectors(monkeypatch)
    fw_alerts = _watch_alerts(monkeypatch)
    st_alerts: list[str] = []

    async def _st_alert(msg):
        st_alerts.append(msg)

    monkeypatch.setattr(st, "_alert", _st_alert)
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)

    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})

    enqueued = {str(i.payload.get("task_id")) for i in
                q.list_items(source="scheduled-task", limit=500)}
    assert enqueued == {"702"}, (
        f"the dispatch tick stopped enqueuing what the board calls due: {enqueued}")
    assert ticks == [], "the source's tick still drives the watchdog"
    assert calls == {"scan": 0, "grossly": 0, "nextrun": 0}, (
        f"the surveillance still runs from inside dispatch: {calls}")
    assert fw_alerts == [] and st_alerts == [], (
        f"a dispatch tick raised an alert: {fw_alerts} {st_alerts}")


# ── clause 3: #938 survives the split ──────────────────────────────────────


async def test_a_model_server_outage_stops_dispatch_and_not_the_watchdog(
        board, monkeypatch, tmp_path):
    """Clause 3: the engine is down, so the fleet is watched AND not fed.

    #938's invariant is that an outage pauses dispatch and not watching. With the
    detectors no longer on the dispatch path at all, the half that must survive is
    the gate's: `enqueue_if_due` still enqueues nothing while the probe fails. The
    watching half is asserted from the other module — the watchdog never asks the
    engine's health, so it cannot be silenced by it, and it still alerts on the
    task dispatch is skipping.
    """
    write_task(board, 703, frequency="daily",
               last_run=_ago(days=5), next_run=_ago(days=5))
    q = WorkQueue(tmp_path / "outage.db")

    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: False)
    calls = _watch_detectors(monkeypatch)
    _streak_at_alerting_threshold(monkeypatch)
    alerts = _watch_alerts(monkeypatch)

    await fw.tick(q)

    assert calls["scan"] >= 1, "the scan skipped a tick taken with the engine down"
    assert calls["grossly"] >= 1 and calls["nextrun"] >= 1, (
        f"the detectors skipped a tick taken with the engine down: {calls}")
    assert len(alerts) == 1 and "703" in alerts[0], (
        f"the stall alarm did not fire through the outage: {alerts}")

    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    assert not q.list_items(source="scheduled-task", limit=50), (
        "the gate let work through with the model server down, so this test's "
        "empty queue would say nothing about clause 3")

    # Control on the next tick: only the probe flips, and the same task enqueues.
    monkeypatch.setattr(st, "_vllm_healthy", lambda *a, **k: True)
    await st.enqueue_if_due(q, {"max_duration_seconds": 1800})
    enqueued = {str(i.payload.get("task_id")) for i in
                q.list_items(source="scheduled-task", limit=500)}
    assert enqueued == {"703"}, (
        f"the healthy tick did not enqueue what the down tick held: {enqueued}")


# ── clause 4's pacing, measured in the module it now lives in ───────────────


async def test_the_moved_alarms_keep_their_streak_and_cooldown_pacing(
        board, monkeypatch, tmp_path):
    """Clause 4: 5 confirming ticks, then its own cooldown, unchanged by the move.

    Driven through `tick` because pacing is what the extraction could most quietly
    have lost: the streaks are module state, and a `tick` that rebuilt `_state` per
    call, or dropped the cooldown comparison, would alert on every 60 s scheduler
    pass — which is exactly how this alarm earned its 6 h cooldown (100 alerts in
    6 days). The count below is the whole assertion: four silent ticks, one alert,
    and silence again for the ticks after it.
    """
    _pacing_constants_unchanged()
    write_task(board, 704, frequency="daily",
               last_run=_ago(days=5), next_run=_ago(days=5))
    q = WorkQueue(tmp_path / "pacing.db")

    # The `next_run` alarm's cooldown seeded as just-fired — its real clock, not a
    # pinned one — which is the state production is in whenever it alerted in the
    # last day, and what makes the alert count below about the one alarm under test.
    _streak_at_alerting_threshold(
        monkeypatch, stall_streak=0,
        nextrun_alerted_at=dt.datetime.now(dt.timezone.utc))
    alerts = _watch_alerts(monkeypatch)

    for _ in range(fw._STALL_ALARM_TICKS - 1):
        await fw.tick(q)
    assert alerts == [], (
        f"alerted before {fw._STALL_ALARM_TICKS} confirming ticks: {alerts}")
    await fw.tick(q)
    assert len(alerts) == 1, (
        f"did not alert on tick {fw._STALL_ALARM_TICKS}: {alerts}")
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, (
        f"re-alerted inside the {fw._STALL_ALERT_INTERVAL_SECONDS}s cooldown: "
        f"{len(alerts)} alerts — the noise the cooldown exists to stop")
    assert "704" in alerts[0], alerts[0]
    assert all("next_run" not in m for m in alerts), (
        "the count above is not about one alarm: " + repr(alerts))


# ── clause 5: one config field read, no key of its own ─────────────────────


async def test_the_watchdog_reads_max_duration_seconds_and_never_the_enabled_flag(
        board, monkeypatch, tmp_path):
    """Clause 5 across the config seam: the threshold that fires the alarm is config's.

    `max_duration_seconds` reaches the starving clause and nowhere else, so it is
    checked three ways — the read, the threshold it sets, the alert that threshold
    decides — because the value appears in NO alert text: the moved alarm quotes the age
    of the oldest row, never its own threshold (clause 4 requires that message
    unchanged). The proof is therefore that the config value decides WHETHER it fires.

    The board is empty, so the starving clause is the only alarm this can produce, and
    the queue row is 20 min old. At the 300 s this hands the source block, 3x is 15 min,
    so the row starves, the streak confirms and one alert goes out; at the 1800 s
    fallback the same 3x bound is 90 min, so the same row is inside the cap, the streak
    never confirms, and there is nothing to find. A watchdog that ignored the source
    block would report the same result for both, and fail the second half.
    """
    def config_at(max_duration):
        block = _NoEnabledKeyConfig({"max_duration_seconds": max_duration})
        monkeypatch.setattr(sources, "get_sources_config",
                            lambda: {"scheduled-task": block}, raising=False)

    db = _aged_queued_row(tmp_path / "starve.db", minutes=20)
    q = WorkQueue(db)
    assert json.loads(q.list_items(source="scheduled-task", limit=1)[0]
                      .payload or "null") is None, (
        "the fixture row carries a task_id, so `_queue_starving` would skip it and both "
        "halves of this test would report zero alerts for the wrong reason")

    config_at(300)
    assert fw._max_duration_seconds() == 300, (
        "the watchdog is not reading the field out of the `scheduled-task` block "
        "through `get_sources_config`, which is the read the clause names")
    # The threshold the read produces, at both values of the field: the alarm's own
    # detector, so the number above is shown to be the one that decides.
    assert fw._queue_starving(q, 300) > 0, "3x 300 s is 15 min, so a 20 min row starves"
    assert fw._queue_starving(q, 1800) == 0.0, "3x 1800 s is 90 min, so it does not"

    alerts = _watch_alerts(monkeypatch)
    _streak_at_alerting_threshold(monkeypatch)
    await fw.tick(q)
    assert len(alerts) == 1, f"expected the one starving-queue alert: {alerts}"
    assert "oldest claimable queue item is 20 min old" in alerts[0], alerts[0]

    # Same row, same state, only the read-out number different: inside the cap now, so
    # no streak confirmation and no alert across a full alarm window.
    alerts.clear()
    config_at(1800)
    _streak_at_alerting_threshold(monkeypatch)
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert alerts == [], (
        "the alarm fired for a row well inside the 3x cap, so the config value is not "
        f"what sets the threshold: {alerts}")


def test_the_watchdog_adds_no_config_key_of_its_own():
    """Clause 5's other half: it needs nothing that is not already in config.yaml.

    A new key would put this item outside what an automod round can land
    (`config.yaml` is denied to a round), so the watchdog reads the one field it needs
    out of the source block instead. The positive control on `max_duration_seconds` is
    what makes the absence assertions mean anything: a pattern empty against the corpus
    reads exactly like an absent key.
    """
    config_text = (Path(__file__).resolve().parent.parent
                   / "config.yaml").read_text()
    src = inspect.getsource(fw)
    # Keys are literals in the AST, so this counts config READS rather than mentions:
    # the module's own prose quotes pool.py's `src_cfg.get("enabled", False)` guard
    # while describing the disarmament, and a substring check on the text would read
    # that description as if it were the read.
    keys = {n.value for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "max_duration_seconds" in keys, (
        "positive control broken: the watchdog no longer reads the field by that name, "
        "so the absence checks below measure nothing")
    assert "get_sources_config" in src, "it no longer reads the source config"
    assert "enabled" not in keys, (
        "the watchdog asks some config whether a source is enabled, which is the "
        "disarmament path #1682 exists to remove")
    assert "fleet_watchdog" not in config_text and "fleet-watchdog" not in config_text, (
        "config.yaml grew a key for the watchdog, which a round may not add")
    assert "from app.config import" not in src, (
        "the watchdog reads CONFIG directly instead of the source block")


async def test_the_nextrun_alarm_keeps_its_own_streak_and_24h_cooldown(
        board, monkeypatch, tmp_path):
    """Clause 4, second alarm: the `next_run` streak and cooldown are its own.

    The two stall alarms were separate assertions before the move and have to stay
    separate after it — separate streak, separate cooldown, separate message — so
    this task is built so ONLY the `next_run` detector can see it: `last_run` absent
    means the grossly-overdue detector has no interval to weigh it against, and an
    empty queue means nothing is starving. Four silent ticks, the alert on the fifth,
    then a window of further ticks with still one alert: the 24 h cooldown, which is a
    different constant from the stall alarm's 6 h and would silently become that one
    if the two ever shared a slot.
    """
    _pacing_constants_unchanged()
    write_task(board, 705, frequency="daily", last_run=None, next_run=_ago(days=2))
    q = WorkQueue(tmp_path / "nextrun.db")
    assert fw._next_run_stalled(q), (
        "the fixture is not visible to the `next_run` detector, so the counts below "
        "could not fail")

    _streak_at_alerting_threshold(monkeypatch, stall_streak=0,
                                  nextrun_streak=0, nextrun_alerted_at=None)
    alerts = _watch_alerts(monkeypatch)

    for _ in range(fw._STALL_NEXTRUN_TICKS - 1):
        await fw.tick(q)
    assert alerts == [], (
        f"the `next_run` alarm fired before its own {fw._STALL_NEXTRUN_TICKS} ticks: "
        f"{alerts}")
    await fw.tick(q)
    assert len(alerts) == 1 and "705" in alerts[0], alerts
    assert fw._state["nextrun_alerted_at"] is not None, alerts

    for _ in range(fw._STALL_NEXTRUN_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, (
        f"re-alerted inside the {fw._STALL_NEXTRUN_ALERT_INTERVAL_SECONDS}s "
        f"(24 h) cooldown: {alerts}")


async def test_the_parse_scan_keeps_its_30_minute_cadence(
        board, monkeypatch, tmp_path):
    """Clause 4, third reader: the scan still scans on its 30 min slot, not per tick.

    The cadence is the only thing keeping a 60 s scheduler loop from walking and
    parsing every task file on every pass, and it is measured against
    `autonomy._utcnow`, the instant the scan itself claims. A broken file is what
    makes the count observable: the scan reports it, so the count is the number of
    scans and the alert is the number of transitions. Two scans, one alert — a second
    alert would mean the transition rule, not the cadence, had broken.
    """
    from app import autonomy
    assert callable(getattr(autonomy, "_utcnow", None)), (
        "`autonomy._utcnow` is gone, so the scan's clock has no single indirection to "
        "pin and the cadence below could not fail")
    now = [dt.datetime(2026, 9, 27, 12, 0, tzinfo=dt.timezone.utc)]
    monkeypatch.setattr(autonomy, "_utcnow", lambda: now[0])

    # Truncated mid front matter: `_parse_task_file` splits on `---\n` and returns
    # None below three parts (app/autonomy.py:148-150), which is also the shape of the
    # 2026-05-28 stall. A YAML error alone would not do — the regex fallback in
    # `parse_frontmatter_text` recovers the fields and the file parses.
    _pacing_constants_unchanged()
    (board / "9999-broken.md").write_text("---\nid: 9999\nname: broken\nfrequency: daily\n")
    assert fw._unparseable_task_files() == ["9999-broken.md"], (
        "the fixture file is not one the scheduler fails to parse, so the scan could "
        "report nothing and the counts below would measure nothing")

    q = WorkQueue(tmp_path / "cadence.db")
    calls = _watch_detectors(monkeypatch)
    _streak_at_alerting_threshold(monkeypatch, stall_streak=0)
    alerts = _watch_alerts(monkeypatch)

    await fw.tick(q)
    assert calls["scan"] == 1, "the first tick did not scan"
    assert len(alerts) == 1 and "unparseable" in alerts[0], alerts

    await fw.tick(q)
    assert calls["scan"] == 1, (
        "the scan ran twice in one cadence slot: at the pool's 60 s loop that is a "
        "full parse of the board every minute")

    now[0] = now[0] + dt.timedelta(seconds=fw._UNPARSEABLE_SCAN_SECONDS)
    await fw.tick(q)
    assert calls["scan"] == 2, "the scan never re-ran, so it is still a one-shot"
    assert len(alerts) == 1, (
        f"the same broken file alerted again on the second scan: {alerts}")


# ── #1735 — the next_run alert cooldown survives the process that set it ─────
#
# `_state["nextrun_alerted_at"]` is the instant the `next_run` alarm last reached a
# person, and it lived only in a module-global dict. A new process therefore starts
# having never alerted, reaches its own 5-tick streak on the same board, and posts
# the alert again: `~/lloyd-data/logs/server.err` shows pool starts at 21:10:13 and
# 22:16:18 on 2026-09-27 with next_run alerts posted at 21:14:20 and 22:20:28 —
# 4 min 07 s and 4 min 10 s after each boot, which is exactly
# `_STALL_NEXTRUN_TICKS` x the pool's 60 s scheduler interval. The 24 h cooldown was
# a property of one process's uptime. These tests drive two processes against ONE
# queue, because the reset is the defect and a test that reuses one process proves
# nothing.


def _fresh_process_state(monkeypatch) -> dict:
    """Replace `_state` with what a second watchdog process actually starts with.

    Every key of the module's own default literal, at its default value — not the
    state the first batch of ticks left behind, which is the thing under test. The
    key-set assertion is what keeps this honest: a `_state` that gained a key would
    silently stop being "a fresh process" and the no-re-alert assertion below would
    then be measuring a fixture rather than the durable read.
    """
    fresh = {"unparseable_scan_at": None, "unparseable_alerted": set(),
             "stall_streak": 0, "stall_alerted_at": None,
             "nextrun_streak": 0, "nextrun_alerted_at": None,
             "nextrun_parked_logged_at": None}
    assert set(fresh) == set(fw._state), (
        f"fleet_watchdog._state has keys {sorted(fw._state)}, so this is no longer a "
        f"fresh process's surveillance state")
    monkeypatch.setattr(fw, "_state", fresh)
    return fresh


def _durable_alert_instants(queue: WorkQueue) -> list[tuple[str, dt.datetime]]:
    """Every watermark row under the watchdog's own source whose value is an instant.

    Read through the queue's public API (`wm_keys`/`wm_get`) rather than a key name
    the module keeps private: the claim being tested is that the alert instant is
    DURABLE QUEUE STATE, and a test that hardcoded the row's key would still pass if
    the value moved somewhere unreadable. Rows that do not parse as an instant — any
    pacing cursor another part of the source keeps beside it — are not the alert
    instant and are skipped.
    """
    found = []
    for key in queue.wm_keys(fw.NAME):
        value = queue.wm_get(fw.NAME, key)
        try:
            found.append((key, dt.datetime.fromisoformat(value)))
        except (TypeError, ValueError):
            continue
    return found


async def test_a_restart_inside_the_cooldown_does_not_re_post_the_nextrun_alarm(
        board, monkeypatch, tmp_path):
    """Clauses 1 + 2: one alert per 24 h, measured across two processes.

    A late un-parked task, five ticks for the first process, and — the part clause 2
    exists for — the assertion that the first process posted exactly one alert
    NAMING THE TASK before any no-re-alert claim is made. Against that, a second
    process with the fresh module default in hand ticks five more times over the
    SAME queue: its streak reaches `_STALL_NEXTRUN_TICKS` again, so the only thing
    that can hold the alert back is the instant the first process recorded. One
    alert total.

    The queue is a `WorkQueue` on a real file, not a mock, because the thing being
    survived is a restart of the process that owns the in-memory dict.
    """
    _pacing_constants_unchanged()
    write_task(board, 706, frequency="daily", last_run=None, next_run=_ago(days=2))
    q = WorkQueue(tmp_path / "restart.db")
    assert fw._next_run_stalled(q) and fw._grossly_overdue(q) == [], (
        "the fixture is not a board only the `next_run` alarm sees, so the alert "
        "counts below would not be attributable to it")

    first = _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)

    for _ in range(fw._STALL_NEXTRUN_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1 and "706" in alerts[0], (
        f"the first process did not post exactly one alert naming the late task: "
        f"{alerts}. Nothing below can be assessed until it does")
    assert first["nextrun_alerted_at"] is not None, alerts
    assert len(_durable_alert_instants(q)) == 1, (
        "the alert instant is not durable queue state, so the next process cannot "
        "read it and the 24 h cooldown still dies with this one")

    second = _fresh_process_state(monkeypatch)
    assert second["nextrun_alerted_at"] is None, (
        "the fixture did not hand the second process a fresh cooldown")
    for _ in range(fw._STALL_NEXTRUN_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, (
        f"a process with a fresh `_state` re-posted the `next_run` alarm inside "
        f"{fw._STALL_NEXTRUN_ALERT_INTERVAL_SECONDS}s (24 h) of the first one: {alerts}")


async def test_an_expired_watermark_alerts_the_same_board_again(
        board, monkeypatch, tmp_path):
    """Clause 3: persistence is a cooldown, not a mute.

    The durable instant aged past `_STALL_NEXTRUN_ALERT_INTERVAL_SECONDS`, on the
    same still-late un-parked board and in a fresh process: the alarm fires again.
    The ageing goes through `wm_set` with an older instant as the value, which is the
    same write the alarm itself makes — nothing reaches for raw SQL, so what is aged
    is the row the alarm reads and not a copy of it. The second process is what a
    day-old row is actually met by, and it also keeps the in-process cache, which
    still holds the un-aged instant, out of the answer.
    """
    _pacing_constants_unchanged()
    write_task(board, 707, frequency="daily", last_run=None, next_run=_ago(days=2))
    q = WorkQueue(tmp_path / "expired.db")
    assert fw._next_run_stalled(q), "the board is not late, so no alert could fire"

    _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_NEXTRUN_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, f"the first alert did not fire: {alerts}"

    rows = _durable_alert_instants(q)
    assert len(rows) == 1, rows
    key, posted = rows[0]
    aged = posted - dt.timedelta(seconds=fw._STALL_NEXTRUN_ALERT_INTERVAL_SECONDS + 60)
    q.wm_set(fw.NAME, key, aged.isoformat())

    _fresh_process_state(monkeypatch)
    for _ in range(fw._STALL_NEXTRUN_TICKS):
        await fw.tick(q)
    assert len(alerts) == 2, (
        f"a persisted alert instant older than "
        f"{fw._STALL_NEXTRUN_ALERT_INTERVAL_SECONDS}s did not alert again, so the "
        f"watermark is a permanent mute: {alerts}")


async def test_a_parked_only_board_never_spends_the_durable_alert_instant(
        board, monkeypatch, tmp_path):
    """Clause 4: #1661's invariant, moved into the durable row.

    A board whose only late task declares its own park posts nothing, and must also
    leave the durable alert instant unwritten — a suppression that wrote it would
    spend a day of cooldown on a park and mute the first genuinely late task for 24 h
    in a state the whole fleet sits in. The watermark is asserted empty, not assumed
    so, and the phase below shows the cooldown is genuinely unspent rather than
    merely unreadable: an un-parked late task added to the same queue in a fresh
    process alerts on its own streak, having never met a watermark.
    """
    _pacing_constants_unchanged()
    write_task(board, 708, frequency="daily", status="draft", last_run=None,
               next_run=_ago(days=3), parked="parked by Alan's 2026-09-17 ruling")
    q = WorkQueue(tmp_path / "parked.db")

    flagged = fw._next_run_stalled(q)
    assert [(e["id"], bool(fw._parked_note(e))) for e in flagged] == [(708, True)], (
        "the board is not parked-only, so the silence asserted below could be the "
        f"silence of no late task at all: {[(e['id'], e['parked']) for e in flagged]}")
    assert fw._grossly_overdue(q) == [], (
        "the due-ness alarm also sees this board, so an empty alert list here would "
        "say nothing about the nextrun alarm")

    _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_NEXTRUN_TICKS):
        await fw.tick(q)
    assert alerts == [], (
        f"a board whose every late task declares its own park reached the alert "
        f"route: {alerts}")
    assert fw._state["nextrun_alerted_at"] is None, alerts
    assert _durable_alert_instants(q) == [], (
        "the parked-only scan wrote the durable alert instant, so a park has spent "
        "the day's cooldown and the first genuinely late task is muted for 24 h")

    write_task(board, 709, frequency="daily", last_run=None, next_run=_ago(days=2))
    _fresh_process_state(monkeypatch)
    for _ in range(fw._STALL_NEXTRUN_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1 and "709" in alerts[0], (
        f"the un-parked late task did not alert after the parked-only scan, so the "
        f"park did spend the cooldown: {alerts}")
    # The helper can see a row on THIS queue, so the empty assertion above was a
    # finding and not an unreadable store.
    assert len(_durable_alert_instants(q)) == 1, (
        "the alert route wrote no durable instant on this queue either, so "
        "`_durable_alert_instants` returning [] above proved nothing")


async def test_an_absent_or_unreadable_watermark_alerts_as_it_does_today(
        board, monkeypatch, tmp_path):
    """Clause 5: the durable read may be missing or broken, the alarm may not be.

    Both shapes in one node, parametrised by monkeypatching the read itself. With no
    row at all the alarm behaves exactly as it did before persistence — five ticks,
    one alert. With `wm_get` raising, the tick must not raise either: this runs
    inside the scheduler loop, and a `next_run` watermark that cannot be read is a
    reason to say so and alert, not a reason to stop watching the fleet.
    """
    _pacing_constants_unchanged()
    write_task(board, 710, frequency="daily", last_run=None, next_run=_ago(days=2))
    q = WorkQueue(tmp_path / "unreadable.db")
    assert fw._next_run_stalled(q), "the board is not late, so no alert could fire"

    _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    assert _durable_alert_instants(q) == [], "the fixture started with a watermark"

    def _boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(WorkQueue, "wm_get", _boom)
    await fw.tick(q)          # the read is on the alert path; must not raise here
    for _ in range(fw._STALL_NEXTRUN_TICKS - 1):
        await fw.tick(q)
    assert len(alerts) == 1 and "710" in alerts[0], (
        f"an unreadable watermark stopped the alarm rather than being read as "
        f"'never alerted': {alerts}")
