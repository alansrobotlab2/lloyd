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


def test_a_starved_backlog_of_another_source_is_detected(tmp_path):
    """Clause 5 (#2496): the age read is about the queue, not about this watchdog's
    own source.

    The read was `queue.list_items(source=NAME, limit=500)` with `NAME =
    "scheduled-task"`, so `_queue_starving` could only ever see rows the scheduler for
    this box enqueues. The backlog that actually starved on 2026-10-10 was 31 rows of
    `owed-check`, `session-distill`, `board-steward`, `bench-mine` and
    `frontend-probe-canary`: 31 claimable rows at attempts=0, `not_before` NULL on every
    one, oldest 2026-10-09T15:53:31Z — 23h37m — while the seven claims the pool made in
    the twenty minutes after 15:20Z all came from the priority-30 and priority-55 bands.
    The alarm returned 0.0 for every hour of that, and that is why a day of starvation
    passed with nobody told.

    A real `WorkQueue`, not a stand-in: a fake that answers a source-scoped read and a
    source-agnostic read the same way makes this node pass either way. The queue below
    holds ONE source and not one `scheduled-task` row — the exact shape that read as
    healthy for 23 hours — and the age is the measured 23h37m, so the node is the
    incident and not a round number.
    """
    q = WorkQueue(tmp_path / "starved.db")
    row_id = q.enqueue(source="owed-check", kind="owed-check:clause",
                       payload={"item_id": 2358, "apply": True})
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE queue SET enqueued_at=? WHERE id=?",
                     (_ago(hours=23, minutes=37), row_id))
        conn.commit()

    age = fw._queue_starving(q, 3600)
    assert age > 3 * 3600, (
        "a 23h37m owed-check backlog is invisible to the starvation alarm, which is the "
        "blind spot that let #2496 run for a day unreported")
    assert pytest.approx(age, abs=120) == 23 * 3600 + 37 * 60, (
        "the figure reported is not the age of the oldest claimable row in the queue")


def test_the_age_read_is_not_a_window_of_the_newest_rows(tmp_path):
    """The second blind spot in the same line, pinned on its own so it cannot come back
    without spending a node.

    `list_items` ends `ORDER BY enqueued_at DESC LIMIT ?`, so the rows a 500-row window
    drops are the OLDEST ones: the alarm under-reported by exactly as much as the
    backlog had grown, which is the only direction the queue moves when it is failing.
    Source scoping is the node above; every row here is `scheduled-task`, the one source
    the old filter could see, so the window is the only thing standing between this
    fixture and an alarm. 501 rows, the oldest one a day behind the rest: one more than
    the limit that hid it, so the shape under test is the shape that failed.
    """
    q = WorkQueue(tmp_path / "window.db")
    recent, old = _ago(minutes=10), _ago(hours=23, minutes=37)
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.executemany(
            "INSERT INTO queue (source, kind, payload_json, priority, state, "
            "enqueued_at) VALUES (?,?,?,?,?,?)",
            [("scheduled-task", "heartbeat", "{}", 30, "queued", recent)] * 500
            + [("scheduled-task", "heartbeat", "{}", 30, "queued", old)])
        conn.commit()
    with sqlite3.connect(str(q.db_path)) as conn:
        held = conn.execute("SELECT count(*) FROM queue WHERE state='queued'").fetchone()[0]
    assert held == 501, f"the fixture is not the shape this node is about: {held} rows"

    age = fw._queue_starving(q, 3600)
    assert pytest.approx(age, abs=120) == 23 * 3600 + 37 * 60, (
        "the oldest queued row was dropped from the age figure by a newest-first window")


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


async def test_a_row_the_grace_suppressed_is_logged_once_and_posts_nothing(
        board, monkeypatch, tmp_path, caplog):
    """#2695 clause 4: when the grace is what closed the door, the tick logs it and posts nothing.

    The nightly board at the tick's instant: #38 `daily`, its newest run record a
    `success` completed 23.9 h ago, its own `next_run` four minutes behind because a
    completion is what moves that stamp and it is mid-run; #42 `daily`, `up_next`,
    `depends_on: 38`, `stale_bypass_hours: 36`, six minutes behind its own stamp, so
    the one-period bound of #421 has it nowhere and #2417's door is the only way in.

    The alerting condition is seeded one tick away — streak at
    `_STALL_NEXTRUN_TICKS - 1` with both cooldown slots clear — so an empty `alerts`
    list below can only be the grace answering, not a watchdog that had not yet counted
    to five. And the cooldown is asserted unspent afterwards: the same shape #1661's
    declared-park suppression keeps, because a benign row must not spend the one alert
    a genuinely slipped night still needs.

    What the log record owes is the reader's half: the dependent, the upstream, and how
    far past the upstream's own `next_run` it sat. Without it a silent alert and a
    suppressed one are the same sentence, which is the failure mode this file's
    declared-park node exists to prevent.
    """
    import logging

    write_task(board, 38, frequency="daily",
               last_run=_ago(hours=23, minutes=54), next_run=_ago(minutes=4))
    _run_record(38, "run_38_20261013_050021",
                completed_at=_ago(hours=23, minutes=54))
    write_task(board, 42, frequency="daily", depends_on=38, stale_bypass_hours=36,
               last_run=_ago(hours=23, minutes=30), next_run=_ago(minutes=6))
    q = WorkQueue(tmp_path / "grace.db")
    # The board has to be live for the silence below to mean anything: the dependent IS
    # past its own `next_run`, its upstream's stamp IS behind, and it is the grace and
    # not an empty board that answers `late_or_dead: False`.
    dep = autonomy._parse_task_file(autonomy._find_task_file(42))
    blocked = fw._upstream_block(dep, autonomy.next_run_gap(dep),
                                 autonomy.dependency_resolution_set(),
                                 dt.datetime.now(dt.timezone.utc))
    assert (blocked["upstream_status"], blocked["late_or_dead"]) == ("success", False), (
        f"the door did not close on the grace for the reason this node is about: {blocked!r}")
    assert blocked["upstream_hours_past"] > 0, (
        "the upstream is not past its own next_run at all, so there was nothing for "
        f"the grace to suppress: {blocked!r}")

    _streak_at_alerting_threshold(monkeypatch, stall_streak=0,
                                  nextrun_streak=fw._STALL_NEXTRUN_TICKS - 1)
    alerts = _watch_alerts(monkeypatch)
    assert fw._last_alert_instant(q, fw._NEXTRUN_WM_KEY, "next_run") is None, (
        "the cooldown slot was already spent by another test's tick, so `no alert` "
        "below would not distinguish the grace from a cooldown")

    # `caplog.records` spans the whole test, and the door read above is a real call to
    # the function that logs — so the count below is measured from here, not from the
    # test's start.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=fw.logger.name):
        await fw.tick(q)

    assert alerts == [], (
        "the nightly line is still posted: a dependent six minutes behind an "
        f"upstream that is mid-run behind a success record: {alerts}")
    assert fw._state["nextrun_streak"] == 0, (
        "the streak kept counting, so the scan still named the chain and the empty "
        f"alert list is the cooldown, not the grace: {fw._state['nextrun_streak']}")
    assert fw._last_alert_instant(q, fw._NEXTRUN_WM_KEY, "next_run") is None, (
        "the suppressed row spent the 24 h alert slot, which is the night a genuinely "
        "slipped chain goes unreported")

    suppressions = [r for r in caplog.records
                    if "did not name the chain" in r.getMessage()]
    assert len(suppressions) == 1, (
        "one suppressed chain must be exactly one log record — zero leaves a silent "
        f"alert indistinguishable from a healthy board: {[r.getMessage() for r in suppressions]}")
    said = suppressions[0].getMessage()
    assert "#42" in said and "#38" in said, (
        f"the record does not name both ends of the chain: {said!r}")
    assert "4 min past its own next_run" in said, (
        "the record does not say how far past the upstream's own next_run it sat, "
        f"which is the number a reader weighs the grace against: {said!r}")
    assert suppressions[0].levelno == logging.WARNING, (
        f"the suppression is not at the level the route precedent uses: {said!r}")


async def test_a_success_upstream_whose_stamp_is_still_ahead_logs_no_suppression(
        board, monkeypatch, tmp_path, caplog):
    """#2950 clause 1: an upstream ahead of its own `next_run` is not inside a grace.

    The record above says "the upstream IS past its own `next_run`, just not by enough
    to name the chain". A stamp still in the FUTURE is the opposite fact, and the gate
    on that record never tested it: `not late_or_dead` is true for every `up_hours <= 0`
    too, because the only bound on lateness in this function is the DEPENDENT's. So the
    live board of #98 -> #24 — #24 `daily`, its stamp ~2.5 h ahead once its own
    completion moved it forward — printed 3,556 records between 2026-10-14 and 2026-10-17
    reading `it sits -167 min past its own next_run, inside the 0.5 h grace`, each one
    attributing a healthy hold to a grace that had nothing to do with it.

    #38 here is that upstream: `daily`, `last_run` 21 h ago, so its `next_run` is 3 h in
    the future, newest run record a `success`. #42 is the shape that gets the door
    consulted at all — `daily`, `depends_on: 38`, `stale_bypass_hours: 36`, six minutes
    behind its own stamp, nowhere near the one-period bound. The verdict must stay
    `late_or_dead: False` (clause 1 moves the record, not the door), and
    `upstream_hours_past` must come back NEGATIVE, which is the datum the record
    misread. Then the tick, seeded one alert away, posts nothing and — the assertion the
    3,556 lines are the premise for — emits no `did not name the chain` record at all.
    """
    import logging

    write_task(board, 38, frequency="daily",
               last_run=_ago(hours=21), next_run=_ago(hours=-3))   # negative age: 3 h ahead
    _run_record(38, "run_38_20261017_050812", completed_at=_ago(hours=21))
    write_task(board, 42, frequency="daily", depends_on=38, stale_bypass_hours=36,
               last_run=_ago(hours=23, minutes=30), next_run=_ago(minutes=6))
    q = WorkQueue(tmp_path / "future-stamp.db")

    dep = autonomy._parse_task_file(autonomy._find_task_file(42))
    blocked = fw._upstream_block(dep, autonomy.next_run_gap(dep),
                                 autonomy.dependency_resolution_set(),
                                 dt.datetime.now(dt.timezone.utc))
    assert (blocked["upstream_status"], blocked["late_or_dead"]) == ("success", False), (
        "the door did not close on the future-stamped upstream, so the silence below is "
        f"not this clause answering: {blocked!r}")
    assert (blocked["upstream_hours_past"] is not None
            and blocked["upstream_hours_past"] < 0), (
        "the fixture's upstream is not ahead of the tick, so this board holds nothing for "
        f"the bound to reject: {blocked!r}")

    _streak_at_alerting_threshold(monkeypatch, stall_streak=0,
                                  nextrun_streak=fw._STALL_NEXTRUN_TICKS - 1)
    alerts = _watch_alerts(monkeypatch)
    assert fw._last_alert_instant(q, fw._NEXTRUN_WM_KEY, "next_run") is None, (
        "the cooldown slot was already spent by another test's tick, so `no alert` "
        "below would not distinguish the grace from a cooldown")

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=fw.logger.name):
        await fw.tick(q)

    said = [r.getMessage() for r in caplog.records
            if "did not name the chain" in r.getMessage()]
    assert said == [], (
        "a `success` upstream whose own `next_run` is still in the future was logged as "
        f"suppressed by the grace — the negative-minute record #2950 is about: {said}")
    assert alerts == [], (
        f"the tick posted an alert for a chain held by a healthy upstream: {alerts}")
    assert fw._last_alert_instant(q, fw._NEXTRUN_WM_KEY, "next_run") is None, (
        "the future-stamped row spent the 24 h alert slot, which is the night a genuinely "
        "slipped chain goes unreported")


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
    # `throughput_day` is #2890's failure-throughput census: the UTC day it last
    # reported, and the only key in here that is a day rather than an instant. It is
    # part of "what a second process starts with" like the rest, so it belongs in this
    # literal — dropping it here would be the guard quietly passing on a state that had
    # gained a key, which is exactly what the assertion below exists to refuse.
    fresh = {"unparseable_scan_at": None, "unparseable_alerted": set(),
             "stall_streak": 0, "stall_alerted_at": None,
             "nextrun_streak": 0, "nextrun_alerted_at": None,
             "nextrun_parked_logged_at": None,
             "throughput_day": None}
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


# ── #2662 — the due-ness alarm's 6 h cooldown died with the process that set it ──
#
# `_state["stall_alerted_at"]` was in-process only, so a restart answered "never
# alerted", the streak recount from `_STALL_ALARM_TICKS` started at zero, and the same
# continuing stall posted again. Re-counted on 2026-10-13 against the anchor #1797's
# tripwire names — `grep -h '^\S* \S* \[ERROR\] lloyd-workers.fleet_watchdog: autonomy
# scheduler may be stalled' ~/lloyd-data/logs/server.err*` — the corpus holds 20 posts
# over 41.9 h (2026-10-11 16:17:42 → 2026-10-13 10:14:02) and ALL 19 consecutive gaps
# land inside `_STALL_ALERT_INTERVAL_SECONDS`: the largest is 4.52 h, so the interval
# held zero times out of nineteen. The smallest, 0.80 h, is one stall posting twice —
# 16:17:42 and 17:05:56 both read `oldest claimable queue item is 185 min old` — and 31
# `Worker pool started` lines sit in the same window. #1797 gated sibling persistence on
# "one real stall posts twice inside its 6 h cooldown"; it posted twice inside it
# nineteen times over, so these nodes are the pre-committed action, not a proposal.
#
# What is deliberately NOT persisted is `nextrun_parked_logged_at`: `:1125` is a
# `logger.warning` that reaches no person, and #1735's rationale is the route to a
# person. One node below pins that the parked-only scan cannot spend the DUE-NESS
# instant either (#1661), which is the half that has to survive persistence.


#: The age of the starving head in this section. 6 h is over 3x the widest
#: `max_duration_seconds` this source has carried (3600 s since 2026-09-03), so the
#: queue starves under the live config and under the 1800 s fallback alike — and
#: `_starving_queue` asserts the figure it measured instead of assuming one.
_STARVING_HEAD_MINUTES = 360


def _starving_queue(tmp_path, name: str) -> WorkQueue:
    """A real queue the due-ness alarm is entitled to alarm on, and nothing else.

    On a real file for the same reason the two-process nodes above are: what has to be
    survived is the death of the process holding `_state`, so the queue has to outlive
    it. Both attribution asserts are the fixture's promise — with no task file to see,
    `_grossly_overdue` and `_next_run_stalled` answer nothing, so every alert counted
    in a node below can only be the due-ness starve clause. Call this BEFORE writing
    any task to the board, or the second assert sees that task.
    """
    db = _aged_queued_row(tmp_path / name, minutes=_STARVING_HEAD_MINUTES)
    q = WorkQueue(db)
    assert fw._queue_starving(q, fw._max_duration_seconds()) > 0, (
        f"a {_STARVING_HEAD_MINUTES} min old claimable row is not starvation at "
        f"max_duration_seconds={fw._max_duration_seconds()}, so no alert below fires "
        "and every silence asserted here would be vacuous")
    assert fw._grossly_overdue(q) == [] and fw._next_run_stalled(q) == [], (
        "the board is not empty of task-side findings, so an alert counted below "
        "would not be attributable to the due-ness starve clause")
    return q


def _names_the_starving_head(msg: str) -> bool:
    """Does this alert read as the due-ness starve clause, naming the head's age?

    Matched on the sentence rather than on the substring "stalled" so a `next_run`
    alert — which begins "N task(s) more than one period past their own next_run" —
    could never satisfy it, and on the printed minutes so an alert that named a
    different cause could not either.
    """
    import re
    m = re.search(r"autonomy scheduler may be stalled: .*oldest claimable queue item "
                  r"is (\d+) min old", msg)
    return bool(m) and int(m.group(1)) >= _STARVING_HEAD_MINUTES - 1


async def test_a_restart_inside_the_cooldown_does_not_re_post_the_due_ness_alarm(
        board, monkeypatch, tmp_path):
    """Clause 1: one alert per 6 h, measured across two processes on one queue.

    The first process carries the assertion the class rule demands before any silence
    is claimed — exactly one `autonomy scheduler may be stalled`, naming the starving
    head by its age. Against that, a second process holding the module's DEFAULT
    surveillance state ticks `_STALL_ALARM_TICKS` again over the SAME starving queue:
    its streak reaches the threshold on its own, so the only thing that can hold the
    message back is the instant the first process recorded.

    At base this node was the 20-post corpus in miniature: the second process re-posted,
    which is exactly the 0.80 h between two posts naming the same 185 min head.
    """
    _pacing_constants_unchanged()
    q = _starving_queue(tmp_path, "due-ness-restart.db")

    first = _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1 and _names_the_starving_head(alerts[0]), (
        f"the first process did not post exactly one due-ness alert naming the "
        f"starving head, so the silence below would be measuring nothing: {alerts}")
    assert first["stall_alerted_at"] is not None, alerts

    second = _fresh_process_state(monkeypatch)
    assert second["stall_alerted_at"] is None, (
        "the fixture did not hand the second process a fresh cooldown")
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, (
        f"a process with a fresh `_state` re-posted the due-ness alert inside "
        f"{fw._STALL_ALERT_INTERVAL_SECONDS}s (6 h) of the first one — the defect that "
        f"posted 20 times in 41.9 h with all 19 gaps inside the cap: {alerts}")


async def test_the_due_ness_alert_instant_is_durable_queue_state_under_its_own_key(
        board, monkeypatch, tmp_path):
    """Clause 2: the hold above is a row on the queue, not a memory in the process.

    Read back through the queue's public API (`wm_keys`/`wm_get`) on a SECOND
    `WorkQueue` object over the same file — a new connection over the same database is
    as close as a test comes to the new process the corpus actually had, and it is the
    boundary the value has to cross. Two things are pinned, because "a row exists" is
    not the claim: the instant is the one it posted, so the durable row is correct and
    not merely present; and it is under a key that is NOT `_NEXTRUN_WM_KEY`, because
    two alarms sharing one row would let a 24 h alert mute a 6 h one and back, and one
    global instant PER ALARM is the policy this file has always held.
    """
    _pacing_constants_unchanged()
    q = _starving_queue(tmp_path, "due-ness-durable.db")

    _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1 and _names_the_starving_head(alerts[0]), alerts
    posted = fw._state["stall_alerted_at"]
    assert posted is not None, alerts

    rows = _durable_alert_instants(WorkQueue(q.db_path))
    assert len(rows) == 1, (
        f"the due-ness instant is not one readable instant under source {fw.NAME!r}: "
        f"{rows} — a restart cannot read what the public API does not show")
    key, durable = rows[0]
    assert key != fw._NEXTRUN_WM_KEY, (
        f"the due-ness alarm wrote the `next_run` alarm's row ({key}), so the two "
        "cooldowns are one row and either alarm can spend the other's silence")
    assert durable == posted, (
        f"the row holds {durable} but the alert posted {posted} — the durable instant "
        "is not the instant a person was told about")


async def test_the_persisted_due_ness_instant_is_a_cooldown_not_a_mute(
        board, monkeypatch, tmp_path):
    """Clause 3: an aged row alerts again, and so does no row at all.

    Both shapes fail the same way if persistence is written as a mute, so both are
    here. Aged: the row is pushed past `_STALL_ALERT_INTERVAL_SECONDS` through `wm_set`
    with an older instant as its value, which is the same write the alarm itself makes
    — no raw SQL against `workers.db`, so what is aged is the row the alarm reads and
    not a copy of it. Absent: a queue that never wrote a row, the shape every one of
    the 20 corpus posts met, and the behaviour the alarm has to keep byte-identical. A
    fresh process before each, because the in-process cache would otherwise answer
    without the row being consulted at all.
    """
    _pacing_constants_unchanged()

    q = _starving_queue(tmp_path, "due-ness-aged.db")
    _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, f"the first alert did not fire: {alerts}"
    rows = _durable_alert_instants(q)
    assert len(rows) == 1, rows
    key, posted = rows[0]
    aged = posted - dt.timedelta(seconds=fw._STALL_ALERT_INTERVAL_SECONDS + 60)
    q.wm_set(fw.NAME, key, aged.isoformat())

    _fresh_process_state(monkeypatch)
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 2, (
        f"a persisted due-ness instant older than "
        f"{fw._STALL_ALERT_INTERVAL_SECONDS}s (6 h) did not alert again, so the "
        f"watermark is a permanent mute: {alerts}")

    absent = _starving_queue(tmp_path, "due-ness-absent.db")
    assert _durable_alert_instants(absent) == [], "the fixture started with a row"
    _fresh_process_state(monkeypatch)
    never_alerted = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(absent)
    assert len(never_alerted) == 1 and _names_the_starving_head(never_alerted[0]), (
        f"with no durable row the alarm stopped firing, which is a mute wearing a "
        f"cooldown's clothes: {never_alerted}")


async def test_an_unreadable_due_ness_watermark_still_alerts_the_starving_queue(
        board, monkeypatch, tmp_path):
    """Clause 4: a watermark that cannot be read reads as "never alerted".

    `wm_get` raises on every call. The tick has to complete anyway: this runs inside
    the loop that enqueues the whole fleet, so an alarm's own read failure taking that
    loop down would be the larger incident, and muting the alarm because the cooldown
    row is unreadable would hide the stall the row exists to pace. Same shape as the
    `next_run` node above, which is the point — the two alarms now share one read
    function and have to share its failure behaviour too.
    """
    _pacing_constants_unchanged()
    q = _starving_queue(tmp_path, "due-ness-unreadable.db")

    def _boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(WorkQueue, "wm_get", _boom)
    _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)          # the read is on the alert path; must not raise here
    for _ in range(fw._STALL_ALARM_TICKS - 1):
        await fw.tick(q)
    assert len(alerts) == 1 and _names_the_starving_head(alerts[0]), (
        f"an unreadable watermark stopped the due-ness alarm rather than being read as "
        f"'never alerted': {alerts}")


async def test_a_parked_only_scan_leaves_the_due_ness_cooldown_untouched(
        board, monkeypatch, tmp_path):
    """Clause 5: #1661 holds through persistence — a park cannot buy 6 h of silence.

    Phase one is a real starvation, and it is there to give phase two something to
    preserve: without a due-ness instant on the queue, "unchanged" would be comparing
    an absent row with itself. Phase two retires the starved row — raw SQL on a fixture
    row the fixture itself aged — and leaves one late task that declares its own park,
    which is the steady state of this fleet (#68 is parked by a standing ruling). It
    then ticks `_STALL_NEXTRUN_TICKS`, enough for the scan to CONFIRM the lateness and
    reach the parked branch, so the silence asserted below is a suppression and not an
    alarm with nothing to see.

    Three things then hold together: the parked-only scan posts nothing; the due-ness
    instant is identical to the one phase one wrote, key included; and the count of
    durable instants under this source stays at one — which is also the assertion that
    the parked branch did not take up persisting its own log slot here, the scope call
    #1735's docstring already made and this item leaves to a human ruling.
    """
    _pacing_constants_unchanged()
    q = _starving_queue(tmp_path, "due-ness-parked.db")

    _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1 and _names_the_starving_head(alerts[0]), (
        f"phase one spent no due-ness alert, so phase two has no cooldown instant to "
        f"leave alone: {alerts}")
    before = _durable_alert_instants(q)
    assert len(before) == 1, before

    write_task(board, 711, frequency="daily", status="draft", last_run=None,
               next_run=_ago(days=3), parked="parked by Alan's 2026-09-17 ruling")
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE queue SET state='done' WHERE state='queued'")
        conn.commit()
    flagged = fw._next_run_stalled(q)
    assert [(e["id"], bool(fw._parked_note(e))) for e in flagged] == [(711, True)], (
        "phase two is not a parked-only board, so an empty alert list would say "
        f"nothing about the suppression: {flagged}")
    assert fw._queue_starving(q, fw._max_duration_seconds()) == 0.0, (
        "the queue is still starving, so phase two's silence is the starvation alarm "
        "declining to fire rather than a park declining to spend anything")

    _fresh_process_state(monkeypatch)
    for _ in range(fw._STALL_NEXTRUN_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, (
        f"a board whose every late task declares its own park reached the alert "
        f"route: {alerts}")
    assert _durable_alert_instants(q) == before, (
        "the parked-only scan advanced or replaced the due-ness cooldown instant, so a "
        "park has bought the 6 h of silence the next genuinely overdue task needs")
    assert fw._state["nextrun_parked_logged_at"] is not None, (
        "the parked branch never took its log slot, so the phase above may have ticked "
        "past the branch it is supposed to be suppressing")


# ── #2342 — the stall line may not blame dispatch for a run the records already name ──
#
# The scan that raises the `next_run` alarm reasoned from row and queue state only:
# `hours` from `autonomy.next_run_gap`, `hold` from `autonomy.hold_reason`, `queued`
# from live queue rows. `git grep -n 'autonomy-runs|workers.db|list_runs' --
# workers/fleet_watchdog.py` returned exactly one hit before this section existed, at
# the docstring of `_last_alert_instant` (then `_last_nextrun_alert`) warning against
# raw SQL — so the file read no
# run record. That is how a task that HAD run got the sentence "nothing holds it;
# dispatch did not enqueue it": `~/lloyd-data/autonomy-runs/79/run_79_20261006_051105.md`
# carries `status: success` / `completed_at: 2026-10-06T05:17:08+00:00` while
# `autonomy/79-retention-sweep.md` did not advance to it (vault commit `ba3ab3fc` moved
# `last_run` from `2026-10-05T05:05:18` straight to `2026-10-07T05:02:42`), and the
# alert at `2026-10-06 22:00:28` in `logs/server.err.1` blamed dispatch for it.
#
# Four nodes, four shapes. Each is a row on disk plus a run record on disk, and the
# string a person reads: the seam is file-to-alert, because the thing that was wrong
# was a sentence, and a stub that returned a verdict would have let the sentence stay.


def _stamp(**delta) -> dt.datetime:
    """An instant measured off the same wall clock the scan reads, as a datetime."""
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(**delta)


def _run_record(task_id, run_id, *, status="success", completed_at):
    """One run record on disk, in the shape `app.autonomy._write_run_record` writes.

    Written by hand, not by the writer: `_write_run_record` is reached from inside
    `run_task`, and the fact under test is the instant the watchdog has to read back
    out of the front matter — `completed_at`, or `started_at` when it is absent.
    """
    d = autonomy.AUTONOMY_RUNS_DIR / str(task_id)
    d.mkdir(parents=True, exist_ok=True)
    fm = {"run_id": run_id, "task_id": task_id, "status": status,
          "started_at": completed_at, "completed_at": completed_at,
          "duration_seconds": 363.0, "summary": "ok", "trigger": "scheduler"}
    (d / f"{run_id}.md").write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# Run\n",
        encoding="utf-8")


def test_a_newer_success_record_makes_the_stall_line_name_a_lost_completion(
        board, tmp_path):
    """Clause 1: #79's own shape, read the way the alert says it.

    The row is a day and an hour behind (`next_run` 25 h old against a `daily`
    frequency, so the strict one-period bound of #421 is crossed by an hour) and its
    `last_run` is 26 h old, while a success record completed 20 h ago — between the
    two row stamps, exactly as `run_79_20261006_051105.md` sits between
    `last_run: 2026-10-05T05:05:18` and the `next_run: 2026-10-06T05:00:00` that was
    24 h stale when the false alert fired.

    The claim is about the sentence, not about a flag: the line has to name a lost
    row completion and quote the record's own instant, and must not contain the
    dispatch accusation. The assertion is on `_nextrun_alert_message`'s output —
    the text `discord_alert` is handed — because a green on an internal boolean
    would still leave the wrong sentence shipping.
    """
    rec_at = _stamp(hours=20).isoformat()
    write_task(board, 79, name="Retention Sweep", frequency="daily",
               last_run=_stamp(hours=26).isoformat(),
               next_run=_stamp(hours=25).isoformat())
    _run_record(79, "run_79_20261006_051105", completed_at=rec_at)

    flagged = {e["id"]: e for e in fw._next_run_stalled(WorkQueue(tmp_path / "lost.db"))}
    assert 79 in flagged, "the row is 25 h past a daily next_run, so it must be flagged"
    msg = fw._nextrun_alert_message(list(flagged.values()))

    assert "row bookkeeping lost a completion" in msg, (
        "the alert still tells the reader the row's story as though it were the "
        f"fleet's: {msg!r}")
    assert rec_at.split("+")[0] in msg, (
        f"the disagreement is not quotable without the record's instant: {msg!r}")
    assert "run_79_20261006_051105" in msg, (
        f"a reader cannot go and look at the record the alert cites: {msg!r}")
    assert "dispatch did not enqueue it" not in msg, (
        f"the cause claim a run record on disk already contradicts: {msg!r}")


def test_a_stall_with_no_newer_record_still_blames_dispatch(board, tmp_path):
    """Clause 2: the alarm keeps catching the failure it was built for.

    The negative half, and the half that keeps clause 1 honest: an older success
    record is not a lost completion. `last_run` tracks the last SUCCESS (#86), so a
    record completed 50 h ago beside a `last_run` 26 h old is the row agreeing with
    its own records, and the row being 25 h past `next_run` with nothing holding it
    IS a dispatch non-enqueue. With the run record absent altogether the sentence is
    the same one for the same reason — so a node that only seeded "no records at all"
    would be satisfied by a fix that simply never read them.
    """
    write_task(board, 711, frequency="daily",
               last_run=_stamp(hours=26).isoformat(),
               next_run=_stamp(hours=25).isoformat())
    _run_record(711, "run_711_20260920_050000", completed_at=_stamp(hours=50).isoformat())

    q = WorkQueue(tmp_path / "real.db")
    flagged = {e["id"]: e for e in fw._next_run_stalled(q)}
    entry = flagged[711]
    assert not entry["hold"], (
        f"a hold reason would make the dispatch sentence unreachable for the wrong "
        f"reason: {entry['hold']!r}")
    assert entry["queued"] is False, (
        "a live queue row belongs to the other branch of `_hold_note`")

    msg = fw._nextrun_alert_message([entry])
    assert "nothing holds it; dispatch did not enqueue it" in msg, (
        f"the real dispatch failure lost the sentence that exists to catch it: "
        f"{msg!r}")
    assert "lost a completion" not in msg, (
        f"a record older than the row's own stamp is agreement, not a disagreement: "
        f"{msg!r}")


def test_a_row_with_no_next_run_is_named_instead_of_skipped(board, tmp_path):
    """Clause 3: the #96 shape, which is invisible to both stall alarms today.

    `autonomy/96-djev-name-prior-probe.md` carries no `last_run`, `last_attempt` or
    `next_run` key at all while `autonomy-runs/96/run_96_20261005_062825.md` carries
    `status: success` / `completed_at: 2026-10-05T06:29:35.824210+00:00`. Because
    `app/autonomy.py` computes `past_next_run` as `bool(interval and hours_past_next
    is not None and ...)`, a row with no `next_run` can never satisfy the predicate,
    and `next_run_gap` answers `None` in its elapsed fields — "no verdict, not a
    healthy verdict". A row that cannot be measured is not a row that is fine, so the
    scan has to say so rather than walk past it.

    Task 97 is the discrimination: an unstamped row with NO run record is a task that
    has simply never run, and flagging that would make this branch the new #421 noise.
    """
    write_task(board, 96, name="djev name prior probe", frequency="weekly",
               last_run=None, next_run=None)
    rec_at = _stamp(days=2).isoformat()
    _run_record(96, "run_96_20261005_062825", completed_at=rec_at)
    write_task(board, 97, frequency="weekly", last_run=None, next_run=None)

    flagged = {e["id"]: e for e in fw._next_run_stalled(WorkQueue(tmp_path / "stampless.db"))}
    assert 96 in flagged, (
        "#96 has a success record and no row stamp, and the gap predicate cannot see "
        f"either fact: {sorted(flagged)}")
    assert 97 not in flagged, (
        "an unstamped row with no run record has never run: naming it is the noise "
        "#421 was filed about")

    entry = flagged[96]
    assert entry["hours"] is None, (
        "an unstamped row has no elapsed figure to report, and inventing one is the "
        f"defect: {entry['hours']!r}")
    msg = fw._nextrun_alert_message([entry])
    assert "#96 (djev name prior probe)" in msg, (
        f"the task is still missing from the alert: {msg!r}")
    assert "no next_run" in msg, (
        f"the reason must be that the row has no stamps, not a guessed cause: {msg!r}")
    assert rec_at.split("+")[0] in msg, (
        f"the records are the only account this row has, so quote them: {msg!r}")
    assert "dispatch did not enqueue it" not in msg, (
        f"a row with no stamps cannot support a claim about its dispatch: {msg!r}")


async def test_unreadable_run_records_fall_back_and_the_alarm_still_fires(
        board, monkeypatch, tmp_path):
    """Clause 4: a record store that cannot be read must not silence the alarm.

    Two shapes of "cannot be read", one outcome. With `AUTONOMY_RUNS_DIR` pointing at
    a plain file there is no directory to walk, and with a run file whose front
    matter is not YAML there is no instant to trust; both answer "nothing", which is
    also the answer of a task that has no records, so the line falls back to the
    wording it has always used and the tick still reaches a person.

    The fallback is the same STRING by design and that is a deliberate reading of the
    clause: "falls back to the current wording and the alarm still fires" — an alarm
    that invented a fresh sentence for an unreadable store would be making a claim
    about records it did not read, which is the defect this section is about, pointed
    the other way. What is asserted here instead is that the reader's silence is not
    a verdict: the alert still arrives, with the task named.
    """
    _pacing_constants_unchanged()
    write_task(board, 712, frequency="daily",
               last_run=_stamp(hours=26).isoformat(),
               next_run=_stamp(hours=25).isoformat())

    # Shape one: no record directory at all.
    q = WorkQueue(tmp_path / "norecords.db")
    assert not (autonomy.AUTONOMY_RUNS_DIR / "712").is_dir(), (
        "task #712 already has a record directory, so 'no records to read' is not "
        "the shape this fixture set up")
    _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)
    for _ in range(fw._STALL_NEXTRUN_TICKS - 1):
        await fw.tick(q)
    assert len(alerts) == 1 and "712" in alerts[0], (
        f"a missing record directory silenced the alarm: {alerts}")
    assert "nothing holds it; dispatch did not enqueue it" in alerts[0], (
        f"the fallback wording is not the wording the alarm had before: {alerts[0]!r}")

    # Shape two: a record file that is there and cannot be parsed.
    d = autonomy.AUTONOMY_RUNS_DIR / "712"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_712_20261006_050000.md").write_text(
        "---\nrun_id: [unclosed\n", encoding="utf-8")
    flagged = {e["id"]: e for e in fw._next_run_stalled(q)}
    msg = fw._nextrun_alert_message([flagged[712]])
    assert "dispatch did not enqueue it" in msg, (
        f"an unparsable record was read as a disagreement it cannot support: {msg!r}")
    assert "lost a completion" not in msg, (
        f"a front matter that does not parse cannot quote an instant: {msg!r}")


# ── The lost-occurrence ledger (#2552) ───────────────────────────────────────
#
# `watermarks` is `PRIMARY KEY (source, key)`: one row per key, updated in place.
# That is why the pause itself is unanswerable after the fact — #2498 owns that
# half — and why the ledger below is keyed per OCCURRENCE rather than per task:
# "how many cycles did that hold cost" is only computable if each one is a row.
#
# The clock AND the zone are pinned, and both pins are load-bearing. The window is
# `preferred_hours`, machine-local hours (#1437 Defect B), and `autonomy
# ._local_tz()` answers with the zone of whatever box runs the suite — so WHICH
# occurrence `_window_losses` selects is a function of that zone and of the minute
# the suite runs. An earlier version of this section left both unpinned and read the
# recorded instants back through `_to_local`, which made the recorded HOURS true
# anywhere while leaving every row-count assertion below a matter of the machine.
# Pinned, the fixture is the 2026-10-10 incident carrying its own numbers, and the
# assertions name them exactly: at UTC-7 the window [22,23,0,1,2,3,4] is
# 2026-10-10T05:00Z -> 2026-10-10T12:00Z, the occurrence the hold really covered.

#: The box's own offset that week. The pause row's persisted `updated_at` is
#: 17:57:57Z, and the window it starved opened at 22:00 local the next morning —
#: which is 05:00Z only at UTC-7.
_INCIDENT_TZ = dt.timezone(dt.timedelta(hours=-7), "UTC-7")
#: The instant the release still had not come: 3 h 10 min after the window closed,
#: local hour 8 — the shape `_is_task_due` refused in.
_INCIDENT_NOW = dt.datetime(2026, 10, 10, 15, 10, tzinfo=dt.timezone.utc)
#: The persisted `(_pool, operator_paused).updated_at` of that hold, verbatim.
_INCIDENT_SINCE = "2026-10-09T17:57:57.490185+00:00"
#: Task #42's own window, read verbatim from ~/obsidian/autonomy/42-*.md.
_LOSS_WINDOW = [22, 23, 0, 1, 2, 3, 4]
#: The occurrence that hold starved, in both clocks it is written in.
_OCC_OPEN = "2026-10-10T05:00:00+00:00"      # 22:00 local on 2026-10-09
_OCC_CLOSE = "2026-10-10T12:00:00+00:00"     # 05:00 local on 2026-10-10
_OCC_KEY = "window_lost:42:2026-10-09T22:00"


def _ledger_keys(q) -> list[str]:
    """Every loss row in the store, and nothing else.

    `WINDOW_LOSS_WM_SOURCE` is the source the scheduled-task source already uses for
    its other watermarks — `nextrun_alerted_at` lives under it — so the ledger's own
    namespace is the key's prefix, and a whole-source listing would count the alarm's
    stamps beside the losses."""
    return sorted(k for k in q.wm_keys(autonomy.WINDOW_LOSS_WM_SOURCE)
                  if k.startswith(autonomy.WINDOW_LOSS_WM_PREFIX))


def _held_over_a_closed_window(tmp_path, board, monkeypatch, *, held=True):
    """The 2026-10-10 board: task 42, its window closed, the pool held across it.

    Clock, zone and local hour are pinned to the incident's (`_utcnow`,
    `_local_tz`, `_local_hour`), so the occurrence `_window_losses` selects is the
    one named in `_OCC_*` on any box at any minute. `held=False` is the same board
    with no operator pause row at all.

    The hold is seeded the way the release leaves it: the queue's
    `(_pool, operator_paused)` row, whose `updated_at` IS the since-instant
    (`workers/pool.py:550-580`), aged with raw SQL because `wm_set` cannot age a
    row through the public API — the very workaround `wm_updated_at`'s own
    docstring names. So the assertion beside it measures the incident's shape, not
    a fixture invented for the test. And because that `since` is the incident's
    real one, it lands AFTER the two earlier occurrences inside the recorder's
    lookback had opened: a block that began inside an occurrence left that
    occurrence unblocked minutes, and the recorder must not claim it as their
    cause — which is what `keys == [_OCC_KEY]` then measures.
    """
    import workers.pool as pool_mod

    monkeypatch.setattr(autonomy, "_utcnow", lambda: _INCIDENT_NOW)
    monkeypatch.setattr(autonomy, "_local_tz", lambda: _INCIDENT_TZ)
    # 15:10Z read at UTC-7. `_is_preferred_hour` evaluates the window against
    # `_local_hour()` rather than the instant it is given (#2552's own finding), so
    # the hour is pinned apart from the clock; 8 is the hour the 10-10 refusal
    # actually evaluated, and it is outside the window.
    monkeypatch.setattr(autonomy, "_local_hour", lambda: 8)
    # `last_run` is the nightly reflection's last completion before the loss (two
    # days, so the elapsed gate is satisfied) and `next_run` the retry stamp its
    # file carried when the window closed.
    write_task(board, 42, preferred_hours=_LOSS_WINDOW,
               last_run="2026-10-08T09:18:03+00:00",
               next_run="2026-10-09T09:18:03+00:00")
    q = WorkQueue(tmp_path / "loss.db")
    if held:
        q.wm_set(pool_mod.PAUSE_WM_SOURCE, pool_mod.PAUSE_WM_KEY, "1")
        conn = sqlite3.connect(str(tmp_path / "loss.db"))
        conn.execute("UPDATE watermarks SET updated_at=? WHERE source=? AND key=?",
                     (_INCIDENT_SINCE,
                      pool_mod.PAUSE_WM_SOURCE, pool_mod.PAUSE_WM_KEY))
        conn.commit()
        conn.close()
    return q


def test_a_window_closed_under_a_hold_is_recorded_with_every_name_it_owes(
        tmp_path, board, monkeypatch):
    """Clause 4. The record names the task, the occurrence and the block, durably.

    Durable means the same table the pause lives in — `watermarks` — so a reader
    holding `workers.db` and no log can answer which cycles a hold cost. On 2026-10-10
    the loss existed only as prose in `memory/2026-10-10.md:22`, 17 hours after the
    window closed, from an alarm structurally one period late; this row is the
    machine-readable half that was missing.
    """
    q = _held_over_a_closed_window(tmp_path, board, monkeypatch)
    rows = fw._window_losses(q)

    assert [r["task_id"] for r in rows] == [42], rows
    row = rows[0]
    assert row["name"] == "task42"
    assert row["window_hours"] == "00-04,22-23", row

    # The occurrence, named exactly: the UTC instants the row carries, and the
    # local hours the window was declared in. The length is the window's own.
    assert row["window_open"] == _OCC_OPEN, row
    assert row["window_close"] == _OCC_CLOSE, row
    o = dt.datetime.fromisoformat(row["window_open"])
    c = dt.datetime.fromisoformat(row["window_close"])
    assert autonomy._to_local(o).hour == 22, row
    assert autonomy._to_local(c).hour == 5, row
    assert c - o == dt.timedelta(hours=len(_LOSS_WINDOW)), row
    assert row["recorded_at"] == _INCIDENT_NOW.isoformat(), row

    keys = _ledger_keys(q)
    # One row, for the one occurrence the hold covered. The two earlier
    # occurrences inside the lookback opened before the hold began, and the
    # key's own local opening hour is what tells them apart.
    assert keys == [_OCC_KEY], keys
    persisted = json.loads(q.wm_get(autonomy.WINDOW_LOSS_WM_SOURCE, keys[0]))
    assert persisted == row, "what the tick returned is not what is on disk"


def test_the_record_names_the_block_and_outlives_the_catch_up_it_granted(
        tmp_path, board, monkeypatch):
    """Clause 4's other half: the pause's own `since`, and a row the run does not eat.

    `pause_since` is the `(_pool, operator_paused)` row's `updated_at` read back — the
    block's identity, not a description of it. And the row survives the completion it
    authorised: what expires is the authorisation, which rides on the task's own
    `last_run`, never the record of the loss.
    """
    import workers.pool as pool_mod

    q = _held_over_a_closed_window(tmp_path, board, monkeypatch)
    rows = fw._window_losses(q)
    assert len(rows) == 1
    assert rows[0]["pause_since"] == _INCIDENT_SINCE, rows[0]
    assert rows[0]["pause_since"] == pool_mod.operator_pause_state(q)["since"], (
        "the record must name the block's own since-instant, not a restatement")

    before = q.wm_get(autonomy.WINDOW_LOSS_WM_SOURCE, _OCC_KEY)

    # The catch-up runs: 3 h 09 min after the occurrence closed, so the completion
    # stamp post-dates it. Nothing else about the task moved.
    write_task(board, 42, preferred_hours=_LOSS_WINDOW,
               last_run="2026-10-10T15:09:00+00:00",
               next_run="2026-10-10T22:00:00-07:00")
    assert fw._window_losses(q) == [], (
        "a task that has run since the window closed was recorded as a fresh loss")
    assert q.wm_get(autonomy.WINDOW_LOSS_WM_SOURCE, _OCC_KEY) == before, (
        "the ledger row was rewritten or removed by the very run it authorised; it "
        "is the record of what was lost, not a token the fix consumes")

    # Same store redirect the scheduler tests use, so the row is read by
    # `window_catch_up_row` the way the engine reads it.
    monkeypatch.setattr("workers.queue.configured_db_path",
                        lambda: tmp_path / "loss.db")
    task = [t for t in autonomy._all_runnable_tasks() if t.get("id") == 42][0]
    assert autonomy.window_catch_up_row(task, now=_INCIDENT_NOW) is None, (
        "the grant survived the run it granted: a standing out-of-window bypass")


def test_no_loss_is_recorded_without_a_block_covering_the_occurrence(
        tmp_path, board, monkeypatch):
    """No witnessed block, no row, nothing to authorise.

    Every row in this ledger is a licence to dispatch outside a window, so writing
    one where the queue names no hold would widen that window by the back door — the
    move the item forbids in its own constraints. A task unrun because it failed, or
    because it simply ran inside the window, must produce nothing here.
    """
    q = _held_over_a_closed_window(tmp_path, board, monkeypatch, held=False)
    assert fw._window_losses(q) == []
    assert _ledger_keys(q) == [], (
        "a loss row appeared with no block on the queue to cover the occurrence")


class _LedgerUnreadable:
    """A queue that answers every watermark read but the loss ledger's.

    Delegation is the point. `operator_pause_state` swallows its own failures and
    reports "not held" (`workers/pool.py:571-575`), so a queue that raised on EVERY
    read would be turned away at `_window_losses`' own early exit and the test
    below would be grading pool's guard instead of the recorder's. Only the loss
    ledger's source refuses, and `refused` records the keys it refused, so the test
    can also see that the recorder reached the ledger at all.
    """

    def __init__(self, real):
        self._real = real
        self.db_path = real.db_path
        self.refused: list[str] = []

    def wm_get(self, source, key):
        if source == autonomy.WINDOW_LOSS_WM_SOURCE:
            self.refused.append(key)
            raise sqlite3.OperationalError("database is locked")
        return self._real.wm_get(source, key)

    def wm_updated_at(self, source, key):
        return self._real.wm_updated_at(source, key)

    def wm_keys(self, source):
        return self._real.wm_keys(source)

    def wm_set(self, source, key, value, **kw):
        if source == autonomy.WINDOW_LOSS_WM_SOURCE:
            self.refused.append(key)
            raise sqlite3.OperationalError("database is locked")
        return self._real.wm_set(source, key, value, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)


async def test_the_recorder_runs_from_the_tick_and_never_silences_the_alarms(
        tmp_path, board, monkeypatch):
    """The wiring: a recorder nothing calls is a file format, not a feature.

    `_window_losses` runs from `tick`, ahead of the stall alarms. Two things about
    that call are observable and both are asserted: the row it owes lands in the
    store, and the alarm sharing the tick still reaches a person — the new call sits
    between `tick` and the `next_run` detector, so a ledger write that cost the
    alarm its alert would show up here as an empty list. The incident board puts
    task 42 a period past its own `next_run`, so the alarm has something to say, and
    the streak starts below the threshold, so it says it on exactly the fifth tick.

    The second half is the failure mode: the same board with a ledger whose reads and
    writes both explode. `refused` proves the recorder reached that ledger — the
    refusal is on the loss path inside `_window_losses`, not on the pause read that
    precedes it — and the swallow is what the empty return and the tick that does not
    raise measure. #1682 is an alarm that could not report its own silencing; a loss
    ledger that skipped both stall alarms would repeat it.
    """
    _pacing_constants_unchanged()
    q = _held_over_a_closed_window(tmp_path, board, monkeypatch)
    assert [e["id"] for e in fw._next_run_stalled(q)] == [42], (
        "the incident board is invisible to the `next_run` detector, so the alert "
        "count below would measure a fixture rather than the alarm")

    _streak_at_alerting_threshold(monkeypatch, stall_streak=0, nextrun_streak=0,
                                  nextrun_alerted_at=None)
    alerts = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_NEXTRUN_TICKS - 1):
        await fw.tick(q)
    assert alerts == [], (
        f"the alarm fired before its own {fw._STALL_NEXTRUN_TICKS} ticks, or the "
        f"ledger write changed its pacing: {alerts}")
    await fw.tick(q)
    assert len(alerts) == 1 and "42" in alerts[0], (
        f"the tick that records a lost occurrence silenced the stall alarm: {alerts}")
    assert _ledger_keys(q) == [_OCC_KEY], (
        "the recorder never ran from the tick")

    broken = _LedgerUnreadable(q)
    assert fw._window_losses(broken) == [], (
        "the recorder let a locked ledger escape: the tick that cannot write a loss "
        "row would also skip both stall alarms, which is #1682's silence")
    assert broken.refused == [_OCC_KEY], (
        "the recorder never reached the loss ledger — the exception was raised on a "
        "path some other guard already covers, so the line above graded that guard "
        "instead of `_window_losses`' own")
    await fw.tick(broken)
    assert len(alerts) == 1, (
        f"a second alert arrived once the ledger began refusing: {alerts}")


# ---------------------------------------------------------------------------
# #2615 clause 4 — the stall alarm names the holder kind the journal records.
# ---------------------------------------------------------------------------


def _held_watchdog_queue(tmp_path, *, kind: str):
    """A real workers.db whose pool is held by `kind`, as the alarm will find it."""
    from workers.pool import WorkerPool as _WP
    from workers.queue import WorkQueue as _WQ

    q = _WQ(tmp_path / "workers.db")
    _WP(q, slots=1).pause(True, holder_kind=kind)
    return q


def test_the_stall_alarm_names_the_kind_of_hold_and_keeps_its_hours(tmp_path):
    """Clause 4: a person's hold and a machine's hold must not read as one sentence.

    The sentence used to be `the worker pool is PAUSED by operator` with the hours
    and the since-instant — and `operator_pause_state` already returned the kind,
    unread. That is the gap #2498's findings named as still owed: a reader could not
    tell "someone halted this on purpose" from "a process died holding the fleet",
    which is the one distinction that decides whether to resume it. The hours and the
    instant stay, because they are why the alarm fires at all; the kind is added,
    read from the same row the hold journal takes its `holder_kind` from.
    """
    import re

    person = fw._operator_pause_clause(_held_watchdog_queue(tmp_path / "p", kind="person"))
    machine = fw._operator_pause_clause(_held_watchdog_queue(tmp_path / "m", kind="machine"))

    assert "PAUSED by operator" in person and "PAUSED by operator" in machine, (
        f"{person}\n{machine}")
    assert "(person hold)" in person, person
    assert "(machine hold)" in machine, machine
    assert person != machine, "the two kinds read identically; the clause says nothing"
    for clause in (person, machine):
        assert re.search(r"held \d+\.\d h since ", clause), (
            f"the duration and the since-instant must survive: {clause}")


def test_the_stall_alarm_reads_the_kind_from_the_row_not_from_a_second_counter(tmp_path):
    """No new alarm, no second stall counter: the clause is one read of one row.

    The hold is aged with raw SQL because `wm_set` cannot age a row through the
    public API — the same workaround `wm_updated_at`'s docstring names — so the
    hours the alarm prints are derived from the row's `updated_at`, exactly as before
    this change. A pre-#2498 database with no holder row at all still alarms, and
    alarms as a person's hold, because that is `HOLDER_DEFAULT` and the safe reading.
    """
    import sqlite3

    from workers.pool import PAUSE_WM_KEY, PAUSE_WM_SOURCE, WorkerPool as _WP
    from workers.queue import WorkQueue as _WQ

    q = _WQ(tmp_path / "workers.db")
    _WP(q, slots=1).pause(True, holder_kind="machine")
    since = "2026-10-10T06:24:42+00:00"
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE watermarks SET updated_at=? WHERE source=? AND key=?",
                     (since, PAUSE_WM_SOURCE, PAUSE_WM_KEY))
        conn.commit()

    clause = fw._operator_pause_clause(q)
    assert "(machine hold)" in clause, clause
    assert since in clause, f"the since-instant must be quoted verbatim: {clause}"

    legacy = _WQ(tmp_path / "legacy.db")
    legacy.wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY, "1")
    assert "(person hold)" in fw._operator_pause_clause(legacy), (
        "a database with no holder row alarms as the kind that is never auto-released")


# ── #2692 — the queue-age clause names its source, and an outranked wait says so ──
#
# The command this section's premise rests on, run again on 2026-10-14 against the live
# logs (`~/lloyd-data/logs/server.err*`, eleven rotated files). The second grep is the
# load-bearing part: every post is written TWICE, once as this module's ERROR line and
# once as a `lloyd-server: discord_alert` echo of the same text, so the bare first grep
# counts 44 LINES for 22 POSTS and any figure read off it is double the truth.
#
#   grep -hF 'autonomy scheduler may be stalled' server.err* | grep -F 'fleet_watchdog:'
#
# It answers 22 posts of the due-ness alarm, and 18 of those 22 are the age-only sentence
# — no source named — running from 2026-10-11 16:17:42 (185 min) through
# 2026-10-12 23:58:32 (915 min) to 2026-10-14 02:05:42 (196 min), which is to say it was
# still firing six hours before this line was written. The four files the item grepped
# (`server.err{,.1,.2,.3}`) hold 5 of the 18; the 915 min one is in `server.err.4`.
#
# Two independent signals say those 18 were not stalls. First, the ages FALL across
# consecutive posts on 2026-10-13 — 892, 570, 456, then 185 min — and a head age can only
# fall if the head row is being claimed and replaced by a younger one; a stalled pool's
# head age rises monotonically. Second, the shape on the queue itself: an `owed-check` row
# at priority 72 waiting while the pool drains work from the eleven sources whose rows ALL
# sit numerically above it (max(priority) below 72 for every one; no source spans 72, and
# no row in the store is at priority 1 or 2 — the `priority=1`/`priority=2` arguments at
# workers/sources/autotriage.py:1165 and owed_check.py:355 belong to
# `run_prompt_in_session`, which is the model-request priority of the session, not the
# priority of a queue row, so they are not in this ranking at all). The
# alarm held the source name in hand — `pending_wait_by_source()`
# is keyed by source, and the max that produces the number is taken over its VALUES — and
# threw the key away one line before printing it.


#: The one config number the starving clause weighs, pinned rather than inherited. The
#: live value read through the loader on 2026-10-14:
#: `workers.sources.get_sources_config()["scheduled-task"]["max_duration_seconds"]` is
#: 3600, and `fw._max_duration_seconds()` — which weighs that source and no other —
#: returns 3600. So the threshold AND the claim window below are 3 x 3600 = 10800 s =
#: 180 min, and every age in this section clears that threshold.
_STARVE_MAX_DURATION_S = 3600
_STARVE_WINDOW_MIN = 3 * _STARVE_MAX_DURATION_S // 60

#: Both ages below are real posts, not round numbers: 915 min is the 2026-10-12 23:58:32
#: line (`server.err.4`), 456 min the 2026-10-13 10:14:02 one. The older is the head the
#: clause has to name, the younger a `scheduled-task` row, so the alarm's own source is
#: waiting too and naming the familiar one instead of the oldest is a visible failure.
_STARVE_HEAD_MIN = 915
_STARVE_YOUNGER_MIN = 456


def _starve_config(monkeypatch):
    """Pin `max_duration_seconds` to the live 3600 s, so the window is 180 min here
    whatever the box's `sources.yaml` says on the day the test runs."""
    monkeypatch.setattr(
        sources, "get_sources_config",
        lambda: {"scheduled-task":
                 _NoEnabledKeyConfig({"max_duration_seconds": _STARVE_MAX_DURATION_S})},
        raising=False)


def _age_queued_row(q, *, row_id: int, minutes: int) -> None:
    """Wind a queued row's `enqueued_at` back, because the age IS the fixture and
    `WorkQueue.enqueue` stamps it from the wall clock on every public write."""
    aged = (dt.datetime.now(dt.timezone.utc)
            - dt.timedelta(minutes=minutes)).isoformat()
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE queue SET enqueued_at=? WHERE id=?", (aged, row_id))
        conn.commit()


def _claim_rows(q, *, source: str, n: int, age_min: int = 0,
                hold: tuple[str, ...] = ("owed-check", "scheduled-task")) -> None:
    """Have the pool claim `n` rows of one source, `age_min` minutes in the past.

    Through `claim_next`, not raw SQL: `claimed_at` is that method's column to write in
    `workers/queue.py`, so the fixture leaves the same trace a real worker does —
    state `claimed`, `claimed_at` stamped, `attempts` bumped. `hold` are the sources
    whose rows must stay queued: it keeps the starving heads unclaimed, which is the
    whole state under test — work going out of the queue past a row that stays. The
    rows this claims are at priority 55 and the held head at 72, so the head is never
    the one handed out even when its own source is the one being drained. Only
    `claimed_at` is wound back, and only when `age_min` says so, because the age of
    that column is the second half of the fixture.
    """
    for i in range(n):
        assert q.enqueue(source=source, kind=f"{source}:row",
                         payload={"i": i}, priority=55) is not None
    for _ in range(n):
        item = q.claim_next("w-2692", exclude_sources=list(hold))
        assert item is not None and item.source == source, (
            f"the pool did not hand out a {source} row, so the claim evidence this "
            f"section turns on was never written: {item}")
    if age_min:
        aged = (dt.datetime.now(dt.timezone.utc)
                - dt.timedelta(minutes=age_min)).isoformat()
        with sqlite3.connect(str(q.db_path)) as conn:
            conn.execute("UPDATE queue SET claimed_at=? WHERE source=?",
                         (aged, source))
            conn.commit()


def _starve_head(q: WorkQueue) -> int:
    """Put the waiting row on the board and wind its age to the one that posted.

    Separate from `_starved_board_queue` because one fixture has to build the claims
    BEFORE this row exists: `claim_next` ranks on the EFFECTIVE priority, and the
    aging credit (#2496, 15 points per 6 h) pulls a 915 min row from 72 down to ~34,
    above everything fresh — so an aged head cannot be walked past from the same
    source, and the state where a source claims rows while its own head waits is only
    reachable by ageing the head after those claims are made.
    """
    head = q.enqueue(source="owed-check", kind="owed-check:clause",
                     payload={"item_id": 2690, "apply": True}, priority=72)
    _age_queued_row(q, row_id=head, minutes=_STARVE_HEAD_MIN)
    return head


def _starved_board_queue(tmp_path, name: str, *, head: bool = True) -> WorkQueue:
    """A queue in the state that produced the five false posts, and nothing else.

    An `owed-check` head at priority 72 (the live figure on 2026-10-14: five
    `owed-check` rows queued at 72, oldest 171 min) aged 915 min, and a younger aged
    `scheduled-task` row, so the alarm's own source is waiting too and the clause has
    to pick the OLDER one rather than the familiar one. No claims here — each node
    adds its own — and no task file, so `_grossly_overdue` and `_next_run_stalled`
    answer nothing and every alert counted below is the due-ness starve clause.
    `head=False` builds the same board one step earlier, for a fixture that has to
    claim rows before the head is old enough to win them.
    """
    q = WorkQueue(tmp_path / name)
    if head:
        _starve_head(q)
    younger = q.enqueue(source="scheduled-task", kind="heartbeat", payload=None)
    _age_queued_row(q, row_id=younger, minutes=_STARVE_YOUNGER_MIN)
    if head:
        _assert_starving(q)
    return q


def _assert_starving(q: WorkQueue) -> None:
    """The fixture's two standing premises, checked wherever the board is finished."""
    assert fw._queue_starving(q, _STARVE_MAX_DURATION_S) > 3 * _STARVE_MAX_DURATION_S, (
        f"a {_STARVE_HEAD_MIN} min head is not starvation at 3 x "
        f"{_STARVE_MAX_DURATION_S} s, so no alert below could fire")
    assert fw._grossly_overdue(q) == [] and fw._next_run_stalled(q) == [], (
        "the board is not empty of task-side findings, so an alert counted below "
        "would not be attributable to the due-ness starve clause")


async def test_an_outranked_wait_names_its_source_and_says_priority_starvation(
        board, monkeypatch, tmp_path):
    """Clauses 1 + 2: the line names the waiting source, and blames the priority order.

    The state is the live one: an `owed-check` head 915 min old, three `autotriage`
    rows claimed inside the 180 min window, and nothing else wrong with the fleet. At
    base this posts `autonomy scheduler may be stalled: oldest claimable queue item is
    915 min old` — which names no source and accuses the one process the evidence
    clears, because three rows went out of that queue inside the window.
    """
    _starve_config(monkeypatch)
    q = _starved_board_queue(tmp_path, "outranked.db")
    _claim_rows(q, source="autotriage", n=3)

    alerts = _watch_alerts(monkeypatch)
    _streak_at_alerting_threshold(monkeypatch)
    await fw.tick(q)

    assert len(alerts) == 1, f"expected exactly the one due-ness alert: {alerts}"
    msg = alerts[0]
    assert "owed-check" in msg, (
        f"the clause still does not say which source is waiting: {msg}")
    assert f"{_STARVE_HEAD_MIN} min old" in msg, (
        f"the age of the waiting row left the clause, and it is the symptom a reader "
        f"acts on: {msg}")
    assert "outranked" in msg and "priority" in msg, (
        f"the clause does not attribute the wait to the priority order: {msg}")
    assert "stalled" not in msg.lower(), (
        f"the word that accuses the scheduler survived on a board where the pool "
        f"claimed three rows inside the window: {msg}")
    assert "autotriage" in msg, (
        f"the clause does not name the work being claimed ahead of the wait: {msg}")
    assert f"inside {_STARVE_WINDOW_MIN} min" in msg, (
        f"the clause does not bound the claim it counts with the window it weighed them "
        f"in, and without the {3 * _STARVE_MAX_DURATION_S // 60} min a reader cannot "
        f"tell an outranked queue from a stalled one: {msg}")


async def test_a_wait_with_nothing_claimed_in_the_window_keeps_today_s_sentence(
        board, monkeypatch, tmp_path):
    """Clause 3's half that lives here: no claims in the window, today's bytes.

    Same board, same ages, the only difference being that the three `autotriage`
    claims sit at 400 min — outside the 180 min window, so nothing proves the pool is
    taking work, and the alarm has to keep the sentence
    `agent-services/guardian/detect.py:243` watches. Character for character,
    including the absence of the source name: an attribution this file cannot
    establish must not leak into the genuine-stall case.
    """
    _starve_config(monkeypatch)
    q = _starved_board_queue(tmp_path, "unclaimed.db")
    _claim_rows(q, source="autotriage", n=3, age_min=400)

    alerts = _watch_alerts(monkeypatch)
    _streak_at_alerting_threshold(monkeypatch)
    await fw.tick(q)

    assert alerts == ["autonomy scheduler may be stalled: "
                      f"oldest claimable queue item is {_STARVE_HEAD_MIN} min old"], (
        f"a genuine stall — nothing claimed inside the window — stopped printing the "
        f"sentence the guardian watch keys on: {alerts}")


async def test_claims_by_the_waiting_source_itself_do_not_explain_the_wait(
        board, monkeypatch, tmp_path):
    """The exclusion that keeps this an attribution and not an excuse.

    If the head's OWN source were claiming rows while the head sat unclaimed, that is
    not priority starvation — `claim_next` takes the best-ranked claimable row, so a
    same-source claim past an older sibling is the anomaly, not the alibi. The claims
    read therefore excludes the waiting source, and the alarm stays the honest
    accusation it is today.
    """
    _starve_config(monkeypatch)
    # The claims are `owed-check`'s own, which is the case that must NOT be read as
    # someone else being ahead of it, so they are made first and the head is aged in
    # afterwards — see `_starve_head` for why the order is forced.
    q = _starved_board_queue(tmp_path, "self-claimed.db", head=False)
    _claim_rows(q, source="owed-check", n=2, hold=("scheduled-task",))
    _starve_head(q)
    _assert_starving(q)

    alerts = _watch_alerts(monkeypatch)
    _streak_at_alerting_threshold(monkeypatch)
    await fw.tick(q)

    assert len(alerts) == 1, alerts
    msg = alerts[0]
    assert "priority starvation" not in msg, (
        f"the waiting source's own claims were read as proof that someone else is "
        f"ahead of it: {msg}")
    assert msg == ("autonomy scheduler may be stalled: "
                   f"oldest claimable queue item is {_STARVE_HEAD_MIN} min old"), msg


async def test_the_starvation_wording_still_posts_once_and_spends_the_instant(
        board, monkeypatch, tmp_path):
    """Clause 4: attribution, never suppression — one alert, cooldown spent, restart quiet.

    Everything clause 2 changes is the WORDS. The alert still goes exactly once, the
    instant still lands on the queue under the due-ness key BEFORE the send (the
    #1704 ordering), and a second process with a fresh `_state` over the same starving
    queue still has nothing to say — the persistence #2662 added, re-pinned on the new
    wording rather than inherited from the old one.
    """
    _pacing_constants_unchanged()
    _starve_config(monkeypatch)
    q = _starved_board_queue(tmp_path, "starved-cooldown.db")
    _claim_rows(q, source="autotriage", n=3)

    first = _fresh_process_state(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, (
        f"the starvation wording did not post exactly one alert: {alerts}")
    assert "owed-check" in alerts[0] and "stalled" not in alerts[0].lower(), alerts[0]
    posted = first["stall_alerted_at"]
    assert posted is not None, (
        "the starvation wording posted without recording the instant, so the next "
        "process is handed a fresh 6 h — the exact defect #2662 closed")

    rows = _durable_alert_instants(WorkQueue(q.db_path))
    assert [(k, v) for k, v in rows if k == fw._STALL_WM_KEY] == [(fw._STALL_WM_KEY,
                                                                  posted)], (
        f"the due-ness instant is not the one a person was told about: {rows}")

    second = _fresh_process_state(monkeypatch)
    assert second["stall_alerted_at"] is None, (
        "the fixture did not hand the second process a fresh cooldown")
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(q)
    assert len(alerts) == 1, (
        f"a process with a fresh `_state` re-posted the starvation alert inside "
        f"{fw._STALL_ALERT_INTERVAL_SECONDS}s of the first: {alerts}")


async def test_the_source_attribution_is_read_on_the_alert_path_and_never_per_tick(
        board, monkeypatch, tmp_path):
    """A call-count rail: the claim read belongs to the alert, not to the tick.

    `_operator_pause_clause` earned its "computed on the alert path only" line because
    the watchdog ticks every minute and an alarm that costs a database read per tick is
    an alarm that quietly becomes the tick's dominant cost. The starvation clause reads
    `queue.claimed_at`, and `claimed_at` has no index of its own, so that read is a scan
    of the `queue` table — 9,520 rows on this box 2026-10-14. A healthy tick must not
    pay it, and a cooled-down tick must not either: the cooldown zeroes `starving` before
    the message is built, and a finding nobody is being told about has no source to look
    up.
    """
    _starve_config(monkeypatch)
    q = _starved_board_queue(tmp_path, "alert-path.db")
    _claim_rows(q, source="autotriage", n=3)

    reads: list[str] = []
    real = WorkQueue.claims_by_source

    def _counted(self, since_iso, exclude_source=None):
        reads.append(self and "claims_by_source")
        return real(self, since_iso, exclude_source=exclude_source)

    monkeypatch.setattr(WorkQueue, "claims_by_source", _counted)
    alerts = _watch_alerts(monkeypatch)

    _streak_at_alerting_threshold(monkeypatch)
    await fw.tick(q)
    assert len(reads) == 1, (
        f"an alerting tick read the claim counts {len(reads)} times; the clause names "
        f"one source from one aggregate, so one read is the whole cost: {alerts}")

    reads.clear()
    _streak_at_alerting_threshold(monkeypatch)
    await fw.tick(q)                       # cooldown engaged — nothing is posted
    assert reads == [], (
        f"the cooldown swallowed the finding and the read came anyway: {len(reads)}. "
        f"A finding nobody is being told about has no source to look up, and this is "
        f"the tick that runs every minute")

    reads.clear()
    alerts.clear()
    empty = WorkQueue(tmp_path / "healthy.db")
    for _ in range(fw._STALL_ALARM_TICKS):
        await fw.tick(empty)
    assert reads == [], (
        f"a healthy tick — nothing overdue, nothing waiting — paid {len(reads)} claim "
        f"reads, so the scan is on the per-tick path: {alerts}")
    assert alerts == [], f"a healthy board must post nothing: {alerts}"


async def test_an_overdue_only_alert_names_no_source_and_binds_no_starvation_clause(
        board, monkeypatch, tmp_path):
    """The overdue-only arm: a task-side finding alone must still build a message.

    Every other node in this section reaches the alert with BOTH findings present, and
    an alert block that binds its starvation wording only inside the waiting-row branch
    breaks HERE instead — an overdue task with a quiet queue is the common case, and it
    is the shape that takes the `head` line without ever running the branch that defines
    the clause it interpolates. The alarm must come out as today's task-side sentence,
    with no source named and no claim read paid.
    """
    # The overdue arm weighs `last_run` against the frequency interval, not `next_run`
    # (`_grossly_overdue`: a task with no `last_run` is skipped outright), so both are
    # set 30 days back — past 10x a daily interval, which is the threshold.
    write_task(board, 971, status="up_next", frequency="daily",
               next_run=_ago(days=30), last_run=_ago(days=30))
    q = WorkQueue(tmp_path / "quiet-queue.db")
    assert fw._queue_starving(q, _STARVE_MAX_DURATION_S) == 0.0, (
        "the queue is not quiet, so this node would not be testing the overdue-only arm")

    reads: list[str] = []
    real = WorkQueue.claims_by_source

    def _counted(self, since_iso, exclude_source=None):
        reads.append("claims_by_source")
        return real(self, since_iso, exclude_source=exclude_source)

    monkeypatch.setattr(WorkQueue, "claims_by_source", _counted)
    alerts = _watch_alerts(monkeypatch)
    _streak_at_alerting_threshold(monkeypatch)
    await fw.tick(q)

    assert len(alerts) == 1, f"expected the one overdue alert: {alerts}"
    msg = alerts[0]
    assert msg.startswith("autonomy scheduler may be stalled: "), msg
    assert "task(s) due and overdue" in msg, msg
    assert "priority starvation" not in msg and "owed-check" not in msg, msg
    assert "claimable queue item" not in msg, (
        f"a quiet queue must not contribute a waiting-row clause: {msg}")
    assert reads == [], (
        f"an alert with no waiting row still paid a claim read: {reads}")

# ── #2890: the worker pool's failure-throughput census ───────────────────────
# The population these nodes seed is `workers.db`'s `runs` table, which no other
# node in this file writes: the two stall alarms read task files and the QUEUE,
# and that is exactly the seam #2890 is about — 137 `failed` runs on 2026-10-16
# against 7 on 2026-10-15 reached no reader, because the fleet's only alarm seat
# had never looked at the run history it was sitting next to.

_THROUGHPUT_DAY = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat()
_THROUGHPUT_PREV = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=2)).isoformat()
_THROUGHPUT_MINUS3 = (dt.datetime.now(dt.timezone.utc).date()
                                 - dt.timedelta(days=3)).isoformat()


def _unsuppressed_census(monkeypatch) -> None:
    """Pin THIS alarm's in-process day to "never posted", for this node only.

    `_throughput_reported_day` takes the LATER of `fw._state` and the watermark row, so a
    day left in the module-global by any node suppresses the census for every node
    collected after it — including against a store that never heard of that day, which is
    how one of these nodes came to fail for a reason its own assertion does not describe.
    `_state` is shared with `scheduled_task` and only `_both_states_restored` guards it, in
    this file; the throughput nodes pin the ONE key they own through monkeypatch, which
    restores that key to its prior state (absence included) at teardown instead of
    rebinding the shared dict underneath another fixture.
    """
    monkeypatch.setitem(fw._state, fw._THROUGHPUT_WM_KEY, None)


def _forget_census_day(monkeypatch) -> None:
    """Drop the in-process day while leaving the STORE's row alone.

    This is the durable-watermark question — does the suppression survive a process that
    forgot it — asked without `_fresh_process_state`, which rebinds the dict
    `scheduled_task` shares and is a far wider intervention than forgetting one key.
    """
    monkeypatch.setitem(fw._state, fw._THROUGHPUT_WM_KEY, None)


def _throughput_queue(path, pairs) -> WorkQueue:
    """Seed a `runs` table. `pairs` is `[(utc_day, summary, 'HH:MM'), …]`, one per row.

    The day is written into `started_at` exactly as this store writes it — ISO with a
    `+00:00` offset — because the census groups on the leading 10 characters of that
    column, and a fixture stamped some other way would test a reader the pool does not
    have. Every row is `status='failed'`: the census counts failures, and seeding a
    success row to pad a count would let a green pass hide a `WHERE` clause that
    dropped the status filter.
    """
    q = WorkQueue(path)
    conn = sqlite3.connect(str(path))
    for n, (day, summary, hhmm) in enumerate(pairs):
        stamp = f"{day}T{hhmm}:00+00:00"
        conn.execute(
            "INSERT INTO runs (run_id, queue_id, source, task_id, status,"
            " started_at, completed_at, summary) VALUES (?,?,?,?,?,?,?,?)",
            (f"run-{n}", None, "owed-check", None, "failed", stamp, stamp, summary))
    conn.commit()
    conn.close()
    return q


def _throughput_rows(day, failed, first_minute=0):
    """`failed` rows on `day`, spread one minute apart from `first_minute`."""
    return [(day, f"run {n} failed: no structured answer",
             f"{first_minute + n // 60:02d}:{first_minute + n % 60:02d}")
            for n in range(failed)]


async def test_the_census_posts_on_a_137_versus_7_day_and_stays_silent_at_8_versus_7(
        board, monkeypatch, tmp_path):
    """Clause 1: the ratio decides, on the two counts the incident actually had.

    137 against 7 is 2026-10-16 over 2026-10-15 as `workers.db` reported them at
    filing — the day the engine refused connections for three windows and the note
    said nothing. 8 against 7 is the ordinary neighbour: 1.14x, nowhere near the 10x
    rule, and the pair that keeps this a census of an event rather than a daily
    report. Both days are seeded as COMPLETE (`_THROUGHPUT_DAY` is yesterday UTC), so
    neither half is passing or failing because of which calendar day it is today.

    The board is empty and the queue is empty, so the two stall alarms have nothing to
    say and every alert captured here came from the census.
    """
    _unsuppressed_census(monkeypatch)
    q = _throughput_queue(tmp_path / "spike.db",
                          _throughput_rows(_THROUGHPUT_DAY, 137)
                          + _throughput_rows(_THROUGHPUT_PREV, 7))
    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)
    assert len(alerts) == 1, f"expected the one throughput alert: {alerts}"
    assert "137 failed runs on UTC" in alerts[0], alerts[0]
    assert f"against 7 on UTC {_THROUGHPUT_PREV}" in alerts[0], alerts[0]
    assert _THROUGHPUT_DAY in alerts[0], alerts[0]

    # The same seat, the same shape of store, one failure fewer on the newer day:
    # silence, and silence for the right reason — the detector says None, so there is
    # no message to suppress downstream.
    quiet = _throughput_queue(tmp_path / "ordinary.db",
                              _throughput_rows(_THROUGHPUT_DAY, 8)
                              + _throughput_rows(_THROUGHPUT_PREV, 7))
    alerts.clear()
    await fw.tick(quiet)
    assert alerts == [], (
        f"an 8-vs-7 day is 1.14x, inside the {fw._THROUGHPUT_RATIO}x rule, yet the "
        f"census posted: {alerts}")
    assert fw._throughput_census(quiet) is None, (
        "the detector returned a verdict for a sub-ratio pair, so the silence above "
        "came from the watermark and not from the rule")


async def test_the_throughput_line_is_repeated_only_once_per_day_by_its_own_watermark(
        board, monkeypatch, tmp_path):
    """Clause 3: two ticks over one pair, one line, and the other alarms' instants intact.

    The dedupe is the alarm's OWN watermark row rather than a cadence, which is what
    makes it observable at all: two consecutive `tick`s in one test are the same pair
    of days read twice, so a second line can only mean the dedupe is broken. The other
    two alarms' rows are SEEDED first, because a watermark that was never written has
    no persisted instant to preserve — writing them here is the only way the second
    half of this clause has anything to compare.
    """
    _unsuppressed_census(monkeypatch)
    q = _throughput_queue(tmp_path / "dupe.db",
                          _throughput_rows(_THROUGHPUT_DAY, 137)
                          + _throughput_rows(_THROUGHPUT_PREV, 7))
    q.wm_set(fw.NAME, fw._STALL_WM_KEY, "2026-10-16T00:00:00+00:00")
    q.wm_set(fw.NAME, fw._NEXTRUN_WM_KEY, "2026-10-16T00:00:01+00:00")
    before = _watermarks(q)

    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)
    await fw.tick(q)
    assert len(alerts) == 1, (
        f"two ticks over the same triggering pair posted {len(alerts)} lines; the "
        f"day watermark is what makes the second one silence: {alerts}")

    after = _watermarks(q)
    assert after.get(fw._THROUGHPUT_WM_KEY, (None, None))[0] == _THROUGHPUT_DAY, (
        "the census posted a day it did not record, so the next process would post "
        f"the same day again: {after}")
    for key in (fw._STALL_WM_KEY, fw._NEXTRUN_WM_KEY):
        assert after[key] == before[key], (
            f"the throughput census moved the persisted instant of `{key}` — "
            f"{before[key]} became {after[key]} — which is one alarm spending another's "
            "silence, the failure #1704/#1735 removed")

    # The durable half of the same watermark. `fw._state` is a process memory, and the
    # row is what survives a restart, so the suppression has to be readable WITHOUT the
    # memory: empty it and the census must still stay silent on the day it already
    # posted. Reusing `_fresh_process_state` rather than hand-writing a literal is the
    # point — that helper asserts its own key set against the module's, so a `_state`
    # that grew a key cannot pass as "a fresh process" here either.
    _forget_census_day(monkeypatch)
    assert fw._state.get(fw._THROUGHPUT_WM_KEY) is None, (
        "the in-process day was not forgotten, so the read-back below is still "
        "reading the cache it was supposed to lose")
    assert fw._throughput_reported_day(q) == _THROUGHPUT_DAY, (
        "a second process could not read the day back from the store, so a restart "
        "would post the same census again")
    before_restart = len(alerts)
    await fw.tick(q)
    assert len(alerts) == before_restart, (
        "with the in-process memory emptied, the census re-posted a day the store "
        "already records")
    # And the other two alarms' instants survived the restart read-back too, since
    # `tick` above ran them against the seeded rows.
    after_restart = _watermarks(q)
    for key in (fw._STALL_WM_KEY, fw._NEXTRUN_WM_KEY):
        assert after_restart[key] == before[key], (
            f"`{key}` moved on the tick that only read the throughput watermark back: "
            f"{before[key]} became {after_restart[key]}")


def _watermarks(q: WorkQueue) -> dict:
    """Every watermark row this watchdog owns, as {key: (value, updated_at)}."""
    conn = sqlite3.connect(str(q.db_path))
    try:
        return {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT key, value, updated_at FROM watermarks WHERE source = ?",
            (fw.NAME,))}
    finally:
        conn.close()


async def test_the_census_is_read_from_the_watch_seat_with_the_source_disabled(
        board, tmp_path, monkeypatch):
    """Clause 4, across the real seam: a STARTED pool, `enabled: false`, one census line.

    The same construction as `test_the_scheduler_loop_watches_the_fleet_with_the_source_disabled`:
    a real `WorkerPool.start()`, the source block saying `enabled: False`, and the
    dispatch half stubbed out — so the only thing on this pool that could have produced
    an alert is `_watch_fleet`. A census that consulted the source's `enabled` key would
    post nothing here and still look healthy to every other node in this file, which is
    the disarmament #1682 exists to remove.

    The board is left EMPTY, so the two stall alarms have nothing to report and the one
    line captured is the census's own.
    """
    _unsuppressed_census(monkeypatch)
    q = _throughput_queue(tmp_path / "disabled.db",
                          _throughput_rows(_THROUGHPUT_DAY, 137)
                          + _throughput_rows(_THROUGHPUT_PREV, 7))
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {"scheduled-task": st},
                        raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {
        "scheduled-task": {"enabled": False, "interval_seconds": 0,
                           "max_duration_seconds": 1800}}, raising=False)
    alerts = _watch_alerts(monkeypatch)

    async def _noop(*_a, **_k):
        return None

    pool = WorkerPool(q, slots=0, poll_idle_seconds=0.01)
    monkeypatch.setattr(pool, "_maybe_sweep_poisoned", _noop)
    monkeypatch.setattr(pool, "_probe_services", _noop)
    monkeypatch.setattr(pool, "_watch_dispatch", _noop)
    # Count the seat's completed passes rather than polling wall clock for a message:
    # the counter is incremented AFTER `tick` returns, so a non-zero count is proof the
    # census was reached in that pass — a wait on `alerts` alone could time out for a
    # reason that has nothing to do with the census, and then the assertion below would
    # be a timing story instead of a verdict about the switch.
    seen = {"watch": 0}
    real_watch = pool._watch_fleet

    async def _watch_spy():
        await real_watch()
        seen["watch"] += 1

    monkeypatch.setattr(pool, "_watch_fleet", _watch_spy)
    await pool.start()
    try:
        for _ in range(300):
            if alerts or seen["watch"] >= 3:
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()

    assert seen["watch"] >= 1, (
        "a started pool ran `_watch_fleet` zero times, so nothing on this seat had a "
        "chance to report anything")
    assert any("worker pool failure throughput" in a for a in alerts), (
        f"the watch seat ran {seen['watch']} pass(es) and did not report the throughput "
        f"census while the `scheduled-task` source block said `enabled: false`; the "
        f"day it would have to suppress is "
        f"{fw._throughput_reported_day(q)!r} of " + repr(alerts))
    assert not q.list_items(source="scheduled-task", limit=50), (
        "a source disabled in config enqueued work, so the line above could have come "
        "from dispatch rather than from the watchdog seat")


async def test_the_census_is_silent_and_raises_nothing_on_a_store_it_cannot_compare(
        board, monkeypatch, tmp_path):
    """Clause 5: the three shapes that are not a comparison, plus an unreadable store.

    An EMPTY `runs` table; a store holding rows for ONE UTC day, which has no
    predecessor to weigh; and a newest complete day whose predecessor is ABSENT (rows
    two days back and one day back with the middle day missing). The last shape is the
    one worth stating: a day the pool did not run is not a day the pool survived, so a
    missing predecessor is read as no comparison, not as a zero that makes any count
    infinite. And a store that cannot be opened at all must cost one log line, because
    this runs on the scheduler's own tick beside two stall alarms.
    """
    _unsuppressed_census(monkeypatch)
    alerts = _watch_alerts(monkeypatch)
    # In `predecessor-absent` the rows two days back are real and the day BETWEEN them
    # and yesterday has no row at all, so a reader that treated an absent day as a zero
    # would report 137 against 0 and post.
    shapes = {
        "empty": [],
        "one-day": _throughput_rows(_THROUGHPUT_DAY, 137),
        "predecessor-absent": (_throughput_rows(_THROUGHPUT_DAY, 137)
                               + _throughput_rows(_THROUGHPUT_MINUS3, 1)),
    }

    for name, pairs in shapes.items():
        q = _throughput_queue(tmp_path / f"{name}.db", pairs)
        alerts.clear()
        await fw.tick(q)                       # must not raise into the caller
        assert alerts == [], f"{name}: posted a line with nothing to compare: {alerts}"
        assert fw._throughput_census(q) is None, (
            f"{name}: the detector invented a comparison for a store that has none")

    # Built as a real queue and then corrupted, because `WorkQueue.__init__` itself
    # opens the file and would raise here rather than at the census: the shape under
    # test is a store that opened once and cannot be read NOW.
    broken = _throughput_queue(tmp_path / "broken.db", [])
    with open(broken.db_path, "r+b") as fh:
        fh.seek(0)
        fh.write(b"not a sqlite database" * 40)
    alerts.clear()
    await fw.tick(broken)                      # a store that refuses to open
    assert alerts == [], f"an unreadable store produced an alert: {alerts}"
    assert fw._throughput_census(broken) is None, (
        "an unreadable store raised or guessed instead of reporting nothing")


# ── #2899: an absolute-floor leg on the SAME census ──────────────────────────
# The ratio leg compares a day with its neighbour, and a comparison is blind on
# exactly the days #2890's owed entry named: `2026-09-29` (187 failed against 27,
# 6.93x) and `2026-10-07` (75 against 12, 6.25x) are the two worst days in the
# live `workers.db` and neither reached a note, because the day before each was
# already bad. The floor leg asks the other question of the SAME reading — is this
# day itself bad, whatever its predecessor looked like — so it inherits the
# newest-complete-UTC-day-with-adjacent-predecessor shape, including the rule that
# an absent predecessor is not a zero (the node above stays green at any count).
# 40 is pinned, not derived: over the 25 complete UTC days that store held on
# 2026-10-17 the sorted per-day `failed` counts are
# [1,5,6,6,7,7,8,8,9,9,10,10,10,12,13,14,15,20,27,27,32,57,75,137,187], and the
# only gap wide enough to sit between an ordinary bad day and the four days worth
# a line is 32→57, so ANY floor inside it fires on exactly {09-24:57, 09-29:187,
# 10-07:75, 10-16:137}. Whether 40 is the right number inside that gap is #2899's
# owed measurement after a week of real traffic, not something this fixture can
# decide.

async def test_the_floor_leg_posts_on_a_40_versus_20_day_the_ratio_leg_is_blind_to(
        board, monkeypatch, tmp_path):
    """Clause 1: the absolute leg fires on a 2x day, and the constant fires it.

    40 failed against 20 is 2.0x — one fifth of `_THROUGHPUT_RATIO` — so nothing
    about this pair can satisfy the ratio rule; it is 2026-10-07's shape (75 v 12,
    6.25x, silent) shrunk to fixture size. The line names both counts and both days
    because a floor reading that said only "40 failed" could not be checked against
    the store it came from. The board and the queue are empty, so the only alert
    this tick can produce is the census's.
    """
    _unsuppressed_census(monkeypatch)
    assert fw._THROUGHPUT_FLOOR == 40, (
        "the floor is a constant pinned on the live distribution, not a baseline — "
        "the 40 and 20 seeded below are that number and its half")
    q = _throughput_queue(tmp_path / "floor.db",
                          _throughput_rows(_THROUGHPUT_DAY, 40)
                          + _throughput_rows(_THROUGHPUT_PREV, 20))
    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)
    assert len(alerts) == 1, f"expected the one floor-leg alert: {alerts}"
    msg = alerts[0]
    assert "40 failed runs on UTC" in msg, msg
    assert f"against 20 on UTC {_THROUGHPUT_PREV}" in msg, msg
    assert _THROUGHPUT_DAY in msg, msg


async def test_the_floor_is_a_constant_so_39_failed_against_20_posts_nothing(
        board, monkeypatch, tmp_path):
    """Clause 2: one failure under the floor is silence, and the SILENCE IS THE RULE'S.

    39 against 20 is 1.95x and one short of `_THROUGHPUT_FLOOR`. The detector-level
    assertion is the point of this node rather than decoration: a sliding baseline —
    a mean of recent days, which would land near 20 here — posts this line, and that
    is the failure the clause exists to keep out. Returning None says the number
    never reached `_alert` at all, so the emptiness above cannot be a watermark's
    doing.
    """
    _unsuppressed_census(monkeypatch)
    q = _throughput_queue(tmp_path / "just-under.db",
                          _throughput_rows(_THROUGHPUT_DAY, 39)
                          + _throughput_rows(_THROUGHPUT_PREV, 20))
    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)
    assert alerts == [], (
        f"39 failed is one below the pinned floor of {fw._THROUGHPUT_FLOOR}, so this "
        f"is a normal day and the census posted it: {alerts}")
    assert fw._throughput_census(q) is None, (
        "the detector returned a verdict for a day under the floor and under the "
        "ratio, so the silence above belongs to the watermark and not to the rule")


async def test_a_day_tripping_both_legs_posts_one_line_on_the_one_day_watermark(
        board, monkeypatch, tmp_path):
    """Clause 3: two legs fire, one line goes out, one row is written.

    137 against 7 is the pair that shipped the ratio leg — 19.6x AND over the floor,
    so both rules trigger on the same reading. The second tick is the dedupe's test:
    the day must be recorded exactly once, under the ONE key `throughput_day`, and a
    second key for the second leg would be two alarms spending one another's
    silence, which is precisely what #1704/#1735 removed for the stall alarms'
    persisted instants. The stall row is seeded first so "no new key" has a baseline
    to be compared against.
    """
    _unsuppressed_census(monkeypatch)
    q = _throughput_queue(tmp_path / "both-legs.db",
                          _throughput_rows(_THROUGHPUT_DAY, 137)
                          + _throughput_rows(_THROUGHPUT_PREV, 7))
    # Both neighbours' rows are SEEDED so the "no new key" comparison below has a
    # baseline of the rows a live store actually holds: with only one of them
    # present, a row some other alarm legitimately writes during the two ticks would
    # read as the floor leg bringing its own key.
    q.wm_set(fw.NAME, fw._STALL_WM_KEY, "2026-10-16T00:00:00+00:00")
    q.wm_set(fw.NAME, fw._NEXTRUN_WM_KEY, "2026-10-16T00:00:01+00:00")
    before = _watermarks(q)
    assert fw._NEXTRUN_WM_KEY in before and fw._STALL_WM_KEY in before, (
        f"the baseline rows were not written, so the key-set comparison below would "
        f"be comparing against an empty store: {sorted(before)}")

    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)
    await fw.tick(q)
    assert len(alerts) == 1, (
        f"two ticks over a pair that trips both legs posted {len(alerts)} lines; the "
        f"legs share the one day watermark, so both together still write one line "
        f"per counted day: {alerts}")
    assert "10x rule" in alerts[0] and "floor rule" in alerts[0], (
        f"a day that trips both rules names only the one that was here first: "
        f"{alerts[0]}")

    after = _watermarks(q)
    assert after.get(fw._THROUGHPUT_WM_KEY, (None, None))[0] == _THROUGHPUT_DAY, (
        "the census posted a day it did not record, so the next process would post "
        f"the same day again: {after}")
    extra = sorted(set(after) - set(before) - {fw._THROUGHPUT_WM_KEY})
    assert extra == [], (
        f"the floor leg brought its own watermark key {extra} into the store; both "
        f"legs dedupe on `{fw._THROUGHPUT_WM_KEY}` alone")
    for key in (fw._STALL_WM_KEY, fw._NEXTRUN_WM_KEY):
        assert after[key] == before[key], (
            f"the floor leg moved the persisted instant of `{key}` — "
            f"{before[key]} became {after[key]} — which is one alarm spending another's "
            "silence, the failure #1704/#1735 removed")


async def test_the_floor_leg_line_names_its_own_rule_and_keeps_the_tallies_and_window(
        board, monkeypatch, tmp_path):
    """Clause 4: the line says WHICH rule fired, and loses nothing the ratio line had.

    `_throughput_alert_message` hardcoded the `(10x rule: N.Nx)` parenthetical, so a
    floor-triggered 40-vs-20 day would have printed "10x rule: 2.0x" — a line that
    misstates its own trigger and sends a reader to audit a ratio that was never the
    reason for it. The class tallies and the dated `failures span …Z to …Z` bounds
    are what made the ratio line triageable (#2890's incident was one availability
    event across three windows), so this node pins them on the floor line too, plus
    the clock sentence, because the census counts UTC days while the note is keyed
    to America/Los_Angeles.
    """
    _unsuppressed_census(monkeypatch)
    pairs = ([(_THROUGHPUT_DAY, "run failed: no structured answer", "03:07")] * 26
             + [(_THROUGHPUT_DAY, "ConnectError: all connection attempts failed",
                 "19:41")] * 10
             + [(_THROUGHPUT_DAY, "unclassified thing happened", "21:02")] * 4
             + [(_THROUGHPUT_PREV, "one-off failure", "06:00")] * 20)
    q = _throughput_queue(tmp_path / "naming.db", pairs)
    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)
    assert len(alerts) == 1, f"expected the one floor-leg alert: {alerts}"
    line = alerts[0]

    assert "floor rule" in line, (
        f"a day that fired only the absolute leg names no rule, or names the wrong "
        f"one: {line}")
    assert "floor rule: >= 40" in line, (
        f"the line names the floor rule but not the value it was pinned at — the "
        f"literal here is deliberate, since the 40 rows seeded above are a literal "
        f"too: {line}")
    assert "10x rule" not in line, (
        f"a 40-vs-20 day is 2.0x, so the ratio rule did not fire and must not be "
        f"printed as the trigger: {line}")
    assert "classes:" in line, f"the floor line carries no class tally: {line}"
    assert "no structured answer 26" in line, line
    assert "connect error 10" in line, line
    assert "other 4" in line, (
        f"the 4 seeded rows that match no needle must show up as `other`, not vanish: "
        f"{line}")
    assert f"failures span {_THROUGHPUT_DAY}T" in line and "Z to " in line, (
        f"the floor line names no window bounds for the failures it counts: {line}")
    assert "UTC" in line and "America/Los_Angeles" in line, (
        f"the floor line names a UTC day on a note keyed to another zone without "
        f"saying so: {line}")


async def test_the_floor_leg_reaches_a_person_from_the_watch_seat_on_a_ratio_blind_day(
        board, tmp_path, monkeypatch):
    """The new leg across the seat's own seam: a STARTED pool, 40 v 20, one line.

    The pool-seat node above seeds a 137-vs-7 pair, which now trips BOTH legs, so
    before this node no floor-only day had ever been read by the caller that actually
    reads the store in production — `WorkerPool._watch_fleet` → `fw.tick` →
    `_census_failure_throughput`'s `run_in_executor` thread opening `workers.db`. The
    other floor nodes call `fw.tick` directly: same function, different caller, and
    the executor hand-off is where a leg that only exists on one call path would show
    up. The source block says `enabled: False` and dispatch is stubbed, so the one
    line this pool can produce is the census's (#1682's disarmament, re-asked of the
    leg that ships in this diff), and the pair is 2.0x — a day the ratio leg, left
    alone, would have passed over in silence exactly as it passed over 2026-10-07.
    """
    _unsuppressed_census(monkeypatch)
    q = _throughput_queue(tmp_path / "floor-seat.db",
                          _throughput_rows(_THROUGHPUT_DAY, 40)
                          + _throughput_rows(_THROUGHPUT_PREV, 20))
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {"scheduled-task": st},
                        raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {
        "scheduled-task": {"enabled": False, "interval_seconds": 0,
                           "max_duration_seconds": 1800}}, raising=False)
    alerts = _watch_alerts(monkeypatch)

    async def _noop(*_a, **_k):
        return None

    pool = WorkerPool(q, slots=0, poll_idle_seconds=0.01)
    monkeypatch.setattr(pool, "_maybe_sweep_poisoned", _noop)
    monkeypatch.setattr(pool, "_probe_services", _noop)
    monkeypatch.setattr(pool, "_watch_dispatch", _noop)
    seen = {"watch": 0}
    real_watch = pool._watch_fleet

    async def _watch_spy():
        await real_watch()
        seen["watch"] += 1

    monkeypatch.setattr(pool, "_watch_fleet", _watch_spy)
    await pool.start()
    try:
        for _ in range(300):
            if alerts or seen["watch"] >= 3:
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()

    assert seen["watch"] >= 1, (
        "a started pool ran `_watch_fleet` zero times, so the seat never had a chance "
        "to read the runs store")
    assert any("40 failed runs on UTC" in a for a in alerts), (
        f"the watch seat ran {seen['watch']} pass(es) over a 40-vs-20 day and said "
        f"nothing — the day-shape the ratio leg cannot see; the day it would have to "
        f"suppress is {fw._throughput_reported_day(q)!r} of " + repr(alerts))
    assert any("floor rule" in a and "10x rule" not in a for a in alerts), (
        f"the line the seat posted does not name the floor rule as its trigger: "
        f"{alerts}")
    assert not q.list_items(source="scheduled-task", limit=50), (
        "a source disabled in config enqueued work, so the line above could have come "
        "from dispatch rather than from the watchdog seat")


# ── #2956: the same line, readable against the host's own clock ──────────────
# The defect here is in a READING, not in a rule. The 2026-10-17 census fired the
# FLOOR leg on 45 failed runs: 43 of them sat inside one ~29-minute window that
# straddled both the UTC midnight boundary and the instant the host came up
# (`uptime -s` = 2026-10-16 17:26 PDT = 2026-10-17T00:26Z), and the other 2 fell in
# the ~23 hours left over. "45 failed runs on UTC 2026-10-17" is a true sentence
# about that day and an incident-shaped one, and nothing on the line said the pool
# had been quiet since the machine came up — a recovery day read as a fresh event.
# What these nodes pin is the number that fixes it: the same day's failures counted
# again from the host boot instant, with the instant and a per-15-minute rate beside
# the day count. Nothing here moves a threshold or adds a leg: the floor stays at 40
# (`test_the_floor_leg_posts_on_a_40_versus_20_day_the_ratio_leg_is_blind_to` pins
# the constant), the ratio stays at 10, and no rule reads the new figure.
#
# Every node holds the clock still TWICE over, because two clocks are in the reading:
# `_host_boot_instant` is monkeypatched to a fixed instant (it is the census's only
# route to the host clock, and `/proc/uptime` moves every tick by construction), and
# `_throughput_census` is handed `now`, so `minutes_since_boot` is arithmetic the node
# writes as a literal rather than a figure that drifts with the wall clock — and the
# day gate cannot roll over under it at midnight.

#: Five minutes into TODAY UTC — the clock these nodes read instead of the real one.
#: `_THROUGHPUT_DAY` is then the newest COMPLETE day and `_THROUGHPUT_PREV` its
#: adjacent predecessor, which is exactly the pair the seeding below writes.
_THROUGHPUT_NOW = dt.datetime.combine(
    dt.datetime.now(dt.timezone.utc).date(), dt.time(0, 5), tzinfo=dt.timezone.utc)
#: The boot instant they pin: 06:00Z on the counted day, so that day's rows split 33
#: before it and 12 at-or-after it, the shape the live day had on its own boot.
_THROUGHPUT_BOOT = dt.datetime.fromisoformat(f"{_THROUGHPUT_DAY}T06:00:00+00:00")


def _boot_pinned(monkeypatch, boot=_THROUGHPUT_BOOT) -> None:
    """Hold the ONE boot route the census has, still, at `boot`.

    The instant moves every tick by construction, so a node that asserted a count
    over the real one could not write its own expected value down. Overriding this
    function is the whole of the fixture because the census reaches the host clock
    through it and through nothing else.
    """
    monkeypatch.setattr(fw, "_host_boot_instant", lambda: boot)


def _boot_split_rows(day=_THROUGHPUT_DAY):
    """45 failures on `day`: 33 before 06:00Z, 10 in 06:00–06:10Z, 2 at 20:00Z.

    2026-10-17's shape at fixture size — a pre-boot cluster, a burst beside the boot
    instant, two ordinary failures hours later — with the first burst row stamped
    EXACTLY at the boot instant so the "at or after" edge is exercised and not
    assumed. Two real classes plus an unclassified remainder, because the tallies the
    line already carries have to survive the new clause.
    """
    return ([(day, "run failed: no structured answer", f"00:{n:02d}")
             for n in range(33)]
            + [(day, "ConnectError: all connection attempts failed", f"06:{n:02d}")
               for n in range(10)]
            + [(day, "unclassified thing happened", "20:00")] * 2)


def test_the_boot_helper_reads_the_uptime_file_and_says_nothing_when_it_is_absent(
        monkeypatch, tmp_path):
    """The helper every node below overrides, tested at the helper itself.

    A node that only ever patches this function would leave the real read unpinned,
    and the real read is the mechanism: `/proc/uptime`'s first field is seconds since
    boot, so boot is `now - uptime`. The 45-minute-old-host case is the one that
    matters for the rate, so the returned instant is checked against the seeded 3600 s
    rather than against a second reading of the same file. Two failure shapes return
    None rather than raising, because the census that calls this runs on the pool's
    scheduler tick beside two stall alarms: garbage in the file, and no file at all
    (which is every non-Linux box this could run on).
    """
    fake = tmp_path / "uptime"
    fake.write_text("3600.5 7200.0\n", encoding="utf-8")
    monkeypatch.setattr(fw, "_PROC_UPTIME_PATH", str(fake))
    boot = fw._host_boot_instant()
    assert boot is not None, "a readable /proc/uptime yielded no instant"
    elapsed = (dt.datetime.now(dt.timezone.utc) - boot).total_seconds()
    assert 3599.0 <= elapsed <= 3602.0, (
        f"{fake} said the host has been up 3600.5 s and the helper returned {boot}, "
        f"{elapsed} s ago; boot is `now - uptime`, not any other reading")
    assert boot.utcoffset() == dt.timedelta(0), (
        f"the boot instant has to be UTC to sit beside `runs.started_at`, and "
        f"{boot} is not")

    fake.write_text("not a number\n", encoding="utf-8")
    assert fw._host_boot_instant() is None, (
        "an unparseable uptime file raised or guessed instead of yielding no instant")
    monkeypatch.setattr(fw, "_PROC_UPTIME_PATH", str(tmp_path / "absent"))
    assert fw._host_boot_instant() is None, (
        "an absent uptime file (a non-Linux box) raised instead of yielding no "
        "instant, which would take the throughput census down with it")

    # No fork per minute: the census is on a 60 s tick, and `uptime -s` is a subprocess
    # per call. Pinned structurally because a slow line is invisible to every other
    # assertion in this file.
    boot_names = set(fw._host_boot_instant.__code__.co_names)
    assert not (boot_names & {"Popen", "popen", "check_output", "subprocess", "system"}), (
        f"_host_boot_instant reaches the host clock through a subprocess "
        f"({sorted(boot_names & {'Popen', 'popen', 'check_output', 'subprocess', 'system'})}), "
        "which forks on every surveillance tick; the item's finding is to read the file")


def test_the_census_counts_the_day_and_the_failures_since_the_host_booted(
        monkeypatch, tmp_path):
    """Clause 1: one reading, four numbers — 45 on the day, 12 of them since boot.

    The seeded day is `_boot_split_rows()`: 33 failures before the pinned 06:00Z boot,
    10 inside 06:00–06:10Z, 2 at 20:00Z, so the since-boot count is 12 and the day
    count stays 45. 20 on the predecessor keeps the pair at 2.25x, so the ONLY thing
    that can produce a verdict is the floor leg, exactly as on the day this came from.
    `minutes_since_boot` is 1085 because the pinned clock is 00:05Z today and the
    pinned boot is 06:00Z yesterday: 18 h 5 min.
    """
    _boot_pinned(monkeypatch)
    q = _throughput_queue(tmp_path / "since-boot.db",
                          _boot_split_rows() + _throughput_rows(_THROUGHPUT_PREV, 20))
    census = fw._throughput_census(q, now=_THROUGHPUT_NOW)
    assert census is not None, (
        "45 failed against 20 trips the pinned floor of 40, so the census said "
        "nothing and there would be no reading to report")
    assert census["day"] == _THROUGHPUT_DAY and census["newer"] == 45, (
        f"the day count the line already prints changed: {census['day']} / "
        f"{census['newer']} — #2956 adds a number, it does not move this one")
    assert census["boot"] == _THROUGHPUT_BOOT, (
        f"the census did not report the host boot instant the helper returned: "
        f"{census['boot']!r} against {_THROUGHPUT_BOOT!r}")
    assert census["since_boot"] == 12, (
        "33 of the seeded 45 failures start before the pinned boot instant, so the "
        f"count at-or-after it must be 12 (the 10:00-window burst plus the two at "
        f"20:00Z), not {census['since_boot']} — the burst row stamped exactly at "
        "06:00:00 has to count, that is the boot ramp")
    assert census["minutes_since_boot"] == 1085, (
        f"06:00Z yesterday to 00:05Z today is 18 h 5 min = 1085 whole minutes, and "
        f"the census reported {census['minutes_since_boot']}; the rate's denominator "
        "is this number floored at 15")


def test_the_throughput_line_prints_the_day_count_and_the_since_boot_reading(
        monkeypatch, tmp_path):
    """Clause 2: the line carries the new numbers and loses nothing it printed before.

    Same census as clause 1, rendered. The since-boot figure is `12 of them`, not a
    replacement count: the day count is what both rules weighed and what the owed
    re-pricing of the floor counts (#2899 owed 1), so a line that swapped one number
    for the other would close the very measurement the floor is waiting on. The rate
    is 12 x 15 / 1085 = 0.1659 -> `0.17 / 15 min`, printed beside the 1085 minutes it
    was divided by and the boot instant in ISO UTC, so a reader can see how long a
    window the rate describes. The fired-rule parenthetical, the class tallies and the
    `failures span …Z to …Z` bounds are what made the 2026-10-16 line triageable
    (#2890, #2899), so they are asserted here rather than assumed to survive.
    """
    _boot_pinned(monkeypatch)
    q = _throughput_queue(tmp_path / "line.db",
                          _boot_split_rows() + _throughput_rows(_THROUGHPUT_PREV, 20))
    census = fw._throughput_census(q, now=_THROUGHPUT_NOW)
    assert census is not None and census["since_boot"] == 12, (
        "this node asserts against a literal rendered from the census clause 1 "
        f"describes, and got a different reading: {census}")
    line = fw._throughput_alert_message(census)

    assert "45 failed runs on UTC" in line, line
    assert f"against 20 on UTC {_THROUGHPUT_PREV}" in line, line
    assert "12 of them since host boot" in line, (
        f"the line carries no since-boot count beside the day count: {line}")
    assert f"since host boot {_THROUGHPUT_BOOT.isoformat()} " in line, (
        f"the since-boot count names no boot instant, or not in ISO UTC — a reader "
        f"cannot tell a 10-minute-old window from a day-old one without it: {line}")
    assert "= 0.17 / 15 min" in line, (
        f"12 failures over 1085 min since boot is 12*15/1085 = 0.1659 per 15 min, so "
        f"the line prints a different rate or none: {line}")
    assert "(1085 min since boot)" in line, (
        f"the rate does not say how wide its own window was: {line}")

    assert "floor rule: >= 40" in line, (
        f"the fired-rule parenthetical is gone or wrong: {line}")
    assert "10x rule" not in line, (
        f"45 against 20 is 2.25x, so the ratio leg did not fire and must not be "
        f"printed as the trigger: {line}")
    assert "classes:" in line and "no structured answer 33" in line, line
    assert "connect error 10" in line and "other 2" in line, line
    assert f"failures span {_THROUGHPUT_DAY}T" in line and "Z to " in line, (
        f"the day's own failure bounds were dropped to make room for the new "
        f"clause: {line}")
    assert "America/Los_Angeles" in line, (
        f"the clock sentence went missing: {line}")


def test_the_since_boot_rate_window_is_floored_at_fifteen_minutes(monkeypatch, tmp_path):
    """Clause 3: 3 failures in the 5 minutes since boot print 3 / 15 min, never 9.

    The window here is genuinely five minutes: boot is pinned to 23:57Z on the counted
    day and the clock to 00:02Z the next, so the census is reading a machine that came
    up five minutes ago — the first minutes after a reboot, when the pool's backlog
    and its connection errors all land at once. Dividing by the elapsed minutes would
    print 3 x 15 / 5 = `9.00 / 15 min`, a figure that describes how recently the
    machine was switched on and reads louder than the incident the alarm exists to
    catch. The denominator is `max(minutes_since_boot, 15)`, so the rate is 3 x 15 / 15
    = `3.00`, and the parenthetical still says `(5 min since boot)` because the
    narrowness of the window is the reader's business too.
    """
    boot = dt.datetime.fromisoformat(f"{_THROUGHPUT_DAY}T23:57:00+00:00")
    now = dt.datetime.combine(dt.datetime.now(dt.timezone.utc).date(),
                              dt.time(0, 2), tzinfo=dt.timezone.utc)
    _boot_pinned(monkeypatch, boot)
    pairs = (_throughput_rows(_THROUGHPUT_DAY, 40)
             + [(_THROUGHPUT_DAY, "run failed: no structured answer", "23:58")] * 2
             + [(_THROUGHPUT_DAY, "ConnectError: all connection attempts failed",
                 "23:59")]
             + _throughput_rows(_THROUGHPUT_PREV, 20))
    q = _throughput_queue(tmp_path / "ramp.db", pairs)
    census = fw._throughput_census(q, now=now)
    assert census is not None, (
        "40 pre-boot failures plus 3 in the boot ramp is 43, over the pinned floor of "
        "40, so this day has to reach a line")
    assert census["since_boot"] == 3 and census["minutes_since_boot"] == 5, (
        f"the window under test is 3 failures in 5 minutes; the census read "
        f"{census['since_boot']} in {census['minutes_since_boot']}")
    line = fw._throughput_alert_message(census)
    assert "= 3.00 / 15 min" in line, (
        f"3 failures with a 5-minute window must be divided by the 15-minute floor: "
        f"{line}")
    assert "9.00" not in line, (
        f"the rate divided by the raw 5-minute window, which is the artefact the "
        f"floor exists to stop: {line}")
    assert "(5 min since boot)" in line, (
        f"the line hid how narrow its window was: {line}")


async def test_the_line_keeps_its_old_shape_when_the_host_will_not_give_a_boot_instant(
        board, monkeypatch, tmp_path):
    """Clause 4: no instant, no rate — and the alarm still posts what it always posted.

    `None` is the helper's real answer wherever `/proc/uptime` is unreadable, so this
    is not a hypothetical branch: the census has to be as informative as it was before
    #2956 and not one bit more inventive. The census is called directly first, because
    "nothing raises past `_throughput_census`" is only witnessed at the census; then
    through `tick`, because the same shape has to still REACH a person — a census that
    returned None on a missing boot anchor would silence the alarm entirely, which is a
    worse failure than the unreadable line it was meant to improve. The printed line is
    then checked for the absence of the new clause and the presence of everything the
    old line had: a rate measured from no anchor is the invention this defect was.
    """
    _unsuppressed_census(monkeypatch)
    monkeypatch.setattr(fw, "_host_boot_instant", lambda: None)
    q = _throughput_queue(tmp_path / "no-boot.db",
                          _boot_split_rows() + _throughput_rows(_THROUGHPUT_PREV, 20))
    census = fw._throughput_census(q, now=_THROUGHPUT_NOW)
    assert census is not None, (
        "a day over the floor stopped triggering just because the host would not say "
        "when it booted")
    assert census["boot"] is None and census["since_boot"] is None \
        and census["minutes_since_boot"] is None, (
        f"the census invented a boot figure out of a missing anchor: {census}")

    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)                        # nothing raises past the census
    assert len(alerts) == 1, (
        f"the census posts on a triggered day only when it knows the boot instant: "
        f"{alerts}")
    line = alerts[0]
    assert "since host boot" not in line, (
        f"the line names a boot instant the census does not have: {line}")
    assert "/ 15 min" not in line, (
        f"the line printed a rate with no window to compute it from: {line}")
    assert "45 failed runs on UTC" in line, line
    assert f"against 20 on UTC {_THROUGHPUT_PREV}" in line, line
    assert "floor rule: >= 40" in line, line
    assert "classes:" in line and "no structured answer 33" in line, line
    assert "other 2" in line, line
    assert f"failures span {_THROUGHPUT_DAY}T" in line and "Z to " in line, line
    assert "America/Los_Angeles" in line, line


# ── #3063: the window the label prints is the window that did the dividing ────
# `_THROUGHPUT_RATE_WINDOW_MINUTES` (= 15) is the denominator every since-boot rate
# above is normalised to. The line computed WITH it (`git grep -n
# '_THROUGHPUT_RATE_WINDOW_MINUTES' workers/fleet_watchdog.py`) and then printed the
# unit as a literal `15`. Today the two agree, which is why nothing was red; the day
# the constant moves, the one artifact a person triages an incident from divides by
# one window and calls it another. These nodes move the constant and read what comes
# out — first the rendered label, then the sentence beside it in the docstring.

async def test_the_since_boot_rate_label_names_the_window_constant_not_a_literal(
        board, monkeypatch, tmp_path):
    """#3063 clause 1: move `_THROUGHPUT_RATE_WINDOW_MINUTES` and the label moves.

    The census under test is the one #2956 clause 1 describes — 12 of the day's 45
    failures at or after the pinned boot, 1085 minutes before the pinned clock —
    because the census never reads the constant (the formatter is its only reader),
    so patching it changes only what the LINE says. At window 30 the rate is
    12 * 30 / max(1085, 30) = 0.3318, so the honest line is `= 0.33 / 30 min`; the
    code before this change printed `= 0.33 / 15 min`, a unit that had already
    drifted from the arithmetic beside it. The absence of `/ 15 min` is most of the
    clause: a line carrying the new number AND the old one is still two windows.

    The line is then driven a second time through `tick`, the seat that actually
    posts it, because the render a person reads is produced there against the REAL
    clock, not at the formatter — so that pass asserts the unit and not the rate, and
    a formatter-only test could not tell a posted line from an unposted one.
    """
    _boot_pinned(monkeypatch)
    _unsuppressed_census(monkeypatch)
    q = _throughput_queue(tmp_path / "window-30.db",
                          _boot_split_rows() + _throughput_rows(_THROUGHPUT_PREV, 20))
    census = fw._throughput_census(q, now=_THROUGHPUT_NOW)
    assert census is not None and census["since_boot"] == 12 \
        and census["minutes_since_boot"] == 1085, (
        "this node renders the #2956 fixture (12 of the day's 45 failures at or after "
        f"the pinned 06:00Z boot, 1085 min before the pinned clock), got {census}")
    monkeypatch.setattr(fw, "_THROUGHPUT_RATE_WINDOW_MINUTES", 30)
    line = fw._throughput_alert_message(census)
    assert "= 0.33 / 30 min" in line, (
        "with the window at 30 the rate is 12 * 30 / 1085 = 0.3318 per 30 min, so the "
        f"line neither divided by 30 nor labelled its own window 30 min: {line}")
    assert "/ 15 min" not in line, (
        f"the window in the label is a literal that survived moving the constant: "
        f"{line}")
    assert "(1085 min since boot)" in line, (
        f"the elapsed window the label belongs to is no longer on the line: {line}")

    alerts = _watch_alerts(monkeypatch)
    await fw.tick(q)
    assert len(alerts) == 1, (
        f"a 45-vs-20 day over the pinned floor posted {len(alerts)} lines, so the "
        f"posted text is not the text under test: {alerts}")
    assert "/ 30 min" in alerts[0], (
        f"the line that reaches a person labels its rate with something other than "
        f"the constant: {alerts[0]}")
    assert "/ 15 min" not in alerts[0], (
        f"the posted line still carries the literal window: {alerts[0]}")


def test_the_docstring_that_states_the_rate_denominator_names_the_constant():
    """#3063 clause 3: the denominator sentence names the constant, not a number.

    `_throughput_alert_message`'s docstring stated the denominator as
    `max(minutes_since_boot, 15)`. The rendered label now interpolates, and a prose
    number beside it drifts the same silent way: whoever moves the constant to 30
    reads a sentence still saying 15 and re-prices the floor against a window nobody
    set. The node takes the function's own source through `inspect`, cuts the ONE
    sentence that states the denominator at its NEXT SENTENCE BOUNDARY (a period, so
    rewording the sentence narrows what is checked instead of raising here), and asks
    of that sentence that the denominator expression name
    `_THROUGHPUT_RATE_WINDOW_MINUTES`. The expression is pinned rather than the digit
    `15` banned, because the same sentence carries the worked `22.5 / 15 min` example
    of an UNFLOORED denominator — arithmetic at the shipped value, not a claim about
    the window, and left alone by the item's own finding.
    """
    src = inspect.getsource(fw._throughput_alert_message)
    start = src.find("The rate's denominator is")
    assert start != -1, (
        "the sentence stating what the since-boot rate divides by is gone, so nothing "
        "here can say whether it names the constant or a number")
    tail = src[start:]
    ends = [i for i in (tail.find(". "), tail.find(".\n")) if i > 0]
    assert ends, (
        f"the denominator sentence never ends, so this node could not bound the text "
        f"it is grading: {tail[:120]!r}")
    sentence = tail[:min(ends) + 1]
    assert ("max(minutes_since_boot, _THROUGHPUT_RATE_WINDOW_MINUTES)"
            in sentence), (
        f"the denominator sentence does not name the constant the divisor reads: "
        f"{sentence!r}")
    assert "max(minutes_since_boot, 15)" not in sentence, (
        f"the denominator sentence still states the window as a bare 15, so it drifts "
        f"the day the constant moves: {sentence!r}")
