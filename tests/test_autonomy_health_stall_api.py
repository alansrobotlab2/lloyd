"""`GET /api/autonomy/health` carries the stall verdict: backlog #1121.

The scheduler-side half of #1121 is pinned in `tests/test_autonomy_scheduler.py`,
where `compute_health` is called as a Python function. That is not the seam
production crosses: `autonomy_health` is an MCP tool whose docstring promises a
caller "Fleet health over the last N days", and the tool does not compute
anything — `agent_mcp/autonomy.py:324` proxies to the backend route
(`app/routers/autonomy.py:322`), which builds its `tasks` list with
`autonomy._iter_task_files()` and its rows with `queue.list_runs_joined`. A
per-task field can exist in the pure function and still never reach the agent
that asks the question, and the 2026-09-17 call that scored the stalled #68
`fail_rate: 0.0` was made through this route, not through the function.

So this file drives the route over a real temp autonomy dir and a real
`WorkQueue` sqlite file. The only seam crossed is the one that leaves the
process: the MCP proxy's HTTP hop, which cannot be exercised here because
`agent_mcp` runs as a separate server process (`agent_mcp/autonomy.py:327-329`).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import autonomy as A
from app.routers import autonomy as ROUTER

# A settled instant, so the elapsed arithmetic in the assertions below is a
# number rather than "whenever pytest happened to run". The route reads the real
# wall clock (`compute_health`'s `now` defaults to `_utcnow`), and pinning that
# one call is the difference between asserting gap_ratio >= 40 and asserting it
# is within a few percent of 50.
PIN = dt.datetime(2026, 9, 18, 12, 0, 0, tzinfo=dt.timezone.utc)
TASK_BODY = "\n## Activity Log\n\n- 2026-09-16T14:14:05Z: run completed\n"


def _task_file(dirn: Path, task_id: int, name: str, status: str, **extra) -> Path:
    fm = {"type": "autonomy", "segment": "autonomy", "id": task_id, "name": name,
          "status": status, "frequency": "every-15min"}
    fm.update(extra)
    path = dirn / f"{task_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n{TASK_BODY}",
                    encoding="utf-8")
    return path


@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    monkeypatch.setattr(A, "AUTONOMY_DIR", dirn)
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", dirn)
    return dirn


@pytest.fixture
def queue(tmp_path):
    """A real `WorkQueue` over a real sqlite file, shared by route and test.

    The window-clamp tests (#1401) have to put a run row at a chosen instant,
    so the queue is a fixture in its own right rather than something built
    inside the client fixture — and it stays a real queue rather than a stub
    because the clamp is derived from `SELECT MIN(completed_at)`, which only
    the real schema answers.
    """
    from workers.queue import WorkQueue

    return WorkQueue(tmp_path / "health-seam.db")


@pytest.fixture
def client(autonomy_dir, queue, monkeypatch):
    """The real route, over a real queue file, with the clock pinned once."""
    monkeypatch.setattr("workers.queue.get_queue", lambda: queue)
    monkeypatch.setattr(A, "_utcnow", lambda: PIN)
    app = FastAPI()
    app.include_router(ROUTER.router)
    with TestClient(app) as c:
        yield c


def _seed_run(queue, when: dt.datetime, task_id: str,
              status: str = "success", duration: float = 60.0) -> str:
    """Write one `scheduled-task` run row stamped `when`, returning its stamp.

    `WorkQueue.record_run` takes an explicit `completed_at`, which is the whole
    reason a 20-hour-old store is testable here instead of a fixture that has
    to wait 20 hours for one.
    """
    stamp = when.isoformat()
    queue.record_run(run_id=f"run-{task_id}-{stamp}", queue_id=None,
                     source="scheduled-task", status=status,
                     started_at=stamp, completed_at=stamp,
                     duration_seconds=duration, summary="ok",
                     task_id=task_id)
    return stamp


def test_the_health_endpoint_reports_a_draft_task_as_stalled(client, autonomy_dir):
    """#68's own row shape, read the way an agent reads it.

    `every-15min`, `draft`, last run 12.5 h before the pinned instant and
    `next_run` 12.25 h before it — the state the fleet's highest-volume task was
    in when the shipped surfaces called it the healthiest job on the board."""
    _task_file(autonomy_dir, 68, "Morning Brief Triage", "draft",
               last_run=(PIN - dt.timedelta(hours=12, minutes=30)).isoformat(),
               next_run=(PIN - dt.timedelta(hours=12, minutes=15)).isoformat())

    body = client.get("/api/autonomy/health", params={"days": 1}).json()
    assert "error" not in body, body

    stalled = {s["task_id"]: s for s in body["stalled"]}
    assert "68" in stalled, (
        f"the endpoint still cannot say it: stalled={sorted(stalled)}")
    assert stalled["68"]["status"] == "draft"
    assert stalled["68"]["gap_ratio"] >= 40, (
        f"gap_ratio {stalled['68']['gap_ratio']} for a 12.5 h gap at "
        "every-15min — the field is not measuring elapsed time over the "
        "declared frequency")
    assert stalled["68"]["fail_rate"] is None, (
        "a task with no in-window run was given a 0.0 rate it cannot support")

    row = next(t for t in body["idle_tasks"] if t["task_id"] == "68")
    assert row["hours_since_last_run"] == pytest.approx(12.5, abs=0.02), (
        f"hours_since_last_run {row['hours_since_last_run']} — the route is not "
        "reading the pinned clock, or the field is not stamp-derived")
    assert row["gap_ratio"] == pytest.approx(stalled["68"]["gap_ratio"], abs=0.01), (
        "the per-task row and the stalled list disagree about the same task")


def test_the_autonomy_health_tool_describes_fields_the_payload_actually_has():
    """The MCP tool description is the only thing telling an agent this answer
    exists, and the MCP process cannot be driven from here — so the description
    is checked against the keys the payload really carries rather than against a
    restatement. A field renamed on one side fails here naming both sides."""
    import json

    from agent_mcp.autonomy import list_tools

    desc = {t.name: t for t in asyncio.run(list_tools())}["autonomy_health"].description
    task = {"id": 68, "name": "Email & Calendar Triage", "status": "draft",
            "frequency": "every-15min",
            "last_run": (PIN - dt.timedelta(hours=12, minutes=30)).isoformat(),
            "next_run": (PIN - dt.timedelta(hours=12, minutes=15)).isoformat()}
    report = A.compute_health([], [task], 1, now=PIN)

    for field in ("hours_since_last_run", "gap_ratio"):
        assert f"`{field}`" in desc, f"the tool advertises no {field}"
        assert field in report["idle_tasks"][0], (
            f"{field} is advertised but absent from the row the tool returns")
    assert "`stalled`" in desc and "stalled" in report, (
        "the tool advertises a top-level `stalled` list the payload lacks")
    assert "`window_clamped_to_hours`" in desc, (
        "the tool still promises 'Fleet health over the last N days' without "
        "telling the agent the payload can say the window was shorter than that")
    assert "window_clamped_to_hours" in report["fleet"], (
        "window_clamped_to_hours is advertised but absent from the fleet block")
    assert json.dumps(report), "the payload must stay JSON-serialisable"


# ── #1401: the payload names the window its verdict actually rests on ─────────
#
# `days` was decoration: `?days=7`, `?days=30` and `?days=90` returned bodies
# identical once `days` and `generated_at` were stripped (measured on live state
# 2026-09-23T16:45Z), because every row in `workers.db` was written after the
# 2026-09-22T20:02Z rebuild — a ~21 h store delivered as a 168 h verdict, with
# no field anywhere naming the age of the oldest input it read.


def test_a_window_wider_than_the_store_reports_the_clamp(client, autonomy_dir, queue):
    """Clause 1: 20 hours of state under a 7-day request reads as a clamp.

    `app/routers/dashboard.py:458-462` already answers this way for the worker
    rollup — `window_hours` beside `window_start`, with the comment that a
    reader must be able to reproduce the number instead of guessing at
    "roughly two days". This endpoint had no equivalent, so the fleet's own
    watchdog could only report a verdict whose denominator it could not name.
    """
    _task_file(autonomy_dir, 30, "Nightly Knowledge Write", "up_next")
    _seed_run(queue, PIN - dt.timedelta(hours=20), "30")

    body = client.get("/api/autonomy/health", params={"days": 7}).json()
    assert "error" not in body, body
    fleet = body["fleet"]
    assert fleet["runs"] == 1, (
        "the seeded row fell outside the window; the fixture is not the "
        "20-hours-of-state-under-a-7-day-request case")
    assert A._parse_iso(fleet["oldest_input"]) == PIN - dt.timedelta(hours=20), (
        f"oldest_input {fleet['oldest_input']!r} is not the oldest run row the "
        "store holds")
    assert fleet["window_clamped_to_hours"] == pytest.approx(20.0, abs=0.02), (
        f"window_clamped_to_hours {fleet['window_clamped_to_hours']} — the "
        "verdict rests on 20 h and the payload does not say so")


def test_the_clamp_follows_the_store_not_the_request(client, autonomy_dir, queue):
    """One store, three requests: `days` alone describes only the covered ones.

    The filed symptom was that days=7/30/90 were indistinguishable with no
    field to tell them apart. This store reaches back 10 days, so a 1-day and a
    7-day request are genuinely covered and say nothing, while days=90 asks for
    nine times what exists — the shape the live box is in right now, where the
    item's own post-landing check is that days=90 names ~21 h.
    """
    _task_file(autonomy_dir, 30, "Nightly Knowledge Write", "up_next")
    _seed_run(queue, PIN - dt.timedelta(days=10), "30")
    _seed_run(queue, PIN - dt.timedelta(hours=2), "30")

    seen = {d: client.get("/api/autonomy/health", params={"days": d}).json()["fleet"]
            for d in (1, 7, 90)}
    for d in (1, 7):
        assert seen[d]["oldest_input"] is None and seen[d]["window_clamped_to_hours"] is None, (
            f"days={d} was reported clamped over a store that reaches back 10 "
            "days — the clamp is the shortfall, not the store's existence")
    assert seen[90]["window_clamped_to_hours"] == pytest.approx(240.0, abs=0.02), (
        f"days=90 reported {seen[90]['window_clamped_to_hours']}: the store is "
        "10 days deep and the payload still calls it a 90-day verdict")
    assert A._parse_iso(seen[90]["oldest_input"]) == PIN - dt.timedelta(days=10)


def test_a_full_window_report_carries_no_clamp(client, autonomy_dir, queue):
    """Clause 2: with a run row older than `now - days`, `days` is the whole story."""
    _task_file(autonomy_dir, 30, "Nightly Knowledge Write", "up_next")
    _seed_run(queue, PIN - dt.timedelta(days=10), "30")
    _seed_run(queue, PIN - dt.timedelta(hours=2), "30")

    body = client.get("/api/autonomy/health", params={"days": 7}).json()
    assert "error" not in body, body
    fleet = body["fleet"]
    assert fleet["runs"] == 1, (
        "the 10-day-old row leaked into a 7-day window; the fixture is not the "
        "covered-window case")
    assert fleet["oldest_input"] is None, (
        f"oldest_input {fleet['oldest_input']!r} on a window the store covers")
    assert fleet["window_clamped_to_hours"] is None, (
        f"window_clamped_to_hours {fleet['window_clamped_to_hours']} on a "
        "window the store covers — a reader cannot tell a full window from a "
        "short one if both carry a number")


def test_a_window_holding_no_rows_still_names_the_store(client, autonomy_dir, queue):
    """Clause 3: the clamp reads the store, so zero in-window rows is not "no data".

    A clamp computed from the window-filtered rows would report nothing here —
    the same empty shape as the clean bill of health this exists to expose. The
    store's own oldest row is the reference, so the response says what it holds.
    """
    old = PIN - dt.timedelta(days=20)
    _task_file(autonomy_dir, 44, "Weekly Hygiene", "up_next",
               last_run=old.isoformat(), next_run=old.isoformat())
    _seed_run(queue, old, "44")

    body = client.get("/api/autonomy/health", params={"days": 7}).json()
    assert "error" not in body, body
    fleet = body["fleet"]
    assert fleet["runs"] == 0, "the fixture's row fell inside the window"
    assert A._parse_iso(fleet["oldest_input"]) == old, (
        f"oldest_input {fleet['oldest_input']!r} — derived from the window's "
        "rows (empty) instead of the store's")
    assert fleet["window_clamped_to_hours"] == 0.0, (
        f"window_clamped_to_hours {fleet['window_clamped_to_hours']}: the "
        "requested window holds no run state at all, which is a clamp of zero "
        "hours, not a full window")


def test_an_idle_task_is_marked_unobserved_rather_than_clean(client, autonomy_dir, queue):
    """Clause 4: a task file whose `last_run` is in-window but has no run row.

    Both rows used to carry the hard-coded `"fail_rate": 0.0, "silent_rate":
    0.0` literal, byte-identical in shape to a task that ran and passed. The
    payload already uses `null` for the unobserved case elsewhere —
    `refuted_or_insufficient_rate` at `app/autonomy.py:3012`, and `stalled[].fail_rate`
    for a task with no in-window row — which is the precedent this follows. A
    row WITH runs keeps its numeric rates: the change marks no observation, it
    does not soften a measurement.
    """
    _task_file(autonomy_dir, 30, "Ran In Window", "up_next")
    _task_file(autonomy_dir, 68, "Last Run In Window No Rows", "up_next",
               last_run=(PIN - dt.timedelta(hours=6)).isoformat(),
               next_run=(PIN + dt.timedelta(hours=6)).isoformat())
    _seed_run(queue, PIN - dt.timedelta(hours=2), "30")

    body = client.get("/api/autonomy/health", params={"days": 7}).json()
    assert "error" not in body, body

    ran = next(t for t in body["tasks"] if t["task_id"] == "30")
    assert ran["runs"] == 1
    assert ran["fail_rate"] == 0.0 and ran["silent_rate"] == 0.0, (
        "a row with runs lost the numeric rates clause 4 says it keeps")
    assert "unobserved_in_window" not in ran, (
        "the marker belongs on the rows that were not observed")

    idle = next(t for t in body["idle_tasks"] if t["task_id"] == "68")
    assert idle["runs"] == 0
    assert idle["fail_rate"] is None, (
        "a zero-run task was still handed a 0.0 rate it cannot support")
    assert idle["silent_rate"] is None, (
        "a zero-run task was still handed a 0.0 silent rate")
    assert idle["unobserved_in_window"] is True, (
        "the row is null-rate but carries no marker a consumer flattening "
        "`tasks` and `idle_tasks` together can read")
    assert idle["never_run"] is False and idle["last_run"], (
        "the task's own stamps are the reason this row is interesting; they "
        "must survive the change")


def test_the_shared_gap_predicate_keeps_its_strict_bound_and_its_numbers():
    """#2342 clause 5: the fix belongs in the watchdog's reporting, not here.

    `next_run_gap` is the ONE predicate the `next_run` stall scan and
    `compute_health` have shared since #1121 — a board cannot be healthy in one
    surface and stalled in the other — and the strict `> one interval` bound is
    what keeps the #421 alarm quiet on an on-cadence nightly job (measured
    2026-09-18: healthy #51 reads 1.01 on the `last_run` reference, #68 reads
    169.4). #2342's disagreement is between a row and its run records, and the
    record cross-check therefore goes in `workers/fleet_watchdog.py`; widening
    this bound to catch an unstamped row would make the alarm the noisy one all
    over again and would move the numbers `compute_health` reports.

    So this node pins the whole returned dict at three shapes, which is the
    tightest available statement of "unchanged": exactly one period past is NOT a
    stall (strict), one second more IS, a measured row returns these six keys with
    these values, and a row with neither stamp returns no verdict. The key set is
    asserted because a field added here is a field `compute_health` and
    `/api/autonomy/health` start carrying.
    """
    keys = {"expected_interval_seconds", "hours_since_last_run",
            "hours_past_next_run", "gap_ratio", "never_run", "past_next_run"}

    day = dt.timedelta(days=1)
    measured = A.next_run_gap({"frequency": "daily",
                               "last_run": (PIN - day).isoformat(),
                               "next_run": (PIN - day).isoformat()}, now=PIN)
    assert set(measured) == keys, f"the shared predicate's shape moved: {sorted(measured)}"
    assert measured == {"expected_interval_seconds": 86400.0,
                        "hours_since_last_run": 24.0,
                        "hours_past_next_run": 24.0,
                        "gap_ratio": 1.0,
                        "never_run": False,
                        # Strict, as #421 shipped it: exactly one period is not
                        # MORE than one period.
                        "past_next_run": False}, (
        f"the bound or the numbers moved: {measured}")

    over = A.next_run_gap({"frequency": "daily",
                           "last_run": (PIN - day - dt.timedelta(seconds=1)).isoformat(),
                           "next_run": (PIN - day - dt.timedelta(seconds=1)).isoformat()},
                          now=PIN)
    assert over["past_next_run"] is True, (
        f"one second past one period must flag, or the bound became >=: {over}")
    assert over["gap_ratio"] == 1.0, (
        f"gap_ratio is elapsed over the declared period, not over the run: {over}")

    # The #96 shape, from this predicate's side: no stamps is no verdict, which is
    # exactly why the watchdog, not this function, has to notice it.
    unstamped = A.next_run_gap({"frequency": "weekly"}, now=PIN)
    assert unstamped == {"expected_interval_seconds": 604800.0,
                         "hours_since_last_run": None,
                         "hours_past_next_run": None,
                         "gap_ratio": None,
                         "never_run": True,
                         "past_next_run": False}, (
        f"a row with no stamps started answering like a healthy one: {unstamped}")


# ── #2619: an alarm NIGHT is a number the fleet can see ──────────────────────
#
# The knowledge-health report exits 2 when an alarm rail fires. From 2026-10-10
# to 2026-10-12 that happened three nights running, and every one of those runs
# is `status: success` in `workers.db` — the runner records the agent TURN, and
# the script's exit code dies inside a Bash call in it. This endpoint is what
# `queue-health-check` reads, so a monitor built to raise an alarm was raising
# nothing the fleet could see: `fail_rate`, `silent_rate`,
# `consecutive_failures` and `empty_runs` all stayed at their clean values.
#
# The marker the runner now writes into `runs.meta_json` (`script_alarm`,
# {"knowledge-health-report.py": 2}) is surfaced here as a per-task count plus a
# fleet total, so an alarm night is a NUMBER in the same table a human already
# reads. No reader of it opens the report file.

#: What a knowledge-health alarm night writes into the run row's `meta_json`.
ALARM_META = {"script_alarm": {"knowledge-health-report.py": 2}}


def _seed_script_run(queue, when: dt.datetime, task_id: str,
                     meta: dict | None = None) -> str:
    """One `scheduled-task` row for `task_id`, with `meta` in `meta_json`."""
    stamp = when.isoformat()
    queue.record_run(
        run_id=f"run-alarm-{task_id}-{stamp}", queue_id=None,
        source="scheduled-task", status="success",
        started_at=stamp, completed_at=stamp, duration_seconds=60.0,
        summary="Knowledge health report written", task_id=task_id,
        meta_json=json.dumps(meta) if meta is not None else "")
    return stamp


def test_the_endpoint_counts_alarm_runs_per_task_and_in_the_fleet(
        client, queue, autonomy_dir):
    """clause 3: an alarm night is visible on this endpoint, per task and fleet-wide.

    Three rows for task 60, all `status: success`, two carrying the marker —
    which is the real shape of 2026-10-10..12: three consecutive alarm nights,
    zero failures. The assertion that carries the weight is the one on
    `fail_rate`: if the alarm somehow also moved `fail_rate`, the count below
    would prove nothing about the gap this field exists to close.
    """
    _task_file(autonomy_dir, 60, "Knowledge health report", "up_next")
    for back in (1, 2, 3):
        _seed_script_run(queue, PIN - dt.timedelta(days=back),
                         "60", ALARM_META if back <= 2 else None)

    body = client.get("/api/autonomy/health", params={"days": 7}).json()
    assert "error" not in body, body
    row = next(t for t in body["tasks"] if t["task_id"] == "60")

    assert row["script_alarm_runs"] == 2, (
        f"two alarm nights in the window, endpoint said "
        f"{row.get('script_alarm_runs')!r}")
    assert row["fail_rate"] == 0.0 and row["consecutive_failures"] == 0, (
        "the control: this is supposed to be a run that looks healthy on every "
        "pre-existing number, and if it does not, the count above is not "
        "measuring what it claims")
    assert body["fleet"]["script_alarm_runs"] == 2, (
        f"fleet total {body['fleet'].get('script_alarm_runs')!r} does not "
        "match the per-task count")


def test_a_task_with_no_marker_reports_zero_not_missing(client, queue,
                                                        autonomy_dir):
    """The field is in the table's SHAPE, so the watchdog can print it unconditionally.

    Two negative cases in one node, because the failure mode is the same: a
    reader must be able to print `script_alarm_runs` for every row without
    first asking whether the row is old enough to predate the marker.
    (a) a run whose `meta_json` is empty — every row written before this change,
    which on 2026-10-12 is every one of the 663 `scheduled-task` rows in
    `~/lloyd-data/workers.db` (0 of them carry `script_alarm`), so this is the
    shape of every row the watchdog will read up to and including the last
    nightly before this lands;
    (b) a run whose `meta_json` will not parse, which must not raise and must
    not be counted;
    (c) an idle task, which has no run at all.
    """
    _task_file(autonomy_dir, 61, "Nightly tidy", "up_next")
    _seed_script_run(queue, PIN - dt.timedelta(hours=3), "61", None)
    stamp = (PIN - dt.timedelta(hours=4)).isoformat()
    queue.record_run(run_id="run-unparseable-61", queue_id=None,
                     source="scheduled-task", status="success",
                     started_at=stamp, completed_at=stamp,
                     duration_seconds=60.0, summary="x", task_id="61",
                     meta_json="{not json")
    _task_file(autonomy_dir, 62, "Never ran", "up_next")

    body = client.get("/api/autonomy/health", params={"days": 7}).json()
    row = next(t for t in body["tasks"] if t["task_id"] == "61")
    assert row["script_alarm_runs"] == 0, (
        f"empty and unparseable meta_json produced "
        f"{row['script_alarm_runs']!r} — an absent marker read as an alarm")
    idle = next(t for t in body["idle_tasks"] if t["task_id"] == "62")
    assert idle["script_alarm_runs"] == 0, (
        "an idle task has no `script_alarm_runs` at all, so a watchdog that "
        "prints the field for every row dies on the idle rows")


# ── #2619: the watchdog's own report has to carry the number ─────────────────
#
# Surfacing a field on the endpoint is half of it. `queue-health-check` is the
# job that reads this endpoint every 15 minutes, and what it PRINTS is what a
# human ever sees: a field the skill's Step 1 does not print and Step 2 does not
# flag is a field that exists in JSON and in no report. Task 60's three alarm
# nights were exactly that — every number the skill prints was healthy.
#
# These nodes read the live skill file rather than a copy, because a claim about
# what a running job prints is only worth what it says about the file the job
# actually loads. Unmarked on purpose — the same trade
# `tests/test_autonomy_run_record.py` documents for this skill's Step 2: the gate
# runner hardcodes `-m "not live_vault"`, so a marked node pins nothing. A
# missing skill FAILS here rather than skipping: a skipped check is exactly how
# the false green below survived, because nothing was reading the line.
#
# The false green, stated plainly because it is the reason these nodes exist:
# vault commit 864733c8 put `script_alarm_runs` into this skill's Step 1 and
# Step 2 while the code half was still in review, so between that commit and
# this one the watchdog prints `script-alarm runs: 0 across 0 tasks` on a box
# with three alarm nights — a check that reads as performed. Landing the count is
# what makes the sentence true; the lines are not this diff's to change.

#: The watchdog skill, on the live board this box's jobs load from.
def _skill_text() -> str:
    import os
    root = os.environ.get("LLOYD_OBSIDIAN_VAULT") or (Path.home() / "obsidian")
    path = Path(root) / "skills" / "queue-health-check" / "SKILL.md"
    if not path.is_file():
        pytest.fail(f"no queue-health-check skill on this box at {path} — the "
                    "job that reads this endpoint loads it from here")
    return path.read_text(encoding="utf-8")


def test_the_watchdog_skill_prints_the_alarm_field():
    """The acceptance check's watchdog half: an alarm night reaches the report.

    Both halves are pinned because they fail differently. Step 1 is the fleet
    table's code block — if it never prints the field, the number is invisible no
    matter what Step 2 says. Step 2 is the criteria list — a printed number
    nothing judges is printed decoration, and the whole reason task 60's alarm
    survived three nights is that no criterion could be met by it.
    """
    text = _skill_text()
    step1, _, rest = text.partition("## Step 2")
    assert step1.count("script_alarm_runs") >= 1, (
        "Step 1 does not print `script_alarm_runs`, so an alarm night never "
        "reaches the report a human reads")
    assert rest.split("## Step 3")[0].count("script_alarm_runs") >= 1, (
        "Step 2 does not judge `script_alarm_runs`, so the printed number "
        "decides nothing")
