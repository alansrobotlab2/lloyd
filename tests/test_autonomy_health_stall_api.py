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
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import autonomy as A
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
def client(autonomy_dir, tmp_path, monkeypatch):
    """The real route, over a real queue file, with the clock pinned once."""
    from workers.queue import WorkQueue

    q = WorkQueue(tmp_path / "health-seam.db")
    monkeypatch.setattr("workers.queue.get_queue", lambda: q)
    monkeypatch.setattr(A, "_utcnow", lambda: PIN)
    app = FastAPI()
    app.include_router(ROUTER.router)
    with TestClient(app) as c:
        yield c


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
    assert json.dumps(report), "the payload must stay JSON-serialisable"
