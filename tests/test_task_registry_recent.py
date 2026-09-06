"""Finished background tasks stay visible.

`Bash(run_in_background=true)` records used to leave the dashboard the
instant the subprocess exited: `list_active` filters on
`status == "running"`, and nothing rendered the rest. A task that died
three seconds in was indistinguishable from one that never started —
the opposite of what a background task most needs to report, since
nobody is watching its terminal.

What is worth pinning:

* finished records are still reachable, newest-finished first;
* the list is **bounded by the limit, not by eviction** — `_records` is
  deliberately kept whole so a later `get(task_id)` can still hand the
  model an output path to Read;
* a running task never appears in it, and a finished one never appears
  in `list_active`.
"""

from __future__ import annotations

import time

import pytest

from agent_mcp import _task_registry as reg


@pytest.fixture(autouse=True)
def _clean():
    reg._records.clear()
    reg._pending_by_session.clear()
    yield
    reg._records.clear()
    reg._pending_by_session.clear()


def _record(task_id: str, *, status: str, finished_at: float | None,
            exit_code: int | None = None, started_at: float = 1000.0):
    """A TaskRecord without a real subprocess behind it."""
    rec = reg.TaskRecord(
        task_id=task_id,
        session_id="s1",
        command=f"echo {task_id}",
        description=f"task {task_id}",
        output_path=reg.TASKS_DIR / f"{task_id}.log",
        process=None,  # type: ignore[arg-type]
        log_fd=-1,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        status=status,
    )
    reg._records[task_id] = rec
    return rec


def test_finished_tasks_are_reachable_after_they_exit():
    _record("bg-done", status="completed", finished_at=1005.0, exit_code=0)
    assert [r.task_id for r in reg.list_recent()] == ["bg-done"]
    assert reg.list_active() == []


def test_a_failure_survives_where_it_used_to_vanish():
    """The whole point: a task that died at second 3 must still be there,
    carrying the exit code that says it died."""
    _record("bg-boom", status="failed", finished_at=1003.0, exit_code=1)
    (row,) = reg.list_recent()
    assert row.status == "failed"
    assert row.exit_code == 1


def test_recent_is_newest_finished_first():
    _record("bg-old", status="completed", finished_at=1001.0, exit_code=0)
    _record("bg-new", status="completed", finished_at=1009.0, exit_code=0)
    _record("bg-mid", status="killed", finished_at=1005.0)
    assert [r.task_id for r in reg.list_recent()] == ["bg-new", "bg-mid", "bg-old"]


def test_running_tasks_are_not_recent():
    _record("bg-live", status="running", finished_at=None)
    _record("bg-done", status="completed", finished_at=1002.0, exit_code=0)
    assert [r.task_id for r in reg.list_recent()] == ["bg-done"]
    assert [r.task_id for r in reg.list_active()] == ["bg-live"]


def test_limit_bounds_the_payload_without_evicting_records():
    """`_records` is the model's lookup table for output paths. Trimming it
    to keep the dashboard small would break a later Read."""
    for i in range(25):
        _record(f"bg-{i:02d}", status="completed", finished_at=1000.0 + i, exit_code=0)

    assert len(reg.list_recent()) == 10
    assert len(reg.list_recent(limit=3)) == 3
    assert [r.task_id for r in reg.list_recent(limit=3)] == ["bg-24", "bg-23", "bg-22"]
    # Every record is still resolvable by id.
    assert len(reg._records) == 25
    assert reg.get("bg-00") is not None


def test_a_finished_record_missing_its_timestamp_still_sorts():
    """Defensive: a record whose waiter was cancelled mid-flight can carry
    a terminal status with no `finished_at`. It must not crash the sort."""
    _record("bg-nots", status="killed", finished_at=None)
    _record("bg-ts", status="completed", finished_at=1004.0, exit_code=0)
    assert [r.task_id for r in reg.list_recent()] == ["bg-ts", "bg-nots"]


def test_finished_row_reports_duration_not_time_since_start():
    """A finished task's elapsed time must stop ticking. Measured against
    `now`, a task that ran for two seconds reads as hours old by evening."""
    from agent_mcp.main import _task_row

    started = time.time() - 3600
    rec = _record("bg-done", status="completed",
                  started_at=started, finished_at=started + 2.0, exit_code=0)
    row = _task_row(rec)
    assert row["elapsed_s"] == 2.0
    assert row["exit_code"] == 0
    assert row["status"] == "completed"


def test_running_row_reports_time_since_start():
    from agent_mcp.main import _task_row

    rec = _record("bg-live", status="running",
                  started_at=time.time() - 5.0, finished_at=None)
    assert _task_row(rec)["elapsed_s"] == pytest.approx(5.0, abs=0.5)
