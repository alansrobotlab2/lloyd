"""Subagent registry lifecycle.

The registry exists so an in-flight `Task` is visible *while* it is
in-flight. The failure that matters is a row that never closes (the
dashboard shows a phantom subagent forever) or one that closes with the
wrong status — which is what a blanket `finally` would have caused, since
`finish` is idempotent and would have stamped every completed run
"cancelled" before the real status arrived.
"""

from __future__ import annotations

import pytest

from agent_mcp import _subagent_registry as reg


@pytest.fixture(autouse=True)
def _clean():
    reg.reset()
    yield
    reg.reset()


def _register(**over):
    kwargs = dict(
        subagent_type="general-purpose",
        description="probe",
        prompt="do the thing",
        parent_session_id="sess-1",
        session_id="task:general-purpose:abcd",
        model="primary",
        max_turns=40,
    )
    kwargs.update(over)
    return reg.register(**kwargs)


def test_a_registered_run_is_immediately_visible_as_active():
    """The whole point: visible while running, not after it returns."""
    record = _register()
    active = reg.list_active()
    assert len(active) == 1
    assert active[0]["run_id"] == record.run_id
    assert active[0]["status"] == "running"
    assert active[0]["subagent_type"] == "general-purpose"
    assert reg.list_recent() == []


def test_progress_is_observable_before_the_run_finishes():
    record = _register()
    record.note_turn()
    record.note_tool("Grep")
    record.note_tool("Read")
    record.note_tool("Grep")

    row = reg.list_active()[0]
    assert row["turns"] == 1
    assert row["tool_call_count"] == 3
    assert row["tool_counts"] == {"Grep": 2, "Read": 1}
    assert row["last_tool"] == "Grep"
    assert row["elapsed_s"] >= 0


def test_finish_moves_the_row_from_active_to_recent():
    record = _register()
    reg.finish(record, status="completed", stop_reason="stop", response_chars=42)

    assert reg.list_active() == []
    recent = reg.list_recent()
    assert len(recent) == 1
    assert recent[0]["status"] == "completed"
    assert recent[0]["response_chars"] == 42
    assert recent[0]["finished_at"] is not None


def test_finish_is_idempotent_and_the_first_status_wins():
    """builtin_task closes on the success path and again on unwind. The
    second call must not overwrite the real status or double-append."""
    record = _register()
    reg.finish(record, status="completed", stop_reason="stop")
    reg.finish(record, status="cancelled", stop_reason="cancelled")

    recent = reg.list_recent()
    assert len(recent) == 1
    assert recent[0]["status"] == "completed"


def test_failed_runs_keep_their_diagnosis():
    record = _register()
    record.note_tool("Bash")
    reg.finish(
        record,
        status="failed",
        stop_reason="max_turns",
        error="exhausted its 40-turn budget",
    )
    row = reg.list_recent()[0]
    assert row["status"] == "failed"
    assert row["stop_reason"] == "max_turns"
    assert "40-turn budget" in row["error"]


def test_active_runs_are_ordered_longest_running_first():
    """The one that has been going longest is the one worth looking at."""
    first = _register(description="first")
    second = _register(description="second")
    first.started_at -= 60  # pretend it started a minute earlier

    assert [r["description"] for r in reg.list_active()] == ["first", "second"]
    assert second.run_id != first.run_id


def test_recent_is_newest_first_and_bounded():
    """This is a live view, not an audit log — event_logs/ holds history."""
    for i in range(reg._RECENT_LIMIT + 5):
        reg.finish(_register(description=f"run-{i}"), status="completed")

    recent = reg.list_recent()
    assert len(recent) == reg._RECENT_LIMIT
    assert recent[0]["description"] == f"run-{reg._RECENT_LIMIT + 4}"


def test_prompt_preview_is_truncated():
    record = _register(prompt="x" * 5000)
    assert len(reg.list_active()[0]["prompt_preview"]) == 200
    reg.finish(record, status="completed")


def test_snapshot_shape_matches_what_the_dashboard_reads():
    running = _register(description="in flight")
    reg.finish(_register(description="done"), status="completed")

    snap = reg.snapshot()
    assert snap["active_count"] == 1
    assert [r["description"] for r in snap["active"]] == ["in flight"]
    assert [r["description"] for r in snap["recent"]] == ["done"]
    assert running.status == "running"
