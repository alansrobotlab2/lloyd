"""Live per-turn activity — what the dashboard's agent panel reads.

"Busy" is equally true of a turn that is prefilling a 160k context, one
four minutes into a `Bash` build, and one wedged on a dead engine. The
activity record is the only thing that tells an operator which of those
they are looking at, so the properties worth pinning are:

* it is **display state, not turn state** — the loop never reads it back,
  so writing it must never be able to raise or block;
* it is written **per streamed token** on the text path, so an unchanged
  state has to be a cheap no-op rather than a fresh timestamp;
* it must not **leak across turns** — a stale `Bash` line under an idle
  session reads as a hung tool call.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from app.sessions_io import (
    SessionTurn,
    _get_or_create_queue,
    _session_queues,
    active_sessions_snapshot,
    set_turn_activity,
    tool_activity_detail,
)


@pytest.fixture(autouse=True)
def _clean_queues():
    _session_queues.clear()
    yield
    _session_queues.clear()


def _running(session_id: str) -> SessionTurn:
    """A session with one turn promoted to current, as the consumer does."""
    q = _get_or_create_queue(session_id)
    turn = SessionTurn(
        turn_id=uuid.uuid4().hex[:8],
        source="user",
        payload={},
        enqueued_at=datetime.now(),
    )
    turn.started_at = datetime.now()
    q.current = turn
    return turn


# ── Detail extraction ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "args,want",
    [
        ('{"command": "supervisorctl restart lloyd-mc:lloyd-backend"}',
         "supervisorctl restart lloyd-mc:lloyd-backend"),
        ('{"file_path": "/home/x/lloyd/app/harness/loop.py"}',
         "/home/x/lloyd/app/harness/loop.py"),
        ('{"pattern": "set_turn_activity", "path": "app"}', "set_turn_activity"),
        # Unlisted tool: first string argument is right far more often
        # than showing nothing at all.
        ('{"widget": "kv-cache"}', "kv-cache"),
        # Newlines would break the one-line row.
        ('{"command": "line one\\nline two"}', "line one line two"),
    ],
)
def test_tool_activity_detail(args, want):
    assert tool_activity_detail(args) == want


def test_tool_activity_detail_never_raises_on_junk():
    """A tool whose arguments failed to parse still deserves its name shown."""
    assert tool_activity_detail("not json") == ""
    assert tool_activity_detail("") == ""
    assert tool_activity_detail("[1, 2, 3]") == ""
    assert tool_activity_detail('{"n": 42}') == ""


def test_tool_activity_detail_truncates():
    got = tool_activity_detail('{"command": "%s"}' % ("x" * 500), limit=40)
    assert len(got) == 40
    assert got.endswith("…")


# ── Recording ──────────────────────────────────────────────────────────


def test_activity_is_recorded_on_the_running_turn():
    turn = _running("s1")
    set_turn_activity("s1", "tool", "Bash", "pytest -q")
    assert turn.activity["kind"] == "tool"
    assert turn.activity["label"] == "Bash"
    assert turn.activity["detail"] == "pytest -q"
    assert turn.activity["at"]


def test_unchanged_state_does_not_churn_the_timestamp():
    """The streaming path calls this per token. Re-stamping every call
    would make an idle-looking `at` field meaningless."""
    turn = _running("s1")
    set_turn_activity("s1", "responding")
    first = turn.activity
    set_turn_activity("s1", "responding")
    assert turn.activity is first

    set_turn_activity("s1", "thinking")
    assert turn.activity is not first


def test_setting_activity_on_an_idle_session_is_a_no_op():
    """Late events after a turn closes must not resurrect a `current`."""
    set_turn_activity("nobody-home", "tool", "Bash")
    assert "nobody-home" not in _session_queues

    q = _get_or_create_queue("s1")
    set_turn_activity("s1", "tool", "Bash")
    assert q.current is None


# ── Snapshot ───────────────────────────────────────────────────────────


def test_snapshot_carries_activity_for_running_turns():
    _running("s1")
    set_turn_activity("s1", "tool", "Bash", "pytest -q")
    (entry,) = active_sessions_snapshot()
    assert entry["session_id"] == "s1"
    assert entry["running"] is True
    assert entry["activity"]["label"] == "Bash"


def test_snapshot_activity_is_a_copy_not_the_live_dict():
    """The snapshot is serialized to JSON on a different task than the one
    mutating it; handing out the live dict invites a torn read."""
    turn = _running("s1")
    set_turn_activity("s1", "tool", "Bash", "pytest -q")
    (entry,) = active_sessions_snapshot()
    assert entry["activity"] is not turn.activity
    assert entry["activity"] == turn.activity


def test_queued_sessions_report_no_activity():
    """A queued turn is not doing anything yet, and showing the previous
    turn's tool under it reads as a hung call."""
    q = _get_or_create_queue("s1")
    q.pending_user.append(SessionTurn(
        turn_id="t1", source="user", payload={}, enqueued_at=datetime.now(),
    ))
    (entry,) = active_sessions_snapshot()
    assert entry["running"] is False
    assert entry["activity"] is None
