"""Dashboard section aggregation.

Two pieces of judgment are encoded here and both are easy to get wrong in
a way that looks fine:

* **Overdue is not "next up."** Sorting every scheduled task by `next_run`
  ascending and labelling the head "next up" makes a fleet whose ticker
  is months behind read as a healthy schedule — the most overdue task is
  displayed exactly where the soonest one belongs.
* **`completed` is not open work.** The queue's depth table is dominated
  by lifetime `completed` rows (3,400+ on this box). Summing it into a
  backlog figure buries the handful of items actually waiting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.routers import dashboard as dash


@pytest.fixture(autouse=True)
def _clear_cache():
    """Sections that walk the vault are TTL-cached; the cache is
    module-global and would leak fixtures between tests."""
    dash._cache.clear()
    yield
    dash._cache.clear()


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Point `Path.home()` at a scratch tree with the two vault dirs."""
    (tmp_path / "obsidian" / "autonomy").mkdir(parents=True)
    (tmp_path / "obsidian" / "backlog").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _iso(**delta) -> str:
    return (datetime.now(timezone.utc) + timedelta(**delta)).isoformat()


def _task(vault, name, **fm):
    """Write one autonomy task file. Only NN-*.md files are tasks."""
    body = "\n".join(f"{k}: {v}" for k, v in fm.items())
    path = vault / "obsidian" / "autonomy" / f"{len(list((vault / 'obsidian' / 'autonomy').glob('*.md'))) + 1:02d}-{name}.md"
    path.write_text(f"---\nname: {name}\n{body}\n---\n\nbody\n")
    return path


def _backlog_item(vault, n, name, status, board="lloyd"):
    path = vault / "obsidian" / "backlog" / f"{n}-{name}.md"
    path.write_text(f"---\nname: {name}\nstatus: {status}\nboard: {board}\n---\n\n# {name}\n")
    return path


# ── Autonomy ───────────────────────────────────────────────────────────


def test_overdue_tasks_are_not_reported_as_upcoming(vault):
    """The regression this split exists for: a task whose next_run passed
    months ago must not head the "next up" list."""
    _task(vault, "stale", status="up_next", frequency="daily", next_run=_iso(days=-70))
    _task(vault, "soon", status="up_next", frequency="daily", next_run=_iso(hours=8))

    out = dash._autonomy()
    assert [t["name"] for t in out["overdue"]] == ["stale"]
    assert [t["name"] for t in out["upcoming"]] == ["soon"]
    assert out["overdue_count"] == 1


def test_overdue_is_worst_first_and_upcoming_is_soonest_first(vault):
    _task(vault, "late-a", status="up_next", next_run=_iso(days=-2))
    _task(vault, "late-b", status="up_next", next_run=_iso(days=-30))
    _task(vault, "next-a", status="up_next", next_run=_iso(hours=20))
    _task(vault, "next-b", status="up_next", next_run=_iso(hours=2))

    out = dash._autonomy()
    assert [t["name"] for t in out["overdue"]] == ["late-b", "late-a"]
    assert [t["name"] for t in out["upcoming"]] == ["next-b", "next-a"]


def test_failed_tasks_are_separated_from_the_schedule(vault):
    """A failed task is not waiting to run; it needs attention."""
    _task(vault, "broken", status="failed", next_run=_iso(days=-1), last_run=_iso(days=-1))
    _task(vault, "fine", status="up_next", next_run=_iso(hours=5))

    out = dash._autonomy()
    assert [t["name"] for t in out["failing"]] == ["broken"]
    assert [t["name"] for t in out["overdue"]] == []
    assert [t["name"] for t in out["upcoming"]] == ["fine"]
    assert out["by_status"] == {"failed": 1, "up_next": 1}


def test_non_task_files_are_ignored(vault):
    """_config.md, reports and notes live in the same directory."""
    _task(vault, "real", status="up_next", next_run=_iso(hours=1))
    (vault / "obsidian" / "autonomy" / "_config.md").write_text("---\nfoo: 1\n---\n")
    (vault / "obsidian" / "autonomy" / "report-2026.md").write_text("---\nstatus: failed\n---\n")

    out = dash._autonomy()
    assert out["total"] == 1
    assert out["by_status"] == {"up_next": 1}


def test_a_task_with_no_next_run_is_neither_overdue_nor_upcoming(vault):
    _task(vault, "unscheduled", status="paused")
    out = dash._autonomy()
    assert out["total"] == 1
    assert out["overdue"] == [] and out["upcoming"] == []


def test_missing_autonomy_directory_is_empty_not_an_error(vault, monkeypatch):
    (vault / "obsidian" / "autonomy").rmdir()
    out = dash._autonomy()
    assert out["total"] == 0
    assert out["by_status"] == {}


# ── Backlog ────────────────────────────────────────────────────────────


def test_open_count_excludes_closed_statuses(vault):
    for i, status in enumerate(["done", "done", "closed", "up_next", "draft", "review"]):
        _backlog_item(vault, 100 + i, f"t{i}", status)

    out = dash._backlog()
    assert out["total"] == 6
    # done x2 + closed x1 are off the board.
    assert out["open_total"] == 3
    assert out["by_status"]["done"] == 2


def test_boards_are_ranked_by_open_work(vault):
    _backlog_item(vault, 1, "a", "up_next", board="lloyd")
    _backlog_item(vault, 2, "b", "up_next", board="lloyd")
    _backlog_item(vault, 3, "c", "up_next", board="alfie")
    _backlog_item(vault, 4, "d", "done", board="alfie")

    out = dash._backlog()
    assert [b["board"] for b in out["by_board"]] == ["lloyd", "alfie"]
    assert out["by_board"][0] == {"board": "lloyd", "open": 2, "total": 2}
    # A board whose items are all done still appears, with open 0.
    assert out["by_board"][1] == {"board": "alfie", "open": 1, "total": 2}


def test_recently_touched_lists_only_open_items(vault):
    _backlog_item(vault, 1, "shipped", "done")
    _backlog_item(vault, 2, "active", "up_next")

    out = dash._backlog()
    assert [t["name"] for t in out["recent_open"]] == ["active"]


# ── Caching ────────────────────────────────────────────────────────────


def test_vault_scans_are_cached_between_polls(vault):
    """331 markdown files must not be re-read every 2 seconds."""
    _backlog_item(vault, 1, "a", "up_next")
    first = dash._backlog()
    assert first["total"] == 1

    _backlog_item(vault, 2, "b", "up_next")
    assert dash._backlog()["total"] == 1, "second poll should hit the cache"

    dash._cache.clear()
    assert dash._backlog()["total"] == 2


# ── Workers ────────────────────────────────────────────────────────────


def test_open_states_exclude_completed():
    """`completed` is the dominant depth key; counting it as open work
    would report thousands of items waiting."""
    assert "completed" not in dash._OPEN_STATES
    assert "queued" in dash._OPEN_STATES and "running" in dash._OPEN_STATES


# ── Agent panel ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_primary_state_names_sessions_by_title(monkeypatch):
    """The panel renders titles, but the snapshot they decorate must stay
    pure in-memory queue state — it is also the selfmod promoter's idle
    gate, and a disk read there would put the filesystem in front of a
    restart decision."""
    from app import session_titles

    monkeypatch.setattr(
        dash.sessions_io, "active_sessions_snapshot",
        lambda: [{
            "session_id": "20260906_214751_iv1620",
            "running": True, "turn_id": "t1", "source": "user",
            "started_at": None, "enqueued_at": None, "preempted": False,
            "activity": {"kind": "tool", "label": "Bash",
                         "detail": "pytest -q", "at": "now"},
            "pending_user": 0, "pending_ambient": 0,
        }],
    )
    monkeypatch.setattr(
        session_titles, "titles_for",
        lambda ids: {i: "Setting up TTS with cloned voice" for i in ids},
    )

    state = await dash._primary_state()
    (session,) = state["sessions"]
    assert session["title"] == "Setting up TTS with cloned voice"
    assert session["activity"]["label"] == "Bash"


@pytest.mark.asyncio
async def test_primary_state_without_active_sessions_reads_no_titles(monkeypatch):
    """No running turns means no disk touched — this endpoint is polled
    every 2 seconds all day."""
    from app import session_titles

    monkeypatch.setattr(dash.sessions_io, "active_sessions_snapshot", lambda: [])

    def _never(ids):
        raise AssertionError("titles_for called with no active sessions")

    monkeypatch.setattr(session_titles, "titles_for", _never)
    assert (await dash._primary_state())["sessions"] == []


# ── Degradation ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failing_section_becomes_an_error_payload_not_an_exception():
    """One broken panel must not blank the page — the dashboard is most
    useful exactly when something is broken."""
    async def _boom():
        raise RuntimeError("supervisord is wedged")

    name, value = await dash._gather("services", _boom())
    assert name == "services"
    assert value == {"error": "RuntimeError: supervisord is wedged"}


# ── Recent chats ───────────────────────────────────────────────────────


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    """Point SESSIONS_DIR at a scratch tree."""
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr("app.paths.SESSIONS_DIR", d)
    return d


def _session(sessions, session_id, last_active, *, mtime=None, **extra):
    """Write one session file. `last_active` is naive local, as on disk."""
    import json
    import os

    data = {
        "session_id": session_id,
        "last_active": last_active.replace(tzinfo=None).isoformat(),
        "preview": "hello",
        "message_count": 2,
        "platform": "mission-control",
        "messages": [],
        **extra,
    }
    path = sessions / f"{session_id}.json"
    path.write_text(json.dumps(data))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _local(**delta):
    return datetime.now() + timedelta(**delta)


def test_recent_chats_are_ordered_by_last_active_not_mtime(sessions, no_live_turns):
    """A background writer — the titler, post-session capture, TodoWrite —
    touches a session file long after the talking stopped. Under an mtime
    sort that silently promotes an old chat to the top of the panel."""
    import time

    now = time.time()
    _session(sessions, "old_but_touched", _local(hours=-9), mtime=now)
    _session(sessions, "actually_recent", _local(minutes=-5), mtime=now - 3600)

    out = dash._recent_sessions()
    assert [s["session_id"] for s in out["sessions"]] == [
        "actually_recent",
        "old_but_touched",
    ]


def test_recent_chats_exclude_sessions_with_a_live_turn(sessions, monkeypatch):
    """A running or queued chat is already on the panel beside this one.
    Showing it in both costs the operator what this half is for."""
    _session(sessions, "live", _local(minutes=-1))
    _session(sessions, "finished", _local(minutes=-2))
    monkeypatch.setattr(
        dash.sessions_io, "active_sessions_snapshot",
        lambda: [{"session_id": "live", "running": True}],
    )

    out = dash._recent_sessions()
    assert [s["session_id"] for s in out["sessions"]] == ["finished"]


def test_a_chat_that_starts_a_turn_leaves_the_panel_within_the_poll(
    sessions, monkeypatch
):
    """The scan is cached for 10s; the live filter must not be. Otherwise a
    chat that just started reads as finished until the cache expires."""
    _session(sessions, "a", _local(minutes=-1))
    _session(sessions, "b", _local(minutes=-2))
    live: list[dict] = []
    monkeypatch.setattr(
        dash.sessions_io, "active_sessions_snapshot", lambda: list(live)
    )

    assert [s["session_id"] for s in dash._recent_sessions()["sessions"]] == ["a", "b"]
    live.append({"session_id": "a", "running": True})
    assert [s["session_id"] for s in dash._recent_sessions()["sessions"]] == ["b"]


def test_scheduled_tasks_are_not_chats(sessions, no_live_turns):
    """Autonomy runs have their own panel."""
    _session(sessions, "task", _local(minutes=-1), platform="autonomy")
    _session(sessions, "chat", _local(minutes=-2))

    out = dash._recent_sessions()
    assert [s["session_id"] for s in out["sessions"]] == ["chat"]


def test_last_active_is_serialised_as_explicit_utc(sessions, no_live_turns):
    """`last_active` on disk is a naive *local* stamp. Handing that
    straight to `new Date()` shifts every row by the box's offset."""
    _session(sessions, "chat", _local(minutes=-1))

    stamp = dash._recent_sessions()["sessions"][0]["last_active"]
    assert stamp.endswith("Z")
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds() - 60) < 5


def test_only_two_chats_are_shown(sessions, no_live_turns):
    for i in range(5):
        _session(sessions, f"s{i}", _local(minutes=-i - 1))
    assert len(dash._recent_sessions()["sessions"]) == 2


# ── Recent chats ───────────────────────────────────────────────────────


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    """Point SESSIONS_DIR at a scratch tree."""
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr("app.paths.SESSIONS_DIR", d)
    return d


def _session(sessions, session_id, last_active, *, mtime=None, **extra):
    """Write one session file. `last_active` is naive local, as on disk."""
    import json
    import os

    data = {
        "session_id": session_id,
        "last_active": last_active.replace(tzinfo=None).isoformat(),
        "preview": "hello",
        "message_count": 2,
        "platform": "mission-control",
        "messages": [],
        **extra,
    }
    path = sessions / f"{session_id}.json"
    path.write_text(json.dumps(data))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _local(**delta):
    return datetime.now() + timedelta(**delta)


@pytest.fixture
def no_live_turns(monkeypatch):
    """For the tests that are about the disk scan, not the run queue."""
    monkeypatch.setattr(dash.sessions_io, "active_sessions_snapshot", lambda: [])


def test_recent_chats_are_ordered_by_last_active_not_mtime(sessions):
    """A background writer — the titler, post-session capture, TodoWrite —
    touches a session file long after the talking stopped. Under an mtime
    sort each of those silently promotes an old chat to the top."""
    import time

    now = time.time()
    _session(sessions, "old_but_touched", _local(hours=-9), mtime=now)
    _session(sessions, "actually_recent", _local(minutes=-5), mtime=now - 3600)

    out = dash._recent_sessions()
    assert [s["session_id"] for s in out["sessions"]] == [
        "actually_recent",
        "old_but_touched",
    ]


def test_recent_chats_exclude_sessions_with_a_live_turn(sessions, monkeypatch):
    """A running or queued chat is already on the panel beside this one.
    Rendering it in both costs the operator what this half is for."""
    _session(sessions, "live", _local(minutes=-1))
    _session(sessions, "finished", _local(minutes=-2))
    monkeypatch.setattr(
        dash.sessions_io, "active_sessions_snapshot",
        lambda: [{"session_id": "live", "running": True}],
    )

    out = dash._recent_sessions()
    assert [s["session_id"] for s in out["sessions"]] == ["finished"]


def test_a_chat_that_starts_a_turn_leaves_the_panel_immediately(
    sessions, monkeypatch
):
    """The scan is cached for 10s; the live filter must not be, or a chat
    that just started reads as finished until the cache expires."""
    _session(sessions, "a", _local(minutes=-1))
    _session(sessions, "b", _local(minutes=-2))
    live: list[dict] = []
    monkeypatch.setattr(
        dash.sessions_io, "active_sessions_snapshot", lambda: list(live)
    )

    assert [s["session_id"] for s in dash._recent_sessions()["sessions"]] == ["a", "b"]
    live.append({"session_id": "a", "running": True})
    assert [s["session_id"] for s in dash._recent_sessions()["sessions"]] == ["b"]


def test_scheduled_tasks_are_not_chats(sessions):
    """Autonomy runs have their own panel."""
    _session(sessions, "task", _local(minutes=-1), platform="autonomy")
    _session(sessions, "chat", _local(minutes=-2))

    out = dash._recent_sessions()
    assert [s["session_id"] for s in out["sessions"]] == ["chat"]


def test_last_active_is_serialised_as_explicit_utc(sessions):
    """`last_active` on disk is a naive *local* stamp. Handing that
    straight to `new Date()` shifts every row by the box's offset."""
    _session(sessions, "chat", _local(minutes=-1))

    stamp = dash._recent_sessions()["sessions"][0]["last_active"]
    assert stamp.endswith("Z")
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds() - 60) < 5


def test_only_two_chats_are_shown(sessions):
    for i in range(5):
        _session(sessions, f"s{i}", _local(minutes=-i - 1))
    assert len(dash._recent_sessions()["sessions"]) == 2
