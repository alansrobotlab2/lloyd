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
