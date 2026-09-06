"""Run timeout precedence and stuck-task recovery.

Why this file exists
--------------------
Two timeout caps apply to every autonomy run — the task's frontmatter
`timeout_seconds` (`autonomy.py`) and the work source's
`max_duration_seconds` (`workers/pool.py`). The recorded rule is "both apply,
min wins", and the correction log spent a cycle discovering that the *opposite*
belief ("frontmatter timeout_seconds is not read") had been written into memory
as fact.

The incident that fixed is worth keeping pinned: when the two values were
**equal**, the pool timer won the race and cancelled the coroutine before its
own handler ran — no run record, no activity-log line, and the task left
`in_progress` on disk. `autonomy.py:682` now subtracts a margin for exactly that
reason.

`AUTONOMY_DIR` is `~/obsidian/autonomy` — a protected vault path — so every test
that touches the filesystem redirects it to a tmp dir. `_find_task_file`,
`_update_task_field` and `_append_activity_log` all resolve through that one
module global, so patching it isolates all three.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import autonomy
from workers import pool as worker_pool

DEFAULT_TASK_TIMEOUT = 1800          # autonomy.py's `or 1800` fallback
MARGIN = autonomy._POOL_TIMEOUT_MARGIN
POOL_DEFAULT = worker_pool._DEFAULT_MAX_DURATION_SECONDS


def resolve_timeout(declared: int, max_duration: int | None) -> int:
    """Mirrors autonomy.py:676-683 verbatim.

    A copy, not an import: the expression is inline inside `run_task`, which
    needs the whole harness to invoke. Extracting it is the right refactor and
    out of scope for a test-only change; until then this pins the rule and the
    constants it depends on, and fails if either moves.
    """
    if max_duration:
        return max(60, min(declared, int(max_duration) - MARGIN))
    return declared


# ── precedence rule ──────────────────────────────────────────────────────────

def test_task_timeout_is_used_when_the_pool_sets_no_cap():
    assert resolve_timeout(600, None) == 600


def test_zero_pool_cap_is_treated_as_absent():
    """`if max_duration:` — a falsy 0 must not clamp everything to 60s."""
    assert resolve_timeout(600, 0) == 600


def test_smaller_task_timeout_wins_over_a_larger_pool_cap():
    assert resolve_timeout(300, 1800) == 300


def test_smaller_pool_cap_wins_over_a_larger_task_timeout():
    assert resolve_timeout(1800, 1200) == 1200 - MARGIN


def test_equal_timeouts_yield_the_pool_cap_minus_the_margin():
    """The race condition, pinned. With both at 1800 the pool won at exactly
    1800 and cancelled the coroutine before its own handler could record the
    run — the task stayed `in_progress` on disk with no run record. Equal must
    never resolve to equal."""
    assert resolve_timeout(1800, 1800) == 1800 - MARGIN
    assert resolve_timeout(1800, 1800) < 1800


def test_the_task_timer_always_fires_before_the_pool_timer():
    """The invariant behind every case: strict inequality, for every cap."""
    for declared in (60, 300, 600, 1800, 3600, 7200):
        for cap in (300, 600, 1800, 3600):
            assert resolve_timeout(declared, cap) < cap, (declared, cap)


def test_margin_does_not_gobble_a_short_cap():
    assert resolve_timeout(1800, 45) == 60


def test_floor_is_sixty_seconds():
    assert resolve_timeout(1800, 1) == 60


def test_a_one_second_margin_cap_still_leaves_working_time():
    assert resolve_timeout(1800, MARGIN + 5) == 60


def test_the_margin_constant_is_a_positive_number_of_seconds():
    assert isinstance(MARGIN, int) and 0 < MARGIN < 600


def test_defaults_are_the_documented_numbers():
    """Both defaults are load-bearing: a task with no `timeout_seconds` gets 30
    minutes, and a source with no `max_duration_seconds` gets 15. The
    entity-resolution-sweep poisoning was diagnosed against the *source* cap."""
    assert DEFAULT_TASK_TIMEOUT == 1800
    assert POOL_DEFAULT == 900
    assert resolve_timeout(DEFAULT_TASK_TIMEOUT, POOL_DEFAULT) == POOL_DEFAULT - MARGIN


def test_the_sweep_cap_pair_is_resolvable():
    """Concrete regression from the incident log: sweep at 1800 against a 1800
    source cap previously resolved to a tie."""
    assert resolve_timeout(1800, 1800) == 1770


# ── stuck-task recovery ──────────────────────────────────────────────────────

@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    d = tmp_path / "autonomy"
    d.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", d)
    return d


def task_file(dir_, task_id, *, status, updated, timeout=None, name="some task"):
    front = [
        "---",
        f"id: {task_id}",
        f"name: {name}",
        f"status: {status}",
        f"updated: {updated}",
    ]
    if timeout is not None:
        front.append(f"timeout_seconds: {timeout}")
    front.append("---")
    path = dir_ / f"{task_id}-{name.replace(' ', '-')}.md"
    path.write_text("\n".join(front) + "\n\n# Body\n", encoding="utf-8")
    return path


def iso(offset_seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)).isoformat()


def test_no_stuck_tasks_is_a_noop(autonomy_dir):
    task_file(autonomy_dir, "10", status="up_next", updated=iso(99999))
    assert autonomy.recover_stuck_tasks() == []


def test_fresh_in_progress_task_is_left_alone(autonomy_dir):
    task_file(autonomy_dir, "11", status="in_progress", updated=iso(30), timeout=1800)
    assert autonomy.recover_stuck_tasks() == []
    assert "in_progress" in (autonomy_dir / "11-some-task.md").read_text()


def test_in_progress_past_its_timeout_is_reset(autonomy_dir):
    task_file(autonomy_dir, "12", status="in_progress", updated=iso(3700), timeout=3600)
    recovered = autonomy.recover_stuck_tasks()
    assert "12" in [str(t) for t in recovered]
    text = (autonomy_dir / "12-some-task.md").read_text()
    assert "status: up_next" in text
    assert "Recovered from in_progress" in text        # activity note appended


def test_missing_timeout_seconds_falls_back_to_thirty_minutes(autonomy_dir):
    task_file(autonomy_dir, "13", status="in_progress", updated=iso(1900))
    assert "13" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_a_short_timeout_recovers_sooner(autonomy_dir):
    """The reason `timeout_seconds` in frontmatter is not decorative — it drives
    recovery speed as well as the run cap."""
    task_file(autonomy_dir, "14", status="in_progress", updated=iso(700), timeout=600)
    assert "14" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_files_without_the_number_prefix_are_ignored(autonomy_dir):
    (autonomy_dir / "_config.md").write_text(
        "---\nid: cfg\nstatus: in_progress\nupdated: 2020-01-01T00:00:00+00:00\n---\n",
        encoding="utf-8")
    assert autonomy.recover_stuck_tasks() == []


def test_malformed_frontmatter_does_not_abort_the_sweep(autonomy_dir):
    (autonomy_dir / "99-broken.md").write_text("no frontmatter at all\n", encoding="utf-8")
    task_file(autonomy_dir, "15", status="in_progress", updated=iso(99999), timeout=60)
    assert "15" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_last_run_is_used_when_updated_is_absent(autonomy_dir):
    path = autonomy_dir / "16-no-updated.md"
    path.write_text(
        "---\nid: 16\nname: no updated\nstatus: in_progress\n"
        f"last_run: {iso(99999)}\ntimeout_seconds: 600\n---\n\n# Body\n", encoding="utf-8")
    assert "16" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_a_task_with_no_timestamp_at_all_is_treated_as_stuck(autonomy_dir):
    """Characterized: with no `updated` and no `last_run`, `stuck_seconds` is
    forced past the timeout, so an in_progress task with no timestamp is
    recovered on the next tick rather than hanging forever."""
    path = autonomy_dir / "17-no-stamp.md"
    path.write_text("---\nid: 17\nname: no stamp\nstatus: in_progress\n"
                    "timeout_seconds: 1800\n---\n\n# Body\n", encoding="utf-8")
    assert "17" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_missing_autonomy_dir_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "absent")
    assert autonomy.recover_stuck_tasks() == []


def test_recovery_only_touches_the_task_it_flagged(autonomy_dir):
    task_file(autonomy_dir, "18", status="in_progress", updated=iso(99999), timeout=600)
    task_file(autonomy_dir, "19", status="in_progress", updated=iso(10), timeout=600)
    autonomy.recover_stuck_tasks()
    assert "up_next" in (autonomy_dir / "18-some-task.md").read_text()
    assert "in_progress" in (autonomy_dir / "19-some-task.md").read_text()
