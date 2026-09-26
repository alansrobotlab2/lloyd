"""The dashboard's `held` bucket is bounded by the task's OWN period (#1519).

`held` used to be binary on "is there a hold reason at all", which is the
right shape for a nightly job at midday and the wrong shape for one that has
not run in four days. Measured live 2026-09-26 03:00Z, `GET /api/dashboard`
answered `overdue_count: 0` with nine rows in `held`, and inside them sat
#42 Knowledge Analysis — 93.7 h past its own `next_run`, 3.91 declared
periods, on the hold string "outside hours 00-04,22-23". The panel an
operator reads reported zero problems while four of the Distil chain's links
and #51 were 3.6-3.9x past their own cadence, every one of them
`fail_rate: null` and `runs: 0`, so no failure-based alarm could see them
either.

The bound is one of the task's own declared periods, measured by the one
shared predicate `autonomy.next_run_gap` (#1121) so this surface and the
health route's `stalled` list cannot disagree about what "more than a period"
means. Two cases stay in `held`, and both are what the split was written for:
a hold shorter than one period (a daily job sitting at midday inside its
22-04 window's off-hours), and a deliberate stop — `draft` or `paused`, the
statuses the scheduler itself reads as a human kill switch — which stays held
however long it runs, because bounding it would light the alarm permanently
over the parked tasks that are parked on purpose (#68).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.routers import dashboard as dash


@pytest.fixture(autouse=True)
def _clear_cache():
    """`_autonomy` is TTL-cached module-globally; the cache would leak a
    fixture's fleet into the next test's verdict."""
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
    """Write one autonomy task file, runnable unless told otherwise.

    `skill_name` and `frequency: daily` are the defaults because `hold_reason`
    refuses a task with neither — a no-skill task can never dispatch, so
    without them every row here would be held for the wrong reason.
    """
    fm = {"skill_name": "demo", "frequency": "daily", **fm}
    body = "\n".join(f"{k}: {v}" for k, v in fm.items())
    directory = vault / "obsidian" / "autonomy"
    path = directory / f"{len(list(directory.glob('*.md'))) + 1:02d}-{name}.md"
    path.write_text(f"---\nname: {name}\n{body}\n---\n\nbody\n")
    return path


def _outside_nightly_window(monkeypatch, autonomy_module):
    """Freeze the local hour at 14:00, so a task confined to 03:00 reads as
    held by its window — the state a daily job is in for most of every day."""
    monkeypatch.setattr(autonomy_module, "_local_hour", lambda: 14)


# ── Clause 1: a hold past its own period is a miss ─────────────────────


def test_a_hold_two_periods_past_next_run_is_overdue_not_held(vault, monkeypatch):
    """#42's state exactly: `up_next`, inside its window's off-hours, and two
    whole periods past the `next_run` the scheduler writes at completion.

    A row that is past due and has not dispatched has been held the entire
    time it was past due — otherwise it would have run — so `next_run`
    backlogs are a lower bound on how long the hold has persisted, and a
    backlog longer than the task's own period is no longer a schedule working
    as configured. The reason still has to travel with the row: the panel
    prints `blocked`, and an operator who is told "overdue" without the "why"
    cannot tell a window that never opened from a dependency that died.
    """
    import autonomy

    _outside_nightly_window(monkeypatch, autonomy)
    _task(vault, "knowledge-analysis", status="up_next",
          next_run=_iso(hours=-50), preferred_hours="[3]")

    out = dash._autonomy()
    assert [t["name"] for t in out["overdue"]] == ["knowledge-analysis"]
    assert out["overdue_count"] == 1
    assert out["held"] == [] and out["held_count"] == 0
    assert out["overdue"][0]["blocked"] == "outside hours 03", (
        "reclassifying the row must not drop the hold reason the badge prints")


# ── Clause 2: a hold shorter than one period stays held ────────────────


def test_a_sub_period_hold_inside_the_off_hours_is_still_held(vault, monkeypatch):
    """The 3x-at-midday false alarm the split exists to suppress stays
    suppressed: a daily job whose window is 03:00 is past its `next_run` for
    most of every day, which is 9 h of a declared 24 h period, not a miss."""
    import autonomy

    _outside_nightly_window(monkeypatch, autonomy)
    _task(vault, "nightly-miner", status="up_next",
          next_run=_iso(hours=-9), preferred_hours="[3]")

    out = dash._autonomy()
    assert out["overdue"] == [] and out["overdue_count"] == 0
    assert [t["name"] for t in out["held"]] == ["nightly-miner"]
    assert out["held_count"] == 1
    assert out["held"][0]["blocked"] == "outside hours 03"


# ── Clause 3: a deliberate stop is never a miss ────────────────────────


def test_a_deliberate_stop_is_never_moved_into_overdue(vault):
    """Ten periods dark is the expected state of a task a person parked.

    `hold_reason` answers a non-`up_next` task with its status verbatim, so
    the exemption is keyed on the hold reason itself against
    `autonomy.DISPATCH_STOPPING_STATUSES` — the same two values the scheduler
    reads as a dispatch kill switch. Bounding these would re-light the alarm
    over Alan's standing ruling that #68 stays in `draft`, and a permanently
    lit alarm is the failure mode this whole split was written against.
    """
    _task(vault, "parked-draft", status="draft", next_run=_iso(days=-10))
    _task(vault, "parked-paused", status="paused", next_run=_iso(days=-10))

    out = dash._autonomy()
    assert out["overdue"] == [] and out["overdue_count"] == 0
    assert sorted(t["name"] for t in out["held"]) == ["parked-draft", "parked-paused"]
    assert sorted(t["blocked"] for t in out["held"]) == ["draft", "paused"]


# ── The bound is the task's own cadence, not a fixed number of hours ───


def test_the_bound_is_the_tasks_own_period_not_a_fixed_number_of_hours(vault,
                                                                       monkeypatch):
    """Two rows, both two hours past `next_run`, both held by their window.

    One is due every 15 minutes, so two hours is eight missed periods and the
    row is a miss; the other is weekly, so two hours is a twelfth of a period
    and the row is on schedule. An absolute threshold would get one of these
    two wrong whichever way it were set, which is why the health route scores
    `gap_ratio` against the declared period too.
    """
    import autonomy

    _outside_nightly_window(monkeypatch, autonomy)
    _task(vault, "fifteen-minute", status="up_next", frequency="every-15min",
          next_run=_iso(hours=-2), preferred_hours="[3]")
    _task(vault, "weekly", status="up_next", frequency="weekly",
          next_run=_iso(hours=-2), preferred_hours="[3]")

    out = dash._autonomy()
    assert [t["name"] for t in out["overdue"]] == ["fifteen-minute"]
    assert [t["name"] for t in out["held"]] == ["weekly"]
    assert out["overdue_count"] == 1 and out["held_count"] == 1


# ── Seam: the bound is the health route's predicate, not a second copy ──


def test_the_dashboard_agrees_with_the_health_routes_past_next_run_predicate(vault,
                                                                            monkeypatch):
    """`/api/autonomy/health` already computes `stalled` and `gap_ratio` and
    nothing consumes them (#1519). The fix must not answer the same question
    twice: this surface has to call the same predicate #1121 unified, on the
    same front matter, or the two surfaces disagree one period either side of
    the bound — which is exactly how #68 came to score `fail_rate 0.0` beside
    an alert naming it as the fleet's worst stall.
    """
    import autonomy

    _outside_nightly_window(monkeypatch, autonomy)
    long_hold = _task(vault, "past-a-period", status="up_next",
                      next_run=_iso(hours=-50), preferred_hours="[3]")
    short_hold = _task(vault, "inside-a-period", status="up_next",
                       next_run=_iso(hours=-9), preferred_hours="[3]")

    out = dash._autonomy()
    overdue_names = [t["name"] for t in out["overdue"]]
    held_names = [t["name"] for t in out["held"]]

    fm_long = dash._frontmatter(long_hold)
    fm_short = dash._frontmatter(short_hold)
    assert autonomy.next_run_gap(fm_long, now=datetime.now(timezone.utc))["past_next_run"] is True
    assert autonomy.next_run_gap(fm_short, now=datetime.now(timezone.utc))["past_next_run"] is False

    assert "past-a-period" in overdue_names and "past-a-period" not in held_names, (
        "`past_next_run` says a period has gone by, so this row must be a miss")
    assert "inside-a-period" in held_names and "inside-a-period" not in overdue_names, (
        "`past_next_run` says the period has not gone by, so this row must not be")
