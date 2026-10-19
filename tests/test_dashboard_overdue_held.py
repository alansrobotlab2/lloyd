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
    from app import autonomy

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
    from app import autonomy

    _outside_nightly_window(monkeypatch, autonomy)
    _task(vault, "nightly-miner", status="up_next",
          next_run=_iso(hours=-9), preferred_hours="[3]")

    out = dash._autonomy()
    assert out["overdue"] == [] and out["overdue_count"] == 0
    assert [t["name"] for t in out["held"]] == ["nightly-miner"]
    assert out["held_count"] == 1
    assert out["held"][0]["blocked"] == "outside hours 03"


# ── Clause 1: a DECLARED park is never a miss, however long it runs ────


def test_a_declared_park_is_never_moved_into_overdue(vault):
    """Ten periods dark is the expected state of a task a person parked ON RECORD.

    Reworded by #3016, which also narrowed what this pins. The version before it
    seeded `parked-draft` and `parked-paused` with no `parked:` key and called them
    deliberate, which is the conflation the item exists to end: `hold_reason` answers a
    non-`up_next` task with its status verbatim, so a bare `status: draft` records that
    a job is not armed, not that anyone decided that — and a `draft` create is the
    deliberately-ungated shape every chain-filed hand-down arrives in
    (`app/harness/policy.py`, "the nightly chain's documented hand-down shape"). What
    makes a stop a decision is the file's own `parked:` line, the same one
    `workers/fleet_watchdog.py:653` suppresses its stall alert on. The rest of the
    reasoning is unchanged: bounding a DECLARED park would re-light the alarm over
    Alan's standing ruling that #68 stays in `draft`, and a permanently lit alarm is
    the failure mode this whole split was written against.
    """
    _task(vault, "parked-draft", status="draft", next_run=_iso(days=-10),
          parked="Alan parked this on purpose 2026-09-17; the row stays draft.")
    _task(vault, "parked-paused", status="paused", next_run=_iso(days=-10),
          parked="paused by hand pending the GPU move.")

    out = dash._autonomy()
    assert out["overdue"] == [] and out["overdue_count"] == 0
    assert sorted(t["name"] for t in out["held"]) == ["parked-draft", "parked-paused"]
    assert sorted(t["blocked"] for t in out["held"]) == ["draft", "paused"]


# ── Clause 2: an undeclared stop past its own period is a miss ─────────


def test_an_undeclared_stop_past_its_own_period_is_a_miss_that_keeps_its_reason(vault):
    """A draft with no `parked:` line is a job nobody armed, and now reads that way.

    Three of these are live on the box today: #100, #101 and #102 were `status: draft`,
    have zero run records under ~/lloyd-data/autonomy-runs/, and were past their own
    declared period — while `GET /api/dashboard` answered `overdue_count: 0` and
    `held_count: 6`, labelling every one of them a deliberate park. The row keeps its
    `blocked` value when it is reclassified, because "why is this not dispatching" and
    "is this a miss" are different questions and the panel still needs the first.
    """
    _task(vault, "never-armed-daily", status="draft", frequency="daily",
          next_run=_iso(days=-2))
    _task(vault, "never-armed-weekly", status="draft", frequency="weekly",
          next_run=_iso(days=-8))
    _task(vault, "never-armed-paused", status="paused", frequency="daily",
          next_run=_iso(days=-2))

    out = dash._autonomy()
    assert sorted(t["name"] for t in out["overdue"]) == [
        "never-armed-daily", "never-armed-paused", "never-armed-weekly"]
    assert out["overdue_count"] == 3
    assert [t["name"] for t in out["held"]] == []
    # The hold reason survives the move; dropping it would trade one blind spot for
    # another, since `draft` is precisely what a reader must see to know the fix is to
    # arm the job rather than debug the scheduler.
    assert sorted(t["blocked"] for t in out["overdue"]) == ["draft", "draft", "paused"]


# ── Clause 3: `parked:` has to say something to count ──────────────────


def test_parked_false_declares_nothing(vault):
    """`parked: false` is a value, not a decision — same rule, same accessor.

    `autonomy.parked_declaration` hands back '' for any non-string that is falsy, so a
    YAML `false` reaches the caller falsy and the exemption does not fire, exactly as
    the watchdog's `t["parked"]` truthiness test treats it. A key present with no value
    declares nothing either: the point of the key is that a later reader can quote it
    back as the ruling, and neither of these can be quoted.
    """
    _task(vault, "parked-false", status="draft", frequency="daily",
          next_run=_iso(days=-2), parked=False)
    _task(vault, "parked-empty", status="draft", frequency="daily",
          next_run=_iso(days=-2), parked="")

    out = dash._autonomy()
    assert sorted(t["name"] for t in out["overdue"]) == ["parked-empty", "parked-false"]
    assert out["overdue_count"] == 2
    assert [t["name"] for t in out["held"]] == []


# ── Clause 4: gating must not turn every short hold into a miss ────────


def test_an_undeclared_stop_inside_its_own_period_is_still_held(vault):
    """Under one declared period dark, an undeclared draft is a hold and not a miss.

    The narrowing is bounded by the task's own cadence, not switched off: a daily 12
    hours past `next_run` and a weekly 2 days past it are both inside the period they
    declared, so they stay in `held` with `overdue_count` at 0. The third row is the
    control that makes the first two evidence of a bound rather than of a blanket rule —
    the same status and cadence, 30 hours old, and it is a miss.
    """
    _task(vault, "draft-daily-young", status="draft", frequency="daily",
          next_run=_iso(hours=-12))
    _task(vault, "draft-weekly-young", status="draft", frequency="weekly",
          next_run=_iso(days=-2))
    _task(vault, "draft-daily-old", status="draft", frequency="daily",
          next_run=_iso(hours=-30))

    out = dash._autonomy()
    assert sorted(t["name"] for t in out["held"]) == [
        "draft-daily-young", "draft-weekly-young"]
    assert out["held_count"] == 2
    assert [t["name"] for t in out["overdue"]] == ["draft-daily-old"]
    assert out["overdue_count"] == 1


# ── The two surfaces must agree about which gaps are decisions ─────────


def test_the_dashboard_and_the_watchdog_read_the_same_declaration(vault):
    """One record of a decision, read through one accessor, by both surfaces.

    The premise of the fix is that `parked:` is that record, so the panel and the
    watchdog must not disagree about the same row. This drives the dashboard's split and
    the predicate the watchdog's rule encodes over ONE vault, across the four shapes a
    fleet actually contains: declared/undeclared × past-period/inside-period. The
    right-hand side is the watchdog's own expression (`not parked and past_one_period`),
    so the node fails if either side is rewired to a second copy of the rule.
    """
    _task(vault, "declared-old", status="draft", frequency="daily",
          next_run=_iso(days=-5), parked="ruled spent 2026-10-17, do not arm")
    _task(vault, "undeclared-old", status="draft", frequency="daily",
          next_run=_iso(days=-5))
    _task(vault, "undeclared-young", status="draft", frequency="daily",
          next_run=_iso(hours=-6))

    out = dash._autonomy()
    overdue_names = {t["name"] for t in out["overdue"]}
    buckets = {t["name"]: ("overdue" if t["name"] in overdue_names else "held")
               for t in out["held"] + out["overdue"]}
    assert buckets == {"declared-old": "held",
                       "undeclared-old": "overdue",
                       "undeclared-young": "held"}, buckets

    from app import autonomy
    now = datetime.now(timezone.utc)
    paths = sorted((vault / "obsidian" / "autonomy").glob("*.md"))
    files = [dash._frontmatter(p) for p in paths]
    resolution = autonomy.dependency_resolution_set(vault / "obsidian" / "autonomy")
    for fm in files:
        fm["status"] = str(fm.get("status") or "draft")
        fm["blocked"] = autonomy.hold_reason(fm, resolution)
        gap = autonomy.next_run_gap(fm)
        assert dash._hold_is_a_miss({"blocked": fm["blocked"]}, fm, now) == (
            not bool(autonomy.parked_declaration(fm)) and gap["past_next_run"]
        ), fm["name"]


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
    from app import autonomy

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
    from app import autonomy

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
