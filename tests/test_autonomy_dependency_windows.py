"""#1437 Defect B — dependency freshness and `next_run` follow the windows.

On 2026-09-23 #38 (reflection signals) and #56 (trajectory extraction) ran at
14:25-14:31Z, outside their `preferred_hours`. Two rules then pinned both
nightly chains for two cycles:

* the elapsed gate re-anchored each upstream at `last_run + interval - slack`
  (13:31Z the next day), after its next window had already closed, so it lost
  that window too; and
* the dependency gate called an upstream fresh only within `interval / 2`
  (12 h) of its run, which no instant inside #42's (05-12Z) or #57's window
  could satisfy, while `stale_bypass_hours: 36` opened only after the window.

The item proves no `stale_bypass_hours` serves both the healthy and the slipped
chain whenever a dependent's window opens before its upstream's. The ruling:
derive both from the windows. Every instant here is pinned, the machine zone is
pinned to PDT (UTC-7, the zone the incident's hours are in), and the local hour
the window gate reads is derived from the same instant.
"""
import datetime as dt
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autonomy  # noqa: E402

PDT = dt.timezone(dt.timedelta(hours=-7))
NIGHT = [22, 23, 0, 1, 2, 3, 4]          # #38, #42: 05:00-11:59Z
LATE = [23, 0, 1, 2, 3, 4]               # #51, #57: 06:00-11:59Z


def Z(day, hh, mm=0, ss=0):
    return dt.datetime(2026, 9, day, hh, mm, ss, tzinfo=dt.timezone.utc)


@pytest.fixture
def aut(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "autonomy")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(autonomy, "_missing_artifact_bypass_warned", {})
    monkeypatch.setattr(autonomy, "_never_ran_bypass_warned", {})
    monkeypatch.setattr(autonomy, "_local_tz", lambda: PDT)
    autonomy.AUTONOMY_DIR.mkdir()
    return autonomy


def at(aut, monkeypatch, now):
    """Pin the instant AND the local hour the window gate reads, together."""
    monkeypatch.setattr(aut, "_utcnow", lambda: now)
    monkeypatch.setattr(aut, "_local_hour", lambda: now.astimezone(PDT).hour)
    return now


def task(tid, *, hours=None, dep=None, last=None, **extra):
    t = {"id": tid, "name": f"task{tid}", "status": "up_next",
         "frequency": "daily", "skill_name": "some-skill",
         "preferred_hours": hours, "depends_on": dep,
         "last_run": last.isoformat() if last else None}
    t.update(extra)
    return t


def by_id(board):
    return {t["id"]: t for t in board}


def complete(t, when):
    t["last_run"] = when.isoformat()


def due(aut, t, board, now):
    return aut._is_task_due(t, board, now=now)


def reason(aut, t, board, now):
    return aut.hold_reason(t, board, now=now)


def _record(aut, task_id, run_id, completed_at):
    d = aut.AUTONOMY_RUNS_DIR / str(task_id)
    d.mkdir(parents=True, exist_ok=True)
    fm = {"run_id": run_id, "task_id": task_id, "status": "success",
          "completed_at": completed_at.isoformat()}
    (d / f"{run_id}.md").write_text(f"---\n{yaml.dump(fm)}---\n\nreport\n")


# ── Incident 1: #38 -> #42 -> #39 -> #40 ────────────────────────────────────

def _reflection_board():
    return [
        task(38, hours=NIGHT, last=Z(23, 14, 31, 42), stale_bypass_hours=36),
        task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16, 36), stale_bypass_hours=36),
        task(39, hours=[1, 2, 3, 4], dep=42, last=Z(22, 8, 6), stale_bypass_hours=36),
        task(40, hours=[2, 3, 4], dep=39, last=Z(22, 9, 5), stale_bypass_hours=36),
    ]


def test_the_slipped_upstream_is_due_at_its_next_window_opening(aut, monkeypatch):
    """#38 done 14:31:42Z on 09-23 comes due at 22:00 local that evening
    (05:00Z 09-24) — not at 13:31Z 09-24, after that window had closed."""
    board = _reflection_board()
    t38 = by_id(board)[38]
    # The run record of the out-of-window run is on disk, as it was: the guard
    # answered "already ran this period" for #38 at the incident's probe.
    _record(aut, 38, "run_38_20260923_142539", Z(23, 14, 31, 42))

    assert aut._elapsed_due_at(t38, Z(23, 14, 31, 42)) == Z(24, 4)
    assert aut._next_run_after(t38, Z(23, 14, 31, 42)) == Z(24, 5)

    now = at(aut, monkeypatch, Z(24, 4, 59))
    assert reason(aut, t38, board, now) == "outside hours 00-04,22-23"
    now = at(aut, monkeypatch, Z(24, 5, 0))
    assert due(aut, t38, board, now) is True
    assert reason(aut, t38, board, now) is None
    # The probe instant of the item (08:02Z): the window is still open and #38
    # is still due, not "already ran this period".
    now = at(aut, monkeypatch, Z(24, 8, 2))
    assert aut._already_ran_this_period(t38, now=now) == ""
    assert due(aut, t38, board, now) is True


@pytest.mark.parametrize("bypass", [True, False])
def test_the_reflection_chain_runs_in_its_next_window_after_the_slip(
        aut, monkeypatch, bypass):
    """Also with every `stale_bypass_hours` removed: the ordinary chain must not
    need the override (it is what forwarded #39 onto nothing on 09-24)."""
    board = _reflection_board()
    if not bypass:
        for t in board:
            t.pop("stale_bypass_hours")
    b = by_id(board)

    # 05:00Z: #38 owes this window a run, so #42 waits for it (the old rule
    # waited too — for 14.5 h of a 12 h bound, then for a 36 h bypass that
    # opened after the window closed).
    now = at(aut, monkeypatch, Z(24, 5, 0))
    assert reason(aut, b[42], board, now) == "waiting on #38"
    complete(b[38], Z(24, 5, 20))

    now = at(aut, monkeypatch, Z(24, 5, 21))
    assert due(aut, b[42], board, now) is True, "#42 follows #38 in the SAME window"
    if not bypass:
        assert reason(aut, b[39], board, now) == "waiting on #42"
    complete(b[42], Z(24, 6, 10))

    now = at(aut, monkeypatch, Z(24, 8, 0))          # #39 opens 01:00 local
    assert due(aut, b[39], board, now) is True
    if not bypass:
        assert reason(aut, b[40], board, now) == "waiting on #39"
    complete(b[39], Z(24, 8, 40))

    now = at(aut, monkeypatch, Z(24, 9, 0))          # #40 opens 02:00 local
    assert due(aut, b[40], board, now) is True


def test_the_old_interval_half_bound_is_what_held_42(aut, monkeypatch):
    """Pin the incident's arithmetic: inside #42's window #38's 14:31Z output was
    14.5-20.5 h old — never within `interval / 2`, never past 36 h. The windows
    call it not-this-cycle for a different reason: #38 is owed at 05:00Z."""
    board = _reflection_board()
    b = by_id(board)
    for now in (Z(24, 5, 0), Z(24, 11, 59)):
        age_h = (now - Z(23, 14, 31, 42)).total_seconds() / 3600
        assert 12 < age_h < 36
        assert aut._window_dependency_fresh(b[42], b[38], Z(23, 14, 31, 42),
                                            now) is False


# ── Incident 2: #56 -> #51 / #57 -> #58 -> #83 ──────────────────────────────

def _trajectory_board(up_last=Z(23, 14, 31, 29)):
    return [
        task(56, hours=[1, 2], last=up_last),
        task(51, hours=LATE, dep=56, last=Z(22, 8, 20, 7), runs_per_day=1),
        task(57, hours=LATE, dep=56, last=Z(22, 8, 18, 53)),
        task(58, hours=[], dep=57, last=Z(22, 11, 7, 23)),
        task(83, dep=58, last=Z(22, 13, 11, 14)),
    ]


def test_the_trajectory_chain_runs_in_its_next_window_after_the_slip(
        aut, monkeypatch):
    board = _trajectory_board()
    b = by_id(board)
    _record(aut, 56, "run_56_20260923_143102", Z(23, 14, 31, 29))

    # 06:00Z 09-24: #51/#57 are open, #56 opens at 08:00Z. The dependents wait
    # for today's extraction rather than consuming the out-of-window one.
    now = at(aut, monkeypatch, Z(24, 6, 0))
    for tid in (51, 57):
        assert reason(aut, b[tid], board, now) == "waiting on #56"
    assert reason(aut, b[56], board, now) == "outside hours 01-02"
    assert reason(aut, b[58], board, now) == "waiting on #57"
    assert reason(aut, b[83], board, now) == "waiting on #58"

    now = at(aut, monkeypatch, Z(24, 8, 0))
    assert due(aut, b[56], board, now) is True, (
        "the slip no longer costs #56 its 09-24 window")
    complete(b[56], Z(24, 8, 5))

    now = at(aut, monkeypatch, Z(24, 8, 6))
    assert due(aut, b[51], board, now) is True
    assert due(aut, b[57], board, now) is True
    complete(b[57], Z(24, 8, 40))

    # #58 and #83 have no window: their edges keep the `interval / 2` rule.
    now = at(aut, monkeypatch, Z(24, 8, 45))
    assert due(aut, b[58], board, now) is True
    complete(b[58], Z(24, 9, 30))
    now = at(aut, monkeypatch, Z(24, 9, 31))
    assert due(aut, b[83], board, now) is True


def test_the_healthy_chain_still_holds_the_dependent_until_todays_upstream(
        aut, monkeypatch):
    """Finding 2's HEALTHY case: #56 ran in window at 08:05Z 09-22. At #57's
    window opening on 09-23 that output is 21.9 h old and today's #56 is due at
    08:00Z, so #57 must hold — the inversion the gate exists to prevent."""
    board = _trajectory_board(up_last=Z(22, 8, 5))
    b = by_id(board)
    b[57]["last_run"] = Z(22, 8, 18).isoformat()
    for probe in (Z(23, 6, 0), Z(23, 7, 59), Z(23, 8, 30)):
        now = at(aut, monkeypatch, probe)
        assert reason(aut, b[57], board, now) == "waiting on #56", probe
    complete(b[56], Z(23, 8, 35))
    now = at(aut, monkeypatch, Z(23, 8, 36))
    assert due(aut, b[57], board, now) is True


def test_the_same_window_healthy_chain_holds_until_the_upstream_ran(
        aut, monkeypatch):
    board = [task(38, hours=NIGHT, last=Z(22, 5, 5)),
             task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16), stale_bypass_hours=36)]
    b = by_id(board)
    now = at(aut, monkeypatch, Z(23, 5, 0))
    assert due(aut, b[38], board, now) is True
    assert reason(aut, b[42], board, now) == "waiting on #38"
    complete(b[38], Z(23, 5, 4))
    now = at(aut, monkeypatch, Z(23, 5, 5))
    assert due(aut, b[42], board, now) is True
    # Consumed: once #42 has run on this #38 output it is not fresh again.
    complete(b[42], Z(23, 5, 30))
    now = at(aut, monkeypatch, Z(23, 6, 0))
    assert reason(aut, b[42], board, now) == "waiting on #38"


def test_an_upstream_that_never_gets_its_window_keeps_the_dependent_waiting(
        aut, monkeypatch):
    """Owed and not delivered: the dependent waits the whole window. Only an
    explicitly declared `stale_bypass_hours` forwards on the old output."""
    board = _trajectory_board()
    b = by_id(board)
    now = at(aut, monkeypatch, Z(24, 11, 30))
    assert reason(aut, b[57], board, now) == "waiting on #56"
    b[57]["stale_bypass_hours"] = 16
    assert due(aut, b[57], board, now) is True, "the override still works"


def test_a_run_just_before_the_window_is_not_repeated_and_feeds_the_chain(
        aut, monkeypatch):
    """A manual #56 at 07:00Z, an hour before its window: the floor of half a
    period keeps it from running again at 08:00Z, and because it owes nothing
    more this cycle its dependents consume that run."""
    board = _trajectory_board(up_last=Z(24, 7, 0))
    b = by_id(board)
    now = at(aut, monkeypatch, Z(24, 8, 0))
    assert due(aut, b[56], board, now) is False
    assert aut._next_run_after(b[56], Z(24, 7, 0)) == Z(25, 8)
    assert due(aut, b[57], board, now) is True


# ── next_run re-anchoring ───────────────────────────────────────────────────

def test_next_run_after_an_out_of_window_completion_lands_in_window(aut):
    t = task(38, hours=NIGHT)
    # The incident's 14:31Z lands on the opening. Later slips land where the
    # half-period floor puts them, still inside a window: 18:00Z is 11 h before
    # the opening, so 06:00Z; 03:00Z is two hours before it, so the next night.
    for done, expect in ((Z(23, 14, 31, 42), Z(24, 5)), (Z(23, 18, 0), Z(24, 6)),
                         (Z(24, 3, 0), Z(25, 5))):
        nxt = aut._next_run_after(t, done)
        assert nxt.astimezone(PDT).hour in NIGHT, (done, nxt)
        assert nxt == expect, (done, nxt)
    # In window, the schedule is unchanged in effect: the next opening.
    assert aut._next_run_after(t, Z(24, 5, 10)) == Z(25, 5)
    # Weekly (#79, [22..2]): out-of-window completion 14:56Z 09-23 -> 22:00
    # local on 09-29, not 13:56Z 09-30 outside the window.
    w = task(79, hours=[22, 23, 0, 1, 2], frequency="weekly")
    assert aut._next_run_after(w, Z(23, 14, 56, 39)) == Z(30, 5)


def test_the_run_record_guard_follows_the_same_due_instant(aut, monkeypatch):
    t = task(38, hours=NIGHT, last=Z(24, 5, 10))
    _record(aut, 38, "run_38_a", Z(24, 5, 10))
    assert aut._already_ran_this_period(t, now=Z(24, 6, 0)) == "run_38_a"
    assert aut._already_ran_this_period(t, now=Z(25, 3, 59)) == "run_38_a"
    assert aut._already_ran_this_period(t, now=Z(25, 4, 0)) == ""
    now = at(aut, monkeypatch, Z(25, 5, 0))
    assert due(aut, t, [t], now) is True


async def test_every_completion_path_writes_the_windowed_next_run(
        aut, tmp_path, monkeypatch):
    """`run_task` is the one completion writer the scheduler, the API and the
    MCP tool share (see `test_every_production_caller_names_itself`)."""
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\nDo it.\n")
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "sys",
                        raising=False)

    async def _rq(messages, options):
        yield {"type": "text_delta", "text": "did the thing"}
        yield {"type": "result", "stop_reason": "stop",
               "usage": {"input_tokens": 1, "output_tokens": 1}, "num_turns": 1}
    monkeypatch.setattr("app.harness.run_query", _rq)
    fm = {"id": 38, "name": "t38", "type": "autonomy", "status": "up_next",
          "frequency": "daily", "skill_name": str(skill), "timeout_seconds": 30,
          "max_retries": 3, "failure_count": 0, "preferred_hours": NIGHT}
    (aut.AUTONOMY_DIR / "38-t38.md").write_text(
        f"---\n{yaml.dump(fm)}---\n\nbody\n\n## Activity Log\n")
    with aut.run_trigger("mcp"):
        result = await aut.run_task(38)
    assert result["success"] is True
    t = aut._parse_task_file(aut._find_task_file(38))
    last = aut._parse_iso(t["last_run"])
    nxt = aut._parse_iso(t["next_run"])
    assert nxt == aut._next_run_after(t, last)
    assert nxt.astimezone(PDT).hour in NIGHT


# ── Tasks without windows keep today's behaviour exactly ────────────────────

def test_windowless_tasks_are_unchanged(aut, monkeypatch):
    t = task(1, last=Z(23, 14, 31))
    assert aut._elapsed_due_at(t, Z(23, 14, 31)) == Z(24, 14, 31)
    assert aut._next_run_after(t, Z(23, 14, 31)) == Z(24, 14, 31)
    up = task(2, last=Z(23, 20, 0))
    down = task(3, dep=2, last=Z(22, 20, 0))
    board = [up, down]
    assert aut._window_dependency_fresh(down, up, Z(23, 20, 0), Z(24, 7)) is None
    now = at(aut, monkeypatch, Z(24, 7, 59))            # 11 h 59 m: fresh
    assert aut._is_dependency_met(down, board, now=now) is True
    now = at(aut, monkeypatch, Z(24, 8, 1))             # past interval / 2
    assert aut._is_dependency_met(down, board, now=now) is False


def test_one_windowless_side_keeps_the_interval_half_rule(aut):
    windowed_dep = task(47, hours=NIGHT, dep=40)
    plain_up = task(40)
    assert aut._window_dependency_fresh(windowed_dep, plain_up, Z(23, 9), Z(24, 5)) is None
    plain_dep = task(58, dep=57)
    windowed_up = task(57, hours=LATE)
    assert aut._window_dependency_fresh(plain_dep, windowed_up, Z(23, 9), Z(24, 5)) is None


def test_sub_daily_windowed_tasks_keep_the_elapsed_rule(aut):
    t = task(5, hours=[1, 2, 3, 4], runs_per_day=4)       # 6 h period
    slack = aut._due_slack_seconds(t)
    assert aut._elapsed_due_at(t, Z(24, 9)) == Z(24, 15) - dt.timedelta(seconds=slack)
    assert aut._next_run_after(t, Z(24, 9)) == Z(24, 15)


# ── The override does not race an upstream that is about to run ─────────────

def test_a_bypass_waits_for_an_upstream_due_in_its_window_this_tick(
        aut, monkeypatch, tmp_path):
    """The live board on 2026-09-25 05:00Z: #38 last ran 38.5 h ago, past #42's
    36 h `stale_bypass_hours`, and its declared artifact is on disk — so the
    override alone would dispatch #42 beside #38, on 09-23's signals. #38 is
    due in its window on this tick, so #42 waits for it instead."""
    art = tmp_path / "signals-latest.md"
    art.write_text("x" * 4096)
    board = [task(38, hours=NIGHT, last=Z(23, 14, 31, 42), output_artifact=str(art)),
             task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16, 36),
                  stale_bypass_hours=36)]
    b = by_id(board)
    now = at(aut, monkeypatch, Z(25, 5, 0))
    assert due(aut, b[38], board, now) is True
    assert reason(aut, b[42], board, now) == "waiting on #38"
    b[38]["status"] = "in_progress"
    assert reason(aut, b[42], board, now) == "waiting on #38"
    b[38]["status"] = "up_next"
    complete(b[38], Z(25, 5, 20))
    now = at(aut, monkeypatch, Z(25, 5, 21))
    assert due(aut, b[42], board, now) is True


def test_the_override_still_forwards_when_the_upstream_will_not_run(
        aut, monkeypatch):
    board = [task(38, hours=NIGHT, last=Z(23, 14, 31, 42), status="failed"),
             task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16, 36),
                  stale_bypass_hours=36)]
    b = by_id(board)
    now = at(aut, monkeypatch, Z(25, 5, 0))
    assert due(aut, b[42], board, now) is True


def test_a_dependency_cycle_cannot_recurse(aut, monkeypatch):
    board = [task(1, hours=NIGHT, dep=2, last=Z(20, 5), stale_bypass_hours=36),
             task(2, hours=NIGHT, dep=1, last=Z(20, 5), stale_bypass_hours=36)]
    now = at(aut, monkeypatch, Z(25, 5, 0))
    for t in board:
        assert aut.hold_reason(t, board, now=now) in (None, f"waiting on #{t['depends_on']}")


# ── Where `interval / 2` could never answer ──────────────────────────────────

@pytest.mark.parametrize("up_hours,down_hours,up_done,probe", [
    # Dependent opens 14 h after its upstream's run, same cycle: 22:00 -> 12:00.
    ([22], [12], Z(24, 5, 5), Z(24, 19, 0)),
    # Dependent consumes the previous afternoon's output before its upstream's
    # next window: upstream 06:00 local (13:00Z), dependent 23:00-04:00 local.
    ([6], LATE, Z(23, 13, 5), Z(24, 6, 0)),
])
def test_a_cycle_whose_hand_off_is_older_than_half_a_day_still_runs(
        aut, monkeypatch, up_hours, down_hours, up_done, probe):
    """The old bound held both shapes for good (13.9 h and 16.9 h > 12 h) unless
    a bypass was declared; the windows see the upstream owes nothing before the
    dependent's window closes, so the dependent runs on the cycle's output."""
    board = [task(1, hours=up_hours, last=up_done),
             task(2, hours=down_hours, dep=1, last=up_done - dt.timedelta(days=1))]
    b = by_id(board)
    assert (probe - up_done).total_seconds() > 12 * 3600
    now = at(aut, monkeypatch, probe)
    assert due(aut, b[2], board, now) is True
    complete(b[2], probe + dt.timedelta(minutes=10))
    later = at(aut, monkeypatch, probe + dt.timedelta(minutes=20))
    assert reason(aut, b[2], board, later) == "waiting on #1", "consumed once"


# ── #1526: #51 / #57 fail forward when #56 is starved past their close ──────

# Captured at import: the `aut` fixture re-points `autonomy.AUTONOMY_DIR` at a tmp
# board, and clause 1 is a claim about the files the scheduler actually reads.
LIVE_BOARD_DIR = autonomy.AUTONOMY_DIR


def _live_bypass_bound(tid):
    """`stale_bypass_hours` as the live board declares it, for task `tid`."""
    path = next(iter(LIVE_BOARD_DIR.glob(f"{tid}-*.md")), None)
    assert path is not None, f"no task file for #{tid} under {LIVE_BOARD_DIR}"
    return (autonomy._parse_task_file(path) or {}).get("stale_bypass_hours")


# The night the pool starved #56: queue row 515 `enqueued_at 2026-09-25T08:00:33Z`
# (inside #56's own 08:00-09:59Z window) and `claimed_at 2026-09-25T19:10:40Z` —
# 11 h 10 m of starvation, so the run finished at 19:14:12Z, 7 h 14 m after
# #51/#57's window had closed at 12:00Z. Its record
# `autonomy-runs/56/run_56_20260925_191040.md` carries `in_window: false`.
LATE_AT = Z(25, 19, 14, 12)


def _starved_board(bound):
    """#51 and #57 as the live files declare them, with #56 as it ran."""
    return [
        task(56, hours=[1, 2], last=LATE_AT),
        task(51, hours=LATE, dep=56, last=Z(22, 8, 20, 7),
             **({"stale_bypass_hours": bound} if bound else {})),
        task(57, hours=LATE, dep=56, last=Z(22, 8, 18, 53),
             **({"stale_bypass_hours": bound} if bound else {})),
    ]


def test_the_starved_leg_declares_its_fail_forward_and_56_does_not():
    """#1526 clause 1. #56 has no `depends_on`, so a bound on it is read by
    nothing; #51 and #57 are the tasks the pool has been holding since
    2026-09-22, and each must declare one short enough to fire on this edge.

    The ceiling is not the clause's 30 h but the arithmetic of the night that
    broke: a completion that lands 7 h 14 m past the dependents' close is
    16 h 45 m old at that close, so a bound above it cannot open that window at
    all — 24 h (the reflection chain's value, where the upstream's slip is a
    night) holds the chain for a whole further day. 12 h is shipped; over 17 h
    fails this clause and re-creates the outage.
    """
    assert _live_bypass_bound(56) is None, "#56 has no depends_on: a bound is noise"
    for tid in (51, 57):
        raw = _live_bypass_bound(tid)
        assert raw is not None, f"#{tid} declares no stale_bypass_hours"
        assert 0 < float(raw) <= 17, (
            f"#{tid} declares {raw} h: a starved completion is only 16.75 h old at "
            "the window's close, so that bound never opens this window")


def test_the_starved_chain_releases_at_its_window_opening(aut, monkeypatch):
    """#1526 clause 2 — the three instants the triage replayed, at the bound read
    off the live board.

    #56's last completion was 2026-09-23T14:31:29Z, so at 23:00 local on 09-24
    (06:00Z on the 25th, when #51/#57's window opens) it is 39.5 h old — past the
    declared bound at the opening, not merely near it. Before #1526 neither task
    declared a bound, and `test_an_upstream_that_never_gets_its_window_keeps_the_
    dependent_waiting` above pins what that means: `waiting on #56` at every hour
    of every window, four windows on the row with no run, #58 and #83 down behind
    them.
    """
    bound = float(_live_bypass_bound(51))
    board = _starved_board(bound)
    board[0]["last_run"] = Z(23, 14, 31, 29).isoformat()
    b = by_id(board)
    assert (Z(25, 6, 0) - Z(23, 14, 31, 29)).total_seconds() / 3600 > bound

    for label, probe in (("23:00 local", Z(25, 6, 0)),      # window opens 06:00Z
                         ("03:00 local", Z(25, 10, 0)),
                         ("04:30 local", Z(25, 11, 30))):   # closes 12:00Z
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) is None, (
                f"#{tid} still held at {label}: the declared bound does not open "
                "inside the window, so the starved leg stays down")
            assert due(aut, b[tid], board, now) is True, (label, tid)


def test_the_release_still_waits_while_56_is_due_inside_its_own_window(
        aut, monkeypatch):
    """#1526 clause 3 — the valve at `autonomy.py:1622-1629` survives the fix.

    At 01:00 and 02:00 local (08:00Z, 09:00Z) #56 sits inside its own
    `preferred_hours` and is elapsed-due, so dispatching #51/#57 now would run
    them beside the upstream on the previous cycle's output — the inversion the
    valve exists to stop. The bound is past at both instants (41.5 h and 42.5 h),
    so they hold only because the valve suppresses it. These are the same probes
    the two tests above release at 23:00 / 03:00 / 04:30: a fix that stopped
    holding here has traded a dead leg for a wrong-data race.
    """
    bound = float(_live_bypass_bound(51))
    board = _starved_board(bound)
    board[0]["last_run"] = Z(23, 14, 31, 29).isoformat()
    b = by_id(board)
    for label, probe in (("01:00 local", Z(25, 8, 0)), ("02:00 local", Z(25, 9, 0))):
        assert (probe - Z(23, 14, 31, 29)).total_seconds() / 3600 > bound, label
        now = at(aut, monkeypatch, probe)
        assert due(aut, b[56], board, now) is True, f"#56 must be due at {label}"
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) == "waiting on #56", (label, tid)


def test_a_completion_starved_past_the_close_releases_that_same_window(
        aut, monkeypatch):
    """#1526's acceptance: the pool-latency night, not merely the wedged one.

    This is the night that actually broke: #56 completed at 19:14:12Z on 09-25,
    7 h 14 m after the dependents' close, so their next window (06:00Z-11:59Z on
    09-26) starts with an upstream that is 10 h 46 m old — fresh by any daily
    measure, and not owed again until 09-27. The elapsed rule therefore calls it
    this cycle's output and holds the window; only the declared bound forwards
    it, and only once the upstream is a day-and-a-half of nothing. With the
    shipped 12 h that is 03:00 and 04:30 local, 3-4.5 h before the close: the
    leg runs the very night it was starved. Strip the bound and all three
    instants hold, which is the outage this item is about.
    """
    bound = float(_live_bypass_bound(51))
    board = _starved_board(bound)
    b = by_id(board)
    held_at = ("23:00 local", "03:00 local", "04:30 local")
    probes = (Z(26, 6, 0), Z(26, 10, 0), Z(26, 11, 30))
    assert (probes[0] - LATE_AT).total_seconds() / 3600 < bound < (probes[1] - LATE_AT).total_seconds() / 3600

    for label, probe in zip(held_at, probes):
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            got = reason(aut, b[tid], board, now)
            if label == "23:00 local":
                assert got == "waiting on #56", (
                    f"#{tid} released at {label} on an upstream only 10.8 h old")
            else:
                assert got is None, f"#{tid} still held at {label}: {bound} h does not open this window"
            assert due(aut, b[tid], board, now) is (got is None), (label, tid)

    stripped = _starved_board(None)
    bs = by_id(stripped)
    for label, probe in zip(held_at, probes):
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert reason(aut, bs[tid], stripped, now) == "waiting on #56", (
                f"#{tid} released at {label} with no bound declared — this test "
                "would pass whatever the board says")
