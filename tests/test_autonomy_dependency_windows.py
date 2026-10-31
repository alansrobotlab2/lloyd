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
from app import autonomy  # noqa: E402

PDT = dt.timezone(dt.timedelta(hours=-7))
NIGHT = [22, 23, 0, 1, 2, 3, 4]          # #38, #42: 05:00-11:59Z
LATE = [3, 4, 5, 6, 7, 8]                # #51, #57: 10:00-15:59Z
# The ORDERED schedule #3578 ruled and the board carries now: their window opens one
# minute AFTER #56's own 08:00Z-09:59Z window closes, so an on-schedule #56 has
# delivered before their first tick. The declared-versus-replayed equality at the end
# of `test_the_starved_leg_declares_its_fail_forward_and_56_does_not` refuses this
# file the moment these hours and the live board disagree, so every instant replayed
# against `LATE` below is an instant of THIS window.


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


def _upstream_due(aut, up, up_last, board, now, seen):
    """The in-window "the upstream dispatches this very tick" prediction, with
    the dependent already in `seen` exactly as the gate calls it."""
    return aut._upstream_due_in_window(up, up_last, board, now, seen)


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
    # opened after the window closed). The refusal names the bound that will
    # release it (#1538 clause 4); `bypass=False` keeps the bare string, since
    # with no bound declared there is no bound to name.
    now = at(aut, monkeypatch, Z(24, 5, 0))
    assert reason(aut, b[42], board, now) == (
        "waiting on #38" if not bypass else
        f"waiting on #38 (inside its {b[42]['stale_bypass_hours']:g} h "
        "stale_bypass window: #38 ran 14.5 h ago)")
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
    """#56 slipped to 14:31Z on 09-23, outside its 08:00Z-09:59Z window.

    Under the ordered schedule #51/#57 open at 10:00Z, one minute after #56's own
    window shuts, so an undelivered #56 is already unrunnable until the next day by
    their first tick: they hold the WHOLE window (measured at 10:00Z and 15:30Z)
    rather than consume the out-of-window 09-23 run. Dispatched in window instead,
    its completion is what their 10:00Z opening consumes.
    """
    board = _trajectory_board()
    b = by_id(board)
    _record(aut, 56, "run_56_20260923_143102", Z(23, 14, 31, 29))

    # Their whole 09-24 window with #56 unrun: #56's own window opened and shut
    # before this one did, so nothing it could still dispatch this cycle exists.
    for probe in (Z(24, 10, 0), Z(24, 15, 30)):
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) == "waiting on #56", probe
        assert reason(aut, b[56], board, now) == "outside hours 01-02", probe
        assert reason(aut, b[58], board, now) == "waiting on #57", probe
        assert reason(aut, b[83], board, now) == "waiting on #58", probe

    # The same cycle seen from #56's own window, which the 09-23 slip does not
    # cost it: it is due at its 08:00Z opening and completes inside the window.
    now = at(aut, monkeypatch, Z(24, 8, 0))
    assert due(aut, b[56], board, now) is True, (
        "the slip no longer costs #56 its 09-24 window")
    complete(b[56], Z(24, 8, 5))

    now = at(aut, monkeypatch, Z(24, 10, 0))
    assert due(aut, b[51], board, now) is True
    assert due(aut, b[57], board, now) is True
    complete(b[57], Z(24, 10, 40))

    # #58 and #83 have no window: their edges keep the `interval / 2` rule.
    now = at(aut, monkeypatch, Z(24, 10, 45))
    assert due(aut, b[58], board, now) is True
    complete(b[58], Z(24, 11, 30))
    now = at(aut, monkeypatch, Z(24, 11, 31))
    assert due(aut, b[83], board, now) is True


def test_the_healthy_chain_still_holds_the_dependent_until_todays_upstream(
        aut, monkeypatch):
    """Finding 2's HEALTHY case: #56 ran in window at 08:05Z 09-22. #57's window
    opens at 10:00Z on 09-23, when that output is 25.9 h old and today's #56 is
    due at its own 08:00Z opening, so #57 must hold across its whole window — the
    inversion the gate exists to prevent, and the reason an ordered schedule does
    not make the dependency gate redundant."""
    board = _trajectory_board(up_last=Z(22, 8, 5))
    b = by_id(board)
    b[57]["last_run"] = Z(22, 8, 18).isoformat()
    for probe in (Z(23, 10, 0), Z(23, 13, 0), Z(23, 15, 30)):
        now = at(aut, monkeypatch, probe)
        assert reason(aut, b[57], board, now) == "waiting on #56", probe
    complete(b[56], Z(24, 8, 35))
    now = at(aut, monkeypatch, Z(24, 10, 0))
    assert due(aut, b[57], board, now) is True


def test_the_same_window_healthy_chain_holds_until_the_upstream_ran(
        aut, monkeypatch):
    board = [task(38, hours=NIGHT, last=Z(22, 5, 5)),
             task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16), stale_bypass_hours=36)]
    b = by_id(board)
    now = at(aut, monkeypatch, Z(23, 5, 0))
    assert due(aut, b[38], board, now) is True
    # 23.9 h after #38 ran: inside this dependent's own 36 h bound, so the
    # refusal says which clock has not run out (#1538 clause 4).
    assert reason(aut, b[42], board, now) == (
        "waiting on #38 (inside its 36 h stale_bypass window: #38 ran 23.9 h ago)")
    complete(b[38], Z(23, 5, 4))
    now = at(aut, monkeypatch, Z(23, 5, 5))
    assert due(aut, b[42], board, now) is True
    # Consumed: once #42 has run on this #38 output it is not fresh again.
    complete(b[42], Z(23, 5, 30))
    now = at(aut, monkeypatch, Z(23, 6, 0))
    assert reason(aut, b[42], board, now) == "waiting on #38"


def test_an_upstream_that_never_gets_its_window_keeps_the_dependent_waiting(
        aut, monkeypatch):
    """Owed and not delivered: the dependent waits the whole window (probed at its
    15:30Z tick, 25.5 h after the last #56 run). Only an explicitly declared
    `stale_bypass_hours` forwards on the old output — the #3578 ruling's stated
    reason for keeping #57's bound, and why it is NOT a dead declaration."""
    board = _trajectory_board()
    b = by_id(board)
    now = at(aut, monkeypatch, Z(24, 15, 30))
    assert reason(aut, b[57], board, now) == "waiting on #56"
    b[57]["stale_bypass_hours"] = 16
    assert due(aut, b[57], board, now) is True, "the override still works"


def test_a_run_just_before_the_window_is_not_repeated_and_feeds_the_chain(
        aut, monkeypatch):
    """A manual #56 at 07:00Z, an hour before its window: the floor of half a
    period keeps it from running again at 08:00Z, and because it owes nothing
    more this cycle its dependents consume that run at their 10:00Z opening."""
    board = _trajectory_board(up_last=Z(24, 7, 0))
    b = by_id(board)
    now = at(aut, monkeypatch, Z(24, 8, 0))
    assert due(aut, b[56], board, now) is False
    assert aut._next_run_after(b[56], Z(24, 7, 0)) == Z(25, 8)
    now = at(aut, monkeypatch, Z(24, 10, 0))
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
    monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "sys",
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


# ── A declared bound outranks a prediction about the next tick (#1538) ──────
#
# `test_a_bypass_waits_for_an_upstream_due_in_its_window_this_tick` pinned the
# opposite rule and is replaced by the two tests below. That test was added by
# #1526 on 2026-09-26 08:20Z — twenty minutes before #1538's own triage
# (08:46Z) measured what the rule it pinned did to #42: with #38 `up_next` and
# due in the shared window, the short-circuit ran BEFORE `stale_bypass_hours` was
# consulted, so the bound was unreachable in exactly the hours the dependent may
# run and #42 stayed frozen from 2026-09-22T05:16:36Z through four cycles, #39
# behind it. Both tests keep what that one was actually protecting — the
# `in_progress` race and the healthy chain — and give up only the claim that a
# prediction can overrule a bound the owner declared.

def test_the_bound_releases_the_dependent_inside_the_shared_window(
        aut, monkeypatch, tmp_path):
    """#1538 clause 1. A daily dependent declaring `stale_bypass_hours: 36`
    sharing its upstream's window, upstream `up_next` and due inside that window,
    its last successful run older than the bound: the dependent is released to
    dispatch at an in-window instant past its own `next_run` (it was held
    here — the refusal came back as `waiting on #38`)."""
    art = tmp_path / "signals-latest.md"
    art.write_text("x" * 4096)
    board = [task(38, hours=NIGHT, last=Z(23, 14, 31, 42), output_artifact=str(art)),
             task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16, 36),
                  stale_bypass_hours=36)]
    b = by_id(board)
    now = at(aut, monkeypatch, Z(25, 5, 0))
    # The upstream is due this very tick, and the dependent is 38.47 h past its
    # own bound (05:00Z 09-25 minus #38's last completion 14:31:42Z 09-23) with
    # its upstream's artifact on disk. That is the acceptance instant: released,
    # not `waiting on #38`.
    assert due(aut, b[38], board, now) is True
    assert 38.47 < (now - Z(23, 14, 31, 42)).total_seconds() / 3600 < 38.48
    assert reason(aut, b[42], board, now) is None
    assert due(aut, b[42], board, now) is True, (
        "#42 was held at this instant for four cycles with failure_count 0")
    # What that rule was genuinely for is still held: an upstream RUNNING is an
    # artifact in progress, and no bound forwards onto one.
    b[38]["status"] = "in_progress"
    assert reason(aut, b[42], board, now) == (
        "waiting on #38 (stale_bypass 36 h passed; #38 is in_progress)")
    assert due(aut, b[42], board, now) is False
    # And the healthy chain still runs in order once the upstream completes.
    b[38]["status"] = "up_next"
    complete(b[38], Z(25, 5, 20))
    now = at(aut, monkeypatch, Z(25, 5, 21))
    assert due(aut, b[42], board, now) is True


def test_the_same_upstream_still_holds_the_dependent_inside_the_bound(
        aut, monkeypatch, tmp_path):
    """#1538 clause 2, the purpose clause for weakening the gate. Same board one
    hour earlier in the upstream's elapsed clock — #38 `up_next`, due this tick
    in the shared window, last run 35.0 h ago, one hour INSIDE #42's 36 h bound —
    and the dependent is still held: the chain cannot dispatch it beside an
    upstream that is about to write this cycle's file."""
    art = tmp_path / "signals-latest.md"
    art.write_text("x" * 4096)
    board = [task(38, hours=NIGHT, last=Z(24, 19, 0), output_artifact=str(art)),
             task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16, 36),
                  stale_bypass_hours=36)]
    b = by_id(board)
    now = at(aut, monkeypatch, Z(26, 6, 0))      # 23:00 local: inside both windows
    assert (now - Z(24, 19, 0)).total_seconds() / 3600 == 35.0
    assert due(aut, b[38], board, now) is True, (
        "the upstream must be due in-window here, or this pins nothing")
    assert reason(aut, b[42], board, now) == (
        "waiting on #38 (inside its 36 h stale_bypass window: #38 ran 35.0 h ago)")
    assert due(aut, b[42], board, now) is False
    # Two hours into the same window the same upstream, two hours older, is
    # released with the upstream still due and still unwritten — the bound is the
    # whole difference, not the prediction about the next tick.
    now = at(aut, monkeypatch, Z(26, 8, 0))      # 01:00 local
    assert (now - Z(24, 19, 0)).total_seconds() / 3600 == 37.0
    assert due(aut, b[38], board, now) is True
    assert reason(aut, b[42], board, now) is None
    assert due(aut, b[42], board, now) is True


def test_the_override_still_forwards_when_the_upstream_will_not_run(
        aut, monkeypatch):
    board = [task(38, hours=NIGHT, last=Z(23, 14, 31, 42), status="failed"),
             task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16, 36),
                  stale_bypass_hours=36)]
    b = by_id(board)
    now = at(aut, monkeypatch, Z(25, 5, 0))
    assert due(aut, b[42], board, now) is True


def test_the_in_window_prediction_applies_the_upstreams_run_record(
        aut, monkeypatch, tmp_path):
    """#1538 clause 3. The "the upstream will dispatch this very tick"
    prediction read only the task FILE, while `_is_task_due` also drops an
    upstream with a successful run record for the current period (#1296 reverted
    #38's `last_run` from 09-25T18:53Z to 09-22T05:16Z while `runs/` still held
    the 09-25 success). An upstream the run record says already ran its period
    cannot be the one about to write this cycle's file.

    Predicted from the file alone, and the refusal on a held dependent says its
    upstream is due in its window this tick; with the run record on disk, the same
    probes do not claim it. On 2026-09-24 `queue` holds no row for #38 at all
    while every in-window probe of #42 read "upstream due in-window"."""
    art = tmp_path / "signals-latest.md"
    art.write_text("x" * 4096)
    board = [task(38, hours=NIGHT, last=Z(24, 19, 0), output_artifact=str(art)),
             task(42, hours=NIGHT, dep=38, last=Z(22, 5, 16, 36),
                  stale_bypass_hours=36)]
    b = by_id(board)
    now = at(aut, monkeypatch, Z(26, 8, 0))      # 01:00 local, 37 h past the bound
    seen = frozenset({"42"})
    up_last = Z(24, 19, 0)
    assert _upstream_due(aut, b[38], up_last, board, now, seen) is True, (
        "fixture: the file alone says #38 is about to dispatch")
    # Hold the dependent on the artifact clause so the refusal reports the
    # prediction, and it does.
    art.unlink()
    assert reason(aut, b[42], board, now) == (
        "waiting on #38 (stale_bypass 36 h passed; #38's declared output_artifact "
        "is not on disk; #38 is due in its window this tick)")
    assert _upstream_due(aut, b[38], up_last, board, now, seen) is True

    # The same instant with a SUCCESSFUL #38 run from the current period on disk,
    # two hours before this probe and 35 h newer than the `last_run` the reverted
    # file still shows — the #1296 shape exactly.
    _record(aut, 38, "run_38_20260926_055934", Z(26, 6, 0))
    assert _upstream_due(aut, b[38], up_last, board, now, seen) is False, (
        "an upstream whose run record says it already ran this period is still "
        "predicted as dispatching")
    assert reason(aut, b[42], board, now) == (
        "waiting on #38 (stale_bypass 36 h passed; #38's declared output_artifact "
        "is not on disk)")
    # And `_is_task_due` agrees, which is the point of the clause: the prediction
    # is no stronger than the gate it predicts.
    assert due(aut, b[38], board, now) is False


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
    # next window: upstream 09:00-09:59 local (16:00-16:59Z), dependent
    # 03:00-08:59 local (10:00-15:59Z, `LATE`). The hand-off is 18.0 h old at the
    # probe, and #1's own window shut before #2's opened, so it owes nothing the
    # probe can wait for. (The pre-#3578 shape here — upstream 13:00Z, dependent
    # opening at 06:00Z — is the INVERSION, and no longer what the live board is.)
    ([9, 10], LATE, Z(23, 16, 5), Z(24, 10, 0)),
])
def test_a_cycle_whose_hand_off_is_older_than_half_a_day_still_runs(
        aut, monkeypatch, up_hours, down_hours, up_done, probe):
    """The old bound held both shapes for good (13.9 h and 18.0 h > 12 h) unless
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


def _live_field(tid, key):
    """One front-matter field as the live board declares it, for task `tid`."""
    path = next(iter(LIVE_BOARD_DIR.glob(f"{tid}-*.md")), None)
    assert path is not None, f"no task file for #{tid} under {LIVE_BOARD_DIR}"
    return (autonomy._parse_task_file(path) or {}).get(key)


def _live_bypass_bound(tid):
    """`stale_bypass_hours` as the live board declares it, for task `tid`."""
    return _live_field(tid, "stale_bypass_hours")


def _window_ages(aut, tid, upstream_done):
    """The two ages of `upstream_done` that decide whether a declared
    `stale_bypass_hours` is usable at all, for the first window task `tid` opens
    after it: its age AT THE OPENING and its age AT THE CLOSE.

    A bound at or below the first never holds the window: the bypass is past on the
    first tick the dependent may run, which is #3365's daily inversion. A bound at
    or above the second never opens it: by the close the bypass still has not
    expired, and the leg waits a further day — #1526's outage. A bound serves a
    night only strictly between them, so the pair, not any number, is the contract;
    both ends come from the live board's own `preferred_hours` through the engine's
    own window derivation (the `aut` fixture's pinned PDT zone is what makes them
    UTC instants), so when either window moves these tests say so with the two ages
    instead of with a ceiling somebody once computed."""
    hours = autonomy._window_hours(
        {"preferred_hours": _live_field(tid, "preferred_hours")})
    assert hours, f"#{tid} declares no window on the live board"
    openings = [occ for occ in autonomy._window_occurrences(hours, upstream_done)
                if occ[0] > upstream_done]
    assert openings, f"no window of #{tid} opens after {upstream_done}"
    opening, closing = openings[0]
    return ((opening - upstream_done).total_seconds() / 3600,
            (closing - upstream_done).total_seconds() / 3600)


# Where #56's output actually exists on a healthy night. `completed_at` in
# `~/lloyd-data/autonomy-runs/56/` is between 08:01:14Z and 08:02:05Z on every day
# from 10-21 to 10-26 — inside the first minutes of its own 08:00Z-09:59Z window —
# so the window's opening, 08:00Z, is the instant its dependents see. Against it
# their 10:00Z opening — one minute after #56's own window shuts, under the ordered
# `[3, 4, 5, 6, 7, 8]` schedule #3578 ruled — carries a 2.00 h-old upstream and
# their 15:59Z close an 8.00 h-old one, which is the band
# `test_the_starved_leg_declares_its_fail_forward_and_56_does_not` grades the live
# bound on. Pre-#3578 those two instants were 06:00Z and 11:59Z, carrying 21.97 h
# and 27.97 h, which is why every bound the fleet has shipped reads differently now
# and why the nights below are replayed at the bound that was live ON them.
UPSTREAM_AT_ITS_OPENING = Z(25, 8, 0)

# The other end of #56's own declared window: `preferred_hours` [1, 2] local is
# 08:00Z-09:59Z, so 09:59Z is the latest an on-schedule #56 can finish. What that
# costs the dependents is what
# `test_a_completion_at_56_s_close_costs_their_next_window_only_above_the_band`
# measures — nothing about that #56 is late, and under the ordered schedule it
# delivered an hour before their window opened.
UPSTREAM_AT_ITS_CLOSE = Z(25, 9, 59)

# The node ids #3617 was filed for, named here so
# `test_no_node_this_item_names_is_skipped_xfailed_or_remarked` can hold the clause
# rather than leave it to a diff reading. Each is a node this item re-derived against
# the ordered schedule; the guard is what refuses a later round marking one of them
# to make this file green again.
ITEM_NAMED_NODES = (
    "test_a_completion_starved_past_the_close_releases_that_same_window",
    "test_the_starved_leg_declares_its_fail_forward_and_56_does_not",
    "test_the_live_bound_costs_the_starved_night_one_window_and_releases_the_next",
    "test_the_live_bound_holds_todays_opening_and_forwards_once_56_ran",
    "test_a_completion_at_56_s_close_costs_their_next_window_only_above_the_band",
)

# The two bounds the fleet has shipped, kept as named constants because the nights
# each was bought for are replayed here as BEHAVIOUR at that value. #1526 shipped
# 12 h on 2026-09-26 and served the starved night below; #3365 raised the live
# board to 26 h on 2026-10-26 (vault commit 8ad529fc); #3617 lowered it to 6 h on
# 2026-10-31 under the ordered windows. A node that read the LIVE bound for one of
# those nights would pin neither of them.
BOUND_AT_1526 = 12.0
BOUND_AT_3365 = 26.0


# The night the pool starved #56: queue row 515 `enqueued_at 2026-09-25T08:00:33Z`
# (inside #56's own 08:00-09:59Z window) and `claimed_at 2026-09-25T19:10:40Z` —
# 11 h 10 m of starvation, so the run finished at 19:14:12Z. Against the
# `[3, 4, 5, 6, 7, 8]` windows #51/#57 declare today, closing at 15:59Z, that is
# 3 h 15 m after their window had shut, and their next one opens with the output
# 14.76 h old. Its record `autonomy-runs/56/run_56_20260925_191040.md` carries
# `in_window: false`.
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


def test_the_starved_leg_declares_its_fail_forward_and_56_does_not(aut):
    """#1526 clause 1, restated as the invariant its ceiling was a proxy for.

    #56 has no `depends_on`, so a bound on it is read by nothing; #51 and #57 are
    the tasks the pool has been holding since 2026-09-22, and each must declare
    one the LIVE SCHEDULE can actually use — which is a band, not a number, and
    the band is derived here from the two windows the board declares.

    The ceiling this test used to carry (`0 < bound <= 17`) was arithmetic on one
    night, and so was the 22-28 h band its successor quoted: both measured a window
    that opened at 06:00Z, BEFORE #56's own. #3578 ruled the ordered schedule the
    board now carries — #51/#57 at `[3, 4, 5, 6, 7, 8]` local, opening one minute
    after #56's 09:59Z close — and that voids both numbers rather than moving them:
    against a #56 finishing at its own 08:00Z opening the band is now (2.00 h,
    8.00 h), which is where the 6 h #3617 landed sits. A ceiling copied out of one
    night's arithmetic outlived the night once already; what a clause can hold
    through a schedule change is the shape, and the shape is tested here — the bound
    strictly between the upstream's age at the window's opening and at its close,
    both ends derived from the board's own `preferred_hours`. The two failure modes
    stay named in the assertion messages below: a bound at or below the opening age
    releases the leg on the first tick of every window beside an upstream still owed
    (the stale read #3365 was filed from), and one at or above the close age never
    opens that window at all (the outage #1526 was filed for — #3365's 26 h does
    exactly that on the starved night, which is the night's own node below).

    The nights stay pinned, each at the bound it was served by:
    `test_a_completion_starved_past_the_close_releases_that_same_window` replays
    2026-09-25 at #1526's 12 h,
    `test_the_live_bound_costs_the_starved_night_one_window_and_releases_the_next`
    says what #3365's 26 h gave up on that same night,
    `test_the_live_bound_holds_todays_opening_and_forwards_once_56_ran` carries the
    hold-then-forward that 26 h bought, replayed at 26 h, and
    `test_the_pre_3365_bound_released_the_window_beside_an_upstream_still_owed`
    is the falsifier for bringing 12 h back.
    """
    assert _live_bypass_bound(56) is None, "#56 has no depends_on: a bound is noise"
    for tid in (51, 57):
        raw = _live_bypass_bound(tid)
        assert raw is not None, f"#{tid} declares no stale_bypass_hours"
        bound = float(raw)
        at_open, at_close = _window_ages(aut, tid, UPSTREAM_AT_ITS_OPENING)
        assert at_open < at_close, f"fixture: #{tid}'s window closes before it opens"
        assert at_open < bound < at_close, (
            f"#{tid} declares {bound:g} h, but an upstream finished at #56's 08:00Z "
            f"opening is {at_open:.2f} h old when #{tid}'s window opens and "
            f"{at_close:.2f} h old when it closes, so {bound:g} h "
            + ("releases it on the first tick of every window — the stale read "
               "#3365 raised the bound for (see "
               "test_the_pre_3365_bound_released_the_window_beside_an_upstream_still_owed)"
               if bound <= at_open else
               "never opens that window at all — the outage #1526 was filed for "
               "(see test_a_completion_starved_past_the_close_releases_that_same_"
               "window)"))

        # The band is only a contract if it excludes things. Both other bounds this
        # file's history has used fall outside it — 12 h is what the board declared
        # when #3365 measured the stale read, 30 h is the ceiling #1526's clause
        # allowed — and under the ordered schedule both sit ABOVE the window's close
        # age, so each would leave a leg waiting a day for a #56 that already ran.
        # The check below is therefore not a window widened until a value fits.
        for dead in (BOUND_AT_1526, 30.0):
            assert not (at_open < dead < at_close), (
                f"#{tid}'s band ({at_open:.2f}, {at_close:.2f}) admits {dead:g} h, "
                "which is the bound that caused one incident (12 h, #3365) or the "
                "ceiling that would have caused the other (30 h, #1526's clause): "
                "the band no longer discriminates")

        # Where the bound's expiry lands inside the window it is supposed to open,
        # stated as a number instead of left to a change note. An upstream finishing
        # at the END of #56's own window — still inside its `preferred_hours`, so
        # nothing about it is late — is `late_open` h old when the dependents' next
        # window opens and `late_close` h old when it closes, so `late_close - bound`
        # is the part of that window still inside the bound. At the live 6 h that is
        # 1 minute: their window opens 10:00Z on an upstream 0.02 h old and the bound
        # expires at 15:59Z, the last minute of it. Both edges are asserted and
        # neither is a tuning target — a bound at or past `late_close` never opens the
        # window (the shape #3365's 26 h has on the starved night, measured in
        # `test_a_completion_at_56_s_close_costs_their_next_window_only_above_the_band`),
        # and one at or below `late_open` has the window entered already released,
        # which is the daily stale read #3365 was filed for. A window move takes the
        # number off this band and re-derives it here.
        late_open, late_close = _window_ages(aut, tid, UPSTREAM_AT_ITS_CLOSE)
        usable = late_close - bound
        assert late_open < bound, (
            f"#{tid} declares {bound:g} h and an upstream finishing at #56's close "
            f"is only {late_open:.2f} h old when their window opens: the bound is "
            "already past on the first tick, so the freshness rule rather than the "
            "board's declared value is what decides that morning")
        assert 0 < usable <= 1.0, (
            f"#{tid} declares {bound:g} h against an upstream finishing at #56's own "
            f"close: {usable:.2f} h of their next window is still inside it. At or "
            "below 0 the bound never opens that window — #3365's 26 h on the starved "
            "night — and further inside it the bound expires so early that it is no "
            "longer sized to the schedule's latest completion")

    # And the band and the replays must describe one fleet. Every behavioural node
    # here runs a synthetic board with hardcoded hours (`LATE` for #51/#57, [1, 2]
    # for #56); the bound above is graded against the LIVE board's `preferred_hours`.
    # Assert the equality here, where a window move first makes a verdict wrong,
    # rather than leaving the two to drift apart silently.
    for tid, replayed in ((51, LATE), (57, LATE), (56, [1, 2])):
        declared = _live_field(tid, "preferred_hours") or []
        assert sorted(declared) == sorted(replayed), (
            f"#{tid} declares preferred_hours {declared} on the live board but the "
            f"replayed nights use {replayed}: the band is computed from one window "
            "and the releases asserted against another")


def test_the_starved_chain_releases_at_its_window_opening(aut, monkeypatch):
    """#1526 clause 2 — the three instants the triage replayed, at the bound read
    off the live board.

    #56's last completion was 2026-09-23T14:31:29Z, so at 03:00 local on 09-25
    (10:00Z, when #51/#57's window opens) it is 43.5 h old — past the declared bound
    at the opening, not merely near it. Before #1526 neither task declared a bound,
    and `test_an_upstream_that_never_gets_its_window_keeps_the_dependent_waiting`
    above pins what that means: `waiting on #56` at every hour of every window, four
    windows on the row with no run, #58 and #83 down behind them.
    """
    bound = float(_live_bypass_bound(51))
    board = _starved_board(bound)
    board[0]["last_run"] = Z(23, 14, 31, 29).isoformat()
    b = by_id(board)
    assert (Z(25, 10, 0) - Z(23, 14, 31, 29)).total_seconds() / 3600 > bound

    for label, probe in (("03:00 local", Z(25, 10, 0)),    # window opens 10:00Z
                         ("06:00 local", Z(25, 13, 0)),
                         ("08:30 local", Z(25, 15, 30))):  # closes 15:59Z
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) is None, (
                f"#{tid} still held at {label}: the declared bound does not open "
                "inside the window, so the starved leg stays down")
            assert due(aut, b[tid], board, now) is True, (label, tid)


def test_the_bound_releases_them_though_56_owes_the_cycle_and_cannot_run(
        aut, monkeypatch):
    """#1538 clause 1 on the #56 → #51 edge, replacing #1526's clause-3 valve.

    Under the ordered schedule this edge is harder than the one #1538 argued: at
    #51/#57's 10:00Z and 13:00Z probes #56's own 08:00Z-09:59Z window has ALREADY
    shut for the day, so #56 owes this cycle and cannot dispatch until tomorrow —
    `due` says False and `hold_reason` says `outside hours 01-02` at the very
    instants the dependents are open. #1526 clause 3 held the dependents while the
    upstream was due in-window; #1538 reversed that, because the bound is an owner's
    decision to forward on the previous cycle's file and #42 sat 98+ h at
    `failure_count: 0` behind this same valve with #38 `up_next`. Here the bound is
    past at both probes (43.5 h and 46.5 h against the live 6 h) and the in-window
    prediction cannot even be evaluated, so the bound decides without an argument to
    have. That is why this node checks `due(b[56]) is False` rather than True: if a
    later round re-adds the valve it will find no instant on this edge to fire it.

    What the valve was for is pinned twice over rather than dropped: the race is
    impossible inside the bound
    (`test_the_same_upstream_still_holds_the_dependent_inside_the_bound`), and
    both refusals below still hold at these very instants — an upstream
    `in_progress` is an artifact being written, and a dependent that declared no
    bound has no fail-forward path at all."""
    bound = float(_live_bypass_bound(51))
    board = _starved_board(bound)
    board[0]["last_run"] = Z(23, 14, 31, 29).isoformat()
    b = by_id(board)
    for label, probe in (("03:00 local", Z(25, 10, 0)), ("06:00 local", Z(25, 13, 0))):
        age_h = (probe - Z(23, 14, 31, 29)).total_seconds() / 3600
        assert age_h > bound, (label, age_h)
        now = at(aut, monkeypatch, probe)
        assert due(aut, b[56], board, now) is False, (
            f"#56 must NOT be dispatchable at {label}: its window shut before the "
            "dependents' opened, which is what makes the #1526 valve unreachable")
        assert reason(aut, b[56], board, now) == "outside hours 01-02", label
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) is None, (
                f"#{tid} held at {label}: {age_h:.1f} h after #56 last ran, past "
                f"its own {bound:g} h bound, with #56 owed and unable to run — the "
                "#42 starvation shape")
            assert due(aut, b[tid], board, now) is True, (label, tid)

    now = at(aut, monkeypatch, Z(25, 13, 0))
    b[56]["status"] = "in_progress"
    for tid in (51, 57):
        assert reason(aut, b[tid], board, now) == (
            f"waiting on #56 (stale_bypass {bound:g} h passed; "
            "#56 is in_progress)"), tid
        assert due(aut, b[tid], board, now) is False, tid
    b[56]["status"] = "up_next"
    for tid in (51, 57):
        b[tid].pop("stale_bypass_hours")
    assert reason(aut, b[51], board, now) == "waiting on #56", (
        "a dependent with no declared bound forwards on nothing, and the bare "
        "refusal it has always printed must not gain a clause about a bound it "
        "does not declare")
    assert due(aut, b[51], board, now) is False


def test_a_completion_starved_past_the_close_releases_that_same_window(
        aut, monkeypatch):
    """#1526's acceptance: the pool-latency night, not merely the wedged one.

    This is the night that actually broke: #56 completed at 19:14:12Z on 09-25,
    3 h 15 m after the dependents' 15:59Z close, so their next window —
    10:00Z-15:59Z on 09-26 — starts with an upstream 14 h 46 m old, fresh by any
    daily measure and not owed again until 09-27. The elapsed rule therefore calls
    it this cycle's output and holds the window; only the declared bound forwards
    it, and only once the upstream is a day-and-a-half of nothing. At the shipped
    12 h the bypass expired at 07:14:12Z, 2 h 46 m BEFORE this window opened, so
    all three of the instants below run free and the leg works the very night it
    was starved. Strip the bound and all three hold, which is the outage this item
    is about.

    Replayed at #1526's own 12 h (`BOUND_AT_1526`), not at the bound the board
    declares today: #3365 raised that to 26 h on 2026-10-26, and 26 h is past at
    neither end of this night, so running it here would assert a release that no
    longer happens. What the bound #3365 shipped does on this exact night — forfeits
    this window, releases the next one — is pinned by
    `test_the_live_bound_costs_the_starved_night_one_window_and_releases_the_next`,
    and the night the board's 6 h IS about by
    `test_the_live_bound_holds_todays_opening_and_forwards_once_56_ran`. A replay
    at a bound chosen for a different night pins neither.

    The instants themselves moved with #3578's ordered schedule. Before it the
    dependents opened at 06:00Z, so this night started 10.8 h old — INSIDE the 12 h
    bound — and the 23:00-local probe below was a hold that named the bound in its
    refusal. The window moving behind #56's put the opening past the expiry, which
    is the schedule change doing the work, not a bound move: the same probe at 26 h
    is still a hold, one node down.
    """
    bound = BOUND_AT_1526
    board = _starved_board(bound)
    b = by_id(board)
    probes = (Z(26, 10, 0), Z(26, 13, 0), Z(26, 15, 30))   # 03:00/06:00/08:30 local
    labels = ("03:00 local", "06:00 local", "08:30 local")
    expiry = LATE_AT + dt.timedelta(hours=bound)
    assert expiry < probes[0], (
        f"fixture: at {bound:g} h the bypass expires {expiry.isoformat()}, at or "
        "after the opening of the window this node replays, so this night is no "
        "longer the one this bound was bought for — re-derive the instants")

    for label, probe in zip(labels, probes):
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) is None, (
                f"#{tid} still held at {label}: {bound:g} h does not open this "
                "window, so the leg starved on 09-25 starves again on 09-26")
            assert due(aut, b[tid], board, now) is True, (label, tid)

    stripped = _starved_board(None)
    bs = by_id(stripped)
    for label, probe in zip(labels, probes):
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert reason(aut, bs[tid], stripped, now) == "waiting on #56", (
                f"#{tid} released at {label} with no bound declared — this test "
                "would pass whatever the board says")


def test_the_live_bound_costs_the_starved_night_one_window_and_releases_the_next(
        aut, monkeypatch):
    """What the 26 h #3365 shipped gave up on #1526's night, said out loud instead
    of left in a change note.

    The night that broke is replayed unchanged — #56 completed 19:14:12Z on 09-25,
    3 h 15 m after the dependents' 15:59Z close — so that output is 14.8 h old when
    their next window opens and 20.3 h old at the last probe inside it, and a bound
    of 26 h is past at neither end. All three of #1526's instants therefore HOLD,
    each refusal naming the bound that holds them; the bypass expires at 21:14:12Z
    on 09-26, 5 h 15 m after that window shut; and the window after it (10:00Z
    09-27, the output 38.8 h old) opens with both legs released.

    So the cost of a 26 h bound on this night is exactly ONE window, and the
    fail-forward path still ends the wait on its own clock rather than leaving the
    leg stuck as it was from 2026-09-22. What it no longer does is release the
    starved night itself — the trade #3365 made against the steady-state night
    pinned in `test_the_live_bound_holds_todays_opening_and_forwards_once_56_ran`,
    where 12 h was forwarding on 22 h-old input every single day. That is why #3617
    brought the board back to 6 h, and why this node replays at `BOUND_AT_3365`
    rather than reading the live file: at today's 6 h the starved night releases at
    its own opening and this node's premise — a bound that forfeits a window — has
    no subject on the live board at all.
    """
    bound = BOUND_AT_3365
    last_in_window = (Z(26, 15, 30) - LATE_AT).total_seconds() / 3600
    assert bound > last_in_window, (
        "fixture: this test is about a bound that forfeits the starved window, and "
        f"{bound:g} h is not past the {last_in_window:.2f} h an upstream is at the "
        "last probe inside it — #1526's replay now owns the other case")
    board = _starved_board(bound)
    b = by_id(board)
    for label, probe in zip(("03:00 local", "06:00 local", "08:30 local"),
                            (Z(26, 10, 0), Z(26, 13, 0), Z(26, 15, 30))):
        age_h = (probe - LATE_AT).total_seconds() / 3600
        assert age_h < bound, (label, age_h)
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) == (
                f"waiting on #56 (inside its {bound:g} h stale_bypass window: "
                f"#56 ran {age_h:.1f} h ago)"), (label, tid)
            assert due(aut, b[tid], board, now) is False, (label, tid)
    expires = LATE_AT + dt.timedelta(hours=bound)
    assert Z(26, 16, 0) < expires < Z(27, 10, 0), (
        "the bound must expire after this window's close and before the next "
        f"one's opening, and it expires at {expires.isoformat()}")
    now = at(aut, monkeypatch, Z(27, 10, 0))
    assert (now - LATE_AT).total_seconds() / 3600 > bound
    for tid in (51, 57):
        assert reason(aut, b[tid], board, now) is None, tid
        assert due(aut, b[tid], board, now) is True, (
            f"#{tid} stayed held into the next window: one forfeited window is "
            "#3365's cost, two is a wedge")


def test_the_live_bound_holds_todays_opening_and_forwards_once_56_ran(
        aut, monkeypatch):
    """#3365's whole case in one node — the behaviour its 26 h bought, replayed at
    26 h, which is what was landed through the vault route with nothing holding it.

    #56 finished at 08:02Z on 09-26 and slipped 09-27, so at the dependents'
    10:00Z opening on 09-27 that output is 25.97 h old: inside a 26 h bound, so both
    legs wait, and each refusal names the bound that is holding them rather than
    printing the bare `waiting on #56` that #1739 could not tell apart from a dead
    chain. Give #56 its file instead — 09:30Z on 09-27, half an hour before their
    window opens — and the SAME tick releases both legs: the dependent dispatched
    after its upstream's file, on that file, which is the ordering #3365 raised the
    bound to get. Skip #56 on 09-27 as well and the bound alone decides, as #1526
    requires: the opening still holds at 25.97 h and 15:30Z (31.76 h) releases both,
    so a missed upstream costs hours inside one window and not another day.

    Replayed at `BOUND_AT_3365`, because that hold is what that value bought. The
    board declares 6 h today: at 6 h this same morning releases at the opening, on an
    upstream 25.97 h old, and that trade is the last arm below — #3617 gave up the
    hold in order to stop the 6 h-era wait that costs a whole window on the starved
    night above. What #3365 bought is therefore still replayed as behaviour, at the
    value that bought it, and this node is the one that says what it costs."""
    bound = BOUND_AT_3365
    board = _starved_board(bound)
    b = by_id(board)
    up_done = Z(26, 8, 2)
    board[0]["last_run"] = up_done.isoformat()
    opening = Z(27, 10, 0)                      # their window opens 10:00Z
    age_h = (opening - up_done).total_seconds() / 3600
    assert age_h < bound, (
        "fixture: the opening must sit INSIDE the 26 h bound #3365 shipped, or the "
        f"hold this node pins is not the bound's doing ({age_h:.2f} h vs {bound:g} h)")
    now = at(aut, monkeypatch, opening)
    for tid in (51, 57):
        assert reason(aut, b[tid], board, now) == (
            f"waiting on #56 (inside its {bound:g} h stale_bypass window: "
            f"#56 ran {age_h:.1f} h ago)"), (opening, tid)
        assert due(aut, b[tid], board, now) is False, (opening, tid)
    complete(b[56], Z(27, 9, 30))
    for tid in (51, 57):
        assert reason(aut, b[tid], board, now) is None, (
            f"#{tid} still held once #56 had written this cycle's file: the "
            "hold was supposed to end when the upstream ran, which is the whole "
            "point of the bound")
        assert due(aut, b[tid], board, now) is True, tid
    board[0]["last_run"] = up_done.isoformat()      # #56 skipped 09-27 entirely
    for probe, releases in ((Z(27, 10, 0), False), (Z(27, 15, 30), True)):
        age_h = (probe - up_done).total_seconds() / 3600
        assert (age_h > bound) is releases, (probe, f"{age_h:.2f} h", bound)
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert (reason(aut, b[tid], board, now) is None) is releases, (
                probe, tid, f"{age_h:.2f} h old against a {bound:g} h bound")
            assert due(aut, b[tid], board, now) is releases, (probe, tid)

    # The same tick on the board that ships today: held by nothing. This is the
    # trade #3617 made — the 6 h bound lets a 25.97 h-old file through at the
    # opening, which is the read #3365 raised the bound to prevent — so a round
    # that reaches for 26 h again finds the reason and the cost in one place.
    live = float(_live_bypass_bound(51))
    board6 = _starved_board(live)
    board6[0]["last_run"] = up_done.isoformat()
    b6 = by_id(board6)
    now = at(aut, monkeypatch, opening)
    assert live < age_h, (
        f"fixture: {live:g} h is not past the {age_h:.2f} h the ordered schedule "
        "puts on this morning, so the two bounds no longer differ here")
    for tid in (51, 57):
        assert due(aut, b6[tid], board6, now) is True, (
            f"#{tid} held at the live {live:g} h bound: the hold pinned above is "
            "meant to be 26 h's behaviour, not the board's")


def test_the_pre_3365_bound_released_the_window_beside_an_upstream_still_owed(
        aut, monkeypatch):
    """The falsifier for restoring #1526's 12 h, and the lower edge of the band
    `test_the_starved_leg_declares_its_fail_forward_and_56_does_not` grades on.

    At the bound the board carried until 2026-10-26 the bypass was past before the
    window could open: #56 finished 08:02Z on 09-26 and is not due again until
    08:00Z on 09-27, so at the opening of the window it then had (06:00Z-11:59Z) its
    output was already 21.97 h old, past a 12 h bound, and both legs dispatched
    beside an upstream that still owed this cycle's run. At the ordered schedule the
    opening is later and the age therefore larger — 25.97 h at 10:00Z, past a 12 h
    bound by 14 h — so the inversion is not a tight edge case any more, it is most of
    a window. That is not a hypothetical replay:
    under the pre-#3578 window `autonomy-runs/51` and `/57` start at 06:00-06:01Z and
    `autonomy-runs/56` completes at 08:00-08:02Z the same UTC day on every day from
    10-23 to 10-26, nine consecutive stale reads, which is the measurement #3365
    raised the bound for. The ordered schedule #3578 ruled removed the geometric cause
    of that pattern, which is why #3617 could bring the bound inside the band at all;
    12 h comes back only if #56's window moves ahead of the dependents' again, and
    this node is what says so when it does.
    """
    board = _starved_board(BOUND_AT_1526)
    up_done = Z(26, 8, 2)
    board[0]["last_run"] = up_done.isoformat()
    b = by_id(board)
    for probe in (Z(27, 10, 0), Z(27, 15, 30)):      # opening and 3.5 h into the window
        assert (probe - up_done).total_seconds() / 3600 > BOUND_AT_1526, (
            "fixture: 12 h must be past at the probe, or the inversion has no edge")
        now = at(aut, monkeypatch, probe)
        assert due(aut, b[56], board, now) is False, (
            "fixture: #56 must still owe this cycle's run at the probe, or nothing "
            "is being inverted — it is #56's 08:00Z window that has not opened yet")
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) is None, (probe, tid)
            assert due(aut, b[tid], board, now) is True, (
                f"#{tid} was held at {probe.isoformat()} at 12 h: what 12 h actually "
                "did was release onto yesterday's file, and that release is the "
                "defect #3365 closed")


def test_a_completion_at_56_s_close_costs_their_next_window_only_above_the_band(
        aut, monkeypatch):
    """What a bound above the band costs, in minutes, and what one inside it does not.

    Nothing about this #56 is late: it finishes at 09:59Z, the last minute of the
    08:00Z-09:59Z window IT declares. Under the ordered schedule the dependents'
    window opens one minute after that, so their morning is decided by the freshness
    rule and nothing else: at today's live bound both legs are due at the 10:00Z
    opening on an upstream 0.02 h old. The same night at the 26 h #3365 shipped is a
    different story — 24.02 h old at their opening is inside that bound, so 09-27
    opens held, is still held at 11:30Z (25.5 h, refusal naming the bound), and
    releases at 12:30Z (26.5 h). Move the completion one minute later, to 10:00Z, and
    the leg is held just the same: the forfeit is the bound, not an off-window flag.

    So the cost this node exists to charge belongs to a bound ABOVE the (2.00 h,
    8.00 h) band, not to the board, and the shape that outlives any particular bound
    is the one in
    `test_the_starved_leg_declares_its_fail_forward_and_56_does_not`: a leg is only
    held while the bound is unexpired, and the band says which bounds that is.

    The 12 h night is pinned as behaviour, on its own synthetic board carrying
    `BOUND_AT_1526`, by
    `test_a_completion_starved_past_the_close_releases_that_same_window`; its scale is
    stated here, since this is the node that compares bounds. #1526's starved
    completion sits 14.76 h before their window opens and 20.76 h before it closes, so
    a 12 h bound expired 2 h 46 m BEFORE the opening and left all 6.00 h of that
    window usable — which is what 12 h bought on its own night, and why no node here
    calls 12 h wrong on #1526's geometry. It was wrong on #3365's, and the band moved
    when the schedule did.
    """
    live = float(_live_bypass_bound(51))
    late_open, late_close = _window_ages(aut, 51, UPSTREAM_AT_ITS_CLOSE)
    usable_min = (late_close - live) * 60
    assert 0 <= usable_min <= 2.0, (
        f"an upstream finishing at #56's own close is {late_open:.2f} h old at the "
        f"dependents' next opening and {late_close:.2f} h old at its close, so the "
        f"live {live:g} h bound leaves {usable_min:.1f} min of that window: this "
        "node's arithmetic no longer describes the live board — re-measure the margin")

    # The scale of the bound #1526 shipped, measured on ITS night rather than on
    # #3365's: expiry before the opening means the window is whole.
    starved_open, starved_close = _window_ages(aut, 51, LATE_AT)
    assert BOUND_AT_1526 < starved_open, (
        f"a {BOUND_AT_1526:g} h bound no longer expires before #1526's night opens "
        f"({starved_open:.2f} h at the opening): #3617's night is the one with a "
        "usable window, and this node's comparison has lost its subject")
    assert (starved_close - starved_open) * 60 == pytest.approx(6 * 60, abs=1.0), (
        "fixture: #1526's night is one 6-hour window wide, so 'the bound left it "
        f"whole' has to mean all {starved_close - starved_open:.2f} h of it")

    # Above the band the forfeit is behaviour, not arithmetic: #56 done at its own
    # 09:59Z close on 09-26, and at 26 h the dependents are still inside the bound at
    # 11:30Z on 09-27 — 25.5 h — so that window is spent waiting.
    board = _starved_board(BOUND_AT_3365)
    b = by_id(board)
    b[56]["last_run"] = UPSTREAM_AT_ITS_CLOSE.replace(day=26).isoformat()
    for probe, age_txt in ((Z(27, 10, 0), "24.0"), (Z(27, 11, 30), "25.5")):
        now = at(aut, monkeypatch, probe)
        for tid in (51, 57):
            assert reason(aut, b[tid], board, now) == (
                f"waiting on #56 (inside its {BOUND_AT_3365:g} h stale_bypass "
                f"window: #56 ran {age_txt} h ago)"), (probe, tid)
            assert due(aut, b[tid], board, now) is False, (probe, tid)
    now = at(aut, monkeypatch, Z(27, 12, 30))        # 26.5 h: past the bound
    for tid in (51, 57):
        assert reason(aut, b[tid], board, now) is None, tid
        assert due(aut, b[tid], board, now) is True, tid

    # A completion one minute later — 10:00Z, one minute outside #56's own window —
    # is held exactly as hard, so what is spending the window is the bound rather
    # than any in-window flag on the completion.
    later = _starved_board(BOUND_AT_3365)
    b2 = by_id(later)
    b2[56]["last_run"] = Z(26, 10, 0).isoformat()
    now = at(aut, monkeypatch, Z(27, 11, 30))
    assert reason(aut, b2[51], later, now) == (
        f"waiting on #56 (inside its {BOUND_AT_3365:g} h stale_bypass window: "
        "#56 ran 25.5 h ago)")

    # And at the bound the board ships, this same morning is a plain release: the
    # ordered schedule put #56's close one minute before their opening, so the
    # freshness rule answers it before any bound is consulted.
    board6 = _starved_board(live)
    b6 = by_id(board6)
    b6[56]["last_run"] = UPSTREAM_AT_ITS_CLOSE.replace(day=26).isoformat()
    now = at(aut, monkeypatch, Z(26, 10, 0))
    for tid in (51, 57):
        assert due(aut, b6[tid], board6, now) is True, (
            f"#{tid} held on an upstream {late_open:.2f} h old at the live "
            f"{live:g} h bound: the forfeit above is 26 h's behaviour, and if it is "
            "the board's again then #3617 has been undone")


def test_no_node_this_item_names_is_skipped_xfailed_or_remarked(aut):
    """Clause 2 of #3368 as a check, not as a diff reading.

    The clause is that the named tests PASS rather than being routed around: nothing
    skipped, xfailed, deleted, or re-marked `live_vault` / `fault_injection` to get
    the file green. That is testable from inside the file: every id in
    `ITEM_NAMED_NODES` is still here, none carries a mark that would drop or mute it,
    and no module-level `pytestmark` does either — the same selector the gate runs
    (`-m "not live_vault and not fault_injection"`) is what would hide them. Add
    `@pytest.mark.skip` to any of them and this one fails, which is the crispness
    proof; the suite cannot go green by muting a node this item named without going
    red here first. #3617 re-derived four of these nodes against the ordered schedule
    and renamed the fifth, which is exactly the set a later round could be tempted to
    quiet down instead of re-measuring.

    `aut` is not used by the assertions and is not decoration: it is the fixture that
    pins the machine-local zone, and importing it proves this module still runs under
    the same engine the other nodes drive.
    """
    import inspect

    mod = __import__(__name__)
    forbidden = {"skip", "skipif", "xfail", "live_vault", "fault_injection"}
    module_marks = {m.name for m in getattr(mod, "pytestmark", [])}
    assert not (module_marks & forbidden), (
        f"this module carries module-level {sorted(module_marks & forbidden)}: "
        "every node in it is being routed around, not fixed")
    for name in ITEM_NAMED_NODES:
        fn = getattr(mod, name, None)
        assert callable(fn), (
            f"{name} is gone from this file — deleting a node satisfies a run and "
            "fails the clause #3368 was filed for")
        marks = {m.name for m in getattr(fn, "pytestmark", [])}
        assert not (marks & forbidden), (
            f"{name} carries {sorted(marks & forbidden)} — the clause requires it to "
            "PASS, not to be excused")
        src = inspect.getsource(fn)
        assert "pytest.skip" not in src and "pytest.xfail" not in src, (
            f"{name} skips or xfails itself in its own body")

    # The other way to satisfy a run without satisfying a clause is to exclude the
    # node from the selection: an `addopts` deselect, a `-k` that drops it, or a
    # `testpaths` that stops collecting this file. Read the project's own pytest
    # config, because the gate runs that and nothing else.
    root = Path(inspect.getfile(mod)).parent.parent
    configured = []
    for name in ("pytest.ini", "tox.ini", "setup.cfg"):
        text = (root / name).read_text(encoding="utf-8", errors="replace") \
            if (root / name).exists() else ""
        configured.append(text)
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        configured.append(pyproject.read_text(encoding="utf-8", errors="replace"))
    blob = "\n".join(configured)
    for needle in ("testpaths", "deselect", "ignore", "--deselect"):
        for line in blob.splitlines():
            if needle in line and "test_autonomy_dependency_windows" in line:
                raise AssertionError(
                    f"pytest config {needle}: {line.strip()} — the file is being "
                    "excluded from the selection, which the clause forbids")


# ── #1739: a chain that COMPLETED reads as a chain that is HELD ───────────────
#
# Measured live at 2026-09-28T09:32Z against `dependency_resolution_set()`: all
# three links of #38 -> #42 -> #39 -> #40 had finished that night's cycle, and
# `hold_reason` answered `waiting on #38` / `waiting on #42` / `waiting on #39`
# for them — the same string a link gated by a dead upstream prints, which is
# the sentence #1737 was filed from. The `last_run` values and run ids below are
# those tasks' own fields and the records they wrote that night
# (`autonomy-runs/42/run_42_20260928_050613.md` and its two siblings).

PROBE_0928 = Z(28, 9, 32)                     # 02:32 local: all three windows open

RUNS_COMPLETED_0928 = {                        # dependent -> (run id, completed)
    42: ("run_42_20260928_050613", Z(28, 5, 13, 15)),
    39: ("run_39_20260928_080049", Z(28, 8, 20)),
    40: ("run_40_20260928_090013", Z(28, 9, 6, 2)),
}


def _completed_0928_chain():
    """The nightly reflection chain as the live board declares it (daily, the
    same `preferred_hours`, `stale_bypass_hours: 36` on every dependent), with
    every link completed inside its own window: #38 last at 05:05:31Z."""
    return [
        task(38, hours=NIGHT, last=Z(28, 5, 5, 31)),
        task(42, hours=NIGHT, dep=38, last=RUNS_COMPLETED_0928[42][1],
             stale_bypass_hours=36),
        task(39, hours=[1, 2, 3, 4], dep=42, last=RUNS_COMPLETED_0928[39][1],
             stale_bypass_hours=36),
        task(40, hours=[2, 3, 4], dep=39, last=RUNS_COMPLETED_0928[40][1],
             stale_bypass_hours=36),
    ]


def _owed_upstream_board(*, bound=None):
    """#42 with an upstream that still OWES this window: #38 last succeeded at
    06:00Z on 09-27, the 26th->27th window occurrence, so at this probe it is
    27.5 h old and today's run has not happened. `bound` is #42's own
    `stale_bypass_hours`, and with none declared there is no bound to name."""
    return [
        task(38, hours=NIGHT, last=Z(27, 6, 0)),
        task(42, hours=NIGHT, dep=38, last=RUNS_COMPLETED_0928[42][1],
             **({"stale_bypass_hours": bound} if bound else {})),
    ]


def test_a_dependent_that_ran_this_period_is_reported_done_not_waiting(
        aut, monkeypatch):
    """#1739 clause 1. A dependent whose run record says this period already
    ran, upstream finished earlier the SAME night, must read `already ran this
    period` and not `waiting on #38`.

    The refusal it has to beat is the undescribed one: `_dependency_refusal`
    returns at its "this cycle's output is already consumed" line, so the
    `_explain` detail is `''` even though `stale_bypass_hours: 36` is declared —
    `_bypass_hold_detail` is never reached, and there is no clock to name. An
    empty detail is the absence of a reason, and the run record on disk is the
    only predicate in `hold_reason` that reads a second account."""
    board = _completed_0928_chain()
    b = by_id(board)
    for tid, (run_id, when) in RUNS_COMPLETED_0928.items():
        _record(aut, tid, run_id, when)
    now = at(aut, monkeypatch, PROBE_0928)
    for tid, upstream in ((42, 38), (39, 42), (40, 39)):
        # Positive controls beside the string, so the row cannot pass on a
        # fixture that never plants a record or never reaches the dependency
        # branch at all.
        assert aut._already_ran_this_period(b[tid], now=now) == (
            RUNS_COMPLETED_0928[tid][0]), tid
        assert aut._dependency_refusal(b[tid], board, now=now,
                                       _explain=True) == "", tid
        assert reason(aut, b[tid], board, now) == aut.RUN_SUCCESS_HOLD, (
            f"#{tid} finished {RUNS_COMPLETED_0928[tid][1].isoformat()} and the "
            f"board still calls it held by #{upstream}")
        # Dispatch is untouched: the scheduler already refused this task, at its
        # OWN last gate, the run record. Only the sentence changes.
        assert due(aut, b[tid], board, now) is False, tid


def test_a_dependent_that_has_not_run_still_prints_what_holds_it(
        aut, monkeypatch):
    """#1739 clause 2, the paired negative control. The same two boards at the
    same instants with NO run record for the dependent, and every string is
    exactly what it was before #1739: this is the condition the fix is gated on,
    so a re-order that ignored the run record would flip these rows and fail
    here while passing the test above.

    Bare when nothing describes the hold — the consumed chain, and an upstream
    that owes the cycle with no bound declared — and `waiting on #N (detail)`
    when `_bypass_hold_detail` has a bound to name (#1538 clause 4)."""
    now = at(aut, monkeypatch, PROBE_0928)

    board = _completed_0928_chain()
    b = by_id(board)
    for tid, upstream in ((42, 38), (39, 42), (40, 39)):
        assert aut._already_ran_this_period(b[tid], now=now) == "", tid
        assert reason(aut, b[tid], board, now) == f"waiting on #{upstream}", tid

    owed = _owed_upstream_board()
    bo = by_id(owed)
    assert aut._already_ran_this_period(bo[42], now=now) == ""
    age_h = (now - Z(27, 6, 0)).total_seconds() / 3600
    assert 27.5 < age_h < 27.6, (
        "#38 must still owe this window here, or the two blocks below pin "
        "nothing about an upstream that is behind")
    assert reason(aut, bo[42], owed, now) == "waiting on #38", (
        "no bound declared, so no bound to name — the bare string is the "
        "pre-#1538 refusal and must stay bare")
    bo[42]["stale_bypass_hours"] = 36
    assert reason(aut, bo[42], owed, now) == (
        "waiting on #38 (inside its 36 h stale_bypass window: "
        "#38 ran 27.5 h ago)")


def test_a_hold_that_names_its_clock_beats_the_completed_cycle_claim(
        aut, monkeypatch):
    """#1739 clause 3. The owed board, the same instant, and now #42 DOES have a
    successful run record for this period — and the detailed hold still wins,
    unchanged from the block above.

    This is the other half of the gate: the fix fires on an EMPTY detail, never
    on the run record alone. A dependent that ran while its upstream still owes
    the cycle, inside its declared bound, is genuinely waiting for the next
    link's input, and `waiting on #38 (inside its 36 h ...)` is the sentence
    that tells a reader which clock has not run out."""
    owed = _owed_upstream_board(bound=36)
    b = by_id(owed)
    _record(aut, 42, RUNS_COMPLETED_0928[42][0], RUNS_COMPLETED_0928[42][1])
    now = at(aut, monkeypatch, PROBE_0928)
    assert aut._already_ran_this_period(b[42], now=now) == (
        RUNS_COMPLETED_0928[42][0])
    assert reason(aut, b[42], owed, now) == (
        "waiting on #38 (inside its 36 h stale_bypass window: "
        "#38 ran 27.5 h ago)"), "the completed-cycle claim overrode a real hold"
    assert due(aut, b[42], owed, now) is False
