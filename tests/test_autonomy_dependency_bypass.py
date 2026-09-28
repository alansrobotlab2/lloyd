"""#1437 — the dependency fail-forward requires the upstream's artifact, and a
run record names who dispatched it.

On 2026-09-24 #39 (knowledge write) was released by its `stale_bypass_hours: 36`
50.7 h after #42 last ran, onto a `knowledge-handoff-{date}.md` that existed at
no spelling on the machine: the gate measured a timestamp and the thing it gates
is a file. And the out-of-window runs of #38/#56 that pinned both nightly chains
could not be attributed, because no run record says who asked for it.

#1551 is the other edge of that fix. A hold is only a hold if it ends. On
2026-09-26 #39 was still refusing to dispatch 4 days after its 36 h bound ran
out, because the `knowledge-handoff-{date}.md` #42 declares went missing with the
2026-09-22 data-home move and nothing writes it back: #1437's fail-closed hold had
quietly become fail-forever, and the bound the task file declared bounded nothing.
Past the bound plus `autonomy.MISSING_ARTIFACT_GRACE_HOURS` (24 h) an absent
artifact now releases the dependent with its OWN warning; up to that horizon the
hold below is unchanged, and an upstream whose artifact IS on disk forwards
exactly as it always did.
"""
import datetime as dt
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import autonomy  # noqa: E402

PIN = dt.datetime(2026, 9, 24, 8, 0, 0, tzinfo=dt.timezone.utc)
UP_LAST = PIN - dt.timedelta(hours=50, minutes=42)      # the measured 50.7 h


@pytest.fixture
def aut(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "autonomy")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(autonomy, "_missing_artifact_bypass_warned", {}, raising=False)
    monkeypatch.setattr(autonomy, "_no_input_bypass_warned", {}, raising=False)
    monkeypatch.setattr(autonomy, "_never_ran_bypass_warned", {})
    autonomy.AUTONOMY_DIR.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n")
    monkeypatch.setattr(autonomy, "_SKILL_FOR_TESTS", str(skill), raising=False)
    monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "sys",
                        raising=False)
    monkeypatch.setattr(autonomy, "_utcnow", lambda: PIN)
    return autonomy


def write_task(aut, task_id, **fm):
    base = {"id": task_id, "name": f"task{task_id}", "type": "autonomy",
            "status": "up_next", "frequency": "daily", "priority": "medium",
            "skill_name": aut._SKILL_FOR_TESTS, "timeout_seconds": 1800,
            "max_retries": 3, "failure_count": 0}
    base.update(fm)
    base = {k: v for k, v in base.items() if v is not None}
    (aut.AUTONOMY_DIR / f"{task_id}-task{task_id}.md").write_text(
        f"---\n{yaml.dump(base)}---\n\nbody\n\n## Activity Log\n")


def board(aut, *ids):
    return [aut._parse_task_file(aut._find_task_file(i)) for i in ids]


def _chain(aut, tmp_path, *, declared=True, up_last=UP_LAST, next_run=None,
           bound=36):
    """The #42 → #39 edge, with #39's `stale_bypass_hours` set to `bound`.

    `bound=None` is #1551 clause 5's dependent: no bound declared, so no
    fail-forward path at all and nothing for the grace horizon to bound.
    """
    tmpl = str(tmp_path / "reflection" / "knowledge-handoff-{date}.md")
    write_task(aut, 42, last_run=up_last.isoformat() if up_last else "",
               next_run=next_run.isoformat() if next_run else None,
               output_artifact=tmpl if declared else None)
    write_task(aut, 39, depends_on=42, stale_bypass_hours=bound,
               last_run=(PIN - dt.timedelta(days=2)).isoformat(),
               next_run=next_run.isoformat() if next_run else None)
    return tmpl


def _write_handoff(tmpl, day, nbytes=5000):
    p = Path(tmpl.replace("{date}", day))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x" * nbytes)
    return p


def _bypass_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-autonomy" and "HELD past its stale_bypass_hours"
            in r.getMessage()]


def _no_input_warnings(caplog):
    """#1551's release line — the one that fires while DISPATCHING, not holding."""
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-autonomy" and "NO input on disk" in r.getMessage()]


def test_a_bypass_onto_a_handoff_that_is_not_on_disk_holds_and_says_so(
        aut, tmp_path, caplog):
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    _chain(aut, tmp_path)
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False, (
        "the fail-forward released #39 onto a handoff nothing ever wrote")
    # #1538 clause 4, the second of the two refusals: the bound HAS run out (50.7 h
    # against 36 h) and something else is holding the dependent. Before #1538 this
    # line printed the same bare `waiting on #42` as the inside-the-window case
    # pinned by `test_inside_the_bypass_window_nothing_changes`, which is how four
    # days of #42 stall alerts could not name the branch (#1538).
    assert aut.hold_reason(tasks[1], tasks, now=PIN) == (
        "waiting on #42 (stale_bypass 36 h passed; "
        "#42's declared output_artifact is not on disk)")
    aut._is_dependency_met(tasks[1], tasks, now=PIN)
    lines = _bypass_warnings(caplog)
    assert len(lines) == 1 and "#42" in lines[0] and "#39" in lines[0], lines


def test_the_same_bypass_releases_once_the_handoff_exists(aut, tmp_path):
    tmpl = _chain(aut, tmp_path)
    _write_handoff(tmpl, UP_LAST.astimezone().strftime("%Y-%m-%d"))
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is True


def test_a_handoff_dated_by_the_run_start_counts(aut, tmp_path):
    """The template is dated when the run STARTS and `last_run` is when it ends,
    so a run crossing local midnight names the day before its completion stamp.

    The upstream sits 48.8 h back (two midnights before `PIN`), not 72.8 h as it
    did until #1551: clause 2 makes an absent artifact stop holding a dependent
    once its `last_run` is more than 24 h past the 36 h bound, so a 3-day-old
    upstream would have released at the `is False` assertion below and the
    midnight spelling would never have been tested. Inside the horizon the held
    verdict that assertion pins is still #1437's, which is what this case is
    about."""
    local_midnight = (PIN - dt.timedelta(days=2)).astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0)
    up_last = (local_midnight + dt.timedelta(minutes=10)).astimezone(dt.timezone.utc)
    tmpl = _chain(aut, tmp_path, up_last=up_last)
    start_day = (up_last - dt.timedelta(seconds=1800)).astimezone().strftime("%Y-%m-%d")
    assert start_day not in {up_last.astimezone().strftime("%Y-%m-%d"),
                             up_last.strftime("%Y-%m-%d")}, "the case must cross midnight"
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False
    _write_handoff(tmpl, start_day)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is True


def test_a_stub_under_the_artifact_floor_is_not_a_handoff(aut, tmp_path):
    tmpl = _chain(aut, tmp_path)
    _write_handoff(tmpl, UP_LAST.astimezone().strftime("%Y-%m-%d"), nbytes=69)
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False


def test_an_upstream_declaring_no_artifact_keeps_the_elapsed_time_rule(aut, tmp_path):
    _chain(aut, tmp_path, declared=False)
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is True


def test_inside_the_bypass_window_nothing_changes(aut, tmp_path):
    """20 h after the upstream: past interval/2, short of 36 h. Held as before,
    and not on the artifact's account (no missing-artifact warning).

    #1538 clause 4, the first of the two refusals: the held reason now says so in
    words, where it used to be the same `waiting on #42` the past-bound case
    printed. The VERDICT is byte-identical to the one above — `_is_dependency_met`
    is False here and there — which is the whole reason the strings were needed."""
    _chain(aut, tmp_path, up_last=PIN - dt.timedelta(hours=20))
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False
    assert autonomy._missing_artifact_bypass_warned == {}
    assert aut.hold_reason(tasks[1], tasks, now=PIN) == (
        "waiting on #42 (inside its 36 h stale_bypass window: #42 ran 20.0 h ago)")


def test_the_two_dependency_refusals_are_distinguishable_in_the_stall_alert(
        aut, tmp_path):
    """#1538 clause 4, the half that reaches a person.

    48 stall alarms naming #42 fired between 2026-09-23 22:44 and 2026-09-26 01:47
    local, each with the right hours and each reading `held: waiting on #38`, and
    none of them could say whether the chain was one run behind (inside the bound,
    releases by itself at the bound) or lost (bound run out, something else still
    refusing). That is why this item had to guess between two candidate branches.

    Drives the real alert route — `_next_run_stalled` builds `hold` from
    `autonomy.hold_reason` and `_nextrun_alert_message` prints it through
    `_hold_note` (both in `workers/fleet_watchdog.py` since #1682) — over one board
    carrying both states at once: #39 behind a #42
    whose 50.7 h silence is past its 36 h bound with the declared handoff absent,
    and #45 behind a #44 that stopped 20.0 h ago, inside its bound."""
    from workers.queue import WorkQueue
    import workers.fleet_watchdog as fw

    two_days = PIN - dt.timedelta(days=2)
    _chain(aut, tmp_path, next_run=two_days)      # #42 → #39, upstream 50.7 h quiet
    tmpl44 = str(tmp_path / "reflection" / "signals-{date}.md")
    write_task(aut, 44, last_run=(PIN - dt.timedelta(hours=20)).isoformat(),
               output_artifact=tmpl44)
    write_task(aut, 45, depends_on=44, stale_bypass_hours=36,
               last_run=two_days.isoformat(), next_run=two_days.isoformat())

    # The two alert functions live in `workers/fleet_watchdog.py` since #1682, which
    # moved the fleet's alarms out of the dispatch source; same bodies, same call.
    stalled = {int(e["id"]): e for e in fw._next_run_stalled(
        WorkQueue(tmp_path / "alert.db"))}
    assert {39, 45} <= set(stalled), (
        "neither dependent reached the alert, so the strings below would pass on "
        f"an empty message: {sorted(stalled)}")
    msg = fw._nextrun_alert_message([stalled[39], stalled[45]])
    assert "#44 ran 20.0 h ago" in msg, msg
    assert "stale_bypass 36 h passed; #42's declared output_artifact is not on disk" in msg, msg
    assert msg.count("waiting on #") == 2, msg


def test_a_never_run_upstream_that_declares_an_artifact_holds(aut, tmp_path):
    _chain(aut, tmp_path, up_last=None)
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False


# ── #1551: the declared bound has to bound the wait ──────────────────────────
#
# `stale_bypass_hours: 36` says "after 36 h of upstream silence, forward on the
# previous cycle's input"; #1437 then added "…but not if the input is absent".
# Both cannot be the whole rule, because an absent artifact that never comes back
# makes the bound unreachable and the dependent waits forever — which is what
# happened to #39 from 2026-09-22 on. `MISSING_ARTIFACT_GRACE_HOURS` (24 h) is the
# horizon where the two stop arguing: up to bound + 24 h the hold is #1437's, past
# it the owner's declared bound wins and the dependent runs on nothing.


def test_the_artifact_hold_still_applies_inside_the_grace_horizon(aut, tmp_path,
                                                                  caplog):
    """Clause 1. #1437's fail-closed hold survives to the END of the grace.

    59.9 h after #42's last run — past #39's 36 h bound by 23.9 h, one tenth of an
    hour short of the 24 h grace — the dependent is still held and the held reason
    still names the artifact as not on disk, byte-for-byte the #1437 sentence
    pinned by `test_a_bypass_onto_a_handoff_that_is_not_on_disk_holds_and_says_so`.
    Nothing on the NO-input side has fired either: a hold that has not reached its
    horizon must not read as a release.
    """
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    _chain(aut, tmp_path, up_last=PIN - dt.timedelta(hours=59, minutes=54))
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False
    assert aut.hold_reason(tasks[1], tasks, now=PIN) == (
        "waiting on #42 (stale_bypass 36 h passed; "
        "#42's declared output_artifact is not on disk)")
    assert autonomy._no_input_bypass_warned == {}, (
        "a dependent inside the grace horizon is held, not forwarding")
    assert _no_input_warnings(caplog) == []


def test_the_hold_releases_only_strictly_past_the_grace_horizon(aut, tmp_path):
    """Clause 2, and the one-second line that defines it.

    At exactly bound + 24 h (60.0 h) the dependent is STILL held — the horizon is
    exceeded, not met, so a case sitting on the boundary is the conservative one.
    One second older and nothing on disk can hold it: the file is as absent as it
    was a second ago, and the only thing that changed is the clock the owner chose
    to bound.
    """
    tmpl = _chain(aut, tmp_path, up_last=PIN - dt.timedelta(hours=60))
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False
    write_task(aut, 42, last_run=(PIN - dt.timedelta(hours=60,
                                                      seconds=1)).isoformat(),
               output_artifact=tmpl)
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is True, (
        "the artifact is exactly as absent as it was one second ago, so what "
        "released the dependent must be the elapsed bound, not the file")


def test_a_dependent_past_the_grace_horizon_dispatches_without_its_input(
        aut, tmp_path):
    """Clause 2 across the seam dispatch actually reads: `get_due_tasks()`.

    Same fixture, one variable — the upstream's age. At 59.9 h (inside the grace)
    #39 is out of the dispatch set; at 61 h it is in it. Asserting only the
    predicate would not show a dependent leaving the set dispatch consumes, and a
    green predicate on a fixture whose dependent was never due would prove nothing
    either, so the held case here is the positive control.
    """
    def _due_ids():
        return {int(t["id"]) for t in aut.get_due_tasks(now=PIN)}

    _chain(aut, tmp_path, up_last=PIN - dt.timedelta(hours=59, minutes=54))
    assert 39 not in _due_ids(), (
        "#39 dispatched inside the grace horizon, so the 61 h assertion below "
        "would pass on a fixture that never held it")
    _chain(aut, tmp_path, up_last=PIN - dt.timedelta(hours=61))
    assert 39 in _due_ids(), (
        "#39 is still absent from the dispatch set 25 h past its declared 36 h "
        "bound: the missing handoff is holding it forever (#1551)")


def test_the_no_input_release_warns_and_is_not_the_hold_warning(aut, tmp_path,
                                                                caplog):
    """Clause 3: a run on nothing can never read as an ordinary stale forward.

    The release line must name the missing artifact, say it is forwarding with NO
    input on disk, and be a DIFFERENT sentence from the #1437 hold — the two
    describe opposite states (will not dispatch / is dispatching) and 33 copies of
    one string could not tell them apart. Deduplicated per (dependent, upstream)
    episode like the hold warning, because dispatch re-answers this every tick.
    """
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    tmpl = _chain(aut, tmp_path, up_last=PIN - dt.timedelta(hours=61))
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is True
    lines = _no_input_warnings(caplog)
    assert len(lines) == 1, lines
    msg = lines[0]
    assert "NO input on disk" in msg, msg
    assert tmpl in msg, f"the warning does not name the missing artifact: {msg}"
    assert "#39" in msg and "#42" in msg, msg
    assert "HELD past its stale_bypass_hours" not in msg, (
        f"the release reused the hold sentence: {msg}")
    assert _bypass_warnings(caplog) == [], msg
    assert autonomy._missing_artifact_bypass_warned == {}, (
        "the episode moved from hold to release, so the hold ledger must clear "
        "and a later regression warns again")
    aut._is_dependency_met(tasks[1], tasks, now=PIN)
    aut._is_dependency_met(tasks[1], tasks, now=PIN)
    assert len(_no_input_warnings(caplog)) == 1, (
        "the release warning is not deduplicated: 1,440 lines a day")


def test_an_upstream_whose_artifact_is_on_disk_still_forwards_past_the_horizon(
        aut, tmp_path, caplog):
    """Clause 4, first half: the present-file case is unchanged by #1551.

    61 h of upstream silence with the handoff on disk is the ordinary stale
    forward #1437 left untouched. It must not pick up the NO-input warning: the
    input IS on disk, and a warning that fires on both states is the conflation
    clause 3 exists to prevent.
    """
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    tmpl = _chain(aut, tmp_path, up_last=PIN - dt.timedelta(hours=61))
    _write_handoff(tmpl, (PIN - dt.timedelta(hours=61)).astimezone()
                   .strftime("%Y-%m-%d"))
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is True
    assert _no_input_warnings(caplog) == []
    assert _bypass_warnings(caplog) == []
    assert autonomy._no_input_bypass_warned == {}


def test_an_upstream_declaring_no_artifact_still_forwards_past_the_horizon(
        aut, tmp_path, caplog):
    """Clause 4, second half: no declared artifact, pure elapsed-time rule.

    The same 61 h gap with #42 declaring nothing releases on the bound alone, as
    #814 and #870 pinned, and says nothing about a missing file — there is no
    declared file to be missing.
    """
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    _chain(aut, tmp_path, declared=False, up_last=PIN - dt.timedelta(hours=61))
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is True
    assert _no_input_warnings(caplog) == []
    assert _bypass_warnings(caplog) == []


def test_a_dependent_declaring_no_bound_is_never_released_by_this_path(
        aut, tmp_path, caplog):
    """Clause 5. No bound declared means no fail-forward at all.

    200 h of upstream silence — more than eight times the horizon any 36 h bound
    would have granted — and #39 stays held in all three artifact states: the
    declared artifact absent, the declared artifact present, and no artifact
    declared at all. The grace horizon releases a dependent that ASKED to forward
    on stale input; it is not a licence for one that never did, and it must not
    print the NO-input warning for a task that was never released.
    """
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    silence = PIN - dt.timedelta(hours=200)
    tmpl = _chain(aut, tmp_path, bound=None, up_last=silence)
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False, (
        "absent artifact: a dependent with no stale_bypass_hours was released")
    _write_handoff(tmpl, silence.astimezone().strftime("%Y-%m-%d"))
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False, (
        "present artifact: a dependent with no stale_bypass_hours was released")
    write_task(aut, 42, last_run=silence.isoformat())      # declares nothing
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False, (
        "undeclared artifact: a dependent with no stale_bypass_hours was released")
    assert autonomy._no_input_bypass_warned == {}
    assert _no_input_warnings(caplog) == []


# ── the trigger field ────────────────────────────────────────────────────────

RESULT = {"type": "result", "stop_reason": "stop",
          "usage": {"input_tokens": 10, "output_tokens": 5}, "num_turns": 1}
TEXT = {"type": "text_delta", "text": "did the thing"}


def _fake_run_query(events):
    async def _rq(messages, options):
        for e in events:
            yield e
    return _rq


def _record(aut, task_id):
    runs = list((aut.AUTONOMY_RUNS_DIR / str(task_id)).glob("run_*.md"))
    assert len(runs) == 1
    return yaml.safe_load(runs[0].read_text().split("---")[1])


@pytest.mark.parametrize("via,expected", [(None, "direct"), ("scheduler", "scheduler"),
                                          ("api", "api")])
async def test_a_run_record_names_its_trigger(aut, monkeypatch, via, expected):
    write_task(aut, 1, timeout_seconds=30)
    monkeypatch.setattr("app.harness.run_query", _fake_run_query([TEXT, RESULT]))
    if via:
        with aut.run_trigger(via):
            result = await aut.run_task(1)
    else:
        result = await aut.run_task(1)
    assert result["success"] is True
    fm = _record(aut, 1)
    assert fm["trigger"] == expected
    assert "in_window" not in fm, "a task with no window has no in/out to report"
    assert aut.RUN_TRIGGER.get() is None, "the trigger leaked out of its block"


async def test_an_out_of_window_run_says_so(aut, monkeypatch):
    monkeypatch.setattr(aut, "_local_hour", lambda: 7)
    write_task(aut, 38, timeout_seconds=30, preferred_hours=[22, 23, 0, 1, 2, 3, 4])
    monkeypatch.setattr("app.harness.run_query", _fake_run_query([TEXT, RESULT]))
    with aut.run_trigger("mcp"):
        await aut.run_task(38)
    fm = _record(aut, 38)
    assert fm["trigger"] == "mcp" and fm["in_window"] is False


async def test_a_failed_run_record_carries_the_trigger_too(aut, monkeypatch):
    write_task(aut, 1, timeout_seconds=1)

    def _slow(events):
        async def _rq(messages, options):
            import asyncio
            await asyncio.sleep(5)
            yield TEXT
        return _rq
    monkeypatch.setattr("app.harness.run_query", _slow([TEXT]))
    with aut.run_trigger("scheduler"):
        result = await aut.run_task(1)
    assert result["success"] is False
    fm = _record(aut, 1)
    assert fm["status"] == "failed" and fm["trigger"] == "scheduler"


def test_every_production_caller_names_itself():
    root = Path(__file__).resolve().parent.parent
    for rel, name in (("workers/sources/scheduled_task.py", "scheduler"),
                      ("app/routers/autonomy.py", "api"),
                      ("agent_mcp/autonomy.py", "mcp")):
        assert f'run_trigger("{name}")' in (root / rel).read_text(), rel
