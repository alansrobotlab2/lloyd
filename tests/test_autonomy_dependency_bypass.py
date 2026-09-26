"""#1437 — the dependency fail-forward requires the upstream's artifact, and a
run record names who dispatched it.

On 2026-09-24 #39 (knowledge write) was released by its `stale_bypass_hours: 36`
50.7 h after #42 last ran, onto a `knowledge-handoff-{date}.md` that existed at
no spelling on the machine: the gate measured a timestamp and the thing it gates
is a file. And the out-of-window runs of #38/#56 that pinned both nightly chains
could not be attributed, because no run record says who asked for it.
"""
import datetime as dt
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autonomy  # noqa: E402

PIN = dt.datetime(2026, 9, 24, 8, 0, 0, tzinfo=dt.timezone.utc)
UP_LAST = PIN - dt.timedelta(hours=50, minutes=42)      # the measured 50.7 h


@pytest.fixture
def aut(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "autonomy")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(autonomy, "_missing_artifact_bypass_warned", {}, raising=False)
    monkeypatch.setattr(autonomy, "_never_ran_bypass_warned", {})
    autonomy.AUTONOMY_DIR.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n")
    monkeypatch.setattr(autonomy, "_SKILL_FOR_TESTS", str(skill), raising=False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "sys",
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


def _chain(aut, tmp_path, *, declared=True, up_last=UP_LAST, next_run=None):
    tmpl = str(tmp_path / "reflection" / "knowledge-handoff-{date}.md")
    write_task(aut, 42, last_run=up_last.isoformat() if up_last else "",
               next_run=next_run.isoformat() if next_run else None,
               output_artifact=tmpl if declared else None)
    write_task(aut, 39, depends_on=42, stale_bypass_hours=36,
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
    so a run crossing local midnight names the day before its completion stamp."""
    local_midnight = (PIN - dt.timedelta(days=3)).astimezone().replace(
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
    `_hold_note` — over one board carrying both states at once: #39 behind a #42
    whose 50.7 h silence is past its 36 h bound with the declared handoff absent,
    and #45 behind a #44 that stopped 20.0 h ago, inside its bound."""
    from workers.queue import WorkQueue
    import workers.sources.scheduled_task as st

    two_days = PIN - dt.timedelta(days=2)
    _chain(aut, tmp_path, next_run=two_days)      # #42 → #39, upstream 50.7 h quiet
    tmpl44 = str(tmp_path / "reflection" / "signals-{date}.md")
    write_task(aut, 44, last_run=(PIN - dt.timedelta(hours=20)).isoformat(),
               output_artifact=tmpl44)
    write_task(aut, 45, depends_on=44, stale_bypass_hours=36,
               last_run=two_days.isoformat(), next_run=two_days.isoformat())

    stalled = {int(e["id"]): e for e in st._next_run_stalled(
        WorkQueue(tmp_path / "alert.db"))}
    assert {39, 45} <= set(stalled), (
        "neither dependent reached the alert, so the strings below would pass on "
        f"an empty message: {sorted(stalled)}")
    msg = st._nextrun_alert_message([stalled[39], stalled[45]])
    assert "#44 ran 20.0 h ago" in msg, msg
    assert "stale_bypass 36 h passed; #42's declared output_artifact is not on disk" in msg, msg
    assert msg.count("waiting on #") == 2, msg


def test_a_never_run_upstream_that_declares_an_artifact_holds(aut, tmp_path):
    _chain(aut, tmp_path, up_last=None)
    tasks = board(aut, 42, 39)
    assert aut._is_dependency_met(tasks[1], tasks, now=PIN) is False


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
