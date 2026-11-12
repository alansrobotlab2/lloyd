"""#4088 — a producer whose run record improves must re-drive its consumer.

The measured instance is 2026-11-11. `#42` (Knowledge Analysis) died at 06:23Z;
`#39` (Knowledge Write) refused the incomplete handoff at 06:44:29Z, and the
refusal recorded itself as a success — `run_39_20261111_064429.md` is
`status: success` / `stop_reason: stop` over a note headed "Verdict: handoff
REFUSED". `#42` retried at 12:44:59Z and completed the SAME handoff at 12:49:46Z
(40,189 bytes, mtime 12:49:29Z) — 10 min 14 s before `#39`'s 09:00-13:00Z window
closed. No `#39` run followed it: the elapsed gate on `last_run` refused the task
for the rest of its period, silently (`grep -ac "inside this task's period"`
across three rotated logs: 0, 0, 0), and `_already_ran_this_period` — the second,
independent account, keyed on that same recorded success — would have refused it
too. The next cycle's `#39` consumed a *different*, freshly-written handoff and
the recovered artifact was read by nobody.

So a link that consumed NOTHING held the chain exactly as long as a link that
consumed the cycle, and the recovery had no owner. Every test here drives the real
due-check (`_is_task_due`, `hold_reason`, `get_due_tasks`) over real task files
and real run records, with the instant, the machine zone and the local hour the
window gate reads pinned together — `#39`'s `[1,2,3,4]` is 09:00-13:00Z only when
the zone is PST, and the whole incident fits inside four minutes of that window.
"""
import datetime as dt
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import autonomy  # noqa: E402

PST = dt.timezone(dt.timedelta(hours=-8))    # America/Los_Angeles in November.
WRITE = [1, 2, 3, 4]                         # #39: 09:00-13:00Z.
ANALYSE = [22, 23, 0, 1, 2, 3, 4]            # #42: 06:00-12:00Z.

# The three instants of the incident, and the tick five minutes after the
# recovery — still inside #39's window, which closes at 13:00:00Z.
REFUSAL = dt.datetime(2026, 11, 11, 6, 44, 29, tzinfo=dt.timezone.utc)
RECOVERY = dt.datetime(2026, 11, 11, 12, 49, 46, tzinfo=dt.timezone.utc)
PROBE = dt.datetime(2026, 11, 11, 12, 55, 0, tzinfo=dt.timezone.utc)
HANDOFF_BYTES = 40_189                        # the orphan's measured size.

#: The tick the module under test believes it is. The window gate reads the local
#: hour from the wall clock, so a test that wants a different instant moves this
#: and every clock moves with it — one instant per reading, the #813 rule.
CLOCK = {"now": PROBE}


def Z(hh, mm=0, ss=0, day=11):
    return dt.datetime(2026, 11, day, hh, mm, ss, tzinfo=dt.timezone.utc)


@pytest.fixture
def aut(tmp_path, monkeypatch):
    """The due-check with its three clocks pinned together, as the incident had them."""
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "autonomy")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(autonomy, "_missing_artifact_bypass_warned", {},
                        raising=False)
    monkeypatch.setattr(autonomy, "_no_input_bypass_warned", {}, raising=False)
    monkeypatch.setattr(autonomy, "_never_ran_bypass_warned", {}, raising=False)
    monkeypatch.setattr(autonomy, "_redrive_traced", {}, raising=False)
    monkeypatch.setattr(autonomy, "_local_tz", lambda: PST)
    monkeypatch.setattr(autonomy, "_utcnow", lambda: CLOCK["now"])
    monkeypatch.setattr(autonomy, "_local_hour",
                        lambda: CLOCK["now"].astimezone(PST).hour)
    autonomy.AUTONOMY_DIR.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n")
    monkeypatch.setattr(autonomy, "_SKILL_FOR_TESTS", str(skill), raising=False)
    return autonomy


def write_task(aut, task_id, *, last_run=None, **fm):
    base = {"id": task_id, "name": f"task{task_id}", "type": "autonomy",
            "status": "up_next", "frequency": "daily", "priority": "high",
            "skill_name": aut._SKILL_FOR_TESTS, "timeout_seconds": 1800,
            "max_retries": 3, "failure_count": 0}
    base.update(fm)
    base["last_run"] = last_run.isoformat() if last_run else ""
    base = {k: v for k, v in base.items() if v is not None}
    (aut.AUTONOMY_DIR / f"{task_id}-task{task_id}.md").write_text(
        f"---\n{yaml.dump(base)}---\n\nbody\n\n## Activity Log\n")


def board(aut, *ids):
    return [aut._parse_task_file(aut._find_task_file(i)) for i in ids]


def record(aut, task_id, run_id, completed_at, status="success"):
    """A run record on disk — the second account, and the one a refusal falsifies."""
    d = aut.AUTONOMY_RUNS_DIR / str(task_id)
    d.mkdir(parents=True, exist_ok=True)
    fm = {"run_id": run_id, "task_id": task_id, "status": status,
          "completed_at": completed_at.isoformat()}
    (d / f"{run_id}.md").write_text(f"---\n{yaml.dump(fm)}---\n")


def handoff(aut, tmpl, day, nbytes=HANDOFF_BYTES):
    p = Path(tmpl.replace("{date}", day))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x" * nbytes)
    return p


def artifact_for(aut, tmpl, when, nbytes=HANDOFF_BYTES):
    """The file the upstream's run at `when` names, written as the writer writes it.

    The name comes out of `_artifact_candidates` rather than from this test's
    opinion, because that resolver — local date first, the other UTC date too, and
    `_upstream_artifact_on_disk` accepting `_ARTIFACT_MIN_BYTES` or more — is the
    thing the re-drive actually asks. A fixture that spelled the date by hand would
    pass by accident when the resolver moved.
    """
    cands = aut._artifact_candidates(tmpl, when)
    assert cands, "fixture: the template must name at least one candidate"
    cands[0].parent.mkdir(parents=True, exist_ok=True)
    cands[0].write_text("x" * nbytes)
    return cands[0]


def _incident(aut, tmp_path, *, consumer_last=REFUSAL, producer_last=RECOVERY,
              handoff_present=True):
    """The #42 → #39 edge as the 2026-11-11 cycle left it, and the two run records.

    Re-calling it rewrites the task files, which is how a test advances a stamp
    the way a completing run does.
    """
    tmpl = str(tmp_path / "reflection" / "knowledge-handoff-{date}.md")
    write_task(aut, 42, preferred_hours=ANALYSE, last_run=producer_last,
               output_artifact=tmpl)
    write_task(aut, 39, preferred_hours=WRITE, depends_on=42,
               last_run=consumer_last, stale_bypass_hours=36,
               output_artifact=str(tmp_path / "reflection"
                                   / "knowledge-write-{date}.md"))
    record(aut, 42, "run_42_20261111_124459", producer_last)
    record(aut, 39, "run_39_20261111_064429", REFUSAL)
    path = None
    if handoff_present:
        path = handoff(aut, tmpl, "2026-11-11")
    return {"board": board(aut, 42, 39), "template": tmpl, "artifact": path}


# ── Clause 1: the improved upstream re-drives the consumer in its window ─────

def test_the_recovered_artifact_redrives_the_consumer_inside_its_window(
        aut, tmp_path):
    """#39 at 12:55Z on 2026-11-11: due, and dispatchable through the real scan.

    The two fixture assertions are the point of the test as much as the last one.
    Without them a green `get_due_tasks` would only prove the gates do not fire
    for some reason nobody meant, and the incident is exactly a case where a gate
    fired for a reason nobody meant.
    """
    f = _incident(aut, tmp_path)
    t39, t42 = f["board"][1], f["board"][0]

    # The elapsed gate IS the refuser at this instant: #39's 06:44:29Z run holds
    # it until the next window (09:00Z 11-12), a day after the artifact expires.
    assert PROBE < aut._elapsed_due_at(t39, REFUSAL), (
        "fixture: the elapsed gate must be refusing #39 here, or this pins nothing")
    # And the second, independent account refuses it too, on the success that
    # consumed nothing (#1296's guard, unchanged by this item).
    assert aut._already_ran_this_period(t39, now=PROBE) == "run_39_20261111_064429"
    # The windows say this recovery IS this cycle's input (`interval / 2` = 12 h
    # would have too, at 5 min old; the windows are the rule the gate applies).
    assert aut._window_dependency_fresh(t39, t42, RECOVERY, PROBE) is True

    assert aut._is_task_due(t39, f["board"], now=PROBE) is True, (
        "the orphaned recovery must re-drive the consumer, not sit unconsumed")
    # The dispatch surface `scheduled_task.py` actually reads, on the same tick.
    due = {int(t["id"]) for t in aut.get_due_tasks(now=PROBE)}
    assert 39 in due, f"#39 not in get_due_tasks: {due}"
    # And the board does not claim the cycle was already consumed (#870).
    assert aut.hold_reason(t39, f["board"], now=PROBE) is None


def test_the_redrive_dispatches_inside_the_window_and_not_outside_it(aut, tmp_path):
    """Clause 1's second half: the re-drive is released into a window, not past one.

    An hour and five minutes after the recovery, and one hour past the close, the
    artifact is exactly as recoverable and #39 exactly as
    eligible, and the answer must flip to No purely because 13:00:00Z passed: a
    re-drive that ignored `preferred_hours` would spend #39's primary-model cycle
    at an hour its own file forbids, which is the #1437 window rule this module took
    two rounds to build and is not this round's to spend.
    """
    f = _incident(aut, tmp_path)
    t39 = f["board"][1]
    assert aut._is_task_due(t39, f["board"], now=PROBE) is True
    CLOCK["now"] = Z(14, 0)          # 13:00:00Z closed; #39's [1,2,3,4] is 09:00-13:00Z.
    try:
        assert aut._is_task_due(t39, f["board"], now=CLOCK["now"]) is False, (
            "an improved upstream must not dispatch #39 outside its window")
    finally:
        CLOCK["now"] = PROBE


def test_no_artifact_on_disk_means_no_redrive(aut, tmp_path):
    """The same instant with the handoff absent: nothing new to consume, no re-run.

    A recovery is only news if there is a recovered artifact. Re-driving a
    consumer onto a missing file spends a primary-model cycle to refuse again —
    the state #1437 built the artifact clause to hold.
    """
    f = _incident(aut, tmp_path, handoff_present=False)
    t39 = f["board"][1]
    assert aut._is_task_due(t39, f["board"], now=PROBE) is False
    assert aut.hold_reason(t39, f["board"], now=PROBE) == autonomy.RUN_SUCCESS_HOLD


# ── Clause 2: bounded by the freshness rule the dependency gate already uses ─

def test_an_upstream_success_older_than_the_freshness_rule_does_not_redrive(
        aut, tmp_path):
    """`interval / 2` is the bound for a windowless pair, and it is the gate's own.

    Neither task declares a window, so `_window_dependency_fresh` cannot answer and
    the freshness rule falls through to this daily dependent's 12 h half-window. The
    dependent declares `stale_bypass_hours: 36` and the upstream's newest success is
    40 h old, which is precisely what the bypass exists for: the DEPENDENCY GATE has
    released this one onto the stale artifact (that is fail-forward, #1437), and the
    re-drive must still be bounded by freshness. An upstream success older than the
    bound is next cycle's input, not a reason to re-run this cycle's consumer —
    otherwise "improved" would mean "newer than me, however old", and the chain
    would re-dispatch onto day-old artifacts, which is the inverted order #1437
    Defect A built this rule to kill.
    """
    tmpl = str(tmp_path / "out" / "thing-{date}.md")
    stale = PROBE - dt.timedelta(hours=40)
    write_task(aut, 60, last_run=stale, output_artifact=tmpl)
    record(aut, 60, "run_60_stale", stale)
    artifact_for(aut, tmpl, stale)
    write_task(aut, 61, depends_on=60, stale_bypass_hours=36,
               last_run=PROBE - dt.timedelta(hours=14))
    record(aut, 61, "run_61_mine", PROBE - dt.timedelta(hours=14))
    tasks = board(aut, 60, 61)
    t61, t60 = tasks[1], tasks[0]

    assert aut._window_dependency_fresh(t61, t60, stale, PROBE) is None, (
        "fixture: this pair must be on the `interval / 2` arm, or the bound "
        "under test is not the one the dependency gate applies")
    assert aut._upstream_output_fresh(t61, t60, stale, PROBE) is False, (
        "fixture: 40 h against a daily dependent's 12 h half-window is stale")
    assert aut._dependency_refusal(t61, tasks, now=PROBE) is None, (
        "fixture: the 36 h bypass must have released the dependency gate at 40 h, "
        "or the run-record hold below is never reached and proves nothing")

    assert aut._consumer_redrive(t61, tasks, PROBE) is None
    assert aut._is_task_due(t61, tasks, now=PROBE) is False
    assert aut.hold_reason(t61, tasks, now=PROBE) == autonomy.RUN_SUCCESS_HOLD, (
        "with the dependency gate released it is the run record that holds, and a "
        "stale upstream must not lift it")

    # Ten hours instead of forty: the same pair, the same instant, the same two
    # run records, and only the upstream's age moved. Now the re-drive fires.
    fresh = PROBE - dt.timedelta(hours=10)
    write_task(aut, 60, last_run=fresh, output_artifact=tmpl)
    record(aut, 60, "run_60_fresh", fresh)
    artifact_for(aut, tmpl, fresh)
    tasks = board(aut, 60, 61)
    assert aut._upstream_output_fresh(tasks[1], tasks[0], fresh, PROBE) is True
    trace = aut._consumer_redrive(tasks[1], tasks, PROBE)
    assert trace is not None and trace["upstream_run_id"] == "run_60_fresh"
    assert aut._is_task_due(tasks[1], tasks, now=PROBE) is True


# ── Clause 3: one re-drive per upstream success ─────────────────────────────

def test_one_redrive_per_upstream_success_even_when_the_rerun_refuses_again(
        aut, tmp_path, caplog):
    """A second tick over an unchanged upstream record must not dispatch a third time.

    The naive shape of this fix is a loop: a consumer that re-runs and refuses
    AGAIN records `status: success` exactly as its predecessor did, so if the
    release were keyed on anything but the two records' order it would re-drive
    forever. This is that shape, replayed — the re-drive fires at 12:55Z, the
    re-driven run refuses again at 12:58Z and records success, and #42's record is
    untouched throughout.
    """
    f = _incident(aut, tmp_path)
    t39 = f["board"][1]
    assert aut._is_task_due(t39, f["board"], now=PROBE) is True

    second = Z(12, 58)
    record(aut, 39, "run_39_20261111_125504", second)
    f = _incident(aut, tmp_path, consumer_last=second)
    t39 = f["board"][1]
    assert aut._consumer_redrive(t39, f["board"], second) is None, (
        "the consumer's own newer success must close the period again")
    assert aut._is_task_due(t39, f["board"], now=second) is False
    assert aut.hold_reason(t39, f["board"], now=second) == autonomy.RUN_SUCCESS_HOLD
    # And the third tick would be refused by `_already_ran_this_period` alone: the
    # #1296 guard is not weakened, it is only released by a NEWER upstream.
    assert aut._already_ran_this_period(t39, now=second) == "run_39_20261111_125504"


# ── Clause 5: the drop leaves one durable trace ─────────────────────────────

def _redrive_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-autonomy" and "RE-DRIVEN" in r.getMessage()]


def test_a_redrive_leaves_one_line_naming_the_link_the_run_and_the_artifact(
        aut, tmp_path, caplog):
    """Zero hits for the in-period refusal across three logs is how #4088 stayed
    invisible; the line has to name all three things needed to reconstruct it."""
    caplog.set_level("INFO", logger="lloyd-autonomy")
    f = _incident(aut, tmp_path)
    t39 = f["board"][1]
    assert aut._is_task_due(t39, f["board"], now=PROBE) is True
    # A second tick on the same improved artifact does not repeat it.
    assert aut._is_task_due(t39, f["board"], now=PROBE) is True
    lines = _redrive_lines(caplog)
    assert len(lines) == 1, lines
    line = lines[0]
    assert "#39" in line, line
    assert "#42" in line, line
    assert "run_42_20261111_124459" in line, line
    assert "knowledge-handoff-2026-11-11.md" in line, line
