"""`policy.first_arm_allowed` — the one dispatch move a file lane may land (#3464).

The premise this file exists because of: a round can build a scheduled series end to end and
can never start it. `scripts/automod/vault_round.py`'s create branch still says so ("create
it `status: draft` and let a human, or a granted call, arm it"), and that granted call never
came — eight items (#2219→#96, #2586→#97, #2897→#98, #2797→#99, #2865→#101, #3308, #3458,
#3464) existed to move one line, and every arm that did happen was an attended vault commit,
one of them `~/obsidian` a01fa34f titled "unattributed dirty state".

The exception is the narrowest thing still worth having: the FIRST arm of a task that has
never dispatched and declares no park on purpose. Everything here is measured against that
shape, and the refusals matter as much as the one allowance — a re-arm, an un-park, a cadence
move or an armed CREATE must keep landing exactly as they do today.

Every node aims `LLOYD_DATA` at its own tree, because clause 2 of the item is precisely that
the runs-directory test resolves through the `${LLOYD_DATA:-~/lloyd-data}/autonomy-runs`
accessor rather than the import-frozen `app.paths.AUTONOMY_RUNS_DIR`. A node that read the
frozen constant would see the real production tree no matter what this file set, and would
pass while pinning nothing: `app.paths.DATA_ROOT` resolves a worktree's data root by TREE,
not by the variable, so in a gate worktree the two answers are different directories.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from app.harness import policy as P

#: Vault-relative, the spelling all three lanes hand the predicate.
TASK = "autonomy/100-eval-instrument-index-daily.md"
TASK_ID = "100"

#: A task file's dispatch fields, in the shape the fleet really writes them. The `parked:`
#: key is deliberately absent: `autonomy/100-*.md` and `autonomy/103-*.md` carry none, and
#: the predicate must answer for the missing key rather than assume one.
DRAFT = {"status": "draft", "skill_name": "eval-instrument-index-daily",
         "frequency": "daily", "preferred_hours": "06:30-08:00"}
ARMED = {**DRAFT, "status": "up_next"}


@pytest.fixture(autouse=True)
def _runs_root_is_this_tests(tmp_path, monkeypatch):
    """A runs root this node owns, with task 100 in it: HAS run.

    Started from the state a live armed task actually has (`_write_run_record` drops a
    `run_<id>_<stamp>.md` under `<runs-root>/<id>/` on every dispatch) rather than from an
    empty directory, so a node that wants the allowed answer has to remove the record on
    purpose. A fixture that began empty would make the True side the default and every
    refusal here a test of a condition nobody had to reach.
    """
    runs = tmp_path / "autonomy-runs" / TASK_ID
    runs.mkdir(parents=True)
    (runs / f"run_{TASK_ID}_20261030_080148.md").write_text("---\nstatus: success\n---\n",
                                                            encoding="utf-8")
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))
    return tmp_path


def _never_run() -> None:
    """Delete the task's run record: this arm would be its first dispatch."""
    root = Path(__import__("os").environ["LLOYD_DATA"]) / "autonomy-runs"
    target = root / TASK_ID
    if target.is_dir():
        import shutil
        shutil.rmtree(target)
    assert not target.exists(), "the run record survived, so this is not a FIRST arm"


def _parked(reason: str = "Alan's ruling 2026-09-17: deliberately parked (#68)") -> dict:
    """The same draft file carrying a park, spelled the way the live parked files spell it.

    `autonomy/68-morning-brief-triage.md:18`, `autonomy/98-retag-untagged-facts.md:34` and
    `autonomy/102-find-all-arms-run.md:31` each carry a ruling STRING, which is the form
    `app.autonomy.parked_declaration` rules on: a reason string or `parked: true` is a park,
    a missing key or `parked: false` is not.
    """
    return {**DRAFT, "parked": reason}


# ───────────────────────────── the one allowance ─────────────────────────────

def test_the_first_arm_of_a_never_run_unparked_task_is_allowed():
    """The whole point: draft -> up_next, never dispatched, nobody parked it.

    Clause 1's True side. `status` is the only dispatch field that moved, so
    `schedule_value_moves` reports exactly one move and the predicate takes it.
    """
    _never_run()
    assert P.first_arm_allowed(TASK, DRAFT, ARMED) is True


def test_the_predicate_reads_the_runs_root_from_lloyd_data_not_the_frozen_constant():
    """Clause 2, pinned by moving the directory and watching the answer move with it.

    Both sides in one node on purpose: create the record, expect False; remove it, expect
    True. A node that only checked the True side would also pass on a predicate that ignored
    the runs root entirely, and one that only checked False would pass on a predicate that
    always refused. It is also the node that fails if the root is taken from
    `app.paths.AUTONOMY_RUNS_DIR`: that constant is frozen at import and, in a worktree,
    resolves by TREE rather than by `LLOYD_DATA`, so the directory this node moved would be
    a directory the predicate never looked at.
    """
    assert P.first_arm_allowed(TASK, DRAFT, ARMED) is False, "has a run record already"
    _never_run()
    assert P.first_arm_allowed(TASK, DRAFT, ARMED) is True


# ─────────────────────────── what stays refused ───────────────────────────

def test_parking_a_task_stays_refused_because_a_park_is_a_ruling():
    """`up_next -> draft` is not an arm — it is the opposite move, and it stays human-owned.

    The predicate is for a task nobody has ever started. Turning a live task off has a
    different owner (the ruling that started it) and a different audit trail, and #3464
    deliberately does not touch it.
    """
    _never_run()   # even a never-run task's downgrade is not this allowance
    assert P.first_arm_allowed(TASK, ARMED, DRAFT) is False


@pytest.mark.parametrize("parked", [
    pytest.param("Alan's ruling 2026-09-17: deliberately parked (#68)", id="a-ruling-string"),
    pytest.param(True, id="parked-colon-true"),
], ids=str)
def test_a_parked_task_is_never_first_armable(parked):
    """The #68 case: a file that DECLARES a park, under the `parked_declaration` rule.

    `parked: <reason>` and `parked: true` are both parks to the rule that already governs
    the fleet; `autonomy/98-*.md:34` and `autonomy/102-*.md:31` are the live refusals of the
    first form. A predicate reading raw key presence instead would get BOTH wrong — it would
    refuse `parked: false`, which the rule says is not a park, and allow `parked: null`,
    which is why those two live files' wording matters.
    """
    _never_run()
    assert P.first_arm_allowed(TASK, DRAFT, {**ARMED, "parked": parked}) is False


def test_parked_in_the_head_bytes_cannot_be_deleted_by_the_arming_write():
    """The old side is read too, or one write could clear the ruling and arm over it.

    The attack this half of the rule exists for: a file whose HEAD carries
    `parked: <ruling>` and whose new bytes simply omit the key. Nothing in the NEW front
    matter mentions a park, so a predicate that read only the new map would arm a task Alan
    parked by name.
    """
    _never_run()
    assert P.first_arm_allowed(TASK, _parked(), ARMED) is False


@pytest.mark.parametrize("field,value", [
    pytest.param("frequency", "weekly", id="cadence"),
    pytest.param("preferred_hours", "07:00-07:30", id="window"),
    pytest.param("skill_name", "some-other-skill", id="skill"),
    pytest.param("depends_on", 42, id="dependency"),
    pytest.param("auto_advance", "in_progress", id="auto-advance"),
    pytest.param("scheduled_at", "2026-11-01T06:30:00", id="one-shot-time"),
])
def test_an_arm_that_arrives_with_a_cadence_field_moves_is_refused(field, value):
    """The allowance is one field wide, and it does not split a mixed write.

    A write that moves `status` AND `frequency` is not "an arm plus something harmless":
    under #2670's ruling the window and the cadence are the half the fleet is measured on,
    so the write is refused whole — including its status half — exactly as it is today.
    """
    _never_run()
    assert P.first_arm_allowed(TASK, DRAFT, {**ARMED, field: value}) is False


@pytest.mark.parametrize("bad", [
    pytest.param("backlog/100-eval-instrument-index-daily.md", id="a-backlog-item-file"),
    pytest.param("knowledge/autonomy-notes.md", id="a-knowledge-note"),
    pytest.param("autonomy/skill-less-task.md", id="a-task-file-without-a-numeric-id"),
    pytest.param("memory/2026-10-30.md", id="a-daily-note"),
])
def test_a_path_that_is_not_an_autonomy_task_file_is_never_first_armable(bad):
    """The rule is about task files, not about a `status` field that happens to move.

    `backlog/<n>-*.md` is the one that could actually be reached by accident: its basename is
    the same `<digits>-<slug>.md` shape, and its own `status` field (`up_next`, `done`) is
    written by this very loop. It is a kanban state read by the backlog tools, not a dispatch
    field, and the required `autonomy/` segment is what keeps the two apart.
    """
    _never_run()
    assert P.first_arm_allowed(bad, DRAFT, ARMED) is False


def test_an_unreadable_front_matter_map_is_not_an_arm():
    """A rail that cannot see the values it is protecting declines rather than waves through.

    Both lanes already treat a `None` front-matter map as a refusal; this pins that the
    predicate does not become the hole in that rule.
    """
    _never_run()
    assert P.first_arm_allowed(TASK, None, ARMED) is False
    assert P.first_arm_allowed(TASK, DRAFT, None) is False


def test_a_missing_status_or_an_armed_create_is_not_this_shape():
    """`draft -> up_next` names both ends; `absent -> up_next` names neither.

    The create branch is untouched on purpose (`policy.create_arms_a_task`, and vault commit
    e66bc6e1, which records that rail refusing 101's own create). On a create there is no
    HEAD baseline and no runs directory can exist, so both guards here would be vacuously
    true and any task file a round invented would arm itself. The lane keeps that create
    refusal in its own branch; this node pins that the predicate cannot be asked to bless it
    by handing it an empty old map.
    """
    _never_run()
    assert P.first_arm_allowed(TASK, {}, ARMED) is False
    assert P.first_arm_allowed(TASK, {"frequency": "daily"}, ARMED) is False


def test_a_run_record_that_cannot_be_listed_is_answered_as_refused(tmp_path, monkeypatch):
    """A rail that cannot see whether the task has run cannot call this the FIRST arm.

    `LLOYD_DATA` aimed at a FILE rather than a directory is the cheapest honest way to make
    the runs root unlistable: `scandir` answers `NotADirectoryError`, and the predicate's
    two answers are "no directory at all" (a box where nothing has ever run — allow) and
    "cannot tell" (refuse). Those are different, and collapsing them would turn a broken
    data root into a free pass.
    """
    _never_run()
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setenv("LLOYD_DATA", str(blocker))
    assert P.first_arm_allowed(TASK, DRAFT, ARMED) is False


def test_a_missing_runs_root_is_answered_as_never_run(tmp_path, monkeypatch):
    """The other side of the same seam: a root that is simply absent means nothing has run.

    Refusing here would deny the first arm forever on a box whose runs tree has not been
    created yet — and the sweeper deletes old run-record directories, so "absent" is a state
    a real machine is in, not a hypothetical.
    """
    empty = tmp_path / "fresh-data-root"
    empty.mkdir()
    monkeypatch.setenv("LLOYD_DATA", str(empty))
    assert P.first_arm_allowed(TASK, DRAFT, ARMED) is True


def test_the_allowance_stamps_what_it_permitted():
    """Every arm this route permits names itself, because the arms that happened did not.

    `~/obsidian` a01fa34f is titled "autonomy-data-pipeline: pre-flight (unattributed dirty
    state)" and it is one of the commits that armed 100 and 103. The predicate's True side
    therefore logs the id, the path, the clock it was given or the clock it read, and the
    reasons that made it True — the sentence a later reader needs and no vault commit carried.

    Captured on the module's OWN logger rather than with `caplog`, because `app.harness.policy`
    sets `logger.propagate = False` on purpose (its records reach the root handler through its
    own, and a root-level capture would silently see nothing — a test that asserted on an empty
    capture would read as "the stamp is missing" in one direction and as "nothing to check" in
    the other).
    """
    import logging

    seen: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    handler = _Collect()
    P.logger.addHandler(handler)
    previous_level = P.logger.level
    P.logger.setLevel(logging.INFO)
    try:
        _never_run()
        stamp = dt.datetime(2026, 10, 30, 14, 0, tzinfo=dt.timezone.utc)
        assert P.first_arm_allowed(TASK, DRAFT, ARMED, stamp) is True
    finally:
        P.logger.removeHandler(handler)
        P.logger.setLevel(previous_level)

    lines = [m for m in seen if "first arm" in m]
    assert len(lines) == 1, seen
    assert TASK_ID in lines[0] and TASK in lines[0], lines[0]
    assert "2026-10-30T14:00:00+00:00" in lines[0], lines[0]
    assert "never run" in lines[0] and "no park" in lines[0], lines[0]


def test_the_predicate_does_not_decide_anything_else_about_the_file():
    """A file that is not moving a dispatch field at all is not this predicate's business.

    `first_arm_allowed` answers one question. A body-only write stays the lane's
    `_autonomy_body_*` check's job, and a node that let this helper be asked about no-move
    cases and answer True would make the lane's body check skippable. Answering False for a
    no-move pair is what keeps the two checks in series rather than in each other's place.
    """
    _never_run()
    assert P.first_arm_allowed(TASK, DRAFT, dict(DRAFT)) is False
    assert P.first_arm_allowed(TASK, ARMED, dict(ARMED)) is False
