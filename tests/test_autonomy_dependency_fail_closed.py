"""#558: an upstream that cannot produce the artifact holds its dependents.

`_is_dependency_met` ends in a freshness rule, and before this round it began
with a lookup that answered "met" when it found nothing: `if not dep_task:
return True`. The lookup's input is the resolution set, so the answer to "may
this chain run?" depended on whether the caller's list happened to contain the
upstream — and on 2026-09-08 that was the whole nightly chain: #42 sat `paused`
(auto-disabled 09-04, re-armed 06:04Z), #39 and #40 dispatched at 06:00-06:03Z
against a vacuously-satisfied gate, and #42's handoff was written at 06:10Z. The
analysis those two runs consumed did not exist yet.

#870 later made `paused`/`draft` upstreams resolve at all (one resolution set for
dispatch and the board), which is why clauses 1-2 are asserted here rather than
implemented: they pin that the resolution set is now consulted AND that the
gate reads the status it found. What this round adds is the other half — an
upstream that resolves to nothing, or to a task the scheduler will not run, is
NOT met, and the escape hatch stays bounded.

Alan's rulings on the two open cases, recorded when the item was reopened
(2026-09-13), are what these tests encode:

* `depends_on` naming no task file HOLDS the dependent and warns (clause 3).
  `stale_bypass_hours` still escapes.
* A `failed` upstream with a fresh-enough last success is ALLOWED (clause 6,
  amended). Exhausting a retry budget stops a task from running again; it does
  not unmake the artifact its last run produced. `test_a_failed_upstream_with_a_
  fresh_success_still_satisfies_its_dependent` pins that permission, and
  `test_failed_upstream_does_not_satisfy_a_dependent` (test_autonomy_scheduler.py)
  pins the stale case, so the two cannot be conflated by a future edit.

Every test here drives the live path — `get_due_tasks()`, the function dispatch
actually calls — and not just the predicate, because the predicate alone cannot
show a dependent leaving the dispatch set. Each one also carries its positive
control: the same fixture with an `up_next` upstream, which must still dispatch.
Without that, an `assert 2 not in due` is green for a fixture that never had a
dependent in the first place (the zero-denominator failure #443/#524 filed).
"""
import datetime as dt
import logging
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autonomy  # noqa: E402

# One pinned instant for every case, through the single clock indirection #813
# introduced. The freshness bound is `interval / 2`, so a case near it is only
# reproducible if the evaluation instant is not the wall clock.
PIN = dt.datetime(2026, 9, 19, 12, 0, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def aut(tmp_path, monkeypatch):
    """Isolated task dir + runs dir, with the clock pinned.

    `_fail_closed_found` is module state that makes the fail-closed warning fire
    once per (dependent, upstream, what-was-found) instead of on every 60 s
    dispatch tick — so a test asserting the warning exists would otherwise depend
    on whether some earlier test in the same process had already emitted the
    identical one. An isolated board gets an isolated warning ledger.
    """
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "autonomy")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    autonomy.AUTONOMY_DIR.mkdir()
    monkeypatch.setattr(autonomy, "_fail_closed_found", {}, raising=False)
    monkeypatch.setattr(autonomy, "_utcnow", lambda: PIN)
    monkeypatch.setattr("prompt_builder.build_system_prompt",
                        lambda **_kw: "sys", raising=False)
    return autonomy


def write_task(aut, task_id, **fm):
    base = {
        "id": task_id, "name": f"task{task_id}", "type": "autonomy",
        "status": "up_next", "frequency": "daily", "priority": "medium",
        "skill_name": "some-skill", "timeout_seconds": 600,
        "max_retries": 3, "failure_count": 0,
    }
    base.update(fm)
    base = {k: v for k, v in base.items() if v is not None}
    path = aut.AUTONOMY_DIR / f"{task_id}-task{task_id}.md"
    path.write_text(f"---\n{yaml.dump(base)}---\n\nbody\n\n## Activity Log\n")
    return path


def _age(days=0, hours=0):
    return (PIN - dt.timedelta(days=days, hours=hours)).isoformat()


def _due_ids(aut):
    return {int(t["id"]) for t in aut.get_due_tasks(now=PIN)}


def _chain(aut, up_over=None, dep_over=None, dep_id=2, up_id=1, dep_on=None):
    """Upstream #1 ran 2 days ago (out of #2's 12 h half-interval, inside the
    36 h bypass window), dependent #2 is 3 days past its own interval and waits
    on it. Returns (upstream, dependent) as parsed off disk.

    `dep_on` overrides what #2 depends on; pass 9999 for the no-file case.
    """
    write_task(aut, up_id, last_run=_age(days=2), **(up_over or {}))
    write_task(aut, dep_id, depends_on=dep_on or up_id, last_run=_age(days=3),
               **(dep_over or {}))
    return (_parse(aut, up_id), _parse(aut, dep_id))


def _parse(aut, task_id):
    path = aut._find_task_file(task_id)
    return aut._parse_task_file(path) if path else None


# ── clauses 1-3: the three shapes that must hold ─────────────────────────────


@pytest.mark.parametrize("up_status", ["paused", "draft"],
                         ids=["paused", "draft_archive"])
def test_a_parked_upstream_holds_its_dependent_out_of_get_due_tasks(
        aut, up_status):
    """Clauses 1 and 2. Upstream parked with a `last_run` older than half the
    dependent's interval, no `stale_bypass_hours`: `get_due_tasks()` excludes the
    dependent.

    Asserted as written in the clauses even though the literal setup was already
    green at base: with the upstream 2 days stale, HEAD's freshness bound
    (`interval / 2` = 12 h) holds the dependent whether or not the status means
    anything, so #870 alone satisfied these two lines. `test_a_parked_upstream_
    holds_the_dependent_even_with_a_fresh_success` is the case where only the
    status can hold it. Both are pinned so a later edit cannot satisfy the clause
    by leaning on the freshness bound.

    Positive control in each: the same fixture with a FRESH `up_next` upstream
    puts #2 back in the set, so the exclusion cannot be an artefact of the
    fixture — a dependent that was never due, a bad skill_name, the hour window.
    The ONLY difference is the upstream's status and its age.
    """
    up, dep = _chain(aut, up_over={"status": up_status})

    assert aut._is_dependency_met(dep, [up, dep], now=PIN) is False
    assert 2 not in _due_ids(aut)
    assert aut.hold_reason(dep, [up, dep], now=PIN) == "waiting on #1"

    write_task(aut, 1, status="up_next", last_run=_age(hours=6))
    up_live = _parse(aut, 1)
    assert aut._is_dependency_met(dep, [up_live, dep], now=PIN) is True
    assert 2 in _due_ids(aut), (
        "control failed: the dependent does not dispatch even with a fresh "
        "up_next upstream, so the assertions above prove nothing about status")


@pytest.mark.parametrize("up_status", ["paused", "draft"],
                         ids=["paused", "draft_archive"])
def test_a_parked_upstream_holds_the_dependent_even_with_a_fresh_success(
        aut, up_status):
    """The half the freshness gate cannot do the work of.

    With a FRESH parked upstream the freshness rule alone would answer "met", so
    this is the case where the status check is the only thing holding the chain —
    which is what makes clauses 1-2 a rule about status and not a restatement of
    the freshness bound.
    """
    _, dep = _chain(aut, up_over={"status": up_status})
    write_task(aut, 1, status=up_status, last_run=_age(hours=6))
    up = _parse(aut, 1)

    assert aut._is_dependency_met(dep, [up, dep], now=PIN) is False
    assert 2 not in _due_ids(aut)
    # Control: same fixture, upstream `up_next`, dispatches.
    write_task(aut, 1, status="up_next", last_run=_age(hours=6))
    assert 2 in _due_ids(aut)


def test_a_depends_on_naming_no_task_file_holds_the_dependent(aut):
    """Clause 3. `depends_on: 9999` with no `9999-*.md` on disk: NOT met, and
    the dependent is out of `get_due_tasks()`.

    This is the case #870 deliberately left where it found it — its docstring
    still says the function is "deliberately NOT a decision about what an absent
    upstream means" — because the ruling on it was Alan's, not the round's. The
    ruling landed 2026-09-13: fail closed. A typo'd id is the same shape as a
    deleted upstream, and under the old rule both silently unblocked every
    dependent of a task that does not exist.

    Control: the same dependent pointed at a real, fresh #1 dispatches.
    """
    write_task(aut, 1, last_run=_age(hours=6))          # real, fresh upstream
    write_task(aut, 2, depends_on=9999, last_run=_age(days=3))
    dep = _parse(aut, 2)
    board = list(aut.dependency_resolution_set())

    assert aut._is_dependency_met(dep, board, now=PIN) is False
    assert 2 not in _due_ids(aut)
    assert aut.hold_reason(dep, board, now=PIN) == "waiting on #9999"

    write_task(aut, 2, depends_on=1, last_run=_age(days=3))
    dep_real = _parse(aut, 2)
    assert aut._is_dependency_met(dep_real, board, now=PIN) is True
    assert 2 in _due_ids(aut), (
        "control failed: the dependent never dispatches even with a fresh real "
        "upstream, so the fail-closed assertion above proves nothing")


def test_an_unparseable_upstream_file_holds_the_dependent(aut):
    """The sharper edge of clause 3: the file EXISTS but the board's parse
    dropped it, so the id resolves to nothing all the same.

    `_all_board_tasks` skips a file whose front matter cannot be recovered, and
    the dashboard test `test_a_dependency_hidden_by_broken_front_matter_is_still_
    a_hold` exists because that drop once read as a satisfied dependency on one
    surface. Under this round's rule the dependent is held on every surface,
    which is the point: the resolution set is what exists.
    """
    write_task(aut, 1, last_run=_age(hours=6))
    path = aut._find_task_file(1)
    path.write_text("not front matter at all, no fences\n")
    write_task(aut, 2, depends_on=1, last_run=_age(days=3))
    dep = _parse(aut, 2)
    board = list(aut.dependency_resolution_set())

    assert board and not any(str(t.get("id")) == "1" for t in board), (
        "the fixture lost the upstream to the parse, so this is the same "
        "assertion as the missing-id case rather than the unparseable one")
    assert aut._is_dependency_met(dep, board, now=PIN) is False
    assert 2 not in _due_ids(aut)


# ── clause 4: the hold has to say what it found ──────────────────────────────


def test_the_fail_closed_hold_warns_naming_the_upstream_and_what_was_found(
        aut, caplog):
    """Clause 4. The warning names the upstream id AND what was found: its
    status for a parked task, the absence of a task file for a dangling id.

    A hold with no reason is the #68 failure mode one layer up — a dependent that
    silently never runs is exactly as hard to diagnose as an upstream silently
    parked for 30 h. Asserted on the real logger (`lloyd-autonomy`) at WARNING,
    and asserted to contain the id and the finding, not just "something was
    logged", so a generic message cannot satisfy it.
    """
    with caplog.at_level(logging.WARNING, logger="lloyd-autonomy"):
        # (a) dangling id
        write_task(aut, 2, depends_on=9999, last_run=_age(days=3))
        dep = _parse(aut, 2)
        board = list(aut.dependency_resolution_set())
        assert aut._is_dependency_met(dep, board, now=PIN) is False
        dangling = [r.getMessage() for r in caplog.records
                    if r.levelno >= logging.WARNING and "9999" in r.getMessage()]
        assert dangling, (
            f"no warning named the unresolved upstream #9999; records: "
            f"{[r.getMessage() for r in caplog.records]}")
        assert any(("no task file" in m) or ("does not exist" in m)
                   for m in dangling), (
            f"the warning did not say that no task file was found: {dangling}")

        # (b) parked upstream: the status has to be in the message
        caplog.clear()
        up, dep = _chain(aut, up_over={"status": "paused"})
        assert aut._is_dependency_met(dep, [up, dep], now=PIN) is False
        parked = [r.getMessage() for r in caplog.records
                  if r.levelno >= logging.WARNING and " #1" in r.getMessage()]
        assert parked, (
            f"no warning named upstream #1; records: "
            f"{[r.getMessage() for r in caplog.records]}")
        assert any("paused" in m for m in parked), (
            f"the warning did not name the upstream's status: {parked}")

        # (c) a satisfied dependency must NOT warn, or the log becomes noise
        caplog.clear()
        write_task(aut, 1, status="up_next", last_run=_age(hours=6))
        up_fresh = _parse(aut, 1)
        assert aut._is_dependency_met(dep, [up_fresh, dep], now=PIN) is True
        assert not [r for r in caplog.records
                    if r.levelno >= logging.WARNING], (
            "a healthy dependency warns, which makes the fail-closed warning "
            "unfindable in the log: "
            f"{[r.getMessage() for r in caplog.records]}")


# ── clause 5: the escape hatch stays, and stays bounded ──────────────────────


def test_stale_bypass_hours_still_dispatches_past_an_unresolved_upstream(aut):
    """Clause 5, first half. A dependent with `stale_bypass_hours: 36` against a
    `paused` upstream whose last success is 2 days old still dispatches.

    This is the exact line the triage probe printed TRUE for and the item says
    must NOT flip: fail closed must not become fail forever. With no upstream
    artifact coming, the declared window is the only signal a chain has that
    waiting is still worth it, so past the window the dependent runs on stale
    input, which is principle 3 of this scheduler.
    """
    up, dep = _chain(aut, up_over={"status": "paused"},
                     dep_over={"stale_bypass_hours": 36})
    assert aut._is_dependency_met(dep, [up, dep], now=PIN) is True
    assert 2 in _due_ids(aut)

    # Bounded, not absent: inside the window (upstream ran 20 h ago, under the
    # 36 h bypass) the same paused upstream still holds it.
    write_task(aut, 1, status="paused", last_run=_age(hours=20))
    up_in = _parse(aut, 1)
    assert aut._is_dependency_met(dep, [up_in, dep], now=PIN) is False
    assert 2 not in _due_ids(aut)


def test_stale_bypass_hours_still_dispatches_past_a_missing_upstream(aut):
    """Clause 5, second half: the dangling-id case with the escape hatch on.

    An absent upstream has no `last_run` to measure a window against, so the
    declared bypass is the whole signal and the dependent forwards. Asserted
    explicitly because it is the one place this round trades a hold for a
    dispatch, and a future edit that made "absent" unforgivable would silently
    stop every bypass-carrying dependent in the fleet.
    """
    write_task(aut, 2, depends_on=9999, last_run=_age(days=3),
               stale_bypass_hours=36)
    dep = _parse(aut, 2)
    board = list(aut.dependency_resolution_set())

    assert aut._is_dependency_met(dep, board, now=PIN) is True
    assert 2 in _due_ids(aut)


def test_the_bypass_still_waits_for_an_upstream_that_is_running(aut):
    """Clause 5, third half. `in_progress` + bypass window elapsed still waits.

    Re-asserted here rather than trusted from
    `test_stale_bypass_waits_for_a_running_upstream` because this round moves the
    call into a new branch: the fail-closed path is where a refactor would drop
    the `in_progress` guard and turn "the upstream is still working" into a
    licence to run on its half-written output.
    """
    up, dep = _chain(aut, up_over={"status": "in_progress"},
                     dep_over={"stale_bypass_hours": 36})
    write_task(aut, 1, status="in_progress", last_run=_age(days=2),
               last_attempt=_age(hours=1))
    up = _parse(aut, 1)
    assert aut._is_dependency_met(dep, [up, dep], now=PIN) is False
    assert 2 not in _due_ids(aut)


# ── clause 6: a task dropped from the runnable set holds its dependents ──────


def test_an_unreadable_grants_block_holds_the_dependent(aut):
    """Clause 6, the half the status filter cannot express.

    #534 drops a task whose `grants:` block will not parse from the runnable set
    — the block IS the human's authorisation, so running it under an authority
    nobody wrote is fail-open with a log line attached. The same block must stop
    anything running OFF that task: an upstream that is not allowed to execute
    cannot certify the artifact its dependent is about to consume.

    Control: the identical chain with a valid grants block dispatches, so the
    hold is about the block being unreadable and not about `grants:` existing.
    """
    bad = [{"tool": "Bash", "predicate": "scope:x", "bogus_key": 1}]
    write_task(aut, 1, last_run=_age(hours=6), grants=bad)
    write_task(aut, 2, depends_on=1, last_run=_age(days=3))
    up, dep = _parse(aut, 1), _parse(aut, 2)
    board = list(aut.dependency_resolution_set())

    assert aut._grant_block_errors(up, Path(up["_path"])), (
        "the fixture's grants block parses, so this proves nothing")
    assert aut._is_dependency_met(dep, board, now=PIN) is False
    assert 2 not in _due_ids(aut)

    # Control 1: no grants block at all → dispatches (the block is not the hold).
    write_task(aut, 1, last_run=_age(hours=6))
    assert 2 in _due_ids(aut), (
        "control failed: a chain with no grants block does not dispatch, so the "
        "hold above is not about the unreadable block")
    # Control 2: a VALID grants block → dispatches (#534's rule is about the
    # block being unreadable, not about a task having authorisation).
    write_task(aut, 1, last_run=_age(hours=6),
               grants=[{"tool": "Bash", "predicate": "len(output)<=1000",
                        "expires_at": (PIN + dt.timedelta(days=30)).isoformat(),
                        "issued_by": "Alan"}])
    assert not aut._grant_block_errors(_parse(aut, 1),
                                       Path(aut.AUTONOMY_DIR / "1-task1.md")), (
        "control 2's grants block does not itself validate, so the assertion "
        "below is not testing a valid block")
    assert 2 in _due_ids(aut), (
        "a valid grants block holds its dependent: the fail-closed rule is "
        "reading `grants` as present rather than unreadable, which would stop "
        "every authorised task in the fleet from having dependents")


def test_a_failed_upstream_with_a_fresh_success_still_satisfies_its_dependent(
        aut):
    """Clause 6 as amended by Alan's ruling: `failed` is ALLOWED.

    `failed` is the one status inside the runnable set, and the two clauses pull
    in opposite directions, so the boundary is stated as a test. A task whose
    retry budget ran out cannot run again — that is what `failed` means — but its
    last SUCCESS still names a real artifact. What governs whether that artifact
    is usable is the freshness rule, not the status word:
    `test_failed_upstream_does_not_satisfy_a_dependent` pins the stale case (2
    days old, out of the half-interval) and this pins the fresh case. An edit
    that added `failed` to the disallowed set would break THIS test, which is the
    point of writing it in the opposite direction from clause 6's headline.
    """
    _, dep = _chain(aut, up_over={"status": "failed", "failure_count": 3})
    write_task(aut, 1, status="failed", failure_count=3,
               last_run=_age(hours=6), last_attempt=_age(hours=1))
    up = _parse(aut, 1)

    assert aut._is_dependency_met(dep, [up, dep], now=PIN) is True
    assert 2 in _due_ids(aut)
    # Freshness still binds: the same failed upstream 2 days stale is NOT met.
    write_task(aut, 1, status="failed", failure_count=3,
               last_run=_age(days=2), last_attempt=_age(hours=1))
    up_stale = _parse(aut, 1)
    assert aut._is_dependency_met(dep, [up_stale, dep], now=PIN) is False


def test_the_rung_that_decides_a_dependent_is_the_rung_that_decides_a_run(aut):
    """The structural claim underneath clauses 1-3 and 6, asserted as a set.

    "May this task run?" and "may this task certify its dependent's input?" are
    one question with one answer, so they must not have two status lists that can
    drift — the drift IS the 2026-09-08 incident. Dispatch enumerates statuses in
    `_all_runnable_tasks`; the dependency rule is asserted to accept exactly the
    statuses that filter admits, plus the grants predicate it applies. `failed`
    is in the set (dispatch admits it) but is excluded from dispatch by
    `_is_task_due`'s own status gate — the one asymmetry in the design, and
    asserted here so it is visible rather than implied.

    Which statuses dispatch admits is established by RUNNING
    `_all_runnable_tasks` and reading back the tasks it returns, not by
    re-testing a status tuple inside this test. An earlier version of this node
    carried `if status in ("up_next", "in_progress", "failed"): admitted.add(status)`
    and then asserted `admitted == {"up_next", "in_progress", "failed"}`: the set
    was filled by its own membership test, so the drift the docstring names — a
    status added to dispatch's filter and not to the gate's — passed it green. A
    pin that cannot fail is worse than no pin; it reads as coverage.
    """
    statuses = ["up_next", "in_progress", "failed", "paused", "draft",
                "suspended"]
    # One board, one task per status, fed to the real dispatch filter. `admitted`
    # is now whatever the filter says, which is the only source that can make the
    # assertions below about the production code rather than about this test.
    for n, status in enumerate(statuses, start=1):
        write_task(aut, n, status=status, last_run=_age(hours=6))
    board = aut._all_board_tasks()
    assert len(board) == len(statuses), "fixture did not parse as written"
    runnables = aut._all_runnable_tasks(board)
    admitted = {str(t.get("status", "")).strip() for t in runnables}
    assert admitted == set(aut.RUNNABLE_STATUSES), (
        f"dispatch admits {sorted(admitted)} but RUNNABLE_STATUSES says "
        f"{sorted(aut.RUNNABLE_STATUSES)}: the filter stopped reading the "
        "constant the dependency gate reads, so the two status lists are drifting")

    for status in statuses:
        write_task(aut, 1, status=status, last_run=_age(hours=6))
        write_task(aut, 2, depends_on=1, last_run=_age(days=3))
        up, dep = _parse(aut, 1), _parse(aut, 2)
        meets = aut._is_dependency_met(dep, [up, dep], now=PIN)
        if status in admitted:
            assert meets is True, (
                f"upstream status {status!r} is admitted by the runnable filter "
                f"but does not satisfy a dependent: the two lists have diverged")
        else:
            assert meets is False, (
                f"upstream status {status!r} is NOT admitted by the runnable "
                f"filter yet still satisfies a dependent — the fail-open this "
                f"item exists to close")
    assert admitted == {"up_next", "in_progress", "failed"}, (
        f"the runnable filter admits {sorted(admitted)}; the dependency rule is "
        f"asserted against exactly that set, so re-read this test if the filter "
        f"changed")
    # And `failed`, the one status the filter admits that still never dispatches:
    # its exclusion is `_is_task_due`'s own status gate, which is the asymmetry
    # clause 6's ruling depends on. Asserted so it stays visible.
    write_task(aut, 1, status="failed", last_run=_age(hours=6))
    assert 1 not in _due_ids(aut), (
        "`failed` started dispatching: the asymmetry this test documents is the "
        "reason that exclusion lives in _is_task_due and not in the filter")
