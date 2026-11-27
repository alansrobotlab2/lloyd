"""#3022: the count #706's "tasks pinned at exactly 0.00 must not increase" is about.

The invariant has been restated since #522 as a claim about a NUMBER, and until
``scripts/maintenance/zero_pinned_baseline.py`` nothing in the tree computed that
number — ``git grep -n "must not increase" -- '*.py'`` at 13d24da8 is 0 hits, and
``tests/test_bench_invariants.py``'s 30 nodes are all structural over the task
manifest, none reading ``_pipeline/research/ledger.jsonl``. So these nodes do not
grade a guard: they pin the four ways a counter like this can quietly report a number
that is not the one a person ruled on, and then pin the declaration to the number
itself.

  * **max, not min.** A task with one 0.00 trial and one 0.50 trial can score; the
    invariant is about tasks whose BEST trial is nothing. A min reads the same fixture
    as a task that always fails, and inflates the count.
  * **null is not zero.** 8,225 of the live ledger's 30,460 per-trial rows carry
    ``"composite_score": null`` (the not-rankable shape of #416/#646/#698) as measured
    on 2026-11-27, and only 3,486 of those 8,225 sit inside a default-length window —
    so the whole-file null rate and the window's not-rankable population are different
    numbers and both are pinned below. Coercing null to ``0.0`` is a one-token bug that
    reports a far larger count than the declared 2 and looks entirely reasonable.
  * **the window is part of the number.** Anchored on 19 October 2026 the live ledger
    gave 9 pinned tasks over 30 days and 11 over 7; anchored on 2026-11-27 it gives 2
    either way, over 21 scored tasks. A module that carried a second window literal, or
    a report line that printed a count without the window it came from, would let one of
    those be forwarded as the other — which is how #706's figure got separated from
    its definition the first time. The window is one literal in one constant, counted
    by AST below rather than trusted.
  * **a task can leave the pinned list without learning to score.** Six of the nine
    tasks the October ruling declared pinned had 581 in-window trials each on
    2026-11-27 and not one numeric score among them, so the count fell to 2 while
    nothing improved. The population printed beside the count exists for that, and the
    nodes below pin that it is window-scoped, score-exclusive, and never confused with
    the whole-file null counter.
  * **the declaration travels with its date and its command.** 2 was ruled on
    2026-11-27 over the all-rows definition, replacing the 9 that was ruled on 19
    October 2026 over 31 scored tasks — when the same filter restricted to ``BASELINE*``
    variants returned 10. A constant whose comment does not say which of the two
    produced it is a number nobody can re-check.

The CLI is driven as a subprocess with ``--ledger``/``--now``, because resolving the
default ledger is the one thing a test must NOT do to the live 36 MB store, and
``tests/conftest.py`` redirects ``LLOYD_DATA`` to a scratch root anyway — so the
default is pinned as a PATH SHAPE (it ends with the research-ledger relpath), never as
a read.
"""
import ast
import importlib.util
import inspect
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "maintenance" / "zero_pinned_baseline.py"

#: The window anchor every fixture is stamped against, so no node moves when the clock
#: does. The date the owed-check ruling re-declared the constant, over this instant's
#: window; the stamps below are all relative to it, so any anchor would do.
NOW = datetime(2026, 11, 27, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def zpb():
    # scripts/maintenance has no __init__.py — the same loader
    # tests/test_gate_to_land_lag.py uses for this directory.
    assert SCRIPT.is_file(), (f"{SCRIPT} is not on disk — the counter went missing and "
                              "#3022's five clauses go unpinned")
    spec = importlib.util.spec_from_file_location("zero_pinned_baseline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: the module is `from __future__ import annotations` with a
    # `@dataclass`, and dataclasses resolves a field's string annotation through
    # `sys.modules[cls.__module__]` — an unregistered module looks up to None and the
    # class body dies with AttributeError, naming dataclasses.py rather than this file.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stamp(days_ago: float, moment: datetime = NOW) -> str:
    """A ``created_at`` in the ledger's own spelling: ISO-8601 UTC with a ``Z``."""
    return (moment - timedelta(days=days_ago)).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def _row(task_id: str, score, days_ago: float = 1.0, **extra) -> dict:
    """One per-trial ledger row: the keys a scoring reader touches, nothing else."""
    row = {"round_id": "R_20261127_000000", "variant_id": "V_one", "task_id": task_id,
           "created_at": _stamp(days_ago), "composite_score": score}
    row.update(extra)
    return row


def _undated_row(task_id: str, **extra) -> dict:
    """A row with no ``composite_score`` key AT ALL — the other null shape."""
    row = {"round_id": "R_20261127_000000", "variant_id": "V_one", "task_id": task_id,
           "created_at": _stamp(2.0)}
    row.update(extra)
    return row


def _ledger(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "ledger.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, cwd=str(ROOT))


# ---------------------------------------------------------------- clause 1 --
def test_pinned_at_zero_keeps_tasks_whose_best_trial_is_zero_and_drops_any_that_can_score(zpb):
    """The function exists, takes rows and a window, and returns sorted ids by MAX.

    Four tasks, one pass: ``bench_dead`` (two 0.00 trials) and ``bench_lonely`` (one)
    are pinned; ``bench_can_score`` has a 0.00 trial AND a 0.50 one, so its MAX is 0.50
    and it is out — the min would have filed it as broken, which is the single way this
    count over-states a corpus; and ``bench_almost`` has a best trial of 0.001, which is
    not exactly 0.00 and must not be swept in by a tolerance.
    """
    rows = [
        _row("bench_dead", 0.0), _row("bench_dead", 0.0),
        _row("bench_can_score", 0.0), _row("bench_can_score", 0.5),
        _row("bench_lonely", 0.0),
        _row("bench_almost", 0.001),
    ]
    assert zpb.pinned_at_zero(rows, 30, now=NOW) == ["bench_dead", "bench_lonely"]
    # Same answer through the denominator-carrying path the CLI uses.
    m = zpb.measure(rows, window_days=30, now=NOW)
    assert list(m.pinned) == ["bench_dead", "bench_lonely"]
    assert m.tasks_scored == 4, "the population the count is a share of"
    assert m.trials_used == 6


def test_the_zero_test_is_exact_so_a_task_that_almost_scores_is_never_pinned(zpb):
    """0.001 is not 0.00: the invariant says "exactly 0.00", so no epsilon is allowed."""
    assert zpb.pinned_at_zero([_row("bench_epsilon", 0.001)], 30, now=NOW) == []


# ---------------------------------------------------------------- clause 2 --
def test_a_fixture_whose_rows_are_all_null_or_absent_scores_returns_no_ids(zpb):
    """All-null rows yield NO ids — not every task in the file.

    Two rows say ``"composite_score": null`` and one omits the key entirely (the
    #416/#646/#698 not-rankable shape, 3,491 live rows of it). Reading either as 0.0
    would return both tasks and read as a corpus suddenly full of broken benches, so
    the answer here is the empty list, and the denominators say where the rows went.
    """
    rows = [_row("bench_null_one", None), _row("bench_null_two", None),
            _undated_row("bench_null_one")]
    assert zpb.pinned_at_zero(rows, 30, now=NOW) == []
    m = zpb.measure(rows, window_days=30, now=NOW)
    assert m.pinned == ()
    assert m.tasks_scored == 0, "a null row scores nothing, so it has no population"
    assert m.rows_read == 3
    assert m.rows_null_score == 3, ("an explicit null and an absent key are the same "
                                    "fact here: no score recorded")
    assert m.rows_bad_score == 0, "nothing was corrupt; nothing belongs in that bucket"


def test_null_rows_beside_one_nonzero_trial_still_return_no_ids(zpb):
    """The same fixture plus one numeric non-zero trial per task: still nothing pinned.

    This is the shape a null-coercing counter survives — with the coercion the null
    rows would pin both tasks on their own; without it, each task's best (and only)
    score is non-zero.
    """
    rows = [_row("bench_null_one", None), _row("bench_null_two", None),
            _undated_row("bench_null_one"),
            _row("bench_null_one", 0.25, days_ago=3.0),
            _row("bench_null_two", 1.0, days_ago=3.0)]
    assert zpb.pinned_at_zero(rows, 30, now=NOW) == []
    assert zpb.measure(rows, window_days=30, now=NOW).tasks_scored == 2


def test_a_non_numeric_score_is_dropped_rather_than_read_as_zero(zpb):
    """A score that is a string, or a bool, is a broken row — not a 0.00 trial."""
    rows = [_row("bench_string", "0.0"), _row("bench_bool", True)]
    assert zpb.pinned_at_zero(rows, 30, now=NOW) == []
    assert zpb.measure(rows, window_days=30, now=NOW).rows_bad_score == 2


def test_event_rows_are_counted_and_never_scored(zpb):
    """The four ``event`` row kinds carry no trial, so they cannot pin a task.

    The live ledger's 929 event rows include ``decision`` rows that name a task, so a
    reader keyed on ``task_id`` alone would pull them in.
    """
    rows = [_row("bench_real", 0.0),
            {"event": "decision", "round_id": "R_20261127_000000", "task_id": "bench_real",
             "created_at": _stamp(1.0), "composite_score": 1.0, "should_promote": False}]
    m = zpb.measure(rows, window_days=30, now=NOW)
    assert list(m.pinned) == ["bench_real"]
    assert m.rows_event == 1
    assert m.trials_used == 1


# ------------------------------------------ the not-rankable population (#4646) --
def test_a_task_whose_in_window_trials_are_all_null_prints_beside_the_pinned_count(zpb):
    """One null-only task and one scored task: the line names the first, with its
    in-window row count, and never the second.

    ``bench_all_null`` ran twice in the window and the judge scored it neither way (the
    #416/#646/#698 not-rankable shape); ``bench_scores`` ran once at 0.50;
    ``bench_pinned`` ran once at 0.00. The count is therefore one and the denominator is
    two, and the task that belongs to NEITHER of those populations is ``bench_all_null``
    — six real benches are in that shape right now, which is why the count alone fell
    from 9 to 2 while nothing about them improved. A line that printed only
    "measured 1 of 2 tasks" is the report that reads as a recovery.
    """
    rows = [_row("bench_all_null", None), _row("bench_all_null", None, days_ago=4.0),
            _row("bench_scores", 0.5), _row("bench_pinned", 0.0)]
    m = zpb.measure(rows, window_days=30, now=NOW)
    assert m.pinned == ("bench_pinned",) and m.tasks_scored == 2
    assert m.not_rankable == ("bench_all_null",)
    assert m.not_rankable_rows_by_task == {"bench_all_null": 2}
    assert m.not_rankable_count == 1 and m.not_rankable_rows == 2
    assert "bench_all_null" not in m.pinned, "a task with no score is never pinned"
    line = zpb.report_line(m)
    assert "\n" not in line, "the population rides on the same line as the count"
    assert "bench_all_null:2" in line, (
        "named beside the count, with how many in-window rows carry no score")
    assert "not-rankable in window: 1 task(s) / 2 rows" in line
    assert "bench_scores" not in line, "a scored task is not part of the population"


def test_the_not_rankable_population_is_window_scoped_and_score_exclusive(zpb):
    """Null rows stamped outside the window attribute to nobody, and a task with both a
    null and a numeric in-window trial is a scored task, never a population member.

    Five tasks, one pass. ``bench_all_null`` has one in-window null row and nothing
    else: named. ``bench_null_out`` has its only null row stamped 45 days back: not
    named. ``bench_null_undated`` has a null row with no parseable stamp: not named.
    ``bench_null_and_score`` has an in-window null row AND an in-window 0.75: it belongs
    to ``tasks_scored`` alone, because "not-rankable" is a claim about every trial in
    the window and one numeric score refutes it. ``bench_pinned`` is the pinned one.
    """
    rows = [_row("bench_all_null", None),
            _row("bench_null_out", None, days_ago=45.0),
            {**_row("bench_null_undated", None), "created_at": ""},
            _row("bench_null_and_score", None), _row("bench_null_and_score", 0.75),
            _row("bench_pinned", 0.0)]
    m = zpb.measure(rows, window_days=30, now=NOW)
    assert m.not_rankable == ("bench_all_null",), (
        "only the task whose in-window trials are all null, and no others")
    assert m.not_rankable_rows_by_task == {"bench_all_null": 1}
    assert m.tasks_scored == 2 and list(m.pinned) == ["bench_pinned"]
    assert "bench_null_and_score" not in m.not_rankable, (
        "a task with one numeric in-window trial is in the scored denominator only")
    assert m.rows_null_score == 4, "every null row in the file, whichever window it sits in"
    assert m.rows_null_unplaced == 2, "the out-of-window null row and the undated one"
    assert m.rows_undated == 0, (
        "an undated null row is attributed to nobody, not also counted as an undated trial")


def test_rows_null_score_stays_whole_file_while_the_population_counts_the_window(zpb):
    """One null row inside the window and one outside it: the counter reports 2, the
    population reports 1 task over 1 row, and neither task is pinned.

    These two denominators measure DIFFERENT THINGS on purpose. ``rows_null_score`` is
    the file's null rate and the number ``--json`` has published since #3022, so moving
    the null branch below the window check would silently redefine a published figure —
    and no other node here would catch it, because the older null fixtures place every
    null row in-window and pass under either reading. This node is the one that cannot:
    it puts a row on each side of the boundary and requires 2 and 1. The second half is
    the coercion this whole module exists to refuse: a pile of null rows pins nothing.
    """
    m = zpb.measure([_row("bench_a", None), _row("bench_b", None, days_ago=45.0)],
                    window_days=30, now=NOW)
    assert m.rows_null_score == 2, "whole-file, unchanged by where the window fell"
    assert m.not_rankable == ("bench_a",) and m.not_rankable_count == 1
    assert m.not_rankable_rows == 1 and m.not_rankable_rows_by_task == {"bench_a": 1}
    assert m.rows_null_unplaced == 1
    assert m.pinned == () and m.tasks_scored == 0, "a null row is not a 0.00 trial"

    heavy = zpb.measure([_row("bench_a", None, days_ago=float(day))
                         for day in (1.0, 2.0, 3.0, 4.0)], window_days=30, now=NOW)
    assert heavy.pinned == (), "four null rows pin nothing, however many the window holds"
    assert heavy.rows_null_score == 4 and heavy.not_rankable == ("bench_a",)
    assert heavy.not_rankable_rows == 4


# ---------------------------------------------------------------- clause 3 --
def test_a_trial_days_old_is_pinned_only_by_a_window_wide_enough_to_reach_it(zpb):
    """One task, one 0.00 trial, stamped 35 days back: in at 40, out at 30.

    This is the clause that makes the window an argument rather than an assumption —
    and the reason #706's 9 must always be quoted with the window it was counted over.
    The default call answers the 30-day way, because the default IS the declared
    window.
    """
    rows = [_row("bench_aged", 0.0, days_ago=35.0)]
    assert zpb.pinned_at_zero(rows, 40, now=NOW) == ["bench_aged"]
    assert zpb.pinned_at_zero(rows, 30, now=NOW) == []
    assert zpb.pinned_at_zero(rows, zpb.DEFAULT_WINDOW_DAYS, now=NOW) == []


def test_the_window_and_ledger_path_are_parameters_with_one_window_literal(zpb):
    """Defaults: the window from the one constant, the ledger from the research store.

    Three things pinned together. (a) ``window_days`` on both the helper and the
    file-level reader, and ``--window-days`` on the CLI, all default to
    ``DEFAULT_WINDOW_DAYS`` — no fourth spelling of the number. (b) The AST holds
    exactly one integer literal equal to the window, and it is the constant's own
    assignment: a second literal anywhere else in the file is a second definition of
    the window the declared baseline was measured under, and it is exactly the kind of
    drift no runtime test can see. (c) The path default resolves under
    ``_pipeline/research/ledger.jsonl``, asserted as a PATH SHAPE because the resolver
    is lazy and a test must not read the live 13 MB ledger to check a default.
    """
    assert zpb.DEFAULT_WINDOW_DAYS == 30
    helper = inspect.signature(zpb.pinned_at_zero).parameters["window_days"]
    file_level = inspect.signature(zpb.measure_from_ledger).parameters["window_days"]
    assert helper.default is zpb.DEFAULT_WINDOW_DAYS
    assert file_level.default is zpb.DEFAULT_WINDOW_DAYS
    assert zpb.build_parser().get_default("window_days") == zpb.DEFAULT_WINDOW_DAYS
    assert zpb.build_parser().get_default("ledger") is None, (
        "None means main() resolves it through default_ledger(), not a copy of the path")

    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    window_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool)
        and isinstance(node.value, int) and node.value == zpb.DEFAULT_WINDOW_DAYS
    ]
    constant_line = next(i for i, line in enumerate(source.splitlines(), start=1)
                         if line.startswith("DEFAULT_WINDOW_DAYS = "))
    assert window_lines == [constant_line], (
        "the window literal must appear exactly once, in its own constant — a second one "
        "is a second definition of the window the declared baseline was measured under")

    assert zpb.DEFAULT_LEDGER.parts[-3:] == ("_pipeline", "research", "ledger.jsonl")
    assert zpb.LEDGER_RELPATH == Path("_pipeline") / "research" / "ledger.jsonl"
    assert str(zpb.default_ledger()).endswith(str(zpb.LEDGER_RELPATH))


def test_the_default_anchor_is_now_so_a_fresh_run_measures_a_fresh_window(zpb):
    """Omitting ``now`` anchors on this instant: yesterday counts, 35 days ago does not.

    The fixtures pass an anchor for determinism; this node pins the default that a
    nightly run actually gets, with stamps built off ``datetime.now`` so it cannot be
    satisfied by a hard-coded answer.
    """
    live = datetime.now(timezone.utc)
    recent = _row("bench_recent", 0.0)
    recent["created_at"] = _stamp(1.0, moment=live)
    aged = _row("bench_aged", 0.0)
    aged["created_at"] = _stamp(35, moment=live)
    assert zpb.pinned_at_zero([recent, aged]) == ["bench_recent"]


# ---------------------------------------------------------------- clause 4 --
def test_the_declared_baseline_constant_is_two_and_its_comment_names_706_and_the_command(zpb):
    """``DECLARED_BASELINE_ZERO_PINNED == 2``, and its comment can be re-checked.

    2 is a dated ruling: #706's constant, re-declared by the owed-check ruling of
    2026-11-27 over 21 tasks with a scored trial in the window, replacing the 9 ruled on
    19 October 2026 over 31, definition unchanged. The constant is only worth having
    because the text beside it names the item it came from and the exact command that
    reproduces it — a bare number in a module is the thing #706's owed entry found with
    no referent — and pinning the number and its date as one pair is what makes a
    re-declaration a change somebody has to notice, rather than an edit that quietly
    agrees with whatever the sliding window measures today.
    """
    assert zpb.DECLARED_BASELINE_ZERO_PINNED == 2
    assert zpb.DECLARED_ON == "2026-11-27"

    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    declared_at = next(i for i, line in enumerate(lines)
                       if line.startswith("DECLARED_BASELINE_ZERO_PINNED ="))
    comment: list[str] = []
    probe = declared_at - 1
    while probe >= 0 and lines[probe].lstrip().startswith("#:"):
        comment.append(lines[probe])
        probe -= 1
    adjacent = "\n".join(reversed(comment))
    assert "#706" in adjacent, "the constant must name the item that ruled it"
    assert "python scripts/maintenance/zero_pinned_baseline.py" in adjacent, (
        "the constant must carry the command that recomputes it")
    assert "#706" in (zpb.__doc__ or ""), "and the module says which invariant it serves"


# ---------------------------------------------------------------- clause 5 --
def test_the_report_line_carries_baseline_and_window_and_count_and_ids_on_one_line(zpb):
    """The line the invariant is worth, in one line, with all four together.

    Declaration, window, count and ids on a SINGLE line, because a count forwarded
    without the window it was measured over is the exact failure that orphaned #706's
    figure from its definition. The fixture pins the shape against a window (40) that
    is NOT the declared one, so a line built from a hard-coded window would fail here.
    """
    rows = [_row("bench_a", 0.0), _row("bench_b", 0.0), _row("bench_b", 0.0),
            _row("bench_scores", 0.75), _row("bench_null", None),
            _row("bench_beyond", 0.0, days_ago=35.0)]
    m = zpb.measure(rows, window_days=40, now=NOW)
    line = zpb.report_line(m)
    assert "\n" not in line, "one line: the four facts cannot be separated across prints"
    assert f"declared baseline {zpb.DECLARED_BASELINE_ZERO_PINNED}" in line
    assert zpb.DECLARED_ON in line
    assert "40 days" in line, "the window this count was measured over"
    assert f"measured {m.pinned_count} of {m.tasks_scored} tasks" in line
    assert m.pinned_count == 3 and m.tasks_scored == 4, (
        "three pinned out of four tasks with a scored trial: bench_a, bench_b, "
        "bench_beyond and bench_scores")
    for task_id in m.pinned:
        assert task_id in line
    assert "bench_beyond" in line, "window 40 reaches the 35-day-old trial"
    assert "#706" in line


def test_a_moved_count_prints_drift_and_still_reads_as_a_measurement_not_an_alarm(zpb):
    """Measured != declared: the drift line names baseline and window and exits level.

    The declared 2 was ruled on one day and the window slides daily, so a moved count is
    evidence for the next ruling. A report that alarmed on it would be crying wolf by
    arithmetic; one that stayed SILENT would hide the move.

    The fixture measures THREE pinned tasks — deliberately not the declared two, because
    a drift node whose fixture equals the constant can no longer fail — beside one task
    whose only in-window trial is null, which pins that the population re-declared by
    #4646 does not leak into the count the drift compares.
    """
    rows = [_row("bench_a", 0.0), _row("bench_b", 0.0), _row("bench_c", 0.0),
            _row("bench_all_null", None)]
    m = zpb.measure(rows, window_days=30, now=NOW)
    assert m.pinned_count == 3, "three pinned, so the fixture still disagrees with a 2"
    assert m.not_rankable == ("bench_all_null",), (
        "the not-rankable task is named beside the count, not added to it")
    assert m.agrees_with_declaration is False
    drift = zpb.drift_line(m)
    assert drift is not None
    assert f"declared {zpb.DECLARED_BASELINE_ZERO_PINNED}" in drift
    assert "30 days" in drift and "measured 3" in drift
    assert zpb.drift_line(zpb.measure(
        [_row(f"bench_{i}", 0.0) for i in range(zpb.DECLARED_BASELINE_ZERO_PINNED)],
        window_days=30, now=NOW)) is None, "at the declared count, no drift sentence"


# ------------------------------------------------------- the CLI, as a process --
def test_the_cli_prints_the_baseline_window_count_and_ids_on_one_line(zpb, tmp_path):
    """End to end across the process boundary: --ledger, --now, and the report line.

    Seven rows on disk — three tasks pinned, one scoring, one whose only in-window trial
    is null, and one whose only trial sits outside the window — driven through ``main``
    as a subprocess, because the CLI is what a person or a nightly report runs and the
    seam between argparse and the counter has never been crossed by anything else.

    Three pinned against a declared two is deliberate: a CLI node whose fixture agreed
    with the constant could no longer print the drift sentence, and the drift line is
    what tells a reader the declaration is stale rather than the corpus broken. The same
    line has to carry the not-rankable task beside the count, because the text report is
    the half a person pastes into a note.
    """
    ledger = _ledger(tmp_path, [
        _row("bench_pinned_one", 0.0),
        _row("bench_pinned_two", 0.0), _row("bench_pinned_two", 0.0),
        _row("bench_pinned_three", 0.0),
        _row("bench_scoring", 0.6),
        _row("bench_null", None),
        _row("bench_out_of_window", 0.0, days_ago=45.0),
    ])
    out = _run_cli("--ledger", str(ledger), "--now", "2026-11-27T12:00:00Z")
    assert out.returncode == 0, out.stderr
    report = [line for line in out.stdout.splitlines()
              if line.startswith("zero-pinned #706")]
    assert len(report) == 1, out.stdout
    line = report[0]
    assert f"declared baseline {zpb.DECLARED_BASELINE_ZERO_PINNED}" in line
    assert "30 days" in line
    assert "measured 3 of 4 tasks" in line
    assert "bench_pinned_one" in line and "bench_pinned_two" in line
    assert "bench_pinned_three" in line
    assert "bench_scoring" not in line and "bench_out_of_window" not in line
    assert "not-rankable in window: 1 task(s) / 1 row" in line, (
        "the task whose only in-window trial is null prints beside the count")
    assert "bench_null:1" in line, "named with its in-window not-rankable row count"
    assert "#706 baseline moved" in out.stdout, "3 measured against a declared 2 is drift"


def test_the_cli_json_carries_the_same_count_and_the_row_denominators(zpb, tmp_path):
    """``--json`` is stdout-pure and reports the drops and the not-rankable population,
    so a nightly row can be honest.

    The same seven-row fixture through ``--json``: the pinned list must agree with the
    text line's, the out-of-window row must show as a DROPPED count rather than a task,
    and the null-only task must arrive as a named population with its in-window row
    count — the machine-readable half of the same fact the text line prints, and the
    half a scheduled report would read. A report whose stdout is not one parseable
    object is a report no nightly job can read.
    """
    ledger = _ledger(tmp_path, [
        _row("bench_pinned_one", 0.0),
        _row("bench_pinned_two", 0.0), _row("bench_pinned_two", 0.0),
        _row("bench_pinned_three", 0.0),
        _row("bench_scoring", 0.6),
        _row("bench_null", None),
        _row("bench_out_of_window", 0.0, days_ago=45.0),
    ])
    out = _run_cli("--ledger", str(ledger), "--now", "2026-11-27T12:00:00Z", "--json")
    assert out.returncode == 0, out.stderr
    doc = json.loads(out.stdout)
    assert doc["pinned"] == ["bench_pinned_one", "bench_pinned_three",
                             "bench_pinned_two"]
    assert doc["pinned_count"] == 3, "three, so the drift stays a drift at a declared 2"
    assert doc["tasks_scored"] == 4
    assert doc["window_days"] == zpb.DEFAULT_WINDOW_DAYS
    assert doc["declared_baseline"] == zpb.DECLARED_BASELINE_ZERO_PINNED
    assert doc["declared_on"] == zpb.DECLARED_ON
    assert doc["agrees_with_declaration"] is False
    assert doc["rows_null_score"] == 1
    assert doc["rows_null_unplaced"] == 0, "the one null row is inside the window"
    assert doc["tasks_not_rankable"] == 1
    assert doc["not_rankable"] == ["bench_null"], (
        "the null-only task is its own population, not a pinned task and not a scored one")
    assert doc["not_rankable_rows"] == 1
    assert doc["not_rankable_rows_by_task"] == {"bench_null": 1}
    assert "bench_scoring" not in doc["not_rankable"], "a scored task is never in it"
    assert doc["rows_outside_window"] == 1
    assert doc["rows_read"] == 7
    assert doc["ledger"] == str(ledger)


def test_a_missing_ledger_is_a_refusal_that_names_the_baseline_and_the_window(zpb, tmp_path):
    """No file: exit nonzero, and the text says WHICH invariant went unchecked.

    A refusal that prints only "no such file" reads as a clean run to whatever parses
    it next, and the whole defect #3022 closes is a number going unread while its
    invariant looked satisfied. So the refusal names the declared baseline, the window
    it could not measure, and the path it tried — and never prints the report line.
    """
    missing = tmp_path / "absent" / "ledger.jsonl"
    out = _run_cli("--ledger", str(missing))
    assert out.returncode == 1
    assert f"declared baseline {zpb.DECLARED_BASELINE_ZERO_PINNED}" in out.stderr
    assert "30 days" in out.stderr
    assert str(missing) in out.stderr
    assert "zero-pinned #706: cannot measure" in out.stderr
    assert "measured" not in out.stdout, "no report line without a measurement"


def test_an_unparseable_anchor_is_refused_rather_than_silently_measured_as_now(zpb, tmp_path):
    """A bad ``--now`` cannot become "this instant by default" — that is a wrong window.

    A reader who typed a window anchor and had it ignored would forward a count over a
    window they did not ask for. Nonzero exit, and the same baseline-and-window
    refusal text.
    """
    ledger = _ledger(tmp_path, [_row("bench_a", 0.0)])
    out = _run_cli("--ledger", str(ledger), "--now", "not-a-stamp")
    assert out.returncode == 2
    assert "unparseable --now" in out.stderr
    assert f"declared baseline {zpb.DECLARED_BASELINE_ZERO_PINNED}" in out.stderr
    assert "30 days" in out.stderr
