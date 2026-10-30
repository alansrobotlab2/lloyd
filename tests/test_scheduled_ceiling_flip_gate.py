#!/usr/bin/env python3
"""The flip gate's reader: four readings, one verdict, and no way to pass on nothing (#2927).

What is being pinned is a decision-maker. `scripts/maintenance/scheduled_ceiling_flip_gate.py`
is the artifact a person consults before setting `LLOYD_SCHEDULED_CEILING_MODE=enforce`
(#2670's owed entry 1: seven shadow days, no would-strip name above 5% of turns, zero
tool-name refusals), so its failure modes are the expensive ones: a report that prints
PASS because it read nothing, a rate taken over the wrong denominator, or a bar quietly
lowered. Every node below therefore states which reading it can go red on.

The seam this crosses is a process boundary in both directions:

  * IN: the ledger is a file under the data root that a DIFFERENT process wrote —
    `app/scheduled_ceiling.TurnSurface.witness` appends one row per scheduled turn from
    inside `run_task`, days before anyone reads it. So every test here writes real JSONL
    bytes and lets the script read them back through `app.scheduled_ceiling.read_rows`,
    never a dict handed to a stub. A malformed tail line is included where it matters: an
    append-only file is caught mid-write more often than not, and skipping it is the
    production parser's documented leniency, which a private re-parse would not have.
  * OUT: the consumer is a human or the owed-check job reading stdout and the exit
    status in a session that wrote none of the rows. So the assertions are on printed
    lines and on the return code, not on the internals of `measure()` — a verdict that
    exists only in a dict is a verdict nobody can act on.

The fixtures are synthetic on purpose: the live ledger gains a row with every scheduled
turn, so a node quoting its current counts would be red by tomorrow and would teach the
next reader to distrust a red. The no-argument path is still exercised, through
`LLOYD_SCHEDULED_CEILING_LEDGER`, the knob `app.scheduled_ceiling.ledger_path()` resolves
— against a file this test wrote rather than against the store the fleet is appending to.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import scheduled_ceiling as sc
from scripts.maintenance import scheduled_ceiling_flip_gate as gate

# ─────────────────────── fixtures: ledger bytes, not dicts ───────────────────────

#: The date spine the fixtures stamp rows with. Seven of these is exactly
#: `MIN_SHADOW_DATES`, so a bar that moved from 7 to 6 reddens the PASS node.
DATES = [f"2026-10-{d:02d}" for d in range(11, 18)]

#: The two dates the default fixture carries the over-bar candidate on: 2 strips in 42
#: rows is 4.8%, under a 5% bar. The PASS arm therefore runs on evidence rather than on
#: an empty table — a reader that printed PASS only when nothing was ever stripped would
#: never be seen working.
DEFAULT_STRIP_DATES = (DATES[2], DATES[4])


def _row(date: str, *, mode: str = "shadow", task_id: int = 5,
         names: list[str] | None = None, compile_error: str = "") -> dict:
    """One row in the shape `TurnSurface.witness` writes.

    A row carrying `compile_error` and NO `would_strip` key is the writer's own
    distinction: an empty list means "measured, and every call was inside the ceiling",
    so a failed compile may not borrow it. The reader has to keep those apart.
    """
    row = {
        "ts": f"{date}T04:00:00+00:00",
        "date": date,
        "mode": mode,
        "task_id": task_id,
        "run_id": f"run_{task_id}_{date.replace('-', '')}_040000",
        "grant_scope": f"autonomy-task:{task_id}",
        "n_calls": 7,
        "envelope_size": 33,
    }
    if compile_error:
        row["compile_error"] = compile_error
    elif names is not None:
        row["would_strip"] = list(names)
        row["n_would_strip"] = len(names)
    return row


def _write_ledger(tmp_path: Path, rows: list[dict], *, tail_junk: bool = False) -> Path:
    """Append-only bytes at a path, which is the only form the reader accepts."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "scheduled-ceiling.jsonl"
    lines = [json.dumps(r, sort_keys=True) for r in rows]
    if tail_junk:
        # Half a row: the append was caught mid-write. `read_rows` skips it.
        lines.append('{"date": "2026-10-18", "mode": "shadow", "would_stri')
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


def _passing_rows(per_date: int = 6, *, strip_dates=DEFAULT_STRIP_DATES,
                  name: str = "automod_vault_land") -> list[dict]:
    """`7 * per_date` rows over the seven dates, `name` on the first row of each date in
    `strip_dates`. Defaults: 42 rows, 2 strips → 4.8%, every bar met."""
    rows: list[dict] = []
    for date in DATES:
        for i in range(per_date):
            names = [name] if (i == 0 and date in strip_dates) else []
            rows.append(_row(date, task_id=40 + i, names=names))
    return rows


def _run(tmp_path: Path, rows: list[dict], *, tail_junk: bool = False,
         argv: list[str] | None = None,
         monkeypatch=None) -> tuple[int, str]:
    """Run the script's real entry point over written bytes → (exit code, stdout)."""
    p = _write_ledger(tmp_path, rows, tail_junk=tail_junk)
    return _run_at(p, argv=argv, monkeypatch=monkeypatch)


def _run_at(path: Path, *, argv: list[str] | None = None,
            monkeypatch=None) -> tuple[int, str]:
    """`main()` against one ledger file, via the `--ledger` flag or via the env knob.

    Default is the flag. `argv=[]` plus a monkeypatch exercises the no-argument path,
    which resolves the file through `app.scheduled_ceiling.ledger_path()` — the same
    function the writer resolves, so that is the seam and not a convenience.
    """
    if argv is None:
        argv = ["--ledger", str(path)]
    if monkeypatch is not None:
        monkeypatch.setenv(sc.LEDGER_ENV, str(path))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gate.main(argv)
    return rc, buf.getvalue()


def _verdict_lines(out: str) -> list[str]:
    """The PRE-FLIP verdict only, with its label stripped so the existing assertions read
    as they always did.

    The report now carries two verdicts (#3415): `verdict: PRE-FLIP <V>` is the four bars
    over shadow rows and `verdict: POST-FLIP HEALTH <V>` is what enforcement has actually
    refused. Both begin with `VERDICT_PREFIX`, so a helper that returned every one of them
    would turn every single-verdict assertion below into a two-element list and fail tests
    whose report is correct. The enforce half is pinned by name in `_post_flip_verdict`
    below and end-to-end in
    `tests/test_scheduled_ceiling_service_unit.py::test_the_bars_and_the_post_flip_verdict_read_their_own_sides_of_the_ledger`.
    """
    return [ln.replace(gate.PRE_FLIP_PREFIX, "", 1)
            for ln in out.splitlines()
            if ln.startswith(gate.VERDICT_PREFIX + gate.PRE_FLIP_PREFIX)]


def _post_flip_verdict(out: str) -> str:
    """The one POST-FLIP HEALTH line. Asserts it exists, so a report that lost the
    enforce verdict cannot pass these tests by going back to a single verdict."""
    lines = [ln for ln in out.splitlines()
             if ln.startswith(gate.VERDICT_PREFIX + gate.POST_FLIP_PREFIX)]
    assert len(lines) == 1, f"want exactly one post-flip verdict, got {lines}:\n{out}"
    return lines[0]


@pytest.fixture(autouse=True)
def _isolated_grant_store(tmp_path, monkeypatch):
    """No reading in this file may consult the fleet's live grant store.

    The post-flip half of the report asks `policy.default_store().live()` which refused name
    was licensed, and `default_store()` resolves through `LLOYD_GRANT_DB`. Left unset, a
    test here would grade enforcement against however many grants happen to exist on the box
    that morning, and a grant minted for a nightly vault-landing job could flip a unit
    test's verdict. One empty DB per test, at a fresh path each time so the store's own
    per-path cache cannot bleed a row from one test into the next: nothing is licensed, which
    is the state these fixtures describe.

    It costs coverage rather than buys it — the licensed branch is exercised where a store is
    built WITH rows, in tests/test_scheduled_ceiling_service_unit.py.
    """
    monkeypatch.setenv("LLOYD_GRANT_DB", str(tmp_path / "grants-isolated.db"))


def _table(out: str) -> list[str]:
    """The name rows of the table, top to bottom (a rate line ends in a `%`)."""
    return [ln.strip().split()[0] for ln in out.splitlines()
            if ln.startswith("  ") and not ln.startswith("  ! ")
            and ln.strip().endswith("%")]


# ──────────────────────────────────── the bars ────────────────────────────────────

def test_the_bars_are_the_ones_2670_owed_and_the_report_shows_both_numbers():
    """The two figures a flip turns on are pinned, not paraphrased.

    #2670's owed entry: "7 shadow days … with no would-strip name above 5% of turns". A
    reader whose bar drifted to 5 dates or to 10% would still print a confident verdict,
    so both constants are asserted at their contract values AND shown to be the numbers
    the output actually quotes.
    """
    assert gate.MIN_SHADOW_DATES == 7
    assert gate.NAME_RATE_CEILING == 0.05
    out = gate.format_report(gate.measure(_passing_rows()))
    assert "every name <= 5.0%" in out, "the table's stated bar is not the code's bar"
    six = [r for r in _passing_rows(strip_dates=()) if r["date"] != DATES[-1]]
    six_out = gate.format_report(gate.measure(six))
    assert "shadow dates 6 < 7 required" in six_out, (
        "the date reason must name both the count and the bar, or a reader cannot tell "
        "a near-miss from a broken reader")


def test_a_ledger_that_meets_every_bar_prints_one_pass_and_exits_zero(tmp_path):
    """The control arm: real bytes, seven dates, a name under the bar, verdict PASS.

    Every reading is asserted on its printed line — the shadow-date count, the name with
    its count and its rate, the enforce-refusal count — and the report is asserted to carry
    EXACTLY ONE line of each verdict kind: `verdict: PRE-FLIP PASS` (the four bars) and
    `verdict: POST-FLIP HEALTH PASS` (what enforcement has refused), one apiece, in that
    order. A report with a table but no verdict would leave the owed-check job re-deriving
    the decision this script exists to make; a report with two PRE-FLIP lines would let a
    grep take the wrong one.
    """
    rc, out = _run(tmp_path, _passing_rows())        # 42 rows, 7 dates, 2 strips
    assert rc == 0, out
    assert _verdict_lines(out) == ["verdict: PASS"], out
    assert _post_flip_verdict(out) == "verdict: POST-FLIP HEALTH PASS", out
    # Order compared between the TWO VERDICT LINES, not between the first mention of each
    # word: the word POST-FLIP also appears in the parenthetical that explains why a
    # zero-enforce reading's post-flip pass is vacuous, which sits above both verdicts.
    lines = out.splitlines()
    assert lines.index("verdict: PRE-FLIP PASS") < lines.index(
        "verdict: POST-FLIP HEALTH PASS"), \
        "the pre-flip verdict must come first: it is the one the exit code and the " \
        "stored reading's `verdict` field have always meant"
    assert "shadow dates: 7" in out
    assert "rows: 42" in out
    name_line = [ln for ln in out.splitlines() if "automod_vault_land" in ln]
    assert len(name_line) == 1, out
    assert "   2   " in name_line[0] and "4.8%" in name_line[0], name_line[0]
    assert "OVER" not in out, "4.8% is under the bar"
    assert "rows refusing a name: 0" in out
    assert "  ! " not in out, f"a PASS must carry no failure lines:\n{out}"


def test_an_enforce_refusal_left_the_pre_flip_bars_and_moves_the_post_flip_one(tmp_path):
    """The fifth bar this file used to hold is now the post-flip verdict's first rule.

    Before #3415 an enforce row that refused anything failed the PRE-FLIP verdict outright
    ("1 enforce-mode row(s) refused a tool name"). That was right while the flip was not
    taken — a stray enforce row meant something had gone wrong — and became wrong the day it
    was: the first refusal of an unlicensed name would have reported FAIL on the ceiling
    doing its job. So the same row now leaves `verdict: PRE-FLIP` alone and drives
    `verdict: POST-FLIP HEALTH`, through the refusal rate (1 of 1 enforce rows = 100.0% >
    5.0%), which is the bar #2670 set applied to the enforce side.

    The two lines are asserted together on purpose: deleting either rule could satisfy one
    assertion and break the other.
    """
    rows = _passing_rows() + [
        _row(DATES[-1], mode="enforce", task_id=99, names=["djev_status"])]
    m = gate.measure(rows, licensed_pairs=set())
    out = gate.format_report(m)
    assert _verdict_lines(out) == ["verdict: PASS"], \
        f"the enforce refusal leaked back into the pre-flip bars:\n{out}"
    assert _post_flip_verdict(out) == "verdict: POST-FLIP HEALTH FAIL", out
    assert "enforce refusal rate 100.0% (1 of 1 enforce rows) > 5.0%" in out, out
    assert "no live grant for that scope" in out, \
        f"the refusal must be attributed to its scope and licensing:\n{out}"


def test_the_rate_is_over_every_row_not_over_the_rows_that_stripped(tmp_path):
    """Clause 1's denominator, pinned the only way it can be: with a wrong one.

    The per-name rate is over the TOTAL row count. The tempting alternative — turns that
    reached for anything at all — differs from the real figure by an order of magnitude
    at exactly the number the flip is decided on: the live ledger today has one
    would-strip name in a small minority of its turns, and dividing only by those turns
    would print 100.0% for it. One stripped row in 42 must read 2.4%, and must never read
    100.0%.
    """
    rows = [dict(r, would_strip=[], n_would_strip=0) for r in _passing_rows()]
    rows[0]["would_strip"] = ["automod_vault_land"]
    rows[0]["n_would_strip"] = 1
    assert len(rows) == 42
    rc, out = _run(tmp_path, rows)
    assert rc == 0, f"1 of 42 is 2.4%, under the bar, and all seven dates are present:\n{out}"
    name_line = [ln for ln in out.splitlines() if "automod_vault_land" in ln]
    assert len(name_line) == 1, out
    assert "2.4%" in name_line[0], name_line[0]
    assert "100.0%" not in out, (
        "the rate was taken over the rows that stripped, not over all rows — the figure "
        "the 5% bar is defined against")


def test_each_bar_fails_alone_and_the_reason_names_the_number(tmp_path):
    """Four bars, each able to fail by itself: the verdict is traceable to its reading.

    The PASS arm above and these fixtures differ by one property each, so no bar can be
    satisfied by an accident of the fixture and none can be dropped without one of these
    going red. The under-bar arm inside the loop is the paired control: a bar wired `>=`
    instead of `>` fails a name at exactly 5.0%-ish, and without it the four failures
    above would read as though the bar were merely strict.
    """
    six_dates = [r for r in _passing_rows(strip_dates=()) if r["date"] != DATES[-1]]
    exactly_under = _passing_rows(strip_dates=(DATES[1], DATES[2]))   # 2/42 = 4.8%
    over_bar = _passing_rows() + [
        _row("2026-10-18", task_id=97, names=["djev_status"])] * 3     # 3/45 = 6.7%
    unmeasured = _passing_rows() + [
        _row(DATES[-1], task_id=98, compile_error="RuntimeError: boom")]

    assert _verdict_lines(gate.format_report(gate.measure(exactly_under))) == [
        "verdict: PASS"], "2 of 42 rows is 4.8% — at-or-under the bar is allowed"

    for rows, expected in (
        (six_dates, "shadow dates 6 < 7 required"),
        (over_bar, "djev_status reached in 3 of 45 rows = 6.7% > 5.0%"),
        (unmeasured, "1 row(s) carry no would_strip measurement "
                     "(1 naming a compile_error)"),
    ):
        out = gate.format_report(gate.measure(rows))
        assert _verdict_lines(out) == ["verdict: FAIL"], f"{expected!r}:\n{out}"
        assert expected in out, f"reason not printed for {expected!r}:\n{out}"


def test_the_table_marks_the_name_that_is_over_and_orders_by_count(tmp_path):
    """The OVER marker fires when it should and not when it should not, and the order is stable.

    A column that never prints is decoration. Two names tied at 2 and one at 1 also pin
    the tie-break (count descending, then name ascending): a table whose order moved on a
    tie would show a daily diff where nothing changed, which is how a real change gets
    missed.
    """
    rows = [_row(d, task_id=40 + i, names=[]) for d in DATES for i in range(6)]  # 42
    rows += [_row(DATES[0], task_id=74, names=["automod_vault_land"]),
             _row(DATES[1], task_id=86, names=["automod_status"]),
             _row(DATES[2], task_id=86, names=["automod_status"]),
             _row(DATES[1], task_id=47, names=["autonomy_write_task"]),
             _row(DATES[2], task_id=47, names=["autonomy_write_task"])]
    assert len(rows) == 47
    out = gate.format_report(gate.measure(rows))
    assert _table(out) == ["automod_status", "autonomy_write_task",
                           "automod_vault_land"], out
    assert "OVER" not in out, out                     # 2/47 = 4.3%, 1/47 = 2.1%

    hot = [_row(DATES[3], task_id=i, names=["automod_vault_land"])
           for i in range(5)] + rows                  # 6/52 = 11.5%
    out2 = gate.format_report(gate.measure(hot))
    over_lines = [ln for ln in out2.splitlines() if "OVER" in ln]
    assert len(over_lines) == 1, out2
    assert "automod_vault_land" in over_lines[0], over_lines
    assert "11.5%" in over_lines[0], over_lines


def test_no_arguments_reads_the_ledger_the_writer_writes_to(tmp_path, monkeypatch):
    """The seam node: `main([])` resolves the file through the module's own reader.

    Two things at once. First, the flag is optional — the script "runs with no
    arguments" — and that path is the same `ledger_path()` the writer in
    `app/scheduled_ceiling` resolves, exercised through `LLOYD_SCHEDULED_CEILING_LEDGER`
    instead of against the live store. Second, the bytes come back through `read_rows`
    and not a private parse: a half-written tail line is skipped and the intact rows
    still produce the reading. A reader that re-implemented the parse would either die on
    that line or count it as a row that stripped nothing, which silently enlarges the
    denominator every rate divides by.
    """
    rc, out = _run(tmp_path / "seam", _passing_rows(), tail_junk=True,
                   argv=[], monkeypatch=monkeypatch)
    assert rc == 0, out
    assert _verdict_lines(out) == ["verdict: PASS"], out
    assert "rows: 42" in out, (
        "the truncated tail line must be skipped, not counted as a turn that stripped "
        "nothing — it would move every rate in the report")
    p = tmp_path / "seam" / "scheduled-ceiling.jsonl"
    assert str(p) in out, "the report has to name the file it read"
    rc2, out2 = _run_at(p)                            # the --ledger route, same bytes
    assert (rc2, out2) == (rc, out), "the two routes must read identically"


def test_an_empty_or_absent_ledger_is_a_fail_and_never_a_clean_read(tmp_path):
    """No rows is not "nothing reached outside the ceiling" — it is "nothing measured".

    The class of bug this box keeps meeting: a guard that reads its own missing input and
    reports the verdict that input implies. An empty ledger, a file that is not there, and
    a fixture whose rows all carry an empty `would_strip` look identical to a naive rate
    loop, and only the third deserves a PASS.
    """
    rc, out = _run(tmp_path / "empty", [])
    assert rc == 1, out
    assert _verdict_lines(out) == ["verdict: FAIL"], out
    assert "the ledger holds 0 rows" in out
    assert "shadow dates 0 < 7 required" in out

    missing = tmp_path / "absent" / "scheduled-ceiling.jsonl"
    assert not missing.exists()
    rc2, out2 = _run_at(missing)
    # Named, not just "no PASS": the report now also carries `verdict: POST-FLIP HEALTH
    # PASS`, which is a VACUOUS pass on a ledger with no enforce rows and must not be
    # mistaken for the bars passing.
    assert rc2 == 1 and "verdict: PRE-FLIP PASS" not in out2, out2

    clean = [_row(d, task_id=i, names=[]) for i, d in enumerate(DATES)]
    rc3, out3 = _run(tmp_path / "clean", clean)
    assert rc3 == 0, out3
    assert "(none — no row recorded a name outside the ceiling)" in out3, out3


def test_an_enforce_row_that_stripped_nothing_is_not_a_refusal(tmp_path):
    """The two enforce readings are separate numbers, and neither may borrow the other.

    Post-flip, `rows refusing a name` is the line that says enforcement is hurting jobs,
    so it has to count rows whose name list is non-EMPTY — an enforced turn that stayed
    inside its ceiling is the good case and must not be reported as a refusal. Same for
    the dates: this fixture's eighth date is enforce-only, so `shadow dates` must stay at
    7. A reader that counted every mode's date would print 8 and hand the flip its own
    missing day, which is the bar #2670 set in shadow traffic.
    """
    rows = _passing_rows() + [
        _row("2026-10-18", mode="enforce", task_id=99, names=[])]
    rc, out = _run(tmp_path, rows)
    assert rc == 0, out
    assert _verdict_lines(out) == ["verdict: PASS"], out
    assert "enforce rows: 1" in out, out
    assert "rows refusing a name: 0" in out, (
        "a clean enforce row was counted as a refusal — the post-flip line would cry "
        "wolf on every healthy turn")
    assert "shadow dates: 7" in out, (
        "an enforce-only date must not count toward the seven-day shadow bar")


@pytest.mark.parametrize("drop", [
    pytest.param("date", id="a_row_without_a_date_never_buys_a_shadow_day"),
    pytest.param("mode", id="a_row_with_no_mode_is_not_shadow"),
])
def test_a_malformed_row_cannot_buy_a_shadow_date(tmp_path, drop):
    """The date bar is the flip's only time-based defence, so a missing key adds nothing.

    A shadow date counts only when the row says `mode: shadow` AND carries a `date`. Strip
    either from seven otherwise-passing dates and the reading has to fall to 0, not stay
    at 7: the alternative is a ledger from a writer that renamed a field printing a PASS
    on days it never recorded.
    """
    rows = _passing_rows()
    for r in rows:
        r.pop(drop)
    rc, out = _run(tmp_path / f"drop-{drop}", rows)
    assert rc == 1, out
    assert "shadow dates: 0" in out
    assert "shadow dates 0 < 7 required" in out, out


# ─────────────────── the recorder: a verdict that outlives the run ───────────────────
#
# Clause map for #3195, so a reader (or a grader) can match each acceptance clause to the
# node that pins it without reading the section twice:
#
#   clause 1 — one row per run, beside the ledger, full reading, stdout/rc unchanged:
#              test_record_appends_one_row_per_run_beside_the_ledger_and_prints_nothing_extra
#              test_the_command_line_form_honours_the_flags_the_report_names  (the same flag
#              through a real `python -m`, which is how a daily caller invokes it)
#   clause 2 — a FAIL is recorded, not escalated; a PASS records too; nothing else written:
#              test_a_failing_verdict_is_recorded_not_escalated_and_a_passing_one_is_recorded_too
#              test_a_reading_that_cannot_be_written_is_said_loudly_and_costs_nothing
#   clause 3 — the row-level refusal count is printed beside the 5% table:
#              test_the_report_prints_how_many_rows_would_refuse_under_enforce
#              test_the_predictor_counts_rows_not_names_and_survives_an_empty_ledger


def _snapshot(d: Path) -> dict:
    """Name → bytes for every file in a directory, so a write has to SHOW UP.

    Clause 2 says the recorder writes nothing but its own readings file. The only way
    to pin an absence of a write is to diff the directory before and after, so that is
    what this returns; a node that merely asserted the readings file existed would also
    pass if the recorder rewrote the ledger, which is the one write here that could
    destroy evidence rather than create it.
    """
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()}


def test_record_appends_one_row_per_run_beside_the_ledger_and_prints_nothing_extra(
        tmp_path, monkeypatch):
    """Clause 1: `--record` durably appends the reading, and changes no other output.

    Three properties that can each fail alone.

      * **Where.** The readings file is the sibling of `app.scheduled_ceiling.ledger_path()`,
        so redirecting the ledger through `LLOYD_SCHEDULED_CEILING_LEDGER` — the knob every
        node in this file already uses — redirects the readings too. A second, independent
        env var would be a second thing to get wrong on a test machine and would let a
        recorded reading land somewhere no later reader thinks to look.
      * **What.** The row carries the reading, not a summary of it: `ts`, `total_rows`, the
        mode counts, the shadow dates, the per-name count/rate list, `enforce_rows`,
        `enforce_refusals`, the failure strings VERBATIM and the verdict. The pre-flip
        reading cannot be recomputed after the flip, because the ledger gains `enforce` rows
        and so the denominator every rate divides by moves; a row that stored only the
        verdict would not answer the question the flip round will be asked later.
      * **Nothing else.** Stdout and the exit code from a `--record` run must be byte-for-byte
        what the same run without the flag produced. This script's stdout is already a
        machine-read artifact — the owed-check job takes the decision off the single
        `verdict:` line — so a record that also printed a confirmation would be a change to
        the consumer under a flag that claims only to save a copy.
    """
    led = _write_ledger(tmp_path, _passing_rows())          # 42 rows, 7 dates, 2 strips
    monkeypatch.setenv(sc.LEDGER_ENV, str(led))

    assert gate.readings_path() == sc.ledger_path().parent / gate.READINGS_FILENAME
    assert gate.readings_path() == tmp_path / "flip-gate-readings.jsonl", (
        "the readings file must sit in the directory ledger_path() resolves, which is "
        "the directory a test redirects and the data root otherwise")
    assert not gate.readings_path().exists(), "the file is created by the run, not before"

    started = dt.datetime.now(dt.timezone.utc)
    rc_plain, out_plain = _run_at(led, argv=[], monkeypatch=monkeypatch)
    rc1, out1 = _run_at(led, argv=["--record"], monkeypatch=monkeypatch)
    rc2, out2 = _run_at(led, argv=["--record"], monkeypatch=monkeypatch)

    assert (rc1, out1) == (rc_plain, out_plain), (
        "--record changed the printed report or the exit code; the report is the "
        "consumer-facing artifact and the flag is only supposed to copy it")
    assert rc2 == rc1 and out2 == out1
    assert _verdict_lines(out1) == ["verdict: PASS"], out1

    recorded = [json.loads(ln) for ln in
                gate.readings_path().read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(recorded) == 2, f"one row per run, so two runs → two rows, got {recorded}"

    row = recorded[1]
    m = gate.measure(sc.read_rows(led))
    for key in ("ts", "total_rows", "modes", "shadow_dates", "names", "enforce_rows",
                "enforce_refusals", "failures", "verdict"):
        assert key in row, f"recorded row is missing {key!r}: {sorted(row)}"
    assert row["total_rows"] == 42 and row["verdict"] == "PASS", row
    assert row["shadow_dates"] == DATES, row            # all seven fixture dates, in order
    assert row["names"] == m["names"], "the per-name list must be the reading itself"
    assert row["enforce_rows"] == 0 and row["enforce_refusals"] == 0, row
    assert row["failures"] == [], row
    assert row["modes"] == {"shadow": 42}, row

    # The stamp is UTC-aware AND is THIS run's. A naive local stamp in a machine-facing
    # row is read as UTC by every later reader, and a stamp with only an upper bound is
    # no bound at all: a hard-coded literal, or a value copied from the row before, would
    # satisfy it, and a readings file of stale timestamps is worse than none — it looks
    # like a daily recorder that has been running.
    stamped = dt.datetime.fromisoformat(row["ts"])
    assert stamped.tzinfo is not None and stamped.utcoffset() == dt.timedelta(0), row["ts"]
    assert started - dt.timedelta(seconds=30) <= stamped \
        <= started + dt.timedelta(minutes=5), (row["ts"], started.isoformat())


def test_a_failing_verdict_is_recorded_not_escalated_and_a_passing_one_is_recorded_too(
        tmp_path, monkeypatch):
    """Clause 2: the recorder survives its own red, and records both verdicts alike.

    A FAIL is the case the recorder exists for: this script's exit status is already a
    notification (a daily check goes red on it), so a `--record` that also raised, or
    that decided a red reading was worth escalating into a write somewhere else, would
    turn an observation into an actor. The paired PASS arm is what makes that
    non-redundant — a recorder that only ever ran on a green ledger would be a
    success-only log, and the reading this item is chasing is a red one.
    """
    hot = _passing_rows() + [                              # 3 more of one name: 3/45 = 6.7%
        _row("2026-10-18", task_id=97, names=["djev_status"])] * 3
    led = _write_ledger(tmp_path, hot)
    monkeypatch.setenv(sc.LEDGER_ENV, str(led))
    before = _snapshot(tmp_path)

    rc, out = _run_at(led, argv=["--record"], monkeypatch=monkeypatch)   # must not raise

    assert rc == 1, out
    assert _verdict_lines(out) == ["verdict: FAIL"], out
    after = _snapshot(tmp_path)
    assert set(after) - set(before) == {gate.READINGS_FILENAME}, (
        f"--record wrote outside its readings file: {sorted(set(after) - set(before))}")
    assert after["scheduled-ceiling.jsonl"] == before["scheduled-ceiling.jsonl"], (
        "the ledger is the witness; a recorder that touched it would be editing its "
        "own evidence")

    rows = [json.loads(ln) for ln in
            gate.readings_path().read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1, rows
    assert rows[0]["verdict"] == "FAIL", rows[0]
    assert rows[0]["failures"] == gate.measure(hot)["failures"], (
        "the failure strings must be recorded verbatim — a paraphrase cannot be "
        "re-checked against the code that made it")
    assert any("6.7% > 5.0%" in f for f in rows[0]["failures"]), rows[0]["failures"]

    # The PASS arm: same flag, same one row, verdict PASS. A reading exists either way.
    ok_led = _write_ledger(tmp_path / "pass", _passing_rows())
    monkeypatch.setenv(sc.LEDGER_ENV, str(ok_led))
    rc2, out2 = _run_at(ok_led, argv=["--record"], monkeypatch=monkeypatch)
    assert rc2 == 0, out2
    ok_rows = [json.loads(ln) for ln in
               gate.readings_path().read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(ok_rows) == 1 and ok_rows[0]["verdict"] == "PASS", ok_rows


def test_a_reading_that_cannot_be_written_is_said_loudly_and_costs_nothing(
        tmp_path, monkeypatch):
    """A lost record is visible, and it does not move the verdict — clause 2's other edge.

    Clause 2 fixes the exit code to the verdict and forbids raising, so a failed write may
    not become an exit 2; but it may not be silent either. A daily caller whose readings
    file never appears reads as a fleet that never ran, which is the same false-clean
    reading the empty-ledger node exists to refuse. So the loss goes to stderr — not the
    report, whose `verdict:` line must stay byte-identical to the run that wrote fine.
    """
    led = _write_ledger(tmp_path, _passing_rows())
    monkeypatch.setenv(sc.LEDGER_ENV, str(led))

    def _boom(*_a, **_k):
        raise OSError("Read-only file system")

    monkeypatch.setattr(gate, "record_reading", _boom)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc, out = _run_at(led, argv=["--record"], monkeypatch=monkeypatch)   # must not raise

    assert rc == 0, f"a write failure changed the exit code:\n{out}"
    assert _verdict_lines(out) == ["verdict: PASS"], out
    assert not gate.readings_path().exists(), "nothing was written, so nothing appeared"
    assert "wrote no reading" in err.getvalue(), err.getvalue()
    assert gate.READINGS_FILENAME in err.getvalue(), err.getvalue()


#: Repo root, so a subprocess node can run the module the way an operator does.
REPO = Path(__file__).resolve().parents[1]


def test_the_command_line_form_honours_the_flags_the_report_names(tmp_path):
    """The operator's seam is a shell, so the flags are tested through a shell's proxy.

    Every other node calls `main([...])` with an explicit argv list, which is exactly why
    the no-argv path could parse an EMPTY argv for all its life without a single red node:
    `python -m scripts.maintenance.scheduled_ceiling_flip_gate --record` then dropped
    `--record` on the floor — no error, and the exit code still looked like the verdict —
    while `--ledger /tmp/copy.jsonl` read the LIVE store behind a report that printed the
    copy's name. A silently ignored flag is the worst kind: the operator gets a confident
    answer from the wrong file. So this node crosses the real process boundary — `sys.
    executable -m`, `cwd=REPO`, flags in `argv`, the ledger redirect in the ENVIRONMENT —
    and reads back what the child process left on disk.

    Both flags are pinned at once, against different files, because only two different
    files can tell them apart: the env knob says 42 passing rows, `--ledger` says 45 rows
    with a name over the bar. Whichever one the child actually read shows up in the exit
    code, and only the copy produces a 1.
    """
    env_ledger = _write_ledger(tmp_path / "env", _passing_rows())        # 42 → PASS
    copy = _write_ledger(tmp_path / "copy", _passing_rows() + [          # 45 → 6.7% FAIL
        _row("2026-10-18", task_id=97, names=["djev_status"])] * 3)
    env = {**os.environ, sc.LEDGER_ENV: str(env_ledger)}

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.maintenance.scheduled_ceiling_flip_gate",
         "--ledger", str(copy), "--record"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180)

    assert proc.returncode == 1, (
        f"--ledger was ignored, so the child read the 42-row store and passed:\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    assert f"ledger: {copy}" in proc.stdout, proc.stdout[-2000:]
    # Named with its label: since #3415 the report carries two verdicts, and the bare
    # `verdict: FAIL` this asserted is no longer a substring of either line. The copy is a
    # shadow-only ledger, so the half this node is about — the one the four bars decide — is
    # the PRE-FLIP line.
    assert "verdict: PRE-FLIP FAIL" in proc.stdout, proc.stdout[-2000:]
    # 2 strips from the passing fixture + 3 over-bar rows = 5 of the 45 the COPY holds.
    # A row-level predictor doubles as proof of which file was read: the env store would
    # have printed 2 of 42.
    assert "rows carrying a would-strip name: 5 of 45 = 11.1%" in proc.stdout, \
        proc.stdout[-2000:]

    readings = tmp_path / "env" / gate.READINGS_FILENAME
    assert readings.exists(), (
        f"--record was ignored by the CLI:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    rows = [json.loads(ln) for ln in
            readings.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1, rows
    assert rows[0]["verdict"] == "FAIL" and rows[0]["total_rows"] == 45, rows[0]
    assert rows[0]["ledger"] == str(copy), (
        "the row must say which file it measured — the readings live beside the "
        "env-resolved ledger, so the field is the only thing distinguishing a copy "
        "from the live store")


# ───────────────── the enforce predictor: the number the table hides ─────────────────

def _many_names_under_bar(per_date: int = 6) -> list[dict]:
    """42 rows over 7 dates, 8 of them stripping — but 4 names at 2 each, so 4.8% apiece.

    Every per-name rate sits under the 5% bar while 8 rows still carry a name. That
    arrangement is the whole reason this reading is printed: it is a PASS-shaped ledger
    that under enforce refuses a tool name on one turn in five.
    """
    names = ["automod_vault_land", "autonomy_write_task", "automod_status", "djev_status"]
    rows = [_row(d, task_id=40 + i, names=[]) for d in DATES for i in range(per_date)]
    for slot, name in enumerate(names):
        rows[slot * 2]["would_strip"] = [name]
        rows[slot * 2]["n_would_strip"] = 1
        rows[slot * 2 + 1]["would_strip"] = [name]
        rows[slot * 2 + 1]["n_would_strip"] = 1
    return rows


def test_the_report_prints_how_many_rows_would_refuse_under_enforce(tmp_path):
    """Clause 3: the row-level refusal count is printed, not left to arithmetic.

    The 5% table answers a different question from the one the flip is gated on. It
    divides EACH name by the total, so four names at 2 rows each all read 4.8% and the
    ledger prints PASS — while 8 of those 42 rows (19.0%) carry a name that under
    `LLOYD_SCHEDULED_CEILING_MODE=enforce` is a refused tool call. Deriving 19.0% from
    the printed table is not possible at all (it needs row-level, not name-level,
    counts), and the live ledger measured 33 of 354 = 9.3% on 2026-10-23 with no name
    above 6.5%, so the gap is not hypothetical. Hence a line of its own.
    """
    rows = _many_names_under_bar()
    assert len(rows) == 42
    m = gate.measure(rows)
    assert m["rows_with_strip"] == 8, m["names"]
    assert all(n["rate"] <= gate.NAME_RATE_CEILING for n in m["names"]), m["names"]

    out = gate.format_report(m)
    line = [ln for ln in out.splitlines() if "carrying a would-strip name" in ln]
    assert len(line) == 1, out
    assert "8 of 42" in line[0] and "19.0%" in line[0], line[0]
    assert "under enforce" in line[0] and "refused a tool name" in line[0], line[0]
    assert max(n["rate"] for n in m["names"]) * 100 < 19.0, (
        "the predictor is only worth printing if it can exceed the table's own rates")
    assert _verdict_lines(out) == ["verdict: PASS"], (
        "the predictor is a reading, not a fifth bar — it must not flip the verdict")

    rc, out2 = _run(tmp_path, rows)                       # same numbers through main()
    assert rc == 0, out2
    assert [ln for ln in out2.splitlines() if "8 of 42" in ln], out2
    assert "rows_with_strip" not in out2                   # the field name is not the line
    assert out2.splitlines().index([ln for ln in out2.splitlines()
                                    if "8 of 42" in ln][0]) < \
        out2.splitlines().index("verdict: PRE-FLIP PASS"), \
        "the predictor sits above the table but below the verdict it predicts"


def test_the_predictor_counts_rows_not_names_and_survives_an_empty_ledger(tmp_path):
    """The predictor is a row count, so one row with three names is ONE refusal.

    A turn that reached for three excluded tools is one turn that lost three tools, but
    it is not three turns; the flip's damage estimate is about how many scheduled jobs
    get hurt, which is the row. An unmeasured row (`would_strip` absent because the
    compile failed) is neither — it measured nothing, so it counts as 0, and the
    `unmeasured` reading already reports it. The empty-ledger arm is the same
    missing-input guard the rates get: 0 of 0 must print 0.0% and not divide.
    """
    rows = _passing_rows(strip_dates=())
    rows[0]["would_strip"] = ["automod_vault_land", "automod_status", "djev_rank"]
    rows[0]["n_would_strip"] = 3
    m = gate.measure(rows)
    assert m["rows_with_strip"] == 1, m          # three names, ONE refusing row
    assert len(m["names"]) == 3, m

    out = gate.format_report(m)
    assert "1 of 42" in out and "2.4%" in out, out

    unmeasured = _passing_rows(strip_dates=()) + [
        _row(DATES[-1], task_id=98, compile_error="RuntimeError: boom")]
    m2 = gate.measure(unmeasured)
    assert m2["rows_with_strip"] == 0, (
        "a row that never measured may not be counted as a row that would refuse")
    assert "0 of 43" in gate.format_report(m2)

    rc, out3 = _run(tmp_path / "empty-predictor", [])
    assert rc == 1, out3
    assert "0 of 0" in out3 and "0.0%" in out3, out3
