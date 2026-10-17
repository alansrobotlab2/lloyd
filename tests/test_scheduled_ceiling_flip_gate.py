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
import io
import json
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
    return [ln for ln in out.splitlines() if ln.startswith(gate.VERDICT_PREFIX)]


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
    its count and its rate, the enforce-refusal count — and the report is asserted to
    carry EXACTLY one `verdict:` line. A second one would let a later grep take the wrong
    one, and a report with a table but no verdict would leave the owed-check job
    re-deriving the decision this script exists to make.
    """
    rc, out = _run(tmp_path, _passing_rows())        # 42 rows, 7 dates, 2 strips
    assert rc == 0, out
    assert _verdict_lines(out) == ["verdict: PASS"], out
    assert "shadow dates: 7" in out
    assert "rows: 42" in out
    name_line = [ln for ln in out.splitlines() if "automod_vault_land" in ln]
    assert len(name_line) == 1, out
    assert "   2   " in name_line[0] and "4.8%" in name_line[0], name_line[0]
    assert "OVER" not in out, "4.8% is under the bar"
    assert "rows refusing a name: 0" in out
    assert "  ! " not in out, f"a PASS must carry no failure lines:\n{out}"


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
        _row("2026-10-18", task_id=97, names=["djem_status"])] * 3     # 3/45 = 6.7%
    enforce_refused = _passing_rows() + [
        _row(DATES[-1], mode="enforce", task_id=99, names=["djem_status"])]
    unmeasured = _passing_rows() + [
        _row(DATES[-1], task_id=98, compile_error="RuntimeError: boom")]

    assert _verdict_lines(gate.format_report(gate.measure(exactly_under))) == [
        "verdict: PASS"], "2 of 42 rows is 4.8% — at-or-under the bar is allowed"

    for rows, expected in (
        (six_dates, "shadow dates 6 < 7 required"),
        (over_bar, "djem_status reached in 3 of 45 rows = 6.7% > 5.0%"),
        (enforce_refused, "1 enforce-mode row(s) refused a tool name"),
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
    assert rc2 == 1 and "verdict: PASS" not in out2, out2

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
