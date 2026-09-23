"""Backlog #835 — `iv_grade.py --since` selects by the clock, not by the separator.

`inner_voice_observations.created_at` is written local-naive ISO with a `T`
(`usage_store.record_inner_voice_observation`). A bound written any other way —
`--since 2026-09-16`, or the `YYYY-MM-DD HH:MM:SS` form SQLite's `datetime('now')`
renders, or `date '+%Y-%m-%d %H:%M:%S'` — carries a *space* where every row carries a
`T`. SQLite compares TEXT with BINARY collation, so `T` (0x54) sorts above space (0x20)
and the comparison is settled at the separator without ever reading the hour. Measured
on the live table at triage (2026-09-16), one instant written two ways through the
shipped CLI: `'2026-09-16 04:00:00'` → 170 observations (its own `scope.first` was
`2026-09-16T00:16:07.490501`, i.e. rows 3 h 44 m *before* the bound were selected);
`'2026-09-16T04:00:00'` → 4. The `datetime('now','-3 hours')` form returned 170 against
an honest 119.

So this file runs the shipped script as a subprocess against a throwaway `usage.db` —
the table's real schema, rows in the live `T` format — and pins:

1. a space-form bound excludes rows by their TIME, not by their shape;
2. the two separator forms of one instant can never disagree again;
3. the window is decided over a MULTI-DAY range, where the failure is a whole
   extra day rather than one hour — the shape the nightly's 26-hour bound lives on;
4. `--help` names the clock `--since` is interpreted in, and the module docstring's
   usage example is a full timestamp rather than a bare date.

The writer is deliberately untouched: `created_at` stays local-naive because the
frontend timeline merge reads it as local time (`usage_store.py:339-342`), and clause 5
of the acceptance lives in `tests/integration/test_iv_guards.py`
(`test_observation_rows_keep_the_local_naive_writer_clock`).

Run: .venvs/lloyd/bin/python -m pytest tests/test_iv_grade_window.py -q
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRADER = ROOT / "scripts" / "iv_grade.py"

#: Copied verbatim from the live table quoted in #835 — the two rows that *are*
#: the discriminating pair at the `04:00:00` bound, sub-second precision included.
LIVE_FORMAT_ROWS = ("2026-09-16T00:16:07.490501", "2026-09-16T04:37:14.904432")


def _make_repo(tmp_path: Path) -> Path:
    """A throwaway repo root holding a copy of the grader and its own `usage.db`.

    Copied, not symlinked: `iv_grade.py` anchors `DB_PATH` at
    the data root of `Path(__file__).resolve().parents[1]`, and a symlink resolves back to the real
    checkout — which would have the suite reading (and this file asserting against)
    the production table instead of the fixture.
    """
    repo = tmp_path / "lloyd"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(GRADER, repo / "scripts" / "iv_grade.py")
    # The grader reads `usage.db` through `app.paths`; a fixture repo that is not
    # the live checkout keeps its data under `<repo>/.lloyd-data` (rule 3).
    (repo / "app").mkdir()
    (repo / "app" / "__init__.py").write_text("")
    shutil.copy2(GRADER.parents[1] / "app" / "paths.py", repo / "app" / "paths.py")
    (repo / ".lloyd-data").mkdir()
    return repo


@pytest.fixture(autouse=True)
def _fixture_repo_owns_its_data(monkeypatch):
    """The copied grader must resolve `usage.db` inside the fixture repo, so the
    suite's own `LLOYD_DATA` is not handed to the child."""
    monkeypatch.delenv("LLOYD_DATA", raising=False)


def _make_db(repo: Path, created_at_values: list[str]) -> Path:
    """A fixture `usage.db` with the live `inner_voice_observations` schema.

    `created_at` is inserted as given, so the stored bytes are exactly what is under
    test — including the `T` separator. Timestamps come in as strings for the same
    reason: a `datetime` object would be formatted by `str()`, and a format change in
    Python would quietly change what the fixture stores.
    """
    db = repo / ".lloyd-data" / "usage.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE inner_voice_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        turn_id TEXT NOT NULL, sequence_in_turn INTEGER NOT NULL,
        trigger TEXT NOT NULL, action TEXT NOT NULL, reason TEXT, content TEXT,
        related_tool TEXT, input_tokens INTEGER, output_tokens INTEGER,
        cache_read INTEGER, cache_create INTEGER, latency_ms INTEGER,
        model TEXT, error TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    for i, stamp in enumerate(created_at_values, start=1):
        conn.execute(
            """INSERT INTO inner_voice_observations
               (session_id, turn_id, sequence_in_turn, trigger, action, reason,
                input_tokens, output_tokens, cache_read, cache_create,
                latency_ms, model, error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("s1", f"t{i}", i, "assistant_message", "noop", "fixture row",
             1000, 5, 0, 0, 400, "primary", None, stamp))
    conn.commit()
    conn.close()
    return db


def _grade(repo: Path, since: str) -> dict:
    """One real invocation of the shipped CLI. Returns its parsed `--json` report."""
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "iv_grade.py"),
         "--json", "--since", since],
        capture_output=True, text=True, check=False, cwd=str(repo),
    )
    assert proc.returncode == 0, (
        f"iv_grade.py --json --since {since!r} exited {proc.returncode}: "
        f"stdout={proc.stdout[:200]!r} stderr={proc.stderr[:300]!r}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            f"--json printed no JSON for --since {since!r}: {proc.stdout[:200]!r}")


def _live_format_repo(tmp_path: Path) -> Path:
    """Fixture carrying the two rows quoted out of the live table."""
    repo = _make_repo(tmp_path)
    _make_db(repo, list(LIVE_FORMAT_ROWS))
    return repo


# ---------------------------------------------------------------------------
# clause 1 — a space-separated bound excludes rows by their time
# ---------------------------------------------------------------------------


def test_space_form_bound_excludes_the_earlier_row_by_time(tmp_path):
    """`--since '<date> 04:00:00'` reports 1 observation, not 2.

    The two fixture rows are the live pair: `00:16:07.490501` sits before the bound,
    `04:37:14.904432` after it, so the honest answer is 1. Before the fix the space
    in the bound settled the comparison before the hour was read, every row of that
    day compared "greater", and the script reported 2 here. On the live table the same
    defect counted 170 observations for a bound whose honest count was 119 (#835,
    2026-09-16) — the inflation, not this fixture.
    """
    repo = _live_format_repo(tmp_path)

    report = _grade(repo, "2026-09-16 04:00:00")

    assert report["cost"]["observations"] == 1, (
        "the space-separated bound selected both fixture rows: the comparison is "
        "still being decided by the 'T' separator rather than by the clock")
    assert report["scope"]["first"] == LIVE_FORMAT_ROWS[1], (
        "the surviving row is not the one after the bound")


def test_a_row_exactly_at_the_space_form_bound_is_kept(tmp_path):
    """The window is `>=`, so the instant equal to the bound is inside it.

    Pinned next to the clause-1 case because a "fix" that normalised into a strict
    `>` — or that truncated the bound to its date — would also bring the count from
    2 down to 1 and pass the clause above.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, ["2026-09-16T03:00:00.000000", "2026-09-16T04:00:00.000000"])

    assert _grade(repo, "2026-09-16 04:00:00")["cost"]["observations"] == 1
    assert _grade(repo, "2026-09-16T04:00:00")["cost"]["observations"] == 1


def test_a_bare_date_bound_still_keeps_the_whole_day(tmp_path):
    """The usage example's shape: `--since 2026-09-16` is local and inclusive.

    A stored `T` sorts above a space, so a date-only bound behaved correctly against
    `T` rows by lexical accident (#835 calls out exactly this). After normalising,
    `'2026-09-16 05:00:00' >= '2026-09-16'` is still true on the longer string, so
    the day stays inclusive and the fix does not silently narrow the shape that
    `--since 2026-08-01` has always had.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, ["2026-09-15T23:59:59.000000", "2026-09-16T00:00:00.000000",
                    "2026-09-16T23:59:59.999999"])

    assert _grade(repo, "2026-09-16")["cost"]["observations"] == 2


# ---------------------------------------------------------------------------
# clause 2 — the two separator forms of one instant cannot disagree
# ---------------------------------------------------------------------------


def test_space_and_T_forms_of_one_bound_report_the_same_window(tmp_path):
    """`cost.observations`, `scope.first` and `scope.last` are identical.

    This is the acceptance check itself: the two forms reported 170 vs 4 across a
    22-hour span on the live table. Compared on all three fields rather than the
    count alone, because two windows with equal counts and different endpoints
    (`scope.first`/`scope.last` are what the nightly's coverage columns are built
    from) are precisely the disagreement that would survive a count-only pin.
    """
    repo = _live_format_repo(tmp_path)

    space = _grade(repo, "2026-09-16 04:00:00")
    iso_t = _grade(repo, "2026-09-16T04:00:00")

    for field in ("observations",):
        assert space["cost"][field] == iso_t["cost"][field], (
            f"cost.{field} disagrees: {space['cost'][field]} (space form) vs "
            f"{iso_t['cost'][field]} (T form)")
    for field in ("first", "last"):
        assert space["scope"][field] == iso_t["scope"][field], (
            f"scope.{field} disagrees: {space['scope'][field]} (space form) vs "
            f"{iso_t['scope'][field]} (T form) — the two forms still select "
            "different row sets")
    assert space["cost"]["observations"] == 1, (
        "guard against both forms selecting 2: agreement on the inflated answer "
        "is not the fix")


# ---------------------------------------------------------------------------
# clause 3 — the window is decided over a multi-day range
# ---------------------------------------------------------------------------


def test_a_multi_day_window_drops_the_previous_day_and_the_early_hours(tmp_path):
    """A `datetime()`-shaped bound over several days: only the bound's hour onward.

    Six rows across four days, bound `2026-09-16 04:00:00`. Correct answer: the
    `09-16 04:00` row and the `09-17 00:05` row — 2 of 6.

    The row that makes this a window rather than a string comparison is
    `2026-09-16T00:16:07.490501`: same DATE as the bound, three hours forty-four
    minutes before its hour. Rows on `09-14`/`09-15` are excluded by the date digits
    whichever way the comparison is done, so a multi-day fixture without that row
    passes pre-fix and proves nothing — the separator is only ever consulted once
    the two strings agree up to it, which is what makes the same-day early row the
    discriminating one. Pre-fix this fixture reports 3; the same shape is what the
    nightly's 26-hour bound runs on, where the leaked rows are the window's oldest
    hours.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [
        "2026-09-14T23:00:00.000000",
        "2026-09-15T00:00:00.000000",
        "2026-09-15T23:59:59.999999",
        "2026-09-16T00:16:07.490501",
        "2026-09-16T04:00:00.000000",
        "2026-09-17T00:05:00.000000",
    ])

    report = _grade(repo, "2026-09-16 04:00:00")

    assert report["cost"]["observations"] == 2, (
        f"the window kept {report['cost']['observations']} of 6 fixture rows; the "
        "same-day row before the bound's hour is being kept on the strength of its "
        "'T' separator")
    assert report["scope"]["first"] == "2026-09-16T04:00:00.000000"
    assert report["scope"]["last"] == "2026-09-17T00:05:00.000000"


#: Fixed reference instant for the SQLite-rendered-bound test, for the same reason
#: `tests/test_iv_metrics_series.py` pins `NOW_LOCAL`: an assertion computed from
#: `datetime.now()` changes shape at midnight, and here the shape IS the test — the
#: discriminating row is only same-day while the rendered bound's date matches it.
REF_LOCAL = "2026-09-16 12:00:00"


def test_a_sqlite_rendered_bound_selects_by_the_clock(tmp_path):
    """A `datetime(<instant>,'-5 hours')` bound is decided over a multi-day range.

    The bound is rendered by SQLite itself — `'2026-09-16 07:00:00'`, space form, the
    exact string `datetime('now', '-5 hours')` produces — so the separator under test
    is SQLite's, not one this file typed. Rows span `09-15` through `09-16 09:00`;
    the honest window is the `07:00` and `09:00` rows (2 of 5), and the pre-fix
    comparison kept `2026-09-16T00:16:07.490501` as well, because a stored `T` sorts
    above the bound's space and the hour was never read.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [
        "2026-09-15T22:00:00.000000",
        "2026-09-15T23:59:59.999999",
        "2026-09-16T00:16:07.490501",
        "2026-09-16T07:00:00.000000",
        "2026-09-16T09:00:00.000000",
    ])

    bound = sqlite3.connect(":memory:").execute(
        "SELECT datetime(?, '-5 hours')", (REF_LOCAL,)).fetchone()[0]
    assert bound == "2026-09-16 07:00:00", bound
    assert " " in bound, "SQLite renders a space — that separator is the whole defect"

    report = _grade(repo, bound)

    assert report["cost"]["observations"] == 2, (
        f"a bound SQLite rendered as {bound!r} kept "
        f"{report['cost']['observations']} of 5 fixture rows")
    assert report["scope"]["first"] == "2026-09-16T07:00:00.000000"


# ---------------------------------------------------------------------------
# clause 4 — the help text names the clock, the example is a full timestamp
# ---------------------------------------------------------------------------


def _help_text() -> str:
    proc = subprocess.run([sys.executable, str(GRADER), "--help"],
                          capture_output=True, text=True, check=False, cwd=str(ROOT))
    assert proc.returncode == 0, f"--help exited {proc.returncode}: {proc.stderr[:200]}"
    return proc.stdout


def test_help_names_the_clock_since_is_interpreted_in():
    """"LOCAL wall clock", and a warning that UTC is the wrong choice.

    `help="ISO date lower bound on created_at"` named no clock, which is what made
    `date -u` look like the natural thing to hand `--since` — and `skills/
    iv-metrics-series/SKILL.md` then had to forbid it in prose. The clock belongs in
    the usage text, so this asserts the phrase and the warning, not a whole sentence.
    """
    help_text = _help_text()

    assert "--since" in help_text
    assert "LOCAL wall clock" in help_text, (
        "--since still does not say which clock the bound is read in")
    assert "UTC" in help_text, (
        "the help has to name the wrong clock it is warning off — 'local' alone "
        "does not stop a caller reaching for `date -u`")


def test_module_docstring_usage_example_is_a_full_timestamp_not_a_bare_date():
    """The example no longer teaches the shape that behaved right by accident."""
    example = next(
        (line.strip() for line in GRADER.read_text().splitlines()
         if "--since" in line and "python" in line),
        None)

    assert example is not None, "the module docstring lost its --since example"
    bound = example.split("--since", 1)[1]
    assert re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", bound), (
        f"the usage example must show a full timestamp, not a bare date: {example!r}")
    assert "LOCAL" in example, (
        f"the example must carry the clock it demonstrates: {example!r}")
