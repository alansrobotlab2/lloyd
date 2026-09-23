"""Backlog #995 — the grader's latency figure says which unit it is in.

`scripts/iv_grade.py` used to print exactly one latency number, labelled
`observer ms / turn`, and it was a SUM: every LLM millisecond in the window over
turns. A turn costs several calls, so the number is an order of magnitude larger
than any single request — measured on the live table for 2026-09-01..09-17, 1,189
turns over 9,967 LLM calls (8.4 calls/turn), so a window whose per-call p50 is
1,750 ms printed `observer ms / turn 21,087`. Backlog #458 read that line as a
round-trip and argued "mean observer latency is 25.8 s/turn against a 12 s
deadline … the deadline is arithmetically doomed". Both observer deadlines are
PER CALL, and the 12 s one sat above p99 (9,231 ms), so the inference and its
preferred fix were unit errors.

So this file runs the shipped script as a subprocess against a throwaway
`usage.db` — the table's real schema, the shape `_was_llm_call` accepts — and
pins the four properties that make the misreading impossible:

1. a PER-CALL line carrying p50, p90 and p99 of `latency_ms` is printed, with the
   label saying per call;
2. error rows are out of that distribution: the same 12,000 ms row moves p99 only
   when it carries no error;
3. the summed figure is still printed with its value untouched
   (`sum(latency_ms) / turns`), and its label now contains `SUMMED`;
4. a window whose LLM rows ALL errored prints the percentile line and exits 0 —
   an empty distribution is `None`, not a `ZeroDivisionError` on a p99 of nothing.

The percentile rule is nearest-rank (`ceil(pct·n/100)`, 1-based) and the fixtures
are 10 and 11 rows wide so every expected value below is derived by hand in the
docstring rather than by re-running the function under test.

Run: .venvs/lloyd/bin/python -m pytest tests/test_iv_grade_latency.py -q
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

#: The ten completed round-trips every fixture below is built on. Deliberately
#: 100…1000 in steps of 100: with n = 10, nearest-rank gives p50 = the 5th value
#: (500), p90 = the 9th (900) and p99 = ceil(9.9) = the 10th (1000), so the three
#: reported percentiles are three DIFFERENT values and a function that returned
#: the same one for all three, or interpolated between them, cannot pass.
HEALTHY_MS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

#: A latency shaped like a deadline cut-off: the observer's async deadline is 12 s
#: (`config.yaml` `async_timeout_seconds`), and a dropped verdict is stamped with
#: the time it was abandoned, so a row at 12,000 ms is the row that must NOT set
#: the reported p99.
SLOW_ERROR_MS = 12000

#: The two turns the standard fixture's rows are spread over. Two, not eleven:
#: the SUMMED figure divides by TURNS, and with every row in its own turn the sum
#: and the per-call mean coincide and the unit error this item exists for is
#: unobservable in the fixture.
STANDARD_ROWS = [
    {"latency_ms": ms, "error": None, "turn_id": "tA" if i < 5 else "tB"}
    for i, ms in enumerate(HEALTHY_MS)
] + [
    {"latency_ms": SLOW_ERROR_MS, "error": "timeout after 12.0s", "turn_id": "tB"},
]


def _make_repo(tmp_path: Path) -> Path:
    """A throwaway repo root holding a copy of the grader and its own `usage.db`.

    Copied, not symlinked: `iv_grade.py` anchors `DB_PATH` at
    the data root of `Path(__file__).resolve().parents[1]`, and a symlink resolves back to the real
    checkout — which would have the suite reading, and asserting against, the
    production table instead of the fixture.
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


def _make_db(repo: Path, rows: list[dict]) -> Path:
    """A fixture `usage.db` with the live `inner_voice_observations` schema.

    Rows come in as dicts so one call can mix completed calls and error rows — the
    pair clauses 2 and 4 are about. `input_tokens` is nonzero on every row so each
    one is an LLM row by `_was_llm_call` (`iv_grade.py:210-218`), which is what
    makes "an error row is a real call whose duration is not a round-trip" the
    property under test rather than an artefact of row selection.
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
    for i, row in enumerate(rows, start=1):
        conn.execute(
            """INSERT INTO inner_voice_observations
               (session_id, turn_id, sequence_in_turn, trigger, action, reason,
                input_tokens, output_tokens, cache_read, cache_create,
                latency_ms, model, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("s1", row.get("turn_id", f"t{i}"), i, "assistant_message",
             "noop", "fixture row", 1000, 5, 0, 0,
             row.get("latency_ms"), "primary", row.get("error")))
    conn.commit()
    conn.close()
    return db


def _run(repo: Path, *flags: str) -> subprocess.CompletedProcess:
    """One real invocation of the shipped CLI, flags included."""
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "iv_grade.py"), *flags],
        capture_output=True, text=True, check=False, cwd=str(repo))
    assert proc.returncode == 0, (
        f"iv_grade.py {' '.join(flags)} exited {proc.returncode}: "
        f"stdout={proc.stdout[:300]!r} stderr={proc.stderr[:300]!r}")
    return proc


def _report(repo: Path) -> dict:
    return json.loads(_run(repo, "--json").stdout)


def _lines_matching(text: str, pattern: str) -> list[str]:
    """The lines a reader's `grep -E` would find. The acceptance check is a grep,
    so the test greps the same way instead of asserting on a substring that could
    sit anywhere in the block."""
    rx = re.compile(pattern)
    return [line for line in text.splitlines() if rx.search(line)]


def _percentile_line(text: str) -> str:
    """The printed PER-CALL line — clause 1's grep target, exactly one of them."""
    hits = [ln for ln in _lines_matching(text, r"p50.*p90.*p99")
            if "per call" in ln.lower()]
    assert len(hits) == 1, (
        f"expected exactly one printed line that is labelled per call and carries "
        f"p50/p90/p99, got {len(hits)}: {hits!r}\nfull stdout:\n{text}")
    return hits[0]


# ---------------------------------------------------------------------------
# clause 1 — a per-call percentile line is printed, and its label says per call
# ---------------------------------------------------------------------------


def test_grader_prints_p50_p90_p99_on_a_line_labelled_per_call(tmp_path):
    """`grep -E 'p50|p90|p99'` on stdout matches a line that says PER CALL.

    Standard fixture: ten completed calls at 100…1000 ms plus one 12,000 ms error
    row. Expected percentiles over the ten non-error rows (n = 10, nearest-rank):
    p50 500 · p90 900 · p99 1,000. Pre-fix stdout matched `observer ms` on one
    summed line and matched the percentile pattern on nothing at all.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, STANDARD_ROWS)

    stdout = _run(repo).stdout
    line = _percentile_line(stdout)

    for expected in ("500", "900", "1,000"):
        assert expected in line, f"{expected} missing from the per-call line: {line!r}"

    cost = _report(repo)["cost"]
    assert cost["latency_ms_per_call_p50"] == 500, cost
    assert cost["latency_ms_per_call_p90"] == 900, cost
    assert cost["latency_ms_per_call_p99"] == 1000, cost
    assert cost["latency_ms_per_call_n"] == 10, (
        "the denominator beside the percentiles must be the count of rows that "
        f"went into them, not llm_calls ({cost['llm_calls']})")


# ---------------------------------------------------------------------------
# clause 2 — error rows are excluded from the distribution
# ---------------------------------------------------------------------------


def test_a_12000ms_row_with_an_error_does_not_move_the_reported_p99(tmp_path):
    """The same row without an error does move it. One fixture, two states.

    With the 12,000 ms row carrying `error = "timeout after 12.0s"`: n = 10 and
    p99 = 1,000 (its own deadline is not a round-trip). Clear that row's error and
    n = 11, so nearest-rank p99 = ceil(10.89) = the 11th value = 12,000. The pair
    is what makes this a test of the filter rather than of the sort: a build that
    ignored `error` entirely reports 12,000 in BOTH cases, and one that dropped
    every slow row reports 1,000 in both.
    """
    with_error = _make_repo(tmp_path / "with")
    _make_db(with_error, STANDARD_ROWS)
    healthy = _make_repo(tmp_path / "healthy")
    _make_db(healthy, [
        {**row, "error": None} if row["latency_ms"] == SLOW_ERROR_MS else row
        for row in STANDARD_ROWS
    ])

    errored = _report(with_error)["cost"]
    cleared = _report(healthy)["cost"]

    assert errored["latency_ms_per_call_p99"] == 1000, errored
    assert cleared["latency_ms_per_call_p99"] == SLOW_ERROR_MS, cleared
    assert errored["latency_ms_per_call_n"] == 10
    assert cleared["latency_ms_per_call_n"] == 11
    # The error row is still a real LLM call — it is excluded from the duration
    # distribution, not from the cost tables. Otherwise "excluded" here would
    # quietly mean "not counted anywhere", which is a different and wrong claim.
    assert errored["llm_calls"] == 11 == cleared["llm_calls"]


# ---------------------------------------------------------------------------
# clause 3 — the summed figure survives, labelled as a sum
# ---------------------------------------------------------------------------


def test_the_summed_figure_is_still_printed_and_labelled_summed(tmp_path):
    """Same value, honest label: `sum(latency_ms) / turns`, `SUMMED` in the text.

    Standard fixture: 17,500 ms of LLM time over 2 turns = 8,750. The window's
    per-call p50 is 500 ms in the same run — an 17.5× gap in one printed block,
    which is the gap #458 collapsed. A fix that renamed the line but recomputed it
    (e.g. as a mean per call) would fail the value assert while passing the label
    assert.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, STANDARD_ROWS)

    hits = _lines_matching(_run(repo).stdout, r"SUMMED")
    assert len(hits) == 1, f"expected one line whose label contains SUMMED: {hits!r}"
    assert "8,750" in hits[0], hits[0]

    cost = _report(repo)["cost"]
    assert cost["observer_ms_per_turn"] == 8750, cost
    assert cost["latency_ms_per_call_p50"] == 500, cost


# ---------------------------------------------------------------------------
# clause 4 (of #995) — an all-error window is a measurement, not a crash
# ---------------------------------------------------------------------------


def test_a_window_whose_llm_rows_all_errored_still_reports_and_exits_zero(tmp_path):
    """Every row carries an error: the percentile line appears, values are absent.

    Four rows at 12,000 ms, all timed out — the shape of an observer whose model
    endpoint is down, and the case where a naive `statistics.quantiles([])` or a
    `vals[-1]` on an empty list raises. The CLI must exit 0 (the window is empty of
    round-trips, not empty of data), print the same labelled line so a reader
    grepping for `p99` still finds it, and report `None` rather than a fabricated
    zero: a `0 ms` p99 would read as the fastest night on record.

    The summed figure is untouched by all this: 48,000 ms over 2 turns = 24,000,
    which is exactly how the old single line made a dead endpoint look like a
    24-second round-trip.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [{"latency_ms": SLOW_ERROR_MS, "error": "timeout after 12.0s",
                     "turn_id": "tA" if i < 2 else "tB"} for i in range(4)])

    stdout = _run(repo).stdout
    line = _percentile_line(stdout)
    assert "not measured" in line, line

    cost = json.loads(_run(repo, "--json").stdout)["cost"]
    assert cost["latency_ms_per_call_n"] == 0, cost
    for key in ("latency_ms_per_call_p50", "latency_ms_per_call_p90",
                "latency_ms_per_call_p99"):
        assert cost[key] is None, f"{key} fabricated a value for an empty window: {cost}"
    assert cost["observer_ms_per_turn"] == 24000, cost
    assert cost["llm_calls"] == 4, cost


def test_a_window_with_no_llm_rows_at_all_does_not_raise(tmp_path):
    """Fast-path-only rows: no LLM row means no denominator anywhere.

    The degenerate case one step earlier than the clause above — rows exist, none
    of them cost a call (`_was_llm_call` is false at `iv_grade.py:210-218`), so the
    distribution is empty for a different reason. Same contract: exit 0, no value
    fabricated.
    """
    repo = _make_repo(tmp_path)
    db = repo / ".lloyd-data" / "usage.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE inner_voice_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        turn_id TEXT NOT NULL, sequence_in_turn INTEGER NOT NULL,
        trigger TEXT NOT NULL, action TEXT NOT NULL, reason TEXT, content TEXT,
        related_tool TEXT, input_tokens INTEGER, output_tokens INTEGER,
        cache_read INTEGER, cache_create INTEGER, latency_ms INTEGER,
        model TEXT, error TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    for i in range(3):
        conn.execute(
            """INSERT INTO inner_voice_observations
               (session_id, turn_id, sequence_in_turn, trigger, action, reason,
                input_tokens, output_tokens, cache_read, cache_create,
                latency_ms, model, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("s1", f"t{i}", i, "pretool", "noop", "fast-path: no candidate",
             0, 0, 0, 0, 0, "primary", None))
    conn.commit()
    conn.close()

    cost = _report(repo)["cost"]
    assert cost["llm_calls"] == 0, cost
    assert cost["latency_ms_per_call_n"] == 0, cost
    assert cost["latency_ms_per_call_p99"] is None, cost
    assert cost["observer_ms_per_turn"] == 0, cost
