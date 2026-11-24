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

import ast
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
from tests.fixture_checkout import copy_app_import_chain  # noqa: E402

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
    # The `app` modules the grader imports are listed once, in
    # `tests/fixture_checkout.py`, instead of per fixture file (#1415, #1656).
    copy_app_import_chain(GRADER.parents[1], repo)
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


# ---------------------------------------------------------------------------
# #4216 clauses 1 and 2 — the token field says which unit it is in, too
# ---------------------------------------------------------------------------


def test_the_token_line_names_its_own_unit_and_still_leaves_exactly_one_summed_line(tmp_path):
    """`input tokens TOTALLED / turn`, and the value it prints is untouched.

    #995 fixed the millisecond trap and left its twin standing: `input_tokens_per_turn`
    printed bare as `input tokens / turn`, while being the very same shape — the
    observer calls' prompt tokens over EVERY review in the window, divided by turns.
    #4216 is what that cost: `architecture/inner-voice.md` set the ~62k rows the field
    wrote before R2 against the plan's ~15–20k target, and the field cannot answer
    that question, because half of any per-turn figure is reviews-per-turn.

    Standard fixture: 11 rows of 1,000 prompt tokens over 2 turns = 5,500 a turn,
    11 reviews over 2 turns = 5.5 reviews a turn, and 1,000 a review. The three
    figures are pairwise distinct, so a line that quietly switched to a per-review
    or a per-call value cannot pass the value assert while keeping the label, and a
    line that renamed the unit without printing the counts cannot pass the unit
    asserts.

    The word is TOTALLED, not a second SUMMED. `observer ms SUMMED / turn` is #995's
    line and `test_the_summed_figure_is_still_printed_and_labelled_summed` pins that
    exactly one printed line carries that word; a token label borrowing it would
    break that assertion. So this node pins BOTH halves in one run: the new label,
    and the one-SUMMED invariant that constrains how it may be worded.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, STANDARD_ROWS)
    out = _run(repo).stdout

    hits = _lines_matching(out, r"input tokens")
    assert len(hits) == 1, f"expected exactly one printed token line: {hits!r}"
    line = hits[0]
    assert "TOTALLED" in line, line
    assert "input tokens / turn" not in line, (
        f"the bare pre-#4216 label is still on the line ({line!r}) — a label that "
        "names no unit is what let a doc read a per-review target out of a per-turn "
        "figure")
    # The unit itself: reviews in the numerator, turns in the denominator.
    assert "11 reviews" in line, line
    assert "2 turns" in line, line
    assert "NOT one review's prompt" in line, line
    # Value unchanged, plus the per-review figure that does isolate the prompt.
    assert "5,500" in line, line
    assert "per review 1,000" in line, line

    cost = _report(repo)["cost"]
    assert cost["input_tokens_per_turn"] == 5500, cost
    assert cost["input_tokens"] == 11000, cost
    assert cost["llm_calls"] == 11 and cost["turns"] == 2, cost

    summed = _lines_matching(out, r"SUMMED")
    assert len(summed) == 1, (
        f"the token label must not add a second SUMMED line — that invariant is "
        f"#995's, asserted above: {summed!r}")


def test_a_window_with_no_llm_calls_still_prints_the_token_line(tmp_path):
    """Fast-path-only window: 0 reviews, so per review is `not measured`, not 0.

    Three rows whose `input_tokens` are all 0 — every review skipped by the fast
    path. `input_tokens_per_turn` is arithmetically 0 and true, but the per-review
    figure would be 0/0, so the line has to print and say the figure is absent
    rather than fabricate one or raise. The window exits 0 today
    (`test_a_window_with_no_llm_rows_at_all_does_not_raise`); the new label
    must not change that.
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

    hits = _lines_matching(_run(repo).stdout, r"input tokens")
    assert len(hits) == 1, f"expected exactly one printed token line: {hits!r}"
    assert "0 reviews" in hits[0], hits[0]
    assert "per review not measured" in hits[0], hits[0]


def test_the_units_block_defines_the_token_field_end_to_end():
    """Clause 2: the block that fixed ms now defines the token quantity as well.

    Read off the module docstring of the shipped script through `ast`, so a block
    parked in a comment — or in the live checkout's copy but not in the file the
    nightly runs — cannot pass. #4216 asks for three things at the field's source:
    its numerator (the observer call's own `prompt_tokens`, summed over the window's
    observation rows), its denominator (DISTINCT `turn_id`s, so one turn that ran
    five reviews contributes five prompts), and the derived per-review cost
    `input_tokens_per_turn / (llm_calls/turns)` — the figure that actually isolates
    the observer prompt from the review count.

    #995's two ms bullets are asserted to survive too: this is an extension of that
    block, and a rewrite that quietly dropped the millisecond pair would un-pin #995
    from inside a change that only mentioned tokens.
    """
    doc = ast.get_docstring(ast.parse(GRADER.read_text(encoding="utf-8")))
    assert doc, "iv_grade.py carries no module docstring to hold the units"
    assert "UNITS —" in doc and "Usage:" in doc, doc[:400]
    units = doc[doc.index("UNITS —"):doc.index("Usage:")]

    assert "prompt_tokens" in units, units
    assert "observation row" in units, units
    assert "distinct" in units.lower() and "turn_id" in units, units
    assert "input_tokens_per_turn / (llm_calls/turns)" in units, units
    assert "observer ms SUMMED / turn" in units, units
    assert "observer ms PER CALL" in units, units


# ---------------------------------------------------------------------------
# #4345 clause 5: the module docstring's own latency verdict carries its dates
# ---------------------------------------------------------------------------

def test_the_module_docstring_dates_the_latency_verdict_it_ships(tmp_path):
    """The docstring's deadline sentence names its window, its bound, and today's trio.

    For seven weeks this file's module docstring ended the #458 story with "The
    deadline sits ABOVE p99 (p50 1.8 s / p90 4.1 s / p99 9.2 s over that window)".
    "That window" was #995's own `--since 2026-09-01` cut measured 2026-09-17 — n
    9,606 non-error calls, p50 1,750 / p90 4,090 / p99 9,231 ms — and no reader
    could tell, because the sentence carried no dates and named no bound. So a
    standing claim about the box outlived the traffic it described: measured
    2026-11-24 over the same `--since 2026-09-01`, the trio is p50 6,980 / p90
    15,106 / p99 21,963 ms over 29,644 non-error calls with 12,110 of them (40.9%)
    over the 8,000 ms bound — the synchronous deadline now sits BELOW p90, the
    opposite of the sentence a reader of this file would have taken away.

    The clause is therefore three things at once, asserted here: the window dates
    are in the text, the bound it compares against is named in the unit the bound is
    kept in (`timeout_seconds` in seconds, so the ms figure is spelled out), and the
    current trio is beside it with the date it was measured. Asserted as source text,
    because the artefact under test is a sentence a model reads before it runs the
    script — the same reason `_units_block` exists in `tests/test_iv_metrics_series.py`.

    The second half is the clause's safety pin: the fix was to the docstring, and the
    `UNITS` entry's "compared against a deadline they do not share" is still true and
    unedited. So this node re-runs the printed block and re-asserts the invariant
    `test_the_summed_figure_is_still_printed_and_labelled_summed` owns — exactly one
    printed line matching `SUMMED`, at 8,750 for this fixture — so a docstring edit
    that quietly reached into the `print` cannot pass as documentation-only.
    """
    doc = ast.get_docstring(ast.parse(GRADER.read_text(encoding="utf-8")))
    assert doc, "iv_grade.py carries no module docstring to hold the verdict"

    # The undated form, which is the shape that went false while still reading as a
    # measurement. It must be gone even in a quoted form.
    assert "over that window)" not in doc, (
        "the sentence is back to pointing at an unnamed window")
    assert "The deadline sits ABOVE p99 (p50" not in doc, (
        "the verdict is back to being asserted unqualified, with no dates and no bound")

    # The dated trio, the current trio, both bounds, and where to read it next.
    for text in ("2026-09-01", "2026-09-17", "9,606",
                 "p50 1,750 / p90 4,090 / p99 9,231 ms",
                 "2026-11-24", "29,644",
                 "p50 6,980 / p90 15,106 / p99 21,963 ms",
                 "8,000 ms", "12,000 ms",
                 "inner_voice.observer.timeout_seconds", "async_timeout_seconds",
                 "40.9%", "latency_p90_over_sync_deadline"):
        assert text in doc, f"the docstring no longer carries {text!r}"

    # The printed block, unchanged: one SUMMED line, same value as the node above.
    repo = _make_repo(tmp_path)
    _make_db(repo, STANDARD_ROWS)
    hits = _lines_matching(_run(repo).stdout, r"SUMMED")
    assert len(hits) == 1, f"expected one line whose label contains SUMMED: {hits!r}"
    assert "8,750" in hits[0], hits[0]
