"""#2576 clauses 1-3 and #2798's displacement columns: the per-job-class join.

Clause → node map (the review rung grades per clause, so every clause names its
nodes here rather than leaving them to be found by title):

* **#2798 clause 1** — `--json` carries `mean_tool_calls`, `errors_per_tool_call`
  and `length_bound_turns`, each over only the rows where that column is non-NULL,
  with its measured-row count beside it, and null rather than 0 for a class nothing
  measured:
  `test_2798_clause_1_the_json_row_carries_the_three_ratios_with_their_denominators`,
  `test_2798_clause_1_a_measured_zero_is_not_the_unmeasured_null`.
* **#2798 clause 2** — both the plain `render` table and `markdown_rows` print the
  three as cells taken from that row's own fields:
  `test_2798_clause_2_every_displacement_cell_is_that_rows_own_in_both_tables`.
* **#2798 clause 3** — mutating one row's tool-call and stop-reason fields moves that
  row's cells and no other's, in the JSON and in both renderings:
  `test_2798_clause_3_mutating_one_witness_row_moves_only_that_row`,
  `test_2798_clause_3_mutating_one_usage_row_moves_only_its_class`,
  `test_no_node_in_this_file_skips`.
* **#2798 clause 4** — the committed 2026-10-11 witness still renders its pinned
  11-column table byte-for-byte, because its rows carry none of the new keys:
  `test_2798_clause_4_the_committed_witness_still_renders_eleven_columns`.
* The report's own prose reading of these columns is pinned in
  `tests/test_thinking_effort_report.py`, in the file that already pins its tables.
* **Clause 1** — one row per job class with mapped-session count, turn count,

* **Clause 1** — one row per job class with mapped-session count, turn count,
  non-null `reasoning_tokens` coverage, mean and p90 reasoning tokens, mean output
  tokens, reasoning share, and success/failed/skipped counts:
  `test_clause_1_join_emits_one_row_per_job_class_with_every_column`,
  `test_clause_1_a_class_with_no_turns_carries_dashes_and_keeps_its_runs`,
  `test_clause_1_the_aligned_render_labels_where_a_dash_came_from`,
  `test_clause_1_the_real_stores_join_too_and_the_class_that_priced_the_knob_is_there`.
* **Clause 2** — a separate `unattributed` row for usage rows named by no run, so no
  turn leaves the corpus without a line saying so:
  `test_clause_2_a_turn_no_run_claims_is_a_row_not_a_footnote`,
  `test_clause_2_unclaimed_usage_rows_become_a_row_never_a_footnote`,
  `test_clause_2_corpus_accounting_prints_every_row_the_window_excludes`.
* **Clause 3** — outcome rate is `success/(success+failed)` with `skipped` and
  `interrupted` out of the denominator but printed beside it, and a thin denominator
  reported with its n:
  `test_clause_3_outcome_rate_excludes_skipped_and_interrupted_but_prints_them`,
  `test_clause_3_a_thin_denominator_is_marked_with_its_n_attached`.

The clause-level numbers themselves — the ones quoted into the vault report — are
pinned in `tests/test_thinking_effort_report.py`, which re-renders the report's tables
from the committed witness instead of comparing them to a typed copy.

Why this needs a real fixture and not just a live-DB poke
---------------------------------------------------------
The join crosses a process boundary that has no code behind it: `usage.db` and
`workers.db` are separate stores, connected only by a `session_ids` array that
`workers/pool.py` happens to write into `runs.meta_json` and `app/usage_store.py`
happens to key turns by. Neither store validates the other, so "unattributed" and
"unmeasured" and "zero" are all reachable from a silent schema change. The fixture
below pins the contract and makes every definition the script's docstring claims
assertable.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.reasoning_by_job_class import (NO_VALUE, THIN_DENOM, UNATTRIBUTED,
                                            build_table, markdown_rows, render,
                                            write_witness)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_WITNESS = REPO_ROOT / "eval" / "measurements" / "2576-class-table-2026-10-11.md"
LIVE_DATA_ROOT = Path("/home/alansrobotlab/lloyd-data")


def _unwrap_witness(path: Path) -> dict:
    """A tracked `.md` witness: prose head, then one fenced JSON block."""
    text = path.read_text()
    start = text.index("```json") + len("```json")
    return json.loads(text[start:text.index("```", start)])


def _measured() -> tuple[dict, str]:
    """The real class table to assert against: the live stores, else their witness.

    Never a skip. A node that skips when the production stores are absent is a node
    that passes on a machine where nothing was checked, and the gate runs on a checkout
    that may be one of those. Both branches run the SAME assertions; what differs is
    which bytes they read, and the witness is tracked, so off-box the claim is still
    measured against the artifact the stores produced. The second return value names
    the source, which the asserting node prints into its failure message — the same
    reason the script prints its own `corpus:` line.
    """
    workers, usage = LIVE_DATA_ROOT / "workers.db", LIVE_DATA_ROOT / "usage.db"
    if workers.exists() and usage.exists():
        return build_table(workers_db=workers, usage_db=usage,
                           since=None), "live stores"
    return _unwrap_witness(CLASS_WITNESS), f"committed witness {CLASS_WITNESS.name}"


def _seed_workers(path: Path, rows: list[tuple[str, str, str, list[str]]]) -> None:
    """`runs` exactly as `workers/pool.py` writes it: class in `source`, sessions in
    `meta_json.session_ids`. Columns omitted here are never read by the join."""
    con = sqlite3.connect(path)
    try:
        con.execute("create table runs (source text, started_at text, status text,"
                    " meta_json text)")
        con.executemany("insert into runs values (?,?,?,?)",
                        [(s, t, st, json.dumps({"session_ids": sids}))
                         for s, t, st, sids in rows])
        con.commit()
    finally:
        con.close()


def _seed_usage(path: Path,
                rows: list[tuple[str | None, str, int | None, int]
                           | tuple[str | None, str, int | None, int,
                                   str | None, int | None, int | None]]) -> None:
    """One row per turn, in the table the live store actually has.

    A 4-tuple is `(session_id, ts, reasoning_tokens, output_tokens)` with the
    telemetry columns left NULL — the shape every row written before
    `app/usage_store.py::_TELEMETRY_COLUMNS` landed has on the live store, which is
    what `tool_rows`/`stop_rows` exist to count. A 7-tuple adds
    `(stop_reason, tool_calls, tool_errors)` for the displacement cases.

    The table comes from `app/usage_store._init_schema`, the same function that
    creates the production store, and not from a column list typed in here. The
    previous fixture typed eleven columns of its own invention — `source`, `kind`,
    `job_id` and `cached_tokens`, four names the live table has never carried — while
    missing `stop_reason`, `tool_calls` and `tool_errors`, so its docstring's claim to
    be "the whole `usage` table" was false twice over and every query naming the
    telemetry columns died with `no such column` on a fixture that purported to
    exercise them. `_init_schema` creates the base table and then applies both
    migration loops over `_TELEMETRY_COLUMNS` (app/usage_store.py:65 and the ALTER
    block at :170-177), so a rename of a real column now breaks this fixture here
    instead of breaking a live read.
    """
    from app.usage_store import _init_schema

    con = sqlite3.connect(path)
    try:
        con.row_factory = sqlite3.Row          # _init_schema reads PRAGMA by name
        _init_schema(con)
        wide = [r + (None, None, None) if len(r) == 4 else r for r in rows]
        con.executemany(
            "insert into usage (ts, session_id, model, input_tokens, output_tokens,"
            " reasoning_tokens, stop_reason, tool_calls, tool_errors)"
            " values (?,?,?,?,?,?,?,?,?)",
            [(t, sid, "Qwen3.8-Flash-Next", 100, out, reason, *telemetry)
             for sid, t, reason, out, *telemetry in wide])
        con.commit()
    finally:
        con.close()


@pytest.fixture
def stores(tmp_path):
    """Two stores, one window, six runs, ten turns — every case the definitions name.

    Shaped like the real corpus rather than tidy: one class with a NULL-heavy engine
    (autotriage), one with no turns at all (queue-maintenance), a `skipped` run that
    must not become a failure, and turns no run claims — including a NULL `session_id`
    and rows outside the window — so a zero in any figure means the code found
    nothing, not that the fixture held nothing.
    """
    workers, usage = tmp_path / "workers.db", tmp_path / "usage.db"
    _seed_workers(workers, [
        # autotriage: one success, one skipped, one interrupted, one failed.
        ("autotriage", "2026-10-05T10:00:00", "success", ["s-tri-a"]),
        ("autotriage", "2026-10-06T10:00:00", "skipped", ["s-tri-b"]),
        ("autotriage", "2026-10-07T10:00:00", "interrupted", ["s-tri-c"]),
        ("autotriage", "2026-10-08T10:00:00", "failed", ["s-tri-d"]),
        # autocode: a session named FIRST by autocode and re-used by a later
        # autotriage run, so the class has to be the earliest run's, deterministically.
        ("autocode", "2026-10-05T09:00:00", "success", ["s-code-a"]),
        ("autotriage", "2026-10-09T10:00:00", "skipped", ["s-code-a"]),
        # A second autocode turn whose ratio is 1/18th of the first's, so the class's
        # SHARE and the mean of its per-turn ratios differ by nearly 3x.
        ("autocode", "2026-10-05T09:05:00", "success", ["s-code-b"]),
        # queue-maintenance: runs and zero turns (a job that never calls the LLM).
        ("queue-maintenance", "2026-10-05T11:00:00", "success", []),
        # board-steward: exactly THIN_DENOM attempted runs and no turns, so the
        # fixture holds a non-thin denominator next to thin ones — the boundary is
        # asserted in `test_a_full_denominator_is_not_marked_thin`, and a fixture with
        # nothing above the bar could not tell a working marker from one that fires
        # always. Sessions empty because the outcome rate needs runs, not turns.
        *[(("board-steward"), f"2026-10-0{5 + i % 5}T12:00:00", "success", [])
          for i in range(30)],
    ])
    _seed_usage(usage, [
        # autotriage: one NULL reasoning, one zero-output, all four of them below.
        # The tail is (#2798) `(stop_reason, tool_calls, tool_errors)`: this class is
        # the mixed case — 4 turns, only 3 of them measured, one of those 3 cut off at
        # the budget, one measured with zero calls, and calls/errors whose class ratio
        # is 3/12 = 0.25 while the mean of the per-turn ratios is (0.25 + 0.25 + 0)/3 =
        # 0.1667. Same trap as `share`, one column over.
        ("s-tri-a", "2026-10-05T10:00:05", 40, 100, "length", 4, 1),
        ("s-tri-b", "2026-10-06T10:00:05", None, 90),
        # Output 0, not None: the live DDL declares `output_tokens INTEGER NOT NULL`,
        # so the NULL this row used to carry is a shape the store cannot hold and the
        # old fixture was exercising `build_table`'s `output is not None` guard against
        # a case production never sends. The class's covered-output sum is 100+0+50 and
        # its share 120/150 either way, so no figure below moves.
        ("s-tri-c", "2026-10-07T10:00:05", 60, 0, "stop", 8, 2),
        # autotriage's own turn, plus autocode's — both covered.
        ("s-tri-d", "2026-10-08T10:00:05", 20, 50, "max_turns", 0, 0),
        ("s-code-a", "2026-10-05T09:00:05", 10, 20, "stop", 10, 0),
        # autocode's second turn keeps NULL telemetry, so the class's tool mean is over
        # ONE row while its `turns` is 2: a figure that divided by `turns` would print
        # 5.0 and quietly price an unmeasured turn as a turn that called nothing.
        ("s-code-b", "2026-10-05T09:05:05", 10, 180),
        # Unattributed: no run names these. Including a NULL session_id, which is a
        # real shape in the store and must land in the residual, not crash the join.
        # s-orphan-1's two calls both errored, so the residual carries the 1.000
        # error-per-call that no other class in this fixture has.
        ("s-orphan-1", "2026-10-05T12:00:00", 5, 20, "length", 2, 2),
        ("s-orphan-2", "2026-10-06T12:00:00", None, 20),
        (None, "2026-10-07T12:00:00", 7, 21),
        # Outside the window entirely, and counted in the footer, not in a row.
        ("s-tri-a", "2020-01-01T00:00:00", 999, 1000),
    ])
    return workers, usage


def _rows(result: dict) -> dict[str, dict]:
    return {row["class"]: row for row in result["rows"]}


# ── clause 1 ────────────────────────────────────────────────────────────────


def test_clause_1_join_emits_one_row_per_job_class_with_every_column(stores):
    """Clause 1: every column the clause names, on one row per class.

    The definitions the clause depends on, each pinned on a class shaped to make
    the wrong answer visible: `meanR`/`p90R`/`meanOut` over the covered subset only;
    p90 nearest-rank at `ceil(0.9*n)`, which on n=3 is the 3rd value, not the
    interpolated one a numpy default would print; and `share` as
    sum(reasoning)/sum(output) — autocode's two turns have per-turn ratios of 0.5 and
    0.056, whose mean is 0.278, while its share is 0.1, and a reader who conflated
    the two would read a 10x-heavy class as a 3.6x one.
    """
    workers, usage = stores
    result = build_table(workers_db=workers, usage_db=usage,
                         since="2026-01-01T00:00:00")
    rows = _rows(result)
    assert set(rows) == {"autotriage", "autocode", "queue-maintenance",
                         "board-steward", UNATTRIBUTED}

    tri = rows["autotriage"]
    assert tri["sessions"] == 4 and tri["turns"] == 4
    assert tri["reasoning_coverage"] == 3          # one row's reasoning is NULL
    assert tri["mean_reasoning_tokens"] == pytest.approx((40 + 60 + 20) / 3)
    assert tri["p90_reasoning_tokens"] == 60       # ceil(0.9*3)=3rd of [20,40,60]
    # Output summed over the COVERED rows only (100+0+50): the NULL-output turn's
    # tokens are unknown, and guessing 0 or 100 would move the share on this row.
    assert tri["mean_output_tokens"] == pytest.approx(150 / 3)
    assert tri["reasoning_share_of_output"] == pytest.approx(120 / 150)

    code = rows["autocode"]
    assert (code["sessions"], code["turns"]) == (2, 2)   # s-code-a counted once
    assert code["reasoning_share_of_output"] == pytest.approx(20 / 200)
    assert code["reasoning_share_of_output"] < 0.2, (
        "the share drifted to the mean of per-turn ratios, which on this fixture is "
        "0.278 — nearly 3x the aggregate the column claims to print")
    assert rows["queue-maintenance"]["turns"] == 0
    assert rows[UNATTRIBUTED]["turns"] == 3


def test_clause_1_a_class_with_no_turns_carries_dashes_and_keeps_its_runs(stores):
    """A zero-turn class is reported, not dropped, and its token cells say "none".

    `queue-maintenance` runs without calling the engine, so it has an outcome rate and
    no token figures. Zero in a token column would say "measured, cheap"; the class
    was never measured at all, which is the distinction #2576's coverage requirement
    exists to keep.
    """
    workers, usage = stores
    row = _rows(build_table(workers_db=workers, usage_db=usage,
                            since="2026-01-01T00:00:00"))["queue-maintenance"]
    for column in ("mean_reasoning_tokens", "p90_reasoning_tokens",
                   "mean_output_tokens", "reasoning_share_of_output"):
        assert row[column] is None, f"{column} invented a number for a class with no turns"
    assert row["outcome_rate"] == 1.0 and row["outcome_n"] == 1


def test_the_markdown_render_prints_the_missing_cell_constant_not_zero(stores):
    """The vault-facing table uses the script's own `NO_VALUE`, never a `0`.

    Read out of the module rather than typed into the test: an em-dash and an en-dash
    are one character apart, and a hand-typed one in a test asserts the test's own
    spelling. `0` in these cells would tell a reader that a class which never called
    the engine was measured and came out free — the exact inversion #2576's coverage
    requirement exists to prevent.
    """
    workers, usage = stores
    md = markdown_rows(build_table(workers_db=workers, usage_db=usage,
                                   since="2026-01-01T00:00:00"))
    row = next(l for l in md.splitlines() if l.startswith("| queue-maintenance"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:4] == ["0", "0", "0"], (
        f"the run/turn/cover counts stopped being real zeros: {cells}")
    assert cells[4:8] == [NO_VALUE] * 4, (
        f"a zero-turn class is printing {cells[4:8]} where the four token columns "
        "should carry the missing-measurement constant")
    assert "0" not in cells[4:8] and NO_VALUE != "0"


def test_clause_1_the_aligned_render_labels_where_a_dash_came_from(stores):
    """Every token cell is a number or `-`; a `-1` or a padded 0 is a fabricated measurement.

    Asserted per row of the rendered text rather than as `"-" in text`, because a
    legend that contains a dash satisfies the loose form no matter what the table
    prints. This node is also where the dash's PROSE lives: the render must say that
    the figures are over the covered subset, or the dash is a reader guessing.
    """
    workers, usage = stores
    result = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    text = render(result)
    body = [l for l in text.splitlines()
            if l.split() and l.split()[0] in _rows(result)]
    assert len(body) == 5, f"expected one printed row per class, got {len(body)}"
    # `-1` is scanned over the TABLE, not the whole page: the window line above it
    # carries ISO timestamps, which contain "-1" as a substring and made a page-wide
    # scan of this string pass or fail on the date.
    assert not [l for l in body if "-1" in l], f"a negative token figure: {body}"
    for line in body:
        fields = line.split()
        if line.startswith(("queue-maintenance", "board-steward")):
            assert fields[4:7] == ["-", "-", "-"], (
                f"{line!r}: a zero-turn class must not carry token figures")
        else:
            for cell in fields[4:7]:
                assert cell.isdigit(), f"{line!r} rendered {cell!r} as a token figure"
    assert "COVERED subset" in text and "reasoning_tokens" in text


def test_clause_1_the_real_stores_join_too_and_the_class_that_priced_the_knob_is_there():
    """The clause against the box's own stores, not a fixture.

    A 13-row join that only ever passes on hand-made data proves the arithmetic and
    nothing about the corpus; and the real stores have a shape the fixture cannot
    invent — a class with 1,615 attempted runs and 2,036 skipped ones, which is the
    denominator mistake clause 3 exists to prevent. The numbers here are pinned in the
    committed witness; what this node pins is that the committed witness is the live
    shape and not a stale one, so it asserts only the properties the report's
    conclusion stands on.

    Runs against the production stores, and where a machine has none it reads the
    committed witness those stores produced (see `_measured()`) rather than skipping: a
    skip is a green result from a box where nothing was looked at, and this file's own
    clause exists to stop an unmeasured corpus reading as a clean bill. The fallback is
    a tracked artifact, so the same assertions run over the bytes the join emitted, and
    every failure below names which source it read.
    """
    result, measured_from = _measured()
    rows = _rows(result)
    assert UNATTRIBUTED in rows
    assert result["runs"] > 1000 and result["usage_rows"] > 1000, (
        f"not measured on a real corpus (source: {measured_from})")

    autocode = rows["autocode"]
    assert autocode["reasoning_coverage"] > 500, (
        "autocode is the report's highest-share class with real n; if its coverage "
        "collapsed, its share is no longer a measurement of that class")

    triage = rows["autotriage"]
    assert triage["outcome_rate"] is not None
    assert triage["skipped"] > triage["outcome_n"], (
        "autotriage's skipped runs outnumber its attempted ones in the real corpus; "
        "counting them into the denominator would move its rate by tens of points")


# ── clause 2 ────────────────────────────────────────────────────────────────


def test_clause_2_a_turn_no_run_claims_is_a_row_not_a_footnote(stores):
    """Clause 2: the interactive residual gets a row, with counts and token figures.

    `usage.db` carries no job class and 3,283 of 10,278 rows were named by no run at
    triage, so "interactive turns" is definable ONLY as that residual. Without this
    row a third of the corpus is missing from a table that reads as complete, and
    without its token figures the missing third is invisible in exactly the column
    (#2576's own reasoning: the residual's share is the third-highest in the table, so
    dropping it would drop a finding, not just a count).
    """
    workers, usage = stores
    result = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    unattr = _rows(result)[UNATTRIBUTED]
    assert unattr["turns"] == 3
    assert unattr["reasoning_coverage"] == 2, "the NULL-reasoning orphan must not count"
    assert unattr["mean_reasoning_tokens"] == pytest.approx((5 + 7) / 2)
    assert unattr["outcome_rate"] is None, "a residual with no runs has no outcome rate"
    assert unattr["success"] == unattr["failed"] == unattr["skipped"] == 0
    assert UNATTRIBUTED in render(result)


def test_clause_2_unclaimed_usage_rows_become_a_row_never_a_footnote(tmp_path):
    """A session named by no run lands in `unattributed`, which carries token figures.

    The clause's wording is the requirement: no turn may leave the corpus without a
    line saying so. So the assertion is a conservation law over the whole store, not
    the presence of a label — attributed + unattributed must equal the rows the window
    held, on every window, or a row vanished somewhere with no line to say which.
    """
    workers, usage = tmp_path / "workers.db", tmp_path / "usage.db"
    _seed_workers(workers, [("autocode", "2026-10-05T09:00:00", "success", ["s1"])])
    _seed_usage(usage, [("s1", "2026-10-05T09:10:00", 50, 100),
                        ("s2", "2026-10-05T09:10:00", 10, 20),
                        ("s2", "2026-10-05T09:11:00", None, 20)])
    for since, expected in ((None, 2), ("2026-01-01T00:00:00", 2),
                            # A window that starts between the two `s2` turns holds
                            # one of them, so the residual moves with the corpus —
                            # asserted per window, because "row exists" would pass on
                            # a scan that dropped two turns and printed the third.
                            ("2026-10-05T09:10:30", 1)):
        result = build_table(workers_db=workers, usage_db=usage, since=since)
        rows = _rows(result)
        assert rows[UNATTRIBUTED]["turns"] == expected, (f"since={since}")
        assert (rows[UNATTRIBUTED]["turns"] + result["turns_attributed"]
                == result["usage_rows"]), (
            f"since={since}: attributed + unattributed != the window's rows, so a "
            "turn is missing from the table with no line saying so")
    assert _rows(build_table(workers_db=workers, usage_db=usage,
                             since="2026-01-01T00:00:00"))[UNATTRIBUTED][
                             "reasoning_coverage"] == 1


def test_clause_2_corpus_accounting_prints_every_row_the_window_excludes(stores):
    """Nothing in the corpus may vanish quietly: excluded rows get a printed count.

    `--days 7` keys its window off the newest run, so rows before it are outside the
    table by construction. A census that reports 9 rows and silently omits the 10th
    is how a stale-looking figure gets quoted into an item as a finding, so the footer
    prints what it left out, how many NULL-session rows there were, and how many
    sessions were named by two classes.
    """
    workers, usage = stores
    result = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    assert result["usage_rows"] == 9
    assert result["usage_rows_outside_window"] == 1      # the 2020 turn
    assert result["null_session_id_rows"] == 1
    # s-code-a is named by an autocode run and later by an autotriage run: the
    # collision is COUNTED, and the turn still belongs to the earliest run's class.
    assert result["multi_class_sessions"] == 1
    assert _rows(result)["autocode"]["turns"] == 2

    tight = build_table(workers_db=workers, usage_db=usage, since="2026-10-07T00:00:00")
    assert tight["usage_rows"] == 3
    assert tight["usage_rows_outside_window"] == 7      # 10 rows, 3 in window
    assert (next(r for r in tight["rows"] if r["class"] == UNATTRIBUTED)["turns"]
            + tight["turns_attributed"] == tight["usage_rows"])
    text = render(tight)
    assert "usage rows outside the window (not in any figure above): 7" in text
    assert "rows with a NULL session_id: 1" in text
    # The collision is only one inside a window holding BOTH runs that name s-code-a.
    # The tight window starts after the autocode run, so its footer must read 0 while
    # the full window's reads 1 — asserted as a pair, because a footer frozen at either
    # number would satisfy a single assertion of this kind.
    assert "sessions named by more than one job class: 0" in text
    assert ("sessions named by more than one job class: 1"
            in render(build_table(workers_db=workers, usage_db=usage,
                                  since="2026-01-01T00:00:00")))


# ── clause 3 ────────────────────────────────────────────────────────────────


def test_clause_3_outcome_rate_excludes_skipped_and_interrupted_but_prints_them(stores):
    """Clause 3: `success/(success+failed)`, with the excluded counts kept beside it.

    `skipped` is a run that never reached the engine (depth gate, landing hold) and
    `interrupted` one a restart cut short; neither is an attempt that could have gone
    either way, so folding them into the denominator would report autotriage's real
    100 % over n=1,615 as ~50 %, and a reader would go looking for a failure that
    does not exist.

    The status vocabulary is asserted too. `workers/pool.py`'s JobStatus is
    `queued/claimed/running/success/failed/skipped/interrupted/lost` — the item's
    triage wrote "ok/fail/skip", which is NOT the vocabulary, and a rate computed with
    `.get("ok")` is silently 0/0 forever on every real store. So the four excluded and
    included names are pinned rather than implied, and the same vocabulary is asserted
    against the live `runs` table in the next-but-one node.
    """
    workers, usage = stores
    result = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    tri = _rows(result)["autotriage"]
    assert (tri["success"], tri["failed"], tri["skipped"], tri["interrupted"]) == (1, 1, 2, 1)
    assert tri["outcome_rate"] == 0.5, "skipped/interrupted leaked into the denominator"
    assert tri["outcome_n"] == 2
    line = next(l for l in render(result).splitlines() if l.startswith("autotriage"))
    assert "50%" in line and "1/1/2/1" in line, (
        "the rate is being printed without the excluded counts beside it")
    assert "skipped/interrupted excluded" in render(result), (
        "the legend that defines the denominator went missing from the output")


def test_the_live_status_vocabulary_is_the_one_the_rate_reads():
    """The denominator cannot be quietly 0/0 because the status names went stale.

    Clause 3 is only true if `success` and `failed` are the strings the store uses,
    and the item's own triage line said `ok/fail/skip` — a wrong-enough guess to pass
    every fixture in this file and report 0 attempted runs for every class on the
    box. Read the real `runs` table and assert the two names the rate multiplies are
    present in it.

    Off-box it reads the committed witness instead of skipping, and says what that
    proves and does not: `select distinct status` cannot be run against aggregates, so a
    store that grew a FIFTH status stays invisible there. What is really read off the
    tracked bytes is the set of outcome names the table emits as count keys — a rename
    in the script changes that set — plus the requirement that each of the four is
    populated by a nonzero count somewhere in the measured corpus, which fails on a
    status the script silently stopped counting. Not a tautology: the key set comes from
    a witness row and the totals are summed across every class in it.
    """
    workers = LIVE_DATA_ROOT / "workers.db"
    names = ("success", "failed", "skipped", "interrupted")
    if workers.exists():
        con = sqlite3.connect(f"file:{workers}?mode=ro", uri=True)
        try:
            statuses = {row[0] for row in con.execute("select distinct status from runs")}
        finally:
            con.close()
        source = "live runs table"
    else:
        witness = _unwrap_witness(CLASS_WITNESS)
        statuses = {k for k in names if k in witness["rows"][0]}
        for name in names:
            if sum(r[name] for r in witness["rows"]) == 0:
                statuses.discard(name)
        source = f"committed witness {CLASS_WITNESS.name} (key set and populated counts)"
    assert set(names) <= statuses, f"{source}: {sorted(statuses)}"
    assert "ok" not in statuses and "fail" not in statuses, (
        f"{source}: the vocabulary changed under the rate; "
        "update both here and the report")


def test_clause_3_a_thin_denominator_is_marked_with_its_n_attached(stores):
    """Clause 3: a class under `THIN_DENOM` attempted runs says so, and says n.

    100 % of 3 runs and 100 % of 623 are not one finding. The clause's own trap,
    stated in #2576's risk notes, is reading a flat outcome on a handful of runs as a
    verdict about thinking — so the marker and the count have to travel together in
    the printed line, and a thin row must read as thin in TEXT and not only in a
    legend the reader may not have reached.
    """
    workers, usage = stores
    result = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    assert THIN_DENOM == 30
    rows = _rows(result)
    assert rows["board-steward"]["outcome_n"] == THIN_DENOM
    assert rows["board-steward"]["outcome_thin"] is False, (
        "a full denominator printed thin, which trains a reader to ignore the marker")
    thin = sorted(c for c, r in rows.items() if r["outcome_thin"])
    assert thin == ["autocode", "autotriage", "queue-maintenance"], (
        "every class under the bar must be marked, and no class above it")
    text = render(result)
    for cls in thin:
        line = next(l for l in text.splitlines() if l.startswith(cls))
        assert "THIN" in line, f"{cls}: a thin denominator printed unmarked"
        assert str(rows[cls]["outcome_n"]) in line, (
            f"{cls}: THIN printed without its n attached")


def test_a_full_denominator_is_not_marked_thin(tmp_path):
    """THIN fires on the denominator, not on the turn count or the coverage.

    Two runs of the same scan over two stores differing only in attempted runs: 30 at
    the bar and 29 below it. The boundary is a threshold a reader reasons about ("its
    n is 131, is that thin?"), so the comparison has to be the one the constant states
    — `< THIN_DENOM`, with a denominator of exactly 30 NOT thin. Both classes carry
    zero turns, so if the marker ever keyed off `turns` or coverage these two rows
    would agree, which is exactly how such a bug hides.
    """
    def scan(attempts: int) -> dict:
        workers, usage = tmp_path / f"w{attempts}.db", tmp_path / f"u{attempts}.db"
        _seed_workers(workers, [("big", f"2026-10-0{5 + i % 5}T09:00:00", "success", [])
                                for i in range(attempts)])
        _seed_usage(usage, [("no-such-session", "2026-10-05T09:00:00", 1, 2)])
        return _rows(build_table(workers_db=workers, usage_db=usage,
                                since=None))["big"]

    at_bar = scan(THIN_DENOM)
    assert (at_bar["outcome_n"], at_bar["outcome_thin"], at_bar["turns"]) == (
        THIN_DENOM, False, 0)
    just_under = scan(THIN_DENOM - 1)
    assert (just_under["outcome_n"], just_under["outcome_thin"],
            just_under["turns"]) == (THIN_DENOM - 1, True, 0)


# ── the witness's own arithmetic, and the report's dependence on it ─────────


def test_the_json_carries_the_sums_the_reported_share_is_derived_from():
    """The rounded columns must be re-derivable exactly, not to within a tolerance.

    `share` is `round(sum(r)/sum(o), 4)` and both means are `round(..., 1)`, so a
    check that re-derives the share from the printed means can only agree to ~1e-4 —
    and a check with a 1e-4 tolerance cannot see a share that drifted from its own
    sums, which is the drift that already happened once here (the report's prose said
    10,403 rows while the accounting line below it said 10,404). The witness therefore
    carries `reasoning_total`/`output_total` at full precision, and this node asserts
    every printed figure against them.
    """
    witness = _unwrap_witness(CLASS_WITNESS)
    for row in witness["rows"]:
        if row["reasoning_coverage"] == 0:
            assert row["reasoning_total"] is None and row["output_total"] is None
            continue
        assert row["reasoning_share_of_output"] == round(
            row["reasoning_total"] / row["output_total"], 4)
        assert row["mean_reasoning_tokens"] == round(
            row["reasoning_total"] / row["reasoning_coverage"], 1)
        assert row["mean_output_tokens"] == round(
            row["output_total"] / row["reasoning_coverage"], 1)
    assert witness["usage_rows"] == witness["turns_attributed"] + next(
        r["turns"] for r in witness["rows"] if r["class"] == UNATTRIBUTED)


def test_the_witness_flag_writes_the_bytes_the_report_reads(tmp_path, stores):
    """The `--witness-out` route round-trips: what the script writes is what it read.

    Both witnesses used to be pasted output, so the tracked bytes had no provenance
    and nothing tied them to the script. Now one flag writes them, so the tie is a
    property of the route rather than of someone's transcription. The wrapper is the
    tracked shape — a `.md` with one fenced JSON block, because `*.json` is ignored
    repo-wide and `.gitignore` is a denied path for a round — so the node unwraps it
    the same way the report-pinning sibling does.
    """
    workers, usage = stores
    out = tmp_path / "nested" / "witness.md"
    result = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    write_witness(result, out, flags="--all")
    text = out.read_text()
    assert text.startswith("# #2576 per-job-class") and "--all" in text.splitlines()[2]
    assert _unwrap_witness(out) == json.loads(json.dumps(result))
    for needle in ("No prompt", "session id"):
        assert needle in text, (
            "the wrapper stopped saying what is NOT in these bytes, which is the "
            "property that lets a census of every turn be committed at all")
    assert "turns_attributed" in text


# ── #2798: the displacement columns ─────────────────────────────────────────


def _md_cells_by_class(text: str) -> dict[str, list[str]]:
    """A rendered markdown table as `{first cell: [cells]}`, header keyed `class`."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        out[cells[0]] = cells
    return out


def _aligned_rows(text: str, classes: set[str]) -> dict[str, list[str]]:
    """The aligned table's body lines as `{class: [whitespace fields]}`.

    Selected by the class names rather than by a pattern over the page: the render's
    legend lines are prose that opens with the columns' own names (`meanR = ...`,
    `turns attributed ...`), so a prefix rule has to be as clever as the format to
    keep them out, and a rule that is wrong here reads as a missing row.
    """
    return {fields[0]: fields for fields in
            (line.split() for line in text.splitlines())
            if fields and fields[0] in classes}


def test_2798_clause_1_the_json_row_carries_the_three_ratios_with_their_denominators(stores):
    """Clause 1: three displacement fields per `--json` row, each over its own subset.

    `autotriage` is the discriminating class: 4 turns, only 3 of them carrying a
    tool-call counter, of which one hit the budget, and calls/errors that aggregate to
    3/12 = 0.25 while the mean of the three turns' own ratios is (0.25 + 0.25 + 0)/3 =
    0.1667. 0.25 and 0.1667 are a 1.5x difference in a class's error rate, and it is
    the same trap `share` pins two nodes above, one column over.
    """
    workers, usage = stores
    result = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    rows = _rows(result)
    for row in result["rows"]:
        for key in ("mean_tool_calls", "errors_per_tool_call", "length_bound_turns",
                    "tool_rows", "stop_rows", "tool_calls_total", "tool_errors_total"):
            assert key in row, f"{row['class']}: the --json row lost {key}"
        # Each printed figure re-derives from its own sums, as `share` does above: a
        # tolerance would not see a ratio that drifted away from its denominator.
        if row["tool_rows"]:
            assert row["mean_tool_calls"] == round(
                row["tool_calls_total"] / row["tool_rows"], 1)
            assert row["errors_per_tool_call"] == round(
                row["tool_errors_total"] / row["tool_calls_total"], 4)
        else:
            assert (row["tool_calls_total"], row["tool_errors_total"]) == (None, None)

    tri = rows["autotriage"]
    assert (tri["turns"], tri["tool_rows"], tri["stop_rows"]) == (4, 3, 3)
    assert (tri["tool_calls_total"], tri["tool_errors_total"]) == (12, 3)
    assert tri["mean_tool_calls"] == 4.0 and tri["errors_per_tool_call"] == 0.25
    assert tri["length_bound_turns"] == 1
    assert tri["errors_per_tool_call"] != pytest.approx((0.25 + 0.25 + 0.0) / 3), (
        "errors per call drifted to the mean of per-turn ratios")

    code = rows["autocode"]
    # One of autocode's two turns has NULL telemetry, so its mean is over ONE row.
    # Divided by `turns` the figure would be 5.0 — half its real value, because the
    # unmeasured turn was priced as a turn that called nothing, which is the inversion
    # `app/usage_store.py::_TELEMETRY_COLUMNS` warns about in its own comment.
    assert (code["turns"], code["tool_rows"], code["stop_rows"]) == (2, 1, 1)
    assert code["mean_tool_calls"] == 10.0 and code["errors_per_tool_call"] == 0.0
    assert code["length_bound_turns"] == 0

    residual = rows[UNATTRIBUTED]
    assert (residual["mean_tool_calls"], residual["errors_per_tool_call"],
            residual["length_bound_turns"]) == (2.0, 1.0, 1)

    # A class with runs and no turns was never measured on either counter, so all
    # three are null. A 0 here would say its turns were counted and never hit the
    # budget — a finding about the mechanism, printed about a class that made none.
    for cls in ("queue-maintenance", "board-steward"):
        row = rows[cls]
        assert (row["tool_rows"], row["stop_rows"]) == (0, 0)
        assert row["mean_tool_calls"] is None, f"{cls} invented a tool mean from nothing"
        assert row["errors_per_tool_call"] is None
        assert row["length_bound_turns"] is None, f"{cls}: null is the finding, not 0"


def test_2798_clause_1_a_measured_zero_is_not_the_unmeasured_null(tmp_path):
    """`0` and NULL are different findings on all three, and each column decides alone.

    Three classes, one per shape: `no-calls` measured a turn that called nothing (real
    zeros, in every rendering); `no-telemetry` measured nothing at all (the sentinel);
    and `stop-only` carries a stop reason with no tool counter, so its length figure is
    a real 0 while its two ratios are null — the shape that tells the columns' two
    denominators apart. Today's live store cannot produce that third shape, because
    `stop_reason`, `tool_calls` and `tool_errors` all arrive from the same migration
    (`stop_reason` is not in the original `create table usage`, and the 2026-10-16
    check found 0 rows where one of the three is NULL and another is not), so the
    distinction is seeded here rather than left unasserted.
    """
    workers, usage = tmp_path / "workers.db", tmp_path / "usage.db"
    _seed_workers(workers, [
        ("no-calls", "2026-10-05T09:00:00", "success", ["s-z"]),
        ("no-telemetry", "2026-10-05T09:00:00", "success", ["s-n"]),
        ("stop-only", "2026-10-05T09:00:00", "success", ["s-s"])])
    _seed_usage(usage, [
        ("s-z", "2026-10-05T09:10:00", 5, 20, "stop", 0, 0),
        ("s-n", "2026-10-05T09:10:00", 5, 20),
        ("s-s", "2026-10-05T09:10:00", 5, 20, "max_turns", None, None)])
    result = build_table(workers_db=workers, usage_db=usage, since=None)
    rows = _rows(result)

    measured = rows["no-calls"]
    assert (measured["tool_rows"], measured["stop_rows"]) == (1, 1)
    assert (measured["mean_tool_calls"], measured["errors_per_tool_call"],
            measured["length_bound_turns"]) == (0.0, 0.0, 0)
    unmeasured = rows["no-telemetry"]
    assert (unmeasured["tool_rows"], unmeasured["stop_rows"]) == (0, 0)
    assert unmeasured["mean_tool_calls"] is None
    assert unmeasured["errors_per_tool_call"] is None
    assert unmeasured["length_bound_turns"] is None
    mixed = rows["stop-only"]
    assert (mixed["tool_rows"], mixed["stop_rows"]) == (0, 1)
    assert (mixed["mean_tool_calls"], mixed["errors_per_tool_call"]) == (None, None)
    assert mixed["length_bound_turns"] == 0, (
        "the class's one turn ended max_turns, which is a measured turn that did NOT "
        "hit the length budget — a null here would erase the negative result")

    cells = _md_cells_by_class(markdown_rows(result))
    assert cells["no-calls"][-3:] == ["0.0 n=1", "0.000", "0 n=1"]
    assert cells["no-telemetry"][-3:] == [NO_VALUE] * 3
    assert cells["stop-only"][-3:] == [NO_VALUE, NO_VALUE, "0 n=1"]
    aligned = _aligned_rows(render(result), set(rows))
    assert aligned["no-calls"][-3:] == ["0.0", "0.000", "0"]
    assert aligned["no-telemetry"][-3:] == ["-", "-", "-"]
    assert aligned["stop-only"][-3:] == ["-", "-", "0"]


def test_2798_clause_2_every_displacement_cell_is_that_rows_own_in_both_tables(stores):
    """Clause 2: every printed cell is formatted from ITS row's fields, in both tables.

    #2576's third review round caught a renderer that printed one equal ratio on every
    row, so "row-scoped" is asserted here by walking the rendered text and re-deriving
    each cell from the `--json` row of the SAME class. A renderer that hoisted one
    class's value into every row, or read the columns off the table instead of the row,
    fails on the first class whose figure differs — and the spread assertion at the end
    closes the case where every class would have printed the same thing anyway.
    """
    workers, usage = stores
    result = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    rows = _rows(result)
    md = markdown_rows(result)
    header = md.splitlines()[0]
    assert [c.strip() for c in header.strip("|").split("|")][-3:] == [
        "meanT", "errPerCall", "lenBound"], header
    cells, aligned = _md_cells_by_class(md), _aligned_rows(render(result), set(rows))
    assert set(cells) - {"class"} == set(rows) == set(aligned)
    for cls, row in rows.items():
        want = ([NO_VALUE, NO_VALUE, NO_VALUE] if not row["tool_rows"] else
                [f"{row['mean_tool_calls']:,.1f} n={row['tool_rows']:,}",
                 f"{row['errors_per_tool_call']:,.3f}",
                 (f"{row['length_bound_turns']:,} n={row['stop_rows']:,}"
                  if row["length_bound_turns"] is not None else NO_VALUE)])
        assert cells[cls][-3:] == want, (
            f"{cls}: markdown printed {cells[cls][-3:]} where its own row says {want}")
        # The aligned table's own sentinel is a hyphen, not the report's en dash: the
        # two tables spell "nothing measured" differently and both are pinned here.
        assert aligned[cls][-3:] == ["-" if w == NO_VALUE else w.split(" n=")[0]
                                     for w in want], (
            f"{cls}: the aligned table printed {aligned[cls][-3:]} for row {want}")
    measured = [r for r in rows.values() if r["tool_rows"]]
    assert len(measured) >= 3, "the fixture lost the classes the spread is read across"
    assert len({(r["mean_tool_calls"], r["errors_per_tool_call"])
                for r in measured}) >= 3, (
        "every measured class printed the same two ratios, which is the shared-value "
        "shape this node exists to catch")


def test_2798_clause_3_mutating_one_witness_row_moves_only_that_row():
    """Clause 3: the committed witness, mutated in one row, re-rendered both ways.

    The 2026-10-11 witness is the artifact the report's table is pinned to, and its
    rows carry none of the displacement keys — so it is the natural corpus to add them
    to in order to prove the cells are row-scoped: two classes gain three cells with
    stated values, every other class may gain ONLY the sentinel, and no class may
    inherit a neighbour's numbers. Both renderings come off the same mutated dict, so
    a renderer reading the table rather than the row shows up as a row that did not
    move.
    """
    witness = _unwrap_witness(CLASS_WITNESS)
    base_md, base_aligned = markdown_rows(witness), render(witness)
    assert "meanT" not in base_md, (
        "the pinned 2026-10-11 render grew columns on its own, so this node's "
        "before-side is no longer the bytes the report pins")

    mutated = json.loads(json.dumps(witness))
    moved = {"autocode": (40.0, 0.25, 3), "owed-check": (1.5, 0.0, 0)}
    for cls, (mean_t, err, length) in moved.items():
        next(r for r in mutated["rows"] if r["class"] == cls).update(
            mean_tool_calls=mean_t, errors_per_tool_call=err,
            length_bound_turns=length, tool_rows=100, stop_rows=100,
            tool_calls_total=100, tool_errors_total=100)

    md, aligned = markdown_rows(mutated), render(mutated)
    cells, base_cells = _md_cells_by_class(md), _md_cells_by_class(base_md)
    assert set(cells) == set(base_cells), "a class appeared or vanished mid-render"
    for cls, (mean_t, err, length) in moved.items():
        assert cells[cls][-3:] == [f"{mean_t:,.1f} n=100", f"{err:,.3f}",
                                   f"{length:,} n=100"], f"{cls}'s own cells"
    for cls, row in cells.items():
        if cls == "class":
            continue
        assert row[:11] == base_cells[cls][:11], (
            f"{cls}: one of the eleven pinned columns moved when only two OTHER rows "
            "were given displacement fields")
        if cls not in moved:
            assert row[-3:] == [NO_VALUE] * 3, (
                f"{cls} printed {row[-3:]} without carrying the fields")
    assert cells["autocode"][-3:] != cells["owed-check"][-3:], (
        "the two mutated classes print the same triple, which is the shared-value bug")
    header = md.splitlines()[0]
    assert header.count("|") == base_md.splitlines()[0].count("|") + 3
    ragged = {cls: len(row) for cls, row in cells.items() if len(row) != 14}
    assert not ragged, f"markdown rows of differing width: {ragged}"

    known = {c for c in cells if c != "class"}
    after, before = _aligned_rows(aligned, known), _aligned_rows(base_aligned, known)
    for cls, (mean_t, err, length) in moved.items():
        assert after[cls][-3:] == [f"{mean_t:,.1f}", f"{err:,.3f}", f"{length:,d}"]
    for cls, fields in after.items():
        assert fields[:-3] == before[cls], (
            f"{cls}: the aligned row changed before its three new cells")
        if cls not in moved:
            assert fields[-3:] == ["-"] * 3


def test_2798_clause_3_mutating_one_usage_row_moves_only_its_class(stores):
    """Clause 3 across the store boundary: one UPDATE on `usage` moves one class.

    The witness node mutates a JSON document; this one mutates the fixture's SQLite
    table, which `_seed_usage` builds with the production `_init_schema`, so the
    statement is the one an operator would run against `usage.db` and a SELECT that
    stopped naming `stop_reason`, `tool_calls` or `tool_errors` would leave every
    figure here unchanged. Rows are compared as whole JSON documents, so a change that
    leaked into a second class is named rather than averaged away.
    """
    workers, usage = stores
    before = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")
    con = sqlite3.connect(usage)
    try:
        con.execute("update usage set stop_reason='length', tool_calls=99, tool_errors=7"
                    " where session_id='s-code-b'")
        con.commit()
    finally:
        con.close()
    after = build_table(workers_db=workers, usage_db=usage, since="2026-01-01T00:00:00")

    changed = {}
    for row in after["rows"]:
        old = next(r for r in before["rows"] if r["class"] == row["class"])
        if json.dumps(row, sort_keys=True) != json.dumps(old, sort_keys=True):
            changed[row["class"]] = (old, row)
    assert set(changed) == {"autocode"}, (
        f"one row's telemetry moved {sorted(changed)}; the columns are not scoped to "
        "the class whose row changed")
    old, new = changed["autocode"]
    assert (old["tool_rows"], new["tool_rows"]) == (1, 2), "the row did not enter the denominator"
    assert (old["mean_tool_calls"], new["mean_tool_calls"]) == (10.0, round(109 / 2, 1))
    assert new["errors_per_tool_call"] == round(7 / 109, 4)
    assert (old["length_bound_turns"], new["length_bound_turns"]) == (0, 1)
    assert markdown_rows(after).count("54.5 n=2") == 1
    assert "54.5 n=2" not in markdown_rows(before)
    assert _aligned_rows(render(after), {"autocode"})["autocode"][-3:] == [
        "54.5", "0.064", "1"]
    assert _aligned_rows(render(before), {"autocode"})["autocode"][-3:] == [
        "10.0", "0.000", "0"]


def test_2798_clause_4_the_committed_witness_still_renders_eleven_columns():
    """Clause 4: the 2026-10-11 witness is untouched, and still renders what it rendered.

    Its rows carry none of the displacement keys, which is the only reason the new
    columns are keyed off the rows: `tests/test_thinking_effort_report.py` byte-pins
    this render inside the vault report and re-derives the report's prose figures from
    it. The header is typed here rather than imported from `MD_COLUMNS` so the
    assertion cannot be true by definition — the module constant and a test that reads
    it agree whatever the renderer grows.
    """
    witness = _unwrap_witness(CLASS_WITNESS)
    for row in witness["rows"]:
        assert not ({"mean_tool_calls", "errors_per_tool_call",
                     "length_bound_turns"} & set(row)), (
            "the committed witness gained displacement keys, so it no longer proves "
            "that an unmeasured corpus still renders the eleven pinned columns")
    md = markdown_rows(witness)
    assert md.splitlines()[0] == ("| class | sess | turns | cover | meanR | p90R "
                                  "| meanOut | meanIn | share | outcome | s/f/s/i |")
    ragged = {cls: len(row) for cls, row in _md_cells_by_class(md).items()
              if len(row) != 11}
    assert not ragged, f"the witness no longer renders 11 cells per row: {ragged}"
    aligned = render(witness).splitlines()
    header_line = next(l for l in aligned if l.startswith("class"))
    assert "meanT" not in header_line, (
        "the aligned table grew a displacement column for a corpus that has none")
    classes = {row["class"] for row in witness["rows"]}
    # Body rows are the header's width, or one wider: `THIN` is a trailing marker, and
    # it is the only thing in this render that ever was.
    too_wide = [l for l in aligned if l.split() and l.split()[0] in classes
                and len(l.split()) > len(header_line.split()) + 1]
    assert not too_wide, f"aligned rows wider than the header: {too_wide}"


def test_no_node_in_this_file_skips():
    """Clause 3's last requirement, as a node that runs with the suite rather than a grep.

    Checked over the whole file including the helpers, because a skip inside a helper
    hides a clause's nodes behind a green run — which is the failure mode `_measured()`
    was written to avoid. Only the executable spellings are searched: the word "skip"
    appears in prose below, and prose is not a node.
    """
    text = Path(__file__).read_text()
    # Each needle is built at run time, because writing any of them as a literal in
    # this file would make this node fail on its own source.
    for needle in ("pytest" + "." + "skip", "@pytest.mark." + "skip", "skip" + "if"):
        assert needle not in text, f"{needle} entered this file"


def test_every_node_named_in_the_clause_map_exists():
    """The clause map in this file's docstring is the review's index, so it cannot rot.

    A node renamed or deleted leaves its clause pointing at nothing, and the review
    rung reads that map as the clause-to-evidence table. Every backticked `test_*`
    name in the module docstring has to be a function defined in this file.
    """
    import re

    text = Path(__file__).read_text()
    docstring = text.split('"""', 2)[1]
    named = set(re.findall(r"`(test_[a-z0-9_]+)`", docstring))
    assert named, "the clause map stopped naming any node, so it grades nothing"
    missing = [name for name in sorted(named) if f"def {name}(" not in text]
    assert not missing, f"the clause map names nodes this file does not define: {missing}"


def test_a_missing_store_is_refused_rather_than_created(tmp_path):
    """`sqlite3.connect` on a bare path fabricates a 0-byte database.

    A census over a just-created empty store prints a beautifully clean empty table,
    and an empty table is the false-negative shape #2576's coverage requirement keeps
    tripping over — so `_read_only` opens `mode=ro` and refuses a path that is not
    there. Asserted because nothing else in this file would notice its removal: the
    other nodes all hand it a store that exists.
    """
    workers, usage = tmp_path / "absent-workers.db", tmp_path / "absent-usage.db"
    with pytest.raises(SystemExit) as caught:
        build_table(workers_db=workers, usage_db=usage, since=None)
    assert "no such store" in str(caught.value)
    assert not workers.exists(), "the refusal still created the store it refused"
