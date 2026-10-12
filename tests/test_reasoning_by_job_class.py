"""#2576 clauses 1-3: the per-job-class join, from two stores and out one table.

Clause → node map (the review rung grades per clause, so every clause names its
nodes here rather than leaving them to be found by title):

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


def _seed_usage(path: Path, rows: list[tuple[str | None, str, int | None, int]]) -> None:
    """One row per turn. A `None` reasoning is the NULL the covered subset exists for.

    The column list is the whole `usage` table from `app/usage_store.py::_SCHEMA`
    except the indexes: a fixture carrying only the three columns the query names
    would stay green after a rename of the other eight, and those eight are what the
    report's `mean output` and `share` columns silently depend on.
    """
    con = sqlite3.connect(path)
    try:
        con.execute("create table usage (ts text, source text, kind text,"
                    " session_id text, job_id integer, model text, input_tokens integer,"
                    " output_tokens integer, reasoning_tokens integer,"
                    " cached_tokens integer, cost_usd real)")
        con.executemany("insert into usage values (?,?,?,?,?,?,?,?,?,?,?)",
                        [(t, "autocode", "turn", sid, None, "Qwen3.8-Flash-Next",
                          100, out, reason, None, 0.0)
                         for sid, t, reason, out in rows])
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
        # autotriage: one covered, one NULL reasoning, one NULL output (guarded).
        ("s-tri-a", "2026-10-05T10:00:05", 40, 100),
        ("s-tri-b", "2026-10-06T10:00:05", None, 90),
        ("s-tri-c", "2026-10-07T10:00:05", 60, None),
        # autotriage's own turn, plus autocode's — both covered.
        ("s-tri-d", "2026-10-08T10:00:05", 20, 50),
        ("s-code-a", "2026-10-05T09:00:05", 10, 20),
        ("s-code-b", "2026-10-05T09:05:05", 10, 180),
        # Unattributed: no run names these. Including a NULL session_id, which is a
        # real shape in the store and must land in the residual, not crash the join.
        ("s-orphan-1", "2026-10-05T12:00:00", 5, 20),
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
