"""Per-mechanism re-prefill attribution (#2027).

`app.usage_store.reprefill_attribution` prices a window's prefix-cache re-prefill
per mechanism. These tests pin the three things that make the number trustworthy
rather than merely printed:

* every mean carries its `n=`, and a rung the window holds no row for is printed
  with `n=0` rather than omitted — an absent row is indistinguishable from a
  healthy machine;
* the Σ over the buckets is exhaustive against ONE named total, computed over every
  row in the window, so the split cannot be a selection of interesting rungs;
* the freeing predicate answers BOTH paths — `turn_start.tokens_freed` and
  top-level `relief_tokens_freed` — because reading only the first is the mistake
  that put every relief-only turn into the control arm in this item's first draft,
  and a row that freed by both paths is credited exactly once.

Rows are written through `record_usage`, the real writer, so a change to how
`compaction` is stored (#1078's column, and its sanitizer that collapses a blank
record to NULL) is inside what these tests cover.
"""

from __future__ import annotations

import json
import re
import sqlite3

from pathlib import Path

import pytest

from app import usage_store


MICROCOMPACT = {
    "mechanisms": ["microcompact"],
    "turn_start": {
        "ran": True, "mechanisms": ["microcompact"], "microcompacted": 3,
        "tokens_freed": 4000, "tokens_before": 215_000, "tokens_after": 211_000,
        "threshold": 210_144, "context_window": 262_144,
        "summarized": False, "summarize_attempted": False,
        "summarize_outcome": "under_threshold", "summary_folds": 0,
        "truncated": False, "restored_files": 0,
    },
}

RELIEF_ONLY = {
    "mechanisms": ["relief:intra_turn"],
    "relief_passes": 2,
    "relief_tokens_freed": 46_734,
    "relief": [
        {"reason": "intra_turn", "passes": 1, "freed_tokens": 902,
         "rungs": ["tool_results"], "rearm": 151_303, "target": 109_274,
         "used_before": 109_758, "used_after": 108_856},
        {"reason": "intra_turn", "passes": 2, "freed_tokens": 45_832,
         "rungs": ["tool_results", "reasoning:6", "truncate:6"], "rearm": 151_303,
         "target": 109_274, "used_before": 153_335, "used_after": 107_503},
    ],
    "turn_start": {
        "ran": True, "mechanisms": [], "microcompacted": 0, "tokens_freed": 0,
        "tokens_before": 3387, "tokens_after": 3387, "threshold": 210_144,
        "context_window": 262_144, "summarized": False,
        "summarize_attempted": False, "summarize_outcome": "under_threshold",
        "summary_folds": 0, "truncated": False, "restored_files": 0,
    },
}

RAN_NOOP = {
    "mechanisms": [],
    "turn_start": {
        "ran": True, "mechanisms": [], "microcompacted": 0, "tokens_freed": 0,
        "tokens_before": 80_000, "tokens_after": 80_000, "threshold": 210_144,
        "context_window": 262_144, "summarized": False,
        "summarize_attempted": False, "summarize_outcome": "under_threshold",
        "summary_folds": 0, "truncated": False, "restored_files": 0,
    },
}

#: A record that names nothing and claims no run — the live shape is a `mechanisms`
#: list with no `turn_start` block behind it. Not the control: it is an absence of
#: evidence, and must not dilute the arm the ratio is measured against.
NO_EVIDENCE = {"mechanisms": []}

BOTH_PATHS = {
    "mechanisms": ["microcompact", "relief:intra_turn"],
    "relief_passes": 1,
    "relief_tokens_freed": 1_200,
    "relief": [{"reason": "intra_turn", "passes": 1, "freed_tokens": 1_200,
                "rungs": ["tool_results"], "rearm": 151_303, "target": 109_274,
                "used_before": 150_000, "used_after": 148_800}],
    "turn_start": {
        "ran": True, "mechanisms": ["microcompact"], "microcompacted": 2,
        "tokens_freed": 5_000, "tokens_before": 214_000, "tokens_after": 209_000,
        "threshold": 210_144, "context_window": 262_144, "summarized": False,
        "summarize_attempted": False, "summarize_outcome": "under_threshold",
        "summary_folds": 0, "truncated": False, "restored_files": 0,
    },
}


def _ts_old(hours: float = 400.0) -> str:
    """A `ts` string safely outside a 168-hour window, in the column's own
    `%Y-%m-%dT%H:%M:%S` spelling."""
    from datetime import datetime, timedelta

    return (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")


def _ts_recent(hours: float = 1.0) -> str:
    """A `ts` string safely INSIDE a 168-hour window, same spelling as `_ts_old`.

    The counterpart exists because the two are the same hazard in opposite
    directions: a row stamped with a literal date and priced by a RELATIVE window
    (`hours=168`) has a shelf life measured in wall-clock time, and when the
    window's left edge slides past it the table comes back empty and an
    attribution check reads as a broken attribution. `#2053` is that failure
    caught in the wild. Stamp relative to the clock the reader reads, or pin an
    absolute interval on both sides; never mix the two.
    """
    from datetime import datetime, timedelta

    return (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh usage.db. `_conn()` reopens whenever the file behind `DB_PATH` is
    not the one its handle writes to, so pointing that path at the temp file is
    the isolation — no cached-handle teardown needed."""
    db = tmp_path / "usage.db"
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    return db


def _write(db, *, compaction, reprefill, input_tokens=150_000, misses=1):
    usage_store.record_usage(
        "sess-a", "test-model",
        input_tokens=input_tokens, output_tokens=40,
        reprefill_tokens=reprefill, prefix_misses=misses,
        compaction=compaction,
    )


def _row(attr, name):
    for row in attr["buckets"]:
        if row["bucket"] == name:
            return row
    raise AssertionError(f"bucket {name!r} is not in the printed table: "
                         f"{[r['bucket'] for r in attr['buckets']]}")


def test_attribution_covers_the_whole_window_and_prints_n_beside_every_mean(store):
    """Clause 1 + clause 2 together: a stated window, one row per bucket with its
    `n=`, and a Σ that equals the total the header itself names."""
    _write(store, compaction=MICROCOMPACT, reprefill=300_000)
    _write(store, compaction=RELIEF_ONLY, reprefill=220_000)
    _write(store, compaction=RAN_NOOP, reprefill=5_000, input_tokens=80_000)
    _write(store, compaction=None, reprefill=4_000, input_tokens=58_000)
    _write(store, compaction=NO_EVIDENCE, reprefill=1_000, input_tokens=77_000)

    attr = usage_store.reprefill_attribution(hours=168)

    # The window is stated on the report's first line, with its row count.
    head = attr["report"].splitlines()[0]
    assert "window: last 168 h" in head and "rows=5" in head

    # Every bucket gets a row, whether or not the window holds one for it.
    for name in usage_store.ATTRIBUTION_BUCKETS:
        assert re.search(rf"^{re.escape(name)}\s+n=\s*\d+", attr["report"], re.M), (
            f"{name} has no printed row, so an absent rung would read as a "
            f"healthy machine:\n{attr['report']}")

    # …and EXACTLY one row each, in `ATTRIBUTION_BUCKETS`' own front-most-break
    # precedence order. Membership alone survives a table that reordered the buckets
    # or printed one twice, and both of those change which rung reads as the cost
    # centre while every number on every line stays correct. A duplicated Σ would
    # also break the exhaustiveness claim, which is why the count is pinned here and
    # not left to the gap below.
    printed = re.findall(r"^(\S+)\s+n=\s*\d+ measured=", attr["report"], re.M)
    assert printed == list(usage_store.ATTRIBUTION_BUCKETS), (
        "the printed table is not one row per bucket in the precedence order the "
        f"attribution rule is stated in: {printed}")
    # The order is pinned to the rule itself, not only to the tuple: a re-cut tuple
    # moves the line above with it, and the summary rewrite is the front-most break
    # the doc prices, so it must head the table.
    assert printed[:5] == ["turn_start:summarized", "turn_start:summary_folds",
                           "turn_start:truncated", "turn_start:microcompact",
                           "turn_start:other"], printed

    # n= is printed beside every mean, on every line that carries one.
    for line in attr["report"].splitlines():
        if "mean=" in line or "mean reprefill=" in line:
            assert "n=" in line, f"a mean printed without its n=: {line}"

    # A rung the window contains none of prints n=0 and NO mean, rather than
    # going missing or printing a mean of 0.
    images = _row(attr, "relief:images")
    assert images["n"] == 0 and images["mean_reprefill_tokens"] is None
    empty_line = next(l for l in attr["report"].splitlines()
                      if l.startswith("relief:images"))
    assert re.search(r"^relief:images\s+n=\s*0\b", empty_line), empty_line
    # The mean field is right-aligned in a width-10 column, so whitespace
    # collapsed the empty rung reads `mean= -`: a bare dash, never a 0.
    collapsed = re.sub(r" +", " ", empty_line)
    assert "mean= - " in collapsed, (
        "an empty rung printed something other than a bare dash for its mean, "
        "which a reader will take for a measured zero: " + empty_line)

    # The Σ is exhaustive against the total named in the same output.
    conn = sqlite3.connect(store)
    try:
        window_total = conn.execute(
            "SELECT COALESCE(SUM(reprefill_tokens), 0) FROM usage WHERE ts >= ?",
            (attr["since"],)).fetchone()[0]
    finally:
        conn.close()
    assert window_total == 530_000 == attr["window_total_reprefill_tokens"]
    assert attr["bucket_sum_reprefill_tokens"] == window_total
    assert attr["gap_pct"] == 0.0
    assert f"{window_total:,}" in attr["report"]
    # ...and the total was taken over EVERY row of the window, the two that freed
    # nothing and the one that recorded nothing included, not over the attributed
    # subset: 300,000 + 220,000 + 5,000 + 4,000 + 1,000.
    assert _row(attr, "no_compaction_record")["reprefill_tokens"] == 4_000
    assert _row(attr, "no_freeing_evidence")["reprefill_tokens"] == 1_000
    assert _row(attr, "ran_noop")["reprefill_tokens"] == 5_000


def test_the_only_relief_signal_puts_the_row_on_a_rung_and_never_on_the_control(store):
    """Clause 3, first half. This is the predicate trap: a row whose turn-start
    pass freed nothing but whose `relief_tokens_freed` is 46,734 is a FREEING turn.
    A query reading only `turn_start.*` files it under the control and prints a
    plausible, weaker number instead of failing."""
    _write(store, compaction=RELIEF_ONLY, reprefill=220_000)
    _write(store, compaction=RAN_NOOP, reprefill=5_000, input_tokens=80_000)

    attr = usage_store.reprefill_attribution(hours=168)

    tool_results = _row(attr, "relief:tool_results")
    assert tool_results["n"] == 1 and tool_results["reprefill_tokens"] == 220_000
    noop = _row(attr, "ran_noop")
    assert noop["n"] == 1 and noop["reprefill_tokens"] == 5_000, (
        "the relief-only row was counted in the control arm")

    # Both paths in the arms predicate, so the ratio is measured against the right
    # populations: B holds exactly the relief-only row, D exactly the noop.
    assert attr["arms"]["B_freed_something"]["n"] == 1
    assert attr["arms"]["B_freed_something"]["reprefill_tokens"] == 220_000
    assert attr["arms"]["D_ran_noop"]["n"] == 1
    assert attr["arms"]["A_no_record"]["n"] == 0
    # A rung that fired but was not the credited owner reports fired_= as well as
    # n=0, so the empty row cannot be read as "the rung never ran": this row's
    # first freeing pass named tool_results first, and its second named
    # tool_results, reasoning and truncate.
    assert _row(attr, "relief:reasoning")["n"] == 0
    assert _row(attr, "relief:reasoning")["fired_rows"] == 1
    assert _row(attr, "relief:truncate")["fired_rows"] == 1


def test_a_row_that_freed_on_both_paths_is_credited_exactly_once(store):
    """Clause 3, second half. `["microcompact", "relief:intra_turn"]` in one row is
    the common shape, not an edge case — the top-reprefill rows all carry it — and
    the record holds one `reprefill_tokens` per row with no per-rung split. So a
    row is credited wholly to one bucket, under the front-most edit, and the
    buckets that did not take it stay at n=0 while still reporting fired_=1."""
    _write(store, compaction=BOTH_PATHS, reprefill=90_000)

    attr = usage_store.reprefill_attribution(hours=168)

    assert _row(attr, "turn_start:microcompact")["n"] == 1
    assert _row(attr, "turn_start:microcompact")["reprefill_tokens"] == 90_000
    assert _row(attr, "relief:tool_results")["n"] == 0
    assert _row(attr, "relief:tool_results")["fired_rows"] == 1
    assert sum(r["n"] for r in attr["buckets"]) == 1, (
        "a row was credited to more than one bucket, or to none")
    assert attr["bucket_sum_reprefill_tokens"] == attr["window_total_reprefill_tokens"]
    # The arms table, which is the item's own predicate, still counts the row as B.
    assert attr["arms"]["B_freed_something"]["n"] == 1


def test_front_most_break_owns_the_row(store):
    """The assignment rule, stated in `ATTRIBUTION_BUCKETS` and pinned here: a row
    is credited to the front-most mechanism that fired, because §4.3 of
    arXiv:2609.37725v1 prices an edit by everything AFTER its break point, so the
    earliest break is what the re-prefill is paying for. A summary rewrite and a
    fold both sit at position ~0, as does a truncation that drops the oldest turns,
    so their relative order is a deterministic tie-break and not a claim about
    which of the three is dearer."""
    summarized = {
        "mechanisms": ["summarize", "microcompact"],
        "turn_start": dict(MICROCOMPACT["turn_start"], summarized=True,
                           summary_folds=1, truncated=True),
    }
    _write(store, compaction=summarized, reprefill=100_000)

    ladder = {
        "mechanisms": ["relief:intra_turn"],
        "relief_passes": 1, "relief_tokens_freed": 900,
        "relief": [{"reason": "intra_turn", "passes": 1, "freed_tokens": 900,
                    "rungs": ["tool_results", "images", "reasoning:4"],
                    "rearm": 1, "target": 1, "used_before": 2, "used_after": 1}],
        "turn_start": dict(RAN_NOOP["turn_start"]),
    }
    _write(store, compaction=ladder, reprefill=70_000)

    attr = usage_store.reprefill_attribution(hours=168)
    assert _row(attr, "turn_start:summarized")["n"] == 1
    assert _row(attr, "turn_start:summary_folds")["n"] == 0
    assert _row(attr, "turn_start:truncated")["n"] == 0
    assert _row(attr, "turn_start:microcompact")["n"] == 0
    # The relief half: the ladder appends its rungs in run order
    # (`app/harness/loop.py:1795-:1884`), so the first name in the list is the
    # earliest rung that edited — `tool_results`, not the `images` behind it.
    assert _row(attr, "relief:tool_results")["n"] == 1
    assert _row(attr, "relief:images")["fired_rows"] == 1


def test_an_unrecognised_rung_name_is_folded_and_reported_not_dropped(store):
    """A rung the ladder constant does not know must not vanish: its row is folded
    into `relief:other` (so the Σ stays exhaustive — appending a second row naming
    the same tokens would double-count it) and its name is printed."""
    novel = {
        "mechanisms": ["relief:intra_turn"],
        "relief_passes": 1, "relief_tokens_freed": 500,
        "relief": [{"reason": "intra_turn", "passes": 1, "freed_tokens": 500,
                    "rungs": ["memories:3"], "rearm": 1, "target": 1,
                    "used_before": 2, "used_after": 1}],
        "turn_start": dict(RAN_NOOP["turn_start"]),
    }
    _write(store, compaction=novel, reprefill=10_000)

    attr = usage_store.reprefill_attribution(hours=168)
    assert _row(attr, "relief:other")["n"] == 1
    assert _row(attr, "relief:other")["reprefill_tokens"] == 10_000
    assert attr["unknown_relief_rungs"] == ["relief:memories"]
    assert "relief:memories" in attr["report"]
    assert attr["gap_pct"] == 0.0


def test_means_are_taken_over_measured_rows_and_a_null_is_not_zero(store):
    """A NULL `reprefill_tokens` is unmeasured, not zero (the same rule
    `prefix_miss_summary` prints `turns_measured` for). Averaging it as a zero
    would halve a rung's mean and the table would still look fine."""
    _write(store, compaction=MICROCOMPACT, reprefill=300_000)
    _write(store, compaction=MICROCOMPACT, reprefill=None, misses=None)

    attr = usage_store.reprefill_attribution(hours=168)
    row = _row(attr, "turn_start:microcompact")
    assert row["n"] == 2 and row["measured"] == 1
    assert row["mean_reprefill_tokens"] == 300_000, (
        "the unmeasured row was averaged in as a zero")
    assert row["reprefill_tokens"] == 300_000
    # Unmeasured rows are out of the arms table, which is measured-only.
    assert attr["arms"]["B_freed_something"]["n"] == 1


def test_the_named_total_is_not_the_space_spelled_sqlite_window(store):
    """The reason this item's own two totals disagreed, and the reason a shell
    re-run has to quote the printed `since` verbatim.

    `ts` carries a literal `T` between date and time
    (`strftime('%Y-%m-%dT%H:%M:%S','now')`), and sqlite renders
    `datetime('now','-7 day')` with a SPACE. String comparison then puts every row
    of the boundary day inside the space-spelled window, because 'T' (0x54) sorts
    above ' ' (0x20) — so that predicate sums a WIDER window, and the pair reads as
    two kinds of total when it is one expression over two windows.
    """
    _write(store, compaction=None, reprefill=100, input_tokens=58_000)
    _write(store, compaction=RAN_NOOP, reprefill=200, input_tokens=80_000)
    boundary = usage_store._since(hours=168)[:10]      # the window's own date

    conn = sqlite3.connect(store)
    try:
        conn.execute("UPDATE usage SET ts = ? WHERE reprefill_tokens = 100",
                     (f"{boundary}T00:01:01",))
        conn.commit()
    finally:
        conn.close()

    attr = usage_store.reprefill_attribution(hours=168)

    assert attr["window_total_reprefill_tokens"] == 200
    # The row at 00:01:01 on the boundary day is inside the space-spelled window
    # and outside this one, so the two totals differ by exactly its 100 tokens.
    assert attr["sqlite_spelling_total_reprefill_tokens"] == 300
    # The behavioural pin is the 200-vs-300 pair above, not the wording of a label
    # this same diff writes in app/usage_store.py.
    assert f"{attr['sqlite_spelling_total_reprefill_tokens']:,}" in attr["report"]
    # And the total the Σ is exhaustive against is the narrow one, printed in the
    # header — not the wider number a careless re-run would arrive at.
    assert f"{attr['window_total_reprefill_tokens']:,}" in attr["report"].splitlines()[1]


def test_the_arms_table_reproduces_the_three_arm_predicate(store):
    """The arms block is the item's own table, so it must land every row where the
    item's predicate lands it: NULL record = A, either freeing path > 0 = B, a
    record that freed nothing = D — and its ratio carries the size caveat instead of
    quoting 52x as though it were controlled."""
    _write(store, compaction=None, reprefill=4_000, input_tokens=58_000)
    _write(store, compaction=MICROCOMPACT, reprefill=300_000)
    _write(store, compaction=RELIEF_ONLY, reprefill=220_000)
    _write(store, compaction=RAN_NOOP, reprefill=5_000, input_tokens=80_000)
    _write(store, compaction=RAN_NOOP, reprefill=6_000, input_tokens=81_000)

    attr = usage_store.reprefill_attribution(hours=168)
    arms = attr["arms"]
    assert (arms["A_no_record"]["n"], arms["B_freed_something"]["n"],
            arms["D_ran_noop"]["n"]) == (1, 2, 2)
    assert arms["B_freed_something"]["reprefill_tokens"] == 520_000
    assert arms["D_ran_noop"]["reprefill_tokens"] == 11_000
    assert attr["ratio_b_over_d_mean"] == pytest.approx(
        (520_000 / 2) / (11_000 / 2), abs=0.2)
    # Not size-controlled, and the output says so rather than quoting the ratio
    # clean: B's mean input is 150,000 against D's 80,500.
    assert attr["size_controlled"] is False
    assert "size-controlled: False" in attr["report"]
    assert "mean input=150,000" in attr["report"]
    assert "mean input=80,500" in attr["report"]


def test_a_row_outside_the_window_is_not_attributed(store):
    """The window is applied at the query, not assumed. `ts` defaults to the insert
    time, so this row is pushed back 400 hours to sit outside a 168-hour window."""
    _write(store, compaction=MICROCOMPACT, reprefill=300_000)
    _write(store, compaction=RELIEF_ONLY, reprefill=220_000)
    conn = sqlite3.connect(store)
    try:
        conn.execute("UPDATE usage SET ts = ? WHERE reprefill_tokens = 220000",
                     (_ts_old(),))
        conn.commit()
    finally:
        conn.close()

    attr = usage_store.reprefill_attribution(hours=168)
    assert attr["rows"] == 1
    assert attr["window_total_reprefill_tokens"] == 300_000
    assert _row(attr, "relief:tool_results")["n"] == 0
    assert sum(r["n"] for r in attr["buckets"]) == 1


def test_an_absolute_window_closes_at_its_until(tmp_path, monkeypatch):
    """`until` is a bound, not a caption.

    `reprefill_attribution` prints an absolute window as `SINCE <= ts < UNTIL`, and
    the published replay passes both. A row at or after `until` has to be out of the
    named total, out of every bucket and out of the arms table: an extract re-cut a
    day after its stated `until` would otherwise widen the window while printing the
    closed interval unchanged, and the Σ would still be internally exhaustive, so the
    gap check would pass on the wider population. Rows go in through
    `replay_usage_extract`, the route the published figures are re-derived by, so the
    bound is tested on the same path a reader re-runs.
    """
    extract = tmp_path / "closed.jsonl"
    extract.write_text(
        json.dumps({"ts": usage_store.REPREFILL_WITNESS_SINCE, "input_tokens": 80_000,
                    "reprefill_tokens": 5_000, "compaction": None}) + "\n"
        # One second before `until`: the last row the window holds.
        + json.dumps({"ts": "2026-10-01T16:28:37", "input_tokens": 150_000,
                      "reprefill_tokens": 400_000,
                      "compaction": MICROCOMPACT}) + "\n"
        # Exactly `until`, and one row behind it: both outside a half-open interval.
        + json.dumps({"ts": usage_store.REPREFILL_WITNESS_UNTIL, "input_tokens": 150_000,
                      "reprefill_tokens": 900_000, "compaction": MICROCOMPACT}) + "\n"
        + json.dumps({"ts": "2026-10-04T09:00:00", "input_tokens": 150_000,
                      "reprefill_tokens": 900_000,
                      "compaction": RELIEF_ONLY}) + "\n")
    db = usage_store.replay_usage_extract(extract, db_path=tmp_path / "closed.db")
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    attr = usage_store.reprefill_attribution(
        since=usage_store.REPREFILL_WITNESS_SINCE,
        until=usage_store.REPREFILL_WITNESS_UNTIL)

    assert attr["window_absolute"] is True and attr["rows"] == 2, attr["rows"]
    assert attr["window_total_reprefill_tokens"] == 405_000, (
        "a row at or after `until` was priced into the named total, which the "
        "header prints as a closed interval")
    assert attr["bucket_sum_reprefill_tokens"] == 405_000 == \
        sum(b["reprefill_tokens"] for b in attr["buckets"])
    assert attr["gap_pct"] == 0.0
    # The post-`until` relief row is out of the buckets AND out of the arms table,
    # so `turn_start:microcompact` holds the one in-window freeing row alone.
    assert _row(attr, "turn_start:microcompact")["n"] == 1
    assert _row(attr, "relief:tool_results")["n"] == 0
    assert _row(attr, "relief:tool_results")["fired_rows"] is None, (
        "no in-window row's record names that rung, so fired_ must read None "
        "(unmeasured) and not 0 — and a post-`until` row must not reach it")
    assert sum(a["n"] for a in attr["arms"].values()) == 2
    # And the closed interval is what the reader is told.
    assert f"{usage_store.REPREFILL_WITNESS_SINCE} <= ts " \
        f"< {usage_store.REPREFILL_WITNESS_UNTIL}" in attr["report"].splitlines()[0]


# --- the vault extract the #2027 verdict is priced on -----------------------
#
# `backlog/data/2026-10-01.2027-reprefill-witness.jsonl` is 3,868 rows of the
# 2026-09-24T16:28:38 .. 2026-10-01T16:28:38 window — the interval the published
# arm table and per-mechanism split were measured on, kept as an absolute window so
# the figures stay checkable after the rolling 168 h moves past them. These nodes
# pin the two ways that extract can silently lie.

def test_a_null_compaction_in_the_extract_stays_sql_null(tmp_path):
    """JSON `null` must replay as SQL NULL, not as the four-character string
    `'null'`.

    The distinction decides an arm: SQL NULL is `no_compaction_record` (arm A, "the
    writer recorded nothing"), while `'null'` parses to a non-mapping and is read as
    `no_freeing_evidence` ("a record exists, and names nothing"). A trial replay
    that dumped nulls through `json.dumps` moved 1,458 live rows out of arm A, took
    7,195,735 tokens off that arm's mean and shifted the headline from 51.8 to 52.3
    — with every Σ intact, so the window-total check passes and the ratio published
    is wrong.
    """
    extract = tmp_path / "extract.jsonl"
    extract.write_text(
        json.dumps({"ts": "2026-09-25T10:00:00", "input_tokens": 60_000,
                    "reprefill_tokens": 1_000, "compaction": None}) + "\n"
        + json.dumps({"ts": "2026-09-25T10:01:00", "input_tokens": 60_000,
                      "reprefill_tokens": 2_000,
                      "compaction": {"mechanisms": []}}) + "\n")
    db = usage_store.replay_usage_extract(extract, db_path=tmp_path / "r.db")
    conn = sqlite3.connect(str(db))
    stored = conn.execute(
        "SELECT compaction IS NULL FROM usage ORDER BY id").fetchall()
    assert [r[0] for r in stored] == [1, 0], (
        "the null row did not replay as SQL NULL, so it will be read as a record "
        "that names nothing rather than as no record at all")
    buckets = [usage_store.attribution_bucket(r[0]) for r in
               conn.execute("SELECT compaction FROM usage ORDER BY id")]
    assert buckets == [usage_store.NO_COMPACTION_RECORD,
                       usage_store.NO_FREEING_EVIDENCE], buckets


def test_a_dropped_microcompacted_renames_the_bucket_with_its_tokens(tmp_path,
                                                                    monkeypatch):
    """Clause 5's rename half, pinned with no vault read anywhere.

    `turn_start:microcompact` is the bucket this item prices the trade at (236 rows,
    68,444,409 tokens, 60.6%), and its name comes from `turn_start.microcompacted` —
    not from `mechanisms`, which says `"microcompact"` either way. The second extract
    built for this item dropped that key, and the bucket changed its name with its
    `n`, its Σ and its share all unchanged, so the window-total check passed while the
    published mechanism was wrong. The vault-reading node catches that only where the
    vault is mounted, so this node catches it on synthetic bytes.

    It crosses the writer→reader seam rather than asserting on a hand-built dict: the
    row is built by `app.compaction_record`, the module the live harness uses to
    produce the stored `compaction` JSON, and read by `attribution_bucket` in a later
    process. A key renamed or dropped on the producer side shows up here and nowhere
    else, because every other node in this file constructs its rows by hand.

    The rows are stamped with `_ts_recent`, not a literal date, and the node asserts
    `rows == 2` before it reads any bucket. Both halves are #2053: this node used to
    stamp `2026-09-25T10:00:00` and `2026-09-25T10:01:00` and then ask for a
    RELATIVE window (`hours=168`), so its own left edge `now - 168 h` passed
    `2026-09-25T10:00:00` at `2026-10-02T10:00Z` and both rows fell out. `rows` was
    0, every bucket printed `n=0`, `gap_pct` stayed 0.0, and the rename check failed
    as `assert 0 == 1` while the attribution code behind it was doing nothing wrong —
    the same shape as a dropped provenance key, arriving on a schedule nobody wrote
    down. An empty table is a fixture that stopped being in the window; it must never
    be able to imitate a mis-named bucket again.
    """
    from app import compaction_record as CR

    comp = {"tokens_before": 215_000, "tokens_after": 211_000, "microcompacted": 3,
            "threshold": 210_144, "context_window": 262_144,
            "summarize_outcome": "under_threshold"}
    turn = CR.TurnCompaction(session_id="sess-a")
    turn.note_turn_start(comp)
    record = turn.to_record()
    assert record["turn_start"]["tokens_freed"] == 4_000, (
        "the producer stopped reporting a freed count, so this node is no longer "
        "describing a front edit that freed something")
    assert usage_store.attribution_bucket(record) == "turn_start:microcompact"

    # The same turn as an extract that never carried the key would arrive.
    stripped = json.loads(json.dumps(record))
    del stripped["turn_start"]["microcompacted"]
    assert stripped["turn_start"]["mechanisms"] == \
        record["turn_start"]["mechanisms"], (
        "`mechanisms` names the layer that acted in both rows; if it differs, the "
        "assertion below is catching a different fault than the rename")
    assert usage_store.attribution_bucket(stripped) == "turn_start:other", (
        "the row kept the microcompact name after its provenance key went, so a "
        "writer-side rename would silently re-title the headline bucket")

    # And in the printed table: one row each, identical Σ, which is exactly why a
    # total check cannot tell a named bucket from an anonymous one.
    extract = tmp_path / "rename.jsonl"
    extract.write_text(
        json.dumps({"ts": _ts_recent(2.0), "input_tokens": 150_000,
                    "reprefill_tokens": 400_000, "compaction": record}) + "\n"
        + json.dumps({"ts": _ts_recent(1.0), "input_tokens": 150_000,
                      "reprefill_tokens": 400_000, "compaction": stripped}) + "\n")
    db = usage_store.replay_usage_extract(extract, db_path=tmp_path / "rename.db")
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    attr = usage_store.reprefill_attribution(hours=168)
    assert attr["rows"] == 2, (
        f"the window priced {attr['rows']} row(s), not the 2 the node just wrote "
        f"(since={attr['since']}), so a bucket assertion below would be reading an "
        "empty table — a fixture outside the window it asks about, not a rename")
    assert _row(attr, "turn_start:microcompact")["n"] == 1
    assert _row(attr, "turn_start:other")["n"] == 1
    assert _row(attr, "turn_start:microcompact")["reprefill_tokens"] == \
        _row(attr, "turn_start:other")["reprefill_tokens"] == 400_000
    assert attr["gap_pct"] == 0.0, (
        "the Σ stopped being exhaustive over the rename, which is the check a "
        "mis-named bucket would actually have to trip instead")


def test_the_extract_replays_to_its_own_published_split(tmp_path, monkeypatch):
    """Every figure the #2027 verdict publishes comes out of the committed extract
    alone — including the bucket NAMES, not just their totals.

    Two extracts in a row were built wrong on this item and both survived a
    Σ check. The first dropped `turn_start.microcompacted` (keeping only
    `mechanisms`), which re-named the window's largest bucket from
    `turn_start:microcompact` to `turn_start:other` with its 68,444,409 tokens and
    60.6% share unchanged. The second encoded nulls as `'null'` (the node above).
    So this asserts the named triples, and it asserts the bucket that carries the
    headline by name: a figure quoted in `architecture/context-window.md` has to be
    reproducible from bytes in the vault, and a reader who cannot re-derive the
    mechanism cannot price the trade.

    Skipped, never failed, when the vault is not on this machine: the assertion is
    about a file in `~/obsidian`, and a checkout without it has no evidence either
    way. What the skip does NOT take with it is the failure this extract taught us:
    the null-as-NULL half and the bucket-rename half are both pinned vault-free, by
    `test_a_null_compaction_in_the_extract_stays_sql_null` and
    `test_a_dropped_microcompacted_renames_the_bucket_with_its_tokens`, on synthetic
    bytes through the same `replay_usage_extract` route. Only the published TOTALS
    need the file — a vault-less machine therefore still catches a mis-named headline
    bucket and a mis-encoded null, and loses only the 68,444,409 figure itself.
    """
    extract = Path.home() / "obsidian" / usage_store.REPREFILL_WITNESS
    if not extract.exists():
        pytest.skip(f"{extract} is not on this machine, so the published TOTALS "
                    "cannot be re-derived here; the two silent-failure halves "
                    "(null-as-NULL, dropped-microcompacted rename) are pinned "
                    "vault-free in the two nodes named above")
    db = usage_store.replay_usage_extract(extract, db_path=tmp_path / "w.db")
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    attr = usage_store.reprefill_attribution(
        since=usage_store.REPREFILL_WITNESS_SINCE,
        until=usage_store.REPREFILL_WITNESS_UNTIL)
    assert attr["window_absolute"] is True
    assert attr["rows"] == 3_868, attr["rows"]
    assert attr["window_total_reprefill_tokens"] == 112_908_568
    assert attr["gap_pct"] == 0.0, attr["gap_pct"]
    assert attr["ratio_b_over_d_mean"] == 51.8, attr["ratio_b_over_d_mean"]
    published = {b["bucket"]: (b["n"], b["reprefill_tokens"])
                 for b in attr["buckets"] if b["n"]}
    assert published == {
        "turn_start:microcompact": (236, 68_444_409),
        "relief:tool_results": (95, 20_903_710),
        "relief:reasoning": (30, 5_865_692),
        "ran_noop": (2_049, 10_181_397),
        "no_freeing_evidence": (12, 317_625),
        "no_compaction_record": (1_446, 7_195_735),
    }, published
