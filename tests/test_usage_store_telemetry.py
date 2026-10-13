"""The P11 telemetry columns on `usage.db` and their three breakdowns.

Additive by migration, like every column before them: a live store keeps its
rows and gains NULLs, and NULL means unmeasured in every query here.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app import sessions_io, usage_store


def _old_schema_db(path):
    """A usage table as it stood before P11 (the #1078 shape), with one row."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
            session_id TEXT, model TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_create INTEGER NOT NULL DEFAULT 0,
            cache_read INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL, duration_ms INTEGER, duration_api_ms INTEGER,
            num_turns INTEGER, reprefill_tokens INTEGER, prefix_misses INTEGER,
            skills TEXT, compaction TEXT);
        INSERT INTO usage (session_id, model, input_tokens) VALUES ('old', 'primary', 123);
    """)
    conn.commit()
    conn.close()


def test_migration_keeps_rows_and_adds_null_columns(tmp_path, monkeypatch):
    db = tmp_path / "usage.db"
    _old_schema_db(db)
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    conn = usage_store._conn()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(usage)")}
    for col, _ in usage_store._TELEMETRY_COLUMNS:
        assert col in cols
    row = dict(conn.execute("SELECT * FROM usage").fetchone())
    assert row["input_tokens"] == 123
    assert all(row[c] is None for c, _ in usage_store._TELEMETRY_COLUMNS)
    # Unmeasured rows are out of every breakdown, not counted as zeros. The
    # bench/production split (#2663) rides in the same rows, so a window with no
    # measured turn is still the empty list, not a row of zeros.
    assert usage_store.stop_reason_breakdown(hours=24) == []
    assert usage_store.tool_error_breakdown(hours=24)["turns_measured"] == 0
    assert usage_store.ttft_summary(hours=24)["first_p50_ms"] is None


def _record(**tel):
    usage_store.record_usage(session_id="s", model="primary",
                             input_tokens=10, output_tokens=5, **tel)


def test_breakdowns():
    _record(stop_reason="stop", tool_calls=3, tool_errors=1,
            tool_errors_by_class={"denied": 1}, tool_ms_total=120,
            ttft_ms_first=200, ttft_ms_max=900, reasoning_tokens=2, wrapped_up=0)
    _record(stop_reason="stop", tool_calls=1, tool_errors=0,
            tool_errors_by_class={}, tool_ms_total=10,
            ttft_ms_first=400, ttft_ms_max=400, wrapped_up=0)
    _record(stop_reason="max_turns", tool_calls=2, tool_errors=2,
            tool_errors_by_class={"transport": 1, "denied": 1},
            ttft_ms_first=1000, ttft_ms_max=1000, wrapped_up=1)
    _record()  # an unmeasured row: in no breakdown

    # Every row here is a production turn: `s` is not a background-shaped id, so
    # the bench halves are 0 and the totals are untouched by the split (#2663).
    assert usage_store.stop_reason_breakdown(hours=24) == [
        {"stop_reason": "stop", "turns": 2, "wrapped_up": 0,
         "bench_turns": 0, "production_turns": 2,
         "bench_wrapped_up": 0, "production_wrapped_up": 0},
        {"stop_reason": "max_turns", "turns": 1, "wrapped_up": 1,
         "bench_turns": 0, "production_turns": 1,
         "bench_wrapped_up": 0, "production_wrapped_up": 1},
    ]
    tools = usage_store.tool_error_breakdown(hours=24)
    assert tools["turns_measured"] == 3
    assert (tools["tool_calls"], tools["tool_errors"]) == (6, 3)
    assert tools["tool_ms_total"] == 130
    assert tools["by_class"] == {"denied": 2, "transport": 1}
    assert list(tools["by_class"]) == ["denied", "transport"]
    ttft = usage_store.ttft_summary(hours=24)
    assert ttft == {"turns_measured": 3, "first_p50_ms": 400,
                    "first_p90_ms": 1000, "max_ms": 1000}
    reasoning = usage_store.reasoning_tokens_summary(hours=24)
    assert reasoning == {"turns_measured": 1, "reasoning_tokens": 2,
                         "output_tokens": 5}


#: Two real bench-family ids from `usage.db`, quoted from #2663. `bench_runner_sdk`
#: mints its recorded id from `RECORDED_SESSION_SLUG = "bench"`; the bench-mining
#: worker hands `new_background_session_id` the source `bench-mine`, whose mint
#: rule keeps only alphanumerics, so the hyphen is gone before the id exists.
_BENCH_ID = "20260924_100311_bench_67d5"
_BENCH_MINE_ID = "20260922_183424_benchmine_6d65"
_CHAT_ID = "20260924_100311_a1b2c3"


def _record_turn(session_id, **tel):
    usage_store.record_usage(session_id=session_id, model="primary",
                             input_tokens=10, output_tokens=5, **tel)


def test_the_bench_predicate_reads_the_session_id_slug_exactly():
    """#2663 clause 2: ONE named predicate, compared as a slug, not a substring.

    `benchmine` is the shape a predicate keyed on the id component `bench` loses
    — 362 rows / 19 `max_turns` in `usage.db` since 2026-09-25 — and `benchmarked`
    is the shape a `LIKE '%bench%'` predicate wrongly gains. The set is derived
    from `sessions_io.BENCH_SOURCES` by the mint rule, so the two spellings of
    one producer cannot drift apart.
    """
    assert sessions_io.BENCH_SESSION_SLUGS == frozenset({"bench", "benchmine"})
    assert sessions_io.is_bench_session_id(_BENCH_ID) is True
    assert sessions_io.is_bench_session_id(_BENCH_MINE_ID) is True
    assert sessions_io.is_bench_session_id(_CHAT_ID) is False
    assert sessions_io.is_bench_session_id("20260924_100311_owedcheck_1f2e") is False
    assert sessions_io.is_bench_session_id("20260924_100311_benchmarked_1f2e") is False
    assert sessions_io.is_bench_session_id("bench_002_recall") is False
    assert sessions_io.is_bench_session_id("worker:arch-review:abc") is False
    assert sessions_io.is_bench_session_id(None) is False


def test_stop_reason_breakdown_separates_bench_trials_from_production_turns():
    """#2663 clauses 1 and 2: the production count is readable from the return
    value, with no caller-side SQL.

    Bench trials and production worker turns share the `worker` platform and the
    `usage` table has no source column, so before this split the published
    number could not say which population it described: `max_turns` reported 142
    turns over 168 h on 2026-10-13, and 140 of them were bench trials.
    """
    _record_turn(_BENCH_ID, stop_reason="max_turns", wrapped_up=1)
    _record_turn(_BENCH_MINE_ID, stop_reason="max_turns", wrapped_up=0)
    _record_turn(_CHAT_ID, stop_reason="max_turns", wrapped_up=1)
    _record_turn(_CHAT_ID, stop_reason="stop", wrapped_up=0)

    assert usage_store.stop_reason_breakdown(hours=24) == [
        {"stop_reason": "max_turns", "turns": 3, "wrapped_up": 2,
         "bench_turns": 2, "production_turns": 1,
         "bench_wrapped_up": 1, "production_wrapped_up": 1},
        {"stop_reason": "stop", "turns": 1, "wrapped_up": 0,
         "bench_turns": 0, "production_turns": 1,
         "bench_wrapped_up": 0, "production_wrapped_up": 0},
    ]


def test_wrapped_up_is_tallied_per_kind_not_inferred_from_the_stop_reason():
    """#2663 clause 4, and the error the new instrument must not rebuild.

    Over 168 h ending 2026-10-13 the bench-mine rows carry 19 `max_turns` with 0
    `wrapped_up`, while production carries 11 `max_turns` / 11 `wrapped_up`, so
    "every `max_turns` is wrapped up" is a production fact and not a bench fact.
    A row that derived the wrap marker from the reason would report
    `bench_wrapped_up=1` for the single row below.
    """
    _record_turn(_BENCH_MINE_ID, stop_reason="max_turns", wrapped_up=0)
    assert usage_store.stop_reason_breakdown(hours=24) == [
        {"stop_reason": "max_turns", "turns": 1, "wrapped_up": 0,
         "bench_turns": 1, "production_turns": 0,
         "bench_wrapped_up": 0, "production_wrapped_up": 0},
    ]
    _record_turn(_CHAT_ID, stop_reason="max_turns", wrapped_up=1)
    assert usage_store.stop_reason_breakdown(hours=24) == [
        {"stop_reason": "max_turns", "turns": 2, "wrapped_up": 1,
         "bench_turns": 1, "production_turns": 1,
         "bench_wrapped_up": 0, "production_wrapped_up": 1},
    ]


#: The query `stop_reason_breakdown` ran before #2663, verbatim. The three keys
#: it returns have to keep answering exactly this after the split.
_PRE_SPLIT_SQL = """SELECT stop_reason,
                          COUNT(*)                     AS turns,
                          COALESCE(SUM(wrapped_up), 0) AS wrapped_up
                     FROM usage
                    WHERE ts >= ? AND stop_reason IS NOT NULL
                 GROUP BY stop_reason
                 ORDER BY turns DESC, stop_reason"""


def test_the_split_rides_along_and_leaves_the_pre_split_totals_readable():
    """#2663 clause 3: the split adds keys, it never replaces the total.

    #1554-era history and every trend comparison across this change is read off
    `turns` / `wrapped_up`, so those two keys have to hold the same numbers the
    unsplit query produced over the same rows — including for the bench rows the
    split now names separately.
    """
    _record_turn(_BENCH_ID, stop_reason="max_turns", wrapped_up=1)
    _record_turn(_BENCH_MINE_ID, stop_reason="max_turns", wrapped_up=0)
    _record_turn(_CHAT_ID, stop_reason="max_turns", wrapped_up=1)
    _record_turn(_CHAT_ID, stop_reason="stop", wrapped_up=0)
    _record_turn(_BENCH_ID, stop_reason="stop", wrapped_up=0)
    _record_turn("s")          # unmeasured: in no breakdown

    since = usage_store._since(hours=24)
    pre_split = [dict(r) for r in usage_store._conn().execute(_PRE_SPLIT_SQL,
                                                             (since,))]
    rows = usage_store.stop_reason_breakdown(hours=24)
    assert [{k: r[k] for k in ("stop_reason", "turns", "wrapped_up")}
            for r in rows] == pre_split
    for r in rows:
        assert r["turns"] == r["bench_turns"] + r["production_turns"]
        assert r["wrapped_up"] == r["bench_wrapped_up"] + r["production_wrapped_up"]


def test_by_class_is_stored_as_json():
    _record(tool_errors_by_class={"mcp_error": 2})
    raw = usage_store._conn().execute(
        "SELECT tool_errors_by_class FROM usage").fetchone()[0]
    assert json.loads(raw) == {"mcp_error": 2}


def test_an_unknown_telemetry_column_is_a_caller_error():
    with pytest.raises(TypeError):
        _record(ttft_ms=5)


def test_a_file_replaced_under_the_store_is_written_to_the_new_one(tmp_path,
                                                                   monkeypatch):
    """A connection is cached per thread and reopened when `DB_PATH` MOVES; the
    same path holding a DIFFERENT file is the other half of that transition, and
    SQLite never complains about it — it writes into the unlinked inode, and
    `record_usage` returns exactly as though the row had landed.

    pytest 9 produces this between two param cases that share a `tmp_path`
    basename (it strips brackets, so `""` and `"[]"` collide), which is how four
    of #1866's junk-compaction nodes lost a row while the store reported
    success. The reader here is read-only on purpose: `sqlite3.connect` on a
    missing path creates one, which would hide the whole defect.
    """
    db = tmp_path / "usage.db"
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    usage_store.record_usage(session_id="first", model="m", input_tokens=1)
    assert db.exists(), "the first write should have created the file"
    db.unlink()                                    # same path, now no file
    usage_store.record_usage(session_id="second", model="m", input_tokens=2)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sessions = [r[0] for r in conn.execute("SELECT session_id FROM usage")]
    assert sessions == ["second"], (
        "a write after the file was replaced must land in the file that sits at "
        f"the path now, not in the inode the cached handle held: {sessions}")
