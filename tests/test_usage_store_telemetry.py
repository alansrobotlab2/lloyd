"""The P11 telemetry columns on `usage.db` and their three breakdowns.

Additive by migration, like every column before them: a live store keeps its
rows and gains NULLs, and NULL means unmeasured in every query here.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app import usage_store


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
    # Unmeasured rows are out of every breakdown, not counted as zeros.
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

    assert usage_store.stop_reason_breakdown(hours=24) == [
        {"stop_reason": "stop", "turns": 2, "wrapped_up": 0},
        {"stop_reason": "max_turns", "turns": 1, "wrapped_up": 1},
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
