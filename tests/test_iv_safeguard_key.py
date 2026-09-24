"""#770 — `inner_voice_observations` carries the deterministic rule's name.

Guard identity lived only in the prose of `reason`, half of it as a bracketed
suffix the grader's prefix regex never matched, so a verdict per safeguard was
a regex over sentences. The column is forward-only: an existing database is
migrated in place and its old rows keep a NULL key.
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import usage_store  # noqa: E402

# The table as it stood before #770: 17 columns, no key.
_OLD_SCHEMA = """CREATE TABLE inner_voice_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
    sequence_in_turn INTEGER NOT NULL, trigger TEXT NOT NULL,
    action TEXT NOT NULL, reason TEXT, content TEXT, related_tool TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER,
    cache_create INTEGER, latency_ms INTEGER, model TEXT, error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""


def _cols(path):
    with sqlite3.connect(path) as c:
        return [r[1] for r in c.execute("PRAGMA table_info(inner_voice_observations)")]


def test_a_17_column_db_is_migrated_in_place_with_its_rows(tmp_path, monkeypatch):
    db = tmp_path / "usage.db"
    with sqlite3.connect(db) as c:
        c.execute(_OLD_SCHEMA)
        c.execute("INSERT INTO inner_voice_observations (session_id, turn_id, "
                  "sequence_in_turn, trigger, action, reason) VALUES "
                  "('s', 't', 1, 'pretool', 'inject', 'deterministic: 3 near-identical Read calls')")
    assert len(_cols(db)) == 17 and "safeguard" not in _cols(db)

    monkeypatch.setattr(usage_store, "DB_PATH", db)
    rows = usage_store.list_inner_voice_observations(session_id="s")

    assert "safeguard" in _cols(db)
    assert len(rows) == 1
    assert rows[0]["reason"].startswith("deterministic:")
    assert rows[0]["safeguard"] is None, "forward-only: an old row is not backfilled"


def test_the_writer_persists_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_store, "DB_PATH", tmp_path / "usage.db")
    usage_store.record_inner_voice_observation(
        "s", "t", 1, "pretool", "inject", reason="deterministic: …",
        safeguard="repetition")
    usage_store.record_inner_voice_observation("s", "t", 2, "assistant_message", "noop")
    got = {r["sequence_in_turn"]: r["safeguard"]
           for r in usage_store.list_inner_voice_observations(session_id="s")}
    assert got == {1: "repetition", 2: None}
