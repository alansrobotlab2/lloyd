"""`state.read_events` decodes the ledger once per change, not once per call.

Until 2026-11-22 it read and decoded the whole file on every call whatever
`limit` said — 0.44 s on the 21.5 MB live ledger — and `/api/automod/status`
called it on the backend's event loop.
"""

from __future__ import annotations

import asyncio
import json
import threading

from app.routers import automod as R
from scripts.automod import state as S


def _write(path, n, start=0):
    with path.open("a", encoding="utf-8") as fh:
        for i in range(start, start + n):
            fh.write(json.dumps({"event": "x", "i": i, "nested": {"k": i}}) + "\n")


def test_repeated_reads_of_an_unchanged_ledger_decode_it_once(tmp_path):
    S.ledger_rows_reset()
    path = tmp_path / "promotions.jsonl"
    _write(path, 50)
    for _ in range(5):
        assert [e["i"] for e in S.read_events(path, limit=3)] == [47, 48, 49]
    assert S.ledger_read_count(path) == 1


def test_an_append_is_seen_by_the_next_read(tmp_path):
    S.ledger_rows_reset()
    path = tmp_path / "promotions.jsonl"
    _write(path, 3)
    assert len(S.read_events(path, limit=100)) == 3
    _write(path, 2, start=3)
    assert [e["i"] for e in S.read_events(path, limit=100)] == [0, 1, 2, 3, 4]
    assert S.ledger_read_count(path) == 2


def test_a_caller_that_writes_into_a_row_does_not_change_the_next_read(tmp_path):
    S.ledger_rows_reset()
    path = tmp_path / "promotions.jsonl"
    _write(path, 2)
    first = S.read_events(path, limit=2)
    first[0]["event"] = "changed"
    first[0]["added"] = True
    again = S.read_events(path, limit=2)
    assert again[0]["event"] == "x" and "added" not in again[0]


def test_a_missing_ledger_and_a_torn_line_read_as_before(tmp_path):
    S.ledger_rows_reset()
    assert S.read_events(tmp_path / "absent.jsonl") == []
    path = tmp_path / "promotions.jsonl"
    path.write_text('{"event": "a"}\n\nnot json\n{"event": "b"}\n{"event": "c', encoding="utf-8")
    assert [e["event"] for e in S.read_events(path)] == ["a", "b"]


def test_the_status_route_reads_the_ledger_off_the_event_loop(monkeypatch):
    seen = {}
    real = S.read_events

    def spy(*args, **kwargs):
        seen["thread"] = threading.get_ident()
        return real(*args, **kwargs)

    monkeypatch.setattr(S, "read_events", spy)

    async def run():
        seen["loop_thread"] = threading.get_ident()
        await R.get_status(limit=5)

    asyncio.run(run())
    assert seen["thread"] != seen["loop_thread"]
