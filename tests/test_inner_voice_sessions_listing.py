"""`/api/inner_voice/sessions` must not parse the whole sessions directory.

A session JSON carries its whole transcript and workers opt into Inner Voice,
so the listing that parsed every file took 7.9 s over 10,503 files on
2026-10-12 — on the event loop, twice per tab switch under React StrictMode.
The walk now goes newest-mtime first and stops once the answer is settled;
these pin that the stop is real and that it does not change the answer.
"""
import json
import os

import pytest

from app.routers import inner_voice


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(inner_voice, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(inner_voice, "_SESSION_ROW_CACHE", {})
    return tmp_path


def _write(sessions_dir, sid, *, epoch, inner_voice=True, mtime=None, **extra):
    from datetime import datetime, timezone
    stamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    path = sessions_dir / f"{sid}.json"
    path.write_text(json.dumps(
        {"session_id": sid, "inner_voice": inner_voice, "messages": [],
         "last_active": stamp, **extra}))
    mtime = epoch if mtime is None else mtime
    os.utime(path, (mtime, mtime))
    return path


def _count_reads(monkeypatch):
    reads = []
    real = json.loads

    def counting(text, *a, **kw):
        reads.append(1)
        return real(text, *a, **kw)

    monkeypatch.setattr(inner_voice.json, "loads", counting)
    return reads


def test_walk_stops_once_the_limit_is_settled(sessions_dir, monkeypatch):
    base = 1_800_000_000
    for i in range(200):
        _write(sessions_dir, f"s{i:03d}", epoch=base + i * 600, inner_voice=i % 2 == 0)
    reads = _count_reads(monkeypatch)

    rows = inner_voice._scan_inner_voice_sessions(5)

    assert [r["session_id"] for r in rows] == ["s198", "s196", "s194", "s192", "s190"]
    assert len(reads) < 20, f"parsed {len(reads)} of 200 files for 5 rows"


def test_a_late_touched_old_session_does_not_displace_a_newer_one(sessions_dir):
    """A titler or capture write moves mtime, never `last_active`: the old
    session heads the walk and must still sort where its conversation ended."""
    base = 1_800_000_000
    for i in range(30):
        _write(sessions_dir, f"s{i:02d}", epoch=base + i * 600)
    _write(sessions_dir, "s03", epoch=base + 3 * 600, mtime=base + 99 * 600)

    rows = inner_voice._scan_inner_voice_sessions(3)

    assert [r["session_id"] for r in rows] == ["s29", "s28", "s27"]


def test_a_rewritten_file_is_re_read(sessions_dir, monkeypatch):
    base = 1_800_000_000
    _write(sessions_dir, "a", epoch=base, title="before")
    assert inner_voice._scan_inner_voice_sessions(5)[0]["title"] == "before"

    reads = _count_reads(monkeypatch)
    assert inner_voice._scan_inner_voice_sessions(5)[0]["title"] == "before"
    assert reads == [], "an unchanged file was parsed again"

    _write(sessions_dir, "a", epoch=base + 60, title="after, and longer")
    assert inner_voice._scan_inner_voice_sessions(5)[0]["title"] == "after, and longer"
