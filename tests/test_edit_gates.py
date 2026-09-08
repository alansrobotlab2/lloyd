"""Read-before-edit and stale-file gates on Write/Edit.

The failure these exist for is not "the model edited a file it hadn't read"
— that one announces itself as `old_string not found`. It is the edit that
*succeeds*: the model Reads a file, something else rewrites it, the model's
old_string still matches, and the edit silently reverts the other writer.
Nothing in the transcript, the tool result or the logs says so.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agent_mcp import builtin_fs as FS

SID = "20260908_120000_test"


@pytest.fixture(autouse=True)
def _clean():
    FS.reset_read_records()
    yield
    FS.reset_read_records()


@pytest.fixture
def bound(monkeypatch):
    """Bind a session id, as main.call_tool does for a real tool call."""
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    return SID


@pytest.fixture
def f(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("alpha\nbeta\n")
    return p


def _text(res):
    return res.content[0].text


def _err(res):
    return json.loads(_text(res)).get("error", "")


def _touch_later(p: Path, content: str):
    """Rewrite `p` with a stat that is unambiguously different.

    mtime_ns alone can collide inside one filesystem tick, and the content
    here is the same length in some cases, so the test would pass or fail on
    timer granularity.
    """
    p.write_text(content)
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


# ── the gate ────────────────────────────────────────────────────────────────

async def test_edit_without_a_read_is_refused(bound, f):
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error
    assert "has not been Read in this session" in _err(res)
    assert "Read it first" in _err(res)
    assert f.read_text() == "alpha\nbeta\n", "the file must not have been touched"


async def test_read_then_edit_succeeds(bound, f):
    await FS.call_tool("Read", {"file_path": str(f)})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error is False
    assert f.read_text() == "ALPHA\nbeta\n"


async def test_a_partial_read_is_enough(bound, f):
    await FS.call_tool("Read", {"file_path": str(f), "offset": 2, "limit": 1})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "beta",
                                      "new_string": "BETA"})
    assert res.is_error is False


async def test_an_out_of_band_write_makes_the_edit_stale(bound, f):
    """This is the silent-revert case: old_string still matches."""
    await FS.call_tool("Read", {"file_path": str(f)})
    _touch_later(f, "alpha\nbeta\ngamma\n")     # e.g. `sed -i` from Bash
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error
    assert "changed on disk since you last Read it" in _err(res)
    assert f.read_text() == "alpha\nbeta\ngamma\n", "the other writer survived"


async def test_re_reading_clears_the_staleness(bound, f):
    await FS.call_tool("Read", {"file_path": str(f)})
    _touch_later(f, "alpha\nbeta\ngamma\n")
    assert (await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                        "new_string": "ALPHA"})).is_error
    await FS.call_tool("Read", {"file_path": str(f)})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error is False


async def test_consecutive_edits_need_only_one_read(bound, f):
    await FS.call_tool("Read", {"file_path": str(f)})
    assert not (await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                            "new_string": "ALPHA"})).is_error
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "beta",
                                      "new_string": "BETA"})
    assert res.is_error is False
    assert f.read_text() == "ALPHA\nBETA\n"


async def test_write_then_edit_needs_no_read(bound, tmp_path):
    p = tmp_path / "new.py"
    assert not (await FS.call_tool("Write", {"file_path": str(p),
                                             "content": "one\n"})).is_error
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "one",
                                      "new_string": "two"})
    assert res.is_error is False


# ── Write ───────────────────────────────────────────────────────────────────

async def test_write_creates_a_new_file_with_no_read(bound, tmp_path):
    p = tmp_path / "sub" / "new.txt"
    res = await FS.call_tool("Write", {"file_path": str(p), "content": "hi"})
    assert res.is_error is False
    assert p.read_text() == "hi"


async def test_write_over_an_unread_existing_file_is_refused(bound, f):
    res = await FS.call_tool("Write", {"file_path": str(f), "content": "clobbered"})
    assert res.is_error
    assert "already exists and has not been Read" in _err(res)
    assert f.read_text() == "alpha\nbeta\n"


async def test_write_over_a_stale_file_is_refused(bound, f):
    await FS.call_tool("Read", {"file_path": str(f)})
    _touch_later(f, "somebody else\n")
    res = await FS.call_tool("Write", {"file_path": str(f), "content": "clobbered"})
    assert res.is_error
    assert "changed on disk since you last Read it" in _err(res)


async def test_write_through_a_dangling_symlink_is_a_create(bound, tmp_path):
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    assert not target.exists()
    res = await FS.call_tool("Write", {"file_path": str(link), "content": "made"})
    assert res.is_error is False
    assert target.read_text() == "made"


# ── keys, sessions, kill switch ─────────────────────────────────────────────

async def test_a_symlink_and_its_target_share_one_record(bound, tmp_path):
    target = tmp_path / "real.py"
    target.write_text("x = 1\n")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    await FS.call_tool("Read", {"file_path": str(link)})
    res = await FS.call_tool("Edit", {"file_path": str(target), "old_string": "x = 1",
                                      "new_string": "x = 2"})
    assert res.is_error is False, _text(res)


async def test_two_sessions_do_not_share_reads(monkeypatch, f):
    monkeypatch.setattr(FS, "get_bound_session", lambda: "session-a")
    await FS.call_tool("Read", {"file_path": str(f)})
    monkeypatch.setattr(FS, "get_bound_session", lambda: "session-b")
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error and "has not been Read" in _err(res)


async def test_the_second_writer_of_two_sessions_is_told(monkeypatch, f):
    for sid in ("session-a", "session-b"):
        monkeypatch.setattr(FS, "get_bound_session", lambda s=sid: s)
        await FS.call_tool("Read", {"file_path": str(f)})
    monkeypatch.setattr(FS, "get_bound_session", lambda: "session-a")
    assert not (await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                            "new_string": "ALPHA"})).is_error
    monkeypatch.setattr(FS, "get_bound_session", lambda: "session-b")
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "beta",
                                      "new_string": "BETA"})
    assert res.is_error and "changed on disk" in _err(res)


async def test_no_bound_session_skips_the_gate(monkeypatch, f):
    """Unit tests and legacy callers dispatch with no aggregator context."""
    monkeypatch.setattr(FS, "get_bound_session", lambda: "")
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error is False


async def test_kill_switch_keeps_recording_but_stops_refusing(bound, f, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness", {**CONFIG.get("harness", {}),
                                            "edit_gates": {"enabled": False}})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error is False
    assert FS._seen(SID, os.path.realpath(f)) is not None, \
        "records must keep accruing so flipping the switch back works at once"


async def test_records_are_bounded_per_session_and_overall(bound, tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_READ_PATHS_MAX", 3)
    monkeypatch.setattr(FS, "_READ_SESSIONS_MAX", 2)
    paths = []
    for i in range(5):
        p = tmp_path / f"f{i}.txt"
        p.write_text(str(i))
        paths.append(p)
        await FS.call_tool("Read", {"file_path": str(p)})
    assert len(FS._read_records[SID]) == 3
    assert FS._seen(SID, os.path.realpath(paths[0])) is None
    assert FS._seen(SID, os.path.realpath(paths[4])) is not None

    for sid in ("s1", "s2", "s3"):
        monkeypatch.setattr(FS, "get_bound_session", lambda s=sid: s)
        await FS.call_tool("Read", {"file_path": str(paths[0])})
    assert len(FS._read_records) == 2


# ── errors that used to escape ──────────────────────────────────────────────

async def test_editing_a_binary_file_is_a_normal_error(bound, tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\xff\xfe\x00\x01not utf8")
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "a",
                                      "new_string": "b"})
    assert res.is_error
    assert "not valid UTF-8" in _err(res)
    assert str(p) in _err(res)


async def test_a_missing_file_records_nothing(bound, tmp_path):
    res = await FS.call_tool("Read", {"file_path": str(tmp_path / "nope.txt")})
    assert res.is_error
    assert FS._read_records.get(SID) in (None, {})


async def test_reading_a_directory_records_nothing(bound, tmp_path):
    res = await FS.call_tool("Read", {"file_path": str(tmp_path)})
    assert res.is_error
    assert FS._read_records.get(SID) in (None, {})


# ── the contract is in the schema, not only the error ───────────────────────

async def test_the_tool_descriptions_state_the_contract():
    tools = {t.name: t for t in await FS.list_tools()}
    assert "Read the file in this session first" in tools["Edit"].description
    assert "OVERWRITING an existing one requires" in tools["Write"].description
