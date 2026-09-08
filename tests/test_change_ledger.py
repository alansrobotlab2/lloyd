"""Per-turn change ledger: what a turn wrote, and putting it back.

`~/lloyd` is production — a saved file is a deploy. The self-modification
loop has a worktree, a gate and an automatic rollback; an ordinary chat turn
that edits three files had none of that, no line saying which three, and no
undo.

The assertion that matters most is
`test_revert_refuses_a_file_something_else_wrote_since`. Restoring a
pre-image over somebody else's later write is the exact damage this feature
exists to prevent, so a revert that cannot prove it is safe must refuse by
name rather than skip quietly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agent_mcp import (
    _change_ledger as CL,
    _subagent_registry as SR,
    _task_registry as TR,
    builtin_fs as FS,
)

SID = "20260908_150000_led"
TURN = "turn-aaaa"


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(CL, "CHANGES_ROOT", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    CL.reset()
    FS.reset_read_records()
    SR.reset()
    yield
    CL.reset()
    FS.reset_read_records()
    SR.reset()


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    tok = TR.current_turn_id.set(TURN)
    ctok = TR.current_call_id.set("call_1")
    yield
    TR.current_turn_id.reset(tok)
    TR.current_call_id.reset(ctok)


def _files():
    return CL.list_changes(SID, TURN)


async def _edit(p, old, new):
    await FS.call_tool("Read", {"file_path": str(p)})
    return await FS.call_tool("Edit", {"file_path": str(p), "old_string": old,
                                       "new_string": new})


# ── recording ───────────────────────────────────────────────────────────────

async def test_an_edit_is_recorded_with_its_pre_image(bound, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    await _edit(f, "one", "two")

    files = _files()
    assert len(files) == 1
    e = files[0]
    assert e["op"] == "edit" and e["path"] == str(f)
    assert e["real"] == os.path.realpath(f)
    assert e["call_id"] == "call_1" and e["snapshot"] == "ok"
    assert CL._pre_path(SID, TURN, e["real"]).read_bytes() == b"one\n"


async def test_a_create_is_recorded_with_no_snapshot(bound, tmp_path):
    p = tmp_path / "new.txt"
    await FS.call_tool("Write", {"file_path": str(p), "content": "hi"})
    e = _files()[0]
    assert e["op"] == "create" and e["snapshot"] == "none"


async def test_overwriting_an_existing_file_is_a_write_with_a_snapshot(bound, tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("before\n")
    await FS.call_tool("Read", {"file_path": str(p)})
    await FS.call_tool("Write", {"file_path": str(p), "content": "after\n"})
    e = _files()[0]
    assert e["op"] == "write" and e["snapshot"] == "ok"
    assert CL._pre_path(SID, TURN, e["real"]).read_bytes() == b"before\n"


async def test_first_writer_wins_across_repeated_edits(bound, tmp_path):
    """A turn that edits one file ten times reverts to where it came in."""
    f = tmp_path / "a.txt"
    f.write_text("v0\n")
    await _edit(f, "v0", "v1")
    await FS.call_tool("Edit", {"file_path": str(f), "old_string": "v1",
                                "new_string": "v2"})
    files = _files()
    assert len(files) == 1 and files[0]["writes"] == 2
    assert CL._pre_path(SID, TURN, files[0]["real"]).read_bytes() == b"v0\n"


async def test_the_rule_survives_an_aggregator_restart_mid_turn(bound, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("v0\n")
    await _edit(f, "v0", "v1")
    CL.reset()                      # the aggregator restarted
    await FS.call_tool("Read", {"file_path": str(f)})
    await FS.call_tool("Edit", {"file_path": str(f), "old_string": "v1",
                                "new_string": "v2"})
    files = _files()
    assert len(files) == 1, "the on-disk index was not consulted"
    assert CL._pre_path(SID, TURN, files[0]["real"]).read_bytes() == b"v0\n"


async def test_a_symlink_and_its_target_are_one_entry(bound, tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("x\n")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    await _edit(link, "x", "y")
    await FS.call_tool("Read", {"file_path": str(target)})
    await FS.call_tool("Edit", {"file_path": str(target), "old_string": "y",
                                "new_string": "z"})
    assert len(_files()) == 1


async def test_a_too_large_file_records_no_snapshot(bound, tmp_path, monkeypatch):
    monkeypatch.setattr(CL, "SNAPSHOT_MAX_BYTES", 10)
    f = tmp_path / "big.txt"
    f.write_text("a long line of text\n")
    await _edit(f, "long", "short")
    assert _files()[0]["snapshot"] == "too_large"


async def test_no_turn_id_means_no_ledger(monkeypatch, tmp_path):
    """Workers and direct run_query callers have no turn and no reader."""
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    await _edit(f, "one", "two")
    assert _files() == []
    assert not (CL.CHANGES_ROOT / f"{SID}.changes").exists()


async def test_the_kill_switch_stops_recording(bound, tmp_path, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "change_ledger": {"enabled": False}})
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    await _edit(f, "one", "two")
    assert _files() == []


async def test_a_ledger_failure_never_fails_the_edit(bound, tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(CL, "begin", boom)
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    res = await _edit(f, "one", "two")
    assert res.is_error is False
    assert f.read_text() == "two\n"


# ── attribution ─────────────────────────────────────────────────────────────

def test_a_subagent_change_is_attributed_to_the_parent_turn():
    rec = SR.register(subagent_type="general-purpose", description="d", prompt="p",
                      parent_session_id="parent-sid", parent_turn_id="parent-turn",
                      session_id="task:gp:abcd1234", model="primary", max_turns=5)
    SR.finish(rec, status="completed")
    assert CL.scope("task:gp:abcd1234", "") == ("parent-sid", "parent-turn")


def test_an_orphan_subagent_session_gets_no_ledger():
    assert CL.scope("task:gp:deadbeef", "") is None


def test_an_ordinary_session_is_its_own_scope():
    assert CL.scope("sid", "turn") == ("sid", "turn")
    assert CL.scope("sid", "") is None
    assert CL.scope("", "turn") is None


async def test_a_subagent_edit_lands_on_the_parents_ledger(tmp_path, monkeypatch):
    rec = SR.register(subagent_type="general-purpose", description="d", prompt="p",
                      parent_session_id=SID, parent_turn_id=TURN,
                      session_id="task:gp:abcd1234", model="primary", max_turns=5)
    monkeypatch.setattr(FS, "get_bound_session", lambda: "task:gp:abcd1234")
    tok = TR.current_turn_id.set("")
    try:
        f = tmp_path / "a.txt"
        f.write_text("one\n")
        await _edit(f, "one", "two")
    finally:
        TR.current_turn_id.reset(tok)
        SR.finish(rec, status="completed")
    files = _files()
    assert len(files) == 1
    assert files[0]["via_session"] == "task:gp:abcd1234"


# ── revert ──────────────────────────────────────────────────────────────────

async def test_revert_restores_an_edit(bound, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    await _edit(f, "one", "two")
    results = CL.revert(SID, TURN)
    assert [r["status"] for r in results] == ["restored"]
    assert f.read_text() == "one\n"
    assert _files()[0]["reverted_at"] is not None


async def test_revert_deletes_a_created_file(bound, tmp_path):
    p = tmp_path / "new.txt"
    await FS.call_tool("Write", {"file_path": str(p), "content": "hi"})
    assert [r["status"] for r in CL.revert(SID, TURN)] == ["deleted"]
    assert not p.exists()


async def test_revert_handles_several_files(bound, tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("A0\n")
    b.write_text("B0\n")
    await _edit(a, "A0", "A1")
    await _edit(b, "B0", "B1")
    results = CL.revert(SID, TURN)
    assert sorted(r["status"] for r in results) == ["restored", "restored"]
    assert a.read_text() == "A0\n" and b.read_text() == "B0\n"


async def test_revert_can_be_scoped_to_named_paths(bound, tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("A0\n")
    b.write_text("B0\n")
    await _edit(a, "A0", "A1")
    await _edit(b, "B0", "B1")
    results = CL.revert(SID, TURN, [str(a)])
    assert len(results) == 1 and results[0]["status"] == "restored"
    assert a.read_text() == "A0\n" and b.read_text() == "B1\n"


async def test_revert_refuses_a_file_something_else_wrote_since(bound, tmp_path):
    """Restoring here would destroy the other write, which is the damage."""
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    await _edit(f, "one", "two")
    f.write_text("somebody else\n")
    results = CL.revert(SID, TURN)
    assert results[0]["status"] == "refused"
    assert "changed since" in results[0]["reason"]
    assert f.read_text() == "somebody else\n"


async def test_revert_refuses_a_created_file_that_changed_since(bound, tmp_path):
    p = tmp_path / "new.txt"
    await FS.call_tool("Write", {"file_path": str(p), "content": "hi"})
    p.write_text("edited by someone else")
    results = CL.revert(SID, TURN)
    assert results[0]["status"] == "refused"
    assert p.exists()


async def test_revert_refuses_when_there_is_no_snapshot(bound, tmp_path, monkeypatch):
    monkeypatch.setattr(CL, "SNAPSHOT_MAX_BYTES", 5)
    f = tmp_path / "a.txt"
    f.write_text("a long line\n")
    await _edit(f, "long", "short")
    results = CL.revert(SID, TURN)
    assert results[0]["status"] == "refused" and "no snapshot" in results[0]["reason"]


async def test_revert_refuses_a_file_that_no_longer_exists(bound, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    await _edit(f, "one", "two")
    f.unlink()
    assert CL.revert(SID, TURN)[0]["status"] == "refused"


async def test_reverting_twice_is_a_skip_not_a_second_restore(bound, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one\n")
    await _edit(f, "one", "two")
    CL.revert(SID, TURN)
    f.write_text("later work\n")
    results = CL.revert(SID, TURN)
    assert results[0]["status"] == "skipped"
    assert f.read_text() == "later work\n"


async def test_a_created_file_already_gone_is_a_skip(bound, tmp_path):
    p = tmp_path / "new.txt"
    await FS.call_tool("Write", {"file_path": str(p), "content": "hi"})
    p.unlink()
    assert CL.revert(SID, TURN)[0]["status"] == "skipped"


# ── retention ───────────────────────────────────────────────────────────────

def test_prune_drops_turn_dirs_past_the_retention_window(monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "change_ledger": {"retention_days": 1, "max_total_mb": 999}})
    old = CL.turn_dir("s1", "t-old")
    new = CL.turn_dir("s1", "t-new")
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "index.json").write_text("{}")
    stale = time.time() - 2 * 86400
    os.utime(old, (stale, stale))
    CL.prune()
    assert not old.exists() and new.exists()


def test_prune_drops_the_oldest_until_under_the_size_cap(monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "change_ledger": {"retention_days": 99,
                                           "max_total_mb": 0.00002}})   # 20 bytes
    now = time.time()
    for i, name in enumerate(("t1", "t2", "t3")):
        d = CL.turn_dir("s1", name)
        d.mkdir(parents=True)
        (d / "a.pre").write_bytes(b"x" * 20)
        # Recent, and ordered: epoch-1970 mtimes would be swept by the
        # retention pass before the size cap ever ran.
        os.utime(d, (now - 30 + i, now - 30 + i))
    CL.prune()
    left = sorted(p.name for p in (CL.CHANGES_ROOT / "s1.changes").iterdir())
    assert left == ["t3"], left


def test_prune_only_ever_touches_changes_directories(monkeypatch):
    """The sessions dir also holds transcripts and spilled tool results."""
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "change_ledger": {"retention_days": 0, "max_total_mb": 0}})
    transcript = CL.CHANGES_ROOT / "s1.json"
    transcript.write_text("{}")
    spill = CL.CHANGES_ROOT / "s1.tool-results"
    spill.mkdir()
    (spill / "x.txt").write_text("kept")
    doomed = CL.turn_dir("s1", "t1")
    doomed.mkdir(parents=True)
    (doomed / "a.pre").write_bytes(b"x")
    CL.prune()
    assert transcript.exists() and (spill / "x.txt").read_text() == "kept"
    assert not doomed.exists()


# ── the wire ────────────────────────────────────────────────────────────────

def test_the_meta_keys_match_on_both_sides():
    from agent_mcp import main as M
    from app.harness import mcp_pool as P
    assert M.META_TURN_ID == P.META_TURN_ID == "lloyd/turn_id"
    assert M.META_CALL_ID == P.META_CALL_ID == "lloyd/call_id"


def test_the_pool_only_sends_the_keys_when_set():
    import inspect
    from app.harness import mcp_pool as P
    src = inspect.getsource(P.MCPPool.call_tool)
    assert "if turn_id:" in src and "if call_id:" in src


def test_the_loop_passes_the_turn_and_call_id():
    from pathlib import Path
    from app.harness import loop as L
    src = Path(L.__file__).read_text()
    assert '"turn_id": getattr(options, "turn_id", "") or ""' in src
    assert '"call_id": tc.get("id", "") or ""' in src


def test_state_reports_the_ledger():
    import inspect
    from agent_mcp import main as M
    assert '"changes": _change_ledger.stats()' in inspect.getsource(M.state)


async def test_the_changes_route_needs_both_params():
    from agent_mcp import main as M

    class _Req:
        def __init__(self, qp):
            self.query_params = qp

    assert (await M.changes(_Req({}))).status_code == 400
    assert (await M.changes(_Req({"session": "s"}))).status_code == 400
    ok = await M.changes(_Req({"session": SID, "turn": TURN}))
    assert ok.status_code == 200
    assert json.loads(ok.body)["files"] == []
