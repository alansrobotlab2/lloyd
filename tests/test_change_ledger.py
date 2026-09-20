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

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent_mcp import (
    _change_ledger as CL,
    _subagent_registry as SR,
    _task_registry as TR,
    builtin_fs as FS,
)

_ROOT = Path(__file__).resolve().parent.parent
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


# ── the vault_write route (#962) ─────────────────────────────────────────────
#
# `vault_write` is the whole-file route every nightly, reflection and digest job
# is told to use, and it was the one such route that recorded nothing: an
# overwrite that truncated a note left no pre-image to restore from and no index
# row saying it happened. Two clobbers are on the record — `lloyd/MEMORY.md` on
# 2026-09-10 (45 lines recovered from one) and a 192-line note on 2026-09-12,
# self-reported in the run record of the run that did it.
#
# The tests below cross the boundary those jobs actually cross: the aggregator's
# own `main.call_tool`, which binds the caller's session from the request `_meta`,
# hands the handler to a worker thread (`asyncio.to_thread` in `vault.py`, which
# copies the contextvars), and serialises the result dict through `_wrap`. Nothing
# here sets a contextvar by hand, so the recorded scope can only have come from
# that route. A recording that worked only on a direct in-process call to
# `_vault_write` — same thread, ambient variables already in place — would not
# have undone a single real clobber.

PRE_CONTENT = "# Note as it came in\n\ncarriage return in it\r\nunicode ✓\n"
POST_CONTENT = "# Note after the overwrite\n"
# What a *second process* commits to the note while this turn's write is queued on
# the lock. Used only by the cross-process test below.
CHILD_CONTENT = "# Committed by another process while this turn waited\n"
NOTE_PATH = "notes/a-note.md"


@pytest.fixture
def vault_route(tmp_path, monkeypatch):
    """A scratch vault, and `vault_write` pointed at it. No session bound.

    Every root is patched on the module that *reads* it. `app.paths.VAULT_ROOT`
    and `atomic_io.SCRATCH_DIR` matter as much as `vault.VAULT` does: without them
    a target in the scratch tree takes the sibling-temp branch instead of the
    off-tree one production takes, and the locks land in the checkout running the
    suite.
    """
    from types import SimpleNamespace

    from agent_mcp import vault as VT
    from app import atomic_io
    from app import paths as app_paths

    vault = tmp_path / "vault"
    note = vault / NOTE_PATH
    note.parent.mkdir(parents=True)
    note.write_bytes(PRE_CONTENT.encode("utf-8"))
    audit = vault / "memory" / "audit"

    monkeypatch.setattr(VT, "VAULT", vault)
    monkeypatch.setattr(VT, "AUDIT_LOG_DIR", audit)
    monkeypatch.setattr(VT, "AUDIT_LOG_FILE", audit / "writes.jsonl")
    monkeypatch.setattr(app_paths, "VAULT_ROOT", vault)
    monkeypatch.setattr(atomic_io, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(atomic_io, "SCRATCH_DIR", tmp_path / "locks" / "tmp")
    yield SimpleNamespace(mod=VT, vault=vault, note=note,
                          audit=audit / "writes.jsonl",
                          locks=tmp_path / "locks")


async def _vault_write(vr, path: str, content: str, *,
                       session: str = SID, turn: str = TURN,
                       call_id: str = "call-1") -> dict:
    """One `vault_write` as a client sends it: args plus the `_meta` envelope."""
    from agent_mcp import main as M

    res = await M.call_tool("vault_write", {"path": path, "content": content}, {
        M.META_SESSION_ID: session, M.META_TURN_ID: turn, M.META_CALL_ID: call_id,
    })
    text = res.content[0].text
    assert not res.is_error, text
    return json.loads(text)


def _audit_rows(vr) -> list[dict]:
    return [json.loads(line) for line in
            vr.audit.read_text(encoding="utf-8").splitlines() if line.strip()]


async def test_a_vault_write_overwrite_is_recorded_with_its_pre_image(vault_route):
    """One entry, both hashes, `snapshot: ok` — what a `Write` already gave you."""
    out = await _vault_write(vault_route, NOTE_PATH, POST_CONTENT)
    assert out["success"] and out["path"] == NOTE_PATH

    files = _files()
    assert len(files) == 1, f"expected exactly one ledger entry, got {files}"
    e = files[0]
    assert e["op"] == "write"
    assert e["path"] == NOTE_PATH
    assert e["real"] == os.path.realpath(vault_route.note)
    assert e["call_id"] == "call-1"     # the aggregator's id, verbatim, unprefixed
    assert e["snapshot"] == "ok", e
    assert e["pre_sha256"] == CL.sha256_bytes(PRE_CONTENT.encode("utf-8"))
    assert e["post_sha256"] == CL.sha256_bytes(POST_CONTENT.encode("utf-8"))
    assert CL._pre_path(SID, TURN, e["real"]).read_bytes() == PRE_CONTENT.encode("utf-8")
    assert vault_route.note.read_bytes() == POST_CONTENT.encode("utf-8")


async def test_reverting_a_vault_write_restores_the_note_byte_for_byte(vault_route):
    """The undo the two recorded clobbers did not have."""
    await _vault_write(vault_route, NOTE_PATH, POST_CONTENT)

    results = CL.revert(SID, TURN, [NOTE_PATH])
    assert [r["status"] for r in results] == ["restored"], results
    assert vault_route.note.read_bytes() == PRE_CONTENT.encode("utf-8")


async def test_a_vault_write_that_creates_a_note_deletes_on_revert(vault_route):
    """A create is `op="create"`, snapshotted as nothing, and undo removes the file."""
    fresh = "notes/fresh.md"
    await _vault_write(vault_route, fresh, POST_CONTENT)

    e = _files()[0]
    assert e["op"] == "create" and e["snapshot"] == "none", e
    assert (vault_route.vault / fresh).read_bytes() == POST_CONTENT.encode("utf-8")

    results = CL.revert(SID, TURN, [fresh])
    assert [r["status"] for r in results] == ["deleted"], results
    assert not (vault_route.vault / fresh).exists()


async def test_two_writes_to_one_note_revert_to_where_it_came_in(vault_route):
    """A turn that overwrites twice still undoes to the bytes it came in with.

    The 2026-09-10 truncation was followed two hours later by an append, so a
    revert that landed the intermediate version would have restored the 1-line
    remnant and called it recovered.
    """
    await _vault_write(vault_route, NOTE_PATH, "v1\n")
    await _vault_write(vault_route, NOTE_PATH, "v2\n")

    files = _files()
    assert len(files) == 1 and files[0]["writes"] == 2, files
    assert files[0]["post_sha256"] == CL.sha256_bytes(b"v2\n")
    CL.revert(SID, TURN, [NOTE_PATH])
    assert vault_route.note.read_bytes() == PRE_CONTENT.encode("utf-8")


async def test_an_unattributable_vault_write_still_writes_the_note(vault_route):
    """A call the ledger cannot key: today's behaviour, and the write still lands.

    No turn id is the shape that occurs — a worker turn (`run_in_background`) or a
    subagent turn shares its session's ambient turn id, which is empty there, so
    `begin` could never be found again by `revert`. Nothing raises and nothing is
    recorded, yet the row keeps both content hashes and the session it does know: a
    call whose *session* id is missing never gets this far at all, since the
    aggregator refuses a state-changing call that arrives with no session (#1053).
    """
    out = await _vault_write(vault_route, NOTE_PATH, POST_CONTENT, turn="")
    assert out["success"], out
    assert vault_route.note.read_bytes() == POST_CONTENT.encode("utf-8")
    assert _files() == []

    row = _audit_rows(vault_route)[-1]
    assert row["session"] == SID and row["turn"] == ""
    assert row["post_sha256"] == CL.sha256_bytes(POST_CONTENT.encode("utf-8"))
    assert row["pre_sha256"] == CL.sha256_bytes(PRE_CONTENT.encode("utf-8"))


async def test_a_failed_snapshot_never_fails_the_vault_write(vault_route, monkeypatch):
    """Undo bookkeeping that blows up mid-record costs a pre-image, not the write.

    The rule `builtin_fs._ledger_begin` already documents: an edit that fails
    because the undo bookkeeping failed is strictly worse than an edit with no
    undo.
    """
    def boom(*a, **k):
        raise RuntimeError("snapshot target went away")

    monkeypatch.setattr(CL, "snapshot_pre", boom)
    out = await _vault_write(vault_route, NOTE_PATH, POST_CONTENT)
    assert out["success"], out
    assert vault_route.note.read_bytes() == POST_CONTENT.encode("utf-8")
    # Nothing durable claims a restorable pre-image, so nothing may claim to restore.
    # The review of round SM_20260920_084122 read the previous line
    # (`all(r["status"] != "restored" for r in CL.revert(...))`) as vacuous on the
    # grounds that `revert` returns an empty list with no index on disk. It does not:
    # `_load` is an in-memory LRU that `begin` already populated, so the call returns
    # one result. What the review caught is real but one level softer — the old line
    # passed on both `[]` and `[refused]`, so it could not tell "one refusal" from
    # "nothing to revert". Pinned to the exact refusal, with the reason it must name,
    # and the observable half of the degrade: the note keeps the new bytes, because
    # the pre-image that would have undone them was never snapshotted.
    assert not CL._index_path(SID, TURN).exists(), \
        "a failed snapshot still wrote an index claiming a restorable pre-image"
    backed = CL.revert(SID, TURN, [NOTE_PATH])
    assert len(backed) == 1, f"expected one refusal naming the missing snapshot: {backed}"
    assert backed[0]["status"] == "refused", backed
    assert backed[0]["reason"] == "no snapshot (none)", backed
    assert vault_route.note.read_bytes() == POST_CONTENT.encode("utf-8")


async def test_a_write_that_cannot_take_the_lock_records_nothing(vault_route, monkeypatch):
    """The one attributable call that returns without writing records nothing either.

    `_vault_write` resolves the scope *before* taking `commit_lock`, so the lock
    timeout is the branch where an attributable call reaches neither `_ledger_record`
    nor `_audit_write`. That is the right order and this pins it: had the entry been
    opened before the lock, a turn whose write was refused would carry an index entry
    whose revert reports a restore that never happened.
    """
    from app import atomic_io as AI

    ready, release = threading.Event(), threading.Event()

    def _hold():
        with AI.commit_lock(vault_route.note):
            ready.set()
            release.wait(20)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert ready.wait(5), "the other writer never took the lock"
    try:
        monkeypatch.setattr(AI, "DEFAULT_LOCK_WAIT", 0.3)   # read at call time
        from agent_mcp import main as M
        res = await M.call_tool("vault_write",
                               {"path": NOTE_PATH, "content": POST_CONTENT},
                               {M.META_SESSION_ID: SID, M.META_TURN_ID: TURN,
                                M.META_CALL_ID: "call-timeout"})
        text = res.content[0].text
        assert res.is_error, f"a write blocked by another writer reported success: {text}"
        assert "could not lock" in text, text
    finally:
        release.set()
        holder.join(5)

    assert vault_route.note.read_bytes() == PRE_CONTENT.encode("utf-8")
    assert _files() == []
    assert not vault_route.audit.exists(), "a refused write left an audit row"


# ── the two seams the first round left untested ──────────────────────────────
#
# Both were named by the review of round SM_20260920_084122. They are process
# boundaries, so the assertions below are built to go red on their own: the child
# writes and the parent reads a different process's bytes, and a subagent call
# arrives over the aggregator with nothing in the parent's context.

_CHILD_PREIMAGE_WRITER = '''"""Another process, holding the vault writer's lock and changing the note under it.

Roots arrive by argument and are echoed back before any work, so the parent can
assert the child used the tree it was handed rather than a module default.
"""
import sys
from pathlib import Path

ROOT, VAULT, LOCKS, TARGET, NEW_BYTES = sys.argv[1:6]
sys.path.insert(0, ROOT)

from app import atomic_io                      # noqa: E402
from app import paths as _paths                # noqa: E402

_paths.VAULT_ROOT = Path(VAULT)
atomic_io.LOCK_DIR = Path(LOCKS)
atomic_io.SCRATCH_DIR = Path(LOCKS) / "tmp"

target = Path(TARGET)
print("LOCK", str(atomic_io.lock_file_for(target)), str(atomic_io.LOCK_DIR), flush=True)
with atomic_io.commit_lock(target, timeout=20):
    print("HELD", flush=True)
    sys.stdin.readline()          # the parent is queued on this lock
    atomic_io.write_text_durable(target, NEW_BYTES)
    print("WROTE", flush=True)
    sys.stdin.readline()          # release
print("RELEASED", flush=True)
'''


async def test_the_pre_image_is_what_a_write_from_another_process_left(
        vault_route, tmp_path):
    """The lock the pre-image read sits inside is a real cross-process lock.

    Round SM_20260920_084122 pinned this only with a holder in another *thread* of
    the pytest process. `flock` excludes processes, and the writers that matter
    here are separate processes — a nightly job, the other MCP module, a script —
    so a same-process holder could not say whether the lock is shared at all.

    The ordering is what makes this an assertion rather than a description, so it is
    spelled out in the awaits. A child takes `commit_lock` on the note; the parent's
    `vault_write` is then started and given the event loop (`await asyncio.sleep`,
    not `time.sleep` — a blocking sleep in the loop thread would start nothing and
    the write would land after the child had already committed, which is a test that
    passes for the wrong reason and was the first draft of this test). Only once the
    parent is demonstrably queued on the lock does the child replace the note's
    bytes. So:

    * if the parent read the file *before* taking the lock, the pre-image would be
      the note's ORIGINAL bytes — the ones that were there when the read happened,
      not the ones this write displaced — and reverting it would silently erase the
      child's change;
    * reading inside the lock makes the pre-image the child's bytes, which is what
      the write actually displaced, and the revert below puts exactly those back.

    The child also prints the lock path it resolved: equal to the parent's, which is
    the claim that this is one lock and not two that happen to agree today.
    """
    from app import atomic_io as AI

    script = tmp_path / "child_preimage_writer.py"
    script.write_text(_CHILD_PREIMAGE_WRITER, encoding="utf-8")
    env = {k: os.environ[k] for k in ("PATH", "USER", "LANG", "LC_ALL") if k in os.environ}
    env["PYTHONPATH"] = str(_ROOT)
    child = subprocess.Popen(
        [sys.executable, str(script), str(_ROOT), str(vault_route.vault),
         str(vault_route.locks), str(vault_route.note), CHILD_CONTENT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        cwd=str(_ROOT), env=env)
    try:
        lock_line = child.stdout.readline().split()
        assert lock_line[:1] == ["LOCK"], f"child died early: {lock_line}"
        assert lock_line[1] == str(AI.lock_file_for(vault_route.note)), (
            f"child locks {lock_line[1]}, the parent locks "
            f"{AI.lock_file_for(vault_route.note)}: two locks is the same lost "
            "update with extra steps")
        assert child.stdout.readline().strip() == "HELD", "child never took the lock"

        # The parent queues behind it. `await`, not `time.sleep`: the handler runs
        # in a worker thread, so the loop has to be yielded for the call to reach
        # commit_lock at all — with a blocking sleep here the coroutine never starts,
        # the child commits first, and both the lock-mutation below and any lost
        # pre-image read go unnoticed.
        task = asyncio.create_task(_vault_write(vault_route, NOTE_PATH, POST_CONTENT))
        await asyncio.sleep(0.4)
        assert not task.done(), (
            "the write did not wait for the other process's lock — with the lock "
            "shared, this call cannot have got past commit_lock while the child "
            "still holds it")
        assert vault_route.note.read_bytes() == PRE_CONTENT.encode("utf-8"), \
            "the parent wrote before the child released the lock"

        child.stdin.write("go\n")
        child.stdin.flush()
        assert child.stdout.readline().strip() == "WROTE", "child never wrote"
        assert vault_route.note.read_bytes() == CHILD_CONTENT.encode("utf-8")

        child.stdin.write("release\n")
        child.stdin.flush()
        out = await asyncio.wait_for(task, timeout=20)
        assert child.wait(10) == 0
        assert child.stdout.readline().strip() == "RELEASED"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(5)

    assert out["success"], out
    files = _files()
    assert len(files) == 1, files
    assert files[0]["pre_sha256"] == CL.sha256_bytes(CHILD_CONTENT.encode("utf-8")), (
        "the recorded pre-image is the content that was on disk before the other "
        "process wrote, not the content this write displaced — the read is not "
        "inside the lock")
    assert files[0]["post_sha256"] == CL.sha256_bytes(POST_CONTENT.encode("utf-8"))

    backed = CL.revert(SID, TURN, [NOTE_PATH])
    assert [r["status"] for r in backed] == ["restored"], backed
    assert vault_route.note.read_bytes() == CHILD_CONTENT.encode("utf-8"), (
        "revert restored the wrong pre-image: it would have discarded a change "
        "that another process committed while this turn was queued")


async def test_a_subagent_vault_write_lands_on_the_parents_ledger_over_the_pool(
        vault_route):
    """A Task child's note reaches the parent turn's index through the aggregator.

    The first round pinned this by calling `_change_ledger.scope` with a
    `task:*` id — a unit assertion on a pure function, which is not the boundary.
    A subagent re-enters `main.call_tool` over loopback `/mcp` from a fresh ASGI
    task, so the parent's turn context does NOT travel: the only thing crossing is
    `_meta`, whose session id is the child's own `task:*` and whose turn key is
    absent, because `loop.py` stamps a turn id only when one is set.

    So the test drives that call — child session id, no turn key — with the child
    registered against a parent (session, turn). Everything the parent's contextvar
    would have supplied is missing here, and the entry still has to arrive on the
    PARENT's index: a human reverts the turn they were watching, and a per-child
    index would be read by nobody. `via_session` is what keeps the child's authorship
    visible inside that entry rather than lost to the redirect.
    """
    from agent_mcp import main as M

    child_sid = "task:general-purpose:abcd1234"
    rec = SR.register(subagent_type="general-purpose", description="d", prompt="p",
                      parent_session_id=SID, parent_turn_id=TURN,
                      session_id=child_sid, model="primary", max_turns=5)
    res = await M.call_tool(
        "vault_write", {"path": NOTE_PATH, "content": POST_CONTENT},
        {M.META_SESSION_ID: child_sid, M.META_CALL_ID: "call-child"})
    assert not res.is_error, res.content[0].text
    SR.finish(rec, status="completed")

    # The parent's index, not the child's: no turn id of its own, and the redirect
    # is the whole point. Asserted from the index path, not from `_files()`, so a
    # stray entry written under the child's own key cannot satisfy it.
    assert not CL._index_path(child_sid, "").exists(), (
        "the child got its own ledger, which nothing reads after the Task returns")
    files = _files()
    assert len(files) == 1, files
    assert files[0]["via_session"] == child_sid, files[0]
    assert files[0]["pre_sha256"] == CL.sha256_bytes(PRE_CONTENT.encode("utf-8"))
    assert files[0]["post_sha256"] == CL.sha256_bytes(POST_CONTENT.encode("utf-8"))
    assert files[0]["call_id"] == "call-child", files[0]

    row = _audit_rows(vault_route)[-1]
    assert (row["session"], row["turn"]) == (SID, TURN), row

    backed = CL.revert(SID, TURN, [NOTE_PATH])
    assert [r["status"] for r in backed] == ["restored"], backed
    assert vault_route.note.read_bytes() == PRE_CONTENT.encode("utf-8")


async def test_the_audit_row_names_the_turn_and_both_hashes(vault_route):
    """The join key: the audit row's hashes are the ledger entry's hashes.

    Before this the row carried `timestamp`/`agent_id`/`path`/`bytes`/`action`
    only, so the log that saw the overwrite named neither the turn that wrote it
    nor whether anything was restorable — given "a note changed at 09:04:12Z"
    there was no way to reach the pre-image that did or did not exist.
    """
    await _vault_write(vault_route, NOTE_PATH, POST_CONTENT)

    row = _audit_rows(vault_route)[-1]
    for key in ("timestamp", "agent_id", "path", "bytes", "action",
                "session", "turn", "pre_sha256", "post_sha256"):
        assert key in row, f"audit row missing {key}: {row}"
    assert (row["session"], row["turn"]) == (SID, TURN)
    assert row["path"] == NOTE_PATH and row["action"] == "write"
    assert row["bytes"] == len(POST_CONTENT.encode("utf-8"))

    e = _files()[0]
    assert row["pre_sha256"] == e["pre_sha256"] == CL.sha256_bytes(PRE_CONTENT.encode("utf-8"))
    assert row["post_sha256"] == e["post_sha256"] == CL.sha256_bytes(POST_CONTENT.encode("utf-8"))
