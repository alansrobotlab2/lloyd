"""Out-of-band tsc diagnostics: coverage, delta, cold start, delivery.

tsc is whole-project (`web/tsconfig.json` includes only `src`) and ~5 s on
this tree, so it cannot ride back on the Edit. It runs debounced in the
background and the answer arrives through the drain the background-Bash tool
already uses. The interesting failures are all about *attribution*: whose
errors are these, and which of them are new.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_mcp import (
    _subagent_registry as SR,
    _task_registry as TR,
    _tsc_runner as T,
    builtin_bash as BB,
    builtin_fs as FS,
)

SID = "20260908_140000_tsc"


@pytest.fixture(autouse=True)
def _clean():
    T.reset()
    TR._pending_by_session.clear()
    SR.reset()
    FS.reset_read_records()
    yield
    T.reset()
    TR._pending_by_session.clear()
    SR.reset()
    FS.reset_read_records()


@pytest.fixture
def tree(tmp_path):
    """A tree shaped like the repo: web/tsconfig.json, web/src, a fake tsc."""
    root = tmp_path / "proj"
    (root / "web" / "src").mkdir(parents=True)
    (root / "web" / "tsconfig.json").write_text('{"include": ["src"]}')
    binp = root / "web" / "node_modules" / ".bin"
    binp.mkdir(parents=True)
    tsc = binp / "tsc"
    tsc.write_text("#!/bin/sh\ncat \"$TSC_FAKE_OUT\"\n")
    tsc.chmod(0o755)
    return root


def _fake_output(tree: Path, text: str, monkeypatch):
    out = tree / "web" / "out.txt"
    out.write_text(text)
    monkeypatch.setenv("TSC_FAKE_OUT", str(out))


ONE_ERROR = ("src/A.tsx(10,3): error TS2339: Property 'x' does not exist "
             "on type 'Y'.\n")
TWO_ERRORS = ONE_ERROR + ("src/B.tsx(4,1): error TS2551: Property 'z' does "
                          "not exist on type 'W'.\n")


# ── coverage ────────────────────────────────────────────────────────────────

def test_project_root_walks_up_to_the_tree_that_owns_the_file(tree):
    f = tree / "web" / "src" / "A.tsx"
    f.write_text("x")
    assert T.project_root(str(f)) == tree


def test_a_worktree_answers_about_itself(tmp_path):
    """A autoimplement round edits under ~/lloyd-work/...; the live tree's tsc
    would say nothing about it."""
    for name in ("live", "work/SM_1/home/lloyd"):
        root = tmp_path / name
        (root / "web" / "src").mkdir(parents=True)
        (root / "web" / "tsconfig.json").write_text("{}")
    wt = tmp_path / "work/SM_1/home/lloyd"
    f = wt / "web" / "src" / "A.tsx"
    f.write_text("x")
    assert T.project_root(str(f)) == wt


def test_files_outside_web_src_are_not_covered(tree):
    for rel in ("web/vite.config.ts", "server.py", "web/src/styles.css"):
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        assert T.project_root(str(p)) is None, rel


def test_a_tree_with_no_tsconfig_is_not_covered(tmp_path):
    f = tmp_path / "web" / "src" / "A.tsx"
    f.parent.mkdir(parents=True)
    f.write_text("x")
    assert T.project_root(str(f)) is None


async def test_no_tsc_binary_means_no_hint_and_no_run(tree, monkeypatch):
    """Promising a check that cannot happen is worse than saying nothing."""
    (tree / "web" / "node_modules" / ".bin" / "tsc").unlink()
    f = tree / "web" / "src" / "A.tsx"
    f.write_text("x")
    assert await T.note_edit(SID, str(f)) == ""
    assert T._pending == {}


async def test_note_edit_returns_the_hint_and_arms_one_timer(tree, monkeypatch):
    monkeypatch.setattr(T, "DEBOUNCE_S", 30.0)
    f = tree / "web" / "src" / "A.tsx"
    f.write_text("x")
    hint = await T.note_edit(SID, str(f))
    assert "tsc check queued" in hint
    assert T._pending[str(tree)][SID] == {"src/A.tsx"}
    first = T._timers[str(tree)]

    g = tree / "web" / "src" / "B.tsx"
    g.write_text("x")
    await T.note_edit(SID, str(g))
    await asyncio.sleep(0)                    # let the cancellation land
    assert first.cancelled() or first.done(), "the debounce must re-arm, not stack"
    assert T._pending[str(tree)][SID] == {"src/A.tsx", "src/B.tsx"}
    T._timers[str(tree)].cancel()


# ── delta and cold start ────────────────────────────────────────────────────

async def test_the_first_run_only_seeds(tree, monkeypatch):
    """Otherwise every pre-existing error lands on whoever edited first."""
    _fake_output(tree, TWO_ERRORS, monkeypatch)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    summary = await T.run_once(tree)
    assert summary["seeded"] is True
    assert TR._pending_by_session == {}


async def test_a_new_error_in_an_edited_file_is_reported(tree, monkeypatch):
    _fake_output(tree, ONE_ERROR, monkeypatch)
    await T.run_once(tree, seed_only=True)

    _fake_output(tree, TWO_ERRORS, monkeypatch)
    T._pending[str(tree)] = {SID: {"src/B.tsx"}}
    await T.run_once(tree)

    recs = TR._pending_by_session[SID]
    assert len(recs) == 1
    assert recs[0].files == ["src/B.tsx"]
    assert recs[0].lines == ["src/B.tsx(4,1): error TS2551: Property 'z' does "
                             "not exist on type 'W'."]


async def test_an_error_in_a_file_this_session_did_not_touch_is_not_reported(
        tree, monkeypatch):
    """Somebody else's breakage is somebody else's news."""
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)
    _fake_output(tree, TWO_ERRORS, monkeypatch)
    T._pending[str(tree)] = {SID: {"src/C.tsx"}}
    await T.run_once(tree)
    assert TR._pending_by_session == {}


async def test_a_pre_existing_error_is_not_reported(tree, monkeypatch):
    _fake_output(tree, ONE_ERROR, monkeypatch)
    await T.run_once(tree, seed_only=True)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)
    assert TR._pending_by_session == {}


async def test_an_error_that_only_moved_is_not_new(tree, monkeypatch):
    _fake_output(tree, ONE_ERROR, monkeypatch)
    await T.run_once(tree, seed_only=True)
    moved = ONE_ERROR.replace("(10,3)", "(90,3)")
    _fake_output(tree, moved, monkeypatch)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)
    assert TR._pending_by_session == {}


async def test_two_copies_of_one_error_is_a_new_error(tree, monkeypatch):
    _fake_output(tree, ONE_ERROR, monkeypatch)
    await T.run_once(tree, seed_only=True)
    _fake_output(tree, ONE_ERROR + ONE_ERROR.replace("(10,3)", "(40,3)"), monkeypatch)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)
    assert len(TR._pending_by_session[SID][0].lines) == 1


async def test_a_clean_run_emits_nothing(tree, monkeypatch):
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)
    assert TR._pending_by_session == {}


async def test_the_baseline_persists_across_a_restart(tree, monkeypatch):
    _fake_output(tree, ONE_ERROR, monkeypatch)
    await T.run_once(tree, seed_only=True)
    assert (tree / "_pipeline" / "tsc" / "baseline.json").exists()

    T._baseline.clear()                       # simulate a process restart
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)
    assert TR._pending_by_session == {}, "the on-disk baseline was not honoured"


async def test_a_failed_run_still_tells_the_session(tree, monkeypatch):
    """A model told a check was queued and never told the outcome waits."""
    tsc = tree / "web" / "node_modules" / ".bin" / "tsc"
    tsc.write_text("#!/bin/sh\nexit 9\n")
    tsc.chmod(0o755)
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)

    T._baseline[str(tree)] = {}
    monkeypatch.setattr(T, "_spawn_tsc", _failing_spawn)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)
    rec = TR._pending_by_session[SID][0]
    assert rec.status == "failed"
    assert "check failed" in TR.format_diagnostics_notification(rec)


async def _failing_spawn(root):
    return "failed", "tsc blew up", 0.2


# ── delivery ────────────────────────────────────────────────────────────────

async def test_a_subagents_result_is_attributed_to_its_parent(tree, monkeypatch):
    """Nothing reads a subagent's drain queue once its Task has returned."""
    rec = SR.register(subagent_type="general-purpose", description="d",
                      prompt="p", parent_session_id="parent-sid",
                      parent_turn_id="turn-1", session_id="task:gp:abcd1234",
                      model="primary", max_turns=5)
    SR.finish(rec, status="completed")

    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)
    _fake_output(tree, ONE_ERROR, monkeypatch)
    T._pending[str(tree)] = {"task:gp:abcd1234": {"src/A.tsx"}}
    await T.run_once(tree)
    assert "parent-sid" in TR._pending_by_session
    assert "task:gp:abcd1234" not in TR._pending_by_session


async def test_an_unknown_task_session_still_gets_its_own_queue(tree, monkeypatch):
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)
    _fake_output(tree, ONE_ERROR, monkeypatch)
    T._pending[str(tree)] = {"task:gp:deadbeef": {"src/A.tsx"}}
    await T.run_once(tree)
    assert "task:gp:deadbeef" in TR._pending_by_session


async def test_the_drain_emits_diagnostics_beside_background_tasks(monkeypatch):
    monkeypatch.setattr(BB, "get_bound_session", lambda: SID)
    await TR.enqueue_diagnostics(TR.DiagnosticsRecord(
        session_id=SID, files=["src/A.tsx"], lines=["src/A.tsx(1,1): error TS1: x"],
        started_at=1.0, finished_at=2.0))
    res = await BB.call_tool("_BackgroundTaskDrain", {})
    payload = json.loads(res.content[0].text)
    n = payload["notifications"][0]
    assert n["kind"] == "diagnostics"
    assert n["files"] == ["src/A.tsx"]
    assert "<diagnostics_notification>" in n["xml"]
    assert "<kind>typescript</kind>" in n["xml"]
    assert "src/A.tsx(1,1)" in n["xml"]
    # A background bash row must keep its own shape.
    assert "task_id" not in n


async def test_diagnostics_records_never_reach_the_task_rows():
    await TR.enqueue_diagnostics(TR.DiagnosticsRecord(session_id=SID))
    assert TR.list_active() == []
    assert all(not isinstance(r, TR.DiagnosticsRecord) for r in TR.list_recent(50))


async def test_the_router_labels_a_diagnostics_notification_apart():
    import inspect
    from app.routers import messages as M
    src = inspect.getsource(M._build_notification_drain)
    assert '"diagnostics_notification"' in src
    assert 'n.get("kind") == "diagnostics"' in src


# ── wiring into the Edit result ─────────────────────────────────────────────

async def test_a_tsx_edit_gets_the_queued_hint(tree, monkeypatch):
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    monkeypatch.setattr(T, "DEBOUNCE_S", 30.0)
    f = tree / "web" / "src" / "A.tsx"
    f.write_text("const a = 1\n")
    await FS.call_tool("Read", {"file_path": str(f)})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "1",
                                      "new_string": "2"})
    assert "tsc check queued" in res.content[0].text
    T._timers[str(tree)].cancel()


async def test_a_python_edit_gets_no_tsc_hint(tree, monkeypatch):
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    f = tree / "web" / "src" / "a.py"
    f.write_text("x = 1\n")
    await FS.call_tool("Read", {"file_path": str(f)})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "1",
                                      "new_string": "2"})
    assert "tsc check queued" not in res.content[0].text


async def test_the_kill_switch_removes_the_hint(tree, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "edit_diagnostics": {"typescript": False}})
    f = tree / "web" / "src" / "A.tsx"
    f.write_text("x")
    assert await T.note_edit(SID, str(f)) == ""


async def test_state_exposes_the_last_run(tree, monkeypatch):
    _fake_output(tree, ONE_ERROR, monkeypatch)
    await T.run_once(tree, seed_only=True)
    st = T.stats()
    assert st["last_run"][str(tree)]["status"] == "ok"
    assert str(tree) in st["baseline_roots"]

    from agent_mcp import main as M
    import inspect
    assert '"tsc": _tsc_runner.stats()' in inspect.getsource(M.state)


async def test_shutdown_cancels_pending_timers(tree, monkeypatch):
    monkeypatch.setattr(T, "DEBOUNCE_S", 30.0)
    f = tree / "web" / "src" / "A.tsx"
    f.write_text("x")
    await T.note_edit(SID, str(f))
    task = T._timers[str(tree)]
    await T.shutdown()
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()
    assert T._timers == {}
