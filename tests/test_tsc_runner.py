"""Out-of-band tsc diagnostics: coverage, delta, cold start, delivery.

tsc is whole-project (`web/tsconfig.json` includes only `src`) and ~5 s on
this tree, so it cannot ride back on the Edit. It runs debounced in the
background and the answer arrives through the drain the background-Bash tool
already uses. The interesting failures are all about *attribution*: whose
errors are these, which of them are new, and whether a fresh error belongs to
a file nobody in the run edited — a caller of whatever was just changed.
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

# The cross-file shape this file is really about: a session edits a component,
# and the error lands in the caller that imports it. The session never opens
# `src/Parent.tsx`, so it is in nobody's edit set — and `record.lines` holds
# exactly these display lines, which is why they carry no trailing newline.
CALLER_BREAK = ("src/Parent.tsx(12,8): error TS2741: Property 'old_prop' is "
                "missing in type 'IntrinsicAttributes'.")
CALLER_BREAK2 = ("src/Parent.tsx(30,1): error TS2322: Type 'number' is not "
                 "assignable to type 'string'.")
# A second error in the file the session *did* edit, for the shared cap.
OWN_BREAK2 = ("src/A.tsx(20,1): error TS2345: Argument of type 'X' is not "
              "assignable to parameter of type 'Y'.")
# Errors that exist before the run and not after it: they keep the
# whole-project total flat while a new error appears somewhere else.
OLD_GONE = "src/Old.tsx(1,1): error TS2304: Cannot find name 'was_here'."
OLD_GONE2 = "src/Old2.tsx(1,1): error TS2304: Cannot find name 'also_was_here'."


def _out(*lines: str) -> str:
    """Fake tsc stdout, one display line per argument."""
    return "".join(ln + "\n" for ln in lines)


# ── coverage ────────────────────────────────────────────────────────────────

def test_project_root_walks_up_to_the_tree_that_owns_the_file(tree):
    f = tree / "web" / "src" / "A.tsx"
    f.write_text("x")
    assert T.project_root(str(f)) == tree


def test_a_worktree_answers_about_itself(tmp_path):
    """An automod round edits under ~/lloyd-work/...; the live tree's tsc
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


# ── cross-file breakage ─────────────────────────────────────────────────────
# The five tests below replace `test_an_error_in_a_file_this_session_did_not_
# touch_is_not_reported`, which asserted `TR._pending_by_session == {}` for the
# first shape (#694). Its policy was right about a file *another session is
# editing* and wrong about a file nobody is editing — clause 3 keeps the half
# that was right.

async def test_a_new_error_in_an_untouched_caller_is_reported_when_total_rises(
        tree, monkeypatch):
    """A session edits a component, the error lands in the caller that imports
    it, and the component itself is clean.

    Clause 1. Under the old filter `tsc` had already reported
    `src/Parent.tsx`, `by_file` held it, and the per-session delta dropped it —
    then `if not new: continue` swallowed the run whole, so the session that
    was handed "tsc check queued" heard nothing while the project got worse.
    The whole-project total rising is the evidence that the breakage is real
    rather than a concurrent session's unfinished edit.
    """
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)                # clean tree: total 0
    _fake_output(tree, _out(CALLER_BREAK), monkeypatch)   # total rises to 1
    T._pending[str(tree)] = {SID: {"src/Child.tsx"}}      # the component is clean
    await T.run_once(tree)

    recs = TR._pending_by_session[SID]
    assert len(recs) == 1, "the editing session was told nothing"
    assert recs[0].lines == [], "src/Child.tsx is clean; nothing is owed for it"
    assert recs[0].elsewhere_lines == [CALLER_BREAK]
    assert recs[0].elsewhere_files == ["src/Parent.tsx"]


async def test_a_flat_total_still_delivers_nothing_about_an_untouched_file(
        tree, monkeypatch):
    """Clause 2, and the control for the test above: same seed, same caller
    error, same single edit — only here the total does not rise, because the
    error that appeared displaced one that disappeared.

    Silence about `src/Parent.tsx` is what keeps every concurrent session's
    unfinished work out of everyone else's queue.
    """
    _fake_output(tree, _out(OLD_GONE), monkeypatch)
    await T.run_once(tree, seed_only=True)                # total 1
    _fake_output(tree, _out(CALLER_BREAK), monkeypatch)   # one in, one out: total 1
    T._pending[str(tree)] = {SID: {"src/Child.tsx"}}
    await T.run_once(tree)
    assert TR._pending_by_session == {}


async def test_a_flat_total_reports_the_edited_file_but_not_the_other_one(
        tree, monkeypatch):
    """Clause 2's other half: the rise gate suppresses the new group, not the
    report. `src/A.tsx` is this session's own edit and its error is new, so it
    is still delivered; `src/B.tsx` is nobody's edit and gains an error in the
    same flat-total run, so it is not.
    """
    _fake_output(tree, _out(OLD_GONE, OLD_GONE2), monkeypatch)
    await T.run_once(tree, seed_only=True)                # total 2
    _fake_output(tree, TWO_ERRORS, monkeypatch)           # A and B, both new: total 2
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)

    recs = TR._pending_by_session[SID]
    assert len(recs) == 1
    assert recs[0].lines == [ONE_ERROR.rstrip()]
    assert recs[0].elsewhere_lines == []
    assert "src/B.tsx" not in TR.format_diagnostics_notification(recs[0])


async def test_a_file_another_session_is_editing_is_not_called_elsewhere(
        tree, monkeypatch):
    """Clause 3. One run coalesces every session's pending files, so the
    per-session filter survives as a per-run exclusion: `src/Parent.tsx` is
    sid-b's own edit, not sid-a's collateral damage. The total rose, so the
    rise gate does not protect sid-a — only this exclusion does.
    """
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)
    _fake_output(tree, _out(CALLER_BREAK), monkeypatch)
    T._pending[str(tree)] = {SID: {"src/Child.tsx"}, "sid-b": {"src/Parent.tsx"}}
    await T.run_once(tree)

    assert SID not in TR._pending_by_session, \
        "sid-b's own error was attributed to the session that did not edit it"
    rec = TR._pending_by_session["sid-b"][0]
    assert rec.lines == [CALLER_BREAK], "it belongs in sid-b's own group"
    assert rec.elsewhere_lines == []


async def test_the_two_groups_are_told_apart_in_the_drained_payload(
        tree, monkeypatch):
    """Clause 4, across the seam the finding actually crosses: `run_once`
    enqueues a record, `_BackgroundTaskDrain` renders it to JSON, and the
    router forwards that `xml` verbatim as a user message. If the two groups
    read as one list, a caller's error looks like the session's own bug.
    """
    monkeypatch.setattr(BB, "get_bound_session", lambda: SID)
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)
    _fake_output(tree, _out(ONE_ERROR.rstrip(), CALLER_BREAK), monkeypatch)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)

    payload = json.loads((await BB.call_tool(
        "_BackgroundTaskDrain", {})).content[0].text)
    xml = payload["notifications"][0]["xml"]
    own_block = xml.split("<errors>")[1].split("</errors>")[0]
    assert "src/A.tsx(10,3)" in own_block
    assert "src/Parent.tsx" not in own_block, "a caller's error was filed as its own"
    assert "file(s) you did not edit" in xml.split("<summary>")[1].split("</summary>")[0]
    other_block = xml.split("<errors_outside_your_edits")[1]
    assert "src/Parent.tsx(12,8)" in other_block
    assert "another session" in other_block, "no attribution caveat reached the model"


async def test_the_two_groups_share_one_max_lines_budget(tree, monkeypatch):
    """Clause 4's cap. It bounds the notification, not each group inside it:
    renaming an export in `src/api.ts` breaks every page that imports it, and a
    per-group budget would double the payload every time that happened. The
    session's own findings keep priority — those it can act on — and the files
    attribute still names the ones that were dropped.
    """
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "edit_diagnostics": {"max_lines": 2}})
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)
    _fake_output(tree, _out(ONE_ERROR.rstrip(), OWN_BREAK2,
                            CALLER_BREAK, CALLER_BREAK2), monkeypatch)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)

    rec = TR._pending_by_session[SID][0]
    assert rec.lines == [ONE_ERROR.rstrip(), OWN_BREAK2]
    assert rec.elsewhere_lines == ["... and 2 more"]
    reported = [ln for ln in rec.lines + rec.elsewhere_lines
                if not ln.startswith("... and ")]
    assert len(reported) == 2, "the reported findings exceeded max_lines"
    assert rec.elsewhere_files == ["src/Parent.tsx"], \
        "the dropped findings left no trace of where to look"


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


async def test_the_router_forwards_the_second_group_into_the_turn(
        tree, monkeypatch):
    """The one real process boundary this feature crosses. Everything above it
    is bookkeeping inside the MCP process; here the aggregator's drain closure
    asks the pool over the wire and splices the answer into `chat_messages` as
    a user turn, so this is the hop where the caller's error either reaches the
    model or does not. The chain runs for real: `run_once`, the registry,
    `_BackgroundTaskDrain`, the router's own closure.
    """
    import app.harness.mcp_pool as MP
    from app.routers import messages as M

    monkeypatch.setattr(BB, "get_bound_session", lambda: SID)
    _fake_output(tree, "", monkeypatch)
    await T.run_once(tree, seed_only=True)
    _fake_output(tree, _out(ONE_ERROR.rstrip(), CALLER_BREAK), monkeypatch)
    T._pending[str(tree)] = {SID: {"src/A.tsx"}}
    await T.run_once(tree)

    class _Pool:
        async def call_tool(self, name, arguments, *, session_id=None):
            res = await BB.call_tool(name, arguments)
            return {"content": res.content[0].text, "is_error": False}

    async def _fake_pool(*args, **kwargs):
        return _Pool()
    monkeypatch.setattr(MP, "get_or_open_pool", _fake_pool)

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        M._event_log, "log_event",
        lambda sid, event, data=None, **kw: events.append((event, data)))
    persisted: list[list[dict]] = []

    async def _fake_append(sid, msgs):
        persisted.append(msgs)
    monkeypatch.setattr(M, "_append_messages", _fake_append)

    harness_msgs = await M._build_notification_drain(SID, "turn-seam")()
    assert len(harness_msgs) == 1, "the caller's error never reached the turn"
    xml = harness_msgs[0]["content"]
    assert harness_msgs[0]["role"] == "user"
    assert "<errors_outside_your_edits" in xml
    assert "src/Parent.tsx(12,8)" in xml
    assert persisted and persisted[0][0]["source"] == "diagnostics_notification"
    ready = [d for e, d in events if e == "brain1.diagnostics_ready"]
    assert len(ready) == 1, "the drain logged no diagnostics_ready event"
    assert ready[0]["files"] == ["src/A.tsx"]
    assert ready[0]["elsewhere_files"] == ["src/Parent.tsx"]


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
