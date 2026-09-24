"""The live tree is shared, and the loop has to keep up with it.

A human works on `main` while rounds are open. `round.start` used to refuse a
dirty tree outright, `run_gate` had no way to record a base that moved, and a
`land_failed` after a passing gate was invisible to the backlog. These pin the
three joints that are not the gate or the promoter themselves.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.automod import backlog as B, round as R, state as S, worktree as W


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture()
def scratch(tmp_path, monkeypatch):
    live = tmp_path / "live"; (live / "app").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(live))
    git(live, "config", "user.email", "t@e.com"); git(live, "config", "user.name", "t")
    (live / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    git(live, "add", "-A"); git(live, "commit", "-q", "-m", "base")
    for name in ("STATE_DIR", "BROKEN_DIR"):
        monkeypatch.setattr(S, name, tmp_path / "state")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state" / "rounds")
    for name, fn in (("LEDGER_PATH", "promotions.jsonl"), ("HALTED_PATH", "halted"),
                     ("BROKEN_PATH", "BROKEN"), ("LOCK_PATH", "lock"), ("CURRENT_PATH", "current.json")):
        monkeypatch.setattr(S, name, tmp_path / "state" / fn)
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    return live


def test_round_start_records_live_dirt_instead_of_refusing(scratch):
    """A worktree is cut from HEAD — committed state — and an uncommitted edit
    in production cannot reach it. Refusing blocked `automod_start` for #448
    over one orphaned file. Recorded, not refused."""
    (scratch / "app" / "wip.py").write_text("x = 1\n", encoding="utf-8")
    out = R.start("a goal", force=True)
    try:
        assert out["live_dirty_paths"] == ["app/wip.py"]
        assert "tolerates them" in out["note"]
        assert Path(out["worktree"]).exists()
        ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_start"][-1]
        assert ev["live_dirty_paths"] == ["app/wip.py"]
    finally:
        W.remove(out["round_id"], repo=scratch)


def test_run_gate_persists_a_base_the_preflight_rebased_onto(scratch, monkeypatch):
    """`run_gate` reads the base from the run spec, and `changed_paths` is
    `base...HEAD`. Leave the old base after a rebase and the next gate sweeps
    the human's commits into the round's diff."""
    out = R.start("a goal", force=True)
    rid, old = out["round_id"], out["base"]
    new_base = "f" * 40

    class Stub:
        def __init__(self, *a, **k): pass
        def run(self):
            return type("Rep", (), {"ok": True, "base": new_base, "head": "e" * 40,
                                    "to_dict": lambda s: {"ok": True, "base": new_base,
                                                          "head": "e" * 40, "rungs": []}})()
    monkeypatch.setattr(R.G, "Gate", Stub)
    try:
        rep = R.run_gate(rid)
        assert rep["base"] == new_base
        spec = (S.ROUNDS_DIR / rid / "run_spec.yaml").read_text()
        assert f"base_commit: {new_base}" in spec and old not in spec
        assert (S.ROUNDS_DIR / rid / "gate.json").exists()
    finally:
        W.remove(rid, repo=scratch)


def test_update_run_spec_base_is_a_noop_without_a_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path)
    assert S.update_run_spec_base("SM_NONE", "a" * 40) is False


def test_a_land_failed_after_a_passing_gate_is_the_rounds_last_verdict(tmp_path, monkeypatch):
    """Every rung passed; then `main` moved faster than the loop could chase.
    Before `land_failed`, that round looked exactly like one that landed —
    and spent the item."""
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", ledger)
    for rung in ("preflight", "static", "tests", "drill"):
        S.append_event({"event": "gate", "round_id": "SM_L", "rung": rung, "ok": True}, path=ledger)
    assert B.externally_blocked_rounds(ledger) == set()
    S.append_event({"event": "land_failed", "round_id": "SM_L", "ok": False,
                    "external_blocker": True, "detail": "fast-forward failed"}, path=ledger)
    assert B.externally_blocked_rounds(ledger) == {"SM_L"}
    # ...and a later gate that passes supersedes it in turn.
    S.append_event({"event": "gate", "round_id": "SM_L", "rung": "drill", "ok": True}, path=ledger)
    assert B.externally_blocked_rounds(ledger) == set()


def test_dirty_paths_reports_renames_on_both_sides(tmp_path):
    repo = tmp_path / "r"; repo.mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    git(repo, "config", "user.email", "t@e.com"); git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("1\n", encoding="utf-8")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "x")
    git(repo, "mv", "a.py", "b.py")
    (repo / "c.py").write_text("3\n", encoding="utf-8")
    assert W.dirty_paths(repo) == ["a.py", "b.py", "c.py"]
    assert W.dirty_paths(repo, limit=1) == ["a.py"]


# ---------------------------------------------------------------------------
# The round is told which interpreter it has (#692)
# ---------------------------------------------------------------------------

def test_round_start_hands_the_round_an_absolute_interpreter(scratch):
    """`.venvs/` is gitignored, so the worktree has none, and the relative form
    every handoff writes — `.venvs/lloyd/bin/python -m pytest …` — dies inside
    the round with `No such file or directory`, which reads exactly like a
    failed acceptance check. The start response names the interpreter instead.
    """
    out = R.start("a goal", force=True)
    try:
        expected = scratch / ".venvs" / "lloyd" / "bin" / "python"
        assert out["venv_python"] == str(expected)
        assert Path(out["venv_python"]).is_absolute(), "cwd-relative is the bug"
    finally:
        W.remove(out["round_id"], repo=scratch)


def test_the_run_spec_carries_that_interpreter_and_still_validates(scratch):
    """Same value, in the file a resumed or detached reader reads.

    `validate_code_run_spec` requires `code.base_commit` and `code.branch` and
    rejects no extra key, so the entry has to ride through unchanged — a spec
    that stopped validating would refuse the round its own gate runs on.
    """
    import yaml

    from scripts.automod import spec as SP
    out = R.start("a goal", force=True)
    try:
        loaded = yaml.safe_load(Path(out["run_spec"]).read_text(encoding="utf-8"))
        assert loaded["code"]["venv_python"] == out["venv_python"]
        assert SP.validate_code_run_spec(loaded) is None
    finally:
        W.remove(out["round_id"], repo=scratch)


async def test_the_mcp_tool_hands_that_interpreter_to_the_agent(scratch, monkeypatch):
    """The process boundary the round actually crosses.

    An implementer never calls `round.start()` — it calls the `automod_start`
    MCP tool, which serialises the dict with `json.dumps`. A key dropped
    between the two is invisible to a test on the in-process return value. The
    IV gate is stubbed: what is under test is the payload, not who may call.
    """
    import json

    import agent_mcp.automod as AM
    monkeypatch.setattr(AM, "_inner_voice_gate", lambda action: None)
    monkeypatch.setattr(AM, "_enabled", lambda: True)
    # The tool wrapper checks `automod.enabled` and then calls `start` WITHOUT
    # `force`, which checks it again inside `state`. Both are config reads, and
    # the loop ships inert in config; what is under test is the payload.
    monkeypatch.setattr(S, "require_enabled", lambda action, root: None)
    rid, payload = None, {}
    try:
        payload = json.loads((await AM.call_tool(
            "automod_start", {"goal": "a goal"})).content[0].text)
        rid = payload.get("round_id")
        assert payload["venv_python"] == str(
            scratch / ".venvs" / "lloyd" / "bin" / "python")
    finally:
        if rid:
            W.remove(rid, repo=scratch)


# ---------------------------------------------------------------------------
# A close preserves and names the live checkout's uncommitted work (#1037)
# ---------------------------------------------------------------------------

def _last_event(kind: str) -> dict:
    rows = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]
    assert rows, f"no {kind} row was written"
    return rows[-1]


def test_abort_copies_the_live_uncommitted_work_into_the_round_state_dir(scratch):
    """An edit made in the live checkout exists in ONE copy, and it is not on
    the branch `abort` keeps: not in HEAD, not in the worktree `W.remove`
    deletes, not in the stash (which a round must never touch). `649193f`
    turned the blanket refusal into tolerated-and-recorded dirt and so removed
    the only moment the loop ever noticed one — measured on 2026-09-18, 45 of
    277 `round_start` rows carried `live_dirty_paths` and 0 of the 250 close
    rows carried anything about it.

    Preservation is a patch plus the untracked source, on disk, under a path
    with the round id in it — never a `git stash` entry, the shared LIFO stack
    whose top an unrelated round popped #573's diff off on 2026-09-11.
    """
    out = R.start("a goal", force=True)
    rid = out["round_id"]
    (scratch / "app" / "m.py").write_text("V = 2\n", encoding="utf-8")          # tracked edit
    (scratch / "app" / "wip.py").write_text("ORPHAN = 1\n", encoding="utf-8")   # untracked source
    res = R.abort(rid, reason="done here")

    assert res["live_dirty_paths"] == ["app/m.py", "app/wip.py"]
    dest = S.ROUNDS_DIR / rid / "live-dirty"
    assert res["live_dirty_dir"] == str(dest)
    patch = Path(res["live_dirty_patch"])
    assert patch == dest / "dirty.patch" and patch.is_file(), "the tracked edit was not preserved"
    assert "+V = 2" in patch.read_text(encoding="utf-8")
    copy = dest / "app" / "wip.py"
    assert copy.is_file() and copy.read_text(encoding="utf-8") == "ORPHAN = 1\n", \
        "git diff HEAD cannot carry an untracked file; the copy is what does"
    assert res["live_dirty_untracked"] == ["app/wip.py"]
    assert "live_dirty_error" not in res

    # What abort always managed is still managed, so preservation did not
    # become the close: worktree gone, branch kept as the forensic record.
    assert not Path(out["worktree"]).exists()
    assert W.branch_exists(scratch, f"automod/{rid}")
    # And the live stash stack is untouched — read it, never write it.
    assert git(scratch, "stash", "list").stdout.strip() == "", "a close must not create a stash"


def test_the_round_aborted_row_carries_the_paths_and_the_patch(scratch):
    """The file is no use to a reader who cannot find it. `git status` is one
    command away, but the ledger row is where a close is recorded, and 250
    consecutive close rows named nothing."""
    out = R.start("a goal", force=True)
    (scratch / "app" / "m.py").write_text("V = 3\n", encoding="utf-8")
    (scratch / "app" / "wip.py").write_text("ORPHAN = 1\n", encoding="utf-8")
    res = R.abort(out["round_id"], reason="orphan edit outstanding")
    ev = _last_event("round_aborted")
    assert ev["live_dirty_paths"] == W.dirty_paths(scratch) == ["app/m.py", "app/wip.py"]
    assert ev["live_dirty_patch"] == res["live_dirty_patch"]
    assert Path(ev["live_dirty_patch"]).is_file(), "the row named a file that is not there"


def test_the_row_caps_the_paths_at_twenty_while_the_copy_keeps_every_one(scratch):
    """The cap is on what rides the event, matching `start`'s `live_dirty_paths`
    bound since `649193f`. It must not quietly become a bound on what is
    preserved, or a 25-file tree loses 5 files to a slice."""
    out = R.start("a goal", force=True)
    rid = out["round_id"]
    for i in range(25):
        (scratch / "app" / f"wip{i}.py").write_text(f"X = {i}\n", encoding="utf-8")
    res = R.abort(rid, reason="a wide tree")
    assert len(W.dirty_paths(scratch)) == 25
    assert _last_event("round_aborted")["live_dirty_paths"] == res["live_dirty_paths"]
    assert len(res["live_dirty_paths"]) == 20
    assert len(res["live_dirty_untracked"]) == 25
    assert (S.ROUNDS_DIR / rid / "live-dirty" / "app" / "wip24.py").is_file()


def test_a_clean_live_tree_writes_no_patch_and_names_no_paths(scratch):
    """The overwhelmingly common case: the close stays as quiet as it was, and
    an empty list means empty rather than "we did not look"."""
    out = R.start("a goal", force=True)
    rid = out["round_id"]
    res = R.abort(rid, reason="nothing to do")
    assert res["live_dirty_paths"] == []
    assert "live_dirty_patch" not in res and "live_dirty_dir" not in res
    assert not (S.ROUNDS_DIR / rid / "live-dirty").exists(), "a clean tree wrote files anyway"
    ev = _last_event("round_aborted")
    assert ev["live_dirty_paths"] == [] and "live_dirty_patch" not in ev


def test_automod_status_reports_the_live_checkouts_changed_paths(scratch):
    """`status()` grew `unit_drift` for systemd copies and never a word about
    the tracked files themselves, so `automod_status` — the tool a session
    reaches for first — cannot see an orphan edit. `/health` publishes dirt as
    a boolean and nothing else."""
    assert R.status()["live_dirty_paths"] == []
    (scratch / "app" / "wip.py").write_text("ORPHAN = 1\n", encoding="utf-8")
    (scratch / "app" / "m.py").write_text("V = 4\n", encoding="utf-8")
    assert R.status()["live_dirty_paths"] == W.dirty_paths(scratch)
    assert R.status()["live_dirty_paths"] == ["app/m.py", "app/wip.py"]


def test_status_says_so_when_the_live_tree_cannot_be_read(scratch, tmp_path, monkeypatch):
    """`W.dirty_paths` returns `[]` for a clean tree AND for a `git status` that
    failed, and the two read alike — the check-reads-its-own-missing-input
    shape this loop has been burned by repeatedly. The empty list has to come
    with the reason, and the way to prove the reason surfaces is a root that
    really is not a repo, not a stub."""
    nowhere = tmp_path / "not-a-repo"
    nowhere.mkdir()
    monkeypatch.setattr(R, "LIVE_ROOT", nowhere)
    st = R.status()
    assert st["live_dirty_paths"] == []
    assert "could not be read" in st["live_dirty_error"]


async def test_the_mcp_tools_carry_the_preserved_paths_across_the_wire(scratch, monkeypatch):
    """The boundary an agent actually crosses: `automod_abort` and
    `automod_status` run in lloyd-mcp and hand back `json.dumps` of these dicts
    (`agent_mcp/automod.py`). A key that survives the in-process return and not
    the serialisation is invisible to every test above, and the agent reads
    only what arrives here."""
    import json

    import agent_mcp.automod as AM
    monkeypatch.setattr(AM, "_enabled", lambda: True)
    out = R.start("a goal", force=True)
    rid = out["round_id"]
    (scratch / "app" / "m.py").write_text("V = 5\n", encoding="utf-8")
    before = json.loads((await AM.call_tool("automod_status", {})).content[0].text)
    assert before["live_dirty_paths"] == ["app/m.py"]
    aborted = json.loads((await AM.call_tool(
        "automod_abort", {"round_id": rid, "reason": "orphan edit"})).content[0].text)
    assert aborted["live_dirty_paths"] == ["app/m.py"]
    assert Path(aborted["live_dirty_patch"]).is_file()
