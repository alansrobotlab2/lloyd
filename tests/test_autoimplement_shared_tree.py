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

from scripts.autoimplement import backlog as B, round as R, state as S, worktree as W


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
    in production cannot reach it. Refusing blocked `autoimplement_start` for #448
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
