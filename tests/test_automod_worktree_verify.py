"""The verify command a handoff writes has to run inside the round (#692).

`.venvs/` is gitignored: it exists only in the live checkout, so a round
worktree has no interpreter at all until the gate's `venv` rung clones one —
and that rung is skipped whole on a requirements-unchanged diff, after the
implement turn in which the verify command actually runs. The vault's default
form (`.venvs/lloyd/bin/python`, `CLAUDE.md:12`, 247 files under `~/obsidian`)
therefore fails inside a round with `No such file or directory`, which is
indistinguishable from a failed acceptance check at the moment it decides
whether the implementer trusts the check.

These two run the SAME pytest invocation on the SAME scratch round worktree,
differing only in which interpreter the round was told to use.
"""
from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.automod import round as R, state as S, worktree as W

PYTEST_TIMEOUT = 300.0


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture()
def live(tmp_path, monkeypatch):
    """A scratch live checkout whose `.venvs/lloyd/bin/python` is executable.

    The stand-in execs the interpreter running this test suite, so `-m pytest`
    works through it exactly as it does through the real venv, and the test
    pins the mechanism (an absolute path, cwd-relative to nothing) rather than
    one particular build of Python.
    """
    root = tmp_path / "live"
    (root / "app").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(root))
    git(root, "config", "user.email", "t@e.com")
    git(root, "config", "user.name", "t")
    (root / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")

    shim = root / ".venvs" / "lloyd" / "bin" / "python"
    shim.parent.mkdir(parents=True)
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    for name in ("STATE_DIR", "BROKEN_DIR"):
        monkeypatch.setattr(S, name, tmp_path / "state")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state" / "rounds")
    for name, fn in (("LEDGER_PATH", "promotions.jsonl"), ("HALTED_PATH", "halted"),
                     ("BROKEN_PATH", "BROKEN"), ("LOCK_PATH", "lock"),
                     ("CURRENT_PATH", "current.json")):
        monkeypatch.setattr(S, name, tmp_path / "state" / fn)
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(R, "LIVE_ROOT", root)
    monkeypatch.setattr(W, "LIVE_ROOT", root)
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    return root


def _one_test_round(live) -> tuple[dict, Path]:
    """Open a round and drop a single passing test into its worktree."""
    out = R.start("a goal", force=True)
    wt = Path(out["worktree"])
    (wt / "tests").mkdir(parents=True, exist_ok=True)
    (wt / "tests" / "test_probe.py").write_text("def test_probe():\n    assert True\n",
                                                encoding="utf-8")
    return out, wt


def test_the_interpreter_the_round_is_handed_runs_pytest_from_the_worktree(live):
    """The acceptance check: the returned `venv_python`, executed with cwd set
    to the new worktree, exits 0. This is the command a handoff writes.
    """
    out, wt = _one_test_round(live)
    try:
        assert not (wt / ".venvs").exists(), "a provisioned worktree would hide the defect"
        r = subprocess.run([out["venv_python"], "-m", "pytest", "tests/test_probe.py", "-q"],
                           cwd=wt, capture_output=True, text=True, timeout=PYTEST_TIMEOUT)
        assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
        assert "1 passed" in r.stdout
    finally:
        W.remove(out["round_id"], repo=live)


def test_the_documented_relative_form_is_still_dead_inside_the_round(live):
    """The premise, pinned so a future that provisions `.venvs` says so here
    rather than leaving the handoff wording unexamined.
    """
    out, wt = _one_test_round(live)
    try:
        r = subprocess.run(["/bin/sh", "-c",
                            ".venvs/lloyd/bin/python -m pytest tests/test_probe.py -q"],
                           cwd=wt, capture_output=True, text=True, timeout=PYTEST_TIMEOUT)
        assert r.returncode == 127, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
        assert "No such file or directory" in r.stderr
    finally:
        W.remove(out["round_id"], repo=live)
