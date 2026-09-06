"""Rollback mechanics against a real throwaway git repo.

Nothing here touches the live tree or the live supervisord: the repo is built
in `tmp_path` and supervisord is a recording fake. What the tests are really
protecting is a short list of properties that are cheap to get wrong and
expensive to get wrong in production:

  * the tree ends on `main`, not detached — `git checkout <sha>` would satisfy
    "HEAD == target" while quietly breaking the isolation model;
  * `git clean` never reaches the repo root, where usage.db / workers.db /
    .env / .venvs live, all gitignored and none replaceable;
  * a dirty tree is preserved, never silently discarded;
  * a rollback with nothing to roll back to refuses rather than guessing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

GUARDIAN_DIR = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import gstate     # noqa: E402
import policy     # noqa: E402
import rollback as rb  # noqa: E402


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)


@pytest.fixture()
def repo(tmp_path):
    """Two commits (A then B), a .gitignore, and irreplaceable root state."""
    r = tmp_path / "lloyd"
    (r / "app").mkdir(parents=True)
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")

    (r / ".gitignore").write_text("*.db\n.env\n.venvs/\n", encoding="utf-8")
    (r / "app" / "mod.py").write_text("VALUE = 'A'\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "A")
    sha_a = git(r, "rev-parse", "HEAD").stdout.strip()

    (r / "app" / "mod.py").write_text("VALUE = 'B'\n", encoding="utf-8")
    (r / "app" / "added_by_b.py").write_text("BROKEN = True\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "B")
    sha_b = git(r, "rev-parse", "HEAD").stdout.strip()

    # Gitignored, irreplaceable, at the repo root — must survive a rollback.
    (r / "usage.db").write_bytes(b"precious")
    (r / "workers.db").write_bytes(b"precious")
    (r / ".env").write_text("SECRET=1\n", encoding="utf-8")
    return {"path": r, "a": sha_a, "b": sha_b}


# ---------------------------------------------------------------------------
# restore_tree
# ---------------------------------------------------------------------------

def test_restore_lands_on_the_target_and_stays_on_main(repo):
    r = repo["path"]
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    rb.verify_tree(str(r), repo["a"])
    assert rb.head_commit(str(r)) == repo["a"]
    # The assertion that matters: `git checkout <sha>` would also satisfy the
    # line above while detaching HEAD and breaking the isolation model.
    assert rb.head_branch(str(r)) == "refs/heads/main"
    assert (r / "app" / "mod.py").read_text() == "VALUE = 'A'\n"


def test_verify_refuses_a_detached_head(repo):
    r = repo["path"]
    git(r, "checkout", "-q", "--detach", repo["a"])
    with pytest.raises(rb.RollbackError, match="expected refs/heads/main"):
        rb.verify_tree(str(r), repo["a"])


def test_verify_refuses_when_the_tree_did_not_move(repo):
    with pytest.raises(rb.RollbackError, match="after reset"):
        rb.verify_tree(str(repo["path"]), repo["a"])


def test_untracked_python_under_a_clean_path_is_removed(repo):
    """A file the bad commit left behind can shadow an import after the reset."""
    r = repo["path"]
    stray = r / "app" / "stray.py"
    stray.write_text("BOOM = 1\n", encoding="utf-8")
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    assert not stray.exists()


def test_gitignored_root_state_survives_the_clean(repo):
    """The data-loss pair. A bare `git clean -fdx` here would be an incident."""
    r = repo["path"]
    rb.restore_tree(str(r), repo["a"], policy.CLEAN_PATHS, policy.PYCACHE_PATHS)
    assert (r / "usage.db").read_bytes() == b"precious"
    assert (r / "workers.db").read_bytes() == b"precious"
    assert (r / ".env").exists()


def test_pycache_under_code_dirs_is_dropped(repo):
    r = repo["path"]
    cache = r / "app" / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-312.pyc").write_bytes(b"stale")
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    assert not cache.exists()


def test_restore_to_a_missing_commit_raises(repo):
    with pytest.raises(rb.RollbackError, match="failed"):
        rb.restore_tree(str(repo["path"]), "0" * 40, ("app",), ("app",))


def test_restore_is_idempotent(repo):
    r = repo["path"]
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    rb.verify_tree(str(r), repo["a"])


# ---------------------------------------------------------------------------
# Evidence preservation
# ---------------------------------------------------------------------------

def test_a_dirty_tree_is_preserved_before_it_is_destroyed(repo, tmp_path):
    """A rollback that erases the bug guarantees you fix it twice."""
    r = repo["path"]
    (r / "app" / "mod.py").write_text("VALUE = 'uncommitted work'\n", encoding="utf-8")
    (r / "app" / "scratch.py").write_text("notes = 1\n", encoding="utf-8")

    ev = rb.preserve_evidence(str(r), tmp_path / "broken" / "ts", "guardian-broken-ts")
    assert ev["tag"] == "guardian-broken-ts"
    assert ev["stash"], "dirty tree was not stashed"
    assert ev["patch"] and Path(ev["patch"]).read_text().strip()
    assert "app/scratch.py" in ev["untracked"]

    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    stashes = git(r, "stash", "list").stdout
    assert "guardian-rollback-guardian-broken-ts" in stashes


def test_the_tag_keeps_the_bad_commit_reachable(repo, tmp_path):
    r = repo["path"]
    rb.preserve_evidence(str(r), tmp_path / "b", "guardian-broken-x")
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    assert rb.commit_exists(str(r), repo["b"])
    assert "guardian-broken-x" in git(r, "tag", "-l").stdout


def test_preserve_on_a_clean_tree_records_no_stash(repo, tmp_path):
    ev = rb.preserve_evidence(str(repo["path"]), tmp_path / "b", "tag-clean")
    assert ev["stash"] is None


# ---------------------------------------------------------------------------
# index.lock
# ---------------------------------------------------------------------------

def test_a_stale_index_lock_is_removed(repo):
    lock = repo["path"] / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    os.utime(lock, (time.time() - 600, time.time() - 600))
    rb._wait_for_index_lock(str(repo["path"]), stale_seconds=60.0, budget=2.0)
    assert not lock.exists()


def test_a_fresh_index_lock_is_waited_on_not_stolen(repo):
    lock = repo["path"] / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    started = time.time()
    rb._wait_for_index_lock(str(repo["path"]), stale_seconds=60.0, budget=1.0)
    assert lock.exists(), "a fresh lock must not be stolen"
    assert time.time() - started >= 0.9


# ---------------------------------------------------------------------------
# Ancestry / floor guards
# ---------------------------------------------------------------------------

def test_ancestry_helpers(repo):
    r = str(repo["path"])
    assert rb.is_ancestor(r, repo["a"], repo["b"])
    assert not rb.is_ancestor(r, repo["b"], repo["a"])
    assert rb.commit_exists(r, repo["a"])
    assert not rb.commit_exists(r, "0" * 40)


# ---------------------------------------------------------------------------
# venv swap
# ---------------------------------------------------------------------------

def test_venv_swap_back_restores_the_previous_clone(repo):
    r = repo["path"]
    venvs = r / ".venvs"
    (venvs / "lloyd").mkdir(parents=True)
    (venvs / "lloyd" / "marker").write_text("candidate", encoding="utf-8")
    (venvs / "lloyd.prev").mkdir()
    (venvs / "lloyd.prev" / "marker").write_text("previous", encoding="utf-8")

    failed = rb.swap_venv_back(str(r))
    assert (venvs / "lloyd" / "marker").read_text() == "previous"
    assert failed and Path(failed).exists(), "the failed clone must be kept for forensics"


def test_venv_swap_back_is_a_noop_without_a_previous(repo):
    (repo["path"] / ".venvs" / "lloyd").mkdir(parents=True)
    assert rb.swap_venv_back(str(repo["path"])) is None


# ---------------------------------------------------------------------------
# Worktree safety
# ---------------------------------------------------------------------------

def test_a_linked_worktree_is_untouched_by_a_rollback(repo, tmp_path):
    """The guardian must never reach into a round's worktree."""
    r = repo["path"]
    wt = tmp_path / "work"
    git(r, "worktree", "add", "-q", "-b", "selfmod/x", str(wt), repo["b"])
    (wt / "app" / "mod.py").write_text("VALUE = 'worktree'\n", encoding="utf-8")

    rb.restore_tree(str(r), repo["a"], policy.CLEAN_PATHS, policy.PYCACHE_PATHS)
    assert (wt / "app" / "mod.py").read_text() == "VALUE = 'worktree'\n"
    assert rb.head_commit(str(wt)) == repo["b"]


# ---------------------------------------------------------------------------
# State: rollback target resolution and its degradation ladder
# ---------------------------------------------------------------------------

def test_target_comes_from_the_lkg_pointer(tmp_path):
    st = gstate.SelfModState(tmp_path)
    st.set_lkg("a" * 40)
    target, source = st.rollback_target()
    assert target == "a" * 40 and "last_known_good" in source


def test_target_falls_back_to_the_ledger_when_the_pointer_is_gone(tmp_path):
    st = gstate.SelfModState(tmp_path)
    gstate.append_event(st.ledger, {"event": "promoted", "commit": "b" * 40,
                                    "parent": "a" * 40})
    target, source = st.rollback_target()
    assert target == "a" * 40 and "ledger" in source


def test_a_malformed_pointer_falls_through_rather_than_crashing(tmp_path):
    st = gstate.SelfModState(tmp_path)
    st.lkg_path.parent.mkdir(parents=True, exist_ok=True)
    st.lkg_path.write_text("{not json", encoding="utf-8")
    gstate.append_event(st.ledger, {"event": "promoted", "commit": "b" * 40,
                                    "parent": "c" * 40})
    target, _ = st.rollback_target()
    assert target == "c" * 40


def test_no_target_anywhere_refuses_rather_than_guessing(tmp_path):
    """There is deliberately no fourth step in the ladder.

    A watchdog that guesses at a commit is worse than one that pages a human.
    """
    st = gstate.SelfModState(tmp_path)
    target, source = st.rollback_target()
    assert target is None and "no usable" in source


def test_a_short_or_non_hex_commit_is_not_accepted(tmp_path):
    st = gstate.SelfModState(tmp_path)
    gstate.write_json_atomic(st.lkg_path, {"schema": 1, "commit": "abc123"})
    assert st.rollback_target()[0] is None


def test_the_floor_is_set_once_and_never_moves(tmp_path):
    st = gstate.SelfModState(tmp_path)
    st.set_lkg("a" * 40)
    st.set_lkg("b" * 40)
    assert st.floor() == "a" * 40


def test_pause_is_capped_by_the_snapshots_own_policy(tmp_path):
    """A forgotten or over-long lease must not disable the watchdog."""
    st = gstate.SelfModState(tmp_path)
    st.pause.parent.mkdir(parents=True, exist_ok=True)
    st.pause.write_text(str(time.time() + 10 * 24 * 3600), encoding="utf-8")
    assert st.pause_remaining(cap=1800.0) == pytest.approx(1800.0, abs=1.0)


def test_an_expired_pause_reads_as_zero(tmp_path):
    st = gstate.SelfModState(tmp_path)
    st.pause.parent.mkdir(parents=True, exist_ok=True)
    st.pause.write_text(str(time.time() - 5), encoding="utf-8")
    assert st.pause_remaining(cap=1800.0) == 0.0


def test_unfinished_rollback_is_detected_for_resume(tmp_path):
    st = gstate.SelfModState(tmp_path)
    gstate.append_event(st.ledger, {"event": "rollback_started", "to": "a" * 40})
    assert st.unfinished_rollback() is not None
    gstate.append_event(st.ledger, {"event": "rollback_succeeded", "restored": "a" * 40})
    assert st.unfinished_rollback() is None


def test_recent_rollbacks_counts_only_inside_the_window(tmp_path):
    st = gstate.SelfModState(tmp_path)
    st.ledger.parent.mkdir(parents=True, exist_ok=True)
    old = {"event": "rollback_succeeded", "ts": time.time() - 10 * 3600}
    recent = {"event": "rollback_succeeded", "ts": time.time() - 60}
    with open(st.ledger, "w", encoding="utf-8") as f:
        f.write(json.dumps(old) + "\n")
        f.write(json.dumps(recent) + "\n")
    assert st.recent_rollbacks(6 * 3600) == 1, "the 10h-old rollback must age out"
    assert st.recent_rollbacks(24 * 3600) == 2
