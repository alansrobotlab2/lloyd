"""One commit on `main` per landing (Alan, 2026-09-17).

The promoter used to fast-forward the round's whole working history: #1204 put
eight commits on `main`, most of them `test(#1204): …` fix-ups answering a
review. The squash happens after the last gate, so the property worth pinning
is that it can never land a tree the gate did not see.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.automod import promote as P, state as S, worktree as W

ROOT = Path(__file__).resolve().parent.parent


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True)


@pytest.fixture
def branch(tmp_path):
    r = tmp_path / "r"
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "a.py").write_text("A = 1\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    git(r, "checkout", "-q", "-b", "automod/SM_X")
    for i, msg in enumerate(["feat(#7): the change", "test(#7): review fix-up", "docs(#7): say so"]):
        (r / "a.py").write_text(f"A = {i + 2}\n")
        (r / f"f{i}.txt").write_text("x\n")
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", f"{msg}\n\nCo-Authored-By: Lloyd <lloyd@example.invalid>")
    return r, base


def test_three_commits_become_one_with_the_same_tree_and_the_history_is_kept(branch):
    r, base = branch
    old = W.head(r)
    old_tree = git(r, "rev-parse", "HEAD^{tree}").stdout.strip()
    msg = P.squash_message("SM_X", "Item #7: the change, as the board names it", r, base)
    new, note = W.squash_onto(r, base, msg, keep_ref="refs/automod/rounds/SM_X")
    assert new and new != old and "squashed 3 commits" in note
    assert git(r, "rev-parse", "HEAD^{tree}").stdout.strip() == old_tree, "what was gated is what lands"
    assert git(r, "rev-list", "--count", f"{base}..HEAD").stdout.strip() == "1"
    assert git(r, "rev-parse", "HEAD^").stdout.strip() == base, "still a fast-forward from live"
    assert git(r, "rev-parse", "refs/automod/rounds/SM_X").stdout.strip() == old, "working history reachable"
    assert "SM_X" not in git(r, "branch", "--list").stdout.replace("automod/SM_X", ""), "not a branch"
    body = git(r, "log", "-1", "--format=%B").stdout
    assert body.startswith("Item #7: the change, as the board names it\n")
    assert "squashed at landing from 3 commit(s)" in body
    assert "feat(#7): the change" in body and "docs(#7): say so" in body
    assert body.count("Co-Authored-By: Lloyd") == 1, "trailers deduplicated, not dropped"


def test_a_single_commit_round_is_left_exactly_as_gated(branch):
    r, base = branch
    git(r, "reset", "-q", "--hard", f"{base}")
    (r / "a.py").write_text("A = 9\n")
    git(r, "commit", "-qam", "one")
    old = W.head(r)
    new, note = W.squash_onto(r, base, "msg")
    assert new is None and "nothing to squash" in note and W.head(r) == old


def test_a_branch_that_is_not_on_top_of_live_is_left_for_the_fast_forward_to_refuse(branch):
    r, base = branch
    old = W.head(r)
    git(r, "checkout", "-q", "main")
    (r / "other.txt").write_text("moved\n"); git(r, "add", "-A"); git(r, "commit", "-q", "-m", "main moved")
    moved = git(r, "rev-parse", "HEAD").stdout.strip()
    git(r, "checkout", "-q", "automod/SM_X")
    new, note = W.squash_onto(r, moved, "msg")
    assert new is None and "not an ancestor" in note and W.head(r) == old


def test_uncommitted_edits_in_the_worktree_stop_the_squash(branch):
    """`reset --soft` + commit would otherwise sweep them into what lands."""
    r, base = branch
    old = W.head(r)
    (r / "a.py").write_text("A = 'not gated'\n")
    new, note = W.squash_onto(r, base, "msg")
    assert new is None and "uncommitted" in note and W.head(r) == old


def test_a_squash_that_cannot_reproduce_the_tree_restores_the_branch(branch, monkeypatch):
    r, base = branch
    old = W.head(r)
    real = W.git

    def sabotage(repo, *args, **kw):
        if args[:1] == ("commit",):
            (Path(repo) / "a.py").write_text("A = 'tampered'\n")
            real(repo, "add", "-A")
        return real(repo, *args, **kw)
    monkeypatch.setattr(W, "git", sabotage)
    new, note = W.squash_onto(r, base, "msg")
    assert new is None and "branch restored" in note
    monkeypatch.setattr(W, "git", real)
    assert W.head(r) == old and git(r, "status", "--porcelain").stdout.strip() == ""


def test_the_promoter_rewrites_its_record_before_the_merge_and_the_switch_defaults_on(monkeypatch):
    src = (ROOT / "scripts/automod/promote.py").read_text().split("def promote(")[1]
    squash, record, merge = (src.index("W.squash_onto("), src.index('current["commit"] = result["commit"] = head'),
                             src.index('"merge", "--ff-only"'))
    assert squash < record < merge, "the guardian rolls back by current.json: it must name what lands"
    assert src.index("_regate_after_move") < squash, "after the last gate, never before one"
    monkeypatch.setattr(S, "landing_cfg", lambda repo=None: {})
    assert P.squash_enabled() is True
    monkeypatch.setattr(S, "landing_cfg", lambda repo=None: {"squash": False})
    assert P.squash_enabled() is False
    monkeypatch.undo()
    assert S.landing_cfg(ROOT).get("squash") is True, "config.yaml ships it on"
