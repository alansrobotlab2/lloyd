"""A round id becomes a path in exactly one place, and that path reaches `rmtree`.

`worktree.remove` deletes `WORK_ROOT / round_id`. `automod_abort` handed it
`str(arguments.get("round_id") or "")` with nothing in between, so an omitted id
deleted the work root (every open round), `".."` the directory above it, and an
absolute id whatever it named — the 2026-09-29 shape (a computed path reaching a
recursive delete) on a tool a worker model calls. These nodes pin the refusal at
both ends: the path is never computed, and the tool says so instead of raising.
"""

from __future__ import annotations

import pytest

from agent_mcp import automod as AM
from scripts.automod import worktree as W

BAD_IDS = ["", ".", "..", "../lloyd", "../lloyd-data", "a/b", "/abs/path", "SM_1/..",
           " SM_1", "-rf", ".hidden", None, 7]
GOOD_IDS = ["SM_20261108_001545", "review_cal_392-small-met", "SM_T", "r1"]


@pytest.fixture()
def work(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    (tmp_path / "precious").mkdir()
    (tmp_path / "precious" / "file").write_text("keep", encoding="utf-8")
    (root / "SM_other" / "home").mkdir(parents=True)
    monkeypatch.setattr(W, "WORK_ROOT", root)
    monkeypatch.setattr(W, "git", lambda *a, **k: None)   # no repo is touched here
    return root


@pytest.mark.parametrize("rid", BAD_IDS)
def test_round_dir_refuses_an_id_that_is_not_one_child_of_the_work_root(work, rid):
    assert not W.valid_round_id(rid)
    with pytest.raises(ValueError):
        W.round_dir(rid)


@pytest.mark.parametrize("rid", GOOD_IDS)
def test_every_id_shape_the_loop_mints_is_accepted(work, rid):
    assert W.valid_round_id(rid)
    assert W.round_dir(rid).parent == work


@pytest.mark.parametrize("rid", ["", "..", "../precious", "SM_other/.."])
def test_remove_deletes_nothing_for_a_bad_id(work, tmp_path, rid):
    with pytest.raises(ValueError):
        W.remove(rid)
    assert (work / "SM_other" / "home").is_dir(), "another round's worktree went with it"
    assert (tmp_path / "precious" / "file").read_text(encoding="utf-8") == "keep"


def test_remove_will_not_follow_a_symlink_out_of_the_work_root(work, tmp_path):
    (work / "SM_link").symlink_to(tmp_path / "precious")
    with pytest.raises(ValueError):
        W.remove("SM_link")
    assert (tmp_path / "precious" / "file").exists()


def test_remove_still_removes_a_real_round_directory(work):
    (work / "SM_mine" / "home").mkdir(parents=True)
    W.remove("SM_mine")
    assert not (work / "SM_mine").exists()
    assert (work / "SM_other").is_dir()


@pytest.mark.parametrize("rid", ["", "..", "../lloyd", "/abs"])
def test_the_abort_tool_refuses_before_anything_is_computed(work, rid):
    refusal = AM._abort_refusal(rid)
    assert refusal and "not a round id" in refusal["error"]
    assert "nothing was removed" in refusal["error"]
