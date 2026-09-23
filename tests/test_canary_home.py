"""The canary's HOME is not the test rungs' HOME.

On 2026-09-22 the gate began running candidate tests under `<round>/home` as a
symlink farm over the real home, `obsidian` included. `materialize_home` built the
canary's scratch vault in that same directory: its `mkdir`s went through the link
into the live vault, and its SOUL.md copy onto itself raised `SameFileError` at
canary_boot for every round from 01:56Z. These build that exact layout in a
tmp dir standing in for both the live tree and the round.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.automod import canary_config as cc


@pytest.fixture
def layout(tmp_path):
    """A fake account (`live/lloyd`, `live/obsidian`) and a round whose test home
    is the gate's farm: `<round>/home/lloyd` is the worktree, `obsidian` a link."""
    live = tmp_path / "acct"
    (live / "lloyd").mkdir(parents=True)
    (live / "lloyd" / ".env").write_text("X=1\n")
    (live / "obsidian" / "lloyd").mkdir(parents=True)
    (live / "obsidian" / "lloyd" / "SOUL.md").write_text("soul\n")
    round_dir = tmp_path / "SM_X"
    worktree = round_dir / "home" / "lloyd"
    worktree.mkdir(parents=True)
    (round_dir / "home" / "obsidian").symlink_to(live / "obsidian")
    return live, round_dir, worktree


def _tree(p: Path) -> list[str]:
    return sorted(str(x.relative_to(p)) for x in p.rglob("*"))


def test_the_canary_builds_beside_the_test_farm_and_leaves_the_live_vault_alone(layout):
    live, round_dir, worktree = layout
    before = _tree(live / "obsidian")
    home = cc.materialize_home(round_dir, worktree, live_root=live / "lloyd")
    assert home == round_dir / "canary-home"
    assert _tree(live / "obsidian") == before, "the canary wrote into the live vault"
    vault = home / "obsidian"
    assert vault.is_dir() and not vault.is_symlink()
    assert (vault / "lloyd" / "SOUL.md").read_text() == "soul\n"
    assert (vault / "autonomy").is_dir() and not any((vault / "autonomy").iterdir())
    assert (home / "lloyd").resolve() == worktree.resolve()
    # The test rungs' farm is untouched: its obsidian still reaches the real vault.
    assert (round_dir / "home" / "obsidian").resolve() == (live / "obsidian").resolve()


def test_materializing_twice_is_a_no_op(layout):
    live, round_dir, worktree = layout
    cc.materialize_home(round_dir, worktree, live_root=live / "lloyd")
    cc.materialize_home(round_dir, worktree, live_root=live / "lloyd")   # a re-gate


def test_a_linked_canary_vault_is_refused_rather_than_written_through(layout):
    live, round_dir, worktree = layout
    (round_dir / "canary-home").mkdir()
    (round_dir / "canary-home" / "obsidian").symlink_to(live / "obsidian")
    with pytest.raises(RuntimeError, match="symlink"):
        cc.materialize_home(round_dir, worktree, live_root=live / "lloyd")


def test_the_canary_env_names_the_home_it_was_built_in(layout, tmp_path):
    live, round_dir, worktree = layout
    home = cc.materialize_home(round_dir, worktree, live_root=live / "lloyd")
    env = cc.canary_env(round_dir, worktree, overlay=tmp_path / "o.yaml",
                        python=Path("/usr/bin/python3"))
    assert env["HOME"] == str(home)
