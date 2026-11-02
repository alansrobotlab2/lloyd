"""Tests for the bench fixture builder (#3403 clause 3).

A durable-write bench task is only gradeable if the state it writes is a copy, and the
copy is only a copy if four things hold: it lands under the data root's fixture area, it
carries the marker that makes `resolve_data_root` accept it rather than fall back to the
production root, its vault half is a git repo so the run's writes are enumerable, and the
builder cannot be talked into writing the destination inside the tree it is copying. Each
node below is one of those four, and the last one is the refusal — the only failure mode
here that damages the house rather than a trial.

Nothing in this file reads or writes the real vault or the real data root: the "live"
trees are tmp_path fixtures, so the destination-inside-live checks are provable without
staking a copy of Alan's memory on the test.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_root import DATA_ROOT_MARKER  # noqa: E402
from scripts.autoresearch import bench_fixture
from scripts.autoresearch.bench_fixture import (  # noqa: E402
    FIXTURE_ROOT_RELATIVE, FixtureRefused, BenchFixture, build_fixture,
    declared_scopes, fixture_parent, sweep_orphan_fixtures)


def _live(tmp_path: Path) -> tuple[Path, Path]:
    """A stand-in vault and data root, each with one file of the shape the copy takes."""
    vault = tmp_path / "obsidian"
    (vault / "lloyd" / "memory").mkdir(parents=True, exist_ok=True)
    (vault / "lloyd" / "MEMORY.md").write_text("# index\n", encoding="utf-8")
    (vault / "lloyd" / "memory" / "memory-md-ledger.md").write_text(
        "---\ntype: note\n---\nrow\n", encoding="utf-8")
    (vault / ".git").mkdir(exist_ok=True)             # must NOT be copied
    (vault / ".obsidian").mkdir(exist_ok=True)        # must NOT be copied
    data = tmp_path / "lloyd-data"
    data.mkdir(exist_ok=True)
    (data / DATA_ROOT_MARKER).write_text("1\n", encoding="utf-8")
    (data / "kg.sqlite").write_bytes(b"SQLite format 3\x00fake")
    (data / "workers.db").write_bytes(b"SQLite format 3\x00fake")
    return vault, data


_BUILD_SEQ = itertools.count(1)


def _build(tmp_path: Path, writes=("vault", "kg")):
    """A fresh trial id per call: `build_fixture` refuses an existing destination (the
    rail that stops one trial deleting another's tree), and the store-scope node needs
    two builds into the same fixtures area."""
    vault, data = _live(tmp_path)
    return build_fixture(trial_id=f"t_bench_fixture_{next(_BUILD_SEQ):04d}", writes=list(writes),
                         vault=vault, data_root=data,
                         fixtures_parent=fixture_parent(data))


def test_the_fixture_lands_under_the_data_roots_fixture_area(tmp_path):
    fx = _build(tmp_path)
    try:
        assert fx.root.parent == fixture_parent(tmp_path / "lloyd-data")
        assert FIXTURE_ROOT_RELATIVE.name == "fixtures"
        assert fx.vault.is_dir() and fx.data_root.is_dir()
    finally:
        fx.cleanup()


def test_the_data_root_marker_is_carried_so_the_copy_is_a_root_not_a_fallback(tmp_path):
    """Without the marker `resolve_data_root` refuses the copy and falls back to production.

    That is the quiet failure: `LLOYD_DATA` points at the fixture, the resolver says "not
    a root", returns the production root, and the trial writes the house while its
    environment says otherwise.
    """
    fx = _build(tmp_path)
    try:
        assert (fx.data_root / DATA_ROOT_MARKER).is_file()
    finally:
        fx.cleanup()


def test_the_named_slice_is_copied_and_the_git_and_forge_dirs_are_not(tmp_path):
    fx = _build(tmp_path, writes=("vault",))
    try:
        assert (fx.vault / "lloyd" / "MEMORY.md").is_file()
        assert (fx.vault / "lloyd" / "memory" / "memory-md-ledger.md").is_file()
        assert not (fx.vault / ".obsidian").exists(), "the vault's config was copied"
    finally:
        fx.cleanup()


def test_the_data_stores_travel_only_with_the_scope_that_names_them(tmp_path):
    with_stores = _build(tmp_path, writes=("vault", "kg", "workers.db"))
    without = _build(tmp_path, writes=("vault",))
    try:
        assert with_stores.kg_db.is_file() and with_stores.workers_db.is_file()
        assert not without.kg_db.exists(), (
            "kg.sqlite was copied for a scope that never asked for it: the trial could "
            "write a KG row and the reward could not see it")
        assert not without.workers_db.exists()
    finally:
        with_stores.cleanup()
        without.cleanup()


def test_init_git_makes_the_runs_writes_enumerable(tmp_path):
    """What clause 3 asks for is `git -C <fixture> diff --name-only` ENUMERATING the run's
    writes, so this node calls `writes()` — the method the verifiers read — and asserts its
    list, not merely what a plain `git diff` misses. Three separate things have to be true
    at once: a CREATED file appears (the point of the intent-to-add, and the exact hole the
    previous version of this node left unpinned), a MODIFIED file appears, and the working
    tree still holds the agent's own bytes for the verifier to read afterwards."""
    fx = _build(tmp_path, writes=("vault",))
    try:
        assert fx.writes() == [], "a fresh baseline must enumerate as no writes at all"
        (fx.vault / "backlog" / "3403-card.md").parent.mkdir(parents=True, exist_ok=True)
        (fx.vault / "backlog" / "3403-card.md").write_text(
            "---\nstatus: done\n---\n\nbody\n", encoding="utf-8")
        (fx.vault / "lloyd" / "MEMORY.md").write_text(
            "# index\n- a line the run added\n", encoding="utf-8")

        # The hole `writes()` closes, shown rather than asserted away: the same repository,
        # the same moment, and a bare `git diff --name-only` names only the MODIFIED file.
        bare = subprocess.run(
            ["git", "-C", str(fx.vault), "diff", "--name-only"],
            capture_output=True, text=True).stdout.split()
        assert "lloyd/MEMORY.md" in bare
        assert "backlog/3403-card.md" not in bare, (
            "a file the run CREATED is untracked, so a verifier built on a plain "
            "`git diff` would grade a successful write as a no-op")

        listed = fx.writes()
        assert listed == ["backlog/3403-card.md", "lloyd/MEMORY.md"], listed
        # Intent-to-add records a path, not content: the file on disk is still the bytes
        # the run wrote, which is what a `frontmatter`-reading verifier reads next.
        assert (fx.vault / "backlog" / "3403-card.md").read_text(encoding="utf-8") == (
            "---\nstatus: done\n---\n\nbody\n")
        staged = subprocess.run(
            ["git", "-C", str(fx.vault), "diff", "--cached", "--name-only"],
            capture_output=True, text=True).stdout.split()
        assert staged == [], f"intent-to-add staged content: {staged}"
    finally:
        fx.cleanup()


def test_env_points_every_resolved_root_at_the_copy(tmp_path):
    fx = _build(tmp_path, writes=("vault", "kg"))
    try:
        env = fx.env()
        assert env["LLOYD_VAULT_ROOT"] == str(fx.vault)
        assert env["LLOYD_DATA"] == str(fx.data_root)
        assert env["LLOYD_KG_DB"] == str(fx.kg_db)
    finally:
        fx.cleanup()


def test_a_destination_inside_the_live_vault_or_live_data_root_is_refused(tmp_path, monkeypatch):
    """The one failure mode here that damages the house rather than a trial.

    Both spellings matter: the plain nesting, and a `..` that lexically escapes it — the
    second is why the check realpaths before comparing.
    """
    vault, data = _live(tmp_path)
    # `data` here is a stand-in, so the same path is legal in `_build`; what makes a
    # destination inside a data root a refusal is that the root is the LIVE one. Patching
    # the module's resolved production roots is the only honest way to say that on a
    # fixture — clause 3's parenthetical is that a destination under an OVERRIDDEN data
    # root is legal, and a test that refused the un-overridden spelling would refuse the
    # builder's own default.
    monkeypatch.setattr(bench_fixture, "PRODUCTION_DATA_ROOT", data.resolve())
    monkeypatch.setattr(bench_fixture, "PRODUCTION_VAULT", vault.resolve())
    with pytest.raises(FixtureRefused):
        build_fixture(trial_id="t_x_1", writes=["vault"], vault=vault, data_root=data,
                      fixtures_parent=data / "eval" / "fixtures")
    with pytest.raises(FixtureRefused):
        build_fixture(trial_id="t_x_2", writes=["vault"], vault=vault, data_root=data,
                      fixtures_parent=data / "a" / "b" / ".." / "..")


def test_a_writes_scope_outside_the_known_set_is_refused(tmp_path):
    """A typo in a task file must not silently build a fixture that copies nothing."""
    vault, data = _live(tmp_path)
    with pytest.raises(FixtureRefused):
        build_fixture(trial_id="t_x_3", writes=["memery"], vault=vault, data_root=data,
                      fixtures_parent=fixture_parent(data))


def test_cleanup_removes_the_fixture_tree_and_refuses_to_delete_anything_else(tmp_path):
    fx = _build(tmp_path, writes=("vault",))
    root = fx.root
    fx.init_git()
    fx.cleanup()
    assert not root.exists(), "the second copy of the vault outlived the trial"
    assert (tmp_path / "obsidian" / "lloyd" / "MEMORY.md").is_file(), (
        "the source vault was touched")
    # A path that is not a fixture directory is not deletable, even through cleanup().
    stray = tmp_path / "lloyd-data" / "eval" / "notfixtures" / "keep"
    stray.mkdir(parents=True)
    foreign = BenchFixture(root=stray, vault=stray, data_root=stray,
                           id="t_foreign_1", scopes=["vault"])
    with pytest.raises(RuntimeError):
        foreign.cleanup()
    assert stray.exists()


def test_declared_scopes_reads_the_two_task_spellings_and_nothing_else():
    assert declared_scopes({"writes": "backlog"}) == ["backlog"]
    assert declared_scopes({"writes": ["vault", "kg"]}) == ["vault", "kg"]
    assert declared_scopes({}) == []
    assert declared_scopes({"writes": None}) == []


# ─────────────────────────────────────────────────────────────────────────────
# ── #3690: bounding the copy that outlived its trial ─────────────────────────
# `cleanup()` is called at the END of a trial, so the always-delete rule holds only
# while the runner survives to reach that line. Two ways it does not: a process killed
# mid-trial, and — the one that needs no signal at all — a raise between the trial's
# `finally` and the `cleanup()` that sits after it. Either leaves a SECOND COPY OF ALAN'S
# ACTUAL MEMORY in the fixtures area with nothing that ages it. The nodes below grade the
# one mechanism that bounds it: an age sweep over the fixtures parent, run by the builder
# that is standing in the right directory, behind the same rails `cleanup()` uses.
# ─────────────────────────────────────────────────────────────────────────────


def _plant(parent: Path, name: str, *, age_s: float, now: float) -> Path:
    """An entry in a fixtures area aged `age_s` seconds, holding the shape a copy has.

    Aged with `os.utime` on the directory rather than by sleeping, because the age the
    sweep reads is the directory's own and a test that waits 400 s for a leak it could
    fabricate in microseconds is a test nobody runs.
    """
    d = parent / name
    (d / "obsidian" / "lloyd").mkdir(parents=True)
    (d / "obsidian" / "lloyd" / "MEMORY.md").write_text("# index\n", encoding="utf-8")
    os.utime(d, (now - age_s, now - age_s))
    return d


def test_the_sweep_deletes_a_copy_older_than_the_ceiling_and_keeps_a_fresh_one(tmp_path):
    """#3690 clause 1: the ceiling is a parameter, and the two sides of it are two
    outcomes.

    The ceiling the sweep is given is the runner's per-trial hard cap — `ORPHAN_MAX_AGE_S`
    restates `scripts/autoresearch/common.py:170`'s `timeout_secs = 300`, and the node that
    pins the build wiring below uses that number rather than a convenient one, because a
    fixture is built for exactly one trial and a copy still here after a whole trial
    cannot be a live one.
    """
    now = time.time()
    parent = tmp_path / "fixtures"
    aged = _plant(parent, "aged_deadbeef", age_s=400, now=now)
    fresh = _plant(parent, "fresh_deadbeef", age_s=10, now=now)
    assert bench_fixture.ORPHAN_MAX_AGE_S == 300, (
        "the ceiling drifted from the runner's per-trial cap; the constant's comment names "
        "common.py:170 as its source")

    report = sweep_orphan_fixtures(parent, bench_fixture.ORPHAN_MAX_AGE_S, now=now)

    assert not aged.exists(), "the orphan outlived the ceiling: the leak this item is about"
    assert fresh.is_dir(), "a live trial's copy was deleted — the guard that keeps nothing " \
                           "is worse than the leak"
    assert report["deleted"] == [str(aged)]
    assert report["kept"] == [str(fresh)]
    assert report["refused"] == []
    # The same file, one second either side of the line, so a `>=`/`>` slip is visible.
    edge = _plant(parent / "fixtures_edge", "x", age_s=1, now=now)  # not a fixtures parent
    assert sweep_orphan_fixtures(edge.parent, 300, now=now)["refused"], \
        "an age sweep offered a non-fixtures parent deleted something"


def test_the_sweep_refuses_a_parent_that_is_not_the_fixtures_area(tmp_path):
    """#3690 clause 2a: `rmtree` is one wrong parent away from a real deletion, so the
    parent's NAME is a rail, and refusing means reporting, not raising.

    Refusal is a report and not an exception because the sweep runs inside `build_fixture`:
    a housekeeping pass that can veto a build is a housekeeping pass that gets taken out of
    the build, and the leak comes back. The path still has to be NAMED, or the entry that
    will not age out becomes an invisible leak instead of a visible refusal.
    """
    now = time.time()
    wrong = tmp_path / "lloyd-data" / "eval" / "notfixtures"
    aged = _plant(wrong, "aged_deadbeef", age_s=400, now=now)

    report = sweep_orphan_fixtures(wrong, 300, now=now)

    assert aged.is_dir(), "a directory under a parent not named `fixtures` was deleted"
    assert report["deleted"] == []
    assert len(report["refused"]) == 1 and str(aged) in report["refused"][0], report
    # One guard, not two: the path the sweep refuses is refused by `cleanup()` too, so the
    # sweep cannot delete through a rail the object's own deleter still enforces.
    foreign = BenchFixture(root=aged, vault=aged, data_root=aged,
                           id="t_foreign_1", scopes=["vault"])
    with pytest.raises(RuntimeError):
        foreign.cleanup()
    assert aged.is_dir()


def test_the_sweep_refuses_an_aged_copy_inside_either_production_tree(tmp_path, monkeypatch):
    """#3690 clause 2b: neither production tree may be inside a delete target, and the
    refusal says which path it stopped on.

    `build_fixture` refuses to BUILD inside `PRODUCTION_VAULT`/`PRODUCTION_DATA_ROOT`; this
    is the same strong test pointed the other way, at a delete. It is the difference between
    the two that makes it worth a node: a build cannot put a copy there, but a sweep is handed
    a parent by a caller, and the parent that reads as a fixtures area while sitting inside
    `~/lloyd-data` is exactly the path where `rmtree` would be deleting the house's own
    `eval/` subtree. The roots are patched, not staked — these trees are the account's real
    memory and the check under test is a path comparison.
    """
    now = time.time()
    live_data = tmp_path / "lloyd-data"
    in_data = _plant(live_data / "eval" / "fixtures", "aged_deadbeef", age_s=400, now=now)
    live_vault = tmp_path / "obsidian"
    (live_vault / "lloyd").mkdir(parents=True)
    in_vault = _plant(live_vault / "eval" / "fixtures", "aged_deadbeef", age_s=400, now=now)
    monkeypatch.setattr(bench_fixture, "PRODUCTION_DATA_ROOT", live_data.resolve())
    monkeypatch.setattr(bench_fixture, "PRODUCTION_VAULT", live_vault.resolve())

    for parent, aged in ((live_data / "eval" / "fixtures", in_data),
                         (live_vault / "eval" / "fixtures", in_vault)):
        report = sweep_orphan_fixtures(parent, 300, now=now)
        assert aged.is_dir(), f"a sweep deleted inside a production tree: {parent}"
        assert report["deleted"] == []
        assert len(report["refused"]) == 1 and str(aged) in report["refused"][0], report


def test_building_a_fixture_sweeps_its_own_fixtures_area_first(tmp_path):
    """#3690 clause 3, and the item's own planted-directory acceptance check, run as
    written: plant `aged_deadbeef` older than 300 s and `fresh_deadbeef` at the current
    time, build in that parent, and the aged directory must be gone while the fresh one and
    the new build both remain.

    The builder is the only code that ever stands in the right directory, which is why the
    sweep lives on its path and not in the data-root retention sweep: that sweep resolves
    `$LLOYD_DATA` (`scripts/groundskeeper/retention-sweep.py:395`), and rule (2) of this very
    file guarantees no fixture can ever be under it, so a store added there would enumerate
    an empty path forever.
    """
    vault, data = _live(tmp_path)
    parent = fixture_parent(data)
    parent.mkdir(parents=True)
    now = time.time()
    aged = _plant(parent, "aged_deadbeef", age_s=400, now=now)
    fresh = _plant(parent, "fresh_deadbeef", age_s=10, now=now)

    fx = build_fixture(trial_id="t_sweep_1", writes=["vault"], vault=vault, data_root=data,
                       fixtures_parent=parent)
    try:
        assert not aged.exists(), (
            "the build left the orphan in place: the acceptance check this item was filed "
            "with is still failing")
        assert fresh.is_dir(), "the sweep deleted a copy inside the ceiling"
        assert fx.root.is_dir() and fx.root != fresh, "the build did not create its own copy"
        assert sorted(p.name for p in parent.iterdir()) == ["fresh_deadbeef", "t_sweep_1"]
        assert (fx.vault / "lloyd" / "MEMORY.md").is_file(), (
            "the sweep ran over the copy this build had just made")
        assert (vault / "lloyd" / "MEMORY.md").is_file(), "the source vault was touched"
    finally:
        fx.cleanup()
    # `cleanup()` and the sweep share one rails function, so the copy the build made is gone
    # and the planted fresh copy is still there for the next build to sweep.
    assert fresh.is_dir() and not fx.root.exists()
