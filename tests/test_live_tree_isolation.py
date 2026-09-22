"""The suite may not address the production tree, and the gate must redirect it.

2026-09-22: an implement round's gate `tests` rung failed with 24 errors, the
model re-ran the full suite with rootdir `/home/alansrobotlab/lloyd` to decide
whether the failures were pre-existing, and at 11:18:50 something under that run
removed the tree — `.git`, `.venvs`, `qmd`, `web/node_modules`, the model
weights, every tracked file, ~35 seconds. Two independent holes made it
reachable and this file pins both shut:

  * the suite would run at all with production as its own tree, and
  * `gate._child_env` handed every child the real `HOME`, so even inside the
    gate `Path.home()/"lloyd"` was production — although `worktree.py` lays a
    round out as `<round>/home/lloyd` for the sole purpose of making
    `HOME=<round>/home` redirect that name.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.automod import gate as G
from scripts.automod import worktree as W

ROOT = Path(__file__).resolve().parent.parent


def _conftest():
    """The real conftest, loaded under its own name so patching it is local."""
    spec = importlib.util.spec_from_file_location(
        "lloyd_conftest_under_test", ROOT / "tests" / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── the suite refuses production ───────────────────────────────────────────

def _production_is(c, monkeypatch, path: Path) -> None:
    monkeypatch.setattr(c, "_production_tree", lambda: path.resolve())


def test_the_suite_refuses_to_run_against_the_production_tree(monkeypatch, tmp_path):
    """The refusal is on the ANCHOR, not on how much of the suite was selected:
    one file is the same fixtures with the same teardowns."""
    c = _conftest()
    prod = tmp_path / "home" / "lloyd"
    prod.mkdir(parents=True)
    _production_is(c, monkeypatch, prod)
    monkeypatch.setattr(c, "ROOT", prod)
    monkeypatch.delenv(c.LIVE_TREE_OPT_IN, raising=False)
    with pytest.raises(pytest.UsageError) as exc:
        c._refuse_the_production_tree()
    msg = str(exc.value)
    assert "worktree add" in msg, "the refusal must name the supported route"
    assert c.LIVE_TREE_OPT_IN in msg, "a guard with an undocumented exit gets deleted"


def test_a_round_worktree_is_never_refused(monkeypatch, tmp_path):
    """The case that would have taken the loop down. The gate now runs the suite
    with `HOME=<round>/home`, so under it `Path.home()/"lloyd"` IS the worktree —
    a guard reading production through `Path.home()` fires on every gate run and
    on nothing else. If this test ever fails, the loop cannot run its tests."""
    c = _conftest()
    prod = tmp_path / "real" / "lloyd"
    prod.mkdir(parents=True)
    round_home = tmp_path / "work" / "SM_1" / "home"
    (round_home / "lloyd").mkdir(parents=True)
    _production_is(c, monkeypatch, prod)
    monkeypatch.setattr(c, "ROOT", round_home / "lloyd")
    monkeypatch.setenv("HOME", str(round_home))       # what the gate now sets
    monkeypatch.delenv(c.LIVE_TREE_OPT_IN, raising=False)
    assert Path.home() / "lloyd" == round_home / "lloyd", "premise of the trap"
    c._refuse_the_production_tree()  # no raise


def test_production_is_read_off_the_passwd_entry_not_the_environment(monkeypatch, tmp_path):
    """`$HOME` is exactly what the gate moves, so it cannot be what names the
    directory this guard protects."""
    c = _conftest()
    monkeypatch.setenv("HOME", str(tmp_path / "somewhere-else"))
    import pwd, os as _os
    assert c._production_tree() == (
        Path(pwd.getpwuid(_os.getuid()).pw_dir) / "lloyd").resolve()


def test_the_human_opt_in_is_honoured(monkeypatch, tmp_path):
    c = _conftest()
    prod = tmp_path / "home" / "lloyd"
    prod.mkdir(parents=True)
    _production_is(c, monkeypatch, prod)
    monkeypatch.setattr(c, "ROOT", prod)
    monkeypatch.setenv(c.LIVE_TREE_OPT_IN, "1")
    c._refuse_the_production_tree()  # no raise


def test_the_refusal_is_wired_at_conftest_import():
    """A guard defined and never called is the shape of the bug it fixes —
    `_is_private_host` sat uncalled in agent_mcp/browser.py for five months."""
    src = (ROOT / "tests" / "conftest.py").read_text()
    assert "\n_refuse_the_production_tree()\n" in src, \
        "the guard must run at conftest import, not merely be defined"


def test_the_refusal_fires_in_a_real_pytest_run(tmp_path):
    """End to end in a child process, against the REAL production path: no test
    may reach into `~/lloyd`, so the run is staged elsewhere and only the
    conftest's own anchor is aimed there."""
    import pwd
    production = Path(pwd.getpwuid(os.getuid()).pw_dir) / "lloyd"
    stage = tmp_path / "stage"
    (stage / "tests").mkdir(parents=True)
    src = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    # ROOT is the checkout under test; pin it at production so the child asks
    # the same question the incident asked, without touching that directory.
    src = src.replace('ROOT = Path(__file__).resolve().parent.parent',
                      f'ROOT = Path({str(production)!r})')
    assert 'ROOT = Path(' in src
    (stage / "tests" / "conftest.py").write_text(src, encoding="utf-8")
    (stage / "tests" / "test_x.py").write_text(
        "def test_one():\n    assert True\n", encoding="utf-8")
    env = {**os.environ}
    env.pop("LLOYD_ALLOW_LIVE_TREE_TESTS", None)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_x.py"],
                       cwd=stage, env=env, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "production tree" in out, out
    assert " 1 passed" not in out, "a refused run must not have executed anything"



# ── the gate redirects HOME for candidate code ─────────────────────────────

def _round(tmp_path, monkeypatch, round_id="SM_HOME"):
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "lloyd-work")
    wt = W.worktree_path(round_id)
    wt.mkdir(parents=True)
    return round_id, wt


def test_round_home_makes_path_home_lloyd_the_worktree(tmp_path, monkeypatch):
    rid, wt = _round(tmp_path, monkeypatch)
    real_home = tmp_path / "real"
    (real_home / "obsidian" / "backlog").mkdir(parents=True)
    (real_home / "lloyd").mkdir()            # production — must NOT be linked
    (real_home / ".gitconfig").write_text("[user]\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: real_home))

    home = W.ensure_round_home(rid)

    assert (home / "lloyd").resolve() == wt.resolve(), \
        "the whole point: Path.home()/'lloyd' is the candidate, not production"
    assert not (home / "lloyd").is_symlink(), \
        "app/paths.py calls .resolve(); a symlink here lands back on production"
    assert (home / "obsidian" / "backlog").is_dir(), "the vault is genuinely shared"
    assert (home / ".gitconfig").read_text() == "[user]\n"


def test_a_round_home_leaves_the_real_home_entries_alone(tmp_path, monkeypatch):
    """Symlinks, not copies — and `shutil.rmtree` refuses a symlinked directory,
    so the farm that keeps the vault readable also makes a teardown aimed at
    `Path.home()/"obsidian"` raise instead of run."""
    import shutil
    rid, _ = _round(tmp_path, monkeypatch)
    real_home = tmp_path / "real"
    (real_home / "obsidian" / "note").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: real_home))
    home = W.ensure_round_home(rid)

    assert (home / "obsidian").is_symlink()
    with pytest.raises(OSError):
        shutil.rmtree(home / "obsidian")
    assert (real_home / "obsidian" / "note").is_dir(), "the vault survived"


def test_removing_a_round_takes_the_links_and_not_their_targets(tmp_path, monkeypatch):
    """`W.remove` ends in `shutil.rmtree(round_dir)`, and the round dir now holds
    a farm of symlinks into the real home — the vault among them.

    `rmtree` unlinks a symlink rather than descending it, so this is safe by
    construction. Pinned anyway: the cost of that being wrong, or of some future
    cleanup switching to a walker that follows links, is the vault and the
    machine, and it would be discovered by losing them.
    """
    import shutil
    rid, _ = _round(tmp_path, monkeypatch, "SM_CLEANUP")
    real_home = tmp_path / "real"
    (real_home / "obsidian" / "backlog").mkdir(parents=True)
    (real_home / ".cache" / "thing").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: real_home))
    W.ensure_round_home(rid)

    shutil.rmtree(W.round_dir(rid))

    assert not W.round_dir(rid).exists()
    assert (real_home / "obsidian" / "backlog").is_dir(), "the vault survived"
    assert (real_home / ".cache" / "thing").is_dir()


def test_ensure_round_home_refuses_rather_than_pretending(tmp_path, monkeypatch):
    """No worktree means no redirection, and the caller must be told so: its
    fallback is the real home, which is the outcome this exists to prevent."""
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "lloyd-work")
    with pytest.raises(RuntimeError, match="no worktree"):
        W.ensure_round_home("SM_MISSING")


def test_the_tests_rung_asks_for_the_isolated_home():
    """Every rung that executes CANDIDATE test code, and no others — the
    tool-choice rungs deliberately run from the live tree."""
    src = (ROOT / "scripts" / "automod" / "gate.py").read_text()
    assert src.count("isolate_home=True") == 3, \
        "expected the suite, the base probe and the flake re-confirm"
    suite = src.split("base_cmd = [str(self.python)", 1)[1][:400]
    assert "self._child_env(isolate_home=True)" in suite, \
        "the tests rung is the one that runs the candidate's own fixtures"


def test_the_child_env_points_home_at_the_round(tmp_path, monkeypatch):
    rid, wt = _round(tmp_path, monkeypatch, "SM_ENV")
    real_home = tmp_path / "real"
    real_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: real_home))
    g = G.Gate(rid, wt, "HEAD", live_root=wt)

    assert g._child_env()["HOME"] == str(real_home), \
        "unchanged for the rungs that run this repo's own scripts"
    env = g._child_env(isolate_home=True)
    assert env["HOME"] == str(W.round_home(rid))
    assert (Path(env["HOME"]) / "lloyd").resolve() == wt.resolve()
    assert "round home" in g.home_isolation


def test_a_home_that_cannot_be_built_falls_back_and_says_so(tmp_path, monkeypatch):
    """Fails open, because a gate that cannot run its tests judges nothing — but
    a silent downgrade here is indistinguishable from the isolation working, so
    the reading rides onto the rung's data either way."""
    rid, wt = _round(tmp_path, monkeypatch, "SM_FALLBACK")
    real_home = tmp_path / "real"
    real_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: real_home))
    monkeypatch.setattr(W, "ensure_round_home",
                        lambda _rid: (_ for _ in ()).throw(RuntimeError("no worktree")))
    g = G.Gate(rid, wt, "HEAD", live_root=wt)

    env = g._child_env(isolate_home=True)
    assert env["HOME"] == str(real_home)
    assert "FELL BACK" in g.home_isolation and "no worktree" in g.home_isolation

    src = (ROOT / "scripts" / "automod" / "gate.py").read_text()
    assert '"home": self.home_isolation' in src and 'counts["home"] = self.home_isolation' in src, \
        "the reading must reach the rung's data on the pass and on the failure"


# ── the prompt ─────────────────────────────────────────────────────────────

def test_the_implement_prompt_forbids_the_live_tree():
    from workers.sources import autocode
    p = autocode.PROMPT
    assert "never from `~/lloyd`" in p, \
        "the rule the 2026-09-22 round did not have"
    assert "base_probe" in p, \
        "name the answer the round already has, or it goes looking for its own"
