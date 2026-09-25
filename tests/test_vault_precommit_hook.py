"""The vault pre-commit hook refuses an off-set knowledge/ `type` (#1512).

Driven through real git in a scratch repo with `core.hooksPath` pointed at the
tracked hook, which is how production installs it — so what is pinned is the
refusal a plain `git commit` meets, not a function call.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "scripts" / "vault" / "hooks"

pytestmark = pytest.mark.skipif(not Path("/usr/bin/python3").exists() or shutil.which("git") is None,
                                reason="needs /usr/bin/python3 and git")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env)


@pytest.fixture
def vault(tmp_path):
    repo = tmp_path / "vault"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "core.hooksPath", str(HOOKS))
    return repo


def _stage(repo: Path, rel: str, type_value: str | None) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    fm = f"---\ntitle: x\ntype: {type_value}\n---\n" if type_value is not None else "---\ntitle: x\n---\n"
    p.write_text(fm + "body\n")
    _git(repo, "add", rel)


def test_the_hook_is_executable():
    assert os.access(HOOKS / "pre-commit", os.X_OK)


def test_a_retired_spelling_in_knowledge_is_refused_by_name(vault):
    _stage(vault, "knowledge/software/a.md", "note")
    r = _git(vault, "commit", "-qm", "x")
    assert r.returncode != 0
    assert "knowledge/software/a.md" in r.stderr and "'note'" in r.stderr and "`notes`" in r.stderr
    assert _git(vault, "rev-parse", "--verify", "HEAD").returncode != 0  # nothing committed


def test_a_canonical_type_commits(vault):
    _stage(vault, "knowledge/software/a.md", "notes")
    assert _git(vault, "commit", "-qm", "x").returncode == 0


def test_outside_knowledge_and_absent_types_are_not_its_business(vault):
    _stage(vault, "projects/a.md", "note")
    _stage(vault, "knowledge/b.md", None)
    assert _git(vault, "commit", "-qm", "x").returncode == 0


def test_the_staged_bytes_are_judged_not_the_working_tree(vault):
    _stage(vault, "knowledge/a.md", "note")
    (vault / "knowledge/a.md").write_text("---\ntype: notes\n---\nfixed on disk, not staged\n")
    assert _git(vault, "commit", "-qm", "x").returncode != 0
