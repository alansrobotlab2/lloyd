"""`worktree.changed_paths` is what the scope fence, the drill decision and the
removed-tests check read, so a rename has to appear as what it is: one path gone
and one path new.

With git's rename detection on, `git mv config.yaml app/old.yaml` listed only
`app/old.yaml` — the denied source never reached `spec.classify`, a protected file
could be renamed to an allowed name with no drill, and a failing test moved out of
`tests/` was not a removed test. `vet.py` already passed `--no-renames` for this
reason; these nodes pin the same for the list the gate judges.
"""

from __future__ import annotations

import subprocess

import pytest

from scripts.automod import worktree as W


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "app").mkdir(parents=True)
    (r / "tests").mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True)
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    body = "".join(f"key_{i}: {i}\n" for i in range(40))
    (r / "config.yaml").write_text(body, encoding="utf-8")
    (r / "tests" / "test_x.py").write_text(
        "".join(f"def test_{i}():\n    assert {i} == {i}\n" for i in range(20)),
        encoding="utf-8")
    (r / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r


def _base(repo):
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def test_a_pure_rename_lists_the_source_as_well_as_the_destination(repo):
    base = _base(repo)
    git(repo, "mv", "config.yaml", "app/old_config.yaml")
    git(repo, "commit", "-q", "-m", "move it")
    # The control: this is the list the fence used to be handed.
    detected = git(repo, "diff", "--name-only", f"{base}...HEAD").stdout.split()
    assert detected == ["app/old_config.yaml"]
    assert W.changed_paths(repo, base) == ["app/old_config.yaml", "config.yaml"]


def test_a_test_file_moved_out_of_tests_is_a_changed_test_path(repo):
    base = _base(repo)
    git(repo, "mv", "tests/test_x.py", "app/old_x.py")
    git(repo, "commit", "-q", "-m", "retire a test by moving it")
    assert "tests/test_x.py" in W.changed_paths(repo, base)


def test_an_ordinary_edit_is_listed_exactly_as_before(repo):
    base = _base(repo)
    (repo / "app" / "m.py").write_text("V = 2\n", encoding="utf-8")
    git(repo, "commit", "-qam", "edit")
    assert W.changed_paths(repo, base) == ["app/m.py"]
