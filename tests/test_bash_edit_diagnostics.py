"""A `.py` rewritten through Bash carries the same pyflakes delta as `Edit` (#695).

Until this, `sed -i`, `patch`, a redirect, a `Path(...).write_text` heredoc or
a formatter could introduce a finding in the tree and the Bash result said
nothing, so an automod implementer learned of it at the gate. The rail lives in
`agent_mcp/_bash_edit_diagnostics.py`, called from `builtin_bash._bash`.

Every scenario runs the real shell inside a real `git init` tree under
`tmp_path`; "outside the tree" is a sibling directory with no `.git`.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_mcp import _bash_edit_diagnostics as R  # noqa: E402
from agent_mcp import _edit_diagnostics as D  # noqa: E402
from agent_mcp import builtin_bash as B  # noqa: E402

CLEAN = "def f():\n    return 1\n"
TOLERATED = "import os\n\n\ndef f():\n    return 1\n"


@pytest.fixture(autouse=True)
def _default_config(monkeypatch):
    monkeypatch.setattr(D, "config", lambda: {
        "python": True, "typescript": True, "blast_radius": True,
        "max_lines": D.DEFAULT_MAX_LINES})


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def bash(command: str, cwd: Path, **extra) -> str:
    return asyncio.run(B._bash({"command": command, "cwd": str(cwd), **extra}))


def blocks(out: str) -> list[str]:
    return [b.split("</diagnostics>")[0] for b in out.split("<diagnostics ")[1:]]


# -- clause 1: each write shape reports what it introduced ------------------

def test_sed_in_place_reports_the_new_finding_with_the_absolute_path(repo):
    f = repo / "pkg" / "mod.py"
    f.write_text(CLEAN)
    out = bash("sed -i '1i import sys' pkg/mod.py", repo)
    [b] = blocks(out)
    assert b.startswith(f'file="{f}" tool="pyflakes" new="1"')
    assert "'sys' imported but unused" in b


def test_a_redirect_that_creates_a_file_reports_it(repo):
    out = bash("printf 'import json\\n' > pkg/new.py", repo)
    [b] = blocks(out)
    assert f'file="{repo / "pkg" / "new.py"}"' in b and "'json'" in b


def test_a_python_heredoc_write_text_reports_it(repo):
    f = repo / "pkg" / "mod.py"
    f.write_text(CLEAN)
    cmd = (f"{sys.executable} - <<'EOF'\n"
           "from pathlib import Path\n"
           "p = Path('pkg/mod.py')\n"
           "p.write_text('import re\\n' + p.read_text())\n"
           "EOF")
    [b] = blocks(bash(cmd, repo))
    assert "'re' imported but unused" in b


def test_git_apply_and_patch_report_the_files_their_diff_names(repo):
    f = repo / "pkg" / "mod.py"
    f.write_text(CLEAN)
    diff = ("--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,2 +1,3 @@\n"
            "+import abc\n def f():\n     return 1\n")
    out = bash(f"git apply <<'EOF'\n{diff}EOF", repo)
    assert "'abc' imported but unused" in "".join(blocks(out)), out

    f.write_text(CLEAN)
    (repo / "fix.diff").write_text(diff.replace("abc", "enum"))
    out = bash("patch -s -p1 < fix.diff", repo)
    assert "'enum' imported but unused" in "".join(blocks(out)), out


@pytest.mark.parametrize("tool,argv", [("black", "black"), ("ruff", "ruff check --fix")])
def test_a_formatter_run_on_a_named_file_reports_it(repo, tmp_path, monkeypatch, tool, argv):
    # Neither formatter is installed here; a stand-in that writes its last
    # argument is what the rail observes either way.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / tool
    fake.write_text('#!/bin/sh\nfor a; do t="$a"; done\nprintf "import glob\\n" >> "$t"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    (repo / "pkg" / "mod.py").write_text(CLEAN)
    [b] = blocks(bash(f"{argv} pkg/mod.py", repo))
    assert "'glob' imported but unused" in b


def test_a_leading_cd_resolves_relative_targets(repo, tmp_path):
    (repo / "pkg" / "mod.py").write_text(CLEAN)
    out = bash(f"cd {repo} && sed -i '1i import sys' pkg/mod.py", tmp_path)
    assert len(blocks(out)) == 1


# -- clause 2: a delta, never absolute -------------------------------------

def test_only_the_new_finding_is_reported_not_the_tolerated_one(repo):
    (repo / "pkg" / "mod.py").write_text(TOLERATED)
    [b] = blocks(bash("sed -i '1a import sys' pkg/mod.py", repo))
    assert 'new="1"' in b and "'sys'" in b and "'os'" not in b


def test_a_write_that_adds_no_finding_appends_no_block(repo):
    (repo / "pkg" / "mod.py").write_text(TOLERATED)
    out = bash("sed -i '1i # a comment' pkg/mod.py", repo)
    assert out == "(no output)"


# -- clause 3: reads, and writes outside the tree, are silent --------------

@pytest.mark.parametrize("cmd", [
    "grep -n import pkg/mod.py",
    "cat pkg/mod.py",
    f"{sys.executable} - <<'EOF'\nfrom pathlib import Path\n"
    "Path('/dev/null').write_text(Path('pkg/mod.py').read_text())\nEOF",
])
def test_a_command_that_only_reads_a_tree_file_appends_nothing(repo, cmd):
    (repo / "pkg" / "mod.py").write_text(TOLERATED)
    assert "<diagnostics" not in bash(cmd, repo)


@pytest.mark.parametrize("where", ["outside", "repo/_pipeline", "repo/sessions"])
def test_a_write_outside_the_tree_appends_nothing(repo, tmp_path, where):
    d = tmp_path / where
    d.mkdir(parents=True, exist_ok=True)
    (d / "scratch.py").write_text(CLEAN)
    out = bash(f"sed -i '1i import sys' {d / 'scratch.py'}", tmp_path)
    assert "import sys" in (d / "scratch.py").read_text()
    assert "<diagnostics" not in out


# -- clause 4: the rail cannot change the command's result -----------------

def _run_twice(repo, monkeypatch, breaker) -> tuple[str, str]:
    f = repo / "pkg" / "mod.py"
    cmd = "sed -i '1i import sys' pkg/mod.py; echo done; exit 3"
    f.write_text(CLEAN)
    with monkeypatch.context() as m:
        m.setattr(R, "snapshot", lambda *a, **k: None)  # the rail absent
        baseline = bash(cmd, repo)
    f.write_text(CLEAN)
    with monkeypatch.context() as m:
        breaker(m)
        broken = bash(cmd, repo)
    return baseline, broken


def _boom(*a, **k):
    raise RuntimeError("rail exploded")


@pytest.mark.parametrize("target", ["python_block", "config"])
def test_a_raising_diagnostics_helper_leaves_the_result_byte_identical(repo, monkeypatch, target):
    baseline, broken = _run_twice(
        repo, monkeypatch, lambda m: m.setattr(D, target, _boom))
    assert baseline == "done\n\n[exit code: 3]"
    assert broken == baseline


@pytest.mark.parametrize("target", ["snapshot", "append"])
def test_a_raising_rail_entry_point_leaves_the_result_byte_identical(repo, monkeypatch, target):
    baseline, broken = _run_twice(
        repo, monkeypatch, lambda m: m.setattr(R, target, _boom))
    assert broken == baseline


def test_a_background_call_returns_only_its_task_handle(repo, monkeypatch):
    seen = []
    monkeypatch.setattr(R, "snapshot", lambda *a, **k: seen.append(a))
    (repo / "pkg" / "mod.py").write_text(CLEAN)
    out = bash("sed -i '1i import sys' pkg/mod.py", repo, run_in_background=True)
    assert "<diagnostics" not in out
    payload = json.loads(out)
    assert set(payload) == {"task_id", "output_file", "started_at", "session_id", "note"}
    assert seen == []
