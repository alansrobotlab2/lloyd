"""The writer-side dead-path rung (#1969).

Two suite nodes read the live vault and go red on main when a skill names a
checkout path that is not there, and three times in three days the writer was a
different job. `scripts/util/skill_path_findings.py` asks the same rule of the
STAGED text inside `vault-commit.sh`, so the committing job reads the finding in
its own output. These nodes pin what makes that worth having: it sees exactly the
shapes that reddened main, it reports only what the edit ADDS, it shares the
suite's rule rather than restating it, and it never blocks the commit.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.util import skill_path_findings as rung

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "util" / "vault-commit.sh"

_SKILL = "skills/nightly-reflection-demo/SKILL.md"
_BASE = ("---\nname: nightly-reflection-demo\n---\n# Demo\n\n"
         "Reads `app/config.py` and an old note about "
         "`eval/a_path_that_was_already_dead_1969.json`.\n")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture
def vault(tmp_path):
    repo = tmp_path / "vault"
    (repo / "skills" / "nightly-reflection-demo").mkdir(parents=True)
    (repo / "autonomy").mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / _SKILL).write_text(_BASE, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _append(repo: Path, text: str, rel: str = _SKILL) -> None:
    path = repo / rel
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")
    _git(repo, "add", "-A")


def test_the_two_shapes_that_reddened_main_are_both_named(vault):
    """The #2026 shape (a backticked repo path quoted in an incident note) and
    the #1969 shape (`~/lloyd/.t`, a dot-directory only the bench rule sees —
    the suite's unmarked rule skips a reference with no slash in it)."""
    _append(vault, "\nOn 2026-09-30 `git add -A` staged "
                   "`eval/uptake/a_report_no_commit_has_1969.json`; "
                   "`ls -d ~/lloyd/.t-never-here-1969` → absent.\n")
    findings, checked = rung.path_findings(vault)
    assert checked == 1
    joined = "\n".join(findings)
    assert "`eval/uptake/a_report_no_commit_has_1969.json`" in joined, findings
    assert "`~/lloyd/.t-never-here-1969`" in joined, findings
    assert len(findings) == 2, findings


def test_only_what_the_edit_adds_is_reported(vault):
    """The base commit already names a dead path; an edit that adds prose and a
    path that exists reports nothing. A permanent alarm is a disabled alarm."""
    _append(vault, "\nAlso reads `tests/test_skill_tool_names.py`.\n")
    findings, checked = rung.path_findings(vault)
    assert (findings, checked) == ([], 1)


def test_a_new_skill_is_checked_in_full_and_other_files_are_not(vault):
    new = "skills/brand-new/SKILL.md"
    (vault / "skills" / "brand-new").mkdir()
    (vault / new).write_text("# New\n\nRun `app/no_such_module_1969.py`.\n",
                             encoding="utf-8")
    (vault / "memory").mkdir()
    (vault / "memory" / "note.md").write_text(
        "`app/no_such_module_1969.py` again\n", encoding="utf-8")
    _git(vault, "add", "-A")
    findings, checked = rung.path_findings(vault)
    assert checked == 1, "only skill pages and task files are this rung's corpus"
    assert findings == [f"{new}: newly names `app/no_such_module_1969.py`, which is "
                        f"not in the checkout at {REPO_ROOT}"]


def test_the_rung_asks_the_suites_own_rule():
    """One definition per rule: the unmarked node's `_absent_refs` is loaded from
    the test module, not copied, and the bench node calls `bench_dead_refs`."""
    guard = rung._load_guard()
    assert Path(guard.__file__) == REPO_ROOT / "tests" / "test_skill_tool_names.py"
    assert guard._absent_refs("x", "see `app/no_such_module_1969.py`", None) == {
        "x::repo:app/no_such_module_1969.py"}
    bench = (REPO_ROOT / "tests" / "test_bench_audit_tasks.py").read_text()
    assert "bench_dead_refs" in bench and "re.compile(r\"(?:~|/home" not in bench


def test_the_wrapper_prints_the_finding_and_still_commits(vault):
    """End to end through the door the nightly jobs use. `LLOYD_PYTHON` is this
    interpreter because a worktree has no `.venvs/`; production resolves the repo
    venv by itself."""
    _append(vault, "\nThe stray `eval/uptake/a_report_no_commit_has_1969.json`.\n")
    proc = subprocess.run(
        ("bash", str(WRAPPER), "nightly: knowledge write (demo)", "--", "skills/"),
        capture_output=True, text=True, timeout=120, cwd=str(vault),
        env={"VAULT_DIR": str(vault), "PATH": "/usr/bin:/bin:/usr/local/bin",
             "LLOYD_PYTHON": sys.executable})
    assert proc.returncode == 0, proc.stderr
    printed = proc.stdout + proc.stderr
    assert "skill-path: 1 staged skill/task file(s) checked, 1 new dead path(s)" in printed
    assert ("skill-path FINDING: " + _SKILL + ": newly names "
            "`eval/uptake/a_report_no_commit_has_1969.json`") in printed
    assert "without a repo-rooted path" in printed
    assert _git(vault, "log", "--format=%s", "-1").strip() == \
        "nightly: knowledge write (demo)"


def test_an_unloadable_rule_skips_and_never_blocks(vault, monkeypatch, capsys):
    monkeypatch.setattr(rung, "GUARD", REPO_ROOT / "tests" / "no_such_guard_1969.py")
    _append(vault, "\n`eval/uptake/a_report_no_commit_has_1969.json`\n")
    assert rung.report(vault) == 0
    assert "skill-path: CHECK SKIPPED" in capsys.readouterr().out
