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

import json
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


def test_a_path_the_skill_records_its_job_wrote_prints_no_finding(vault):
    """#2223's seam. The unmarked node exempts a path a document says its subject
    WROTE, whether at a flag, a redirection or in a plain sentence, and this rung
    loads that node's `_absent_refs` — so the prose half arrives here for free, but
    only on the leg that loads it.

    The bench leg is defined in `skill_path_findings.py` with its own narrower
    matcher, so the two legs of one commit's report have to subtract the same
    creation sites or a single sentence prints a finding and no finding at once.
    This node goes red from either half failing: take out the node's prose rule and
    the unmarked leg speaks, take out the subtraction and the bench leg does.

    The phantom is named `name_prior_…json` because that shape is what makes the
    node test the rule instead of the ignore list. `.gitignore` exempts an ignored
    path from the unmarked leg, and the eval tree's JSON is ignored except for the
    `!eval/djev/name_prior_*.json` negation — which is exactly why task 96's report
    was counted as drift at all: the negation says those reports are meant to be
    committed, so nothing about the path is ignorable and only the sentence's
    meaning can excuse its absence.
    """
    _append(vault, "\nThe probe ran. It wrote\n"
                   "`~/lloyd/eval/djev/name_prior_a_created_2223.json` there.\n")
    findings, checked = rung.path_findings(vault)
    assert (findings, checked) == ([], 1), findings


def test_a_path_the_skill_tells_a_run_to_read_prints_on_both_legs(vault):
    """The positive control the node above cannot be satisfied by: an instruction
    to open a file the checkout does not have is the violation this rung exists to
    catch before the push, and it prints — once, because the bench leg subtracts
    what the unmarked leg already named rather than repeating it. An exemption
    widened past creation statements, to a whole document or to every absent path a
    skill page happens to mention, passes the node above and fails here. Same
    phantom shape as that node and the same reason: an ignorable path would silence
    the leg that carries the node's rule.
    """
    _append(vault, "\nRead `~/lloyd/eval/djev/name_prior_a_read_2223.json` now.\n")
    findings, checked = rung.path_findings(vault)
    assert checked == 1 and len(findings) == 1, (
        f"an instruction to open a phantom stopped being a finding: {findings}")
    assert "eval/djev/name_prior_a_read_2223.json" in findings[0], findings


def test_a_path_the_skill_records_its_run_crashed_on_prints_no_finding(vault):
    """#2223's other half at the seam: an Activity Log line quoting
    `FileNotFoundError: <path>` records that the file was MISSING, which is the
    fence's own verdict and not a citation of it.

    This is the node that pins the seam to `_recorded_sites` rather than to
    `_creation_sites`. The unmarked leg stays quiet about the path either way,
    because the rule lives in the loaded module; what this leg can get wrong on its
    own is the bench matcher defined in `skill_path_findings.py`, whose narrower
    path rule has never heard of exception messages. Point the subtraction back at
    `_creation_sites` and the bench leg prints this finding while the suite's node
    prints nothing — the two legs of one commit disagreeing about one bullet, which
    is the whole reason this file exists.
    """
    _append(vault, "\nThe run died. Its last line was\n"
                   "`FileNotFoundError: "
                   "~/lloyd/eval/djev/name_prior_a_crashed_2223.json`\n")
    findings, checked = rung.path_findings(vault)
    assert (findings, checked) == ([], 1), (
        f"a skill page quoting the error its run raised was charged for the "
        f"missing file: {findings}")


_DANGLING_2484 = "scripts/a_script_the_dangling_report_lists_2484.py"


def _quoted_dangling_bullet(*, suffix=":610", block=True) -> str:
    """A skill page quoting one bullet of a referential-integrity report, in the shape
    `scripts/maintenance/referential_integrity.py` writes it: citing half, target half
    carrying the line the cite sat on, and the `ri:dangling` block that proves the bullet
    list is a generated sweep and not a hand-written page full of arrows.

    `suffix=""` spells the target without a line number and `block=False` drops the block,
    so the node below can take the proof away from the same phantom and show it printing.
    """
    target = f"~/lloyd/{_DANGLING_2484}{suffix}"
    lines = ["", "The nightly sweep's own bullet read:",
             f"- `lloyd/USER.md:28` → `{target}` — "
             "stat:repo_root|vault_root|data_root|citing_dir (none exist)"]
    if block:
        lines += ["", "<!-- ri:dangling "
                      + json.dumps([f"lloyd/USER.md:28 -> {target}"]) + " -->"]
    return "\n".join(lines) + "\n"


def test_a_line_numbered_dangling_bullet_a_skill_quotes_prints_no_finding(vault):
    """#2484 clause 4 at the process boundary. The suite's unmarked node and this
    pre-flight leg read one definition — `_recorded_sites`, loaded out of
    `tests/test_skill_tool_names.py` by `_load_guard` — so the leg has to agree that a
    dangling report's own `` `…/x.py:610` `` bullet is the sweep talking about a dead
    path, not the skill page citing it.

    The seam is a subprocess boundary in production and an importlib load here: this
    module is a file the running suite never imports, loaded fresh into `vault-commit.sh`
    at the door the nightly jobs commit through. Nothing but a call through
    `path_findings` crosses it, which is why the claim is asserted as findings and not as
    a helper's return value. Before #2484 this leg charged the page: the loaded
    `_absent_refs` returned the row because the recorded-site pair it subtracts kept the
    `:610` and so matched no row `_named_paths` had yielded — and `bench_dead_refs`, whose
    own matcher knows nothing about report blocks, was subtracted against a `created` set
    bearing that same stale suffix, so it had its own copy of the finding ready and was
    hidden only because the loaded leg named the path first. The suite's node charged the
    same citation, so one bullet made the pre-flight print and main go red; the fix is
    reader-side and reaches both surfaces together, which is the whole point of the shared
    name.
    """
    _append(vault, _quoted_dangling_bullet())
    findings, checked = rung.path_findings(vault)
    assert (findings, checked) == ([], 1), (
        f"a skill page quoting a generated report's own bullet was charged for the "
        f"path that report proved dead: {findings}")


def test_the_same_line_numbered_path_without_the_sweep_block_prints(vault):
    """The control the node above cannot be satisfied by: take the `ri:dangling` block out
    of the same text and the bullet is a plain citation of a missing script, which is what
    this rung was written to say before the commit.

    Without it the fixture would go green for an exemption widened to "any page whose
    prose contains an arrow", and the node above would measure nothing. Same phantom, same
    bullet, same suffix: the only difference is the machine block that proves a sweep, not
    a reader, wrote the list — so the one finding printed here is the path from the node
    above, and it prints once because the bench leg subtracts what the loaded rule already
    named.
    """
    _append(vault, _quoted_dangling_bullet(block=False))
    findings, checked = rung.path_findings(vault)
    assert checked == 1
    assert findings == [
        f"{_SKILL}: newly names `{_DANGLING_2484}`, which is not in the checkout at "
        f"{REPO_ROOT}"], (
        f"a bare citation of an absent script stopped being a finding: {findings}")
