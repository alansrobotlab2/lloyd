"""Tests for #3881: the wrapper refuses a skill clock literal THIS commit writes.

`scripts/skill_timezone.py` defines the class once. Until this round its only caller was
`scripts/automod/vault_round.py::skill_timezone_errors`, whose selector is
`p.startswith("skills/") and p.endswith("/SKILL.md")` — so the one violation the vault
actually carries was invisible at land time, because it is not in a SKILL.md. It is
`skills/nightly-reflection-knowledge-write/step-1-and-commit-detail.md` line 110, which
interpolates `$(date +%Y-%m-%d)` into a `vault-commit.sh` subject, and the writer that
follows that instruction commits through the wrapper this suite drives. Measured over
`~/obsidian/skills/**/*.md` on 2026-11-07: 1 violation outside `skills/.archived/` (that
line) and 14 across 8 archived SKILL.md files, in 567 `.md` files under `skills/`.

The clauses these nodes pin, in order:

  1. the wrapper shells out to `scripts/util/skill_timezone_findings.py` as a pre-flight
     rung and does not restate the rule (`test_the_wrapper_...` nodes);
  2. a staged edit that ADDs a violating line is refused — nonzero exit, no commit, and
     the path, the line number and the `$(date -u +…)` remedy all named;
  3. a commit that only MODIFIES other lines of a file whose pre-image carries the
     violation still commits;
  4. the scope is every `.md` under `skills/`: the measured detail-file shape is refused
     and the same line with `-u` commits;
  5. fail-open like the six rungs beside it: no matcher or no interpreter prints
     `CHECK SKIPPED` and commits, and a matcher exit above 1 prints `CHECK FAILED`
     without deciding.

These fixtures are real git repos with a `main` branch, because the wrapper exits on any
other branch and reads the INDEX, so a fixture that stops at `git init` would be asserting
against a branch name and a check that never ran. The vault is always passed as
`--vault`/`LLOYD_VAULT`, never inferred.

Run from a throwaway checkout — `tests/conftest.py` refuses the production tree.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "util" / "vault-commit.sh"
MODULE = REPO / "scripts" / "util" / "skill_timezone_findings.py"
RULE = REPO / "scripts" / "skill_timezone.py"

#: The violating line, in the shape the vault really carries it: a `vault-commit.sh`
#: subject with a local `date` interpolated in. Line 10 below is the same line number it
#: holds in the live file only by coincidence of this fixture's layout; every assertion
#: that needs a number computes it, and this one is stated so the fixture is auditable.
BAD_SUBJECT = ('LLOYD_JOB=nightly-reflection-knowledge-write '
               '~/lloyd/scripts/util/vault-commit.sh '
               '"nightly: knowledge write $(date +%Y-%m-%d)" -- memory/ lloyd/')
#: The fix the rule blesses, measured by `test_the_rule_blesses_every_shape_the_fix_uses`
#: over in tests/test_skill_timezone_literals.py. Only the flag differs from BAD_SUBJECT,
#: so a refusal that cannot tell the two apart cannot survive this file.
GOOD_SUBJECT = BAD_SUBJECT.replace("$(date +%Y-%m-%d)", "$(date -u +%Y-%m-%d)")
#: Shape 1 of the same rule: a displayed time whose abbreviation is a season's, not a
#: clock's. Included so "the rung enforces the class" means both shapes the one
#: definition has, and not just the one this round was filed for.
BAD_DISPLAY = "- generated: YYYY-MM-DD HH:MM PST"


def _detail(bad: bool = True, extra: str = "First step prose.") -> str:
    """A skill DETAIL file (not a SKILL.md), with the clock line on line 10."""
    subject = BAD_SUBJECT if bad else GOOD_SUBJECT
    return (
        "---\n"
        "type: note\n"
        "segment: skills\n"
        "---\n"
        "# Nightly knowledge write — step 1 and commit detail\n"
        "\n"
        f"{extra}\n"
        "\n"
        "```bash\n"
        f"{subject}\n"
        "```\n"
    )


def _git(vault, *args, check=True):
    return subprocess.run(["git", "-C", str(vault), *args],
                          capture_output=True, text=True, check=check)


def _vault(tmp_path):
    """A scratch vault already on `main`, with one committed file."""
    vault = tmp_path / "vault"
    vault.mkdir()
    _git(vault, "init", "-q", "-b", "main")
    _git(vault, "config", "user.email", "test@example.invalid")
    _git(vault, "config", "user.name", "Test")
    _git(vault, "config", "commit.gpgsign", "false")
    (vault / "seed.md").write_text("# Seed\n\nCommitted before the run under test.\n")
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "seed")
    return vault


def _env(vault, **extra):
    env = dict(os.environ)
    env.update(LLOYD_VAULT=str(vault), VAULT_DIR=str(vault), HOME=str(Path.home()),
               GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
               GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid")
    # Nothing is attributed to a job here: `witness.py` runs only with `LLOYD_JOB`, and
    # a job identity would need the live pipeline directory to write a witness into.
    for k in ("LLOYD_JOB", "LLOYD_JOB_WRITES", "LLOYD_PYTHON", "LLOYD_DATA"):
        env.pop(k, None)
    env.update(extra)
    return env


def _run(vault, *args, env=None, script=None):
    """The wrapper by its own shebang (`#!/usr/bin/env bash`), not through python."""
    return subprocess.run([str(SCRIPT if script is None else script), *args],
                          capture_output=True, text=True,
                          env=env or _env(vault), cwd=REPO)


def _module(vault, *args, script=None):
    return subprocess.run([sys.executable,
                           str(MODULE if script is None else script),
                           "--vault", str(vault), *args],
                          capture_output=True, text=True, cwd=REPO)


def _head(vault) -> str:
    return _git(vault, "rev-parse", "HEAD").stdout.strip()


def _head_paths(vault) -> list[str]:
    """Every path in HEAD — recursed into subtrees, the reason #3545 states for
    `git show --name-only` alone being the wrong assertion."""
    r = _git(vault, "show", "--pretty=format:", "--name-only", "-r", "HEAD")
    return [p for p in r.stdout.splitlines() if p.strip()]


def _copy_rung_dir(dest: Path, *, with_module: bool = True, stub_exit: int | None = None):
    """A scripts/util/ copy in `dest`, to hold the wrapper's `$SCRIPT_DIR` fixed.

    The wrapper resolves its rungs against its own directory (#2696: a login shell's
    `cd` into `~` made that relative resolve to `~/scripts/...`), so a test for the
    missing-matcher branch has to move the copy, not the vault.
    """
    util = dest / "scripts" / "util"
    util.mkdir(parents=True)
    shutil.copy(SCRIPT, util / "vault-commit.sh")
    if with_module:
        shutil.copy(MODULE, util / "skill_timezone_findings.py")
    if stub_exit is not None:
        # A matcher that decides nothing and exits `stub_exit`: what the bash branch for
        # "exit > 1" is for, and unreachable any other way — the real module exits 3 only
        # when git itself disagrees, which no fixture can arrange on demand.
        (util / "skill_timezone_findings.py").write_text(
            f"import sys\nprint('stub: cannot look', file=sys.stderr)\nsys.exit({stub_exit})\n"
        )
    return util / "vault-commit.sh"


# ---------------------------------------------------------------- clause 1: the rung

def test_the_wrapper_shells_out_to_the_shared_matcher_and_restates_no_rule():
    """The bash half of the seam exists, and the rule stays in the one module.

    `grep -cin timezone scripts/util/vault-commit.sh` is the item's own acceptance
    measurement; `sys.executable` is not (bash cannot import). Restating the patterns in
    bash is the #2468 failure — the two doors disagreeing about what the class is — so
    this also asserts the matcher carries no zone pattern of its own and loads the rule
    by path instead.
    """
    script = SCRIPT.read_text(encoding="utf-8")
    assert "skill_timezone_findings.py" in script, script
    # `grep -cin timezone` counts MATCHING LINES, case-insensitively, and the item's
    # acceptance is that this is > 0 where it measured 0.
    matching = [ln for ln in script.splitlines() if "timezone" in ln.lower()]
    assert matching, "the wrapper still never mentions a timezone check"
    assert any("skill_timezone_findings.py" in ln for ln in matching)

    matcher = MODULE.read_text(encoding="utf-8")
    assert "template_clock_violations" in matcher, (
        "the rung must call the one definition, not a copy of it")
    for pattern in ("P[SD]T", "HH:MM", "DATE_SUBST", "UTC_FLAG"):
        assert pattern not in matcher.replace("_GIT_MESSAGE", ""), (
            f"the matcher restates the rule's {pattern!r} instead of importing it")
    assert RULE.exists() and RULE.read_text(encoding="utf-8").count(
        "def template_clock_violations") == 1


def test_the_rung_refuses_instead_of_printing_like_the_two_rungs_above_it():
    """Exit 1 is a refusal path, not a warning beside the warning-shaped majority.

    Three of the seven rungs are report-only by explicit ruling, so 'print it' is
    readable as house style; #3545 settled that a refusal is satisfiable and printing is
    not, on the same diff-scoping argument clause 2 uses. Pinned structurally: the block
    that reads `TIMEZONE_RC` at 1 contains `exit 1`, and `git reset` of the named paths,
    and does not contain the commit line — and the rung sits BEFORE the index readback,
    so a refused path never reaches the commit body or the #1070 unattributed list.
    """
    script = SCRIPT.read_text(encoding="utf-8")
    at = script.find('"$TIMEZONE_RC" -eq 1')
    assert at != -1, "no branch acting on the matcher's exit code 1"
    block = script[at:script.find("\nfi\n", at)]
    assert "exit 1" in block, block
    assert "reset -q --" in block, "the refused paths must leave the index (#3545)"
    assert "TIMEZONE_RC" in block
    readback = script.find("git diff --cached --name-only -z")
    assert readback != -1 and script.find("TIMEZONE_RUNG=") < readback, (
        "the rung must run before the index readback, like the rungs above it")


def test_the_matcher_is_started_by_path_and_imports_the_rule_without_a_package():
    """`sys.path[0]` is `scripts/util/`, so a root-package import would exit 1.

    Exit 1 is this tool's REFUSAL code: a top-level `from scripts...` that fails would
    refuse every vault commit in the tree while naming no path (#3403). The sibling
    modules' answer is to load the rule lazily by path, which is asserted here and
    exercised for the bash-launched interpreter by the sweep in
    tests/test_no_runtime_paths_in_code.py.
    """
    src = MODULE.read_text(encoding="utf-8")
    head = src.split("def ", 1)[0]
    assert not re.search(r"(?m)^from (app|scripts)\.([\w.]+) import", head), head
    assert "importlib" in src, (
        "the rule is loaded lazily by path, so a failed load is a 3 and not a 1")


# ---------------------------------------------------- clause 2: a fresh line refused

def test_a_fresh_clock_literal_in_a_staged_skill_edit_is_refused_and_nothing_committed(tmp_path):
    """The rung's refusal: nonzero exit, HEAD unmoved, the path out of the index.

    Nothing is `git add`-ed by hand here on purpose: the wrapper's own staging step is
    part of the seam, and a fixture that pre-staged the file would pass even if the rung
    read a path the wrapper never staged.
    """
    vault = _vault(tmp_path)
    before = _head(vault)
    (vault / "skills" / "nightly-reflection-knowledge-write").mkdir(parents=True)
    (vault / "skills" / "nightly-reflection-knowledge-write"
     / "step-1-and-commit-detail.md").write_text(_detail(), encoding="utf-8")

    r = _run(vault, "nightly: knowledge write", "--", "skills/")

    assert r.returncode != 0, r.stdout + r.stderr
    assert "SKILL CLOCK LITERAL REFUSED" in r.stderr, r.stderr
    assert _head(vault) == before, "a refused check must not commit anything"
    assert "step-1-and-commit-detail.md" not in _head_paths(vault)
    # Refused, not deleted (#3545, #2468): the writer's prose survives on disk, so
    # editing the one line and re-staging is the entire remedy.
    assert "$(date +%Y-%m-%d)" in (
        vault / "skills" / "nightly-reflection-knowledge-write"
        / "step-1-and-commit-detail.md").read_text(encoding="utf-8")


def test_the_refusal_names_the_path_the_line_and_the_utc_remedy(tmp_path):
    """A refusal the caller can act on without opening the file.

    The line number is checked against the fixture's own layout — recomputed, never
    hardcoded — because a refusal pointing at the wrong line is a refusal a writer cannot
    fix, and this project has already been bitten by a check whose reported location was
    wrong (#3455). The remedy strings are the two the rule blesses.
    """
    vault = _vault(tmp_path)
    rel = "skills/nightly-reflection-knowledge-write/step-1-and-commit-detail.md"
    (vault / "skills" / "nightly-reflection-knowledge-write").mkdir(parents=True)
    body = _detail()
    want_line = body.split("\n").index(BAD_SUBJECT) + 1
    (vault / rel).write_text(body, encoding="utf-8")
    # Staged, not merely written: the matcher's whole subject is the INDEX, so a fixture
    # that never ran `git add` would be asserting that the rung found nothing — true, and
    # for the wrong reason. `_run` below stages through the wrapper's own `git add` step.
    _git(vault, "add", "-A", "--", "skills/")

    r = _module(vault)

    assert r.returncode == 1, r.stderr
    assert r.stdout.strip().splitlines() == [rel], (
        "stdout carries the refused paths and nothing else: that is the bash contract")
    assert f"{rel}: line {want_line}" in r.stderr, r.stderr
    assert "$(date -u +" in r.stderr, "the fix must be spelled in the refusal"
    assert "date -u +%Y-%m-%dT%H:%M:%SZ" in r.stderr, r.stderr
    wrapper = _run(vault, "nightly: knowledge write", "--", "skills/")
    assert wrapper.returncode == 1 and rel in wrapper.stderr, wrapper.stderr


# ------------------------------------------------- clause 3: the pre-image still lands

def test_a_commit_that_only_edits_other_lines_of_a_violating_file_still_commits(tmp_path):
    """The violating line is unchanged context, so it is not this commit's doing.

    This is the clause that keeps the rung off the nightly writer's back. The tree carries
    one live violation (`step-1-and-commit-detail.md:110`) and 14 in archived skills, and
    the wrapper's batches span `memory/ lloyd/ people/ projects/ knowledge/ skills/
    backlog/ autonomy/`; a whole-body check over touched files would hold every one of
    those segments on a line nobody wrote — #2245's stranded-snapshot failure, and the
    reason clause 2 is diff-relative rather than file-relative.
    """
    vault = _vault(tmp_path)
    rel = "skills/nightly-reflection-knowledge-write/step-1-and-commit-detail.md"
    (vault / "skills" / "nightly-reflection-knowledge-write").mkdir(parents=True)
    (vault / rel).write_text(_detail(), encoding="utf-8")
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "seed the detail file with the violation in its body")

    # One unrelated line moves; line 10 stays exactly where git puts it as context.
    (vault / rel).write_text(
        _detail(extra="First step prose, reworded by the round under test."),
        encoding="utf-8")
    before = _head(vault)

    r = _run(vault, "nightly: knowledge write", "--", "skills/")

    assert r.returncode == 0, r.stdout + r.stderr
    assert "SKILL CLOCK LITERAL REFUSED" not in r.stderr, r.stderr
    assert _head(vault) != before, "the unrelated edit must land"
    assert rel in _head_paths(vault)
    assert "$(date +%Y-%m-%d)" in (vault / rel).read_text(encoding="utf-8"), (
        "the pre-existing line is still there, untouched and unrepaired")


def test_renaming_a_file_that_carries_a_violation_still_commits(tmp_path):
    """A `git mv` moves a violation without writing one.

    Rename detection is why this needs a node of its own: `git diff -M` given only the
    destination pathspec answers `new file mode` with every line inside a `+` range, so
    passing one path instead of both would refuse a rename for content nobody typed.
    Measured on git 2.43 while writing the matcher; the matcher passes both operands.
    """
    vault = _vault(tmp_path)
    old = "skills/nightly-reflection-signals/SKILL.md"
    new = "skills/nightly-reflection-signals/step-1-and-commit-detail.md"
    (vault / "skills" / "nightly-reflection-signals").mkdir(parents=True)
    (vault / old).write_text(_detail(), encoding="utf-8")
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "seed the skill that is about to be split")

    _git(vault, "mv", old, new)
    before = _head(vault)

    r = _run(vault, "rename the commit step into a detail file", "--", "skills/")

    assert r.returncode == 0, r.stdout + r.stderr
    assert "SKILL CLOCK LITERAL REFUSED" not in r.stderr, r.stderr
    assert _head(vault) != before
    assert new in _head_paths(vault)


def test_deleting_a_violating_skill_commits(tmp_path):
    """A removal adds no line, so the one place the violation leaves is never refused."""
    vault = _vault(tmp_path)
    rel = "skills/deprecated-skill/SKILL.md"
    (vault / "skills" / "deprecated-skill").mkdir(parents=True)
    (vault / rel).write_text(_detail(), encoding="utf-8")
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "seed the skill that is about to go")

    _git(vault, "rm", "-q", rel)
    before = _head(vault)

    r = _run(vault, "retire a skill", "--", "skills/")

    assert r.returncode == 0, r.stdout + r.stderr
    assert _head(vault) != before
    # `ls-tree` and not `_head_paths`: `git show --name-only` lists a deletion commit's
    # removed path, which is the very thing being asserted away here — the same
    # flatten-the-tree trap, with its polarity reversed.
    in_head = _git(vault, "ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines()
    assert rel not in in_head, in_head


# ------------------------------------------------------- clause 4: scope of the scope

def test_a_detail_file_carrying_the_measured_subject_shape_is_refused(tmp_path):
    """Clause 3's scope: a step-detail file, not a SKILL.md, in the live file's shape.

    The land rail's selector is `skills/*/SKILL.md`, which is exactly why the vault's one
    live violation — a detail file — has never been refused anywhere. This fixture is that
    file's shape: the `vault-commit.sh "… $(date +%Y-%m-%d)"` subject inside a fenced
    block in a non-SKILL.md under a skill directory.
    """
    vault = _vault(tmp_path)
    rel = "skills/some-other-skill/step-1-and-commit-detail.md"
    (vault / "skills" / "some-other-skill").mkdir(parents=True)
    (vault / rel).write_text(_detail(bad=True), encoding="utf-8")

    r = _run(vault, "nightly: knowledge write", "--", "skills/")

    assert r.returncode != 0, "a detail file must be judged like a SKILL.md"
    assert rel in r.stderr, r.stderr


def test_the_same_subject_written_with_the_u_flag_commits(tmp_path):
    """The clean sibling of the fixture above: identical bytes except the one flag."""
    assert GOOD_SUBJECT != BAD_SUBJECT and "-u" in GOOD_SUBJECT
    vault = _vault(tmp_path)
    rel = "skills/some-other-skill/step-1-and-commit-detail.md"
    (vault / "skills" / "some-other-skill").mkdir(parents=True)
    (vault / rel).write_text(_detail(bad=False), encoding="utf-8")
    before = _head(vault)

    r = _run(vault, "nightly: knowledge write", "--", "skills/")

    assert r.returncode == 0, r.stdout + r.stderr
    assert "SKILL CLOCK LITERAL REFUSED" not in r.stderr, r.stderr
    assert _head(vault) != before
    assert rel in _head_paths(vault)


def test_a_skill_md_carrying_the_displayed_time_shape_is_refused(tmp_path):
    """Both shapes of the one definition, and the SKILL.md half of the scope too."""
    vault = _vault(tmp_path)
    rel = "skills/report-writer/SKILL.md"
    body = ("---\ntype: note\nsegment: skills\n---\n# Report writer\n\n"
            "Header line of the template:\n\n"
            f"{BAD_DISPLAY}\n")
    want_line = body.split("\n").index(BAD_DISPLAY) + 1
    (vault / "skills" / "report-writer").mkdir(parents=True)
    (vault / rel).write_text(body, encoding="utf-8")

    r = _run(vault, "add the header rule", "--", "skills/")

    assert r.returncode != 0, r.stdout + r.stderr
    assert f"{rel}: line {want_line}" in r.stderr, r.stderr
    assert "hardcoded zone abbreviation" in r.stderr, r.stderr


def test_a_clock_literal_added_outside_skills_is_not_this_rungs_business(tmp_path):
    """The scope cuts both ways: this rung governs `skills/`, not the whole vault.

    A `$(date +%F)` in a memory note is a different class with its own owner (the writer
    template and #2542's `utc_today()`); refusing it here would put two doors on one
    sentence and neither of them owns the ruling.
    """
    vault = _vault(tmp_path)
    (vault / "memory" / "learnings").mkdir(parents=True)
    (vault / "memory" / "learnings" / "a-note.md").write_text(
        "# Note\n\n"
        'Run `~/lloyd/scripts/util/vault-commit.sh "nightly: signals $(date +%F)" '
        '-- memory/` a day.\n', encoding="utf-8")
    before = _head(vault)

    r = _run(vault, "add a note", "--", "memory/")

    assert r.returncode == 0, r.stdout + r.stderr
    assert "SKILL CLOCK LITERAL REFUSED" not in r.stderr, r.stderr
    assert _head(vault) != before


def test_the_worktree_copy_does_not_clear_a_violation_the_index_still_ships(tmp_path):
    """The check reads the staged blob, never the working tree.

    `orphan_topic_findings._staged_blob` is the precedent and the reason is #1867/#2245:
    this wrapper commits pathspec-scoped batches, so an index can hold a line the
    worktree no longer does — and `git commit` takes the index. A matcher that opened the
    file on disk would bless that commit and ship the defect it just inspected, the same
    blindness as a rung that exits 0 on a file it never opened.

    Asked of the matcher and not of the wrapper, because the wrapper's own pathspec
    `git add` closes the window first: by the time its rungs run, the index already holds
    the worktree bytes. What this pins is which bytes the matcher reads.
    """
    vault = _vault(tmp_path)
    rel = "skills/some-other-skill/step-1-and-commit-detail.md"
    (vault / "skills" / "some-other-skill").mkdir(parents=True)
    (vault / rel).write_text(_detail(bad=False), encoding="utf-8")
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "seed the clean detail file")
    # The violating line goes into the INDEX…
    (vault / rel).write_text(_detail(bad=True), encoding="utf-8")
    _git(vault, "add", "-A")
    # …and the fix stays in the worktree, unstaged. `git commit` takes the index.
    (vault / rel).write_text(_detail(bad=False), encoding="utf-8")

    r = _module(vault)

    assert r.returncode == 1, (
        f"the staged blob still carries the violation that commit would ship: {r.stderr}")
    assert rel in r.stderr, r.stderr


# ----------------------------------------------------- clause 5: fail-open, unchanged

def test_a_missing_matcher_prints_check_skipped_and_still_commits(tmp_path):
    """The wrapper's own fail-open, and the one that answers 'is the rung reachable?'.

    #2696: the login shell's `cd` into `~` used to make every relative rung path
    resolve outside the checkout, so every check ran against nothing and the guard was a
    no-op for months. This node cannot be satisfied by an in-place module copy: the
    wrapper resolves its rungs against its own directory, so the copy has to live in a
    `scripts/util/` of its own.
    """
    vault = _vault(tmp_path)
    (vault / "skills" / "demo").mkdir(parents=True)
    (vault / "skills" / "demo" / "SKILL.md").write_text(
        _detail(bad=True).replace("# Nightly knowledge write — step 1 and commit detail",
                                  "# Demo skill"), encoding="utf-8")
    stub = _copy_rung_dir(tmp_path / "nocheck", with_module=False)

    r = _run(vault, "demo: commit despite the missing check", "--", "skills/",
             script=stub)

    assert r.returncode == 0, r.stderr
    assert "CHECK SKIPPED" in r.stderr and "skill timezone-literal" in r.stderr, r.stderr
    assert "skills/demo/SKILL.md" in _head_paths(vault), "fail open: it commits"
    assert "SKILL CLOCK LITERAL REFUSED" not in r.stderr


def test_no_python_interpreter_prints_check_skipped_and_still_commits(tmp_path):
    """The other fail-open leg: the matcher is on disk but there is nothing to run it.

    The wrapper's `$PY` probe is `python3` then `python`, so both have to be absent from
    `PATH` — see `_interpreterless_bin` for why that is a farm and not `/nonexistent`.
    """
    vault = _vault(tmp_path)
    (vault / "skills" / "demo").mkdir(parents=True)
    (vault / "skills" / "demo" / "SKILL.md").write_text(
        _detail(bad=True).replace("# Nightly knowledge write — step 1 and commit detail",
                                  "# Demo skill"), encoding="utf-8")

    r = _run(vault, "demo: commit with no interpreter", "--", "skills/",
             env=_env(vault, PATH=_interpreterless_bin(tmp_path)))

    assert r.returncode == 0, r.stderr
    assert "CHECK SKIPPED" in r.stderr and "skill timezone-literal" in r.stderr, r.stderr
    assert "skills/demo/SKILL.md" in _head_paths(vault)


def _interpreterless_bin(tmp_path) -> str:
    """A PATH holding every system binary except an interpreter.

    Farming the real `/usr/bin` and `/bin` rather than a hand-picked list, because the
    wrapper's commit path calls more than the rungs: `git`, `sed`, `grep`, `date`,
    `mktemp`, `dirname`, `tr`, `rm`, `tail`, and `bash` itself — the shebang is
    `/usr/bin/env bash`, and `env` finds `bash` through THIS PATH. Excluding by name
    prefix catches `python`, `python3`, `python3.11` and `pypy` in one rule, which is the
    point: `/nonexistent` would take `git` with it and the wrapper would die at its first
    `git rev-parse` instead of reaching the rung.
    """
    farm = tmp_path / "no-python-bin"
    farm.mkdir()
    seen: set[str] = set()
    for d in ("/usr/bin", "/bin", "/usr/local/bin"):
        root = Path(d)
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            name = entry.name
            # A `seen` SET and not `(farm / name).exists()`: that call follows the link
            # it just wrote, so a source entry that is itself dangling (`/usr/bin/
            # QtWebEngineProcess6` here) reads as "not linked yet" and the repeat link
            # over the same name raises FileExistsError.
            if name.startswith(("python", "pypy")) or name in seen:
                continue
            seen.add(name)
            os.symlink(entry, farm / name)
    assert not (farm / "python3").exists()
    assert (farm / "git").exists() and (farm / "bash").exists()
    return str(farm)


def test_a_matcher_exiting_above_one_prints_check_failed_and_does_not_decide(tmp_path):
    """A broken check commits the batch, but says it went blind instead of implying clean.

    The exit code is the only thing distinguishing these cases, so the wrapper has to
    read it and not merely survive it. Same shape as the orphan-topic node's
    `test_a_broken_check_exits_three_and_is_never_laundered_into_a_clean_zero`, and
    asserted the same way: exit 0, the CHECK FAILED line naming the exit code, no refusal
    line, and the violating file actually committed.
    """
    vault = _vault(tmp_path)
    (vault / "skills" / "demo").mkdir(parents=True)
    (vault / "skills" / "demo" / "SKILL.md").write_text(
        _detail(bad=True).replace("# Nightly knowledge write — step 1 and commit detail",
                                  "# Demo skill"), encoding="utf-8")
    stub = _copy_rung_dir(tmp_path / "exitthree", with_module=True, stub_exit=3)

    r = _run(vault, "demo: commit despite a broken check", "--", "skills/", script=stub)

    assert r.returncode == 0, r.stderr
    assert "CHECK FAILED" in r.stderr and "exit 3" in r.stderr, r.stderr
    assert "SKILL CLOCK LITERAL REFUSED" not in r.stderr, r.stderr
    assert "skills/demo/SKILL.md" in _head_paths(vault)
    assert "$(date +%Y-%m-%d)" in (vault / "skills" / "demo" / "SKILL.md").read_text(
        encoding="utf-8"), "fail-open does not mean the check got to edit the prose"


def test_the_matcher_exits_three_when_the_index_cannot_be_read(tmp_path):
    """The module's own 3-branch: a vault git will not answer for is not a clean vault."""
    not_a_repo = tmp_path / "loose-files"
    not_a_repo.mkdir()
    (not_a_repo / "skills").mkdir()

    r = _module(not_a_repo)

    assert r.returncode == 3, r.stderr
    assert "CHECK FAILED (not a clean read)" in r.stderr, r.stderr
    assert "REFUSED" not in r.stderr, (
        "a 1 from a crash would refuse every commit in the tree, naming no path")


def test_the_matcher_is_quiet_when_no_staged_path_is_under_skills(tmp_path):
    """A commit touching nothing under `skills/` costs one git call and no verdict.

    The `skills/` pathspec is the whole scope clause; this node is here so a future
    widening to the whole vault has to pass through this assertion on purpose.
    """
    vault = _vault(tmp_path)
    (vault / "memory").mkdir()
    (vault / "memory" / "x.md").write_text("# x\n\n"
                                           'commit -m "s $(date +%F)"\n',
                                           encoding="utf-8")
    _git(vault, "add", "-A")

    r = _module(vault, "--json")

    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout
