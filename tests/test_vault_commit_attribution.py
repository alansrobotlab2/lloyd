"""A vault commit cannot be read as authorship of the content it carries: #1070.

`scripts/util/vault-commit.sh` staged the whole tree — bare `git add -A` over
`~/obsidian` — and committed it under the *calling job's* message. It also took
exactly one argument, so no caller could narrow it: there was no pathspec
parameter to pass. The result is that `git log` answers "who wrote this line"
with "whoever committed last", and for three real incidents that answer is wrong:

* `0860640e` "autonomy-data-pipeline: pre-flight 2026-09-10" (21 files) captured
  `lloyd/MEMORY.md` truncated to one line. The clobber record the file itself now
  carries says the truncating writer is *still unidentified*, because the
  truncating writer was not the committer.
* `fe1cada6` "skills-mgmt: pre-run snapshot" (34 files) captured the earlier
  MEMORY.md copy of SOUL.md; backlog #464 closed with the writer unidentified.
* `3274963e` "skills-mgmt: pre-run snapshot" (44 files, +1599 −99) is where
  autonomy task #51's silent `model: secondary` → `primary` revert first appears,
  under a subject about skills management (#937, closed unattributed).

This is not historical. 55 `pre-flight`/`pre-run snapshot` commits landed after
this item was filed on 2026-09-13, and the shape is the same: `05aeb28a`
"autonomy-data-pipeline: pre-flight 2026-09-17" = 46 files, +2629 −248; `dd042a74`
"nightly-reflection: pre-flight 2026-09-16" = 32 files. A pre-flight step runs
*before* the job writes anything, so by construction every one of those commits is
some other job's dirty state under an innocent job's name.

Two behaviours replace it, and the pair is what makes `git log` honest:

1. **A pathspec.** `vault-commit.sh "msg" -- <path>…` stages and commits exactly
   the named paths, and the rest of the tree stays dirty and uncommitted.
2. **An honest fallback.** Invoked with no pathspec while the tree holds changes,
   the wrapper writes an `unattributed dirty state:` block into the commit body
   naming every path the commit carries, and prints the same list to stderr. The
   commit no longer claims to be the committer's own work.

The wrapper is one script whose commit step now also carries #341's branch guard,
#668's job identity (`LLOYD_JOB` → author + `Job:` trailer) and #1127's
autonomy-status rung. Rewriting the staging/commit step can silently drop any of
them, so each is pinned here too rather than assumed.

The boundary under test is bash plus the git index: a job reaches this script as a
shell command and the claim is about what lands in a real commit, so every test
here runs the actual wrapper as a subprocess against a real fixture repo and reads
the result back with `git show`/`git log --format=%B`. Asserting on the script's
text would not fail if `git add` stopped being scoped or `-m` stopped being
repeated. The clause about the *skills* is graded against the live vault, which no
round controls, so those tests carry `live_vault` like the #668 tests next door.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "util" / "vault-commit.sh"
VAULT_SKILLS = Path.home() / "obsidian" / "skills"

#: The literal the wrapper puts at the head of both the commit-body block and the
#: stderr report. A `git log` reader and a job's own grep both key on it, so it is
#: asserted as a constant rather than paraphrased per test.
UNATTRIBUTED = "unattributed dirty state:"

#: The job used where a test crosses the #668 identity seam as well. Spelled as a
#: real skill directory, because that slug is the job's identity everywhere else.
JOB = "nightly-reflection-knowledge-write"


def _git(repo: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(("git", "-C", str(repo), *args),
                          capture_output=True, text=True, timeout=120,
                          env=env if env is not None else os.environ.copy())
    assert proc.returncode == 0, f"git {' '.join(args)}: {proc.stderr}"
    return proc


def _env(repo: Path, job: str | None = None) -> dict:
    """A minimal env: no HOME, so no `~/.gitconfig` leaks an identity into a
    fixture, and `LLOYD_PYTHON` names this interpreter for the #1127 rung the
    wrapper runs, rather than depending on what `/usr/bin/python3` has installed.
    """
    env = {"VAULT_DIR": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin",
           "LLOYD_PYTHON": sys.executable}
    if job is not None:
        env["LLOYD_JOB"] = job
    return env


def _write(repo: Path, name: str, text: str | None = None) -> Path:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or f"written by this test: {name}\n", encoding="utf-8")
    return path


@pytest.fixture
def vault_repo(tmp_path):
    """A real git repo on `main` holding one committed file, so "a path the commit
    did not touch" exists to be distinguished from "a path it carried"."""
    repo = tmp_path / "vault"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "alan@example.com")
    _git(repo, "config", "user.name", "alansrobotlab")
    _write(repo, "seed.md", "committed before the run\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _run_wrapper(repo: Path, msg: str, *, paths: list[str] | None = None,
                 job: str | None = None) -> subprocess.CompletedProcess:
    """Run the real wrapper as a job would. `paths=None` means the invocation has
    no `--` at all, which is the eight legacy call sites' shape."""
    argv = ["bash", str(WRAPPER), msg]
    if paths is not None:
        argv.append("--")
        argv.extend(paths)
    return subprocess.run(argv, capture_output=True, text=True, timeout=120,
                          cwd=str(repo), env=_env(repo, job))


def _commit_paths(repo: Path, sha: str = "HEAD") -> list[str]:
    out = _git(repo, "show", "--stat", "--format=", "--name-only", sha).stdout
    return [line for line in out.splitlines() if line.strip()]


def _body(repo: Path) -> str:
    return _git(repo, "log", "-1", "--format=%B").stdout


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _paths_in(block: str) -> list[str]:
    """The path lines of an `unattributed dirty state:` block: indented entries
    under the header, which is the only shape the wrapper emits."""
    return [line.strip() for line in block.splitlines()
            if line.startswith("    ") and line.strip()]


def _dirty_paths(repo: Path) -> list[str]:
    """Every path the working tree holds uncommitted, path field only (porcelain v1
    prefixes each record with two status columns and a space)."""
    out = _git(repo, "status", "--porcelain=v1", "--untracked-files=all",
               "--no-renames").stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Clause 1: `vault-commit.sh "msg" -- <path>…` commits exactly those paths.
# ---------------------------------------------------------------------------

def test_a_pathspec_commit_carries_the_named_path_and_none_of_the_other(vault_repo):
    """The clause's own scenario: two paths dirty in one repo, one named."""
    _write(vault_repo, "memory/mine.md")
    _write(vault_repo, "knowledge/theirs.md")
    proc = _run_wrapper(vault_repo, "job: my own writes", paths=["memory/"])
    assert proc.returncode == 0, proc.stderr
    assert _commit_paths(vault_repo) == ["memory/mine.md"], _body(vault_repo)
    # The un-named path is not in the commit AND is still dirty afterwards: the
    # wrapper must leave it for its own writer, not absorb it and not discard it.
    assert "knowledge/theirs.md" in _dirty_paths(vault_repo), _dirty_paths(vault_repo)


def test_a_named_path_the_job_did_not_write_is_skipped_and_not_fatal(vault_repo):
    """A job names every segment it writes and usually writes two of them.

    `git add -A -- <path>` is fatal on a pathspec that matches nothing ("fatal:
    pathspec 'nope/' did not match any files", exit 128), so a wrapper that passed
    the caller's list straight through would turn a quiet night — the common case
    for a nightly job — into a failed commit step that strands every file the job
    did write. The named-but-absent directory below is the case a skill hits
    literally, since `projects/` is not in this fixture at all.
    """
    _write(vault_repo, "memory/one.md")
    proc = _run_wrapper(vault_repo, "job: writes",
                        paths=["memory/", "lloyd/", "people/", "projects/"])
    assert proc.returncode == 0, proc.stderr
    assert _commit_paths(vault_repo) == ["memory/one.md"]


def test_a_pathspec_matching_nothing_creates_no_commit(vault_repo):
    """`-- <paths>` where none of them changed: exit 0, no commit, nothing staged.

    This is the "nothing to commit" path a pre-flight-style caller already relies
    on, reached through the new argument form instead of a clean tree.
    """
    _write(vault_repo, "knowledge/theirs.md")
    before = _head(vault_repo)
    proc = _run_wrapper(vault_repo, "job: nothing here", paths=["lloyd/"])
    assert proc.returncode == 0, proc.stderr
    assert _head(vault_repo) == before
    assert _dirty_paths(vault_repo) == ["knowledge/theirs.md"], _dirty_paths(vault_repo)


def test_a_named_file_owns_itself_and_none_of_its_directory(vault_repo):
    """Naming one file claims that file and nothing else: a sibling in the same
    directory is neither committed nor absorbed, so a job that writes one line of
    a shared file cannot also take responsibility for what a neighbour put in the
    same directory."""
    _write(vault_repo, "memory/audit/writes.jsonl")
    _write(vault_repo, "memory/audit/other.jsonl")
    proc = _run_wrapper(vault_repo, "job: one file",
                        paths=["memory/audit/writes.jsonl"])
    assert proc.returncode == 0, proc.stderr
    assert _commit_paths(vault_repo) == ["memory/audit/writes.jsonl"]
    assert UNATTRIBUTED not in _body(vault_repo), (
        "a commit holding only the named path carries nothing to disclose")
    assert _dirty_paths(vault_repo) == ["memory/audit/other.jsonl"], (
        _dirty_paths(vault_repo))


def test_a_pathspec_that_names_the_whole_tree_is_refused(vault_repo):
    """`-- .` is the unscoped commit with a pathspec's appearance, and the
    appearance is what a later reader would trust. The wrapper has to answer it the
    same way it answers an argument it cannot interpret."""
    _write(vault_repo, "memory/mine.md")
    before = _head(vault_repo)
    for spec in (".", "./", "*", "/memory"):
        proc = _run_wrapper(vault_repo, "job: whole tree claimed", paths=[spec])
        assert proc.returncode == 3, f"{spec!r}: {proc.stderr}"
        assert _head(vault_repo) == before
    assert _dirty_paths(vault_repo) == ["memory/mine.md"]


def test_a_second_argument_that_is_not_a_pathspec_separator_is_refused(vault_repo):
    """A pathspec the wrapper silently ignores is the original defect wearing the
    fix's clothes: the commit would still be whole-tree under the job's name. So an
    unexpected second argument is an invocation error, not a no-op.
    """
    _write(vault_repo, "memory/mine.md")
    before = _head(vault_repo)
    proc = subprocess.run(("bash", str(WRAPPER), "job: msg", "memory/mine.md"),
                          capture_output=True, text=True, timeout=120,
                          cwd=str(vault_repo), env=_env(vault_repo))
    assert proc.returncode == 3, proc.stderr
    assert _head(vault_repo) == before


# ---------------------------------------------------------------------------
# Clause 2: with no pathspec, the commit says so and names what it carries.
# ---------------------------------------------------------------------------

def test_without_a_pathspec_the_commit_body_names_every_path_it_carries(vault_repo):
    """`git log --format=%B` must answer "not this job" for each carried path."""
    _write(vault_repo, "lloyd/MEMORY.md")
    _write(vault_repo, "autonomy/51-x.md")
    proc = _run_wrapper(vault_repo, "nightly-reflection: pre-flight 2026-09-21")
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert UNATTRIBUTED in body, body
    carried = _paths_in(body.split(UNATTRIBUTED, 1)[1])
    assert sorted(carried) == ["autonomy/51-x.md", "lloyd/MEMORY.md"], body
    # `seed.md` is in the repo and untouched: naming it would make the list a
    # listing of the repository rather than of the commit.
    assert "seed.md" not in carried


def test_the_unattributed_list_is_printed_to_stderr_as_well(vault_repo):
    """The job cannot read its own commit body from the terminal it is running in,
    and clause 5 makes it copy the list into its run record. That is only possible
    if the same list is on stderr, so the two lists are compared to each other
    rather than each to a fixture-shaped constant.
    """
    _write(vault_repo, "lloyd/MEMORY.md")
    _write(vault_repo, "memory/audit/writes.jsonl")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight")
    assert proc.returncode == 0, proc.stderr
    assert UNATTRIBUTED in proc.stderr, proc.stderr
    body = _body(vault_repo)
    listed = _paths_in(proc.stderr.split(UNATTRIBUTED, 1)[1])
    assert sorted(listed) == ["lloyd/MEMORY.md", "memory/audit/writes.jsonl"], proc.stderr
    assert sorted(listed) == sorted(_paths_in(body.split(UNATTRIBUTED, 1)[1]))


def test_a_pathspec_commit_does_not_claim_unattributed_state(vault_repo):
    """The block is conditional, and this is the assertion that makes the previous
    two tests able to fail: a wrapper that always emitted the block would satisfy
    clause 2 and tell the reader nothing."""
    _write(vault_repo, "memory/mine.md")
    _write(vault_repo, "knowledge/theirs.md")
    proc = _run_wrapper(vault_repo, "job: scoped", paths=["memory/"])
    assert proc.returncode == 0, proc.stderr
    assert UNATTRIBUTED not in _body(vault_repo), _body(vault_repo)
    assert UNATTRIBUTED not in proc.stderr, proc.stderr


def test_a_pathspec_commit_still_names_state_that_was_already_staged(vault_repo):
    """`git commit` commits the index, not just what this call added.

    If another writer left something staged, a path-scoped invocation still ships
    it, and the honest answer is the same block naming it — the clause is about
    what the message claims, not about which argument form was used.
    """
    _write(vault_repo, "backlog/1070-x.md")
    _git(vault_repo, "add", "backlog/1070-x.md")
    _write(vault_repo, "memory/mine.md")
    proc = _run_wrapper(vault_repo, "job: scoped", paths=["memory/"])
    assert proc.returncode == 0, proc.stderr
    committed = _commit_paths(vault_repo)
    assert "backlog/1070-x.md" in committed, committed  # the hazard is real
    body = _body(vault_repo)
    assert UNATTRIBUTED in body, body
    assert "backlog/1070-x.md" in _paths_in(body.split(UNATTRIBUTED, 1)[1]), body
    assert "memory/mine.md" not in _paths_in(body.split(UNATTRIBUTED, 1)[1]), body


# ---------------------------------------------------------------------------
# Clause 3: what the wrapper already guaranteed, and must still guarantee.
# ---------------------------------------------------------------------------

def test_the_wrapper_is_executable_not_just_runnable_through_bash():
    """Every skill invokes the wrapper as `~/lloyd/scripts/util/vault-commit.sh "…"`,
    which needs the mode bit; running it as `bash <path>` in a test hides a rewrite
    that dropped it. `git diff --summary` is what says so: it prints
    `mode change 100755 => 100644` for the same defect."""
    assert WRAPPER.stat().st_mode & 0o111, oct(WRAPPER.stat().st_mode)


def test_a_clean_tree_still_exits_zero_without_creating_a_commit(vault_repo):
    before = _head(vault_repo)
    proc = _run_wrapper(vault_repo, "vault-maintenance: pre-run snapshot")
    assert proc.returncode == 0, proc.stderr
    assert _head(vault_repo) == before
    assert "nothing to commit" in proc.stderr, proc.stderr


def test_a_non_main_head_is_forced_back_to_main_before_the_commit_lands(vault_repo):
    """The #341 guard, re-pinned after the rewrite. It is the reason a stranded
    `experiment-*` HEAD stops being a data-loss event, and it sits three lines
    above the staging step this item changed."""
    _write(vault_repo, "memory/mine.md")
    _git(vault_repo, "checkout", "-q", "-b", "experiment-accuracy")
    proc = _run_wrapper(vault_repo, "job: after a stray branch")
    assert proc.returncode == 0, proc.stderr
    assert _git(vault_repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    assert "Forcing checkout to main" in proc.stderr, proc.stderr
    assert _commit_paths(vault_repo) == ["memory/mine.md"]


def test_the_job_identity_and_the_status_rung_survive_the_new_commit_step(vault_repo):
    """One seam the fix itself crosses: the commit is now built from two `-m`
    arguments instead of one, and #668's `--trailer` plus #1127's staged-tree rung
    hang off that same command.

    A second `-m` placed after `--trailer`, or a reordered command, loses one of
    them silently: the trailer would sit above the block and a reader who greps
    the last paragraph for `Job:` finds nothing, and the rung's finding is the
    only trace a dispatch-killing `status` flip leaves.
    """
    task = vault_repo / "autonomy" / "68-morning-brief-triage.md"
    task.parent.mkdir(exist_ok=True)
    task.write_text("---\ntype: autonomy\nid: 68\nname: Morning Brief Triage\n"
                    "status: up_next\nfrequency: every-15min\n---\n\nbody\n",
                    encoding="utf-8")
    _git(vault_repo, "add", "-A")
    _git(vault_repo, "commit", "-qm", "task 68 exists and is dispatchable")
    # An *added* task file is never a transition, so the flip has to be a
    # modification of a file HEAD already holds — otherwise this test would be
    # asserting that the rung fires on a shape it deliberately ignores.
    task.write_text("---\ntype: autonomy\nid: 68\nname: Morning Brief Triage\n"
                    "status: draft\nfrequency: every-15min\n---\n\nbody\n",
                    encoding="utf-8")
    _write(vault_repo, "memory/mine.md")
    proc = _run_wrapper(vault_repo, "nightly: knowledge write 2026-09-21", job=JOB)
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert f"Job: {JOB}" in body, body
    assert UNATTRIBUTED in body, body
    assert "autonomy-status FINDING:" in proc.stdout + proc.stderr, proc.stdout + proc.stderr
    assert "status up_next -> draft" in proc.stdout + proc.stderr, proc.stdout + proc.stderr
    author = _git(vault_repo, "log", "-1", "--format=%ae").stdout.strip()
    assert author == f"{JOB}@jobs.lloyd.local", author


# ---------------------------------------------------------------------------
# Clause 4: the eight call sites in the seven skills.
# ---------------------------------------------------------------------------

def _invocations() -> list[tuple[Path, int, str]]:
    """Every non-comment line in a live skill that reaches the wrapper."""
    found = []
    for skill in sorted(VAULT_SKILLS.glob("*/SKILL.md")):
        for n, line in enumerate(skill.read_text(encoding="utf-8").splitlines(), 1):
            if "scripts/util/vault-commit.sh" in line and not line.strip().startswith("#"):
                found.append((skill, n, line.strip()))
    return found


@pytest.mark.live_vault
def test_every_wrapper_call_site_in_a_skill_scopes_its_commit_or_labels_it():
    """Clause 4, graded against the live vault.

    Each site must do one of two things: name a pathspec, so the commit is exactly
    that job's own writes; or say in its own message that the tree may hold state
    the job did not write. What must not exist is a site whose commit reads as the
    job's authorship while `git add -A` sweeps someone else's work into it — which
    is what every one of the eleven sites looked like before this change.
    """
    seen = _invocations()
    assert len(seen) >= 11, f"expected the documented call sites, found {len(seen)}"
    unlabelled = []
    for skill, n, line in seen:
        if " -- " in line or " --\n" in line:
            continue
        if "unattributed" in line.lower():
            continue
        unlabelled.append(f"{skill.parent.name}/SKILL.md:{n}: {line}")
    assert not unlabelled, "call sites committing under a bare job name:\n" + "\n".join(unlabelled)


@pytest.mark.live_vault
def test_the_scoped_call_sites_execute_and_leave_the_rest_of_the_tree_alone(tmp_path):
    """Clause 4's mechanism, executed rather than read.

    A pathspec in prose is worth nothing if the line does not parse — a continuation
    slipped in the wrong place turns the job's commit step into a shell error at
    02:00 with the run's output left unstaged, which is the #341 failure this wrapper
    exists to prevent. `autonomy-data-pipeline`'s post-flight invocation is therefore
    taken verbatim out of the live skill and run against a fixture vault.

    The other writer in the fixture is `.obsidian/workspace.json`, which Obsidian
    itself rewrites continuously and no nightly job is a writer of — so it is the
    path whose survival in the tree, uncommitted, is the thing a scoped commit has to
    guarantee.
    """
    skill = VAULT_SKILLS / "autonomy-data-pipeline" / "SKILL.md"
    line = next((ln.strip() for ln in skill.read_text(encoding="utf-8").splitlines()
                 if "scripts/util/vault-commit.sh" in ln and " -- " in ln), None)
    assert line is not None, "post-flight step no longer passes a pathspec"
    repo = tmp_path / "vault"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "alan@example.com")
    _git(repo, "config", "user.name", "alansrobotlab")
    _write(repo, "seed.md")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _write(repo, "memory/audit/writes.jsonl")
    _write(repo, ".obsidian/workspace.json")

    command = line.replace("~/lloyd/scripts/util/vault-commit.sh", str(WRAPPER))
    proc = subprocess.run(("bash", "-c", command), capture_output=True, text=True,
                          timeout=120, cwd=str(repo), env=_env(repo))
    assert proc.returncode == 0, proc.stderr
    assert _commit_paths(repo) == ["memory/audit/writes.jsonl"], proc.stderr
    assert ".obsidian/workspace.json" in _dirty_paths(repo), _dirty_paths(repo)
    assert UNATTRIBUTED not in _body(repo), _body(repo)


# ---------------------------------------------------------------------------
# Clause 5: the warning in nightly-reflection-signals becomes an instruction.
# ---------------------------------------------------------------------------

@pytest.mark.live_vault
def test_the_signals_skill_tells_the_job_to_copy_the_list_into_its_run_record():
    """Clause 5. `skills/nightly-reflection-signals/SKILL.md` already *described*
    the whole-tree commit as a "prose warning" while its own pre-flight step kept
    doing it, which is how a job report could stay silent about a change its commit
    was showing. The section must now instruct the job to carry the wrapper's exact
    list into the run record it writes.
    """
    text = (VAULT_SKILLS / "nightly-reflection-signals" / "SKILL.md").read_text(
        encoding="utf-8")
    phase = text.split("## Phase 1: Pre-Flight Snapshots", 1)
    assert len(phase) == 2, "the pre-flight section is gone; clause 5 has no target"
    section = phase[1].split("## Phase 2", 1)[0]
    assert UNATTRIBUTED in section, section
    assert "run record" in section, section
    assert "MANDATORY" in section, section
    # The old form described the behaviour passively; the instruction is the fix,
    # so the imperative verb is asserted rather than the topic.
    assert "Copy" in section, section
