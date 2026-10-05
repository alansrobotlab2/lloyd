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
3. **A declared write list** (#1867). `LLOYD_JOB_WRITES="a.md:b.md"` alongside
   `LLOYD_JOB` and a pathspec makes ownership mean *authorship* instead of
   *vicinity*: the block then names every carried path that is not on the list.
   Without it, ownership stays "under one of the named paths", which is what a
   job that names whole segments — `-- memory/ backlog/ …` — silently turns into
   a claim on whatever else was dirty in those segments. Opt-in: unset, both
   modes behave exactly as item 1 and item 2 describe.

The wrapper is one script whose commit step now also carries #341's branch guard,
#668's job identity (`LLOYD_JOB` → author + `Job:` trailer), #1127's autonomy-status
rung and #1867's declared-write ownership. Rewriting the staging/commit step can
silently drop any of them, so each is pinned here too rather than assumed.

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
import re
import subprocess
import sys
import time
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


def _env(repo: Path, job: str | None = None,
         writes: str | None = None) -> dict:
    """A minimal env: no HOME, so no `~/.gitconfig` leaks an identity into a
    fixture, and `LLOYD_PYTHON` names this interpreter for the #1127 rung the
    wrapper runs, rather than depending on what `/usr/bin/python3` has installed.

    `writes` is #1867's declared write list. `None` leaves the variable out of the
    environment, which is the unset case every call site has today; `""` is the
    distinct case of a job that set it from a variable that was not itself set.
    """
    env = {"VAULT_DIR": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin",
           "LLOYD_PYTHON": sys.executable}
    if job is not None:
        env["LLOYD_JOB"] = job
    if writes is not None:
        env["LLOYD_JOB_WRITES"] = writes
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
                 job: str | None = None,
                 writes: str | None = None) -> subprocess.CompletedProcess:
    """Run the real wrapper as a job would. `paths=None` means the invocation has
    no `--` at all, which is the eight legacy call sites' shape. `writes` is the
    declared write list (#1867): `None` leaves `LLOYD_JOB_WRITES` out of the
    environment entirely, which is the unset case the other tests run in."""
    argv = ["bash", str(WRAPPER), msg]
    if paths is not None:
        argv.append("--")
        argv.extend(paths)
    return subprocess.run(argv, capture_output=True, text=True, timeout=120,
                          cwd=str(repo), env=_env(repo, job, writes))


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
# #2245: another writer's already-staged deletion must not cost this commit.
#
# The staging step builds its path list from
# `git status --porcelain=v1 --untracked-files=all --no-renames` and hands it to
# `git add -A --`. `--no-renames` reports a staged rename as its SOURCE with a
# staged-deletion status (`D  p`), and `git add -A -- p` on an absent `p` is fatal:
# "fatal: pathspec 'p' did not match any files", rc 128, which under
# `set -euo pipefail` aborts the wrapper with no commit at all. That is what happened
# at 15:52Z on 2026-10-05: another job's staged `git mv` of the promotions ledger to
# a dated witness name. The rename's SOURCE sat under `backlog/`, so it was inside
# task #24's own pathspec too — and so the
# pre-flight rollback snapshot and the post-flight commit both died, for ~20 minutes,
# until the owning job committed its rename (the vault's `7786c598`). The fallback
# the run fell back to, `842ad773`, bypassed this wrapper's attribution entirely.
#
# These are the three shapes the fix has to separate: a deletion ALREADY IN THE
# INDEX whose path is gone (skip — the index already says it), a deletion only in the
# worktree (add it — `git add -A` succeeds because the pathspec matches the index
# entry), and a pathspec that merely holds no change (skip it, which the wrapper has
# done since #1070 and must keep doing).
# ---------------------------------------------------------------------------

#: The pair the fixture renames, in the shape the incident's pair had. The real pair
#: is the retired promotions mirror and its dated witness name — vault `7786c598`
#: landed it — and this file cites that sha rather than the path, because
#: tests/test_automod_vault_round.py refuses a new file that names the mirror.
LEDGER = "backlog/data/ledger.jsonl"
WITNESS = "backlog/data/2026-10-05.2216-witness.jsonl"


def _stage_rename(vault_repo: Path) -> None:
    """Commit one file, then `git mv` it and leave the rename UNCOMMITTED, which is
    the state that blocked task #24."""
    _write(vault_repo, LEDGER, "board,at,round\nlloyd,2026-10-04,SM_x\n")
    _git(vault_repo, "add", LEDGER)
    _git(vault_repo, "commit", "-qm", "the promotions ledger exists at the old path")
    _git(vault_repo, "mv", LEDGER, WITNESS)


def test_a_staged_rename_left_by_another_writer_does_not_stop_the_commit(vault_repo):
    """The item's clause 1, and the incident itself: the rename is the ONLY thing
    dirty, and the invocation is the pre-flight one: no pathspec.

    Before the fix this exits 128 and commits nothing, which is how a job lost its
    rollback snapshot: the `git status` read the staging step scans reports the
    rename's SOURCE as `D  <ledger's old path>`, that path is not in the
    worktree, and `git add -A` refuses to name it. The assertion is not merely that
    the wrapper survived: the rename has to be IN the commit, because the index is
    what `git commit` reads and a wrapper that skipped the whole thing would also
    exit 0.
    """
    _stage_rename(vault_repo)
    before = _head(vault_repo)
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-10-05")
    assert proc.returncode == 0, (
        f"another writer's staged rename cost this commit again: {proc.returncode} "
        f"{proc.stderr}")
    assert _head(vault_repo) != before, (
        f"exit 0 with no commit is the other way to fail this clause: {proc.stderr}")
    assert _git(vault_repo, "status", "--porcelain=v1",
                "--untracked-files=all").stdout.strip() == "", (
        "the rename this invocation swept up is still uncommitted afterwards")
    body = _body(vault_repo)
    assert UNATTRIBUTED in body, body
    assert WITNESS in _paths_in(body.split(UNATTRIBUTED, 1)[1]), body


def test_a_lone_staged_deletion_with_nothing_else_dirty_still_lands_a_commit(
        vault_repo):
    """The same index state with no rename destination to add, which is the shape
    that turns the fix into a silent no-op if it is written carelessly.

    A `git rm` staged and left uncommitted by a crashed run makes the skipped record
    the tree's ONLY record. Dropping it from the `git add` list is right, but a fix
    that then falls through to the wrapper's "nothing to commit (clean tree on main)"
    fast exit would exit 0, commit nothing, and tell the reader the tree was clean
    when the index plainly is not. The deletion has to reach a commit.
    """
    _write(vault_repo, LEDGER, "board,at,round\n")
    _git(vault_repo, "add", LEDGER)
    _git(vault_repo, "commit", "-qm", "the promotions ledger exists at the old path")
    _git(vault_repo, "rm", "-q", LEDGER)
    before = _head(vault_repo)
    proc = _run_wrapper(vault_repo, "vault-maintenance: pre-run snapshot")
    assert proc.returncode == 0, proc.stderr
    assert "nothing to commit" not in proc.stderr, (
        f"a tree with a staged deletion is not clean: {proc.stderr}")
    assert _head(vault_repo) != before, proc.stderr
    assert _git(vault_repo, "show", "--format=", "--name-only", "HEAD").stdout.split() \
        == [LEDGER], _commit_paths(vault_repo)
    assert _git(vault_repo, "status", "--porcelain=v1").stdout.strip() == ""


def test_a_staged_rename_source_under_a_named_directory_does_not_stop_a_scoped_commit(
        vault_repo):
    """The item's clause 2: task #24's second dead route, the post-flight commit.

    `-- memory/ backlog/ autonomy/` is that run's literal, and the rename source sits
    under `backlog/`, so narrowing did not help — the directory pathspec pulled the
    same `D ` record into the scan and the same fatal followed. The named directory
    with nothing in it (`autonomy/`, committed and untouched here) is the other half
    of this clause: a pathspec holding no change has to stay skipped rather than
    become fatal, which is the #1070 behaviour the staging rewrite must not lose.
    """
    _stage_rename(vault_repo)
    _write(vault_repo, "memory/vault-maintenance/2026-10-05.md", "## Run 15:52Z\n")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-10-05",
                        paths=["memory/", "backlog/", "autonomy/"])
    assert proc.returncode == 0, proc.stderr
    committed = _commit_paths(vault_repo)
    assert "memory/vault-maintenance/2026-10-05.md" in committed, (
        f"the caller's own write did not land: {committed}")
    assert WITNESS in committed, (
        f"the rename the index already held is not in the commit: {committed}")


def test_a_worktree_deletion_nothing_has_staged_is_still_staged_and_committed(
        vault_repo):
    """The item's clause 3, and the trap the skip rule could walk into.

    A file deleted in the worktree with nothing staged (` D p`) is ALSO absent from
    the worktree, and `git add -A -- p` succeeds for it because the pathspec matches
    the index entry the path still has — so a filter keyed on worktree-absence alone
    would quietly stop this wrapper ever committing a deletion, and every existing
    node here would stay green doing it. The separator is the INDEX column, not the
    missing file.

    Both shapes are in one tree here, which is the case that would pass a weaker
    test: the staged rename source has to be skipped and the unstaged deletion
    staged, in the same run, into the same commit.
    """
    _stage_rename(vault_repo)
    Path(vault_repo / "seed.md").unlink()
    records = _git(vault_repo, "status", "--porcelain=v1", "--no-renames").stdout
    assert f"D  {LEDGER}" in records.splitlines(), records
    assert " D seed.md" in records.splitlines(), (
        f"the fixture is not the two-shape tree this node is about: {records}")
    proc = _run_wrapper(vault_repo, "vault-maintenance: clean up seed.md")
    assert proc.returncode == 0, proc.stderr
    committed = _git(vault_repo, "show", "--format=", "--name-status",
                     "HEAD").stdout.split()
    assert "D" in committed and "seed.md" in committed, (
        f"a worktree-only deletion was dropped from the commit: {committed}")
    assert LEDGER not in _dirty_paths(vault_repo), _dirty_paths(vault_repo)


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
# The declared write list (#1867) — `LLOYD_JOB_WRITES` turns "under a path I
# named" into "a file I wrote".
# ---------------------------------------------------------------------------

#: The 2026-09-30 01:17Z autonomy-data-pipeline run in fixture form. It ran the
#: Post-Flight literal `-- memory/ knowledge/ backlog/ autonomy/` and produced
#: commit 441c35d9: its own run-log note, another job's in-flight edit to a
#: backlog item (95 insertions), and the audit logger's append. All three lie
#: inside a named segment, so #1070's ownership rule called all three this job's
#: own and printed nothing — while the same run's no-`--` pre-flight commit,
#: 4e2fd7b9, printed the block for all 15 paths it carried.
OWN_NOTE = "memory/vault-maintenance/2026-09-30.md"
FOREIGN_ITEM = "backlog/1866-item.md"
AUDIT_APPEND = "memory/audit/writes.jsonl"

#: The Post-Flight literal's scope, verbatim.
PIPELINE_SCOPE = ["memory/", "knowledge/", "backlog/", "autonomy/"]


def _the_0117_tree(vault_repo):
    """One write this job made and two it did not, all inside the named segments."""
    _write(vault_repo, OWN_NOTE, "## Run 01:17Z\n")
    _write(vault_repo, FOREIGN_ITEM, "another job's in-flight item edit\n")
    _write(vault_repo, AUDIT_APPEND, '{"job": "audit-logger"}\n')


def test_a_declared_write_list_names_a_foreign_path_inside_a_named_directory(vault_repo):
    """The hole #1867 names, closed: a path-scoped commit carrying two paths
    absent from the declared list emits the block naming both.

    The same invocation was silent before this variable existed, because both
    foreign paths lie under a segment the caller named and #1070 compared
    ownership against the *pathspec*, never against what the job wrote.
    """
    _the_0117_tree(vault_repo)
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-09-30",
                        paths=PIPELINE_SCOPE, job="autonomy-data-pipeline",
                        writes=OWN_NOTE)
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert UNATTRIBUTED in body, (
        "the commit carried a backlog item another job was editing and the audit "
        "logger's append and said nothing — the silent attribution 441c35d9 made")
    listed = _paths_in(body.split(UNATTRIBUTED, 1)[1])
    assert sorted(listed) == sorted([FOREIGN_ITEM, AUDIT_APPEND]), listed
    assert OWN_NOTE not in listed, (
        f"the one path this job did write was named as unattributed: {listed}")
    for path in (FOREIGN_ITEM, AUDIT_APPEND):
        assert path in proc.stderr, (
            f"{path} is in the commit body but not on stderr, and a job cannot read "
            f"its own commit body from where it is running")
    assert "Job: autonomy-data-pipeline" in body, (
        f"the block arrived by way of dropping #668's trailer: {body}")


def test_the_block_lists_exactly_the_committed_paths_that_are_not_declared(vault_repo):
    """The declared half of the rule: a path on the list is never named, and an
    entry may be a directory as well as a file.

    `memory/audit/` is declared as a directory here, which is how a job that owns a
    subtree says so — the entry has to cover the paths *under* it, not merely equal
    one. Everything declared stays out of the block, so what is left is exactly what
    the job cannot show it wrote.
    """
    _the_0117_tree(vault_repo)
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-09-30",
                        paths=PIPELINE_SCOPE, job="autonomy-data-pipeline",
                        writes=f"{OWN_NOTE}:memory/audit/")
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert UNATTRIBUTED in body, proc.stderr
    assert _paths_in(body.split(UNATTRIBUTED, 1)[1]) == [FOREIGN_ITEM], body
    assert sorted(_commit_paths(vault_repo)) == sorted([FOREIGN_ITEM, OWN_NOTE,
                                                        AUDIT_APPEND]), (
        "the commit is still the scoped sweep it always was — the declared list "
        "changes what is reported, not what is staged")


def test_a_job_that_declared_every_path_it_carried_commits_without_a_word(vault_repo):
    """The other side of the same rule, and the reason this is not a tax: a run
    that names its own writes and carries nothing else commits with no block in the
    body and nothing on stderr."""
    _write(vault_repo, OWN_NOTE, "## Run 01:17Z\n")
    _write(vault_repo, "knowledge/mine.md")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-09-30",
                        paths=PIPELINE_SCOPE, job="autonomy-data-pipeline",
                        writes=f"{OWN_NOTE}:knowledge/mine.md")
    assert proc.returncode == 0, proc.stderr
    assert sorted(_commit_paths(vault_repo)) == sorted([OWN_NOTE,
                                                        "knowledge/mine.md"])
    body = _body(vault_repo)
    assert UNATTRIBUTED not in body, body
    assert UNATTRIBUTED not in proc.stderr, proc.stderr


def test_the_declared_list_splits_on_newlines_and_ignores_padding(vault_repo):
    """The one syntactic claim the header makes: `:` and newline are both
    separators, and whitespace around a separator is not part of a path.

    Worth a node because the failure is quiet — an entry read as
    `" memory/audit/"` matches nothing, so a job that wrote its list with padding
    would find its own file reported as unattributed.
    """
    _the_0117_tree(vault_repo)
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-09-30",
                        paths=PIPELINE_SCOPE, job="autonomy-data-pipeline",
                        writes=f"{OWN_NOTE}\n memory/audit/ ")
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert UNATTRIBUTED in body, proc.stderr
    assert _paths_in(body.split(UNATTRIBUTED, 1)[1]) == [FOREIGN_ITEM], body


def test_an_unset_or_empty_declared_list_leaves_a_pathspec_asking_the_old_question(
        vault_repo):
    """Opt-in, in both spellings of it. With the variable absent — every human
    commit, and every call site that has not adopted it — and with it present but
    empty (`LLOYD_JOB_WRITES="$MAYBE_UNSET_VAR"`, which is how a run forgets),
    ownership is still "under one of the named paths" and a foreign path inside a
    named directory stays unreported. That is the behaviour
    `test_a_pathspec_commit_does_not_claim_unattributed_state` pins, re-pinned here
    with `LLOYD_JOB` set, which that test does not exercise.
    """
    _the_0117_tree(vault_repo)
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-09-30",
                        paths=PIPELINE_SCOPE, job="autonomy-data-pipeline")
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert UNATTRIBUTED not in body, body
    assert UNATTRIBUTED not in proc.stderr, proc.stderr
    assert len(_commit_paths(vault_repo)) == 3, "the scoped sweep still swept"

    _write(vault_repo, "knowledge/empty-list.md")
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-09-30 (empty list)",
                        paths=PIPELINE_SCOPE, job="autonomy-data-pipeline",
                        writes="")
    assert proc.returncode == 0, proc.stderr
    assert _commit_paths(vault_repo) == ["knowledge/empty-list.md"]
    assert UNATTRIBUTED not in _body(vault_repo), (
        "an empty declared list was read as 'this job wrote nothing', which would "
        "name the committer's own file as unattributed: " + _body(vault_repo))


def test_the_sweep_still_names_every_carried_path_when_nothing_is_declared(vault_repo):
    """Clause 3's other mode, with a job name attached (the pre-existing sweep test
    runs without one): no pathspec and no declared list is still #1070's honest
    snapshot, naming all three paths."""
    _the_0117_tree(vault_repo)
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-09-30",
                        job="autonomy-data-pipeline")
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert UNATTRIBUTED in body, proc.stderr
    listed = _paths_in(body.split(UNATTRIBUTED, 1)[1])
    assert sorted(listed) == sorted([OWN_NOTE, FOREIGN_ITEM, AUDIT_APPEND]), listed


def test_the_unset_block_keeps_the_sentence_1070_wrote_in_both_modes(vault_repo):
    """The exact-sentence half of clause 3. "Behaves exactly as today" is a claim
    about bytes, not about counts: eight call sites' run records quote these two
    sentences, and the sweep prose is what a job is told to copy. So the whole
    sentences are pinned here, with `LLOYD_JOB` set — the case the pre-existing
    tests do not run, and the only one in which a new "declared write list" branch
    could have reached in and reworded them."""
    _the_0117_tree(vault_repo)
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-09-30",
                        job="autonomy-data-pipeline")
    assert proc.returncode == 0, proc.stderr
    assert ("unattributed dirty state: 3 path(s) in this commit that this invocation "
            "did not name as its own writes." in _body(vault_repo)
            ), _body(vault_repo)
    assert ("unattributed dirty state: 3 path(s) this commit carries that this "
            "invocation did not name — copy this exact list" in proc.stderr), proc.stderr
    assert "LLOYD_JOB_WRITES" not in proc.stderr, (
        "the unset sweep was told about a variable it never set: " + proc.stderr)


def test_a_declared_list_without_a_pathspec_does_not_quiet_the_sweep(vault_repo):
    """The declared list is scoped to the mode that needs it. With no `--`, the
    block keeps naming every carried path *including* the declared one: a
    pre-flight snapshot cannot buy back authorship by declaring, and a run that
    sets the variable and forgets the pathspec has to end up over-disclosing, never
    under."""
    _the_0117_tree(vault_repo)
    proc = _run_wrapper(vault_repo, "autonomy-data-pipeline: pre-flight 2026-09-30",
                        job="autonomy-data-pipeline", writes=OWN_NOTE)
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert UNATTRIBUTED in body, proc.stderr
    listed = _paths_in(body.split(UNATTRIBUTED, 1)[1])
    assert OWN_NOTE in listed, (
        "a declaration silenced the sweep for its own path; with no pathspec there "
        f"is nothing to scope, so the block must name all three: {listed}")
    assert sorted(listed) == sorted([OWN_NOTE, FOREIGN_ITEM, AUDIT_APPEND]), listed


def test_a_declared_list_is_ignored_when_the_invocation_has_no_job_name(vault_repo):
    """An invocation that does not say which job it is has no own writes to check
    against, so the list is inert and the commit stays the #1070 unattributed one.
    Honouring a declaration from an anonymous caller would hand any commit the power
    to silence the guard that exists to interrogate it."""
    _the_0117_tree(vault_repo)
    proc = _run_wrapper(vault_repo, "somebody: 2026-09-30",
                        paths=PIPELINE_SCOPE, writes=OWN_NOTE)
    assert proc.returncode == 0, proc.stderr
    body = _body(vault_repo)
    assert UNATTRIBUTED not in body, body
    assert "Job:" not in body, body


def test_a_declared_entry_that_is_not_a_scoped_path_is_refused_as_an_invocation_error(
        vault_repo):
    """`LLOYD_JOB_WRITES=.` claims every path in the vault as this job's own work —
    the same claim `-- .` is refused for, by the same shape of rail, and here it is
    refused by the guard it would disarm. So is a glob, an absolute path and a `..`,
    because a declared entry that matches no staged path is indistinguishable, in a
    commit body, from a job that declared nothing. Exit 3, named by variable, nothing
    committed."""
    _the_0117_tree(vault_repo)
    before = _head(vault_repo)
    for claim in (".", "*", "memory/.."):
        proc = _run_wrapper(vault_repo, "job: claiming the vault",
                            paths=PIPELINE_SCOPE, job="autonomy-data-pipeline",
                            writes=claim)
        assert proc.returncode == 3, f"'{claim}': {proc.stderr}"
        assert "LLOYD_JOB_WRITES" in proc.stderr, proc.stderr
    assert _head(vault_repo) == before
    assert len(_dirty_paths(vault_repo)) == 3, (
        f"the refusal left the tree it refused to commit in another state: "
        f"{_dirty_paths(vault_repo)}")


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
def test_the_pipeline_post_flight_step_declares_the_writes_it_is_committing():
    """Clause 4, read off the live skill: the Post-Flight step names its own writes
    alongside the directory scope, not just the scope.

    The directory form alone is what let the 2026-09-30 01:17Z run commit another
    job's `backlog/1866-*.md` edit and the audit logger's append under
    `autonomy-data-pipeline: 2026-09-29` with nothing said: `-- memory/ knowledge/
    backlog/ autonomy/` asserts authorship of everything inside four segments. The
    run-log note is the one path every run writes, so it is the entry that has to be
    in the literal for the literal to be a declaration rather than a placeholder.
    """
    skill = VAULT_SKILLS / "autonomy-data-pipeline" / "SKILL.md"
    line = next((ln.strip() for ln in skill.read_text(encoding="utf-8").splitlines()
                 if "scripts/util/vault-commit.sh" in ln and " -- " in ln), None)
    assert line is not None, "post-flight step no longer passes a pathspec"
    assert "LLOYD_JOB_WRITES=" in line, line
    assert "memory/vault-maintenance/" in line, (
        f"the declared list does not name the run-log note, the write every run "
        f"makes, so the declaration covers nothing: {line}")
    assert "LLOYD_JOB=autonomy-data-pipeline" in line, (
        f"a declared list without a job name is inert by design: {line}")


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

    Since #1867 the same literal carries a declared list naming only the run-log
    note, so the fixture's `memory/audit/writes.jsonl` is foreign by that rule even
    though it sits in a named segment: the commit still ships it, and now has to say
    so. That is the 01:17Z run reproduced with the skill's own line.
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
    # The path the literal declares, under the same UTC date the literal expands.
    _write(repo, f"memory/vault-maintenance/{time.strftime('%Y-%m-%d', time.gmtime())}.md",
           "## Run 01:17Z\n")
    _write(repo, "memory/audit/writes.jsonl")
    _write(repo, ".obsidian/workspace.json")

    command = line.replace("~/lloyd/scripts/util/vault-commit.sh", str(WRAPPER))
    proc = subprocess.run(("bash", "-c", command), capture_output=True, text=True,
                          timeout=120, cwd=str(repo), env=_env(repo))
    assert proc.returncode == 0, proc.stderr
    assert sorted(_commit_paths(repo)) == ["memory/audit/writes.jsonl",
                                           f"memory/vault-maintenance/"
                                           f"{time.strftime('%Y-%m-%d', time.gmtime())}.md"], \
        proc.stderr
    assert ".obsidian/workspace.json" in _dirty_paths(repo), _dirty_paths(repo)
    body = _body(repo)
    assert UNATTRIBUTED in body, (
        "the pipeline's own Post-Flight line carried the audit logger's append and "
        f"reported nothing — the 01:17Z run's silent commit: {body}")
    listed = _paths_in(body.split(UNATTRIBUTED, 1)[1])
    assert listed == ["memory/audit/writes.jsonl"], listed


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


# ---------------------------------------------------------------------------
# #2184 clauses 3 and 4: the run record's own precedent lines, corrected.
# ---------------------------------------------------------------------------

VAULT = Path.home() / "obsidian"
_VAULT = str(VAULT)
def _section(text: str, when: str) -> str:
    """One `## Run <when> (task #24, data pipeline)` section of the run record."""
    head = f"## Run {when} (task #24, data pipeline)"
    parts = text.split(head, 1)
    assert len(parts) == 2, f"the {when} run section is gone"
    rest = parts[1]
    nxt = rest.find("\n## Run ")
    return rest if nxt < 0 else rest[:nxt]


VAULT_RECORD_2026_10_04 = Path.home() / "obsidian" / "memory" / "vault-maintenance" / \
    "2026-10-04.md"
#: The form three of the four 2026-10-04 run records used: whole commit object
#: (message included) grepped with a bare word list.
FLAWED_FORM = "git show --name-only | grep -iE 'MEMORY|USER|SOUL'"
#: The form #2184 puts in the wrapper: message suppressed, pattern anchored to the
#: curated filenames `app/harness/protected_paths.py` protects.
FIXED_PATTERN = r"'(^|/)lloyd/(MEMORY|USER|SOUL)\.md'"


def _no_word_list_grep(section: str, label: str) -> None:
    """Assert the section cites no message-inclusive word-list grep, however the
    line is wrapped. Comparing on collapsed whitespace is the point: the shape that
    fooled three 2026-10-04 run records is `git show --name-only | grep -iE
    'MEMORY|USER|SOUL'`, and a note may legitimately break it across two lines, which
    is exactly how a one-line substring check misses it and calls the clause green."""
    flat = " ".join(section.split())
    flawed = " ".join(FLAWED_FORM.split())
    assert flawed not in flat, f"{label}: the message-inclusive word-list grep is still cited"
    # and any *unanchored* pattern that is not accompanied by its own measured count
    for line in section.splitlines():
        if "grep -icE" in line and FIXED_PATTERN not in line:
            assert "→" in line, (
                f"{label}: an unanchored pattern is quoted without its measured count, "
                f"so a reader cannot see it is not the deciding form: {line}")


def _figure(measurements: dict[str, int], sha: str, *, pretty: bool | None = None,
            anchored: bool | None = None) -> int:
    """The stated count for the one quoted command that reads `sha`, optionally
    filtered by whether it suppresses the commit message and whether its pattern is
    path-anchored. Looking the line up by its properties rather than rebuilding its
    text is what keeps this node from failing on the note's own formatting."""
    hits = []
    for cmd, stated in measurements.items():
        if sha not in cmd:
            continue
        if pretty is not None and ("--pretty=format:" in cmd) != pretty:
            continue
        if anchored is not None and (FIXED_PATTERN in cmd) != anchored:
            continue
        hits.append((cmd, stated))
    assert len(hits) == 1, f"expected exactly one quoted command for {sha} " \
                           f"(pretty={pretty}, anchored={anchored}), got {hits}"
    return hits[0][1]


def _stated_measurements(section: str) -> list[tuple[str, int]]:
    """Every `git … | grep … → N` line a section quotes, as (command, stated count).

    The run records carry their evidence as copy-pasteable shell with the figure the
    run saw beside it. Reading the figure out of the note and re-running the command
    on the other side is the whole point of #2184: the note's numbers become
    executable, so a quoted count cannot outlive the git state that produced it."""
    out = []
    for line in section.splitlines():
        line = line.strip()
        if "→" not in line or not line.startswith("git -C"):
            continue
        cmd, _, figure = line.rpartition("→")
        out.append((cmd.strip(), int(figure.strip())))
    assert out, "the section quotes no `git … → N` measurement line"
    return out


def _run_measurements(section: str, label: str) -> None:
    """Re-run each stated measurement and fail naming the one that moved."""
    for cmd, stated in _stated_measurements(section):
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                              timeout=120)
        # `grep -c` exits 1 when it prints 0, which is this check's passing answer.
        assert proc.returncode in (0, 1), f"{label}: {cmd} → rc={proc.returncode}"
        got = int(proc.stdout.strip() or 0)
        assert got == stated, f"{label}: {cmd} → the note says {stated}, git says {got}"


@pytest.mark.live_vault
def test_the_08_56Z_record_states_the_anchored_command_and_not_the_returned_nothing_claim():
    """Clause 3. That section certified `a49a256d` with the flawed grep and described
    it as coming back empty; the same command returns 7 lines on that commit. The
    section now quotes the message-suppressed, path-anchored command, and every figure
    it quotes in that block is re-run here against the real vault repository."""
    text = VAULT_RECORD_2026_10_04.read_text(encoding="utf-8")
    section = _section(text, "08:56Z")
    assert FLAWED_FORM not in section, "the flawed grep is still the cited evidence"
    _no_word_list_grep(section, "08:56Z")
    assert "--pretty=format:" in section and FIXED_PATTERN in section, section
    _run_measurements(section, "08:56Z")
    # The block has to contain the deciding figure (0 on a49a256d) AND the positive
    # control, or the anchored pattern could be inert and the section still green.
    stated = dict(_stated_measurements(section))
    assert _figure(stated, "a49a256d", pretty=True, anchored=True) == 0, \
        "the deciding figure: the anchored, message-supplied check reports 0"
    assert _figure(stated, "a49a256d", pretty=False, anchored=False) == 7, \
        "the flawed form's own count is stated, not hidden"
    assert _figure(stated, "30dae7c6", pretty=True, anchored=True) == 1, \
        "positive control: a commit that really did edit lloyd/MEMORY.md registers 1, " \
        "so the anchored pattern is not simply inert"
    for cmd, _ in _stated_measurements(section):
        assert "a49a256d" in cmd or "30dae7c6" in cmd, \
            f"08:56Z: a quoted figure does not name the commit it measures: {cmd}"



@pytest.mark.live_vault
def test_the_13_15Z_record_states_the_measured_split_of_the_flawed_form():
    """Clause 4. Its first correction reported the flawed form's 3 lines on
    `ca546bac` as prose from the message, and that suppression alone would leave 0.
    Measured: 3 = 1 message line + 2 filenames, message-suppressed-unanchored is 2,
    anchored is 0. The section states that split and the three figures in its block
    are re-run here."""
    text = VAULT_RECORD_2026_10_04.read_text(encoding="utf-8")
    section = _section(text, "13:15Z")
    _no_word_list_grep(section, "13:15Z")
    _run_measurements(section, "13:15Z")
    stated = dict(_stated_measurements(section))
    assert _figure(stated, "ca546bac", pretty=False, anchored=False) == 3
    assert _figure(stated, "ca546bac", pretty=True, anchored=False) == 2, \
        "message suppression alone leaves 2: the anchor is load-bearing"
    assert _figure(stated, "ca546bac", pretty=True, anchored=True) == 0
    for cmd, _ in _stated_measurements(section):
        assert "ca546bac" in cmd, \
            f"13:15Z: a quoted figure does not name the commit it measures, which is " \
            f"how one section's numbers get read as another's: {cmd}"

    # The split's substance is which lines they were, so the section is pinned on the
    # filenames it enumerates (and on the three figures above), not on how it phrases
    # them: a rewording that keeps both names and all three counts is still correct.
    attribution = [f for f in ("memory/audit/writes.jsonl",
                               "memory/vault-maintenance/2026-10-04.md")
                   if f in section]
    assert len(attribution) == 2, \
        f"the 2-filename half of the split has to name them; found {attribution}"
    assert re.search(r"1 message line", section), \
        "the 1-message-line half of the split is no longer attributed to the message"

WITNESS_COPY = REPO_ROOT / "tests" / "data" / "loaded-memory-proof-2026-10-04.md"


def _corrected_proof_text() -> str:
    """The graded text #2184 clauses 3 and 4 are about, in the copy committed to THIS
    repository. The run record itself lives in the vault repository, so a reviewer
    standing in `~/lloyd` cannot read it — and the last review of this item reached
    for a `git show` of a vault file, read the commit *message* instead, and reported
    the live file as still carrying the flawed grep. These bytes exist so that
    judgment can be made on the file it names;
    `test_the_witness_copy_matches_the_vault_record` is what stops it drifting."""
    return WITNESS_COPY.read_text(encoding="utf-8")


def test_the_in_repo_witness_of_the_proof_carries_the_corrected_form_only():
    """Clauses 3 and 4, on bytes a reader of this repository actually holds.

    Every check here is textual on purpose: no vault commit is needed, so this node
    runs in the suite with no live-vault marker, and it fails if either passage ever
    cites the message-inclusive word-list grep (in any line wrapping), quotes an
    unanchored pattern without its measured count, or drops the measured split."""
    text = _corrected_proof_text()
    flat = " ".join(text.split())
    assert " ".join(FLAWED_FORM.split()) not in flat, \
        "the flawed grep is cited again, even in a note describing it"
    assert "--pretty=format:" in text and FIXED_PATTERN in text, text[:200]
    assert "1 message line" in flat and "2 of the" in flat, \
        "the 1-message-line + 2-filename split of the flawed form's 3 lines is gone"
    assert "suppressing the message would leave 0" in flat, \
        "the correction's own miscount has to stay named, or the anchor's role is lost"
    anchored = [ln for ln in text.splitlines() if "→" in ln and FIXED_PATTERN in ln]
    assert len(anchored) == 3, \
        f"want the anchored figures for a49a256d, ca546bac and the 30dae7c6 control: {anchored}"
    for line in text.splitlines():
        if "grep -icE" in line and FIXED_PATTERN not in line:
            assert "→" in line, f"unanchored form quoted without its count: {line}"


@pytest.mark.live_vault
def test_the_witness_copy_matches_the_vault_record():
    """The witness is a copy, so it is only evidence while it agrees with the record.
    Comparing the passages as text (not a hash) means a later edit to the run record
    that leaves the proof intact still fails this node, and says why: the copy is
    stale, and the numbers in `~/lloyd` are no longer the ones in `~/obsidian`."""
    live = VAULT_RECORD_2026_10_04.read_text(encoding="utf-8")
    witness = _corrected_proof_text()
    body = witness.split("-->\n", 1)[1]
    for when in ("08:56Z", "13:15Z"):
        start = body.index(f"### {when} run")
        stop = body.find("\n### ", start + 5)
        passage = body[body.index("\n\n", start) + 2:
                       len(body) if stop < 0 else stop].rstrip()
        section = _section(live, when)
        assert passage in section, (
            f"the {when} witness no longer appears verbatim in the run record; "
            f"re-cut {WITNESS_COPY.relative_to(REPO_ROOT)} from the vault file")


# ---------------------------------------------------------------------------
# #2184 clauses 1 and 2: the wrapper emits the check, and the check decides.
# ---------------------------------------------------------------------------

#: What the emitted line has to be: the commit's file list only (message
#: suppressed), anchored to the curated loaded-memory filenames, and safe to paste.
EMITTED_FLAGS = ("--name-only", "--pretty=format:", r"'(^|/)lloyd/(MEMORY|USER|SOUL)\.md'",
                 "|| true")


def _emitted_check(stderr: str) -> str:
    """The one command line the wrapper emitted on stderr, or a failure naming what
    it saw. Deliberately strict on count: two emitted checks would mean two forms in
    circulation, which is how three run records on 2026-10-04 each picked their own."""
    lines = [ln for ln in stderr.splitlines()
             if ln.startswith("git ") and "grep" in ln]
    assert len(lines) == 1, (
        f"expected exactly one emitted check line at column 0, got {lines}\n"
        f"in stderr:\n{stderr}")
    return lines[0]


def test_the_wrapper_emits_one_copyable_loaded_memory_check_beside_the_unattributed_list(
        vault_repo):
    """Clause 1. #1070 makes an unattributed commit a thing a job has to explain, and
    the run records then proved the obvious follow-on question ("is a curated
    loaded-memory file in it?") with a grep over `git show` output — whose input,
    because the wrapper deliberately writes the same path list into the commit
    message, contains the claim. The wrapper now has to supply the command that
    actually answers it, and supplying it must not disturb the block above it."""
    _write(vault_repo, "memory/audit/writes.jsonl")
    proc = _run_wrapper(vault_repo, "pipeline: snapshot", paths=None, job=JOB,
                        writes="knowledge/somebody-elses.md")
    assert proc.returncode == 0, proc.stderr
    cmd = _emitted_check(proc.stderr)
    for flag in EMITTED_FLAGS:
        assert flag in cmd, f"the emitted check is missing {flag}: {cmd}"
    # Column 0, because an indented line under the header reads as one of its path
    # entries — to `_paths_in` below, and to the run record that copies the block.
    assert cmd.startswith("git "), cmd
    assert f'git -C "{vault_repo}"' in cmd, f"the check must name this repo: {cmd}"
    # The emitted check comes BEFORE the list: #1070 makes the list stderr's tail so
    # a job can copy it verbatim, and the tail is what this file's other nodes pin.
    stderr = proc.stderr
    assert stderr.index(cmd) < stderr.index(UNATTRIBUTED), \
        "the check is printed after the list, so it is the tail a job copies instead"
    # and it does not contaminate the list it sits beside
    listed = _paths_in(stderr[stderr.index(UNATTRIBUTED):])
    assert listed == ["memory/audit/writes.jsonl"], listed
    # The line is copy-pasteable in the literal sense: it runs, under a caller's
    # `set -e`, and prints 0 here — `grep -c` exits 1 on the zero it is asked for.
    run = subprocess.run(["bash", "-c", f"set -e\n{cmd}"], capture_output=True,
                         text=True, cwd=str(vault_repo), timeout=120)
    assert run.returncode == 0, f"pasting the emitted line under `set -e` failed: {run.stderr}"
    assert run.stdout.strip() == "0", run.stdout


def test_the_emitted_check_reads_only_the_commit_file_list_the_grepped_form_cannot(
        vault_repo):
    """Clause 2, on the item's own fixture: a commit whose MESSAGE BODY names
    `lloyd/MEMORY.md` and whose DIFF carries only `memory/audit/writes.jsonl`. The
    emitted check must report 0 there (no loaded-memory path is in the commit), while
    each half of the flawed form reports at least 1 — the message because the body
    names the file, the pattern because `memory/...` contains the word `MEMORY`
    case-insensitively. Asserting the pair is the point: a check that merely never
    matches anything would pass the first half and be useless."""
    sha = "HEAD"
    _write(vault_repo, "memory/audit/writes.jsonl")
    # The body cites the curated file by its full vault-relative path, the way a real
    # run record does. That matters: `git show` indents the message body by four
    # spaces, so a bare `lloyd/MEMORY.md` in prose matches neither `^lloyd/` nor
    # `/lloyd/` and the fixture would leave the message in the input without ever
    # proving that its being there is the defect.
    proc = _run_wrapper(vault_repo,
                        "pipeline: snapshot — no curated file is in this commit; "
                        "~/obsidian/lloyd/MEMORY.md, lloyd/USER.md and lloyd/SOUL.md "
                        "are untouched by this run",
                        paths=None, job=JOB, writes="knowledge/somebody-elses.md")
    assert proc.returncode == 0, proc.stderr
    cmd = _emitted_check(proc.stderr).replace("HEAD", sha)
    repo = str(vault_repo)

    def count(*, pretty: bool, anchored: bool) -> int:
        pattern = EMITTED_FLAGS[2] if anchored else "'MEMORY|USER|SOUL'"
        line = (f"git -C '{repo}' show --name-only"
                + (" --pretty=format:" if pretty else "")
                + f" {sha} | grep -icE {pattern}")
        out = subprocess.run(["bash", "-c", line], capture_output=True, text=True,
                             timeout=120)
        assert out.returncode in (0, 1), out.stderr
        return int(out.stdout.strip() or 0)

    # the fixture is the fixture only if the body really names the file and the diff
    # really carries only the audit log
    assert "lloyd/MEMORY.md" in _body(vault_repo), "the message must name the file"
    assert _commit_paths(vault_repo, sha) == ["memory/audit/writes.jsonl"], \
        _commit_paths(vault_repo, sha)
    emitted = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                             timeout=120)
    assert emitted.returncode in (0, 1), emitted.stderr
    assert emitted.stdout.strip() == "0", (
        f"the emitted check reports a loaded-memory path in a commit that has none: "
        f"{emitted.stdout!r}")
    # each flawed half on its own still fires on this commit
    assert count(pretty=False, anchored=False) >= 1, (
        "the message-inclusive form was expected to fire on a body naming "
        "lloyd/MEMORY.md; if it does not, the fixture stopped exercising the defect")
    assert count(pretty=True, anchored=False) >= 1, (
        "the message-suppressed unanchored form was expected to fire on the "
        "memory/audit/writes.jsonl filename; if it does not, the fixture is inert")
    # Suppressing the message is load-bearing even with the anchor in place: the body
    # cites `~/obsidian/lloyd/MEMORY.md`, and `/lloyd/MEMORY.md` satisfies the anchor.
    assert count(pretty=False, anchored=True) >= 1, (
        "the anchor alone was expected to be defeated by a body that cites the path "
        "with any leading directory; it is not, so message suppression is decorative")
    # ...and the emitted line agrees with the anchored form it spells
    assert int(emitted.stdout.strip() or 0) == count(pretty=True, anchored=True)


def test_the_pattern_is_keyed_to_the_curated_set_and_not_to_the_bare_filenames(
        vault_repo):
    """The other half of the emitted pattern's job. `app/harness/protected_paths.py`
    guards `~/obsidian/lloyd/{SOUL,USER,MEMORY}.md` specifically, and a note named
    `MEMORY.md` anywhere else in the vault is ordinary content the pipeline is
    expected to commit. An anchored-but-unqualified pattern would report those as the
    incident, which is the failure mode that gets a guard switched off: this fixture
    commit carries one such file, and the check the wrapper emits has to walk past it."""
    # The real shape: a sweep commit that carries the file without the job having
    # declared it, which is what makes the wrapper print the list and, with it, the
    # check (clause 1 emits the check beside the list — with nothing unattributed
    # there is nothing to explain and nothing is emitted).
    _write(vault_repo, "memory/MEMORY.md", "a daily-note file that happens to be named MEMORY.md\n")
    proc = _run_wrapper(vault_repo, "pipeline: snapshot of a note",
                        paths=["memory/MEMORY.md"], job=JOB,
                        writes="knowledge/somebody-elses.md")
    assert proc.returncode == 0, proc.stderr
    cmd = _emitted_check(proc.stderr)
    assert _commit_paths(vault_repo) == ["memory/MEMORY.md"], _commit_paths(vault_repo)
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120)
    assert out.returncode in (0, 1), out.stderr
    assert out.stdout.strip() == "0", (
        f"a note called memory/MEMORY.md is not a curated loaded-memory file, and the "
        f"emitted check reported otherwise: {out.stdout!r}")
    # positive control that the emitted pattern is not simply inert: the curated file
    # itself, in the same repository, does register
    _write(vault_repo, "lloyd/MEMORY.md", "the curated loaded-memory file\n")
    _git(vault_repo, "add", "lloyd/MEMORY.md")
    _git(vault_repo, "commit", "-qm", "a writer that edits the curated file")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "1", (
        f"the emitted check missed a commit that really does carry lloyd/MEMORY.md: "
        f"{out.stdout!r}")
