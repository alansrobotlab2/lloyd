"""A vault commit says which job wrote it, in git metadata, not in its subject: #668.

The claim this pins is narrow and it is about attribution, not behaviour. Every
commit in `~/obsidian` — 1,642 of 1,645 at triage — carries
`alansrobotlab <gestalt73@gmail.com>` as both author and committer, including the
nightly family (`3b499469` "nightly: knowledge write 2026-09-17 (third pass)",
`2145ef3d` "nightly-reflection: post-improvement 2026-09-17"), and those commit
bodies are *empty*: no trailer of any kind. So `git blame` on a line a job
rewrote names a human, and the only record that a machine wrote it is prose in
the subject — prose the audit path cannot parse and a job rewrites freely. That
inverts the accountability signal #527 wants to build: a rejected-quality change
would carry Alan's name with no accept ever having happened.

Two metadata fields carry the job, and both are needed:

* the author identity, `lloyd-<job> <<job>@jobs.lloyd.local>`, which is what
  `git blame` and `git log --format=%ae` show today with no new tooling;
* a `Job: <job>` trailer, which survives a re-author and is what the shipped
  query prefers.

The trailer alone would serve the query but not blame; the author field alone
would serve blame but is one field, so a mis-typed `-c` override — the `~/lloyd`
side of this defect, where one machine identity drifts across four spellings and
defeats `scorecard.py:415`'s case-exact filter — would lose the signal with no
second copy to notice against.

What is NOT a signal, and one test per reason:

* **The subject line.** It is the field the defect says is untrustworthy, so
  nothing in `vault_commit_identity.py` reads it: `Commit` has no message field,
  and `test_classification_is_unchanged_when_only_the_messages_are_reworded`
  proves the sets are byte-identical across a reword rather than asserting that
  the code looks message-shaped.
* **A `Co-Authored-By: Claude …` trailer.** 29 of the last 500 vault commit
  bodies carry one and they come from harness-side, mostly *human*-initiated
  sessions, while the nightly job commits have empty bodies. Keying on it
  classifies this item's worst case backwards.
* **`LLOYD_JOB` being unset.** Then the wrapper commits under the ambient
  identity with no trailer, which is what leaves the human path and the
  automod-round path byte-for-byte where they were.

The two halves sit either side of a language boundary — bash builds the identity,
Python parses it — so
`test_the_wrapper_and_the_query_spell_the_identity_the_same_way` runs both over
one commit and refuses drift, the way `test_autonomy_status_change_audit.py` pins
`DISPATCH_STOPPING_STATUSES` across the same seam.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "util" / "vault-commit.sh"
QUERY = REPO_ROOT / "scripts" / "util" / "vault_commit_identity.py"
VAULT_SKILLS = Path.home() / "obsidian" / "skills"

# `scripts/util/` is not an importable package — `scripts/` has no __init__.py and
# its neighbours are hyphenated — so load the module from its path. It is a
# script first: `scripts/util/autonomy_status_findings.py` next door is invoked
# the same way by a shell.
_SPEC = importlib.util.spec_from_file_location("vault_commit_identity", QUERY)
VCI = importlib.util.module_from_spec(_SPEC)
# Registered before exec: `@dataclass` resolves its own annotations through
# `sys.modules[cls.__module__]`, which is unset for a module built this way, and
# the failure is an AttributeError at import that looks nothing like its cause.
sys.modules[_SPEC.name] = VCI
_SPEC.loader.exec_module(VCI)

#: The identity the defect is about: the human's own, as the fixture's ambient config.
HUMAN_NAME = "alansrobotlab"
HUMAN_EMAIL = "gestalt73@gmail.com"
#: Clause 4's job, spelled as its skill directory is spelled, because that slug is
#: already the job's identity in every autonomy task file (`skill_name:`).
JOB = "nightly-reflection-knowledge-write"


def _git(repo: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(("git", "-C", str(repo), *args),
                          capture_output=True, text=True, timeout=120,
                          env=env if env is not None else os.environ.copy())
    assert proc.returncode == 0, f"git {' '.join(args)}: {proc.stderr}"
    return proc


def _wrapper_env(repo: Path, job: str | None) -> dict:
    """A minimal env for the wrapper. No HOME, so no `~/.gitconfig` can leak in and
    the fixture's own `user.name` is the only ambient identity there is. The
    autonomy-status rung the wrapper runs is stdlib-only, so `sys.executable` is
    named for it rather than relying on what `/usr/bin/python3` happens to have.
    """
    env = {"VAULT_DIR": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin",
           "LLOYD_PYTHON": sys.executable}
    if job is not None:
        env["LLOYD_JOB"] = job
    return env


def _touch(repo: Path, name: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"content of {name}\n", encoding="utf-8")


def _run_wrapper(repo: Path, msg: str, *, job: str | None = None,
                 dirty: str | None = None) -> subprocess.CompletedProcess:
    """Run the real wrapper as a subprocess — the boundary that matters, since a
    job reaches it as a shell command, never as an import."""
    if dirty is not None:
        _touch(repo, dirty)
    return subprocess.run(("bash", str(WRAPPER), msg),
                          capture_output=True, text=True, timeout=120,
                          cwd=str(repo), env=_wrapper_env(repo, job))


@pytest.fixture
def vault_repo(tmp_path):
    """A real git repo on `main` whose ambient identity is Alan's, so "not under
    Alan's identity" is an assertion about the real name and not one I invented."""
    repo = tmp_path / "vault"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", HUMAN_EMAIL)
    _git(repo, "config", "user.name", HUMAN_NAME)
    _touch(repo, "seed.md")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _head(repo: Path, fmt: str) -> str:
    """HEAD rendered with `fmt`, minus the newline `git log` appends — every
    comparison against a literal identity is against the value, not the value
    plus a delimiter."""
    return _git(repo, "log", "-1", f"--format={fmt}").stdout.strip()


def _rev(repo: Path, back: int) -> str:
    return _git(repo, "rev-parse", f"HEAD~{back}").stdout.strip()


def _fixture_history(repo: Path) -> tuple[str, str]:
    """Land exactly two commits over the base: one job's, one a human's.

    Returns `(job_sha, non_job_sha)`. One-and-one is the smallest history that can
    show both sets populated, so a classifier that put everything in one bucket
    cannot pass the disjointness assertions below.
    """
    _run_wrapper(repo, "nightly: knowledge write 2026-09-20", job=JOB,
                 dirty="knowledge/job-authored.md")
    job_sha = _rev(repo, 0)
    _run_wrapper(repo, "alan: edited the daily note", dirty="memory/learnings/log.md")
    return job_sha, _rev(repo, 0)


def _reword_history(repo: Path, messages: list[str]) -> None:
    """Reword every commit reachable from HEAD, oldest first, changing nothing else.

    `git commit --amend` reaches only HEAD, and a helper that re-authored commits
    would be testing its own override instead of the classifier. So each field the
    convention is built from — author name, email, date, tree — is read back and
    handed to `git commit-tree` untouched, with only `-m` differing, and the branch
    is moved to the end of the rebuilt chain. Committer fields are left to the
    ambient config: they are not part of the classification.
    """
    shas = list(reversed(_git(repo, "rev-list", "HEAD").stdout.split()))
    assert len(shas) == len(messages), f"{len(shas)} commits, {len(messages)} messages"
    sep = "\x1f"
    fmt = sep.join(["%an", "%ae", "%aI", "%T", "%B"])
    env = os.environ.copy()
    new: str | None = None
    for sha, msg in zip(shas, messages):
        raw = _git(repo, "log", "-1", sha, f"--format={fmt}").stdout.rstrip("\n")
        fields = raw.split(sep)
        assert len(fields) == 5, fields
        an, ae, adate, tree, _old_body = fields
        env["GIT_AUTHOR_NAME"] = an
        env["GIT_AUTHOR_EMAIL"] = ae
        env["GIT_AUTHOR_DATE"] = adate
        args = ["commit-tree", tree] + (["-p", new] if new else []) + ["-m", msg]
        new = subprocess.run(("git", "-C", str(repo), *args), capture_output=True,
                             text=True, timeout=120, check=True, env=env).stdout.strip()
    _git(repo, "update-ref", "refs/heads/main", new)


# --------------------------------------------------------------------------
# Clause 1: LLOYD_JOB set -> non-human author, job-named email, Job: trailer.
# --------------------------------------------------------------------------

def test_wrapper_with_the_job_env_authors_as_the_job_and_adds_the_trailer(vault_repo):
    """Clause 1, through the shell: the commit exists, is not Alan's, and names the
    job in both metadata fields the convention uses."""
    proc = _run_wrapper(vault_repo, "nightly: knowledge write 2026-09-20",
                        job=JOB, dirty="knowledge/note.md")
    assert proc.returncode == 0, proc.stderr
    assert _head(vault_repo, "%an") != HUMAN_NAME
    assert _head(vault_repo, "%ae") != HUMAN_EMAIL
    assert _head(vault_repo, "%ae") == f"{JOB}@{VCI.JOB_EMAIL_DOMAIN}"
    assert _head(vault_repo, "%an") == f"{VCI.JOB_NAME_PREFIX}{JOB}"
    # The committer moves with the author: a job write must not read as "Alan
    # committed a machine's edit" either, which is the other half of blame.
    assert _head(vault_repo, "%ce") == f"{JOB}@{VCI.JOB_EMAIL_DOMAIN}"
    body = _head(vault_repo, "%B")
    assert f"Job: {JOB}" in body.splitlines(), body
    # The subject is untouched by the convention: the job signal is additive.
    assert body.splitlines()[0] == "nightly: knowledge write 2026-09-20"


def test_query_reads_the_job_off_that_commit_without_the_subject(vault_repo):
    """Clause 1's second half: the commit the wrapper just landed is answerable by
    the shipped query, which is the reason for landing it."""
    _run_wrapper(vault_repo, "nightly: knowledge write 2026-09-20",
                 job=JOB, dirty="knowledge/note.md")
    commits = VCI.commit_records(vault_repo)
    assert commits[0].sha == _rev(repo=vault_repo, back=0)
    assert commits[0].job == JOB
    assert commits[0].job_authored is True
    # The base commit, written by nobody's job, is not silently swept in.
    assert [c.job for c in commits[1:]] == [None] * (len(commits) - 1)


def test_the_wrapper_and_the_query_spell_the_identity_the_same_way(vault_repo):
    """The bash half and the Python half must be one convention, not two.

    `vault-commit.sh` builds `lloyd-$JOB` / `$JOB@jobs.lloyd.local` by string
    concatenation, because a commit that silently loses its identity is worse than
    one that refuses; `vault_commit_identity.py.job_identity()` holds the same two
    strings in Python and is what the query side trusts. Drift between them would
    make every job-authored commit classify as not-a-job — the defect's own shape —
    so this compares the two over a commit the wrapper really made.
    """
    other = "autonomy-data-pipeline"
    _run_wrapper(vault_repo, "autonomy-data-pipeline: 2026-09-20",
                 job=other, dirty="autonomy/40-config.md")
    name, email = VCI.job_identity(other)
    assert (_head(vault_repo, "%an"), _head(vault_repo, "%ae")) == (name, email)
    assert VCI.job_from_email(_head(vault_repo, "%ae").strip()) == other


def test_wrapper_refuses_a_job_name_that_could_not_form_an_identity(vault_repo):
    """A typo in the env must not become a commit under Alan's name — that is the
    failure this item exists to stop, so it is a refusal (exit 3), not a fallback."""
    for i, bad in enumerate(("my job", "-x", "a/b")):
        _touch(vault_repo, f"scratch/{i}.md")
        proc = subprocess.run(("bash", str(WRAPPER), "refused"),
                              capture_output=True, text=True, timeout=120,
                              cwd=str(vault_repo), env=_wrapper_env(vault_repo, bad))
        assert proc.returncode == 3, (bad, proc.returncode, proc.stderr)
        assert "not a usable job name" in proc.stderr
    # Nothing was committed and nothing was staged: the refusal is total.
    assert _head(vault_repo, "%s").strip() == "base"
    assert _git(vault_repo, "diff", "--cached", "--name-only").stdout.strip() == ""


# --------------------------------------------------------------------------
# Clause 2: LLOYD_JOB unset -> ambient identity, no trailer, path unchanged.
# --------------------------------------------------------------------------

def test_wrapper_without_the_job_env_keeps_the_identity_and_adds_no_trailer(vault_repo):
    """Clause 2: the human and automod-round paths keep exactly what they had."""
    proc = _run_wrapper(vault_repo, "human: reword the daily note",
                        dirty="memory/learnings/today.md")
    assert proc.returncode == 0, proc.stderr
    assert _head(vault_repo, "%an") == HUMAN_NAME
    assert _head(vault_repo, "%ae") == HUMAN_EMAIL
    body = _head(vault_repo, "%B")
    assert "Job:" not in body, body
    assert body.strip() == "human: reword the daily note"


def test_the_two_paths_differ_only_in_identity_when_the_subject_is_identical(vault_repo):
    """The negative half of clause 3, set up as the sharpest case: the same subject
    line committed twice, once with a job named and once without. If any part of the
    classification read the message, these two commits would be indistinguishable
    and one of the two assertions below must fail.
    """
    subject = "nightly: knowledge write 2026-09-20"
    _run_wrapper(vault_repo, subject, dirty="a.md")
    _run_wrapper(vault_repo, subject, job=JOB, dirty="b.md")
    commits = VCI.commit_records(vault_repo)
    assert [c.job for c in commits] == [JOB, None, None]
    assert VCI.partition(commits) == ([commits[0].sha],
                                      [commits[1].sha, commits[2].sha])


# --------------------------------------------------------------------------
# Clause 3: the shipped query — disjoint, total, message-independent.
# --------------------------------------------------------------------------

def test_query_returns_two_disjoint_sets_for_a_job_and_a_non_job_commit(vault_repo):
    """Clause 3's first assertion on a one-and-one fixture: the partition is
    disjoint, total, and puts each commit where its metadata says."""
    job_sha, human_sha = _fixture_history(vault_repo)
    commits = VCI.commit_records(vault_repo)
    job_shas, non_job_shas = VCI.partition(commits)
    assert job_shas == [job_sha]
    assert human_sha in non_job_shas
    assert not (set(job_shas) & set(non_job_shas)), "the two sets must be disjoint"
    assert sorted(job_shas + non_job_shas) == sorted(c.sha for c in commits), (
        "every commit lands in exactly one set — a commit in neither is an audit hole")


def test_classification_is_unchanged_when_only_the_messages_are_reworded(vault_repo):
    """Clause 3's second assertion, and the load-bearing one.

    Reword all three commits, then re-classify. `classification_lines()` carries
    the tree hash rather than the sha precisely so this comparison is meaningful:
    a sha-bearing listing cannot be byte-stable across a message edit by
    definition, so asserting stability on the shas would be vacuous. If any part
    of the classification read the subject or body, this is where it shows up.
    """
    job_sha, human_sha = _fixture_history(vault_repo)
    before = VCI.classification_lines(VCI.commit_records(vault_repo))
    assert len(before) == 3, before
    assert len({job_sha, human_sha}) == 2

    _reword_history(vault_repo, ["base",
                                 "something the job titled completely differently",
                                 "alan reworded this himself"])
    after = VCI.classification_lines(VCI.commit_records(vault_repo))
    assert after == before, (
        "the classification moved when only the messages changed:"
        f"\nbefore={before}\nafter ={after}")

    # And the sets still separate the two, under their new shas.
    job_shas, non_job_shas = VCI.partition(VCI.commit_records(vault_repo))
    assert len(job_shas) == 1 and len(non_job_shas) == 2
    assert human_sha not in job_shas, "the reworded human commit is still not a job"


def test_rewording_actually_changed_the_shas_it_left_the_trees_alone(vault_repo):
    """The control for the test above: if rewording were a no-op, byte-identical
    output would prove nothing. The shas must move and the trees must not.
    """
    _fixture_history(vault_repo)
    shas_before = [c.sha for c in VCI.commit_records(vault_repo)]
    trees_before = [c.tree for c in VCI.commit_records(vault_repo)]
    _reword_history(vault_repo, ["base reworded", "job commit reworded",
                                 "human commit reworded"])
    after = VCI.commit_records(vault_repo)
    assert [c.sha for c in after] != shas_before, "rewording did not rehash anything"
    assert [c.tree for c in after] == trees_before, "rewording changed a tree"
    # Every one of the three moved, including the root commit.
    assert len({c.sha for c in after}) == 3 == len(after)


def test_a_claude_co_author_trailer_does_not_make_a_commit_look_job_authored(vault_repo):
    """The backwards signal, pinned as a negative test.

    Real vault history: 29 of the last 500 commit bodies carry
    `Co-Authored-By: Claude … <noreply@anthropic.com>` and belong mostly to
    human-initiated sessions, while the nightly job commits have empty bodies. A
    classifier looking for a machine marker in the body would invert exactly the
    case #668 is about, so a human commit wearing that trailer has to land in the
    not-a-job set.
    """
    _touch(vault_repo, "backlog/668-item.md")
    _git(vault_repo, "add", "-A")
    _git(vault_repo, "commit", "-qm",
         "backlog: needs-human sweep 2026-09-14 — rulings applied\n\n"
         "Co-Authored-By: Claude <noreply@anthropic.com>")
    _run_wrapper(vault_repo, "nightly: knowledge write 2026-09-20", job=JOB,
                 dirty="lloyd/MEMORY.md")
    commits = VCI.commit_records(vault_repo)
    assert [c.job for c in commits] == [JOB, None, None]
    claude_commit = commits[1]
    printed = _git(vault_repo, "log", "-1", "--format=%B", claude_commit.sha).stdout
    assert "Co-Authored-By" in printed, "fixture did not land the trailer it claims"
    assert claude_commit.job_authored is False


def test_the_query_prefers_the_trailer_when_the_author_field_does_not_name_a_job(vault_repo):
    """The two metadata copies are independent, and the trailer wins.

    A commit can lose its author identity and keep its trailer. The `~/lloyd` side
    of this defect is exactly that: one machine identity drifting across four
    spellings, which is what lets `scorecard.py:415`'s case-exact filter count 4
    machine commits as human touch. Preferring the trailer means the answer
    survives that drift instead of depending on one `-c` flag being typed right.
    """
    _touch(vault_repo, "knowledge/note.md")
    _git(vault_repo, "add", "-A")
    _git(vault_repo, "commit", "-qm", f"nightly: knowledge write\n\nJob: {JOB}")
    commits = VCI.commit_records(vault_repo)
    assert commits[0].author_email == HUMAN_EMAIL, "fixture is the drift case"
    assert commits[0].job == JOB


def test_the_four_drifted_machine_spellings_are_not_jobs(vault_repo):
    """Each of the four spellings `~/lloyd` history actually carries is a not-a-job
    answer here.

    None of them sits on the convention's domain, so the query calls them what
    they are: commits whose author is not a person either. Fixing `scorecard.py`'s
    filter is its own item, but the answer this query gives is pinned so the two
    cannot silently disagree about what those addresses mean.
    """
    for i, email in enumerate(("lloyd@local", "lloyd@localhost",
                               "Lloyd@local", "Lloyd@localhost")):
        _touch(vault_repo, f"scratch/{i}.md")
        _git(vault_repo, "add", "-A")
        _git(vault_repo, "commit", "-qm", f"round: {email}",
             env={**os.environ, "GIT_AUTHOR_NAME": "lloyd", "GIT_AUTHOR_EMAIL": email,
                  "GIT_COMMITTER_NAME": "lloyd", "GIT_COMMITTER_EMAIL": email})
    jobs, non_jobs = VCI.partition(VCI.commit_records(vault_repo))
    assert jobs == []
    assert len(non_jobs) == 5, "base + the four spellings, none of them a job"


# --------------------------------------------------------------------------
# The shipped query, as a command an audit can run.
# --------------------------------------------------------------------------

def test_query_cli_answers_from_metadata_in_both_documented_forms(vault_repo):
    """The process boundary: a caller runs the script rather than importing it, and
    the pure-`git log` form the module docstring tells an auditor to run has to be
    the real placeholder, not prose.
    """
    job_sha, human_sha = _fixture_history(vault_repo)
    proc = subprocess.run((sys.executable, str(QUERY), "--repo", str(vault_repo),
                           "-f", "classification"),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 3
    assert sum(1 for ln in lines if ln.endswith(f"\t{JOB}")) == 1

    jobs = subprocess.run((sys.executable, str(QUERY), "--repo", str(vault_repo),
                           "--jobs-only"),
                          capture_output=True, text=True, timeout=120)
    assert jobs.returncode == 0, jobs.stderr
    assert jobs.stdout.strip().split("\t")[0] == job_sha
    assert human_sha not in jobs.stdout

    raw = _git(vault_repo, "log",
               f"--format=%H%x09%(trailers:key={VCI.JOB_TRAILER_KEY},valueonly)").stdout
    assert job_sha in raw, "the documented git-only form does not find the job"
    assert raw.count(f"\t{JOB}") == 1


def test_query_refuses_a_repo_it_cannot_read_instead_of_answering_empty(tmp_path):
    """An empty answer here means "no job has written anything" — the wrong
    reassurance to hand an audit path — so a missing or unreadable repo is an
    error, never a `[]`.
    """
    with pytest.raises(VCI.CommitIdentityError):
        VCI.commit_records(tmp_path / "nope")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(VCI.CommitIdentityError):
        VCI.commit_records(plain)
    proc = subprocess.run((sys.executable, str(QUERY), "--repo", str(plain)),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "failed in" in proc.stderr


# --------------------------------------------------------------------------
# Clause 4: the job's own commit step, and the live vault, are the point.
# --------------------------------------------------------------------------

KNOWLEDGE_WRITE_SKILL = VAULT_SKILLS / "nightly-reflection-knowledge-write" / "SKILL.md"


def _commit_step_body() -> str:
    """The bash block under `## Step 3: Commit`, as the job reads it."""
    text = KNOWLEDGE_WRITE_SKILL.read_text(encoding="utf-8")
    step = text.split("## Step 3: Commit", 1)
    assert len(step) == 2, "the skill has no '## Step 3: Commit' section any more"
    return step[1].split("```bash", 1)[1].split("```", 1)[0]


@pytest.mark.live_vault
def test_nightly_knowledge_write_skill_commits_through_the_wrapper_with_its_job():
    """Clause 4, graded against the live skill file.

    The step at `skills/nightly-reflection-knowledge-write/SKILL.md:248` was
    `git add memory/ … && git commit -m "nightly: knowledge write $(date +%Y-%m-%d)"`
    — a raw commit, under whoever `user.name` says, with an empty body. It must
    name the wrapper and the job instead, and no raw `git commit` may survive the
    commit step, or the convention is a suggestion the next job ignores.
    """
    body = _commit_step_body()
    assert "scripts/util/vault-commit.sh" in body, body
    assert f"LLOYD_JOB={JOB}" in body, body
    assert VCI.JOB_NAME_RE.fullmatch(JOB), JOB
    assert "git commit" not in body, f"a raw commit is still prescribed:\n{body}"


@pytest.mark.live_vault
def test_the_skill_commit_invocation_lands_a_job_authored_commit_when_executed(vault_repo):
    r"""Clause 4's other half: the line in the skill, run, not read.

    Asserting on the text cannot see a broken line continuation, a misplaced `\`, or a
    quoting slip — every one of which turns the job's commit step into a shell error at
    02:00 with the run's output left unstaged, which is the #341 failure the wrapper
    exists to prevent. So the invocation is executed over a fixture vault with
    `VAULT_DIR` pointed at it, and the resulting commit has to be the one clauses 1-3
    describe: not Alan's, and carrying the trailer.

    One substitution, named because it is the whole point of the test: the skill names
    the wrapper by its deployed path, `~/lloyd/scripts/util/vault-commit.sh`, so that
    literal is swapped for this tree's `WRAPPER` — otherwise the test exercises whatever
    is installed rather than the change under review. Everything that actually parses —
    the `LLOYD_JOB=` prefix, the continuation, the message — is the skill's own text.
    """
    invocation = "\n".join(
        line.replace("~/lloyd/scripts/util/vault-commit.sh", str(WRAPPER))
        for line in _commit_step_body().splitlines()
        if line.strip().startswith(("LLOYD_JOB=", "~/lloyd/scripts/util/vault-commit.sh"))
    )
    assert "vault-commit.sh" in invocation, "no invocation found to execute"
    _touch(vault_repo, "lloyd/MEMORY.md")
    proc = subprocess.run(("bash", "-c", invocation),
                          capture_output=True, text=True, timeout=120,
                          cwd=str(vault_repo), env=_wrapper_env(vault_repo, None))
    assert proc.returncode == 0, proc.stderr
    commit = VCI.commit_records(vault_repo)[0]
    assert commit.job == JOB, commit
    assert commit.author_email == VCI.job_identity(JOB)[1], commit
    assert commit.author_name == VCI.job_identity(JOB)[0], commit
    subject = _git(vault_repo, "log", "-1", "--format=%s").stdout.strip()
    assert subject.startswith("nightly: knowledge write "), subject


@pytest.mark.live_vault
def test_the_live_vault_answers_the_query_and_the_partition_is_total():
    """The acceptance check, run against `~/obsidian` rather than a fixture.

    Two things must hold on the real repo: the query answers at all (a
    `CommitIdentityError` is an unreadable store, which must never be reported as
    "no job commits"), and the partition is total and disjoint over real history.
    The positive control is the human identity's own count — the defect is that it
    is nearly the whole repo, so if that reads 0 this test is not looking at the
    history it describes.
    """
    commits = VCI.commit_records(Path.home() / "obsidian", limit=200)
    assert commits, "no commits readable in ~/obsidian"
    jobs, non_jobs = VCI.partition(commits)
    assert not (set(jobs) & set(non_jobs))
    assert len(jobs) + len(non_jobs) == len(commits)
    human = [c for c in commits if c.author_email == HUMAN_EMAIL]
    assert human, (f"no {HUMAN_EMAIL} commit in the newest {len(commits)} — this "
                   "assertion is not reading the history it describes")
    # Every job name the query returns has to be one the convention can produce, so
    # a stray `Job: whatever` in a body cannot pass as a job identity.
    for c in commits:
        if c.job is not None:
            assert VCI.JOB_NAME_RE.fullmatch(c.job), f"{c.sha}: {c.job!r}"


@pytest.mark.live_vault
def test_every_wrapper_caller_that_names_a_job_names_a_parseable_one():
    """The blast radius of the convention, measured rather than asserted.

    Ten call sites in seven skills reach the wrapper. Each either declares its job
    or stays under the ambient identity, which clause 2 guarantees is not a
    regression. What must not exist is a caller whose `LLOYD_JOB` cannot parse:
    that turns a nightly commit into exit 3 and strands a whole job's output, which
    is the #341 failure this wrapper was written to prevent.
    """
    seen = []
    for skill in sorted(VAULT_SKILLS.glob("*/SKILL.md")):
        for line in skill.read_text(encoding="utf-8").splitlines():
            if "scripts/util/vault-commit.sh" in line and not line.strip().startswith("#"):
                seen.append(line)
                if "LLOYD_JOB=" in line:
                    job = line.split("LLOYD_JOB=", 1)[1].split()[0].strip("\"'")
                    assert VCI.JOB_NAME_RE.fullmatch(job), f"{skill.name}: {line}"
    assert len(seen) >= 10, f"expected the ten documented call sites, found {len(seen)}"
