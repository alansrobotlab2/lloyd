#!/usr/bin/env python3
"""Which job wrote this vault commit — answered from git metadata alone (#668).

The convention
--------------
A vault commit written by a scheduled job is authored as::

    lloyd-<job> <<job>@jobs.lloyd.local>

and carries one trailer in its message::

    Job: <job>

Both halves are git metadata, so an audit can separate a job's own write from a
human's without reading the subject line. `scripts/util/vault-commit.sh` is what
applies it, from `LLOYD_JOB`; a commit with `LLOYD_JOB` unset is unchanged —
ambient identity, no trailer — which is what keeps the human path and the
automod-round path exactly where they were.

Why the trailer *and* the author field
--------------------------------------
The defect this closes (`backlog #668`) is that every job write in `~/obsidian`
carried `alansrobotlab <gestalt73@gmail.com>` — 1,642 of 1,645 commits measured
2026-09-20, the 3 others a single round's `-c` override — so `git blame`
attributed unreviewed machine edits to the human, and the only trace of the job
was prose in the subject. Subject
prose cannot be the signal, because it is exactly the part a job rewrites freely
and the part a human-initiated session can imitate: 29 of the last 500 vault
commit bodies carry a `Co-Authored-By: Claude …` trailer, and those are mostly
the *human*-started sessions, so "has a Claude trailer ⇒ machine-authored"
classifies the worst case backwards.

So the author email carries the job and the trailer records it independently:
either one alone answers the question, and this module prefers the trailer. The
two spellings are held in one place (`job_identity`); the wrapper builds them in
bash because a commit that silently loses its identity is worse than one that
refuses, and `tests/test_vault_commit_identity.py` pins the two against each
other rather than trusting a comment.

Usage
-----
::

    scripts/util/vault_commit_identity.py                    # every commit reachable from HEAD
    scripts/util/vault_commit_identity.py -n 50 -f records
    scripts/util/vault_commit_identity.py -f classification   # sha-free: stable across a reword
    scripts/util/vault_commit_identity.py --jobs-only

A repo that cannot be queried raises `CommitIdentityError`. It never returns an
empty list for "I could not look": an empty answer here reads as "no job has
written anything", which is the wrong reassurance to give an audit path.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

#: Domain every job-authored vault commit is addressed under.
JOB_EMAIL_DOMAIN = "jobs.lloyd.local"
#: Trailer key that names the job, independent of who is in the author field.
JOB_TRAILER_KEY = "Job"
#: Prefix on the author *name*, so `git log` reads as a job without parsing the email.
JOB_NAME_PREFIX = "lloyd-"
#: A job name that can survive being put into an email address and a `git -c`
#: override. `vault-commit.sh` applies the same rule and exits 3 on a mismatch.
JOB_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")

# `%(` is only a placeholder when it is not followed by a space; the record
# separator that closes the trailer field is what keeps it one. The trailer is
# last because `valueonly` emits a newline after the value, and a trailing field
# can absorb that with a strip.
#
# Unit separator, not NUL. NUL is the usual choice for this and it cannot be
# used here: the separator travels inside the `--format=` argv string, and an
# argument containing a NUL makes `execve` refuse the process with
# "embedded null byte" — the failure surfaces in the caller, several frames from
# here, as a ValueError from subprocess.
_RECORD_SEP = "\x1f"
_FORMAT = ("%H%x09%an%x09%ae%x09%cn%x09%ce%x09%T"
           f"%x09%(trailers:key={JOB_TRAILER_KEY},valueonly)" + _RECORD_SEP)


class CommitIdentityError(RuntimeError):
    """The identity query could not be answered — never report that as "no jobs"."""


@dataclass(frozen=True)
class Commit:
    """One commit, as much of it as this convention needs. No message field: a
    classification that read the subject could not claim to be message-independent.
    """

    sha: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    tree: str
    job: str | None

    @property
    def job_authored(self) -> bool:
        return self.job is not None


def job_identity(job: str) -> tuple[str, str]:
    """Return `(author_name, author_email)` for a job — the one canonical spelling.

    Raises `ValueError` for a name that cannot form a git identity, so a typo in
    `LLOYD_JOB` is a refusal rather than a commit under
    `lloyd-my job <my job@jobs.lloyd.local>`.
    """
    if not isinstance(job, str) or not JOB_NAME_RE.fullmatch(job):
        raise ValueError(
            f"not a usable job name: {job!r} (want {JOB_NAME_RE.pattern})")
    return f"{JOB_NAME_PREFIX}{job}", f"{job}@{JOB_EMAIL_DOMAIN}"


def job_from_email(email: str) -> str | None:
    """The job named by an author email, or None if the address is not a job's."""
    local, _, domain = (email or "").rpartition("@")
    if domain == JOB_EMAIL_DOMAIN and JOB_NAME_RE.fullmatch(local):
        return local
    return None


def _job_from_trailer(raw: str) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    # `valueonly` prints one line per matching trailer; the first wins. A commit
    # carrying two different `Job:` values would be two jobs' write, which the
    # wrapper cannot produce, so the first is the answer rather than a guess.
    return value.splitlines()[0].strip() or None


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(("git", "-C", str(repo), *args),
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise CommitIdentityError(
            f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()[:300]}")
    return proc.stdout


def require_repo(repo: str | Path) -> Path:
    """Resolve `repo` as a git repository or raise.

    A missing repo and an empty repo are different answers, and neither is "no
    job commits": this is the only way a caller can tell them apart.
    """
    path = Path(repo).expanduser()
    if not path.is_dir():
        raise CommitIdentityError(f"not a directory: {repo}")
    _git(path, "rev-parse", "--git-dir")
    return path


def commit_records(repo: str | Path = "~/obsidian", *, rev: str = "HEAD",
                   limit: int | None = None) -> list[Commit]:
    """Every commit reachable from `rev`, newest first, with its job resolved.

    `job` comes from the `Job:` trailer, falling back to the author email's
    local-part when it sits on `JOB_EMAIL_DOMAIN`. Nothing here reads the commit
    subject or body prose, so rewording a commit cannot move it between the two
    sets.
    """
    root = require_repo(repo)
    _git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    args = ["log", f"--format={_FORMAT}", rev]
    if limit is not None:
        args.insert(1, f"-n{int(limit)}")
    out = _git(root, *args)
    commits: list[Commit] = []
    for blob in out.split(_RECORD_SEP):
        line = blob.strip("\n")
        if not line:
            continue
        fields = line.split("\t", 6)
        if len(fields) != 7:
            raise CommitIdentityError(f"unparsable record from git log: {line[:120]!r}")
        sha, an, ae, cn, ce, tree, trailer = fields
        job = _job_from_trailer(trailer) or job_from_email(ae)
        commits.append(Commit(sha=sha, author_name=an, author_email=ae,
                              committer_name=cn, committer_email=ce,
                              tree=tree, job=job))
    return commits


def partition(commits: list[Commit]) -> tuple[list[str], list[str]]:
    """`(job_shas, non_job_shas)`, log order. Total and disjoint by construction."""
    job, other = [], []
    for c in commits:
        (job if c.job_authored else other).append(c.sha)
    return job, other


def classification_lines(commits: list[Commit]) -> list[str]:
    """The answer without any sha in it: `tree<TAB>author<TAB>email<TAB>job`.

    The sha is deliberately absent. Rewording a commit rehashes it, so a
    sha-bearing listing cannot be byte-stable across a message edit no matter how
    good the classifier is — the tree hash is the field that survives one, which
    is what makes this the form to diff when the question is "did rewording the
    messages change what you can conclude?". It is a content key, not a unique
    one: two commits with identical trees collide, as repeated no-op job writes do.
    """
    return [f"{c.tree}\t{c.author_name}\t{c.author_email}\t{c.job or ''}"
            for c in commits]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vault_commit_identity.py",
        description="Which job wrote each vault commit, from git metadata only.")
    ap.add_argument("--repo", default="~/obsidian",
                    help="git repository to query (default: ~/obsidian)")
    ap.add_argument("--rev", default="HEAD", help="rev to walk (default: HEAD)")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="only the N newest commits")
    ap.add_argument("-f", "--format", default="records",
                    choices=("records", "classification", "json"),
                    help="records: sha, job, author email. classification: the same "
                         "classification without shas, so it survives a reword. "
                         "json: every field.")
    ap.add_argument("--jobs-only", action="store_true",
                    help="print only the commits some job wrote")
    args = ap.parse_args(argv)
    try:
        commits = commit_records(args.repo, rev=args.rev, limit=args.limit)
    except CommitIdentityError as exc:
        print(f"vault_commit_identity.py: {exc}", file=sys.stderr)
        return 2
    if args.jobs_only:
        commits = [c for c in commits if c.job_authored]
    if args.format == "json":
        print(json.dumps([asdict(c) for c in commits], indent=2))
    elif args.format == "classification":
        for line in classification_lines(commits):
            print(line)
    else:
        for c in commits:
            print(f"{c.sha}\t{c.job or '-'}\t{c.author_email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
