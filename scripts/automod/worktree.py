"""Round worktrees.

Layout is dictated by the isolation model: the worktree must live at
`<round>/home/lloyd` so that `HOME=<round>/home` makes `Path.home()/"lloyd"`
and `app.paths.LLOYD_HOME` the same directory. It must be a real directory,
not a symlink — `app/paths.py` calls `.resolve()`.

A worktree shares the live repo's object store (19 MB, 555 tracked files), so
creating one is cheap and a commit made inside it is immediately reachable
from the live repo. That is what lets landing be a pure fast-forward with no
fetch and no push.

`main` is checked out in the live tree, and git refuses to check out the same
branch twice, so every round gets its own `automod/<round_id>` branch.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
WORK_ROOT = Path.home() / "lloyd-work"


def git(repo: Path, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=timeout, check=False)


def round_dir(round_id: str) -> Path:
    return WORK_ROOT / round_id


def worktree_path(round_id: str) -> Path:
    return round_dir(round_id) / "home" / "lloyd"


def round_home(round_id: str) -> Path:
    """`HOME` for a child process that runs CANDIDATE code.

    The layout in this module's docstring exists for exactly this and nothing
    else set it. With `HOME=<round>/home`, `Path.home()/"lloyd"` and
    `app.paths.LLOYD_HOME` are the same directory — the worktree — so a test
    that resolves its tree off `Path.home()` addresses the candidate.

    Until 2026-09-22 `gate._child_env` handed every child the real home, so the
    redirection this layout was built to provide was never actually applied: a
    candidate test that deleted `Path.home()/"lloyd"` deleted production, and on
    that day something did — the whole tree, `.git`, `.venvs`, `qmd`, the model
    weights, in about 35 seconds. `Path.home()` is the name the isolation turns
    on; `LLOYD_HOME` was already safe because `app/paths.py` resolves it from
    `__file__`.
    """
    return round_dir(round_id) / "home"


#: Never linked into a round's home. `lloyd` is the worktree itself, and it must
#: stay a real directory — `app/paths.py` calls `.resolve()`, and a symlink here
#: would resolve straight back to production, silently undoing the redirection.
#: `lloyd-data` is production's runtime data (sessions, workers.db, logs); a
#: round gets an empty one of its own, which `round_data_root` names.
HOME_LINK_SKIP = frozenset({"lloyd", "lloyd-data"})


def round_data_root(round_id: str) -> Path:
    """The round's own data root: `<round>/home/lloyd-data`, never production's.

    What `LLOYD_DATA` is set to for every rung that runs candidate code, so a
    candidate's `sessions/`, `workers.db` and logs land beside its worktree —
    the isolation a data dir inside the tree used to give for free.
    """
    return round_home(round_id) / "lloyd-data"


def ensure_round_home(round_id: str) -> Path:
    """Make `<round>/home` usable as a home, and return it.

    A directory holding only a checkout is not a home. The vault
    (`~/obsidian`), `~/.gitconfig`, and the caches every tool writes all live in
    the real one and are genuinely shared, so every entry but `lloyd` is
    symlinked across. Symlinks rather than copies because the goal is
    redirecting ONE name, not isolating the machine — and because `shutil.rmtree`
    refuses a symlink to a directory, so the same farm that keeps the vault
    readable also makes `rmtree(Path.home() / "obsidian")` raise instead of run.

    Raises if the redirection cannot be proved, because the caller's fallback is
    the real home and that has to be a decision somebody recorded rather than a
    thing that quietly happened.
    """
    wt = worktree_path(round_id)
    if not wt.is_dir():
        raise RuntimeError(f"no worktree at {wt} — cannot build a round home")
    home = round_home(round_id)
    home.mkdir(parents=True, exist_ok=True)
    real = Path.home()
    for entry in sorted(real.iterdir()):
        if entry.name in HOME_LINK_SKIP:
            continue
        link = home / entry.name
        # `is_symlink()` as well as `exists()`: a link whose target has gone is
        # still an entry, and `symlink_to` on top of it raises FileExistsError.
        if link.is_symlink() or link.exists():
            continue
        try:
            link.symlink_to(entry)
        except OSError:
            # One unlinkable entry is a missing convenience; failing the whole
            # farm over it would return the caller to the real home, which is
            # the outcome this exists to prevent.
            pass
    round_data_root(round_id).mkdir(parents=True, exist_ok=True)
    resolved = (home / "lloyd").resolve()
    if resolved != wt.resolve():
        raise RuntimeError(
            f"round home is not redirecting: {home / 'lloyd'} resolves to "
            f"{resolved}, expected {wt.resolve()}")
    return home


def create(round_id: str, base: str = "HEAD", repo: Path | None = None) -> Path:
    """Create `<work>/<round_id>/home/lloyd` on branch `automod/<round_id>`."""
    repo = repo or LIVE_ROOT
    wt = worktree_path(round_id)
    wt.parent.mkdir(parents=True, exist_ok=True)
    branch = f"automod/{round_id}"
    r = git(repo, "worktree", "add", "-q", "-b", branch, str(wt), base)
    if r.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {r.stderr.strip()[:400]}")
    return wt


def branch_exists(repo: Path, branch: str) -> bool:
    return git(repo, "rev-parse", "--verify", "-q", f"refs/heads/{branch}").returncode == 0


def create_from_branch(round_id: str, branch: str, repo: Path | None = None) -> Path:
    """Create the round's worktree on a NEW branch that starts where `branch`
    left off — the resume path for a round the review rung sent back. The
    caller rebases onto live HEAD immediately and deletes the old branch."""
    repo = repo or LIVE_ROOT
    wt = worktree_path(round_id)
    wt.parent.mkdir(parents=True, exist_ok=True)
    r = git(repo, "worktree", "add", "-q", "-b", f"automod/{round_id}", str(wt), branch)
    if r.returncode != 0:
        raise RuntimeError(f"git worktree add from {branch} failed: {r.stderr.strip()[:400]}")
    return wt


def delete_branch(repo: Path, branch: str) -> bool:
    return git(repo, "branch", "-D", branch).returncode == 0


def changed_paths(worktree: Path, base: str) -> list[str]:
    """Repo-relative paths changed between `base` and the worktree's HEAD."""
    r = git(worktree, "diff", "--name-only", f"{base}...HEAD")
    if r.returncode != 0:
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def head(worktree: Path) -> str | None:
    r = git(worktree, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def squash_onto(worktree: Path, base: str, message: str, *,
                keep_ref: str = "") -> tuple[str | None, str]:
    """Collapse `base..HEAD` of the worktree's branch into ONE commit on `base`.

    `(new_sha, detail)`; `new_sha` is None when nothing was changed, and the
    branch is then exactly as it was. The promoter calls this after the last
    gate and before the fast-forward, so `main` gets one commit per landing
    instead of the round's working history (#1204 put eight on it). Three
    things make that safe to do after the gate rather than before it:

    - **The tree is proved identical.** The new commit's tree is compared with
      the gated HEAD's; any difference resets the branch back and returns None.
      What was tested is what lands, byte for byte — only its history differs.
    - **Only a linear branch on top of `base`.** Anything else could not have
      fast-forwarded either, and is left for that refusal to report.
    - **The working history is kept**, under `keep_ref` (a ref outside
      `refs/heads`, so it never shows in a branch list and gc never takes it).

    A single-commit branch is left alone: there is nothing to squash, and
    rewriting it would change a sha the gate recorded for no gain.
    """
    old = head(worktree)
    if not old:
        return None, "no HEAD"
    if git(worktree, "merge-base", "--is-ancestor", base, old).returncode != 0:
        return None, f"{base[:8]} is not an ancestor of the branch"
    count = git(worktree, "rev-list", "--count", f"{base}..{old}")
    if count.returncode != 0 or int(count.stdout.strip() or 0) < 2:
        return None, "one commit or none: nothing to squash"
    if git(worktree, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        return None, "worktree has uncommitted changes"
    old_tree = git(worktree, "rev-parse", f"{old}^{{tree}}").stdout.strip()
    if git(worktree, "reset", "--soft", base).returncode != 0:
        return None, "reset --soft failed"
    commit = git(worktree, "commit", "-q", "--no-verify", "-m", message)
    new = head(worktree)
    new_tree = git(worktree, "rev-parse", "HEAD^{tree}").stdout.strip() if new else ""
    if commit.returncode != 0 or not new or new == base or new_tree != old_tree:
        git(worktree, "reset", "--hard", old)
        return None, (f"squash did not reproduce the gated tree "
                      f"({(commit.stderr or '').strip()[:160]}); branch restored")
    if keep_ref:
        git(worktree, "update-ref", keep_ref, old)
    return new, f"squashed {count.stdout.strip()} commits ({old[:8]} → {new[:8]}), same tree"


def is_clean(repo: Path) -> bool:
    r = git(repo, "status", "--porcelain")
    return r.returncode == 0 and not r.stdout.strip()


def commit_pending(worktree: Path, message: str) -> tuple[str | None, str]:
    """Commit everything uncommitted in a round's worktree: `(new_sha, detail)`.

    `new_sha` is None when there was nothing to commit or the commit failed,
    and `detail` says which. For the reaper, which gates a round whose turn
    stopped without gating (`autocode._gate_if_ungated`): the gate judges
    commits only, and the alternative — `abort` — runs `worktree remove
    --force`, which destroys exactly the edits a stalled turn left behind.
    `add -A` honours `.gitignore`, so caches and `graphify-out/` stay out.
    """
    r = git(worktree, "status", "--porcelain")
    if r.returncode != 0:
        return None, f"git status failed: {(r.stderr or '').strip()[:160]}"
    if not r.stdout.strip():
        return None, "nothing uncommitted"
    if git(worktree, "add", "-A").returncode != 0:
        return None, "git add -A failed"
    c = git(worktree, "commit", "-q", "--no-verify", "-m", message)
    if c.returncode != 0:
        return None, f"commit failed: {(c.stderr or c.stdout or '').strip()[:160]}"
    return head(worktree), f"committed {len(r.stdout.strip().splitlines())} uncommitted path(s)"


def dirty_paths(repo: Path, limit: int | None = None) -> list[str]:
    """Every path `git status` reports as modified or untracked.

    "live tree is dirty" sent round SM_20260909_081105 hunting: four tool
    calls to find the one uncommitted file, then half an hour polling for it
    to clear. The paths are one command away and belong in the refusal — and
    now in the decision too, because the gate and the promoter tolerate dirt
    that is disjoint from the round's own diff. A rename reports both sides.
    """
    r = git(repo, "status", "--porcelain")
    if r.returncode != 0:
        return []
    out: list[str] = []
    for ln in r.stdout.splitlines():
        if not ln.strip():
            continue
        entry = ln[3:].strip() if len(ln) > 3 else ln.strip()
        for part in entry.split(" -> "):
            part = part.strip().strip('"')
            if part and part not in out:
                out.append(part)
    return out[:limit] if limit else out


def rebase_onto(worktree: Path, onto: str, *, upstream: str | None = None) -> tuple[bool, str, list[str]]:
    """Rebase the round's branch onto `onto`. `(ok, detail, conflicting_paths)`.

    The tree is shared. A human commits to `main` while a round is open, and
    until 2026-09-09 that turned every later gate and every landing into a
    refusal — "something landed under you; abort and re-cut", which threw
    away the round's diff to reapply it by hand onto a base one commit newer.
    A rebase is that reapplication, done by git, and what follows it in the
    ladder is the retest.

    Fails closed and leaves nothing half-done: a conflict is aborted so the
    worktree is exactly as it was, and the conflicting paths come back so the
    round can resolve them by hand and gate again. A worktree with uncommitted
    changes is refused rather than autostashed — those changes are not in the
    round's diff either way, and carrying them silently across a rebase is
    how a round comes to believe it gated work it never committed.

    `upstream` is the round's recorded base, and passing it keeps the replay
    to the round's own commits (`git rebase --onto <onto> <upstream>`). A bare
    `git rebase <onto>` replays everything reachable from the branch and not
    from `onto` — which, when the base was a promotion the guardian has since
    reset away, is that promotion too, re-landed under a new sha and a tree
    hash the denylist cannot match.
    """
    if not is_clean(worktree):
        return False, "worktree has uncommitted changes — commit them before gating", []
    r = git(worktree, *(("rebase", "--onto", onto, upstream) if upstream else ("rebase", onto)))
    if r.returncode == 0:
        return True, "", []
    c = git(worktree, "diff", "--name-only", "--diff-filter=U")
    conflicts = [ln.strip() for ln in c.stdout.splitlines() if ln.strip()]
    git(worktree, "rebase", "--abort")
    return False, (r.stderr or r.stdout).strip()[:400], conflicts


def has_merge_commits(repo: Path, base: str, head_ref: str) -> bool:
    r = git(repo, "rev-list", "--count", "--merges", f"{base}..{head_ref}")
    return r.returncode == 0 and r.stdout.strip() not in ("", "0")


def remove(round_id: str, *, keep_branch: bool = False, repo: Path | None = None) -> None:
    """Remove the worktree. On failure we keep the branch — it is the only
    forensic record of what was attempted."""
    repo = repo or LIVE_ROOT
    wt = worktree_path(round_id)
    if wt.exists():
        git(repo, "worktree", "remove", "--force", str(wt))
    git(repo, "worktree", "prune")
    if not keep_branch:
        git(repo, "branch", "-D", f"automod/{round_id}")
    rd = round_dir(round_id)
    if rd.exists():
        shutil.rmtree(rd, ignore_errors=True)


def prune(repo: Path | None = None) -> None:
    """Drop registrations whose directories are gone, without reading the list."""
    git(repo or LIVE_ROOT, "worktree", "prune")


class WorktreeListUnavailable(RuntimeError):
    """`git worktree list` could not be read, so its emptiness means nothing."""


def list_registered(repo: Path | None = None) -> list[str]:
    """Every registered worktree path. Raises rather than returning `[]` on a
    failed read.

    `git()` runs with `check=False`, so the failure mode of the old inline
    read was an empty stdout and an empty list — indistinguishable from "this
    repo has no worktrees". Measured on 2026-09-17: `prune_orphans` against a
    directory that is not a repo returned `[]` with no exception, exit 128.
    Both callers of that list are guards, and an empty list satisfies both of
    them: the implement loop's depth check passes at zero, and a landing check
    iterated over zero rounds finds no landing. A read that cannot be taken is
    a different answer from a read that came back empty, and it has to be
    sayable — hence the raise.
    """
    repo = repo or LIVE_ROOT
    r = git(repo, "worktree", "list", "--porcelain")
    if r.returncode != 0:
        raise WorktreeListUnavailable(
            f"git -C {repo} worktree list failed rc={r.returncode}: "
            f"{(r.stderr or '').strip()[:200]}")
    return [line.split(" ", 1)[1] for line in r.stdout.splitlines()
            if line.startswith("worktree ")]


def prune_orphans(repo: Path | None = None) -> list[str]:
    """Drop worktree registrations whose directories are gone."""
    repo = repo or LIVE_ROOT
    git(repo, "worktree", "prune")
    return list_registered(repo)
