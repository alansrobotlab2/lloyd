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
branch twice, so every round gets its own `autoimplement/<round_id>` branch.
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


def create(round_id: str, base: str = "HEAD", repo: Path | None = None) -> Path:
    """Create `<work>/<round_id>/home/lloyd` on branch `autoimplement/<round_id>`."""
    repo = repo or LIVE_ROOT
    wt = worktree_path(round_id)
    wt.parent.mkdir(parents=True, exist_ok=True)
    branch = f"autoimplement/{round_id}"
    r = git(repo, "worktree", "add", "-q", "-b", branch, str(wt), base)
    if r.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {r.stderr.strip()[:400]}")
    return wt


def changed_paths(worktree: Path, base: str) -> list[str]:
    """Repo-relative paths changed between `base` and the worktree's HEAD."""
    r = git(worktree, "diff", "--name-only", f"{base}...HEAD")
    if r.returncode != 0:
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def head(worktree: Path) -> str | None:
    r = git(worktree, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def is_clean(repo: Path) -> bool:
    r = git(repo, "status", "--porcelain")
    return r.returncode == 0 and not r.stdout.strip()


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


def rebase_onto(worktree: Path, onto: str) -> tuple[bool, str, list[str]]:
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
    """
    if not is_clean(worktree):
        return False, "worktree has uncommitted changes — commit them before gating", []
    r = git(worktree, "rebase", onto)
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
        git(repo, "branch", "-D", f"autoimplement/{round_id}")
    rd = round_dir(round_id)
    if rd.exists():
        shutil.rmtree(rd, ignore_errors=True)


def prune_orphans(repo: Path | None = None) -> list[str]:
    """Drop worktree registrations whose directories are gone."""
    repo = repo or LIVE_ROOT
    git(repo, "worktree", "prune")
    r = git(repo, "worktree", "list", "--porcelain")
    return [line.split(" ", 1)[1] for line in r.stdout.splitlines()
            if line.startswith("worktree ")]
