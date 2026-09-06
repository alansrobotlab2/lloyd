"""Restore the repo to its last known good commit and bring the stack back.

Stdlib only. Every step is idempotent and the whole procedure is resumable:
the `rollback_started` intent record is written *before* anything is touched,
so a guardian that dies mid-rollback picks up where it left off on restart.

Two orderings in here are load-bearing and non-obvious:

**Stop the writers before moving the floor.** The agent is the thing writing
into this repo. Running `git reset --hard` while an `Edit` tool call is
mid-write produces a half-applied revert, which is strictly worse than either
version. So the backend goes down first (it drives the MCP), then the
aggregator, and only then does the tree move. This is also why
`lloyd-backend.conf` and `lloyd-mcp.conf` gained `stopasgroup`/`killasgroup` —
without those a Bash tool's child process outlives the stop and keeps writing.

**`git clean` is path-scoped, never repo-root.** The root holds `usage.db`,
`workers.db`, `mc-state.json`, `.env` and `.venvs/` — all gitignored, none of
them replaceable. A bare `git clean -fdx` here would be a data-loss event. The
scoped clean exists only to remove new `.py` files a bad commit added, which
`reset --hard` alone leaves behind and which can shadow an import.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


class RollbackError(RuntimeError):
    pass


def _git(repo: str, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def head_commit(repo: str) -> str | None:
    r = _git(repo, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def head_branch(repo: str) -> str | None:
    r = _git(repo, "symbolic-ref", "--quiet", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def commit_exists(repo: str, sha: str) -> bool:
    return _git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def is_ancestor(repo: str, ancestor: str, descendant: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def _wait_for_index_lock(repo: str, stale_seconds: float, budget: float = 10.0) -> None:
    lock = Path(repo) / ".git" / "index.lock"
    deadline = time.time() + budget
    while time.time() < deadline:
        if not lock.exists():
            return
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            return
        if age > stale_seconds:
            # A stale lock from a killed git blocks everything downstream and
            # is the common case after a hard stop.
            try:
                lock.unlink()
            except OSError:
                pass
            return
        time.sleep(0.5)


def _drain_writers(repo: str, budget: float) -> list[int]:
    """Wait for processes holding write fds under `repo` to exit. Best-effort."""
    deadline = time.time() + budget
    holders: list[int] = []
    repo_real = os.path.realpath(repo)
    while time.time() < deadline:
        holders = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            try:
                cwd = os.readlink(entry / "cwd")
            except OSError:
                continue
            if not cwd.startswith(repo_real):
                continue
            holders.append(pid)
        if not holders:
            return []
        time.sleep(1.0)
    return holders


def preserve_evidence(repo: str, broken_dir: Path, tag: str) -> dict:
    """Tag HEAD, stash the dirty tree, and copy untracked source aside.

    A rollback that erases the bug guarantees you fix it twice. The tag also
    pins the commit against garbage collection once the branch ref moves off
    it.
    """
    out: dict = {"tag": None, "stash": None, "patch": None, "untracked": []}
    dest = Path(broken_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if _git(repo, "tag", tag, "HEAD").returncode == 0:
        out["tag"] = tag

    diff = _git(repo, "diff", "HEAD")
    if diff.returncode == 0 and diff.stdout.strip():
        patch = dest / "dirty.patch"
        patch.write_text(diff.stdout, encoding="utf-8")
        out["patch"] = str(patch)

    ls = _git(repo, "ls-files", "--others", "--exclude-standard")
    if ls.returncode == 0:
        for rel in ls.stdout.split():
            if not rel.endswith(".py"):
                continue
            src = Path(repo) / rel
            try:
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                out["untracked"].append(rel)
            except OSError:
                pass

    stash_name = f"guardian-rollback-{tag}"
    st = _git(repo, "stash", "push", "-u", "-m", stash_name)
    if st.returncode == 0 and "No local changes" not in st.stdout:
        out["stash"] = stash_name
    return out


def restore_tree(repo: str, target: str, clean_paths: tuple[str, ...],
                 pycache_paths: tuple[str, ...]) -> None:
    """reset --hard to `target`, scoped-clean, and drop stale bytecode."""
    r = _git(repo, "reset", "--hard", target)
    if r.returncode != 0:
        raise RollbackError(f"git reset --hard {target[:8]} failed: {r.stderr.strip()[:300]}")

    # NOT `git checkout <sha>` — that detaches HEAD, and the isolation model
    # requires the live tree stay on `main`. reset --hard moves the branch ref
    # and the worktree together.
    existing = [p for p in clean_paths if (Path(repo) / p).exists()]
    if existing:
        _git(repo, "clean", "-fd", "--", *existing)

    for rel in pycache_paths:
        base = Path(repo) / rel
        if not base.is_dir():
            continue
        for cache in base.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)


def verify_tree(repo: str, target: str, expected_branch: str = "main") -> None:
    """Confirm the tree moved AND is still on a branch, not detached.

    The branch check is not cosmetic: `git checkout <sha>` would satisfy the
    commit assertion while detaching HEAD, and the isolation model requires
    the live tree stay on its branch so the next promotion can fast-forward.
    """
    actual = head_commit(repo)
    if actual != target:
        raise RollbackError(f"tree is at {actual} after reset, expected {target}")
    branch = head_branch(repo)
    want = f"refs/heads/{expected_branch}"
    if branch != want:
        raise RollbackError(f"HEAD is {branch!r}, expected {want} — refusing to restart")


def swap_venv_back(repo: str) -> str | None:
    """Undo a promoted venv clone by renaming the previous one back."""
    venvs = Path(repo) / ".venvs"
    live, prev = venvs / "lloyd", venvs / "lloyd.prev"
    if not prev.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    failed = venvs / f"lloyd.failed-{stamp}"
    if live.exists():
        os.rename(live, failed)
    os.rename(prev, live)
    return str(failed)
