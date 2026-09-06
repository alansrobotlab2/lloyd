"""Read the current git commit without shelling out.

`/health` reports the commit the running process booted from, and that field
is what proves a self-modification rollback actually took effect: `git reset`
proves the *filesystem* changed, but only `/health.commit == last_known_good`
proves the *running service* changed.

Deliberately not `git rev-parse HEAD`:

  * a subprocess costs 5-15ms, and /health must stay cheap enough for a
    watchdog to poll every few seconds;
  * `git rev-parse` can transiently fail mid-rewrite (index.lock held), and a
    health endpoint that flaps because git is busy is worse than useless;
  * the guardian needs the same logic in a stdlib-only process.

Handles `.git` as a directory and as a worktree pointer file (`gitdir: ...`),
follows symbolic refs, and falls back to `packed-refs`. Returns None rather
than raising — a missing commit is reported as null, never as a 500.
"""

from __future__ import annotations

from pathlib import Path

_HEX = set("0123456789abcdef")


def _is_sha(value: str) -> bool:
    return len(value) == 40 and all(c in _HEX for c in value.lower())


def _resolve_git_dir(root: Path) -> Path | None:
    """Return the .git directory for `root`, following a worktree pointer."""
    dot_git = root / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        # A linked worktree: ".git" is a file containing "gitdir: /abs/path".
        try:
            text = dot_git.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if text.startswith("gitdir:"):
            target = Path(text.split(":", 1)[1].strip())
            if not target.is_absolute():
                target = (root / target).resolve()
            if target.is_dir():
                return target
    return None


def _read_packed_ref(git_dir: Path, ref: str) -> str | None:
    packed = git_dir / "packed-refs"
    if not packed.exists():
        # In a linked worktree, packed-refs lives in the main .git dir.
        commondir = git_dir / "commondir"
        if commondir.exists():
            try:
                rel = commondir.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            candidate = (git_dir / rel).resolve() / "packed-refs"
            if candidate.exists():
                packed = candidate
            else:
                return None
        else:
            return None
    try:
        for line in packed.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "^")):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1] == ref and _is_sha(parts[0]):
                return parts[0].lower()
    except OSError:
        return None
    return None


def head_commit(root: Path | str) -> str | None:
    """Return the 40-hex commit HEAD points at, or None if it can't be read."""
    try:
        root = Path(root)
        git_dir = _resolve_git_dir(root)
        if git_dir is None:
            return None
        head_file = git_dir / "HEAD"
        raw = head_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if _is_sha(raw):
        return raw.lower()  # detached HEAD

    if not raw.startswith("ref:"):
        return None
    ref = raw.split(":", 1)[1].strip()

    # Loose ref first. In a linked worktree, refs/heads/* resolve through
    # commondir, so try both locations.
    candidates = [git_dir / ref]
    commondir = git_dir / "commondir"
    if commondir.exists():
        try:
            rel = commondir.read_text(encoding="utf-8").strip()
            candidates.append((git_dir / rel).resolve() / ref)
        except OSError:
            pass
    for candidate in candidates:
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if _is_sha(value):
            return value.lower()

    return _read_packed_ref(git_dir, ref)


def head_branch(root: Path | str) -> str | None:
    """Return the branch name HEAD is on, or None when detached/unreadable."""
    try:
        git_dir = _resolve_git_dir(Path(root))
        if git_dir is None:
            return None
        raw = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw.startswith("ref: refs/heads/"):
        return raw[len("ref: refs/heads/"):].strip() or None
    return None
