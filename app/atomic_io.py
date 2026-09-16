"""Atomic text-file writes, and the lock that orders them.

Two halves, because the two failure shapes are different:

`atomic_write_text` writes to a sibling tmp file and `os.replace()`s it onto
the destination, so a crash, OOM-kill or power loss mid-write can never leave
a truncated file — and, just as important, a *reader* can never see one. A
plain `write_text` opens with `O_TRUNC`, so anyone reading while it writes
gets an empty file or a prefix. That half protects every reader, cooperative
or not, and it is why the memory files and every MCP-tool write go through it.

`locked_file` / `commit_lock` serialize the *writers*: two callers load the
same state, each makes a locally correct change, and the second save silently
erases the first. The read has to happen inside the lock, not before it (see
`scripts/memory/repair_fact_ids.py`) — a lock around only the write turns a
lost update into a slightly later lost update.

Used for state that is gitignored and expensive or impossible to recreate
(session transcripts, the fact-relationships graph), for config.yaml
rewrites, and for the shared memory files that are loaded into every system
prompt (`lloyd/MEMORY.md`, `lloyd/USER.md`) — those are written by three
separate lanes, so they use `commit_lock`, not a sibling lock file; see
`lock_file_for`.
"""
import contextlib
import errno
import fcntl
import hashlib
import os
import time
from pathlib import Path

from app.paths import LOGS_DIR

# Wait, don't queue forever: a chat turn that blocks on a stuck nightly writer
# has to come back with an error the model can act on, not hang. Callers that
# want to trade latency for waiting pass `timeout=`.
DEFAULT_LOCK_WAIT = 30.0

# Where locks for shared-state writes live. Deliberately outside any tree the
# write lands in: `~/obsidian/lloyd/.research-queue.lock` is the counterexample
# — a sibling lock file is a real file in the vault, so it shows up in
# `git status`, in Obsidian's file tree, and in whatever walks the vault, and
# that one is tracked in git. Content-addressed by realpath, so one file has
# one lock no matter which lane spells its path differently.
#
# Under `logs/` because that directory is already this repo's ignored runtime
# state (`/logs/`): a lock dir anywhere else in a checkout shows up as an
# untracked path and dirties the tree of whoever runs a memory writer — including
# an automod worktree, where a dirty tree aborts the round. Still under the repo
# root, which is what the vault-cleanliness clause needs.
LOCK_DIR = Path(os.environ.get("LLOYD_LOCK_DIR") or (LOGS_DIR / "locks"))

# Half-written temps go to the same off-tree directory, for the same reason:
# `os.replace` needs a *rename*, and a temp is a file that exists until the
# commit succeeds. A writer killed mid-write leaves it behind, and inside the
# vault that stray is exactly what the item this came from is about — the vault
# has no `*.tmp` rule in its .gitignore, `agent-obsidian-sync` pushes anything it
# finds, and Obsidian indexes it. On the same filesystem (both trees are on the
# `/home` btrfs here) the rename is atomic and off-tree costs nothing; on a
# different one `os.replace` raises EXDEV and the writer falls back to a sibling
# temp, which is the old behaviour and still correct.
SCRATCH_DIR = LOCK_DIR / "tmp"


def scratch_dir_for(path: Path | str) -> Path | None:
    """Where this writer's temp belongs, or None for a sibling temp.

    Only a target inside the vault needs the off-tree treatment: nothing else in
    the tree keeps a clean-file invariant that a stray breaks, and the existing
    callers under `~/lloyd` are already covered by that repo's ignore rules.
    """
    from app.paths import VAULT_ROOT
    try:
        inside = Path(os.path.realpath(str(path))).is_relative_to(
            os.path.realpath(str(VAULT_ROOT)))
    except (OSError, ValueError):
        return None
    return SCRATCH_DIR if inside else None


def write_text_durable(path: Path | str, text: str, *, fsync: bool = True) -> None:
    """`atomic_write_text` that leaves no sidecar inside the vault.

    The lane every MCP tool that writes shared state goes through.
    """
    atomic_write_text(path, text, fsync=fsync, tmp_dir=scratch_dir_for(path))


def lock_file_for(path: Path | str) -> Path:
    """The one lock file that serializes writers of `path`.

    Keyed on realpath, not on the spelling the caller used. `memory_add` builds
    `MEMORIES_ROOT / name`, `vault_write` takes a vault-relative path, and
    `Write`/`Edit` get an absolute path from the model — three spellings of
    `lloyd/MEMORY.md` must not pick three locks, because three locks is the
    same lost update with extra steps.
    """
    real = os.path.realpath(str(path))
    digest = hashlib.sha256(real.encode("utf-8")).hexdigest()[:32]
    return LOCK_DIR / f"{Path(real).name}-{digest}.lock"


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = False,
    tmp_dir: Path | str | None = None,
) -> None:
    """Atomically replace `path` with `text`.

    `fsync=True` flushes data to disk before the rename — use it for files
    whose loss is unrecoverable. Callers are expected to serialize concurrent
    writes to the same path themselves: fact-file writers take
    `locked_file(path)`, and every writer of a shared file — the memory files,
    any note through the MCP tool layer — takes `commit_lock(path)` and reads
    inside it.

    The temp file is named per-process so two writers racing on one path
    cannot clobber each other's half-written temp. With `tmp_dir` it lives there
    instead of beside the target — see `SCRATCH_DIR`; if the two are on different
    filesystems the rename cannot cross them, so the writer falls back to a
    sibling temp rather than failing the commit.
    """
    path = Path(path)
    tmp = _tmp_path(path, tmp_dir)
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    try:
        os.replace(tmp, path)
    except OSError as exc:
        # EXDEV: the scratch dir and the target are on different filesystems, so
        # no rename can join them. A sibling temp can.
        if tmp_dir is None or getattr(exc, "errno", None) != errno.EXDEV:
            raise
        tmp = _tmp_path(path, None)
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)


def _tmp_path(path: Path, tmp_dir: Path | str | None) -> Path:
    name = f"{path.name}.{os.getpid()}.tmp"
    if tmp_dir is None:
        return path.with_name(name)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _sweep_scratch(tmp_dir)
    return tmp_dir / name


# How old a leftover temp has to be before a writer collects it. Well past any
# write that is still in flight, short enough that one crash does not leave a
# permanent file.
SCRATCH_MAX_AGE_S = 3600.0


def _sweep_scratch(tmp_dir: Path) -> None:
    """Delete leftover temps a crashed writer left behind.

    The dir holds only this lane's own temps, and each is named
    `<target>.<pid>.tmp` — one per pid, so a crash leaves one file and the next
    crash leaves another, forever, while every `ls` of the directory and every
    disk-size probe counts them. Age-gated rather than pid-gated: a pid is reused
    and a writer alive 30 seconds ago is not a leftover. Best-effort — a sweep
    that cannot unlink must not fail the commit it was helping.
    """
    import time as _t
    cutoff = _t.time() - SCRATCH_MAX_AGE_S
    try:
        for stray in tmp_dir.glob("*.tmp"):
            try:
                if stray.stat().st_mtime < cutoff:
                    stray.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _flock_exclusive(lock_path: Path, timeout: float):
    """Open `lock_path` and block until it is ours. Returns the open fd."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"could not lock {lock_path} within {timeout}s")
                time.sleep(0.02)
    except BaseException:
        os.close(fd)
        raise


def _flock_release(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


@contextlib.contextmanager
def locked_file(path: Path | str, *, timeout: float = DEFAULT_LOCK_WAIT):
    """Hold an exclusive advisory lock covering writes to `path`.

    The lock lives on a sibling `<name>.lock` rather than the file itself, so
    it survives the `os.replace` in `atomic_write_text` — locking the target
    directly would leave every waiter holding a descriptor to the replaced
    inode.

    Needed because the extractor runs four worker threads and `fact_add` can
    fire from a chat turn at the same moment, and both do read-modify-write on
    one fact file. Without this, the later write silently drops whatever the
    earlier one added.

    Advisory, so it only excludes writers that also take it. Every writer of a
    fact file must. For state that lives inside a tree worth keeping clean
    (the vault) use `commit_lock` instead: it takes the same lock from a
    directory outside the tree.
    """
    path = Path(path)
    fd = _flock_exclusive(path.with_name(path.name + ".lock"), timeout)
    try:
        yield path
    finally:
        _flock_release(fd)


@contextlib.contextmanager
def commit_lock(path: Path | str, *, timeout: float = DEFAULT_LOCK_WAIT):
    """Serialize the read-modify-write of one shared file, off-tree.

    The critical section is the commit only — read the current bytes, apply the
    change, replace — not the caller's reasoning, network, or model call. Reads
    outside the lock stay concurrent, which is the point: fan-out sub-agents and
    parallel sessions keep working, they just cannot both commit.

    Raises `TimeoutError` after `timeout` rather than queueing: see
    `DEFAULT_LOCK_WAIT`.
    """
    lock_path = lock_file_for(path)
    fd = _flock_exclusive(lock_path, timeout)
    try:
        yield lock_path
    finally:
        _flock_release(fd)


def hash_bytes(data: bytes) -> str:
    """sha256 of the bytes actually read.

    The extractor hashed the file again after processing it. If the file
    changed in between — a note being appended to while the run walked the
    vault — the new content was recorded as already extracted and never was.
    """
    return hashlib.sha256(data).hexdigest()
