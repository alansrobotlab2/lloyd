"""Atomic text-file writes.

Write to a sibling tmp file, then os.replace() onto the destination so a
crash, OOM-kill, or power loss mid-write can never leave a truncated file.
Used for state that is gitignored and expensive or impossible to recreate
(session transcripts, the fact-relationships graph) and for config.yaml
rewrites.
"""
import contextlib
import fcntl
import hashlib
import os
import time
from pathlib import Path


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = False,
) -> None:
    """Atomically replace `path` with `text`.

    `fsync=True` flushes data to disk before the rename — use it for files
    whose loss is unrecoverable. Callers are expected to serialize concurrent
    writes to the same path themselves: fact-file writers take
    `locked_file(path)`, everything else is single-writer.

    The temp file is named per-process so two writers racing on one path
    cannot clobber each other's half-written temp.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def locked_file(path: Path | str, *, timeout: float = 30.0):
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
    fact file must.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + timeout
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"could not lock {lock_path} within {timeout}s")
                time.sleep(0.02)
        try:
            yield path
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def hash_bytes(data: bytes) -> str:
    """sha256 of the bytes actually read.

    The extractor hashed the file again after processing it. If the file
    changed in between — a note being appended to while the run walked the
    vault — the new content was recorded as already extracted and never was.
    """
    return hashlib.sha256(data).hexdigest()
