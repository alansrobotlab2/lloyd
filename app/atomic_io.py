"""Atomic text-file writes.

Write to a sibling tmp file, then os.replace() onto the destination so a
crash, OOM-kill, or power loss mid-write can never leave a truncated file.
Used for state that is gitignored and expensive or impossible to recreate
(session transcripts, the fact-relationships graph) and for config.yaml
rewrites.
"""
import os
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
    whose loss is unrecoverable (e.g. the relationships index). Callers are
    expected to serialize concurrent writes to the same path themselves
    (all current call sites already hold a per-file lock or are
    single-writer).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
