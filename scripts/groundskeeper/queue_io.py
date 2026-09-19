#!/usr/bin/env python3
"""The one guarded writer for the groundskeeper queue (backlog #899).

On 2026-03-30 a fix loop processed four ``BROKEN_LINK`` items and wrote only
those four back over a queue holding ~2,700, destroying the accumulated
``done``/``skipped`` status of everything else; by the queue's own idempotency
rule that loss is unrecoverable (``memory/learnings/2026-03-30-groundskeeper-queue-corruption.md``).
The survey grew a verify-then-rename guard after it
(``scripts/groundskeeper/groundskeeper-survey.py``), but the two consumers in
``scripts/memory/`` — ``process-groundskeeper-queue.py`` and
``batch-process-orphans.py`` — kept finishing with a bare
``open(queue, 'w'); json.dump`` and re-dumped the whole queue nightly from
outside that guard, leaving no record of which process wrote the file.

Every caller now shares this module:

* ``write_queue_atomic`` serialises to ``<path>.tmp``, re-reads the temp file,
  and renames only when the re-read item count equals the in-memory count. On
  any mismatch it deletes the temp file, raises ``QueueWriteError`` and leaves
  the original queue byte-identical — so a partial write is refused instead of
  installed.
* Every accepted write appends one attribution row to
  ``groundskeeper-writes.jsonl`` naming the writing script, its pid, the number
  of items it stamped, the ``generated_at`` value it read and a UTC timestamp,
  so a future unattributed pass is identifiable from that file alone.
* ``append_process_log`` writes the per-item rows to the existing
  ``groundskeeper-log.jsonl`` in one open, so each stamped item carries its own
  reason instead of one writer stamping a single reason for everything.

The queue path is injectable — ``write_queue_atomic`` takes it as an argument,
and both this module's callers (the survey and the two consumers) read their
default from ``GROUNDKEEPER_QUEUE`` — because a guard that can only be exercised
against the live 28k-item queue is a guard no test can prove.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

#: Sidecar naming the process behind every accepted queue write.
WRITES_LOG_NAME = "groundskeeper-writes.jsonl"
#: Pre-existing per-item process log, one row per stamped item.
PROCESS_LOG_NAME = "groundskeeper-log.jsonl"


class QueueWriteError(RuntimeError):
    """A queue write was refused before it could replace the live queue."""


def _utc_now() -> str:
    """UTC second, matching the format the queue and log already carry."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialise(queue: dict, fh) -> None:
    """Write the queue body. Its own function so a test can inject a short write."""
    json.dump(queue, fh, indent=2)


def _writer_name(writer: str | None) -> str:
    """The writing script's own name, for the attribution row."""
    if writer:
        return writer
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] else ""
    return os.path.basename(argv0) or "<unknown>"


def writes_log_for(queue_path) -> Path:
    """The attribution sidecar that belongs to this queue file."""
    return Path(queue_path).with_name(WRITES_LOG_NAME)


def append_process_log(log_path, rows) -> int:
    """Append stamped-item rows to the process log, one JSON row each.

    An empty batch writes nothing and does not create the file: a pass with
    nothing to stamp must not leave the impression that it did something.
    """
    rows = list(rows)
    if not rows:
        return 0
    with open(log_path, "a") as log:
        for row in rows:
            log.write(json.dumps(row) + "\n")
    return len(rows)


def write_queue_atomic(
    path,
    queue: dict,
    *,
    stamped: int = 0,
    writer: str | None = None,
    writes_path=None,
) -> dict:
    """Replace the queue at ``path`` only after the temp copy reads back whole.

    Returns the attribution row it appended. Raises ``QueueWriteError`` — with
    the live queue untouched — when the queue carries no ``items`` list, when
    serialising or re-reading the temp file fails, or when the re-read item
    count disagrees with the in-memory count.
    """
    path = Path(path)
    items = queue.get("items")
    if items is None:
        raise QueueWriteError(
            f"refusing to write {path}: queue has no 'items' list "
            f"(keys: {sorted(queue)})"
        )
    expected = len(items)
    # The queue it is replacing, read before anything is written: a rebuild
    # that dropped generated_at is a rebuild that went wrong.
    read_generated_at = queue.get("generated_at")

    temp = Path(str(path) + ".tmp")
    try:
        with open(temp, "w") as fh:
            _serialise(queue, fh)
        # Verify from disk, not from memory: the whole point is that the bytes
        # on disk may not be the bytes we meant to send.
        with open(temp, "r") as fh:
            actual = len(json.load(fh).get("items", []))
    except Exception as exc:
        if temp.exists():
            temp.unlink()
        raise QueueWriteError(f"queue write to {temp} failed: {exc}") from exc

    if actual != expected:
        temp.unlink()
        raise QueueWriteError(
            f"Queue corruption detected: item count mismatch, expected {expected}, "
            f"got {actual}; {path} left unchanged"
        )

    os.rename(temp, path)

    row = {
        "written_at": _utc_now(),
        "writer": _writer_name(writer),
        "pid": os.getpid(),
        "stamped": int(stamped),
        "queue_generated_at": read_generated_at,
        "items": expected,
        "queue": str(path),
    }
    writes_path = Path(writes_path) if writes_path else writes_log_for(path)
    with open(writes_path, "a") as log:
        log.write(json.dumps(row) + "\n")
    return row
