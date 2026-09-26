"""The next-session channel (#1516): what a nightly pass leaves for tomorrow.

An ambient signal is a message to a *session that is already open*. The queue that
carries it (`app.sessions_io._ambient_prefetch_queue`) is an in-process dict keyed
by session id, drained only by a turn for that exact id, and it dies with the
backend process. A note written at 03:00 for the first turn in the morning has
nowhere to go on that path, for two independent reasons that both hold tonight:

  * resolution — `get_active_session_id()` returns None when nothing qualifies,
    and a producer is documented to treat that as a no-op; and
  * deadline — even when a stale last-chat session *does* resolve, the ambient
    tier's default `ttl_seconds` is 3600, so a 03:00 note is purged by ~04:00,
    hours before anyone sits down.

So this is a different store, and the difference is the point: it is a file, so it
outlives the process that wrote it and needs no session to be addressed to; and its
deadline is a day, because that is the interval it is built to cross. What the
channel is *not* is a second memory: it holds at most `NEXT_SESSION_NOTE_CAP`
notes, each one delivered to exactly one turn and then gone.

Delivery is by the only position the architecture leaves free. Position 0 is frozen
for the turn (`architecture/harness.md`) and all dynamic context is prepended to
the *user message* (`architecture/subliminal.md`), so the renderer
(`prefetch._format_context`) renders these notes as a part of the `<context>` block
and nothing on this path may touch the system prompt.

Concurrency: writes come from the backend router and from anything a human or a
nightly task runs directly, so both the read and the write happen inside
`app.atomic_io.locked_file` — a lock around only the save would turn a lost update
into a slightly later one — and the save itself is an `os.replace`, so a reader can
never see a half-written store.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.atomic_io import locked_file, write_text_durable
from app.paths import DATA_ROOT

logger = logging.getLogger("lloyd-server")

#: Env override for tests and tools that must point the channel somewhere else
#: than the running tree's data root. Same shape as `LLOYD_DATA`, for the same
#: reason: the path is decided once, by the process, not by each caller.
STORE_PATH_ENV = "LLOYD_NEXT_SESSION_NOTES"

#: The file's name inside `DATA_ROOT`, beside `SESSIONS_DIR` and `MC_STATE_PATH`.
#: Runtime state, not a build input: repo `data/**` is the latter and is denied to
#: the self-modification loop.
FILE_NAME = "next-session-notes.json"

SCHEMA_VERSION = 1

# ── the two numbers the channel is specified by, declared exactly once ───────
#
# `CAP` bounds the store, not a turn: every note waiting is delivered, so a turn
# that opens onto three notes reads three notes and the store is empty after it.
# `TTL` is a day because the interval this crosses is a night; the ambient tier's
# hour-scale default is the number that cannot cross it, which is what the channel
# exists for. Both are read from here by the router, the renderer and the tests —
# a second copy of either is the defect, not the value.
NEXT_SESSION_NOTE_CAP = 5
NEXT_SESSION_NOTE_TTL_SECONDS = 24 * 3600

#: Render caps, charged before the text reaches a turn. Truncation is the honest
#: cost of carrying a note in a prompt rather than in a tool result, and the eval's
#: `sleep_notes` arm measures exactly that cost against the prefetch arm.
NEXT_SESSION_SUMMARY_MAX = 720
NEXT_SESSION_CONTENT_MAX = 2000


def store_path() -> Path:
    """Where the channel lives. Resolved per call, so `LLOYD_DATA` and the test
    override both take effect without re-importing the module.
    """
    override = (os.environ.get(STORE_PATH_ENV) or "").strip()
    if override:
        return Path(override)
    return DATA_ROOT / FILE_NAME


@dataclass
class NextSessionNote:
    """One note waiting for the next chat turn.

    `expires_at == 0.0` means no deadline was ever set — the ambient queue's
    sentinel and for the same reason (#910): read as an instant, epoch zero is
    permanently past, and every note that was never given a deadline would
    silently vanish.
    """
    source: str
    summary: str
    content: str = ""
    dedup_key: str = ""
    enqueued_at: float = 0.0
    expires_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary,
            "content": self.content,
            "dedup_key": self.dedup_key,
            "enqueued_at": self.enqueued_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NextSessionNote":
        return cls(
            source=str(raw.get("source") or ""),
            summary=str(raw.get("summary") or ""),
            content=str(raw.get("content") or ""),
            dedup_key=str(raw.get("dedup_key") or ""),
            enqueued_at=float(raw.get("enqueued_at") or 0.0),
            expires_at=float(raw.get("expires_at") or 0.0),
        )


def _expired(note: NextSessionNote, now: float) -> bool:
    """Has this note's deadline passed? The sentinel is checked first and is
    never expired, exactly as the ambient queue's filter does it.
    """
    return note.expires_at != 0.0 and note.expires_at <= now


def _read(path: Path) -> list[NextSessionNote]:
    """The store as notes. A missing file is an empty channel; unreadable bytes
    (a crash between two `os.replace`s is impossible, a hand-edit is not) are an
    empty channel too — a turn must not fail because a store it does not own is
    malformed, and the empty save that follows a drain replaces the damage.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    notes = raw.get("notes") if isinstance(raw, dict) else None
    if not isinstance(notes, list):
        return []
    out = []
    for item in notes:
        if isinstance(item, dict):
            with contextlib.suppress(TypeError, ValueError):
                out.append(NextSessionNote.from_dict(item))
    return out


def _save(path: Path, notes: list[NextSessionNote]) -> None:
    write_text_durable(path, json.dumps(
        {"version": SCHEMA_VERSION, "notes": [n.to_dict() for n in notes]},
        ensure_ascii=False, indent=1))


def write_next_session_note(source: str, summary: str, content: str = "",
                            dedup_key: str = "", ttl_seconds: int | None = None,
                            now: float | None = None) -> dict[str, Any]:
    """Add one note for the next chat turn. No target session, and none needed.

    Behaviour, all of it inside the lock so the read cannot go stale under a
    second writer:

      * A note already past its deadline is purged on write, and named in
        `dropped`. The ambient queue learned this the hard way (#910): a signal
        queued where nobody reads it was also a signal nothing reclaimed.
      * `dedup_key` defaults to `source`, as it does on the `/inject-prefetch`
        door, and collapses any note already waiting under that key — newest
        wins. A nightly pass that re-runs replaces its own brief instead of
        stacking copies and evicting everything else for nothing.
      * Above `NEXT_SESSION_NOTE_CAP` the oldest note goes, and its source is
        named in `dropped`.

    A `ttl_seconds` of 0 or less sets no deadline, which is the ambient queue's
    sentinel and means what it means there: the note waits until drained or
    evicted, never until a clock says so.

    Returns `{queued, queue_depth, dropped, deduped, path}` for the producer echo.
    """
    now = time.time() if now is None else float(now)
    ttl = NEXT_SESSION_NOTE_TTL_SECONDS if ttl_seconds is None else int(ttl_seconds)
    note = NextSessionNote(
        source=source, summary=summary, content=content,
        dedup_key=dedup_key or source,
        enqueued_at=now,
        expires_at=now + ttl if ttl > 0 else 0.0,
    )
    path = store_path()
    dropped: list[str] = []
    with locked_file(path):
        notes = _read(path)
        alive = []
        for n in notes:
            if _expired(n, now):
                dropped.append(n.source)
            else:
                alive.append(n)
        notes = alive

        # Newest wins under one key. The key defaults to the source, so a pass
        # that re-runs replaces its own brief instead of stacking copies and
        # evicting everything else for nothing.
        deduped = False
        if note.dedup_key:
            keep = [n for n in notes if n.dedup_key != note.dedup_key
                    or dropped.append(n.source)]
            deduped = len(keep) != len(notes)
            notes = keep
        # Sorted before the cap so `pop(0)` is provably the oldest note and not
        # merely the first one the file happened to hold: a producer may write
        # with a backdated `now`, and eviction that quietly drops the newest note
        # instead would be invisible until a morning turned up with no brief.
        notes.append(note)
        notes.sort(key=lambda n: n.enqueued_at)
        while len(notes) > NEXT_SESSION_NOTE_CAP:
            dropped.append(notes.pop(0).source)
        _save(path, notes)

    return {"queued": 1, "queue_depth": len(notes), "dropped": dropped,
            "deduped": deduped, "path": str(path)}


def drain_next_session_notes(now: float | None = None) -> list[NextSessionNote]:
    """Take every note waiting, and leave the channel empty: delivery consumes.

    Called by `prefetch._prefetch_prepare` on the caller's thread, in request
    order, so two turns arriving at once cannot both read the same note. Notes
    past their deadline never reach the turn; they are dropped and named in the
    log, because a path that removes a user-visible signal without a trace is the
    shape that hid #910.
    """
    now = time.time() if now is None else float(now)
    path = store_path()
    with locked_file(path):
        notes = _read(path)
        if not notes:
            return []
        alive = [n for n in notes if not _expired(n, now)]
        dead = [n.source for n in notes if _expired(n, now)]
        if dead:
            logger.info("next-session notes: dropped %d expired note(s): %s",
                        len(dead), ", ".join(dead))
        if len(alive) != len(notes) or notes:
            _save(path, [])
    return alive


def peek_next_session_notes() -> list[NextSessionNote]:
    """Snapshot without consuming — for the debug endpoint and for a human
    checking tomorrow whether a nightly pass delivered anything.
    """
    with locked_file(store_path()):
        return _read(store_path())
