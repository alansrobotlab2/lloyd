"""Per-session NDJSON event log + content-addressed blob store (#345 Stage 0).

The chat-transcript JSON (`~/lloyd/sessions/<id>.json`) is the human-facing
record. This module is the *machine-facing* record — every machination on
both the agent and the critic sides is captured here, append-only, never
overwritten.

Storage layout:
    ~/lloyd/event_logs/<session_id>.events.jsonl   # one JSON object per line
    ~/lloyd/event_logs/blobs/<sha256>.txt          # large fields by hash

Why:
    SQLite tables (`inner_voice_critiques`, `inner_voice_interventions`)
    capture decision summaries — what fired, what severity, what action.
    They discard prompts, raw responses, intermediate parse failures, and
    the causal chain between critique → intervention → outcome. The event
    log is the authoritative substrate for forensic analysis. Ships in
    Stage 0, before the critic is even wired up, so we have baseline data
    from session 1.

Design constraints:
    - Append-only writes. Never `vault_write`-style truncate.
    - Atomic per-line writes — partial events on crash are unacceptable.
    - Large fields (prompts, raw responses) deduplicate via content-address.
    - Never log secrets. Callers are responsible for filtering API keys /
      session cookies / tokens before passing payloads in.
    - No retention cap by default. Disk is cheap; the audit trail is the
      point. Rotation is a separate concern (cron, not online).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-server")

# Layout. Resolved relative to the lloyd repo root via paths.py would be
# tidier but creating a circular import isn't worth it — the layout has
# been stable since project inception.
_LLOYD_ROOT = Path(__file__).resolve().parent.parent  # ~/lloyd
EVENT_LOGS_DIR = _LLOYD_ROOT / "event_logs"
BLOBS_DIR = EVENT_LOGS_DIR / "blobs"

# Default threshold: fields larger than this go to the blob store and the
# event references them by `{"$blob": "<sha256>"}`. Tuned in Stage 0; can
# move into config.yaml later without changing the wire format.
DEFAULT_BLOB_THRESHOLD_BYTES = 4096

# Per-process file-handle lock keyed on session_id. Multiple turns run
# serially per session (the queue enforces this), but the consumer task
# may overlap with the post-capture ensure_future from a previous turn,
# so we still gate writes per session to avoid interleaved partial lines.
_session_locks: dict[str, threading.Lock] = {}
_session_locks_meta_lock = threading.Lock()


def _ensure_dirs() -> None:
    EVENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    BLOBS_DIR.mkdir(parents=True, exist_ok=True)


def _get_lock(session_id: str) -> threading.Lock:
    with _session_locks_meta_lock:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _store_blob(content: str) -> str:
    """Write `content` to a content-addressed blob and return its sha256.

    Repeated writes of the same bytes are no-ops at the filesystem level
    — the file is created once and reused. This is how repeated persona
    prompts (same persona, same context) deduplicate naturally.
    """
    _ensure_dirs()
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    path = BLOBS_DIR / f"{sha}.txt"
    if not path.exists():
        # Atomic-ish: write to .tmp, fsync, rename. Crash leaves either no
        # file or the complete file. The dedup property means even a
        # double-write at the same hash is safe.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    return sha


def _maybe_externalize(value: Any, threshold_bytes: int) -> Any:
    """Replace large strings with `{"$blob": "<sha>"}` references.

    Recurses into dicts and lists. Anything other than a "long string" is
    passed through unchanged. Threshold applies to the UTF-8 encoded
    length of the string, not its character count.
    """
    if isinstance(value, str):
        if len(value.encode("utf-8")) >= threshold_bytes:
            return {"$blob": _store_blob(value), "size": len(value)}
        return value
    if isinstance(value, dict):
        return {k: _maybe_externalize(v, threshold_bytes) for k, v in value.items()}
    if isinstance(value, list):
        return [_maybe_externalize(v, threshold_bytes) for v in value]
    return value


def log_event(
    session_id: str,
    event: str,
    data: dict[str, Any] | None = None,
    *,
    turn_id: str | None = None,
    blob_threshold_bytes: int = DEFAULT_BLOB_THRESHOLD_BYTES,
) -> int | None:
    """Append one event to `~/lloyd/event_logs/<session_id>.events.jsonl`.

    Schema:
        {
          "ts": "2026-05-01T14:02:33.142Z",
          "session_id": "...",
          "turn_id": "...",        # optional
          "event": "brain1.tool_call_proposed",
          "data": { ... event-specific ... }
        }

    Returns the 0-indexed line number where this event landed, computed
    under the same per-session lock as the append. Stage 2+ persists this
    offset on `inner_voice_critiques` rows so SQLite can pivot back to the
    raw event for click-to-detail UI. Returns None on any error (caught and
    logged) — historical callers ignore the return value, so this remains
    backward-compatible.

    Failures are caught and logged at WARNING — this is best-effort
    telemetry; we never want a misformed event to break the chat path.
    """
    if not session_id:
        # Defensive: a session_id-less event would write to ".events.jsonl"
        # which is meaningless. Caller bug — log + drop.
        logger.warning(f"log_event dropped (no session_id): {event}")
        return None
    try:
        _ensure_dirs()
        payload: dict[str, Any] = {
            "ts": _utc_iso(),
            "session_id": session_id,
            "event": event,
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
        if data is not None:
            payload["data"] = _maybe_externalize(data, blob_threshold_bytes)
        else:
            payload["data"] = {}
        line = json.dumps(payload, ensure_ascii=False, default=str)
        path = EVENT_LOGS_DIR / f"{session_id}.events.jsonl"
        with _get_lock(session_id):
            # Compute the offset under the lock so concurrent appends from
            # other tasks can't shift it. `count_events` is a separate code
            # path that doesn't take the same lock — racy by design, but
            # only Inner Voice cares about the precise offset and it always
            # goes through this function.
            offset = 0
            if path.exists():
                with path.open("rb") as fh:
                    offset = sum(1 for _ in fh)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")
            return offset
    except Exception as e:
        logger.warning(f"log_event failed (session={session_id}, event={event}): {e}")
        return None


async def log_event_async(
    session_id: str,
    event: str,
    data: dict[str, Any] | None = None,
    *,
    turn_id: str | None = None,
    blob_threshold_bytes: int = DEFAULT_BLOB_THRESHOLD_BYTES,
) -> None:
    """Async-friendly wrapper. Currently runs sync inline (writes are
    cheap, lock is per-session, fsync is left to the OS) but exposing
    the async signature lets us swap in an executor pool later without
    touching call sites.
    """
    log_event(
        session_id, event, data,
        turn_id=turn_id,
        blob_threshold_bytes=blob_threshold_bytes,
    )


def read_events(
    session_id: str,
    *,
    offset: int = 0,
    limit: int = 200,
    expand_blobs: bool = False,
) -> list[dict[str, Any]]:
    """Read parsed events from a session's log.

    `offset` is a line index (0-based). `limit` caps the return slice.
    `expand_blobs` resolves `{"$blob": ...}` references back to inline
    strings — costs disk reads, off by default for paginated UI use.

    Lines that fail to parse are skipped and logged. Missing files
    return an empty list.
    """
    path = EVENT_LOGS_DIR / f"{session_id}.events.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i < offset:
                    continue
                if len(out) >= limit:
                    break
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"skipping malformed event line {i} in {path.name}: {e}")
                    continue
                if expand_blobs:
                    obj = _expand_blobs(obj)
                out.append(obj)
    except Exception as e:
        logger.warning(f"read_events failed for session {session_id}: {e}")
    return out


def count_events(session_id: str) -> int:
    """Cheap line count. Used by /api/inner_voice/event_log for pagination."""
    path = EVENT_LOGS_DIR / f"{session_id}.events.jsonl"
    if not path.exists():
        return 0
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def read_blob(sha: str) -> str | None:
    """Resolve a blob reference back to its content. Returns None on miss."""
    path = BLOBS_DIR / f"{sha}.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"read_blob failed for {sha}: {e}")
        return None


def _expand_blobs(obj: Any) -> Any:
    """Recursively replace `{"$blob": <sha>, ...}` with the resolved string."""
    if isinstance(obj, dict):
        if "$blob" in obj and isinstance(obj["$blob"], str):
            content = read_blob(obj["$blob"])
            if content is not None:
                return content
            return {"$blob_missing": obj["$blob"]}
        return {k: _expand_blobs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_blobs(v) for v in obj]
    return obj
