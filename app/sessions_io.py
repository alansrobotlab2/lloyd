"""Session metadata persistence and per-session turn queue.

Session data lives as one JSON file per session under `SESSIONS_DIR`.

Concurrency model (task #296):
    Each session_id has a `SessionQueue` with two FIFO deques — one for
    user turns, one for ambient (background-producer) turns. User turns
    always win: the consumer pops `pending_user` before `pending_ambient`,
    and enqueueing a user turn while ambient is running sets the
    current-turn `cancel_event` so the user turn can take over.

    A lazy per-session consumer task drains both deques serially —
    replacing the old `_active_streams: dict[str, Event]` check-then-act
    model that raced on near-simultaneous POSTs.
"""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from app.paths import SESSIONS_DIR

logger = logging.getLogger("lloyd-server")


TurnSource = Literal["user", "ambient", "system"]


@dataclass
class SessionTurn:
    """A single enqueued turn. Payload is opaque; producer owns its shape.

    `events` is an unbounded asyncio.Queue. The consumer's `_run_turn`
    pushes `{"event": <name>, "data": <dict>}` or the sentinel `None` on
    end-of-turn. SSE responders subscribe by awaiting from this queue.
    """
    turn_id: str
    source: TurnSource
    payload: dict[str, Any]
    enqueued_at: datetime
    started_at: Optional[datetime] = None
    # Non-None iff turn was preempted by a user turn while running.
    preempted: bool = False
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class SessionQueue:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_user: deque = field(default_factory=deque)
    pending_ambient: deque = field(default_factory=deque)
    current: Optional[SessionTurn] = None
    # Fresh Event per turn — consumer rotates this when a turn is promoted
    # to current. Readers must go through `get_cancel_event()` which also
    # checks that a turn is actually running.
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    consumer_task: Optional[asyncio.Task] = None


_session_queues: dict[str, SessionQueue] = {}


def _get_or_create_queue(session_id: str) -> SessionQueue:
    q = _session_queues.get(session_id)
    if q is None:
        q = SessionQueue()
        _session_queues[session_id] = q
    return q


def is_session_active(session_id: str) -> bool:
    """True if a turn is running OR pending for this session."""
    q = _session_queues.get(session_id)
    if q is None:
        return False
    return q.current is not None or bool(q.pending_user) or bool(q.pending_ambient)


def get_cancel_event(session_id: str) -> Optional[asyncio.Event]:
    """Cancel-event for the currently-running turn, or None if idle."""
    q = _session_queues.get(session_id)
    if q is None or q.current is None:
        return None
    return q.cancel_event


def get_current_turn(session_id: str) -> Optional[SessionTurn]:
    q = _session_queues.get(session_id)
    if q is None:
        return None
    return q.current


def get_queue_state(session_id: str) -> dict[str, Any]:
    """Snapshot of queue state for the /queue endpoint."""
    q = _session_queues.get(session_id)
    if q is None:
        return {
            "current": None,
            "pending_user": 0,
            "pending_ambient": 0,
            "depth": 0,
        }
    cur = q.current
    return {
        "current": {
            "turn_id": cur.turn_id,
            "source": cur.source,
            "started_at": cur.started_at.isoformat() if cur.started_at else None,
        } if cur else None,
        "pending_user": len(q.pending_user),
        "pending_ambient": len(q.pending_ambient),
        "depth": len(q.pending_user) + len(q.pending_ambient),
    }


# Max queued ambient turns per session. Producers can spam —
# drop the oldest beyond this cap so the queue doesn't grow unbounded.
AMBIENT_QUEUE_CAP = 3


async def enqueue_turn(session_id: str, turn: SessionTurn, consumer_factory) -> dict[str, Any]:
    """Enqueue a turn on the appropriate tier.

    `consumer_factory` is a zero-arg callable returning a coroutine — used
    to lazily spawn the per-session consumer task (passed in to avoid a
    circular import between sessions_io and routers.messages).

    User turns: appended to `pending_user`. If an ambient turn is
    currently running, sets `cancel_event` to preempt it.

    Ambient turns: appended to `pending_ambient`, with two policies:
      - `dedup_key` in payload collapses duplicates (newest wins — old
        entry is dropped so producers can spam safely).
      - Queue is capped at `AMBIENT_QUEUE_CAP`; oldest ambient is
        dropped when the cap is exceeded.

    Returns: turn_id, source, preempted, dropped (list of dropped turn_ids
    from dedup+cap), dedup (bool).
    """
    q = _get_or_create_queue(session_id)
    preempted = False
    dropped: list[str] = []
    deduped = False
    async with q.lock:
        if turn.source == "user":
            q.pending_user.append(turn)
            if q.current is not None and q.current.source == "ambient":
                q.current.preempted = True
                q.cancel_event.set()
                preempted = True
        else:
            dedup_key = turn.payload.get("dedup_key") if isinstance(turn.payload, dict) else None
            if dedup_key:
                # Collapse: newest wins. Drop any queued ambient sharing this key.
                keep = deque()
                for t in q.pending_ambient:
                    t_key = t.payload.get("dedup_key") if isinstance(t.payload, dict) else None
                    if t_key == dedup_key:
                        t.preempted = True
                        try: t.events.put_nowait(None)
                        except Exception: pass
                        t.done.set()
                        dropped.append(t.turn_id)
                        deduped = True
                    else:
                        keep.append(t)
                q.pending_ambient = keep
            q.pending_ambient.append(turn)
            # Cap: drop oldest until within limit.
            while len(q.pending_ambient) > AMBIENT_QUEUE_CAP:
                old = q.pending_ambient.popleft()
                old.preempted = True
                try: old.events.put_nowait(None)
                except Exception: pass
                old.done.set()
                dropped.append(old.turn_id)
        if q.consumer_task is None or q.consumer_task.done():
            q.consumer_task = asyncio.create_task(consumer_factory())
    # Broadcast state to the currently-running turn's subscribers so they
    # see queue depth change in real time.
    await _broadcast_queue_state(session_id)
    return {
        "turn_id": turn.turn_id,
        "source": turn.source,
        "preempted": preempted,
        "dropped": dropped,
        "deduped": deduped,
    }


async def _broadcast_queue_state(session_id: str) -> None:
    """Push a `queue_state` event into the running turn's broker so any
    SSE subscriber sees queue changes.

    No-op if no turn is currently running. (Clients with no active
    subscription fall back to polling GET /api/sessions/{id}/queue.)
    """
    q = _session_queues.get(session_id)
    if q is None or q.current is None:
        return
    state = get_queue_state(session_id)
    try:
        await q.current.events.put({"event": "queue_state", "data": state})
    except Exception:
        pass


async def drain_pending(session_id: str, source: Optional[TurnSource] = None) -> int:
    """Remove queued turns. If source is None, drains ambient only
    (the documented behavior for /cancel?drain_pending=true — user turns
    are never silently dropped). Returns number drained.
    """
    q = _session_queues.get(session_id)
    if q is None:
        return 0
    drained = 0
    async with q.lock:
        if source is None or source == "ambient":
            while q.pending_ambient:
                t = q.pending_ambient.popleft()
                t.preempted = True
                try:
                    t.events.put_nowait(None)
                except Exception:
                    pass
                t.done.set()
                drained += 1
        if source == "user":
            while q.pending_user:
                t = q.pending_user.popleft()
                try:
                    t.events.put_nowait(None)
                except Exception:
                    pass
                t.done.set()
                drained += 1
    if drained:
        await _broadcast_queue_state(session_id)
    return drained


# Per-session file locks. All mutations to a session's JSON file must go
# through `mutate_session` (or helpers built on it) to prevent concurrent
# read-modify-write from clobbering each other's changes. Previously,
# post_capture would read a snapshot, await for 10+ seconds on a
# secondary-model call, then write the stale snapshot back — wiping any
# messages that had been appended in the meantime.
_file_locks: dict[str, asyncio.Lock] = {}


def _get_file_lock(session_id: str) -> asyncio.Lock:
    lock = _file_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _file_locks[session_id] = lock
    return lock


async def mutate_session(session_id: str, fn) -> bool:
    """Atomic read-modify-write on a session file.

    Acquires the per-session lock, reads fresh data from disk, calls
    `fn(data)` (which mutates in place), writes back. Returns True if the
    mutation ran, False if the file doesn't exist.

    `fn` MUST be synchronous and fast — never await or do I/O inside it.
    Expensive work (LLM calls, etc.) must happen OUTSIDE this helper;
    only apply the result via a small `fn` callback.
    """
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    async with _get_file_lock(session_id):
        if not meta_path.exists():
            return False
        data = json.loads(meta_path.read_text())
        fn(data)
        meta_path.write_text(json.dumps(data, indent=2))
        return True


async def _save_session_meta(session_id: str, model: str, preview: str = ""):
    """Save session metadata to JSON file (creates if missing)."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    now = datetime.now().isoformat()
    async with _get_file_lock(session_id):
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            data["last_active"] = now
            if preview:
                data["preview"] = preview[:60]
            data["message_count"] = data.get("message_count", 0) + 1
        else:
            data = {
                "session_id": session_id,
                "model": model,
                "created_at": now,
                "last_active": now,
                "preview": preview[:60],
                "message_count": 1,
                "messages": [],
                "platform": "mission-control",
            }
        meta_path.write_text(json.dumps(data, indent=2))


async def _append_messages(session_id: str, new_messages: list[dict]):
    """Append messages to session metadata file."""
    def _append(data):
        msgs = data.get("messages", [])
        msgs.extend(new_messages)
        data["messages"] = msgs
        data["last_active"] = datetime.now().isoformat()
    await mutate_session(session_id, _append)
