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


# ---------------------------------------------------------------------------
# Active-session tracking (task #295)
#
# Producers (autonomy, cron, pipelines) need to target "the user's current
# session" without knowing session IDs. We track the last session that
# received a user turn in-memory; fallback is mtime-sorted mission-control
# session files.
# ---------------------------------------------------------------------------

_last_user_session_id: Optional[str] = None


def set_last_user_session(session_id: str) -> None:
    """Record that a user turn just enqueued for this session. Called from
    `post_message_stream` so `get_active_session_id()` can resolve "current
    session" without scanning the filesystem on the hot path.
    """
    global _last_user_session_id
    _last_user_session_id = session_id


def get_active_session_id(max_age_hours: float = 24.0) -> Optional[str]:
    """Best-effort "current user session" for ambient producers.

    Resolution:
      1. `_last_user_session_id` if set AND the session JSON still exists.
      2. Most-recent `platform: mission-control` session by mtime within
         `max_age_hours`. Explicitly excludes `platform: autonomy` so an
         autonomy task's own session never receives its own injection.

    Returns None if nothing qualifies — producers should treat this as a
    no-op (the user simply has no active chat session to notify).
    """
    import time as _time

    if _last_user_session_id:
        p = SESSIONS_DIR / f"{_last_user_session_id}.json"
        if p.exists():
            return _last_user_session_id

    if not SESSIONS_DIR.exists():
        return None

    cutoff = _time.time() - (max_age_hours * 3600)
    best: tuple[float, str] | None = None
    for sf in SESSIONS_DIR.glob("*.json"):
        try:
            mtime = sf.stat().st_mtime
            if mtime < cutoff:
                continue
            data = json.loads(sf.read_text())
            platform = data.get("platform", "mission-control")
            if platform == "autonomy":
                continue
            if best is None or mtime > best[0]:
                best = (mtime, data.get("session_id", sf.stem))
        except Exception:
            continue
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Ambient prefetch queue (task #295, Mechanism 1)
#
# Producers push entries here for priority=`ambient` injections. On the
# user's next turn, `prefetch.py` drains pending entries for the target
# session and appends them to the <context> block. No SDK turn is fired;
# this is the cheap passive path.
# ---------------------------------------------------------------------------

@dataclass
class AmbientPrefetchEntry:
    source: str                # e.g. "autonomy:task-42"
    summary: str               # one-liner for the <context> block
    content: str = ""          # optional fuller body
    dedup_key: str = ""        # collapses with earlier entries of same key
    expires_at: float = 0.0    # unix ts; 0 means no expiry
    enqueued_at: float = 0.0


_ambient_prefetch_queue: dict[str, list[AmbientPrefetchEntry]] = {}
AMBIENT_PREFETCH_CAP = 5       # max stored per session before oldest is dropped
AMBIENT_PREFETCH_DRAIN_MAX = 3 # max injected into a single turn's <context>


def enqueue_ambient_prefetch(session_id: str, entry: AmbientPrefetchEntry) -> dict[str, Any]:
    """Push an ambient prefetch entry for `session_id`.

    Behavior:
      - If `dedup_key` is set, collapses any existing entry sharing that
        key (newest wins — old is dropped so rapid producer re-fires don't
        stack).
      - Caps at `AMBIENT_PREFETCH_CAP`; oldest beyond cap is evicted.

    Returns a small dict describing what happened (for producer logging).
    """
    q = _ambient_prefetch_queue.setdefault(session_id, [])
    dropped: list[str] = []
    deduped = False
    if entry.dedup_key:
        keep: list[AmbientPrefetchEntry] = []
        for e in q:
            if e.dedup_key == entry.dedup_key:
                dropped.append(e.source)
                deduped = True
            else:
                keep.append(e)
        q[:] = keep
    q.append(entry)
    while len(q) > AMBIENT_PREFETCH_CAP:
        old = q.pop(0)
        dropped.append(old.source)
    return {
        "queued": 1,
        "queue_depth": len(q),
        "dropped": dropped,
        "deduped": deduped,
    }


def drain_ambient_prefetch(session_id: str) -> list[AmbientPrefetchEntry]:
    """Pop and return (up to `AMBIENT_PREFETCH_DRAIN_MAX`) unexpired entries
    for `session_id`. Expired entries are silently evicted. Safe to call
    when the queue is empty (returns []).
    """
    import time as _time
    q = _ambient_prefetch_queue.pop(session_id, [])
    if not q:
        return []
    now = _time.time()
    alive = [e for e in q if e.expires_at == 0.0 or e.expires_at > now]
    # Keep newest first for drain, then put any overflow back for next turn.
    alive.sort(key=lambda e: e.enqueued_at, reverse=True)
    drained = alive[:AMBIENT_PREFETCH_DRAIN_MAX]
    leftover = alive[AMBIENT_PREFETCH_DRAIN_MAX:]
    if leftover:
        _ambient_prefetch_queue[session_id] = leftover
    return drained


def peek_ambient_prefetch(session_id: str) -> list[AmbientPrefetchEntry]:
    """Snapshot without popping — for debug endpoints and tests."""
    return list(_ambient_prefetch_queue.get(session_id, []))


# ---------------------------------------------------------------------------
# Ambient-turn decision state (task #295, Slice 4)
#
# When an agent calls `ambient_decide(session_id, surface=False)` during an
# ambient turn, we record the decision here. `_run_turn`'s finally clause
# consults this dict and, if surface=False, suppresses normal assistant
# persistence and writes a muted breadcrumb instead.
# ---------------------------------------------------------------------------

_ambient_decisions: dict[str, dict[str, Any]] = {}


def set_ambient_decision(session_id: str, decision: dict[str, Any]) -> None:
    _ambient_decisions[session_id] = decision


def take_ambient_decision(session_id: str) -> Optional[dict[str, Any]]:
    """Pop and return the decision for a session, or None if unset."""
    return _ambient_decisions.pop(session_id, None)


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
                # Inner Voice (#345): A/B linkage tag. Sessions sharing an
                # experiment_id (typically Chat-tab + Inner-Voice-tab runs
                # of the same task) can be joined for meta-review.
                "experiment_id": None,
                # Inner Voice opt-in flag — Brain 2 ensemble fires only on
                # sessions where this is True. Stage 0 always False.
                "inner_voice": False,
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
