"""Session metadata persistence and per-session turn queue.

Session data lives as one JSON file per session under `SESSIONS_DIR`.

Concurrency model (task #296, Phase 1):
    Each session_id has a `SessionQueue` holding an `asyncio.Lock`, a FIFO
    `pending` deque of `SessionTurn` objects, a reference to the currently
    running turn, and a `cancel_event` that targets that current turn.

    A lazy per-session consumer task drains `pending` serially — this
    replaces the old `_active_streams: dict[str, Event]` check-then-act
    model that raced on near-simultaneous POSTs.

    The dataclass has headroom for Phase 3 (ambient/system turns, dedup,
    per-turn broker subscribers). Phase 1 only wires `source="user"`.
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
    Unbounded is fine while the only producer is a single user turn;
    ambient producers in Phase 3 need a bounded variant.
    """
    turn_id: str
    source: TurnSource
    payload: dict[str, Any]
    enqueued_at: datetime
    started_at: Optional[datetime] = None
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class SessionQueue:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: deque = field(default_factory=deque)
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
    return q.current is not None or bool(q.pending)


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


def _save_session_meta(session_id: str, model: str, preview: str = ""):
    """Save session metadata to JSON file."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    now = datetime.now().isoformat()
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


def _append_messages(session_id: str, new_messages: list[dict]):
    """Append messages to session metadata file."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return
    data = json.loads(meta_path.read_text())
    msgs = data.get("messages", [])
    msgs.extend(new_messages)
    data["messages"] = msgs
    data["last_active"] = datetime.now().isoformat()
    meta_path.write_text(json.dumps(data, indent=2))
