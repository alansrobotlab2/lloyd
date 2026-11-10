"""A voice room's view of its session: every turn's events, and what was heard.

Two small pieces of per-session state the voice worker needs from the backend,
kept here so neither the chat router nor the voice router owns the other's.

**The tap.** A turn's `events` queue has exactly one reader — whoever enqueued
it. A spoken turn is read by the worker (`/api/voice/inject?stream`), but a
typed turn in a session with an open voice room is read by the browser, so
until this the worker could only find a typed answer by polling the transcript
every 500 ms, waiting for it to finish, and having the primary rewrite it as a
summary (#1445) — 13 s of speech per answer, with "session poll failed"
warnings. `_emit` now also hands each event to any tap open on the turn's
session, and `GET /api/voice/listen` streams a tap to the worker, which speaks
typed turns through the same clause path as spoken ones.

A tap is lossy by design: its queue is bounded and a full one drops, because a
worker that stopped reading must never back up a turn. The turn's own queue,
which persistence and the browser depend on, is untouched.

**What was heard.** When the listener talks over Lloyd, the transcript still
holds the whole reply, so the next turn's model believes it said all of it.
LiveKit Agents truncates its chat context at the playout position; the Lloyd
equivalent is a note the next turn carries in its prompt tail (so the cached
prefix is untouched), which the chat then records as that turn's subliminal
context like every other injection.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger("lloyd-server")

#: Events a tap carries. Everything the worker needs to speak a turn and to
#: know one started; thinking deltas and queue chatter are not worth the copy.
TAP_EVENTS = frozenset({"session", "text_delta", "tool_start", "tool_complete",
                        "done", "error"})
TAP_QUEUE_MAX = 4000

_taps: dict[str, set[asyncio.Queue]] = {}

#: Turns the worker injected and reads from their own stream. Their events never
#: reach a tap: registering the id there would race the tap's copy of the same
#: `session` event, and the loser would speak the reply twice.
_voice_turn_ids: "deque[str]" = deque(maxlen=256)


def mark_voice_turn(turn_id: str) -> None:
    """Called before a spoken turn is enqueued, so no tap ever sees it."""
    if turn_id:
        _voice_turn_ids.append(turn_id)


def is_voice_turn(turn_id: str) -> bool:
    """Whether `turn_id` is a turn the voice worker injected (and reads)."""
    return bool(turn_id) and turn_id in _voice_turn_ids


#: Spoken turns whose first model request has already been reported.
_stats_sent: "deque[str]" = deque(maxlen=256)


def voice_turn_stats(turn: Any, evt: dict, usage: dict) -> dict:
    """The first model request of a spoken turn, as the voice worker records it.

    `prompt_tokens`/`cached_tokens` say whether the prewarm's prefix was hit,
    `ttft_ms` is the engine's own first-token time for that request, and
    `queue_wait_s` is how long the turn waited in the session queue before it
    ran — the three hypotheses for a slow first word (prefix miss, engine
    admission, a turn in front) read off one row.
    """
    def _int(value):
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    out = {
        "prompt_tokens": _int(usage.get("input_tokens") or usage.get("prompt_tokens")),
        "cached_tokens": _int(usage.get("cache_read") or usage.get("prompt_tokens_cached")
                              or 0),
        "ttft_ms": _int(evt.get("ttft_ms")),
        "request_ms": _int(evt.get("request_ms")),
    }
    started, enqueued = getattr(turn, "started_at", None), getattr(turn, "enqueued_at", None)
    if started is not None and enqueued is not None:
        out["queue_wait_s"] = round(max(0.0, (started - enqueued).total_seconds()), 3)
    return out


async def emit_first_turn_stats(turn: Any, evt: dict, usage: dict, emit) -> None:
    """Send one `turn_stats` frame on a spoken turn's first model request.

    Called by the message path (app/routers/messages.py `_run_turn`) after every
    completed model request; a typed turn, or any request after a spoken turn's
    first, is a no-op. The voice worker (agent-services/livekit_worker.py) keeps
    the frame on the turn's latency row.
    """
    turn_id = getattr(turn, "turn_id", "")
    if not is_voice_turn(turn_id) or turn_id in _stats_sent:
        return
    _stats_sent.append(turn_id)
    await emit(turn, "turn_stats", voice_turn_stats(turn, evt, usage))


def open_tap(session_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=TAP_QUEUE_MAX)
    _taps.setdefault(session_id, set()).add(q)
    return q


def close_tap(session_id: str, q: asyncio.Queue) -> None:
    taps = _taps.get(session_id)
    if not taps:
        return
    taps.discard(q)
    if not taps:
        _taps.pop(session_id, None)


def has_listener(session_id: str) -> bool:
    """True while a voice worker is streaming this session's turns."""
    return bool(_taps.get(session_id))


def publish(session_id: str, event: str, data: dict) -> None:
    """Hand one turn event to every tap on `session_id`. Never raises."""
    if not session_id or event not in TAP_EVENTS:
        return
    taps = _taps.get(session_id)
    if not taps:
        return
    if data.get("turn_id") in _voice_turn_ids:
        return
    for q in list(taps):
        try:
            q.put_nowait({"event": event, "data": dict(data)})
        except asyncio.QueueFull:
            logger.warning("voice tap for %s is full — dropping %s", session_id, event)
        except Exception:
            pass


# ── What was heard ─────────────────────────────────────────────────────────

#: A note older than this is about a conversation that has moved on.
HEARD_NOTE_TTL_S = 600.0

_heard_notes: dict[str, tuple[float, str]] = {}


def record_interruption(session_id: str, heard: str, unheard_chars: int = 0) -> str:
    """Remember that the last spoken reply was cut off after `heard`."""
    heard = " ".join((heard or "").split())
    if len(heard) > 600:
        heard = "…" + heard[-600:]
    # The note says what was heard AND what to do about it. Without the second
    # half the model finished the interrupted reply: on 2026-10-12 a
    # three-word acknowledgement became a turn, and the model read the whole
    # corrected answer (629 characters, 10 s) to a listener who had moved on.
    tail = (
        " They interrupted to say something new: answer what they just said. "
        "Do not resume, repeat or finish the interrupted reply unless they ask "
        "for it; if what they said needs no answer, one short sentence is "
        "enough.</system-reminder>\n\n"
    )
    if heard:
        note = (
            "<system-reminder>The listener interrupted your previous spoken "
            f"reply. They heard only this much of it: \"{heard}\" — the rest "
            "was never said aloud, so do not assume they know it." + tail
        )
    else:
        note = (
            "<system-reminder>The listener interrupted your previous spoken "
            "reply before any of it was heard." + tail
        )
    _heard_notes[session_id] = (time.monotonic(), note)
    return note


def take_heard_note(session_id: str) -> Optional[str]:
    """Pop the interruption note for the next turn, if one is fresh."""
    item = _heard_notes.pop(session_id, None)
    if item is None:
        return None
    at, note = item
    if time.monotonic() - at > HEARD_NOTE_TTL_S:
        return None
    return note


def reset_for_tests() -> dict[str, Any]:
    _taps.clear()
    _heard_notes.clear()
    _voice_turn_ids.clear()
    return {}
