"""Short, human-readable titles for chat sessions.

A session id is a timestamp. That is a fine primary key and a useless
label: finding "the conversation about cloning the TTS voice" in the chat
history means opening sessions one at a time until one looks right. Every
surface in Mission Control that names a session — the chat history list,
the dashboard's agent panel, the Inner Voice picker, the chat header —
renders this title instead, falling back to the id when there isn't one.

Titles are written by the **secondary** model, never the primary. Titling
is background work and must not sit in front of the user's next token;
this is the slot post-session capture, fact extraction, and voice
summaries already use. That slot is single-tenant (llama.cpp
``--parallel 1``), so title calls are kept small — a ~30-token completion
over a truncated head of the transcript — and rare: `_should_title`
re-titles on a geometric schedule (after the 1st user message, then the
3rd, 9th, 27th…), because a conversation's subject drifts but not on
every turn, and a title call that fires per turn would sit in the same
queue as the agent's own work.

The title is advisory display text. Nothing reads it back, so every
failure path here is "leave the session untitled and move on".
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

from app.paths import SESSIONS_DIR
from app.sessions_io import is_user_session, mutate_session

logger = logging.getLogger("lloyd-server")


# A title is a label, not a sentence. These bounds keep it renderable in a
# 256px sidebar row without an ellipsis eating the only distinguishing
# word. The prompt asks for 3-6 words; these are the safety net, so the
# char bound is the one that normally bites — a word cap set at the
# prompt's own target lops the last word off titles that were fine
# ("Debugging Task 75 AI Engineer Monitor Failure" is 45 characters).
_MAX_WORDS = 8
_MAX_CHARS = 48

# Re-title once the user-message count reaches this multiple of the count
# the current title was written from. Geometric, so a long session pays
# for ~5 title calls rather than one per turn.
_RETITLE_GROWTH = 3

# Below this the transcript is a greeting, not a conversation.
_MIN_TRANSCRIPT_CHARS = 20

# Head of the transcript handed to the model. The subject of a session is
# established early; feeding the tail would title a 50-turn debugging
# session after whatever it happened to be doing last.
_TRANSCRIPT_MAX_CHARS = 2000
_TRANSCRIPT_MAX_MESSAGES = 8

# Injected blocks are machinery, not conversation. Same prefix list
# post_capture filters on — a session whose first user message is a
# `<context>` dump would otherwise be titled after the prefetcher.
_SYNTHETIC_PREFIXES = (
    "<daily_notes>", "<memory>", "<context>", "<system-reminder>",
    "[cron:", "[System Message]", "[autonomy:",
)

# Models like to answer the question rather than do the task. Strip the
# common wrappers before deciding whether what's left is a title.
_LABEL_RE = re.compile(r"^(?:the\s+)?(?:session\s+)?title\s*[:\-—]\s*", re.I)
_REFUSALS = {"none", "trivial", "n/a", "na", "untitled", "unknown", "title"}


# ── Cleaning ───────────────────────────────────────────────────────────


def clean_title(raw: str) -> str:
    """Reduce a model's reply to a usable title, or "" if there isn't one.

    Deliberately strict. A bad title is worse than no title: the id at
    least tells you which session you are looking at, whereas
    ``Here is a title for the conversation`` tells you nothing and looks
    like a bug.
    """
    if not raw:
        return ""

    text = raw.strip()
    # Fenced or inline code — take the contents, not the fence.
    text = text.strip("`").strip()
    # First non-empty line only; anything after it is commentary.
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not line:
        return ""

    line = _LABEL_RE.sub("", line)
    line = line.strip().strip("\"'“”‘’*").strip()
    # Markdown heading / list markers the model may have volunteered.
    line = re.sub(r"^[#\-*•]+\s*", "", line).strip()
    line = re.sub(r"\s+", " ", line)
    line = line.rstrip(".,;:!—-").strip()

    if not line or line.lower() in _REFUSALS:
        return ""

    words = line.split(" ")
    if len(words) > _MAX_WORDS:
        line = " ".join(words[:_MAX_WORDS])
    if len(line) > _MAX_CHARS:
        line = line[:_MAX_CHARS].rsplit(" ", 1)[0].strip() or line[:_MAX_CHARS].strip()
    line = line.rstrip(".,;:!—-").strip()

    if len(line) < 3:
        return ""

    # Sentence-case the opener, but never touch a first word that already
    # carries internal capitals — "vLLM restart loop" must not become
    # "VLLM restart loop".
    first = line.split(" ", 1)[0]
    if line[0].islower() and not any(c.isupper() for c in first[1:]):
        line = line[0].upper() + line[1:]
    return line


# ── Policy ─────────────────────────────────────────────────────────────


def _message_text(msg: dict) -> str:
    """Flatten a session message's content blocks to plain text."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        )
    return ""


def _is_real_user_message(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    text = _message_text(msg).strip()
    if len(text) < 2:
        return False
    return not any(text.startswith(p) for p in _SYNTHETIC_PREFIXES)


def user_message_count(data: dict) -> int:
    """User messages the human actually typed, excluding injected blocks."""
    return sum(1 for m in data.get("messages", []) if _is_real_user_message(m))


def should_title(data: dict) -> bool:
    """Whether this session is due for a (re)title.

    Untitled sessions with at least one real user message always qualify.
    Titled ones qualify again only once the conversation has grown by
    `_RETITLE_GROWTH`×, which is what keeps this off the per-turn path.
    """
    # Background sessions are titled at creation (`sessions_io.create_session`),
    # so they never need this — which matters more than it sounds. The titler
    # runs on the single-tenant secondary, where agent turns already queue, and
    # ~180 background runs a day would put 180 model calls in front of them for
    # labels nobody asked to have refreshed.
    if not is_user_session(data):
        return False
    count = user_message_count(data)
    if count < 1:
        return False
    if not (data.get("title") or "").strip():
        return True
    titled_at = data.get("title_at_count") or 0
    if titled_at < 1:
        # Titled by an older build that didn't record the count. Anchor it
        # to now rather than re-titling on every turn forever.
        return False
    return count >= titled_at * _RETITLE_GROWTH


def build_transcript(data: dict) -> str:
    """Head of the conversation, as `USER:` / `LLOYD:` lines."""
    lines: list[str] = []
    for msg in data.get("messages", []):
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        if role == "user" and not _is_real_user_message(msg):
            continue
        text = _message_text(msg).strip()
        if not text:
            continue
        label = "USER" if role == "user" else "LLOYD"
        lines.append(f"{label}: {text[:400]}")
        if len(lines) >= _TRANSCRIPT_MAX_MESSAGES:
            break
    return "\n".join(lines)[:_TRANSCRIPT_MAX_CHARS]


def _enabled() -> bool:
    from app.config import CONFIG

    return bool((CONFIG.get("session_titles") or {}).get("enabled", True))


# ── Generation ─────────────────────────────────────────────────────────


# Titling runs as a fire-and-forget task off turn completion. Two turns
# finishing close together would otherwise queue two calls for the same
# session on a single-tenant engine and race each other's write.
_in_flight: set[str] = set()


async def maybe_title_session(session_id: str) -> Optional[str]:
    """Title `session_id` if it's due. Returns the new title, or None.

    Safe to fire-and-forget after every turn: the cheap checks (disk read,
    message count) run first and the model is only called when the session
    is actually due.
    """
    if not _enabled() or session_id in _in_flight:
        return None

    path = SESSIONS_DIR / f"{session_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not should_title(data):
        return None

    transcript = build_transcript(data)
    if len(transcript.strip()) < _MIN_TRANSCRIPT_CHARS:
        return None
    count = user_message_count(data)

    _in_flight.add(session_id)
    try:
        from app.secondary_models import _sync_secondary_title

        raw = await asyncio.to_thread(_sync_secondary_title, transcript)
        title = clean_title(raw or "")
        if not title:
            logger.info(
                "session title: %s — secondary returned nothing usable (%r)",
                session_id, (raw or "")[:80],
            )
            return None

        def _apply(d: dict) -> None:
            d["title"] = title
            d["title_at_count"] = count
            d["title_generated_at"] = datetime.now().isoformat()

        if not await mutate_session(session_id, _apply):
            return None
        invalidate(session_id)
        logger.info(
            "session title: %s -> %r (after %d user messages)",
            session_id, title, count,
        )
        return title
    except Exception as e:
        logger.warning("session title failed for %s: %s", session_id, e)
        return None
    finally:
        _in_flight.discard(session_id)


# ── Reads ──────────────────────────────────────────────────────────────


# The dashboard asks for the titles of every running session every 2s, and
# a session JSON is the whole transcript — megabytes on a long turn. Cache
# on a TTL rather than on mtime: mtime changes on every appended message,
# so an mtime cache would re-parse that file on every single poll. A title
# is stale for at most `_TITLE_TTL_S` seconds, and `invalidate` closes
# that window the moment we write a new one.
_TITLE_TTL_S = 30.0
_cache: dict[str, tuple[float, str]] = {}


def invalidate(session_id: str) -> None:
    _cache.pop(session_id, None)


def title_for(session_id: str) -> str:
    """Cached title for one session. "" when untitled or unreadable."""
    hit = _cache.get(session_id)
    now = time.monotonic()
    if hit is not None and now - hit[0] < _TITLE_TTL_S:
        return hit[1]

    path = SESSIONS_DIR / f"{session_id}.json"
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Keep serving the last known title on a transient read failure —
        # a torn read mid-write should not blank the panel.
        return hit[1] if hit else ""

    title = (data.get("title") or "").strip()
    _cache[session_id] = (now, title)
    return title


def titles_for(session_ids: list[str]) -> dict[str, str]:
    """Titles for several sessions. Blocking — call off the event loop."""
    return {sid: title_for(sid) for sid in session_ids}
