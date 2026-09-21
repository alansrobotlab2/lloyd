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
import contextvars
import json
import logging
import time as _time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from app.atomic_io import atomic_write_text
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
    # What the turn is doing right now, for the dashboard's agent panel:
    # {"kind", "label", "detail", "at"}. Display-only and lossy by design
    # — the loop never reads it back. See `set_turn_activity`.
    activity: Optional[dict[str, Any]] = None


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

# Sessions that exist so a machine can run a turn in them, not so a human can
# read them. Neither is ever "the user's session": an ambient producer that
# resolves to one delivers its notification to nobody. `autonomy` was excluded
# from the start. `worker` joined it on 2026-09-07, when the morning brief
# (MockBOT meeting that night) was injected into a backlog-triage session that
# had finished 77 seconds earlier — worker turns go through the chat path, so
# it was the last session to receive a user-source turn — and answered there.
# A deny-list on purpose: a client this list has never heard of must keep
# receiving its briefs rather than silently losing them.
NON_USER_PLATFORMS = frozenset({"autonomy", "worker"})


def is_user_session(data: dict) -> bool:
    """True if a human reads this session. Missing platform means the web UI."""
    return (data.get("platform") or "mission-control") not in NON_USER_PLATFORMS


#: A background session's id has four underscore-separated parts
#: (`20260910_120001_autocode_9f2a`); a chat session's has three
#: (`20260910_120001_9f2a1c`). That is a cheap discriminator, and it has to
#: exist because the alternative is parsing every session JSON in the
#: directory to answer "is this one a chat?" — at ~240 background sessions a
#: day against ~14 chats (measured over the week to 2026-09-10), that is the
#: difference between a bounded listing and one that grows with the fleet's
#: throughput.
#:
#: Where the file is opened, its `platform` is the authority. But the chat
#: listings do not open a four-part file at all, so for that shape the NAME
#: decides — and that is safe only while nothing that creates a user session
#: mints a four-part id. Three mints exist: the chat path (`<ts>_<6hex>`),
#: `POST /api/sessions/create` (`<ts>_iv<4hex>`), and `new_background_session_id`
#: below, the only four-part one. `tests/test_session_platform_checks.py`
#: pins all three, because a future producer that named a user session in four
#: parts would vanish from the history without a word.
def is_background_session_name(name: str) -> bool:
    """True if this session *filename* looks like a background run's.

    Conservative in one direction only: an id this rule does not recognise
    reads as a chat session, is parsed, and is then classified by its
    `platform` — one wasted read. The other direction is not conservative — a
    four-part user session would be hidden from the history unread — and is
    closed at the creators instead of here.
    """
    stem = str(name or "")
    if stem.endswith(".json"):
        stem = stem[:-5]
    parts = stem.split("_")
    if len(parts) < 4:
        return False
    return (len(parts[0]) == 8 and parts[0].isdigit()
            and len(parts[1]) == 6 and parts[1].isdigit())


def new_background_session_id(slug: str) -> str:
    """Mint a four-part background session id from a producer slug."""
    import uuid as _uuid
    clean = "".join(ch for ch in str(slug or "bg") if ch.isalnum())[:12] or "bg"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{clean}_{_uuid.uuid4().hex[:4]}"


#: Sessions created while a worker-pool job is claimed. The pool binds an
#: empty list around each job and reads it back when it writes the run row, so
#: every background run's record names the transcript(s) it produced —
#: whatever path made them, and without each source having to remember to
#: return the id. Two records of the same run with nothing joining them is
#: exactly what made a suspect autonomy run unreviewable on 2026-09-10.
#:
#: Default `None` means "nobody is collecting", which is what an interactive
#: turn and a bare script get; `create_session` then does nothing extra.
current_run_sessions: contextvars.ContextVar[Optional[list[str]]] = \
    contextvars.ContextVar("lloyd_run_sessions", default=None)


def note_run_session(session_id: str) -> None:
    """Attribute a session to the job currently claimed, if any."""
    bucket = current_run_sessions.get()
    if bucket is None or not session_id or session_id in bucket:
        return
    bucket.append(session_id)


def create_session(session_id: str, *, platform: str, model: str = "",
                   title: str = "", source: str = "",
                   inner_voice: bool = False, preview: str = "",
                   inner_voice_evaluate_user_turns: Optional[bool] = None,
                   experiment_id: Optional[str] = None,
                   exist_ok: bool = True,
                   sessions_dir: Optional[Path] = None) -> str:
    """Create a session file for a run that is about to start.

    One writer for every *create* — the background recorder,
    `workers.sources._common.new_worker_session`, and `POST
    /api/sessions/create` (the Inner Voice "+ new chat" pre-creator) all come
    through here. The chat path keeps `_save_session_meta`, which has to merge
    into an existing file on every turn; this one is a create, and a create is
    the only thing a background run and a pre-created stub both need. Until
    2026-09-20 the endpoint did not come through here: it built its own dict
    and wrote it with a bare `write_text`, so every session it had minted
    carried no `id` and no `source`, and it was the only create path that wrote
    non-atomically — a torn file there makes one live chat vanish from one
    dashboard poll. One field set, one atomic write, is the point of this
    helper.

    `sessions_dir` is for the one caller that owns the directory it reads back
    — the sessions router keeps its own module-level `SESSIONS_DIR` and the
    collision check, the write and the listing have to be about the same file.
    `exist_ok=False` raises `FileExistsError` instead of returning quietly,
    which is what lets that endpoint answer 409 rather than adopt a stranger's
    session.

    **The title is set here, at creation**, for two reasons that apply to
    different sessions. A direct-path run (autonomy, `run_prompt_on_primary`)
    never goes near the LLM titler — it is fired from the chat path only — so
    the title set here is the only one it will ever have. A session-backed
    worker does go through the chat path, ~70 a day in the week to
    2026-09-10, and `session_titles.should_title` now refuses it: the titler
    runs on the single-tenant secondary, where every chat turn already queues,
    and those would be ~70 model calls a day to relabel rows nobody asked
    about.

    Both `session_id` and `id` are written. `/api/sessions` reads
    `session_id` and falls back to the filename; the worker sessions that
    predate this helper wrote only `id`, and something may yet read it.
    `source` is the *producer* attribution — a session the web UI created has
    no producer, so it is written empty rather than invented: `_session_identity`
    in the message router feeds it to the grant gate, and a made-up value there
    is a policy change. `inner_voice_evaluate_user_turns` defaults to the master
    flag (`None`) but a caller that collected the two flags separately passes
    them separately — that is the difference between a critic that fires on
    chat turns and one that does not.
    Existing files are left alone — a multi-call job lands in one transcript.
    """
    note_run_session(session_id)
    directory = sessions_dir or SESSIONS_DIR
    path = directory / f"{session_id}.json"
    if path.exists():
        if not exist_ok:
            raise FileExistsError(f"session {session_id} already exists")
        return session_id
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    evaluate_user_turns = (bool(inner_voice) if inner_voice_evaluate_user_turns is None
                           else bool(inner_voice_evaluate_user_turns))
    data = {
        "session_id": session_id,
        "id": session_id,
        "title": (title or "").strip()[:80],
        "model": model,
        "platform": platform,
        "source": source,
        "created_at": now,
        "last_active": now,
        "preview": preview[:60],
        "message_count": 0,
        "messages": [],
        "experiment_id": experiment_id,
        "inner_voice": bool(inner_voice),
        "inner_voice_evaluate_user_turns": evaluate_user_turns,
    }
    atomic_write_text(path, json.dumps(data, indent=2))
    return session_id


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
      1. `_last_user_session_id` if set, the session JSON still exists, AND
         it is a user session. Worker turns arrive through the chat path and
         set this too, which is how a brief once went to a triage session.
      2. Most-recent user session by mtime within `max_age_hours`.

    Both rules apply `is_user_session`: an `autonomy` or `worker` session
    never receives an injection, its own or anyone else's.

    Returns None if nothing qualifies — producers should treat this as a
    no-op (the user simply has no active chat session to notify).
    """
    if _last_user_session_id:
        p = SESSIONS_DIR / f"{_last_user_session_id}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except (OSError, ValueError):
                data = None          # mid-write: keep the old answer
            if data is None or is_user_session(data):
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
            if not is_user_session(data):
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


def ambient_clock_stamp(epoch_seconds: float | None) -> str:
    """Render one measured instant as the wall clock the model is allowed to use.

    #1197. The autotriage brief of 2026-09-16 wrote `Brief + Triage — 2026-09-17`
    into its own header, asked the calendar for that invented day, got an empty
    list back legitimately, and reported Ben's Birthday as `(no events)` — 58 of
    147 dated headers named a day other than the run's own. The run had no clock
    in its context, so it composed one in prose, and nothing strips a composed
    date. Both ambient delivery paths now carry a measured one: this is the only
    formatter either path uses — the `<ambient-signals>` drain in
    `prefetch._format_context` and the `<ambient …>` envelope in
    `app.routers.messages.build_ambient_turn`.

    Box-local zone with its abbreviation, because that is the zone the user reads
    a calendar in; `astimezone()` with no argument is the system zone, so this
    tracks DST (`PST` in January, `PDT` in September) instead of a hard-coded
    offset.

    An unset instant — `0.0`, the dataclass default, meaning no producer ever
    stamped it — returns "". No measurement, no stamp: rendering it would hand
    the model a 1969 date as if it were evidence.
    """
    if not epoch_seconds:
        return ""
    try:
        when = datetime.fromtimestamp(float(epoch_seconds)).astimezone()
    except (TypeError, ValueError, OSError):
        return ""
    return when.strftime("%Y-%m-%d %H:%M %Z")


def _is_expired(entry: AmbientPrefetchEntry, now: float) -> bool:
    """Has this entry's deadline passed? `expires_at == 0.0` means none was set.

    The sentinel is checked first and is never expired. It is the dataclass
    default, so every producer that never thought about a TTL — and every test
    that enqueues with the field unset — arrives as 0.0. Reading it as an instant
    (epoch zero, permanently past) would make any purge silently delete signals
    that were never given a deadline, which is a worse failure than the one #910
    fixes: the old bug kept a signal too long, this one would drop it forever.
    """
    return entry.expires_at != 0.0 and entry.expires_at <= now


def _release_if_empty(session_id: str) -> None:
    """Drop `session_id`'s key once its list is empty; keep it otherwise.

    The dict lives for the process lifetime, so an empty list parked under a key
    is a permanent entry per session id that ever received a signal — the
    unbounded dimension #910 names, since `AMBIENT_PREFETCH_CAP` bounds entries
    but not keys. Callers use this after any step that can empty a list; the
    invariant is "a key is present iff it holds entries", so a drain that
    overflows `AMBIENT_PREFETCH_DRAIN_MAX` must NOT release its key, and neither
    must cap eviction, which drops the oldest and still leaves the cap.
    """
    if not _ambient_prefetch_queue.get(session_id):
        _ambient_prefetch_queue.pop(session_id, None)


def clear_ambient_prefetch(session_id: str) -> int:
    """Discard every ambient prefetch entry queued for `session_id`.

    For a caller that is destroying the session outright: the entries are
    drained only by a turn for this exact id, so once the session JSON is
    unlinked nothing in the process will ever read them. `DELETE
    /api/sessions/{id}` calls this — it already drained both tiers of the *turn*
    queue for the same reason (#909), and the prefetch dict was the one store it
    left behind.

    Returns how many entries were discarded.
    """
    dropped = _ambient_prefetch_queue.pop(session_id, [])
    return len(dropped)


def enqueue_ambient_prefetch(session_id: str, entry: AmbientPrefetchEntry) -> dict[str, Any]:
    """Push an ambient prefetch entry for `session_id`.

    Behavior:
      - Expiry is applied here as well as at the drain (#910). Expiry used to
        live only inside `drain_ambient_prefetch`, whose sole caller is
        `prefetch._prefetch_prepare` — so a signal's TTL could not fire until a
        turn ran **for its own session id**, and a producer that named a session
        which never takes one left the entry and its key in RAM for the life of
        the process. Purging on write makes every producer that touches a session
        reclaim that session's dead entries, and its `dropped` names them.
      - A signal already past its deadline is not stored at all. `queued` is 0
        then, because `session_inject_context` echoes this body as what was
        delivered and must not report a signal that can never reach anyone.
      - If `dedup_key` is set, collapses any existing entry sharing that
        key (newest wins — old is dropped so rapid producer re-fires don't
        stack). A signal that will not be stored does not spend a live entry.
      - Caps at `AMBIENT_PREFETCH_CAP`; oldest beyond cap is evicted.
      - The session's key is released whenever the list ends up empty.

    Returns a small dict describing what happened (for producer logging).
    """
    now = _time.time()
    q = _ambient_prefetch_queue.setdefault(session_id, [])
    dropped: list[str] = []

    expired = [e for e in q if _is_expired(e, now)]
    if expired:
        dropped.extend(e.source for e in expired)
        q[:] = [e for e in q if not _is_expired(e, now)]

    if _is_expired(entry, now):
        # Refused at the door: it could only ever be drained as expired. The
        # depth reported is what the session still holds — a refusal is not an
        # empty queue, and a producer reading `queue_depth: 0` would conclude its
        # earlier signals had already been delivered.
        dropped.append(entry.source)
        _release_if_empty(session_id)
        return {
            "queued": 0,
            "queue_depth": len(_ambient_prefetch_queue.get(session_id, [])),
            "dropped": dropped,
            "deduped": False,
        }

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
    # No release check on this path: the append above put one entry in, so the
    # list cannot be empty and cap eviction cannot empty it (`AMBIENT_PREFETCH_CAP
    # = 5` — eviction only runs above 5 and stops at 5). The one write that can
    # end with nothing stored is the refusal above.
    return {
        "queued": 1,
        "queue_depth": len(q),
        "dropped": dropped,
        "deduped": deduped,
    }


def drain_ambient_prefetch(session_id: str) -> list[AmbientPrefetchEntry]:
    """Pop and return (up to `AMBIENT_PREFETCH_DRAIN_MAX`) unexpired entries
    for `session_id`. Safe to call when the queue is empty (returns []).

    An entry past its deadline is dropped and **named in the log**, the way the
    write path names its casualties in `dropped`. This filter is the backstop, not
    the primary — the write path refuses to store an expired signal at all — but a
    signal can still pass its deadline here, because it may wait in the queue
    between one write and the next drain. A silently-falling-signal path is what
    kept #910 invisible for as long as it did, so this drop is not silent either.

    Takes the key with the list, then puts back only what actually overflows —
    so an exhausted queue leaves no key behind (#910) while a drain that hit the
    per-turn cap keeps exactly the leftover entries under it.
    """
    q = _ambient_prefetch_queue.pop(session_id, [])
    if not q:
        return []
    now = _time.time()
    alive = [e for e in q if not _is_expired(e, now)]
    dead = [e.source for e in q if _is_expired(e, now)]
    if dead:
        logger.info(
            "ambient prefetch: dropped %d expired signal(s) for session %s: %s",
            len(dead), session_id, ", ".join(dead),
        )
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


def active_turn_summary() -> dict:
    """Process-wide turn liveness: {"active": N, "queued": M, "sessions": [...]}.

    Nothing else answers "is any turn in flight anywhere". `is_session_active`
    needs a session id; `/api/sessions/active` returns a heuristic *current*
    session for ambient producers rather than a liveness answer; and
    `/api/sessions/active-procs` only finds legacy SDK subprocesses, which the
    in-process harness no longer spawns.

    This is the idle gate for self-modification: the promoter must not restart
    the backend out from under a running turn. Cheap by construction — a dict
    walk over `_session_queues`, no filesystem and no locking (a torn read
    just means the promoter waits one more poll).
    """
    active = 0
    queued = 0
    busy_sessions: list[str] = []
    for sid, q in list(_session_queues.items()):
        running = q.current is not None
        pending = len(q.pending_user) + len(q.pending_ambient)
        if running:
            active += 1
        queued += pending
        if running or pending:
            busy_sessions.append(sid)

    # Session queues are not the whole story. Worker jobs, the IDE routes and
    # post-session capture all call `run_query` directly and never appear in
    # `_session_queues`, so an idle gate built on the queues alone reads a box
    # running a ten-minute research job as perfectly quiet — and restarts the
    # backend out from under it. `harness_runs` is the process-wide count of
    # agent loops actually in flight, whoever started them.
    try:
        from app.harness.loop import active_run_count
        harness_runs = active_run_count()
    except Exception:
        harness_runs = 0

    return {"active": active, "queued": queued, "harness_runs": harness_runs,
            "sessions": sorted(busy_sessions)}


def get_cancel_event(session_id: str) -> Optional[asyncio.Event]:
    """Cancel-event for the currently-running turn, or None if idle."""
    q = _session_queues.get(session_id)
    if q is None or q.current is None:
        return None
    return q.cancel_event


# Argument worth showing for a tool call, in preference order. A tool not
# listed falls back to its first short string argument, which names the
# thing being worked on far more often than it doesn't.
_ACTIVITY_ARG_KEYS = (
    "command", "file_path", "pattern", "path", "query",
    "description", "prompt", "url", "content",
)


def tool_activity_detail(args_json: str, limit: int = 80) -> str:
    """One-line summary of a tool call's arguments, for display.

    `Bash` alone says nothing; `Bash · supervisorctl restart` says what the
    agent is doing. Never raises — a tool whose arguments failed to parse
    still deserves to have its name shown.
    """
    try:
        args = json.loads(args_json or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    if not isinstance(args, dict):
        return ""

    def _flatten(value: str) -> str:
        one_line = " ".join(value.split())
        return one_line[: limit - 1] + "…" if len(one_line) > limit else one_line

    for key in _ACTIVITY_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _flatten(value)
    for value in args.values():
        if isinstance(value, str) and value.strip():
            return _flatten(value)
    return ""


def set_turn_activity(
    session_id: str, kind: str, label: str = "", detail: str = ""
) -> None:
    """Record what the running turn is doing right now.

    Feeds the dashboard's agent panel, which otherwise can only say that a
    session is busy — true of a turn that is thinking, one that has been
    running `Bash` for four minutes, and one that is wedged, which are not
    the same situation to an operator.

    Best-effort: a no-op when nothing is running, and unchanged states are
    dropped so the streaming path can call this per token without churning
    a timestamp.
    """
    q = _session_queues.get(session_id)
    cur = q.current if q else None
    if cur is None:
        return
    prev = cur.activity
    if (
        prev is not None
        and prev.get("kind") == kind
        and prev.get("label") == label
        and prev.get("detail") == detail
    ):
        return
    cur.activity = {
        "kind": kind,
        "label": label,
        "detail": detail,
        "at": datetime.now().isoformat(),
    }


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


def active_sessions_snapshot() -> list[dict[str, Any]]:
    """Every session with a running or queued turn, for the dashboard.

    Exposed as a function rather than letting callers walk
    `_session_queues` themselves: the dict is mutated by the turn
    consumer under `q.lock`, and a reader iterating it directly races
    that. Building the list in one pass here keeps the private state
    private and the snapshot self-consistent.
    """
    out: list[dict[str, Any]] = []
    for session_id, q in list(_session_queues.items()):
        cur = q.current
        if cur is None and not q.pending_user and not q.pending_ambient:
            continue
        out.append({
            "session_id": session_id,
            "running": cur is not None,
            "turn_id": cur.turn_id if cur else None,
            "source": cur.source if cur else None,
            "started_at": cur.started_at.isoformat() if cur and cur.started_at else None,
            "enqueued_at": cur.enqueued_at.isoformat() if cur else None,
            "preempted": bool(cur.preempted) if cur else False,
            "activity": dict(cur.activity) if cur and cur.activity else None,
            "pending_user": len(q.pending_user),
            "pending_ambient": len(q.pending_ambient),
        })
    # Running turns first, then longest-queued — the order an operator
    # scanning the panel wants.
    out.sort(key=lambda e: (not e["running"], e["started_at"] or ""))
    return out


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


#: What `drain_pending` may be asked to clear. `"all"` is the one value here
#: that is not a `TurnSource`: it names both tiers at once, for a caller that
#: is destroying the session and must leave nothing queued (#909).
DrainSource = Literal["user", "ambient", "system", "all"]


async def drain_pending(session_id: str, source: Optional[DrainSource] = None) -> int:
    """Remove queued turns. If source is None, drains ambient only
    (the documented behavior for /cancel?drain_pending=true — user turns
    are never silently dropped). Returns number drained.

    Pass `source="all"` for both tiers in one call: it drains the ambient
    queue then the user queue and returns the summed count. The pop is
    permanent — a drained turn never runs — so only a caller that discards
    the session outright may ask for it. `DELETE /api/sessions/{id}` does: a
    queued user turn it left behind would run to completion for a transcript
    whose file is already unlinked, and every write it made would no-op in
    `mutate_session`.
    """
    q = _session_queues.get(session_id)
    if q is None:
        return 0
    drained = 0
    async with q.lock:
        if source is None or source == "ambient" or source == "all":
            while q.pending_ambient:
                t = q.pending_ambient.popleft()
                t.preempted = True
                try:
                    t.events.put_nowait(None)
                except Exception:
                    pass
                t.done.set()
                drained += 1
        if source == "user" or source == "all":
            while q.pending_user:
                t = q.pending_user.popleft()
                # Same drop signature as the ambient tier. `preempted` is what
                # the dashboard snapshot reports and what the ambient-cancel
                # breadcrumb reads, so a dropped turn has to carry it or the
                # panel shows a queue that is empty and nothing saying why.
                t.preempted = True
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
        atomic_write_text(meta_path, json.dumps(data, indent=2))
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
                # Inner Voice opt-in flag — the critic ensemble fires only on
                # sessions where this is True. Stage 0 always False.
                "inner_voice": False,
            }
        atomic_write_text(meta_path, json.dumps(data, indent=2))


async def _append_messages(session_id: str, new_messages: list[dict]):
    """Append messages to session metadata file."""
    def _append(data):
        msgs = data.get("messages", [])
        msgs.extend(new_messages)
        data["messages"] = msgs
        data["last_active"] = datetime.now().isoformat()
    await mutate_session(session_id, _append)
