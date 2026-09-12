"""Subagent registry — live state for in-flight `Task` runs.

A `Task` call runs a nested `run_query` inside the lloyd-mcp process and,
until this module existed, was completely opaque: the tool blocks for as
long as the subagent works (minutes, sometimes) and the only evidence
anything is happening was the parent turn sitting on an unreturned tool
call. Session 20260905_151355_iv5174 burned 231s on a subagent that had
already gone wrong, with nothing to watch.

This registry is what the Mission Control dashboard reads. It records
one row per Task invocation and mutates it in place as the run
progresses — turns taken, tools dispatched, how long it has been going —
so a stuck subagent is visible while it is stuck rather than after it
returns.

The registry is also the *control* surface, not only a view (#411). A row
publishes the live conversation list its loop is reading and the event that
stops it, so `steer()` and `cancel_run()` can reach a child that is running
rather than only reporting on one. Both refuse a caller that is not the
orchestrating session, and a refusal adds nothing.

Lifetime is process-scoped, matching `_task_registry` (background bash
tasks) next door. Completed runs stay in a bounded ring so the dashboard
can show what *just* happened, not only what is happening; the ring is
capped because this is a live view, not an audit log — `event_logs/` is
the durable record.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("lloyd-subagent-registry")

# How many finished runs to keep for the "recent" panel. Small on
# purpose: the dashboard shows a handful and the event log holds history.
_RECENT_LIMIT = 20

_active: dict[str, "SubagentRecord"] = {}
_recent: deque["SubagentRecord"] = deque(maxlen=_RECENT_LIMIT)

_counter = 0
_task_counter = 0

# ---------------------------------------------------------------------------
# Steering framing — fixed here, in one place, on purpose (#411 clause 5)
# ---------------------------------------------------------------------------
#
# An injected message is stored in the child's own conversation and is
# replayed to it on every later iteration *and* on a resume, so its shape is
# load-bearing in two directions:
#
#   * the role cannot be `assistant`, or a rebuilt history reads the
#     orchestrator's instruction as the child's own turn — it would be
#     agreeing with itself, and the harness's own assistant-only repair pass
#     would treat it as authored content;
#   * it cannot be `system`, because vLLM rejects a system turn arriving
#     mid-stream — which is exactly why the Inner Voice observer uses `user`
#     for its own injections (`app/inner_voice/observer.py`).
#
# The prefix is what makes the injected line self-identifying to the model
# and greppable in a spilled transcript. Both are asserted in
# `tests/test_task_steering.py`; nothing else may build an injected message.
STEER_ROLE = "user"
STEER_PREFIX = "[ORCHESTRATOR] "


class SteeringRefused(Exception):
    """A steer/cancel that will not happen, with the reason for the caller."""


@dataclass(frozen=True)
class CallerScope:
    """Who is acting. `session_id` is the identity authority is checked on.

    Built from the contextvars `agent_mcp.main.call_tool` binds from the
    request `_meta` — see `builtin_task.current_caller_scope()`. An empty
    session id means "nobody in particular" and is never authorised.
    """

    session_id: str
    turn_id: str = ""


def steering_message(text: str) -> dict[str, str]:
    """The exact message an orchestrator append puts into the child's list."""
    return {"role": STEER_ROLE, "content": STEER_PREFIX + text.strip()}


@dataclass
class SubagentRecord:
    run_id: str
    # Stable across continuations, unlike `run_id` — one dashboard row per
    # RUN, one task_id per line of work. A caller resumes by task_id.
    task_id: str
    subagent_type: str
    description: str
    prompt_preview: str
    parent_session_id: str
    # The parent's turn id, when the spawning call carried one. A subagent's
    # own drain queue is never read after Task returns, so anything that
    # arrives late — a tsc result, a change-ledger entry — has to reach the
    # parent instead.
    parent_turn_id: str
    session_id: str
    model: str
    max_turns: int
    started_at: float
    finished_at: float | None = None
    status: str = "running"  # running | completed | failed | error
    turns: int = 0
    stop_reason: str = ""
    error: str = ""
    response_chars: int = 0
    tool_calls: list[str] = field(default_factory=list)
    # The run_id this one continues, when it is a resume.
    continuation_of: str = ""
    # The LIVE conversation list the child's loop is reading — the same
    # object `RunOptions.chat_messages_handle` holds, not a copy — and the
    # event that stops it. Published so a control surface can reach a running
    # child at all: before #411 the only reference to that list left the
    # process through `store_history()` inside `_close`, i.e. after the run
    # was already over, which is observability, not control.
    #
    # Never serialised. See `to_dict`.
    chat_messages: list[dict[str, Any]] | None = None
    cancel_event: asyncio.Event | None = None
    # How many orchestrator appends landed. Observable on the dashboard row.
    injected: int = 0

    @property
    def elapsed_s(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def note_tool(self, name: str) -> None:
        self.tool_calls.append(name)

    def note_turn(self) -> None:
        self.turns += 1

    def to_dict(self) -> dict[str, Any]:
        # `tool_counts` rather than the raw list: a subagent that ran 40
        # Greps is far more legible as "Grep x40" than as forty rows, and
        # the raw list is unbounded.
        counts: dict[str, int] = {}
        for name in self.tool_calls:
            counts[name] = counts.get(name, 0) + 1
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "continuation_of": self.continuation_of,
            "subagent_type": self.subagent_type,
            "description": self.description,
            "prompt_preview": self.prompt_preview,
            "parent_session_id": self.parent_session_id,
            "parent_turn_id": self.parent_turn_id,
            "session_id": self.session_id,
            "model": self.model,
            "max_turns": self.max_turns,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(self.elapsed_s, 1),
            "status": self.status,
            "turns": self.turns,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "response_chars": self.response_chars,
            "tool_call_count": len(self.tool_calls),
            "tool_counts": counts,
            "last_tool": self.tool_calls[-1] if self.tool_calls else "",
            # Capabilities, not handles. `snapshot()` is a JSON surface
            # (`main.py` → Mission Control): an `asyncio.Event` is not
            # serialisable and a conversation is unbounded, so the row says
            # whether this run can be steered or killed, and how many times
            # it has been — the objects themselves stay in the process.
            "steerable": self.chat_messages is not None,
            "cancellable": self.cancel_event is not None,
            "injected": self.injected,
        }


def _new_run_id() -> str:
    global _counter
    _counter += 1
    return f"sub-{time.strftime('%H%M%S')}-{_counter:03d}"


def _new_task_id() -> str:
    global _task_counter
    _task_counter += 1
    return f"sub-{time.strftime('%H%M%S')}-t{_task_counter:03d}"


def register(
    *,
    subagent_type: str,
    description: str,
    prompt: str,
    parent_session_id: str,
    session_id: str,
    model: str,
    max_turns: int,
    parent_turn_id: str = "",
    task_id: str = "",
    continuation_of: str = "",
    chat_messages: list[dict[str, Any]] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> SubagentRecord:
    """Open a row for a Task run that is about to start.

    `task_id` is carried in on a resume so the continuation keeps the same
    identity; a fresh Task mints one. `run_id` stays per-run either way —
    the dashboard shows runs, and a resume is a new run of the same task.

    `chat_messages` / `cancel_event` are the child's live handles. They must
    be the objects handed to `RunOptions`, and register must happen BEFORE
    the run starts — a row published after `run_query` returns is a
    post-mortem, which is the state #411 is about.
    """
    record = SubagentRecord(
        run_id=_new_run_id(),
        task_id=task_id or _new_task_id(),
        continuation_of=continuation_of,
        subagent_type=subagent_type,
        description=description,
        prompt_preview=prompt[:200],
        parent_session_id=parent_session_id,
        parent_turn_id=parent_turn_id,
        session_id=session_id,
        model=model,
        max_turns=max_turns,
        started_at=time.time(),
        chat_messages=chat_messages,
        cancel_event=cancel_event,
    )
    _active[record.run_id] = record
    return record


def finish(
    record: SubagentRecord,
    *,
    status: str,
    stop_reason: str = "",
    error: str = "",
    response_chars: int = 0,
) -> None:
    """Close a run and move it from active to the recent ring.

    Safe to call twice — the second call is a no-op, so a `finally` that
    races an explicit close can't double-append to the ring.
    """
    if record.run_id not in _active:
        return
    record.finished_at = time.time()
    record.status = status
    record.stop_reason = stop_reason
    record.error = error
    record.response_chars = response_chars
    _active.pop(record.run_id, None)
    _recent.appendleft(record)


def list_active() -> list[dict[str, Any]]:
    """In-flight runs, longest-running first — the ones worth watching."""
    return [
        r.to_dict()
        for r in sorted(_active.values(), key=lambda r: r.started_at)
    ]


def list_recent(limit: int = _RECENT_LIMIT) -> list[dict[str, Any]]:
    """Most recently finished runs, newest first."""
    return [r.to_dict() for r in list(_recent)[:limit]]


def snapshot() -> dict[str, Any]:
    active = list_active()
    return {
        "active": active,
        "active_count": len(active),
        "recent": list_recent(),
    }


# ---------------------------------------------------------------------------
# Control surface: steer a running child, or stop it (#411)
# ---------------------------------------------------------------------------
#
# `Task` blocks its caller until the subagent returns, so the caller of a
# steering control surface is never the blocked tool call itself — it is
# something else acting in this process for the same session. That is why
# authority is a *session* and not a turn: a turn that is blocked on a wedged
# child cannot also be the one killing it, and the useful case after that is
# the parent session's next turn cleaning up the child it left behind.
#
# Session-scoped, fail-closed, and the narrowest of the shapes the item
# named. Widening it — the Inner Voice observer, a Mission Control button —
# is a decision this module deliberately does not make for whoever asks
# first; see `backlog/411-*` ("Needs a person").


def active_run(task_id: str) -> SubagentRecord | None:
    """The live row for a task_id, or None. One active run per task_id: a
    second resume of the same id is refused by `claim_history`."""
    for record in _active.values():
        if record.task_id == task_id:
            return record
    return None


def _require_authority(record: SubagentRecord, caller: CallerScope, verb: str) -> None:
    """Refuse, and say so in the log, unless `caller` IS the orchestrator.

    The refusal is a raise *and* a log line because both audiences matter:
    the caller needs the reason, and whoever is watching a subagent that got
    steered without anyone admitting it needs the record.
    """
    if not caller.session_id or caller.session_id != record.parent_session_id:
        detail = (
            f"steering refused: {verb} on subagent {record.task_id} "
            f"(run {record.run_id}) by caller session "
            f"{caller.session_id or '<unbound>'!r}, which is not the "
            f"orchestrating session "
            f"{record.parent_session_id or '<unbound>'!r}"
        )
        logger.warning(detail)
        raise SteeringRefused(detail)


def steer(task_id: str, text: str, *, caller: CallerScope,
          reason: str = "") -> dict[str, str]:
    """Append one orchestrator message to a running child's live conversation.

    Returns the message that was appended — the same dict now in the child's
    list, so a caller can assert what the child will actually see. Raises
    `SteeringRefused` and appends nothing if the run is gone, has no
    published handle, carries no text, or the caller is not the orchestrating
    session.

    Ordering caveat, stated rather than hidden: an out-of-band append lands
    wherever the list is at that instant, which may be before the assistant
    turn it answers. The Inner Voice path avoids that by injecting after the
    turn is appended; an external caller has no such moment to wait for.
    """
    record = active_run(task_id)
    if record is None:
        raise SteeringRefused(
            f"steering refused: no active subagent run for task_id {task_id!r} "
            "(finished, evicted, or never spawned)"
        )
    _require_authority(record, caller, "append")
    if record.chat_messages is None:
        raise SteeringRefused(
            f"steering refused: run {record.run_id} published no conversation "
            "handle, so there is nothing to append to"
        )
    if not text or not text.strip():
        raise SteeringRefused("steering refused: nothing to say")

    message = steering_message(text)
    record.chat_messages.append(message)
    record.injected += 1
    logger.info(
        "subagent %s steered by session %r (%d injected%s)%s",
        record.task_id, caller.session_id, record.injected,
        f" — {reason}" if reason else "",
    )
    return message


def cancel_run(task_id: str, *, caller: CallerScope, reason: str = "") -> bool:
    """Ask a running child to stop, via the event its loop already checks.

    Not a kill: `app/harness/loop.py` checks this event between iterations,
    between SSE chunks and before each tool dispatch, and races an in-flight
    MCP tool call — so the child stops at its next safe point with
    `stop_reason == "cancelled"`, which is also what lets `_close` store its
    history and keep the task_id resumable.
    """
    record = active_run(task_id)
    if record is None:
        raise SteeringRefused(
            f"cancelling refused: no active subagent run for task_id {task_id!r}"
        )
    _require_authority(record, caller, "cancel")
    if record.cancel_event is None:
        raise SteeringRefused(
            f"cancelling refused: run {record.run_id} published no cancel event"
        )
    record.cancel_event.set()
    logger.info(
        "subagent %s cancelled by session %r%s",
        record.task_id, caller.session_id, f" — {reason}" if reason else "",
    )
    return True


# ---------------------------------------------------------------------------
# Resumable history
# ---------------------------------------------------------------------------
#
# A Task used to start from nothing every call. That is right for a
# fire-and-forget fan-out and wrong for the case that keeps recurring: a
# subagent burns its budget mid-investigation, the caller reads the partial
# answer, and the only way to ask a follow-up is to pay for the whole
# investigation again — a fresh prompt, a cold KV cache, and no memory of the
# forty tool results it just collected.
#
# The store is a *bounded* one, and deliberately process-scoped. After an
# aggregator restart every task_id reads as `unknown or evicted`, which is
# honest: the conversation it would have continued is gone with the process.


class HistoryUnavailable(Exception):
    """A resume that cannot happen, with a reason the model can act on."""


@dataclass
class SubagentHistory:
    task_id: str
    subagent_type: str
    profile: dict
    model: str
    base_url: str
    # The `task:<type>:<hex8>` id. Reused on resume so the continuation keeps
    # its tool_search LoadedToolSet and its spill directory.
    session_id: str
    description: str
    # THE list the loop mutated, sanitised. Passed back as
    # `chat_messages_handle`, which is why the follow-up user message is
    # appended to it rather than sent as `messages` — `run_query` ignores
    # `messages` entirely when the handle is non-empty.
    chat_messages: list[dict]
    last_run_id: str
    runs: int
    finished_at: float
    chars: int
    active: bool = False


_HISTORY_KEEP = 8
_HISTORY_TTL_S = 1800
_HISTORY_MAX_CHARS = 3_000_000

_history: "OrderedDict[str, SubagentHistory]" = OrderedDict()


def _sanitise(messages: list[dict]) -> list[dict]:
    """Drop a trailing assistant message whose tool calls were never answered.

    That is the only invalid shape the loop can leave behind: a cancel or an
    exception between the stream ending and the dispatch completing
    (`loop.py` around the tool-dispatch block). Replaying it would send vLLM
    an assistant `tool_calls` with no matching `tool` messages, which every
    engine rejects.
    """
    out = list(messages)
    while out:
        last = out[-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            out.pop()
            continue
        break
    return out


def _sizeof(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        total += len(c) if isinstance(c, str) else len(str(c))
    return total


def store_history(*, task_id: str, subagent_type: str, profile: dict, model: str,
                  base_url: str, session_id: str, description: str,
                  chat_messages: list[dict], run_id: str, runs: int) -> None:
    """Keep a finished run's conversation so a follow-up can continue it."""
    messages = _sanitise(chat_messages)
    if not messages:
        return
    now = time.time()
    # TTL sweep first, then oldest-first until both caps hold.
    for key in [k for k, h in _history.items()
                if now - h.finished_at > _HISTORY_TTL_S and not h.active]:
        _history.pop(key, None)

    entry = SubagentHistory(
        task_id=task_id, subagent_type=subagent_type, profile=dict(profile),
        model=model, base_url=base_url, session_id=session_id,
        description=description, chat_messages=messages, last_run_id=run_id,
        runs=runs, finished_at=now, chars=_sizeof(messages), active=False,
    )
    _history[task_id] = entry
    _history.move_to_end(task_id)

    while len(_history) > _HISTORY_KEEP:
        _history.popitem(last=False)
    while sum(h.chars for h in _history.values()) > _HISTORY_MAX_CHARS and len(_history) > 1:
        _history.popitem(last=False)


def claim_history(task_id: str) -> SubagentHistory:
    """Take a stored conversation for a resume, or say exactly why not."""
    entry = _history.get(task_id)
    if entry is None:
        raise HistoryUnavailable("unknown or evicted")
    if entry.active:
        raise HistoryUnavailable("still running")
    if time.time() - entry.finished_at > _HISTORY_TTL_S:
        _history.pop(task_id, None)
        raise HistoryUnavailable("expired")
    entry.active = True
    _history.move_to_end(task_id)
    return entry


def release_history(task_id: str) -> None:
    """Un-claim a resume whose run raised before it produced anything."""
    entry = _history.get(task_id)
    if entry is not None:
        entry.active = False


def history_stats() -> dict:
    return {"tasks": len(_history),
            "chars": sum(h.chars for h in _history.values()),
            "ids": list(_history)}


def parent_scope(session_id: str) -> tuple[str, str] | None:
    """The (session, turn) a `task:*` subagent session belongs to.

    Anything produced by a subagent that arrives *after* its Task returns
    has nowhere to go: nothing reads a subagent's drain queue once the run
    is over. Attributing it to the parent is the only delivery that reaches
    a reader. Active runs are searched first, then the recent ring, because
    a result can land in the seconds after the run closed.
    """
    for record in _active.values():
        if record.session_id == session_id:
            return record.parent_session_id, record.parent_turn_id
    for record in _recent:
        if record.session_id == session_id:
            return record.parent_session_id, record.parent_turn_id
    return None


def reset() -> None:
    """Drop all state. Tests only."""
    _active.clear()
    _recent.clear()
    _history.clear()
