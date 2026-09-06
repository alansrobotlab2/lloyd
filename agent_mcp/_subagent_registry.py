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

Lifetime is process-scoped, matching `_task_registry` (background bash
tasks) next door. Completed runs stay in a bounded ring so the dashboard
can show what *just* happened, not only what is happening; the ring is
capped because this is a live view, not an audit log — `event_logs/` is
the durable record.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# How many finished runs to keep for the "recent" panel. Small on
# purpose: the dashboard shows a handful and the event log holds history.
_RECENT_LIMIT = 20

_active: dict[str, "SubagentRecord"] = {}
_recent: deque["SubagentRecord"] = deque(maxlen=_RECENT_LIMIT)

_counter = 0


@dataclass
class SubagentRecord:
    run_id: str
    subagent_type: str
    description: str
    prompt_preview: str
    parent_session_id: str
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
            "subagent_type": self.subagent_type,
            "description": self.description,
            "prompt_preview": self.prompt_preview,
            "parent_session_id": self.parent_session_id,
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
        }


def _new_run_id() -> str:
    global _counter
    _counter += 1
    return f"sub-{time.strftime('%H%M%S')}-{_counter:03d}"


def register(
    *,
    subagent_type: str,
    description: str,
    prompt: str,
    parent_session_id: str,
    session_id: str,
    model: str,
    max_turns: int,
) -> SubagentRecord:
    """Open a row for a Task run that is about to start."""
    record = SubagentRecord(
        run_id=_new_run_id(),
        subagent_type=subagent_type,
        description=description,
        prompt_preview=prompt[:200],
        parent_session_id=parent_session_id,
        session_id=session_id,
        model=model,
        max_turns=max_turns,
        started_at=time.time(),
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


def reset() -> None:
    """Drop all state. Tests only."""
    _active.clear()
    _recent.clear()
