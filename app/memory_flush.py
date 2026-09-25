"""Memory flush before compaction (review 2026-09-24, P3). Ships off.

A long chat reaches the compaction wall and its older turns are summarised or
dropped. Whatever durable fact, decision or preference lived only in those
turns survives only as well as the summary kept it. OpenClaw's answer, and
this one: shortly BEFORE the wall, give the model one quiet turn to write what
is worth keeping into memory (`memory_add`, `fact_add`), while the turns are
still verbatim in its context.

The pieces, and where they live:

* **Trigger** — the end of a chat turn (`_run_turn`'s `result` branch), not
  inside `load_and_compact_session`: the flush is a whole agent turn and the
  compaction stack runs inside the next one. `should_flush` fires when the
  engine-reported peak prompt of the turn just finished is at least
  ``trigger_fraction`` × the compaction threshold, and no flush has happened in
  this compaction cycle.
* **The turn** — `app.routers.messages.build_flush_turn`: an ambient-tier turn
  (runs when the session is idle, a user turn preempts it) with
  `RunOptions.allowed_tools` restricted to the memory tools, tool search off,
  no Inner Voice. Its `turn_id` carries `app.compaction.FLUSH_TURN_PREFIX`, so
  every row it writes is kept in the transcript and dropped from history.
* **Bookkeeping** — ``data["compaction"]["flush"]``, beside the D2 summary
  record (`app/compaction_state.py`, whose `save_record` carries it over). A
  cycle ends when the summary record is updated after the flush, or when a
  later turn's start summarised or truncated (`note_turn_start`), which is the
  only signal there is with ``compaction.persist_summary`` off.
* **Measure** — event ``compaction.memory_flush`` when the flush turn ends, and
  ``flushed_before_summary`` on the turn-start compaction record.

Off switch: ``compaction.memory_flush.enabled`` (false). Off, nothing is
enqueued and nothing is written; rows a past flush wrote are still dropped
from history, which is what they always were meant to be.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Iterable

from app.compaction import FLUSH_TURN_PREFIX

logger = logging.getLogger("lloyd-memory-flush")

#: The ambient producer name. `_iv_should_fire_on_turn` skips it, and the
#: queue's dedup key is the same string, so two flushes never stack.
PRODUCER = "memory_flush"

#: The tools a flush turn is advertised and may dispatch. `fact_search` in the
#: plan does not exist in the aggregator (`fact_get` covers the read).
DEFAULT_TOOLS = ("memory_read", "memory_add", "fact_get", "fact_add")

#: A flush queued this long ago that never reported back (dropped from the
#: queue, backend restarted) no longer holds the cycle closed.
STALE_QUEUED_SECONDS = 3600.0

_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "trigger_fraction": 0.85,
    "max_turns": 6,
    "tools": list(DEFAULT_TOOLS),
    "once_per_cycle": True,
}

FLUSH_PROMPT = (
    "This conversation is about to reach its context limit, and its older "
    "turns will then be summarised. Before that happens, save what is worth "
    "keeping.\n\n"
    "Read memory first (`memory_read`) so you do not duplicate an entry. Then "
    "persist anything durable from this conversation that memory does not "
    "already hold:\n"
    "- a ruling or correction the user gave you → `memory_add` with "
    "type `feedback`;\n"
    "- a fact about the user → `memory_add` to USER.md with type `user`;\n"
    "- a decision, an open commitment or working state that outlives this "
    "chat → `memory_add` with type `project`;\n"
    "- where something lives (a path, a URL, a doc) → `memory_add` with type "
    "`reference`;\n"
    "- a fact about a named entity (a person, a project, a system) → "
    "`fact_add`.\n\n"
    "One entry per fact, one sentence each. Skip anything already saved, "
    "anything true only of this moment, and anything you are unsure of. If "
    "nothing needs saving, save nothing.\n\n"
    "Do not address the user and do not continue the conversation. End with "
    "exactly one line naming what you saved (or `Saved nothing.`)."
)


def flush_cfg() -> dict[str, Any]:
    """``compaction.memory_flush`` with defaults. Read per call, like
    `app.compaction._compaction_cfg`, so a config edit applies next turn."""
    try:
        from app.config import CONFIG  # type: ignore
        raw = (CONFIG.get("compaction") or {}).get("memory_flush") or {}
    except Exception:  # noqa: BLE001
        raw = {}
    cfg = dict(_DEFAULTS)
    if isinstance(raw, dict):
        cfg.update({k: v for k, v in raw.items() if v is not None})
    cfg["tools"] = [str(t) for t in (cfg.get("tools") or DEFAULT_TOOLS)]
    return cfg


def new_turn_id() -> str:
    """A SessionTurn id every row of the flush turn will carry (X3)."""
    return f"{FLUSH_TURN_PREFIX}{uuid.uuid4().hex[:12]}"


def is_flush_turn_id(turn_id: str | None) -> bool:
    return isinstance(turn_id, str) and turn_id.startswith(FLUSH_TURN_PREFIX)


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------


def _record(data: dict | None) -> dict:
    rec = (data or {}).get("compaction") if isinstance(data, dict) else None
    return rec if isinstance(rec, dict) else {}


def flush_entry(data: dict | None) -> dict | None:
    entry = _record(data).get("flush")
    return entry if isinstance(entry, dict) else None


def _in_this_cycle(entry: dict | None, data: dict | None, now: float) -> bool:
    """Does ``entry`` belong to the compaction cycle that is still open?"""
    if not entry or entry.get("consumed"):
        return False
    at = float(entry.get("at") or 0)
    updated = _record(data).get("updated_at")
    # D2: a summary record updated after the flush means a fold has run since,
    # which is the end of that cycle.
    if isinstance(updated, (int, float)) and at < float(updated):
        return False
    status = entry.get("status") or ""
    if status in ("cancelled", "failed"):
        return False
    if status == "queued" and now - at > STALE_QUEUED_SECONDS:
        return False
    return True


def flushed_this_cycle(data: dict | None) -> bool:
    """A flush has FINISHED in the open cycle (`flushed_before_summary`)."""
    entry = flush_entry(data)
    return (_in_this_cycle(entry, data, time.time())
            and (entry or {}).get("status") == "done")


def should_flush(data: dict | None, *, peak_tokens: int, threshold: int,
                 cfg: dict | None = None, now: float | None = None) -> bool:
    cfg = cfg if cfg is not None else flush_cfg()
    if not cfg.get("enabled"):
        return False
    if threshold <= 0 or peak_tokens <= 0:
        return False
    if peak_tokens < float(cfg.get("trigger_fraction") or 0.85) * threshold:
        return False
    if cfg.get("once_per_cycle", True) and _in_this_cycle(
            flush_entry(data), data, time.time() if now is None else now):
        return False
    return True


# ---------------------------------------------------------------------------
# Writes. Each is one small `mutate_session`; the key stays after `messages`
# (X6), because the retention sweep reads a 4 KB prefix for `last_active`.
# ---------------------------------------------------------------------------


def _set_flush(data: dict, update: dict, *, replace: bool = False) -> None:
    rec = data.pop("compaction", None)
    rec = rec if isinstance(rec, dict) else {}
    cur = rec.get("flush") if isinstance(rec.get("flush"), dict) else {}
    rec["flush"] = dict(update) if replace else {**cur, **update}
    data["compaction"] = rec


async def mark_queued(session_id: str, *, turn_id: str, trigger_tokens: int,
                      threshold: int) -> bool:
    from app import sessions_io

    entry = {"turn_id": turn_id, "at": time.time(), "status": "queued",
             "trigger_tokens": int(trigger_tokens), "threshold": int(threshold)}
    try:
        return await sessions_io.mutate_session(
            session_id, lambda d: _set_flush(d, entry, replace=True))
    except Exception as e:  # noqa: BLE001
        logger.warning("memory_flush: mark_queued failed for %s: %s", session_id, e)
        return False


async def mark_consumed(session_id: str) -> bool:
    """A turn start summarised or truncated: the open cycle is over."""
    from app import sessions_io

    def _fn(d: dict) -> None:
        entry = flush_entry(d)
        if entry and not entry.get("consumed"):
            _set_flush(d, {"consumed": True, "consumed_at": time.time()})
    try:
        return await sessions_io.mutate_session(session_id, _fn)
    except Exception as e:  # noqa: BLE001
        logger.warning("memory_flush: mark_consumed failed for %s: %s", session_id, e)
        return False


def count_saves(tool_names: Iterable[str]) -> dict[str, int]:
    names = list(tool_names)
    return {"memory_adds": sum(1 for n in names if n == "memory_add"),
            "fact_adds": sum(1 for n in names if n == "fact_add")}


async def record_done(session_id: str, *, turn_id: str, tool_names: Iterable[str],
                      duration_ms: int, stop_reason: str) -> dict[str, Any]:
    """The flush turn ended: book it and emit ``compaction.memory_flush``."""
    from app import sessions_io
    from app.harness.telemetry import log_harness_event

    counts = count_saves(tool_names)
    status = "cancelled" if stop_reason == "cancelled" else "done"
    trigger = 0

    def _fn(d: dict) -> None:
        nonlocal trigger
        entry = flush_entry(d) or {}
        trigger = int(entry.get("trigger_tokens") or 0)
        # Only this flush's own entry: a newer one has replaced it otherwise.
        if entry.get("turn_id") not in (None, "", turn_id):
            return
        _set_flush(d, {"turn_id": turn_id, "status": status,
                       "done_at": time.time(), "duration_ms": int(duration_ms),
                       "stop_reason": stop_reason, **counts})
    try:
        await sessions_io.mutate_session(session_id, _fn)
    except Exception as e:  # noqa: BLE001
        logger.warning("memory_flush: record_done failed for %s: %s", session_id, e)
    event = {"turn_id": turn_id, "trigger_tokens": trigger, "status": status,
             "stop_reason": stop_reason, "duration_ms": int(duration_ms), **counts}
    log_harness_event(session_id, "compaction.memory_flush", event, turn_id=turn_id)
    return event


async def after_turn(session_id: str, *, turn_id: str, turn_start: dict | None,
                     peak_tokens: int, threshold: int, enqueue) -> bool:
    """Called at the end of every non-flush chat turn. True when a flush
    was enqueued. ``enqueue(turn_id)`` builds and enqueues the flush turn;
    it is passed in so this module never imports the router.

    Never raises: this runs on the path that answers the user.
    """
    try:
        import json

        from app import sessions_io

        cfg = flush_cfg()
        if not cfg.get("enabled") or is_flush_turn_id(turn_id):
            return False
        ts = turn_start or {}
        if ts.get("summarized") or ts.get("truncated"):
            await mark_consumed(session_id)
        path = sessions_io.SESSIONS_DIR / f"{session_id}.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        if not should_flush(data, peak_tokens=peak_tokens, threshold=threshold, cfg=cfg):
            return False
        flush_turn_id = new_turn_id()
        await mark_queued(session_id, turn_id=flush_turn_id,
                          trigger_tokens=peak_tokens, threshold=threshold)
        await enqueue(flush_turn_id)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("memory_flush: after_turn failed for %s: %s", session_id, e)
        return False


__all__ = [
    "PRODUCER",
    "DEFAULT_TOOLS",
    "FLUSH_PROMPT",
    "flush_cfg",
    "new_turn_id",
    "is_flush_turn_id",
    "flush_entry",
    "flushed_this_cycle",
    "should_flush",
    "mark_queued",
    "mark_consumed",
    "count_saves",
    "record_done",
    "after_turn",
]
