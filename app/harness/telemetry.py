"""The harness's one event sink (review 2026-09-24, X1).

`log_harness_event` lived in `loop.py` as `_log_harness_event`, which meant a
module that wanted to record a harness event had to import the agent loop to
do it. `hooks.py`, `mcp_pool.py` and the compaction state all sit *below* the
loop — the loop imports them — so none of them could, and the events they
should emit (a hook that raised, a discovery refresh, a persisted summary) had
nowhere to go. This module imports nothing from the harness, so anything may
import it. `loop._log_harness_event` stays as an alias so no caller or test
that names it moves.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Any

logger = logging.getLogger("lloyd-harness-telemetry")

# Per-turn tallies of the events written here (P11). A turn's usage writer
# binds a dict (`bind_event_counts`) before it drives `run_query`, and every
# `log_harness_event` in that context bumps it — including ones from hooks and
# parallel tool tasks, which inherit the context and so share the same dict.
# That is how `harness.hook_raised` (hooks.py) and `harness.stream_retried`
# reach the usage row without either module knowing a row exists. Counted
# before the session check: a bare `run_query` caller with no session still
# has a turn whose row wants the number.
_event_counts: contextvars.ContextVar[dict[str, int] | None] = \
    contextvars.ContextVar("lloyd_harness_event_counts", default=None)


def bind_event_counts() -> dict[str, int]:
    """Start counting harness events for the current context; return the dict.

    The binding is left in place rather than reset: the writers that call
    this own the turn's task, and the next turn binds a fresh dict over it.
    """
    counts: dict[str, int] = {}
    _event_counts.set(counts)
    return counts


def log_harness_event(
    session_id: str, event: str, data: dict[str, Any],
    *, turn_id: str | None = None,
) -> None:
    """Append one event to the session's event log, or give up quietly.

    Lazy and guarded: `app.harness` is importable without a full app
    bootstrap (bench scripts and `app/harness/tests` rely on that), and a
    turn must never die because its diagnostics could not be written.

    `turn_id` is optional because most callers have only ever had a
    session; the relief record passes the run's own, so one firing can be
    attributed to the turn that needed it (#1078).
    """
    counts = _event_counts.get()
    if counts is not None:
        counts[event] = counts.get(event, 0) + 1
    if not session_id:
        return
    try:
        from app import event_log

        event_log.log_event(session_id, event, data, turn_id=turn_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("telemetry: could not log %s: %s", event, exc)
