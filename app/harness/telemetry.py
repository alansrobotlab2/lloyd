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

import logging
from typing import Any

logger = logging.getLogger("lloyd-harness-telemetry")


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
    if not session_id:
        return
    try:
        from app import event_log

        event_log.log_event(session_id, event, data, turn_id=turn_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("telemetry: could not log %s: %s", event, exc)
