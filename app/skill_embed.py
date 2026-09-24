"""Where skill bytes enter a prompt, and how many (#624).

A SKILL.md reaches the model by three routes and they cost very different
amounts. The chat turn-start injection is capped (`prefetch.SKILL_BODY_MAX`, 6000
chars, with an excerpt for the runner-up), so a 1,600-line skill costs a chat turn
the same as a 200-line one. The autonomy task prompt (`autonomy._build_task_prompt`)
and the worker prompt (`workers.sources._common.build_skill_prompt`) splice the
whole file in with no cap and carry it for the run. #624's premise — that shorter
bodies save tokens — is only true on the second kind, so the question "which route
cost what" has to be answerable before a spill pass is judged on it.

`record_skill_embed` writes one `skill.embedded` event into the run's own session
event log (the per-session machine record, `app/event_log.py`) and one
`SKILL_EMBED` INFO line, per skill per turn or run, tagged with the route. A
report over `event_logs/*.events.jsonl` can then sum `embedded_chars` by route.

Accounting, never the turn: nothing here raises.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("lloyd-server")

EVENT = "skill.embedded"


def record_skill_embed(session_id: str | None, *, route: str, skill: str,
                       embedded_chars: int, source_chars: int | None = None,
                       truncated: bool | None = None,
                       turn_id: str | None = None) -> dict[str, Any]:
    """Record one skill body entering a prompt by `route`; return the record.

    `embedded_chars` is what the prompt carries; `source_chars` is the file's own
    size when the caller has it, so a capped route shows what it left out.
    Without a session id only the log line is written.
    """
    record: dict[str, Any] = {"route": route, "skill": skill,
                              "embedded_chars": int(embedded_chars)}
    if source_chars is not None:
        record["source_chars"] = int(source_chars)
    if truncated is not None:
        record["truncated"] = bool(truncated)
    try:
        logger.info("SKILL_EMBED route=%s skill=%s embedded=%dc source=%s session=%s",
                    route, skill, record["embedded_chars"],
                    f"{source_chars}c" if source_chars is not None else "-",
                    session_id or "-")
        if session_id:
            from app import event_log
            event_log.log_event(session_id, EVENT, record, turn_id=turn_id)
    except Exception as exc:  # noqa: BLE001 — a lost record is never an error
        logger.debug("skill_embed: record failed for %s: %s", session_id, exc)
    return record


def record_context_skills(session_id: str | None, context_text: str,
                          turn_id: str | None = None) -> list[dict[str, Any]]:
    """Record every skill a turn-start `<context>` block carried, with its size."""
    try:
        from app.harness.skill_dispatch import skill_delivery_sizes
        sizes = skill_delivery_sizes(context_text or "")
    except Exception as exc:  # noqa: BLE001
        logger.debug("skill_embed: could not read the context block: %s", exc)
        return []
    return [record_skill_embed(session_id, route=d["route"], skill=d["name"],
                               embedded_chars=d["chars"], truncated=d["truncated"],
                               turn_id=turn_id)
            for d in sizes]
