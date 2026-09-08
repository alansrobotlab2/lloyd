"""Research registry tools — propose a topic, see what came of one.

The registry (`app.research_store`) is written from two processes: the
backend's `deep-research` worker, and this module inside the MCP aggregator.
That is the same shape `workers.db` has, and the store handles it in SQL.

**`research_next` is deliberately peek-only.** Only the worker claims a topic,
because a claim taken by a chat turn that then wanders off leaves the topic
stuck in `researching` until `reclaim_stale` sweeps it. A human who researches
something by hand records it afterwards with `research_complete`.

Every store call is hopped onto a thread: the aggregator awaits `call_tool` on
its one event loop, and that loop is serving every tool call for every session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import Any, Callable

from mcp.types import Tool

from agent_mcp._shared import get_bound_session, text_result

logger = logging.getLogger("lloyd-mcp.research")

_DOMAINS = ("robotics", "local-llm-serving", "agent-architecture",
            "knowledge-memory", "computer-vision", "tooling-infra", "voice-tts")

#: Mirrors `app.research_store.STATUSES`. A literal rather than an import,
#: because `list_tools` must not be able to raise and an import of the store
#: here would open a code path to SQLite just to build a schema string.
#: `tests/test_research_tools.py` pins the two together.
_STATUSES = ("queued", "researching", "written", "nothing_found",
             "duplicate", "archived", "rejected")


async def list_tools() -> list[Tool]:
    """The advertised surface.

    Pure and unable to raise on purpose: `agent_mcp/main.py` records a module
    whose `list_tools` throws as `ok: False`, and any such module turns the
    whole aggregator's `/health` into a 503. Nothing here touches the database.
    """
    return [
        Tool(
            name="research_propose",
            description=(
                "Propose a topic for the deep-research worker to investigate. "
                "Deduplicates automatically: a topic already in the registry is "
                "returned rather than added again, and the response lists similar "
                "existing topics with their outcomes so you can drop or sharpen a "
                "near-duplicate instead of researching it twice."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "One self-contained line, roughly 40-200 characters. No "
                            "pronouns and no bare 'recent'/'latest' — name the paper, "
                            "library, technique or system so it still reads in a month."
                        ),
                    },
                    "domain": {
                        "type": "string",
                        "description": f"Subject area, e.g. one of: {', '.join(_DOMAINS)}",
                    },
                    "signal": {
                        "type": "string",
                        "description": (
                            "Where the idea came from: health-report, "
                            "session-distill-gaps, backlog, daily-note, or human. "
                            "Recorded so a later reader can tell which signals earn "
                            "their keep."
                        ),
                    },
                    "priority": {
                        "type": "integer",
                        "description": "0-100, lower is researched sooner. Default 50.",
                    },
                    "proposed_by": {
                        "type": "string",
                        "description": (
                            "Who is proposing. Defaults to the calling session id; "
                            "pass 'human' when relaying a request from the user."
                        ),
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="research_next",
            description=(
                "Peek at the topics the deep-research worker will take next, most "
                "urgent first. Read-only: it does not claim anything, so calling it "
                "never takes work away from the worker or leaves a topic stuck."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "How many topics to look at, 1-10. Default 1.",
                    },
                },
            },
        ),
        Tool(
            name="research_complete",
            description=(
                "Record the outcome of researching a topic: written with the note it "
                "produced, nothing_found when the search came up empty, duplicate when "
                "the vault already covers it, or rejected when it should not have been "
                "proposed. An outcome that is already recorded is never rewritten."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic_id": {
                        "type": "integer",
                        "description": "Registry id of the topic, from research_next or research_list.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["written", "nothing_found", "duplicate", "rejected"],
                        "description": (
                            "nothing_found is a real result, not a failure: it is what "
                            "stops the topic being proposed again."
                        ),
                    },
                    "artifact_path": {
                        "type": "string",
                        "description": "Path of the note that was written, for status=written.",
                    },
                    "note": {
                        "type": "string",
                        "description": "One line on what was found, or why nothing was.",
                    },
                    "duplicate_of": {
                        "type": "integer",
                        "description": "Registry id this duplicates, for status=duplicate.",
                    },
                },
                "required": ["topic_id", "status"],
            },
        ),
        Tool(
            name="research_list",
            description=(
                "List registry topics, newest activity first, filtered by status, "
                "domain or recency. Use since_days to see what has already been "
                "researched or ruled out before proposing anything new."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": list(_STATUSES),
                        "description": "Only topics in this state.",
                    },
                    "domain": {"type": "string", "description": "Only topics in this subject area."},
                    "since_days": {
                        "type": "integer",
                        "description": "Only topics proposed or settled within this many days.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return, up to 200. Default 50.",
                    },
                },
            },
        ),
        Tool(
            name="research_stats",
            description=(
                "Registry health in one call: how many topics sit in each state, how "
                "many were settled today, the depth and age of the queue, and the last "
                "note written. Start here before proposing or reporting on research."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def call_tool(name: str, arguments: dict):
    handlers: dict[str, Callable[[dict], dict]] = {
        "research_propose": _handle_propose,
        "research_next": _handle_next,
        "research_complete": _handle_complete,
        "research_list": _handle_list,
        "research_stats": _handle_stats,
    }
    handler = handlers.get(name)
    if handler is None:
        return text_result(json.dumps({"error": f"unknown tool: {name}"}))
    # Off the loop: every one of these opens SQLite, and this coroutine is
    # awaited directly by the aggregator's dispatcher.
    payload = await asyncio.to_thread(_guarded, handler, arguments)
    return text_result(json.dumps(payload, default=str))


def _guarded(handler: Callable[[dict], dict], arguments: dict) -> dict:
    """Run a handler, turning every failure into a readable error payload.

    Handlers never raise out of this module: an exception escaping `call_tool`
    reaches the model as a transport error with no explanation of what it did
    wrong, and `text_result` already renders `{"error": ...}` as `isError`.
    """
    from app.research_store import StoreUnavailable

    try:
        return handler(arguments or {})
    except ValueError as exc:
        return {"error": str(exc)}
    except StoreUnavailable as exc:
        logger.error("research registry unavailable: %s", exc)
        return {"error": f"research registry unavailable: {exc}"}
    except sqlite3.OperationalError as exc:
        # The store's busy_timeout is 5s; past that something is holding a
        # write transaction open, which is a person, not contention.
        return {"error": f"research registry busy: {exc}"}
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("research tool failed: %s", exc, exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


def _store():
    from app.research_store import store
    return store()


def _handle_propose(params: dict) -> dict:
    topic = str(params.get("topic") or "").strip()
    if not topic:
        return {"error": "topic is required"}
    out = _store().propose(
        topic,
        domain=str(params.get("domain") or ""),
        signal=str(params.get("signal") or ""),
        priority=int(params.get("priority") or 50),
        proposed_by=str(params.get("proposed_by") or "") or get_bound_session() or "unbound",
    )
    if out.get("refused"):
        out["message"] = out["refused"]
    elif out["created"]:
        out["message"] = f"queued as #{out['id']}"
    else:
        out["message"] = (f"already known as #{out['id']} ({out['status']}); "
                          f"not proposed again")
    return out


def _handle_next(params: dict) -> dict:
    n = max(1, min(int(params.get("n") or 1), 10))
    return {"topics": _store().next(n), "note": "peek only; the worker claims"}


def _handle_complete(params: dict) -> dict:
    topic_id = params.get("topic_id")
    if topic_id is None:
        return {"error": "topic_id is required"}
    status = str(params.get("status") or "")
    if status not in ("written", "nothing_found", "duplicate", "rejected"):
        return {"error": f"status must be written|nothing_found|duplicate|rejected, "
                         f"not {status!r}"}
    return {"topic": _store().finish(
        int(topic_id), status,
        artifact_path=str(params.get("artifact_path") or ""),
        note=str(params.get("note") or ""),
        duplicate_of=params.get("duplicate_of"),
        session_id=get_bound_session(),
    )}


def _handle_list(params: dict) -> dict:
    rows = _store().list(
        status=str(params.get("status") or "") or None,
        domain=str(params.get("domain") or "") or None,
        since_days=int(params["since_days"]) if params.get("since_days") else None,
        limit=int(params.get("limit") or 50),
    )
    return {"topics": rows, "count": len(rows)}


def _handle_stats(params: dict) -> dict:
    return _store().stats()
