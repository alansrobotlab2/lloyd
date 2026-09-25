"""Running `Task` subagents: list, stop, steer (review 2026-09-24, P8).

GET  /api/subagents                    — the aggregator's live subagent rows.
POST /api/subagents/{task_id}/cancel   — {session_id, reason?}: stop a running child.
POST /api/subagents/{task_id}/steer    — {session_id, text, reason?}: append an
                                         `[ORCHESTRATOR]` line to its conversation.

`Task` children run inside the lloyd-mcp process, so the backend has no handle
on them and proxies across the same seam the dashboard's `/state` read and the
Browser tab's URL bar cross (`app.aggregator_config`, URL and credential
together). The request names the session it acts for; the aggregator applies
`_subagent_registry.policy_allows` (`orchestrator-session`) to it, so a button
can stop a child only on behalf of the session that spawned it — the same rule
a tool call from that session meets. The status code is passed through, so a
refusal reads as a refusal and not as a broken proxy.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.aggregator_config import auth_headers_for, subagent_route
from app.aggregator_config import route as aggregator_route

logger = logging.getLogger("lloyd-server")

router = APIRouter()

_TIMEOUT_S = 5.0


async def _forward(task_id: str, verb: str, body: dict) -> JSONResponse:
    url = subagent_route(task_id, verb)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.post(url, json=body, headers=auth_headers_for(url))
    except Exception as e:
        logger.warning("subagents/%s: aggregator unreachable: %s", verb, e)
        return JSONResponse({"error": f"aggregator unreachable: {e}"}, status_code=502)
    try:
        data = r.json()
    except Exception:
        data = {"error": r.text or f"{verb} failed (HTTP {r.status_code})"}
    return JSONResponse(data, status_code=r.status_code)


def _body(payload: dict | None, *keys: str) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    return {k: payload[k] for k in keys if k in payload}


@router.post("/api/subagents/{task_id}/cancel")
async def cancel_subagent(task_id: str, payload: dict | None = None):
    return await _forward(task_id, "cancel", _body(payload, "session_id", "reason"))


@router.post("/api/subagents/{task_id}/steer")
async def steer_subagent(task_id: str, payload: dict | None = None):
    return await _forward(task_id, "steer",
                          _body(payload, "session_id", "text", "reason"))


@router.get("/api/subagents")
async def list_subagents():
    """The running rows (and the last few finished), off the aggregator's `/state`."""
    url = aggregator_route("state")
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(url, headers=auth_headers_for(url))
        r.raise_for_status()
        return JSONResponse((r.json() or {}).get("subagents") or
                            {"active": [], "active_count": 0, "recent": []})
    except Exception as e:
        return JSONResponse({"active": [], "active_count": 0, "recent": [],
                             "error": f"aggregator unreachable: {e}"})
