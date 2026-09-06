"""Dashboard endpoint — one aggregated snapshot of system + agent state.

`GET /api/dashboard` returns everything the Mission Control dashboard
renders. Deliberately one endpoint rather than eight: the page polls on a
short interval, and eight concurrent requests per tick (times however
many browser tabs are open) is a lot of load to put on a box whose whole
job is to hold a 262k-token KV cache steady.

Every section is gathered independently and degrades on its own. A
wedged supervisord or an unreachable MCP aggregator turns one panel into
an error string; it never blanks the page. That is the point of a
dashboard — it is most useful exactly when something is broken, so it
must not be the second thing to break.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import host_metrics, sessions_io, vllm_metrics

logger = logging.getLogger("lloyd-server")

router = APIRouter()

# The aggregator's loopback state route. Same host, same machine — if
# this is slow the aggregator is wedged, and a short timeout keeps that
# from becoming the dashboard's problem too.
_MCP_STATE_URL = "http://127.0.0.1:8500/state"
_MCP_TIMEOUT_S = 2.0


async def _gather(name: str, coro) -> tuple[str, Any]:
    """Await one section, converting any failure into an `error` payload."""
    try:
        return name, await coro
    except Exception as exc:
        logger.warning("dashboard section %r failed: %s", name, exc)
        return name, {"error": f"{type(exc).__name__}: {exc}"}


async def _to_thread(fn, *args) -> Any:
    """Run a blocking call off the event loop."""
    return await asyncio.to_thread(fn, *args)


# ── Sections ───────────────────────────────────────────────────────────


async def _agent_state() -> dict[str, Any]:
    """Subagents + background bash tasks, read from the MCP aggregator.

    Both live in the lloyd-mcp process, not this one — see the docstring
    on `agent_mcp.main.state`.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get(_MCP_STATE_URL, timeout=_MCP_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()


async def _primary_state() -> dict[str, Any]:
    """What the primary agent loop is doing right now."""
    from app.config import CONFIG

    active = sessions_io.active_sessions_snapshot()
    models = CONFIG.get("models") or {}
    default_alias = (CONFIG.get("model") or {}).get("default", "primary")
    default_cfg = models.get(default_alias) or {}

    running = [s for s in active if s["running"]]
    return {
        "model": default_alias,
        "base_url": default_cfg.get("base_url", ""),
        "context_length": default_cfg.get("context_length"),
        "max_turns": (CONFIG.get("agent") or {}).get("max_turns"),
        "permission_mode": (CONFIG.get("agent") or {}).get("permission_mode", ""),
        "preserve_thinking_iterations": (CONFIG.get("harness") or {}).get(
            "preserve_thinking_iterations"
        ),
        "sessions": active,
        "running_count": len(running),
        "queued_count": sum(
            s["pending_user"] + s["pending_ambient"] for s in active
        ),
        "busy": bool(running),
    }


def _focus_session() -> dict[str, Any]:
    """Goal / plan / todos for the session the operator is looking at.

    Prefers a session with a turn actually running; falls back to the
    tab focus mirrored in `mc_state`. These live in the session JSON
    (written by the goal/plan/todo built-ins), so this is a disk read,
    hence the thread hop at the call site.
    """
    import json

    from app import mc_state
    from app.paths import SESSIONS_DIR

    running = [s for s in sessions_io.active_sessions_snapshot() if s["running"]]
    session_id = running[0]["session_id"] if running else ""

    if not session_id:
        focus = mc_state.get_focus_snapshot().get("focus_by_tab", {})
        for tab in ("chat", "inner_voice"):
            entry = focus.get(tab) or {}
            if entry.get("kind") == "session" and entry.get("id"):
                session_id = entry["id"]
                break
    if not session_id:
        return {}

    path = SESSIONS_DIR / f"{session_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"session_id": session_id}

    todos = data.get("todos") or []
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}
    goal = data.get("goal") if isinstance(data.get("goal"), dict) else {}
    return {
        "session_id": session_id,
        "preview": (data.get("preview") or "")[:160],
        "message_count": data.get("message_count"),
        "inner_voice": bool(data.get("inner_voice")),
        "platform": data.get("platform", ""),
        "goal": goal.get("text", ""),
        "goal_set_at": goal.get("set_at"),
        "goal_achieved": bool(goal.get("achieved_at")),
        "plan_mode": bool(plan.get("plan_mode")),
        "plan_stages": len(plan.get("stages") or []),
        "todos": [
            {
                "content": t.get("content", ""),
                "status": t.get("status", ""),
                "activeForm": t.get("activeForm", ""),
            }
            for t in todos
            if isinstance(t, dict)
        ],
        "todo_counts": _count_todos(todos),
    }


def _count_todos(todos: list) -> dict[str, int]:
    counts = {"pending": 0, "in_progress": 0, "completed": 0}
    for t in todos:
        if isinstance(t, dict):
            status = t.get("status", "")
            if status in counts:
                counts[status] += 1
    return counts


def _services() -> dict[str, Any]:
    """supervisord health for infra + lloyd processes."""
    from app.supervisor_client import (
        _INFRA_SERVICES,
        _LLOYD_SERVICES,
        _health,
        _port_open,
        _sup_state,
        _supervisor_all,
    )

    procs = _supervisor_all()
    out: list[dict[str, Any]] = []
    for group, table in (("infra", _INFRA_SERVICES), ("lloyd", _LLOYD_SERVICES)):
        for sid, (name, port) in table.items():
            proc = procs.get(sid)
            active, sub = _sup_state(proc)
            port_healthy = _port_open(port) if port else None
            out.append({
                "id": sid,
                "name": name,
                "group": group,
                "port": port,
                "state": active,
                "sub_state": sub,
                "port_healthy": port_healthy,
                "health": _health(active, port_healthy),
                "uptime": (proc or {}).get("description"),
            })
    unhealthy = [s["id"] for s in out if s["health"] != "healthy"]
    return {"services": out, "unhealthy": unhealthy, "total": len(out)}


def _workers() -> dict[str, Any]:
    """Worker pool + queue depth."""
    from workers.queue import get_queue

    q = get_queue()
    depth = q.depth_by_source() if hasattr(q, "depth_by_source") else {}
    by_state: dict[str, int] = {}
    for _src, states in (depth or {}).items():
        if isinstance(states, dict):
            for state, n in states.items():
                by_state[state] = by_state.get(state, 0) + int(n)
    return {"depth_by_source": depth, "by_state": by_state}


def _usage() -> dict[str, Any]:
    """Local token accounting from usage.db.

    This is Lloyd's own record of what it spent against the local
    engines — distinct from the vLLM counters, which are per-engine and
    reset on restart.
    """
    import usage_store

    return {
        "last_hour": usage_store.summary(hours=1),
        "last_24h": usage_store.summary(hours=24),
        "last_7d": usage_store.summary(days=7),
        "daily": usage_store.history_daily(days=7),
        "by_model_24h": usage_store.model_breakdown(hours=24),
    }


# ── Endpoint ───────────────────────────────────────────────────────────


@router.get("/api/dashboard")
async def get_dashboard():
    """One snapshot: host, engines, primary agent, subagents, services."""
    sections = await asyncio.gather(
        _gather("host", host_metrics.collect()),
        _gather("vllm", vllm_metrics.collect(vllm_metrics.configured_engines())),
        _gather("primary", _primary_state()),
        _gather("focus", _to_thread(_focus_session)),
        _gather("agents", _agent_state()),
        _gather("services", _to_thread(_services)),
        _gather("workers", _to_thread(_workers)),
        _gather("usage", _to_thread(_usage)),
    )
    payload: dict[str, Any] = dict(sections)
    payload["timestamp"] = time.time()
    return JSONResponse(payload)
