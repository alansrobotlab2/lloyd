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


# Sections that walk the vault are cached: the backlog is 300+ markdown
# files and its status counts do not change between 2-second polls.
# Live sections (engines, host, running turns) are never cached — they
# are the whole point of the page.
_VAULT_SCAN_TTL_S = 10.0
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fn) -> Any:
    hit = _cache.get(key)
    now = time.monotonic()
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (now, value)
    return value


def _frontmatter(path, limit: int = 3000) -> dict:
    """First YAML block of a markdown file, or {} if there isn't one."""
    import yaml

    try:
        head = path.read_text(encoding="utf-8")[:limit]
    except OSError:
        return {}
    if not head.startswith("---"):
        return {}
    parts = head.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}


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


# Queue states that mean "not finished". `completed` dominates the depth
# table (3,400+ rows on this box) and would drown the live numbers.
_OPEN_STATES = ("queued", "claimed", "running", "pending", "ready")


def _workers() -> dict[str, Any]:
    """Worker pool, per-source depth, and what just ran."""
    from app.config import CONFIG
    from workers.pool import get_pool
    from workers.queue import get_queue

    q = get_queue()
    depth = q.depth_by_source() if hasattr(q, "depth_by_source") else {}

    by_state: dict[str, int] = {}
    for _src, states in (depth or {}).items():
        if isinstance(states, dict):
            for state, n in states.items():
                by_state[state] = by_state.get(state, 0) + int(n)

    pool = None
    try:
        p = get_pool()
        pool = p.status() if p else {"running": False}
    except Exception:
        pool = {"running": False}

    # One row per configured source, with its open (unfinished) backlog
    # split out from the lifetime completed count.
    sources_cfg = (CONFIG.get("workers") or {}).get("sources") or {}
    sources = []
    for name, cfg in sources_cfg.items():
        d = depth.get(name, {}) or {}
        sources.append({
            "name": name,
            "enabled": bool((cfg or {}).get("enabled", False)),
            "open": sum(int(d.get(st, 0)) for st in _OPEN_STATES),
            "running": int(d.get("running", 0)),
            "completed": int(d.get("completed", 0)),
            "failed": int(d.get("failed", 0)),
            "poisoned": int(d.get("poisoned", 0)),
        })
    sources.sort(key=lambda r: (-r["running"], -r["open"], r["name"]))

    runs = []
    try:
        for r in (q.list_runs(limit=6) or []):
            runs.append({
                "run_id": r.get("run_id", ""),
                "source": r.get("source", ""),
                "status": r.get("status", ""),
                "started_at": r.get("started_at", ""),
                "duration_seconds": r.get("duration_seconds"),
                "summary": (r.get("summary") or "")[:120],
            })
    except Exception:
        pass

    return {
        "enabled": bool((CONFIG.get("workers") or {}).get("enabled", False)),
        "pool": pool,
        "depth_by_source": depth,
        "by_state": by_state,
        "open_total": sum(by_state.get(st, 0) for st in _OPEN_STATES),
        "poisoned_total": by_state.get("poisoned", 0),
        "sources": sources,
        "recent_runs": runs,
    }


def _autonomy() -> dict[str, Any]:
    """Scheduled-task fleet: what is due, what is running, what is broken.

    Task definitions are markdown under ~/obsidian/autonomy/, so the file
    walk is TTL-cached; the in-flight list comes from the live pool and
    is not.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    autonomy_dir = Path.home() / "obsidian" / "autonomy"

    def _scan() -> dict[str, Any]:
        if not autonomy_dir.exists():
            return {"total": 0, "by_status": {}, "upcoming": [], "failing": []}
        by_status: dict[str, int] = {}
        upcoming: list[dict[str, Any]] = []
        failing: list[dict[str, Any]] = []
        total = 0
        for path in autonomy_dir.glob("*.md"):
            # Only NN-name.md task files — skip _config.md, reports, notes.
            if not path.name[:1].isdigit():
                continue
            fm = _frontmatter(path)
            if not fm:
                continue
            total += 1
            status = str(fm.get("status") or "draft")
            by_status[status] = by_status.get(status, 0) + 1
            row = {
                "name": str(fm.get("name") or path.stem),
                "status": status,
                "frequency": str(fm.get("frequency") or ""),
                "next_run": _iso(fm.get("next_run")),
                "last_run": _iso(fm.get("last_run")),
            }
            if status == "failed":
                failing.append(row)
            elif row["next_run"]:
                upcoming.append(row)

        # Split overdue from genuinely-upcoming. Sorting them together and
        # calling the result "next up" is how 23 tasks whose next_run is
        # months in the past read as a healthy schedule: the soonest-first
        # sort puts the most overdue at the top, labelled as if it were
        # the next thing to run.
        now_iso = datetime.now(timezone.utc).isoformat()
        overdue = [r for r in upcoming if (r["next_run"] or "") < now_iso]
        pending = [r for r in upcoming if (r["next_run"] or "") >= now_iso]
        overdue.sort(key=lambda r: r["next_run"] or "")        # worst first
        pending.sort(key=lambda r: r["next_run"] or "")        # soonest first
        return {
            "total": total,
            "by_status": by_status,
            "overdue": overdue[:5],
            "overdue_count": len(overdue),
            "upcoming": pending[:5],
            "failing": failing[:5],
        }

    out = _cached("autonomy", _VAULT_SCAN_TTL_S, _scan)

    # Live: which scheduled tasks the pool is actually executing now.
    running: list[dict[str, Any]] = []
    try:
        from workers.pool import get_pool

        pool = get_pool()
        status = pool.status() if pool else {}
        now = datetime.now(timezone.utc)
        for job_id, meta in (status.get("in_flight") or {}).items():
            if (meta or {}).get("source") != "scheduled-task":
                continue
            started = _iso(meta.get("started_at"))
            elapsed = None
            if started:
                try:
                    dt = datetime.fromisoformat(started)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    elapsed = (now - dt).total_seconds()
                except ValueError:
                    pass
            running.append({
                "job_id": str(job_id),
                "kind": (meta or {}).get("kind", ""),
                "started_at": started,
                "elapsed_s": elapsed,
            })
    except Exception:
        pass

    return {**out, "running": running, "running_count": len(running)}


def _backlog() -> dict[str, Any]:
    """Backlog board: open work by status and by board."""
    from pathlib import Path

    backlog_dir = Path.home() / "obsidian" / "backlog"

    def _scan() -> dict[str, Any]:
        if not backlog_dir.exists():
            return {"total": 0, "by_status": {}, "by_board": [], "open_total": 0}
        by_status: dict[str, int] = {}
        # board -> {open, total}
        boards: dict[str, dict[str, int]] = {}
        total = 0
        recent: list[dict[str, Any]] = []
        for path in backlog_dir.glob("*.md"):
            if not path.name[:1].isdigit():
                continue
            fm = _frontmatter(path)
            if not fm:
                continue
            total += 1
            status = str(fm.get("status") or "draft")
            board = str(fm.get("board") or "default")
            by_status[status] = by_status.get(status, 0) + 1
            entry = boards.setdefault(board, {"open": 0, "total": 0})
            entry["total"] += 1
            if status not in _BACKLOG_CLOSED:
                entry["open"] += 1
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                recent.append({
                    "name": str(fm.get("name") or path.stem),
                    "status": status,
                    "board": board,
                    "mtime": mtime,
                })
        recent.sort(key=lambda r: r["mtime"], reverse=True)
        return {
            "total": total,
            "by_status": by_status,
            "by_board": sorted(
                ({"board": b, **v} for b, v in boards.items()),
                key=lambda r: -r["open"],
            ),
            "open_total": sum(
                n for st, n in by_status.items() if st not in _BACKLOG_CLOSED
            ),
            "recent_open": recent[:5],
        }

    return _cached("backlog", _VAULT_SCAN_TTL_S, _scan)


# Statuses that take a task off the board.
_BACKLOG_CLOSED = frozenset({"done", "closed", "cancelled", "wontfix"})


def _iso(value: Any) -> str:
    """Frontmatter dates arrive as str or datetime depending on the writer."""
    from datetime import date, datetime

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value) if value else ""


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
        _gather("autonomy", _to_thread(_autonomy)),
        _gather("backlog", _to_thread(_backlog)),
        _gather("usage", _to_thread(_usage)),
    )
    payload: dict[str, Any] = dict(sections)
    payload["timestamp"] = time.time()
    return JSONResponse(payload)
