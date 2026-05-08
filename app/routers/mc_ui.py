"""Mission Control UI state mirror — HTTP + SSE endpoints.

GET    /api/mc/state     — read the mirrored {tab, focus, ...}.
POST   /api/mc/state     — frontend reports its current {tab, focus}.
POST   /api/mc/navigate  — agent moves the user's UI; returns brief detail.
GET    /api/mc/events    — SSE channel: navigate commands fan-out to clients.

Frontend hooks call the first two; the agent's mc_get_state / mc_navigate
MCP tools call /state and /navigate over the local loopback. The Layout
component subscribes to /events and reflects navigate commands.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import mc_state
from app.paths import LLOYD_HOME, SESSIONS_DIR

logger = logging.getLogger("lloyd-server")

router = APIRouter()


# ── State endpoints ────────────────────────────────────────────────────

@router.get("/api/mc/state")
async def get_mc_state():
    snap = await mc_state.get_state()
    return JSONResponse(snap)


@router.post("/api/mc/state")
async def post_mc_state(request: Request):
    """Frontend reports its current tab + focus.

    Body: {tab?: string, focus?: {kind, id, label?} | null}
    A null/missing focus clears the active tab's focus entry.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")

    tab = body.get("tab")
    focus = body.get("focus")
    try:
        snap = await mc_state.set_state(tab if isinstance(tab, str) else None, focus)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(snap)


# ── Navigate (agent → frontend) ────────────────────────────────────────

@router.post("/api/mc/navigate")
async def post_mc_navigate(request: Request):
    """Push a navigate command to all subscribed frontends.

    Body: {tab: string, focus_id?: string|null}
    Returns: {tab, focus_id, detail} where detail is a per-tab brief
    summary computed server-side.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")

    tab = body.get("tab")
    if not isinstance(tab, str) or tab not in mc_state.VALID_TABS:
        raise HTTPException(
            status_code=400,
            detail=f"tab must be one of: {sorted(mc_state.VALID_TABS)}",
        )
    focus_id = body.get("focus_id")
    if focus_id is not None and not isinstance(focus_id, str):
        raise HTTPException(status_code=400, detail="focus_id must be a string or null")

    # Pre-validate path-shaped focus on tabs that load files. Surfacing the
    # 404 here turns a silent broken navigation into a clear tool error the
    # caller can correct (e.g. by retrying with vault_search results).
    if tab == "memory" and focus_id and ("/" in focus_id or focus_id.endswith(".md")):
        vault_path = (Path.home() / "obsidian" / focus_id).resolve()
        vault_root = (Path.home() / "obsidian").resolve()
        if not str(vault_path).startswith(str(vault_root) + "/"):
            raise HTTPException(status_code=400, detail=f"path escapes vault: {focus_id}")
        if not vault_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"vault path does not exist: {focus_id}. "
                    "Use vault_search to find the correct path."
                ),
            )

    try:
        await mc_state.publish_navigate(tab, focus_id)
    except Exception as e:
        logger.warning("mc/navigate publish failed: %s", e)

    detail = await _summarize_tab(tab)
    return JSONResponse({"tab": tab, "focus_id": focus_id, "detail": detail})


# ── SSE event bus (backend → frontend) ─────────────────────────────────

async def _navigate_sse(request: Request):
    """Yield SSE events from the navigate fan-out queue."""
    q = mc_state.subscribe()
    try:
        # Initial hello so the client knows the channel is live.
        yield f"event: hello\ndata: {json.dumps({'ok': True})}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                evt = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Heartbeat keeps the connection from being reaped by
                # intermediaries that close idle SSE sockets.
                yield ": ping\n\n"
                continue
            yield f"event: navigate\ndata: {json.dumps(evt)}\n\n"
    except asyncio.CancelledError:
        raise
    finally:
        mc_state.unsubscribe(q)


@router.get("/api/mc/events")
async def get_mc_events(request: Request):
    return StreamingResponse(_navigate_sse(request), media_type="text/event-stream")


# ── Per-tab brief summarizers ──────────────────────────────────────────
#
# Each summarizer returns a small dict — counts + a few identifiers —
# composed from the same on-disk sources the corresponding tab's GET
# endpoint already reads. Failures are swallowed into {} so a broken
# fixture in one tab can't crash the navigate response.

_VAULT = Path.home() / "obsidian"
_VAULT_SEGMENTS = ("memory", "knowledge", "projects", "agents", "personal", "work", "skills")


def _safe(coro_or_value):
    """Run a sync callable safely, returning {} on failure."""
    try:
        return coro_or_value() if callable(coro_or_value) else coro_or_value
    except Exception as e:
        logger.debug("mc_ui summarizer failed: %s", e)
        return {}


def _summarize_inner_voice() -> dict:
    snap = {}
    try:
        from app import event_log as _event_log  # late import — avoids cycles
    except Exception:
        return snap
    # Lightweight: just count sessions with IV enabled.
    iv_sessions = 0
    total = 0
    for sf in SESSIONS_DIR.glob("*.json"):
        total += 1
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            if data.get("inner_voice"):
                iv_sessions += 1
        except Exception:
            continue
    return {"sessions_total": total, "iv_enabled_count": iv_sessions}


def _summarize_chat() -> dict:
    sessions = []
    for sf in sorted(SESSIONS_DIR.glob("*.json"),
                      key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            if data.get("platform") == "autonomy":
                continue
            sessions.append({
                "id": data.get("session_id", sf.stem),
                "preview": (data.get("preview") or "")[:120],
                "model": data.get("model", ""),
            })
        except Exception:
            continue
        if len(sessions) >= 5:
            break
    return {
        "total_session_count": sum(
            1 for sf in SESSIONS_DIR.glob("*.json")
        ),
        "recent_sessions": sessions,
    }


_BACKLOG_DIR = Path.home() / "obsidian" / "backlog"
_AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"


def _summarize_backlog() -> dict:
    if not _BACKLOG_DIR.exists():
        return {}
    by_status: dict[str, int] = {}
    by_board: dict[str, int] = {}
    total = 0
    for tf in _BACKLOG_DIR.glob("*.md"):
        try:
            head = tf.read_text(encoding="utf-8")[:2000]
            if not head.startswith("---"):
                continue
            parts = head.split("---", 2)
            if len(parts) < 3:
                continue
            fm = yaml.safe_load(parts[1]) or {}
            total += 1
            status = str(fm.get("status") or "draft")
            by_status[status] = by_status.get(status, 0) + 1
            board = fm.get("board")
            if board is not None:
                by_board[str(board)] = by_board.get(str(board), 0) + 1
        except Exception:
            continue
    return {"total": total, "by_status": by_status, "by_board": by_board}


def _summarize_autonomy() -> dict:
    if not _AUTONOMY_DIR.exists():
        return {}
    statuses: dict[str, int] = {}
    total = 0
    for tf in _AUTONOMY_DIR.glob("*.md"):
        if tf.name == "_config.md":
            continue
        try:
            head = tf.read_text(encoding="utf-8")[:2000]
            if not head.startswith("---"):
                continue
            parts = head.split("---", 2)
            if len(parts) < 3:
                continue
            fm = yaml.safe_load(parts[1]) or {}
            total += 1
            status = str(fm.get("status") or "scheduled")
            statuses[status] = statuses.get(status, 0) + 1
        except Exception:
            continue
    return {"total": total, "by_status": statuses}


def _summarize_workers() -> dict:
    try:
        from workers.queue import get_queue
        q = get_queue()
    except Exception:
        return {}
    depth = q.depth_by_source() if hasattr(q, "depth_by_source") else {}
    pending_by_source: dict[str, int] = {}
    pending_total = 0
    for src, by_state in (depth or {}).items():
        if not isinstance(by_state, dict):
            continue
        n = sum(int(v) for k, v in by_state.items() if k in ("pending", "ready", "queued"))
        if n:
            pending_by_source[src] = n
            pending_total += n
    return {"pending_total": pending_total, "pending_by_source": pending_by_source}


def _summarize_memory() -> dict:
    if not _VAULT.exists():
        return {"vault_file_count": 0}
    count = 0
    for seg in _VAULT_SEGMENTS:
        seg_dir = _VAULT / seg
        if seg_dir.is_dir():
            count += sum(1 for _ in seg_dir.rglob("*.md"))
    return {"vault_file_count": count}


def _summarize_tools() -> dict:
    try:
        from app.config import CONFIG
    except Exception:
        return {}
    servers = CONFIG.get("mcp_servers", {})
    enabled_servers = sum(1 for v in servers.values() if v.get("enabled", True))
    return {
        "server_count": len(servers),
        "enabled_server_count": enabled_servers,
        "servers": [
            {
                "id": name,
                "enabled": cfg.get("enabled", True),
                "disabled_tool_count": len(cfg.get("disabled_tools") or []),
            }
            for name, cfg in servers.items()
        ],
    }


def _summarize_skills() -> dict:
    try:
        from app.config import CONFIG
    except Exception:
        return {}
    total = 0
    for dir_path in CONFIG.get("skills", {}).get("directories", []):
        expanded = Path(dir_path.replace("~", str(Path.home())))
        if not expanded.exists():
            continue
        for entry in expanded.iterdir():
            if entry.is_dir() and (entry / "SKILL.md").exists():
                total += 1
    return {"skill_count": total}


def _summarize_services() -> dict:
    try:
        from app.supervisor_client import (
            _supervisor_all,
            _INFRA_SERVICES,
            _LLOYD_SERVICES,
            _sup_state,
        )
    except Exception:
        return {}
    procs = _supervisor_all()
    counts = {"running": 0, "stopped": 0, "other": 0}
    for sid in (*_INFRA_SERVICES.keys(), *_LLOYD_SERVICES.keys()):
        proc = procs.get(sid, {})
        active, _sub = _sup_state(proc)
        if active == "active":
            counts["running"] += 1
        elif active == "inactive":
            counts["stopped"] += 1
        else:
            counts["other"] += 1
    return {"counts": counts}


_SUMMARIZERS = {
    "inner_voice": _summarize_inner_voice,
    "chat": _summarize_chat,
    "backlog": _summarize_backlog,
    "autonomy": _summarize_autonomy,
    "workers": _summarize_workers,
    "memory": _summarize_memory,
    "tools": _summarize_tools,
    "skills": _summarize_skills,
    "services": _summarize_services,
    "architecture": lambda: {},
    "settings": lambda: {},
    "graph": lambda: {},
}


async def _summarize_tab(tab: str) -> dict:
    fn = _SUMMARIZERS.get(tab)
    if fn is None:
        return {}
    return _safe(fn)
