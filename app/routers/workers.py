"""Workers router — HTTP surface for the unified work queue.

GET  /api/workers/status     — pool state + queue depth by source
GET  /api/workers/queue      — list queue items (filterable)
GET  /api/workers/runs       — list recent runs (filterable)
POST /api/workers/enqueue    — manually enqueue (for testing / agent hooks)
POST /api/workers/pause      — pause/resume worker draining
POST /api/workers/enable     — toggle workers.enabled in config.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import CONFIG
from app.paths import LLOYD_HOME
from workers.queue import get_queue
from workers.pool import get_pool

router = APIRouter()
logger = logging.getLogger("lloyd-server")


@router.get("/api/workers/status")
async def workers_status():
    try:
        q = get_queue()
    except RuntimeError:
        return JSONResponse({"initialized": False})
    pool = get_pool()
    depth = q.depth_by_source()
    sources_cfg = CONFIG.get("workers", {}).get("sources", {}) or {}

    # Compose per-source health row.
    sources = []
    for name, src_cfg in sources_cfg.items():
        sources.append({
            "name": name,
            "enabled": bool(src_cfg.get("enabled", False)),
            "interval_seconds": src_cfg.get("interval_seconds"),
            "max_inflight": src_cfg.get("max_inflight"),
            "depth": depth.get(name, {}),
        })

    return JSONResponse({
        "initialized": True,
        "workers_enabled": bool(CONFIG.get("workers", {}).get("enabled", False)),
        "pool": pool.status() if pool else {"running": False},
        "depth": depth,
        "sources": sources,
    })


@router.get("/api/workers/queue")
async def workers_queue(state: str = "", source: str = "", limit: int = 100):
    try:
        q = get_queue()
    except RuntimeError:
        return JSONResponse({"items": []})
    items = q.list_items(
        state=state or None,
        source=source or None,
        limit=min(max(1, limit), 500),
    )
    return JSONResponse({"items": [i.to_dict() for i in items]})


@router.get("/api/workers/runs")
async def workers_runs(source: str = "", task_id: str = "", limit: int = 50):
    try:
        q = get_queue()
    except RuntimeError:
        return JSONResponse({"runs": []})
    runs = q.list_runs(
        source=source or None,
        task_id=task_id or None,
        limit=min(max(1, limit), 500),
    )
    return JSONResponse({"runs": runs})


@router.post("/api/workers/enqueue")
async def workers_enqueue(request: Request):
    try:
        q = get_queue()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="queue not initialized")
    data = await request.json()
    source = data.get("source")
    kind = data.get("kind", "manual")
    if not source:
        raise HTTPException(status_code=400, detail="source required")

    new_id = q.enqueue(
        source=source,
        kind=kind,
        payload=data.get("payload") or {},
        priority=int(data.get("priority", 50)),
        dedup_key=data.get("dedup_key"),
    )
    if new_id is None:
        return JSONResponse({"coalesced": True})
    return JSONResponse({"id": new_id})


@router.post("/api/workers/pause")
async def workers_pause(request: Request):
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="pool not running")
    data = await request.json() if (await request.body()) else {}
    paused = bool(data.get("paused", True))
    pool.pause(paused)
    return JSONResponse({"paused": pool.paused})


@router.post("/api/workers/enable")
async def workers_enable(request: Request):
    data = await request.json() if (await request.body()) else {}
    enabled = bool(data.get("enabled", True))
    CONFIG.setdefault("workers", {})["enabled"] = enabled
    try:
        cfg_path = LLOYD_HOME / "config.yaml"
        cfg_path.write_text(yaml.dump(CONFIG, default_flow_style=False, allow_unicode=True))
    except Exception as e:
        logger.warning("Failed to persist workers.enabled: %s", e)
    return JSONResponse({"enabled": enabled})


# ── Startup hook — registered in server.py ───────────────────────────────


async def start_worker_pool() -> None:
    """Initialize the queue, register sources (via import), and start the pool."""
    cfg = CONFIG.get("workers", {}) or {}
    if not cfg.get("enabled", False):
        logger.info("Worker pool disabled in config — not starting")
        return

    db_path = Path(cfg.get("db_path", "~/lloyd/workers.db")).expanduser()
    slots = int(cfg.get("slots", 8))
    max_attempts = int(cfg.get("max_attempts", 3))

    # Instantiate queue singleton with this db path.
    from workers.queue import get_queue, _queue_instance  # noqa: F401
    try:
        q = get_queue()
    except RuntimeError:
        from workers.queue import WorkQueue
        import workers.queue as _q
        _q._queue_instance = WorkQueue(db_path)
        q = get_queue()

    # Importing sources registers them into SOURCE_REGISTRY.
    import workers.sources  # noqa: F401

    from workers.pool import start_pool
    await start_pool(q, slots=slots, max_attempts=max_attempts)
    logger.info("Worker pool startup complete (slots=%d, db=%s)", slots, db_path)
