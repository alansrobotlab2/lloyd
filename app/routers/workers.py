"""Workers router — HTTP surface for the unified work queue.

GET  /api/workers/status           — pool state + queue depth by source
GET  /api/workers/queue            — list queue items (filterable)
GET  /api/workers/runs             — list recent runs (filterable)
POST /api/workers/enqueue          — manually enqueue (for testing / agent hooks)
POST /api/workers/pause            — pause/resume worker draining
POST /api/workers/enable           — toggle workers.enabled in config.yaml
GET  /api/workers/pending          — list pending-research artifacts
GET  /api/workers/pending/read     — read one artifact's full content
POST /api/workers/pending/promote  — move artifact to a canonical vault location
POST /api/workers/pending/reject   — move artifact to pending-research/_rejected/
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import CONFIG, save_tool_overrides
from app.paths import VAULT_ROOT, VAULT_PENDING_RESEARCH_DIR as PENDING_ROOT
from workers.queue import get_queue
from workers.pool import get_pool

REJECTED_ROOT = PENDING_ROOT / "_rejected"

# Default promotion destination per source (relative to vault root).
# Tuned so "just click Promote" does the right thing for the easy cases.
_DEFAULT_DEST: dict[str, str] = {
    "domain-research": "knowledge",
    "bench-mine": "lloyd/bench",
}

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
    """Turn the pool on or off, and make it stick across a restart.

    This used to `yaml.dump(CONFIG)` over `config.yaml`, and each of the three
    things wrong with that was serious on its own. `CONFIG` is the *loaded*
    config, so the dump wrote the expanded values back: `${LIVEKIT_API_SECRET}`
    would have been replaced by the secret itself, in a tracked file. It
    flattened every comment in a 600-line file that is mostly comments. And it
    dirtied the live tree, which `scripts/automod/gate.py` and `promote.py`
    both refuse — so one click here silently stopped the self-modification
    loop until a human committed the damage. That is the identical defect the
    Tools page was moved off `config.yaml` to avoid; this endpoint kept it
    because nothing in the frontend calls it yet.

    So it takes the same route the Tools page does: the UI-mutable slice goes
    to the untracked overrides file, which `app.config` merges at boot.
    """
    data = await request.json() if (await request.body()) else {}
    enabled = bool(data.get("enabled", True))
    CONFIG.setdefault("workers", {})["enabled"] = enabled
    try:
        save_tool_overrides()
    except Exception as e:
        logger.warning("Failed to persist workers.enabled: %s", e)

    # And make it true now, not only after the next restart. A flag that says
    # "off" beside a pool that is still draining the queue is worse than no
    # flag at all.
    pool = get_pool()
    started = False
    if enabled and (pool is None or not pool.status().get("running")):
        await start_worker_pool()
        started = True
    elif not enabled and pool is not None and pool.status().get("running"):
        await pool.stop()

    return JSONResponse({"enabled": enabled, "started": started,
                         "pool": (get_pool().status() if get_pool() else {"running": False})})


# ── Pending-research review surface ──────────────────────────────────────


def _safe_pending_path(path_str: str) -> Path:
    """Resolve a claimed artifact path; 400 unless it's under pending-research/."""
    p = Path(path_str).expanduser().resolve()
    try:
        p.relative_to(PENDING_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="path must be under pending-research/")
    return p


def _safe_vault_dest(rel_or_abs: str) -> Path:
    """Resolve a destination path; 400 unless it stays under the obsidian vault."""
    p = Path(rel_or_abs).expanduser()
    if not p.is_absolute():
        p = VAULT_ROOT / p
    p = p.resolve()
    try:
        p.relative_to(VAULT_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="destination must be under obsidian vault")
    return p


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---\n"):
        return {}, content
    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return {}, content
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, parts[2]


@router.get("/api/workers/pending")
async def workers_pending(source: str = "", limit: int = 200):
    """List pending-research artifacts with frontmatter + short preview."""
    if not PENDING_ROOT.exists():
        return JSONResponse({"items": [], "sources": []})

    items: list[dict] = []
    sources_seen: set[str] = set()
    for src_dir in PENDING_ROOT.iterdir():
        if not src_dir.is_dir() or src_dir.name.startswith("_") or src_dir.name.startswith("."):
            continue
        sources_seen.add(src_dir.name)
        if source and src_dir.name != source:
            continue
        for artifact in src_dir.rglob("*.md"):
            if artifact.name == "README.md":
                continue
            try:
                stat = artifact.stat()
                content = artifact.read_text(encoding="utf-8")
            except OSError:
                continue
            fm, body = _parse_frontmatter(content)
            preview = body.strip().split("\n\n", 1)[0].strip()[:220]
            items.append({
                "path": str(artifact),
                "source": src_dir.name,
                "date": artifact.parent.name,
                "filename": artifact.name,
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "frontmatter": fm,
                "preview": preview,
            })

    items.sort(key=lambda i: i["mtime"], reverse=True)
    return JSONResponse({
        "items": items[: max(1, min(limit, 1000))],
        "sources": sorted(sources_seen),
    })


@router.get("/api/workers/pending/read")
async def workers_pending_read(path: str):
    p = _safe_pending_path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    try:
        content = p.read_text(encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"read failed: {e}")
    fm, body = _parse_frontmatter(content)
    return JSONResponse({
        "path": str(p),
        "source": p.relative_to(PENDING_ROOT).parts[0] if PENDING_ROOT in p.parents else None,
        "frontmatter": fm,
        "body": body,
        "raw": content,
    })


@router.post("/api/workers/pending/promote")
async def workers_pending_promote(request: Request):
    """Move a pending artifact to a canonical vault location.

    Body: { path, destination?, filename? }
      - path: artifact to promote (must be under pending-research/)
      - destination: directory under obsidian vault (absolute or relative).
        Defaults per-source — see _DEFAULT_DEST. Required for sources without
        a default (gap-fill, session-distill).
      - filename: override destination filename (defaults to the artifact's name).
    """
    data = await request.json()
    src = _safe_pending_path(data.get("path", ""))
    if not src.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    src_name = src.relative_to(PENDING_ROOT).parts[0]
    dest_dir_str = data.get("destination") or _DEFAULT_DEST.get(src_name)
    if not dest_dir_str:
        raise HTTPException(
            status_code=400,
            detail=f"no default destination for source '{src_name}' — provide 'destination'",
        )
    dest_dir = _safe_vault_dest(dest_dir_str)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = data.get("filename") or src.name
    dest_path = _safe_vault_dest(str(dest_dir / filename))

    if dest_path.exists():
        raise HTTPException(status_code=409, detail=f"destination exists: {dest_path}")

    # Update frontmatter review_status before move.
    try:
        content = src.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(content)
        if fm:
            fm["review_status"] = "promoted"
            fm["promoted_at"] = datetime.now(timezone.utc).isoformat()
            new_content = (
                "---\n"
                + yaml.dump(fm, default_flow_style=False, allow_unicode=True)
                + "---\n"
                + body
            )
        else:
            new_content = content
        dest_path.write_text(new_content, encoding="utf-8")
        src.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"promote failed: {e}")

    return JSONResponse({
        "promoted": True,
        "from": str(src),
        "to": str(dest_path),
    })


@router.post("/api/workers/pending/reject")
async def workers_pending_reject(request: Request):
    """Move a pending artifact to pending-research/_rejected/ (recoverable)."""
    data = await request.json()
    src = _safe_pending_path(data.get("path", ""))
    if not src.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    rel = src.relative_to(PENDING_ROOT)
    dest = REJECTED_ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = dest.with_name(f"{dest.stem}-{int(datetime.now().timestamp())}{dest.suffix}")
    shutil.move(str(src), str(dest))
    return JSONResponse({"rejected": True, "from": str(src), "to": str(dest)})


# ── Startup hook — registered in server.py ───────────────────────────────


async def start_worker_pool() -> None:
    """Initialize the queue, register sources (via import), and start the pool."""
    cfg = CONFIG.get("workers", {}) or {}
    if not cfg.get("enabled", False):
        logger.info("Worker pool disabled in config — not starting")
        return

    from workers.queue import configured_db_path, get_queue

    db_path = configured_db_path()
    # Same default as `WorkerPool.__init__`. They disagreed (8 here, 4 there),
    # so "what happens with no `workers.slots` in config" had two answers
    # depending on which one you read.
    slots = int(cfg.get("slots", 4))
    max_attempts = int(cfg.get("max_attempts", 3))

    q = get_queue(db_path)

    # Importing sources registers them into SOURCE_REGISTRY.
    import workers.sources  # noqa: F401

    from workers.pool import start_pool
    await start_pool(q, slots=slots, max_attempts=max_attempts)
    logger.info("Worker pool startup complete (slots=%d, db=%s)", slots, db_path)
