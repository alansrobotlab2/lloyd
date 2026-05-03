"""Autonomy endpoints (status/enable/disable/run + file-backed task CRUD)
plus the background scheduler ticker registered at app startup.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.discord_notify import _discord_notify_task_complete


router = APIRouter()
logger = logging.getLogger("lloyd-server")

_AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
_AUTONOMY_RUNS_DIR = Path.home() / "lloyd" / "autonomy-runs"


# ── Runtime control ──────────────────────────────────────────────────────────


@router.post("/api/autonomy/run")
async def autonomy_run(request: Request):
    try:
        from autonomy import run_task
        data = await request.json()
        task_id = data.get("task_id")
        if not task_id:
            raise HTTPException(status_code=400, detail="task_id required")
        result = await run_task(int(task_id))
        return JSONResponse(result)
    except ImportError:
        raise HTTPException(status_code=501, detail="Autonomy module not available")


# ── File-backed task CRUD ────────────────────────────────────────────────────


def _autonomy_parse(path: Path) -> dict | None:
    """Parse an autonomy task markdown file into a normalized dict."""
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        fm = yaml.safe_load(parts[1])
        if not isinstance(fm, dict):
            return None

        def _to_iso(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val.strftime("%Y-%m-%dT%H:%M:%SZ")
            return str(val) if val else None

        raw_id = fm.get("id", 0)
        try:
            id_val = int(raw_id)
        except (ValueError, TypeError):
            id_val = 0
        return {
            "id": id_val,
            "name": fm.get("name", ""),
            "description": fm.get("description", ""),
            "status": fm.get("status", "draft"),
            "priority": fm.get("priority", "medium"),
            "frequency": fm.get("frequency") or None,
            "scheduled_at": _to_iso(fm.get("scheduled_at")),
            "last_run": _to_iso(fm.get("last_run")),
            "next_run": _to_iso(fm.get("next_run")),
            "auto_advance": bool(fm.get("auto_advance", False)),
            "preemptible": bool(fm.get("preemptible", True)),
            "pipeline_mode": bool(fm.get("pipeline_mode", False)),
            "notify_on_complete": bool(fm.get("notify_on_complete", True)),
            "tags": fm.get("tags", []) or [],
            "created_at": _to_iso(fm.get("created", fm.get("created_at"))) or "",
            "updated_at": _to_iso(fm.get("updated", fm.get("updated_at"))) or "",
            "runs_per_day": fm.get("runs_per_day"),
            "depends_on": fm.get("depends_on"),
            "pipeline": fm.get("pipeline"),
            "agent_id": fm.get("agent_id") or None,
            "skill_name": fm.get("skill_name", fm.get("skill_path")) or None,
            "model": fm.get("model") or None,
            "timeout_seconds": fm.get("timeout_seconds", 1800),
            "max_retries": fm.get("max_retries", 3),
            "preferred_hours": fm.get("preferred_hours") or None,
            "cron_id": fm.get("cron_id"),
            "body": parts[2] if len(parts) > 2 else "",
        }
    except Exception:
        return None


def _autonomy_find_file(task_id: int) -> Path | None:
    if not _AUTONOMY_DIR.exists():
        return None
    matches = [p for p in _AUTONOMY_DIR.glob(f"{task_id}-*.md") if p.name != "_config.md"]
    return matches[0] if matches else None


def _autonomy_next_id() -> int:
    if not _AUTONOMY_DIR.exists():
        return 1
    max_id = 0
    for p in _AUTONOMY_DIR.glob("*.md"):
        if p.name == "_config.md":
            continue
        parts = p.name.split("-", 1)
        if parts[0].isdigit():
            max_id = max(max_id, int(parts[0]))
    return max_id + 1


def _autonomy_slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]


def _autonomy_write_file(task_dict: dict) -> Path:
    """Write a task dict back to its markdown file."""
    task_id = task_dict.get("id", 0)
    name = task_dict.get("name", "unnamed")
    _AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    existing = _autonomy_find_file(task_id)
    path = existing if existing else _AUTONOMY_DIR / f"{task_id}-{_autonomy_slugify(name)}.md"
    fm = {}
    for key in ("type", "id", "name", "description", "status", "priority", "frequency",
                 "agent_id", "model", "tags", "auto_advance", "preemptible", "pipeline_mode",
                 "timeout_seconds", "max_retries", "failure_count", "skill_name", "cron_id",
                 "runs_per_day", "scheduled_at", "last_run", "next_run", "depends_on",
                 "preferred_hours", "notify_on_complete", "pipeline", "created", "updated"):
        if key in task_dict and task_dict[key] is not None:
            fm[key] = task_dict[key]
    if "type" not in fm:
        fm["type"] = "autonomy"
    body = task_dict.get("body", "")
    content = f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n\n{body}"
    path.write_text(content, encoding="utf-8")
    return path


@router.get("/api/autonomy/tasks")
async def autonomy_tasks(status: str = "", tag: str = ""):
    """List autonomy tasks from ~/obsidian/autonomy/."""
    if not _AUTONOMY_DIR.exists():
        return JSONResponse({"tasks": []})
    tasks = []
    for path in _AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        task = _autonomy_parse(path)
        if task is None:
            continue
        if status and task.get("status") != status:
            continue
        if tag and tag not in (task.get("tags") or []):
            continue
        tasks.append(task)
    return JSONResponse({"tasks": tasks})


@router.post("/api/autonomy/task-write")
async def autonomy_task_write(request: Request):
    """Create or update an autonomy task."""
    data = await request.json()
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    task_id = data.get("id", 0)
    if not task_id:
        name = data.get("name", "")
        if not name:
            raise HTTPException(status_code=400, detail="name required for new task")
        new_id = _autonomy_next_id()
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        task_dict = {
            "type": "autonomy",
            "id": new_id,
            "name": name,
            "description": data.get("description", ""),
            "status": data.get("status") or "draft",
            "priority": data.get("priority") or "medium",
            "frequency": data.get("frequency", ""),
            "skill_name": data.get("skill_name", data.get("skill_path", "")),
            "agent_id": data.get("agent_id") or "memory",
            "model": data.get("model", ""),
            "timeout_seconds": data.get("timeout_seconds") or 1800,
            "tags": tags,
            "auto_advance": data.get("auto_advance", False),
            "preemptible": data.get("preemptible", True),
            "pipeline_mode": data.get("pipeline_mode", False),
            "notify_on_complete": data.get("notify_on_complete", True),
            "max_retries": data.get("max_retries", 3),
            "scheduled_at": data.get("scheduled_at", ""),
            "depends_on": data.get("depends_on"),
            "pipeline": data.get("pipeline", ""),
            "created": now,
            "updated": now,
            "body": "",
        }
        _autonomy_write_file(task_dict)
        return JSONResponse({"task": {"id": new_id}})
    else:
        path = _autonomy_find_file(task_id)
        if not path:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        task = _autonomy_parse(path)
        if not task:
            raise HTTPException(status_code=500, detail=f"Failed to parse task {task_id}")
        for key in ("name", "description", "status", "priority", "frequency", "skill_name",
                     "agent_id", "model", "scheduled_at", "pipeline", "auto_advance",
                     "preemptible", "pipeline_mode", "notify_on_complete", "timeout_seconds",
                     "max_retries", "depends_on", "preferred_hours", "cron_id", "runs_per_day"):
            if key in data:
                task[key] = data[key]
        if "tags" in data:
            tags = data["tags"]
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            task["tags"] = tags
        task["created"] = task.get("created_at") or task.get("created", "")
        task["updated"] = now
        _autonomy_write_file(task)
        return JSONResponse({"task": {"id": task_id}})


@router.post("/api/autonomy/task-delete")
async def autonomy_task_delete(request: Request):
    """Delete an autonomy task."""
    data = await request.json()
    task_id = data.get("id", 0)
    if not task_id:
        raise HTTPException(status_code=400, detail="id required")
    path = _autonomy_find_file(task_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    path.unlink()
    return JSONResponse({"success": True, "id": task_id})


@router.get("/api/autonomy/runs")
async def autonomy_runs(task_id: int = 0, limit: int = 20):
    """Get recent runs for an autonomy task."""
    if not task_id:
        return JSONResponse({"runs": []})
    runs_dir = _AUTONOMY_RUNS_DIR / str(task_id)
    if not runs_dir.exists():
        return JSONResponse({"runs": []})
    runs = []
    run_files = sorted(runs_dir.glob("*.md"), key=lambda p: p.name, reverse=True)
    for run_path in run_files[:limit]:
        try:
            content = run_path.read_text(encoding="utf-8")
            parts = content.split("---\n", 2)
            if len(parts) < 3:
                continue
            rfm = yaml.safe_load(parts[1])
            if not isinstance(rfm, dict):
                continue

            def _to_iso(val):
                if val is None:
                    return None
                if isinstance(val, datetime):
                    return val.strftime("%Y-%m-%dT%H:%M:%SZ")
                return str(val) if val else None

            runs.append({
                "run_id": rfm.get("run_id", 0),
                "task_id": rfm.get("task_id", task_id),
                "status": rfm.get("status", ""),
                "duration_seconds": rfm.get("duration_seconds"),
                "started_at": _to_iso(rfm.get("started_at")),
                "completed_at": _to_iso(rfm.get("completed_at")),
                "body": parts[2] if len(parts) > 2 else "",
            })
        except Exception:
            continue
    return JSONResponse({"runs": runs})


# ── Startup hook (recovery only — scheduling lives in workers/pool.py) ──────


async def start_autonomy_ticker():
    """Recover stuck tasks on startup. Scheduling is driven by the worker pool."""
    try:
        from autonomy import recover_stuck_tasks
        recovered = recover_stuck_tasks()
        if recovered:
            logger.info("Autonomy startup: recovered %d stuck task(s): %s", len(recovered), recovered)
    except ImportError:
        pass
    logger.info("Autonomy startup complete — worker pool handles task execution")
