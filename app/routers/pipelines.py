"""Pipeline run listing, detail, and abort endpoints."""

import json
import logging
from datetime import datetime

import psutil
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.paths import PIPELINE_RUNS_DIR


router = APIRouter()
logger = logging.getLogger("lloyd-server")


def _pipeline_summary(run: dict) -> dict:
    """Reduce a full pipeline run to the fields needed by the UI."""
    task = run.get("task", "")
    task_preview = next((l.strip() for l in task.splitlines() if l.strip()), task[:80])
    stages = run.get("stages", [])
    idx = run.get("current_stage_index", 0)
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status", "unknown"),
        "task_preview": task_preview[:120],
        "current_stage": run.get("current_stage", stages[idx] if idx < len(stages) else ""),
        "stage_index": idx,
        "stage_count": len(stages),
        "stages": stages,
        "model": run.get("model", ""),
        "created_at": run.get("created_at", ""),
        "updated_at": run.get("updated_at", ""),
        "completed_at": run.get("completed_at", ""),
        "blocked_reason": run.get("blocked_reason", ""),
    }


@router.get("/api/pipelines")
async def list_pipelines(status: str = ""):
    """List pipeline runs, optionally filtered by status."""
    if not PIPELINE_RUNS_DIR.exists():
        return JSONResponse({"runs": []})
    runs = []
    for p in sorted(PIPELINE_RUNS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            run = json.loads(p.read_text(encoding="utf-8"))
            if status and run.get("status") != status:
                continue
            runs.append(_pipeline_summary(run))
        except Exception:
            pass
    return JSONResponse({"runs": runs[:50]})


@router.get("/api/pipelines/{run_id}")
async def get_pipeline(run_id: int, log_tail: int = 8000):
    """Get full details for a pipeline run, including live log tail."""
    run_path = PIPELINE_RUNS_DIR / f"{run_id}.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail=f"Pipeline run {run_id} not found")
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    log_path = PIPELINE_RUNS_DIR / f"{run_id}.log"
    log_content = ""
    if log_path.exists():
        try:
            raw = log_path.read_text(encoding="utf-8", errors="replace")
            log_content = raw[-log_tail:] if len(raw) > log_tail else raw
        except Exception:
            pass

    summary = _pipeline_summary(run)
    summary["task"] = run.get("task", "")
    summary["stage_outputs"] = run.get("stage_outputs", {})
    summary["skills"] = [s.get("name", "") for s in run.get("skills", [])]
    summary["live_log"] = log_content
    return JSONResponse(summary)


@router.post("/api/pipelines/{run_id}/abort")
async def abort_pipeline(run_id: int):
    """Mark a pipeline run as aborted and kill its claude subprocess."""
    run_path = PIPELINE_RUNS_DIR / f"{run_id}.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail=f"Pipeline run {run_id} not found")

    # 1. Cooperative abort — the pipeline thread checks this flag between stages
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("status") not in ("running", "blocked"):
            return JSONResponse({"success": True, "message": f"Run already {run.get('status')}"})
        run["status"] = "aborted"
        run["completed_at"] = datetime.now().isoformat()
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update run file: {e}")

    # 2. Hard kill — find claude subprocesses spawned by the MCP server process
    killed_pids = []
    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if "pipeline" not in cmdline and "mcp_server" not in cmdline:
                    continue
                for child in proc.children(recursive=True):
                    try:
                        child_cmd = " ".join(child.cmdline())
                        if "claude" in child_cmd and "vscode" not in child_cmd:
                            child.terminate()
                            killed_pids.append(child.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        logger.warning(f"Failed to kill claude subprocesses for run {run_id}: {e}")

    return JSONResponse({
        "success": True,
        "run_id": run_id,
        "killed_pids": killed_pids,
        "message": f"Aborted. Killed {len(killed_pids)} subprocess(es).",
    })
