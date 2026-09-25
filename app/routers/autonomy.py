"""Autonomy endpoints (status/enable/disable/run + file-backed task CRUD)
plus the background scheduler ticker registered at app startup.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_mcp._shared import (
    AUTONOMY_TASK_FIELDS, parse_frontmatter_text, with_body_contract)


router = APIRouter()
logger = logging.getLogger("lloyd-server")

_AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
from app.paths import AUTONOMY_RUNS_DIR as _AUTONOMY_RUNS_DIR


# ── Runtime control ──────────────────────────────────────────────────────────


@router.post("/api/autonomy/run")
async def autonomy_run(request: Request):
    """Run a task immediately. Bypasses due-checks, but not the in-progress guard."""
    try:
        from autonomy import run_task, run_trigger
        data = await request.json()
        task_id = data.get("task_id")
        if not task_id:
            raise HTTPException(status_code=400, detail="task_id required")
        with run_trigger("api"):
            result = await run_task(int(task_id))
        if result.get("skipped"):
            # A run is already in flight for this task.
            raise HTTPException(status_code=409, detail=result.get("error", "already running"))
        return JSONResponse(result)
    except ImportError:
        raise HTTPException(status_code=501, detail="Autonomy module not available")


# ── File-backed task CRUD ────────────────────────────────────────────────────


def _autonomy_parse(path: Path) -> dict | None:
    """Parse an autonomy task markdown file into a normalized dict.

    Graduated recovery, same as the scheduler and the MCP reader, over the SAME
    field list (`agent_mcp._shared.AUTONOMY_TASK_FIELDS`). This reader used to
    `yaml.safe_load` and `except Exception: return None`, and its caller does
    `if task is None: continue` — so a YAML-broken task was simply absent from
    Mission Control while the scheduler loaded and dispatched it. The degraded
    fleet this protects against (2026-05-28, 34 of 40 files) therefore presented
    as "N tasks, none broken": board and scheduler disagreed about the fleet in
    the direction that looks healthy, with no signal anywhere (#1014).

    A file that comes back from the regex fallback is LISTED, flagged
    `yaml_broken: True`, and refused by the write endpoint below — visible, not
    rewritable in place."""
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        fm = parse_frontmatter_text(
            parts[1],
            fallback_fields=AUTONOMY_TASK_FIELDS,
            log_label=f"autonomy-api:{path.name}",
        )
        if not isinstance(fm, dict):
            return None
        degraded = bool(fm.get("_yaml_broken"))

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
            "failure_count": fm.get("failure_count") or 0,
            "last_attempt": _to_iso(fm.get("last_attempt")),
            # #1085's infra ceiling. These two are not decoration on the row: the
            # list handler below feeds THIS dict to `autonomy.hold_reason`, whose
            # `_in_infra_rest` reads `infra_rest_until` — so a projection that
            # omitted it answered "nothing is holding this task" for a task the
            # scheduler was refusing for a whole declared period. The board is the
            # surface a human reads during a `weekly` rest, and a row that looks
            # due while the fleet holds is the #1014 failure shape (board and
            # scheduler disagreeing in the direction that looks healthy) wearing a
            # new field.
            "infra_failure_count": fm.get("infra_failure_count") or 0,
            "infra_rest_until": _to_iso(fm.get("infra_rest_until")),
            "stale_bypass_hours": fm.get("stale_bypass_hours"),
            "expected_error_patterns": fm.get("expected_error_patterns") or [],
            "preferred_hours": fm.get("preferred_hours") or None,
            "cron_id": fm.get("cron_id"),
            # The flag is the point of the whole change: a row that looks
            # healthy next to 31 healthy rows is how a degraded fleet hides.
            # Same name the scheduler's own dict, `agent_mcp/backlog.py` and
            # `_reject_broken_fm` use, and underscore-prefixed like them: every
            # writer here strips `_`-prefixed keys before dumping, so a degraded
            # flag can never be written into a task file.
            "_yaml_broken": degraded,
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


# Frontmatter keys this endpoint is allowed to set. Anything else already in
# the file is preserved verbatim (see _autonomy_write_file) — the old behaviour
# rebuilt frontmatter from this list alone, so every key not named here was
# silently destroyed by any UI edit or task-write call. `tags` was never in the
# list at all, and neither were fields added later (stale_bypass_hours,
# expected_error_patterns, last_attempt, archived_reason, segment).
_WRITABLE_TASK_KEYS = (
    "type", "id", "name", "description", "status", "priority", "frequency",
    "agent_id", "model", "auto_advance", "preemptible", "pipeline_mode",
    "timeout_seconds", "max_retries", "failure_count", "skill_name", "cron_id",
    "runs_per_day", "scheduled_at", "last_run", "last_attempt", "next_run",
    "depends_on", "preferred_hours", "notify_on_complete", "pipeline",
    "created", "updated", "tags", "stale_bypass_hours", "expected_error_patterns",
)


def _autonomy_write_file(task_dict: dict) -> Path:
    """Write a task dict back to its markdown file.

    Overlays the provided fields onto the file's EXISTING frontmatter rather
    than rebuilding it, so keys this endpoint doesn't know about survive.
    """
    task_id = task_dict.get("id", 0)
    name = task_dict.get("name", "unnamed")
    _AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    existing = _autonomy_find_file(task_id)
    path = existing if existing else _AUTONOMY_DIR / f"{task_id}-{_autonomy_slugify(name)}.md"

    fm: dict = {}
    if existing and existing.exists():
        try:
            from agent_mcp._shared import parse_frontmatter_text
            parts = existing.read_text(encoding="utf-8").split("---\n", 2)
            if len(parts) >= 3:
                prior = parse_frontmatter_text(parts[1], log_label=f"task-write:{existing.name}")
                if isinstance(prior, dict):
                    fm = {k: v for k, v in prior.items() if not str(k).startswith("_")}
        except Exception as e:
            logger.warning("task-write: could not read existing frontmatter for #%s: %s", task_id, e)

    for key in _WRITABLE_TASK_KEYS:
        if key in task_dict and task_dict[key] is not None:
            fm[key] = task_dict[key]
    if "type" not in fm:
        fm["type"] = "autonomy"
    # Stamped in the writer, not in the create branch, for the reason named in
    # `agent_mcp/autonomy._write_task_file`: this is the second of the two routes
    # that can put a task file on disk (the Mission Control autonomy page writes
    # through here), and the pin reads the directory, not the call site.
    body = with_body_contract(task_dict.get("body", ""))
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
        if not re.match(r"\d+-", path.name):
            continue  # only NN-name.md task files; skip _config.md, reports, notes
        task = _autonomy_parse(path)
        if task is None:
            continue
        if status and task.get("status") != status:
            continue
        if tag and tag not in (task.get("tags") or []):
            continue
        tasks.append(task)
    # Why the scheduler is holding each task, from the scheduler itself. The
    # board used to derive "overdue" from elapsed/interval alone, which paints
    # a nightly job red for the eighteen hours a day it is not allowed to run.
    try:
        import autonomy as _a

        # One resolution input, shared with dispatch (#870). This endpoint used to
        # assemble its own `everything` — every parsed task regardless of status,
        # deliberately, because at the time an unresolvable `depends_on` id counted
        # as met and a filtered view would have reported every dependency
        # satisfied — while dispatch resolved the same `depends_on` against the
        # status-filtered runnable set.
        # Same gate, two inputs: for a `paused` upstream the board printed
        # `waiting on #N` in the same second the scheduler dispatched the
        # dependent. The whole-board reasoning now lives in
        # `autonomy.dependency_resolution_set()`, which the dispatcher reads too.
        #
        # It is passed `_AUTONOMY_DIR`, not left to default to `autonomy.AUTONOMY_DIR`,
        # because this handler lists the tasks from `_AUTONOMY_DIR`: taking the
        # default would make one request read TWO directory globals and report a
        # board that lists one tree while gating it against another — silent until
        # someone relocates one of them. One request, one directory, so the set the
        # rows came from is by construction the set they were resolved against.
        resolution = _a.dependency_resolution_set(_AUTONOMY_DIR)
        for task in tasks:
            try:
                task["blocked"] = _a.hold_reason(task, resolution)
            except Exception:
                task["blocked"] = None
    except Exception as e:
        logger.warning("autonomy hold_reason unavailable: %s", e)
        for task in tasks:
            task["blocked"] = None
    return JSONResponse({"tasks": tasks})


@router.post("/api/autonomy/task-write")
async def autonomy_task_write(request: Request):
    """Create or update an autonomy task."""
    data = await request.json()
    # `timezone.utc`, not the bare `datetime.now()` this line used: the latter is
    # naive — the host's wall clock, `America/Los_Angeles (PDT, -0700)` — while
    # the `Z` it was formatted with asserts UTC. Every write through the
    # autonomy page therefore recorded an instant 7.00 h behind the one it
    # happened at (#1128). That is load-bearing, not cosmetic: `_parse_iso` reads
    # a trailing `Z` as UTC, and `recover_stuck_tasks` plus `run_task`'s
    # double-run guard compare the result against a real UTC now — so a task
    # claimed from the UI read as 25,200 s old on the tick it was written, past
    # its own 1,800 s timeout, and was flipped back to `up_next` with a
    # fabricated `Recovered from in_progress after 25200s` in its log. Pinned by
    # tests/test_autonomy_task_write_timestamp.py.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    task_id = data.get("id", 0)
    if not task_id:
        name = data.get("name", "")
        if not name:
            raise HTTPException(status_code=400, detail="name required for new task")
        new_id = _autonomy_next_id()
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
        if task.get("_yaml_broken"):
            # Listing a degraded task (#1014) must not make it rewritable in
            # place: `_autonomy_write_file` rebuilds the file from what it could
            # read, so saving here would write the recovered fields back over the
            # file and drop every key the regex fallback could not see — turning a
            # visible defect into a lost schedule. The backlog board refuses the
            # same way (`_reject_broken_fm`), and the MCP writer now agrees.
            raise HTTPException(
                status_code=409,
                detail=f"{path.name} has malformed YAML frontmatter; fix it by hand "
                       "before editing it here (rewriting would drop the fields the "
                       "fallback parse could not read).",
            )
        prior_status = task.get("status")
        for key in ("name", "description", "status", "priority", "frequency", "skill_name",
                     "agent_id", "model", "scheduled_at", "pipeline", "auto_advance",
                     "preemptible", "pipeline_mode", "notify_on_complete", "timeout_seconds",
                     "max_retries", "depends_on", "preferred_hours", "cron_id", "runs_per_day"):
            if key in data:
                task[key] = data[key]
        task["created"] = task.get("created_at") or task.get("created", "")
        task["updated"] = now
        # A status change from outside the scheduler records itself in the task's
        # own markdown (#1127 clause 4). `draft` and `paused` are never dispatched,
        # so before this the endpoint could park a task with no reason anywhere —
        # #68's disable read as unattributable for a day for exactly that reason.
        # The write is NOT blocked: a legitimate claim (`up_next` -> `in_progress`)
        # is an ordinary write, so the rung is a record, not a gate.
        #
        # Read from the clock again rather than reusing `now`: this line is what
        # dates an audit entry, and it was already passing
        # `datetime.now(timezone.utc)` when `now` above was the naive host clock
        # (#1128) — the two stamps were correct and wrong in the same request.
        # `now` is true UTC too now, so the two agree; the separate read stays
        # because an audit stamp should be the moment the note is written.
        from autonomy import append_activity_line, status_change_note
        status_note = status_change_note(prior_status, task.get("status"))
        if status_note:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            task["body"] = append_activity_line(task.get("body", ""), status_note, stamp)
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


@router.get("/api/autonomy/health")
async def autonomy_health(days: int = 7):
    """Fleet health: per-task failure rate, GPU-hours, timeouts, empty runs.

    Reads workers.db rather than the per-task run records, so pool-level
    failures are included — those were previously written without a task_id and
    were unreachable from any per-task view.

    `fleet.oldest_input` / `fleet.window_clamped_to_hours` name the store's own
    age when it is younger than `days`, because `days` is a request and not a
    measurement (#1401): the 2026-09-22 rebuild left a ~21 h table reporting a
    7-day `fail_rate` of 0.0, and the payload had no field a reader could check
    it against.
    """
    import asyncio as _asyncio
    from datetime import timedelta

    days = max(1, min(90, int(days)))
    try:
        from workers.queue import get_queue
        queue = get_queue()
    except Exception as e:
        return JSONResponse(
            {"error": f"work queue unavailable: {e}", "days": days}, status_code=503)

    import autonomy as _autonomy
    # ONE clock reading, used both to bound the query and to date the verdict.
    # The window used to come from `datetime.now` while `compute_health` read
    # `_utcnow()`, so the two edges of the same window were two instants; the
    # clamp reports the span this query actually ran over, so it has to be
    # measured from the instant the query was built at.
    now = _autonomy._utcnow()
    since = (now - timedelta(days=days)).isoformat()
    loop = _asyncio.get_event_loop()
    try:
        rows = await loop.run_in_executor(
            None, queue.list_runs_joined, "scheduled-task", since)
        # Unfiltered by the window on purpose: the case the clamp exists for is
        # a store whose every row predates the window, where the filtered read
        # above is empty and cannot say how old the data behind it was.
        oldest_input = await loop.run_in_executor(
            None, queue.oldest_run_completed_at, "scheduled-task")
        tasks = await loop.run_in_executor(None, _autonomy._iter_task_files)
        return JSONResponse(_autonomy.compute_health(
            rows, tasks, days, now=now, oldest_input=oldest_input))
    except Exception as e:
        logger.error("autonomy health failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e), "days": days}, status_code=500)


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
    """Recover stuck tasks on startup. Scheduling is driven by the worker pool.

    `recover_stuck_tasks()` REWRITES task files under the autonomy dir. A
    self-modification canary sets `autonomy.ticker_enabled: false` so a gate
    run cannot mutate real task state. (The canary also runs with its own
    HOME, so the dir it would touch is an empty scratch one — this flag is the
    second, explicit layer.)
    """
    from app.config import CONFIG
    if not (CONFIG.get("autonomy") or {}).get("ticker_enabled", True):
        logger.info("autonomy.ticker_enabled=false → skipping stuck-task recovery")
        return
    try:
        from autonomy import recover_stuck_tasks
        recovered = recover_stuck_tasks()
        if recovered:
            logger.info("Autonomy startup: recovered %d stuck task(s): %s", len(recovered), recovered)
    except ImportError:
        pass
    logger.info("Autonomy startup complete — worker pool handles task execution")
