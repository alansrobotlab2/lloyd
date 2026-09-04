#!/usr/bin/env python3
"""
Lloyd MCP Server: Autonomy — task management tools.

Provides tools for creating, listing, running, and managing autonomy tasks
stored in ~/obsidian/autonomy/ as markdown files with YAML frontmatter.

Tools: autonomy_tasks, autonomy_write_task, autonomy_get_task,
       autonomy_delete_task, autonomy_config, autonomy_run_task
"""

import asyncio
import datetime
import json
import re
import sys
from pathlib import Path

import yaml
from mcp.types import Tool

from agent_mcp._shared import parse_frontmatter_text, text_result

AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"


def _slugify(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name[:50]


def _parse_task_file(path: Path) -> dict | None:
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        # Resilient parse: bad YAML degrades to regex extraction instead of
        # returning None (a None here silently drops the task from listings —
        # the same failure mode that dormant-killed 34/40 scheduler tasks).
        frontmatter = parse_frontmatter_text(
            parts[1],
            fallback_fields=(
                "id", "name", "description", "status", "priority", "frequency",
                "scheduled_at", "next_run", "last_run", "agent_id", "skill_name",
                "timeout_seconds", "board_id",
            ),
            log_label=f"autonomy:{path.name}",
        )

        def _to_iso(val):
            if val is None:
                return None
            if isinstance(val, datetime.datetime):
                return val.strftime("%Y-%m-%dT%H:%M:%SZ")
            return val

        return {
            "id": frontmatter.get("id", 0),
            "name": frontmatter.get("name", ""),
            "description": frontmatter.get("description", ""),
            "status": frontmatter.get("status", "draft"),
            "priority": frontmatter.get("priority", "medium"),
            "frequency": frontmatter.get("frequency", ""),
            "scheduled_at": _to_iso(frontmatter.get("scheduled_at", "")),
            "last_run": _to_iso(frontmatter.get("last_run")),
            "next_run": _to_iso(frontmatter.get("next_run")),
            "auto_advance": bool(frontmatter.get("auto_advance", False)),
            "preemptible": bool(frontmatter.get("preemptible", True)),
            "board_id": frontmatter.get("board_id", 4),
            "created_at": _to_iso(frontmatter.get("created", frontmatter.get("created_at", ""))),
            "updated_at": _to_iso(frontmatter.get("updated", frontmatter.get("updated_at", ""))),
            "skill_name": frontmatter.get("skill_name", frontmatter.get("skill_path", "")),
            "agent_id": frontmatter.get("agent_id", "memory"),
            "model": frontmatter.get("model", ""),
            "timeout_seconds": frontmatter.get("timeout_seconds", 1800),
            "depends_on": frontmatter.get("depends_on"),
            "pipeline": frontmatter.get("pipeline"),
            "runs_per_day": frontmatter.get("runs_per_day"),
            "run_count": frontmatter.get("run_count"),
            "failure_count": frontmatter.get("failure_count", 0),
            "max_retries": frontmatter.get("max_retries", 3),
            "notify_on_complete": frontmatter.get("notify_on_complete", True),
            "preferred_hours": frontmatter.get("preferred_hours", []),
            "last_attempt": _to_iso(frontmatter.get("last_attempt")),
            "stale_bypass_hours": frontmatter.get("stale_bypass_hours"),
            "expected_error_patterns": frontmatter.get("expected_error_patterns") or [],
            "cron_id": frontmatter.get("cron_id"),
            "type": frontmatter.get("type", "autonomy"),
            "body": parts[2] if len(parts) > 2 else "",
        }
    except Exception:
        return None


def _write_task_file(task_dict: dict) -> Path:
    task_id = task_dict.get("id", 0)
    name = task_dict.get("name", "unnamed")
    slug = _slugify(name)
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)

    existing = _find_task_file(task_id)
    path = existing if existing else AUTONOMY_DIR / f"{task_id}-{slug}.md"

    # Start from the file's EXISTING frontmatter so keys this module doesn't
    # model survive the write. The old code rebuilt frontmatter from the map
    # below alone, which destroyed `tags`, `segment`, `stale_bypass_hours`,
    # `expected_error_patterns`, `archived_reason` and anything else added
    # later — and, worse, substituted a DEFAULT for keys the caller omitted
    # (e.g. an update that didn't mention preferred_hours reset it to []).
    frontmatter: dict = {}
    if existing and existing.exists():
        try:
            from agent_mcp._shared import parse_frontmatter_text
            prior_parts = existing.read_text(encoding="utf-8").split("---\n", 2)
            if len(prior_parts) >= 3:
                prior = parse_frontmatter_text(
                    prior_parts[1], log_label=f"autonomy_write_task:{existing.name}")
                if isinstance(prior, dict):
                    frontmatter = {k: v for k, v in prior.items()
                                   if not str(k).startswith("_")}
        except Exception:
            pass

    updates = {
        "type": task_dict.get("type") or "autonomy",
        "id": task_dict.get("id", 0),
        "name": task_dict.get("name"),
        "description": task_dict.get("description"),
        "status": task_dict.get("status"),
        "priority": task_dict.get("priority"),
        "frequency": task_dict.get("frequency"),
        "agent_id": task_dict.get("agent_id"),
        "model": task_dict.get("model"),
        "auto_advance": task_dict.get("auto_advance"),
        "preemptible": task_dict.get("preemptible"),
        "pipeline_mode": task_dict.get("pipeline_mode"),
        "timeout_seconds": task_dict.get("timeout_seconds"),
        "max_retries": task_dict.get("max_retries"),
        "failure_count": task_dict.get("failure_count"),
        "skill_name": task_dict.get("skill_name", task_dict.get("skill_path")),
        "cron_id": task_dict.get("cron_id"),
        "runs_per_day": task_dict.get("runs_per_day"),
        "scheduled_at": task_dict.get("scheduled_at"),
        "last_run": task_dict.get("last_run"),
        "last_attempt": task_dict.get("last_attempt"),
        "next_run": task_dict.get("next_run"),
        "depends_on": task_dict.get("depends_on"),
        "preferred_hours": task_dict.get("preferred_hours"),
        "notify_on_complete": task_dict.get("notify_on_complete"),
        "pipeline": task_dict.get("pipeline"),
        "stale_bypass_hours": task_dict.get("stale_bypass_hours"),
        "expected_error_patterns": task_dict.get("expected_error_patterns"),
        "created": task_dict.get("created_at", task_dict.get("created")),
        "updated": task_dict.get("updated_at", task_dict.get("updated")),
    }
    frontmatter.update({k: v for k, v in updates.items() if v is not None})
    frontmatter = {k: v for k, v in frontmatter.items() if v is not None}
    body = task_dict.get("body", "")
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{body}"
    path.write_text(content, encoding="utf-8")
    return path


def _find_task_file(task_id: int) -> Path | None:
    if not AUTONOMY_DIR.exists():
        return None
    patterns = [p for p in AUTONOMY_DIR.glob(f"{task_id}-*.md") if p.name != "_config.md"]
    return patterns[0] if patterns else None


def _next_task_id() -> int:
    if not AUTONOMY_DIR.exists():
        return 1
    max_id = 0
    for path in AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        name = path.name
        if "-" in name:
            id_str = name.split("-")[0]
            if id_str.isdigit():
                max_id = max(max_id, int(id_str))
    return max_id + 1


def _parse_run_file(path: Path) -> dict | None:
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        frontmatter = yaml.safe_load(parts[1])
        if not isinstance(frontmatter, dict):
            return None
        return {
            "run_id": frontmatter.get("run_id", 0),
            "task_id": frontmatter.get("task_id", 0),
            "status": frontmatter.get("status", ""),
            "duration_seconds": frontmatter.get("duration_seconds"),
            "started_at": frontmatter.get("started_at"),
            "completed_at": frontmatter.get("completed_at"),
            "body": parts[2] if len(parts) > 2 else "",
        }
    except Exception:
        return None


def _read_config() -> dict:
    config_path = AUTONOMY_DIR / "_config.md"
    if not config_path.exists():
        return {}
    try:
        content = config_path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) >= 2:
            frontmatter = yaml.safe_load(parts[1])
            if isinstance(frontmatter, dict):
                return frontmatter
    except Exception:
        pass
    return {}


def _write_config(config: dict) -> None:
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    config_path = AUTONOMY_DIR / "_config.md"
    content = f"---\n{yaml.dump(config, default_flow_style=False, allow_unicode=True)}---\n\n"
    config_path.write_text(content, encoding="utf-8")


# ── Tool definitions ──────────────────────────────────────────────────────────

async def list_tools():
    return [
        Tool(name="autonomy_tasks", description="List the scheduled autonomy tasks, optionally filtered by status, frequency or agent. Returns task objects with their schedule and last-run state, not their full history.", inputSchema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter by status (draft, up_next, in_progress)"},
                "frequency": {"type": "string", "description": "Filter by frequency"},
                "agent_id": {"type": "string", "description": "Filter by agent_id"},
            },
        }),
        Tool(name="autonomy_write_task", description="Create or update (upsert) autonomy task. If id omitted → CREATE, if id provided → UPDATE.", inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Task ID to update (0 for create)"},
                "name": {"type": "string", "description": "Task name/title"},
                "description": {"type": "string", "description": "Task description"},
                "status": {"type": "string", "description": "Task status (draft, up_next, in_progress)"},
                "priority": {"type": "string", "description": "Task priority (low, medium, high)"},
                "frequency": {"type": "string", "description": "Task frequency"},
                "skill_name": {"type": "string", "description": "Skill slug (e.g. 'autonomy-data-pipeline'). Resolves to ~/obsidian/skills/<slug>/SKILL.md"},
                "agent_id": {"type": "string", "description": "Agent ID to run the task"},
                "model": {"type": "string", "description": "Model to use"},
                "timeout_seconds": {"type": "integer", "description": "Timeout in seconds"},
                "auto_advance": {"type": "boolean", "description": "Move the task to the next status automatically when a run succeeds"},
                "preemptible": {"type": "boolean", "description": "Allow the scheduler to interrupt this run for a higher-priority task"},
                "scheduled_at": {"type": "string", "description": "ISO timestamp for the next run; overrides the frequency for one cycle"},
                "depends_on": {"type": "integer", "description": "Task ID that must complete successfully before this one runs"},
                "pipeline": {"type": "string", "description": "Named pipeline this task belongs to; tasks in one pipeline run in order"},
                "activity_note": {"type": "string", "description": "Note to append to activity log"},
            },
        }),
        Tool(name="autonomy_get_task", description="Get one autonomy task in full: its definition, schedule, dependencies and its most recent run records with exit status.", inputSchema={
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "Task ID to retrieve"}},
            "required": ["id"],
        }),
        Tool(name="autonomy_delete_task", description="Delete an autonomy task, or archive it by setting status back to draft. Archiving is reversible; deletion is not.", inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Task ID to delete"},
                "archive": {"type": "boolean", "description": "If true, set status to draft instead of deleting"},
            },
            "required": ["id"],
        }),
        Tool(name="autonomy_config", description="Read or change autonomy scheduler configuration. Called with no key it returns the whole config; with a key and no value it reads one setting.", inputSchema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Config key to get/set (empty to get all)"},
                "value": {"type": "string", "description": "Value to set (None to get key)"},
            },
        }),
        Tool(name="autonomy_run_task", description="Run an autonomy task immediately, outside its schedule. The run happens in the background; use autonomy_get_task to see the result.", inputSchema={
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "Task ID to run"}},
            "required": ["id"],
        }),
        Tool(name="autonomy_health", description=(
            "Fleet health over the last N days: per-task runs, failure rate, timeouts, "
            "empty runs, [SILENT] rate, GPU-hours, wasted hours, consecutive failures, "
            "plus fleet totals. Use this to find tasks that are burning GPU without "
            "producing anything."), inputSchema={
            "type": "object",
            "properties": {"days": {"type": "integer", "description": "Window in days (default 7, max 90)"}},
        }),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "autonomy_tasks":
        return text_result(_handle_tasks(arguments))
    elif name == "autonomy_write_task":
        return text_result(_handle_write(arguments))
    elif name == "autonomy_get_task":
        return text_result(_handle_get(arguments))
    elif name == "autonomy_delete_task":
        return text_result(_handle_delete(arguments))
    elif name == "autonomy_config":
        return text_result(_handle_config(arguments))
    elif name == "autonomy_run_task":
        text = await _handle_run(arguments)
        return text_result(text)
    elif name == "autonomy_health":
        text = await _handle_health(arguments)
        return text_result(text)
    return text_result(json.dumps({"error": f"Unknown tool: {name}"}))


async def _handle_health(params: dict) -> str:
    """Proxy to the backend's /api/autonomy/health.

    agent_mcp runs in its own process, so the work queue singleton isn't
    initialised here — the backend owns it.
    """
    days = int(params.get("days") or 7)
    try:
        from agent_mcp._shared import make_http_client
        from app.config import service_url
        base = service_url("backend", "http://127.0.0.1:8080")
        async with make_http_client(timeout=30.0) as client:
            r = await client.get(f"{base}/api/autonomy/health", params={"days": days})
            if r.status_code >= 400:
                return json.dumps({"error": f"backend returned {r.status_code}",
                                   "body": r.text[:500]})
            return r.text
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _handle_tasks(params: dict) -> str:
    if not AUTONOMY_DIR.exists():
        return json.dumps({"tasks": []})
    status = params.get("status", "")
    frequency = params.get("frequency", "")
    agent_id = params.get("agent_id", "")
    tasks = []
    for path in AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        task = _parse_task_file(path)
        if task is None:
            continue
        if status and task.get("status") != status:
            continue
        if frequency and task.get("frequency") != frequency:
            continue
        if agent_id and task.get("agent_id") != agent_id:
            continue
        tasks.append(task)
    return json.dumps({"tasks": tasks})


def _handle_write(params: dict) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    task_id = params.get("id", 0)

    if task_id == 0:
        name = params.get("name", "")
        if not name:
            return json.dumps({"error": "name is required when creating a task"})
        new_id = _next_task_id()
        task_dict = {
            "id": new_id,
            "name": name,
            "description": params.get("description", ""),
            "status": params.get("status", "") or "draft",
            "priority": params.get("priority", "") or "medium",
            "frequency": params.get("frequency", ""),
            "skill_name": params.get("skill_name", params.get("skill_path", "")),
            "agent_id": params.get("agent_id", "") or "memory",
            "model": params.get("model", ""),
            "timeout_seconds": params.get("timeout_seconds", 0) or 1800,
            "auto_advance": params.get("auto_advance", False),
            "preemptible": params.get("preemptible", True),
            "scheduled_at": params.get("scheduled_at", ""),
            "depends_on": params.get("depends_on") if params.get("depends_on", 0) else None,
            "pipeline": params.get("pipeline", ""),
            "created_at": now,
            "updated_at": now,
            "body": "",
        }
        _write_task_file(task_dict)
        return json.dumps({"task": task_dict})
    else:
        existing_path = _find_task_file(task_id)
        if not existing_path:
            return json.dumps({"error": f"Task #{task_id} not found"})
        task_dict = _parse_task_file(existing_path)
        if task_dict is None:
            return json.dumps({"error": f"Failed to parse task #{task_id}"})
        for key in ("status", "priority", "frequency", "skill_name", "agent_id", "model",
                     "scheduled_at", "pipeline", "description"):
            if params.get(key):
                task_dict[key] = params[key]
        if params.get("timeout_seconds", 0) > 0:
            task_dict["timeout_seconds"] = params["timeout_seconds"]
        if params.get("depends_on", 0) > 0:
            task_dict["depends_on"] = params["depends_on"]
        if "auto_advance" in params:
            task_dict["auto_advance"] = params["auto_advance"]
        if "preemptible" in params:
            task_dict["preemptible"] = params["preemptible"]
        task_dict["updated_at"] = now
        activity_note = params.get("activity_note", "")
        if activity_note:
            body = task_dict.get("body", "")
            if "## Activity Log" not in body:
                body += "\n\n## Activity Log\n"
            body += f"\n- {now}: {activity_note}\n"
            task_dict["body"] = body
        _write_task_file(task_dict)
        return json.dumps({"task": task_dict})


def _handle_get(params: dict) -> str:
    task_id = params.get("id", 0)
    if task_id == 0:
        return json.dumps({"error": "id is required"})
    path = _find_task_file(task_id)
    if not path:
        return json.dumps({"error": f"Task #{task_id} not found"})
    task = _parse_task_file(path)
    if task is None:
        return json.dumps({"error": f"Failed to parse task #{task_id}"})
    runs_dir = AUTONOMY_DIR / "runs" / str(task_id)
    runs = []
    if runs_dir.exists():
        run_files = sorted(runs_dir.glob("*.md"), key=lambda p: p.name, reverse=True)
        for run_path in run_files[:10]:
            run = _parse_run_file(run_path)
            if run:
                runs.append(run)
    runs = sorted(runs, key=lambda r: r.get("started_at", "") or "", reverse=True)
    task["runs"] = runs
    return json.dumps({"task": task})


def _handle_delete(params: dict) -> str:
    task_id = params.get("id", 0)
    if task_id == 0:
        return json.dumps({"error": "id is required"})
    archive = params.get("archive", True)
    path = _find_task_file(task_id)
    if not path:
        return json.dumps({"error": f"Task #{task_id} not found"})
    if archive:
        task = _parse_task_file(path)
        if task:
            task["status"] = "draft"
            task["updated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _write_task_file(task)
            return json.dumps({"success": True, "id": task_id})
        return json.dumps({"error": f"Failed to parse task #{task_id}"})
    else:
        path.unlink()
        return json.dumps({"success": True, "id": task_id})


def _handle_config(params: dict) -> str:
    key = params.get("key", "")
    value = params.get("value", None)
    if not key:
        return json.dumps(_read_config())
    if value is not None:
        config = _read_config()
        config[key] = value
        _write_config(config)
        return json.dumps({"set": key, "value": value})
    else:
        config = _read_config()
        if key in config:
            return json.dumps({key: config[key]})
        return json.dumps({"error": f"Config key not found: {key}"})


async def _handle_run(params: dict) -> str:
    task_id = params.get("id", 0)
    if task_id == 0:
        return json.dumps({"error": "id is required"})
    try:
        lloyd_home = Path.home() / "lloyd"
        if str(lloyd_home) not in sys.path:
            sys.path.insert(0, str(lloyd_home))
        from autonomy import run_task
        result = await run_task(task_id)
        return json.dumps(result)
    except ImportError as e:
        return json.dumps({"error": f"autonomy scheduler module not available: {e}"})
    except Exception as exc:
        return json.dumps({"error": str(exc), "task_id": task_id})

