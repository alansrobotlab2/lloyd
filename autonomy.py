"""
Lloyd Autonomy Scheduler — task scheduler and executor powered by Claude Agent SDK.

Evaluates which autonomy tasks are due based on frequency, last_run,
depends_on, preferred_hours, and priority. Executes the highest-priority
due task by calling query() from the Claude Agent SDK.

Usage:
    from autonomy import autonomy_tick, run_task
    autonomy_tick()       # Called every 60s from server.py
    run_task(task_id=48)  # Run a specific task on demand
"""

import asyncio
import datetime
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("lloyd-autonomy")

AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
AUTONOMY_RUNS_DIR = AUTONOMY_DIR / "runs"
LLOYD_HOME = Path(__file__).parent

# ── State ─────────────────────────────────────────────────────────────────────

_ticker_enabled = False
_ticker_running = False
_last_tick_time: float = 0.0
_current_task_id: Optional[int] = None


def recover_stuck_tasks() -> list:
    """Reset any tasks stuck in_progress longer than their timeout. Called on startup."""
    recovered = []
    if not AUTONOMY_DIR.exists():
        return recovered
    now = datetime.datetime.now(datetime.timezone.utc)
    for path in AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        task = _parse_task_file(path)
        if not task or str(task.get("status", "")).strip() != "in_progress":
            continue
        timeout = int(task.get("timeout_seconds") or 1800)
        updated = _parse_iso(task.get("updated") or task.get("last_run"))
        stuck_seconds = (now - updated).total_seconds() if updated else timeout + 1
        if stuck_seconds >= timeout:
            task_id = task.get("id")
            _update_task_field(task_id, status="up_next", updated=now.isoformat())
            _append_activity_log(task_id, f"Recovered from in_progress after {stuck_seconds:.0f}s (timeout={timeout}s)")
            logger.warning("Recovered stuck task #%s (%s) after %.0fs", task_id, task.get("name"), stuck_seconds)
            recovered.append(task_id)
    return recovered


def get_status() -> dict:
    return {
        "enabled": _ticker_enabled,
        "running": _ticker_running,
        "last_tick": _last_tick_time,
        "current_task_id": _current_task_id,
    }


def set_enabled(enabled: bool) -> None:
    global _ticker_enabled
    _ticker_enabled = enabled
    logger.info("Autonomy scheduler %s", "enabled" if enabled else "disabled")
    try:
        config_path = LLOYD_HOME / "config.yaml"
        config = yaml.safe_load(config_path.read_text()) or {}
        config.setdefault("autonomy", {})["enabled"] = enabled
        config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    except Exception as e:
        logger.warning("Failed to persist autonomy enabled state: %s", e)


# ── Task file I/O ─────────────────────────────────────────────────────────────

def _parse_task_file(path: Path) -> Optional[dict]:
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        fm = yaml.safe_load(parts[1])
        if not isinstance(fm, dict):
            return None
        fm["body"] = parts[2] if len(parts) > 2 else ""
        fm["_path"] = str(path)
        return fm
    except Exception as e:
        logger.error("Failed to parse %s: %s", path, e)
        return None


def _find_task_file(task_id) -> Optional[Path]:
    task_id = str(task_id)
    for path in AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        if path.name.startswith(f"{task_id}-"):
            return path
    return None


def _update_task_field(task_id, **fields) -> None:
    path = _find_task_file(task_id)
    if not path:
        return
    content = path.read_text(encoding="utf-8")
    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return
    fm = yaml.safe_load(parts[1])
    if not isinstance(fm, dict):
        return
    fm.update(fields)
    new_content = f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n{parts[2]}"
    path.write_text(new_content, encoding="utf-8")


def _append_activity_log(task_id, note: str) -> None:
    path = _find_task_file(task_id)
    if not path:
        return
    content = path.read_text(encoding="utf-8")
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_line = f"\n- {now_str}: {note}\n"
    if "## Activity Log" in content:
        content = content.rstrip() + log_line
    else:
        content = content.rstrip() + "\n\n## Activity Log\n" + log_line
    path.write_text(content, encoding="utf-8")


def _write_run_record(task_id: int, run_id: str, status: str,
                      started_at: str, completed_at: str,
                      duration_seconds: float, summary: str, body: str) -> Path:
    runs_dir = AUTONOMY_RUNS_DIR / str(task_id)
    runs_dir.mkdir(parents=True, exist_ok=True)
    fm = {
        "run_id": run_id, "task_id": task_id, "status": status,
        "started_at": started_at, "completed_at": completed_at,
        "duration_seconds": round(duration_seconds, 1), "summary": summary,
    }
    content = f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n{body}"
    path = runs_dir / f"{run_id}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ── Scheduling logic ──────────────────────────────────────────────────────────

def _all_runnable_tasks() -> list[dict]:
    if not AUTONOMY_DIR.exists():
        return []
    tasks = []
    for path in AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        task = _parse_task_file(path)
        if not task:
            continue
        status = str(task.get("status", "")).strip()
        if status in ("up_next", "in_progress"):
            tasks.append(task)
    return tasks


def _parse_iso(s) -> Optional[datetime.datetime]:
    if not s or str(s).strip().lower() in ("null", "none", ""):
        return None
    try:
        s = str(s).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except Exception:
        return None


def _frequency_interval_seconds(task: dict) -> Optional[float]:
    freq = str(task.get("frequency", "")).strip().lower()
    rpd = task.get("runs_per_day")
    if rpd:
        try:
            rpd_f = float(rpd)
            if rpd_f > 0:
                return 86400.0 / rpd_f
        except (TypeError, ValueError):
            pass
    freq_map = {"hourly": 3600, "every-15min": 900, "daily": 86400, "weekly": 604800}
    return freq_map.get(freq)


def _is_dependency_met(task: dict, all_tasks: list[dict]) -> bool:
    dep_id = task.get("depends_on")
    if not dep_id or str(dep_id).strip().lower() in ("null", "none", ""):
        return True
    try:
        dep_id = int(dep_id)
    except (TypeError, ValueError):
        return True
    dep_task = None
    for t in all_tasks:
        try:
            if int(t.get("id", 0)) == dep_id:
                dep_task = t
                break
        except (TypeError, ValueError):
            continue
    if not dep_task:
        return True
    dep_last_run = _parse_iso(dep_task.get("last_run"))
    if not dep_last_run:
        return False
    my_last_run = _parse_iso(task.get("last_run"))
    if not my_last_run:
        return True
    return dep_last_run > my_last_run


def _is_preferred_hour(task: dict) -> bool:
    pref = task.get("preferred_hours")
    if not pref or not isinstance(pref, list) or len(pref) == 0:
        return True
    current_hour = datetime.datetime.now().hour
    try:
        return current_hour in [int(h) for h in pref]
    except (TypeError, ValueError):
        return True


def _is_task_due(task: dict, all_tasks: list[dict]) -> bool:
    skill = str(task.get("skill_path", "") or "").strip()
    if not skill or skill.lower() in ("null", "none"):
        return False
    interval = _frequency_interval_seconds(task)
    if interval is None:
        return False
    last_run = _parse_iso(task.get("last_run"))
    now = datetime.datetime.now(datetime.timezone.utc)
    if last_run:
        elapsed = (now - last_run).total_seconds()
        if elapsed < interval:
            return False
    if not _is_dependency_met(task, all_tasks):
        return False
    if not _is_preferred_hour(task):
        return False
    return True


def _priority_key(task: dict) -> tuple:
    prio_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "background": 0}
    prio = prio_map.get(str(task.get("priority", "medium")).lower(), 2)
    last_run = _parse_iso(task.get("last_run"))
    overdue = (datetime.datetime.now(datetime.timezone.utc) - last_run).total_seconds() if last_run else 999999
    return (-prio, -overdue)


def get_due_tasks() -> list[dict]:
    all_tasks = _all_runnable_tasks()
    due = [t for t in all_tasks if _is_task_due(t, all_tasks)]
    due.sort(key=_priority_key)
    return due


# ── Task execution via Claude Agent SDK ───────────────────────────────────────

def _load_skill_content(skill_path: str) -> Optional[str]:
    expanded = Path(skill_path.replace("~", str(Path.home())))
    if not expanded.exists():
        alt = Path.home() / "obsidian" / skill_path
        if alt.exists():
            expanded = alt
        else:
            return None
    try:
        return expanded.read_text(encoding="utf-8")
    except Exception:
        return None


def _build_task_prompt(task: dict, skill_content: str) -> str:
    silent_hint = (
        "[SYSTEM: If you have a meaningful status report or findings, "
        "send them — that is the whole point of this task. Only respond "
        'with exactly "[SILENT]" (nothing else) when there is genuinely '
        "nothing new to report. [SILENT] suppresses delivery to the user. "
        "Never combine [SILENT] with content — either report your "
        "findings normally, or say [SILENT] and nothing more.]\n\n"
    )
    skill_name = task.get("name", "autonomy-task")
    parts = [
        silent_hint,
        f'[SYSTEM: You are executing autonomy task #{task.get("id")}: "{skill_name}". '
        f"Follow the skill instructions below.]",
        "",
        skill_content,
    ]
    description = str(task.get("description", "")).strip()
    if description:
        parts.extend(["", f"Task description: {description}"])
    return "\n".join(parts)


def _get_model_env(model_name: str) -> dict:
    """Get environment variable overrides for a model."""
    config_path = LLOYD_HOME / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
        models = config.get("models", {})
        if model_name in models:
            return models[model_name].get("env", {})
        for name, cfg in models.items():
            if cfg.get("alias") == model_name:
                return cfg.get("env", {})
    except Exception:
        pass
    return {}


def run_task(task_id) -> dict:
    """Execute a single autonomy task via Claude Agent SDK."""
    global _current_task_id

    path = _find_task_file(task_id)
    if not path:
        return {"success": False, "error": f"Task #{task_id} not found"}

    task = _parse_task_file(path)
    if not task:
        return {"success": False, "error": f"Failed to parse task #{task_id}"}

    skill_path = str(task.get("skill_path", "") or "").strip()
    if not skill_path or skill_path.lower() in ("null", "none"):
        return {"success": False, "error": f"Task #{task_id} has no skill_path"}

    skill_content = _load_skill_content(skill_path)
    if not skill_content:
        return {"success": False, "error": f"Skill not found: {skill_path}"}

    _update_task_field(task_id, status="in_progress")
    prompt = _build_task_prompt(task, skill_content)

    # Resolve model
    task_model = str(task.get("model", "") or "").strip()
    if not task_model or task_model.lower() in ("null", "none"):
        try:
            cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
            task_model = cfg.get("model", {}).get("default", "")
        except Exception:
            pass

    model_env = _get_model_env(task_model)
    timeout = int(task.get("timeout_seconds") or 1800)

    now = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat()
    run_id = f"run_{task_id}_{now.strftime('%Y%m%d_%H%M%S')}"

    logger.info("Running task #%s: %s (model=%s)", task_id, task.get("name"), task_model)
    _current_task_id = task_id
    started_at = now_iso

    try:
        from claude_code_sdk import query as sdk_query, ClaudeCodeOptions
        from prompt_builder import build_system_prompt

        system_prompt = build_system_prompt()

        # Get MCP server configs
        config = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
        mcp_servers = {}
        disallowed_tools = list(config.get("tools", {}).get("disabled_builtin", []))
        for name, cfg in config.get("mcp_servers", {}).items():
            if not cfg.get("enabled", True):
                continue
            server_type = cfg.get("type", "stdio")
            if server_type in ("sse", "http"):
                mcp_servers[name] = {"type": server_type, "url": cfg["url"]}
            else:
                mcp_servers[name] = {"command": cfg.get("command", "python"), "args": cfg.get("args", [])}
            for tool_name in cfg.get("disabled_tools", []):
                disallowed_tools.append(f"mcp__{name}__{tool_name}")

        options = ClaudeCodeOptions(
            model=task_model,
            system_prompt=system_prompt,
            max_turns=config.get("agent", {}).get("max_turns", 60),
            permission_mode="bypassPermissions",
            mcp_servers=mcp_servers,
            disallowed_tools=disallowed_tools,
        )

        # Set model env
        old_env = {}
        for key, value in model_env.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            final_response = ""

            async def _run():
                nonlocal final_response
                async for message in sdk_query(prompt=prompt, options=options):
                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "text"):
                                final_response += block.text

            asyncio.run(_run())
        finally:
            for key, old_val in old_env.items():
                if old_val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_val

        if not final_response:
            final_response = "(No response)"

        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        duration = (datetime.datetime.now(datetime.timezone.utc) - now).total_seconds()

        _write_run_record(
            task_id=task_id, run_id=run_id, status="success",
            started_at=started_at, completed_at=completed_at,
            duration_seconds=duration, summary=final_response[:200],
            body=f"## Prompt\n\n{prompt[:500]}...\n\n## Response\n\n{final_response}",
        )

        _update_task_field(task_id, status="up_next", last_run=completed_at,
                           updated=completed_at, failure_count=0)
        _append_activity_log(task_id, f"Run {run_id} — success ({duration:.0f}s)")

        logger.info("Task #%s completed in %.1fs", task_id, duration)
        return {
            "success": True, "task_id": task_id, "run_id": run_id,
            "duration_seconds": round(duration, 1),
            "response_preview": final_response[:300],
        }

    except Exception as e:
        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        duration = (datetime.datetime.now(datetime.timezone.utc) - now).total_seconds()
        error_msg = f"{type(e).__name__}: {e}"

        current_failures = int(task.get("failure_count") or 0)
        _update_task_field(task_id, status="up_next",
                           failure_count=current_failures + 1, updated=completed_at)

        _write_run_record(
            task_id=task_id, run_id=run_id, status="failed",
            started_at=started_at, completed_at=completed_at,
            duration_seconds=duration, summary=error_msg[:200],
            body=f"## Error\n\n```\n{error_msg}\n\n{traceback.format_exc()}\n```",
        )

        _append_activity_log(task_id, f"Run {run_id} — FAILED: {error_msg[:100]}")
        logger.error("Task #%s failed: %s", task_id, error_msg)
        return {"success": False, "task_id": task_id, "run_id": run_id,
                "error": error_msg, "duration_seconds": round(duration, 1)}

    finally:
        _current_task_id = None


# ── Tick ──────────────────────────────────────────────────────────────────────

def autonomy_tick() -> dict:
    global _ticker_running, _last_tick_time

    if not _ticker_enabled:
        return {"skipped": True, "reason": "disabled"}
    if _ticker_running:
        return {"skipped": True, "reason": "already_running"}

    _ticker_running = True
    _last_tick_time = time.time()

    try:
        due = get_due_tasks()
        if not due:
            return {"skipped": False, "reason": "no_tasks_due", "checked": True}

        task = due[0]
        task_id = task.get("id")
        logger.info("Tick: running task #%s (%s) — %d due total", task_id, task.get("name"), len(due))
        result = run_task(int(task_id))
        result["tasks_due"] = len(due)
        return result

    except Exception as e:
        logger.error("Tick failed: %s", e)
        return {"skipped": False, "error": str(e)}

    finally:
        _ticker_running = False
