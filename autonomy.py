"""Lloyd autonomy helpers — task-file I/O and single-task execution.

Scheduling, KG pipeline dispatch, and worker orchestration now all live in
the unified work queue (see workers/ and docs/21-unified-work-queue.md).
This module provides the task-file CRUD + `run_task()` that the
`scheduled-task` source and the `/api/autonomy/run` endpoint both call.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import subprocess
import logging
import os
import traceback
from pathlib import Path
from typing import Optional


# High-precision phrases that suggest a tool/subprocess failed even though the
# agent's turn completed successfully. Used by _detect_silent_failures() to flag
# runs whose summary should not be trusted as a clean success.
_SILENT_FAILURE_PATTERNS = [
    re.compile(r"\bfailed because\b", re.IGNORECASE),
    re.compile(r"\bTraceback \(most recent call last\)"),
    re.compile(r"\bexit code\s+[1-9]\d*\b"),
    re.compile(r"\b(?:FileNotFoundError|PermissionError|ModuleNotFoundError|"
               r"ImportError|KeyError|AttributeError|TypeError|ValueError)\b\s*:"),
]


def _detect_silent_failures(text: str) -> list[str]:
    """Return the failure-indicator snippets found in text, or []."""
    if not text:
        return []
    hits = []
    for pat in _SILENT_FAILURE_PATTERNS:
        m = pat.search(text)
        if m:
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 80)
            hits.append(text[start:end].strip().replace("\n", " "))
    return hits

import yaml

from agent_mcp._shared import parse_frontmatter_text

logger = logging.getLogger("lloyd-autonomy")

AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
AUTONOMY_RUNS_DIR = Path.home() / "lloyd" / "autonomy-runs"
LLOYD_HOME = Path(__file__).parent

def recover_stuck_tasks() -> list:
    """Reset any tasks stuck in_progress longer than their timeout."""
    recovered = []
    if not AUTONOMY_DIR.exists():
        return recovered
    now = datetime.datetime.now(datetime.timezone.utc)
    for path in AUTONOMY_DIR.glob("*.md"):
        if not re.match(r"\d+-", path.name):
            continue  # only NN-name.md task files; skip _config.md, reports, notes
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


# ── Task file I/O ─────────────────────────────────────────────────────────────

def _parse_task_file(path: Path) -> Optional[dict]:
    """Parse a task file with graduated recovery (shared parser in
    agent_mcp._shared): plain YAML → orphaned-tags repair → regex field
    extraction. A task can come back degraded (`_yaml_broken: True`) but it
    can never silently vanish from the scheduler — that failure mode
    dormant-killed 34/40 tasks on 2026-05-28 (see
    project_autonomy_silent_task_drop memory). The next yaml.dump write
    (e.g. _update_task_field) normalizes a repaired file on disk."""
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        fm = parse_frontmatter_text(
            parts[1],
            fallback_fields=(
                "id", "name", "description", "status", "priority", "frequency",
                "scheduled_at", "next_run", "last_run", "agent_id", "skill_name",
                "timeout_seconds", "preemptible", "auto_advance", "depends_on",
                "max_retries", "failure_count", "runs_per_day", "preferred_hours",
            ),
            log_label=f"scheduler:{path.name}",
        )
        fm["body"] = parts[2] if len(parts) > 2 else ""
        fm["_path"] = str(path)
        return fm
    except Exception as e:
        logger.error("Failed to parse %s: %s", path, e)
        return None


def _find_task_file(task_id) -> Optional[Path]:
    task_id = str(task_id)
    for path in AUTONOMY_DIR.glob("*.md"):
        if not re.match(r"\d+-", path.name):
            continue  # only NN-name.md task files; skip _config.md, reports, notes
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


# ── Scheduling logic (used by scheduled-task source) ──────────────────────────

def _all_runnable_tasks() -> list[dict]:
    if not AUTONOMY_DIR.exists():
        return []
    tasks = []
    for path in AUTONOMY_DIR.glob("*.md"):
        if not re.match(r"\d+-", path.name):
            continue  # only NN-name.md task files; skip _config.md, reports, notes
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


_no_skill_warned: set[str] = set()


def _is_dependency_met(task: dict, all_tasks: list[dict]) -> bool:
    dep_id = task.get("depends_on")
    if not dep_id or str(dep_id).strip().lower() in ("null", "none", ""):
        return True
    dep_id = str(dep_id).strip()
    dep_task = None
    for t in all_tasks:
        if str(t.get("id", "")).strip() == dep_id:
            dep_task = t
            break
    if not dep_task:
        return True
    dep_last_run = _parse_iso(dep_task.get("last_run"))
    if not dep_last_run:
        return False
    # Freshness gate: the dependency must have completed within the current
    # scheduling cycle (half this task's interval), not just "since my last
    # run". Without this, yesterday's upstream run satisfies the gate and the
    # nightly pipelines settle into a stable inverted order where downstream
    # tasks always consume day-old upstream artifacts (observed June 2026:
    # reflection ran 39→38/40→42, trajectory ran 57 before 56).
    interval = _frequency_interval_seconds(task) or 86400.0
    now = datetime.datetime.now(datetime.timezone.utc)
    if (now - dep_last_run).total_seconds() > interval / 2:
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
    # A task is runnable if it has a skill_name (slug) or a full skill_path.
    # If both are empty, skip it.
    skill_name = str(task.get("skill_name", "") or "").strip()
    skill_path = str(task.get("skill_path", "") or "").strip()
    if not skill_name and not skill_path:
        # Loud skip: an empty skill_name otherwise dead-letters the task
        # forever with no signal (bit task 79 in June 2026).
        task_id = str(task.get("id", "?"))
        if task_id not in _no_skill_warned:
            _no_skill_warned.add(task_id)
            logger.warning(
                "Task #%s (%s) has no skill_name/skill_path — it will NEVER "
                "run until one is set", task_id, task.get("name"))
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

def _load_skill_content(skill_name: str) -> Optional[str]:
    """Resolve a skill name (slug) or path to SKILL.md content.

    Priority:
      1. If skill_name looks like a filesystem path (contains / or .md), use it directly.
      2. Otherwise treat it as a slug → ~/obsidian/skills/<slug>/SKILL.md.
    """
    skill_name = str(skill_name or "").strip()
    if not skill_name or skill_name.lower() in ("null", "none"):
        return None

    # If it's already a path, try it directly first
    if "/" in skill_name or skill_name.endswith(".md"):
        expanded = Path(skill_name.replace("~", str(Path.home())))
        if expanded.exists():
            try:
                return expanded.read_text(encoding="utf-8")
            except Exception:
                return None
        # Path didn't exist — fall through to slug resolution below
        return None

    # Treat as slug: ~/obsidian/skills/<slug>/SKILL.md
    expanded = Path.home() / "obsidian" / "skills" / skill_name / "SKILL.md"
    if expanded.exists():
        try:
            return expanded.read_text(encoding="utf-8")
        except Exception:
            return None
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


async def run_task(task_id) -> dict:
    """Execute a single autonomy task via Claude Agent SDK."""
    path = _find_task_file(task_id)
    if not path:
        return {"success": False, "error": f"Task #{task_id} not found"}

    task = _parse_task_file(path)
    if not task:
        return {"success": False, "error": f"Failed to parse task #{task_id}"}

    skill_name = str(task.get("skill_name", "") or "").strip()
    # Backward compat: if skill_name is empty but skill_path exists (old format), use it
    skill_path = str(task.get("skill_path", "") or "").strip() if not skill_name else ""

    if skill_name:
        skill_content = _load_skill_content(skill_name)
        if not skill_content:
            return {"success": False, "error": f"Skill not found: {skill_name}"}
    elif skill_path:
        skill_content = _load_skill_content(skill_path)
        if not skill_content:
            return {"success": False, "error": f"Skill not found: {skill_path}"}
    else:
        return {"success": False, "error": f"Task #{task_id} has no skill_name or skill_path"}

    now = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat()
    run_id = f"run_{task_id}_{now.strftime('%Y%m%d_%H%M%S')}"

    _update_task_field(task_id, status="in_progress", updated=now_iso)
    prompt = _build_task_prompt(task, skill_content)

    # Resolve model
    task_model = str(task.get("model", "") or "").strip()
    if not task_model or task_model.lower() in ("null", "none"):
        try:
            cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
            task_model = cfg.get("model", {}).get("default", "")
        except Exception:
            pass
    if not task_model:
        task_model = ""

    model_env = _get_model_env(task_model)
    timeout = int(task.get("timeout_seconds") or 1800)

    logger.info("Running task #%s: %s (model=%s)", task_id, task.get("name"), task_model)
    started_at = now_iso
    # Capture tool/script failures that happen INSIDE the run (e.g. a Bash command
    # exiting non-zero). The harness returns these to the model as tool_results with
    # is_error=True rather than raising, so without this they never reach the run
    # record. Defined before the try so the except path can reference it safely.
    tool_errors: list[str] = []

    try:
        from app.harness import run_query, RunOptions
        from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_URL
        from prompt_builder import build_system_prompt

        system_prompt = build_system_prompt()

        config = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
        disallowed_tools: list[str] = []
        for name, cfg in config.get("mcp_servers", {}).items():
            for tool_name in cfg.get("disabled_tools", []):
                disallowed_tools.append(f"mcp__{name}__{tool_name}")

        options = RunOptions(
            model=task_model,
            base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
            system_prompt=system_prompt,
            max_turns=config.get("agent", {}).get("max_turns", 60),
            permission_mode="bypassPermissions",
            mcp_servers={"lloyd-mcp": {"type": "sse", "url": DEFAULT_LLOYD_MCP_URL}},
            disallowed_tools=disallowed_tools,
            env=model_env,
            priority=1,
        )

        messages = [{"role": "user", "content": prompt}]
        final_response = ""

        try:
            async with asyncio.timeout(timeout):
                async for evt in run_query(messages, options):
                    if evt["type"] == "text_delta":
                        final_response += evt["text"]
                    elif evt["type"] == "tool_result" and evt.get("is_error"):
                        content = evt.get("content")
                        if isinstance(content, list):
                            content = " ".join(
                                str(b.get("text", b)) if isinstance(b, dict) else str(b)
                                for b in content
                            )
                        tool_errors.append(str(content)[:600])
        except asyncio.TimeoutError:
            # Without this bookkeeping the task file stays in_progress on
            # disk with no run record or activity-log line — the run just
            # vanishes (recover_stuck_tasks was written for the fallout but
            # was never wired up).
            logger.warning("Task #%s timed out after %ds", task_id, timeout)
            completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            duration = (datetime.datetime.now(datetime.timezone.utc) - now).total_seconds()
            partial = f"## Partial response before timeout\n\n{final_response}" \
                if final_response else "(no output before timeout)"
            _write_run_record(
                task_id=task_id, run_id=run_id, status="failed",
                started_at=started_at, completed_at=completed_at,
                duration_seconds=duration,
                summary=f"timed out after {timeout}s",
                body=f"## Prompt\n\n{prompt[:500]}...\n\n{partial}",
            )
            current_failures = int(task.get("failure_count") or 0)
            _update_task_field(task_id, status="up_next",
                               failure_count=current_failures + 1,
                               updated=completed_at)
            _append_activity_log(
                task_id, f"Run {run_id} — FAILED: timed out after {timeout}s")
            return {
                "success": False, "error": f"Task #{task_id} timed out after {timeout}s",
            }

        if not final_response:
            final_response = "(No response)"

        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        duration = (datetime.datetime.now(datetime.timezone.utc) - now).total_seconds()

        silent_failures = _detect_silent_failures(final_response)
        body_parts = [f"## Prompt\n\n{prompt[:500]}...", f"## Response\n\n{final_response}"]
        if tool_errors:
            errs_md = "\n\n".join(f"```\n{e}\n```" for e in tool_errors[:10])
            body_parts.insert(1, f"## ⚠ Tool/script errors during run ({len(tool_errors)})\n\n{errs_md}")
        if silent_failures:
            indicators_md = "\n".join(f"- `{s}`" for s in silent_failures[:5])
            body_parts.insert(1, f"## ⚠ Silent failure indicators detected\n\n{indicators_md}")
            logger.warning(
                "Task #%s ran to completion but final response contains failure indicators: %s",
                task_id, silent_failures[:3],
            )

        _write_run_record(
            task_id=task_id, run_id=run_id, status="success",
            started_at=started_at, completed_at=completed_at,
            duration_seconds=duration, summary=final_response[:200],
            body="\n\n".join(body_parts),
        )

        interval = _frequency_interval_seconds(task)
        completed_dt = datetime.datetime.fromisoformat(completed_at)
        next_run_iso = (completed_dt + datetime.timedelta(seconds=interval)).isoformat() if interval else None
        _update_task_field(task_id, status="up_next", last_run=completed_at,
                           updated=completed_at, failure_count=0,
                           **({"next_run": next_run_iso} if next_run_iso else {}))
        if silent_failures:
            _append_activity_log(
                task_id,
                f"Run {run_id} — success ({duration:.0f}s) ⚠ silent-failure indicators: "
                f"{silent_failures[0][:120]}",
            )
        elif tool_errors:
            _append_activity_log(
                task_id,
                f"Run {run_id} — success ({duration:.0f}s) ⚠ {len(tool_errors)} "
                f"tool error(s); see autonomy-runs/{task_id}/{run_id}.md",
            )
        else:
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

        tool_errs_md = ""
        if tool_errors:
            joined = "\n\n".join(f"```\n{e}\n```" for e in tool_errors[-10:])
            tool_errs_md = f"\n\n## Tool/script errors before failure ({len(tool_errors)})\n\n{joined}"

        _write_run_record(
            task_id=task_id, run_id=run_id, status="failed",
            started_at=started_at, completed_at=completed_at,
            duration_seconds=duration, summary=error_msg[:200],
            body=f"## Error\n\n```\n{error_msg}\n\n{traceback.format_exc()}\n```{tool_errs_md}",
        )

        # Keep enough of the error to be diagnosable from the activity log itself,
        # and point at the full run record for the traceback + tool stderr.
        _append_activity_log(
            task_id,
            f"Run {run_id} — FAILED: {error_msg[:280]} "
            f"[full: autonomy-runs/{task_id}/{run_id}.md]",
        )
        logger.error("Task #%s failed: %s", task_id, error_msg)
        return {"success": False, "task_id": task_id, "run_id": run_id,
                "error": error_msg, "duration_seconds": round(duration, 1)}
