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


def _detect_silent_failures(text: str, expected: Optional[list] = None) -> list[str]:
    """Return the failure-indicator snippets found in text, or [].

    `expected` (task frontmatter `expected_error_patterns`) suppresses matches a
    task deliberately provokes — #48's dry-run is REQUIRED to raise
    FileNotFoundError while the graph is missing, which produced 33 false
    positives in a week and taught everyone to ignore the indicator.
    """
    if not text:
        return []
    patterns = []
    for pat in (expected or []):
        try:
            patterns.append(re.compile(str(pat), re.IGNORECASE))
        except re.error:
            patterns.append(None)  # fall back to substring below
    hits = []
    for pat in _SILENT_FAILURE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 80)
        snippet = text[start:end].strip().replace("\n", " ")
        suppressed = False
        for exp, compiled in zip(expected or [], patterns):
            if compiled is not None:
                if compiled.search(snippet):
                    suppressed = True; break
            elif str(exp).lower() in snippet.lower():
                suppressed = True; break
        if not suppressed:
            hits.append(snippet)
    return hits

import yaml

from agent_mcp._shared import parse_frontmatter_text

logger = logging.getLogger("lloyd-autonomy")

AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
from app.paths import AUTONOMY_RUNS_DIR  # anchored to LLOYD_HOME
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
                "scheduled_at", "next_run", "last_run", "last_attempt", "agent_id",
                "skill_name", "timeout_seconds", "preemptible", "auto_advance",
                "depends_on", "max_retries", "failure_count", "runs_per_day",
                "preferred_hours", "model", "stale_bypass_hours",
                "expected_error_patterns",
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
    # Use the graduated-recovery parser, not plain yaml.safe_load: a file whose
    # frontmatter is degraded (orphaned tags, etc.) would otherwise be
    # unwritable, and this function is called from the FAILURE handler — the
    # exact moment a task most needs its status and failure_count recorded.
    fm = parse_frontmatter_text(parts[1], log_label=f"update:{path.name}")
    if not isinstance(fm, dict):
        return
    fm = {k: v for k, v in fm.items() if not str(k).startswith("_")}
    fm.update(fields)
    # An explicit None clears the key rather than writing `key: null`.
    fm = {k: v for k, v in fm.items() if v is not None}
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


# ── Failure backoff ───────────────────────────────────────────────────────────
# A failed run used to leave `last_run` untouched, and `_is_task_due` gates only
# on `last_run` — so a task that timed out was due again on the very next 60s
# tick, forever. That produced 12 consecutive 600s timeouts on #36 in one night
# and ~130 on #69 over two days (~21 GPU-hours on a task whose script was gone).
# `failure_count` and `max_retries` were parsed, stored and displayed, but no
# scheduling decision read either.
_FAILURE_BACKOFF_BASE = 600          # 10 min, doubling per consecutive failure
_FAILURE_BACKOFF_CAP_SECONDS = 21600  # 6h floor for the cap
_DEFAULT_MAX_RETRIES = 5
# An empty response this fast, with no tool call, is the model server hiccuping
# (a thinking-only turn or a 200 with no content), not the task failing.
_INFRA_EMPTY_MAX_SECONDS = 15
# Keep the task-level timeout strictly under the pool's, so run_task's own
# handler wins the race and the run is always recorded.
_POOL_TIMEOUT_MARGIN = 30
_INFRA_EXC_NAMES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout", "PoolTimeout",
    "RemoteProtocolError", "ConnectionRefusedError", "ConnectionResetError",
})


def _failure_cooldown_seconds(task: dict) -> float:
    """Exponential backoff keyed on consecutive failures: 10m, 20m, 40m, ...
    capped at the task's own interval or 6h, whichever is larger."""
    n = max(1, int(task.get("failure_count") or 0))
    interval = _frequency_interval_seconds(task) or 86400.0
    cap = max(interval, _FAILURE_BACKOFF_CAP_SECONDS)
    return float(min(_FAILURE_BACKOFF_BASE * (2 ** (n - 1)), cap))


def _in_failure_cooldown(task: dict, now: datetime.datetime) -> bool:
    """True while a failed task is serving its cooldown.

    The last attempt failed iff `last_attempt` is newer than `last_run`.
    `last_run` deliberately keeps meaning "last SUCCESSFUL completion" — it
    feeds the dependency freshness gate, so bumping it on failure would let a
    broken upstream satisfy its downstream tasks.
    """
    last_attempt = _parse_iso(task.get("last_attempt"))
    if not last_attempt:
        return False
    last_run = _parse_iso(task.get("last_run"))
    if last_run and last_attempt <= last_run:
        return False  # most recent attempt succeeded
    return (now - last_attempt).total_seconds() < _failure_cooldown_seconds(task)


def _write_run_record(task_id: int, run_id: str, status: str,
                      started_at: str, completed_at: str,
                      duration_seconds: float, summary: str, body: str,
                      extra: Optional[dict] = None) -> Path:
    runs_dir = AUTONOMY_RUNS_DIR / str(task_id)
    runs_dir.mkdir(parents=True, exist_ok=True)
    fm = {
        "run_id": run_id, "task_id": task_id, "status": status,
        "started_at": started_at, "completed_at": completed_at,
        "duration_seconds": round(duration_seconds, 1), "summary": summary,
    }
    # stop_reason / usage / num_turns were never recorded, so a run cut off at
    # max_turns looked identical to one that finished its work.
    for k, v in (extra or {}).items():
        if v is not None:
            fm[k] = v
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
        # `failed` is included so a disabled upstream stays FINDABLE by
        # _is_dependency_met — otherwise the dependency lookup misses it and
        # returns True, letting dependents run off a broken upstream. It is
        # excluded from dispatch by the status gate in _is_task_due.
        if status in ("up_next", "in_progress", "failed"):
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


def _dependency_bypassed(task: dict, dep_task: dict,
                         dep_last_run: Optional[datetime.datetime],
                         now: datetime.datetime) -> bool:
    """Implement `stale_bypass_hours` — the documented "fail forward" rule.

    The field was set on #38/#40 and described in the architecture doc as
    letting a dependent run with stale input rather than blocking the chain,
    but nothing ever read it. Bypass only when the upstream is not actively
    running, so a merely-late upstream is still waited for.
    """
    try:
        bypass_hours = float(task.get("stale_bypass_hours") or 0)
    except (TypeError, ValueError):
        return False
    if bypass_hours <= 0:
        return False
    if str(dep_task.get("status", "")).strip() == "in_progress":
        return False
    if dep_last_run is None:
        return True
    return (now - dep_last_run).total_seconds() > bypass_hours * 3600


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
        # Never succeeded — still eligible for a stale bypass.
        return _dependency_bypassed(task, dep_task, None,
                                    datetime.datetime.now(datetime.timezone.utc))
    # Freshness gate: the dependency must have completed within the current
    # scheduling cycle (half this task's interval), not just "since my last
    # run". Without this, yesterday's upstream run satisfies the gate and the
    # nightly pipelines settle into a stable inverted order where downstream
    # tasks always consume day-old upstream artifacts (observed June 2026:
    # reflection ran 39→38/40→42, trajectory ran 57 before 56).
    interval = _frequency_interval_seconds(task) or 86400.0
    now = datetime.datetime.now(datetime.timezone.utc)
    if (now - dep_last_run).total_seconds() > interval / 2:
        return _dependency_bypassed(task, dep_task, dep_last_run, now)
    my_last_run = _parse_iso(task.get("last_run"))
    if not my_last_run:
        return True
    return dep_last_run > my_last_run


_HHMM_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})")


def _local_hour() -> int:
    """Current machine-local hour. Indirection exists so tests can pin it."""
    return datetime.datetime.now().hour


def _effective_preferred_hours(task: dict) -> Optional[list]:
    """preferred_hours, falling back to the hour in `scheduled_at`.

    Several tasks documented a schedule in `scheduled_at` (#60 "04:30", #81
    "05:00 is deliberate") while leaving `preferred_hours` null, so nothing
    enforced it — #81 ran at 17:57 and took the qmd daemon offline mid-afternoon.
    Hours are machine-local; a cron-expression scheduled_at yields no window.
    """
    pref = task.get("preferred_hours")
    if isinstance(pref, list) and len(pref) > 0:
        try:
            return [int(h) for h in pref]
        except (TypeError, ValueError):
            return None
    m = _HHMM_RE.match(str(task.get("scheduled_at") or ""))
    if m and int(m.group(1)) < 24:
        return [int(m.group(1))]
    return None


def _is_preferred_hour(task: dict) -> bool:
    hours = _effective_preferred_hours(task)
    if not hours:
        return True
    return _local_hour() in hours


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
    # Only up_next dispatches. in_progress previously stayed "due", so the same
    # task could be enqueued while a copy of it was still running; `failed` is in
    # the runnable set purely so dependency lookups can see it.
    if str(task.get("status", "")).strip() != "up_next":
        return False
    interval = _frequency_interval_seconds(task)
    if interval is None:
        return False
    last_run = _parse_iso(task.get("last_run"))
    now = datetime.datetime.now(datetime.timezone.utc)
    if last_run:
        elapsed = (now - last_run).total_seconds()
        # last_run is a COMPLETION time, so due-time drifts later by the run's
        # own duration every cycle. For a task pinned to a one-hour window that
        # drift eventually steps past the window and skips a day, so allow a
        # little slack when a window is in force.
        slack = min(3600.0, interval * 0.25) if _effective_preferred_hours(task) else 0.0
        if elapsed < interval - slack:
            return False
    # A failed run keeps last_run untouched, so without this gate the task is
    # due again on the next tick — the retry storm.
    if _in_failure_cooldown(task, now):
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


async def _record_failure(task: dict, task_id, run_id: str, started_at: str,
                          started_dt: datetime.datetime, *, summary: str, body: str,
                          kind: str = "task", extra: Optional[dict] = None,
                          alert: bool = True) -> dict:
    """Single failure path for run_task: run record, backoff, activity log, alert.

    `kind="task"` means the task itself failed (timeout, exception mid-run, an
    empty response after real work) and increments `failure_count`, escalating
    to `status: failed` at `max_retries`. `kind="infra"` means the model server
    hiccuped (fast empty response, connection error); it gets a flat cooldown
    and never counts toward the retry budget, so an outage can't disable the
    whole fleet — on 2026-09-01 every task returned empty for 11 hours straight.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    completed_at = now.isoformat()
    duration = (now - started_dt).total_seconds()

    _write_run_record(
        task_id=task_id, run_id=run_id, status="failed",
        started_at=started_at, completed_at=completed_at,
        duration_seconds=duration, summary=summary[:200], body=body,
        extra={**(extra or {}), "failure_kind": kind},
    )

    failures = int(task.get("failure_count") or 0)
    max_retries = int(task.get("max_retries") or _DEFAULT_MAX_RETRIES)
    fields: dict = {"status": "up_next", "last_attempt": completed_at,
                    "updated": completed_at}
    disabled = False
    if kind == "task":
        failures += 1
        fields["failure_count"] = failures
        if failures >= max_retries:
            fields["status"] = "failed"
            disabled = True

    if disabled:
        fields["next_run"] = None
    else:
        cooldown = (_failure_cooldown_seconds({**task, **fields}) if kind == "task"
                    else float(_FAILURE_BACKOFF_BASE))
        fields["next_run"] = (now + datetime.timedelta(seconds=cooldown)).isoformat()
    _update_task_field(task_id, **fields)

    note = f"Run {run_id} — FAILED ({kind}): {summary[:280]} [full: autonomy-runs/{task_id}/{run_id}.md]"
    if disabled:
        note += (f" — DISABLED after {failures} consecutive failures; "
                 f"set status back to up_next to re-enable")
    _append_activity_log(task_id, note)

    if disabled and alert:
        try:
            from app.discord_notify import discord_alert
            await discord_alert(
                f"Autonomy task #{task_id} ({task.get('name')}) disabled after "
                f"{failures} consecutive failures. Last: {summary[:300]}"
            )
        except Exception as e:
            logger.warning("Alert dispatch failed for task #%s: %s", task_id, e)

    logger.error("Task #%s failed (%s, %d/%d): %s", task_id, kind, failures,
                 max_retries, summary[:200])
    return {
        "success": False, "status": "failed", "task_id": task_id, "run_id": run_id,
        "error": summary, "duration_seconds": round(duration, 1),
        "failure_kind": kind, "disabled": disabled,
        "meta": {**(extra or {}), "failure_kind": kind, "disabled": disabled},
    }


async def run_task(task_id, *, max_duration: int | None = None) -> dict:
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

    # Refuse to start a second copy of a run that is already going. `_is_task_due`
    # never checked status and the manual/MCP entry points skip due-checks
    # entirely, so two runs of #38 once started 9 seconds apart and interleaved.
    if str(task.get("status", "")).strip() == "in_progress":
        updated = _parse_iso(task.get("updated"))
        stale_after = int(task.get("timeout_seconds") or 1800)
        if updated and (now - updated).total_seconds() < stale_after:
            msg = (f"Task #{task_id} is already in_progress (since "
                   f"{updated.isoformat()}); not starting a second run")
            logger.info("%s", msg)
            return {"success": False, "skipped": True, "status": "skipped",
                    "task_id": task_id, "error": msg}

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

    # Honour the `secondary_enabled` switch: when the secondary slot is off,
    # route its tasks back to primary rather than at a dead port. Everything
    # else in the codebase goes through this helper; the autonomy path did not,
    # so a task pinned to `secondary` would fail instead of falling back.
    try:
        from app.config import resolve_model_alias
        resolved = resolve_model_alias(task_model)
        if resolved != task_model:
            logger.info("Task #%s: model %s -> %s (secondary_enabled=false)",
                        task_id, task_model, resolved)
            task_model = resolved
    except Exception as e:
        logger.warning("Model alias resolution failed for #%s: %s", task_id, e)

    model_env = _get_model_env(task_model)
    declared_timeout = int(task.get("timeout_seconds") or 1800)
    timeout = declared_timeout
    if max_duration:
        # Stay strictly under the caller's cap. When the two were equal the POOL
        # timer won, cancelling this coroutine before its own handler could run:
        # no run record, no activity-log line, task left in_progress on disk.
        timeout = max(60, min(declared_timeout, int(max_duration) - _POOL_TIMEOUT_MARGIN))
        if declared_timeout > timeout:
            logger.warning(
                "Task #%s timeout_seconds=%ds exceeds the caller cap %ds; using %ds",
                task_id, declared_timeout, max_duration, timeout)

    logger.info("Running task #%s: %s (model=%s, timeout=%ds)",
                task_id, task.get("name"), task_model, timeout)
    started_at = now_iso
    # Capture tool/script failures that happen INSIDE the run (e.g. a Bash command
    # exiting non-zero). The harness returns these to the model as tool_results with
    # is_error=True rather than raising, so without this they never reach the run
    # record. Defined before the try so the except path can reference it safely.
    tool_errors: list[str] = []

    try:
        from app.harness import run_query, RunOptions
        from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
        from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
        from prompt_builder import build_system_prompt

        system_prompt = build_system_prompt()

        config = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
        # Resolve the tool surface through the same helpers the chat and voice
        # routers use (app/routers/messages.py:1243). Two things were wrong
        # with building it here by hand:
        #
        #   * the raw yaml.safe_load bypassed ${VAR} expansion and
        #     data/tool_overrides.yaml — the same defect the 2026-09-04 review
        #     fixed in builtin_task, in a second location;
        #   * tool_search kwargs were never passed, so tool_search_baseline
        #     stayed empty and the harness fell back to _DEFAULT_BASELINE_TOOLS
        #     (Bash, Read, Write, Edit, Grep, Glob, Task). Every autonomy run
        #     therefore had Bash permanently visible while http_search and
        #     http_fetch sat behind a ToolSearch round-trip — and the nightly
        #     research jobs are what generate the trajectories the skill miner
        #     learns from, so the bias fed itself.
        disallowed_tools = _get_disallowed_tools()

        options = RunOptions(
            model=task_model,
            base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
            system_prompt=system_prompt,
            max_turns=config.get("agent", {}).get("max_turns", 60),
            permission_mode="bypassPermissions",
            mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
            disallowed_tools=disallowed_tools,
            env=model_env,
            priority=1,
            **_get_harness_kwargs(),
        )

        messages = [{"role": "user", "content": prompt}]
        final_response = ""
        stop_reason = None
        usage = None
        num_turns = None
        saw_tool_call = False

        try:
            async with asyncio.timeout(timeout):
                async for evt in run_query(messages, options):
                    if evt["type"] == "text_delta":
                        final_response += evt["text"]
                    elif evt["type"] == "tool_call":
                        saw_tool_call = True
                    elif evt["type"] == "result":
                        stop_reason = evt.get("stop_reason")
                        usage = evt.get("usage")
                        num_turns = evt.get("num_turns")
                    elif evt["type"] == "tool_result" and evt.get("is_error"):
                        content = evt.get("content")
                        if isinstance(content, list):
                            content = " ".join(
                                str(b.get("text", b)) if isinstance(b, dict) else str(b)
                                for b in content
                            )
                        tool_errors.append(str(content)[:600])
        except asyncio.TimeoutError:
            logger.warning("Task #%s timed out after %ds", task_id, timeout)
            partial = (f"## Partial response before timeout\n\n{final_response}"
                       if final_response else "(no output before timeout)")
            errs = ("\n\n## Tool/script errors before timeout\n\n"
                    + "\n\n".join(f"```\n{e}\n```" for e in tool_errors[-5:])
                    ) if tool_errors else ""
            return await _record_failure(
                task, task_id, run_id, started_at, now,
                summary=f"timed out after {timeout}s",
                body=f"## Prompt\n\n{prompt[:500]}...\n\n{partial}{errs}",
                kind="task",
                extra={"timeout": True, "timeout_seconds": timeout,
                       "stop_reason": stop_reason, "usage": usage,
                       "num_turns": num_turns, "tool_errors": len(tool_errors)},
            )
        except asyncio.CancelledError:
            # The worker pool cancels via asyncio.wait_for. CancelledError is a
            # BaseException, so neither the timeout branch nor `except Exception`
            # below used to catch it: the run vanished with no record at all and
            # the task file was left in_progress until recover_stuck_tasks found
            # it. Record it, then re-raise so cancellation still propagates.
            duration = (datetime.datetime.now(datetime.timezone.utc) - now).total_seconds()
            logger.warning("Task #%s cancelled after %.0fs", task_id, duration)
            partial = (f"## Partial response before cancellation\n\n{final_response}"
                       if final_response else "(no output before cancellation)")
            await _record_failure(
                task, task_id, run_id, started_at, now,
                summary=f"cancelled by the worker pool after {duration:.0f}s",
                body=f"## Prompt\n\n{prompt[:500]}...\n\n{partial}",
                kind="task", alert=False,
                extra={"cancelled": True, "stop_reason": stop_reason,
                       "usage": usage, "num_turns": num_turns,
                       "tool_errors": len(tool_errors)},
            )
            raise

        duration = (datetime.datetime.now(datetime.timezone.utc) - now).total_seconds()

        # An empty response used to be relabelled "(No response)" and recorded as
        # a SUCCESS: last_run advanced, failure_count reset, dependents unblocked.
        # That is how #79 (retention) went dark for a week on 0.6s "successes",
        # and how ~180 phantom runs passed during the 2026-09-01 empty window.
        if not final_response.strip():
            infra = (not saw_tool_call) and duration < _INFRA_EMPTY_MAX_SECONDS
            kind = "infra" if infra else "task"
            summary = (f"empty response after {duration:.0f}s "
                       f"(stop_reason={stop_reason}, turns={num_turns}, "
                       f"tool_errors={len(tool_errors)})")
            errs = ("\n\n## Tool/script errors\n\n"
                    + "\n\n".join(f"```\n{e}\n```" for e in tool_errors[:10])
                    ) if tool_errors else ""
            return await _record_failure(
                task, task_id, run_id, started_at, now,
                summary=summary,
                body=(f"## Prompt\n\n{prompt[:500]}...\n\n## Response\n\n"
                      f"(empty — the model returned no text){errs}"),
                kind=kind,
                extra={"empty": True, "stop_reason": stop_reason, "usage": usage,
                       "num_turns": num_turns, "tool_errors": len(tool_errors),
                       "saw_tool_call": saw_tool_call},
            )

        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        silent_failures = _detect_silent_failures(
            final_response, task.get("expected_error_patterns"))
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

        is_silent = final_response.strip() == "[SILENT]"
        meta = {"stop_reason": stop_reason, "usage": usage, "num_turns": num_turns,
                "tool_errors": len(tool_errors), "silent": is_silent,
                "silent_failure_indicators": len(silent_failures)}

        _write_run_record(
            task_id=task_id, run_id=run_id, status="success",
            started_at=started_at, completed_at=completed_at,
            duration_seconds=duration, summary=final_response[:200],
            body="\n\n".join(body_parts), extra=meta,
        )

        interval = _frequency_interval_seconds(task)
        completed_dt = datetime.datetime.fromisoformat(completed_at)
        next_run_iso = (completed_dt + datetime.timedelta(seconds=interval)).isoformat() if interval else None
        # last_attempt tracks EVERY attempt; last_run only successes. The
        # cooldown gate reads "last_attempt newer than last_run" as "the most
        # recent attempt failed", so a success must set both.
        _update_task_field(task_id, status="up_next", last_run=completed_at,
                           last_attempt=completed_at,
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

        logger.info("Task #%s completed in %.1fs (stop_reason=%s)",
                    task_id, duration, stop_reason)
        return {
            "success": True, "status": "success", "task_id": task_id, "run_id": run_id,
            "duration_seconds": round(duration, 1),
            "response_preview": final_response[:300],
            "meta": meta,
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        tool_errs_md = ""
        if tool_errors:
            joined = "\n\n".join(f"```\n{err}\n```" for err in tool_errors[-10:])
            tool_errs_md = (f"\n\n## Tool/script errors before failure "
                            f"({len(tool_errors)})\n\n{joined}")
        # A model-server outage shouldn't burn the retry budget of every task.
        kind = "infra" if type(e).__name__ in _INFRA_EXC_NAMES else "task"
        return await _record_failure(
            task, task_id, run_id, started_at, now,
            summary=error_msg,
            body=f"## Error\n\n```\n{error_msg}\n\n{traceback.format_exc()}\n```{tool_errs_md}",
            kind=kind,
            extra={"exception": type(e).__name__, "tool_errors": len(tool_errors)},
        )


# ── Fleet health ──────────────────────────────────────────────────────────────
# Nothing aggregated run outcomes before this: /api/autonomy/runs required a
# task_id, so the 237 pool-timeout rows with a NULL task_id (73.6 GPU-hours)
# were unreachable by design, and no view existed for failure rate, GPU-hours,
# [SILENT] rate or consecutive failures. A task that timed out on every single
# run was indistinguishable from a healthy one.

def _iter_task_files() -> list[dict]:
    """Every NN-*.md task, whatever its status (including paused/failed)."""
    out = []
    if not AUTONOMY_DIR.exists():
        return out
    for path in AUTONOMY_DIR.glob("*.md"):
        if not re.match(r"\d+-", path.name):
            continue
        task = _parse_task_file(path)
        if task:
            out.append(task)
    return out


def _row_task_id(row: dict) -> Optional[str]:
    tid = row.get("task_id")
    if tid:
        return str(tid)
    payload = row.get("queue_payload_json")
    if payload:
        try:
            pid = json.loads(payload).get("task_id")
            if pid is not None:
                return str(pid)
        except (ValueError, TypeError):
            pass
    return None


def compute_health(rows: list[dict], tasks: list[dict], days: int) -> dict:
    """Aggregate run rows into per-task and fleet health. Pure function."""
    by_task: dict[str, dict] = {}
    task_by_id = {str(t.get("id")): t for t in tasks}

    for row in rows:
        status = row.get("status")
        if status == "skipped":
            continue
        tid = _row_task_id(row) or "unattributed"
        try:
            meta = json.loads(row.get("meta_json") or "{}")
        except (ValueError, TypeError):
            meta = {}
        summary = str(row.get("summary") or "")
        response = str(row.get("response_json") or "")
        duration = float(row.get("duration_seconds") or 0.0)

        # Historical rows predate meta_json, and empty responses were recorded
        # as successes — reclassify them so the numbers reflect reality.
        empty = bool(meta.get("empty")) or response.strip() == "(No response)" \
            or summary.strip() == "(No response)"
        timeout = bool(meta.get("timeout")) or bool(meta.get("pool_timeout")) \
            or "timed out" in summary or summary.startswith("TimeoutError")
        failed = status != "success" or empty
        silent = "[SILENT]" in response or bool(meta.get("silent"))

        e = by_task.setdefault(tid, {
            "task_id": tid, "runs": 0, "successes": 0, "failures": 0,
            "timeouts": 0, "empty": 0, "silent": 0, "silent_indicator_runs": 0,
            "max_turns_runs": 0, "tool_error_runs": 0,
            "gpu_hours": 0.0, "wasted_hours": 0.0, "total_seconds": 0.0,
            "max_seconds": 0.0, "consecutive_failures": 0, "last_success": None,
            "_streak_open": True,
        })
        e["runs"] += 1
        e["total_seconds"] += duration
        e["max_seconds"] = max(e["max_seconds"], duration)
        e["gpu_hours"] += duration / 3600.0
        if failed:
            e["failures"] += 1
            e["wasted_hours"] += duration / 3600.0
            if e["_streak_open"]:
                e["consecutive_failures"] += 1
        else:
            e["successes"] += 1
            e["_streak_open"] = False
            if not e["last_success"]:
                e["last_success"] = row.get("completed_at")
        if timeout:
            e["timeouts"] += 1
        if empty:
            e["empty"] += 1
        if silent:
            e["silent"] += 1
        if meta.get("stop_reason") == "max_turns":
            e["max_turns_runs"] += 1
        if int(meta.get("silent_failure_indicators") or 0) > 0:
            e["silent_indicator_runs"] += 1
        if int(meta.get("tool_errors") or 0) > 0:
            e["tool_error_runs"] += 1

    out_tasks = []
    for tid, e in by_task.items():
        e.pop("_streak_open", None)
        runs = e["runs"] or 1
        e["fail_rate"] = round(e["failures"] / runs, 3)
        e["silent_rate"] = round(e["silent"] / runs, 3)
        e["avg_seconds"] = round(e["total_seconds"] / runs, 1)
        e["gpu_hours"] = round(e["gpu_hours"], 2)
        e["wasted_hours"] = round(e["wasted_hours"], 2)
        e["max_seconds"] = round(e["max_seconds"], 1)
        e.pop("total_seconds", None)
        t = task_by_id.get(tid)
        if t:
            e.update({
                "name": t.get("name"), "status": t.get("status"),
                "frequency": t.get("frequency"),
                "failure_count": int(t.get("failure_count") or 0),
                "timeout_seconds": t.get("timeout_seconds"),
                "last_run": t.get("last_run"), "last_attempt": t.get("last_attempt"),
            })
        else:
            e["name"] = "(unattributed)" if tid == "unattributed" else f"task {tid}"
            e["status"] = "unknown"
        out_tasks.append(e)

    out_tasks.sort(key=lambda x: x["wasted_hours"], reverse=True)

    # Tasks with a file but no runs in the window are worth seeing too.
    seen = {t["task_id"] for t in out_tasks}
    idle = [{"task_id": str(t.get("id")), "name": t.get("name"),
             "status": t.get("status"), "frequency": t.get("frequency"),
             "runs": 0, "successes": 0, "failures": 0, "timeouts": 0, "empty": 0,
             "silent": 0, "fail_rate": 0.0, "silent_rate": 0.0, "gpu_hours": 0.0,
             "wasted_hours": 0.0, "avg_seconds": 0.0, "max_seconds": 0.0,
             "consecutive_failures": 0, "last_success": None,
             "failure_count": int(t.get("failure_count") or 0),
             "last_run": t.get("last_run")}
            for t in tasks if str(t.get("id")) not in seen]

    total_runs = sum(t["runs"] for t in out_tasks)
    total_fail = sum(t["failures"] for t in out_tasks)
    return {
        "days": days,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fleet": {
            "runs": total_runs,
            "failures": total_fail,
            "fail_rate": round(total_fail / total_runs, 3) if total_runs else 0.0,
            "gpu_hours": round(sum(t["gpu_hours"] for t in out_tasks), 2),
            "wasted_hours": round(sum(t["wasted_hours"] for t in out_tasks), 2),
            "empty_runs": sum(t["empty"] for t in out_tasks),
            "timeout_runs": sum(t["timeouts"] for t in out_tasks),
            "active_tasks": len([t for t in tasks if str(t.get("status")) == "up_next"]),
            "failed_tasks": [str(t.get("id")) for t in tasks
                             if str(t.get("status")) == "failed"],
            "paused_tasks": [str(t.get("id")) for t in tasks
                             if str(t.get("status")) == "paused"],
        },
        "tasks": out_tasks,
        "idle_tasks": idle,
    }
