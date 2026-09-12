"""Lloyd autonomy helpers — task-file I/O and single-task execution.

Scheduling, KG pipeline dispatch, and worker orchestration now all live in
the unified work queue (see workers/ and architecture/workers.md).
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
                "expected_error_patterns", "inner_voice",
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

def _all_board_tasks(directory=None) -> list[dict]:
    """Every parseable task file, whatever its status or grants block.

    This is the BOARD — the set `depends_on` gets resolved against (#870). An
    upstream that exists on disk is a fact about the schedule whether it is
    `paused`, `draft` or mid-flight; whether it may itself dispatch is a
    separate question that `_all_runnable_tasks` answers for the dispatch
    candidate only.
    """
    # `directory` exists because one caller — `app/routers/dashboard.py`, the
    # Mission Control landing strip — reads a vault-derived autonomy dir rather
    # than `AUTONOMY_DIR`. Without a parameter it had no way to ask for the
    # shared set, which is why it still had its own.
    root = AUTONOMY_DIR if directory is None else Path(directory)
    if not root.exists():
        return []
    tasks = []
    for path in sorted(root.glob("*.md")):
        if not re.match(r"\d+-", path.name):
            continue  # only NN-name.md task files; skip _config.md, reports, notes
        task = _parse_task_file(path)
        if not task:
            continue
        tasks.append(task)
    return tasks


def dependency_resolution_set(directory=None) -> list[dict]:
    """THE input `depends_on` is resolved against — scheduler and board alike.

    Both surfaces used to hand `_is_dependency_met` their own list: dispatch
    passed `_all_runnable_tasks()` (status-filtered), the Mission Control board
    passed every parsed task. With an upstream in `paused` the scheduler could
    not see it, `if not dep_task: return True` fired, and the dependent
    dispatched while the board printed `waiting on #N` for the same task in the
    same second. One resolution set, one answer, so the two surfaces cannot
    disagree — and no test can be written against one and pass while the other
    is broken.

    Deliberately NOT a decision about what an *absent* upstream means: an id
    with no file still resolves to nothing and is still treated as met. That
    boundary is #558's to move, not this function's.

    `directory` is for the one caller whose board lives somewhere else (see
    `_all_board_tasks`); it is still the same function, the same membership rule,
    and the same parse.
    """
    return _all_board_tasks(directory)


def _all_runnable_tasks(board: Optional[list[dict]] = None) -> list[dict]:
    """Dispatch CANDIDATES: board tasks a status filter and #534's gate let run.

    Not the dependency resolution set — see `dependency_resolution_set`. The
    two roles were one list until #870, which is exactly how the scheduler came
    to be blind to a paused upstream. Pass `board` to filter a snapshot already
    in hand, so one caller does not read the directory twice and get two
    different boards out of it.
    """
    tasks = []
    for task in (board if board is not None else _all_board_tasks()):
        status = str(task.get("status", "")).strip()
        # `failed` is included so a disabled upstream stays FINDABLE by
        # _is_task_due's own status gate — it is excluded from dispatch there,
        # not dropped here, so a broken upstream still reads as an upstream.
        if status not in ("up_next", "in_progress", "failed"):
            continue
        # #534: a `grants:` block the loader cannot read is not the same thing
        # as a task with no grants. The block IS the human's authorization, so
        # parsing it badly and running anyway is fail-open with a log line
        # attached — the task runs durable-external actions under an authority
        # nobody actually wrote. Drop it from the runnable set: it is not due,
        # it is not enqueued, it does not drain. The fix is legible in the
        # error and the file is one edit away.
        _errors = _grant_block_errors(task, Path(str(task.get("_path") or "")))
        if _errors:
            continue
        tasks.append(task)
    return tasks


def _grant_block_errors(task: dict, path: Path) -> list[str]:
    """Validate a task's declared `grants:` block; empty list means runnable."""
    if "grants" not in task:
        return []
    from app.harness.policy import validate_task_grants

    _, errors = validate_task_grants(task.get("grants"))
    if errors:
        logger.error(
            "Task #%s (%s) declares an unreadable grants: block and is NOT "
            "runnable — fix the block, the task stays stopped until then: %s",
            task.get("id"), path.name, "; ".join(errors))
    return errors


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


def _utcnow() -> datetime.datetime:
    """Current UTC instant. Indirection exists so tests can pin it.

    Same reason `_local_hour` below is a one-line function. The dependency gate
    read `datetime.datetime.now` twice inside its own body, so nothing could ask
    "would this task have been due at instant T?" and a case near the
    `interval / 2` freshness bound flipped depending on when the probe ran
    (#813). Pin this, and the whole due-decision path is reproducible.
    """
    return datetime.datetime.now(datetime.timezone.utc)


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


def _is_dependency_met(task: dict, all_tasks: list[dict], *,
                       now: Optional[datetime.datetime] = None) -> bool:
    """Is `task`'s `depends_on` satisfied right now?

    `all_tasks` is the resolution set — pass `dependency_resolution_set()` in
    production; a test passes the tasks it means to exist. `now` is the instant
    the answer is about: the gate read the wall clock inside its own body until
    #813, which made it not a function of its inputs — two probes of a case near
    the freshness bound could legitimately disagree, so neither the gate nor a
    differential over it could be replayed. One evaluation reads one instant.
    """
    if now is None:
        now = _utcnow()
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
        # Absent from the resolution set. Whether this should read as NOT met is
        # #558's open decision; this round deliberately keeps the existing
        # answer so the gate's agreement is not smuggled in as a verdict.
        return True
    dep_last_run = _parse_iso(dep_task.get("last_run"))
    if not dep_last_run:
        # Never succeeded — still eligible for a stale bypass.
        return _dependency_bypassed(task, dep_task, None, now)
    # Freshness gate: the dependency must have completed within the current
    # scheduling cycle (half this task's interval), not just "since my last
    # run". Without this, yesterday's upstream run satisfies the gate and the
    # nightly pipelines settle into a stable inverted order where downstream
    # tasks always consume day-old upstream artifacts (observed June 2026:
    # reflection ran 39→38/40→42, trajectory ran 57 before 56).
    interval = _frequency_interval_seconds(task) or 86400.0
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


def _is_task_due(task: dict, all_tasks: list[dict], *,
                 now: Optional[datetime.datetime] = None) -> bool:
    # `all_tasks` is the dependency resolution set, not the dispatch candidates
    # — pass `dependency_resolution_set()`. `now` is one instant for this whole
    # evaluation, so `hold_reason` can be asked the same question at the same
    # instant and get the same answer (#870).
    if now is None:
        now = _utcnow()
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
    if not _is_dependency_met(task, all_tasks, now=now):
        return False
    if not _is_preferred_hour(task):
        return False
    return True


def _hour_windows(hours) -> str:
    """[23,0,1,2,3,4] -> "00-04,23". Contiguous runs collapse; gaps survive."""
    try:
        hs = sorted({int(h) % 24 for h in hours})
    except (TypeError, ValueError):
        return ""
    out: list[str] = []
    start = prev = None
    for h in hs:
        if start is None or h != prev + 1:
            if start is not None:
                out.append(f"{start:02d}" if start == prev else f"{start:02d}-{prev:02d}")
            start = h
        prev = h
    if start is not None:
        out.append(f"{start:02d}" if start == prev else f"{start:02d}-{prev:02d}")
    return ",".join(out)


def hold_reason(task: dict, all_tasks: list[dict], *,
                now: Optional[datetime.datetime] = None) -> Optional[str]:
    """Why this task will not dispatch right now, or None if nothing holds it.

    Mirrors `autonomy._is_task_due`'s gates in its order and calls the very
    same predicates, because a second private definition of "due" is exactly
    how the Mission Control panel came to report six tasks overdue on a night
    when this scheduler considered none of them late. Four were nightly jobs
    sitting outside their `preferred_hours` window and two were `paused` — all
    six behaving precisely as configured. A nightly task is "late" for the
    eighteen hours a day it is not allowed to run, so `overdue_count` was
    never zero and therefore said nothing.

    The distinction any display has to draw is between *held* (something is
    deliberately keeping this task from running) and *overdue* (nothing is,
    and it still has not run). Only the second is worth a colour.
    """
    if (not str(task.get("skill_name") or "").strip()
            and not str(task.get("skill_path") or "").strip()):
        # `_is_task_due` warns once and skips forever. A task that can never
        # run is the one thing here that always deserves a human.
        return "no skill"
    status = str(task.get("status") or "").strip()
    if status != "up_next":
        return status or "no status"
    if _frequency_interval_seconds(task) is None:
        return "no frequency"
    if now is None:
        now = _utcnow()
    if _in_failure_cooldown(task, now):
        return "failure cooldown"
    if not _is_dependency_met(task, all_tasks, now=now):
        return f"waiting on #{task.get('depends_on')}"
    if not _is_preferred_hour(task):
        window = _hour_windows(_effective_preferred_hours(task) or [])
        return f"outside hours {window}" if window else "outside hours"
    return None


def _priority_key(task: dict) -> tuple:
    prio_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "background": 0}
    prio = prio_map.get(str(task.get("priority", "medium")).lower(), 2)
    last_run = _parse_iso(task.get("last_run"))
    overdue = (datetime.datetime.now(datetime.timezone.utc) - last_run).total_seconds() if last_run else 999999
    return (-prio, -overdue)


def get_due_tasks(*, now: Optional[datetime.datetime] = None) -> list[dict]:
    """Dispatch candidates, gated against the one dependency resolution set.

    Two inputs were one variable until #870: `_all_runnable_tasks()` was both
    the list to dispatch and the list `depends_on` was resolved against, while
    the board resolved against every task file. Same gate, different inputs,
    opposite answers for one task in one second. Candidates keep the status and
    grants filters (#534 fails a bad `grants:` block closed); resolution takes
    the whole board, so a `paused` upstream stays FINDABLE — the intent the
    `failed` status was let into the old runnable set to serve (see
    `dependency_resolution_set`).
    """
    if now is None:
        now = _utcnow()
    resolution = dependency_resolution_set()
    due = [t for t in _all_runnable_tasks(resolution)
           if _is_task_due(t, resolution, now=now)]
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


# ── Verifier-bound evidence claims (#525) ──────────────────────────────────
# A task's final text used to be the whole of its run record: `summary` is that
# text front-sliced, and nothing re-checked any number in it against the disk.
# The corrections log is the cost — the 08-24 handoff that reported the entity
# graph "restored to 12,131 relationships" against a same-night health report
# reading `| Total relationships | 0 |`, the 09-03 counts of 96/21 against 121/64
# on disk, the 09-04 "307KB" signals-latest.md that was 13,503 bytes. The fix so
# far has been a skill instruction ("verify on disk before claiming"), which
# binds only a willing model.
#
# The pilot is these four tasks and nothing else until the two-week numbers
# exist, which is why this is a literal set rather than a config key: widening it
# should be a change someone reads, and retiring it deletes four lines.
EVIDENCE_PILOT_TASK_IDS = frozenset({38, 42, 39, 40})


def _evidence_pilot(task_id) -> bool:
    try:
        return int(task_id) in EVIDENCE_PILOT_TASK_IDS
    except (TypeError, ValueError):
        return False


def _evidence_gap_list(task_id) -> list[str]:
    """The gap list this task's previous run left behind, if any.

    Read from the queue's watermarks, not from the task file: the gap list is
    ledger state, and `~/obsidian/autonomy/*.md` is a human-edited file that a
    worker has no business rewriting hourly.
    """
    try:
        from workers.evidence import gaps_key, parse_gap_list
        from workers.queue import WorkQueue, configured_db_path
        q = WorkQueue(configured_db_path())
        return parse_gap_list(q.wm_get("scheduled-task", gaps_key(task_id)))
    except Exception as e:
        # Unevaluable carry-forward is worth a log line, but never worth losing
        # the run over.
        logger.warning("could not read the evidence gap list for task #%s: %s",
                       task_id, e)
        return []


def _evidence_prompt(task_id, gaps: Optional[list[str]] = None) -> str:
    """The evidence section appended to a pilot task's prompt.

    Static instruction first, dynamic gap list last: the whole point of #520 is
    that this harness is prefix-cache-sensitive, so the part that never changes
    goes where it can stay cached and the part that changes goes at the tail.
    The system prompt is not touched at all.
    """
    if not _evidence_pilot(task_id):
        return ""
    from workers.evidence import CLAIMS_INSTRUCTION, gaps_prompt
    return CLAIMS_INSTRUCTION + gaps_prompt(
        _evidence_gap_list(task_id) if gaps is None else gaps)


def _evidence_claims(final_response: str) -> list[dict]:
    """Pull the claims block out of a finished run's own final text."""
    from workers.evidence import parse_claims_block
    return parse_claims_block(final_response)


# Wall-clock budget anchor. The chat path warns a turn that it is running out of
# ITERATIONS (app/routers/messages.py::_build_state_anchor, 75%/90% of
# max_turns); an autonomy run is bounded by neither of those — it dies on
# `asyncio.timeout(timeout_seconds)` — and nothing told the model that clock
# existed. So a run that had the answer in hand at t-1s was killed without ever
# being asked for it, and the run record read "(no output before timeout)".
#
# That is the whole of the 2026-09-08 failure of #80: `validate_okf.py` takes
# 2.4s, the budget was 300s, and three consecutive runs spent all of it
# investigating and reported nothing. A timeout the model cannot see is a
# deadline it cannot meet.
from app.deadline_anchor import build_deadline_anchor


def _task_inner_voice(task: dict) -> bool:
    """Whether the Inner Voice observer watches this task's run.

    Frontmatter beats config, and the fleet default is **off**. Recording is
    cheap and universal; observing is not — the observer runs on the PRIMARY
    at priority 1 and spends a goal-extraction call plus a critique per
    observed turn. At ~120 autonomy runs a day, defaulting it on would put
    that load behind every chat turn for runs nobody asked to have watched.

    An unset value on the task means "ask the fleet default", which is what
    makes `autonomy.inner_voice: true` in config.yaml a working master switch
    while a single task can still opt in on a fleet that is off.
    """
    raw = task.get("inner_voice")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("true", "yes", "on", "1"):
            return True
        if text in ("false", "no", "off", "0"):
            return False
        raw = None
    if isinstance(raw, bool):
        return raw
    try:
        # `app.config.CONFIG`, not a fresh `yaml.safe_load` of config.yaml —
        # the same rule `_common._worker_run_options` learned the hard way.
        # Reading the file directly skips `${VAR}` expansion, the
        # `LLOYD_CONFIG_OVERLAY` a canary boots with, and the
        # `data/tool_overrides.yaml` merge, so a canary run would resolve this
        # against production's answer rather than its own.
        from app.config import CONFIG
        return bool((CONFIG.get("autonomy") or {}).get("inner_voice", False))
    except Exception:
        return False


def _build_deadline_anchor(timeout_s: int):
    """The task-flavoured wall-clock anchor. One definition, in
    `app.deadline_anchor` — a second caller (the autocode worker turn) needs
    the identical behaviour, and two copies of "how close is the deadline"
    drift in the direction nobody is watching."""
    return build_deadline_anchor(timeout_s, what="task")


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
    failed = {
        "success": False, "status": "failed", "task_id": task_id, "run_id": run_id,
        "error": summary, "duration_seconds": round(duration, 1),
        "failure_kind": kind, "disabled": disabled,
        "meta": {**(extra or {}), "failure_kind": kind, "disabled": disabled},
    }
    # A failed pilot run still gets a bundle — an empty one, which the verifier
    # records as "nothing this run asserted was checked". Silence in the ledger
    # would look like an un-piloted task, and the pilot's own coverage would be
    # the first thing nobody could measure.
    if _evidence_pilot(task_id):
        failed["claims"] = []
    return failed


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
    # Every background run is recorded. This path used to leave nothing at
    # all — no transcript, no event log, no session — so the record of what a
    # scheduled task did was a 200-character summary, and on 2026-09-10 a task
    # could be neither confirmed nor cleared as the cause of a vault wipe
    # because its tool calls had never been written down.
    #
    # Recording is not observation: the session is created with Inner Voice
    # OFF unless the task asks for it, because the observer runs on the
    # primary and costs a goal-extraction call per turn, while the transcript
    # costs a few file appends.
    from app.sessions_io import create_session, new_background_session_id
    session_id = new_background_session_id("autonomy")

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
    # Appended payload only — see `_evidence_prompt` for the cache reasoning.
    prompt += _evidence_prompt(task_id)

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
    task_inner_voice = _task_inner_voice(task)
    try:
        create_session(
            session_id, platform="autonomy", model=task_model,
            title=f"#{task_id} {str(task.get('name') or '').strip()}"[:80],
            source=f"autonomy-task:{task_id}",
            inner_voice=task_inner_voice,
            preview=str(task.get("description") or task.get("name") or ""),
        )
    except Exception as exc:  # noqa: BLE001 — a record is not the run
        logger.warning("Task #%s: could not create its session %s: %s",
                       task_id, session_id, exc)
    started_at = now_iso
    # Capture tool/script failures that happen INSIDE the run (e.g. a Bash command
    # exiting non-zero). The harness returns these to the model as tool_results with
    # is_error=True rather than raising, so without this they never reach the run
    # record. Defined before the try so the except path can reference it safely.
    tool_errors: list[str] = []
    # Declared out here so the `finally` at the bottom can always close it,
    # including when the run fails before the observer is attached.
    iv_state = None

    try:
        from app.harness import run_query, RunOptions
        from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
        from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
        from prompt_builder import build_system_prompt

        system_prompt = build_system_prompt(platform="autonomy")

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

        # #534 — this is a SECOND dispatch path, not the worker one: run_task
        # builds its own RunOptions, so gating `_worker_run_options` alone
        # would have left every scheduled task running ungated. Re-validating
        # here matters because the manual and HTTP entry points skip the
        # scheduler's load-time check entirely.
        from app.harness import HookRegistry
        from app.harness.policy import (
            GRANT_MINT_TOOL, GrantError, default_store, install_policy_hook,
            sync_task_grants, validate_task_grants,
        )

        grant_scope = f"autonomy-task:{task_id}"
        _specs, _errors = validate_task_grants(task.get("grants"))
        if _errors:
            return {"success": False, "task_id": task_id,
                    "error": f"Task #{task_id} declares an unreadable grants: "
                             f"block; refusing to run ungated: "
                             + "; ".join(_errors)}
        try:
            if _specs:
                _new = sync_task_grants(default_store(), task_id=task_id,
                                        scope=grant_scope, grants=task["grants"])
                if _new:
                    logger.info("Task #%s: materialized %d declared grant(s)",
                                task_id, _new)
        except (GrantError, OSError) as e:
            return {"success": False, "task_id": task_id,
                    "error": f"Task #{task_id} could not materialize its "
                             f"grants: {e}"}

        disallowed_tools = list(disallowed_tools) + [
            GRANT_MINT_TOOL, f"mcp__lloyd-mcp__{GRANT_MINT_TOOL}"]

        task_hooks = HookRegistry()
        install_policy_hook(task_hooks, scope=grant_scope)

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
            hooks=task_hooks,
            state_anchor=_build_deadline_anchor(timeout),
            # Both, and both for a concrete reason. `session_id` routes this
            # run's tool-result spills under its own session instead of the
            # process-wide default; `turn_id` is what switches on the per-turn
            # change ledger (`agent_mcp/_change_ledger.py`), so a scheduled
            # task's file writes now leave pre-images and are revertable. An
            # unattended run is the one that most needs an undo, and it was
            # the only path that had none.
            session_id=session_id,
            turn_id=run_id,
            **_get_harness_kwargs(),
        )

        messages = [{"role": "user", "content": prompt}]
        # The Inner Voice observer, when this task asked for one. It attaches
        # to the direct path rather than needing the HTTP one:
        # `attach_observer_for_turn` creates a registry only if none exists,
        # so passing this run's `task_hooks` keeps the #534 grant gate AND
        # adds the observer. Its own gate fires for `turn_source="ambient"`
        # when the session's `inner_voice` is on, which is what
        # `create_session` wrote above.
        #
        # The ambient and clarify callbacks are None deliberately: both exist
        # to reach a human mid-turn, and nobody is reading. The observer's
        # remaining levers — inject, and the record it writes — are the ones
        # that mean anything here.
        #
        # Importing a router helper into this module is a layering smell.
        # Accepted rather than relocated: the alternative is a second
        # definition of "how a turn is watched", and the codebase has paid for
        # second definitions of "due", "healthy" and "tell the human" already.
        iv_cancel = asyncio.Event()
        if task_inner_voice:
            try:
                from app.routers._messages_inner_voice import (
                    attach_observer_for_turn,
                )
                iv_state = await attach_observer_for_turn(
                    session_id=session_id, turn_id=run_id,
                    turn_source="ambient", producer_source="autonomy",
                    user_request=prompt, options=options,
                    chat_messages_handle=messages, cancel_event=iv_cancel,
                )
                # The observer's `cancel` lever has to reach the loop or it is
                # a knob nothing reads. On the chat path `_run_turn` wires the
                # turn's own cancel event here; this path had none. It is the
                # lever that matters most for unattended work — a run going
                # somewhere it should not is the case this whole workstream
                # was opened for — and it is reachable only for a task that
                # asked to be watched.
                if iv_state is not None:
                    options.cancel_event = iv_cancel
            except Exception as exc:  # noqa: BLE001 — watching is not the run
                logger.warning("Task #%s: could not attach the observer: %s",
                               task_id, exc)
        final_response = ""
        stop_reason = None
        usage = None
        num_turns = None
        saw_tool_call = False

        try:
            from app.run_recorder import record_events
            recorded = record_events(
                run_query(messages, options),
                session_id=session_id, turn_id=run_id, prompt=prompt,
                model=task_model, source="autonomy",
            )
            async with asyncio.timeout(timeout):
                async for evt in recorded:
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
                       "num_turns": num_turns, "tool_errors": len(tool_errors),
                       "session_id": session_id},
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
                       "tool_errors": len(tool_errors),
                       "session_id": session_id},
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
                       "saw_tool_call": saw_tool_call,
                       "session_id": session_id},
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
                "silent_failure_indicators": len(silent_failures),
                # The join from a run record to the transcript of that run.
                # Without it the two records exist and nothing connects them.
                "session_id": session_id}

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
        result = {
            "success": True, "status": "success", "task_id": task_id, "run_id": run_id,
            "duration_seconds": round(duration, 1),
            "response_preview": final_response[:300],
            "meta": meta,
        }
        # Out unverified on purpose: the pool runs the checks when it writes the
        # row, so nothing that happened during the run — including this model's
        # own tool calls, which could have edited the file being claimed — can
        # affect the measurement of what the record asserts. An empty list is
        # still a bundle: "the model emitted no claims" is a gap the ledger has
        # to show, not an absence.
        if _evidence_pilot(task_id):
            result["claims"] = _evidence_claims(final_response)
        return result

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
            extra={"exception": type(e).__name__, "tool_errors": len(tool_errors),
                   "session_id": session_id},
        )
    finally:
        # Always. A turn that ends any way other than through the `result`
        # event leaves non-terminal judgments running against a dead turn,
        # where they write observation rows and can append an inject to a
        # message list nobody will read again. Synchronous by contract.
        if iv_state is not None:
            try:
                from app.routers._messages_inner_voice import close_observer
                close_observer(iv_state)
            except Exception as ce:  # noqa: BLE001 — cleanup must not raise
                logger.warning("close_observer failed for %s: %s", run_id, ce)


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


def _row_claims(row: dict) -> Optional[dict]:
    """Decode a run row's verified evidence bundle, or None when it has none.

    None means "no bundle" — a source outside the #525 pilot, a row from before
    it shipped, or a write that failed. It is never treated as a clean run: a
    task whose rows are all None reports a rate of None, because a metric that
    reads its own missing input as zero is the failure mode this file has now
    been burned on three times.
    """
    raw = row.get("claims_json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


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
            "runs_with_bundle": 0, "runs_without_bundle": 0,
            "claims_checked": 0, "claims_verified": 0, "claims_refuted": 0,
            "claims_insufficient": 0, "gap_runs": 0,
            "gpu_hours": 0.0, "wasted_hours": 0.0, "total_seconds": 0.0,
            "max_seconds": 0.0, "consecutive_failures": 0, "last_success": None,
            "_streak_open": True,
        })
        e["runs"] += 1
        # #525: the claims a run asserted and what the verifier found on disk.
        # Counted from the per-claim statuses rather than trusting the bundle's
        # own rate field, so a source cannot write its own score.
        bundle = _row_claims(row)
        if bundle is None:
            e["runs_without_bundle"] += 1
        else:
            e["runs_with_bundle"] += 1
            for claim in bundle.get("claims") or []:
                status = (str(claim.get("status") or "")
                          if isinstance(claim, dict) else "")
                if status == "verified":
                    e["claims_verified"] += 1
                elif status == "refuted":
                    e["claims_refuted"] += 1
                elif status == "insufficient":
                    e["claims_insufficient"] += 1
            if bundle.get("gap"):
                e["gap_runs"] += 1
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
        # Denominator is the claims that were CHECKED, not the runs. A [SILENT]
        # run asserted nothing, so counting it in a run-level denominator would
        # pull a task's refutation rate toward zero every time it said less —
        # which is how a fleet reports itself clean by doing nothing (#525).
        e["claims_checked"] = (e["claims_verified"] + e["claims_refuted"]
                               + e["claims_insufficient"])
        unverified = e["claims_refuted"] + e["claims_insufficient"]
        e["refuted_or_insufficient_rate"] = (
            round(unverified / e["claims_checked"], 3) if e["claims_checked"]
            else None)
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
             "runs_with_bundle": 0, "runs_without_bundle": 0,
             "claims_checked": 0, "claims_verified": 0, "claims_refuted": 0,
             "claims_insufficient": 0, "gap_runs": 0,
             "refuted_or_insufficient_rate": None,
             "consecutive_failures": 0, "last_success": None,
             "failure_count": int(t.get("failure_count") or 0),
             "last_run": t.get("last_run")}
            for t in tasks if str(t.get("id")) not in seen]

    total_runs = sum(t["runs"] for t in out_tasks)
    total_fail = sum(t["failures"] for t in out_tasks)
    total_claims = sum(t["claims_checked"] for t in out_tasks)
    total_unverified = (sum(t["claims_refuted"] for t in out_tasks)
                        + sum(t["claims_insufficient"] for t in out_tasks))
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
            # #525 pilot coverage next to the rate it produces. A window with a
            # low rate but most runs in `runs_without_bundle` is a window where
            # the rate describes a slice nobody chose — so the two are read
            # together or not at all.
            "claims_checked": total_claims,
            "claims_verified": sum(t["claims_verified"] for t in out_tasks),
            "claims_refuted": sum(t["claims_refuted"] for t in out_tasks),
            "claims_insufficient": sum(t["claims_insufficient"] for t in out_tasks),
            "runs_with_bundle": sum(t["runs_with_bundle"] for t in out_tasks),
            "runs_without_bundle": sum(t["runs_without_bundle"] for t in out_tasks),
            "refuted_or_insufficient_rate": (
                round(total_unverified / total_claims, 3) if total_claims else None),
            "active_tasks": len([t for t in tasks if str(t.get("status")) == "up_next"]),
            "failed_tasks": [str(t.get("id")) for t in tasks
                             if str(t.get("status")) == "failed"],
            "paused_tasks": [str(t.get("id")) for t in tasks
                             if str(t.get("status")) == "paused"],
        },
        "tasks": out_tasks,
        "idle_tasks": idle,
    }
