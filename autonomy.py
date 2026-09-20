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
                "skill_name", "timeout_seconds", "max_turns", "preemptible", "auto_advance",
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


# ── Status changes that stop a task dispatching ───────────────────────────────
# Dispatch reads exactly two things off a task's `status`: `_all_runnable_tasks`
# drops anything outside `RUNNABLE_STATUSES`, and the
# queue's own source (`workers/sources/scheduled_task.py`) skips anything
# that is not `up_next`. So every other value — `draft` and `paused` above all —
# is a dispatch kill switch that leaves the task looking configured: live
# `next_run`, normal-looking board row, no log line saying it was parked. #68
# (`frequency: every-15min`, the fleet's highest-volume task) sat in `draft` for
# ~30 h — ~120 missed cycles — after a vault pre-flight commit flipped it there,
# and no record named the change or its author. Hence: report it before the
# commit (`scripts/util/autonomy_status_findings.py`) and record it when written
# (`status_change_note`, used by every writer that is not the scheduler).
DISPATCH_STOPPING_STATUSES = ("draft", "paused")


def status_change_note(old_status, new_status) -> Optional[str]:
    """An activity-log note for a task status transition, or None if there is none.

    Returns None when the new status is missing or equals the old, so a writer
    that only touched other front matter adds no line. A transition into a
    dispatch-stopping status is labelled, because that is the one a reader cannot
    infer from the file: the task still looks scheduled.
    """
    old = str(old_status or "").strip()
    new = str(new_status or "").strip()
    if not new or old == new:
        return None
    note = f"status changed: {old or '(no status recorded)'} -> {new}"
    if new in DISPATCH_STOPPING_STATUSES:
        note += " (dispatch-stopping)"
    return note


def append_activity_line(body: str, note: str, now_str: str) -> str:
    """Append one `- <ts>: <note>` line to a body's `## Activity Log` heading.

    The same shape `_append_activity_log` writes for the scheduler, for callers
    that hold the body in memory instead of the file: both non-scheduler writers
    serialise the whole task file themselves, so they append here and still write
    exactly once. Creates the heading when the body has none.
    """
    text = body or ""
    log_line = f"\n- {now_str}: {note}\n"
    if "## Activity Log" in text:
        return text.rstrip() + log_line
    return text.rstrip() + "\n\n## Activity Log\n" + log_line


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

# The statuses dispatch may run a task in, in ONE place. `_all_runnable_tasks`
# filters on it for the dispatch candidate; `_is_dependency_met` filters on it for
# the upstream (#558). Those were two ad-hoc literals in two functions, and "may
# this task run?" and "may this task certify its dependent's input?" are the same
# question with the same answer — a dependent consuming an artifact its upstream
# is not allowed to produce is fail-open with a log line attached. When the lists
# diverged on 2026-09-08 the scheduler could not see a `paused` upstream,
# `if not dep_task: return True` fired, and #39/#40 dispatched at 06:00-06:03Z
# against a vacuously-satisfied gate while #42's handoff landed at 06:10Z.
# `tests/test_autonomy_dependency_fail_closed.py::test_the_rung_that_decides_a_
# dependent_is_the_rung_that_decides_a_run` asserts the two agree status by
# status, so this tuple cannot drift from the filter silently again.
#
# `failed` is IN this set on Alan's ruling when #558 was reopened (2026-09-13):
# exhausting a retry budget stops a task running AGAIN, it does not unmake the
# artifact its last run produced. Whether that artifact is still usable is the
# freshness rule's job, not the status word's. `failed` is nonetheless excluded
# from DISPATCH — by `_is_task_due`'s own status gate, not by this tuple — and
# that asymmetry is asserted, not implied.
RUNNABLE_STATUSES = ("up_next", "in_progress", "failed")


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
    with no file resolves to nothing here, and `_is_dependency_met` is what
    decides that a resolution which cannot name its upstream does not certify the
    run (#558, on Alan's ruling that a `depends_on` pointing at no task file
    holds its dependent). That verdict is why the whole-board membership below
    matters as much as the shared parse: `paused` and `draft` upstreams are in
    this set, so they are found and then judged by their status, rather than
    silently resolving to nothing.

    `directory` is for the one caller whose board lives somewhere else (see
    `_all_board_tasks`); it is still the same function, the same membership rule,
    and the same parse.
    """
    return _all_board_tasks(directory)


def _dispatch_blockers(task: dict) -> list[str]:
    """Why this task is not something dispatch may run. Empty list means it may.

    ONE predicate behind two questions: "may this task run?"
    (`_all_runnable_tasks`) and "may this task certify the artifact its
    dependent is about to consume?" (`_is_dependency_met`, #558). They are the
    same question — an upstream that is not permitted to execute cannot have
    produced the input downstream is reading — so they must not be two ad-hoc
    lists that can drift. The drift is what burned the 2026-09-08 nightly chain:
    dispatch could not see the `paused` upstream, `if not dep_task: return True`
    fired, and #39/#40 consumed an analysis that had not been written yet.

    The messages are the warning text clause 4 asks for, so they name what was
    found (the status, or the grant errors) rather than just "unavailable".
    """
    blockers: list[str] = []
    status = str(task.get("status", "") or "").strip()
    if status not in RUNNABLE_STATUSES:
        blockers.append(f"status {status or '(blank)'!r}, which dispatch does not run")
    errors = _grant_block_errors(task, Path(str(task.get("_path") or "")))
    if errors:
        blockers.append("unreadable grants: block — " + "; ".join(errors[:2]))
    return blockers


_fail_closed_found: dict = {}


def _warn_fail_closed(task: dict, dep_id: str, finding: str) -> None:
    """Warn once per (dependent, upstream, finding) that a chain is held shut.

    The hold has to be legible: a dependent that silently never runs is the #68
    failure mode one layer up, and the 2026-09-08 inversion was undiagnosable for
    exactly that reason — nothing said "#39 ran while #42 was paused".

    Deduplicated on the FINDING rather than on the pair, so this repeats at most
    once per episode instead of every 60 s dispatch tick, while a change in what
    was found (paused → draft, or parked → deleted) still prints. `_no_skill_warned`
    set the precedent for deduplicating a dispatch-tick warning; that one dedupes
    forever, which is right for a missing skill and wrong here — a recurrence of
    the parked-upstream shape is the news this warning exists to report, so the
    entry is dropped the moment the upstream resolves and runnable again.
    """
    key = (str(task.get("id", "")), dep_id)
    finding = f"upstream #{dep_id}: {finding}"
    if _fail_closed_found.get(key) == finding:
        return
    _fail_closed_found[key] = finding
    logger.warning("Task #%s is held: its depends_on %s — dispatching it would "
                   "consume an artifact that will not be produced. Set "
                   "stale_bypass_hours on the dependent to forward on stale "
                   "input instead of waiting forever.",
                   task.get("id"), finding)


def _clear_fail_closed(task: dict, dep_id: str) -> None:
    """Forget a resolved fail-closed warning so its recurrence can warn again."""
    _fail_closed_found.pop((str(task.get("id", "")), dep_id), None)


def _all_runnable_tasks(board: Optional[list[dict]] = None) -> list[dict]:
    """Dispatch CANDIDATES: board tasks a status filter and #534's gate let run.

    Not the dependency resolution set — see `dependency_resolution_set`. The
    two roles were one list until #870, which is exactly how the scheduler came
    to be blind to a paused upstream. Pass `board` to filter a snapshot already
    in hand, so one caller does not read the directory twice and get two
    different boards out of it.

    The filter itself is `_dispatch_blockers`, shared with the dependency gate so
    the two cannot answer "is this task able to run?" differently (#558).
    """
    tasks = []
    for task in (board if board is not None else _all_board_tasks()):
        # `failed` is in `RUNNABLE_STATUSES` so a disabled upstream stays
        # FINDABLE by _is_task_due's own status gate — it is excluded from
        # dispatch there, not dropped here, so a broken upstream still reads as
        # an upstream, and its last SUCCESS still counts (#558 clause 6, amended).
        if _dispatch_blockers(task):
            continue
        # #534: a `grants:` block the loader cannot read is not the same thing
        # as a task with no grants. The block IS the human's authorization, so
        # parsing it badly and running anyway is fail-open with a log line
        # attached — the task runs durable-external actions under an authority
        # nobody actually wrote. Drop it from the runnable set: it is not due,
        # it is not enqueued, it does not drain. The fix is legible in the
        # error and the file is one edit away.
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


def next_run_gap(task: dict,
                 now: Optional[datetime.datetime] = None) -> dict:
    """How far this task is behind its OWN declared frequency, as two numbers.

    ONE predicate, shared by the two surfaces that had to agree (#1121): the
    scheduler's next_run stall scan and `compute_health`. Before this they were
    one private copy of the arithmetic each, which is exactly how the aggregate
    view came to score #68 `fail_rate 0.0` while the alert beside it — had it
    seen the task at all — would have named it. A board cannot be healthy in one
    surface and stalled in the other when both read this function.

    The two numbers, and why both are needed:

    * `hours_since_last_run` — elapsed against `last_run`, window-independent.
      A run-count window cannot see a task that stopped running: #68 scored
      `runs 30 / successes 30` over a 1-day window while dark for half of it,
      because the 30 runs that did happen were all successes.
    * `gap_ratio` — that elapsed divided by the declared period, so the number
      scales with the task's own cadence instead of a threshold somebody
      guesses: #68 at `every-15min` is past 50 periods by lunchtime.

    `past_next_run` is the stall predicate, and it stays keyed on `next_run` for
    the reason #421 shipped it that way. `last_run`-based staleness crosses 1.0
    at the moment a task becomes due, so a `gap_ratio > 1` alarm fires on every
    on-cadence nightly job — measured on the live board 2026-09-18, healthy
    task #51 reads 1.01 on that reference while #68 reads 169.4. `next_run` is
    written at completion, so being a whole period past it means a full extra
    period went by. The bound is strict and unchanged: > one interval.

    A task that has never run has no `last_run`. Then `gap_ratio` is measured
    against `next_run` and floored at 0.0, so "weekly, not yet due" reads 0.0
    and "daily, never ran, three periods past" reads 3.0 — and `never_run` says
    which shape the reader is looking at, because `run_count: 0` on its own
    cannot tell an overdue job from one that has simply never been due.

    A task with neither stamp yields None in every elapsed field: no verdict,
    not a healthy verdict. Nothing here reports a rate it cannot compute."""
    if now is None:
        now = _utcnow()
    interval = _frequency_interval_seconds(task)
    last = _parse_iso(task.get("last_run"))
    nxt = _parse_iso(task.get("next_run"))
    hours_since_last = (now - last).total_seconds() / 3600.0 if last else None
    hours_past_next = (now - nxt).total_seconds() / 3600.0 if nxt else None
    reference = hours_since_last
    if reference is None and hours_past_next is not None:
        reference = max(0.0, hours_past_next)
    gap_ratio = (round(reference * 3600.0 / interval, 2)
                 if interval and reference is not None else None)
    return {
        "expected_interval_seconds": interval,
        "hours_since_last_run": (round(hours_since_last, 2)
                                 if hours_since_last is not None else None),
        "hours_past_next_run": (round(hours_past_next, 2)
                                if hours_past_next is not None else None),
        "gap_ratio": gap_ratio,
        "never_run": last is None,
        # Strictly more than one period, exactly the bound the alert has used
        # since #421. Widening WHICH statuses are scanned is #1121's change;
        # widening this would make the alarm the noisy one all over again.
        "past_next_run": bool(interval and hours_past_next is not None
                              and hours_past_next * 3600.0 > interval),
    }


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


_never_ran_bypass_warned: dict = {}


def _warn_never_ran_bypass(dependent: dict, dep_task: dict,
                           bypass_hours: float) -> None:
    """Warn once per (dependent, upstream) that a bypass forwards on nothing.

    The verdict on this branch is deliberately unchanged by #814 — fail-forward
    is what the field exists to do — so the only way anyone learns that a
    dependent is consuming an artifact nobody has ever produced is this line.
    `hold_reason` cannot carry it: it is binary held/not-held, a bypassed
    dependent IS due, and `test_board_and_scheduler_agree_for_every_upstream_status`
    pins that board and scheduler agree on exactly that.

    Deduplicated on the (dependent, upstream) pair for the same reason
    `_warn_fail_closed` dedupes: dispatch re-answers this every 60 s tick, and a
    warning that prints 1,440 times a day is read by nobody. Unlike that one, the
    ledger here is dropped as soon as the upstream has ANY `last_run`, so a chain
    that recovers and then loses its upstream again warns again — the recurrence
    is the news.
    """
    key = (str(dependent.get("id", "")), str(dep_task.get("id", "")))
    if _never_ran_bypass_warned.get(key):
        return
    _never_ran_bypass_warned[key] = True
    logger.warning(
        "Task #%s bypasses its depends_on upstream #%s, which has NEVER RUN: "
        "stale_bypass_hours=%s on the dependent forwards on an artifact that "
        "has never been produced. This branch has no elapsed-time condition — "
        "there is no upstream last_run to measure the window against — so any "
        "window at all lets the dependent dispatch.",
        dependent.get("id"), dep_task.get("id"), bypass_hours)


def _dependency_bypassed(dependent: dict, dep_task: dict,
                         dep_last_run: Optional[datetime.datetime],
                         now: datetime.datetime) -> bool:
    """Implement `stale_bypass_hours` — the documented "fail forward" rule.

    WHOSE FIELD THIS IS: `stale_bypass_hours` is read from the DEPENDENT — the
    task whose `depends_on` is being gated — never from the upstream. The
    parameter used to be called `task` while this docstring talked about #38 and
    #40, which are UPSTREAMS; `stale_bypass_hours: 36` written on #38, a root
    whose own `depends_on` is null, is read by nothing, which is why the
    #38→#42 edge had no fail-forward path at all and a completed 12,700-byte
    report stalled the nightly chain ~10 h on 2026-09-10/11.

    TWO BRANCHES THAT ARE DELIBERATELY DIFFERENT. An upstream that HAS RUN is
    bypassed only once `now` sits further past its `last_run` than the
    dependent's window: there the field is a real elapsed-time condition. An
    upstream that has NEVER RUN (`last_run` empty — brand new, renamed, or never
    succeeded; or no task file answering the id at all) bypasses IMMEDIATELY,
    with no elapsed-time condition of any kind, because there is no timestamp to
    compare and any window at all forwards. #814 considered the alternative —
    treat never-ran as infinitely stale and hold the first cycle — and rejected
    it: the field exists to stop exactly the stall described above, a fresh or
    renamed chain is at its most fragile on its first cycle, and #870 already
    pinned the fail-forward verdict
    (`test_both_gate_branches_under_one_pinned_instant`, `assert met[6] is True`),
    so flipping it means editing a done item's assertion, not this function. What
    #814 does change is that this branch is no longer silent: it logs one WARNING
    naming the upstream and saying it has never run.

    The upstream being `in_progress` holds the dependent from BOTH branches, and
    the check sits above them on purpose: an upstream running its first-ever run
    is an artifact in progress, not a missing one.
    """
    try:
        bypass_hours = float(dependent.get("stale_bypass_hours") or 0)
    except (TypeError, ValueError):
        return False
    if bypass_hours <= 0:
        return False
    if str(dep_task.get("status", "")).strip() == "in_progress":
        return False
    if dep_last_run is None:
        _warn_never_ran_bypass(dependent, dep_task, bypass_hours)
        return True
    key = (str(dependent.get("id", "")), str(dep_task.get("id", "")))
    _never_ran_bypass_warned.pop(key, None)
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

    FAILS CLOSED on an upstream that cannot be the source of the artifact
    (#558): no task file answers the id, or the file exists but
    `_dispatch_blockers` says dispatch will not run it — a parked status
    (`paused`, `draft`, `suspended`, a blank status) or an unreadable `grants:`
    block. Both answer NOT met and log a warning naming the upstream and what
    was found. The rule is Alan's ruling on the reopened item (2026-09-13), not
    this round's judgement: a `depends_on` naming nothing is the same shape as a
    deleted upstream, and under the old `return True` both silently unblocked
    every dependent of a task that will never produce its input. That is the
    2026-09-08 nightly chain: #42 `paused`, #39/#40 dispatched against a
    vacuously-satisfied gate, #42's handoff written 6 minutes later.

    `stale_bypass_hours` is the escape and it still works, past its own window
    and never while the upstream is `in_progress` — fail closed must not become
    fail forever (principle 3: forwarding on stale input beats not running). The
    one status that IS accepted with a warning-free pass is `failed`: dispatch
    will not run it again, its last SUCCESS still names a real artifact, and the
    freshness rule below is what decides whether that artifact is usable.
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
        # No task file answers this id: a typo, a deleted upstream, or a file
        # whose front matter the board could not parse. There is no `last_run` to
        # measure freshness against, so the declared bypass window is the only
        # signal there is — hence the synthetic dep_task, which
        # `_dependency_bypassed` reads for its `in_progress` guard alone.
        _warn_fail_closed(task, dep_id, "no task file answers this depends_on id")
        return _dependency_bypassed(task, {"id": dep_id, "status": "unknown"},
                                    None, now)
    blockers = _dispatch_blockers(dep_task)
    if blockers:
        _warn_fail_closed(task, dep_id, "; ".join(blockers))
        return _dependency_bypassed(task, dep_task,
                                    _parse_iso(dep_task.get("last_run")), now)
    _clear_fail_closed(task, dep_id)
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


# Budget anchors, because an autonomy run is bounded by TWO clocks.
#
# The wall clock is `asyncio.timeout(timeout_seconds)`. Nothing told the model
# it existed until #80: `validate_okf.py` takes 2.4s, the budget was 300s, and
# three consecutive runs on 2026-09-08 spent all of it investigating and
# reported nothing. A timeout the model cannot see is a deadline it cannot meet.
#
# The iteration clock is `RunOptions.max_turns` — `agent.max_turns` below, which
# is 60. The comment here used to read that an autonomy run "is bounded by
# neither of those — it dies on `asyncio.timeout(timeout_seconds)`", and for the
# population that is now common that was false: `app/harness/loop.py` breaks
# with `stop_reason="max_turns"` the iteration past the cap, and
# `app/harness/finalizer.py` records no verdict for that death at all.
# `workers.db` holds 20 scheduled-task runs that died exactly there between
# 2026-09-04 and 2026-09-18, six of them task #39 — whose 2400s wall clock puts
# the deadline anchor's first level at 1680s, while all six died inside 1114s.
# The warning they needed was the chat path's, and it lived as a closure inside
# `app/routers/messages.py` with nothing to import (#1061).
#
# Both builders are in `app.deadline_anchor`, shared with the chat path, so each
# warning is one string rather than two that drift.
from app.deadline_anchor import (
    build_deadline_anchor,
    build_iteration_anchor,
    compose_state_anchors,
)


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


def _resolve_task_max_turns(task: dict, global_max_turns: int) -> int:
    """The iteration budget this one task gets, per task (#823).

    The task's own frontmatter wins — the same way `timeout_seconds` does — and
    `agent.max_turns` is the fallback for a task that declares nothing, which is
    every task on the fleet today, so nothing's budget moves when this lands.

    Why a per-task key rather than a bigger global: `agent.max_turns` is not an
    autonomy-only knob. `app/routers/messages.py:157` and `app/routers/voice.py:127`
    take the same key as the default iteration budget for an interactive and a
    voice turn, so raising 60 to spare one nightly job re-bounds every chat turn
    on the primary too. `run_task` is the only site that can hand one job a
    larger budget without costing the fleet anything.

    A declared value is FLOORED at the global, never honoured as-is when it is
    smaller. Two task files carry a number that cannot mean an iteration budget —
    `autonomy/48-entity-resolution-sweep.md` declares `max_turns: 6` and
    `84-fact-improvement.md` declares `max_turns: 4`, each added believing the key
    capped something else — and a too-small budget is precisely the failure this
    item exists to fix: `app/harness/loop.py` breaks at the cap,
    `app/harness/finalizer.py` records no verdict for that death, and the run
    reports nothing after minutes of GPU. Honouring a 4 would hand two working
    tasks that exact death, so the mismatch is logged at WARNING naming the task
    and both numbers instead of being applied or swallowed.

    The other direction is unbounded on purpose: a declared 5,000 is still cut
    off by `asyncio.timeout(timeout_seconds)` around the run, so the wall clock —
    not the iteration cap — is what bounds the GPU minutes an operator pays for.
    """
    task_id = task.get("id", "?")
    raw = task.get("max_turns")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return global_max_turns
    try:
        declared = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "Task #%s: max_turns=%r is not a number; running at the global "
            "agent.max_turns=%d instead", task_id, raw, global_max_turns)
        return global_max_turns
    if declared <= 0:
        logger.warning(
            "Task #%s: max_turns=%d is not a positive iteration budget; running "
            "at the global agent.max_turns=%d instead",
            task_id, declared, global_max_turns)
        return global_max_turns
    if declared < global_max_turns:
        logger.warning(
            "Task #%s declares max_turns=%d, below the global agent.max_turns=%d; "
            "running it at %d — a turn budget under the fleet default is never "
            "applied as-is (#823)",
            task_id, declared, global_max_turns, global_max_turns)
        return global_max_turns
    return declared


def _build_deadline_anchor(timeout_s: int):
    """The task-flavoured wall-clock anchor. One definition, in
    `app.deadline_anchor` — a second caller (the autocode worker turn) needs
    the identical behaviour, and two copies of "how close is the deadline"
    drift in the direction nobody is watching."""
    return build_deadline_anchor(timeout_s, what="task")


def _build_task_anchor(timeout_s: int, max_turns: int):
    """The one `state_anchor` a scheduled task gets, carrying both of its clocks.

    `timeout_s` must be the clamped value `asyncio.timeout` actually receives
    and `max_turns` the value handed to `RunOptions` — each warning is a
    fraction of one of those, and a fraction of a budget the run does not have
    arrives either too late to act on or after the run is already dead.

    A run with no wall clock still has a turn cap and still dies on it, so this
    returns an iteration-only anchor rather than `None`; `None` only when
    neither budget exists. That is the whole of #1061 — the wall-clock half has
    been here since #80, and 20 runs between 2026-09-04 and 09-18 died at
    `turns=61` with nothing said about iterations.
    """
    # Iteration levels first, then the wall clock — the order the chat path has
    # always emitted them in, so one run reading both messages sees them the way
    # every chat turn has.
    return compose_state_anchors(
        build_iteration_anchor(max_turns),
        _build_deadline_anchor(timeout_s),
    )


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


# ── Declared output artifact as evidence for run status (#832) ────────────────
#
# Run status used to be derived from one thing: whether the terminal assistant
# block had any characters in it. That block does not always exist. When the
# agent loop's last act is a tool call there is no text block after it, and five
# recorded runs ended that way — 2026-09-11 #38 (248 s, 19 turns, `signals-latest.md`
# written complete), 09-14 #57 (215 s, verdict ledger line landed), 09-16 #39,
# and 09-17 #39 twice (615.7 s / 52,579 output tokens and a 439.4 s retry), each
# of which committed a full nightly knowledge write under git. Every one was
# recorded `failed`, because `"".strip()` is empty.
#
# The cost is not the wrong word in a log line. `_record_failure` writes
# `last_attempt` and never `last_run`, and `_is_dependency_met` consumes only
# `last_run` — so an upstream that WORKED was indistinguishable from one that
# died, and the dependent in the nightly chain waited for a duplicate GPU run
# that survived only because it happened to sign off with prose.
#
# The 2026-09-01 change that stopped recording an empty response as a SUCCESS
# was correct and stays: this adds a second piece of evidence for the runs that
# demonstrably produced their deliverable, it does not restore any leniency. A
# run with no declared artifact, none on disk, a stub, or one older than its own
# start is still `failure_kind: task` and still leaves `last_run` absent.
_ARTIFACT_MIN_BYTES = 512
# Measured on this box 2026-09-19 over the declared outputs of the four nightly
# reflection tasks: the two truncated stubs are `knowledge-write-2026-09-14.md`
# (68 B) and `knowledge-write-2026-09-16.md` (69 B, front matter still
# `status: in-progress`, written by a 61-turn run scored `success` on non-empty
# preamble text), while every complete write in the window is at least 5,741 B
# (knowledge-write 5,741-14,153; knowledge-handoff 20,929-35,489; learnings
# 9,141-13,143). 512 B therefore sits ~11x under the smallest complete
# deliverable (5,741) and ~7x over the largest stub (69), which is the only place
# a floor like this can live
# without being either decorative or a coin flip. `wc -c` the two directories
# above before moving the number.
_DATE_MACRO = "{date}"


def _artifact_candidates(declared: str,
                         run_started: datetime.datetime) -> list[Path]:
    """Path(s) the task's `output_artifact` template names for THIS run.

    `{date}` resolves to the run's start date. Local (`America/Los_Angeles`)
    first, because the fleet's date-keyed deliverables are local-dated —
    `scripts/extract-trajectories.py:23-28` fixes that explicitly after UTC
    bucketing misfiled sessions a day late. But #42's handoff lands ~22:15 local,
    which is already the next UTC day, and #38/#39/#42 all declare `{date}`, so
    the other date of the same instant is a candidate as well. That is not
    leniency: the mtime test in `_declared_artifact_evidence` still has to pass,
    so a candidate can only win by having been written DURING this run. Without
    both spellings the mechanism exists and never fires on the chain it was
    built for.
    """
    text = str(declared or "").strip()
    if not text:
        return []
    if _DATE_MACRO not in text:
        return [Path(text).expanduser()]
    out: list[Path] = []
    for day in {run_started.astimezone().strftime("%Y-%m-%d"),
                run_started.strftime("%Y-%m-%d")}:
        p = Path(text.replace(_DATE_MACRO, day)).expanduser()
        if p not in out:
            out.append(p)
    return out


def _declared_artifact_evidence(task: dict,
                                run_started: datetime.datetime) -> Optional[dict]:
    """Qualifying evidence that this run wrote its declared output, else None.

    Qualifying = the task declares `output_artifact`, the file exists, it is at
    least `_ARTIFACT_MIN_BYTES` (512) bytes, and its mtime is at or after the
    run's start. `>=` and not `>`: a deliverable whose last write lands on the
    same microsecond the run began was written by this run as much as one a
    microsecond later, and a boundary that depends on a coin flip is not a
    gate. Returns the path, the byte count and the mtime so the run record can
    name its own evidence instead of merely asserting a status.

    `run_started` is when THIS run started, never the current time — "the file
    exists" is not the property; "this run produced it" is. Yesterday's
    8,918-byte knowledge write is real and says nothing about tonight.
    """
    declared = task.get("output_artifact")
    if not declared:
        return None
    for path in _artifact_candidates(str(declared), run_started):
        try:
            st = path.stat()
        except OSError:
            continue  # missing, or an unreadable ancestor: no evidence either way
        if st.st_size < _ARTIFACT_MIN_BYTES:
            continue
        modified = datetime.datetime.fromtimestamp(
            st.st_mtime, tz=datetime.timezone.utc)
        if modified < run_started:
            continue
        return {"path": str(path), "bytes": st.st_size,
                "modified_at": modified.isoformat()}
    return None


async def _record_artifact_success(task: dict, task_id, run_id: str,
                                   started_at: str,
                                   started_dt: datetime.datetime, duration: float,
                                   artifact: dict, *, stop_reason, usage,
                                   num_turns, tool_errors: list,
                                   session_id: str) -> dict:
    """Record a run whose terminal text was empty but whose declared deliverable
    is on disk and fresh. Single writer, so the run record, the activity line and
    the task fields can never disagree about which basis decided the status.
    """
    completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rel = artifact["path"]
    summary = (f"empty terminal text; status from artifact "
               f"{Path(rel).name} ({artifact['bytes']} B)")
    # Deliberately absent: `silent` and `silent_failure_indicators` are written
    # only when a terminal text block existed to scan, and #832's 4th instance is
    # precisely a run whose `silent_failure_indicators: 0` read as "checked and
    # clean" when there was nothing to scan. Absent means not evaluated; the
    # health report counts it as zero either way (`meta.get(...) or 0`).
    meta = {"stop_reason": stop_reason, "usage": usage, "num_turns": num_turns,
            "tool_errors": len(tool_errors),
            "session_id": session_id,
            # `empty` stays: the terminal block WAS empty, and that is a fact
            # worth querying for. `status_basis` is what makes the two kinds of
            # success distinguishable — a text-confirmed success carries no
            # `status_basis` at all.
            "empty": True, "status_basis": "artifact",
            "output_artifact": artifact}
    _write_run_record(
        task_id=task_id, run_id=run_id, status="success",
        started_at=started_at, completed_at=completed_at,
        duration_seconds=duration, summary=summary,
        body=(f"## Response\n\n(empty — the loop's last block was a tool call, "
              f"so there was no terminal text to read; #832)\n\n"
              f"## Status evidence\n\nRecorded SUCCESS on its declared output "
              f"artifact, not on terminal text:\n\n"
              f"- path: `{artifact['path']}`\n"
              f"- bytes: {artifact['bytes']} (floor {_ARTIFACT_MIN_BYTES})\n"
              f"- modified_at: {artifact['modified_at']}\n"
              f"- run started: {started_dt.isoformat()}\n"),
        extra=meta,
    )

    interval = _frequency_interval_seconds(task)
    completed_dt = datetime.datetime.fromisoformat(completed_at)
    next_run_iso = (completed_dt + datetime.timedelta(seconds=interval)).isoformat() if interval else None
    # Both stamps, exactly as on a text-confirmed success: `last_run` is what
    # `_is_dependency_met` reads, and the cooldown gate reads
    # "last_attempt newer than last_run" as "the most recent attempt failed".
    _update_task_field(task_id, status="up_next", last_run=completed_at,
                       last_attempt=completed_at, updated=completed_at,
                       failure_count=0,
                       **({"next_run": next_run_iso} if next_run_iso else {}))
    _append_activity_log(
        task_id,
        f"Run {run_id} — SUCCESS ({duration:.0f}s) ⚠ empty terminal text; "
        f"status from artifact {Path(rel).name} ({artifact['bytes']} B); see "
        f"autonomy-runs/{task_id}/{run_id}.md")
    logger.info("Task #%s completed in %.1fs on artifact evidence (%s, %d B, "
                "stop_reason=%s)", task_id, duration, Path(rel).name,
                artifact["bytes"], stop_reason)
    # `status` is "success", NOT a new "artifact_success" value. This dict
    # crosses into `workers/pool.normalize_result`, which validates it against
    # the closed set `RUN_STATUSES` and coerces anything else — logging "source
    # scheduled-task returned unknown status" and silently rewriting the value
    # (pool.py:135-141). An invented status would therefore have been renamed on
    # the way to the queue row anyway, so it buys nothing there, and it would
    # cost a warning on every occurrence plus a status that disagrees between the
    # two tables about what the same run was. The distinguishability clause 2
    # asks for lives in `status_basis`/`output_artifact` here and on the run
    # record's own front matter, which is the surface a human reads first, and
    # the text-confirmed path writes neither.
    result = {
        "success": True, "status": "success", "task_id": task_id,
        "run_id": run_id, "duration_seconds": round(duration, 1),
        "response_preview": "", "meta": meta,
    }
    if _evidence_pilot(task_id):
        result["claims"] = _evidence_claims("")
    return result


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

        # ONE number for the cap and for the warning about the cap. The harness
        # stops the run at `max_turns` (`app/harness/loop.py`), and the anchor
        # below counts down to it; read twice, the two drift and the warning
        # arrives at an iteration the run never reaches. The task's own
        # frontmatter decides that one number; `agent.max_turns` is both its
        # fallback and its floor, so a task that declares nothing is built
        # exactly as it was before this line existed (#823).
        global_max_turns = int(config.get("agent", {}).get("max_turns", 60))
        max_turns = _resolve_task_max_turns(task, global_max_turns)

        options = RunOptions(
            model=task_model,
            base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
            system_prompt=system_prompt,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
            disallowed_tools=disallowed_tools,
            env=model_env,
            priority=1,
            hooks=task_hooks,
            state_anchor=_build_task_anchor(timeout, max_turns),
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
                    # Nobody reads a scheduled task's reply either, so the
                    # unattended profile applies here for the same reason.
                    platform="autonomy", source=str(task.get("name") or ""),
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
        # The run's TERMINAL assistant block: the text of the last
        # `assistant_message` event that carried any. `final_response` is the
        # join of every `text_delta` of every iteration, so a run that narrated
        # while it worked ("Now let me classify the hits…") can never equal the
        # `[SILENT]` sentinel however it signed off — the contract the harness
        # prints in every autonomy prompt was unreachable for exactly the runs
        # that did work. 65 of the 104 task-#68 records from Sep 14-16 that
        # ended `## Response` on `[SILENT]` were written `silent: false`.
        #
        # `None` means "no block event arrived", which is what the pre-existing
        # `text_delta`-only test fixtures and any stream that ends on a
        # tool-call iteration produce; those fall back to the join rather than
        # reading an empty terminal block. Only non-empty blocks count, so a
        # run that signs off with a tool call and no prose is still judged on
        # the last thing it said.
        last_block: Optional[str] = None
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
                    elif evt["type"] == "assistant_message":
                        # One event per agent-loop iteration, carrying THAT
                        # block's own text (`app/harness/events.py`, emitted at
                        # `app/harness/loop.py` end-of-iteration). The last
                        # non-empty one is how the run ended.
                        if evt.get("text"):
                            last_block = evt["text"]
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
            # First evidence, and it is not the text: a task that declares its
            # deliverable and left it on disk, fresh, at 8,918 bytes, did the
            # work whatever the loop's last block happened to be. See
            # `_declared_artifact_evidence` for the five runs this is about and
            # for why the 2026-09-01 "an empty response is not a success" rule
            # is intact — with no qualifying artifact this falls straight
            # through to the failure paths below, unchanged.
            artifact = _declared_artifact_evidence(task, now)
            if artifact:
                return await _record_artifact_success(
                    task, task_id, run_id, started_at, now, duration, artifact,
                    stop_reason=stop_reason, usage=usage, num_turns=num_turns,
                    tool_errors=tool_errors, session_id=session_id)
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

        # What the run's answer IS, as opposed to everything it said on the way
        # to it. The transcript, the `## Response` body, the record's summary
        # and `response_preview` all stay on the join (those are #642's and
        # #832's to change); only the two readings below — a verdict about the
        # run's terminal message — move to the last block. An indicator phrase
        # in mid-run narration ("this probe failed because the port is closed")
        # used to brand a run that signed off `[SILENT]` a silent failure. The
        # mirror case — a run whose terminal block was the sentinel but which
        # had narrated first — is the one that recorded `silent: false`. The
        # substring form of the test lives in `workers/sources/scheduled_task.py`
        # against `response_preview` and is #642's to change, not this one's.
        terminal_text = final_response if last_block is None else last_block

        silent_failures = _detect_silent_failures(
            terminal_text, task.get("expected_error_patterns"))
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

        # The whole terminal block, never a substring of anything: the sentinel
        # is how a run declines to be surfaced, and a run that said the words
        # mid-flight and then reported normally is not that.
        is_silent = terminal_text.strip() == "[SILENT]"
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


_GAP_ROW_KEYS = ("expected_interval_seconds", "hours_since_last_run",
                 "hours_past_next_run", "gap_ratio", "never_run")


def _health_gap_fields(task: dict, now: datetime.datetime) -> dict:
    """The elapsed-time half of a health row, from `next_run_gap`.

    Carried on EVERY row — a task with runs in the window, a task with none, and
    a task that has never run — because the row this field exists to catch is
    not the one with 0 runs in it. `past_next_run` is deliberately left out of
    the row: the verdict lives in one place, the top-level `stalled` list, so a
    reader cannot be handed a number and a bool that disagree."""
    gap = next_run_gap(task, now=now)
    return {k: gap[k] for k in _GAP_ROW_KEYS}


def compute_health(rows: list[dict], tasks: list[dict], days: int,
                   now: Optional[datetime.datetime] = None) -> dict:
    """Aggregate run rows into per-task and fleet health. Pure function.

    `now` is the one instant every elapsed-time field is measured against, and
    it defaults to `_utcnow()` for the same reason `_is_task_due` takes one: the
    gap arithmetic below has to be askable at a chosen instant or it cannot be
    tested near a bound, and a fleet report that mixes two clocks is two
    reports. Production passes nothing — the route reads the wall clock once,
    here, and every field below inherits that single reading.
    """
    if now is None:
        now = _utcnow()
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
        # #832: a run whose empty terminal text was overruled by its declared
        # artifact sitting fresh on disk is recorded `success` and advances
        # `last_run`, so the health report has to agree with the scheduler or
        # the two surfaces disagree about the same row — and this reclassification
        # exists precisely to catch rows the scheduler got wrong, so a row it
        # decided on evidence it could read has to count as the run it says it
        # was. Every other empty row keeps its verdict, including the
        # pre-meta_json phantoms of the 2026-09-01 window and #79's dark week,
        # which carry no `status_basis` because no run record from that era has
        # a meta block at all.
        artifact_backed = (meta.get("status_basis") == "artifact"
                           and bool(meta.get("output_artifact")))
        timeout = bool(meta.get("timeout")) or bool(meta.get("pool_timeout")) \
            or "timed out" in summary or summary.startswith("TimeoutError")
        failed = status != "success" or (empty and not artifact_backed)
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
            e.update(_health_gap_fields(t, now))
        else:
            e["name"] = "(unattributed)" if tid == "unattributed" else f"task {tid}"
            e["status"] = "unknown"
            # No task file, so no declared frequency and no stamps: None, not a
            # 0.0 that would read as "on cadence" to anything summing this field.
            e.update({k: None for k in _GAP_ROW_KEYS[:-1]})
            e["never_run"] = False
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
             "last_run": t.get("last_run"),
             # An idle row is where a stopped task actually lands, so the gap
             # fields have to be here as much as on a row with runs: `runs: 0`
             # on its own cannot tell an overdue daily job from a weekly task
             # that has never yet been due.
             **_health_gap_fields(t, now)}
            for t in tasks if str(t.get("id")) not in seen]

    # The stall list, and the reason it is built from `tasks` rather than from
    # the rows in the window: a task that is not running is precisely the task
    # whose rows are missing, so anything derived from `by_task` could not
    # contain it — the shape #68 was in. Same predicate the scheduler's alert
    # uses (`next_run_gap`), same instant, so the report and the alert agree.
    # `runs_in_window` and `fail_rate` are carried ALONGSIDE the entry rather
    # than as a filter on it: #68's 30 in-window successes scored 0.0 and were
    # the reason nothing listed it. A passing rate suppresses nothing here.
    stalled = []
    for t in tasks:
        gap = next_run_gap(t, now=now)
        if not gap["past_next_run"]:
            continue
        row = by_task.get(str(t.get("id")))
        stalled.append({
            "task_id": str(t.get("id")), "name": t.get("name"),
            "status": t.get("status"), "frequency": t.get("frequency"),
            "expected_interval_seconds": gap["expected_interval_seconds"],
            "hours_since_last_run": gap["hours_since_last_run"],
            "hours_past_next_run": gap["hours_past_next_run"],
            "gap_ratio": gap["gap_ratio"],
            "never_run": gap["never_run"],
            "runs_in_window": (row["runs"] if row else 0),
            "fail_rate": (row["fail_rate"] if row else None),
            "hold": hold_reason(t, tasks, now=now),
        })
    stalled.sort(key=lambda s: (-(s["gap_ratio"] or 0.0), s["task_id"]))

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
        # Tasks more than one period past their own next_run, whatever their
        # status — the field whose absence let a `fail_rate: 0.0` stand in for
        # "healthy" about a job that had not run in ~50 cycles (#1121).
        "stalled": stalled,
    }
