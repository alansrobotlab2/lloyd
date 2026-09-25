"""Lloyd autonomy helpers — task-file I/O and single-task execution.

Scheduling, KG pipeline dispatch, and worker orchestration now all live in
the unified work queue (see workers/ and architecture/workers.md).
This module provides the task-file CRUD + `run_task()` that the
`scheduled-task` source and the `/api/autonomy/run` endpoint both call.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import functools
import json
import re
import subprocess
import logging
import os
import threading
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

from agent_mcp._shared import AUTONOMY_TASK_FIELDS, parse_frontmatter_text

logger = logging.getLogger("lloyd-autonomy")

AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
from app.paths import AUTONOMY_RUNS_DIR  # anchored to DATA_ROOT
from app.run_acceptance import GRADE_KEY as _ACCEPTANCE_GRADE_KEY, DispatchTrace, grade_run
LLOYD_HOME = Path(__file__).parent

def recover_stuck_tasks() -> list:
    """Reset tasks stuck in_progress past their timeout, and rearms retired ones.

    Two recovery paths share this scan because they are the same shape — a task
    in a resting state with nothing that ever leaves it — and because the pool's
    tick and the startup hook already call exactly this one function
    (`workers/sources/scheduled_task.py`, `app/routers/autonomy.py`). A second
    scan would be a second place for the same predicate to drift.

    Returns the ids recovered by EITHER path.
    """
    recovered = []
    if not AUTONOMY_DIR.exists():
        return recovered
    now = datetime.datetime.now(datetime.timezone.utc)
    for path in AUTONOMY_DIR.glob("*.md"):
        if not re.match(r"\d+-", path.name):
            continue  # only NN-name.md task files; skip _config.md, reports, notes
        task = _parse_task_file(path)
        if not task:
            continue
        status = str(task.get("status", "")).strip()
        if status == "failed":
            # A task retired at max_retries has no other way back (#1086).
            if _rearm_after_one_period(task, now):
                recovered.append(task.get("id"))
            continue
        if status != "in_progress":
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
    (e.g. _update_task_field) normalizes a repaired file on disk.

    The field list is `AUTONOMY_TASK_FIELDS`, imported, not a local tuple: the
    MCP tool reader and the Mission Control board read the SAME files, and this
    tuple is what each of them recovers from a degraded one. When it was a
    private 25-field tuple here and a private 13-field tuple there, a broken
    file read by the MCP reader came back window-less/chain-less/model-less and
    — because that reader writes from its own read — got rewritten that way,
    while the scheduler kept dispatching the task on values only it could still
    see (#1014)."""
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        fm = parse_frontmatter_text(
            parts[1],
            fallback_fields=AUTONOMY_TASK_FIELDS,
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


# ── Activity Log entry shape ────────────────────────────────────────────────
# One note is one physical line, and a note repeated back-to-back is one counted
# entry. Both rules exist because the retention sweep bounds an Activity Log by
# counting *entry bullets* — lines beginning `- ` at column 0 — so a line that is
# not a bullet is content the cap never counts. It used to be content the sweep
# never removed either, which is how `autonomy/24-data-pipeline.md` came to carry
# 359 bare lines through every weekly sweep of that exact file while its own
# marker counted 6,841 entries pruned (#845). The sweep removes such lines now;
# these two rules are why it should never have to. The bare lines got here because
# a run summary is markdown prose and the failure note truncates it by characters
# (`f"Run … FAILED …: {summary[:280]}"`), so one failed run could write thirty.
ACTIVITY_LOG_HEADING = "## Activity Log"
#: Fold for a note's continuation lines. Visible in the log, greppable, and it
#: cannot re-introduce a line break the way the raw newline it replaces would.
ACTIVITY_NOTE_FOLD = " | "
#: What a collapsed entry ends with: `… ×3`. Read back on the next occurrence
#: so the counter increments instead of a second line appearing.
ACTIVITY_REPEAT_RE = re.compile(r"\s*×(\d+)\s*$")
#: The stamp of a note's own run id (`run_<task>_<YYYYMMDD>_<HHMMSS>`, built in
#: `run_task`), which a failure note embeds twice — once as the run, once
#: inside `[full: autonomy-runs/<task>/<run>.md]`.
_RUN_STAMP_RE = re.compile(r"\d{8}_\d{6}")
#: Any human-readable instant: the per-entry UTC stamp, an ISO timestamp in a
#: summary, or a bare date.
_ENTRY_STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?"
                             r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?")
#: Only a failed run earns the counter. Two successes are two facts about the
#: schedule, not one fact twice: collapsing a pair would delete the earlier
#: run's timestamp from the only per-task history anyone reads, and a loop of
#: identical successes is the silent-failure signal the log exists to show.
_REPEAT_COLLAPSE_MARK = "FAILED"


def fold_activity_note(note) -> str:
    """The note's text, on exactly one line.

    Every newline inside the note becomes `ACTIVITY_NOTE_FOLD` and empty
    segments are dropped, so a multi-line summary contributes continuation text
    to its own entry instead of bare lines the entry cap cannot count.
    """
    text = str(note or "").replace("\r\n", "\n").replace("\r", "\n")
    return ACTIVITY_NOTE_FOLD.join(seg.strip() for seg in text.split("\n") if seg.strip())


def _activity_entry_key(line: str) -> str:
    """What identifies an entry, minus everything that varies from run to run.

    Byte-equality is unusable here: every entry begins with a per-call stamp and
    every failure note carries a unique run id, so two notes from one failure
    loop are never byte-equal and a guard keyed on the raw line could never fire.
    Stripping the `×N` a previous collapse wrote is what lets the N+1-th
    occurrence match the first one.
    """
    text = ACTIVITY_REPEAT_RE.sub("", line.strip())
    if text.startswith("- "):
        text = text[2:]
    text = _RUN_STAMP_RE.sub("<stamp>", text)
    text = _ENTRY_STAMP_RE.sub("<ts>", text)
    return re.sub(r"\s+", " ", text).strip()


def _activity_repeat_count(line: str) -> int:
    """How many occurrences the entry already stands for (1 if never collapsed)."""
    m = ACTIVITY_REPEAT_RE.search(line.strip())
    return int(m.group(1)) if m else 1


def _last_activity_entry(lines: list[str], start: int) -> int:
    """Index of the last entry bullet in the log region from `start`; -1 if none.

    An entry is `- ` at column 0 and the prune marker is `- …` — the same two
    spellings `scripts/groundskeeper/retention-sweep.py` counts with, deliberately:
    the cap and the writer have to agree on what an entry is, or the writer adds
    lines to a window the cap is not counting. An indented bullet is therefore not
    an entry here either — it is leaked continuation text, and the sweep removes
    it, so counting a run onto it would be counting onto a line about to vanish.

    The region stops at the next heading so a file that ever grows a section
    below its log cannot have that section's bullets read as entries.
    """
    last = -1
    for i in range(start, len(lines)):
        if lines[i].lstrip().startswith("#"):
            break
        if lines[i].startswith("- ") and not lines[i].startswith("- …"):
            last = i
    return last


def append_activity_line(body: str, note: str, now_str: str) -> str:
    """Record one note as one entry under a body's `## Activity Log` heading.

    One physical line, always: a note carrying newlines has its continuation
    text folded into that entry (`fold_activity_note`), because a bare line is
    content the retention cap does not count and therefore never removes.

    A back-to-back repeat of the entry already last in the log does not get a
    second line. A task failing in a loop used to fill the entire retained
    window with one error string, so the log — the only per-task history a human
    or a later run reads — carried no information at all. The existing entry is
    re-stamped with this occurrence and gains a `×N` count instead, and the
    match ignores the two things that always differ between two runs: the stamp
    and the run id. Only a note reporting `FAILED` collapses; a note whose text
    differs once those are normalised still appends, so a real change of state
    is never swallowed by the counter.

    This is the one implementation of the shape. `_append_activity_log` hands it
    the whole file for the scheduler; the MCP tool and the HTTP route call it
    directly with a body they hold in memory, so both non-scheduler writers
    serialise the file themselves and still write exactly once. Creates the
    heading when the body has none.
    """
    folded = fold_activity_note(note)
    new_line = f"- {now_str}: {folded}"
    lines = (body or "").split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().lower() == ACTIVITY_LOG_HEADING.lower()), None)
    if start is None:
        return (body or "").rstrip() + f"\n\n{ACTIVITY_LOG_HEADING}\n\n{new_line}\n"
    idx = _last_activity_entry(lines, start + 1)
    if (idx >= 0 and _REPEAT_COLLAPSE_MARK in folded
            and _activity_entry_key(lines[idx]) == _activity_entry_key(new_line)):
        lines[idx] = f"{new_line} ×{_activity_repeat_count(lines[idx]) + 1}"
        return "\n".join(lines)
    return (body or "").rstrip() + f"\n{new_line}\n"


def _append_activity_log(task_id, note: str) -> None:
    """Append one note to the task file's Activity Log.

    Reads the file, applies `append_activity_line` — one line per note, a
    repeated failure counted on the line it already occupies — writes once.
    """
    path = _find_task_file(task_id)
    if not path:
        return
    content = path.read_text(encoding="utf-8")
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(append_activity_line(content, note, now_str), encoding="utf-8")


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


# `append_activity_line` moved up beside `_append_activity_log` (above, in this
# file): the scheduler's writer and the two body-in-memory callers —
# `agent_mcp/autonomy.py` and `app/routers/autonomy.py` — are one shape, and two
# copies of it is how a multi-line note could fold on one surface and spill bare
# lines on the other.


# ── Failure backoff ───────────────────────────────────────────────────────────
# A failed run used to leave `last_run` untouched, and `_is_task_due` gates only
# on `last_run` — so a task that timed out was due again on the very next 60s
# tick, forever. That produced 12 consecutive 600s timeouts on #36 in one night
# and ~130 on #69 over two days (~21 GPU-hours on a task whose script was gone).
# `failure_count` and `max_retries` were parsed, stored and displayed, but no
# scheduling decision read either.
_FAILURE_BACKOFF_BASE = 600          # 10 min floor for one retry step
_FAILURE_BACKOFF_CAP_SECONDS = 21600  # 6h ceiling for one retry step
# One retry step is this fraction of the task's OWN declared period, doubling
# per consecutive failure and bounded by the period and by the 6h ceiling.
_FAILURE_BACKOFF_PERIOD_FRACTION = 4.0
# How long a task stays retired before the scheduler brings it back on its own.
_FAILURE_REARM_PERIODS = 1.0
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
# Consecutive infra-classified failures one task may collect inside one declared
# period before it rests until the start of its next one (#1085). `1d1bbb6`
# established that an infra failure must not spend the retry budget "so an
# outage can't disable the whole fleet", and shipped only that half: with no
# ceiling the same classification re-dispatched one task every
# `_FAILURE_BACKOFF_BASE` seconds forever, with no `status: failed` resting
# state and no alert (the disable alert is gated on `disabled and alert`). This
# bounds the per-task loop WITHOUT reclassifying infra as task — the ceiling
# counts dispatches, the budget still counts only task failures, so an outage
# rests a task for one period instead of retiring 30 schedules.
_INFRA_CEILING = 5
# The hold string. `hold_reason` is what the board's `blocked` field, the
# dispatch loop's "Holding #" line and the next-run stall alert all print, so
# this must be distinguishable from "failure cooldown", which reads as
# "retrying shortly" and is precisely the false reassurance #1085 is about.
_INFRA_CEILING_HOLD = "infra ceiling"


def _failure_cooldown_seconds(task: dict) -> float:
    """One retry step: a fraction of the task's OWN period, doubling per
    consecutive failure, bounded by the period itself and by a 6h ceiling.

    Before #1086 the step was `10m * 2**(n-1)` with the cap taken as the
    LARGER of the task's interval and 6h, which made the interval a term that
    could not bind until failure_count 11 (640 min is the first step that
    reaches the 6h floor; a weekly 604800 s needs 17 doublings). With 30 of 31
    task files declaring `max_retries: 3`, the largest step any task ever
    computed was 20 minutes — so `frequency` contributed nothing to retry
    spacing, and a weekly job's entire retry budget was spent inside half an
    hour (measured on the old code: 600.0 s at failure_count 1 for weekly,
    daily and hourly alike).

    The bounds, so both failure modes are pinned: never below the 10-minute
    floor, never above the task's own period (a retry may not outlive the
    schedule it is retrying), and the FIRST step never above 6h — a later step
    may double past 6h, since applying the ceiling to every step would clamp
    every period of a day or more to a flat 6h and stop the doubling this is
    here to restore. Weekly therefore runs 6h/12h/24h where it used to run
    10m/20m, hourly runs 15m/30m/60m. A frequency the parser cannot read gets
    the OLD ladder, because inventing a period for it would be a guess with a
    long tail: `6x-daily` (task #24) and the spaced `every 15 min` both yield
    no period, and treating either as daily would space their retries 6h apart,
    well past their own cadence. An unknown period is not the same fact as a
    daily one, so it must not be coerced into it.
    """
    n = max(1, int(task.get("failure_count") or 0))
    interval = _frequency_interval_seconds(task)
    if interval is None:
        # No period, so no period may be a term: this is the pre-#1086 ladder
        # (10m doubling, 6h ceiling), kept deliberately. Falling back to a daily
        # period would hand a 15-minute or 6x-daily task a 6h first retry —
        # longer than the schedule it is retrying — because the parser cannot
        # read its spelling. Unknown must not be allowed to imply daily.
        return float(min(_FAILURE_BACKOFF_BASE * (2 ** (n - 1)),
                         _FAILURE_BACKOFF_CAP_SECONDS))
    first = interval / _FAILURE_BACKOFF_PERIOD_FRACTION
    first = min(max(first, _FAILURE_BACKOFF_BASE),
                _FAILURE_BACKOFF_CAP_SECONDS, interval)
    return float(min(first * (2 ** (n - 1)), interval))


_REARM_ALERT_TITLE = "✅ Autonomy task re-armed"


def _rearm_alert_id(stem: str) -> str:
    """The id that names one task file's rearm, in the log line and the alert.

    The disable and the rearm travel to the same channel with different titles,
    so this id is what lets a reader pair the two messages — and its absence
    from a rearm alert is what distinguishes "the rearm fired and the post
    failed" from "nothing was ever retired"."""
    return f"AUTOREARM {stem}"


def _notify_alert_channel(title: str, message: str):
    """Send one `discord_alert` from a synchronous caller. Returns the Thread.

    `app.discord_notify.discord_alert` is async and the recovery scan that
    rearms a task is sync, running inside the worker pool's own event loop via
    `run_in_executor` — so `asyncio.run` cannot be called on this stack and the
    post goes to a detached daemon loop. The Thread is returned so a caller that
    cares (a test) can join it; the scheduler does not, because a rearm that
    cannot post is still a rearm and the task's Activity Log carries the same
    line whatever the transport does.
    """
    try:
        from app.discord_notify import discord_alert

        def _run():
            try:
                asyncio.run(discord_alert(message, title=title))
            except Exception as exc:   # pragma: no cover - transport, not fatal
                logger.debug("rearm alert post failed: %s", exc)

        thread = threading.Thread(target=_run, daemon=True, name="alert-notify")
        thread.start()
        return thread
    except Exception as exc:  # pragma: no cover - import failure is not fatal
        logger.debug("rearm alert skipped (%s): %s", title, exc)
        return None


def _rearm_after_one_period(task: dict, now: datetime.datetime) -> Optional[str]:
    """Bring one retired task back on its own; return its rearm note, or None.

    A task disabled at `max_retries` used to have exactly one way out: a human
    editing its file. `_is_task_due` refuses `failed`, `_hold_reason` reports
    it, and both stall alarms skip it twice over — once on the status, once on
    the `next_run: None` the disable deliberately writes — so a transient
    half-hour of trouble retired a `weekly` schedule permanently with no alert
    saying so. #78 lost a run and its slot on 2026-09-09 and was revived by a
    manual run that made the schedule look like it had healed itself; #85 lost
    ~21 h on 2026-09-16 and was revived by another job editing the file.

    The clock this measures is `last_attempt`: the disable is itself a failure
    record, so `last_attempt` is the instant of the retirement for every task
    this path can select (the success write moves both stamps together). One
    declared period of silence is the evidence the outage has passed, and it
    costs the fleet at most one extra cycle per retirement.

    The rearm does four things at once, and the third is the one a fix that
    "just resets the status" would silently skip: `next_run` is rewritten,
    because a disabled task with no `next_run` is invisible to both stall
    alarms, so rearming without it re-creates the blind spot. `failure_count`
    resets, so a re-armed task has to spend a whole retry budget before it is
    retired again — and the disable alert therefore fires once per retirement
    rather than every period.

    `failed` is the ONLY status this touches, and that is the boundary with
    human intent: a person who means a task to stop uses `paused` or `draft`,
    the dispatch-stopping statuses named in `DISPATCH_STOPPING_STATUSES`, and
    those are left exactly as they were found. `failed` has no human-meaning
    writer — it is set in one place, `_record_failure`'s escalation — so
    rearming it cannot undo an instruction.
    """
    task_id = task.get("id")
    interval = _frequency_interval_seconds(task)
    if not task_id or not interval:
        return None
    last_attempt = _parse_iso(task.get("last_attempt"))
    if not last_attempt:
        return None
    idle = (now - last_attempt).total_seconds()
    if idle < interval * _FAILURE_REARM_PERIODS:
        return None

    path = _find_task_file(task_id)
    stem = path.stem if path else str(task_id)
    next_run = (now + datetime.timedelta(seconds=interval)).isoformat()
    _update_task_field(task_id, status="up_next", failure_count=0,
                       next_run=next_run, updated=now.isoformat())
    alert_id = _rearm_alert_id(stem)
    hours = idle / 3600.0
    period_hours = interval / 3600.0
    note = (f"AUTO-REARMED after {hours:.1f}h at status failed (>= its own "
            f"{period_hours:.1f}h period): status back to up_next, "
            f"failure_count reset, next_run set. Retire-by-failure is not "
            f"permanent; alert id {alert_id}.")
    _append_activity_log(task_id, note)
    logger.warning("Auto-rearmed retired task #%s (%s) after %.1fh idle",
                   task_id, task.get("name"), hours)
    _notify_alert_channel(
        _REARM_ALERT_TITLE,
        f"task #{task_id} ({task.get('name')}) auto-rearmed to up_next after "
        f"{hours:.1f}h retired — {period_hours:.1f}h period elapsed since it "
        f"hit max_retries. It alerts once per retirement, not per period. "
        f"[{alert_id}]")
    return note


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


# ── The infra ceiling (#1085) ────────────────────────────────────────────────
# Two persisted fields carry this state, both in the task file's front matter:
#
#   `infra_failure_count` — consecutive infra failures, cleared on a success, on
#       crossing the ceiling, and whenever failures arrive a whole period apart.
#   `infra_rest_until`    — the instant the ceiling's rest ends. Its presence in
#       the future IS the hold; there is no third flag.
#
# The counter is deliberately separate from `failure_count`. Ingesting infra into
# that budget would give an outage the fleet-wide disable `1d1bbb6` exists to
# prevent — and `tests/test_autonomy_scheduler.py` pins both of those
# properties (`test_fast_empty_response_is_infra_and_does_not_escalate`,
# `test_connection_error_is_infra`), so the separation is not merely stylistic.


def _infra_failure_count(task: dict) -> int:
    """Consecutive infra-classified failures recorded on this task file."""
    try:
        return max(0, int(task.get("infra_failure_count") or 0))
    except (TypeError, ValueError):
        return 0


def _infra_ceil_window_seconds(task: dict) -> float:
    """One declared period, for both counting the ceiling and resting on it.

    The task's own `frequency`/`runs_per_day` period, falling back to the 6 h
    retry ceiling for a frequency this module cannot parse (`6x-daily`, a cron
    expression): such a task is never due anyway — `_is_task_due` stops at "no
    frequency" — so the fallback only ever applies to a run someone invoked by
    hand, and it must still be a bound rather than nothing.
    """
    return _frequency_interval_seconds(task) or float(_FAILURE_BACKOFF_CAP_SECONDS)


def _next_infra_failure_count(task: dict, now: datetime.datetime) -> int:
    """This infra failure, counted: the running total, or 1 to start a new window.

    "N consecutive infra failures INSIDE ONE DECLARED PERIOD" is two conditions,
    and the success path only gives the first one. Failures a whole period apart
    do not accumulate: a server that hiccups once a week for five weeks is five
    separate hiccups, not a ceiling crossing, and resting a weekly task for a
    week on that evidence would be the outage-shaped punishment #1085 is careful
    not to inflict.
    """
    prev = _parse_iso(task.get("last_attempt"))
    if prev is None:
        return 1
    return (_infra_failure_count(task) + 1
            if (now - prev).total_seconds() <= _infra_ceil_window_seconds(task)
            else 1)


def _in_infra_rest(task: dict, now: datetime.datetime) -> bool:
    """True while a task is resting on the infra ceiling (#1085)."""
    until = _parse_iso(task.get("infra_rest_until"))
    return until is not None and now < until


def _infra_rest_reason(task: dict) -> str:
    """The board's sentence for a ceilinged task, with its resume time."""
    until = _parse_iso(task.get("infra_rest_until"))
    return (f"{_INFRA_CEILING_HOLD} ({_INFRA_CEILING} consecutive infra "
            f"failures), resting until {until.isoformat() if until else 'unknown'}")


def _infra_rest_seconds(task: dict) -> float:
    """How long a ceilinged task rests: until its next declared period starts.

    Two shapes, because a period is declared two ways. A task with
    `preferred_hours` (or an hour in `scheduled_at`, which
    `_effective_preferred_hours` also reads) rests the LATER of its raw period and
    the wait to the next hour IN that window — the window can only lengthen a rest
    onto an allowed hour, never shorten the period below itself. Where the window
    is the longer of the two the resume lands inside it, which is the case that
    matters: resting a windowed job until an hour outside its window leaves the
    board showing `outside hours` for a task the ceiling is holding, and the hour
    the window reopens is a fact of `preferred_hours` and the crossing hour, not of
    the minute the clock happens to read. Where the raw period is longer — a daily
    job whose window reopens in two hours — the resume is outside the window and
    `_is_preferred_hour` holds it the extra minutes, so the hold's NAME changes
    mid-rest. That is the shape the auto-rearm already has; the alternative is two
    timestamps, one for the due gate and one for `next_run`, which is what puts the
    board and the stall alarms telling different stories about one task. A
    windowless task rests its raw period, which is the number that matters: the
    flat 600 s cooldown already covered the sub-hour case, and a `weekly` task
    resting 604800 s instead of 600 s is the whole of this fix's worth.
    """
    period = _infra_ceil_window_seconds(task)
    hours = _effective_preferred_hours(task)
    if hours:
        window = set(hours)
        now_hour = _local_hour()
        ahead = next((k for k in range(1, 25) if (now_hour + k) % 24 in window), 24)
        return max(period, ahead * 3600.0)
    return period


# ── The run-summary seam (#642) ───────────────────────────────────────────────
# Both ledger summaries — `runs.summary` in `workers.db` and the front-matter
# `summary:` of `autonomy-runs/<task_id>/<run_id>.md` — used to be the LEADING
# edge of the model's final message, at two lengths (200 and 300) that disagreed
# with each other and with the 500-char caps downstream. The consequence is a
# ledger of opening narration: over the 92 `scheduled-task` rows completed since
# 2026-09-18, 82 sit at the cap and the join to their own artifacts puts 10/10 of
# them at the head of the response and 0/10 at its closing — rows ending
# `…Per job (gap = primary − seco` and `…scored (`primary` = Qwen3.8-Flash-Next-nvfp4 `.
# The full text is already persisted in the run record, so the closing is free.
#
# One cap, then, and one named slice per kind. The direction differs by kind on
# purpose, and the difference is the thing a caller cannot infer:
#
#   success  → the CLOSING. A report's outcome is its last line, and a run whose
#              response ends in a markdown table kept only the header row before
#              this existed (`| Check | Result |\n|---|---|\n| Bro`), so the
#              verdict a person comes to the ledger for was the part cut away.
#   failure  → the OPENING. The identity of a failure is its first token —
#              `TimeoutError: exceeded max_duration_seconds=...` — and both the
#              activity-log note and `compute_health`'s fallback for pre-`meta_json`
#              rows read exactly that prefix. Tail-slicing an error would delete
#              the only signal those rows carry.
RUN_SUMMARY_CAP = 300


def _outcome_summary(final_response: str) -> str:
    """The ledger summary of a completed run: the CLOSING of its response.

    Suffix, not prefix — see the seam note on `RUN_SUMMARY_CAP`. A run that
    signs off with a table keeps the row that answered the question instead of
    the header that opened it. This is the ONLY slice of `final_response` in the
    module; both summary call sites go through here.
    """
    final_response = final_response or ""
    return final_response[-RUN_SUMMARY_CAP:]


def _failure_summary(summary: str) -> str:
    """The ledger summary of a failed run: the OPENING of its error.

    Head-sliced deliberately, the one place the direction is reversed. The
    exception name is at the front, and it is what the activity log and the
    health classifier key on.
    """
    summary = summary or ""
    return summary[:RUN_SUMMARY_CAP]


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
    # Inside `run_task` only: a record written for a run nobody dispatched here
    # (a recovery sweep) names no trigger rather than a wrong one.
    for k, v in (_ACTIVE_RUN.get() or {}).items():
        if v is not None:
            fm.setdefault(k, v)
    content = f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n{body}"
    path = runs_dir / f"{run_id}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ── The change-ledger line, and the undo it points at (#963) ──────────────────
# Pre-images have existed for every scheduled run since `ef294bf` (2026-09-10),
# which armed the ledger by putting `turn_id=run_id` on the RunOptions below.
# Only the capture half got built. The single restore caller in the tree is the
# aggregator's `POST /changes/revert`, so "restore on failure" was in practice
# "restore on request, from someone who already knows the session id, the turn
# id and that a 7-day retention window is running" — three facts the one document
# somebody reads at 03:00 (the run record) named none of. Across 359 turn indexes
# and 1,051 entries measured at triage, `reverted_at` was set on zero: the undo
# was unreachable, not merely unused.
#
# The line below is what makes it reachable from the record alone. It reports,
# it does not restore: a nightly that wrote 6 of its 9 notes before dying has 6
# notes of real progress, and an automatic revert would destroy the progress to
# save the invariant. Whether to put a run's writes back is a decision, and a
# decision needs a reader — so the record names the scope and the count, and
# `revert_run_writes` below is the caller that acts on it.

CHANGE_LEDGER_HEADING = "## Change ledger"


def _change_ledger_note(session_id: str, run_id: str) -> tuple[dict, str]:
    """Front-matter fields and a body block naming this run's pre-images.

    Returns ``({}, "")`` only when there is no session id to name — a
    ``_record_failure`` call from outside ``run_task`` has no turn of its own,
    and inventing a directory for it would be worse than saying nothing.

    The count comes from ``_change_ledger.list_changes``, and never from anything
    the run asserted. That distinction is the whole reason the line is worth
    writing: the run that died mid-write is the worst possible witness about its
    own writes, and the tool calls were executed — and the pre-images flushed to
    ``sessions/<session_id>.changes/<run_id>/index.json`` — over in the
    aggregator, so the index is the account of record and this process holds no
    cache of its own to mistake for it. Zero entries reads as zero, not as an
    omission, because "this run touched no files" is a fact worth having on the
    record next to "this run touched seven".

    Never raises. A run record that cannot be written because the ledger was
    unreadable would be the second bug, and the more damaging one.
    """
    if not session_id or not run_id:
        return {}, ""
    try:
        from agent_mcp import _change_ledger
        entries = _change_ledger.list_changes(session_id, run_id)
        scope = _change_ledger.scope_label(session_id, run_id)
    except Exception as exc:  # noqa: BLE001 — the record must still be written
        logger.warning("Task run %s: change ledger unreadable (%s); "
                       "recording the failure without it", run_id, exc)
        return {}, ""

    count = len(entries)
    extra = {"changes": f"{scope} ({count} files)", "changes_files": count}
    lines = [f"{scope} ({count} files) — pre-images recorded by the aggregator.",
             ""]
    if count:
        lines += [
            "Nothing restored them automatically: a dead run's partial writes are",
            "often the progress worth keeping, so restoring is a decision made from",
            "this record, not a side effect of dying. To put them all back:",
            "",
            "```",
            f"autonomy.revert_run_writes({session_id!r}, {run_id!r})",
            "```",
            "",
            f"Files ({count}):"]
        lines += [_ledger_entry_line(e) for e in entries]
    else:
        lines.append("This run recorded no file writes, so there is nothing to "
                     "put back.")
    return extra, f"{CHANGE_LEDGER_HEADING}\n\n" + "\n".join(lines)


def _ledger_entry_line(entry: dict) -> str:
    """One bullet per recorded file, saying what a revert would do to it.

    The qualifier is the actionable half. An entry whose pre-image was never
    kept (`too_large`, or a snapshot that failed to write) is one revert will
    REFUSE, and a bullet that just names the file promises an undo that will not
    arrive. `create` is exempt: it reverts by unlinking, so it needs no
    pre-image at all.
    """
    path = str(entry.get("path") or entry.get("real") or "(unknown path)")
    if entry.get("reverted_at") is not None:
        return f"- `{path}` — already put back"
    if entry.get("op") != "create" and entry.get("snapshot") != "ok":
        return (f"- `{path}` — no pre-image ({entry.get('snapshot')}), "
                f"a revert will refuse this one")
    return f"- `{path}`"


def revert_run_writes(session_id: str, run_id: str,
                      paths: Optional[list[str]] = None) -> dict:
    """Put back one autonomy run's writes, by the two ids its record names.

    ``run_task`` passes ``turn_id=run_id``, so the run id IS the ledger's turn
    id: the ``session_id`` and ``run_id`` printed in a run record's front matter
    are the complete address, and this is the in-process call that takes them —
    instead of a hand-composed ``POST /changes/revert {session, turn}`` body,
    which is the only form that existed and is why nothing was ever restored.

    Answers per path, never with a bare success, because per-path is the only
    granularity that can be honest: a file another writer has changed since the
    snapshot comes back ``refused`` with its name and reason, and is left
    exactly as that later writer left it. Restoring over it would destroy the
    later write, which is the damage the ledger exists to prevent; skipping it
    silently would make the undo untrustworthy. Refused, by name, is the answer
    that can be acted on.

    No failure path calls this. Nothing in this module reverts anything
    unattended.

    Returns ``{scope, dir, recorded, counts, results}`` — ``results`` being the
    per-path list from ``_change_ledger.revert``, and ``recorded`` the entry
    count of that turn's index, so a caller can see at a glance when the number
    of files it was promised and the number it got do not match.
    """
    from agent_mcp import _change_ledger
    entries = _change_ledger.list_changes(session_id, run_id)
    scope = _change_ledger.scope_label(session_id, run_id)
    results = _change_ledger.revert(session_id, run_id, paths)
    counts: dict[str, int] = {}
    for row in results:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {"scope": scope, "dir": str(_change_ledger.turn_dir(session_id, run_id)),
            "recorded": len(entries), "counts": counts, "results": results}


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


#: The `frequency:` vocabulary the scheduler reads, in seconds. ONE definition
#: (#815): `_frequency_interval_seconds` answers from it, `_is_task_due` warns
#: from it, and `scripts/autonomy/validate_tasks.py` imports it to lint a
#: task file — so a consumer enumerates the recognised intervals instead of
#: inventing one. Anything outside it resolves to no interval, which is *not
#: due, ever*; `runs_per_day` is consulted first and is why #24's
#: `frequency: 6x-daily` dispatches at all.
FREQUENCY_INTERVALS: dict[str, float] = {
    "hourly": 3600.0, "every-15min": 900.0, "daily": 86400.0, "weekly": 604800.0,
}


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
    return FREQUENCY_INTERVALS.get(freq)


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
# Same once-per-process shape, for the interval gate below: a `frequency:`
# outside FREQUENCY_INTERVALS with no `runs_per_day` used to return False in
# silence (#815) — the no-skill path warned, this one did not, and dropping
# #24's `runs_per_day: 6` would have parked a nightly pipeline unannounced.
_no_frequency_warned: set[str] = set()


#: The `status:` `_write_run_record` gives a run that finished its work. Written
#: into `AUTONOMY_RUNS_DIR/<id>/run_<id>_<stamp>.md`. That tree is gitignored
#: (`/autonomy-runs/`, `.gitignore:24`), which is exactly why it is the account
#: of what happened that a vault sweep cannot undo — see #1296.
RUN_STATUS_SUCCESS = "success"


def _due_slack_seconds(task: dict) -> float:
    """How early a task may come due relative to its own interval.

    `last_run` is a COMPLETION time, so a due-gate measured from it drifts later
    by the run's own duration every cycle; for a task pinned to a one-hour window
    that drift eventually steps past the window and skips a day. One function
    because the two places that need this — the elapsed gate in `_is_task_due`
    and the period the run-record guard measures a success against — must agree
    to the second, or the guard holds a task longer than the healthy path ever
    would and pushes it out of its own window.
    """
    interval = _frequency_interval_seconds(task) or 0.0
    return min(3600.0, interval * 0.25) if _effective_preferred_hours(task) else 0.0


def _run_period_start(task: dict, *, now: datetime.datetime) -> Optional[datetime.datetime]:
    """Start of the period a success at `now` would still be held for.

    For a window-anchored task (#1437) this is only the outer bound: the
    elapsed gate may release it earlier, from the window opening its run
    belonged to, so `_already_ran_this_period` also asks each success
    `_elapsed_due_at` — the gate's own question.

    `now - (interval - slack)`, which is deliberately the exact window the
    elapsed gate would have granted had the completion stamp survived: a
    successful run at S refuses the next dispatch while `now <= S + interval -
    slack`, and the healthy path refuses while `now - S < interval - slack`. Same
    inequality, so the guard can never delay a task that the intact file would
    have dispatched — it substitutes for the lost stamp and nothing more.

    Trailing `now`, therefore: never anchored on `last_run` (the field this guard
    exists to survive — a verdict computed from it inherits its corruption) and
    never on `next_run`, which lives in the same vault front matter and is
    reverted by the same sweep. A hold anchored on a stale `last_run` would keep
    every tick inside one period and silence the job outright, which is a worse
    failure than the duplicate it prevents.
    """
    interval = _frequency_interval_seconds(task)
    if interval is None:
        return None
    return now - datetime.timedelta(seconds=max(1.0, interval - _due_slack_seconds(task)))


def _successful_run_this_period(
    task_id, *, period_start: datetime.datetime, holds=None,
) -> tuple[bool, str]:
    """Did a run record for this task report success inside the current period?

    Returns ``(succeeded, run_id)``; `run_id` is the record that answered, empty
    when nothing did, so a caller can name the run it refused to re-dispatch.

    Why due-ness needs a second account of what already ran (#1296). The front
    matter `last_run` is the ONLY thing `_is_task_due` measured, and it is a
    file in the vault — a tree other writers sweep. On 2026-09-20 an arch-review
    turn ended 48 s after task #82 (Nightly Retrieval Eval, `daily`,
    `preferred_hours: [6]`) completed at 13:07:04Z and reverted that task file
    to HEAD, restoring `last_run: 2026-09-19T13:07:31Z`. Elapsed was a whole
    24 h again on the next tick (13:08:17Z, 27 s after the run finished), the
    task dispatched a second time, and that dispatch spent 259 s and
    2,254,592 tokens of a primary-engine turn reaching
    "Today's baseline already exists — this is a duplicate dispatch". The run
    record it refused to re-run is gitignored and was untouched by the sweep.

    Reads are bounded to the task's own directory and to records that completed
    inside the period, so a weekly job's month-old successes cost nothing and a
    nightly job pays for one directory of ~17 files.
    """
    task_dir = AUTONOMY_RUNS_DIR / str(task_id)
    if not task_dir.is_dir():
        return False, ""
    newest = None
    try:
        paths = sorted(task_dir.glob("run_*.md"))
    except OSError:
        return False, ""
    for path in paths:
        try:
            parts = path.read_text(encoding="utf-8").split("---\n", 2)
            if len(parts) < 2:
                continue
            fm = yaml.safe_load(parts[1])
        except (OSError, ValueError):
            continue
        if not isinstance(fm, dict) or not isinstance(fm.get("status"), str):
            continue
        if fm.get("status").strip().lower() != RUN_STATUS_SUCCESS:
            continue
        # Prefer the instant the work FINISHED; `started_at` is the fallback so a
        # record missing `completed_at` still counts. The stamp in the filename
        # names the run's start, not its end, and the sort here is a comparison
        # against a completion-derived window — so the content decides, not the
        # path.
        when = _parse_iso(fm.get("completed_at")) or _parse_iso(fm.get("started_at"))
        if when is None or when < period_start:
            continue
        if holds is not None and not holds(when):
            continue
        if newest is None or when > newest[0]:
            newest = (when, str(fm.get("run_id") or path.stem))
    if newest is None:
        return False, ""
    return True, newest[1]


#: The hold reason the board shows when the run record vetoes a dispatch. One
#: constant because `hold_reason` and `_is_task_due` are answering two questions
#: about one fact, and every previous pair of those two disagreed by
#: construction (#870, #558).
RUN_SUCCESS_HOLD = "already ran this period"


def _already_ran_this_period(task: dict, *, now: datetime.datetime) -> str:
    """Run id of a successful run inside this task's own period, else ''.

    The single source for both gates. Returns '' for a task with no declared
    period — `frequency` absent and no `runs_per_day` — because there is then no
    defensible window and a stale success must not silence a job forever.
    """
    task_id = task.get("id")
    if task_id is None:
        return ""
    period_start = _run_period_start(task, now=now)
    if period_start is None:
        return ""
    # A window-anchored task can come due before `period_start` has passed its
    # last success (`_elapsed_due_at` counts from the window opening), so each
    # success is also asked the elapsed gate's own question — does it still
    # hold at `now`? `period_start` stays as the cheap outer bound: it admits
    # every record that test keeps, and more.
    def _still_holds(when):
        return now < _elapsed_due_at(task, when)
    holds = _still_holds if _window_anchored(task) is not None else None
    ok, run_id = _successful_run_this_period(
        task_id, period_start=period_start, holds=holds)
    return run_id if ok else ""


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


_missing_artifact_bypass_warned: dict = {}


def _upstream_artifact_on_disk(dep_task: dict,
                               dep_last_run: Optional[datetime.datetime],
                               now: datetime.datetime) -> Optional[str]:
    """The upstream's declared `output_artifact` as it stands on disk, for a bypass.

    Returns None when the upstream declares no artifact (there is nothing to
    require, and the bypass stays a pure elapsed-time rule), the path found when
    one candidate exists at `_ARTIFACT_MIN_BYTES` or more, and "" when the
    upstream declares an artifact and none of its candidates is there.

    Candidates are the `{date}` spellings `_artifact_candidates` resolves, taken
    at the upstream's last completion AND at that minus its `timeout_seconds`,
    because the template is dated by the run's START and `last_run` is its END —
    a run that crosses local midnight names yesterday. There is no mtime test,
    unlike `_declared_artifact_evidence`: a bypass forwards on stale input by
    design, so the question here is only whether the input exists at all. An
    upstream that never ran is resolved at `now`.
    """
    declared = str(dep_task.get("output_artifact") or "").strip()
    if not declared:
        return None
    if dep_last_run is None:
        anchors = [now]
    else:
        try:
            span = float(dep_task.get("timeout_seconds") or 1800)
        except (TypeError, ValueError):
            span = 1800.0
        anchors = [dep_last_run, dep_last_run - datetime.timedelta(seconds=span)]
    seen: set = set()
    for anchor in anchors:
        for path in _artifact_candidates(declared, anchor):
            if path in seen:
                continue
            seen.add(path)
            try:
                if path.stat().st_size >= _ARTIFACT_MIN_BYTES:
                    return str(path)
            except OSError:
                continue
    return ""


def _warn_missing_artifact_bypass(dependent: dict, dep_task: dict) -> None:
    """Once per (dependent, upstream) episode: a bypass refused for want of input."""
    key = (str(dependent.get("id", "")), str(dep_task.get("id", "")))
    if _missing_artifact_bypass_warned.get(key):
        return
    _missing_artifact_bypass_warned[key] = True
    logger.warning(
        "Task #%s is HELD past its stale_bypass_hours: upstream #%s declares "
        "output_artifact %r and no candidate for its last run (%s) is on disk. "
        "Fail-forward means running on stale input, not on none (#1437).",
        dependent.get("id"), dep_task.get("id"), dep_task.get("output_artifact"),
        dep_task.get("last_run") or "never")


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

    AND THE INPUT MUST EXIST (#1437). A window measures a timestamp; what the
    dependent consumes is a file. On 2026-09-24 #39 was released 50.7 h past
    #42's last run onto a `knowledge-handoff-{date}.md` that was on disk under
    no spelling, so it could only take its no-artifact path and spend the cycle.
    When the upstream declares `output_artifact`, a bypass now also needs one of
    its candidates on disk (`_upstream_artifact_on_disk`); without it the
    dependent is held and a warning names the missing file. An upstream that
    declares nothing keeps the elapsed-time rule alone, which is every case #814
    and #870 pinned.
    """
    try:
        bypass_hours = float(dependent.get("stale_bypass_hours") or 0)
    except (TypeError, ValueError):
        return False
    if bypass_hours <= 0:
        return False
    if str(dep_task.get("status", "")).strip() == "in_progress":
        return False
    key = (str(dependent.get("id", "")), str(dep_task.get("id", "")))
    if dep_last_run is not None:
        _never_ran_bypass_warned.pop(key, None)
        if (now - dep_last_run).total_seconds() <= bypass_hours * 3600:
            return False
    if _upstream_artifact_on_disk(dep_task, dep_last_run, now) == "":
        _warn_missing_artifact_bypass(dependent, dep_task)
        return False
    _missing_artifact_bypass_warned.pop(key, None)
    if dep_last_run is None:
        _warn_never_ran_bypass(dependent, dep_task, bypass_hours)
    return True


def _upstream_due_in_window(dep_task: dict, dep_last_run: datetime.datetime,
                            all_tasks: list[dict], now: datetime.datetime,
                            seen: frozenset) -> bool:
    """Will the scheduler dispatch this windowed upstream on this very tick?

    Asked only for a window pair whose upstream still owes this cycle a run
    (`_window_dependency_fresh` False), before the `stale_bypass_hours`
    override is consulted. The upstream's own gates, in `_is_task_due`'s
    order, at `now`: `up_next`, no rest or cooldown, its elapsed-due instant
    passed, inside its window, and its own `depends_on` met. That last one
    recurses up the chain; `seen` stops a cycle, which then reads as "not
    about to run" so the override stays reachable."""
    dep_id = str(dep_task.get("id", ""))
    if dep_id in seen:
        return False
    hours = _window_hours(dep_task)
    if hours is None or str(dep_task.get("status", "")).strip() != "up_next":
        return False
    if _in_infra_rest(dep_task, now) or _in_failure_cooldown(dep_task, now):
        return False
    due_at = _elapsed_due_at(dep_task, dep_last_run)
    if due_at is None or now < due_at:
        return False
    if _first_in_window_at_or_after(hours, now) != now:
        return False
    return _is_dependency_met(dep_task, all_tasks, now=now, _seen=seen)


def _is_dependency_met(task: dict, all_tasks: list[dict], *,
                       now: Optional[datetime.datetime] = None,
                       _seen: frozenset = frozenset()) -> bool:
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
    # `or 86400.0` is defensive only. Both production callers exclude a None
    # interval before reaching this gate — `_is_task_due` returns False (and
    # warns once) and `hold_reason` returns "no frequency" — so a task whose
    # frequency is outside FREQUENCY_INTERVALS meets this line only by a direct
    # call, where it is judged against a day's 12 h half-window rather than
    # crashing (#815; tests/test_autonomy_dependency_fail_closed.py pins it).
    #
    # WINDOWS FIRST (#1437 Defect B). When both tasks declare a window and the
    # dependent runs daily or slower, freshness is a fact of the two windows,
    # not of an hour count: `_window_dependency_fresh` asks whether the
    # upstream still owes a run before the dependent's current window closes.
    # `interval / 2` straddled the window whenever the dependent's opened
    # before its upstream's — safe in the healthy chain needed a bound over
    # 21.9 h, releasing after a slip needed one under 21.0 h — so no
    # `stale_bypass_hours` could serve both. The bypass stays an explicit
    # override below; the ordinary chain no longer needs it.
    interval = _frequency_interval_seconds(task) or 86400.0
    fresh = _window_dependency_fresh(task, dep_task, dep_last_run, now)
    if fresh is None:
        fresh = (now - dep_last_run).total_seconds() <= interval / 2
    elif not fresh and _upstream_due_in_window(
            dep_task, dep_last_run, all_tasks, now,
            _seen | {str(task.get("id", ""))}):
        # The upstream is inside its window and due this very tick, so the
        # output this cycle consumes is about to exist: a bypass now would
        # dispatch the dependent BESIDE its upstream on the previous cycle's
        # file. On 2026-09-25 05:00Z that is #42 next to #38, 38.5 h past
        # #38's last run and so past its 36 h `stale_bypass_hours`.
        return False
    if not fresh:
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


# ── Window occurrences (#1437 Defect B) ──────────────────────────────────────
# `preferred_hours` are machine-local hours, so "the window" is a recurring
# span of local time — [22,23,0,1,2,3,4] opens at 22:00 and closes at 05:00 the
# next morning. Two things are measured against its OCCURRENCES, never against
# a fixed hour count: when a windowed task is next due (`_elapsed_due_at`) and
# whether an upstream's output belongs to a dependent's cycle
# (`_window_dependency_fresh`). On 2026-09-23 #38 and #56 ran at 14:25-14:31Z,
# outside their windows; `last_run + interval` then re-anchored both outside
# again, each lost its next window, and `interval / 2` could not call their
# output fresh anywhere inside a dependent's window — four daily jobs held two
# cycles, with no `stale_bypass_hours` value able to release them safely.

def _local_tz():
    """tzinfo the window hours are read in; None = this machine's zone.

    Indirection so a test can pin a fixed offset. None goes through
    `astimezone()` with no argument, which applies the zone's DST rule for each
    instant rather than today's offset."""
    return None


def _to_local(instant: datetime.datetime) -> datetime.datetime:
    """`instant` as naive machine-local wall time."""
    return instant.astimezone(_local_tz()).replace(tzinfo=None)


def _from_local(naive: datetime.datetime) -> datetime.datetime:
    """Naive machine-local wall time back to an aware UTC instant."""
    tz = _local_tz()
    aware = naive.astimezone() if tz is None else naive.replace(tzinfo=tz)
    return aware.astimezone(datetime.timezone.utc)


def _window_hours(task: dict) -> Optional[list]:
    """The task's window as sorted hours, or None when it has no real window.

    A window naming all 24 hours has no opening and is no window at all."""
    hours = _effective_preferred_hours(task)
    if not hours:
        return None
    hs = sorted({int(h) % 24 for h in hours})
    return hs if 0 < len(hs) < 24 else None


def _window_occurrences(hours: list, around: datetime.datetime,
                        days: int = 2) -> list:
    """(open, close) UTC instants of every window occurrence within `days`
    local days of `around`, sorted. A run of hours that wraps midnight is one
    occurrence, opening on the evening's date."""
    hs = set(hours)
    base = _to_local(around).date()
    out = []
    for d in range(-days - 1, days + 1):
        day = base + datetime.timedelta(days=d)
        for h in sorted(hs):
            if (h - 1) % 24 in hs:
                continue
            length = 1
            while (h + length) % 24 in hs:
                length += 1
            start = datetime.datetime.combine(day, datetime.time(h))
            out.append((_from_local(start),
                        _from_local(start + datetime.timedelta(hours=length))))
    out.sort()
    return out


def _latest_window_open(hours: list,
                        at: datetime.datetime) -> Optional[tuple]:
    """The occurrence that most recently OPENED at or before `at`."""
    best = None
    for occ in _window_occurrences(hours, at):
        if occ[0] <= at:
            best = occ
    return best


def _first_in_window_at_or_after(hours: list,
                                 at: datetime.datetime) -> datetime.datetime:
    """`at` itself when it is inside the window, else the next opening."""
    for start, end in _window_occurrences(hours, at):
        if start <= at < end:
            return at
        if start > at:
            return start
    return at  # unreachable for a real window; a due check must never raise


def _window_anchored(task: dict) -> Optional[list]:
    """Window hours when this task's CADENCE follows its window, else None.

    Only for a period of at least a day: a window occurs once per local day, so
    a sub-daily task's cycles cannot be window occurrences, and it keeps the
    pure elapsed-time rule. A task with no window keeps it too — unchanged."""
    interval = _frequency_interval_seconds(task)
    if interval is None or interval < 86400.0:
        return None
    return _window_hours(task)


def _elapsed_due_at(task: dict,
                    last_run: datetime.datetime) -> Optional[datetime.datetime]:
    """The instant a run that completed at `last_run` stops holding the task.

    ONE definition for the elapsed gate in `_is_task_due`, the run-record guard
    (`_already_ran_this_period`) and the `next_run` a completion writes
    (`_next_run_after`), so the guard can never hold what the intact stamp
    would release (#1296) and the board names the slot dispatch will use.

    Windowless, or sub-daily: `last_run + interval - slack`, exactly as before.

    Window-anchored: a run belongs to the window occurrence that last OPENED at
    or before it — for an out-of-window run, the one it slipped from — and the
    next cycle is that opening plus the interval, less the slack. So one
    out-of-window completion can no longer carry the next due instant past the
    next window: #38 done at 14:31Z comes due for 05:00Z the next morning, not
    13:31Z (after its window had closed). Floored at `last_run + interval / 2`
    so a run made just before a window opens is not repeated inside it. Never
    later than the old rule: the opening is at or before `last_run`, and the
    slack is at most a quarter period.
    """
    interval = _frequency_interval_seconds(task)
    if interval is None:
        return None
    slack = datetime.timedelta(seconds=_due_slack_seconds(task))
    period = datetime.timedelta(seconds=interval)
    hours = _window_anchored(task)
    occ = _latest_window_open(hours, last_run) if hours else None
    if occ is None:
        return last_run + period - slack
    return max(occ[0] + period - slack, last_run + period / 2)


def _next_run_after(task: dict,
                    completed: datetime.datetime) -> Optional[datetime.datetime]:
    """The `next_run` a successful completion writes.

    Windowless tasks keep `completed + interval`. A window-anchored task gets
    the first in-window instant at or after its elapsed-due instant — after an
    out-of-window run, `completed + interval` named an hour the window forbids,
    so the board showed a slot the scheduler could never use."""
    interval = _frequency_interval_seconds(task)
    if interval is None:
        return None
    hours = _window_anchored(task)
    if hours is None:
        return completed + datetime.timedelta(seconds=interval)
    return _first_in_window_at_or_after(hours, _elapsed_due_at(task, completed))


def _window_dependency_fresh(task: dict, dep_task: dict,
                             dep_last_run: datetime.datetime,
                             now: datetime.datetime) -> Optional[bool]:
    """Is the upstream's last run the one this dependent's cycle consumes?

    None when the windows cannot answer — either task has no window, or the
    dependent's cadence is sub-daily — and the caller keeps `interval / 2`.

    The dependent's cycle is its window occurrence that last opened at or
    before `now`, closing at C. The upstream OWES a run in that cycle when its
    own next slot — `_next_run_after(upstream, its last_run)`, the instant the
    scheduler will next dispatch it — falls before C. Owed means not fresh:
    the dependent waits for this cycle's upstream, whether the upstream's
    window opens before its own (#38 → #42, the same hours) or inside it (#56
    at local 01:00 → #57, open from 23:00). Not owed means the output on hand
    is the newest this cycle will get, in window or not. After #38 slipped to
    14:31Z it is owed 05:00Z, runs there, and #42 follows it in the same
    window, instead of being held 14.5-20.5 h past a 12 h bound.
    """
    mine = _window_anchored(task)
    if mine is None or _window_hours(dep_task) is None:
        return None
    occ = _latest_window_open(mine, now)
    slot = _next_run_after(dep_task, dep_last_run)
    if occ is None or slot is None:
        return None
    return slot >= occ[1]


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
        task_id = str(task.get("id", "?"))
        if task_id not in _no_frequency_warned:
            _no_frequency_warned.add(task_id)
            logger.warning(
                "Task #%s (%s) has frequency %r and no runs_per_day — not in "
                "%s, so it will NEVER run until one is set",
                task_id, task.get("name"), task.get("frequency"),
                sorted(FREQUENCY_INTERVALS))
        return False
    last_run = _parse_iso(task.get("last_run"))
    if last_run:
        # last_run is a COMPLETION time, so due-time drifts later by the run's
        # own duration every cycle. For a task pinned to a one-hour window that
        # drift eventually steps past the window and skips a day, so allow a
        # little slack when a window is in force — and a daily-or-slower
        # windowed task counts its period from the window opening its run
        # belonged to, so one out-of-window run cannot cost it the next window
        # (#1437 Defect B, `_elapsed_due_at`).
        if now < _elapsed_due_at(task, last_run):
            return False
    # The infra ceiling (#1085), ABOVE the retry cooldown on purpose: a task
    # that just crossed it is inside both, and the cooldown would win the
    # explanation while being the weaker claim — 600 s against a whole period.
    # `hold_reason` answers this same predicate at the same instant, in this
    # same position, which is the #870 rule: one definition of due-ness.
    if _in_infra_rest(task, now):
        return False
    # A failed run keeps last_run untouched, so without this gate the task is
    # due again on the next tick — the retry storm.
    if _in_failure_cooldown(task, now):
        return False
    if not _is_dependency_met(task, all_tasks, now=now):
        return False
    if not _is_preferred_hour(task):
        return False
    # ── The work already ran this period ─────────────────────────────────
    # Last, on purpose: every gate above reads only the task file, so this is
    # the one place that asks a second, independent account — the run record on
    # disk — and it is only paid for by tasks that everything else already
    # called due. It exists because the file every gate above reads is in the
    # vault, and vault writers exist: task #82's completion stamp was reverted
    # by an arch-review sweep 48 s after it landed on 2026-09-20, `last_run`
    # went back a day, elapsed read a whole 24 h again on the next tick, and the
    # task dispatched a second time 27 s after it finished. `hold_reason` asks
    # the same helper, so the board cannot show runnable what the scheduler
    # refuses to run (#1296, and the #870/#558 pair this file keeps citing).
    ran_id = _already_ran_this_period(task, now=now)
    if ran_id:
        logger.info(
            "Task #%s (%s): not due — run record %s reports `status: success` "
            "inside this task's period, whatever `last_run` (%r) says",
            task.get("id"), task.get("name"), ran_id, task.get("last_run"))
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
    if _in_infra_rest(task, now):
        return _infra_rest_reason(task)
    if _in_failure_cooldown(task, now):
        return "failure cooldown"
    if not _is_dependency_met(task, all_tasks, now=now):
        return f"waiting on #{task.get('depends_on')}"
    if not _is_preferred_hour(task):
        window = _hour_windows(_effective_preferred_hours(task) or [])
        return f"outside hours {window}" if window else "outside hours"
    # Same helper, same position in the order, so the board never paints green a
    # dispatch the scheduler is refusing. This is the state a reverted stamp
    # leaves behind: `last_run` a day stale, the window open, and the run record
    # saying the job already ran — which a person has to be able to SEE, or it
    # reads as a scheduler that silently skipped a night (#1296).
    if _already_ran_this_period(task, now=now):
        return RUN_SUCCESS_HOLD
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


def _configured_model_names() -> list[str]:
    """Every name a task may put in `model:`, in config order, deduplicated.

    The `models:` slot keys plus each slot's declared `alias` — the two forms
    `app.config._get_model_cfg` resolves, so the two forms that can reach the
    engine. Read through the module attribute rather than a snapshot so a test
    (or a reload) that replaces `MODEL_CONFIGS` is answered by the replacement.

    An empty list means CANNOT ADJUDICATE, never "nothing is valid": a validator
    that fails closed when its own config read failed would take the whole
    fleet down with the config file, which is a bigger outage than the one this
    exists to prevent.
    """
    try:
        from app.config import MODEL_CONFIGS
    except Exception:  # pragma: no cover - import failure is not fatal
        return []
    names: list[str] = []
    for key, cfg in (MODEL_CONFIGS or {}).items():
        for name in (key, (cfg or {}).get("alias")):
            if name and name not in names:
                names.append(str(name))
    return names


def _model_dispatch_refusal(task_id, task_model: str) -> Optional[str]:
    """Why this dispatch must not reach the engine, or None to let it through.

    #1209. Task #85 dispatched for the first time on 2026-09-16 carrying
    `model: eco` — an alias `config.yaml` never defined — and failed three
    times in 38 minutes at 0.3 s / 5.6 s / 0.3 s with `vLLM returned 404: The
    model `eco` does not exist`, then self-disabled. The string was passed
    straight to the engine because nothing between the task file and
    `create_session(model=...)` asked whether it named a configured slot:
    `resolve_model_alias` only rewrites `secondary`→`primary` when the secondary
    switch is off, and this module's `_get_model_env` swallows a miss into `{}`
    with no log line at all. So the first signal anyone got was the engine's
    404, and the second was the row reading `status: failed` — which, with
    `next_run` nulled, both stall alarms skip by construction.

    A typo is fully detectable here for the price of one membership test, so it
    is refused HERE: no engine call, no run record, no `failure_count`, and the
    text names both the offending value and the slots that exist, because
    "unknown model" alone would send whoever reads it looking at the engine.

    An empty `task_model` is deliberately let through. It is not an unvalidated
    name, it is the absence of one — the value a task with no `model:` lands on
    when `model.default` is unreadable too — and refusing it would turn a
    missing key into a new outage class for a bug that only ever came from a
    name that WAS present.
    """
    if not task_model:
        return None
    known = _configured_model_names()
    if not known or task_model in known:
        return None
    return (f"Task #{task_id}: model '{task_model}' is not a configured alias; "
            f"known: {', '.join(known)} — refusing dispatch instead of sending "
            f"an unvalidated model id to the engine")


# Refusals already written to a task's activity log, keyed `"<task id>:<model>"`.
# Same one-mark-per-process shape as `_no_skill_warned` above, for the same reason:
# the scheduler re-enqueues a refused task on the tick after the refusal (a
# refusal must not move the row — clause 3 of #1209 — so due-ness is untouched),
# so anything written unconditionally is written once per tick forever.
_model_refusal_logged: set[str] = set()


def _record_model_refusal(task_id, task_model: str, refusal: str) -> bool:
    """Mark the task file with its own refused dispatch. True if this call wrote it.

    The refusal returns a `skipped` queue row and a `logger.error`, and both are
    true at the instant they are written; what neither is is a record on the
    thing that is broken. The review rung of SM_20260921_091912 named exactly
    that: "the refusal is loud but not durable, and nothing backs a task off".
    The task file's own Activity Log is the durable surface that already exists
    here — it is where `DISABLED after 3 consecutive failures` landed for #85 and
    what the autonomy board renders — so the refusal gets one line there, naming
    the offending value and the real slots, written once per (task, model) per
    process. The repeated refusals themselves stay at log level on purpose: a
    config typo that outlives a person's attention must not append a hundred
    identical lines to a curated file, which is the same judgement
    `_no_skill_warned` reached from the other direction.
    """
    key = f"{task_id}:{task_model}"
    if key in _model_refusal_logged:
        return False
    _model_refusal_logged.add(key)
    try:
        _append_activity_log(task_id, refusal)
    except Exception as exc:  # noqa: BLE001 — a note is never worth the dispatch
        logger.warning("Task #%s: could not record the model refusal in its log: %s",
                       task_id, exc)
    return True


def _resolve_task_model(task_id, task: dict) -> str:
    """The model name this dispatch will hand the engine, after two rewrites.

    Both rewrites predate #1209 and are unchanged by it: an absent/`null`
    `model:` falls back to `model.default`, and `secondary` becomes `primary`
    when `secondary_enabled` is off — every other caller in the codebase goes
    through `resolve_model_alias` for that second one and the autonomy path did
    not, so a task pinned to a switched-off secondary failed instead of falling
    back. Split out of `run_task` so the refusal below has something to check
    that is not inline arithmetic, and so the empty-string case (no key, no
    readable default) is visibly the absence of a name rather than a name.
    """
    task_model = str(task.get("model", "") or "").strip()
    if not task_model or task_model.lower() in ("null", "none"):
        try:
            cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
            task_model = cfg.get("model", {}).get("default", "")
        except Exception:
            pass
    if not task_model:
        task_model = ""

    try:
        from app.config import resolve_model_alias
        resolved = resolve_model_alias(task_model)
        if resolved != task_model:
            logger.info("Task #%s: model %s -> %s (secondary_enabled=false)",
                        task_id, task_model, resolved)
            task_model = resolved
    except Exception as e:
        logger.warning("Model alias resolution failed for #%s: %s", task_id, e)
    return task_model


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


def _acceptance_grade(task: dict, dispatch, final_text: str) -> Optional[dict]:
    """The run's acceptance grade (#623), or None when the task has `grader:
    false`. Never raises: a grading bug costs the grade, not the run record."""
    try:
        return grade_run(task, dispatch.as_trace(final_text))
    except Exception as e:
        logger.warning("Task #%s: acceptance grading failed: %s", task.get("id"), e)
        return {"grade": "grader_error", "error": f"{type(e).__name__}: {e}"}


async def _record_artifact_success(task: dict, task_id, run_id: str,
                                   started_at: str,
                                   started_dt: datetime.datetime, duration: float,
                                   artifact: dict, *, stop_reason, usage,
                                   num_turns, tool_errors: list,
                                   session_id: str,
                                   grade: Optional[dict] = None) -> dict:
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
    if grade is not None:
        meta[_ACCEPTANCE_GRADE_KEY] = grade
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
    next_run_dt = _next_run_after(task, completed_dt) if interval else None
    next_run_iso = next_run_dt.isoformat() if next_run_dt else None
    # Both stamps, exactly as on a text-confirmed success: `last_run` is what
    # `_is_dependency_met` reads, and the cooldown gate reads
    # "last_attempt newer than last_run" as "the most recent attempt failed".
    _update_task_field(task_id, status="up_next", last_run=completed_at,
                       last_attempt=completed_at, updated=completed_at,
                       failure_count=0,
                       # Clause 2 of #1085 is about a SUCCESSFUL RUN, and a run
                       # has two shipped writers — this one, and the
                       # text-confirmed write in `run_task` that already clears the
                       # pair. Clearing on one path is not clearing: an
                       # artifact-backed success leaves the count standing, and
                       # because every success stamps `last_attempt` to its own
                       # completion, the next infra failure is inside the counting
                       # window by construction — so a 4 carried across a clean
                       # stretch rests the task a whole period on one later
                       # hiccup, which is a ceiling punishing a healthy task.
                       infra_failure_count=0, infra_rest_until=None,
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
    if grade is not None:
        result[_ACCEPTANCE_GRADE_KEY] = grade
    if _evidence_pilot(task_id):
        result["claims"] = _evidence_claims("")
    return result


# ── Fast-failure alert (#1209) ────────────────────────────────────────────────
# A failure that completes in under half a minute is not a task that failed at
# its work. A run that reaches the engine and does something takes minutes;
# 0.3 s is a request refused on the way in — a model alias the server does not
# have, a route that is not there, a payload it rejects. #1209's task #85 logged
# exactly that (0.3 s / 5.6 s / 0.3 s, `vLLM returned 404: The model `eco` does
# not exist`) and the only trace was the task row reading `status: failed` with
# `next_run` nulled, which both stall alarms skip by construction and the one
# alert that fires is a no-op on an unconfigured transport. The shape is worth
# its own line in the one place a person reads daily.
_FAST_FAILURE_ALERT_SECONDS = 30.0
_FAST_FAILURE_ALERT_STREAK = 3


def _daily_note_dir() -> Path:
    """Where today's daily note lives: the vault's `memory/`, unless overridden.

    `LLOYD_DAILY_NOTE_DIR` exists for the test suite — the same env-var-at-read
    shape `tests/conftest.py` uses for the automod, guardian and manifest state
    dirs, for the same reason: the default is the live vault, and a fixture that
    fails three runs to prove an alert exists would otherwise write fiction into
    today's note. `app/post_capture._append_daily_note` hardcodes the vault
    path; this is the same target, made redirectable rather than duplicated.
    """
    env = os.environ.get("LLOYD_DAILY_NOTE_DIR")
    if env:
        return Path(env)
    return Path.home() / "obsidian" / "memory"


def _fast_task_failure_seconds(path: Path, *,
                               max_seconds: float) -> Optional[float]:
    """This run record's duration, when it IS a sub-`max_seconds` task failure.

    `None` means "this record is not part of a fast-failure streak", which the
    caller reads as STOP LOOKING, not as "skip it and keep going" — the
    distinction is the whole meaning of `consecutive`, see
    `_fast_failure_streak`. Includes a record whose front matter cannot be read
    or which carries no numeric `duration_seconds`: a streak the reader cannot
    certify is not a streak, and guessing from the neighbour either way is what
    this helper exists to refuse.
    """
    try:
        parts = path.read_text(encoding="utf-8").split("---\n", 2)
        if len(parts) < 2:
            return None
        fm = yaml.safe_load(parts[1])
    except (OSError, ValueError):
        return None
    if not isinstance(fm, dict) or fm.get("status") != "failed":
        return None
    if str(fm.get("failure_kind") or "task") != "task":
        return None
    duration = fm.get("duration_seconds")
    if not isinstance(duration, (int, float)) or duration >= max_seconds:
        return None
    return float(duration)


def _fast_failure_streak(task_id, *, max_seconds: float = _FAST_FAILURE_ALERT_SECONDS,
                         limit: int = 12) -> list[float]:
    """Durations of this task's trailing run of sub-`max_seconds` failures.

    Oldest first, so the caller can print them as a sequence. Consecutive in the
    literal sense: the walk goes newest-record-first and STOPS at the first
    record that is not itself a sub-`max_seconds` `failure_kind: task` failure —
    a success, an `infra` failure, a slow failure, an unreadable record. The
    first cut of this reader collected only the qualifying records and then
    measured the tail of that filtered list, so `0.3s, 0.3s, SUCCESS, 0.3s` read
    as a streak of three and the alert said "consecutive" about a task that had
    succeeded in the middle of it. A filter cannot express "consecutive";
    stopping can, which is what `test_a_successful_run_between_two_fast_failures_
    does_not_alert` now pins.

    An `infra` failure breaks the streak for the same reason it does not count
    toward it: the 2026-09-01 eleven-hour empty-server window produced nothing
    but those, and its remedy is patience, not a look at the task file. What a
    run of them does change is the *sequence* — after an outage the clock starts
    again, so the alert speaks about failures that are all post-outage.

    Read off the run records rather than a counter kept somewhere: the only
    per-task counter that survives a restart is `failure_count` in the task
    file, which cannot distinguish three 0.3 s refusals from three 600 s
    timeouts — and the durations ARE the signal here, and are already on disk in
    the one place every run is recorded. Filenames are the order: a run id is
    `run_<task>_<UTC date>_<HHMMSS>` made at dispatch, so a lexical sort of the
    directory is a chronological one and needs no timestamp parsing to agree
    with itself. Newest-first up to `limit` records, so a weekly job's month of
    history costs one directory of ~17 small files.
    """
    task_dir = AUTONOMY_RUNS_DIR / str(task_id)
    if not task_dir.is_dir():
        return []
    try:
        paths = sorted(task_dir.glob("run_*.md"))
    except OSError:
        return []
    streak: list[float] = []
    for path in reversed(paths[-limit:]):
        duration = _fast_task_failure_seconds(path, max_seconds=max_seconds)
        if duration is None:
            break
        streak.append(duration)
    return list(reversed(streak))


def _append_fast_failure_alert(task: dict, task_id, durations: list[float]) -> bool:
    """Write the one line a fast-failing task never had. True if it landed.

    Deliberately NOT routed through `app.discord_notify.discord_alert`, which is
    what the disable alert further below calls: this box carries
    `discord.home_channel: null` and an empty token, so `discord_alert` logs a
    warning and returns (app/discord_notify.py:52-56). #85's disable alert
    therefore "fired" and reached nobody — an alert whose only transport is
    unconfigured is not an alert, and routing a new one through that transport
    would inherit the dead end. The daily note is a surface that demonstrably
    works here: it is the file `app/post_capture._append_daily_note` appends
    session summaries to, on the same America/Los_Angeles date filename, so the
    line lands where a person already reads rather than where a webhook would.

    A failure to write is logged and reported as False; it never propagates,
    because a note that could not be written must not change what the run record
    or the retry budget say.
    """
    from zoneinfo import ZoneInfo

    try:
        la = ZoneInfo("America/Los_Angeles")
        now = datetime.datetime.now(la)
        path = _daily_note_dir() / f"{now.strftime('%Y-%m-%d')}.md"
        name = str(task.get("name") or "").strip() or "unnamed task"
        listed = ", ".join(f"{d:.1f}s" for d in durations)
        entry = (
            f"\n- {now.strftime('%H:%M %Z')} — **Autonomy #{task_id} ({name}): "
            f"{len(durations)} consecutive failures under "
            f"{_FAST_FAILURE_ALERT_SECONDS:.0f}s each ({listed}).** A run that "
            f"reached the engine takes minutes; these were refused before any "
            f"work — check `model:` and the engine route in the task file. "
            f"Details: autonomy-runs/{task_id}/\n"
        )
        if not path.exists():
            # Same OKF-conformant header a fresh daily note gets from
            # `app/post_capture._append_daily_note`: `type` is required
            # (scripts/vault/validate_okf.py) and a note born without it is a
            # conformance violation from its first line.
            frontmatter = yaml.safe_dump(
                {"segment": "memory", "tags": ["memory", "daily-notes"],
                 "type": "note",
                 "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S")},
                sort_keys=False, allow_unicode=True, default_flow_style=False,
            ).rstrip()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"---\n{frontmatter}\n---\n\n"
                f"# {now.strftime('%Y-%m-%d')} Daily Notes\n{entry}",
                encoding="utf-8")
        else:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(entry)
        return True
    except Exception as exc:  # noqa: BLE001 — a note is never worth the run
        logger.warning("Task #%s: could not write the fast-failure alert: %s",
                       task_id, exc)
        return False


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
    The infra path is bounded separately (#1085): `_INFRA_CEILING` consecutive
    infra failures inside one declared period rest the task until its next
    period starts, alert once, and clear themselves then. Never reclassify an
    infra failure as `task` to make it consume the budget — that is the
    fleet-wide disable this split exists to prevent.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    completed_at = now.isoformat()
    duration = (now - started_dt).total_seconds()

    # The count this run is charged under, computed BEFORE the record is
    # written so the record can carry it. Zero of 4,800 run records carried an
    # attempt number before #1086, so reading one file could not tell the 1st
    # try from the 3rd: #78's five runs on 2026-09-08/09 had to be reconstructed
    # by hand from five timestamps. An infra failure does not spend the budget
    # (see below), so it carries the unchanged count — the number the next task
    # failure will be counted against either way.
    failures = int(task.get("failure_count") or 0) + (1 if kind == "task" else 0)
    # Name the pre-images before anything else, so the record a person opens at
    # 03:00 says what is revertable and how. Reads the ledger's index; writes
    # nothing, and reverting is not this path's call to make (#963).
    ledger_extra, ledger_block = _change_ledger_note(
        str((extra or {}).get("session_id") or ""), run_id)
    if ledger_block:
        body = f"{body}\n\n{ledger_block}"
    _write_run_record(
        task_id=task_id, run_id=run_id, status="failed",
        started_at=started_at, completed_at=completed_at,
        duration_seconds=duration, summary=_failure_summary(summary), body=body,
        extra={**(extra or {}), **ledger_extra,
               "failure_kind": kind, "failure_count": failures},
    )

    max_retries = int(task.get("max_retries") or _DEFAULT_MAX_RETRIES)
    fields: dict = {"status": "up_next", "last_attempt": completed_at,
                    "updated": completed_at}
    disabled = False
    if kind == "task":
        fields["failure_count"] = failures
        if failures >= max_retries:
            fields["status"] = "failed"
            disabled = True

    # #1085: the ceiling on the infra path. `kind == "task"` above owns the
    # retry budget and `status: failed`; nothing here may touch either, because
    # "an outage must not disable the fleet" is the property `1d1bbb6` was
    # written for. What the infra path gets instead is its own persisted
    # counter and, at the ceiling, a rest of one declared period — a bound on
    # how often this task may re-dispatch, which the flat 600 s cooldown never
    # was (`_in_failure_cooldown` recomputes `n = max(1, 0) = 1` for a task
    # whose budget was never spent, so it re-answers the same 600 s forever).
    infra_ceiling_crossed = False
    if kind == "infra":
        infra_n = _next_infra_failure_count(task, now)
        if infra_n >= _INFRA_CEILING:
            infra_ceiling_crossed = True
            fields["infra_rest_until"] = (now + datetime.timedelta(
                seconds=_infra_rest_seconds(task))).isoformat()
            # Spent. The rest is the state now; a counter left reading 5 would
            # put the next single hiccup one step from a second rest, and clause
            # 4 is a self-resuming task, not a permanently twitchy one.
            infra_n = 0
        elif _parse_iso(task.get("infra_rest_until")) and not _in_infra_rest(task, now):
            # An already-expired rest is dropped, so the file never carries a
            # resume time that `next_run` contradicts. An UNEXPIRED one is left
            # exactly as found: a run invoked by hand while the task is resting
            # must not lift the rest.
            fields["infra_rest_until"] = None
        fields["infra_failure_count"] = infra_n

    if disabled:
        fields["next_run"] = None
    elif infra_ceiling_crossed:
        # ONE timestamp for both surfaces: the hold reads `infra_rest_until` and
        # the stall alarms read `next_run`, so two values here would be two
        # stories about when this task comes back.
        fields["next_run"] = fields["infra_rest_until"]
    else:
        cooldown = (_failure_cooldown_seconds({**task, **fields}) if kind == "task"
                    else float(_FAILURE_BACKOFF_BASE))
        fields["next_run"] = (now + datetime.timedelta(seconds=cooldown)).isoformat()
    _update_task_field(task_id, **fields)

    note = f"Run {run_id} — FAILED ({kind}): {summary[:280]} [full: autonomy-runs/{task_id}/{run_id}.md]"
    if disabled:
        note += (f" — DISABLED after {failures} consecutive failures; "
                 f"set status back to up_next to re-enable")
    if infra_ceiling_crossed:
        note += (f" — INFRA CEILING: {_INFRA_CEILING} consecutive infra failures in one "
                 f"declared period, resting until {fields['infra_rest_until']}. Resumes "
                 f"by itself then; the retry budget is untouched, so nobody has to "
                 f"re-enable it")
    _append_activity_log(task_id, note)

    # The sub-30s streak gets one line in today's daily note (#1209), written
    # here rather than at the disable below because the two are different facts:
    # a task with `max_retries: 10` can burn three refusals and still be
    # retrying, and a task with `max_retries: 1` disables on its FIRST fast
    # failure — waiting for the disable to say anything would have kept #85's
    # first two refusals silent. `alert` gates both transports together: a
    # caller that says do-not-notify-this-failure means it for the note too.
    #
    # Computed after the record above, so this run's own duration is in the
    # streak it is counted as.
    if kind == "task" and alert:
        streak = _fast_failure_streak(task_id)
        # Exactly the threshold, never "threshold or more": a fourth and fifth
        # fast failure of the same streak must not append a fourth identical line
        # to the same note. One line per streak is what "exactly one alert line"
        # asks for, and the streak length is the only thing that can tell the
        # third failure from the fourth.
        if len(streak) == _FAST_FAILURE_ALERT_STREAK:
            if _append_fast_failure_alert(task, task_id, streak):
                logger.error("Task #%s: %d consecutive failures under %.0fs (%s) — "
                             "alerted in today's daily note", task_id, len(streak),
                             _FAST_FAILURE_ALERT_SECONDS,
                             ", ".join(f"{d:.1f}s" for d in streak))

    if disabled and alert:
        try:
            from app.discord_notify import discord_alert
            await discord_alert(
                f"Autonomy task #{task_id} ({task.get('name')}) disabled after "
                f"{failures} consecutive failures. Last: {_failure_summary(summary)}"
            )
        except Exception as e:
            logger.warning("Alert dispatch failed for task #%s: %s", task_id, e)

    # Exactly one message per crossing, from the same seam as the disable alert
    # the infra path could never reach (`disabled` is only ever True for
    # `kind == "task"`, which is why a 600 s loop ran silently for three weeks).
    # One, not one-per-failure-afterwards: the counter is zeroed by the crossing
    # and nothing dispatches during the rest, so the next message here is the
    # next crossing. `alert` gates it with the same flag the disable uses.
    #
    # The cadence that buys, stated so nobody mistakes it for silence. A crossing
    # needs 5 failures and consecutive dispatches are at least
    # `_FAILURE_BACKOFF_BASE` apart (both `next_run` and `_in_failure_cooldown`
    # floor there), so two crossings for one task cannot come closer than
    # 4 x 600 s of failures plus the rest between them — about one message an
    # hour even for the loudest possible talker, an `every 15 min` task whose
    # rest is only its own 900 s period, and far less for anything slower. The
    # whole infra corpus on this box is 8 records across 5 days, so the observed
    # rate is orders under that bound.
    if infra_ceiling_crossed and alert:
        try:
            from app.discord_notify import discord_alert
            await discord_alert(
                f"Autonomy task #{task_id} ({task.get('name')}) hit the infra "
                f"ceiling: {_INFRA_CEILING} consecutive infra-classified failures "
                f"inside one declared period. Resting until "
                f"{fields['infra_rest_until']}, and resuming by itself then — its "
                f"retry budget is untouched ({failures}/{max_retries}), because an "
                f"outage must not disable a schedule. Last: {_failure_summary(summary)}"
            )
        except Exception as e:
            logger.warning("Infra-ceiling alert dispatch failed for task #%s: %s",
                           task_id, e)

    logger.error("Task #%s failed (%s, %d/%d): %s", task_id, kind, failures,
                 max_retries, _failure_summary(summary))
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


# Who asked for a run, stamped on its run record as `trigger` (#1437). On
# 2026-09-23 #38 and #56 ran at 14:25-14:31Z, outside their windows, and that
# slip pinned both nightly chains for two cycles — yet 0 of 43 run records
# carried any field saying who dispatched them, so the event the damage traces
# to could not be attributed. Callers name themselves with `run_trigger(...)`;
# a call that does not is recorded `direct`. `in_window` says whether the local
# hour sat inside the task's `preferred_hours` when the run started (absent for
# a task with no window): an out-of-window run is the one that re-anchors
# `last_run` and moves the chain, so it is the one worth finding.
RUN_TRIGGER: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "autonomy_run_trigger", default=None)
_ACTIVE_RUN: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar(
    "autonomy_active_run", default=None)


@contextlib.contextmanager
def run_trigger(name: str):
    """`with run_trigger("scheduler"): await run_task(...)` — names the caller."""
    token = RUN_TRIGGER.set(name)
    try:
        yield
    finally:
        RUN_TRIGGER.reset(token)


def _records_trigger(fn):
    """Scope the run's trigger to exactly one `run_task` call."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        token = _ACTIVE_RUN.set({"trigger": RUN_TRIGGER.get() or "direct"})
        try:
            return await fn(*args, **kwargs)
        finally:
            _ACTIVE_RUN.reset(token)
    return wrapper


@_records_trigger
async def run_task(task_id, *, max_duration: int | None = None) -> dict:
    """Execute a single autonomy task via Claude Agent SDK."""
    path = _find_task_file(task_id)
    if not path:
        return {"success": False, "error": f"Task #{task_id} not found"}

    task = _parse_task_file(path)
    if not task:
        return {"success": False, "error": f"Failed to parse task #{task_id}"}
    active = _ACTIVE_RUN.get()
    if active is not None and _effective_preferred_hours(task):
        active["in_window"] = _is_preferred_hour(task)

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

    # Resolve the model, then refuse a dispatch whose `model:` names no slot the
    # engine has (#1209) — and refuse it HERE, above the `in_progress` flip, so a
    # refusal leaves the task file exactly as it was found. Below the flip the
    # refusal would strand the row at `in_progress` until the stale-run window
    # expired, which is a second bug wearing the first one's clothes.
    task_model = _resolve_task_model(task_id, task)
    refusal = _model_dispatch_refusal(task_id, task_model)
    if refusal:
        logger.error("%s", refusal)
        _record_model_refusal(task_id, task_model, refusal)
        # `skipped`, not `failed`: no run happened, so nothing may be charged
        # against the retry budget, and `failed` here would let one typo in one
        # field disable a schedule — the exact outcome this check exists to stop.
        # `skipped` carries the text because the queue's `normalize_result`
        # honours a bare `skipped` key as both the status and the summary, so the
        # board row reads as a refusal that names its reason.
        return {"success": False, "status": "skipped", "skipped": refusal,
                "task_id": task_id, "error": refusal,
                "refusal": "unconfigured_model"}

    _update_task_field(task_id, status="in_progress", updated=now_iso)
    prompt = _build_task_prompt(task, skill_content)
    # #624: this route splices the whole SKILL.md in, uncapped, for the run —
    # the one where a shorter body is a real saving, so it is the one measured.
    from app.harness.skill_dispatch import ROUTE_AUTONOMY_TASK
    from app.skill_embed import record_skill_embed
    record_skill_embed(session_id, route=ROUTE_AUTONOMY_TASK,
                       skill=skill_name or skill_path, turn_id=run_id,
                       embedded_chars=len(skill_content),
                       source_chars=len(skill_content))
    # Appended payload only — see `_evidence_prompt` for the cache reasoning.
    prompt += _evidence_prompt(task_id)

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
        from app.harness.outbound_content import install_outbound_content_gate
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
        # #1136 — the sibling guard on the same registry. Armed with the task's
        # own scope rather than the contextvar default, because a scheduled
        # task is the path where a leak has no reviewer: the grant gate is
        # already armed with `grant_scope` two lines above, and a content gate
        # that read a different scope would be a second convention.
        install_outbound_content_gate(task_hooks, scope=grant_scope)
        # The deterministic senses (stall rescue, repetition, failure
        # payloads), which the observer's opt-in used to gate. The session and
        # turn ids are read off the options the loop binds.
        from app.harness.turn_guards import install_turn_guards
        install_turn_guards(task_hooks, platform="autonomy",
                            source=str(task.get("name") or ""))

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
            # A scheduled task has nobody at Mission Control; see
            # agent_mcp.annotations.hidden_on_surface.
            surface="worker",
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
            # D4: the scope `task_hooks` was armed with, so a Task child
            # re-arms the same gate. Explicit because the manual and HTTP
            # entry points run outside the pool's `current_scope`.
            grant_scope=grant_scope,
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
                    from app.deadline_anchor import compose_state_anchors
                    from app.routers._messages_inner_voice import goal_card_anchor
                    options.state_anchor = compose_state_anchors(
                        options.state_anchor, goal_card_anchor(iv_state))
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
        # The run's own dispatch record, for the acceptance grade (#623): what
        # the harness dispatched, not what the final text says it did.
        dispatch = DispatchTrace()

        try:
            from app.harness.events import trim_discarded
            from app.run_recorder import record_events
            recorded = record_events(
                run_query(messages, options),
                session_id=session_id, turn_id=run_id, prompt=prompt,
                model=task_model, source="autonomy",
                disallowed_tools=list(
                    getattr(options, "disallowed_tools", None) or []),
            )
            async with asyncio.timeout(timeout):
                async for evt in recorded:
                    if evt["type"] in ("tool_call", "tool_result"):
                        dispatch.observe(evt)
                    if evt["type"] == "text_delta":
                        final_response += evt["text"]
                    elif evt["type"] == "iteration_retry":
                        # A broken stream was re-requested (D7): its deltas
                        # are not part of the answer.
                        final_response, _ = trim_discarded(final_response, "", evt)
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
                    tool_errors=tool_errors, session_id=session_id,
                    grade=_acceptance_grade(task, dispatch, ""))
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
        # Recorded beside the literal `status="success"` below, never instead of
        # it: the grade decides nothing yet (#623, grade-only).
        grade = _acceptance_grade(task, dispatch, terminal_text)
        if grade is not None:
            meta[_ACCEPTANCE_GRADE_KEY] = grade

        _write_run_record(
            task_id=task_id, run_id=run_id, status="success",
            started_at=started_at, completed_at=completed_at,
            duration_seconds=duration, summary=_outcome_summary(final_response),
            body="\n\n".join(body_parts), extra=meta,
        )

        interval = _frequency_interval_seconds(task)
        completed_dt = datetime.datetime.fromisoformat(completed_at)
        # The due gate's own definition: an out-of-window completion writes the
        # next in-window slot, not `completed + interval` (#1437).
        next_run_dt = _next_run_after(task, completed_dt) if interval else None
        next_run_iso = next_run_dt.isoformat() if next_run_dt else None
        # last_attempt tracks EVERY attempt; last_run only successes. The
        # cooldown gate reads "last_attempt newer than last_run" as "the most
        # recent attempt failed", so a success must set both.
        _update_task_field(task_id, status="up_next", last_run=completed_at,
                           last_attempt=completed_at,
                           updated=completed_at, failure_count=0,
                           # Clause 2 of #1085: one success ends a run of hiccups
                           # outright, so a server that stubs one turn an hour for
                           # a week never accumulates to a ceiling.
                           infra_failure_count=0, infra_rest_until=None,
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
            "response_preview": _outcome_summary(final_response),
            "meta": meta,
        }
        if grade is not None:
            result[_ACCEPTANCE_GRADE_KEY] = grade
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


# ── Cross-task / cross-artifact claim rollup (#713) ──────────────────────────
#
# `compute_health` answers "how is task N doing?". The corrections log asks a
# different question — *which artifact has this fleet repeatedly misreported
# about?* — and no per-task sum can answer it, because the refutations about one
# artifact are scattered across the tasks that each made a claim about it. The
# case in the item: #39 asserts something about
# `_pipeline/reflection/signals-latest.md` that #38's run later disproves. Each
# refutation reaches only its own task's next prompt (the `evidence_gaps:<id>`
# watermark), so the pair never meets.
#
# Three rules carried over from #525, in the order they bite:
#
# 1. **The denominator is claims, never runs.** A `[SILENT]` run asserted
#    nothing, so a run-level denominator would pull every rate toward zero
#    every time a job said less. `claims_checked` here is summed from the same
#    per-claim statuses `compute_health` counts, from the same decode, so the
#    two can only disagree if one of them stops calling `_claim_status`.
# 2. **`runs_without_bundle` dominates ⇒ no rate.** A window where most runs
#    carry no bundle at all has a rate computed over a slice nobody chose, and a
#    clean 0.000 there is the report that reads as healthy while measuring
#    nothing. It reports `None` plus the count instead.
# 3. **A claim with no `check.path` is counted, never dropped.** It cannot be
#    attributed to an artifact, so it lands in `claims_without_path` beside the
#    totals rather than silently lowering them — otherwise the rollup and the
#    per-task sums diverge by exactly the rows that were malformed, which is the
#    divergence the reconciliation test exists to catch.

_ROLLUP_STATUSES = ("verified", "refuted", "insufficient")


def _claim_status(claim: dict) -> str:
    """The verifier's verdict on one stored claim, or "" when it carries none.

    One extraction for both consumers (`compute_health`'s per-task tallies and
    the artifact rollup), so a status rename cannot land in one and not the
    other. Anything the verifier did not write — absent, misspelt, a claim that
    is not an object — is "" and is counted by neither, exactly as #525's
    per-task block already behaved.
    """
    if not isinstance(claim, dict):
        return ""
    status = str(claim.get("status") or "")
    return status if status in _ROLLUP_STATUSES else ""


def _claim_path(claim: dict) -> str:
    """The artifact a claim is about: its check's `path`, or "" for none.

    `check.path` is the group-by key because it is the thing the corrections log
    names — a relative artifact path — and it is the only field the verifier
    itself resolves. A missing, non-string or blank path yields "", which routes
    the claim to `claims_without_path`.
    """
    check = claim.get("check")
    if not isinstance(check, dict):
        return ""
    path = check.get("path")
    return path.strip() if isinstance(path, str) and path.strip() else ""


def _tally() -> dict:
    """One accumulator: the four claim counts plus the bundle split.

    Same key names as a `compute_health` task row, on purpose — the equality
    this module promises is between the two of them, and names that match are a
    comparison a reader can do by eye.
    """
    return {"runs_with_bundle": 0, "runs_without_bundle": 0,
            "claims_checked": 0, "claims_verified": 0, "claims_refuted": 0,
            "claims_insufficient": 0}


def _tally_add(tally: dict, status: str) -> None:
    tally[f"claims_{status}"] += 1
    tally["claims_checked"] += 1


def _tally_rate(tally: dict) -> Optional[float]:
    """Refuted-or-insufficient over the claims that were CHECKED.

    `None` over zero claims — the unevaluable reading, never 0.0.
    """
    checked = tally["claims_checked"]
    if not checked:
        return None
    unverified = tally["claims_refuted"] + tally["claims_insufficient"]
    return round(unverified / checked, 3)


def _rollup_new() -> dict:
    return {"paths": {}, "by_task": {}, "claims_without_path": 0}


def _rollup_add(rollup: dict, tid: str, bundle: Optional[dict]) -> None:
    """Fold one run row's bundle in, under both its task and its artifact.

    A claim about an artifact another task also claimed about lands in that
    entry's `task_ids` AND its `per_task` map under both ids: the entry is the
    artifact's record, and which jobs made the claim is half of what makes it
    worth reading.
    """
    task = rollup["by_task"].setdefault(tid, _tally())
    if bundle is None:
        task["runs_without_bundle"] += 1
        return
    task["runs_with_bundle"] += 1
    for claim in bundle.get("claims") or []:
        status = _claim_status(claim)
        if not status:
            continue
        _tally_add(task, status)
        path = _claim_path(claim)
        if not path:
            rollup["claims_without_path"] += 1
            continue
        entry = rollup["paths"].setdefault(path, {**_tally(), "task_ids": set(),
                                                 "per_task": {}})
        entry["task_ids"].add(tid)
        _tally_add(entry, status)
        _tally_add(entry["per_task"].setdefault(tid, _tally()), status)


def _rollup_finalize(rollup: dict) -> dict:
    """Shape the accumulated rollup: entries most-misreported first, plus the marker.

    `entries` holds the artifacts carrying at least one refuted-or-insufficient
    claim, because that is the list a reader came for; `artifacts_checked` is
    printed beside it so an empty `entries` reads as "N artifacts, all clean"
    and not as "the fleet asserts nothing".
    """
    entries = []
    for path, e in rollup["paths"].items():
        if not (e["claims_refuted"] + e["claims_insufficient"]):
            continue
        entries.append({
            "path": path,
            "claims_checked": e["claims_checked"],
            "claims_verified": e["claims_verified"],
            "claims_refuted": e["claims_refuted"],
            "claims_insufficient": e["claims_insufficient"],
            "refuted_or_insufficient": (e["claims_refuted"]
                                        + e["claims_insufficient"]),
            "refuted_or_insufficient_rate": _tally_rate(e),
            "task_ids": sorted(e["task_ids"]),
            "per_task": {tid: {**t, "refuted_or_insufficient_rate": _tally_rate(t)}
                         for tid, t in sorted(e["per_task"].items())},
        })
    # Most-misreported first. Count before rate: three refutations out of four
    # claims outranks one out of one, because the corrections log chases the
    # artifact the fleet gets wrong most often, not the one with the prettiest
    # ratio. Ties break on the claim count then the path, so the order is a
    # function of the rows and not of dict insertion order.
    entries.sort(key=lambda x: (-x["refuted_or_insufficient"],
                                -x["claims_checked"], x["path"]))

    totals = _tally()
    for t in rollup["by_task"].values():
        for k in totals:
            totals[k] += t[k]
    # Rule 2: the window-wide rate is a measurement only when most runs actually
    # carried a bundle. `>` is "strictly more", so a 50/50 window still reports
    # a rate and names its own `runs_without_bundle`.
    unevaluable = totals["runs_without_bundle"] > totals["runs_with_bundle"]
    return {
        "entries": entries,
        "artifacts_checked": len(rollup["paths"]),
        "artifacts_with_refutations": len(entries),
        "runs_with_bundle": totals["runs_with_bundle"],
        "runs_without_bundle": totals["runs_without_bundle"],
        "claims_checked": totals["claims_checked"],
        "claims_verified": totals["claims_verified"],
        "claims_refuted": totals["claims_refuted"],
        "claims_insufficient": totals["claims_insufficient"],
        "claims_without_path": rollup["claims_without_path"],
        "refuted_or_insufficient_rate": (
            None if unevaluable else _tally_rate(totals)),
        "unevaluable": unevaluable,
        "unevaluable_reason": (
            f"{totals['runs_without_bundle']} of "
            f"{totals['runs_with_bundle'] + totals['runs_without_bundle']} runs "
            "in the window carry no evidence bundle"
            if unevaluable else None),
        "per_task": {tid: {**t, "refuted_or_insufficient_rate": _tally_rate(t)}
                     for tid, t in sorted(rollup["by_task"].items())},
    }


def claim_artifact_rollup(rows: list[dict]) -> dict:
    """Roll verified evidence claims up across tasks and by artifact path.

    The window is whatever the caller passed in: production hands it the same
    `queue.list_runs_joined("scheduled-task", since)` rows `compute_health`
    reads, so the two agree over identical rows by construction — the row filter
    (`skipped`), the task attribution (`_row_task_id`) and the claim extraction
    (`_row_claims` + `_claim_status`) are all shared. `compute_health` emits this
    same structure under its `artifacts` key; this function is that same query
    standing on its own, for a caller that wants the artifacts and not the whole
    fleet report.
    """
    rollup = _rollup_new()
    for row in rows:
        if row.get("status") == "skipped":
            continue
        _rollup_add(rollup, _row_task_id(row) or "unattributed", _row_claims(row))
    return _rollup_finalize(rollup)


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


def _health_window_fields(now: datetime.datetime, days: int,
                          oldest_input: Optional[str],
                          observed: bool) -> dict:
    """How much of the requested window the report can actually speak over (#1401).

    Two keys, both null when the store covers the window so that `days` alone
    describes the verdict:

    * `oldest_input` — the ISO stamp of the oldest run row the store holds for
      this source, named only when the verdict does not span what it claims.
    * `window_clamped_to_hours` — the hours of the requested window the verdict
      rests on: `now - max(window_start, oldest_input)`, and `0.0` when the
      window holds no usable row at all. Present (non-null) only when that is
      less than `days * 24`, i.e. whenever a reader who took `days` at face
      value would be over-reading the evidence by an unknown factor.

    The reference is the caller-supplied store age, never the window-filtered
    rows: those are empty in the case that matters most (every row predates the
    window), and an absent field there would wear the same face as the clean
    bill of health this exists to expose. `observed` is False when the window
    returned no row that reached the tallies — no rows, or only `skipped` ones —
    which is a verdict over nothing, and nothing is shorter than any window.

    Shaped this way because `app/routers/dashboard.py:458-462` already does the
    same job for the worker rollup with `window_hours` beside `window_start`,
    and its comment says why: a reader must be able to reproduce the number
    instead of guessing at "roughly two days"."""
    requested_hours = float(days) * 24.0
    oldest = _parse_iso(oldest_input)
    if oldest is None:
        return {"oldest_input": None, "window_clamped_to_hours": None}
    window_start = now - datetime.timedelta(hours=requested_hours)
    if not observed:
        span_hours = 0.0
    else:
        span_hours = (now - max(window_start, oldest)).total_seconds() / 3600.0
    if span_hours >= requested_hours - 1e-9:
        return {"oldest_input": None, "window_clamped_to_hours": None}
    return {"oldest_input": oldest_input,
            "window_clamped_to_hours": round(span_hours, 2)}


def compute_health(rows: list[dict], tasks: list[dict], days: int,
                   now: Optional[datetime.datetime] = None,
                   oldest_input: Optional[str] = None) -> dict:
    """Aggregate run rows into per-task and fleet health. Pure function.

    `oldest_input` is the age of the store the rows came from — the caller reads
    it with `WorkQueue.oldest_run_completed_at`, unfiltered by the window — and
    it is what lets the payload say "this 7-day verdict rests on 21 hours"
    instead of asserting 7 days it never saw. See `_health_window_fields`.

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
    # #713: the cross-task artifact rollup, accumulated in THIS pass over the
    # rows. A second pass over the same list would be a second row filter, and
    # two filters that drift are the divergence this payload must not have: the
    # whole reason `artifacts` is worth reading is that its per-task totals are
    # the same numbers the `tasks` rows above carry.
    rollup = _rollup_new()

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
        # #1137: a run killed by a restart or a pool cancel is recorded by the
        # boot-time sweep as `status='interrupted'` and carries no timeout meta
        # — it never reached a timeout, the process holding it died. Reading
        # only the meta and the summary text is what made those runs invisible:
        # they spent the wall clock, wrote no row, and the fleet reported zero
        # timeouts while items were re-queued from scratch. The status is the
        # fact; the summary is prose and is not consulted for it.
        # #642: the text fallback now needs a row that is actually not a success.
        # It matched `runs.summary`, which since this round is the CLOSING of the
        # model's response rather than its opening — so it reads ordinary report
        # prose, and prose about timeouts is what a report that recovered from one
        # contains. `run_scheduled-task_20260909_084821_30159b` is `status='success'`
        # with no timeout in its meta and was counted as a timeout because its
        # summary says "already retrying the CV files that timed out last run"; the
        # same sentence arriving inside the last 300 characters is now the common
        # case, not the accident. So narrative decides only for a row that is not a
        # success, which is every row the fallback was written for: 143 of the 163
        # `TimeoutError`-prefixed rows in `workers.db` are `status='failed'` with no
        # `meta_json` at all, and a head-sliced `TimeoutError: ...` prefix is the only
        # evidence they carry. A `success` row states its own verdict, and prose
        # cannot overturn it.
        narrative_timeout = bool(status) and status != "success" and (
            "timed out" in summary or summary.startswith("TimeoutError"))
        timeout = bool(meta.get("timeout")) or bool(meta.get("pool_timeout")) \
            or status == "interrupted" or narrative_timeout
        # An interrupted row is a failure too — the work did not complete — so
        # its `duration_seconds` lands in `wasted_hours` below and a restart
        # shows up in the number that exists to catch wasted GPU.
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
        _rollup_add(rollup, tid, bundle)
        if bundle is None:
            e["runs_without_bundle"] += 1
        else:
            e["runs_with_bundle"] += 1
            for claim in bundle.get("claims") or []:
                # One extraction, shared with the rollup, so the two tallies of
                # the same claim cannot be counted by two different rules.
                status = _claim_status(claim)
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
    #
    # `runs: 0` with a `fail_rate` of 0.0 was the defect (#1401): a task whose
    # task-file `last_run` sits inside the window while its run row does not —
    # which is what a young store does to every task older than the rebuild —
    # rendered byte-identically to a task that ran and passed. A rate over zero
    # runs is no rate, so these two are null, the same way
    # `refuted_or_insufficient_rate` below is null over zero claims and
    # `stalled[].fail_rate` is null over zero rows; `unobserved_in_window` is
    # the marker for a consumer flattening `tasks` and `idle_tasks` into one
    # table, where the list a row came from is no longer visible.
    seen = {t["task_id"] for t in out_tasks}
    idle = [{"task_id": str(t.get("id")), "name": t.get("name"),
             "status": t.get("status"), "frequency": t.get("frequency"),
             "runs": 0, "successes": 0, "failures": 0, "timeouts": 0, "empty": 0,
             "silent": 0, "fail_rate": None, "silent_rate": None, "gpu_hours": 0.0,
             "unobserved_in_window": True,
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
    # ONE read of the evidence totals. Summing them again here — as this block
    # did until #713 — gave the payload two answers to one question: the rollup
    # applies the unevaluable rule (a window whose runs carry no bundle reports
    # no rate), the old sum divided whenever any claim existed, and on the real
    # 2026-09-24 window they differed by exactly that (3 bundles / 46 bare rows:
    # `artifacts.refuted_or_insufficient_rate: null` next to
    # `fleet.refuted_or_insufficient_rate: 0.0`). A counting guard whose two
    # denominators are computed twice is the failure mode this file keeps
    # re-recording, so the fleet block now quotes the rollup instead of redoing
    # its arithmetic.
    artifacts = _rollup_finalize(rollup)
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
            # together or not at all. Quoted from the rollup rather than summed a
            # second time: see the note above the return, and `artifacts` below
            # for the same numbers grouped by artifact. A null rate here means
            # `artifacts.unevaluable` — over zero claims, or over a window whose
            # runs carry no bundle.
            "claims_checked": artifacts["claims_checked"],
            "claims_verified": artifacts["claims_verified"],
            "claims_refuted": artifacts["claims_refuted"],
            "claims_insufficient": artifacts["claims_insufficient"],
            "runs_with_bundle": artifacts["runs_with_bundle"],
            "runs_without_bundle": artifacts["runs_without_bundle"],
            "refuted_or_insufficient_rate": artifacts["refuted_or_insufficient_rate"],
            "evidence_unevaluable_reason": artifacts["unevaluable_reason"],
            "active_tasks": len([t for t in tasks if str(t.get("status")) == "up_next"]),
            "failed_tasks": [str(t.get("id")) for t in tasks
                             if str(t.get("status")) == "failed"],
            "paused_tasks": [str(t.get("id")) for t in tasks
                             if str(t.get("status")) == "paused"],
            # `oldest_input` / `window_clamped_to_hours`: how much of `days` the
            # rows above actually cover. Null when they cover it. Without these
            # the block reports `fail_rate` under a label ("the last N days")
            # that a young store silently contradicts (#1401) — which is how
            # this endpoint read "fleet healthy, nothing to pause" over ~21 h.
            **_health_window_fields(now, days, oldest_input, bool(by_task)),
        },
        "tasks": out_tasks,
        # #713: the artifact half of the same evidence, grouped by `check.path`
        # over these very rows. `tasks` answers "which job is unhealthy"; this
        # answers "which artifact does the fleet keep getting wrong", which no
        # per-task sum can — the refutations about one path are spread over the
        # tasks that each claimed something about it.
        "artifacts": _rollup_finalize(rollup),
        "idle_tasks": idle,
        # Tasks more than one period past their own next_run, whatever their
        # status — the field whose absence let a `fail_rate: 0.0` stand in for
        # "healthy" about a job that had not run in ~50 cycles (#1121).
        "stalled": stalled,
    }
