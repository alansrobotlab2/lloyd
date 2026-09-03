#!/usr/bin/env python3
"""Groundskeeper Retention Sweep

Bounds the two unbounded-growth stores (2026-06-11 architecture review,
Tier 3.1):

1. ~/lloyd/_pipeline/tasks/   — background-bash task logs, never evicted
   by the harness. DELETE entries older than TASK_LOG_MAX_AGE_DAYS.
2. ~/lloyd/sessions/*.json    — session transcripts. GZIP (never delete —
   nightly trajectory extraction has long since processed them, and the
   .gz keeps them recoverable) sessions whose `last_active` is older than
   SESSION_ARCHIVE_AGE_DAYS. Live consumers all glob *.json, so archived
   sessions intentionally drop out of session lists and session_recall.

Age signal: sessions are aged by the `last_active` field in the JSON
(mtime lies — any reprocessing touches the file); task logs by mtime.

Usage:
    retention-sweep.py            # dry run — report only
    retention-sweep.py --apply    # actually delete/gzip
"""

import argparse
import gzip
import json
import re
from datetime import datetime, timezone
import shutil
import sys
import time
from pathlib import Path

TASKS_DIR = Path.home() / "lloyd" / "_pipeline" / "tasks"
SESSIONS_DIR = Path.home() / "lloyd" / "sessions"
AUTONOMY_RUNS_DIR = Path.home() / "lloyd" / "autonomy-runs"
AUTONOMY_TASKS_DIR = Path.home() / "obsidian" / "autonomy"
CANDIDATES_DIR = Path.home() / "lloyd" / "_pipeline" / "skills" / "candidates"

TASK_LOG_MAX_AGE_DAYS = 30
SESSION_ARCHIVE_AGE_DAYS = 90
RUN_RECORD_MAX_AGE_DAYS = 30
ACTIVITY_LOG_MAX_ENTRIES = 200
CANDIDATE_MAX_AGE_DAYS = 30
# candidates still awaiting action are never pruned regardless of age
CANDIDATE_KEEP_STATUSES = ("pending", "proposed", "flagged_for_authoring")
_CANDIDATE_STATUS_RE = re.compile(r"^status:\s*(\S+)", re.MULTILINE)

# last_active sits near the top of the session JSON (insertion order,
# indent=2) — read a small prefix instead of parsing 390MB of transcripts.
_LAST_ACTIVE_RE = re.compile(r'"last_active":\s*"([^"]+)"')


def _session_age_days(path: Path, now: float) -> float:
    try:
        head = path.open("r", encoding="utf-8", errors="replace").read(4096)
        m = _LAST_ACTIVE_RE.search(head)
        if m:
            from datetime import datetime
            ts = datetime.fromisoformat(m.group(1)).timestamp()
            return (now - ts) / 86400.0
    except Exception:
        pass
    try:
        return (now - path.stat().st_mtime) / 86400.0
    except OSError:
        return 0.0


def sweep_task_logs(apply: bool, now: float) -> tuple[int, int]:
    """Delete _pipeline/tasks entries older than TASK_LOG_MAX_AGE_DAYS.
    Returns (count, bytes)."""
    count = freed = 0
    if not TASKS_DIR.exists():
        return 0, 0
    cutoff = now - TASK_LOG_MAX_AGE_DAYS * 86400
    for entry in TASKS_DIR.iterdir():
        try:
            if entry.is_symlink() or entry.stat().st_mtime >= cutoff:
                continue
            size = (
                sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                if entry.is_dir() else entry.stat().st_size
            )
            if apply:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            count += 1
            freed += size
        except OSError as e:
            print(f"  ! skip {entry.name}: {e}", file=sys.stderr)
    return count, freed


def sweep_sessions(apply: bool, now: float) -> tuple[int, int]:
    """Gzip sessions inactive for SESSION_ARCHIVE_AGE_DAYS+.
    Returns (count, bytes_saved)."""
    count = saved = 0
    if not SESSIONS_DIR.exists():
        return 0, 0
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            if path.is_symlink():
                continue
            if _session_age_days(path, now) < SESSION_ARCHIVE_AGE_DAYS:
                continue
            gz_path = path.with_suffix(".json.gz")
            orig = path.stat().st_size
            if apply:
                with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                # Sanity: archive must round-trip as JSON before the
                # original is deleted.
                with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
                    json.load(fh)
                path.unlink()
                saved += orig - gz_path.stat().st_size
            else:
                saved += orig  # dry-run: report candidate size
            count += 1
        except Exception as e:
            print(f"  ! skip {path.name}: {e}", file=sys.stderr)
    return count, saved


_RUN_TS_RE = re.compile(r"^(?:completed_at|started_at):\s*'?([0-9]{4}-[0-9]{2}-[0-9]{2}"
                        r"[T ][0-9:.+-]*)", re.MULTILINE)


def _run_record_age_seconds(path: Path, now: float) -> float | None:
    """Age of a run record, from its frontmatter timestamp.

    mtime is not usable here: a bulk operation on 2026-08-22 reset the mtime of
    every run record, which made this sweep silently inert — `find autonomy-runs
    -name 'run_*.md' -mtime +30` matched 0 of 3,350 files. The frontmatter
    timestamp is what the record actually means. Falls back to mtime when the
    frontmatter is unreadable.
    """
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:600]
        m = _RUN_TS_RE.search(head)
        if m:
            ts = m.group(1).strip().replace(" ", "T")
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return now - dt.timestamp()
    except (OSError, ValueError):
        pass
    try:
        return now - path.stat().st_mtime
    except OSError:
        return None


def sweep_autonomy_runs(apply: bool, now: float) -> tuple[int, int]:
    """Delete autonomy run records older than RUN_RECORD_MAX_AGE_DAYS.

    Matches both the current `run_<task>_<ts>.md` names and the legacy
    epoch-millisecond names (e.g. `1774837994780.md`) written by the
    pre-2026-03 scheduler — 844 of those were permanently exempt from retention
    because the old glob only matched `run_*.md`. Non-record files such as
    wiki-sweep-latest.json are still left alone.
    """
    count = freed = 0
    if not AUTONOMY_RUNS_DIR.exists():
        return 0, 0
    max_age = RUN_RECORD_MAX_AGE_DAYS * 86400
    for path in AUTONOMY_RUNS_DIR.glob("*/*.md"):
        name = path.name
        if not (name.startswith("run_") or name[:-3].isdigit()):
            continue
        try:
            if path.is_symlink():
                continue
            age = _run_record_age_seconds(path, now)
            if age is None or age < max_age:
                continue
            size = path.stat().st_size
            if apply:
                path.unlink()
            count += 1
            freed += size
        except OSError as e:
            print(f"  ! skip {path.name}: {e}", file=sys.stderr)
    return count, freed


def sweep_activity_logs(apply: bool) -> tuple[int, int]:
    """Truncate '## Activity Log' sections in autonomy task files to the
    last ACTIVITY_LOG_MAX_ENTRIES entries. Returns (files_touched,
    lines_removed)."""
    files = removed = 0
    if not AUTONOMY_TASKS_DIR.exists():
        return 0, 0
    for path in sorted(AUTONOMY_TASKS_DIR.glob("[0-9]*.md")):
        try:
            lines = path.read_text(encoding="utf-8").split("\n")
        except OSError as e:
            print(f"  ! skip {path.name}: {e}", file=sys.stderr)
            continue
        try:
            start = next(i for i, ln in enumerate(lines)
                         if ln.strip().lower() == "## activity log") + 1
        except StopIteration:
            continue
        entries = [i for i in range(start, len(lines))
                   if lines[i].lstrip().startswith("- ")
                   and not lines[i].lstrip().startswith("- …")]
        excess = entries[:-ACTIVITY_LOG_MAX_ENTRIES] \
            if len(entries) > ACTIVITY_LOG_MAX_ENTRIES else []
        if not excess:
            continue
        # fold any prior marker lines into the new one
        prior = n_markers = 0
        for i in range(start, len(lines)):
            stripped = lines[i].lstrip()
            if stripped.startswith("- …"):
                m = re.search(r"(\d+) older entries", stripped)
                prior += int(m.group(1)) if m else 0
                n_markers += 1
                excess.append(i)
        drop = set(excess)
        kept = [ln for i, ln in enumerate(lines) if i not in drop]
        marker = (f"- … {len(drop) - n_markers + prior} older entries "
                  f"pruned by retention sweep "
                  f"(keeping last {ACTIVITY_LOG_MAX_ENTRIES})")
        kept.insert(start, marker)
        if apply:
            path.write_text("\n".join(kept), encoding="utf-8")
        files += 1
        removed += len(drop) - n_markers
    return files, removed


def sweep_candidates(apply: bool, now: float) -> tuple[int, int]:
    """Delete processed skill-candidate files older than
    CANDIDATE_MAX_AGE_DAYS. Candidates whose status is still actionable
    (CANDIDATE_KEEP_STATUSES) are kept. Returns (count, bytes)."""
    count = freed = 0
    if not CANDIDATES_DIR.exists():
        return 0, 0
    cutoff = now - CANDIDATE_MAX_AGE_DAYS * 86400
    for path in CANDIDATES_DIR.glob("candidate-*.md"):
        try:
            if path.is_symlink() or path.stat().st_mtime >= cutoff:
                continue
            m = _CANDIDATE_STATUS_RE.search(
                path.read_text(encoding="utf-8", errors="replace"))
            status = (m.group(1) if m else "").lower()
            if any(status.startswith(k) for k in CANDIDATE_KEEP_STATUSES):
                continue
            size = path.stat().st_size
            if apply:
                path.unlink()
            count += 1
            freed += size
        except OSError as e:
            print(f"  ! skip {path.name}: {e}", file=sys.stderr)
    return count, freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete/gzip (default: dry-run report)")
    args = ap.parse_args()
    now = time.time()
    mode = "APPLY" if args.apply else "DRY RUN"

    logs_n, logs_b = sweep_task_logs(args.apply, now)
    sess_n, sess_b = sweep_sessions(args.apply, now)
    runs_n, runs_b = sweep_autonomy_runs(args.apply, now)
    act_f, act_l = sweep_activity_logs(args.apply)
    cand_n, cand_b = sweep_candidates(args.apply, now)

    print(f"[retention-sweep] {mode}")
    print(f"  task logs >{TASK_LOG_MAX_AGE_DAYS}d:  "
          f"{logs_n} deleted, {logs_b / 1024:.0f} KiB freed")
    print(f"  sessions >{SESSION_ARCHIVE_AGE_DAYS}d inactive: "
          f"{sess_n} gzipped, {sess_b / 1024 / 1024:.1f} MiB "
          f"{'saved' if args.apply else 'candidate'}")
    print(f"  autonomy runs >{RUN_RECORD_MAX_AGE_DAYS}d: "
          f"{runs_n} deleted, {runs_b / 1024 / 1024:.1f} MiB freed")
    print(f"  activity logs: {act_f} task files truncated, "
          f"{act_l} entries pruned (keep last {ACTIVITY_LOG_MAX_ENTRIES})")
    print(f"  skill candidates >{CANDIDATE_MAX_AGE_DAYS}d processed: "
          f"{cand_n} deleted, {cand_b / 1024:.0f} KiB freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
