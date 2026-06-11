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
import shutil
import sys
import time
from pathlib import Path

TASKS_DIR = Path.home() / "lloyd" / "_pipeline" / "tasks"
SESSIONS_DIR = Path.home() / "lloyd" / "sessions"

TASK_LOG_MAX_AGE_DAYS = 30
SESSION_ARCHIVE_AGE_DAYS = 90

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete/gzip (default: dry-run report)")
    args = ap.parse_args()
    now = time.time()
    mode = "APPLY" if args.apply else "DRY RUN"

    logs_n, logs_b = sweep_task_logs(args.apply, now)
    sess_n, sess_b = sweep_sessions(args.apply, now)

    print(f"[retention-sweep] {mode}")
    print(f"  task logs >{TASK_LOG_MAX_AGE_DAYS}d:  "
          f"{logs_n} deleted, {logs_b / 1024:.0f} KiB freed")
    print(f"  sessions >{SESSION_ARCHIVE_AGE_DAYS}d inactive: "
          f"{sess_n} gzipped, {sess_b / 1024 / 1024:.1f} MiB "
          f"{'saved' if args.apply else 'candidate'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
