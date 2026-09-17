#!/usr/bin/env python3
"""Groundskeeper Retention Sweep

Bounds the unbounded-growth stores — the two from the 2026-06-11
architecture review (Tier 3.1), the autonomy stores added in June, and
the transcript scratch home from backlog #566:

1. ~/lloyd/_pipeline/tasks/   — background-bash task logs, never evicted
   by the harness. DELETE entries older than TASK_LOG_MAX_AGE_DAYS.
2. ~/lloyd/sessions/*.json    — session transcripts. GZIP (never delete —
   nightly trajectory extraction has long since processed them, and the
   .gz keeps them recoverable) sessions whose `last_active` is older than
   SESSION_ARCHIVE_AGE_DAYS. Live consumers all glob *.json, so archived
   sessions intentionally drop out of session lists and session_recall.
3. ~/lloyd/_pipeline/tmp/     — raw YouTube transcript scratch, the one
   directory skills/youtube-transcript and skills/youtube-content name as
   TRANSCRIPT_DIR (backlog #566). DELETE files older than
   TRANSCRIPT_MAX_AGE_DAYS. Deleting is safe here specifically because
   every video note records transcript_path + transcript_md5: the note is
   the durable artifact, the scratch file only has to survive long enough
   to be re-read, and a missing input stays detectable. Before this store
   existed nothing bounded it — and the sessions that skipped the named
   directory wrote into /tmp, where systemd-tmpfiles-clean.timer reaps
   them on a clock nobody in Lloyd controls.

7. ~/lloyd/sessions/*.tool-results/ — the tool-result spill dirs the harness
   writes BESIDE the transcripts (app/harness/tool_result_spill.py names them
   `<session_id>.tool-results`, and rung 2's `*.json` glob cannot see them).
   DELETE a directory whose NEWEST inner file is older than
   SPILL_MAX_AGE_DAYS, unless the session that owns it is still inside its own
   archive window — the point of spilling is that the model re-reads the file
   via Read, so a still-open session's spill is not garbage.

Age signal: sessions are aged by the `last_active` field in the JSON
(mtime lies — any reprocessing touches the file); task logs and transcript
scratch by mtime (a transcript is written once and only ever read back); a
spill directory by the newest mtime inside it, because a spill dir is created
on the session's first oversized tool result and then only ever has files
added to it — its own mtime is the age of the OLDEST spill in it.

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
# The one transcript scratch home. Both youtube skills name it as TRANSCRIPT_DIR, so this
# constant and that literal are the same directory — tests/test_youtube_artifact_phase.py
# pins the pair, because a scratch dir the sweep has never heard of is an unbounded store
# that reads as bounded. Deliberately not under /tmp (systemd-tmpfiles-clean.timer reaps it
# daily) and not in the vault (raw inputs are not what the Obsidian Sync quota is for).
TRANSCRIPT_SCRATCH_DIR = Path.home() / "lloyd" / "_pipeline" / "tmp"

TASK_LOG_MAX_AGE_DAYS = 30
SESSION_ARCHIVE_AGE_DAYS = 90
# Background runs — autonomy tasks and worker jobs — are recorded now, and at
# ~240 background sessions a day against ~14 chats (measured over the week to
# 2026-09-10) they are the overwhelming majority of the directory by count. They archive sooner than a conversation because they
# are read for a different reason and over a different span: a chat is
# something the user may come back to for months, a background transcript is
# forensics for "what did the thing that ran last night actually do". Same
# gzip-never-delete rule, so a run from six months ago is still recoverable —
# it is only out of the listings. That is how long a transcript stays
# openable by id; the Background tab itself is a recent view — the newest 150
# runs out of at most 600 files scanned, about the last 15 hours on screen.
BACKGROUND_SESSION_ARCHIVE_AGE_DAYS = 30
RUN_RECORD_MAX_AGE_DAYS = 30
ACTIVITY_LOG_MAX_ENTRIES = 200
CANDIDATE_MAX_AGE_DAYS = 30
# How long a raw transcript survives after it was fetched. Long enough for the two follow-ups
# that actually re-read it — a re-digest within the week, and an audit asking whether a digest
# matched its input — and short enough that the directory holds a rolling month instead of
# everything ever extracted. Measured 2026-09-14: 5 files, 208,885 B, oldest mtime 2026-09-08
# — every byte of it written by a session and none of it ever deleted. Deleting is safe at
# all only because the video note carries transcript_path + transcript_md5.
TRANSCRIPT_MAX_AGE_DAYS = 30
# The sessions store's sidecar: one `<session_id>.tool-results/` directory per
# session that ever spilled, written by app/harness/tool_result_spill.py next to the
# transcript. Rung 2 globs `*.json`, so these were invisible to every age rule from the
# day the spill module landed — the same silent-inertness shape `_run_record_age_seconds`
# records for run records, caused here by the store changing shape under a fixed glob.
# Measured 2026-09-16: 804 dirs / 5,617 files / 130.1 MiB = 23.9 % of the store, and 372
# of the 804 dirs (61.4 MiB) name an id with no `sessions/<id>.json` at all — task
# subagents, bench trials and the e2e harness spill under ids that never get a transcript,
# so no rule that joins to a session can reach them and the window has to sit on the
# directory itself. 30 d matches BACKGROUND_SESSION_ARCHIVE_AGE_DAYS: a background
# transcript is out of the listings by then, so its spilled tool results are forensics on
# a run nobody is reading any more. The name of the directory is derived from
# SPILL_DIR_SUFFIX, and tests/test_retention_sweep.py pins that suffix against the writer's
# own `_spill_dir()` — a spill root that moved or renamed must fail a test, not read as
# "0 candidates, therefore bounded".
SPILL_DIR_SUFFIX = ".tool-results"
SPILL_DIR_GLOB = f"*{SPILL_DIR_SUFFIX}"
SPILL_MAX_AGE_DAYS = 30
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


_PLATFORM_RE = re.compile(r'"platform":\s*"([^"]+)"')

#: Kept in step with `app.sessions_io.NON_USER_PLATFORMS`. Restated rather
#: than imported because this script is stdlib-only and runs from cron with no
#: venv; `tests/test_session_platform_checks.py` pins that the two agree.
NON_USER_PLATFORMS = ("autonomy", "worker")


def _session_platform(path: Path) -> str:
    """`platform` from the session JSON's head, or "" if unreadable.

    Same 4 KB prefix read as `_session_age_days`, for the same reason: the
    directory is hundreds of megabytes of transcript and this script wants two
    scalar fields out of each file.
    """
    try:
        head = path.open("r", encoding="utf-8", errors="replace").read(4096)
        m = _PLATFORM_RE.search(head)
        return m.group(1) if m else ""
    except Exception:
        return ""


def _archive_age_for(path: Path) -> int:
    """How long this session is kept in the live listings.

    Falls back to the *conversation* policy when the platform cannot be read.
    A truncated or unreadable head must never shorten a retention window:
    keeping a background run three months too long costs a few kilobytes,
    archiving a real conversation two months early loses it from the history
    the user actually reads.
    """
    platform = _session_platform(path)
    if platform in NON_USER_PLATFORMS:
        return BACKGROUND_SESSION_ARCHIVE_AGE_DAYS
    return SESSION_ARCHIVE_AGE_DAYS


def sweep_sessions(apply: bool, now: float) -> tuple[int, int]:
    """Gzip inactive sessions — SESSION_ARCHIVE_AGE_DAYS for a conversation,
    BACKGROUND_SESSION_ARCHIVE_AGE_DAYS for a background run.
    Returns (count, bytes_saved)."""
    count = saved = 0
    if not SESSIONS_DIR.exists():
        return 0, 0
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            if path.is_symlink():
                continue
            if _session_age_days(path, now) < _archive_age_for(path):
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


def sweep_transcript_scratch(apply: bool, now: float) -> tuple[int, int]:
    """Delete raw transcript scratch older than TRANSCRIPT_MAX_AGE_DAYS.

    Only plain files directly in TRANSCRIPT_SCRATCH_DIR are touched, and symlinks are
    skipped — the directory is written by ad-hoc extraction sessions, so what is in it is
    not guaranteed to be a transcript, and a sweep of a scratch dir must never follow a
    symlink out of it. Returns (count, bytes)."""
    count = freed = 0
    if not TRANSCRIPT_SCRATCH_DIR.exists():
        return 0, 0
    cutoff = now - TRANSCRIPT_MAX_AGE_DAYS * 86400
    for path in TRANSCRIPT_SCRATCH_DIR.iterdir():
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_mtime >= cutoff:
                continue
            size = path.stat().st_size
            if apply:
                path.unlink()
            count += 1
            freed += size
        except OSError as e:
            print(f"  ! skip {path.name}: {e}", file=sys.stderr)
    return count, freed


def _spill_contents(dir: Path) -> tuple[float | None, int]:
    """(newest mtime, total bytes) over the plain files inside a spill dir.

    One walk, because the byte total reported for a candidate has to be the bytes
    the delete actually reclaims. Symlinks are skipped rather than followed: the
    harness writes plain files here, so a link in a spill dir is not a spill, and
    rmtree would take the link either way — counting its target's size would report
    a reclaim the delete does not deliver.
    """
    newest: float | None = None
    total = 0
    for entry in dir.rglob("*"):
        try:
            if entry.is_symlink() or not entry.is_file():
                continue
            st = entry.stat()
        except OSError:
            continue
        if newest is None or st.st_mtime > newest:
            newest = st.st_mtime
        total += st.st_size
    return newest, total


def sweep_session_spills(apply: bool, now: float) -> tuple[int, int]:
    """Delete spill dirs whose newest inner file predates SPILL_MAX_AGE_DAYS.

    Returns (count, bytes).

    A directory is a candidate on the age of its NEWEST file, never the dir's own
    mtime and never its oldest spill: taking a directory that a session is still
    writing into would delete the newest results, which are the ones in use.

    One thing spares a candidate: the session that owns it is still inside its own
    archive window. Spill exists so the model can re-read a large tool result from
    disk with `Read` (tool_result_spill.py:10-14), so the read-back target of a live
    session must survive even when the directory has been on disk past the window —
    which is exactly what a long-running conversation's spill dir looks like. Once
    `sessions/<sid>.json` is itself past its archive line the session is about to
    leave the listings anyway and its spill goes with it; when there is no transcript
    at all (the task/bench/e2e id class) the age rule is the only thing that can
    reach the directory, and it applies unconditionally — that is the class this
    rung exists for.

    `*.changes` is deliberately NOT swept even though it sits in the same directory
    with the same shape: those directories are the change ledger's own pre-images,
    which `_change_ledger.py`'s `prune()` ages on its own clock and a sweep would
    destroy the very state the ledger keeps.
    """
    count = freed = 0
    if not SESSIONS_DIR.exists():
        return 0, 0
    cutoff = now - SPILL_MAX_AGE_DAYS * 86400
    for path in SESSIONS_DIR.glob(SPILL_DIR_GLOB):
        try:
            if path.is_symlink() or not path.is_dir():
                continue
            newest, size = _spill_contents(path)
            if newest is None:
                # An empty directory holds nothing to age by; the directory itself is
                # the only signal, and an empty spill dir is worthless regardless.
                newest = path.stat().st_mtime
            if newest >= cutoff:
                continue
            session_json = SESSIONS_DIR / (path.name[: -len(SPILL_DIR_SUFFIX)] + ".json")
            if (session_json.is_file()
                    and _session_age_days(session_json, now) < _archive_age_for(session_json)):
                continue
            if apply:
                shutil.rmtree(path)
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
    scr_n, scr_b = sweep_transcript_scratch(args.apply, now)
    spill_n, spill_b = sweep_session_spills(args.apply, now)

    print(f"[retention-sweep] {mode}")
    print(f"  task logs >{TASK_LOG_MAX_AGE_DAYS}d:  "
          f"{logs_n} deleted, {logs_b / 1024:.0f} KiB freed")
    print(f"  sessions >{SESSION_ARCHIVE_AGE_DAYS}d inactive "
          f"(background >{BACKGROUND_SESSION_ARCHIVE_AGE_DAYS}d): "
          f"{sess_n} gzipped, {sess_b / 1024 / 1024:.1f} MiB "
          f"{'saved' if args.apply else 'candidate'}")
    print(f"  autonomy runs >{RUN_RECORD_MAX_AGE_DAYS}d: "
          f"{runs_n} deleted, {runs_b / 1024 / 1024:.1f} MiB freed")
    print(f"  activity logs: {act_f} task files truncated, "
          f"{act_l} entries pruned (keep last {ACTIVITY_LOG_MAX_ENTRIES})")
    print(f"  skill candidates >{CANDIDATE_MAX_AGE_DAYS}d processed: "
          f"{cand_n} deleted, {cand_b / 1024:.0f} KiB freed")
    print(f"  transcript scratch >{TRANSCRIPT_MAX_AGE_DAYS}d: "
          f"{scr_n} deleted, {scr_b / 1024:.0f} KiB freed")
    # Same line in both modes, like the other six: the operator approves `--apply` from
    # the dry run's numbers, so a dry run that differs in shape from an applied run is a
    # line nobody can compare against the run they just approved.
    print(f"  session spill dirs (*{SPILL_DIR_SUFFIX}) >{SPILL_MAX_AGE_DAYS}d: "
          f"{spill_n} deleted, {spill_b / 1024:.0f} KiB freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
