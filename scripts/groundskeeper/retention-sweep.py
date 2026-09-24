#!/usr/bin/env python3
"""Groundskeeper Retention Sweep

Bounds the unbounded-growth stores — the two from the 2026-06-11
architecture review (Tier 3.1), the autonomy stores added in June, and
the transcript scratch home from backlog #566:

1. ~/lloyd-data/_pipeline/tasks/ — background-bash task logs, never evicted
   by the harness. DELETE entries older than TASK_LOG_MAX_AGE_DAYS.
2. ~/lloyd-data/sessions/*.json — session transcripts. GZIP (never delete —
   nightly trajectory extraction has long since processed them, and the
   .gz keeps them recoverable) sessions whose `last_active` is older than
   SESSION_ARCHIVE_AGE_DAYS. Live consumers all glob *.json, so archived
   sessions intentionally drop out of session lists and session_recall.
3. ~/lloyd-data/_pipeline/tmp/ — raw YouTube transcript scratch, the one
   directory skills/youtube-transcript and skills/youtube-content name as
   TRANSCRIPT_DIR (backlog #566). DELETE files older than
   TRANSCRIPT_MAX_AGE_DAYS. Deleting is safe here specifically because
   every video note records transcript_path + transcript_md5: the note is
   the durable artifact, the scratch file only has to survive long enough
   to be re-read, and a missing input stays detectable. Before this store
   existed nothing bounded it — and the sessions that skipped the named
   directory wrote into /tmp, where systemd-tmpfiles-clean.timer reaps
   them on a clock nobody in Lloyd controls.

7. ~/lloyd-data/sessions/*.tool-results/ — the tool-result spill dirs the harness
   writes BESIDE the transcripts (app/harness/tool_result_spill.py names them
   `<session_id>.tool-results`, and rung 2's `*.json` glob cannot see them).
   DELETE a directory whose NEWEST inner file is older than
   SPILL_MAX_AGE_DAYS, unless the session that owns it is still inside its own
   archive window — the point of spilling is that the model re-reads the file
   via Read, so a still-open session's spill is not garbage.

8. ~/lloyd-data/workers.db table `runs` — the queue's own run history, appended by
   `WorkQueue.record_run` and pruned by nothing until now: `DELETE FROM runs` appeared
   in no commit in this repository's history (#1018). DELETE rows whose `completed_at`
   is older than WORKER_RUN_MAX_AGE_DAYS. The db is reached through `WORKERS_DB`, which
   is `workers.queue.configured_db_path()` — the queue's own resolution, imported rather
   than restated, because a second rule here would prune a file the queue never writes to
   and report success. No VACUUM: a DELETE frees pages for the next INSERT to reuse, which
   bounds the file, and VACUUM takes an exclusive lock on a WAL database the live pool is
   writing. The sibling `queue` table is NOT pruned — its horizon is an open decision on
   #1018. A database the pool will not release inside the bounded busy wait is reported
   `SKIPPED (database locked …)` and exits 0, never a traceback: a weekly sweep that
   fails because the queue was busy reads as "the store is fine" to everything downstream.

Age signal: sessions are aged by the `last_active` field in the JSON
(mtime lies — any reprocessing touches the file); task logs and transcript
scratch by mtime (a transcript is written once and only ever read back); a
spill directory by the newest mtime inside it, because a spill dir is created
on the session's first oversized tool result and then only ever has files
added to it — its own mtime is the age of the OLDEST spill in it.

Which root: every store above sits under `DATA_ROOT`, resolved by the one copy
of `app.paths`' three rules (`app/data_root.py`, stdlib-only so this file can
import it with no venv): `$LLOYD_DATA`, else `<passwd home>/lloyd-data` for the
production checkout — and only while that root carries `.lloyd-data-root`, since
without the marker the sweep refuses and exits 2 rather than fall back — else
`<tree>/.lloyd-data` for any other checkout. So a sandbox's or a round's
`--apply` reaches only that tree's own data (#1415). The run prints the root it
resolved, in dry run and in `--apply` alike, above the numbers it describes. One
rung writes outside the data root: bounding the activity logs in the vault's
`autonomy/*.md`, which `LLOYD_VAULT_ROOT` points at a copy. The bound is two rules,
not one — the last `ACTIVITY_LOG_MAX_ENTRIES` entry bullets are kept and every other
non-blank line under the heading is removed, because a line the entry cap does not
count is a line the cap could never remove (#845).

Usage:
    retention-sweep.py            # dry run — report only
    retention-sweep.py --apply    # actually delete/gzip
"""

import argparse
import gzip
import json
import re
from datetime import datetime, timedelta, timezone
import shutil
import sqlite3
import sys
import time
from pathlib import Path

# Import `app.data_root` — the stdlib-only half of the resolver — not `app.paths`,
# which drags in the package and needs the project venv this script does not have
# when cron, the weekly autonomy task #79, or an `sh -c` child runs it.
_TREE = Path(__file__).resolve().parents[2]
for _p in (_TREE, _TREE / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
del _p  # a module-level Path left by that loop is a name a test could delete from
try:
    from app.data_root import (DataRootMissing, resolve_data_root_for_tree,
                               vault_root)
except ModuleNotFoundError:
    # `app` is a name a stray `app.py` on the path can shadow, and nothing else in
    # this file imports `app.*`. The second spelling is the same file, and
    # tests/test_retention_sweep.py pins both against the one resolver.
    from data_root import (DataRootMissing, resolve_data_root_for_tree,
                           vault_root)

# The runtime data root — `app.paths.DATA_ROOT`'s three rules, read from the one
# copy of them (#1415). It used to be restated here as
# `${LLOYD_DATA:-~/lloyd-data}`, which is rule 2 with neither the marker check
# nor rule 3, so an unset `LLOYD_DATA` — the normal state of every shell, and of
# a sandbox or a round's checkout — pointed this script's deletes, gzips and
# rewrites at the LIVE root while its own report said it had swept nothing of the
# tree it was run from. A misdirected sweep was also uncatchable: the Bash delete
# guard parses command strings and never sees a Python `unlink`, and the tripwire
# needs 10 % and 200 files gone inside 15 minutes, ~5,600 of the live root's
# ~56,000.
try:
    DATA_ROOT = resolve_data_root_for_tree(_TREE)
except DataRootMissing as exc:
    # Refuse, loudly, before a single file is opened. Falling back to
    # `<tree>/.lloyd-data` here would start the second copy of everything the
    # data move exists to prevent, and falling back to the tree would put this
    # sweep's destructive reach inside a code checkout.
    print(f"[retention-sweep] REFUSING to run: {exc}", file=sys.stderr)
    sys.exit(2)

TASKS_DIR = DATA_ROOT / "_pipeline" / "tasks"
SESSIONS_DIR = DATA_ROOT / "sessions"
AUTONOMY_RUNS_DIR = DATA_ROOT / "autonomy-runs"
# The one store this sweep touches outside the data root: it truncates the
# `## Activity Log` section of the autonomy task files in the vault. Derived the
# way `app.paths.VAULT_ROOT` is (`Path.home() / "obsidian"`), which under a
# gate's HOME is the round's link into the LIVE vault —
# `scripts/automod/worktree.py::HOME_LINK_SKIP` skips `lloyd` and `lloyd-data`
# and links everything else, and `worktree.py:86` is explicit that the symlink
# stops `rmtree`, not `write_text`. So this rung gets the override
# `vaultwatch.py:46` and `scripts/backup/backup-vault.sh:28` already read: a
# round can point it at a copy of the task files instead of the live ones.
AUTONOMY_TASKS_DIR = vault_root() / "autonomy"
CANDIDATES_DIR = DATA_ROOT / "_pipeline" / "skills" / "candidates"
# The one transcript scratch home. Both youtube skills name it as TRANSCRIPT_DIR, so this
# constant and that literal are the same directory — tests/test_youtube_artifact_phase.py
# pins the pair, because a scratch dir the sweep has never heard of is an unbounded store
# that reads as bounded. Deliberately not under /tmp (systemd-tmpfiles-clean.timer reaps it
# daily) and not in the vault (raw inputs are not what the Obsidian Sync quota is for).
TRANSCRIPT_SCRATCH_DIR = DATA_ROOT / "_pipeline" / "tmp"

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
# The queue's run history (`workers.db` table `runs`). Same horizon as the markdown run
# records above, because they are the same event written twice — once by the task runner
# into `autonomy-runs/`, once by the queue into sqlite — and a reader who compares "what
# ran last month" across the two surfaces should not get two different answers. The name
# is the one `skills/retention-sweep/SKILL.md` documents; the two are pinned together by
# tests/test_retention_sweep.py.
WORKER_RUN_MAX_AGE_DAYS = 30
# How long to wait for a write lock the live queue pool is holding before giving up on
# this store. Bounded on purpose: the sweep is a weekly background job and the queue is
# the thing users wait on, so the sweep yields rather than contends — and the bound is
# what makes the skip reportable at all, since an unbounded wait would hang the job that
# is supposed to report it.
DB_BUSY_TIMEOUT_MS = 5000
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


#: An Activity Log entry is a bullet at column 0 (`autonomy.append_activity_line`
#: writes `- <ts>: <note>`, one physical line). An indented bullet is the leaked
#: continuation of a multi-line note, not an entry, so it is junk like any other
#: non-bullet — which is why this test is `startswith` and not `lstrip().startswith`:
#: the lstrip form was half of why the junk could not be removed (#845).
ENTRY_BULLET_PREFIX = "- "
#: The one line the sweep itself writes. `- …` is a bullet, so a check that
#: counts non-bullets as junk has to name it, and `prior`/`prior_lines` below are
#: parsed back out of it.
PRUNE_MARKER_PREFIX = "- …"


def sweep_activity_logs(apply: bool) -> tuple[int, int, int]:
    """Bound '## Activity Log' sections: the last ACTIVITY_LOG_MAX_ENTRIES
    entries, and nothing else.

    Returns (files_touched, entries_pruned, non_entry_lines_removed).

    Two shapes are removed here, and they are counted apart in the marker because
    they mean different things. Entries past the cap are the log doing what a
    rolling window should; a non-bullet line under the heading is a note that
    wrote its own continuation lines into the file, and it is the reason the cap
    never worked. `sweep_activity_logs` used to build its entry list from
    `lstrip().startswith("- ")`, so a bare line was in neither set: not an entry
    the cap could count, not a line any branch removed. It therefore survived
    every sweep forever, and the file grew past its declared bound while the
    marker reported a healthy prune — `24-data-pipeline.md` stood at 359 such
    lines for a year while its own marker counted 6,841 entries pruned, and every
    one of those sweeps ran on that exact file. So: non-blank lines that are
    neither entries nor the prune marker go, and a file shedding 359 junk lines
    reports 359 lines, not 359 runs.
    """
    files = pruned = dropped = 0
    if not AUTONOMY_TASKS_DIR.exists():
        return 0, 0, 0
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
        # The log ends at the next heading, so the section bound is what decides
        # which lines are junk. Swept to end-of-file instead, a task file that
        # ever grows a section below its log would have that section's prose
        # deleted as though it were leaked note text.
        end = next((i for i in range(start, len(lines))
                    if lines[i].lstrip().startswith("#")), len(lines))
        entries = [i for i in range(start, end)
                   if lines[i].startswith(ENTRY_BULLET_PREFIX)
                   and not lines[i].startswith(PRUNE_MARKER_PREFIX)]
        junk = [i for i in range(start, end)
                if lines[i].strip()
                and not lines[i].startswith(ENTRY_BULLET_PREFIX)]
        excess = entries[:-ACTIVITY_LOG_MAX_ENTRIES] \
            if len(entries) > ACTIVITY_LOG_MAX_ENTRIES else []
        # fold any prior marker lines into the new one
        prior = prior_lines = n_markers = 0
        for i in range(start, end):
            if lines[i].startswith(PRUNE_MARKER_PREFIX):
                m = re.search(r"(\d+) older entries", lines[i])
                prior += int(m.group(1)) if m else 0
                m = re.search(r"non-entry lines removed: (\d+)", lines[i])
                prior_lines += int(m.group(1)) if m else 0
                n_markers += 1
        if not excess and not junk:
            continue
        drop = set(excess) | set(junk) | {i for i in range(start, end)
                                         if lines[i].startswith(PRUNE_MARKER_PREFIX)}
        kept = [ln for i, ln in enumerate(lines) if i not in drop]
        marker = (f"{PRUNE_MARKER_PREFIX} {prior + len(excess)} older entries "
                  f"pruned by retention sweep "
                  f"(keeping last {ACTIVITY_LOG_MAX_ENTRIES})")
        if prior_lines + len(junk):
            # Counted apart, because "358 older entries pruned" about a file that
            # ran 200 times and shed 358 junk lines is a number pointing at a
            # failure loop that never happened. Suffixed form, not `N lines`, so
            # the line reads the same at 1 as at 359.
            marker += f"; non-entry lines removed: {prior_lines + len(junk)}"
        kept.insert(start, marker)
        if apply:
            path.write_text("\n".join(kept), encoding="utf-8")
        files += 1
        # What *this* run removed. The marker above carries the folded history, so
        # returning those counts too would report a year of accumulated prunes as
        # today's work — the old rung reported this-run-only, and the printed
        # number is what an operator compares against last week's.
        pruned += len(excess)
        dropped += len(junk)
    return files, pruned, dropped


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


# --------------------------------------------------------------------------
# workers.db — the queue's own run history
# --------------------------------------------------------------------------

def _resolve_workers_db() -> Path:
    """The queue's database, resolved by the queue's own rule.

    Imported from `workers.queue.configured_db_path` rather than restated: that
    function folds in `config.yaml`'s `workers.db_path` and the data-root move, and
    a second rule written here would prune whichever file the literal points at
    while the queue kept writing somewhere else — a sweep that reports "3 deleted"
    about a database nobody reads is worse than one that never runs.

    Falls back to `DATA_ROOT / "workers.db"` when the import is unavailable. That is
    the common case this script is written for — cron and autonomy task #79 invoke it
    with a bare `python3`, and `workers.queue` pulls in `app.config`, which needs the
    project venv. The fallback is `configured_db_path`'s own default (`app.paths`'
    `DATA_ROOT / "workers.db"`) reached by the resolver this module already trusts, so
    the two agree in a venv and disagree only into a report that says which file it
    counted.
    """
    try:
        from workers.queue import configured_db_path

        return Path(configured_db_path())
    except Exception:  # noqa: BLE001 - no-venv invocation is the expected path
        return DATA_ROOT / "workers.db"


#: The ONE path this module opens the queue database at. The test fixture redirects
#: this constant into tmp_path, and that fixture's guard treats a `_DB`-suffixed
#: Path as destructive exactly like a `_DIR`, so a new caller that forgets to patch
#: it fails a test instead of pruning the live queue.
WORKERS_DB = _resolve_workers_db()


def _db_skip_note(exc: BaseException) -> str:
    """The report form `skills/retention-sweep/SKILL.md` documents for a lock.

    A skip is reported with its reason and never as `0 deleted`, because `0` and
    "the prune did not run" arrive at the same number and mean opposite things: one
    says the window was empty, the other says the store is still unbounded.
    """
    text = str(exc)
    if "locked" in text or "busy" in text:
        return f"SKIPPED (database locked: {text})"
    return f"SKIPPED (database error: {text})"


def sweep_worker_runs(apply: bool, now: float, *, db: Path | None = None,
                      busy_timeout_ms: int = DB_BUSY_TIMEOUT_MS) -> tuple[int, int, str]:
    """DELETE `runs` rows whose `completed_at` is older than WORKER_RUN_MAX_AGE_DAYS.

    Returns `(rows, bytes, skip_note)`; `skip_note` is "" when the store was reached.
    Bytes is a size delta and is 0 in dry run — the row count is what the horizon
    bounds, and a DELETE that reuses its pages for the next INSERT frees no bytes on
    disk while deleting exactly as many rows as it was asked to.

    The count and the delete share one write transaction: `BEGIN IMMEDIATE` takes the
    lock before either, so the number reported is the number removed and a row the
    queue inserts in between is in neither. Taking that lock is also what makes a busy
    queue observable — with a deferred begin the read succeeds and only the DELETE
    notices, which is a report of `0` for a store that was never swept.
    """
    path = db if db is not None else WORKERS_DB
    if not path.is_file():
        # Absent db is the empty case, not a repair: a round, a sandbox or a fresh
        # install has no queue yet, and "nothing to prune" is the true answer.
        return 0, 0, ""

    try:
        size_before = path.stat().st_size
        conn = sqlite3.connect(str(path), timeout=busy_timeout_ms / 1000.0)
    except sqlite3.Error as exc:
        return 0, 0, _db_skip_note(exc)

    try:
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()
        if table is None:
            # A database with no runs table is the same case as no database at all —
            # a fresh or relocated store, not something to repair. The sweep's job is
            # to bound rows, and creating or migrating schema here would put a weekly
            # cleanup tool in the path of the queue's own migration.
            return 0, 0, ""
        cutoff = (datetime.fromtimestamp(now, tz=timezone.utc)
                  - timedelta(days=WORKER_RUN_MAX_AGE_DAYS)).isoformat()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            return 0, 0, _db_skip_note(exc)
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE completed_at < ?", (cutoff,)
            ).fetchone()[0]
            if apply and rows:
                conn.execute("DELETE FROM runs WHERE completed_at < ?", (cutoff,))
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            return 0, 0, _db_skip_note(exc)
        freed = max(0, size_before - path.stat().st_size) if apply else 0
        return rows, freed, ""
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete/gzip (default: dry-run report)")
    args = ap.parse_args()
    now = time.time()
    mode = "APPLY" if args.apply else "DRY RUN"

    # The header goes out before anything is counted or deleted, and in both
    # modes. An operator approves `--apply` from the dry run's eight numbers, and
    # all eight describe whichever root this run resolved — a value that depends
    # on the caller's tree, its environment and its `$HOME`, and appears nowhere
    # in the report today (#1415). Naming the root above the numbers is what makes
    # them approvable, and is the only line that says which tree they describe.
    print(f"[retention-sweep] {mode}")
    print(f"  data root: {DATA_ROOT}")

    logs_n, logs_b = sweep_task_logs(args.apply, now)
    sess_n, sess_b = sweep_sessions(args.apply, now)
    runs_n, runs_b = sweep_autonomy_runs(args.apply, now)
    act_f, act_e, act_l = sweep_activity_logs(args.apply)
    cand_n, cand_b = sweep_candidates(args.apply, now)
    scr_n, scr_b = sweep_transcript_scratch(args.apply, now)
    spill_n, spill_b = sweep_session_spills(args.apply, now)
    wr_n, wr_b, wr_skip = sweep_worker_runs(args.apply, now)

    print(f"  task logs >{TASK_LOG_MAX_AGE_DAYS}d:  "
          f"{logs_n} deleted, {logs_b / 1024:.0f} KiB freed")
    print(f"  sessions >{SESSION_ARCHIVE_AGE_DAYS}d inactive "
          f"(background >{BACKGROUND_SESSION_ARCHIVE_AGE_DAYS}d): "
          f"{sess_n} gzipped, {sess_b / 1024 / 1024:.1f} MiB "
          f"{'saved' if args.apply else 'candidate'}")
    print(f"  autonomy runs >{RUN_RECORD_MAX_AGE_DAYS}d: "
          f"{runs_n} deleted, {runs_b / 1024 / 1024:.1f} MiB freed")
    # The two numbers are the two shapes the sweep removes, and one line carrying
    # their sum is how a prune of 359 leaked note lines came to be reported as 359
    # runs pruned. Same line in both modes, like every other rung here.
    print(f"  activity logs: {act_f} task files truncated, "
          f"{act_e} entries pruned, {act_l} non-entry lines removed "
          f"(keep last {ACTIVITY_LOG_MAX_ENTRIES})")
    print(f"  skill candidates >{CANDIDATE_MAX_AGE_DAYS}d processed: "
          f"{cand_n} deleted, {cand_b / 1024:.0f} KiB freed")
    print(f"  transcript scratch >{TRANSCRIPT_MAX_AGE_DAYS}d: "
          f"{scr_n} deleted, {scr_b / 1024:.0f} KiB freed")
    # Same line in both modes, like the other six: the operator approves `--apply` from
    # the dry run's numbers, so a dry run that differs in shape from an applied run is a
    # line nobody can compare against the run they just approved.
    print(f"  session spill dirs (*{SPILL_DIR_SUFFIX}) >{SPILL_MAX_AGE_DAYS}d: "
          f"{spill_n} deleted, {spill_b / 1024:.0f} KiB freed")
    # Same line in both modes for the same reason as the spill line above. A skip is put
    # in place of the counts, never beside a `0`, so the number `0` on this store keeps
    # its only honest meaning — the window held nothing.
    if wr_skip:
        print(f"  workers.db runs >{WORKER_RUN_MAX_AGE_DAYS}d: {wr_skip} — nothing pruned")
    else:
        print(f"  workers.db runs >{WORKER_RUN_MAX_AGE_DAYS}d: "
              f"{wr_n} deleted, {wr_b / 1024:.0f} KiB freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
