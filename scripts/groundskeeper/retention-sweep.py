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
   writing. A database the pool will not release inside the bounded busy wait is reported
   `SKIPPED (database locked …)` and exits 0, never a traceback: a weekly sweep that
   fails because the queue was busy reads as "the store is fine" to everything downstream.

9. ~/lloyd-data/workers.db table `queue` — the queue's items themselves (#1466, the
   horizon #1018 left open). DELETE rows in a TERMINAL state (`completed`, `poisoned`,
   `quarantined`) whose `completed_at` (or `enqueued_at`, for a terminal row that
   carries none) is older than QUEUE_MAX_AGE_DAYS — 30, the same horizon as `runs`,
   so an item and the run rows that name it (`runs.queue_id`) leave together. A live
   state (`queued`, `claimed`, `running`) is never touched, whatever its age: those are
   work in flight or waiting, and the pool and the maintenance sweep own them. Same
   database, same busy wait, same `SKIPPED` form, no VACUUM.

10. ~/lloyd-data/_pipeline/groundskeeper-queue.json + groundskeeper-writes.jsonl — the
    groundskeeper survey's queue and its write ledger. Both were written only through
    `scripts/groundskeeper/queue_io.py`, which `201901b4` (#1012) deleted together with
    the survey and the weekly summary, and task #36 is archived — so nothing in the tree
    opens either file to write it and nothing reads them either. Measured on 2026-09-27,
    when this store was added: the queue was 3,128,444 bytes of 8,000 items whose
    `generated_at` is 2026-09-23T02:41:48, the last record in
    the ledger names a writer that no longer exists, and `scripts/backup/backup-graph.sh`
    copies `_pipeline` wholesale, so the pair rides into every backup. DELETE each file
    past GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS — 30, the same mtime window the other file
    stores use, so a survey revived later cannot have a queue it just built pruned under
    it. Aged by mtime, like task logs and transcript scratch: mtime is the only age a
    file with no writer left can have, and the window is uniform so the pair leaves on the
    same clock as the stores around it.

11. ~/lloyd-work/<round_id>/ — the self-mod loop's round worktree homes, the store
    outside the data root (#1644, from #1037's ruling). `worktree.remove()` rmtree's a
    round's directory at close, but a round the gate refused or a human aborted keeps it,
    and `worktree.prune_orphans` runs `git worktree prune` plus a listing — it never
    removes a directory. Reclaim a round's directory WORKTREE_DIR_MAX_AGE_DAYS (7) after
    that round's NEWEST event in `promotions.jsonl`, and never while `git worktree list`
    names a path under it or while `current.json` names the round: an in-flight round is
    never a candidate whatever its age. A directory with no ledger row at all is never
    deleted either — its age is unknowable, and a test scratch dir (`SM_TEST`, `SM_T`) is
    indistinguishable from a round whose row was lost — and its count is reported as its
    own named number, never folded into the reclaimed count or hidden behind a `0`.

12. refs/heads/automod/<round_id> — the round branches, 226 of them at triage against
    82 at filing. DELETE a branch BRANCH_MAX_AGE_DAYS (30) after the round settled, and
    only when `git merge-base --is-ancestor <tip> main` holds: that is the state that
    says the work is in the tree, and it is what makes an irreversible delete
    forensic-safe. A tip that is not reachable from main is a refused or aborted round
    whose branch is the ONLY record of what it attempted, so it is held and NEVER
    deleted: that is settled, not an open question, and no age makes a sole record
    deletable. BRANCH_UNREACHABLE_HOLD_DAYS (90) is a reporting threshold, not a
    deletion horizon — past it the tip is counted and printed, nothing more. The delete
    question reopens only on a measurement: more than 1000 unreachable tips, or a
    `.git` above 500 MB that an unreachable-branch pin is holding open. The held count
    on the report line is the series to watch for the first of those.

13. ~/.local/state/lloyd-automod/promotions.jsonl — the promotion ledger: the loop's own
    append-only audit trail, 33,113,707 bytes / 28,678 rows in the copy the vault committed
    at vault commit `4bc93177` (`git show 4bc93177:backlog/data/promotions.jsonl`), and
    growing 1,992,001 bytes a day on the 7-day mean of that copy's own rows — both figures
    re-derived from those committed bytes by `tests/test_retention_sweep.py` rather than
    typed. The commit and not a file, because #2050 owed 5 retired the rolling working-tree
    copy at `backlog/data/promotions.jsonl`: refreshing it re-commits the whole 33,113,707-byte
    ledger into the vault's history every time, on a repository with no LFS and no remote to
    put it on, and this store's own window rewrites the ledger that copy was cut from — so the
    copy could not stay true even while it was being paid for. #2054 re-aimed every reader at
    the commit, which is the dated extract: history is the thing that cannot be rewritten
    under a reader's feet. `board_health` and the two rungs above
    read it whole on every pass, and nothing had taken a row out of it until #1975 gave it a
    window and #2043 set its value. ARCHIVE OUT — never fold in place, never delete: a row
    older than LEDGER_ARCHIVE_AGE_DAYS (14 days: 14 x 1,992,001 = 27,888,014 bytes of live
    file, 41.6% of the `state.py` decode-cache ceiling, and inside the ±20% band the cold
    cycle's ledger budget is calibrated on) leaves the live file for
    `promotions-archive-<YYYYMM>.jsonl.gz` beside it, bucketed by the month of the row's OWN
    age and copied byte-for-byte, so the archive holds the rows and not a summary of them.
    Two rules are what keep the two rungs above intact while the file shrinks: each round's
    NEWEST row is never archived, because `_settle_times` takes a max and a round with no
    live row is a round both rungs count `no ledger row` and therefore never delete —
    bounding this store would otherwise unbound stores 11 and 12. #2043's 14 is what makes
    that first rule load-bearing rather than a courtesy: a round's rows go archivable a
    fortnight before the 30-day branch horizon asks this ledger when the round settled, so
    the keep, and not the window, is what keeps stores 11 and 12 bounded. And the rewrite
    happens only on a PROOF, `board_health()` run over the live file
    and over the new file as a copy beside it, refusing to write if any key of the two
    payloads differs. Rewrite is temp-file + `os.replace` with every line appended since
    the read grafted back verbatim and `st_size` re-checked before the rename, because
    `state.append_event` appends and fsyncs under no lock and an unconditional rename
    would rename a live audit row out of existence.

Stores 11, 12 and 13 are the three the loop leaves behind, and 13 is the one the other two
read. None of the three is under `DATA_ROOT`, and store 12 is not even on the filesystem:
it is the live repo's refs. That is the hazard the production-checkout guard exists for —
a round's worktree shares those refs, so `git branch -D` run from inside a gate would
delete production branches. `automod_rung_refusal()` therefore refuses all three loop
rungs — the ledger included, which rewrites no ref but rewrites the loop's own audit file,
and a tree that is not the live checkout has no business deciding to compress it — unless
the tree this script was loaded from IS the production checkout (`app.data_root`'s own
two predicates for that, not a restatement), and `--apply` exits
NOT_PRODUCTION_EXIT (2) naming the tree it resolved, in the manner of the
`.lloyd-data-root` refusal above. A dry run still exits 0 from any tree, because it
deletes nothing and the root it reports is the point of running it (#1415).

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
resolved, in dry run and in `--apply` alike, above the numbers it describes. Two rungs
write outside the data root at a path a knob points at a copy: bounding the activity logs
in the vault's `autonomy/*.md`, which `LLOYD_VAULT_ROOT` points at a copy, and archiving
the promotion ledger, which `LLOYD_AUTOMOD_STATE` points at a copy — the same shape as
`scripts/automod/state.py`'s own resolution, so a round's or a sandbox's `--apply` can
only ever reach its own ledger. The bound is two rules,
not one — the last `ACTIVITY_LOG_MAX_ENTRIES` entry bullets are kept and every other
non-blank line under the heading is removed, because a line the entry cap does not
count is a line the cap could never remove (#845).

Deliberately NOT a store: ~/lloyd-data/_pipeline/trajectories — the mined trajectory
corpus, `scripts/memory/conversation_relations.py`'s TRAJECTORY_DIR. No rung here opens
it, and that exclusion is a ruling rather than an omission (#1674, recorded on the board
2026-09-28 and nowhere in the tree until this paragraph). It is the corpus the live
conversation-relations floors are measured against, and `tests/test_conversation_relations.py`
requires at least 5 day-files, at least 200 raw co-access pairs and at least 100 aggregate
pairs out of these very files. So an mtime window here would bound a store that a green
suite cannot see: `test_the_bare_invocation_deletes_the_pair_it_resolves` plants its
untouched-`_pipeline` decoy as a SIBLING of this directory, and nothing planted inside it,
so a store naming the corpus would sweep it and report a line every test still liked.
And unlike every store above, it cannot be rebuilt after the fact. Mining reads session
transcripts, and store 2 gzips those out of the `*.json` globs the extractors use — the
2026-09-22 deletion took `_pipeline/trajectories` and the 09-11..09-22 mining window with
it, which is why that test now calls an absent corpus "a fact about the machine".
Nor is there pressure to bound it: measured 2026-09-28 the corpus is 7 day-files, 31 MB,
about 4.4 MB a day, sitting at 2,374 raw and 225 aggregate pairs against the same floors.
When it does need bounding, the discipline is pair-preserving compaction inside the corpus
— merging day-files so the co-access pairs survive — and never an mtime archive: the corpus
is daily-grained, so any window short of its span breaks the 5-file volume gate in days,
long before the pair floors bind. `tests/test_retention_sweep.py` is what keeps this
paragraph from rotting into a comment: it runs the shipped script over a corpus planted
inside the data root, and it refuses any store path this module resolves that lands on or
under the directory.

Usage:
    retention-sweep.py            # dry run — report only
    retention-sweep.py --apply    # actually delete/gzip
"""

import argparse
import gzip
import json
import os
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
try:
    # The same module as an OBJECT, for the two predicates the automod rungs'
    # production guard reads. Through the module rather than by name, so a caller
    # that redirects `app.data_root.live_checkout` — as
    # `tests/test_retention_sweep.py::_load` already does for the data-root rules —
    # redirects this guard too. A name imported here would have been a frozen copy
    # of the pre-patch function, and the guard would have been untestable.
    from app import data_root as _DATA_ROOT_MODULE
except ModuleNotFoundError:  # pragma: no cover - same fallback as above
    import data_root as _DATA_ROOT_MODULE

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
# The retired survey's queue pair (#1574). Named as two files, not a directory: the
# survey wrote only these two under `_pipeline/`, and a glob over `_pipeline` would put
# this sweep's `unlink` next to the task logs, the skill candidates and the transcript
# scratch that the rungs above own with their own rules. Both are module-level constants
# because `tests/test_retention_sweep.py`'s fixture redirects every path constant this
# module holds — a tuple of the two built here would freeze the LIVE paths past that
# redirection, which is exactly the fixture's guard exists to prevent.
GROUNDSKEEPER_QUEUE_FILE = DATA_ROOT / "_pipeline" / "groundskeeper-queue.json"
GROUNDSKEEPER_WRITES_FILE = DATA_ROOT / "_pipeline" / "groundskeeper-writes.jsonl"
#: 30, the same horizon as the other mtime-bounded file stores here (task logs,
#: transcript scratch, skill candidates, spill dirs). Uniform rather than chosen for this
#: pair: the queue was last written 2026-09-23 and any window under 30 d would reclaim it
#: sooner for no reason a reader can check, and a window over 30 d would let a survey
#: revived later have a queue it just built pruned while its own writer is still running.
#: `tests/test_retention_sweep.py` asserts this value rather than trusting the literal.
GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS = 30

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
# The queue's items (`workers.db` table `queue`), terminal states only (#1466). Its own
# constant so the two horizons can move apart, but 30 today: a run row outliving the
# item it names, or the reverse, is a join that answers differently depending on which
# table the reader started from.
QUEUE_MAX_AGE_DAYS = 30
#: States a queue row can be pruned in. Everything else — `queued`, `claimed`,
#: `running`, and any state a future writer invents — is kept, because an unknown
#: state is not evidence the work is over.
QUEUE_TERMINAL_STATES = ("completed", "poisoned", "quarantined")
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


def sweep_groundskeeper_queue(apply: bool, now: float) -> tuple[int, int]:
    """Delete the retired survey's queue pair past GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS.

    Returns (count, bytes), counting files — so the line reads `2 deleted` for the pair
    and `0 deleted` for a tree that has neither file, which is the same answer a fresh
    install gives and the only honest one: `0` here means the window held nothing, not
    that the store was unreachable. There is no skip form because there is nothing to
    skip on — two `unlink`s, no lock and no schema.

    The files are looked up as module globals on every call rather than iterated from a
    tuple, so the test fixture's redirection of the two constants is what actually gets
    deleted. A symlink is passed over rather than unlinked: this store retires two
    specific files that nothing writes any more, so a path that has become a link is not
    one of them, and removing the link would report the store's real target gone while
    leaving it in place.
    """
    count = freed = 0
    cutoff = now - GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS * 86400
    for path in (GROUNDSKEEPER_QUEUE_FILE, GROUNDSKEEPER_WRITES_FILE):
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


# ---------------------------------------------------------------------------
# automod round residue — the two stores the loop leaves outside the data root
# ---------------------------------------------------------------------------
#
# Every rung above bounds something under `DATA_ROOT`. The self-modification loop
# leaves two things that are not under it and that nothing in this repository has
# ever deleted (#1644, carrying #1037's ruling):
#
#   ~/lloyd-work/<round_id>/   The round's worktree home. `worktree.remove()`
#                              rmtree's it at round close, but a round the gate
#                              refused or a human aborted keeps it, and
#                              `worktree.prune_orphans` is `git worktree prune` plus a
#                              listing — it never removes a directory. Measured at
#                              triage (2026-09-27): 525 MB across 12 dirs, the oldest
#                              (SM_20260910_104045, 272 MB) 17 days past its last
#                              ledger event and unregistered in `git worktree list`.
#   refs/heads/automod/<rid>   The round's branch. `abort()` keeps it deliberately —
#                              "it is the only forensic record of what was attempted" —
#                              and outside a resume nothing has ever deleted one.
#                              Measured: 226 branches at triage, 82 at filing.
#
# `~/.local/state/lloyd-automod/rounds/` (825 dirs / 24 MB) is deliberately NOT a
# store here: #1037's ruling keeps the state dirs indefinitely, so `AUTOMOD_STATE_ROUNDS`
# exists only so a test can point at the place and prove an `--apply` left it standing.

#: Reclaim a settled round's `~/lloyd-work/<round_id>/` directory this many days after
#: that round's newest ledger event (#1037's ruling, as recorded on #1644).
WORKTREE_DIR_MAX_AGE_DAYS = 7
#: Delete an `automod/<round_id>` branch this many days after the round settled — but
#: only once its tip is an ancestor of `main`, which is the state that says the work is
#: in the tree and is what makes an irreversible delete forensic-safe.
BRANCH_MAX_AGE_DAYS = 30
#: An unreachable tip is HELD and NEVER deleted — settled, not an open question. The
#: branch of a refused or aborted round is the only thing left saying what it attempted,
#: and no age changes that. Measured reason at triage: of the 226 branch tips, 37 are
#: ancestors of `main` and 0 are reachable from `refs/automod/rounds/*` — `squash_onto`
#: keeps the PRE-squash HEAD at the keep-ref while the branch moves to the squashed
#: commit — so the unreachable set is 189 refused and aborted rounds, every one of them
#: a sole record.
#:
#: This age is therefore a REPORTING THRESHOLD, not a deletion horizon: past it the tip
#: is counted (`due_ruling`) and printed, and nothing else happens to it. The delete
#: question reopens only on a measurement — more than 1000 unreachable tips, or a `.git`
#: directory above 500 MB that an unreachable-branch pin is holding open — and the held
#: count on the weekly report line is the series to watch for the first of those. Both
#: bounds are below, and the printed line prints them, so a reader of the report never
#: has to open this file to learn what would change the answer.
BRANCH_UNREACHABLE_HOLD_DAYS = 90
#: Unreachable tips above which the never-delete decision is reopened for measurement,
#: not for an opinion. 189 at triage; 1000 is a headroom of ~5x, chosen because the cost
#: of holding is forensic and the cost of deleting a sole record is unrecoverable.
BRANCH_UNREACHABLE_REOPEN_TIPS = 1000
#: A `.git` above this many MB, with unreachable automod tips in it, is the other
#: measurement that reopens the question: it says the hold is costing the repo itself,
#: which is the only price the 90 d figure was ever standing in for.
BRANCH_UNREACHABLE_REOPEN_GIT_MB = 500
#: The ref a branch tip must be an ancestor of to count as landed.
MAIN_REF = "main"
#: Round ids are `SM_<YYYYMMDD>_<HHMMSS>`; the prefix also covers the `SM_TEST`,
#: `SM_T`, `SM_TRIAL`, `SM_SEAM`, `SM_PINS` scratch dirs the loop's own tests leave
#: behind, which is deliberate — they fall into the no-ledger-row bucket below.
ROUND_ID_PREFIX = "SM_"
#: The loop's own branch namespace, spelled as `worktree.py:128` and `:383` spell it.
BRANCH_NAMESPACE = "automod/"
#: Exit status for an `--apply` that refused, the same status the `.lloyd-data-root`
#: refusal at the top of this file exits with.
NOT_PRODUCTION_EXIT = 2


class AutomodStoreUnavailable(RuntimeError):
    """An automod store could not be READ, so its count would mean nothing.

    The same reason `worktree.list_registered` raises instead of returning `[]`:
    a `git` call that failed and a call that came back empty arrive at this rung
    as the same zero, and "nothing is due" is the opposite of "nothing could be
    checked". A rung that cannot read its store says so and deletes nothing.
    """


def _automod_module(name: str):
    """Import one `scripts.automod` helper, or None if this tree cannot give it.

    Imported rather than restated for three rules this file would otherwise
    duplicate: where `~/lloyd-work` is, how a branch is named for a round, and how
    the registered worktrees are read (that reader raises on a failed `git` call,
    which is the behaviour this rung needs and would not get from a local copy).
    Cron and autonomy task #79 run this file with a bare `python3`, and
    `scripts.automod.worktree` / `.state` are stdlib-only, so the import is
    available — but it is not ASSUMED: a tree that cannot give it gets the skip
    note below, never a guessed path.
    """
    try:
        return __import__(f"scripts.automod.{name}", fromlist=["*"])
    except Exception:  # noqa: BLE001 - absence is a state, not a bug
        return None


def _automod_paths() -> dict[str, Path]:
    """Where the loop keeps its state, resolved by the loop's own module.

    `scripts.automod.state` reads `LLOYD_AUTOMOD_STATE` at import, so pointing
    that variable is the sanctioned way to run this sweep against a copy — the
    same shape as `LLOYD_VAULT_ROOT` for the vault-touching rung. Falls back to
    `state.py`'s own formula (`~/.local/state/lloyd-automod`) when the module is
    unavailable, and says which it used by printing the resolved paths.
    """
    state = _automod_module("state")
    if state is not None:
        return {"ledger": Path(state.LEDGER_PATH), "current": Path(state.CURRENT_PATH),
                "rounds": Path(state.ROUNDS_DIR)}
    root = Path(os.environ.get("LLOYD_AUTOMOD_STATE",
                               Path.home() / ".local" / "state" / "lloyd-automod"))
    return {"ledger": root / "promotions.jsonl", "current": root / "current.json",
            "rounds": root / "rounds"}


def _automod_work_root() -> Path:
    """`~/lloyd-work`, by `scripts.automod.worktree.WORK_ROOT` — not a copy of it."""
    worktree = _automod_module("worktree")
    if worktree is not None:
        return Path(worktree.WORK_ROOT)
    return Path.home() / "lloyd-work"


_paths = _automod_paths()

#: The round worktree homes. Named `_ROOT`, not `_DIR`, for the honest reason: it is
#: not under `DATA_ROOT`, so it must not be swept into the set
#: `test_the_one_resolution_owns_every_directory_the_sweep_touches` holds to that root.
#: `tests/test_retention_sweep.py`'s fixture redirects it anyway — its guard is
#: name-blind and every `Path` in here is a delete target by default.
AUTOMOD_WORK_ROOT = _automod_work_root()
#: The loop's append-only audit trail — the only thing that says when a round settled.
AUTOMOD_LEDGER = _paths["ledger"]
#: Which round is landing RIGHT NOW. Deleted at settle, so a missing file means no
#: round is in flight, not that the check was skipped.
AUTOMOD_CURRENT = _paths["current"]
#: The state dirs the ruling keeps indefinitely. Not a store: no rung writes here, and
#: a test holds this path to prove `--apply` walked past it.
AUTOMOD_STATE_ROUNDS = _paths["rounds"]
#: The checkout whose refs the branch arm may delete from. Separate from the tree this
#: script was loaded from on purpose: `_TREE` is a sys.path detail and the
#: `tests/test_retention_sweep.py` fixture guard enumerates module `Path` constants, so
#: the one place a `git branch -D` can be aimed has to be enumerable by that guard.
#: Defaulting it to the tree the script lives in is the production behaviour — the
#: groundskeeper runs from the live checkout, which is exactly the case the guard below
#: admits and no other.
AUTOMOD_REPO = Path(_TREE)


def automod_rung_refusal(tree: Path | None = None) -> str | None:
    """Why the automod rungs must not run from this tree, or None if they may.

    The hazard is specific: a round's worktree shares the LIVE repository's refs, so a
    `git branch -D` issued from inside a gate removes production branches — and unlike
    every file store above, there is no data root between this rung and the thing it
    destroys. So the rung runs only where the code tree IS the production checkout,
    decided by the same two predicates `app.data_root.resolve_data_root` uses for that
    word (the tree equals `<passwd home>/lloyd`, and it is not a linked worktree),
    called through the module so `LLOYD_DATA`-style test redirection reaches them.

    Returned as a sentence naming both trees because the refusal is the report line: an
    operator who ran `--apply` from a sandbox has to be able to see, from the line
    alone, which root it declined.
    """
    here = _TREE if tree is None else Path(tree)
    production = _DATA_ROOT_MODULE.live_checkout()
    if _DATA_ROOT_MODULE.tree_is_worktree(here):
        return (f"REFUSED: this sweep is running from {here}, a linked git worktree, "
                f"not the production checkout {production} — its refs are the live "
                f"repo's, so no branch or round directory was touched")
    if here.resolve() != Path(production).resolve():
        return (f"REFUSED: this sweep is running from {here}, not the production "
                f"checkout {production} — no branch or round directory was touched")
    return None


def _settle_times(ledger: Path) -> dict[str, float]:
    """`round_id` -> seconds-since-epoch of that round's NEWEST ledger event.

    Newest, not first: a round that was gated twice or resumed has several events, and
    the age that matters for "is this residue still wanted" is when the loop last
    touched it. A row with no numeric `ts` falls back to its `created_at` ISO stamp; a
    row with neither, or an undecodable line, is skipped rather than guessed at — the
    ledger is append-only under a lock and this rung must not be the reason a
    malformed line becomes an age.
    """
    out: dict[str, float] = {}
    try:
        text = ledger.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        rid = row.get("round_id")
        if not isinstance(rid, str) or not rid:
            continue
        seconds = row.get("ts")
        if not isinstance(seconds, (int, float)):
            seconds = _iso_seconds(row.get("created_at"))
        if seconds is None:
            continue
        if seconds > out.get(rid, float("-inf")):
            out[rid] = float(seconds)
    return out


def _iso_seconds(value) -> float | None:
    """Seconds for the ledger's `created_at` form, or None if it is not one."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _current_round(current: Path) -> str | None:
    """The round `current.json` names, or None when no round is in flight."""
    try:
        doc = json.loads(current.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rid = doc.get("round_id") if isinstance(doc, dict) else None
    return rid if isinstance(rid, str) and rid else None


def _bytes_under(dir: Path) -> int:
    """Total file size under `dir`, following no symlink and never raising.

    Only ever called on directories this rung has already decided to remove, so the
    walk is bounded by what a reclaim was going to delete anyway.
    """
    total = 0
    for root, _dirs, files in os.walk(dir):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _names_registered(dir: Path, registered: list[str]) -> bool:
    """Whether `git worktree list` named any path at or under `dir`.

    Both sides go through `realpath` because the round layout is a symlink farm by
    design (`worktree.ensure_round_home` links into the live tree), and a registered
    path spelled through a link would otherwise read as "not registered" while the
    worktree is live — the one mistake this guard exists to make impossible.
    """
    real = Path(os.path.realpath(dir))
    for path in registered:
        candidate = Path(os.path.realpath(path))
        if candidate == real or real in candidate.parents:
            return True
    return False


def _registered_worktrees(repo: Path) -> list[str]:
    """The loop's own reader, which raises rather than returning `[]` on a failed read."""
    worktree = _automod_module("worktree")
    if worktree is None:
        raise AutomodStoreUnavailable(
            "scripts.automod.worktree is not importable from this tree, so the "
            "registered worktrees cannot be read")
    try:
        return list(worktree.list_registered(repo))
    except Exception as exc:  # WorktreeListUnavailable and any git failure
        raise AutomodStoreUnavailable(str(exc)) from exc


def sweep_automod_worktrees(apply: bool, now: float, *, work_root: Path | None = None,
                            ledger: Path | None = None, current: Path | None = None,
                            repo: Path | None = None,
                            registered: list[str] | None = None) -> dict:
    """Reclaim `~/lloyd-work/<round_id>/` past WORKTREE_DIR_MAX_AGE_DAYS.

    Returns the counts the report line prints, with `skip` non-empty when the store
    could not be read at all. The keys are each a NAMED outcome, because the failure
    this rung has to avoid is a `0 reclaimed` that means four different things:

      reclaimed   directories removed (under `--apply`) or due for removal (dry run)
      bytes       their total file size
      young       settled inside the horizon — kept, still recent forensics
      untracked   NO ledger row, so no settle time — never deleted (clause 3)
      registered  `git worktree list` names a path under it — never deleted, any age
      live        the round `current.json` names — never deleted, any age
      not_a_round an entry whose name is not a round id, or a symlink — never touched
      failed      `rmtree` refused (permissions, a busy mount) — reported, not swallowed
    """
    root = Path(work_root if work_root is not None else AUTOMOD_WORK_ROOT)
    times = _settle_times(Path(ledger if ledger is not None else AUTOMOD_LEDGER))
    live = _current_round(Path(current if current is not None else AUTOMOD_CURRENT))
    out = {"reclaimed": 0, "bytes": 0, "young": 0, "untracked": 0, "registered": 0,
           "live": 0, "not_a_round": 0, "failed": 0, "skip": ""}
    if not root.is_dir():
        return out
    if registered is None:
        try:
            registered = _registered_worktrees(Path(repo if repo is not None else _TREE))
        except AutomodStoreUnavailable as exc:
            out["skip"] = f"SKIPPED (registered worktrees unreadable: {exc})"
            return out
    cutoff = now - WORKTREE_DIR_MAX_AGE_DAYS * 86400
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        out["skip"] = f"SKIPPED ({root} unreadable: {exc})"
        return out
    for dir in entries:
        try:
            if dir.is_symlink() or not dir.is_dir() or not dir.name.startswith(ROUND_ID_PREFIX):
                out["not_a_round"] += 1
                continue
            if live is not None and dir.name == live:
                out["live"] += 1
                continue
            if _names_registered(dir, registered):
                out["registered"] += 1
                continue
            settled = times.get(dir.name)
            if settled is None:
                out["untracked"] += 1
                continue
            if settled > cutoff:
                out["young"] += 1
                continue
            size = _bytes_under(dir)
            if apply:
                try:
                    shutil.rmtree(dir)
                except OSError as exc:
                    print(f"  ! skip {dir.name}: {exc}", file=sys.stderr)
                    out["failed"] += 1
                    continue
            out["reclaimed"] += 1
            out["bytes"] += size
        except OSError as exc:
            print(f"  ! skip {dir.name}: {exc}", file=sys.stderr)
            out["failed"] += 1
    return out


def _branch_tips(repo: Path) -> dict[str, str]:
    """Every `automod/*` branch and its tip, in one `for-each-ref` call.

    Raises when the read fails: an empty dict from a failed call and an empty dict
    from a repo with no automod branches look identical, and the first one must not
    arrive as `0 deleted`.
    """
    worktree = _automod_module("worktree")
    if worktree is None:
        raise AutomodStoreUnavailable(
            "scripts.automod.worktree is not importable from this tree, so its "
            "`git` runner is unavailable")
    # `refs/heads/` is prefixed here and not inside the constant: `BRANCH_NAMESPACE`
    # is how a branch is NAMED (`automod/<round_id>`, the spelling `worktree.py` uses
    # in `git branch -D`), and the test below asserts the enumerated names against
    # that spelling. The full ref path is this call's business.
    r = worktree.git(repo, "for-each-ref", "--format=%(refname:lstrip=2)\t%(objectname)",
                     f"refs/heads/{BRANCH_NAMESPACE.rstrip('/')}/")
    if r.returncode != 0:
        raise AutomodStoreUnavailable(
            f"git -C {repo} for-each-ref failed rc={r.returncode}: "
            f"{(r.stderr or '').strip()[:200]}")
    tips: dict[str, str] = {}
    for line in r.stdout.splitlines():
        name, _, sha = line.partition("\t")
        if name.startswith(BRANCH_NAMESPACE) and sha:
            tips[name] = sha.strip()
    return tips


def _commits_reachable_from(repo: Path, ref: str) -> set[str]:
    """Every commit reachable from `ref`, in one call.

    Membership in this set IS `git merge-base --is-ancestor <tip> main`, and
    `tests/test_retention_sweep.py` asserts it against that very command for every
    seeded branch rather than trusting the equivalence. One `rev-list` is 1,638 lines
    and 6 ms on this repository against one `merge-base` per branch (233 at triage);
    the reason it is worth a pinned test instead of the loop of calls is that the
    command names the predicate, so a change here that broke the identity could not
    pass unnoticed.
    """
    worktree = _automod_module("worktree")
    if worktree is None:
        raise AutomodStoreUnavailable("scripts.automod.worktree is unavailable")
    r = worktree.git(repo, "rev-list", ref)
    if r.returncode != 0:
        raise AutomodStoreUnavailable(
            f"git -C {repo} rev-list {ref} failed rc={r.returncode}: "
            f"{(r.stderr or '').strip()[:200]}")
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


def sweep_automod_branches(apply: bool, now: float, *, ledger: Path | None = None,
                           current: Path | None = None, repo: Path | None = None,
                           work_root: Path | None = None,
                           registered: list[str] | None = None,
                           main_ref: str = MAIN_REF) -> dict:
    """Delete `automod/<round_id>` past BRANCH_MAX_AGE_DAYS when its tip is in main.

    The tip condition is the whole safety story: branch deletion is irreversible, and
    an ancestor of `main` is a round whose content is in the tree, so the branch is
    scaffolding. An unreachable tip is held and counted (`held_unreachable`, of which
    `due_ruling` are past BRANCH_UNREACHABLE_HOLD_DAYS) and NEVER deleted: for a refused
    or aborted round that branch is the only record of what was attempted, which is a
    settled decision rather than this rung's call to re-litigate. `due_ruling` is a
    reporting count over a threshold, and the two measurements that would reopen the
    question — `BRANCH_UNREACHABLE_REOPEN_TIPS` and `BRANCH_UNREACHABLE_REOPEN_GIT_MB` —
    are recorded at their constants and printed on the report line; this rung acts on
    neither of them.

    Same named-outcome shape as the worktree rung, plus `deleted`/`failed` for the
    `git branch -D` itself, whose exit status is checked by `worktree.delete_branch`
    and a refusal is counted, never ignored.
    """
    root = Path(work_root if work_root is not None else AUTOMOD_WORK_ROOT)
    here = Path(repo if repo is not None else _TREE)
    times = _settle_times(Path(ledger if ledger is not None else AUTOMOD_LEDGER))
    live = _current_round(Path(current if current is not None else AUTOMOD_CURRENT))
    out = {"deleted": 0, "young": 0, "held_unreachable": 0, "due_ruling": 0,
           "untracked": 0, "registered": 0, "live": 0, "failed": 0, "skip": ""}
    try:
        tips = _branch_tips(here)
        reachable = _commits_reachable_from(here, main_ref)
        if registered is None:
            registered = _registered_worktrees(here)
    except AutomodStoreUnavailable as exc:
        out["skip"] = f"SKIPPED ({exc})"
        return out
    delete_cutoff = now - BRANCH_MAX_AGE_DAYS * 86400
    hold_cutoff = now - BRANCH_UNREACHABLE_HOLD_DAYS * 86400
    for branch, tip in sorted(tips.items()):
        rid = branch[len(BRANCH_NAMESPACE):]
        if live is not None and rid == live:
            out["live"] += 1
            continue
        if _names_registered(root / rid, registered):
            out["registered"] += 1
            continue
        settled = times.get(rid)
        if settled is None:
            out["untracked"] += 1
            continue
        if settled > delete_cutoff:
            out["young"] += 1
            continue
        if tip not in reachable:
            out["held_unreachable"] += 1
            if settled <= hold_cutoff:
                out["due_ruling"] += 1
            continue
        if apply:
            worktree = _automod_module("worktree")
            if worktree is None or not worktree.delete_branch(here, branch):
                print(f"  ! skip {branch}: git branch -D refused", file=sys.stderr)
                out["failed"] += 1
                continue
        out["deleted"] += 1
    return out


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
    return _prune_db_rows(apply, now, db=db, busy_timeout_ms=busy_timeout_ms,
                          table="runs", where="completed_at < ?", params=(),
                          max_age_days=WORKER_RUN_MAX_AGE_DAYS)


def sweep_queue_rows(apply: bool, now: float, *, db: Path | None = None,
                     busy_timeout_ms: int = DB_BUSY_TIMEOUT_MS) -> tuple[int, int, str]:
    """DELETE terminal `queue` rows older than QUEUE_MAX_AGE_DAYS (#1466).

    Only `QUEUE_TERMINAL_STATES`; a live row is never counted or deleted. Aged by
    `completed_at`, which every terminal writer stamps, falling back to `enqueued_at`
    (NOT NULL) so a terminal row missing its stamp is still bounded rather than kept
    forever. Same contract as `sweep_worker_runs`: `(rows, bytes, skip_note)`.
    """
    marks = ",".join("?" for _ in QUEUE_TERMINAL_STATES)
    return _prune_db_rows(apply, now, db=db, busy_timeout_ms=busy_timeout_ms,
                          table="queue",
                          where=(f"state IN ({marks}) "
                                 f"AND COALESCE(completed_at, enqueued_at) < ?"),
                          params=QUEUE_TERMINAL_STATES,
                          max_age_days=QUEUE_MAX_AGE_DAYS)


def _prune_db_rows(apply: bool, now: float, *, db: Path | None, busy_timeout_ms: int,
                   table: str, where: str, params: tuple,
                   max_age_days: int) -> tuple[int, int, str]:
    """Count, and under `apply` delete, `table` rows matching `where` in one write txn.

    `where` ends in the cutoff placeholder; `params` fill the ones before it.
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
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,)).fetchone()
        if present is None:
            # A database without the table is the same case as no database at all —
            # a fresh or relocated store, not something to repair. The sweep's job is
            # to bound rows, and creating or migrating schema here would put a weekly
            # cleanup tool in the path of the queue's own migration.
            return 0, 0, ""
        cutoff = (datetime.fromtimestamp(now, tz=timezone.utc)
                  - timedelta(days=max_age_days)).isoformat()
        args = (*params, cutoff)
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            return 0, 0, _db_skip_note(exc)
        try:
            rows = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {where}", args).fetchone()[0]
            if apply and rows:
                conn.execute(f"DELETE FROM {table} WHERE {where}", args)
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            return 0, 0, _db_skip_note(exc)
        freed = max(0, size_before - path.stat().st_size) if apply else 0
        return rows, freed, ""
    finally:
        conn.close()


def _worktree_line(w: dict) -> str:
    """One report line for the round-directory store, identical in both modes.

    Every kept count is named, because a bare `0 reclaimed` is the number four
    different states produce: the horizon held nothing, no directory has a ledger row,
    every worktree is registered, or the directory list could not be read. Only the
    first of those is a clean bill.
    """
    if w["skip"]:
        return f"  ~/lloyd-work round dirs >{WORKTREE_DIR_MAX_AGE_DAYS}d: " \
               f"{w['skip']} — nothing reclaimed"
    kept = [f"{w['young']} <{WORKTREE_DIR_MAX_AGE_DAYS}d",
            f"{w['untracked']} no ledger row",
            f"{w['registered']} registered", f"{w['live']} live round"]
    if w["not_a_round"]:
        kept.append(f"{w['not_a_round']} not a round id")
    if w["failed"]:
        kept.append(f"{w['failed']} FAILED to remove")
    return (f"  ~/lloyd-work round dirs >{WORKTREE_DIR_MAX_AGE_DAYS}d: "
            f"{w['reclaimed']} reclaimed, {w['bytes'] / 1024 / 1024:.1f} MiB freed "
            f"(kept: {', '.join(kept)})")


def _branch_line(b: dict) -> str:
    """One report line for the branch store, identical in both modes.

    The unreachable count is the one an operator reads first: at triage it is 189 of
    226 tips, and the decision over them is made — they are never deleted, because a
    refused or aborted round's tip is the sole record of what it attempted. The
    `past 90d` figure beside it counts how many have been held past the reporting
    threshold, which is the series to watch: it reopens the question only above
    BRANCH_UNREACHABLE_REOPEN_TIPS tips or a `.git` above BRANCH_UNREACHABLE_REOPEN_GIT_MB
    MB, and this sweep deletes none of them at either figure. The line says all of that
    so the weekly report cannot advertise a settled decision as an open question — the
    alarm-that-survives-its-retraction shape, where the retraction is never reprinted
    and only the ask is.
    """
    if b["skip"]:
        return (f"  automod/* branches >{BRANCH_MAX_AGE_DAYS}d & ancestor of "
                f"{MAIN_REF}: {b['skip']} — nothing deleted")
    kept = [f"{b['young']} <{BRANCH_MAX_AGE_DAYS}d",
            f"{b['held_unreachable']} unreachable held: never deleted, sole record "
            f"({b['due_ruling']} past {BRANCH_UNREACHABLE_HOLD_DAYS}d)",
            f"{b['untracked']} no ledger row", f"{b['registered']} registered",
            f"{b['live']} live round"]
    if b["failed"]:
        kept.append(f"{b['failed']} FAILED to delete")
    return (f"  automod/* branches >{BRANCH_MAX_AGE_DAYS}d & ancestor of {MAIN_REF}: "
            f"{b['deleted']} deleted (kept: {', '.join(kept)}) "
            f"— the delete question reopens only above {BRANCH_UNREACHABLE_REOPEN_TIPS} "
            f"held tips or a .git above {BRANCH_UNREACHABLE_REOPEN_GIT_MB} MB")


# ── 13. promotions ledger — archive the rows the loop no longer reads ──────────
#
# Alan's ruling on #1975 (2026-10-01) chose archive-out over digest-in-place after
# triage measured that the cold cycle's cost tracks ROW COUNT, not payload bytes: the
# same 28,476 rows padded from 32.8 MB to 61.8 MB decoded in 0.26-0.29 s (+49% bytes,
# +0.01 s) while cutting to 21,050 rows decoded in 0.20 s. So the rows leave. What the
# ruling did not lift, and what the two rungs above depend on, is that no row is lost:
# the gzip beside the ledger holds the original line, byte for byte.

#: Age after which a promotions-ledger row leaves the live file for the monthly
#: archive (#1975, from #1858's owed entry 2; the VALUE is #2043's ruling). 14 is
#: deliberately NOT the horizon every other store in this file uses. At the
#: 1,992,001 B/day the committed copy of this ledger adds over its own last 7 days,
#: 14 days is 14 x 1,992,001 = 27,888,014 bytes of live file, 41.6% of the
#: `state.py` decode-cache ceiling and inside the ±20% band the cold cycle's ledger
#: budget is calibrated on. The 30-day alternative — every other horizon here —
#: steady-states above the TOP of that band, so it would bound this store by spending
#: the budget the store exists to protect, and on a ledger whose whole history is
#: younger than the window it would leave the fold with nothing to move at all. The
#: shortened value is also what makes `newest-of-round` load-bearing rather than a
#: courtesy (see `_archive_plan`): a round's rows go archivable a fortnight before the
#: 30-day branch horizon asks this ledger when that round settled, and a round with no
#: live row is a round the two rungs above count `no ledger row` and therefore never
#: delete.
LEDGER_ARCHIVE_AGE_DAYS = 14

#: Prefix of the monthly gzip that receives the archived rows, beside the ledger. One
#: file per month of the ledger's OWN history, never one per sweep, so a reader who wants
#: September's rows opens September's file whatever week the sweep ran in.
LEDGER_ARCHIVE_PREFIX = "promotions-archive-"

#: How many times the settled rewrite re-reads the tail before it gives up. The ledger is
#: appended to by `state.append_event` with no lock (`LOCK_PATH` guards round
#: transitions, not appends), and `os.replace` is unconditional, so the only thing
#: standing between this rung and a vanished audit row is re-reading the bytes that
#: appeared since the read that built the new file, and refusing to rename while the file
#: will not hold still.
LEDGER_REWRITE_ATTEMPTS = 5


def _ledger_row_seconds(row: dict) -> float | None:
    """The age `sweep_promotions_ledger` window-members on — `_settle_times`'s own rule.

    `ts` when it is numeric, else the `created_at` ISO stamp, else None. Not a second
    clock: the rungs that age round residue read exactly this pair through
    `_settle_times`, so a row that is "31 days old" to this rung is the same 31 days to
    them. Two clocks here is how a row ends up archived on one and still load-bearing on
    the other.
    """
    seconds = row.get("ts")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        seconds = _iso_seconds(row.get("created_at"))
    return float(seconds) if seconds is not None else None


def _archive_plan(rows: list[tuple[bytes, dict]],
                  now: float) -> tuple[list[int], dict[int, str]]:
    """Which row indexes archive, and why each other row stayed.

    A row archives when it is past `LEDGER_ARCHIVE_AGE_DAYS` AND it is not the NEWEST row
    of its round. The second condition is the one that keeps stores 11 and 12 bounded
    rather than unbounded: `_settle_times` answers "when did this round last matter" with
    a MAX over the round's rows, and both deletion rungs refuse to touch a round whose
    answer is missing — `sweep_automod_branches` counts it `untracked` and prints it as
    `no ledger row` forever. Archiving a settled round's last live row would therefore
    take its settle time with it. #1975 re-triage finding 3 saw that coming at a 30-day
    window, where this store's horizon and those rungs' coincided by construction; #2043's
    14-day window makes the rule load-bearing rather than redundant, because a round's rows
    go archivable a fortnight before either rung asks about it at all — so this condition,
    and not the size of the window, is what keeps stores 11 and 12 bounded. Keeping one row
    per round costs one row per round and preserves every answer `_settle_times` gives — max
    untouched, exactly.

    Returns `(archive_indexes, kept_reason)` where `kept_reason` names, per kept index,
    the rule that kept it (`young`, `newest-of-round`, `no-age`, `not-a-dict`,
    `malformed`) so the report can tell "the window held nothing" from "every candidate
    was somebody's newest row".
    """
    newest: dict[str, float] = {}
    for _line, row in rows:
        if not isinstance(row, dict):
            continue
        rid = row.get("round_id")
        seconds = _ledger_row_seconds(row)
        if not isinstance(rid, str) or not rid or seconds is None:
            continue
        if seconds > newest.get(rid, float("-inf")):
            newest[rid] = seconds

    archive: list[int] = []
    kept: dict[int, str] = {}
    horizon = LEDGER_ARCHIVE_AGE_DAYS * 86400
    for index, (_line, row) in enumerate(rows):
        if not isinstance(row, dict):
            kept[index] = "malformed"
            continue
        seconds = _ledger_row_seconds(row)
        if seconds is None:
            kept[index] = "no-age"
            continue
        if (now - seconds) < horizon:
            kept[index] = "young"
            continue
        rid = row.get("round_id")
        if isinstance(rid, str) and rid and seconds >= newest.get(rid, seconds):
            # Somebody's newest row, on the same clock `_settle_times` answers with.
            kept[index] = "newest-of-round"
            continue
        archive.append(index)
    return archive, kept


def _board_health_diff(before, after, path: str = "") -> list[str]:
    """Dotted paths where two `board_health()` payloads disagree, [] if they agree.

    Compares values, not bytes: `board_health()` returns a dict and its consumers read
    keys, so a payload that re-serialises in a different order is the same output. A type
    change at an equal-valued pair (`1` vs `True`) is a difference. The list is the
    refusal message, so it names the keys and stops at a handful.
    """
    if before == after and type(before) is type(after):
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        out: list[str] = []
        for key in sorted(set(before) | set(after)):
            out.extend(_board_health_diff(before.get(key), after.get(key),
                                          f"{path}.{key}"))
        return out
    return [path or "<root>"]


def _board_health_payload(ledger: Path, backlog_dir: Path | None = None, *,
                          now: float):
    """`board_health()` over one specific ledger file, or None if it cannot be had.

    Resolved through `scripts.automod.backlog`, the module that DEFINES it, so the proof
    is against the function the dashboard calls and not a copy of its logic. Returns None
    — which the caller reads as "neutrality unprovable" and refuses to write — when the
    module cannot be imported or does not answer that name. Both halves matter:
    `scripts/automod/state.py` is imported by path from a tree whose repo root is not on
    `sys.path` when a test loads this file with `importlib`, so a missing module must read
    as an absent proof rather than a green one, and a module that is present but has been
    refactored away from `board_health` must not raise `AttributeError` and take the run
    down mid-sweep instead of refusing.

    `now` is REQUIRED and keyword-only, and it is the whole reason the two calls of one
    proof are comparable. `board_health` stamps `.decisions.window.since/.until` from its
    own `now` (`board_decisions.py:339`), second precision, and the calls are 1-3 s apart,
    so with a per-call clock the payload differs at those two keys over BYTE-IDENTICAL
    bytes — measured on this tree at 1.44 s and 0.87 s a call, diff
    `['.decisions.window.since', '.decisions.window.until']`. That made the shipped probe
    refuse EVERY fold, in production as in the suite, and the fold never wrote (#1975
    review, upheld on reproduction). A keyword default of `None` would leave that clock
    reachable, so a caller that forgets it gets a `TypeError` the rung reports as a
    refusal naming the type — not a silent verdict that the store is unsafe to bound.
    """
    try:
        backlog = _automod_module("backlog")
        if backlog is None or not hasattr(backlog, "board_health"):
            return None
        return backlog.board_health(ledger, backlog_dir=backlog_dir, now=now)
    except Exception as exc:  # noqa: BLE001 - an unprovable fold is a refused fold
        print(f"  promotions ledger: board_health() unreadable ({exc}); not archiving")
        return None


def _ledger_archive_path(ledger: Path, age_seconds: float) -> Path:
    """`promotions-archive-<YYYYMM>.jsonl.gz` beside the ledger, for that row's month.

    Keyed off the row's own age in seconds, which the caller has already proven it has
    (window membership needs it) — never off a `created_at` string that a `ts`-only row
    does not carry, which is how the first draft of this rung produced a file called
    `promotions-archive-None.jsonl.gz` (#1975 round finding). UTC month, because
    `created_at` is written UTC and a month that shifts with the operator's timezone
    splits one month of rows across two files.
    """
    month = time.strftime("%Y%m", time.gmtime(age_seconds))
    return ledger.parent / f"{LEDGER_ARCHIVE_PREFIX}{month}.jsonl.gz"


def _archive_append(target: Path, lines: list[bytes]) -> int:
    """Append the lines not already in `target`, and return how many went.

    Read-then-append rather than blind append because the rewrite that follows can still
    refuse: rows are archived BEFORE they leave the live file — the order that makes loss
    impossible — so a refused rewrite leaves them in both places, and a later run would
    append them again. A duplicate is the lesser evil against deleting a row that never
    reached an archive, and this closes it anyway. Memory is bounded by `lines`, not by
    the archive: the pending set shrinks as matching lines stream past.
    """
    pending = set(lines)
    if not pending:
        return 0
    if target.is_file():
        try:
            with gzip.open(target, "rb") as fh:
                for line in fh:
                    pending.discard(line)
        except OSError:
            pass
    rest = [ln for ln in lines if ln in pending]
    if not rest:
        return 0
    with gzip.open(target, "ab") as fh:
        for ln in rest:
            fh.write(ln)
    return len(rest)


def _rewrite_live_ledger(ledger: Path, body: bytes, read_bytes: int, *,
                        attempts: int = LEDGER_REWRITE_ATTEMPTS,
                        on_attempt=None) -> tuple[bool, str | None]:
    """Rename `body` over the live ledger only once it has stopped growing.

    The race this exists for: `state.append_event` opens the ledger in append mode and
    fsyncs under no lock, at ~2.3 appends a minute on 2026-10-01, while building this
    file takes seconds. `os.replace` is unconditional, so a rename of a snapshot taken
    before those appends silently renames live audit rows out of existence. So: everything
    appended after the snapshot at `read_bytes` is grafted on verbatim — those rows are
    minutes old, never archive candidates — and the rename only goes ahead when
    `st_size` still says `read_bytes + len(appended)`, i.e. nothing landed between the
    tail read and the rename. Five attempts, then a refusal that leaves the live file
    exactly as it was: a sweep that cannot get the ledger to hold still is a sweep that
    does not touch it.

    Returns `(settled, refusal)`. `on_attempt` exists for the test that proves the graft:
    the same hook production never passes, called with the attempt index immediately
    before each rename.
    """
    tmp = ledger.parent / f".{ledger.name}.archiving"
    # The snapshot's end, FIXED for the whole loop. Advancing it to the current size
    # between attempts is the off-by-one that loses a row: the bytes that made this
    # attempt unsafe are exactly the bytes a re-read from the new end would step over.
    # Every attempt therefore re-reads the entire tail since the snapshot.
    seen = read_bytes
    landed = 0
    for attempt in range(attempts):
        try:
            with open(ledger, "rb") as src:
                src.seek(seen)
                appended = src.read()
        except OSError as exc:
            return False, f"ledger unreadable ({exc})"
        payload = body + appended
        try:
            with open(tmp, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            _unlink_quietly(tmp)
            return False, f"temp file unwritable ({exc})"
        if on_attempt is not None:
            on_attempt(attempt)
        try:
            settled = ledger.stat().st_size == seen + len(appended)
        except OSError as exc:
            _unlink_quietly(tmp)
            return False, f"ledger vanished ({exc})"
        if settled:
            os.replace(tmp, ledger)
            return True, None
        landed = max(landed, ledger.stat().st_size - seen)
    _unlink_quietly(tmp)
    return False, (f"ledger kept growing under the archive ({landed} bytes landed in "
                   f"{attempts} attempts); live file untouched")


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def sweep_promotions_ledger(apply: bool, now: float, *, ledger: Path | None = None,
                            backlog_dir: Path | None = None, health=None,
                            on_attempt=None) -> dict:
    """Archive promotions-ledger rows older than LEDGER_ARCHIVE_AGE_DAYS.

    `~/.local/state/lloyd-automod/promotions.jsonl` is the loop's own audit trail: 32
    MB, ~28,600 rows, +2 MB a day, decoded whole by `board_health()` on every dashboard
    cycle and pruned by nothing until now. Rows past the window are copied into
    `promotions-archive-<YYYYMM>.jsonl.gz` beside the ledger and taken out of the live
    file; each round's newest row stays (see `_archive_plan`), and the whole write
    happens only after `board_health()` answers identically over the file as it is and the
    file as it would be.

    Order of one applied run, which is the order that makes a crash survivable:

    1. Read the ledger once, as BYTES. The bytes are the snapshot every later decision is
       taken from, and the size the graft compares against.
    2. Choose the rows (`_archive_plan`).
    3. Prove neutrality over two temp files — the snapshot's bytes, and the kept bytes —
       so a row appended while the proof ran cannot make the diff non-empty and cannot
       make it empty either. Non-empty diff, or no `board_health` to ask: refuse, live
       file untouched, and the byte pair reported is the size the store still is.
    4. Append the archived lines to their monthly gzips (deduped, so a retry of a run
       that refused at step 5 cannot double-write).
    5. Rename over the live file through `_rewrite_live_ledger`, grafting the tail.

    A dry run stops after step 3 and writes nothing at all — not even the archive — so
    the numbers an operator approves are the numbers this function computed over the bytes
    it actually read.

    `health` injects the probe (`(path) -> payload`) so a test never pays for a real board
    walk: the real `board_health()` costs ~1.4-1.6 s over an EMPTY board and this calls it
    twice. `backlog_dir` redirects the board for the same reason. With `health` absent the
    shipped probe asks `board_health()` twice on THIS run's `now` — see
    `_board_health_payload` — so the only thing a non-empty diff can be is a real change in
    what the board panel answers, which is what makes a write admissible.
    """
    ledger = Path(ledger) if ledger is not None else Path(AUTOMOD_LEDGER)
    out = {"moved": 0, "bytes": 0, "archived": 0, "before": 0, "after": 0,
           "refused": None, "candidates": 0, "kept_newest": 0}
    try:
        raw = ledger.read_bytes()
    except OSError:
        out["after"] = 0
        return out
    out["before"] = len(raw)
    out["after"] = len(raw)

    lines = raw.splitlines(keepends=True)
    rows: list[tuple[bytes, dict]] = []
    for line in lines:
        body = line.strip()
        try:
            row = json.loads(body.decode("utf-8", "replace")) if body else None
        except ValueError:
            row = None
        rows.append((line, row if isinstance(row, dict) else None))

    archive_idx, kept = _archive_plan(rows, now)
    out["kept_newest"] = sum(1 for why in kept.values() if why == "newest-of-round")
    out["candidates"] = len(archive_idx) + out["kept_newest"]
    if not archive_idx:
        return out

    dropped = set(archive_idx)
    moved = [rows[i][0] for i in archive_idx]
    kept_bytes = b"".join(line for i, (line, _r) in enumerate(rows) if i not in dropped)
    out["bytes"] = sum(len(ln) for ln in moved)

    # ── the neutrality proof, in both modes ───────────────────────────────────
    # It runs on a dry run too, because it is the only thing that says this archive is
    # safe, and a safety check that only runs when it is already too late to change its
    # mind is a log line and not a gate. Both sides are COPIES of the snapshot: comparing
    # the live file against a copy would let a row that landed in the 3 s between the two
    # reads decide the verdict.
    probe = health or (lambda path: _board_health_payload(path, backlog_dir, now=now))
    before_copy = ledger.parent / f".{ledger.name}.neutral-before"
    after_copy = ledger.parent / f".{ledger.name}.neutral-after"
    try:
        before_copy.write_bytes(raw)
        after_copy.write_bytes(kept_bytes)
    except OSError as exc:
        out["refused"] = f"proof copy unwritable ({exc})"
    try:
        if out["refused"] is None:
            try:
                bh_before, bh_after = probe(before_copy), probe(after_copy)
            except Exception as exc:  # noqa: BLE001 - see _board_health_payload
                bh_before = bh_after = None
                out["refused"] = f"board_health() raised {type(exc).__name__}: {exc}"
            if out["refused"] is None:
                if bh_before is None or bh_after is None:
                    out["refused"] = "board_health() unavailable — neutrality unprovable"
                else:
                    moved_keys = _board_health_diff(bh_before, bh_after)
                    if moved_keys:
                        out["refused"] = ("board_health() changed at "
                                          f"{', '.join(moved_keys[:5])}")
    finally:
        _unlink_quietly(before_copy)
        _unlink_quietly(after_copy)
    if out["refused"] is not None:
        # `X -> 0 bytes live` is the one sentence this store must never print while the
        # audit ledger is still 32 MB: a refused archive leaves the live file at the size
        # it was, so the pair repeats the before count instead of inventing an after one.
        out["after"] = out["before"]
        out["moved"] = out["bytes"] = out["archived"] = 0
        return out

    out["moved"] = len(moved)
    if not apply:
        return out

    months: dict[str, list[bytes]] = {}
    for index in archive_idx:
        seconds = _ledger_row_seconds(rows[index][1])
        month = time.strftime("%Y%m", time.gmtime(seconds))
        months.setdefault(month, []).append(rows[index][0])
    for month in sorted(months):
        target = ledger.parent / f"{LEDGER_ARCHIVE_PREFIX}{month}.jsonl.gz"
        out["archived"] += _archive_append(target, months[month])

    settled, refusal = _rewrite_live_ledger(ledger, kept_bytes, len(raw),
                                           on_attempt=on_attempt)
    if not settled:
        out["refused"] = refusal
        out["moved"] = out["archived"] = 0
        out["after"] = out["before"]
        return out
    try:
        out["after"] = ledger.stat().st_size
    except OSError:
        out["after"] = len(kept_bytes)
    return out


def _ledger_line(l: dict) -> str:
    """The thirteenth store line, identical in dry run and `--apply`.

    `0 archived` means the window held nothing, and the byte pair is then the same number
    twice — the honest shape of "this run changed nothing". A refusal is not a `0`: it is
    printed as a refusal repeating the live byte count, because `0 archived` would read to
    the operator who approves this run as an empty window when in fact a 32 MB audit file
    is sitting there unbounded and the rung knows it.
    """
    if l["refused"]:
        return (f"  promotions ledger: REFUSED ({l['refused']}) — live file untouched, "
                f"{l['before']} bytes")
    if l["moved"] == 0 and l["kept_newest"]:
        return (f"  promotions ledger: 0 archived ({l['before']} -> {l['after']} bytes "
                f"live) — all {l['kept_newest']} rows past "
                f"{LEDGER_ARCHIVE_AGE_DAYS}d are their round's newest, kept so the "
                f"round-dir and branch rungs still have a settle time")
    return (f"  promotions ledger: {l['moved']} archived ({l['before']} -> "
            f"{l['after']} bytes live)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete/gzip (default: dry-run report)")
    args = ap.parse_args()
    now = time.time()
    mode = "APPLY" if args.apply else "DRY RUN"

    # The header goes out before anything is counted or deleted, and in both
    # modes. An operator approves `--apply` from the dry run's numbers, and
    # every one of them describes whichever root this run resolved — a value that depends
    # on the caller's tree, its environment and its `$HOME`, and appears nowhere
    # in the report today (#1415). Naming the root above the numbers is what makes
    # them approvable, and is the only line that says which tree they describe.
    print(f"[retention-sweep] {mode}")
    print(f"  data root: {DATA_ROOT}")

    # The two automod stores answer to a different question than the data root above:
    # is THIS TREE the production checkout? One of them is the live repo's refs, and a
    # round's worktree shares them, so a `git branch -D` from inside a gate would delete
    # production branches with no data root in between. The refusal is therefore printed
    # in both modes — a rung that simply vanished from the dry run's list is a rung the
    # operator never saw missing — and an `--apply` that was refused ends the run
    # `NOT_PRODUCTION_EXIT`, the status the `.lloyd-data-root` refusal above exits with.
    # A dry run still ends 0 from any tree: it deletes nothing, and the root it names is
    # the reason for running it (#1415, pinned by
    # `test_a_bare_interpreter_reports_the_root_it_resolved`).
    refusal = automod_rung_refusal()
    dirs = branches = ledger_rows = None
    if refusal is None:
        dirs = sweep_automod_worktrees(args.apply, now, repo=AUTOMOD_REPO)
        branches = sweep_automod_branches(args.apply, now, repo=AUTOMOD_REPO)
        # LAST of the three, because the two rungs above have just read this file: an
        # archive that ran first would move the settle times they are about to be
        # reported on, and the dry run an operator approved would not be the run that
        # happened.
        ledger_rows = sweep_promotions_ledger(args.apply, now)

    logs_n, logs_b = sweep_task_logs(args.apply, now)
    sess_n, sess_b = sweep_sessions(args.apply, now)
    runs_n, runs_b = sweep_autonomy_runs(args.apply, now)
    act_f, act_e, act_l = sweep_activity_logs(args.apply)
    cand_n, cand_b = sweep_candidates(args.apply, now)
    scr_n, scr_b = sweep_transcript_scratch(args.apply, now)
    spill_n, spill_b = sweep_session_spills(args.apply, now)
    gq_n, gq_b = sweep_groundskeeper_queue(args.apply, now)
    wr_n, wr_b, wr_skip = sweep_worker_runs(args.apply, now)
    wq_n, wq_b, wq_skip = sweep_queue_rows(args.apply, now)

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
    # One line for the pair, counted in files, and identical in both modes like every
    # line above it: the operator approves `--apply` from these numbers, and the bytes
    # here are the whole of what approving them gives up.
    print(f"  groundskeeper queue ({GROUNDSKEEPER_QUEUE_FILE.name} + "
          f"{GROUNDSKEEPER_WRITES_FILE.name}) >{GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS}d: "
          f"{gq_n} deleted, {gq_b / 1024:.0f} KiB freed")
    # Same line in both modes for the same reason as the spill line above. A skip is put
    # in place of the counts, never beside a `0`, so the number `0` on this store keeps
    # its only honest meaning — the window held nothing.
    if wr_skip:
        print(f"  workers.db runs >{WORKER_RUN_MAX_AGE_DAYS}d: {wr_skip} — nothing pruned")
    else:
        print(f"  workers.db runs >{WORKER_RUN_MAX_AGE_DAYS}d: "
              f"{wr_n} deleted, {wr_b / 1024:.0f} KiB freed")
    queue_label = (f"  workers.db queue ({'/'.join(QUEUE_TERMINAL_STATES)}) "
                   f">{QUEUE_MAX_AGE_DAYS}d")
    if wq_skip:
        print(f"{queue_label}: {wq_skip} — nothing pruned")
    else:
        print(f"{queue_label}: {wq_n} deleted, {wq_b / 1024:.0f} KiB freed")
    # Last, so the two lines that can name a production ref are the last thing an
    # operator reads before deciding whether the run did what they asked.
    if refusal:
        print(f"  automod stores: {refusal}")
    else:
        print(_worktree_line(dirs))
        print(_branch_line(branches))
        # Printed only here, inside the same guard that refused the other two: the fold
        # rewrites the loop's own audit trail, so a run from a round's worktree or a
        # sandbox declines it too and still prints ten store lines plus one refusal line.
        print(_ledger_line(ledger_rows))
    if refusal and args.apply:
        return NOT_PRODUCTION_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
