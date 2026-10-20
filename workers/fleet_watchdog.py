"""Fleet surveillance for the autonomy scheduler — the alarms, not the dispatch.

Three alarms and one scan live here, and none of them is about enqueuing:

- the grossly-overdue / starving-queue stall alarm (#24/#48/#75, #1550), with the
  operator-pause clause that says when a held pool, not a stuck scheduler, owns a
  backlog, and the queue clause that names which source is waiting and says priority
  starvation when the pool is demonstrably claiming other work (#2692);
- the second stall assertion (#421), keyed on each task's own `next_run`, on its own
  streak and its own 24 h cooldown;
- the unparseable-task-file scan (#939), on its own 30 min cadence, which is the only
  thing that notices a task file the scheduler silently drops.

They used to sit inside `workers/sources/scheduled_task.py:enqueue_if_due`, and that
cost more than length: `workers/pool.py` skips that call outright for a source whose
config block says `enabled: false`, so a dispatch-side switch disarmed every alarm
here and raised no alert about the disarmament — an alarm cannot report its own
silencing (#1682). They were also paced by the dispatch tick and paid its watermark
round-trip, while their own cooldowns are 6 h, 24 h and 30 min.

The seat is `WorkerPool._watch_fleet`, beside `_maybe_sweep_poisoned` and
`_probe_services`, whose docstring is the precedent: a reader that has to run when
the fleet is not running cannot be a work source, because a work source is starved
exactly when the fleet is backed up.

What stayed on the dispatch side, and why:

- `_vllm_healthy` and the gate that acts on it, plus `_note_vllm_outage` and the
  outage accounting it drives. #938's invariant is that an outage pauses DISPATCH and
  not WATCHING, and the gate is dispatch: `enqueue_if_due` probes it, accounts for the
  tick, and returns before enqueuing. If the gate moved here the invariant would hold
  only while nothing sat behind it, which is how #938 broke the first time; #1683 owns
  the outage alarm's threshold, so it stays put for that round too.
- `get_due_tasks`, the per-task model-server skip, and the enqueue itself.

The pacing came with the functions: the 5-tick streaks, the 6 h and 24 h cooldowns and
the 30 min scan are the constants below, unchanged, and the pool's own tick (60 s)
paces them exactly as the source's tick did. The one value read from the source's
config block is `max_duration_seconds` — never its `enabled` key, which is the whole
point of the move.
"""

from __future__ import annotations

import asyncio
import logging

from workers.queue import WorkQueue

logger = logging.getLogger("lloyd-workers.fleet_watchdog")

#: The queue `source` id of the dispatch path these alarms watch, and the
#: sources-config block whose `max_duration_seconds` the starving clause weighs the
#: backlog against. The same string as `scheduled_task.NAME`, spelled out here rather
#: than imported: the watchdog must not have to import the dispatcher to ask it what it
#: is called, and a key of the watchdog's own in `config.yaml` would be a config change
#: outside what a round of this size may land.
NAME = "scheduled-task"
#: The fallback the dispatch tick applied before the split (`src_cfg.get(
#: "max_duration_seconds", 1800)`), kept rather than trusted from the live config so
#: a missing block reads as the old default instead of as zero.
DEFAULT_MAX_DURATION_SECONDS = 1800

# Stall alarm: a task overdue by more than this multiple of its interval (with its
# dependency met) should have run; if any are for N consecutive checks, alert once.
_STALL_INTERVAL_MULT = 2.5
_STALL_ALARM_TICKS = 5
# The alarm fired 100 times in 6 days because it could not tell "dispatch is
# broken" from "there is no capacity": #24/#48/#68/#75 are permanently overdue
# when demand exceeds the 2 shared slots. Re-alerting at most this often keeps
# a genuine stall visible instead of drowned.
_STALL_ALERT_INTERVAL_SECONDS = 6 * 3600
# The second, quieter stall assertion (#421). The alarm above is keyed on
# due-ness and `last_run` and its three exclusions are load-bearing for noise —
# they are also exactly the shapes that sat silent for 41 h and 60 h on
# 2026-09-07/08, so this one is keyed on each task's own `next_run` and reads
# none of those three inputs. Quieter by construction: the key is a whole
# period (not 2.5x a partial one) and the cooldown is a day, not 6 hours. Its
# streak is a separate counter so neither alarm can reset the other's.
_STALL_NEXTRUN_TICKS = 5
_STALL_NEXTRUN_ALERT_INTERVAL_SECONDS = 24 * 3600
#: How far past its OWN `next_run` an upstream may sit, with a `success` newest run
#: record, before its lateness is a reason to name the chain: one hour. #2695 set the
#: shape and the width it shipped with (30 min); #2950 widened the width to what its
#: own measurement said.
#:
#: A completion is what moves `next_run` (`app/autonomy.py`, `next_run_gap`'s own
#: docstring), so between an upstream's launch and its finish its stamp is behind by
#: exactly the length of its run — and on the 05:00 nightly chain that window is the
#: whole 05:00-05:09 slot the watchdog's own tick lands in. Without a threshold here,
#: `up_hours > 0` admitted the dependent every single night: 2026-10-08 through
#: 2026-10-13 each carry the alert, #42 named at 0.1 h behind an upstream that was
#: four minutes into a run that finished eight minutes later. An alarm that fires on
#: the arithmetic of a healthy chain is not an alarm, and its 24 h cooldown made it a
#: metronome.
#:
#: What the grace does NOT touch is the other disjunct. A newest run record that is
#: not a `success` opens the door at any lateness, including none, because that is the
#: fault #2417 was built for — #38 dying at 05:04:09Z on 2026-10-08 and its three
#: dependents going 6.2 h / 3.4 h / 2.7 h late behind it, unpinned byte for byte by
#: `test_a_dependent_three_hours_behind_a_failed_upstream_is_flagged`. Sibling in
#: reasoning: `MISSING_ARTIFACT_GRACE_HOURS` (`grep -n '^MISSING_ARTIFACT_GRACE_HOURS'
#: app/autonomy.py`) bounds a hold that no longer has a thing to wait for, the same
#: shape of judgement on the dispatch side.
#:
#: The width is a ruling, and a measured one — #2695's `ruling` field, executed here.
#: Over 2026-10-14..10-17 the readings this grace actually suppressed ran 1…29 min
#: (`grep -ho "sits [0-9]* min past its own" ~/lloyd-data/logs/server.err*` over the
#: retained logs), so the 30 min it shipped with sat at 97% of its own tail and one
#: slightly longer upstream run would have re-opened the alarm #2695 killed; #24's
#: 28.8-min run at 2026-10-17T16:18Z is that same reading from the run side. One hour
#: is 2x the worst measured run. The alternative #2695 weighed and REJECTED is deriving
#: the grace as a fraction of the UPSTREAM's own interval, and it stays rejected for two
#: reasons: an interval says nothing about run length — #24 is `daily` and its run was
#: 28.8 min, so a fraction wide enough for it is nowhere near wide enough for an
#: `every-15min` upstream, which is exactly where the nightly arithmetic #2695 closed
#: lives — and a bound that moves with a schedule field cannot be read off the line this
#: record prints, which is the only evidence a reader of `server.err` has. One fixed
#: named constant, widened when a measurement says the tail moved, is the ruling.
#:
#: One scan says so in the log per suppressed chain, unsuppressed by any cooldown: the
#: window this closes runs from zero to `_NEXTRUN_UPSTREAM_GRACE_HOURS`, so the record
#: cannot repeat the way the declared-park log could, and that log's 24 h slot is what
#: keeps "nothing is late" and "everything late was parked" two sentences rather than
#: this one's budget. Its LOWER edge is #2950: an upstream whose own stamp is still in
#: the future is not a late upstream being forgiven, and printing it as one is what put
#: 3,556 lines reading `sits -167 min past its own next_run` for #98 -> #24 into four
#: days of log.
_NEXTRUN_UPSTREAM_GRACE_HOURS = 1.0
#: Where the `next_run` alarm's last-reached-a-person instant is persisted: the
#: queue's own watermark table, under this source, in the same row pattern the
#: operator pause uses (`workers/pool.py:PAUSE_WM_SOURCE`). See
#: `_last_alert_instant` for why the instant is the VALUE rather than the row's
#: `updated_at`.
_NEXTRUN_WM_KEY = "nextrun_alerted_at"
#: The due-ness alarm's own row beside it (#2662) — same source, its own key, so
#: neither alarm can spend the other's cooldown. The key string is also the `_state`
#: key it mirrors, which is what lets one pair of functions serve both alarms: the
#: `_state` name and the watermark name for an alarm's instant are the same word.
_STALL_WM_KEY = "stall_alerted_at"

# How often to re-scan the task files for ones the scheduler cannot parse (#939).
# This used to be a one-shot per process: the flag was set on the first tick and
# every later tick inherited that first tick's answer, so a file that turned
# unparseable while the scheduler was up — a half-written file from a crashed
# vault sync, a bulk edit cut off before its closing `---` fence — stopped being
# dispatched and produced no alert until the next backend restart, which a round
# under observation can defer for a day. Re-scanning is the cheap half; neither
# stall alarm can cover this class, because both iterate sets built from files
# that ALREADY parse, so a file parsing to `None` is missing from their input as
# well as from dispatch. 30 min costs one directory scan per interval at the
# pool's 60 s tick and sits inside the one-hour ceiling #939 sets.
_UNPARSEABLE_SCAN_SECONDS = 30 * 60
# ── the worker pool's failure-throughput census (#2890) ──────────────────────
#: The day-over-day ratio that is worth a line: the newer complete UTC day must
#: carry at least this many times the older day's `failed` count, with the
#: denominator floored at 1 so a quiet predecessor does not divide away a day of
#: real failures. Priced on the live store rather than guessed — over the 24
#: adjacent pairs of COMPLETE UTC days that `workers.db` could evaluate on
#: 2026-10-17, this fires on 2: `2026-09-23` (10 failed vs 1) and `2026-10-16`
#: (137 vs 7). Two lines across the 26 days of `runs` rows that store spans is a
#: census of an event, which is why this is not the daily-note flap counter Config
#: Application #40 declined on 2026-10-01 (`run_40_20261001_090101.md:67`) for
#: instrumenting a channel whose flapper was already fixed: that one measured a
#: producer known to be quiet, this one writes for a population no writer covered.
_THROUGHPUT_RATIO = 10
#: The absolute floor, #2899's second leg on the SAME reading. A ratio is a
#: comparison, and a comparison is blind on exactly the days #2890's owed entry
#: named: `2026-09-29` (187 `failed` runs against 27 = 6.93x) and `2026-10-07` (75
#: against 12 = 6.25x) are the two worst days the live store holds and neither
#: reached a note, because the day before each was ALREADY bad. This leg asks the
#: other question of the same newest-complete-day pair — is this day itself bad,
#: whatever its predecessor looked like — so it inherits the adjacency gate and the
#: rule that an absent predecessor is not a zero, rather than counting a day the
#: pool never ran. Pinned rather than derived, because a baseline computed from
#: recent days is the very blindness being repaired: over the 25 complete UTC days
#: `workers.db` held on 2026-10-17 the sorted per-day `failed` counts are
#: [1,5,6,6,7,7,8,8,9,9,10,10,10,12,13,14,15,20,27,27,32,57,75,137,187], and the
#: only gap wide enough to separate an ordinary bad day from the four worth a line
#: is 32→57, so ANY floor inside it fires on exactly {09-24:57, 09-29:187,
#: 10-07:75, 10-16:137} — three lines in 25 days beyond the one the ratio already
#: caught. Where inside that gap to put it, and whether 40 is still the right place
#: after a week of real traffic, is #2899's owed measurement, not a call this
#: constant can make for a future store.
_THROUGHPUT_FLOOR = 40
#: This alarm's own "we already told someone" row, holding the UTC DAY it last
#: reported — a day string, not an instant. The dedupe unit is a day, so a
#: cooldown in seconds would let the first tick after the pair rolls over post the
#: SAME day again, and a promise about a day cannot be kept with a stopwatch. Keyed
#: beside `_STALL_WM_KEY` and `_NEXTRUN_WM_KEY` under this module's own name, so no
#: alarm can spend another's silence and no alarm's persisted instant moves when
#: this one fires. It is also the ONLY throughput row: the ratio leg and the floor
#: leg (#2899) are two rules over one reading of one day, so they share this
#: watermark rather than each carrying a key of its own — a per-leg key would let a
#: day that trips both reach a person twice, which is one alarm spending its own
#: silence.
_THROUGHPUT_WM_KEY = "throughput_day"
#: Failure classes, read off `runs.summary` as substrings. These are the strings
#: the 2026-10-16 census actually found (99 / 24 / 3 / 6 / 3 / 1 of its 137), not a
#: taxonomy imposed on it — and the census reports an `other` remainder beside them
#: regardless, so the tallies always sum to the count on the line and a failure mode
#: nobody has classified yet reads as "other grew" instead of as silence.
_THROUGHPUT_CLASSES: tuple[tuple[str, str], ...] = (
    ("no structured answer", "no structured answer"),
    ("connect error", "ConnectError"),
    ("connect timeout", "ConnectTimeout"),
    ("peer closed connection", "peer closed connection"),
    ("turn never completed", "turn never completed"),
    ("model server refused the task", "deferring task"),
)
#: The floor on the since-boot window, in minutes (#2956). The line expresses the
#: day's failures that fall at or after the host boot instant as a rate per this many
#: minutes, so a boot-window burst and an hour of live pressure are readable on one
#: scale. It is a FLOOR on the denominator, not a fixed denominator: a census that
#: runs two minutes after a boot with 3 failures behind it would divide by 2 and
#: print 22.5 / 15 min, which is arithmetic about how recently the machine came up
#: rather than a rate anything can be compared to. Fifteen minutes is the unit the
#: defect was argued in ("with a post-boot rate of 0 / 15 min, report the post-boot
#: rate beside the raw day count"), so the line keeps that unit.
_THROUGHPUT_RATE_WINDOW_MINUTES = 15
#: Where the host's uptime is read (#2956). `/proc/uptime`'s first field is seconds
#: since boot, so boot is `now - uptime`, and that is one `open()` beside the
#: `workers.db` read the census already does. `uptime -s` returns the same instant and
#: costs a fork on a tick that runs every 60 s, which is the reason this reads the
#: file rather than shelling out. Absent outside Linux, and the caller reads that as
#: no boot knowledge and prints the line it prints today — never as a guess.
_PROC_UPTIME_PATH = "/proc/uptime"
# Surveillance state, module-global for the reason the source had it: the streaks and
# cadence slots have to survive from one scheduler tick to the next, and
# `tests/test_autonomy_scheduler.py` restores this dict's CONTENTS around every test
# that reads it so no test inherits another's streak (#939, #1682). The outage's three
# keys stayed in `scheduled_task._state` with the alarm that writes them.
_state = {
          # `_unparseable_scan_at` is the instant the parse scan last ran (None
          # until the first tick takes its slot) and `_unparseable_alerted` the
          # file names it has already raised. Together they make the scan a
          # cadence with transition-only alerting, so neither an unchanged bad
          # set nor a healthy one sends a message every half hour.
          "unparseable_scan_at": None, "unparseable_alerted": set(),
          "stall_streak": 0, "stall_alerted_at": None,
          # `nextrun_alerted_at` is the instant the `next_run` alarm last REACHED A
          # PERSON, and is written only on that route. A scan that found nothing but
          # declared parks never reaches one, so it writes the key beside it instead —
          # its own log cadence, so the record repeats at the alarm's interval without
          # spending the alert cooldown that a real stall will need (#1661). Of these
          # Both alert instants are mirrored into the queue — `nextrun_alerted_at`
          # under `_NEXTRUN_WM_KEY` (#1735) and `stall_alerted_at` under
          # `_STALL_WM_KEY` (#2662) — because each is the promise "once per 24 h" /
          # "once per 6 h" and that promise outlives the process that posted the
          # message. A streak and a log slot stay in here: they are this tick's
          # business, and #1735's rationale was always the route to a person, not the
          # name of one key.
          "nextrun_streak": 0, "nextrun_alerted_at": None,
          "nextrun_parked_logged_at": None,
          # `throughput_day` is the UTC day the failure-throughput census last put on
          # the note (#2890) — a day string, mirroring `_THROUGHPUT_WM_KEY`, and the
          # only key in here that is not an instant. The census has no streak: a day
          # either carried the ratio or it did not, and re-confirming that across five
          # ticks would add nothing a single read of the store does not already say.
          "throughput_day": None}


def _source_cfg() -> dict:
    """The `scheduled-task` block of the sources config, re-read every tick.

    The import is function-local because it is the cycle that matters, not the taste:
    `workers.sources` imports `workers.pool`, and `workers.pool` reaches this module
    from `_scheduler_loop`. Re-reading is also what makes `/api/workers/enable` and the
    Tools page take effect without a restart — and the one field taken out of the
    result is `max_duration_seconds`, never `enabled`.
    """
    from workers.sources import get_sources_config
    return dict(get_sources_config().get(NAME) or {})


def _max_duration_seconds() -> int:
    """What the starving clause weighs the oldest claimable item against.

    Read here instead of handed in as `src_cfg`, so the watchdog needs nothing from the
    dispatch path, and falls back to the same 1800 s the dispatch tick fell back to
    rather than to zero — a zero cap would call every queued item starving.
    """
    try:
        return int(_source_cfg().get("max_duration_seconds")
                   or DEFAULT_MAX_DURATION_SECONDS)
    except (TypeError, ValueError):
        return DEFAULT_MAX_DURATION_SECONDS


def _unparseable_task_files() -> list[str]:
    """Task files the scheduler cannot parse (invisible to dispatch).

    Reads `autonomy.AUTONOMY_DIR`, the one directory constant the scans beside it
    already use (`recover_stuck_tasks` at app/autonomy.py:96, `_find_task_file` at
    :161), rather than spelling `Path.home() / "obsidian" / "autonomy"` a second
    time: the value is the same in production — that is how the constant is
    defined, at app/autonomy.py:77 — and the indirection is what lets a test point the
    real scan at a scratch fleet instead of the live board it would alert about."""
    import re
    from app import autonomy
    d = autonomy.AUTONOMY_DIR
    bad = []
    for p in d.glob("*.md"):
        if not re.match(r"\d+-", p.name):
            continue
        if autonomy._parse_task_file(p) is None:
            bad.append(p.name)
    return bad


def _active_task_ids(queue: WorkQueue) -> set:
    """Task ids with a live queue row (queued/claimed/running)."""
    try:
        items = queue.list_items(source=NAME, limit=500)
    except Exception:
        return set()
    return {str(i.payload.get("task_id")) for i in items
            if i.state in ("queued", "claimed", "running")
            and i.payload.get("task_id") is not None}


def _grossly_overdue(queue: WorkQueue) -> list:
    """Tasks that are DUE RIGHT NOW yet overdue by > _STALL_INTERVAL_MULT *
    interval AND have no queue row waiting for them.

    The queue-row exclusion is the whole point: a task sitting in the queue
    behind a saturated pool is starved for capacity, not stalled, and flagging
    it forever is what turned this alarm into noise. A task that is due, this
    stale, and NOT in the queue means the dispatch path itself is broken (the
    2026-05-28 silent stall this was built for)."""
    from app import autonomy
    # One resolution set and one instant, the same pair `get_due_tasks` uses
    # (#870): this used to hand `_is_task_due` the runnable set to resolve
    # `depends_on` with, so a `paused` upstream was invisible here too and the
    # alarm's idea of "due" was not dispatch's idea of due.
    resolution = autonomy.dependency_resolution_set()
    active = _active_task_ids(queue)
    now = autonomy._utcnow()
    overdue = []
    for t in autonomy._all_runnable_tasks(resolution):
        if not autonomy._is_task_due(t, resolution, now=now):
            continue
        if str(t.get("id")) in active:
            continue
        interval = autonomy._frequency_interval_seconds(t)
        last = autonomy._parse_iso(t.get("last_run"))
        if not interval or not last:
            continue
        if (now - last).total_seconds() > _STALL_INTERVAL_MULT * interval:
            overdue.append(t.get("id"))
    return overdue


def _row_vs_records(task: dict) -> dict:
    """What a task's own run records claim, set beside what its row's stamps claim.

    The watchdog's "did it actually run" half, asked of the records (#2342). This is
    the file's only read of them, and it is a recent one: before this function
    existed the scan reasoned from the row alone — `hours` from
    `autonomy.next_run_gap`, `hold` from `autonomy.hold_reason`, `queued` from live
    queue rows — and a search of this file for the run-record tree, its database or
    a run listing answered with one hit, prose in `_last_alert_instant`'s (then
    `_last_nextrun_alert`'s) docstring
    warning against raw SQL. So `last_run`, a field in a vault file another writer
    may never have advanced, was the only account this alarm had, and a row that lost
    a completion produced a cause claim about dispatch that the run's own record on
    disk contradicted: `run_79_20261006_051105.md` finished at
    `2026-10-06T05:17:08+00:00` while the alert at 22:00:28 that evening read a
    `next_run: 2026-10-06T05:00:00` as 24 h of silence.

    Four fields, and what each is for:

    * `record_at` / `record_run` — the newest `success` record's finished instant and
      its id, quoted into the alert so a reader can open the file. `''` when the
      records answer nothing, which covers "no directory", "unreadable front matter"
      and "no success" alike: `app.autonomy.newest_successful_run` cannot tell those
      apart and neither can a sentence, because inventing a claim about records this
      reader did not read is the same defect pointed the other way. Such a task keeps
      the wording it has always had, and still alerts.
    * `row_last` — the row's own `last_run`, echoed so the alert can show BOTH sides
      of a disagreement instead of asserting one.
    * `lost_completion` — a success the row never recorded. Strictly newer than
      `last_run`, because `last_run` tracks the last SUCCESS (#86): a record from
      before the row's own stamp is the two accounts agreeing.

    The shape this function cannot describe — a row with no `next_run` at all, which
    `past_next_run` can never flag because the bound requires
    `hours_past_next is not None` — is admitted by the scan itself and shows up here
    as `row_last: ''` beside a real `record_at`. That is #96, unalerted on both
    alarms since before this file existed while its 2026-10-05 success record sat on
    disk.

    Cost, measured on the live board the evening this landed rather than estimated:
    38 rows in the resolution set, 2 shortlisted for a record walk (#68, 499 h past
    its `next_run`, and #96 with no stamp at all), so 2 walks of one small directory
    per tick. That is not a new kind of cost here: `_is_task_due` and `hold_reason`
    each walk one of these same directories for any task everything else called due
    (`app/autonomy.py:2282`, `:2407`, `:2419`), on the scheduler's own 60 s tick.
    """
    from app import autonomy
    at, run_id = autonomy.newest_successful_run(task.get("id"))
    row_last = autonomy._parse_iso(task.get("last_run"))
    return {
        "record_at": at.isoformat() if at else "",
        "record_run": run_id,
        "row_last": row_last.isoformat() if row_last else "",
        "lost_completion": bool(at is not None
                                and (row_last is None or at > row_last)),
    }


def _upstream_block(task: dict, gap: dict, resolution: list[dict], now) -> dict:
    """What the upstream of a late dependent is itself doing — #2417's early door.

    The nightly chain of 2026-10-08 is the shape this exists for: #38's run died at
    `05:04:09Z`, and #42, #39 and #40 went ~6.2 h / 3.4 h / 2.7 h late as a block
    while the only alarm that could see them waited a full declared period apiece.
    Two facts answer that door, both about the task named in `depends_on`: what its
    NEWEST run record says, from `autonomy.newest_run_record` — status-agnostic,
    because `newest_successful_run` answers "failed" and "no records at all" with
    the same `(None, "")` — and whether its own `next_run` stamp is behind.

    The verdict is `late_or_dead`, and it is the union of those two facts and
    nothing else: the newest record is not a `success`, OR the upstream is itself
    past its own `next_run` by more than `_NEXTRUN_UPSTREAM_GRACE_HOURS`. The second
    disjunct's threshold is #2695, and it exists because the stamp is not a health
    signal on its own: `next_run` moves at COMPLETION (`next_run_gap`'s own
    docstring, `app/autonomy.py`), so an upstream that launched on time and is still
    running has a stamp behind it by exactly the length of that run. On the 05:00
    nightly chain the watchdog's tick lands inside that window every single morning,
    which made `> 0` a line the fleet could not stop printing — six nights running,
    about a chain that completed minutes later each time.

    The suppression RECORD that sits beside the verdict carries BOTH edges of that
    threshold, and that is #2950: it is emitted only for
    `0 < up_hours <= _NEXTRUN_UPSTREAM_GRACE_HOURS`, because what it prints asserts the
    upstream is behind, while `not late_or_dead` is equally true of a negative
    `up_hours` — a `daily` upstream whose own completion moved its stamp hours into the
    future, which is the ordinary state of the fleet's most reliable chain and was the
    3,556-line `sits -167 min` record. The VERDICT is not part of that change: an
    upstream ahead of its own stamp has never named a chain, and the dependent's own
    lateness stays the one-period bound's sentence.

    What the grace lets through is unchanged in kind: an early report here still
    means the upstream has not finished what the dependent is waiting on, either
    because its newest record says so or because it is late doing it. A `paused` or
    `draft` upstream that last completed cleanly is still NOT named early — its
    lateness is a decision somebody wrote down, and the one-period bound still
    catches it.

    Two shapes answer `{}`: no `depends_on` (no chain to name), and a `depends_on`
    whose id names no task file — #558's case, where the hold is already held and
    there is no record and no stamp to name in a sentence. Such a row keeps the
    bound it has always had.

    Cost is one walk of the UPSTREAM's record directory, paid only for a row that
    already names an upstream AND is past its own `next_run`. A healthy board has
    no such row, for the same reason the verdict comes out False there: completing
    moved its stamp forward. `gap` is handed in rather than recomputed, so the
    dependent's own arithmetic stays in one place.
    """
    from app import autonomy
    import datetime as _dt
    up_id = task.get("depends_on")
    hours_past = gap.get("hours_past_next_run")
    if up_id is None or not (hours_past is not None and hours_past > 0):
        return {}
    up = next((u for u in resolution
               if u.get("id") is not None and str(u.get("id")) == str(up_id)),
              None)
    if up is None:
        return {}
    up_hours = autonomy.next_run_gap(up, now=now)["hours_past_next_run"]
    status, when, run_id = autonomy.newest_run_record(up_id)
    # Only the STAMP disjunct carries the grace, and only when the record is clean:
    # a `success` newest record plus a stamp minutes behind is a chain mid-flight,
    # which is the shape `_NEXTRUN_UPSTREAM_GRACE_HOURS` exists to hold shut. The
    # record disjunct is untouched, so a dead upstream opens this door at any
    # lateness and `late_or_dead` says so with the same words it always used.
    late_or_dead = (status != autonomy.RUN_STATUS_SUCCESS
                    or (up_hours is not None
                        and up_hours > _NEXTRUN_UPSTREAM_GRACE_HOURS))
    # Both edges of the grace, not just the upper one (#2950). The record below claims
    # "the upstream IS past its own `next_run`, just not by enough to say so", and
    # `not late_or_dead` alone does not establish the first half: the only lateness bound
    # in this function is the DEPENDENT's (`hours_past` above), so the conjunction was
    # true for every `up_hours <= 0` — a healthy upstream whose completion moved its
    # stamp forward, still hours ahead of the tick. #98 -> #24 printed 3,556 of those
    # between 2026-10-14 and 2026-10-17, each attributing a healthy hold to the grace.
    # The VERDICT is unchanged by this: an upstream ahead of its own stamp never names a
    # chain, and the dependent's own lateness stays the one-period bound's sentence.
    if (status == autonomy.RUN_STATUS_SUCCESS
            and up_hours is not None
            and 0.0 < up_hours <= _NEXTRUN_UPSTREAM_GRACE_HOURS):
        # The grace is what closed the door: the upstream IS past its own `next_run`,
        # just not by enough to say so. One record names both ends of the chain and
        # the minutes, so a reader of the log can see the suppression and the size of
        # it — and no alert, which is the whole point of #2695.
        logger.warning(
            "nextrun stall alarm did not name the chain #%s -> #%s: the upstream's "
            "newest run record is a success and it sits %.0f min past its own "
            "next_run, inside the %.1f h grace",
            task.get("id"), up.get("id"), up_hours * 60.0,
            _NEXTRUN_UPSTREAM_GRACE_HOURS)
    return {
        "upstream_id": up.get("id"),
        "upstream_status": status,
        # The form the alert quotes: one instant, UTC, whole seconds, and a `Z` so
        # a reader cannot mistake it for local wall clock. `_row_vs_records` prints
        # `isoformat()` for its own fields; this one is a new sentence and follows
        # the shape #2417's report asked for.
        "upstream_at": (when.astimezone(_dt.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ") if when else ""),
        "upstream_run": run_id,
        "upstream_hours_past": up_hours,
        "late_or_dead": late_or_dead,
    }


def _next_run_stalled(queue: WorkQueue) -> list[dict]:
    """Tasks of ANY status sitting more than one period past their OWN `next_run`.

    The companion to `_grossly_overdue`, and deliberately not a variant of it:
    the only task state it reads is `next_run` (plus `frequency` to know what a
    period is), so the three filters that make the noisy alarm survivable — skip
    not-due, skip queue-waiting, skip never-run — cannot hide a stall here. Those
    three are the shapes #421 was filed for: #51 41 h and #40 60 h overdue,
    `failure_count: 0`, no run record, because an upstream's run was never
    recorded and that makes the dependent not-due forever.

    It used to read `status` as well, and that is the one thing #1121 had to
    change: line 174 of this file was `if status != "up_next": continue`, which
    dropped `draft`, `paused` and `failed` out of the only mechanism built to
    catch a stall. An unattributed `up_next -> draft` flip is exactly how #68
    (frequency `every-15min`, the fleet's highest-volume task) stopped, and the
    alarm shipped for that purpose reported zero stalls while #68 sat ~50 cycles
    past its own `next_run`. The status is not a reason to skip the scan any
    more; it is the REASON, carried in `status` and in `hold` — `hold_reason`
    returns the status string itself for any non-`up_next` task, so widening the
    scan needed no new reason code.

    Each entry therefore carries the reason dispatch itself would give —
    `hold_reason`, the same function the dispatch loop's "Holding #" line uses —
    so the alert says WHY it has not run rather than only THAT it has not. A task
    with no hold reason was skipped below this loop: a per-task model-server
    health check, or the model-server gate. Since #938 that gate sits BELOW this
    scan rather than above it, so an `up_next` task past its own period with no
    queue row is flagged while the engine is down too, and reads as a task
    nothing holds: dispatch skipped it because the model server was
    unreachable, which is the outage alert's own sentence, and the two arrive
    together.

    The bound is unchanged and strict (> one interval), and it comes from
    `autonomy.next_run_gap`, the same predicate `compute_health` reads, so the
    alert and the fleet report cannot call the same board two different things.
    Since #2342 the predicate still decides which stamped rows are listed, and the
    one row it structurally cannot see — no `next_run` key — is admitted beside it
    and then judged by its run records, because widening the shared bound is what
    #421 was filed against. So each entry also carries what the RECORDS say, from
    `_row_vs_records`: `record_at`, `record_run`, `row_last` and `lost_completion`.
    `past_period` records which of those two routes the row came through — True for
    a row the bound itself admitted — and since #2417 a third route exists, a row the
    bound can measure but has not crossed: `blocked_by`, from `_upstream_block`, is
    non-empty exactly when the row is past its own `next_run` and the upstream named
    in its `depends_on` is late or has a newest run record that is not a `success`.
    Nothing about the bound moved; this is a second question, asked only of rows that
    already name an upstream, that lets a chain be reported the hour it goes late
    instead of a day later. `hours` is None exactly on the admitted-without-a-stamp shape,
    and the return drops such an entry when its records are empty too — a row with
    no stamps and no record has never run, and is the noise this alarm's budget is
    spent avoiding.

    The instant is `autonomy._utcnow()`, like the alarm beside it, so one clock
    pin moves both.

    Each entry carries `parked`: the declaration from that task's OWN file, via
    `autonomy.parked_declaration`, or '' when it says nothing. The scan does not
    filter on it, and that is deliberate twice over. First, `_nextrun_alert_message`
    is the surface a person reads, so that is where a declared park is taken out and
    counted — see its docstring for why the count has to survive; second, the list
    here is what a reader of the mechanism, not of the alert, consults, and a task
    silently missing from it would be indistinguishable from one the gap predicate
    never looked at.

    The declaration is asked of the file and never inferred from `status`, because a
    status is precisely what #1121 proved can flip by accident: #68, parked on
    purpose by Alan on 2026-09-17 with the ruling that nobody restore it or re-file
    an item to re-enable it, sits `draft` exactly like an accidental flip does. A
    suppression derived from `draft` would have undone that widening in one line, a
    few above the test that pins it."""
    from app import autonomy
    now = autonomy._utcnow()
    resolution = autonomy.dependency_resolution_set()
    active = _active_task_ids(queue)
    stalled = []
    for t in resolution:
        gap = autonomy.next_run_gap(t, now=now)
        # Two doors. The first is the predicate, untouched by #2342. The second is the
        # row with NO `next_run` key at all: such a row fails `past_next_run` on its
        # own `hours_past_next is not None` guard, so #96 was invisible HERE exactly as
        # it is to the due-ness alarm, and the only thing that knew it had run is its
        # own record. `gap` is not consulted for it — there is nothing to measure.
        # `next_run_gap` itself is unchanged (clause 5): this is a reporter noticing a
        # shape the predicate is built not to see, not a widening of the bound.
        #
        # Cost stays bounded because the door is keyed on a missing field, not on a
        # slow one: `grep -L '^next_run:' ~/obsidian/autonomy/[0-9]*-*.md` names #96
        # alone fleet-wide, so the record walk below runs for the flagged rows plus
        # that one, not for all 38. Whether the unstamped row is reportable is then
        # answered by the records themselves, in the filter on the return.
        # Three doors. The first is the predicate, untouched by #2342 and by #2417.
        # The second is the row with NO `next_run` key at all: such a row fails
        # `past_next_run` on its own `hours_past_next is not None` guard, so #96 was
        # invisible HERE exactly as it is to the due-ness alarm, and the only thing
        # that knew it had run is its own record. `gap` is not consulted for it —
        # there is nothing to measure. `next_run_gap` itself is unchanged (clause 5
        # of #2342): these are a reporter noticing shapes the predicate is built not
        # to see, not a widening of the bound.
        #
        # The third is #2417's, and it is the only one that admits a row the bound
        # can measure: a dependent that is PAST its own `next_run` but by less than a
        # period, whose upstream is demonstrably late or dead. It goes through
        # `_upstream_block`, which is what keeps the general bound intact — a
        # dependency-held row 3 h late behind a HEALTHY upstream still waits its
        # period, or this alarm becomes the noise #2342 removed.
        #
        # Cost stays bounded because door two is keyed on a missing field and door
        # three on a stamp that is already behind, neither of which a healthy board
        # has: `grep -L '^next_run:' ~/obsidian/autonomy/[0-9]*-*.md` names #96 alone
        # fleet-wide, and completing a run moves `next_run` forward, so the record
        # walks below and in `_upstream_block` run for the flagged rows and nothing
        # else, not for all 38. Whether an unstamped row is reportable is then
        # answered by the records themselves, in the filter on the return.
        blocked = _upstream_block(t, gap, resolution, now)
        if not (gap["past_next_run"]
                or autonomy._parse_iso(t.get("next_run")) is None
                or blocked.get("late_or_dead", False)):
            continue
        stalled.append({
            "id": t.get("id"),
            "name": t.get("name"),
            "status": str(t.get("status", "")).strip(),
            "hours": gap["hours_past_next_run"],
            "gap_ratio": gap["gap_ratio"],
            # Which door this row came through, kept as its own field rather than
            # inferred from `hours`: `gap["past_next_run"]` is the shared one-period
            # predicate, and `_nextrun_alert_message` has to count the rows it held
            # separately from the ones #2417's door reached, or the head sentence
            # claims a bound it did not apply.
            "past_period": bool(gap["past_next_run"]),
            "hold": autonomy.hold_reason(t, resolution, now=now),
            "queued": str(t.get("id")) in active,
            "parked": autonomy.parked_declaration(t),
            # What the named upstream is doing, {} when there is nothing to name.
            "blocked_by": blocked,
            # Read AFTER the shortlist, so the record walk costs what it claims to
            # cost in `_row_vs_records`: the flagged rows, 0 or 1 on a healthy board,
            # not all 38 of them every tick.
            **_row_vs_records(t),
        })
    return [e for e in stalled
            if e["hours"] is not None or e["record_at"]]


#: How long ago a lost occurrence may be and still be recorded. `_window_occurrences`
#: already yields only ±2 local days, so this is a filter on what that list contains,
#: not a scan cost — and it is what stops a tick whose clock was wound back from
#: re-litigating a month of windows.
_WINDOW_LOSS_LOOKBACK_SECONDS = 2 * 86400


def _window_losses(queue: WorkQueue) -> list[dict]:
    """Record every window occurrence that closed unrun with claims held.

    The durable half of #2552. On 2026-10-10 the operator pause was held from
    2026-10-09T17:57:57Z to 2026-10-10T14:49:34Z; #30, which declares
    `preferred_hours: []`, ran 11.5 min after the release on the queue row it had
    accumulated during the hold, while the three windowed jobs of the nightly
    reflection chain — #42 `[22,23,0,1,2,3,4]`, #39 `[1,2,3,4]`, #40 `[2,3,4]` —
    lost the night outright, because `preferred_hours` is an ENQUEUE gate and a
    hold on claims therefore starves a windowed task's whole occurrence while it
    costs a windowless task nothing. No run record, no queue row, no
    `memory/learnings/2026-10-10.md`; the chain resumed the next night and nothing
    anywhere said a cycle had been dropped rather than delayed.

    Why the writer sits here and not beside the gate it concerns: `app/autonomy.py`
    could not see the block at all (`git grep -n operator_paused -- app/autonomy.py`
    was empty before this item), and the dispatch path is precisely what the block
    is refusing. This module is the reader that already runs while claims are held —
    that is #1682's whole reason for this seat — and `operator_pause_state` is the
    one accessor both surfaces agree on (#1550).

    Two conditions, both read from the row rather than inferred:

    * the occurrence is CLOSED (`close <= now`), its close is inside
      `_WINDOW_LOSS_LOOKBACK_SECONDS`, and the pause's `since` is at or before the
      occurrence's OPEN — the block held across the entire window, so the task never
      had one unblocked minute in it. A hold taken mid-window is not claimed as the
      cause: the task had unblocked time and something else may have held it.
    * the completion stamp is older than the opening. The stamp alone, not the run
      record: this is a reader on the pool's 60 s tick, and the backstop that reads
      the disk lives in `_is_task_due`, where a stamp reverted by a vault sweep
      spends its false grant and is refused (#1296).

    The key carries the occurrence's opening instant because `watermarks` is
    `PRIMARY KEY (source, key)` — one row updated in place — and the question this
    item was filed with ("how many cycles did this lose?") is unanswerable if a
    second loss overwrites the first. So each lost occurrence is its own row.

    Never raises. A ledger that could not be written leaves the fleet exactly where
    it already is, and an exception out of here would skip the two stall alarms
    that share this tick — an alarm cannot report its own silencing (#1682).
    """
    written: list[dict] = []
    try:
        import json

        from app import autonomy
        from workers.pool import operator_pause_state

        now = autonomy._utcnow()
        state = operator_pause_state(queue)
        if not state["paused"]:
            return written          # the common case, and it costs one indexed read
        since = _parse_iso_safe(state["since"])
        if since is None:
            # A hold with no durable instant (`automod`'s, which is deliberately
            # never persisted) is not evidence that a block covered a window, and
            # inventing one is exactly the false authorisation this ledger must
            # never become.
            return written

        for task in autonomy.dependency_resolution_set():
            if str(task.get("status", "")).strip() != "up_next":
                continue
            if autonomy.parked_declaration(task):
                continue            # it declares itself parked on purpose
            hours = autonomy._window_hours(task)
            if not hours:
                continue
            stamp = task.get("last_run")
            last_run = autonomy._parse_iso(stamp)
            if stamp and last_run is None:
                continue            # an unparseable stamp is not evidence of a loss
            for occ_open, occ_close in autonomy._window_occurrences(hours, now):
                if occ_close > now:
                    continue
                if (now - occ_close).total_seconds() > _WINDOW_LOSS_LOOKBACK_SECONDS:
                    continue
                if since > occ_open:
                    continue        # the hold began inside the window: not its doing
                if last_run is not None and last_run >= occ_open:
                    continue        # it ran in that occurrence
                key = autonomy.window_loss_key(task.get("id"), occ_open)
                if queue.wm_get(autonomy.WINDOW_LOSS_WM_SOURCE, key) is not None:
                    continue        # already recorded; a loss is recorded once
                row = {
                    "task_id": task.get("id"),
                    "name": task.get("name"),
                    "window_hours": autonomy._hour_windows(hours),
                    "window_open": occ_open.isoformat(),
                    "window_close": occ_close.isoformat(),
                    "pause_since": state["since"],
                    "recorded_at": now.isoformat(),
                }
                queue.wm_set(autonomy.WINDOW_LOSS_WM_SOURCE, key,
                             json.dumps(row))
                written.append(row)
                logger.warning(
                    "preferred_hours occurrence lost: task #%s (%s) window %s "
                    "[%s -> %s] closed with the task unrun while the operator pool "
                    "was held since %s — recorded as %s, and one out-of-window "
                    "catch-up is now authorised for it",
                    task.get("id"), task.get("name"), row["window_hours"],
                    row["window_open"], row["window_close"], row["pause_since"],
                    key)
    except Exception as e:
        logger.error("window-loss ledger failed: %s", e, exc_info=True)
    return written


def _parked_note(entry: dict) -> str:
    """The declaration carried on a flagged entry, '' if its file declares nothing."""
    return str(entry.get("parked") or "").strip()


def _nextrun_alert_message(stalled: list[dict]) -> str:
    """Alert text for the scan above, describing the statuses it actually found.

    The count line used to read `N up_next task(s)` unconditionally, which was
    only true while the scan filtered on that status. Now it names the statuses
    in the flagged set: an alert that misdescribes its own contents is the
    failure this item is about, because a reader who knows #68 is `draft` and
    reads "up_next task(s)" concludes the message is about some other task and
    drops it. When every flagged task is `up_next` — late against its own
    `next_run` and held by nothing — the wording is exactly what it was before
    the widening, so a reader of the old alert reads the same sentence.

    A task whose file declares it parked is then taken OUT of that list, and the
    asymmetry is why the count must ride along rather than the entry quietly
    disappearing: suppressing a declared park removes a named task, so silence after
    the suppression is indistinguishable from the silence #1121 was filed about. Zero
    used to mean nothing was late; it can now also mean every late task talked itself
    out of the alert. A count is the cheapest marker that keeps "nothing is stalled"
    and "everything stalled was parked" two different sentences, so it is appended
    even when nothing un-suppressed is left — and the caller alerts on the scan's
    list, not on this string's arithmetic, which is what lets that case be said at
    all. The names behind the count stay in the task files: a park is a decision
    somebody wrote down, and the reader needs proof the scan looked, not a rerun of
    a ruling they made.

    The head carries a second count since #2342, for the entries the predicate cannot
    measure at all (no `next_run` key), and the first count covers only the rows it
    did measure. That split is the same rule as the status one two paragraphs up: a
    line that says "N tasks more than one period past their own next_run" may only
    count tasks that are. On a board where every flagged row has its stamps — the
    shape every test of the legacy alarm seeds — the string is byte-identical.

    #2417 adds a third count for the same reason, not a decoration. Its door admits
    rows that are past their own `next_run` by LESS than a period, so once a single
    one is present the first count would be naming tasks the bound never crossed
    under the very clause that says "more than one period" — the misdescribing-alert
    defect again, one door over. The counts are read off `past_period`, which is the
    bound's own verdict, and each new clause says what its rows actually are."""
    suppressed = [e for e in stalled if _parked_note(e)]
    late = [e for e in stalled if not _parked_note(e)]
    # The head sentence claims "more than one period past their own next_run", and a
    # row with no `next_run` is not that — it is the row with nothing to measure
    # against. Counting it in that clause would reproduce, one level up, the exact
    # defect #1121 was filed about: an alert that misdescribes its own contents. So
    # the count covers the measured entries and the unstamped ones get their own
    # clause, which also keeps the legacy string byte-identical on a board where
    # every flagged row has its stamps.
    measured = [e for e in late if e.get("hours") is not None
                and e.get("past_period")]
    # #2417's rows: measurable, past their stamp, but inside the period. They are
    # here because an upstream was caught late or dead, and the clause below says so
    # rather than lending them the sentence above.
    short = [e for e in late if e.get("hours") is not None
             and not e.get("past_period")]
    unstamped = [e for e in late if e.get("hours") is None]
    statuses = sorted({str(e.get("status") or "unknown") for e in late})
    # `up_next task(s)` alone reproduces the pre-widening string byte for byte,
    # so anything that matches on the old alert text keeps matching.
    scope = f"{'/'.join(statuses)} task(s)" if late else "task(s)"
    lines = "; ".join(_stall_line(e) for e in late[:15])
    head = (f"{len(measured)} {scope} more than one period past "
            f"their own next_run, which the due-ness stall alarm cannot see")
    if short:
        head += (f" | {len(short)} task(s) past their own next_run by less than one "
                 f"period, which that bound cannot reach, held behind an upstream "
                 f"that is itself past its next_run or whose newest run record is "
                 f"not a success")
    if unstamped:
        head += (f" | {len(unstamped)} task(s) whose row carries no next_run stamp "
                 f"at all, which that predicate cannot see either")
    tail = (f" | {len(suppressed)} task(s) suppressed as declared parked in "
            f"their own file (`parked:`) and not counted above"
            if suppressed else "")
    return f"{head}: {lines}{tail}" if lines else f"{head}{tail}"


def _stall_line(entry: dict) -> str:
    """One flagged task, in the sentence the alert actually shows.

    Three shapes, chosen by what the scan could prove rather than by what it wishes.
    The middle one is the whole point of #2342: the row being late AND a run record
    having completed after the row's own `last_run` is a BOOKKEEPING miss, and
    reporting that as a dispatch failure costs the same attention as a real stall
    while teaching the reader to ignore this alarm.

    `hours` is None on the unstamped shape only, so the two measured branches never
    guard their `:.1f`; a caller that handed an unstamped entry with an elapsed
    figure attached would raise here rather than print `Noneh`.

    Since #2417 the third shape also carries `_blocked_note` — the upstream's id and
    what that upstream's own newest run record says — because a chain going late is
    one event and the reader needs its cause in the same breath as its symptom. The
    other two shapes deliberately do NOT gain it: the middle one's whole finding is
    that the row's bookkeeping, not the chain, is what is wrong, and the unstamped
    one's is that the row carries no stamps at all. Naming a second cause beside
    either would put two explanations in one line where the evidence supports one,
    and the reader would have to decide which the alarm meant."""
    if entry.get("hours") is None:
        missing = ("no next_run and no last_run" if not entry.get("row_last")
                   else "no next_run")
        said = (f"run record {entry['record_run']} says it succeeded at "
                f"{entry['record_at']}" if entry.get("record_at")
                else "its run records answer nothing")
        return (
            f"#{entry['id']} ({entry['name']}) has {missing} on its row while {said} "
            f"— a row with no stamps can never satisfy the past_next_run predicate, "
            f"so no other alarm can see it")
    if entry.get("lost_completion"):
        return (
            f"#{entry['id']} ({entry['name']}) row bookkeeping lost a completion: "
            f"run record {entry.get('record_run') or '?'} succeeded at "
            f"{entry.get('record_at')}, after the row's last_run "
            f"{entry.get('row_last') or '(absent)'}, and the row is still "
            f"{entry['hours']:.1f}h past its next_run — that elapsed figure is "
            f"measured from a stamp the task's own records contradict")
    return (f"#{entry['id']} ({entry['name']}) is {entry['hours']:.1f}h past its "
            f"next_run — {_hold_note(entry)}{_blocked_note(entry)}")


def _hold_note(entry: dict) -> str:
    """Why a flagged task has not dispatched, in dispatch's own words.

    Reached only for an entry the run records do NOT contradict. Since #2342 the
    caller asks `_row_vs_records` first, because this function reasons from the row
    alone, and a row that lost a completion turns 'dispatch did not enqueue it' into
    a claim the run record on disk refutes — the sentence that fired at 2026-10-06
    22:00:28 about #79, whose run had finished at 05:17:08 that same morning.
    """
    if entry["hold"]:
        return f"held: {entry['hold']}"
    if entry["queued"]:
        return "nothing holds it; a queue row is already waiting for capacity"
    return "nothing holds it; dispatch did not enqueue it"


def _blocked_note(entry: dict) -> str:
    """The chain behind a late dependent, named: which upstream, and what it did.

    #2417. `held: waiting on #38` is dispatch's own reason, and #1739 made sure it is
    not said over a completed cycle — but it has never been able to say anything
    about #38 itself, and the reader of the 2026-10-08 alert needed exactly that: the
    upstream had DIED at 05:04:09Z, which is the fact that turns "three tasks are
    late" into "one run failed and nobody retried it". Handed the first sentence
    without the second, a reader goes and looks at three schedulers instead of one
    run record.

    Three branches, chosen by the strongest fact `_upstream_block` could prove:

    * a newest record that is not a `success` — `which failed at
      2026-10-08T05:04:09Z`, the shape #2417's report asked for, with the upstream's
      own lateness appended when it has one, since "failed and not retried" is the
      whole news;
    * a `success` record and a stamp behind it by more than the grace — nothing
      upstream is broken except its clock, so the lateness IS the finding. Since #2695
      this branch cannot be reached by a run that is merely still going: the minutes
      have to exceed `_NEXTRUN_UPSTREAM_GRACE_HOURS` first;
    * no readable record at all, which is not the same thing as a success and says
      so rather than going quiet.

    It returns '' whenever `late_or_dead` is False, so on a healthy chain the line is
    byte-identical to the pre-#2417 sentence, and whenever the entry came through a
    door with no upstream to name. The clause rides AFTER `_hold_note`'s rather than
    replacing it: `waiting on #38` remains why dispatch refused, and this is why that
    refusal now deserves a person rather than another day of waiting.
    """
    from app import autonomy
    blocked = entry.get("blocked_by") or {}
    if not blocked.get("late_or_dead"):
        return ""
    up_id = blocked.get("upstream_id")
    status = str(blocked.get("upstream_status") or "")
    at = str(blocked.get("upstream_at") or "")
    hours = blocked.get("upstream_hours_past")
    late_tail = (f" and is {hours:.1f}h past its own next_run"
                 if hours is not None and hours > 0 else "")
    if status and status != autonomy.RUN_STATUS_SUCCESS:
        said = (f"which {status} at {at}" if at
                else f"which last reported `{status}` with no completed stamp")
        return f", blocked-by #{up_id}, {said}{late_tail}"
    if status == autonomy.RUN_STATUS_SUCCESS:
        # A `success` upstream can only have opened this door through its own stamp,
        # so there is always an `hours` to print here.
        return (f", blocked-by #{up_id}, which is "
                f"{hours:.1f}h past its own next_run")
    return (f", blocked-by #{up_id}, whose run records report no success"
            f"{late_tail}")


def _queue_starving(queue: WorkQueue, max_duration: int) -> float:
    """Age in seconds of the oldest claimable queued item, if it is older than
    3x max_duration. Catches the opposite failure: dispatch fine, workers dead.

    **Every source, and the oldest row rather than the newest 500.** This read used
    to be `queue.list_items(source=NAME, limit=500)`, and asking one question about
    one source bought two blind spots out of it. `NAME` is `scheduled-task`, so a
    backlog of `owed-check`, `session-distill`, `board-steward`, `bench-mine` or
    `frontend-probe-canary` rows was invisible to the alarm that exists to notice a
    backlog: on 2026-10-10 the queue held 31 claimable rows at attempts=0, oldest
    23h37m, every one of them in the 65-80 priority bands and none of them
    `scheduled-task`, and this function returned 0.0 for every hour they waited. And
    `list_items` ends `ORDER BY enqueued_at DESC LIMIT ?`, so once more than 500 rows
    are queued the rows it drops are the OLDEST ones — the figure under-reports by
    exactly as much as the backlog has grown, which is the one direction that matters.

    `WorkQueue.pending_wait_by_source` is the read that already had this right:
    source-agnostic, no `since` window, no LIMIT, and the same two exclusions this
    loop made by hand — a row whose `not_before` is still in the future is waiting by
    design (`mark_failed`'s back-off, not pool starvation), and a row whose
    `enqueued_at` cannot be parsed contributes no age. Both are stated on that
    method, so they are not restated as rules here; the max over its per-source
    figures is all this detector needs, and the alarm and the Mission Control panel
    cannot disagree about how long the fleet has been waiting.

    What has not moved is the threshold and the shape of the answer: 3x
    `max_duration`, and 0.0 when nothing qualifies, which the alert's wording and the
    streak logic in `tick()` both depend on. A read that raises still returns 0.0 —
    an unreadable queue is not evidence of a stall, and a watchdog that alarmed on its
    own exception would spend a person's cooldown on a lock.
    """
    try:
        waits = queue.pending_wait_by_source()
    except Exception:
        return 0.0
    oldest = max((entry["max_wait_seconds"] for entry in waits.values()),
                 default=0.0)
    return oldest if oldest > 3 * max_duration else 0.0


def _oldest_wait(queue: WorkQueue) -> tuple[float, str | None]:
    """The longest wait on the queue, and WHICH SOURCE is serving it.

    The same read `_queue_starving` makes, with the key it discards. `waits` is keyed by
    source, so carrying the name costs nothing, and one line of the produced alert is the
    reason it has to be carried: a post reading `oldest claimable queue item is 915 min
    old` names nobody, so a reader cannot tell a queue that stopped moving from a
    low-priority stream sitting behind a busy high-priority one — and those two want
    opposite actions. Same exception rule: an unreadable queue is `(0.0, None)`, which
    never crosses the threshold, so a watchdog never alarms on its own failure to read.
    """
    try:
        waits = queue.pending_wait_by_source()
    except Exception:
        return 0.0, None
    if not waits:
        return 0.0, None
    source, entry = max(waits.items(), key=lambda kv: kv[1]["max_wait_seconds"])
    return float(entry["max_wait_seconds"]), source


def _outranked_clause(
    queue: WorkQueue, starving: float, starving_source: str | None,
    max_duration: int,
) -> str:
    """A clause naming the work being claimed ahead of the waiting row, or "".

    Empty string UNLESS the fleet is demonstrably claiming: some source other than the
    waiting one has rows in `claimed` or `running` stamped inside the same
    3 x `max_duration` window the wait is weighed against. That is what "the pool is
    taking work" means, read off `queue.claimed_at` — the column `claim_next` stamps on
    every claim — rather than inferred from run rows, which only appear when a job
    finishes. The waiting source is excluded from the count on purpose: an `owed-check`
    row claiming work while its own older sibling waits is the anomaly (the pool takes
    the best-ranked claimable row), not evidence that someone is ahead of it.

    The alarm drops the word "stalled" whenever this clause is non-empty, because on that
    board the word is false. Measured on this box 2026-10-14 with the command at the head
    of `tests/test_fleet_watchdog.py`'s #2692 section — which pipes through
    `grep -F fleet_watchdog:` because each post is written TWICE, once as this module's
    ERROR line and once as a `lloyd-server: discord_alert` echo, so a bare `grep -c`
    returns 44 lines for 22 posts — 22 posts of this alarm over the eleven rotated
    `server.err*` files, 18 of them age-only and 4 carrying the #1550 pause clause. The
    18 run from 2026-10-11 16:17:42 (185 min) to 2026-10-14 02:05:42 (196 min), and both
    of those instants are the endpoints of all 22 posts too: 892, 570, 456 then 185
    minutes across 2026-10-13 is a head age that FALLS, which is only possible while the
    head is being claimed and replaced. A stalled pool's head age rises. What keeps its
    bytes exactly is the case this returns "" for: a queue nothing has claimed, whose
    sentence `agent-services/guardian/detect.py:243` cites as the chronic signature and
    which
    `tests/test_autonomy_scheduler.py::test_a_starving_queue_with_no_pause_alerts_byte_identically`
    pins.

    What makes the clause decidable rather than a judgement call is that the claim count
    is a fact about the queue. On this box 2026-10-14, `select source, min(priority),
    max(priority), count(*) from queue group by source` puts ELEVEN sources wholly above
    `owed-check`'s 72 — every one of their rows beats it, because max(priority) is below 72
    for all eleven and NO source spans the 72 line at all (so this is not the min()
    projection outranking on one lucky row), and no row in the store sits at priority 1 or 2
    (the `priority=1` / `priority=2` arguments at `workers/sources/autotriage.py:1165` and
    `workers/sources/owed_check.py:355` are `run_prompt_in_session`'s model-request
    priority, not a queue row's, so they are not in this ranking):
    `scheduled-task` 20-50, `autocode` 40, `autotriage` 40-55, `youtube-digest` 45,
    `autoresearch` 60, `arch-review` 62, `backlog-cluster` 65, `board-steward` 68, and
    `automod-regression`, `deep-research` and `session-distill` at 70. What actually makes
    the wait is the aggregate rate, not any one of them: 426 claims from 14 sources between
    2026-10-13T04:00 and 2026-10-14T04:00, 384 of them not `owed-check`'s own. A source's
    rows inherit its configured priority at enqueue (`int(src_cfg.get("priority", ...))`),
    so this ranking is the one `sources.yaml` holds today, not a property of these rows.

    Computed on the alert path only, for the same reason `_operator_pause_clause` is: the
    read is `SELECT source, COUNT(*) ... WHERE claimed_at >= ? GROUP BY source`, and
    `claimed_at` carries no index of its own, so the cost is a scan of the `queue` table —
    9,520 rows on this box 2026-10-14, which at alert-path frequency is nothing, and which
    is why the read is not hoisted into the tick's ordinary path.
    """
    if starving < 3 * max_duration or starving_source is None:
        return ""
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    since = (now - _dt.timedelta(seconds=3 * max_duration)).isoformat()
    try:
        claims = queue.claims_by_source(since, exclude_source=starving_source)
    except Exception:
        # A claims read that will not answer is not evidence the pool is busy. The
        # sentence falls back to the one that accuses nothing beyond the wait itself,
        # and the alarm is never silenced: this branch changes the words, not the post.
        return ""
    if not claims:
        return ""
    leader, count = claims[0]
    ahead = f"{count} {leader} row" + ("s" if count != 1 else "")
    others = len(claims) - 1
    if others:
        ahead += f", plus work from {others} other source" + ("s" if others != 1 else "")
    return (
        f" — priority starvation: it is outranked by higher-priority work,"
        f" {ahead} claimed inside {3 * max_duration // 60} min"
    )


def _operator_pause_clause(queue: WorkQueue) -> str:
    """A clause naming an engaged operator pause and how long it has been held.

    Empty string when no operator pause is held, which is the overwhelmingly
    common case and must leave the alert byte-identical to what it said before
    #1550 — `agent-services/guardian/detect.py` greps the surrounding sentence, and
    an accusation that leaked into the unpaused case would both break that watch and
    blame a person who touched nothing.

    Why this exists: an operator pause is durable and has no TTL, so a hold taken
    by hand in Mission Control stops claims for as long as nobody resumes it. On
    2026-09-24 one lasted 16.5 h straight through the 22:00-04:00 nightly window —
    queue row 510 enqueued 05:00:19Z and was claimed at 18:44:52.78Z, one second
    after the resume line at 18:44:51.495 — and every alert in that stretch read
    `autonomy scheduler may be stalled: oldest claimable queue item is 637 min old`
    (then 791 min). The scheduler was healthy. The pause was the entire cause, and
    nothing said so.

    The verdict is read from the persisted `(_pool, operator_paused)` watermark
    through `workers.pool.operator_pause_state`, the same read `WorkerPool` uses to
    re-engage a hold after a restart. It cannot use `get_pool()`: this tick runs in
    the backend process, which after a restart has a pool whose in-memory flags came
    from that row anyway — and during an outage where no pool started at all there
    is no object to ask. The row is the durable fact; memory is a copy of it.
    """
    try:
        from workers.pool import HOLDER_PERSON, operator_pause_state
        state = operator_pause_state(queue)
    except Exception:
        logger.exception("stall alarm: could not read the operator pause; "
                         "alarming without it")
        return ""
    if not state["paused"]:
        return ""
    import datetime as _dt
    held_h = 0.0
    since = state["since"]
    if since:
        taken = _parse_iso_safe(since)
        if taken is not None:
            now = _dt.datetime.now(_dt.timezone.utc)
            held_h = max(0.0, (now - taken).total_seconds() / 3600.0)
    # The kind, from the row the hold journal takes its `holder_kind` from (#2615
    # clause 4). `by operator` alone made a harness's measurement window and a
    # person's deliberate halt read as one sentence, so the reader could not tell
    # "someone is doing this on purpose" from "a process died holding the fleet" —
    # the two things #2498's distinction exists to separate. The journal is the
    # history; this is the live hold, and both now name the same kind.
    holder = state["holder"] or HOLDER_PERSON
    return (f"the worker pool is PAUSED by operator ({holder} hold), held "
            f"{held_h:.1f} h since {since} — claims are held by that pause, not by "
            f"this scheduler; resume it to drain the backlog")


def _parse_iso_safe(value):
    from app import autonomy
    return autonomy._parse_iso(value)


def _last_alert_instant(queue: WorkQueue, key: str, alarm: str):
    """When `alarm` last reached a person, as the FLEET knows it.

    The later of what this process remembers and what the queue's watermark row
    says, because an alert cooldown is a fact about the fleet and `_state` is a fact
    about one process's uptime. That is #1735 for the `next_run` alarm:
    `nextrun_alerted_at` lived only in the module-global dict, so a new process
    satisfied "never alerted" immediately, reached its own `_STALL_NEXTRUN_TICKS`
    streak on the same late board, and re-posted — `~/lloyd-data/logs/server.err` has
    pool starts at 21:10:13 and 22:16:18 on 2026-09-27 with next_run alerts at
    21:14:20 and 22:20:28, 4 min 07 s and 4 min 10 s after each boot, which is exactly
    five of the pool's 60 s scheduler ticks.

    #2662 is the same defect on the due-ness alarm, and it cost 20 posts rather than
    two. Counted on `~/lloyd-data/logs/server.err*` on 2026-10-13, anchored on
    `[ERROR] lloyd-workers.fleet_watchdog: autonomy scheduler may be stalled`: 20 posts
    over 41.9 h (2026-10-11 16:17:42 to 2026-10-13 10:14:02), and ALL 19 consecutive
    gaps fall inside `_STALL_ALERT_INTERVAL_SECONDS` — the largest 4.52 h, so the 6 h
    interval did not hold once in the corpus. Two of them are one continuing stall:
    16:17:42 and 17:05:56 on 10-11, 0.80 h apart, both reading
    `oldest claimable queue item is 185 min old`. No live `_state` can produce that
    pair, because the first post writes the instant and the second needed 6 h.

    `key` is the `_state` key and the watermark key at once, which is why one pair of
    functions serves both alarms: each alarm's instant is stored under the same word
    in both places, and the two rows cannot spend each other's cooldown.

    Why the LATER of two reads rather than the row alone: `_state` keeps the key as
    this process's own cache, so the two can disagree only by one being behind, and
    the later one is the sentence "we already told someone". Reading the row alone
    would let a process that had just alerted, on a queue it cannot read, hand the
    day's silence back and re-post; reading `_state` alone is the defect.

    Why the LATER of two reads rather than the row alone: `_state` keeps the key as
    this process's own cache, so the two can disagree only by one being behind, and
    the later one is the sentence "we already told someone". Reading the row alone
    would let a process that had just alerted, on a queue it cannot read, hand the
    day's silence back and re-post; reading `_state` alone is the defect.

    Why the instant is the row's VALUE and not its `updated_at`, which is where
    `workers/pool.operator_pause_state` reads its clock: that row's value is a
    boolean, so `updated_at` is the only place the instant can go, and #1550 put it
    there so the alert and the Mission Control panel quote one accessor. Here the
    instant IS the payload — `wm_set` writes value and stamp in the same statement,
    so they cannot disagree — and storing it as the value is what keeps the row
    aged through the public API. A row whose `updated_at` is the only clock can only
    be made to look a day old with raw SQL against `workers.db`, which is precisely
    the workaround `wm_updated_at`'s own docstring names as the reason it exists.

    An unreadable or unparseable row is read as no durable knowledge, loudly, and
    the in-process answer stands: this runs inside the scheduler loop, and a
    watermark this tick cannot read is not a reason to stop watching the fleet.
    """
    in_process = _state.get(key)
    try:
        raw = queue.wm_get(NAME, key)
    except Exception:
        logger.exception("%s stall alarm: could not read the persisted alert "
                         "instant; falling back to this process's own record", alarm)
        return in_process
    persisted = _parse_iso_safe(raw)
    if raw and persisted is None:
        logger.warning("%s stall alarm: the persisted alert instant %r under "
                       "(%s, %s) does not parse; ignoring it",
                       alarm, raw, NAME, key)
    if persisted is None or (in_process is not None and in_process > persisted):
        return in_process
    return persisted


def _record_alert_instant(queue: WorkQueue, key: str, when, alarm: str) -> None:
    """Say that `alarm` reached a person, here and durably.

    Both halves of one fact, written on the one route that reaches a person — a
    scan that found nothing but declared parks never gets here (#1661), so a park
    cannot spend the cooldown. The write is guarded and the alert still goes out if
    it fails: an alarm that could not record itself is still an alarm, and the
    failure it produces is the old one — a later process may re-post — rather than a
    silenced one.

    Called on the due-ness route too since #2662, under `_STALL_WM_KEY`. The ordering
    that matters is the caller's: record first, dispatch second, so a send that fails
    still spends the cooldown rather than handing the next process a fresh window
    (#1704).
    """
    _state[key] = when
    try:
        queue.wm_set(NAME, key, when.isoformat())
    except Exception:
        logger.exception("%s stall alarm: alert posted but its instant could not be "
                         "persisted; a restart inside the cooldown may re-post it",
                         alarm)


async def _alert(message: str) -> None:
    try:
        from app.discord_notify import discord_alert
        await discord_alert(message)
    except Exception as e:
        logger.error("alert dispatch failed: %s", e)


def _utc_day(now=None) -> str:
    """Today's UTC day as `YYYY-MM-DD` — the key `runs.started_at` is grouped by."""
    import datetime as _dt

    return (now or _dt.datetime.now(_dt.timezone.utc)).date().isoformat()


def _day_minus_one(day: str) -> str:
    """The calendar day before `day`, which is also what "adjacent" means here."""
    import datetime as _dt

    return (_dt.date.fromisoformat(day) - _dt.timedelta(days=1)).isoformat()


def _host_boot_instant():
    """The HOST's boot instant in UTC, or None when this machine will not say.

    The anchor is the machine's, never this process's: the watchdog restarts ~12x a
    day (#1735, #2662), so a process-relative "since boot" would reset a dozen times
    a day and the figure it prints would describe the restart rather than the pool.
    One module-level helper is the census's only route to it, which is also what lets
    a test hold the anchor still — the instant moves every tick by construction, so a
    node that read the real one could not assert a count over it.

    `/proc/uptime`'s first field is seconds since boot, so boot is `now - uptime`,
    truncated to whole seconds because that is finer than the counter is read; on
    this box the result agreed with `uptime -s` to the second (#2956). Reading the
    file rather than shelling out is deliberate: this is called from the 60 s
    surveillance tick, where a fork per minute buys nothing.

    Any failure to read or parse — no `/proc/uptime` at all, a short file, a
    non-numeric field — returns None rather than raising: the census that calls this
    has to print the line it prints today instead of dying on the pool's scheduler
    loop beside the two stall alarms.
    """
    import datetime as _dt

    try:
        with open(_PROC_UPTIME_PATH, encoding="utf-8") as fh:
            uptime_seconds = float(fh.read().split()[0])
    except Exception:
        logger.debug("failure-throughput census: %s could not be read, so the "
                     "since-boot figure is being left off the line",
                     _PROC_UPTIME_PATH)
        return None
    now = _dt.datetime.now(_dt.timezone.utc)
    return (now - _dt.timedelta(seconds=uptime_seconds)).replace(microsecond=0)


def _throughput_census(queue: WorkQueue, now=None) -> dict | None:
    """The newest COMPLETE UTC day's failed runs, against its own predecessor.

    What the reading is, stated exactly because every number on the line depends on
    it: a day is COMPLETE when it is strictly older than today's UTC day, so the
    census never compares a whole day against a partial one; the pair must be
    calendar-ADJACENT, so a day with no rows at all cannot be silently skipped into a
    comparison across a gap; and the day is the leading 10 characters of
    `runs.started_at`, which this store writes in UTC
    (`2026-10-17T09:24:16.250412+00:00`) — the same reading every count quoted on
    #2890 used, hazards and all, including that a row stamped in a local zone would
    be grouped by its own date digits.

    TWO rules read that one pair (#2890 shipped the first, #2899 the second), and
    either one alone is enough to post: the day-over-day RATIO, `newer >=
    _THROUGHPUT_RATIO * older`, which catches a day that is loud RELATIVE to its
    neighbour; and the absolute FLOOR, `newer >= _THROUGHPUT_FLOOR`, which catches a
    day that is loud on its own terms whatever the neighbour looked like — the case
    the ratio structurally cannot see, because two bad days in a row make the second
    one look ordinary. Both read the same pair, so the floor leg inherits every shape
    above rather than loosening any of them: a single-day store with 137 rows is
    still not a comparison and still says nothing. The point of the second leg is a
    different threshold, not a different query.

    NEITHER rule is asked what it is counting its failures against, and that is the
    defect #2956 is about: the 2026-10-17 line fired the floor on 45 failures, 43 of
    them inside one ~29-minute window that straddled both the UTC midnight boundary
    and the instant the host came up, so a day that was quiet for the other ~23 hours
    reached a person as an incident. The trigger stays exactly as #2899 priced it —
    this reads the same day again, from the rows it already fetched, and reports three
    more numbers with it: `boot`, the host's boot instant from `_host_boot_instant()`
    (None when the machine will not say, in which case the two counts beside it are
    None too and the line keeps the shape it has today); `since_boot`, how many of
    THIS day's failed rows start at or after that instant; and
    `minutes_since_boot`, the whole minutes from the instant to `now`, the clock
    reading this census took. Those three are a reading, not a rule: no threshold
    reads them, no leg fires on them, and `_THROUGHPUT_FLOOR` is untouched. They are
    what lets a reader tell "43 failures in a boot ramp" from "45 failures spread
    across a day of pressure" — the first is the machine coming up, the second is the
    pool failing now.

    `now` is the census's clock, defaulted to wall clock: the day gate and
    `minutes_since_boot` are both read from it, so a test that pins it pins the whole
    reading instead of racing the tick against midnight.

    Returns None — nothing to say, and no line — for each of the shapes the store can
    hand back that is not a comparison: no complete day at all, a newest complete day
    whose predecessor has no rows (an absent predecessor is NOT read as a zero,
    because a day the pool did not run is not a day the pool survived), a count that
    trips NEITHER leg, or a store it cannot open. It never raises: this runs on the
    pool's scheduler tick, beside two stall alarms, and #1682's whole point is that a
    fault in one reader here may not be the thing that stops the others.

    The connection is closed explicitly rather than left to `with`, unlike the
    precedent read at `app/failure_ledger.py:615`: that one runs once per ledger
    build, this one runs on every 60 s tick, so a handle that only commits on exit
    would be a descriptor leaked once a minute for the life of the pool.
    """
    if now is None:
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
    conn = None
    try:
        today = _utc_day(now)
        conn = queue._connect()
        complete = [r[0] for r in conn.execute(
            "SELECT DISTINCT substr(started_at, 1, 10) FROM runs "
            "WHERE substr(started_at, 1, 10) < ? ORDER BY 1", (today,))]
        if not complete:
            return None
        day = complete[-1]
        prev = _day_minus_one(day)
        if prev not in complete:
            return None
        counts = dict(conn.execute(
            "SELECT substr(started_at, 1, 10), count(*) FROM runs "
            "WHERE status = 'failed' GROUP BY 1"))
        newer, older = counts.get(day, 0), counts.get(prev, 0)
        # Two independent rules over one reading, and WHICH one fired travels with
        # the result: `_throughput_alert_message` has to say so, because a line that
        # printed the ratio on a floor-triggered 2.0x day would send a reader to
        # audit a rule that was never the reason it exists.
        by_ratio = newer >= _THROUGHPUT_RATIO * max(older, 1)
        by_floor = newer >= _THROUGHPUT_FLOOR
        if not (by_ratio or by_floor):
            return None
        rows = conn.execute(
            "SELECT summary, started_at FROM runs WHERE status = 'failed' "
            "AND substr(started_at, 1, 10) = ?", (day,)).fetchall()
        # Read after the legs, so a tick that trips neither — the ordinary case, every
        # minute of the day — never opens the uptime file for a figure nobody prints.
        boot = _host_boot_instant()
    except Exception:
        logger.exception("failure-throughput census: the runs store could not be "
                        "read; saying nothing rather than guessing")
        return None
    finally:
        if conn is not None:
            conn.close()

    # Tallied in Python, over the day's own rows, because the classes are substrings
    # of a free-text `summary` and several can match one row: an `if` chain would
    # file a `ConnectTimeout` under `connect error` and a `CASE` ladder in SQL would
    # do the same and cost a recompile per tick. A row that matches more than one
    # needle is counted once per class it matches, so the tallies can OVERCOUNT the
    # headline number and the line says "classes" rather than a partition; `other` is
    # the remainder that matched none, floored at zero for exactly that reason.
    tallies = {label: 0 for label, _needle in _THROUGHPUT_CLASSES}
    matched = 0
    stamps = []
    # The since-boot count walks the SAME rows the tallies walk rather than issuing a
    # second query: the population it is asked about is this day's failed runs, which
    # `rows` already is, and a second `SELECT` would be a second reading of a store
    # the census is supposed to have taken once. `_parse_iso_safe` is this file's own
    # reader for a stored instant — it returns an aware datetime (naive read as UTC,
    # `Z` normalised) or None — so an unparseable stamp is left out of the count, as
    # it is left out of `stamps`, instead of being compared against an aware instant
    # and raising past the census. A row stamped exactly at the boot instant counts:
    # the boot ramp's failures are the whole reason the figure exists.
    since_boot = 0 if boot is not None else None
    for summary, started_at in rows:
        text = summary or ""
        hit = False
        for label, needle in _THROUGHPUT_CLASSES:
            if needle in text:
                tallies[label] += 1
                hit = True
        matched += 1 if hit else 0
        if started_at:
            stamps.append(started_at)
            if boot is not None:
                stamp = _parse_iso_safe(started_at)
                if stamp is not None and stamp >= boot:
                    since_boot += 1
    # Whole minutes, floored: `// 60` on seconds is the same reading the rate below
    # takes, and a fractional minute is precision the line cannot use. A boot in the
    # future — a clock step, an NTP correction — goes negative here and is clamped by
    # the window floor at print time rather than by a second rule invented here.
    minutes_since_boot = (int((now - boot).total_seconds() // 60)
                          if boot is not None else None)
    return {
        "day": day, "prev": prev, "newer": newer, "older": older,
        "ratio": newer / max(older, 1),
        # Which of the two rules fired. At least one of these is True by the time a
        # dict exists: the only way out of the census with a verdict is a triggered
        # day, and a day that trips both says so twice rather than picking a winner.
        "by_ratio": by_ratio, "by_floor": by_floor,
        "classes": [(label, tallies[label]) for label, _ in _THROUGHPUT_CLASSES],
        "other": max(newer - matched, 0),
        "first": min(stamps) if stamps else None,
        "last": max(stamps) if stamps else None,
        # #2956's reading, and all three of them travel together: no boot instant
        # means no count and no elapsed window, so the formatter can print the line it
        # printed before this existed rather than a rate measured from an anchor it
        # does not have. None is also the shape a test overrides the helper to return.
        "boot": boot, "since_boot": since_boot,
        "minutes_since_boot": minutes_since_boot,
    }


def _throughput_alert_message(census: dict) -> str:
    """The one line the census writes. Every number in it came from those rows.

    Both counts are named because "137 failed runs" alone cannot be checked against
    anything, and the day is named twice with the clock on it: the census counts UTC
    days while the note file is keyed to `NOTE_DAY_TZ = America/Los_Angeles`
    (`app/discord_notify.py:126`), so the 00:xxZ failures of a UTC day land in the
    PREVIOUS note. A line that said only "yesterday" would be read against the wrong
    day by the one person it is written for, so it says which clock it counts and
    that this note is keyed to another.

    The parenthetical names EVERY rule that fired, because it used to be hardcoded
    to the ratio (#2899): a floor-triggered 40-vs-20 day would have printed "10x
    rule: 2.0x", a line that misstates its own trigger and points a reader at a
    comparison nobody made. A day that trips both legs says so twice, which is also
    what makes the two rules distinguishable from the notes alone — the owed
    re-pricing of the floor is a count of these lines.

    The since-boot clause is #2956, and it exists because the two counts above cannot
    answer the one question a reader of that line actually had. 2026-10-17 fired the
    floor on 45 failures: 43 of them landed inside one ~29-minute window that straddled
    both UTC midnight and the instant the host came up, and the other 2 were spread over
    the remaining ~23 hours. "45 failed runs against 20" and "45 ≥ 40" are both true of
    that day and neither says whether the pool is failing NOW or was failing before the
    machine came up, which is the difference between an incident and a reboot. So the
    line prints the same day's failures again, measured from the host boot instant, with
    the instant itself on it so the reader can see how long the window was: `12 of them
    since host boot 2026-10-17T00:26:00+00:00 = 0.13 / 15 min (1415 min since boot)` is
    the reading that day would have got, beside the 45 it still says. The day count is
    NOT replaced — the day is what the rules weighed and what a reader re-pricing the
    floor counts — and no rule reads the new number: it is legibility, not a third leg.

    The rate's denominator is
    `max(minutes_since_boot, _THROUGHPUT_RATE_WINDOW_MINUTES)`, not the elapsed
    minutes: three failures two minutes after a boot are 22.5 / 15 min un-floored,
    a figure that describes how recently the machine was switched on rather than how
    hard the pool is failing, and would itself read as the incident this clause is
    meant to discount.

    When the machine would not give an instant the clause is absent entire — no count,
    no rate, no anchor — because the alternative is printing a rate measured from a boot
    the census does not know, which is the invention the defect being fixed was. The
    line is otherwise byte-identical to the one before this clause existed.
    """
    parts = [f"{label} {n}" for label, n in census["classes"] if n]
    parts.append(f"other {census['other']}")
    rules = []
    if census["by_ratio"]:
        rules.append(f"{_THROUGHPUT_RATIO}x rule: {census['ratio']:.1f}x")
    if census["by_floor"]:
        rules.append(f"floor rule: >= {_THROUGHPUT_FLOOR}")
    bounds = ""
    if census["first"] and census["last"]:
        bounds = (f"; failures span {census['day']}T{census['first'][11:16]}Z to "
                  f"{census['day']}T{census['last'][11:16]}Z")
    boot_part = ""
    if census["boot"] is not None:
        window = max(census["minutes_since_boot"], _THROUGHPUT_RATE_WINDOW_MINUTES)
        rate = (census["since_boot"] * _THROUGHPUT_RATE_WINDOW_MINUTES / window)
        # The unit is interpolated for the same reason the divisor is (#3063): a
        # literal `15` here agrees with the constant today and silently stops agreeing
        # the day it moves, leaving a line that normalises to one window and labels
        # itself another.
        boot_part = (f"; {census['since_boot']} of them since host boot "
                     f"{census['boot'].isoformat()} = {rate:.2f} "
                     f"/ {_THROUGHPUT_RATE_WINDOW_MINUTES} min "
                     f"({census['minutes_since_boot']} min since boot)")
    return (
        f"worker pool failure throughput: {census['newer']} failed runs on UTC "
        f"{census['day']} against {census['older']} on UTC {census['prev']} "
        f"({'; '.join(rules)}){boot_part}; classes: "
        f"{', '.join(parts)}{bounds}. Times and days are UTC; this note is keyed to "
        f"America/Los_Angeles, so the day named is not necessarily this note's day.")


def _throughput_reported_day(queue: WorkQueue) -> str | None:
    """The UTC day this alarm last reached a person, fleet-wide.

    The LATER of the two reads, for the reason `_last_alert_instant` gives at length:
    `_state` is one process's cache and the watermark row is the fleet's memory, and
    the pool restarts ~12x a day (#1735, #2662). ISO day strings sort lexicographically,
    so "later" is `max` here and no parsing is needed. An unreadable row is read as no
    durable knowledge, loudly, with the in-process answer standing.
    """
    in_process = _state.get(_THROUGHPUT_WM_KEY)
    try:
        raw = queue.wm_get(NAME, _THROUGHPUT_WM_KEY)
    except Exception:
        logger.exception("failure-throughput alarm: could not read the persisted "
                        "reported day; falling back to this process's own record")
        return in_process
    if not raw:
        return in_process
    return max(raw, in_process) if in_process else raw


def _record_throughput_day(queue: WorkQueue, day: str) -> None:
    """Say that the census for `day` reached a person, here and durably.

    Same ordering as the alarms beside it (#1704): record before dispatch, so a send
    that fails still spends the day rather than handing the next process a fresh one.
    A failed write is logged and the alert still goes out — an alarm that could not
    record itself risks one duplicate line, and a recording failure that silenced it
    would risk the silence this file exists to end.
    """
    _state[_THROUGHPUT_WM_KEY] = day
    try:
        queue.wm_set(NAME, _THROUGHPUT_WM_KEY, day)
    except Exception:
        logger.exception("failure-throughput alarm: the day was posted but its "
                        "watermark could not be persisted; a restart may re-post it")


async def _census_failure_throughput(loop, queue: WorkQueue) -> None:
    """Post the day's failure-throughput census once, on its own watermark.

    Read on every tick and posted at most once per day named: the store read is one
    `GROUP BY` over `runs`, the same shape `WorkQueue.run_rollup_by_source` already
    pays for the Background tab's poll, and the dedupe is the watermark rather than a
    cadence, so two ticks over the same pair are one line and a restart inside the day
    cannot post a second. A cadence here would make the SECOND tick of the pair a
    non-event and the dedupe unobservable, which is the thing clause 3 of #2890 asks
    to be true of the watermark and not of the clock.

    Never raises past `_throughput_census`, which is where every store read lives.
    """
    census = await loop.run_in_executor(None, _throughput_census, queue)
    if census is None:
        return
    day = census["day"]
    if _throughput_reported_day(queue) == day:
        return
    _record_throughput_day(queue, day)
    msg = _throughput_alert_message(census)
    logger.error("%s", msg)
    await _alert(msg)

async def _scan_unparseable_task_files(loop) -> None:
    """Scan the task files and alert on a CHANGE in the unparseable set (#939).

    The alert is keyed on the transition, not on the scan: `_state
    ["unparseable_alerted"]` holds the names already raised, so a file that stays
    broken is named once and not again, and a file that heals gets a recovery
    message and leaves the set — which is what lets a later re-corruption be heard
    at all. A per-scan alert would be one message every `_UNPARSEABLE_SCAN_SECONDS`
    for as long as the file is broken, which is the noise that made the stall alarm
    beside it unreadable (see `_STALL_ALERT_INTERVAL_SECONDS`).

    The recovery half is not decoration. Without it the alert channel records the
    injury and never its repair, and the only surviving account of a resolved
    incident is an instruction to act on it.

    Runs on the caller's event loop with the directory walk on an executor thread,
    like every other scan here: it parses every task file, and the pool's scheduler
    loop is the same loop that dispatches.
    """
    bad = set(await loop.run_in_executor(None, _unparseable_task_files))
    alerted = set(_state.get("unparseable_alerted") or ())
    _state["unparseable_alerted"] = bad
    newly_bad = bad - alerted
    if newly_bad:
        still = alerted & bad
        msg = (f"{len(newly_bad)} autonomy task file(s) unparseable and INVISIBLE "
               f"to the scheduler: {', '.join(sorted(newly_bad))}"
               + (f" (already alerted, still unparseable: "
                  f"{', '.join(sorted(still))})" if still else ""))
        logger.error("%s", msg)
        await _alert(msg)
    recovered = alerted - bad
    if recovered:
        msg = (f"{len(recovered)} autonomy task file(s) parseable again and back "
               f"on the scheduler: {', '.join(sorted(recovered))}")
        logger.warning("%s", msg)
        await _alert(msg)


async def tick(queue: WorkQueue) -> None:
    """One pass of fleet surveillance, called from the pool's scheduler loop.

    The body is the surveillance the dispatch tick used to run, in the order it ran it,
    with the dispatch half left behind: the vLLM probe and gate, `recover_stuck_tasks`
    (which resets claims so dispatch can resume them) and `get_due_tasks` all stay in
    `scheduled_task.enqueue_if_due`. It never probes the model server: watching needs
    nothing from it, which is exactly what #938 was about.

    Raises are the seat's business — `WorkerPool._watch_fleet` catches, because an
    alarm's own failure must not take the loop that enqueues everything down with it.
    Every detector here is read-only.
    """
    from app import autonomy

    loop = asyncio.get_event_loop()


    # Validation of the task files against the scheduler's own parser, on a
    # cadence rather than once per process: any file the scheduler cannot parse is
    # silently dropped (the 2026-05-28 stall), and neither stall alarm can see it.
    # The slot is claimed BEFORE the scan so a scan that raises — an unreadable
    # directory, a mount mid-flip — costs the next interval, not one executor call
    # per tick. It needs no model server, which since #1682 is a fact about this
    # module rather than a note about sitting above the vLLM gate.
    scan_at = _state.get("unparseable_scan_at")
    scan_now = autonomy._utcnow()
    if (scan_at is None
            or (scan_now - scan_at).total_seconds() >= _UNPARSEABLE_SCAN_SECONDS):
        _state["unparseable_scan_at"] = scan_now
        await _scan_unparseable_task_files(loop)

    # Stall alarm — due, grossly overdue, and NOT waiting in the queue.
    import datetime as _dt
    max_dur = _max_duration_seconds()
    overdue = await loop.run_in_executor(None, _grossly_overdue, queue)
    starving = await loop.run_in_executor(None, _queue_starving, queue, max_dur)
    # `_queue_starving` keeps its `(queue, max_duration) -> float` signature: `tick`
    # calls it positionally through `run_in_executor`, three test files call it
    # directly, and tests/test_autonomy_scheduler.py replaces it with a two-argument
    # spy that asserts the value stays a number. So the source of that same wait comes
    # from a second read — taken only once there IS a wait to name, and only on this
    # alert path, never as a per-tick cost (#2692).
    starving_source = None
    if overdue or starving:
        _state["stall_streak"] += 1
        now = _dt.datetime.now(_dt.timezone.utc)
        # Read off the queue as well as out of `_state` (#2662): the pool restarts
        # ~12x a day, and an instant that dies with the process let the same
        # sentence re-post on every fresh boot — six posts of one finding in 21 h
        # on 2026-10-10/11, four in 15 h on 2026-10-09. Read BEFORE the streak check,
        # and an unreadable read still alerts (None): a transient lock may lose a
        # cooldown, it may not silence the alarm.
        last_alert = _last_alert_instant(queue, _STALL_WM_KEY, "due-ness")
        due_for_alert = (last_alert is None or
                         (now - last_alert).total_seconds() >= _STALL_ALERT_INTERVAL_SECONDS)
        if _state["stall_streak"] >= _STALL_ALARM_TICKS and due_for_alert:
            # Recorded before the message is built and dispatched, the ordering
            # #1704 fixed for the alarm beside it: a send that fails still spends
            # the cooldown rather than handing the next process a fresh 6 h.
            _record_alert_instant(queue, _STALL_WM_KEY, now, "due-ness")
            # #2692: the source of the oldest wait comes from a second read of
            # `pending_wait_by_source()` because `_queue_starving` keeps its
            # `(queue, max_duration) -> float` signature — `tick` calls it positionally
            # through `run_in_executor`, three test files call it directly, and
            # `tests/test_autonomy_scheduler.py` replaces it with a two-argument spy that
            # asserts the value stays a number. Like `_operator_pause_clause` below, this
            # read belongs to the alert and never to the tick.
            parts = []
            # The age above came from `_queue_starving`, and the streak, the cooldown and
            # the printed number all weigh that one reading. The source is the only fact
            # this second read adds, so the age is deliberately NOT re-derived here: if the
            # head row were claimed in between, re-deriving could drop the wait under the
            # threshold and post an alert whose parts list holds NO finding at all — a
            # message ending at its own colon. Keeping the confirmed age costs something
            # smaller and self-healing instead: at worst one post names a source that
            # stopped waiting a moment earlier. And the race can never invent the word
            # "stalled", because that prefix is only chosen when this clause is empty,
            # which requires claims this tick can see.
            outranked = ""
            if starving:
                _, starving_source = await loop.run_in_executor(
                    None, _oldest_wait, queue)
            if overdue:
                parts.append(f"{len(overdue)} task(s) due and overdue "
                             f">{_STALL_INTERVAL_MULT}x their interval with no queue row "
                             f"(ids: {overdue[:15]})")
            if starving:
                # Name the source the age belongs to ONLY where the queue can prove the
                # pool is taking work. Without that proof the clause below is today's
                # string character for character — the guardian watch at
                # `agent-services/guardian/detect.py:243` cites it, and
                # `test_a_starving_queue_with_no_pause_alerts_byte_identically` pins its
                # bytes — so a source name never leaks into the genuine-stall case.
                outranked = _outranked_clause(queue, starving, starving_source, max_dur)
                if outranked:
                    parts.append(f"the {starving_source} stream has a claimable row "
                                 f"{starving/60:.0f} min old{outranked}")
                else:
                    parts.append(f"oldest claimable queue item is "
                                 f"{starving/60:.0f} min old")
            # #1550: attribute a starving queue to a held pool before attributing it
            # to anything. Computed on the alert path only — one indexed read of the
            # watermark row, not a per-tick cost — and empty unless an operator pause
            # is actually persisted, so the unpaused message is untouched.
            pause_clause = _operator_pause_clause(queue)
            if pause_clause:
                parts.append(pause_clause)
            # The clause above can name what the wait actually is, and then the word
            # that accuses the scheduler is false on that board: an alert that said
            # both would be two findings arguing with each other. So the prefix moves
            # with it. The else branch is today's prefix character for character.
            head = ("queue behind higher-priority work: " if outranked
                    else "autonomy scheduler may be stalled: ")
            msg = head + "; ".join(parts)
            logger.error("%s", msg)
            await _alert(msg)
    else:
        _state["stall_streak"] = 0

    # Second stall assertion (#421), keyed on each task's own `next_run` so it
    # sees the three shapes the alarm above is built to exclude. Separate
    # streak, separate cooldown, separate message: nothing about the noisy
    # alarm's exclusions or text moved.
    #
    # The streak below stays on the scan's own list, declared parks included, so a
    # board whose only late tasks are parked is still a board the scan CONFIRMED late.
    # Filtering the scan would make it indistinguishable from a healthy one, which is
    # the silence #1121 was filed about. What is suppressed here is the ROUTE, not the
    # finding (#1661): a board that flags nothing but tasks which declare their own
    # park has nothing a person has not already written down, and on this fleet that is
    # the steady state — #68 is parked by a standing ruling — so posting it was one
    # daily-note line per restart on the one surface the fallback exists to reach.
    # The finding is logged instead, carrying the same message, so "nothing is late"
    # and "everything late was parked" remain two different sentences in the log as
    # they are in the alert.
    # The lost-cycle ledger (#2552). Ahead of this alarm rather than after it: a
    # next_run stall alert that fires in the same minute an occurrence is recorded
    # as lost should be reading a board that already says which occurrence the
    # block took, and `_window_losses` never raises, so nothing below it is ever
    # silenced by it.
    await loop.run_in_executor(None, _window_losses, queue)

    stalled = await loop.run_in_executor(None, _next_run_stalled, queue)
    _state["nextrun_streak"] = (
        (_state.get("nextrun_streak", 0) + 1) if stalled else 0)
    if stalled and _state["nextrun_streak"] >= _STALL_NEXTRUN_TICKS:
        nr_now = _dt.datetime.now(_dt.timezone.utc)
        # One test decides both branches, and it is the test the message applies:
        # `_parked_note`, so a task cannot be counted as suppressed by the string and
        # still counted as late by the trigger.
        if any(not _parked_note(e) for e in stalled):
            last_nextrun = _last_alert_instant(queue, _NEXTRUN_WM_KEY, "next_run")
            if (last_nextrun is None
                    or (nr_now - last_nextrun).total_seconds()
                    >= _STALL_NEXTRUN_ALERT_INTERVAL_SECONDS):
                # Recorded before the message is dispatched, so a failed send still
                # spends the cooldown rather than handing the next process a fresh
                # day (#1704's ordering), and recorded durably as well as here, so
                # the next PROCESS is included in that same sentence (#1735).
                _record_alert_instant(queue, _NEXTRUN_WM_KEY, nr_now,
                                    "next_run")
                msg = _nextrun_alert_message(stalled)
                logger.error("%s", msg)
                await _alert(msg)
        else:
            # Nothing late is unaccounted for, so no alert is posted and the ALERT
            # cooldown is deliberately not consumed: spending a day's silence on a
            # park would mute the first genuinely late task for 24 h, which is the
            # silence this file exists to prevent. The log record has its own slot at
            # the same interval, so the scan says it looked without becoming a line
            # every 60 s for as long as the park stands.
            last_parked_log = _state.get("nextrun_parked_logged_at")
            if (last_parked_log is None
                    or (nr_now - last_parked_log).total_seconds()
                    >= _STALL_NEXTRUN_ALERT_INTERVAL_SECONDS):
                _state["nextrun_parked_logged_at"] = nr_now
                logger.warning(
                    "nextrun stall alarm not posted, declared parks only: %s",
                    _nextrun_alert_message(stalled))

    # The worker pool's failure-throughput census (#2890). Last in the tick, which is
    # the whole of its safety argument and is stated here rather than hidden in a
    # try/except: `tick` guards none of its alarms individually — the seat that calls
    # it does, at `workers/pool.py` `_watch_fleet`, whose "Never raises" is what keeps a
    # fault here from taking the loop and every enqueue in the fleet with it. Adding a
    # per-alarm handler would be a second guard on a path that already has one, so what
    # this placement buys instead is an ordering property: by the time the census is
    # read, both stall alarms and the lost-cycle ledger have already run in this tick,
    # so no bug in the new reader can be the reason a stall alert did not go out.
    # And nothing here consults a source config's `enabled` key: the census counts what
    # the POOL ran, which no source's dispatch switch has any standing to define away
    # (#1682).
    await _census_failure_throughput(loop, queue)
