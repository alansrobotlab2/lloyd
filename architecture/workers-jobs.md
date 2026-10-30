---
segment: architecture
tags: [architecture, lloyd, workers]
type: reference
status: implemented
date: 2026-10-25
---

# The worker jobs

One entry per source registered in `workers/sources/__init__.py`. [[workers]]
is the machinery they share — the queue, the claim, the KV gate, the result
contract, the event-loop rule — and this is what each one actually *does*: what
wakes it, what it reads, what it writes, and the one thing about it worth
knowing before changing it.

A **source** is a module with `NAME`, `DEFAULT_PRIORITY`, `enqueue_if_due` and
`execute`. Nothing else is a worker job: the KG pipeline steps, the nightly
chain and the morning triage are all autonomy *task files* that reach the pool
through the single `scheduled-task` source — [[autonomy]] is that mechanism
and [[autonomy-jobs]] is what each of those jobs is for, not this document.

The sixteen are grouped here by **what they are for**, not by priority, because
the families share more than the members do:

| § | family | sources | what it is for |
|---|---|---|---|
| §3 | **dispatch** | `scheduled-task` | one door onto the autonomy fleet |
| §4 | **self-mod** | `arch-review`, `backlog-cluster`, `board-steward`, `owed-check`, `autotriage`, `autocode`, `automod-regression`, `autoresearch`, `frontend-probe-canary` | change Lloyd's own code, behind a gate |
| §5 | **intake** | `youtube-digest`, `deep-research` | turn outside text into vault knowledge |
| §6 | **mining** | `session-distill`, `bench-mine`, `failure-ledger`, `cost-investigate` | turn Lloyd's own exhaust into staged notes, and into dated failure issues |

§1's roster is the other view — the same sixteen in priority order, which is
the order the pool considers them.

---

## 1. The roster

Sixteen sources are registered.

All sixteen have a block under `workers.sources`, and none is registered-but-off:
`failure-ledger` was the last to get one, on 2026-10-06 (#2309). Until that block
existed `WorkerPool._scheduler_pass` read an empty dict for the source
(`workers/pool.py:1613`) and stepped past it at `:1614` without calling its
`enqueue_if_due`, which is why its **on** column was the only **no** in the table and
why its cadence was a module constant rather than a configured interval (see §6's
last entry). Priority is `DEFAULT_PRIORITY` unless config overrides it —
`youtube-digest` is the only real override (45, not the default 60); `arch-review` and
`board-steward` state 62 and 68 in config, though those equal their defaults — and
**lower runs sooner**.

*That first sentence is one clause ending in a full stop, on purpose:*
`tests/test_automod_doc_claims.py::test_workers_jobs_counts_and_rosters_every_registered_source`
parses `^(\w+) sources are registered\.` and compares the word with the number of
`register(` lines in `workers/sources/__init__.py`. The rail exists because §1 said
twelve while thirteen were registered; it fired on this round's first version of the
paragraph, which had joined the count to the next clause with a comma and so read to
the parser as the count sentence being gone.

| source | family | prio | cadence | inflight | turn path | KV-gated | IV | on |
|---|---|---|---|---|---|---|---|---|
| `scheduled-task` | dispatch | 10–70 | 60 s | 2 | `run_query` direct, recorded | no | per task | yes |
| `autocode` | self-mod | 40 | 900 s | 2 | session | **yes** | **on** | yes |
| `youtube-digest` | intake | **45** | 300 s | 1 | session | no | off | yes |
| `autotriage` | self-mod | 55 | 900 s | 1 | session | **yes** | **on** | yes |
| `autoresearch` | self-mod | 60 | 14400 s | 1 | own script | no | — | **yes** |
| `arch-review` | self-mod | **62** | 1800 s | 1 | session | **yes** | off | yes |
| `backlog-cluster` | self-mod | 65 | 3600 s poll | 1 | none (numpy) | no | — | yes |
| `board-steward` | self-mod | 68 | 900 s | 1 | session (primary) | no | off | yes |
| `owed-check` | self-mod | 72 | 120 s | 1 | session (primary) | **yes** | off | yes |
| `deep-research` | intake | 70 | 3600 s | 1 | session | **yes** | off | yes |
| `session-distill` | mining | 70 | 1800 s | 1 | direct (primary) | no | — | yes |
| `automod-regression` | self-mod | 70 | 900 s | 1 | none (subprocess) | no | — | yes |
| `cost-investigate` | mining | **92** | armed at a run's completion, not a timer | 1 | direct, no model turn | no | — | yes |
| `failure-ledger` | mining | 75 | every 1 h poll; **one sweep a day** (`min_interval_seconds`) | 1 | direct (primary) | no | — | yes |
| `bench-mine` | mining | 80 | 7200 s | 1 | direct (primary) | no | — | yes |
| `frontend-probe-canary` | self-mod | 80 | 3600 s poll, one run a day | 1 | none (subprocess) | no | — | yes |

**KV-gated** is `LONG_LIVED = True`: tens of iterations each re-submitting a
100–200k context, so the pool will not *claim* one while the primary's
one-minute median KV usage is over `workers.kv_gate.max_kv_usage` (0.60). Nine
sources declare the attribute — `True` on the five long-session turns
(`autocode`, `autotriage`, `arch-review`, `deep-research`, `owed-check`),
`False` explicitly on `backlog-cluster`, `board-steward`, `failure-ledger`
and `frontend-probe-canary`.

**IV** is `workers.sources.<name>.inner_voice`, and `—` is not "off": it is
meaningful only on a source that runs through `run_prompt_in_session`, because
the observer is wired in `app/routers/messages.py` and nowhere else.
`/api/workers/health` reports it tri-state for exactly that reason. Session
sources set the key explicitly — `true` on `autocode` and `autotriage` since
2026-09-25, `false` on the rest — and a session source with no key
reads as off too: `source_inner_voice` falls back to False since #1015, when
`board-steward` shipped unkeyed and was observed by nobody's decision.
`tests/test_background_inner_voice.py` discovers the session sources from
`workers/sources/*.py` rather than naming them.

## 2. What actually ran

Seven days to 2026-10-28 (`runs.completed_at >= '2026-10-22' AND
completed_at < '2026-10-29'`), read 2026-10-29T18:48Z from
`~/lloyd-data/workers.db`. The upper bound is the change: the read this one
replaced had a start and **no upper bound**, so its window was whatever the store
held at the moment of the read, and §2 and §6 could both say "the same week" while
disagreeing by the runs that landed between their two stamps. A bounded window has
one number. §6's mining figures are `/api/workers/health`'s own unclamped 7-day read
taken at its own instant, so they stay a few runs off these rows — `bench-mine` 101
ok of 123 against 107 of 129 here, `session-distill` 12 ok of 13 against 11 of 12 —
and neither section is wrong: the endpoint's floor moves with the read, this one
does not, and quoting either without its stamp is what made them look contradictory.
Rows are the registered sources that ran; `ok` is `status = 'success'`, and
`failed` / `skipped` are the other two statuses a run can end in, with the residue
`interrupted` (1 `bench-mine`, 1 `youtube-digest`, 1 `arch-review`). `runs` is pruned
at 30 days, so re-measure before quoting this table — what it is for is the gap
between what a source's config implies and what its row says. The stamp before this
one, `2026-09-29T02:19Z`, is the cautionary case, and it stays quoted as one: its own
predecessor `2026-09-28T05:32Z` was thirteen hours old by the time this section's
`automod-regression` row was checked against the store, and it had that source
reporting nothing at all while the window already held two runs of it — a window with
no end cannot be age-checked against a stamp, which is what the upper bound above
removes.

| source | family | runs | ok | failed | skipped | avg |
|---|---|---|---|---|---|---|
| `autotriage` | self-mod | 1573 | 549 | 0 | 1024 | 74 s |
| `owed-check` | self-mod | 715 | 684 | 27 | 4 | 185 s |
| `board-steward` | self-mod | 354 | 343 | 11 | 0 | 90 s |
| `scheduled-task` | dispatch | 261 | 257 | 4 | 0 | 385 s |
| `autocode` | self-mod | 260 | 239 | 13 | 8 | 1797 s |
| `bench-mine` | mining | 129 | 107 | 11 | 10 | 387 s |
| `youtube-digest` | intake | 96 | 54 | 41 | 0 | 205 s |
| `backlog-cluster` | self-mod | 62 | 62 | 0 | 0 | 9 s |
| `autoresearch` | self-mod | 41 | 41 | 0 | 0 | 568 s |
| `deep-research` | intake | 24 | 23 | 1 | 0 | 444 s |
| `arch-review` | self-mod | 22 | 19 | 2 | 0 | 495 s |
| `session-distill` | mining | 12 | 11 | 1 | 0 | 154 s |
| `frontend-probe-canary` | self-mod | 7 | 7 | 0 | 0 | 77 s |

- **The mining pair's cap deaths are history, and so is the code that set the
  cap.** Over the 7 days to 2026-09-11 both failed most of their runs at
  `empty response (stop_reason=max_turns) — nothing written`, against a turn
  budget fixed in the source (15 and 8) — §6 has those numbers and the shape.
  Both budgets are config now: `workers.sources.bench-mine.max_turns` (#896,
  2026-09-24) and `workers.sources.session-distill.max_turns` (#1460,
  2026-09-25), each carried in the queue payload so an item runs under the
  budget it was enqueued with. Over these same 7 days to 2026-09-28, at the stamp
  above, `session-distill` is 34 ok of 43 and still has no row naming
  `max_turns`: 7 of its 9 failures are one `ConnectError` burst on 2026-09-28
  18:51–19:14Z and the other two are empty responses at `stop_reason=stop`
  (`turns=14`, `turns=1`) — while 30 of `bench-mine`'s 33 failures are one
  `ConnectError` burst on 2026-09-24 and 3 are cap deaths, 2 of them in §6's
  earlier read of this same open-ended window. An empty turn is correctly a
  failure ([[workers]] §4); the budget was what was wrong, and moving it was free.
- **The two mining sources are still one shape, and the shared shape is the
  diagnosis**: both are `run_prompt_on_primary` with a budget read from
  `src_cfg` into the payload and a `write_staging_note` at the end. §6 is what
  they have in common; this is what it costs.
- **`autoresearch` is the row not doing what its config implies**: 35 of its 58
  ended `TimeoutError: exceeded max_duration_seconds=1800`, all of them the
  pre-#1546 shape §4 walks; rounds since that cap raise complete.
- **`automod-regression` has two rows, both `success`, and the row above used to
  say zero.** A zero here is the source declining to offer anything, never the
  store failing to hold a row, and two functions decide that between them:
  `enqueue_if_due()` declines the offer silently unless `_runner_needed()` is true
  — `pending_promotions()` non-empty **and** no `runner_alive()`
  (`workers/sources/automod_regression.py:455`) — because offering unconditionally
  would write ninety-six "nothing to measure" rows a day over the handful worth
  reading; and `start_runner()`, the job body, re-checks and returns
  `_skipped("no promotion is waiting to be measured")` (def `:1248`, skip `:1256`). So
  **a zero means no promotion was pending in the window** — or one was and a runner
  was already on it, which is the same `_runner_needed()` answer — and it is a
  declined offer, so unlike a `queued → skipped` transition elsewhere in this table
  it leaves no row at all. "Nothing to mine, which is the design working" is the
  reading that let the placeholder pass for health. These two rows started the detached runner for
  promotion `e6213520` at 18:59:33Z and 19:14:57Z on 2026-09-28 (pids 940170,
  982144), each recording a pid and each completing without writing a
  `guardian_check` row — the `check_runner` half of §4's walk, which is what #1706
  fixed. §4 measured the same zero on 2026-09-27 and read it the same wrong way.
- **One unregistered name does write rows: `queue-maintenance`, 22 of them as of
  2026-09-29T02:19Z** — dated because it grows: `run_sweep()` calls
  `queue.record_run()` when it acts, and the sweep fires every 900 s, so rows
  accumulate on however many sweeps find something (those 22 span 5 days; the
  undated "19 of them" this line carried went stale inside a day). What the count is
  NOT: `avg 0.0 s` is a bookkeeping pass, not 22 worker turns, and a sweep that
  changed nothing records nothing but a watermark — so no row is not a stalled sweep.
  The name is deliberately not a registered work source: the comment above `SOURCE =
  "queue-maintenance"` in `workers/maintenance.py` says so ("nothing enqueues into
  it"), so there is no queue and no runner claim, and its rows arrive from the
  scheduler's own sweep instead of from `run_worker`. That is also why the gap this
  table exists to catch — a configured source with no row — cannot be expressed
  for it at all: it has no `workers.sources` block, and
  `test_config_configures_only_registered_sources` fails the moment one is added,
  because the registry is what a `workers.sources` key may name. So its missing row
  here, and its missing row in §1's 13-name roster, are the design working — unlike
  a zero in the `automod-regression` row above, which is not.
- **The store on this box begins 2026-09-22T20:03Z** (the 2026-09-22 data wipe), so
  no window `runs` can show holds the old names §7 lists, and a "7 days" window
  here has never been shortened by pruning.

---

## 3. Dispatch — one door onto the autonomy fleet

One source, and it is not really a job. It is the seam the entire autonomy
fleet comes through: 45 task files in the vault, each with its own schedule,
priority, skill and dependencies, all reaching the pool as
`scheduled-task:<id>`. Everything the pool knows about the nightly KG chain or
the morning triage, it knows through this one module — which makes it the widest
fan-in in the table rather than the busiest: one module for 45 schedules, but
257 runs over the 7 days to 2026-10-27 (`/api/workers/health`, window unclamped),
fifth behind `autotriage`, `owed-check`, `autocode` and `board-steward` — the
rank §2's own table gave it at its 2026-09-28 stamp too. It is also the source
with the least to say about what it does.

### `scheduled-task`

**Wakes** every 60 s. **Reads** `~/obsidian/autonomy/*.md`, evaluates due-ness
per task through `autonomy.get_due_tasks`, and enqueues each due task under
`scheduled-task:<id>`. **Executes** by delegating to `autonomy.run_task`, which
writes the per-task run record to `autonomy-runs/<id>/<run_id>.md`.

Priority is the task's own frontmatter mapped through `_PRIORITY_MAP` —
critical/high/medium/low/background to 10/20/30/50/70 — which is why this one
source can both preempt everything and sit behind everything in the same tick.

Four behaviours are load-bearing:

- **It gates on the health of the model each task pins**, not just the
  primary. A task pinned to a dead secondary is skipped, not enqueued into a
  retry loop that becomes a ConnectError flood. The named-model leg has no member
  on this box as of 2026-10-27 — every one of the 45 task files pins
  `model: primary` — and a pin `config.models` does not define resolves to the
  *primary's* endpoint (`_model_health_url`, `workers/sources/scheduled_task.py:113`
  falls through to `_primary_health_target()`), so until a pin names a real second
  model this gate is a second copy of the fleet gate.
- **It passes the pool's cap into `run_task`** (`max_duration_seconds: 3600`;
  `run_task` (`app/autonomy.py:4585`) keeps its own timeout 30 s under the value
  it is handed, the margin being `_POOL_TIMEOUT_MARGIN`) so the task's own timer
  always wins. When the two were equal the pool cancelled the handler before it could
  write anything, and 237 runs and 73.6 GPU-hours recorded with a NULL `task_id`
  were that bug — but the pair is `config.yaml`'s comment on the cap, not a row
  anyone can re-query: `SELECT count(*) FROM runs WHERE source='scheduled-task'
  AND task_id IS NULL` returned 0 on 2026-10-27, because those runs pre-date the
  2026-09-22 wipe §2 records.
- **It recovers claims the pool never finished**, ahead of both the gate and
  `get_due_tasks`: `enqueue_if_due` calls `autonomy.recover_stuck_tasks` on every
  tick (`workers/sources/scheduled_task.py:291`), because a worker that dies
  mid-run leaves the task file `in_progress` and the task stops coming due. It
  sits *before* the vLLM gate, so a dead claim is recovered even while dispatch is
  paused.
- **It does not watch the fleet. `workers/fleet_watchdog.py` does.** Until #1682
  the alarms below sat in this source's `enqueue_if_due`, so `enabled: false` on
  this one source — a dispatch-side switch — disarmed all of them and raised no
  alert about the disarmament. They now run from `WorkerPool._watch_fleet`, on the
  scheduler loop's own seat, whether or not this source is enabled or even raises:
  - **A task file the scheduler cannot parse is announced**, on a 30 min cadence
    and on state transitions. A file that does not parse is invisible to the
    scheduler, which is the 2026-05-28 stall; silence there is indistinguishable
    from a healthy fleet.
  - **There are two stall alarms**, each with its own streak and cooldown. The
    grossly-overdue one: a task due past a multiple of its own interval with no
    queue row, or a claimable item going stale past 3x `max_duration_seconds`
    (the source's own config value, which is the only thing the watchdog reads
    from it), after `_STALL_ALARM_TICKS` ticks, rate-limited 6 h. And the
    `next_run` one, for a task whose own `next_run` is a full period stale with no
    queue row at all, after its own streak, rate-limited 24 h.
  - **One alarm on that seat is not about due-ness, and did not come over with
    #1682**: the failure-throughput census (#2890, added 2026-10-17; absolute floor
    #2899, since-boot rate floor over `_THROUGHPUT_RATE_WINDOW_MINUTES = 15` min
    #2956) counts the UTC day's failed runs across every source and posts at most
    once a day on its own `throughput_day` watermark. `fleet_watchdog.tick` runs it
    last, reading `runs` rather than task files — so §2's table is its subject, not
    this group's due-ness, and a roster copied from `WorkerPool._watch_fleet`'s
    docstring misses it: that docstring still names only the two stall alarms, the
    starving clause and the scan.
  - **The invariant across the two stall alarms** (#938): an outage of the model server pauses
    dispatch, never the watching. The vLLM gate stays here because it is dispatch,
    and its own threshold is 45 min of failed `/health` before `logger.error`
    plus discord. Which `/health` it checks is config-derived — the default
    model's `models.<name>.base_url` through
    `app.config.default_model_base_url_source` — so a port move under `models:`
    moves the gate, and the alert names the config key it read; the literal
    `:8096` survives in that module only as the fallback for an unreadable config
    (#1684). (The co-watcher's port is a separate literal table,
    `app/supervisor_client.py:_INFRA_SERVICES`.) It is one of TWO watchers of the
    primary's `:8096`: the other is
    `workers/service_probe.py`, which watches the PORT under supervisord rather
    than `/health` and announces the closed port to the journal and a toast at its
    own 30 min grace, from the pool's scheduler loop and outside this gate. One
    outage therefore produces two lines at two thresholds, the
    `workers/service_probe.py` line first, and the gate reads its elapsed minutes
    off that watcher's streak rather than counting its own ticks (#1683). Both are
    announcement-shaped: neither writes a guardian ledger row for an engine
    outage. The detectors above are read-only and never touch the model server.

### When the loop itself is stopped — what `workers.enabled` silences, and what notices

#1682 moved the alarms above off the `scheduled-task` source and onto the pool's
own seat, which closed the source-level route: disabling one module could no longer
disarm all of them. One level up, two supported routes still stop the seat itself,
and each stops the reporting of the problem along with the work:

- **`workers.enabled: false` at boot.** `start_worker_pool()`
  (`app/routers/workers.py:962`) returns before `WorkerPool` is constructed, so
  `_scheduler_loop` (`workers/pool.py:1566`) never runs at all.
- **`POST /api/workers/enable {"enabled": false}`.** `app/routers/workers.py:559`
  awaits `pool.stop()`, which clears `_running` and cancels `_scheduler_task`
  (`workers/pool.py:839`).

What either one silences is everything that rides that loop, not just dispatch: the
parse announcement and both stall alarms above, `workers/service_probe.py`'s port
watch, and — because the sources that *create* work are polled from `_scheduler_pass`
on the same loop (`workers/pool.py:1596`, body `:1612`) — the entire autonomy/nightly chain,
including the job that would have written the alert. `/api/workers/status` and
`/health` both publish the state honestly (`{"initialized": false}` on the boot
route, with no `pool` key at all, because `get_queue()` raises to tell "switched off"
from "running and empty"; `{"pool": {"running": false}}` on the enable route) and
both are pull surfaces that nothing alarms on.

The silence is therefore reported from outside the pool:
`agent-services/guardian/poolwatch.py`, seated in `Guardian.check_pool` above
`tick()`'s early returns, which alerts once on the ordinary guardian surface
(`alert()` → notifier → `ALERT.md`, ledger, daily note) after
`policy.POOL_SILENT_STREAK = 3` consecutive ticks reading no running pool while the
snapshot's `/health` verdict is ok, and repeats at most once per
`policy.ALERT_REPEAT_SECONDS` window from the watermark in `pool-silence.json` — on
disk rather than in `_alert_seen`, because the silence outlives a guardian restart
and that dict does not. It deliberately does **not** report a pause
(`running: true, paused: true`), which leaves the loop and its alarms alive, and says
nothing while the backend is unreachable, leaving one outage to the existing liveness
alert. Pinned by `tests/test_guardian_pool_watch.py` (#1747).

Long version: [[autonomy]] for the mechanism, [[autonomy-jobs]] for the 45
jobs it dispatches — the reflection chain, trace2skill, the graph chain, vault
hygiene, inbound signal.

---

## 4. Self-modification — the loop that changes Lloyd's own code

Nine sources are in this family. Five of them are one loop over the backlog,
and they read best in pipeline order rather than priority order:

```
arch-review  →  backlog-cluster  →  autotriage  →  autocode  →  automod-regression
read one doc    group the drafts    judge one       implement one   measure whether the
against the                         item or one     confirmed item  landing made anything
tree, file                          cluster         behind the gate worse
what is wrong
```

`board-steward` and `owed-check` are not stages of that chain, and neither is
the family's ninth member, `frontend-probe-canary`: it judges no board and
touches no code, it checks that one gate rung's instrument still catches a
broken build. The two judgment services each run beside the state machine —
on the primary, one turn per tick or per item, with an `apply` flag that
decides whether its answer is written or merely recorded.

`arch-review` is the loop's one *supplier* rather than a stage of it: nothing
downstream waits on it, and what it files enters the board as ordinary drafts.
It is here because its findings are about this tree and it writes to this tree —
one `architecture/*.md` doc per run, and nothing else.

What the five share is that **no member acts on its own conclusion**.
`backlog-cluster` writes a file that triage is free to ignore; `autotriage`
reaches a verdict and implements nothing; `autocode` lands only through thirteen
rungs and a second reader; `automod-regression` measures and never reverts; and
`arch-review` may correct the doc it read and must *file* every other finding,
including the one-line skill fix it could obviously make itself.
Each one's output is the next one's input, and every handoff is a file on disk
rather than a call — so a member that is down stalls the loop rather than
corrupting it.

The family's outlier is `autoresearch`. It is the one that rewrote a prompt
surface with no gate at all, and what it did with that is the last entry in
this section — it is here as the counterexample the others are shaped against.
It runs again (re-armed 2026-09-16, behind the bench-and-landing gate it
lacked), but its gate-less week is why the loop above is shaped as it is.

Long version: [[automod]], [[backlog]].

### `arch-review` — one doc, checked against the tree it describes

**Wakes** every 30 min and takes the unit that has rested longest from a
picklist: every top-level `architecture/*.md` is a unit, plus one per
functional *group* of [[autonomy-jobs]] and this doc (hand-kept in
`workers.sources.arch-review.groups`). A unit rests 30 days after a review, so
the first pass takes about twelve days at `daily_max: 4` (36 docs + 12 groups =
48 units as of 2026-10-25) and steady state is a review a month per unit.

**Executes** as one session (Inner Voice off) that checks every backticked path,
count, tool name and config key against the tree — `Read`, `Grep`,
`graph_affected`, `git log`, and `curl` on `/api/workers/health` and
`/api/autonomy/health` for anything the doc states as measured — then reviews
the code the unit names. A group gets three lenses instead: does the grouping
still hold, what the members' own code and SKILL.md bodies do, and where the
section's stated invariants and its members have come apart.

**Writes** exactly one file: that doc. Findings become `arch-review` drafts
tagged `spawned-by-review`, and a fix that belongs in a skill, an autonomy task
file or anywhere else is filed rather than made. Everything else the turn wrote
— anywhere in this repo, and anywhere in the vault bar `backlog/` and the
scheduler's own `autonomy/` task files, whose `last_run` stamp a revert would
take back and re-dispatch the run (#915, #82) — is reverted from a `git status`
baseline taken before the turn. The doc's own diff is thrown away if it exceeds
400 changed lines, deletes more than 30% of **the unit under review** — a
group's own section, not the file, so a 43-line section dies at 13 deleted
lines — breaks the front matter, or (for a group) has a hunk outside its
section, which the rail counts strictly and therefore also rejects a refreshed
`date:` line in the front matter. What
survives, the **source** commits: the model never runs `git`.

It stages nothing under `pending-research/`, so it has no `_DEFAULT_DEST` entry
and does not appear in the Review tab — its output is a commit and a set of
backlog drafts, both already durable.

Long version: [[arch-review]].

### `backlog-cluster` — the deterministic half of group triage

**Wakes** hourly and runs on either of two age gates; the ledger row records
which one fired as `trigger: nightly|exhausted`. *nightly*: `clusters.json` is
older than `min_age_seconds` (20 h), so "nightly" is the age of the output
rather than a wall-clock hour and a restart never doubles it up. *exhausted*:
the file is younger, but at least `exhausted_min_age_seconds` (2 h) old — that
floor — and `select_cluster` has nothing left to take from it. The second gate
is what runs the job on a caught-up board: every rebuild rewrites the file and
resets its age, so while the clusterable pool is empty the age never reaches
20 h and the rebuild lands every 2-3 h instead — 94 of the 98 runs logged
2026-09-11→29 carry `trigger: exhausted`. **Executes** off the loop
(`asyncio.to_thread`): a numpy pass over qmd's stored backlog vectors, plus
shared file paths and parent links, plus an optional pair-judge on the
secondary for ambiguous edges. **Writes** `clusters.json` in the automod state
dir, which `autotriage`'s group mode consumes.

**What it is producing is close to nothing.** 8 of its 10 runs on 2026-09-27
returned `0 clusters over 0 items`: its input is the untriaged-`draft` pool,
and the sweep keeps that pool empty on a caught-up board. On an empty input its
summary says `no items in the triage pool`; `no vectors available, clustered on
paths and parents only` is reserved for a run that asked for items and got no
vectors back (#1690). Before that split both zeros printed the same clause, so
an idle night read as a dead embedding pipeline with the denominator, not the
vectors, at zero. The edge into `autotriage`'s group mode is therefore
carried by the days on which new drafts outrun the sweep, not by this job's
cadence.

No session, deliberately: a clustering pass is arithmetic, not a judgement
anyone needs to review.

Long version: [[automod]] §clustering.

### `board-steward` — one judgment beside the state machine

**Wakes** every 900 s and enqueues one pass (`board-steward:tick`). **Executes**
as one session on the **primary** (`inner_voice: false`): it reads the ledger
events since its last run and up to 80 open items, and answers with one
structured object — a list of board moves and the item `autocode` should take
next. It runs on the primary because on the secondary it made ordering errors
in two of five dry-runs and its live tick died at `max_turns` on tool reads
with no answer; a two-minute judgment every fifteen minutes is a cost the
primary carries, and the round hold keeps it off the engine while a round runs.

**Writes** nothing to the board while `apply: false`, which is the shipped
state: it records how far its answer agrees with what the state machine would
have done, and that record is what decides whether `apply` flips. The state
machine in `scripts/automod/backlog.py` is still the writer today — the
module docstring's "the replacement" is the plan, not the present. With
`apply: true` one mechanical writer would apply its moves and
`select_confirmed` would take its pick.

**That record was not evidence of anything when it was first read** (measured
2026-09-27, 136 ticks over 7 d). Its successful ticks reported `0 move(s)
proposed, agreement 100% (0
agree, 0 disagree, 0 missed)` — a `rate` of 1.0 computed over zero judged
decisions, which is the failure mode `config.yaml` itself warns of above the
flip — and 17 of the 136 failed with `finalizer failed: output truncated at
8192 tokens`, so a twelfth of the ticks do not even write their row. (Since #1606
a tick's record opens with the steward's own sentence and these counts follow
it, because the counts took ~110 of the summary's 500 characters and the
sentence up to 400, so half of all ticks lost the tail of their board state to
the cap. What the counts *say* is unchanged; only their position is.) It is
evidence now, and the evidence is negative: over the rolling 7 d to 2026-10-27
`judgement` reads 127 judged, 101 agree, 26 disagree — **rate 0.795, under the
0.90 the flip needs** — on 279 ticks (267 ok, 12 failed, and the failures are no
longer truncation: they are `output is not JSON`, an empty answer, a diverging
finalizer and `ConnectTimeout`). So `apply: false` is no longer an unmade
decision waiting on an
unmeasured quantity; it is the reading of a bar this week does not clear.

**Below the floor, the disagreement now has a reader** (#3519). A measured fail
is actionable in a way an unmeasured one is not, and until now the 26 disagreeing
ids were published on a health route that nobody consumed: `autotriage` triages
items, `owed-check` answers owed entries, `arch-review` reads docs. So
`workers/board_disagreement.py` runs as the last act of an ordinary tick
(`execute`, after the watermark and after the tick's own `board_steward` ledger
row): it asks `judgement_aggregate` — the same function this section's `judgement`
figures come from — and only when the window is **measured and under the floor**
(summed `judged` ≥ 50 with agree `rate` < 0.90, and not `truncated`) does it file
**one** draft item on the `lloyd` board, tagged `board-steward-disagreement`. Its
rows name both sides of each disagreement — the status the steward proposed, from
its parsed moves, and the status `desired_statuses` wants, with that rule's
`why` — because the proposed move is stored nowhere else (every board-steward
`runs` row has an empty `response_json`) and the question "which side is wrong"
cannot be answered from the store. A later below-floor tick **adds its new ids to
the same item** rather than filing a second one, and an item a round is already
working (`in_progress`) is left unrewritten. Each file-or-merge appends one
`board_steward_disagreement` row to the promotions ledger carrying that window's
`judged`, `disagree` and `rate`, since the run summary is capped at 500 characters
(#1606) and 77 of 154 rows already stop at exactly 500. Filing is a report on the
tick, never its outcome: it cannot reach the tick's `status` or `summary`, it
changes no item's status, and a filing that raises is logged and swallowed — the
agreement record is built from tick *success*, so a reader that could break the
instrument would make the flip harder to earn. **The flip itself is unchanged:
`apply` still turns only on owed-check's bar** (summed `judged` ≥ 50 and agree
`rate` ≥ 0.90, #3089 asked / #3329 ruled), which this module restates as its
trigger and does not own.

Since #1688 the record no longer claims what it did not measure. `agreement()`
carries `judged` — the decisions both sides made — on the run meta and the
`board_steward` ledger row, and its `rate` is `None` when `judged` is 0; such a
tick's summary reads `agreement not measured, 0 judged (…)` and one that judged
something reads `agreement 50% of 2 judged (…)`. The flip is judged on a sum of
`judged` over the ticks, never on `rate` alone: since #3089 (2026-10-20) the
comment above the flip states both numbers, and owed-check settled them on
2026-10-25 (#3329) — summed `judged` ≥ 50 and agree `rate` ≥ 0.90 over the
rolling 7-day window — and `GET /api/workers/health` publishes them as
`judgement` on this source's row, so the gate and the figure that clears it are
read from one place. The floor came down because #3089's 100 sat ~20% above the
measured weekly plateau under it, not because 100 was unreachable — the oldest
of the four weeks the ruling read (163/82/78/82, ending 2026-10-25) cleared it.
And `rate` is a pre-flip measure only: `disagree` is scored on the moves a tick
proposes, so after `apply: true` an applied wrong status is never re-proposed and
the standing divergence shows up in `missed` and the `judged` denominator
instead, where nothing new ever reads as a disagreement.

**It may never set `done`.** Closing is gated on a settled promotion whose
outcome said `met` and stays mechanical in `close_settled_items`: a closed
item is never re-triaged, so it is the one move the loop cannot recover from.

### `owed-check` — settle what an item still owes

**Wakes** every 120 s and, when any item has a due entry on its `owed` list
(`scripts/automod/owed.py`), enqueues up to `batch` jobs — `owed-check:item:<id>`,
one row per item, an open item owed a decision first and then oldest owed entry first, `batch` defaulting to 3 (#1909: one
item per tick settled fewer entries a day than the loop filed). **Executes**
as one session on the **primary** per item: it measures each owed entry
against the live system (run records, `server.err`, the ledger, the code) and
answers per entry with one of `settled`, `recheck` (a date, clamped to 30 days;
an entry rechecked four times is ruled on instead), `ruling`, `work` (a draft
follow-up item, at most `spawn_cap` per item), `reopen` (which may carry
`amend_clause` — a review disagreement is this job's to decide, and it may
reword the clause both reviews called wrong as written), `close`, or `outside`.
The session may read and run commands but not edit files or write the board;
the apply step is the only writer.

**Why it exists.** The owed list replaced the `needs-human` tag on 2026-09-27.
A met landing with a post-landing check, a path the loop may not write, a
spent attempt and a parking triage verdict (`not_code`, `unverifiable`,
`human-only`) all used to park on Alan with the tag, and nothing came back for
it: 257 closed items piled up, and a hand sweep found 125 of them already done
or moot. Alan's ruling: nothing parks on him, and Lloyd approves his own
choices. `outside` is the only class that reaches him — physical or his alone:
hardware, sudo on the host, a secret or account he holds, money, a post under
his name — listed on Mission Control's backlog panel, never as a tag. Deleting
or moving data is not `outside`: it is ruled on and filed as work for a gated
round; neither is a protected path, a `config.yaml` edit or a decision. His
second ruling (2026-10-05) is in the prompt verbatim: the loop goes with its
own recommendation, and only something physical waits for him
(`architecture/automod.md` §3.2k).

**Writes** for real since 2026-09-27, when `apply` flipped to `true` after a
dry run of 9 items against the hand sweep: the 4 dated checks came back
`recheck` as the sweep had them, and of the 5 the sweep had left for Alan, 2
came back `outside` and 3 settled or ruled on evidence the sweep predated. It
now rules, closes, and files its follow-up drafts; every answer is still an
`owed_check` ledger row, and an item a dry run already answered is still
skipped. Measured the same day: 14 runs, 14 applied — rulings with items filed
(#1572 → #1680, #1418 → #1671, #1241 → #1668) and one `outside` reaching the
Mission Control panel.

### `autotriage` — is this backlog item still true?

**Wakes** every 900 s, enqueues one item under `autotriage:triage` with the
budgets carried *in the payload*, so the run uses the config that was live when
it was queued. **Executes** one session turn, Inner Voice **on** again since
2026-09-25 (`inner_voice: true`, with the worker note and the report-ask rail
that replaced the reasons it was switched off), `max_inflight: 2` the same day,
and `retry_seconds: 60` so a board with drafts waiting is worked about every
minute rather than every 900 s.

Three modes share this one source and the run picks in a different order from
the pipeline diagram above. A `high` item goes before everything; then **sweep
mode** (`sweep: true`, shipped on) takes `sweep_batch: 8` never-read items in
one turn while any open item is unread, retiring or ranking each; then a
**cluster** goes before the single pool — when `clusters.json` holds a cluster
with ≥ `group_min_items` untriaged members, the run consolidates it instead:
closes duplicates, retires the stale, folds the rest under one umbrella item,
which is finite work where the single pool is not. Even that yields when the
single candidate's own priority outranks the cluster. On a caught-up board the
first two rules are the whole story: its runs on 2026-09-27 are single
confirmations and `skipped: every open backlog item has been triaged`.

- **It implements nothing.** Triage is read-only: a verdict plus the evidence
  for it. Splitting it from implementation is the whole point — the failure
  worth avoiding is a confident, tested, gated change that solves a problem
  nobody has, and that is only reachable if implementation can start from an
  unverified premise.
- **Retiring an item is a success**, and for a backlog this age it is the
  expected outcome.
- **The verdict is structured, with the regex kept as a fallback.**
  `final_schema` (`TRIAGE_VERDICT_SCHEMA`) asks the harness for one extra
  completion restating the conclusion as JSON. The finalizer is skipped when
  the turn did not end of its own accord, so a pipeline with no fallback would
  turn a transient engine error into a lost triage. `verdict_source` on the
  ledger is how you tell a finalizer that quietly stopped working from one that
  is working.
- **The item body is truncated from the END** — that is where appended triage
  sections live, so the original survives.
- **Its own output is quarantined from its own queue.** A pass that files items
  into the pool it reads from has R > 1; it went 19 → 122 open items in 48
  hours before `backlog.is_quarantined`.

Long version: [[automod]] §triage, [[backlog]].

### `autocode` — one confirmed backlog item becomes landed code

**Wakes** every 900 s, and does housekeeping *first*, whether or not a round
can start: reap abandoned rounds, close settled items, reconcile statuses,
expire stale spawns. Each is wrapped so it can never take the scheduler down.

**Then** it checks `_loop_is_free` — automod enabled, not halted, not BROKEN,
no promotion under observation, no rollback pending, no round open — and
`select_confirmed`, and enqueues at most one round under `autocode:round`.

**Executes** one turn in a real session following the `automod-change-own-code`
skill: open a worktree round, do the work, run the thirteen-rung gate, land it or
abort. Landing runs detached, exactly as when a human drives it.

- **Two gates stand in front of it.** Triage must have reached `confirmed`
  *with an acceptance check* — an item confirmed without one is skipped, not
  guessed at, because a round with no contract cannot fail. And the loop must
  be free, re-checked at run time because a queued item can sit.
- **It goes through `/api/message/stream`, not `run_query`.** The reason is the
  observer and the transcript: the chat path is the only one that attaches Inner
  Voice, and it is on for this source again since 2026-09-25, with the round
  landing in the Inner Voice history where anyone can read what it did. The
  older reason is retired — `automod_start` does **not** refuse a turn with no
  Inner Voice attached any more (`automod.require_inner_voice` defaults false
  since 2026-09-24, IV plan R5: every turn is recorded and every turn runs the
  deterministic guards).
- **The loop finishes rounds the turn did not.** Since 2026-09-18 the
  scheduler is a second writer of a round's outcome: `land_passed_gates` lands
  a round whose gate passed at the commit it still holds, `regate_unreviewed`
  re-gates one whose only failed rung was a review that could not run, and
  `gate_ungated` (2026-09-25) commits the leftovers of a turn that ended with
  unjudged work and gates them once, landing on a pass. `continue_session`
  then runs the next item on that slot in the previous round's session. So one
  turn no longer equals one round: an aborted turn is not necessarily the end
  of the work it did.
- **The clock is not the throttle; the gates are.** 14400 → 3600 → 900 s, each
  cut for the same reason: `_loop_is_free`, the dedup key and `max_inflight: 2`
  (raised from the default 1 on 2026-09-17, climbed to 4 while automod
  throughput was being pushed, cut back to 2 by Alan on 2026-09-24) decide
  whether a round may start, so the interval's only job is to ask them
  often enough. At an hour it did not — one round settled at 19:16 and the next
  poll was 19:50, with four confirmed items waiting.
- The longest-running job in the pool: ~1810 s mean over its 350 completed runs
  (measured 2026-10-27), though its last ten ok runs averaged ~800 s. Over the 7 d
  to 2026-10-27: 280 of 301 runs ok (13 failed, 8 skipped) for 152.8 GPU-hours —
  just over half the pool's 295.2 measured hours.

### `automod-regression` — did the landing make anything worse?

**Wakes** every 900 s under `automod:regression`. The comparison no longer runs
here: since 2026-09-18 it runs **detached**, and the promoter starts that runner
the moment a landing is verified, so the measurement really does arrive shortly
after a landing rather than on this source's poll. What is left in the pool is
the safety net for a runner that never started — `execute` is
`await asyncio.to_thread(start_runner)`, a few milliseconds with no engine,
which is why the source is exempt from the round hold — and it decides what is
owed a measurement from the **ledger**, not from `last_settled.json`: every
`promoted` row of the last day that was not rolled back and has no
`regression_check` row that measured anything, oldest first, behind
`regression.lock`. That is what makes "one check per promotion" true at any
landing rate; measuring "the latest promotion" lost them whenever two landed
inside one check. Measured 2026-09-27: zero runs in the 7 d window, which is
what a safety net that has not had to fire looks like.

- **It is a paired A/B on identical data, and that shape is the whole point.**
  Check the promotion's **parent** out into a scratch worktree, point both arms
  at the *live* fact tree and knowledge graph via `LLOYD_FACTS_ROOT` /
  `LLOYD_KG_DB`, and run them in the same window. Vault drift cancels; what is
  left is the code. Comparing against a number recorded at the last promotion
  would measure how much the vault moved.
- **It deliberately does not compare the autoresearch composite score.** Three
  identical baseline runs scored 0.719 / 0.542 / 0.624 — a spread of 0.177
  against a promotion threshold of 0.05. A detector built on that fires on
  sampling noise and is switched off within a week. `eval/run_eval.py` has no
  LLM in it and five consecutive runs against an unchanged vault produced
  *identical* quality metrics, so any movement there is signal.
- **A missing noise file means "cannot evaluate", never "no regression".**
  `_skipped(reason)` returns a real `status: skipped`, not a bare `{"skipped":
  …}` — that older shape read as 22 successes with empty summaries before
  `pool.normalize_result` learned it.
- **It compares against the parent, not the LKG**, because by the time a
  promotion settles the LKG *is* the promoted commit.
- It is the source that taught the event-loop rule, with two 900-second eval
  arms and a `git worktree add` between them.

### `frontend-probe-canary` — is the gate's frontend check still catching broken builds?

**Wakes** on the 3600 s poll and runs at most once per `min_interval_seconds`
(86400), held by its own `last_enqueued` watermark so a restart or a finished
run cannot let a second one in. `execute` spawns
`python -m scripts.automod.frontend_probe_canary --seeds 16` as a subprocess off
the event loop, with no `env=`: the child inherits the pool's environment, which
sets no `LLOYD_AUTOMOD_STATE`, so the artifact lands at
`~/.local/state/lloyd-automod/frontend_probe_canary/latest.json` — the file
#1601's owed entries read, which until #1981 nothing but a hand run could write
(every test redirects the state dir).

- **The source is only the schedule.** Seeds, the 90% bar and the artifact are
  the script's; it is not a gate, and the check it measures stays observe-only.
- **`--seeds 16` is the whole must-detect table** (`SEEDS` in the source, raised
  12 → 16 on #2130 when four layout seeds joined it; a test pins the constant to
  the table). The script's default floor of 10 would let a seed drop out of `n`
  on top of the one miss the bar allows.
- **The exit code is the verdict.** 0 is a success carrying `detected`,
  `seeds_measured`, `rate`, `rate_counting_blind_as_misses` and `run_at` (in the
  result and in the run's `meta`). 1 is a returned failure carrying the same
  measured numbers. 2, a crash, a timeout, or an exit 0 with no artifact from
  this run is a returned failure that says "not measured" and carries no rate —
  `latest.json` may still hold an older run.
- **Measured state exists** (2026-10-27): 7 runs over the 7 d, all ok, and
  `latest.json` reads
  `seeds_measured: 16`, `detected: 16`, `rate: 1.0` with one blind seed
  (`rate_counting_blind_as_misses` 0.94) — clear of the script's 90% bar. That is
  **one run a day, not one an hour**: `min_interval_seconds: 86400` floors the
  3600 s poll, so the `interval_seconds` figure is how often it asks, not how
  often it measures.

### `autoresearch` — the gate-less week, and why it runs again now

**Off 2026-09-08 → 2026-09-16; re-armed since, and gated.** One
prompt-optimisation round per `interval_seconds` — 14400, every four hours and
not the hourly cadence of the incident below: a round now runs nearly all the
time and the source is exempt from the round hold, so the interval is the share
of the primary it may take (at most 30 min in 4 h). Wraps
`scripts/autoresearch/run_round.py`, dedupped on `autoresearch:round`; a round
is bounded to the source's `max_duration_seconds` (below, #1546).

It promoted generated prompt variants straight over the live `lloyd/SOUL.md`
and `MEMORY.md` every hour with **no gate, no test, no review and no revert**.
It produced the #464 `MEMORY.md` clobber, and at 09:54 on 2026-09-08 it
overwrote the operating contract *during* the round that was trimming it —
re-inflating the gate stack to 64% ninety minutes after #377 had cut it to 48%,
with a variant whose own hypothesis was the technique #377 was filed against.

That is the property the rest of this family is built around, stated by
its absence: a member of a self-modification loop may not promote its own
conclusion.

`promote()` refuses a variant that breaks the prompt-surface invariants and
commits what it does apply through `automod_vault_land` (#506's route, since
closed), so the writer is bounded rather than removed. What made re-arming
possible: #876 (landed 2026-09-15) restored variant benching — before it,
`run_round` benched nothing and a round could promote on baseline-only data or
not at all — and with benching back, the promotion floors do the gatekeeping:
`autoresearch.promotion` demands `min_composite_delta` 0.05,
`min_bench_win_fraction` 0.5 (held at 0.5 on 2026-09-13, #428's false-positive
measurement) and `require_safety_pass`. Alan pre-approved the re-arm on
2026-09-13; config flipped it on 2026-09-16. The 7 d to 2026-09-27 read 16
success / 35 failed, all 35 the pre-#1546 shape —
`TimeoutError: exceeded max_duration_seconds=1800`, the last of them at 05:46Z
on 09-27. That failure mode is closed: over the 7 d to 2026-10-27 the source is
41 runs, 41 ok, 0 failed, 6.18 GPU-hours, every round `stopped at deadline: no`
at 212–849 s and reporting `3 variants evaluated, 0 promotable, winner=none`.
It still promotes nothing, and the reason has moved off the cap twice over. The
projection (#1605, below) shrinks the matrix on 40 of those 41 rounds
(`matrix_shrunk` 40, `deadline_cut` 0), and separately at least one task in the
round's targeted or held-out split scores on no arm — the 2026-10-27 22:13Z
round's held-out `bench_005_replay_memory_update` is "not mechanically
checkable", so all three variants end `HOLD — partial_heldout_coverage` before
any delta is read (#549 / #416). Over the 176 rounds since 2026-09-26 the
ledger holds 513 decisions and zero promotions: 309 `partial_heldout_coverage`,
121 `partial_targeted_coverage`, 61 `safety_regression`, 22 `deadline_stopped`.
Read that as the doc's older sentence being wrong: the gate **is** exercised on
finished rounds now, and what refuses is coverage, not the floors doing their
job. Every round does
now score the frozen behavioural scenarios (#1549, landed 2026-09-26 alongside
#1546) — report-only by construction: an axis in the report that feeds no
promotion verdict until it has discriminated on a real one.

**A round ends inside its cap (#1546, 2026-09-26).** From 2026-09-24 20:07Z
every round was cancelled at `max_duration_seconds=1800` mid-matrix, retried
three or four times, and quarantined: 8 rows, 32 runs, 16 h of the primary, no
report and no measurement, since a killed round writes only `run_spec.yaml`. The
cause was `fa8a5bac` (#885), which routed the `requires_runtime` bench tasks
through the agent loop. A round grew from 91 trials with 1 such task to 144
trials with 5 (8 variants × 5 tasks, ~110 s a trial, one at a time: ~75 min).
The direct arm's 104 trials take ~80 s. Nothing bounded the round: the payload
carried a hard-coded `budget_minutes: 60` that `run_round` never read, and the
`bench_limit` / `max_variants` / `max_parallel` knobs had no way in from config.

- **The budget is derived from the cap.** `budget_minutes_for` takes the pool
  cap in whole minutes, lowered (never raised) by a payload or config
  `budget_minutes`. It is applied at enqueue, and again at execute, because a
  queued row can predate the cap.
- **The round keeps a clock** (`run_round.trial_deadline`): trials must be over
  by budget − max(20 %, 180 s), which leaves the judge and the report their
  share. Both runners start no trial with too little time left (20 s direct,
  90 s agent loop), cut a trial that starts near the deadline to the time left,
  and flag a trial cut that way `deadline_cut`. They walk the matrix task by
  task, so a stopped round has every variant measured on the same tasks.
- **A stopped round is a result, not a failure.** `complete_matrix` drops any
  task a variant missed or had cut, for every variant at once; the rest is
  judged, ranked, ledgered and reported, and the report says `stopped at
  deadline: yes` and names what was dropped. The round returns normally, so the
  pool records it completed and does not re-run the same work. It **never
  promotes** (every decision is `deadline_stopped`, HOLD), never compares or
  restores, and writes no `round_summary` row: a baseline mean over fewer tasks
  is not comparable. Only reports that promoted are ever read back, so its
  report cannot become a reference either.
- **Sized to finish — and it still does not.** `max_variants: 3` in the source
  block and `autoresearch.default_budget_minutes` for a hand-run round
  (`run_spec.yaml` records `budget.budget_minutes`) were meant to fit the matrix
  inside ~20 min. Measured 2026-09-27 they do not: all five rounds ran 1480–
  1537 s and stopped with 2 or 3 bench tasks never reached, which is precisely
  the state that writes no `round_summary` row and promotes nothing. `909476a5`
  (#1605, 2026-09-27) projects the trial matrix against the deadline *before*
  starting trials, so a round picks a matrix it can actually complete; no round
  has run since it landed, so the fix is unmeasured and this line is the thing
  to re-read next pass.

`tests/test_autoresearch_deadline.py` pins every link.

---

## 5. Intake — the outside world in

Two sources whose input is text nobody in this system wrote: a YouTube
transcript and a fetched web page. Both run one session turn per unit of work,
both write a vault note under `knowledge/`, and both arrived at the same three
rules — independently, which is the sign the rules belong to the family rather
than to either job:

- **The input is untrusted, so the turn's reach is a declared set rather than
  whatever nobody thought to ban.** Since #2269 (2026-10-06) each source names
  the capability groups its own output needs (`SOURCE_CAPABILITIES`,
  `app/harness/capabilities.py`) and the envelope hands the turn that expansion
  as `allowed_tools`, so anything absent from it is refused before dispatch:
  23 tools of the live 147-tool pool for `youtube-digest`, 20 for
  `deep-research`, in `tests/fixtures/worker_capability_baseline.json`,
  recomputed from the compile on every run of
  `tests/test_worker_capability_allowlist.py`. The per-source `DISALLOWED`
  tuples are still passed and still name refusals — the paper child's egress
  half is subtracted that way, which is the one thing a declared group cannot
  do — but they are the second gate now, not the policy. Neither turn reaches
  the automod tools, the queue writers or the autonomy writers, and
  `deep-research` has no `READ_FILES` at all: no `Bash`, `Read`, `Write`,
  `Edit`, `Grep` or `Glob`, its note going out through `vault_write`. What the
  example attack does *not* close is worth stating, and since #3524 its two halves
  have different answers. **The read half is closed** — by a rule that is not a
  capability: `PROTECTED_READ_ROOTS` (`app/harness/protected_paths.py`, the one
  owner of the table) names the credential files no digest, distill or research job
  has a reason to read (`~/lloyd/.env`, `~/.ssh`, `~/.config/gh`, `~/.openclaw`),
  and `app/harness/secret_read_deny.py` arms a `PreToolUse` deny over `Read`,
  `Grep` and `Glob` for a turn whose source is in
  `capabilities.UNTRUSTED_INGEST_SOURCES` — these two plus `session-distill`. The
  envelope is deliberately unchanged by it: the video turn is still 23 tools with
  `READ_FILES` in it, because the measured census says the job needs the reader
  (2947 `Read` calls over the 458 retained transcripts, and 2666 of the 4217 read
  calls land outside the bundle directory — `lloyd/eval/lloyd_profile.md` alone
  399), so dropping the group would have stopped the job and bounding `Read` to the
  bundle would have broken it. **The egress half is still open**: both sources
  declare `WEB_READ` (`http_search`/`http_fetch`), `deep-research` additionally
  holds `BROWSER_READ`, none of those three verbs is in `TIER2_TOOLS` or
  `TIER3_TOOLS` so the outbound content gate never arms for them, and
  `harness.egress_policy` runs with `enforce: false` — a destination is recorded,
  never refused. `attacker.example/?k=…` is therefore still reachable from both
  turns, and stays so until a human flips that flag (#628 closed with the flip
  explicitly delegated; `deep-research`'s whole job is fetching arbitrary hosts, so
  the flip needs an allow list widened around it). It is owed, not open work. One
  further deliberate limits: the read deny is armed per source, never fleet-wide —
  an attended turn reads `~/.ssh` legitimately (the `github-auth` skill), so the
  same rule there would be an operator outage rather than a fix — and it is a rule
  over three *path-taking* verbs. `session-distill`, the third member, also holds
  `Bash`, whose command string no path rule reads (that is the fixture's
  `untrusted_ingest_sources_with_unmeasured_bash` residual, and `--ro-bind` makes a
  credential un-writable, not unreadable), so for that one member a shell read of the
  same file remains available and rides on the egress ruling above with it.
- **Disk decides, not the turn's own text.** The source computes the note path,
  puts it in the prompt, and checks the file afterwards. A confident `RESULT`
  block over an empty directory is a failed attempt, not a note — and supplying
  the path is what makes the check possible at all.
- **The retry lives outside the queue, but the SPACING does not, and since #1714
  it cannot be left with nothing.** The pool records an in-band
  `{"status": "failed"}` and then calls `mark_completed` **regardless** — only a
  *raised* exception reaches `mark_failed` and the queue's backoff. So a source
  that returns `failed` and expects a retry does not get one from the queue. Each
  of these two keeps its own registry of attempts (`seen.json`, `research.db`) and
  decides there when to offer the work again. This is the part most likely to be
  got wrong by someone reading `pool.py`. What that rule used to leave unstated is
  the case where the source decides *nothing* — `youtube-digest`'s infra-shaped
  branch deliberately touches neither registry (no `--fail`, so `seen.json` records
  no attempt against a video that never got a turn), which left the re-offer
  spaced by the source's `interval_seconds` alone: 300 s, forever, with a fresh
  row and `attempts=1` each time. A source in that position now returns
  `defer_seconds: <n>` alongside the `failed`, and the pool releases the row with
  `mark_completed(item_id, not_before=now+n)`: the run is still recorded, the row
  goes back to `queued` behind that timestamp and **keeps its `dedup_key`**, so the
  next tick's `enqueue` coalesces against it instead of minting a row. For
  `youtube-digest` that n is `INFRA_DEFER_SECONDS` (900 s — the scanner's own
  `RETRY_INTERVAL_SECONDS`), and the ceiling is the queue's ordinary
  `max_attempts` (3), because `claim_next` raises `attempts` on every claim: a
  video that keeps losing its turn to the engine is offered three times, at least
  900 s apart, and `_poison_at_cap` then ends it at `state='poisoned'`, which
  the queue-maintenance pass moves on to the terminal `quarantined` through
  `WorkQueue.quarantine` (`workers/maintenance.py`), `poisoned` being reserved
  for "needs attention" — so `quarantined` is the state a capped video is
  actually found in (`workers.db` held two `youtube-digest` video rows there on
  2026-10-28, each at `attempts=3`). The
  no-penalty half is unchanged — `seen.json` still learns nothing from an infra
  turn, so a video the engine never served is not charged for it either way.

A fourth follows from the third: because each owns its own retry, each can
afford to treat **a `DrainActive` as not the work's fault**. The backend landing
a code update while a turn was starting releases the item with no penalty and
reports `skipped`, so it is offered again on the next tick rather than counting
as an attempt.

### `youtube-digest` — one video, one session, one verdict

**Wakes** every 300 s and refills the queue to `batch` (3) videos, keyed
`youtube-digest:<channel>:<video_id>`. `batch` bounds what sits in the queue,
not what gets done, and channels are **interleaved** so a channel with 280
videos in its window does not push one with 55 to the end of the day. Each tick
also runs `--register-new` to pick up new uploads. **Executes**
one session turn, Inner Voice off: read the fetched transcript bundle, write the
vault note under `~/obsidian/knowledge/youtube/<Channel>/`, judge the video
against `eval/lloyd_profile.md`, and file a draft backlog item when something
there would improve Lloyd.

The split is the design. `scripts/youtube_channel_monitor.py` is the
deterministic half — the `CHANNELS` registry, `seen.json`, listing uploads,
fetching the transcript and metadata into a bundle. This source is the
judgement half.

- **The script owns the retry, not the queue.** A failed session is reported
  back with `--fail`, which counts the attempt in `seen.json`, and
  `_is_retry_eligible` decides when `--pending` offers it again. A
  `DrainActive` is not the video's fault: the row stays `fetched` and is offered
  next tick.
- **Disk decides.** The note must exist at the path the source chose and carry
  the video's id, or the turn failed however confident its text reads. A
  `FILED: #n` claim is checked against `~/obsidian/backlog` before it is
  recorded; an unverifiable id is kept as `filed_unverified`.
- **The transcript is untrusted text**, so the turn's envelope is the narrowest
  declared set in the family: `READ_FILES`, `READ_VAULT`, `READ_MEMORY`,
  `READ_FACTS`, `TURN_STATE`, `WEB_READ`, `FILE_NOTE` (`Write` + `vault_write`),
  `FILE_BACKLOG`, `FACT_WRITE`, and the two `djev_*` deciders — 23 tools, and
  the whole browser surface is absent because `BROWSER_READ` is simply not
  declared. No shell, no `Edit`, no subagents, nothing that seeds queues or
  touches the self-modification loop. `Read`/`Write` stay because the note is
  the job — which also means `Read` reaches any file under `$HOME`, the residue
  #3524 is about.
- **Priority 45, not the source default 60.** `scheduled-task` may hold both
  slots with low/background work; at 60 three digests sat queued 25 minutes
  behind two low-priority KG jobs. At 45 a digest beats those for a slot and an
  `autocode` round (40) still goes first.
- The **eval rule is Alan's** and lives in the prompt: an open-source
  framework, tool or model may be proposed for direct adoption; a commercial
  product never is, only the aspects worth recreating locally are named.
- Its failure population is a closed burst, and the sentence this line carried
  before #3527 published it as a standing rate. Read 2026-10-28 over
  `/api/workers/health?days=7`, unclamped (`window_clamped_to_hours: null`,
  `oldest_input: null`, so a real 7 days): 100 runs, 51 failed (50 `failed` + 1
  `interrupted`), 5.96 h GPU — 51 % of that window, but a rate whose failure
  population ran 2026-10-18 to 2026-10-23 and had been empty for five days
  by the read. The payload could not say that, so the number was read as a live
  defect and #3523 was filed at `high`; the per-source block now carries
  `last_failure` and `consecutive_failures` beside `last_completed` (#3527),
  which is what dates a failure population without opening `workers.db`. Where
  the 51 sit: 9 on 2026-10-21, 25 on 2026-10-22, 14 on 2026-10-23, 3 on
  2026-10-26; 47 carry `meta_json.handoff_refusal` with a summary of
  `turn ended (stop, N iterations) without a note at <path>`, N between 3 and
  13, which is the two-stage `stage-a-reader`/`stage-b-writer` pilot's own
  signature — 79 such rows lifetime, 2026-10-18 to 2026-10-23, and none
  since `2a5778b6` reverted that path (committed 2026-10-23T18:13Z). The other
  four are not the pilot: one `ConnectTimeout` on 2026-10-23, one `interrupted`
  at pool start, and two `turn produced nothing` for `kRJjSOYqW0I` on
  2026-10-26 whose detail is the backend answering `All connection attempts
  failed`. What this line claimed before that is still refuted: **0** are the
  `--fetch rc=2` argparse burst, last seen 2026-09-09, that it called the whole
  of its 92 failures. And the retry design below is working as described — the
  burst cost GPU and latency, not the digests. `uTrY8slys0w` needed 8 attempts
  (7 failed, then a success) and its note is on disk from the successful
  attempt (vault commit `470a9aae`, 2026-10-23T16:27:46Z); `laQ8wMysnkk` and
  `STRtnhlf9CI` each have exactly one success run and a note; the two rows that
  reached `quarantined` (`queue.id` 10883, 15975, both at `attempts=3`) were
  re-offered on fresh rows and digested, `16031` succeeding at `attempts=1`.
  Exactly one confirmed permanent loss over the burst that ran 2026-10-18
  through 2026-10-23: `YvQ4BVTWfHI`, 12 failed / 0 success on 2026-10-21 and
  2026-10-22, with no note naming that id anywhere under
  `~/obsidian/knowledge/youtube/`.

**A cited paper is a second kind of row** (`kind="paper"`, keyed
`youtube-digest:paper:<arxiv_id>`), added 2026-10-09. Until then the script
looked for arXiv ids in the transcript only and wrote an abstract stub per hit;
counted over the 517 bundles on disk that day it had found none and written no
note, because a link is written in the description and never spoken — 21
descriptions held an id, and most of Discover AI's name the paper by title and
authors with no id at all. Now:

- **Registration is the script's, twice.** `--fetch` scans the transcript
  *and* the description and registers each arXiv id it finds. After the video's
  turn the source passes the `PAPERS:` line of the RESULT block to
  `--paper-register`: ids as they are, titles through an arXiv title search
  that accepts only a near-exact match (ratio 0.92, or a subtitle-less prefix of
  four words or more) — a wrong paper filed under a video is worse than none,
  and what does not resolve is recorded as `papers_unresolved` on the run. Four
  papers per video at most.
- **One registry for every channel**, `~/.local/share/youtube-papers/papers.json`
  with its bundles beside it. A registered paper gets an abstract-only note at
  once (`digest: abstract`) at `knowledge/papers/<id>-<slug>.md`, so the video
  note's wiki link resolves before the paper is read; its `## Discussed in`
  section lists the videos, which is the only link for a paper the session named
  after the video note was written.
- **A note with no `digest:` key is someone's hand-written note**: registered as
  `completed`, never queued, never rewritten.
- **The read is a session**, one paper per tick ahead of the videos (the
  channels never run dry, so a paper offered after them would never get the
  room). `--paper-fetch` puts the full text on disk — arXiv's HTML rendering with
  the bibliography dropped and formulas as TeX, `pdftotext` over the PDF when
  there is no HTML — and the turn follows the `paper-digest` vault skill:
  Summary, Key Findings, Method, Results, Limitations, Relevance to Lloyd,
  Discussed in. It files nothing (`PAPER_DISALLOWED` is the video deny list plus
  `backlog_write_task` and `fact_add`), and because the text is already on disk
  the whole egress half goes as well — `TURN_ENVELOPES["paper-note-turn"]` adds
  `_EGRESS_DENY`, taking `http_search`, `http_fetch` and every browser verb, so
  this is the one turn in the family with no route to the outside: it reads a
  file and writes a note. That subtraction is a deny on top of a declared group,
  which is the one thing an allow-list cannot do itself.
- **Disk decides, more strictly than for a video**: the note must carry this
  paper's `arxiv_id`, `digest: full`, a `## Discussed in` tail and more than a
  placeholder's bytes. The placeholder says `abstract`, so `full` is proof this
  turn wrote it. A failure is counted in the registry (four attempts, 900 s
  apart); a drain, an empty turn and a missing skill are not.
- `workers.sources.youtube-digest.papers.enabled: false` stops both halves.
  `--papers-backfill` registers the ids that completed videos' bundles already
  name; it cannot see a title-only citation, which needs a re-digest
  (`--since-days N --requeue`).

### `deep-research` — one registry topic through the deep-dive skill

**Wakes** hourly, claims one topic from `research.db` under
`deep-research:<topic_id>`, bounded by `daily_max` (3) — the registry can hold
40 queued topics and the generator proposes 5–8 a night, so without a per-day
bound one night's proposals become one day of GPU. That is the per-day ceiling;
`LONG_LIVED = True` is the per-moment one, because a research turn is 14–29
iterations each re-submitting a 120k+ context, which is precisely the shape that
comes back to a cold re-prefill under pressure. **Executes** one session turn
running the vault's `deep-dive-research` skill. **Writes**
`knowledge/research/<date>-<slug>.md` and records the outcome back on the topic.

It replaced `domain-research`, and the difference is not the storage: that
source asked the model to write a knowledge note from `vault_recall` alone — its
prompt never mentioned searching, though the turn had `http_search` and
`http_fetch` the whole time. Over its life, 142 notes, 90 of them empty, five
citing any URL, none ever promoted. The skill this one runs is the one that
produced the 102 notes people kept.

**Disk is the source of truth for `written`.** The source computes the note
path, puts it in the prompt, and checks the file afterwards; a file under
`_MIN_NOTE_BYTES` (400) does not count, because a lone heading is what a turn
that gave up leaves behind. Three consequences:

- A claim of `written` with nothing on disk is a failed attempt, not a note.
- A note on disk with no `RESULT` block is `written` anyway — the model just
  did not sign off, and retrying would produce a second note.
- A note left by an attempt that died before recording is recovered on the next
  claim without spending a turn.
- A verified note without `segment:` gets `segment: knowledge` inserted as one
  line after `type:` (`ensure_segment`, #1167) — the skill's template is vault
  text a skills pass can rewrite, and on 2026-09-23 two notes came out keyless.

**The source picks the filename, not the model.** The skill used to say "run
`date +%F` via bash, never guess it" — a workaround for a problem that existed
only because the model was choosing, and it produced notes misdated by days,
some dated in the future. Supplying the path is also what lets `written` be
verified against disk, and it is why `Bash` is on the deny list.

**Retries live in the registry, not the queue.** `store.release(error,
backoff_seconds)` puts the topic back with a `not_before` and this source's
`interval_seconds` is the retry cadence; after `workers.max_attempts`,
`exhaust()` settles it `nothing_found` with the reason recorded. Without that
last step the failure is a cycle rather than a retry — `enqueue_if_due` would
offer the same topic every tick forever.

**A `DrainActive` is deliberately not a failure.** The backend landing a code
update while the turn was starting releases the topic with **no** backoff and
reports `skipped`, so it is claimable on the very next tick instead of sitting
out the hour. A landing is not the topic's fault.

**Inner Voice is off**, and it is a setting rather than a literal only since
2026-09-10. A research note is read by a human before anything acts on it, and
this source runs often enough that observing it would put a goal extraction plus
a critique per turn in front of chat all day.

**The turn fetches arbitrary web pages, so its reach is declared rather than
inferred from a deny union** — and before this source, no session-backed worker
passed a list at all. `run_prompt_on_primary` bakes the automod ban into its own
`RunOptions`, and `/api/message/stream` used to build `disallowed_tools` from
config plus whatever the request body named, so a worker session was handed
exactly a chat's toolbox (the endpoint also reads `platform`, but only to arm the
#534 grant gate and ban `grant_create`, which gates tier-2 and tier-3 tools and
nothing a research turn uses). Since #2269 the ceiling crosses the loopback POST
as `allowed_tools` too, so the live envelope is `envelope_for` of
`SOURCE_CAPABILITIES["deep-research"]` — 20 tools — and the source's
`DISALLOWED` tuple is the floor beneath it. The turn cannot reach `Bash`, `Read`,
`Write`, `Edit`, `Grep`, `Glob`, `Task`, `http_request`, the browser mutators,
the task boards, the autonomy writers, the automod tools, or the registry's own
writers. It keeps `http_search`, `http_fetch`, `browser_navigate`, the vault
readers, `vault_write`, `fact_add`, `backlog_get_task` and `graph_explain`, which
are the job. The "read `~/lloyd/.env`" half of that threat is genuinely closed
here — there is no file-reading verb in the set at all — the "navigate to
attacker.example/?k=…" half is not, because `WEB_READ` and `BROWSER_READ` *are*
the job: what cannot leave is only what the turn could not already see. A
`vault_write` landing outside `knowledge/` is caught by a `git status` **diff
against a snapshot taken before the turn** (`_unexpected_vault_writes`) and
recorded on the topic — a detector, never a revert; the first cut read the
post-turn status alone and reported the scheduler's ordinary churn as twenty
"unexpected writes". Since #3190 (2026-10-23) one shape is refused outright
rather than detected: a card declaring `pipeline: research` may only be written
to `knowledge/research/`, guarded through `scripts/vault/research_route.py` on
both write surfaces (`agent_mcp/vault.py`, `agent_mcp/builtin_fs.py`).

**Correlation.** The topic row carries `queue_id` (the `workers.db` item that
researched it) and `session_id` (the transcript); `runs.queue_id` joins back to
the run record, the same recovery `autonomy.compute_health` uses. The pool never
passes its `run_id` into `execute`, so the queue id is the link.

Two accounting limits, both noted because they live somewhere nobody would look:

- **A tick that never researched anything still spends an attempt.** `claim`
  increments `attempts` before the turn runs and `release` does not roll it
  back, so a topic handed straight back for a landing, or held by a backoff, is
  one attempt nearer the `exhaust()` that the *next* real failure triggers. Not
  seen in practice — a drain window is minutes against an hourly cadence — and
  the accounting is in `claim`, not in the failure path.
- **There is no UI.** `research_stats` and `research_list` from a chat, or
  `sqlite3`. The Workers page's Recent Runs shows what the source did, because
  the run summary names the topic and its outcome.

The registry it drains, its seven states and the two producers that fill it:
[[research-pipeline]].

---

## 6. Mining — Lloyd's own exhaust back out

Two sources that read what this system already produced — finished chats and
failed autonomy runs — and turn it into a note a human promotes. They are the
same program twice, and that is worth saying once rather than twice:

- **A direct turn on the primary** through `run_prompt_on_primary`. No session,
  no Inner Voice, no transcript anyone reviews.
- **A turn budget read from `src_cfg` and carried in the queue payload.** Both
  started with a literal at the call site (15 and 8); `bench-mine` got the key
  first (#896, 2026-09-24) and `session-distill` followed (#1460, 2026-09-25),
  so today `workers.sources.session-distill.max_turns` (15) and
  `workers.sources.bench-mine.max_turns` (12) are the only way to move either,
  over code defaults of 15 and 8. An item runs under the budget it was enqueued
  under (`_turn_budget` reads the payload, not the live config).
- **`_common.write_staging_note(source=NAME, …)` at the end**, which fixes the path to
  `pending-research/<source>/<date>/` and stamps `review_status: pending`.
  `bench-mine` then overwrites that field itself: `_record_calibration` writes
  `pending` or `out_of_band` from the edge verdict, so for that source the
  staging helper's stamp is only the initial value. The
  Review tab is what promotes them; only `bench-mine` has a default destination
  in `_DEFAULT_DEST`, so for `session-distill` a human must name where it goes.

- **A bound on the pile, not on the clock: `max_staged_pending`, under
  `workers.sources.<source>` (#3530).** `write_staging_note` writes for a human, and for the family's whole
  life nothing bounded a producer against how much of that nobody had read:
  measured 2026-10-28 the tree held 624 notes (bench-mine 519 — 206 `pending`,
  194 `out_of_band`, 119 `uncalibrated`; `session-distill` 105, all pending)
  while `~/obsidian/lloyd/bench/` held 28 files, **none** carrying the
  `review_status: promoted` + `promoted_at` pair the promote route stamps, and the
  reject lane had never run either — zero promotions
  and zero rejections,
  against 13.87 GPU-hours in `bench-mine`'s 7-day window. `interval_seconds`
  bounds frequency and `max_enqueue_per_tick` bounds one tick; neither could see
  the queue. So `enqueue_if_due` in both sources now asks
  `_common.staged_pending_count(source)` first and returns `DECLINED` — offering
  from **neither** input, and moving **no** watermark, so a drained pile re-mines
  the same slate — while that count exceeds the cap: code defaults **150** and
  **60**, deliberately below the 206 and 105 pending then on disk, so the bound
  bites the day it lands and an absent key is not an open door. The skip is a
  named outcome (`skip_reason=staged_pending_cap`, one INFO line with the count
  and the cap, no run row), and the same figure the bound reads is published on
  `/api/workers/health` as that source's `staged_pending`, which is why the count
  lives in `_common` rather than in either source.

  The count is `review_status: pending` **alone**, and that is the whole design:
  the 313 `out_of_band`/`uncalibrated` rows are notes somebody already labelled
  droppable, so a cap over every file could never be cleared by promoting and
  would have been a permanent halt wearing a throttle. The values themselves are
  owed a ruling — 150 and 60 are positions, not measurements, because a number
  describing how much backlog is tolerable needs a consumer to be measured
  against, and to that date the lane has taken none.

The shared helper is also why **both docstrings used to name a staging
directory that does not exist** — `distill/` and `bench/<date>/`, from before
the path was centralised. Harmless, and the kind of drift that makes a grep for
the real path fail; `tests/test_research_doc_claims.py` pins the real leaf now.

This was the family that did not work: measured over the 7 days to 2026-09-11
it failed 42% and 82% of its runs, both at `max_turns` with nothing written, and
§2 carried those numbers in the present tense until #1713 re-measured that
section on 2026-09-28. The two config keys
above are what closed it. Over the 7 days to 2026-09-28 `session-distill` is 28
ok of 29 runs and `bench-mine` 118 ok of 163, and of `bench-mine`'s 32 failures
30 are one `ConnectError` burst on 2026-09-24, not cap deaths — 2 runs in the
whole window died at the ceiling with nothing written. The shared shape is still
the diagnosis, but the defect that shape now shares is not the budget: it was a
`done:` marker the ledger input wrote and never read (#1711, landed 2026-09-28 —
that input now reads the same watermark set the failed-runs input always did),
beside a calibration gate that scored the staging envelope instead of the
candidate. Both are now closed: #1710 landed 2026-09-28 05:02Z, minutes after the
readings stamped below were taken, and #1769 relabelled the notes the old route had
already mis-stamped. What is still open in `bench-mine` is the cost, not the gate.
The family's third member, `gap-fill`, never ran at all and was retired on
2026-09-24 (§7).

### `session-distill` — mine a finished chat for patterns

**Wakes** every 1800 s, scans `~/lloyd-data/sessions/*.json` (`app.paths.SESSIONS_DIR`
— the data root, not the code tree) and enqueues one item
per eligible session.
**Executes** a direct turn on the primary (`workers.sources.session-distill.max_turns`,
15 since #1460 landed 2026-09-25; the code's `DEFAULT_MAX_TURNS` is 15 too, so the
key exists to move it, not to hold a different value)
and writes findings to `pending-research/session-distill/<date>/`.

A session is distilled **once**, **after it goes quiet** (30 min since the
file's last write), **not while a turn is running in it**, and **only if a user
wrote it**. Each gate is scar tissue:

- *Once.* Selection was keyed on an `mtime > last_mtime` watermark, and a
  session's mtime advances with every message — so an active chat re-crossed it
  on the very next tick, forever. One chat was distilled 44 times, another 22,
  and the worst eight account for 138 runs. Per-session `done:` markers replace
  the moving cursor, and fix its mirror image too: a session skipped for being
  busy used to be stranded below the advanced watermark and never looked at
  again.
- *Quiet, and no turn running.* Two gates, not one: a turn that has been
  thinking for forty minutes has a file older than the quiet window and a
  conversation that is not finished.
- *A user session.* `platform: worker` and `platform: autonomy` sessions are
  this system talking to itself; 12 of Lloyd's own triage transcripts had been
  mined back in as observations about the user before this gate existed.
  `sessions_io.NON_USER_PLATFORMS` is the one definition.

**Current state (7 days to 2026-10-28, `/api/workers/health`, window unclamped):**
12 ok of 13 runs, 1 failed, 0 skipped, 0.57 GPU-hours. The shape is unchanged from
the 7 days to 2026-09-28 — 28 ok of 29, 28 notes staged, 1.3 GPU-hours, its one
failure not a cap death but an empty turn that stopped under the ceiling
(`stop_reason=stop`, 14 of 15). 105 notes sit staged on 2026-10-28 and every one is
still `review_status: pending`: the distiller works and nothing has read what it
produced. The 154-of-369 figure §2 carried before #1713 re-measured it is the
window to 2026-09-11 and no longer reproduces.

### `bench-mine` — new bench tasks from failure signal

**Wakes** every 7200 s with two deliberately independent inputs, each capped at
`MAX_ENQUEUE_PER_TICK` (3): `enqueue_if_due` resolves
`workers.sources.bench-mine.max_enqueue_per_tick` once and passes that one
number, with `FAILURE_WINDOW_DAYS` (7 days), to both selectors, so the key
bounds every item the source offers. Until #1712 it reached the failed-runs
selector alone, and the ledger input — the one that actually fires, 138 of the
139 `bench-mine` queue items all-time to 2026-09-28 — ran at
`_recent_ledger_losers`'s own default `limit=5`, overrunning the declared
budget on 24 of the 36 enqueue ticks in that window. The inputs: **failed autonomy runs**
(`autonomy-runs/**/run_*.md` with `status: failed`, 7-day window) and **ledger
losers** (bench tasks the baseline scored under 0.6). **Executes** a direct turn
on the primary (`workers.sources.bench-mine.max_turns`, 12 since 2026-09-24;
the code's default is 8, and the value rides the queue payload like the
other session sources' — #896) and stages a candidate task under
`pending-research/bench-mine/<date>/`, which a human promotes into
`~/obsidian/lloyd/bench/`.

- **#522 found it enabled, registered, and never once enqueued** while both
  inputs were wide open — the failed-runs input was advertised in the docstring
  from the day it was written and the directory was never opened. What that
  directory held when the input landed is stated once, with its date, in
  `architecture/workers.md` §8; it is not repeated here, because the sweep that
  prunes `autonomy-runs/**` past 30 days has made that reading unreproducible
  and a copy of it on a second page could only go staler, not truer.
- **The ledger input can go quiet for days** — only an autoresearch round
  appends to it — and had a defect of its own: the writer mints `BASELINE_<int>`
  while the filter matched lowercase `baseline`, a case-sensitive comparison
  that never held, so the selector returned nothing for the ledger's entire
  life. #625 made it case-insensitive and restricted losers to trials whose
  `trace_status` is `success`, since `judge_trace` in
  `scripts/autoresearch/judge.py` scores an incomplete trace
  0.0 and every errored baseline row is therefore a "loser" for harness rather
  than model reasons. The input fires now: 1,678 ledger rows and 238 baseline
  rows on 2026-09-28, 38 usable losers inside the 7-day window, and 132 completed
  queue items of kind `mine` in that window against 0 on 2026-09-19. It fires too
  often. 128 of those 132 items are one task — `bench_007_skill_invocation`,
  baseline 0.05 — offered 24 to 29 times per round across 5 rounds, because
  `_enqueue_ledger_losers` did not consult the `done:` markers this source writes
  (all 10 of those `done:ledger:*` keys existed, the earliest of them before most
  of those runs) while `mark_completed` released the dedup key behind it. The
  failed-runs input always read that watermark; the ledger input was the half of
  the module that skipped its own "a failure corpus is an infinite loop without
  markers" rule, and #1711 closed that on 2026-09-28: it now builds the same
  `done:` set and skips a `(task_id, round_id)` pair already retired under
  `_ledger_key`. What that fix did not do is apply the test before the slice:
  marked rows still spent the per-tick budget, so a tick whose head of the slice
  was marked offered nothing while unmined losers waited behind it. That is what
  the ruling on #1711's owed entry decided against, and #1774 applied it on
  2026-09-29: `_recent_ledger_losers` now drops a retired pair inside the scan,
  before `rows[:limit]`, and reports how many it dropped. Measured the same day,
  the slice was three retired rows at composite 0.05 with three unmined 0.05 rows
  directly behind them, and the input had offered nothing for 20.9 hours while 44
  unmined losers sat inside the window. What #1774 did not move is the budget: the
  selector is still asked for `max_enqueue_per_tick` rows and the caller still
  asks once, so a scan that skips does not buy a wider slate.
- **Every candidate carries a `calibration` block** — N trials against the
  canonical prompt, and whether the composite landed strictly inside the
  capability edge. A task the learner always passes and one it always fails both
  move the bench mean by noise rather than signal; of the 11 tasks with baseline
  rows in the ledger, none sits at mean 0.00 on 2026-09-28, and the live corpus is
  28 files as of 2026-10-25, not the twenty this line used to count (filed). **It
  used to score the wrong document**: `_load_candidate` handed `load_bench_tasks`
  the staging *directory*, so the thing given 10 real trials was
  `write_staging_note`'s envelope — no `prompt`, no `objective_checks`, the layer
  the judge then awards full marks for. All 118 notes staged in the 7 days to
  2026-09-28 carried `status: ok` with an empty `error` and means spanning 0.51 to
  1.00, a perfect score ten times over among them, on candidates mined from a task
  the baseline scored 0.05. **#1710 closed that on 2026-09-28**:
  `calibrate_candidate` now takes the parsed candidate its caller already has
  (`task=`), `_load_candidate` is a fallback for a real task file only, and
  `_indiscriminable` returns a reason and spends **zero** trials on a dict with no
  `prompt` or no non-empty `objective_checks` rather than handing back a verdict on
  the wrong object. #1769 relabelled the 119 notes the old route had stamped —
  `review_status: uncalibrated`, `calibration.status: stale_envelope`,
  `measured_against: staging envelope (pre-#1710)` — so the
  145-`pending`-against-one-`out_of_band` split this line used to report is not the
  pile's shape
  either (measured 2026-10-28: 519 `bench-mine` notes, 206 `pending`, 194
  `out_of_band`, 119 `uncalibrated`). The gate is a measurement now, and it is
  still the expensive step: 10 real bench trials per candidate, 13.87 GPU-hours for
  the source in the 7 days to 2026-10-28 against 6.42 in the window to 2026-09-28
  (filed).
- **The human promotion step is the honesty gate.** Lloyd writing the tasks
  that grade Lloyd is the known self-grading failure mode; the human gate, the
  mechanical-check requirement, and mining from real failures rather than
  invented ones are what hold against it. As of 2026-09-28 the gate had never
  opened: none of the 20 live tasks then under `~/obsidian/lloyd/bench/` carried a
  staging provenance field, against 118 `bench-mine` candidates sitting staged —
  113 of them the same slug, `mined-from-bench-007-skill-invocation`. Measured
  2026-10-28 it has *still* never opened, and the ratio has moved the wrong way: 28
  live files (highest `bench_028`) against 519 staged notes. The one-slug pile is
  gone (430 distinct candidate ids, #1711 and #1774), so the duplication that closed
  is not why nothing lands, and the tab's own rails are not the blocker either — it
  refuses a candidate whose task block cannot be graded and refuses a stem the bench
  already declares, which is a working gate waiting for a human. Six weeks of mining
  and six weeks of nothing on the other side of it. The only consumer of this group's
  output is that tab. One cause of the pile's shape is now fixed (#3531): the mining
  prompt used to show `id: bench_XXX_<slug>` and leave the number to the turn, but the
  number a turn can read advances only when a human promotes, so every candidate since
  `bench_028` answered `bench_029` and 324 of the notes staged by 2026-10-29 declared
  that one id. The number is now minted by `_mint_bench_id` at staging time — one past
  the highest held by the live corpus and the staging pile together — and the prompt
  asks only for a slug.
- The idea is copied from Terminal-Universe (arXiv:2609.04148): a recorded
  trajectory, *including the failed ones*, reconstructed into a task with a
  deterministic pass/fail check.

**Current state (7 days to 2026-10-28, `/api/workers/health`, window unclamped):**
101 ok of 123 runs, 12 failed, 10 skipped — a 9.8% fail rate for 13.87 GPU-hours.
11 of the 12 failures are `ConnectError: All connection attempts failed` on a
single day, 2026-10-26, with the primary unreachable, and 8 of the source's 9
quarantined queue rows carry that same error: an outage, not a source defect. All
10 skips are "no parseable bench-task frontmatter" — a turn that ran to the end and
returned a candidate the promote route would refuse — spread over five tasks
(`bench_027` ×4, `bench_024` ×3, `bench_023`, `bench_007`, `bench_006`), so the ~8%
that end unparseable is no longer `bench_007`'s quirk but a standing shape, and a
skip is charged its duration and appears in no fail rate. The window to 2026-09-28
measured 163 runs, 118 ok, 32 failed — 30 of those one `ConnectError` burst on
2026-09-24 and 2 genuine cap deaths — plus 13 skipped, 11 of them "no parseable
bench-task frontmatter" on `bench_007_skill_invocation`, for 6.42 GPU-hours; the
failure that mattered there was not in the failed column but 118 successes writing
113 copies of one candidate, the shape #1711 and #1774 closed. What is open on
2026-10-28 is the ratio: 2.2× that window's GPU-hours for zero promotions. §2's
former 49-of-60 was the window to 2026-09-11.

### `failure-ledger` — the same fault counted, and one run a day sent after it (#2079)

`app/failure_ledger.py` is the store; this source is its only caller. One pass is
ingest → sweep → detect → dispatch, and the two properties worth knowing before
touching it are both about not lying to the reader.

- **Ingest reads the rotated archive *and* the live ledger.** Over the committed
  fixtures the guardian's `Service down, but no promotion to revert` family is 18
  rows with `first_seen` 2026-09-06T18:34Z; the live file alone is 2 rows starting
  three weeks later. A store whose whole job is a `first_seen` that does not move
  cannot afford a reader that opens one file and invents an onset.
- **The daily cap is a row, not a variable.** `failure_dispatches` is keyed on the
  day, so a second dispatch the same day writes nothing, and the findings the cap
  held back stay in `failure_findings` with `dispatched = 0` — which is the figure
  that says how much was deliberately left uninvestigated.
- **Growth counts the day BEFORE the pass, is stamped with the pass's own day, and
  has two arms.**
  One sweep a day means the day a pass's clock names holds only the ~2 h before it
  woke — the three logged passes ran at 01:51:42, 02:03:27 and 02:38:29 UTC — so
  counting `ts[:10] == day_of(now)` looked at a slice too thin to contain the burst
  the detector exists for. Measured over the live store on 2026-10-17: of the 8
  signature-days that reached `GROWTH_MIN_COUNT` (3) across 22 event days, 6 had zero
  events before 02:40 UTC on their own day (2026-09-06 n=6, 09-09 n=3, 09-14 n=3,
  09-29 n=3, 09-30 n=11, 10-02 n=8), and an 83-event store held 0 rows in
  `failure_findings` and 0 in `failure_dispatches` (#2893). So growth counts
  `app/failure_ledger.growth_review_day(day)`, which is `day` minus
  `GROWTH_REVIEW_LAG_DAYS` = 1 — a CLOSED day rather than a "since the last pass"
  window whose start this store cannot verify without reading `workers.db`, which its
  header rule forbids — and the finding is stamped `detected_on` with the pass's own
  day, because `dispatch_findings` selects `WHERE detected_on = ?` on the day it is
  called with. Move one without the other and findings accrue with `dispatched = 0`
  forever while the pass logs a sensible-looking `0 finding(s)`-shaped summary:
  `tests/test_failure_ledger_dispatch.py` pins the pair from a frozen 02:10 clock
  through `enqueue_if_due` to the queue row.
- **What the counted day is compared against — two arms, and only one of them is a
  comparison.** Where the trailing `GROWTH_WINDOW_DAYS` = 7 window recorded events, the
  bar is `GROWTH_MEDIAN_MULTIPLE` = 2.0 times `median_active`, the median over ONLY the
  days that had an event (put the zero-days in and an intermittent family's median is
  0.0, which either trips on nothing or, guarded, never). Where that window recorded NO
  event at all, `median_active` is None and `GROWTH_MIN_COUNT` = 3 on the reviewed day
  is the whole bar — the burst-from-silence arm #3281 added, and the only burst shape
  this store has ever produced: measured over the ledger copy on 2026-10-25, 3 of the 9
  signature-days that reached 3 across 25 event days had an empty window
  (`04f90c87b03a` 09-25 n=23 and 10-21 n=8, `a1afbe2e3b04` 09-06 n=6), and the single
  day that cleared the 2x median (`9cd285a288f7` 09-29 n=3 against 1.0) is one no D-1
  pass can ever review. The pass that got 8 events of the silent shape wrote `0
  finding(s), 8 event(s) new, investigation queue_id=None`. An empty window is its own
  arm and NEVER a median coerced to zero — `median or 0` rates every family as
  quiet-then-growing — and a family that does have a rate is judged by the bar alone:
  09-30's 11 events against a median of 13.5 and 10-02's 8 against 11.0 both stay
  quiet. `tests/test_failure_ledger.py` pins both arms; `tests/
  test_failure_ledger_dispatch.py` carries the silent one through `enqueue_if_due`.
- **`last_enqueued` is the last sweep, not the last enqueued row.** The source stamps
  it unconditionally after `sweep_and_dispatch` returns, whether or not a dispatch
  happened, so it is the interval gate's clock and answers exactly one question — has
  this source swept — and nothing about the day's outcome: a day with findings and a
  day with none move it identically. The pool's own pair answers a different question
  and neither one is "a pass completed": `last_enqueue_check` is stamped on every ask
  whether or not the call returned (`workers/pool.py:1636`) and `last_enqueue_ok` only
  on a clean return (`workers/pool.py:1647`, read by `workers/dispatch_watch.py`), so
  both advance once per pool ask, per source. A fresh `last_enqueue_ok` therefore says
  the pool is still asking, not that this source swept: `enqueue_if_due` returns early
  inside its own `min_interval_seconds` of the last sweep
  (`workers/sources/failure_ledger.py:243`), and that early clean return stamps the
  pair like any other. The only stamp saying this source swept recently is its own
  `last_enqueued`, and it says no more than that: 0 `failure_dispatches` rows sit
  against a stamp that moves daily (measured 2026-10-30T00:28Z). Stamping a separate
  enqueue key on a real dispatch, or renaming this one `last_sweep`, is #3532's owed
  scope call.

The prompt names at most `DISPATCH_MAX_RECORDS` finding ids plus a window of
`DISPATCH_WINDOW_DAYS` around `first_seen`, and `execute` pastes the ledger's own
rows in above it: a worker turn has `Read` and `Bash`, not a reader for a second
sqlite file, so forwarding the prompt alone would send a run off to describe rows
it cannot open. `tests/test_failure_ledger_dispatch.py` pins exactly that — the
sample text it asserts on is in the store and nowhere in the queue payload. **That
turn is also the one in this section with no ceiling over it**:
`failure-ledger` has no entry in `app/harness/capabilities.py:288`
`SOURCE_CAPABILITIES`, whose keys are `session-distill`, `youtube-digest`,
`deep-research`, `scheduled-task`, `autotriage` and `bench-mine`, and
`envelope_for(source)` returns `None` for an undeclared name — which every
`workers/sources/_common.py` call site reads as "no ceiling": no `tool_allowlist`
reaches `turn_options`, and the empty `model` on `run_prompt_on_primary` leaves the
legacy unbounded path (filed).

**Registered, and scheduled as of 2026-10-06.** #2079 is measured on nothing yet, and
this is the half of the reason: `WorkerPool._scheduler_pass` reaches a source's
`enqueue_if_due` only through a block under `workers.sources`, and the module shipped
without one, so between 2026-10-03 and #2309 the sweep ran only inside its own tests.
Its block is in `config.yaml` now — `enabled: true`, `interval_seconds: 3600`,
`min_interval_seconds: 86400`, `max_inflight: 1`, `max_duration_seconds: 900` — and
what the wait cost was measurement, not mechanism: counting was proven against the
real ledger and the real guard logs before it, and this is what it had done
(measured 2026-10-03): the store counted 313 events across 30 signatures, the top
two families holding 80 and 41 occurrences — the two figures `sweep()` returns as
`{"signatures", "events"}` (`app/failure_ledger.py:395`), not `reconcile()`, which
reports queue depths and run outcomes and cannot count an event. That pair reopens
against no surviving state of the file: the #2893 witness
(`backlog/data/failure-ledger.db`, cut 2026-10-17T14:46Z) holds 83 events across 7
signatures, and the live store on 2026-10-28 holds 95 events across 10 signatures
over 25 event days, with `failure_findings` and `failure_dispatches` both still at
0 rows. Counting is what the sweep does
alone; acting needs the pool. **The denominator is `runs`, not that event count** —
`app/failure_ledger.py:670` `reconcile()` reports `runs_total` (`:701`) as
`SELECT status, COUNT(*) FROM runs WHERE source = ? GROUP BY status` run with
`source='failure-ledger'` against `workers.db` (`:690-696`, its own docstring at `:679`)
already names it "the denominator for 'how much did the ledger actually do?'"), and
`runs` rows are written only by `_insert_run()` (`_RUN_INSERT_SQL`) when a worker finishes
one. So `runs_total` is the one
figure here the thing being judged cannot produce, `runs.status` splits it into
outcomes, and `runs_total: 0` is not a quiet ledger: the events say what the sweep
found, and only a run row says anything read them. Before this module existed the
rows were worse still — the pool looked the source name up, missed, and marked each
one poisoned at `attempts = 0`, which is terminal.

**What decides it, and what was never the obstacle.** Seven consecutive days with a
`runs` row per day, then the before/after count of identical unanswered notices
against named, dated open issues: owed-check's ruling, read on the daily note and the
`[guardian]` board rather than the parked morning brief, and the #543 four-week
delete-or-keep clock for the ledger starts at that first real run, not at this block's
landing. **Measured 2026-10-30T00:28Z that first real run has still not
happened: 0 runs and 0 queue rows for `source='failure-ledger'`, both
all-time** — and the reason is quieter than the one this paragraph carried
until #3532: the store has produced no qualifying burst since the detector that
looks for one landed. The schedule is not the blocker either, and its stamps
are not the same fact: the pool's ask pair (`last_enqueue_check`,
`last_enqueue_ok` — each stamped once per pool ask, not once per sweep) read
2026-10-29T23:34Z, and the source's own sweep stamp `last_enqueued` read
2026-10-29T05:22Z, all three from one read, where the ask pair sits 53 min and
the sweep stamp 19 h before the `runs` count above. The pool has asked within the
hour and the sweep last ran 19 h ago; neither is a completed diagnosis — the ask
pair says nothing about whether this source swept, and the sweep stamp says
nothing about whether it dispatched. **An arrival is not missing.** A reviewed
day holding ≥3 events of one signature — `app/failure_ledger.py:482`
`GROWTH_MIN_COUNT = 3` — whose trailing-7-day window is empty fires the
silent-baseline arm in `detect_growth` (`:576`, in main since 2026-10-25), so
the store's next quiet burst of that shape arrives at the pool on the pass the
day after it. The rate is a fact about the store and not a hope: counted
2026-10-30T00:28Z, 9 signature-days reach the floor across the store's 27
event days (2026-09-06 → 2026-10-27: 27 days carrying events inside a 52-day
calendar span), roughly one a week. What has not arrived is that day *since
the arm*: the last qualifying day is 2026-10-21 (`04f90c87b03a`, n=8 counted
2026-10-30T00:28Z), four days ahead of #3281's settling (2026-10-25), and the
only event days after it — n=1 (2026-10-22), n=2 (2026-10-24), n=1
(2026-10-27) — all sit under the floor. Two facts about that 2026-10-21 burst
are the whole of the confusion, and both are in #3281's round record: the
scheduled pass DID review it, on 2026-10-22, and the pre-#3281 guard
`if median is None: continue` then discarded its empty window — the arm's
measured flip is `[]` at base `752f5ade` versus `[1]` after, replayed on a
copy of the store because `detect_growth` commits. What is permanently
unreviewable is bounded, and it is about days rather than about reach: because
the pass reviews only D-1 (`GROWTH_REVIEW_LAG_DAYS` = 1, `:503`) with no
catch-up, a burst that closed before 2026-10-25 can never be re-reviewed —
`04f90c87b03a` 2026-09-25 n=23 and `a1afbe2e3b04` 2026-09-06 n=6 among them —
so those went unreported and stay that way. The seven-day criterion and #543's
clock are therefore not early, and they are not stuck either: the live
authority on the residual zero is #3281's owed entry, which owed-check ruled
on 2026-10-27T03:03Z ("the source is live, not stalled … the zero is 'shape
not seen', not a defect") with its recheck owed 2026-11-03. What kept the loop
from scheduling the source was never a fence around
`config.yaml`: that claim was retired on 2026-10-05 by the
`scripts/automod/spec.py::config_value_change` lane, whose denied set
(`spec.py::CONFIG_DENIED_KEYS`)
covers the lock-out, the rollback path, the engine's identity and the loop's own three
worker sources, and no path under `workers.sources` belonging to anything else.

---

## 7. Retired and renamed

`runs` rows outlive the source that wrote them, so the table carries names no
longer in `SOURCE_REGISTRY`.

| in `runs` | now | family |
|---|---|---|
| `backlog-selfmod`, `backlog-implement`, `autoimplement` | `autocode` | self-mod |
| `selfmod-regression`, `autoimplement-regression` | `automod-regression` | self-mod |
| `domain-research` | retired 2026-09-08 → `deep-research` | intake |
| `gap-fill` | retired 2026-09-24, no successor | mining |

The renames all landed on 2026-09-09 and are history, not drift.
`domain-research` is gone from config.yaml, the registry and, since #1278,
`app/routers/workers.py::_DEFAULT_DEST`: the 142 staged notes that kept that
entry alive were lost with the 2026-09-22 data wipe, so there is nothing left
under `pending-research/domain-research/` for the Review tab to promote.

`gap-fill` (#897) leaves nothing behind: it resolved facts carrying
`label: gap` or `provenance: GAP`, and no extractor ever emitted one, so in
the life of `workers.db` it had zero `runs` rows and zero staged notes while
stat-walking the ~70k-file facts tree every 300 s. Giving it a producer meant
adding a field to the fact schema, not pointing the scanner elsewhere, so it
went the way `domain-research` did: module, registry entry and config block
gone, nothing to keep readable. The wikilink
`[[project_gap_fill_event_loop_freeze]]` in `bench_mine.py` names the incident
that put every disk-heavy scan on a thread, and stays.

---

## 8. Where the long versions live

| family | job | doc |
|---|---|---|
| dispatch | `scheduled-task`, and the fleet it runs | [[autonomy]], [[autonomy-jobs]] |
| self-mod | `backlog-cluster`, `autotriage`, `autocode`, `automod-regression` | [[automod]], [[backlog]] |
| self-mod | `arch-review`, and the picklist it walks | [[arch-review]] |
| intake | the registry `deep-research` drains | [[research-pipeline]] |
| all | every job's transcript and recording | [[background-runs]] |
| all | the queue, gates and contracts they share | [[workers]] |
| all | the KV gate's measurements | [[vllm]] |

## Review log

- 2026-09-25 — **stale**: the doc's measured window (09-11) had drifted —
  `autoresearch` was re-armed 2026-09-16 (#876 restored variant benching; runs
  every 4 h behind the promotion floors, 11/11 success), `automod-regression`
  polls 900 s not hourly, `autocode` is `max_inflight: 2` not 1, the gate is
  eleven rungs not nine, `arch-review`/`autotriage`/`youtube-digest` run with
  Inner Voice **off** (config says so; the prose said on, against the doc's own
  roster), the fleet is 36 task files not 32, `bench-mine`'s turn budget is a
  config key since #896 (only `session-distill`'s remains a literal), and the
  `eval/durable_write_judge/judge.py` citation is dead — the zero-score is `judge_trace`. Findings
  outside this doc filed as #1458 (stale automod-regression config comment),
  #1459 (dead judge.py citation in `bench_mine.py`), #1460 (session-distill
  budget has no config key).
- 2026-10-25 — **stale**: §1 disagreed with itself and with `config.yaml` —
  the roster said `autocode`/`autotriage` IV **off** while §4 and config say
  `inner_voice: true` (re-armed 2026-09-25), and `cost-investigate` said on
  **no** though its block shipped `enabled: true` in #2510. `LONG_LIVED`
  declarations are 9 not 5 (True on 5, False also on `failure-ledger` and
  `frontend-probe-canary`); the gate is thirteen rungs not eleven; the fleet is
  45 task files not 36; the picklist is 48 units not 42; the board-steward flip
  gate is stated since #3089, whose floor owed-check re-ruled to 50 on
  2026-10-25 (#3329; the rate bar held); the canary runs
  `--seeds 16` (#2130) and has measured state; `autocode`'s last-ten avg is
  ~1200 s not ~2000; the bench corpus is 28 files. Re-pointed the drifted
  `file:line` citations in §2/§3 (`pool.py`, `app/routers/workers.py`,
  `automod_regression.py`) and fixed §6's two dangling cross-references to
  §2's pre-#1713 figures. Findings filed: #3283 (no rail reads doc
  `file:line` citations), #3284 (canary module docstring still says
  `--seeds 12`), #3285 (config comment claims every unattended turn is IV off).
