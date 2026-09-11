---
segment: architecture
tags:
- architecture
- lloyd
status: active
type: reference
timestamp: '2026-07-06T14:36:37'
---
# Autonomy System Architecture

**Created:** 2026-03-22
**Last updated:** 2026-09-11
**Status:** Active — 32 task files, dispatched by the `scheduled-task` worker source

## Overview

A scheduled task is a markdown file. `~/obsidian/autonomy/NN-slug.md` carries a
skill name, a frequency and a set of gates in its frontmatter; the
`scheduled-task` worker source reads that directory every 60 s, asks
`autonomy._is_task_due` which files are ready, and enqueues them into the one
work queue. `autonomy.run_task` then loads the named skill, runs it as a real
agent turn, and writes what happened to four places at once.

Until mid-2026 this was a different system, and its vocabulary outlived it:
four "agent types" — memory, operator, idler, researcher — pulled work with LLM
dispatch gated on GPU utilization, polling every 250 ms and dispatching under
30%, so that background work never competed with a foreground turn. That loop
is gone. `workers/pool.py` is the only dispatcher now and
`architecture/workers.md` is its document; the surviving trace of the old model
is the `agent_id:` field, still written on every task file (23 `memory`, 4
`researcher`, 3 `worker`, 1 `operator`) and read by nothing that schedules.

**Storage:** Vault markdown files (migrated from SQLite 2026-03-29). Task files
at `~/obsidian/autonomy/{id}-{slug}.md`,run records at
`~/lloyd/autonomy-runs/{task_id}/run_{task_id}_{YYYYmmdd_HHMMSS}.md`
(`app.paths.AUTONOMY_RUNS_DIR`, anchored to the repo rather than the vault),and
a small key/value block at `~/obsidian/autonomy/_config.md` that only the
`autonomy_config` MCP tool reads or writes. qmd indexes both collections —
`autonomy` over the vault directory and `autonomy-runs` over `*/run_*.md`
(`~/.config/qmd/index.yml`).

**Six readers**, where this overview used to name three (MCP tools, an MC
extension, the idler daemon):

| reader | what it does |
|---|---|
| `autonomy.py` | task-file CRUD, the due predicates, `run_task`, `compute_health` |
| `workers/sources/scheduled_task.py` | enqueues what is due; executes a claimed item |
| `app/routers/autonomy.py` | `/api/autonomy/{run,tasks,task-write,task-delete,health,runs}` |
| `agent_mcp/autonomy.py` | seven `autonomy_*` MCP tools |
| `app/routers/dashboard.py::_autonomy` | the overdue/held split on Mission Control |
| `app/routers/mc_ui.py::_summarize_autonomy` | the tab summary the agent reads |

Every one of them skips a file whose name does not start with a digit, so
`_config.md` and the stray reports in that directory are not tasks. The walk is
`glob("*.md")` and never `rglob`, which is why `_archived/` — 14 retired tasks —
is invisible to the scheduler rather than merely inactive.

## Every run is recorded (2026-09-10)

> Every task runs through the `scheduled-task` worker source into
> `autonomy.run_task`; `architecture/workers.md` is the queue's document and
> `architecture/background-runs.md` the long version of this section.

Until 2026-09-10 `run_task` called `run_query` directly and kept only the
text, so the record of what a task *did* was a 200-character summary in its
run file. That day an autonomy task was the prime suspect in a full vault
wipe and could be neither confirmed nor cleared: its tool calls had never
been written down. At 851 runs in the week to 2026-09-10 (122 a day), it was
the largest single share of what this machine did unattended — 851 of 2,115
background runs that week.

Each run now leaves four records, all joined:

| record | where | keyed by |
|---|---|---|
| session + transcript | `sessions/<ts>_autonomy_<4hex>.json`, `platform: autonomy`, `source: autonomy-task:<id>`, titled `#<id> <name>` at creation | session id |
| event log | `event_logs/<session>.events.jsonl` — the same `brain1.*` trail a chat turn writes | session id + `turn_id` |
| run file | `autonomy-runs/<id>/<run_id>.md`, whose frontmatter now carries `session_id` on **every** outcome — success, timeout, cancellation, empty response, exception | `run_id` |
| change ledger | `sessions/<session>.changes/<run_id>/` — pre-images of every file the run wrote, revertable | `turn_id = run_id` |

- **It is a passthrough.** `app/run_recorder.py` wraps the existing
  `run_query` loop, persists each event and re-yields it unchanged, so the
  #534 grant hook, the deadline anchor, `saw_tool_call`, `tool_errors` and the
  timeout partial are all untouched. Routing the run through the chat
  endpoint instead — a stale sandbox branch tried it — drops the grant gate,
  which that endpoint does not install on its own registry's behalf.
- **A run killed at its deadline keeps what it had.** Persistence is
  incremental and the final flush is shielded from the cancellation that
  ended the run, which is how the interesting ones end.
- **`turn_id` is the load-bearing option.** It is what switches on the
  per-turn change ledger, so a scheduled task's file writes can be undone. It
  was the only turn path with no undo.

**Observation is a separate opt-in.** A task file's `inner_voice: true` makes
the Inner Voice observer watch the run; `autonomy.inner_voice` in config.yaml
is the fleet default and ships **off**, because the observer runs on the
primary at priority 1 and spends a goal extraction plus a critique per turn.
Frontmatter beats config in both directions, and the key survives the
degraded parser (`fallback_fields`). The observer attaches to this direct
path by passing `run_task`'s own `HookRegistry` — `attach_observer_for_turn`
creates one only when none exists — so an observed task keeps its grant gate.
Its `cancel` lever is wired to the loop; its ambient and clarify callbacks
are not, because both exist to reach a human mid-turn and nobody is reading.

`architecture/background-runs.md` is the long version.

## Design Principles

1. **Sense -> Analyze -> Act** — ingest raw data, reflect on it, then improve from it
2. **One job per task** — each task has a single clear responsibility with no overlap
3. **Fail forward** — `stale_bypass_hours` lets dependent tasks run with stale input rather than blocking the whole chain when one step fails
4. **Priority orders the queue; nothing preempts** — `preemptible: true` sits on
   almost every task file and is read by no scheduling decision. What priority
   actually buys is a position in the work queue (`_PRIORITY_MAP`) and a place
   in `get_due_tasks`'s sort. A running task runs to its timeout.
5. **Closed-loop learning** — trajectories feed skill mining, reflection feeds dream, dream feeds skills management. Nothing is write-only.

## How a task becomes a run

Five steps, three processes, and the seams are where it has gone wrong.

1. **The file.** `~/obsidian/autonomy/NN-slug.md`. Frontmatter carries the
   schedule and the gates; `skill_name` names the skill under
   `~/obsidian/skills/<slug>/SKILL.md` that is the actual instruction.
2. **Due-ness.** Every 60 s (`workers.sources.scheduled-task.interval_seconds`)
   `enqueue_if_due` calls `autonomy.get_due_tasks()`, which runs `_is_task_due`
   over the runnable set and sorts by priority, then by staleness.
3. **The queue.** Due tasks are enqueued under `dedup_key =
   scheduled-task:<id>`, so one already waiting is never enqueued twice, at a
   priority mapped from the frontmatter word:
   `critical|high|medium|low|background → 10|20|30|50|70`, default 30. Lower
   runs sooner. `max_inflight: 2`.
4. **The run.** `execute` calls `autonomy.run_task(id, max_duration=3600)`,
   which builds the prompt, resolves the model, opens a session and streams a
   real agent turn through `run_query`.
5. **The records.** Four of them, joined — see above.

`autonomy.py` owns 1, 2 and 4; `workers/sources/scheduled_task.py` owns 3 and
the health gates around it. Nothing schedules inside `autonomy.py` any more,
which is what its docstring means by scheduling having moved to the unified
work queue.

## The five gates

`_is_task_due` is the whole of "should this run now", and it is five questions
asked in order. Anything that answers no is a **hold**, not a failure:

1. **A skill.** No `skill_name` and no `skill_path` means the task can never
   run, so it warns **once** (`_no_skill_warned`) and skips forever. Silence
   here dead-lettered #79 in June 2026.
2. **`status == "up_next"`.** Only that one word dispatches. `in_progress` used
   to stay "due", so a task could be enqueued while a copy of itself was still
   running — two runs of #38 once started nine seconds apart and interleaved.
   `_all_runnable_tasks` nonetheless *admits* `failed` into the set, purely so
   `_is_dependency_met` can still see a disabled upstream; without that the
   lookup misses it, returns True, and a dependent runs off a broken input.
3. **The interval has elapsed.** `_frequency_interval_seconds` reads
   `runs_per_day` first (`86400 / n`), then a four-word vocabulary: `hourly`
   3600, `every-15min` 900, `daily` 86400, `weekly` 604800. Anything else
   yields `None`, which is **not due, ever**. #24's `frequency: 6x-daily` is
   not in that map and runs only because `runs_per_day: 6` is consulted first.
4. **No failure cooldown, and the dependency is met.**
5. **The current hour is a preferred hour.**

`hold_reason(task, all_tasks)` mirrors those gates *in the same order and
through the same predicates*, returning the first that bites — `"no skill"`,
the raw status, `"no frequency"`, `"failure cooldown"`, `"waiting on #42"`,
`"outside hours 00-04,23"` — or `None`. It exists because Mission Control had
grown a second, private definition of "due" out of elapsed time alone, and on
2026-09-06 reported six tasks overdue on a night this scheduler considered none
of them late: four nightly jobs outside their window, two paused, all six
behaving exactly as configured. A nightly task is past due for the eighteen
hours a day it is not allowed to run, so that counter was never zero and
therefore said nothing.

Both readers call it rather than restating it — `/api/autonomy/tasks` per row
as `blocked`, and `dashboard._autonomy` to split past-due rows into `overdue`
(nothing holds it) and `held` (something does). The dashboard reports
`classifier: "naive"` when `import autonomy` failed and every past-due task is
being called overdue again, because a downgrade that looks like success is the
failure this split exists to prevent. Note that the dependency gate resolves
`depends_on` by id and treats an unresolvable id as **met**, so both callers
must hand it the *whole* board: `/api/autonomy/tasks?status=up_next`
classified against its own filtered list would report every dependency
satisfied, since the upstream is usually the task that just left that status.

### Dependencies, and failing forward

`depends_on` is a single id. The gate is freshness, not mere completion: the
upstream must have succeeded within **half this task's interval**, not merely
"since my last run". Without that, yesterday's upstream satisfies today's gate
and the chains settle into a stable inverted order where every downstream task
consumes a day-old artifact — observed June 2026, with reflection running
39→38/40→42 and trajectory running 57 before 56.

`stale_bypass_hours` is principle 3 made real, and for a long time it was not:
the field was set on the reflection chain and described in this document as
letting a dependent run with stale input, and **nothing read it**.
`_dependency_bypassed` does now. It bypasses only while the upstream is not
`in_progress`, so a merely-late upstream is still waited for, and an upstream
that has never succeeded is bypassable. The chains on disk today:

| task | depends on | bypass |
|---|---|---|
| #42 Reflection: Knowledge Analysis | #38 Signals | 36 h |
| #39 Reflection: Knowledge Write | #42 | 36 h |
| #40 Reflection: Config | #39 | 36 h |
| #47 Dream Consolidation | #40 | — |
| #48 Entity Resolution Sweep | #24 | — |
| #51 Conversation Relation Linking | #56 | — |
| #57 Trajectory Mining | #56 | — |
| #58 Skill Consolidation | #57 | — |
| #74 KG Mention Classifier | #48 | — |
| #83 Nightly Skills Management | #58 | — |

### Hours, and the window that drifts shut

`preferred_hours` is a list of **machine-local** hours — this box runs
America/Los_Angeles, and the Autonomy page's input placeholder still says UTC
while the scheduler does not. Empty means no window.
`_effective_preferred_hours` falls back to the hour in `scheduled_at` when it
parses as `HH:MM`, because several tasks documented a schedule there and left
`preferred_hours` null, so nothing enforced it — #81 ran at 17:57 and took the
qmd daemon offline mid-afternoon. A cron-expression `scheduled_at` yields no
window.

`last_run` is a *completion* time, so the due moment drifts later by the run's
own duration every cycle; for a task pinned to a one-hour window that drift
eventually steps past the window and skips a day. So when a window is in force
the interval check allows `min(3600, interval * 0.25)` of slack. Thirteen of
the 32 tasks set a real window.

## Failure is a state machine, not a log line

A failed run used to leave `last_run` untouched — and `_is_task_due` gates on
`last_run` — so a task that timed out was due again on the very next 60 s tick,
forever. That produced 12 consecutive 600 s timeouts on #36 in one night and
~130 on #69 over two days, roughly 21 GPU-hours spent on a task whose script
was gone. `failure_count` and `max_retries` were parsed, stored and displayed,
and no scheduling decision read either.

`_record_failure` is now the single failure path — run record, backoff,
activity log, alert — and it splits two kinds:

- **`task`**: the task itself failed. Increments `failure_count`, and at
  `max_retries` sets `status: failed` and alerts once. Cooldown is
  `600 × 2^(n-1)`, capped at `max(interval, 6 h)`.
- **`infra`**: the model server hiccuped. Flat 600 s cooldown, and it never
  touches the retry budget, so an outage cannot disable the whole fleet — on
  2026-09-01 every task returned empty for eleven hours straight. An exception
  whose type name is in `_INFRA_EXC_NAMES` (the eight httpx/socket connection
  errors) is infra, and so is an empty response inside
  `_INFRA_EMPTY_MAX_SECONDS` (15 s) **with no tool call**: that fast and that
  silent, it is a thinking-only turn or a 200 with no content, not work.

`_DEFAULT_MAX_RETRIES` is 5, but every live task file carries `max_retries: 3`
and that is what the create path writes, so 3 is the number in practice.

**`last_run` means the last success; `last_attempt` means every attempt.** The
cooldown reads "last_attempt newer than last_run" as "the most recent attempt
failed", so a success must write both — and `last_run` has to keep meaning
success, because it is what the dependency freshness gate reads. Bumping it on
failure would let a broken upstream satisfy everything downstream of it.

**An empty response is a failure.** It used to be relabelled "(No response)"
and recorded as a success: `last_run` advanced, `failure_count` reset,
dependents unblocked. That is how #79 went dark for a week on 0.6 s
"successes", and how ~180 phantom runs passed during the 2026-09-01 window.
`compute_health` still reclassifies those historical rows as it reads them,
because what is on disk says success.

**And a run that completes can still have failed.** `_detect_silent_failures`
greps the final text for four high-precision shapes — `failed because`, a
traceback header, a non-zero `exit code`, a bare builtin exception name — and
marks the run record and activity log without changing the status. A task's own
`expected_error_patterns` suppress what it deliberately provokes: #48's dry-run
is *required* to raise `FileNotFoundError` while the graph is missing, which
produced 33 false positives in a week and taught everyone to ignore the
indicator. Tool errors are collected separately from `tool_result` events with
`is_error` — the harness hands those back to the model rather than raising, so
without that collection they never reach the record at all.

**`[SILENT]`** is the one sentinel in the prompt: exactly that string and
nothing else means "nothing to report", and it suppresses the Discord
completion embed. Combining it with content is explicitly refused, because a
delivery decision made by substring against a long answer is one made by
accident. `compute_health` tracks the rate.

## Two timeouts, and the thirty seconds between them

Both caps apply and the smaller wins: `timeout_seconds` in the task's own
frontmatter, enforced by `run_task`'s `asyncio.timeout`, and
`workers.sources.scheduled-task.max_duration_seconds` (3600), enforced by the
pool's `asyncio.wait_for`. When the two were equal the **pool** timer won the
race, cancelling `run_task` before its own handler could run: no run record, no
activity-log line, the task file left `in_progress`. 237 such runs — 73.6
GPU-hours — sit in `workers.db` with a NULL `task_id`.

So `run_task` derives `max(60, min(declared, cap − 30))`
(`_POOL_TIMEOUT_MARGIN`) and the task handler always wins.
`tests/test_autonomy_timeout.py` pins the pair, including that the margin is a
positive number of seconds and that a short cap still leaves working time. The
sibling constant on the session-backed worker path,
`_common.POOL_TIMEOUT_MARGIN_SECONDS`, is 60 — deliberately wider, because that
path has an HTTP call to make on its way out.

**A `CancelledError` is not an exception.** The pool cancels through
`wait_for`, and `CancelledError` is a `BaseException`, so neither the timeout
branch nor `except Exception` caught it: the run vanished with no record and
the file stayed `in_progress` until `recover_stuck_tasks` found it. It is now
caught, recorded (with `alert=False`, since nobody chose this failure) and
re-raised so cancellation still propagates.

### The budget the model can see

`asyncio.timeout` is invisible to the thing it kills. On 2026-09-08 task #80
ran `validate_okf.py` — a 2.4 second script — on a 300 s budget, failed three
consecutive times recording `(no output before timeout)`, and auto-disabled at
`max_retries`. It had not hung: the run records show 39 completions and real
findings in the partial text. The model had the answer inside the first minute,
kept investigating, and was killed without ever being asked to write it down.
#78 and #24 died the same way in the same window.

`app/deadline_anchor.py::build_deadline_anchor` is passed as
`RunOptions.state_anchor` and fires once at **70%** and once at **90%** of the
budget (`BUDGET_WARN_FRACTIONS`). Three things about it are load-bearing. It is
built from the **resolved** timeout, not the declared one, because the pool
clamps the frontmatter value and a warning at 70% of the declared budget can
land after the kill. The two levels say different things: at 70% it is "do not
open new lines of investigation", at 90% "stop calling tools and write the
report from what you have", and a partial report beats a failed run. And each
level fires once, because a warning re-sent every iteration is one the model
learns to skip.

Resolution is one iteration — a single tool call longer than the remaining
budget still overruns. That is the accepted limit: the failure being fixed is
twenty short iterations past the stopping point, not one long one. The module
lives in `app/` rather than privately in `autonomy.py` because the automod
worker turn needs identical behaviour, and two copies of "how close is the
deadline" drift in the direction nobody is watching.
`tests/test_autonomy_budget_anchor.py` pins it, including that `run_task` wires
it to the resolved timeout.

## The task file

Read by the scheduler, and therefore worth getting right:

| field | what it does |
|---|---|
| `id`, `name`, `description` | identity; `description` is appended to the prompt |
| `status` | `up_next` dispatches; `in_progress` is a live run; `failed` is disabled-after-retries; `draft`/`paused` are held |
| `skill_name` | slug under `~/obsidian/skills/<slug>/SKILL.md`, or a path; `skill_path` is the legacy spelling |
| `frequency`, `runs_per_day` | the interval; `runs_per_day` wins |
| `preferred_hours`, `scheduled_at` | the window, machine-local |
| `depends_on`, `stale_bypass_hours` | the chain, and the fail-forward escape |
| `timeout_seconds` | the task-side cap (default 1800) |
| `max_retries`, `failure_count` | the disable ladder |
| `last_run`, `last_attempt`, `next_run`, `updated` | success / attempt / display / stuck-detection |
| `model` | alias, resolved through `config.resolve_model_alias` |
| `expected_error_patterns` | suppressions for the silent-failure detector |
| `inner_voice` | per-task observer opt-in |
| `notify_on_complete` | whether a non-`[SILENT]` result posts to Discord |
| `grants` | #534 authority; an unreadable block makes the task **not runnable** |

Written and displayed, read by no scheduling decision: `agent_id`,
`preemptible`, `auto_advance`, `pipeline_mode`, `pipeline`, `cron_id`. They are
the old four-agent model's vocabulary, kept because deleting a key a human
still reads is worse than carrying one nothing acts on.

`grants` is the exception that is not inert. `_all_runnable_tasks` drops any
task whose block fails `validate_task_grants`, because the block **is** the
human's authorization: parsing it badly and running anyway is fail-open with a
log line attached, and the task would then take durable external actions under
an authority nobody wrote. Such a task is not due, not enqueued, and does not
drain; the error names the problem and the file is one edit away. No live task
declares one today.

**Unknown keys survive a write.** `/api/autonomy/task-write` overlays
`_WRITABLE_TASK_KEYS` onto the file's *existing* frontmatter rather than
rebuilding it from that list, which is what it used to do — so every key not
named there was silently destroyed by any UI edit or task-write call. `tags`
was never in the list at all, and neither were the fields added later.

### The parser can never drop a task

On 2026-05-28 a bulk edit corrupted the `tags` field of 34 of 40 task files (an
inline list followed by orphan block-list items), `yaml.safe_load` raised, and
the scheduler silently skipped each one: no failure record, no alert. About 85%
of the fleet was dormant for six days before anyone noticed.

`_parse_task_file` uses the shared graduated-recovery parser
(`agent_mcp._shared.parse_frontmatter_text`): plain YAML → orphaned-tag repair →
regex extraction of a named field list. A task can come back degraded
(`_yaml_broken: True`) but it can never silently vanish, and the next
`yaml.dump` write normalizes the file on disk. That `fallback_fields` list is
the contract — a field missing from it is lost on a degraded file, which is why
`inner_voice` was added there and not only to the reader.

Two things guard the same failure from outside:
`scripts/autonomy/validate_tasks.py` is a fail-loud linter (exit 1 unparseable,
exit 2 structural), and the source's first tick after boot runs
`_unparseable_task_files` and alerts with the filenames.

`_update_task_field` uses the same recovering parser rather than
`yaml.safe_load`, because it is called from the *failure* handler — the exact
moment a task most needs its status and `failure_count` recorded.

### Stuck, and recovered

A worker that dies mid-run leaves `status: in_progress` on disk, which gate 2
then holds forever. `recover_stuck_tasks` resets any `in_progress` task whose
`updated` is older than its own `timeout_seconds` (1800 default) back to
`up_next`, with an activity-log line. It runs at backend startup
(`start_autonomy_ticker`) and again at the top of every enqueue tick.

It **rewrites vault files**, so `autonomy.ticker_enabled: false` switches the
startup half off — an automod canary sets it as a second explicit layer behind
its own scratch `HOME`. That startup hook is all that remains of the ticker the
function is named after; scheduling is the pool's.

`run_task` holds the matching guard on the other side: it refuses to start a
second run of a task whose `in_progress` is still fresh and returns `skipped`,
which `/api/autonomy/run` turns into a 409. The manual and MCP entry points
bypass due-checks entirely, so this guard is the only thing between an
impatient human and two interleaved copies of #38.

## The source: two health gates and a stall alarm

`workers/sources/scheduled_task.py` wraps the enqueue in gates that exist
because a wedged engine turns every due task into a ConnectError flood:

- **Before enqueuing at all**, a `/health` probe of the primary (`:8096`).
  Down, the whole tick is skipped, logged once on the way down and once on the
  way back up.
- **Per task**, a probe of the engine that task actually runs on.
  `_model_health_url` resolves `model:` through `resolve_model_alias` and
  `_get_model_env`, so a dead secondary skips only its own tasks instead of
  turning each into a retry loop.
- **Inside `execute`**, up to 90 s (18 × 5 s) of polling for recovery before
  spending an attempt, because the enqueue-side gate cannot help items already
  in the queue when the engine wedges under them.

**A task failure is reported in-band, never raised.** Raising sends the item
back through the queue's retry path, so a single timeout became up to
`max_attempts` full re-runs — 3 × 600 s on #36 — before the scheduler's own
cooldown was ever consulted. By then `run_task` has already written the run
record, bumped `failure_count` and set the cooldown.

The **stall alarm** watches the opposite failure, the 2026-05-28 silent stall:
a task that is due *right now*, overdue by more than 2.5× its interval, and has
**no queue row**. The queue-row exclusion is the whole point — a task sitting
behind a saturated pool is starved for capacity, not stalled, and flagging it
forever is what fired this alarm 100 times in six days. It needs 5 consecutive
ticks and re-alerts at most every 6 h. `_queue_starving` catches the mirror
case (dispatch fine, workers dead) from the age of the oldest claimable item.

Alerts and completion notices go to Discord (`app/discord_notify.py`). That is
this subsystem's own channel and predates the guardian's six-channel fan-out;
`agent-services/guardian/notify.py` does not cover it.

## Fleet health

`GET /api/autonomy/health?days=7` and the `autonomy_health` MCP tool read
**`workers.db`**, not the per-task run records, because the 237 pool-timeout
rows with a NULL `task_id` were unreachable from any per-task view by
construction — and `/api/autonomy/runs` requires a `task_id`. A task that timed
out on every single run was indistinguishable from a healthy one.

`autonomy.compute_health` is a pure function over those rows: per-task and
fleet failure rate, GPU-hours, wasted hours, timeouts, empty runs, `[SILENT]`
rate, `max_turns` runs, tool-error runs, consecutive failures — plus
`idle_tasks`, the tasks with a file and no runs in the window, which a
rows-only view cannot see at all.

It carries the #525 evidence bundle beside those numbers.
`EVIDENCE_PILOT_TASK_IDS = {38, 42, 39, 40}` — the reflection chain, a literal
set rather than a config key because widening it should be a change someone
reads — get a claims block appended to their prompt, binding checkable
assertions (`file_exists`, `count_eq`, `json_key`, `regex`) to paths on disk.
The pool verifies them when it writes the row, *after* the run, so nothing the
run did — including its own tool calls, which could have edited the file being
claimed — can affect the measurement of what it asserted. Refuted claims are
carried into the next run of that task by name, through the queue's watermarks
rather than the task file: the gap list is ledger state, and a human-edited
vault file is not somewhere a worker should be writing hourly. A run with no
bundle counts as `runs_without_bundle` and never as clean, because a metric
that reads its own missing input as zero is the failure mode this file has been
burned on three times.

## The fleet today

32 task files: 30 `up_next`, one `in_progress`, one `draft` (#85, pinned to a
`model: eco` that no `models:` block defines). 30 run on `primary`, #68 on
`secondary`, and **no task sets `inner_voice: true`** — the fleet is recorded
and unobserved. Fourteen more sit in `_archived/`, invisible to a
non-recursive walk rather than merely inactive.

The nightly chain is **#38 → #42 → #39 → #40 → #47**, and two of the ids this
document used to give were wrong: Knowledge Analysis is **#42**, never "#39a",
and Dream Consolidation depends on #40, runs weekly, at `medium`.

This section used to be a hand-maintained inventory of 19 tasks, and by
2026-09-11 it was four months stale in the way such a table always goes stale —
it still listed #25 Memory Capture, #29 Self-Improvement Loop and #45/#46
Trajectory Extraction and Mining, all since retired to `_archived/` and
superseded by #56/#57/#58/#83, and #33/#34 Groundskeeper Loop and Research and
#41 Email+Calendar Monitor, which are gone from the fleet altogether (#68
Email & Calendar Triage covers the old #41's ground). The *schedule* is not
reproduced here again: `GET /api/autonomy/tasks` is the current one, the
Autonomy tab renders it with each row's `blocked` reason, and
`python scripts/autonomy/validate_tasks.py` is the structural check.

What each job is **for**, on the other hand, does not change on a nightly
clock, and that is [[autonomy-jobs]] — one entry per task, grouped by the chain
it belongs to, including the orderings no single file reveals: the nightly
reflection chain, trace2skill, and the knowledge-graph chain below.

## Config

`autonomy:` in config.yaml has four keys and two of them are read:

| key | status |
|---|---|
| `inner_voice: false` | **live** — the fleet default for the observer; a task's own `inner_voice:` beats it |
| `ticker_enabled` | **live** — default true; `false` skips the startup stuck-task recovery (the automod canary sets it) |
| `task_dir: ~/obsidian/autonomy` | dead — every reader hardcodes `Path.home() / "obsidian" / "autonomy"` |
| `tick_interval: 60` | dead — the real cadence is `workers.sources.scheduled-task.interval_seconds`, which is also 60 |
| `enabled: true` | dead — nothing reads it; the live switch is the source's own `enabled` |

The dead three are worth naming rather than deleting: `task_dir` in particular
reads like the way to relocate the fleet, and moving it would change nothing at
all.

## Knowledge-Graph Chain

Ordered because each step's input is the previous step's output. #24 produces
entities and `mentions` edges; #48 settles canonical names; #74 types the
edges; #60 measures the result; #82 measures whether retrieval improved.

| ID | Name | Freq | Depends | Applies? | Role |
|----|------|------|---------|----------|------|
| #24 | Data Pipeline | 6x/day | -- | yes | Extract facts from the `sources.paths` corpus; emit `mentions` edges with source_doc + evidence |
| #51 | Conversation Relation Linking | daily | #56 | yes | Co-access pairs from trajectories -> `co_accessed` edges (provenance INFERRED) |
| #48 | Entity Resolution Sweep | daily | #24 | **no** | Cluster near-duplicates, run the semantic gate, print the plan and #67's proposals. A merge is a human action |
| #67 | Semantic Entity Resolution | weekly | -- | **no** | LLM-judge the pairs string rules cannot; write `semantic-proposals-latest.jsonl` for #48's review list |
| #74 | KG Mention Classifier | daily | #48 | yes | Re-type `mentions` into typed relations via `edges.retype`, one transaction |
| #60 | Knowledge Health Report | daily | -- | no | God/thin/orphan entities, contamination, provenance. **Exits 2 and alerts** on store-unreadable, below-baseline, contamination, or duplicate fact IDs |
| #82 | Nightly Retrieval Eval | daily | -- | no | 20-query eval with production's recall defaults; records the trend |

Only #51 and #74 write to the edge store outside extraction, and **nothing in
this chain moves fact files unattended**. #48 is the one that could — it is the
sweep that merges — but it never passes `--apply`, so its scheduled form
reports a plan and stops; #67 proposes into
`semantic-proposals-latest.jsonl` and stops too (its own `--apply` was retired
2026-09-04). Everything else here reports. That split is deliberate: the
2026-08-22 wipe and the 2026-09-03 151-merge incident were both unattended
applies.

**One store.** Edges, aliases, the entity registry and the fact index live in
`_pipeline/vault-derived/kg.sqlite`, behind `app.kg_store`. Nothing else opens
it. See `architecture/knowledge-graph.md`.
