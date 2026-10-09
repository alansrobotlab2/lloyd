---
segment: architecture
tags:
- architecture
- lloyd
status: implemented
type: reference
timestamp: '2026-09-13T00:00:00'
---
# Autonomy System Architecture

**Created:** 2026-03-22
**Last updated:** 2026-09-13
**Status:** Active — 32 task files, dispatched by the `scheduled-task` worker source

## Overview

**Autonomy is for recurring work only.** There is no run-once task: every
task re-dispatches on its `frequency` until its file is deleted, and `done` is not a
scheduler status. A one-off goes on the backlog, where autocode works it and closes
it on its clauses. (2026-09-24: three one-off arXiv digests, #87–#89, were filed
here with `frequency: daily` and re-ran daily until deleted 2026-09-26.)

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
`~/lloyd-data/autonomy-runs/{task_id}/run_{task_id}_{YYYYmmdd_HHMMSS}.md`
(`app.paths.AUTONOMY_RUNS_DIR`, anchored to the repo rather than the vault),and
a small key/value block at `~/obsidian/autonomy/_config.md` that only the
`autonomy_config` MCP tool reads or writes. qmd indexes both collections —
`autonomy` over the vault directory and `autonomy-runs` over `*/run_*.md`
(`~/.config/qmd/index.yml`).

**Six readers**, where this overview used to name three (MCP tools, an MC
extension, the idler daemon):

| reader | what it does |
|---|---|
| `app/autonomy.py` | task-file CRUD, the due predicates, `run_task`, `compute_health` |
| `workers/sources/scheduled_task.py` | enqueues what is due; executes a claimed item |
| `app/routers/autonomy.py` | `/api/autonomy/{run,tasks,task-write,task-delete,health,runs}` |
| `agent_mcp/autonomy.py` | seven `autonomy_*` MCP tools |
| `app/routers/dashboard.py::_autonomy` | the overdue/held split on Mission Control |
| `app/routers/mc_ui.py::_summarize_autonomy` | the tab summary the agent reads |

**The rule is one sentence, and it is written down in three shapes.** A task file is
named `NN-slug.md`, so the config block and the stray reports in that directory are not
tasks whatever their frontmatter says. It is NOT true that every reader skips a file
whose name does not start with a digit: the shared pattern `_TASK_NAME_RE` also needs a
**hyphen** after the digits, `app/routers/dashboard.py:668` settles for
`path.name[:1].isdigit()`, and `agent_mcp/autonomy.py` skipped exactly one filename by
literal until #1692. The three shapes:

* **Shared** — `app/routers/autonomy.py::_TASK_NAME_RE` (`re.compile(r"\d+-")`, applied
  with `.match`, so effectively `^\d+-`) gates the task route, the tab summary
  `mc_ui._summarize_autonomy` through `autonomy_task_files()` / `list_parsed_tasks()`
  (#1594), and since #1692 `agent_mcp/autonomy.py::_handle_tasks` — the reader #1594
  did not enumerate. It globbed the directory, excluded one filename by literal, and
  accepted everything else on frontmatter alone, so `meta-analysis-2026-06-03.md`, a
  prose note with legal frontmatter and no `name:` key, reached every `autonomy_tasks`
  call as `{id: 0, name: "", status: "draft"}`: 33 rows for the 32 task files on disk.
  Pinned by `tests/test_agent_mcp_autonomy_tasks_gate.py`.
* **Re-spelled inline** — the identical pattern appears four times in the scheduler as
  `re.match(r"\d+-", path.name)`: `recover_stuck_tasks` (`app/autonomy.py:102`),
  `_find_task_file` (`:167`), `_all_board_tasks` (`:1012`), `_iter_task_files`
  (`:4116`). Same behaviour, four more copies of the literal, none of them importing
  the constant. `workers/sources/scheduled_task.py` has no gate of its own; it reaches
  the fleet through `app.autonomy` and inherits these.
* **Looser by one character** — `app/routers/dashboard.py:668`, inside `_autonomy()`,
  gates on `path.name[:1].isdigit()`. Because the shared regex requires the hyphen,
  a file named `9x-notes.md` counts on the Mission Control panel and is not a task to
  the route, the tab or the MCP tool. (`:812` is the same idiom over the *backlog*
  directory, where nothing dispatches on the difference.) The two id allocators,
  `_autonomy_next_id` and `_next_task_id`, parse the numeric prefix themselves and
  still skip the config block by literal.

The walk is `glob("*.md")` and never `rglob`, which is why `_archived/` — 14 retired
tasks — is invisible to the scheduler rather than merely inactive. #1594 closed a
different divergence in this same discussion: `mc_ui._summarize_autonomy` used to gate
on frontmatter, count a prose note as a task, and drop three real tasks whose
frontmatter ran past its 2000-byte read window — 32 task files reported as 30.

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
- **The failure record says what is undoable, and undoes nothing.** A failed or
  timed-out run's front matter carries `changes: sessions/<sid>.changes/<run_id>/
  (N files)` with N read from that turn's `index.json` — not from anything the
  dying run asserted, which is why the number can be trusted — and the body lists
  the paths (`autonomy._change_ledger_note`). Nothing reverts on that path: a
  nightly that wrote 6 of its 9 notes before dying has 6 notes of real progress,
  so restoring stays a decision made from the record, through
  `autonomy.revert_run_writes(session_id, run_id)`. That call answers per path,
  and a file another writer moved since the snapshot comes back `refused` by name
  instead of being overwritten or skipped.

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
   `enqueue_if_due` calls `autonomy.get_due_tasks()`, which reads the directory
   once into **two** lists — `dependency_resolution_set()` (the whole board) to
   resolve `depends_on` against, `_all_runnable_tasks(resolution)` (status- and
   grant-filtered) for the candidates — pins one `now` for the whole pass, then
   runs `_is_task_due` over the runnable set and sorts by priority, then by
   staleness.
3. **The queue.** Due tasks are enqueued under `dedup_key =
   scheduled-task:<id>`, so one already waiting is never enqueued twice, at a
   priority mapped from the frontmatter word:
   `critical|high|medium|low|background → 10|20|30|50|70`, default 30. Lower
   runs sooner. `max_inflight: 2`.
4. **The run.** `execute` calls `autonomy.run_task(id, max_duration=3600)`,
   which builds the prompt, resolves the model, opens a session and streams a
   real agent turn through `run_query`.
5. **The records.** Four of them, joined — see above.

`app/autonomy.py` owns 1, 2 and 4; `workers/sources/scheduled_task.py` owns 3 and
the health gates around it. Nothing schedules inside `app/autonomy.py` any more,
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
   `runs_per_day` first (`86400 / n`), then the four-word vocabulary in
   `FREQUENCY_INTERVALS` (the one definition, #815 — `validate_tasks.py`
   imports it): `hourly` 3600, `every-15min` 900, `daily` 86400, `weekly`
   604800. Anything else yields `None`, which is **not due, ever**, and warns
   once per process like the no-skill gate (`_no_frequency_warned`). #24's
   `frequency: 6x-daily` is not in that map and runs only because
   `runs_per_day: 6` is consulted first; the linter warns on such a file the
   day `runs_per_day` goes missing, **and** on the opposite error (#2445) — a
   word that IS in the map contradicting a `runs_per_day` that is present, which
   is what #30 did for 47 days (`daily` + `3`, resolved 28800 s, dispatched 3×/day)
   while the linter passed all 38 files. The two are silent on `weekly` +
   `0.14` (2.04% apart: two decimals of 1/7) and on `6x-daily` + `6` (the label
   is outside the map on purpose).
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

One exception to that order, and it exists only on this side of the seam
(#1739): where the dependency gate bites and its detail comes back EMPTY,
`hold_reason` consults `_already_ran_this_period` before returning the string, and
answers `"already ran this period"` instead of `"waiting on #N"`. An undescribed
`waiting on #N` is not a claim about the chain, it is the absence of one:
`_dependency_refusal` returns `''` both when the dependent's own `last_run` has
already consumed this cycle — the HEALTHY chain, seconds after its last link —
and when the dependent declares no `stale_bypass_hours` for the detail to name. So
a completed chain and a chain behind a dead upstream printed one identical string:
measured 2026-09-28T09:32Z, #42, #39 and #40 had each finished that night's cycle
and `/api/autonomy/tasks`, `dashboard._autonomy` and the stall alert all read
`waiting on #38` / `#42` / `#39` over their own run records. A detail that names a
clock still wins, so a dependent that ran while its upstream still owes the cycle
inside its declared bound keeps `waiting on #N (inside its 36 h ...)`.
`_is_task_due`'s order is untouched: there the run-record scan stays LAST because
it is the only gate that pays a disk read, and dispatch is not served by a better
sentence (#1296).

Both readers call it rather than restating it — `/api/autonomy/tasks` per row
as `blocked`, and `dashboard._autonomy` to split past-due rows into `overdue`
(nothing holds it) and `held` (something does). The dashboard reports
`classifier: "naive"` when `from app import autonomy` failed and every past-due task is
being called overdue again, because a downgrade that looks like success is the
failure this split exists to prevent. Note that the dependency gate resolves
`depends_on` against the set it is handed, and since #558 a dependency it
cannot find, or finds unpromisable, is **not met** — so every caller must
resolve against the *whole* board. `/api/autonomy/tasks?status=up_next`
classified against its own filtered list is the wrong direction now: the
upstream is usually the task that just left that status, so the filtered view
would report a satisfied dependency as `waiting on #N` and the hold would be
invisible rather than under-reported. Either way one caller reading a
half-board gets the wrong answer, which is the point of the next paragraph.
That is no longer a convention each caller honouring on its own — #870 made it
one function, `autonomy.dependency_resolution_set(directory)`, and dispatch
(`get_due_tasks`), `_grossly_overdue`, `/api/autonomy/tasks` and
`dashboard._autonomy` all call it. Before that, dispatch resolved the same
`depends_on` against the status-filtered runnable set while the board resolved
it against everything: a `paused` upstream was invisible to the scheduler, and
the board printed `waiting on #42` in the same second the scheduler dispatched
the task underneath it.

### Dependencies, and failing forward

`depends_on` is a single id. The gate is freshness, not mere completion.
Without it, yesterday's upstream satisfies today's gate and the chains settle
into a stable inverted order where every downstream task consumes a day-old
artifact — observed June 2026, with reflection running 39→38/40→42 and
trajectory running 57 before 56. What "fresh" means depends on the windows:

- **Both tasks windowed, dependent daily or slower** (#1437 Defect B,
  `_window_dependency_fresh`): the dependent's cycle is its window occurrence
  that last opened, closing at C; the upstream's output is fresh unless the
  upstream still OWES a run before C — its next slot (`_next_run_after`, the
  instant dispatch will next take it) falls before C. So #57 (open 23:00
  local) waits for #56 (01:00 local) of the same night, and #42 follows #38
  in their shared window, whatever the hour count. The old bound, half the
  dependent's interval, straddled the window whenever a dependent's window
  opened before its upstream's: in the healthy chain it needed a bypass over
  21.9 h to stay safe and after a slip one under 21.0 h to release, so no
  `stale_bypass_hours` could serve both. When the upstream is owed, a dependent
  that declares no `stale_bypass_hours` waits the whole window; one that declares
  it is released by the bound alone (#1538). Until then the gate also asked
  `_upstream_due_in_window` — is the upstream inside its own window and due on this
  very tick? — and held before the bound was consulted, which made a declared bound
  unreachable in exactly the hours its dependent may run: #42 sat from
  2026-09-22T05:16:36Z through four cycles with #38's last success 36.1 h and 38.5 h
  old against `stale_bypass_hours: 36`, #39 starved behind it, `failure_count: 0`
  the whole time. The bound is the owner's decision to forward on the previous
  cycle's file, so a prediction about the next tick no longer overrules it. What
  still refuses is the bound's own elapsed check (upstream 35.0 h old against a
  36 h bound: held, so the race the prediction guarded against stays impossible),
  an upstream `in_progress`, and #1437's requirement that a declared
  `output_artifact` be on disk. `_upstream_due_in_window` survives as the sentence
  in `hold_reason`, and now applies `_already_ran_this_period` as `_is_task_due`
  does, so a reverted `last_run` cannot make it predict a dispatch the scheduler
  will veto (#1296).
- **Otherwise** (either side windowless, or a sub-daily dependent): the
  upstream must have succeeded within **half this task's interval**, as
  before.

Either way the upstream's run must be newer than the dependent's own last run.

A dependency the gate cannot name holds its dependent — **fail closed** (#558).
Two shapes reach that: no task file answers the id at all, and an upstream that
exists but could not itself dispatch (parked `paused`/`draft`, disabled without
ever having succeeded, or dropped for an unreadable `grants:` block — the same
`_dispatch_blockers` list `_is_task_due` uses, so "may this task run?" and "may
this task certify its dependent's input?" cannot diverge again, which is the
2026-09-08 inversion: #42 sat `paused`, the scheduler could not see it, and
#39/#40 ran at 06:00–06:03Z against a vacuously-satisfied gate while #42's
handoff landed at 06:10Z). The hold is not silent: `_warn_fail_closed` logs the
dependent, the upstream id, and what was found, deduplicated per episode so a
60 s dispatch tick cannot shout. `stale_bypass_hours` is the escape and still
fires, so a genuinely-dead upstream forwards on stale input rather than
retiring the chain — and the bypass still refuses while the upstream is
`in_progress`, which is why a wrong-id misconfiguration cannot use the bypass to
dispatch both tasks on top of each other.

The gate is `_dependency_refusal`, which answers *why not* — `None` when the
dependent may run, else the detail `hold_reason` puts inside `waiting on #N`.
`_is_dependency_met` is its boolean projection, and dispatch calls only that, so
the detail (a run-record scan plus an artifact stat) is built only for the board
and for the stall alert, never on the per-tick path (#1538). The detail matters
because one bare string used to stand for two opposite states: 48 alarms naming
#42 fired over three days, each reading `held: waiting on #38`, and none could say
whether the chain was one run behind (`inside its 36 h stale_bypass window: #38
ran 14.5 h ago`, which releases by itself at the bound) or lost (`stale_bypass
36 h passed; #38 is in_progress`, which does not). A dependent that declares no
bound prints the bare `waiting on #N` exactly as before.

`stale_bypass_hours` is principle 3 made real, and for a long time it was not:
the field was set on the reflection chain and described in this document as
letting a dependent run with stale input, and **nothing read it**.
`_dependency_bypassed` does now. It bypasses only while the upstream is not
`in_progress`, so a merely-late upstream is still waited for, and an upstream
that has never succeeded is bypassable. Since #1437 a bypass also needs the
upstream's declared `output_artifact` on disk (`_upstream_artifact_on_disk`: a
`{date}` candidate for its last run, dated by completion or by start, at least
512 bytes; no mtime test, since stale input is the point): fail-forward means
stale input, not none, and on 2026-09-24 #39 was released 50.7 h past #42 onto a
handoff that existed under no spelling. Held that way, it logs one warning per
episode naming the missing file; an upstream that declares no artifact keeps the
elapsed-time rule alone. Every run record also carries `trigger` (`scheduler`,
`api`, `mcp`, or `direct` for an unnamed caller, set with `run_trigger(...)`)
and, for a windowed task, `in_window` — the out-of-window runs of #38/#56 on
2026-09-23 that pinned both chains were attributable to nobody. The chains on
disk today:

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
the interval check allows `min(3600, interval * 0.25)` of slack.

A daily-or-slower windowed task also counts its period from the window
occurrence its run belonged to, not from the run (#1437 Defect B,
`_elapsed_due_at`): due again at that opening plus the interval less the
slack, floored at half an interval after the run. Before this, one
out-of-window run re-anchored the task outside its window — #38 finished at
14:31Z on 2026-09-23, came due at 13:31Z the next day after its window had
closed, and lost that night too. Now it is due at 05:00Z. The run-record guard
asks the same function of each success, and a completion writes `next_run` as
the first in-window instant at or after it (`_next_run_after`), so the board
names the slot dispatch will use. Windowless and sub-daily tasks are unchanged:
`last_run + interval - slack`, `next_run = last_run + interval`. Fourteen of
the 32 tasks end up with an enforced window: thirteen set
`preferred_hours`, and #84's empty list falls through to the hour in its
`scheduled_at`.

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
  whose type name is in `_INFRA_EXC_NAMES` is infra: the httpx transport and
  socket connection errors, `ToolDiscoveryError` when the aggregator never
  answered tool discovery, and `StreamStalledError` when an engine that had
  already started producing goes quiet mid-generation and the loop's own retry
  cannot take it back. Membership is the frozenset in `app/autonomy.py` — read
  it there rather than counting the prose, because the sentence this replaced
  said "the eight" while the set already held nine. And an empty response
  inside `_INFRA_EMPTY_MAX_SECONDS` (15 s) **with no tool call** is infra too:
  that fast and that silent, it is a thinking-only turn or a 200 with no
  content, not work.

`_DEFAULT_MAX_RETRIES` is 5. Thirty of the 31 runnable task files carry
`max_retries: 3`, which is also what the create path writes, so 3 is the number
in practice; #84 carries 2, and #83 carries none, so a failure of that task
counts against 5 rather than 3.

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
accident. `compute_health` tracks the rate, and the rate is almost entirely a
cadence artefact rather than a behaviour: **337 of 808 runs in the week to
2026-09-13 came back `[SILENT]`**, 306 of them from two tasks — #68 Email &
Calendar Triage at every-15-minutes (132 of 428) and #75 AI Engineer YouTube
Monitor (174 of 180), which was archived on 2026-09-09 and will age out of the
window.

## Two timeouts, and the thirty seconds between them

Both caps apply and the smaller wins: `timeout_seconds` in the task's own
frontmatter, enforced by `run_task`'s `asyncio.timeout`, and
`workers.sources.scheduled-task.max_duration_seconds` (3600), enforced by the
pool's `asyncio.wait_for`. When the two were equal the **pool** timer won the
race, cancelling `run_task` before its own handler could run: no run record, no
activity-log line, the task file left `in_progress`. 309 such runs — 96.9
GPU-hours, the last one on 2026-09-03 and none since — sit in `workers.db` with
a NULL `task_id`.

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
lives in `app/` rather than privately in `app/autonomy.py` because the automod
worker turn needs identical behaviour, and two copies of "how close is the
deadline" drift in the direction nobody is watching.
`tests/test_autonomy_budget_anchor.py` pins it, including that `run_task` wires
it to the resolved timeout.

#### The second clock: iterations (moved from CLAUDE.md, 2026-09-25)

A run is bounded by **two** clocks, and for years it was warned about neither,
then about one. Beside the wall clock (`asyncio.timeout(timeout_seconds)` in
`autonomy.run_task`) is the iteration clock, `RunOptions.max_turns`
(`agent.max_turns`, 60), which `app/harness/loop.py` enforces and
`app/harness/finalizer.py` records **no verdict for at all**. The chat path has
warned at 75%/90% of `max_turns` since `_build_state_anchor` landed, but that
warning was a closure inside that function, so `run_task` — which calls
`run_query` directly — had nothing to import and passed no `state_anchor`.

20 scheduled-task runs between 2026-09-04 and 09-18 died at
`stop_reason=max_turns, turns=61`. Six of those predate the wall-clock anchor
(`af038eb`, 2026-09-08 20:21 −07:00); of the 14 after it, every one finished its
61 iterations below 70% of its own clamped timeout — the shortest at 269 s, the
longest at 2059 s against a 2499 s level — so the warning that did exist could
not have fired on a single one (#1061). So `_build_task_anchor(timeout,
max_turns)` is now the single `RunOptions.state_anchor`, carrying both clocks:
70%/90% of the resolved wall-clock budget, plus 75%/90% of `max_turns` in the
chat path's exact `<budget>Iteration N of M` wording. Both builders live in
`app/deadline_anchor` — `build_deadline_anchor` and `build_iteration_anchor`,
joined by `compose_state_anchors`.

**One callable, and it is not `None` just because the wall clock is absent.**
The harness takes exactly one `state_anchor`, and `app/harness/loop.py` calls it
inside a `try` that swallows what it raises and logs a warning — so a composed
anchor that misbehaves fails the same way a missing one does: a warning nobody
hears and no run record. A task with `timeout_seconds: 0` has no wall clock but
still has a turn cap and still dies on it, so `_build_task_anchor` returns an
iteration-only anchor there, and `None` only when neither budget exists. (The
70% level is computed from `timeout`, never `declared_timeout`: the pool clamps
the frontmatter value to `max_duration - _POOL_TIMEOUT_MARGIN`.)

#### The skills matter as much as the mechanism

A budget only helps a task that knows when it is done. All three of the
2026-09-08 deaths had an unbounded step:

- **#80** named `okf_migrate.py --apply` as the repair path without saying "not
  from this task", and its last successful report *ended in a question* ("Want
  me to run the migrate pass?"). Nobody answers an autonomy report, so the next
  three runs went looking for their own permission to act. Its runs also spent
  time diagnosing a bubblewrap sandbox that did not exist then (nothing in
  `agent_mcp/builtin_bash.py` sandboxed anything; the bench/eval sandbox in
  `agent_mcp/_tool_sandbox.py` came 2026-09-14 and applies to bench sessions
  only).
- **#78** Step 2 asked the model to hand-filter "unreferenced notes" — but only
  711 of 4,448 vault files contain a wikilink at all, so the sweep returned
  ~3,600 files, 84% of the vault. Its Step 3 was `[! -f "$file" ]`, which is not
  valid shell: it raises `[!: command not found` every iteration and has never
  detected a missing skill.
- **#24** Step 1 ran `nightly_extraction.py` in the foreground. The Bash tool
  defaults to 120 s and caps at 600 s; that script's last seven real runs took
  545–1666 s, so the call *cannot* complete and the model improvises a `nohup`.
  On 2026-09-08 the improvised run finished fine 40 minutes later — the task had
  already timed out, so `files_processed=133 facts=4969 failed=1048` was
  reported to nobody. (Those 1048 were one incident, not 1048 problems: the
  extractor talks to the primary on :8096 and that engine restarted mid-run, so
  every remaining document raised `Connection refused`. Unhashed files retry.)

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
| `notify_on_complete` | whether the completion notice posts to Discord — read by `execute` *after* the run, and only for a successful non-`[SILENT]` result, so a failed or timed-out run cannot notify whatever this says |
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

### The dispatch-affecting fields, and the three surfaces that write them

Seven front-matter fields decide whether a task runs: `SCHEDULE_STATE_FIELDS` =
`status`, `scheduled_at`, `depends_on`, `auto_advance`, `frequency`, `skill_name`,
`preferred_hours` (`app/harness/policy.py:313`). **#724's rail holds `vault_write`
too, the third of the three surfaces that can write them** — with
`autonomy_write_task` and the vault-round landing route — **and the page names the
lanes it does not reach.** The first is the tool: `effective_tier` demotes an
`autonomy_write_task` call that moves none of them back to tier 1, so re-arming or
parking a task needs a grant and appending an activity note does not. The second is
the landing route an unattended turn can use instead of the tool —
`scripts/automod/vault_round.py` classifies `autonomy/**` as validated and used to
check only that the file still parses, so a round denied the call could write the
same field into the file. On 2026-10-01 one did exactly that:
`autonomy/92-vllm-prefix-miss-daily.md` landed at 18:51:05Z, eighteen seconds after
the rail denied that round's `autonomy_write_task` (the denial row is in
`~/lloyd-data/safety/denials.jsonl`, and its witness copy is
`~/obsidian/backlog/data/denials.jsonl`). It came in `status: draft`, so nothing was
armed — but nothing stood in the way of `up_next` either.

Since #2190 `vault_round.schedule_state_errors()` refuses it, from the same
frozenset rather than a copied list, and it refuses on the **diff**: the baseline is
`git show HEAD:autonomy/<file>`, so a round that rewrites a description, appends an
activity note, or re-quotes the YAML lands, and one that moves a value does not. Two
asymmetries are deliberate. A **create** is judged the way `policy` judges one — only
`status: up_next` dispatches, so a new task written `draft` with its skill and window
set is the nightly chain's hand-down shape and passes. A **deleted** task file is not
refused at all, because the lifecycle retires tasks by moving them out of the live set
and #777 is what a lander made unable to retire anything does. The tool gate, by
contrast, cannot see the disk and so denies a *resend* of `up_next` to a task that is
already armed; the vault route reads HEAD and lets an unchanged value through, which
is the difference between guarding a field and refusing to write the file.

The **third** surface is `vault_write` (#2362), and it is the one that proves the rail
had a hole rather than a design: it is tier 1 and consults only the protected-path
deny-set, whose entries do not include `~/obsidian/autonomy/`, so a call the rail denied
could be spelled one tool later and land in the same file. On 2026-10-05 a turn whose
`autonomy_write_task` was refused at 06:38:16Z had already created
`autonomy/96-djev-name-prior-probe.md` with `status: up_next` through `vault_write` at
06:27:38Z, and `run_scheduled-task_20261005_062825_b3ceb1` started 47 seconds after that
write — armed through the open door, refused through the closed one. It is not a rare
shape either: 17 of the 2514 rows in `~/obsidian/memory/audit/writes.jsonl` are
`vault_write` onto `autonomy/`, across 7 sessions. `agent_mcp/vault.py`,
`_autonomy_schedule_refusal`, now refuses it, and it shares its rule rather than
restating it: `policy.front_matter_map`, `policy.schedule_value_moves` and
`policy.create_arms_a_task` are one definition read by both file lanes, so widening
`SCHEDULE_STATE_FIELDS` widens every lane with no edit to a lane, and both fail closed on
a front matter they cannot parse and on a rail they cannot import.

**The gate is on the file, not on the spelling of its path.** The first version of
this lane tested `path.startswith("autonomy/")` on the string
`_normalize_vault_path` returned, and that function validated `Path(p).parts` —
where a `./` is already gone — while handing back the caller's dots.
`VAULT / './autonomy/96-x.md'` is the same file as `VAULT / 'autonomy/96-x.md'`, so
one leading dot-segment reached a live task file with both phases returning `None`:
the bypass this section documents, rebuilt inside the guard written to close it.
`_normalize_vault_path` now returns the reduced path (`Path(*parts).as_posix()`),
and that is the root fix rather than a second prefix test, because the value it
returns is the choke point for everything downstream — the two `knowledge/`-only OKF
guards, the `path` field of the row appended to `memory/audit/writes.jsonl` (which is
what the owed-after-landing check on this item greps for), and the `path` in the
success result. `_is_autonomy_task_path` is the second line, so a phase reached
through another caller cannot reintroduce it. Pinned by
`tests/test_vault_write_schedule_guard.py::test_the_normalizer_hands_every_guard_one_spelling_of_a_path`
and the two spelling-parametrised nodes beside it.

**What the rail does not reach, stated rather than implied.** `memory_add` /
`memory_replace` / `memory_remove` are not lanes at all: that tool's `file` argument is
grammar-bound to `MEMORY.md|USER.md|topics/<slug>` (`app/memory_ceiling.py`,
`agent_mcp/session.py`), so it cannot name a task file. What stays open after #2362 is
`Write`/`Edit` (`agent_mcp/builtin_fs.py` carries no reference to `autonomy`) and a Bash
child, which `agent_mcp/_path_sandbox.py` sandboxes with one read-only bind per
`PROTECTED_WRITE_ROOTS` entry — and `~/obsidian/autonomy/` is not one of those entries, so
closing it that way would also refuse the sanctioned nightly writers of these very files.
That is a scope ruling, recorded as owed on #2362, not something a guard can quietly
assume. The asymmetry above still stands with it: a task file that leaves the live set is
retired, not refused, and none of these three surfaces can delete one.

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

**The list is the scheduler's, and only the scheduler's.** Two other readers
parse the same files with their own shorter lists:
`agent_mcp/autonomy.py::_parse_task_file` names 13 fields — no `preferred_hours`,
`depends_on`, `model`, `stale_bypass_hours`, `last_attempt` or `inner_voice`,
and one (`board_id`) the scheduler's list has never carried — and
`app/routers/autonomy.py::_autonomy_parse` passes no fallback list at all, so a
YAML-broken file returns `None` and vanishes from `/api/autonomy/tasks` while
the scheduler still dispatches it. The invariant holds where it has to; the
display and MCP paths can report a degraded task as window-less, chain-less and
model-less, and no test pins the two lists against each other.

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
  Down, nothing is enqueued and no task is executed — but the tick is not
  abandoned there. The probe's verdict is taken first and acted on only after
  both stall alarms and the outage's own alert have run. Returning at the probe
  used to silence dispatch AND the watchdogs together, and left one deduped
  `logger.warning` line behind (#938): an outage stops work, it does not stop
  watching. An outage continuous past `_VLLM_DOWN_ALERT_SECONDS` raises one
  Discord alert of its own, which is the only route that names the model server
  at all — `agent-services/guardian/policy.py` watches `lloyd-backend` and
  `lloyd-mcp`, never `:8096`, and the vault tasks that probe it are autonomy
  tasks dispatched through this gate, so they go down with it.
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

A **second, quieter assertion** sits beside it (`_next_run_stalled`, #421,
2026-09-12), keyed on each task's own `next_run` rather than on due-ness: any
`up_next` task more than one whole period past its `next_run`, on a separate
streak counter with a 24 h cooldown, so neither alarm can reset the other's. It
exists because the three exclusions that make the first alarm survivable are
exactly the shapes that sat silent for 41 h on #51 and 60 h on #40 — a
dependent whose upstream's run was never recorded is *not due*, so the due-ness
alarm cannot see it. Each entry carries `hold_reason`, dispatch's own words, so
the alert says why it has not run rather than only that it has not.

Two shapes sit outside that period and are reported anyway — each a row the period
cannot measure, not a widening of it. #2342 admits the row whose file carries **no
`next_run` at all** and trusts its own run records. #2417 admits a row that *is*
past its `next_run` but by less than a period, and only when the upstream named in
its `depends_on` is itself past its own `next_run` or has a newest run record that is
not a `success`; the line then names the chain — `blocked-by #38, which failed at
2026-10-08T05:04:09Z` — because a late dependent is the symptom and the upstream's
run record is the cause. That upstream read goes through
`autonomy.newest_run_record`, the status-agnostic twin of `newest_successful_run`,
which filters to `success` before it sorts and so hears a dead upstream as "no
records". Neither door touched the bound: a task with no `depends_on` still waits one
whole period, which is what keeps the false stalls #2342 retired from coming back.

Alerts and completion notices are sent through `app/discord_notify.py`, and on
this box an alert lands on today's daily note (`memory/<date>.md`), not on Discord.
Discord is unconfigured by decision rather than by accident: `config.yaml` ships
`discord.home_channel: null` and the bot token is empty, and
`tests/test_autonomy_failure_alert.py` reads that null off the disk and fails a
round that configures it, because whether Discord is ever set is Alan's edit. The
two calls behave differently when the transport is missing:

- **An alert falls back.** `discord_alert` logs its warning and hands the message
  to `_survive_the_dropped_alert`, which appends a "Scheduler alert not delivered"
  line to the daily note through `app.autonomy.append_daily_alert_line` (#1592).
- **A completion notice is dropped.** `_discord_notify_task_complete` returns
  without posting and has no fallback; a finished task's record is its run file.

The append has a read-back contract (#1736): a returned `True` means the line was
readable back from the note at the moment the call returned, not that the alarm
can no longer be lost — a later whole-file rewrite from an older snapshot still
removes it. The witness after the return is the ledger
`~/lloyd-data/alerts/daily-note-appends.jsonl`, one row per confirmed append
(#1799), and its reader is the `daily_note_appends` component of the system
health check (`~/obsidian/skills/system-health-check/system_health_check.py`),
which re-reads the last 48 h of rows and reports a line that is no longer in its
note.

This is the subsystem's own route and predates the guardian's fan-out:
`agent-services/guardian/notify.py` does not carry autonomy's alerts. Both stacks
do write `memory/<date>.md` — the guardian through its own
`agent-services/guardian/daily_note.py`, sharing the header rendering in
`app/daily_note.py` — so the daily note is the one surface where the two meet.

## Fleet health

`GET /api/autonomy/health?days=7` and the `autonomy_health` MCP tool read
**`workers.db`**, not the per-task run records, because the 309 pool-timeout
rows with a NULL `task_id` were unreachable from any per-task view by
construction — and `/api/autonomy/runs` requires a `task_id`. A task that timed
out on every single run was indistinguishable from a healthy one.

`autonomy.compute_health` is a pure function over those rows: per-task and
fleet failure rate, GPU-hours, wasted hours, timeouts, empty runs, `[SILENT]`
rate, `max_turns` runs, tool-error runs, consecutive failures — plus
`idle_tasks`, the tasks with a file and no runs in the window, which a
rows-only view cannot see at all.

It carries the #525 evidence bundle beside those numbers.
`EVIDENCE_PILOT_TASK_IDS = {38, 42, 39, 40, 53}` — the reflection chain plus the
Documentation Digester, which joined 2026-10-09 (#2482), a literal set rather than a
config key because widening it should be a change someone reads — get a claims block
appended to their prompt, binding checkable
assertions (`file_exists`, `count_eq`, `json_key`, `regex`) to paths on disk.
The pool verifies them when it writes the row, *after* the run, so nothing the
run did — including its own tool calls, which could have edited the file being
claimed — can affect the measurement of what it asserted. Refuted claims are
carried into the next run of that task by name, through the queue's watermarks
rather than the task file: the gap list is ledger state, and a human-edited
vault file is not somewhere a worker should be writing hourly. A run with no
bundle counts as `runs_without_bundle` and never as clean, because a metric
that reads its own missing input as zero is the failure mode this file has been
burned on three times. **Today that is exactly what the pilot is, for the whole
fleet: across 808 runs in the week to 2026-09-13, `runs_with_bundle` is 0 and
`claims_checked` is 0.** Not because the four tasks stopped emitting claims —
because `scheduled-task.execute` returns a hand-written result dict that never
includes the `claims` key `pool.py` reads, so the bundle is dropped one hop
before the verifier is called. The counting rule below is sound and its input is
empty; #945 (umbrella #966) is the fix, and until it lands every
`claims_checked: 0` in a report means "never measured", which is the reading the
rule was written to force.

**A declared acceptance is graded, and decides nothing yet (#623).** A task may
carry an `acceptance:` block — `objective_checks` drawn only from
`scripts/autoresearch/judge.py`'s `CHECK_TYPES`, plus `rubric` names — and
`app/run_acceptance.py` grades every successful run against it with the judge's
deterministic layer, over the run's own dispatch record (`tool_trace_authoritative`,
hook-refused calls filed as denied), never over what its text says it did. The
grade (`graded_pass` / `graded_fail` / `no_acceptance` / `acceptance_invalid` /
`not_measurable` / `grader_error`) rides on the run's `meta` as
`acceptance_grade`, so it is in the run record's front matter and in
`runs.meta_json`; `false_completion_rate` reads the share of `success` rows
graded `graded_fail`, per source or per task, `None` where nothing was graded.
The status stays the literal `success`, nothing is requeued, the rubric is
recorded as `rubric_ungraded` rather than scored (no model call), and
`grader: false` removes the key entirely. Gating on the grade waits on a
hand-labelled check of the grader.

## The fleet today

32 task files: **31 `up_next` and one `draft`** (#85, pinned to a `model: eco`
that no `models:` block defines) — nothing sits `in_progress` between runs,
which is the point of the recovery path rather than a normal state. 30 run on
`primary`, #68 on `secondary`, #85 on that undefined alias, and **no task sets
`inner_voice: true`** — the fleet is recorded and unobserved. Fourteen more sit
in `_archived/`, invisible to a non-recursive walk rather than merely inactive;
three of those fourteen (#52, #59, #73) still have run rows in the seven-day
window, because the retention sweep does not delete run history and
`compute_health` reports a task id with no file as `task: null` and never as a
ghost.

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

`autonomy:` in config.yaml has five keys and two of them are read:

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

## Review log

- **2026-09-13 — `current`.** Every mechanism described here still runs: the
  five gates and their order, the two-list dispatch (`dependency_resolution_set`
  + `_all_runnable_tasks`, #870), the failure ladder and its two kinds, both
  timeout margins, the deadline anchor, the four joined records, the graduated
  parser. What had moved is the snapshots: 30 `up_next` + 1 `in_progress` is now
  31 `up_next` + 1 `draft`; the pool-timeout residue is 309 rows / 96.9 GPU-h and
  closed on 2026-09-03, not 237 / 73.6 and ongoing; the enforced-window count is
  14 (thirteen `preferred_hours` plus #84 via its `scheduled_at` hour), not 13;
  and `max_retries: 3` is on thirty of thirty-one files, with #84 at 2 and #83
  carrying none. Two statements were aspirational in the bad sense and are now
  marked: the #525 evidence pilot has verified **nothing** — 0 bundles and 0
  `claims_checked` across 808 runs — because `scheduled-task.execute` returns a
  result dict with no `claims` key for `pool.py` to read (#945, umbrella #966);
  and `[SILENT]`'s 42% fleet rate is two high-frequency tasks, not a behaviour.
  Added the second stall alarm (`_next_run_stalled`, #421) beside the due-ness
  one, and one sentence naming what the display readers do with a degraded task
  (#1014).
