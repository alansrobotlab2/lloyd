---
segment: architecture
tags: [architecture, lloyd, workers]
type: reference
status: implemented
date: 2026-09-08
---

# The worker pool and the unified work queue

Everything Lloyd does without being asked runs through here. Scheduled
autonomy tasks, research jobs, session mining, the backlog triage that judges
his own code and the round that then changes it are all one queue drained by
a small number of workers.

This replaced three separate loops — a time-based autonomy scheduler, a
vault-change pipeline and a standalone autoresearch runner — and the reason to
keep them merged is that they compete for the same GPU. Three schedulers with
three ideas about what is busy is how you get a research job and a nightly
extraction on the same 24 GB card.

---

## 1. Shape

```
 config.yaml workers.sources.*
        │
        ▼
 SCHEDULER  ── enqueue_if_due(queue, cfg) ──►  QUEUE (sqlite, WAL)
 (1 task,                                        │
  60s tick)                                      │ claim_next(worker_id, quota)
                                                 ▼
                                        WORKERS  (N asyncio tasks)
                                                 │
                                                 ├─ execute(item) ──► run record
                                                 └─ mark_completed / mark_failed
```

Two loops, both inside the **backend** process (`server.py`):

- **The scheduler** wakes every 60 s, and for each enabled source whose
  `interval_seconds` has elapsed calls `enqueue_if_due`. It is the only thing
  that creates work.
- **The workers** — `workers.slots` of them — claim one item at a time and
  await its source's `execute`.

`workers.db` lives beside the repo (`~/lloyd/workers.db`, gitignored) and is
opened in WAL mode, because the **MCP aggregator is a second process** that
writes to it: `autoresearch_run` enqueues a round from there.

---

## 2. The queue

Three tables. `queue` is the work, `runs` is the history, `watermarks` is
whatever a source needs to remember between ticks.

An item walks one path:

```
queued ──claim──► claimed ──► running ──┬─ ok ──────────► completed
   ▲                                    │
   └────── requeue + not_before ────────┴─ fail ─┬─► queued  (attempts < max)
                                                 └─► poisoned (attempts ≥ max)
```

### Claiming

Ordered `priority ASC, enqueued_at ASC` — **a lower priority number runs
sooner**, which is worth saying out loud because it has been read backwards
once already: `backlog-implement` sat at 80, behind research jobs at 70 that
arrive every few minutes, and the first unattended round had no path to a
slot. It is at 40 now.

`max_inflight` per source is applied **in the SQL**, by excluding saturated
sources from the query. It used to select fifty rows and skip over-quota ones
in Python, which is a different thing entirely once a saturated source has
more than fifty items queued: they fill the window, the claimable row behind
them is never seen, and the pool reads the queue as empty and sleeps. Three
sources sharing two slots on this box makes that an ordinary Tuesday.

### Dedup

`dedup_key` coalesces: enqueuing a key that already exists in
`queued|claimed|running` is dropped. **Completion releases the key**, because
most sources want to run it again later — `scheduled-task` dedups on
`scheduled-task:<id>` and has to run that task on its next interval.

A source that wants *once ever* must keep its own record. `session-distill`
does, with `done:<file>` watermark rows, and section 5 explains why it had to
learn that the expensive way.

### Failure

Two kinds, and the difference is the most useful thing in this file:

- **Raised** — infrastructure. The item is requeued with an exponential
  backoff (30 s, 60 s, 120 s, … capped at 10 min) and poisoned after
  `max_attempts`. The backoff exists so a wedged vLLM does not turn every item
  into a tight failure loop.
- **Returned** as `{"status": "failed", ...}` — the *job* failed, and re-running
  it will not help. It is recorded and the row is completed. Raising instead
  is what made one timed-out autonomy task re-run three times at 600 s each
  before the scheduler's own cooldown was ever consulted.

### Recovery

`pool.start()` returns **every** `claimed|running` row to `queued` — not just
the ones matching its current slot names. Worker ids are positional
(`worker-0` … `worker-{slots-1}`), so filtering by them stranded anything a
higher-numbered slot held the moment `workers.slots` was lowered. Those rows
stay `running` forever, and since `claim_next` counts `claimed|running`
toward the quota, a source with `max_inflight: 1` was then switched off
permanently and silently.

---

## 3. What a source is

A module in `workers/sources/` with:

```python
NAME: str
DEFAULT_PRIORITY: int
async def enqueue_if_due(queue, src_cfg) -> None
async def execute(item) -> dict
```

registered in `workers/sources/__init__.py`.

### The result contract

`execute` returns a dict. `pool.normalize_result` is the authority:

| key | meaning |
|---|---|
| `status` | `success` \| `failed` \| `skipped`. Absent means success. |
| `summary` | one line, ≤500 chars — this is what a human reads later |
| `artifact_path` | what it produced |
| `response` | full output, ≤50k |
| `task_id` | for the per-task views |
| `meta` | structured detail (stop reason, flags) |

**`skipped` is not a failure and not a success**: the source looked and there
was nothing to do. That distinction is not cosmetic. `selfmod-regression`
signalled "I could not measure anything" by returning `{"skipped": reason}` —
a key where a status belongs — so all 22 of its runs were recorded as
successes with an empty summary. For a detector whose entire premise is *a
missing noise floor means cannot evaluate, never no regression*, that is the
one failure it cannot afford, and it was invisible in the runs table and on
the dashboard alike. `normalize_result` now reads a bare `skipped` key as a
status, and logs any result with neither summary nor artifact.

### Event-loop discipline

**Everything here runs on the backend's one event loop** — the loop that
serves every HTTP request and streams every chat turn. A source that blocks it
does not slow the pool down, it stops Lloyd answering.

So: no `subprocess.run`, no `urlopen`, no unbounded file walk directly inside
`execute` or `enqueue_if_due`. Put the body in a plain function and hand it to
`asyncio.to_thread`. `selfmod_regression.execute` broke this with two
900-second eval arms and a `git worktree add` between them; the 109-second run
in its history is 109 seconds during which the backend answered nothing.
`test_workers_pool.py::test_no_source_blocks_the_event_loop_in_execute` greps
for it, and the queue's own writes are hopped onto threads for the same
reason.

---

## 4. Two ways to run a turn, and how to choose

| | `run_prompt_on_primary` | `run_prompt_in_session` |
|---|---|---|
| calls | `run_query` directly | `POST /api/message/stream` |
| session | none | a real one, `platform: worker` |
| Inner Voice | no | yes |
| transcript | discarded | persisted, reviewable |
| used by | research and mining | anything judging or changing Lloyd's code |

`app/routers/messages.py` is the **only** turn path that attaches the
observer, so work a human must be able to audit goes through it rather than
having the observer wiring copied into a second place. `selfmod_start` refuses
a turn with no Inner Voice, which is what makes that structural rather than a
convention.

### An empty turn is a failed turn

`run_prompt_on_primary` returns a `TurnResult`, not a string. It used to
return the concatenated text and nothing else, and a turn that ends at
`max_turns`, or on a tool call, or against a wedged engine produces no text —
indistinguishable from a short answer once the stop reason has been thrown
away.

What that cost: **225 of the 498 notes under `pending-research/` have the body
`(no response)`.** The source responsible, `domain-research`, wrote the empty
note, ticked its topic off in the queue file so it could never be retried, and
returned success. It has since been retired for that whole class of reason.
Nothing anywhere said a research job had failed. Sources now check
`turn.ok` and write nothing when it is false.

### The turn's budget must sit under the pool's

The pool wraps `execute` in `asyncio.wait_for(…, max_duration_seconds)`.
Whoever's timer fires first decides what happens next, and if it is the
pool's, it cancels the *HTTP request* — while the chat path is explicitly
built to keep running when its client disconnects. The turn is then orphaned:
the pool records a failure and backs off, and for `backlog-selfmod` the retry
re-selects the same item, because no verdict was ever written. A second
90-iteration triage, racing the first one that never stopped.

So `run_prompt_in_session` defaults its own wall-clock bound to
`turn_timeout_for(source)` — the source's `max_duration_seconds` minus
`POOL_TIMEOUT_MARGIN_SECONDS` (60 s) — and on expiry it **cancels the turn in
the backend** before raising `TurnTimeout`. Same shape as
`autonomy.run_task`'s cap against the same pool timer, with a wider margin
because this path has to make an HTTP call on its way out.

Note that `httpx`'s timeout is per-read, not total: handing it the budget
bounds nothing on a stream that keeps producing.

---

## 5. The sources

| source | what it does | turn path |
|---|---|---|
| `scheduled-task` | runs `~/obsidian/autonomy/*.md` via `autonomy.run_task` | its own |
| `backlog-selfmod` | triages one backlog item — read-only, reaches a verdict | session |
| `backlog-implement` | one gated selfmod round per confirmed item | session |
| `selfmod-regression` | paired A/B eval after a promotion | none (subprocess) |
| `autoresearch` | one prompt-optimisation round | its own |
| `deep-research` | one registry topic, through the deep-dive-research skill | session, IV off |
| `session-distill` | mines finished chats for gaps and patterns | direct |
| `gap-fill` | resolves `label: gap` facts | direct |
| `bench-mine` | new bench tasks from baseline losses | direct |

**`deep-research` owns its own retries, and that is not a preference.** The
pool records an in-band `{"status": "failed"}` and then calls `mark_completed`
on the item regardless — only a *raised* exception reaches `mark_failed` and
the queue's backoff. So a source that returns `failed` and expects to be tried
again is simply not. That source puts the topic back in its own registry with
a `not_before`, and its `interval_seconds` is the retry cadence.
`architecture/research-pipeline.md` is the long version, including why the
markdown checklist it replaced could not hold an outcome.

`scheduled-task` carries the most traffic by far, and two of its behaviours
are load-bearing. It gates on the health of **the model each task pins**, not
just the primary, so a dead secondary does not turn its tasks into a
ConnectError flood. And it passes the pool's cap into `run_task` so the task's
own timer always wins — when the two were equal the pool cancelled the handler
before it could write anything, which is where 237 runs and 73.6 GPU-hours of
NULL `task_id` came from.

### session-distill selects three ways, and each one is scar tissue

A session is distilled **once**, **after it goes quiet**, and **only if it is a
user session**.

- *Once.* Selection keyed on an `mtime > last_mtime` watermark, and a
  session's mtime advances with every message. An active chat therefore
  crossed the watermark again on the very next tick, forever: **one chat was
  distilled 44 times, another 22, and the worst eight account for 138 runs.**
  The dedup key could not help, because completion releases it by design.
  Per-session `done:` markers replace the moving cursor — and they fix its
  mirror image too, where a session skipped for being busy was left stranded
  below the advanced watermark and never looked at again.
- *Quiet.* A chat still being typed into is not a transcript to learn from.
- *A user session.* `platform: worker` and `platform: autonomy` sessions are
  this system talking to itself; 12 of Lloyd's own triage transcripts had been
  mined back in as observations about the user before this gate existed.
  `sessions_io.NON_USER_PLATFORMS` is the one definition of that.

---

## 5.1 Structured verdicts

`backlog-selfmod`'s verdict was parsed out of `VERDICT:` / `SURFACE:` lines by
regex. That works until a turn words it slightly differently, and then a
`confirmed` is recorded as `unverifiable` and an item is retired for a
formatting reason.

`run_prompt_in_session(..., final_schema=...)` asks the harness for one extra
completion after the turn ends, restating its conclusion under a JSON schema
(`app/harness/finalizer.py`). It returns `structured` and `structured_error`
beside `text` and `stop_reason`.

- `scripts/selfmod/backlog.TRIAGE_VERDICT_SCHEMA` is built from
  `VERDICTS`/`SURFACES`, not restated — one list, or a new verdict lands in the
  grammar and not the validator. It carries no `maxLength`: that is enforced by
  the guided decoder, so the model would stop mid-sentence at the limit rather
  than write something shorter. The clamps stay in `parse_verdict`.
- **The regex stays and the `VERDICT:` block stays in the prompt.** The
  finalizer is *skipped* whenever the turn did not end of its own accord —
  forcing a verdict out of a `max_turns` turn recreates exactly the failure
  `INCOMPLETE` was added to fix — and it can also fail. A verdict pipeline with
  no fallback turns a transient engine error into a lost triage.
- The ledger event carries `verdict_source` (`structured` | `regex` | `none`)
  and `structured_error`, because a finalizer that quietly stopped working
  otherwise looks exactly like one that is working. Watch the fallback rate,
  not the feature flag.
- The router honours a schema only for a session whose platform is in
  `sessions_io.NON_USER_PLATFORMS`. A chat turn that quietly ran a second
  completion under a grammar would be paying tokens for something nobody reads.
- Kill switch `workers.sources.backlog-selfmod.structured_verdict`, carried in
  the queue payload like the budgets so a queued item runs under the config
  that was live when it was enqueued.

Follow-ups, not done: schemas for `deep_research.parse_result` and
`backlog_implement.parse_spawned_line`.

## 6. Configuration

```yaml
workers:
  db_path: ~/lloyd/workers.db
  enabled: true
  slots: 2                      # concurrent workers
  max_attempts: 3               # before an item is poisoned
  sources:
    <name>:
      enabled: true
      interval_seconds: 1800    # how often enqueue_if_due is called
      max_inflight: 1           # concurrent items from this source
      max_duration_seconds: 3600  # the pool's wait_for cap
      priority: 55              # optional; defaults to the source's own
```

`get_sources_config()` re-reads `CONFIG` on every call, so a toggle takes
effect without a restart.

**`workers.enabled` is UI-mutable and therefore does not live only in
config.yaml.** `POST /api/workers/enable` writes it to
`data/tool_overrides.yaml`, which is untracked and merged over config.yaml at
boot. It used to `yaml.dump(CONFIG)` over `config.yaml` itself, and all three
consequences were serious: `CONFIG` is the *loaded* config, so the dump would
have written `${LIVEKIT_API_SECRET}` out expanded into a tracked file; it
flattened the comments out of a file that is mostly comments; and it left the
tree dirty, which `gate.py` and `promote.py` both refuse — so one click
silently stopped the self-modification loop. That is the same defect the Tools
page was moved off config.yaml to avoid, and this endpoint had kept it only
because nothing in the frontend calls it yet.

---

## 7. Operating it

```bash
curl -s localhost:8080/api/workers/status | jq       # pool + depth by source
curl -s 'localhost:8080/api/workers/runs?limit=20' | jq
curl -sX POST localhost:8080/api/workers/pause -d '{"paused":true}'
sqlite3 ~/lloyd/workers.db \
  "SELECT source,state,COUNT(*) FROM queue GROUP BY source,state;"
```

**Pause and drain before restarting the backend.** The pool lives in that
process, so a restart kills whatever is in flight — and if a worker turn dies
mid-flight during a selfmod landing, the connection errors it logs on the way
down land inside the guardian's observation window and get blamed on the
promotion. That is the 2026-09-06 20:14 false-positive rollback exactly. The
promoter now drains the pool itself, and `run_prompt_on_primary` refuses to
start a turn while a landing is draining.

The dashboard reads the pool through `app/routers/dashboard.py::_workers`,
off the loop via `asyncio.to_thread`; `completed` is excluded from the "open"
counts because it dominates the depth table.

---

## 8. Known limits

- **Sources cannot express a dependency on each other.** Chaining is done in
  `~/obsidian/autonomy/` frontmatter with `depends_on` and runs through
  `scheduled-task`; the queue itself has no edges.
- **One pool per machine.** `claim_next` is safe across processes, but
  `recover_claimed()` at startup assumes nothing else is running.
- **`gap-fill` and `bench-mine` have never enqueued anything here.** There is
  one gap-labelled fact in the tree and no recent ledger losers. They are
  correct and idle, not broken — but nothing has exercised their `execute`
  path against real input, so treat it as untested in production.
- **Run history grows without bound.** 4,894 runs and 4,544 queue rows today,
  6.3 MB. Fine for now; there is no retention job.
