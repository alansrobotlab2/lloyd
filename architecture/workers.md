---
segment: architecture
tags: [architecture, lloyd, workers]
type: reference
status: implemented
date: 2026-09-11
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
  60s tick)                                      │ claim_next(worker_id, quota, held)
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

`workers.db` lives beside the repo (`~/lloyd/workers.db`, gitignored by the
`*.db` rule) and is opened in WAL mode, because the **MCP aggregator is a
second process** that writes to it: `autoresearch_round` enqueues a round from
there, the effect ledger writes `tool_effects` rows from inside a tool call,
and the authority gate reads its grants. A process outside the backend never
runs `start_worker_pool`, so the singleton is never initialised there and a
bare `get_queue()` raises — which is how the round tool answered "work queue
not available" for its entire life, with not one row in `workers.db` carrying
the `targets` payload it sends. Callers outside the backend ask
`configured_db_path()` instead of inventing a path.

---

## 2. The queue

Three tables are this file's own. `queue` is the work, `runs` is the history,
`watermarks` is whatever a source needs to remember between ticks.

Three more share the database, and each is here because *this* file owns it:
`authority_grants` and `grant_dispatch` (#534) are declared in `queue.py`'s
`GRANT_DDL` and read by `app/harness/policy.py` rather than redeclared there —
two definitions of a table whose NOT-NULL `expires_at` is the whole safety
property is how one of them stops being true — and `tool_effects` (#544) is
created by `agent_mcp/_tool_effects.py` against `configured_db_path()`. So
`.tables` shows six, and only the first three belong to the pool.

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
once already: `autocode` sat at 80, behind research jobs at 70 that
arrive every few minutes, and the first unattended round had no path to a
slot. It is at 40 now.

`max_inflight` per source is applied **in the SQL**, by excluding saturated
sources from the query. It used to select fifty rows and skip over-quota ones
in Python, which is a different thing entirely once a saturated source has
more than fifty items queued: they fill the window, the claimable row behind
them is never seen, and the pool reads the queue as empty and sleeps. Three
sources sharing two slots on this box makes that an ordinary Tuesday.

### The KV budget gate

A source that declares `LONG_LIVED = True` — `autocode`, `autotriage`,
`deep-research` — is not claimed while the primary's KV usage is above
`workers.kv_gate.max_kv_usage` (0.60); every other source claims as before.
Held sources join the same `NOT IN` as the saturated ones, for the same
reason, and a held item keeps its place and its attempt.

It exists because of what the 09-09 stall turned out to be
(`vllm.md`): long-lived agent loops evicting each
other's prefixes. Each iteration re-submits a 100-200k context; between
iterations that prefix sits only in the engine's free pool, and three or
four long residents on the old 398k pool meant one of them came back cold on
most iterations. Short-lived jobs never came back to miss — seven youtube
digests at 81.5% KV ran clean. So the gate decides *who* starts, not how
many, and the classification is a static module attribute a reviewer can
read rather than a runtime guess at context size.

It reads `app/engine_pressure.py`'s background samples, never the engine,
so a claim does not wait on HTTP, and it judges their **median over the last
minute** (`kv_gate.window_seconds`), not the newest one. Measured on the FP8
build: a cold 200k prefill drives the gauge from 0.20 to 0.96 over its 21 s
and it drops to 0.50 the moment the prompt is in — a prompt being built
references ~2.5x its resident footprint — so a last-sample gate would hold
every long-lived job for the length of every prefill. No reading — sampler
off, engine down, a sample older than 30 s — means open. State is `pool.kv_gate` in
`/api/workers/status` and a row on the dashboard's worker panel, and each
transition logs one line.

### What a claim binds

Two contextvars and a collector are set around the claimed job, and the pair
of scopes is the part worth keeping straight because they answer opposite
questions about the same run.

- **`grant_scope_for(item)`** — *whose authority is this turn borrowing.* An
  autonomy task is `autonomy-task:<id>`, everything else `worker:<source>`,
  because that is the difference between a human granting `email_send` to the
  nightly mail job and granting it to whatever runs on that source next.
- **`effect_scope_for(item)`** — *which run's effects must not happen twice.*
  `item:<source>:<id>`, the queue item, because that is the unit a retry
  re-runs: `run_id` is minted per attempt, so keying on it would let attempt 2
  fire a fresh effect, which is the bug. Deliberately **not** the grant scope —
  `autonomy-task:39` is stable across every run that task will ever have,
  which is right for a permission and would suppress a legitimate second
  effect forever.
- **`current_run_sessions`** — the transcripts the job created, collected by
  the pool rather than returned by each source. §7.

Both scopes are contextvars bound in the pool's own task, so they do **not**
survive the loopback POST a session-backed source makes; they travel in the
request body instead (§4).

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
| `claims` | `{claim, check}` pairs to verify against disk — see below |

**`skipped` is not a failure and not a success**: the source looked and there
was nothing to do. That distinction is not cosmetic. `automod-regression`
signalled "I could not measure anything" by returning `{"skipped": reason}` —
a key where a status belongs — so all 22 of its runs were recorded as
successes with an empty summary. For a detector whose entire premise is *a
missing noise floor means cannot evaluate, never no regression*, that is the
one failure it cannot afford, and it was invisible in the runs table and on
the dashboard alike. `normalize_result` now reads a bare `skipped` key as a
status, and logs any result with neither summary nor artifact.

### Evidence, when a source offers any

`summary` is what the model *said*. `claims` (#525) is what was checked.
A source may end its run with `{claim, check}` pairs; the pool re-runs each
check against the filesystem at the moment the ledger row is written — off the
loop, because a check reads files — and stores the graded bundle in
`runs.claims_json`. `workers/evidence.py` is stdlib-only and never LLM-judged,
because a claim graded by a model is the narration this replaces, one layer up.
Two rules carry it:

- **A claim that cannot be evaluated is `insufficient`, not `verified`.** A
  gate that reads its own missing input as a pass is a failure mode this tree
  has recorded three separate times. A bundle with nothing in it reports a gap
  and a rate of `None`, never a clean zero.
- **Presence of the key is the scope switch.** A source that emits no `claims`
  gets no bundle at all and is counted as `runs_without_bundle` rather than
  scored as a clean check. An *empty list* is different and deliberate: the
  source is in the pilot and its model asserted nothing, which is a gap.

Whatever failed to verify is carried into the next run of that same task, as a
`gaps:<task_id>` watermark — a gap that lives only in the prose of the last run
is a gap nobody reads — and an empty list is written on a clean run so a
resolved gap stops being re-litigated. Still a pilot: **no row in `runs`
carries a bundle yet** (checked 2026-09-11), so every number the health view
derives from one is currently derived from nothing.

### Event-loop discipline

**Everything here runs on the backend's one event loop** — the loop that
serves every HTTP request and streams every chat turn. A source that blocks it
does not slow the pool down, it stops Lloyd answering.

So: no `subprocess.run`, no `urlopen`, no unbounded file walk directly inside
`execute` or `enqueue_if_due`. Put the body in a plain function and hand it to
`asyncio.to_thread`. `automod_regression.execute` broke this with two
900-second eval arms and a `git worktree add` between them; the 109-second run
in its history is 109 seconds during which the backend answered nothing. It is
the shape of the fix that is worth copying: `execute` is now three lines that
`await asyncio.to_thread(_execute_blocking)`, and the whole comparison lives in
the sync function — async only at the edge, so there is no seam inside it where
a later edit can reintroduce a blocking call on the loop.
`test_workers_pool.py::test_no_source_blocks_the_event_loop_in_execute` greps
for it, and the queue's own writes are hopped onto threads for the same
reason.

---

## 4. Two ways to run a turn, and how to choose

| | `run_prompt_on_primary` | `run_prompt_in_session` |
|---|---|---|
| calls | `run_query` directly | `POST /api/message/stream` |
| session | yes, `platform: worker` | yes, `platform: worker` |
| transcript | recorded by `app/run_recorder.py` | persisted by the chat path |
| Inner Voice | never | per source, `workers.sources.<name>.inner_voice` |
| #534 grant gate | its own `HookRegistry` | installed by the endpoint, keyed on the session's platform |
| change ledger | yes — a `turn_id` is minted per run | yes |
| used by | research and mining | anything judging or changing Lloyd's code |

Until 2026-09-10 the left column read *none* and *discarded*: once a
`gap-fill` or `session-distill` run's text had been collected, what it did
existed nowhere. Both paths are **recorded** now, and the difference between
them is **observation** — which is the axis worth choosing on.
`app/routers/messages.py` is still the only turn path that attaches the
observer, so work a human must be able to audit as it happens goes through it
rather than having the observer wiring copied into a second place.
`automod_start` refuses a turn with no Inner Voice, which is what makes that
structural rather than a convention; a transcript is not an observer.
`architecture/background-runs.md` is the long version of both axes.

Both scopes §2 binds have to cross a process seam to reach the right column.
`run_prompt_on_primary` is in the pool's own task and reads the contextvars
directly; `run_prompt_in_session` posts to the backend, which handles the
request in a different task, so `grant_scope` and `effect_scope` ride in the
request body and the endpoint honours them only for a non-user platform. The
endpoint arms the grant gate on the session's platform regardless, so a caller
that omits the scope is still gated — what the scope buys is the *right* one.
Skipping that hop is not a degraded gate, it is the wrong bill: the effect
ledger ran with 25 items and 0 rows for its first sixteen hours because every
session-backed source read an empty contextvar.

There is a third in-process shape, `run_prompt_with_run_state` (#529): the same
model, priority and tool policy — all three turn builders go through
`_worker_run_options`, and a test asserts none of them constructs its own
`RunOptions` — but each step carries a schema-validated state object instead of
one transcript that only ever grows. It writes its own record
(`run_dir/state-trace.ndjson`) rather than a conversation, and **nothing in
production calls it yet**, so it is not a third column above.

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
the pool records a failure and backs off, and for `autotriage` the retry
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

One row each below; [[workers-jobs]] is one section each — what wakes a source,
what it reads, what it writes, and the measured state of it.

| source | prio | what it does | turn path |
|---|---|---|---|
| `scheduled-task` | 10–70 | runs `~/obsidian/autonomy/*.md` via `autonomy.run_task` | its own, recorded; IV per task |
| `autotriage` | 55 | triages one backlog item, or consolidates one cluster | session, IV on |
| `autocode` | 40 | one gated automod round per confirmed item | session, IV on |
| `backlog-cluster` | 65 | nightly clustering of the open board for the above | none (numpy, off-loop) |
| `arch-review` | 62 | one `architecture/` doc or one functional group: check it against the tree, edit it, file the rest | session, IV on |
| `automod-regression` | 70 | paired A/B eval after a promotion | none (subprocess on a thread) |
| `autoresearch` | 60 | one prompt-optimisation round | its own |
| `deep-research` | 70 | one registry topic, through the deep-dive-research skill | session, IV off |
| `youtube-digest` | 45 | one tracked-channel video: transcript → vault note → Lloyd eval → backlog draft | session, IV on |
| `session-distill` | 70 | mines finished chats for gaps and patterns | direct |
| `gap-fill` | 50 | resolves `label: gap` facts | direct |
| `bench-mine` | 80 | new bench tasks from failed autonomy runs, and from baseline losses when the ledger has any it can read | direct |

Priorities are each module's `DEFAULT_PRIORITY` unless config overrides it, and
two do: `youtube-digest` (45, so a digest beats low/background autonomy work
for a slot while an autocode round still goes first) and `arch-review` (62,
which is also its module default — set explicitly because doc maintenance must
sit below the digest and above the night's clustering, and reading that from
one file rather than two is worth the duplication). `scheduled-task` is a
range because it maps each task's own frontmatter through `_PRIORITY_MAP` —
critical/high/medium/low/background to 10/20/30/50/70 — which is why it can
both preempt everything and sit behind everything in the same tick.
`autoresearch` has been `enabled: false` since 2026-09-08: it promoted
generated prompt variants straight over the live `lloyd/SOUL.md` with no gate,
test, review or revert. config.yaml carries the full reason and what has to
close before it is re-armed.

`autotriage` takes a **cluster** before it takes a single item — one turn
closing several duplicates is finite work where the single pool is not — which
is what `backlog-cluster` exists to prepare. `architecture/automod.md` is the
long version of both.

The IV column is `workers.sources.<name>.inner_voice`, read through
`_common.source_inner_voice` (default `True`, since that is what the
session-backed sources do). A `scheduled-task` run is watched when its task
file says `inner_voice: true`, which beats the fleet default
`autonomy.inner_voice` (off).

**It is meaningful only on a session row, and `/api/workers/health` reports it
tri-state to say so**: `null` means the source does not set the key, which for
a direct-path source is not "off, and you could turn it on" — nothing there can
be observed at all, because the observer is wired in `app/routers/messages.py`
and nowhere else. Reporting a flat `false` there would invite a knob that reads
as broken the one time somebody uses it. Which is exactly what config.yaml does
to `backlog-cluster` today: it sets `inner_voice: false` on a source whose
`execute` runs no agent turn at all — a numpy pass over stored vectors plus a
few JSON calls — so the panel renders the one source that *cannot* be observed
as the one that has been switched off. Harmless (nothing reads the key on that
path) and wrong in the direction the tri-state was built to prevent; the honest
value is no key.

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

### session-distill selects four ways, and each one is scar tissue

A session is distilled **once**, **after it goes quiet**, **not while a turn is
running in it**, and **only if it is a user session**.

- *Once.* Selection keyed on an `mtime > last_mtime` watermark, and a
  session's mtime advances with every message. An active chat therefore
  crossed the watermark again on the very next tick, forever: **one chat was
  distilled 44 times, another 22, and the worst eight account for 138 runs.**
  The dedup key could not help, because completion releases it by design.
  Per-session `done:` markers replace the moving cursor — and they fix its
  mirror image too, where a session skipped for being busy was left stranded
  below the advanced watermark and never looked at again.
- *Quiet.* A chat still being typed into is not a transcript to learn from:
  `_QUIET_SECONDS` is 30 minutes since the file's last write.
- *No turn running.* `is_session_active` is a second, cheaper gate than the
  clock, and it is not redundant with it — a turn that has been thinking for
  forty minutes has a file older than the quiet window and a conversation that
  is not finished.
- *A user session.* `platform: worker` and `platform: autonomy` sessions are
  this system talking to itself; 12 of Lloyd's own triage transcripts had been
  mined back in as observations about the user before this gate existed.
  `sessions_io.NON_USER_PLATFORMS` is the one definition of that.

---

## 5.1 Structured verdicts

`autotriage`'s verdict was parsed out of `VERDICT:` / `SURFACE:` lines by
regex. That works until a turn words it slightly differently, and then a
`confirmed` is recorded as `unverifiable` and an item is retired for a
formatting reason.

`run_prompt_in_session(..., final_schema=...)` asks the harness for one extra
completion after the turn ends, restating its conclusion under a JSON schema
(`app/harness/finalizer.py`). It returns `structured` and `structured_error`
beside `text` and `stop_reason`.

- `scripts/automod/backlog.TRIAGE_VERDICT_SCHEMA` is built from
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
- Kill switch `workers.sources.autotriage.structured_verdict`, carried in
  the queue payload like the budgets so a queued item runs under the config
  that was live when it was enqueued.

Follow-ups, not done: schemas for `deep_research.parse_result` and
`autocode.parse_spawned_line`.

## 6. Configuration

```yaml
workers:
  db_path: ~/lloyd/workers.db
  enabled: true
  slots: 2                      # concurrent workers
  max_attempts: 3               # before an item is poisoned
  kv_gate:                      # §2, "The KV budget gate"
    enabled: true
    max_kv_usage: 0.60          # hold LONG_LIVED sources above this
    window_seconds: 60          # the median window, not the last sample
  sources:
    <name>:
      enabled: true
      interval_seconds: 1800    # how often enqueue_if_due is called
      max_inflight: 1           # concurrent items from this source
      max_duration_seconds: 3600  # the pool's wait_for cap (900 if absent)
      priority: 55              # optional; defaults to the source's own
      inner_voice: true         # session-backed sources only; see §4
```

Anything past those keys is the source's own: `autotriage` reads
`group_min_items` and `structured_verdict`, `deep-research` a `daily_max`,
`backlog-cluster` a `min_age_seconds`. The pool never looks at them — it
forwards `src_cfg` to `enqueue_if_due`, and most sources copy what they need
into the queue payload so a queued item runs under the config that was live
when it was enqueued.

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

**`workers.sources.<name>.inner_voice` is UI-mutable the same way.** The
override merge honours that one key per source and nothing else: an override
may not add a source, enable one, or change its budget. A disagreement with
config.yaml is logged, like the other overridden keys, because a fresh clone
boots into whatever the tracked file says.

---

## 7. Operating it

```bash
curl -s localhost:8080/api/workers/status | jq       # pool + depth by source
curl -s localhost:8080/api/workers/health | jq       # per-source outcomes + recent runs
curl -s 'localhost:8080/api/workers/runs?limit=20' | jq
curl -s 'localhost:8080/api/background/sessions?limit=20' | jq   # the transcripts
curl -sX POST localhost:8080/api/workers/pause -d '{"paused":true}'
sqlite3 ~/lloyd/workers.db \
  "SELECT source,state,COUNT(*) FROM queue GROUP BY source,state;"
```

**Pause and drain before restarting the backend.** The pool lives in that
process, so a restart kills whatever is in flight — and if a worker turn dies
mid-flight during an automod landing, the connection errors it logs on the way
down land inside the guardian's observation window and get blamed on the
promotion. That is the 2026-09-06 20:14 false-positive rollback exactly. The
promoter now drains the pool itself, and `run_prompt_on_primary` refuses to
start a turn while a landing is draining.

The dashboard reads the pool through `app/routers/dashboard.py::_workers`,
off the loop via `asyncio.to_thread`; `completed` is excluded from the "open"
counts because it dominates the depth table.

**A run row names its transcripts.** `meta.session_ids` lists every session
the claimed job created. The pool collects them itself
(`sessions_io.current_run_sessions`, bound around the job like
`current_scope`), not by asking the source to return them, and writes them on
the timeout and exception branches too — a timed-out run is the one most
worth opening. Mission Control's **Background** tab is where both are read:
runs grouped by producer, and a Sources view over `/api/workers/health`.
That endpoint exists because `/api/workers/status` reports what a source is
*allowed* to do and how much is queued, and nothing joined a source to its
outcomes. Its `fail_rate` is `null` over zero runs rather than 0.0.

**`/health` lists every source the config names, plus every source `runs` or
`queue` has ever seen**, which is why an operator reading it today finds
`domain-research`, `backlog-selfmod`, `selfmod-regression`, `autoimplement`
and `backlog-implement` beside the live ones. They are retirements and
renamings (see MEMORY.md's naming history), not drift, and their history is
worth keeping readable — a row with `configured: false` is the one to look at.

The other half of the router is the **pending-research review surface** —
`GET /api/workers/pending`, `/pending/read`, and `POST /pending/promote` /
`/pending/reject`. That is the human gate on everything the direct-path sources
write: a `bench-mine` candidate is staged, not adopted, and promotion moves it
into the vault under a per-source default destination (`_DEFAULT_DEST`), with
`review_status` rewritten on the way. Lloyd writing the tasks that grade Lloyd
is the known self-grading failure mode, and this is where it is stopped.

---

## 8. Known limits

- **Sources cannot express a dependency on each other.** Chaining is done in
  `~/obsidian/autonomy/` frontmatter with `depends_on` and runs through
  `scheduled-task`; the queue itself has no edges.
- **One pool per machine.** `claim_next` is safe across processes, but
  `recover_claimed()` at startup assumes nothing else is running.
- **`bench-mine` advertised an input it never opened, and still has one it
  cannot read.** Before #522 its `enqueue_if_due` returned early on the ledger
  mtime, so the failed-run input its own docstring named since the first commit
  was unreachable and the queue held zero rows for its entire life. It now
  scans `AUTONOMY_RUNS_DIR/**/run_*.md` for `status: failed` on every tick,
  gated by nothing the ledger does — 104 eligible failures in the last 7 days.
  Each candidate is calibrated (10 trials, composite kept only strictly inside
  0.05–0.95, and a calibration that scored fewer than 3 is a measurement of the
  engine rather than of the task) and tagged with one escalation direction. The
  ledger half is still dead, and "idle" was never the right word for it:
  `variant_sandbox.py` writes `BASELINE_<int>` while the filter matches
  lowercase `baseline`, so 3,893 baseline rows — 648 of them recent losers —
  have never been visible to it. That is #625, still open.

  **Production has since answered the last sentence of this paragraph, and the
  answer is that the input fires and the mining turn does not.** 60 runs since
  2026-09-09: 6 success, 5 `skipped` (a candidate with no mechanical check is
  rejected rather than counted against the source), and **49 failed, every one
  of them `empty response (stop_reason=max_turns) — nothing written`**. The
  run-failure items are enqueued and claimed exactly as #522 intended; the turn
  that has to read a failed run and write a bench task dies at its 8-iteration
  ceiling with no text, which `TurnResult.ok` catches and records in-band, so
  nothing is written and the candidate is retired after repeats. That is the
  budget being wrong for the job, not the input being unreachable — and it is
  the same failure `TurnResult` exists to make visible, working.
- **`gap-fill` is idle and still untested in production.** Three
  `label: gap` occurrences across two fact files today, and **zero rows in
  `runs` for its entire life** — not "few", none. Correct and idle, not broken,
  but nothing has ever exercised its `execute` path against real input.
- **Run history grows without bound.** 6,268 runs and 5,910 queue rows on
  2026-09-11, 9.4 MB — up ~1,400 runs in the three days since this was last
  counted at 4,894 / 4,544 / 6.3 MB. Fine for now; there is no retention job.
  The session files these runs now write *are* archived — at 30 days against a
  conversation's 90, by `scripts/groundskeeper/retention-sweep.py` — and the
  `tool_effects` rows sharing the database prune themselves (14 days settled,
  30 unknown). `queue` and `runs` are the two that only ever grow.
