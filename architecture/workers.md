---
segment: architecture
tags: [architecture, lloyd, workers]
type: reference
status: implemented
date: 2026-09-25
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
  that creates work. The same tick carries two deterministic chores that must
  not wait for a free slot: the poison sweep (`workers/maintenance.py`) and
  the service probe (`workers/service_probe.py`, #1359), which announces a
  supervised infra program whose declared port stays closed past its grace
  while supervisord still reports it up — the crash loop that reads RUNNING.
- **The workers** — `workers.slots` of them — claim one item at a time and
  await its source's `execute`.

`workers.db` lives in the data root (`~/lloyd-data/workers.db`, outside the
code tree, [[data-home]]) and is opened in WAL mode, because the **MCP aggregator is a
second process** that writes to it: the effect ledger writes `tool_effects`
rows from inside a tool call, `autoresearch_status` reads the queue, and the
authority gate reads its grants. A process outside the backend never runs
`start_worker_pool`, so the singleton is never initialised there and a bare
`get_queue()` raises — which is how the `autoresearch_round` tool (retired
2026-09-23) answered "work queue not available" for most of its life, with not
one row in `workers.db` carrying the `targets` payload it sent. Callers outside
the backend ask `configured_db_path()` instead of inventing a path.

---

## 2. The queue

Three tables are this file's own. `queue` is the work, `runs` is the history,
`watermarks` is whatever a source needs to remember between ticks.

Three more share the database, and each is here because *this* file owns it:
`authority_grants` and `grant_dispatch` (#534) are declared in `queue.py`'s
`GRANT_DDL` and read by `app/harness/policy.py` rather than redeclared there —
two definitions of a table whose NOT-NULL `expires_at` is the whole safety
property is how one of them stops being true — and `tool_effects` (#544) is
created by `agent_mcp/_tool_effects.py` against `configured_db_path()`. One
more shares the file without being *owned* here: `egress_events` (#628),
created by `agent_mcp/egress.py`, the egress guard's record of every
destination it saw. So `.tables` shows seven, and only the first three
belong to the pool.

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
them is never seen, and the pool reads the queue as empty and sleeps. A few
saturated sources sharing the handful of slots on this box makes that an
ordinary Tuesday.

### The KV budget gate

A source that declares `LONG_LIVED = True` — `autocode`, `autotriage`,
`deep-research`, `arch-review` — is not claimed while the primary's KV usage
is above `workers.kv_gate.max_kv_usage` (0.60); every other source claims as
before. The flag in `workers/sources/<name>.py` is the truth and this list is
a copy of it: `tests/test_kv_gate_docs.py` fails when a source flips the flag
without this sentence, the `kv_gate` comment in `config.yaml` or `vllm.md`
§6.3 being updated (#1341 — `arch-review` had been gated for a week while
both enumerations named three).
Held sources join the same `NOT IN` as the saturated ones, for the same
reason, and a held item keeps its place and its attempt.

The KV gate is one of **three** holds applied there. **`round_hold`**
(`workers.round_hold`, #1101's `exempt_bound`) keeps everything off its
exempt list unclaimed while an automod round is in flight;
`architecture/automod.md` owns its semantics. **`primary_hold`** probes
whether the primary engine is answering *at all* and holds the sources it
names (default `autocode`) while it is not — the backend coming back before
the 95 GiB table is resident used to spend items' attempt budgets on
connection errors (#1430). Its kill switch is deliberately absent from
tracked config (a gate the loop must not be able to disarm), and it has no
long-form doc yet (#1465). All three report beside `pool.kv_gate` in
`/api/workers/status`.

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
async def enqueue_if_due(queue, src_cfg) -> None | str
async def execute(item) -> dict
```

registered in `workers/sources/__init__.py`.

The scheduler calls `enqueue_if_due` once `interval_seconds` have passed
since the source's `last_enqueue_check`, then stamps it. A source that looked
and could not act *yet* returns `DECLINED` (`workers.sources`); with
`retry_seconds` in its config the stamp is back-dated so it is due again that
much sooner (`WorkerPool._scheduler_pass`). Anything else — `None`,
`ENQUEUED` — advances the full interval. autocode is the one user: a poll
that finds a promotion under observation — or whose enqueue coalesced against
the previous round's row, still `running` through its finalizer — retries in
60 s instead of 900, and its board housekeeping keeps the 900 s clock on its
own `last_housekeeping` watermark (`architecture/automod.md` §4.5d, §3.2d).

A source that sets `REPOLL_ON_COMPLETE = True` is also made due the moment one
of its runs ends: the worker loop's `finally` back-dates its
`last_enqueue_check` to the epoch (`WorkerPool._repoll_on_complete`), so the
next scheduler pass — at most 60 s later — asks it again. autocode sets it;
without it the next round was queued up to a full interval after the last one
ended. A failed write costs the early look, never the run's record.

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
resolved gap stops being re-litigated. Still a pilot: the first bundles
landed on 2026-09-23 — 3 rows in `runs` carry one (nightly
`scheduled-task` reflection runs) as of 2026-09-25 — so every rate the
health view derives from a bundle rests on a sample of three.

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
`automod_start` still refuses a *chat* turn with no Inner Voice
(`automod.require_inner_voice`), but since 2026-09-12 a worker or autonomy
session passes: the observer's measured effect on rounds was negative
(#874), and every background run has been recorded since 2026-09-10, so a
round driven from a worker session is as reviewable as one driven from an
IV session. A refusal that made `inner_voice: false` on `autocode` stop
every round from opening would be the opposite of a switch.
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

What that cost: **225 of the 498 notes then staged under `pending-research/`
had the body `(no response)`** — that staging tree did not survive the
2026-09-22 deletion of `~/lloyd` (§8); 29 notes are staged now and none is
empty. The source responsible, `domain-research`, wrote the empty
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
| `autotriage` | 55 | triages one backlog item, or consolidates one cluster | session, IV off |
| `autocode` | 40 | one gated automod round per confirmed item | session, IV off |
| `backlog-cluster` | 65 | nightly clustering of the open board for the above | none (numpy, off-loop) |
| `arch-review` | 62 | one `architecture/` doc or one functional group: check it against the tree, edit it, file the rest | session, IV off |
| `board-steward` | 68 | one board pass: proposed moves and the next item for `autocode`, recorded beside the state machine's | session (primary), IV off |
| `automod-regression` | 70 | paired A/B eval after a promotion | none (subprocess on a thread) |
| `autoresearch` | 60 | one prompt-optimisation round | its own |
| `deep-research` | 70 | one registry topic, through the deep-dive-research skill | session, IV off |
| `youtube-digest` | 45 | one tracked-channel video: transcript → vault note → Lloyd eval → backlog draft | session, IV off |
| `session-distill` | 70 | mines finished chats for gaps and patterns | direct |
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
`autoresearch` was `enabled: false` from 2026-09-08 to 2026-09-16: it had
promoted generated prompt variants straight over the live `lloyd/SOUL.md`
with no gate, test, review or revert. It is **re-armed since 2026-09-16**
(Alan pre-approved 09-13 once #506 and #876 closed) behind the conditions
that were the reason for the stop: `promote()` refuses a variant that
breaks the prompt-surface invariants and commits through
`automod_vault_land`, and `autoresearch.promotion` sets the bench bar. It
runs every 4 h, exempt from the round hold — the share of the primary it
may take beside a live round, at most 30 min in 4 h. config.yaml carries
the full history.

`autotriage` takes a **cluster** before it takes a single item — one turn
closing several duplicates is finite work where the single pool is not — which
is what `backlog-cluster` exists to prepare. `architecture/automod.md` is the
long version of both.

The IV column is `workers.sources.<name>.inner_voice`, read through
`_common.source_inner_voice` (fallback **`False`** since #1015: an unkeyed
source now reads as unobserved, because `board-steward` arrived unkeyed on
2026-09-12 and inherited the old `True` — watched on the primary every
900 s nobody had chosen). **Since 2026-09-12 ("cut 1 of
senses-not-supervision") every session-backed source sets the key false**:
observer injects were measurably harmful on unattended turns (#874 —
abandoned at iteration 38 with 44 minutes left on an invented premise,
sixteen false repetition fires in a day), and the stall-rescue, budget and
context-pressure anchors now do deterministically what the observer used to
attempt. Observation stays on for chat, where a human reads it and its
value was measured there. A `scheduled-task` run is watched when its task
file says `inner_voice: true`, which beats the fleet default
`autonomy.inner_voice` (off).

**It is meaningful only on a session row, and `/api/workers/health` reports it
tri-state to say so**: `null` means the source does not set the key, which for
a direct-path source is not "off, and you could turn it on" — nothing there can
be observed at all, because the observer is wired in `app/routers/messages.py`
and nowhere else. Reporting a flat `false` there would invite a knob that reads
as broken the one time somebody uses it. The promise is being broken today,
just not where an earlier draft of this file said: `config.yaml` no longer
sets `inner_voice` on `backlog-cluster` — #1015 removed the key and a
comment marks the omission deliberate — but the UI-written override file
re-adds `inner_voice: false` for it (among six sources), so the panel still
renders the one source that *cannot* be observed as the one that has been
switched off. Harmless (nothing reads the key on that path) and wrong in
exactly the direction the tri-state was built to prevent; the honest value
is no key, and the stray one is filed (#1464).

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
- No kill switch: `workers.sources.autotriage.structured_verdict` was
  retired on 2026-09-24, and every triage turn asks for the object.

`deep-research` and `youtube-digest` take the same path since #710: each
module's `RESULT_SCHEMA` is built from the tuples its `parse_result` validates
against, `parse_verdict(text, structured)` prefers the object and falls back to
the `RESULT:` block (which stays in both prompts), the run's `meta` carries
`verdict_source` and `structured_error`, and each has its own
`structured_verdict` kill switch carried in the payload (default on, no config
key). One behaviour moved with it: once the finalizer has run on a
youtube-digest turn, "a clean stop with text" no longer counts as a turn that
ran over a pre-existing note — it needs a verdict with a known `RESULT`, since
the finalizer only runs on a clean stop and the old fallback would pass every
such turn. Follow-up, not done: a schema for `autocode.parse_spawned_line`.

## 6. Configuration

`workers.slots` is the one number a reader is most likely to find stale in a
doc like this one. The authoritative value lives only in `config.yaml`; it is
pinned by `tests/test_loop_depth.py` to `rounds + triages + 1` (the number of
autocode rounds plus the number of autotriage turns, plus one slot so a
scheduled task is never queued behind them). The copy in the block below is a
sample of the shape, not the live number — `tests/test_doc_worker_slots_parity.py`
fails if it ever contradicts `config.yaml`.

```yaml
workers:
  db_path: ~/lloyd-data/workers.db
  enabled: true
  slots: 6                      # concurrent workers — see the note above; truth is config.yaml
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
`group_min_items` and `spawn_cap`, `deep-research` a `daily_max`,
`backlog-cluster` a `min_age_seconds`. The pool never looks at them — it
forwards `src_cfg` to `enqueue_if_due`, and most sources copy what they need
into the queue payload so a queued item runs under the config that was live
when it was enqueued.

`get_sources_config()` re-reads `CONFIG` on every call, so a toggle takes
effect without a restart.

**`workers.enabled` is UI-mutable and therefore does not live only in
config.yaml.** `POST /api/workers/enable` writes it to
`~/lloyd-data/data/tool_overrides.yaml`, which is untracked and merged over
config.yaml at boot (it moved out of the code tree with the rest of the
runtime data on 2026-09-22, after a pytest fixture teardown deleted
`~/lloyd` outright — commit `6426668b`). It used to `yaml.dump(CONFIG)` over `config.yaml` itself, and all three
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
sqlite3 ~/lloyd-data/workers.db \
  "SELECT source,state,COUNT(*) FROM queue GROUP BY source,state;"
```

**Pause and drain before restarting the backend.** The pool lives in that
process, so a restart kills whatever is in flight — and if a worker turn dies
mid-flight during an automod landing, the connection errors it logs on the way
down land inside the guardian's observation window and get blamed on the
promotion. That is the 2026-09-06 20:14 false-positive rollback exactly. The
promoter now drains the pool itself, and `run_prompt_on_primary` refuses to
start a turn while a landing is draining.

**A person's pause survives a restart; a landing's does not.** Until
2026-09-22 the pause lived only in the backend process, so every restart came
back running, and three times that day the pool claimed 3-6 jobs (4 of them
autocode rounds) before a re-pause could land. Now `WorkerPool.pause` has two
owners. An **operator** pause (the default: the route, Mission Control, the
guardian's vault trip) is written to `workers.db` (`watermarks`, source `_pool`)
and read back when the pool is built, so a restarted pool claims nothing. An
**automod** pause (`{"owner": "automod"}`, sent only by
`promote.set_pool_paused`) stays in memory, because the promoter leaves its pause
for the landing's own restart to clear; persisted, every landing would leave
the pool paused for good. An automod resume lifts only its own pause, and an
operator resume lifts both. `status().paused_by` says which is holding.
`vaultwatch.py clear` does not resume the pool; it prints the command.

**A pause stops claims, not runs.** `_worker_loop` reads the flag before
`claim_next` and nowhere else, so a job already in `source.execute` runs to its
end. That is the design the landing depends on — pause, then wait for what is
in flight — not a gap; the stop for a run in flight is its turn's
`cancel_event` (`POST /api/sessions/{id}/cancel`, the Inner Voice cancel).
`scripts/mitigation_drill.py` (#703) fires both at a synthetic run, offline and
in-process, and classifies each by what it measured: `session_cancel` as
`in-flight` with seconds-to-stop, `pool_pause` as `dispatch-only`, and a
control stubbed to a no-op as `no-op`, which exits non-zero. It refuses while
`round_hold` is engaged.

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

**`/health` lists every source the config names, plus every source the
window's rollup or queue depth has seen** — a row with `configured: false`
is the one to look at. What such a row *is* has changed: the 2026-09-22
deletion of `~/lloyd` reset both tables (§8) and the retention sweep prunes
`runs` past 30 days, so the retirements and renamings earlier drafts of
this file enumerated there (`domain-research`, `backlog-selfmod`,
`selfmod-regression`, `autoimplement`, `backlog-implement` — see MEMORY.md's
naming history) no longer appear in any window it can ask for. The
unconfigured row an operator finds today is `queue-maintenance`: the poison
sweep's own ledger identity (`workers/maintenance.py`'s `SOURCE`) — a
mechanism that records runs without being a scheduled source.
`configured: false` now has two readings: a swept-away retirement, or a
mechanism wearing a source's name.

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
- **`bench-mine` advertised an input it never opened, and had a second it
  could not read.** Before #522 its `enqueue_if_due` returned early on the ledger
  mtime, so the failed-run input its own docstring named since the first commit
  was unreachable and the queue held zero rows for its entire life. It now
  scans `AUTONOMY_RUNS_DIR/**/run_*.md` for `status: failed` on every tick,
  gated by nothing the ledger does — 104 eligible failures in the last 7 days.
  Each candidate is calibrated (10 trials, composite kept only strictly inside
  0.05–0.95, and a calibration that scored fewer than 3 is a measurement of the
  engine rather than of the task) and tagged with one escalation direction. The
  ledger half was dead for a second, dumber reason, and "idle" was never the
  right word for it: `variant_sandbox.py` writes `BASELINE_<int>` while the
  filter matched lowercase `baseline` — a case-sensitive comparison that held
  for no row the ledger ever wrote. `workers.db` therefore held 0 queue rows of
  kind `mine` against 134 of kind `mine-run` as of 2026-09-19, and the ledger
  itself held 4,080 baseline rows (370 distinct ids among 32,411 rows), none of
  them selectable. #625 made the comparison case-insensitive
  (`BM.BASELINE_ID_PREFIX`) and added the guard the widening made reachable: a
  loser may only be a trial whose `trace_status` is `success`, because
  `judge.py:177` zeroes the composite of any trace that did not complete and
  all 77 `error` baseline rows sit below the 0.6 line by construction. Both are
  pinned in `tests/test_workers_sources.py`, one of those tests crossing the
  writer→ledger→selector seam by asking `materialize_baseline` for its own id.
  **This is a mechanism fix, not an observed one.** Production has to answer
  whether the input now fires, on rows #876 had not yet frozen and then
  un-froze; until `SELECT count(*) FROM queue WHERE source='bench-mine' AND
  kind='mine'` is non-zero, nothing here has been demonstrated end to end. Note
  too that mtime cannot be read as "the ledger is fresh": a watermark matched
  the file's `stat` across an eight-day gap in which no row was appended, which
  only proves the inode was touched.

  **Production has since answered the run-failure half of this paragraph, and
  the answer is that the input fires and the mining turn does not.** 60 runs since
  2026-09-09: 6 success, 5 `skipped` (a candidate with no mechanical check is
  rejected rather than counted against the source), and **49 failed, every one
  of them `empty response (stop_reason=max_turns) — nothing written`**. The
  run-failure items are enqueued and claimed exactly as #522 intended; the turn
  that has to read a failed run and write a bench task dies at its 8-iteration
  ceiling with no text, which `TurnResult.ok` catches and records in-band, so
  nothing is written and the candidate is retired after repeats. That is the
  budget being wrong for the job, not the input being unreachable — and it is
  the same failure `TurnResult` exists to make visible, working.

  **The same ceiling kills `session-distill` more often than it kills
  `bench-mine`, and this section named only the smaller instance.** 30 days to
  2026-09-20: `bench-mine` 123 failures of 146 runs, **120 of them
  `stop_reason=max_turns`**; `session-distill` 238 failures of 632, **209 the
  same string** — 209 deaths against 394 successes, on a source whose remaining
  failures are 21 `ConnectError` (engine-side) and 3 wall-clock
  `TimeoutError`s. Exclude those two buckets from any before/after count on the
  ceiling: they are different mechanisms, and mixing them moves the number
  without moving anything. `session-distill` runs `iterations_per_step=15`
  (`workers/sources/session_distill.py`) against `bench-mine`'s then-literal
  `max_turns=8` (`workers/sources/bench_mine.py`; since #896 on 2026-09-24 it is
  `workers.sources.bench-mine.max_turns`, 12, carried in the payload, default 8)
  and dies identically — it reads a whole
  transcript and runs out of iterations before it writes a note. Raising
  session-distill's ceiling is #980's decision. What #1050 changed is that the turn was never
  told: `_worker_run_options` set no `state_anchor`, so this path had the kill
  and no warning while the chat, session-backed-worker and autonomy paths all had
  the warning, and inner voice is off for these sources by design, so no observer
  could say it instead. Both clocks now ride every direct worker turn — 75 %/90 %
  of `max_turns`, 70 %/90 % of `turn_timeout_for(source)` — pinned in
  `tests/test_worker_budget_anchor.py`. Whether the warning changes what those
  turns do is a post-landing traffic question, and this line does not claim it.
- **`gap-fill` is retired (#897, 2026-09-24).** It resolved `label: gap`
  facts and no extractor ever emitted one, so it had zero rows in `runs` for
  its entire life while stat-walking the facts tree every 300 s.
  [[workers-jobs]] §7 has the rest.
- **The `queue` table still grows without bound; `runs` no longer does.**
  The 6,268 / 5,910 / 9.4 MB reading of 2026-09-11 became history the moment
  `~/lloyd` was deleted whole on 2026-09-22 — a pytest fixture teardown, and
  every gitignored byte in it went with the code; commit `6426668b` moved
  all runtime data to `~/lloyd-data`, where both tables' oldest rows now
  read `2026-09-22T20:02Z`. `scripts/groundskeeper/retention-sweep.py` now
  prunes `runs` at 30 days (`WORKER_RUN_MAX_AGE_DAYS`, the same horizon as
  the markdown run records and the session archive — against a
  conversation's 90), and the `tool_effects` rows sharing the database prune
  themselves (14 days settled, 30 unknown). Measured 2026-09-25: 534 runs /
  493 queue rows / 1.5 MB. The sweep's own header names its sibling `queue`
  table as deliberately NOT pruned — that horizon is an open decision with
  no owner (#1466).

## Review log

- 2026-09-25: **stale** — every mechanism checked out (claim SQL, backoff,
  dedup release, recovery sweep, KV gate, source protocol, result contract,
  hold-before-claim, pause ownership, routes); the live-status prose had
  moved. Corrected: `autoresearch` re-armed 2026-09-16 behind promotion
  gates (was "off since 09-08"); Inner Voice fallback `False` since #1015
  and every session-backed source off since 2026-09-12, with
  `automod_start`'s refusal chat-only (the table said "IV on" four times);
  the seventh shared table `egress_events`; first evidence bundles 2026-09-
  23; `tool_overrides.yaml`'s path; `/health`'s `configured: false` roster
  (now `queue-maintenance`, not the swept-away retirements); §8 counts —
  both tables reset by the 2026-09-22 deletion, `runs` pruned at 30 days,
  `queue` still growing. Filed #1463–#1466.
