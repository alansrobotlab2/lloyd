---
segment: architecture
tags: [architecture,lloyd]
status: active
type: reference

---

# Autonomy System Architecture

**Created:** 2026-03-22
**Last updated:** 2026-09-03 (full rewrite from audited reality — the previous
version was truncated mid-table, described 19 tasks and a four-agent model that
no longer exists, and pointed at a run-record path that was never used.)

## Overview

Scheduled markdown tasks, each running one skill through the in-process agent
harness against a local vLLM server. There are no "agent types" — that model is
gone. A task is a file; the scheduler decides when it is due; the worker pool
runs it.

## Storage

| What | Where |
|---|---|
| Task files | `~/obsidian/autonomy/{id}-{slug}.md` |
| Archived tasks | `~/obsidian/autonomy/_archived/` |
| Run records | `~/lloyd/autonomy-runs/{task-id}/run_{id}_{ts}.md` |
| Queue + run history | `~/lloyd/workers.db` (`queue`, `runs`, `watermarks`) |
| Skills | `~/obsidian/skills/{skill_name}/SKILL.md` |

Run records live under `~/lloyd`, **not** in the vault, and are not indexed by
qmd. A task whose findings need to be searchable must write them into an indexed
vault segment (task #78 does this correctly, via `memory/vault-maintenance/`).

## Dispatch path

```
workers/pool.py  _scheduler_loop     every 60s
  └─ workers/sources/scheduled_task.py  enqueue_if_due
       ├─ recover_stuck_tasks()           reset in_progress past its timeout
       ├─ per-model vLLM /health gate     skip tasks whose server is down
       ├─ stall alarm                     due + overdue + NOT queued
       └─ autonomy.get_due_tasks()  ──►  queue.enqueue(dedup_key=task:<id>)

workers/pool.py  _worker_loop  x2 slots
  └─ scheduled_task.execute
       └─ autonomy.run_task(task_id, max_duration=<source cap>)
            └─ app.harness.run_query  ──►  vLLM
```

`workers.slots: 2` is shared across **all** sources (scheduled-task,
autoresearch, session-distill, gap-fill, domain-research, bench-mine), so
autonomy competes with knowledge acquisition for the same two slots.

## When is a task due?

`autonomy._is_task_due` — every gate must pass:

1. `status == "up_next"`. `in_progress` is excluded (it used to be dispatchable,
   so a running task could be started again).
2. Has a `skill_name` or `skill_path` that resolves.
3. Interval elapsed since `last_run`. Interval comes from `runs_per_day` if set,
   else `frequency` (`hourly`/`every-15min`/`daily`/`weekly`). A task with an
   active hour-window gets `min(1h, 25% of interval)` of slack, because
   `last_run` is a *completion* time and drifts later every cycle.
4. **Not in failure cooldown** — see below.
5. `depends_on` satisfied: the dependency succeeded within half this task's
   interval AND more recently than this task's own last success. A failed
   upstream does not satisfy it. `stale_bypass_hours` overrides the freshness
   requirement once the dependency is that stale and not currently running.
6. `preferred_hours` contains the current **machine-local** hour. If unset, the
   hour is derived from `scheduled_at` when it parses as `HH:MM`.

## Failure handling

A run ends in exactly one of three states, and each is recorded:

| Outcome | `failure_count` | Status | Notes |
|---|---|---|---|
| success | reset to 0 | `up_next` | sets `last_run` **and** `last_attempt` |
| `task` failure | +1 | `up_next`, or `failed` at `max_retries` | timeout, exception, or an empty response after real work |
| `infra` failure | unchanged | `up_next` | fast empty response, connection error — a model-server hiccup must not disable the fleet |

**Cooldown:** `min(600 · 2^(n-1), max(interval, 6h))` — 10m, 20m, 40m, 80m…
The gate is "`last_attempt` is newer than `last_run`", which is what "the last
attempt failed" means. `last_run` deliberately tracks only successes, because
the dependency freshness gate reads it.

**Terminal state:** at `max_retries` consecutive task-failures the status becomes
`failed`, one Discord alert fires, and the task stops consuming GPU until a
human drags it back to Up Next. Keep `max_retries` at 3 or more: with a real
terminal state, 1 or 2 means a single transient timeout disables a healthy task.

**Empty responses are failures.** They were once recorded as successes, which
advanced `last_run`, reset `failure_count` and unblocked dependents — a run that
did nothing was indistinguishable from one that worked.

**Timeouts:** the effective timeout is `min(timeout_seconds, source cap − 30s)`,
so `run_task`'s own handler always wins the race and writes a record. When the
two caps were equal the pool's `wait_for` cancelled the coroutine first and the
run vanished with no record at all.

## Models

| Alias | Port | GPU | Used by |
|---|---|---|---|
| `primary` | 8096 | RTX PRO 6000 (96 GB) | analysis, the nightly chain, anything context-heavy |
| `secondary` | 8091 | RTX 3090 #2 (24 GB) | thin script wrappers and mechanical reports |

Set `model: secondary` in a task's frontmatter to route it; `run_task` resolves
the base URL through `config.models.<alias>.env.ANTHROPIC_BASE_URL`. vLLM runs
with `--scheduling-policy priority`: interactive chat sends 0, autonomy sends 1,
batch jobs (e.g. graph classification) should send 2.

## Observability

- `GET /api/autonomy/health?days=N` — per-task runs, failure rate, timeouts,
  empty runs, `[SILENT]` rate, GPU-hours, wasted hours, consecutive failures,
  plus fleet totals. Joins `runs` to `queue` to recover the task id for rows
  written before the pool passed one through.
- MCP tool `autonomy_health` returns the same JSON (it proxies the endpoint,
  because agent_mcp is a separate process and does not hold the queue singleton).
- The Autonomy page shows this as a strip above the kanban board.
- Task #76 consumes it daily and can pause a chronically failing task.

## Task inventory

| ID | Task | Freq | Model | Dep | Hours | Timeout | Status |
|----|------|------|-------|-----|-------|---------|--------|
| #24 | Data Pipeline | 24/day | primary | — | — | 1800s | up_next |
| #30 | Intelligence Pipeline Scan & Score | 3/day | primary | — | — | 1800s | up_next |
| #35 | Daily Backlog Triage | daily | primary | — | — | 1800s | up_next |
| #36 | Groundskeeper Survey | daily | primary | — | — | 180s | up_next |
| #38 | Nightly Reflection — Signals | daily | primary | — | 22,23,0,1,2,3,4 | 1800s | up_next |
| #39 | Nightly Reflection — Knowledge Write | daily | primary | #42 | 23,0,1,2,3,4 | 2400s | up_next |
| #40 | Nightly Reflection — Config | daily | primary | #39 | 23,0,1,2,3,4 | 1800s | up_next |
| #42 | Nightly Reflection — Knowledge Analysis | daily | primary | #38 | 22,23,0,1,2,3,4 | 1800s | up_next |
| #47 | Dream Consolidation | 0.14/day | primary | #40 | — | 1800s | up_next |
| #48 | Entity Resolution Sweep | daily | primary | — | — | 1800s | up_next |
| #51 | Conversation Relation Linking | daily | secondary | #56 | 23,0,1,2,3,4 | 1800s | up_next |
| #52 | Deep Dive Research | daily | primary | #65 | — | 1800s | up_next |
| #53 | Documentation Digester | daily | primary | — | — | 900s | up_next |
| #54 | Cross-Domain Synthesis | 0.14/day | primary | — | — | 1800s | up_next |
| #56 | Nightly Trajectory Extraction | daily | primary | — | 1,2 | 300s | up_next |
| #57 | Nightly Trajectory Mining | daily | primary | #56 | 23,0,1,2,3,4 | 300s | up_next |
| #58 | Nightly Skill Consolidation | daily | primary | #57 | — | 600s | paused |
| #60 | Knowledge Health Report | daily | primary | — | 4 | 600s | up_next |
| #65 | Research Queue Generator | daily | primary | — | — | 900s | up_next |
| #67 | Semantic Entity Resolution | 0.14/day | primary | — | — | 3000s | up_next |
| #68 | Email & Calendar Triage | every-15min | primary | — | — | 300s | up_next |
| #70 | Skill Lint Sweep | weekly | secondary | — | — | 300s | up_next |
| #74 | KG Mention Classifier | daily | primary | #48 | — | 3000s | up_next |
| #75 | AI Engineer YouTube Monitor | 96/day | secondary | — | — | 600s | up_next |
| #76 | Queue Health Check | daily | primary | — | 6 | 600s | up_next |
| #77 | Weekly Backlog Hygiene | weekly | secondary | — | — | 600s | up_next |
| #78 | Orphaned Reference Cleanup | weekly | secondary | — | — | 600s | up_next |
| #79 | Retention Sweep | weekly | secondary | — | 22,23,0,1,2 | 600s | up_next |
| #80 | OKF Conformance Check | weekly | secondary | — | — | 300s | up_next |
| #81 | QMD Index Maintenance | daily | primary | — | 5 | 3000s | up_next |
| #82 | Nightly Retrieval Eval | daily | primary | — | 6 | 900s | up_next |

Paused/archived tasks keep their files; `_archived/` holds retired ones with an
`archived_reason`.

## Design principles

1. **One job per task.** Overlap produced eight duplicate tasks by June 2026.
2. **Fail loudly, then back off.** Silence is the dangerous failure mode: a
   parse error once removed 34 of 40 tasks from the schedule with no signal.
3. **Claim your output early.** A job that investigates exhaustively and writes
   at the end produces nothing when it runs out of turns. Write a skeleton
   first and enrich it in place (task #42 is the reference).
4. **Scripts belong in timers, not in agent turns.** A 40-minute scan cannot
   survive a tool-call timeout; run it from systemd and let the task read the
   result (task #36).
5. **Nothing is write-only.** A task whose output has no consumer is waste.
6. **Measure, do not assume.** #82 evaluates retrieval nightly so the nightly
   writes are checked rather than believed.
