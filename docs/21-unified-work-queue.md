# Unified Work Queue for Lloyd — Autonomy + KG + Background Research

## Context

The local GPU (RTX 6000 Blackwell, Qwen3.5-122B on vLLM, 8 decode slots) sits idle most hours. Three separate execution paths currently exist:

1. **Time-based autonomy scheduler** ([autonomy.py](autonomy.py)) — 60s ticker, one task at a time, serialized via `_ticker_running` lock. Reads task defs from `~/obsidian/autonomy/*.md`. Most tasks are hourly/daily/nightly, so most ticks are idle.
2. **Vault-change KG pipeline** ([docs/20-realtime-kg-pipeline.md](docs/20-realtime-kg-pipeline.md)) — qmd inotify watcher → `POST /api/autonomy/vault-change` → 4 sequential skills on secondary Gemma model. Serialized via `_realtime_pipeline_running` lock + `_rerun_pending` coalescing. Idle whenever vault is quiet.
3. **Autoresearch** ([docs/19-autoresearch-gameplan.md](docs/19-autoresearch-gameplan.md), scaffold uncommitted at `scripts/autoresearch/` + `agent_mcp/autoresearch.py`) — Karpathy-style variant → eval → promote. Designed to run nightly for 120 min. ~22 hours/day unused.

On top of that, `label=gap` facts accumulate from live sessions with no resolver. No knowledge-acquisition loop exists.

**Goal:** collapse all three execution paths into a single **persistent work queue** drained by a continuous worker pool. The queue becomes the one place work gets scheduled, prioritized, and observed. The user's design decisions:

- **Single pool, primary 122B only, depth up to 8.** Rely on vLLM priority scheduling — interactive user requests at `priority=0` preempt workers at `priority=2`. No separate secondary-model pool.
- **No idleness check.** Just keep workers draining; vLLM handles fairness.
- **Task definition files stay canonical.** `~/obsidian/autonomy/*.md` remain human-editable. A new `scheduled-task` source reads them and enqueues due tasks.
- **Queue replaces KG inotify trigger.** The `kg-pipeline` source polls vault state each tick and enqueues the 4-step chain when changes detected. Drop `/api/autonomy/vault-change` endpoint and the inotify→HTTP hook.
- **Four knowledge-acquisition outputs:** gap-fill, session-distill, bench-mine, domain-research — all staged to `~/obsidian/pending-research/` for review before promotion.

## What this replaces / removes

| Today | After |
|---|---|
| `autonomy_tick()` executes due tasks inline | `autonomy_tick()` only enqueues via `scheduled-task` source |
| `_ticker_running` serialization lock | Removed. Queue dedup + `max_inflight` per source replaces it |
| `POST /api/autonomy/vault-change` endpoint | Removed |
| qmd-watcher's post-embed curl hook | Removed (the watcher still does `qmd update && qmd embed`; KG pipeline runs from queue on timer) |
| `_realtime_pipeline_running` lock + `_rerun_pending` flag | Removed. Dedup key `kg-pipeline:chain` handles coalescing |
| Secondary Gemma as realtime target (port 8091/8093) | KG steps run on primary 122B at `priority=2` alongside everything else |
| Autoresearch as nightly 120-min window | Autoresearch enqueues hourly via the queue |
| `_pipeline/autonomy-watermarks.json` | Kept for now (memory_capture session mtime, etc.); new sources use SQL `watermarks` table |

Latency tradeoff on KG: vault change → KG graph update goes from ~2s (inotify) to ~60s worst-case (next tick). Acceptable — KG updates are asynchronous; live sessions read vault via MCP tools, not from KG cache.

## Approach

A long-lived asyncio worker pool inside the existing [server.py](server.py) process, drained from a SQLite-backed queue. Each source owns its own `enqueue_if_due()` (evaluated every 60s via `autonomy_tick`); up to 8 workers claim + execute items concurrently. Autoresearch, KG pipeline, and scheduled tasks are **wrapped**, not redesigned.

```
autonomy_tick (60s)
    └─→ for source in sources: source.enqueue_if_due(queue)

worker_pool (N=8 asyncio tasks, always running)
    └─→ claim highest-priority queued item
    └─→ source.execute(item) via priority-proxy @ priority=2
    └─→ write run record + artifact
```

All agent LLM calls route through the existing priority proxy at port 8097 with `priority=2`. vLLM's priority scheduler preempts them for interactive `priority=0` user traffic.

## Work sources

| Source | Interval | Writes to | Wraps / reuses |
|---|---|---|---|
| `scheduled-task` | 60s | existing `autonomy-runs/{task_id}/` | reads `~/obsidian/autonomy/*.md`; respects `frequency`, `preferred_hours`, `depends_on` |
| `autoresearch` | 3600s | `_pipeline/research/` | `scripts/autoresearch/run_round.py` |
| `gap-fill` | 300s | `pending-research/gaps/` | `mcp-servers/memory.py::fact_query(label="gap")` |
| `session-distill` | 1800s | `pending-research/distill/` | `periodic-memory-capture-lloyd.md` watermark pattern |
| `bench-mine` | 7200s | `pending-research/bench/` | session failures + autoresearch ledger losers |
| `domain-research` | 600s | `pending-research/domain/` | reads new `~/obsidian/lloyd/research-queue.md` |
| KG pipeline steps | per-task frequency | vault (existing behavior) | ordinary `scheduled-task` entries in `~/obsidian/autonomy/` chained via `depends_on` |

The KG chain (data-pipeline → conversation-relation-linking → entity-resolution-sweep) does NOT get a dedicated source. It's three `scheduled-task` entries (#24, #51, #48) with `frequency: every-15min` and `depends_on` pointing up the chain. The scheduled-task source fires them in order; queue dedup on `scheduled-task:{id}` handles bursts. This keeps a single code path and makes the chain visible/editable in the Autonomy UI.

## Files to create

- [workers/__init__.py](workers/__init__.py)
- [workers/queue.py](workers/queue.py) — SQLite CRUD, atomic claim via `UPDATE ... RETURNING`, dedup on `UNIQUE(dedup_key)`
- [workers/pool.py](workers/pool.py) — asyncio TaskGroup, N-slot semaphore, SDK invocation lifted from `autonomy.py::_run_realtime_step`
- [workers/sources/scheduled_task.py](workers/sources/scheduled_task.py) — scans autonomy task dir, evaluates `due-ness` per file
- [workers/sources/autoresearch.py](workers/sources/autoresearch.py)
- [workers/sources/gap_fill.py](workers/sources/gap_fill.py)
- [workers/sources/session_distill.py](workers/sources/session_distill.py)
- [workers/sources/bench_mine.py](workers/sources/bench_mine.py)
- [workers/sources/domain_research.py](workers/sources/domain_research.py)
- [app/routers/workers.py](app/routers/workers.py) — `GET /api/workers/status`, `/queue`, `/runs`, `POST /api/workers/{enable,enqueue,pause}`
- `~/obsidian/pending-research/README.md` — staging semantics
- `~/obsidian/lloyd/research-queue.md` — user-curated backlog for `domain-research`

## Files to modify

- [server.py](server.py) — register workers router; `app.on_event("startup")` launches worker pool coroutine
- [autonomy.py](autonomy.py):
  - `autonomy_tick()` loops over source `enqueue_if_due()` — no longer executes tasks directly
  - Remove `_ticker_running` serialization; sources handle their own dedup
  - Remove `_realtime_pipeline_running`, `_rerun_pending`, `run_vault_change_pipeline()`, `_run_realtime_step()` (logic moves into `workers/pool.py` + `workers/sources/kg_pipeline.py`)
  - `get_status()` returns `queue_depth`, `worker_slots_in_use`, `source_health`
- [app/routers/autonomy.py](app/routers/autonomy.py) — remove `POST /api/autonomy/vault-change`; keep `GET /api/autonomy/tasks`, `/runs`, `/task-write` (still useful for reading task files)
- [app/routers/__init__.py](app/routers/__init__.py) — export workers router
- [config.yaml](config.yaml) — add `workers:` block (below); autonomy block `tick_seconds` stays
- [agent_mcp/autoresearch.py](agent_mcp/autoresearch.py) — `autoresearch_round` tool enqueues instead of running in a thread; `autoresearch_status` reads from queue/runs tables
- `agent-services/scripts/qmd-watcher.sh` — remove the post-embed curl to `/api/autonomy/vault-change`; keep `qmd update && qmd embed`
- [web/src/App.tsx](web/src/App.tsx) + new WorkersPage in [web/src/components/pages/](web/src/components/pages/); update existing AutonomyPage to read runs from the new endpoint

## Schemas (workers.db)

SQLite WAL mode, path `~/lloyd/workers.db`, not committed.

```sql
CREATE TABLE queue (
  id              INTEGER PRIMARY KEY,
  source          TEXT NOT NULL,
  kind            TEXT NOT NULL,
  priority        INTEGER DEFAULT 50,   -- 0 = highest
  payload_json    TEXT NOT NULL,
  dedup_key       TEXT UNIQUE,
  state           TEXT DEFAULT 'queued',-- queued|claimed|running|completed|failed|poisoned
  attempts        INTEGER DEFAULT 0,
  enqueued_at     TEXT NOT NULL,
  claimed_at      TEXT,
  claimed_by      TEXT,
  completed_at    TEXT,
  error           TEXT
);
CREATE INDEX idx_queue_state_prio ON queue(state, priority, enqueued_at);

CREATE TABLE runs (
  run_id           TEXT PRIMARY KEY,
  queue_id         INTEGER REFERENCES queue(id),
  source           TEXT NOT NULL,
  task_id          TEXT,                 -- scheduled-task source sets this; others NULL
  status           TEXT NOT NULL,        -- success|failed|timeout
  started_at       TEXT NOT NULL,
  completed_at     TEXT NOT NULL,
  duration_seconds REAL,
  summary          TEXT,
  artifact_path    TEXT,
  response_json    TEXT
);

CREATE TABLE watermarks (
  source     TEXT,
  key        TEXT,
  value      TEXT,
  updated_at TEXT,
  PRIMARY KEY (source, key)
);
```

Priority defaults: `scheduled-task`=30 (task frontmatter can override), `kg-pipeline`=40, `gap-fill`=50, `autoresearch`=60, `session-distill`=70, `domain-research`=70, `bench-mine`=80. Lower number = sooner.

## Config additions (`config.yaml`)

```yaml
workers:
  enabled: false              # opt-in; flip on per phase
  slots: 8                    # depth = all vLLM slots; vLLM preempts on priority=0
  db_path: ~/lloyd/workers.db
  staging_root: ~/obsidian/pending-research
  max_attempts: 3
  priority_proxy_url: http://127.0.0.1:8097
  worker_priority: 2
  sources:
    scheduled-task:  { enabled: true,  interval_seconds: 60 }
    autoresearch:    { enabled: true,  interval_seconds: 3600, max_inflight: 1 }
    gap-fill:        { enabled: true,  interval_seconds: 300,  max_inflight: 2 }
    session-distill: { enabled: true,  interval_seconds: 1800, max_inflight: 1 }
    bench-mine:      { enabled: false, interval_seconds: 7200, max_inflight: 1 }  # Phase 4
    domain-research: { enabled: true,  interval_seconds: 600,  max_inflight: 1 }
    kg-pipeline:     { enabled: true,  interval_seconds: 60,   max_inflight: 1, cooldown_seconds: 900 }
```

`max_inflight` prevents a single source from monopolizing all 8 slots (autoresearch rounds take 30-60 min each).

## Phased rollout

Each phase is a reviewable commit. Feature flag via `workers.enabled` and per-source `enabled`.

1. **Phase 1 — Queue + scheduled-task migration.**
   Queue schema, worker pool (N=4), `scheduled-task` source only. Migrate the existing time-based scheduler to enqueue-via-queue. Remove `_ticker_running`. **Behavior parity with today** for all existing autonomy tasks. Confirms queue mechanics without new work types. ~500 LOC.
2. **Phase 2 — KG pipeline migration.**
   Move `kg-pipeline` to the queue. Remove `/api/autonomy/vault-change` endpoint, `_realtime_pipeline_running`, `_rerun_pending`, the post-embed curl in qmd-watcher. One less lock, one less endpoint.
3. **Phase 3 — Autoresearch.**
   Turn on `autoresearch` source (hourly enqueue). Extend `agent_mcp/autoresearch.py` to use queue. Inspect ledger.jsonl growth and promotion behavior.
4. **Phase 4 — Knowledge-acquisition sources + UI.**
   `gap-fill`, `session-distill`, `domain-research` (then `bench-mine` once the others are stable). `pending-research/` staging convention + `/workers` web UI + daily digest. **Gate:** do not enable these sources at scale until the review UI ships.

## Biggest risk + mitigation

**`pending-research/` clutter becomes unreviewable.** Eight workers running all day will produce hundreds of notes per week. If promotion to canonical is purely manual, the pile grows forever and users stop trusting that the pool is doing anything.

Mitigation:
1. **Structured staging:** every artifact under `pending-research/{source}/{yyyy-mm-dd}/` with YAML frontmatter `{confidence, review_status, rationale, source_refs}`. No free-form directories.
2. **Auto-promote on high confidence:** autoresearch winners that pass the existing promotion gate go straight to canonical — same as today. Gap-fill resolutions with ≥2 verified sources also auto-promote.
3. **`/workers` review surface:** 10 oldest pending items + Discord-notified daily digest. Phase 4 ships UI before scaling knowledge-acquisition sources.

## Verification

1. **Phase 1 parity** — after migration, confirm each existing autonomy task runs on its prior schedule (Task #25 hourly memory-capture, Task #59 nightly knowledge-health-report, reflection chain 38→39a→39→40→47→48). Compare last 24h of run records to pre-migration baseline.
2. **Queue mechanics** — `POST /api/workers/enqueue` a dummy `kind=noop` item; observe claim/complete; inspect via `GET /api/workers/runs`.
3. **Priority proxy path** — start a worker round, tail priority-proxy logs at 8097, confirm `priority=2` tags.
4. **User-session non-interference** — with 8 workers saturating the pool, submit a real chat turn; verify TTFT/throughput match baseline (measure via `usage_store.py`).
5. **KG latency** — touch a vault file, confirm a `kg-pipeline:chain` item enqueues within 60s and the 4-step chain completes.
6. **Autoresearch end-to-end** — enqueue one round via MCP tool; confirm ledger entry at `_pipeline/research/ledger.jsonl` and round summary at `_pipeline/research/rounds/<id>.md`.
7. **Gap-fill e2e** — synthetically insert a `label=gap` fact via `fact_add`; within 5 minutes, observe a resolution note at `pending-research/gaps/<date>/<fact_id>.md`.
8. **Restart resilience** — kill `lloyd-mc:lloyd-backend` mid-run; restart; verify `state=queued` items resume, `state=running` items requeue with `attempts+1`, `state=completed` items remain.

## Critical files for implementation

- [autonomy.py](autonomy.py) — tick becomes enqueue-only; remove old locks & vault-change path
- [server.py](server.py) — register workers router + startup hook
- [app/routers/autonomy.py](app/routers/autonomy.py) — remove vault-change endpoint
- [agent-services/scripts/qmd-watcher.sh](agent-services/scripts/qmd-watcher.sh) — remove post-embed curl
- [config.yaml](config.yaml) — add `workers:` block
- [workers/pool.py](workers/pool.py) (new)
- [workers/queue.py](workers/queue.py) (new)
- [workers/sources/scheduled_task.py](workers/sources/scheduled_task.py) (new)
- [workers/sources/kg_pipeline.py](workers/sources/kg_pipeline.py) (new)
- [app/routers/workers.py](app/routers/workers.py) (new)
