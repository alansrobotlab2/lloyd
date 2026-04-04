# Pipeline Workflow Migration: .openclaw → .hermes

## What "Pipeline" Means Here

In `.openclaw`, a **pipeline** is a multi-stage agentic run: Lloyd dispatches work by calling `pipeline_dispatch` with a task and an ordered list of stage names (e.g., `plan → implement → review`). Each stage spawns a fresh worker session with a stage-specific subliminal injected, and the pipeline advances through stages by detecting `SIGNAL:STAGE_COMPLETE` in the worker's output.

The full system lives across three files in `.openclaw/extensions/mission-control/`:
- `pipeline-hooks.ts` — core: stage state machine, context injection, signal parsing, autonomy DB updates
- `autonomy-service.ts` — read/write to autonomy SQLite DB
- `index.ts` — tool registration (`pipeline_dispatch`, `pipeline_status`) and stage file loading from `~/obsidian/agents/worker/stages/`

In `.hermes`, none of this exists yet. The closest analogs are the `autonomy` plugin (task management) and the `subliminal` plugin (per-call context injection). The pipeline layer sits between them.

---

## Current State of Hermes Plugins

| Plugin | Status | OpenClaw Equivalent |
|--------|--------|---------------------|
| `subliminal` | **Done** — vault recall + Operating Contract injection | `mcp-tools` prefill + subliminal frontmatter |
| `autonomy` | **Done** — task CRUD in `~/obsidian/autonomy/` | Autonomy DB (SQLite) |
| `backlog` | **Done** — kanban board tools | Backlog DB (SQLite) |
| `mission-control` | **Done** (thin) — session list/send via CLI | Mission Control REST API + WebSocket |
| `http-tools` | **Done** — search + fetch | `http_search`, `http_fetch` in mcp-tools |
| `next-gen-memory` | **Done** — fact CRUD | Vault + memory writes |
| `thunderbird-tools` | **Done** — email/calendar via MCP bridge | `thunderbird-tools` extension |
| **`pipeline`** | **Missing** | `pipeline-hooks.ts` + `pipeline_dispatch` tool |
| **`worker`** | **Missing** | Worker agent + stages dir |
| **Nightly jobs** | **Missing** | Cron: memory capture, vault maintenance, skills, reflection |
| **Signal parsing** | **Partial** (subliminal Phase 4 planned) | `pipeline-hooks.ts` signal detection |

---

## Pipeline Architecture: OpenClaw

```
Lloyd (main agent)
  │
  ├─ pipeline_dispatch(task, stages=["plan","implement","review"])
  │     │
  │     └─ pipeline-hooks.ts (registered via session_spawn hook)
  │           │
  │           ├─ Loads stage definitions from ~/obsidian/agents/worker/stages/*.md
  │           ├─ Loads matching skills from ~/obsidian/skills/
  │           ├─ Stores PipelineState in activePipelines Map (in-memory)
  │           └─ Spawns worker session with stage[0] subliminal injected
  │
  │     Worker session (stage 0)
  │       └─ pre_llm_call / context hook: injects stage subliminal each turn
  │       └─ Emits SIGNAL:STAGE_COMPLETE when done
  │
  ├─ pipeline-hooks.ts detects signal via agent_end hook
  │       ├─ currentStageIndex++
  │       └─ if more stages: spawn new worker session with stage[N] subliminal
  │          else: completeTask() → update autonomy DB, notify requester
  │
  └─ pipeline_status(taskId) → returns current stage + completion state
```

**Stage file format** (`~/obsidian/agents/worker/stages/<name>.md`):
```yaml
---
default_model: opus
signal: SIGNAL:STAGE_COMPLETE
---
<stage-specific instructions, injected as subliminal for this stage>
```

**Model aliases** (defined in `pipeline-hooks.ts`):
```
122b  → local-llm-120b/Qwen3.5-122B-A10B
35b   → local-llm-35b/Qwen3.5-35B-A3B
opus  → anthropic/claude-opus-4-6
sonnet → anthropic/claude-sonnet-4-6
```

**Skill injection**: At dispatch time, `pipeline_dispatch` searches `~/obsidian/skills/` for skills relevant to the task. Matched skills are loaded and appended to each stage's subliminal. A `novel` flag tracks whether any skills matched (used to nudge skill creation at TASK_COMPLETE).

---

## Hermes Hook System (Target)

Hermes has two complementary hook entry points:

1. **`pre_llm_call` plugin hook** (`run_agent.py:6640`) — fires before every LLM API call in a session. Plugins return `{"context": "..."}` to inject ephemeral context. The `subliminal` plugin already uses this.

2. **Gateway hooks** (`gateway/hooks.py`) — async event handlers in `~/.hermes/hooks/<name>/handler.py`. Events: `agent:start`, `agent:step`, `agent:end`, etc. Used for signal parsing and inter-session coordination.

The pipeline plugin needs **both**: `pre_llm_call` for per-call stage injection, and a gateway `agent:end` hook for signal detection and stage advancement.

The worker sessions are started as separate Hermes CLI invocations (same as how `mission-control` uses `subprocess` to call `hermes chat`), but the pipeline needs to control them from a parent process.

---

## Implementation Plan

### Phase 1 — Stage File Convention + Worker Config

**Goal**: Establish where stage definitions live and how the worker agent is configured.

**Stage dir**: `~/obsidian/agents/worker/stages/` (same as OpenClaw — no reason to change).

**Stage file format** (same frontmatter, compatible with OpenClaw):
```yaml
---
default_model: 122b
signal: SIGNAL:STAGE_COMPLETE
---
You are the planning stage. Your job is to...
SIGNAL:STAGE_COMPLETE when the plan is written.
```

**Worker personality**: Add a `worker` personality entry to `config.yaml`:
```yaml
personalities:
  worker:
    system_prompt_file: ~/.hermes/SOUL.md
    reasoning_effort: high
    max_turns: 80
    note: "Used for pipeline stages. Receives stage subliminal via pre_llm_call."
```

**Deliverable**: Stage files exist and can be loaded. Worker can be invoked with `hermes chat --personality worker`.

---

### Phase 2 — Pipeline Plugin (`plugins/pipeline/`)

**Goal**: Implement `pipeline_dispatch` and `pipeline_status` tools as a Hermes plugin.

**File**: `~/.hermes/plugins/pipeline/__init__.py`

**Core data structures**:
```python
# Stored per-pipeline in ~/.hermes/pipeline-runs/<task_id>.json
{
  "task_id": 52,
  "stages": ["plan", "implement", "review"],
  "current_stage_index": 0,
  "status": "running",   # running | complete | blocked
  "requester_session": "...",
  "skills": ["...content..."],
  "novel": true
}
```

**`pipeline_dispatch` tool**:
1. Accept `task: str, stages: list[str], model: str | None`
2. Load stage definitions from `~/obsidian/agents/worker/stages/`
3. Search `~/obsidian/skills/` for matching skills (keyword match against task)
4. Write run JSON to `~/.hermes/pipeline-runs/<task_id>.json`
5. Spawn worker session: `hermes chat --personality worker --session-key pipeline:<task_id>:stage:0`
6. Return task_id and initial status

**`pipeline_status` tool**:
1. Accept `task_id: int`
2. Read run JSON
3. Return current stage, status, and completion info

**Stage content injection**: The pipeline plugin also implements `pre_llm_call`. It checks if the current session key matches `pipeline:*`, reads the run JSON to find the current stage, loads that stage's content + skills, and returns it as context. This replaces the static subliminal for pipeline worker sessions.

**Session key convention**: `pipeline:<task_id>:stage:<n>` — allows `pre_llm_call` and the gateway signal handler to unambiguously identify pipeline sessions.

---

### Phase 3 — Signal Handler (Gateway Hook)

**Goal**: Detect `SIGNAL:STAGE_COMPLETE` in worker output and advance the pipeline.

**File**: `~/.hermes/hooks/pipeline-signals/handler.py`

**Registered for**: `agent:end` event

**Logic**:
```python
SIGNAL_RE = re.compile(r'\bSIGNAL:(STAGE_COMPLETE|TASK_COMPLETE|BLOCKED(?::.+)?)\b')

async def handle(event_type, context):
    session_key = context.get("session_key", "")
    if not session_key.startswith("pipeline:"):
        return

    # Parse session key: pipeline:<task_id>:stage:<n>
    parts = session_key.split(":")
    task_id = int(parts[1])
    
    response = context.get("response", "")
    match = SIGNAL_RE.search(response)
    if not match:
        return  # Stage didn't signal — let it continue next turn
    
    signal = match.group(1)
    run = load_run(task_id)
    
    if signal == "STAGE_COMPLETE":
        run["current_stage_index"] += 1
        if run["current_stage_index"] >= len(run["stages"]):
            # All stages done
            run["status"] = "complete"
            save_run(run)
            mark_autonomy_task_complete(run["task_id"])
            notify_requester(run)
        else:
            # Advance to next stage
            save_run(run)
            spawn_next_stage(run)
    
    elif signal.startswith("BLOCKED"):
        run["status"] = "blocked"
        run["blocked_reason"] = signal[8:]  # after "BLOCKED:"
        save_run(run)
        notify_requester(run)
    
    elif signal == "TASK_COMPLETE":
        run["status"] = "complete"
        save_run(run)
        mark_autonomy_task_complete(run["task_id"])
        notify_requester(run)
```

**`spawn_next_stage`**: calls `hermes chat --personality worker --session-key pipeline:<task_id>:stage:<n>` as a subprocess, same pattern as `mission-control`'s `run_hermes_command`.

**`notify_requester`**: sends a `chat_send` to the requester session (stored in run JSON) with a summary of completion or blockage.

---

### Phase 4 — Autonomy Integration

**Goal**: Wire pipeline completion to the autonomy task system.

The `autonomy` plugin already provides `autonomy_write_task`. The signal handler can call it directly (import the plugin function) or via the tool interface.

**Flow**:
- `pipeline_dispatch` accepts an optional `autonomy_task_id` param
- On TASK_COMPLETE, signal handler calls `autonomy_write_task(task_id, status="done")`
- This closes the loop between the high-level task backlog and the pipeline execution

**Autonomy idler** (future): The OpenClaw system has an "idler" that polls the autonomy DB and calls `POST /api/mc/pipeline-init` to auto-dispatch backlog tasks when Lloyd is idle. In Hermes, this would be a cron trigger or a background hook polling `~/obsidian/autonomy/`.

---

### Phase 5 — Nightly Jobs

The three nightly OpenClaw cron jobs need Hermes equivalents. They currently run as scheduled tasks in `.openclaw/cron/jobs.json` and invoke the `memory` agent via `sessions_spawn`.

| Job | OpenClaw Schedule | OpenClaw Mechanism | Hermes Target |
|-----|-------------------|--------------------|---------------|
| Memory capture | Every 15min | `periodic-memory-capture` → memory agent | `hermes chat --personality memory` on cron |
| Vault maintenance | 2:00 AM | Nightly job → Opus | Pipeline: `vault-maintenance` stages |
| Skills management | 3:00 AM | Nightly job → Opus | Pipeline: `skills-management` stages |
| Nightly reflection | 4:00 AM | Nightly job → Opus | Pipeline: `nightly-reflection` stages |

**Recommended approach**: Use the `schedule` skill to create cron triggers that call `hermes chat` with appropriate prompts. The memory capture job and each nightly job become scheduled remote agents rather than managed cron jobs in a separate framework.

**Memory capture specifics**: The OpenClaw version runs a transcript extraction script (`extract-transcript.py`) that feeds the session JSONL to the memory agent for distillation. The Hermes equivalent needs:
1. A script (or tool) that exports the current session transcript
2. A `hermes chat --personality memory` invocation with the transcript content

---

## File Map: OpenClaw → Hermes

| OpenClaw | Hermes | Notes |
|----------|--------|-------|
| `extensions/mission-control/pipeline-hooks.ts` | `plugins/pipeline/__init__.py` | Core pipeline logic |
| `extensions/mission-control/pipeline-hooks.ts` (signal handler) | `hooks/pipeline-signals/handler.py` | Gateway `agent:end` hook |
| `extensions/mission-control/autonomy-service.ts` | `plugins/autonomy/__init__.py` | Already done |
| `openclaw.json` agents.worker | `config.yaml` personalities.worker | Worker config |
| `~/obsidian/agents/worker/stages/*.md` | Same path | Stage definitions — no change |
| `~/obsidian/skills/` | Same path | Skills — no change |
| `cron/jobs.json` | `schedule` triggers | Nightly jobs |
| `agents/memory/` workspace | `~/.hermes/plugins/memory/` (or existing next-gen-memory) | Memory agent |

---

## What Doesn't Exist Yet in Hermes

1. **`post_llm_call` plugin hook** — needed for Phase 5 skill enforcement (skill check is optional; subliminal instruction may suffice)
2. **Worker agent invocation** — the pipeline plugin spawns `hermes chat --personality worker` as subprocess; verify this works with session key passing before building Phase 2
3. **`agent:end` gateway hook event** — confirm this fires with `session_key` and full response in context before building Phase 3
4. **Pipeline run storage dir** — `~/.hermes/pipeline-runs/` needs to exist (create on first use)
5. **Autonomy idler** — not in Hermes yet; manual dispatch via `pipeline_dispatch` tool works for now

---

## Build Order

1. **Validate worker invocation** — manually run `hermes chat --personality worker --session-key pipeline:test:stage:0` and confirm session key is accessible to plugins
2. **Phase 1** — create stage files and worker personality config
3. **Phase 2** — pipeline plugin with `pipeline_dispatch` and `pipeline_status` tools + `pre_llm_call` stage injection
4. **Phase 3** — gateway signal handler; test end-to-end with a 2-stage pipeline
5. **Phase 4** — wire to autonomy tasks
6. **Phase 5** — port nightly jobs as scheduled triggers

---

## Open Questions

- Does `hermes chat` accept a `--session-key` flag or equivalent? The pipeline needs stable, predictable session keys to correlate hooks with runs. If not, the run JSON can store the session ID returned by the spawn call.
- Does the `agent:end` gateway hook fire for subprocessed sessions (spawned by another plugin), or only for the primary gateway session?
- The `novel` flag (no matching skills found) was used in OpenClaw to nudge skill creation at pipeline completion. Is that behavior wanted in Hermes, or does the subliminal's skill-check rule cover it?
- Should pipeline runs be stored in `~/.hermes/pipeline-runs/` (file-based, like autonomy) or in the autonomy DB itself as a run record?
