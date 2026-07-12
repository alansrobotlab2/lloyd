---
segment: architecture
type: architecture
tags: [agents,architecture,lloyd,pipeline,wiggam]

---

# Staged Pipeline Architecture

> Single-session staged execution driven by the wiggam hook. Agent runs each stage,emits a signal,hook injects next stage and continues.

## Status: Working (2026-03-28)

Implemented and tested. Password strength library completed all 4 stages (plan → implement → test → review,62 tests passing) and FizzBuzz completed 2 stages (implement → review,17 tests passing) — both on local 122B model.

## Core Concept

**One session,multiple stages,full context continuity.**

Instead of spawning separate agents per stage,a single session progresses through ordered stages. At each stage boundary:

1. Agent emits `SIGNAL:STAGE_COMPLETE` (or `SIGNAL:TASK_COMPLETE` for final stage)
2. Gateway's `extractSignals()` detects the signal in the agent's last output
3. `agent_end` hook fires → `pipeline-hooks.ts` looks up the pipeline state
4. If more stages remain: loads next stage's `.md` content → returns `{ continue: true,prompt }` 
5. Gateway's **wiggam loop** re-prompts the agent in the same session
6. Agent sees full conversation history + new stage instructions

No context loss. No file re-reads. No orchestrator. The review stage was present for planning and implementation — it has full context.

## Key Components

### Gateway (lloyd-main branch)

| Commit | What |
|--------|------|
| `0541fb0` | `context` plugin hook (ephemeral message injection) + enriched `agent_end` (signals[],toolNames[],lastAssistantText) |
| `b9490ee` | Wiggam hook — `agent_end` upgraded from void to modifying. Returns `{ continue: true,prompt }`. `runEmbeddedPiAgent` loops up to 10 continuations. |
| `dff5d6d` | Added `STAGE_COMPLETE` to `extractSignals()` |

### Mission Control Extension

**`pipeline-hooks.ts`** (~620 lines) — the brain:
- `context` hook: ephemeral subliminal injection (stage + agent subliminals)
- `llm_output` hook: passive debug logging
- `agent_end` hook: **all lifecycle management** — signal detection,stage advancement,continuation prompts,completion/failure handling
- `POST /api/mc/pipeline-init` — register a session for pipeline tracking
- `GET /api/mc/pipeline-status` — inspect active pipelines

### Stage Definitions

Files at `~/obsidian/agents/worker/stages/{name}.md`:

| Stage | Signal | Purpose |
|-------|--------|---------|
| `plan` | STAGE_COMPLETE | Analyze requirements,survey codebase,produce plan |
| `implement` | STAGE_COMPLETE | Execute the plan — write code,make changes |
| `test` | STAGE_COMPLETE | Write tests,run them,fix failures |
| `review` | TASK_COMPLETE | Adversarial quality audit — final gate |
| `research` | STAGE_COMPLETE | Fetch,extract,synthesize from sources |
| `audit` | STAGE_COMPLETE | Security and ops review |

Each file has YAML frontmatter (`name`,`default_model`,`signal`) and body content used as the continuation prompt.

## Dispatch Protocol

### Step 1: Spawn the worker
```
sessions_spawn({
  agentId: "worker",
  runtime: "subagent",
  mode: "run",
  model: "local-llm-120b/Qwen3.5-122B-A10B",
  task: "<prompt with stage instructions and SIGNAL: markers>"
})
→ returns childSessionKey
```

### Step 2: Register the pipeline
```
POST /api/mc/pipeline-init
{
  "sessionKey": "<childSessionKey>",
  "taskId": <number>,
  "stages": ["plan","implement","test","review"]
}
```

Stage names must match filenames in `agents/worker/stages/`. Order matters.

### Automatic from here
- Stage transitions: `agent_end` detects signal → loads next stage `.md` → returns continuation
- Notifications: toast per stage transition,chat announce on completion/failure
- Finalization: `completeTask()` + `completeActiveRun()` on TASK_COMPLETE

## Common Pipeline Patterns

| Pattern | Stages | When to use |
|---------|--------|-------------|
| Surgical fix | `["implement"]` | One-file fix,no planning needed |
| Standard | `["implement","review"]` | Most tasks |
| Full | `["plan","implement","review"]` | Complex changes |
| With tests | `["plan","implement","test","review"]` | Library/tool creation |
| Research-first | `["research","plan","implement","review"]` | Unknown domain |

## Agent_end Decision Tree

```
agent_end fires for a pipeline session:
├── TASK_COMPLETE signal?
│   → completeTask(),toast + chat announce,cleanup. Done.
├── BLOCKED signal?
│   → handleBlocked(),chat announce,cleanup. Done.
├── STAGE_COMPLETE signal?
│   → advance currentStageIndex
│   ├── more stages? → load next .md → toast → return { continue: true,prompt }
│   └── last stage? → completeTask(). Done.
├── error or failure?
│   → handleTaskFailed(),chat announce,cleanup. Done.
└── no signal (agent just stopped)?
    → return { continue: true,prompt: "keep going..." } (safety net)
```

Non-pipeline tasks (registered with empty stages) finalize on success without continuation.

## Notification Routing

| Event | Delivery | Content |
|-------|----------|---------|
|