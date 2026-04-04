# Claude Code Dream Feature: Complete Analysis & Implementation Guide

> Based on a thorough read of the Claude Code source in `~/Projects`.  
> "Dream" is Claude Code's **background memory consolidation system** — an autonomous subagent that periodically synthesizes session history into durable, well-organized memory files between turns.

---

## Table of Contents

1. [Conceptual Overview](#1-conceptual-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Complete File Manifest](#3-complete-file-manifest)
4. [Data Flow](#4-data-flow)
5. [Key Types & Function Signatures](#5-key-types--function-signatures)
6. [Integration Points](#6-integration-points)
7. [Configuration & Feature Flags](#7-configuration--feature-flags)
8. [UI Surface](#8-ui-surface)
9. [Consolidation Prompt](#9-consolidation-prompt)
10. [Analytics Events](#10-analytics-events)
11. [Error Handling & Resilience](#11-error-handling--resilience)
12. [Task Lifecycle](#12-task-lifecycle)
13. [Disabled Conditions](#13-disabled-conditions)
14. [Manual Trigger (KAIROS Mode)](#14-manual-trigger-kairos-mode)
15. [Settings Schema](#15-settings-schema)
16. [Key Implementation Decisions](#16-key-implementation-decisions)
17. [Implementation Guide for Hermes](#17-implementation-guide-for-hermes)

---

## 1. Conceptual Overview

Dream is a **background memory consolidation feature** — an autonomous, asynchronous subagent that periodically processes and synthesizes memories while the main session is idle or between turns. It performs a reflective pass over accumulated session transcripts and memory files, consolidating what the AI has learned into durable, well-organized memory structures for future sessions.

**Key properties:**
- Runs automatically at turn-end when gates pass (time + session count)
- Runs as a fully independent forked subagent (own context, own tools)
- Does not block the main session (side-effect, not awaited)
- Can be manually triggered via `/dream` in KAIROS mode
- Gated behind GrowthBook feature flag `tengu_onyx_plover`
- User-controllable via `autoDreamEnabled` setting

---

## 2. Architecture Overview

Dream consists of three main subsystems:

### A. Core Auto-Dream Engine (`src/services/autoDream/`)
Manages background execution logic with time/session gating, lock acquisition, and forked agent launch.

### B. Dream Task Management (`src/tasks/DreamTask/`)
UI-facing task registry entry that surfaces the dream process in the background tasks dialog.

### C. UI Components (`src/components/tasks/`)
Visual representations: background task list item, detail view dialog, and memory file selector integration.

---

## 3. Complete File Manifest

### Core Logic
| File | Purpose |
|------|---------|
| `src/services/autoDream/autoDream.ts` | Main execution engine, gates, lock, fork |
| `src/services/autoDream/config.ts` | Feature flag & setting gating |
| `src/services/autoDream/consolidationLock.ts` | Concurrency control via lock file |
| `src/services/autoDream/consolidationPrompt.ts` | Prompt generation (4-phase) |

### Task Management
| File | Purpose |
|------|---------|
| `src/tasks/DreamTask/DreamTask.ts` | Task state, lifecycle functions, kill handler |

### UI Components
| File | Purpose |
|------|---------|
| `src/components/tasks/BackgroundTask.tsx` | Dream task in list view |
| `src/components/tasks/BackgroundTasksDialog.tsx` | Dialog container |
| `src/components/tasks/DreamDetailDialog.tsx` | Detail view with live progress |
| `src/components/memory/MemoryFileSelector.tsx` | Memory UI toggle/status |

### Supporting Systems
| File | Purpose |
|------|---------|
| `src/memdir/memdir.ts` | Memory directory management |
| `src/memdir/paths.ts` | Memory path resolution |
| `src/Task.ts` | Task type definitions |
| `src/query/stopHooks.ts` | Turn-end hook integration |
| `src/skills/bundled/index.ts` | Skill registration |
| `src/utils/attachments.ts` | Daily log flushing |
| `src/utils/cronTasks.ts` | Cron task config |
| `src/utils/settings/types.ts` | Settings schema |
| `src/tasks/pillLabel.ts` | Status text rendering |

---

## 4. Data Flow

```
Turn End (stopHooks.ts)
└─ executeAutoDream() called
   │
   └─ runAutoDream() in closure (autoDream.ts)
      │
      ├─ Gate 1: isGateOpen()
      │  ├─ !getKairosActive()        (skip if assistant mode)
      │  ├─ !getIsRemoteMode()        (skip if remote)
      │  ├─ isAutoMemoryEnabled()     (require memory enabled)
      │  └─ isAutoDreamEnabled()      (check GB flag + settings)
      │
      ├─ Gate 2: Time Gate
      │  └─ readLastConsolidatedAt() < minHours
      │     └─ Lock file mtime check (.consolidate-lock)
      │
      ├─ Gate 3: Scan Throttle
      │  └─ 10min throttle to prevent spam when time passes
      │
      ├─ Gate 4: Session Gate
      │  ├─ listSessionsTouchedSince(lastAt)
      │  └─ Need >= minSessions sessions modified (excl. current)
      │
      ├─ Gate 5: Lock Acquisition
      │  └─ tryAcquireConsolidationLock()
      │     ├─ Check existing lock mtime and PID
      │     ├─ If live process and mtime recent: bail
      │     └─ Write PID, verify write succeeded
      │
      ├─ Register Dream Task (UI)
      │  └─ registerDreamTask(setAppState, {...})
      │     ├─ status='running', phase='starting'
      │     └─ id='dream-{8-char-random}'
      │
      ├─ Fork Subagent
      │  └─ runForkedAgent({
      │     ├─ promptMessages: [buildConsolidationPrompt()]
      │     ├─ canUseTool: createAutoMemCanUseTool(memoryRoot)
      │     ├─ querySource: 'auto_dream'
      │     ├─ skipTranscript: true
      │     └─ onMessage: makeDreamProgressWatcher()
      │
      ├─ Monitor Progress
      │  └─ makeDreamProgressWatcher() on each message
      │     ├─ Extract text blocks
      │     ├─ Count tool_use blocks
      │     ├─ Track file_path from FILE_EDIT_TOOL / FILE_WRITE_TOOL
      │     ├─ Flip phase to 'updating' when files touched
      │     └─ Call addDreamTurn() to update UI
      │
      ├─ Completion
      │  ├─ Success → completeDreamTask()
      │  │  ├─ Set status='completed', notified=true
      │  │  ├─ Append completion message to main session
      │  │  └─ Log 'tengu_auto_dream_completed'
      │  │
      │  └─ Failure → failDreamTask() OR user kill
      │     ├─ Set status='failed'|'killed'
      │     └─ rollbackConsolidationLock(priorMtime)
      │        ├─ If priorMtime=0: delete lock file
      │        └─ Otherwise: rewind mtime to pre-acquire
      │
      └─ User kills task
         └─ DreamTask.kill(taskId, setAppState)
            ├─ Abort the AbortController
            ├─ Set status='killed'
            └─ rollbackConsolidationLock(priorMtime)
```

---

## 5. Key Types & Function Signatures

### `autoDream.ts`

```typescript
function initAutoDream(): void
// Called once at startup. Initializes closure with lastSessionScanAt.

async function executeAutoDream(
  context: REPLHookContext,
  appendSystemMessage?: AppendSystemMessageFn,
): Promise<void>
// Entry point from stopHooks. Executes dream if all gates pass.

function getConfig(): AutoDreamConfig
// Returns {minHours, minSessions} from GrowthBook with fallbacks.

function isGateOpen(): boolean
// Returns false if: KAIROS active, remote mode, memory disabled, or feature disabled.

function makeDreamProgressWatcher(
  taskId: string,
  setAppState: SetAppState,
): (msg: Message) => void
// Returns callback tracking turns and touched file paths from assistant messages.
```

### `consolidationLock.ts`

```typescript
async function readLastConsolidatedAt(): Promise<number>
// Returns lock file mtime (0 if absent). One stat() cost.

async function tryAcquireConsolidationLock(): Promise<number | null>
// Acquire lock: write PID, verify. Return prior mtime for rollback, or null if blocked.

async function rollbackConsolidationLock(priorMtime: number): Promise<void>
// Rewind lock mtime to priorMtime. If priorMtime=0, delete lock file.

async function listSessionsTouchedSince(sinceMs: number): Promise<string[]>
// Return session IDs with mtime > sinceMs from project transcript dir.

async function recordConsolidation(): Promise<void>
// Manual /dream stamp. Write lock file with current PID.
```

### `DreamTask.ts`

```typescript
type DreamPhase = 'starting' | 'updating'

type DreamTurn = {
  text: string
  toolUseCount: number
}

type DreamTaskState = TaskStateBase & {
  type: 'dream'
  phase: DreamPhase
  sessionsReviewing: number
  filesTouched: string[]       // Files edited/written during dream
  turns: DreamTurn[]           // Recent assistant responses
  abortController?: AbortController
  priorMtime: number           // For lock rollback on kill
}

function registerDreamTask(
  setAppState: SetAppState,
  opts: {
    sessionsReviewing: number
    priorMtime: number
    abortController: AbortController
  }
): string
// Register new dream task, return task ID.

function addDreamTurn(
  taskId: string,
  turn: DreamTurn,
  touchedPaths: string[],
  setAppState: SetAppState,
): void
// Append turn to task, flip phase to 'updating' if files touched.

function completeDreamTask(taskId: string, setAppState: SetAppState): void
function failDreamTask(taskId: string, setAppState: SetAppState): void

const DreamTask: Task = {
  async kill(taskId, setAppState) { ... }
}
```

---

## 6. Integration Points

### Turn-End Hooks (`query/stopHooks.ts`)
- Called at end of every turn (after assistant response)
- Entry: `executeAutoDream(context, appendSystemMessage)`
- Non-blocking: fired as side effect, doesn't block response

### Task Registry (`Task.ts`)
- Task type: `'dream'` (shortcode: `'d'`)
- Implements `Task` interface with `kill()` method
- Lifecycle: registered → running → completed | failed | killed

### UI (`components/tasks/`)
- Appears in Shift+Down background tasks dialog
- Clickable to open detail dialog with live progress
- Kill with `x`, close with Esc/Space/Enter

### Memory System (`memdir/`)
- Reads from auto-memory directory
- Reads project transcript JSONL files
- Writes updated memory files and MEMORY.md index
- Bash restricted to read-only commands

### Forked Agents (`utils/forkedAgent.ts`)
- Runs as separate forked subagent with own context window
- `createAutoMemCanUseTool()` controls file access
- `skipTranscript: true` — dream not recorded in main session
- Cache-enabled: reads cache from main context for efficiency

---

## 7. Configuration & Feature Flags

### GrowthBook Feature Flag: `tengu_onyx_plover`

```typescript
interface TenguOnyxPlover {
  enabled?: boolean       // Master kill switch
  minHours?: number       // Time gate (default: 24)
  minSessions?: number    // Session gate (default: 5)
}
```

Resolution order for `enabled`:
1. `settings.autoDreamEnabled` — user setting (takes precedence if defined)
2. GrowthBook `tengu_onyx_plover.enabled` — server-side default
3. Default: enabled

### Time and Session Gates

```typescript
const DEFAULTS: AutoDreamConfig = {
  minHours: 24,     // Must wait 24h since last consolidation
  minSessions: 5,   // At least 5 sessions touched memory since then
}

const SESSION_SCAN_INTERVAL_MS = 10 * 60 * 1000  // 10 min scan throttle
```

### Lock File

```
Location:  {getAutoMemPath()}/.consolidate-lock
Content:   PID of lock holder
Mtime:     timestamp of last consolidation
Stale after: 1 hour (HOLDER_STALE_MS)
```

---

## 8. UI Surface

### Status Bar / Memory Selector

```
Auto-dream: on · last consolidated 3h ago
Auto-dream: off · /dream to run
```

Emits `tengu_auto_dream_toggled` when user toggles.

### Background Tasks Dialog (Shift+Down)

Dream tasks appear in the selectable list:
- Navigate: ↑/↓
- View detail: Enter or Space
- Kill (while running): `x`
- Close: Esc

### Dream Detail Dialog

Live progress view showing:
- Status: `running` | `completed` | `failed` | `killed`
- Phase: `starting` → `updating` (once first file touched)
- Sessions reviewed: e.g., "reviewing 5 sessions"
- Files touched: e.g., "touched 3 files"
- Recent turns: up to 6 (earlier turns collapse to count)
- Elapsed time: e.g., "3m 21s"
- Shortcuts: ← back, Esc/Enter/Space close, x stop

---

## 9. Consolidation Prompt

The 4-phase prompt sent to the forked subagent:

```
# Dream: Memory Consolidation

You are performing a dream — a reflective pass over your memory files.
Synthesize what you've learned recently into durable, well-organized memories
so that future sessions can orient quickly.

Memory directory: `{memoryRoot}`
Session transcripts: `{transcriptDir}` (large JSONL files — grep narrowly,
don't read whole files)

## Phase 1 — Orient
- `ls` the memory directory to see what already exists
- Read `MEMORY.md` to understand the current index
- Skim existing topic files so you improve them rather than creating duplicates

## Phase 2 — Gather recent signal
Look for new information worth persisting. Sources in priority order:
1. Daily logs (`logs/YYYY/MM/YYYY-MM-DD.md`) if present
2. Existing memories that drifted (facts that contradict current codebase)
3. Transcript search (grep narrowly, don't read whole files)

## Phase 3 — Consolidate
For each thing worth remembering, write or update a memory file.
- Merge new signal into existing topic files (avoid near-duplicates)
- Convert relative dates to absolute dates
- Delete contradicted facts

## Phase 4 — Prune and index
Update `MEMORY.md` so it stays under 200 lines AND under ~25KB.
Each entry: `- [Title](file.md) — one-line hook`
- Remove stale/wrong/superseded pointers
- Shorten verbose entries
- Add pointers to newly important memories
- Resolve contradictions

Return a brief summary of what you consolidated, updated, or pruned.
```

---

## 10. Analytics Events

```typescript
'tengu_auto_dream_fired' — {
  hours_since: number,
  sessions_since: number,
}

'tengu_auto_dream_completed' — {
  cache_read: number,
  cache_created: number,
  output: number,
  sessions_reviewed: number,
}

'tengu_auto_dream_failed' — {}

'tengu_auto_dream_toggled' — {
  enabled: boolean,
}
```

---

## 11. Error Handling & Resilience

### Lock-Based Concurrency
- Only one dream runs at a time (first to acquire wins)
- PID reuse guard: lock stale after 1 hour even if PID exists
- Lock rollback on failure: mtime rewound so time-gate passes again on retry
- Lock file location: inside auto-memory dir (git-root scoped)

### Graceful Degradation
- Gate failures are silent no-ops (don't surface to user)
- Scan throttle prevents spam on repeated misses
- Lock acquire failures logged but don't block
- User can kill at any time; lock rolls back for retry
- Cache used aggressively to reduce per-turn API cost

### Memory Directory Availability
- Assumption: memory dir exists (created by `ensureMemoryDirExists`)
- Bash restricted to: `ls`, `find`, `grep`, `cat`, `stat`, `wc`, `head`, `tail`
- All file writes go through `createAutoMemCanUseTool()` permission layer

---

## 12. Task Lifecycle

```
registerDreamTask() → status='running', phase='starting'
      │
      ├─ addDreamTurn() updates turns/filesTouched (phase → 'updating' on first file)
      │
      ├─ User presses 'x' → kill()
      │  ├─ abortController.abort()
      │  ├─ status='killed'
      │  └─ rollbackConsolidationLock()
      │
      ├─ Fork succeeds → completeDreamTask()
      │  ├─ status='completed', notified=true
      │  └─ appendSystemMessage("Improved N files")
      │
      └─ Fork throws → failDreamTask()
         ├─ status='failed', notified=true
         └─ rollbackConsolidationLock()

All terminal states: notified=true → UI evicts after timeout
```

---

## 13. Disabled Conditions

Dream will **not** fire if any of the following are true:

1. `isGateOpen()` returns false:
   - KAIROS mode active (uses `/dream` skill instead)
   - Remote mode active
   - Auto-memory disabled
   - `isAutoDreamEnabled()` returns false

2. Time gate: < `minHours` (default 24h) since last consolidation

3. Scan throttle: < 10 min since last scan attempt

4. Session gate: < `minSessions` (default 5) sessions touched since last consolidation

5. Lock acquire: another process holds a live, recent lock

6. Force flag is false (testing override, disabled in production)

---

## 14. Manual Trigger (KAIROS Mode)

In KAIROS feature mode, `/dream` skill is registered (`skills/bundled/dream.js`, feature-gated):
- Runs same consolidation prompt as auto-dream
- No time/session gates (manual override)
- Calls `recordConsolidation()` to stamp the lock file after completion
- Still respects bash read-only restrictions

---

## 15. Settings Schema

```typescript
// src/utils/settings/types.ts
autoDreamEnabled?: boolean
// "Enable background memory consolidation (auto-dream).
//  When set, overrides the server-side default."
```

User opt-out via `settings.json`:
```json
{
  "autoDreamEnabled": false
}
```

---

## 16. Key Implementation Decisions

### Why 4-phase prompt without phase detection?
The consolidation prompt has explicit Phase 1–4 sections but the harness doesn't parse or enforce them. The agent follows the phases autonomously. UI state (`starting` → `updating`) is derived solely from whether files have been touched. This keeps the agent autonomous — the harness watches, it doesn't orchestrate.

### Why turn collapse in UI?
Dream agents can be verbose. The UI keeps only the most recent 6 turns visible; tool use blocks collapse to counts. Users care about work done, not individual tool calls.

### Why `skipTranscript: true`?
Dream is background bookkeeping, not part of the user's conversation. Skipping the transcript prevents dream runs from bloating the main session JSONL and avoids confusing the context of future sessions.

### Why a forked agent?
The dream needs independent tool access (editing memory files, reading transcripts) and must not interfere with the main session's context window. A forked agent can be killed independently, uses its own context, and can share prompt cache with the main session for efficiency.

### Why a lock file rather than in-memory state?
The lock file survives process restarts and is visible to concurrent processes (e.g., multiple Claude Code windows on the same project). The mtime doubles as the "last consolidated at" timestamp, avoiding a separate file. PID in content enables stale lock detection.

---

## 17. Implementation Guide for Hermes

This section describes how to implement a dream-equivalent system in the Hermes project.

### Core Requirements

To replicate dream in Hermes, you need:

1. **Memory directory** with a `MEMORY.md` index and per-topic `.md` files
2. **Session transcripts** stored as JSONL (or any log format greppable by a subagent)
3. **A forked/subagent mechanism** that can run independently of the main session
4. **Gate logic**: time gate + session count gate
5. **Lock file**: to prevent concurrent consolidation runs
6. **Turn-end hook**: where the gate logic is evaluated and dream is potentially triggered

### Minimal Implementation Steps

#### Step 1: Lock File Module

```typescript
// services/dream/lock.ts
const LOCK_FILE = path.join(getMemoryRoot(), '.consolidate-lock')
const STALE_MS = 60 * 60 * 1000  // 1 hour

async function readLastConsolidatedAt(): Promise<number> {
  try {
    const stat = await fs.stat(LOCK_FILE)
    return stat.mtimeMs
  } catch {
    return 0
  }
}

async function tryAcquireLock(): Promise<number | null> {
  const priorMtime = await readLastConsolidatedAt()
  // Check for live holder...
  await fs.writeFile(LOCK_FILE, String(process.pid))
  // Verify we won...
  return priorMtime
}

async function rollbackLock(priorMtime: number): Promise<void> {
  if (priorMtime === 0) {
    await fs.unlink(LOCK_FILE).catch(() => {})
  } else {
    await fs.utimes(LOCK_FILE, new Date(priorMtime), new Date(priorMtime))
  }
}
```

#### Step 2: Gate Logic

```typescript
// services/dream/gates.ts
const MIN_HOURS = 24
const MIN_SESSIONS = 5
const SCAN_THROTTLE_MS = 10 * 60 * 1000

let lastScanAt = 0

async function shouldRunDream(): Promise<{ run: boolean; reason?: string }> {
  if (!isAutoMemoryEnabled()) return { run: false, reason: 'memory disabled' }
  if (!isAutoDreamEnabled()) return { run: false, reason: 'dream disabled' }

  const now = Date.now()
  if (now - lastScanAt < SCAN_THROTTLE_MS) return { run: false, reason: 'throttled' }
  lastScanAt = now

  const lastAt = await readLastConsolidatedAt()
  const hoursSince = (now - lastAt) / 3_600_000
  if (hoursSince < MIN_HOURS) return { run: false, reason: 'too recent' }

  const sessions = await listSessionsTouchedSince(lastAt)
  if (sessions.length < MIN_SESSIONS) return { run: false, reason: 'too few sessions' }

  return { run: true }
}
```

#### Step 3: Consolidation Prompt

Use the 4-phase structure from Section 9, substituting your actual memory root and transcript directory paths. Key constraints:
- Bash tools should be **read-only** during Phase 1 & 2 (`ls`, `find`, `grep`, `cat`, `stat`, `wc`, `head`, `tail`)
- Write tools allowed during Phase 3 & 4 (file edit/write restricted to memory root)
- Instruct agent to `grep narrowly` on transcripts, never read whole JSONL files

#### Step 4: Turn-End Hook

```typescript
// hooks/onTurnEnd.ts
export async function onTurnEnd(context: TurnContext): Promise<void> {
  // ... other hooks ...
  
  // Fire dream check (non-blocking)
  executeDream(context).catch(err => {
    console.error('dream error:', err)
  })
}

async function executeDream(context: TurnContext): Promise<void> {
  const { run } = await shouldRunDream()
  if (!run) return

  const priorMtime = await tryAcquireLock()
  if (priorMtime === null) return

  const abortController = new AbortController()
  
  try {
    await runSubagent({
      prompt: buildConsolidationPrompt(),
      abortSignal: abortController.signal,
      skipTranscript: true,
      onMessage: trackProgress,
    })
    // stamp completed — lock mtime is already set by tryAcquireLock
  } catch (err) {
    await rollbackLock(priorMtime)
    throw err
  }
}
```

#### Step 5: Progress Tracking

The progress watcher should extract, per assistant message:
- Text content (to show user what agent is reasoning about)
- Tool use count (collapsed in UI)
- File paths from any `file_edit` or `file_write` tool calls

```typescript
function makeProgressWatcher(onUpdate: (turn: DreamTurn) => void) {
  return (msg: AssistantMessage) => {
    const text = msg.content
      .filter(b => b.type === 'text')
      .map(b => b.text)
      .join('')
    
    const toolUseCount = msg.content.filter(b => b.type === 'tool_use').length
    
    const touchedPaths = msg.content
      .filter(b => b.type === 'tool_use' && isFileWriteTool(b.name))
      .map(b => b.input.file_path)
      .filter(Boolean)
    
    onUpdate({ text, toolUseCount, touchedPaths })
  }
}
```

### Hermes-Specific Adaptations

Given that Hermes uses:
- `mc_server.py` as the backend (needs restart after changes)
- A memory system at `/home/alansrobotlab/.claude/projects/.../memory/`
- The `MEMORY.md` index format already in use

The dream implementation should:

1. Use the **same memory directory** as the existing auto-memory system
2. Read session transcripts from **wherever Hermes stores conversation history** (check `mc_server.py` for transcript storage location)
3. Integrate the **turn-end trigger** into whatever `mc_server.py` calls after processing a response
4. Restrict the consolidation subagent's bash to **read-only** to protect backend state
5. Scope the write permissions to the memory root only — never let the dream agent write to `mc_server.py` or other source files

### What Dream Is NOT

Dream is not:
- A chat summary (it writes to memory files, not into the conversation)
- A full re-read of transcripts (Phase 2 instructs the agent to grep narrowly)
- Synchronous with user turns (it runs after the turn ends, non-blocking)
- A one-shot operation (gated by time + session count to amortize cost)

---

*Generated 2026-04-02 from analysis of the Claude Code source in `~/Projects`.*
