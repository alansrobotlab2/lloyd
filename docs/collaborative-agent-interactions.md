# Collaborative Agent Interactions for Agent Orchestrator

**Date:** 2026-03-04
**Type:** Design Document
**Status:** Draft

## Problem

The agent orchestrator plugin (`extensions/agent-orchestrator/`) uses a fire-and-forget model: `cc_orchestrate`/`cc_spawn` return an instance ID immediately, the Agent SDK query runs in the background, and a completion notification arrives when done. There's no mechanism for subagents to ask clarifying questions, request permission for destructive actions, or surface progress — all the collaborative patterns that make Claude Code effective.

## SDK Primitives Available (Currently Unused)

The Claude Agent SDK (`@anthropic-ai/claude-agent-sdk` v0.2.68) already provides:

| Primitive | SDK Interface | Purpose |
|-----------|--------------|---------|
| `canUseTool` callback | `Options.canUseTool` | Intercept every tool call for approval/denial/modification |
| `interrupt()` | `Query.interrupt()` | Pause execution, return control |
| `streamInput()` | `Query.streamInput()` | Inject user messages into a running query |
| `stopTask()` | `Query.stopTask(taskId)` | Abort a specific subagent mid-run |
| `AskUserQuestion` tool | Built-in agent tool | Structured clarification questions |
| Task lifecycle messages | `task_started`, `task_progress`, `task_notification` | Real-time subagent progress |
| `setPermissionMode()` | `Query.setPermissionMode()` | Change permission level mid-execution |
| `setModel()` | `Query.setModel()` | Switch model mid-execution |

## Notification Channel: HTTP Hooks Endpoint

OpenClaw has a built-in HTTP hooks system that provides push notifications into Lloyd's session. This replaces the complex gateway WebSocket + device auth approach currently used for completion notifications.

**Endpoint:** `POST http://127.0.0.1:18789/hooks/agent`

```typescript
await fetch("http://127.0.0.1:18789/hooks/agent", {
  method: "POST",
  headers: {
    "Authorization": "Bearer <hooks-token>",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    message: "[Agent Question] coder wants to write auth.ts. cc_respond(q-1234, allow/deny)",
    sessionKey: "agent:main:main",
  }),
});
```

**Why this is better than gateway WebSocket:**
- 5 lines vs 160 lines of WebSocket + Ed25519 auth code
- Already proven (voice pipeline uses it for ASR transcript injection)
- Push, not poll — message lands in Lloyd's session and **wakes Lloyd up**
- Token from `openclaw.json` → `hooks.token`
- Config: `hooks.allowRequestSessionKey: true` enables routing to `agent:main:main`

**Key implication for collaborative pattern:** When a subagent hits a `canUseTool` gate, the plugin POSTs a question to `/hooks/agent`. Lloyd's session receives it as an incoming message, Lloyd sees the question, asks the user, and calls `cc_respond`. No polling loop needed.

## Architecture: The Promise-Based Gate Pattern

```
Subagent calls tool (e.g. Write)
  → SDK fires canUseTool callback (async, blocks indefinitely)
    → Plugin creates PendingQuestion + Promise
      → POST /hooks/agent pushes question into Lloyd's session
        → Lloyd wakes up, presents question to user
          → User answers
            → Lloyd calls cc_respond(questionId, answer)
              → Promise resolves
                → canUseTool returns allow/deny
                  → SDK resumes tool execution
```

## Data Flow Diagram

```
┌──────────┐    cc_orchestrate     ┌──────────────────┐
│          │   (interactive:true)   │  Agent SDK        │
│  Lloyd   │ ──────────────────────▶│  query()          │
│  (main)  │                        │                   │
│          │    cc_respond           │  ┌─────────────┐ │
│          │ ──────────────────────▶ │  │ Orchestrator │ │
│          │                        │  │  (Sonnet)    │ │
│          │ ◀── POST /hooks ────── │  │      │       │ │
│          │   [Agent Question]     │  │  Task tool   │ │
│          │                        │  │      │       │ │
│          │ ◀── POST /hooks ────── │  │  ┌───▼─────┐ │ │
│          │   [Agent Progress]     │  │  │ coder   │ │ │
│          │                        │  │  │ (Opus)  │ │ │
│          │ ◀── POST /hooks ────── │  │  │  Write ─┼─┼─┼── canUseTool ──▶ PendingQuestion
│          │   [Completion]         │  │  └─────────┘ │ │                        │
└──────────┘                        │  └─────────────┘ │           cc_respond ◀──┘
                                    └──────────────────┘
```

---

## Phase 1: Pending Questions Infrastructure + `cc_respond` Tool

**Foundation that all other phases build on.**

**Files:**
- New: `extensions/agent-orchestrator/pending-questions.ts`
- Modify: `extensions/agent-orchestrator/types.ts`
- Modify: `extensions/agent-orchestrator/index.ts`

### 1a. New type `PendingQuestion` in types.ts

```typescript
export type QuestionType = "permission" | "clarification" | "escalation";

export interface PendingQuestion {
  id: string;                          // short UUID (8 chars)
  instanceId: string;
  type: QuestionType;
  agentId?: string;                    // from canUseTool options.agentID
  toolName?: string;                   // for permission type
  toolInput?: Record<string, unknown>;
  question: string;
  options?: string[];
  createdAt: number;
  timeoutMs: number;                   // default 5min, auto-deny on expiry
  status: "pending" | "answered" | "timeout" | "cancelled";
  answer?: string;
  _resolve?: (answer: { action: string; text?: string; updatedInput?: Record<string, unknown> }) => void;
}
```

Add to `CcInstance`: `pendingQuestions: PendingQuestion[]` and `interactive?: boolean`.

### 1b. New module `pending-questions.ts`

Manages the question store and lifecycle:

- `createQuestion(opts)` → returns `{ question, promise }` — the promise blocks until resolved
- `resolveQuestion(id, answer)` → resolves the blocking promise, unblocks the canUseTool callback
- `listPendingQuestions(instanceId?)` → returns pending questions for cc_status display
- `cancelAllForInstance(instanceId)` → cleanup on abort (resolves all pending with deny)
- Auto-timeout via `setTimeout` — auto-deny after `timeoutMs` to prevent indefinite hangs

### 1c. New tool: `cc_respond`

```
cc_respond(questionId: string, action: "allow"|"deny"|"answer", text?: string)
```

Lloyd calls this to answer pending questions from subagents. Resolves the matching promise, which unblocks the `canUseTool` callback and lets the SDK continue.

### 1d. Update `cc_status` to surface pending questions

Add `pendingQuestions` array to status output so Lloyd (and Mission Control) can see what needs answering:

```json
{
  "id": "inst-abc",
  "status": "running",
  "pendingQuestions": [
    { "id": "q-1234", "type": "permission", "agentId": "coder", "toolName": "Write", "question": "Write to src/auth.ts" }
  ]
}
```

### 1e. Replace `injectGatewayMessage` with `injectHookMessage`

New lightweight notification function using the HTTP hooks endpoint:

```typescript
const HOOKS_URL = "http://127.0.0.1:18789/hooks/agent";

async function injectHookMessage(message: string, logger: any): Promise<void> {
  try {
    const hooksToken = loadHooksToken(); // from openclaw.json → hooks.token
    await fetch(HOOKS_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${hooksToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        sessionKey: "agent:main:main",
      }),
    });
  } catch (err: any) {
    logger.warn?.(`agent-orchestrator: hook inject failed: ${err.message}`);
  }
}
```

This replaces the 160-line `injectGatewayMessage()` function (WebSocket + Ed25519 device auth). Used for:
- Question notifications (new)
- Completion notifications (existing, simplified)
- Progress notifications (Phase 5)

The existing `injectGatewayMessage` can be removed once `injectHookMessage` is verified working.

### 1f. Cleanup on abort

`cc_abort` calls `cancelAllForInstance()` to resolve all pending promises with deny, preventing memory leaks.

---

## Phase 2: Approval Gates for Destructive Actions

**The "permission prompt" pattern — subagent file writes and commands require user approval.**

**Files:**
- Modify: `extensions/agent-orchestrator/index.ts`

### 2a. Add `interactive: boolean` parameter to `cc_orchestrate` and `cc_spawn`

Opt-in flag. Default `false` preserves existing fire-and-forget behavior.

### 2b. `buildCanUseTool` callback factory

```typescript
const GATED_TOOLS = new Set(["Write", "Edit", "Bash"]);
const SAFE_BASH = [/^(cat|ls|head|tail|wc|find|grep|rg|git\s+(status|log|diff|show))\b/];
```

Logic:
- Read-only tools (Read, Glob, Grep, Task, MCP reads) → **auto-allow**
- Safe bash patterns (ls, cat, git status, etc.) → **auto-allow**
- Destructive tools (Write, Edit, non-safe Bash) → **create PendingQuestion**, POST to `/hooks/agent`, await promise, return allow/deny based on user response

### 2c. Wire into query() options

When `interactive: true`:
- `permissionMode: "default"` (not `bypassPermissions`)
- `allowedTools` limited to read-only set (destructive tools trigger canUseTool)
- `canUseTool: buildCanUseTool(instance, logger)`

When `interactive: false`: unchanged behavior (bypassPermissions, no callback).

### 2d. Timeout behavior

When a pending question times out (5 minutes default), the promise resolves with deny. The subagent sees a tool error and can retry or proceed differently. The instance is NOT killed — only that specific tool call is denied. Timeout is configurable per-instance.

---

## Phase 3: Mid-Execution Clarification (AskUserQuestion Interception)

**Subagents can pause and ask questions when they encounter ambiguity.**

**Files:**
- Modify: `extensions/agent-orchestrator/index.ts` (buildCanUseTool)
- Modify: `extensions/agent-orchestrator/agents/coder.ts`, `planner.ts`, `operator.ts`

### 3a. Add `AskUserQuestion` to agent tool lists

Agents that should be able to escalate get `AskUserQuestion` in their `tools` array. Disallowed when not in interactive mode via `disallowedTools`.

### 3b. Intercept in `canUseTool`

When `toolName === "AskUserQuestion"`:
1. Extract question text and options from tool input
2. Create PendingQuestion with `type: "clarification"`
3. POST to `/hooks/agent` → pushes question into Lloyd's session
4. Await promise (blocks until user answers or timeout)
5. Return `{ behavior: "deny", message: "[User Response] ${answer.text}" }`

**Key insight:** The denial message carries the user's answer. The agent sees it as tool output and incorporates the response. This is simpler than `interrupt()` + `streamInput()` and works within the existing `canUseTool` contract.

Agent prompts get a note: *"If AskUserQuestion is denied, the denial message contains the user's response."*

---

## Phase 4: Interactive Planning Tool

**Collaborative requirements gathering before execution begins.**

**Files:**
- Modify: `extensions/agent-orchestrator/index.ts`
- Modify: `extensions/agent-orchestrator/orchestrator-prompt.ts`

### 4a. New tool: `cc_plan_interactive`

Async like `cc_orchestrate` (returns instance ID immediately), but configured for collaborative planning:
- `permissionMode: "default"` + canUseTool callback
- Opus model for deeper analysis
- `AskUserQuestion` + read-only tools only (no writes)
- System prompt appended with interactive planning instructions
- Lower budget ($1) and fewer turns (15)

Lloyd's expected workflow (no polling needed — hooks push questions):
1. `cc_plan_interactive(task)` → returns instanceId
2. Hook pushes question into Lloyd's session → Lloyd wakes up
3. Lloyd presents question to user → gets answer
4. `cc_respond(questionId, answer)` → planner continues
5. Repeat until completion hook arrives
6. `cc_result(instanceId)` → retrieve final plan

### 4b. Planning prompt variant in orchestrator-prompt.ts

```
You are in INTERACTIVE PLANNING mode. Your job is to:
1. Explore the codebase thoroughly using Read/Glob/Grep
2. Ask clarifying questions using AskUserQuestion when you encounter ambiguity
3. Produce a detailed, actionable execution plan
Do NOT modify any files. Ask about: scope, preferences, constraints, edge cases.
```

---

## Phase 5: Progress Streaming

**Real-time visibility into what subagents are doing.**

**Files:**
- Modify: `extensions/agent-orchestrator/query-consumer.ts`
- Modify: `extensions/agent-orchestrator/types.ts`

### 5a. Process task lifecycle messages in consumeQuery

The SDK emits `task_started`, `task_progress`, `task_notification` system messages that `consumeQuery` currently ignores. Add handling:

- Update `instance.activity` with current task description
- Push to `recentMessages` ring buffer for Mission Control
- Log to instance JSONL
- When `interactive: true`, POST milestone notifications to `/hooks/agent` (only `task_notification` completions, not every progress tick — avoid flooding)

### 5b. Extend InstanceMessage types

Add `task_progress`, `question_pending`, `question_answered` message types for richer Mission Control display.

---

## Prompt Engineering for Lloyd

Add to Lloyd's agent prompt (AGENTS.md or soul):

```markdown
## Interactive Agent Orchestration

When you receive an [Agent Question] message from a running instance:
1. Present the question to the user and get their answer
2. Use cc_respond(questionId, action, text) to deliver the answer immediately
3. Agents are blocked waiting — never leave pending questions unanswered

When using cc_orchestrate/cc_spawn with interactive: true, or cc_plan_interactive:
- Questions and progress are pushed to your session automatically via hooks
- No need to poll cc_status — you'll receive notifications
- Use cc_status only to check overall progress or if you haven't heard back
```

---

## Tool Summary

| Tool | Status | Description |
|------|--------|-------------|
| `cc_orchestrate` | Modified | Adds `interactive: boolean` parameter |
| `cc_spawn` | Modified | Adds `interactive: boolean` parameter |
| `cc_status` | Modified | Shows pending questions in output |
| `cc_abort` | Modified | Cleans up pending questions on abort |
| `cc_respond` | **New** | Answer a pending question from a subagent |
| `cc_plan_interactive` | **New** | Launch interactive planning session |
| `cc_result` | Unchanged | Retrieve completed results |

Total tools: 7 (was 5).

---

## Bonus: Simplify Existing Completion Notifications

The existing `injectGatewayMessage()` (lines 47–266 of index.ts) can be replaced by `injectHookMessage()` for completion notifications too. This removes ~160 lines of WebSocket + device identity + Ed25519 signing code. The gateway WebSocket helpers (`loadDeviceIdentity`, `loadDeviceToken`, `base64UrlEncode`, `derivePublicKeyRaw`, `publicKeyRawBase64Url`, `signPayload`) become dead code and can be removed.

---

## Edge Cases & Mitigations

| Risk | Mitigation |
|------|------------|
| **Timeout cascading** — user slow to respond, questions pile up | Each question has independent timeout (5min default). Agent sees sequential denials and adapts. |
| **Lloyd's context window** — injected hook messages consume context | Keep notifications concise. Rely on cc_status for details, not the notification body. |
| **Concurrent interactive instances** — questions from different instances interleave | questionId includes instanceId prefix. cc_respond requires specific questionId. |
| **canUseTool + bypassPermissions** — must not conflict | Interactive mode uses `permissionMode: "default"`. Non-interactive unchanged. |
| **AskUserQuestion in subagents** — do subagents inherit canUseTool? | Yes — canUseTool is on top-level Options, fires for all agents including Task-spawned subagents. `options.agentID` identifies which one. |
| **Memory leaks** — unresolved promises if instance crashes | `cancelAllForInstance()` on abort + timeout auto-deny prevents leaked promises. |
| **Hooks endpoint down** — gateway not running | Log warning, question still exists in pending store. Lloyd can discover via cc_status as fallback. |

---

## Verification Plan

1. **Phase 1**: Spawn non-interactive instance → verify cc_status works as before. Verify `injectHookMessage` delivers to Lloyd's session. Call cc_respond with invalid ID → verify error response.
2. **Phase 2**: `cc_spawn("coder", task, interactive: true)` → coder attempts Write → verify question pushed to Lloyd's session via hook → cc_respond allow → verify file written. Test deny path. Test timeout auto-deny.
3. **Phase 3**: Agent with AskUserQuestion → verify clarification question pushed to Lloyd → answer delivered via denial message → agent continues incorporating the answer.
4. **Phase 4**: `cc_plan_interactive(task)` → verify planner explores codebase and asks questions → answer flow works → plan produced.
5. **Phase 5**: Long-running orchestration → verify task lifecycle messages appear in cc_status and (if interactive) pushed to Lloyd's session via hook.

Gateway restart between phases:
```bash
distrobox-enter lloyd -- bash -c "kill $(lsof -ti :18789) 2>/dev/null; sleep 2"
distrobox-enter lloyd -- systemctl --user start openclaw-gateway.service
```
