---
segment: architecture
type: architecture
tags: [ambient,architecture,autonomy,context-injection,lloyd,mcp]

---

# Ambient Context Injection

> How background producers (autonomy tasks,pipelines,cron jobs) surface context into the user's active chat session. Backlog #295,shipped 2026-04-19.

## Status: Shipped (2026-04-19)

All five verification tests pass. Merged to main as `932a71c`. Builds on #296 (per-session turn queue) for the notable/urgent path.

## Core Concept

Producers call one MCP tool — `session_inject_context` — with a priority. The harness picks the delivery mechanism. Producers never touch the turn queue directly.

| Priority | Cost | Mechanism | When the agent sees it |
|----------|------|-----------|------------------------|
| `ambient` | Zero SDK cost | Prefetch queue (passive) | Next user turn's `<context>` block |
| `notable` | Full SDK call | Synthetic turn via #296 queue | Immediately,wrapped in `<ambient>` envelope |
| `urgent` | Full SDK call | Synthetic turn,stronger framing | Immediately,with "surface now" instruction |

**Key invariant:** producers declare *intent*,not mechanism. They never need to think about turn queues or SDK calls.

## Mechanism 1 — Prefetch Drain (ambient)

Passive. The agent does NOT wake up. Context only surfaces when the user sends their next message.

### Flow

```
Producer (autonomy task,cron,pipeline)
  → session_inject_context(priority="ambient",source,summary,content?)
  → MCP tool POSTs to /api/sessions/{sid}/inject-prefetch
  → Backend enqueues AmbientPrefetchEntry in _ambient_prefetch_queue[sid]
  → [NOTHING HAPPENS — queue sits passively]

                    ... time passes ...

User sends message
  → messages.py → prefetch_context(text,session_id)
  → drain_ambient_prefetch(session_id)
      ├─ filters expired (TTL)
      ├─ caps at N per drain
      └─ empties the queue
  → _format_context renders <ambient-signals> block inside <context>
  → Combined <context> prepended to user's text
  → SDK sees: <context>...<ambient-signals>...</ambient-signals>...</context>\n<user text>
```

### Rendered Context Block

```xml
<ambient-signals>
Background producers queued these signals for you. The user did NOT ask —
reference them only if naturally relevant to what they're saying now.
- **[autonomy:task-42]** 3 new emails worth reviewing
  > From: client@foo.com ("contract update"),...
- **[pipeline:research-117]** Background research on quantum compilers finished
  > Key finding: tket compiler beats qiskit on circuits >10q by 14%...
</ambient-signals>
```

### Key Properties

- **Dedup by `dedup_key`** — same key re-firing replaces the previous unsent entry (newest wins). Default `dedup_key = source`.
- **TTL** — entries have `expires_at`. Expired entries filtered at drain time.
- **Cap** — `AMBIENT_PREFETCH_CAP = 5` entries max per session (FIFO evict).
- **No SDK cost** — nothing fires until user engages naturally.

## Mechanism 2 — Synthetic Turn (notable / urgent)

Full SDK call. The agent wakes up,reads the signal,and decides whether to surface it to the user or stay silent via `ambient_decide`.

### Flow

```
Producer
  → session_inject_context(priority="notable",source,summary,content?)
  → MCP tool POSTs to /api/sessions/{sid}/inject
  → build_ambient_turn() wraps text in envelope:

      <ambient priority="notable" source="autonomy:task-42" session_id="20260419_..."> 
      {summary + content}
      </ambient>

      This is a background signal from `autonomy:task-42`. You were not asked —
      consider whether to mention this to the user. If it is not worth interrupting
      them,call ambient_decide(session_id="...",surface=false,reasoning="...").

  → Enqueued as ambient-tier Turn in SessionQueue (from #296)
  → Queue consumer picks it up when no user turn is active

                    ... turn runs ...

Agent either:
  A) Replies normally
     → Assistant message lands,tagged source="ambient"
  B) Calls ambient_decide(session_id,surface=false,reasoning="...")
     → Server stores decision in _ambient_decisions[sid]
     → cancel_event.set() → SDK generation stops
     → On ResultMessage: take_ambient_decision consumed
     → messages.py writes muted breadcrumb instead of assistant message:
         "(ambient: Lloyd reviewed and chose not to surface — <reasoning>)"
     → Emits ambient_silent SSE event
```

### Preemption

If the user sends a message while an ambient turn is running or queued:

- **Queued ambient** → dropped (optional via `drain_pending=true` on cancel)
- **Running ambient** → preempted,`cancel_event.set()`,transcript gets breadcrumb noting the interrupt

User turns always win.

## Components

### State (`app/sessions_io.py`)

| Name | Type | Purpose |
|------|------|---------|
| `_last_user_session_id` | `Optional[str]` | Hint for `get_active_session_id()` — most recent session that received a user turn |
| `_ambient_prefetch_queue` | `dict[session_id,list[AmbientPrefetchEntry]]` | Mec