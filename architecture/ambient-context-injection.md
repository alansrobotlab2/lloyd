---
segment: architecture
type: architecture
status: implemented
date: 2026-09-12
tags: [ambient,architecture,autonomy,context-injection,lloyd,mcp]

---

# Ambient Context Injection

> How background producers (autonomy tasks, pipelines, cron jobs) surface context into the user's active chat session. Backlog #295, shipped 2026-04-19.

## Status: shipped 2026-04-19, and still the whole mechanism

Five things were checked by hand the night it shipped: the drain rendering a
live `<ambient-signals>` block, `ambient_decide`'s gate and set/take semantics,
three fires of one `dedup_key` collapsing to one entry, a 1-second TTL enforced
at drain, and the resolver's hint path and filesystem fallback. That manual
list is what "all five verification tests pass" meant. The durable pins today
are `tests/test_session_queue.py` (nine cases over the turn queue),
`tests/test_active_session_resolution.py` (five over the resolver) and the
ambient cases in `tests/test_prefetch.py` — 63 tests, all green when re-run on
2026-09-12.

The ship note recorded the merge as `932a71c`. The repository's history has
been rewritten since, so that hash resolves to nothing; the same commit, same
message — `feat(ambient): session_inject_context + ambient_decide (#295)` — is
`deec25f` today. Builds on #296 (per-session turn queue) for the
notable/urgent path.

## Core Concept

Producers call one MCP tool — `session_inject_context` — with a priority. The tool picks the delivery mechanism, branching on `priority` to choose which endpoint it POSTs to. Producers never touch the turn queue directly.

| Priority | Cost | Mechanism | When the agent sees it |
|----------|------|-----------|------------------------|
| `ambient` | Zero — no model call | Prefetch queue (passive) | Next user turn's `<context>` block |
| `notable` | A full agent turn | Synthetic turn via the #296 queue | Immediately, wrapped in an `<ambient>` envelope |
| `urgent` | A full agent turn | Synthetic turn, stronger framing | Immediately, with "surface now" framing |

**Key invariant:** producers declare *intent*, not mechanism. They never need to think about turn queues or what a turn costs.

The tool's own description still sells the cheap tier as "zero SDK cost"
(`agent_mcp/ambient.py`, the `session_inject_context` description), and this
page said the same until 2026-09-11. It is a fossil of the claude-agent-sdk
era: the in-process harness replaced the SDK, so the expensive tiers now cost a
`run_query` turn against the local vLLM engine rather than a subprocess. The
economics the tiers were built around are unchanged — one is free and two are
not — which is why the wording survived this long without misleading anyone.

## Mechanism 1 — Prefetch Drain (ambient)

Passive. The agent does NOT wake up. Context only surfaces when the user sends their next message.

### Flow

```
Producer (autonomy task, cron, pipeline)
  → session_inject_context(priority="ambient",source,summary,content?)
  → session_id=="" → GET /api/sessions/active resolves the target
  → MCP tool POSTs to /api/sessions/{sid}/inject-prefetch
  → Backend enqueues AmbientPrefetchEntry in _ambient_prefetch_queue[sid]
  → [NOTHING HAPPENS — queue sits passively]

                    ... time passes ...

User sends message
  → messages.py → prefetch_context_async(text,session_id)
  → _prefetch_prepare drains FIRST, before the MIN_MESSAGE_LEN guard
  → drain_ambient_prefetch(session_id)
      ├─ filters expired (expires_at)
      ├─ sorts newest-first, takes AMBIENT_PREFETCH_DRAIN_MAX = 3
      └─ puts the overflow back for the next turn
  → _format_context renders <ambient-signals> FIRST inside <context>
  → "<context>…</context>" + "\n\n" + the user's text
  → model sees: <context>...<ambient-signals>...</ambient-signals>...</context>
                \n\n<user text>
```

Two ordering rules in there are load-bearing. The drain runs **above** the
`MIN_MESSAGE_LEN` guard, so "ok" still surfaces a queued signal — a short
message must not suppress an injection the producer already decided was worth
showing. And it runs on the **caller's** thread, not in the worker thread the
budgeted search phase uses (`_prefetch_prepare` vs `_prefetch_run`): the queue
is mutated by producers on the event loop, so draining it off-thread would race
them.

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
- **TTL** — entries carry `expires_at`, from `ttl_seconds` (default 3600, and the ambient tier only). Expired entries are evicted silently at drain time —
  and expiry is evaluated *only* there: `drain_ambient_prefetch`
  (`app/sessions_io.py:334-352`) is both the only TTL check and the only place a
  session's key leaves the dict, and it runs only when that session next takes a
  turn. A signal queued against a session that never takes one — a worker or
  autonomy session, which `/inject-prefetch` does not refuse — is therefore
  neither delivered nor reclaimed, and its key persists for the process
  lifetime. Backlog #910.
- **Two caps, and they answer different questions.** `AMBIENT_PREFETCH_CAP = 5`
  bounds what a session may *hold*: the oldest is evicted on enqueue.
  `AMBIENT_PREFETCH_DRAIN_MAX = 3` bounds what one turn may *see*: the drain
  takes the three newest and puts the remainder back. A burst is therefore
  spread across turns rather than dropped on the floor, and the newest signal
  is never the one that waits.
- **`content` is truncated at 800 chars** when rendered, with an ellipsis.
- **No model call** — nothing fires until the user engages naturally.
- **A short message still drains.**
  `tests/test_prefetch.py::test_short_message_skips_search_but_keeps_ambient`
  pins all three halves of that: the signal lands, the search phase does not
  run, and a second call returns the bare message because the queue is empty.

## Mechanism 2 — Synthetic Turn (notable / urgent)

A full agent turn. The agent wakes up, reads the signal, and decides whether to surface it to the user or stay silent via `ambient_decide`.

### Flow

```
Producer
  → session_inject_context(priority="notable",source,summary,content?)
  → MCP tool POSTs to /api/sessions/{sid}/inject
  → 409 if the target is not a user session (see "A worker session is nobody")
  → build_ambient_turn() wraps text in envelope:

      <ambient priority="notable" source="autonomy:task-42" session_id="20260419_..."> 
      {summary + content}
      </ambient>

      This is a background signal from `autonomy:task-42`. You were not asked a
      question — consider whether to mention this to the user. If it is not worth
      interrupting them, call
      `ambient_decide(session_id="...",surface=false,reasoning="...")` and stop.
      If it is worth surfacing, reply briefly and naturally.

  → Enqueued on the SessionQueue's ambient tier (from #296): dedup_key collapses
    a queued duplicate, and AMBIENT_QUEUE_CAP = 3 drops the oldest beyond it
  → Queue consumer picks it up when no user turn is pending or running

                    ... turn runs ...

Agent either:
  A) Replies normally
     → Assistant message lands, tagged source="ambient"
  B) Calls ambient_decide(session_id,surface=false,reasoning="...")
     → Server stores decision in _ambient_decisions[sid]
     → cancel_event.set() 0.5s LATER, via _deferred_ambient_cancel
     → On the harness `result` event: take_ambient_decision consumed
     → messages.py writes muted breadcrumb instead of assistant message:
         "(ambient: Lloyd reviewed and chose not to surface — <reasoning>)"
     → Emits ambient_silent SSE event onto the turn's broker queue
```

`urgent` differs from `notable` by one verb in that envelope — "surface this
now if the user should know" against "consider whether to mention this to the
user". Everything else about the two tiers is identical.

**The cancel is deferred on purpose, and a synchronous one would have been a
self-inflicted wound.** `ambient_decide` is dispatched as a tool call, and that
in-flight call races `cancel_event` in the harness loop; setting the event
inside the handler cancels the dispatch that is still running, and the agent
reads back `Tool 'ambient_decide' cancelled by user` for a call that in fact
succeeded. `_deferred_ambient_cancel` sleeps 0.5 s so the tool result
propagates first, and the loop's top-of-iteration check then stops the turn
cleanly. The event reference is captured up front, so a turn that ends
naturally in the meantime just sets an orphaned event — a no-op for any later
turn, not a cancel leaking forward.

**The decision is read before the cancelled path, not after.** Because the
server schedules that cancel, a silenced ambient turn can finish down either
branch — normal completion or cancelled-mid-stream — and only the earlier check
gets the breadcrumb written in both cases.

### Preemption

If the user sends a message while an ambient turn is running or queued:

- **Running ambient** → preempted. `enqueue_turn` marks `current.preempted` and
  sets `cancel_event`; the consumer's `finally` writes
  `(ambient turn interrupted by user input)` into the transcript so history
  shows the injection was cut off rather than simply absent.
- **Queued ambient** → **not** dropped. This page said it was until 2026-09-11,
  and the queue has never behaved that way: the consumer pops `pending_user`
  before `pending_ambient`, so a queued ambient waits and runs once the user
  has been served. Dropping one is always a separate, explicit act —
  `POST /api/sessions/{id}/cancel?drain_pending=true`,
  `DELETE /api/sessions/{id}`, a `dedup_key` collision, or the
  `AMBIENT_QUEUE_CAP` eviction.

User turns always win, and `drain_pending` never drops one: `source=None` means
"ambient only" (`app/sessions_io.py:671-698`) and its `source == "user"` branch
has no caller anywhere. This page previously named `DELETE /api/sessions/{id}`
as the exception that "really does want both" — it does not. That handler passes
`source=None` under a `# drain all queued turns` comment the code does not
honour (`app/routers/sessions.py:767`), so a queued **user** turn survives the
wipe, still runs, and writes nothing: `_append_messages` goes through
`mutate_session`, which no-ops once the session JSON is gone. Backlog #909.

## Components

### State (`app/sessions_io.py`)

| Name | Type | Purpose |
|------|------|---------|
| `_last_user_session_id` | `Optional[str]` | Hint for `get_active_session_id()` — most recent session that received a user turn |
| `_ambient_prefetch_queue` | `dict[session_id, list[AmbientPrefetchEntry]]` | Mechanism 1's per-session queue. In memory only: a backend restart loses undelivered signals |
| `_ambient_decisions` | `dict[session_id, dict]` | What `ambient_decide` recorded, popped by the turn's own completion path via `take_ambient_decision` |
| `SessionQueue.pending_ambient` | `deque[SessionTurn]` | Mechanism 2's tier, popped only when `pending_user` is empty |
| `AMBIENT_PREFETCH_CAP` / `AMBIENT_PREFETCH_DRAIN_MAX` | `5` / `3` | Hold cap and per-turn cap |
| `AMBIENT_QUEUE_CAP` | `3` | Queued ambient *turns* per session; oldest evicted |

### Endpoints (`app/routers/sessions.py`)

| Route | Does |
|---|---|
| `POST /api/sessions/{id}/inject-prefetch` | Mechanism 1. 404 on an unknown session; 400 without `source` and `summary` |
| `POST /api/sessions/{id}/inject` | Mechanism 2. 409 on a non-user session, 404 on an unknown one |
| `POST /api/sessions/{id}/ambient-decide` | 400 unless the session's *current* turn has `source == "ambient"` |
| `GET /api/sessions/active` | The resolver a producer gets when it passes `session_id=""` |
| `GET /api/sessions/{id}/prefetch-queue` | Debug peek that does not drain (`peek_ambient_prefetch`) |
| `POST /api/sessions/{id}/cancel?drain_pending=true` | Drops queued ambients, never queued user turns |

### A worker session is nobody (2026-09-07)

`get_active_session_id` is what every producer means by "the user", and for
most of this system's life it meant "the last session to receive a user-source
turn". Worker turns arrive through the chat path — that is how they get Inner
Voice — so they set that hint too. On 2026-09-07 the morning brief (a MockBOT
meeting that night) was injected into a backlog-triage worker session that had
finished 77 seconds earlier, and was answered there, to nobody.

`NON_USER_PLATFORMS = {"autonomy", "worker"}` and `is_user_session` are the one
definition now, and **both** resolution rules apply it: the in-memory hint is
re-validated against the session's platform before it is returned, and the
mtime fallback skips machine sessions while scanning. It is a deny-list on
purpose — a client this code has never heard of must keep receiving its briefs
rather than silently losing them, so an unknown platform stays eligible and a
missing one reads as the web UI.

`/inject` refuses such a session with **409 rather than a 200 "skipped"**, so
`session_inject_context` reports `ok=false` and no producer records the
notification as delivered. Note the asymmetry: `/inject-prefetch` carries no
such gate — it checks only that the session file exists — because the cheap
tier's defence is the resolver, and a producer that passes an explicit worker
`session_id` there will queue a signal nobody drains.
`tests/test_active_session_resolution.py` pins the resolver and the 409.

### Inner Voice is the second producer

Mechanism 2 is not reached only from MCP. The Inner Voice observer's `ambient`
lever builds a turn through the same `build_ambient_turn` + `enqueue_ambient`
pair (`_iv_enqueue_ambient_cb` in `app/routers/messages.py`), which is why the
callback takes a `producer` argument: it lands in the payload as
`producer_source`, and `_iv_should_fire_on_turn` reads it back to decide
whether the follow-up itself gets observed. A discretionary `inner_voice`
ambient must not be — the observer would re-judge its own work without bound —
while an `inner_voice_goal` retry must be, or the `/goal` completion loop stops
after a single attempt.

`autonomy.run_task` also passes `turn_source="ambient"` when attaching an
observer, but that is the observer's own gate on the direct path, not a turn
on this queue.

### What the user sees, and what the model sees again

- **The envelope is shown, not hidden.** `_extract_subliminal_prefix` classifies
  a `<ambient ...>` wrapper as kind `ambient_envelope` and persists it as a
  `role: "subliminal"` row (#306), so the chat can surface exactly what the
  agent was handed. The whole `prefetched_text` is the injection here, because
  the producer's text sits *inside* the wrapper rather than after it.
- **Breadcrumbs are `role: "system"`, `source: "ambient"`** — both the silent
  one and the interrupted one. `ChatPanel`'s `MessageRow` has no `system` case,
  so they fall through to the plain-text branch and render as an ordinary
  left-aligned card.
- **Neither breadcrumb re-enters the prompt.** Compaction's conversation filter
  keeps `system` rows, but `_prepare_messages_for_harness` keeps only
  `user`/`assistant`/`tool`, so the model never re-reads its own decision not
  to speak.
- **The `ambient_silent` frame goes onto the turn's broker queue**, which only a
  client subscribed to *that turn's* SSE stream can see. A producer POSTing
  `/inject` gets a `turn_id` back and does not subscribe, so in practice the
  durable record of a silent decision is the persisted breadcrumb, not the
  frame.

### Configuration

There is no `ambient:` block in config.yaml, and no kill switch for the
mechanism as a whole — the switches are all at the tool layer.
`mcp_servers.lloyd-mcp.disabled_tools` is the real one. `ambient_decide` sits
in `harness.tool_search.baseline_tools` and `session_inject_context` does not,
which costs nothing today because `tool_search.enabled` is `false` in both
config.yaml and `data/tool_overrides.yaml` and every tool is advertised; if
progressive disclosure is ever switched back on, the agent-facing half stays
always-visible while the producer-facing half is one discovery round-trip away.
`ambient_decide` is also in `PLAN_MODE_ALWAYS_ALLOWED` — it records a routing
choice for the current turn and reaches nothing outside the process, so plan
mode has no reason to block it.

## Review log

- 2026-09-12 — **stale**. Both mechanisms verified live against the tree and
  still accurate: drain order and caller-thread rule, the 5 / 3 / 3 caps,
  800-char truncation, the envelope text and its one-verb `urgent` variant, the
  deferred 0.5 s cancel, the `worker`/`autonomy` deny-list and the `/inject` 409,
  the Inner Voice second-producer path, every config and `PLAN_MODE_ALWAYS_ALLOWED`
  claim, and 63 green tests across the three pinned files. Two claims were wrong
  and are corrected in place: `DELETE /api/sessions/{id}` does *not* drain queued
  user turns (#909), and TTL is enforced only at drain, so signals aimed at a
  session that never takes a turn are neither delivered nor reclaimed (#910).
