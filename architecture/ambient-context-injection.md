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
- **TTL** — entries carry `expires_at`, from `ttl_seconds` (default 3600, and the ambient tier only). Expiry is evaluated on **both** doors, so reclaiming a signal never depends on a turn arriving for the session it was queued against. `enqueue_ambient_prefetch` (`app/sessions_io.py:564`) purges that session's dead entries before it stores the new one, names each casualty's `source` in `dropped`, refuses a signal that was already expired at the moment it was written, and releases the session's key whenever the purge empties it; `drain_ambient_prefetch` (`app/sessions_io.py:635`) keeps its own filter as defence in depth, because an entry can pass its deadline while it waits rather than at write — and names each casualty in the log, since a path that drops a signal without a trace is how #910 stayed invisible. What `expires_at == 0.0` means is **no deadline**, not the epoch — every caller that omits `ttl_seconds` gets that value, so a purge that read it as "already expired" would silently delete nearly every ambient signal in the system and still report a clean queue. `tests/test_ambient_prefetch_retention.py` pins both halves of that boundary: expiry-on-write, key release through every door (a dead signal into an empty session, purge-to-empty, drain-to-empty, cap eviction), the drain's overflow re-insert and its logged drop of a signal that expired while it waited, that drain driven through its only production caller `prefetch._prefetch_prepare` rather than called directly, the `DELETE` and `/inject-prefetch` routes, and this page's own `app/sessions_io.py:<line>` citations (`test_the_pages_line_citations_point_at_the_functions_they_name`), since a stale pointer reads as a discovered defect. What the page said before, and what was wrong with it, is in the 2026-09-20 review-log entry below — and the three sentences that read as current but described the old tree are pinned absent by `test_the_architecture_page_describes_the_retention_that_shipped`, which is the machine-readable version of that entry.
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

User turns always win, and `/cancel` never drops one: `drain_pending`'s
`source=None` default means "ambient only" (`app/sessions_io.py:986-1030`), which
is what `drain_pending=true` on that route relies on. Draining the user tier is
the explicit `source="all"` sentinel — both tiers, ambient then user, summed
count — and `DELETE /api/sessions/{id}` is its only caller
(`app/routers/sessions.py:756-785`). That route wipes the session outright, so a
queued **user** turn it left behind would still run and still write nothing:
`_append_messages` goes through `mutate_session`, which no-ops once the session
JSON is gone — a full model turn spent on a transcript nobody will read again.

This paragraph read the other way until 2026-09-20: it said the delete handler
asked for the ambient-only default under a comment claiming it drained every
queued turn, that the `source == "user"` branch had no caller at all, and that
this route was therefore *not* the exception that "really does want both". All
three were true of the tree that page reviewed, and all three are false now —
the branch is reachable, through `"all"`. Fixed by #909.

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
| `POST /api/sessions/{id}/inject-prefetch` | Mechanism 1. 404 on an unknown session; 409 on a non-user one (#910); 400 without `source` and `summary` |
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
notification as delivered. `/inject-prefetch` makes the same refusal for the same
reason, since #910. It used to check only that the session file existed, on the
theory that the cheap tier's defence is the resolver — but a producer that passes
an explicit worker `session_id` got a 200, and because expiry then lived only in
the drain, that response queued a signal nobody would drain and nothing could
reclaim. `tests/test_active_session_resolution.py` pins the resolver and `/inject`'s
409; `tests/test_ambient_prefetch_retention.py` pins this route's.

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

- 2026-09-20 — **#910 landed**: expiry moved onto the write path, so reclaiming an
  ambient signal no longer requires a turn for the session it was queued against.
  `enqueue_ambient_prefetch` purges that session's dead entries, reports their
  `source`s in `dropped`, refuses a signal already expired at the moment it was
  written, and releases the session's key when the purge empties it; the drain
  keeps its filter as defence in depth and now **logs each entry it drops** rather
  than evicting silently — a signal that dies without a trace is how this defect
  stayed invisible, and until this change no test in the tree had exercised the
  drain's expiry at all: `drain_ambient_prefetch` appears in no other test file,
  and none of the five `AmbientPrefetchEntry` constructions elsewhere in `tests/`
  passes `expires_at`, so every pre-existing entry arrived as the no-deadline
  sentinel. (Stated about this dataclass, not the field name — `expires_at` also
  appears in seven other test files as grant and mail-object expiry, which is why
  the narrower wording is the true one.) `DELETE /api/sessions/{id}` now
  discards prefetch entries along with queued turns, seeded in its test through
  `POST /inject-prefetch` so what gets cleared is an entry the route itself stored
  rather than one a helper invented. `POST
  /api/sessions/{id}/inject-prefetch` answers 409 for a `NON_USER_PLATFORMS`
  session — the refusal `/inject` has returned since 2026-09-07 and the one this
  route lacked, which is how a producer naming a worker or autonomy id got a 200
  for a signal nobody would ever read. `app/routers/messages.py` also lost its
  unused `enqueue_ambient_prefetch` / `AmbientPrefetchEntry` imports, so the
  producer set is as small as it actually is: `sessions.py` is the only writer.
  Pinned by 16 nodes (15 functions, one parametrised over both non-user
  platforms) in the new `tests/test_ambient_prefetch_retention.py`, one
  per door a signal or a key can leave through — including the drain driven
  through `prefetch._prefetch_prepare`, its only production caller, so the
  seam a turn actually crosses is covered and not just the function that
  changed — plus this page.
  Verified 2026-09-20: `pytest tests/test_ambient_prefetch_retention.py` → 16
  passed; with the three source files reverted to their pre-fix versions and the
  same file re-run, 10 of the 16 fail, each for the reason its clause names (the
  other six are the controls that must not change, one of which — the
  `_prefetch_prepare` seam — passes at base by design, because that caller's
  behaviour is being preserved, not changed). Blast radius, taken from
  `graph_affected` over `enqueue_ambient_prefetch` and `drain_ambient_prefetch`
  rather than from a filename guess, and re-measured 2026-09-21: **275 passed** over
  the 14 test files that reach this store — the nine the graph walk named
  (`test_prefetch`, `test_ambient_inject`, `test_session_queue`,
  `test_session_doc_claims`, `test_active_session_resolution`,
  `test_session_platform_checks`, `test_ambient_prefetch_retention`,
  `test_api_contracts`, `test_files_changed_surface`) plus five that touch it by
  name (`test_session_titles`, `test_component_manifest_prefetch_seam`,
  `test_brief_triage_clock_skill`, `test_background_inner_voice`,
  `test_grant_gate_session_path`). An earlier cut of this entry claimed "every
  ambient and session test in the tree → 160 passed" while naming seven files: the
  seven do measure 160, but they are not every such file, and a count that outruns
  the set it names is exactly the defect this page keeps recording.
- 2026-09-20 — **#909 landed**, which makes the 2026-09-12 entry's first
  correction historical rather than current. `DELETE /api/sessions/{id}` now
  drains both tiers through `drain_pending(source="all")`, so a queued user
  turn can no longer run into a deleted transcript; the ambient-only
  `source=None` default and `POST /api/sessions/{id}/cancel?drain_pending=true`
  are unchanged. Pinned by seven new tests in `tests/test_session_queue.py` (16
  in that file now) — one per branch that reaches the queues: the both-tiers
  delete, the drained turn that must not run, the `"all"` sentinel, the
  `source=None` default, the `user`-only drain, flag-less `/cancel`, and
  `/cancel?drain_pending=true` — plus six in the new
  `tests/test_session_doc_claims.py`, including the caller grep itself and what
  keeps this page's drain paragraph from re-staling. Verified 2026-09-20: `pytest
  tests/test_session_queue.py tests/test_session_doc_claims.py` → 22 passed,
  and four mutations of the fix — revert the delete call to `source=None`, cut
  the `"all"` arm off the `pending_user` pop, widen `/cancel` to `"all"`, flip
  `drain_pending`'s default to `"all"` — each failed the test that owns it.
 Both mechanisms verified live against the tree and
  still accurate: drain order and caller-thread rule, the 5 / 3 / 3 caps,
  800-char truncation, the envelope text and its one-verb `urgent` variant, the
  deferred 0.5 s cancel, the `worker`/`autonomy` deny-list and the `/inject` 409,
  the Inner Voice second-producer path, every config and `PLAN_MODE_ALWAYS_ALLOWED`
  claim, and the green tests across the pinned files. (This entry originally read
  "63 green tests across the three pinned files"; that number names a file set no
  later reader can identify — the plausible triads measure 27
  (`test_session_queue.py` + `test_session_doc_claims.py` +
  `test_active_session_resolution.py`), 33 (the first two +
  `test_session_platform_checks.py`) and 58 (`test_ambient_inject.py` +
  `test_active_session_resolution.py` + `test_session_platform_checks.py`) on
  2026-09-21, none of them 63. The count is dropped rather than replaced with a
  guess, and a set is named wherever a count is quoted.) Two claims were wrong
  and are corrected in place: `DELETE /api/sessions/{id}` does *not* drain queued
  user turns (#909), and TTL was enforced only at drain, so a signal aimed at a
  session that never takes one sat unreclaimed — both since fixed, #909 first and
  #910 on 2026-09-20; the entries above record each.
