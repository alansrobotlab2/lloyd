---
segment: architecture
type: architecture
status: implemented
date: 2026-10-12
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
are named by file, not by count: `tests/test_session_queue.py` and
`tests/test_active_session_resolution.py` over the queue and the resolver,
`tests/test_ambient_prefetch_retention.py` and `tests/test_session_doc_claims.py`
over this page's own sentences, and the ambient cases in `tests/test_prefetch.py`.
Re-measured 2026-10-16 by `--collect-only -q` on each file — collected nodes, not
passing ones, so the figure ages rather than reassures:
`tests/test_session_queue.py` 18, `tests/test_active_session_resolution.py` 5,
`tests/test_ambient_prefetch_retention.py` 16,
`tests/test_session_doc_claims.py` 12 (51 across those four), and
`tests/test_prefetch.py` 93. Those five are the mechanism's surface, and
`tests/test_ambient_decision_turn_scope.py`, which came in with #2599, is a sixth
that pins the decision store rather than the queue. Another file
reads this page's own sentences —
`tests/test_architecture_coverage_doc_claims.py` — so a claim that the page is
"green" has to say whether it ran that one too, which is exactly what #2606 was
filed about; its node for this slug is named in the review log below. The
"(nine cases)" and "63 tests" this paragraph carried are
the un-identified counts the review log below already retired for the same
reason: a number with no named set drifts and nobody notices.

The ship note recorded the merge as `932a71c`. The repository's history has
been rewritten since, so that hash resolves to nothing; the same commit, same
message — `feat(ambient): session_inject_context + ambient_decide (#295)` — is
`deec25f` today. Builds on #296 (per-session turn queue) for the
notable/urgent path.

## Core Concept

Producers call one MCP tool — `session_inject_context` — with a priority. The tool picks the delivery mechanism, branching on `priority` to choose which endpoint it POSTs to. Producers never touch the turn queue directly.

That sentence has one exception, and it is the branch *before* the priority one:
if no session resolves, `priority` no longer selects an endpoint. Since #1516 an
unaddressable producer falls through to the file-backed **next-session channel**
whatever it asked for (`agent_mcp/ambient.py:185-216` → `POST
/api/next-session-note`), so a 03:00 pass with no session to name still leaves its
signal for the morning. Two consequences the table below does not show: an
unresolvable `notable` or `urgent` is delivered as a passive note, with no agent
turn, no `ambient_decide` gate and no breadcrumb while the response says
`ok=true` (filed, #2600); and the note channel is drained by *any* turn that runs
prefetch, including a worker's, which consumes the note for the human who was
meant to receive it (#2598).

| Priority | Cost | Mechanism | When the agent sees it |
|----------|------|-----------|------------------------|
| `ambient` | Zero — no model call | Prefetch queue (passive) | Next user turn's `<context>` block |
| `notable` | A full agent turn | Synthetic turn via the #296 queue | Immediately, wrapped in an `<ambient>` envelope |
| `urgent` | A full agent turn | Synthetic turn, stronger framing | Immediately, with "surface now" framing |

**Key invariant:** producers declare *intent*, not mechanism. They never need to think about turn queues or what a turn costs.

The cheap tier is still sold as costing nothing, but the exact phrase "zero SDK
cost" is not in `agent_mcp/ambient.py` — it is the first line of the
`/api/sessions/{id}/inject-prefetch` docstring (`app/routers/sessions.py:1366`).
The tool description a producer actually reads says the signal is queued "at no
cost" (`agent_mcp/ambient.py:65`) and the module docstring says "cheap prefetch
drain (no SDK turn)" (`:11`); this page attributed the wording to the wrong file
until 2026-10-12. Either way it is a fossil of the claude-agent-sdk
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
  → resolves to nothing → POST /api/next-session-note instead, at ANY priority
      (the file-backed channel, #1516 — see the note under Core Concept)
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
  → drain_next_session_notes() (#1516) — a global file, and consuming is the same
      act as delivering, so since #2598 the drain asks the reader first:
      `is_user_session` over this turn's session JSON. A worker/autonomy turn
      skips it, and the note waits for a turn that can reach a human
  → _format_context renders <ambient-signals> FIRST inside <context>
  → "<context>…</context>" + "\n\n" + the user's text
  → model sees: <context>...<ambient-signals>...</ambient-signals>...</context>
                \n\n<user text>
```

Two ordering rules in there are load-bearing. The drain runs **above** the
`MIN_MESSAGE_LEN` guard, so "ok" still surfaces a queued signal — a short
message must not suppress an injection the producer already decided was worth
showing; the guard's `not ambient_entries and not notes` clause covers both
drains. And it runs on the **caller's** thread, not in the worker thread the
budgeted search phase uses (`_prefetch_prepare` vs `_prefetch_run`): the queue
is mutated by producers on the event loop, so draining it off-thread would race
them, and the next-session channel holds a note for exactly one turn — drained
twice, it delivers twice.

### Rendered Context Block

```xml
<ambient-signals>
Background producers queued these signals for you. The user did NOT ask —
reference them only if naturally relevant to what they're saying now.
- **[autonomy:task-42]** 3 new emails worth reviewing
  _server clock when queued: 2026-09-16 09:14 PDT_
  > From: client@foo.com ("contract update"),...
- **[pipeline:research-117]** Background research on quantum compilers finished
  _server clock when queued: 2026-09-16 09:15 PDT_
  > Key finding: tket compiler beats qiskit on circuits >10q by 14%...
</ambient-signals>
```

The stamp under each summary is the **server's** reading of the instant the entry
was queued, never a date the producer wrote (#1197: an autotriage brief invented
`2026-09-17` in its own header, queried the calendar for that day, and reported a
birthday as "(no events)"). `ambient_clock_stamp` (`app/sessions_io.py:558`) is
the only formatter either ambient path uses, and it renders nothing for an unset
instant — no measurement, no stamp, rather than a 1969 date offered as evidence.

### Key Properties

- **Dedup by `dedup_key`** — same key re-firing replaces the previous unsent entry (newest wins). Default `dedup_key = source`.
- **TTL** — entries carry `expires_at`, from `ttl_seconds` (default 3600, and the ambient tier only). Expiry is evaluated on **both** doors, so reclaiming a signal never depends on a turn arriving for the session it was queued against. `enqueue_ambient_prefetch` (`app/sessions_io.py:633`) purges that session's dead entries before it stores the new one, names each casualty's `source` in `dropped`, refuses a signal that was already expired at the moment it was written, and releases the session's key whenever the purge empties it; `drain_ambient_prefetch` (`app/sessions_io.py:704`) keeps its own filter as defence in depth, because an entry can pass its deadline while it waits rather than at write — and names each casualty in the log, since a path that drops a signal without a trace is how #910 stayed invisible. What `expires_at == 0.0` means is **no deadline**, not the epoch — every caller that omits `ttl_seconds` gets that value, so a purge that read it as "already expired" would silently delete nearly every ambient signal in the system and still report a clean queue. `tests/test_ambient_prefetch_retention.py` pins both halves of that boundary: expiry-on-write, key release through every door (a dead signal into an empty session, purge-to-empty, drain-to-empty, cap eviction), the drain's overflow re-insert and its logged drop of a signal that expired while it waited, that drain driven through its only production caller `prefetch._prefetch_prepare` rather than called directly, the `DELETE` and `/inject-prefetch` routes, and this page's own `app/sessions_io.py:<line>` citations (`test_the_pages_line_citations_point_at_the_functions_they_name`), since a stale pointer reads as a discovered defect. What the page said before, and what was wrong with it, is in the 2026-09-20 review-log entry below — and the three sentences that read as current but described the old tree are pinned absent by `test_the_architecture_page_describes_the_retention_that_shipped`, which is the machine-readable version of that entry.
- **Two caps, and they answer different questions.** `AMBIENT_PREFETCH_CAP = 5`
  bounds what a session may *hold*: the oldest is evicted on enqueue.
  `AMBIENT_PREFETCH_DRAIN_MAX = 3` bounds what one turn may *see*: the drain
  takes the three newest and puts the remainder back. A burst is therefore
  spread across turns rather than dropped on the floor, and the newest signal
  is never the one that waits.
- **`content` is truncated at 800 chars** when rendered, with an ellipsis. The
  summary line has its own, tighter cap the page did not used to state:
  `AMBIENT_SUMMARY_MAX = 240` (`app/prefetch.py:145`), marked with
  ` [... truncated]`, applied to every ambient line — and the next-session
  channel caps its own fields at 720 / 2000 before a note is ever stored
  (`NEXT_SESSION_SUMMARY_MAX` / `NEXT_SESSION_CONTENT_MAX`,
  `app/next_session_notes.py:69-76`).
- **A summary carrying a date that contradicts itself is refused, not rendered.**
  `app/date_fidelity.refusal_detail` runs before the target check on all three
  doors — `/inject`, `/inject-prefetch` and `/api/next-session-note` — and
  answers 400 so a producer fixes its payload rather than blaming a missing
  session (#1149).
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

      <ambient priority="notable" source="autonomy:task-42" session_id="20260419_..."
               server_clock="2026-09-16 09:14 PDT">
      {summary + content}
      </ambient>

      The server clock when this was queued was **2026-09-16 09:14 PDT** — that is
      a measured reading from this machine, not a date the producer wrote. Treat
      it as the current day: anything the signal above says about "today", a date,
      or a window has to agree with it or be reported as unverified (#1197).
      This is a background signal from `autonomy:task-42`. You were not asked a
      question — consider whether to mention this to the user. If it is not worth
      interrupting them, call
      `ambient_decide(session_id="...",surface=false,reasoning="...")` and stop.
      If it is worth surfacing, reply briefly and naturally — the user will see
      your message as a normal assistant turn.

      # One measured instant feeds both the envelope attribute and the turn's
      # queue position, so the clock shown is the clock it was enqueued at.

  → Enqueued on the SessionQueue's ambient tier (from #296): dedup_key collapses
    a queued duplicate, and AMBIENT_QUEUE_CAP = 3 drops the oldest beyond it
  → Queue consumer picks it up when no user turn is pending or running

                    ... turn runs ...

Agent either:
  A) Replies normally
     → Assistant message lands, tagged source="ambient"
  B) Calls ambient_decide(session_id,surface=false,reasoning="...")
     → Server stores decision in _ambient_decisions[sid], stamped with the
       deciding turn's turn_id
     → cancel_event.set() 0.5s LATER, via _deferred_ambient_cancel
     → On the harness `result` event: take_ambient_decision(sid, turn_id) hands
       the decision back to the turn that recorded it and to nobody else
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

**And only the turn that decided gets to be silenced by it.** The read is
`take_ambient_decision(sid, turn_id)` and it answers only the turn whose id the
entry carries, because the alternative was #2599: the pop is that one branch, so
a turn ending any other way leaves its entry standing, and with one slot per
session the next ambient turn is the one that finds it. Keyed by session alone,
that orphan replaced the successor's reply with the dead turn's breadcrumb and
its `ambient_silent` — a reply the user never got, attributed to a decision
nobody made for it. A late entry is therefore discarded, removed so no third
turn can spend it either, and logged naming both the recording and the asking
turn id.

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
`source=None` default means "ambient only" (`app/sessions_io.py:1109`), which
is what `drain_pending=true` on that route relies on. Draining the user tier is
the explicit `source="all"` sentinel — both tiers, ambient then user, summed
count — and `DELETE /api/sessions/{id}` is its only caller
(`app/routers/sessions.py:1141-1195`). That route wipes the session outright, so a
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
| `_ambient_decisions` | `dict[session_id, dict]` | What `ambient_decide` recorded, stamped with the deciding turn's `turn_id` and handed back by `take_ambient_decision(session_id, turn_id)` to that turn alone — the pop is the one place the slot is read, so an entry left behind by a turn that never reached its `result` event is discarded and logged rather than spent on the session's next ambient turn (#2599) |
| `SessionQueue.pending_ambient` | `deque[SessionTurn]` | Mechanism 2's tier, popped only when `pending_user` is empty |
| `AMBIENT_PREFETCH_CAP` / `AMBIENT_PREFETCH_DRAIN_MAX` | `5` / `3` | Hold cap and per-turn cap |
| `AMBIENT_QUEUE_CAP` | `3` | Queued ambient *turns* per session; oldest evicted |
| *not here:* `app/next_session_notes.py` | file-backed, cap 5, TTL 24 h | The third store this subsystem writes to, and the only one that survives a restart: `~/lloyd-data/next-session-notes.json` (#1516). Its own caps (`NEXT_SESSION_NOTE_CAP`, `..._SUMMARY_MAX`, `..._CONTENT_MAX`) and its own no-deadline sentinel rule are declared and tested there — a second copy of either, here or in the queue module, is the defect |

### Endpoints (`app/routers/sessions.py`)

| Route | Does |
|---|---|
| `POST /api/sessions/{id}/inject-prefetch` | Mechanism 1. 404 on an unknown session; 409 on a non-user one (#910); 400 without `source` and `summary`, or on a date the summary contradicts (#1149) |
| `POST /api/sessions/{id}/inject` | Mechanism 2. 409 on a non-user session, 404 on an unknown one, and the same date guard first |
| `POST /api/next-session-note` | The third door (#1516): the same payload minus the session id, so it is the only one a 03:00 producer can use. 400 on the date guard, then 400 without `source`/`summary`; writes through `write_next_session_note`. No 409 is possible — nothing names a session |
| `GET /api/next-session-notes` | The read door for that same store (#2601): a debug peek that does not drain (`peek_next_session_notes`), in the entry shape of the prefetch peek below so the two queues read side by side. Expired notes are listed with `expires_at` in the past rather than hidden — expiry is logged only inside a drain, so a note no turn ever reached has no other record, and hiding it would collapse "expired" into "never written" |
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

The invariant is enforced at **both doors**, and a rule held only at the door is
held by nothing where the consumer reads — the shape the next-session channel had
until #2598. Its file is global, so the drain in `prefetch._prefetch_prepare` ran
for whichever session turned up first, worker included, since worker turns arrive
through the chat path for the Inner-Voice reason this section already names, and a
worker turn between the overnight write and the first human turn read the brief
into a transcript nobody reads and left the store empty. The consumer end now
applies the same deny-list the doors apply: `sessions_io.is_user_session` over the
draining turn's own session JSON, read once in `_prefetch_prepare` beside the
plan-mode flag it also answers. Being a deny-list is the part that keeps delivery
working — a session with no JSON on disk, a platform the list has never heard of,
or no session id at all still drains and still receives the note; `platform`
values a human reads (`mission-control`, `browser`) are not in the list, and
widening it is a separate decision. Pinned by `tests/test_next_session_notes.py`,
which drives `_prefetch_prepare` rather than the drain, because the drain's only
production caller is the seam a turn crosses. One related
nuance so the two predicates are not collapsed into one: `is_user_session` answers
"can a signal reach a human through this session", and since #1064 the *listings*
and exports ask a stricter question — `is_conversation_session`, an allow-list of
`INTERACTIVE_PLATFORMS` where an unknown platform is excluded — so a session can
legitimately be missing from Mission Control and still be a valid ambient target.

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
on this queue. A third occupant of the tier is not a producer at all: the
memory-flush turn (`build_flush_turn`, off by default at
`memory_flush.enabled: false`) enqueues with `source="ambient"` to save durable
facts before compaction, surfaces nothing by design, and is named back to Inner
Voice as `memory_flush` so the observer skips it.

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
config.yaml and the live override — `TOOL_OVERRIDES_PATH`, which is
`$LLOYD_DATA/data/tool_overrides.yaml` (`app/paths.py:119`:
`~/lloyd-data/data/tool_overrides.yaml`, not a file in the checkout;
`data/**` under the repository is build input and is denied to the self-mod
loop) — and every tool is advertised; if
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
- 2026-10-12 — reviewed against `1bf72f2d` (status: **stale**, corrected). Both
  mechanisms still ship exactly as described; what moved is around them. Added:
  the third door (#1516, `POST /api/next-session-note`, `app/next_session_notes.py`
  — cap 5, TTL 24 h, drained by the same `_prefetch_prepare`) and the fact that
  `session_inject_context` branches on resolution *before* it branches on
  `priority`, which the Core Concept's one-sentence invariant had gone quiet
  about; the `server_clock` the envelope and every rendered ambient line now
  carry (#1197, `ambient_clock_stamp`); the `AMBIENT_SUMMARY_MAX = 240` cap beside
  the 800-char one; the #1149 date guard on all three doors; the memory-flush
  turn as a third occupant of the ambient tier; and `is_conversation_session` as
  the listing predicate that is deliberately *not* the delivery predicate.
  Corrected: two drifted line pointers `drain_pending` is
  `app/sessions_io.py:1109`, not `1030-1074`; the `DELETE` route is
  `app/routers/sessions.py:1141-1195`, not `811-840`), the "zero SDK cost"
  attribution — that phrase is the `/inject-prefetch` docstring, not the tool
  description, which says "at no cost" — the override path, which is
  `$LLOYD_DATA/data/tool_overrides.yaml` rather than a file in the checkout, and
  the Status paragraph's "(nine cases)" and "63 tests", replaced with the
  per-file `--collect-only -q` collected counts the Status paragraph now carries.
  What that pass published instead — "17 / 5 / 16 / 6 = 44 green across the four
  files that pin this page, 73 in `test_prefetch.py`" — was a four-file selection
  reported as though it were the pinning surface, over a wrong number: the same
  command collects 93 in `tests/test_prefetch.py`, and the fifth file that reads
  this page's own sentences was red on the tree this page landed into — node
  `test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree[ambient-context-injection.md]`
  of `tests/test_architecture_coverage_doc_claims.py`, whose whole file the pass
  never ran. A count is now worth writing only with
  the command that produces it beside it and the node id of anything excluded
  named, which is the rule #2606 filed and
  `test_the_ambient_page_states_the_command_its_collected_counts_came_from` holds.
  Filed from the same
  pass: **#2598** (the notes drain has no user-session check, so a worker's turn
  consumes the note meant for the human), **#2600** (`notable`/`urgent` with no
  resolvable session are downgraded to the passive note and reported `ok=true`),
  **#2599** (`_ambient_decisions` is keyed by session while the deferred 0.5 s
  cancel lets a decide call land on the next ambient turn), and **#2601**
  (`peek_next_session_notes` promises a debug endpoint that does not exist).
- 2026-10-12 — **#2598 landed**: the notes drain stopped consuming the channel for
  a machine session. `prefetch._prefetch_prepare` now reads the draining turn's
  session JSON once — the same read the plan-mode fallback answers, which is why
  the old `_read_plan_mode` is `_read_session_meta` plus `_plan_mode_from_meta` —
  and calls `sessions_io.is_user_session` on it before `drain_next_session_notes`,
  so a `worker` or `autonomy` turn leaves the note for the next readable turn
  instead of deleting it. No second rule was written: inside the code that decides
  the drain there is no platform set and no platform name as a literal, one test
  asserts that absence, and another proves the decision goes through the module
  attribute by monkeypatching `is_user_session` and watching the drain stop. The
  fail-open cases are the design, not an oversight: an empty dict (no session id,
  no file, unreadable bytes) reads as a reader, so `prefetch_context`'s sessionless
  callers and every client the list has never heard of keep getting notes —
  including the eval harnesses, all of which pass `session_id=None` explicitly or
  by default, and the memory eval's sleep-note arm, which drains directly and is
  untouched. Blast radius from `graph_affected` over `_prefetch_prepare`: depth 1
  is `prefetch_context` and `prefetch_context_async` in `app/prefetch.py` plus four
  test nodes; depth 2 adds `post_message_stream` (`app/routers/messages.py`),
  `voice_inject` (`app/routers/voice.py`), the eval entry points and
  `tests/test_prefetch.py`. Two things the change cannot show and the item stays
  open on: whether real traffic delivers now — the live store has never been
  written outside a test and a consumed note leaves no trace, so only a note
  surviving several nights of machine turns and arriving in a chat prompt is
  evidence — and whether `browser` counts as a reader, which is a ruling on the
  platform list, not on this gate.
- 2026-10-12 — **#2599 landed**: an ambient `surface=false` silences the turn that
  wrote it and no other. `post_ambient_decide` stamps the stored decision with the
  deciding turn's `turn_id` and answers with that same recorded value;
  `take_ambient_decision(session_id, turn_id)` hands it back only on a match, and on a
  mismatch removes the entry — so no third turn can spend it either — and logs a
  warning naming the session and both turn ids. Keying the slot by session was only
  half of it; the read sits inside `_run_turn`'s harness `result` branch rather than a
  cleanup path, so an ambient turn ending any other way left its `surface=false`
  standing and the *next* ambient turn of that session persisted the dead turn's muted
  breadcrumb in place of its own reply — no race required. The deferred cancel reaches
  the same slot from the other side, a decide call landing up to 0.5 s after its own
  turn stopped being current. Both halves are pinned by
  `tests/test_ambient_decision_turn_scope.py`, which crosses the router/store seam with
  the real route and the real dict (a mocked store would pin nothing), and the comment
  above `_ambient_decisions` no longer promises the `finally` clause that made the leak
  look impossible to anyone who read it. Ambient traffic here is thin enough that no
  crossed silencing had left a fingerprint, which is a coverage gap, not a clean bill;
  the scan that would settle it is recorded as owed on #2599.

- 2026-10-12 — **#2601 landed**: the notes channel got the read door its peek had
  been promising since #1516. `GET /api/next-session-notes` serves
  `peek_next_session_notes` unchanged — `{depth, notes[]}` in the prefetch peek's
  entry shape, consuming nothing — and the peek's docstring now names that route
  instead of "the debug endpoint", the pointer a reader followed to nowhere. The gap
  was never the file; it was the window. Delivery consumes, and expiry is logged only
  inside a drain, so a note no turn ever reached left no record of having existed, and
  the only way to ask "did tonight's pass write anything, and is it still here?" was to
  read the store file by hand between turns — on a box that runs background jobs
  continuously, a race with the turn that takes the note. The peek filters nothing, so
  an `expires_at` in the past is what separates "expired" from "never written", and the
  entry keys are pinned against the *other* route's live response rather than a literal
  list, because the claim worth holding is that the two queues read alike. Both doors
  of the channel now sit adjacent in the Endpoints table above. Two things this round
  could not settle and the item stays open on: whether a TTL death with no drain should
  log at all, and one real nightly pass checked through the route rather than the file.
  The #2598 motive the item carried — a worker turn eating the note — was already closed
  by the gate two entries above and is not what this door is for.
