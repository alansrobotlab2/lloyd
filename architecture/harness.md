---
segment: architecture
tags: [architecture, lloyd, harness]
type: reference
status: implemented
date: 2026-09-11
---

# The agent harness

`app/harness/` is the in-process agent loop. It replaced the Claude Agent SDK
on 2026-09: one async generator, `run_query(messages, options)`, streams a
turn against a local OpenAI-compatible engine, dispatches tool calls through
the MCP aggregator, and yields normalised events that the chat router, the
worker pool and the autonomy scheduler all consume the same way.

This doc is the map and, since 2026-09-25, the long version too: `CLAUDE.md`
§ "Agent Harness" keeps one line per rule and points here, and the
mechanism, the incidents and the measurements are under "The long versions"
below.

## Modules

| Module | Owns |
|---|---|
| `loop.py` | the iteration loop: stream → parse → tool dispatch → append → repeat; history shaping (`_assistant_message_for_history`, `_commit_tool_calls`, `_prune_reasoning`) |
| `turn_state.py` | `TurnState` (the turn's locals) and `Iteration` (one pass's) — the state `run_query`'s phases share (P13.1) |
| `client.py` | the httpx SSE stream to `/v1/chat/completions`; `read=None`, so stalls are bounded by `stream_chunk_timeout_s`, never time-to-first-byte |
| `options.py` | `RunOptions` — every knob a caller can set; the router and the worker sources build one per turn |
| `events.py` | the `NormalizedEvent` constructors: `system`, `text_delta`, `thinking_delta`, `thinking_done`, `tool_call`, `tool_result`, `assistant_message`, `result`, `stream_raw` |
| `mcp_pool.py` | `MCPPool`: discovery with annotations, dispatch, the process-wide pool cache (`get_or_open_pool`). The HTTP path opens a session per call and is unlocked — the 2026-07-28 core is stateless, so a held connection buys nothing; only the stdio path shares one `ClientSession` and takes a per-server lock. Raises `ToolDiscoveryError` on an empty pool. The catalog is one immutable value swapped whole, refreshed at turn start on a TTL (P12) |
| `tool_schema.py` | MCP `inputSchema` → OpenAI tool schema; bare-name advertise with the legacy `mcp__server__tool` form still parsed for replay; `add_summary_param` |
| `tool_search.py`, `tool_search_cache.py` | progressive disclosure: a baseline toolset plus `ToolSearch` when the pool exceeds `threshold_tools` |
| `hooks.py` | `HookRegistry` — pre/post tool-use callbacks (Inner Voice, safety, skill dispatch, the #534 grant gate) |
| `safety.py` | the destructive-Bash patterns; the only thing between an unattended turn and `rm -rf` |
| `policy.py` | authority grants (#534): scopes, tiers, `current_scope` / `current_effect_scope` contextvars |
| `skill_dispatch.py` | `/skill` invocation from a turn |
| `microcompact.py` | intra-turn compaction: clears the oldest tool results, spilling each to disk first so the marker can name the file. Budget-driven since 2026-09-05 — it fires on the prompt's fraction of the truncation threshold, not on a tool count; the old count rule cleared 93 of 97 results on a review using 17% of its window and is now disabled outright |
| `tool_result_spill.py` | tool results over 50k chars spilled to `sessions/<sid>.tool-results/` and replaced inline by a `<persisted-output>` marker naming the file |
| `finalizer.py` | one extra completion under a JSON schema after the turn (`final_schema`) |
| `run_state.py` | schema-validated execution state for long *worker* turns (#529): a step's patch rides the `final_schema` path, `Σ` is merged with null-deletion, the reasoning is discarded, and every applied and rejected patch goes to `state-trace.ndjson`. Imported by `workers/sources/_common.py`, not by the router — and explicitly **not** for the interactive loop, where rewriting state every step is directly opposed to keeping position 0 stable |
| `errors.py` | `HarnessError` (base), `ParseError`, `ToolDispatchError`, `MaxTurnsExceeded`, `ContextOverflowError`, `StreamStalledError`, `ToolDiscoveryError` |

## Invariants worth knowing before touching it

- **Position 0 is frozen for the turn.** The system prompt is built once and
  inserted at index 0; the loop only appends. That keeps the prompt prefix
  KV-cached across every iteration. Anything that must change mid-turn is
  *appended* (`state_anchor`, `notification_drain`), never re-rendered.
- **Preserved thinking rides under two keys.** vLLM reads `reasoning`,
  llama.cpp reads `reasoning_content`; each ignores the other silently.
  `_prune_reasoning` drops the pair together. Window:
  `harness.preserve_thinking_iterations`, applied at turn entry
  (`_cap_history_reasoning`) and — only when the meter says the prompt is over
  target — again by relief rung 2, down to
  `context_relief.reasoning_keep_under_pressure`. Never per iteration (#520).
- **An empty tool pool is the worst failure.** With no `tools` array vLLM
  never engages the tool parser and the model narrates tool calls as prose.
  `MCPPool.open()` raises on empty discovery so `get_or_open_pool` evicts and
  re-discovers; `run_query` refuses a toolless turn.
- **A tool call's transport budget must sit above the call's own ceiling, and
  a failed call is only re-sent when the server says that is safe.**
  `CALL_TIMEOUT_SECONDS` is 660 s and `HTTP_READ_TIMEOUT_SECONDS` is 690;
  until 2026-09-11 the pool built its HTTP client on the SDK default
  (`read=300`), so `automod_gate` — 7 to 12 minutes with its review rung —
  died at the transport and the pool's retry started a *second* gate of the
  same round while the first was still grading. 18 rounds, 17 aborts, 0
  landings in one day. `_retry_safe` now re-sends only a call the server
  annotated `readOnlyHint` or `idempotentHint`: a transport error says
  nothing about whether the server ran the call, and for a long one it almost
  certainly did.
- **A context overflow is recovered, not raised.** vLLM's 400 for a prompt
  past the window comes back as `ContextOverflowError`, distinct from every
  other `HTTPStatusError` so the loop recovers from this one failure only:
  it truncates the largest tool results in place, appends a synthetic note
  saying so, and retries the same turn, bounded by
  `max_context_overflow_recoveries` so truncation that cannot free enough
  budget does not spin.
- **Every tool is advertised under its bare name**, built-ins included —
  `Bash`, `Read`, `Write`, `Edit`, `Grep`, `Glob`, `Task`. The
  `mcp__<server>__<tool>` prefix was an SDK artifact; `disallowed_tools`
  still blocks both forms, and old session JSON still replays. The cost of
  bare names is that `build_tool_list` must raise on a cross-server
  collision rather than silently shadow one. A tool whose name starts with
  `_` is never advertised at all — the harness can still dispatch it
  directly, which is how `_BackgroundTaskDrain` works.
- **Parallel dispatch is read-only-only and ships off.** A batch overlaps only
  when every call carries `readOnlyHint`; one `Bash` or mutating tool makes
  the batch sequential. History is written in wire order regardless.
- **Every tool carries a `summary` parameter** — except one that already
  declares its own (`session_inject_context`, where it is a real required
  argument), which is why `add_summary_param` returns the set it actually
  injected into rather than letting the loop strip by name. It is lifted off
  `_args_dict` before dispatch and kept in the replayed `arguments`, because
  the replayed call is the model's own most recent example of that tool.
- **Each reasoning phase is its own `role="thinking"` row** in the session
  JSON (`app/routers/_messages_thinking.py`). The role is what keeps it out
  of every transcript producer; do not change it.
- **The finalizer sends the identical `tools` array with
  `tool_choice: "none"`.** Dropping it diverges the rendered prompt at token
  41 and costs a full re-prefill (measured 1.2 s vs 25 s). It runs only when
  the turn stopped on its own.

## Who calls it

| Caller | Path | Recorded | Observed |
|---|---|---|---|
| chat (`/api/message/stream`) | `app/routers/messages.py` | session JSON + event log | Inner Voice per session |
| worker, session-backed | `workers/sources/_common.run_prompt_in_session` → the chat endpoint | yes | per source (`inner_voice`) |
| worker, direct | `_common.run_prompt_on_primary` → `run_query` via `app/run_recorder.py` | yes | never |
| autonomy task | `autonomy.run_task` → `run_query` via the recorder | yes | per task |
| `Task` subagent | `agent_mcp/builtin_task.py`, inside the aggregator process | parent's turn | parent's observer |

See [[background-runs]] for the recorded/observed split and
[[workers]] for the queue.

## Which engine a turn reaches

`RunOptions` carries a model *alias*, not an endpoint, and
`config.resolve_model_alias` turns one into the other. Two things bend that
resolution, and neither is visible from the call site:

- **`secondary_enabled: false` rewrites `secondary` to `primary`** for every
  caller, logged once per name because it is otherwise undetectable. Inner
  Voice's config said `model: secondary` for four months and ran on the
  primary throughout — [[infrastructure]] § Model slots has the flag and the
  4B episode that followed when it was flipped.
- **Worker turns name `model="primary"` outright** in
  `workers/sources/_common.py`. The one exception is the scheduled-task
  source: `autonomy.run_task` resolves the task file's own `model:`
  frontmatter, which is the only path by which anything reaches the secondary
  deliberately.

The Inner Voice observer runs on the primary (`inner_voice.model`, pinned
there after that episode) at **the priority of the turn it watches** —
`attach_observer_for_turn` passes `options.priority` straight through, so a
chat's observer runs at 0 instead of queueing at the configured `1` behind
every worker iteration. vLLM orders equal priorities by arrival, so an
observer call still cannot preempt the turn that spawned it.

A `Task` subagent resolves nothing of its own — it inherits the calling
turn's model and base URL, below.

## Subagents

`Task` runs a nested `run_query` inside lloyd-mcp with its own `RunOptions`
built from `subagents.<type>` in `config.yaml`. It inherits the calling
turn's model and base URL through the MCP request `_meta`
(`lloyd/model`, `lloyd/base_url`); recursion is capped at one level. Every
result carries a `task_id`; passing it back with a follow-up prompt resumes
the stored conversation (8 tasks, 30 minutes, 3M chars, process-scoped).
`agent_mcp/_subagent_registry.py` opens a dashboard row before the run
starts and each exit path closes it with its own status. A `task:*` run
writes no session JSON of its own — it leaves a spill directory under
`sessions/` and its conversation lives in that process-scoped registry, so
after an aggregator restart every `task_id` honestly reads `unknown or
evicted`. What it wrote to *disk* is attributed to the parent's turn in the
change ledger, which is where to look for it.

## Config

`harness:` in `config.yaml` — `stream_chunk_timeout_seconds`,
`todo_anchor_interval_iterations`, `preserve_thinking_iterations`,
`tool_search`, `parallel_tool_calls`, `finalizer`, `thinking_trace`,
`background_recording`, `prefix_miss`, `tool_call_summaries`, `edit_gates`,
`edit_diagnostics`, `change_ledger`, `effect_ledger`. The *state* anchor is a
`RunOptions` callback the router (or `autonomy._build_task_anchor`) builds per
turn; each of its parts has an off switch read at the moment it would speak —
`budget_anchor.enabled` (both budget clocks, `app/deadline_anchor.py`),
`todo_anchor.enabled`, `context_anchor.enabled` — and an absent key means on
(#769; only `context_anchor` is in config.yaml today). Anchor messages are not
persisted, so the loop records each one it appends as a `harness.anchor_fired`
event in `event_logs/<session_id>.events.jsonl`, naming the anchor, its level,
the iteration and the turn: `grep -h '"harness.anchor_fired"'
event_logs/*.events.jsonl` counts them.
`compaction:` owns the between-turn and in-turn compaction walls.
`subagents:` owns the `Task` profiles.

`tool_search` is the one block here the UI can rewrite: the Tools page
persists to `data/tool_overrides.yaml`, which is merged over config.yaml at
load and shadows this key wholesale. `app/config.py` warns when the two
disagree, because config.yaml is the state a fresh clone boots into and
`enabled` decides whether the model is handed the whole catalog or a
baseline plus `ToolSearch`.

## Tests that pin the above

`tests/test_preserved_thinking.py`, `tests/test_mcp_pool_discovery_failure.py`,
`tests/test_mcp_pool_retry_policy.py`, `tests/test_tool_call_summaries.py`,
`tests/test_thinking_trace_transcripts.py`, `tests/test_task_registry_wiring.py`,
`tests/test_task_resume.py`, `tests/test_transcript_entries.py`,
`tests/test_run_state.py`.

The harness also carries its own unit suite *inside the package*, at
`app/harness/tests/` — `test_loop_inject_ordering.py`,
`test_parallel_dispatch.py`, `test_dispatch_split.py`, `test_stream_stall.py`,
`test_finalizer.py`, `test_preserve_thinking.py`, `test_state_anchor.py`,
`test_tool_search.py`, `test_loop_tool_search.py`, `test_harness_unit.py`.
Both directories run under `pytest`; a change to `loop.py` usually breaks
something in the second one first.

## The long versions (moved from CLAUDE.md, 2026-09-25)

CLAUDE.md keeps one line and a pointer per rule; the mechanism, the incidents
and the measurements behind each one live here.

### The event stream

`run_query(messages: list[dict], options: RunOptions) -> AsyncIterator[NormalizedEvent]`.
The constructors in `app/harness/events.py` are the authority, and these keys
are not the OpenAI wire names:

- `system` — `{type, session_id, model}` — turn opened.
- `text_delta` — `{type, text}` — streaming text chunk.
- `thinking_delta` — `{type, text}` — reasoning content chunk.
- `thinking_done` — `{type, text, duration_ms}` — reasoning phase complete.
  `duration_ms` spans the first reasoning chunk to the last, not the
  iteration's wall clock, which also covers prefill and the answer written
  afterwards. It reaches the chat's collapsed thinking panel as `reasoning_ms`
  on that phase's own `role="thinking"` row (see "The thinking trace" below),
  so the header reads the same on reload as it did live — the event lands
  *after* that iteration's text, so the browser opens the row on the first
  *delta* and measures the timestamps itself until the real number arrives.
- `tool_call` — `{type, call_id, name, args_json, args_dict, summary}` — tool
  invocation. `summary` is the model's own one-liner for the transcript; it is
  absent from `args_dict` and kept in `args_json` (see "Tool-call summaries").
- `tool_result` — `{type, call_id, name, content, is_error}` (+ `duration_ms`,
  `handshake_ms`, `error_class` when known, P11).
- `assistant_message` — `{type, text, tool_calls, thinking, usage,
  duration_ms, iteration, finish_reason, ttft_ms, request_ms, cache_ratio}` —
  one agent-loop iteration. `usage` and `duration_ms` are per-iteration, not
  per-turn.
- `result` — `{type, stop_reason, usage, num_turns, duration_ms,
  response_text}` — turn complete.
- `stream_raw` — `{type, raw, error}` — raw SSE line on parse failure.

### The position-0 rule (mid-turn state)

The system prompt is built once per turn and inserted at index 0; the loop
only ever appends. That keeps the whole prompt prefix KV-cached across every
iteration, so a 160k-token turn re-prefills nothing. The cost is that anything
rendered into the system prompt — `<active_todos>`, the plan, the goal — is
frozen at turn start. **Never refresh the system prompt mid-turn**; re-anchor
by appending instead (`RunOptions.state_anchor`, mirroring
`notification_drain`). A turn that creates its own todo list would otherwise
never see it again — see `app/routers/messages.py::_build_state_anchor`.
Across turns, P1's `harness.prompt_layout` switches move that state and memory
edits to the user message's tail (§P1 below).

### Preserved thinking and the two reasoning keys

Assistant messages carry their reasoning back into history under **both**
`reasoning` and `reasoning_content`, bounded to
`harness.preserve_thinking_iterations` recent iterations. Qwen3.8-Flash-Next
renders it into each prior turn's `<think>` block; dropping it showed the model
turn after turn in which it had apparently thought nothing. A/B it with
`eval/run_preserve_thinking_eval.py` before changing the window.

The two spellings are not redundant — the engines disagree, and each one
ignores the other's field *silently*:

| Engine | Reads | Ignores |
|---|---|---|
| vLLM 0.28 (primary) | `reasoning` | `reasoning_content` |
| llama.cpp (secondary, Qwen3.6) | `reasoning_content` | `reasoning` |

vLLM accepts both on the wire but only populates the template from
`reasoning` (`entrypoints/chat_utils.py:2000`); Qwen3.6's own
`chat_template.jinja:91` reads `reasoning_content` and never looks at
`reasoning`. Sending one spelling preserves thinking on one engine and quietly
discards it on the other, which is the exact failure this mechanism exists to
prevent. The builder is `_assistant_message_for_history`. `_prune_reasoning`
must drop the pair together or the token bound stops bounding anything.
`tests/test_preserved_thinking.py` pins both halves; llama.cpp's
`POST /apply-template` will show you the rendered prompt if you need to
re-verify.

Scope is **intra-turn only**: history is rebuilt from the session JSON on each
user turn (`load_and_compact_session`), which does not carry per-iteration
reasoning, so the window resets at every turn boundary. That is where the cost
was anyway — the motivating turn ran 52 iterations inside one turn.

### An empty tool pool is the worst failure in the system

`client.stream_chat` omits `tools` from the request when the list is falsy, so
vLLM never engages the `qwen3_xml` tool parser. The model still reads its
whole toolbox in the system prompt, reasons its way to "call Bash", and then
has no channel to emit a tool call on. What comes out is an empty message, or
the call written as prose (`{"name":"Bash","input":...}` — an Anthropic shape
that appears nowhere in this repo), or invented tool *output*. Nothing in the
stream says "no tools"; it reads exactly like the model having forgotten how
to use them, and Inner Voice's only lever — injecting more text — cannot help,
because the intent was never missing, the capability was.

`MCPPool.open()` used to log a warning, `continue`, and set `_opened = True`
even when its only server failed discovery. `get_or_open_pool` caches
process-wide and short-circuits on `_opened`, so one transient error (the
aggregator restarting) pinned an empty pool for the life of the backend.
`open()` now raises `ToolDiscoveryError` when discovery yields nothing, which
makes `get_or_open_pool`'s **existing** eviction path fire so the next caller
re-discovers — the recovery already existed, nothing ever failed loudly enough
to trigger it. A *partial* failure still degrades gracefully; that is what the
`continue` is for. `run_query` refuses a turn whose pool advertised nothing.

Two things this cost on 2026-09-06, both invisible as tool failures: a
30-minute chat where Lloyd narrated `sqlite3` commands instead of running
them, and four `domain-research` jobs killed at the 600s cap — a toolless
research job cannot research, so it spins until the timer. The same job
finished in 17s once tools came back. `tests/test_mcp_pool_discovery_failure.py`
pins it. The tell in the log is `no server claims tool '_BackgroundTaskDrain'`
firing right after `mcp_pool: failed to discover` — the drain shares the pool,
so it is the cheapest early warning that every turn has gone toolless.

### Stream stalls

`harness.stream_chunk_timeout_seconds` bounds the gap *between* SSE lines once
the engine has started producing, raising `StreamStalledError`. It
deliberately does **not** bound time-to-first-line: prefill emits no bytes,
and the secondary runs llama.cpp with `--parallel 1`, so a queued request
legitimately sits silent for as long as the one ahead of it.
`client.stream_chat` sets httpx `read=None`, so without this a wedged engine
mid-generation hangs the turn until the client gives up. The key existed from
the start and was read by nothing until 2026-09-06. A broken stream is retried
once while no tool-call delta arrived, else ends `stream_error` (§D7,
`harness.stream_retry`). A turn that raises or is cancelled still books its
tokens, once, from the running totals (§D12).

### Concurrent tool dispatch (read-only batches only)

`harness.parallel_tool_calls.enabled` lets one iteration's tool calls overlap.
A batch qualifies only when **every** call in it is annotated `readOnlyHint`
(or is a parse error, or ToolSearch). One `Bash`, `Edit`, `Write` or mutating
MCP tool makes the whole batch sequential, byte-for-byte the old path.

Read-only-only is not caution, it is the only classification available that
is not a guess. `mcp_pool._list_tools` carries `annotations` through, so
qualification comes from the server's own hint rather than a second private
list of names — the pattern `agent_mcp/annotations.py` was written to replace.
A server that sets no hints qualifies nothing, which is that file's contract.
`Bash(cat …)` serialises its batch: classifying shell commands as read-only is
the guessing game the safety hook deliberately refuses to play.

Since P13.3 this is ONE path for every batch (`loop._dispatch_batch`),
concurrency 1 unless the batch qualifies; at 1 it is the old sequential loop
exactly (§P13.1-3). Three phases, and each one exists for a reason:

- **Phase 1 runs in wire order and stays sequential.** `_pre_dispatch` covers
  the parse error, the disabled-tool gate, the ToolSearch intercept (which
  mutates the shared `LoadedToolSet`) and the hook deny. Every `tool_call`
  event is yielded here, which is why both frames reach the UI before the
  first result. (Since P13.3 a batch wider than the semaphore announces a call
  when it is admitted, not all up front — §P13.1-3.)
- **Phase 2 overlaps only `_execute_tool_call`,** under a
  `Semaphore(max_concurrency)`. No `TaskGroup`: it cancels its siblings on the
  first exception, and one tool failing is a `tool_result`, not a reason to
  abandon the batch. Results are yielded as they land — the frontend and
  `messages.py` key on `call_id` — and a `finally` cancels outstanding tasks
  if the generator is closed mid-batch.
- **Phase 3 writes history in wire order** regardless of who finished first,
  so the replayed conversation matches the assistant message's own
  `tool_calls` array.

Caption bookkeeping also runs in wire order (`_account_captions`): the ratchet
is about the *first miss*, and the first call to come back is arbitrary.

`MCPPool._invoke` takes a per-server lock on the stdio path — one
`ClientSession` over one pair of pipes, and two concurrent `call_tool`s
interleave their JSON-RPC frames. The HTTP path opens a session per call and
is unlocked. The lock map outlives `_reopen`.

Subagents read the same config keys (`builtin_task`), since a Task is the
fan-out case this exists for and constructs its own `RunOptions`.

Ships **off**. Soak checklist before flipping it: `mcp_pool:` warnings,
`[iv.observer] inject` placement in transcripts, and
`harness.empty_terminal_iteration` counts.

`lloyd_rpc` (P9, ships off, `harness.rpc.enabled`) is the other way a turn
makes several read-only calls for one engine round trip — §P9.

### Tool naming

Built-in tools (Bash, Read, Write, Edit, Grep, Glob, Task) are advertised to
vLLM under bare names. This keeps session JSON and SOUL.md deny rules working
unchanged. Disabled tools are enforced via `RunOptions.disallowed_tools` as
`mcp__<server>__<tool>`; the bare-name aliasing in `tool_schema.py` blocks
both the bare and namespaced form at advertise and dispatch time, so
disabling `Bash` via `mcp_servers.lloyd-mcp.disabled_tools: [Bash]` blocks
the model from calling either `Bash` or `mcp__lloyd-mcp__Bash`
([[tools]] §8 has the #727 dispatch-side normalisation).

### Tool-call summaries

Every advertised tool carries one extra string parameter, `summary`: a short
phrase the model writes saying what the call is doing ("Reading server.py",
"Restarting the backend"). The collapsed tool bubble in the chat and Inner
Voice transcripts renders it as **`ToolName`** — summary, which is the whole
point — a wall of `Bash`, `Bash`, `Read`, `Bash` says nothing about what a
50-iteration turn actually did.

It is display metadata riding in the one channel a tool call has — its
arguments — which makes *where it is removed* the whole design:

- **`tool_schema.add_summary_param`** injects it and returns *which tools got
  it*. That return value is load-bearing: `session_inject_context` already has
  a required top-level `summary` of its own (1 of the 129 tools advertised
  when this landed), and popping that one before dispatch would delete a real
  argument. Injection **replaces** the `parameters` object rather than
  mutating it — it arrives as the very `inputSchema` dict held in
  `MCPPool.discovered`, which is process-shared for the life of the pool, so
  an in-place write would make the *next* turn read `summary` back as the
  tool's own parameter, skip injection, and stop stripping.
- **`loop._commit_tool_calls`** lifts the value onto the tool call's
  `_summary` and pops it from `_args_dict` — and *only* from there. The two
  records of a call deliberately disagree: `_args_dict` is what reaches MCP
  and the tool's handler — nothing validates `args` against the inputSchema,
  so a leaked `summary` is not rejected — an unknown key is handed to the
  handler (and read as a real argument by a tool that has one), while
  `arguments` is what gets replayed to the engine and is **the only record of
  this call the model will ever see again**. **Stripping the caption from
  `arguments` too is what broke the first cut of this**, and it broke it
  invisibly: session `20260907_184351_ivec8d` shows the first call of each
  tool name carrying a summary and every repeat carrying none — 5/5 vs 0/31.
  The schema said `required`; the model's own most recent example of that
  tool said otherwise, and the example won. A few-shot channel you are
  writing into cannot be edited for brevity. It costs ~10 tokens per
  historical tool call to keep, which is the price of the field working past
  its first use.
- **`summary` is injected first** in `properties` and in `required`. Property
  order is the order the schema is shown to the model and roughly the order it
  emits arguments in, so a caption placed after Bash's `command` is one
  written after a 40-line heredoc.
- `messages.py` persists it on the tool call (omitted when empty, so every
  pre-existing session reads the same), puts it on the `tool_start` SSE frame
  so the live bubble has it before the result lands, and prefers it over
  `tool_activity_detail` for the dashboard's live activity line.
- The transcript's expanded **Arguments** block hides the key, because that
  block shows what was *dispatched* and the header already shows the caption.
  It is dropped only when the header is rendering it, so a tool with a real
  `summary` parameter of its own still shows it there.

The Inner Voice observer reads the caption in two of its three tool-call
inputs, and the third exclusion is deliberate:

- **`build_assistant_message_summary`** renders
  `Bash — Checking root disk usage` per call instead of
  `['Bash','Bash','Bash']`. The observer's job is judging whether the primary
  is still on the user's request, and a wall of identical names is the least
  informative possible input for that. `observer_prompt._tool_call_labels`
  accepts all three shapes a caption arrives in — `_summary` (live harness
  event), `summary` (rebuilt from session JSON), and `summary` inside the raw
  `arguments` string.
- **`build_pretool_event_summary`** states it before the arguments: the
  caption is what the primary *said* it was doing and the arguments are what
  it actually did, so the two disagreeing is the signal. (This path is dormant
  while `pretool_llm_enabled: false`.)
- **`guards.tool_call_signature` must never see it.** `exact` is the full
  `key=value` rendering for every tool but Bash, so a caption in the args
  makes two byte-identical calls compare as different — and rewording is
  exactly what a looping model does. This is why `fire_pre_tool_use` carries
  the caption as its own `tool_summary` key rather than merging it into
  `tool_input`: `tool_input` is what safety matching and the repetition guard
  read, and it stays clean. `tests/test_tool_call_summaries.py` pins all three.

**No tool may ask for the caption twice.** `Bash` used to declare a
`description` argument — "Short human-readable description (informational
only)" — and `Task` a `description`, "Short label for the task
(informational)". Both restated, one key later in the same object, exactly
what the injected `summary` asks for, and a model answers that question once.
On 2026-09-07 it began answering into the wrong half: sessions
`20260907_235236_backlogs_a8fd` and `20260908_000804_backlogi_3828` emitted
`{"command": ..., "description": "check"}` for 49 consecutive Bash calls with
no `summary` on any of them, and the chat rendered 49 bare `Bash` rows.
**Nothing errored**, because `description` was a real Bash argument — the
caption was not dropped, it was filed where only a background task would read
it.

What makes an ambiguous schema expensive here is the ratchet described above:
`arguments` is replayed as history, so the first miss becomes the model's own
most recent example of calling that tool and the session locks into it.
Across the 16 sessions since the feature landed, every one whose *first* Bash
call carried a summary stayed above 95%; both that missed stayed below 26%.
One field decides a whole session, which is why the fix is to delete the
competing field rather than to reword it.

The two tools still need their label — a background-task row and a subagent
row are both read by a human later — so the caption travels the way the
session id and the calling turn's model already do: in the request's `_meta`,
as `lloyd/summary`, lifted into `_task_registry.current_call_summary` by
`agent_mcp/main.py::call_tool`. It must not be handed back through `args`
instead: that is what reaches the tool's handler unvalidated, and it is what
the repetition guard hashes. `tests/test_tool_call_summaries.py` pins that
Bash and Task advertise no second caption field, and that the caption reaches
MCP through `_meta` only.

`harness.tool_call_summaries: false` removes the parameter from every schema;
the UI falls back to the bare tool name. Worth reaching for if a model ever
starts spending its tool-call budget on the caption.

### The thinking trace

The harness has always emitted one `thinking_done` per agent-loop iteration.
The router kept almost none of them: a single `accumulated_thinking` buffer
held the current phase, each new phase **replaced** it (`messages.py`, not
`+=`), and it only reached disk on an iteration that produced both tool calls
*and* non-empty text. A tool-only iteration — the common shape — never
flushed, so a forty-iteration turn persisted exactly one reasoning phase, the
last one, and the chat could only ever show that. On the verification turn for
this feature the first phase was the one that chose both tool calls and wrote
no text at all: precisely what used to be discarded.

Each phase is now its own message entry, `role="thinking"`, built by
`app/routers/_messages_thinking.py`:

```json
{"id": "think_<turn>_<seq>", "role": "thinking", "content": [],
 "reasoning": "…", "reasoning_ms": 11800,
 "thinking": {"chars": 3201, "iteration": 7, "turn_id": "…"}}
```

- **Ordering is free, and that is why it is written on `thinking_done`.** The
  loop yields that event before `_commit_tool_calls` and before the
  `assistant_message` that flushes a text segment, so appending there lands
  the thought ahead of the tool and text rows it produced. No sorting logic;
  Inner Voice's timeline sorts on `timestamp` and slots it in for free.
- **The role is what keeps reasoning out of the transcripts, and it is the
  only thing that does.** Every producer generated from a session log branches
  on role first — the vault exporter and `_build_capture_transcript` in
  `app/post_capture.py`, `session_titles.build_transcript`, the
  `scripts/memory/*` renderers, `scripts/extract-trajectories.py`,
  `session_recall`'s corpus — and none has a `"thinking"` case. Measured, not
  assumed: change the role to `assistant` and six of the seven leak; leave the
  role alone and filling `content` leaks from none. `content` stays empty as a
  *second* layer, against a future producer that walks content without
  checking role. Do not read the empty content as the reason this works and
  conclude the role is free to change. `tests/test_thinking_trace_transcripts.py`
  pins both directions.
- **It never re-enters the prompt.** `thinking` is not one of compaction's
  conversation roles, so the rows are dropped before
  `_prepare_messages_for_harness` is reached. Preserved thinking
  (`loop._assistant_message_for_history`) is a separate, in-flight mechanism
  and is untouched.
- **A hard compaction discards the trace**, exactly as it already discards
  `subliminal` rows — the rewrite sets `data["messages"]` to the
  conversation-only set. The event log's `brain1.thinking_block_emitted` stays
  the durable record. Called out at that site so it does not read as a bug.
- **`accumulated_thinking` is cleared when a phase is flushed**, so what
  remains at the cancel and error paths is only a phase that streamed deltas
  and never reached `thinking_done` — which is exactly what those paths should
  still attach to their own message. Nothing is lost to a cancel
  mid-reasoning, and no phase is written twice.
- **`seq` rides on the SSE frame only when the trace is on.** That is how the
  browser knows a row is going to be persisted for this phase; without it, it
  withdraws its provisional row and falls back to hanging the reasoning off
  the assistant bubble. The kill switch therefore changes live rendering and
  reload rendering together rather than leaving them disagreeing.

The UI is one component — `ChatPanel`'s `ThinkingRow`, which the chat, the
right-hand chat sidebar and the Inner Voice timeline all mount, so none of
them needed its own work. The row must be handled **above** `MessageRow`'s
content guard: it has no content blocks and would be dropped before it
rendered. It opens on the first `thinking_delta` and ticks locally, because
`thinking_done` arrives after the iteration's text and a row created there
would sort below the answer it preceded; `makeThinkingTracker` holds that
bookkeeping in one place because the file's two stream handlers are
near-duplicates and drift between them is the standing hazard there.

`harness.thinking_trace.enabled: false` restores the old behaviour.

### Structured verdicts (the finalizer)

Worker verdicts used to be parsed out of `VERDICT:` / `SURFACE:` lines by
regex. That works until a turn words it slightly differently, and then a
`confirmed` is recorded as `unverifiable` and an item is retired for a
formatting reason.

`RunOptions.final_schema` asks the loop for one extra completion after the
turn ends, restating its conclusion as a JSON object
(`app/harness/finalizer.py`). Four things decide whether it is worth having:

- **The extra request must send the identical `tools` array with
  `tool_choice: "none"`.** Qwen renders the tools array inside the system
  message, so dropping it diverges the rendered prompt at token 41 of 1536 —
  2.7% in — and everything after that is a cache miss. Measured in the
  production shape (four tools-bearing iterations on a 180k conversation, then
  one finalizer): keeping tools reuses 177,600 of 180,068 tokens and takes
  1.19 s; dropping them reuses nothing and takes **25.00 s**. 21x, for one
  object. `eval/measurements/finalizer-2026-09-08.md` has the runs, and the
  two ways to measure this wrongly — `cached_tokens` reads 0 for the first two
  requests of any prefix on this engine, and an alternating A/B ends up
  caching both shapes, which is not what production does.
- **It is skipped unless `stop_reason` is `stop`/`end_turn`.** Forcing a
  verdict out of a turn that died at `max_turns` recreates the failure
  `INCOMPLETE` was added to fix. The reason is reported as `structured_error`,
  so a caller can tell "the turn never reached a verdict" from "the model
  refused to produce one".
- **The regex stays.** The finalizer can be skipped or can fail, and a verdict
  pipeline with no fallback turns a transient engine error into a lost triage.
  `parse_verdict(text, structured)` prefers the object when its verdict is
  known and records `source`; the ledger carries `verdict_source` and
  `structured_error` so a finalizer that quietly stopped working does not look
  exactly like one that is working.
- **The budget is the whole completion, thinking included.** The grammar
  applies only after `</think>`, so reasoning tokens are spent before the
  object starts. At `finalizer_max_tokens: 1024`, 14 of the first 34 triage
  verdicts (41%) came back as well-formed JSON cut mid-string inside
  `check`/`evidence` — 13 of them `confirmed`, the verbose verdicts that
  matter — and every one was recorded `output is not JSON`, the same message a
  model writing prose would get. The default is 4096 now, `finalizer.py` names
  a truncation as one (`finish_reason: length`, or an unclosed `{`), and the
  ledger carries `finalizer_tokens` per verdict so the budget is a number
  beside the failure rather than a regex rate to be inferred.

`TRIAGE_VERDICT_SCHEMA` is built from `VERDICTS`/`SURFACES` rather than
restated — one list, or a new verdict lands in the grammar and not the
validator. It carries **no `maxLength`**: that is enforced by the guided
decoder, so the model would stop mid-sentence at the limit rather than write
something shorter. The clamps stay in Python, after the fact.

The router honours `final_schema` only for a session whose platform is in
`sessions_io.NON_USER_PLATFORMS`. A chat turn that quietly ran a second
completion under a grammar would be paying tokens for something nobody reads.

Every triage turn asks for the object; its kill switch
(`autotriage.structured_verdict`) was retired on 2026-09-24 with
`close_on_settle`, `reopen_reverted` and `unfold_spent_umbrellas`, all on
since landing.

## Review 2026-09-24

### D9 — one anchor per iteration; no orphaned tool calls; Stop during prefill

- **A retried iteration runs its head once.** The overflow and multimodal
  recoveries `num_turns -= 1; continue`, which re-entered the loop head with
  the same iteration number, so the notification drain and the state anchor
  ran again and a second copy of the anchor was appended onto a prompt being
  retried *because* it was too big. `TurnState.prelude_done_for` records
  the iteration whose head already ran. `test_state_anchor.py` pins `[1, 2]`.
- **Cancelling the dispatcher cancels the MCP call.** `asyncio.wait` does not
  cancel what it waits on; `_execute_tool_call` now cancels `tool_task` on any
  `BaseException` from the wait (a generator closed mid-batch, the pool's
  `wait_for`) and still cancels the cancel-watcher in `finally`.
  `test_dispatch_split.py::test_cancelling_the_dispatcher_cancels_the_pool_call`.
- **Stop lands mid-prefill.** `client.stream_chat` checked `cancel_event`
  only after a line arrived, and a 200k-token prefill emits no line for as
  long as it takes. `_next_line` races every read against one
  `cancel_event.wait()` task for the stream; a cancel leaves `cli.stream`,
  which closes the connection and aborts the request in vLLM.
  `chunk_timeout_s` still bounds only the gap after the first line.
  `test_stream_stall.py::test_a_cancel_during_prefill_ends_the_stream_promptly`.

### X1 — one event sink

`app/harness/telemetry.py::log_harness_event(session_id, event, data, *,
turn_id=None)` is the harness's event sink. It imports nothing from the
harness, so modules the loop imports (`hooks.py`, `mcp_pool.py`, the
compaction state) can record events without importing `loop`.
`loop._log_harness_event` is an alias of it; nothing that names the old symbol
moved.

### X2 — `iteration_retry`

`events.iteration_retry(reason, attempt, discarded_text_chars,
discarded_thinking_chars)` says the iteration in flight is thrown away and
requested again, and how many characters of each kind it had already yielded
as deltas. `events.trim_discarded` is the one trim: `messages.py` applies it
to `full_response` / `accumulated_thinking`, forwards a `retry` SSE frame and
logs `harness.iteration_retry`; `run_recorder` trims its own buffers the same
way. The browser ignores the frame today (`api.ts`'s SSE switch has no
default arm). Nothing emits the event yet: the stream retry (D7) and the echo
guard's `tool_choice` mode (P6a) will. `tests/test_iteration_retry.py`.

### X3 — `turn_id` on every row

The user row (`_run_turn`, and the recorder's opening row), the tool-call row
and the tool-result row now carry the `turn_id` of the turn that wrote them,
as the assistant text and thinking rows already did. `build_tool_call_entry`,
`build_tool_result_entry` and `build_user_entry` take an optional `turn_id`
and omit the key when it is empty, so a caller without one writes what it
wrote before. `tests/test_transcript_entries.py`.

### X4 — `app/tool_bans.py`

`WORKER_AUTOMOD_BAN` and `WORKER_GRANT_MINT_BAN` live in stdlib-only
`app/tool_bans.py`, readable from the aggregator without importing the worker
package; `workers/sources/_common.py` re-exports the same tuples and the chat
router imports them from `app/`.

### X5, X6 — decisions only

The transcript pointer for a tool result is `maybe_spill`'s
`<persisted-output>` block, with no second marker (X5). The persisted
compaction record is a top-level `data["compaction"]` key written after
`messages`, never a message row (X6).

### P13.0 — behaviour, not source text

The loop's ordering invariants were pinned with `inspect.getsource(run_query)`
substring checks, which certify that a line is still there rather than that
the loop still behaves. They are now driven through the real `run_query` on
`app/harness/tests/_replay.py`: `ReplayEngine` replaces `loop.stream_chat`
(SSE-shaped chunks — reasoning under either key, content, `tool_calls` name and
argument fragments, finish, usage — and a snapshot of every request's messages
and `tools`), and `ReplayPool` replaces `loop._build_pool` (answers from a
dict, `delay_by_call_id` to reorder completions, start/completion/cancel order
and peak overlap). `app/harness/tests/test_loop_invariants.py` is the index:
position 0 inserted once and history append-only, wire order on both dispatch
paths, `_pre_dispatch` sequential before any call, both reasoning keys, the
finalizer's tools array equal to the last request's, overflow recoveries ≤ 2.
The stdio lock is driven through `MCPPool.call_tool` with fake sessions
(`test_dispatch_split.py`), including a reopen while a call holds the lock.
Write new loop tests on these seams; do not reintroduce source pins.

### D3 — restored files reach the engine

`compaction_llm.restore_recent_files` returns ONE `role: "user"` row,
`<restored-context><file path=… truncated_to=…>…</file>…</restored-context>`,
with the file count on the row (`restored_files`, read by
`restored_file_count`; the adapter never forwards it). The old per-file
`role: "system"` rows were dropped by `_prepare_messages_for_harness`, which
keeps only `user`/`assistant`/`tool`, while `tokens_after` still counted them.
`tokens_after` now estimates only those three roles. `/compact` no longer
restores at all (still true after D11, which stopped it rewriting history): its history was persisted, and a persisted `user` row would be
read by every transcript producer as something the user typed (the old system
rows were persisted and never sent, so nothing reached the engine either way).
Pinned by `tests/test_compaction_llm_restore.py`.

### D6 — rung 1 budgets from the meter

`loop._intra_turn_microcompact(chat_messages, *, options, meter, …) -> int`
reads `meter.used` and `ContextMeter.offset` instead of re-deriving an offset
from `total_usage["input_tokens"]`, a turn PEAK: after one relief pass the
peak booked every freed token as fixed cost and the next pass cleared to the
`keep_recent` floor, and on the overflow path (whose rejected size never
reaches `total_usage`) rung 1 under-triggered. `_relieve_context` no longer
takes `total_usage`. Unmeasured meter → the estimate alone, as before.
Pinned by `test_context_meter.py::test_a_second_pass_budgets_from_the_relieved_size_not_the_peak`,
`::test_an_unmeasured_meter_still_lets_rung_one_run_on_the_estimate` and
`tests/test_overflow_recovery.py::test_the_rejected_size_reaches_rung_one`.

### D10 — rung 1 deny-list; error results spill

`compaction.microcompact.non_compactable_tools` (config.yaml ships
`DEFAULT_NON_COMPACTABLE`: TodoWrite, ToolSearch, Enter/ExitPlanMode,
SetGoal, ClearGoal) switches both the turn-start pass and rung 1 to deny-list
mode: every tool result may be cleared (spilled first) except those, so MCP
domain results no longer go straight to truncation. Deleting the key (or
`null`) restores the `compactable_tools` allow-list. It reaches rung 1 via
`mcp_discovery.intra_turn_compaction_kwargs` →
`RunOptions.intra_turn_microcompact_non_compactable`. `Task` is not listed
(`keep_recent` protects the newest result); skill-delivery results are short
and left alone by size. `_execute_tool_call` now `maybe_spill`s error results
too; the empty-result marker stays success-only. `/compact` ran no microcompact
pass at all after D11.

### P7 — the context-rot curve sets the compaction trigger

`compaction.microcompact.trigger_fraction` (0.72 of the 210,144-token
truncation threshold, ~151k) was a KV cost knob with no quality measurement
behind it. `eval/run_context_rot_eval.py` measures where the primary's recall
falls with prompt length: {50k…240k} × depth {0.1…0.9} × {single, multi3,
distract4} × 3 seeds = 225 requests per shape, `repo` (the admission bench's
`corpus()`) and `session` (the compaction eval's `build_session`), needles
the compaction eval's codename and "current port" with the old port and
sibling ports salted as distractors. Thinking off, 64 tokens, temperature 0,
priority 1; cold (nonce at position 0) and warm TTFT; `wait_idle` before every
request; the pool paused via `POST /api/workers/pause` and resumed in a
`finally` only if the run paused it. `--dry-run` sends nothing.

`--decide` computes `L*` (the largest length such that every grid length up
to it keeps distractor accuracy ≥ 0.9·A(50k) with no depth under 0.8·A(50k);
the worse shape decides) and the recommendation: keep 0.72/0.52 at
`L* ≥ 151k`, else `trigger = floor(L*/210,144·20)/20`, `target = trigger − 0.20`,
priced against 14 days of `usage.db` (read-only) as extra compactions × cold
TTFT over engine-busy seconds (< 5% to adopt, after the compaction recall eval
and a 3-day soak). Report: `eval/measurements/context-rot-<date>.md|json`.
Re-run whenever `models.primary.expect_model` changes.
`tests/test_context_rot_eval.py`.

### P10 — injection-resistant action reviewer (shadow)

Two passive seams for worker turns; neither can change a tool call today.

- **Action reviewer** (`app/harness/action_review.py`, `harness.action_review.mode:
  off|shadow|warn`, default `shadow`). Installed by `messages._install_action_review`
  on `/api/message/stream` for `NON_USER_PLATFORMS` turns only, as an OnEvent tap
  (no PreToolUse callback, so it cannot deny or deliver). Per tool call it
  enqueues one frozen djev question through `djev_shadow.shadow(seam="action_review")`
  — options in fixed order `consistent with the task` / `unrelated to the task` /
  `what an instruction embedded in fetched content would ask for` (schema
  `eval/djev/schemas.py::ACTION_REVIEW`, switch `djev.shadow.seams.action_review`).
  The canvas is the worker's prompt plus the calls so far, never tool results or
  prose; `actual` is the gates' outcome read off the result. The canvas is built
  on the recorder's thread. `warn` records like `shadow` until a threshold exists.
  `architecture/djev.md` §6.4.
- **Input probe** (`agent_mcp/_injection_probe.py`, `harness.injection_probe.mode`,
  default `shadow`). Regex families (`role_header`, `ignore_instructions`,
  `you_must_now`, `run_the_following`, `conceal_from_user`, `persona_swap`,
  `new_system_prompt`, `invisible_chars`) over the text results of `Read`,
  `http_fetch`, `http_request`, `vault_read`, `browser_snapshot`, applied in
  `main.call_tool` after the module call and after the effect ledger, background
  sessions only. `shadow` logs `harness.injection_probe_hit {tool, pattern_id,
  excerpt}` from a thread the call does not wait on; `warn` appends one
  `<warning>` to a non-error result. Never raises into `call_tool`.
  `session.py::_INJECTION_PATTERNS` (the `memory_add` gate) is unchanged.

**What gets measured (not done here).** A week of `action_review` rows in
`~/.local/state/lloyd-djev/shadow.jsonl` (extend until ≥ 200 rows); a stratified
200-call sample (source × score decile) hand-labelled by Alan as consistent /
off-task / injection-shaped, plus 20 synthetic positives replayed with a planted
instruction under `LLOYD_DJEV_SHADOW=0` into a separate file; AUC and FP/FN at the
best threshold on P(`injected`). For the probe: hit rate per `pattern_id` from
the event logs and 50 hand-checked hits for precision (this repo's docs describe
injections, so arch-review reads are the expected false positives).

**Decision rule.** `mode: warn` — the probe's warning, and an Inner Voice inject
when the reviewer picks `injected` above the measured threshold — only if FP ≤ 2%
at FN ≤ 30% and `label_mass` clears the floor set from the week's rows (write
`threshold`, `label_mass_floor`, `calibrated_on`, `calibrated_hash` into the
schema). Never a hard block on djev alone; deny stays with the safety and grant
hooks. `tests/test_action_review.py`, `tests/test_injection_probe.py`.

### D5 — deny hooks fail closed

`HookRegistry.add_pre_tool_use(matcher, cb, *, fail_closed=False)`. A raising
PreToolUse callback writes one `harness.hook_raised` event through
`telemetry.log_harness_event`; fail-open (the default: the Inner Voice
observer, `skill_dispatch`) then passes as before, fail-closed
returns a deny naming the gate and the error at once, ahead of any held
deliver. The three gates — `safety._safety_pretool_cb`,
`outbound_content._content_pretool_cb`, `policy._policy_pretool_cb` — register
fail-closed, and so does the bench-contamination guard
`bench_corpus._bench_corpus_pretool_cb`. `_pre` stays `(matcher, cb)` pairs (tests unpack them); the flag
lives index-for-index in `_pre_fail_closed`. Pins:
`app/harness/tests/test_harness_unit.py` (the four `a_raising_*` /
`a_raised_hook_*` tests) and
`tests/test_harness_safety_docstring_pointers.py::test_the_three_gates_are_registered_fail_closed`.

### D8 — observer inject race; observer text

- **An inject made on the assistant_message is moved behind the batch.**
  The observer judges an iteration with tool calls on its `assistant_message`
  and can inject then, before the batch runs; `batch_base` was taken after
  that hook, so the inject sat outside the slice `_reorder_batch_messages`
  may move and the next request went out as `assistant(tool_calls) → user →
  tool`. It is now `chat_msgs_len_before_hook`, shared by both dispatch
  paths.
  `test_loop_inject_ordering.py::test_an_inject_landing_during_the_assistant_hook_is_reordered_behind_the_tool_results`
  drives both.
- **The observer sees the primary's text.** Its `accumulated_text` was fed
  from `text_delta`, which the loop never fires OnEvent for, so every
  per-event prompt said "(none yet)" and the /goal evaluator got nothing when
  `result` carried no `response_text`. It is appended from each
  `assistant_message` now; the `text_delta` branch is a no-op guard against
  double counting. `hooks.py`'s docstrings name the four events that do fire
  (`assistant_message`, `tool_call`, `tool_result`, `result`).
  `tests/test_observer_accumulated_text.py`.

### D1 — transcript rows are pointers, not 2k cuts

A tool result over `TOOL_RESULT_MAX_CHARS` (2,000) used to be stored as its
first 2 KB and nothing else, and the next turn's history is rebuilt from that
row — so a 20 KB `Read` the model saw whole in turn N was 2 KB with no way back
in turn N+1. `transcript_entries.shape_tool_result_for_transcript` now runs
`maybe_spill` at that threshold: the full text goes to
`<sid>.tool-results/<call_id>.{txt,json}` and the row carries the
`<persisted-output>` block (path, 2 KB preview, a recovery sentence built
from the writing turn's deny list). Both writers call it (`messages.py`'s
`tool_result` branch, `run_recorder`), and `build_tool_result_entry` reads
the path back off the block into `stats.persisted_path` (omitted otherwise).
A block the live turn already spilled at 50k is stored whole — the old cut
sliced off its recovery sentence. No session id, a failed write, or
`compaction.transcript_spill.enabled: false` all give the old cut.

- **No file-name collision.** The live spill fires only at 50k, and then this
  function sees its block and writes nothing; below 50k the only other writer
  of `<call_id>.<ext>` is microcompact's `persist_for_compaction`, with the
  same content and so the same extension. Screenshots are `<call>.img<n>.<ext>`.
- **Microcompact already handles it.** The turn-start pass drops an old
  pointer's preview to its header (path kept) for compactable tools older
  than `keep_recent_tools`, with no pressure needed; the recent window keeps
  its preview. `tests/test_compaction.py` drives it through
  `load_and_compact_session`. MCP-domain results are outside that list and
  keep their preview (D10's scope).
- **Readers.** `GET /api/sessions/{sid}/tool-results/{name}` already serves
  the files (`txt`/`json` media types). The chat renders the block verbatim in
  the collapsed tool bubble's `<pre>` (legible; the path is not a link yet).
  The two vault exporters show `tool_result_preview` (preview, not tag line
  and absolute path) in their 300 chars; `session_recall` indexes only
  user/assistant rows. `DELETE /api/sessions/{id}` now removes the spill
  directory too; `sweep_session_spills` ages the rest.
- `tests/test_transcript_entries.py` pins the four shaping cases, the switch,
  the delete and the two writers agreeing on a 10k pointer.

### P11 — harness telemetry

The loop measured almost nothing about itself: no time to first token, no tool
latency, no reason a tool call failed, and `completion_tokens_details` (the
reasoning count) was dropped by `_merge_usage` exactly as the cache hits once
were. Now:

- **`assistant_message`** carries `ttft_ms`, `request_ms` and `cache_ratio`.
  The request clock starts after the pre-request relief pass
  (`request_started_at`, just after `request_msgs_len`), so relief shows in
  `duration_ms` (still the whole iteration) and never as a slow prefill. No
  chunk at all is `ttft_ms: None`; no reported prompt is `cache_ratio: None`.
- **`tool_result`** carries `duration_ms` (the MCP call, cancel race
  included), `handshake_ms` when the pool reports `result["timing"]` (P12 adds
  it; absent until then) and `error_class` on every failure:
  `parse_error`/`disabled`/`denied` from `_pre_dispatch`,
  `transport`/`dispatch_failed`/`cancelled` from `_execute_tool_call`'s exits,
  `mcp_error` for a protocol refusal (the pool's `MCP error calling` prefix),
  `tool_error` for the tool's own failure. `events.TOOL_ERROR_CLASSES` is the
  list. An early return that never reached MCP has no `duration_ms` — absent,
  not zero.
- **`result.stop_reason`**'s Literal now names every value the loop emits,
  including `context_exhausted` (emitted for months, missing from it),
  `length` and `tool_calls` (a terminal iteration's own finish reason) and
  `stream_error` (D7). `test_result_stop_reason_literal_matches_what_the_loop_emits`
  greps the loop's assignments against it.
- **`_merge_usage`** keeps `reasoning_tokens`; `_accumulate_iteration_usage`
  sums it.
- **`harness.overflow_recovered`** is written once per overflow recovery.
  `telemetry.log_harness_event` also bumps a per-turn tally bound by
  `telemetry.bind_event_counts()`; that is how `harness.hook_raised` (D5) and
  `harness.stream_retried` (D7) reach the usage row without either emitter
  knowing a row exists (hooks and parallel tool tasks inherit the context, so
  they share the dict).
- **`app/turn_usage.py::TurnTelemetry`** folds a turn's events into the new
  `usage.db` columns — `stop_reason, reasoning_tokens, ttft_ms_first,
  ttft_ms_max, tool_calls, tool_errors, tool_errors_by_class (JSON),
  tool_ms_total, stream_retries, overflow_recoveries, hook_raised,
  wrapped_up` — for all three writers (streaming chat, the sync endpoint,
  `run_recorder`), so they cannot drift (`tests/test_turn_usage.py` drives the
  chat router and the recorder over one stream and compares the rows).
  `partial_row()` is the running peak-prompt / summed-output total D12 books
  for a turn that dies before `result`. Columns are additive in
  `usage_store._init_schema`; NULL is unmeasured, never zero.
- **Queries and dashboard**: `usage_store.stop_reason_breakdown`,
  `tool_error_breakdown`, `ttft_summary`, `reasoning_tokens_summary`, served as
  `stop_reasons_24h`, `tool_errors_24h`, `ttft_24h`, `reasoning_tokens_24h` by
  `dashboard._usage`; the Tokens panel shows a turn-endings strip and a TTFT /
  tool-error line under the prefix-miss line.

Verify on live traffic:
`sqlite3 ~/lloyd-data/usage.db "select stop_reason,count(*) from usage where ts>datetime('now','-1 day') group by 1"`
and `grep -h '"harness\.' ~/lloyd-data/event_logs/*.events.jsonl | jq -r .event | sort | uniq -c`.

### D4 — a Task child inherits the grant gate and the deny list

The loop stamps two more `_meta` keys on a call: `lloyd/grant_scope` (the
option `RunOptions.grant_scope`, which the router and `autonomy.run_task` set
beside their `install_policy_hook(scope=…)`; else `policy.current_scope` only
where the pool bound it — its default `"worker"` is never read, or every chat
Task would be gated) and, on `Task` calls only, `lloyd/disallowed_tools` (the
iteration's refreshed deny list). `agent_mcp/main.call_tool` lifts them into
`builtin_task.current_parent_grant_scope` / `current_parent_disallowed` (and
binds `policy.current_scope` when a scope came in) around the dispatch;
`_task` unions the parent's list and `app.tool_bans.WORKER_AUTOMOD_BAN` in
both spellings into the child's deny list and installs the policy hook when a
scope is present. `tests/test_task_subagent_authority.py`;
[[editing-safeguards]] has the gate map.

### D7 — broken streams are `stream_error`, retried once while nothing was dispatched

One handler in the stream loop for every way a stream breaks: `ParseError`,
`StreamStalledError`, httpx connect / connect-timeout / read /
remote-protocol errors and a 5xx `HTTPStatusError`. A 4xx is the request's
own fault and re-raises. While the iteration has sent **no tool-call delta**,
the turn is not cancelled and `RunOptions.stream_retry_max` (1) is not spent,
the loop yields `iteration_retry` (reason `stream_stalled`, `parse_error`,
`http_<code>`, `transport`; the discarded text/thinking counts), logs
`harness.stream_retried`, sleeps `stream_retry_backoff_s` (2 s, woken by
Stop) and re-requests the same messages with `num_turns -= 1; continue` — D9's
`prelude_done_for` guard keeps the anchor from being appended twice. Otherwise
a `ParseError` ends the turn as `stop_reason="stream_error"`: no tool call is
committed (a half-parsed call used to be dispatched), the assistant message
carries the text that streamed, and the finalizer skips it. Every other error
raises as before. A `ParseError` after the finish frame (a lost usage line) is
not a broken completion and proceeds as it always did. Config
`harness.stream_retry: {max_attempts: 1, backoff_seconds: 2}`; `max_attempts:
0` is the off switch. Task children and the direct worker path take the
`RunOptions` defaults (they do not splat `_get_harness_kwargs`).

Every consumer that accumulates `text_delta` trims on `iteration_retry` with
`events.trim_discarded` — the chat router and `run_recorder` (X2), plus
`autonomy.run_task`, `_common.run_prompt_on_primary`,
the IDE query, `bench_runner_sdk`, `replay_run_state`, and the Discord bot on
the router's `retry` SSE frame. The IDE's streaming completion cannot take
text back off the wire, so a retry after text ends it. The observer takes its
text from `assistant_message` (D8), which only the kept attempt produces. The
voice worker speaks deltas as they arrive and is not trimmed. Pins:
`app/harness/tests/test_stream_retry.py`,
`test_finalizer.py::test_a_broken_stream_has_no_verdict_to_restate`.

### D2 — the summary is persisted, folded forward and bounded

Once a session crossed the wall, every turn re-summarised the whole older block
(up to 120 s before first token, a different text each time so the prefix cache
died every turn, and no input bound, so a big block 400'd the summariser and
the turn fell back to drop-oldest — every turn). `app/compaction_state.py` now
keeps it as `data["compaction"]` (X6), under `compaction.persist_summary`
(ships **false** until the recall eval's `summary_legacy` /
`summary_persisted` arms are compared; `eval/run_compaction_recall_eval.py`,
run past the truncation threshold).

- **Record**: `{version, summary, covers_through_entry_id,
  covers_through_index, covered_sha, covered_rows, covered_turn_ids,
  files_touched, created_at, updated_at, model, folds, source, instructions}`.
  Indexes are into the stack's conversation-role filter
  (`app.compaction._CONVERSATION_ROLES`, one definition). `covered_sha` is over
  `id:role` lines, so microcompact rewriting contents never invalidates it; a
  rewritten past (legacy `/compact`) does → `compaction.record_invalidated`,
  rebuilt from the rows.
- **Applied whenever it validates**, under the threshold too — a stable summary
  is the cache win. The threshold decides only whether the delta past the
  boundary is folded: `chunk_by_turns` on turn boundaries to
  `min(summary_input_budget_tokens, summary window − 8k − 2k) − prior summary`
  (re-chunked after each fold; an over-budget turn is reduced, never split),
  `compaction_llm.summarize_incremental(prior, chunk)` with fixed sections
  Goal / Constraints / Progress / Decisions / Next steps, the record saved
  after **each** fold (`save_record` → `sessions_io.mutate_session(path=…)`,
  writes only that key, re-validated under the lock, always placed after
  `messages` so the retention sweep's 4 KB prefix read still finds
  `last_active`), at most `max_folds_per_turn`. Leftover delta stays verbatim;
  drop-oldest then truncates the rows behind the summary, never the summary.
- **Files touched** is rendered by us from `agent_mcp/_change_ledger` over the
  newly covered `turn_id`s (X3) and carried forward on the record (the ledger
  keeps 7 days); the prompt tells the model not to list files.
- A failed fold keeps the prior record (`summarize_outcome: empty_summary`).
  New outcome `reused`; `turn_start_record` gains `summary_reused`,
  `summary_folds`, `summary_covered_rows`; one `compaction.summary_updated`
  event per fold. Summarize mode only — `mode: truncate` (voice) ignores the
  record, and with the switch off a record on disk is ignored entirely —
  except a record `/compact` made, which D11 applies either way.
- `tests/test_compaction_persisted_summary.py`,
  `tests/test_compaction_record.py::test_turn_start_record_reports_reuse_and_folds`.

### P12 — discovery refresh, per-tool timeouts, handshake timing

- **One catalog, swapped whole.** `mcp_pool._Catalog` (discovered, routes,
  schemas, annotations, timeouts) is built off to the side by `open()`,
  `_reopen()` and `ensure_fresh()` and published in one assignment. `_reopen`
  no longer `.clear()`s the routes, so a concurrent call never reads "no
  server claims tool" mid-rediscovery, and a reopen that fails leaves the old
  catalog standing. `_tool_routes`/`_schemas`/`_annotations` remain as
  properties over the current catalog.
- **Refresh at the turn boundary.** `loop._build_pool` awaits
  `pool.ensure_fresh()`, which re-lists the HTTP servers once the catalog is
  older than `harness.mcp_pool.discovery_ttl_s` (300; 0 = once per pool, the
  old behaviour). A failed server keeps its previous entry, a refresh that
  lists nothing publishes nothing (the empty pool stays unreachable this way),
  an unchanged listing keeps the catalog object, every attempt is stamped (a
  down aggregator costs one ≤10 s try per TTL), and a turn that finds another
  refreshing does not wait. stdio servers are not re-listed. Within a turn the
  advertised tools never change. `notifications/tools/list_changed` is **not
  implementable**: the HTTP session lives for one call, so nothing is open for
  the server to notify; the TTL poll is the substitute (the aggregator's own
  `ttl_ms` on tools/list is the same idea from its side).
- **Per-tool call budget.** `agent_mcp/annotations.py::TIMEOUT_SECONDS`
  (Bash 630, http_fetch/http_search 120, http_request 180,
  automod_gate_wait 600 — each above the tool's own internal bound) is served
  as `lloyd/timeoutSeconds` in the Tool's **`_meta`**, not on
  `ToolAnnotations`: mcp 2.1 models ToolAnnotations with pydantic's default
  `extra="ignore"`, so an extra key there is silently dropped (verified; the
  test pins it). `MCPPool.timeout_for` clamps to `[1, CALL_TIMEOUT_SECONDS]`
  and `call_tool` uses it whenever the caller passed no `timeout_seconds` —
  in the pool rather than in `_execute_tool_call`, so every caller (Task
  children, thunderbird) gets it. Off switch
  `harness.mcp_pool.per_tool_timeouts: false`.
- **Handshake timing.** `_invoke`'s HTTP path times entering `_http_session`
  (connect + `initialize()`, paid per call) apart from `session.call_tool`
  and `call_tool` returns them as `result["timing"] = {handshake_ms,
  call_ms}` — which lights up P11's optional `tool_result.handshake_ms`.
  stdio reports none. `pool.stats()` (catalog size/age, refreshes, failures,
  handshake count/avg/max) is on `/health/deep` under `mcp.pools`.
  `tests/test_mcp_pool_discovery_refresh.py`.

### P4 — bounded, typed long-term memory (build half; deploy gated on the eval)

`lloyd/MEMORY.md` is 73,002 B against a 73,728 B ceiling and ~18k tokens on
every user turn. The proposal is an index: one typed line per entry, the detail
in `~/obsidian/lloyd/memory/<slug>.md` topic files that are pulled, never
rendered. This landing is additive; production prompts are byte-identical.

- **Tools (agent_mcp/session.py, lloyd-mcp restart).** `memory_add(type=…)`
  writes `- [type] (date) text` (`user|feedback|project|reference`; default
  `user` for USER.md, `project` otherwise — the plan said `project` everywhere,
  which would tag user facts wrongly), switch `memory_tools.typed_entries`
  (config on, code default off). A tag or date the writer already wrote is not
  doubled. All four memory tools accept `file="topics/<slug>"`, validated by
  `app.memory_ceiling.topic_slug` (`[a-z0-9-]{1,48}`, the whole traversal
  defence); `TOPIC_FILE_CEILING_BYTES = 32_768` is enforced through
  `memory_write_error`, so Write/Edit/vault_write refuse the same growth.
- **Refusal hint.** `prompt_surface.size_error` appends the three largest `## `
  sections and the untyped-entry count.
- **Render-time overflow** (`prompt_builder._bound_memory_render`):
  `memory.render_overflow: annotate | render_all`. **Default `render_all`**, a
  deliberate deviation from the plan's `annotate`, so nothing changes until the
  eval promotes; `annotate` cuts at an entry boundary under the file's own
  ceiling, appends `<memory_overflow file dropped_bytes>`, logs ERROR and
  announces once a day.
- **Deployed 2026-09-25** on Alan's call after two runs at 7/10 on the pull
  check with answers non-inferior at half the prompt: vault commit 1a72649f
  (20,466 B index + 24 topic files in `lloyd/memory/`, both nightly skills
  patched), `MEMORY_MD_CEILING_BYTES = MEMORY_MD_INDEX_CEILING_BYTES`,
  `memory.render_overflow: annotate`. Watch criterion (d) — `memory_read` per
  user turn ≤ 0.5 — for 7 days from the event logs.
- **The ceiling is one constant** (history of the flip below).
  `MEMORY_MD_INDEX_CEILING_BYTES = 25_600` sits beside
  `MEMORY_MD_CEILING_BYTES = 73_728`; the deploy is
  `MEMORY_MD_CEILING_BYTES = MEMORY_MD_INDEX_CEILING_BYTES` in the same change
  that writes the consolidated index, then
  `git -C ~/obsidian apply scripts/maintenance/vault-memory-index-skills.patch`
  (the #39 knowledge-write and #47 dream skills; the plan's
  `skills/nightly-knowledge-write` is really `nightly-reflection-knowledge-write`).
- **Scripts.** `scripts/memory/validate_memory_index.py` (stdlib; `structure`
  or `full`; the live test runs `full` automatically once the ceiling flips) and
  `scripts/memory/consolidate_memory_index.py --dry-run --out <overlay>`
  (deterministic, lossless — every unit verbatim in a topic file — refuses an
  `--out` inside the vault; on today's file: 73,002 B → 20,466 B index, 85
  entries, 24 topics).
- **Eval.** `eval/run_memory_index_ab.py` over `eval/memory_index_probes.yaml`
  (10 index / 10 topic / 10 feedback, each anchored on verbatim MEMORY.md text
  and located in the built arm by `--check`; `--with-trim-probes` adds #1425's
  20). `memory_read` in a trial is answered from the arm's overlay by a
  PreToolUse deliverer, because the live tool reads the vault, which holds no
  topics yet. Decision (a)–(c) is computed; (d), live `memory_read` per user
  turn, is reported pending. `tests/test_memory_index_cap.py`. The first run
  (2026-09-25) missed (c) at 7/10: three topic probes were answered by grepping
  the ledger, git and the code, not from their index lines (the reaped-round
  line carried nothing about it). Those probes now ask for incident history the
  machine does not hold, and every topic probe names `answer_terms` that must be
  absent from the index, SOUL.md and USER.md. The consolidator was not changed.

### D12 — errored turns book usage; a cancelled task saves its text

`usage.db` saw only turns that reached `result`: `stream_stats` is filled by
that handler, so a turn that raised after two iterations booked nothing, and
one whose consumer task was cancelled lost its partial text too. The chat
router's error arm now fills `stream_stats` from `TurnTelemetry.partial_row()`
(peak prompt, summed output and cache-create, the iteration count) when the
result never came, books the row with `stop_reason="error"`, `num_turns` and
the wall-clock `duration_ms` — in the no-text branch as well — and its `done`
frame carries `stop_reason: "error"` and the error clipped to 300 chars. An
`except asyncio.CancelledError` ahead of it persists the unwritten tool pairs
and the text (`cancelled: true`, shielded against a second cancel), books
`stop_reason="cancelled"`, sends `done(cancelled)` and re-raises. A
`usage_booked` flag makes the booking once-only: a fault inside the `result`
handler after its `record_usage` used to book the turn a second time from the
error arm (`test_usage_skill_breakdown.py` had pinned the two rows).
`run_recorder.close_interrupted` books the same running totals
(`cancelled` for CancelledError/TimeoutError, else `error`), once. The browser
ignores both new `done` fields; `_common.run_prompt_in_session` maps
`stop_reason: "error"` back to `None`, which its callers read as "never
completed" (autocode's `infra_failed`), exactly as an errored turn read before,
and appends the error to `errors`. Pins: `tests/test_messages_errored_turn_usage.py`.

### P6 — `tool_choice` levers

`client.stream_chat(..., tool_choice="auto")` sends the value only beside a
non-empty `tools` array. The template renders `tools`, not `tool_choice`, so a
forced value changes what the engine may emit and leaves the prompt
byte-identical — the property the finalizer already relies on for `"none"`.

- **(a) Echo guard** — `harness.echo_guard.mode`: `nudge` (shipped, the old
  behaviour) or `tool_choice`. In `tool_choice` mode a fenced shell block with
  no tool call (and no observer inject) is discarded: the assistant message is
  popped, `iteration_retry(reason="echo_guard")` takes its deltas back off
  every consumer, and the identical request goes out with `"required"` without
  spending an iteration. Stays `nudge` until `"required"` is measured: vLLM
  satisfies it with a JSON grammar, not qwen3_xml's own format.
- **(b) Max-turns wrap-up** — `harness.max_turns_wrapup: {enabled, models}`.
  At the budget the loop appends one user message ("You have used all N
  iterations…") and sends one more request with `"none"`; the answer ends
  `response_text`, a tool call in it is dropped (never dispatched, never in
  history or events), `stop_reason` stays `max_turns` (INCOMPLETE, finalizer
  skip unchanged), and `result.wrapped_up` is true (P11's `wrapped_up`
  column). No drain or budget anchor runs on that request. **Per engine**:
  `models` resolves to base URLs (`mcp_discovery.max_turns_wrapup_kwargs`) and
  the loop wraps up only when the run's own `base_url` is listed. Shipped
  `[primary]`: vLLM 0.28.1 honours `"none"` with tools present
  (`exclude_tools_when_tool_choice_none` false, no tool parser attached —
  verified for `finalizer.py`). The llama.cpp secondary is not verified and is
  left out; add it only after checking `none` there. Task children splat the
  same kwargs, so a child that exhausts its budget returns its wrap-up as
  `response` marked `truncated`. Not wired into `_worker_run_options`
  (`run_prompt_on_primary`), whose `TurnResult.ok` reads non-empty text as
  success. Pins: `app/harness/tests/test_tool_choice_levers.py`,
  `tests/test_task_subagent_answer.py::test_a_wrapped_up_budget_run_returns_its_summary_marked_truncated`.

### D11 — `/compact` is a queued turn that folds into the same record

`/compact` was a one-shot SSE generator beside the queue: it read a snapshot,
awaited the summariser (up to 120 s), then replaced `data["messages"]` — so
anything appended meanwhile was deleted, every `thinking`/`subliminal` row in
the session went with it, it ran the legacy count-rule microcompact, and it
recorded nothing.

- `post_message_stream` enqueues `SessionTurn(source="user", payload={"kind":
  "compact", instructions, model, meta_path})` (`_compact_turn`); `_run_turn`
  dispatches it to `_run_compact_turn`, so it waits behind a running turn and
  holds the slot while it folds.
- `compaction_state.manual_compact` folds everything but the last
  `keep_recent_turns` past the record's boundary through D2's `fold(...,
  source="manual", instructions=...)`, at most `compaction.manual.max_folds`
  (12) folds. **It never writes `data["messages"]`**; only `data["compaction"]`
  changes. No files are restored (D3), and the count-rule pass is gone.
- Events keep their keys: `compact_start {session_id, instructions}`,
  `compact_done {session_id, microcompacted, summarized, restored_files,
  truncated, tokens_before, tokens_after, context_window}` plus `folds` and
  `covered_rows`; `error {detail}` when nothing changed (summariser failed).
  `web/src` reads none of them today. It is recorded: `compaction_record.start_turn`
  + `note_turn_start`, and one `compaction.manual` event (the turn-start record
  plus `source`, `instructions`, `folds`, `covered_rows`; `ok: false` on failure).
- **With `persist_summary` off (as D2 ships), a manual record is still applied.**
  The record is all `/compact` leaves behind now, so ignoring it would make the
  command a no-op. A record carries `manual_at` (set by a manual fold, carried
  forward through every later automatic fold); `compaction_state.is_manual`
  puts the session on the persisted path whatever the flag says. In summarize
  mode later over-the-wall turns fold into it; in truncate mode (voice) it is
  applied but never folded.
- `tests/test_compaction_manual.py`,
  `tests/test_session_queue.py::test_a_compact_request_waits_behind_the_running_turn_and_a_row_appended_meanwhile_survives`.

### P5 — skills index with descriptions; body by tool (build half; deploy gated on the eval)

Ships with today's behaviour byte for byte; the eval decides the flips.

- **Index** (`prompt_builder._load_skills_index` → `_skills_index_lines`,
  `skills.index` in config.yaml). `descriptions: false` (default) is the
  names-only line, pinned by `test_off_is_bytewise_today`. On: one
  `- name — description` line per skill, the description being `/api/skills`'
  field whitespace-folded and clipped at `max_description_chars` (100); who gets
  one is decided by 30-day use (`offers + loaded + loaded_by_read`, scanned once
  per UTC day per process, ~0.8 s) under `budget_chars` (12,000); the tail is
  names-only, no skill is ever dropped, and the lines render alphabetically so a
  re-rank moves the cached prefix at most daily and says nothing by position.
  On the live vault (189 skills) the index is 11,981 chars with 78 described.
- **Pull arm** (`prefetch.skills.push`, default `true`). `false` makes
  `prefetch._skill_injection_plan` plan nothing — no body, no excerpt — while
  `_emit_skill_match_events` still writes every offer (`landed: false`), and
  the index note becomes "load it with skills_read(name)".
  `prompt_builder.skills_push_enabled` is the one reader for both.
- **Telemetry.** `skill_injection_counts` gains `loaded_by_read {skill: n}` and
  `reads` — `skills_read` calls off `brain1.tool_call_proposed` rows — as a
  separate mapping so offer entries keep their shape and `no_telemetry` stays
  about offers.
- **Lint.** `scripts/skill_lint.py` DESCRIPTION bucket, measured with the
  index's own rule and clip: `missing` (name-only; a finding) and `clipped`
  (advisory). Premise correction: the live vault has **0** skills without a
  description (the plan's 48 is stale); 172 of 189 exceed the 100-char clip
  (mean 191 chars), which is what the bucket now surfaces.
- **Eval.** `eval/run_prefetch_cost_eval.py` arms `desc_push` (descriptions
  on, push on) and `desc_pull` (descriptions on, push off, the turn's `<skill>`
  section dropped — what an empty plan renders), each under its own system
  prompt built by the production builder with the two keys swapped in and
  charged past its own probed prefix; `--queries eval/skill_match_queries.yaml`
  loads the 51 labelled turns; column `right_skill_reached` (injected, or
  `skills_read(expected)` in the trace), contrasts against `injected`, and
  `pull_lost_vs_push` for the decision rule. Not run live yet:
  `--n 100 --arms injected,injected_aa,desc_push,desc_pull --label skills-index`
  in a paused-pool window. `tests/test_skills_index_descriptions.py`,
  `tests/test_skill_injection_telemetry.py`.

### P3 — memory flush before compaction (ships off)

Shortly before the compaction wall, one quiet turn writes what is worth keeping
into memory while the older turns are still verbatim (OpenClaw's pre-compaction
flush). `app/memory_flush.py`; switch `compaction.memory_flush.enabled`
(**false** until the recall eval's `summary_legacy` / `memory_flush` arms are
compared, `eval/run_compaction_recall_eval.py`, run past the threshold).

- **Trigger** at the end of a chat turn, in `_run_turn`'s `result` branch
  (`_memory_flush_after_turn`), not inside `load_and_compact_session`: when the
  turn's engine-reported peak prompt ≥ `trigger_fraction` (0.85) × the
  compaction threshold and no flush is open in this cycle, it enqueues
  `build_flush_turn` as an ambient turn (dedup key `memory_flush`; runs when
  the session is idle, a user turn preempts it).
- **Cycle** bookkeeping is `data["compaction"]["flush"]` = `{turn_id, at,
  status queued|done|cancelled, trigger_tokens, threshold, memory_adds,
  fact_adds, duration_ms, consumed}`. A cycle ends when D2's record
  `updated_at` passes the flush's `at`, or when a later turn start summarised
  or truncated (`consumed`, the only signal with `persist_summary` off). A
  cancelled flush, or one queued over an hour ago that never reported, does not
  hold the cycle closed. `compaction_state.save_record` carries the entry over;
  `load_record` ignores a `compaction` dict holding only a flush.
- **The turn**: `RunOptions.allowed_tools` (new; None = no allow-list) =
  `memory_read, memory_add, fact_get, fact_add` (the plan's `fact_search` does
  not exist). Applied where the catalog is built (the left-out tools join the
  surface-hidden set, so every iteration's dispatch set too) and explicitly in
  `_pre_dispatch` (an undiscovered name or ToolSearch is refused, `disabled`).
  An allow-list turn gets an uncached LoadedToolSet, so the chat's loaded tools
  survive it; tool search is off unless ToolSearch is listed. Task children
  build their own RunOptions and are unaffected (and `Task` is not listed).
  `priority=1`, `max_turns` 6, session system prompt (cache reuse), safety +
  grant hooks, no Inner Voice (`_iv_should_fire_on_turn` skips `memory_flush`).
  The prompt asks for typed `memory_add` entries (P4's `type`) and one closing
  line; it does not address the user.
- **Rows**: the flush turn's id is `mflush-<hex>` (`FLUSH_TURN_PREFIX`), so
  every row it writes carries it (X3). `app.compaction.is_history_row` is the
  one filter for the turn-start stack and `compaction_state.conversation_rows`:
  flush rows stay in the transcript and never re-enter the prompt, the summary
  or its index. (The user row is not separately tagged `producer`: the id is
  the tag, and it kept this item's hunks out of `_run_turn`'s top.)
- **Measure**: event `compaction.memory_flush {turn_id, trigger_tokens,
  memory_adds, fact_adds, duration_ms, status, stop_reason}` when the flush turn
  ends; `turn_start_record.flushed_before_summary` on the next rewrite.
- `tests/test_memory_flush.py`.
### P1 — cross-turn prefix reuse (switched; ships as today)

The system prompt heads every request, so anything in it that moves between
turns re-prefills the whole previous conversation on the next turn's first
iteration. Two things moved it: the session state (`<goal>`, `<plan>`,
`<active_todos>`, rendered ahead of the static harness hints) and the memory
files (re-read every turn, so one `memory_add` anywhere invalidated every open
session). Three switches under `harness.prompt_layout`, all shipping off:

- **`session_state: system_head | system_tail | user_tail`**
  (`prompt_builder.session_state_layout`, unknown values read as
  `system_head`). `system_head` is today's output byte for byte (checked
  against `f75f4004` over seven input shapes with the real vault; pinned by
  `test_system_head_is_byte_identical_to_today`). `system_tail` renders one
  `build_session_state_block` (`<session_state>` around the same three
  renderers) after every static paragraph, registered as the `session_state`
  component. `user_tail` leaves it out of the system prompt;
  `app/prompt_layout.turn_tail` builds it and the four turn paths (stream,
  ambient, sync in `messages.py`; `voice._voice_turn_setup`) append it to
  `prefetched_text`, so it lands at the tail of the sent user message.
  P3's flush turn renders the same (frozen) system prompt but gets no tail;
  D11's compact turn builds no prompt at all.
- **`freeze_memory`** (`app/memory_snapshot.py`): the first turn writes the
  memory body `_load_memories` renders for the session's platform to
  `sessions/<sid>.tool-results/_memory_snapshot.md`; later turns pass it as
  `build_system_prompt(memories_text=…)`. Edits since are a `<memory_delta>`
  note (+/− line counts and the added lines, ≤2000 chars) on the turn tail —
  in every layout, since a system-prompt copy would defeat the freeze. The
  bypass is at the caller, not inside `_load_memories`.
- **`replay_injected_context`**: `load_and_compact_session` re-joins each past
  user row with its turn's `subliminal` row (`prompt_layout.replay_injected`,
  after P3's `is_history_row` filter) so history replays what was sent. The subliminal rows never reach
  `_prepare_messages_for_harness` (compaction drops the role first), so the
  re-join sits in compaction, not the adapter the plan named.

`_messages_subliminal._split_subliminal(prefetched, text) → (prefix, tail)`
recognises a tail by its opening tag (`prompt_layout.TAIL_TAGS`); the row
shows prefix then tail and records `tail_chars` (omitted when empty). With no
tail it equals the old `_extract_subliminal_prefix`.

**Measurement, always on:** `prefix_miss.record_turn_start` writes
`brain1.turn_start_prefix {iteration, input_tokens, cache_read, reuse,
ttft_ms}` on the first `assistant_message` of each turn from all three
writers — the iteration `prefix_miss` deliberately skips. Rollout is
`system_tail`, then an A/B of `user_tail`; adopt when turn-N+1 first-iteration
`reuse` rises and `eval/run_eval.py` does not fall. `tests/test_prompt_layout.py`.

### P8 — parallel read-only Task fan-out; subagent defaults

- **Fan-out.** `subagents.<type>.parallel_safe: true` (shipped on
  `read-only` only) is read by `mcp_discovery.parallel_safe_task_profiles()`
  into `RunOptions.parallel_safe_task_profiles`. `loop._batch_is_read_only`
  accepts a `Task` call whose `subagent_type` is in that set and that carries
  no `task_id` (a resume's stored profile wins over the argument, and two
  resumes of one id race `claim_history`); no `subagent_type` means
  `general-purpose`, which is not safe. A batch that is **only** such Tasks
  (`_is_task_fanout`) overlaps even with `harness.parallel_tool_calls.enabled`
  off, under the same `max_concurrency`; a Task mixed with a Read waits for
  the general flag like any other batch.
- **Why it is safe.** Not the prompt: `builtin_task` holds a parallel-safe
  child to the `readOnlyHint` set (`_parallel_safe_blocked`, plan mode's
  derivation, re-read per iteration through `disallowed_tools_refresh`
  because the tool universe is recorded when the child's own pool opens).
  Each overlapped call is still its own `_execute_tool_call`, so the D4
  `_meta` grant scope and this iteration's deny list reach every child.
  Kill switch: remove `parallel_safe` (children run sequentially and keep
  only their profile's deny list).
- **Defaults.** A profile with an empty prompt gets `_DEFAULT_SUBAGENT_PROMPT`
  (final message is all the caller sees; concise; absolute paths; say what
  was not verified). Every child gets `build_iteration_anchor(max_turns)`
  (75%/90%, appended). `Task` takes an optional `final_schema` (a JSON Schema
  with `type: object`) → `RunOptions.final_schema`; the result then carries
  `structured` and `structured_error` (a bad schema or a skipped finalizer is
  reported there, never a failed Task). The `read-only` profile's prompt no
  longer names the Bash it is denied (`tests/test_subagent_profiles.py`).
- **Stop/steer from Mission Control.** `POST /api/subagents/{task_id}/cancel`
  and `/steer` (`app/routers/subagents.py`) proxy to the aggregator's
  `POST /subagents/{task_id}/{verb}` (`_subagent_registry.control_request`),
  URL and credential from `app.aggregator_config.subagent_route`. The body
  names the session it acts for and `policy_allows`
  (`orchestrator-session`) decides — 403 for any other session, 404 for no
  active run. `GET /api/subagents` lists the rows; `RunningAgentsPanel` shows
  running children with a stop button acting for each row's
  `parent_session_id`. The policy is not widened: the operator acts as the
  spawning session, the same authority that session's next turn has.
- Pins: `app/harness/tests/test_parallel_dispatch.py` (P8 block),
  `tests/test_task_subagent_answer.py` (default prompt, iteration anchor,
  `final_schema` round trip, read-only child), `tests/test_subagent_profiles.py`,
  `tests/test_task_steering.py` (Mission Control half, over the real
  credential-wrapped routes). Measure: wall time of a 3-way `read-only`
  fan-out and vLLM batch occupancy against the same prompts sequential.
### D2e — the summary arms measure a summary

The first live run of `summary_legacy` vs `summary_persisted` (240k/280k,
tool shape) was 24/24 `error`, and fixed it would still have compared nothing.
`eval/run_compaction_recall_eval.py` (header, SUMMARY ARMS) now:

- **Warm-up**: fitted to the window (`fit_warmup`: JSON chars / 3 per token,
  4k reserve), clipped from the END to a head of whole turns — the part a
  prefix cache can share — or skipped; attempted once per run whatever
  happens, its failure recorded as `warmup.status: error`. It used to be sent
  unclipped at 240k, 400 inside the loop's stream, and be re-sent on every
  overflow-recovery attempt.
- **A shape on which the summary fires** (`--shape conversation`, the default
  for an all-summary arm list): one Read plus a long `architecture/*.md`
  discussion per turn, the planted facts restated in the assistant's reply,
  `--sizes` counting the non-tool rows. Microcompact stays on: turning it off
  instead would hand the unbounded legacy summariser a ~280k-token block, a
  guaranteed 400.
- **Gate**: dropped before the probe unless the summariser ran this turn
  (`summarize_outcome == summarized`), its row survived truncation, and the
  fact is not still verbatim. Rows carry `summary_has` (the facts read off
  the summary text), `summarizer` (calls, wall, chars), `turn_start_wall_s`;
  `--dry` stubs the summarisers. `paired_vs_summary_legacy` in the output.
- **Finding**: legacy can summarise only while the cleared history is within
  ~40k of the threshold — past it its request exceeds the window (conversation
  195k → 249k tokens, 210k → 263k) — so the fair band is conversation
  186k–192k. The eval also arms the safety floor / outbound-content gate
  (`GATE_ARM_POINTS` 9 → 12, with the two other live evals that armed it
  without joining the roster).
- **P3's `memory_flush` arm** is a summary arm on the same shape, sizes and
  gate. Its flush turn (now armed too) is sent `flush_history`: the head of
  whole turns before the first whose turn-start-compacted prompt crosses
  `trigger_fraction` x the threshold — the crossing production flushes at —
  not the whole uncompacted session, which at these sizes is past the window
  exactly as the warm-up was. That head is ~40% of the rows, so depth 0.5 is
  a fact that arrived after the flush (`flush.planted_in_history`).

### P13.4 — one options builder (`app/routers/turn_options.py`)

- **What it replaced.** Four hand-built turn sites — `post_message_stream`,
  `build_ambient_turn`, `build_flush_turn` (P3) and `voice._voice_turn_setup`
  — plus the sync route (P13.6) each built their own `RunOptions`, system
  prompt and hook registry. `SessionSnapshot.load` reads the session's small
  fields once; `build_turn_options(snapshot, body, kind)` returns a `TurnBuild`
  (options, system prompt, hooks, turn tail). Every kind difference is a named
  branch: voice keeps its chat-shaped prompt (no platform, no goal), its
  `extra_body`, no surface and no grant gate; flush keeps memory tools only,
  tool search off, no refresher and no tail; the stream kind alone gets the
  context meter, final schema, effect scope and action review, and skill
  dispatch through `arm_skill_dispatch` after the prefetch.
- **Unchanged, measured.** `tests/fixtures/turn_options_head.json` was recorded
  on base 8ac6f4a8 by `tests/test_turn_options.py` (every `RunOptions` field,
  the system prompt's arguments, the turn tail, hooks in order, the refreshed
  deny list) for 4 kinds x 5 session shapes; the table test holds the builder
  to it. One base artifact was corrected before recording: voice's refresher
  read `app.paths.SESSIONS_DIR` while the rest read the router's, so the first
  recording took plan mode from the wrong directory in a test (same directory
  in production).
- **Parses per POST.** Counted on base: 9 reads of the session JSON for a chat
  POST, 11 for a worker POST, then the meta save. Now 1
  (`SessionSnapshot.aload`, off the loop) plus the save. The gate helpers
  (`_authority_scope_for`, `_ban_automod_for_workers`, `_effect_scope_for`,
  `_final_schema_for`) take the snapshot's identity instead of reading.
- Roster: `GATE_ARM_POINTS` names `turn_options.py` in place of `messages.py`
  and `voice.py` (12 → 11 entries; the D2e evals stay armed and listed). Pins moved: `tests/test_grant_gate_session_path.py`,
  `tests/test_outbound_content_gate.py`, `tests/unit/test_skill_dispatch.py`,
  `tests/test_action_review.py`, `tests/test_tool_effects.py`.

### P13.5 — session writes off the loop; cached small-field reads

- `sessions_io.mutate_session` and `_save_session_meta` run read → `fn` →
  dump (`indent=2` kept) → atomic write in `asyncio.to_thread`, with the
  per-session lock held by the awaiting coroutine. `fn` therefore runs off the
  loop thread and must not touch asyncio objects.
- `sessions_io.read_session_fields(path)` parses the small fields (model,
  platform, source, todos, plan, goal, Inner Voice flags, user-turn count) at
  most once per file version, keyed by `(ino, mtime_ns, ctime_ns, size)`. A file
  younger than 50 ms when read is parsed but not cached (git's racily-clean
  rule; file times tick at the kernel's coarse clock). `_load_session_todos`,
  `_load_session_plan`, `_load_session_goal` and the per-iteration plan-mode
  refresher read through it and return deep copies.
- **Measured** (`eval/measure_session_io_stalls.py`: 1.11 MB synthetic
  session, watchdog waking every 1 ms, a wake-up more than 10 ms late counted
  as a stall; 3 runs each, load average ~35):

  | workload | base 8ac6f4a8 | P13.5 |
  |---|---|---|
  | 40 appends | 40 stalls every run, max 16–25 ms | 0 stalls, max 6–8 ms |
  | 10 stream POSTs | 10 stalls >25 ms every run, max 104–119 ms | 0–2 stalls, max 8–14 ms |
  | 60 plan-mode refreshes | 0–4 stalls, max 6–12 ms, 0.57–0.62 s wall | 0 stalls, max 4 ms, 0.33 s wall |

  What is left is GIL time: `json.loads` holds the GIL for its whole ~4 ms on
  this file, so a parse in a worker thread still delays the loop by that much.
  `json.dumps` with `indent` is the pure-Python encoder and yields. An
  append-only `<sid>.msgs.jsonl` journal (no full rewrite per row) is the next
  lever and its own item.
- Pins: `tests/test_turn_options.py` (one parse per POST, racy-fresh files not
  cached, a cached reader cannot poison the cache, a rewritten file is re-read).

### P13.6 — `POST /api/message` deleted

Callers searched before deleting: the whole repo, `web/src`,
`agent-services/`, `scripts/`, `workers/`, and the vault's skills (only
archived skills from the pre-harness Hermes days name it). No caller: the
one reference was `api.sendMessage` in `web/src/api.ts`, defined and never
called (nor in the built chrome extension), removed with the route
(`tests/test_api_contracts.py` checks every client path is a route). Removed
with it: its usage-row writer (two writers remain in `messages.py`, both in
`_run_turn`), its chat-id mint, and the tests that drove it
(`tests/test_compaction_record.py::test_a_loopback_post_lands_the_same_record`).
Pin: `tests/test_workers_router.py::test_the_sync_message_route_is_gone`.

### P13.1-3 — `run_query` in phases; one dispatch path

- **State.** `app/harness/turn_state.py`: `TurnState` holds what `run_query`
  kept as ~30 locals across iterations, `Iteration` what it re-initialised per
  pass. A pure move — same names, same initial values, same reset points.
- **Phases.** `run_query` (~80 lines) is `_open_turn` → per iteration
  `_prelude` (max_turns + wrap-up, Stop, drain, anchor, disallowed refresh,
  pre-request relief) → `_stream_iteration` (request, D7/overflow/multimodal
  recovery, commit, `assistant_message`, history, hook) →
  `_end_without_tools` (observer inject, echo guard, stop reason) or
  `_dispatch_batch` → `_after_batch` (microcompact) → `_close_turn`
  (finalizer, `result`). A phase says how the loop goes on in `it.flow`
  (`continue` / `break` / fall through). Phases that yield are re-yielded
  through `contextlib.aclosing`, so closing the turn mid-batch still cancels
  the batch's outstanding calls at once.
- **One dispatch path.** `_dispatch_batch` runs every batch; concurrency is
  `parallel_tool_calls_max_concurrency` when the batch qualifies (read-only
  and the flag on, or a P8 Task fan-out), else 1. Calls are admitted lazily in
  wire order — announced, OnEvent fired, `_pre_dispatch`ed — only when a slot
  is free, and an early result holds its slot until yielded, so at 1 the
  stream is `call₁, result₁, call₂, result₂` and a hook for call₂ runs with
  result₁ already in history, exactly as the deleted sequential loop did. At 1
  the MCP call is awaited in the turn's own task (an unexpected raise still
  ends the turn; no contextvar is lost to a copied context); above 1 each call
  is a task, a raise becomes `dispatch_failed`, results are yielded as they
  land. Captions are one wire-order pass (`_account_captions`); history is
  appended as a wire-order prefix as results land, then
  `_reorder_batch_messages` and the image cap run once. `_dispatch_one_tool_call`
  is gone.
- **What changed on purpose, both on formerly-parallel batches only** (the
  flag ships off, so production reaches them only through P8 fan-out): a
  batch wider than the semaphore announces a call when it is admitted, not
  all up front; and a caption nudge that lands on an early result (a denied or
  parse-error call) now reaches history too, as it always did sequentially —
  the old parallel path yielded the nudged event but wrote the un-nudged one.
- **Proof.** `eval/run_harness_replay_diff.py` replays persisted sessions
  through base and HEAD with scripted engine and pool seams and diffs events,
  final history, the session event log and a hook trace; 200 sessions / 236
  turns / 2 arms: zero diffs after steps 1+2, and after step 3 zero in history,
  event log and the `seq` arm, 2 whitelisted interleaves in `par` (both
  batches of 4-5 calls against concurrency 3). A/A clean both times.
  `eval/measurements/harness-replay-diff-2026-09-24.md`.
- Pins: `app/harness/tests/test_dispatch_split.py`
  (`test_a_sequential_batch_pre_dispatches_each_call_after_the_last_result`,
  `test_the_old_per_call_dispatcher_is_gone`), the P13.0 replay tests, and
  `tests/test_tool_choice_eval_cost.py` (the yield-before-dispatch citation now
  names `_dispatch_batch`).

### P9 — programmatic tool calling, `lloyd_rpc` (v1 read-only; build half, ships off)

- **What.** A `Bash` command may call Lloyd's read-only tools itself and print
  only what it needs, so twenty backlog reads cost one engine round trip, not
  twenty. Client: `agent-services/rpc/lloyd_rpc.py` (stdlib; `call`, `map`
  with `concurrency`), `agent-services/bin/lloyd_rpc` on the command line.
  Policy: `app/harness/rpc_policy.py`. Server state: `agent_mcp/_rpc.py`.
- **The turn decides.** `harness.rpc.enabled` (false) is read where the turn
  runs: the loop stamps `lloyd/rpc_deny` on **Bash calls only** (this
  iteration's disallowed set + `FIXED_DENY`: `Bash`, `Task`, `ToolSearch`,
  `automod_*`, `desktop_*`, `_*`), and `prompt_builder` adds one
  `harness_hints` paragraph. Off, the prompt and every call's `_meta` are byte
  for byte what they were. The aggregator serves exactly the Bash calls that
  carry the stamp (`rpc_policy.override` lets an eval arm switch it per
  context with no restart).
- **The env** (`builtin_bash._bash`, both spawn branches): `LLOYD_SESSION_ID`,
  `LLOYD_TURN_ID`, `LLOYD_PARENT_CALL_ID`, `LLOYD_EFFECT_SCOPE`,
  `LLOYD_SURFACE`, `LLOYD_RPC_DENY`, `LLOYD_RPC_DEADLINE` (now + timeout − 5 s;
  a background child gets the 600 s foreground ceiling), `LLOYD_RPC_DEPTH=1`,
  `LLOYD_RPC_URL`, `LLOYD_RPC_TOKEN_FILE` = the aggregator's own credential
  file, only when it is a 0600 regular file of this uid. **A sandboxed
  (bench/eval) session gets none of it**, and bwrap now unsets
  `LLOYD_MCP_TOKEN*` and every `LLOYD_RPC_*` and binds `/dev/null` over the
  token file, so a trial cannot read the secret at all.
- **Server-authoritative.** Spawning a stamped Bash registers a *parent* (id,
  session, turn, scope, surface, deny list, deadline). `main.call_tool` sends
  any request carrying `lloyd/rpc_parent_call_id` through `_rpc_call`: refused
  — above the effect-ledger claim — for an unknown/finished/expired parent,
  depth ≠ 1, a name on the PARENT's recorded deny list (never the script's
  copy), or a tool outside `READ_ONLY`; otherwise re-entered as an ordinary
  call with the parent's `_meta`, so the sandbox, desktop, sessionless-write
  and effect-ledger gates see the parent's session, and a nested `Read`
  satisfies the read-before-edit gate for that session.
  `harness.rpc.allow_mutating` is read and ignored (logged) in v1: a nested
  call skips every harness PreToolUse hook — grant gate, Inner Voice, the
  repetition guard — and read-only tools are the class none of them decides.
  Egress still goes through `egress.guard` inside the tool, under the parent's
  scope. Not containment against code already running as this uid (the
  `aggregator_auth` limit, unchanged): the property is that the honest path
  opens no way around a gate.
- **Record.** Each nested call → `harness.rpc_call {parent_call_id, tool,
  args_digest, ms, is_error[, refused]}` in the parent session's event log;
  the Bash result ends with `[lloyd_rpc: 12 calls — Read×10 Grep×2, 0 errors,
  3.1 s]`. Nothing enters `chat_messages`; nested rows in the chat timeline
  keyed on `parent_call_id` are a UI follow-on.
- **Measure** (`eval/run_rpc_eval.py`, `eval/rpc_tasks.yaml`: board pass over
  20 items, grep+read over the `agent_mcp/` modules, autonomy frontmatter
  reconciliation; arms `off`/`off_rep`/`on`, 5 reps, objective checks from
  disk). Promote if `prompt_tokens_sum` drops ≥ 30% on ≥ 2 of 3 tasks, the on
  pass rate is not below off − A/A, and the on arm has zero Bash timeouts.
  **It runs live, unsandboxed Bash** — a sandboxed session cannot have rpc by
  construction — so it refuses without `--live-bash`; bounded by fixed
  read-only prompts, a read-only-plus-Bash hook, the safety hook and
  background-id dispatch guards. Not run yet; that exception is Alan's call.
- Pins: `tests/test_lloyd_rpc.py` (env from bound meta; unstamped gets none;
  sandboxed gets none and bwrap hides the credential; parent-denied, mutating,
  recursion, expired refused above the ledger; a script cannot shorten the
  deny list; calls logged with the parent id; trailer; rpc Read then Edit
  passes the gate; client deadline refusal; loop stamps Bash only; prompt
  bytewise off; a real shell through the real client and aggregator app).

### D13 — cleanup; docstrings describe the code as it ends up

- **Dead `RunOptions` knobs removed**: `history`, `permission_mode`, `env`
  (Claude Agent SDK leftovers the loop never read) and
  `context_relief_send_max_tokens_reservation` (an A/B nothing wired), with
  their kwargs at every construction site (the options builder, the worker
  seam, autonomy, `ide.py`, the tool-choice eval, the autoresearch runner),
  the two fields they fed in `brain1.options_built`, and
  `harness.context_relief.send_max_tokens_reservation`. `permission_mode` was
  never enforced: Discord's non-owner tier still sends `"default"` in the body
  and the builder no longer reads it — that tier's real restriction was
  always `extra_disallowed` (`agent_mcp/discord_bot.py` left untouched). `agent.permission_mode` stays in config.yaml
  as the value the dashboard displays. Pin:
  `test_harness_unit.py::test_run_options_carries_no_dead_knobs`; the P13.4
  fixture lost those four keys and nothing else.
- **The overflow bound is an option**: `RunOptions.max_context_overflow_recoveries`
  (2) from `harness.context_relief.max_overflow_recoveries`, replacing the
  `TurnState` constant. `test_overflow_recovery.py::test_the_recovery_bound_is_an_option`.
- **Relief report**: rung 1 is named `tool_results:<n>` only when it cleared
  n results (it was appended on every pass); a pass whose other rungs ran and
  freed ~0 still records. Rung 0 (old screenshots) runs only over target,
  like rungs 2–4 — dropping one rewrites a cached message.
- **Discovery errors**: a cross-server tool-name collision in
  `build_tool_list` raises `ToolDiscoveryError` (was `ValueError`); an
  over-64-char tool name is still left out, now with a warning naming it.
- **Doc drift**: `run_query` (the handle wins over `messages`), `options.py`
  and config.yaml's `tool_call_summaries` comment (the caption is stripped
  from the dispatched args and KEPT in the replayed `arguments`),
  `events.tool_call` (absent from `args_dict`, present in `args_json`), the
  thinking window above (turn entry and rung 2), and the system prompt's
  platform line, `Lloyd (Claude Agent SDK)` → `Lloyd (local harness)` — one
  cold prefill per open session after the restart.
- **Red tests on main fixed test/doc-side**: the live bench census
  (`test_bench_split.py`, four new synthetic `bench_014`–`017` files named);
  `architecture/vllm.md` §10 now names the running shape (autocode
  `max_inflight` 2); `test_task_steering.py` clears the close-handoff map it
  fills and `test_mcp_transport.py` judges only what its own spawn adds.
  P13.1-3's `eval/run_harness_replay_diff.py` sets its perturbation kwargs
  after construction instead of splatting them, so the outbound-content roster
  reads it as what it is (a scripted pool, no sender tool reachable).
- **Replay diff, P9 → D13** (30 sessions, 68 turns, both arms): A/A clean;
  A/B history and hook traces identical; 42 turns differ only in relief
  telemetry — a `harness.context_relief` event is no longer written for a
  pass in which no rung acted, and the overflow notice and
  `harness.overflow_recovered` name `tool_results:<n>`. That is the change.

### Closing summary

| id | landed |
|---|---|
| D9 | a85f38e6 |
| X1–X4 | 735c6be3 |
| P13.0 | 8ee4c1d2 |
| D3, D6, D10 | c9246ca2 |
| P7 | e2b26f00 (runner), 09d6d704 (cost side) |
| P10 | dfe18fdf |
| D5 | 02d46ef4 |
| D8 | a19759d2 |
| D1 | 82421732 |
| P11 | d677acec |
| D4 | f75f4004 |
| D7 | d11c4486 |
| D2 | f01e6357; D2e 4764a840 |
| P12 | c83c3912 |
| P4 | ed6d699e |
| D12 | e468a326 |
| P6 | c366ce6e |
| D11 | 9bd457b1 |
| P5 | 7beb765a |
| P3 | b4ac86f6 |
| P1 | 9f41bb74 |
| P8 | 8ac6f4a8 |
| P13.4–6 | 6a866ae1 |
| P13.1–3 | this landing |
| P9 | this landing |
| D13 | this landing |

**Ships on**: every D fix (D1–D13) and X1–X4; P11 telemetry; P12 (discovery
TTL 300 s, per-tool timeouts); P13 (behaviour-preserving, replay-diffed);
P6's max-turns wrap-up (`harness.max_turns_wrapup`, primary only); P8's
fan-out for the `read-only` profile (`parallel_safe`); D10's deny-list
(`non_compactable_tools`); D7's one retry. **Ships in shadow**: P10
(`action_review.mode`, `injection_probe.mode`). **Ships off / as today**:
D2's `compaction.persist_summary`, P3's `compaction.memory_flush.enabled`,
P1's `prompt_layout` (`system_head`, `freeze_memory: false`), P4's memory
index (`memory.render_overflow: render_all`, today's full render), P5's
`skills.index.descriptions`, P6's `echo_guard.mode: nudge`, P9's
`harness.rpc.enabled`, and the general `parallel_tool_calls`.

**Pending measurements and decisions** (each deploys only on a measured gain;
a negative result is a clean `rejected`):

- **P7** — the context-rot run (tonight's window) sets
  `compaction.microcompact.trigger_fraction` via `--decide`.
- **D2** — `eval/run_compaction_recall_eval.py` `summary_legacy` vs
  `summary_persisted`, then `persist_summary: true`.
- **P3** — the same eval's `memory_flush` arm, then `memory_flush.enabled`.
- **P4** — the memory index A/B (`eval/run_memory_trim_ab.py`); on a win, the
  render flip, the MEMORY.md ceiling change and the vault patch.
- **P5** — the skills eval's `desc_push` / `desc_pull` arms, then
  `skills.index.descriptions` and the pull switch.
- **P1** — rollout `system_tail`, then the `user_tail` A/B
  (`brain1.turn_start_prefix`), then `freeze_memory`.
- **P6** — whether `echo_guard.mode: tool_choice` beats `nudge`.
- **P9** — its eval needs a session with live, unsandboxed Bash, which the
  bench/eval sandbox rule (CLAUDE.md, "The vault is protected at the tool
  layer") forbids by construction: Alan's call on how it is run.
- **P10** — a labelling week (200 calls, hand-labelled) before any threshold
  is written and `warn` is considered.

## Cleared tool results: where they go and how they come back (#1514, #1481, #1499)

A cleared result has left the prompt but not the machine. Every rung that drops
tool content spills it first and names the file: rung 1 (microcompact, turn start
and in turn), rung 3 (argument bodies) and rung 4 (truncation), all through
`tool_result_spill.persist_for_compaction` into `sessions/<sid>.tool-results/`.
The transcript row written when the result landed is in `sessions/<sid>.json`
(whole, or a D1 pointer into the same directory). Rung 2 (preserved reasoning) is
the one rung with no in-band handle; its reasoning is persisted as
`role="thinking"` rows in that same record.

- **The free route (#1514).** `tool_result_spill.session_record_route(sid, deny)`
  is the one sentence naming both paths, offering Grep, else Read, else nothing
  (a route the turn's deny list refuses is #1066's false promise).
  `compaction.microcompact.name_session_record` appends it to rung 1's and rung 4's
  markers — **off**: measured on the #600 harness it lowered ambiguous recall
  (−0.25 [−0.45, −0.10] vs `tool_clear`, n = 20) for ~250 chars per marker.
- **Observation stubs (#1481).** `compaction.microcompact.observation_stubs`
  (+ `observation_head_chars`, 400) makes a clear leave `[observation <id> — call
  — N chars cleared …]` plus a verbatim head, never a paraphrase; the id is the
  spill file's stem, and `recall_observation(id)` (`agent_mcp/recall_observation.py`,
  `READ_ONLY`) returns the file — only from the calling session's own spill
  directory, never a path, a symlink out, or another session's id. **Off**: no
  recall gain over `tool_clear` (+0.05/+0.05, ns) for +1.0 s TTFT per turn.
  `loop._open_turn` hides the tool from any turn whose relief writes no stubs, so
  with the switch off the catalog is byte-identical to before.
- **Re-relief is idempotent (on, unconditionally).** `_is_cleared_stub` keeps an
  already-reduced `<persisted-output>` block and a stub out of both selections, so
  a second pass over its own output changes no byte and counts no clear. Before,
  every pass re-reduced every old pointer (one duplicate `[preview dropped …]`
  line, then a clear counted on each pass): 35 of the 60 relief passes naming rung
  1 in `usage.db` freed 0 tokens.
- **Relief events carry what rungs 3 and 4 cut** (`argument_chars_freed`,
  `truncated_chars_freed`), as the usage record already did (#1499 step 1).
- **The #600 harness had a cross-session leak** until 2026-09-25: its stub Grep
  searched every arm's and seed's session record, each planting a different port.
  `tool_clear`'s 12/20 was mostly that; fixed, it is 17/20 and 19/20.

`eval/measurements/cleared-results-2026-09-25.md` has the numbers; the arms are
`self_record`, `observation`, `production_self_record`, `production_observation`,
`rung4`, `rung4_lossy`, `rung4_self_record`. Tests: `tests/test_compaction.py`
(stubs, idempotence, the route), `tests/test_recall_observation.py`,
`tests/test_compaction_recall_eval.py`.
