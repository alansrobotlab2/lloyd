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

`CLAUDE.md` § "Agent Harness" carries the operational detail and the scar
tissue; this doc is the map.

## Modules

| Module | Owns |
|---|---|
| `loop.py` | the iteration loop: stream → parse → tool dispatch → append → repeat; history shaping (`_assistant_message_for_history`, `_commit_tool_calls`, `_prune_reasoning`) |
| `client.py` | the httpx SSE stream to `/v1/chat/completions`; `read=None`, so stalls are bounded by `stream_chunk_timeout_s`, never time-to-first-byte |
| `options.py` | `RunOptions` — every knob a caller can set; the router and the worker sources build one per turn |
| `events.py` | the `NormalizedEvent` constructors: `system`, `text_delta`, `thinking_delta`, `thinking_done`, `tool_call`, `tool_result`, `assistant_message`, `result`, `stream_raw` |
| `mcp_pool.py` | `MCPPool`: discovery with annotations, dispatch, the process-wide pool cache (`get_or_open_pool`). The HTTP path opens a session per call and is unlocked — the 2026-07-28 core is stateless, so a held connection buys nothing; only the stdio path shares one `ClientSession` and takes a per-server lock. Raises `ToolDiscoveryError` on an empty pool |
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
  `harness.preserve_thinking_iterations`.
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
`edit_diagnostics`, `change_ledger`, `effect_ledger`. The *state* anchor has
no config key — it is a `RunOptions` callback the router builds per turn;
only the todo anchor's interval is configured.
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
