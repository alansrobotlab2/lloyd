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
| `mcp_pool.py` | one persistent client per MCP server (`MCPPool`), discovery with annotations, per-server lock on stdio; raises `ToolDiscoveryError` on an empty pool |
| `tool_schema.py` | MCP `inputSchema` → OpenAI tool schema; bare-name aliasing for built-ins; `add_summary_param` |
| `tool_search.py`, `tool_search_cache.py` | progressive disclosure: a baseline toolset plus `ToolSearch` when the pool exceeds `threshold_tools` |
| `hooks.py` | `HookRegistry` — pre/post tool-use callbacks (Inner Voice, safety, skill dispatch, the #534 grant gate) |
| `safety.py` | the destructive-Bash patterns; the only thing between an unattended turn and `rm -rf` |
| `policy.py` | authority grants (#534): scopes, tiers, `current_scope` / `current_effect_scope` contextvars |
| `skill_dispatch.py` | `/skill` invocation from a turn |
| `microcompact.py` | intra-turn compaction of old tool results once a turn is long |
| `tool_result_spill.py` | large tool results spilled to `sessions/<sid>.spill/` and referenced by path |
| `finalizer.py` | one extra completion under a JSON schema after the turn (`final_schema`) |
| `run_state.py` | per-run bookkeeping the router reads (activity line, usage) |
| `errors.py` | `ParseError`, `ToolDispatchError`, `MaxTurnsExceeded`, `StreamStalledError` |

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
- **Built-ins are bare names.** `Bash`, `Read`, `Write`, `Edit`, `Grep`,
  `Glob`, `Task` are advertised bare; `disallowed_tools` blocks both the bare
  and the `mcp__<server>__` form.
- **Parallel dispatch is read-only-only and ships off.** A batch overlaps only
  when every call carries `readOnlyHint`; one `Bash` or mutating tool makes
  the batch sequential. History is written in wire order regardless.
- **Every tool carries a `summary` parameter.** It is lifted off `_args_dict`
  before dispatch and kept in the replayed `arguments`, because the replayed
  call is the model's own most recent example of that tool.
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

## Subagents

`Task` runs a nested `run_query` inside lloyd-mcp with its own `RunOptions`
built from `subagents.<type>` in `config.yaml`. It inherits the calling
turn's model and base URL through the MCP request `_meta`
(`lloyd/model`, `lloyd/base_url`); recursion is capped at one level. Every
result carries a `task_id`; passing it back with a follow-up prompt resumes
the stored conversation (8 tasks, 30 minutes, 3M chars, process-scoped).
`agent_mcp/_subagent_registry.py` opens a dashboard row before the run
starts and each exit path closes it with its own status.

## Config

`harness:` in `config.yaml` — stream stall bound, todo/state anchors,
preserved-thinking window, `tool_search`, `parallel_tool_calls`,
`thinking_trace`, `background_recording`, `prefix_miss`, `tool_call_summaries`,
`edit_gates`, `edit_diagnostics`, `change_ledger`, `effect_ledger`.
`compaction:` owns the between-turn and in-turn compaction walls.
`subagents:` owns the `Task` profiles.

## Tests that pin the above

`tests/test_preserved_thinking.py`, `tests/test_mcp_pool_discovery_failure.py`,
`tests/test_tool_call_summaries.py`, `tests/test_thinking_trace_transcripts.py`,
`tests/test_loop_inject_ordering.py`, `tests/test_task_registry_wiring.py`,
`tests/test_transcript_entries.py`.
