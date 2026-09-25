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

## Review 2026-09-24

### D9 — one anchor per iteration; no orphaned tool calls; Stop during prefill

- **A retried iteration runs its head once.** The overflow and multimodal
  recoveries `num_turns -= 1; continue`, which re-entered the loop head with
  the same iteration number, so the notification drain and the state anchor
  ran again and a second copy of the anchor was appended onto a prompt being
  retried *because* it was too big. `prelude_done_for` in `run_query` records
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
`events.trim_discarded` — the chat router and `run_recorder` (X2), plus the
sync `POST /api/message`, `autonomy.run_task`, `_common.run_prompt_on_primary`,
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
- **The ceiling is one constant, not yet flipped.**
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
  turn, is reported pending. `tests/test_memory_index_cap.py`.

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
