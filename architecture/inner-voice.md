---
title: Inner Voice — Architecture
status: active
created: 2026-05-02
updated: 2026-05-03
related:
  - architecture/agents.md
  - architecture/verification-system.md
  - architecture/ambient-context-injection.md
---

# Inner Voice — Architecture

A second agent that watches the primary agent's stream and intervenes when it sees a problem. One LLM,one prompt,five levers,one goal card. The Python is plumbing — all judgment lives in the observer's system prompt.

## Design intent

> "A second agent watching the first agent to steer and nudge as necessary as a second look at the progress of the primary agent."

The implementation is a thin event-tap + observer LLM whose job is to **keep the primary on task and working** against an explicit goal contract. To tune behavior,edit the system prompt. Adding a Python heuristic is almost always the wrong move.

## High-level shape

```
                ┌──────────────────────────┐
                │  Primary agent loop      │   yields NormalizedEvent
                │  (app/harness/loop.py)   │ ─────────┬──────────────┐
                └──────────────────────────┘          │              │
                          ▲                           │              │
                          │ PreToolUse gate           ▼              ▼
                          │             ┌──────────────────┐  to existing
                          └────────────►│ Observer task    │  consumer
                                        │ (per-turn)       │  (messages._run_turn)
                                        │                  │
                                        │  • goal card     │
                                        │  • event queue   │
                                        │  • fast-path     │
                                        │  • LLM judgment  │
                                        │  • five levers   │
                                        │  • cross-event   │
                                        │    memory        │
                                        └──────────────────┘
```

One observer task per primary turn. At turn start it runs a goal-extraction LLM call that produces a structured goal card (success criteria,out-of-scope concerns,completion signals). It then subscribes to the harness's `NormalizedEvent` stream via an `OnEvent` hook and a synchronous `PreToolUse` hook. Each subsequent event runs through a cheap deterministic fast-path; if the fast-path can't decide,it escalates to a focused vLLM call with the goal card and its prior decisions threaded into the prompt.

## Goal contract (per turn)

Before the primary runs,the observer extracts a **goal card** from the user request:

```json
{
  "success_criteria": ["concrete things that must be true to be done"],
  "out_of_scope": ["things the agent should NOT pursue"],
  "completion_signals": ["specific outputs that mean the goal is fully addressed"]
}
```

The goal card is the observer's reference for "is the primary still on task?" Every per-event prompt includes it. When all `completion_signals` are present and the primary is starting another iteration,the observer can `cancel` to stop padding. When the primary drifts into `out_of_scope` territory,the observer can `inject` to redirect.

If goal extraction fails (timeout,parse error,conversational request with no actionable goal),the observer runs in lighter-touch mode: only intervene on destructive or clearly off-task behavior.

Cost: one extra LLM call per IV-enabled turn at attach time. Latency hides inside turn-startup overhead since the primary's first chat-completion is much longer.

## Observer's contract per event

### What it sees

- The original user request and the extracted goal card.
- Live tap on the primary's `NormalizedEvent` stream: `assistant_message`,`tool_call`,`tool_result`,`result`. Text deltas flow into `accumulated_text` for context but don't trigger an LLM call per token.
- Synchronous PreToolUse callback for every tool dispatch.
- **Its own prior decisions this turn** (last ~8) — so it can escalate when the primary ignored an earlier inject,or back off when it's been wrong.

### What it can do — five levers

| Action | Effect | Trigger contexts |
|---|---|---|
| `noop` | Default. Watched and chose not to act. | All triggers. |
| `inject` | Append `{"role": "user","content": "[INNER VOICE] ..."}` to the primary's chat history before its next iteration. | `assistant_message`,`tool_result` |
| `cancel` | `options.cancel_event.set()`. Harness exits cleanly with `stop_reason="cancelled"`. Used both for "going wrong,stop" and "task complete,save effort." | `assistant_message`,`tool_result` |
| `ambient` | Enqueue a follow-up ambient turn via `build_ambient_turn` + `enqueue_ambient`. | `assistant_message`,`tool_result`,`result` |
| `clarify` | Surface a short question to the user as an assistant message and pause the primary (set `cancel_event`). The primary's next user turn will respond to the question. | `assistant_message`,`tool_result` |
| `deny_tool` | Return a permission-deny dict from PreToolUse. Tool dispatch is blocked. | `pretool` only |
| `allow` | Pass-through; tool dispatches normally. | `pretool` only |

### Lever invariants

- **Inject role is `user`,not `system`.** vLLM rejects mid-stream system messages (must be at position 0). The `[INNER VOICE]` prefix makes the source explicit to the primary.
- **Cancel for "task complete" is a feature,not abuse.** When the goal card's success criteria are all addressed,`cancel` short-circuits unnecessary padding. The observer has explicit prompt guidance for this case.
- **Clarify pauses the primary.** It persists the question as an assistant message via a session-scoped callback,then sets `cancel_event` so the harness exits cleanly. The user sees the question in chat and replies via the next user turn.
- **Result-trigger inject auto-translates to ambient.** By the time the `result` event fires,the harness has already emitted its terminal event; injecting into chat history is wasted (no further iteration will read it). When the observer wants to inject on `result`,the dispatcher rewrites it to `ambient` if a callback is wired,otherwise drops it as `noop_inject_on_result`.
- **Result-trigger cancel/clarify auto-degrade to noop.** Same reasoning — turn is over.
- **PreToolUse failures fail-open.** If the observer times out or errors on a pretool gate,the tool dispatches anyway. Latency-safe over deny-safe.
- **Pretool denies don't count against the intervention budget.** Safety is a separate axis from nudges.

### Per-turn budget

Default cap: **3 interventions per turn** (config: `inner_voice.observer.intervention_budget`). After the budget is exhausted,the observer can only `noop` or `allow` — further inject/cancel/ambient/clarify decisions get downgraded to `noop_budget_exhausted` and persisted with that label so the UI shows what *would* have happened.

## Tiered triggering

Most events shouldn't hit the LLM. Each pretool / tool_result event runs through a cheap deterministic fast-path first.

### Pretool fast-allow

These tool calls auto-allow without an LLM round-trip:

- **Read-only built-ins:** `Read`,`Glob`,`Grep`.
- **Safely read-only Bash:** first word in an allowlist (`ls`,`cat`,`head`,`tail`,`grep`,`find`,`df`,`du`,`echo`,`date`,`pwd`,`whoami`,`wc`,`sort`,`uniq`,`diff`,`stat`,`file`,`ps`,`uptime`,`which`,`printf`,`seq`,`env`,`printenv`,etc.) AND no risk patterns (rm/mv/cp/chmod/chown/sudo/dd/mkfs,mount/kill/systemctl,docker,git push|reset|rebase|checkout,`--force`,`--hard`,redirect to non-`/tmp` path,command substitution,pipe-to-shell).
- **Read-shaped MCP tools:** tool name contains `read`,`list`,`get`,`search`,`find`,`query`,`fetch`,`view`,`show`,`lookup`,etc.,AND args are small (<1 KB).

Anything else escalates to the LLM. The fast-path is purely a latency optimization — if it misses a safe tool,we just pay the LLM call. If it wrongly fast-allowed a destructive tool we'd lose safety,so the rules are conservative.

### Tool result fast-noop

These results auto-noop without an LLM round-trip:

- Non-error tool result smaller than 2 KB.
- Error result whose content is `"Tool call arguments could not be parsed as JSON: ..."` (the primary handles its own parse-retry; observer doesn't need to inspect).

Larger results,real errors,and unexpected payloads escalate to the LLM.

### When tiering doesn't apply

`assistant_message` and `result` events always hit the LLM. These are the iteration boundaries where progress against the goal card is checked.

Configurable via `inner_voice.observer.fast_path_enabled` (default `true`).

## Cross-event memory

The observer's per-event prompt includes a "YOUR PRIOR DECISIONS THIS TURN" block listing up to the last 8 decisions: `(trigger,action,reason,related_tool)`. This gives the observer continuity within a turn:

- "I already injected once telling you to summarize — if you ignored it,escalate to cancel."
- "I keep injecting and the primary keeps doing the same thing — maybe my read is wrong,back off."
- "I allowed three Bash calls in a row — pattern looks fine,keep allowing."

Cost: extra prompt tokens,bounded by the intervention budget plus 8-entry truncation. Typical overhead ~150–300 input tokens per call.

## Continue-on-inject

When the observer injects on the final iteration of a turn (model produced no tool calls),the harness would normally break out of the loop and emit `result`. That would lose the inject — the model would never get to read it.

The harness handles this by snapshotting `len(chat_messages)` before firing the OnEvent hook. If the list grew during the hook (i.e. the observer injected) and there were no tool calls to dispatch,the loop continues for one more iteration so the model sees the injected message and can respond.

## Clarify mechanics

When the observer chooses `clarify`:

1. Calls `clarify_callback(question,reason)`. The callback (provided by `messages._run_turn`) appends an assistant message to the session: `{"role": "assistant","content": "[INNER VOICE] " + question,"source": "inner_voice_clarify"}`.
2. Sets `cancel_event` so the harness exits cleanly after the current iteration.
3. Persists the observation row.

User experience: the chat shows the primary's partial work (if any),followed by an assistant message with the clarification question. The user types a reply,which becomes the next user turn — the observer attaches to that turn fresh,with a new goal card derived from the user's clarification.

## System prompt

All judgment lives in [`app/inner_voice/observer_prompt.py`](app/inner_voice/observer_prompt.py):

- `GOAL_EXTRACTION_SYSTEM_PROMPT` — used for the one extraction call at turn start.
- `SYSTEM_PROMPT` — used for every per-event observer call. Encodes the five levers,when to use each,the goal-card progress check,Bash safety rules,and the "WHEN NOT TO INTERVENE" guardrails (mid-thought,tool-only iterations are normal,substantive answers don't need padding nudges,etc.).

To tune behavior,edit the prompt — not Python.

## JSON parsing

The observer's response is a single JSON object: `{"action":"...","reason":"...","content":"..."}`. Local models are inconsistent about prefill conventions,so the parser handles five shapes:

1. Complete JSON object (model ignored the `{"action":` prefill).
2. Prefill continuation (`"value","field":...}`).
3. Prefill continuation with dropped opening quote (`value","field":...}`).
4. JSON object embedded in surrounding prose.
5. Bare action word (`noop` or `noop"}`).

The goal-card extraction has a parallel parser keyed on the `{"success_criteria":` prefill. On total parse failure the observer falls back to `noop` (or `allow` for pretool,or `None` for goal extraction) and logs the raw response. The primary stream is never blocked by parse failures.

## File map

```
app/inner_voice/
├── __init__.py             # public surface
├── observer.py             # client + lever dispatch + parser + fast-path
└── observer_prompt.py      # SYSTEM_PROMPT + goal-extraction prompt
                            # + per-event prompt builders

app/harness/
├── hooks.py                # add_on_event + fire_on_event
├── loop.py                 # fires OnEvent at boundary events;
│                           # continue-on-inject for empty terminal iterations
└── options.py              # chat_messages_handle field for shared mutation

app/routers/
├── _messages_inner_voice.py  # session-flag gate + async attach helper
│                             # (runs goal extraction before observer install)
├── messages.py               # observer attach in _run_turn;
│                             # provides ambient + clarify callbacks;
│                             # synthetic placeholder for empty terminal
│                             # iterations even when IV is off
└── inner_voice.py            # /api/inner_voice/{observations,state,sessions,event_log}

usage_store.py              # inner_voice_observations table + record/list helpers

web/src/components/pages/
└── InnerVoicePage.tsx      # observations timeline UI

tests/integration/
├── test_observer.py        # 45 unit tests
└── smoke_observer_e2e.py   # live e2e against running backend
```

## Persistence schema

Single table: `inner_voice_observations`. One row per observer decision (including fast-path noops/allows).

```sql
CREATE TABLE inner_voice_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    sequence_in_turn INTEGER NOT NULL,
    trigger TEXT NOT NULL,-- assistant_message | tool_call | tool_result | result | pretool
    action TEXT NOT NULL,-- noop | inject | cancel | ambient | clarify | deny_tool | allow | noop_*
    reason TEXT,-- one short phrase from the model (or "fast-path: ...")
    content TEXT,-- inject text | ambient body | clarify question | deny reason
    related_tool TEXT,-- for pretool / tool_result
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms INTEGER,
    model TEXT,
    error TEXT,-- non-null on parse_failed / timeout / http error
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_iv_obs_session ON inner_voice_observations(session_id);
CREATE INDEX idx_iv_obs_turn    ON inner_voice_observations(turn_id);
```

The `noop_*` action variants record what the observer *intended* before the dispatcher downgraded:

- `noop_budget_exhausted` — would have intervened,but already at budget.
- `noop_empty_content` — `inject`/`ambient` returned empty content.
- `noop_no_ambient_channel` — `ambient` chosen but no callback wired.
- `noop_ambient_failed` — ambient enqueue raised.
- `noop_no_clarify_channel` — `clarify` chosen but no callback wired.
- `noop_clarify_failed` — clarify callback raised.
- `noop_inject_on_result` / `noop_cancel_on_result` / `noop_clarify_on_result` — chosen on `result` trigger (turn already over).

Fast-path decisions are persisted with `reason` prefixed `"fast-path: ..."` so they're distinguishable from LLM-judged ones in the UI.

## API surface

`/api/inner_voice/`:

- `GET /observations?session_id=X[&turn_id=Y]&limit=N` — observations newest first.
- `GET /state?session_id=X` — `{inner_voice_enabled,evaluate_user_turns,observations_count_by_action,last_observation_at}`.
- `GET /sessions?limit=N` — sessions opted into IV.
- `GET /event_log?session_id=X[&offset=...&limit=...&expand_blobs=true]` — raw event log.
- `GET /event_log/blob/{sha}` — externalized blob lookup.

## Session opt-in

Two flags in the session JSON:

```json
{
  "inner_voice": true,
  "inner_voice_evaluate_user_turns": true
}
```

- `inner_voice` is the master switch. When false,observer never attaches.
- `inner_voice_evaluate_user_turns` controls whether observer fires on user-typed turns. When false,observer only fires on ambient/autonomy turns.

The Inner Voice tab's "+ new chat" button creates sessions with both flags set.

## Hook integration

`HookRegistry` (in `app/harness/hooks.py`) exposes `add_on_event(cb)` / `fire_on_event(evt)`. The harness calls `fire_on_event` after yielding the four boundary events (`assistant_message`,`tool_call`,`tool_result`,`result`). Text deltas don't fire the hook — observer accumulates them via the `assistant_message` event's text field.

`PreToolUse` is the existing synchronous hook; the observer registers itself with `matcher=None` so it sees every tool dispatch.

## Failure modes and recovery

| Failure | Behavior |
|---|---|
| Goal extraction LLM error | Observer runs in lighter-touch mode (no goal card). Logged at WARNING. |
| Goal extraction parse failure | Same as above — best-effort. |
| Observer LLM HTTP error | Decision = `noop` (or `allow` for pretool),`error` populated,primary stream unblocked. |
| Observer LLM timeout | Same as above. Pretool default timeout = 3s,post-event default = 5s,goal extraction = 8s. |
| JSON parse failure | Decision = `noop`/`allow`,raw response logged,observation persisted with `error="parse_failed"`. |
| Observer hook callback raises | `HookRegistry.fire_on_event` swallows the exception with a warning log. Primary stream continues. |
| `chat_messages` 400 from vLLM | Should never happen — inject role is `user`,not `system`. If it does,the harness's main `try/except` block catches it and persists whatever content existed. |
| Empty terminal iteration | Two safety nets: (a) observer's `inject` on `assistant_message` triggers continue-on-inject in the harness; (b) [`messages.py`](app/routers/messages.py) surfaces a synthetic placeholder if the turn ends with no text. |
| Clarify callback raises | Decision degrades to `noop_clarify_failed`; primary continues unchanged. |

## Configuration

```yaml
# config.yaml
inner_voice:
  model: primary                            # observer uses the same vLLM endpoint
  observer:
    max_tokens: 400                          # observer's response budget per event
    timeout_seconds: 5                       # post-event LLM call timeout
    pretool_timeout_seconds: 3               # PreToolUse-gate LLM call timeout
    intervention_budget: 3                   # max interventions per turn
    primary_text_window_chars: 4000          # window of accumulated primary text in observer prompt
    goal_extraction_enabled: true            # run goal extraction at turn start
    goal_extraction_timeout_seconds: 8       # one-shot extraction timeout
    goal_extraction_max_tokens: 600
    fast_path_enabled: true                  # cheap pretool/tool_result deterministic checks
```

Tighter timeouts on pretool because it gates tool dispatch. Most observer calls land in 150–400ms; pretool denies are typically 250–600ms.

## Verification

- `tests/integration/test_observer.py` — 45 unit tests covering JSON parsing,action normalization,each lever (including clarify),result-trigger downgrades,the HTTP layer with mocked vLLM,end-to-end `install_observer` plumbing,goal extraction,fast-path classifiers,cross-event memory rendering.
- `tests/integration/smoke_observer_e2e.py` — live e2e: creates an IV session,posts a message via SSE,polls observations until the result-trigger row lands,verifies enabled state and turn-end coverage.
- Manual smoke prompts that exercise specific paths:
  - `"What's 2+2?"` — vanilla noop chain with empty goal card.
  - `"Run echo hello && date -u via Bash"` — pretool fast-allow + tool result fast-noop + result LLM call.
  - `"Run rm -rf /home/alansrobotlab/lloyd/sessions"` — pretool deny path.
  - `"Full systems check"` — exercises goal-card-tracked progress with multiple iterations.
  - `"Fix the bug"` (no specifics) — exercises clarify lever (genuine ambiguity).

## Design boundaries

These are deliberate constraints in the current implementation. Documented here so future work has the context for revisiting them.

### One observer per turn

A single observer task watches one primary turn from start to finish. It does not persist across turns; it has no memory of prior turns beyond what's already in the primary's chat history. The goal card is re-extracted per turn (the user's clarification on turn 2 implies a different goal than turn 1).

If diverse perspectives are ever needed (red-team vs. continuation,safety vs. progress),the right place is the prompt — e.g. "first list three concerns from different angles,then pick the strongest." Adding a Python orchestration layer to fan out parallel personas would re-introduce judgment scattered across regex,thresholds,and aggregation logic instead of one editable prompt.

### Cross-turn memory is not built in

The observer doesn't see prior turn observations when attaching to a new turn. Each turn is a fresh start.

If we want patterns like "the observer noticed a user-frustration trend across turns and adjusts tone," that requires persisting observer-level state across turns and threading it into the goal-extraction prompt. The schema supports it (just query prior turn rows); the integration is just not wired today.

### No retrospective grading

Observer decisions aren't scored after the fact. The persistence schema captures everything needed (raw decisions,content,timing,related tool),but it's not wired into the chat path.

If we later want intervention-quality telemetry,the right shape is a separate read-only analysis pass over `inner_voice_observations` — not a load-bearing dependency on the chat path.

### Single intervention budget axis

The 3-per-turn cap applies to inject/cancel/ambient/clarify combined. Per-trigger sub-budgets (e.g. "max 1 cancel per turn") are not enforced.

If observer over-intervention on a specific trigger becomes a pattern in the wild,sub-budgets are the cheapest fix.

### Observer uses the same model as the primary

Same vLLM endpoint,same model alias. Observer adds 1 + N LLM calls per turn (one goal extraction + one per non-fast-path event). The fast-path keeps N small in practice (~2–4 LLM calls per typical multi-iteration turn).

A separate smaller/faster observer model is a reasonable future direction if observer latency starts dominating tool dispatches. Endpoint resolution lives in `_resolve_endpoint()` in `observer.py` and reads `inner_voice.model` from config — adding a separate model alias is a config change,not a code change.

### Mid-stream injection only between iterations

The observer cannot inject during a streaming text response. The earliest it can act on a model decision is after the model emits `assistant_message` (end of an iteration's stream). Token-level intervention would require taking on the streaming completeness problem (when is partial text "enough" to judge?) and is out of scope.

### No system-role mid-stream injection

vLLM rejects it. Inject uses `user` role with the `[INNER VOICE]` prefix. If we ever migrate to a backend that allows multi-position system messages,the role choice could be revisited — but the prefix is also semantically clearer for the primary,so there's no strong reason to.

### Goal-card extraction is one-shot

The goal card is extracted once at turn start and not refined as the primary works. If the user's request was ambiguous,the observer's recourse is `clarify` (ask the user to disambiguate,get a new turn with a new goal card) rather than re-deriving the goal mid-turn.

A future "goal refinement" lever — observer revises the goal card based on what it's learning — is plausible but adds complexity. Single extraction is sufficient for the current failure modes.

### Fast-path is conservative by design

The deterministic pretool/tool_result fast-path is purely a latency optimization. False negatives (escalating a benign call to LLM) cost an LLM round-trip; false positives (fast-allowing a destructive call) cost safety. The Bash risk-pattern regex and tool-name keyword lists are intentionally tight; broadening them needs an explicit threat-model revisit.
