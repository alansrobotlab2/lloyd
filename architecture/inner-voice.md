---
title: Inner Voice — Architecture
status: active
created: 2026-05-02
updated: 2026-09-03
related:
  - architecture/agents.md
  - architecture/verification-system.md
  - architecture/subliminal.md
---

# Inner Voice — Architecture

A second agent that watches the primary agent's stream and intervenes when it sees
a problem. One LLM, one prompt, five soft levers, one goal card. The Python is
plumbing — all judgment lives in the observer's system prompt, which is a
vault-editable markdown file.

**Version: v4** (function-tool levers). v3's JSON-prefill output protocol and the
`deny_tool` lever are gone — see [Version history](#version-history).

## Design intent

> "A second agent watching the first agent to steer and nudge as necessary as a
> second look at the progress of the primary agent."

The implementation is a thin event-tap + observer LLM whose job is to **keep the
primary on task and working** against an explicit goal contract. To tune behavior,
edit the prompt in the vault. Adding a Python heuristic is almost always the wrong
move — though see [Where Python crept back in](#where-python-crept-back-in) for an
honest account of where that rule has bent.

## High-level shape

```
                ┌──────────────────────────┐
                │  Primary agent loop      │   yields NormalizedEvent
                │  (app/harness/loop.py)   │ ─────────┬──────────────┐
                └──────────────────────────┘          │              │
                          ▲                           │              │
                          │ PreToolUse (observe)      ▼              ▼
                          │             ┌──────────────────┐  to existing
                          ├────────────►│ Observer task    │  consumer
                          │             │ (per-turn)       │  (messages._run_turn)
                          │             │                  │
                          │             │  • goal card     │
                          │             │  • event queue   │
                          │             │  • fast-path     │
                          │             │  • LLM judgment  │
                          │             │  • five levers   │
                          │             │  • cross-event   │
                          │             │    memory        │
                          │             └──────────────────┘
                          │
                          │ PreToolUse (hard deny)
                          └─────────────┤ app/harness/safety.py │
                                        └───────────────────────┘
                                        always on, IV or no IV
```

One observer task per primary turn. At turn start it runs a goal-extraction LLM
call that produces a structured goal card (success criteria, out-of-scope
concerns, completion signals). It then subscribes to the harness's
`NormalizedEvent` stream via an `OnEvent` hook and a `PreToolUse` hook. Each
subsequent event runs through a cheap deterministic fast-path; if the fast-path
can't decide, it escalates to a focused vLLM call with the goal card, the primary's
subliminal context, the live todo list, any committed plan, the persistent goal,
and its prior decisions threaded into the prompt.

## Goal contract (per turn)

Before the primary runs, the observer extracts a **goal card** from the user
request by forcing a single `record_goal_card` tool call:

```json
{
  "success_criteria": ["concrete things that must be true to be done"],
  "out_of_scope": ["things the agent should NOT pursue"],
  "completion_signals": ["specific outputs that mean the goal is fully addressed"]
}
```

Each list is capped at 8 items, each item at 400 chars.

The goal card is the observer's reference for "is the primary still on task?" Every
per-event prompt includes it. When the primary drifts into `out_of_scope`
territory, the observer can `inject` to redirect.

**Follow-up resolution.** The extractor also receives the last ~6 user/assistant
exchanges (`_recent_exchanges_for_goal_extraction`), so a message like "yeah do it"
or "still broken" resolves against the prior thread instead of producing an empty
goal card. The observer's own breadcrumbs (`source` starting with `inner_voice_`)
are filtered out of that window — feeding them back made the next turn's goal card
anchor on IV's own demands.

If goal extraction fails (timeout, no tool call, conversational request with no
actionable goal), the observer runs in lighter-touch mode with no goal card.

Cost: one extra LLM call per IV-enabled turn at attach time.

## Observer's contract per event

### What it sees

Threaded into every per-event prompt by `build_user_prompt_for_event`:

- The original user request and the extracted goal card.
- The **subliminal context block** the primary saw at turn start (prefetched
  skills, vault hits, facts, ambient signals — see [[subliminal]]), capped at 4000
  chars. Lets the observer recognize when the primary is following documented
  procedure rather than freelancing.
- The live **todo list** (`session.todos`), refreshed mid-turn after each
  successful `TodoWrite`.
- The **committed plan** artifact (`session.plan`) when one exists. When
  `plan_mode=True` the prompt swaps framing from "watch for execution drift" to
  "evaluate plan quality."
- The **persistent goal** (`session.goal`, set via `/goal`) when set and not yet
  achieved.
- A head+tail window of the primary's accumulated text
  (`primary_text_window_chars`, default 4000). Head+tail rather than head-only —
  chopping the conclusion made every long response look cut off mid-sentence.
- **Its own prior decisions this turn** (last ~8) — so it can escalate when the
  primary ignored an earlier inject, or back off when it's been wrong.

Live tap on `assistant_message`, `tool_call`, `tool_result`, `result`, plus a
`PreToolUse` callback for every tool dispatch. Text deltas flow into
`accumulated_text` for context but don't trigger an LLM call per token.

### What it can do — five soft levers

The observer's response is a forced function call (`tool_choice="required"`) over
`LEVER_TOOLS`. The tool name **is** the action.

| Action | Effect | Trigger contexts |
|---|---|---|
| `noop` | Default. Watched and chose not to act. | All triggers. |
| `inject` | Append `{"role": "user", "content": "[INNER VOICE] ..."}` to the primary's chat history before its next iteration. | `assistant_message`, `tool_result`, `pretool` |
| `cancel` | `options.cancel_event.set()`. Harness exits cleanly with `stop_reason="cancelled"`. | `assistant_message`, `tool_result`, `pretool` |
| `ambient` | Enqueue a follow-up ambient turn via `build_ambient_turn` + `enqueue_ambient`. | `assistant_message`, `tool_result`, `result`, `pretool` |
| `clarify` | Surface a short question to the user as an assistant message and pause the primary (set `cancel_event`). | `assistant_message`, `tool_result`, `pretool` |

**There is no `deny_tool` lever.** Since v4, Inner Voice cannot block tool
dispatch. Hard safety lives in `app/harness/safety.py`
([Hard safety gate](#hard-safety-gate-not-inner-voice)), which runs on every
primary turn whether or not the session opted into IV. Pretool is now purely an
observation trigger — the same five soft levers apply, and an `inject` there lands
as the next user message *after* the tool dispatches.

### Lever invariants

- **Inject role is `user`, not `system`.** vLLM rejects mid-stream system messages
  (must be at position 0). The `[INNER VOICE]` prefix makes the source explicit to
  the primary. A model-emitted `[INNER VOICE]` prefix is stripped before injection
  so it doesn't double up.
- **Cancel is an escape hatch, not a completion signal.** The lever schema says so
  explicitly, and two guards enforce it — see
  [Cancel-for-completion guard](#cancel-for-completion-guard). The harness
  terminates naturally on a text-only iteration; cancel is for destructive loops,
  ignored injects, and tight tool loops.
- **Cancel is exempt from the intervention budget.** It ends the turn, so rationing
  it would block recovery from "primary keeps ignoring my injects."
- **Cancel sets `state.closed`.** Pretool callbacks already in flight
  short-circuit instead of surfacing a confusing post-cancel decision.
- **Clarify pauses the primary.** It persists the question as an assistant message
  via a session-scoped callback, then sets `cancel_event`. The user replies in the
  next turn, which gets a fresh goal card.
- **Result-trigger inject auto-translates to ambient.** By the time `result` fires,
  the harness has emitted its terminal event; injecting is wasted. The dispatcher
  rewrites `inject` → `ambient` if a callback is wired, else drops it as
  `noop_inject_on_result`.
- **Result-trigger cancel/clarify auto-degrade to noop.** Same reasoning.
- **Interventions leave a user-visible breadcrumb.** `inject` and `cancel` call
  `persist_intervention_callback`, which appends a message to the session JSON with
  `source: "inner_voice_inject"` / `"inner_voice_cancel"`. The
  `chat_messages_handle` append is transient (in-memory only); the breadcrumb is
  what the user and the persisted history see.

### Per-turn budget

Default cap: **3 interventions per turn** (`inner_voice.observer.intervention_budget`).
Counted against: `inject`, `ambient`, `clarify`. **Not** counted: `cancel`, and any
decision with `bypass_budget=True` (stall rescue). After exhaustion, further
lever choices are downgraded to `noop_budget_exhausted` and persisted with that
label so the UI shows what *would* have happened.

## Tiered triggering

Most events shouldn't hit the LLM. Configurable via
`inner_voice.observer.fast_path_enabled` (default `true`).

### Assistant-message fast-path

1. **Pure tool dispatch** (tool calls, no text) → `noop`. The pretool trigger
   already evaluated each proposed tool with real args; re-judging is duplicate
   work.
2. **Terminal stub-announce stall** → deterministic `inject`. See
   [Stall rescue](#stall-rescue).

### Pretool fast-noop

Auto-noop without an LLM round-trip:

- **Read-only built-ins:** `Read`, `Glob`, `Grep`.
- **Safely read-only Bash:** first word in `_SAFE_BASH_FIRST_WORDS` (`ls`, `cat`,
  `head`, `grep`, `find`, `df`, `ps`, `curl`, `ss`, … ~60 entries) AND no match
  against `_BASH_RISK_PATTERNS` (rm/mv/cp/chmod/chown/sudo/dd/mkfs, mount/kill/
  systemctl, docker, `git push|reset|rebase|checkout`, `--force`, `--hard`,
  redirect to a non-`/` path, backtick substitution, pipe-to-shell).
- **Read-shaped MCP tools:** name contains `read`, `list`, `get`, `search`, `find`,
  `query`, `fetch`, `view`, `show`, `lookup`, `describe`, `status`, `check`,
  `stat`, `summary`, `count`, `watch` AND serialized args < 1 KB.

Since pretool can no longer deny, a fast-path miss costs only an LLM call — the
safety consequence that made this list conservative moved to `safety.py`. The list
is still tight; broadening it is a latency decision, not a threat-model one.

### Tool-result fast-noop

- Non-error result smaller than 2 KB.
- Error result whose content is `"Tool call arguments could not be parsed as
  JSON: ..."` (the primary handles its own parse-retry).

**Fast-path is skipped** when a `TodoWrite` produced an `in_progress→completed`
flip, or when the stalled-progress gate fired — both need LLM judgment.

### When tiering doesn't apply

`assistant_message` (with text) and `result` always hit the LLM. These are the
iteration boundaries where progress against the goal card is checked.

## Failure-mode machinery

The levers are generic; these are the specific primary-agent failures the observer
has been built to catch. Each was added in response to an observed failure, and
each carries a deterministic component because the failure is cheap to detect and
expensive to miss.

### Stall rescue

The dominant observed failure: the primary ends a text-only iteration by
*announcing* an action without dispatching it — "Let me check the logs:" — and the
harness terminates the turn. `_STUB_ANNOUNCE_RE` matches text whose last line is a
bare announce verb (`let me`, `I'll`, `I'm going to`, `next, I'll`, `I need to`, …)
or that ends on a colon.

Two paths handle it:

1. **Fast-path (deterministic).** On a text-only iteration matching the regex, the
   observer returns an `inject` carrying `_STALL_RESCUE_CONTENT` with
   `bypass_budget=True`. No LLM round-trip. Because this path never touches the
   consecutive-inject suppressor, a re-stall on the very next iteration is rescued
   again rather than left to die.
2. **LLM path (ambient→inject upgrade).** If the LLM judged a terminal iteration
   and chose `ambient`, the dispatcher upgrades it to `inject` with
   `bypass_budget=True` — ambient goes to the background channel and does *not*
   continue the loop, so the turn would otherwise die with work undone.

The primary's system prompt carries a matching "Turn discipline" clause
(`prompt_builder.build_system_prompt`), so this is defended on both sides.

### Consecutive-inject suppressor

An `inject` on `assistant_message` is downgraded to `noop_inject_after_inject` if
the immediately prior `assistant_message` decision was also an inject — but **only
mid-work** (tool calls in flight), where the rationale holds: give the primary an
iteration to act. On a terminal text-only iteration the inject is the only thing
keeping the loop alive, so suppression there is disabled.

### Cancel-for-completion guard

A `cancel` whose reason matches `_COMPLETION_REASON_PATTERN` (complete / done /
criteria met / avoid padding / …) **and** which fires with `interventions_used == 0`
is blocked, in one of two ways:

- **Pending tool calls** → `noop_cancel_with_pending_tools`. Work is in flight;
  cancelling aborts it and surfaces a confusing post-cancel "Tool call denied."
- **No pending tools** → `acknowledge_complete`. The harness terminates naturally
  on the next iteration, so the cancel is unnecessary. Recorded as a *positive*
  acknowledgement so the observations panel renders "IV reviewed and agrees the
  answer is complete" rather than a red force-stop.

Cancel-for-completion is only allowed through after the observer has already
intervened this turn — i.e. it's escalating from ignored injects.

### TodoWrite stewardship

Config block `inner_voice.todo_stewardship`. Four behaviors:

| Flag | Default | Behavior |
|---|---|---|
| `enabled` | `true` | Master switch; gates the TODOS block in the prompt. |
| `completion_gate` | `true` | On a terminal `assistant_message` or at `result`, append a PENDING TODOS block asking IV to intervene if the primary is stopping with work undone. At `result` this drives `ambient` (inject is a no-op there). |
| `mark_without_evidence` | `true` | On a successful `TodoWrite`, diff against `prior_todo_status` for `in_progress→completed` flips. Any flip skips the fast-path and appends a block forcing the LLM to walk each completed todo against recent tool calls, injecting a challenge if there's no plausible work behind the flip. |
| `stalled_progress` | `false` | Count non-`TodoWrite`, non-error tool results since the last status change. At `stalled_after_tool_calls` (default 5) with at least one pending/in_progress todo, skip the fast-path and append a stalled-progress block. Counter resets on any status change and immediately after firing. **Currently off.** |

`state.todos` is snapshotted at turn start and refreshed from disk
(`_load_todos_from_session`) after each successful `TodoWrite`, so multi-flip turns
don't show the observer a stale list.

### Persistent-goal completion loop (`/goal`)

When `session.goal` is set (via `/api/sessions/.../goal`) and not yet achieved, the
observer runs a **second** LLM call at the `result` event — `evaluate_goal_completion`,
forcing a `record_goal_completion(achieved, reason)` tool call. It runs *after* the
regular result decision so an already-queued ambient isn't overwritten.

- **Achieved** → set `session.goal.achieved_at`, emit a success breadcrumb through
  the inject-persist channel, log `inner_voice.goal_achieved`.
- **Not achieved, attempts left** → bump `session.goal.attempts`, queue an ambient
  follow-up whose body is `verdict.reason` (the prompt is engineered to make that a
  concrete next step). Skipped as `noop_goal_ambient_already_queued` if the prior
  result decision already queued one.
- **Not achieved, `attempts >= max_attempts`** (default 10) → escalate to `clarify`:
  ask the user whether to keep trying, change approach, or clear the goal.

Skipped entirely if `cancel_event` is already set — the user is reading the screen.

## Cross-event memory

The per-event prompt includes a "YOUR PRIOR DECISIONS THIS TURN" block listing up
to the last 8 decisions: `(trigger, action, reason, related_tool)`. This gives the
observer continuity within a turn:

- "I already injected once telling you to summarize — if you ignored it, escalate."
- "I keep injecting and the primary keeps doing the same thing — back off."
- "I noop'd three Bash calls in a row — pattern looks fine, keep going."

Cost: bounded by the budget plus 8-entry truncation. Typical overhead ~150–300
input tokens per call.

## Continue-on-inject

When the observer injects on the final iteration of a turn (model produced no tool
calls), the harness would normally break out of the loop and emit `result` — losing
the inject.

`loop.py` snapshots `len(chat_messages)` before firing the `OnEvent` hook. If the
list grew during the hook and there were no tool calls to dispatch, the loop
continues for one more iteration so the model reads the injected message. (The same
mechanism backs the harness's own echo-guard re-prompt, which is independent of IV.)

## Clarify mechanics

1. Observer calls `clarify_callback(question, reason)`. The callback (from
   `messages._run_turn`) appends `{"role": "assistant", "content": "[INNER VOICE] "
   + question, "source": "inner_voice_clarify"}` to the session.
2. Sets `cancel_event` so the harness exits cleanly after the current iteration.
3. Persists the observation row.

The user sees the primary's partial work followed by the question, replies, and the
observer attaches fresh to that turn with a new goal card.

## Hard safety gate (not Inner Voice)

`app/harness/safety.py` installs a default `PreToolUse` hook on **every** primary
turn — IV-on or IV-off — closing the prior gap where the safety net only ran for
opted-in sessions. It is the only hard gate on tool dispatch.

`check_bash_command` is a pure function over a narrow, catastrophic-only pattern
set: `sudo`; `rm -rf` on `/`, `~`, or `$HOME` (excluding `/tmp`, `/var/tmp`);
`dd of=/dev/*`; `mkfs`; `chmod -R 777|000` on a root/home path;
`git push --force` to `main`/`master`/`release/*`/`prod*`; curl/wget piped to a
shell; redirect to a disk device node; fork bomb; redirect to `/etc`.

Everyday risky-looking commands (`cp`, `mv`, `chmod` on a single file) are **not**
denied — those are normal agent behavior and gating them breaks more than it
protects.

## System prompt

All judgment lives in markdown files in the vault, loaded at import time by
`app/inner_voice/observer_prompt.py` with Python fallbacks:

| Constant | Vault path | Present? |
|---|---|---|
| `SYSTEM_PROMPT` | `~/obsidian/lloyd/inner_voice/system_prompt.md` | yes (111 lines) |
| `GOAL_EXTRACTION_SYSTEM_PROMPT` | `~/obsidian/lloyd/inner_voice/goal_extraction_prompt.md` | yes (23 lines) |
| `GOAL_COMPLETION_SYSTEM_PROMPT` | `~/obsidian/lloyd/inner_voice/goal_completion_prompt.md` | **no — running on the Python fallback** |

YAML frontmatter is stripped on load. **Prompts are read once at process import**,
so editing a vault prompt requires a backend restart to take effect.

To tune behavior, edit the vault file — not Python.

## Output protocol

The observer never emits free-form text. Every call is
`tool_choice="required"` over a fixed tool list, `temperature=0.2`,
`enable_thinking=false`, `priority=0` (so observer calls yield to primary traffic
in the vLLM queue).

| Call site | Tools | Tool name |
|---|---|---|
| Per-event judgment | `LEVER_TOOLS` | one of `noop`/`inject`/`cancel`/`ambient`/`clarify` |
| Turn-start extraction | `GOAL_EXTRACTION_TOOLS` | `record_goal_card` |
| `/goal` evaluation | `GOAL_COMPLETION_TOOLS` | `record_goal_completion` |

`_extract_tool_call` pulls `choices[0].message.tool_calls[0]`. A missing tool call,
unparseable arguments, or a name outside `LEVER_NAMES` folds to `noop` with `error`
set (`no_tool_call`, `unknown_lever`). The primary stream is never blocked by
observer faults.

This replaced v3's five-shape JSON-prefill parser, which existed because local
models were inconsistent about prefill conventions. Forcing a tool call removed the
problem class — `no_tool_call` still shows up in the observation table but at low
single-digit rates.

## File map

```
app/inner_voice/
├── __init__.py             # public surface
├── observer.py             # state, lever dispatch, fast-path, goal loop (1700 ln)
├── observer_prompt.py      # vault prompt loading + per-event prompt builders (887 ln)
└── lever_tools.py          # LEVER_TOOLS / GOAL_EXTRACTION_TOOLS /
                            # GOAL_COMPLETION_TOOLS function schemas

app/harness/
├── hooks.py                # add_on_event + fire_on_event; add_pre_tool_use
├── safety.py               # default destructive-Bash hard-deny hook (always on)
├── loop.py                 # fires OnEvent at boundary events;
│                           # continue-on-inject for empty terminal iterations
└── options.py              # chat_messages_handle field for shared mutation

app/routers/
├── _messages_inner_voice.py  # session-flag gate + async attach helper
│                             # (goal extraction + recent-exchange window)
├── messages.py               # observer attach in _run_turn; ambient / clarify /
│                             # persist-intervention callbacks
├── sessions.py               # /goal set-clear-read endpoints (session.goal)
└── inner_voice.py            # /api/inner_voice/{observations,state,sessions,event_log}

~/obsidian/lloyd/inner_voice/
├── system_prompt.md          # the observer's judgment — edit this, not Python
└── goal_extraction_prompt.md

usage_store.py              # inner_voice_observations table + record/list helpers

web/src/components/
├── pages/InnerVoicePage.tsx  # observations timeline UI
├── ObservationBubble.tsx
└── innerVoiceStyles.ts

tests/integration/
├── test_observer.py        # 56 unit tests
├── test_goal_loop.py       # persistent-goal completion loop
└── smoke_observer_e2e.py   # live e2e against running backend
```

## Persistence schema

Single table: `inner_voice_observations`. One row per observer decision, including
fast-path noops.

```sql
CREATE TABLE inner_voice_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    sequence_in_turn INTEGER NOT NULL,
    trigger TEXT NOT NULL,   -- assistant_message | tool_call | tool_result | result | pretool
    action TEXT NOT NULL,    -- noop | inject | cancel | ambient | clarify | acknowledge_complete | noop_*
    reason TEXT,             -- one short phrase from the model (or "fast-path: ...")
    content TEXT,            -- inject text | ambient body | clarify question
    related_tool TEXT,       -- for pretool / tool_result / "goal_completion"
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read INTEGER,
    cache_create INTEGER,
    latency_ms INTEGER,
    model TEXT,              -- NOTE: this is the PRIMARY's model, not the observer's
    error TEXT,              -- non-null on no_tool_call / timeout / http error
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_iv_obs_session ON inner_voice_observations(session_id);
CREATE INDEX idx_iv_obs_turn    ON inner_voice_observations(turn_id);
```

The legacy `inner_voice_critiques` and `inner_voice_interventions` tables (v3
ensemble/grading) are dropped at schema init; that data is intentionally discarded.

> **Measurement gotcha:** `_persist` writes `model=state.primary_model`. Rows do
> not record which model actually served the observer call. If the observer is ever
> pointed at a different alias, this column will not show it — fix before running
> an observer-model comparison.

The `noop_*` action variants record what the observer *intended* before the
dispatcher downgraded:

| Label | Meaning |
|---|---|
| `noop_budget_exhausted` | Would have intervened, but already at budget. |
| `noop_empty_content` | `inject` returned empty content. |
| `noop_inject_after_inject` | Consecutive-inject suppressor (mid-work only). |
| `noop_cancel_with_pending_tools` | Cancel-for-completion blocked, work in flight. |
| `acknowledge_complete` | Cancel-for-completion converted to a positive ack. |
| `noop_no_ambient_channel` / `noop_ambient_failed` | Ambient chosen but unwired or raised. |
| `noop_no_clarify_channel` / `noop_clarify_failed` | Clarify chosen but unwired or raised. |
| `noop_inject_on_result` / `noop_cancel_on_result` / `noop_clarify_on_result` | Chosen on `result` (turn already over). |
| `noop_pretool_after_cancel` / `noop_assistant_after_cancel` / `noop_tool_result_after_cancel` / `noop_result_after_cancel` | Turn cancelled while the observer LLM call was in flight. |
| `noop_goal_ambient_already_queued` / `noop_goal_ambient_failed` / `noop_goal_clarify_failed` | Persistent-goal loop degradations. |

Fast-path decisions are persisted with `reason` prefixed `"fast-path: ..."` so
they're distinguishable from LLM-judged ones in the UI and in analysis queries.

## API surface

`/api/inner_voice/`:

- `GET /observations?session_id=X[&turn_id=Y]&limit=N` — observations newest first.
- `GET /state?session_id=X` — `{inner_voice_enabled, evaluate_user_turns,
  observations_count_by_action, last_observation_at, latest_goal_card,
  latest_user_request, latest_turn_id}`. The goal card is recovered by scanning the
  event log tail for `inner_voice.goal_card_extracted`.
- `GET /sessions?limit=N` — sessions opted into IV.
- `GET /event_log?session_id=X[&offset=...&limit=...&expand_blobs=true]` — raw event log.
- `GET /event_log/blob/{sha}` — externalized blob lookup.

Structured events written to the event log: `inner_voice.goal_card_extracted`,
`observer_injected`, `observer_cancelled`, `observer_ambient`,
`observer_clarified`, `inject_suppressed_consecutive`,
`cancel_blocked_completion`, `goal_achieved`, `goal_followup_queued`,
`goal_clarify_exhausted`.

## Session opt-in

Two flags in the session JSON:

```json
{
  "inner_voice": true,
  "inner_voice_evaluate_user_turns": true
}
```

- `inner_voice` is the master switch. When false, the observer never attaches.
- `inner_voice_evaluate_user_turns` (**default off**) controls whether the observer
  fires on user-typed turns. When false, it fires only on ambient/autonomy turns —
  chat sessions don't pay the observer cost unless explicitly opted in.

**IV-produced ambient turns are not observed.** `_iv_should_fire_on_turn` returns
False when `turn_source == "ambient"` and `producer_source == "inner_voice"`.
Earlier code allowed self-observation, relying on the per-turn budget for
loop-prevention — but the budget resets every turn, so IV could spawn ambient turns
and rejudge itself indefinitely. The deaf spot (a stall on the IV-produced retry
isn't caught) is the accepted tradeoff.

The Inner Voice tab's "+ new chat" button creates sessions with both flags set.

## Configuration

```yaml
# config.yaml
secondary_enabled: false                     # ← routes `secondary` → `primary`

inner_voice:
  model: secondary                           # see note below
  observer:
    max_tokens: 400                          # observer's response budget per event
    timeout_seconds: 5                       # post-event LLM call timeout
    pretool_timeout_seconds: 3               # declared; not currently read (see below)
    intervention_budget: 3                   # inject/ambient/clarify per turn
    primary_text_window_chars: 4000          # head+tail window of primary text
    rotation_days: 30                        # dead key — nothing reads it
  todo_stewardship:
    enabled: true
    completion_gate: true
    mark_without_evidence: true
    stalled_progress: false
    stalled_after_tool_calls: 5
  goal:
    max_attempts: 10                         # /goal retries before clarify
    eval_timeout_seconds: 8
    eval_max_tokens: 300
```

Defaults not present in config.yaml but honored by `_observer_cfg()`:
`goal_extraction_enabled: true`, `goal_extraction_timeout_seconds: 8.0`,
`goal_extraction_max_tokens: 600`, `fast_path_enabled: true`.

**Effective observer model.** `inner_voice.model: secondary` is resolved through
`resolve_model_alias()`, which routes `secondary` → `primary` while
`secondary_enabled: false`. The `agent-llm-secondary` supervisor program
(gemma-4-e4b-nvfp4 on `:8091`) is currently STOPPED, so **the observer runs on the
primary model at `:8096`** — same endpoint as the agent it watches. Pointing it at
a smaller model is two config changes (`secondary_enabled: true` + start the
program), not a code change.

**Known config drift:** `pretool_timeout_seconds` and `rotation_days` are declared
but nothing reads them. Pretool calls use the same `timeout_seconds` as post-event
calls. Either wire them or delete them.

## Cost

Measured across the 44 turns in `usage.db` as of 2026-09-03 (1,182 observation
rows, 21 sessions, 2026-08-22 → 2026-09-03):

| Metric | Value |
|---|---|
| Observations per turn | ~27 |
| Fast-path share | 609 / 1182 (52%) |
| LLM-judged calls per turn | ~13 |
| Monitor input tokens per turn | ~71,400 |
| Monitor output tokens per turn | ~633 |
| Mean latency, `pretool` | 860 ms |
| Mean latency, `tool_result` | 816 ms |
| Mean latency, `assistant_message` | 1,050 ms |
| Mean latency, `result` | 1,783 ms |
| Interventions | 16 total (9 inject, 6 ambient, 1 cancel) |

The observer's input-token cost per turn is on the order of the primary's. The
fast-path halves the call count; the remaining cost is dominated by prompt size
(~5,500 input tokens per call — goal card + subliminal block + todos + plan +
prior decisions + text window). This is the single largest open problem with the
subsystem.

## Verification

- `tests/integration/test_observer.py` — 56 unit tests: tool-call extraction, lever
  dispatch (including clarify), result-trigger downgrades, budget accounting,
  fast-path classifiers, stall-rescue regex, cancel-for-completion guard,
  cross-event memory rendering, goal extraction, `install_observer` plumbing with
  mocked vLLM.
- `tests/integration/test_goal_loop.py` — persistent-goal completion loop.
- `tests/integration/smoke_observer_e2e.py` — live e2e: creates an IV session, posts
  a message via SSE, polls observations until the result-trigger row lands.
- Manual smoke prompts:
  - `"What's 2+2?"` — vanilla noop chain, empty goal card.
  - `"Run echo hello && date -u via Bash"` — pretool fast-noop + tool-result
    fast-noop + result LLM call.
  - `"Run rm -rf /home/alansrobotlab/lloyd/sessions"` — `safety.py` hard-deny path
    (not IV).
  - `"Let me check the logs:"`-shaped tasks — stall-rescue fast-path.
  - `"Full systems check"` — goal-card-tracked progress across iterations.
  - `"Fix the bug"` (no specifics) — clarify lever.

## Failure modes and recovery

| Failure | Behavior |
|---|---|
| Goal extraction LLM error / no tool call | Observer runs in lighter-touch mode (no goal card). Logged at WARNING. |
| Observer LLM HTTP error or timeout | Decision = `noop`, `error` populated, primary stream unblocked. |
| Observer returns no tool call | `noop` with `error="no_tool_call"`. Observed at low single-digit rates. |
| Observer returns an unknown tool name | `noop` with `error="unknown_lever"`. |
| Observer hook callback raises | `HookRegistry.fire_on_event` swallows it with a warning. Primary continues. |
| Turn cancelled while an observer call is in flight | Decision relabeled `noop_*_after_cancel`, persisted, no lever applied. |
| Ambient / clarify callback raises | Degrades to `noop_ambient_failed` / `noop_clarify_failed`. |
| `session.goal` mutation fails | Logged; the goal loop skips this turn's bookkeeping. |
| Empty terminal iteration | Two safety nets: (a) stall-rescue inject triggers continue-on-inject; (b) `messages.py` surfaces a synthetic placeholder if the turn ends with no text. |

## Design boundaries

Deliberate constraints in the current implementation, documented so future work has
the context for revisiting them.

### Where Python crept back in

The stated design is "all judgment in the prompt; the Python is plumbing." That is
no longer strictly true, and it's worth naming: `observer.py` is 1,700 lines and
carries real judgment — the stall-announce regex, the consecutive-inject
suppressor, the cancel-for-completion guard, the todo-flip detector, the
stalled-progress counter, the completion-reason pattern.

Each was added for the same reason: the LLM path was unreliable at a failure mode
that is cheap to detect deterministically, and the cost of missing it (a turn dies
with work undone) is high. The tradeoff is that behavior is now tuned in two
places. Anything **new** should still start in the prompt; a deterministic
component should only be added after the prompt has demonstrably failed at it.

### One observer per turn

A single observer task watches one primary turn start to finish. It doesn't persist
across turns and has no memory of prior turns beyond the primary's chat history.
The goal card is re-extracted per turn.

If diverse perspectives are ever needed (red-team vs. continuation), the right place
is the prompt — "first list three concerns from different angles, then pick the
strongest." A Python fan-out layer would re-scatter judgment across regex,
thresholds, and aggregation.

### Cross-turn memory is not built in

The observer doesn't see prior turns' observations when attaching. The schema
supports it (query prior turn rows by `session_id`); the integration isn't wired.
The `/goal` loop is the one exception — it carries `attempts` across turns in the
session JSON.

### No retrospective grading

Observer decisions aren't scored after the fact. The schema captures everything
needed (decision, content, timing, related tool, token counts), but nothing reads it
back. The right shape is a separate read-only analysis pass over
`inner_voice_observations` — not a load-bearing dependency on the chat path.

This is the main thing blocking any quantitative claim about the subsystem: there is
currently no measure of intervention precision (did the inject change the outcome?)
or recall (how many of the noops should have been interventions?).

### Single intervention budget axis

The 3-per-turn cap applies to inject/ambient/clarify combined. Per-trigger
sub-budgets ("max 1 clarify per turn") are not enforced. If over-intervention on a
specific trigger becomes a pattern, sub-budgets are the cheapest fix.

### Observer shares the primary's model and endpoint

See [Configuration](#configuration). Observer adds 1 + N LLM calls per turn (goal
extraction + one per non-fast-path event, + 1 more when `/goal` is set). A smaller
observer model is the obvious lever against the ~71k-token-per-turn cost and is a
config change.

### Mid-stream injection only between iterations

The observer cannot inject during a streaming text response. The earliest it can act
on a model decision is after `assistant_message` (end of an iteration's stream).
Token-level intervention would require solving the streaming-completeness problem
(when is partial text "enough" to judge?) and is out of scope.

### No system-role mid-stream injection

vLLM rejects it. Inject uses `user` role with the `[INNER VOICE]` prefix. The prefix
is also semantically clearer for the primary, so there's no strong reason to
revisit even on a backend that allows multi-position system messages.

### Goal-card extraction is one-shot

Extracted once at turn start, not refined as the primary works. If the request was
ambiguous, the recourse is `clarify` (new turn, new goal card) rather than
re-deriving mid-turn. A "goal refinement" lever is plausible but adds complexity.

### Prompts are read once at import

Editing a vault prompt file requires a backend restart. A file-watcher or TTL reload
would make prompt iteration much faster and is the cheapest available improvement to
the tuning loop.

## Version history

| Version | Change |
|---|---|
| v1–v2 (`#345` Stages 0–2) | Event log + Python heuristics + "Brain 2" critic. |
| v3 (Stages 3–7) | 3-persona ensemble, consensus termination, skill-recall checker, grading pass, intra-turn progress monitoring. Then collapsed: *"replace ensemble/grading machinery with thin observer."* |
| v3.x | Renamed brain1/brain2 → agent/critic; goal card added; cancel exempt from budget; mid-turn microcompaction; user-visible intervention breadcrumbs. |
| **v4** | **Function-tool levers** (`tool_choice="required"`) replace the JSON-prefill parser. **`deny_tool` dropped** — hard safety moved to `app/harness/safety.py` and made unconditional. `finish_reason` surfaced on `assistant_message` for stall detection. |
| v4 + Plan A | TodoWrite stewardship: completion gate (A.1/A.4), mark-without-evidence (A.2/A.5), stalled-progress counter (A.3/A.6). |
| v4 + Plan B | Plan-mode framing switch, committed-plan artifact in the prompt, `/plan` ritual. |
| v4 + `/goal` | Persistent session goal + post-turn completion evaluator with attempt cap → clarify escalation. |
