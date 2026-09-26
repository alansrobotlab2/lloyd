---
title: Inner Voice — Architecture
status: implemented
created: 2026-05-02
updated: 2026-09-24
related:
  - ~/obsidian/knowledge/software/lloyd-inner-voice-history-to-v5.4-2026-09-24.md
  - architecture/harness.md
  - architecture/background-runs.md
---

# Inner Voice — Architecture

Inner Voice is three separable things that used to share one name and one
opt-in. Since the IV plan (R1–R5, 2026-09-24) they are separate:

1. **Senses** — deterministic turn guards, free, on **every** turn
   (`app/harness/turn_guards.py`).
2. **One terminal review** — an opt-in LLM second reader that judges, at the
   moment a turn is about to end, whether it delivered what the user asked
   for (`app/inner_voice/observer.py`).
3. **A measurement loop** that can say whether (2) helped: an on/off A/B, a
   human label on each intervention, a tracked corpus, a weekly outcome score.

The history through v5.4 — five levers, per-event triggers, the unattended
profile, every incident that shaped them — is in
`~/obsidian/knowledge/software/lloyd-inner-voice-history-to-v5.4-2026-09-24.md`. This doc is the current shape only.

## Why it is shaped this way

The evidence the plan read (surviving `usage.db` rows 09-22 → 09-24, 268
`[INNER VOICE]` lines in the transcripts recovered after the 09-22 wipe):

- **Every useful intervention was "the primary stopped before finishing"**:
  a stub announce, a search loop, a subagent that returned nothing, a
  delivered answer with the todo list left open. All cheap to detect without
  a model. They lived inside the observer, so the 09-12 cut that took the
  observer off every worker took them off too.
- **The model-judged observer earned about one useful chat intervention a
  week**, every one a terminal-boundary judgment ("deliver now", "answer the
  actual question"). Its one confident mid-turn scope judgment (08-30,
  `20260830_182633_ive386`) injected three times and cancelled the turn the
  user asked for.
- **Cost was in triggers with zero yield**: `result` 740k input tokens and
  `tool_result` 186k for no interventions, in a window where the whole
  observer cost ~62k tokens a turn.
- **The observer judged fragments** — 1,200 chars of text, 300 chars of a
  tool result — which is the surface its confabulations grew from.
- **Nothing had ever measured whether it helped.**

## 1. Turn guards — the senses

`install_turn_guards(hooks, *, session_id, turn_id, platform, source,
chat_messages_handle, persist_intervention_callback)` registers one PreToolUse
and one OnEvent callback. Idempotent per registry (`hooks.turn_guards`): a
second call fills in what the first did not know. Installed at:

| path | where |
|---|---|
| chat: stream, ambient, voice | `app/routers/messages.py::_run_turn` (all three routes pass through it) |
| chat: sync route | `app/routers/messages.py` beside the safety hook |
| direct worker turns | `workers/sources/_common.py::_worker_run_options` |
| scheduled tasks | `autonomy.py::run_task` |
| `Task` subagents | `agent_mcp/builtin_task.py`, after the skill deliverer and grant gate |
| any observed turn | `install_observer` calls it too, so a bare registry still gets them |

The senses (pure predicates in `app/inner_voice/guards.py`):

| guard | fires when | says |
|---|---|---|
| `stall_rescue` | a text-only iteration announces an action and stops (`is_terminal_stall`) | chat: do it or deliver; worker: do it, no report; worker with a round open: commit and gate |
| `repetition` | the same target queried `threshold`+1 times inside `window` calls (`repetition_verdict`, ambient terms and polling tools excluded) | names the terms; claims nothing about results |
| `failure_payload` | a non-error result whose head says `[stopped: max_turns]`, a timeout, or an empty response — once per tool name per turn | do not treat it as a result |
| `todo_gate` | a turn that called `TodoWrite` ends with its own items still open — once per turn | mark them or keep working; do not restate the answer |
| `round_open` | an unattended turn ends with an automod round it opened still open — once per turn | commit, gate, land or abort |
| `capability_fault` | the terminal text carries a tool call written as prose (`looks_like_prose_tool_call`) | **no inject**: `logger.error`, a `harness.capability_fault` event, and a guardian `announce(level="warning")` at most every 30 min — more text cannot supply a missing tool pool |

Mechanics:

- **An inject reaches the loop through `HookRegistry.bind_run`.** `run_query`
  binds the list it actually reads; a caller with no `chat_messages_handle`
  has only a private copy otherwise. The loop continues a terminal iteration
  when a hook grew that list, drops the inject under the context floor, and
  `_reorder_batch_messages` puts a mid-batch append back in wire order.
- **Capped**: `deterministic_inject_budget` (5) fires a turn; past it a fire
  is recorded as `noop_deterministic_budget_exhausted`.
- **Recorded** as an `inner_voice_observations` row with `safeguard` = the
  guard's name and `model` NULL, plus an `inner_voice.observer_injected`
  event with `deterministic: true` and `guard` — so `iv_grade` and
  `iv_outcome_score` read them as they read the old deterministic rows. On a
  chat turn the router's breadcrumb callback writes the `[INNER VOICE]` line
  into the session, where the user sees it.
- Config `inner_voice.turn_guards`: `enabled` and one switch per guard, plus
  `repetition_window`, `repetition_threshold`, `repetition_exempt_tools`,
  `deterministic_inject_budget`; the old `inner_voice.observer.*` keys still
  read as fallback.

## 2. The observer — one terminal review

Opt-in per session (`inner_voice` + `inner_voice_evaluate_user_turns`, or the
A/B arm below); off for every worker source and, by default, every scheduled
task. `app/routers/_messages_inner_voice.py::attach_observer_for_turn` wires it.

### When it is called

| event | what happens |
|---|---|
| terminal `assistant_message` (no tool calls) | **the review** — synchronous, because the loop decides whether to continue by whether the list grew. Skipped when a turn guard already injected on this iteration, when the text is a prose tool call, or when the context is under the loop's floor. |
| non-terminal `assistant_message` | a **mid-turn review**, async, only when due: every `review_interval(max_turns)` iterations (a sixth of the budget, 4–12; `review_every_iterations` pins it), at a silent streak (`silent_iterations_before_review`), or when a todo flip / stalled-progress block is waiting |
| `pretool` | no judgment, no row — the call is appended to the signature ring and the trajectory |
| `tool_result` | no judgment, no row — the trajectory entry gets its outcome; todo flips and stalled progress become blocks for the next review |
| `result` | no judgment. The `/goal` evaluator still runs here; it has a real loop behind it |

### What it reads

`build_review_summary`: the **trajectory** — one line per tool call this turn:
iteration, tool, the primary's own caption (or the argument preview when it gave
none), and the harness's outcome (`ok 1.2kB`, `ERROR`, `returned, but reports it
did not complete`) — then the iteration itself: the terminal text in full
(`terminal_text_window_chars`, 8000, head+tail), the goal-card walk, the
pending-todos block, and any waiting mark-without-evidence or stalled-progress
block. Plus, as before: the user request, the goal card, todos, plan, `/goal`,
subliminal context, this turn's decisions and earlier turns' interventions,
iteration and context pressure notes.

Tool output that reaches a prompt is framed `<untrusted_tool_output>` and the
prompt states the rule (`UNTRUSTED_RULE`): it can carry instructions; they are
not addressed to the observer.

### Three levers

`app/inner_voice/lever_tools.py`: `noop`, `inject`, `cancel`, forced with
`tool_choice="required"`. `ambient` and `clarify` are no longer offered (5 uses
all time); the `/goal` loop still uses the ambient and clarify channels itself.

Decision guards (`_apply_decision_guards`), in order:

- consecutive-inject suppression and the inject cooldown (unchanged);
- **one scope inject per turn** (`_is_scope_judgment`) — a second is
  `noop_scope_repeat`;
- the unread-inject and cancel-for-completion downgrades (unchanged);
- **cancel only on what can be observed** (`_cancel_is_observable`): a
  destructive loop, or a loop reason on a turn where the repetition guard has
  already fired. Anything else is `noop_cancel_not_observable`.

The discretionary budget (`intervention_budget`, 3) is unchanged; guard fires
never spend it.

### The call

Thinking on (`inner_voice.observer.thinking`, `thinking_max_tokens` 2048,
`thinking_timeout_seconds` 20). Measured 2026-09-24 on the live primary with an
observer-shaped 5k-token prompt: 1.2 s without, 1.6–1.9 s with, same verdicts.
Served by `inner_voice.model` (pinned `primary`) at the watched turn's priority,
one step lower on an unattended turn. The observer must not be weaker than the
primary it watches — the 09-03 4B episode (40% intervention rate against 1.7%)
is the negative result.

### Goal card

Extracted **beside** iteration 1, not before it (it used to add up to 8 s to
the first token). The extraction is one of the observer's pending tasks, so the
terminal review drains it first. `_goal_source_text` reads the user's own words:
system-injected blocks are stripped, and an ambient turn takes its goal from
the last message the user typed (the 08-30 card was built from a
notification). The primary is shown the card by `goal_card_anchor`, composed
into `options.state_anchor` after the turn's own anchors: first at the
iteration boundary after it exists, then every
`harness.todo_anchor_interval_iterations` (IV plan R4 — a weaker primary
forgets the contract faster).

### Prompt

`~/obsidian/lloyd/inner_voice/system_prompt.md` (v5, "one job"), with an
embedded fallback in `observer_prompt.py`. The vault is live and shared: an
edit lands at once, so a prompt change that assumes new code ships with it.

## 3. The measurement loop

- **On/off A/B** (`app/inner_voice/ab.py`, `inner_voice.ab`, ships off). A new
  chat created inside `[start, end)` is assigned an arm by
  `sha256(name:session_id)` against `fraction_on`, at file creation in
  `sessions_io._save_session_meta`, as the same `inner_voice` /
  `inner_voice_evaluate_user_turns` flags every reader already honours, plus
  `inner_voice_ab: {experiment, arm, assigned_on}` (also on `/state` as
  `ab_arm`). Sessions created with explicit flags and worker sessions are never
  enrolled. `scripts/iv_ab_report.py` compares the arms from the event log:
  bad-stop rate, tool-error rate, correction proxy (z against the other arm),
  median iterations and duration; a session flipped by hand is `crossed_over`
  and left out. **This is the acceptance test for the concept.**
- **Human labels**: thumbs on `ObservationBubble` for inject / cancel /
  ambient / clarify rows → `POST /api/inner_voice/observations/{id}/verdict`
  → `verdict`, `verdict_at`.
- **Tracked corpus** `eval/iv/` (`scripts/iv_corpus.py`): `seed` wrote the 270
  recovered lines (242 repetition, 26 model-written, 2 cancels; 20 of the 26
  carry the 09-24 hand reading: helped / unactionable / obsolete / harmful /
  low_value / failed); `export` appends thumbed rows from usage.db.
- **Observer-model replay** (`scripts/iv_observer_model_eval.py`): each labelled
  case becomes one terminal review against a chosen model and prompt. Cases
  with no recorded terminal text are not scored (they replay as an empty
  stop). The recovered corpus keeps text for only 2 of its labelled cases, so
  the noop side waits on thumbed rows. The secondary-as-observer run needs
  GPU 2, which serves djev; it is an attended window, not a round.
- **Weekly outcome score**: `scripts/iv_metrics_record.py` attaches
  `outcome_score` (a child `iv_outcome_score.py --json` over 7 days) to the
  nightly metrics row once a week. The scorer leaves out the six test-suite
  sessions (`TEST_SESSION_IDS`) whose logs hold 580 of the 583
  `observer_injected` events on disk.
- `scripts/iv_grade.py` is unchanged — proxies, and it says so.

## Persistence

`inner_voice_observations` (usage.db): `id, session_id, turn_id,
sequence_in_turn, trigger, action, reason, content, related_tool, input_tokens,
output_tokens, cache_read, cache_create, latency_ms, model, error, created_at,
safeguard, verdict, verdict_at`. `model` is what served the observer's call,
NULL for a guard. `safeguard` names the deterministic rule that decided, NULL
for the model. Turn guards and the observer share one `sequence_in_turn`
counter per turn. Rows are written off the event loop.

Event log: `inner_voice.observer_injected` (with `deterministic`, `guard`),
`inner_voice.goal_card_extracted`, `inner_voice.goal_achieved`,
`harness.capability_fault`, plus the observer's suppression events.

## API

- `GET /api/inner_voice/observations?session_id=&turn_id=&limit=`
- `POST /api/inner_voice/observations/{id}/verdict` `{"verdict": "up"|"down"|null}`
- `GET /api/inner_voice/state?session_id=` — flags, `ab_arm`, counts, latest goal card
- `GET /api/inner_voice/events` — the session event log

## Retired (IV plan R5)

- The observer's **unattended profile**: `_is_unattended`, the deterministic
  terminal words, the unattended cancel gate, the PLATFORM note. The words a
  worker turn needs live in `guards.py` and the turn guards speak them.
- **`pretool` rows** on every tool call (102 of 350 rows in the window said
  only "observation-only").
- **`automod.require_inner_voice`** now defaults false: since 09-12 it refused
  only chat-driven rounds, and every turn is recorded and guarded now.
- The `result` judgment, `tool_result` sampling, `ambient` / `clarify` as
  model levers, `unattended_*` sampling config.

## Workers: back on for autocode and autotriage (2026-09-25)

Off for every worker from 2026-09-12 on three incidents: #874 abandoned at
iteration 38 on "deliver the final report now" and a confabulated "working
tree clean"; 16 repetition fires in a day on round-id polling; an inject at
241k tokens. None was a controlled measurement, and the cut also removed the
one intervention the recovered record shows helping — "the primary stopped
before finishing" — which R1 later restored as turn guards. Alan's rule
(09-25): fix what got in the way inside Inner Voice, don't switch it off. So:

| 09-12 harm | fixed by |
|---|---|
| round-id polling read as a loop | polling tools and `SM_…` ids exempt in the repetition guard (09-11) |
| inject with no room to answer | the terminal review skips under the context floor (R2) |
| confabulated state from fragments | the review reads the whole trajectory and 8k of terminal text (R2) |
| "deliver the final report" on a worker | `observer._worker_note` → `observer_prompt.build_worker_note` in every worker review: PLATFORM, no human reader, the finalizer collects the outcome, and — from the turn guards' own `round_open` — whether a round is open and its only correct ending (commit, `automod_gate`, `automod_gate_wait`, land or abort). It is placed late in the prompt, beside the pressure notes it overrides |
| the same, if the model ignores the note | `_apply_decision_guards` rewrites any inject `guards.asks_for_report` matches on a worker turn: to `UNATTENDED_ROUND_OPEN_CONTENT` with a round open, to `noop_worker_report` without one (safeguard `worker_content`, event `inner_voice.worker_report_ask_rewritten`). Negated asks ("do not write a report") are not asks |

R5 had deleted exactly these two with the unattended profile, because no
worker was observed then; re-enabling without them would have recreated #874.
Alongside: the `round_open` turn guard may fire a second time when the turn did
more work after the first nudge and stopped again (`ROUND_GATE_MAX_FIRES` 2),
and the autocode reaper's 20-minute grace no longer follows `inner_voice` — it
waited for the post-turn follow-up R2 retired (`abandon_grace_seconds`, 0).
Worker sessions are not in the `iv-chat-1` A/B; a round's outcome with the
observer on reads off the automod ledger joined by `session_id`.

## Known defects

- **Guard races under `async_nonterminal`** (09-04 review A4): two async
  mid-turn reviews can both clear the suppressor. Much rarer now that a
  mid-turn review is due every several iterations rather than on sampled
  results; still a soft cap, not an exact one.
- **`stalled_progress` has never been measured** for precision.
- **`iteration_pressure_enabled` guards the prompt note only**, not whether
  the review runs.

## Verification

`tests/test_turn_guards.py` (one test per guard on a worker turn with no
observer, one through the real loop), `tests/test_iv_one_review.py` (R2/R4),
`tests/test_iv_measurement.py` (R3), and the existing `tests/test_iv_*`,
`tests/integration/test_iv_*`, `tests/integration/test_observer.py`. After R2
the nightly series should show `input_tokens_per_turn` and
`observer_ms_per_turn` falling (predicted ~15–20k tokens a turn against ~62k).
