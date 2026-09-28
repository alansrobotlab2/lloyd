---
segment: architecture
tags: [architecture, lloyd, context, compaction, prompt]
type: reference
status: implemented
date: 2026-09-28
---

# Lloyd — the context window

Everything between the session file on disk and the prompt the engine
accepts. The harness is stateless per request, so each turn's conversation is
rebuilt from the persisted session, fitted to the model's window, and shipped.
Four modules do the fitting — `app/compaction.py`, `app/compaction_llm.py`,
`app/compaction_state.py` and `app/compaction_record.py`, 2,511 lines between
them — and the machinery around them decides what the window costs: the spill,
the meter, the prompt layout, and the counter that says whether the prefix cache
was hit at all.

It had no doc of its own before #1699: the material existed as decisions inside
`harness.md` (§P7 for the trigger, §D11 for `/compact`) in the same unit that
covers `run_query`, subagents, engine resolution and RPC, which is why a change
to compaction was reviewed only if a reader of `doc:harness` happened to be
looking. `architecture/index.md` now lists this one; the loop's turn mechanics
stay in [[harness]].

## Four layers, cheapest first

`app/compaction.py::load_and_compact_session` walks them in order of what they
cost, and each layer gets a chance to make the next one unnecessary:

1. **Microcompaction** (`app/harness/microcompact.py`) — a structural pass with
   no LLM call: older results from the compactable tools are replaced by a short
   marker, oldest first, until the conversation is back under its target. It has
   to spill a result to disk before it clears it, so the marker can name the file
   the model can re-read (`app/harness/tool_result_spill.py`).
2. **The persisted summary** (`app/compaction_state.py`) — one record per session
   in `data["compaction"]`, folded forward rather than regenerated. History is
   rebuilt as `[summary] + rows past the boundary`.
3. **LLM summarisation** (`app/compaction_llm.py`) — asked for the older block
   when the layers under it did not fit it.
4. **Truncate** (`app/compaction.py::truncate_conversation`) — drop-oldest. The
   fallback when `compaction.mode` is `truncate`, and the silent one when
   summarisation fails; a turn that lands here emits
   `compaction.summarize_fallback`.

Layer 2 is the recent fix and it is worth keeping straight, because it is a
performance story that reads as a correctness one. Before `app/compaction_state.py`
(review 2026-09-24, D2), the between-turn summary was never stored: once a session
crossed the wall, every turn re-summarised the whole older block — up to 120
seconds before the first token, a *different* summary each time so the prefix
cache was invalidated every turn, and an older block with no bound on its size, so
a big one 400'd the summariser and the turn fell back to drop-oldest, every turn,
quietly. `app/compaction_record.py` is the record's shape and its fold;
`compaction.record_invalidated` is the event that says a fold was thrown away.

## One number, three readers

`app/harness/context_meter.py` answers "how much of the window has this turn
used" once, for three consumers: the loop deciding when to relieve pressure and
whether a terminal inject can still be answered, the `<context>` state anchor
telling the model, and the Inner Voice observer deciding whether to nudge a turn
that has no room to act on a nudge ([[inner-voice]]).

The shared object is the point. Each reader had its own partial view: the in-turn
microcompaction pass recomputed the estimate every iteration and discarded it,
the anchor had no view at all, and the observer judged a 241k turn exactly as it
judged a 40k one. Three private estimators disagreeing is the same defect as
three private definitions of a protected path ([[authority-surfaces]]).

## The trigger, and what a fold costs

`config.yaml` `compaction:` owns the thresholds; `app/compaction.py` holds only
`setdefault` fallbacks, so a number read in code may not be the number in force —
`app/compaction.py:360` defaults the micro pass to a `trigger_fraction` of 0.8
while `config.yaml` ships 0.72.

| Key | In force | What it does |
|---|---|---|
| `compaction.mode` | `summarize` | the layer-3 strategy; `truncate` skips it |
| `compaction.keep_recent_turns` | 5 | how much verbatim tail survives |
| `compaction.persist_summary` | false | layer 2. Off until the `summary_persisted` arm of `eval/run_compaction_recall_eval.py` is compared |
| `compaction.summary_input_budget_tokens` | 48000 | bounds one fold's delta |
| `compaction.max_folds_per_turn` | 3 | leftover delta folds on later turns |
| `compaction.memory_flush.enabled` | false | layer 0 in spirit: a quiet turn to write durable facts *before* the wall |
| `microcompact.trigger_fraction` / `target_fraction` | 0.72 / 0.52 | start clearing at ~151k of the 210k threshold, stop at the target |
| `microcompact.keep_recent_tools` | 15 | floor on results kept inline |
| `microcompact.min_chars_to_clear` | 2000 | below this the marker costs more than it saves |

Two things the config comments are explicit about and a reader should not lose.
A fold is not free: every compaction rewrites the middle of the prompt and forces
a cold re-prefill of everything after it, so the 0.2-wide band is deliberate —
narrowing it would trade resident KV for more of exactly those stalls. And
`count_threshold: 20` is legacy: `app.compaction` passes a token budget instead,
because the old count-only rule cleared 93 of 97 results on a review using 17% of
its window.

`compaction.memory_flush` (`app/memory_flush.py`) ships off for the same reason
`persist_summary` does: both are measured by arms in [[measurement]] and neither
has been compared yet. Its shape is one ambient turn at 0.85 of the threshold,
`max_turns: 6`, tool pool restricted to `memory_read`, `memory_add`, `fact_get`
and `fact_add`, once per cycle, whose rows stay in the transcript and never
re-enter history. What it writes is in [[memory]].

Manual compaction is `/compact`, a queued turn rather than an inline rewrite
(`harness.md` §D11), and a record it makes is applied whatever
`persist_summary` says — a person's compaction is not gated on a flag awaiting an
eval.

## The prompt that goes out

`app/prompt_builder.py` decides what the system prompt holds;
`app/prompt_layout.py` decides what rides on the turn's user message instead
(P1) — the `<session_state>` block when
`harness.prompt_layout.session_state` is `user_tail`, and the `<memory_delta>`
note when `harness.prompt_layout.freeze_memory` is on and the memory files moved
since the session's snapshot (`app/memory_snapshot.py`). Both exist to keep the
system prompt byte-stable, because the system prompt is the front of the prefix
and anything that moves inside it moves everything behind it.

`app/prefix_miss.py` is the instrument for that cost. Every iteration is a fresh
request re-submitting the whole conversation; while the engine still holds the
prefix, an iteration pays only for what was appended — 5-7k tokens on a 207k
prompt, measured 2026-09-10 on the FP8 build. When it has been evicted, the whole
prompt is prefilled again from cold, one 8,192-token chunk per engine step, and
every other request on the engine gets one token per step until it finishes. The
09-09 "5 tok/s" stall was that: all 27 episodes in that day's engine log were
100-200k prompts re-prefilled from cold beside a chat. FP8 made each one cheaper
(a 1.74x pool, 3200-token pages); this module is what says whether they still
happen.

## What this doc does not cover

- **Turn mechanics.** `run_query`, events, the position-0 rule, the tool pool,
  subagents and engine resolution are [[harness]]. Two of its decisions are
  quoted above because they name a threshold, not because the loop moved here.
- **The engine's own window.** The context length in force, FP8 KV, YaRN and
  paging are [[vllm]]; `app/compaction.py::get_context_window` looks the number
  up, it does not own it.
- **What is retrieved into the window.** Skills, facts and vault docs arriving
  as `<context>` is [[subliminal]]; how recall answers is [[retrieval]].
- **The durable half of a flush.** This doc covers the trigger and the turn pool;
  what `memory_add` and `fact_add` write is [[memory]] and [[knowledge-graph]].
- **Whether the thresholds are right.** They are measurements with an owner: the
  context-rot and compaction arms named in `architecture/measurement.md`, and
  `app/prefix_miss.py` for the stalls. Nothing here asserts a threshold is
  correct, only what it is, what it costs and which flag turns it on.

## Review log

- **2026-09-28 — created (#1699).** The four compaction modules' line counts and
  the config values in the table were read from this tree at `f7cf29f4`, not from
  `harness.md`. Two facts the item got wrong are corrected here rather than
  inherited: `tool_result_spill.py` is `app/harness/tool_result_spill.py`, and the
  context-meter and microcompact modules moved under `app/harness/` in the 2026-09
  root move (`f9863c64`), so the paths the item quoted are stale.
