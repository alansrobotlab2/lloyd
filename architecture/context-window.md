---
segment: architecture
tags: [architecture, lloyd, context, compaction, prompt]
type: reference
status: implemented
date: 2026-09-29
---

# Lloyd — the context window

Everything between the session file on disk and the prompt the engine
accepts. The harness is stateless per request, so each turn's conversation is
rebuilt from the persisted session, fitted to the model's window, and shipped.
Four modules hold the fitting and its record — `app/compaction.py`,
`app/compaction_llm.py`, `app/compaction_state.py` and `app/compaction_record.py`,
2,511 lines between them, three of them fitting and the fourth saying what the
other three did — and the machinery around them decides what the window costs:
the spill,
the meter, the prompt layout, and the counter that says whether the prefix cache
was hit at all.

It had no doc of its own before #1699: the material existed as decisions inside
`harness.md` (§P7 for the trigger, §D11 for `/compact`) in the same unit that
covers `run_query`, subagents, engine resolution and RPC, which is why a change
to compaction was reviewed only if a reader of `doc:harness` happened to be
looking. `architecture/index.md` now lists this one; the loop's turn mechanics
stay in [[harness]].

## Four layers, cheapest first

`app/compaction.py::load_and_compact_session` reaches for them in order of what
they cost, and each layer gets a chance to make the next one unnecessary. Three
steps, not four: layers 2 and 3 are the two implementations of the *same* step,
and a turn runs one or the other (`app/compaction.py:721` / `:737`) — it never
summarises persistently and then again from scratch.

1. **Microcompaction** (`app/harness/microcompact.py`) — a structural pass with
   no LLM call: old results are replaced by a short marker, oldest first, until
   the conversation is back under its target. Which results are eligible is a
   deny-list today (`compaction.microcompact.non_compactable_tools`, D10, threaded at
   `app/compaction.py:678`): everything but the six bookkeeping tools is
   clearable and the `compactable_tools` allow-list is ignored. It has to spill
   a result to disk before it clears it, so the marker can name the file the
   model can re-read (`app/harness/tool_result_spill.py`).
2. **The persisted summary** (`app/compaction_state.py`) — that step, chosen
   under `compaction.persist_summary` or for a session someone hand-compacted
   (`compaction_state.py:241`). One record per session in `data["compaction"]`,
   folded forward rather than regenerated; the fold calls the same summariser
   layer 3 does (`compaction_state.fold` → `compaction_llm.summarize_incremental`,
   `app/compaction_state.py:489`). History is rebuilt as
   `[summary] + rows past the boundary`.
3. **LLM summarisation** (`app/compaction_llm.py`) — the other implementation of
   that step, and the one that runs with the flag off: one fresh summary of the
   whole older block, every turn. Either way the summary is followed by Layer C
   (`app/compaction.py:508` / `:782`) — up to `compaction.restore.max_files`
   files, `budget_tokens` in all, re-read from disk and re-injected — so what
   survives a summary is the verbatim tail *plus* those bodies.
4. **Truncate** (`app/compaction.py::truncate_conversation`) — drop-oldest. The
   fallback when `compaction.mode` is `truncate`, and the silent one when
   summarisation fails; a turn that lands here emits
   `compaction.summarize_fallback`. Under a persisted summary the record's head
   is what drop-oldest never drops (`app/compaction.py:806`), so a capped fold
   loses verbatim rows and the next turn keeps folding.

Layer 2 is the recent fix and it is worth keeping straight, because it is a
performance story that reads as a correctness one. Before `app/compaction_state.py`
(review 2026-09-24, D2), the between-turn summary was never stored: once a session
crossed the wall, every turn re-summarised the whole older block — up to 120
seconds before the first token, a *different* summary each time so the prefix
cache was invalidated every turn, and an older block with no bound on its size, so
a big one 400'd the summariser and the turn fell back to drop-oldest, every turn,
quietly. The summary record's shape and its fold are `app/compaction_state.py`
(`RECORD_KEY = "compaction"`), and `compaction.record_invalidated` is the event
that says it was discarded and rebuilt. Do not read that event's subject as
`app/compaction_record.py`, which is a different record: the per-turn *rewrite*
telemetry — which mechanism acted, how much it removed (#1078). Two records
share the word in this subsystem; only one of them is history.

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

Every key below lives under `config.yaml`'s `compaction:` — the microcompact ones
nested one level deeper at `compaction.microcompact:` (`config.yaml:424`) — which
owns the thresholds, while `app/compaction.py` holds only `setdefault` fallbacks.
So a number read in code may not be the number in force: `app/compaction.py:360`
defaults the micro pass to a `trigger_fraction` of 0.8 while `config.yaml` ships
0.72.

| Key | In force | What it does |
|---|---|---|
| `compaction.mode` | `summarize` | the summarise step's strategy; `truncate` skips it |
| `compaction.keep_recent_turns` | 5 | how many recent turns stay verbatim past the summary boundary. Truncation has its own count, `TURNS_TO_KEEP` = 20 (`app/compaction.py:90`) |
| `compaction.persist_summary` | false | fold vs regenerate, one or the other. Compared against `summary_legacy` on 2026-09-25 and kept off: behind on recall, 42 s slower at turn start ([[measurement]] §Arms that are flag values) |
| `compaction.summary_input_budget_tokens` | 48000 | bounds one fold's delta |
| `compaction.max_folds_per_turn` | 3 | leftover delta folds on later turns |
| `compaction.restore.budget_tokens` / `max_files` | 50000 / 5 | what Layer C puts back after a summary: re-read file bodies, on top of the verbatim tail |
| `compaction.memory_flush.enabled` | false | layer 0 in spirit: a quiet turn to write durable facts *before* the wall |
| `compaction.microcompact.trigger_fraction` / `target_fraction` | 0.72 / 0.52 | start clearing at ~151k of the 210k threshold, stop at the target |
| `compaction.microcompact.keep_recent_tools` | 15 | floor on results kept inline |
| `compaction.microcompact.min_chars_to_clear` | 2000 | below this the marker costs more than it saves |

Two things the config comments are explicit about and a reader should not lose.
A fold is not free: every compaction rewrites the middle of the prompt and forces
a cold re-prefill of everything after it, so the 0.2-wide band is deliberate —
narrowing it would trade resident KV for more of exactly those stalls. And
`compaction.microcompact.count_threshold: 20` is legacy: `app.compaction` passes
a token budget instead,
because the old count-only rule cleared 93 of 97 results on a review using 17% of
its window.

`compaction.memory_flush` (`app/memory_flush.py`) ships off for the same reason
`persist_summary` does: both are `--arms` presets of
`eval/run_compaction_recall_eval.py`, and both were measured against
`summary_legacy` on 2026-09-25 — twelve kept rows each. `summary_persisted` came
in behind on recall (-0.25, not significant in direction) and 42 s slower at
turn start (significant); `memory_flush` gained 0.08, and of the eight sessions
whose flush saw the planted fact it wrote the distinctive one down in one, so the
gain is not the flush's. The run, its numbers and the stay-off verdict are
[[measurement]] §Arms that are flag values; what the pair waits on now is not a
first comparison but the cross-turn one that runner has no preset for. The
flush's shape is one ambient turn at 0.85 of the threshold,
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
(P1). Three layouts exist — `prompt_builder.SESSION_STATE_LAYOUTS` — and what
ships is `system_tail` (`harness.prompt_layout.session_state`,
`config.yaml:555`, since 2026-09-25): the `<session_state>` block is rendered
last, after every static paragraph, so a goal/plan/todo edit moves the tail of
the system prompt and nothing under it. The `<memory_delta>` note goes to the
user tail when `harness.prompt_layout.freeze_memory` is on (`config.yaml:556`
ships it off) and the memory files moved since the session's snapshot
(`app/memory_snapshot.py`). Both exist to keep the system prompt as static as it
can be, because it is the front of the prefix and anything that moves inside it
moves everything behind it — which is the difference between the three layouts,
and why `user_tail` — the layout that leaves the system prompt byte-identical
across state changes — is the next A/B step rather than the shipped one.

`app/prefix_miss.py` is the instrument for that cost. Every iteration is a fresh
request re-submitting the whole conversation; while the engine still holds the
prefix, an iteration pays only for what was appended — 5-7k tokens on a 207k
prompt, measured 2026-09-10 on the FP8 build. When it has been evicted, the whole
prompt is prefilled again from cold, one chunk per engine step — 4,096 tokens
since 2026-09-10 (`MAX_NUM_BATCHED_TOKENS` in
`agent-services/supervisor/conf.d/agent-llm-primary.conf`, `vllm.md` §5.1; the
module docstring still says vLLM's default 8,192, which is #1785), and
every other request on the engine gets one token per step until it finishes. The
09-09 "5 tok/s" stall was that: all 27 episodes in that day's engine log were
100-200k prompts re-prefilled from cold beside a chat. FP8 made each one cheaper
(a 1.74x pool, 3200-token pages); this module is what says whether they still
happen.

## What this doc does not cover

- **Turn mechanics.** `run_query`, events, the position-0 rule, the tool pool,
  subagents and engine resolution are [[harness]] — as is the mid-turn relief
  ladder under `harness.context_relief`, which is the other half of "the loop
  deciding when to relieve pressure": this doc's four layers run at turn start,
  that one runs between iterations against the same meter. Two of its decisions
  are quoted above because they name a threshold, not because the loop moved
  here.
- **The engine's own window.** The context length in force, FP8 KV, YaRN and
  paging are [[vllm]]; `app/compaction.py::get_context_window` looks the number
  up, it does not own it.
- **What is retrieved into the window.** Skills, facts and vault docs arriving
  as `<context>` is [[subliminal]]; how recall answers is [[retrieval]].
- **The durable half of a flush.** This doc covers the trigger and the turn pool;
  what `memory_add` and `fact_add` write is [[memory]] and [[knowledge-graph]].
- **Whether the thresholds are right.** They are measurements with an owner: the
  context-rot and compaction-recall pins and runners indexed by
  `architecture/measurement.md`, and `app/prefix_miss.py` for the stalls.
  Nothing here asserts a threshold is correct, only what it is, what it costs
  and which flag turns it on.

## Review log

- **2026-09-28 — created (#1699).** The four compaction modules' line counts and
  the config values in the table were read from this tree at `f7cf29f4`, not from
  `harness.md`. Two facts the item got wrong are corrected here rather than
  inherited: `tool_result_spill.py` is `app/harness/tool_result_spill.py`, and the
  context-meter and microcompact modules moved under `app/harness/` in the 2026-09
  root move (`f9863c64`), so the paths the item quoted are stale.
- **2026-09-29 — stale.** Five corrections, all read from this tree at
  `86aef7e5`, plus the key paths the table wrote so they resolve: the "four
  layers" read as a sequential walk, but the persisted summary and
  `compaction_llm` are one branch (`app/compaction.py:721` / `:737`) followed by
  Layer C's re-injected files, which were undocumented; `compaction_record.py` is
  the per-turn rewrite telemetry, not the summary record's shape and fold;
  microcompaction runs on a deny-list today, not the allow-list; the cold-prefill
  step is 4,096 tokens, not vLLM's default 8,192, since 2026-09-10 (#1785); and
  the shipped `prompt_layout` is `system_tail`, not the `user_tail` condition the
  paragraph named. Filed: #1785 and #1786 are the stale module docstrings that
  were the source of two of these, and #1787 is the two unrun arms this doc
  delegated to [[measurement]], which names neither of them.
