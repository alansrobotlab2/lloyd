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

## What a summary re-injects as

A summary is not a note back to the reader, it is a row in the assistant's own
voice. `app/compaction_state.py::summary_message` returns the rendered record with
`role` set to `assistant`, and `app/compaction.py::_persisted_summary_layer` puts it at
index 0 — first element of the new conversation — so everything the summariser carried re-enters
every later turn as text Lloyd apparently wrote itself: the fence it arrived inside is
gone, and so is any label saying where it came from. The summariser strips one thing,
the `<analysis>` scratchpad, and nothing else. That is a persistence channel for a
prompt injection: content the model was told to treat as data gets promoted to
first-party instructions by the act of being summarised.

It is worse than a bare copy because of what the summariser is asked to do. Its prompt
(`app/compaction_llm.py`) orders the model to keep user messages near-verbatim, to list
pending tasks that were "explicitly requested", and to name the single most logical
next step — three ways to promote a directive into the part of the summary a later turn
will act on — and nothing in it says that folded-window content may be untrusted data.
One partial mitigation is in force and undocumented elsewhere: the same prompt says
tool results are persisted to disk and should be referred to by tool and intent, which
narrows a payload riding in a large tool result and does nothing for one quoted in as a
user message.

Measured so far: the mechanism, from the tree, and it is real. What has not been
measured is the rate, and the instrument that will measure it is in
`eval/run_injection_canary.py`. Its persistence scenarios put the payload on the
arrival turn, force a fold over every arrival row with `keep_recent_turns` at zero, and
probe on the next turn from the summary row alone — so the only way the instruction can
reach the probe is through the summariser. An episode whose fold did not cover every
arrival row is reported not-run, because its probe turn still had the payload
verbatim. It reports the leak as planted tokens found in the rendered summary — matched
verbatim, each planted token against the exact text `render_summary` produces, which is
the text the summary row actually carries — beside a benign control's own survival
count, so a summary emptied enough to look clean shows as a control failure. That
verbatim match is why the rate is a **lower bound**: the summariser's own prompt invites
it to restate things "in their own words", and a directive that survived in other words
is not counted. A low number is therefore evidence the channel is narrow, never evidence
it is shut. The verdict on the channel is therefore open, and this is the command that
closes it:

```bash
python -m eval.run_injection_canary run --only persistence-web-digest persistence-relay-email persistence-control-handover
python -m eval.run_injection_canary grade
```

The first line needs the live aggregator and the engine, which is why it is owed after
landing rather than run from a round. Where a fix would go if the number is not
negligible is the summariser's output boundary — restating a surviving directive as
inert data, or keeping its provenance with it — and not a wider allowance for the
model to edit its own context, which is the one thing the same class of paper argues
against. Whether the summary row keeps its `assistant` role is a separate ruling, and
changing it would move every later turn's reading of that text.

## What a rewrite costs, priced per mechanism

A rewrite of the history is not free at the engine: `vllm.md` §2.2 prices a prefix
break as the re-prefill of the suffix past it, and arXiv:2609.37725v1 §4.3 puts the
same rule the other way round — an edit is priced by everything **after** its break
point, so an edit near the **front** is the expensive kind. Both of Lloyd's
destructive layers break at the front. `app/compaction.py:505` is
`new_convo: list[dict] = [CS.summary_message(record)]`: the compacted conversation
begins with the summary, so position 0 changed and the longest common prefix of the
old and new prompt is about zero — the whole compacted conversation re-prefills.
`app/harness/microcompact.py:11` clears the oldest tool results "needed to get under
``token_budget``, oldest first, and stops", and the module states at its line 28
that "Clearing is oldest-first by design", so the first cleared message is the break
and the surviving suffix is nearly the whole conversation.

What that costs is measurable, because #1078 put a `compaction` record on every
`usage` row and #1078's companion column `reprefill_tokens` holds the turn's measured
re-prefill. The split is by **arm** — whether the pass changed anything — and that
split is the whole instrument:

```sql
-- Quote BOTH bounds literally, the `since` and the `until` the report prints: the
-- interval is `since <= ts < until`, the same bound reprefill_attribution(since=…,
-- until=…) closes its row scan with. Drop the upper one and this stops being the
-- query the table below answers: a re-run on or after 2026-10-01T16:28:38 keeps every
-- row that arrives after the window closes while the table still says 3,868 rows and
-- 112,908,568 tokens. `ts` carries a literal 'T' between date and time, so a
-- comparison written datetime('now','-7 day') — which sqlite renders with a
-- SPACE — is lexicographically WIDER ('T' is 0x54, above ' ' at 0x20, so every row
-- of the boundary day passes) and reaches back to 00:00 of that day: 233 more rows,
-- 22,940,221 more tokens on 2026-10-01. That is why #2027 read "135,848,789 over
-- 4094 rows" and "112,908,568 over 3861 turns" as two kinds of total; both are
-- COALESCE(SUM(reprefill_tokens),0), over two different windows.
with b as (select case when compaction is null then 'A_no_record'
    when coalesce(json_extract(compaction,'$.turn_start.tokens_freed'),0)>0
      or coalesce(json_extract(compaction,'$.relief_tokens_freed'),0)>0 then 'B_freed_something'
    else 'D_ran_noop' end arm, input_tokens it, reprefill_tokens re
  from usage where ts>='2026-09-24T16:28:38' and ts<'2026-10-01T16:28:38' and reprefill_tokens is not null)
select arm, count(*) n, cast(avg(it) as int), cast(avg(re) as int),
  round(100.0*avg(cast(re as real)/nullif(it,0)),1), sum(re) from b group by arm;
```

Run against the machine's usage database — `app/paths.py`'s `USAGE_DB`, which on this box is
lloyd-data/usage.db under the data root — at 2026-10-01T16:28Z over the last 168 hours
(3,868 rows, 112,908,568 tokens of re-prefill — the total this doc's percentages are
against):

| arm | n | mean input | mean reprefill | miss % of prompt | Σ reprefill |
|---|---|---|---|---|---|
| `A_no_record` (no `compaction` at all) | 1,446 | 58,170 | 4,976 | 3.6% | 7,195,735 |
| `D_ran_noop` (offered, freed nothing) | 2,061 | 80,383 | 5,094 | 3.9% | 10,499,022 |
| `B_freed_something` | 361 | 155,630 | 263,750 | 161.6% | 95,213,811 |

The control is the point of the table: `D` ran the pass on sessions **bigger** than
`A`'s and re-prefilled 5,094 tokens on average, 3.9% of the prompt, so the
instrument is not measuring "big sessions evict". Against it a freeing turn
re-prefills **51.8×** more, on 9.3% of measured turns, and carries **84.3%** of the
week's re-prefill. A ratio above 100% of the prompt is not an error to fix:
`reprefill_tokens` is summed over a turn's iterations while `input_tokens` is one
request.

**That 51.8× is not size-controlled, and is stated as such.** Arm `B`'s mean prompt
(155,630) is 1.9× arm `D`'s (80,383), so some of the ratio is the sessions being
larger rather than the edit costing more. The within-session pairing that would
settle it — a free turn's next turn against a noop turn's next turn in the same
session — is owed on #2027 and no controlled ratio is quoted here.

Per mechanism, `usage_store.reprefill_attribution(hours=168)` (its `["report"]` is
the text to print; `python -m pytest tests/test_reprefill_attribution.py` runs it in
the small) assigns every row in the window to exactly one bucket — front-most break
first, because §4.3 prices an edit by everything after its break, and the record
holds one `reprefill_tokens` per row with no per-rung split, so the split has to be
an assignment rule and not a partition of the tokens. Same window; `n=` is rows
credited to the bucket, `fired_` is rows whose record names that rung in a freeing
pass and is **not** additive, so a rung can sit at `n=0` while having fired
hundreds of times behind an earlier rung:

| bucket | n | Σ reprefill | mean | share | fired_ |
|---|---|---|---|---|---|
| `turn_start:microcompact` | 236 | 68,444,409 | 290,019 | 60.6% | — |
| `relief:tool_results` | 95 | 20,903,710 | 220,039 | 18.5% | 379 |
| `relief:reasoning` | 30 | 5,865,692 | 195,523 | 5.2% | 13,517 |
| `ran_noop` (the control) | 2,049 | 10,181,397 | 4,969 | 9.0% | — |
| `no_compaction_record` | 1,446 | 7,195,735 | 4,976 | 6.4% | — |
| `no_freeing_evidence` | 12 | 317,625 | 26,469 | 0.3% | — |
| `relief:truncate` | 0 | 0 | — | 0.0% | 1,856 |
| `relief:arguments` | 0 | 0 | — | 0.0% | 516 |
| `turn_start:summarized` / `:summary_folds` / `:truncated` / `:other`, `relief:images`, `relief:other` | 0 | 0 | — | 0.0% | — |

The three freeing buckets account for 95,213,811 tokens exactly — all of arm `B` —
and every bucket sums to the named total to the token (the function prints the gap,
and `tests/test_reprefill_attribution.py` pins that it is 0.0%).

These figures came off this box's live usage database, and a rolling window moves
them, so the rows they were measured on are committed beside them, in the vault's
backlog/data/2026-10-01.2027-reprefill-witness.jsonl — 3,868 rows over the absolute
window named by `REPREFILL_WITNESS_SINCE` and `REPREFILL_WITNESS_UNTIL`.
`usage_store.replay_usage_extract` loads it into a fresh database through the
module's own schema and the same `reprefill_attribution(since=…, until=…)` answers
over it; `test_the_extract_replays_to_its_own_published_split` asserts the replayed
bucket names, `n`s and Σ against the table above. Two extracts were built wrong
before that assertion existed and both passed a Σ check — one dropped
`turn_start.microcompacted`, which renamed the 60.6% bucket to
`turn_start:other` with its tokens unchanged, and one encoded JSON nulls as the
string `'null'`, which moved 1,458 arm-A rows into `no_freeing_evidence` and the
headline from 51.8 to 52.3. Check the bucket names, not just the total.

**The verdict.** Keep paying the break, and stop describing it as one number. The
week's price is not the summary rewrite: `turn_start:summarized`,
`:summary_folds` and `:truncated` are each `n=0` in this window — `persist_summary`
is off and the 210,144 threshold was not crossed — so `app/compaction.py:505`, the
front-most edit of the three, cost nothing in the week measured. Two mechanisms own
it: oldest-first microcompaction at the turn-start pass (60.6% of the window, 71.9%
of arm `B`, on 236 turns) and the intra-turn ladder's `tool_results` rung (18.5%, 22.0%
of `B`, on 95 turns), which together are 79.1% of the total re-prefill Lloyd
generated in seven days. What that buys is the session: without the pass, those 361
turns' prompts sit at or over the 210,144 threshold and the turn is either truncated
or rejected, so the re-prefill is the price of a long session surviving rather than a
tax on it. The remaining open question is the ladder's re-arm level, not whether to
free: `closed()` releases the latch as soon as the prompt is back above `_relief_rearm_level`, and the
window's worst turn — 2,580,081 tokens re-prefilled — records a re-arm level of
151,303 against its own turn-start threshold of 210,144, with 137 relief passes whose
`used_after` sits at or above ~155k on essentially every one, most of them freeing
tens of tokens. So a session that frees three times in one turn is a session
**parked above** the re-arm level, not one whose prompt regrew to it. That ruling, and the paired
size-controlled ratio, are owed on #2027; nothing here changes a threshold.

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
- **2026-10-01 — added §What a rewrite costs, priced per mechanism (#2027).** The
  arm table and the per-mechanism split were run against the machine's usage
  database (`app/paths.py`'s `USAGE_DB`) at
  2026-10-01T16:28Z over 168 hours, not copied from the item: the item's headline
  "68M tokens/week" is reproducible from no query (arm `B` is 95,213,811 and the
  window total 112,908,568); its 84.3% is a correct like-for-like share, and the
  triage's "like-for-like 70.1%" is the mistaken one — it divides the same arm-`B`
  sum by the SPACE-spelled window's 135,848,789, and the two spellings differ only
  because `ts` carries a literal 'T' that sorts above the space `datetime()` emits;
  and its warning not to "fix" a share above 100% is well meant but answers a defect
  no query produces (`reprefill_tokens` is per turn, `input_tokens` per request). The 51.8×
  headline is written as not size-controlled, and the re-arm-level reading is stated
  as owed rather than settled.
