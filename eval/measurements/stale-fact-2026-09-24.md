# #622 — does Lloyd act on a superseded fact? (2026-09-24)

Runner: `eval/run_stale_fact_eval.py`. Raw reports (every trial, answer, tool
call and judge verdict): `stale-fact-2026-09-24/run1-five-arms.json.gz`,
`stale-fact-2026-09-24/run2-dated.json.gz`. The runner writes `.json` next to
this file, which `*.json` in `.gitignore` hides, so the committed copies are
gzipped.

## Question and design

When a fact is superseded, does Lloyd act on the old value? A pinned synthetic
corpus: 32 fictional entities (services, people, projects, devices) with
4 facts each = 128 planted, 20 superseded, 12 never-superseded controls. Every
fact is fictional, so the stateless arm scores 0 by construction and the
stateful rate is the memory gain. It is planted into a SHADOW memory (a temp
`LLOYD_FACTS_ROOT` / `LLOYD_KG_DB` and a temp `MEMORY.md`, never the live
store; the runner exits 2 with `LiveStoreRefused` on a live path) through the
real writers: `facts._fact_add`, `facts._fact_invalidate` and
`session._memory_add`. Only the clock is faked, so planted rows sit in early
July and supersessions on 2026-09-02, as they would in life.

Each probe is action-shaped ("give me the curl command", "which cost code goes
on the form") and needs the CURRENT value. One trial runs the production
primary (Qwen3.8-Flash-Next) with SOUL.md as its system prompt, plus the
memory block when the arm has one. It opens with a `fact_get(entity)` exchange
answered from that arm's store. The model may then call `fact_get` /
`memory_read` itself (real handlers against the shadow state, 4 iterations
max) before it answers. The answer is scored `correct` when it names the new
value only, `stale` when it names the old only (a miss, never "unanswered"),
and `none` when it names neither. When it names both, the primary judges which
one it acted on (thinking on, temperature 0, JSON grammar): `current` →
correct, `superseded` → stale, `both_unresolved` → **hedged**. n = 20
superseded probes × 3 samples = 60 per arm, and 12 × 3 = 36 control trials.
Default sampling. No live tools are offered, so no trial can touch the machine.

Arms. They differ only in what the memory holds; the system-prompt skeleton,
the probe, the tools and the sampling are identical.

| arm | how the supersession is written | old value in context? |
|---|---|---|
| stateless | nothing planted | no |
| facts_expired | `fact_add(new)` + `fact_invalidate(old)` | **no**: `fact_get` filters expired rows (0/20) |
| facts_appended | `fact_add(new)` only, which is what the extractor does in practice (25 expired rows of 306k live) | yes, 20/20, with `created_at` |
| prose_appended | `memory_add(old)` … `memory_add(new)`, both lines kept (the `lloyd` segment) | yes, 20/20, no dates |
| prose_consolidated | as above, then rewritten the way a nightly consolidation rewrites the file: one `##` section per entity, lines in no order | yes, 20/20, no dates, no order |
| prose_*_dated | as above with `memory_tools.date_stamp_entries` on | yes, 20/20, each line dated |

## Results

Stale-action rate on superseded probes, audited (see "Judge noise" below).
Wilson 95% intervals. Controls were 36/36 in every stateful arm.

| arm | stale | Wilson 95% | hedged | correct | memory gain (paired 95%) |
|---|---|---|---|---|---|
| stateless | 0/60 | [0, 6.0%] | 0 | 0 | — |
| facts_expired | 0/60 | [0, 6.0%] | 0 | 60 | 1.00 [1.00, 1.00] |
| facts_appended | **0/60** (1 measured, judge misread) | [0, 6.0%] | 5 | 54 | 0.94 [0.88, 0.98] |
| prose_appended | **0/60** (1 measured, judge misread) | [0, 6.0%] | 3 | 55 | 0.95 [0.90, 0.99] |
| prose_consolidated, run 1 | **25/59** | [30.6%, 55.1%] | 12 | 22 | 0.60 [0.45, 0.76] |
| prose_consolidated, run 2 | **21/59** | [24.6%, 48.3%] | 19 | 19 | 0.57 [0.41, 0.73] |
| prose_consolidated_dated | **0/60** | [0, 6.0%] | 0 | 60 | 1.00 [1.00, 1.00] |
| prose_appended_dated | 0/60 | [0, 6.0%] | 0 | 60 | 1.00 [1.00, 1.00] |

The noise floor comes from the same arm run twice, with a different
consolidation layout: 25/59 vs 21/59. The fix's effect is paired by probe
within run 2 (same layout, same session): the stale rate falls by −0.35
[−0.53, −0.18], p < 0.001. The dated arms also stop hedging (12–19 hedged → 0),
so the gain rises 0.57 → 1.00.

### Mechanism

The model does not act on a superseded value when anything in context says
which value is newer. `fact_get`'s `created_at` does this for the fact layer,
and file order does it for an append-only MEMORY.md. In the appended arms most
answers name both values, take the newer one and flag the older one as older:
49/60 (facts) and 51/60 (prose). It fails exactly when no recency cue survives. In the consolidated
arm the answer follows the line that sits LAST:

| consolidated, by position | stale |
|---|---|
| old line after new (run 1 / run 2) | 25/33 · 20/30 |
| new line after old (run 1 / run 2) | 0/27 · 1/30 |

The answers say so ("2964 listed after 7277, so it's likely the current one").
So in retrieval terms both values are returned, and in prompt terms no
supersession marker is shown. Once a rewrite reorders the file, the model's
heuristic "last written wins" picks the old value about as often as the file
puts it last.

### Fix (behind a flag)

`memory_tools.date_stamp_entries` (config.yaml, default **off**; the code
lives in `agent_mcp/session.py::_memory_add`) prefixes each `memory_add` entry
with its UTC write date: `- (2026-09-02) Quillmark runs on godwit-33.`. The
date travels with the line through any rewrite that keeps the line. Measured:
consolidated stale 21/59 → 0/60, hedged 19 → 0, no cost on the appended arm
(0/60 → 0/60), controls unchanged. The cost is ~13 bytes per entry against
MEMORY.md's byte ceiling.

**Limits.** The synthetic consolidation keeps lines verbatim. A real nightly
rewrite (an LLM) may paraphrase an entry and drop its stamp, and then this
fix does nothing for that line. How often live rewrites reorder a
contradicting pair was not measured. Live MEMORY.md/USER.md carry prose-level
dates ("09-08") on some entries, not a per-line stamp.

### Judge noise

166 of 480 answers in run 1 named both values and went to the judge. The first
judge ran with thinking off and misread 2 of the 6 stale verdicts it produced
on the appended arms. A diacritic bug also counted "Valparaíso" as not naming
"Valparaiso". Both are fixed (`_fold`, judge thinking on), and the reports were
re-judged in place (`--rejudge`). The re-judge still reads the two Saltmarsh
answers as stale while each commits to 16,000 (the new value), so they were
audited by hand as correct. Eight randomly drawn consolidated-arm stale
verdicts were audited by hand: all eight are genuine (the command uses the old
value). One judge reply per run was unreadable and is scored `error`, which
takes the consolidated denominator to 59.

## Verdict

- **Stale action on today's channels is rare: REJECT** the item's premise that
  it is frequent. It was 0/60 on the fact layer when supersession is only
  appended (the realistic writer), 0/60 on the append-only prose memory, and
  0/60 on the correctly expired fact path. Wilson upper bound 6.0% each.
- **It is frequent, 36–42% (25/59, 21/59), once a rewrite strips the recency
  cue**, and a small fix measurably removes that: stale 0/60, paired
  Δ −0.35 [−0.53, −0.18]. The fix is landed behind a default-off flag.
  Recommended: `memory_tools.date_stamp_entries: true`.

## #82's budget

Decision: **not in the nightly task. Keep it as its own manual/per-change
script.** One five-arm run took 1,414 s of wall time and the four-arm re-run
1,773 s, both at concurrency 12 on a shared engine. That is far past
`82-nightly-retrieval-eval.md`'s budget (900 s when the item was written,
1,800 s today, and that budget is already spent on retrieval). A pinned
synthetic corpus also measures model and memory-writer behaviour, which moves
only when the model, SOUL.md or a memory writer changes, so nightly repeats
would mostly re-measure noise. Run it when one of those changes:
`flock -s <primary.lock> .venvs/lloyd/bin/python eval/run_stale_fact_eval.py --samples 3`
(`--arms` to narrow it, `--dry-run` for the corpus and stale-evidence counts
alone, with no engine).
