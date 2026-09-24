# Lloyd — profile for the improvement eval

This is what a model is shown when asked "does this video hold anything that
would improve Lloyd?" Keep it current and short: it is a rubric input, not
documentation. Figures here go stale, so a line that carries one also names the
call or the file to read the live value from, and a number that names no source
is a hint, not a measurement. This file is what the sessions that *propose* work
read, so whatever it asserts becomes the premise of every idea written against
it.

## What Lloyd is

A fully local AI agent that runs its own in-process agent loop against a local
inference server and exposes every tool through one MCP aggregator. Nothing
leaves the machine: no cloud model, no hosted vector store, no SaaS memory.
Backend is FastAPI + SSE; frontend is React. A single owner (Alan) uses it for
research, coding, voice conversation, and running the house's other robots'
projects. It also modifies its own code through a gated loop.

## Hardware and models

- Two 24 GB RTX 3090-class GPUs. Anything that needs more than ~22 GB per GPU,
  or a cloud API, is not adoptable as-is.
- Primary: Qwen3.8-Flash-Next (NVFP4) on vLLM, 262k context, OpenAI-compatible
  API, `qwen3_xml` tool parser, reasoning parser, prefix caching on.
- Secondary: Qwen3.6-35B-A3B (GGUF Q3, llama.cpp, `--parallel 1`, full 256k
  window). Used for session titles, post-session capture, voice summaries.
- TTS: Qwen3-TTS with a cloned voice, shaped client-side (EQ + WSOLA speed).
- ASR + LiveKit for voice mode; wake word path is unfinished.

## Harness (the agent loop)

- `run_query(messages, options)` async generator; streams SSE from vLLM,
  dispatches tool calls to the MCP aggregator, loops until the model stops.
- System prompt built once per turn at position 0 and never refreshed, so the
  whole prefix stays KV-cached across every iteration; mid-turn state
  (todos, plan, notifications) is re-anchored by *appending*.
- Preserved thinking: the last N iterations' reasoning is carried back in
  history (both `reasoning` and `reasoning_content`, because vLLM and
  llama.cpp read different keys).
- Every tool call carries a model-written `summary` caption; captions are
  replayed in history so the model keeps writing them.
- 131 tools; progressive disclosure via a baseline set plus a ToolSearch tool.
- Stream-stall timeout between SSE lines; max_turns budget with an anchor
  telling the model when it is at 75%/90%.
- Subagents (`Task`) run inside the aggregator process and inherit the
  calling turn's model.
- Inner Voice: a second-model observer that reads the primary's transcript
  and can inject text or stop a loop (repetition guard keyed on tool-call
  signatures). It is observation-only on tool dispatch — since v4 it cannot
  block a tool call, and it is not a safety control. The hard gate is the
  deterministic destructive-Bash check in `app/harness/safety.py`
  (`check_bash_command`), which the aggregator also enforces at dispatch.
- History is rebuilt from the session JSON every user turn and compacted
  (`load_and_compact_session`); there is no learned or summarised long-term
  compaction beyond that.

## Memory and knowledge

- The vault: an Obsidian markdown tree (`~/obsidian`) — SOUL.md, USER.md,
  skills (`SKILL.md` per skill; 187 loaded on 2026-09-24, counted as
  `len(list(agent_mcp.skills._iter_skills()))`, which drops dirs whose front
  matter says `status: archived`; on disk the same day there were 191
  top-level skill dirs and 352 `SKILL.md` files, the extra ones nested or
  under `.archived/` — only the loaded count is what retrieval sees),
  knowledge notes, people, projects,
  daily notes, autonomy task files, backlog items. Nightly jobs rewrite
  SOUL.md/USER.md/skills; that is the rewritable layer.
- Facts layer: one markdown file per entity and category
  (`facts/<Entity>/<Entity>-<category>.md`), extracted from the vault by a
  pipeline with an allow-list of source directories.
- Knowledge graph: SQLite store behind one module (`app.kg_store`): entities,
  aliases, typed edges (expire, never delete), fact index. No scale figures are
  quoted here, deliberately: the store grows nightly and every count this
  bullet used to carry went stale by an order of magnitude. Read them live
  through the one sanctioned door — `from app.kg_store import store`, then
  `store().stats()`, which returns `entities`, `aliases`, `edges_total`,
  `edges_active`, `facts`. Nothing else may open the database file.
- Retrieval runs on two paths, and they are not interchangeable. **Prefetch**
  is the automatic one: before each turn it pulls candidate vault context into
  the prompt from a fixed set of legs — skills, facts, vault lexical, vault
  hybrid (`qmd`, a local hybrid BM25 + embedding index over the vault), recent
  sessions, backlog refs — and it has **no graph arm**: `prefetch.py` never
  imports `app.kg_store`, and its fact leg is a name→facts lookup ordered by
  confidence, not a walk over neighbours. **`vault_recall`** is the tool that
  does graph seed-and-expand over that store (neighbour expansion plus
  graph-weighted reranking) along with a source-code grep fallback, and it runs
  only when the model calls it. Graph expansion is therefore available to
  Lloyd but is not part of what every turn sees — a proposal that assumes the
  per-turn prompt already carries graph neighbours is assuming something no
  code does.
- Nightly retrieval eval on the pinned corpus. The figures below are **one
  dated run**, the 2026-09-23 nightly (`summary.overall` of
  `eval/baselines/nightly-20260923-20260923-073952.json` — that directory is
  `app.paths.EVAL_BASELINES_DIR`, which lives under the data root, not in the
  code checkout), not a standing property: entity_hit_rate 0.37,
  doc_hit_rate 0.716, entity_recall_avg 0.394, doc_recall_avg 0.645,
  mrr_doc 0.368, ndcg10 0.416, fact_entity_recall_avg 0.381, latency_ms_avg
  766 ms. Latency is graded against a ceiling named in code, not a remembered
  one: this run sits inside `LATENCY_BUDGET_MS[CONTEXT_NIGHTLY]` = 4800 ms in
  `workers/sources/automod_regression.py`, and every baseline records its own
  `latency_budget` block, so the verdict travels with the file.
  Read *Live measurements* below for the current night — a figure quoted here
  with no run behind it is how an item came to adopt a bar the tree had already
  left. The known weak spot is **entity identification**:
  when a query names something obliquely, seeds land on the wrong entity rows.
- Session distillation: a worker mines finished user sessions into
  observations; a memory-capture pipeline writes daily learnings and
  trajectories; nightly consolidation ("dream") merges them into USER.md and
  skills. Quality is uneven and several of these have had truncation and
  bucketing defects.

## Autonomy, workers, self-modification

- One SQLite job queue drained by asyncio workers inside the backend runs
  every unasked thing: scheduled autonomy tasks (~37 task files with
  frequency, preferred hours, dependencies, cooldowns), deep research off a
  topic registry, session distillation, backlog triage, and the automod
  implementer. Priority ASC, per-source `max_inflight`, `failed` vs raised
  distinction, `skipped` as a third outcome.
- Backlog: markdown items on a kanban board (~450 items, ~120 open). A
  triage pass re-checks old claims against live code; an implement pass runs
  one automod round per confirmed item. Self-filed items are quarantined
  from triage for 30 days because the pass once filed 2 items per item it
  closed.
- Self-modification loop: cut a worktree from live main, make the change,
  run an 8-rung gate (pyflakes, tests, tsc/vite for frontend, retrieval eval
  regression against a pinned index, service-definition checks), land only
  when the system is idle (drains the worker pool first), observe for a
  window, roll back via an out-of-process guardian (systemd unit, stdlib
  only) that reads `logs/server.err`. Every rollback so far was a false
  positive; the design pressure is "do not invent a bad build".
- An autoresearch loop proposes hypotheses (prompt/config overlays) and
  measures them; its hypothesis generator currently truncates at 8000 output
  tokens because it echoes SOUL.md/MEMORY.md back.
- Skills: SKILL.md files in the vault, loaded into the prompt by name; a
  nightly consolidation job creates/merges/archives them from session
  patterns. Many are thin; there is no state carried across invocations of
  a skill beyond what the skill text says to write to disk.

## Evaluation surfaces that already exist

- Nightly retrieval eval (above), pinned corpus, paired comparison for
  automod rounds.
- Tool-choice eval (`eval/run_tool_choice_eval.py`), prefetch eval,
  preserved-thinking A/B, parallel-dispatch timing
  (`eval/measure_parallel_dispatch.py`), counterfactual retrieval
  perturbation (`eval/counterfactual.py`), review-rung grader calibration
  (`eval/review_calibration/`), secondary-routing replay
  (`eval/secondary_routing_eval.py`).
- Instruction-density compliance (`eval/instruction_compliance_eval.py`,
  backlog #630): how many required words an engine still carries verbatim
  as the count grows, scored mechanically, with an N* (largest N at ≥95 %)
  per engine. `eval/run_skill_dispatch_probe.py` reports a *predicted*
  compliance and disclaims it — a rule that fires proves the protocol
  reached the model, not that the model obeyed — so this is the arm that
  measures the obeying, and until it existed no compliance arm did.
- Guardian error-rate observation, worker run ledgers, usage store.

## Standing problems worth solving (an idea that hits one of these is more valuable)

1. Entity identification in retrieval (0.55 hit rate): synonyms, oblique
   references, sibling entities with thin facts.
2. Long-turn context management: 50+ iteration turns, compaction is
   truncation-shaped; no learned summarisation or working-memory scratchpad
   beyond todos.
3. Skill quality and reuse: skills are text; no compilation, versioning,
   testing, or state; consolidation is lossy.
4. Memory write quality: distillation and nightly jobs produce duplicates and
   over-long entries; no principled forgetting or importance scoring.
5. Autoresearch signal: hypotheses are noisy and the measurement is one
   retrieval eval; no broader behavioural benchmark for "is Lloyd better".
6. Harness robustness: tool-call parsing edge cases, empty tool pools,
   model drift into prose tool calls, repetition loops.
7. Inference efficiency on 24 GB: KV cache pressure at long context,
   NVFP4/GGUF quality trade-offs, speculative decoding not in use.
8. Voice: ~38 s cold start, latency from ASR to first spoken word, wake word.
9. Research pipeline: deep research off a registry; quality of synthesis and
   dedupe of topics is unmeasured.

## Live measurements — check these before deciding

When a video's claim touches a number Lloyd already measures, fetch the
number before choosing a verdict. "Lloyd already caches the prefix" or
"already batches" is only as true as the counter says, and the first digest
session (2026-09-09) rated a talk `worth_a_look` on the assumption that the
prefix stayed cached, while the live counter read 68.7%.

**From a digest session:** there is no Bash and `http_fetch` refuses
loopback by design, so do not try the URLs below from there. Instead Read
`measurements.json` in your bundle directory (beside `transcript.txt`): it
holds the vLLM counters, the dashboard's `vllm`/`workers`/`host`/`usage`
sections and the newest retrieval-eval baseline, captured when the bundle
was fetched, with an `errors` list naming any source that was unreachable.
The URLs are for a human or a shell.

- **Inference** — `http://127.0.0.1:8096/metrics`:
  `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` is the
  hit rate since boot (an append-only loop should be well above 90%);
  `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`, and the
  time-to-first-token histogram if present. The rate form is on
  `http://127.0.0.1:8080/api/dashboard` under `vllm`.
- **Retrieval** — the newest `eval/baselines/nightly-*.json` in
  `app.paths.EVAL_BASELINES_DIR` (the mutable data root — `~/lloyd-data` on
  this box, deliberately *not* inside the code checkout) is current: the
  numbers quoted above, per query, with the corpus and the latency-budget
  verdict the run saw.
- **Workers, autonomy, backlog** — `http://127.0.0.1:8080/api/dashboard`
  (`workers`, `autonomy`, `backlog` sections) and
  `http://127.0.0.1:8080/api/workers/runs` for recent run records.
- **Prompt surface and sessions** — the same dashboard's `primary` section
  for active turns; the prompt-surface measurements live with backlog #377.

## Things Lloyd already does (do not re-propose as new)

Prefix/KV caching across iterations; preserved reasoning in history; a
knowledge graph with typed edges and aliases; hybrid BM25 + embedding
retrieval; pre-turn prefetch; an observation-only observer model (Inner Voice) that
can inject text and stop a loop; a deterministic destructive-Bash gate; a gated
self-modification loop with automatic rollback; a task scheduler with
dependencies and preferred hours; skills as markdown files; session
distillation into a user model; nightly consolidation; a backlog with
automated triage; tool-call captions; progressive tool disclosure; a local
cloned voice.
