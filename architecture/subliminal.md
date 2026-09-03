---
segment: architecture
tags: [architecture,subliminal,memory,prefetch]
type: architecture
updated: 2026-09-03
---

# Context Injection & Prefetch System

## Overview

Automatic context retrieval injected before every LLM call. The agent sees relevant
skills, facts, vault documents, recent sessions, backlog tasks, and background
signals without explicit tool calls.

The load-bearing constraint: **the system prompt stays byte-stable across turns so
vLLM's prefix cache hits.** All dynamic context is prepended to the *user message*
instead. Putting it in the system prompt — which is what the first implementation
did — invalidates the cache and forces a full re-prefill of the system prompt every
turn. This is why there is no timestamp in the platform hint.

Three layers of context injection operate independently:

1. **System prompt** (per-session, static): SOUL.md + memories + skills index +
   goal/plan/todos
2. **Prefetch** (per-message, dynamic, ≤300 ms): skills, facts, vault, sessions,
   backlog refs, ambient signals, IDE state
3. **Agent tools** (on-demand): `vault_recall`, `vault_search`, `fact_get`,
   `skills_search`, `memory_read`

## Architecture

```
User message arrives at app/routers/messages.py (or voice.py)
    │
    ├─→ build_system_prompt(todos, plan, goal)      [per-session, static]
    │   ├─ ANTICOMPLIANCE_DIRECTIVE (pre-identity frame)
    │   ├─ SOUL.md (identity + operating contract)
    │   ├─ <memory> — MEMORY.md + USER.md
    │   ├─ <available_skills> (names only)
    │   ├─ <goal> / <plan> / <active_todos>
    │   ├─ Platform hint  ← deliberately timestamp-free (prefix cache)
    │   ├─ Background-bash contract
    │   └─ Turn-discipline clause  ← pairs with IV stall rescue
    │
    ├─→ prefetch_context(text, session_id)          [per-message, ≤300ms hard budget]
    │   │
    │   ├─→ drain_ambient_prefetch(session_id)      [first, before any guard]
    │   │
    │   ├─→ SessionFocus.update(text)               [0ms, in-memory]
    │   │   Accumulate weighted keywords, decay old by 0.75/turn
    │   │
    │   └─→ ThreadPoolExecutor(max_workers=5)       [parallel, budget-capped]
    │       ├─ Worker 1: _search_skills()           Token overlap, metadata-hit required
    │       ├─ Worker 2: _search_facts()            Entity extraction → fact lookup
    │       ├─ Worker 3: _search_vault(focus)       QMD hybrid search, focus-enriched
    │       ├─ Worker 4: _search_recent_sessions()  Temporal queries only
    │       └─ Worker 5: _search_backlog_refs()     #NNN / bare-number task IDs
    │       ⤷ stragglers abandoned at the deadline, left running detached
    │
    ├─→ _format_ide_state()                         [in-process MC state mirror]
    │
    ├─→ Memory nudge injection                      [every 20 user turns]
    │   <system-reminder> nudge to capture undocumented decisions
    │
    ├─→ persist role="subliminal" session entry     [#306, UI visibility]
    │
    └─→ run_query(messages, options)
        │
        ↓
    LLM sees: [system prompt] [history] [<context> block + user msg]
              ←── prefix-cached ──→     ←── uncached (~2-8KB) ──→
```

`prefetch_context` is called from [messages.py](../app/routers/messages.py) (chat +
ambient paths) and [voice.py](../app/routers/voice.py).

## Latency budget

`PREFETCH_BUDGET_MS = 300` is a **hard wall**. Workers are submitted to a
`ThreadPoolExecutor` and collected with `wait(..., FIRST_COMPLETED)` in a loop
against the remaining budget. Anything unfinished at the deadline is dropped for
this turn and logged:

```
prefetch budget=300ms exceeded, dropped=vault
```

The pool is deliberately **not** used as a context manager — `with` blocks on
`__exit__` until every future finishes, which would defeat the whole point.
`pool.shutdown(wait=False)` lets stragglers finish detached; Python GCs the pool
once their threads return.

Per code comments, the fast workers (skills ~37 ms, facts ~20–75 ms, sessions <1 ms,
backlog cached) always land. The vault QMD call is the one that gets dropped:
~290 ms warm on an embedding-cache hit, ~800–1700 ms on a novel query (embedding
dominates), ~2.5 s cold. A dropped vault search is recoverable — the agent can call
`vault_recall` explicitly.

> Two code comments disagree on warm vault latency: `prefetch.py` says ~290 ms warm
> and `vault._qmd_daemon_search` says ~100 ms warm end-to-end with `skip_rerank`.
> Neither has been re-measured recently. Measure before citing either.

## Prefetch workers

### Worker 1: Skill search

- **Source:** `~/obsidian/skills/` + `~/lloyd/skills/`, via `_iter_skills()`
- **Method:** `_score_skill(skill, tokens, require_metadata_hit=True)` —
  `name×3.0 + desc×2.0 + tags×1.5 + min(body_hits, 4)×0.3`
- **Metadata-hit gate:** a skill with zero name/desc/tag overlap scores 0.0
  regardless of body accidents. Fixes #311, where generic stopword queries pulled
  powerpoint/youtube skills into graph-classifier sessions. Body hits are a
  tiebreaker, not a qualifier.
- **Output:** first skill full body (≤6000 chars), second skill excerpt (≤500 chars)
- **Thresholds:** ≥3.0 for the first, ≥4.0 for the second
- **Cache:** 5-minute TTL skill-list cache in memory
- **Quarantine:** a skill whose frontmatter `status:` is one of `inactive`,
  `archived`, `disabled`, `retired`, `quarantined` is excluded from retrieval
  entirely while staying on disk. This is the lever for pulling a misbehaving skill
  out of circulation without deleting it — e.g. auto-generated bash-runbook skills
  the model echoes verbatim instead of acting on. The norm is `status: active`.
- **Plan-mode augmentation:** when `session.plan.plan_mode` is true, the query
  tokens are unioned with `{plan, mode, authoring, planning, plan-mode,
  plan-mode-authoring}` so the `plan-mode-authoring` skill auto-loads regardless of
  what the user typed.

### Worker 2: Entity fact search

- **Source:** `~/obsidian/facts/{entity}/` fact files
- **Method:** `_extract_entities_from_query()` regex entity matching against fact
  directory names → `_get_facts_sync(entity)` → sort by confidence
- **Output:** top 2 entities × 3 facts each = ≤6 fact lines
- **No LLM:** pure regex + file reads

### Worker 3: Vault search (QMD)

- **Source:** QMD daemon on `:8181` — hybrid BM25 + vector search
- **Collections:** `memory, knowledge, projects, lloyd, work, sessions`
  (facts and skills are excluded — they have dedicated workers)
- **Method:** dual `lex` + `vec` queries in one payload; both legs get the same
  stopword-stripped query. Conversational framing ("tell me about X") drifts the
  vec embedding away from content, so the strip applies to both; lex and vec still
  produce complementary signal on identical input.
- **`skip_rerank=True`** in prefetch. The reranker rarely changes top-1 and mostly
  shuffles within top-5 — not worth the tax on a latency-critical path. Explicit
  `vault_recall` calls keep reranking on.
- **Output:** top 5 results, ≤500-char snippets, minimum score 0.5 (0.3 was too
  noisy)
- **Focus enrichment:** the query is augmented with conversation focus keywords
  (below)
- **Gating:** skipped when the *effective* (focus-enriched) query is < 25 chars

### Worker 4: Recent-session search

- **Gate:** only fires when `_TEMPORAL_RE` matches — "today", "yesterday",
  "earlier", "what did we", "last session", "discussed", "decided", …. Avoids
  adding session noise to non-temporal queries.
- **Method:** load the session index for the last 3 days, score with
  `_score_session` plus a +0.3 temporal baseline. A pure-temporal query with no
  content tokens ("what did we work on today?") returns the most recent sessions
  by date.
- **Output:** top 3, ≤200-char snippets

### Worker 5: Backlog task-ID refs

Task IDs in user text ("what's left on #294", "302 is resolved") consistently lose
the fact-lookup competition to entities sharing common words. This is a
precision-first parallel source that sidesteps that entirely.

- **Method:** regex `(?:#(\d{2,4})|(?<!\d)(\d{3,4})(?!\d))` casts a wide net; the
  precision gate is **existence in the live backlog index**. Unknown numbers are
  dropped silently. The negative lookarounds mean `20260421` yields no match.
- **Source:** `~/obsidian/backlog/*.md`, indexed by leading task ID, 60-second TTL
  cache. A stale cache is retained on rescan failure rather than breaking prefetch.
  Frontmatter parsing is deliberately silent — several backlog files have malformed
  YAML and a warning per rebuild would flood `server.err`.
- **Output:** ≤3 refs, each with current title / status / priority / board plus a
  ≤300-char body excerpt. More authoritative than fact-store snapshots, so it
  renders ahead of `<facts>`.

## Ambient signal drain

Background producers (autonomy tasks, cron, Inner Voice) push
`AmbientPrefetchEntry` objects into a per-session in-memory queue via
`enqueue_ambient_prefetch`. `prefetch_context` drains it **first**, before the
`MIN_MESSAGE_LEN` guard — a short message shouldn't suppress an injection a
producer already decided was worth showing.

- `dedup_key` collapses repeat entries from the same producer (newest wins).
- Queue caps at `AMBIENT_PREFETCH_CAP = 5` per session; oldest evicted.
- At most `AMBIENT_PREFETCH_DRAIN_MAX = 3` are injected per turn; overflow is put
  back for the next turn.
- Entries carry `expires_at`; expired ones are evicted silently on drain.
- Rendered **first** in the context block, framed explicitly as passive: *"The user
  did NOT ask — reference them only if naturally relevant."*

## Conversation focus tracking

Per-session keyword accumulator that maintains awareness of what the conversation is
about. Solves the "stateless prefetch" problem — without it, "what about the PID
gains?" produces garbage vault results; with it, the query inherits "alfie servo
shoulder joint" from earlier turns.

### Tier 1: Keyword accumulator (zero cost)

In-memory `SessionFocus` per session:

```python
class SessionFocus:
    keyword_weights: dict[str, float]  # word → cumulative weight
    turn_count: int
    topics: list[str]                  # secondary-model topic phrases
    topics_turn: int
    last_access: float
```

- **Update:** each message decays all existing weights by `FOCUS_DECAY = 0.75`,
  evicts anything below 0.1, then adds this message's keywords at +1.0.
- **Output:** `enrich_query()` appends extracted topics + top-6 focus keywords
  (weight ≥ 0.3) to the vault search query.
- **Noise filtering:** ~150-word `_FOCUS_NOISE` set stripped before accumulation.
- **LRU eviction:** max 50 tracked sessions; least-recently-accessed evicted.
- **State:** in-process only, lost on restart. Acceptable — focus rebuilds within
  2–3 turns.

### Tier 2: Secondary-model topic extraction (background, async)

Every 5 turns (after ≥3 turns of context), `_maybe_extract_focus` fires a background
call extracting 3–5 topic phrases from the last 10 user/assistant messages:

```
User messages → secondary model → ["servo PID tuning", "Alfie shoulder joint", "motor temperature"]
```

- **Endpoint:** resolved via `resolve_model_alias("secondary")` in
  `app/secondary_models.py`. With `secondary_enabled: false` this routes to the
  **primary** model at `:8096` — the `secondary` alias points at `:8091`
  (gemma-4-e4b-nvfp4) but that program is currently stopped.
- **Timing:** background `asyncio.ensure_future()` after the turn completes — zero
  user-facing latency.
- **One-turn delay:** results are available on the NEXT message, not the current one.
- **Filtering:** injected context blocks (`<context>`, `<system-reminder>`,
  `<memory>`, `<daily_notes>`) are stripped from the transcript so the extractor
  doesn't summarize its own prior injections.
- **Prompt:** "Extract 3-5 short topic phrases (2-4 words each), one per line"
- **Temperature:** 0.0. **Timeout:** 15 s, non-fatal. Kept only if 2–6 words.

### Focus example

```
Turn 1: "lets look at the alfie servo configuration"
  Focus: [alfie, servo, configuration]

Turn 3: "try increasing the D term to reduce oscillation"
  Focus: [pid, gains, shoulder, oscillation, alfie, servo]

Turn 5: "what about the PID gains?"  ← only 25 chars
  Enriched query: "what about the PID gains? pid gains motor temperature alfie servo"
  Vault finds: Alfie Shoulder GPIO Setup, Closed-Loop Stepper Control
```

## Context block format

Sections render in this order (`_format_context`):

```xml
<context>
<ambient-signals>
Background producers queued these signals for you. The user did NOT ask —
reference them only if naturally relevant to what they're saying now.
- **[autonomy:task-42]** Nightly reflection finished with 2 config fixes
  > [≤800 char body]
</ambient-signals>
<skill name="autonomy-error-handling" score="14.5">
[full skill body, ≤6000 chars]
</skill>
<skill name="upstream-bug-triage" score="13.5" excerpt="true">
[excerpt, ≤500 chars]
</skill>
<backlog-refs>
- [Task #294] "Harness stall rescue" — in_progress, high priority, lloyd board
  [≤300 char body excerpt]
</backlog-refs>
<facts>
- [Lloyd] Vault search added to prefetch pipeline (confidence: 0.75)
- [QMD] QMD is installed at version 1.0.8 (confidence: 1.0)
</facts>
<vault-context>
- **Alfie Servo Settings** (score: 0.93): PID gains shoulder=0.8/0.1/0.05...
- **2026-04-13 Daily Notes** (score: 0.56): Discussed servo tuning approach...
</vault-context>
<recent-sessions>
- [2026-09-02T11:19] (primary, 24 msgs): servo tuning | shoulder oscillation
</recent-sessions>
<skill-hint>No skills matched automatically. If this looks like a repeatable
workflow, call skills_search before proceeding.</skill-hint>
<ide_state>
  open_folder: /home/alansrobotlab/lloyd
  visible_file: prefetch.py
  open_tabs: [prefetch.py, config.yaml]
</ide_state>
</context>

[original user message]
```

**Skill hint** fires only when no skill matched AND the message isn't a
continuation. `_CONTINUATION_RE` matches openers like "ok", "yes please", "let's
go", "continue", "sounds good"; messages under 15 chars also count. Continuations
never start a new workflow, so the nudge would be pure noise. The old
low-confidence branch (hint when a skill matched but scored below
`SKILL_HIGH_CONFIDENCE = 6.5`) was dropped — the model trusted low-confidence
auto-picks ~95% of the time regardless, so it cost ~65 tokens per firing for
nothing.

**IDE state** is read from the in-process MC state mirror (`app.mc_state.get_ide_snapshot`)
— no HTTP round-trip — and spliced in just before the closing `</context>` tag.

## Subliminal capture (#306)

The injected prefix is ephemeral: it goes to the harness but the session JSON would
otherwise never see it. `app/routers/_messages_subliminal.py` recovers it by
diffing `prefetched_text` against the original `text` and persists it as a
`role="subliminal"` message right after the user message, so the chat UI can show
what the agent actually saw.

Three injection shapes are classified (first match wins):

| Kind | Marker |
|---|---|
| `memory_nudge` | leading `<system-reminder>` |
| `ambient_envelope` | leading `<ambient ` (text is *inside* the wrapper, so the whole thing is the injection) |
| `prefetch` | leading `<context>` |

Source badges are detected by tag presence: `ambient`, `skills`, `backlog`,
`facts`, `vault`, `sessions`, `hint`. Scripts that filter by role skip these
entries for free.

The same prefix is passed to Inner Voice as `subliminal_context` (capped at 4000
chars) so the observer can tell when the primary is following documented procedure
rather than freelancing — see [[inner-voice]].

## Post-session enrichment

Background tasks fired via `asyncio.ensure_future` after a turn completes
(`app/post_capture.py`):

### `_post_session_capture`

1. Skips autonomy sessions and already-`captured` sessions.
2. Exports the session as searchable markdown into the vault sessions collection
   (which is why `sessions` is a QMD collection the vault worker searches).
3. Builds a ≤4000-char transcript and asks the secondary model for a summary. A
   `TRIVIAL` verdict marks the session captured and stops.
4. Appends the summary to the daily note.
5. With ≥3 user messages, extracts durable facts and writes them to
   `~/obsidian/facts/{entity}/`. Confidence 0.75, provenance `EXTRACTED`, linked to
   the source session. Cuts fact-extraction latency from the 11-hour nightly cycle
   to seconds.
6. Marks `captured` via `mutate_session` — never writes the stale snapshot back,
   because the model call can take 10+ seconds during which new turns may append.

### `_maybe_extract_focus`

Tier-2 topic extraction (above).

### Memory preservation nudge

- **Trigger:** every 20 user turns within a session
- **Method:** `<system-reminder>` prepended to the (already prefetched) user message
- **Content:** reminds the agent to call `memory_add` or `fact_add` for undocumented
  decisions
- **Purpose:** prevents context loss during compaction

## Tuning constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `PREFETCH_BUDGET_MS` | 300 | Hard wall on the parallel phase; stragglers dropped |
| `MIN_MESSAGE_LEN` | 10 chars | Skip search workers (ambient still drains) |
| `SKILL_THRESHOLD_FIRST` | 3.0 | Min score to inject first skill |
| `SKILL_THRESHOLD_SECOND` | 4.0 | Min score to inject second skill (excerpt) |
| `SKILL_HIGH_CONFIDENCE` | 6.5 | Legacy; low-conf hint branch was removed |
| `SKILL_BODY_MAX` | 6000 chars | Max first skill body |
| `SKILL_EXCERPT_MAX` | 500 chars | Max second skill excerpt |
| `_BODY_HITS_CAP` | 4 | Cap on body-token hits in skill scoring |
| `FACT_MAX_ENTITIES` | 2 | Max entities to look up |
| `FACT_MAX_PER_ENTITY` | 3 | Max facts per entity |
| `VAULT_MAX_RESULTS` | 5 | Max vault results |
| `VAULT_SNIPPET_MAX` | 500 chars | Max chars per vault snippet |
| `VAULT_MIN_SCORE` | 0.5 | Min vault relevance score |
| `VAULT_MIN_QUERY_LEN` | 25 chars | Skip vault for short queries (post-enrichment) |
| `SESSION_PREFETCH_LIMIT` | 3 | Max recent sessions |
| `SESSION_PREFETCH_DAYS` | 3 | Session lookback window |
| `SESSION_PREFETCH_SNIPPET_MAX` | 200 chars | Max chars per session snippet |
| `BACKLOG_MAX_REFS` | 3 | Max backlog refs per turn |
| `BACKLOG_CACHE_TTL` | 60 s | Backlog index rescan interval |
| `BACKLOG_BODY_EXCERPT_MAX` | 300 chars | Max backlog body excerpt |
| `AMBIENT_PREFETCH_CAP` | 5 | Max queued ambient entries per session |
| `AMBIENT_PREFETCH_DRAIN_MAX` | 3 | Max ambient entries injected per turn |
| `FOCUS_DECAY` | 0.75 | Per-turn keyword weight decay |
| `FOCUS_TOP_K` | 6 | Focus keywords used to enrich vault query |
| `FOCUS_MIN_WEIGHT` | 0.3 | Min weight for a keyword to be used |
| `FOCUS_EXTRACT_INTERVAL` | 5 turns | Topic extraction frequency |
| `FOCUS_MAX_SESSIONS` | 50 | Max tracked sessions before LRU eviction |
| `_SKILL_CACHE_TTL` | 300 s | Skill-list cache TTL |

## Files

| File | Purpose |
|------|---------|
| `~/lloyd/prefetch.py` | Prefetch layer: 5-worker parallel search under a hard budget, `SessionFocus`, context formatting, IDE state |
| `~/lloyd/prompt_builder.py` | System prompt assembly (SOUL.md + memories + skills index + goal/plan/todos + turn discipline) |
| `~/lloyd/app/routers/messages.py` | Integration: prefetch call, 20-turn memory nudge, subliminal persistence, post-session task dispatch |
| `~/lloyd/app/routers/voice.py` | Same prefetch call on the voice path |
| `~/lloyd/app/routers/_messages_subliminal.py` | #306 injected-prefix extraction, classification, and `role="subliminal"` entry shaping |
| `~/lloyd/app/post_capture.py` | Post-session capture, markdown export, fact extraction, focus topic extraction |
| `~/lloyd/app/secondary_models.py` | Secondary-model endpoint resolution + capture/fact/focus prompts |
| `~/lloyd/app/sessions_io.py` | Ambient prefetch queue (`enqueue_ambient_prefetch` / `drain_ambient_prefetch`) |
| `~/lloyd/agent_mcp/vault.py` | `_qmd_daemon_search`, `_qmd_strip_stopwords`, QMD payload construction |
| `~/lloyd/agent_mcp/facts.py` | `_extract_entities_from_query`, `_get_facts_sync` |
| `~/lloyd/agent_mcp/skills.py` | `_iter_skills`, `_score_skill`, `_tokenize`, `_query_tokens` |
| `~/lloyd/agent_mcp/session.py` | `_load_session_index`, `_score_session` |
| `~/lloyd/agent_mcp/subliminal.py` | Legacy `subliminal_recall` tool (keyword extraction + SOUL.md) — rarely used |

## Infrastructure

| Service | Port | GPU | Purpose |
|---------|------|-----|---------|
| QMD daemon | 8181 | GPU 0 | Hybrid BM25 + vector search, embedding model |
| vLLM primary (qwen3.8-flash-next) | 8096 | GPU 1 (RTX PRO 6000, `--gpu-memory-utilization 0.93`) | Main agent model; also serves capture/fact/focus extraction while `secondary_enabled: false` |
| vLLM secondary (gemma-4-e4b-nvfp4) | 8091 | configurable | Intended host for capture/fact/focus + Inner Voice. **Currently STOPPED.** |
| Qwen3-TTS | 8090 | GPU 0 | Voice output |
| lloyd-mcp aggregator | 8500 | — | All agent tools |

## Evaluation

`eval/run_eval.py` + `eval/vault_recall_queries.yaml` — 21 hand-written queries over
the live personal vault, scored for `entity_hit` / `doc_hit` / `topk_overlap`
against expected entities and documents. Written to baseline retrieval before and
after graph-vote re-ranking changes.

Current official baseline (2026-08-06; all prior baselines void for entity-side
metrics after the #380 Phase 0b expectation audit):

| Metric | Value |
|---|---|
| Overall MRR | 0.359 |
| entity_hit | 0.65 |
| ent_recall | 0.47 |
| fER | 0.392 |
| **multi-hop MRR** | **0.022** |
| Latency | 5,063 ms |

Two things to be honest about:

1. **Multi-hop retrieval effectively does not work** (MRR 0.022). This is the
   largest known quality gap in the subsystem.
2. This measures `_vault_recall`, the *explicit tool path* with reranking on — not
   the prefetch path, which uses `skip_rerank=True`, a 300 ms budget, and
   focus-enriched queries. **There is no eval for the prefetch path as deployed.**
   The gap matters: the injected-vs-explicit paths differ in ranking, latency, and
   query text.

The #380 Phase 0b audit is worth remembering as a methodology note: 9 of 46
`expect_entities` were *unsatisfiable* — no entity directory contained the string —
so they silently capped `ent_recall`/`fER` and masked graph changes. The rule
applied was to remove an expectation only when factually wrong, and replace it only
with the genuinely correct answer, never an easier target.

## History

- **2026-03-24:** Phase 3 deployed via `appendSystemContext` in OpenClaw — cache
  invalidation discovered
- **2026-03-27:** Cache problem analyzed; design doc for context event migration
- **2026-03-28:** Phase 3.5 — `context` plugin hook added to OpenClaw core,
  subliminal migrated to ephemeral injection
- **2026-04-05:** Lloyd migration — system moved off OpenClaw; subliminal replaced
  by `prefetch.py`
- **2026-04-14:** Active Memory (#289) — vault search added as a third parallel
  worker, conversation focus tracking, inline fact extraction, memory nudge
- **2026-04-20:** qmd `searchVec` patch + embedding LRU cache; `PREFETCH_BUDGET_MS`
  hard budget introduced with drop-on-timeout
- **2026-06:** Ambient signal drain (#295 Mechanism 1); subliminal capture as
  `role="subliminal"` session entries (#306)
- **2026-07:** Skill scoring `require_metadata_hit` gate (#311); low-confidence
  skill-hint branch removed
- **2026-08-06:** Retrieval eval expectation audit (#380 Phase 0b); new official
  baseline
- **2026-08:** Backlog task-ID ref worker; recent-session worker; IDE state block;
  plan-mode synthetic skill tokens
- **2026-09-03:** Doc rewritten against the code — worker count corrected 3 → 5,
  latency budget, ambient/backlog/session/IDE sources, #306 capture, eval baseline
  and its scope caveat documented
