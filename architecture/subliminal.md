---
segment: architecture
tags: [architecture,subliminal,memory,prefetch]
type: architecture
updated: 2026-09-11
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
    ├─→ build_system_prompt(todos, plan, goal, session_id)  [per-session, static]
    │   ├─ SOUL.md (identity + operating contract, and since 2026-09-08 the
    │   │   only home of the anti-compliance rules — the Python
    │   │   ANTICOMPLIANCE_DIRECTIVE that used to be prepended ahead of it
    │   │   was a second, already-diverged copy of the same six rules, and
    │   │   only the vault one was reachable by the promotion path (#465))
    │   ├─ <memory> — MEMORY.md + USER.md
    │   ├─ <available_skills> (names only)
    │   ├─ <goal> / <plan> / <active_todos>
    │   ├─ Platform hint  ← deliberately timestamp-free (prefix cache)
    │   ├─ Background-bash contract
    │   └─ Turn-discipline clause  ← pairs with IV stall rescue
    │
    ├─→ prefetch_context_async(text, session_id, plan_mode)   [per-message]
    │   │
    │   ├─→ drain_ambient_prefetch(session_id)      [loop thread, before any guard]
    │   │
    │   ├─→ SessionFocus.update(text)               [loop thread, in-memory]
    │   │   Accumulate weighted keywords, decay old by 0.75/turn
    │   │
    │   └─→ asyncio.to_thread → ThreadPoolExecutor(max_workers=6)  [≤300ms hard budget]
    │       ├─ Worker 1:  _search_skills()          Token overlap, metadata-hit required
    │       ├─ Worker 2:  _search_facts()           Entity extraction → fact lookup
    │       ├─ Worker 3a: _search_vault(lex)        QMD lex leg, short AND terms — lands when warm
    │       ├─ Worker 3b: _search_vault(lex+vec)    QMD hybrid; starts after 3a returns,
    │       │                                       straggles, result carried to next turn
    │       ├─ Worker 4:  _search_recent_sessions() Temporal queries only
    │       └─ Worker 5:  _search_backlog_refs()    #NNN / bare-number task IDs
    │       ⤷ stragglers abandoned at the deadline, left running detached
    │
    ├─→ _format_ide_state()                         [in-process MC state mirror]
    │
    ├─→ Memory nudge injection            [stream path only, every 20 user turns]
    │   <system-reminder> nudge to capture undocumented decisions
    │
    ├─→ log_turn_prompt_budget(prefetched_text)     [one PROMPT_BUDGET line/turn]
    │
    ├─→ persist role="subliminal" session entry     [#306, UI visibility]
    │
    └─→ run_query(messages, options)
        │
        ↓
    LLM sees: [system prompt] [history] [<context> block + user msg]
              ←── prefix-cached ──→     ←── uncached (~2-8KB) ──→
```

`prefetch_context_async` is awaited from [messages.py](../app/routers/messages.py)
(stream + sync chat paths) and [voice.py](../app/routers/voice.py). Its cheap half
(ambient drain, focus update) runs on the event-loop thread in request order; the
budgeted search phase runs in a worker thread via `asyncio.to_thread`, so the
loop keeps serving other sessions, SSE streams, and Inner Voice while it waits.
Before 2026-09-03 the sync call blocked the loop for the full budget on every
message. Callers pass `plan_mode` through so prefetch does not re-read the
session JSON they just loaded. Ambient turns (`build_ambient_turn`) skip prefetch.
The sync `prefetch_context` remains for scripts and tests.

Two things follow from *where* the call sits that are easy to miss. Nothing in
this path gates on platform, and every session-backed worker source posts to
`/api/message/stream` — so an `autocode`, `autotriage`, `youtube-digest` or
`deep-research` turn is prefetched exactly like a chat turn and leaves a
`role="subliminal"` row of its own. (A direct-path run — `run_prompt_on_primary`,
`autonomy.run_task` — calls `run_query` itself and is prefetched by nothing; see
[[background-runs]].) And the 20-turn memory nudge lives in the **streaming**
handler only: the sync chat path and the voice path call `prefetch_context_async`
without it, so a session driven entirely through either never sees one.

## Latency budget

`PREFETCH_BUDGET_MS = 300` is a **hard wall**. Workers are submitted to a
`ThreadPoolExecutor` and collected with `wait(..., FIRST_COMPLETED)` in a loop
against the remaining budget. Anything unfinished at the deadline is dropped for
this turn and logged:

```
prefetch budget=300ms exceeded, dropped=facts
```

The pool is deliberately **not** used as a context manager — `with` blocks on
`__exit__` until every future finishes, which would defeat the whole point.
`pool.shutdown(wait=False)` lets stragglers finish detached; Python GCs the pool
once their threads return.

Measured 2026-09-03 on this host (warm process, 283 skills, 328 backlog files,
QMD daemon on GPU 0):

| Worker | Before | After | Note |
|---|---|---|---|
| skills | ~83 ms | ~0.2 ms | token sets memoized on the cached skill dicts |
| backlog (index refresh) | ~216 ms every 60 s | ~1.2 ms | incremental mtime scan |
| facts | ~5 ms warm | ~5 ms | ~150 ms once per process (entity index) |
| sessions | <5 ms | <5 ms | temporal queries only |
| vault lex leg | — | 6–50 ms warm; 0.5–1.3 s per cold term | 1–4 short calls; soft-waited 150 ms, then carried over |
| vault lex+vec | 1.1–2.6 s | 1.1–2.6 s | never lands; carried to the next turn |

Before the fix every one of the last 20 logged turns hit the wall (301–348 ms):
vault was dropped 20/20 and facts 12/20. The vault number was structural — the
vec leg's query embedding costs 1.1 s even on a repeated query and 2.4–2.7 s on
a novel one, so the "~290 ms warm" that the old code comment cited was never
achievable. Facts (5 ms of work) were being starved: skill scoring re-tokenized
1.5 MB of skill bodies every turn and the backlog index re-read and YAML-parsed
328 files once a minute, both holding the GIL inside the budget window.

Two QMD facts shape the vault design:

- **The daemon serializes requests.** It is one node process; a lex query
  fired 50 ms after a vec query waits ~1.2 s behind it. So the hybrid leg is
  started only after the lex leg has returned (`after=` future), never
  alongside it.
- **The vec leg cannot be made to fit.** Rather than drop it every turn (and
  waste the daemon's GPU work), its result is stashed on the `SessionFocus` and
  injected on the *next* turn, labelled as such. See Worker 3.

- **A term the daemon has never seen is expensive.** The first lex query
  containing a new term costs 0.5–1.3 s (FTS pages cold); the same term is
  6–50 ms afterwards. So the lex leg is waited on for only
  `VAULT_LEX_SOFT_WAIT_MS` (150 ms) once every other required worker is in.
  If it is still running it straggles like the hybrid leg and its result
  carries over — the first turn on a new topic pays ~150 ms instead of
  pinning at 300 ms for nothing, and the second turn sees both legs' hits.

Because the daemon is serial, any other QMD client holding it — the previous
turn's hybrid straggler if turns come faster than ~3 s, or an autonomy task
calling `vault_search` with reranking on — pushes the lex leg past the budget
for that turn; the carried-over result from the previous turn still lands.
**Neither vault leg can ever appear in `dropped=`.** The drop list is built by
skipping both of them by name, because a straggling leg is not a loss — its
result carries over — so the only names that line can carry are `skills`,
`facts`, `sessions` and `backlog`. A lex leg that missed the wall shows up in
the debug line instead, as `vault=N(lex-straggling)`.
The wait loop only waits on the required
workers, so a turn ends as soon as the slowest of them returns rather than
sitting at the wall; the debug line `prefetch NNNms: ... landed=skills@1ms
facts@6ms vault@140ms` shows when each one arrived.

## Prefetch workers

### Worker 1: Skill search

- **Source:** `~/obsidian/skills/` + `~/lloyd/skills/`, via `_iter_skills()`
  (`agent_mcp/skills.py::SKILLS_DIRS`). Only the vault root exists today —
  ~190 skills; the repo-local root is in the list and absent on disk, which
  `_iter_skills` skips silently. `prompt_builder` carries the same two roots
  and imports `_QUARANTINE_STATUSES` from the MCP module rather than
  restating it, so the advertised index and the retrievable set cannot drift.
- **Method:** `_score_skill(skill, tokens, require_metadata_hit=True)` —
  `name×3.0 + desc×2.0 + tags×1.5 + min(body_hits, 4)×0.3`
- **Metadata-hit gate:** a skill with zero name/desc/tag overlap scores 0.0
  regardless of body accidents. Fixes #311, where generic stopword queries pulled
  powerpoint/youtube skills into graph-classifier sessions. Body hits are a
  tiebreaker, not a qualifier.
- **Output:** first skill full body (≤6000 chars), second skill excerpt (≤500 chars)
- **Thresholds:** ≥3.0 for the first, ≥4.0 for the second
- **Cache:** in-memory skill list, rebuilt only when a `SKILL.md` mtime changes
  (a `(dir, mtime)` signature over both roots is re-checked at most every 15 s,
  ~1 ms). Name/description/tag/body token sets are memoized on each cached dict
  by `skills._skill_token_sets` under a private `_tok` key, so scoring the
  whole set is one set intersection per skill (~0.2–1 ms; 283 skills when
  this was measured, ~190 today) instead of re-tokenizing 1.5 MB of bodies
  per turn.
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

- **Source:** the fact tree at `_pipeline/vault-derived/facts/<Entity>/`
  (`app.paths.VAULT_FACTS_ROOT`, overridable by `LLOYD_FACTS_ROOT` so a
  rebuild can extract into a fresh tree). It was `~/obsidian/facts/` until
  the 2026-09 store migration moved the derived half out of the vault — see
  [[knowledge-graph]].
- **Method:** `_extract_entities_from_query()` ranks known entity *directories*
  against the query — task-id dispatch first (`#299`, `backlog_18` →
  `Task #N`), then verbatim-name substring, then squared token overlap
  normalized by both token counts, with deterministic tie-breaks →
  `_get_facts_sync(entity)` → drop expired/invalidated → sort by confidence.
  Both helpers live in `agent_mcp/retrieval.py` since the #340 module split
  and are re-exported from `agent_mcp/facts.py` under their old underscore
  names, which is what prefetch still imports.
- **Output:** top 2 entities × 3 facts each = ≤6 fact lines
- **No LLM:** pure regex + file reads

### Worker 3: Vault search (QMD)

- **Source:** QMD daemon on `:8181` — BM25 (`lex`) + vector (`vec`) search
- **Collections:** `memory, knowledge, projects, lloyd, work, sessions,
  architecture, autonomy, backlog`. facts and skills are excluded (dedicated
  workers); personal and people are left to the explicit tool. The last three
  were added 2026-09-03: 8 of 20 eval queries expect docs there, and the
  agent's own design docs and task notes are what it needs mid-task. Warm lex
  cost for 6 → 9 collections: +5–30 ms.
  This list is also what keeps prefetch on qmd's **per-collection** path. Since
  2026-09-09 `_qmd_daemon_search` sends `collections: []` — one global ANN scan
  instead of eleven exact ones — but only when the caller's list is a superset
  of `VAULT_SEGMENTS`. Prefetch's nine omit `personal`, `people` and `skills`
  and add `sessions`, so it never qualifies and never will while the exclusions
  above hold. That is the intended trade, not an oversight: a global top-k can
  starve a small collection, and prefetch is already scoped deliberately.
- **Two legs, sequenced:**
  - **3a `lex` only** — 6–50 ms per call once the terms are warm, 0.5–1.3 s
    on a term's first sight. Waited on for at most 150 ms after the other
    workers land; if still running it straggles and stashes its result for
    the next turn (`_search_vault_lex_and_stash`). No new call starts within
    90 ms of the 2 s straggle deadline.
    Injected as this turn's `<vault-context>`. QMD's lex leg is FTS5 with
    implicit **AND**: every term must match, `OR` is not passed through, and
    one unmatched term returns nothing ("alfie servo shoulder pid" → 2 hits,
    plus "zzzqqq" → 0). A long focus-enriched query therefore returned
    nothing lexically. The leg now runs up to three short sub-queries (the
    message's own terms ordered by focus weight, the top focus keywords, up
    to two Tier-2 topic phrases; ≤4 terms each) and, when one returns
    nothing, drops its lowest-weight term and retries down to 2 terms (or 1,
    when the query itself is a single term like "qmd") — at most 6 daemon
    round-trips, fewer when the deadline is near. Results are merged by file,
    best score kept. Measured end to end: 21–25 ms on warm-term turns, ~150 ms
    on cold-term turns, with every other worker in by ~6 ms.
  - **3b `lex+vec` hybrid** — started only after 3a returns (the daemon
    serializes requests). Takes 1.1–2.6 s, so it always straggles. Its result
    is stashed on the session's `SessionFocus` and merged into the **next**
    turn's `<vault-context>`, deduped against that turn's lex hits and rendered
    with `semantic hit from the previous turn's query`. Its two legs get
    different text: the lex leg the message's short term list (AND
    semantics), the vec leg the focus-enriched sentence — with one string for
    both, the hybrid's lex component returned nothing on enriched queries.
    In the merge, `VAULT_CARRIED_MIN_SLOTS` (2) of the 5 slots are reserved for
    carried hits when there are any: lex scores are normalized per query, so a
    page of confident-looking lex misses used to evict the one semantic hit
    the previous turn found. A stash older than
    `VAULT_CARRY_MAX_AGE_S` (15 min) is discarded. If 3b ever does land in
    budget, its (superset) result is used directly and the stash is cleared.
  - Why carry over rather than drop: across five sample queries the vec leg
    contributed 1–4 results above the 0.5 floor that lex alone missed (e.g.
    "inner voice observer stall rescue": hybrid 4 hits, lex 0). Consecutive
    turns share most of their focus-enriched query, so a one-turn delay keeps
    most of that recall at zero added latency and zero added daemon load — the
    hybrid request was already being fired and thrown away every turn.
- **Query:** stopword-stripped on *both* legs — conversational framing ("tell
  me about X") drifts the vec embedding away from content, so the strip is not
  a lex-only concern. The two legs no longer get the same *text*: `lex_query`
  overrides the lex leg with the short AND-friendly term list while the vec leg
  keeps the focus-enriched sentence (see 3b). Before that they shared one
  string and the hybrid's lex component returned nothing on every enriched
  query.
- **`skip_rerank=True`** in prefetch. The reranker rarely changes top-1 and
  mostly shuffles within top-5 — not worth the tax on a latency-critical path.
  Explicit `vault_recall` calls keep reranking on.
- **Output:** top 5 results, ≤500-char snippets, minimum score 0.5 (0.3 was too
  noisy)
- **Focus enrichment:** the query is augmented with conversation focus keywords
  (below)
- **Gating:** the `VAULT_MIN_QUERY_LEN` floor applies to the **hybrid leg
  only**. It is checked against the *effective* (focus-enriched) query inside
  `_search_vault`, and the lex ladder calls that function with
  `min_query_len=1` and `focus=None` — it has already built its own short term
  lists, and laddering a two-word query down is the whole point of 3a. So a
  12-character message still gets lexical hits and gets no semantic ones.

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
  Live IDs span 9–890 (9–387 when this was written; the board has roughly
  doubled twice since), which collides with HTTP status codes and millisecond
  values, so a **bare** number is also rejected when followed by a unit
  (`300ms`, `512mb`, `3 sec`) or preceded by code/port context (`HTTP 302`,
  `returned a 404`, `port 8080`, `BUDGET_MS = 300`). `#NNN` is never filtered.
- **Source:** `~/obsidian/backlog/*.md`, indexed by leading task ID. The
  directory is re-stat'ed every 60 s (~1 ms for 330 files; ~830 today) and only files whose
  mtime changed are re-read and parsed; the old rebuild re-parsed every file
  (~216 ms of GIL-held CPU landing inside the budget once a minute). A stale
  cache is retained on rescan failure rather than breaking prefetch.
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
  back for the next turn. The drain sorts **newest first**, so what gets held
  over is the oldest — a producer that fired twice while the user was typing
  surfaces its fresher entry this turn, not its stalest.
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
  evicts anything below 0.1, then adds this message's keywords at +1.0. Trailing
  punctuation is stripped first so a sentence-final `servo.` accumulates onto
  `servo` rather than a separate key.
- **Thread safety:** the tracker is mutated from the loop thread (`update`,
  topic extraction) and read from prefetch worker threads (`enrich_query`, the
  vault carry-over stash), so every access takes the instance lock; the registry
  itself is guarded by a module lock around LRU eviction.
- **Output:** `enrich_query()` appends extracted topics + top-6 focus keywords
  (weight ≥ 0.3, excluding words already in the message) to the hybrid vault
  query; `lex_subqueries()` builds the short term lists for the lex leg.
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
  `app/secondary_models.py`. `secondary_enabled` has been **true** since
  2026-09-03 (`de893d7`), so these calls land on the real secondary slot —
  `:8091`, Qwen3.6-35B-A3B UD-Q3_K_XL under llama.cpp on GPU 2 since
  2026-09-06. While the flag was false the alias rewrote to the primary at
  `:8096` and every caller asking for the secondary silently got Flash-Next;
  that rewrite still fires if anyone turns it back off, and logs itself once
  per name when it does. The slot is single-tenant (`--parallel 1`), so a
  topic extraction queues behind whatever post-session summary, voice rewrite
  or session title is already there — which is the whole reason this call is
  fired after the turn and never inside it.
- **Timing:** background `asyncio.ensure_future()` after the turn completes — zero
  user-facing latency.
- **One-turn delay:** results are available on the NEXT message, not the current one.
- **Filtering:** injected context blocks (`<context>`, `<system-reminder>`,
  `<memory>`, `<daily_notes>`) are stripped from the transcript so the extractor
  doesn't summarize its own prior injections.
- **Prompt:** "Extract 3-5 short topic phrases (2-4 words each), one per line"
- **Temperature:** 0.0. **Timeout:** 15 s, non-fatal. Kept only if 2–6 words.
- **Attempt accounting:** `mark_topic_attempt()` is called whether or not
  topics came back. Previously an empty or failed extraction left
  `topics_turn` untouched, so the model call re-fired on every subsequent turn
  instead of every 5. Both outcomes now log at INFO. Note the tracker is
  in-process: a backend restart resets `turn_count`, so a session has to reach
  5 turns without a restart before Tier 2 fires at all.

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
go", "continue", "sounds good"; messages under 15 chars also count. Only the
closed forms of "please …" and "let's …" count — "please review the module" and
"let's build a new feature" start new work and keep the hint. Continuations
never start a new workflow, so the nudge would be pure noise. The old
low-confidence branch (hint when a skill matched but scored below
6.5 — the constant has since been removed) was dropped — the model trusted low-confidence
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
`facts`, `vault`, `sessions`, `hint`, `ide`. Scripts that filter by role skip
these entries for free.

The prefix is recovered by stripping `"\n\n" + text` from the end of the
prefetched text; a single `"\n"` separator is accepted too. Without that, a
memory nudge firing on a turn with no `<context>` block (nudge + `"\n"` + text)
fell through to the envelope branch and the user's own text was persisted as
part of the "injection".

**A `subliminal` row is a record, never an input.** It is written to the session
JSON *after* the user message and is read back by nobody who talks to the model.
`load_and_compact_session` filters to `user`/`assistant`/`tool`/`system` before
it counts a single token, and `_prepare_messages_for_harness` filters again to
`user`/`assistant`/`tool` — so the injected block reaches the engine exactly
once, inside the turn that built it, as the prefix on that turn's user message.
That is the point: the prefix is rebuilt fresh every turn from the current
message and the current focus, and replaying last turn's would be both stale and
paid for twice. The same two filters are why a script that walks a session by
role skips these entries for free.

**`role="thinking"` is the same trick, used again.** `_messages_thinking.py`
follows this module's precedent deliberately — one entry per reasoning phase,
a role no transcript producer has a branch for, `content` left empty as a
second layer — and it lands on the same side of both filters. See [[harness]].
The one place the two roles share a fate worth knowing about: the `/compact`
slash command — `_slash_compact_sse`, handled inline by `post_message_stream`
before the queue, which force-summarizes with no threshold check because the
user explicitly asked — rewrites `data["messages"]` to the conversation-only
set. So it discards every `subliminal` and `thinking` row in the session, not
just those in the block it summarized. That is called
out at the write site as known and deliberate — putting UI-only rows back at
the right positions is a merge that swap cannot express — and for the thinking
trace the event log's `brain1.thinking_block_emitted` remains the durable
record. The subliminal rows have no such backup: after a hard compaction, what
the agent saw on those turns is gone.

The same prefix is passed to Inner Voice as `subliminal_context`, trimmed to
`_SUBLIMINAL_PROMPT_CHAR_CAP` (4000) chars so the observer can tell when the
primary is following documented procedure rather than freelancing — see
[[inner-voice]]. The trim is two-stage and head+tail at both stages: each
`<skill>` body is windowed to 1800 chars *first*, then the whole block to 4000.
A single head-only cut used to spend the entire budget on one skill body and
drop exactly the small sections — facts, vault hits, IDE state — that tell the
observer what the primary actually knew.

## Post-session enrichment

Background tasks fired via `asyncio.ensure_future` after a turn completes
(`app/post_capture.py`):

### `_post_session_capture`

1. Skips already-`captured` sessions.
2. Exports the session as searchable markdown. A user session goes to
   `vault-derived/sessions/` — which is why `sessions` is a QMD collection the
   vault worker searches — and a background one to
   `vault-derived/sessions-background/`, which qmd does not watch. The watcher
   *embeds* on every change, and the background sessions that reach this task
   are ~70 a day (479 in the week to 2026-09-10, against 99 chats). A
   background session then stops here, marked `captured`, without steps 3–5:
   the summary, the daily note and fact extraction are about the user's day.
   Only chat-path turns fire this task, so what arrives is the session-backed
   workers; an autonomy or `run_prompt_on_primary` run's record is its session
   JSON. See [[background-runs]].
3. Builds a ≤4000-char transcript and asks the secondary model for a summary. A
   `TRIVIAL` verdict marks the session captured and stops.
4. Appends the summary to the daily note.
5. With ≥3 user messages, extracts durable facts and writes them through
   `_fact_add` into `_pipeline/vault-derived/facts/<Entity>/` under category
   `session-extracted`. Confidence 0.75, provenance `EXTRACTED`, linked to
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
| `VAULT_CARRY_MAX_AGE_S` | 900 s | Max age of a carried-over hybrid vault result |
| `VAULT_LEX_MAX_TERMS` | 4 | Terms per lex sub-query (AND semantics) |
| `VAULT_LEX_MIN_TERMS` | 2 | Ladder floor when a sub-query returns nothing |
| `VAULT_LEX_MAX_CALLS` | 6 | Cap on lex round-trips per turn |
| `VAULT_LEX_DEADLINE_MARGIN_S` | 0.09 s | No new lex call this close to the ladder deadline |
| `VAULT_LEX_SOFT_WAIT_MS` | 150 | Max extra wait for the lex leg once other workers are in |
| `VAULT_LEX_STRAGGLE_MAX_S` | 2.0 s | Ladder deadline once the lex leg is straggling |
| `VAULT_CARRIED_MIN_SLOTS` | 2 | Vault slots reserved for carried-over hits |
| `_SKILL_CACHE_CHECK_S` | 15 s | How often the skill-dir mtime signature is re-checked |
| `PROMPT_BUDGET_CHARS` | 80,000 | Shared system+injected budget `log_turn_prompt_budget` measures the turn against (`prompt_builder`). Nothing is truncated at it — an overrun is a loud `over_budget=True` in the `PROMPT_BUDGET` line, not a silently cut identity file |

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
| `~/lloyd/agent_mcp/retrieval.py` | `extract_entities_from_query`, `get_facts_sync` — the real home since the #340 split; `agent_mcp/facts.py` re-exports both under their `_`-prefixed names, which is what prefetch imports |
| `~/lloyd/app/compaction.py` + `app/routers/_messages_harness_adapter.py` | the two role filters that keep `subliminal` (and `thinking`) rows out of the prompt |
| `~/lloyd/agent_mcp/skills.py` | `_iter_skills`, `_score_skill`, `_tokenize`, `_query_tokens` |
| `~/lloyd/agent_mcp/session.py` | `_load_session_index`, `_score_session` |
| `~/lloyd/agent_mcp/subliminal.py` | **Removed 2026-09-03.** Legacy `subliminal_recall` tool (keyword extraction + a full SOUL.md dump). Zero calls across the 40 sessions on disk, and SOUL.md is already in the system prompt. The `subliminal` QMD collection in `qmd-index.yml` is unrelated and stays. |
| `~/lloyd/eval/run_prefetch_eval.py` | Prefetch-path retrieval eval: lex leg, hybrid straggler, and their merge, scored against `vault_recall_queries.yaml` expect_docs |
| `~/lloyd/tests/test_prefetch.py` | Hermetic tests: capture helpers, focus tracking, task-ref precision, skill-score memo, budget drop + carry-over, leg ordering, session-index window, incremental backlog index |

## Infrastructure

| Service | Port | GPU | Purpose |
|---------|------|-----|---------|
| QMD daemon | 8181 | GPU 0 | Hybrid BM25 + vector search, embedding model. Single node process — serializes requests; vec leg 1.1–2.6 s, lex leg 10–80 ms and AND-only |
| vLLM primary (Qwen3.8-Flash-Next) | 8096 | GPU 1 (RTX PRO 6000, `--gpu-memory-utilization 0.9345`) | Main agent model. Served capture/fact/focus extraction too while `secondary_enabled` was false |
| llama.cpp secondary (Qwen3.6-35B-A3B UD-Q3_K_XL) | 8091 | GPU 2 (RTX 3090) | Capture / fact / focus / title extraction. Running, `secondary_enabled: true`. Single-tenant (`--parallel 1`) — these jobs queue behind each other. **Not** Inner Voice: the observer is pinned to `primary` in config.yaml after a 4B briefly landed in this slot and intervened on 40% of judged events |
| Qwen3-TTS | 8090 | GPU 0 | Voice output |
| lloyd-mcp aggregator | 8500 | — | All agent tools |

## Evaluation

`eval/run_eval.py` + `eval/vault_recall_queries.yaml` — 20 hand-written queries over
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
   focus-enriched queries. The prefetch path has its own eval (below); the two
   are not comparable — different limits, score floors, and query text.

### Prefetch-path eval

`eval/run_prefetch_eval.py` runs the same 20 queries through the vault leg as
deployed, as a *first turn* (fresh `SessionFocus`, no prior context — the
hardest case), and scores `doc_hit` / MRR against `expect_docs` for the lex
leg, the hybrid straggler, and their merge (what the next turn injects). Run it
twice back to back to see cold-term and warm-term latency; hit rates are the
same either way. Outputs land in `eval/baselines/` (gitignored).

| Metric | 2026-09-03 before | after (cold terms) | after (warm terms) |
|---|---|---|---|
| lex doc_hit | 0.20 | 0.35 | 0.35 |
| hybrid doc_hit | 0.20 | 0.50 | 0.50 |
| merged doc_hit | 0.25 | 0.60 | 0.55–0.60 |
| merged MRR | 0.127 | 0.253 | 0.24–0.25 |
| lex p50 / p90 | 940 / 1462 ms | 1336 / 1641 ms | 39–41 / 76–695 ms |
| lex landed in budget | 25% | 10% | 85–95% |
| hybrid p50 | 2.35 s | 2.18 s | 1.32 s |

"Before" is the first run, with 6 collections, one enriched string for both
hybrid legs, a 2-term ladder floor that skipped single-term queries, and no
reserved carried slots. The hit-rate gain comes from the three added
collections and the per-leg hybrid queries; the latency gain is the soft wait
(cold-term turns no longer sit at the wall) plus warm FTS pages. Known
remaining misses: expectations under `skills/` (excluded by design — the
skills worker covers them), `technical` queries whose expected docs are source
files (`agent_mcp/…`, `app/…` — the vault leg does not grep code; the explicit
tool does), and multi-hop questions.

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
- **2026-09-03 (perf/efficacy review):** prefetch moved off the event loop
  (`prefetch_context_async`); skill token sets memoized + mtime-gated skill
  cache; incremental backlog index; vault split into lex leg (lands) + hybrid
  straggler carried to the next turn, sequenced because QMD serializes; bare
  task-number unit/code exclusions; session-index cache keyed on window; focus
  keyword punctuation, locking, and topic-attempt accounting; single-newline
  nudge capture; `ide` badge; `tests/test_prefetch.py` added. Follow-ups the
  same day: `agent_mcp/subliminal.py` deleted; observer subliminal cap made
  head+tail with `<skill>` blocks trimmed first; continuation regex no longer
  swallows "please X" / "let's X"; memory-nudge turn counter ignores ambient
  and background-task rows; `eval/run_prefetch_eval.py` added with a first
  baseline. The eval then drove a second round: three more collections,
  per-leg hybrid queries, the single-term ladder floor fix, a 150 ms soft
  wait with lex carry-over, and reserved carried slots — merged doc_hit
  0.25 → 0.55–0.60
- **2026-09-08:** `ANTICOMPLIANCE_DIRECTIVE` deleted from `prompt_builder.py`
  (#465 — SOUL.md's own copy is the only one now); `log_turn_prompt_budget`
  added to `prefetch.py` so the injected half of the turn is measured against
  `PROMPT_BUDGET_CHARS` beside the system half (#466)
- **2026-09-11:** audited against the code again. Corrected: the fact tree
  moved out of the vault to `_pipeline/vault-derived/facts/`; the secondary
  slot is live and is a 35B on llama.cpp, not a stopped gemma-4B; `dropped=`
  can never name a vault leg; the 25-char floor binds the hybrid leg only;
  the two hybrid legs no longer share a query string; 20 eval queries, not
  21; backlog ids now span 9–890. Added: what becomes of a `subliminal` row
  after it is written (both role filters, and the hard compaction that
  destroys it), the `role="thinking"` sibling, worker turns being prefetched
  exactly like chat turns, and why prefetch keeps qmd's per-collection path
