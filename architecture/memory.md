---
segment: architecture
tags: [architecture, lloyd, memory]
type: reference
status: implemented
date: 2026-09-11
---

# Memory: the `improve` loop and the four-verb surface

Backlog #376. Written while implementing it, so the numbers below are measured,
not proposed. Where the item's assumption did not survive contact with the
store, that is recorded rather than smoothed over.

## 1. What the surface looked like, and what it is now

Before: 19 memory-family tools at three abstraction levels — 10 `fact_*`
(`agent_mcp/facts.py`), 5 `vault_*` (`agent_mcp/vault.py`), 4 `memory_*`
(`agent_mcp/session.py`). Two prior consolidation efforts (#174 "17→7", #340
the module split) moved the count **up**.

Now: four verbs in `agent_mcp/memory_ops.py`, and the 19 unchanged underneath.

| verb | delegates to | the one thing it adds |
|---|---|---|
| `remember` | `fact_add` | refuses to re-add a fact the entity already carries verbatim |
| `recall` | `vault_recall` | documents + facts + graph in one call; `grep_code` off by default |
| `forget` | `fact_invalidate` | refuses an unscoped "forget everything about X" |
| `improve` | `fact_improvement.run_improvement` | the feedback loop, dry-run by default |

**Routers, not replacements.** This deliberately does not shrink the tool count.
A caller that wants `fact_neighbors(min_confidence=0.7)` still calls it; hiding
parameters behind a facade would trade the actual complaint (no obvious entry
point) for a worse one (the escape hatch is gone). Each verb exists because it
adds a guard its underlying tool lacks, not because it renames it.

`forget`'s refusal is the interesting one: naming an entity is not a decision
about its entire history, and the same reasoning made `fact_resolve`
stop defaulting to `auto_resolve=true`.

## 2. `improve`: the detector is evidence, never a verdict

`_detect_contradictions_sync` (`agent_mcp/facts.py:104`) fires when two of an
entity's facts share >0.6 token overlap. That is *phrased alike*, not
*disagrees* — measured on `Lloyd`: 32,857 "contradictions" from 5,489 facts,
and the scan is O(n²). So the loop requires an independent reason per action:

| evidence | action | why it is admissible |
|---|---|---|
| confidences differ | expire the lower-confidence side | the graph already says which claim was believed harder |
| one written ≥1 day after the other | expire the older | a later write about the same contradiction supersedes |
| same confidence, same day | **nothing** | no basis; this is the false-positive class |

A second, stricter filter turned out to be load-bearing. The detector's
*triggers* are opposing terms **or** `_token_overlap > 0.6` — the second one
labels two facts that say nearly the same thing a contradiction, and on this
tree that is its dominant mode (the auto-capture pipeline writes the same event
twice with different confidence numbers). Acting on those pairs is not
correction, it is dedupe decided by whoever lost a coin flip, and it is exactly
what ate useful facts. So `REQUIRE_OPPOSING_TERMS = True`: only polarity flips
(working/broken, enabled/disabled, true/false) are actionable. Near-duplicate
pairs are counted in the record as `near_duplicates` and left alone. On a wide
pass this changed 749 candidate actions into 17.

The `Lloyd` measurement above is now itself history, and worth keeping as the
reason for the bound rather than deleting as out of date: that 113-second,
15-million-comparison scan is why `_detect_contradictions_sync` refuses any
entity holding more than `FACT_GODNODE_THRESHOLD` (50, `agent_mcp/retrieval.py:112`)
facts at all. So the entity that produced the 32,857 false positives can no
longer be scanned, and `plan_entity` propagates that as `refused` rather than
as "nothing found" — the two must never read alike. It bites harder than it
sounds: in the 09-09 pass restricted to the eval's own expected entities,
**18 of 30 were refused as god-nodes**, so most of the entities the metric
depends on are ones this loop is structurally unable to examine. The nightly
shape is milder (1–5 refused of 40) because drift selects on recent writes,
not on size.

Two bounds sit under the evidence rules, and they are what the CLI's exit 3
is checking: `MAX_ACTIONS_PER_ENTITY = 5` and `MAX_ACTIONS_PER_RUN = 25`.
Expiring twenty facts because a heuristic shrugged is not an improvement, and
a god-node's contradiction list is long enough to do exactly that. Both
counters are also deliberately un-fakeable in the other direction:
`_active_count` returns **-1** when the store cannot answer and
`_fact_entity_recall` returns **None** when it could not run, so "did not
measure" can never be recorded as "measured zero" — the same rule
`knowledge-health-report.py` runs on.

Signal sources, both real, neither invented:
- **corrections** — entities named in `~/obsidian/memory/corrections.md`, the
  user's own log of things gotten wrong. Consumed before this only by the
  behaviour/prompt loops; nothing routed it into fact quality. An entry is
  credited to an entity only if a token in its heading is a **registered
  entity**, so a heading can never invent an entity to delete facts about.
- **drift** — entities with a fact file written inside N days. Uses fact-file
  mtime, not `facts_idx.created_at`: the 09-03 rebuild rewrote every row's
  date, and directory mtimes don't move when a file is edited in place.

Writers are the existing tools called as functions. Three writers of
`expired_at` would be the drift bug `fact_profile` and the router already had
— `_reindex_files` (`agent_mcp/facts.py:80-92`), whose docstring is the rule:
the markdown is the fact layer, `facts_idx` is what the router, `fact_profile`
and the health report actually read, and a write that skips the reindex leaves
the two disagreeing. Every run writes a JSON record under
`_pipeline/improvement/` with before/after active-fact counts and one reason
string per action.

## 3. Why it does not call `fact_resolve`, even though that is the right verb

`fact_resolve(auto_resolve=true)` selects losers by **fact id**, and fact ids
are per-file counters: `Assistant/state-001` and `Assistant/usage-001` are
different facts with the same id (19 facts in that entity share `fact-001`).
Measured on a copy of the live store, one plan of 2 actions invalidated **25
facts** — `Assistant` went from 29 active facts to 4 — because each id lookup
hit every fact sharing the loser's id across all category files.

So `apply_action` aims `fact_invalidate` at a substring verified to match
**exactly one** stored fact (growing 60 → 120 → full text, dropped if still
ambiguous), and the run records it as an error and stops that entity if a
writer ever expires more than one fact for one action. The `confidence` vs
`superseded` distinction is preserved in the reason string so the fields can be
re-tagged (`invalid_at` rather than `expired_at`) once ids are entity-unique.
Filed as its own item.

## 4. What it did to the metric it is required to move

`fact_entity_recall` from `eval/run_eval.py`, measured on **copies** of the live
fact tree and `kg.sqlite` via `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB`; the live store
was never opened. Production knobs passed explicitly — `run_eval()`'s
*signature* defaults are the pre-#322 configuration (`graph_rerank=False`,
`alpha=0.5`) while only the *argparse* defaults come from `agent_mcp.vault`, so
calling `run_eval(queries)` measures a configuration nothing serves. The trap
is the gap between the signature (`eval/run_eval.py:184-188`) and the argparse
block (`:428-442`, which imports `RECALL_GRAPH_RERANK` / `RECALL_RERANK_ALPHA`
from `agent_mcp.vault` and carries the comment recording the 2026-09-03 review
that found it); `fact_improvement._fact_entity_recall` fell into it on the
first version.

| pass | facts expired | active facts | `fact_entity_recall` |
|---|---|---|---|
| baseline | — | 205,738 | **0.375** |
| default nightly, guarded (drift ≤3d, 40 entities) | 3 | — (dry run) | 0.375 |
| wide backfill, guarded (drift ≤120d, 2,000 entities) | 17 | 205,738 → 205,721 | **0.375** (Δ 0.000) |
| *unguarded wide pass, for contrast* | 649 | 205,738 → 205,087 | 0.375 |
| *before the one-fact-per-action guard* | 33, **25 of them collateral** | `Assistant` 29 → 4 | 0.375 |

An intermediate unguarded pass over the four entities that dominate the
queries' fact pools moved the metric **0.35 → 0.30** — a regression, and the
clearest evidence in this whole change that "delete more" is not "improve".

**Re-measured independently on 2026-09-09 (the promotion round), same method,
fresh copies, and the flat result reproduced.** Baseline `fact_entity_recall`
**0.375** on the live corpus (20 queries, `errors=0`); the guarded nightly
policy over 40 correction/drift entities found **0 admissible actions** (136
detector pairs, 0 of them opposing-terms — every one was the near-duplicate
class the loop refuses to delete); an unguarded pass expired 24 facts
(255,812 → 255,788) → **0.375**; a further pass restricted to the **30
distinct entities** named by the 20 eval queries' 43 `expect_entities` slots
expired 8 more — 18 of those 30 refused outright as god-nodes — including
**5 of `Task #363`'s 34 facts** (34 → 29) → **0.375**. Probe records:
`_pipeline/improvement/20260909-175*-apply.json`.

**The honest result: the loop runs correctly and does not move the metric.**
#376's acceptance says "a change that cannot move that metric is not this
feature". Two findings say the criterion is mis-aimed rather than the code dead:

1. **The metric is pool-composition-bound.** `_rank` (`agent_mcp/vault.py:1023`)
   caps the returned fact list at `FACT_RANK_CAP_SEED=10`
   (`agent_mcp/retrieval.py:116`; graph-expanded neighbours get their own
   `FACT_RANK_CAP_GRAPH=5`) across *all* seed entities — the only per-entity
   guardrail is a query-token filter, and it applies only above
   `FACT_GODNODE_THRESHOLD=50`. So
   `fact_entity_recall` records which entities won 10 slots, and expiring
   individual facts only changes that when it shifts the score ordering.
   Removing the *lowest-scoring* facts of an entity changes nothing; removing
   its best facts ejects the entity from the pool (measured: expiring the
   highest-scoring claim of `Assistant`, `Autonomy System`, `Backlog System`
   and `Data Pipeline` moved the metric **0.35 → 0.30**). Movement is available
   and it is mostly downward.
2. **The headroom is capped.** Of 30 unmet expected-entity slots, **8 name
   entities with no fact directory at all** — no fact-side change can reach
   them; only ingestion can, which #376 explicitly puts out of scope. Re-counted
   2026-09-11: 9 of the 43 slots, 7 distinct entities (`1X Neo`,
   `Backlog Item #363`, `Isaac GR00T N1.7`, `Semantic Entity Resolution`,
   `TGS-RAG Implementation`, `lloyd`, `qmd`). The number moves with ingestion,
   not with anything this loop does, which is the point.

So the nightly consumer is shipped in **plan mode**: it reads the signals,
pairs, explains, logs counts, and writes nothing. Applying is an explicit act.
The apply-mode policy decision, with the numbers above, is filed as its own
item rather than being taken quietly at 3am by a scheduler.

It is running, and the record says so rather than the schedule saying so: the
pass of 2026-09-11 21:01 UTC (task #84's 14:00 local window) read 40 signal
entities, found 133 detector pairs, classed **every one** of them a
near-duplicate, planned nothing and wrote nothing — the same shape as 09-09's
136-of-136. The number underneath it moved anyway: 255,812 active facts on
09-09, **274,167** on 09-11. Ingestion added ~18,000 facts in two days while
this loop expired none, which is the honest proportion to keep in view — the
quality of the corpus is set by what writes into it, and an expiry loop
bounded at 25 actions a run is not the lever that moves it.

Filed out of this round, so the next reader does not re-derive them: **#659**
(the metric is insensitive to the mechanism — what to measure instead), **#660**
(`fact_resolve(auto_resolve=true)` selects by colliding fact ids — the
2 → 25 collateral above, reproduced minimally 09-09), **#661** (nothing checks
that a skill's named script exists, which is how this item's vault half shipped
a day before its code half).

## 5. Files

| path | role |
|---|---|
| `agent_mcp/fact_improvement.py` | the loop: signals → plan → apply → record |
| `agent_mcp/memory_ops.py` | `remember` / `recall` / `forget` / `improve` |
| `scripts/memory/fact-improvement.py` | CLI; exits 2 on failure, 3 on blast-radius overrun |
| `tests/test_memory_improvement.py` | 27 tests, incl. the one-fact-per-action and near-duplicate guards, and the schedule wiring (task armed + script named by the skill exists) |
| `skills/fact-improvement/SKILL.md` + `autonomy/84-fact-improvement.md` | the scheduled consumer — armed `up_next`, daily, window 14:00 local, plan mode only |
