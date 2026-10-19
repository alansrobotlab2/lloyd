---
segment: architecture
tags: [architecture, lloyd, memory]
type: reference
status: implemented
date: 2026-10-19
---

# Memory: the `improve` loop and the memory tool surface

Backlog #376. Written while implementing it, so the numbers below are measured,
not proposed. Where the item's assumption did not survive contact with the
store, that is recorded rather than smoothed over.

## 1. What the surface looked like, and what it is now

Before: 19 memory-family tools at three abstraction levels — 10 `fact_*`
(`agent_mcp/facts.py`), 5 `vault_*` (`agent_mcp/vault.py`), 4 `memory_*`
(`agent_mcp/session.py`). Two prior consolidation efforts (#174 "17→7", #340
the module split) moved the count **up**.

Then (2026-09-09 to 2026-09-23; the module was retired in `ed78a3a9`): four verbs
in `agent_mcp/memory_ops.py`, layered over the 19 as routers, each adding one
guard its target lacked.

| verb | delegated to | the one thing it added | where that guard lives now |
|---|---|---|---|
| `remember` | `fact_add` | refused a fact the entity already carried verbatim | `fact_add`'s own write-time duplicate refusal (#499), across every category |
| `recall` | `vault_recall` | `grep_code` off by default | nothing: it was `vault_recall` with one default |
| `forget` | `fact_invalidate` | refused an unscoped "forget everything about X" | `fact_invalidate`, which also defaults `ended` to today |
| `improve` | `fact_improvement.run_improvement` | the feedback loop as a tool | `scripts/memory/fact-improvement.py`, which the nightly task #84 runs |

Now: the verbs are retired (2026-09-23). They made the catalog bigger by four
while their three guards were already in, or could move into, the tools they
wrapped, and no transcript or vault file had ever called them. A router is
worth its schema only while it guards something its target does not. The
same pass retired `fact_check` (the same read as `fact_resolve`) and
`fact_profile`, whose per-category cap and `query` ranking moved into
`fact_get`: uncapped, `fact_get` on a hub entity (`Lloyd`, 5,489 facts) had
put every fact into the context.

Naming an entity is still not a decision about its entire history.
`fact_invalidate` refuses without `fact_substring` or `category`, the reasoning
that also took `auto_resolve` out of `fact_resolve` altogether: it is a report
now, and the write verb is a separate one, `fact_resolve_apply`, which keys each
mark on `(source_file, fact_id)` through `app.fact_ids.fact_identity`.

## 2. `improve`: the detector is evidence, never a verdict

`_detect_contradictions_sync` (`agent_mcp/facts.py:430`) fires when two of an
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
entity holding more than `FACT_GODNODE_THRESHOLD` (50, `agent_mcp/retrieval.py:114`)
facts at all. So the entity that produced the 32,857 false positives can no
longer be scanned, and `plan_entity` propagates that as `refused` rather than
as "nothing found" — the two must never read alike. It used to bite harder than
that: the refusal was the end of the entity, so a 55-fact entity spread over five
category files — `Assistant`, live, largest file 26 — was dropped with zero
actions while not one of its files was large enough to refuse on its own. In the
09-09 pass restricted to the eval's own expected entities, **18 of 30 were refused
as god-nodes**; that is the pre-#1251 shape and it overstates today's, because an
entity is now refused only when *every* one of its category files is over the bound.

What an entity over the bound gets instead is the retry the refusal string itself
names: one scan per category (`plan_entity` → `_rescan_by_category`), the categories
taken from the rows of the refused read rather than from a file listing, so the
facts planned against and the facts scanned cannot disagree. The bound is not
raised — each piece goes through the same check, and a category over 50 is skipped
and named in `categories_skipped`, its size kept alongside in `facts_excluded`.
Cost falls with the coverage: Σn_c² ≤ 50·n comparisons against n², which is what
took 113 seconds on `Lloyd`. Cost is not the figure a scan's reach is read from,
though: a whole-entity pair count is quadratic in a fact count that moves with
every re-extraction, so a hub's coverage is stated as a fraction of its facts.
`QMD`, re-measured 2026-10-18 by counting `^- entity:` rows per category file, is
1,028 facts over 8 files, of which `state` (579) and `event` (413) sit over the
bound and are skipped — **36 of 1,028 facts scanned, 3.5% of the entity**. That
fraction is what the plan and the record now publish, as `facts_excluded` and
`facts_total` on the row and as a `facts scanned=N of M` percentage on the CLI
line, because until #3007 the record named the skipped categories and published
their pair counts while their sizes went unread: the 2026-10-18 pass covered 442
of 1,638 facts across its partial rows — 27.0% — and printed a number that could
only be read as a quiet tree. What the
partition cannot do is pair a fact with one in a *different* category, which is why
it runs only on the refusal path: an entity at or under the bound is still scanned
whole, and an entity that reaches the retry had nothing scanned at all before. The
plan and the record therefore carry `scan_scope` (`entity` / `by_category`),
`categories_skipped`, and the fact denominator `facts_excluded` and `facts_total` —
after #1251, `refused: false` no longer means "scanned whole", and only those three
say which it was. A refused entity is the one shape where `checked` is not
coverage: it is the size of the read that was never scanned, so the plan counts it
as excluded and nothing as scanned.

Two bounds sit under the evidence rules — `MAX_ACTIONS_PER_ENTITY = 5`, applied
by `plan_entity` to one entity's plan, and `MAX_ACTIONS_PER_RUN = 25`, the budget
the write loop spends down — and the second is what the CLI's exit 3 checks: it
fires only when `actions_taken` exceeds the run's `--max-actions`, so an
over-planned entity is invisible to the exit code.
Expiring twenty facts because a heuristic shrugged is not an improvement, and
a god-node's contradiction list is long enough to do exactly that. Both
counters are also deliberately un-fakeable in the other direction:
`_active_count` returns **-1** when the store cannot answer and
`_fact_entity_recall` returns **None** when it could not run, so "did not
measure" can never be recorded as "measured zero" — the same rule
`knowledge-health-report.py` runs on. Since #702 the record keeps the evidence
behind that score as well: `fact_entity_recall_detail` (`_recall_detail`) carries
the query count, how many queries answered and `run_eval`'s per-query hits, and is
null exactly when the score is — two apply runs that expired 24 and 8 facts had
both reported a bare 0.375 with nothing to say whether 20 queries moved by zero or
the eval never ran. The same rule reaches the per-entity rows: a refused entity is
written with `contradictions: null`, its `checked` fact count and the detector's
`skipped_reason`, never the `contradictions: 0` a scanned-and-clean entity gets,
and the CLI's `entities scanned=` line takes the refusals off the count and names
them (`refused=7 (AI Agents, LLM, …)`).

Signal sources, both real, neither invented:
- **corrections** — entities named in the correction logs, read from **three**
  surfaces since #2789 and returned newest-first with the source on every signal
  (`corrections_paths()`): `memory/corrections.md` (`## <date> — <text>`),
  `lloyd/USER.md`'s `## corrections_log` bullets, and the `lloyd/MEMORY.md`
  index's `[feedback]` bullets — the last is the live dated channel, because
  USER.md's two standing bullets carry no date of their own and its file-clock
  fallback (`timestamp: 2026-07-11`) sits outside the window, so USER.md
  contributing 0 signals is the design and not a broken channel. An entry is
  credited to an entity only if a token in it is a **registered entity** name
  (checked against the store, dates and prose stripped first), so an entry can
  never invent an entity to delete facts about.
- **drift** — entities with a fact row *created* inside N days: the newest
  per-row `created_at` in the fact files' front matter, read by regex (~1.4 s over
  11,959 dirs in September; 15,243 dirs on 2026-10-18, and the nightly record
  publishes the denominator as `drift_corpus_total`), not file mtimes (#1461). The nightly rebuild rewrites the
  whole tree, so on 2026-09-25 11,806 of 11,807 dirs had a fresh mtime and the
  pool was the corpus. A full re-extraction still re-stamps every row, so the
  record carries the census — `drift_corpus_total`, `drift_pool_fraction`,
  `drift_status` (`slice` / `sweep` at ≥90% of the corpus / `stale` when the
  pool is empty, with `drift_newest_fact`) — and the CLI prints the pool as a
  fraction of entity dirs plus a `[drift] sweep:` or `[drift] 0 candidates …
  stale since <date>` line beside the counts.

Writers are the existing tools called as functions. Three writers of
`expired_at` would be the drift bug `fact_get` and the router already had
— `_reindex_files` (`agent_mcp/facts.py:159`), whose docstring is the rule:
the markdown is the fact layer, `facts_idx` is what the router, `fact_get`
and the health report actually read, and a write that skips the reindex leaves
the two disagreeing. Every run writes a JSON record under
`_pipeline/improvement/` with before/after active-fact counts and one reason
string per action.

## 3. Why it does not call `fact_resolve`, even though that is the right verb

`fact_resolve` used to take `auto_resolve=true`, and it selected losers by
**fact id**, and fact ids are per-file counters: `Assistant/state-001` and
`Assistant/usage-001` are different facts with the same id (19 facts in that
entity share `fact-001`). Measured on a copy of the live store, one plan of 2
actions invalidated **25 facts** — `Assistant` went from 29 active facts to 4 —
because each id lookup hit every fact sharing the loser's id across all category
files.

That defect is closed at the writers, and the loop moved with it: `apply_action`
no longer calls `fact_invalidate` but the marking route underneath it,
`agent_mcp/facts.py::_apply_fact_marks`, which `fact_resolve_apply` and
`fact_invalidate` also share, so there is one front-matter writer again and it
keys every mark on `(file, id)` (`app.fact_ids.fact_identity`). A `confidence`
action whose loser carries an attribution is aimed by that identity, which is the
only route allowed to attach the `conflicts_with` trace (#1817); a `superseded`
action keeps the substring route — `_aim_substring` grows 60 → 120 → full text and
drops the action if the probe still matches two stored facts — with
`stop_after_first=True`. That flag caps the route at one mark **per file**, not per
action: both `if stop_after_first: break` statements sit inside the per-fact loop,
while the loop above it walks every file in the scope, so an action aimed at a phrase
two category files share leaves one mark in each.

That is the route the guard above the writer is still reachable from, and the only
one. `apply_action`'s scope is
`[FACTS_ROOT / action["loser_source_file"]]` when the loser carries an attribution
and `list(entity_dir.glob("*.md"))` when it does not — the plan writes `""` for an
unattributed loser — so a `superseded` action whose loser came from a view with no
`source_file` scans every fact file of the entity, one mark per file, and returns
`expired_count` above 1 on ordinary data (a phrase two categories happen to share).
The `(file, id)` identity route is the capped one: its scope is the single attributed
file and `stop_after_first=True` stops at that file's first mark, so it cannot mark
two facts and cannot trip the guard, duplicated ids notwithstanding. The
`invalid_at`/`invalid_reason` vs `expired_at`/`expire_reason` split this section used
to promise for later is in place today. What remains of the old protection is that
tripwire: `run_improvement` records an error and halts the entity if a mark ever
lands on more than one fact. It used to halt into the record and one stdout line
only, so the CLI exited 0 and a halted apply was a successful run to the scheduler;
it now returns **4** and prints `LLOYD_SCRIPT_EXIT: fact-improvement.py exit=4` on
both streams (#3006), because the run that tripped the guard has already retired a
fact and the operator has to be told before the next reader trusts the record. The
alarm keys on the halt's VALUE, so the ordinary `run action budget` stop that ends a
capped apply still exits 0, and a budget overrun still returns 3. The live collision
count is in `scripts/memory/measure_fact_id_collisions.py`.

## 4. What it did to the metric it is required to move

`fact_entity_recall` from `eval/run_eval.py`, measured on **copies** of the live
fact tree and `kg.sqlite` via `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB`; the live store
was never opened (the record's `isolated`, `facts_root` and `kg_db` say which
copies a score came from). There *was* a knob trap here, and it is closed at the
source rather than by remembering: `run_eval()`'s signature defaults
(`eval/run_eval.py:777-786`) are now the same `RECALL_*` constants its argparse
defaults use (`:2312`, `:2329`), imported at `:46-48` — #498 (`0a7c239e`,
2026-09-14) deleted the restated literals, one of which, `rerank_alpha=0.5`
against production's 0.3, is what made this section's first version measure a
configuration nothing served. The knobs are still passed explicitly, so the
measurement never depends on the pinning holding.

The one knob this consumer deliberately does not take from production is
`expand_graph=True`, and the honest way to say that is a field, not a verdict: for
this arm `expand_graph_matches_production` is False while
`matches_production_defaults` stays True. The flag stays green because #1000
(`7389a1fc`, 2026-09-22) removed `and not args.no_graph` from that conjunction — a
`store_true` flag compared with its own absence, which could only ever switch the
verdict off — and gave the knob its own field instead, as
`tests/test_eval_scorer.py:774` pins. Production's own default,
`RECALL_EXPAND_GRAPH = False` (`agent_mcp/vault.py:510`), is not a recent change
either: that commit *named* the literal that was already there
(`params.get("expand_graph", False)` → the constant), so nothing flipped, and what
the eval adds is a knob the caller may pass rather than a configuration nothing
serves. Reading the non-parity as a knob bug is what #3005 was filed for; its
mechanism never existed post-#1000 and the claim is retired here.

What genuinely did redefine this metric between one record and the next is what the
record now carries instead: `fact_entity_recall_detail.run_config` is
`eval/run_eval.py::build_run_config`'s own output — the same 15 keys the nightly
baseline stamps, `seed_top_k` and `semantic_seeding` among them — because
`RECALL_SEED_TOP_K` 5→10 (2026-09-11) and semantic seeding #1486 (`dbfde750`,
2026-09-25) both changed what the score means while its shape did not (#3009).

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
Read it as a **blast-radius** number and not a quality one: that pass condemned
2 facts and `fact_resolve(auto_resolve=true)` selected its losers by fact `id`,
which `app/fact_ids.py` numbers *within one category file*, so each id matched
every fact in the entity sharing it and 25 were invalidated where 2 were
condemned (`Assistant` went 29 active → 4). The metric moved because the writer
deleted the surviving twins that were making the expected entities retrievable,
not because anything became more correct. The identity scoping is fixed at the
writers (`agent_mcp/facts.py::_apply_fact_marks`; the live collision count is in
`scripts/memory/measure_fact_id_collisions.py`), and the number that can show a
correction working is the pass's own pair count, not this average — see
`eval/run_eval.py::count_overreach_regressions` for the number that reads an
over-reach as a regression.

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

1. **The metric is pool-composition-bound.** `_rank` (`agent_mcp/vault.py:2637`,
   nested in `_vault_recall_base` at `:2363`, over the pool `_collect` builds at
   `:2599`) caps the returned fact list at `FACT_RANK_CAP_SEED=10`
   (`agent_mcp/retrieval.py:118`; graph-expanded neighbours get their own
   `FACT_RANK_CAP_GRAPH=5`, `:119`, on the leg production does not run while
   `RECALL_EXPAND_GRAPH` is False) across *all* seed entities — the only per-entity
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

Both of those figures are history now, and the shape is the part to keep: the
2026-10-18 nightly read 40 entities and planned 0 actions, `before_active`
**161,255** — down from 09-11 because `general` facts were quarantined out of the
tree (`scripts/memory/quarantine_general_facts.py`) — and its
`fact_entity_recall` and `fact_entity_recall_detail` were both **null**, as they
are in every record back to 2026-10-03, because the shipped invocation does not
pass `--report-eval`. The metric named as the acceptance bar is therefore measured
only when someone remembers to ask for it, which is what #3004 is about; the
fields' existence is not evidence that a score was taken.

Filed out of this round, so the next reader does not re-derive them: **#659**
(the metric is insensitive to the mechanism — what to measure instead), **#660**
(`fact_resolve(auto_resolve=true)` selected by colliding fact ids — the
2 → 25 collateral above, reproduced minimally 09-09; the selection is by
`(file, id)` now, so this reads as history, not as an open defect), **#661** (nothing checks
that a skill's named script exists, which is how this item's vault half shipped
a day before its code half).

## 5. Files

| path | role |
|---|---|
| `agent_mcp/fact_improvement.py` | the loop: signals → plan → apply → record |
| `scripts/memory/fact-improvement.py` | CLI; exits 2 on an unreadable store, 3 when the run's action budget is overrun, **4** when the apply loop halted on the blast-radius guard, with `LLOYD_SCRIPT_EXIT: fact-improvement.py exit=4` on both streams so the halt reaches `meta_json.script_alarm` / `script_alarm_runs` instead of exiting 0 unseen (#3006). The halt is reachable only from the **text route whose `loser_source_file` is empty** — scope `entity_dir.glob("*.md")`, one mark per file, so a phrase two category files share marks one in each; the `(file, id)` identity route is capped at one mark and cannot trip it (§3) |
| `tests/test_memory_improvement.py` | 129 tests (2026-10-19), incl. the one-fact-per-action and near-duplicate guards, the per-category rescan's scope rules (#1251), the exit-code ladder and the blast-radius halt's 4 + sentinel line (#3006), and the schedule wiring (task armed + script named by the skill exists) |
| `skills/fact-improvement/SKILL.md` + `autonomy/84-fact-improvement.md` | the scheduled consumer — armed `up_next`, daily, window 14:00 local, plan mode only |

## Review log

- 2026-10-19 — **stale**. The loop, the god-node bound and the plan-mode consumer
  are as described; the parts that had moved were the writer route (§3 now goes
  through `_apply_fact_marks` keyed on `(file, id)`, with the `invalid_at` split
  already landed and `fact_resolve` no longer taking `auto_resolve` at all), the
  §4 knob-trap paragraph (closed at the source by #498; the live divergence is
  `expand_graph=True` against a production that defaults it False), the
  corrections channel (three surfaces since #2789, `lloyd/MEMORY.md` the live
  dated one), and four pointers/counts (`facts.py:159`, `vault.py:2637`, the
  drift corpus at 15,243 dirs, 124 tests). Filed #3004 (the nightly never scores
  the metric this doc calls the acceptance bar), #3005 (the metric's arm is not
  production's, undistinguishable from a knob bug), #3006 (a blast-radius halt
  exits 0) and #3007 (a partial hub scan publishes pair counts with no fact
  denominator).

- 2026-10-19 — **corrected, same day.** The line above describes what the pass
  believed when it was written and is left standing, but two readings of it were
  wrong and §4 no longer carries them. `matches_production_defaults` does NOT read
  False for the graph-expanded arm: `7389a1fc` removed `and not args.no_graph` from
  that conjunction and added `expand_graph_matches_production`, which is where the
  non-parity is recorded, and the same commit only *named* the literal that
  production had already defaulted False — no flip occurred, so "an arm nothing
  serves" was false too (#3005 closes `stale`: its mechanism never existed
  post-#1000). What does make one record's `fact_entity_recall` incomparable with
  the next is the knobs the record never carried — `RECALL_SEED_TOP_K` 5→10
  (2026-09-11) and semantic seeding #1486 (`dbfde750`) — so the pass now stamps
  `fact_entity_recall_detail.run_config` from `eval/run_eval.py::build_run_config`
  over the same parsed namespace it scores with (#3009).
