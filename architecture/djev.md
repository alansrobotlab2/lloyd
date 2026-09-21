---
segment: architecture
tags: [architecture, lloyd, djev, decisions, calibration]
type: reference
status: implemented
date: 2026-09-20
---

# djev — the structured-decision engine on GPU 2

DiffusionGemma 26B-A4B NVFP4 (`github.com/mmastrac/djev-spark`), serving Jev's
`POST :8011/v1/systemone`: typed questions — yes/no (`noul`), one-of-N
(`choice`), ordered scale (`score`) — read off one diffusion canvas. It
replaced the Qwen3.6 secondary on 2026-09-20 and was wired into Lloyd the same
day.

**GPU 1 runs at 100% utilization. GPU 2 sits at 0%** with 23.7 GiB of weights
loaded. Every decision moved here is pure throughput, and a decision costs
~40 ms plus ~0.25 ms per prompt token.

It is **not a chat slot**. It is absent from `models:` and from
`resolve_model_alias`, the model dropdown has never heard of it, and nothing
routes a turn to it. That stays true.

---

## 1. What it is for, and the line

Measured on 2026-09-20 over 40 labelled pairs, four framings of one question:

| framing | AUC | acc@0.5 | best acc | optimal threshold |
|---|---|---|---|---|
| `choice`, "distinct" listed first | **0.833** | 0.80 | 0.82 | 0.39 |
| `choice`, "related" listed first | 0.795 | 0.50 | 0.75 | **0.03** |
| `choice`, mean of both orders | 0.815 | 0.53 | 0.80 | 0.30 |
| `noul` (no option list at all) | 0.723 | 0.50 | 0.55 | **0.01** |

Order bias |P_A − P_B|: mean **0.324**, max **0.851**. Re-measured on 200 pairs
of `cluster_judgments.jsonl` the same day: mean **0.245**, max **0.947**.

Four rules follow, and every one of them is a thing this tree gets wrong if
left to reasoning rather than measurement:

1. **Ranking is trustworthy; the probability attached to it is not.** AUC holds
   across every framing, so the *ordering* is real signal. The number attached
   to it moves by a third from a cosmetic change, and each framing has its own
   optimal cutoff. **A fixed 0.5 threshold is meaningless.** `config.yaml`'s
   `djev:` block said "calibrated probabilities" until this landed; it now says
   "self-consistent scores", because the measurement contradicts the first.
2. **Order-averaging is not the fix.** It lands between the two arms (AUC 0.815)
   and `acc@0.5` stays 0.53. The fix is a threshold measured per schema, with
   the schema — **option order included** — frozen once calibrated.
   `eval/djev/schemas.py` holds a hash over the question JSON with key order
   preserved, and `Schema.check()` raises `SchemaDrift` when a rewording
   outlives the number measured under it. That is a test, not a convention
   (`tests/test_djev_schemas.py`).
3. **`noul` is not automatically safer than `choice`.** The yes/no form has no
   option list to order and measured *worst*. Pick framings by measurement.
4. **So ranking, ordering and shortlisting are safe today, and every yes/no
   gate ships with a calibrated threshold or it does not ship.** Nothing in
   this round gates anything: `Schema.gate_ready` is False on all five schemas
   and each carries a `gate_blocked_reason`.

---

## 2. The three layers

### 2.1 `app/djev.py` — the one client

Stdlib at its core, on `urllib.request`, because the callers are hot paths and
leaves: `backlog_similar` sits on the `backlog_write_task` write path, `vault`
on every recall, and the entity sweep runs outside the backend entirely.
`app/backlog_status.py` and `app/qmd_health.py` are the precedent; httpx
appears only behind the function-local import in `ask()`.

**Three-valued, like `semantic_candidates`.** `None` on any failure at all —
unreachable, non-200, malformed, slot switched off. Never `[]`, never an
exception. A caller must be able to tell "djev had no opinion" from "djev did
not answer", because only the first is evidence.

**Enablement goes through `app/llm_slots.py::is_enabled("agent-djev")`**, not a
second read of `djev.enabled`. Six surfaces disagreeing about the secondary is
the failure that module exists to prevent.

**It normalizes three traps the server's own response shape creates:**

- `noul` returns `{"type": "noul", "noul": p}` and nothing else — no
  `confidence`, no `probabilities`. A caller reading `confidence` off a mixed
  answer set gets `None` for exactly the yes/no questions and, unguarded,
  reads an absent confidence as zero. The client fills both, with
  `confidence = max(p, 1-p)`: P(yes)=0.1 is a *confident no*.
- `score` returns both `score` (the 0-based expected value over the levels)
  and `confidence` (the modal probability). Different numbers answering
  different questions; collapsing them turns "how severe is this" into "how
  sure are you".
- `label_mass` and `argmax_is_label` live in `diagnostics.questions.<id>`, one
  nesting level away from the answers, so the natural read of the response
  misses the only honest confidence signal the server produces.

**`label_mass` is how much of the model's probability landed on legal label
tokens.** The returned `probabilities` are renormalized over the label set
regardless, so they still sum to 1 and still look confident while 92–98% of the
mass sat elsewhere. Every answer carries it.

**Two trust flags, per answer, never a refusal.** `low_trust` when
`label_mass` is under the schema's floor; `uninformative` when every answer in
a ranking carried the same value. The second exists because a floor is
structurally unable to see that shape — the n=4 listwise run returned every
score 0.0 with `label_mass` 0.987, so the mass was legal and the answer was
empty. The client returns the flags *with* the values; the caller decides.

There is deliberately **no `is_same()`** and there must not be one. A helper
that answered a yes/no from a hardcoded cutoff is the exact mistake §1 rules
out.

### 2.2 `app/djev_shadow.py` — the recorder

`shadow(seam=…, state=…, questions=…, actual=…)` returns `None`, always,
having done one `put_nowait`. It has no return value a caller could branch on
by accident.

- **Bounded queue, drops on overflow.** djev is `--max-num-seqs 1`: strictly
  serialized with unbounded queueing in front of it, which is the old
  secondary's trap where post-session jobs queued behind agent turns. A shadow
  call must never sit in front of a production decision. Losing an observation
  costs a row; blocking the recall path costs the recall.
- **The worker does the expensive part.** A seam enqueues ids and text it
  already holds; anything needing a disk read — the dedupe seam's candidate
  heads — arrives as a zero-argument callable the *worker* runs.
  `tests/test_djev_shadow.py` pins which thread it runs on.
- **Two process shapes.** In the aggregator the worker lives for the process
  and `agent_mcp/djev.py::shutdown()` flushes it through the MODULES hook
  `main.lifespan` already runs. In a script the process exits minutes after
  its last call and a daemon thread takes the queue with it, so
  `entity-resolution-sweep.py` calls `flush()` in a `finally`. What a landing
  restart still loses is counted into `dropped_at_shutdown` on the next
  process's first row — the stack restarts several times a night and a silent
  loss reads exactly like a quiet seam.
- **The floor and schema hash come from the registry, not from the seam.**
  Three seams each remembering to pass their own floor is three places for it
  to go stale after one recalibration.
- **What the bound does NOT protect, stated plainly.** It keeps a shadow call
  off the *caller's thread*; it cannot keep one out of the engine's queue.
  djev serves one sequence at a time, so a shadow read in flight can delay a
  `djev_rank` tool call by up to ~1 s. That is acceptable only because nothing
  in this round depends on djev for a decision — every consumer is advisory —
  so the only thing a shadow row can delay is other advice. **A future round
  that makes a djev answer load-bearing has to revisit this**, either by
  giving production calls a lane or by switching the seam off while one is in
  flight.
- **Muted by `LLOYD_DJEV_SHADOW=0`**, which `eval/run_eval.py` sets for itself
  and `scripts/automod/evalpin.py::env_for` sets for every pinned-corpus child
  — one place, because every regression arm, noise run and warm-up passes
  through it. A replay recorded as production traffic would poison the very
  `label_mass` distribution the floors are read off, and the rerank arm calls
  `_vault_recall`, where the lead seam lives.

Writes `~/.local/state/lloyd-djev/shadow.jsonl`. Outside the repo, like the
automod state dir: a landing rewrites the tree while this is being appended to.

### 2.3 `agent_mcp/djev.py` — three tools

| tool | shape |
|---|---|
| `djev_rank` | query + candidates → scores, ordered. Owns the ≤12 default / 16 ceiling and refuses to sort across canvas chunks |
| `djev_decide` | one state + typed questions → answers with probabilities and trust signals |
| `djev_status` | slot enabled, reachable, recent latency per call site, shadow queue depth and drops, per-schema floors and whether each may gate |

**`list_tools()` is offline and must stay that way.** No reachability probe, no
config read that can raise. A module that degrades makes the aggregator answer
`/health` with a 503, and `agent-services/guardian/detect.py::
mcp_degraded_is_fatal` reads that as a rollback trigger — so a djev engine that
is merely *stopped* would revert whatever landed last. Verified with the engine
unreachable: all three tools still advertise, `djev_rank` answers with an error
result, `djev_status` reports `reachable: false`.

There is **no module `enabled` flag**, for the same reason `code_graph` has
none: an `enabled: false` that emptied `list_tools()` breaks the
annotation-staleness test. The kill switch is
`mcp_servers.lloyd-mcp.disabled_tools`.

All three are `READ_ONLY` in `agent_mcp/annotations.py`. A decision is one
stateless read off a seeded canvas: nothing on the machine changes, the engine
keeps no conversation. Read-only also buys `MCPPool._retry_safe` re-sending a
dropped call and the parallel-dispatch batch overlapping it — both right here
and both wrong for a writer.

---

## 3. The seams

### 3.1 Reranking (lead) — `agent_mcp/vault.py`

The hook sits between the daily-log demote sort and the `[:limit]` slice,
**unconditionally**, where `documents` holds qmd's reranked pool in the order
production returns it. `_vault_recall` has exactly two callers — the
`vault_recall` tool and `memory_ops.recall` (`agent_mcp/memory_ops.py:105`) —
and both pass through it. The earlier "315-378 prefetch turns a day" estimate
had no verified caller behind it and there is no prefetch caller; the real
daily volume is read off the first day of rows, not guessed.

**The first draft pointed at dead code.** It went inside `_graph_rerank`, which
is only reached under `if graph_rerank:` while `RECALL_GRAPH_RERANK` is False
— the 2026-09-04 sweep measured "off wins at every alpha". A shadow call there
would have fired zero times in production and read as a quiet seam.
`tests/test_djev_rerank_arm.py` asserts `_graph_rerank`'s source contains no
`djev` at all.

**The arm and the shadow are exclusive.** With `djev_rerank` on, djev *is* the
decision, and a row comparing djev's ordering against djev's ordering is not an
observation.

### 3.2 Backlog dedupe — `agent_mcp/backlog.py::_dedupe`

At the `backlog_write_task` caller that holds `name` and `description`, **not**
at `merge_target` — which is handed rows of
`{id, title, status, score, lexical, shared, created, source}` and no body text
at all, so the one thing a language model needs to answer the question is the
one thing that function has never seen. The candidate heads are read by the
worker.

### 3.3 Entity SAME/DIFFERENT — `scripts/memory/entity_semantic_gate.py`

After the judges have spoken and before the record is written, on the
**uncached** path only: a cached verdict is not a decision being made, and
recording re-runs would weight the calibration toward whatever the sweep
re-walks most.

**This gate is one judge today, not two.** `default_judges()` adds the
secondary only while `resolve_model_alias("secondary")` still returns
`secondary`, and it has not since `secondary_enabled: false`. The "unanimity"
rule is a single primary vote. djev as the restored *second* judge is the
obvious follow-on round, and §4.2 is what that round has to beat.

---

## 4. What the measurements said

Reproduced 2026-09-20 with `agent-services/bin/bench-djev.py` and
`eval/djev/replay.py`. The original harness lived in a session scratchpad and
no copy survived it, which is why both now exist in the tree.

### 4.1 Performance — there is no problem to fix

| case | this 3090 | upstream GB10 |
|---|---|---|
| 3-question ticket, `samples=1` | **42.9 ms** p50 | 104.3 ms |
| state ~11.6k tok, cold | **2,686 ms** | 5,420 ms (at 8.7k) |
| state ~11.6k tok, warm | **80.7 ms** | 140 ms (at 8.7k) |
| structured-server overhead | **0.8 ms** | — |

Batching: 35.8 / 39.1 / 46.5 / 46.3 / 65.4 ms at 1 / 3 / 6 / 12 / 24 questions
— least-squares fit **35.5 ms fixed + 1.2 ms per extra decision**. That is the
economic fact that makes a sweep batch (8 items × 2 questions) and a clause set
(6 × 2) one request each.

Prefill is the whole cost model: 594 / 2,193 / 8,756 / 21,951 tokens cost
131 / 442 / 1,928 / 6,004 ms cold against 44 / 51 / 73 / 122 ms warm —
**0.20–0.27 ms/token**, with throughput falling from 6,778 to 3,732 tok/s as
the quadratic term from the 5 full-attention layers bites (25 of 30 are
sliding-window-1024).

**Leave the idle clocks alone.** GPU 2 sits in P8 at 210 MHz with nothing
calling it, costing ~25 ms on a sporadic decision; the request finishes
*before* the clocks ramp. Locking clocks would burn ~90 W continuously to save
25 ms. Refused, and this is the record of why.

The **one** action: a boot warmup, added to `start-djev.sh`. The ~1 s spike is
Triton JIT of `_fill_logprob_token_ids_kernel` and `_topk_log_softmax_kernel`,
exactly twice in the whole of `agent-djev.log`'s history, both on the first
structured read after boot. Since the first caller after a restart is now
usually a shadow row, the spike would otherwise land in the latency
distribution the seams are judged on.

### 4.2 Accuracy — the reason nothing gates

**Listwise capacity**, and two hard limits invisible without the diagnostics:

| candidates | state tok | latency | chunks | chunk sizes | min `label_mass` |
|---|---|---|---|---|---|
| 4 | 497 | 126 ms | 1 | [4] | 1.000 |
| 8 | 913 | 182 ms | 1 | [8] | 1.000 |
| 16 | 1,766 | 320 ms | 1 | [16] | 0.965 |
| 32 | 3,478 | 697 ms | 1 | [32] | 0.873 |
| 48 | 4,560 | 1,650 ms | **2** | [33, 15] | 0.303 |
| 64 | 5,600 | 2,373 ms | **3** | [33, 30, **1**] | 0.005 |

Above 32 questions the canvas splits into separate shared contexts, and
upstream states plainly that a partitioned listwise score is not comparable
across them. The lone one-item chunk at n=64 held a candidate scored against
nothing. A reranker that fans a 240-row pool across chunks and sorts the union
produces exactly that artefact, silently. And `label_mass` collapses as N
grows while the probabilities stay renormalized, so they keep looking like
confident scores.

**Replay over the recorded corpora** (`eval/djev/replay.py`, 2026-09-20):

| corpus | n | shape | AUC | acc@0.5 | best thr → acc | what the label is |
|---|---|---|---|---|---|---|
| `entities` | 150 | 1/canvas | **0.942** | 0.827 | 0.263 → 0.893 | the gate's own SAME/REVIEW |
| `clusters` | 200 | 8/canvas | 0.699 | 0.660 | 0.217 → 0.685 | a retired 35B pair-judge |
| `dedupe` | 150 | 3/canvas | 0.607 | 0.587 | 0.208 → 0.627 | what the reranker+Jaccard rule did |
| `reverted` | 137 | 8/canvas | n/a | **0.562** | — | **151 human-verified bad merges** |
| `edges` | 200 | 8/canvas | agreement **0.580** vs a 0.310 majority floor | | | a v4 classifier prompt |

Four things to carry out of that table:

- **Only `reverted` is ground truth.** Everything else measures agreement with
  another model, and a 0.94 AUC against a judge is not accuracy.
- **The threshold measured on ordinary pairs does not transfer to the hard
  ones.** On the 137 definition-carrying pairs of the 151 verified-bad merges,
  `acc@0.5` was 0.562 and **50 of 137 scored above 0.9** while `label_mass`
  stayed healthy (p50 0.991) — confidently wrong, not confused. Those 151 are
  a hard-negative set *by construction*, being exactly the cases the old rule
  got wrong, so 0.942 against a judge and 0.562 against the reverted set are
  consistent and the second is the one a gate would meet. This is why the
  entity seam ships as shadow and `gate_ready` is False.
- **A replay of a definition-less pair measures name shape, not djev.** The
  first run of `reverted` scored 0.364 with **zero** of 302 entity definitions
  loaded, because the replay had its own private definition reader that parsed
  none of the 288 overviews the gate's reader parses. It now imports
  `entity_semantic_gate.entity_definition` and skips the 14 pairs that still
  have no definition, reporting the count — the gate's own rule, for the gate's
  own reason.
- **`clusters` fell from AUC 0.833 on 40 pairs to 0.699 on 200.** The larger
  sample is the truer one. Its reliability curve is plainly miscalibrated: the
  0.0–0.1 bin observed 0.338 and the 0.9–1.0 bin observed 0.690.

**`edges` is the largest prize and the one to take next.** 1,183–3,817
decisions a day at two sequential primary calls each today; djev nearly doubles
the majority-class floor at 34 ms a decision, and 35,802 labelled rows already
exist. What is missing is a human-checked subset — 0.580 agreement with the v4
prompt is agreement, not correctness.

### 4.3 The floors, and why one of them is unset

`label_mass` floors are **per schema and per request shape**, and this is
measured rather than argued:

| schema | shape | min | p01 | p05 | p50 | floor |
|---|---|---|---|---|---|---|
| `entity` | 1 pair / canvas | 0.454 | 0.508 | 0.620 | 0.796 | **0.40** |
| `entity` | 8 pairs / canvas | 0.053 | 0.166 | — | 0.995 | — |
| `dedupe` | 3 pairs / canvas | 0.562 | 0.694 | 0.815 | 0.934 | **0.50** |
| `clusters` | 8 pairs / canvas | 0.878 | 0.930 | 0.967 | 0.993 | **0.80** |
| `edges` | 8 / canvas | 0.680 | 0.908 | 0.965 | 0.997 | **0.60** |
| `rerank` | 12–16 / canvas | — | — | — | — | **unset** |

The two `entity` rows are the same schema, the same corpus and the same engine
— **the request shape alone moved the distribution by an order of magnitude**.
A floor lifted from the batch-8 run would never fire on the seam, and a
"reasonable" 0.5 would flag 1% of perfectly healthy batch-1 reads and 40% of
batch-8 ones. Each floor here sits just under everything observed healthy for
its own shape, which is what a flag that never refuses should mean — "outside
the distribution we measured", not "in the bottom 5% of normal traffic". The
entity floor fires on 0% of its own corpus.

**`rerank` stays unset, and that is the design's own rule holding.** Listwise
`label_mass` at n=12–16 has measured 0.446, 0.807, 0.965 and 1.000 across four
runs on different corpora — a fourfold spread inside the window the design
calls safe. No floor read off one of them means anything for the others, so
the shadow rows, which are the seam's own requests on the seam's own corpus,
set this one. `eval/djev/replay.py --floors` prints the distribution as it
accumulates.

---

## 5. Adoption: the arm, not the log

`RECALL_DJEV_RERANK = False` in `agent_mcp/vault.py`, with
`RECALL_DJEV_RERANK_TOP = 12`. `eval/run_eval.py --djev-rerank` passes it, and
`build_run_config` records `matches_production_defaults: false` for such a run
so a djev-reranked baseline never compares as production's by accident.

**An eval arm is code in the handler, not a log.** The adoption question is
"does djev's ordering beat qmd's own reranker on the labelled set", and a
shadow record has no labels. Flipping the constant is the adoption decision and
is a separate commit with the eval result in its message.

**Measured 2026-09-20**, six runs across two pinned corpora. Every arm in a
run shares one frozen qmd index, so the flag is the only difference left:

| arm | MRR | NDCG@10 | latency avg |
|---|---|---|---|
| baseline | 0.511 | 0.608 | 4,591 ms |
| baseline (repeat) | 0.511 | 0.607 | 4,907 ms |
| `--djev-rerank` (top 12) | **0.605** | **0.697** | 4,841 ms |
| `--djev-rerank` (top 12, repeat) | 0.596 | 0.686 | 4,925 ms |
| `--djev-rerank` (top 12, third) | 0.605 | 0.698 | 5,448 ms |
| `--djev-rerank-top 8` | 0.603 | 0.682 | **4,797 ms** |

**+0.085 to +0.094 MRR, three times out of three**, against a baseline that is
deterministic on a pinned index (0.511 both times) — so the spread in the arm
is djev's own and it is small. `doc_hit_rate` stays 1.000 and
`entity_hit_rate` 0.650 in every arm, which is the check that the arm is
reordering the pool rather than changing what is in it.

**Top-8 is the one to reach for if the budget bites.** It buys the same MRR
(0.603) and lands at 4,797 ms against the nightly's 4,800 ms ceiling, where
top-12 runs 41-648 ms over. The plan anticipated exactly this and called it a
result rather than a tuning, which it is.

**Carry the caveat anyway.** The labelled set is 20 queries / 50 doc labels
and `agent_mcp/vault.py:172` puts the noise floor at 0.02 MRR. Reproducing
+0.09 three times against a deterministic baseline is much stronger than one
run of it, but it is still twenty questions, and the absolute latencies above
were taken on a box that was also running the test suite — the comparison is
fair because both arms paid it, the ceiling comparison is not. `latency_ms_avg`
is reported, never gated; quality is what the paired promotion check compares.

The inverse-cloze result the arm exists to re-test properly: djev MRR 0.766 /
recall@1 0.64 over 16 candidates, against 0.498 / 0.29 for lexical Jaccard and
0.146 / 0.00 for random, at 662 ms for 16. **Jaccard is a weak baseline** and
that comparison decides nothing.

---

## 6. Running it

```bash
# reproduce the tables above
.venvs/lloyd/bin/python agent-services/bin/bench-djev.py all
.venvs/lloyd/bin/python agent-services/bin/bench-djev.py headline --idle-probe

# calibrate against a recorded corpus
.venvs/lloyd/bin/python eval/djev/replay.py --corpus clusters --limit 200 --balance --swap-probe
.venvs/lloyd/bin/python eval/djev/replay.py --corpus reverted --limit 151
.venvs/lloyd/bin/python eval/djev/replay.py --floors      # from the shadow log

# the adoption arm
.venvs/lloyd/bin/python eval/run_eval.py --label djev-rerank --djev-rerank
```

`_pipeline/` is gitignored derived data, so a clone or a worktree has an empty
one and every corpus reads as "no rows" — indistinguishable from a corpus that
ran out. `LLOYD_DJEV_CORPUS_ROOT` names the tree that holds it, and a run that
finds nothing prints where it looked.

The bench refuses a busy engine through `vllm_metrics.wait_idle`, the one
definition of idle. djev serves one sequence at a time, so a neighbour does not
merely add noise — it serializes in front of every read and the whole table
shifts.

---

## 7. Rules, short form

- A fixed 0.5 threshold is meaningless. Calibrate per schema; freeze the
  schema — option order included — once calibrated.
- Order-averaging is not the fix, and `noul` is not automatically safer than
  `choice`. Pick framings by measurement.
- Never sort a ranking across canvas chunks.
- Surface `label_mass` everywhere, and set its floor from data **of the same
  request shape**.
- A floor cannot catch a degenerate ranking. The equality check is a second,
  separate flag.
- Fail open at every layer. A dead engine costs the advice, never the work.
- A shadow hook goes where production's decision is made, not where a knob
  would make one.
- An eval arm is code in the handler, not a log.
- djev stays out of `models:` and out of `resolve_model_alias`.
