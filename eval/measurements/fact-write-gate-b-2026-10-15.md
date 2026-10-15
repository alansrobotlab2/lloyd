---
type: note
timestamp: '2026-10-15T19:20:00'
item: 2768
---

# Measurement (b): paired LloydMemEval `knowledge_update` gain for the fact write gate

**Result: the gate's retirement reached a scored question's `<facts>` block, and the
paired `knowledge_update` gain it produced is +0.000, 95% CI [-0.113, +0.094] over
n=53.** Bar (b) is **UNMET** as a gain demonstration and **UNMET** as a
disproof of arming: the interval straddles zero on two flips each way, and the
channel the gate writes into is 6 lines wide. The run is recorded here so bar (b)
is no longer "untaken"; the arming recommendation is at the end, and the decision
is Alan's. `knowledge_graph.write_gate.mode` is still `noop` in this diff.

## What bar (b) asked for, and why it had never been taken

The 2026-10-11 ruling left arming the fact write gate on two bars: (a) the
labelled precision/recall measurement, closed by `fact-write-gate-2026-10-08.md`
(5 E / 55 P), and (b) a paired LloydMemEval `knowledge_update` gain, which
`eval/fact_write_gate_snapshot.py` builds the tree pair for and
`eval/run_memory_eval.py --fact-snapshot` scores. #2344 built the instrument,
#2775 and #2778 fixed two things about it, and nobody ran it. Running it as-shipped
produced a null that had nothing to do with the gate, for three reasons found on the
way and fixed in this round:

1. **The fixture is not a corpus.** `WRITE_SEQUENCE` is three facts about one
   synthetic `Relay` entity, which no scored question mentions, so the pair retired
   a prior no `<facts>` block renders and the delta is 0.000 by construction. Fixed
   by `--replay-from` (`eval/fact_write_gate_snapshot.py`), whose list is this
   note's `replay-list.jsonl`.
2. **There was no cross-artifact join.** `run`'s only paired interval is arm-vs-arm
   inside one process (`paired_bootstrap_ci` takes two vectors over shared records),
   while a gate-mode pair is one arm across two processes. Added as
   `run_memory_eval.py join`.
3. **One named bar cost the whole dev set.** 1,275 prompts per leg measured
   2026-09-25, and `--limit` cannot narrow it because it keeps the first N of
   *every* category. Added `run --categories`, and `run --dump-blocks`, which
   commits the lines each leg rendered — `PREFETCH_META_KEYS` kept the block's char
   count and not its content, which is exactly why the previous pair could report a
   0.000 with neither witness nor counter-witness.

## Inputs — the set, the seed tree, and the snapshot root

| | |
|---|---|
| Set | `eval/memory_eval/v3` — `set_sha` `abe162a0de484b04…`, `label_status: pilot`, 264 dev questions, **53** in `knowledge_update`. Named, not defaulted: `DEFAULT_VERSION` is `v1`, which has no label audit and is refused by `require_label_audit` anyway. |
| Seed facts tree (both legs) | `…/.lloyd-data/eval/fact-write-gate/corpus/facts` — 50,846 files in 11,959 dirs, a `cp -a` copy of the frozen `#1480` eval corpus facts tree, made outside the live data root before the builder ran. |
| Snapshot root | `…/.lloyd-data/eval/fact-write-gate/pair-2026-10-15`, legs `noop/` and `on/` (`.lloyd-data/…` is the round worktree's own data root, which `.gitignore` keeps untracked). |
| Replay list | `replay-list.jsonl` beside this note — 262 writes, one per distinct fact the `prefetch_rel` block renders for the 53 questions, plus the witness entity's event-category prior. Each `fact` is a rendered fact's text plus the superseding clause the shipped gate's own threshold was calibrated on (0.55 / 0.75, `agent_mcp/fact_write_gate.py:76-79`), so `UPDATE` is the shipped threshold's answer, not a prompt engineered for this note. |
| Answerer | `Qwen3.8-Flash-Next-nvfp4` at `http://127.0.0.1:8096`, `temperature=0` (a gate-mode pair whose temperature is not pinned has no reproducible flips, and flips are the whole measurement at n=53) |
| Judge | `djev` on GPU 2 (`check_judge` refuses a set judged by its own generator, and it did not refuse) |

Every input and output path above is outside `production_data_root()`. That was not
a formality: the frozen corpus this pair is seeded from is a live store at
`<live data root>/eval/1480/corpus`, and `fact_write_gate_snapshot.py` refuses
`--out`/`--seed-from` there by the same rule as everywhere else — so the seed was
copied out first. The builder seeds each leg by **hardlink** (`seed_cost`: 50,846
linked, 0 copied per leg), which is safe only because the leg then replaces every
file it writes rather than writing through a shared link.

**Where the trees are now.** They were built and scored under the round worktree's
`.lloyd-data/` (the paths in the manifest), and after scoring moved out of every repo
tree to `~/lloyd-eval-scratch/2768/` — `corpus/`, `pair-2026-10-15/`, `artifacts/`.
A seeded corpus cannot sit under a worktree's `.lloyd-data/` even though that path is
gitignored: `tests/test_vault_commit_attribution.py` walks the repo with
`rglob("*.md")`, which does not skip ignored paths and dies with `IsADirectoryError` on
`corpus/facts/mental-models.md` — a DIRECTORY whose name matches `*.md`. Recorded here
rather than fixed: it is a property of holding a corpus inside a repo tree, and the fix
belongs to whoever next has a reason to keep one there.

```
$ .venvs/lloyd/bin/python eval/fact_write_gate_snapshot.py \
    --out "$ROOT/pair-2026-10-15" \
    --seed-from "$ROOT/corpus/facts" \
    --replay-from "$ROOT/replay-list.jsonl"
```

`$ROOT` is `…/.lloyd-data/eval/fact-write-gate` in the round worktree's data root.
The builder's own output, with the differential, is committed as
`build-manifest.jsonl`; the numbers it reported:

| | |
|---|---|
| Writes seen | `noop` 265, `on` 265 — the same 265 through the same `gate_write` in the same order; the pair differs by `LLOYD_FACT_WRITE_GATE` and by nothing else |
| Writes **applied** | `noop` 0, `on` 224 |
| Differential | 224 facts `expired_at`-stamped only in `on`, 0 only in `noop` |

## The two legs

Both legs scored `prefetch_rel`, the arm whose `<facts>` ordering `config.yaml`
ships (`prefetch.facts.rank: relevance`). This is the #2778 rail applied to the pair:
`--arms` defaults to `closed_book,history,prefetch`, and the `prefetch` arm renders
the **confidence** order unconditionally, so a gain measured there is not a gain for
the ordering anyone runs. Both artifacts and both block dumps carry the runner's
disclosure header with `"mismatch": []` — the empty list is the claim that the
scored arm is the shipped ordering.

```
$ LLOYD_FACT_WRITE_GATE=noop .venvs/lloyd/bin/python eval/run_memory_eval.py run \
    --set eval/memory_eval/v3 --arms prefetch_rel --categories knowledge_update \
    --judge djev --concurrency 4 \
    --fact-snapshot "$ROOT/pair-2026-10-15/noop" \
    --out-dir "$OUT" --label fact-write-gate-b-2026-10-15-noop \
    --dump-blocks "$DUMPS/knowledge-update-blocks-noop.jsonl"

$ LLOYD_FACT_WRITE_GATE=on .venvs/lloyd/bin/python eval/run_memory_eval.py run \
    --set eval/memory_eval/v3 --arms prefetch_rel --categories knowledge_update \
    --judge djev --concurrency 4 \
    --fact-snapshot "$ROOT/pair-2026-10-15/on" \
    --out-dir "$OUT" --label fact-write-gate-b-2026-10-15-on \
    --dump-blocks "$DUMPS/knowledge-update-blocks-on.jsonl"
```

The legs are `LLOYD_FACT_WRITE_GATE`-pinned (`run` refuses a pin other than the
snapshot's leg), so the pair is leg-pinned as well as path-pinned. Cost: 210.0 s and
205.3 s — 53 prompts per leg, against the 1,275 a whole-set pair costs.

| arm `prefetch_rel`, `knowledge_update`, n=53 | `noop` | `on` |
|---|---|---|
| `correct` | 35/53 = 0.660 [0.526, 0.773] | 35/53 = 0.660 [0.526, 0.773] |
| `correct_strict` | 30/53 = 0.566 | 28/53 = 0.528 |
| `evidence_in_context` | 33/53 = 0.623 | 35/53 = 0.660 |
| `stale` | 0/53 | 0/53 |

## The paired delta, and how it was computed

Neither number above is the headline: two absolute Wilson intervals that overlap
almost completely cannot support a 2-flip difference. The delta is the paired
bootstrap over the questions both legs scored, from `run_memory_eval.py join`:

```
$ .venvs/lloyd/bin/python eval/run_memory_eval.py join \
    --leg noop="$OUT/lloydmemeval-fact-write-gate-b-2026-10-15-noop.json" \
    --leg on="$OUT/lloydmemeval-fact-write-gate-b-2026-10-15-on.json" \
    --arm prefetch_rel --category knowledge_update \
    --out eval/measurements/fact-write-gate-b-2026-10-15/knowledge-update-join.jsonl
```

| metric | `noop` | `on` | paired delta | 95% CI | n | flips each way |
|---|---|---|---|---|---|---|
| **`correct`** | 0.6604 | 0.6604 | **+0.000** | **[-0.113, +0.094]** | **53** | 4 / 4 |
| `correct_strict` | 0.5660 | 0.5283 | -0.038 | [-0.113, +0.038] | 53 | 1 / 3 |
| `evidence_in_context` | 0.6226 | 0.6604 | +0.038 | [+0.000, +0.094] | 53 | 2 / 0 |
| `stale` | 0.0 | 0.0 | +0.000 | [+0.000, +0.000] | 53 | 0 / 0 |

Bootstrap: `seed=20260921`, `n_resamples=10000` — `eval.stats.SEED` /
`N_RESAMPLES`, the same defaults `run`'s in-process comparisons use, which is what
makes a joined interval comparable to one the runner prints itself. `p = 1.0` for
the primary metric. The interval is wide because n=53 is small, not because the
legs disagreed: they disagreed on 8 of 53 questions and split them evenly.

`knowledge-update-join.jsonl` is that result plus one row per scored question, with
the seed and `n_resamples` in its header row.

## The witness — the legs demonstrably differ where a question reads

The interval above would be meaningless if the two trees rendered the same text,
which is what the unseeded fixture guaranteed. They do not. `run --dump-blocks`
committed the lines each leg rendered for all 53 questions:

- `knowledge-update-blocks-noop.jsonl`
- `knowledge-update-blocks-on.jsonl`

`diff` of the two is non-empty on **all 53** rows: 344 lines differ, across 169
entities. **`lme-kn-013`** is the witness — the only question of the 53 whose block
carries `Hornet.dev` at all:

```
noop  - [Hornet.dev] Hornet.dev claims to offer the most efficient BM25 implementation,
          … single-node indexing of 100 million web documents.      (conf 0.8)
on    - [Hornet.dev] Hornet.dev claims to offer the most efficient BM25 implementation,
          … single-node indexing of 100 million web documents, per the 2026-10-14 sync
          that replaces the earlier reading on record               (conf 1.0)
```

The mechanism, per the committed fact rows (`witness-entity-facts-noop.jsonl`,
`witness-entity-facts-on.jsonl`) and the builder's per-write ledger in
`build-manifest.jsonl`: the prior is
**`Hornet.dev/Hornet.dev-state.md::stat-002`**, `expired_at: null` in the `noop` leg
and `expired_at: "2026-10-15T18:39:…"` in the `on` leg, both legs holding the
identical text and confidence — so the block difference is the gate's retirement and
not a different tree.

One correction to the clause that named this witness: it pointed at
`Hornet.dev-event.md::evnt-001`, and the measured ledger says that write was **not**
an UPDATE — `target_fact: null`, `took: "add"` in **both** legs, whose fact file
still shows `evnt-001` active at conf 0.9. The retirement that reaches the block is
`stat-002`. The tests pin the id the ledger shows.

## Why a real retirement moved the score by zero

`prefetch._search_fact_records` shows `FACT_MAX_ENTITIES = 2` entities
(`app/prefetch.py:108`) × `FACT_MAX_PER_ENTITY = 3` facts (`:109`) — six lines per
turn, ordered by query relevance. The gate retired 224 facts and exactly one of them
sat inside that six-line window for a scored question. The others were retired
outside the only channel the eval reads, so the eval could not have moved: arming
the gate changes text a scored question never renders. That is a property of the
block, not a defect of this run — and a re-run of the same pair without widening the
block will return the same 0.000.

It is also the reason this run does **not** falsify the gate: `stale` is 0/53 in both
legs, so the retirements removed no answer the model needed, and
`evidence_in_context` moved the *positive* way (+0.038, 2 flips to 0). At n=53 and
two flips, both readings are inside the noise the interval already states.

## The bundled re-score: not-triggered

The 2026-10-11 ruling attaches one single re-score to a round taking measurement
(b), triggered by an S/D row on record. There are none:
`eval/measurements/fact-write-gate-2026-10-08/labels.txt` is **5 E / 55 P** over 60
rows — **0 S/D rows** (verified by counting that file's label column). The bundled
re-score is therefore **not-triggered**, and no second opinion was consumed here.
Any later S/D row reopens that question; this run neither creates nor waives one.

## Verdict on bar (b), and the arming recommendation

**Bar (b): UNMET.** The paired `knowledge_update` gain is **+0.000, 95% CI
[-0.113, +0.094]** (n=53, seed 20260921, 10,000 resamples) on the ordering
production ships, with the witness block demonstrably differing. An interval that
straddles zero on two flips each way demonstrates no gain — and it equally fails to
disarm anything, because 0/53 `stale` says no needed answer was retired and
`evidence_in_context` moved positive. Read it as "the channel is too narrow to
carry a signal at n=53", which is a statement about the eval, not about the gate.

**Arming recommendation: do not arm the gate on this evidence.** Nothing here is
evidence that arming helps, and nothing here is evidence that it harms; a bar whose
instrument cannot move is not a bar, so arming must not be justified by this
measurement having been taken. Two things would make bar (b) decidable, and neither
is inside this round: widening the `<facts>` channel the eval reads (2 entities ×
3 lines is 6 lines out of a 224-fact retirement), or a category whose questions
depend on a *retired* fact by construction rather than by chance.

**The arming decision is Alan's, and remains Alan's**: whether
`knowledge_graph.write_gate.mode` moves off `noop` is a separate ship action (#2556
owed entry 2), gated on his ship-time spot-check of the 5 E rows plus 10 P rows
(#2556 owed entry 1), which no round may take or simulate. **This diff leaves
`config.yaml` `knowledge_graph.write_gate.mode: "noop"`.** Whether bar (b) is
re-runnable or closes as tried is owed entry 3's ruling, not this note's.
