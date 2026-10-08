# Fact write gate (#1487): UPDATE verdicts hand-labelled at n = 60, measured 2026-10-08

**What this note is.** Measurement **(a)** of the two the arming bar asks for: a
false-supersede CI over >= 60 hand-labelled UPDATE verdicts (#1487 human clause 2,
settled by #2344's owed ruling, and quoted in `config.yaml`'s `write_gate`
comment). It sits beside `eval/measurements/fact-write-gate-2026-09-25.md` and
supplements it; it replaces nothing. Read the last section before quoting any
number from this one.

## Sample

| | |
|---|---|
| source | `~/lloyd-data/_pipeline/vault-derived/fact-write-gate.jsonl`, read as an extract: `verdict == "update"` rows only. The file holds 25,123 rows and is over 1 MiB, so it is never ordered whole by this measurement |
| frame | the **644** `"verdict": "update"` rows the log held as of 2026-10-08, `ts` spanning 2026-09-26 to 2026-10-08 |
| draw | `random.Random(2441).shuffle(frame)`, take the first 60 — **seed 2441** (the item id), the same draw shape as `eval/run_fact_write_gate_eval.py replay` |
| sample | **n = 60**; the drawn rows' own `ts` range is **2026-09-26 to 2026-10-08**; 49 distinct entities across 4 categories (`event`, `goal`, `relationship`, `state`); target Jaccard 0.400 to 0.909; djev asked on all 60 and failed open on none |
| labeler | **Lloyd**, the #2441 implementation session (round `SM_20261008_220449`) — see "Who labelled" below |
| label file | scored from `~/lloyd-data/eval/2441/labels.txt`; those same bytes are committed at `eval/measurements/fact-write-gate-2026-10-08/labels.txt` (sha256 `743efc8349da8f71…`), and the 60 rows they label at `eval/measurements/fact-write-gate-2026-10-08/decisions.jsonl` (sha256 `c4e83b627cbba064…`) |
| calibration overlap | **0 of the 60** sampled new facts have `app.kg_store.text_hash(fact)` among the **160** `new_hash` values in `~/lloyd-data/eval/1487/pairs.json` — the set the shipped thresholds were tuned on, so as far as that set goes this sample is held out from it |

The 60 rows, their labels, the 160 calibration hashes and the seed are all
committed under `eval/measurements/fact-write-gate-2026-10-08/`, because the
frame is a moving target: the log gains roughly 50 UPDATE rows a day, so the
*draw* is only reproducible against a log of this size, while the *score* has to
be reproducible forever. The four committed files are `decisions.jsonl` (the 60
rows as the scorer reads them), `labels.txt` (one `<row> <E|S|P|D>` line each),
`sample.jsonl` (the seed, the frame size and the window the draw was taken from)
and `calibration-new-hashes.jsonl` (the 160 `new_hash` values, one per line, so
the overlap is re-derivable without `~/lloyd-data`). All four are line-oriented
rather than `.json` because `.gitignore:42` ignores `*.json` repo-wide except for
named negations, and a witness written as `.json` would never have been
committed.

## Where the labels came from, and why no replay ran

The item's clause 3 offered two routes and the first one is not a thing the
tooling does. `cmd_score` opens only `--decisions` and `--labels`
(`eval/run_fact_write_gate_eval.py:143-145`); it never opens a store. The
live-`kg.sqlite` refusal is in `_open_copy` (`eval/run_fact_write_gate_eval.py:60-61`),
which only `replay` reaches. "Score against a `.backup` copy of the store"
therefore describes no code path; the clause's other branch — labels from log
rows, no replay needed — is the one that ran.

Each of the 60 rows was labelled from the gate log's own `fact` (the new fact)
and its `target.fact` / `candidates[0].fact` (the existing fact that an UPDATE
would have expired): the two strings `decide()` was given, which is all the
label needs. So **no replay and no `.backup` store copy were needed**. The
decisions file is those log rows re-keyed to the shape `cmd_score` reads
(`row`, `verdict`, `asked`, `latency_ms`, `target.fact`); nothing in it was
re-decided, and djev was not asked a single question for this note. Reproduce
the numbers from the committed bytes with:

```
python eval/run_fact_write_gate_eval.py score \
    --decisions eval/measurements/fact-write-gate-2026-10-08/decisions.jsonl \
    --labels    eval/measurements/fact-write-gate-2026-10-08/labels.txt
```

## The labels

E/S/P/D as `eval/run_fact_write_gate_eval.py:23-29` and the 09-25 note define
them: **E** the same claim; **S** the new fact is contained in the existing one;
**P** the new fact contains the existing one and says more; **D** different
claims.

**55 P, 5 E, 0 S, 0 D.** The five E rows, 0-indexed into the sample: 16, 37, 47,
50, 57 — each is one claim in two wordings (an exact talk title standing for a
paraphrase, a dropped subject restored, a relation verb fixed). No sampled UPDATE
was found to drop a claim the existing fact made (S), and none was found to
contradict it (D).

## The numbers

```
writes replayed: 60  verdicts: {'update': 60}
djev asked on 60/60; failed-open 0
djev latency ms p50 321 p95 1143 max 1671
NOOP right (E|S):       n=0
UPDATE safe (E|P):      60/60 = 1.000 [0.940, 1.000]
false supersede (S|D):  0/60 = 0.000 [0.000, 0.060]
any-loss among actions: 0/60 = 0.000 [0.000, 0.060]
```

`NOOP right` is `n=0` because the frame is UPDATE rows only: a NOOP verdict would
be a different sample, and the 09-25 note's NOOP row is that sample.

| measurement | this note, 2026-10-08 | the 2026-09-25 baseline |
|---|---|---|
| UPDATE safe (E\|P) | **60/60 = 1.000 [0.940, 1.000]** | 3/3 [0.438, 1.0] — `fact-write-gate-2026-09-25.md:85`, the 3 held-out UPDATEs |
| false supersede (S\|D) | **0/60 = 0.000 [0.000, 0.060]** | 0/12, Wilson upper bound 0.243 — `fact-write-gate-2026-09-25.md:150-152`, 3 held-out + 9 calibration supersedes |

The 09-25 note carries two different baseline figures and they are not the same
measurement: the 3/3 is held-out UPDATE-safe at n=3 — its lower bound of 0.438 is
exactly why that note called the sample "too few to judge a verdict that destroys
a stored fact" — while the 0/12 is the supersedes across held-out *and*
calibration pairs. Both are quoted above as they read there today, and neither is
replaced by the row above it: this note supplements both.

The number the ship decision weighs first is the false-supersede upper bound, and
it is the one this note was filed to tighten: **0.243 at n=12 becomes 0.060 at
n = 60**. That is not a claim of zero risk. An UPDATE that destroys a true fact
one time in seventeen is still inside [0.000, 0.060], and sixty pairs read by one
labeler cannot say more than the interval already says.

## Who labelled

**Lloyd labelled this sample** — specifically the #2441 implementation session,
round `SM_20261008_220449`, reading each (new, existing) pair as text. The clause
forbids djev as labeler because djev is the system under test, and this sample
does not use it: the `answers` block on each row travels in the decisions file for
provenance and played no part in any label.

That does not settle the harder half of the clause, which #2441 leaves owed:
whether a Lloyd-agent reader is the reader the bar demands, or whether the sample
has to be Alan's. If the ruling comes back "Alan's", the rows are committed, the
label format is one line per row, and the label file is the thing to over-label —
the sample does not have to be drawn again.

## Two caveats the shape of the frame imposes

1. **The frame is the gate's own UPDATE decisions, not a sample of writes.**
   Every row in it already passed `contains(new, old)` and took
   P(superseded) >= 0.3, which is the guard doing much of the work the CI then
   grades. That is the right frame for "how often is an UPDATE a false supersede",
   and the frame the 09-25 numbers use, but it is not evidence that these 60 rows
   are representative of writes in general.
2. **Every row is `mode: "noop"` with `applied: "add"`, so nothing was expired.**
   These are *would-be* supersedes: the CI describes what `mode: "on"` would have
   done to those 60 facts and reports no damage already done. It is also why a
   reader at all was needed — the log carries no ground-truth field, so no logged
   row can say whether expiring its fact was safe, which is what #2167 pinned.

## What this does and does not authorize

The bar for `knowledge_graph.write_gate.mode: "on"` is **two measurements plus a
decision**, per #2344's owed ruling and the reservation carried in `config.yaml`'s
`write_gate` comment:

- **(a)** a false-supersede CI over >= 60 hand-labelled UPDATE verdicts — **taken
  by this note**: 0/60 = 0.000 [0.000, 0.060].
- **(b)** a paired LloydMemEval `knowledge_update` gain from
  `eval/fact_write_gate_snapshot.py` and `eval/run_memory_eval.py
  --fact-snapshot` — **not taken**. The instrument exists and is tested
  (`tests/test_fact_write_gate_snapshot.py`); no paired run is on record.

plus Alan's explicit ship decision. This note completes measurement (a) only. It
does not authorize flipping the mode, and no threshold, cutoff, branch or config
value moved in the change that wrote it: `knowledge_graph.write_gate.mode` stays
`"noop"`, and
`tests/test_automod_spec.py::test_a_round_cannot_arm_the_fact_write_gate` is the
fence that keeps a round from moving the value at all.
