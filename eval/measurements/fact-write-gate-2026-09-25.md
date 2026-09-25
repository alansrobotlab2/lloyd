# Fact write gate (#1487): ADD / UPDATE / NOOP on djev, measured 2026-09-25

**Verdict: built, measured, landed OFF** (`knowledge_graph.write_gate.mode: "off"`).
The `noop` mode is precise enough to ship: 76 of 77 held-out NOOPs were right,
and a NOOP destroys nothing already stored. Whether to ship it is Alan's call
(the item's first human clause). The `on` mode, which also expires facts, has
3 held-out UPDATEs (all safe). That is too few to judge a verdict that destroys
a stored fact. Run `shadow` or `noop` to collect them.

Code: `agent_mcp/fact_write_gate.py` (the decision),
`agent_mcp/facts.py::_fact_add`, which covers the MCP tool and
`app/post_capture.py`, and `scripts/memory/next-gen-memory/fact_extractor.py::_gate_paraphrases`
(the nightly writer that bypasses `_fact_add`). `kg_rebuild` carry-over opts
out. Replay harness: `eval/run_fact_write_gate_eval.py`. Raw runs and hand
labels: `~/lloyd-data/eval/1487/`.

## What the writers duplicate

On the live store, of the **10,615** facts written after the 09-23 rebuild
(09-24 and 09-25, almost all nightly extraction), each was compared with the
closest same-entity, same-category fact written before it (token Jaccard):

| best prior Jaccard | share of writes | labelled duplicates (E/S/P) in the band |
|---|---|---|
| < 0.2 | 0.486 | not sampled |
| 0.2–0.3 | 0.116 | 2/16 |
| 0.3–0.5 | 0.102 | 14/36 |
| 0.5–0.7 | 0.120 | 33/36 |
| 0.7–0.9 | 0.123 | 35/36 |
| ≥ 0.9 | 0.053 | 35/36 |

Band-weighted, **~33.5% of recent writes restate a fact the entity already
holds**. Mostly this is the extractor re-reading a changed document and saying
the same claims in new words. #499's verbatim key catches none of it.

Labels, for a (new, existing) pair:
- **E**: the same claim.
- **S**: the new fact is contained in the existing one.
- **P**: the new fact contains the existing one and says more.
- **D**: different claims.

What each verdict should be:
- NOOP is right on E or S.
- UPDATE is safe on E or P.
- A **false supersede** is an UPDATE on S or D, and it destroys a true fact.

## Choosing the question (160 hand-labelled pairs, one pair per canvas)

The pairs were stratified by band, 16/36/36/36/36, labelled by hand, and asked
three framings (`~/lloyd-data/eval/1487/framing_raw.jsonl`):

| signal | AUC, "new adds nothing" (E∪S) | AUC, P |
|---|---|---|
| token Jaccard (no model) | 0.883 | 0.404 |
| three-way `choice`: P(restated) | **0.958** | 0.223 |
| three-way `choice`: P(superseded) | 0.178 | **0.917** |
| coverage (two yes/no choices): P(old covers new) | 0.963 | 0.127 |
| same-claim `choice`: P(same) | 0.920 | 0.185 |

The three-way `choice` (`different` / `restated` / `superseded`, negative first)
won. It is one question that answers both verdicts, at the latency of one read.

- **NOOP at P(restated) ≥ 0.8:** right on 74 of 76 (0.974, Wilson [0.909, 0.993]).
  It catches 82% of the E∪S pairs.
- **UPDATE on P(superseded) alone was unsafe at every cutoff.** Between 18% and
  38% of its supersedes landed on D or S pairs: a changed date, a closed item
  against a demoted one, a narrower qualifier.
- So UPDATE also requires `contains(new, old)`: every word and number of the
  old fact appears in the new one, and the new fact says more. djev's
  P(superseded) ≥ 0.3 is the second vote.
- On the calibration pairs that rule made **0 false supersedes in 9**, and
  caught 8 of 29 P pairs.

## Held-out replay, production shape

`eval/run_fact_write_gate_eval.py replay` read a `.backup` copy of `kg.sqlite`
(`kg-copy.sqlite`) and drew 400 random writes from 09-24 onward, excluding the
160 calibration writes. It rebuilt each write's prior state from `created_at`
and put each write through `decide()` with real djev. I hand-audited every
NOOP and UPDATE.

| shape | NOOP | NOOP right | UPDATE | UPDATE safe | any loss among actions |
|---|---|---|---|---|---|
| up to 6 candidates on one canvas | 81 | 69 (0.852 [0.759, 0.913]) | 3 | 2 | 13/84 |
| **top-1 candidate (shipped)** | **77** | **76 (0.987 [0.930, 0.998])** | **3** | **3 ([0.438, 1.0])** | **1/80 (0.013 [0.002, 0.067])** |

- **Six candidates on one canvas failed.** djev read "restated" onto several
  candidates at once, so a fact adding a version number or a backend list was
  judged a restatement of a vaguer neighbour.
- **One candidate fixed it.** That is also the shape the thresholds were set
  on. So `SHORTLIST_K = 1`.
- **The one K=1 loss:** "PR #75669 fixes announcements being dropped" was
  judged a restatement of "…fall through to gateway calls". The new fact's
  wording of the symptom was lost. No stored fact was touched.

**Near-duplicate reduction:**
- The gate acted on 80 of 400 writes: **20.0% of all writes** [0.164, 0.242].
  That is consistent with the band-weighted estimate from the calibration
  pairs: 21.2% of writes caught, out of ~33.5% that restate, so
  **~63% of near-duplicate writes removed**.
- Misses sit mostly in the 0.3–0.7 band, where djev's P(restated) lands at
  0.5–0.8. Four held-out adds in [0.6, 0.8) were labelled 3 right and 1 wrong.
  Lowering the cutoff to 0.6 buys about 1% of writes at ~75% precision, so the
  cutoff was not lowered.

**The motivating pair is not caught.** "stream_chat races SSE line reads
against cancel_event…" and "…races line reads against a cancel event…" scored
P(restated) = 0.64, three reads identical, below 0.8. At a cutoff that catches
it, precision drops, and precision was weighted first.

**Latency, djev on GPU 2:**
- djev was asked on 149 of 400 writes (37%). The rest have no candidate at
  Jaccard ≥ 0.3.
- Per asked write: p50 **341 ms**, p95 1,075 ms, max 1.4 s, with other agents
  sharing djev.
- Timeout is 3 s, fail-open. Of the held-out writes, 0 failed open.

**End to end on a reflinked copy** (`facts-copy/` + `kg-e2e.sqlite`, via
`LLOYD_FACTS_ROOT` / `LLOYD_KG_DB`), 60 real `_fact_add` calls:
- `_fact_add` p50 went from 65 ms to 372 ms on asked writes, and from 27 ms to
  43 ms on the rest.
- Every NOOP left the file's active count unchanged, every ADD added one, and
  the UPDATE added one while expiring one.
- The offline verdict agreed on 57 of 60. The three that differ saw later facts
  in the copy's file.

## #622: a changed value is never swallowed

All 20 superseded probes of `eval/run_stale_fact_eval.py`'s corpus were put to
the gate, pairing "X listens on port 8080" with "…port 9090". The result was
**20/20 ADD**, with P(restated) at most 0.006. `contains()` refuses an UPDATE
whenever the old value is missing.

So the gate never turns a value change into a NOOP, which would lose the new
value. It also does not fix #622's appended-stale case: both rows stay active,
as today. Expiring on a value change is the Mem0 UPDATE this gate deliberately
does not make from a djev label.

## Owed to a person

- **The ship call** (#1487's human clause 1): `mode: "noop"`, or `"shadow"`
  first to log against real traffic. The measured case for `noop` is precision
  of 0.987 and 20% fewer writes, with nothing stored ever expired.
  - The cost is ~340 ms of djev per asked write, on a MAX_SEQS=1 slot that also
    ranks every vault recall.
  - At night the extractor would ask ~2k times, about 12 minutes of djev. A
    recall made during extraction queues behind it.
  - No downstream recall metric can see this change yet: P0 (#1480) does not
    exist.
- **Whether UPDATE may expire at all** (human clause 2): `on` has 3 held-out
  and 9 calibration supersedes, all safe (0/12, Wilson upper bound 0.243). That
  is not enough to rule out a 1-in-5 failure. `shadow` rows in
  `~/lloyd-data/_pipeline/vault-derived/fact-write-gate.jsonl` would build the
  sample.
- **The post-landing trend** (human clause 3): the next improve-loop
  `near_duplicates` after a flip.
- The flag is read per call (`LLOYD_FACT_WRITE_GATE` env beats config). A
  config change reaches `lloyd-mcp` at its next restart and the nightly
  extractor at its next run.
