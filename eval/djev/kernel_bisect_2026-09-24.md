# djev kernel bisect — 2026-09-24 (#1361, blocks #1357 clause 4)

Alan-attended window on GPU 2, worker pool paused, `regression.lock` and the
session's `retrieval.lock` held for every boot and measurement. Each variant
was booted through `agent-djev.conf`'s `environment=` line (stop, wait for GPU
2 to free, write the conf, `reread`, `update`, `start`). The original conf was
restored byte for byte (sha256 `c451f873…`) and djev was booted back onto it and
verified: RUNNING, `Using 'MARLIN' NvFp4 MoE backend`, 148,269 KV tokens,
`:8011/health` ok. Backend, qmd `:8181` and primary `:8096` answered 200 before
every boot. Raw data: `kernel_bisect_2026-09-24.jsonl`; per-arm run records:
`~/lloyd-data/eval/baselines/bisect1361-*.json`.

## Verdict

1. **The item's instrument cannot pass any configuration.**
   `scripts/djev_determinism_probe.py` sends `/v1/completions` with no
   `diffusion_seed_canvas`. For such a request the DiffusionGemma overlay
   initialises the canvas with `torch.randint` (`diffusion_gemma.py`
   `init_canvas`), so the label logprobs it compares move with that RNG, whatever
   the kernels do. No variant printed 0.0000 on it, and none can.
2. **Production reads differently.** `structured_server.one_read` always sends a
   seeded canvas (`build_canvas(template, slots, seed)`, seed 42 by default) and
   reads once (`diffusion_read_only`). Measured in that shape with
   `seeded_canvas_probe.py` (16 canvas positions, 5 requests x 3 reps per regime):
   - on the incumbent, the warm regime is 0.0000–0.3730 nats and the cold regime
     (fresh `cache_salt`) is 0.98–2.17 nats;
   - under `BATCH_INVARIANT=1` both regimes read **0.0000** at both context
     lengths.
3. **`BATCH_INVARIANT=1` is deterministic end to end.** Its unseeded probe
   numbers (cold 2.4802 / warm 4.9329, and 1.6983 / 3.0975) repeated to four
   decimals across two separate boots (65536 and 81920). With deterministic
   kernels, even the "random" canvas replays from the engine's fixed seed. What
   the unseeded probe reports there is pure canvas-RNG spread.
4. **Neither MoE kernel is the cause, nor the flags.** Humming at BI=0 (with or
   without Humming linears) is as noisy as Marlin, and so is autotune+fuse off.
   The fix is the batch-invariant mode as a whole (deterministic
   matmul/softmax/norm reductions, Humming MoE, emulated NvFp4 linears), not a
   backend swap.
5. **Cost of the winner:** recall p50 547.5 ms vs 521.4 ms for the incumbent in
   the same window. That is +26 ms, 2.5 ms under the 550 ms ceiling. Context
   drops from 131072 to 81920 (BI=1 loads 19.02 GiB of weights against 17.93, and
   the KV pool tops out near 83,400 tokens). Recall quality is unchanged within
   noise (table).
6. **Shipped defaults: not flipped.** Flipping `BATCH_INVARIANT` in the script
   alone would boot OOM, because `agent-djev.conf` pins `MAX_MODEL_LEN="131072"`.
   The real change is two keys in that conf's `environment=`
   (`BATCH_INVARIANT="1"`, `MAX_MODEL_LEN="81920"`). That is a production
   context cut, with a 2.5 ms latency margin, which #1357 leaves to Alan. It also
   cannot satisfy clause 1 as written: that clause is graded on the unseeded
   probe, which no configuration passes.

## Per-variant table

Unseeded nats: worst of 2 probe runs x 5 requests per regime (the existing
probe, table definition unchanged). Seeded nats: `seeded_canvas_probe.py`, worst
of 3 x 5. Recall: 2 fresh (non-replayed) arms of `eval/run_eval.py` on one pinned
qmd corpus held across the whole window, 86 queries each. p50 is from the
per-record `latency_ms`. "top-10 differs" counts the queries (of 86) whose
top-10 document order differed between a variant's two arms.

| variant (MOE_BACKEND / BATCH_INVARIANT, ctx) | boot | MoE backend logged | KV tokens | unseeded cold / warm | seeded warm / cold | recall p50 ms (arms) | doc_hit | MRR | NDCG@10 | top-10 / top-1 differs |
|---|---|---|---|---|---|---|---|---|---|---|
| auto / 0, 131072 (incumbent, no restart) | — | MARLIN | 148,269 | 6.8751 / 4.6786 | 0.3730 / 2.1685 | 511.2, 509.4, 521.4, 512.5 | 0.651–0.663 | 0.326–0.350 | 0.361–0.383 | 86 / 12–13 |
| humming / 0, 131072 | refused: KV needs 3.48 GiB, 2.57 free (max 83,200) | HUMMING | — | — | — | — | — | — | — | — |
| humming / 0, 81920 | ok | HUMMING | 82,647 | 11.7241 / 3.2681 | 1.3152 / 2.8502 | 533.8, 531.8 | 0.628, 0.663 | 0.330, 0.339 | 0.363, 0.379 | 86 / 14 |
| humming + `--linear-backend humming` / 0, 131072 | refused (same KV ceiling) | HUMMING | — | — | — | — | — | — | — | — |
| humming + `--linear-backend humming` / 0, 81920 | ok | HUMMING | 82,647 | 7.6118 / 3.4701 | 1.3833 / 2.3929 | 530.4, 531.5 | 0.663, 0.651 | 0.336, 0.330 | 0.370, 0.378 | 86 / 14 |
| auto / 1, 98304 | refused: needs 2.85 GiB, 2.57 free (max 83,488) | HUMMING | — | — | — | — | — | — | — | — |
| **auto / 1, 81920** | ok | HUMMING (+ emulated linears) | 84,614 | 2.4802 / 4.9329 † | **0.0000 / 0.0000** | 546.8, 547.5 | 0.651, 0.663 | 0.326, 0.322 | 0.373, 0.371 | 31 / 1 |
| auto / 1, 65536 | ok | HUMMING | 77,179 | 2.4802 / 4.9329 † | 0.0000 / 0.0000 (1 rep) | 552.9 ‡, 550.4 | 0.651, 0.663 | 0.325, 0.322 | 0.373, 0.371 | 36 / 2 |
| auto + autotune off + `fuse_act_quant` off / 0, 131072 | ok | MARLIN | 146,728 | 9.7482 / 3.7832 | 0.3820 / 1.8333 | 512.7, 511.0 | 0.628, 0.651 | 0.341, 0.333 | 0.378, 0.370 | 86 / 14 |
| emulation / 0 (131072 and 81920) | refused: `fp8e4nv not supported in this architecture` (SM89+) | EMULATION | — | — | — | — | — | — | — | — |
| triton / 0 | refused: `moe_backend='triton' is not supported for NvFP4 MoE` | — | — | — | — | — | — | — | — | — |
| batched_triton, triton_unfused | not booted: same `map_nvfp4_backend` refusal as triton (the NvFp4 map holds b12x, cutlass, flashinfer_*, marlin, humming, emulation only) | | | | | | | | | |
| marlin / 0 | not booted separately: `auto` resolves to MARLIN (logged every boot), so the incumbent rows are this kernel | | | | | | | | | |

† Identical to four decimals on both boots; see Verdict 3.
‡ Arm 0 of the 65536 boot overlapped my first manual seeded-probe run (20
extra requests), so 552.9 ms is inflated. Arm 1 read 550.4 ms. The 81920 arms
had no overlap.

The 2026-09-22 incumbent row (5.0156 / 8.2064 / 510.5) stays first in the
script's table. Today's incumbent re-measure is its own row.

### Recall-level stability, and what remains under BI=1

With BI=1 the two arms still differ in top-10 order on 31 of 86 queries, down
from 86 of 86. Some of that happens before djev ranks: for `backlog-363` the two
arms' candidate lists contain different documents (`skills/backlog-as-handoff`
and `architecture/protocols/handoff.md` appear in only one). So at least part of
what remains comes from upstream of djev, not from its scores. Clause 1's post-landing check
(nightly `eval-noise.json` `metrics_fresh_ranker` stdevs falling toward the
`metrics` arm's 0.0) is the right production measure for that.

## What is left for a person

- **The decision:** switch production to `BATCH_INVARIANT="1"` and
  `MAX_MODEL_LEN="81920"` in `agent-services/supervisor/conf.d/agent-djev.conf`
  (plus the script defaults and `# shipped defaults:` line, so
  `test_the_defaults_line_and_the_script_defaults_cannot_drift_apart` and the
  shipped-row check hold). This buys bit-identical production reads for +26 ms
  recall p50 and 49,152 fewer tokens of djev context. Before cutting, check the
  longest prompt production sends djev (a recall rank is ~32 rows of 160
  characters; `djev_decide` and the shadow seams take arbitrary text).
- **The instrument:** `scripts/djev_determinism_probe.py` should send the
  production shape (seeded canvas, read-only). As it stands, clause 1 of #1361
  cannot be met by any configuration. `eval/djev/seeded_canvas_probe.py` is the
  shape to adopt.
- **Reading `architecture/djev.md` §8.2 against this:** that measurement
  replayed the structured server's own (seeded) body and saw 1–3 nats, which
  matches today's seeded incumbent (cold 0.98–2.17). So it was a real kernel
  effect, and BI=1 removes it. The #1357 probe does not send that body, so its
  rows mix canvas RNG with kernel noise.

## Method notes

- Every boot waited for `nvidia-smi` GPU 2 < 600 MiB before the next started;
  all freed in < 3 s. Refused boots were stopped on the first `BACKOFF`/error
  line, so no crash loop ran.
- The run was three passes: a driver aggregation bug aborted pass 1 before any
  restart; pass 2 was interrupted after the BI=1@65536 boot to add the seeded
  probe; pass 3 ran the rest. Each pass ended with the incumbent restored and
  verified.
- djev was down for ~2–3 min per boot (13 boots, restores included). Recall fell back to
  qmd's cross-encoder in those windows (`app/qmd_health.py` counts them).
