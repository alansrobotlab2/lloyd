# Engine output-integrity probe: the idle floor (#1268), 2026-09-24

Engine: `vllm-flash-next-main`, vLLM `0.2.1.dev19+gdff1bde84`, Qwen3.8-Flash-Next
NVFP4, FP8 KV (844,969 tokens, 3200-token blocks, `mamba_cache_mode=align`),
MTP drafter (3 speculative tokens), `MAX_NUM_BATCHED_TOKENS=4096`. Worker pool
paused; every run was taken under `primary.lock`, each after three quiet 1 s
polls of `num_requests_running == 0`. Every record carries `running_before`
per prompt, and it read 0 everywhere.

Instrument: `eval/engine_output_probe.py` over `eval/engine_output/corpus.yaml`
(21 prompts at temperature 0, `logprobs: 1`, 48 max tokens). The prompts are
10 short raw or chat completions, 4 qwen3_xml tool-schema chat prompts
(~385 tokens), 5 questions sharing one ~7k-token system prefix, and 2 long
prompts (~19.7k tokens).

## Question 1: is greedy output reproducible idle? Partly, and prompt length decides

Ten runs: 5 in production's regime (`idle/`, where the prefix cache hits
from run 2 on) and 5 with a fresh `cache_salt` per request (`cold/`, every
prefill cold). That gives 45 pairs per prompt.

| tier | prompts | prompt tokens | result over 45 pairs |
|---|---|---|---|
| exact | 14: all short, chat and tool-schema prompts | 5–389 | bitwise identical: same tokens and same logprobs, every pair |
| jitter | 7: the 5 shared-prefix prompts and 2 long prompts | 6,968–19,760 | logprobs move in every pair; tokens diverge in 5 of 7 |

The jitter prompts in detail:

| prompt | same tokens | first divergence | largest agreed-token logprob delta (nats) |
|---|---|---|---|
| sys_restart | 0/45 | token 2 | 0.386 |
| sys_missing_file | 8/45 | token 1 | 0.576 |
| sys_summary | 6/45 | token 7 | 0.473 |
| long_needle | 20/45 | token 9 | 0.248 |
| long_continue | 28/45 | token 17 | 0.847 |
| sys_rule_lookup | 45/45 | – | 0.148 |
| sys_count | 45/45 | – | 0.130 |

**Cold does not fix it.** The salted runs jitter on exactly the same 7
prompts (worst agreement 0.031 cold, 0.140 warm). So the cause is neither a
prefix-cache hit nor a cached-versus-recomputed state.

**The boundary is length.** I sent one filler continuation at increasing
lengths, cold, 4 repeats each (`idle_length_boundary.json`). It was bitwise
reproducible at 1,807 prompt tokens and not at 2,714 or anything longer (max
logprob delta 0.01–0.23 nats). At 4,168 tokens the tokens themselves split.
The boundary lines up with neither the 3,200-token block size nor the
4,096-token prefill chunk. Where it sits in the kernels is not established
here. A length-dependent split in the linear-attention or MoE path is the
likely place, and finding it is a separate question.

## Question 2: a threshold that does not fire on an unchanged engine

Every false-positive count below is leave-one-out. For each run, the floor is
built from the other runs, and the held-out run is compared against each
remaining run as the reference.

- **A naive floor (max of every pair, on every axis): 66 of 90 comparisons
  fired** on an unchanged engine (18 of 20 warm-only, 20 of 20 cold-only).
  A max over 45 pairs does not bound the next pair's divergence index or
  the median of a different continuation.
- **The tiered floor: 0 of 90.** An `exact` prompt is decided on every axis
  at the recorded values, which are zero. A `jitter` prompt is decided only
  on its largest agreed-token logprob delta, against `jitter_margin` times its
  own floor. The margin is derived, not chosen: the worst held-out ratio was
  1.97, and ×1.25 rounded up to 0.5 gives **2.5**.
- **Out of sample:** three fresh idle runs taken after `floor.json` was
  written (`heldout/`), each compared against all 10 floor runs, gave 30
  comparisons and **0 past floor**. The first cut also called a first-token
  flip on a jitter prompt "past floor". heldout-3 flipped sys_summary's first
  token, which none of the 45 floor pairs had done, and that rule fired on 10
  of 30 comparisons. It was removed, and a test pins its absence.

## What the floor can and cannot see

- **Exact tier (14 prompts, 579 sampled tokens a run):** any flipped token
  and any moved logprob. This is the tier that catches a build change in
  short-prompt decode, the tool-call format or the chat template. The
  comparator tests pin a single flipped token (reported with its index) and
  a single shifted logprob.
- **Jitter tier (7 prompts):** only an agreed-token logprob shift larger than
  2.5 × its idle maximum, which is 0.32–2.12 nats depending on the prompt.
  A confident-gibberish failure (AI21's) drops sampled-token logprobs by
  several nats and diverges at token 0–1, so it would register here only
  through the logprob axis, on whatever prefix still agrees. On long prompts
  this canary is weaker, and that is measured rather than assumed.
- **Batch variance is not in the floor.** Every run was single-stream. A
  probe run beside other traffic (the `preempt` arm's `probe`) will diverge
  on the jitter tier for reasons this floor never saw.

## Preemption arm

`preempt --load-requests 2 --load-prompt-words 3000 --load-max-tokens 256`
(two ~13.7k-token load requests alongside the corpus). The idle guard passed,
peak KV use was 0.157, and **`preemptions_reached: false`**, recorded as such
(`preempt/`). The counter read **2.0 before the run**, where it was 0.0 at
triage. Some other load on this boot, not this probe, drove the preempt path
twice. Reaching preemption deliberately was not attempted: the task called
that engine arm human-only, and a load big enough to force it on an
844,969-token pool (8 × ~100k prompts with forced decode) risks the only
primary.

## Reproduce

    python eval/engine_output_probe.py run --label idle-N --out-dir eval/engine_output/idle
    python eval/engine_output_probe.py run --label cold-N --cache-salt fresh --out-dir eval/engine_output/cold
    python eval/engine_output_probe.py floor eval/engine_output/idle/*.json eval/engine_output/cold/*.json
    python eval/engine_output_probe.py compare --current <new run> --reference eval/engine_output/idle/<idle-1>.json
