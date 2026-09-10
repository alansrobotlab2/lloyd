---
segment: architecture
tags: [architecture, lloyd, vllm, primary, kv-cache, workers, throughput]
type: reference
status: implemented
date: 2026-09-10
---

# Primary throughput: the 09-09 stall, and what keeps it fixed

On 2026-09-09 Alan's chat with Lloyd dropped to 1–5 tok/s for the first
sentences of its answers, over and over, while the worker pool was busy. His
expectation — three or four concurrent streams should aggregate 400–500 tok/s
— was right for decode. The engine was spending its time somewhere else.

This file is the plan that came out of the two-day investigation (this
session, lloyd-14's FP8 / prefill-cliff work and lloyd-be's load tests), what
was built from it, and what was measured doing so. Memory note
`vllm-admission-stall` is the running record of the investigation itself.

## 1. What it was

**Cold re-prefills of 100–200k-token prompts.** Every agent-loop iteration is
a fresh HTTP request that re-submits the whole conversation. When the engine
still holds that prefix, the iteration pays for what was appended since the
last one — 4–7k tokens on a 200k prompt. When the prefix was evicted in
between, the whole prompt is prefilled again, one 8,192-token chunk per engine
step, and every other request on the engine gets one token per step until it
finishes.

- **Why cold:** most likely eviction under KV pressure. The 09-09 stalls sat
  at 60–100% of a 398,175-token pool with three or four long-lived
  re-admitting turns (autocode rounds, autotriage, deep-research, chats)
  churning the free pool. Short-lived youtube digests (~100k, 3–11
  iterations) never came back to miss, which is why seven of them at once at
  81.5% KV ran clean the night before.
- **Why ~1 s a chunk:** the prefill kernel's page-count cost near BF16's
  ~150-page cliff (1600-token pages). lloyd-14's sweep: 233k prefilled in
  26 s, 238k in 130 s.
- **Evidence:** all 27 stall episodes in the 09-09 engine log show KV
  climbing 50–190k tokens *during* the episode with ~0 prompt tokens counted,
  then 100–360k landing afterwards; per-iteration `stats` show the misses
  directly (`20260910_003817_autocode_308f` iteration 44: 147,117 in, 0
  cached). Fleet baseline over 09-08/09, valid data only: 194 of 1,754
  iterations at ≥100k re-prefilled ≥50k uncached tokens — 20.6M tokens,
  ~34 minutes of prefill, none at iteration 1–2.

**What fixed it:** the FP8 KV cutover (`e35ab2b`, booted 2026-09-10 11:34):
vLLM main with PR #55557, a **692,263-token pool** (×1.74) and 3200-token
pages, which puts the prefill cliff past `max_model_len`. lloyd-be's
reproduction on it read zero stall-shaped windows.

Exonerated along the way, so nobody re-derives them: the 09-08 vLLM tuning
(the stalls predate it), `harness.parallel_tool_calls` (never on), the
worker → Inner Voice migration, `#529` (no callers), the "widest row sets the
decode clock" hypothesis, and vLLM `priority` (it orders the queue and cannot
shorten a chunk).

## 2. What keeps it fixed

Ordered by protection per unit of work. Nothing below touches the engine
except §2.4, and that one only through its launcher.

### 2.0 The fix is pinned and asserted

Precedent: on 2026-09-06 an automod rollback reverted the secondary's
launcher under the same alias and port.

- `agent-services/supervisor/conf.d/agent-llm-primary.conf` carries
  `VLLM_VENV=…/vllm-flash-next-main` and `KV_CACHE_DTYPE=fp8` in its own
  `environment=`. The launcher's fallbacks are the BF16 worker build, so the
  pins cannot live there. `tests/test_flash_next_launcher.py` runs the
  launcher with `DRY_RUN=1` under exactly that environment.
- `agent-services/bin/flash-next-bootfacts.sh` ends in asserts on the last
  boot (sliced at the launcher's `A/B config:` line, the same rule
  `flash-next-run-arm.sh` uses): the engine config must say
  `kv_cache_dtype=fp8` and the pool must be ≥ 600,000 tokens. Exit 1 on a
  regression, 2 while the boot has not logged its pool yet.
  `EXPECT_KV_DTYPE=` / `EXPECT_KV_POOL_MIN=0` waive them for a deliberate arm.
  Verified against the real log: today's 11:29 boot passes, the 06:27 BF16
  boot fails on both counts.
- `app/model_identity.py` reads `vllm:cache_config_info` off the engine's
  /metrics — it carries `cache_dtype` and `kv_cache_size_tokens` as labels —
  against `models.primary.expect_kv_cache_dtype` /
  `expect_kv_pool_tokens_min`, and logs ERROR on a regression at boot. The
  identity check alone would have called a BF16 boot healthy: it is the
  right model, served the old way.

### 2.1 A miss is counted, and announced

The data was always on disk — `cache_read` per iteration on each assistant
row's `stats`, since `5531f21` — and what was missing was a number and a
bell. `app/prefix_miss.py` is both, and all three usage writers call it: the
streaming chat path and the sync endpoint in `app/routers/messages.py`, and
`app/run_recorder.py` for direct background runs (which now also write the
usage row they never had).

- **A prefix miss** is an iteration ≥ 3 whose prompt is ≥ 100k tokens and
  less than half cached. It is logged as `brain1.prefix_miss` on the
  session's event log and as one INFO line (the guardian reads only
  ERROR/CRITICAL, so a slow iteration cannot roll back a promotion).
- **The turn's usage row** carries `reprefill_tokens` and `prefix_misses`
  (new columns, added in place by `usage_store`'s migration), and so does
  the final assistant row's `stats`. The dashboard's Tokens panel shows the
  last 24 h beside the number of turns that were measured at all.
- **The announcement** is `announce()` on the guardian's one fan-out
  (`agent-services/guardian/notify.py`): news, not an incident. It fires when
  a turn's misses pass 100k tokens while another request was running during
  the miss, once per turn, at most every 30 minutes, journal and toast only
  (`harness.prefix_miss.announce_voice: false`). A cold prefill on an
  otherwise idle engine hurts nobody and is not news.

Three rules decide whether the count means anything, each one of the
investigation's traps:

1. **Iterations 1–2 are never counted.** A turn's first iteration
   legitimately re-admits history the engine may not have seen for hours,
   and the old build reported 0 cached for the first two requests of any
   prefix.
2. **A turn that reads zero on every counted iteration is unmeasured, not
   cold.** Before `5531f21` nothing parsed `cached_tokens` and every session
   read 0 everywhere; a regression of that parser would look exactly like
   every iteration missing. A candidate miss therefore waits until the same
   turn shows a non-zero read, and a turn that never does stores NULL — never
   a zero standing in for "could not tell".
3. **Only a miss counts toward `reprefill_tokens`.** The plan's first draft
   summed the uncached tail of *every* iteration ≥ 3. A healthy iteration's
   tail is the tool result it just appended plus up to a page of alignment —
   measured below at 3.8–6.5k — so a 60-iteration round with a perfect cache
   would read ~300k and fire the alert on health. What is summed is the
   uncached part of iterations that missed, which is also what the fleet
   baseline above summed.

`app/engine_pressure.py` supplies the "was anything else running" half: one
task scrapes the primary's /metrics every 5 s into a five-minute ring,
through a stateless parser (`vllm_metrics.gauges_from_text`) so the
dashboard's own rate baseline is not disturbed. The alert asks for the peak
`num_requests_running` inside the missing iteration's own window, minus
itself; with no sample in that window it reads the engine once, after the
fact, when whatever is running is by definition someone else. The dashboard's
KV meter shows the ring's p50 / p90 / max against a 65% line.

### 2.2 Keep prefixes alive

- **The KV gate** (`workers/pool.py`, `architecture/workers.md` §2): a source
  that declares `LONG_LIVED = True` — autocode, autotriage, deep-research —
  is not claimed while the primary's KV is over `workers.kv_gate.max_kv_usage`
  (0.60). Everything else claims as before. The exclusion goes into the same
  SQL `NOT IN` as `max_inflight`, a hold keeps the item's attempt, and no
  reading means open. It judges the **median over the last minute**, not the
  newest sample, for the reason in §3.2. Visible as `pool.kv_gate` on
  `/api/workers/status` and on the dashboard's worker panel.
- **`workers.slots` stays 2.** Drop it to 1 only if §2.1 shows misses.
- **The compaction wall** moved from 0.8/0.6 to 0.72/0.52 of the 210k
  threshold — trigger ≈ 168k → 151k, target ≈ 126k → 109k. The target moved
  with the trigger so the band stays 0.2 wide: every compaction rewrites the
  middle of the prompt and forces a cold re-prefill of what follows, so a
  narrower band would trade resident KV for more of exactly those. A cost
  knob now, not a fix. **The in-turn pass had never read these fractions** —
  `loop._intra_turn_microcompact` ran on `RunOptions` defaults — so lowering
  them would have moved the wall for the first request of a turn and not for
  the sixty after it. `app/mcp_discovery.intra_turn_compaction_kwargs` now
  feeds both `_get_harness_kwargs` and `_worker_run_options`.
- **`harness.parallel_tool_calls` stays off.** See §4.
- **The iteration-boundary hold** was not built. The plan made it
  conditional on §2.1 showing pressure returning, and there is no production
  data yet (§5).

### 2.3 The Inner Voice observer runs at the watched turn's priority

`install_observer` takes the turn's `RunOptions.priority` and every observer
call — goal extraction included — goes out at it, so a chat's second opinion
runs at 0 beside the chat instead of queueing at 1 behind every worker
iteration. The old rule, always 1 "to yield to the agent it is watching",
guarded something equal priority already guarantees: vLLM orders
equal-priority requests by arrival, so an observer call cannot preempt the
in-flight request of the turn that spawned it. Not a throughput lever; an
intent fix.

### 2.4 Shrink the residual cost — the chunk budget

A cold long prompt beside a chat costs the chat one engine step per chunk.
`MAX_NUM_BATCHED_TOKENS` is a launcher knob now (empty = vLLM's 8192), and
`flash-next-run-arm.sh` takes `SKIP_BENCH=1` for arms whose question is
admission rather than decode. **Production runs 4096**, pinned in the
program's `environment=` beside the FP8 pins; the arms and the reasoning are
in §3.3.

## 3. Measurements

All on the production engine (`agent-llm-primary`, vLLM
`0.28.1rc1.dev661+g6ee5bb0a0`, FP8 KV, MTP k=3), with the rest of the stack
down, through `agent-services/bin/bench-admission-stall.py` — the durable
reproducer that replaced the scratchpad drafts every earlier measurement was
made with. A = a ~119k-token context decoding continuously; B = ~200k.
Every request at priority 1.

### 3.1 Layer 1, end to end (`verify`)

B ran fifteen iterations as an agent loop through the harness's own client,
`_merge_usage` and `app.prefix_miss`, with iteration 13 re-admitted with its
first message changed:

| iteration | prompt | cached | uncached | TTFT | prefix_miss |
|---|---|---|---|---|---|
| 1 | 199,821 | 0 | 199,821 | 21.5 s | — (not counted) |
| 2 | 200,293 | 195,200 | 5,093 | 0.93 s | — (not counted) |
| 3–12 | 200.8k–205.4k | 195,200–201,600 | 3,795–6,547 | 0.84–1.13 s | none |
| **13** | **206,077** | **0** | **206,077** | **21.3 s** | **yes** |
| 14 | 206,606 | 201,600 | 5,006 | 0.94 s | — |
| 15 | 207,173 | 201,600 | 5,573 | 0.99 s | — |

- **Field mapping on the new build:** `cache_read` is populated, and reuse
  engages on the *second* request of a prefix. The old build's rule —
  nothing cached until pass 3 (`bench-prefix-reuse.py`) — does not hold on
  this one, so a cold re-admission is one miss, not two.
- **Exactly one `prefix_miss` and one announcement**, delivered to the
  journal and the desktop toast (`channels: journal, desktop`; voice off by
  config), naming one neighbour.
- A's step gaps: alone p50 17.8 ms; beside the warm loop p50 20.7 ms, p99
  380 ms, max 459; **during the cold re-prefill p50 622 ms, p90 663, max 673,
  30 of 55 steps over 500 ms** — lloyd-be's 615 / 667 exactly. Zero
  preemptions.

**Re-run after the 4096 budget was adopted** (§3.3), on the booted
production config: the same result where it matters — 0 misses across the
warm iterations (reuse 195,200–201,600), exactly one `prefix_miss` at
iteration 13 (206,065 tokens), the announcement decision taken (suppressed
for the re-run: the channel was already proven). What moved is the
neighbour: during the cold re-prefill A's steps were p50 349 ms, p90 366,
with **1 of 84 over 500 ms** (a single 702 ms step) against 30 of 55 at
8192; beside the warm loop p99 316 / max 379 ms against 380 / 459. KV peaked
at 0.53 against 0.96. The price in this shape — a re-admission carrying its
own history — was 23.2 s of prefill against 21.3 s, **+9%**, a little more
than the +7% the cold shape measured.

### 3.2 The KV gauge lies during a cold prefill

`vllm:kv_cache_usage_perc` every 2 s through that run:

| moment | reading |
|---|---|
| A (119k) decoding alone | 0.203 |
| A + B decoding, warm loop | 0.50–0.52 |
| during A's own cold 119k prefill | climbs to 0.43, then 0.203 once it lands |
| during B's cold 200k prefill beside A | climbs 0.20 → **0.956** over 21 s, then **0.498** the moment the prompt is in |

At the production 8192-token chunk budget, a prompt being prefilled
references ~2.5× its resident footprint and releases the excess the moment it
lands. **The overshoot belongs to the chunk budget, not to the prompt:** at
4096-token chunks the same cold 200k prompt beside the same A climbs only to
0.51 — A's 0.20 plus its own 0.30 — and A's own cold 119k warm-up peaks at
0.21 instead of 0.41 (§3.3). The mechanism is not pinned; the per-block
prefix-cache checkpoints of the linear-attention state
(`mamba_cache_mode=align`, 3200-token blocks) are the likeliest suspect, since
an 8192-token chunk never ends on a block boundary. What matters is where the
excess comes from: the free pool, which is exactly where paused turns'
cached prefixes wait for their next iteration. At 8192 every cold long
prefill transiently takes ~70% of the pool and can evict the neighbours whose
prefixes it displaces — a miss that manufactures the next miss. Two
consequences for this layer:

- **A gate on the last sample would engage on every cold prefill** and hold
  every long-lived job for its ~20 s. So the gate reads the one-minute
  median, which sees the residents; `tests/test_kv_gate.py` pins that a
  spike to 0.96 amid a minute at 0.30 does not engage it and a sustained 0.66
  does.
- **The plan's "KV p90 < 60% on a normal day" acceptance is stricter than it
  looks**: p90 is mostly prefills now. The dashboard shows p50 beside it.
  Two cold 200k prefills at once would ask for ~1.5× the pool, which is the
  case the gate exists to make rare.

### 3.3 Layer 3 — the chunk budget

Cold shape: A decoding, B one nonce-prefixed ~200k prompt admitted beside it,
then the same B again with A stopped. The plan's bar: take a smaller budget if
A's p50 during B's prefill drops under 300 ms for at most 10% more prefill
time.

| max_num_batched_tokens | A gap p50 / p90 / max during B's prefill | steps > 500 ms | B prefill beside A | B prefill alone | KV peak |
|---|---|---|---|---|---|
| 8192 (vLLM default) | 608 / 651 / 660 ms | 29 of 54 | 20.19 s | 19.91 s | 0.90 |
| **4096 (production since 2026-09-10)** | **333 / 351 / 354 ms** | **0 of 85** | **21.64 s (+7.2%)** | **21.28 s (+6.9%)** | **0.51** |
| 2048 | 195 / 219 / 224 ms | 0 of 146 | 25.68 s (+27.2%) | 25.32 s (+27.2%) | 0.51 |

A decoding alone stepped at 17.8–17.9 ms p50 in all three: the budget costs
decode nothing. Every boot kept the FP8 pool (692,263 tokens) and passed the
§2.0 asserts.

**4096 is adopted, and it misses the plan's line.** Neither arm cleared both
halves of the bar: 4096 is inside the cost budget and 33 ms over the latency
line, 2048 is under the line at nearly three times the cost. 4096 is taken for
two reasons the bar did not weigh. It is the only arm that stays inside the
cost budget, and halving the step removes every step over half a second. And
it removes the KV overshoot of §3.2: at 8192 a cold 200k prefill climbs to
0.90 of the pool beside one decoding stream, at 4096 and 2048 alike to 0.51,
which is its own footprint. That acts on the stall's *mechanism* — eviction
of paused turns' prefixes — where the bar only priced the symptom. 2048 buys
another 140 ms of neighbour latency for +27% prefill time and no further KV
benefit. The budget lives in `agent-llm-primary.conf`'s `environment=`
(`MAX_NUM_BATCHED_TOKENS="4096"`); dropping it goes back to 8192.

A budget below 8192 makes vLLM log `max_num_scheduled_tokens is set to N
based on the speculative decoding settings. This may lead to suboptimal
performance` — once from the API server and once from the engine core, on
the 4096 and 2048 boots and never on the 8192 one. It is about draft-token
slots: with MTP k=3 every running request takes four of the step's tokens,
32 at `max_num_seqs=8`, which a 2048-token budget absorbs. It cost nothing
measurable: A's step time decoding alone was 17.8 ms p50 at both 8192 and
4096.

## 4. Parallel tool calls: not flipped, and the plan's reason was wrong

The plan listed `harness.parallel_tool_calls.enabled: true` "after the soak
checklist", on the grounds that fewer iterations mean fewer re-admissions.
Two things stand in the way:

- **It does not change the number of iterations.** The flag changes how an
  iteration's tool calls are *dispatched*, not how many the model emits; the
  prompt is identical either way. The model already batches — asked for the
  first line of three files, it emitted three `Read` calls in one iteration
  on three of three tries. What parallel dispatch saves is the tool wall time
  inside an iteration, which for `Read`/`Grep` is milliseconds and matters
  for slow read-only tools (`http_fetch`, `vault_search`). Its effect on
  prefix survival is correspondingly small.
- **The soak cannot run now.** The checklist (`mcp_pool:` warnings in
  `logs/server.err`, `[iv.observer] inject` placement in transcripts,
  `harness.empty_terminal_iteration` counts) is about the aggregator's
  concurrency under real traffic, and the stack is down under the vault-wipe
  hold.

Flip it when the stack is back, watch those three for a day, keep it if they
stay flat:

```bash
grep -c 'mcp_pool:' logs/server.err
grep -h '"harness.empty_terminal_iteration"' event_logs/*.events.jsonl | wc -l
```

## 5. Still open

- **One normal day** with slots = 2 and Alan chatting: success is zero
  iterations ≥ 100k with < 90% reuse after iteration 2, KV p50 well under
  the gate, and no two-request window under 15 tok/s that is not a cold
  admission. It needs the stack up, and the stack is down pending the
  vault-wipe root cause. Hold it for a week and Layers 2–3 are insurance.
- **If misses show up at low KV pressure**, eviction was not the cause and
  the draft-group limitation (`kv_cache_utils.py:1871`, all four Mamba
  groups treated as draft groups) is — upstream, and it would also mean a
  KV-offload tier "stores without ever serving a hit".
- **Capacity past 692k**: 2× RTX PRO 6000 at TP=2 measured a 3.12M-token
  pool upstream (memory `primary-kv-capacity-options`). 1M on one card is
  impossible.

## 6. Found on the way

- **Direct worker turns ignore every `harness.*` key.**
  `workers/sources/_common._worker_run_options` builds `RunOptions` without
  `_get_harness_kwargs()`, whose docstring claims every construction site
  splats it. So `run_prompt_on_primary` jobs run with no stream-stall bound
  (`stream_chunk_timeout_s` 0 against config's 60), no preserved thinking (0
  against 6), and `tool_search` **on** (the dataclass default) where config
  and the override file both say off. Only the compaction fractions are
  passed now; the rest changes worker behaviour and is left for a decision.
- **Tests wrote rows into the live `usage.db`.** `usage_store.DB_PATH`
  resolves from `__file__`, and nothing isolated it. `tests/conftest.py`
  redirects it now, and `_conn` reopens when the path moves, so a worker
  thread's cached connection cannot carry one test's writes into the next.
- **A launcher `DRY_RUN` consumed a staged arm.** The one-shot arm env is
  sourced and deleted before the `DRY_RUN` exit, so a test or a human
  checking the command line during a sweep would have eaten the arm.
  `ARM_ENV` overrides the path.

## 7. Don'ts

- Don't raise KV bytes on BF16, touch the 300 W power cap (the Xid 79
  mitigation), or revisit `--async-scheduling`, `GDN_PREFILL_BACKEND` or the
  MTP arms — none was the cause.
- Don't read `tokens/step` or `prompt_tokens_total` as evidence about
  prefill: on this build both are blind until a request completes. Read KV
  climb and per-iteration `cache_read`. Don't compute a reuse rate from
  `usage.db`'s aggregate columns (`input_tokens` is the peak, `cache_read`
  the sum). Don't select sessions by filename date — worker ids were minted
  in UTC until 2026-09-10.
- Don't use `pool.pause()` as a mitigation — every landing's `round restart`
  clears it — and don't restart the primary with bare supervisorctl except
  through `flash-next-run-arm.sh`, one boot at a time, `MemAvailable` > 150
  GiB between boots.
- Don't start a second Flash-Next engine on GPU 1 while `agent-llm-primary`
  is up: two 95 GiB PLE tables is the 2026-09-08 unit-wide OOM.
