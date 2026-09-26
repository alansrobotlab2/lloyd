---
segment: architecture
relations:
  related-to: [infrastructure, engines, workers, harness]
tags: [architecture, lloyd, vllm, primary, kv-cache, gpu, throughput, benchmarks]
type: reference
status: implemented
date: 2026-09-21
---

# The primary engine: vLLM setup, tuning, throughput

`agent-llm-primary` is the only fast model on this box, and everything the
agent does goes through it. This page is what it is configured as, what the
card underneath it can survive, what has been measured, and what keeps it
from degrading. [[infrastructure]] § Model slots covers the slots as a set and
their identity checks; this is the primary's own engine.

The throughput half of this page began as the plan that came out of the
2026-09-09 stall investigation. That investigation is closed — §6 keeps it as
the reason the current configuration is what it is, because every value in §3
and §5 was chosen by it.

## 1. What actually runs

One process, launched by `agent-services/bin/start-qwen38-flash-next.sh` under
supervisord as `agent-llm-primary`. The live command line, verbatim:

```
.venvs/vllm-flash-next-main/bin/python -m vllm.entrypoints.openai.api_server
  --model      agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4
  --served-model-name Qwen3.8-Flash-Next-nvfp4 primary
  --port 8096 --host 127.0.0.1 --trust-remote-code
  --tensor-parallel-size 1
  --max-model-len 262144
  --max-num-seqs 8
  --gpu-memory-utilization 0.9345
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --no-enable-log-requests
  --scheduling-policy priority
  --async-scheduling
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --speculative-config {"method": "mtp", "num_speculative_tokens": 3}
  --kv-cache-memory-bytes 15032385536
  --language-model-only
  --max-num-batched-tokens 4096
  --gdn-prefill-backend flashinfer --no-enable-flashinfer-autotune
  --kv-cache-dtype fp8
  --engram-config {"cpu_offload": true}
```

The last two are the flags this configuration exists for, and both come from
the conf's environment rather than the launcher's defaults: `--kv-cache-dtype
fp8` is emitted from `KV_CACHE_DTYPE` (§3), and `--engram-config
'{"cpu_offload": true}'` is the uva build's spelling of the PLE host-RAM
offload (§8) — on the older worker build the same request goes out as
`VLLM_PLE_CPU_OFFLOAD=1` plus `--distributed-executor-backend mp` instead.

vLLM main at `dff1bde84dd6` (the wheel reports `0.2.1.dev19+gdff1bde84`) — a
**main** build, not a release, because the FP8 QSA path §3 depends on is newer
than any. Until 2026-09-17 that path was PR #55557 overlaid from a
contributor's fork onto `0.28.1rc1.dev661+g6ee5bb0a0`; the PR merged as
`dff1bde84dd6` on 09-16 and the venv was rebuilt at exactly that commit, which
also brings #55309 (the PLE outer residual fused into the short-conv kernel,
1.44x on that kernel at bs=1; the QSA output gate fused into the attention
epilogue, 1.10x single-row). The old build is kept at
`.venvs/vllm-flash-next-main-0910` as the revert target.

Two served names, deliberately: `primary` is the alias every caller in this
repo uses, and `Qwen3.8-Flash-Next-nvfp4` is what `models.primary.expect_model`
substring-matches at boot.

**The load-bearing values live in supervisord's `environment=`, not in the
launcher.** `agent-llm-primary.conf` carries `VLLM_VENV`, `KV_CACHE_DTYPE=fp8`,
`MAX_NUM_BATCHED_TOKENS=4096` and `KV_CACHE_MEMORY_BYTES=15032385536` (§3). The
launcher's own fallbacks are the older
BF16 worker build (`VLLM_VENV` defaults to `.venvs/vllm-qwen38-flash-next`,
`KV_CACHE_DTYPE` to empty), so a conf that loses those lines boots a
perfectly healthy engine with §3 undone. That is exactly the failure §3.1
asserts against, and the precedent is real: on 2026-09-06 an automod rollback
reverted the *secondary's* launcher under the same alias and port.

### 1.1 Slots: identity, the other cards, and who a subagent talks to

(From CLAUDE.md's "Model slots", 2026-09-25; [[infrastructure]] § "Model slots"
carries the slot table and the identity probe in full.)

- **`models.<alias>` is only the endpoint.** What answers there is decided by
  the supervisord program's `environment=MODEL=...` and its start script — three
  places that drift. The 2026-09-06 drift above was `agent-llm-secondary.conf`
  reverted to a launcher branch serving a 4B under the 35B's alias and port.
  `models.<alias>.expect_model` is what catches it, and a slot with no
  `expect_model` reports `unchecked`, so **update it whenever a slot's occupant
  changes** or the check is inert. `expect_kv_cache_dtype` /
  `expect_kv_pool_tokens_min` are the same idea one level down (§3.1).
- **The two venvs.** `start-qwen38-flash-next.sh` adapts to whichever
  `VLLM_VENV` names: `vllm-qwen38-flash-next` (the PLE-offload-worker build, the
  script's own fallback default) and `vllm-flash-next-main` (vLLM main, UVA
  offload — what `agent-llm-primary.conf` serves since 2026-09-10).
- **The secondary (`:8091`, Qwen3.6-35B-A3B UD-Q3_K_XL, llama.cpp,
  `--parallel 1` because llama.cpp divides `--ctx-size` across slots and the
  full 256K window was the point) is off since 2026-09-20** —
  `secondary_enabled: false`, and `app/secondary_models.py` routes its jobs
  (titles, post-session and voice summaries) to the primary. GPU 2 runs djev
  instead: not a chat slot, absent from `models:` and `resolve_model_alias`,
  nothing routes a turn to it. Rank with it, never gate on a fixed cutoff —
  [[djev]].
- **Subagents inherit the calling turn's model.** `subagents.<type>.model: ''`
  means "whatever spawned me"; the harness ships it in the MCP request `_meta`
  (`lloyd/model`, `lloyd/base_url`), since `Task` runs in the aggregator process
  and has no other way to know. Pin an alias there to override. An empty
  `base_url` resolves from `models:` for the chosen model — *not* from
  `default_model_base_url()`, which always returns the primary's endpoint.

## 2. The card underneath it

GPU 1, an RTX PRO 6000 Blackwell Workstation Edition, 96 GB. It is clamped
below rated spec, and that clamp is not a tuning preference — it is the
mitigation for a hardware defect.

*(This section absorbed `architecture/gpu-power-limit-persist.md`, retired to
`.archive/` on 2026-09-11. The power clamp only ever mattered because of what
runs on that card, and splitting the two meant reading both to understand
either.)*

### 2.1 Why the clamp exists

The card repeatedly drops off the PCIe bus under sustained inference — Xid 79
→ Xid 154, unrecoverable without a power cycle. **14 events in the 47 days to
2026-07-21**, across two GSP versions, two engine builds and unrelated models,
and since 2026-07-18 reproducible on demand in **~11 minutes** with a
saturation workload. The card is stable only when clamped below spec. RMA
open; the report is `~/rma/gpu-xid79-falloff-report.md`.

The unit's own `Documentation=` line points at
`file:///home/alansrobotlab/lloyd/gpu-xid79-falloff-report.md`, which has
never existed in this tree. Harmless, but do not go looking for it there.

### 2.2 Why it needs a unit at all

`nvidia-smi -pl <watts>` sets the power **limit** and is runtime-only: it
resets every boot. Without the unit the PRO 6000 comes back at its 600 W
default — the exact envelope that reproduces the fault.

`-pm 1` sets persistence **mode**, which keeps the driver resident so settings
do not drop when the GPU idles. It does not set or remember a power limit, and
it also resets on reboot. The supported mechanism is the `nvidia-persistenced`
daemon, which is what runs here. This driver's `--help` (610.57.04) does not
flag `-pm` as deprecated, so treat "deprecated" as NVIDIA's guidance rather
than something the tool will tell you.

Neither flag survives a reboot on its own.

### 2.3 The unit, and the values in force

`nvidia-power-limit.service`, since 2026-08-22 (`7a12496`). Two files, each
with a repo source of truth and an installed copy that must be refreshed by
hand:

| Role | Repo | Installed |
|---|---|---|
| unit | `agent-services/systemd/nvidia-power-limit.service` | `/etc/systemd/system/` |
| script | `agent-services/bin/set-gpu-power-limit.sh` | `/usr/local/sbin/` |

`ExecStart` runs from `/usr/local/sbin`, **not** the checkout, deliberately: it
executes as root, and a user-writable script running as root is a
privilege-escalation path. The cost is that editing the repo copy does nothing
until you reinstall:

```bash
sudo install -m 0755 agent-services/bin/set-gpu-power-limit.sh /usr/local/sbin/
sudo install -m 0644 agent-services/systemd/nvidia-power-limit.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl restart nvidia-power-limit.service
```

`Type=oneshot`, `RemainAfterExit=yes`, `WantedBy=multi-user.target`, enabled.

| GPU | Card | Clamp | Card default | Accepted range |
|---|---|---|---|---|
| 0 | RTX 3090 (01:00.0) | 275 W | 350 W | 100–375 W |
| 1 | RTX PRO 6000 Blackwell (41:00.0) | **450 W** | 600 W | 150–600 W |
| 2 | RTX 3090 (61:00.0) | 275 W | 350 W | 100–375 W |

`GPU_POWER_LIMIT_W` (300 W) is the fallback for any index without its own
`GPU_POWER_LIMIT_W_<n>`. The numbers have moved three times: **500 W** flat
while the report was being written (it is the figure in the report's own
system table), **300 W** flat from 2026-08-22, per-card **275/400/275** from
2026-09-11 (`bb0dbca`), and **275/450/275** since 2026-09-17 (#1107).

**GPU 1 runs above the only pairing ever described as stable.** That pairing
was 400 W with a **≤2400 MHz** graphics clock, and no clock cap is set
anywhere on this box. SM read 2527–2707 MHz against a 3090 MHz max on
2026-09-11. This driver answers applications-clock queries with "Requested
functionality has been deprecated", so the old `-ac` route is not available
either. Both raises, 300 → 400 W and then 400 → 450 W, were deliberate choices
made with that gap open. **If Xid 79/154 returns on GPU 1, drop this value
before suspecting anything else**, including anything in §3–§5.

**Runtime can disagree with the unit.** Nothing reconciles the two: the unit
is `oneshot` and exited long ago, so a hand-set `nvidia-smi -pl` stands until
the next boot and is then overwritten without any message. That happened twice
before the unit carried 450 W. GPU 1 was raised to 450 W by hand at 08:43 on
2026-09-11, thirteen minutes after the unit applied 400 W. The 2026-09-15 boot
put it back to 400 W and logged a success line, and it was then raised by hand
again. `nvidia-smi` tells you what is in force; the unit tells you what will
be.

The script bounds rather than trusts: a value outside a card's own
`[power.min_limit, power.max_limit]` is clamped into range instead of being
handed to `nvidia-smi` to reject, every card's result is printed to the
journal with the knob it came from, and a failure sets a non-zero exit.
Integer math only (no `bc` on this host), and fields are queried one at a time
because GPU names contain spaces and cannot be split with `read`.

**The index is not a stable hardware id**, and the known defect is precisely a
card falling off the bus. If GPU 1 drops, the remaining cards shift down and a
3090 inherits the override meant for the PRO 6000. The per-card range clamp is
what bounds that: 450 W against a 3090's 375 W maximum lands at 375 W, not at
an error.

### 2.4 The index trap

Three numbering schemes on this box, and only two of them agree:

| `nvidia-smi` | PCI | `/dev/nvidiaN` | Card | Serves |
|---|---|---|---|---|
| 0 | 01:00.0 | `/dev/nvidia2` | RTX 3090 | Qwen3-TTS, the voice worker, qmd |
| 1 | 41:00.0 | `/dev/nvidia1` | RTX PRO 6000 | **this engine** |
| 2 | 61:00.0 | `/dev/nvidia0` | RTX 3090 | the llama.cpp secondary |

`-i <n>` is nvidia-smi enumeration, i.e. PCI bus order — the same order the
engine launchers pin with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the clamp table
and `CUDA_VISIBLE_DEVICES` agree. The **device-node number does not**, and is
very nearly reversed. When attributing GPU usage from `/proc` or an fd list,
map through the PCI bus id and never trust the node number. CUDA's default
order is FASTEST_FIRST, so any process that forgets the pin puts its
"device 0" on the Blackwell. `start-qwen38-flash-next.sh` exports the pin
itself rather than taking it from the conf — see [[infrastructure]] § GPUs.

### 2.5 Persistence mode rides on this unit

All three cards report persistence mode enabled and `nvidia-persistenced` is
running — yet its unit file is `disabled`. It is started by the
`Wants=nvidia-persistenced.service` line in `nvidia-power-limit.service` and
by nothing else: `systemctl list-dependencies --reverse` shows exactly one
reverse dependency. **Disabling the power clamp takes persistence mode with
it.**

`Wants=` does not order. It pulls the daemon in; it does not sequence against
it. There is no `After=nvidia-persistenced.service`, only
`After=multi-user.target`. (This page's predecessor claimed the `Wants=` line
ordered the two — it never did.) That is tolerable because `-pl` does not need
the daemon to have started first, but do not read the line as ordering.

A decoy worth knowing about: `~/.config/systemd/user/nvidia-power-limit.service`
is a symlink to the repo copy, state `linked`, pulled in by no target and
never started. The real unit is the system one, and a user-scope copy could
not do the job anyway — setting a board power limit needs root.

```bash
systemctl status nvidia-power-limit.service      # active (exited) + per-card lines
journalctl -u nvidia-power-limit.service -b      # what it actually set this boot
nvidia-smi --query-gpu=index,name,power.limit,enforced.power.limit --format=csv
```

Read the journal lines, not the unit state: `oneshot` + `RemainAfterExit`
means a unit that set nothing still shows `active (exited)`, so the per-card
output is the only real evidence.

## 3. The KV cache is FP8, and that is the single most important setting

Booted 2026-09-10 (`e35ab2b`). Live `vllm:cache_config_info`:

| label | value |
|---|---|
| `cache_dtype` | `fp8` |
| `kv_cache_size_tokens` | **692,263** (BF16 held 398,175 — ×1.74); **844,969** since 2026-09-15 at `KV_CACHE_MEMORY_BYTES` 14.0 GiB in the program's conf (3.22× at 262k; GPU 1 has nothing else resident and read 5.9 GiB free at the 4096 chunk, so 14.0 leaves ~3.3 GiB) |
| `block_size` / `mamba_block_size` | **3200** (BF16 used 1600) |
| `kv_cache_memory_bytes` | 15,032,385,536 (14.0 GiB, pinned explicitly in the program's `environment=` since 2026-09-15; the launcher's own default is still 11.5 GiB) |
| `kv_cache_max_concurrency` | 2.64 at 11.5 GiB, 3.22 at 14.0 — the pool holds ~3 full-length contexts |
| `enable_prefix_caching` | `True` |
| `mamba_cache_mode` | `align` |

Two separate wins, and the second is the one that fixed the 09-09 stall:

- **Capacity.** ×1.74 more tokens resident, so fewer prefixes get evicted.
- **The prefill cliff moved past `max_model_len`.** The prefill kernel's cost
  near BF16's ~150-page wall is a *page count*, not a token count. At
  1600-token pages that wall sat inside the usable context: 233k prefilled in
  26 s, 238k in **130 s**. At 3200-token pages the same wall lands beyond
  262,144, so it cannot be reached.

The cost, measured: decode is **0–17% slower depending on text type**, because
the MTP drafter accepts fewer tokens through an e4m3 cache. The step time
itself is unchanged. Quality sat at the noise floor.

**The venv guard.** `start-qwen38-flash-next.sh` refuses `KV_CACHE_DTYPE=fp8`
on a venv whose QSA kernel has no fp8 path (it greps for `IS_FP8` in
`qsa.py`), rather than booting BF16 quietly. That refusal is why
`VLLM_VENV=.venvs/vllm-flash-next-main` and `KV_CACHE_DTYPE=fp8` must travel
together.

### 3.1 Pinned and asserted, twice, from two sources

A BF16 boot serves perfectly well while undoing everything above, which makes
it the kind of regression nobody notices. Three independent checks:

- **`agent-services/bin/flash-next-bootfacts.sh`** asserts against the last
  boot log (sliced at the launcher's `A/B config:` line): the engine config
  must say `kv_cache_dtype=fp8` and the pool must be ≥ 600,000 tokens. Exit 1
  on a regression, **2** while the boot has not logged its pool yet.
  `EXPECT_KV_DTYPE=` / `EXPECT_KV_POOL_MIN=0` waive them for a deliberate arm.
  It also fails a boot with no image input (no non-zero `limit_mm_per_prompt`
  image count), since `models.primary.supports_vision` depends on it;
  `EXPECT_IMAGE_INPUT=` waives that for a text-only arm (#1420).
- **`app/model_identity.py`** reads the same two facts off the engine's live
  `/metrics` against `models.primary.expect_kv_cache_dtype` and
  `expect_kv_pool_tokens_min`, and logs ERROR at boot on a miss. The plain
  identity check would have called a BF16 boot healthy: right model, served
  the wrong way.
- **`tests/test_flash_next_launcher.py`** runs the launcher with `DRY_RUN=1`
  under exactly the production environment.

## 4. Context: 262k native, and YaRN is built but switched off

`--max-model-len 262144`, matching the checkpoint's own
`text_config.max_position_embeddings` and `models.primary.context_length`.
The served model's rope block is plain: `rope_type: "default"`.

**YaRN exists as a staged, opt-in arm and is not in use.** A shadow checkpoint
sits at `agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4-yarn2` —
every file symlinked to the original except `config.json`, which carries
`rope_type: "yarn"`, `factor: 2.0`,
`original_max_position_embeddings: 262144`, i.e. a 524,288 horizon. It is
generated by `agent-services/bin/flash-next-yarn-model.py --factor 2.0` and
booted with:

```bash
MODEL_DIR=.../Inferact-Qwen3.8-Flash-Next-NVFP4-yarn2 MAX_MODEL_LEN=524288 \
  bash agent-services/bin/start-qwen38-flash-next.sh
```

Three things about it are worth not rediscovering:

- **It must be a shadow `config.json`, never `--hf-overrides`.** vLLM applies a
  *dict* override to the target model only —
  `SpeculativeConfig.compose_draft_hf_overrides` forwards callables to the MTP
  draft's config and deliberately drops dicts. This model's draft is a full QSA
  layer reading `rope_parameters` from the same `text_config`, so an
  `--hf-overrides` YaRN block leaves the **drafter on plain RoPE with a 262,144
  horizon while the target runs past it** — speculation degrades beyond native
  and nothing in the log says so.
- **YaRN goes in `text_config` only; the outer block stays native** (`83b42f5`).
  The first shadow died before loading a weight with
  `AttributeError: 'Qwen4ExpConfig' object has no attribute
  'max_position_embeddings'`: vLLM's outer config copies `rope_parameters` from
  `text_config` before the transformers constructor runs, and that dataclass's
  `__post_init__` evaluates `self.max_position_embeddings` eagerly for
  `rope_type: "yarn"` — key present or not. The outer config has no such
  attribute. So the shadow passes the native block outward and puts YaRN only
  where the QSA attention, the MTP draft and the `max_model_len` derivation
  read it. The generator refuses a source config whose rope is already scaled,
  rather than stacking.
- **It is static, so it reaches every request — and on text quality that
  costs nothing measurable** (2026-09-21,
  `eval/measurements/yarn-2026-09-21.md`). The model card warns YaRN
  "potentially impact[s] performance on shorter texts"; only 12 of the 48
  layers use RoPE, on a quarter of each head. Measured on the production build
  against production reproducing itself bit-for-bit: mean NLL on 450 short
  windows +0.0014 [−0.004, +0.007]; next-token NLL at 2k/20k/150k/250k inside
  zero; exact-match retrieval 160/160 vs 160/160 (1 code among 4) and 79/80 vs
  78/80 (1 among 24 confusable names) from 16k to 250k, and 72/72 at 300k,
  400k and 480k. What it does do is flip close calls: 19% of short-window
  top-1 tokens, the same rate as changing the request's window length.
- **The one signal against it is tool choice.** Across the 35 web and
  code-nav tool-choice queries, three runs each, production 95/105 and YaRN
  89/105: five queries worse, one better (sign test p ≈ 0.22), with one flip
  persistent under an identical prompt ("is `_prune_reasoning` dead code?" →
  `graph_explain` 3/3 on production, grep 3/3 on YaRN). Because it is static,
  that would reach every worker round, not only long threads. Settling it
  needs a few hundred distinct decisions; until then it stays opt-in.

Capacity and cost, if it is turned on: ~845k tokens at the 14.0 GiB the conf
carries since 2026-09-15 (the YaRN boot *reports* 940,884 from the same memory
and block size — unexplained). On `dff1bde` a 480k prefill peaks at 95.4 GiB
with 2.4 GiB free, so the production budget is safe to full length, and
prefill is 26 s at 300k, 35 s at 400k, 43 s at 480k (61/74/129 s on the 09-10
build — the page-count cliff is gone). A chat decoding beside a cold 400k
prefill stays responsive (max gap 0.32 s) but runs at 5.2 tok/s for 37 s, where
a 200k one costs 7.3 tok/s for 18 s. Turning it on does not by itself raise
`models.primary.context_length`, which is what compaction keys off; raising
that moves every session's truncation from ~210k to ~472k and is a separate
decision. Past 480k needs more card — 2× RTX PRO 6000 at TP=2 measured a
3.12M-token pool upstream; 1M on one card is impossible.

## 5. The tuning knobs

Every knob is an environment variable the launcher reads, so an arm is one
`env` away and production overrides live in the program's `environment=`.

| Knob | Default | Production | Notes |
|---|---|---|---|
| `VLLM_VENV` | `vllm-qwen38-flash-next` | **`vllm-flash-next-main`** | the BF16 worker build vs the FP8 main build |
| `KV_CACHE_DTYPE` | *(empty = BF16)* | **`fp8`** | §3 |
| `MAX_NUM_BATCHED_TOKENS` | *(empty = vLLM's 8192)* | **`4096`** | §5.1 |
| `MAX_MODEL_LEN` | `262144` | default | §4 |
| `MODEL_DIR` | the NVFP4 checkpoint | default | the YaRN shadow is the alternative |
| `MAX_NUM_SEQS` | `8` | default | |
| `GPU_MEMORY_UTILIZATION` | `0.9345` | default | the headroom is vLLM's own prefill scratch, not the desktop's |
| `KV_CACHE_MEMORY_BYTES` | `12348030976` | **`15032385536`** | pinned rather than derived; raised to 14.0 GiB 2026-09-15 |
| `MTP_ENABLED` / `MTP_TOKENS` | `1` / `3` | default | multi-token prediction, k=3 |
| `GDN_PREFILL_BACKEND` | `flashinfer` | default | |
| `FLASHINFER_AUTOTUNE` | `0` | default | |
| `LANGUAGE_MODEL_ONLY` | `1` | default | |
| `MOE_BACKEND` | *(empty)* | default | |
| `EXTRA_ARGS` / `ARM_ENV` | — | — | one-shot arm plumbing; since 2026-09-17 the arm file is sourced above **every** row of this table (it used to sit below the first five, so an arm could not move them and said nothing) |

`--scheduling-policy priority` matters to the rest of the stack: the harness
sends a per-request priority, and vLLM orders *equal* priorities by arrival.
That is why the Inner Voice observer running at its watched turn's priority
(§6.4) is safe — an observer call cannot preempt the turn that spawned it.

### 5.1 The chunk budget is 4096, not vLLM's 8192

A cold long prompt beside a chat costs the chat one engine step per chunk, so
this knob is the one that decides what a *neighbour* feels. Measured in §7.3:
halving the budget cut the neighbour's step from 608 ms to 333 ms for +7%
prefill time, and removed every step over half a second.

It also removes a KV overshoot that acts on the stall's own mechanism — see
§7.2. Dropping the variable returns to 8192.

A budget below 8192 makes vLLM log `max_num_scheduled_tokens is set to N based
on the speculative decoding settings. This may lead to suboptimal performance`,
once from the API server and once from the engine core. It is about draft-token
slots: with MTP k=3 every running request takes four of the step's tokens, 32
at `max_num_seqs=8`, which even a 2048-token budget absorbs. It cost nothing
measurable — decoding alone stepped at 17.8 ms p50 at both 8192 and 4096.

## 6. Throughput: the 09-09 stall, and what keeps it fixed

### 6.1 What it was

On 2026-09-09 chat dropped to 1–5 tok/s for the first sentences of its
answers, repeatedly, while the worker pool was busy. The expectation — three
or four concurrent streams aggregating 400–500 tok/s — was right for decode.
The engine was spending its time elsewhere.

**Cold re-prefills of 100–200k-token prompts.** Every agent-loop iteration is a
fresh HTTP request re-submitting the whole conversation. When the engine still
holds that prefix, the iteration pays for what was appended — 4–7k tokens on a
200k prompt. When the prefix was evicted in between, the whole prompt is
prefilled again, one chunk per engine step, and every other request gets one
token per step until it finishes.

- **Why cold:** eviction under KV pressure. The stalls sat at 60–100% of the
  then-398,175-token pool with three or four long-lived re-admitting turns
  churning the free pool. Short-lived youtube digests (~100k, 3–11 iterations)
  never came back to miss, which is why seven at once at 81.5% KV ran clean
  the night before.
- **Why ~1 s a chunk:** the page-count cliff of §3 — 233k prefilled in 26 s,
  238k in 130 s.
- **Evidence:** all 27 stall episodes in the 09-09 engine log show KV climbing
  50–190k tokens *during* the episode with ~0 prompt tokens counted, then
  100–360k landing afterwards. Fleet baseline over 09-08/09, valid data only:
  194 of 1,754 iterations at ≥100k re-prefilled ≥50k uncached tokens — 20.6M
  tokens, ~34 minutes of prefill, none at iteration 1–2.

**What fixed it:** the FP8 cutover of §3. A reproduction on the new build read
zero stall-shaped windows.

Exonerated along the way, so nobody re-derives them: the 09-08 vLLM tuning
(the stalls predate it), `harness.parallel_tool_calls` (never on), the
worker → Inner Voice migration, `#529` (no callers), the "widest row sets the
decode clock" hypothesis, and vLLM `priority` (it orders the queue and cannot
shorten a chunk).

### 6.2 A miss is counted, and announced

The data was always on disk — `cache_read` per iteration since `5531f21` —
what was missing was a number and a bell. `app/prefix_miss.py` is both, and
all three usage writers call it: the streaming and sync paths in
`app/routers/messages.py`, and `app/run_recorder.py` for direct background
runs.

- **A prefix miss** is an iteration ≥ 3 whose prompt is ≥ 100k tokens and less
  than half cached. Logged as `brain1.prefix_miss` on the session event log
  and as one INFO line — the guardian reads only ERROR/CRITICAL, so a slow
  iteration cannot roll back a promotion.
- **The turn's usage row** carries `reprefill_tokens` and `prefix_misses`, and
  so does the final assistant row's `stats`. The dashboard's Tokens panel
  shows the last 24 h beside how many turns were measured at all.
- **The announcement** is `announce()` on the guardian fan-out: news, not an
  incident. It fires when a turn's misses pass 100k tokens *while another
  request was running*, once per turn, at most every 30 minutes, journal and
  toast only. A cold prefill on an idle engine hurts nobody.

Three rules decide whether the count means anything, each one a trap the
investigation hit:

1. **Iterations 1–2 never count.** A turn's first iteration legitimately
   re-admits history the engine may not have seen for hours.
2. **A turn reading zero on every counted iteration is *unmeasured*, not
   cold** — stored NULL. Before `5531f21` nothing parsed `cached_tokens` and
   every session read 0 everywhere; a regression of that parser would look
   exactly like every iteration missing.
3. **Only a miss counts toward `reprefill_tokens`.** A healthy iteration's
   uncached tail is 3.8–6.5k, so summing every iteration would make a
   60-iteration round with a perfect cache read ~300k and fire the alert on
   health.

`app/engine_pressure.py` supplies the "was anything else running" half: one
task scrapes /metrics every 5 s into a five-minute ring through a stateless
parser, so the dashboard's own rate baseline is undisturbed. It serves all
three readers (the announcement, the KV gate, the dashboard), and the
dashboard's KV meter carries the ring's 5-minute p90 against a 65% line. The
gauge counts blocks *referenced by running requests*; a paused turn's cached
prefix sits in the free remainder, which is where it has to survive until its
next iteration.

### 6.3 Keep prefixes alive

- **The KV gate** (`workers/pool.py`, [[workers]] §2): a source declaring
  `LONG_LIVED = True` — autocode, autotriage, deep-research, arch-review — is
  not claimed while the primary's KV is over `workers.kv_gate.max_kv_usage`
  (0.60). It judges the **one-minute median**, not the newest sample, for the
  reason in §7.2. A hold keeps the item's attempt; no reading means open.
- **`workers.slots` is 6, not the 2 this page prescribed until 2026-09-17.**
  `e20fdba0` took it to 5 and `e0098082` to 6 on the rule "autocode rounds +
  triages + 1" (four rounds today), which `tests/test_loop_depth.py` pins — so
  the slot count is no longer the bound on how many long-lived contexts are
  resident at once. The KV gate above is. That trade is deliberate; §10 carries
  the bar re-stated for this shape and the first day counted at it
  (2026-09-23): KV sat well under the gate and the misses came anyway.
- **The compaction wall** (`compaction.microcompact`) moved from 0.8/0.6 to 0.72/0.52 of the 210k
  threshold — trigger ≈168k → 151k, target ≈126k → 109k. The target moved with
  the trigger so the band stays 0.2 wide: every compaction rewrites the middle
  of the prompt and forces a cold re-prefill of what follows, so a narrower
  band would trade resident KV for more of exactly those. **The in-turn pass
  had never read these fractions** — it ran on `RunOptions` defaults — so
  lowering them would have moved the wall for the first request of a turn and
  not the sixty after it.
- **`harness.parallel_tool_calls` stays off.** See §8.

### 6.4 The observer runs at its watched turn's priority, or one step below

`attach_observer_for_turn` passes `_observer_priority(options, platform)`
(`app/routers/_messages_inner_voice.py:252`, since `97a86cc0` on 2026-09-12),
which is two-valued: **the turn's own `RunOptions.priority`** for a user
platform, so a chat's second opinion runs at 0 beside the chat instead of
queueing at 1 behind every worker iteration; **`priority + 1`** for a platform
in `NON_USER_PLATFORMS`, because a worker round's observer calling the primary
at the round's own priority puts a second-opinion request in front of a human's
first token, and no human is waiting on the round. The original rule — always
1, "to yield to the agent it is watching" — guarded something equal priority
already guarantees (§5). Not a throughput lever; an intent fix, widened once
the observer started watching unattended turns too.

## 7. Benchmarks

All on the production engine with the rest of the stack down, through
`agent-services/bin/bench-admission-stall.py` — the durable reproducer that
replaced the scratchpad drafts every earlier measurement used. A = a ~119k
context decoding continuously; B = ~200k. Every request at priority 1.

**Do not run it casually**: it wants the engine to itself — the worker pool
paused and drained or the backend down — and it waits for an idle engine and
refuses to start otherwise (`wait_idle`, :128). Every request it issues goes at
priority 1 (`PRIORITY`, :61), so a live chat still outranks it; the reason for
the idle gate is that the measurement is A's step gap, and any other tenant is
noise in the number.

### 7.1 End to end (`verify`)

B ran fifteen iterations as an agent loop through the harness's own client,
with iteration 13 re-admitted with its first message changed:

| iteration | prompt | cached | uncached | TTFT | prefix_miss |
|---|---|---|---|---|---|
| 1 | 199,821 | 0 | 199,821 | 21.5 s | — (not counted) |
| 2 | 200,293 | 195,200 | 5,093 | 0.93 s | — (not counted) |
| 3–12 | 200.8k–205.4k | 195,200–201,600 | 3,795–6,547 | 0.84–1.13 s | none |
| **13** | **206,077** | **0** | **206,077** | **21.3 s** | **yes** |
| 14 | 206,606 | 201,600 | 5,006 | 0.94 s | — |
| 15 | 207,173 | 201,600 | 5,573 | 0.99 s | — |

- **Field mapping on this build:** `cache_read` is populated and reuse engages
  on the *second* request of a prefix. The old build's rule — nothing cached
  until pass 3 — does not hold here, so a cold re-admission is one miss, not
  two.
- Exactly one `prefix_miss` and one announcement, journal + desktop, voice off.
- A's step gaps: alone p50 17.8 ms; beside the warm loop p50 20.7 ms, p99
  380 ms; **during the cold re-prefill p50 622 ms, 30 of 55 steps over
  500 ms**. Zero preemptions.

Re-run on the adopted 4096 budget: same result where it matters — 0 misses
across warm iterations, exactly one `prefix_miss` at iteration 13. What moved
is the neighbour: **p50 349 ms, 1 of 84 steps over 500 ms** against 30 of 55
at 8192; KV peaked at 0.53 against 0.96. The price was 23.2 s of prefill
against 21.3 s, **+9%**.

### 7.2 The KV gauge lies during a cold prefill

`vllm:kv_cache_usage_perc` every 2 s:

| moment | reading |
|---|---|
| A (119k) decoding alone | 0.203 |
| A + B decoding, warm loop | 0.50–0.52 |
| during A's own cold 119k prefill | climbs to 0.43, then 0.203 once it lands |
| during B's cold 200k prefill beside A | climbs 0.20 → **0.956** over 21 s, then **0.498** the moment the prompt is in |

At 8192 a prompt being prefilled references ~2.5× its resident footprint and
releases the excess the moment it lands. **The overshoot belongs to the chunk
budget, not the prompt:** at 4096 the same cold 200k prompt climbs only to
0.51. The mechanism is not pinned; per-block prefix-cache checkpoints of the
linear-attention state (`mamba_cache_mode=align`, 3200-token blocks) are the
likeliest suspect, since an 8192-token chunk never ends on a block boundary.

What matters is where the excess comes from: **the free pool, which is exactly
where paused turns' cached prefixes wait for their next iteration.** At 8192
every cold long prefill transiently took ~70% of the pool and could evict the
neighbours whose prefixes it displaced — a miss that manufactures the next
miss. Two consequences:

- **A gate on the last sample would engage on every cold prefill.** Hence the
  one-minute median; `tests/test_kv_gate.py` pins that a spike to 0.96 amid a
  minute at 0.30 does not engage it and a sustained 0.66 does.
- **"KV p90 < 60% on a normal day" is stricter than it looks** — p90 is mostly
  prefills. The dashboard shows p50 beside it.

### 7.3 The chunk budget

Cold shape: A decoding, B one nonce-prefixed ~200k prompt admitted beside it,
then the same B with A stopped. The bar: take a smaller budget if A's p50
during B's prefill drops under 300 ms for at most 10% more prefill time.

| `max_num_batched_tokens` | A gap p50 / p90 / max | steps > 500 ms | B prefill beside A | B alone | KV peak |
|---|---|---|---|---|---|
| 8192 (vLLM default) | 608 / 651 / 660 ms | 29 of 54 | 20.19 s | 19.91 s | 0.90 |
| **4096 (production)** | **333 / 351 / 354 ms** | **0 of 85** | **21.64 s (+7.2%)** | **21.28 s (+6.9%)** | **0.51** |
| 2048 | 195 / 219 / 224 ms | 0 of 146 | 25.68 s (+27.2%) | 25.32 s (+27.2%) | 0.51 |

A decoding alone stepped at 17.8–17.9 ms p50 in all three: the budget costs
decode nothing. Every boot kept the FP8 pool and passed the §3.1 asserts.

**4096 is adopted, and it misses the bar.** Neither arm cleared both halves:
4096 is inside the cost budget and 33 ms over the latency line; 2048 is under
the line at nearly three times the cost. 4096 is taken for two reasons the bar
did not weigh — it is the only arm inside the cost budget while removing every
step over half a second, and it removes the KV overshoot of §7.2, which acts
on the stall's *mechanism* where the bar priced only the symptom. 2048 buys
another 140 ms of neighbour latency for +27% prefill and no further KV benefit.

## 8. Operating it

- **Boot is slow and `startsecs=900` is deliberate.** Warm caches ~240–265 s
  (170 GiB read); ~775 s after a venv change. Below the real boot time a boot
  failure becomes an *unexpected exit* that `autorestart` retries forever
  instead of parking in FATAL. `supervisorctl start` will not return for up to
  900 s.
- **Never boot it onto a host that cannot take it.** `supervisorctl stop`
  returns when processes are signalled, not when the kernel has reclaimed their
  memory, and the engine holds a **95.37 GiB** BF16 n-gram table in *host* RAM.
  On 2026-09-08 an A/B sweep started the next boot before the old one's pages
  were freed, twice, and `systemd-oomd` killed the **whole
  `agent-supervisord.service` unit** — 953 processes the first time, 793 the
  second. Peak RSS 230.3 GiB. It has happened twice since (2026-09-15 23:52Z,
  620 processes; 2026-09-17 17:09).
- **The RAM wait is not what keeps oomd off the unit** — read the header of
  `agent-services/bin/ram-boot-gate.sh`, which says so in full, before leaning on
  it. On 2026-09-17 that wait
  passed at 198 GiB and the unit was killed 129 s later, because **one** boot
  drives its own cgroup to ~226 GiB (a 170 GiB checkpoint read plus a 95 GiB
  shared mapping, page cache charged to the reader's cgroup) and so consumes the
  very gauge the wait reads — `MemAvailable` fell to 79 GiB mid-load. What
  protects the stack is `Slice=lloyd.slice` on `agent-supervisord.service`
  (oomd watches only `app.slice`); the wait answers the narrower question it
  was written for — has the *previous* engine's mapping been released — and
  raising its floor will not stop a kill.
- **Two routes, one definition.** Both thresholds live in
  `agent-services/bin/ram-boot-gate.sh` and neither route keeps a copy: the
  landing route reads it through `scripts/automod/ram_gate.py`, the sweep route
  shells in the same file. Production restarts go through
  `scripts.automod.round restart --only agent-llm-primary`, whose
  `_restart_primary` (in `scripts/automod/promote.py` — find it by that symbol,
  this page used to carry a line number and it was 596 lines wrong) waits
  `MemAvailable` back to `PRIMARY_RAM_FLOOR_GIB` and refuses under
  `PRIMARY_RAM_ABORT_GIB`. `flash-next-run-arm.sh` is the A/B sweep runner, and
  waits to `SWEEP_RAM_WAIT_GIB` and aborts below `SWEEP_RAM_ABORT_GIB` — a lower
  bar by design (the definition's header says why), so it is not the production
  restart route even though it restarts the same program. Both pairs are
  machine- and day-specific; read that file and `/proc/meminfo`, not this page.
- **A venv switch compiles, and the compile is its own memory event.** Both
  primary venvs carry the same flashinfer, whose `build.ninja` embeds the venv's
  absolute include paths — so every switch between `vllm-flash-next-main` and
  `-main-0910` rebuilt all ~64 `fused_moe_120` objects `nproc` (32) wide: on
  2026-09-18 09:39 that put 83 GiB of `cicc` on top of the PLE table, hit
  `MemAvailable` 0, and the boot had to be stopped. Since `0150e88b` the
  launcher defaults `FLASHINFER_WORKSPACE_BASE` to `$VLLM_VENV` (one kernel
  cache per venv, so a switch costs nothing) and `MAX_JOBS` to 8 (a real
  rebuild peaks near 45 GiB).
- **Don't start a second Flash-Next engine on GPU 1** while this one is up:
  two 95 GiB PLE tables is that same OOM.
- `flash-next-bootfacts.sh` after any boot you did not watch.

**Parallel tool calls remain off.** The original plan listed
`harness.parallel_tool_calls.enabled: true` on the grounds that fewer
iterations mean fewer re-admissions. That reason was wrong: the flag changes
how an iteration's tool calls are *dispatched*, not how many the model emits,
and the prompt is identical either way. What it saves is tool wall time inside
an iteration. Its soak checklist — `mcp_pool:` warnings in `logs/server.err`,
`[iv.observer] inject` placement, `harness.empty_terminal_iteration` counts —
has still not been run as of 2026-09-11.

## 9. Don'ts

- Don't raise KV bytes on BF16, touch the power cap (§2), or revisit
  `--async-scheduling`, `GDN_PREFILL_BACKEND` or the MTP arms — none was the
  cause of anything in §6.
- Don't read `tokens/step` or `prompt_tokens_total` as evidence about prefill:
  on this build both are blind until a request completes. Read KV climb and
  per-iteration `cache_read`. Don't compute a reuse rate from `usage.db`'s
  aggregate columns (`input_tokens` is the peak, `cache_read` the sum). Don't
  select sessions by filename date — worker ids were minted in UTC until
  2026-09-10.
- Don't use `pool.pause()` as a mitigation — every landing's `round restart`
  clears it.
- Don't put YaRN in `--hf-overrides` (§4).

## 10. Still open

- **The acceptance bar, for the fleet as it runs today** — `workers.slots` 6
  and `workers.sources.autocode.max_inflight` 2 (4 until the evening of
  2026-09-24), so two long-lived rounds resident beside triage, the scheduled tasks and a chat. One normal day at
  that shape with Alan chatting passes when (a) no chat turn carries a prefix
  miss, (b) the fleet's misses and re-prefilled tokens per day are at or under
  §6.1's per-day baseline, (c) KV p50 is under the gate
  (`workers.kv_gate.max_kv_usage`, 0.60), and (d) no two-request window runs
  under 15 tok/s that is not a cold admission. The bar written for slots = 2
  asked for *zero* misses after iteration 2; with several rounds each idle for
  minutes between iterations (below) that cannot hold on one pool, so the
  re-stated bar asks for what a person feels (a, d) and a budget for the rest
  (b). Clearing it needs a day of real chat, which the window below is not.
- **The counted reading: 2026-09-23 00:00 → 2026-09-24 00:00 UTC** (one full
  day, slots 6, four rounds, the 844,969-token FP8 pool), from `usage.db`, the
  `brain1.prefix_miss` events and the engine's status lines, derived by
  `scripts/vllm_prefix_miss_window.py` over the extract committed as
  `tests/fixtures/vllm_prefix_miss_2026-09-23.json`: **219 turns, 219
  measured, 65 turns carrying 172 misses and 16,084,128 re-prefilled tokens**,
  worst turn 601,503. Taken at four rounds; the shape now running (slots 6,
  2 rounds) has not been counted yet. By session kind: autocode 158 misses (14.5M), then
  deep-research 4, arch-review 3, youtube-digest 3, review 3, autonomy 1.
  Chat 0 — but only 4 of the 219 turns were chat, so (a) was not tested and
  this is not the normal day the bar asks for. 197 miss iterations logged, 85
  of them fully cold (nothing cached) and 112 partial.
- **Against §6.1, per day:** §6.1's baseline is 194 misses and 20.6M tokens
  over *two* days (09-08/09) — 97 misses and 10.3M tokens a day. This window
  is 172 and 16.1M in *one* day, about 1.8x the misses and 1.6x the tokens a
  day of the stall that §6 exists for; (b) fails. The definitions differ at
  the edge (§6.1 counted ≥ 50k uncached, the counter counts < 50% cached, both
  at ≥ 100k), which does not close a 1.8x gap. The 2026-09-11 spot reading
  this page used to quote (135 misses, 20.4M, 24 h) and the 09-19..21 days
  read off the old database by #1339's triage are history now: `usage.db`
  starts at 2026-09-22 20:17 UTC, after the 09-22 wipe, and cannot re-derive
  them.
- **Where a KV history can come from.** Not `engine_pressure`: its ring is
  300 s (`DEFAULT_WINDOW_S`, `engine_pressure.window_seconds`) in the backend's
  memory, emptied by every restart. The engine's own status line is the only
  record — `Running: N reqs, Waiting: N reqs, GPU KV cache usage: NN.N%`
  every 10 s while it has work, in `logs/services/agent-llm-primary.log*`,
  stamped in local time with no year — and rotation keeps about two days of
  it. Over the window, 5,761 such lines: **KV p50 0.26 / p90 0.55 / max
  0.83**, three requests running at the median. Idle intervals print no line,
  so the true p50 is lower still. (c) passes.
- **Eviction, but not by pressure at the gate.** Each miss joined to the lines
  in its *gap* — from the previous iteration's last proposed tool call, when
  its request ended and its blocks went back to the free queue, to the miss
  iteration's request: gap p50 240 s; KV peak in the gap **p50 0.544 / p90
  0.713**, max 0.795; 54 of 197 at or over the 0.60 gate, **none over 0.90**.
  The pool never came close to full, so this is not the 09-09 shape. What the
  gaps do show is churn: vLLM's free queue is LRU, and in **117 of 197**
  gaps the engine computed at least as many tokens as the free pool held at
  the gap's tightest, which is enough to have reclaimed a paused prefix at
  moderate KV. That is eviction the KV gate cannot see — it judges occupancy,
  and the cost here is gap length times fleet throughput. The other 80 misses
  had no such churn and are not explained by eviction; they are the
  candidates for the branch below. Whether to chase that upstream or accept
  the loss is Alan's call, not a round's.
- **The unannotated draft group.** `_warn_if_unannotated_eagle_mamba` in
  vLLM's `v1/core/kv_cache_utils.py` — name the function, not the line: in
  the **production** venv at `dff1bde` the def is 2190 and its
  `logger.warning` 2215; the 2133/2166 this page used to quote is the
  `-0910` revert target's copy, and the `kv_cache_utils.py:1871` in older
  notes is the *worker* build's. It fires on every production boot, and in
  this build it says only "Speculative decoding (method=mtp) is enabled but no
  KV cache group could be identified as the draft model's" — the older
  builds' "prefix-cache reuse across requests will be disabled" is not in
  its text. Reuse across requests is not off wholesale here: 112 of the
  window's 197 misses still read part of their prompt from cache.
- **YaRN's tool-choice effect is unsettled** (§4). Its short-prompt,
  long-context, retrieval, memory and prefill costs were measured on
  2026-09-21 and are nil or acceptable; 35 tool-choice queries leaned 95 vs
  89 of 105 against it, which is not conclusive and is the only thing
  standing between the staged arm and a decision.
- **The ≤2400 MHz clock cap from the only configuration described as stable
  on GPU 1 is not set** (§2.3), and the unit now clamps the card at 450 W, not
  that configuration's 400 W.

## 11. Found on the way

- **Direct worker turns ignore every `harness.*` key.**
  `workers/sources/_common._worker_run_options` builds `RunOptions` without
  `_get_harness_kwargs()`, whose docstring claims every construction site
  splats it. So `run_prompt_on_primary` jobs run with no stream-stall bound
  (0 against config's 60), no preserved thinking (0 against 6), and
  `tool_search` **on** where config and the override file both say off. Only
  the compaction fractions are passed now; the rest is left for a decision.
- **Tests wrote rows into the live `usage.db`.** `usage_store.DB_PATH`
  resolved from `__file__` and nothing isolated it. `tests/conftest.py`
  redirects it now, and `_conn` reopens when the path moves.
- **A launcher `DRY_RUN` consumed a staged arm.** The one-shot arm env is
  sourced and deleted before the `DRY_RUN` exit, so a test or a human checking
  the command line during a sweep would have eaten the arm. `ARM_ENV`
  overrides the path.

## Review log

- 2026-09-21 — **stale.** §3's cache table, §2.3's clamp values (275/450/275), §2.4's index table, §2.5's persistence-mode claim, §5's knob defaults, §3.1's three asserts and the `workers.slots`-adjacent KV gate all verified against the live engine (`vllm:cache_config_info`: fp8, 844,969 tokens, block 3200) and the tree. Corrected: §1's verbatim command line was missing `--kv-cache-dtype fp8` and `--engram-config`; §6.3's "`workers.slots` stays 2" (now 6, `config.yaml:1163`); §6.4's observer priority (one step lower on unattended platforms since `97a86cc0`); §7's "priority-0 requests" (the bench runs at 1); §8's OOM account (four kills, not two — and `Slice=lloyd.slice`, not the RAM wait, is what keeps oomd off the unit, plus the FlashInfer JIT rebuild since `0150e88b`) and its restart route; §10's venv line numbers, which had drifted onto the `-0910` revert target. Filed #1337 #1338 #1339 #1340 #1341.
