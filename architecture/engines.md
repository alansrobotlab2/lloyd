---
segment: architecture
tags: [architecture, lloyd, models, vllm, llamacpp]
type: reference
status: implemented
date: 2026-09-11
---

# Model slots and inference engines

Two LLM slots and one TTS engine, all local, all under supervisord. A slot is
an *endpoint* in `config.yaml` (`models.<alias>`); which model answers there
is decided by the supervisord program's `environment=` and its start script.
Those three places can drift, and did (2026-09-06, a rollback put a 4B under
the 35B's alias), which is why every slot carries an identity check.

## The slots

| Slot | Port | Engine | Model | GPU | Program |
|---|---|---|---|---|---|
| `primary` | 8096 | vLLM (main branch venv `vllm-flash-next-main`) | Qwen3.8-Flash-Next, FP8 KV | GPU 1, RTX PRO 6000 96 GB | `agent-llm-primary` → `agent-services/bin/start-qwen38-flash-next.sh` |
| `secondary` | 8091 | llama.cpp `llama-server`, `--parallel 1` | Qwen3.6-35B-A3B UD-Q3_K_XL | GPU 2, RTX 3090 24 GB | `agent-llm-secondary` → `start-secondary.sh` |
| TTS | 8090 | Qwen3-TTS (`.venvs/qwen3-tts`) | cloned voice `clone:dave_cullen` | GPU 0, RTX 3090 24 GB | `agent-tts` → `start-qwen3-tts.sh` |

GPU numbers are `nvidia-smi` indices with `CUDA_DEVICE_ORDER=PCI_BUS_ID`
pinned in every program's environment; `/dev/nvidiaN` does not match them.

## Primary: what keeps it fast

The long version is [[vllm-throughput-mitigation]]. The short version:

- **FP8 KV cache** (`KV_CACHE_DTYPE=fp8` in the program's `environment=`):
  a 692k-token pool at 3200-token pages, ×1.74 over BF16, and it sidesteps
  the >200k-token prefill cliff. `flash-next-bootfacts.sh` and
  `models.primary.expect_kv_cache_dtype` / `expect_kv_pool_tokens_min` assert
  it at boot.
- **Chunk budget 4096** (`MAX_NUM_BATCHED_TOKENS`), not vLLM's 8192: a cold
  200k prefill beside a chat costs it 333 ms a step instead of 608 and no
  longer transiently references 2.5× its KV.
- **Prefix misses are counted** (`app/prefix_miss.py`) on every usage writer
  and announced once per turn through the guardian fan-out when they pass
  100k tokens with another request running.
- **Long-lived workers wait for room.** `app/engine_pressure.py` samples
  `/metrics` every 5 s; the pool's KV gate holds `LONG_LIVED` sources while
  the one-minute median is over 60%.
- **The 95 GiB n-gram table lives in host RAM.** Never restart the primary
  twice in quick succession: two tables coexisting on a 251 GiB box got the
  whole supervisord unit OOM-killed twice on 2026-09-08.
  `flash-next-run-arm.sh` waits for `MemAvailable` > 150 GiB.

## Secondary: single-tenant by design

llama.cpp divides `--ctx-size` across slots and the full 262,144 window was
the point, so `--parallel 1`. Everything on the secondary queues: session
titles (`app/session_titles.py`, geometric schedule), post-session
summaries and voice summaries (`app/secondary_models.py`), the cluster
pair-judge, and any agent turn routed there. `vllm_metrics._translate_llamacpp`
renames its Prometheus vocabulary into vLLM's so one dashboard card serves
both; KV occupancy and TTFT are reported `None` there, not `0`.

## Identity and health

- `models.<alias>.expect_model` is a case-insensitive substring checked
  against vLLM `/v1/models` or llama.cpp `/props` by `app/model_identity.py`
  — at boot (detached, with retries) and on
  `GET /api/models/identity?refresh=1`. A slot with no `expect_model` reads
  `unchecked`. **Update it whenever a slot's occupant changes.**
- `app/vllm_metrics.py` scrapes both engines for the dashboard; counters
  are reported as rates against the previous scrape, and a counter that goes
  backwards yields `None`.
- Two venvs can serve the primary: `vllm-flash-next-main` (what is served)
  and `vllm-qwen38-flash-next` (the PLE-offload-worker build, the script's
  own fallback). `SETUP.md` Part 4 and Part 9 carry the builds and the
  weights.

## Subagents and workers follow the caller

`subagents.<type>.model: ''` means "whatever spawned me", shipped in the MCP
request `_meta`. Worker sources name `model="primary"` explicitly. The
Inner Voice observer runs on the primary at the priority of the turn it
watches.

## Related

[[harness]] (the client), [[voice]] (TTS shaping happens client-side in
`agent-services/tts_shaping.py`), [[infrastructure]] (programs and ports).
