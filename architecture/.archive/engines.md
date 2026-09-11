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

GPU numbers are `nvidia-smi` indices under `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
`/dev/nvidiaN` does not match them. Neither LLM program sets that variable in
its supervisord `environment=` — each **launcher** exports it, next to its own
`CUDA_VISIBLE_DEVICES`, which is the only reason the pair is ever consistent
("CUDA_DEVICE_ORDER is mandatory — without it the runtime reorders by
capability", `start-qwen38-flash-next.sh`). Only `agent-tts`, the voice worker
and the two qmd programs pin it in the conf. So read a launcher's *exports* to
find out which card a slot is on, and not its prose: `start-qwen3-tts.sh` still
says "GPU 1" in its header line while exporting `CUDA_VISIBLE_DEVICES=0`.

## Primary: what keeps it fast

The long version is [[vllm]]. The short version:

- **FP8 KV cache** (`KV_CACHE_DTYPE=fp8` in the program's `environment=`):
  a 692,263-token pool at 3200-token pages, ×1.74 over BF16's 398,175 — and
  the page size is what moves the prefill cliff, because the kernel's cost
  there is a page *count*. BF16's ~150-page wall at 1600 tokens a page is
  why 233k prefilled in 26 s and 238k took 130 s; 3200-token pages put the
  same wall past `max_model_len`. Asserted twice over, against two different
  sources: `flash-next-bootfacts.sh` (`EXPECT_KV_DTYPE=fp8`,
  `EXPECT_KV_POOL_MIN=600000`) reads the boot log, and
  `models.primary.expect_kv_cache_dtype` / `expect_kv_pool_tokens_min` read
  the running engine's own `vllm:cache_config_info`. Only the vLLM-main venv
  can serve it — the launcher refuses `fp8` on the worker build, whose QSA
  kernel has no fp8 path (PR #55557), rather than booting BF16 quietly.
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
  `flash-next-run-arm.sh` waits for `MemAvailable` to climb back above
  150 GiB between boots, and aborts the arm outright below 120.
- **`startsecs=900` is a boot budget, not a guess.** A warm boot is
  240–265 s, but a venv change empties the torch.compile / Triton cache and
  the first boot after one sat silent for ~8 minutes between "Model loading
  took" and the KV cache line — 775 s to health, measured 2026-09-10. At
  300 s supervisord would have called that healthy boot a crash and, with
  `autorestart`, torn it down three times over. The opposite setting is the
  other failure: at the 27B's `startsecs=5` the process is marked RUNNING
  long before it can serve, so a crash *during* load reads as an unexpected
  exit and autorestart retries it forever, pinning the GPU. At 900 s a
  boot-time failure is a start failure instead — capped by `startretries`
  and parked in FATAL where it is visible. The cost is that
  `supervisorctl start` will not return for up to 900 s; poll :8096.

## Secondary: single-tenant by design

llama.cpp divides `--ctx-size` across slots and the full 262,144 window was
the point, so `--parallel 1`. Everything on the secondary queues: session
titles (`app/session_titles.py`, geometric schedule), post-session
summaries and voice summaries (`app/secondary_models.py`), the cluster
pair-judge, and any agent turn routed there. `vllm_metrics._translate_llamacpp`
renames its Prometheus vocabulary into vLLM's so one dashboard card serves
both; KV occupancy and TTFT are reported `None` there, not `0`, because it
publishes neither an occupancy gauge nor a per-request TTFT count and a `0.0`
would render a full cache as an empty one. Two more special cases hang off the
same `is_llamacpp` flag: a reachable llama.cpp server is **awake** by
definition — it has no sleep-state gauge, and falling through to the vLLM
check renders a healthy engine "asleep" forever — and its model name comes
from a `/props` probe cached per engine lifetime, since it labels none of its
metrics and the loaded GGUF cannot change while the process lives.

**`secondary_enabled` is the slot's real switch, and it outranks supervisord.**
`server.py::_sync_secondary_llm_state` reconciles the program against that flag
on every backend boot, so the conf's own `autostart=true` is not the deciding
vote: leaving the flag false silently stopped the secondary three seconds after
each backend restart. The same flag reroutes the *callers* —
`config.resolve_model_alias` rewrites `secondary` to `primary` while it is
false, logged once per name because it is otherwise undetectable. Inner Voice's
config said `model: secondary` from 2026-05-07 onward and ran on the primary
the whole time for exactly that reason; when `de893d7` flipped the flag on for
the autonomy scheduler, the observer silently moved to what this slot then ran,
a 4B, and nothing in any log said so. Both paths stand down when
`services.sync_secondary_llm` is false, which is what stops an automod canary
booted from a worktree from reconciling the live engines against its own config.

## Identity and health

- `models.<alias>.expect_model` is a case-insensitive substring checked
  against vLLM `/v1/models` (`root`, `id`) or llama.cpp `/props`
  (`model_path`, `model_alias`) by `app/model_identity.py` — at boot
  (detached, 6 attempts 15 s apart) and on
  `GET /api/models/identity?refresh=1`. It retries only while something is
  `unreachable`, because the 35B takes minutes to page 17 GB onto a 3090
  while a `MISMATCH` is conclusive on the first look and waiting would only
  delay the alarm. The KV cache carries its own second verdict beside it
  (`ok` / `REGRESSION` / `unknown` / `unreachable`), since the right model
  can still be served the wrong way. A slot with no `expect_model` reads
  `unchecked`. **Update it whenever a slot's occupant changes.** The sweep
  only ever reports: restarting a slot is precisely the operation that would
  have swapped the model in the first place.
- `app/vllm_metrics.py` scrapes both engines for the dashboard; counters
  are reported as rates against the previous scrape, and a counter that goes
  backwards yields `None`.
- Two venvs can serve Flash-Next: `vllm-flash-next-main` (what the program
  names, and the only one with the fp8 QSA path) and `vllm-qwen38-flash-next`
  (the PLE-offload-worker build — and the launcher's *own* fallback default,
  so a dropped `VLLM_VENV` boots BF16, which is the regression
  `expect_kv_cache_dtype` exists to catch). A third, `vllm-qwen3.8`, belongs
  to the revert target the conf names at the top: the 27B at
  `start-qwen3.8-27b-nvfp4.sh`. `SETUP.md` Part 4 (venvs) and Part 9 (LLM
  models) carry the builds and the weights.

## Subagents and workers follow the caller

`subagents.<type>.model: ''` means "whatever spawned me", shipped in the MCP
request `_meta` as `lloyd/model` and `lloyd/base_url` — `Task` runs in the
aggregator process and has no other way to know. It was pinned to `primary`,
so a turn running on the secondary delegated its subagents back to the primary
and the secondary was never exercised by the fan-out work Task exists for.

Worker turns built by `workers/sources/_common.py` name `model="primary"`
outright. The exception is the scheduled-task source: `autonomy.run_task`
resolves the task file's own `model:` frontmatter through
`resolve_model_alias`, which is the one path by which anything reaches the
secondary deliberately. The Inner Voice observer runs on the primary
(`inner_voice.model`, pinned there after the 4B episode above) at the priority
of the turn it watches — `attach_observer_for_turn` passes `options.priority`
straight through, so a chat's observer runs at 0 rather than queueing at the
configured `1` behind every worker iteration.

## Related

[[harness]] (the client), [[voice]] (TTS shaping happens client-side in
`agent-services/tts_shaping.py`), [[infrastructure]] (programs and ports).
