---
segment: architecture
relations:
  related-to:
  - architecture/tools.md
  - architecture/infrastructure.md
  - architecture/vllm.md
  - architecture/backlog.md
  - architecture/knowledge-graph.md
  - architecture/memory.md
  - architecture/index.md
tags: [architecture, lloyd, djev, decisions, calibration]
summary: GPU 2's structured-decision engine — how it is served, its wire
  protocol, the one client, three tools, the schema registry, three shadow
  seams, the eval arm, and the measurements that say rank with it and do not
  gate on it.
type: reference
status: implemented
date: 2026-09-21
---

# djev — the structured-decision engine on GPU 2

DiffusionGemma 26B-A4B NVFP4 (`github.com/mmastrac/djev-spark`) serves Jev's
`POST :8011/v1/systemone`. That endpoint answers typed questions read off one
diffusion canvas: yes/no (`noul`), one-of-N (`choice`) and an ordered scale
(`score`). It replaced the Qwen3.6 secondary on 2026-09-20 and was wired into
Lloyd the same day.

**GPU 1 runs at 100% utilization. GPU 2 sits at 0%** with 23.7 GiB of weights
loaded. Every decision moved here is pure throughput. One read costs ~40 ms
plus ~0.25 ms per prompt token. A call from this client takes one read or
four, and the first live rows' latency says it is usually four (§3.3).

It is **not a chat slot**. It is absent from `models:` and from
`resolve_model_alias`, the model dropdown has never heard of it, and nothing
routes a turn to it. That stays true.

```
 GPU 2 · RTX 3090 · 24 GiB · SM86        supervisord program agent-djev (autostart=false;
 ┌─────────────────────────────────┐     server.py starts/stops it from djev.enabled)
 │ vLLM + djev overlay      :8010  │ ◀── Mission Control engine card (/metrics)
 │ DiffusionGemma 26B-A4B NVFP4    │
 └────────────────▲────────────────┘
                  │ completions with logprobs
 ┌────────────────┴────────────────┐
 │ structured_server.py     :8011  │  POST /v1/systemone · GET /health
 └────────────────▲────────────────┘
                  │
      app/djev.py — the one client: ask · ask_sync · rank; None on any failure
        ▲                    ▲                           ▲
 agent_mcp/djev.py     app/djev_shadow.py          agent_mcp/vault.py
 djev_rank             bounded queue → worker      RECALL_DJEV_RERANK (off)
 djev_decide           → shadow.jsonl              ← eval/run_eval.py --djev-rerank
 djev_status              ▲       ▲        ▲
                        rerank  dedupe  entity      the three shadow seams

 eval/djev/schemas.py   frozen question shapes, floors, thresholds, gate_ready
 eval/djev/replay.py    calibration against recorded corpora
```

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

Four rules follow. This tree gets every one of them wrong when it reasons
instead of measuring:

1. **Ranking is trustworthy; the probability attached to it is not.** AUC holds
   across every framing, so the *ordering* is real signal. The number attached
   to it moves by a third from a cosmetic change, and each framing has its own
   optimal cutoff. **A fixed 0.5 threshold is meaningless.** `config.yaml`'s
   `djev:` block said "calibrated probabilities" until this landed. It now says
   "self-consistent scores", because the measurement contradicts the first.
2. **Order-averaging is not the fix.** It lands between the two arms (AUC 0.815)
   and `acc@0.5` stays 0.53. The fix is a threshold measured per schema, with
   the schema frozen once calibrated, **option order included**.
   `eval/djev/schemas.py` holds a hash over the question JSON with key order
   preserved, and `Schema.check()` raises `SchemaDrift` when a rewording
   outlives the number measured under it. That is a test, not a convention
   (`tests/test_djev_schemas.py`).
3. **`noul` is not automatically safer than `choice`.** The yes/no form has no
   option list to order, and it measured *worst*. Pick framings by measurement.
4. **So ranking, ordering and shortlisting are safe today, and a yes/no gate
   ships with a calibrated threshold or not at all.** Nothing in this round
   gates anything: `Schema.gate_ready` is False on all five schemas, and each
   carries a `gate_blocked_reason`.

---

## 2. Serving

### 2.1 The engine

| | |
|---|---|
| Model | `nvidia-diffusiongemma-26B-A4B-it-NVFP4`, under `agent-services/llm/models/` |
| Upstream | `agent-services/llm/djev-spark` (github.com/mmastrac/djev-spark): the vLLM overlay and `server/structured_server.py` |
| venv | `.venvs/vllm-djev`: a stock vLLM nightly plus the overlay's nine Python files. Built by `agent-services/setup/setup-djev.sh`, which replaces upstream's Dockerfile and runs its three safety checks |
| Start script | `agent-services/bin/start-djev.sh`: upstream's `entrypoint.sh` adapted from a DGX Spark to a discrete 24 GiB card |
| Program | `agent-djev` in `agent-services/supervisor/conf.d/agent-djev.conf` |
| Ports | `:8010` vLLM (OpenAI API + `/metrics`), `:8011` structured decisions, `5300` vLLM's internal ZMQ port (moved so it cannot collide with the primary's default) |
| Logs | `agent-services/logs/agent-djev.log`, `.err` |

**Why an SM86 card can serve NVFP4 at all.** `hf_quant_config.json` excludes
`lm_head`, `*self_attn*`, `*mlp*` and `*router*`. Only the 128 routed experts
are 4-bit, and everything else stays BF16. vLLM serves that through the NVFP4
**Marlin W4A16** path, which gates at capability 75 and supports exactly this
checkpoint's group size of 16. The Blackwell-only CUTLASS and FlashInfer MoE
backends reject themselves on capability, and selection falls through to
Marlin. It is weight-only: the weights unpack to 16-bit for the GEMM, so it is
slower than native FP4 but numerically the same weights. FlashInfer is
deliberately not installed, because its MoE backends sit ahead of Marlin in the
selection order.

### 2.2 How it runs

`start-djev.sh` runs three things:

1. **vLLM** on `:8010`, with the overlay's `--diffusion-config
   '{"canvas_length": 128}'`. The canvas width bounds the answer template and
   any thought block. `--override-generation-config '{"max_new_tokens": null}'`
   lets the canvas, not a generation cap, bound a read. `--enable-prefix-caching`
   and `--max-logprobs 32` are required by the structured reads.
2. **The structured server** on `:8011`, in a loop that restarts it whenever it
   exits. Its code can then be reloaded without restarting vLLM, which is the
   expensive half. Only vLLM exiting ends the script.
3. **A detached warm-up read.** It absorbs the ~1 s Triton JIT of two logprob
   kernels that happens on the first structured read after boot and never
   again. The first caller after a restart is usually a shadow row, so without
   it the spike would land in the latency distribution the seams are judged on.

A **VRAM preflight** refuses to start unless the card has room for the weights
(measured from the safetensors on disk), a KV floor, the sampler transient and
overhead. When it refuses, it prints what is on the card.

**The tuning knobs live in the program's `environment=`**, not in
config.yaml. A few more (`EXTRA_ARGS` = `--async-scheduling`, `KV_CACHE_GB`,
`MAX_NUM_BATCHED_TOKENS`, `OVERHEAD_MIB`, `MIN_KV_MIB`, `WAIT_SECS`) exist only
as defaults in `start-djev.sh`, and the conf can override any of them. Two files that both claim to set an engine's context is how the
secondary's config and the model it served drifted apart on 2026-09-06.

| Knob | Value | Why not upstream's |
|---|---|---|
| `GPU` | `2` | `CUDA_DEVICE_ORDER=PCI_BUS_ID` is mandatory, or index 2 lands on another card |
| `MAX_MODEL_LEN` | `131072` | vLLM's own measured ceiling here is 154 976. The full 262 144 needs 5.98 GiB of bf16 KV on top of 17.53 GiB of weights, against 23.6 GiB of card |
| `MAX_SEQS` | `1` | profiling materialises ~10 fp32 copies of `[MAX_SEQS × CANVAS, vocab]`, which is 40 GiB at upstream's 32×128 and 1.25 GiB at 1×128 |
| `CANVAS` | `128` | the served canvas width |
| `GPU_UTIL` | `0.97` | upstream's 0.40 is a fraction of a Spark's 121 GiB unified memory. On a 24 GiB card it would not hold the weights |
| `KV_CACHE_DTYPE` | `bfloat16` | the checkpoint asks for FP8 KV, which is SM89+. Ampere has none, and this is what makes the full window impossible |
| `ATTN` | `TRITON_ATTN` | |

**Widening `CANVAS` was measured and rejected (#1345, 2026-09-21).** The
checkpoint's own `config.json` says `canvas_length: 256`; 128 is served to save
the sampler transient.

- **A wider canvas boots only with the KV pool sized by hand.** vLLM's profiling
  under-reserves the transient: at 256 it gave KV 3.81 GiB and then OOMed in
  warm-up on one 256 MiB `[256 × vocab]` buffer.
- **KV also carries ~0.9 GiB of fixed sliding-window overhead.** 1 GiB serves
  ~4.7k tokens and 2 GiB ~53k.
- **What booted:**
  - 256: `KV_CACHE_GB=2` with `MAX_MODEL_LEN=49152` (splits at 65 rank rows);
  - 384: `--kv-cache-memory 1400000000` in `EXTRA_ARGS` with `MAX_MODEL_LEN=12288`
    (splits at 97). `KV_CACHE_GB` takes whole GiB only.
- **Why it was not worth it.** The recall eval over deeper pools found no gain:
  44 rows was +0.011 doc_hit (noise) for +0.22 s, and 92 rows was −0.115. The
  served width also changes djev's answers to an identical request (66/87
  identical between 256 and 384).

### 2.3 Switching it on and off

- **`djev.enabled` in config.yaml is the switch.** `server.py::_sync_llm_slots`
  starts or stops `agent-djev` to match it at backend boot. That is why the
  program is `autostart=false`: with `autostart=true`, supervisord would load
  17.6 GiB onto GPU 2 at every boot and the backend would kill it seconds
  later. Every process reads config.yaml once, at boot, and the tools and two
  of the three seams live in the aggregator. So to change it, edit the flag and
  run `round restart` (both legs). A backend-only restart stops the engine
  while the aggregator keeps calling it and writing `djev: null` rows. The
  entity sweep picks the flag up on its next run.
- **GPU 2 holds djev or the secondary, never both.** `secondary_enabled` and
  `djev.enabled` are an either/or. With both true, `server.py` stops both
  programs and logs an ERROR, and `start-djev.sh` refuses on the VRAM check.
- **`app/llm_slots.py::is_enabled("agent-djev")` is the one reader** for the
  boot reconcile, the client, the shadow recorder, the Services tab and the
  agent's own service view. Six surfaces disagreeing about the secondary is the
  failure that module exists to prevent. The dashboard card
  (`vllm_metrics.configured_engines`) reads the same flag directly. With the
  flag off, and both processes restarted, every client call returns `None`,
  every seam records nothing, and the card disappears.
- **A tuning change** is an edit to `agent-djev.conf`, then `supervisorctl
  reread` and `update`. `update` restarts a changed program itself, so a
  `restart` after it boots the engine a second time. Unlike the primary, djev is not a
  `round restart` leg. Nothing in production waits on it, so a restart costs
  advice only.

---

## 3. The wire protocol

### 3.1 The request

```json
{"model": "djev",
 "state": "The nightly backup job failed three times this week with a disk-full error on /mnt/backup.",
 "questions": {
   "urgent":   {"type": "noul",   "instructions": "Does this need attention today?"},
   "area":     {"type": "choice", "instructions": "Which area does this belong to?",
                "criteria": {"storage": "disks, backups, filesystems",
                             "network": "connectivity, DNS, routing",
                             "application": "application code and services"}},
   "severity": {"type": "score",  "instructions": "How severe is this?",
                "criteria": ["cosmetic", "minor", "major", "critical"]}}}
```

The `criteria` shape follows the type:

| Type | `criteria` | Answer `value` in the client |
|---|---|---|
| `noul` | optional `{"true": …, "false": …}` descriptions | P(yes) |
| `choice` | option name → description (or null). **Insertion order is what the model reads** | the chosen option's name |
| `score` | an **ordered** list of levels, worst first | the expected level index (0-based) |

Optional body keys the server accepts: `instructions` (context for every
question), `samples` (`"auto"` by default, or N), `auto_threshold`,
`auto_max`, `seed`, `steps`, `think` (a thought budget in tokens, 0–4096),
`ask`, `chunk_rows`, `chunk_prompt`, `sequential`. Per question: `depends_on`,
`ask_if` (`{id: [answers]}`, asked only when the answer matched; a skipped
answer comes back null) and `alone`. Lloyd uses `instructions`, `samples` and
`seed` and nothing else.

The same port also serves `/v1/chat/completions` (the same decision from an
OpenAI-shaped call) and `/v1/raw/chat/completions` (passed through to vLLM).
Lloyd uses neither.

### 3.2 The response

The server's answers, from that request (live, 2026-09-20):

```json
"answers": {
  "urgent":   {"type": "noul", "noul": 0.9608},
  "area":     {"type": "choice", "choice": "storage", "confidence": 0.9997,
               "probabilities": {"storage": 0.9997, "network": 0.0002, "application": 0.0001}},
  "severity": {"type": "score", "score": 2.046, "confidence": 0.9523,
               "legend": {"0": "cosmetic", "1": "minor", "2": "major", "3": "critical"},
               "probabilities": {"0": 0.0001, "1": 0.0007, "2": 0.9523, "3": 0.0469}}},
"usage": {"input_tokens": 176, "output_tokens": 17},
"diagnostics": {
  "chunks": [["urgent", "area", "severity"]],
  "samples": {"n": 4, "policy": {"mode": "auto", "max": 4, "threshold": 0.1, "extended": true}},
  "timing": {"total_ms": 156.3, "reads": 4},
  "questions": {"urgent":   {"label_mass": 0.801, "argmax_is_label": true},
                "area":     {"label_mass": 0.834, "argmax_is_label": true},
                "severity": {"label_mass": 0.975, "argmax_is_label": true}}}
```

Three things in that shape are traps, and the client exists partly to remove
them (§4):

- **`noul` returns P(yes) and nothing else**: no `confidence`, no
  `probabilities`.
- **`score` returns two different numbers.** `score` is the expected level
  index, and `confidence` is the modal probability. They answer different
  questions ("how severe" against "how sure").
- **The only honest confidence signal is one level down**, in
  `diagnostics.questions.<id>.label_mass`. That is how much of the model's
  probability landed on legal label tokens. The returned `probabilities` are
  renormalized over the label set regardless, so they always sum to 1 and look
  confident, even when 92–98% of the mass sat elsewhere.

### 3.3 Samples, chunks and latency

- **The server's default is `samples: "auto"`, and it is all or nothing.** It
  takes one read. If any question's first-read entropy is above
  `auto_threshold` 0.1, it takes `auto_max − 1` (3) more at once, so a chunk
  costs 1 read or 4. The client sends no `samples`, so it gets that default.
  The call above took four reads in 156 ms, and the first live shadow rows
  have a median of ~160 ms for 1–3 questions. The ~40 ms figure in §7.1 is
  `samples=1`.
- **A schema whose answer template does not fit the 128-token canvas is split
  into chunks**, and each chunk is a separate shared context
  (`diagnostics.chunks`). **Answers from different chunks are not
  comparable.** The split is by template rows, not by question count, so where
  it falls depends on the ids and types. Measured with rank-shaped questions:
  34 → chunks of 33 and 1, 48 → 33 and 15, 64 → 33, 30 and 1. The client's
  `CANVAS_CHUNK_QUESTIONS` 32 is a conservative bound under that, and it
  refuses to sort a ranking the server split anyway.
- The same question moves with the request shape. `urgent` read `label_mass`
  0.80 beside two other questions, and 0.47 beside one. That is §7.3's rule
  showing up in a single example.

---

## 4. `app/djev.py` — the one client

It is stdlib at its core, on `urllib.request`, because the callers are hot
paths and leaves. The dedupe seam sits on the `backlog_write_task` write path,
the rerank seam on every recall, and the entity sweep runs outside the backend
entirely. `app/backlog_status.py` and `app/qmd_health.py` are the precedent.
httpx appears only behind the function-local import in `ask()`.

| Function | Returns |
|---|---|
| `ask_sync(state, questions, *, timeout=15, seam="", floor=None, samples=None, instructions=None, seed=None)` | `Answers` or `None` |
| `ask(...)` | the async twin, for callers already on the event loop. A tool handler calling `ask_sync` on the loop would block it for as long as the engine takes |
| `rank(query, candidates, *, timeout, seam="rank", floor, levels)` | `[{index, score, label, confidence, label_mass, argmax_is_label, low_trust}]` best first, `[]` for no candidates, or `None`. **Raises** `ValueError` above 16 candidates |
| `rank_questions(candidates, levels)`, `rank_state(query, candidates, chars=1200)` | the question map (`c0`…`cN`) and canvas state a ranking sends, shared with the shadow seam |
| `enabled()`, `structured_url()` | the slot switch (through `llm_slots`) and `djev.structured_url` |
| `reachable(timeout=2)` | a live `GET /health`. Never called from `list_tools()` |
| `stats()` | call counts per outcome, and a 64-call latency ring per seam. Offline |

Constants: `DEFAULT_TIMEOUT_S` 15 (the server's own upstream read is 600 s and
has no 504, so this is the only bound on a wedged engine), `RANK_DEFAULT_N` 12,
`RANK_MAX_N` 16, `CANVAS_CHUNK_QUESTIONS` 32, and `RANK_LEVELS` (`irrelevant`,
`tangential`, `partly answers it`, `directly answers it`). The levels are
ordered worst to best, because `score` is an expected value over the level
*index*. Reversing the list reverses every ranking silently.

**Three-valued, like `semantic_candidates`.** `ask` and `ask_sync` return
`None` on any failure: unreachable, non-200, malformed, empty, or the slot
switched off. Never `[]`, never an exception. (`rank` adds two cases of its
own: `[]` for no candidates, and `ValueError` above 16.) A caller must be able to tell "djev had no opinion" from
"djev did not answer", because only the first is evidence. Every caller in the
tree treats `None` as "carry on unchanged".

**Every answer is normalized** into one `Answer`, whatever its type:

| Field | Meaning |
|---|---|
| `value` | P(yes) for `noul`, the option name for `choice`, the expected level for `score` |
| `label` | `yes`/`no`, the option, or the modal level's name |
| `confidence` | always the modal probability. For `noul` it is `max(p, 1-p)`, so P(yes)=0.1 is a *confident no*. For `score` it is **not** `value` |
| `probabilities` | filled for `noul` too (`{"yes": p, "no": 1-p}`) |
| `label_mass`, `argmax_is_label` | lifted out of `diagnostics` onto every answer |
| `low_trust` | `label_mass` under the schema's floor. Always False when no floor was passed |
| `uninformative` | every answer in a same-typed set carried one value |

`Answers` wraps them with `latency_ms`, `server_ms`, `prompt_tokens`,
`chunks`, `cross_chunk`, `uninformative`, `min_label_mass`, `floor` and
`seam`. `floor` is
reported so that "no answer was low-trust" and "nothing has been calibrated
yet" never read the same.

**The two trust flags are flags, never refusals.** `uninformative` exists
because a floor is structurally unable to see a degenerate ranking: the n=4
listwise run returned every score 0.0 with `label_mass` 0.987, so the mass was
legal and the answer was empty. The client returns the flags *with* the values,
and the caller decides.

There is deliberately **no `is_same()`**, and there must not be one. A helper
that answered a yes/no from a hardcoded cutoff is the exact mistake §1 rules
out.

---

## 5. The pieces around the client

### 5.1 `agent_mcp/djev.py` — three tools

All three are `READ_ONLY` in `agent_mcp/annotations.py`. A decision is one
stateless read off a seeded canvas: nothing on the machine changes, and the
engine keeps no conversation. Read-only also lets `MCPPool._retry_safe` re-send
a dropped call and lets the parallel-dispatch batch overlap it. Both are right
here, and both would be wrong for a writer.

**`djev_rank`** — `query`, `candidates` (strings) required; `levels`,
`timeout_seconds` optional.

- Each candidate is cut to 1 200 characters. A 12-candidate pool is ~4k tokens
  of state: ~1 s cold, ~70 ms warm.
- **More than 16 candidates is refused, not truncated.** A caller that hands
  over 40 rows means to rank 40, and scoring the first 16 would return a
  confident ordering of a slice nobody chose. The error says to shortlist first.
- A server-side canvas split is refused too, as "djev did not answer".
- `levels` needs at least two entries, worst first, or the default is used.
- It returns `ranked: [{rank, index, score, level, label_mass,
  argmax_is_label, candidate[:160]}]`, `min_label_mass`, a `note` that the
  scores are not calibrated, and `low_label_mass_indexes`: the indexes under
  0.5. That list is a pointer for a reader and refuses nothing (§11).

```
djev_rank(query="how do I restart the backend safely", candidates=[
  "Use round restart --only lloyd-backend; it pauses the pool and drains first.",
  "The vault is protected at the tool layer.",
  "supervisorctl restart lloyd-mc:lloyd-backend kills whatever worker job is mid-flight.",
  "Voice alerts respect quiet hours."])
→ ranked: #1 index 0 score 2.998 "directly answers it"
          #2 index 2 score 0.553 "irrelevant"
          #3 index 3 score 0.003,  #4 index 1 score 0.001     (live, 2026-09-20)
```

**`djev_decide`** — `state`, `questions` required; `instructions`, `samples`,
`timeout_seconds` optional.

- `questions` may arrive as a JSON string.
- It is validated before any call. There must be 1–32 questions. A `choice`
  needs a `criteria` object with at least two options. A `score` needs an
  ordered list of at least two levels.
- It returns `Answers.as_dict()` plus the `note`. It adds `warning` when the
  server split the canvas, and `warning_uninformative` when every answer came
  back the same.

**`djev_status`** — no required arguments (an optional `verbose` is accepted
and ignored). It returns the client's `stats()`,
`reachable`, the rank limits, the shadow recorder's counters and per-seam
switches, and every schema's `as_dict()`: hash, threshold, floor,
`gate_ready`, `gate_blocked_reason` and `drifted`.

**`list_tools()` is offline and must stay that way.** No reachability probe,
and no config read that can raise. A module that degrades makes the aggregator
answer `/health` with a 503, and `agent-services/guardian/detect.py::
mcp_degraded_is_fatal` reads that as a rollback trigger. A djev engine that is
merely *stopped* would then revert whatever landed last. Verified with the
engine unreachable: all three tools still advertise, `djev_rank` answers with
an error result, and `djev_status` reports `reachable: false`.

There is **no module `enabled` flag**, for the same reason `code_graph` has
none. An `enabled: false` that emptied `list_tools()` would break the
annotation-staleness test. The kill switch is
`mcp_servers.lloyd-mcp.disabled_tools`. `shutdown()` drains the shadow queue
for up to 5 s through the MODULES hook `main.lifespan` already runs.

### 5.2 `eval/djev/schemas.py` — the schema registry

A djev schema is not configuration. It is a frozen question shape, with what
was measured about it attached:

| Field | Meaning |
|---|---|
| `spec` | everything that decides the wording the model sees, **option order included** |
| `hash` | sha1 of `spec` serialized with `sort_keys=False`. Sorting would hide the one change most likely to invalidate a threshold |
| `threshold`, `label_mass_floor` | `None` until measured, and `None` is a real state: rank but do not gate; report `label_mass` but flag nothing |
| `calibrated_on`, `calibrated_hash` | what the numbers were measured against, and under which hash. `check()` raises `SchemaDrift` on a mismatch |
| `gate_ready`, `gate_blocked_reason` | whether the threshold may **decide** anything. A separate field on purpose: collapsing it into "has a threshold" is how "calibrated against another model's opinions" becomes "calibrated" |
| `seam` | where the decision is made in production, for the shadow recorder |

The five schemas today:

| Schema | Seam | Question | Threshold | Floor | Calibrated on | May gate |
|---|---|---|---|---|---|---|
| `rerank` | `rerank` | `score` per candidate over `RANK_LEVELS` | — (an order has no cutoff) | **unset** | — | no: not a gate |
| `dedupe` | `dedupe` | `choice`: `different` / `same` finding | 0.208 | 0.50 | 150 rows of `dedupe.jsonl`, 3 pairs per canvas: AUC 0.607 | no: barely above chance, and the corpus compares a title against a body |
| `entity` | `entity` | `choice`: `different` / `same` thing | 0.263 | 0.40 | 150 rows of `semantic-verdicts.jsonl`, 1 pair per canvas: AUC 0.942 | no: 0.562 on the 151 verified-bad merges |
| `clusters` | — | `choice`: `distinct` / `related` work | 0.217 | 0.80 | 200 rows of `cluster_judgments.jsonl`, 8 per canvas: AUC 0.699 | no: replay corpus only |
| `edges` | — | 10-way `choice` of relationship type | — (argmax) | 0.60 | 200 rows of `classified-v4-batch.jsonl`: agreement 0.580 | no: agreement is not correctness |

The negative option is listed **first** in every pair schema, because that arm
measured AUC 0.833 against 0.795 and an optimal threshold of 0.39 against 0.03.
`pair_state(a_title, a_body, b_title, b_body)` is the one canvas layout for
every pair schema, so dedupe, entity and clusters cannot drift apart in how
they present a pair. `BY_SEAM` maps a seam to its schema. A seam with no
schema still records, with an empty `schema` hash and no floor.
`tests/test_djev_schemas.py::test_every_live_seam_has_a_schema` covers only the
names in `djev_shadow.SEAMS`, so a new seam must be added there too.

**"The seam's shape" means the batch size, not the prompt.** `pair_state` is
shared, but the text around it is not. The dedupe seam appends " (Item B is
backlog #N.)" to the instruction and joins its pairs with `---`. The replay
adds `=== pair i ===` headers and " (This is pair i.)". The entity seam sends
the bare pair. The schema hash covers the spec, not these decorations, so it
cannot show the difference, and a floor measured by replay is an estimate of
the seam's distribution, not a sample of it.

### 5.3 `app/djev_shadow.py` — the recorder

`shadow(seam=…, state=…, questions=…, actual=…, meta=…)` always returns `None`,
having done one `put_nowait`. It has no return value a caller could branch on
by accident.

- **Bounded queue, drops on overflow** (`queue_max` 256). djev is
  `--max-num-seqs 1`: strictly serialized, with unbounded queueing in front of
  it. That is the old secondary's trap, where post-session jobs queued behind
  agent turns. A shadow call must never sit in front of a production decision.
  Losing an observation costs a row; blocking the recall path costs the recall.
- **The worker does the expensive part.** A seam enqueues ids and text it
  already holds. Anything that needs a disk read, like the dedupe seam's
  candidate heads, arrives as a zero-argument callable that the *worker* runs.
  `tests/test_djev_shadow.py` pins which thread it runs on.
- **Two process shapes.** In the aggregator the worker lives as long as the
  process, and `agent_mcp/djev.py::shutdown()` flushes it. In a script the
  process exits minutes after its last call, and a daemon thread takes the
  queue with it, so `entity-resolution-sweep.py` calls `flush()` in a
  `finally`. What a landing restart still loses is counted into
  `dropped_at_shutdown` on the next process's first row. The stack restarts
  several times a night, and a silent loss reads exactly like a quiet seam.
- **The floor and schema hash come from the registry, not from the seam.**
  Three seams each remembering to pass their own floor is three places for it
  to go stale after one recalibration.
- **What the bound does NOT protect.** It keeps a shadow call off the
  *caller's thread*. It cannot keep one out of the engine's queue. A shadow
  read in flight can delay a `djev_rank` tool call by up to ~1 s. That is
  acceptable only because every consumer today is advisory, so a shadow row
  can only delay other advice. **A round that makes a djev answer
  load-bearing has to revisit this**, by giving production calls a lane or by
  switching the seam off while one is in flight.
- **Muted by `LLOYD_DJEV_SHADOW=0`**, read per call. `eval/run_eval.py` sets it
  for itself, and `scripts/automod/evalpin.py::env_for` sets it for every
  pinned-corpus child: every regression arm and noise run passes through
  that one place. A replay recorded as production traffic would poison
  the `label_mass` distribution the floors are read from. The rerank arm calls
  `_vault_recall`, which is where the lead seam lives.

Rows go to `~/.local/state/lloyd-djev/shadow.jsonl`. That is outside the repo,
like the automod state dir, because a landing rewrites the tree while this file
is being appended to. One row per decision:

| Field | |
|---|---|
| `ts`, `seam`, `schema` (hash) | |
| `inputs_digest`, `n_questions` | sha1 over state + questions |
| `actual` | **what production decided**: the whole value of the row |
| `djev` | `Answers.as_dict()`, or `null` when djev did not answer |
| `meta` | the seam's context (query and scores, item name and candidates, the entity pair) |
| `queue_ms`, `worker_ms` | time waiting, and time working |
| `dropped_at_shutdown` | on a process's first row only, when the last one lost some |
| `error` | instead of `djev`, when the seam's own callable raised |

Counters (`djev_status`): `enqueued`, `dropped`, `written`, `errors`,
`skipped`, plus `queue_depth`, `worker_alive`, `log_rows` and
`pending_shutdown_drops`.

---

## 6. The seams

### 6.1 Reranking (lead) — `agent_mcp/vault.py`

The hook sits between the daily-log demote sort and the `[:limit]` slice,
**unconditionally**. At that point `documents` holds qmd's reranked pool in the
order production returns it. It records the head of the pool
(`RANK_DEFAULT_N`, 12), with `actual` = production's path order and
`meta` = the query and qmd's scores. `_vault_recall` has two production callers,
the `vault_recall` tool and `memory_ops.recall` (`agent_mcp/memory_ops.py:105`),
and both pass through it. `eval/run_eval.py` is a third, muted one. An earlier estimate of "315-378 prefetch turns a
day" had no verified caller behind it, and there is no prefetch caller. The
real daily volume will be read from the rows, not guessed.

**The first draft pointed at dead code.** It went inside `_graph_rerank`, which
is only reached under `if graph_rerank:` while `RECALL_GRAPH_RERANK` is False.
The 2026-09-04 sweep measured "off wins at every alpha". A shadow call there
would have fired zero times in production and read as a quiet seam.
`tests/test_djev_rerank_arm.py` asserts `_graph_rerank`'s source contains no
`djev` at all.

**The arm and the shadow are exclusive.** With `djev_rerank` on, djev *is* the
decision, and a row comparing djev's ordering against djev's ordering is not an
observation.

### 6.2 Backlog dedupe — `agent_mcp/backlog.py::_dedupe`

The hook sits at the `backlog_write_task` caller that holds `name` and
`description`, **not** at `merge_target`. That function is handed rows of
`{id, title, status, score, lexical, shared, created, source}` and no body text
at all, so the one thing a language model needs to answer the question is the
one thing it has never seen. It asks one `same_finding` `choice` per candidate
(the top three) in a single canvas. The candidate heads are read by the worker.
`actual` is what production did (`merged_into`, `rule`). The shadow fires for
every create that has candidates, including human, `force`, `blocker` and
`umbrella` writes that were never eligible to merge. For those, `merged_into:
null` means "not eligible", not "merge declined", and the row does not record
which it was — and unlike the first draft of this section, that bit cannot be
recovered afterwards: `meta` carries `name` and `candidates` and nothing else
(measured 2026-09-21 — every `dedupe` row in the live log has the meta key set
`('candidates', 'name')`), and the tags and `force` flag that decided
`mergeable` at `backlog.py:458-460` are never passed to the hook, though all
three are in scope at its call site. Filed as #934.

### 6.3 Entity SAME/DIFFERENT — `scripts/memory/entity_semantic_gate.py`

After the judges have spoken and before the record is written, on the
**uncached** path only. A cached verdict is not a decision being made, and
recording re-runs would weight the calibration toward whatever the sweep
re-walks most. `actual` is the gate's `SAME`/`REVIEW` and each judge's vote.

**This gate is one judge today, not two.** `default_judges()` adds the
secondary only while `resolve_model_alias("secondary")` still returns
`secondary`, and it has not since `secondary_enabled: false`. The "unanimity"
rule is a single primary vote. djev as the restored *second* judge is the
obvious follow-on round, and §7.2 is what that round has to beat.

---

## 7. What the measurements said

Reproduced 2026-09-20 with `agent-services/bin/bench-djev.py` and
`eval/djev/replay.py`. The original harness lived in a session scratchpad and
no copy survived it, which is why both now exist in the tree.

### 7.1 Performance — there is no problem to fix

| case | this 3090 | upstream GB10 |
|---|---|---|
| 3-question ticket, `samples=1` | **42.9 ms** p50 | 104.3 ms |
| state ~11.6k tok, cold | **2,686 ms** | 5,420 ms (at 8.7k) |
| state ~11.6k tok, warm | **80.7 ms** | 140 ms (at 8.7k) |
| structured-server overhead | **0.8 ms** | — |

Batching: 35.8 / 39.1 / 46.5 / 46.3 / 65.4 ms at 1 / 3 / 6 / 12 / 24 questions.
The least-squares fit is **35.5 ms fixed + 1.2 ms per extra decision**. That is
the economic fact that makes a sweep batch (8 items × 2 questions) and a clause
set (6 × 2) one request each.

Prefill is the whole cost model. 594 / 2,193 / 8,756 / 21,951 tokens cost
131 / 442 / 1,928 / 6,004 ms cold, against 44 / 51 / 73 / 122 ms warm:
**0.20–0.27 ms/token**. Throughput falls from 6,778 to 3,732 tok/s as the
quadratic term from the 5 full-attention layers bites (25 of 30 are
sliding-window-1024).

These are single-read numbers. The client's calls run under the server's
`samples: "auto"`, which takes one read or four (§3.3).

**Leave the idle clocks alone.** GPU 2 sits in P8 at 210 MHz with nothing
calling it, which costs ~25 ms on a sporadic decision: the request finishes
*before* the clocks ramp. Locking clocks would burn ~90 W continuously to save
25 ms. Refused, and this is the record of why.

The **one** action taken was a boot warm-up (§2.2).

### 7.2 Accuracy — the reason nothing gates

**Listwise capacity**, and two hard limits that are invisible without the
diagnostics:

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
produces exactly that artefact, silently. And `label_mass` collapses as N grows
while the probabilities stay renormalized, so they keep looking like confident
scores.

**Replay over the recorded corpora** (`eval/djev/replay.py`, 2026-09-20):

| corpus | n | shape | AUC | acc@0.5 | best thr → acc | what the label is |
|---|---|---|---|---|---|---|
| `entities` | 150 | 1/canvas | **0.942** | 0.827 | 0.263 → 0.893 | the gate's own SAME/REVIEW |
| `clusters` | 200 | 8/canvas | 0.699 | 0.660 | 0.217 → 0.685 | a retired 35B pair-judge |
| `dedupe` | 150 | 3/canvas | 0.607 | 0.587 | 0.208 → 0.627 | what the reranker+Jaccard rule did |
| `reverted` | 137 | 8/canvas | n/a | **0.562** | — | **151 human-verified bad merges** |
| `edges` | 200 | 8/canvas | agreement **0.580** vs a 0.310 majority floor | | | a v4 classifier prompt |

Four things to take from that table:

- **Only `reverted` is ground truth.** Everything else measures agreement with
  another model, and a 0.94 AUC against a judge is not accuracy.
- **The threshold measured on ordinary pairs does not transfer to the hard
  ones.** On the 137 definition-carrying pairs of the 151 verified-bad merges,
  `acc@0.5` was 0.562, and **50 of 137 scored above 0.9** while `label_mass`
  stayed healthy (p50 0.991). Confidently wrong, not confused. Those 151 are a
  hard-negative set *by construction*, being exactly the cases the old rule got
  wrong. So 0.942 against a judge and 0.562 against the reverted set are
  consistent, and the second is the one a gate would meet. This is why the
  entity seam ships as shadow and `gate_ready` is False.
- **A replay of a definition-less pair measures name shape, not djev.** The
  first run of `reverted` scored 0.364 with **zero** of 302 entity definitions
  loaded: the replay had its own private definition reader, and it parsed none
  of the 288 overviews the gate's reader parses. It now imports
  `entity_semantic_gate.entity_definition` and skips the 14 pairs that still
  have no definition, reporting the count. That is the gate's own rule, for the
  gate's own reason.
- **`clusters` fell from AUC 0.833 on 40 pairs to 0.699 on 200.** The larger
  sample is the truer one. Its reliability curve is plainly miscalibrated: the
  0.0–0.1 bin observed 0.338 and the 0.9–1.0 bin observed 0.690.

**`edges` is the largest prize and the one to take next.** It is 1,183–3,817
decisions a day, at two sequential primary calls each today. djev nearly
doubles the majority-class floor at 34 ms a decision, and 35,802 labelled rows
already exist. What is missing is a human-checked subset: 0.580 agreement with
the v4 prompt is agreement, not correctness.

### 7.3 The floors, and why one of them is unset

`label_mass` floors are **per schema and per request shape**. This was
measured, not argued:

| schema | shape | min | p01 | p05 | p50 | floor |
|---|---|---|---|---|---|---|
| `entity` | 1 pair / canvas | 0.454 | 0.508 | 0.620 | 0.796 | **0.40** |
| `entity` | 8 pairs / canvas | 0.053 | 0.166 | — | 0.995 | — |
| `dedupe` | 3 pairs / canvas | 0.562 | 0.694 | 0.815 | 0.934 | **0.50** |
| `clusters` | 8 pairs / canvas | 0.878 | 0.930 | 0.967 | 0.993 | **0.80** |
| `edges` | 8 / canvas | 0.680 | 0.908 | 0.965 | 0.997 | **0.60** |
| `rerank` | 12–16 / canvas | — | — | — | — | **unset** |

The two `entity` rows are the same schema, the same corpus and the same engine.
**The request shape alone moved the distribution by an order of magnitude.** A
floor taken from the batch-8 run would never fire on the seam, and a
"reasonable" 0.5 would flag 1% of perfectly healthy batch-1 reads and 40% of
batch-8 ones. Each floor here sits just under everything observed healthy for
its own shape, which is what a flag that never refuses should mean: "outside
the distribution we measured", not "in the bottom 5% of normal traffic". The
entity floor fires on 0% of its own corpus.

**`rerank` stays unset, and that is the design's own rule holding.** Listwise
`label_mass` at n=12–16 has measured 0.446, 0.807, 0.965 and 1.000 across four
runs on different corpora: a fourfold spread inside the window the design calls
safe. No floor taken from one of them means anything for the others. The shadow
rows are the seam's own requests on the seam's own corpus, so they set this
one. `eval/djev/replay.py --floors` prints the distribution as it accumulates.
Read §11 before trusting it.

---

## 8. Adoption: the arm, not the log

`RECALL_DJEV_RERANK = False` in `agent_mcp/vault.py`, with
`RECALL_DJEV_RERANK_TOP = 12`. `vault_recall` also takes `djev_rerank` and
`djev_rerank_top` per call. `eval/run_eval.py --djev-rerank` passes the flag,
and `build_run_config` records `matches_production_defaults: false` for such a
run, so a djev-reranked baseline never compares as production's by accident.
With the arm on, only the head of the pool is reordered, and the tail keeps
qmd's order.

**An eval arm is code in the handler, not a log.** The adoption question is
"does djev's ordering beat qmd's own reranker on the labelled set", and a
shadow record has no labels. Flipping the constant is the adoption decision.
It is a separate commit, with the eval result in its message.

**Measured 2026-09-20**, six runs across two pinned corpora. Every arm in a run
shares one frozen qmd index, so the flag is the only difference left:

| arm | MRR | NDCG@10 | latency avg |
|---|---|---|---|
| baseline | 0.511 | 0.608 | 4,591 ms |
| baseline (repeat) | 0.511 | 0.607 | 4,907 ms |
| `--djev-rerank` (top 12) | **0.605** | **0.697** | 4,841 ms |
| `--djev-rerank` (top 12, repeat) | 0.596 | 0.686 | 4,925 ms |
| `--djev-rerank` (top 12, third) | 0.605 | 0.698 | 5,448 ms |
| `--djev-rerank-top 8` | 0.603 | 0.682 | **4,797 ms** |

**+0.085 to +0.094 MRR, three times out of three**, against a baseline that is
deterministic on a pinned index (0.511 both times). So the spread in the arm is
djev's own, and it is small. `doc_hit_rate` stays 1.000 and `entity_hit_rate`
0.650 in every arm. That is the check that the arm reorders the pool rather
than changing what is in it.

**Top-8 is the one to reach for if the budget bites.** It buys the same MRR
(0.603) and lands at 4,797 ms against the nightly's 4,800 ms ceiling, where
top-12 runs 41–648 ms over. The plan anticipated exactly this and called it a
result rather than a tuning, which it is.

**Re-measured at n=87, 2026-09-21 (#1335): not adopted.** The gold set grew to
87 queries and the recall moved to global fusion over a 40-row pool, so the table
above describes neither. One pinned snapshot, two runs per arm, paired per query
against a baseline that repeated exactly (MRR 0.191):

| arm | MRR | paired ΔMRR, 95% interval | better / worse | latency avg |
|---|---|---|---|---|
| baseline | 0.191 | — | — | 1,757–1,782 ms |
| `--djev-rerank` (top 12) | 0.246 / 0.244 | +0.055 [−0.002, +0.112] / +0.053 [−0.001, +0.107] | 15/8, 16/7 | 2,300–2,315 ms |
| `--djev-rerank-top 8` | 0.241 / 0.239 | +0.050 [−0.005, +0.107] / +0.048 [−0.007, +0.104] | 13/9, 13/10 | 2,194–2,279 ms |

0 djev misses in 696 calls, 0 qmd rerank fallbacks. Consistent in sign and half
the n=20 step, never clear of zero, and ~0.5 s slower. A change that is slower
has to show a gain, so `RECALL_DJEV_RERANK` stays off. A djev that *replaces*
the cross-encoder only has to match it, which is #1336.

**Carry the caveat anyway.** The labelled set is 20 queries and 50 doc labels,
and `agent_mcp/vault.py:172` puts the noise floor at 0.02 MRR. Reproducing +0.09
three times against a deterministic baseline is much stronger than one run, but
it is still twenty questions. The absolute latencies above were taken on a box
that was also running the test suite. The comparison between arms is fair,
because both paid that cost; the comparison with the ceiling is not.
`latency_ms_avg` is reported, never gated. Quality is what the paired promotion
check compares.

The inverse-cloze result the arm exists to re-test properly: djev MRR 0.766 /
recall@1 0.64 over 16 candidates, against 0.498 / 0.29 for lexical Jaccard and
0.146 / 0.00 for random, at 662 ms for 16. **Jaccard is a weak baseline**, and
that comparison decides nothing.

### 8.1 djev ranks the recall (#1336, 2026-09-21)

The arm above puts djev on top of qmd's cross-encoder, which is slower and has
to show a gain. What landed instead **replaces** the cross-encoder, which only
has to match it (Alan: equal accuracy plus throughput is a win).

- **The pool is the ceiling, not djev.** No ~32-row pool holds as many findable
  documents as collection-240 did (phase 0 on #1336). One 128-token canvas holds
  32 rank rows, so djev ranks at most 32 (`rank(max_n=32)`, capped at
  `CANVAS_CHUNK_QUESTIONS`).
- **On the same ≤32 rows, djev beat qmd's cross-encoder** (87-query pinned eval):
  doc_hit 0.471 vs 0.425 (+0.046 [+0.011, +0.092]), NDCG@10 0.274 vs 0.214
  (+0.060 [+0.004, +0.114]), 0.51 s vs 1.04 s per recall. Full-length
  candidates ranked worse than 160 chars, and `samples: "auto"` was 200 ms slower
  and worse than one read. A repeat of one read gave the same recall outcome on
  87/87 queries, which is not the same thing as a deterministic read: the
  scores behind it do not repeat (§8.2).
- **Deployed shape:** global fusion's 20-row head + floors 2/2/2 for autonomy,
  architecture and skills, cross-encoder off, `rank(chars=160, samples=1,
  max_n=32, timeout=4)`, seam `recall_rank`. Against the cross-encoder path on
  one pin: doc_hit 0.517 vs 0.494, MRR 0.256 vs 0.201, NDCG 0.275 vs 0.234,
  doc_recall 0.357 vs 0.361. Every paired interval includes zero, and p50 is
  0.52 s vs 2.18 s.
- **Fail open, to the better answer.** `None` or a raise from `rank` re-runs
  the recall on the cross-encoder path rather than serving fusion order (0.05
  MRR worse). `app/qmd_health.py::note_ranker` counts it, logs it and rings
  `announce()` once per 30 minutes. The shadow seam does not run when djev
  ranks, nor inside that fallback.
- **The line holds:** this orders documents and gates nothing. `RANK_MAX_N`
  stays 16 for every other caller.

### 8.2 djev does not repeat itself (2026-09-21)

Identical requests get different scores. Replayed straight at vLLM `:8010` with
the structured server's own body for one 32-row recall rank:

- the argmax at all 123 canvas positions was identical on every run, and the
  seeded canvas came back the same (91 of 123 positions reproduce it, every run);
- the label logprobs the rank score is built from moved by **1-3 nats** between
  identical requests (slot 0, second label: -6.96, -5.67, -5.81). The top pick
  holds; ranks 3 and below shuffle;
- **ruled out, by booting djev by hand with one change each:** the prefix cache
  (a fresh `cache_salt` per request varies as much), CUDA graphs and compile
  (`--enforce-eager` varies), async scheduling (removed, still varies), and
  sampling (the read-only path draws nothing; one denoise step);
- **left:** the kernels, the MARLIN NvFp4 MoE backend (weight-only FP4 on the
  3090) or TRITON_ATTN. The bisect is #1357, and it needs djev down for about
  two minutes a variant.

For a recall this costs little: the top result is stable. For a paired eval it
is fatal, because two arms running the same code ask djev the same questions
and get different answers. On 2026-09-21 the regression check rolled back two
promotions that touched no retrieval code, for one query's worth of doc_hit.
So `app/djev.py` has a **replay**, used only by evals. One comparison's arms
share an sqlite file (`LLOYD_DJEV_REPLAY`); a request the anchor arm already
asked is answered from its answer, a request only this arm asks is drawn fresh
and counted, and a failure is counted and never stored. The regression check
reads those counts to pick its noise floor and to refuse an arm djev did not
answer (`architecture/automod.md` §8.1b). Production never sets the variable.

---

## 9. Using it

### 9.1 From a turn

- **`djev_rank` is a final stage over a shortlist**, never the retrieval
  itself. Shortlist with `vault_search`, `recall` or a grep, then rank the best
  12. Use the *order*. Do not read meaning into a score's absolute value, or
  compare scores from two different calls.
- **`djev_decide` is for classifying and ordering one piece of text** when the
  primary would otherwise spend a turn on it: which of N buckets, how severe,
  does it mention X. Batch the questions into one call; the 25th question
  costs ~1 ms. Keep the option order fixed across calls you mean to compare.
  Never act on `value > 0.5`.
- Read `label_mass` before trusting an answer, and treat
  `warning`/`warning_uninformative` as "no answer".
- An error result means djev did not answer. Carry on with what you had.

### 9.2 From code

```python
from app import djev

out = djev.ask_sync(state, questions, seam="my-seam", floor=None)   # or: await djev.ask(...)
if out is None:
    ...                              # djev did not answer: carry on unchanged
elif out.cross_chunk or out.uninformative:
    ...                              # not an answer you can use
else:
    a = out["severity"]              # Answer: value, label, confidence, label_mass, low_trust

rows = djev.rank(query, shortlist)   # ≤16 candidates, else ValueError; None on failure
```

- Name a `seam` on every call. It is how `djev_status` reports latency per
  caller.
- On the event loop, use `ask`, or `asyncio.to_thread(djev.rank, …)` as the
  tool does. Never call `ask_sync` on the loop.
- Question shapes that a result depends on belong in `eval/djev/schemas.py`,
  not in a literal at the call site. Copy `criteria` with `dict(...)` so
  option order survives.

### 9.3 Adding a seam

1. Freeze the question as a `Schema` in `eval/djev/schemas.py`, with `seam`
   set. It starts with `threshold=None` and `label_mass_floor=None`.
2. Add the seam name to `djev_shadow.SEAMS`, and give it a line under
   `djev.shadow.seams` in config.yaml. A seam absent from that map is **on**.
3. Call `djev_shadow.shadow(...)` **where production's decision is made**, with
   `actual` = what production decided. Pass anything that needs I/O as a
   callable. Wrap the call in `try/except: pass`: a recorder may never reach
   its caller.
4. If a replay corpus exists, calibrate against it **at the seam's own request
   shape**: `replay.py --corpus <c> --batch <decisions per canvas the seam
   sends>`. Write the threshold, the floor, `calibrated_on` and
   `calibrated_hash` into the schema by hand, as the existing five do.
5. Let the shadow rows accumulate, then set the floor from them
   (`replay.py --floors`) — **after** quarantining the fixture rows still in
   the log from before #1324 landed (§11), or the floor is a floor set from
   tests. Once it is live, a run appends nothing: `wc -l` on the log across a
   full suite moves only with production traffic.

### 9.4 Turning a schema into a gate

No schema is `gate_ready` today, and flipping one is its own round. Setting
the flag is the last step, after all of these hold:

- a threshold measured **at the seam's request shape** against labels that
  are ground truth, not another model's verdicts, including the hard cases
  (for entities: the 151 verified-bad merges);
- the schema's hash frozen, with `check()` passing;
- an answer to engine contention (§5.3): a production lane, or the seam's
  shadow switched off while a gated call is in flight;
- the eval result in the commit message that flips it.

### 9.5 Switches

| To turn off | Do |
|---|---|
| everything (engine, client, seams, card) | `djev.enabled: false`, then `round restart` (backend and aggregator) |
| one seam's recording | `djev.shadow.seams.<seam>: false` |
| all recording | `djev.shadow.enabled: false` |
| recording in one process | `LLOYD_DJEV_SHADOW=0` |
| the tools | `mcp_servers.lloyd-mcp.disabled_tools: [djev_rank, …]` |
| the rerank arm | it is off: `RECALL_DJEV_RERANK = False` |

---

## 10. Running and checking it

```bash
curl -s http://127.0.0.1:8011/health                       # {"status": "ok"}
~/.local/share/uv/tools/supervisor/bin/supervisorctl -c agent-services/supervisor/supervisord.conf status agent-djev
tail -f agent-services/logs/agent-djev.log

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

`djev_status` from a turn gives the same picture from inside the aggregator.

`_pipeline/` is gitignored derived data, so a clone or a worktree has an empty
one, and every corpus reads as "no rows". That is indistinguishable from a
corpus that ran out. Set `LLOYD_DJEV_CORPUS_ROOT` to the live
`_pipeline/memory-graph` directory. It moves the `entities`, `reverted` and
`edges` corpora. `dedupe` and `clusters` always read
`~/.local/state/lloyd-automod`, and entity definitions come from
`VAULT_FACTS_ROOT`/`LLOYD_FACTS_ROOT`. A run that finds nothing prints where it
looked.

The bench refuses a busy engine through `vllm_metrics.wait_idle`, the one
definition of idle. djev serves one sequence at a time, so a neighbour does not
merely add noise. It serializes in front of every read, and the whole table
shifts.

Tests: `tests/test_djev_client.py`, `test_djev_tools.py`,
`test_djev_schemas.py`, `test_djev_shadow.py`, `test_djev_rerank_arm.py`.

| Path | Holds |
|---|---|
| `app/djev.py` | the client |
| `app/djev_shadow.py` | the shadow recorder |
| `app/llm_slots.py` | the slot switch shared with the secondary |
| `agent_mcp/djev.py` | the three tools |
| `agent_mcp/vault.py`, `agent_mcp/backlog.py`, `scripts/memory/entity_semantic_gate.py` | the three seams |
| `scripts/service_health_check.py` | `SERVICES["agent-djev"]` declares `"port": 8011`, so a supervisor `RUNNING` line with no listener behind it shows up as a red probe, and `_switched_off()` reads `llm_slots.slots()` so a `djev.enabled: false` verdict is not an outage |
| `eval/djev/schemas.py`, `eval/djev/replay.py` | the registry, and calibration |
| `agent-services/bin/start-djev.sh`, `agent-services/setup/setup-djev.sh`, `agent-services/bin/bench-djev.py` | serving, setup, the bench |
| `agent-services/supervisor/conf.d/agent-djev.conf` | the program and every tuning knob |
| `~/.local/state/lloyd-djev/` | `shadow.jsonl`, `dropped_at_shutdown.json` |

---

## 11. Known gaps (2026-09-20)

- **`SchemaDrift`'s message names `replay.py --calibrate`, which does not
  exist.** Calibration is `--corpus <c> --batch <n>` and a hand edit (§9.3).
- **`djev_rank`'s `low_label_mass_indexes` uses a fixed 0.5.** It refuses
  nothing, but it is the one fixed cutoff in the tree, inside the window
  where healthy n=16 reads measured 0.446.
- **The client's default is `samples: "auto"`**, one read or four. The ~40 ms
  headline is `samples=1`. No seam has measured whether one read is enough for
  its schema. The `djev_decide` input schema tells the model `samples`
  defaults to 1, which is wrong: omitted, it is `"auto"`.
- **Engine contention** between shadow reads and tool calls (§5.3). It is
  harmless while every consumer is advisory.
- **Scores do not repeat across identical requests** (§8.2, #1357). Evals
  that compare arms must replay (`app.djev.replay_env`), or they measure djev's
  noise as the change.

### Closed 2026-09-21 — test runs wrote to production's shadow log (#1324)

**What it was.** `STATE_DIR`, `SHADOW_LOG` and `PENDING_DROPS` are
`Path.home()` literals bound at import (`app/djev_shadow.py:68-71`) and
`_write()` opens `SHADOW_LOG` on the recorder's daemon thread at *write* time
(`:295`), so a per-test patch could not hold: `tests/test_djev_shadow.py`
patched the three names and a queued job still drained after its `monkeypatch`
tore down, into the corpus §9.3 step 5 reads. `tests/test_backlog_dedupe.py`
and `test_backlog_spawn_loop.py` reached the seam with no shadow isolation at
all — `_djev_shadow_dedupe` gates on the lexical `similar[:3]`, which
conftest's `_isolate_backlog_dedupe` keeps supplying. Re-measured 2026-09-21
08:19Z over the live log, 96 rows spanning 01:55:15Z→08:19:15Z: 36 of its 44
`dedupe` rows carry one of the two fixture titles (32 ×
"http_fetch error body says only the status code on a quarter of calls" at
`tests/test_backlog_dedupe.py:58`, 4 × "graph_refresh is advertised but never
called by any tool" at `tests/test_backlog_spawn_loop.py:792`), and 27 of its
32 `rerank` rows carry an empty `meta` — the shape of `test_djev_shadow.py`'s
direct `shadow(seam="rerank", state="s", questions={"q": {}})` calls. 63 fixture
rows in a log six hours old, and `_write()` only appends: nothing rotates them
and nothing decays.

**What closes it.** `tests/conftest.py::_isolate_djev_shadow`, one
session-scoped `autouse` fixture that moves all three globals into the session
tmp dir and **does not restore them**. The first cut restored them at session
teardown and still left a row in `$HOME/.local/state/lloyd-djev/shadow.jsonl`:
neither backlog module calls `flush()`, so a read in flight when pytest finishes
lands wherever the module points *then*, and the restore put the home path back
at exactly that moment. Session scope is also what the per-test patch needed —
`test_djev_shadow.py`'s `_isolated` now restores *to* the session dir, so a late
drain has nowhere production to go. Paths are redirected rather than
`LLOYD_DJEV_SHADOW=0` set, because
`test_djev_rerank_arm.py::test_the_eval_mutes_the_shadow_recorder` reads that
variable out of the process to prove `eval/run_eval.py:35` set it; an ambient
mute would make that assertion hold with the line under test deleted.

**The check.** `FH=$(mktemp -d); HOME=$FH .venvs/lloyd/bin/python -m pytest
tests/test_backlog_dedupe.py tests/test_backlog_spawn_loop.py -q` → 72 passed,
and `$FH/.local/state/lloyd-djev/shadow.jsonl` **does not exist**. That command
is `tests/test_djev_shadow_isolation.py::test_a_child_pytest_run_leaves_no_shadow_log_in_its_home`,
which failed on the base commit naming `['dedupe', 'dedupe']` in the file. The
pin runs one extra node — this file's own create test, which calls `flush()` —
so a recorder that recorded nothing anywhere could not pass it, and then
requires a `dedupe` row naming that fixture to exist somewhere in the child's
tree and not under its `HOME`: the positive control is the parsed row, not a
directory that `mktemp` made anyway. Its three siblings pin the three globals
together, the seam driven by a real `backlog_write_task` create, and the drain
race at its mechanism; `tests/test_djev_doc_claims.py` pins this paragraph.
`HOME` is redirected rather than the path patched because the globals
bind at import — a patch cannot imitate a fresh process, and a fresh process is
what a test run is. `scripts/automod/gate.py::_child_env` hands its pytest the
real `HOME` (`grep -n DJEV scripts/automod/gate.py` → 0 hits), so this fixture,
not the gate's environment, is what covers the suite every promotion runs — and
what the gate's own suite cannot show is the real log's line count across a full
run, which is why that count is a person's post-landing check.

**What is left, and it is a person's call.** The 63 fixture rows counted at
08:19Z — 36 `dedupe` under the two titles above plus 27 `rerank` with an empty
`meta` — stay in the log; a code round does not edit a live state file. The
count is a snapshot and rises by a couple of rows every time anyone falsifies
this pin by removing the fixture, so quarantine by the RULE (drop `rerank` rows
with an empty `meta`, and `dedupe` rows whose `meta.name` is one of those two
titles), never by taking the last 63. Until that happens any floor read by
`replay.py --floors` is fixture-dominated. Once this is live,
`wc -l ~/.local/state/lloyd-djev/shadow.jsonl` across a full-suite run should
move only with production traffic.

---

## 12. Rules, short form

- A fixed 0.5 threshold is meaningless. Calibrate per schema, and freeze the
  schema, option order included, once calibrated.
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
- djev orders the vault recall (#1336); qmd's cross-encoder is its fallback,
  and the fallback is counted, never silent.
- djev's scores do not repeat. An eval that compares two arms replays its
  answers per request (§8.2), and never takes djev down while a regression
  check holds `regression.lock`.

## Review log

- **2026-09-21 — #1324 closed, and the entry below is superseded on this
  point.** It recorded "§11's fixture counts … with the leak still live
  (#1324)"; §11 now carries that leak under a Closed subsection instead, and
  `tests/conftest.py::_isolate_djev_shadow` is what closed it — one
  session-scoped autouse fixture that moves `STATE_DIR`, `SHADOW_LOG` and
  `PENDING_DROPS` into the session tmp dir and never restores them. Measured on
  this change: `HOME=$FH pytest tests/test_backlog_dedupe.py
  tests/test_backlog_spawn_loop.py` → 72 passed and no
  `$FH/.local/state/lloyd-djev/shadow.jsonl`; the rows are still recorded, but
  in the session shadow dir under pytest's basetemp (under `$FH` only when
  `TMPDIR` says so), which is the whole point — and the gate-shaped full suite
  (7,768 passed, 15 skipped, 2 xfailed)
  left the real log at 96 rows before and after. What that does NOT settle is
  in §11: the 63 fixture rows counted at 08:19Z are still in the file until
  someone quarantines them by rule, so a floor read by `replay.py --floors` is
  still fixture-dominated.

- **2026-09-21 — `current`.** Checked every path, port, constant, config key and
  measured table against HEAD `485fa6c03e31`: the engine, the three tools, the
  five-schema registry with its three thresholds and four floors, the three
  seams and their
  guards, the kill switch through `llm_slots`, and the 4,800 ms nightly ceiling
  (`workers/sources/automod_regression.py:231`) all still describe what runs.
  Two things changed: §6.2 had claimed a calibration can recover merge
  eligibility from the tags in `meta`, which is false — no tags are written
  (#934) — and §11's fixture counts are re-measured at 62 of 91 rows with the
  leak still live (#1324). `scripts/service_health_check.py` is added to the
  consumer table. The 20-query caveat on the §8 arm table is now also a stale
  denominator, since the gold set grew to 87 queries in `98fa216b` (#1319).
