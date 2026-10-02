# Lloyd

A voice-first personal AI agent that runs entirely on your own hardware.

Lloyd runs its own agent loop against a local vLLM server, exposes every tool
through a single MCP aggregator, keeps its long-term memory in an Obsidian
vault, and rewrites its own code behind a gate and a rollback watchdog. No
model inference leaves the machine — there is no Anthropic, OpenAI, or other
inference API in the loop. (Lloyd can still reach the internet when you ask it
to search or browse; it is the *thinking* that stays local, not the network.)

> **This is one person's system, not a product.** It is built for, and pinned
> to, a specific machine — Arch Linux, three NVIDIA GPUs, ~311 GB of local model
> weights. It is published because the design is worth reading and the pieces
> are worth stealing, not because it will `git clone && run` on your laptop.
> [SETUP.md](SETUP.md) is honest about exactly how much is involved.

---

## How it fits together

```
        browser  ·  voice  ·  Chrome side panel  ·  Discord
                          │
                          ▼
        ┌─────────────────────────────────┐      ┌──────────────────────────┐
        │  FastAPI backend  :8080         │◄─────│  worker pool (in-process)│
        │  chat turns (SSE), dashboard    │      │  autonomy · research ·   │
        └───────────────┬─────────────────┘      │  automod rounds          │
                        │                        └──────────────────────────┘
                        ▼
        ┌─────────────────────────────────┐
        │  agent harness (app/harness/)   │
        │  stream → tool dispatch → loop  │
        └───────┬─────────────────┬───────┘
                │                 │
                ▼                 ▼
    ┌───────────────────┐   ┌──────────────────────────┐
    │  vLLM  :8096      │   │  lloyd-mcp  :8500/mcp    │
    │  primary, GPU 1   │   │  every tool, one server  │
    └───────────────────┘   └──────────┬───────────────┘
                                       │
              ┌────────────────┬───────┴────────┬─────────────────┐
              ▼                ▼                ▼                 ▼
       Obsidian vault     qmd  :8181       djev, GPU 2       Bash / files /
       (~/obsidian)       (vault search)   (ranks recall)    browser / desktop

    guardian (its own systemd unit, outside all of the above):
    watches the stack, performs rollbacks, guards the vault and the data home
```

Each turn rebuilds the full conversation from the persisted session JSON and
sends it to vLLM as an OpenAI-format `messages` list. There is no `resume=` —
history is reconstructed every time, which is what makes compaction and editing
past turns tractable.

Code lives in `~/lloyd`; everything Lloyd *produces* (sessions, databases,
logs) lives in `~/lloyd-data`, outside the tree.

---

## Architecture

One doc per area under [architecture/](architecture/), each describing what
actually runs. [architecture/index.md](architecture/index.md) is the map;
the tl;drs below are the short way in.

### Runtime

| Doc | tl;dr |
|---|---|
| [infrastructure](architecture/infrastructure.md) | Everything runs on the host under one supervisord (a systemd `--user` unit) — no containers. Host, GPUs, model slots, ports, alerts, remote access. |
| [vllm](architecture/vllm.md) | The primary engine end to end: served config, FP8 KV cache, the GPU's power clamp, tuning knobs and the benchmarks behind them. |
| [djev](architecture/djev.md) | A diffusion model on GPU 2 that answers typed questions (yes/no, one-of-N, score). It ranks; nothing gates on it. |

### The agent

| Doc | tl;dr |
|---|---|
| [harness](architecture/harness.md) | `run_query(messages, options)`: one async generator that streams a turn, dispatches tool calls, and yields normalised events to chat, workers and autonomy alike. |
| [context-window](architecture/context-window.md) | Four layers fit a session to the model's window — microcompaction, a persisted summary, LLM summarisation, truncation — and what each costs in prefix-cache misses. |
| [tools](architecture/tools.md) | The lloyd-mcp aggregator: one `Server("lloyd")` mounting every tool (built-in Bash/Read/Write/Edit included) as a flat namespace; the dispatch path in order. |
| [skills](architecture/skills.md) | On-demand procedures: plain `SKILL.md` files in the vault that Lloyd reads before executing. |
| [subliminal](architecture/subliminal.md) | Context retrieved and injected before every LLM call — skills, facts, vault docs, sessions, backlog — while the system prompt stays byte-stable. |
| [ambient-context-injection](architecture/ambient-context-injection.md) | How background producers surface signals into the user's active chat. |
| [inner-voice](architecture/inner-voice.md) | Deterministic turn guards on every turn, plus an opt-in LLM second reader at the end of one. |
| [voice](architecture/voice.md) | The whole spoken round trip: LiveKit, wake word, VAD, end-of-turn, ASR, speaker ID, streaming cloned-voice TTS, barge-in. |
| [desktop](architecture/desktop.md) | Desktop computer use: capture freely, act only on a lease a human grants. |

### Safety

| Doc | tl;dr |
|---|---|
| [authority-surfaces](architecture/authority-surfaces.md) | Every guard in one list, in the order an action meets it. |
| [guard-coverage](architecture/guard-coverage.md) | The union of what no guard sees, each gap with its mechanism and the command that proves it. |
| [editing-safeguards](architecture/editing-safeguards.md) | What stands between a turn and the tree: read-before-edit, edit diagnostics, the change and effect ledgers, the code graph. |
| [vault-protection](architecture/vault-protection.md) | Two vault wipes and the four layers built after them: tool sandbox, wholesale-delete refusal, guardian tripwire, 15-minute snapshots. |
| [data-home](architecture/data-home.md) | Runtime data lives in `~/lloyd-data`, never in the code tree: the one path resolver, hourly read-only snapshots, restore. |

### Memory and knowledge

| Doc | tl;dr |
|---|---|
| [memory](architecture/memory.md) | The four-verb memory tool surface and the loop that improves stored facts. |
| [knowledge-graph](architecture/knowledge-graph.md) | Facts as human-readable markdown; edges, aliases and entities in one SQLite store with a single writer. |
| [qmd](architecture/qmd.md) | The vault search engine: Lloyd's fork, run as a daemon on GPU 0 — BM25 plus embeddings, fused. |
| [retrieval](architecture/retrieval.md) | How `vault_recall` finds notes: qmd builds the candidate pool, djev orders it; every change measured against a gold set. |
| [research-pipeline](architecture/research-pipeline.md) | The topic registry that turns a question Lloyd cannot answer into a note in the vault. |

### Unattended work

| Doc | tl;dr |
|---|---|
| [workers](architecture/workers.md) | One SQLite queue drained by asyncio workers inside the backend; everything Lloyd does unasked runs through it. |
| [workers-jobs](architecture/workers-jobs.md) | One entry per worker source: what wakes it, what it writes, what it is actually doing. |
| [background-runs](architecture/background-runs.md) | Every background run leaves the same record a chat turn does; Inner Voice observation is a separate opt-in. |
| [autonomy](architecture/autonomy.md) | The scheduled-task mechanism (recurring work only): due-gates, failure backoff, the deadline anchor. |
| [autonomy-jobs](architecture/autonomy-jobs.md) | What each scheduled job is *for* — reflection, skill mining, graph upkeep, vault hygiene, inbound signal. |
| [backlog](architecture/backlog.md) | The markdown kanban in the vault that people, tools and the self-modification loop all read and write. |
| [automod](architecture/automod.md) | Self-modification: a worktree per round, a gate ladder, a review rung, a promoter, and a guardian that rolls back. |
| [testing](architecture/testing.md) | The test suite: synthetic vs live-data tests, parallel isolation, and why it never runs against production. |
| [measurement](architecture/measurement.md) | The eval and bench surface: where every number Lloyd quotes about itself comes from. |
| [arch-review](architecture/arch-review.md) | The worker pass that keeps these docs honest, one doc per session. |

### Front end

| Doc | tl;dr |
|---|---|
| [mission-control](architecture/mission-control.md) | The React + Vite UI: chat, sessions, one polled dashboard endpoint whose sections degrade independently. |
| [browser-side-panel](architecture/browser-side-panel.md) | A Chrome side panel with one Lloyd session per tab and page, created only on request. |

### Where the code is

| Path | Holds |
|---|---|
| [server.py](server.py), [app/routers/](app/routers/) | FastAPI backend |
| [app/harness/](app/harness/) | the agent loop |
| [agent_mcp/](agent_mcp/) | the MCP aggregator and every tool |
| [workers/](workers/) | the job queue and its sources |
| [scripts/automod/](scripts/automod/) | the self-modification loop |
| [agent-services/](agent-services/) | supervisord confs, launchers, the voice worker, the guardian |
| [web/](web/), [chrome-extension/](chrome-extension/) | Mission Control and the side panel |
| [eval/](eval/), [tests/](tests/) | benches and the test suite |

---

## Models and hardware

Bus order matters; every GPU program pins `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

| GPU | Card | Runs |
|---|---|---|
| 0 | RTX 3090 (24 GB) | Qwen3-TTS (`:8090`), the voice worker, qmd embeddings |
| 1 | RTX PRO 6000 Blackwell (96 GB) | the primary LLM: Qwen3.8-Flash-Next NVFP4 on vLLM (`:8096`) |
| 2 | RTX 3090 (24 GB) | djev: DiffusionGemma, structured decisions and recall ranking — not a chat slot |

The primary serves a 262 k context with an FP8 KV cache and multi-token
speculative decoding, launched with `--enable-auto-tool-choice
--tool-call-parser qwen3_xml --reasoning-parser qwen3`. Every chat turn, worker
job and subagent runs on it. A llama.cpp secondary slot (`:8091`) exists but
has been off since 2026-09-20; its jobs moved to the primary. The voice
worker's own models (Silero VAD, openWakeWord, Smart Turn, Parakeet, CAM++)
are CPU ONNX.

Details: [infrastructure](architecture/infrastructure.md) § GPUs and § Model
slots, [vllm](architecture/vllm.md), [djev](architecture/djev.md).

---

## Running it

Everything runs directly on the host under supervisord, installed as the
`agent-supervisord.service` systemd `--user` unit. There is no container.

```bash
alias lsup='/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl \
  -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf'

lsup status
lsup tail -f agent-llm-primary stderr
```

**Restart the backend, the aggregator or the engine with the round CLI, not a
bare `supervisorctl restart`:**

```bash
.venvs/lloyd/bin/python -m scripts.automod.round restart --reason "why"
.venvs/lloyd/bin/python -m scripts.automod.round restart --only lloyd-backend
```

It pauses and drains the worker pool, waits for idle, restarts with a health
wait, and records the restart — a bare restart reads to the guardian as a
crash and kills whatever worker job is mid-flight. Frontend edits are picked
up by Vite HMR without a restart.

Mission Control is at **https://localhost:5173** (Vite terminates TLS with a
local CA and proxies `/api` to the backend).

---

## Configuration

[config.yaml](config.yaml) holds models, MCP servers, voice, autonomy, worker
sources, and agent settings. It is **read-only at boot** — UI toggles persist to
`~/lloyd-data/data/tool_overrides.yaml` and are merged over it. If a
hand-edited change to `config.yaml` doesn't take, check that the override file
isn't shadowing the same key.

Secrets live in `.env` (gitignored) and reach `config.yaml` through `${VAR}`
placeholders expanded at boot. **Never put a literal secret in `config.yaml`** —
it is tracked. See [.env.example](.env.example).

Tools are disabled either server-wide (`mcp_servers.<name>.enabled: false`) or
individually (`mcp_servers.<name>.disabled_tools: [Bash, ...]`, using bare tool
names).

---

## Setup

**[SETUP.md](SETUP.md)** is the authority for a rebuild from a fresh OS: system
packages, the uv/bun/npm toolchain, all five venvs, supervisord and the systemd
unit, model downloads, and — importantly — **what to back up before you wipe**,
since several runtime assets are untracked and not re-downloadable.

```bash
agent-services/setup/setup-all.sh --check   # reports what's missing, changes nothing
```

---

## More documentation

- **[SETUP.md](SETUP.md)** — full bare-metal rebuild
- **[CLAUDE.md](CLAUDE.md)** — the rules to know before touching an area, for coding agents
- **[agent-services/README.md](agent-services/README.md)** — the service layer, day to day
- **[architecture/index.md](architecture/index.md)** — the full architecture map, including what was retired

---

## License

MIT — see [LICENSE](LICENSE).
