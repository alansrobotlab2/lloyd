# Lloyd

A voice-first personal AI agent that runs entirely on your own hardware.

Lloyd runs its own agent loop against a local vLLM server, exposes every tool
through a single MCP aggregator, and keeps its long-term memory in an Obsidian
vault. No model inference leaves the machine — there is no Anthropic, OpenAI, or
other inference API in the loop. (Lloyd can still reach the internet when you
ask it to search or browse; it is the *thinking* that stays local, not the
network.)

> **This is one person's system, not a product.** It is built for, and pinned
> to, a specific machine — Arch Linux, three NVIDIA GPUs, ~311 GB of local model
> weights. It is published because the design is worth reading and the pieces
> are worth stealing, not because it will `git clone && run` on your laptop.
> [SETUP.md](SETUP.md) is honest about exactly how much is involved.

---

## What's in the box

| Piece | What it does | Where |
|---|---|---|
| **Agent harness** | In-process agent loop: SSE stream from vLLM → tool dispatch → repeat. Handles compaction, tool-result spill, and a searchable tool index. | [`app/harness/`](app/harness/) |
| **Backend** | FastAPI, 18 routers, SSE streaming to the browser. Port 8080. | [`server.py`](server.py), [`app/routers/`](app/routers/) |
| **MCP aggregator** | One `Server("lloyd")` on `:8500/sse` fronting every tool — including the built-in Bash/Read/Write/Edit/Grep/Glob/Task. | [`agent_mcp/`](agent_mcp/) |
| **Mission Control** | React + Vite dashboard: chat, sessions, token usage, service health, an embedded editor. Port 5173. | [`web/`](web/) |
| **Voice** | LiveKit room → wake word (openwakeword) → VAD → faster-whisper STT → speaker ID → harness → Qwen3-TTS. | [`agent-services/livekit_worker.py`](agent-services/livekit_worker.py) |
| **Memory** | Obsidian vault at `~/obsidian`, searched by a qmd daemon over seven per-segment collections. | [`architecture/memory.md`](architecture/memory.md) |
| **Autonomy** | A task scheduler plus a worker pool — autoresearch, gap-fill, session distillation, nightly reflection. | [`autonomy.py`](autonomy.py), [`workers/`](workers/) |
| **Inner Voice** | A second, smaller model that watches the primary's tool calls and can intervene. | [`app/inner_voice/`](app/inner_voice/) |
| **Usage tracking** | SQLite token and cost accounting per session and model. | [`usage_store.py`](usage_store.py) |

### Models

| Role | Model | Served at | Notes |
|---|---|---|---|
| Primary | `unsloth-Qwen3.8-27B-NVFP4` | `127.0.0.1:8096` | vLLM, 262 k context, speculative decode via an MTP head |
| Secondary | `gemma-4-e4b-nvfp4` | `127.0.0.1:8091` | Inner Voice observer. `autostart=false` — start it by hand |

vLLM is launched with `--enable-auto-tool-choice --tool-call-parser qwen3_xml
--reasoning-parser qwen3`.

---

## How it fits together

```
                 browser  ·  voice  ·  Discord
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  FastAPI backend  :8080         │
        │  /api/message/stream (SSE)      │
        └───────────────┬─────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │  agent harness (app/harness/)   │
        │  stream → tool dispatch → loop  │
        └───────┬─────────────────┬───────┘
                │                 │
                ▼                 ▼
    ┌───────────────────┐   ┌──────────────────────┐
    │  vLLM  :8096      │   │  lloyd-mcp  :8500    │
    │  (local weights)  │   │  every tool, one SSE │
    └───────────────────┘   └──────────┬───────────┘
                                       │
                     ┌─────────────────┼─────────────────┐
                     ▼                 ▼                 ▼
              Obsidian vault      qmd  :8181       Bash / files /
              (~/obsidian)        (vault search)   browser / HTTP
```

Each turn rebuilds the full conversation from the persisted session JSON and
sends it to vLLM as an OpenAI-format `messages` list. There is no `resume=` —
history is reconstructed every time, which is what makes compaction and editing
past turns tractable.

---

## Hardware

Bus order matters; every service pins it with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

| GPU | Card | Used by |
|---|---|---|
| 0 | RTX 3090 (24 GB) | qmd daemon + watcher, Qwen3-TTS |
| 1 | RTX PRO 6000 Blackwell (96 GB) | vLLM primary |
| 2 | RTX 3090 (24 GB) | spare |

---

## Running it

Everything runs directly on the host under supervisord, installed as the
`agent-supervisord.service` systemd `--user` unit. There is no container.

```bash
alias lsup='/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl \
  -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf'

lsup status
lsup restart lloyd-mc:lloyd-backend    # after editing server.py
lsup tail -f agent-llm-primary stderr
```

The backend, frontend, and MCP aggregator live in the **`lloyd-mc` group** and
must be addressed with that prefix — `lloyd-mc:lloyd-backend`, never
`lloyd-backend`. Frontend edits are picked up by Vite HMR without a restart.

Mission Control is then at **http://localhost:5173**.

---

## Configuration

[`config.yaml`](config.yaml) holds models, MCP servers, voice, autonomy, worker
sources, and agent settings. It is **read-only at boot** — UI toggles persist to
`data/tool_overrides.yaml` (gitignored) and are merged over it. If a
hand-edited change to `config.yaml` doesn't take, check that the override file
isn't shadowing the same key.

Secrets live in `.env` (gitignored) and reach `config.yaml` through `${VAR}`
placeholders expanded at boot. **Never put a literal secret in `config.yaml`** —
it is tracked. See [`.env.example`](.env.example).

Tools are disabled either server-wide (`mcp_servers.<name>.enabled: false`) or
individually (`mcp_servers.<name>.disabled_tools: [Bash, ...]`, using bare tool
names).

---

## Setup

**[SETUP.md](SETUP.md)** is the authority for a rebuild from a fresh OS: system
packages, the uv/bun/npm toolchain, all four venvs, supervisord and the systemd
unit, model downloads, and — importantly — **what to back up before you wipe**,
since several runtime assets are untracked and not re-downloadable.

```bash
agent-services/setup/setup-all.sh --check   # reports what's missing, changes nothing
```

---

## Documentation

- **[SETUP.md](SETUP.md)** — full bare-metal rebuild
- **[CLAUDE.md](CLAUDE.md)** — orientation for coding agents working in this repo
- **[agent-services/README.md](agent-services/README.md)** — the service layer, day to day
- **[architecture/](architecture/)** — per-subsystem design docs

---

## License

MIT — see [LICENSE](LICENSE).
