# Lloyd Services

All services are managed from `~/Projects/lloyd-services/`. Service unit files live in `systemd/` and are symlinked to `~/.config/systemd/user/` via `setup/install-services.sh`.

Manage with: `systemctl --user {start|stop|restart|status} <service>`

## Services

### lloyd-llm.service
**Local LLM — Qwen3.5-35B-A3B (llama-server)**

| | |
|---|---|
| Script | `bin/start-llm.sh` |
| Port | 8091 |
| Restart | on-failure, 15s delay |
| Env | `LD_LIBRARY_PATH=/opt/cuda/lib64`, `CUDA_PATH=/opt/cuda` |
| GPU | 0 (RTX 5090, 32GB) |
| Depends on | network |
| Setup | `bash setup/setup-llm.sh` |

Runs the local LLM via llama-server. No distrobox wrapper — runs directly on host with CUDA access. 128K context, all layers GPU-offloaded.

---

### lloyd-tts.service
**Orpheus TTS Server**

| | |
|---|---|
| Script | `bin/start-orpheus-tts.sh` |
| Port | 8090 |
| Restart | on-failure, 10s delay |
| GPU | 1 |
| Depends on | network |
| Setup | `bash setup/setup-orpheus.sh` |

Text-to-speech server (Orpheus 3B model). Alternative: CosyVoice3 via `bin/start-cosyvoice-tts.sh` (same port, run one at a time).

---

### lloyd-voice-mode.service
**Lloyd Voice Mode (headless)**

| | |
|---|---|
| Script | `uv run python voice_mode.py --headless` |
| Port | 8092 |
| Restart | on-failure, 5s delay |
| Depends on | `lloyd-tts.service`, `lloyd-llm.service` |

Headless voice pipeline — ASR + response generation + TTS orchestration. Exposes HTTP API on :8092 for voice interactions.

---

### lloyd-voice-mcp.service
**Lloyd Voice MCP Server (SSE)**

| | |
|---|---|
| Script | `uv run voice_services.py --transport sse --port 8094` |
| Port | 8094 |
| Restart | always, 3s delay |
| Depends on | `lloyd-voice-mode.service` |

MCP server exposing voice tools over SSE. Bridges OpenClaw to the voice pipeline.

---

### lloyd-tool-mcp.service
**Lloyd Tool Services MCP (SSE)**

| | |
|---|---|
| Script | `uv run tool_services.py --transport sse --port 8093` |
| Port | 8093 |
| Restart | always, 3s delay |
| Depends on | network |

MCP server exposing general tool services (file tools, memory, web search, etc.) over SSE.

---

### openclaw-gateway.service
**OpenClaw Gateway**

| | |
|---|---|
| Script | `openclaw gateway --allow-unconfigured --port 18789` |
| Port | 18789 |
| Restart | always, 5s delay |
| Env | `CUDA_VISIBLE_DEVICES=0` |
| Depends on | `lloyd-tool-mcp.service`, `lloyd-voice-mcp.service`, network |

Main OpenClaw gateway. Connects to both MCP servers for tool and voice capabilities. Accepts webhook and agent API requests.

**Note:** When restarting, use `bin/restart-openclaw.sh` to kill stale processes first.

---

### openclaw-cert.service
**OpenClaw Certificate Download Page**

| | |
|---|---|
| Script | `python3 -m http.server` |
| Port | 18790 |
| Restart | always, 5s delay |
| Depends on | network |

Serves the gateway's self-signed TLS certificate for download over plain HTTP. LAN devices visit `http://192.168.50.108:18790/` to download and install the CA cert so they trust the gateway's HTTPS.

---

## Startup Order

```
network
├── lloyd-llm.service (:8091)
├── lloyd-tts.service (:8090)
│   └── (both) lloyd-voice-mode.service (:8092)
│               └── lloyd-voice-mcp.service (:8094)
├── lloyd-tool-mcp.service (:8093)
│   └── (both MCP) openclaw-gateway.service (:18789)
└── openclaw-cert.service (:18790)
```

## Port Summary

| Port | Service |
|------|---------|
| 8090 | TTS (Orpheus or CosyVoice) |
| 8091 | LLM (Qwen3.5 via llama-server) |
| 8092 | Voice Mode (HTTP) |
| 8093 | Tool MCP (SSE) |
| 8094 | Voice MCP (SSE) |
| 18789 | OpenClaw Gateway |
| 18790 | Certificate Download (HTTP) |

## GPU Assignment

| GPU | Service |
|-----|---------|
| 0 | LLM (Qwen3.5-35B-A3B, RTX 5090 32GB) |
| 1 | TTS (Orpheus or CosyVoice) |
| 2 | OpenClaw Gateway (inference) |

## Setup

First-time setup (all services):
```bash
cd ~/Projects/lloyd-services
bash setup/setup-all.sh
```

Individual service setup:
```bash
bash setup/setup-llm.sh        # LLM server (llama.cpp + model)
bash setup/setup-orpheus.sh     # Orpheus TTS venv
bash setup/setup-cosyvoice.sh   # CosyVoice TTS venv (alternative)
bash setup/install-services.sh  # Install systemd unit files
```

## Venvs

| Venv | Python | Location | Purpose |
|------|--------|----------|---------|
| `.venv/` | 3.11 | Project root | Voice pipeline + MCP tools (managed by `uv`) |
| `venvs/orpheus/` | 3.11 | Project root | Orpheus TTS (PyTorch cu128, orpheus-speech) |
| `venvs/cosyvoice/` | 3.10 | Project root | CosyVoice3 TTS (PyTorch cu121, numpy 1.26.4) |

Separate venvs are needed because CosyVoice pins numpy<=1.26.4 (conflicts with main project).
