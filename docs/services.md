# Lloyd Services

All services are systemd user units in `~/.config/systemd/user/` and run inside the `lloyd` distrobox container.

Manage with: `systemctl --user {start|stop|restart|status} <service>`

## Services

### lloyd-llm.service
**Local LLM — Qwen3.5-35B-A3B (llama-server)**

| | |
|---|---|
| Script | `~/Projects/lloyd/scripts/start-qwen35.sh` |
| Restart | on-failure, 15s delay |
| Env | `LD_LIBRARY_PATH=/opt/cuda/lib64`, `CUDA_PATH=/opt/cuda` |
| Depends on | network |

Runs the local LLM via llama-server. No distrobox wrapper — runs directly on host with CUDA access.

---

### lloyd-tts.service
**Orpheus TTS Server**

| | |
|---|---|
| Script | `~/Projects/lloyd/scripts/start-orpheus-tts.sh` |
| Restart | on-failure, 10s delay |
| Depends on | network |

Text-to-speech server (Orpheus model).

---

### lloyd-voice-mode.service
**Lloyd Voice Mode (headless)**

| | |
|---|---|
| Script | `cd ~/Projects/lloyd-services && uv run python voice_mode.py --headless` |
| Port | 8092 |
| Restart | on-failure, 5s delay |
| Depends on | `lloyd-tts.service`, `lloyd-llm.service` |

Headless voice pipeline — ASR + response generation + TTS orchestration. Exposes HTTP API on :8092 for voice interactions.

---

### lloyd-voice-mcp.service
**Lloyd Voice MCP Server (SSE)**

| | |
|---|---|
| Script | `cd ~/Projects/lloyd-services && uv run voice_services.py --transport sse --port 8094` |
| Port | 8094 |
| Restart | always, 3s delay |
| Depends on | `lloyd-voice-mode.service` |

MCP server exposing voice tools over SSE. Bridges OpenClaw to the voice pipeline.

---

### lloyd-tool-mcp.service
**Lloyd Tool Services MCP (SSE)**

| | |
|---|---|
| Script | `cd ~/Projects/lloyd-services && uv run tool_services.py --transport sse --port 8093` |
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
| Env | `CUDA_VISIBLE_DEVICES=2` |
| Depends on | `lloyd-tool-mcp.service`, `lloyd-voice-mcp.service`, network |

Main OpenClaw gateway. Connects to both MCP servers for tool and voice capabilities. Accepts webhook and agent API requests.

**Note:** When restarting, kill the old process first — see CLAUDE.md for the restart procedure.

---

### lloyd-clawdeck.service
**ClawDeck API Server (Rails)**

| | |
|---|---|
| Script | `cd ~/Development/clawdeck && bin/rails server -p 3001 -b 127.0.0.1` |
| Port | 3001 |
| Restart | on-failure, 5s delay |
| Depends on | network |

Rails API server for the ClawDeck dashboard. Independent of the voice/LLM stack.

---

## Startup Order

```
network
├── lloyd-llm.service
├── lloyd-tts.service
│   └── (both) lloyd-voice-mode.service (:8092)
│               └── lloyd-voice-mcp.service (:8094)
├── lloyd-tool-mcp.service (:8093)
│   └── (both MCP) openclaw-gateway.service (:18789)
└── lloyd-clawdeck.service (:3001)
```

## Port Summary

| Port | Service |
|------|---------|
| 3001 | ClawDeck (Rails) |
| 8092 | Voice Mode (HTTP) |
| 8093 | Tool MCP (SSE) |
| 8094 | Voice MCP (SSE) |
| 18789 | OpenClaw Gateway |
