# Lloyd

Lloyd is a personal AI assistant built on [OpenClaw](https://github.com/openclaw/openclaw) — an open-source AI gateway that connects LLMs to messaging platforms, tools, and services.

This repo is the full configuration and customization layer for a self-hosted OpenClaw deployment running on a Linux workstation (Arch, via distrobox container). It's a working example of what a mature OpenClaw setup looks like after weeks of daily use and iteration.

## Key Features

- **Custom Mission Control** — a React-based web dashboard (`extensions/mission-control/`) with real-time chat, token usage stats, API monitoring, service health, and multi-session management. This is the primary interface for interacting with Lloyd.
- **13 specialist agents** — orchestrated via the Claude Agent SDK, including coder, researcher, reviewer, planner, tester, auditor, and operator agents dispatched automatically by a pipeline coordinator.
- **Voice assistant** — real-time voice interaction with custom TTS voice cloning, Whisper ASR, and VAD, powered by services from the companion [lloyd-services](https://github.com/alansrobotlab2/lloyd-services) repo.
- **Three-tier memory system** — periodic capture (local LLM), nightly reflection (Claude Opus), and behavioral correction tracking.
- **Discord integration** — separate agents for direct messages (full-access Lloyd-Prime) and server/friend conversations (scoped Lloyd-Discord).

## Companion Repo: lloyd-services

Lloyd's voice pipeline, MCP tool servers, and TTS engine run as systemd services defined in [**lloyd-services**](https://github.com/alansrobotlab2/lloyd-services). That repo provides:

- **Voice pipeline** — STT (Whisper), VAD, speaker identification, TTS (Qwen3-TTS with voice cloning)
- **MCP tool server** — unified tool proxy for memory search, web tools, file operations
- **Voice MCP server** — voice-specific tools (utterance retrieval, speaker enrollment)

The OpenClaw extensions in this repo (`voice-tools`, `mcp-tools`) proxy through those services.

## What's In Here

```
├── agents/              # 13 specialist agents (system prompts, tool configs)
│   ├── main/            # Lloyd-Prime — primary assistant (Claude Sonnet 4.6)
│   ├── discord-lloyd/   # Lloyd-Discord — friend-facing Discord agent
│   ├── memory/          # Periodic memory capture agent (local Qwen 3.5)
│   ├── coder/           # Code generation specialist
│   ├── researcher/      # Web research + knowledge capture
│   ├── reviewer/        # Code review
│   ├── planner/         # Task planning and decomposition
│   ├── operator/        # System operations
│   ├── tester/          # Test generation and execution
│   ├── auditor/         # Security and code auditing
│   ├── orchestrator/    # Pipeline coordinator (dispatches other agents)
│   └── ...
├── extensions/          # Custom OpenClaw plugins
│   ├── agent-orchestrator/   # Claude Agent SDK orchestration pipeline
│   ├── mission-control/      # React dashboard (chat, usage stats, services)
│   ├── mcp-tools/            # Unified MCP tool proxy (memory, web, files)
│   ├── voice-tools/          # Voice assistant integration (STT/TTS/VAD)
│   ├── thunderbird-tools/    # Email/calendar/contacts via Thunderbird MCP
│   └── timing-profiler/      # Per-tool latency tracking
├── cron/                # Scheduled jobs (nightly reflection, memory capture)
├── completions/         # Shell completions (zsh, bash, fish, powershell)
├── tools/               # Utility scripts
└── docs/                # Internal documentation
```

## Architecture

**Primary model:** Claude Sonnet 4.6 (Anthropic, via OAuth)
**Local model:** Qwen 3.5 35B-A3B (for memory capture, low-stakes tasks)
**Voice pipeline:** Qwen3-TTS with voice cloning, Whisper ASR, custom VAD — via [lloyd-services](https://github.com/alansrobotlab2/lloyd-services)
**Messaging:** Discord (direct + server), Mission Control web UI
**Knowledge base:** Obsidian vault with BM25 search + tag graph (separate repo)

### Agent Orchestration

The `agent-orchestrator` extension uses the [Claude Agent SDK](https://github.com/anthropics/claude-code) to spawn specialist agents as isolated Claude Code instances. The orchestrator analyzes tasks, plans which agents to dispatch (coder → tester → reviewer), and coordinates the pipeline.

### Memory System

Three-tier memory architecture:
1. **Periodic capture** — local Qwen model extracts key events from session transcripts every 15 minutes
2. **Nightly reflection** — Claude Opus reviews daily notes, distills long-term memory, updates behavioral learnings
3. **Corrections tracking** — positive/negative signals logged and fed back into agent behavior

## Setup

This repo is meant to be cloned into `~/.openclaw`:

```bash
git clone https://github.com/alansrobotlab2/lloyd.git ~/.openclaw
```

You'll also need:
- [OpenClaw](https://github.com/openclaw/openclaw) installed (`npm i -g openclaw`)
- [lloyd-services](https://github.com/alansrobotlab2/lloyd-services) for voice and MCP tool services
- An `openclaw.json` config file (not included — contains secrets). Use `openclaw.example.json` as a template.
- Environment variables for secrets (see below)

### Secrets

All credentials are stored as environment variables, referenced via OpenClaw's SecretRef system:

| Variable | Purpose |
|----------|---------|
| `OPENCLAW_DISCORD_TOKEN` | Discord bot token |
| `OPENCLAW_GATEWAY_TOKEN` | Gateway auth token |
| `OPENCLAW_HOOKS_TOKEN` | Webhook auth token |
| `OPENCLAW_ANTHROPIC_TOKEN` | Anthropic API key |
| `OPENCLAW_TTS_OPENAI_KEY` | TTS API key |
| `OPENCLAW_LOCAL_LLM_KEY` | Local LLM API key |

The `openclaw.json` config uses SecretRef objects instead of plaintext:
```json
"token": { "source": "env", "provider": "default", "id": "OPENCLAW_DISCORD_TOKEN" }
```

### Running

```bash
openclaw gateway --allow-unconfigured --port 18789
```

Or via systemd (recommended for persistent deployment).

## What's NOT in This Repo

- `openclaw.json` — main config file (contains SecretRef pointers, gitignored)
- `identity/` — device keys and auth tokens
- `devices/` — paired device registry
- `certs/` — TLS certificates
- Session transcripts, memory databases, logs

All sensitive and ephemeral data is gitignored. See `.gitignore` for the full list.

## License

This is a personal configuration repo shared as a reference. Use whatever's useful to you.
