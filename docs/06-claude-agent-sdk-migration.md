# 06 — Migration Plan: Hermes to Lloyd (Claude Agent SDK)

## Context

The current system uses a large custom agent framework (`hermes-agent`, ~8,800-line `AIAgent` class) deployed at `~/.hermes/` with a web UI, autonomy scheduler, 8 plugin directories, ~50+ skills, and supporting infrastructure managed from `~/agent-services/`.

**Lloyd** (`~/lloyd/`) is the successor — a clean-slate deployment built on the **Claude Agent SDK** (`claude-code-sdk`). It replaces hermes-agent's custom orchestration while preserving all custom extensions as MCP tool servers.

## Validated: Local Models Work with the SDK

Testing confirmed (2026-04-03) that the Claude Agent SDK works with local Qwen models via vLLM's Anthropic-compatible API endpoint:

```bash
ANTHROPIC_BASE_URL="http://127.0.0.1:8096"          # no /v1 suffix
ANTHROPIC_API_KEY="no-key-required"
ANTHROPIC_CUSTOM_MODEL_OPTION="Qwen3.5-122B-A10B"
ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen 122B Local"
```

| Test | Result |
|------|--------|
| Basic `query()` with Qwen 122B | Works |
| Built-in tool calling (Read, Bash) | Works |
| Custom MCP server tools | Works |
| Streaming event inspection | Works — full event stream with all block types |

Test scripts: `~/.hermes/tests/claude-sdk/01-04*.py`
SDK venv: `~/agent-services/.venvs/claude-agent-sdk/` (Python 3.12, claude-code-sdk v0.0.25)

## Design Decisions

- **Single mode**: All agent interactions go through Claude Agent SDK. No hermes-agent dependency. Model switching (local Qwen vs Claude API) handled via environment variables per session.
- **SDK interface**: `query()` for all interactions (web chat + autonomy). Simpler than `ClaudeSDKClient`, and hooks can be added later via `can_use_tool` callback.
- **Clean-slate project**: `~/lloyd/` is a new repo, not a fork of `~/.hermes/`. Extensions are migrated and converted, not symlinked.
- **Architecture**: Each session spawns a `claude` subprocess (managed by SDK). MCP servers provide custom tools. FastAPI provides the web API + SSE bridge.

---

## ~/lloyd/ Project Structure

```
~/lloyd/
├── server.py                  # FastAPI backend (SDK-powered, replaces mc_server.py)
├── autonomy.py                # Task scheduler (SDK-powered, replaces autonomy_scheduler.py)
├── prompt_builder.py          # System prompt assembly (extracted from hermes-agent)
├── config.yaml                # Lloyd configuration
├── SOUL.md                    # Agent identity/persona
├── .env                       # API keys, model endpoints
│
├── mcp-servers/               # Custom tools as MCP servers
│   ├── autonomy.py            # Task CRUD + config + run (from plugins/autonomy/)
│   ├── backlog.py             # Kanban board CRUD (from plugins/backlog/)
│   ├── memory.py              # Knowledge graph (from plugins/next-gen-memory/)
│   ├── mission_control.py     # Session management (from plugins/mission-control/)
│   ├── subliminal.py          # Vault recall + injection (from plugins/subliminal/)
│   ├── http_tools.py          # DuckDuckGo + extract (from plugins/http-tools/)
│   ├── thunderbird.py         # Email/calendar (from plugins/thunderbird-tools/)
│   └── pipeline.py            # Multi-stage workers (from plugins/pipeline/)
│
├── web/                       # React frontend (migrated from mc-web/)
│   ├── src/
│   ├── package.json
│   └── ...
│
├── skills/                    # Skill definitions (migrated from ~/.hermes/skills/ + ~/obsidian/skills/)
│
├── tests/                     # Test suite
│   └── sdk/                   # SDK integration tests (migrated from ~/.hermes/tests/claude-sdk/)
│
├── memories/                  # Persistent memory (MEMORY.md, USER.md)
├── sessions/                  # Session metadata
├── logs/                      # Application logs
└── docs/                      # Documentation
```

---

## Migration Phases

### Phase 1: Scaffold + MCP Servers

**Goal**: Create `~/lloyd/`, port all 8 plugins to MCP servers, verify they work with the SDK.

#### 1a. Scaffold the project

```bash
mkdir -p ~/lloyd/{mcp-servers,web,skills,memories,sessions,logs,docs,tests/sdk}
```

Copy over:
- `SOUL.md` from `~/.hermes/SOUL.md`
- `memories/` from `~/.hermes/memories/`
- Skills from `~/.hermes/skills/` and `~/obsidian/skills/`

#### 1b. Port ALL plugins to MCP servers

Each plugin in `~/.hermes/plugins/` becomes a standalone MCP server in `~/lloyd/mcp-servers/`:

| Source | Target | Tools |
|--------|--------|-------|
| `plugins/autonomy/__init__.py` | `mcp-servers/autonomy.py` | 6 tools: task CRUD + config + run |
| `plugins/backlog/__init__.py` | `mcp-servers/backlog.py` | 4 tools: board CRUD |
| `plugins/next-gen-memory/__init__.py` | `mcp-servers/memory.py` | entity/fact/relation CRUD |
| `plugins/mission-control/__init__.py` | `mcp-servers/mission_control.py` | chat_send, chat_list_sessions |
| `plugins/subliminal/__init__.py` | `mcp-servers/subliminal.py` | vault recall + context injection |
| `plugins/http-tools/__init__.py` | `mcp-servers/http_tools.py` | DuckDuckGo search + content extract |
| `plugins/thunderbird-tools/__init__.py` | `mcp-servers/thunderbird.py` | email/calendar |
| `plugins/pipeline/__init__.py` | `mcp-servers/pipeline.py` | multi-stage worker coordination |

Pattern for each:
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("lloyd-autonomy")

@app.list_tools()
async def list_tools():
    return [Tool(name="...", description="...", inputSchema={...}), ...]

@app.call_tool()
async def call_tool(name, arguments):
    # Port the handler logic from the plugin — rewrite, don't import
    # (lloyd is a clean break, no dependency on hermes-agent internals)
    ...
```

#### 1c. Test each MCP server standalone

Verify via `mcp dev` or direct stdio before SDK integration.

---

### Phase 2: Server + SSE Bridge

**Goal**: Create `server.py` (FastAPI) that uses the Claude Agent SDK for all agent interactions, with SSE streaming to the web UI.

#### 2a. Core server

```python
# ~/lloyd/server.py
from claude_code_sdk import query, ClaudeCodeOptions, SystemMessage, AssistantMessage, UserMessage, ResultMessage
from claude_code_sdk import TextBlock, ToolUseBlock, ToolResultBlock

MCP_SERVERS = {
    "autonomy":    {"type": "stdio", "command": "python", "args": ["mcp-servers/autonomy.py"]},
    "backlog":     {"type": "stdio", "command": "python", "args": ["mcp-servers/backlog.py"]},
    "memory":      {"type": "stdio", "command": "python", "args": ["mcp-servers/memory.py"]},
    "mc":          {"type": "stdio", "command": "python", "args": ["mcp-servers/mission_control.py"]},
    "subliminal":  {"type": "stdio", "command": "python", "args": ["mcp-servers/subliminal.py"]},
    "http":        {"type": "stdio", "command": "python", "args": ["mcp-servers/http_tools.py"]},
    "thunderbird": {"type": "stdio", "command": "python", "args": ["mcp-servers/thunderbird.py"]},
    "pipeline":    {"type": "stdio", "command": "python", "args": ["mcp-servers/pipeline.py"]},
}
```

#### 2b. Model configuration

```python
MODEL_CONFIGS = {
    "Qwen3.5-122B-A10B": {
        "env": {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8096",
            "ANTHROPIC_API_KEY": "no-key-required",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen3.5-122B-A10B",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen 122B",
        },
        "alias": "122b",
    },
    "Qwen3.5-35B-A3B": {
        "env": {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8091",
            "ANTHROPIC_API_KEY": "no-key-required",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen3.5-35B-A3B",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen 35B",
        },
        "alias": "35b",
    },
    "claude-sonnet-4-6": {"env": {}, "alias": "sonnet"},
    "claude-opus-4-6": {"env": {}, "alias": "opus"},
}
```

#### 2c. SSE streaming bridge

Map SDK events to existing SSE format (frontend-compatible):

| SDK Message | Maps to SSE |
|-------------|-------------|
| `SystemMessage(subtype="init")` | `session` (session_id) |
| `AssistantMessage` → `TextBlock` | `text_delta` |
| `AssistantMessage` → `ToolUseBlock` | `tool_start` (id, name, input) |
| `UserMessage` → `ToolResultBlock` | `tool_complete` (tool_use_id, content) |
| `ResultMessage` | `done` (result text, session_id) |

#### 2d. Session management

- New session: `query()` without `resume`, capture `session_id` from `SystemMessage`
- Resume: `query()` with `resume=session_id`
- Store session_id + model + metadata in `~/lloyd/sessions/` JSON files

**Files to create:** `server.py`

---

### Phase 3: Autonomy Scheduler

**Goal**: Port `autonomy_scheduler.py` to use `query()` for task execution.

- Reads task definitions from `~/obsidian/autonomy/` (unchanged)
- Executes via `query()` with task skill as `system_prompt`
- Writes run records to `~/obsidian/autonomy/runs/`
- Background ticker (60s interval) triggered from `server.py`

**Files to create:** `autonomy.py`

---

### Phase 4: System Prompt + Config

**Goal**: Create `prompt_builder.py` and `config.yaml`.

#### prompt_builder.py

Extract system prompt assembly into standalone module:
- SOUL.md identity
- Memory (MEMORY.md, USER.md from `memories/`)
- Skills index (list from `skills/`)
- Platform hints, timestamp, personality

#### config.yaml

Simplified from hermes config — only what lloyd needs:
- Model defaults + model configs
- MCP server list
- Autonomy settings
- Display/personality preferences
- Skills paths

**Files to create:** `prompt_builder.py`, `config.yaml`

---

### Phase 5: Frontend Migration

**Goal**: Migrate `mc-web/` to `~/lloyd/web/`.

- Copy React app from `~/.hermes/mc-web/`
- Update API base URL to point to lloyd's `server.py`
- SSE event types are identical — should work with zero changes to parsing
- Update model list to show both local + Claude models
- `npm install && npm run build`

**Files to copy + modify:** `web/` (from `mc-web/`)

---

### Phase 6: Infrastructure Migration (from agent-services)

**Goal**: Migrate operational infrastructure from `~/agent-services/` to `~/lloyd/`.

#### What moves from agent-services:

| Component | Source | Target | Notes |
|-----------|--------|--------|-------|
| SDK venv | `.venvs/claude-agent-sdk/` | `~/lloyd/.venv/` | Single venv for lloyd |
| Supervisor: backend | `conf.d/hermes-mc-backend.conf` | `conf.d/lloyd-backend.conf` | Points to `~/lloyd/server.py` |
| Supervisor: frontend | `conf.d/hermes-mc-frontend.conf` | `conf.d/lloyd-frontend.conf` | Points to `~/lloyd/web/` |
| LLM configs | `conf.d/agent-llm-*.conf` | stays in agent-services | LLM serving is infra, not app |
| Logs | `logs/hermes-mc-*.log` | `~/lloyd/logs/` | New log paths |
| TTS service | `services/tts/` | evaluate later | May become MCP server |
| Discord voice | `services/discord-voice-bridge/` | evaluate later | May become MCP server |
| Thunderbird MCP | `services/thunderbird-mcp/` | absorbed into `mcp-servers/thunderbird.py` | Already being ported |
| Idle worker | `services/idle-worker/` | evaluate later | May become autonomy task |

#### What stays in agent-services:

- LLM model serving (vllm configs, llama.cpp)
- Voice pipeline (whisper, TTS)
- Model files (`llm/models/`)
- sglang/vllm venvs

#### New supervisor configs:

```ini
[program:lloyd-backend]
command=/home/alansrobotlab/lloyd/.venv/bin/python /home/alansrobotlab/lloyd/server.py
directory=/home/alansrobotlab/lloyd
environment=HOME="/home/alansrobotlab",ANTHROPIC_API_KEY="%(ENV_ANTHROPIC_API_KEY)s"
autorestart=true
stdout_logfile=/home/alansrobotlab/lloyd/logs/server.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/server.err

[program:lloyd-frontend]
command=/usr/bin/npm --prefix /home/alansrobotlab/lloyd/web run dev
directory=/home/alansrobotlab/lloyd/web
autorestart=true
stdout_logfile=/home/alansrobotlab/lloyd/logs/frontend.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/frontend.err
```

---

## Execution Order

| Phase | Effort | Dependencies | Notes |
|-------|--------|-------------|-------|
| 1a: Scaffold ~/lloyd | 0.5 session | None | Directory structure + copy assets |
| 1b: MCP servers (all 8) | 2-3 sessions | 1a | Port all plugin logic |
| 1c: Test MCP servers | 0.5 session | 1b | Verify each standalone |
| 4: prompt_builder + config | 0.5 session | 1a | Parallel with 1b |
| 2: server.py + SSE bridge | 1-2 sessions | 1b, 4 | Core backend |
| 3: autonomy.py | 0.5 session | 1b, 4 | Task scheduler |
| 5: Frontend migration | 0.5 session | 2 | Copy + point at new backend |
| 6: Infrastructure migration | 1 session | 2, 5 | Supervisor, venv, logs |

**Start with**: Phase 1a (scaffold) → Phase 1b (MCP servers) + Phase 4 (prompt/config) in parallel → Phase 2 (server).

---

## What Gets Retired

After lloyd is running:

- `~/.hermes/mc_server.py` — replaced by `~/lloyd/server.py`
- `~/.hermes/autonomy_scheduler.py` — replaced by `~/lloyd/autonomy.py`
- `~/.hermes/plugins/` — replaced by `~/lloyd/mcp-servers/`
- `~/.hermes/mc-web/` — replaced by `~/lloyd/web/`
- `~/.hermes/config.yaml` — replaced by `~/lloyd/config.yaml`
- `~/Projects/hermes-agent/` — no longer needed (SDK replaces AIAgent entirely)
- `agent-services/conf.d/hermes-mc-*.conf` — replaced by `lloyd-*.conf`
- `agent-services/.venvs/hermes/` — replaced by `~/lloyd/.venv/`

What persists (shared data, not application code):
- `~/obsidian/autonomy/` — task definitions (lloyd reads these)
- `~/obsidian/backlog/` — kanban items (lloyd reads these)
- `~/obsidian/skills/` — user skills (lloyd reads these)
- `~/.hermes/state.db` — historical sessions (read-only archive)
- `agent-services/` — LLM serving, voice, model infra (unchanged)

---

## Verification Plan

1. **MCP servers**: Test each standalone with `mcp dev`
2. **SDK + local**: `query()` with MCP servers against Qwen 122B
3. **SDK + Claude**: Same test against `claude-sonnet-4-6`
4. **Web chat streaming**: Send message via web UI, verify SSE events
5. **Session resume**: Start chat, close browser, reopen, verify resume
6. **Autonomy**: Trigger task via `/api/autonomy/run`
7. **Model switching**: Switch between 122B / 35B / sonnet from web UI
8. **Tool coverage**: Test every MCP tool via chat
9. **Supervisor**: Start lloyd-backend + lloyd-frontend via supervisord

---

## Appendix: Existing Test Scripts

Test scripts at `~/.hermes/tests/claude-sdk/` (will be copied to `~/lloyd/tests/sdk/`):

- `01_basic_query.py` — basic `query()` with local model
- `02_tool_calling.py` — built-in Read/Bash tools
- `03_mcp_server.py` — custom MCP server tools
- `04_streaming_events.py` — full event stream inspection
