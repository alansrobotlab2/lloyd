---
segment: architecture
relations:
  related-to:
  - architecture/voice.md
  - architecture/memory.md
  - architecture/autonomy-system.md
  - architecture/evaluation-engine.md
  - architecture/backlog.md
  - architecture/index.md
  - architecture/infrastructure.md
  - architecture/morning-briefing.md
  - autonomy/38-nightly-reflection-signals.md
  - architecture/nightly-skills-management.md
  - architecture/nightly-vault-maintenance.md
  - architecture/skills.md
tags: [architecture]
summary: 41-tool Python FastMCP server providing memory,web,filesystem,system,
  backlog,and skills tools to the OpenClaw Gateway via SSE transport.
type: reference

---



















# MCP Tools

The MCP Tools Server is a consolidated Python service providing 41 tools to the [[index|OpenClaw Gateway]] via SSE transport.

## Server

- **Source:** `~/agent-services/tool_services.py`
- **Framework:** Python,FastMCP
- **Service:** `lloyd-tool-mcp` (supervisord inside distrobox `lloyd-services`)
- **Command:** `uv run tool_services.py --transport sse --port 8093`
- **Port:** 8093 (SSE transport)
- **Architecture:** Single consolidated server replacing 5 separate per-plugin `server.py` processes
- **Dependencies:** mcp[cli],pyyaml,ddgs,httpx,readability-lxml,html2text,uvicorn,starlette,sse-starlette

## OpenClaw Plugin

- **Path:** `~/.openclaw/extensions/mcp-tools/`
- **Implementation:** `index.ts` registers tools + prefill hook + work mode commands
- **Transport:** Uses `McpSseClient` to proxy tool calls to the Python MCP server
- **Additional responsibilities:**
  - Mode switching (`/work`,`/personal`,`/general`,`/mode` commands)
  - Daily memory file creation on `session_end` hook
  - Mode-aware scope injection via `before_tool_call` hook

## Tool Allowlists

Each agent in `~/.openclaw/openclaw.json` has a tool allowlist controlling which of the 41 tools it can access. The gateway enforces these per-agent,so skill-only agents never see file or bash tools.

## Tool Inventory (41 Tools)

### Memory / Vault (6)

| Tool | Description |
|------|-------------|
| `tag_search` | Search vault by tags |
| `tag_explore` | Explore tag relationships |
| `vault_overview` | Vault structure overview |
| `mem_search` | BM25 FTS5 full-text search |
| `mem_get` | Read vault file by path |
| `mem_write` | Write to vault file |

### Prefill (1)

| Tool | Description |
|------|-------------|
| `prefill_context` | Tag match + BM25 + GLM keywords for context injection |

### Web (3)

| Tool | Description |
|------|-------------|
| `http_search` | DuckDuckGo web search |
| `http_fetch` | Fetch URL with readability extraction |
| `http_request` | Raw HTTP request |

### File System (6)

| Tool | Description |
|------|-------------|
| `file_read` | Read file contents |
| `file_write` | Write file contents |
| `file_edit` | Edit file with string replacement |
| `file_patch` | Apply patch to file |
| `file_glob` | Find files by glob pattern |
| `file_grep` | Search file contents with regex |

All file operations are sandboxed to `$HOME`.

### System (3)

| Tool | Description |
|------|-------------|
| `run_bash` | Shell command execution (120s timeout) |
| `bg_exec` | Background process execution (long-running) |
| `bg_process` | Background process management |

### Backlog (4)

| Tool | Description |
|------|-------------|
| `backlog_boards` | List all boards |
| `backlog_tasks` | List/filter tasks |
| `backlog_get_task` | Full task details by ID |
| `backlog_write_task` | Create or update a task |

See [[backlog]] for the backlog system architecture.

### Skills (3)

| Tool | Description |
|------|-------------|
| `skills_search` | Search local + ClawhHub catalog skills |
| `skills_get` | Get skill content (local or `clawhub:<slug>`) |
| `skills_install` | Install ClawhHub skill with security validation |

See [[skills]] for the skill system.

### Autonomy (6)

| Tool | Description |
|------|-------------|
| `autonomy_tasks` | List/filter autonomy tasks |
| `autonomy_get_task` | Get full task detail + recent runs |
| `autonomy_write_task` | Create or update (upsert) autonomy task |
| `autonomy_run_task` | Trigger immediate task execution |
| `autonomy_delete_task` | Delete or archive a task |
| `autonomy_config` | Get or set autonomy system config |

See [[autonomy-system]] for the autonomy system architecture.

## Architecture

```mermaid
graph LR
    subgraph "OpenClaw Gateway (Node.js)"
        Plugin["mcp-tools/index.ts"]
        Hooks["Hooks:<br/>before_prompt_build<br/>before_tool_call<br/>session_end"]
        Commands["Commands:<br/>/work /personal /general /mode"]
    end

    subgraph "Python MCP Server :8093"
        FastMCP["tool_services.py<br/>(FastMCP)"]
    end

    subgraph "Data Stores"
        Vault["Obsidian Vault"]
        SQLite["backlog.db"]
        FS["File System ($HOME)"]
    end

    subgraph "External"
        DDG["DuckDuckGo"]
        LocalLLM["Local LLM :8091"]
    end

    Plugin -->|"SSE (McpSseClien