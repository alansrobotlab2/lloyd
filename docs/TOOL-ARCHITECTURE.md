# Tool Architecture

How tools are assembled and exposed to the agent at runtime.

## Assembly Pipeline

```
@mariozechner/pi-coding-agent        → base codingTools [read, bash, edit, write]
     ↓
createOpenClawCodingTools()          → wraps/replaces base, adds exec + process + apply_patch
     ↓
  + listChannelAgentTools()          → channel-specific tools (e.g. whatsapp_login)
  + createOpenClawTools()            → 14-18 OpenClaw built-in tools
     ↓
  + resolvePluginTools()             → discovers plugin-registered tools (~25)
     ↓
  applyToolPolicyPipeline()          → filters by policy (provider, group, sandbox, subagent)
     ↓
  final tools[] → serialized as "tools" in LLM API request
```

## Layer 1: Base Coding Tools

Source: `@mariozechner/pi-coding-agent` (`node_modules/@mariozechner/pi-coding-agent/dist/core/tools/index.js`)

The upstream library defines `codingTools = [readTool, bashTool, editTool, writeTool]`.

`createOpenClawCodingTools()` in `reply-Duq0R59W.js:69984` transforms these:

- **read** → wrapped by `createOpenClawReadTool` (adds image sanitization, context-window-aware output budget)
- **edit** → wrapped by `createHostWorkspaceEditTool` (workspace root guard)
- **write** → wrapped by `createHostWorkspaceWriteTool` (workspace root guard)
- **bash** → removed; replaced by `exec` (see below)

Then adds:

| Tool | Factory | Notes |
|------|---------|-------|
| `exec` | `createExecTool()` | Replaces `bash`. Supports approval requests, sandbox, background exec. |
| `process` | `createProcessTool()` | Background process management. Gated by policy. |
| `apply_patch` | `createApplyPatchTool()` | Conditional — only for OpenAI-provider models. |

## Layer 2: OpenClaw Built-in Tools

Source: `createOpenClawTools()` in `reply-Duq0R59W.js:68393`

| Tool | Factory | Conditional |
|------|---------|-------------|
| `browser` | `createBrowserTool()` | Always |
| `canvas` | `createCanvasTool()` | Always |
| `nodes` | `createNodesTool()` | Always |
| `cron` | `createCronTool()` | Always |
| `message` | `createMessageTool()` | Omitted if `disableMessageTool` |
| `tts` | `createTtsTool()` | Always |
| `gateway` | `createGatewayTool()` | Always |
| `agents_list` | `createAgentsListTool()` | Always |
| `sessions_list` | `createSessionsListTool()` | Always |
| `sessions_history` | `createSessionsHistoryTool()` | Always |
| `sessions_send` | `createSessionsSendTool()` | Always |
| `sessions_spawn` | `createSessionsSpawnTool()` | Always |
| `subagents` | `createSubagentsTool()` | Always |
| `session_status` | `createSessionStatusTool()` | Always |
| `web_search` | `createWebSearchTool()` | Config: `tools.web.search.enabled` (currently **disabled**) |
| `web_fetch` | `createWebFetchTool()` | Config: `tools.web.fetch.enabled` (currently **disabled**) |
| `image` | `createImageTool()` | Requires `agentDir` |
| `memory_search` | `createMemorySearchTool()` | Requires memory backend (QMD active) |
| `memory_get` | `createMemoryGetTool()` | Requires memory backend (QMD active) |

## Layer 3: Plugin-Registered Tools

Source: `resolvePluginTools()` in `reply-Duq0R59W.js:68490`

After built-in tools are assembled, `resolvePluginTools()` collects tools registered by plugins via `api.registerTool()`. It checks `existingToolNames` to skip collisions with already-registered built-ins.

### mcp-tools plugin (16 tools)

File: `extensions/mcp-tools/index.ts`
Transport: MCP/SSE proxy to `127.0.0.1:8093` (`tool_services.py`)

| Tool | Category | Timeout |
|------|----------|---------|
| `tag_search` | Vault | default |
| `tag_explore` | Vault | default |
| `vault_overview` | Vault | default |
| `mem_search` | Memory | default |
| `mem_get` | Memory | default |
| `mem_write` | Memory | default |
| `http_search` | Web | 20s |
| `http_fetch` | Web | 20s |
| `http_request` | Web | 20s |
| `file_read` | File | default |
| `file_write` | File | default |
| `file_edit` | File | default |
| `file_glob` | File | default |
| `file_grep` | File | default |
| `run_bash` | System | 120s |

Also registers a `before_prompt_build` hook for `prefill_context` (not a callable tool — fires automatically before the first LLM call).

### voice-tools plugin (3 tools)

File: `extensions/voice-tools/index.ts`
Transport: MCP/SSE proxy to `127.0.0.1:8094` (`voice_services.py`)

| Tool | Description |
|------|-------------|
| `voice_last_utterance` | Last spoken utterance with speaker ID |
| `voice_enroll_speaker` | Enroll speaker from recent audio |
| `voice_list_speakers` | List enrolled voice profiles |

Also hooks `message_sending` to extract `<summary>` tags → TTS via `127.0.0.1:8092`.

### backlog tools (6 tools, via mcp-tools)

Source: `~/Projects/lloyd-services/tool_services.py`
Data: SQLite at `~/.openclaw/data/backlog.db`

| Tool | Description |
|------|-------------|
| `backlog_boards` | List kanban boards |
| `backlog_tasks` | List tasks with filters |
| `backlog_next_task` | Next assigned task |
| `backlog_get_task` | Get task by ID |
| `backlog_update_task` | Update status/blocked/activity |
| `backlog_create_task` | Create new task |

Registered as MCP tools in tool_services.py, proxied through extensions/mcp-tools/index.ts.

### Non-tool plugins

| Plugin | Role |
|--------|------|
| `timing-profiler` | Observational hooks only, logs latency to `logs/timing.jsonl` |
| `model-router` | Model routing hooks only |
| `mission-control` | Dashboard only |

## Layer 4: Policy Filtering

Before tools reach the LLM, `applyToolPolicyPipeline()` filters based on:

- **Tool profiles** — per-provider and per-model allow/deny lists
- **Group policy** — channel/sender restrictions
- **Sandbox policy** — container-level restrictions
- **Subagent policy** — depth-based restrictions for spawned subagents
- **Owner-only gating** — some tools require `senderIsOwner`
- **Message-provider filtering** — some tools only available on specific channels

## Name Collisions

The MCP plugin previously registered `memory_search`, `memory_get`, `web_search`, and `web_fetch` — names that collided with built-in tools. These have been renamed to avoid collisions:

- MCP `mem_search` / `mem_get` — no longer shadow built-in `memory_search` / `memory_get`
- MCP `http_search` / `http_fetch` — no longer collide with built-in `web_search` / `web_fetch`
- Built-in `web_search` / `web_fetch` remain **disabled** in config (`tools.web.search.enabled: false`)
- Built-in `memory_search` / `memory_get` are active (QMD backend enabled)
- `resolvePluginTools()` checks `existingToolNames` and skips plugin tools that collide with already-registered built-ins

## Tool Count Summary

| Source | Count |
|--------|-------|
| Base coding (read, edit, write) | 3 |
| OpenClaw exec/process | 2 |
| OpenClaw built-in (browser, canvas, cron, etc.) | ~15 |
| mcp-tools plugin | 16 |
| voice-tools plugin | 3 |
| backlog tools (via mcp-tools) | 6 |
| **Approximate total** | **~45** |

Exact count varies by session depending on which conditional tools are active and policy filtering.

## Key Source Files

| File | Role |
|------|------|
| `openclaw.json` | Root config — plugin discovery, tool toggles, memory backend |
| `extensions/mcp-tools/index.ts` | MCP tool proxy plugin (16 tools) |
| `extensions/voice-tools/index.ts` | Voice tool proxy plugin (3 tools) |
| `~/Projects/lloyd-services/tool_services.py` | Backlog MCP tools (6 tools, SQLite-backed) |
| `~/Projects/lloyd-services/tool_services.py` | MCP server implementation (memory, web, file, system) |
| `~/Projects/lloyd-services/voice_services.py` | Voice MCP server implementation |
| `~/.npm-global/lib/node_modules/openclaw/dist/plugin-sdk/reply-Duq0R59W.js` | Runtime tool assembly (`createOpenClawCodingTools`, `createOpenClawTools`, `resolvePluginTools`) |
| `~/.npm-global/lib/node_modules/openclaw/node_modules/@mariozechner/pi-coding-agent/dist/core/tools/index.js` | Base `codingTools` definition |
