# MCP Tool Migration: OpenClaw Extensions → lloyd-services

## Context

The OpenClaw chat gateway is crashing repeatedly due to a fragile MCP subprocess architecture. Five separate plugins each spawn their own independent `server.py` subprocess over stdio, leading to:

- `anyio.ClosedResourceError` crashes when the parent closes the pipe before the Python subprocess finishes responding
- Stuck sessions that leave the chat window permanently unresponsive
- Plugin tool name conflicts between `memory-core` (built-in) and `memory-graph` (extension)
- 4-5 redundant Python processes all loading the full tool set but only using a subset each

**Goal:** Consolidate all MCP tools into a single dedicated server in `~/Projects/lloyd-services/`, configure OpenClaw to connect to it directly, and strip out the proxy plugin layer from `.openclaw/extensions/`.

---

## Tool Inventory (16 tools to migrate)

### Memory & Vault Tools
- [x] `tag_search` — search vault by frontmatter tags (AND/OR mode, type filter)
- [x] `tag_explore` — tag co-occurrence and bridging documents
- [x] `vault_overview` — vault statistics (doc counts, type distribution, hubs)
- [x] `mem_search` — BM25 full-text search via qmd subprocess
- [x] `mem_get` — read vault file by relative path (with line range)
- [x] `mem_write` — create/overwrite vault file

### Prefill Pipeline
- [x] `prefill_context` — full memory prefill pipeline (tag match + BM25 + GLM keywords + merge/rank + format)

### Web Tools
- [x] `http_search` — DuckDuckGo search
- [x] `http_fetch` — fetch URL + readability extraction → markdown/text
- [x] `http_request` — raw HTTP request (GET/POST/PUT/PATCH/DELETE/HEAD)

### File System Tools
- [x] `file_read` — read file (within $HOME, 2MB limit)
- [x] `file_write` — write file (within $HOME)
- [x] `file_edit` — exact string replacement in file (unique match required)
- [x] `file_glob` — glob pattern file search (200 match limit)
- [x] `file_grep` — regex content search across files

### System Tools
- [x] `run_bash` — execute shell command (within $HOME, 1-120s timeout)

### NOT migrating (separate concerns)
- **voice tools** — already in lloyd-services as `voice_mcp_server.py`, separate server
- **backlog tools** — now MCP-based (SQLite), part of tool_services.py
- **timing-profiler** — hook-only plugin, no tools
- **model-router** — hook-only plugin, no tools

---

## Target Architecture

```
lloyd-services/
├── voice_mcp_server.py          (existing — voice tools)
├── voice_pipeline.py            (existing)
├── voice_tui.py                 (existing)
├── openclaw_mcp_server.py       (NEW — all 16 tools above)
├── pyproject.toml               (update deps)
└── ...
```

**Runtime:** Single long-running `openclaw_mcp_server.py` process, started via stdio by OpenClaw's native MCP config or a single thin plugin.

**Key difference from current arch:** One process instead of five. Subprocess lifecycle managed by one connection instead of five racing `process.on("exit")` handlers.

---

## Migration Steps

### Phase 1: Create the new server

1. Create `~/Projects/lloyd-services/openclaw_mcp_server.py`
2. Copy the core infrastructure from the existing `server.py`:
   - `DocMeta` dataclass, `TagIndex` class, vault scanning (`_scan_vault`, `_walk_markdown`)
   - Frontmatter parser, importance scoring, type boost constants
   - All hardcoded paths/constants (VAULT, QMD, HOME, etc.)
   - Lifespan context manager with `_refresh_index_loop()`
   - Private IP blocking (`_is_private_host`)
   - Safe path validation (`_safe_path`)
   - Prefill pipeline (all helper functions: `_extract_user_query`, `_extract_topic_keywords`, `_extract_keywords_via_glm`, `_run_tag_match`, `_merge_and_rank`, `_format_unified_context`, `_log_prefill`, cache)
3. Register all 16 tools with `@mcp.tool()` decorators
4. Update `pyproject.toml` to add dependencies: `pyyaml`, `ddgs`, `readability-lxml`, `html2text` (httpx and mcp[cli] already present)
5. Add `uv` script metadata header (like existing server.py)

### Phase 2: Validate the new server standalone

6. Run `uv run openclaw_mcp_server.py` and verify startup (vault scan, index build)
7. Test each tool category with manual JSON-RPC calls over stdio
8. Verify `prefill_context` works (requires local vLLM on :8091)

### Phase 3: Wire up to OpenClaw

Since OpenClaw doesn't support native `mcpServers` config, we need a single thin plugin to connect:

9. Create a new unified plugin `extensions/mcp-tools/index.ts` that:
   - Spawns ONE `McpStdioClient` pointing to `~/Projects/lloyd-services/openclaw_mcp_server.py`
   - Registers all 16 tools as proxies
   - Implements the `before_prompt_build` hook for `prefill_context`
   - Sets appropriate timeouts (120s for `run_bash`, 15s for web tools, 10s default)
10. OR: Replace the existing `memory-graph/index.ts` to point to the new server and consolidate all tool registrations there

### Phase 4: Strip old plugins

11. Disable/remove these extension directories (they become dead code):
    - `extensions/mcp-server/` — the old Python server
    - `extensions/file-tools/` — was just a proxy
    - `extensions/run-bash/` — was just a proxy
    - `extensions/web-local/` — was just a proxy
12. Simplify `extensions/memory-graph/` — either becomes the unified plugin or gets removed if replaced by `mcp-tools`
13. Remove `memory-core` from `plugins.allow` in openclaw.json (if present) to eliminate the tool name conflicts

### Phase 5: Additional mitigations

14. **Subprocess restart backoff** — add exponential backoff in McpStdioClient when subprocess crashes (prevent crash loops)
15. **Graceful pipe handling** — add try/catch around `sendRaw()` in McpStdioClient for EPIPE errors
16. **Health check** — add a `ping` tool or use MCP's built-in ping to detect dead connections before tool calls
17. **Prefill hook resilience** — ensure the `before_prompt_build` hook has a hard timeout so a hung MCP server can't block the session forever

---

## Key Constants to Preserve

```python
# Paths
VAULT = Path.home() / "obsidian"
QMD = Path.home() / ".bun/bin/qmd"
PREFILL_LOG = Path.home() / ".openclaw/logs/timing.jsonl"

# GLM (local Qwen for keyword extraction)
GLM_URL = "http://127.0.0.1:8091/v1/chat/completions"
GLM_MODEL = "Qwen3-30B-A3B-Instruct-2507"

# Prefill tuning
PREFILL_TIMEOUT_S = 2.0
PREFILL_CACHE_TTL_S = 60.0
PREFILL_REFRESH_S = 600  # 10 min index refresh
MAX_CONTEXT_CHARS = 8000
W_VECTOR = 0.55, W_TAG = 0.35, W_CROSS = 0.10

# File safety
FILE_MAX_READ_BYTES = 2_000_000
MAX_FILE_SIZE = 512 * 1024

# Web
WEB_TIMEOUT_S = 15.0
WEB_MAX_RESPONSE_BYTES = 2_000_000
```

---

## Dependencies (for pyproject.toml)

```toml
# Add to existing lloyd-services dependencies
pyyaml >= 6.0
ddgs >= 7.0
readability-lxml >= 0.8
html2text >= 2024.0
# Already present: mcp[cli], httpx
```

---

## Verification

1. **Startup test:** `cd ~/Projects/lloyd-services && uv run openclaw_mcp_server.py` — should print vault scan stats to stderr
2. **Tool smoke test:** Send JSON-RPC `tools/call` for each tool over stdin
3. **Integration test:** Restart gateway, send a chat message, verify:
   - Prefill context appears in logs
   - Memory tools work (tag_search, mem_search, mem_get)
   - Web tools work (http_search, http_fetch)
   - File tools work (file_read, file_glob)
   - run_bash works
4. **Crash resilience:** Kill the MCP server process mid-call, verify the gateway session doesn't get stuck
5. **No duplicate warnings:** Verify `[plugins] plugin tool name conflict` messages are gone from logs

---

## Status: COMPLETE (2026-02-28)

All 16 tools migrated to `~/Projects/lloyd-services/openclaw_mcp_server.py`. Single unified plugin at `extensions/mcp-tools/` proxies all tools through one MCP subprocess. Old plugins disabled (renamed to `.disabled`): file-tools, run-bash, web-local, memory-graph, mcp-server. Voice-tools updated to import from mcp-tools and corrected server path.

**Verified:**
- [x] Standalone server starts cleanly (303 docs, 164 tags indexed)
- [x] Gateway loads all plugins without errors or tool name conflicts
- [x] mcp-tools: 16 tools + prefill hook registered
- [x] voice-tools: 3 tools + message_sending hook registered
- [x] McpStdioClient has exponential restart backoff, EPIPE handling, 5s prefill timeout
