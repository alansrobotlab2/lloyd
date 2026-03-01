# OpenClaw Work Log

## 2026-03-01 — Migrate Memory Tools to MCP-Owned

**Type:** Feature / Migration
**Files modified:** `~/Projects/lloyd-services/tool_services.py`, `extensions/mcp-tools/index.ts`, `openclaw.json`

- **Goal:** Take ownership of `memory_search` and `memory_get` tools — move from built-in memory-core (QMD) to MCP-owned versions in tool_services.py
- **memory_search:** Now returns structured JSON `{results: [{path, score, snippet, startLine, endLine, source, citation}], mode}`. Switched from `qmd query` (CUDA, 2.5min cold start) to `qmd search` (BM25, 0.3s). Added `min_score` param
- **memory_get:** Now returns JSON `{path, text}` instead of raw text. Changed `end_line` to `num_lines` (count-based, matching built-in's `from`/`lines` pattern)
- **Disabled built-in:** Removed `memory` section from openclaw.json entirely (prevents memory-core from registering duplicate tools)
- **Tested:** Both tools work through MCP SSE proxy, gateway has no tool conflicts, agent successfully calls MCP versions

---

## 2026-02-28 — Fix Workspace Bootstrap File Loading

**Type:** Bugfix
**Files modified:** `~/obsidian/lloyd/{soul,agents,tools,identity,user,heartbeat,memory,workflow_auto}.md` (replaced with symlinks)

- **Root cause:** OpenClaw's `openVerifiedFileSync` rejects files with `nlink > 1` (hardlink security check, `rejectHardlinks: true` default in `openBoundaryFileSync`). Workspace bootstrap files were hardlinked between `~/.openclaw/workspace/` and `~/obsidian/lloyd/`, causing all 7 files to be silently marked `missing: true` with `rawChars: 0`
- **Fix:** Replaced obsidian-side hardlinks with symlinks pointing to workspace files (`ln -s ~/.openclaw/workspace/SOUL.md ~/obsidian/lloyd/soul.md` etc). Workspace files now have `nlink=1` (pass security check), Obsidian accesses content via symlinks
- **Verified:** After gateway restart, `systemPromptReport.injectedWorkspaceFiles` shows all 7 files with `missing: false` and correct `rawChars`. System prompt grew from ~19k chars to ~43k chars (24k of workspace context restored)
- **Key code:** `agent-scope-DZ5jocOf.js` lines 498/512 (`nlink > 1` check), line 578 (`rejectHardlinks ?? true` default)

---

## 2026-02-28 — MCP Tool Migration: 5 Subprocesses → 1

**Type:** Bugfix / Refactor
**Files created:** `~/Projects/lloyd-services/openclaw_mcp_server.py`, `extensions/mcp-tools/index.ts`, `extensions/mcp-tools/mcp-client.ts`, `extensions/mcp-tools/openclaw.plugin.json`, `docs/mcp-tool-migration.md`
**Files modified:** `~/Projects/lloyd-services/pyproject.toml`, `extensions/voice-tools/index.ts`
**Files disabled:** `extensions/file-tools.disabled/`, `extensions/run-bash.disabled/`, `extensions/web-local.disabled/`, `extensions/memory-graph.disabled/`, `extensions/mcp-server.disabled/`

- **Root cause:** Chat window stuck permanently due to cascading `anyio.ClosedResourceError` crashes — 5 plugins each spawning independent `server.py` subprocesses with racing `process.on("exit")` handlers
- **Fix:** Consolidated all 16 MCP tools into single `openclaw_mcp_server.py` in lloyd-services, proxied via one `mcp-tools` plugin instead of five separate plugins
- **Mitigations added:** Exponential restart backoff (500ms→30s) in McpStdioClient, EPIPE handling in `sendRaw()`, hard 5s timeout on prefill hook
- **Also fixed:** voice-tools import path (was pointing to disabled memory-graph) and server path (`lloyd` → `lloyd-services`)
- **Verified:** Gateway starts clean — 16 tools + 3 voice tools + prefill hook, no tool name conflict warnings

---

## 2026-02-26 — Base Tool Set Expansion (run_bash, http_request, memory_write)

**Type:** Feature
**Files modified:** `extensions/mcp-server/server.py`, `extensions/memory-graph/index.ts`, `extensions/memory-graph/mcp-client.ts`
**Files created:** `extensions/run-bash/index.ts`, `extensions/run-bash/openclaw.plugin.json`

Added three new tools to complete the base MCP tool set accessible from any agent:
- `run_bash` — shell command execution sandboxed to $HOME, max 120s; unlocks git, tests, builds, package managers
- `http_request` — full HTTP client (GET/POST/PUT/PATCH/DELETE/HEAD) with raw response; loopback allowed for local services; blocks other private IPs
- `memory_write` — CRUD completion for Obsidian vault (create/overwrite by vault-relative path)
- Fixed `mcp-client.ts:callTool()` to accept optional `timeoutMs` parameter (was hardcoded 10s); `run-bash` plugin passes 120s
- New `run-bash` plugin auto-discovered from `extensions/run-bash/`; gateway logs show clean load
- Server tool count: 13 → 16; memory-graph proxies: 5 → 6

---

## 2026-02-26 — Phase 7: Prefill Pipeline Migrated to Python MCP Tool

**Type:** Refactor / Feature
**Files modified:** `extensions/mcp-server/server.py`, `extensions/memory-graph/prefill.ts`, `extensions/memory-graph/index.ts`
**Files deleted:** `extensions/memory-graph/tag-index.ts`, `extensions/memory-graph/scanner.ts`, `extensions/memory-graph/query.ts`, `extensions/memory-graph/format.ts`

Moved the full context-prefetch pipeline from TypeScript into a dedicated `prefill_context` MCP tool in `server.py`:

- **`server.py`**: Added constants, LRU cache (60s), query helpers, async GLM extractor (`httpx.AsyncClient`), `_Candidate` dataclass, `_run_tag_match`, `_merge_and_rank`, `_format_unified_context`, timing log, background 10-min index refresh loop (via FastMCP `lifespan`), and the `prefill_context` async MCP tool. Parallelism preserved via `asyncio.gather` + `asyncio.to_thread`; 2s hard timeout via `asyncio.timeout`.
- **`prefill.ts`**: Reduced from 410 lines to 35 — now a thin wrapper that calls `prefill_context` via McpStdioClient and returns `{ prependContext }`.
- **`index.ts`**: Removed TagIndex, scanVault, refreshLock, setInterval, buildIndex/refreshIndex, getTagIndex. Simplified to MCP client + hook registration + 5 proxyTool calls.
- **Deleted**: 4 TypeScript files (~850 lines total) that are now dead code: `tag-index.ts`, `scanner.ts`, `query.ts`, `format.ts`.
- **Verified**: Gateway starts cleanly with `memory-graph v4: registered (Python-delegated prefill + 5 tools via MCP)`. Python now the single source of truth for all search logic.

## 2026-02-26 — MCP Server Phase 4: All 5 Tools via MCP in OpenClaw

**Type:** Feature
**Files modified:** `extensions/memory-graph/index.ts`, `extensions/mcp-server/server.py`, `extensions/memory-graph/prefill.ts`

Completed tool unification — all 5 memory tools now route through the MCP server subprocess in OpenClaw:

- **`index.ts`**: added `proxyTool()` calls for `memory_search` and `memory_get`; plugin registration silently overrides the built-in `memory-core` tools (no collision errors); startup log now reads "5 tools via MCP"
- **`server.py`**: added `json_output: bool = False` to `memory_search`; when `True`, returns `{"results": [{path, score, snippet}]}` JSON instead of human-readable text
- **`prefill.ts`**: both `memory_search` calls now pass `json_output: true`; result parsing handles both built-in QMD format (`details.results`) and MCP plugin format (`content[].text` parsed as JSON) for graceful fallback compatibility
- **Verified**: gateway HTTP test confirms both tools return MCP-backed results; prefill vector search preserved via `json_output` JSON path

## 2026-02-26 — MCP Server Phase 3: Fully Standalone Tools

**Type:** Feature / Refactor
**Files modified:** `extensions/mcp-server/server.py`, `extensions/mcp-server/requirements.txt`

Decoupled the MCP server from the OpenClaw gateway entirely:

- **memory_search**: replaced httpx proxy with `subprocess.run([qmd, "search", ..., "--json"])` — BM25 search via `~/.bun/bin/qmd`; strips `qmd://obsidian/` prefix from paths and `@@...@@` diff markers from snippets
- **memory_get**: replaced httpx proxy with direct `(VAULT / path).read_text()` — includes path traversal check and optional line range slicing
- **Removed**: `httpx` from deps/imports, `GATEWAY_URL` constant, `_gateway_invoke()` function
- **Verified**: both tools return correct results with gateway explicitly stopped

## 2026-02-26 — MCP Server Phase 2: OpenClaw Plugin Proxy

**Type:** Feature
**Files created:** `extensions/memory-graph/mcp-client.ts`
**Files modified:** `extensions/memory-graph/index.ts`

Updated the memory-graph OpenClaw plugin to consume the MCP server instead of running inline tool implementations:

- **McpStdioClient** (`mcp-client.ts`): zero-dependency JSON-RPC 2.0 stdio client; spawns `uv run server.py` on first tool call, handles initialize handshake, routes calls with 10s timeout
- **index.ts**: removed 165 lines of inline tag_search/tag_explore/vault_overview implementations; replaced with `proxyTool()` helper that delegates all 3 to McpStdioClient; prefill hook and tagIndex unchanged
- MCP server subprocess path resolved relative to mcp-client.ts via `import.meta.url`
- Gateway restarted; confirmed `"3 tools via MCP"` in startup log

## 2026-02-26 — Memory Tools MCP Server (Phase 1)

**Type:** Feature
**Files created:** `extensions/mcp-server/server.py`, `extensions/mcp-server/requirements.txt`, `~/.claude/mcp.json`
**Output:** [docs/mcp-server.md](docs/mcp-server.md)

Built a standalone Python MCP server exposing all 5 memory tools via the MCP stdio protocol:

- **FastMCP** (`mcp[cli]`) with inline script deps — `uv run server.py` handles all deps automatically
- **Tag tools** (tag_search, tag_explore, vault_overview): Python vault scanner + in-process tag index; ports TypeScript logic from memory-graph/scanner.ts and tag-index.ts; 286 docs / 144 tags indexed
- **Gateway proxy tools** (memory_search, memory_get): thin `httpx` wrappers to OpenClaw gateway at `127.0.0.1:18789`
- **Claude Code config**: `~/.claude/mcp.json` with `distrobox-enter lloyd -- uv run server.py` so gateway is reachable
- Phase 2 (OpenClaw plugin proxy via mcp-client.ts) deferred

## 2026-02-24 — Context Fill Pipeline Evaluation

**Type:** Analysis
**Files examined:** `extensions/memory-prefetch/index.ts`, `extensions/memory-graph/index.ts`, `extensions/memory-graph/tag-index.ts`, `extensions/memory-graph/format.ts`, `extensions/memory-graph/scanner.ts`
**Output:** [docs/context-fill-pipeline-evaluation.md](docs/context-fill-pipeline-evaluation.md)

Mapped the full `before_prompt_build` pipeline where memory-prefetch (priority 0) and memory-graph (priority 200) both inject context before the first LLM call. Key findings:

- The two plugins are **complementary** (vector search vs tag matching) but produce **overlapping documents** without deduplication
- memory-graph has **no size cap** — combined injection can exceed 10k+ chars
- Both plugins perform **duplicate keyword extraction** (same filler removal, same envelope stripping)
- memory-prefetch uses **HTTP gateway round-trips** to call memory_search/memory_get instead of direct invocation
- memory-graph injects **metadata only** (no content), so tag-only matches still require a follow-up tool call

No code changes made. Improvement options documented for future implementation.

## 2026-02-24 — Unified Context Prefill Hook (Option C)

**Type:** Feature
**Files created:** `extensions/memory-graph/query.ts`, `extensions/memory-graph/gateway.ts`, `extensions/memory-graph/prefill.ts`
**Files modified:** `extensions/memory-graph/index.ts`, `extensions/memory-graph/format.ts`, `extensions/memory-graph/openclaw.plugin.json`
**Files disabled:** `extensions/memory-prefetch/openclaw.plugin.json` → `.disabled`
**Output:** [docs/context-fill-pipeline-evaluation.md](docs/context-fill-pipeline-evaluation.md)

Merged memory-prefetch and memory-graph into a single unified `before_prompt_build` hook in memory-graph v3:

- **Single hook** runs tag matching (sync, 0ms) and vector search (async) in parallel within 2s budget
- **Deduplicated** by path — each document appears once in the output
- **Unified ranking**: `finalScore = 0.55*vectorScore + 0.35*tagScore + 0.10*crossBonus`
- **Tiered output**: top 3 get full content via memory_get, next 5 get metadata/snippet only
- **Single `<memory_context>` block** under shared 8000-char budget replaces separate `<memory_prefetch>` + `<vault_context>`
- **Shared keyword extraction** in query.ts eliminates duplicate filler removal
- Tag tools (tag_search, tag_explore, vault_overview) unchanged

## 2026-02-26 — MCP Server Phase 6: File Tools + Prefill Hook Gateway Removal

**Type:** Feature / Refactor
**Files modified:** `extensions/mcp-server/server.py`, `extensions/file-tools/index.ts` (new), `extensions/file-tools/openclaw.plugin.json` (new), `extensions/memory-graph/prefill.ts`, `extensions/memory-graph/index.ts`
**Files deleted:** `extensions/memory-graph/gateway.ts`
**Docs:** `docs/mcp-server.md`

Phase 6a — Added 5 file system tools to `server.py` (12 total), all stdlib/no new deps:
- `file_read`, `file_write`, `file_edit`, `file_glob`, `file_grep` — all sandboxed to `$HOME` via `resolve()` check
- New `extensions/file-tools/` plugin (index.ts + openclaw.plugin.json) proxies all 5 via McpStdioClient

Phase 6b — Fully removed gateway dependency from the memory-graph prefill hook:
- `prefill.ts`: replaced all `gatewayInvoke` HTTP calls (→ `:18789`) with direct `McpStdioClient.callTool()`
- Added `callToolWithAbort` helper so the 2s abort signal properly cancels MCP calls in `Promise.allSettled`
- Fixed `maxResults` → `max_results` (snake_case for Python MCP server)
- `index.ts`: `McpStdioClient` now created before the prefill hook; shared by prefill + tool proxies (one subprocess)
- `gateway.ts` deleted — no longer referenced anywhere
