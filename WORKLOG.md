# OpenClaw Work Log

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
