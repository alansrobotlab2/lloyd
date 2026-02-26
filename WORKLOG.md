# OpenClaw Work Log

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
