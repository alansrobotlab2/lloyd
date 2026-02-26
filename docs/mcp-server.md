# OpenClaw MCP Server

**Date:** 2026-02-26
**Type:** Feature

## Summary

Expose all five OpenClaw memory tools as a standalone Python MCP server usable by Claude Code and any other MCP-compatible client. The server is fully self-contained — no dependency on the OpenClaw gateway or any external service.

---

## Motivation

Memory tools (tag_search, tag_explore, vault_overview, memory_search, memory_get) existed only inside OpenClaw's agent environment. Making them available via MCP enables:
- Use from Claude Code sessions (this project)
- Use from Claude Desktop or other MCP clients
- A single canonical implementation shared by all consumers

---

## Architecture (Phase 4 — current)

```
extensions/mcp-server/
    server.py          ← FastMCP server, 5 tools — fully standalone
    requirements.txt   ← mcp[cli], pyyaml

    tag_search / tag_explore / vault_overview:
        Walk ~/obsidian, parse YAML frontmatter, in-process tag index (TagIndex)
        IDF-based ranking, co-occurrence graph for tag_explore

    memory_search(query, max_results, json_output=False):
        subprocess → ~/.bun/bin/qmd search <query> -c obsidian -n N --json
        BM25 full-text search, no GPU required
        json_output=True → {"results": [{path, score, snippet}]} for prefill hook

    memory_get:
        Direct read from ~/obsidian/<path>
        Optional line range (start_line, end_line)
        Path traversal guard (is_relative_to check)

extensions/memory-graph/
    mcp-client.ts      ← Zero-dep JSON-RPC 2.0 stdio client
    index.ts           ← OpenClaw plugin; proxyTool() for ALL 5 tools via McpStdioClient
    prefill.ts         ← Unified before_prompt_build hook; uses json_output=true for
                         structured memory_search results; dual-format parser for compatibility

~/.claude/mcp.json     ← Claude Code config (distrobox-enter lloyd -- uv run server.py)
```

Both Claude Code and the OpenClaw agent share the same MCP server subprocess. All 5 tools have a single canonical implementation.

```
OpenClaw agent:
  all 5 tools → McpStdioClient → server.py (MCP, standalone)
  prefill hook → gatewayInvoke → plugin tool → McpStdioClient → server.py

Claude Code:
  all 5 tools → ~/.claude/mcp.json → distrobox-enter → uv run server.py (same server)
```

---

## Files

| File | Action |
|------|--------|
| `extensions/mcp-server/server.py` | Created (Phase 1), updated (Phase 3, 4) |
| `extensions/mcp-server/requirements.txt` | Created (Phase 1), updated (Phase 3) |
| `extensions/memory-graph/mcp-client.ts` | Created (Phase 2) |
| `extensions/memory-graph/index.ts` | Modified (Phase 2, 4) |
| `extensions/memory-graph/prefill.ts` | Modified (Phase 4) |
| `~/.claude/mcp.json` | Created (Phase 1) |

---

## Key Decisions

- **FastMCP** (`mcp.server.fastmcp`): simpler than raw `mcp.Server`; auto-generates JSON Schema from Python type hints
- **uv inline script deps** (`# /// script` block): `uv run server.py` auto-installs deps in an isolated venv — no pip, no manual setup
- **Python frontmatter parsing**: `yaml.safe_load` on `---`-fenced blocks; self-contained, mirrors TypeScript scanner.ts logic
- **Tag index in-process**: built on startup from vault scan; ~280 docs / ~140 tags, sub-second startup
- **memory_search via qmd BM25**: `qmd search` (BM25) works without GPU; `qmd vsearch` (vector) requires CUDA + 1.28GB model — not used
- **memory_get via filesystem**: direct `(VAULT / path).read_text()`; simpler and more reliable than gateway proxy; `qmd get` by path is broken (returns "Document not found")
- **Run inside lloyd**: container shares host network namespace; vault at `~/obsidian` and `qmd` at `~/.bun/bin/qmd` are accessible
- **McpStdioClient**: zero-dependency TypeScript JSON-RPC 2.0 client in OpenClaw plugin; spawns server subprocess on first tool call, handles initialize handshake, 10s timeout per call
- **Plugin overrides built-in**: registering `memory_search`/`memory_get` via `api.registerTool()` silently overrides the built-in `memory-core` tools — no collision errors, last-registration-wins
- **`json_output` param**: `memory_search` accepts `json_output=True` to return structured `{"results":[...]}` JSON; the prefill hook uses this to get parseable data while the LLM sees human-readable output
- **Dual-format parser in prefill**: `prefill.ts` handles both the old QMD gateway format (`details.results`) and the new MCP plugin format (`content[].text` parsed as JSON) for graceful fallback compatibility

---

## Claude Code Config (`~/.claude/mcp.json`)

```json
{
  "mcpServers": {
    "openclaw": {
      "command": "distrobox-enter",
      "args": ["lloyd", "--", "uv", "run",
        "/home/alansrobotlab/.openclaw/extensions/mcp-server/server.py"]
    }
  }
}
```

---

## Phase History

### Phase 1 — Python MCP server
Created `extensions/mcp-server/server.py` with all 5 tools. Tag tools fully in-process. `memory_search` and `memory_get` proxied to OpenClaw gateway at `127.0.0.1:18789` via httpx.

### Phase 2 — OpenClaw plugin proxy
Updated `extensions/memory-graph/index.ts` to remove 165 lines of inline tool implementations. Added `mcp-client.ts` (zero-dep JSON-RPC 2.0 stdio client). Plugin now proxies tag_search, tag_explore, vault_overview through the same MCP server subprocess via `proxyTool()`.

### Phase 3 — Fully standalone
Removed httpx, `GATEWAY_URL`, and `_gateway_invoke()`. `memory_search` replaced with `subprocess.run([qmd, "search", ..., "--json"])`. `memory_get` replaced with direct filesystem read. Server works with OpenClaw gateway stopped. All 5 tools are now self-contained.

### Phase 4 — All 5 tools via MCP in OpenClaw
Added `memory_search` and `memory_get` to `proxyTool()` in `index.ts` — plugin registration silently overrides the built-in `memory-core` tools. Added `json_output` param to `memory_search` in `server.py`. Updated `prefill.ts` to pass `json_output=true` and parse the MCP plugin response format. All 5 tools now share a single canonical implementation for both Claude Code and OpenClaw. Gateway startup log: "5 tools via MCP".

---

## Testing

```bash
# Verify standalone — stop gateway first
distrobox-enter lloyd -- bash -c "
  systemctl --user stop openclaw-gateway.service
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"memory_search\",\"arguments\":{\"query\":\"alfie arm\"}}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null
"
# Expect: BM25 results, no gateway error

# Restart gateway
distrobox-enter lloyd -- systemctl --user start openclaw-gateway.service
```
