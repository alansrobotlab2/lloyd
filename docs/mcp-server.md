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

## Architecture (Phase 5 — current)

```
extensions/mcp-server/
    server.py          ← FastMCP server, 7 tools — fully standalone
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

    web_search(query, count=5):
        ddgs (DuckDuckGo) → search results (title, URL, snippet)

    web_fetch(url, extract_mode="markdown", max_chars=50000):
        httpx GET → readability-lxml article extraction → html2text markdown conversion
        Private IP blocking, 2MB body cap, 15s timeout

extensions/memory-graph/
    mcp-client.ts      ← Zero-dep JSON-RPC 2.0 stdio client
    index.ts           ← OpenClaw plugin; proxyTool() for all 5 memory tools via McpStdioClient
    prefill.ts         ← Unified before_prompt_build hook; uses json_output=true for
                         structured memory_search results; dual-format parser for compatibility

extensions/web-local/
    index.ts           ← OpenClaw plugin; proxies web_search + web_fetch via McpStdioClient

~/.claude/mcp.json     ← Claude Code config (distrobox-enter lloyd -- uv run server.py)
```

Both Claude Code and the OpenClaw agent share the same MCP server. All 7 tools have a single canonical Python implementation.

```
OpenClaw agent:
  memory tools (5) → memory-graph plugin → McpStdioClient → server.py
  web tools (2)    → web-local plugin    → McpStdioClient → server.py (separate subprocess)
  prefill hook     → memory-graph plugin → McpStdioClient → server.py

Claude Code:
  all 7 tools → ~/.claude/mcp.json → distrobox-enter → uv run server.py (same server)
```

---

## Files

| File | Action |
|------|--------|
| `extensions/mcp-server/server.py` | Created (Phase 1), updated (Phase 3, 4, 5) |
| `extensions/mcp-server/requirements.txt` | Created (Phase 1), updated (Phase 3) |
| `extensions/memory-graph/mcp-client.ts` | Created (Phase 2) |
| `extensions/memory-graph/index.ts` | Modified (Phase 2, 4) |
| `extensions/memory-graph/prefill.ts` | Modified (Phase 4) |
| `extensions/web-local/index.ts` | Modified (Phase 5) |
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
- **web_search via ddgs (DuckDuckGo)**: `googlesearch-python` scrapes Google HTML which is actively blocked; `ddgs` uses DuckDuckGo's API which works reliably without API keys; note: the old TypeScript `execFile` path was also broken (the lloyd venv had no `googlesearch` installed)
- **web_fetch via httpx + readability-lxml + html2text**: `httpx.Client` (synchronous — avoids async conflicts with FastMCP); `readability-lxml` for article extraction; `html2text` with `body_width=0` prevents hard line-wrapping that would corrupt URLs
- **Private IP blocking**: string-pattern regex, no DNS resolution — mirrors TypeScript patterns exactly (localhost, 127.x, 10.x, 172.16–31.x, 192.168.x, 169.254.x, IPv6 loopback/ULA/link-local)
- **`extractMode` → `extract_mode` translation**: TypeScript plugin keeps camelCase LLM-facing param; translates to snake_case before MCP call
- **Second subprocess**: web-local spawns its own `McpStdioClient` instance; two `server.py` subprocesses run concurrently (one from memory-graph, one from web-local) — acceptable, both are stateless

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

### Phase 5 — Web tools migrated to MCP
Added `web_search` and `web_fetch` to `server.py` (7 tools total). New uv dependencies: `ddgs`, `httpx`, `readability-lxml`, `html2text`. Switched from `googlesearch-python` (blocked by Google, was also broken in old TypeScript path) to `ddgs` (DuckDuckGo). Replaced `web-local/index.ts` inline implementation (324 lines using `execFile` + `linkedom` + `@mozilla/readability`) with a 90-line `McpStdioClient` proxy. Both web tools now share a single canonical Python implementation available to Claude Code (via `~/.claude/mcp.json`) and the OpenClaw agent (via web-local plugin proxy).

---

## Testing

```bash
# Verify all 7 tools listed
distrobox-enter lloyd -- bash -c "
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null | tail -1 | python3 -c 'import json,sys; [print(t[\"name\"]) for t in json.load(sys.stdin)[\"result\"][\"tools\"]]'
"
# Expect: tag_search tag_explore vault_overview memory_search memory_get web_search web_fetch

# Test memory_search (standalone — no gateway)
distrobox-enter lloyd -- bash -c "
  systemctl --user stop openclaw-gateway.service
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"memory_search\",\"arguments\":{\"query\":\"alfie arm\"}}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null
"
# Expect: BM25 results, no gateway error

# Test web_search
distrobox-enter lloyd -- bash -c "
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"python fastmcp\",\"count\":3}}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null | tail -1
"
# Expect: JSON with numbered search results

# Test web_fetch
distrobox-enter lloyd -- bash -c "
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"web_fetch\",\"arguments\":{\"url\":\"https://example.com\",\"max_chars\":2000}}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null | tail -1 | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"result\"][\"content\"][0][\"text\"][:300])'
"
# Expect: markdown content of example.com

# Test private IP blocking
distrobox-enter lloyd -- bash -c "
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"web_fetch\",\"arguments\":{\"url\":\"http://192.168.1.1/\"}}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null | tail -1 | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"result\"][\"content\"][0][\"text\"])'
"
# Expect: web_fetch error: Blocked — private/internal hostname "192.168.1.1"

# Restart gateway and verify web-local startup log
distrobox-enter lloyd -- systemctl --user start openclaw-gateway.service
distrobox-enter lloyd -- journalctl --user -u openclaw-gateway.service -n 30 --no-pager -o cat | grep web-local
# Expect: web-local: registering web_search + web_fetch (proxied through MCP server)
```
