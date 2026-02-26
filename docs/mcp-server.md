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

## Architecture (Phase 6 — current)

```
extensions/mcp-server/
    server.py          ← FastMCP server, 12 tools — fully standalone
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

    file_read(path, start_line=0, end_line=0):
        Direct filesystem read; path expanded via ~ then sandboxed to $HOME via resolve()
        Optional line range; 2MB size cap

    file_write(path, content):
        Write (create/overwrite) a file within $HOME; creates parent dirs automatically

    file_edit(path, old_text, new_text):
        Exact-string replacement (first occurrence); fails if 0 or >1 matches

    file_glob(pattern, root="~"):
        pathlib.Path.glob() from root; returns up to 200 matches relative to root

    file_grep(pattern, path="~", file_glob="**/*", max_results=50):
        Python re.search() across files; returns filename:lineno: content; max 200 results

extensions/memory-graph/
    mcp-client.ts      ← Zero-dep JSON-RPC 2.0 stdio client
    index.ts           ← OpenClaw plugin; McpStdioClient shared by prefill hook + 5 tool proxies
    prefill.ts         ← Unified before_prompt_build hook; calls MCP server directly via McpStdioClient
                         (memory_search with json_output=true, memory_get); in-process TypeScript
                         TagIndex used for fast synchronous tag matching only

extensions/web-local/
    index.ts           ← OpenClaw plugin; proxies web_search + web_fetch via McpStdioClient

extensions/file-tools/
    index.ts           ← OpenClaw plugin; proxies all 5 file tools via McpStdioClient

~/.claude/mcp.json     ← Claude Code config (distrobox-enter lloyd -- uv run server.py)
```

Both Claude Code and the OpenClaw agent share the same MCP server. All 12 tools have a single canonical Python implementation.

```
OpenClaw agent:
  memory tools (5) → memory-graph plugin → McpStdioClient → server.py
  web tools (2)    → web-local plugin    → McpStdioClient → server.py (separate subprocess)
  file tools (5)   → file-tools plugin   → McpStdioClient → server.py (separate subprocess)
  prefill hook     → memory-graph plugin → McpStdioClient → server.py (shared client, no gateway)

Claude Code:
  all 12 tools → ~/.claude/mcp.json → distrobox-enter → uv run server.py (same server)
```

---

## Files

| File | Action |
|------|--------|
| `extensions/mcp-server/server.py` | Created (Phase 1), updated (Phase 3, 4, 5, 6) |
| `extensions/mcp-server/requirements.txt` | Created (Phase 1), updated (Phase 3) |
| `extensions/memory-graph/mcp-client.ts` | Created (Phase 2) |
| `extensions/memory-graph/index.ts` | Modified (Phase 2, 4, 6b) |
| `extensions/memory-graph/prefill.ts` | Modified (Phase 4, 6b) |
| `extensions/memory-graph/gateway.ts` | Deleted (Phase 6b) |
| `extensions/web-local/index.ts` | Modified (Phase 5) |
| `extensions/file-tools/index.ts` | Created (Phase 6) |
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
- **Third subprocess**: file-tools spawns a third `McpStdioClient` instance — three stateless `server.py` processes run concurrently, acceptable overhead
- **file_read sandbox**: `Path(path).expanduser()` then `p.resolve()` → check `startswith(str(HOME))`; blocks `/etc/passwd` style escapes; does NOT call `is_relative_to` to allow symlinks within HOME (workspace/ symlinks to ~/obsidian/lloyd)
- **file_edit uniqueness**: rejects edits where `old_text` appears 0 or >1 times — forces caller to include more context; mirrors Claude Code's Edit tool behavior
- **file_glob via pathlib**: `root_path.glob(pattern)` — stdlib, no subprocess; returns paths relative to root, sorted, capped at 200
- **file_grep via Python re**: iterates files from `root.glob(file_glob)`, applies `re.search` per line; avoids `rg`/`grep` subprocess dependency; caps at 200 results; skips files >2MB
- **Shared McpStdioClient in memory-graph**: single `McpStdioClient` instance created before both the prefill hook and the `proxyTool()` registrations; both share the same server.py subprocess — avoids a fourth process
- **prefill hook no longer uses gateway**: `gatewayInvoke` (HTTP → 18789) replaced with direct `McpStdioClient.callTool()`; `gateway.ts` deleted; prefill now works without the gateway HTTP endpoint
- **`callToolWithAbort` helper**: wraps `McpStdioClient.callTool()` with `Promise.race` against the prefill hook's `AbortSignal` (2s timeout) so `Promise.allSettled` doesn't block for the full 10s MCP timeout when the prefill is aborted
- **`maxResults` → `max_results`**: prefill hook fixed the camelCase mismatch when calling MCP server's `memory_search` (Python uses snake_case)
- **In-process TypeScript TagIndex retained**: synchronous tag matching in the prefill hook must be instant; calling MCP server (subprocess round-trip) for this would add unacceptable latency. Both the TypeScript TagIndex and Python TagIndex exist in parallel — acceptable redundancy for correctness

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

### Phase 6 — File system tools + prefill hook fully migrated to MCP

**6a — File tools**: Added `file_read`, `file_write`, `file_edit`, `file_glob`, `file_grep` to `server.py` (12 tools total). No new uv dependencies — all implemented with Python stdlib (`pathlib`, `re`). Created `extensions/file-tools/index.ts` + `openclaw.plugin.json` as a thin `McpStdioClient` proxy for all 5 tools. Path safety: all paths expanded via `~` then verified via `resolve()` to stay within `$HOME`. All 5 tools available to Claude Code (via `~/.claude/mcp.json`) and the OpenClaw agent (via file-tools plugin).

**6b — Prefill hook gateway removal**: Replaced `gatewayInvoke` (HTTP to `:18789`) in `prefill.ts` with direct `McpStdioClient.callTool()` calls. The single `McpStdioClient` instance is now created first in `index.ts` and passed to both the prefill hook and the `proxyTool()` registrations — one shared server.py subprocess. Added `callToolWithAbort` helper to properly propagate the 2s abort signal. Fixed `maxResults` → `max_results` snake_case. Removed `gateway.ts`. The prefill hook now requires no gateway HTTP endpoint — fully standalone with the MCP server.

---

## Testing

```bash
# Verify all 12 tools listed
distrobox-enter lloyd -- bash -c "
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null | tail -1 | python3 -c 'import json,sys; [print(t[\"name\"]) for t in json.load(sys.stdin)[\"result\"][\"tools\"]]'
"
# Expect: tag_search tag_explore vault_overview memory_search memory_get web_search web_fetch file_read file_write file_edit file_glob file_grep

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

# Test file_glob
distrobox-enter lloyd -- bash -c "
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"file_glob\",\"arguments\":{\"pattern\":\"**/*.ts\",\"root\":\"~/.openclaw/extensions\"}}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null | tail -1 | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"result\"][\"content\"][0][\"text\"])'
"
# Expect: list of .ts files relative to extensions/

# Test file_read with line range
distrobox-enter lloyd -- bash -c "
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"file_read\",\"arguments\":{\"path\":\"~/.openclaw/openclaw.json\",\"start_line\":1,\"end_line\":5}}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null | tail -1 | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"result\"][\"content\"][0][\"text\"])'
"
# Expect: first 5 lines of openclaw.json

# Test path escape guard
distrobox-enter lloyd -- bash -c "
  printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"file_read\",\"arguments\":{\"path\":\"/etc/passwd\"}}}\n' | \
    uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py 2>/dev/null | tail -1 | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"result\"][\"content\"][0][\"text\"])'
"
# Expect: Error: path escapes home directory: '/etc/passwd'

# Restart gateway and verify startup logs
distrobox-enter lloyd -- systemctl --user start openclaw-gateway.service
distrobox-enter lloyd -- journalctl --user -u openclaw-gateway.service -n 30 --no-pager -o cat | grep -E "web-local|file-tools"
# Expect:
#   web-local: registering web_search + web_fetch (proxied through MCP server)
#   file-tools: registering file_read, file_write, file_edit, file_glob, file_grep (proxied through MCP server)
```
