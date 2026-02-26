# Memory Tools → MCP Conversion

**Date:** 2026-02-26
**Type:** Feature

## Summary

Expose all five OpenClaw memory tools as a standalone Python MCP server so Claude Code (and other MCP-compatible clients) can use them directly. OpenClaw continues to use its native plugin tools in Phase 1; a proxy integration is planned for Phase 2 to unify both on the same canonical server.

---

## Motivation

Memory tools (tag_search, tag_explore, vault_overview, memory_search, memory_get) existed only inside OpenClaw's agent environment. Making them available via MCP enables:
- Use from Claude Code sessions (this project)
- Use from Claude Desktop or other MCP clients
- A single canonical implementation shared by all consumers (Phase 2)

---

## Architecture

```
extensions/memory-mcp-server/
    server.py          ← FastMCP server, 5 tools
    requirements.txt   ← mcp[cli], httpx, pyyaml

    tag_search/tag_explore/vault_overview:
        Walk ~/obsidian, parse YAML frontmatter, in-process tag index

    memory_search / memory_get:
        HTTP proxy → OpenClaw gateway at 127.0.0.1:18789
```

Claude Code spawns `python3 server.py` inside the `lloyd` distrobox container via stdio transport. The container shares the host network namespace, so the gateway port is reachable.

---

## Files

| File | Action |
|------|--------|
| `extensions/memory-mcp-server/server.py` | Created |
| `extensions/memory-mcp-server/requirements.txt` | Created |
| `~/.claude/settings.json` | Modified (added mcpServers) |

---

## Key Decisions

- **FastMCP** (`mcp.server.fastmcp`): simpler than raw `mcp.Server`; auto-generates JSON Schema from Python type hints
- **Python frontmatter parsing**: uses `yaml.safe_load` on `---`-fenced blocks; self-contained, no dependency on TypeScript tag-index.ts
- **Tag index in-process**: built on startup from vault scan; no refresh in Phase 1 (restart server to update)
- **Gateway proxy for memory_search/memory_get**: avoids reimplementing QMD vector search; requires OpenClaw gateway to be running
- **Run inside lloyd**: ensures gateway at `127.0.0.1:18789` is reachable; vault at `~/obsidian` is accessible
- **Stdio transport**: standard for MCP; Claude Code spawns server as a subprocess

---

## Claude Code Config

In `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "openclaw-memory": {
      "type": "stdio",
      "command": "distrobox-enter",
      "args": [
        "lloyd", "--",
        "python3",
        "/home/alansrobotlab/.openclaw/extensions/memory-mcp-server/server.py"
      ]
    }
  }
}
```

---

## Phase 2 (planned)

Update `extensions/memory-graph/index.ts` to:
- Remove native `api.registerTool()` registrations for tag_search/tag_explore/vault_overview
- Add `mcp-client.ts`: minimal JSON-RPC 2.0 stdio client (zero external deps)
- Proxy tool calls through the Python MCP server subprocess

This creates a single canonical implementation shared by both Claude Code and OpenClaw.

---

## Testing

1. Standalone server test via raw JSON-RPC or `mcp dev server.py`
2. Claude Code: `/mcp` shows 5 tools under `openclaw-memory`
3. Smoke test: tag_search, vault_overview, memory_search from Claude Code
4. Graceful degradation: memory_search returns error (not crash) when gateway is down
