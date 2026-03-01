# OpenClaw → IronClaw Migration

**Status:** Planning — not yet started
**Last updated:** 2026-02-27

---

## Overview

IronClaw is a Rust-based reimplementation of OpenClaw by NEAR AI, focused on security and
local data ownership. It provides WASM-sandboxed tools, a single native binary, native MCP
protocol support, and PostgreSQL+pgvector for memory storage.

**Migration goal:** Full cutover. IronClaw connects to the existing Python MCP servers
(already running as independent services) rather than managing them itself.

---

## Architecture Comparison

| Aspect | OpenClaw (current) | IronClaw (target) |
|--------|-------------------|-------------------|
| Language | TypeScript / Node.js | Rust |
| Tool sandboxing | Docker | WASM (capability-based) |
| Memory storage | SQLite + QMD (Obsidian vault) | PostgreSQL 15+ with pgvector |
| Plugin system | TypeScript hooks (`api.on()`) | WASM tools + MCP clients |
| MCP support | Plugins spawn MCP subprocesses | Native MCP client (connects to running servers) |
| Binary | Node.js app | Single native binary |
| Channels | Gateway on `:18789` | REPL, HTTP webhooks, WASM channels, web gateway |
| Default provider | OpenRouter → Sonnet 4.6 | OpenAI-compatible (any provider) |

---

## What Transfers (near-zero rewrite)

All 19 custom tools are implemented in two Python MCP servers. Since IronClaw is a native
MCP client, these carry over by simply pointing IronClaw at the running server endpoints.

### MCP Server 1 — `extensions/mcp-server/server.py` (16 tools)

| Tool | Category |
|------|----------|
| `tag_search` | Memory |
| `tag_explore` | Memory |
| `vault_overview` | Memory |
| `memory_search` | Memory |
| `memory_get` | Memory |
| `memory_write` | Memory |
| `web_search` | Web |
| `web_fetch` | Web |
| `http_request` | Web |
| `file_read` | Files |
| `file_write` | Files |
| `file_edit` | Files |
| `file_glob` | Files |
| `file_grep` | Files |
| `run_bash` | Shell |
| `prefill_context` | Memory prefill |

### MCP Server 2 — `~/Projects/lloyd/voice_mcp_server.py` (3 tools)

| Tool | Purpose |
|------|---------|
| `voice_last_utterance` | Get last transcribed voice input with speaker ID |
| `voice_enroll_speaker` | Register a speaker profile |
| `voice_list_speakers` | List all enrolled speaker profiles |

---

## What Needs Adaptation

### Provider / Model Configuration

Map `openclaw.json` providers to IronClaw's config format. All three providers use
OpenAI-compatible APIs so they should map directly.

| Provider | Base URL | Key Models |
|----------|----------|------------|
| `local-llm` | `http://127.0.0.1:8091/v1` | Qwen3.5-35B-A3B (32k ctx) |
| `anthropic` (via OpenRouter or native) | `https://openrouter.ai/api/v1` | claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6 |
| `openrouter` | `https://openrouter.ai/api/v1` | Qwen3 Next 80B A3B, Sonnet 4.6 |

Default model: `claude-sonnet-4-6` (no regression from current behavior).
Auth: the existing `anthropic:manual` API key in `agents/main/agent/auth-profiles.json`.

### Memory Prefill

OpenClaw's `memory-graph` plugin injects relevant vault context before each LLM call via the
`before_prompt_build` hook, which calls the `prefill_context` MCP tool (GLM keyword
extraction + IDF-ranked BM25, 8000-char budget).

IronClaw has its own native memory search (PostgreSQL + pgvector). Two options:

1. **Validate IronClaw's native memory** — if it auto-injects relevant notes with comparable
   quality, nothing more is needed.
2. **Explicit prefill via routine** — if quality is insufficient, write a minimal IronClaw
   routine that calls the `prefill_context` MCP tool and injects the result as a system-prompt
   prefix at the start of each conversation.

Validate by asking a few memory-sensitive questions and comparing output quality before
committing to either path.

### Timing & Observability

OpenClaw's `timing-profiler` plugin uses 5 fine-grained hooks (`llm_input`, `llm_output`,
`before_tool_call`, `after_tool_call`, `agent_end`) to write per-run metrics to
`logs/timing.jsonl` with this schema:

```jsonl
{"ts":"...","event":"run_end","runId":"...","sessionId":"...","totalMs":...,"llmMs":...,"toolMs":...,"overheadMs":...,"roundTrips":...,"toolCallCount":...,"toolCalls":[...],"success":true}
```

IronClaw has built-in audit logging. Steps:
1. Confirm IronClaw's audit log format and what events it captures (run duration, tool calls,
   model, tokens).
2. If it covers the same metrics: update `tools/timing-profile.js` to read the new log path
   and format.
3. If per-tool timing granularity is lost: accept the coarser format or revisit after initial
   migration is stable.

### Multi-Tier Model Routing

The routing plugin (`extensions/model-router/`) was designed but **not yet implemented** in
OpenClaw — so there is no regression from skipping it initially. See
`docs/multi-tier-model-routing.md` for the full design.

For IronClaw, options in priority order:
1. **Default only (no regression)** — set Sonnet 4.6 as default. Ship the migration first.
2. **Manual overrides** — if IronClaw supports slash commands or per-message model selection,
   implement `/fast` (local Qwen), `/deep` (Opus) as follow-on work.
3. **Rule-based router** — port the Stage 1+2 heuristics from `multi-tier-model-routing.md`
   if IronClaw exposes a pre-prompt hook or scripting API.

---

## What Is Lost

These OpenClaw-specific mechanisms have no IronClaw equivalent:

| Feature | Status |
|---------|--------|
| TypeScript `api.on()` / `api.registerTool()` plugin hook API | Not present in IronClaw |
| `openclaw.plugin.json` plugin manifests | Replaced by IronClaw tool/MCP config |
| `mcp-client.ts` (zero-dep JSON-RPC stdio client) | Not needed — IronClaw is its own MCP client |
| `memory-prefetch` plugin (explicit gateway prefetch) | Superseded by IronClaw native memory or prefill routine |
| `before_model_resolve` hook (planned model router) | No direct equivalent; use IronClaw provider config instead |

The TypeScript extensions (`extensions/*/index.ts`) become obsolete after migration. They
can be kept for reference but will no longer run.

---

## Migration Phases

### Phase 0 — Research IronClaw Config Format

IronClaw is not yet installed. Before any configuration work, fetch actual documentation:

- Config file format (TOML / JSON / YAML?) and provider declaration syntax
- How MCP client connections are declared (HTTP URL? stdio? SSE endpoint?)
- Whether a pre-prompt system-prompt injection API exists (for memory prefill)
- What audit/timing logging is available and its log format
- Multi-provider model routing API, if any
- Install command and prerequisites

**Source:** `https://github.com/nearai/ironclaw` README and docs/

---

### Phase 1 — Install IronClaw + PostgreSQL

```bash
# Option A — Official installer (inside lloyd distrobox)
distrobox-enter lloyd -- bash -c \
  "curl --proto '=https' --tlsv1.2 -LsSf <installer-url> | sh"

# Option B — Cargo
distrobox-enter lloyd -- cargo install ironclaw

# PostgreSQL 15+ with pgvector (if not already present)
# (distro-specific — confirm packages for lloyd's base OS)
```

After install:
- Verify `ironclaw` binary in PATH
- Run `ironclaw init` (or equivalent) to generate initial config
- Confirm PostgreSQL service running with pgvector extension enabled

---

### Phase 2 — Configure Providers and Default Model

Edit IronClaw config to declare all three providers and set Sonnet 4.6 as default.
API keys: copy from `agents/main/agent/auth-profiles.json`.

```
# Providers to configure:
# 1. local-llm   http://127.0.0.1:8091/v1   (Qwen3.5-35B-A3B)
# 2. openrouter   https://openrouter.ai/api/v1 (Sonnet 4.6, Qwen3 Next)
# 3. anthropic    native or via OpenRouter      (Haiku 4.5, Sonnet 4.6, Opus 4.6)
# Default: claude-sonnet-4-6
```

---

### Phase 3 — Start MCP Servers as Persistent Services

The MCP servers must be running before IronClaw connects to them.

**server.py** currently runs as a stdio subprocess spawned by OpenClaw. For standalone
operation, switch to FastMCP's HTTP/SSE transport:

```bash
# FastMCP HTTP mode (confirm flag syntax from FastMCP docs)
uv run extensions/mcp-server/server.py --transport sse --port 18790
```

Create a `systemd --user` service unit so it starts automatically:

```ini
# ~/.config/systemd/user/openclaw-mcp.service
[Unit]
Description=OpenClaw MCP Server
After=network.target

[Service]
ExecStart=distrobox-enter lloyd -- uv run /home/alansrobotlab/.openclaw/extensions/mcp-server/server.py --transport sse --port 18790
Restart=on-failure

[Install]
WantedBy=default.target
```

Do the same for `voice_mcp_server.py` on a separate port (e.g. 18791).

---

### Phase 4 — Connect MCP Servers in IronClaw Config

Declare both MCP server endpoints in IronClaw's config (exact format TBD from Phase 0):

```
# MCP server 1 — memory, web, file, bash tools (16 tools)
# endpoint: http://127.0.0.1:18790/sse  (or whatever transport is confirmed)

# MCP server 2 — voice tools (3 tools)
# endpoint: http://127.0.0.1:18791/sse
```

**Verify:** Run `ironclaw tools list` (or equivalent) — should show all 19 tools.

---

### Phase 5 — Validate Memory Prefill Quality

1. Ask 3–5 memory-sensitive questions (topics with known vault notes).
2. Compare context quality to OpenClaw responses.
3. If IronClaw's native memory injection is comparable: no further action needed.
4. If quality is noticeably worse: write a minimal IronClaw routine that calls
   `prefill_context` (the MCP tool in server.py) and prepends the result to each
   conversation's system prompt.

---

### Phase 6 — Validate Timing / Observability

1. Run a few multi-tool sessions in IronClaw.
2. Inspect IronClaw's audit log location and format.
3. Compare against current `timing.jsonl` schema.
4. Update or rewrite `tools/timing-profile.js` to work with the new format.
5. Accept any loss of granularity that can't be recovered cheaply.

---

### Phase 7 — Parallel Run & Validation

Run IronClaw alongside OpenClaw for 2–3 days:
- Daily workflows: confirm no missing tools or degraded quality
- Memory: verify vault notes are surfaced correctly
- Voice: test speaker identification round-trip
- Model: confirm default Sonnet 4.6 routes correctly

---

### Phase 8 — Cutover

Once parallel run is stable:

```bash
# Stop OpenClaw gateway
distrobox-enter lloyd -- bash -c "kill \$(lsof -ti :18789) 2>/dev/null"
distrobox-enter lloyd -- systemctl --user stop openclaw-gateway.service
distrobox-enter lloyd -- systemctl --user disable openclaw-gateway.service
```

Update `WORKLOG.md` with migration summary.

---

## Key Risks

| Risk | Mitigation |
|------|------------|
| **MCP transport mode** — `server.py` runs via stdio today; needs HTTP/SSE for standalone use | Verify FastMCP supports `--transport sse`; test before declaring Phase 3 done |
| **Memory prefill quality gap** — IronClaw native memory may produce different context injection than the tuned GLM+IDF pipeline | Phase 5 validation with real prompts; fall back to explicit `prefill_context` routine if needed |
| **Timing granularity loss** — the 5-hook profiler cannot be replicated without a hook API | Accept IronClaw's built-in audit log; assess whether per-tool timing matters enough to rebuild |
| **Container context** — MCP servers currently live inside `lloyd` distrobox; IronClaw may run on host | Decide up front: run IronClaw inside lloyd (simpler) or on host (needs port forwarding for MCP servers) |
| **Provider auth** — IronClaw defaults to NEAR AI accounts; must configure OpenRouter/Anthropic keys explicitly | Copy keys from `auth-profiles.json` during Phase 2 |

---

## Files Reference

| File | Purpose |
|------|---------|
| `extensions/mcp-server/server.py` | Core MCP server — all 16 tools |
| `~/Projects/lloyd/voice_mcp_server.py` | Voice MCP server — 3 tools |
| `openclaw.json` | Provider URLs, model IDs, auth profile, memory paths |
| `agents/main/agent/auth-profiles.json` | API keys for Anthropic / OpenRouter |
| `extensions/timing-profiler/index.ts` | Reference for metrics to replicate |
| `extensions/memory-graph/prefill.ts` | Reference for prefill behavior |
| `docs/multi-tier-model-routing.md` | Design doc for model router (follow-on work) |
| `tools/timing-profile.js` | Timing log analysis script (needs update in Phase 6) |
