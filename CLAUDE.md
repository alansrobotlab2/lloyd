# Lloyd — Claude Code Context

## Project Overview

Lloyd is a fully local AI agent. It runs its own in-process agent loop (`app/harness/`) against a local vLLM server and exposes all tools through a unified MCP aggregator (`agent_mcp/`). The backend is FastAPI + SSE; the frontend is React (Vite).

- **Backend**: `server.py` (FastAPI, port 8080)
- **Frontend**: `web/` (Vite dev server, proxied through backend)
- **Config**: `config.yaml`
- **MCP aggregator**: `agent_mcp/main.py` (unified `Server("lloyd")` on `:8500/sse`)
- **Agent harness**: `app/harness/` (`run_query(messages, options)` — async generator)
- **Venv**: `.venvs/lloyd/bin/python`

## Setup

Rebuilding on a fresh OS: **[SETUP.md](SETUP.md)** is the authority — system
packages, the uv/bun/npm-global toolchain, all four venvs, supervisord + the
systemd unit, and what must be backed up first (several runtime assets are
untracked and not re-downloadable). `agent-services/setup/setup-all.sh --check`
reports what's missing without changing anything.

Secrets live in `.env` (gitignored) and reach `config.yaml` through `${VAR}`
placeholders that `app/config.py` expands at boot. Never put a literal secret in
`config.yaml` — it is tracked.

## Service Management

Lloyd runs **directly on the host** under supervisord (installed as the `agent-supervisord.service` systemd `--user` unit; supervisord itself is the uv tool at `~/.local/bin/supervisord`). There is no longer any distrobox container in the loop. Use supervisorctl directly:

```bash
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf restart lloyd-mc:lloyd-backend
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf restart lloyd-mc:lloyd-frontend
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf status
```

The process group is `lloyd-mc`, not `lloyd-backend` bare. Always use `lloyd-mc:lloyd-backend` and `lloyd-mc:lloyd-frontend`.

After editing `server.py`, restart `lloyd-mc:lloyd-backend` for changes to take effect.  
After editing frontend files, Vite HMR usually picks up changes automatically (no restart needed).

## Architecture

```
~/lloyd/
├── server.py            # FastAPI backend — all API endpoints + SSE bridge
├── config.yaml          # Model configs, MCP server list, agent settings
├── prompt_builder.py    # System prompt assembly (SOUL.md + memories + skills)
├── autonomy.py          # Task scheduler
├── usage_store.py       # SQLite usage tracking
│
├── app/harness/         # In-process agent loop (replaces claude-agent-sdk)
│   ├── __init__.py      # Exports: run_query, RunOptions, HookRegistry
│   ├── options.py       # RunOptions dataclass
│   ├── events.py        # NormalizedEvent TypedDict types
│   ├── client.py        # httpx SSE stream → vLLM /v1/chat/completions
│   ├── loop.py          # Agent loop: stream → tool dispatch → loop
│   ├── hooks.py         # HookRegistry (pre/post tool-use callbacks)
│   ├── mcp_pool.py      # Persistent SSE client to lloyd-mcp aggregator
│   ├── tool_schema.py   # MCP tools → OpenAI tool schema translation
│   └── errors.py        # ParseError, ToolDispatchError, MaxTurnsExceeded
│
├── agent_mcp/           # Unified MCP aggregator (Server("lloyd") on :8500/sse)
│   ├── main.py          # Aggregates all modules; MCP SSE endpoint
│   ├── builtin_bash.py  # Bash tool (timeout, truncation)
│   ├── builtin_fs.py    # Read, Write, Edit, Grep, Glob tools
│   ├── builtin_task.py  # Task subagent (in-process, recursion cap = 1)
│   └── ...              # Domain modules: ambient, facts, vault, session, etc.
│
├── web/src/
│   ├── api.ts           # All API calls + TypeScript types
│   └── components/pages/
│       └── ToolsPage.tsx
│
├── .venvs/lloyd/        # Python venv (use this python for all lloyd scripts)
└── logs/                # server.log, server.err, frontend.log
```

## Agent Harness

`run_query(messages: list[dict], options: RunOptions) -> AsyncIterator[NormalizedEvent]`

Events yielded by type:
- `text_delta` — `{type, text}` — streaming text chunk
- `thinking_delta` — `{type, text}` — vLLM reasoning content chunk
- `thinking_done` — `{type}` — reasoning phase complete
- `tool_call` — `{type, id, name, input}` — tool invocation
- `tool_result` — `{type, tool_call_id, content, is_error}` — tool result
- `assistant_message` — `{type, content, tool_calls}` — full assistant turn
- `result` — `{type, stop_reason, usage}` — turn complete
- `stream_raw` — `{type, line}` — raw SSE line on parse failure

**Tool naming**: Built-in tools (Bash, Read, Write, Edit, Grep, Glob, Task) are advertised to vLLM under bare names. This keeps session JSON, SOUL.md deny rules, and Inner Voice `pretooluse_deny` patterns working unchanged.

## Tools

Every tool lives inside an MCP server — built-ins (Bash/Read/Write/Edit/Grep/Glob/Task) live inside the lloyd-mcp aggregator. Tool enable/disable state:

- Server-level: `mcp_servers.<name>.enabled: false`
- Tool-level: `mcp_servers.<name>.disabled_tools: [tool_name, ...]` (use the bare tool name)

config.yaml holds the hand-edited defaults and is **read-only at boot**; UI toggles (`/api/tool-toggle`, `/api/tool-discovery`) persist to `data/tool_overrides.yaml` (gitignored), which is merged over config.yaml at load (`app/config.py:_merge_tool_overrides`). To change tool state by hand, edit config.yaml and check `data/tool_overrides.yaml` isn't shadowing the same key.

Disabled tools are enforced via `RunOptions.disallowed_tools` as `mcp__<server>__<tool>`. The harness's bare-name aliasing in `tool_schema.py` blocks both the bare and namespaced form at advertise + dispatch time, so disabling `Bash` via `mcp_servers.lloyd-mcp.disabled_tools: [Bash]` blocks the model from calling either `Bash` or `mcp__lloyd-mcp__Bash`.

## Knowledge graph

Two layers, and the distinction matters:

- **Fact layer** — markdown, one dir per entity under
  `_pipeline/vault-derived/facts/<Entity>/<Entity>-<category>.md`. Human
  readable, editable, diffable.
- **Store** — `_pipeline/vault-derived/kg.sqlite`, behind `app/kg_store.py`.
  Edges, aliases, the entity registry and a fact index derived from the
  markdown.

**Nothing opens the store except `app.kg_store`.** Not a script, not a
router, not a test fixture. Before 2026-09 the same state lived in two JSON
blobs that six programs rewrote whole with no lock, which produced the
2026-08-22 wipe (12,131 edges) and the 2026-09-03 merge incident (151
entities fused against a 2-edge graph).

```python
from app.kg_store import store
s = store()
s.edges.add({"source": "Lloyd", "target": "vLLM", "type": "uses"}, origin="fact_relate")
s.aliases.resolve("vllm")     # -> "vLLM"
s.facts_idx.for_entity("Lloyd", category="state")
```

Rules worth not relearning:

- A store that will not open raises `StoreUnavailable`. Never return an empty
  graph on a read failure — a writer will persist that emptiness.
- Expire edges, never delete them. `rewrite_endpoint` returns `(old_id,
  new_id)` pairs so a merge is exactly revertable.
- `LLOYD_FACTS_ROOT` / `LLOYD_KG_DB` point the fact tree and the store
  elsewhere — that is how a rebuild extracts without touching the live one.
- The extraction corpus is an allow-list in
  `scripts/memory/next-gen-memory/pipeline_config.yaml`. Edit the config,
  not the walk.

`architecture/knowledge-graph.md` is the long version.

## Config Structure (config.yaml)

```yaml
model:
  default: primary

models:
  primary:
    alias: primary
    base_url: http://127.0.0.1:8096
    context_length: 262144
    env:
      ANTHROPIC_BASE_URL: "http://127.0.0.1:8096"
      ANTHROPIC_API_KEY: "no-key-required"
      ANTHROPIC_CUSTOM_MODEL_OPTION: "primary"
      ANTHROPIC_CUSTOM_MODEL_OPTION_NAME: "Primary"

harness:
  stream_chunk_timeout_seconds: 60
  max_tools_per_request: 128

subagents:
  general-purpose:
    system_prompt: ""
    max_turns: 20
    disallowed_tools: []
    model: primary

mcp_servers:
  lloyd-mcp:
    type: sse
    url: http://127.0.0.1:8500/sse
    disabled_tools: []  # bare tool names, e.g. [Bash, browser_screenshot]

agent:
  max_turns: 60
  permission_mode: bypassPermissions
```

## Development Notes

- Each turn reconstructs the full conversation from the persisted session JSON (`load_and_compact_session`) and sends it as an OpenAI-format `messages` list to vLLM.
- vLLM tool calling: `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`
- Session continuity: no `resume=` — history is rebuilt from `sessions/<id>.json` each turn.
- The `/api/message/stream` endpoint uses SSE. The frontend connects via `fetch` + `ReadableStream`.
