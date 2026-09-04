---
segment: architecture
relations:
  related-to:
  - architecture/voice.md
  - architecture/memory.md
  - architecture/autonomy-system.md
  - architecture/evaluation-engine.md
  - architecture/backlog.md
  - architecture/index.md
  - architecture/infrastructure.md
  - architecture/morning-briefing.md
  - autonomy/38-nightly-reflection-signals.md
  - architecture/nightly-skills-management.md
  - architecture/nightly-vault-maintenance.md
  - architecture/skills.md
tags: [architecture]
summary: The lloyd-mcp aggregator — 124 tools across 21 modules, served to the
  in-process harness over Streamable HTTP on :8500/mcp.
type: reference

---



















# MCP Tools

Every tool Lloyd can call lives in the **lloyd-mcp aggregator**, a single
`Server("lloyd")` that mounts 21 modules and advertises their tools as one flat
namespace. There is no per-plugin server and no gateway: the in-process harness
(`app/harness/`) is the only client.

> Regenerate this inventory rather than editing it by hand:
> `.venvs/lloyd/bin/python -c "import asyncio; from agent_mcp import main as m; print(len(asyncio.run(m.list_tools())))"`
> This page was stale for months — it described a 41-tool FastMCP server on
> port 8093 behind an OpenClaw gateway, listing `mem_get`, `file_read` and
> `prefill_context`, none of which have existed for some time. A doc that names
> tools which do not exist is the same defect that taught the agent to shell out
> to `curl` (see the archived `websearch` skill, 2026-09-04).

## Server

- **Source:** `agent_mcp/main.py` — module list at `agent_mcp/main.py:79-102`
- **Service:** `lloyd-mc:lloyd-mcp` under supervisord, on the host (no distrobox)
- **Transport:** Streamable HTTP, stateless, at `:8500/mcp` (spec 2026-07-28).
  The legacy HTTP+SSE pair is gone.
- **Discovery caching:** `tools/list` carries `ttl_ms` / `cache_scope`
  (`TOOLS_LIST_TTL_MS`, `agent_mcp/main.py:157`), so a tool toggled in the Tools
  page reaches a running harness within the TTL.
- **Health:** `/health`, which reports per-module discovery failures — one
  module failing does not take the tool surface down.

## Contract

A module is anything exposing `async def list_tools() -> list[Tool]` and
`async def call_tool(name, arguments) -> list[TextContent]`, optionally
`async def shutdown()`. Modules do not own an `mcp.server.Server`. Import fails
loudly if a module does not satisfy this (`_check_module`, `main.py:105-131`).

Tool names are advertised **bare** (`Bash`, not `mcp__lloyd-mcp__Bash`); the
legacy prefixed form is still parsed so old session JSON replays.

## Annotations

`agent_mcp/annotations.py` is the central behaviour table — `readOnlyHint`,
`destructiveHint`, `idempotentHint`, `openWorldHint`. It is security-relevant:
the plan-mode block list is derived from `readOnlyHint`, so a tool missing from
the table is treated as not read-only and is blocked while drafting a plan.

## Progressive disclosure

With 124 tools, advertising the whole catalog every request cost ~25.8k tokens
and degraded selection accuracy. `harness.tool_search` advertises a small
baseline plus a `ToolSearch` meta-tool and loads the rest on demand
(`app/harness/tool_search.py`). The baseline is the set the model can always
see — keep it small, and keep it honest: anything not in it competes only after
a discovery round-trip, which is why `http_search`/`http_fetch` sat unused while
`Bash` was always visible.

## Tool Inventory (124 tools, 21 modules)

| Module | n | Tools |
|--------|---|-------|
| `builtin_bash` | 2 | `Bash`, `_BackgroundTaskDrain` (internal, never advertised) |
| `builtin_fs` | 5 | `Read`, `Write`, `Edit`, `Grep`, `Glob` |
| `builtin_goal` | 2 | `SetGoal`, `ClearGoal` |
| `builtin_plan` | 2 | `EnterPlanMode`, `ExitPlanMode` |
| `builtin_task` | 1 | `Task` |
| `builtin_todo` | 1 | `TodoWrite` |
| `ambient` | 2 | `session_inject_context`, `ambient_decide` |
| `autonomy` | 7 | `autonomy_tasks`, `autonomy_write_task`, `autonomy_get_task`, `autonomy_delete_task`, `autonomy_config`, `autonomy_run_task`, `autonomy_health` |
| `autoresearch` | 7 | `autoresearch_round`, `autoresearch_status`, `autoresearch_bench_list`, `autoresearch_bench_add`, `autoresearch_ledger_query`, `autoresearch_promote`, `autoresearch_rollback` |
| `backlog` | 4 | `backlog_boards`, `backlog_tasks`, `backlog_get_task`, `backlog_write_task` |
| `browser` | 14 | `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_press`, `browser_tabs`, `browser_screenshot`, `browser_evaluate`, `browser_fill`, `browser_wait`, `browser_select`, `browser_drag`, `browser_cookies` |
| `discord_bot` | 4 | `discord_send`, `discord_send_embed`, `discord_list_channels`, `discord_get_home_channel` |
| `facts` | 10 | `fact_get`, `fact_add`, `fact_profile`, `fact_check`, `fact_resolve`, `fact_invalidate`, `fact_relate`, `fact_relationships`, `fact_path`, `fact_neighbors` |
| `vault` | 5 | `vault_read`, `vault_write`, `vault_overview`, `vault_search`, `vault_recall` |
| `session` | 5 | `memory_read`, `memory_add`, `memory_replace`, `memory_remove`, `session_recall` |
| `mission_control` | 2 | `chat_list_sessions`, `chat_get_session` |
| `mission_control_ui` | 3 | `mc_get_state`, `mc_navigate`, `mc_close_modal` |
| `ide` | 3 | `ide_open_folder`, `ide_open_file`, `ide_close_tab` |
| `skills` | 2 | `skills_search`, `skills_read` |
| `http_tools` | 3 | `http_search`, `http_fetch`, `http_request` |
| `thunderbird` | 40 | `email_*` (24), `calendar_*` (7), `tasks_*` (3), `contacts_*` (5) — runs as an MCP stdio bridge through `MCPPool` |

### Web (3)

| Tool | Use |
|------|-----|
| `http_search` | Search the public web (DuckDuckGo); returns ranked titles, URLs, snippets |
| `http_fetch` | Fetch a public URL as markdown (headings, lists, tables, `[text](href)` links) or plain text, via trafilatura. GET only. Blocks private hosts by design. Does not parse PDFs |
| `http_request` | Raw request — any verb, custom headers, body; returns status, headers, unparsed body |

`Bash` + `curl` stays correct for localhost (which `http_fetch` blocks) and for
the structured-API pipelines individual skills document. For everything else on
the public web, the `http_*` tools are the answer — see the `web-lookup` skill.

## Enable / disable

- Server level: `mcp_servers.<name>.enabled: false`
- Tool level: `mcp_servers.<name>.disabled_tools: [bare_tool_name, ...]`

`config.yaml` holds hand-edited defaults and is read-only at boot; the Tools
page writes `data/tool_overrides.yaml`, which is merged over it
(`app/config.py:_merge_tool_overrides`). Resolve the effective set through
`app.mcp_discovery._get_disallowed_tools()` / `_get_tool_search_kwargs()` —
reading `config.yaml` directly misses both the overrides and `${VAR}` expansion.
