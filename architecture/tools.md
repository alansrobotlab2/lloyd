---
segment: architecture
relations:
  related-to:
  - architecture/voice.md
  - architecture/memory.md
  - architecture/autonomy.md
  - architecture/backlog.md
  - architecture/index.md
  - architecture/infrastructure.md
  - autonomy/38-nightly-reflection-signals.md
  - architecture/autonomy-jobs.md
  - architecture/skills.md
  - architecture/harness.md
  - architecture/editing-safeguards.md
tags: [architecture]
summary: The lloyd-mcp aggregator — 152 tools across 26 modules, served to the
  in-process harness over Streamable HTTP on :8500/mcp.
type: reference
status: implemented
date: 2026-09-11

---



















# MCP Tools

Every tool Lloyd can call lives in the **lloyd-mcp aggregator**, a single
`Server("lloyd")` that mounts 26 modules and advertises their tools as one flat
namespace. There is no per-plugin server and no gateway: the in-process harness
(`app/harness/`) is the only client.

> Regenerate this inventory rather than editing it by hand. The cheapest
> authority is the running server, which counts exactly what it dispatches:
> `curl -s http://127.0.0.1:8500/health` — `tools`, `modules` and a per-module
> breakdown. Offline, or in a worktree:
> `.venvs/lloyd/bin/python -c "import asyncio; from agent_mcp import main as m; print(len(asyncio.run(m.list_tools())))"`
> This page was stale for months — it described a 41-tool FastMCP server on
> port 8093 behind an OpenClaw gateway, listing `mem_get`, `file_read` and
> `prefill_context`, none of which have existed for some time. A doc that names
> tools which do not exist is the same defect that taught the agent to shell out
> to `curl` (see the archived `websearch` skill, 2026-09-04). It then went stale
> a second, quieter way: the count sat at 124/21 while four modules were mounted
> under it — `code_graph` (2026-09-08), `memory_ops` and `research` (2026-09-08),
> `builtin_grants` (#534, 2026-09-09) — and the automod tools were renamed out
> from under the table. A number that is only *slightly* wrong invites nobody to
> re-derive it.

## Server

- **Source:** `agent_mcp/main.py` — module list at `agent_mcp/main.py:120-149`
- **Service:** `lloyd-mc:lloyd-mcp` under supervisord, on the host (no distrobox)
- **Transport:** Streamable HTTP, stateless, at `:8500/mcp` (spec 2026-07-28).
  The legacy HTTP+SSE pair is gone.
- **Port:** resolved through `services.lloyd_mcp` in config.yaml
  (`_resolve_port`, `main.py:93-114`), the same registry every *client*
  resolves — it used to bind a hardcoded 8500, so moving the port in config
  pointed every client at a server still on the old one. `LLOYD_MCP_PORT`
  overrides, so a canary can take a second port without a config edit.
- **Discovery caching:** `tools/list` carries `ttl_ms` / `cache_scope`
  (`TOOLS_LIST_TTL_MS` = 60 s, `agent_mcp/main.py:236`), so a tool toggled in
  the Tools page reaches a running harness within the TTL.
- **Health:** `GET /health`, which reports per-module discovery failures — one
  module failing does not take the tool surface down. It answers **503** when
  any module is degraded, which is why the guardian excludes this endpoint from
  its down predicate and judges it by `mcp_degraded_is_fatal` instead: a closed
  Thunderbird is not a dead aggregator.
- **Responses are JSON, not SSE** (`json_response=True`). Not a preference —
  Streamable HTTP's SSE framing runs through httpx2's 1 MiB
  `DEFAULT_MAX_EVENT_SIZE_BYTES` with no way to raise it, so a tool result
  above that (`fact_get` on a well-connected entity returns ~1.4 MB) died as
  "SSE stream ended without a response" with the real cause in a debug log.
  Nothing here streams partial results, so the framing bought a size ceiling
  and nothing else.
- **DNS-rebinding protection is on** (`TransportSecuritySettings`,
  `main.py:563-571`). The aggregator binds loopback with no auth, so the only
  thing between a page in the user's browser and this tool surface is the
  SDK's host/origin check. Hosts are matched by **name with a wildcard port**:
  the threat is a page whose domain resolves to 127.0.0.1, which carries the
  attacker's hostname in `Host`/`Origin`. Pinning the port instead would be
  actively harmful — `/health` answers happily on another port while every MCP
  request fails 421 "Invalid Host header", which is the silent partial failure
  this was added to remove.

### The other four routes on :8500

The aggregator is not only an MCP endpoint. Four Starlette routes ride beside
`/mcp` (`main.py:791-797`), all for the same reason: `Task`, background Bash
children and Playwright all live in **this** process, so the backend has no
handle on them and crosses loopback rather than sharing a file.

| Route | Serves |
|---|---|
| `GET /state` | Mission Control's agent panel: `subagents`, `background_tasks` (`active` **and** `recent`), `tsc`, `changes`, `tools` |
| `POST /browser/navigate` | the Browser tab's URL bar — a route, not a tool, because the user typing a URL is not the agent calling something and must not be logged as one |
| `GET /changes` | what a turn wrote (`?session=&turn=`) — the per-turn change ledger |
| `POST /changes/revert` | undo those writes, per file, refusing by name anything that moved since |

`background_tasks.recent` is there because `list_active` filters on
`status == "running"`: before it, a background bash left the dashboard the
instant it exited, so a task that died three seconds in was indistinguishable
from one that never started. See [[mission-control]] for the consuming panel
and [[editing-safeguards]] for the ledger behind `/changes`.

## Contract

A module is anything exposing `async def list_tools() -> list[Tool]` and
`async def call_tool(name, arguments) -> CallToolResult | list[TextContent]`,
optionally `async def shutdown()`. Modules do not own an `mcp.server.Server` —
they used to, and those instances were dead weight, since the SDK's
`@server.list_tools()` decorator returns the function unchanged and the
per-module handler map was never dispatched. Import fails loudly if a module
does not satisfy the contract (`_check_module`, `main.py:161-177`), which is
cheaper than discovering it on the first `tools/list` in production: a module
missing `call_tool` would otherwise register its tools fine and fail every
dispatch. `shutdown()` is what releases a Chromium or a Node bridge —
before it, every restart orphaned both.

Three registration rules, each one a failure that happened:

- **A raising module is skipped, not fatal.** It is logged, recorded in
  `_discovery_status` and left out. Before that guard one bad module failed
  `tools/list`, which failed the harness pool open, which left the agent with
  no tools at all — the worst failure in the system (see [[harness]]).
- **A duplicate name is dropped, not shadowed.** The first module to claim a
  name keeps it and the second is logged at ERROR. Silent shadowing was
  possible here: the last module won and the tool appeared twice in the
  advertised list.
- **`_dispatch` is rebound, never mutated.** `list_tools` builds a fresh dict
  and rebinds in one statement, which is atomic under the GIL. The old code
  cleared the dict and refilled it across 22 `await` boundaries, so any call
  arriving from another MCP session in that window routed to nothing and the
  model was told "Unknown tool". `tests/test_mcp_layer.py::test_dispatch_never_observed_empty_during_rebuild`
  pins it.

Names are capped at `TOOL_NAME_MAX` (64, OpenAI's limit) at registration, so a
bad name fails on the first `list_tools()` rather than mid-conversation in the
harness translator.

Tool names are advertised **bare** (`Bash`, not `mcp__lloyd-mcp__Bash`); the
legacy prefixed form is still parsed (`tool_schema.resolve_tool_name`) so old
session JSON replays. Three consequences live in
`app/harness/tool_schema.py::build_tool_list`:

- **A cross-server name collision raises** rather than silently shadowing —
  bare advertising leaves no way to disambiguate. It is also the standing
  argument against mounting a second MCP server (see the code-graph section).
- **A name starting with `_` is never advertised.** The harness can still
  dispatch it directly through the pool, which is exactly how
  `_BackgroundTaskDrain` works.
- **`disallowed_tools` blocks both forms.** `RunOptions.disallowed_tools`
  carries bare names, and `build_tool_list` skips a tool whose bare name *or*
  whose `mcp__<server>__<tool>` spelling appears in the set — at advertise
  time, so a disabled tool is never in the payload the model reads.

## Annotations

`agent_mcp/annotations.py` is the central behaviour table — `readOnlyHint`,
`destructiveHint`, `idempotentHint`, `openWorldHint`. It is security-relevant:
the plan-mode block list is derived from `readOnlyHint`, so a tool missing from
the table is treated as not read-only and is blocked while drafting a plan.

A central table rather than annotations on each `Tool(...)` because the
classification is easier to review as one ordered file than as 152 constructor
calls spread over 26 modules, and because the sets are explicit rather than
pattern-matched: a regex over tool names gets `email_apply_filters` and
`autonomy_get_task` wrong in opposite directions. `annotate()` is a default,
not an override — a module that sets its own annotations keeps them.

Five sets, and the last two are not cosmetic:

| Set | Means | Read by |
|---|---|---|
| `READ_ONLY` | observes, changes nothing anywhere | the plan-mode gate, `side_effecting`, parallel dispatch, `_retry_safe` |
| `DESTRUCTIVE` | may irreversibly destroy data | UI badging and blast-radius reasoning — *not* the plan-mode gate |
| `IDEMPOTENT` | a second identical call adds no further change | `side_effecting`, `_retry_safe` |
| `REPEAT_EXPECTED` | repeating it is normal operation, so a replay would be a stale answer delivered as a fresh one | the #544 effect ledger |
| `PLAN_MODE_ALWAYS_ALLOWED` | not read-only, but must stay callable in plan mode | the plan-mode gate |

`annotations_for` marks everything in `READ_ONLY` idempotent too, and derives
`openWorldHint` from a prefix list (`http_`, `browser_`, `email_`, `calendar_`,
`contacts_`, `tasks_`, `discord_`) plus `Task` and `Bash`.

Four consumers, each of which used to keep its own private list of names:

- **The plan-mode gate.** `plan_mode_blocked_tools(universe)` is everything
  not read-only, minus `PLAN_MODE_ALWAYS_ALLOWED`, minus `_`-prefixed.
  `app/mcp_discovery.py::_plan_mode_blocked` calls it; the three-name tuple
  `PLAN_MODE_BLOCKED_TOOLS = ("Write", "Edit", "Bash")` survives only as the
  **floor** used before discovery has populated `_TOOL_UNIVERSE`. That floor
  used to be the whole gate, which left `email_send`, `vault_write`,
  `fact_add`, `discord_send`, `browser_click` and ~55 others sailing straight
  through a "read-only" plan-mode turn. `PLAN_MODE_ALWAYS_ALLOWED` exists
  because `ExitPlanMode` is itself not read-only, and a plan mode that blocks
  its own exit is a deadlock rather than a guard.
- **`side_effecting(name)`** — the #544 effect ledger's classifier. Excludes
  `READ_ONLY`, `IDEMPOTENT` and `REPEAT_EXPECTED`; everything else is a
  candidate. Its first cut consulted only the first and third, which ledgered
  20 idempotent tools for no protective value — `Write` among them. See
  [[editing-safeguards]] for what the ledger then does.
- **Parallel dispatch** qualifies a batch only when every call carries
  `readOnlyHint` — the server's own hint, carried through discovery, rather
  than a second private list in the harness.
- **`MCPPool._retry_safe`** re-sends a transport-failed call only when the
  server annotated it `readOnlyHint` or `idempotentHint`. A transport error
  says nothing about whether the server ran the call, and for a long one it
  almost certainly did.

The table can only go stale in one direction, and a test closes it:
`tests/test_mcp_layer.py::test_annotation_tables_have_no_stale_entries` fails
on any name in the four tables that no longer exists — exempting the
Thunderbird names when that module is degraded, because the bridge is a
gitignored build artifact and exports nothing inside an automod worktree.
`test_every_tool_is_annotated` covers the other direction.

## Progressive disclosure

Advertising the whole catalog every request is billed as input tokens every
turn — at 124 tools that was ~25.8k — and tool-call accuracy degrades past
~30-50 simultaneously loaded tools. `harness.tool_search` advertises a small
baseline plus a `ToolSearch` meta-tool and loads the rest on demand
(`app/harness/tool_search.py`, `tool_search_cache.py`). The model sees a
`role: system` reminder listing every deferred tool's name and a one-line
gist — no schemas; `ToolSearch(query=…)` is intercepted in the harness with no
MCP round-trip, marks the matches loaded, and returns their schemas in a
`<functions>` block. The baseline is the set the model can always see — keep it
small, and keep it honest: anything not in it competes only after a discovery
round-trip, which is why `http_search`/`http_fetch` sat unused while `Bash` was
always visible.

**It is switched off today.** `harness.tool_search.enabled` is `false` in both
config.yaml and the live override, so all 152 tools minus the disabled ones are
advertised on every request; `threshold_tools` is 30 and `baseline_tools` holds
28 names (`max_results_default: 5`, `max_results_cap: 20`). The threshold only
bites when the flag is on — `GET /api/tool-discovery` reports that join as
`active`. Read the effective values through `app.mcp_discovery._get_harness_kwargs()`,
which exposes them as `tool_search_enabled` / `tool_search_threshold_tools` /
`tool_search_baseline` / `tool_search_max_results_default` /
`tool_search_max_results_cap`.

Both files must carry the same block: the override shadows this key wholesale,
so an unpaired edit to config.yaml changes only what a fresh clone boots into.
`tests/test_tool_overrides.py::test_config_yaml_agrees_with_the_live_override`
is what catches that, and the merge logs a warning on any key where the two
disagree.

## Tool Inventory (152 tools, 26 modules)

Verified against `GET :8500/health` on 2026-09-11. The count is what the
aggregator *dispatches*; what the model is shown is smaller — subtract
`_BackgroundTaskDrain` (never advertised) and whatever sits in
`disabled_tools`, which today is the four `discord_*` tools, for 147.

| Module | n | Tools |
|--------|---|-------|
| `builtin_bash` | 2 | `Bash`, `_BackgroundTaskDrain` (internal, never advertised) |
| `builtin_fs` | 5 | `Read`, `Write`, `Edit`, `Grep`, `Glob` |
| `builtin_goal` | 2 | `SetGoal`, `ClearGoal` |
| `builtin_grants` | 3 | `grant_create`, `grant_list`, `grant_revoke` — the human's mint path for #534 scope-bound authority; minting refuses any session a human does not read |
| `builtin_plan` | 2 | `EnterPlanMode`, `ExitPlanMode` |
| `builtin_task` | 1 | `Task` |
| `builtin_todo` | 1 | `TodoWrite` |
| `ambient` | 2 | `session_inject_context`, `ambient_decide` |
| `autonomy` | 7 | `autonomy_tasks`, `autonomy_write_task`, `autonomy_get_task`, `autonomy_delete_task`, `autonomy_config`, `autonomy_run_task`, `autonomy_health` |
| `autoresearch` | 7 | `autoresearch_round`, `autoresearch_status`, `autoresearch_bench_list`, `autoresearch_bench_add`, `autoresearch_ledger_query`, `autoresearch_promote`, `autoresearch_rollback` |
| `backlog` | 4 | `backlog_boards`, `backlog_tasks`, `backlog_get_task`, `backlog_write_task` |
| `browser` | 14 | `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_press`, `browser_tabs`, `browser_screenshot`, `browser_evaluate`, `browser_fill`, `browser_wait`, `browser_select`, `browser_drag`, `browser_cookies` |
| `code_graph` | 6 | `graph_explain`, `graph_affected`, `graph_path`, `graph_hubs`, `graph_status`, `graph_refresh` — structural navigation over graphify's AST extraction of a tree |
| `discord_bot` | 4 | `discord_send`, `discord_send_embed`, `discord_list_channels`, `discord_get_home_channel` |
| `facts` | 10 | `fact_get`, `fact_add`, `fact_profile`, `fact_check`, `fact_resolve`, `fact_invalidate`, `fact_relate`, `fact_relationships`, `fact_path`, `fact_neighbors` |
| `memory_ops` | 4 | `remember`, `recall`, `forget`, `improve` — #376's four cognitive verbs. **Routers, not replacements**: each one wraps a `fact_*`/`vault_*` tool and adds the single guard it lacks (a dedupe check, a refusal to expire unscoped), so the surface grew by four rather than shrinking by fifteen |
| `vault` | 5 | `vault_read`, `vault_write`, `vault_overview`, `vault_search`, `vault_recall` |
| `session` | 5 | `memory_read`, `memory_add`, `memory_replace`, `memory_remove`, `session_recall` |
| `mission_control` | 2 | `chat_list_sessions`, `chat_get_session` |
| `mission_control_ui` | 3 | `mc_get_state`, `mc_navigate`, `mc_close_modal` |
| `ide` | 3 | `ide_open_folder`, `ide_open_file`, `ide_close_tab` |
| `research` | 5 | `research_propose`, `research_next`, `research_list`, `research_stats`, `research_complete` — the topic registry. `research_next` is deliberately peek-only: only the `deep-research` worker claims, or a chat turn that wanders off strands a topic in `researching` |
| `automod` | 10 | `automod_start`, `automod_gate`, `automod_gate_wait`, `automod_land`, `automod_abort`, `automod_status`, `automod_amend_clause`, `automod_rollback`, `automod_vault_land`, `automod_vault_revert` — a thin surface over `scripts/automod/`. There is deliberately no `automod_write_code`: a round is a worktree path plus ordinary Edit/Write/Bash. Every mutating tool refuses while `automod.enabled` is false |
| `skills` | 2 | `skills_search`, `skills_read` |
| `http_tools` | 3 | `http_search`, `http_fetch`, `http_request` |
| `thunderbird` | 40 | `email_*` (26), `calendar_*` (6), `tasks_*` (3), `contacts_*` (5) — `node mcp-bridge.cjs` over stdio through `MCPPool`, discovery cached 300 s. It degrades to **zero** tools when Thunderbird is not running, which is the usual reason `/health` answers 503 |

### Code graph (6)

`agent_mcp/code_graph.py` reads `<root>/graphify-out/graph.json` — a
deterministic AST extraction, no LLM calls — and answers structural
questions a grep cannot: who calls this, what breaks if I change it, how
do these two connect.

| Tool | Use |
|------|-----|
| `graph_explain` | Inbound/outbound edges of a symbol, each with the caller's own call site |
| `graph_affected` | Reverse-BFS blast radius, grouped by depth, with the file list |
| `graph_path` | Shortest dependency path between two symbols (containment edges excluded) |
| `graph_hubs` | Most-connected symbols, optionally scoped by path prefix |
| `graph_status` | Counts, built-at commit vs HEAD, why it is stale. Never builds |
| `graph_refresh` | Force a rebuild (~15 s on this repo) |

`root` is explicit and never inferred from the calling session — nothing on
disk links a chat session to an open automod round, so a "bound session's
worktree" default would confidently answer about the wrong checkout. It
defaults to `LLOYD_HOME`, accepts an `SM_…` round id, or an absolute path.

Staleness is a commit mismatch **or** an uncommitted source file newer than
`graph.json`. The second rule is the load-bearing one: inside a round HEAD
does not move while the model edits, so a commit-only rule would call the
graph fresh for exactly the window it is most wrong in. Rebuilds triggered
that way are debounced (`code_graph.min_refresh_interval_s`); a debounced
query still answers, and says it is stale.

The graph is an extraction of one tree, so it is blind across process
seams — an HTTP call from the backend to the aggregator, or an MCP dispatch
from `run_query` into a tool handler, is not an edge. Grep is still the
right tool for string keys, route paths and config names.

`graphify-out/` is gitignored (unanchored, so worktrees inherit it): a
build inside a round would otherwise dirty the tree, and both
`scripts/automod/gate.py` and `promote.py` refuse a dirty tree.

Config lives under `code_graph:` — `graphify_bin`, `auto_refresh: true`,
`refresh_timeout_s: 120`, `min_refresh_interval_s: 30`, `max_cached_roots: 4`,
`max_lines: 60`. There is deliberately **no `enabled` flag**: the kill switch
is `mcp_servers.lloyd-mcp.disabled_tools`, because an `enabled: false` that
emptied `list_tools()` would break the annotation-staleness test above. The
same six tools are also called *passively*, by the blast-radius rail on every
interface-changing edit — see [[editing-safeguards]].

### Web (3)

| Tool | Use |
|------|-----|
| `http_search` | Search the public web (DuckDuckGo); returns ranked titles, URLs, snippets |
| `http_fetch` | Fetch a public URL as markdown (headings, lists, tables, `[text](href)` links) or plain text, via trafilatura. GET only. Blocks private hosts by design. `max_chars` clamped to 1000–200000, default 50000 |
| `http_request` | Raw request — any verb, custom headers, body; returns status, headers, unparsed body. Blocks private hosts **except loopback**, mirroring the browser's policy |

`http_fetch` has read PDFs since 2026-09-04 (`1a38392`) — pymupdf, page by
page, returning `content_type: "pdf"`. This page claimed it did not for as
long as it did.

`Bash` + `curl` stays correct for localhost (which `http_fetch` blocks) and for
the structured-API pipelines individual skills document. For everything else on
the public web, the `http_*` tools are the answer — see the `web-search-and-fetch` skill.

## Enable / disable

- Server level: `mcp_servers.<name>.enabled: false`
- Tool level: `mcp_servers.<name>.disabled_tools: [bare_tool_name, ...]`

`config.yaml` holds hand-edited defaults and is read-only at boot; the Tools
page writes `data/tool_overrides.yaml`, which is merged over it
(`app/config.py:_merge_tool_overrides`). That file is gitignored on purpose
and must stay that way: the Tools page rewrites it on every toggle, and while
it was tracked a single click left the live tree dirty, which the automod
gate and promoter both refuse. A fresh clone therefore has no override file
and boots on `config.yaml`, so the tracked defaults must describe the state
actually being served — the merge warns when they disagree. Resolve the effective set through
`app.mcp_discovery._get_disallowed_tools()` / `_get_harness_kwargs()` —
reading `config.yaml` directly misses both the overrides and `${VAR}` expansion.

The merge is deliberately narrow (`app/config.py:83-190`). Only the UI-mutable
slice is honoured — per-server `enabled` / `disabled_tools`, the
`harness.tool_search` block, `workers.enabled`, and
`workers.sources.<name>.inner_voice` — so a stray write into that file cannot
shadow hand-edited config. An override may adjust only what config.yaml
already defines: one naming an unknown server, or an unknown worker source, is
ignored rather than introducing it. An unreadable file leaves the config
unchanged. `save_tool_overrides()` is the
single writer: every caller rebuilds the whole file from `CONFIG`, so two
endpoints writing different slices cannot clobber each other.

Both tool-facing halves warn on disagreement, and both warnings were paid for. A
`disabled_tools` entry the override re-enables is logged by name — the override
wins, deliberately, since the Tools page is the live authority, but
`browser_screenshot` sat disabled in config.yaml for months while the override
kept it advertised (2026-09-04). `harness.tool_search` went without a warning
for longer: it was a bare `.update()`, so config.yaml claimed `enabled: true`
while the override served `false`, with nothing logged either way (2026-09-07).
Agreement stays silent, because the Tools page rewrites the whole block on
every toggle and a warning on agreement would fire each boot and stop meaning
anything.

`tests/test_tool_overrides.py` pins all of it, including the half that is easy
to get backwards: the end-to-end test *writes* the file through the real writer
and asserts git never sees it — the `.tmp` sibling `atomic_write_text` lands
before renaming included. It used to assert the file simply `exists()`, which
in an automod worktree demanded a file that is by design never checked out, and
failed every round for fifteen hours.

Both routes live in `app/routers/tools.py`: `POST /api/tool-toggle` for server
and tool enablement, `GET`/`POST /api/tool-discovery` for the `tool_search`
block.

## Subagents

`Task` is one tool in `builtin_task`, but it runs a nested `run_query` inside
this process with its own `RunOptions` from `subagents.<type>`. Recursion is
capped at `MAX_TASK_DEPTH = 1`, and the calling turn's model and endpoint
arrive in the request `_meta` (`lloyd/model`, `lloyd/base_url`) because a tool
running in the aggregator has no other way to know what spawned it.

Every result carries a `task_id`, and passing it back with a follow-up prompt
**resumes** that subagent rather than starting from nothing. The stored run's
identity wins — type, profile, model, base URL and the `task:*` session id all
come from the history, so a continuation keeps its `ToolSearch` `LoadedToolSet`
and its spill directory; only `disallowed_tools` is merged live, so a tool
switched off since the first run is honoured. Moving a half-finished
conversation to another engine would re-prefill all of it, which is why
`current_parent_model` is deliberately not consulted on a resume.

The store is process-scoped and bounded three ways in `_subagent_registry.py`:
`_HISTORY_KEEP` = 8 tasks, `_HISTORY_TTL_S` = 1800 s, `_HISTORY_MAX_CHARS` =
3,000,000 — TTL swept first, then oldest-first until both caps hold. After an
aggregator restart every id reads `unknown or evicted`, which is honest: the
conversation went with the process. The three refusal reasons stay distinct
(`unknown or evicted`, `expired`, `still running`) because they call for
different next moves.

`_sanitise` drops a trailing assistant message whose tool calls were never
answered — the one invalid shape the loop can leave behind, from a cancel or an
exception between the stream ending and the dispatch completing, and the one
every engine rejects on replay. `tests/test_task_resume.py` and
`tests/test_task_registry_wiring.py` pin the store and the dashboard row; see
[[harness]] for the run loop itself.
