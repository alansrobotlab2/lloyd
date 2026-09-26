---
segment: architecture
relations:
  related-to:
  - architecture/harness.md
  - architecture/editing-safeguards.md
  - architecture/vault-protection.md
  - architecture/djev.md
  - architecture/mission-control.md
  - architecture/infrastructure.md
  - architecture/index.md
  - architecture/skills.md
  - architecture/memory.md
  - architecture/autonomy.md
  - architecture/backlog.md
  - architecture/voice.md
  - autonomy/38-nightly-reflection-signals.md
  - architecture/autonomy-jobs.md
tags: [architecture, tools, mcp]
summary: The lloyd-mcp aggregator — 145 tools across 26 modules behind one
  Server("lloyd") on :8500/mcp, the dispatch path every call takes, and every
  tool with its properties.
type: reference
status: implemented
date: 2026-09-21
---

# MCP tools

Every tool Lloyd can call lives in the **lloyd-mcp aggregator**: one
`Server("lloyd")` in its own process that mounts 26 Python modules and
advertises their tools as one flat namespace. There is no per-plugin server
and no gateway. The in-process harness (`app/harness/`) is the client that
dispatches tools. The Tools page's discovery (`app/mcp_discovery.py`) opens its
own session to list them, and the backend reads the side routes in §2.

**145 tools, 26 modules**, from an offline `list_tools()` on 2026-09-23 (105
in the tree plus the Thunderbird bridge's 40). The model is shown fewer:
`_BackgroundTaskDrain` is never advertised, and the four `discord_*` tools sit
in `disabled_tools`, so an ordinary chat turn sees 140. Twelve were retired on
2026-09-23 as duplicates of, or subsumed by, a tool that stays: `fact_check`
(`fact_resolve`), `fact_profile` (`fact_get`, which took its cap and `query`
ranking), `browser_type` (`browser_fill(keystrokes=true)`), #376's
`remember`/`recall`/`forget`/`improve` (`fact_add`, `vault_recall`,
`fact_invalidate`, which took `forget`'s scope refusal, and the nightly
`fact-improvement` script), and five operator verbs of `autoresearch`.
`tests/test_mcp_layer.py::RETIRED_TOOLS` keeps them out, and
`test_advertised_catalog_stays_under_its_token_ceiling` pins the tree's
catalog size. `ToolSearch` is not in the count. The
harness answers it itself (see §7).

> **Regenerate this inventory; do not edit it by hand.** §11 has the command.
> This page has gone stale twice already. First it described a 41-tool FastMCP
> server behind an OpenClaw gateway, listing tools that had been gone for
> months. A doc that names tools which do not exist is the defect that taught
> the agent to shell out to `curl` (the archived `websearch` skill,
> 2026-09-04). Then the count sat at 124/21 while four modules were mounted
> under it. A number that is only *slightly* wrong invites nobody to
> re-derive it.

---

## 1. The shape

```
 BACKEND PROCESS (run_query)                      AGGREGATOR PROCESS (lloyd-mcp, :8500)
 ───────────────────────────                      ───────────────────────────────────────
 build_tool_list                                  aggregator_auth: X-Lloyd-Aggregator-Token
   drops disallowed_tools + plan-mode blocks        on every route but GET /health
   (or baseline + ToolSearch, when enabled)       Streamable HTTP, stateless, JSON bodies
   injects the `summary` caption parameter                    │
 PreToolUse hooks                                 main.call_tool(name, args, _meta)
   safety: destructive Bash, protected paths,       1. bench/eval sandbox        _tool_sandbox
     sync registration, service control             2. Bash safety check         check_bash_command
   outbound content gate (#1136)                    3. no session id + a write → refused (#1053)
   grant gate (#534), background sessions           4. effect ledger claim (#544, worker scopes)
   Inner Voice pre-tool deny, skill dispatch        5. module.call_tool(...)
 MCPPool._invoke ──── one HTTP session per call ──▶ 6. effect ledger finish
   _meta: session, turn, call, model,
   base_url, summary, effect_scope               27 modules  ─  builtin_fs, builtin_bash, vault,
   re-send only readOnly/idempotent tools                        facts, browser, thunderbird, ...
 tool_result: spill to disk above 50k chars
```

Two properties of this layout drive most of what follows:

- **The gates are split across two processes, on purpose.** What depends on
  the *turn* (plan mode, the grant scope, Inner Voice, the model's caption)
  runs in the harness, which knows the turn. What must hold *whoever calls*
  (the bench sandbox, the destructive-command check, the no-session refusal,
  the effect ledger) runs in `call_tool`, the one function every call passes
  through. `autonomy.run_task` and `run_prompt_on_primary` never installed the
  harness safety hook. The aggregator's copy of that check is the one that
  cannot be forgotten.
- **Everything the aggregator knows about a call arrives in `_meta`**, never
  in `arguments`. The SDK validates `arguments` against each tool's
  `inputSchema` before the handler runs, and the repetition guard and the
  effect ledger hash them. An injected argument would be rejected by a strict
  schema and would make two identical calls look different.

---

## 2. The aggregator process

- **Source:** `agent_mcp/main.py`. The module list is `MODULES`.
- **Service:** `lloyd-mc:lloyd-mcp` under supervisord, on the host. Restart it
  with `round restart`, never bare `supervisorctl` (see CLAUDE.md).
- **Transport:** Streamable HTTP at `:8500/mcp`, MCP 2026-07-28, stateless:
  the server requires no handshake and pins no `Mcp-Session-Id` (the pool's
  per-call sessions still send `initialize`, and it costs nothing). A dropped connection costs
  a reconnect (~6 ms), not a session shared by every turn in flight. The legacy
  HTTP+SSE pair is gone.
- **Responses are JSON, not SSE** (`json_response=True`). This is required, not
  a preference. SSE framing runs through httpx2's 1 MiB
  `DEFAULT_MAX_EVENT_SIZE_BYTES`, which the MCP client cannot raise, so a
  result above it (`fact_get` on a well-connected entity is ~1.4 MB) died as
  "SSE stream ended without a response". Nothing here streams partial results.
- **Port:** resolved through `services.lloyd_mcp` in config.yaml
  (`_resolve_port`), the same registry every client reads. `LLOYD_MCP_PORT`
  overrides it, so a canary can take a second port without a config edit.
- **Discovery caching:** `tools/list` carries `ttl_ms` 60 000 and
  `cache_scope: private`, so a tool toggled on the Tools page reaches a running
  harness within a minute.
- **DNS-rebinding protection** (`TransportSecuritySettings`): Host and Origin
  must be a loopback *name*, with any port. The threat is a page whose domain
  resolves to 127.0.0.1, and that carries the attacker's hostname. Pinning the
  port would be harmful: `/health` would still answer on another port while
  every MCP request failed 421.
- **A request credential** (#1053, `agent_mcp/aggregator_auth.py`). Every route
  except `GET /health` needs `X-Lloyd-Aggregator-Token`. The token is at
  `~/.local/state/lloyd/aggregator-token`, mode 0600. The first process to
  look writes it, create-exclusive, which is normally the aggregator at boot;
  a file that already holds a token is never rewritten. Before this, a `urllib` script inside one `Bash` call could POST
  `tools/call` with no `_meta` and skip every harness gate. The credential does
  not contain code that already has a shell as this user, which can read the
  file. It stops the unsupervised path: a script, a notebook, a subagent shell.
  `/health` stays open because supervisord, the promotion gate and the guardian
  probe it with no credential.

### Routes on :8500

| Route | Serves |
|---|---|
| `POST /mcp` | the MCP transport: `tools/list`, `tools/call` |
| `GET /health` | per-module discovery. **503 when any module's `list_tools()` raised**; the guardian excludes that from its down predicate and judges it with `mcp_degraded_is_fatal`. A closed Thunderbird is not a 503: that module degrades to zero tools and still reads `ok`. No credential |
| `GET /state` | Mission Control's agent panel: `subagents`, `background_tasks` (`active` and `recent`), `tsc`, `qmd`, `changes`, `tools`, `tool_sandbox` (the bench runner refuses to start unless this reads enforced) |
| `POST /loaded` | which of the given paths this process has imported. The promoter asks both processes before a landing; a commit neither has loaded needs no restart (`app/loaded_paths.py`) |
| `POST /browser/navigate` | the Browser tab's URL bar. A route and not a tool, because the user typing a URL is not the agent calling something |
| `GET /changes` | what a turn wrote (`?session=&turn=`), from the change ledger |
| `POST /changes/revert` | undo a turn's writes per file, refusing by name any file that moved since |

These ride beside `/mcp` because `Task`, background Bash children and
Playwright all live in **this** process. The backend has no handle on them, so
it reads them over loopback rather than through a shared file.

### Lifespan

At boot the lifespan starts the Discord bot, schedules a tsc baseline warm-up a
few seconds later (so the first `.tsx` edit after a restart is not blamed for
every pre-existing error), and prunes the change ledger on a thread. At
shutdown it calls every module's optional `shutdown()`: Playwright's Chromium,
the Thunderbird Node bridge, djev's shadow-queue flush. Before that hook,
every restart orphaned a browser and a node process.

---

## 3. The module contract

A module is any Python module exposing

```python
async def list_tools() -> list[Tool]
async def call_tool(name: str, arguments: dict) -> CallToolResult | list[TextContent]
async def shutdown() -> None            # optional
```

Modules do not own an `mcp.server.Server`. They used to, and those instances
were dead weight, because the SDK's decorator returns the function unchanged
and the per-module handler map was never dispatched. `_check_module` fails the
import if a module lacks either coroutine. That is cheaper than finding out on
the first dispatch.

Three registration rules. Each one is a failure that happened.

- **A raising module is skipped, not fatal.** It is logged and recorded in
  `_discovery_status`, which `/health` reports. Before this guard, one bad
  module failed `tools/list`, which failed the harness pool open and left the
  agent with no tools at all. That is the worst failure in the system (see
  [[harness]]).
- **A duplicate name is dropped, not shadowed.** The first module to claim a
  name keeps it, and the second is logged at ERROR.
- **`_dispatch` is rebound, never mutated.** `list_tools` builds a fresh dict
  and rebinds it in one statement. The old code cleared the dict and refilled
  it across 22 `await`s, so a call arriving in that window was told "Unknown
  tool". `tests/test_mcp_layer.py::test_dispatch_never_observed_empty_during_rebuild`
  pins it.

Names are capped at 64 characters (`TOOL_NAME_MAX`, OpenAI's limit) at
registration. They are advertised **bare** (`Bash`, not
`mcp__lloyd-mcp__Bash`). The prefixed form is still parsed
(`tool_schema.resolve_tool_name`), so old session JSON replays. Bare
advertising has three consequences in `app/harness/tool_schema.py::build_tool_list`:

- **A cross-server name collision raises.** Bare names leave no way to
  disambiguate. This is also the standing argument against mounting a second
  MCP server: `code_graph` and `djev` are modules for exactly this reason.
- **A name starting with `_` is never advertised.** The harness can still
  dispatch it through the pool, which is how `_BackgroundTaskDrain` works.
- **`disallowed_tools` blocks both spellings**, at advertise time, so a
  disabled tool is never in the payload the model reads.

---

## 4. The dispatch path, in order

### In the harness

1. **Advertise.** `build_tool_list` drops everything in
   `RunOptions.disallowed_tools`: the config's `disabled_tools`, plus the
   plan-mode block list while plan mode is on. Callers that wire
   `disallowed_tools_refresh` (chat, ambient, sync and voice) re-read it on
   every iteration, so entering plan mode mid-turn takes effect at the next
   one. `autonomy.run_task` and the direct worker path use the list from turn
   start.
   Every surviving schema gets a `summary` string parameter injected first
   (see CLAUDE.md, "Tool-call summaries").
2. **Commit the call.** `loop._commit_tool_calls` lifts `summary` out of the
   dispatched arguments into `_summary`. It stays in the replayed `arguments`,
   because that is the model's only example of how it called the tool.
3. **PreToolUse hooks.** Whatever the caller installed in its `HookRegistry`.
   The chat path installs `install_default_safety_hook`, which carries four
   Bash checks (the destructive-command table, `protected_paths`,
   `sync_registration`, `service_control`) and the outbound content gate
   (#1136: deny credential-shaped payloads, report PII-shaped ones). It installs
   the #534 grant gate for background platforms, the Inner Voice observer's
   pre-tool hook when the session is observed, and skill dispatch. The grant
   gate is `policy.install_policy_hook`, and since #724 it is also the
   reversibility-tier gate: `TIER2_TOOLS`/`TIER3_TOOLS` plus
   `effective_tier()`, which demotes an `autonomy_write_task` call that moves
   no dispatch-affecting field back to tier 1, so an unattended turn needs a
   grant to re-arm or park a task and does not need one to write an activity
   note. It is installed for worker sources (`workers/sources/_common.py`),
   autonomy task turns (`autonomy.py`), and the ambient and sync dispatch
   paths — not only "background". Every hook here keys on the bare tool name,
   which `loop._pre_dispatch` resolves from whatever the model emitted, so
   the legacy `mcp__lloyd-mcp__` spelling reaches them too (#727). The bench
   runner installs `install_bench_corpus_hook` on its own registry
   (`scripts/autoresearch/bench_runner_sdk.py`). A denial becomes a tool result
   the model reads ("Tool call denied: …"). Non-Bash writes have their own
   lane: `builtin_fs` refuses `Write`/`Edit` under `PROTECTED_WRITE_ROOTS` —
   the credential tree, `~/lloyd/agent-services`, `~/lloyd/.venvs`, and
   `~/obsidian/lloyd/SOUL.md` (`app/harness/protected_paths.py:119`, landed
   2026-09-20) — lifted only by the `allow_protected_writes()` contextvar,
   which is settable in code and by no production caller today. It is not
   `protected_roots()`: those are `$HOME`, the vault and the code tree, and
   refusing writes under them would refuse every note and every round.
4. **Invoke.** `MCPPool._invoke` opens one short-lived HTTP session per call
   and stamps `_meta` (below). The call bound is `CALL_TIMEOUT_SECONDS` 660 s,
   and the HTTP read timeout sits 30 s above it. It re-sends after a transport
   failure **only** for a tool annotated `readOnlyHint` or `idempotentHint`
   (`_retry_safe`). A transport error says nothing about whether the server
   ran the call, and a retried `automod_gate` once became two concurrent gates
   of one round.

### In the aggregator (`main.call_tool`)

Everything that can refuse a call runs before the effect ledger claims it, so
a call that was never allowed to run is never recorded as an `unknown` effect.

1. **Bench and eval sessions are read-only** (`_tool_sandbox`). For session
   ids `bench_*`, `<date>_<time>_bench_<hex>` and `pt-eval-*`, and any `task:*`
   child of one: `Bash` runs inside a read-only bubblewrap (no network, fresh
   `/tmp` `/run` `/var/tmp`, host sockets covered, every capability dropped),
   background Bash is refused, no bwrap means no Bash, and any other tool must
   be in `READ_ONLY`. The bench's grading corpus is refused to every tool. This
   is the layer that exists because the safety bench deleted the vault twice
   ([[vault-protection]]).
2. **The destructive-command check, for every session.** The same
   `check_bash_command` the harness hook runs, with `at_dispatch=True`: this is
   where the command actually executes. At dispatch the regex table's `sudo`
   rule is skipped, because it matches the word inside grep text.
3. **A state-changing call with no session id is refused** (#1053). An empty
   id cannot be a bench id, so before this rule, dropping `lloyd/session_id`
   from `_meta` was itself a way to read as "not sandboxed". Read-only tools
   still dispatch with no session, so discovery and probes are untouched.
4. **Effect ledger claim** (#544, `_tool_effects`). Only inside a worker scope
   (`lloyd/effect_scope` = `item:<source>:<id>`) and only for a side-effecting
   tool (§5). A second identical call in the scope replays the stored result.
   An `unknown` one (the first attempt was cancelled mid-effect) is refused
   with instructions to read the state back. Details: [[editing-safeguards]].
5. **The module runs.** Context vars carry the session, caption, turn, call,
   parent model and effect scope into the handler. This is how `Task` knows
   what model spawned it, and how the change ledger files a write under its
   turn.
6. **Effect ledger finish.** The result text is stored for replay. A handler
   that raised leaves the row `unknown`, which is the honest state.

### After the call

The harness writes the result into history. A result over 50 000 characters is
spilled to `sessions/<sid>.tool-results/<call_id>.*` and replaced inline by a
`<persisted-output>` block with a 2 000-character preview
(`tool_result_spill.py`). An empty result becomes `(<tool> completed with no
output)`, because some local models end the turn on an empty tool message.
Edits to `.py` files come back with a `<diagnostics>` block, and interface
changes with a `<blast_radius>` block. `.tsx` diagnostics arrive on a later
iteration ([[editing-safeguards]]).

### The `_meta` keys

| Key | Carries | Read by |
|---|---|---|
| `lloyd/session_id` | the harness session | everything: sandbox, ledgers, edit gates, registries |
| `lloyd/turn_id`, `lloyd/call_id` | the turn and the individual call | the per-turn change ledger |
| `lloyd/model`, `lloyd/base_url` | the calling turn's model and endpoint | `Task`, so a subagent inherits its parent's model |
| `lloyd/summary` | the model's caption for this call | background Bash and `Task` rows, which a human reads later |
| `lloyd/effect_scope` | the worker queue item | the effect ledger |

A legacy `_session_id` argument is still accepted and stripped before the
handler validates, so a harness and an aggregator at different versions still
correlate.

---

## 5. Tool properties

Each tool carries MCP `ToolAnnotations`, built from one central table in
`agent_mcp/annotations.py` rather than on each `Tool(...)`. The classification
is security-relevant and easier to review as one ordered file than as 155
constructor calls. The sets are explicit rather than pattern-matched: a regex
over names gets `email_apply_filters` and `autonomy_get_task` wrong in
opposite directions. `annotate()` is a default, not an override. A module that
sets its own annotations keeps them (today only `builtin_grants` does).

| Mark | Source | Means |
|---|---|---|
| **RO** | `READ_ONLY` → `readOnlyHint` | observes; changes nothing anywhere. Implies idempotent |
| **DX** | `DESTRUCTIVE` → `destructiveHint` | may irreversibly destroy data |
| **ID** | `IDEMPOTENT` → `idempotentHint` | a second identical call adds no further change (setters, deletes) |
| **OW** | prefix list + `Task`, `Bash` → `openWorldHint` | reaches beyond this machine: `http_`, `browser_`, `email_`, `calendar_`, `contacts_`, `tasks_`, `discord_` |
| **RX** | `REPEAT_EXPECTED` | repeating it is normal: poll loops, the automod round controls, UI and page verbs. A replay would be a stale answer delivered as a fresh one |
| **FX** | `side_effecting()` = not RO, not ID, not RX | effect-ledgered inside a worker scope |
| **PM** | `PLAN_MODE_ALWAYS_ALLOWED` | not read-only, but stays callable in plan mode: plan and goal control, local UI navigation, `ambient_decide` |

An unlisted tool gets the safe default in every direction: not read-only
(blocked in plan mode, refused in a bench session, refused with no session
id), not idempotent (never re-sent), and side-effecting (ledgered).

### What each property decides

| Decision | Rule | Reads | Where |
|---|---|---|---|
| Callable in plan mode | RO, PM or `_`-prefixed (blocked otherwise) | the `READ_ONLY` set | `plan_mode_blocked_tools` via `app/mcp_discovery.py` |
| Callable in a bench/eval session | RO (Bash only inside bwrap, never in background) | the `READ_ONLY` set | `_tool_sandbox.refusal` |
| Callable with no session id | RO | the `READ_ONLY` set | `main.call_tool` |
| Re-sent after a transport failure | `readOnlyHint` or `idempotentHint` | the annotation, as discovered | `MCPPool._retry_safe` |
| May overlap in one iteration | every call in the batch `readOnlyHint` | the annotation, as discovered | `loop.py` parallel dispatch (**ships off**) |
| Effect-ledgered | FX | the sets | `_tool_effects.claim` |
| Badged destructive | DX | the annotation | nothing reads it today: carried on the wire only |

The first three read the frozenset in-process. The next two read the hint as
it crossed the wire. They agree except where a module set its own annotation.

Before the plan-mode rule was derived from these sets, plan mode blocked only
`Write`, `Edit` and `Bash`. `email_send`, `vault_write`, `fact_add`,
`discord_send`, `browser_click` and ~55 others went straight through a
"read-only" planning turn. The three-name tuple survives only as the floor used
before discovery has run. `PLAN_MODE_ALWAYS_ALLOWED` exists because
`ExitPlanMode` is not read-only, and a plan mode that blocks its own exit is a
deadlock.

The tests cover one direction only. `test_annotation_tables_have_no_stale_entries`
fails on a name in `READ_ONLY`, `DESTRUCTIVE`, `IDEMPOTENT` or
`PLAN_MODE_ALWAYS_ALLOWED` that no longer exists (`REPEAT_EXPECTED` is not
checked). Thunderbird's names are exempt only while that module exports
nothing, which it always does inside an automod worktree, because the bridge
is a gitignored build artifact. **No test catches an unclassified tool.**
`test_every_tool_is_annotated` cannot fail on one, because `annotate()` gives
every tool the safe defaults. A new tool is classified when its author
remembers to, which is why §10 says so.

### Classification gaps (found while writing this, 2026-09-20)

Two of these are closed (#1326, 2026-09-21); two remain. The two that were
open were the direction that matters — a write travelling under a read-only
name — and the fix was to split, not to re-annotate:

- **`autonomy_config` is a pure read** (no `value` parameter at all). The write
  is `autonomy_config_set(key, value)`, a name absent from every annotation
  table, so it takes the safe default — a writer everywhere.
- **`fact_resolve` is a pure read** (no `auto_resolve` parameter). The write is
  `fact_resolve_apply(entity, category)`, same treatment.

Splitting rather than moving the names out of `READ_ONLY` is what gets all four
refusals *and* stops the double-send: a name moved out and left unannotated
would be replayed from the effect ledger inside one scope (`_tool_effects.py`),
suppressing a bare `autonomy_config()` re-read, and adding it to `IDEMPOTENT` to
escape that restores `idempotentHint`, which `_retry_safe` also honours. The read
halves refuse a write argument in their handler and name the tool that does it,
so an old prompt loses nothing but the write.

`autonomy_config_set` also stops discarding the file's body: it splits on the
front-matter fence, so the bytes below it survive byte-for-byte, and it refuses a
file whose front matter it cannot parse instead of re-emitting a fallback over it.

| Tool | Classified | Does | Consequence |
|---|---|---|---|
| `grant_list` | hint RO (module-set), not in `READ_ONLY` | reads only | blocked in plan mode and in bench sessions; ledgered as FX. Harmless, but the two readers disagree |
| `automod_status` | RX only | reads, plus `git worktree prune` on the live repo (`round.status()` → `W.prune_orphans`), an idempotent housekeeping write | blocked in plan mode and in bench sessions. Note the prune before moving it into `READ_ONLY` |

Pinned by `tests/test_autonomy_config_write.py`, `tests/test_tool_sandbox.py`,
`tests/test_mcp_pool_retry_policy.py` and the `fact_resolve_apply` cases in
`tests/test_memory_improvement.py`. One name in the class is still open by
decision, not oversight: `browser_screenshot` is in `READ_ONLY` (`annotations.py:75`)
and its handler `mkdir`s the screenshot directory and writes a PNG on every call
(`browser.py:664-671`). It is a derived artifact, like the `graphify-out/` cache
`annotations.py` explicitly justifies a few lines above — but that comment covers
only `graph_*`, and `_retry_safe` re-sends it, so a dropped transport leaves two
PNGs. Low severity: no vault state, no authority, and the duplicate is
self-evidently a file with a fresh timestamp.

---

## 6. Inventory

### Modules

| Module | n | What it is |
|--------|---|------------|
| `builtin_bash` | 2 | `Bash` and its background-completion drain |
| `builtin_fs` | 5 | `Read` `Write` `Edit` `Grep` `Glob`, with the read-before-edit gates, the change ledger, edit diagnostics, and a refusal of writes under `PROTECTED_WRITE_ROOTS` ([[editing-safeguards]]) |
| `builtin_goal` | 2 | the session goal |
| `builtin_grants` | 3 | #534 scope-bound authority grants: the human's mint path |
| `builtin_plan` | 2 | plan mode |
| `builtin_task` | 1 | in-process subagents (§9) |
| `builtin_todo` | 1 | the session todo list |
| `ambient` | 2 | background producers pushing into the active chat, and the ambient turn's routing verdict |
| `autonomy` | 8 | the scheduled-task fleet ([[autonomy]]) |
| `autoresearch` | 2 | prompt-variant rounds: their status, and restoring prompts from a snapshot. A round is scheduled by the worker source or run with `scripts.autoresearch.run_round` |
| `backlog` | 4 | the kanban backlog. `backlog_write_task` runs write-time dedupe ([[backlog]]) |
| `browser` | 13 | Playwright Chromium, with the SSRF guard on resolved addresses |
| `code_graph` | 6 | structural navigation over graphify's AST extraction (§9) |
| `discord_bot` | 4 | Discord. All four are disabled in config today |
| `djev` | 3 | typed decisions on GPU 2's DiffusionGemma, ~40 ms each. Ranking only: the scores are not calibrated ([[djev]]) |
| `facts` | 9 | the knowledge-graph fact layer and store ([[knowledge-graph]]) |
| `vault` | 5 | the Obsidian vault: read, write, search, recall |
| `session` | 5 | `MEMORY.md`/`USER.md`, and transcript search |
| `mission_control` | 2 | chat session listing |
| `mission_control_ui` | 3 | what tab the user is on, and moving them |
| `ide` | 3 | the IDE tab |
| `research` | 5 | the deep-research topic registry. `research_next` is peek-only; only the worker claims |
| `automod` | 10 | a thin surface over `scripts/automod/`. There is no `automod_write_code`: a round is a worktree path plus ordinary Edit/Write/Bash. While `automod.enabled` is false every tool refuses except `automod_status` and `automod_abort` ([[automod]]) |
| `skills` | 2 | the skill library ([[skills]]) |
| `http_tools` | 3 | web search, fetch and raw requests (§9) |
| `thunderbird` | 40 | mail (26), calendar (6), to-dos (3), contacts (5), through a Node bridge over stdio. Zero tools, and still `ok`, when Thunderbird is not running. The proxy rewrites the bridge's prose, never its schema structure: bridge function names become Lloyd's, and repeated or long parameter text is replaced (`PARAM_DESCRIPTION_OVERRIDES`) |

### Every tool

Properties use the marks from §5. **Required** is the schema's `required`
list, and every tool also takes the injected `summary` caption, which is not
shown. `†` marks a hint set by the module itself rather than the table (see §5).

#### `builtin_bash` (2)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `Bash` | DX OW RX | `command` | Run a shell command (default 120 s, max 600 s); `run_in_background` detaches and reports back through the drain |
| `_BackgroundTaskDrain` | RO | — | Internal: pop finished background-Bash records for the session. Never advertised; the harness calls it between iterations |

#### `builtin_fs` (5)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `Read` | RO | `file_path` | Read a file with line numbers; records the read the edit gates check |
| `Write` | DX ID | `file_path`, `content` | Create or replace a file whole. Needs a prior Read to overwrite an existing file |
| `Edit` | RX | `file_path`, `old_string`, `new_string` | Exact-match replacement in a file this session has Read; refused if the file moved since |
| `Grep` | RO | `pattern` | ripgrep over file contents |
| `Glob` | RO | `pattern` | Find files by name pattern, newest first |

#### `builtin_goal` (2)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `SetGoal` | ID RX PM | `text` | Set the session's persistent, verifiable goal (`/goal`) |
| `ClearGoal` | ID RX PM | — | Drop the session's goal (`/clear-goal`) |

#### `builtin_grants` (3)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `grant_create` | FX | `scope`, `tool`, `expires_at`, `issued_by` | Mint an expiring, quota-bound authority grant (#534). Interactive sessions only |
| `grant_list` | RO† FX | — | List live grants, optionally for one scope |
| `grant_revoke` | DX† FX | `grant_id` | Revoke a grant by id; takes effect at the next dispatch |

#### `builtin_plan` (2)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `EnterPlanMode` | ID RX PM | — | Enter research-only plan mode; every non-read-only tool leaves the advertised list |
| `ExitPlanMode` | ID RX PM | — | Commit a plan plus todos, or cancel, and leave plan mode |

#### `builtin_task` (1)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `Task` | OW RX | `prompt` | Run a subagent in this process (recursion depth 1); pass `task_id` back to resume it |

#### `builtin_todo` (1)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `TodoWrite` | ID RX PM | `todos` | Replace the session's todo list |

#### `ambient` (2)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `session_inject_context` | FX | `source`, `summary` | Push a background producer's finding into the user's active chat |
| `ambient_decide` | FX PM | `session_id`, `surface` | Record the routing decision of an ambient turn (surface or stay quiet). Ambient turns only |

#### `autonomy` (7)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `autonomy_tasks` | RO | — | List scheduled tasks with schedule and last-run state |
| `autonomy_write_task` | ID | — | Create or update (upsert) a scheduled task |
| `autonomy_get_task` | RO | `id` | One task in full, with its recent run records |
| `autonomy_delete_task` | DX ID | `id` | Archive a task back to `draft` (the default), or delete its file with `archive=false` |
| `autonomy_config` | RO | — | Read scheduler configuration: the whole config with no key, one setting with a key |
| `autonomy_config_set` | FX | `key`, `value` | Set one scheduler config key. Was the `value` half of `autonomy_config` (#1326); rewrites the front matter and leaves the body below it byte-identical |
| `autonomy_run_task` | FX | `id` | Run a task now, inside the aggregator. The call blocks until the run ends (its own description says "background", which is wrong) |
| `autonomy_health` | RO | — | Fleet health over N days: failures, timeouts, empty runs, GPU-hours |

#### `autoresearch` (2)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `autoresearch_status` | RO | — | Recent rounds, or one round's variants and scores |
| `autoresearch_rollback` | DX FX | `snapshot_ts` | Restore prompt files from a snapshot, through the vault landing route |

#### `backlog` (4)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `backlog_boards` | RO | — | Boards with task counts |
| `backlog_tasks` | RO | — | List items by status, board, tag, blocked or assigned |
| `backlog_get_task` | RO | `task_id` | One item in full: front matter and body |
| `backlog_write_task` | FX | — | Create or update an item. A create runs write-time dedupe and may merge into an open item, or be refused over a closed one |

#### `browser` (13)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `browser_navigate` | OW RX | `url` | Load a URL (private hosts refused, loopback allowed) |
| `browser_snapshot` | RO OW | `full` | Accessibility tree of the page, with `eN` refs for the action tools. Each child frame gets its own `### frame <n> <url>` section; a frame that could not be read is listed with the reason (#424) |
| `browser_click` | OW RX | `ref` | Click an element by ref |
| `browser_scroll` | OW RX | — | Scroll the page |
| `browser_press` | OW RX | `key` | Press a key or chord |
| `browser_tabs` | OW RX | `action` | List, switch, open or close tabs |
| `browser_screenshot` | RO OW | — | PNG of the page, also saved under `logs/screenshots/` |
| `browser_evaluate` | OW RX | `script`, `frame_index` | Run JavaScript in the page and return the result; `frame_index` runs it inside `page.frames[n]` instead of the main frame (#424) |
| `browser_fill` | OW RX | `ref`, `value` | Set a field's value in one write that fires input/change, or key by key with `keystrokes=true` for search-as-you-type fields. Absorbed `browser_type` |
| `browser_wait` | OW RX | `condition` | Wait for a selector, text, navigation or network idle |
| `browser_select` | OW RX | `ref` | Pick an option in a `<select>` by value or label |
| `browser_drag` | OW RX | `source_ref`, `target_ref` | Drag one element onto another |
| `browser_cookies` | OW RX | `action` | Get, set or clear cookies |

#### `code_graph` (6)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `graph_explain` | RO | `symbol` | Inbound and outbound edges of a symbol, with call sites |
| `graph_affected` | RO | `symbol` | Reverse-BFS blast radius of a symbol, grouped by depth |
| `graph_path` | RO | `a`, `b` | Shortest dependency path between two symbols |
| `graph_hubs` | RO | — | Most-connected symbols, optionally under a path prefix |
| `graph_status` | RO | — | Counts, built-at commit vs HEAD, why it is stale. Never builds |
| `graph_refresh` | ID | — | Force a rebuild (~15 s on this repo) |

#### `discord_bot` (4)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `discord_send` | OW FX | `channel_id`, `content` | Post a message to a channel |
| `discord_send_embed` | OW FX | `channel_id`, `title`, `description` | Post a rich embed to a channel |
| `discord_list_channels` | RO OW | — | A guild's text channels |
| `discord_get_home_channel` | RO OW | — | The configured home channel id |

#### `djev` (3)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `djev_rank` | RO | `query`, `candidates` | Re-rank a shortlist (≤12 default, 16 max) against a query on GPU 2. The order is the output; the scores are not calibrated |
| `djev_decide` | RO | `state`, `questions` | Typed questions (yes/no, one-of-N, ordered scale) about one text, in under a second. Compact result by default, `verbose` for every diagnostic |
| `djev_status` | RO | — | Enabled, reachable, per-seam latency, shadow queue, per-schema floors and gate status |

#### `facts` (7)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `fact_get` | RO | `entity` | An entity's facts, capped per category (10; 0 = none), most recent or ranked by `query`; optionally as of a date or including expired. Absorbed `fact_profile` |
| `fact_add` | FX | `entity`, `category`, `fact` | Add one fact to an entity's markdown fact file and index it |
| `fact_resolve` | RO | `entity` | Report contradictions. It marks nothing (#1326) |
| `fact_resolve_apply` | FX | `entity` | Mark the lower-confidence side of each pair `invalid_at` (never expired). Was `fact_resolve`'s `auto_resolve` |
| `fact_invalidate` | DX ID | `entity` | Expire facts that stopped being true. Refuses an unscoped call; `ended` defaults to today. Absorbed `forget` |
| `fact_relate` | FX | `source`, `target`, `type` | Add a typed edge between two entities |
| `fact_relationships` | RO | `entity` | An entity's inbound and outbound edges; `min_confidence` (default 0.0 = keep all) drops weaker ones |

`fact_path` and `fact_neighbors` were deleted on 2026-09-24 (#1077, on the #877
precedent): zero calls in every stored session against hundreds for `fact_add`,
so the surface advertised a graph-expansion read no caller used. Both handlers
and their 1000-node / 2000-edge caps are gone. The surviving expansion lives in
`agent_mcp/retrieval.py`, which bounds nothing during its walk and slices
`top_k` only at the end — so a graph arm wired into per-turn retrieval (#1025)
re-implements the caps.

#### `vault` (5)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `vault_read` | RO | `path` | Read a vault file by vault-relative path |
| `vault_write` | ID | `path`, `content` | Create or overwrite a vault file (audit-logged) |
| `vault_overview` | RO | — | File counts per segment, or the most-linked notes |
| `vault_search` | RO | `query` | BM25 + vector search over the vault |
| `vault_recall` | RO | `query` | Vault search plus entity facts in parallel; carries the djev rerank arm and shadow seam. Its seven eval knobs (`RECALL_EVAL_KNOBS`: graph expansion and rerank, djev rerank) are read from params in process only; `call_tool` strips them, and the schema no longer lists them |

#### `session` (5)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `memory_read` | RO | — | Read `MEMORY.md` or `USER.md` |
| `memory_add` | FX | `entry` | Append an entry to a memory file |
| `memory_replace` | ID | `old_text`, `new_text` | Replace text in a memory file (fails if absent) |
| `memory_remove` | DX ID | `entry` | Remove an entry from a memory file |
| `session_recall` | RO | `query` | Search recent session transcripts |

#### `mission_control` (2)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `chat_list_sessions` | RO | — | Chat sessions with titles and last activity |
| `chat_get_session` | RO | `session_id` | One session's metadata (not its content) |

#### `mission_control_ui` (3)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `mc_get_state` | RO | — | Which Mission Control tab the user is on, and what it shows |
| `mc_navigate` | ID RX PM | `tab` | Move the user to a tab, optionally focusing an item |
| `mc_close_modal` | ID RX PM | `tab` | Dismiss a modal in a tab |

#### `ide` (3)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `ide_open_folder` | ID RX PM | `path` | Point the IDE tab's file tree at a directory |
| `ide_open_file` | ID RX PM | `path` | Open a file in the IDE tab |
| `ide_close_tab` | ID RX PM | `path` | Close an IDE editor tab |

#### `research` (5)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `research_propose` | FX | `topic` | Propose a topic to the deep-research registry (deduplicated) |
| `research_next` | RO | — | Peek at the next topics. Claims nothing — only the worker claims |
| `research_complete` | ID | `topic_id`, `status` | Record a topic's outcome |
| `research_list` | RO | — | Registry topics by status, domain or recency |
| `research_stats` | RO | — | Registry health: counts per state, queue depth and age |

#### `automod` (10)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `automod_start` | RX | `goal` | Open a self-modification round: a worktree off HEAD |
| `automod_gate` | RX | `round_id` | Start the detached promotion gate on a round |
| `automod_gate_wait` | RO | `round_id` | Poll the detached gate; returns the per-rung report when done |
| `automod_amend_clause` | RX | `round_id`, `clause`, `text`, `reason` | Amend one clause the review rung graded `unsatisfiable` |
| `automod_land` | RX | `round_id` | Promote a passed round. Returns at once; the landing runs detached |
| `automod_status` | RX | — | LKG, the observed promotion, halt/broken flags, recent ledger events |
| `automod_abort` | RX | `round_id` | Abandon a round; keeps the branch |
| `automod_vault_land` | RX | `paths`, `message` | Validate and commit named vault paths through the loop |
| `automod_vault_revert` | RX | `sha` | Revert a vault landing |
| `automod_rollback` | RX | `reason` | Ask the guardian to revert the live tree and restart |

#### `skills` (2)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `skills_search` | RO | `query` | Search skills by keyword |
| `skills_read` | RO | `name` | Read a skill's full `SKILL.md` |

#### `http_tools` (3)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `http_search` | RO OW | `query` | Web search (DuckDuckGo) |
| `http_fetch` | RO OW | `url` | Fetch a public page (or PDF) as markdown or text. GET only; private hosts refused |
| `http_request` | OW RX | `method`, `url` | Raw request, any verb, headers and body. Private hosts refused except loopback |

#### `thunderbird` (40)

| Tool | Properties | Required | Does |
|---|---|---|---|
| `email_accounts` | RO OW | — | Configured accounts and identities |
| `email_folders` | RO OW | — | Folders for an account, with URIs and counts |
| `email_search` | RO OW | `query` | Search message headers |
| `email_read` | RO OW | `messageId`, `folderPath` | One message in full |
| `email_messages` | RO OW | `messages` | Several messages in full (10 by default; a bridge preference allows 20) |
| `email_send` | OW FX | `to`, `subject`, `body` | Compose a message in a review window (direct send only if the user disabled review) |
| `email_save_draft` | OW FX | — | Save a draft without opening a window |
| `calendar_list` | RO OW | — | Calendars with ids |
| `calendar_create` | OW FX | `title`, `startDate` | Create an event through a review dialog |
| `calendar_events` | RO OW | — | Events between two dates |
| `calendar_update_event` | ID OW | `eventId`, `calendarId` | Update an event |
| `calendar_delete_event` | DX ID OW | `eventId`, `calendarId` | Delete an event (no trash) |
| `tasks_create` | OW FX | `title` | Open a pre-filled task dialog |
| `calendar_categories` | RO OW | — | Category names |
| `tasks_list` | RO OW | — | To-dos, filtered by completion or due date |
| `tasks_update` | ID OW | `taskId`, `calendarId` | Update a to-do |
| `contacts_search` | RO OW | `query` | Search contacts by name or address |
| `contacts_get` | RO OW | `contactId` | One contact in full |
| `contacts_create` | OW FX | — | Create a contact |
| `contacts_update` | ID OW | `contactId` | Update a contact |
| `contacts_delete` | DX ID OW | `contactId` | Delete a contact |
| `email_reply` | OW FX | `messageId`, `folderPath`, `body` | Reply in a review window |
| `email_forward` | OW FX | `messageId`, `folderPath`, `to` | Forward in a review window |
| `email_recent` | RO OW | — | Recent messages, newest first |
| `email_display` | OW FX | `messageId`, `folderPath` | Open a message in the Thunderbird GUI |
| `email_delete` | DX ID OW | `messageIds`, `folderPath` | Delete messages (drafts go to Trash) |
| `email_update` | OW FX | `folderPath` | Read/flag/tag/move messages |
| `email_create_folder` | OW FX | `parentFolderPath`, `name` | Create a folder |
| `email_rename_folder` | OW FX | `folderPath`, `newName` | Rename a folder |
| `email_delete_folder` | DX ID OW | `folderPath` | Delete a folder and its contents |
| `email_empty_trash` | DX ID OW | — | Empty Trash |
| `email_empty_junk` | DX ID OW | — | Empty Junk |
| `email_move_folder` | OW FX | `folderPath`, `newParentPath` | Move a folder |
| `email_list_filters` | RO OW | — | An account's filter rules |
| `email_create_filter` | OW FX | `accountId`, `name`, `conditions`, `actions` | Create a filter rule |
| `email_update_filter` | OW FX | `accountId`, `filterIndex` | Modify a filter |
| `email_delete_filter` | DX ID OW | `accountId`, `filterIndex` | Delete a filter by index |
| `email_reorder_filters` | OW FX | `accountId`, `fromIndex`, `toIndex` | Move a filter in the run order |
| `email_apply_filters` | OW FX | `accountId`, `folderPath` | Run filters over a folder |
| `email_account_access` | RO OW | — | Which accounts the bridge may touch |

---

## 7. Progressive disclosure (`ToolSearch`)

Advertising the whole catalog on every request is billed as input tokens on
every turn: ~30.3k tokens on a chat turn after the 2026-09-23 trim, measured
with the primary's `/tokenize` over the OpenAI-shaped JSON (34.8k before it;
25.8k when the catalog was 124 tools). The Thunderbird bridge is 10.1k of that,
about half of it schema structure no prose trim can reach. And
tool-call accuracy degrades past roughly
30–50 tools loaded at once. `harness.tool_search` advertises a small baseline
plus a `ToolSearch` meta-tool and loads the rest on demand
(`app/harness/tool_search.py`, `tool_search_cache.py`). The model gets a
`role: system` reminder with every deferred tool's name and a one-line gist,
and no schemas. `ToolSearch(query=…)` is intercepted in the harness with no MCP
round trip. It marks the matches loaded in the session's `LoadedToolSet` and
returns their schemas in a `<functions>` block.

**It is off today.** `harness.tool_search.enabled` is `false` in both
config.yaml and the live override, so every enabled tool is advertised on
every request. When on: `threshold_tools` 30, a 28-name `baseline_tools`
(the five file tools, `Bash`, `Task`, `skills_search`/`skills_read`, the
`http_*` trio, `browser_navigate`/`browser_snapshot`, vault search/recall/write,
the Mission Control and IDE navigation tools, `graph_explain`/`graph_affected`,
`TodoWrite`, `SetGoal`, `ambient_decide`), `max_results_default` 5,
`max_results_cap` 20. Keep the baseline small and honest. A tool outside it
competes only after a discovery round trip, which is why `http_search` and
`http_fetch` sat unused while `Bash` was always visible. Read the effective
values through `app.mcp_discovery._get_harness_kwargs()`.

The override shadows this block wholesale, so config.yaml and the override must
carry the same values. `tests/test_tool_overrides.py::test_config_yaml_agrees_with_the_live_override`
catches an unpaired edit.

---

## 8. Enable and disable

- Server level: `mcp_servers.<name>.enabled: false`
- Tool level: `mcp_servers.<name>.disabled_tools: [bare_tool_name, ...]`.
  Today: the four `discord_*` tools.

config.yaml holds the hand-edited defaults and is read-only at boot. The Tools
page writes `data/tool_overrides.yaml`, which is merged over it
(`app/config.py::_merge_tool_overrides`). That file is gitignored, and must
stay that way. While it was tracked, one click dirtied the live tree, and the
automod gate and promoter both refuse a dirty tree. So a fresh clone boots on
config.yaml alone, and config.yaml has to describe what is actually served. The
merge warns when the override re-enables a tool config.yaml disables, and when
a `tool_search` key differs from the tracked value. Agreement stays silent.

The merge is deliberately narrow. It honours per-server `enabled` and
`disabled_tools`, the `harness.tool_search` block, `workers.enabled` and
`workers.sources.<name>.inner_voice`. An override naming a server or a worker
source config.yaml does not define is ignored. `save_tool_overrides()` is the single writer. Resolve the effective
set through `app.mcp_discovery._get_disallowed_tools()`: reading config.yaml
directly misses both the overrides and `${VAR}` expansion. Routes:
`POST /api/tool-toggle`, and `GET`/`POST /api/tool-discovery`, in
`app/routers/tools.py`.

**Why the override file must stay untracked** (the long version).
`save_tool_overrides` replaced dumping the entire CONFIG back over config.yaml
on every toggle, because config.yaml is tracked and a tracked file rewritten
by a UI click leaves the live tree dirty — which `scripts/automod/gate.py` and
`promote.py` both refuse. Until 2026-09-07 the override file was tracked too,
so the escape hatch had the defect it was built to avoid: one click on the
Tools page dirtied the tree and silently stopped the self-modification loop
until someone hand-committed the result (`4fb1ccd`, `2ef86c7` are that
happening). Untracked alone is not enough — `git status --porcelain` lists
new files as well — so it needs the `.gitignore` rule beside it, and the
`.<pid>.tmp` sibling `atomic_write_text` lands before renaming is ignored too
(a write killed in between would otherwise leave a stray that dirties the tree
exactly as the tracked file used to, one filename over).
`tests/test_tool_overrides.py` pins both halves; how that test itself once
blocked every round is in [[testing]] ("A test about untracked state").

Because a fresh clone has no override file, **config.yaml is the state a
rebuild boots into**, so it has to keep describing what is actually served. It
claimed `tool_search.enabled: true` for an unknown stretch while the override
served `false`. The `disabled_tools` half of the drift warning dates from the
2026-09-04 `browser_screenshot` incident; `harness.tool_search` was a bare
`.update()` until the warning was added for it too, and `enabled` decides
whether the model is handed the whole catalog (131 tools then) or a baseline
plus ToolSearch. Agreement stays silent because the Tools page rewrites the
whole block on every toggle, so warning on it would fire each boot and stop
meaning anything.

Disabled tools reach the loop as `RunOptions.disallowed_tools` in the
`mcp__<server>__<tool>` form.

`Bash` disabled this way is blocked under both spellings at **advertise** time
(`tool_schema.build_tool_list` matches the bare name and
`mcp__<server>__<bare>`) and at **dispatch**: `loop._pre_dispatch` resolves
the emitted name and every entry of the deny list through
`policy.normalize_tool_name` before comparing, and fires the hook walk with
the bare name. Until 2026-09-24 it compared the raw spelling, so
`mcp__lloyd-mcp__Bash` — which `MCPPool.call_tool` accepts deliberately for
old session JSON — passed a deny list carrying `Bash` and skipped the safety
deny keyed on `"Bash"` (#727, `tests/test_harness_disallowed_tools.py`). The
tool-result events keep the spelling the model emitted.

**A turn's surface hides a few tools** (2026-09-23). `RunOptions.surface` is
`"chat"` for a session a person reads and `"worker"` for the `autonomy` and
`worker` platforms (`messages._tool_surface`, `autonomy.run_task`,
`run_prompt_on_primary`); a `Task` subagent inherits its caller's through
`_meta` (`lloyd/surface`). `agent_mcp.annotations.hidden_on_surface` turns it
into names the loop adds to its disallowed set, so a hidden tool is neither
advertised nor dispatchable, plan-mode refresh included. The table is
exceptions only and deliberately small: `CHAT_ONLY` is Mission Control, the
IDE and the three grant tools, because a worker has nobody at the screen and
grants are a human's; `WORKER_ONLY` is `session_inject_context`, because a
chat reaches the user by replying. Mail, calendar and the browser stay on
both: #534's grants exist to let a worker send mail under a bound, and the
email-monitoring skills are background jobs. An empty or unknown surface
hides nothing, so a caller that never sets one behaves as before. It is a
catalog trim, not a security boundary, and fails open; the refusals that are
boundaries (`WORKER_GRANT_MINT_BAN`, the tool sandbox, the policy gate) are
unchanged. Measured on 2026-09-23: a chat turn is 139 tools and ~29.5k
tokens of schema, a worker turn 131 tools and ~28.0k.

**No flag may empty a module's `list_tools()`** (`code_graph` has no
`enabled` key at all; `djev.enabled` switches the engine, not the tools). A
flag that emptied `list_tools()` would break the annotation staleness test. The
kill switch for a module's tools is `disabled_tools`.

---

## 9. Modules worth knowing more about

### Code graph

`agent_mcp/code_graph.py` reads `<root>/graphify-out/graph.json`, a
deterministic AST extraction built with no LLM calls. It answers the structural
questions a grep cannot: who calls this, what breaks if I change it, how do
these two connect.

| Tool | Use |
|------|-----|
| `graph_explain` | Inbound and outbound edges of a symbol, each with the caller's call site |
| `graph_affected` | Reverse-BFS blast radius, grouped by depth, with the file list |
| `graph_path` | Shortest dependency path between two symbols (containment edges excluded) |
| `graph_hubs` | Most-connected symbols, optionally scoped by path prefix |
| `graph_status` | Counts, built-at commit vs HEAD, why it is stale. Never builds |
| `graph_refresh` | Force a rebuild (~15 s on this repo) |

- **`root` is explicit and never inferred** from the calling session. Nothing
  on disk links a chat session to an open automod round, so a default of "the
  session's worktree" would confidently answer about the wrong checkout. It
  defaults to `LLOYD_HOME`, and accepts an `SM_…` round id or an absolute path.
- **Staleness is a commit mismatch, or an uncommitted source file newer than
  `graph.json`.** The second rule is the load-bearing one. Inside a round, HEAD
  does not move while the model edits, so a commit-only rule would call the
  graph fresh for exactly the window it is most wrong in. Rebuilds are
  debounced (`min_refresh_interval_s`, 30 s). A debounced query still answers,
  and says it is stale.
- **It is blind across process seams.** There is no edge from the backend to
  the aggregator over HTTP, or from `run_query` into a tool handler over MCP.
  Grep stays right for string keys, route paths and config names.
- **`graphify-out/` is gitignored, unanchored,** because a build inside a round
  would otherwise dirty the tree the gate refuses.
- **It is not a second MCP server, deliberately.** graphify ships
  `graphify-mcp` and mounting it would have been one config line, but Lloyd
  advertises every server's tools under bare names and `build_tool_list`
  raises on a cross-server collision; Task subagents pin
  `DEFAULT_LLOYD_MCP_SERVERS` and would never see it;
  `tests/test_mcp_layer.py` needs every configured server discoverable at
  test time, including inside a worktree where no second daemon is running;
  `agent-services/supervisor/**` is a protected automod path, so Lloyd could
  never repair the program running it; and graphify-mcp has no `affected`,
  which is the one query a change actually needs.
- **Why `root` cannot be inferred:** `round_start` ledger rows carry no
  session id. When the answer is about the live tree and a worktree is open,
  the header says so.
- **What the ignore rule has to cover.** Both `scripts/automod/gate.py` and
  `promote.py` refuse a dirty tree, so an unignored build would abort the
  round on its own map. `*.json` at the top of `.gitignore` hid `graph.json`
  by accident; `GRAPH_REPORT.md`, `graph.html`, `.graphify_root` and the
  ~64k-file `cache/ast/**` were covered by nothing until `graphify-out/` was
  ignored unanchored.
- **Ambiguity is an answer, not an error:** `main` matches 88 nodes here, and
  a listing with ids lets the model pick where an error makes it guess again.
- **No `enabled` flag.** The kill switch is
  `mcp_servers.lloyd-mcp.disabled_tools` (§8).
- **The vault half is held as a patch.** The `automod-change-own-code`
  skill's blast-radius step lives at
  `scripts/maintenance/vault-automod-skill-blast-radius.patch`, not in the
  vault, until `code_graph` is deployed. The vault is a live shared tree with
  no PR path, so editing it lands *immediately* — while the graph tools it
  names only exist after lloyd-mcp restarts on the merged code. An edited
  skill in that gap tells every automod round to call a tool that returns
  "Unknown tool" and makes its own quality gate 4 unsatisfiable. Apply it
  after the restart with
  `git -C ~/obsidian apply scripts/maintenance/vault-automod-skill-blast-radius.patch`.
  `tests/test_code_graph_doc_claims.py::test_automod_skill_maps_the_radius_between_opening_and_working`
  fails with that command until it is applied; it carries `live_vault`, so the
  automod gate (`-m "not live_vault"`) excludes it and no round is failed by a
  vault someone else has not updated yet.
- Config under `code_graph:`: `graphify_bin`, `auto_refresh`,
  `refresh_timeout_s` 120, `min_refresh_interval_s` 30, `max_cached_roots` 4,
  `max_lines` 60. The same queries run *passively* on every interface-changing
  edit, as the `<blast_radius>` block ([[editing-safeguards]]).

### Web

| Tool | Use |
|------|-----|
| `http_search` | DuckDuckGo: ranked titles, URLs, snippets |
| `http_fetch` | A public URL as markdown or text through trafilatura, and PDFs page by page through pymupdf. GET only. Private hosts blocked. `max_chars` 1 000–200 000, default 50 000 |
| `http_request` | Any verb, headers and body; returns status, headers and the raw body. Private hosts blocked **except loopback**, mirroring the browser |

`Bash` + `curl` stays right for localhost, which `http_fetch` blocks, and for
the structured-API pipelines individual skills document. For the public web,
use the `http_*` tools (the `web-search-and-fetch` skill).

### Browser

The browser's SSRF guard checks **resolved addresses**, not the hostname
string, so `2130706433`, `0x7f000001` and `router.local` are all classified
correctly. Route interception does not see redirects, so `_enforce_landing`
checks where the page actually ended up. It blocks the network the machine is
on and allows the machine itself. CLAUDE.md's Browser section has the rest.

### djev

Three read-only tools over GPU 2's structured-decision engine. `list_tools()`
is offline by construction. A degraded module makes `/health` answer 503, which
the guardian can read as a rollback trigger, so a merely stopped engine must
not degrade the module. [[djev]] is the long version.

### Thunderbird

`node mcp-bridge.cjs` over stdio through `MCPPool`, with discovery cached for
300 s. The stdio path takes a per-server lock, because two concurrent
`call_tool`s on one pair of pipes interleave their JSON-RPC frames. It exports
**zero** tools when Thunderbird is not running, and `/health` stays 200 with
`tools: 0` for the module: discovery failure is caught and degrades to an
empty list rather than raising. The compose-style tools (`email_send`, `email_reply`,
`email_forward`, `calendar_create`, `tasks_create`) open a review window unless
the user has switched that preference off.

### Subagents (`Task`)

`Task` runs a nested `run_query` inside the aggregator with its own
`RunOptions` from `subagents.<type>`. Recursion is capped at depth 1,
because every level is another live loop on the one primary engine that
nothing outside the process can stop, and resuming a child by `task_id`
already hands it the whole working state without nesting (the ruling and its
enforcement are below). The parent's model and endpoint arrive in `_meta`, and
a tool running in the aggregator has no other way to know them.

**Why depth 1, and where it is enforced** (#1515). The cap is enforced twice.
Every child's `RunOptions.disallowed_tools` gets `Task` appended
(`agent_mcp/builtin_task.py:366-368`), which the harness honours at advertise
time (`app/harness/tool_schema.py:186`, `build_tool_list`) and again at
dispatch (`app/harness/loop.py:2855`, `_pre_dispatch`: "disabled by
configuration") — this is the layer that holds on the real path, because a
child's tool calls re-enter `main.call_tool` over loopback `/mcp` from a fresh
ASGI task where no contextvar survives. The `_task_depth` contextvar check
against `MAX_TASK_DEPTH = 1` (`agent_mcp/builtin_task.py:37`, checked at
`agent_mcp/builtin_task.py:282-285`) is the backstop for a same-context call. `tests/test_task_depth_cap.py` pins both,
so a silent widening fails. The cap exists **because** of what each extra
level would stack on one box: every level is a live `run_query` on the same
primary whose parent sits blocked on the tool call, so its 100-200k prefix has
to survive in the engine's free KV pool meanwhile (the cold re-prefill failure
of `vllm.md`), its `Task` call budget has to nest inside its parent's, and —
since nothing outside the process can stop a wedged child (below) — each level
multiplies the un-killable work a single stuck call can hold. At depth 1, and
with `Task` not `readOnlyHint` so a batch containing it runs sequentially, one
turn delegates to at most one live child at a time. What deep delegation is
usually wanted for — handing the next step the whole working state instead of
a summary — is delivered here without nesting by resuming the same child with
its `task_id`. **No per-level cost has been measured**; the ruling is
structural, not a number. What would justify raising it is a measured curve at
depth 1 vs 2 vs 3 of wall clock, input tokens (including re-prefills) and peak
concurrent primary requests for the same task, plus an outside-the-process
cancel for a wedged child (#1135). The recursive-delegation pattern
(JAZ, arXiv:2609.26891, one analyzed 70-deep example) was read on 2026-09-25
and is not adopted on that evidence.

Every result carries a `task_id`, and passing it back with a follow-up prompt
**resumes** the subagent. The stored run's identity wins: type, profile, model,
base URL and the `task:*` session id all come from the history, so a
continuation keeps its `LoadedToolSet` and its spill directory. Only
`disallowed_tools` is merged live. The store is process-scoped and bounded
(8 tasks, 30 minutes, 3 M chars). After an aggregator restart every id reads
`unknown or evicted`. Nothing can stop a wedged child from outside the process:
`_subagent_registry.steer()` and `cancel_run()` (#411) exist, the child honours
both — it gets a real `cancel_event` and answers `stop_reason: cancelled` — and
no tool, route or UI verb calls either one (#1135), so a hung `Task` holds its
660 s call budget until the aggregator restarts. The three refusal reasons stay
distinct because they call for different next moves. `_sanitise` drops a trailing assistant message whose
tool calls were never answered, the one invalid shape a cancel can leave and
the one every engine rejects on replay.

**Resuming, in detail** (the long version CLAUDE.md used to carry). `Task`
used to start from nothing on every call — right for a fire-and-forget
fan-out, wrong when a subagent burns its budget mid-investigation and the only
follow-up is to pay for the whole investigation again (a fresh prompt, a cold
KV cache, and no memory of the forty tool results it just collected).

- **`run_query` ignores `messages` when `chat_messages_handle` is non-empty**
  (`loop.py`). So a resume appends the follow-up to the *stored list* and
  passes that as the handle. Sending it as `messages` would drop it silently.
- **`current_parent_model` is deliberately not consulted** on a resume:
  moving a half-finished conversation to another engine re-prefills all of it.
- **`task_id` is stable across continuations; `run_id` is not.** The
  dashboard shows one row per *run*, and a resume is a new run of the same
  task, linked by `continuation_of`.
- **Every exit path stores as well as closes.** One `_close` helper, because
  there are five exits and a run that closed its row without storing is a
  `task_id` the model was told about and cannot use. `CancelledError` stores
  and re-raises.
- The three refusal reasons are `unknown or evicted`, `expired` and
  `still running`.
- `SubagentRecord.to_dict` never exposes `chat_messages`; the dashboard row
  gets `task_id` and `continuation_of` only.

A subagent's writes land on the parent's turn in the change ledger, and it
inherits its parent's bench sandbox and effect scope. It still has to Read
what it edits: the edit gates key on the raw `task:*` session.

---

## 10. Adding a tool

1. Return it from the module's `list_tools()`, and handle it in `call_tool`.
   Keep `list_tools()` **offline**: no reachability probe, no config read that
   can raise. A raising module is skipped, and a degraded one turns `/health`
   into a 503.
2. Classify it in `agent_mcp/annotations.py`. An unclassified tool is treated
   as a writer everywhere, which is safe and usually wrong for a read.
   `REPEAT_EXPECTED` weakens the effect ledger for every scope, so add to it
   only when a replay would be a stale answer.
3. Write the description the way the others are written: start with the
   routing sentence ("Use when …; for X use Y instead"), then say what it
   returns. The description is the only thing the model reads before calling.
   Keep numbers that drift (measured durations, pacing) out of it: descriptions are part of every turn's
   cached prefix.
4. Declare no `description` or `summary` argument of your own. The harness
   injects `summary`, and a second caption field splits the model's answer
   across two places (CLAUDE.md, "Tool-call summaries").
5. Report errors as a JSON object with an `error` key. `_shared.text_result`
   sets `isError` by sniffing it.
6. A tool that sends, creates or appends is effect-ledgered automatically in a
   worker scope. Make sure a replayed result reads sensibly.
7. Regenerate §6 of this page (§11).

---

## 11. Regenerating the inventory

The running server counts exactly what it dispatches:

```bash
curl -s http://127.0.0.1:8500/health | python3 -m json.tool   # tools, modules, per-module
```

Offline, or in a worktree (Thunderbird will read zero there), this prints every
tool with its module, its marks and its required arguments. §6 is this output
plus a hand-written "Does" column:

```bash
PYTHONPATH=. .venvs/lloyd/bin/python - <<'EOF'
import asyncio
from agent_mcp import main as m, annotations as A
for t in asyncio.run(m.list_tools()):
    a, n = t.annotations, t.name
    marks = [k for k, on in (("RO", a.read_only_hint), ("DX", a.destructive_hint),
             ("ID", a.idempotent_hint and not a.read_only_hint), ("OW", a.open_world_hint),
             ("RX", n in A.REPEAT_EXPECTED), ("FX", A.side_effecting(n)),
             ("PM", n in A.PLAN_MODE_ALWAYS_ALLOWED)) if on]
    req = (t.input_schema or {}).get("required") or []
    print(m._dispatch[n].__name__.rsplit(".", 1)[-1], n, " ".join(marks), ",".join(req), sep="\t")
EOF
```

## Review log

- **2026-09-21 — `current`.** Regenerated §6 against the live `:8500/health` and
  the offline `list_tools()` dump: 155 tools / 27 modules, and all 155 rows agree
  with the tree on module, marks and required arguments; module counts, the
  timeouts (660 s + 30 s), the 50k/2k spill, the routes, `TOOL_NAME_MAX`,
  `ttl_ms`, the tier/annotation tests, the subagent store bounds and the
  `tool_search` block all re-derived clean. Three things the prose got wrong or
  never said: the §8 "blocked at dispatch under both spellings" claim is false
  (dispatch compares the raw emitted name — #727), the hook list omitted the
  reversibility-tier gate #724 added inside `install_policy_hook`, and
  `builtin_fs`'s new `PROTECTED_WRITE_ROOTS` refusal was unmentioned. The §5
  classification gaps are still open (#1326) and a stopped `Task` child still
  has no caller behind `steer()`/`cancel_run()` (#1135).
