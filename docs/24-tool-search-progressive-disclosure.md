# 24 — ToolSearch: Progressive Tool Disclosure

## Context

Lloyd's harness historically advertises every MCP tool to vLLM on every chat
completion request. With the catalog at ~72 tools across 17 modules (and growing),
this is approaching two failure modes:

1. **Token tax.** Tool schemas are billed as input tokens on *every* turn.
   At ~150–250 tokens per tool, the catalog adds 10–20k input tokens per turn —
   a fixed overhead that grows with the catalog and dominates short turns.
2. **Accuracy degradation.** Function-calling fine-tunes start losing tool-selection
   accuracy somewhere around 30–50 simultaneously-loaded tools. Past 50, the model
   begins hallucinating tool names or picking near-misses.

This is the same wall the OpenAI ecosystem hits — Assistants API caps tool counts
at 128, and even Chat Completions degrades well before that. The Claude Agent SDK
sidesteps it with a mechanism called **tool search**: instead of advertising every
tool, advertise a small baseline plus a `ToolSearch` meta-tool, then load full
schemas on demand when the model asks for them. Once a tool's schema has been
seen, it stays loaded for the rest of the session.

Branch [harness/tool-search](https://github.com/) ports this pattern into Lloyd's
in-process harness.

## What landed

End-to-end behavior when active:

```
TODAY:                                  WITH TOOL SEARCH:
pool.discovered (~72 tools)             pool.discovered (~72 tools)
  ↓ build_tool_list()                     ↓ LoadedToolSet.visible_tools()
tools=[all 72]  ──→ vLLM               tools=[baseline + ToolSearch + loaded] ──→ vLLM
                                         ↑                                ↑
                                         │   model calls ToolSearch       │
                                         │   harness intercepts locally,  │
                                         │   adds matched names ──────────┘
                                         │   to LoadedToolSet
```

The model sees a `<system-reminder>` block listing every tool's advertised name +
one-line description (no schemas). When it needs a tool whose schema isn't loaded,
it calls `ToolSearch(query=...)`. The harness intercepts the call locally — no
MCP round-trip — runs a substring/keyword search over the catalog, marks the
matches as loaded, and returns a `<functions>` block whose format mirrors the
schema dump at the top of Claude Code's prompt. Next iteration's `tools=` array
includes the newly-loaded tools, and vLLM dispatches the model's follow-up call
normally.

## Architecture

### Components

**[app/harness/tool_search.py](../app/harness/tool_search.py)** — pure functions and
the per-session state object.

- `LoadedToolSet` dataclass: holds `(catalog, baseline, loaded, enabled,
  catalog_signature)`. `visible_tools()` returns the filtered OpenAI-shaped
  tool list to send to vLLM each turn.
- `search_tools(query, max_results, catalog)`: resolves three query forms:
    - `select:Name1,Name2` — exact lookup by advertised name
    - `+token rest` — require `token` substring in the name; rank by `rest`
    - `keyword phrase` — tokenized substring scoring (name×3, description×1,
      shorter-name tiebreak)
- `format_catalog_reminder(catalog)`: produces the `<system-reminder>` block
  listing all advertised tool names + first-line descriptions.
- `format_tool_as_function_block(tool)`: produces one
  `<function>{...json...}</function>` line, the same encoding Claude Code uses
  at the top of its prompt and in ToolSearch's response.
- `TOOLSEARCH_OPENAI_TOOL`: the OpenAI-shaped tool definition advertised to
  vLLM so the model can call ToolSearch like any other tool.

**[app/harness/tool_search_cache.py](../app/harness/tool_search_cache.py)** —
process-wide `session_id → LoadedToolSet` map.

- `get_or_create(session_id, catalog, baseline, enabled)`: returns the
  session's `LoadedToolSet`, creating one on first call. Persistence is
  **session-scoped**: subsequent `run_query` calls in the same session reuse
  the same loaded set, so the model only has to discover each tool once per
  conversation.
- Invalidation: each `LoadedToolSet` carries a `catalog_signature` (sha1 of
  sorted name+description pairs). On reuse, if the live catalog signature
  doesn't match the cached one (e.g. config reload added/removed tools), the
  cached entry is replaced and the loaded set drops to empty. This is
  conservative — small description changes invalidate too — and the
  consequence is just one extra round-trip the next time the model needs
  one of the affected tools.
- Blank `session_id` is uncached: each call gets a fresh `LoadedToolSet`.
  Keeps callers without session correlation safe — they just don't get
  cross-call persistence.

**[app/harness/loop.py](../app/harness/loop.py)** — integration into `run_query`.

- `_resolve_loaded_tool_set(options, catalog)`: builds the per-session
  `LoadedToolSet` honoring the activation rule. Activation requires
  `tool_search_enabled` *and* ToolSearch not in `disallowed_tools` *and*
  `len(catalog) >= threshold_tools`. Baseline defaults to
  `BUILTIN_BARE_NAMES` (Bash, Read, Edit, Write, Grep, Glob, Task) intersected
  with the catalog and minus `disallowed_tools`.
- `_inject_catalog_reminder(chat_messages, loaded_set)`: appends the
  `<system-reminder>` block into the leading system message's content,
  guarded by an idempotency marker. **Critical:** must append into the
  existing system message rather than insert a separate one — vLLM rejects
  any second `role: system` message (see "vLLM single-system constraint"
  below).
- ToolSearch dispatch interception in `_dispatch_one_tool_call`: when
  `name == "ToolSearch"` and the loaded set is enabled, run `search_tools`
  locally, mark matched names loaded, and return the `<functions>` block as
  the tool result. The MCP pool is never asked to dispatch ToolSearch.
- Defensive intercept: when the loaded set is enabled and the model calls a
  tool that isn't in `visible_tools()` but *is* in the catalog, return a
  guidance error (`"Tool 'foo' requires schema loading. Call
  ToolSearch(query='select:foo') first."`) instead of dispatching to MCP.
  Catches the case where vLLM's `qwen3_xml` tool parser dispatches an
  unadvertised tool.

**[app/mcp_discovery.py](../app/mcp_discovery.py)** — `_get_tool_search_kwargs()`
helper that resolves `harness.tool_search.*` config into RunOptions kwargs.
Splatted into every `RunOptions(...)` construction site.

**[agent_mcp/builtin_task.py](../agent_mcp/builtin_task.py)** — subagent
isolation. Each Task subagent invocation gets a unique
`session_id=f"task:{subagent_type}:{uuid.hex[:8]}"` so different
`disallowed_tools` profiles (e.g. `read-only` vs `general-purpose`) don't share
a `LoadedToolSet`, and concurrent subagent runs don't bleed state into each
other.

### Configuration

In [config.yaml](../config.yaml):

```yaml
harness:
  tool_search:
    enabled: true             # master switch
    threshold_tools: 30       # activate when catalog has >= this many tools
    baseline_tools: []        # empty → BUILTIN_BARE_NAMES
    max_results_default: 5    # used when model omits max_results
    max_results_cap: 20       # hard cap regardless of model's request
```

The threshold is intentionally low (30) — today's 72-tool catalog activates
immediately. Setting it above the live catalog count (e.g. 100) leaves the
mechanism dormant until the catalog grows.

### Activation rule (effective)

```
active = tool_search_enabled
         AND ToolSearch NOT in disallowed_tools
         AND len(catalog) >= tool_search_threshold_tools
```

When inactive, `LoadedToolSet.visible_tools()` returns the full catalog and
behavior is byte-for-byte identical to the pre-ToolSearch harness (verified by
unit test). This is the regression guard.

## The vLLM single-system constraint

**Discovered the hard way during smoke testing.** The original plan called for
injecting the catalog reminder as a separate `{role: "system"}` message right
after the existing system prompt. This worked fine in unit tests (no live
vLLM). The first live `/api/message` call returned:

```json
{"error":{"message":"System message must be at the beginning.",
          "type":"BadRequestError","code":400}}
```

Both vLLM endpoints in the Lloyd stack — qwen3 at `127.0.0.1:8096` and the
gpt-oss priority-proxy at `127.0.0.1:8097` — enforce the same chat-template
rule: **at most one `role: system` message, and it must be at index 0**. A
second system message *anywhere* (including immediately following the first)
produces the 400 above.

The blast radius was wider than just the user-facing call: every backend path
that calls `run_query` — autonomy ticks, scheduled-task workers,
session-distill, conversation-relation-linking — started 400'ing too, since
they all run through the same loop integration point.

**Fix:** [`_inject_catalog_reminder`](../app/harness/loop.py) appends the
reminder into the existing leading system message's content rather than
inserting a second one. The marker comment `<!--lloyd-toolsearch-catalog-reminder-->`
guards against duplicate appends across iterations. If no system message
exists yet (rare — typically when `system_prompt` is empty and history has no
prior system), exactly one is inserted at index 0 with just the reminder body.

This constraint is now documented in
[memory/project_vllm_single_system_constraint.md](~/.claude/projects/-home-alansrobotlab-lloyd/memory/project_vllm_single_system_constraint.md)
so future system-prompt extensions don't repeat the mistake.

## Files modified

```
app/harness/
├── tool_search.py             # NEW — LoadedToolSet, search_tools, formatters
├── tool_search_cache.py       # NEW — session-scoped cache
├── options.py                 # 5 new tool_search_* RunOptions fields
└── loop.py                    # _resolve_loaded_tool_set, _inject_catalog_reminder,
                               #   ToolSearch dispatch interception, defensive intercept

app/harness/tests/
├── test_tool_search.py        # NEW — 24 unit tests (LoadedToolSet, search_tools,
                               #   formatters, catalog_signature, cache)
└── test_loop_tool_search.py   # NEW —  8 integration tests (full run_query flow)

app/
├── mcp_discovery.py           # _get_tool_search_kwargs() helper
└── routers/
    ├── messages.py            # 3 RunOptions construction sites wired
    └── voice.py               # 1 RunOptions construction site wired

agent_mcp/
└── builtin_task.py            # Per-invocation subagent session_id (uuid suffix)

config.yaml                    # New harness.tool_search block
```

## Edge cases handled

| Case | Behavior |
|------|----------|
| Catalog smaller than `threshold_tools` | Tool search stays inactive — full catalog advertised. |
| `ToolSearch` itself in `disallowed_tools` | Falls back to advertising full catalog with a `WARNING` log. Defeats the purpose, but doesn't lock the model out. |
| Model calls a tool not in `visible_tools()` | Defensive intercept returns guidance to call `ToolSearch(query="select:<name>")` first. MCP is not invoked. |
| Empty/vague query (`""`, `"   "`) | Returns first N tools alphabetically with a hint to be more specific. |
| `select:` with non-existent names | Matched names are returned; missing names listed in a `(not found: ...)` note. |
| Subagents (Task tool) | Each invocation gets a unique `session_id` so loaded sets don't bleed across concurrent subagent runs or across different `disallowed_tools` profiles. |
| Baseline tools in `disallowed_tools` | Filtered out of the baseline at construction — `disabled_tools: [Write]` actually hides Write. |
| Catalog signature change mid-session | Cache invalidates and the loaded set drops to empty. Model re-searches as needed on the next turn. |
| `chat_messages_handle` shared with observer | Catalog reminder is injected into the shared list; idempotency marker prevents duplication on observer re-injection. |

## Testing

### Unit tests — [app/harness/tests/test_tool_search.py](../app/harness/tests/test_tool_search.py)

24 tests covering:
- `LoadedToolSet.visible_tools` enabled/disabled behavior, `is_visible`,
  `mark_loaded` ignores unknown names
- `search_tools` for every query form (select, +token, keyword, empty)
- Match-not-found and missing-name notes
- `max_results` capping
- `format_tool_as_function_block` JSON shape
- `format_catalog_reminder` content + empty-catalog handling
- `catalog_signature` stability + invalidation triggers
- Cache: session reuse preserves `loaded`; signature change invalidates;
  blank session_id is uncached; `drop()` forgets entries; `get_or_create`
  refreshes baseline/enabled on reuse but keeps `loaded`

### Integration tests — [app/harness/tests/test_loop_tool_search.py](../app/harness/tests/test_loop_tool_search.py)

8 tests using a fake `MCPPool` and a scripted `stream_chat` that record what
`tools=` and `messages=` get sent each turn:
- Disabled mode sends full catalog (regression guard)
- Enabled mode first turn sends only baseline + ToolSearch
- Catalog below threshold leaves tool search inactive
- ToolSearch call → matched tools appear in next turn's `tools=`,
  MCP pool is not invoked for the ToolSearch call itself
- Catalog reminder appended to leading system message exactly once
  across all turns; no second `role: system` ever produced
- Defensive intercept on unloaded tool call returns guidance, MCP not invoked
- Session-scoped persistence across consecutive `run_query` calls
- ToolSearch in `disallowed_tools` falls back to full catalog

### Live smoke test (2026-05-03)

Two consecutive turns to the same session against live qwen3 vLLM:

**Turn 1** — fresh session, prompt asks for `vault_search`:
```
12:54:11  loop: ToolSearch query='select:vault_search,vault_recall' max_results=2 matched=0
12:54:12  loop: ToolSearch query='vault search' max_results=3 matched=3 (loaded set now 3)
```
The model first tried bare names (got 0 — tools are advertised as
`mcp__lloyd-mcp__vault_search`), self-recovered with a keyword query, and
completed in 5 turns / 475 output tokens with a correct answer.

**Turn 2** — same session, asks for autonomy notes:
- No ToolSearch call at all. Vault tools persisted from turn 1's loaded set.
- Completed in 4 turns / 494 output tokens with a correct answer.

Session-scoped persistence and end-to-end progressive disclosure both
verified live.

## What we explicitly didn't do

- **No fuzzy / embedding search.** Substring + token scoring is good enough at
  this catalog size. Embeddings would add a dependency, cold-start cost, and
  per-call latency for marginal benefit at ~100 tools. Worth revisiting if the
  catalog grows past several hundred.
- **No automatic baseline tuning.** The baseline is the static
  `BUILTIN_BARE_NAMES` set. We don't track which non-baseline tools the model
  loads frequently and promote them. Could be a future optimization once we
  have telemetry.
- **No estimated-context-pct activation.** The plan considered an alternate
  trigger of "advertised tool schemas exceed N% of context_length", but
  `threshold_tools` alone is simpler and sufficient. Add later if a tool's
  schema starts ballooning beyond expectations.
- **No persistence to disk.** The session-scoped cache is in-memory only.
  Backend restart drops every session's loaded set, and the model re-discovers
  on next turn. Acceptable cost for the simplicity gain.
- **No metric/UI exposure.** ToolSearch activity is logged at INFO level
  (`loop: ToolSearch query=... matched=N (loaded set now N)`) but not exposed
  in the usage stats panel. Add if it becomes useful for debugging.
