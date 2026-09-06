# Lloyd — Claude Code Context

## Project Overview

Lloyd is a fully local AI agent. It runs its own in-process agent loop (`app/harness/`) against a local vLLM server and exposes all tools through a unified MCP aggregator (`agent_mcp/`). The backend is FastAPI + SSE; the frontend is React (Vite).

- **Backend**: `server.py` (FastAPI, port 8080)
- **Frontend**: `web/` (Vite dev server, proxied through backend)
- **Config**: `config.yaml`
- **MCP aggregator**: `agent_mcp/main.py` (unified `Server("lloyd")` on `:8500/mcp`, Streamable HTTP)
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

## Self-modification

Lloyd can change his own code through a gated loop with automatic rollback.
`architecture/self-modification.md` is the long version. The master switch is
`selfmod.enabled` in config.yaml and it defaults to **false**.

```bash
python -m scripts.selfmod.round status              # state + ledger + guardian
python -m scripts.selfmod.round start "goal"        # cuts a worktree
python -m scripts.selfmod.round gate  SM_<id>       # 7 rungs, ~2 min
python -m scripts.selfmod.round land  SM_<id>       # idle-gated, verified
python -m scripts.selfmod.rehearse --yes-i-mean-it  # prove rollback still works
```

Four things worth knowing before touching any of it:

- **The guardian is a systemd unit, not a supervisord program.**
  `agent-supervisord.service` sets `KillMode=control-group`, so a
  supervisord-managed watchdog dies exactly when it is needed. It is stdlib
  only, runs on `/usr/bin/python3`, and never imports from `app/` — it must not
  share a failure domain with what it guards.
- **It runs from a pinned snapshot** at `~/.local/state/lloyd-guardian/bin/`,
  staged only if the candidate compiles and passes its own selftest. Editing
  `agent-services/guardian/` does nothing until a staged copy proves itself, so
  a broken guardian degrades to a *stale* watchdog, never to none.
- **State lives at `~/.local/state/lloyd-selfmod/`**, outside the repo, because
  the guardian must read its rollback target while the repo is being rewritten.
- **`HEAD == last-known-good` never rolls back.** Everything broken with
  nothing promoted is infrastructure, not a bad change.

Errors are read from `logs/server.err`, never `server.log` — `basicConfig`
writes to stderr, so `server.log` is uvicorn's access log and holds zero
error-shaped lines.

### Development happens in ~/lloyd-sandbox

`/home/alansrobotlab/lloyd` is production: a saved file is a deploy. Non-trivial
work belongs in the `~/lloyd-sandbox` clone (remotes: `origin` = GitHub,
`live` = the production tree), pushed as a PR. The autonomous loop is the
exception — it cuts worktrees from live `main` and lands offline, because a PR
step in an auto-landing loop is either ceremony or a contradiction.


## Architecture

```
~/lloyd/
├── server.py            # FastAPI backend — all API endpoints + SSE bridge
├── config.yaml          # Model configs, MCP server list, agent settings
├── prompt_builder.py    # System prompt assembly (SOUL.md + memories + skills)
├── autonomy.py          # Task scheduler
├── usage_store.py       # SQLite usage tracking
│
├── app/host_metrics.py  # CPU/RAM/disk/GPU for the dashboard
├── app/vllm_metrics.py  # vLLM /metrics scrape + rate derivation
├── app/routers/dashboard.py  # GET /api/dashboard (one aggregated snapshot)
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
├── agent_mcp/           # Unified MCP aggregator (Server("lloyd") on :8500/mcp)
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

Events yielded by type (constructors in `app/harness/events.py` are the
authority — these keys are not the OpenAI wire names):
- `system` — `{type, session_id, model}` — turn opened
- `text_delta` — `{type, text}` — streaming text chunk
- `thinking_delta` — `{type, text}` — reasoning content chunk
- `thinking_done` — `{type, text}` — reasoning phase complete
- `tool_call` — `{type, call_id, name, args_json, args_dict}` — tool invocation
- `tool_result` — `{type, call_id, name, content, is_error}` — tool result
- `assistant_message` — `{type, text, tool_calls, thinking, usage,
  duration_ms, iteration, finish_reason}` — one agent-loop iteration.
  `usage` and `duration_ms` are per-iteration, not per-turn.
- `result` — `{type, stop_reason, usage, num_turns, duration_ms,
  response_text}` — turn complete
- `stream_raw` — `{type, raw, error}` — raw SSE line on parse failure

**Mid-turn state (the position-0 rule)**: the system prompt is built once
per turn and inserted at index 0; the loop only ever appends. That keeps the
whole prompt prefix KV-cached across every iteration, so a 160k-token turn
re-prefills nothing. The cost is that anything rendered into the system
prompt — `<active_todos>`, the plan, the goal — is frozen at turn start.
**Never refresh the system prompt mid-turn**; re-anchor by appending instead
(`RunOptions.state_anchor`, mirroring `notification_drain`). A turn that
creates its own todo list would otherwise never see it again — see
`app/routers/messages.py::_build_state_anchor`.

**Preserved thinking**: assistant messages carry their reasoning back into
history under **both** `reasoning` and `reasoning_content`, bounded to
`harness.preserve_thinking_iterations` recent iterations. Qwen3.8-Flash-Next
renders it into each prior turn's `<think>` block; dropping it showed the
model turn after turn in which it had apparently thought nothing. A/B it with
`eval/run_preserve_thinking_eval.py` before changing the window.

The two spellings are not redundant — the engines disagree, and each one
ignores the other's field *silently*:

| Engine | Reads | Ignores |
|---|---|---|
| vLLM 0.28 (primary) | `reasoning` | `reasoning_content` |
| llama.cpp (secondary, Qwen3.6) | `reasoning_content` | `reasoning` |

vLLM accepts both on the wire but only populates the template from
`reasoning` (`entrypoints/chat_utils.py:2000`); Qwen3.6's own
`chat_template.jinja:91` reads `reasoning_content` and never looks at
`reasoning`. Sending one spelling preserves thinking on one engine and
quietly discards it on the other, which is the exact failure this mechanism
exists to prevent. `_prune_reasoning` must drop the pair together or the
token bound stops bounding anything. `tests/test_preserved_thinking.py`
pins both halves; llama.cpp's `POST /apply-template` will show you the
rendered prompt if you need to re-verify.

Scope is **intra-turn only**: history is rebuilt from the session JSON on each
user turn (`load_and_compact_session`), which does not carry per-iteration
reasoning, so the window resets at every turn boundary. That is where the cost
was anyway — the motivating turn ran 52 iterations inside one turn.

**Stream stalls**: `harness.stream_chunk_timeout_seconds` bounds the gap
*between* SSE lines once the engine has started producing, raising
`StreamStalledError`. It deliberately does **not** bound time-to-first-line:
prefill emits no bytes, and the secondary runs llama.cpp with `--parallel 1`,
so a queued request legitimately sits silent for as long as the one ahead of
it. `client.stream_chat` sets httpx `read=None`, so without this a wedged
engine mid-generation hangs the turn until the client gives up. The key
existed from the start and was read by nothing until 2026-09-06.

**Two engines, two reasoning keys**: assistant messages carry reasoning back
as **both** `reasoning` and `reasoning_content`. vLLM populates the template
only from the former; Qwen3.6's own jinja (what llama.cpp applies) reads only
the latter. Sending one breaks preserved thinking on the other engine
silently. See `_assistant_message_for_history`.

**Tool naming**: Built-in tools (Bash, Read, Write, Edit, Grep, Glob, Task) are advertised to vLLM under bare names. This keeps session JSON, SOUL.md deny rules, and Inner Voice `pretooluse_deny` patterns working unchanged.

## Tools

Every tool lives inside an MCP server — built-ins (Bash/Read/Write/Edit/Grep/Glob/Task) live inside the lloyd-mcp aggregator. Tool enable/disable state:

- Server-level: `mcp_servers.<name>.enabled: false`
- Tool-level: `mcp_servers.<name>.disabled_tools: [tool_name, ...]` (use the bare tool name)

config.yaml holds the hand-edited defaults and is **read-only at boot**; UI toggles (`/api/tool-toggle`, `/api/tool-discovery`) persist to `data/tool_overrides.yaml` (gitignored), which is merged over config.yaml at load (`app/config.py:_merge_tool_overrides`). To change tool state by hand, edit config.yaml and check `data/tool_overrides.yaml` isn't shadowing the same key.

Disabled tools are enforced via `RunOptions.disallowed_tools` as `mcp__<server>__<tool>`. The harness's bare-name aliasing in `tool_schema.py` blocks both the bare and namespaced form at advertise + dispatch time, so disabling `Bash` via `mcp_servers.lloyd-mcp.disabled_tools: [Bash]` blocks the model from calling either `Bash` or `mcp__lloyd-mcp__Bash`.

## Mission Control dashboard

The `dashboard` tab (first in the sidebar, desktop landing tab) polls one
aggregated endpoint, `GET /api/dashboard`, every 2s. It is deliberately a
single endpoint rather than one per panel: the page is open all day, and
eight requests per tick times however many tabs are open is real load on
a box whose job is holding a 262k-token KV cache steady.

Sections are gathered concurrently and **degrade independently** — a
wedged supervisord turns one panel into an error string and leaves the
rest live. A dashboard is most useful when something is broken, so it
must not be the second thing to break.

Where each section comes from:

| Section | Source |
|---|---|
| `host` | `app/host_metrics.py` — psutil + `nvidia-smi` (2s cache) |
| `vllm` | `app/vllm_metrics.py` — scrapes `<base_url>/metrics` per configured model |
| `primary` | `sessions_io.active_sessions_snapshot()` |
| `focus` | goal / plan / todos out of the session JSON |
| `agents` | **the lloyd-mcp process**, over loopback — see below |
| `services` | `app/supervisor_client.py` |
| `workers` | `workers.queue` + `workers.pool` — pool slots, per-source depth, recent runs |
| `autonomy` | `~/obsidian/autonomy/*.md` frontmatter + the pool's in-flight `scheduled-task` jobs |
| `backlog` | `~/obsidian/backlog/*.md` frontmatter |
| `usage` | `usage_store` |

Sections that walk the vault (`autonomy`, `backlog`) are TTL-cached for
10s — the backlog is 300+ markdown files and its status counts do not
change between 2-second polls. Live sections are never cached; they are
the point of the page.

**Overdue is not "next up."** `_autonomy` splits scheduled tasks on
`next_run` vs now and returns them as separate lists. Sorting them
together ascending and labelling the head "next up" is how a fleet whose
ticker is months behind renders as a healthy schedule — the most overdue
task lands exactly where the soonest one belongs. Likewise `completed`
is excluded from worker "open" counts (`_OPEN_STATES`): it dominates the
depth table and would bury the handful of items actually waiting.

**Subagents and background bash tasks live in the lloyd-mcp process, not
the backend.** The aggregator owns the `Task` tool and spawns
`Bash(run_in_background=true)` children, so the backend has no handle on
either. `agent_mcp/main.py` exposes `GET :8500/state` beside `/health`
and `app/routers/dashboard.py` reads it over loopback. Adding a new
agent-side live panel means extending that route, not the backend.

`agent_mcp/_subagent_registry.py` opens a row **before** the Task run
loop starts — a `Task` blocks its caller for minutes, so a row created on
completion would only ever describe runs that no longer need watching.
Closing it is the subtle part: `finish` is idempotent and
first-writer-wins, so a blanket `finally: finish("cancelled")` runs
*before* the success path and silently stamps every completed run
cancelled. Each exit path closes the row with its own real status;
`tests/test_task_registry_wiring.py` pins that.

**Not every engine is vLLM.** The secondary slot (:8091) runs llama-server,
because a GGUF Q3 is the only build of Qwen3.6-35B-A3B that fits a 24 GB
3090 at the full 262144 window — unsloth's NVFP4 needs SM100+ and vLLM's
GGUF path does not cover this hybrid linear-attention MoE. It serves the
same OpenAI API, so the harness is unchanged, but it publishes `llamacpp:`
Prometheus names. `vllm_metrics._translate_llamacpp` renames them into the
vLLM vocabulary so one snapshot path and one dashboard card serve both.
Two things genuinely do not exist there and are reported as `None` rather
than `0`: live KV occupancy (no gauge) and TTFT (no per-request count).
A llama.cpp engine is `awake` whenever it is reachable — it has no
sleep-state gauge, and falling through to the vLLM check renders a healthy
engine "asleep". Its model name comes from a `/props` probe cached per
engine lifetime, since llama.cpp does not label its metrics.

**Counters vs. gauges.** vLLM exposes both. Gauges (`num_requests_running`,
`kv_cache_usage_perc`) are read straight. Counters
(`prompt_tokens_total`, `prefix_cache_hits_total`) are monotonic since
engine boot, and their absolute value says nothing useful, so
`vllm_metrics` keeps the previous scrape per engine and reports a rate.
A counter that goes backwards (engine restarted) yields `None`, never a
number — otherwise a restart renders as a one-second spike of the
engine's entire history. An unreachable engine drops its baseline for the
same reason.

## Model slots

`models.<alias>` in config.yaml is only the *endpoint*. Which model actually
answers there is decided by the supervisord program's `environment=MODEL=...`
and its start script — three places that can drift apart, and did on
2026-09-06 when a selfmod rollback reverted `agent-llm-secondary.conf` to a
launcher branch serving a 4B under the same alias and port as the 35B.

`models.<alias>.expect_model` is a case-insensitive substring checked against
what the engine reports (vLLM `/v1/models`.root, llama.cpp `/props`.model_path).
`app/model_identity.py` sweeps it at boot — detached, with retries, because a
cold 35B takes minutes to load — and logs ERROR on a mismatch.
`GET /api/models/identity?refresh=1` re-probes on demand. A slot with no
`expect_model` reports `unchecked`, so **update it whenever you swap a slot's
occupant** or the check is inert.

Current occupants: primary `:8096` = Qwen3.8-Flash-Next (vLLM, GPU 1);
secondary `:8091` = Qwen3.6-35B-A3B UD-Q3_K_XL (llama.cpp, GPU 2, single-tenant
at ~21.7 of 24 GiB). The secondary serialises (`--parallel 1`) because
llama.cpp divides `--ctx-size` across slots and the full 256K window was the
point — `secondary_models.py` post-session jobs and voice summaries queue
behind agent turns there.

**Subagents inherit the calling turn's model.** `subagents.<type>.model: ''`
means "whatever spawned me"; the harness ships it in the MCP request `_meta`
(`lloyd/model`, `lloyd/base_url`) since Task runs in the aggregator process
and has no other way to know. Pin an alias there to override. Empty
`base_url` resolves from `models:` for the chosen model — *not* from
`default_model_base_url()`, which always returns the primary's endpoint.

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
    expect_model: Qwen3.8-Flash-Next   # identity check; see below
    env:
      ANTHROPIC_BASE_URL: "http://127.0.0.1:8096"
      ANTHROPIC_API_KEY: "no-key-required"
      ANTHROPIC_CUSTOM_MODEL_OPTION: "primary"
      ANTHROPIC_CUSTOM_MODEL_OPTION_NAME: "Primary"

harness:
  stream_chunk_timeout_seconds: 60      # gap BETWEEN SSE lines, not TTFB
  todo_anchor_interval_iterations: 10   # re-append session.todos this often
  preserve_thinking_iterations: 6       # carry N iterations' reasoning back
  tool_search:            # progressive disclosure; baseline + ToolSearch
    enabled: true
    threshold_tools: 30
    baseline_tools: [Bash, Read, Edit, http_search, http_fetch, ...]

subagents:
  general-purpose:
    system_prompt: ""
    max_turns: 40
    disallowed_tools: []
    model: ''        # '' = inherit the calling turn's model
    base_url: ''     # '' = resolve from `models:` for the chosen model

mcp_servers:
  lloyd-mcp:
    type: streamable-http
    url: http://127.0.0.1:8500/mcp
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
