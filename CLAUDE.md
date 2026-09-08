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
python -m scripts.selfmod.round gate  SM_<id>       # 8 rungs, ~2.5 min
python -m scripts.selfmod.round land  SM_<id>       # idle-gated, verified
python -m scripts.selfmod.round bless               # HEAD becomes last-known-good
python -m scripts.selfmod.round recover             # clear BROKEN, restart the stack
python -m scripts.selfmod.rehearse --yes-i-mean-it  # prove rollback still works
```

**Nothing inside the blast radius performs a rollback, and a landing runs
detached.** A rollback stops the backend and the aggregator, so code doing one
from inside either process issues the stop that kills its own caller and never
reaches `git reset` — the stack goes down and the tree does not move. They
write `rollback_request.json` and the guardian performs it. Likewise the
promoter restarts `lloyd-mcp`, which is where the MCP tool calling it lives,
so `selfmod_land` spawns it in a new session (`state.spawn_detached`) — a
process group signal cannot reach that. Before this, the CLI path worked only
because the Bash tool already spawns that way, so the loop worked when a human
drove it and would have failed the first time Lloyd did.

Things worth knowing before touching any of it:

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
- **A rollback reverts in place when HEAD has moved past the promotion.**
  `reset --hard` to the parent is right only while HEAD *is* the promotion.
  Nightly jobs commit straight to live `main`, so a 15-minute window can close
  over work the loop never touched, and resetting past it destroys commits
  nobody asked the guardian to judge. The route is chosen per rollback; a
  conflicting revert escalates rather than guessing.
- **A rollback restores the promotion's own `rollback_target`, not the LKG.**
  `gstate.rollback_target(current)` prefers what `current.json` recorded at
  landing time, then its `parent`, and only then the LKG pointer. The LKG is
  a blunt fallback because it advances *only when a promotion settles*: two
  rollbacks in a row strand it wherever it last settled while HEAD keeps
  moving with ordinary human commits. On 2026-09-06 that turned one false
  positive into **26 discarded commits** — LKG had sat at 14:24 all day, so
  reverting a promotion whose parent was six hours newer took the whole
  evening with it. The promoter had written the correct target and nothing
  read it. The failure is self-reinforcing, which is what makes it worth a
  rule: every rollback that does not settle makes the next one wider.
- **The log cursor advances on every tick, not only while observing.**
  `Guardian.drain_logs()` runs at the top of `tick()`, above every early
  return, and `evaluate_errors` reads the buffer it fills. Reading used to
  live inside `evaluate_errors`, which only runs during an observation
  window — so between rounds the cursor stood still and the first tick of a
  new window read *everything since the last one*. That is what fired on
  2026-09-06: a healthy promotion reverted four seconds after landing, on
  nine `ConnectError` lines from 11:47–11:56 that morning. Errors are still
  *judged* only inside a window; what changed is that the tape always moves.
  The paused path additionally **discards** its buffer rather than skipping
  it, because the promoter holds that pause across its own supervisord
  restart and the window for that very deploy opens seconds later.

Errors are read from `logs/server.err`, never `server.log` — `basicConfig`
writes to stderr, so `server.log` is uvicorn's access log and holds zero
error-shaped lines.

- **A turn that dies at its budget is not the end of the round.** The
  budget anchor (`<budget>` at 75%/90% of `max_turns`) tells the model to
  gate-and-land or abort while it still can; the observer's ambient
  follow-up is the first responder when it did not (that is what landed
  #278); `backlog_implement.reap_abandoned_rounds` is the backstop, twenty
  minutes later, branch kept. Never abort a round at turn end.
- **The idle gate drains first, then waits.** `wait_idle` used to arm the
  drain only *after* three quiet polls — the one moment it is no longer
  needed. Against a worker pool that starts a research job every few minutes
  that is a lottery, and the first landing of the unattended era
  (SM_20260907_233449) lost it: 900 s watching `harness_runs` flicker, never
  drained. Now the drain is armed before the first poll, refreshed inside its
  TTL, and released on give-up.
- **The idle gate counts `harness_runs`, not just session queues.** Worker
  jobs call `run_query` directly and never enter a queue, so a ten-minute
  research job was invisible to the gate that exists to avoid killing it. The
  landing then restarted the backend underneath it, and the `ConnectError`
  lines it logged on the way down landed inside the observation window — so
  the promotion was reverted for damage its own landing caused. That is the
  2026-09-06 20:14 rollback exactly.
- **An answered non-200 is not a refused connection.** Three probe classes,
  three budgets: refused 3 ticks, timeout 24, http-error 36. The aggregator's
  503 (any degraded module, e.g. a closed Thunderbird) is excluded from the
  down predicate entirely and judged by `mcp_degraded_is_fatal` instead.
- **Every rollback this loop has performed has been a false positive.** The
  failure mode to design against is inventing a bad build, not missing one.

### Alerts: one fan-out, six channels

`agent-services/guardian/notify.py` is the **only** producer of user-facing
alerts. That is a recent property, not an accident: `lloyd-guardian-nag.service`
used to run its own inline `notify-send`, a second private definition of "tell
the human" that was structurally incapable of gaining any channel this module
grew. It now calls `nag.py`, which goes through the same `Notifier`. Anything
that wants to announce something goes through here.

Two entry points, and picking the wrong one is the trap:

- **`alert()`** — an incident. Fans out to all six: ledger, ALERT.md, journal,
  desktop toast, voice, vault note, plus a backlog task when critical.
- **`announce()`** — news, with no bookkeeping. Journal, toast and voice only.
  Used by a successful promotion and by the 15-minute nag. The nag is the
  reason `announce` takes a `level`: the state really is critical and should
  look it, but it has **already** been recorded, and re-announcing through
  `alert` would append a ledger row and file a fresh backlog task every 15
  minutes, burying the task the rollback filed under copies of itself.

Promotion *success* is announced too. Before, every notify-send in the tree
hung off a guardian alert, so a loop that rewrites the running system in the
background was silent whenever it worked — backwards, since the successful
landings are the ones nobody is watching a terminal for.

### The spoken channel

`speak.py` says the alert aloud in the same cloned voice as voice mode. It
lives in `agent-services/guardian/` because `guardian-stage.sh` stages
`guardian/*.py` and nothing else — a module the guardian imports must be in
that directory or it will not exist in the pinned snapshot.

- **It reports *dispatched*, not *heard*.** The unit watchdogs the loop at
  `WatchdogSec=90` against a 5s tick, so synthesis and playback happen in a
  detached child and `alert()` returns in milliseconds. What actually came out
  of the speaker is in `voice.log` in the guardian state dir.
- **The child runs the venv python, and that is deliberate.** The guardian's
  stdlib-only rule exists so the watchdog cannot be taken down by what it
  watches. This child runs *after* all five reliable channels have fired,
  nothing waits on its exit, and its failure cannot reach the loop — so
  spending the venv buys the presence EQ (scipy) at no cost to the property
  the rule protects. A wrecked venv costs a duller voice, never an alert.
- **Shaping degrades in tiers and says which one ran.** EQ needs scipy, the
  WSOLA speed needs only numpy, so a system-python fallback still fixes the
  pace. The tier is logged because the first cut of this module called
  `OutputShaper.enabled()` — it is a `@property` — and shipped *unshaped*
  audio while looking perfectly healthy. A silent downgrade is
  indistinguishable from success.
- **Suppression is on disk**, keyed by alert title, because the two producers
  are different processes: the daemon and the nag oneshot. In-memory dedupe
  (`guardian.py::_alert_seen`, `ALERT_REPEAT_SECONDS` = 900s) cannot see the
  nag. `policy.VOICE_REPEAT_SECONDS` is 3600s — a toast you have already seen
  costs a glance, a sentence you have already heard costs the whole sentence,
  and at 900s an unresolved incident would say the same thing aloud four times
  an hour indefinitely.
- **Quiet hours gate the clock, and only the sound.** `guardian.voice.quiet_hours`
  in config.yaml (23–07 by default) withholds speech; the toast, journal,
  ledger, vault note and backlog task all still fire, so nothing is lost — it
  is waiting in the morning. That is what makes it safe to default on.
  `allow_critical: true` lets a rollback wake you anyway. The window is
  checked **before** `should_speak`, which records as it decides: recording a
  quiet-hours drop would spend the hourly slot on an utterance nobody heard,
  and the 08:00 repeat of an 03:00 alert would then stay silent for the wrong
  reason. A window that wraps midnight is the normal shape, and `start == end`
  means *no* window rather than a full day of silence.
  It lives under `guardian.voice`, **not** `livekit.tts`, and the split is
  load-bearing: `livekit_worker` reads `livekit.tts`, and a voice conversation
  that went mute at 23:00 because an alert policy leaked into it would be a
  real bug. Only alerts are gated by the hour.
- **`LLOYD_VOICE_ALERTS=0`** keeps every other channel and drops only speech.
  `tests/conftest.py` sets it for every test — otherwise `pytest tests/` talks
  to the room from a process that outlives the test.
- **Voice sits below the `external` gate**, like the vault note and the
  backlog task: the drill runs a real guardian against a throwaway repo, and a
  rehearsal that announces a rollback out loud is indistinguishable from a
  production incident to anyone in the room.

`config.yaml`'s `livekit.tts` stays the single source for the voice.
`agent-services/bin/sync-voice-config.py` pushes it into the guardian's state
dir at stage time (the guardian has no yaml, and must not read the repo on a
critical path). If it never runs, `speak.py`'s built-in defaults still sound
right — the sync only stops the two drifting after a voice *change*.

### Unattended: triage, then implement

`backlog-selfmod` triages; `backlog-implement` runs one round per `confirmed`
item with an acceptance check, behind every gate the loop enforces. Both are
off by default and both run **in a real session** via `run_prompt_in_session`,
which is the only way a worker turn gets Inner Voice and a transcript — never
`run_prompt_on_primary` for anything that judges or changes this code. Budget
exhaustion records `incomplete`, not a verdict; the item comes back once.

Two things the first unattended run (#229) taught, both pinned by tests:
a worker session is **never the user's session** — worker turns arrive
through the chat path, so "the last session to receive a user turn" was one
nobody reads, and the morning brief was delivered there.
`sessions_io.NON_USER_PLATFORMS` (`autonomy`, `worker`) is the one definition
of that, and `/inject` refuses such a session with 409 so no producer counts
it delivered. And a verdict's `ACCEPTANCE` is a contract only if
`backlog.acceptance_text` says so: the old template's `else: ->` was copied
verbatim, and `strip("-")` left a truthy `>`. And **a finding that lives only
in EVIDENCE is lost**: #229 said two claims "belong in two new items" and filed
none, in a turn told to write nothing. Both prompts now require filing whatever
the in-focus item does not cover, via `backlog_write_task`, and report it under
`SPAWNED:`; ids are verified on disk before the ledger links them.

**But the pass may not eat what it files.** That filing requirement met
`select_candidate`, which takes the oldest untriaged *open* item, and
`OPEN_STATUSES` includes `draft` — the status `backlog_write_task` writes. So
every item triage filed re-entered the queue it came out of. Over the loop's
first 48 hours: 40 triage runs closed 28 items and filed 78, a reproduction
number of **1.95**, or +46 open items a day at the then-cadence of one run per
30 minutes. The open board went **19 → 122**, and 110 of the 122 were the
loop's own output. R > 1 is the entire bug: the queue doubles rather than
drains, however good the verdicts are. Oldest-first ordering hid it, because
self-filed items sort to the back and the pass reads as healthy right up to
the moment the real backlog runs out — which was 6 items away when this
landed. `backlog.is_quarantined` holds a self-filed item (tagged
`spawned-by-triage` or `spawned-by-selfmod`) out of the candidate pool until
it is `SPAWN_TRIAGE_MIN_AGE_DAYS` old. Quarantine, not exclusion: an item
nobody implements really can go stale, and then the question is real again.
The gate keys on those tags and **not** on `draft`, which is the status of
most of a stale backlog — a rule that skipped drafts would switch the pass
off rather than bound it.

An exhausted queue therefore has two meanings, and `triage_pool` returns the
held count so the skip summary can say which one it is. "Every open backlog
item has been triaged" was true, and misleading, on a board of 122 where 106
were this loop's own drafts.

`SPAWN_CAP` (3, both sources) bounds fan-out per run; overflow goes into one
"Further findings from…" item rather than being dropped, since #229's lesson
still holds. It is **recorded, not enforced** — the items exist on disk before
`SPAWNED:` is parsed, so unfiling them would destroy real findings — and the
ledger carries `spawn_cap`/`spawned_over_cap` on both event types.
`tests/test_backlog_spawn_loop.py` pins all of it, including the
counterfactual: with the window set to zero the same run grows the queue.

The verdict's `SURFACE:` picks the implementer's route. `code` and `frontend`
run a worktree round through the gate — `web/src/**` is in scope since the
`frontend` rung (tsc delta + `vite build`) exists. `vault` runs
`scripts/selfmod/vault_round.py` (`selfmod_vault_land`): the vault is a live,
shared tree with no worktree, so the route is validate the named paths (front
matter; the real prompt/skill/task loaders for `skills/**`, `lloyd/**`,
`autonomy/**`), commit exactly those paths on `main`, revert on failure. A
`confirmed` whose fix needs a path the loop may never touch begins its
acceptance with `human-only:` and is skipped, not attempted.
`architecture/self-modification.md` §3.2.

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
├── app/session_titles.py # few-word session names (secondary model)
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
- `thinking_done` — `{type, text, duration_ms}` — reasoning phase
  complete. `duration_ms` spans the first reasoning chunk to the last,
  not the iteration's wall clock, which also covers prefill and the
  answer written afterwards. It reaches the chat's collapsed thinking
  panel as `reasoning_ms` on the persisted assistant message, so the
  header reads the same on reload as it did live — the event lands
  *after* that iteration's text, so the browser measures the delta
  timestamps itself until the real number arrives.
- `tool_call` — `{type, call_id, name, args_json, args_dict, summary}` — tool
  invocation. `summary` is the model's own one-liner for the transcript;
  it is absent from `args_json`/`args_dict` (see "Tool-call summaries").
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

**An empty tool pool is the worst failure in the system.** `client.stream_chat`
omits `tools` from the request when the list is falsy, so vLLM never engages
the `qwen3_xml` tool parser. The model still reads its whole toolbox in the
system prompt, reasons its way to "call Bash", and then has no channel to emit
a tool call on. What comes out is an empty message, or the call written as
prose (`{"name":"Bash","input":...}` — an Anthropic shape that appears nowhere
in this repo), or invented tool *output*. Nothing in the stream says "no
tools"; it reads exactly like the model having forgotten how to use them, and
Inner Voice's only lever — injecting more text — cannot help, because the
intent was never missing, the capability was.

`MCPPool.open()` used to log a warning, `continue`, and set `_opened = True`
even when its only server failed discovery. `get_or_open_pool` caches
process-wide and short-circuits on `_opened`, so one transient error (the
aggregator restarting) pinned an empty pool for the life of the backend.
`open()` now raises `ToolDiscoveryError` when discovery yields nothing, which
makes `get_or_open_pool`'s **existing** eviction path fire so the next caller
re-discovers — the recovery already existed, nothing ever failed loudly enough
to trigger it. A *partial* failure still degrades gracefully; that is what the
`continue` is for. `run_query` refuses a turn whose pool advertised nothing.

Two things this cost on 2026-09-06, both invisible as tool failures:
a 30-minute chat where Lloyd narrated `sqlite3` commands instead of running
them, and four `domain-research` jobs killed at the 600s cap — a toolless
research job cannot research, so it spins until the timer. The same job
finished in 17s once tools came back. `tests/test_mcp_pool_discovery_failure.py`
pins it. The tell in the log is `no server claims tool '_BackgroundTaskDrain'`
firing right after `mcp_pool: failed to discover` — the drain shares the pool,
so it is the cheapest early warning that every turn has gone toolless.

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

**The override file must stay untracked, and that is the whole reason it
exists.** `save_tool_overrides` replaced dumping the entire CONFIG back over
config.yaml on every toggle, because config.yaml is tracked and a tracked
file rewritten by a UI click leaves the live tree dirty — which
`scripts/selfmod/gate.py` and `promote.py` both refuse. Until 2026-09-07 the
override file was tracked too, so the escape hatch had the defect it was
built to avoid: one click on the Tools page dirtied the tree and silently
stopped the self-modification loop until someone hand-committed the result
(`4fb1ccd`, `2ef86c7` are that happening). Untracked alone is not enough —
`git status --porcelain` lists new files as well — so it needs the
`.gitignore` rule beside it. `tests/test_tool_overrides.py` pins both halves.

**A test about untracked state may not require that state to be present.**
The end-to-end test asserted `(ROOT / "data/tool_overrides.yaml").exists()`
with `ROOT` resolved from `__file__` — so in a selfmod worktree it demanded a
file that is, by design, never checked out. `data/` has no tracked contents at
all, so it failed for **every round from `d11ad8c` onward, whatever the diff
under test**, and the `tests` rung is a hard rung: three rounds aborted on it
in fifteen hours while filing fourteen new backlog items and landing nothing.
The loop's only drain was blocked by a test asserting that a deliberately
untracked file had been checked out. It now *writes* the file through the real
writer — `app.paths.LLOYD_HOME` resolves from `__file__` too, so the writer and
the test address the same tree in a worktree — and asserts git never sees it,
which is also the stronger check: the old form could only observe a file
somebody else had already written and reverted. The `.<pid>.tmp` sibling
`atomic_write_text` lands before renaming is now ignored and asserted too; a
write killed in between leaves a stray that dirties the tree exactly as the
tracked file used to, one filename over.

**And a gate rung must not depend on the wall clock.** Two
`tests/test_guardian_speak.py` tests asserted that an alert dispatches voice,
while `speak.dispatch` consults `in_quiet_hours` against the real clock and a
23→07 default window — so they failed whenever the gate ran overnight, which
is when the unattended loop runs. Rounds gated at 23:52 and 04:07 failed;
the same code at 08:13 did not. They pin the policy off explicitly now
(`_AWAKE`), as the tests that are *about* quiet hours already pinned it on.

Because a fresh clone has no override file, **config.yaml is the state a
rebuild boots into**, so it has to keep describing what is actually served.
It claimed `tool_search.enabled: true` for an unknown stretch while the
override served `false`. The merge now warns on a key whose override differs
from the tracked value — the `disabled_tools` half has warned since the
2026-09-04 `browser_screenshot` incident, but `harness.tool_search` was a
bare `.update()`, and `enabled` decides whether the model is handed all 131
tools or a baseline plus ToolSearch. Agreement stays silent: the Tools page
rewrites the whole block on every toggle, so warning on it would fire each
boot and stop meaning anything.

Disabled tools are enforced via `RunOptions.disallowed_tools` as `mcp__<server>__<tool>`. The harness's bare-name aliasing in `tool_schema.py` blocks both the bare and namespaced form at advertise + dispatch time, so disabling `Bash` via `mcp_servers.lloyd-mcp.disabled_tools: [Bash]` blocks the model from calling either `Bash` or `mcp__lloyd-mcp__Bash`.

### Tool-call summaries

Every advertised tool carries one extra string parameter, `summary`: a
short phrase the model writes saying what the call is doing ("Reading
server.py", "Restarting the backend"). The collapsed tool bubble in the
chat and Inner Voice transcripts renders it as **`ToolName`** — summary,
which is the whole point — a wall of `Bash`, `Bash`, `Read`, `Bash` says
nothing about what a 50-iteration turn actually did.

It is display metadata riding in the one channel a tool call has —
its arguments — which makes *where it is removed* the whole design:

- **`tool_schema.add_summary_param`** injects it and returns *which tools
  got it*. That return value is load-bearing: `session_inject_context`
  already has a required top-level `summary` of its own (1 of the 129
  tools advertised today), and popping that one before dispatch would
  delete a real argument. Injection **replaces** the `parameters` object
  rather than mutating it — it arrives as the very `inputSchema` dict
  held in `MCPPool.discovered`, which is process-shared for the life of
  the pool, so an in-place write would make the *next* turn read `summary`
  back as the tool's own parameter, skip injection, and stop stripping.
- **`loop._commit_tool_calls`** lifts the value onto the tool call's
  `_summary` and pops it from `_args_dict` — and *only* from there. The
  two records of a call deliberately disagree: `_args_dict` is what
  reaches MCP, which validates against each tool's real inputSchema (a
  leaked `summary` is a dispatch error, not a spare field), while
  `arguments` is what gets replayed to the engine and is **the only
  record of this call the model will ever see again**.
  **Stripping the caption from `arguments` too is what broke the first
  cut of this**, and it broke it invisibly: session
  `20260907_184351_ivec8d` shows the first call of each tool name
  carrying a summary and every repeat carrying none — 5/5 vs 0/31. The
  schema said `required`; the model's own most recent example of that
  tool said otherwise, and the example won. A few-shot channel you are
  writing into cannot be edited for brevity. It costs ~10 tokens per
  historical tool call to keep, which is the price of the field working
  past its first use.
- **`summary` is injected first** in `properties` and in `required`.
  Property order is the order the schema is shown to the model and
  roughly the order it emits arguments in, so a caption placed after
  Bash's `command` is one written after a 40-line heredoc.
- `messages.py` persists it on the tool call (omitted when empty, so
  every pre-existing session reads the same), puts it on the
  `tool_start` SSE frame so the live bubble has it before the result
  lands, and prefers it over `tool_activity_detail` for the dashboard's
  live activity line.
- The transcript's expanded **Arguments** block hides the key, because
  that block shows what was *dispatched* and the header already shows
  the caption. It is dropped only when the header is rendering it, so a
  tool with a real `summary` parameter of its own still shows it there.

The Inner Voice observer reads the caption in two of its three tool-call
inputs, and the third exclusion is deliberate:

- **`build_assistant_message_summary`** renders
  `Bash — Checking root disk usage` per call instead of
  `['Bash','Bash','Bash']`. The observer's job is judging whether the
  primary is still on the user's request, and a wall of identical names
  is the least informative possible input for that.
  `observer_prompt._tool_call_labels` accepts all three shapes a caption
  arrives in — `_summary` (live harness event), `summary` (rebuilt from
  session JSON), and `summary` inside the raw `arguments` string.
- **`build_pretool_event_summary`** states it before the arguments:
  the caption is what the primary *said* it was doing and the arguments
  are what it actually did, so the two disagreeing is the signal. (This
  path is dormant while `pretool_llm_enabled: false`.)
- **`guards.tool_call_signature` must never see it.** `exact` is the
  full `key=value` rendering for every tool but Bash, so a caption in
  the args makes two byte-identical calls compare as different — and
  rewording is exactly what a looping model does. This is why
  `fire_pre_tool_use` carries the caption as its own `tool_summary` key
  rather than merging it into `tool_input`: `tool_input` is what safety
  matching and the repetition guard read, and it stays clean.
  `tests/test_tool_call_summaries.py` pins all three.

**No tool may ask for the caption twice.** `Bash` used to declare a
`description` argument — "Short human-readable description (informational
only)" — and `Task` a `description`, "Short label for the task
(informational)". Both restated, one key later in the same object, exactly
what the injected `summary` asks for, and a model answers that question
once. On 2026-09-07 it began answering into the wrong half: sessions
`20260907_235236_backlogs_a8fd` and `20260908_000804_backlogi_3828` emitted
`{"command": ..., "description": "check"}` for 49 consecutive Bash calls
with no `summary` on any of them, and the chat rendered 49 bare `Bash`
rows. **Nothing errored**, because `description` was a real Bash argument —
the caption was not dropped, it was filed where only a background task
would read it.

What makes an ambiguous schema expensive here is the ratchet described
above: `arguments` is replayed as history, so the first miss becomes the
model's own most recent example of calling that tool and the session locks
into it. Across the 16 sessions since the feature landed, every one whose
*first* Bash call carried a summary stayed above 95%; both that missed
stayed below 26%. One field decides a whole session, which is why the fix
is to delete the competing field rather than to reword it.

The two tools still need their label — a background-task row and a subagent
row are both read by a human later — so the caption now travels the way the
session id and the calling turn's model already do: in the request's
`_meta`, as `lloyd/summary`, lifted into `_task_registry.current_call_summary`
by `agent_mcp/main.py::call_tool`. It must not be handed back through `args`
instead: that is what the aggregator validates against each tool's real
inputSchema, and it is what the repetition guard hashes.
`tests/test_tool_call_summaries.py` pins that Bash and Task advertise no
second caption field, and that the caption reaches MCP through `_meta` only.

`harness.tool_call_summaries: false` removes the parameter from every
schema; the UI falls back to the bare tool name. Worth reaching for if a
model ever starts spending its tool-call budget on the caption.

## Code graph

`agent_mcp/code_graph.py` answers "who calls this" and "what breaks if I
change it" from graphify's deterministic AST extraction of a tree
(`<root>/graphify-out/graph.json`, ~15 s to build, zero LLM calls). Six
tools: `graph_explain`, `graph_affected`, `graph_path`, `graph_hubs`,
`graph_status`, `graph_refresh`.

- **It is not a second MCP server, deliberately.** graphify ships
  `graphify-mcp` and mounting it would have been one config line, but Lloyd
  advertises every server's tools under bare names and `build_tool_list`
  raises on a cross-server collision; Task subagents pin
  `DEFAULT_LLOYD_MCP_SERVERS` and would never see it;
  `tests/test_mcp_layer.py` needs every configured server discoverable at
  test time, including inside a worktree where no second daemon is running;
  `agent-services/supervisor/**` is a protected selfmod path, so Lloyd could
  never repair the program running it; and graphify-mcp has no `affected`,
  which is the one query a change actually needs.
- **`root` is explicit and never inferred.** Nothing on disk links a chat
  session to an open round — `round_start` ledger rows carry no session id —
  so a "bound session's worktree" default would silently answer about the
  wrong checkout. Defaults to `LLOYD_HOME`; accepts an `SM_…` round id or an
  absolute path. When the answer is about the live tree and a worktree is
  open, the header says so.
- **Staleness is commit mismatch OR an uncommitted source file newer than
  `graph.json`.** The second rule is mandatory: inside a round HEAD does not
  move while the model edits, so a commit-only rule calls the graph fresh
  for exactly the window it is most wrong in. Auto-rebuilds from the dirty
  rule are debounced by `min_refresh_interval_s` (30 s); a debounced query
  still answers and says `STALE`.
- **`graphify-out/` must stay gitignored, unanchored.** A build inside a
  round dirties the tree, and both `scripts/selfmod/gate.py` and
  `promote.py` refuse a dirty tree — so an unignored build would abort the
  round on its own map. `*.json` at the top of `.gitignore` hid `graph.json`
  by accident; `GRAPH_REPORT.md`, `graph.html`, `.graphify_root` and the
  ~64k-file `cache/ast/**` were covered by nothing.
- **The graph is blind across process seams.** It is an AST extraction of
  one tree: there is no edge from `run_prompt_in_session` to `run_query`,
  because that call crosses HTTP, and none from `run_query` into a tool
  handler, because that crosses MCP. Keep Grep for string keys, route paths
  and config names.
- Ambiguity is an answer, not an error: `main` matches 88 nodes here, and a
  listing with ids lets the model pick where an error makes it guess again.
- There is no `enabled` flag. The kill switch is
  `mcp_servers.lloyd-mcp.disabled_tools`; an `enabled: false` that emptied
  `list_tools()` would break the annotation-staleness test.

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
| `primary` | `sessions_io.active_sessions_snapshot()` + `session_titles` |
| `recent` | the last chats to stop talking — bounded scan of `sessions/` |
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

`recent` is the third cached section and the one with a trap. A session
JSON carries its whole transcript (100 files, 7.5 MB today), so the scan
is bounded twice: only the newest `_RECENT_CANDIDATES` files by **mtime**
are opened, and the parse is cached for 10s. The mtime window is safe
only because mtime is never *earlier* than `last_active` — background
writers (the titler, post-session capture, TodoWrite) push a file's mtime
later than its last real message, so mtime can promote a stale chat but
never demote a fresh one out of the window. The rows are then sorted on
`last_active`, which is what `GET /api/sessions` sorts on too.

The live filter — dropping sessions with a running or queued turn, which
the panel beside it already shows — is applied **outside** that cache.
Caching it would leave a chat that just started reading as finished for
up to ten seconds. Cache the expensive scan, never the cheap freshness.

A section can also be *missing*, not just failed: a browser tab left open
across a backend restart polls the new build with the old snapshot shape,
and `section.error` on an undefined section throws inside render and
blanks the whole page — the one thing this design exists to prevent. Use
`sectionError(section)` from `api.ts`, not `section.error`.

**Overdue is not "next up."** `_autonomy` splits scheduled tasks on
`next_run` vs now and returns them as separate lists. Sorting them
together ascending and labelling the head "next up" is how a fleet whose
ticker is months behind renders as a healthy schedule — the most overdue
task lands exactly where the soonest one belongs. Likewise `completed`
is excluded from worker "open" counts (`_OPEN_STATES`): it dominates the
depth table and would bury the handful of items actually waiting.

**And overdue is not "held."** The clock is only one of five gates the
scheduler applies. A task also needs a skill, `up_next` status, no failure
cooldown, a satisfied `depends_on`, and the current hour inside its
`preferred_hours`. `autonomy.hold_reason` mirrors `_is_task_due`'s gates in
order and returns the first one that bites (`"paused"`, `"waiting on #42"`,
`"outside hours 00-04,23"`, `"no skill"`) or `None`; the panel calls a
past-due task **overdue** only when nothing holds it, and **held** otherwise.
Both `_autonomy` and `GET /api/autonomy/tasks` call that one function rather
than restating the gates — a second private definition of "due" is what this
fixed. On 2026-09-06 the dashboard showed six overdue while the scheduler
considered none of them late: four nightly jobs outside their window and two
paused. A nightly task is past due for the eighteen hours a day it is not
allowed to run, so the counter was never zero and therefore said nothing.
The `classifier` field reports `naive` when `autonomy` could not be imported
and every past-due task is being called overdue, because a downgrade that
looks like success is the failure this whole split exists to prevent. Note
that the dependency gate resolves `depends_on` by id and treats an
unresolvable id as *met*, so it must always be handed the **whole** board —
`/api/autonomy/tasks?status=up_next` classified against its own filtered list
would report every dependency satisfied.

**Front matter is bounded by its closing `---`, not by a byte count.**
`_frontmatter` reads in 4 KB chunks up to a 64 KB ceiling and stops at a
line-anchored `^---$`. The previous flat 3000-byte prefix silently dropped
five backlog items, and the selection was causal rather than random: an item
grows its `activity_log` precisely by being worked on, so the two it hid were
the two that were `in_progress` — the board reported zero. A cap that hides
whatever is most active is the worst possible reading of "bounded". Splitting
on bare `"---"` is the matching trap: it also fires inside quoted log prose
and truncates the block somewhere plausible. A block that parses to a list or
a string returns `{}`, since the caller's first move is `.get`.

**Subagents and background bash tasks live in the lloyd-mcp process, not
the backend.** The aggregator owns the `Task` tool and spawns
`Bash(run_in_background=true)` children, so the backend has no handle on
either. `agent_mcp/main.py` exposes `GET :8500/state` beside `/health`
and `app/routers/dashboard.py` reads it over loopback. Adding a new
agent-side live panel means extending that route, not the backend.

`background_tasks` carries `active` **and** `recent`. `list_active`
filters on `status == "running"`, so before that a background bash left
the dashboard the instant it exited — a task that died three seconds in
was indistinguishable from one that never started, which is the opposite
of what a background task most needs to report when nobody is watching
its terminal. `list_recent` is bounded by its limit rather than by
eviction: `_records` is kept whole so a later `get(task_id)` can still
hand the model an output path to Read. A finished row's `elapsed_s` is
measured against `finished_at`, not `now`, or a task that ran for two
seconds reads as hours old by evening.

**Workers are not in that panel.** The worker pool lives in the backend
(`workers.queue` + `workers.pool`, rendered by `WorkersPanel`), while
subagents and background bash live in the aggregator. A worker job whose
prompt calls `Task` does put subagent rows there — via
`workers/sources/_common.py::run_prompt_on_primary` — but anonymously:
nothing on the row says which worker source it came from.

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

## Session titles and live activity

Every Mission Control surface that names a session — the chat history
list, the chat header, the dashboard's agent panel, the Inner Voice
picker — renders a few-word **title** rather than the timestamp id.
`app/session_titles.py` owns it end to end; the id survives as the
element's `title=` tooltip.

Titles are written by the **secondary** model
(`_sync_secondary_title`), fired and forgotten off turn completion
beside `_post_session_capture`. That slot is single-tenant
(llama.cpp `--parallel 1`) and agent turns already queue behind it, so
`should_title` re-titles on a **geometric** schedule — after the 1st
real user message, then the 3rd, the 9th, the 27th — recorded in
`title_at_count`. A per-turn title call would put a model call in that
queue for a label nobody asked to be refreshed.

`clean_title` is strict on purpose and returning `""` is a normal
outcome: a bad title is worse than none, because the id at least
identifies the row while `Here is a title for the conversation` just
looks like a bug. Consumers share one fallback chain —
`web/src/lib/sessionLabel.ts`, title → preview → id — so a session never
reads as two different sessions in two panels.

`title_for` caches on a **TTL, not on mtime**. The session JSON is
rewritten on every appended message, so an mtime-keyed cache would
re-parse a multi-megabyte transcript on every 2-second dashboard poll,
which is the exact cost the cache exists to avoid. `invalidate` closes
the staleness window when a title is written.

**Live activity** is the second half. `SessionTurn.activity`
(`{kind, label, detail, at}`) is stamped by the turn runner as it
streams — `starting` → `prefill` → `thinking`/`responding` → `tool` →
`working` — and surfaces through `active_sessions_snapshot`. "Busy" is
equally true of a turn prefilling 160k tokens, one four minutes into a
`Bash` build, and one wedged on a dead engine; this line is what tells
them apart. It is display state the loop never reads back, so writing it
is a no-op when nothing is running and a no-op again when the state is
unchanged (the text path calls it per token).

The snapshot itself stays **pure in-memory queue state** — it is also
the selfmod promoter's idle gate, and a disk read there would put the
filesystem in front of a restart decision. Titles are joined on in
`_primary_state`, off the loop via `asyncio.to_thread`.

Each row on that panel is a button that opens the session in the Inner
Voice tab, through the same `setPendingFocus` + `setCurrentTab` pair the
agent's `mc_navigate` uses — so `InnerVoicePage` never has to know who
asked. Two things that panel taught us:

- **A page that applies incoming focus must not race its own list
  fetch.** `loadSessions` used to read `selectedSession` out of its
  closure to decide whether to default to the newest session. On mount
  that closure captures `null`, the fetch resolves *after* the focus has
  been applied, and the stale `null` overwrites it — so every row on the
  dashboard opened the same chat. Use the functional updater
  (`setSelectedSession(prev => prev ?? list[0].session_id)`) and keep the
  callback's deps empty; anything else reintroduces the race.
- **The Inner Voice picker holds only IV-enabled sessions**, but focus
  can point anywhere. A `Select` whose value matches no option renders an
  empty trigger, so the picker carries an out-of-list selection in as its
  own option and names it from `/api/sessions/{id}/meta`.

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

## Voice output

The cloned voice (`clone:dave_cullen`, config `livekit.tts`) is synthesised by
Qwen3-TTS at :8090 and then **shaped client-side** by
`agent-services/tts_shaping.py` before it reaches LiveKit. Two things the
server does not do, both fixed in the worker because the TTS tree is
gitignored and a rebuild would silently delete a fix made there:

- **The 12 Hz speech tokenizer rolls off above ~1.5 kHz.** Against the
  clone's own reference clip the output matches the real speaker to within
  0.5 dB below 1.5 kHz and is then 1.8–6.2 dB down all the way up. That
  missing presence band is the "he's in a broom closet" sound. Two high
  shelves (`livekit.tts.shaping.shelves`) put it back. It is a vocoder
  property, not a bad reference — built-in voices with no cloning measure the
  same, and re-cutting the reference from another source video did not move
  it. **Do not "fix" it by cutting 300 Hz**: that band already matches the
  reference to 0.1 dB, and cutting it trades hollow for thin. The trap is the
  metric — gate frames on total RMS and raising the highs swaps vowel frames
  for fricatives in your own measurement, which reads a +9 dB shelf as +24 dB.
  Gate on sub-1 kHz energy, which the correction cannot move.
- **`speed` is dropped by the server's streaming path.**
  `generate_voice_clone_streaming` has no such parameter while the
  non-streaming path applies `librosa.effects.time_stretch`, and voice mode
  always streams — so `livekit.tts.speed` was inert for the only path that
  uses it. `WsolaStretch`
  applies it in the worker, and the request now sends `speed: 1.0` so a future
  server-side implementation cannot stretch twice. WSOLA rather than a phase
  vocoder: a phase vocoder adds exactly the smeared quality the shelves exist
  to remove.

Pace is set by ear, not by matching the reference's words/second: the w/s
match puts `speed` at 0.70 and that is audibly too slow, because the reference
is one deliberate segment and the model places its pauses differently. Every
setting from 0.70 to 1.00 measures *faster* than the reference by w/s. Sweep a
single synthesis across settings rather than re-synthesising per setting, or
sampling noise reads as the effect of the knob.

An utterance also ends in `livekit.tts.tail_silence_ms` of silence and waits on
`AudioSource.wait_for_playout()`: `_stream_utterance` returns when audio is
*queued*, not played, so `on_utterance_end` fired early and a following
`interrupt()` → `clear_queue()` cut the last syllable off.

Both stages hold state across chunk boundaries and are reset per utterance —
an interrupt mid-stream must drain the shaper or the next utterance opens with
the tail of the one the user talked over. `tests/test_tts_output_shaping.py`
pins chunk-invariance, rate stability, and pitch preservation. Measurements and
listen files: `~/obsidian/projects/lloyd/voice/voice-source-dave-cullen.md`.

A voice change needs **`lloyd-agent-worker`** restarted (it reads
`livekit.tts` once at construction); `agent-livekit-server` is the SFU binary
and never reads TTS config. It also needs the guardian **re-staged**
(`systemctl --user restart lloyd-guardian`), because the same voice is used
for spoken alerts and `sync-voice-config.py` pushes `livekit.tts` across at
stage time — see "The spoken channel" above. Neither restart is required for
the voice to *work*, only for a change to reach that consumer.

## Workers: one queue, everything unasked

Scheduled autonomy tasks, research, session mining, backlog triage and the
selfmod round that implements a confirmed item all run through one SQLite
queue drained by `workers.slots` asyncio workers **inside the backend
process**. `architecture/workers.md` is the long version.

- **`priority ASC` — a lower number runs sooner.** Read backwards once
  already: `backlog-implement` sat at 80, behind research jobs at 70 that
  arrive every few minutes, and the rarest, most valuable job in the pool had
  no path to a slot.
- **`max_inflight` is applied in the SQL, not to a window of rows.**
  `claim_next` used to select 50 and skip over-quota rows in Python, so fifty
  queued items from one saturated source hid every claimable row behind them
  and the pool read the queue as empty.
- **A raised failure retries; a returned `{"status": "failed"}` does not.**
  That is the whole difference between "infrastructure hiccuped" and "the job
  failed and re-running it will not help". Raising for the latter is what made
  one timed-out autonomy task re-run three times at 600 s before the
  scheduler's own cooldown was consulted.
- **`skipped` is a third outcome and must be said as a status.**
  `selfmod-regression` signalled "I could not measure anything" by returning
  `{"skipped": reason}` — a key where a status belongs — so all 22 of its runs
  read as successes with an empty summary. `pool.normalize_result` is the
  contract now, and it reads that shape.
- **Nothing in a source may block the event loop.** It is the loop that serves
  every HTTP request and streams every chat turn, so a `subprocess.run` inside
  `execute` does not slow the pool, it stops Lloyd answering.
  `selfmod_regression` ran two 900-second eval arms there.
  `tests/test_workers_pool.py` greps for the pattern.
- **An empty turn is a failed turn.** `run_prompt_on_primary` returns a
  `TurnResult`, not a string, because a turn that dies at `max_turns` yields
  no text and an empty string is indistinguishable from a short answer. It
  used to return just the text: **225 of the 498 notes under
  `pending-research/` have the body `(no response)`**, and the source
  responsible ticked each topic off on the way, so none can be retried. That
  source, `domain-research`, has since been retired — see
  `architecture/research-pipeline.md`.
- **Research runs off a registry now, not a checklist.** `research.db`
  (`app/research_store.py`) holds topics with a lifecycle; task #65 proposes
  into it through `research_propose` and the `deep-research` source drains it
  through the deep-dive skill. The checklist it replaced held 2,839 items of
  which 314 were unique, and reading it cost 1.02M tokens a night.
  **Retries live in the registry**, because the pool completes an in-band
  `failed` item and never retries it.
- **A session-backed turn's own timer must beat the pool's.**
  `run_prompt_in_session` bounds itself at `max_duration_seconds` minus 60 s
  and cancels the turn in the backend on expiry. If the pool's `wait_for`
  wins, it cancels the HTTP request — and the chat path deliberately keeps
  running when its client disconnects, so the turn is orphaned while the pool
  requeues and re-selects the same item.
- **`session-distill` distils a session once, quiet, and only if a user wrote
  it.** Its watermark moved with each new message, so an active chat re-
  qualified on every tick — one was distilled 44 times — and worker sessions
  were mined back in as observations about the user.
- **`workers.enabled` is UI-mutable, so it lives in the override file.**
  `POST /api/workers/enable` used to `yaml.dump(CONFIG)` over the tracked
  `config.yaml`, which would have written expanded secrets into the tree,
  flattened its comments, and dirtied it — stopping the selfmod loop. Same
  route as the Tools page now.

Pause and drain the pool before restarting the backend
(`POST /api/workers/pause`): a worker turn killed mid-flight logs connection
errors that land in the guardian's observation window and get blamed on
whatever just landed.

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

session_titles:
  enabled: true    # false => surfaces fall back to the preview text
```

## Development Notes

- Each turn reconstructs the full conversation from the persisted session JSON (`load_and_compact_session`) and sends it as an OpenAI-format `messages` list to vLLM.
- vLLM tool calling: `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`
- Session continuity: no `resume=` — history is rebuilt from `sessions/<id>.json` each turn.
- The `/api/message/stream` endpoint uses SSE. The frontend connects via `fetch` + `ReadableStream`.
