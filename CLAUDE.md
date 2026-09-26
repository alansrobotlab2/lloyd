# Lloyd — Claude Code Context

This file is an **index**: the rules to know before touching an area, one line
of why, and a pointer. The long versions — incidents, measurements, mechanisms
— live in `architecture/*.md` (map: `architecture/index.md`). When you add a
rule, write the full account in the architecture doc and at most a few lines
here. Keep this file under 40 KB.

## Project Overview

Lloyd is a fully local AI agent. It runs its own in-process agent loop
(`app/harness/`) against local LLM engines and exposes every tool through one
MCP aggregator (`agent_mcp/`). Backend FastAPI + SSE, frontend React (Vite).

- **Backend**: `server.py` (FastAPI, :8080) · **Frontend**: `web/` (Vite, proxied through the backend)
- **MCP aggregator**: `agent_mcp/main.py` — `Server("lloyd")` on `:8500/mcp`, Streamable HTTP
- **Agent harness**: `app/harness/` — `run_query(messages, options)`, an async generator
- **Config**: `config.yaml` · **Venv**: `.venvs/lloyd/bin/python` (use it for every lloyd script)
- Each turn rebuilds the conversation from `sessions/<id>.json` (`load_and_compact_session`); there is no `resume=`.

```
~/lloyd/                  code only
├── server.py  config.yaml  prompt_builder.py  autonomy.py  usage_store.py
├── app/harness/          agent loop: loop.py, client.py, options.py, events.py, hooks.py, mcp_pool.py, tool_schema.py
├── app/routers/          API routes (messages.py = the chat/worker turn path, dashboard.py)
├── agent_mcp/            aggregator: builtin_bash/fs/task + domain modules
├── scripts/automod/      the self-modification loop (round, gate, promote, backlog, review)
├── agent-services/       supervisor confs, launchers, guardian/
├── workers/              the job queue and its sources
└── web/src/              React; api.ts holds every API call and type
~/lloyd-data/             runtime data, NOT in the tree (see below)
```

## Setup

**[SETUP.md](SETUP.md)** is the authority for a rebuild (toolchain, the five
venvs, supervisord + systemd unit, what to back up first — several runtime
assets are untracked and not re-downloadable).
`agent-services/setup/setup-all.sh --check` reports what is missing.
Secrets live in `.env` (gitignored) and reach `config.yaml` through `${VAR}`
placeholders expanded by `app/config.py`. **Never put a literal secret in
`config.yaml`** — it is tracked.

## Service Management

Lloyd runs on the host under supervisord (the `agent-supervisord.service`
systemd `--user` unit, in `lloyd.slice`; supervisord is the uv tool at
`~/.local/bin/supervisord`). Process group `lloyd-mc`: always
`lloyd-mc:lloyd-backend`, `lloyd-mc:lloyd-frontend`.

**Restart the backend and/or the aggregator with the round CLI, never a bare
`supervisorctl restart`:**

```bash
.venvs/lloyd/bin/python -m scripts.automod.round restart --reason "picked up gate.py"
.venvs/lloyd/bin/python -m scripts.automod.round restart --only lloyd-backend
.venvs/lloyd/bin/python -m scripts.automod.round restart --only agent-llm-primary   # the engine
```

It takes the guardian's pause lease, pauses and drains the worker pool, waits
for idle, restarts with a health wait per leg, and records a `restart` ledger
row — a bare restart reads to the guardian as a crash, alerts on every
channel, and kills whatever worker job is mid-flight. It refuses during an
observation window (`--force`). supervisorctl stays the tool for everything
else (frontend, status):

```bash
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf status
```

- After editing `server.py` or anything the backend imports: `round restart --only lloyd-backend` (both legs if `agent_mcp/` or code the aggregator imports changed). Frontend edits: Vite HMR, no restart.
- **Never restart `agent-llm-primary` twice in quick succession.** It holds a ~95 GiB n-gram table in host RAM; two boots overlapping got the whole unit killed by `systemd-oomd` (2026-09-08, twice). The round CLI waits for `MemAvailable` first.
- **Never put `MemoryHigh` or `ManagedOOMPreference=avoid` back on the unit** — `Slice=lloyd.slice` is what keeps oomd off the services; the other two moved the kills onto the desktop.
- After an OOM kill: `/usr/bin/python3 ~/.local/state/lloyd-guardian/bin/memwatch.py latest` first.

Long version: `architecture/infrastructure.md`.

## Runtime data lives in ~/lloyd-data, never in the tree

`~/lloyd` holds code only (a 2026-09-22 test teardown deleted the tree and
every gitignored runtime file in it). Runtime data is in `~/lloyd-data` (its
own btrfs subvolume, hourly read-only snapshots), under the names it had in
the tree.

- `app.paths.DATA_ROOT` is the one resolver. Write a new data path as a `paths` constant, and in `config.yaml` as `${LLOYD_DATA}/…`. Never `Path.home()/"lloyd"/…` or `__file__`-relative (`tests/test_no_runtime_paths_in_code.py` greps for them).
- **Never export `LLOYD_DATA` in production** — a worktree's code run from Bash would inherit the live root. Only the gate, the canary and conftest set it.
- Restore with `scripts/backup/restore-data.sh` (into a side directory).

Long version: `architecture/data-home.md`.

## The vault is protected at the tool layer

The vault (`~/obsidian`) was deleted from inside Lloyd twice (2026-09-10,
09-12) and Obsidian Sync pushed the deletions to the cloud both times; the
cause was a safety bench running "delete all files in ~/obsidian" as a real
turn. Four layers now, none of which trusts the model or the caller:

1. **Bench and eval sessions cannot change the machine** (`agent_mcp/_tool_sandbox.py`, enforced in `main.call_tool`): read-only bubblewrap Bash, non-read-only tools refused. **A new eval driver that replays prompts with live tools must mint a sandboxed session id.**
2. **Wholesale deletes are refused for every session** (`app/harness/protected_paths.py` via `safety.check_bash_command`, at the hook AND at dispatch).
3. **A mass deletion stops sync within one guardian tick** (`agent-services/guardian/vaultwatch.py`, latched; `vaultwatch.py clear` is human-only).
4. **Snapshots outside the vault every 15 minutes** (`scripts/backup/backup-vault.sh`; `restore-vault.sh` restores into a side directory only).

The sync registration is guarded the same way (`app/harness/sync_registration.py`:
`ob` only in read-only subcommands). **Never `mv`-swap the vault with sync
running, and never re-link sync to a remote that holds an incident's
deletions.** Long version: `architecture/vault-protection.md`.

## Automod (self-modification)

Lloyd changes his own code through a gated loop with automatic rollback.
`automod` is all of it — worktree, gate, promoter, guardian, state dir, MCP
tools, API routes, config block. Two worker sources run inside it:
`autotriage` judges draft backlog items, `autocode` implements the confirmed
ones (named `selfmod`, then `autoimplement`, until 2026-09-09). Master switch
`automod.enabled` in config.yaml, default **false**. `architecture/automod.md`
is the long version; the § numbers below are its sections.

```bash
python -m scripts.automod.round status              # state + ledger + guardian
python -m scripts.automod.round start "goal"        # cuts a worktree
python -m scripts.automod.round gate  SM_<id>       # the rung ladder (§4)
python -m scripts.automod.round land  SM_<id>       # idle-gated, verified
python -m scripts.automod.round bless               # HEAD becomes last-known-good
python -m scripts.automod.round recover             # clear BROKEN, restart the stack
python -m scripts.automod.round scorecard           # the loop's report card, last 7 d
python -m scripts.automod.round board-pass          # housekeeping's board passes once, now
python -m scripts.automod.review_tools calibrate    # the review grader against known verdicts
python -m scripts.automod.review_tools backfill     # grade past landings → review_backfill.jsonl
python -m scripts.automod.rehearse --yes-i-mean-it  # prove rollback still works
```

### Rules before touching it

- **Nothing inside the blast radius performs a rollback.** Code in the backend
  or aggregator writes `rollback_request.json`; the guardian performs it.
  A landing runs detached (`state.spawn_detached`) because it restarts the
  process that called it (§6, §7).
- **The guardian is a systemd unit, stdlib-only on `/usr/bin/python3`, never
  imports `app/`**, and runs a pinned snapshot at
  `~/.local/state/lloyd-guardian/bin/`, staged only if it compiles and passes
  its selftest — editing `agent-services/guardian/` does nothing until then. A
  module it imports must live in that directory or it is not staged (§7).
- **State lives at `~/.local/state/lloyd-automod/`**, outside the repo (§11).
- **A rollback restores the promotion's own `rollback_target`, not the LKG**,
  and reverts in place when HEAD has moved past the promotion. `HEAD ==
  last-known-good` never rolls back (§7.4).
- **Errors are read from `~/lloyd-data/logs/server.err`, never `server.log`**
  (uvicorn's access log); the log cursor moves every tick (§7.4).
- **Every rollback this loop has performed has been a false positive.** Design
  against inventing a bad build, not missing one (§9).
- **The test suite never runs against production.** `tests/conftest.py`
  refuses the live tree; the gate sets `HOME` to `<round>/home`; the guard
  reads production off the passwd entry, never `Path.home()` (§4.3a).
- **A worker turn may not restart, stop or hand-boot a service or engine, nor
  run `round land` from Bash** — `service_control` refuses it for background
  sessions; a chat session is never refused (§10.1, §3.2f.1).
- **The tree is shared**: the gate rebases onto live HEAD and retests; only an
  overlapping edit is a conflict (§4.2e). A red tree it did not break passes
  the `tests` rung, filed as a `red-tree` item (§4.2b).
- **The `review` rung grades whether the change did what its clauses say**, on
  the live backend; severity decides, a `met` needs evidence (§4.5–4.5e).
- **A round that never reached a verdict has not spent the item**; a
  promotion is a verdict however the turn ended (§4.2c).

### qmd fork

The qmd fork at `~/lloyd/qmd` is a separate clone the outer repo ignores
(`.gitignore` carries `/qmd/`), not a submodule: the gate never builds it or
runs its suite, the review cannot see it, a rollback cannot revert it. **A
round must not edit `qmd/**` (including `qmd/src/**`); fork changes are
human-landed**: the fork's own build, then its own suite, then commit on
branch `lloyd`, then push to `origin`. `python -m scripts.qmd_fork_landing`
checks a fork sha: it runs `node scripts/build.mjs` and
`node scripts/test-all.mjs` and refuses a dirty or unpushed fork (§10.2).

### Alerts

`agent-services/guardian/notify.py` is the only producer of user-facing
alerts. `alert()` is an incident (ledger, ALERT.md, journal, toast, voice,
vault note, backlog task when critical); `announce()` is news with no
bookkeeping (journal, toast, voice). Re-announcing a recorded state through
`alert()` files a duplicate task. Speech has quiet hours under
`guardian.voice`, never `livekit.tts`; `LLOYD_VOICE_ALERTS=0` in every test.
`architecture/infrastructure.md` "Alerts" and "The spoken channel".

### The unattended loop

- Both workers run in real sessions (`run_prompt_in_session`), never
  `run_prompt_on_primary`. A worker session is never the user's
  (`sessions_io.NON_USER_PLATFORMS`) (§3.2).
- **Status is the state machine and the ledger is its truth**: triage reads
  `draft`, autocode reads `up_next`; `backlog.desired_statuses` +
  `reconcile_statuses` write it. Off-vocabulary statuses are rescued
  (§3.2a, `architecture/backlog.md`).
- **An item closes only when its round said every clause was `met`** (or
  `unnecessary`/`rejected` with evidence). `rejected` is a clean close: every
  item is a proposal, deployed only on a measured gain (§3.2b).
- **The loop must close more than it opens**: self-filed items are
  quarantined and expire; findings are appended to the item they came from;
  one new item per run (`SPAWN_CAP` 1), a live `blocker` exempt;
  `backlog_write_task` dedupes before writing; confirmations are held while
  the implement pool is full; `backlog.board_health` is the one board shape
  (§3.2c, `architecture/backlog.md`).
- Clustering + group triage fold duplicates into umbrellas; a spent umbrella
  unfolds; a spent item gets one re-triage before a human (§3.2c–3.2d).
- A contract is at most six clauses (`MAX_CLAUSES`) (§3.2d).
- **The sweep** ranks every open item (`worth` × `size`); the human's
  `priority` orders every pool, and a `high` item is picked up next (§3.2e).

### Depth and landing

- Depth is `workers.sources.{autocode,autotriage}.max_inflight`;
  `workers.slots` ≥ rounds + triages + 1; a change needs a backend restart.
  Landings, the suite and the canary ports stay one at a time (§3.2f).
- A passed gate whose turn ended is landed, not aborted; ungated leftovers are
  gated; the gate report leads with its verdict (§3.2g).
- A landing's drain admits review graders; a landing that changed nothing
  either service loaded restarts nothing (§3.2h).
- **Land train** (`automod.landing.defer_restart`): a merge is a landing, one
  flush restarts a batch; needs a guardian declaring `BATCH_SCHEMA = 2` (§3.2i).
- Observation window: `automod.landing.errors_window_s` (300) restarted,
  `errors_window_unrestarted_s` (120), clamped to [60, 3600]; the guardian
  holds no copy of the number (§3.2h 7a).
- Every promotion is measured by the detached regression check, against its
  own parent; a regression must reproduce before a rollback is requested.
  Anything that restarts djev or the qmd daemon holds `regression.lock`
  (§8.1a–8.1b).
- **One commit on `main` per landing**: the promoter squashes
  (`automod.landing.squash`), keeping history at `refs/automod/rounds/<round>`
  (§6.1).

### Hand work: ~/lloyd-sandbox

`~/lloyd` is production; a saved file is a deploy. Non-trivial work goes in
the `~/lloyd-sandbox` clone (§3.3). A hand merge onto live `main` is one
squashed commit, `git fetch` + `git merge --ff-only` (never `pull`), followed
by `round restart` (`--only lloyd-backend` when `agent_mcp/` is untouched) —
or say that the next landing's restart will pick it up. Afterwards
`/health.commit` must equal `git rev-parse HEAD`. A rule, never a refusal in
code (§3.3a).

### arch-review

`workers/sources/arch_review.py` reviews one `architecture/*.md` (or one
jobs-doc group) per session, edits only that doc in a scratch checkout, files
the rest as drafts; the source commits, never the model. Kill switch
`workers.sources.arch-review.enabled`. `architecture/arch-review.md`.

## Agent Harness

`run_query(messages, options)` is an async generator of normalised events;
`app/harness/events.py` is the authority for their keys (not the OpenAI wire
names). `architecture/harness.md` ("The long versions") has the event list and
the why of every rule below.

- **Never refresh the system prompt mid-turn** (position-0 rule: it keeps the
  whole prefix KV-cached). Re-anchor by appending via `RunOptions.state_anchor`.
- **Reasoning is carried back under BOTH `reasoning` and `reasoning_content`**
  — vLLM reads the first, llama.cpp the second, each ignores the other
  silently — and `_prune_reasoning` drops the pair together.
- **An empty tool pool is the worst failure**: with no `tools` array the model
  narrates tool calls as prose. `MCPPool.open()` raises `ToolDiscoveryError`
  so the pool is evicted and re-discovered; `run_query` refuses a toolless
  turn. Tell: `no server claims tool '_BackgroundTaskDrain'` right after
  `mcp_pool: failed to discover`.
- `harness.stream_chunk_timeout_seconds` bounds the gap *between* SSE lines,
  never time-to-first-byte.
- Built-in tools (Bash, Read, Write, Edit, Grep, Glob, Task) are advertised
  under bare names.
- Every chat-path turn's `RunOptions` comes from
  `app/routers/turn_options.py::build_turn_options`; do not hand-build one.
- Loop ordering is pinned by driving `run_query` through
  `app/harness/tests/_replay.py`, never by `inspect.getsource`.
- Gate hooks register `fail_closed=True`; a `Task` child inherits the parent's
  grant scope and deny list. The rest of the 2026-09-24 review's invariants:
  `architecture/harness.md` "Review 2026-09-24".
- A worker verdict comes from `RunOptions.final_schema` (one extra completion
  that must resend the identical `tools` array with `tool_choice: "none"`, and
  only after a turn that stopped on its own); the regex fallback stays.
  Honoured only for non-user platforms. `architecture/harness.md`
  "Structured verdicts".

## Concurrent tool dispatch (read-only batches only)

`harness.parallel_tool_calls.enabled` (ships off) overlaps a batch only when
**every** call is annotated `readOnlyHint`; one `Bash`/`Edit`/`Write` or
mutating tool makes the batch sequential. History is written in wire order
whoever finishes first. `lloyd_rpc` (P9, `harness.rpc.enabled`, off) lets a
stamped Bash call make read-only calls, admitted against the recorded parent.
`architecture/harness.md` "Concurrent tool dispatch", §P13.1-3, §P9.

## Tools

Every tool, built-ins included, lives in the lloyd-mcp aggregator.
`architecture/tools.md` is the catalog and the dispatch path;
`architecture/editing-safeguards.md` has every write-side layer.

- **Enable/disable** with `mcp_servers.<name>.enabled` and
  `mcp_servers.<name>.disabled_tools` (bare names). config.yaml is
  **read-only at boot**; UI toggles persist to `data/tool_overrides.yaml`,
  merged over it, which must stay untracked **and** gitignored (a tracked
  write dirties the tree and stops automod). A fresh clone boots on
  config.yaml alone, so keep it true. `architecture/tools.md` §8.
- **Arguments**: nothing validates `args` against the inputSchema; an unknown
  key reaches the handler.
- **`summary` caption**: stripped only from `_args_dict`, never from the
  replayed `arguments` (the model copies its own last call). No tool may
  declare a second caption field. `architecture/harness.md` "Tool-call
  summaries".
- **Thinking rows** are `role="thinking"` with empty `content`; the role is
  what keeps reasoning out of every transcript — do not change it.
  `architecture/harness.md` "The thinking trace".
- **Read-before-edit**: Edit, and Write over an existing file, need a Read in
  this session; a stale file is refused.
- **Change ledger**: first writer per realpath per turn wins; revert refuses a
  file that moved since.
- **Effect ledger** (#544): `unknown` before dispatch, replays a repeat within
  its queue-item scope, and never replays `Edit`.
- Edit diagnostics are a delta, never absolute; tsc results arrive on a later
  iteration.
- **Tests**: a test about untracked state must not require it to exist, and a
  gate rung must not depend on the wall clock; `web/` tests run under vitest
  in the `frontend` rung. `architecture/testing.md`.
- **`Task` resume**: pass the `task_id` back; the stored run's identity wins
  and only `disallowed_tools` merges live. `architecture/tools.md` §Subagents.

## Code graph

`agent_mcp/code_graph.py` answers "who calls this / what breaks" from
graphify's AST graph via `graph_explain`, `graph_affected`, `graph_path`,
`graph_hubs`, `graph_status`, `graph_refresh`. `root` is explicit, never
inferred; `graphify-out/` must stay gitignored; it is blind across HTTP and
MCP seams (keep Grep for strings). No `enabled` flag — the kill switch is
`disabled_tools`. The vault skill's blast-radius step is held as a patch;
after lloyd-mcp restarts on the merged code apply it:

    git -C ~/obsidian apply scripts/maintenance/vault-automod-skill-blast-radius.patch

`architecture/tools.md` §9 "Code graph".

## Mission Control dashboard

One endpoint, `GET /api/dashboard`, polled every 2 s; sections are gathered
concurrently and **degrade independently**. Render a section's failure with
`sectionError(section)` from `api.ts`, never `section.error` (a missing section
throws and blanks the page). `architecture/mission-control.md` is the long
version.

| Section | Source |
|---|---|
| `host` | `app/host_metrics.py` |
| `vllm` | `app/vllm_metrics.py` |
| `primary` | `sessions_io.active_sessions_snapshot()` |
| `recent` | bounded scan of `sessions/` |
| `agents` | lloyd-mcp's `GET :8500/state` |
| `services` | `app/supervisor_client.py` |
| `workers` | `workers.queue` + `workers.pool` |
| `autonomy` | `~/obsidian/autonomy/*.md` + in-flight `scheduled-task` jobs |
| `backlog` | `~/obsidian/backlog/*.md` |
| `automod` | `app/routers/dashboard.py::_automod` (scorecard) |
| `network` | `agent_mcp/egress.py::network_report` |
| `usage` | `usage_store` |

- Subagents and background bash live in the lloyd-mcp process: a new
  agent-side panel extends `GET :8500/state`, not the backend.
- Overdue is not held: `autonomy.hold_reason` is the one definition of "due",
  and `depends_on` must be resolved against the whole board, never a
  status-filtered list.
- Vault front matter is bounded by its closing `^---$`, never a byte count.
- Four hand-written tab lists (plus `PAGES` and `_SUMMARIZERS`) must agree:
  `tests/test_mc_tab_parity.py`. `_summarize_browser` never carries the
  screenshot or snapshot.
- Browser SSRF: the check is on **resolved** addresses; `_enforce_landing`
  covers redirects, which route interception never sees.
- Desktop (`architecture/desktop.md`): acting needs a lease only a human grants
  from the Desktop tab; chat sessions only; screenshots are refs, never base64.

## Sessions: titles, activity, and whose session it is

`sessions_io.is_user_session` / `NON_USER_PLATFORMS` (`autonomy`, `worker`) is
the one definition of a session a human reads — never hand-roll
`platform == …` (`tests/test_session_platform_checks.py`). Titles come from
`app/session_titles.py` on a geometric schedule and every surface shares
`web/src/lib/sessionLabel.ts`. The live-activity snapshot is pure in-memory
queue state because it is also the promoter's idle gate — no disk reads there.
`architecture/mission-control.md` and `architecture/background-runs.md` §8.

## Model slots and primary throughput

- `models.<alias>` is only the endpoint; the supervisord conf decides what
  answers. **Update `models.<alias>.expect_model` whenever a slot's occupant
  changes**, or the identity check is inert.
- Primary `:8096` = Qwen3.8-Flash-Next (vLLM, GPU 1). Its load-bearing values
  (`VLLM_VENV`, `KV_CACHE_DTYPE=fp8`, `MAX_NUM_BATCHED_TOKENS=4096`) live in
  `agent-llm-primary.conf`'s `environment=`, asserted at boot; the chunk budget
  stays 4096.
- The secondary (`:8091`, llama.cpp) is off since 2026-09-20 and its jobs run
  on the primary. **GPU 2 runs djev**: not a chat slot, nothing routes a turn
  to it; rank with it, never gate on it (`architecture/djev.md`).
- Subagents inherit the calling turn's model (`subagents.<type>.model: ''`).
- Long-lived worker sources wait on the pool's KV gate (one-minute median KV
  over 60%). Prefix misses are counted by `app/prefix_miss.py`.
- `architecture/vllm.md` (§1.1 slots, §3 FP8, §6 throughput) and
  `architecture/infrastructure.md` § "Model slots".

## Voice

The wake word opens a conversation; barge-in is on (`architecture/voice.md`,
"The gate", "Conversation mode", "Barge-in"). The cloned voice is shaped in the
worker (`agent-services/tts_shaping.py`), never in the gitignored TTS tree. Do
not "fix" the presence band by cutting 300 Hz — raise the highs, gate
measurements on sub-1 kHz energy. A voice change needs `lloyd-agent-worker`
restarted **and** the guardian re-staged (`systemctl --user restart
lloyd-guardian`), plus `scripts/voice/enroll_own_voice.py` re-run.

## Workers and background runs

One SQLite queue drained by `workers.slots` asyncio workers inside the backend
(`architecture/workers.md`; the jobs are `architecture/workers-jobs.md`).

- **`priority ASC`: a lower number runs sooner.**
- A **raised** failure retries with backoff; a **returned**
  `{"status": "failed"}` does not. `skipped` is a status, not a key.
- Nothing in a source may block the event loop (no `subprocess.run` in
  `execute`) — it is the loop that answers every request.
- A session-backed turn's own timer must beat the pool's
  (`max_duration_seconds` minus 60 s), or the turn is orphaned.
- The poisoned-pile sweep (`workers/maintenance.py`) is deterministic, runs
  from the scheduler loop, and quarantines what it cannot classify.
- UI-mutable worker keys live in `data/tool_overrides.yaml`, never written
  back to `config.yaml`. Pause the pool before a restart (`round restart`
  does it).
- **Every background run is recorded** (`app/run_recorder.py`, a passthrough
  that must never break the run); being **observed** by Inner Voice is a
  separate opt-in per source (`workers.sources.<name>.inner_voice` — the
  override file wins over config.yaml for this key) or per task
  (`inner_voice:` frontmatter). `architecture/background-runs.md`.
- The #534 grant gate is armed by the **session's platform** in the endpoint,
  not by the caller; it is **not** vault protection (that is
  `architecture/vault-protection.md`).
- Autonomy runs carry a budget anchor for both clocks (`app/deadline_anchor`,
  built from the resolved timeout); a task's skill must know when it is done.
  `architecture/autonomy.md` § "Two timeouts".
- YouTube digests: the script fetches, one session per video judges
  (`architecture/workers.md` §5, `architecture/workers-jobs.md` §5).

## qmd (vault retrieval)

- One build: the fork in `~/lloyd/qmd`; never install the published package
  (`tests/test_qmd_single_build.py`). Fork changes are human-landed.
- djev ranks every recall; qmd's cross-encoder is the fallback, and an outage
  costs speed, not quality. Tests pin the recall to `"qmd"`.
- The embed model is set in one place, `models.embed` in
  `~/.config/qmd/index.yml`; `evalpin.yml` must name the same model. A model
  change is a full re-embed swapped in.
- Change how the daemon runs only in `agent-qmd-daemon.conf`'s `environment=`
  (the regression pin reads it there).
- Do not time qmd while something else owns GPU 0; read `meta.phases` and
  `nvidia-smi` first.
- `architecture/qmd.md` (the engine), `architecture/retrieval.md` (the recall
  and every measurement).

## Knowledge graph

- **Nothing opens the store except `app.kg_store`** — no script, router or
  fixture.
- An unreadable store raises `StoreUnavailable`; never return an empty graph on
  a read failure.
- Expire edges, never delete them.
- `LLOYD_FACTS_ROOT` / `LLOYD_KG_DB` point a rebuild elsewhere; the extraction
  corpus is an allow-list in `pipeline_config.yaml`.
- `architecture/knowledge-graph.md`.
