---
segment: architecture
tags: [architecture, lloyd, tools, safety]
type: reference
status: implemented
date: 2026-09-13
---

# Editing safeguards: what stands between a turn and the tree

`~/lloyd` is production and a saved file is a deploy. The self-modification
loop has a worktree, a gate and a rollback ([[automod]]); an ordinary turn
that edits three files has the layers below. The first five live inside the
lloyd-mcp aggregator and switch under `harness:` in `config.yaml`; the last
three are backend-side and have no switch at all. Read the Switch column
before assuming a layer can be turned off — or that it is installed.

| Layer | Module | Switch | What it prevents |
|---|---|---|---|
| Read-before-edit and stale-file gates | `agent_mcp/builtin_fs.py` (`_read_records`, `_gate_check`) | `harness.edit_gates.enabled` | the edit that **succeeds** over another writer's change and silently reverts it |
| Diagnostics on the edit | `agent_mcp/_edit_diagnostics.py` | `harness.edit_diagnostics.python` | learning about a broken edit minutes later at the gate; reports only *introduced* pyflakes findings |
| Blast radius on the edit | same, via `agent_mcp/code_graph.py` | `harness.edit_diagnostics.blast_radius` | changing an interface without seeing its callers in other files |
| TypeScript diagnostics, later | `agent_mcp/_tsc_runner.py` | `harness.edit_diagnostics.typescript` | a `.tsx` edit that breaks the build; whole-project tsc is ~5 s, so it rides the drain queue |
| The per-turn change ledger | `agent_mcp/_change_ledger.py` | `harness.change_ledger.enabled` | an edit with no record and no undo; `GET /changes`, `POST /changes/revert` |
| The effect ledger | `agent_mcp/_tool_effects.py` | `harness.effect_ledger.enabled` | a requeued worker item re-landing every side effect its first attempt made |
| Destructive-Bash patterns | `app/harness/safety.py` | — no config read at all, so no kill switch | `rm -rf` and friends — on the turns that install it, which is not all of them (below) |
| The grant gate (#534) | `app/harness/policy.py`, `agent_mcp/builtin_grants.py` | armed per session platform | an unattended turn minting its own authority |
| Automod ban on worker turns | `workers/sources/_common.WORKER_AUTOMOD_BAN` | — | a worker driving the self-modification loop |

## The rules that are easy to get wrong

- **The Read's stat is taken before the file is opened**, so a write landing
  in between makes the later Edit refuse as stale — the safe direction.
- **`Write` over an existing file needs a Read too**; creating a new file
  needs nothing. A successful writer *refreshes* the record, so consecutive
  Edits to one file need one Read between them — `sed -i` from Bash does not,
  and the next Edit is refused until the file is re-Read. Intended.
- **Diagnostics are a delta, never absolute**, position-insensitive for
  pyflakes and a multiset for tsc — one shared normaliser,
  `app/lint_findings.py`, so the model is never told about a finding the gate
  will not mind. The tolerated count depends entirely on scope: `app/` alone
  is 34, the whole tree including `scripts/` is ~160, and `config.yaml`'s
  "~69" is neither. A post-image syntax error is always reported; a
  pre-image-only syntax error reports nothing.
- **A foreground Bash command gets the pyflakes delta too** (#695,
  `agent_mcp/_bash_edit_diagnostics.py`). A command carrying a write marker
  (`sed -i`, `patch`/`git apply`, a redirect, `.write_text(`, a formatter)
  has the `.py` files it names — inside a git work tree, outside `_pipeline/`
  and `sessions/` — read before the shell starts; afterwards each file whose
  bytes changed is handed to the same `python_block`. The byte comparison,
  not the extraction, decides what is reported, so a file the command only
  read produces nothing. Background and sandboxed Bash get no block, and
  the blast radius, tsc and the change ledger still see Edit/Write only.
- **The blast radius reports the module-level symbols the edit touched**
  (`_edit_diagnostics._touched_symbols`, both images, Python only, and a
  created file has no pre-edit callers so it reports nothing) against the
  graph of the tree *before* the edit, drops anything over `FANOUT_CEILING`
  (20) call sites, and runs in a thread the edit abandons after
  `RAIL_BUDGET_S` (90 ms).
  Its switch is the one key here with no line in `config.yaml` — absent means
  the default, which is on; adding the line is a human-only edit.
- **The change ledger's first writer per realpath per turn wins**, so a turn
  that edits a file ten times reverts to where it came in. Revert refuses a
  file whose sha moved since. A subagent's writes land on the parent's turn.
  Only `builtin_fs` writes are recorded — a Bash `sed -i` is invisible to it.
  `turn_id` is what switches the whole thing on, so until 2026-09-10 every
  background run left no pre-images and no undo — the gap that made an
  autonomy task both the prime suspect in the vault wipe and impossible to
  clear. Both background paths mint one now (`autonomy.py`,
  `workers/sources/_common.py`); a bare `run_query` caller still has none.
  A failed autonomy run's record names the scope and the count read from that
  turn's index, and `autonomy.revert_run_writes(session_id, run_id)` is the
  in-process undo that takes those two ids; the failure path never calls it, so
  a dead run's partial writes stay on disk until someone decides otherwise.
- **The effect ledger writes `unknown` before dispatch** and settles after;
  a second identical call in scope replays, an `unknown` one is refused.
  Idempotent tools are not ledgered at all — `Write`, `vault_write`, every
  delete and setter, because a replay there could only ever be staler than
  the real call. `Edit` is excluded for the opposite reason: probed A→B,
  B→A, A→B in one scope, the third call was answered from the ledger and the
  file stayed at **A** — a silent non-edit delivered by the guard meant to
  stop duplicates. It is `REPEAT_EXPECTED` instead, where its own
  `old_string` match is the idempotency check the model can see. The scope is
  `item:<source>:<id>`, carried across the loopback POST in the payload and
  honoured only for a non-user platform. A chat turn is handed
  `turn:<session>:<turn>` instead (#767): recorded and counted, never
  replayed or refused, and left out of the suppression counter.
  `_tool_effects.shadow_repeats()` reads the within-turn and across-turn
  repeat counts the replay-vs-re-fire decision (Alan's) is waiting on.
- **The grant gate is not vault protection.** Tier 1 is everything
  unclassified and returns before the store is even opened, so `Bash`,
  `Edit`, `Write`, `backlog_write_task` and `vault_write` all pass. What it
  gates is the durable-external surface — email, calendar, contacts, tasks.
- **And `safety.py` is not installed everywhere.** It is a PreToolUse hook,
  so it exists only where a caller built a registry with it:
  `app/routers/messages.py` — three sites (`:1907` streaming, `:2045` ambient,
  `:2175` sync), which covers the session-backed workers too, since those post
  to the chat endpoint — and `builtin_task.py` for subagents. Two paths build
  options and get *no* registry at all: **voice**
  (`app/routers/voice.py:123` passes no `hooks=`, and `_run_turn` fills in
  every other option but never that one, so an IV-off voice turn has neither
  this nor the grant gate) and the bare `HookRegistry` that
  `attach_observer_for_turn` creates when it finds none. The two *direct*
  paths build their own registries and install the grant gate only, so an
  `autonomy.run_task` or `run_prompt_on_primary` turn has neither this nor
  anything else between it and `rm -rf`; `builtin_bash.py` deliberately does
  not self-check, and says so in its docstring, which names this file and
  `agent_mcp/main.call_tool`'s dispatch check as the gate. Voice is the same shape as the one #534 closed for the
  grant gate, one layer down and still open.
- **A gate that raises denies** (review 2026-09-24, D5). `safety.py`, the
  outbound content gate and the grant gate register with
  `add_pre_tool_use(..., fail_closed=True)`: if the callback raises, the call
  is denied with `gate <name> raised <Type>: <msg>`, and that deny beats a
  deliver held from earlier in the walk. Before this every raise was a pass,
  so an import failure inside `_safety_pretool_cb` let every Bash through.
  Observers and the skill deliverer stay fail-open. Every raise, either kind,
  writes one `harness.hook_raised` event (`hook`, `tool`, `error`,
  `fail_closed`, `tool_use_id`). The aggregator's own dispatch check is
  unaffected.

- **A `Task` child inherits its parent's authority, not a fresh one** (review
  2026-09-24, D4). The child's registry is built in the aggregator, so the
  grant gate reaches it only through `_meta`: `lloyd/grant_scope` re-arms
  `install_policy_hook` in the child, and `lloyd/disallowed_tools` carries the
  parent's live deny list (plan mode, worker bans, `grant_create`), unioned
  with `app.tool_bans.WORKER_AUTOMOD_BAN` in both spellings. A chat parent
  sends no scope and its child stays ungated, like the chat route. Before
  this a worker's subagent could call every tier-2 tool unasked.
  `tests/test_task_subagent_authority.py`.

## The long versions (moved from CLAUDE.md, 2026-09-25)

CLAUDE.md keeps one line per layer and points here.

### Read-before-edit and stale-file gates

`Edit` used to be exact-match against whatever is on disk right now, with no
record of whether this session had ever looked at the file. The failure worth
designing against is not the edit that fails — `old_string not found`
announces itself — it is the edit that **succeeds**: the model Reads a file,
something else rewrites it, the `old_string` still matches, and the edit
silently reverts the other writer. Nothing in the transcript, the tool result
or the logs says so.

`builtin_fs._read_records` is `session_id -> realpath -> (mtime_ns, size)`,
written by a successful Read, Write or Edit and consulted by `_gate_check`.

- **The Read's stat is taken *before* the file is opened.** A write landing in
  between then makes the recorded key older than the bytes returned, so a
  later Edit is refused as stale — the safe direction. Stat'ing afterwards
  would record the other writer's key and wave that Edit through.
- **Keys are `os.path.realpath`**, so a symlink and its target are one file.
- **A writer refreshes the record**, so consecutive edits to one file need one
  Read; `sed -i` from Bash in between does not, and the next Edit is refused
  until it is re-Read. That is the intended cost.
- **Write over an existing file needs a Read too**, which is the rule most
  likely to surprise; creating a new file (including through a dangling
  symlink) needs nothing. Both live behind `harness.edit_gates.enabled`.
- **No bound session means no gate.** Unit tests and legacy callers dispatch
  straight into the handlers with no aggregator context, and refusing them
  protects nothing while breaking `tests/test_mcp_layer.py`.
- Records survive the kill switch: `enabled: false` stops refusing but keeps
  accruing, so flipping it back works immediately rather than after everyone
  re-Reads everything.
- Containers are bounded twice (256 sessions, 2000 paths each, LRU) and
  guarded by a `threading.Lock` — the handlers run on worker threads via
  `asyncio.to_thread`, so two sessions genuinely race.

Editing a binary file is a normal error rather than a `UnicodeDecodeError`
escaping the aggregator as an MCP exception with no path in it.

`app/lint_findings.py` holds the pyflakes/tsc normalisers that `gate.py` used
to own privately. The aggregator cannot import `scripts.automod.gate` — that
pulls the whole self-modification package behind every tool call — and two
private definitions of "is this finding new?" is exactly how the gate and the
model would come to disagree about the same edit.

### Diagnostics on the edit, not on the gate

A successful `Edit`/`Write` on a `.py` file carries a `<diagnostics>` block
listing what it *introduced*. Before this, the model learned about a broken
edit at the automod gate — minutes later, with ten more edits built on top.
`agent_mcp/_edit_diagnostics.py` runs pyflakes in-process against the
pre-image and post-image and reports the difference.

Delta, and the three rules the delta needs, all inherited from the gate:

- **Never absolute.** The tree carries tolerated pyflakes findings (34 in
  `app/`, ~160 whole-tree; the "~69" CLAUDE.md quoted named neither scope —
  see the 2026-09-13 review below). Reporting them on every edit trains the
  model to skip the block, which is worse than not having one.
- **Position is not identity.** `lint_findings.normalize_pyflakes_line` drops
  line and column, so inserting a line at the top does not report the whole
  file as new. Display still uses post-image positions — that is what the
  model can act on.
- **A multiset, not a set.** A second normalised-identical finding is a second
  finding, and `Counter(post) - Counter(pre)` keeps it visible.

Two asymmetries in pyflakes decide the rest. `check()` calls `syntaxError` and
returns *without* running the checker, so flakes and a syntax error are never
both present; a post-image syntax error is therefore always reported (tagged
`pre_existing="true"` when the pre-image was broken too). And when only the
*pre*-image failed to parse there is no flake baseline at all — reporting the
post-image's findings there would dump every tolerated finding in the file
onto the edit that just *fixed* the syntax, so that case reports nothing.

The block is appended only to a **success** string. `_shared.text_result` sets
`isError` by sniffing a leading JSON object with an `"error"` key, so
appending to an error payload would break the JSON and make a lint finding
read as a failed edit. Nothing in this path may raise: `_append_diagnostics`
swallows everything, because an edit with no diagnostics beats an edit that
failed because the linter did.

**The reach of all of that is one file, because pyflakes is.** So a second
block, `<blast_radius>`, carries what a per-file linter structurally cannot:
when an edit changes a module-level interface — a `def`/`class`/assignment
name, a signature, a base, an assigned value, or a `return`/`yield`
expression — the edit result also names the inbound callers living in *other*
files (`symbol → path.py:88`, advisory, never an error). The store is
`agent_mcp/code_graph.py`, the one the graph tools read; the point is that
this half is **passive**, because the graph was always available and only
ever fired when the model remembered to ask. Three rules keep it worth
reading, all the same shape as the three above: only an *interface* change
fires (a local variable adds nothing); only the **pre**-image's symbols are
queried — the graph describes the tree before the edit, which is the right
source for "who calls this today", so a stale graph is correct here, a name
that exists only afterwards silently does not resolve, and a rename is caught
from the old side; and a symbol above `FANOUT_CEILING` call sites is dropped
rather than listed, because a helper with 62 callers produces a wall of text
that gets skipped. Measured on this tree: `kg_store.store` (62 sites) and
`RunOptions` (70) are both suppressed, which is the rail working, not failing.

`_append_diagnostics` runs off the event loop via `asyncio.to_thread` (#726),
and its graph half runs in a daemon thread it joins for `RAIL_BUDGET_S`
(90 ms) and then abandons. Abandoning is not waste: the thread finishes the
210 ms load and fills a per-root cache, so the next edit in that tree gets its
answer. `RAIL_MAX_SOURCE_BYTES` is 120 KB because the fingerprint pass costs
~0.28 ms/KB and would break the latency promise on its own above that;
pyflakes still runs on bigger files.

`harness.edit_diagnostics.python: false` removes the pyflakes block and
`.blast_radius: false` the cross-file one — separate switches, separate
stores. Neither key is in `config.yaml` yet, and adding one is human-only.

### TypeScript diagnostics arrive later, on purpose

pyflakes on one file is milliseconds, so Python diagnostics ride back on the
Edit. tsc cannot: `web/tsconfig.json` includes only `src` and there is no
per-file mode that resolves imports, so it is whole-project and ~5 s. Paying
that on every `.tsx` edit taxes the common case to serve the rare one.

`agent_mcp/_tsc_runner.py` debounces 1.5 s, runs one type-check at a time
process-wide, and delivers the result on a later iteration through the same
drain the background-Bash tool uses. The Edit result says a check was queued
— a model told nothing assumes nothing is coming and either re-checks by hand
or moves on.

- **Cold start seeds, it does not report.** With no baseline the first run
  attributes every pre-existing error in the tree to whoever edited first, so
  `main.lifespan` schedules `warm_baseline()` a few seconds after boot.
  Without that the first `.tsx` edit of every restart is the one with the
  wrong answer. The baseline persists to `_pipeline/tsc/baseline.json` (under
  the data root, [[data-home]]).
- **A whole-project run becomes a per-session answer** by keeping the
  previous run's per-file counts and reporting `run[f] - baseline[f]` only for
  the files *that session* edited. Somebody else's breakage is somebody else's
  news.
- **The root is derived from the edited path**, not from `LLOYD_HOME`: an
  automod round edits under `~/lloyd-work/…` and the live tree's tsc would say
  nothing about it. No `web/node_modules/.bin/tsc` under that root means no
  run *and no hint* — promising a check that cannot happen is worse than
  silence.
- **Delivery is a `DiagnosticsRecord`, not a `TaskRecord`.** That dataclass
  carries a `process` and a `log_fd`, and `format_notification`, `_task_row`
  and `list_active` are all specific to a background bash child. It rides the
  per-session drain queue and never enters `_records`, so `/state`'s task rows
  are untouched.
- **A `task:*` session's result goes to the parent.** Nothing reads a
  subagent's drain queue once its Task has returned
  (`_subagent_registry.parent_scope`).
- A failed or timed-out run is still reported, for the same reason the hint
  exists.

`/state.tsc` shows the last run and what is pending.

### The per-turn change ledger

The self-modification loop has a worktree, a gate and an automatic rollback;
an ordinary chat turn that edits three files had none of that, no line saying
which three, and no undo.

`agent_mcp/_change_ledger.py` writes one record per (session, turn) under
`sessions/<sid>.changes/<turn_id>/` (under the data root) — an `index.json`
naming every file the turn wrote and a `<sha1(realpath)>.pre` beside it. The
layout follows `tool_result_spill.py` for the same reason: per-turn side data
that has to survive an aggregator restart and be findable from a session id
alone.

- **First writer per realpath per turn wins.** The pre-image is what the file
  looked like when *this turn* first touched it, so a turn that edits one file
  ten times reverts to where it came in. `begin` loads the on-disk index on a
  miss, so an aggregator restart mid-turn does not restart the rule.
- **Revert refuses a file that moved since.** If the current sha does not
  match what the turn wrote, something else has written and restoring the
  pre-image would destroy *that* — the exact damage this exists to prevent.
  Refused by name and reported, never silently skipped. Same for a `create`
  whose file was edited afterwards.
- **`turn_id` and `call_id` ride in `_meta`**, like the session id and the
  caption, because `args` is what reaches each tool's handler and is what the
  repetition guard hashes. A caller with no `turn_id` gets no ledger, which
  used to mean every background run: since 2026-09-10 both background paths
  mint one, so an unattended run's writes leave pre-images and are revertable
  ([[background-runs]]). A bare `run_query` caller still has none.
- **A subagent's writes land on the parent's turn.** `ledger.scope()`
  redirects a `task:*` session through `_subagent_registry.parent_scope`; the
  entry carries `via_session` so the row can still say it came from a Task.
  The read-before-edit gate keeps using the raw `task:*` session — the
  subagent must still Read what it edits — and only attribution moves.
- **`prune` only ever walks `*.changes/`.** The sessions directory also holds
  the transcripts and the spilled tool results, and a prune that reached those
  would be deleting the record it exists to protect.
- Known imperfection, documented: a create made *through* a dangling symlink
  reverts by unlinking the symlink path, not the target it pointed at.

`GET /changes?session=&turn=` and `POST /changes/revert` on the aggregator;
`/state.changes` for the dashboard. `harness.change_ledger.enabled: false`
turns it off.

### The effect ledger (#544): exactly-once effect, not exactly-once scheduling

A worker item cancelled at `max_duration_seconds` is requeued and re-runs its
whole turn, so every side-effecting call the first attempt landed lands again.
`agent_mcp/_tool_effects.py` keeps one row per (tool, canonical arguments,
**queue item**) in `workers.db`, written `unknown` **before** dispatch — the
only record that survives a cancel — and settled `ok`/`error` after. A second
identical call in scope replays the stored result; an `unknown` one is refused
with instructions to read the state back; the ledger fails **open** if the
database is unusable. `harness.effect_ledger.enabled` is the switch. The scope
is `item:<source>:<id>` (`workers/pool.py::effect_scope_for`) and never the
grant scope, which is stable across every future run of a task and would
suppress a legitimate second effect forever.

The first cut landed 2026-09-10 through the unattended loop with every gate
rung green, and a review the same day found four things the tests had not:

- **The scope has to cross two process seams, and it crossed neither.** The
  pool binds `policy.current_effect_scope` in its own task; every
  session-backed source (autocode, autotriage, youtube-digest, deep-research)
  POSTs to `/api/message/stream` and the backend runs the turn in another
  task, so the contextvar read empty — 25 items, 0 rows, in the first sixteen
  hours. It travels in the payload now (`_common.run_prompt_in_session` →
  `messages._effect_scope_for` → `RunOptions.effect_scope`, honoured only for
  `NON_USER_PLATFORMS`), and `loop.py` prefers the option over the
  contextvar. A `Task` subagent re-enters `main.py::call_tool` over loopback
  `/mcp` from a fresh ASGI task; `call_tool` binds the incoming scope around
  the dispatch so the nested loop stamps it on. `e3ff863` had fixed exactly
  this hop for `grant_scope` eight hours after the ledger landed.
- **Classification must honour `IDEMPOTENT`.** `annotations.side_effecting`
  consulted only `READ_ONLY` and `REPEAT_EXPECTED`, leaving 20 idempotent
  tools ledgered for no protective value — `Write` among them. A setter or a
  delete repeated adds nothing, so a replay can only be staler than the call.
- **`Edit` is never replayed.** Probed A→B, B→A, A→B in one scope: the third
  call was answered "Edited (1 replacement)" and the file stayed at A — the
  silent revert the read-before-edit gate exists to prevent, delivered by the
  guard meant to stop duplicates. `Edit` is `REPEAT_EXPECTED`; its own
  `old_string` match is the idempotency check the model can see.
- **A test that asserts `x == [] or True` pins nothing**, and the round that
  wrote it declared its acceptance `deferred` to an empty list. Both are why
  the automod gate grew a review rung ([[automod]] §4.5).

`unknown` rows are pruned on a longer clock than settled ones
(`unknown_retention_days`): they are the only record an effect may have
landed, and pruning them re-arms the duplicate. `tests/test_tool_effects.py`
pins all of it, including the probe.

## The code graph

`agent_mcp/code_graph.py` answers "who calls this" and "what breaks if I
change it" from graphify's AST extraction of a tree
(`<root>/graphify-out/graph.json`, gitignored, ~15 s to build). Six tools:
`graph_explain`, `graph_affected`, `graph_path`, `graph_hubs`,
`graph_status`, `graph_refresh`. `root` is explicit (`LLOYD_HOME`, an `SM_…`
round id, or a path) and never inferred; staleness is commit mismatch *or* an
uncommitted source newer than the graph. The second rule is not a nicety:
inside an automod round HEAD does not move while the model edits, so a
commit-only rule would call the graph fresh for exactly the window it is most
wrong in. Rebuilds triggered that way are debounced by
`min_refresh_interval_s` (30 s) and a debounced query still answers, saying
`STALE`. It is blind across process seams — no edge from
`run_prompt_in_session` to `run_query`, none from `run_query` into a tool
handler — so keep Grep for string keys and routes.

## Tests

`tests/test_edit_gates.py`, `tests/test_change_ledger.py`,
`tests/test_bash_edit_diagnostics.py`, `tests/test_harness_safety_docstring_pointers.py`,
`tests/test_tool_effects.py`, `tests/test_edit_diagnostics.py`,
`tests/test_tsc_runner.py`, `tests/test_lint_findings.py`,
`tests/test_code_graph.py`, `tests/test_code_graph_doc_claims.py`,
`tests/test_mcp_layer.py` — which is why the gates skip an unbound session:
a gate that fired there would fail that file rather than protect anything —
and `tests/test_automod_hardening.py::test_worker_turns_cannot_drive_the_loop`.

## Related

[[harness]], [[automod]], [[tools]], [[background-runs]].

## Review log

- 2026-09-13 — **stale.** The safety-hook coverage claim was wrong in the
  dangerous direction: voice turns build a `RunOptions` with no `hooks=` and
  `_run_turn` never fills one, so `safety.py` is absent there too (#869);
  `builtin_bash.py`'s docstring still names the deleted
  `app/inner_voice/heuristics.py` (#695); "~69 tolerated pyflakes findings" is
  `app/`-scoped-34 or whole-tree-~160 depending on scope, so the number now
  names its scope; the symbol-diff helper is `_touched_symbols`, not
  `diff_interfaces`; `config.yaml`'s "workers have no turn id and no reader"
  is half wrong (#963); `safety.py` has no config key, hence no kill switch.
  Code-graph, effect-ledger and read-stat sections checked claim-by-claim and
  held.
