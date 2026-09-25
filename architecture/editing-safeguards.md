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
