---
segment: architecture
tags: [architecture, lloyd, tools, safety]
type: reference
status: implemented
date: 2026-09-11
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
| Destructive-Bash patterns | `app/harness/safety.py` | — | `rm -rf` and friends — on the turns that install it, which is not all of them (below) |
| The grant gate (#534) | `app/harness/policy.py`, `agent_mcp/builtin_grants.py` | by session platform | an unattended turn minting its own authority |
| Automod ban on worker turns | `workers/sources/_common.WORKER_AUTOMOD_BAN` | — | a worker driving the self-modification loop |

## The rules that are easy to get wrong

- **The Read's stat is taken before the file is opened**, so a write landing
  in between makes the later Edit refuse as stale — the safe direction.
- **`Write` over an existing file needs a Read too**; creating a new file
  needs nothing. A successful writer *refreshes* the record, so consecutive
  Edits to one file need one Read between them — `sed -i` from Bash does not,
  and the next Edit is refused until the file is re-Read. Intended.
- **Diagnostics are a delta, never absolute** (~69 tolerated findings in the
  tree), position-insensitive, and a multiset. A post-image syntax error is
  always reported; a pre-image-only syntax error reports nothing.
- **The blast radius queries the pre-image's symbols** against the graph of
  the tree *before* the edit, drops anything over `FANOUT_CEILING` (20) call
  sites, and runs in a thread the edit abandons after `RAIL_BUDGET_S` (90 ms).
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
  honoured only for a non-user platform.
- **The grant gate is not vault protection.** Tier 1 is everything
  unclassified and returns before the store is even opened, so `Bash`,
  `Edit`, `Write`, `backlog_write_task` and `vault_write` all pass. What it
  gates is the durable-external surface — email, calendar, contacts, tasks.
- **And `safety.py` is not installed everywhere.** It is a PreToolUse hook,
  so it exists only where a caller built a registry with it:
  `app/routers/messages.py` — which covers every chat turn and every
  session-backed worker, since those post to the chat endpoint — and
  `builtin_task.py` for subagents. The two *direct* paths build their own
  registries and install the grant gate only, so an `autonomy.run_task` or
  `run_prompt_on_primary` turn has neither this nor anything else between it
  and `rm -rf`; `builtin_bash.py` deliberately does not self-check, and says
  so in its docstring. That asymmetry is the same shape as the one #534
  closed for the grant gate, one layer down and still open.

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
`tests/test_tool_effects.py`, `tests/test_edit_diagnostics.py`,
`tests/test_tsc_runner.py`, `tests/test_lint_findings.py`,
`tests/test_code_graph.py`, `tests/test_code_graph_doc_claims.py`,
`tests/test_mcp_layer.py` — which is why the gates skip an unbound session:
a gate that fired there would fail that file rather than protect anything —
and `tests/test_automod_hardening.py::test_worker_turns_cannot_drive_the_loop`.

## Related

[[harness]], [[automod]], [[tools]], [[background-runs]].
