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
that edits three files has the layers below, all inside the lloyd-mcp
aggregator, all switchable under `harness:` in `config.yaml`.

| Layer | Module | Switch | What it prevents |
|---|---|---|---|
| Read-before-edit and stale-file gates | `agent_mcp/builtin_fs.py` (`_read_records`, `_gate_check`) | `harness.edit_gates.enabled` | the edit that **succeeds** over another writer's change and silently reverts it |
| Diagnostics on the edit | `agent_mcp/_edit_diagnostics.py` | `harness.edit_diagnostics.python` | learning about a broken edit minutes later at the gate; reports only *introduced* pyflakes findings |
| Blast radius on the edit | same, via `agent_mcp/code_graph.py` | `harness.edit_diagnostics.blast_radius` | changing an interface without seeing its callers in other files |
| TypeScript diagnostics, later | `agent_mcp/_tsc_runner.py` | — | a `.tsx` edit that breaks the build; whole-project tsc is ~5 s, so it rides the drain queue |
| The per-turn change ledger | `agent_mcp/_change_ledger.py` | `harness.change_ledger.enabled` | an edit with no record and no undo; `GET /changes`, `POST /changes/revert` |
| The effect ledger | `agent_mcp/_tool_effects.py` | `harness.effect_ledger.enabled` | a requeued worker item re-landing every side effect its first attempt made |
| Destructive-Bash patterns | `app/harness/safety.py` | — | `rm -rf` and friends from any turn |
| The grant gate (#534) | `app/harness/policy.py`, `agent_mcp/builtin_grants.py` | by session platform | an unattended turn minting its own authority |
| Automod ban on worker turns | `workers/sources/_common.WORKER_AUTOMOD_BAN` | — | a worker driving the self-modification loop |

## The rules that are easy to get wrong

- **The Read's stat is taken before the file is opened**, so a write landing
  in between makes the later Edit refuse as stale — the safe direction.
- **`Write` over an existing file needs a Read too**; creating a new file
  needs nothing. `sed -i` from Bash invalidates the record and the next Edit
  is refused until re-Read. Intended.
- **Diagnostics are a delta, never absolute** (~69 tolerated findings in the
  tree), position-insensitive, and a multiset. A post-image syntax error is
  always reported; a pre-image-only syntax error reports nothing.
- **The blast radius queries the pre-image's symbols** against the graph of
  the tree *before* the edit, drops anything over `FANOUT_CEILING` call
  sites, and runs in a thread the edit abandons after 90 ms.
- **The change ledger's first writer per realpath per turn wins**, so a turn
  that edits a file ten times reverts to where it came in. Revert refuses a
  file whose sha moved since. A subagent's writes land on the parent's turn.
  Only `builtin_fs` writes are recorded — a Bash `sed -i` is invisible to it.
- **The effect ledger writes `unknown` before dispatch** and settles after;
  a second identical call in scope replays, an `unknown` one is refused.
  `Edit` is never replayed (it reverted A→B→A→B once); idempotent tools are
  not ledgered at all. The scope is `item:<source>:<id>`, carried across the
  loopback POST in the payload.
- **The grant gate is not vault protection.** Tier 1 is everything
  unclassified, so `Bash`, `Edit`, `Write`, `vault_write` all pass. Only
  `safety.py` stands between an unattended turn and `rm -rf`.

## The code graph

`agent_mcp/code_graph.py` answers "who calls this" and "what breaks if I
change it" from graphify's AST extraction of a tree
(`<root>/graphify-out/graph.json`, gitignored, ~15 s to build). Six tools:
`graph_explain`, `graph_affected`, `graph_path`, `graph_hubs`,
`graph_status`, `graph_refresh`. `root` is explicit (`LLOYD_HOME`, an `SM_…`
round id, or a path) and never inferred; staleness is commit mismatch *or* an
uncommitted source newer than the graph. It is blind across process seams —
no edge from `run_prompt_in_session` to `run_query`, none from `run_query`
into a tool handler — so keep Grep for string keys and routes.

## Tests

`tests/test_tool_overrides.py`, `tests/test_tool_effects.py`,
`tests/test_mcp_layer.py`, `tests/test_code_graph_doc_claims.py`,
`tests/test_automod_hardening.py::test_worker_turns_cannot_drive_the_loop`.

## Related

[[harness]], [[automod]], [[tools]], [[background-runs]].
