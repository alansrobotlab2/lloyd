---
segment: architecture
tags: [architecture, lloyd, safety, authority, guards]
type: reference
status: implemented
date: 2026-09-28
---

# Lloyd — write authority, across every surface

What stops an agent action, and in what order. Each area doc describes the guard
inside its own area, which is precisely the blind spot this doc is for: the
defect class is **a guard that lives on one of two write surfaces is not a
guard**, and a per-area review structurally cannot see it, because
`workers/sources/arch_review.py::doc_slugs` gives a reviewer one area's doc and
the diff for that area. Instances of it are on the record —
`architecture/vault-protection.md`, [[editing-safeguards]] and [[automod]] each
carry one — and none of them was found by a review.

So: enumerate the surfaces, name the question each answers, and state where the
same rule is spelled more than once. This is an inventory with a date, not a
proof of completeness — see the last section for why it cannot be one.

## The ladder, in the order an action meets it

An action from a primary turn meets these in roughly this order. The distinction
that matters is not strength but kind: the first is advice the model can ignore
under enough adversarial pressure, and everything under it is code that does not
ask.

| # | Surface | Module | Question it answers |
|---|---|---|---|
| 1 | L0 classes and the block signal | `lloyd/SOUL.md`, pinned by `app/prompt_surface.py` | What the model is told to refuse, and the exact signal it must emit. The prose is the vault identity file; the module holds what a trim may not drop (`GATE_HEADS`, `LOAD_BEARING`, the ceilings). Prompt, therefore the weakest layer — and it is **not** a trigger for the tool-choice eval: `Gate.PROMPT_SURFACE_PATHS` (`scripts/automod/gate.py:1303`) keys on `app/prompt_builder.py`, `app/prefetch.py` and the three loaded vault files, so a diff that loosens a ceiling here is scored by no behavioural rung |
| 2 | Deterministic Bash denies | `app/harness/safety.py` | Is this command catastrophic (`check_bash_command` at `app/harness/safety.py:372`)? Installed as a default `PreToolUse` hook on every primary turn, Inner Voice on or off — it replaced the LLM-judgment lever that only ran when IV ran |
| 3 | Protected trees, two policies | `app/harness/protected_paths.py` | What may be **destroyed** (the structural shell check that refuses a wholesale delete) and what may be **written** (the deny-set every MCP write lane now consults — `Write`/`Edit` through `agent_mcp/builtin_fs.py:238`, Bash through `check_bash_command`, and `vault_write` through `agent_mcp/vault.py:_protected_write_refusal` since #1757; `PROTECTED_WRITE_ROOTS` at `app/harness/protected_paths.py:148`, `write_deny_reason` at `app/harness/protected_paths.py:195`). `automod_vault_land` is the one exempt lane, by design: it validates the diff before it commits, which is what the refusal text offers a writer. One module, two questions, deliberately co-located so they cannot disagree about what is sacred |
| 4 | The read-only sandbox | `agent_mcp/_tool_sandbox.py` | Is this session one that must never change the machine? Bench and eval sessions, enforced in `agent_mcp/main.py::call_tool` — the one function every tool call from every caller passes through, not in a runner, because runner copies live in worktrees |
| 5 | Unattended tool bans | `app/tool_bans.py` | Which tools may an unattended turn call at all? Two tuples (`WORKER_AUTOMOD_BAN`, `WORKER_GRANT_MINT_BAN`), four readers: the shared worker turn path bakes both in (`workers/sources/_common.py:427`, `:433`), the chat router arms them off the session's own platform (`app/routers/turn_options.py:222-228`) — the automod ban for every non-user session except the one source whose job IS the loop (`AUTOMOD_DRIVER_SOURCES`, `app/routers/messages.py:589`), the grant ban wherever an authority scope is in force (`messages.py:564`) — the review grader's deny list spreads the automod ban (`workers/sources/arch_review.py:140`), and a `Task` child has to inherit it (`agent_mcp/builtin_task.py:378`) |
| 6 | Scope-bound grants | `agent_mcp/builtin_grants.py` | Did a human mint authority for this specific scope, with an expiry? The interactive half is `grant_create`/`grant_list`/`grant_revoke`; the other half is `grants:` front matter on an autonomy task file, where editing the file *is* the grant. The protected-write escape in layer 3 (`allow_protected_writes`) is the same idea scoped to one call |
| 7 | Effect authority | `agent_mcp/_tool_effects.py` | Has this *effect* already happened (#544)? A timeout means unknown, not failure, so a retry is the dangerous path — which makes this an authority question, not a reliability one |
| 8 | Egress | `agent_mcp/egress.py` | Is this destination allowed at all (#628)? Telemetry first, then a default-deny policy. Before this the outbound path had no destination concept: `http_fetch`/`http_request` checked only whether a host was private, `browser.py` mirrored that check, and nothing recorded where a call went |
| 9 | The browser's own SSRF guard | `agent_mcp/browser.py` | May the browser be pointed at a private or loopback address? A second, narrower question than layer 8, on a different surface — the reason #628 had to touch both |
| 10 | RPC policy | `app/harness/rpc_policy.py` | Which tools may `lloyd_rpc` call from inside a `Bash` command (P9)? Programmatic tool calling bypasses the model's turn loop, so it needs its own allow-list rather than inheriting the loop's |
| 11 | The desktop lease | `app/desktop_lease.py`, `app/routers/desktop.py` | Did Alan grant the lease, from Mission Control's Desktop tab? Capture stays read-only; acting requires the lease, and moving the mouse revokes it. The one surface where the human is in the loop physically ([[desktop]]) |
| 12 | The review session's deny list | `workers/sources/arch_review.py` | What may the doc reviewer edit, and what may it never touch — a deny list that must stay a superset of layer 5 |
| 13 | Sync and snapshot gates | `architecture/vault-protection.md` §2 | What may rewrite the vault's registration, and what happens to sync when a mass deletion is seen |

## Where one rule is spelled more than once

Every repetition here is a drift hazard, and the list of repetitions is itself
the useful part of the doc.

- **"Protected."** `PROTECTED_WRITE_ROOTS` declares membership in one constant
  and says so — `agent_mcp/builtin_fs.py` holds no path literal of its own,
  because "the four earlier spellings of 'protected' on this box" (L0 prose, the
  Bash regexes, the automod diff globs, and `protected_roots`) diverged. Its
  entries are **home-relative templates**, not `app.paths` constants, and the
  reason is the cross-cutting trap in one sentence: `LLOYD_HOME` is
  *code*-relative, so inside a worktree it names the worktree, and a deny-set
  that followed the code would protect the copy while leaving the live tree open.
  `app/paths.py:109` is where `VAULT_ROOT` agrees with it. What the constant does
  *not* cover is the vault lane's own answer to "is this path allowed" —
  `agent_mcp/vault.py:1608` decides by containment under `VAULT` alone and never
  asks `write_deny_reason`, so membership in the set does not mean the file is
  closed. The prompt's copy of the same rule is still prose, and prose is checked by
  `tests/test_prompt_surface_guard.py`, not by the code.
- **Loopback.** mTLS was dropped on 2026-06-14 (`server.py:73-75`,
  [[mission-control]]), so no origin presents a certificate: the control is
  `ApiPeerGate`, which admits `/api/*` from a loopback peer or a peer inside
  `server.trusted_networks` (default Tailscale's `100.64.0.0/10`) and checks the
  per-device allowlist only when a `x-client-fingerprint` arrives anyway.
  `chrome-extension`'s service worker reaches `http://127.0.0.1:8080` because it
  is loopback, not because a cert was waived for it ([[browser-side-panel]] still
  says the former — filed). The egress
  policy and the browser guard both reason about private ranges, from separate
  code.
- **Compaction thresholds.** The values live in `config.yaml`; `app/compaction.py`
  carries `setdefault` fallbacks, and the two already differ
  (`context-window.md` names the pair). Same shape, different subject.
- **The human-only marker.** Triage writes it into an acceptance, the dispatch
  skip reads it, the clause splitter reads it, and the ledger's human-only id set
  reads it — the defect that took #1698 was that the prompt renders the marker in
  markdown backticks and one of those readers did not tolerate them.

## Human-only paths

The loop's own scope limit: a fix that needs a path automod may never write is
not a round's work, whatever its clauses say. `config.yaml`, runtime `data/**`,
`.env*`, `pytest.ini`, `.gitignore` and the frontend's build inputs are in that
set — as `DENIED_GLOBS` in `scripts/automod/spec.py`, enforced by `check_scope`
on the round's changed paths, which is what actually stops the bytes landing —
and the marker for it is `human-only:` at the head of a triage acceptance
(`HUMAN_ONLY_PREFIX`, `is_human_only` in `scripts/automod/backlog.py`). The skip
that consumes it sits in the implement-pool selection, so a human-only item is
never handed an attempt; the same predicate decides whether clause splitting
letters a contract and whether the hold-on gate in `workers/sources/autotriage.py`
holds. When any of those readers disagrees, an item that only a person can fix
spends an unattended round discovering that.

## What this doc does not cover

- **Any area's own guard in detail.** The vault layers are
  `architecture/vault-protection.md`; the lease protocol is [[desktop]]; per-tool
  properties and the dispatch path are [[tools]]; the worktree/gate/scope checks
  are [[automod]]. This doc says which question each answers and where it is
  enforced, and stops there. Two docs describing one guard in full is how one
  gets corrected and the other does not.
- **Judgment.** Inner Voice's turn guards, the observer and its opt-in review are
  [[inner-voice]]. Layer 1 above is prompt-level by nature; nothing here claims
  the model's own refusal is load-bearing, which is why everything under it is
  code.
- **The protected-path list itself.** `protected_roots()` and
  `protected_write_roots()` are the list, generated from the live tree. This doc
  does not copy it, for the reason `index.md` gives for not copying a fleet
  count: a second definition of a set is wrong within a week.
- **Whether the coverage is complete.** It is not provable from here, and an
  inventory that reads like a proof is the failure mode
  `architecture/arch-review.md` opens with. The honest test is a property, not a
  list: enumerate the write endpoints and ask which ones a guard sits on — that
  is how the unprotected surface was found in every instance on the record, most
  recently the `vault_write` lane this pass filed.
  A new tool, a new runner or a new write route should be checked against the
  thirteen rows above, and a row missing means this doc is the thing that is
  stale, not that the surface is safe.

## Review log

- **2026-09-28 — created (#1699).** All thirteen surfaces read from this tree at
  `f7cf29f4`, with the enforcing function named where one exists. The list is
  ordered by when an action meets the layer, not by severity; where a layer is
  enforced in a shared choke point (`agent_mcp/main.py::call_tool`,
  `app/harness/safety.py`'s hook) that is stated, because a guard's position in
  the ladder is only real if every caller passes it.
- **2026-09-28 — `stale` (arch-review, whole-doc unit).** Four claims were wrong against
  the tree: the Loopback row's mTLS story (dropped 2026-06-14; the control is
  `ApiPeerGate`'s peer-address rule), row 1's claim that a diff to
  `app/prompt_surface.py` runs the tool-choice eval (it is not in
  `PROMPT_SURFACE_PATHS`), row 3's claim that the `vault_write` lane consults the
  write deny-set (it does not, by test-pinned design — filed), and the two
  `protected_paths.py` line numbers; row 5's reader count and the never-write
  path set's enforcement point (`spec.py::DENIED_GLOBS`) were made precise.
