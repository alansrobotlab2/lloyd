---
segment: architecture
tags: [architecture, lloyd, safety, authority, guards]
type: reference
status: implemented
date: 2026-10-29
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
| 1 | L0 classes and the block signal | `~/obsidian/lloyd/SOUL.md`, pinned by `app/prompt_surface.py` | What the model is told to refuse, and the exact signal it must emit. The prose is the vault identity file; the module holds what a trim may not drop (`GATE_HEADS`, `LOAD_BEARING`, the ceilings). Prompt, therefore the weakest layer — and it **is** a trigger for the tool-choice eval: `Gate.PROMPT_SURFACE_PATHS` (`scripts/automod/gate.py`) keys on `app/prompt_builder.py`, `app/prefetch.py`, this module and the three loaded vault files, so a diff that loosens a ceiling here gets scored by the `prompt_surface` rung (#1758, after a round that raised `GATE_STACK_CEILING` or dropped a `LOAD_BEARING` entry used to clear the whole ladder unobserved). Which modules qualify is not hand-kept either: `tests/test_prompt_surface_coverage.py` derives every module that assigns one of those invariant names at module level and refuses a round whose definition site is unlisted |
| 2 | Deterministic Bash denies | `app/harness/safety.py` | Is this command catastrophic (`check_bash_command` at `app/harness/safety.py:417`)? Installed as a default `PreToolUse` hook on every primary turn, Inner Voice on or off — it replaced the LLM-judgment lever that only ran when IV ran |
| 3 | Protected trees, three policies | `app/harness/protected_paths.py` | What may be **destroyed** (the structural shell check that refuses a wholesale delete), what may be **written** (the deny-set every MCP write lane now consults — `Write`/`Edit` through `_protected_path_refusal` (`agent_mcp/builtin_fs.py:222`), Bash through `check_bash_command`, and `vault_write` through `_protected_write_refusal` (`agent_mcp/vault.py:1708`) since #1757; `PROTECTED_WRITE_ROOTS` and `write_deny_reason`, both defined in `app/harness/protected_paths.py`), and — since #3524, the module's third policy — what an untrusted-input turn may **read** (`PROTECTED_READ_ROOTS` and `read_deny_reason`, armed per source by `app/harness/secret_read_deny.py`, and lifted by no grant, because no turn reading a transcript has a sanctioned reason to read a private key). `automod_vault_land` is the one exempt lane, by design: it validates the diff before it commits, which is what the refusal text offers a writer. One module, three questions, deliberately co-located so they cannot disagree about what is sacred |
| 4 | The read-only sandbox | `agent_mcp/_tool_sandbox.py` | Is this session one that must never change the machine? Bench and eval sessions, enforced in `agent_mcp/main.py::call_tool` — the one function every tool call from every caller passes through, not in a runner, because runner copies live in worktrees |
| 5 | The Bash substrate's kernel bind | `agent_mcp/_path_sandbox.py` | Can a shell child *write* into a protected tree at all? The Bash children of every session that is **not** bench/eval — the interactive session and every unattended source alike: `agent_mcp/builtin_bash.py` wraps each spawn in bwrap, `--ro-bind`ing `PROTECTED_SHELL_RO_ROOTS`, which is **derived** from `PROTECTED_WRITE_ROOTS` rather than copied from it (`protected_shell_ro_paths()` resolves the templates), so a tree added to the write deny set is read-only to the shell from the same edit and cannot be added to one and forgotten in the other, plus `~/obsidian/lloyd/USER.md` and `MEMORY.md`, which the shell binds read-only although the deny set deliberately does not cover them. Its off-state is total and it fails open **on purpose**: bwrap unable to build a namespace means the command runs **unsandboxed** with only the string guard left and a per-command `logger.warning`, because refusing Bash for every attended and worker session over a kernel hiccup is an outage of the whole agent (#2109). The structured witness is `fallback: true` on the `protected_path_sandbox` key at `/state`, and a green `/health` says nothing either way. The nameable alarm is `scripts/service_health_check.py`'s `check_path_sandbox`, which reads that key and reports red; the full description is `architecture/vault-protection.md` §2.1, not this row. |
| 6 | Unattended tool bans | `app/tool_bans.py` | Which tools may an unattended turn call at all? Two tuples (`WORKER_AUTOMOD_BAN`, `WORKER_GRANT_MINT_BAN`), and too many readers for one row to count: the shared worker turn path bakes both in (`workers/sources/_common.py`'s `_worker_run_options`, `:666` and `:672`), the capability pool folds them in as standing bans whatever a source declares (`_standing_bans`, `app/harness/capabilities.py:155`), the chat router arms them off the session's own platform (`build_turn_options`, `app/routers/turn_options.py:222`) — the automod ban for every non-user session except the one source whose job IS the loop (`AUTOMOD_DRIVER_SOURCES`, `app/routers/messages.py:590`), the grant ban wherever an authority scope is in force (`app/routers/messages.py`) — the review grader's deny list spreads the automod ban (`workers/sources/arch_review.py`) and so do three sibling sources' own lists (`autotriage`, `deep_research`, `youtube_digest`), and a `Task` child has to inherit it (`agent_mcp/builtin_task.py:378`). For a source that declared a capability set these two tuples are the **floor** and not the whole answer: they only ever remove names, and the capability-ceiling row below (`app/harness/capabilities.py`) narrows the reachable set further, so a name can be refused on an unattended turn without ever appearing in either tuple |
| 7 | The capability ceiling | `app/harness/capabilities.py` | Which tools can an unattended turn actually reach, once nobody has to remember to forbid one? Row 6 (`app/tool_bans.py`) is its **floor** and this is the ceiling: a source listed in `SOURCE_CAPABILITIES` is dispatched under `allowed = pool − declared`, so a name is unreachable because no one declared it rather than because someone listed it, and the refusal carries no deny-list entry to read back. The declared set is **derived, not hand-kept**: `SOURCE_CAPABILITIES` names what that source's own prompt or RESULT block can consume, checked against a measured call census over `~/lloyd-data/sessions` (the per-source dates and counts are in the module's own header), so the reachable set follows the job's declared output instead of a listkeeper's memory. `envelope_for` is the single compile point for every turn shape that declares one, and `loop._allow_list_hidden` is what turns that value into the catalog the turn is shown and the set every iteration dispatches under. The two postures are not the same and must not be merged. On a **direct worker turn** the compiled envelope is what the turn is dispatched under today (`_worker_run_options` in `workers/sources/_common.py`, and the same value carried across the loopback POST by `run_prompt_in_session` into `app/routers/turn_options.py`). The scheduled-task compile in `app/scheduled_ceiling.py` is **shadow** rather than enforced: `mode()` answers `shadow` unless an operator sets `LLOYD_SCHEDULED_CEILING_MODE=enforce`, so a shadow turn runs with `allowed_tools=None` exactly as it did before the ceiling existed and what it would have refused is written to the ledger that module's `ledger_path` names, rather than refused. Six sources declare one today (`session-distill`, `youtube-digest`, `deep-research`, `scheduled-task`, `autotriage`, `bench-mine`); a source that declares nothing gets `None` from `envelope_for`, which is row 6's deny floor and nothing else, never the whole pool |
| 8 | Scope-bound grants | `agent_mcp/builtin_grants.py` | Did a human mint authority for this specific scope, with an expiry? The interactive half is `grant_create`/`grant_list`/`grant_revoke`; the other half is `grants:` front matter on an autonomy task file, where editing the file *is* the grant — with one exception the code says out loud: a declared grant is never re-minted over a row that still covers its pair, a spent one included, so a used-up `quota:` is restored by `grant_revoke` plus a later `expires_at:` in the file, not by the edit alone (#2021; the sync warning and the denial both name that route). The protected-write escape in layer 3 (`allow_protected_writes`) is the same idea scoped to one call |
| 9 | Effect authority | `agent_mcp/_tool_effects.py` | Has this *effect* already happened (#544)? A timeout means unknown, not failure, so a retry is the dangerous path — which makes this an authority question, not a reliability one |
| 10 | Egress | `agent_mcp/egress.py` | Is this destination allowed at all (#628)? Telemetry first, then a default-deny policy. Before this the outbound path had no destination concept: `http_fetch`/`http_request` checked only whether a host was private, `browser.py` mirrored that check, and nothing recorded where a call went |
| 11 | The browser's own SSRF guard | `agent_mcp/browser.py` | May the browser be pointed at a private or loopback address? `browser_navigate` asks layer 10 first (`egress.aguard`), and then this floor, which is what the row is for: it re-checks where the navigation *landed* (`_enforce_landing` blanks the page when a redirect, a meta-refresh or a script put a private host in front of the model), and its loopback rule is stricter than the policy's — every resolved address must be loopback, and `browser_navigate` is given no loopback allowance at all. The two are why #628 had to touch both surfaces: the policy sees the destination that was asked for, this floor sees where the browser actually ended up |
| 12 | RPC policy | `app/harness/rpc_policy.py` | Which tools may `lloyd_rpc` call from inside a `Bash` command (P9)? Programmatic tool calling bypasses the model's turn loop, so it needs its own allow-list rather than inheriting the loop's |
| 13 | The desktop lease | `app/desktop_lease.py`, `app/routers/desktop.py` | Did Alan grant the lease, from Mission Control's Desktop tab? Capture stays read-only; acting requires the lease, and moving the mouse revokes it. The one surface where the human is in the loop physically ([[desktop]]) |
| 14 | The review session's deny list | `workers/sources/arch_review.py` | What may the doc reviewer edit, and what may it never touch — a deny list that must stay a superset of layer 6 |
| 15 | Sync and snapshot gates | `architecture/vault-protection.md` §2 | What may rewrite the vault's registration, and what happens to sync when a mass deletion is seen |

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
  `app.paths.VAULT_ROOT` is the constant the vault entry agrees with — a symbol, not a
  line pointer, because the pointer that sat here was stale and this doc copied it on.
  The vault lane is now *on* the set rather than outside it: `_protected_write_refusal`
  (`agent_mcp/vault.py:1708`) asks `write_deny_reason` ahead of every other gate in
  that lane since #1757, so all three lanes ask one predicate and membership in the
  constant is the whole answer. What is still spelled twice is the prompt's copy of
  the same rule: prose, checked by `tests/test_prompt_surface_guard.py`, not by the
  code.
- **Loopback.** mTLS was dropped on 2026-06-14 (`server.py`,
  [[mission-control]]), so no origin presents a certificate: the control is
  `ApiPeerGate`, which decides on the peer address and nothing else
  (`_is_trusted_peer`) and admits `/api/*` from a loopback peer or a peer inside
  `server.trusted_networks` (default Tailscale's `100.64.0.0/10`), checking the
  per-device allowlist only when a `x-client-fingerprint` arrives anyway.
  `chrome-extension`'s service worker reaches `http://127.0.0.1:8080` because it
  is loopback, not because a cert was waived for it ([[browser-side-panel]]). The
  egress policy and the browser guard are the pair that *was* spelled twice and no
  longer is: #2584 moved the resolving private-host answer into `agent_mcp/egress.py`
  and `browser.py` calls it, so what the browser still owns is its stricter loopback
  rule and its own `block_private_hosts` knob. What the split cost is on record in
  that file's own comment: while `agent_mcp/egress.py` resolved addresses and
  `agent_mcp/http_tools.py` prefix-matched the hostname *text*, `localhost.localdomain`
  was refused by `browser_navigate` and fetched by `http_fetch` — one rule, two
  answers, which is the drift this list is for.
- **Compaction thresholds.** The values live in `config.yaml`; `app/compaction.py`
  carries `setdefault` fallbacks, and the two already differ
  (`context-window.md` names the pair). Same shape, different subject.
- **The human-only marker.** Triage writes it into an acceptance, the dispatch
  skip reads it, the clause splitter reads it, and the ledger's human-only id set
  reads it — the defect that took #1698 was that the prompt renders the marker in
  markdown backticks and one of those readers did not tolerate them.

## Human-only paths

The loop's own scope limit: a fix that needs a path automod may never write is
not a round's work, whatever its clauses say. Runtime `data/**`, `.env*`,
`pytest.ini`, `.gitignore` and the frontend's build inputs are in that set — as
`DENIED_GLOBS` in `scripts/automod/spec.py`, enforced by `check_scope` on the
round's changed paths, which is what actually stops the bytes landing — along
with every path outside `ALLOWED_GLOBS` and the vault. `config.yaml` is denied
by path and judged by content: a comment-only edit lands, and so does a value
change outside `spec.CONFIG_DENIED_KEYS` / `CONFIG_LOOP_SOURCE_LEAVES`. The
marker for the set is `human-only:` at the head of a triage acceptance
(`HUMAN_ONLY_PREFIX`, `is_human_only` in `scripts/automod/backlog.py`), **and
since 2026-10-05 the marker is checked against the rail it cites**
(`_names_a_guard`): it guards only when the path it names is one `spec.classify`
calls denied or unlisted, lies outside the repo and the vault other than the
loop's own runtime data root (`_RUNTIME_DATA_ROOTS`, #3027 — no rail refuses bytes
there and a round cannot write them either, so a RUN over runtime data is a human
clause, not a guard), sits in a vault
subtree a vault round may not write (the autonomy tasks are the #724 dispatch rail's),
or the head names something physical. A marker on a PROTECTED path is not a
guard — protected paths land, with the drill — and neither is one on
`config.yaml` or one that names only a decision: 48 contracts in the fourteen
days before carried the marker, most on a path the gate would have landed, and
each parked its item for a decision no one needed (`architecture/automod.md`
§3.2k). The older half of the same rule: `human-only: not required` is triage
answering the prompt's "name the path" with "there is none", and since #1909 that
is read as an ordinary contract rather than a guard, because reading the prefix
alone parked landable items with a decision no one had asked for. What a round
actually reports at the end — its `human_paths` — is owed only when it names a
path too (`usable_path`, `human_paths_owed`): #1843 carried "apply \`None\`: None"
for a day off a finalizer that reported the word. The skip
that consumes the marker sits in the implement-pool selection, so a human-only item
is never handed an attempt; the same predicate decides whether clause splitting
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
- **The protected-path list itself.** `protected_roots()`, `protected_write_roots()`
  and `protected_read_roots()` are the lists, generated from the live tree (the
  shell's bind list, `protected_shell_ro_paths()`, is a projection of the write set).
  This doc does not copy them, for the reason `index.md` gives for not copying a
  fleet count: a second definition of a set is wrong within a week.
- **Whether the coverage is complete.** It is not provable from here, and an
  inventory that reads like a proof is the failure mode
  `architecture/arch-review.md` opens with. The honest test is a property, not a
  list: enumerate the write endpoints and ask which ones a guard sits on — that
  is how the unprotected surface was found in every instance on the record, the
  `vault_write` lane row 3 now names among them (#1757).
  A new tool, a new runner or a new write route should be checked against the
  fifteen rows above, and a row missing means this doc is the thing that is
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
- **2026-09-28 — #1759 (follow-up on the same row).** The Loopback bullet now says
  out loud what it decides on (the peer address, `_is_trusted_peer`) and its
  `server.py` citation was re-rooted to the lines holding the drop; the note
  recording that [[browser-side-panel]] still carried the old claim is gone, that
  doc being corrected by the same change. Both docs are now in
  `tests/test_stale_mtls_comment_claims.py`, whose ban is on the claim shape — an
  exemption for some peer, or a surviving requirement to produce a certificate —
  so the history above stays writable while the drift cannot come back.
- **2026-09-29 — symbol citations (#1760).** Row 3 cited `PROTECTED_WRITE_ROOTS` and
  `write_deny_reason` by line number inside `app/harness/protected_paths.py`, and the
  "Protected." bullet cited the vault-root agreement by line number inside
  `app/paths.py`. Both numbers were copied from that module's own comment, and the
  comment's number had already moved: the eleventh line of `app.paths` is prose about
  the gate's `HOME` symlink farm, nowhere near `VAULT_ROOT`. Both places now cite the
  symbol, and the module's comment states no size for the deny-set at all — the count
  it carried said four while the tuple held five, which is the drift
  `tests/test_protected_paths.py` now refuses against the module's own text.
- **2026-09-29 — row 1 rewritten (#1758).** The `stale` entry above corrected row 1
  to say `app/prompt_surface.py` is *not* a prompt-surface trigger, which was true
  at that tree and is the bug the item filed. `Gate.PROMPT_SURFACE_PATHS` now names
  the module, so the row states the opposite of that correction: a diff here is
  scored by the `prompt_surface` rung, and `tests/test_prompt_surface_coverage.py`
  refuses a definition site left off that list. Read the two entries together — a
  correction that is later superseded is still a dated record of the tree it
  measured.
- **2026-10-29 — `stale` (arch-review, whole-doc unit).** Row 3 was one policy short of
  the module it names: `app/harness/protected_paths.py` has carried a third since
  #3524 (`PROTECTED_READ_ROOTS` and `read_deny_reason`, armed per source by
  `app/harness/secret_read_deny.py`, lifted by no grant), so "two policies" and "two
  questions" became three. Two sentences disagreed with row 3 of the same doc and with
  the tree: the "Protected." bullet's claim that the vault lane never asks
  `write_deny_reason` (it has since #1757 — `_protected_write_refusal`), and the
  Loopback bullet's "both reason about private ranges, from separate code" (#2584 made
  that one definition, so the pair left this section's list rather than staying in it).
  Row 5's `:427`/`:433` were lines of `reconcile_fact_writes`, not of the ban loops
  (which are `_worker_run_options`' `:666`/`:672`), and its "four readers" had been
  outgrown by `_standing_bans` and three more sources' deny lists. Filed: the ladder
  has no row for the capability ceiling (#3560), the write deny-set's grant lift has no
  caller anywhere (#3563), and the Bash substrate sandbox — the one every non-bench
  session gets, with a `fallback: true` off-state — is not in the ladder either
  (#3564).
- **2026-10-30 — `implemented` (#3560).** The first of the three filed absences is filled:
  the capability ceiling is row 7, and the rows that were 7–14 became 8–15. Two numbers
  moved with it, because both are positions rather than facts — `browser_navigate`'s
  egress citation is now `layer 10`, and the completeness paragraph counts fifteen rows.
  The 2026-09-28 entry's "All thirteen surfaces" keeps its wording: it counts the tree at
  `f7cf29f4`, and a log rewritten to today's row count is a doc that can no longer say
  what its own numbers used to mean. The row states two postures and does not merge them
  — a direct worker turn is dispatched under the compiled envelope, the scheduled-task
  compile is **shadow** until an operator sets `LLOYD_SCHEDULED_CEILING_MODE=enforce`,
  which is #2670's owed entry 1 and has not been asked for. What the row could not fix in
  this round is the same silence in the sibling docs: `workers-jobs.md` documents the
  ceiling from its own side already, while `guard-coverage.md` (whose table's axis is what
  each guard *sees*, and a ceiling sees no content), `workers.md`, `autonomy-jobs.md` and
  `harness.md` are left as a scope call (#3560's owed notes).
