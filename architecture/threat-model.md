---
segment: architecture
tags: [architecture, lloyd, security, threat-model, stride, atlas, trust-boundaries]
type: reference
status: implemented
date: 2026-11-11
---

# Lloyd — threat model: the boundaries, the threats over them, and what they predicted

Backlog #2563 asked for Silverman's procedure run against this repo instead of a
generic app: scope the system, name the seams, enumerate threats as *capabilities*
over those seams, attach the control in this codebase that answers each one, and
publish the threats with no control as the open set. [[authority-surfaces]]
already inventories the guards in the order an action meets them and
[[guard-coverage]] publishes the union of what none of them sees. Neither states
the other half: **what is being attacked**. Without that, the rungs of the ladder
in [[authority-surfaces]] cannot be ranked, a security backlog item cannot be shown
to answer a
threat, and the next argument about whether a rail is too broad (#2269's
`Bash` narrowing, which shipped and was reverted) is settled by intuition plus a
broken canary again.

This page states no rung count, and does not need one. #4046 is the record of a
count that rotted twice here, and the ladder gains a row every time a guard is
filed, so the number is derived from that table by `tests/test_threat_model_doc.py`
on every suite run — which is also where the claim that every rung has a seam over
it is now checked.

Shape of what follows: a boundary map derived from chokepoints in the tree, a
threat table over it, an open set, and a retrospective that scores the model
against incidents this box already wrote up. Pinned by
`tests/test_threat_model_doc.py` (row shape, the miss vocabulary, the arithmetic)
and by `tests/test_architecture_coverage_doc_claims.py`, whose path node resolves
every cited path and line anchor here — and since #2582 that node reads every
top-level `architecture/*.md` rather than a named few, asking each citation
whether it exists under the checkout, the vault, the data root or the account
home, so a cell pointing at `~/lloyd-data/...` is graded rather than waved
through. What it still cannot do is tell a live pointer from a sentence that
names an absent path on purpose: that is what the exemption list in that file is,
per doc and per citation, with a reason and a node refusing an entry no doc still
makes. A line anchor is no longer
checked only to be shorter than its file: since #2583 the node asks that the line
fall inside a definition the file actually defines, and since #3047 that a citation
whose sentence names a symbol lands inside *that* symbol's span. This doc's table
cells name a path and never an identifier, so containment alone decides them and
they sit outside the `NAMED_SYMBOL_RAIL_DOCS` opt-in (#3580) — which is exactly how
the two cells that once pointed at `main()` and at a grant function (#2583) could
drift again without reddening anything, and two of them have (#4045).

## How this model was built, and the one way it lies

Boundaries came from enumerating the chokepoints the code actually funnels
through — the one function every tool call passes, the one constant the deny-set
lives in, the socket a call has to reach — not from a checklist. Threats came from
walking each seam asking what it lets something through, in STRIDE terms, plus the
classes ATLAS adds for an agent that acts: indirect prompt injection, poisoning,
excessive agency, skills-based persistence, misinformation-to-action. Each threat
is a **capability** ("a turn can destroy a protected tree in one command"), never
an occurrence; the test refuses a row that narrates a date.

**The inflation this section exists to warn about.** This model was written by a
reader of `knowledge/software/`, on a tree whose guards are overwhelmingly scars —
`agent_mcp/vault.py` consults the deny-set because a lane once did not,
`app/harness/sync_registration.py` exists because Sync pushed five thousand
deletes. A model that names a control the incident produced, then counts the
incident as predicted, has scored its own authorship. So the retrospective carries
a per-incident `Control predates the damage?` column, and **two numbers**: raw
coverage, and independent coverage counted only where the control existed before
the damage and fired. Read the second one. Where a boundary exists in the map only
because incidents in this sitting forced it, the map row says so and those
incidents stay counted as misses — a boundary added to hold a miss is not a
boundary that predicted it.

## Scope

In scope: the local agent loop, its tool surface, its durable stores, the unattended
fleet, the self-modification route, the loopback API, and the desktop act path —
everything an instruction that did not come from Alan can reach, and everything an
agent instruction can reach in turn. Out of scope: multi-tenant governance, and
everybody else's machine. Single-owner box, one human, no org to police; the
governance half of the source workshop does not transfer and nothing here proposes
it.

## Boundaries

A seam is a boundary when something crosses it with authority, and the authority
has to be granted by code rather than by the model's good intentions. "Who
authorizes" is therefore the question worth a column: where the answer is the
model's own reading of prose, the seam is a boundary with the weakest possible
control on it, which is exactly what the source workshop's derived rule — the
prompt cannot be access control — is for.

| id | Seam | What crosses | Who authorizes | Chokepoint |
|---|---|---|---|---|
| B1 | external text → turn context | fetched pages, API bodies, transcripts, search results | the model's judgement, plus a destination table | `agent_mcp/egress.py`, `agent_mcp/http_tools.py` |
| B2 | vault text → turn context | prefetched notes, skills excerpts, facts, backlog rows | the subliminal matcher's relevance score, and `app/harness/capabilities.py`'s per-source ceiling, which decides whether the lane the note can reach is even allowed to hold the tool the injection aims at (T-17) | `app/prefetch.py:1037` |
| B3 | loaded memory → every system prompt | `MEMORY.md`, `USER.md`, the identity file | whatever the last writer did | `app/prompt_builder.py:927`, `app/prompt_surface.py` |
| B4 | skill body → instruction | a `SKILL.md` delivered as a dispatch marker | the skill's own name and its front matter | `app/harness/skill_dispatch.py:276` |
| B5 | tool result → next turn's input | every tool's own output, including a subprocess's stdout | nothing: results are trusted as authored | `agent_mcp/_injection_probe.py:121` (records) |
| B6 | model output → shell | an argv string | four deciders, not one: the deny table, twice; the sync-registration parse, which `check_bash_command` runs as its own third check and so fires at both of the deny table's points, refusing an argv that would change the vault's sync registration however the deny table read it; the kernel's bind the argv then runs inside, which **is absent by design when bwrap cannot build a namespace** (T-42); and the effect ledger's verdict on what the call will have *done*, which is a second decision on a call the first two already let through | `app/harness/safety.py:417`, `agent_mcp/main.py:531`, `app/harness/sync_registration.py`, `agent_mcp/_path_sandbox.py`, `agent_mcp/_tool_effects.py` |
| B7 | model output → filesystem and vault | a path plus bytes | one constant, three lanes — and the constant is a Python function: a shell's redirection does not call it, it meets the kernel's bind, which is B6's row, is a second decider of the same question, and is off by design when the namespace cannot be built (T-42) | `app/harness/protected_paths.py::write_deny_reason` |
| B8 | model output → the network | a URL, a body, a header | a destination table that **records rather than refuses** on this box: `harness.egress_policy.enforce` is set `false` at `~/lloyd/config.yaml:806`, so the allow-list half of `egress.guard` never runs and every lane answers from `_is_private_host` instead — one resolving predicate for all of them since #2584, which retired the text-matching copy the HTTP lane carried. #3846 split that seam, and this cell's single verdict used to flatten the split: the **refusal** half is what `enforce` disarms, while the **content** half of the same `guard` call is armed whatever that flag says — `app/harness/credential_store.py::substitute_headers` fills a placeholder in the outbound headers with the real value only for a destination the credential is bound to and sends the placeholder itself anywhere else, so a table that refuses nothing still decides which bytes go on the wire (T-08, T-20). The second decider is upstream of the address: `app/harness/capabilities.py`'s `durable_external` decides, from the source that armed the run, whether it may hold a browser profile at all (T-19). The browser's own floor is a guard inside this same crossing rather than a seam of its own: `_enforce_landing` re-checks where a navigation came to rest, and T-18 credits it | `agent_mcp/egress.py`, `app/harness/outbound_content.py:859`, `app/harness/capabilities.py` |
| B9 | an unattended session → the tool surface | every tool call from a worker or a round | two tuples and the session's platform — a floor, not a ceiling. Below it a declared capability set can only narrow what the platform allowed (`app/harness/capabilities.py`: the declared set, then expand subtracts every standing ban, and a widened declaration is not an error), and a source can narrow it again with a deny the tuples do not mention — which is what `workers/sources/arch_review.py` does for a review session, `DISALLOWED` taking the `automod_*` verbs and the write-and-schedule tools away from a job whose whole unit is reading a diff (T-43) | `app/tool_bans.py:23`, `app/harness/capabilities.py`, `workers/sources/arch_review.py` |
| B10 | a grant → authority | a scope predicate, an expiry, a quota | a human at Mission Control, or a task file's front matter | `agent_mcp/builtin_grants.py`, `app/harness/policy.py:1624` |
| B11 | an automod round → live `main` | a diff, then a restart of the running system | the gate ladder and the guardian | `scripts/automod/spec.py::check_scope`, `Gate.run` at `scripts/automod/gate.py:2731` |
| B12 | scheduler → a worker turn | a queue row, and the turn it becomes | the queue's own due-gates | `workers/queue.py`, `claim_next()` |
| B13 | a run's output → a durable store | loaded memory, facts, the entity store, eval baselines, `-latest` pointers | the job that owns the file, with no cross-job reader | `app/memory_flush.py`, `app/kg_store.py` |
| B14 | a network peer → the backend API | an HTTP request that can start a turn | the peer's address, and a fingerprint if one arrives | `./server.py:181`, `./server.py:241` |
| B15 | a nested RPC call → tool dispatch | a tool call issued from inside a `Bash` child | the allowlist stamped when the parent call began | `app/harness/rpc_policy.py:76` |
| B16 | an act → Alan's desktop session | a click, a keystroke, a window focus | a lease only a human grants, revoked by mouse motion | `app/desktop_lease.py:123` |
| B17 | a new dependency → this machine | a package name, a resolver, an installer | an install-command matcher | `app/harness/supply_chain.py:421` |
| B18 | a bench or eval trial → the production tool surface | a full `run_query` with the real tool pool | the task's `requires_runtime:` flag | `scripts/autoresearch/bench_runner_sdk.py`, `agent_mcp/_tool_sandbox.py` |
| B19 | a guard's own inputs → its verdict *(added by I-03, I-10, I-11 in this sitting)* | the file set a counter walks, the argv a matcher reads, the config value a mode resolves | the guard itself, with no second reader | `app/config.py`, `app/harness/denial_journal.py` |
| B20 | the vault → its only off-box copy | the vault's tree as sync pushes it, the sync registration a re-link rewrites, the 15-minute snapshot that is the only copy outside the vault | the guardian's mass-deletion trip, which stops sync and is latched; sync's start script and the snapshot timer, which both refuse while the trip stands; and `vaultwatch.py clear`, which only a human runs | `agent-services/guardian/vaultwatch.py`, `agent-services/bin/start-obsidian-sync.sh`, `scripts/backup/backup-vault.sh` |

B19 is the seam every other row assumes away: a guard reads something, and its
verdict is only as good as that read. It is in this table because three separate
incidents crossed it and the map had nowhere to put them — see the retrospective.

## Threats

Classes are STRIDE's — S spoofing, T tampering, R repudiation, I information
disclosure, D denial of service, E elevation of privilege — with the ATLAS classes
an agent adds marked `inject`, `poison`, `agency`, `persist`, `chain` and `M2A`
(misinformation-to-action). "Control" names the module that decides, or the
measurement that stands where a decision should. A control cell saying none never
appears in this table: those threats are in the next one.

| id | B | Class | Capability | Control |
|---|---|---|---|---|
| T-01 | B6 | T | A turn destroys a protected tree with one shell command | `app/harness/safety.py:417`, and the bind under the argv: `agent_mcp/_path_sandbox.py` — which is the decider that answers this threat when the string guard is the one that let it through, and which is absent by design when the namespace cannot be built (T-42) |
| T-02 | B6 | T | A turn truncates or overwrites a sacred file without deleting it | `app/harness/safety.py:417`, `agent_mcp/_path_sandbox.py` |
| T-03 | B6 | T | A turn runs a denied shape on a path that never installed the hook | `agent_mcp/main.py:531` |
| T-04 | B6 | E | A turn reaches a protected root through a symlink pointing into one | `app/harness/protected_paths.py::protected_write_roots` |
| T-05 | B7 | T | A turn writes into a protected root through the generic file lane | `agent_mcp/builtin_fs.py:239` |
| T-06 | B7 | T | A turn writes into a protected root through the vault lane | `agent_mcp/vault.py:1967` |
| T-07 | B7 | E | The one exempt write lane commits bytes no loader validated | `scripts/automod/vault_round.py` |
| T-08 | B7 | I | A vault write copies a credential into a file the sync client pushes | `app/harness/outbound_content.py:859` (`scan_outbound_payload`, the module's payload control — not the `redact` this cell used to cite at `:805`, which only keeps a secret out of a findings-ledger excerpt and never sees a request), and `agent_mcp/egress.py::guard`'s destination-keyed credential substitution (`substitute_headers` is called at :997 of that file), which is the armed half of #3846 and is what acts on the bytes of a marked credential that does leave over a guarded lane (`app/harness/credential_store.py::substitute_headers`; T-20) |
| T-09 | B3 | T | A durable memory line becomes an instruction every later turn obeys | `app/prompt_surface.py` |
| T-10 | B3 | T | A scheduled writer shrinks loaded memory and the correction is gone | `app/memory_ceiling.py:273` |
| T-11 | B3 | D | Loaded memory grows past the window and every turn pays for it | `app/memory_ceiling.py:151` |
| T-12 | B3 | R | An unattributed writer changes the prompt and nothing says who | `app/memory_snapshot.py:89` |
| T-13 | B4 | persist | A skill body delivered as a dispatch marker carries an instruction nobody re-reads | `app/harness/skill_dispatch.py:276` |
| T-14 | B4 | S | A skill named to collide with a real one is delivered instead of it | `app/harness/skill_dispatch.py:212` |
| T-15 | B1 | inject | Prose from a fetched page instructs the turn that fetched it | `agent_mcp/_injection_probe.py:121` |
| T-16 | B5 | inject | A tool result's own content instructs the next turn | `agent_mcp/_injection_probe.py:52` |
| T-17 | B2 | inject | A vault note's text is matched as relevant and then obeyed | `app/prefetch.py:1037` is the only decider that runs *at the seam*; upstream of it `app/harness/capabilities.py` answers a different question — `SOURCE_CAPABILITIES` maps a source id to the tools its runs may hold, so the lane a prefetched note can reach is a lane already barred from the verbs the note would ask for, and an over-broad declaration re-narrows rather than erroring |
| T-18 | B1 | E | Fetched content points the browser at a loopback or private address | `egress._is_private_host` — one resolving predicate since #2584, which retired the text-only copy the HTTP lane used to carry. `agent_mcp/browser.py`'s `_host_block_reason`, `http_fetch` and `http_request` all call it, each hop of a redirect chain included, and `egress.floor_reason` answers from it too, so arming `enforce` adds the allow-list rather than changing this answer. The browser answers a second question the destination table cannot: `_enforce_landing` re-checks where the navigation actually came to rest and blanks the page when a redirect, a meta-refresh or a script put a private host in front of the model, and its loopback rule is stricter than the policy's |
| T-19 | B8 | I | Data leaves to a destination the policy never named | `agent_mcp/egress.py` — the measurement, not the gate: its `DECISION_DENY` branch is armed only by `harness.egress_policy.enforce`, false here, so the table holds zero denies (#2580). Re-checked live 2026-11-11 at this head: `/api/dashboard`'s `network.by_decision` over a full 7-day window is `allow 711 / deny 0 / grant-required 8`, with `window_clamped_to_hours` null — so the zero is an unarmed branch, not an unlooked window. `app/harness/capabilities.py` is the one control on this row that refuses instead of recording, and it refuses upstream of the address: a source listed in `UNTRUSTED_INGEST_SOURCES` has its declared tier-2/3 durable-external tools — the browser and the messaging verbs — taken out of the ceiling its own runs are given, so the population most likely to be instructed to send something somewhere is the population that cannot hold the sender, whatever `enforce` eventually says |
| T-20 | B8 | I | A credential rides out inside a request body the model composed | **Partial cell — the named control covers only half of this capability.** The armed half is the destination-keyed credential substitution inside `agent_mcp/egress.py::guard` (`substitute_headers` is called at :997 there), the content control #3846 put inside `guard` itself: it replaces a marked placeholder with the real value only for a destination the credential is bound to, sends the placeholder unchanged anywhere else, and is armed whether or not `harness.egress_policy.enforce` is true (T-19) — so an escape attempt that carries the placeholder gets a token the far side cannot redeem. Its limit is the half no control closes: it substitutes **request headers only** — `app/harness/credential_store.py::substitute_headers` is that module's only substitution function, reached from the HTTP lane's own sender `agent_mcp/http_tools.py::_http_request` alone, whose `client.request(..., content=body.encode())` sends a composed body past it untouched, so a value a model writes into a request body leaves unsubstituted for any destination, bound or not. The cell used to name `redact` at `:805`, a text redactor for findings-ledger excerpts, never wired to a network request; the module's outbound-payload control is `app/harness/outbound_content.py:859`, which reports what a payload looks like and, since #4634, also refuses: its `lloyd-aggregator-credential` rule denies an outbound argument carrying this box's aggregator credential — compared against the value `agent_mcp/aggregator_auth.py::read_token` answers with at scan time, in every argument shape including the one that carries the credential under its own header name as an argument key (the shape no name-adjacency rule of this table can reach, because keys are never scanned as text), and against the header name beside a credential-shaped run when that file cannot be read. Every other credential still only records. |
| T-21 | B17 | chain | A turn installs a package nobody reviewed | `app/harness/supply_chain.py:421` |
| T-22 | B17 | chain | A skill's installer pulls arbitrary code through a channel the matcher does not read | `eval/supply-chain/` |
| T-23 | B11 | T | A round lands a diff in a path the loop may never write | `scripts/automod/spec.py::check_scope` |
| T-24 | B11 | E | A round edits its own gate to let itself through | nothing refuses the diff: `check_scope` at `scripts/automod/spec.py:766` puts `scripts/automod/**` in protected scope, which is permitted and turns on the `drill` rung, so the check is `Gate.rung_review` at `scripts/automod/gate.py:4529` and a person |
| T-25 | B11 | T | An agent-authored change weakens a guard and the suite stays green | `Gate.rung_review` at `scripts/automod/gate.py:4529` (a second reader grades the diff against the item's clauses) |
| T-26 | B11 | D | A promotion ships a crash loop into the running system | `agent-services/guardian/guardian.py:1035` |
| T-27 | B11 | T | A promotion silently deletes data and looks like a clean landing | `agent-services/guardian/vaultwatch.py` |
| T-28 | B9 | E | An unattended turn reaches a tool it was never allowed to hold | `app/tool_bans.py:23` |
| T-29 | B10 | E | An unattended turn mints itself the authority it was banned from using | `app/tool_bans.py:35` |
| T-30 | B10 | E | A grant outlives its scope, its quota or its expiry | `app/harness/policy.py:185` |
| T-31 | B12 | E | A child turn inherits a wider surface than the turn that spawned it | `agent_mcp/builtin_task.py` |
| T-32 | B14 | S | A peer outside loopback starts a turn over the API | `./server.py:181` |
| T-33 | B14 | S | A presented fingerprint is read as identity where none was required | `./server.py:241` |
| T-34 | B15 | E | A nested call performs a write, skipping every pre-dispatch hook | `app/harness/rpc_policy.py:76` |
| T-35 | B16 | E | The desktop is acted on with no live lease, or after the lease was revoked | `app/desktop_lease.py:123` |
| T-36 | B18 | agency | A trial told to be destructive is handed the real tool surface | `agent_mcp/_tool_sandbox.py` |
| T-37 | B13 | T | A derivation run rewrites the entity store and the graph is unrecoverable | `app/kg_store.py` |
| T-38 | B13 | poison | A night's bad output is the input the next night is graded against | `scripts/maintenance/corpus_shape.py` |
| T-39 | B19 | D | A guard refuses work that only describes an action, and the refusal is the damage | `app/harness/denial_journal.py` |
| T-40 | B19 | D | A guard's measurement set includes the machinery it is measuring | `app/config.py` |
| T-41 | B6 | M2A | A believable wrong answer becomes a command, because nothing checks the call against the task | `app/harness/action_review.py:103` |
| T-42 | B6 | D | The bind that would have made the string guard's judgement moot cannot be built, so the command runs unsandboxed and the guard that did see the redirect is left alone with it | `agent_mcp/_path_sandbox.py` — the trade is deliberate, and it is the shape of the trade that belongs on this map: `_unshare_flags` refuses to pretend a subset of the namespaces works, `_should_fallback` answers "build failed, `bwrap` missing, or an opt-out" by running the command anyway, `_build_argv` then re-invokes with no `bwrap` in the argv at all, and `app/harness/protected_paths.py::protected_shell_ro_paths` is what decided which trees the bind would have covered. What witnesses the trade is one `fallback: true` on the `protected_path_sandbox` key at `/state` (#2109) plus `describe_launch` in the launch log — a breadcrumb, not a second decider |
| T-43 | B9 | E | A job whose unit of work is reading a diff is dispatched holding the verbs that land a diff, and nothing in the standing bans would have caught it | `workers/sources/arch_review.py`'s `DISALLOWED`, handed to `envelope_for` as `extra_disallowed` by the shared worker turn path (`workers/sources/_common.py`): the `automod_*` verbs plus the write and schedule tools, and a deny a source carries itself rather than one `app/tool_bans.py` knows about — which is why the row's own rule is that the list must stay a superset of layer 6, and why the ceiling is the seam: a declared set narrower than the floor is narrowed by `expand`, never widened past it |
| T-44 | B6 | D | A call whose outcome the caller was never told is retried, and the retry fires an effect that already landed — the send, the write, the fact | `agent_mcp/_tool_effects.py` — authority keyed on the effect rather than on the argv, which is why it is a second decision and not a second copy of the first: one row per (tool, canonical arguments, scope) claimed before dispatch, so the `unknown` state is the only record that survives being cancelled mid-call, and `Claim.may_dispatch` replays a stored result for an effect already `ok` and refuses outright when the state is still `unknown` (#544). The deny table at `app/harness/safety.py::check_bash_command` never sees this decision, and cannot: it answers a question about one command string |
| T-45 | B20 | T | A mass deletion of the vault reaches its only off-box copy, or is recorded by the snapshot as the new normal, before anyone has judged the deletion | `agent-services/guardian/vaultwatch.py` — the trip runs every guardian tick, stops sync, pauses the pool and latches `vault-tripped.json`; its `sync-gate` is what `agent-services/bin/start-obsidian-sync.sh` asks before starting a client, `scripts/backup/backup-vault.sh` refuses to snapshot while the marker stands, and only a human's `vaultwatch.py clear` lifts it. A re-link that rewrites the registration from inside a turn is B6's fourth decider, `app/harness/sync_registration.py` |

T-41 is what [[guard-coverage]] §8 already states as fact: every guard standing
between an injected instruction and a call refuses on *shape*, and none asks
whether the call is the one the job asked for. Its control cell names the recorder
that exists rather than a control that decides — the capability is live, and the
answer to it on this box is a measurement, which is why `warn` was removed from
that set instead of being kept.

T-42, T-43 and T-44 are #4046's, and each is the threat over a rung of the ladder
that this page had no seam for: row 5, the bind under the shell; row 7, the
capability ceiling; row 9, effect authority; row 14, a deny list one source carries
itself. None of the four needed a new boundary row — each is a decision taken on a
crossing the map already drew, which is the point the rewritten authority cells make
by naming more than one decider where they used to name one. The provenance is worth
stating exactly, because the obvious story is false: rows 9 and 14 were in the ladder
before this map was drawn, so they were **omitted at authoring**, not overtaken; only
rows 5 and 7 postdate it. And the ceiling had never been named here at all: B9
credited two tuples and a platform, which is the floor, so the ladder row's actual
claim — that a tool is unreachable because nobody declared it rather than because
someone listed it, and that the refusal leaves no deny-list entry to read back — had
no threat over it. A boundary whose authority cell names one decider reads as settled;
that is how four rungs stayed off this map without anyone deciding to leave them off.
T-45 is the one #4046 left as a scope call and #4286 answered: ladder row 15's deciders
are tracked code, and the half of them reached from the guardian's tick and a systemd
timer rather than from any turn crosses no seam B1–B19 draw, so B20 is a boundary and
T-45 is the threat over it; the re-link half rides B6.

## Open holes

Threats with no control in this tree. An empty set here would be the surprising
result, not the good one: this box's security backlog is a list of reactions to
incidents, and a model that closed every row it wrote has inventoried the reactions
and not the system. The clause that lets a row go uncited is also the clause that
makes it visible, which is the only reason to keep the two tables apart.

| id | B | Class | Capability | Nearest existing work | Control |
|---|---|---|---|---|---|
| O-01 | B4 | persist | Nothing checks where a skill came from, or whether it changed since review, at the moment its body is delivered as an instruction | the index-description tests, and nothing upstream of delivery | **none** |
| O-02 | B13 | poison | A worker's own run report reaches next night's context through the same lane as everything else, with no sanitizer on that seam | [[memory]] documents the four `memory_*` verbs and their guards — all four **retired 2026-09-23**, their guards folded into the tools they wrapped — and says nothing about the provenance of a write | **none** |
| O-03 | B3 | T | A loaded-memory file other than the identity file is writable by every lane that can reach the vault | the ceiling guard bounds size, never authorship | **none** |
| O-04 | B18 | agency | A read-only capture is published to a peer that granted nothing, and its content is an input no scanner reads | the lease governs acting, never seeing | **none** |
| O-05 | B19 | R | A guard's verdict has no second reader, so an empty log cannot tell "nothing arrived" from "nothing looked" | #1948 closed the mode-typo case for two guards and no others | **none** |

## Retrospective: scoring the model against the incident corpus

The claim being tested is Silverman's, and he never prints his number: that a
threat model written on paper before the damage predicts the damage. Here the gold
set is free — this box writes its incidents up in prose with root causes — so the
question is answerable exactly: for each incident, did this model, over these
boundaries, name the seam the damage crossed?

**13 incidents assessed, 7 predicted, 6 missed.** Of the 6 misses, 5 are
`boundary absent` and 1 is `boundary present, threat not enumerated`.

The `Control predates the damage?` column is not decoration. It records whether the
module in the threat row's control cell existed and fired before this incident
happened, and it is what turns a self-congratulatory number into a usable one.

| id | Incident | B | Control predates the damage? | Verdict | Class of miss | Source |
|---|---|---|---|---|---|---|
| I-01 | The entity graph is destroyed and rebuilt wrong | B13 | no — the single-writer rule is the fix | predicted | — | `~/obsidian/knowledge/ai/2026-08-22-entity-graph-incident.md` |
| I-02 | Loaded memory is replaced by a verbatim copy of the identity file | B3 | no — the duplicate detector came after | predicted | — | `~/obsidian/knowledge/ai/2026-08-26-memory-md-corruption-recovery.md` |
| I-03 | A `data_damage` trip counts `.git` objects and reverts a gate-passing promotion | B19 | none | missed | boundary absent | `~/obsidian/knowledge/software/guardian-data-damage-false-trip.md` |
| I-04 | Loaded memory runs at 15 lines for 19 hours; 44 curated lines gone, no report says so | B3 | no — the ceiling guard landed 2026-09-23 and permits a shrink | predicted | — | `~/obsidian/knowledge/software/memory-md-clobber-2026-09-10.md` |
| I-05 | A background bench grid deletes the vault; Sync pushes thousands of remote deletes | B18 | no — the sandbox and the sync gate are both fixes | predicted | — | `architecture/vault-protection.md` |
| I-06 | A round re-runs its own destructive bench; five Bash calls, none denied | B18 | yes, failed to fire | predicted | — | `architecture/vault-protection.md` |
| I-07 | A suite run from the production checkout deletes the production checkout | B6 | no — the refusal to start there is the fix | predicted | — | `~/obsidian/knowledge/software/gate-round-home-red-tests.md` |
| I-08 | Nightly eval baselines are gone and unrecoverable after a data-root move | B13 | none | missed | boundary absent | `~/obsidian/knowledge/software/eval-baseline-history-loss-2026-09-22.md` |
| I-09 | `~/lloyd` is deleted and re-cloned; the scheduler dispatches into it 24 seconds later | B19 | none | missed | boundary absent | `~/obsidian/knowledge/software/half-restored-tree-outage-2026-09-29.md` |
| I-10 | A deny matcher refuses a command for *mentioning* a denied token | B19 | none | missed | boundary absent | `architecture/guard-coverage.md` |
| I-11 | A mistyped mode boots, the recorder silently stops, and an empty corpus reads as a clean window | B19 | none | missed | boundary absent | `architecture/guard-coverage.md` |
| I-12 | A `-latest` pointer carries an unfinished document, and the next job reads a skeleton as a report | B13 | no — the rule is prose | missed | boundary present, threat not enumerated | `~/obsidian/memory/mental-models.md` |
| I-13 | The vault write lane accepts a protected path because a second lane never asked | B7 | no — consulting the constant is the fix | predicted | — | `architecture/authority-surfaces.md` |

### What the misses say

The split is the finding, and it is not the split the item expected. **Five of six
misses are `boundary absent`, and all five crossed B19 — a guard reading its own
inputs.** That seam was not in the first draft of this map because a threat model
built by enumerating *chokepoints* finds seams where authority is granted, and B19
is where a guard takes a measurement. Nothing in the codebase is a chokepoint for
it: `data_damage` counting loose git objects, a matcher reading the word inside a
`grep -rn`, a mode falling back to a default, a scheduler that cannot see that the
tree it dispatches into was just re-cloned, a store whose root moved out from under
a stored command — five writers, one mechanism, no chokepoint, no control. The map
is wrong about the class of thing to map, not thin.

The single `boundary present, threat not enumerated` miss is the more tractable
half: B13 is a real chokepoint-shaped seam, and the model enumerated a derivation
run destroying the store (T-37) and a night's output poisoning the next (T-38) but
never "an artifact read as complete while it is still being written". That is a
row to add, not a new way of modelling.

### The number that matters

Raw coverage is 7 of 13. Independent coverage — counted only where the control
existed *before* the damage and fired — is **1 of 13, and that one is I-06, where
the control was in place and did not fire.** Every other prediction rests on a
module its own incident produced.

That is the honest answer to the item's premise, and it is not the answer the item
expected. A threat model derived from this tree's controls cannot predict this
tree's incidents, because the controls *are* the incidents, mineralised. It is the
hand-maintained-allowlist failure read from the other end: enumerating what has
happened cannot forecast what will. So the artifact's value is not the score. It
is the two things a score cannot produce: B19 as a named seam nobody had written
down, and five open rows (O-01…O-05) that no incident had yet reached.

## Marginal value against an ad-hoc pass

**Not measured in this round.** The control arm — a fresh agent with no threat
model, asked for its best attack, scored against this table in both directions —
needs an interactive pass and is owed by the check job, including the verdict
that the artifact added nothing. Clause 3 of the item's own acceptance says a
negative here is the honest deliverable, and this page will take it.

## What this doc does not cover

- **Whether a threat is likely.** This is a capability enumeration; it ranks
  nothing and estimates no exploitability. The gate's ladder is where a rung would
  be ranked against a threat id, and that is deliberately not done here: reordering
  rungs needs the gate's latency budget argued first (#2563 step 7 is owed to a
  future round, not this page).
- **Whether a cited control still exists, or a `#NNNN` is still open.**
  Path and line resolution is `tests/test_architecture_coverage_doc_claims.py`'s
  job, and #2582 shipped: that node now reads every top-level `architecture/*.md`,
  so this doc is inside the sweep rather than excepted from it, and the sentence
  claiming a five-doc scope here was the false half of a disagreement with the
  intro. Its line check is a span check (#2583, #3047), not a length comparison —
  but a cell in this table names a path and no identifier, so containment alone
  grades it and a number that drifted into a neighbouring definition of the right
  file stays green (#4045). Neither file re-derives a prose figure:
  the retrospective headline is recomputed from its rows, but the `1 of 13`
  independent-coverage number and every count in the review log are checked by a
  human reading the table, which is where they will drift. Backlog liveness is
  deliberately *not*
  pinned, because a suite node that goes red when a human closes an item is holding
  the board still rather than testing anything.
- **The guards themselves.** Their order, their modules and where one rule is
  spelled twice are [[authority-surfaces]]; what none of them sees is
  [[guard-coverage]]; the vault's layers are §2 of `architecture/vault-protection.md`.
  This page adds the threat each answers and nothing about how it works — two docs
  describing one guard in full is how one gets corrected and the other does not.
- **Anything about Alan.** The desktop lease is modelled as an authority seam and
  no further. Physical presence, the human in the loop, and the judgement that
  decides whether a control is worth its cost are outside a model of this kind, and
  Inner Voice's turn guards are [[inner-voice]] — layer 1 of the ladder is prompt
  prose, which this doc treats as the weakest control everywhere it appears.
- **A second model.** Every boundary here overlaps a row of [[authority-surfaces]]
  or a section of [[guard-coverage]]. Where this page and those two disagree, the
  other two are the inventory of what runs and this one is a reading of it: they
  win about the machine, this one wins about the threat. The overlap is now asserted
  in the direction that broke, too:
  `test_every_ladder_row_that_names_a_module_is_a_seam_this_map_cites` reads the
  ladder's own table and refuses a row whose module no seam or threat on this map
  credits, so a round that files a guard and does not revisit this page goes red
  rather than leaving the sentence above true-by-habit. A row that names no module is
  skipped, not failed; no row of today's ladder is one, since row 15 names its
  deciders beside its [[vault-protection]] pointer. Two rulings decide what that node
  asks for, both made by the owed-check job on #4046 and published here by #4286.
  **The seam criterion:** a ladder row owes a new boundary row only when its decider
  is reached from a crossing no B row models; a second guard inside a crossing the map
  already draws is named on that row's authority cell. That is why row 15 split —
  `app/harness/sync_registration.py` is B6's fourth decider, while the trip, the sync
  gate and the snapshot, reached from no turn at all, are B20 — and why row 11 gets no
  row: `_enforce_landing` is a guard inside B8's crossing, named on B8's authority cell,
  with its control still credited in T-18's. **The ownership ruling:** the overlap
  node's red belongs to the round that files the ladder row, and is fixed in that same
  diff; there is no separate map round to hand it to.
- **The ATLAS mapping in full.** STRIDE covers the six rows whose class letter is
  one of S/T/R/I/D/E; the five ATLAS-shaped classes are named in the table header
  and not formally cross-walked to technique ids, which a next reader should do if
  the enumeration ever grows past this size.

## Review log

- **2026-10-11 — created (#2563).** Nineteen boundaries from the tree's chokepoints,
  41 modelled threats, 5 open holes, and a 13-incident retrospective whose raw 7/13
  is reported beside the 1/13 that survives the predate test. Two things this pass
  learned and did not expect: B19 exists as a seam and explains five of the six
  misses on its own, and a model built from this tree's guards cannot predict this
  tree's damage because the guards are what the damage left behind.
- **2026-10-11 — `current` (arch-review, whole-doc unit).** All 41 threat rows,
  19 boundaries and 8 incident sources re-read against the tree at `1d694d45`:
  the table's shape, the miss arithmetic and the 12-rung gate count hold. Four
  cells did not. B8 and T-19 credited egress default-deny as an authorizer while
  `harness.egress_policy.enforce` is `false` on this box, and reading what falls
  back when it is off found the reason underneath: T-18's SSRF control exists
  twice and the HTTP lane's copy is the pattern-only one (#2584, #2580). Two
  anchors landed on non-deciders — the guardian line pointed at `main()`, the
  lease line at `grant()` — because the path node compares a line
  number to a file's length (#2583); O-02's nearest-work pointed at the four
  `memory_*` verbs, retired 2026-09-23. The intro and the not-cover bullet both
  claimed path resolution ran "as for every other doc in this directory": it
  covers five of thirty-six (#2582), and neither file re-derives a prose count.
- **2026-11-11 — `stale` (arch-review, whole-doc unit).** The map's mechanisms all
  still run, but five statements about the tree around them had moved. The intro
  and the not-cover bullet disagreed with each other about the path node's scope,
  and #2582 has since shipped — the node reads every top-level
  `architecture/*.md`, so the "five-doc scope / not the same check for the rest of
  this directory" sentence was the false side and is gone. Both places also still
  said a line anchor is checked only for length (#2583), which is now a span check
  with a named-symbol half (#3047): true for no cell in this table, whose
  identifier-free citations are graded by containment and sit outside the #3580
  rail: B2 and T-17 had slid onto `drop_sections`, an ablation helper, and are
  re-pointed at the relevance gate that decides the row's authority; two gate
  anchors drifted into the wrong definitions entirely and are filed (#4045, with
  #4046 for four authority surfaces the ladder grew since this map was drawn and
  #4047 for #3846's credential substitution, an egress control armed while
  `enforce` is off). The B8 config anchor had drifted by ninety-four lines and the
  flag is still false there; the guard ladder this page asks to rank is fifteen
  rows, not twelve; T-19's zero is now a dated live witness rather than a bare
  claim. What held: all 19 boundaries, 41 threats and 13 incident sources resolve,
  the arithmetic still sums to 7 predicted / 6 missed (5 absent-boundary),
  #2584's single resolving predicate is what every lane calls, both recorders run
  in `shadow`, `app/config.py` still boots-checks exactly two guard modes so
  O-05's "two guards and no others" stands, and skill delivery still has no
  provenance check, so O-01 is still open.
- **2026-11-11 — #4046, the four rungs this map had no seam for.** The premise came
  from a derived check, not a reading: parse the Module column of the ladder table,
  grep each module against this page. It failed on exactly rows 5, 7, 9 and 14, and
  a fifth — row 11, `agent_mcp/browser.py` — failed for a different reason worth
  separating: the page credited it as a bare ``browser.py`` span, which the sibling
  path node cannot grade because a citation without a directory cannot be resolved.
  Fixed by qualifying the span and by crediting the landing re-check row 11 is
  actually for. The four real ones were folded into the boundaries that already own
  their crossings rather than given rows of their own: B6 now names the bind and the
  effect ledger beside the deny table, B7 says out loud that its constant is a Python
  function a shell never calls, B8 names the per-source ceiling above the address,
  B9 says its two tuples are a floor and that a source can carry a deny the tuples do
  not know about, and B2 names the ceiling too because a prefetched note arrives in a
  lane that may already be barred from the verbs it would ask for. Three threats
  (T-42, T-43, T-44), each with a cited decider, and the control cells of T-01, T-02,
  T-17, T-18 and T-19 now credit the deciders they were answering with all along. What
  the diff does not do is add a boundary: nineteen rows, none of them bare, because
  `test_no_mapped_seam_is_bare_every_boundary_carries_a_threat_row` now refuses a seam
  nobody threatens. The intro lost its rung count for the same reason the arithmetic
  lost its manual re-count — a stated number is a number nobody re-runs.



