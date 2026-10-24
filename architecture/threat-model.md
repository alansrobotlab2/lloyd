---
segment: architecture
tags: [architecture, lloyd, security, threat-model, stride, atlas, trust-boundaries]
type: reference
status: implemented
date: 2026-10-11
---

# Lloyd — threat model: the boundaries, the threats over them, and what they predicted

Backlog #2563 asked for Silverman's procedure run against this repo instead of a
generic app: scope the system, name the seams, enumerate threats as *capabilities*
over those seams, attach the control in this codebase that answers each one, and
publish the threats with no control as the open set. [[authority-surfaces]]
already inventories the guards in the order an action meets them and
[[guard-coverage]] publishes the union of what none of them sees. Neither states
the other half: **what is being attacked**. Without that, eleven rungs — now
twelve — cannot be ranked, a security backlog item cannot be shown to answer a
threat, and the next argument about whether a rail is too broad (#2269's
`Bash` narrowing, which shipped and was reverted) is settled by intuition plus a
broken canary again.

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
makes. A line anchor is only
checked to be shorter than its file, which is how two cells in this very table
pointed at `main()` and at a grant function until the 2026-10-11 pass re-anchored
them (#2583).

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
| B2 | vault text → turn context | prefetched notes, skills excerpts, facts, backlog rows | the subliminal matcher's relevance score | `app/prefetch.py:1840` |
| B3 | loaded memory → every system prompt | `MEMORY.md`, `USER.md`, the identity file | whatever the last writer did | `app/prompt_builder.py:925`, `app/prompt_surface.py` |
| B4 | skill body → instruction | a `SKILL.md` delivered as a dispatch marker | the skill's own name and its front matter | `app/harness/skill_dispatch.py:276` |
| B5 | tool result → next turn's input | every tool's own output, including a subprocess's stdout | nothing: results are trusted as authored | `agent_mcp/_injection_probe.py:121` (records) |
| B6 | model output → shell | an argv string | the deny table, twice | `app/harness/safety.py:417`, `agent_mcp/main.py:531` |
| B7 | model output → filesystem and vault | a path plus bytes | one constant, three lanes | `app/harness/protected_paths.py:259` |
| B8 | model output → the network | a URL, a body, a header | a destination table that **records rather than refuses** on this box: `harness.egress_policy.enforce` is set `false` at `~/lloyd/config.yaml:712`, so the allow-list half of `egress.guard` never runs and every lane answers from `_is_private_host` instead — one resolving predicate for all of them since #2584, which retired the text-matching copy the HTTP lane carried | `agent_mcp/egress.py`, `app/harness/outbound_content.py:526` |
| B9 | an unattended session → the tool surface | every tool call from a worker or a round | two tuples and the session's platform | `app/tool_bans.py:23` |
| B10 | a grant → authority | a scope predicate, an expiry, a quota | a human at Mission Control, or a task file's front matter | `agent_mcp/builtin_grants.py`, `app/harness/policy.py:1363` |
| B11 | an automod round → live `main` | a diff, then a restart of the running system | the gate ladder and the guardian | `scripts/automod/spec.py::check_scope`, `scripts/automod/gate.py:3532` |
| B12 | scheduler → a worker turn | a queue row, and the turn it becomes | the queue's own due-gates | `workers/queue.py`, `claim_next()` |
| B13 | a run's output → a durable store | loaded memory, facts, the entity store, eval baselines, `-latest` pointers | the job that owns the file, with no cross-job reader | `app/memory_flush.py`, `app/kg_store.py` |
| B14 | a network peer → the backend API | an HTTP request that can start a turn | the peer's address, and a fingerprint if one arrives | `server.py:241`, `server.py:181` |
| B15 | a nested RPC call → tool dispatch | a tool call issued from inside a `Bash` child | the allowlist stamped when the parent call began | `app/harness/rpc_policy.py:76` |
| B16 | an act → Alan's desktop session | a click, a keystroke, a window focus | a lease only a human grants, revoked by mouse motion | `app/desktop_lease.py:123` |
| B17 | a new dependency → this machine | a package name, a resolver, an installer | an install-command matcher | `app/harness/supply_chain.py:421` |
| B18 | a bench or eval trial → the production tool surface | a full `run_query` with the real tool pool | the task's `requires_runtime:` flag | `scripts/autoresearch/bench_runner_sdk.py`, `agent_mcp/_tool_sandbox.py` |
| B19 | a guard's own inputs → its verdict *(added by I-03, I-10, I-11 in this sitting)* | the file set a counter walks, the argv a matcher reads, the config value a mode resolves | the guard itself, with no second reader | `app/config.py`, `app/harness/denial_journal.py` |

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
| T-01 | B6 | T | A turn destroys a protected tree with one shell command | `app/harness/safety.py:417` |
| T-02 | B6 | T | A turn truncates or overwrites a sacred file without deleting it | `app/harness/safety.py:417` |
| T-03 | B6 | T | A turn runs a denied shape on a path that never installed the hook | `agent_mcp/main.py:531` |
| T-04 | B6 | E | A turn reaches a protected root through a symlink pointing into one | `app/harness/protected_paths.py:171` |
| T-05 | B7 | T | A turn writes into a protected root through the generic file lane | `agent_mcp/builtin_fs.py:238` |
| T-06 | B7 | T | A turn writes into a protected root through the vault lane | `agent_mcp/vault.py:1709` |
| T-07 | B7 | E | The one exempt write lane commits bytes no loader validated | `scripts/automod/vault_round.py` |
| T-08 | B7 | I | A vault write copies a credential into a file the sync client pushes | `app/harness/outbound_content.py:526` |
| T-09 | B3 | T | A durable memory line becomes an instruction every later turn obeys | `app/prompt_surface.py` |
| T-10 | B3 | T | A scheduled writer shrinks loaded memory and the correction is gone | `app/memory_ceiling.py:272` |
| T-11 | B3 | D | Loaded memory grows past the window and every turn pays for it | `app/memory_ceiling.py:150` |
| T-12 | B3 | R | An unattributed writer changes the prompt and nothing says who | `app/memory_snapshot.py:89` |
| T-13 | B4 | persist | A skill body delivered as a dispatch marker carries an instruction nobody re-reads | `app/harness/skill_dispatch.py:276` |
| T-14 | B4 | S | A skill named to collide with a real one is delivered instead of it | `app/harness/skill_dispatch.py:212` |
| T-15 | B1 | inject | Prose from a fetched page instructs the turn that fetched it | `agent_mcp/_injection_probe.py:121` |
| T-16 | B5 | inject | A tool result's own content instructs the next turn | `agent_mcp/_injection_probe.py:52` |
| T-17 | B2 | inject | A vault note's text is matched as relevant and then obeyed | `app/prefetch.py:1840` |
| T-18 | B1 | E | Fetched content points the browser at a loopback or private address | `egress._is_private_host` — one resolving predicate since #2584, which retired the text-only copy the HTTP lane used to carry. `browser.py`'s `_host_block_reason`, `http_fetch` and `http_request` all call it, each hop of a redirect chain included, and `egress.floor_reason` answers from it too, so arming `enforce` adds the allow-list rather than changing this answer |
| T-19 | B8 | I | Data leaves to a destination the policy never named | `agent_mcp/egress.py` — the measurement, not the gate: its `DECISION_DENY` branch is armed only by `harness.egress_policy.enforce`, false here, so the table holds zero denies (#2580) |
| T-20 | B8 | I | A credential rides out inside a request body the model composed | `app/harness/outbound_content.py:526` |
| T-21 | B17 | chain | A turn installs a package nobody reviewed | `app/harness/supply_chain.py:421` |
| T-22 | B17 | chain | A skill's installer pulls arbitrary code through a channel the matcher does not read | `eval/supply-chain/` |
| T-23 | B11 | T | A round lands a diff in a path the loop may never write | `scripts/automod/spec.py::check_scope` |
| T-24 | B11 | E | A round edits its own gate to let itself through | `scripts/automod/gate.py:3532` |
| T-25 | B11 | T | An agent-authored change weakens a guard and the suite stays green | `scripts/automod/gate.py:2648` |
| T-26 | B11 | D | A promotion ships a crash loop into the running system | `agent-services/guardian/guardian.py:869` |
| T-27 | B11 | T | A promotion silently deletes data and looks like a clean landing | `agent-services/guardian/vaultwatch.py` |
| T-28 | B9 | E | An unattended turn reaches a tool it was never allowed to hold | `app/tool_bans.py:23` |
| T-29 | B10 | E | An unattended turn mints itself the authority it was banned from using | `app/tool_bans.py:35` |
| T-30 | B10 | E | A grant outlives its scope, its quota or its expiry | `app/harness/policy.py:185` |
| T-31 | B12 | E | A child turn inherits a wider surface than the turn that spawned it | `agent_mcp/builtin_task.py` |
| T-32 | B14 | S | A peer outside loopback starts a turn over the API | `server.py:241` |
| T-33 | B14 | S | A presented fingerprint is read as identity where none was required | `server.py:181` |
| T-34 | B15 | E | A nested call performs a write, skipping every pre-dispatch hook | `app/harness/rpc_policy.py:76` |
| T-35 | B16 | E | The desktop is acted on with no live lease, or after the lease was revoked | `app/desktop_lease.py:123` |
| T-36 | B18 | agency | A trial told to be destructive is handed the real tool surface | `agent_mcp/_tool_sandbox.py` |
| T-37 | B13 | T | A derivation run rewrites the entity store and the graph is unrecoverable | `app/kg_store.py` |
| T-38 | B13 | poison | A night's bad output is the input the next night is graded against | `scripts/maintenance/corpus_shape.py` |
| T-39 | B19 | D | A guard refuses work that only describes an action, and the refusal is the damage | `app/harness/denial_journal.py` |
| T-40 | B19 | D | A guard's measurement set includes the machinery it is measuring | `app/config.py` |
| T-41 | B6 | M2A | A believable wrong answer becomes a command, because nothing checks the call against the task | `app/harness/action_review.py:103` |

T-41 is what [[guard-coverage]] §8 already states as fact: every guard standing
between an injected instruction and a call refuses on *shape*, and none asks
whether the call is the one the job asked for. Its control cell names the recorder
that exists rather than a control that decides — the capability is live, and the
answer to it on this box is a measurement, which is why `warn` was removed from
that set instead of being kept.

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
  job for this doc — the same node's five-doc scope is #2582, and its line check
  is a length comparison, not a symbol one, which is #2583. It is not the same
  check for the rest of this directory. Neither file re-derives a prose figure:
  the retrospective headline is recomputed from its rows, but the `1 of 13`
  independent-coverage number and every count in the review log are checked by a
  human reading the table, which is where they will drift. Backlog liveness is
  deliberately *not*
  pinned, because a suite node that goes red when a human closes an item is holding
  the board still rather than testing anything.
- **The guards themselves.** Their order, their modules and where one rule is
  spelled twice are [[authority-surfaces]]; what none of them sees is
  [[guard-coverage]]; the vault's four layers are `architecture/vault-protection.md`.
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
  win about the machine, this one wins about the threat.
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



