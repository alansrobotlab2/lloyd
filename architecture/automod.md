---
segment: architecture
tags: [architecture, lloyd, automod]
type: reference
status: implemented
date: 2026-09-12
---

# Automod: self-modification with automatic rollback

Lloyd changes his own code through a gated loop that can undo itself. This
describes the framework, the process for using it, and the reasoning behind
the parts that look arbitrary — most of which are scar tissue from something
that actually went wrong.

---

## 1. Why this needed machinery at all

**The git working tree is production.** `conf.d/lloyd-backend.conf` runs
`server.py` out of `/home/alansrobotlab/lloyd`. No build step, no release
artifact, no staging. A saved file is a deploy, pending only a restart.

**The repair mechanism lives inside the thing that breaks.** The agent loop,
the tool dispatcher and the autonomy scheduler all run in one process —
`server.py` starts the autonomy ticker on its own startup event. If
`lloyd-backend` goes FATAL, nothing is left alive to fix it. Break
`agent_mcp/main.py` instead and the backend survives with no tools: it can see
the problem and cannot touch it.

That asymmetry is why the pre-existing autoresearch loop mutates only SOUL.md,
MEMORY.md and USER.md. A bad prompt makes Lloyd dumb; a bad import makes him
dead. Extending self-modification to code needed two things prompts never
did — something that proves a build boots, and something that survives it not
booting.

---

## 2. Shape

```
 propose ──► WORKTREE ──► GATE (§4 rungs) ─► PROMOTER ──► live tree
                              │                  │
                              └── fail ──────────┤
                                                 ▼
                                          GUARDIAN (systemd, stdlib)
                                    liveness · errors · data · regression
                                                 │
                                                 └──► rollback to last-known-good
```

Four components, one state directory, and a hard rule about which of them may
write what.

| Component | Lives in | May write |
|---|---|---|
| Round | `scripts/automod/round.py` | its worktree |
| Gate | `scripts/automod/gate.py` | nothing in the live tree |
| Promoter | `scripts/automod/promote.py` | live tree, `current.json` |
| Guardian | `agent-services/guardian/` | live tree, `last_known_good.json`, `last_settled.json` |

### 2.1 Names

| Name | What it is | Where it lives |
|---|---|---|
| `automod` | the whole architecture: round, gate, promoter, guardian, state, tools, routes, config | `scripts/automod/`, `agent_mcp/automod.py` (`automod_*` tools), `app/routers/automod.py` (`/api/automod/*`), `automod:` in config.yaml, `~/.local/state/lloyd-automod/`, `automod/<round>` branches |
| `autotriage` | worker source: judges one `draft` item per run and records a verdict; never opens a round | `workers/sources/autotriage.py` |
| `autocode` | worker source: takes one `up_next` item and runs one round on it through the gate | `workers/sources/autocode.py`, dedup key `autocode:round`, tag `spawned-by-autocode` |
| `automod-regression` | worker source: paired A/B eval after a promotion, read by the guardian's regression detector | `workers/sources/automod_regression.py` |
| `SM_<stamp>` | a round id; the prefix predates both renames below and stays | `automod/SM_…` branches, `~/lloyd-work/SM_…` worktrees |

The rule: `automod` names the machinery and anything only the machinery
touches; a source is named for its job. A round is automod's unit of work
whoever opened it — a human at the CLI, `autocode`, or the vault route — so
"an automod round" is right and "an autocode round" is not.

**Two earlier names survive on disk, and that is history, not drift.** The
loop was `selfmod` (sources `backlog-selfmod` / `backlog-implement`) until
2026-09-09 11:21, then `autoimplement` — one word for both the loop and its
implement source, which is the ambiguity this split removes — until 23:06 the
same day. Deliberately not rewritten either time: the ledger's event types
(`backlog_triage`, `backlog_implement`; append-only), the `runs` and `queue`
rows in `workers.db` under retired source names, the `spawned-by-selfmod` /
`spawned-by-autoimplement` tags and `selfmod_landed` / `autoimplement_landed`
markers on backlog items (`scripts/automod/backlog.py` reads all of them
beside the current ones — a quarantine that stopped recognising an old tag
would re-admit every item carrying it to the triage pool at once), session ids
with the `autoimpl_` slug, and the round-id prefix. Both renames landed the
same way: commit, bridge the old state dir with a symlink so the pinned
guardian keeps reading, `round restart` under the guardian lease, re-stage the
guardian, remove the symlink.

**The promoter never advances last-known-good and the guardian never
promotes.** That split is what makes LKG mean *observed healthy in
production* rather than *passed a pre-flight*, and it guarantees a rollback
always targets a commit that already survived a full observation window.

**Nothing inside the blast radius rolls back inline.** A rollback stops the
backend and the aggregator, so any code performing one from inside either
process issues the stop that kills its own caller and never reaches `git
reset` — the stack goes down and the tree does not move, which is worse than
both doing it and not doing it. The backend, the aggregator and the quality
worker all *request* a rollback (`rollback_request.json`); the guardian is the
one component outside that radius and it performs every one. For the same
reason a **landing runs detached** (`state.spawn_detached`, a new session,
which is what supervisord's `stopasgroup` signal cannot reach): the promoter
restarts the very process it is usually called from.

**The round is a wrapper, not a code generator.** `automod_start` returns a
worktree path; Lloyd edits it with the ordinary Edit/Write/Bash tools and
commits. There is deliberately no `automod_write_code` tool — proposing a
change is just normal work that happens somewhere safe.

**Retired 2026-09-24 (IV plan R5): `automod.require_inner_voice` now defaults
to false.** Since 2026-09-12 worker and autonomy sessions were already exempt,
so the rule below refused only a chat-driven round, and both of its reasons are
now met without the observer: every turn is recorded and every turn runs the
deterministic turn guards (`app/harness/turn_guards.py`). Setting the key true
restores the refusal described here for chat turns.

**A round runs under Inner Voice, or not at all.** `automod_start` refuses a
turn with no observer attached. Two reasons, and the second is the one that
survives a round nobody was watching live: the observer catches the loop
drifting off the request while it happens, and an IV session is the only kind
the Inner Voice tab lists, so this is what makes self-modification reviewable
afterwards. A round driven from a plain session leaves a ledger row and a diff,
which say what changed and nothing about how the agent got there.

It **enables the flag and then refuses once**, rather than just switching it
on, because the observer attaches at turn start — `attach_observer_for_turn`
runs before `run_query`. Flipping the flag mid-call covers the next turn and
not the one that flipped it, the same shape as the position-0 rule for the
system prompt. A single-turn round would otherwise report itself observed while
running blind. The retry runs observed.

Two paths stay unobserved on purpose. The **canary smoke turn** is a gate rung,
not an automod job, and the observer would add LLM calls and non-determinism to
a check whose value is failing if and only if code under test is broken.
**Worker turns** have no session at all, so `messages.py` cannot attach an
observer to them; that is why workers and Task subagents are barred from the
automod tools rather than trusted with a prompt that says not to.

---

## 3. The process

### 3.1 For Lloyd (autonomous)

```bash
python -m scripts.automod.round start "make X faster"   # cuts a worktree
#   ...edit that worktree, commit inside it...
python -m scripts.automod.round gate  SM_<id>           # the rung ladder (§4)
python -m scripts.automod.round land  SM_<id>           # idle-gated, verified
```

Or the same steps as MCP tools: `automod_start`, `automod_gate` (returns
at once; the gate runs detached), `automod_gate_wait` (blocks in slices and
hands back the per-rung report), `automod_land`, `automod_status` (plus
`automod_abort`, `automod_amend_clause` and `automod_rollback`; the vault
route adds `automod_vault_land` and `automod_vault_revert`). Every mutating
tool refuses while `automod.enabled` is false, which is the default — with
one deliberate exception, `automod_abort`, because abandoning a round has to
stay possible after the switch has gone off. §4.5b says why the gate is
detached.

### 3.2 Unattended

Two worker sources, off by default, chained so that implementation can never
start from an unverified premise:

- **`autotriage`** records a verdict with evidence. It takes a *cluster* from
  `clusters.json` before it takes a single item (§3.2c); failing that, one
  open item per run, oldest first. `stale` and `already_done` close the item;
  `confirmed` records an acceptance check and stops. It never opens a round.
  Items this loop filed itself are held out of the single-item candidate pool
  — see "The pass may not eat what it files" below.
- **`autocode`** takes a still-open `confirmed` item that has an acceptance
  check and runs one round on it through the normal gate. Not simply the
  oldest: `select_confirmed` orders nearest-to-landing first, then fresh
  confirmations, then other re-offers (§4.5). It refuses while the loop is
  anything but free — disabled, halted, BROKEN, a promotion under observation,
  a rollback pending, a round open — and checks that twice, once before
  queueing and again before spending the turn. **One attempt per item**, where
  an attempt means a round that reached a verdict on the change; §4.2c is what
  a round has to do to spend one. The attempt is recorded before the turn
  starts, so a crash cannot put the item back on the pile; a second try is a
  human's call.

Both run **in a real session through `POST /api/message/stream`**, not through
`run_query` directly. That is the only turn path that attaches the Inner Voice
observer, and it is what leaves a transcript in the Inner Voice tab — the same
way the three hand-driven rounds ran. `run_prompt_in_session` in
`workers/sources/_common.py` is that path; `run_prompt_on_primary` is the
unobserved one, and it must not be used for anything that judges or changes
this code. (It stopped being session-less on 2026-09-10 — every background run
records a transcript now — but a transcript is not an observer.) Whether the
observer watches is `workers.sources.<name>.inner_voice`, `true` for both of
these sources, and since the same date the chat endpoint installs the #534
grant gate on their sessions by platform. Before that, the two sources that
rewrite this repo were among the four session-backed workers that ran
ungated. `architecture/background-runs.md` is the long version.

Three things were measured to be in the way before this was safe to leave
alone, each now pinned by a test:

| | was | measured need | now |
|---|---|---|---|
| iteration budget | 30 | 45, 65, 76 on the three hand-driven triages | 90, per-request via `max_turns`, ceiling 120 |
| item body | 6,000 chars | next item 12,279; largest 21,300 | 30,000, cut from the end |
| observer | none | — | attached, transcript kept |

The ceiling is enforced at the endpoint, not by the source, and it binds:
`messages._turn_budget` (`app/routers/messages.py:132`) clamps whatever a worker
asks for to
`agent.max_turns_ceiling` (120). Triage asks for 90 and gets it; `autocode`
asks for 150 — the value `workers.sources.autocode.max_turns` carries, and
the one its own `DEFAULT_MAX_TURNS` argues for — and is served 120. Raising
one without the other moves nothing.

**Running out of budget is not a conclusion.** The loop stops cleanly at
`max_turns` with whatever text it has, which for a triage means no verdict
block. That used to be recorded as `unverifiable`, a *terminal* verdict, so
the hardest items on the board were retired for good on first contact for a
reason indistinguishable from "states no checkable claim". It is now recorded
as `incomplete`, the item comes back, and only a second exhaustion retires it —
with evidence that says exactly that and names the transcript.

Two more were found by the first unattended run itself (#229, 2026-09-07:
`stale`, closed, 9 of 90 iterations), which is what a dry run is for:

- **A worker session is not the user's session.** 77 seconds after the
  triage turn ended, the morning brief — a MockBOT meeting that night — was
  injected into that session and answered there, to nobody.
  `session_inject_context` with no target asks `GET /api/sessions/active`,
  whose first rule is "the last session to receive a user turn", and a worker
  turn arrives through the chat path precisely so that it gets Inner Voice.
  `sessions_io.NON_USER_PLATFORMS` now excludes `worker` beside `autonomy` in
  both resolution rules, and `/inject` refuses such a session with **409**
  rather than a 200 "skipped", so the producer's `ok` is false and nothing
  records the brief as delivered. It is a deny-list on purpose: a client the
  list has never heard of must keep receiving its briefs.
  `tests/test_active_session_resolution.py` pins both.
- **The verdict template taught its own placeholder.** The prompt spelled the
  not-confirmed case as `ACCEPTANCE: <… else: ->` and the model copied the
  template's closing bracket verbatim, so the ledger read `acceptance: "->"`.
  `select_confirmed`'s guard stripped the `-` and was left with `>` — truthy —
  so a `confirmed` written the same way would have handed the implementer `>`
  as its contract. `backlog.acceptance_text` is now the one definition of
  blank (no alphanumerics, or a lone placeholder word), the parser records
  `""`, and the prompt says "otherwise the word none".
- **A finding that lives only in EVIDENCE is lost.** #229's verdict said two
  surviving claims "belong in two new items, not this one" — and filed
  nothing, in a turn whose prompt forbade writing anything. The item was then
  closed. Both prompts now make filing a **required step**: anything real the
  in-focus item does not cover — a claim that survives a `stale`, a bug seen
  on the way, scope the implementer's acceptance check does not reach — is
  filed with `backlog_write_task` (tags `spawned-by-triage` /
  `spawned-by-autocode`, first line naming the source item) *before* the
  verdict, and listed under a new `SPAWNED:` field. The triage prompt's
  read-only rule now says read-only **on the code**, because as written it
  forbade the very write. What the model says it filed is a claim and the
  file is the fact: `backlog.existing_ids` checks each id on disk, the ledger
  records `spawned` and `spawned_unverified` separately, and the closed item's
  activity log names the ids so the split is followable from either end.

  **The pass may not eat what it files.** That required filing step met
  `select_candidate`, which takes the oldest untriaged *open* item, and
  `OPEN_STATUSES` contains `draft` — the status `backlog_write_task` writes.
  So every item triage filed re-entered the queue it came out of, and the
  pass became its own supplier. Measured over the loop's first 48 hours
  (2026-09-06 to 09-08): **40 triage runs closed 28 items and filed 78**, a
  reproduction number of **1.95**, or +46 open items a day at the then-cadence
  of one run per 30 minutes. The open board went **19 → 122**, with 110 of the
  122 written by the loop. R > 1 is the whole defect — the queue doubles
  rather than drains, however good the verdicts are, and no cap on spawns per
  run changes that shape. Only cutting the edge does.

  Oldest-first ordering is what hid it. Self-filed items sort to the back, so
  the pass works the real backlog first and reads as healthy right up to the
  moment there is nothing else left; on 2026-09-08 that moment was three hours
  and 6 items away. `backlog.is_quarantined` holds an item tagged
  `spawned-by-triage` or `spawned-by-autocode` out of the single-item pool.
  The first cut released it at `SPAWN_TRIAGE_MIN_AGE_DAYS` (30) on the theory
  that an unimplemented item can go stale; by 2026-09-11 that was 291 items
  due back in triage in October, each spawning ~2 more, so **age no longer
  releases**. The exits are the ones that do not re-enter the queue: the
  clustering pass and a group triage `keep` (§3.2c), expiry, and a human
  reopen. The gate keys on those tags and **not** on `draft`, which is the
  status of most of a stale backlog — a rule that skipped drafts would switch
  the pass off rather than bound it. A **live blocker** — an open `blocker`
  whose blocked item is still open (`backlog.live_blockers`) — is exempt from
  quarantine, the depth-gate hold and expiry, and is taken first by triage
  and early by autocode; until 2026-09-14 it was quarantined and then expired,
  orphaning the clause deferred to it. `architecture/backlog.md` has the table.

  An exhausted queue therefore has two meanings. `backlog.triage_pool` returns
  the held count beside the candidates so the skip summary can say which one:
  "every open backlog item has been triaged" was true, and misleading, on a
  board of 122 where 106 were this loop's own drafts. `SPAWN_CAP` bounds
  fan-out per run — 1 for both since 2026-09-13. For autocode the one thing
  that may become an item is a **blocker**; for triage it is one survivor of
  a closing verdict; every other finding goes onto an item as `## Findings`
  (§3.2c), never nowhere (#229's lesson still holds). Triage's cap was 3 plus
  a "Further findings from…" overflow item until then. It is **recorded, not enforced**: the items are on disk before
  `SPAWNED:` is parsed, so unfiling them would destroy real work. Both event
  types carry `spawn_cap` and `spawned_over_cap`.
  `tests/test_backlog_spawn_loop.py` pins it, including the counterfactual —
  with the spawn tags unrecognised the same run grows the queue.
  The proving run on #278 (`confirmed`, 16 iterations) filed #402 and #403
  before its verdict, after checking `backlog_tasks` for duplicates — and
  exposed that the acceptance check, the implementer's contract, was cut at
  600 chars in the ledger mid-way through its regression guards and written
  nowhere in the item. It is now kept whole (3,000) and `record_verdict`
  writes it into the item as **Acceptance — what must become true**, because
  the item is the handoff and a contract only the ledger holds is one the
  item's next reader never sees.

**Surfaces.** The verdict block carries `SURFACE: code|frontend|vault|mixed|
external`, and it decides the implementer's route. `code` and `frontend` go
through a worktree round and the gate (the frontend rung builds `web/src`).
`vault` goes through `scripts/automod/vault_round.py`, because the vault is a
separate git repo and a *live* tree — Lloyd, the nightly jobs and Alan write
into it at once, and `prompt_builder` and `autonomy` read it straight from
disk — so there is no candidate to gate in isolation. The route is therefore
*validate → commit only these paths → revert on failure*: `.obsidian/**`,
`.git/**` and `.trash/**` are denied; every changed `.md` must have front
matter that parses to a mapping; for paths that feed a prompt or the
scheduler (`skills/**`, `lloyd/**`, `autonomy/**`) the real loaders run in a
fresh interpreter, scoped to what changed — the system prompt must still
build, a touched skill must still load, a touched task must still parse. A
failure puts the round's paths back (tracked ones to HEAD, new ones deleted),
because on a live tree "nothing lands" has to mean "nothing stays". Success
commits exactly those paths on the vault's `main` (the same branch guard as
`scripts/util/vault-commit.sh`) and records a `vault_land` ledger event with
the sha — and, when the vault review passed, its per-clause verdicts as
`review_clauses` (§3.2b); `automod_vault_revert` is a plain `git revert`,
also recorded. Both
tools sit behind the Inner Voice gate like `automod_start`, and both are
denied to session-less worker turns. `not_code` is now reserved for
`external` — hardware, robots, third-party services — and vault items get
real verdicts.

**Drain first, then wait for idle.** The promoter used to poll for three
consecutive quiet ticks and only *then* arm the drain — the one moment it is
no longer needed. Against a worker pool that starts a research or distill job
every few minutes, three quiet polls in a row never arrive: the first landing
of the unattended era (SM_20260907_233449, the frontend rung's proof round)
spent its entire 900 s budget watching `harness_runs` flicker between 1 and 2
and never drained at all. The hand-driven rounds only ever landed because
the pool was quieter then. `wait_idle` now arms the drain before its first
poll (chat turns and worker jobs both honour it, so nothing new starts and
what is in flight finishes), re-arms it inside its 180 s TTL, and releases it
on give-up so a failed landing does not leave the backend refusing turns.

**A turn that dies at its budget is not the end of the round.** The second
unattended implement attempt on #278 (session `20260908_000804_backlogi_3828`)
wrote the whole feature — a headless default for Chromium, an SSE route, a
Browser page in Mission Control, 356 lines of tests — and died at iteration
101 of 100 without ever calling the gate. Three things now stand between
that and a lost round, in order:

1. **The budget anchor.** `_build_state_anchor` appends a `<budget>` message
   at 75% and 90% of `max_turns`: how many iterations remain, and that a
   round left open must be gated and landed or aborted now. Deterministic
   and free; the observer's own `iteration_pressure` nudge is LLM-judged and
   subject to its inject cap, which this very round exhausted on repetition
   guards before any budget warning reached it.
2. **The observer's ambient follow-up**, which is what actually rescued
   #278: two minutes after the cut-off it queued "the round ended at the cap
   with no report; if the gate passed, land it" into the same session, and
   that turn gated, landed, and the feature was live at 17:34. The first
   responder.
3. **`reap_abandoned_rounds`**, the backstop: a round this source opened,
   still open, nothing under observation, its session idle, is aborted with
   its branch kept and the item told where the work is. How long it waits
   follows `autocode.inner_voice`. Observed, twenty minutes after the turn
   ended, on the scheduler tick — at turn end it would have raced #278's
   rescue and thrown away 875 lines the gate then passed. Unobserved (since
   2026-09-12), no wait: nothing can rescue the round, and `_run_and_record`
   reaps it the moment the turn ends (see §3.2d). Either way it never reaps a
   round whose detached gate (`gate.running`) or landing (`land.running`) is
   still alive — protections the twenty minutes used to provide by accident.

The reaper reads terminal rows (`finished`, `infra_failed`), and a turn killed
*with the backend* writes neither: its item's last row is `started`, which is
exactly what a live turn looks like. On 2026-09-15 systemd-oomd took the whole
unit down at 04:48:34Z two minutes into #1131's round; supervisord had
everything back in half a minute, and every autocode poll declined "a round is
already open" for thirteen and a half hours. `settle_orphaned_turns` closes
that gap once per backend process, at the first poll after boot: a turn runs
inside the backend, so a `started` row older than the process cannot still be
running, and it is written `infra_failed` (`stop_reason: backend_restarted`) —
an `infra` re-offer, not a spent attempt — naming the round it opened, which
the reaper then closes under its usual guards. The round is the last
`round_start` between that row and the boot whose `run_spec.yaml` does not bind
another item; `_round_opened_since` has no upper bound, and after a restart
its answer can be the next item's live round. Outside the backend (no pool in
the process) the pass does nothing, since from a CLI every live turn started
before the process did.

**Human-only.** Some paths the loop may never touch remain: `config.yaml`,
`data/**`, `.env*`, `pytest.ini`, `.gitignore`, and the frontend's build
inputs. A triage whose fix needs one records `confirmed` with an acceptance
that begins `human-only:`, and `select_confirmed` skips it — the alternative
was an implement round spent discovering it, which is what #278 cost before
`web/src` was allowed. A *condition* only a person can satisfy — an audit,
a sign-off, a measurement that needs real traffic — is the other shape, and
it goes in `human_clauses`, never in `acceptance_clauses`: the item is still
implemented, the reviewer does not grade those, and the landing leaves it
open tagged `needs-human` (§4.5c).

### 3.2a Status is the state machine

Two jobs, one board, and until 2026-09-09 neither job wrote to it. The
ledger decided everything (`backlog_triage` verdicts, `backlog_implement`
phases) and `status` decided only open-versus-done — so the board could not
tell you where anything was, and nine landed items sat as `up_next`.

```
draft ──autotriage: confirmed──▶ up_next ──autocode: round opens──▶ in_progress
  │                                 ▲                                        │
  └──autotriage: already_done/stale─┼──── attempt ended without a verdict ◀──┤
                          done ◀────┘◀── landed & met, or unnecessary / rejected ┘
```

`rejected` (2026-09-16) is Alan's rule that every backlog item is a
**proposal for research and eval**, deployed only when the measurement says
it improves things. A round that built or measured the idea and found no
gain closes the item `done`, tagged `rejected`, with the measurement in its
activity log — no re-triage, no `needs-human`. Before this the loop had no
honest exit for a negative result: a round either forced a landing or spent
its attempts as `not_met`. The scorecard's row 9 counts rejections beside
landings, because the loop is judged on items *resolved*, not shipped.

`autotriage` reads `draft` only; `autocode` reads `up_next` only. Every
other transition is derived, not scattered: `backlog.desired_statuses`
computes the status each open item *should* have from the ledger — in flight
(`started` with nothing after it), landed-and-awaiting, promoted-and-observing,
an outcome that re-offers (`up_next`) or spends the attempt (`draft`, tagged
`needs-human` — the pool means implement will take it and it will not, and
`draft` is 250 deep, so the tag is what makes a decision findable; it comes
off when a reopen moves the item back), a confirmed verdict, a non-confirmed
verdict — and `reconcile_statuses` writes the differences, on
every implement poll and after every turn. One table, so the migration of the
existing board and the steady state are the same code.

Two rules keep it from fighting a human: `done` is terminal for this writer,
and the loop rewrites only statuses it has a ledger opinion about. The one
exception is an untriaged item parked in `up_next` — nothing can pull it from
there, so it goes back to `draft`, where triage looks.

**A status outside that vocabulary is invisible to the loop and open on the
board**, and the pass above is structurally unable to fix it: `set_status` and
`reconcile_statuses` both reach items through `open_items`, which filters *on
status*. So the one defect this state machine cannot see is a status that is
wrong in exactly this way — the dashboard counts it as open work while
`OPEN_STATUSES` cannot see it at all. #287 (`review`) and #304 (`closed`) sat
there from April 2026 until 2026-09-09, stranded when the vocabulary was
narrowed to four words and nothing migrated what was already on disk.
`app/backlog_status.py` is the one definition, and `rescue_off_vocabulary`
runs at the *top* of every reconcile — walking by **board** rather than by
status, and before the main pass, so a rescued item gets a real verdict in
that same pass instead of waiting for the next one. The mapping is
deliberately lopsided: only words already known to be terminal (`closed`,
`cancelled`, `wontfix`) reach `done`, everything else becomes `draft`. Calling
a word terminal when it is not buries a live item where nothing will look
again; calling it live when it is not costs one triage run that closes it.

### 3.2b The item is closed when the round says so, once the landing settles

The loop wrote three records about a landing — the promotion, the guardian's
`settled`, the turn's `finished` — and read none of them back to the item.
Nine settled landings, nine open items. `implemented_ids` kept them from
being re-picked, so the failure was invisible from inside the loop and
visible only as an open count that never went down.

An implement turn now ends the way a triage turn does: one more completion
under a grammar (`IMPLEMENT_OUTCOME_SCHEMA`) restating the result as
`{landed, acceptance: met|not_met|deferred|unnecessary|rejected, clause_outcomes:
[{clause, outcome, evidence, deferred_to}], deferred_to, summary, spawned}`,
recorded on the `finished` event as `outcome`, with `outcome_error` beside it
so a finalizer that quietly stopped working does not look like one that is
working. `close_settled_items` runs on every implement poll: for each
settled promotion (or vault landing) whose item is still open and not yet
marked, it writes `automod_landed: <sha>` and an activity line, and sets
`status: done` **only when every clause said `met`**.

**A landing a rollback took off `main` is not a landing, whatever the row
names** (2026-09-21). `rollback_succeeded` names one commit: the promotion the
guardian was observing, or HEAD. A `reset` removes everything between HEAD and
the commit it restored, and seven readers took the named commit as the whole
of it. That day a regression check blamed a802b979 (#763) while 1e219da9
(#1038) was under observation on top of it. The guardian reset both away and
wrote `commit: 1e219da9`, so a802b979 read as settled and #763 closed as
landed, a minute after its change was reverted. #939 (dbec85aa) went the same
way an hour later. `state.reverted_commits` is the one definition now: a
`reset` counts every promotion on the `parent` chain from `head_before` back to
`restored`, plus the commit the `rollback_requested` named. Every reader of
"reverted" uses it (`backlog`, `scorecard`, `review_tools`, the regression
queue). The regression check reads a promotion after it settles, so its
rollback usually lands on an item this closer already closed.
`reopen_reverted_landings` runs beside the closer in autocode's housekeeping and
in `round board-pass`. It reopens an item that is `done`, whose
`automod_landed` names a reverted commit, and whose newest `item_landed` row is
this loop's own close of that commit. It moves the item to `up_next` (the
`rolled_back` re-offer) or `draft`, swaps the marker for `automod_reverted`, and
writes `item_reopened {by: rollback}`. An item a human closed is never touched.
The `rolled_back` re-offer names the commit to cherry-pick: the landed commit
outlives the rollback, held by the guardian tag, and a squashed round's history
is at `refs/automod/rounds/<round>`. No switch (`autocode.reopen_reverted`
was retired 2026-09-24). Why the guardian reset past the
newer promotion at all is #1358. `tests/test_backlog_unattended.py` pins it
with the incident's rows.

The acceptance is **clauses** since the review rung (§4.5): triage writes
`acceptance_clauses` (schema field, `ACCEPTANCE_CLAUSES:` numbered lines in
the text block, and the item's front matter), the implementer reports per
clause, and `parse_outcome` derives the overall word from them. **A deferral
that names no id is `not_met`.** #544 declared `deferred` with
`deferred_to: []`; the sweep wrote "deferred to an unnamed follow-up; close
this when that closes" and `desired_statuses`' landed branch parked it
`in_progress` with nothing ever re-reading the empty list. A landed round
whose own outcome says a clause is `not_met` is offered **once more for
exactly those clauses** (`implement_outcomes` → `partial`, cap 1); the branch
is gone with the landing, so that round starts from live main.

Only the round judged the acceptance, and it is asked in a grammar rather
than read out of prose. Everything else is noted and left open: `deferred`
names the ids it waits on, `not_met` says so, and a round from before the
finalizer says "a human decides". A closed item is never re-triaged, which is
the whole reason not to guess — and the reason the prompt tells the model
that `met` on an unverified acceptance is the one claim the loop cannot
recover from. Items confirmed before clauses existed carry prose only; every
reader (`acceptance_clauses_of`) treats that prose as one clause rather than
refusing the whole current pool.

**A vault item has a second judge, and it is used when the round has none.**
#575 (2026-09-14) landed its fix through `automod_vault_land` at 21:25Z; the
vault review graded all five clauses `met`, and so did three code reviews
after it. The item was re-offered twice and a third round was stopped by
hand, because four things combined:

- The contract and the prompt both read as a code round for every surface.
  Triage ended every clause "— tests/<file>.py", and the implement prompt
  said `automod_start`, a test pinning each clause, the gate — so the vault
  turn cut a round for a test file about a script `~/lloyd` does not own, and
  8 of the first 15 vault turns had done the same. A `vault` clause now ends
  with the vault path that shows it, and a `vault` item's prompt carries
  `VAULT_SURFACE_RULES`: a vault change is not a round, the vault review asks
  for no test in `~/lloyd`, and a pinning test is a finding.
- `settled_landings` read a finished row with a `round_id` only as a code
  landing, so a vault landing beside a round that never promoted was
  invisible. It counts now for a `vault` surface. A promotion of the round
  that has not settled holds it; once settled, the code landing is recorded.
  A landing still waiting for idle has no `promoted` row yet, so the vault
  landing is recorded first and the later promotion is not processed again —
  right for a `vault` contract, which the review graded against the vault.
- A turn dead at `max_turns` gets no finalizer, so the landing had no
  outcome. The vault grader's per-clause verdicts now ride on the
  `vault_land` event (`review_clauses`, one row per contract clause,
  recorded only on a `pass`), and `vault_review_outcome` builds a `met`
  outcome from the **newest** of the turn's vault landings when every row is
  `met` — `partial`, `not_met`, `post_landing`, a gap, a newest landing the
  grader did not pass (a 503 during a drain) or a reverted commit all answer
  nothing. It fills a missing outcome only, on a `vault` surface only, on
  every path (no round, a round, a settled promotion); an outcome the turn
  reported is never overridden, and a landing beside a round closes on a
  reported `met` only when the review agrees. `item_landed` carries
  `acceptance_source: vault_review | round`. The wall-clock exit
  (`TurnTimeout`) now records `vault_commits` and `surface` too, or the same
  landing was invisible by that door.
- The sweep ran only in housekeeping, once per `interval_seconds`, while
  `REPOLL_ON_COMPLETE` asks for the next round at once: the turn-end reconcile
  put #575 back in `up_next` at 22:06:38Z and the next round took it at
  22:08:44Z. `execute` now sweeps at turn end, before the reconcile —
  only after a turn that made a vault landing, because the sweep walks the
  board on the event loop (~2.3 s measured) and a code landing has the
  guardian's window to wait out anyway. It is synchronous like the reconcile
  beside it, and `close_settled_items` takes a module lock, because
  housekeeping sweeps in a worker thread at the same time.

Landings recorded before `review_clauses` existed carry none, so none of
them changes. `tests/test_vault_surface_churn.py` pins all of it.

### 3.2c Bounding the board: the loop must close more than it opens

Quarantine bounded one edge. Over 2026-09-07 → 09-11 the board still took
**453 new items against 49 closed**: triage filed 1.9 per run and closed
0.46; implement rounds filed 2.1 per turn and closed 0.12, seventeen for
one; 84 parent items had 282 children linked only in prose, and a
re-offered round re-derived and re-filed the same findings (#549 ran four
times in 110 minutes and filed ten children, three of them one finding).
Both closers were one item wide and both producers were unbounded. Six
mechanisms, landed 2026-09-11 across three commits — `03e98d8` the inflow
cuts, `30be051` the nightly clustering pass, `eb224d5` group triage and
umbrellas — and none of them deletes anything.

**Inflow is cut at the source.**

- **`backlog_write_task` checks the board before it writes**
  (`agent_mcp/backlog_similar.py`). The triage prompt used to say "run
  `backlog_tasks` to be sure no item already covers it" — a tool with no
  text search that returns ~800 titles. Now every create runs the qmd
  daemon's reranked vector search over the `backlog` collection it already
  embeds, plus a lexical Jaccard over the item heads on disk. A create
  tagged `spawned-by-*` whose finding an open item already covers is
  **appended** to that item under a "Merged finding" heading (the activity
  log names the session) and the result carries `merged_into`; a human's
  write is only ever advised (`similar`). The reranker score alone never
  merges — an unrelated query still scored 0.75 on its top hit — so rule A
  needs the lexical leg to agree, and rule B (a strong lexical match on an
  item created in the last ten minutes) covers the qmd watcher's debounce.
  `umbrella` and `blocker` writes are never merged; `force: true` bypasses;
  every decision is logged to `dedupe.jsonl`; it fails open. Config
  `backlog.dedupe`, `merge: false` is observation mode.
- **A finding an implement round turns up goes onto the item it came from**
  (`## Findings (round …)`, appended with `backlog_write_task`), counted off
  the file into `findings_appended`. Only a blocker — a finding that stops a
  clause of the round's contract from becoming true — becomes an item, one
  per round. The first round under the new prompt (#578) appended ten
  findings and filed none.
- **A re-offered round is told what earlier rounds filed** (`prior_spawned`
  in `_reoffer_block`) and told to append rather than re-file.
- **A single triage appends as well** (2026-09-13). After the cuts above the
  board still netted +48 and +47 on 09-11 and 09-12, and scorecard row 4 read
  triage at 0.989 items filed per item closed: 80 of the first 100
  `confirmed` single triages filed at least one item (0/1/2/3/4 in
  20/34/29/15/2 runs) and not one row carried an append. The prompt was
  formatted with six fields — nothing about tags, parent, who filed it or a
  group `keep` — and step 6 told every verdict "the item you are triaging is
  about to be closed", which is false for `confirmed`. Step 6 is now the
  implement rule: `confirmed`, `unverifiable` and `not_code` append each
  finding to the item and file nothing; `stale`/`already_done` send a
  survivor to an open item that covers it, then the parent, then at most
  `spawn_cap` (1) new item. `render_prompt` carries an `<origin>` block
  (tags; parent and its status; `spawn_origin` off the ledger's
  `backlog_triage.spawned`/`backlog_implement.spawned`/`arch_review.filed`
  rows; the group `keep` with its date; `prior_triage_spawned`, `incomplete`
  rows included; the count of `## Findings` sections). `findings_appended` is
  counted off the file before `record_verdict` rewrites it, and the overshoot
  tolerance lost its `+1` with the overflow item.
- **Spawn accounting is mechanical**: `max_item_id()` before the turn, and
  `split_claimed` reads an id at or below it as a merge, above it as a
  spawn — #370's finished row had listed itself and a pre-existing #221 as
  spawns. Both event types carry `merged` and `id_floor`.

**Expiry is the hard bound.** `backlog.expire_stale_spawns` runs in
autocode's housekeeping and closes a self-filed `draft` that nothing
triaged, implemented, clustered or tagged in `spawn_expiry_days()` —
`done`, tagged `expired`, text kept. The bound is
`workers.sources.autocode.expire_spawns_after_days`, 7 since 2026-09-14 (14
from 09-13, when it still had never fired: the oldest open self-spawn was 7.0
days old),
failing open to `SPAWN_EXPIRY_DAYS` (30; `SPAWN_TRIAGE_MIN_AGE_DAYS` is its
old alias). At a hardcoded 30 it had never triggered — the oldest self-spawn was
six days old — while 400 self-spawned items sat open, and `over_bound 0`
read healthy throughout. Expiry, autotriage's skip summary and the scorecard
gauge all read the one function, or `over_bound` would be measured against
a different bound from the one expiry runs at. A human setting its status back to
`draft` reopens it: the reconciler strips the tag and `expired_ids` keeps it
released, so it is never expired twice. Never `grouped`, `umbrella` or
`needs-human` items, never a human's draft. Scorecard row 4 carries the open
self-spawned count and `over_bound`, which should read 0. Kill switch
`workers.sources.autocode.expire_spawns`.

**Clustering puts the split items back together.**
`scripts/automod/cluster.py` groups the open drafts by three signals already
on disk, deterministically and offline: cosine over the chunk-0 vectors qmd
keeps for the `backlog` collection (read through qmd's own bundled
`vec0.so`, 0.12 s for the board; chunk 0 only, because mean-pooling pulls
long items toward the corpus centroid); shared file paths named in
backticks, by basename; and a common `parent`, parsed from the prose first
line and persisted to frontmatter once. A shared parent is an edge on its
own — #549's twelve children are one consolidation job whatever their
pairwise cosine, and a cut that demanded a cosine to confirm it dropped that
family entirely. A giant component is **peeled** into hub-centred groups of
at most 12, never trimmed (trimming dropped most of the board from the
output); a pair is a cluster. The optional pair-judge (the secondary,
priority 2, cached by body hash in `cluster_judgments.jsonl`) adjudicates
only ambiguous edges and an error keeps the edge. Quarantine is deliberately
not applied — its question is staleness, this one is sameness. The
`backlog-cluster` worker source runs it nightly ("nightly" = the output is
older than `min_age_seconds`, so a restart never doubles it up) and writes
`clusters.json` in the state dir; `round cluster --write` runs it by hand.
Since 2026-09-14 it also rebuilds a used-up file early — at least
`exhausted_min_age_seconds` (2 h) old, with nothing `select_cluster` would
take — and the ledger row carries `trigger: nightly|exhausted`;
`clusterable_items` leaves out ids a group triage already judged, so a
rebuild offers only fresh items.
First live run: 53 clusters over 322 of 407 drafts, 200 pairs judged, 291
parents persisted.

**Group triage judges a cluster per run.** `autotriage` takes a cluster
before it takes a single item (`backlog.select_cluster`: re-validated
against disk, ids a group run already judged dropped, duplicate pairs kept
together, largest surviving cluster, `group_min_items` 2 /
`group_max_items` 4, 8 until 2026-09-14). One turn, `GROUP_PROMPT`, `GROUP_TRIAGE_SCHEMA` built
from `RETIRING`/`SURFACES`, per-item verdicts:

- `duplicate_of #t` → `done`, `duplicate_of` in frontmatter, a `stale`
  triage row. Chains resolve to the terminal survivor against the verdicts
  *as given* (resolving against the rewrites in progress let 4→5→4 come out
  as "duplicate of a keep"); a cycle or a target outside the cluster is
  `keep`.
- `stale` / `already_done` → as today.
- `fold` → `group: <umbrella>`, tag `grouped`, stays `draft`; ledger verdict
  `folded`, outside `VERDICTS` so `triaged_ids` does not count it as judged.
  A member is out of both pools and the reconciler parks it `draft` whatever
  else the ledger says.
- `keep` → an activity note, no triage row, and the id is released from
  quarantine so single triage reaches it.
- The **umbrella** the turn filed (tags `umbrella`, `spawned-by-triage`) is
  confirmed through `record_verdict` exactly like any confirmed item —
  `up_next`, `acceptance_clauses` on disk, ≤`MAX_CLAUSES` (6) — plus `members`. Two
  folds minimum: one fold is a keep. No umbrella on disk turns every fold
  into a keep and records `umbrella_missing`.

One `backlog_triage` row per member and one `backlog_group_triage` summary
(`judged: {id: verdict}`), so the scorecard, the status pipeline and
`triaged_ids` see ordinary verdicts. Unparsed degrades to `keep`, never to a
close; budget exhaustion is `incomplete` once and `abandoned` the second
time, writing nothing on the items. The first live run (c-6047a94c, eight
items, twelve turns, structured): three `already_done`, two folded into
umbrella #858, three kept. Kill switch
`workers.sources.autotriage.group_triage`.

**Implement stays one item per round**, so the gate, the review rung and
the rollback unit are untouched. An umbrella is an ordinary confirmed item
whose prompt carries `<member>` blocks and whose review contract appends the
members as context under its own clauses. When it settles `met`,
`close_settled_items` closes every still-open member with "landed via
umbrella #u" and an `item_closed {by: umbrella}` event
(`close_members_on_settle`); `not_met`, `deferred` and no-outcome leave them
folded; `unnecessary` closes the umbrella but tags it `needs-human` with the
members still folded — a wrong `unnecessary` on six findings is the one
claim the loop should not make alone. `backlog.unfold_umbrella(id, reason)`
is the human escape hatch; `unfold_spent_umbrellas` applies it to every open
umbrella whose attempt is spent (§3.2d). Scorecard row 11 counts all of it; folds and
duplicates count as closures on row 4.

**Single triage pauses while the implement pool is full** (2026-09-13).
Triage confirmed 59 items on 09-12 and 28 by 08:00 on 09-13; `record_verdict`
moves each to `up_next` unconditionally and no code counted the depth, so 78
of 89 `up_next` items had never been attempted. After the cluster pick and
before `select_candidate`, `autotriage.execute` computes
`backlog.ready_confirmed` — the readiness filter `select_confirmed` used to
hold inline (up_next, not grouped, acceptance present, not human-only, not
spent), extracted so the gate counts exactly what autocode would take — and
`implement_pool_bound`: `max(implement_pool_floor, landed_items_trailing(7))`.
The trailing count is **distinct items** with a settled code landing or an ok
`vault_land`, not rows, because a re-offered item lands repeatedly. At or
above the bound the run was `skipped` with both numbers in the summary and no
ledger row; group triage kept running, since it is net negative on open
items. `implement_pool_floor` (20) and `spawn_cap` ride in the queue payload
like the budgets. On landing day it read 87 ready against a bound of 55.

**…and since 2026-09-14 it holds the confirmation instead.** Skipping the run
stopped triage's retirements with its confirmations — 23 `stale` /
`already_done` closes the day before, the loop's largest closer — and in the
gate's first ten hours the loop filed 6 items and closed 3, with ready (103)
too far above the bound (58) to reopen for days. Now the run always goes
ahead. When the verdict is `confirmed` (and not human-only) the gate is asked
again at record time (`backlog.implement_pool_full`, since a turn runs for
minutes); if full, `record_verdict(hold=True)` leaves the item `draft` tagged
`confirmed-held` and the ledger row carries `held: true`.
`backlog.held_confirmations` reads that row minus any later
`backlog_confirm_released`; `desired_statuses` keeps a held item `draft`;
`release_held_confirmations` moves held items to `up_next` oldest first while
`bound − ready > 0`, from the top of every triage run and from autocode's
housekeeping (which reads triage's config block, so a disabled triage cannot
strand them). A held item a human moved out of `draft` is released where it
stands (`reconcile_statuses` runs that first). Group triage passes `hold` to
`record_group_verdict` for its umbrella. `hold_confirmations: false` restores
the skip and releases everything held.

**One definition of the board's shape.** `backlog.board_health` returns the
open counts, a `draft` partition (grouped > needs-human > held > triaged >
quarantined > pool, each item once), `up_next`
(umbrellas/singles/never-attempted/ready/unready), 24 h and 7 d flow
(`board_flow`: `created` in, `completed` else `updated` out — `record_verdict`
now stamps `completed` on a close like the other closers; each stamp is read
in its writer's clock, `created` and a `completed`-less `updated` as local
time, `completed` as UTC), the open
self-spawned count and the pool bound. Three readers: the dashboard's backlog
panel (a `health` sub-object on its own 60 s cache, `by_status` left raw),
scorecard row 13 (board net flow, plus `spawn.triage_findings_appended`), and
the board steward's `<board_health>` block, recorded on its ledger row. No
announce: board health is a dashboard and scorecard number.

Triage polls every 15 minutes again (it was raised to an hour on 09-08 as
the second half of the quarantine fix, for a pass that filed two items per
item it read). It never opens a round, so it may overlap an autocode round
or its observation window; two pool slots and the KV gate arbitrate for the
primary. Autocode's own gate (`_loop_is_free`) is what keeps two rounds
apart: no worktree open, no promotion under observation (an `observing` one
is allowed with `automod.chamber` on, §3.2d), no rollback pending, checked at
enqueue and again at run.

### 3.2d Throughput: the loop must not idle between rounds

Measured over the week to 2026-09-14 (`promotions.jsonl`, `workers.db`,
`logs/server.err`): 137 rounds, 54 promoted, 48.3 h of autocode turns
running and **61.6 h idle between a run's end and the next enqueue** — 31.7 h
of landing plus observation window (46 landings, median 20.9 min), ~20 h of
an enqueue race (56 gaps, median 12.6 min), ~7 h of reaper grace (15 rounds,
median 26 min). And of the rounds that ran, the review rung refused 109 of 147
reviews. Five changes, each with a switch.

**Pacing.** Three idle sources, three fixes:

- *The coalesce race.* `enqueue_if_due` returned `DECLINED` only when
  `_loop_is_free` said no. When the model called `automod_abort` (worktree
  gone) and then spent one to three minutes on its finalizer, the loop read
  free while the round's queue row was still `running`, so the enqueue
  coalesced on `autocode:round`, the source returned None, and the pool
  stamped a full `interval_seconds`. A coalesced enqueue returns `DECLINED`.
- *Nothing woke the source.* A source with `REPOLL_ON_COMPLETE = True` has its
  `last_enqueue_check` back-dated to the epoch in `WorkerPool._worker_loop`'s
  `finally`, so the next 60 s scheduler pass asks it. Autocode opts in; it
  covers a `skipped` run that took milliseconds and a round shorter than the
  interval.
- *The reaper's grace.* See §3.2 item 3. The grace existed for the observer's
  ambient follow-up, which cannot come with `autocode.inner_voice: false`, and
  27 of 136 turns ended with the round open. `_run_and_record` calls the
  reaper (off the event loop) after the `finished` row and in the
  `TurnTimeout` branch, passing the session that just ended so its own
  wind-down is not read as busy; not on `infra_failed`. Housekeeping's 900 s
  pass stays the backstop. `land.running` is the gate marker's twin:
  `_land_detached` writes it with the child's pid before `automod_land`
  returns, `round.land` rewrites it with its own pid and clears only its own
  in a `finally`, a dry run never touches it, and a second landing of one round
  is refused.

**The chamber** (`automod.chamber`, **ships false**). `_loop_is_free` refused
a new round for a promotion's whole window, though only `land` needs it — the
turn and gate touch the worktree and the canary ports, and triage and digest
turns already run on the live backend during observation. With the switch on
an `observing` promotion is free; `landing`, a rollback request, halt and
BROKEN are not. The next round's turn and gate (~33 min median) then hide the
observation window entirely (7a; 15 min until 2026-09-20), and only its
landing waits:
`promote.wait_for_settle`, called by `round.land` *outside* the automod lock
(round starts and arch-review commits take it) and before `wait_idle` pauses
the pool, polls `current.json` every 10 s for up to `settle_max_wait()` (the
larger observation window + 120 s slack; see 7a),
then asks halt, BROKEN and the rollback request again. Each way it ends
without a landing is an external `land_failed` (`waited_for_settle: true`),
so the item keeps its attempt; `promote`'s "still under observation" refusal
stays behind it for any caller that did not wait, and a moved `main` takes
the existing `_regate_after_move` path. Accepted failure mode: a rollback of
the observed promotion restarts the stack under the next round's turn, which
ends `infra_failed` and is re-offered as `infra`. It ships off because a round
waiting to land is exactly what the turn-end reaper must not touch, and that
wants a few days of `land.running` in production first. True two-wide
concurrency was considered and not taken: five single-writer resources
(`current.json`, fixed canary ports, the worktree count, `S.Lock`,
`select_confirmed`) would need sharding, and two 150k-token turns contend for
one primary's KV pool.

**The clause budget.** The per-clause not-met rate was 13%, so six clauses
pass together ~43% of the time and twelve ~19%; first reviews by clause count
promoted 3/16 at ≤4, 21/52 at 5–8, 3/11 at 9+; 62 confirmations in the week
carried 9–12. `MAX_CLAUSES` 12 → 6, capped on all four parse paths (single
and group, regex and structured) through `cap_new_clauses` and again in
`record_verdict`, with `clauses_dropped` on the verdict row;
`SINGLE_MAX_CLAUSES` (5) is interpolated into the single prompt; each clause
ends with the test file that pins it (the grader downgrades a `met` with no
test node). A contract already **on disk** is read at `READ_MAX_CLAUSES` (12):
56 umbrellas carried 8–12, and capping the readers would grade a round on
half its contract, close the item on the half it met, and let `amend_clause`
write the truncation back. `group_max_items` 8 → 4.

**Spent umbrellas unfold.** Umbrellas were 56 of 92 `up_next`, 5 of 68 formed
had landed, and 156 members sat `grouped` — expiry-exempt, triageable by
nothing. `unfold_spent_umbrellas` (autocode housekeeping, before the
reconcile): an open umbrella whose outcome is `spent` and that carries no
landed marker is unfolded, closed `done` tagged `unfolded` with
`needs-human` removed, and a `backlog_umbrella_unfolded` row written. Members
are not re-clustered (`group_triaged_ids`) and self-filed ones stay
quarantined and expire; `grouped` stays in `EXPIRY_EXEMPT_TAGS`, because a
member of a live umbrella closed by expiry would be misattributed by
`close_settled_items`. No switch (`autotriage.unfold_spent_umbrellas` was
retired 2026-09-24).

**One automatic second life.** A spent attempt parked the item for a human:
~35 a week, and 42 of the 67 a person reopened later landed.
`retriage_spent_items` (autocode housekeeping, after the unfold, before the
reconcile) sends an item back through *triage* — because what failed was
usually the contract — once. Eligible: outcome `spent`, latest row not
`started`, not an umbrella or member, no live promoted round and no landed
marker (a landed `met` item owing human clauses is a person's), fewer than
`RETRIAGE_CAP` (1) prior re-triages. It is moved to `draft`, tagged
`re-triage`, loses `needs-human` and `review-disagreement`, and gets a
`backlog_retriage` row with the refused round, the review's findings, its last
per-clause verdicts and `unmet_twice`. That row is a **mark**:
`retriage_marks` feeds `triaged_ids`, `confirmed_verdicts` (and through it
`held_confirmations`), `incomplete_counts`, `implement_outcomes`,
`review_events_for_item`, `last_review_all_met` and the review rung's
`prior_reviews`, which all ignore rows at or before it; `released_ids`
includes it. The item is therefore untriaged, unattempted and unquarantined,
and oldest-first puts it near the front. Single triage's `<origin>` carries a
`RE-TRIAGED` line and its prompt says: a new contract without the
twice-refused clauses, within the cap, or retire. The second spend parks for
a human as before. `implement_outcomes` also resets its attempt count at a
human `reopen_item` now — it reset only the latest row, so a reopened item's
first new round could already be past every re-offer cap. While a re-triage is
still owed, a review disagreement is not announced as "needs you". Switch
`workers.sources.autocode.retriage_spent`.

**What the first review of this section found** (same day, before landing).
`spent` is not "over": `implement_outcomes` reads it for a turn in flight, for
a round whose landing is still waiting for idle or re-gating, for an observed
round and for a promotion the settle sweep has not reached. The unfold pass had
no guard at all and would have closed nearly every umbrella under its own
round; re-triage guarded only two of the shapes. `items_with_unfinished_rounds`
is the one guard both read. `implement_history` drops a `started`/`skipped`
pair (a landing drain refused the turn) so a turn that never ran is not a
spend. `desired_statuses` keeps a marked item `draft` until a post-mark
confirmation, or a reopen or a raced landing would have stranded it in
`up_next`. The re-triage mark clears the refused contract from the front
matter. The review-disagreement escalation had run before the `finished` row,
read an empty detail and never fired; it runs after it. The enqueue asks the
queue for a live round row (`WorkQueue.has_live`) before the two-second board
walk. With the chamber, a wait that ends with `current.json` clear but no
`settled` row for the commit waited on is a rollback, not a settle, and
preflight rebases `--onto <live> <base>` so a reset-away base is never
replayed. Smaller: dropped clause text is kept (row and item), the writers
coerce string `tags`, a zombie child's marker reads dead, the reaper reaps
nothing when it cannot read the session snapshot, and a YouTube filing merged
into another item verifies.

**youtube-eval is loop output** (Alan, 2026-09-14). `EVAL_SPAWN_TAGS`
(`youtube-eval`) is in `LOOP_SPAWN_TAGS` (expiry, gauge) and a new
`QUARANTINE_TAGS`, with `SPAWN_TAGS` itself unchanged; the MCP dedupe merges a
`youtube-eval` write like a `spawned-by-*` one. Readers call
`quarantine_tags()` / `loop_spawn_tags()`, which honour
`workers.sources.youtube-digest.loop_spawned`. 125 were filed in the week and
101 were open. Quarantine as well as expiry, because a confirmed eval item
would become `confirmed-held`, which expiry exempts. The scorecard gauge now
coerces tags through `normalize_tags`, since the digest has written them as a
string.

### 3.2e The sweep: switching gears to read the whole board once

**The measurement (2026-09-15, 7 d of `promotions.jsonl` and `workers.db`).**
Triage was not the slow part: 50–66 single runs a day at 216 s each, 73%
`confirmed`. Implement was: 178 rounds ended, 57 promoted (32%), the first
review refusing 74%, 88 h of 168 h idle between implement turns, ~10.6
distinct items landed a day (5 of them vault). Confirmations therefore
arrived ~3× faster than landings — 156 confirmed items queued (87 ready, 69
held), 64 of 88 `up_next` never attempted — and the board could not be
"caught up" by implementing. Two structures made it worse: 56 of the 88
`up_next` were umbrellas, every one with 8–12 clauses from before
`group_max_items` fell to 4, landing 5 of 27 rounds against 52 of 151 for
singles, with 158 members folded under them; and 82 quarantined drafts had
never been read by anything, because quarantine's only exits were
clustering (exhausted), a group `keep`, a human, and expiry — 27 were due to
close unread within 24 h, 29 of the 82 were `youtube-eval` ideas. Expiry,
the bound §3.2c added, had become the mechanism by which the board lost
sight of work.

**The shape.** "Caught up" means every item read and ranked by a reader,
within days, and the loop then working the best items first while the long
tail waits in the open. Four moves, each with a switch:

- *Sweep mode* (`workers.sources.autotriage.sweep`, ships off). While
  `backlog.sweep_pool` is non-empty, `autotriage.execute` takes
  `sweep_batch` (8) items in one turn before it considers a cluster or a
  single item. The pool is every open `draft` or `up_next` item not yet
  swept or parked, not grouped, not an umbrella, not `needs-human` or
  `expired`, and not in a batch abandoned twice — quarantine does **not**
  apply, since reading a self-filed item once is the point; never-judged
  items first, then oldest. `SWEEP_PROMPT` asks one question per item —
  is it dead? — answered from the text or one cheap check, and otherwise
  `keep` with `worth` (high / medium / low, the rubric in the prompt) and
  `size` (small / medium / large). The turn runs under `SWEEP_DISALLOWED`
  (no `Edit`, `Write`, `backlog_write_task`, `vault_write`, `Task`, the
  automod and autonomy tools), so it is read-only by construction rather
  than by instruction, and it files and appends nothing. `parse_sweep_verdict`
  reads the finalizer's `SWEEP_SCHEMA` object or the `SWEEP_VERDICTS:` block;
  an unlisted member is **not** filled in as `keep` — it stays unswept and is
  offered again, because a rank the turn never wrote is not a rank.
  `record_sweep_verdicts` writes retirements through `record_verdict` with an
  ordinary `backlog_triage` row (`sweep_batch` on it), so `triaged_ids`, the
  status pipeline and the scorecard see a normal close; `duplicate_of` may
  name any open item on the board, resolved by `_resolve_duplicates` with
  the open external targets treated as members, and a closed, parked or
  missing target makes it a `keep`. A `keep` writes `worth`/`size` and the
  `swept` tag; a `low` draft is also `parked`. A `low` confirmed item is only
  ranked — the implement order sorts it last — because parking a
  confirmation would fight the reconciler. The summary row is `backlog_sweep`
  (`batch_id`, `judged`, `ranked`, counts). A batch that reaches no verdict
  block is `incomplete` once — the same batch is re-selected, its items
  being still unswept — and `abandoned` the second time (`item_ids` on the
  row, read by `sweep_abandoned_ids`), after which those items are left to
  the ordinary passes.
- *Parked replaces expiry as the exit.* `parked` is out of `triage_pool`,
  `select_candidate` and `select_cluster`, counted as its own `draft`
  bucket in `board_health` and on the dashboard, and in `EXPIRY_EXEMPT_TAGS`
  with `swept`. A person promotes it by removing the tag. Expiry
  (`expire_spawns_after_days`) went 7 → 30 the same day and reaches only
  what the sweep has not read.
- *The rank orders every pool — under the human's priority.* `Item` carries
  `worth`, `size` and `clause_count` from front matter; `rank_key` is
  (worth, size) with unranked between medium and low. Since 2026-09-16
  `priority_key` (the item's `priority`: high, medium, low; anything else
  low) sorts ahead of it everywhere, on Alan's rule that the pace is
  tolerable if the tag is honoured. `select_candidate` sorts by live
  blocker, priority, rank, age; `select_confirmed` by priority, near tier,
  rank, clause count, then the existing keys; `release_held_confirmations`
  fills room by priority then rank; `sweep_pool` by never-judged, priority,
  age; `select_cluster` takes the cluster whose best member has the highest
  priority, largest among equals, and keeps high members through the trim.
  Across the two triage pools, a single candidate that outranks every member
  of the cluster group triage would take runs first (`priority_beats`, in
  `autotriage.execute`; the cluster is the fallback if the depth gate pauses
  single triage). The default on both writers is `low`, so a value nobody
  chose never outranks one somebody did; `round priority-backfill` wrote it
  onto the files that had none.
- *A `high` item is picked up next* (`is_high`, `select_urgent`). Alan's
  ask: submit a high item and have it taken next. In triage it goes ahead
  of a sweep batch (`sweep_pool` never lists it), ahead of any cluster,
  ahead of a live blocker, and past the depth gate's pause with holding
  off; its confirmation is never `held` (on 2026-09-16 the pool read 73
  ready against a bound of 73, so a held high would have waited for
  never), and `release_held_confirmations` moves a held item a person
  raises to high without waiting for room. In the implement pool it is
  first by `priority_key`, and **within `high` the newest goes first**
  (`recency_key`), the sweep's rank, the contract length and
  fresh-before-re-offer not applying: the board carried 87 open highs that
  day, 17 ready, most months old, and oldest-first would have queued the one
  just raised behind all of them — and on 2026-09-17 the fresh-first key
  alone put that item, re-offered with its branch, 27th behind 26 untried
  highs. Only the near tier precedes recency (a high one fix cycle from
  landing).
  Latency is one autotriage interval to the contract, then the next round.
  `round priority-backfill --reset-open` is the stronger reading of
  "default low for existing items": every open item written `low`, the old
  value in its activity line, so the high tier starts empty; human-only.
  `tests/test_autotriage.py` ("picked up next") and
  `tests/test_backlog_priority.py` pin it.
- *Swept ids are in `released_ids`*, so a swept self-spawn enters single
  triage. The single prompt's `<origin>` says the sweep's rank so the
  contract fits the size.
- *2026-09-17, what #1199's three rounds taught.* (1) A worker turn may
  not restart or stop an engine or a service: `app/harness/service_control.py`
  is the fourth check in `safety.check_bash_command`, keyed on the session
  id like `_tool_sandbox`, parsed like `protected_paths`, read-only verbs
  allowed, chat sessions never refused. (2) A tool-opened round no implement
  row names is an orphan: `round_start` carries `opened_by` and `session_id`,
  and `reap_abandoned_rounds` has a second pass that aborts one whose opener
  is quiet after `ORPHAN_ROUND_MIN_AGE_SECONDS`. (3) The implement prompt:
  a review refusal with fewer than 25 iterations left is `automod_abort`,
  never an edit or a `land`. (4) The triage prompt: a clause pins the
  change, never an invariant the tree does not already hold.
- *Autocode can yield* (`workers.sources.autocode.yield_to_sweep`, ships
  off): `enqueue_if_due` returns `DECLINED` while `sweep_pending` > 0, so
  the sweep runs back to back rather than in the gaps the round hold
  leaves it. It ran on for the sweep's first three batches (22:24–22:50Z)
  and was switched off on Alan's rule that an autocoder round runs 100% of
  the time. `autotriage` is exempt from the round hold instead
  (`workers.round_hold.exempt`): a sweep batch is ~40k tokens beside a
  round's ~150k on the 692k FP8 pool, and the KV gate still holds it over
  60% median KV. Measured before the change, 88% of the previous 24 h had
  an implement turn in flight, across 21 gaps with the largest 22 min.
- *Umbrellas.* `form_umbrellas: false` on autotriage appends
  `NO_UMBRELLA_RULE` to the group prompt and records a `fold` as `keep`
  with no umbrella confirmed, so group triage still closes duplicates and
  retires the stale while the pool is over its bound.
  `unfold_oversized_umbrellas` (`round unfold-oversized`, `--min-clauses 8`,
  `--dry-run`) unfolds every never-attempted open umbrella with at least
  that many clauses and closes it `done` tagged `unfolded`; a member's
  `folded` row is not a verdict, so it goes back to `draft` untriaged where
  the sweep reads it, and it does not re-cluster (`group_triaged_ids`). An
  attempted, landing or landed umbrella is not touched.

`round sweep-status` reports `board_health.sweep` (`unswept`, `swept`,
`parked`, `worth`), worth × size over the open items and the last batches.
Sprint settings to undo when `unswept` reads 0: `autotriage.interval_seconds`
300 → 900; `form_umbrellas` back on once `implement_pool.ready` is under
`bound`. `tests/test_backlog_sweep.py` pins the pool, the parser, the
recorder, the rank ordering in all three pools, the yield, the umbrella
switch and the unfold.

### 3.2f Depth: more than one round, more than one triage turn

Added 2026-09-17. The measurements that prompted it, over the seven days
before: 191 implement turns (median 24.3 min; two gate runs per round at
~8.5 min of `tests` + `review` each), 66 landed, 68.2 h idle between turns,
worker pool occupancy 0/1/2 jobs for 24% / 43% / 32% of the window at
`slots: 2`. Review refused 134 of 206 graded attempts: 61 with an unmet
clause, 27 with every clause met and only a skip-pattern precheck blocking.

| what | where | rule |
|---|---|---|
| depth | `workers.sources.{autocode,autotriage}.max_inflight` | claim cap, slot keys, tolerated worktrees; 1 when absent |
| slots | `workers.slots` | ≥ rounds + triages + 1 |
| landings | `round._land_lock` | queued, one at a time; chamber re-checked after the lock |
| suite, canary ports | `state.GATE_TESTS_LOCK_PATH`, `GATE_CANARY_LOCK_PATH` | queued per gate, `Gate.SERIAL_MAX_WAIT`; `review` overlaps |
| landing vs the other turn | `promote.wait_for_rounds` then `wait_idle` | phase 1 pauses nothing; `landing` state holds new rounds |
| idle budget | `automod.landing.idle_max_wait_s`, `idle_hard_max_wait_s` | budget burns only while the pool is empty |
| turn → round | `autocode._round_opened_since` | by `session_id` / `item_id`, not by time alone |
| triage selection | `backlog.claim_for_triage`, `autotriage._claiming` | in-memory claim under an `asyncio.Lock` |
| skip precheck | `review._skip_is_conditional` | conditional → advisory; unconditional → blocking |
| external landing failure | `backlog._external_budget_left` | capped on its own count |

What a second round costs: the second landing waits for the first's
observation window (the chamber), then rebases onto it and re-gates. So depth
2 is a pipeline, not a doubling — the gain is the gate time and the idle time
that now overlap another round's turn. Going back to 1 and 1 is two numbers
in config.yaml (and `slots` back to 3) and a backend restart; nothing else
changes shape. `tests/test_loop_depth.py` pins all of it.

### 3.2g Between the gate and `main`: a finished change is not lost

Added 2026-09-18, from one day read end to end: 22 rounds, 12 promotions, no
rollback — and **250 of that day's 994 implement minutes (25%) spent redoing
changes that had already passed every rung**. Six rounds passed their gate and
were then thrown away, by three mechanisms; and two items were closed
"tried it and rejected it" minutes before their changes promoted. None of it
showed as a failure anywhere: each lost round was re-offered, and landed.

**The gate outgrew the turn.** `backlog.gate_duration_stats` is the measure:

| day | full gates | median (s) | p90 | `tests` rung | `review` rung |
|---|---|---|---|---|---|
| 09-12 | 6 | 311 | 354 | 153 | 158 |
| 09-14 | 32 | 483 | 674 | 231 | 241 |
| 09-16 | 26 | 692 | 841 | 343 | 330 |
| 09-18 | 23 | 995 | 1234 | 476 | 401 |

The suite went from ~4,900 to ~5,830 tests and runs serially (no single hot
test: the slowest is 12 s); at depth 2 the second gate queues 400–470 s for
`gate-tests.lock`; the grader shares the primary with four turns. The turn's
clock stayed 3540 s, and the prompt still said "the first `automod_gate` by
minute 30; after minute 40 start nothing you cannot gate", written when a gate
took five minutes. The four rounds the clock cost opened their gates at minutes
37, 40, 42 and 45 — each as told. Three turns ended with the gate still
running; it passed 1, 5 and 7 minutes later. The fourth saw the pass with 146 s
left and obeyed the 90% anchor's "stop calling tools".

| what | where | rule |
|---|---|---|
| a passed gate whose turn is over is **landed** | `autocode._land_if_passed`, from `reap_abandoned_rounds` | gate `ok` at the commit the worktree still holds; the turn did not end on `unnecessary`/`rejected`; no `promoted`/`land_failed`/`land_rescued` row yet; not halted, not BROKEN, enabled. Ledger row `land_rescued`. Switch `workers.sources.autocode.land_passed_gates` |
| one definition of "start a landing" | `round.land_detached(round_id, by=)` | the checks, `spawn_detached`, the marker with the child's pid — `automod_land` and the reaper both call it |
| the reaper keeps up | `autocode.enqueue_if_due` → `_reap_quietly` | also on every look the loop declines (`retry_seconds`), not only at turn end and with housekeeping: a gate that finished after its turn sat up to 900 s, holding the other slots through `_rounds_about_to_land` |
| pacing is measured | `autocode._pacing_marks` → `{gate_minutes}`, `{first_gate_by}` | median for "a full gate takes about N minutes now"; `first_gate_by = clamp(turn − 1.5 × p90 − 5, 10, 30)` — the slow gate, one refusal, the half-length re-gate. 09-18's ledger: 16 and 23 |
| gate late rather than not at all | the prompt's pacing block | "commit and gate, however late: a gate that passes after your turn ends is landed by the loop, a refused one comes back with its findings"; "when a gate passes, `automod_land` at once"; "run the tests you changed, never the whole suite — the gate runs it" (22 turns launched the full suite 31 times that day; one slept 12 minutes on it) |
| the tool says how long | `automod._gate_minutes_note` in `automod_gate`'s RESULT | never in the tool's description: that is part of every turn's cached prompt prefix, and a number that moved with the ledger would re-prefill every session |

This is #278's rescue, made deterministic. That round died at its cap with the
change written, and the Inner Voice observer sent "if the gate passed, land it"
into the session. Cut 1 of senses-not-supervision (09-12) switched the observer
off for unattended turns and replaced its stall, budget and context nudges with
deterministic anchors — but not that one.

**A model that misreads a pass.** Four rounds in four days were aborted by
their author 20–90 s after passing every rung, each reporting a refusal that is
on no ledger row: #1131 (09-15, "both review attempts spent" on a 5-of-5 pass),
#1190 (09-16, "review refused clause 4" on a first-attempt pass and a green
rollback drill), #1053 twice (09-18, quoting "REVIEW refused: fixable
problems" — a phrase in no tool result of that session). The model had the
whole report each time: `transcript_entries.TOOL_RESULT_MAX_CHARS` clamps the
stored transcript, never what the model is sent, and the spill threshold is
50 kB against a 10–21 kB report. What it had was `gate.json` verbatim —
`"ok": true` on line five, and several hundred lines later the reviewer's
**advisory** findings, worded as a refusal's are ("cannot fail", "is still
grep-only"). `_gate_wait` added a `next` only for an external blocker, because
only that misreading had happened yet.

| what | where | rule |
|---|---|---|
| verdict first | `automod._with_headline` | `verdict` and `next` lead every finished report: PASSED → `automod_land` now; REFUSED at review → which attempt, and what that leaves; NOT JUDGED (external) → wait and re-gate; NOT GRADED; FAILED at `<rung>` |
| advisories are labelled | same | on a pass they move out of the review rung into `notes_that_did_not_block`, which says they did not block and are already on the item; preflight's `allowed` bucket (a second copy of `changed_paths`) is dropped |
| an abort asks first | `automod._abort_refusal` | refused once for a round whose gate PASSED at its current commit (`discard_passed_gate=true` to mean it), and always while the round's landing is running — `round.abort` removes the worktree. The CLI and the reaper call `round.abort` directly and are not asked |

**An item verdict from a round that is landing.** `unnecessary` and `rejected`
close the item at the end of the turn, and `parse_outcome` keeps them "as
stated whatever the clauses say". `rejected` was recorded twice in its first
two days and was wrong both times: #1242 (`landed: false`, no clauses, a
summary saying it "cannot state `landed: true` on evidence I do not have" —
its own `automod_land` was in flight) and #1053 (`landed: true`, four clauses
`met`, empty summary). Both landed. `close_settled_items` walks open items, so
neither landing reached its item, and had either promotion failed the item
would never have been offered again. In a third turn that day the finalizer's
`summary` invented a review refusal while the review was still running.

| what | where | rule |
|---|---|---|
| the verdict carries its own evidence | `backlog.settle_item_verdict`, called by `autocode.execute` before the close | `unnecessary` stands only when nothing landed (the schema: "close WITHOUT a landing"). `rejected` stands only with a `summary` of at least `MIN_REJECTION_SUMMARY` (25) characters AND either nothing landed or a clause reported `not_met`. Either is refused when the ledger shows a landing and the outcome says `landed: false`. Acceptance re-derived from the clauses, `""` with none. `outcome_refused` on the `finished` row |
| what "landing" means | `autocode._landing_seen`, or a vault commit this turn made | the live land marker, `current.json`, a `promoted`/`land_failed`/`land_rescued` row, or `vault_commits` — never the turn's word |
| the review stands in for a missing outcome | `backlog.code_review_outcome`, in `settled_landings` | the vault rule (§3.2b) for the other surface: newest graded review of the round, not blocking, clauses 1..n all `met`. A reported outcome is never overridden |
| said where it is read | `automod_land`'s result (`outcome`), the finalizer's own prompt | what `landed` means costs the length-bounded template nothing; the template gained one sentence — a clause is judged on the change, "never that it is not promoted yet" (two turns reported every clause `not_met` for a finished, gated change) |

What still stands: `unnecessary` with nothing landing, and `rejected` with its
measurement — including a round that landed its instrument and rejected the
idea (`landed: true`, a clause not met, the numbers in `summary`).

The first cut of that check listed contradictions, and a third false `rejected`
walked through it the same evening: #982, a vault landing the vault review had
just graded 4 of 4 met, whose finalizer wrote `landed: true`, no clauses, and
the summary `"placeholder"`. Not empty; no clauses to be all met; and a vault
turn has no round for a landing to be seen on. All three were the structured
finalizer degenerating at the end of a long turn (333–865 tokens against ~1,000
for a real outcome), and a degenerate object is still schema-valid — so the
rule asks what a closing verdict SHOWS rather than what it contradicts.
Replayed over the ledger: three item verdicts ever recorded, all `rejected`,
all false, all refused; `unnecessary` has never been used.

**A deferral waits for what it deferred to.** `backlog._open_deferral_targets`:
`retriage_spent_items` skips an item whose last round deferred to a still-open
item. #1069 deferred to blocker #1242, was re-triaged 25 minutes later on a
tree that still had the obstacle, got a `human-only:` contract citing #1242 —
and #1242 landed two hours after that with nothing left to read #1069 again.

Scorecard row 9 counts `landings_rescued` and `item_verdicts_refused`. Tests:
`test_reaper_lands_passed_gates.py`, `test_gate_report_headline.py`,
`test_outcome_item_verdict.py`, `test_prompt_pacing_and_ordering.py`.

### 3.2h A landing must not starve the rounds beside it (2026-09-19)

Depth went to four rounds at a time on 2026-09-18, and that night's ledger
(18:00–08:00 PDT) shows what a landing cost the other three:

| | |
|---|---|
| rounds started / promoted | 44 / 19, no rollback |
| review attempts that could not run | 17 of 51 (16 `HTTP 503 … Lloyd is landing a code update`, 1 timeout) |
| minutes the backend spent draining | 189 of 840, in seven spans of 12–36 min; every other landing drained < 2 min |
| rounds aborted with the change finished and green | ~11 — #832 took five rounds, #800 four, #1250 #789 #874 three each |
| promotions that changed nothing either service had loaded | 9 of 19 |

**The cycle.** A landing arms the drain and waits for every turn in flight. A
sibling round's turn is waiting on its detached gate. That gate's review rung
needs a grader turn on the live backend, which the drain refuses with a 503.
Nothing moves until the grader's 420 s give-up — usually twice — after which
the turn, told "the grader, not the diff", writes its report and stops with
the round open, and the reaper aborts it. `wait_for_rounds` (§3.2f) exists to
keep a landing out of the drain while sibling turns run, and did not hold: it
took ONE unanswered 5 s probe of `/api/workers/status` as "the backend cannot
say" and went straight on. At 21:45:58 a post-session capture was on the event
loop; the drain armed at 21:46:13 beside three live turns and stayed armed for
24 minutes. Nothing recorded what the wait had seen.

Four changes, each with its own switch or none needed:

1. **The drain admits a review grader while something else is running**
   (`app/routers/automod.py::drain_admits`). Only a session the review rung
   minted (`source: automod-review`, a non-user platform), and only while
   `active_turn_summary()` is non-zero. The second half keeps the drain's
   guarantee: the promoter restarts after N consecutive QUIET polls, so a
   grader admitted onto a quiet backend could be the turn the restart kills,
   while one admitted beside a running turn resets the count and is itself
   counted until it ends. Admitting it can only shorten the wait — the landing
   is already waiting on the turn that is waiting on the grader. Fails closed.
2. **A round whose only failed rung was an unreachable grader is gated again,
   not aborted** (`autocode._regate_if_unreviewed`, ledger `gate_rescued`,
   switch `workers.sources.autocode.regate_unreviewed`). The sibling of
   `_land_if_passed` one step earlier: gate report at the commit the worktree
   still holds, exactly one red rung, `review`, carrying `external_blocker`;
   no item verdict; nothing has tried to land it; at most `REGATE_CAP` (2)
   times. The re-gate reuses what the ledger has for that commit, so it is
   the review's few minutes rather than a new 40-minute turn. A pass is then
   landed by `_land_if_passed` on the reaper's next look; a graded refusal
   closes and re-offers with its findings, as before. `round.gate_detached`
   is the one spawn, for the tool and the reaper both. While that gate or
   landing runs the item's outcome still reads `external` — a re-offer — so
   `backlog.items_being_gated_or_landed` holds it out of `ready_confirmed`:
   live markers only, never a bare worktree, which an abort that failed would
   leave behind for good.
3. **`wait_for_rounds` needs `ROUNDS_UNREADABLE_POLLS` (6, a minute) unreadable
   probes in a row** before it stops waiting, and `round.land` writes what it
   saw as `land_wait_rounds {ok, detail, waited_s}` either way.
4. **A landing that changes nothing either service has loaded restarts
   nothing** (`promote.restart_needed`, switch `automod.landing.skip_restart`).
   The drain, the wait for siblings and the restart exist so that what landed
   is what runs; when no changed file is a module the backend or the
   aggregator has loaded, what runs is unchanged by the merge. Python is
   judged by ASKING the two processes (`app/loaded_paths.py`, `POST
   /api/automod/loaded` on the backend, `POST /loaded` on the aggregator,
   inside its credential) rather than by directory, because the backend
   imports `scripts/automod/**`, `scripts/autoresearch/**` and `scripts/vault/`
   and a list would not have known about the next one. Everything that is not
   Python is an allowlist (`.md`, `tests/`, `architecture/`, `eval/`);
   `agent-services/**` always restarts (the guardian runs from a snapshot, the
   rest belongs to supervisord programs). It fails closed at every step — a
   server that does not answer, a wrong-shaped answer, an unknown path, an
   empty diff. A module imported lazily after the question loads the landed
   file, so it needs nothing; the promoter asks a second time after the merge
   to catch an import that raced it, and then drains and restarts late.
   On this path `round.land` does not call `wait_for_rounds`, `promote` does
   not pause the pool, arm the drain or take the restart lease, and the
   verification is "live HEAD is the commit and `/health` still answers"
   (`/health.commit` names the BOOT commit and proves nothing without a
   restart). The record carries `restart: false`, the `promoted` row
   `restarted: false` with the reason. The observation window, the settle and
   the LKG advance are unchanged — **except that the guardian does not blame a
   crash or an error spike on a promotion that replaced no running code**
   (`guardian.tick`, `unrestarted`): a crash reads "Service down, but no
   promotion to revert", and `evaluate_errors` is skipped. Data damage is still
   judged, since a changed script can be run by a job inside the window.
   `bless` accepts a served commit that differs from HEAD only by such paths
   (`_served_code_is_head`), as it already did for `.md`.

5. **What was held for the round stays held for its landing**
   (`WorkerPool._landing_in_flight`, `state.rounds_landing`). The round hold
   ended with the last round's turn, which is exactly when a waiting landing
   proceeds: the first landing under the changes above (#608) waited 430 s for
   a sibling turn, three held sources were released 25 s before it paused the
   pool, and it then waited fourteen minutes for them. A live land marker now
   keeps the hold engaged; exempt sources are unaffected. Asked at most every
   5 s; fails open.

6. **A gated round that will restart nothing holds no slot**
   (`autocode._rounds_about_to_land`, `_landing_restarts`). The hold on new
   rounds exists so a freed slot does not start a turn the landing's restart
   would kill. A landing under 4 kills nothing, yet its passed gate kept every
   slot empty until it landed — up to a whole observation window behind the
   promotion before it. The verdict is `restart_needed(..., in_backend=True)`:
   `_loop_is_free` runs on the backend's event loop, so it reads its own
   `sys.modules` rather than requesting itself over HTTP, and gives the
   aggregator two seconds. Cached per `(round, head)` for two minutes; an
   unreadable report, no paths, no head, or a verdict that raises all hold as
   before. `tests/test_loop_depth.py`.

7a. **The observation window is two numbers, and both come from config**
   (2026-09-20, `promote.errors_window`, `automod.landing.errors_window_s` /
   `.errors_window_unrestarted_s`, defaults 450 / 120, and 300 / 120 since
   2026-09-24 — Alan's ruling with the land train, §3.2i). It was a flat 900 s
   constant, and on 2026-09-20 that was **13.5 h of a 24 h window** across 54
   promotions — while every rollback the window has EVER caused fired within
   5.5 minutes of the landing:

   | trigger | when | fired after landing | verdict |
   |---|---|---|---|
   | `error_rate` | 09-07 | 4 s | false positive (the landing's own restart) |
   | `crash` | 09-06 | 147 s | false positive |
   | `data_damage` | 09-10 | 262 s | the `bench_010` vault wipe, blamed on whichever promotion was observing |
   | `data_damage` | 09-17 | 327 s | a `.git` repack the file counter did not skip |

   The other five rollbacks in the ledger's history came from `promote_failed`
   (4, the promoter's own inline failure) and `regression` (1, detached since
   `a54ccda`) — neither needs a window. So 450 clears the longest signal ever
   observed with margin, at half the serialization, and a landing that
   restarted nothing gets 120: the guardian already skips liveness and the
   error rate for such a promotion (5 above), and what it still judges — data
   damage from a changed script a job runs — is a guard weak enough by
   construction (a nightly changed at 14:00 runs at 02:00) that shortening it
   costs nothing the window was really providing. Both are clamped to
   `[60, 3600]`: a window under a minute settles a promotion nothing judged and
   advances the LKG anyway, quietly turning "last known good" into "last
   landed", while above the ceiling the loop stops landing.

   `errors_window_s` had sat in `automod.landing` unread since that block was
   written — the same defect as the two idle keys found dead on 2026-09-17 —
   and `liveness_window_s` beside it named a `LIVENESS_WINDOW` constant already
   deleted as dead, so it is gone. `idle_quiet_polls` and `drain_ttl_s` are
   read now too. `S.landing_cfg` gained the `ledger_rows` mtime cache in the
   same change: config.yaml is 89 KB and `yaml.safe_load` costs **45 ms** a
   call, which the idle budget was already paying twice per landing and the
   window keys would have made six. The promotion record and the `promoted` row
   both carry `errors_window_s`, and `promotion_announcement` states the window
   it was given rather than formatting the constant.

   **The guardian is untouched by design.** It is a pinned stdlib-only snapshot
   that never imports from `scripts/`, so a shorter window reaches it only as a
   smaller `errors_until_ts` on the record; `tests/test_observation_window.py`
   asserts `ERRORS_WINDOW` appears nowhere in `guardian.py`, because a second
   copy of the number would move the promoter and leave the watchdog on the old
   one. No restage, no protected path, no drill.

   **Known and accepted: the LKG's `eval` slot goes staler.** `maybe_settle`
   folds `eval_last.json` into the LKG record only when it names the commit
   settling right then, and the regression check takes a **median 368 s** after
   the promotion (n=103 over the three days to 2026-09-20; min 339 s). At 900 s
   71 of those 103 landed inside their window and at 450 s 69 — the
   distribution is bimodal, so the restarted window costs almost nothing — but
   **none finished inside 120 s**, so an unrestarted promotion will not carry
   its own measurement onto the LKG. It carries the previous record's instead
   (`set_lkg` keeps `existing["eval"]` when handed None), so this is staleness,
   not loss: the check still runs, still writes `eval_last.json`, still records
   its `regression_check` row and still raises a rollback request if it finds
   one. Nothing reads the LKG's `eval` as a control input — the runner
   baselines each commit against `subject["parent"]`, never against LKG — it is
   a record shown by `round status` and `/api/automod`. The obvious fix, letting
   the runner write LKG itself, is what `maybe_settle`'s docstring refuses: the
   guardian being the only writer of `last_known_good.json` is what makes LKG
   mean *observed healthy in production*.

   The settle wait derives from the **larger** of the two windows
   (`settle_max_wait`), never the waiting round's own: what it waits on is
   somebody else's promotion. And it now records — `land_wait_settle`, with
   `waited_s` and `behind`, mirroring `land_wait_rounds`. A successful settle
   wait wrote nothing at all before, so the cost this change is about was
   invisible; scorecard row 14 carries both waits, because they have different
   fixes (the window's length, and depth) and the `promotion_gap` idle class cannot
   tell them apart.

7. **A landing that lost the race for a settled window waits out the
   winner's too** (`promote.wait_for_settle`, `SETTLE_QUEUE_CAP` 4). Landings
   queue behind an observed promotion; when it settles one of them lands
   within seconds, and the other used to be held to what was left of the
   FIRST window: `854144ab` settled 21:45:25, `0c15ba3d` promoted 21:45:32,
   and at 21:47:22 a green round was refused "still under observation after
   waiting 17 min" and reaped. It first happened the day landings got fast
   enough for two to queue. Each new promotion restarts the clock; every one
   queued behind must have a `settled` row, not merely have cleared.

Scorecard row 9 carries `gates_rescued`, `reviews_unavailable` (should fall to
near zero) and `landed_without_restart`. `tests/test_landing_review_deadlock.py`
and `tests/test_landing_without_restart.py` pin all four; `tests/conftest.py`
points the promoter's aggregator URL at the discard port as well as the
backend's, so an unstubbed test gets "restart", never production's answer.

### 3.2i The land train: a merge is a landing, a restart is a flush (2026-09-24)

**Why.** In the week to 2026-09-24 the loop had 58 idle hours, and 32 of them
were gaps around landings. Replayed from the ledger they decompose as: the
settle wait 5.3 h, chase re-gates 4.8 h, and **31.4 h of drain after the last
sibling turn ended** — `_land_lock` → `wait_idle` pausing the pool and waiting
out triage turns and scheduled jobs → restart → verify, median 4.5 min, p90
31 min, recorded by no ledger row. All 172 restarting landings ran with zero
implement turns in flight: `_rounds_about_to_land` held new rounds while
`wait_for_rounds` waited out the old ones, so the loop drained itself to zero
before every restart. The idle is `restarts × drain`, and only fewer restarts
move it. Capping `wait_for_rounds` would have killed ~91 turns a week and
recovered none of the drain; dropping the hold starts turns the restart then
kills.

**The model.** `promote` splits in two:

- **`merge_round`** — everything up to the fast-forward, as before: halt,
  BROKEN, the gated head, the denylist before and after a chase, the chase
  itself (`_regate_after_move`), overlapping live dirt, the squash. Then the
  merge, a `promoted` row carrying `deferred: true`, `restart_pending`, and
  `errors_until: null`, the regression runner, and an entry on
  `pending_restart.json`. No settle wait (an observed promotion is running
  other code; a merge changes nothing that runs), no idle wait, no drain, no
  pool pause, no lease, **no window**. The rollback point is still written
  before the tree moves: the entry is appended, verified, first; the flush
  drops an entry whose commit never reached `main`.
- **`flush_pending(reason, kill_turns=False)`** — today's post-merge block,
  applied to the batch: `wait_for_rounds` and `wait_for_settle` outside the
  automod lock (#1215's rule), then under it the pending list trimmed
  (`_live_pending`: settled already, or no longer an ancestor of HEAD →
  `pending_dropped`), the record written in state `landing`, `wait_idle`, the
  restart legs under the lease, and proof that `/health.commit` contains the
  newest batch commit and `boot_id` changed. The record then reads `observing`
  with `commits: [oldest..newest]`, `rollback_target` = the oldest entry's
  parent, `entries` (each commit's own paths and tree hash), and the kg/vault
  counts taken before the FIRST merge. `restart_flushed` carries the idle wait
  (`waited_s`), the rounds and settle waits, and the batch.

  Two shortcuts: nothing pending needs a restart → a `restart: false` record
  with the 120 s window and no drain; every restart-needed entry already
  contained in the running `/health.commit` (the guardian's rollback restart,
  a human's, a crash booted it) → no restart, the window opens on what is
  running (`already_live`). `kill_turns` (`round flush --now`) skips both
  waits; the pool is paused and every turn in flight dies and is re-offered.

**Triggers** (`promote.flush_due`): nothing pending needs a restart; the oldest
entry has waited `restart_after_s` (2700); `restart_batch_max` (4) need one; or
the loop is at a **natural gap** — no implement turn in flight, where today's
landings restarted anyway and the drain is cheapest. Never while a record is
`landing`/`observing`, a flush or landing runs, promotions are halted, BROKEN
or a rollback request is pending, and never a two-entry batch the staged
guardian cannot judge. Asked by the implement source on every look BEFORE the
free check (a flush wins the gap over the next round, and its marker then
holds it) and at every turn end with the asking turn's own slot row excluded
(`autocode._maybe_flush`), and by `round land` after a merge. The spawn is
`round.flush_detached`, which writes `flush.running` with the child's pid
before it returns. A human's `round restart` of the backend or aggregator IS a
flush while anything is pending (`restart_stack` → `flush_pending(force_restart=True)`),
or the restart would make pending code live without a window.

**Eager landings** (`promote.landing_is_eager`, which names the reasons): a
candidate venv, anything under `agent-services/`, or the train off. They take
today's path — wait for rounds, the chamber's settle wait, drain, restart —
and carry whatever the train holds: the record is stamped with the batch
(`_write_record`), the pending list empties, and `restart_flushed` is written
`by: land`. With the train off the batch is empty and every write is exactly
`S.write_verified(CURRENT_PATH, …)`: the landing path is unchanged.

**The interlock.** A guardian that predates this reads a batch record as one
commit with an older rollback target, and resets from the newest commit to the
oldest one's parent — over any foreign commit between. So
`state.guardian_batch_aware()` reads the STAGED snapshot
(`~/.local/state/lloyd-guardian/bin/gstate.py`, as text) for `BATCH_SCHEMA = 2`,
and while it is false `restart_deferred()` reads the flag as off. A guardian
change reaches the repo long before it is staged, and only the staged copy
decides what a rollback does. `defer_restart` ships **false**: land, restage,
run `rehearse --yes-i-mean-it --batch`, then flip it.

**After a batch rollback, landings go singly** for
`batch_suspend_after_rollback_s` (21600): every rollback this loop has
performed has been a false positive, and a batch rollback blames several
changes at once; re-landing them one at a time lets each be judged on its own
window, so a real regression, if there is one, is named alone.

**What waits on a flush.** `flush.running` is a landing to everything that
waits for one: `_rounds_about_to_land` holds every slot while it is live (and
no longer holds for a gated round that will only merge);
`WorkerPool._landing_in_flight` keeps the round hold engaged;
`bless` refuses while restart-needed entries are pending (HEAD is not what is
served); `round status` / `automod_status` carry `pending_restart` (entries,
ages, `flush_due` and why). Scorecard row 14's `waits` gains `flush`, and
`_duty_cycle` classes a gap holding `restart_flushed` with the promotion gaps.

**Failure.** Before the restart: the record is cleared, `flush_failed`
(`restarted: false`), the entries stay pending for the next trigger. After it:
the batch is undone inline by the guardian's own rule (`_undo_batch`: reset only
on an exact range, `revert_commits` otherwise), `rollback_succeeded` names every
commit, and if that fails the batch goes to the guardian as ONE request
(`request_rollback(commits=…)`), which escalates if it cannot either.

**Expected** (ledger replay): restarts 172 → ~60–95 a week, 20–26 of the 32
idle hours back, `land_wait_settle` gone for merges, no turns killed by the
loop. Watch the first night: `restart_flushed` count and batch sizes, row 14's
zero-turn gaps, `infra_failed` within 3 minutes of a flush (must stay 0), and
any `rollback_succeeded` with `batch > 1`. `tests/test_land_train.py`,
`tests/test_guardian_rollback.py` (batch), `tests/test_loop_depth.py` (train).

### 3.3 For humans (this repo's development)

`/home/alansrobotlab/lloyd` is production. Non-trivial work belongs in the
**`~/lloyd-sandbox` clone** and reaches main as a PR:

```
~/lloyd-sandbox   remotes:  origin = GitHub (PRs)
                            live   = /home/alansrobotlab/lloyd
```

A clone rather than a worktree, deliberately: a worktree shares the live
object store, so a bad `gc` or ref contention reaches production. A clone
cannot. Landing from it stays offline — `git fetch /home/alansrobotlab/lloyd-sandbox <branch>`
reads the filesystem, no network.

The sandbox needs a venv to run tests. On btrfs that is nearly free:

```bash
cp --reflink=always -a /home/alansrobotlab/lloyd/.venvs/lloyd .venvs/lloyd
# measured: 3.2s, ~0 allocation, 6.2GB across 48k files
ln -s /home/alansrobotlab/lloyd/.env .env   # gitignored, absent from clones
```

**Why the autonomous loop does not use PRs.** A PR step in an auto-landing
loop is either ceremony (auto-merged, adding a GitHub dependency to the
failure path) or a contradiction (not auto-merged, reintroducing the human
tier that was deliberately removed). The loop cuts worktrees from live `main`
and lands offline. The sandbox is for work that is *too big to auto-land* —
which is also the natural home for dependency changes and guardian edits.

**Friction worth knowing:** nightly jobs commit directly to live `main`, so
`main` is written from two directions and the sandbox needs regular rebasing.

---

## 3.3 Map the blast radius before editing

Both the `automod-change-own-code` skill and `autocode`'s PROMPT now
put one step between opening a round and doing the work: `graph_refresh`, then
`graph_affected` for each symbol about to change and `graph_explain` for its
callers (`agent_mcp/code_graph.py`, an AST extraction with no LLM calls). The
round report names the depth-1 callers and the file list, and quality gate 4
requires it.

Two things about it that are easy to get wrong:

- **Pass `root=<worktree>` on every call.** The default is `LLOYD_HOME`, the
  live checkout — an implementer that omits it reads confidently about a tree
  it is not editing. `root` is never inferred from the session: nothing on disk
  links a chat session to an open round, since `round_start` ledger rows carry
  no session id.
- **The graph is blind across process seams.** There is no edge from
  `run_prompt_in_session` to `run_query` (HTTP), and none from `run_query` into
  a tool handler (MCP). Grep is still the right tool for string keys, route
  paths and config names.

`graphify-out/` is gitignored unanchored so a worktree inherits the rule; a
build there would otherwise dirty the tree, and both `gate.py` and
`promote.py` refuse a dirty tree — the round would abort on its own map.

## 4. The gate

Eleven rungs, cheapest first, short-circuiting. **Every rung fails closed** —
`_rung` catches exceptions and records them as failures, because with no human
review tier a rung that errors and reads as "didn't fail" silently removes a
check.

The gate always runs **from the live tree** against the worktree, so a
candidate that weakens the gate is judged by the old gate.

| Rung | Typical | Catches |
|---|---|---|
| preflight | ~0s | dirty tree, moved base, merge commits, out-of-scope paths; an item with no clauses, a code diff with no test |
| vet | <1s | **observe-only (#679)**: a tracked file non-empty at base and empty at HEAD, a newly-added binary outside the allowlist, a diff over `automod.gate.max_diff_lines`. Records to `gate.json` and the ledger; blocks nothing during the soak |
| static | ~2s | syntax errors, **import failures**, new pyflakes findings |
| frontend | ~5s | new tsc errors, a broken vite build (only when `web/` changed) |
| tests | ~70s on 8 workers (~10m serial) | the full suite, plus floors on collected AND passed — §4.2f |
| prompt_surface | conditional | a scored regression in what the model is told (only when the prompt surface moved) |
| review | 3.5-6.5m | a diff that does not do what the item asked — §4.5 |
| venv | 3s–5m | only when `requirements.txt` or `requirements.lock` changed — not `requirements-dev.txt`, §10 |
| canary_boot | ~2-30s | a build that will not start |
| canary_smoke | ~5-15s | a build that starts but cannot dispatch a tool |
| drill | ~35s | only when the diff touches the rollback path |

Timings above are the measured runtimes of the last full ladder
(`gate.json` per round, 2026-09-11/12): static 2s, tests 127-152s, review
215-394s, canary_boot 2.3s, canary_smoke 6-7s, drill 33s. A week later the
same serial `tests` rung took 430-600 s and `review` a median 401 s
(`backlog.gate_duration_stats`, §3.2g); `tests` has run in parallel since
2026-09-18 (§4.2f), and the prompt and the gate tool read the current figure
off the ledger rather than from this table.

Every rung is in `gate.json`; not every rung is on the ledger. Since
2026-09-24 a **skipped** `frontend`, `prompt_surface`, `venv` or
`canary_smoke` writes no `gate` row (`Gate.QUIET_SKIP_RUNGS`) — ~1,900
zero-second rows a week that said only "not this round", read by nothing.
`drill` is always written, skipped or not, because
`backlog._last_gate_per_round` and `gate_passed_unlanded_rounds` key a full
pass on an ok `drill` row; a failed skip (none exists today) would still be
written.

Two rungs carry most of the weight.

**static** runs `python -c "import server, agent_mcp.main"`. An import-time
failure is the number-one way a self-modification bricks the boot, and this
catches it in seconds without binding a port or running a startup hook.

**canary_boot** starts both services from the worktree on 18080/18500 under a
throwaway supervisord and asserts the backend resolved MCP to *its own* port.
That last assertion is the point: a round that broke `service_url()` would
leave the canary talking to the *live* aggregator, and every other check would
still pass.

`pytest` substitutes for neither. It imports modules in its own process and
never starts `server.py`, so it stays green through a broken startup event, a
port collision, or an aggregator that registers no tools.

### 4.5 The review rung: did it do what was asked?

Eight rungs asked whether the change *broke* something. None asked whether
it did what the acceptance check said, and the only verdict on that was the
round's own self-report. Over the loop's first four days every landed round
was green on every rung, seven items closed, and #544 — 3,271 tests, canary,
smoke, a flat regression eval — had one of five acceptance clauses skipped,
half the worker fleet uncovered by the ledger it added, an `Edit` replay that
silently reverted files, and a test asserting `x == [] or True`. A human with
the item and the diff found all four in forty minutes. That is the grader's
whole input.

`scripts/automod/review.py`. A **fresh session on the live backend** — the
primary model, `platform: worker`, Inner Voice off, `priority: 1` — is handed
the item, its clauses, `git diff base...HEAD` and the changed test files, and
**not** the triage evidence or the round's report: the author's prose primes
agreement. It reads the worktree at its absolute path and may run tests only
through `<round>/gate-state/run_tests.sh`, which bakes in the gate's
interpreter, cwd and scratch state so a grader-run suite cannot write live
automod state. It answers under `REVIEW_SCHEMA`: a premise verdict, one entry
per clause with `evidence_path`, `test_node_id` and `how_verified`, a
`test_honesty` list and a `seams_unverified` list.

Not the canary, deliberately. The canary is candidate code end to end
(`canary_config.py` puts the worktree on `PYTHONPATH`): its harness,
finalizer, prompt builder and tool policy are the round's own, so a round
touching `app/harness/finalizer.py` would grade itself with itself — the
gate's founding rule in reverse. The canary's sessions dir is also deleted
with the worktree, and the grader transcript is what the scorecard reads.

Three rails keep the verdict honest without trusting the model:

- **`parse_review` downgrades `met` to `partial`** when `evidence_path` is not
  in the worktree, `test_node_id` is not in a test file the diff changed, or
  `how_verified` is `inferred`. A clause the grader did not mention is
  `partial`. A lazy grader cannot pass a round. The path is *normalised*
  before it is judged (`normalize_evidence_path`): the schema asks for a bare
  worktree-relative file and graders write `app/x.py:164`,
  `scripts/a.py:224,253`, `autonomy.py::_pin`, `~/obsidian/…`, or an absolute
  path into the review snapshot that has since been removed — every one of the
  first four backfill rows had a `met` downgraded for a path that was real. A
  line suffix, a trailing symbol or anchor, and a dead absolute prefix whose
  tail resolves in the worktree are all accepted; anything left is a genuine
  miss, and the downgrade reason records what the grader actually wrote, so a
  rail that fires wrongly is visible rather than silent. The cited *line* is
  bounded too (`evidence_line_past_eof`, #1254): a file inside the worktree is
  the commit under review, so an `evidence_line` past its end is recorded
  under `citation_unresolved` on every verdict and downgrades a `met` — 23
  `met` clauses across 15 rounds had cited past EOF before this and nothing
  checked. A vault path is live and shared, so its line is not checked, and
  neither is a path the rail resolved through a later citation token — the
  number describes the file the grader named first.
  Three shapes stand as `met` besides a changed test, each recorded on the
  clause as `accepted` (§4.5d): a suite-level node (`tests/ -k expr`, a test
  file with no `::`) or an existing `tests/` node outside the diff, both only
  when the grader `ran` it and the gate's own `tests` rung passed on this
  commit; and an `evidence_path` naming something the diff deleted (in the
  diff's paths and gone, or marked `(absent)`/`(deleted)`/`(removed)`) beside
  a node in a changed test file — never stacked on the suite-level waiver.
  "Test file" means one of `pytest.ini`'s `testpaths` — `tests/` and
  `app/harness/tests/` alike, and a collectable `test_*.py` under `scripts/` —
  through the one predicate in `scripts/automod/testpaths.py`, which the
  changed-test set, the prechecks, the grader prompt, the node rail, the tests
  rung's partial narrowing and `review_tools` all call. A root-only
  `startswith("tests/")` downgraded every clause pinned in the harness suite
  (#1322, SM_20260921_030016: four of five).
- **Deterministic honesty checks run first**, as a delta against the base
  version of each changed test file: `or True`, `assert True`, a new skip or
  xfail, and — when the item has clauses and code changed — no new `def
  test_`. A hit is a finding whatever the grader says.
- **An unverified seam blocks — if a test could have crossed it.** The three
  worst defects in #544 were all cross-process (a contextvar lost over a
  loopback POST, a `Task` subagent, `_meta` over MCP) and the code graph is
  blind across every one of them. Since 2026-09-11 each seam carries
  `testable_before_landing`, and only a testable one refuses: #870 met all
  eight clauses on its second attempt and was refused on two seams the grader
  raised fresh on the last pass, one of which the grader itself called a
  post-landing check — a real pool tick in the worker process, which no test
  in this repo can drive. An untestable seam is recorded on the event
  (`seams_untestable`) and rides the findings. #544's lesson is about seams a
  test *could* have crossed. A bare-string seam keeps the old reading, so the
  calibration cases and the backfill are unchanged. A testable seam refuses
  only while `automod.review.seams_block` says so — attempt 1 under `first`,
  every attempt under `always` — and never when the grader calls it a repeat;
  the rule was dead code until 2026-09-13 (§4.5d). **Shipped policy is `never`
  since 2026-09-24**: over the week before, 72 of 130 send-backs had every
  clause `met` and were refused only on a seam on attempt 1, each costing a
  review (457 graded, median 326 s), a ~15 min fix cycle and a full re-gate.
  A seam now always rides as a post-landing check; `first`/`always` stay in
  code so `review_tools redecide --seams-policy` replays history. What a passing review did not
  refuse on is written onto the item (`post_landing_seams` and one activity
  line, `backlog.note_review_advisories`) and into the rung's data as
  `advisory_seams`/`advisory_findings`. It is recorded, never held open: 23
  of the grader era's first 30 seams were untestable by the grader's own word.

The rung's four outcomes, and where each goes:

| grader says | rung | then |
|---|---|---|
| every clause `met`, no findings | pass | ladder continues |
| premise sound; a clause unmet/partial, a `blocking` honesty finding, a seam a test could cross before landing only under `seams_block: first`/`always` (advisory honesty entries and post-landing seams ride the findings without refusing — #866 and #870, the evening the rung first ran) | fail, `review_retry` + `review_findings` on the gate event | the round fixes what it names, commits, and gates again; the second graded refusal of a *distinct commit* says *abort and report*; a third is refused without asking the model. Re-gating the same commit is answered from the ledger with the same findings and spends nothing. A turn that ends unlanded hands the item back as `implement_outcomes` → `review_retry` (cap 2), findings in `reoffer_reason`, branch kept — the next round passes it as `automod_start(from_branch=…)` and begins where this one stopped, rebased onto live main |
| premise sound; a clause `unsatisfiable` as written | fail, `review_retry`, the findings name `automod_amend_clause`; **spends no attempt** | a refusal of the contract, not the diff: the author amends exactly that clause (refused for any other), gates again, and the next review sees the amendment as an `<amendments>` block and ratifies it or refuses it and restores the old text (§4.5c). `REVIEW_HARD_CAP` — 5 graded reviews per round, passes included — is the ceiling that keeps the free move from looping |
| premise unsound | fail, `review_premise_unsound` | the existing `spent` path: `draft` + `needs-human`, tag `review-premise`, the grader's summary on the item |
| grader unreachable, 503, timeout, unusable object | fail, `external_blocker` | the engine, not the diff; the item keeps its attempt. Never a SKIPPED pass — a waived review is the #544 shape |

It sits **after `tests`**, so it can trust a green tree and is handed the
counts, and **before `venv`**, so a refusal saves the build, the boot, the
smoke and the drill (and because `canary_smoke` must immediately precede
`drill`). It is skipped, and recorded as skipped, only for a round with no
item bound — a human's round has no contract to grade against.

A green `tests` rung is not always a green tree since 2026-09-24: the rung
passes over failures that reproduce at the round's base (§4.2b). The grader is
told their ids ("fail here and at base; not this diff's; a clause leaning on
one is at most `partial`"), and `parse_review` enforces it — a `met` whose
`test_node_id` is one of them, or sits in a file that holds one, is
downgraded, whatever `how_verified` says (`_cites_pre_existing`).

**Termination.** Three bounds end the author/grader loop: two graded
refusals of distinct commits per round (a grader timeout and a duplicate
review of one commit spend nothing — §4.5b), `REVIEW_RETRY_CAP = 2` across
rounds, and an early exit — when the same clause comes back unmet on two
consecutive reviews of an item *on two different commits*
(`review_disagreement`), the item is `spent` at once, tagged
`review-disagreement`, and announced through the guardian's fan-out. A
third round would re-run the same argument; a human resolves it.

A clause Python **downgraded** does not count toward that disagreement. A
`met` the grader wrote and the evidence rails demoted is not the grader
disagreeing with the author, it is the grader agreeing without receipts:
#860's clause 8 ("the suite exits 0") was downgraded on both reviews, once
for the test node and once for the path, and would have parked the item as a
disagreement nobody had.

`select_confirmed` orders the pool so this cannot monopolise it, in three
keys. **Nearest to landing first**: a re-offer whose last graded review met
every clause has one small task left — a test across a seam, an amendment to
ratify — and lands in one gate, while a fresh item costs an hour and two
review attempts. On 2026-09-11 five such re-offers sat behind fresh umbrellas
that each took the hour and aborted. Then fresh confirmations, then other
re-offers, then oldest first: oldest-first alone let a sent-back item be
re-picked on the very next round for as long as its cap allowed, while the
rest of the pool waited.

Vault rounds have no worktree and no ladder — the edit is live the moment it
is saved — so the same reader runs inside `vault_round.land`, between
validation and `git add`, over the staged diff (`vault_round.GRADER`, wired
by the aggregator; `None` in tests and from the CLI records `review:
skipped`). The first refusal leaves the edit in place for the model to fix;
the second, or an unsound premise, reverts it — so the nightly vault sweep
cannot land unreviewed text under someone else's commit.

It grades only a **`vault`-surface** item's clauses, and skips with a note
otherwise. #551 was a `code` item whose round landed a skill and a task file
first; grading the whole contract against that half refused it for the code it
had not written yet — and would have, every time. A `code` or `mixed` item's
clauses belong to the code gate; the vault half is still validated through the
real loaders either way.

**Calibrating the grader.** A rung that blocks from day one has to be
checked against diffs whose verdict is already known.
`scripts/automod/review_tools.py` runs the grader exactly as the rung does —
same prompt, schema and live backend — over a detached worktree of an
already-landed commit, so nothing it does can touch a round or the ledger.
`eval/review_calibration/*.json` are the cases: #544's own landing
(`0f019f9`; expected `retry`, clause 4 not met, a test-honesty finding — the
human review's verdict), the same commit with its test file stripped
(`strip_tests`; expected `retry` on clauses alone), and two small landings
whose authors said `met`, marked `provisional` until a human confirms.
`calibrate` exits 1 on any disagreement and must pass before `backfill` —
which grades every settled landing with an item into
`review_backfill.jsonl` beside the author's verdict — means anything. That
file is the scorecard's audit-delta baseline, and it is a separate file on
purpose: a backfilled verdict measures the grader, and `implement_outcomes`
must never read it as a verdict on the round. Items confirmed before clauses
existed mostly wrote `(a) … (b) …` inline; `acceptance_clauses_of` splits
those, so #544's prose reads as the five clauses its reviewer graded.

### 4.5b The verdict has to reach the author: transport, attempts, snapshots

The rung shipped on 2026-09-10 at 18:32 PT. In the eighteen hours after,
eighteen rounds started, seventeen aborted and none landed — and every
review-gated round ended the same way: two review rows on the ledger, then a
0.02 s "already sent this round back 2 times; abort and report". It read as
the grader being strict. It was the transport eating the verdict, four links
long, and each link is now pinned by `tests/test_review_transport.py` and
`tests/test_mcp_pool_retry_policy.py`.

1. **A gate with the review rung runs seven to twelve minutes; the MCP
   tool call could stay silent for five.** `mcp_pool._http_session` built
   its HTTP client with the SDK default, `read=300`, so the stream died at
   300 s while the gate was still grading. The pool's `CALL_TIMEOUT_SECONDS`
   of 660 was never reached. The client now carries
   `HTTP_READ_TIMEOUT_SECONDS`, above that ceiling.
2. **The pool then "reconnected and retried once", which started a second
   gate of the same round.** #578: gate one preflight 09:22:18, the retry
   warning at 09:27:17, gate two preflight at 09:27:18, transport error to
   the model at 09:32:17. A transport error says nothing about whether the
   server ran the call — for a long one it almost certainly did — so a retry
   is only safe for a call the server annotates read-only or idempotent.
   `_retry_safe` reads the annotations `_list_tools` already carries;
   anything else is surfaced as a tool error that says the call may have
   run. `automod_gate` itself now returns immediately and runs detached
   (`state.spawn_detached`, the landing's pattern), with `gate.running` as
   the one-gate-per-round marker `run_gate` owns and clears; the model polls
   `automod_gate_wait`, which blocks in slices well under any transport
   budget and hands back the same per-rung report. It refuses to start while
   a background task of the session is in flight — #578 launched its only
   baseline run and gated over it, then killed the run at abort.
3. **The review cap counted every row on the round id**, duplicates and
   grader timeouts included, so both attempts were gone before the model
   had read one finding. An attempt is now a *graded refusal of a distinct
   commit*: the event carries `head`; a timeout (`ok: false`) spends
   nothing; the same head refused before is answered from the ledger
   without a grading turn ("nothing has been committed since"), which also
   makes re-gating without committing free instead of fatal.
4. **`review_disagreement` read the duplicate pair as the author refusing
   to fix a clause** and parked #578 `needs-human` on its single attempt.
   Two refusals of one head are one refusal, delivered twice; rows without
   a head predate the field and keep the old reading.

Two smaller things rode along. The grader reads a **detached checkout of
the commit** (`Gate._review_snapshot`, removed after grading), never the
author's working tree: the second review of #578 caught a test file
mid-write and graded a head the author had already moved past. And the
backlog's own filing of the incident, #838, had the symptom right and the
mechanism wrong (it blamed lineage across resumed rounds); the ledger, not
the prose, is what to read for a cause.

### 4.5c Three things the loop can now settle for itself

The day also showed three dead ends the loop could only escalate:

- **A clause no diff can satisfy.** #578's clause 4 asked for an A/B whose
  held-out half was already written into the file under test — a date split
  cannot separate a corpus from memory files written from the same notes.
  The reviewer's only verdicts were `unmet` (wrong: the diff was not at
  fault) or `unsound` (wrong: the item was real). `unsatisfiable` is the
  fourth clause verdict, and it names a remedy: the author calls
  `automod_amend_clause(round_id, clause, text, reason)`, which
  `backlog.amend_clause` refuses for any clause the last graded review did
  not so mark, writes the new text onto the item with a `pending` record
  under `clause_amendments`, and the **next** review sees it as an
  `<amendments>` block and ratifies it (`amendments_ok`) or refuses it and
  restores the old text. The second reader's judgment, not the author's:
  a loop that rewrote its own acceptance unchecked could declare anything
  met. An `unsatisfiable` clause on two distinct heads without an amendment
  still escalates through `review_disagreement`.

  **A check that can only run after the change lands is `unsatisfiable`
  before it, never `partial`** — a day of traffic, a nightly run, a number
  only production produces, a script over live data. No diff can carry that
  evidence, so grading it `partial` refuses a round for the one thing it
  could not have done: #859 was refused twice on exactly this shape and
  parked, with the mechanism complete on both commits. The reviewer names
  what the pre-landing clause would be — the mechanism plus the test that
  pins it — and the post-landing check becomes a person's (`human_clauses`)
  or a deferral the author names.
- **A clause only a person can satisfy.** Clause 5 asked for ten
  human-audited items; the round could not produce them and the reviewer
  correctly refused to accept an agent-authored audit as one. Triage now
  separates `human_clauses` from `acceptance_clauses` (both parse paths, the
  schema, `record_verdict`), the implement prompt shows them as "not yours
  to do, and not yours to simulate", the review prompt shows them as not
  graded, and `close_settled_items` leaves a `met` landing **open** and
  tagged `needs-human` with the outstanding conditions in its note. The
  loop's half lands; the item waits for the person.
- **A gate over a running measurement.** Covered in 4.5b: the gate tool
  reads `_task_registry.list_active()` for the bound session and refuses.

### 4.5d The grader era's first day: 4 of 23 landed

`automod.review.policy` flipped to `grader` at 2026-09-12 23:46 (`4c9eb35`),
on a calibration that showed the grader policy's blocking set inside the
table's *at attempt 1*. (The `table` policy and the `review.policy` key were
retired on 2026-09-24; `decide` is the grader policy alone.) Over the next day 4 of 23 graded rounds landed,
against 8.1 landings a day in the 30 hours before, and the board's ready
queue sat at 87 against a depth bound of 55. Re-deciding the recorded
refusals on their own grader output named the causes, heaviest first:

1. **Advisory findings refused rounds.** `decide_by_grader` blocked any
   test-honesty entry the grader called `actionable_in_round` and ignored its
   `severity`. A docstring number is always fixable, so always actionable: 60
   advisory entries against one blocking, and test honesty led 15 of 21
   refusals. #484 met every clause in four rounds and was refused on a fresh
   nit each time. Severity decides now; `actionable` only demotes a blocking
   entry (to "blocking, but not fixable in this round"), never promotes one.
2. **A suite-level clause could not be `met`.** "The autoresearch tests
   pass" has no single node; #860's clause 8 was downgraded three times for
   writing `tests/ -k autoresearch`. The three `accepted` shapes in §4.5.
3. **A stale amendment bypassed the cap.** `pending_amendments` read the
   whole item while `settle_amendments` settled only the current round's, so
   an amendment left by a round two days gone skipped the same-head check,
   patch-id reuse and the two-attempt cap and bought #860 a third graded
   review ("3/2"). Amendments are per round now: `orphan_stale_amendments`
   marks another round's `pending` record `orphaned` and restores the clause
   at `round start` and at the review rung, quoting the amended text on the
   item so the open round can re-amend it in one call; the rung also filters
   to the round's own, so the cap holds even if that write fails. The same
   pass found the patch-id reuse branch reading `attempt` before assigning
   it — that branch had never run.
4. **Seams blocked on every attempt.** `seams_block` was never consulted
   under `grader` (decisive in 4 of 21 refusals), and an advisory seam on a
   pass was discarded although the prompt promised the grader it would be
   recorded. Since 2026-09-24 the shipped `seams_block` is `never` (§4.5):
   the seam-only refusals that remained were 72 of a week's 130.
5. **A declined poll spent the whole interval.** The pool stamped
   `last_enqueue_check` after every `enqueue_if_due`, so an autocode poll that
   found a promotion still under observation waited 900 s to look again —
   median round-end → next-start 9.6–14.5 min, ~12 h of a free loop over
   fifty rounds. A source may now return `DECLINED`; with `retry_seconds`
   (autocode: 60) the pool looks again that much sooner, and autocode's four
   board passes keep the 900 s clock on their own `last_housekeeping`
   watermark. The observation window is a separate clock and was untouched by
   that change; 7a is where it moved.
6. **The implementer was never told what the grader checks,** and a
   re-offered round's grader saw no history. The implement prompt now names
   seams, the suite-run evidence, graded test prose and the two-attempt cap;
   a `review_retry` re-offer carries the last review's per-clause verdicts;
   the grader is shown the item's last three graded reviews across rounds,
   each naming its round, while attempts are still counted on the round.
   Checking the first re-offer after landing found that **no re-offer banner
   had ever been delivered**: `execute` wrote the attempt's `started` row
   before building the prompt, `implement_outcomes` reads an item by its
   latest row, and a `started` row with no round reads as `spent` — so the
   banner was empty in all 92 autocode sessions on record. It is built before
   that row now, and `tests/test_backlog_unattended.py` pins it through
   `execute` rather than by calling the block builder.

Both policies then shared the severity, seam-attempt and refused-amendment
rules, and a review object without the two judgment fields decided the same
under both at attempts 1 and 2 — which is why retiring `table` (2026-09-24)
changed no verdict; `tests/test_review_grader_policy.py` pins those decisions.
`--seams-policy first|always|never` replays under another `seams_block`.
`python -m scripts.automod.review_tools redecide --since 2026-09-12T23:46`
replays every graded review under today's rules without asking a model; at
landing it re-decided 18 of 22 recorded refusals as passes (14 without the
step-2 approximation it prints) and turned no recorded pass into a refusal.

### 4.1 pyflakes is a diff, not a bar

The tree carries hundreds of pre-existing findings — 215 unique ones measured
over every tracked `.py` on 2026-09-12, up ~3× from the 69 this section cited,
because the count is over all files and nothing re-measures it. An absolute rule would be switched
off within a day, so the rung compares changed files against the same files at
the merge base and fails only on *new* findings.

### 4.2 The collected-count floor is not enough on its own

`pytest -q` exits 0 if a round deletes the test that was failing. Under
auto-landing that is not hypothetical, so the rung asserts a minimum
collected count and refuses a diff that removes files under a testpath.

It exits 0 just as happily having **collected everything and run none of it** —
one broad `skipif`, or a conftest import that degrades to a module-level skip.
A collected floor alone is satisfied by a suite that did nothing, so the rung
also floors `passed` and caps `skipped`. Three numbers, because the failure
being prevented is a round that reaches green by removing the question rather
than by answering it.

### 4.2b A failing tests rung says whose breakage it is

The rung is a diff too, for the same reason 4.1 is — it just took an incident
to notice. `pytest` failing says the tree is red; it does not say the round
made it red. On **2026-09-08** three rounds aborted here on the same three
failures, none of which any diff under test had written:

| Item | Round | Gated (UTC) | Failures |
|---|---|---|---|
| #361 | `SM_20260908_065238` | 07:00 | guardian_speak ×2, tool_overrides |
| #370 | `SM_20260908_105946` | 11:05 | guardian_speak ×2, tool_overrides |
| #376 | `SM_20260908_145622` | 15:14 | tool_overrides |

Two tests asserted the wall clock against a 23→07 quiet-hours window, so they
failed exactly when the unattended loop runs; one asserted that a deliberately
untracked file had been checked out, so it failed in *every* worktree whatever
the diff. `8138f1c` fixed them at 16:39 UTC — after all three — and nothing
went back for any of them, because `implemented_ids` counts any finished round
as the item's one attempt. Three items were consumed by breakage none of them
wrote, and each of those rounds had *proved* it: #361's cut a worktree at its
own base and reproduced the failures with none of its code present, #370's
filed #479 saying the gate was unlandable for every round, #376's ended "the
branch is ready as-is". All of it in prose, in reports that are read once.

So on a failure the rung re-runs the failing **files** at the round's base, in
a throwaway worktree and — since 2026-09-24 — a throwaway data root
(`<round>/baseline-data`, `LLOYD_DATA` for the probe only), and classifies:

- in a file the round's own diff touches → **the round's**, even if it also
  fails at base (the touched-file rule: a round that edits a red test file
  and leaves it red owns it; such nodes are not even probed)
- reproduces at base → **pre-existing**
- does not reproduce, but passes a repeat run with the diff present (§4.2f,
  #1196) → **flaky**
- anything else → **new**, the round's own, and the detail names it

**Since 2026-09-24 the rung passes when nothing is new.** Every failure
pre-existing or flaky → `ok`, with `pre_existing_failures`, `flaky_node_ids`,
`base_probe`, `red_set_cached` and `red_tree_item` in the rung data, the lists
copied onto the gate event (capped at 50 ids), and the report headline's
`notes_that_did_not_block.pre_existing_failures` telling the round who owns
them. The pass goes through the same tail as a green run (`_tests_pass`), so
the collected / passed / skipped floors and the removed-files check still
apply. One new failure fails the rung as before, and the mixed case still
records `pre_existing_failures` so the author does not spend the fix cycle
on them. An INCONCLUSIVE probe grants nothing.

**The probe runs in its own data root** (#1436, `<round>/baseline-data`,
`LLOYD_DATA` for the probe only, removed with the worktree). It used to inherit
the tests rung's env, whose `LLOYD_DATA` is the round data root that run had
just written into — so a store the candidate's own tests provisioned (a 0-row
`kg.sqlite` from `kg_store.configure(VAULT_KG_DB)` in a test the round added)
was present "at base", the corpus guard read it as the store of record there
too, and `SM_20260924_063720` was excused for its own defect, twice. Now that a
pre-existing verdict passes the rung, a probe that can see the candidate's
residue would pass it outright. `HOME` stays the round's: the round home is a
symlink farm over the real one, so `lloyd-data` is the only residue it can hold.

It used to **fail** with `external_blocker: true`, on the theory that a red
tree would open the promotion's observation window against a broken baseline.
The guardian never runs pytest, and 85 of the week's 257 landings restarted
nothing; what the refusal did was kill rounds. In the week to 2026-09-24 the
rung failed 284 of 895 runs, 221 of them pre-existing; 157 rounds hit it and
21 ever landed; red episodes ran up to 59 h (`tests/test_uptake.py`) across 39
distinct red bases, and nothing in the loop fixed the tree — it only refused
to land on it. The `tests` rung no longer writes `external_blocker`, so the
red-tree `external` verdict in `implement_outcomes` disappears by construction;
`preflight`, `review`, `land_failed` and `round.py` still write it, and
historical `tests` rows are read as before.

**The probe's answer is cached per base** (`state.read_red_set` /
`write_red_set`, `red_set.json` in the state dir, newest 8 bases). Before
probing, the rung asks whether every failing id is already recorded as failing
at this exact base sha, fresher than `automod.gate.red_set_max_age_s` (6 h);
if so the probe is skipped and `red_set_cached` says so. A conclusive probe
merges its answer in; an INCONCLUSIVE one writes nothing; a full green run
records `[]` for its base. The set is only consulted for nodes that failed at
HEAD, so a stale entry can skip a probe, never hide a failure; a preflight
rebase is a new base and a new entry. `tests/test_gate_red_set.py`.

**The tree heals itself** (`backlog.file_red_tree_item`,
`automod.gate.file_red_tree`). A conclusive pre-existing set is filed as one
open item tagged `red-tree` — priority `high`, confirmed at birth through
`record_verdict` with one clause per failing file plus "no test skipped,
xfailed, deleted or marked `live_vault` to get there", and an `auto: true,
red_tree: true` triage row — so `select_confirmed` takes it next and the depth
gate never holds it. `red-tree` is not a `spawned-by-*` tag: expiry and the
write-time merge leave it alone. A report on the same base unions into the open
item (a clause per new file), a report on a descendant base replaces it, an
older view is ignored, and the round's own files are never filed. An item a
round is already working (`in_progress`) is never rewritten: what it covers is
that round's, and anything new goes to another open red-tree item or a new one
(2026-09-24: a sibling gate had appended a clause to #1454 twenty minutes into
#1454's own turn). That round's gate headline says the listed failures are its
own contract (`red_tree_item_is_own`), not "do not fix them". A full green
run at a base descending from the item's closes it `already_done`
(`close_healed_red_tree`) — most red trees are healed by a hand commit, and a
`high` item left open would spend a round proving nothing — unless the round's
diff touched the item's files, in which case that round is the fix and its
landing closes it. A closed item suppresses re-filing for 24 h only for an
*older* view of the healed tree: the same failure reproducing at the heal's own
base means the green run's diff fixed it and never landed, and it is filed
again. Ledger rows: `red_tree_filed` (`action`: created / merged / replaced)
and `red_tree_closed`. Filing is try/except inside the rung — never the gate's
verdict — and happens only when the gate judges the production tree, so a test
Gate over a throwaway repo cannot reach the board. `tests/test_backlog_red_tree.py`.

Things that decide whether it works:

- **It probes files, not node ids.** Handed a node id that does not exist at
  base — a test the round just wrote — pytest exits `ERROR: not found:` and
  runs *nothing*, so one new test would hide every pre-existing failure beside
  it and a red tree would read as green. Found while building this; pinned by
  `test_a_test_the_round_added_does_not_hide_the_pre_existing_ones`.
- **It fails closed in every direction.** A worktree that will not create, a
  probe that times out, an unparseable summary — all are INCONCLUSIVE, which
  passes nothing and caches nothing. Being wrong that way costs a round;
  being wrong the other way lands a change nobody checked.
- **The grader is told** (§4.5): the review prompt lists the pre-existing ids,
  and `parse_review(pre_existing_failures=…)` never lets a `met` stand on a
  node in the set or in a file that holds one, whatever `how_verified` says.
  The residual risk is a `met` on a different changed-test node while the
  clause really depends on a red unchanged test — the same residual as before.

### 4.2c A round that never reached a verdict has not spent the item

`implemented_ids` counted any finished round as the item's one attempt,
"whatever it did". Across the loop's first seventeen unattended attempts,
**six** were spent by something that was never a judgment on the change:

| Item | What ended it | Work left behind |
|---|---|---|
| #361, #370, #376 | pre-existing test failures (§4.2b) | 3 branches |
| #446 | wall clock, 14 s and one `automod_gate` call short of landing | 757 lines |
| #447 | `preflight: live tree is dirty` — an unrelated uncommitted edit in production | 587 lines |
| #392 | the turn never ran; the backend was down | none |

`implement_outcomes` classifies each attempt instead, and only `spent` closes
the item. Everything else is offered again, each bounded by its own cap
because `select_confirmed` takes the *oldest* ready item — an uncapped
re-offer is re-picked every round for as long as the cause persists, starving
everything behind it.

- **`incomplete`** — `turn_timeout` or `max_turns` on a round that did not
  promote. Triage has recorded budget exhaustion this way since #229 ("the
  item comes back once"); implement never got the rule. #446 is why: it
  committed 757 lines at 06:43:36 and was killed at 06:43:50, with 32 of its
  100 iterations unspent. **Once means once, counted on budget deaths
  alone** (since 2026-09-14): the rule used to read the shared attempt count
  as `n <= 1 + INCOMPLETE_RETRY_CAP`, which re-offered a second budget death
  as well, while counting every attempt would spend it on an unrelated
  earlier verdict. A second budget death is `spent`, which
  `retriage_spent_items` (§3.2d) sends back through triage once rather than
  running the same contract a third time. On that day's ledger it moved one
  item: #577, dead at its budget on 09-11 and 09-13.
- **`infra`** — the turn never reported completion. `run_prompt_in_session`
  returns `stop_reason=None` when the stream closes without a `done` frame.
  #392 was recorded as its item's one attempt *one second* after starting, on
  a session holding a single user message, while the guardian was alerting
  that supervisord was unreachable. New runs record `infra_failed`; the read
  path also recognises the old shape, so history heals without a backfill —
  but only an **explicit null**, never a missing key, so a writer whose shape
  we do not know falls through to `spent`.
- **`external`** — preflight's two live-tree refusals, an unreachable review
  grader and an external `land_failed`. A red tree was the first case (§4.2b)
  until 2026-09-24, when the `tests` rung started passing over one instead.
- **`rolled_back`** — the round landed and the guardian reverted it. Nothing
  joined those two facts before: the promotion carries the round id, the
  rollback carries only the commit. Every rollback this loop has performed has
  been a false positive, which is the argument for re-offering rather than
  against it. The landing deletes the branch, so the redo really is a redo and
  the detail says where the tree is (the `guardian-broken-*` tag).

**A promotion is a verdict however the turn ended,** and that check runs
first. #278 died at `max_turns` and the observer's ambient follow-up gated and
landed it anyway; without the short-circuit the `incomplete` rule would
re-offer a change already in `main`.

The reason travels: `reoffer_reason` puts it at the top of the next round's
prompt, because a re-offer is not a fresh start — the branch may still hold
the work. (#446's and #447's both still rebase onto `main` cleanly.)

### 4.2d A worker turn can see its wall clock

`agent.max_turns` has been warned about since `_build_state_anchor`, but
iterations are not the budget unattended work dies on. `autonomy.run_task`
got a deadline anchor on 2026-09-08 after #80, #78 and #24 each died holding
an answer they were never asked to write down; #446 is the same failure
against the autocode worker's clock. `app.deadline_anchor` is now the one
definition — autonomy delegates to it, and `/api/message/stream` takes a
`deadline_seconds` the anchor announces at 70% and 90%.

Only a caller that *enforces* a clock sends one. A chat turn has none, and
telling a human's turn it has 900 seconds left would be a lie.

### 4.2e The tree is shared: rebase and retest, tolerate dirt that is not ours

`~/lloyd` is production and a human works on `main` while rounds are open.
Until 2026-09-09 the loop treated that as an error in three places: `start`
refused a dirty tree, `preflight` refused a dirty tree or a moved HEAD
("something landed under you; abort and re-cut"), and `promote` refused both
again at landing. Each refusal threw away work that was fine — the round's
diff, to be reapplied by hand onto a base one commit newer — and on
2026-09-09 the loop's only drain was blocked three separate times by an
uncommitted file nobody was going to commit for an hour.

**A moved `main` is rebased onto, and the ladder is the retest.**
`rung_preflight` runs `git rebase <live HEAD>` in the worktree, records
`rebased: {from, onto, old_head, new_head}`, and continues. Every rung below
then judges the round's change *on top of what landed* — which is the only
build that was ever going to be live, and the one nothing had tested. Only a
conflict fails, it names the files, and `rebase --abort` leaves the worktree
exactly as it was. That failure is `external_blocker`: someone else's change
collided.

The base has three homes and a rebase must move all of them: `report.base`
(what `gate.json` carries and `land` reads), `run_spec.yaml`'s `code.base_commit`
(what the *next* `run_gate` reads), and the promotion record's `parent` /
`rollback_target`. Leave the spec's copy stale and the next gate computes the
round's diff against a commit that is no longer its parent, sweeping the
human's commits into the round's changed paths, its scope check and its
tree hash. `run_gate` writes it back whenever `report.base` moved.

**The promoter chases too, twice.** It runs detached and the human is still
committing: once before the idle wait, and once more inside the drain after
it, `promote` re-runs the gate with the *old* base — whose preflight does the
rebase — and lands the retested head. Not a second implementation of the
rebase; one. A third miss is a `fast-forward failed` and the loop stops
chasing: `main` is moving faster than it can retest, which is a reason to
say so, not to keep up. The round is left rebased, gated and open; the
reaper closes it; the branch is kept.

**A refusal at landing is now a verdict.** `land_failed` is the ledger event,
and `backlog._last_gate_per_round` reads it beside `gate` — ordered by `ts`,
because a round can pass every rung and *then* lose the race, and that later
event is the one that decides whether the item's attempt was spent.

**Uncommitted edits in production are tolerated when they are outside the
round's diff.** `merge --ff-only` never touches a file it is not merging, so
they stay where they are, in the editor they are open in. Overlap — two
writers on one file — is refused by name at both the gate and the promoter,
before the pool is paused; git would refuse the merge anyway. `start`
records the dirt on the `round_start` event rather than refusing. What was
uncommitted at landing is written to `current.json` as `live_dirty_paths`,
because the guardian's window will blame errors on the promotion and a
half-finished human edit live in the same process is the other suspect.

One consequence had to be handled before any of this was safe: the
promoter's inline rollback runs the guardian's `restore_tree`, which is
`reset --hard` after writing uncommitted edits to `broken/<stamp>/dirty.patch`.
It used to run for *pre-merge* failures too — a drain-handshake failure
stopped, restored and restarted the services for a tree that had not moved.
With dirt tolerated in that tree, that path would have stashed the human's
edits out from under their editor for nothing. A `merged` flag gates it now;
a post-merge failure still restores, and the event names the patch.

### 4.2f The suite runs in parallel; a failure is re-asked serially

Added 2026-09-18. The `tests` rung was the largest fixed cost of every round
and the one that compounds with depth: serial pytest on a 32-core box, 153 s on
09-12 and ~600 s on 09-18 (4,900 → 5,900 tests, the slowest of them 12 s — it
is slow by breadth), holding `gate-tests.lock` for all of it, so each further
round in flight queues that long again (400–470 s measured at depth 2; depth
went to 3 the same day). `Gate._run_suite` runs a FULL suite on
`automod.gate.test_workers` xdist workers, `--dist loadfile`:

| run | workers | wall | result |
|---|---|---|---|
| serial, gate env | 1 | 602 s | 5895 passed |
| trial 1–3 | 8 | 76 / 67 / 69 s | 1 failed, the same test each time |
| trial 4–6, after the budget fix below | 8 | 69 / 67 / 70 s | green, 5896 passed |
| trial 7 | 12 | 58 s | green |
| trial 8 | 16 | 56 s | green |

8, not 16: doubling the workers buys thirteen seconds, and the box is also
serving two engines and up to four turns.

**What parallelism costs is load, so a failure under it is never believed.**
The one test that failed, three runs in three, asserts that the blast-radius
rail answers inside its 90 ms budget; under eight workers the first call of
that file (it pays the graph module's imports) did not. That is a fact about
the box during the run. Every file with a failure is run again serially, and
that run is the verdict:

- it passes → the rung passes, and `parallel_only_failures` in the rung data
  (and in the pass line) names what flinched, so a load-sensitive test is a
  number on the ledger rather than a mystery;
- it fails → the failure is judged exactly as a serial run's always was: by the
  serial run's node ids, through the same base probe (§4.2b). The counts stay
  the whole suite's;
- a parallel failure that names no file (a crashed worker, a timeout), or more
  than `PARALLEL_RETRY_MAX_FILES` (40 — that is a broken tree, not load),
  re-runs the whole suite serially.

`loadfile` keeps one file's tests on one worker, in order, so a module-scoped
fixture (a booted uvicorn, a temp repo) is built once per file as before. A
partial run — the changed test files a re-gate gets — stays serial. Serial too,
and saying so (`parallel_unavailable`), whenever `xdist` is not importable in
the venv the gate runs tests with: a candidate venv, a fresh clone, a box where
nobody installed it must all still gate. `pytest-xdist` is in
`requirements.txt` and the lock; it pulls in `execnet` and nothing else.

The test itself was fixed rather than left to flinch on every gate:
`_edit_diagnostics.callers_block` bound `RAIL_BUDGET_S` as a default argument
at import, so nothing could move it, and every test about what the rail SAYS
was also a test of whether the box could load a graph in 90 ms while it ran.
It is read at call time now; the content tests take a budget no load can spend
and the budget keeps its own test. A gate rung must not depend on the wall
clock (§4, the quiet-hours tests) — nor on how busy the machine is.
`tests/test_gate_parallel_tests.py`.

### 4.3 Rungs that run candidate code run it against scratch state

Only the canary redirected `LLOYD_AUTOMOD_STATE`. The static, tests and venv
rungs executed candidate code against the **live** state directory, so a
candidate test that forgot its isolation fixture could write a real `BROKEN`
or `promotions-halted` flag, or append to the production audit trail — from
inside the gate whose entire contract is read-only judgment. `Gate._child_env`
points all three at a per-round scratch dir and sets `LLOYD_VOICE_ALERTS=0`.

**That held for one test file, from 2026-09-09 to 2026-09-18.**
`test_automod_hardening.isolated_state` — "Point the state module at a scratch
dir. Never the live one." — tore down with `delenv("LLOYD_AUTOMOD_STATE")` and
`importlib.reload(S)`. A reload re-reads the environment, so `S.STATE_DIR` came
back as `~/.local/state/lloyd-automod` for every test that ran after that file
in the session. A two-test probe under the gate's exact environment shows it:
scratch alone, production after one `isolated_state` test. It was latent until
two things met it:

- `round.land` installs SIGTERM/SIGHUP handlers and never gave them back. The
  CLI's process ends with the landing; a test's does not. A round's model
  `pkill`ed its own background suite run before gating, twice in a day, and the
  handler a finished `land("SM_L")` had left behind wrote `land_failed` for
  that fixture round into the production ledger.
- `test_a_landing_owns_its_marker…` stubs `wait_for_settle` and fakes the lock,
  and `round._land_lock` re-checks `current.json` after taking the lock and
  loops while it reads `observing` — with no sleep. Reading production's
  `current.json`, it spun at one core for as long as production had a promotion
  under observation: 822 s in one run, three of 09-18's gate runs stretched by
  three to seven minutes, and one turn slept twelve minutes on its own suite.

Fixed four ways, because each alone leaves the next one open: the fixture
restores the variable it found; `tests/conftest.py` sets both state variables
to a scratch dir at import when the caller set none, so a model's plain
`pytest` cannot address production either; `land` restores the handlers in its
`finally`; and the re-check sleeps `LAND_LOCK_POLL` before it loops.
`tests/test_automod_state_isolation.py` pins all four, the first as a child
pytest session.

### 4.3a The suite's own tree, and the HOME the layout was built for

On 2026-09-22 the production tree was deleted from inside a round — `.git`,
`.venvs`, `qmd`, `data`, `web/node_modules`, the model weights, every tracked
file, in about 35 seconds at 11:18:50. Three directories survived with their
original inodes (`~/lloyd`, `web/`, `event_logs/`) because live writers
re-created files inside them mid-walk and the closing `rmdir` hit `ENOTEMPTY` —
which is what identifies it as a recursive delete of the root rather than a
`git clean`, and the only reason the shape could be reconstructed at all: the
transcripts were among the files it removed.

What had reach: round `SM_20260922_174331`'s gate `tests` rung failed at
11:18:17 with 24 errors, and the turn re-ran the **whole suite against
production** to decide whether those failures were pre-existing — 9,716 node
ids, rootdir `/home/alansrobotlab/lloyd`, from 11:15:57 (`.pytest_cache`
re-created at 11:19:14 is the record; the run's own reasoning says "At the live
tree (base 3e8748fd)"). The exact call that deleted the tree cannot be named,
because naming it needs the transcript.

Two anchors decide which tree a test addresses, and both were production there:
`app.paths.LLOYD_HOME`, which resolves from `__file__`, and
`Path.home()/"lloyd"`. Two holes, closed separately:

- **The suite refuses production as its own tree**
  (`tests/conftest.py::_refuse_the_production_tree`, opt-out
  `LLOYD_ALLOW_LIVE_TREE_TESTS=1`). It runs at conftest import, before a single
  fixture, because that is the one point every invocation passes through
  however it is spelled. It cannot live lower down: `protected_paths` parses
  Bash COMMAND STRINGS, `pytest tests/` is not destructive on its face, and the
  deletion happened inside the test process, in Python, which is not a tool
  call. The refusal names the supported route and the opt-out — a guard whose
  exit is undocumented gets deleted rather than used. Not conditional on the
  selection: one file is the same fixtures with the same teardowns, and "how
  much of the suite" was never what made this safe.
- **The gate redirects `HOME` for candidate code** (`W.ensure_round_home`,
  `Gate._child_env(isolate_home=True)`). `worktree.py` has laid a round out as
  `<round>/home/lloyd` since the beginning *so that* `HOME=<round>/home` makes
  `Path.home()/"lloyd"` the worktree — its module docstring says exactly that —
  and `_child_env` set `"HOME": str(Path.home())` anyway, so the isolation the
  layout exists to provide was never applied. The round home is a symlink farm
  over the real one, every entry but `lloyd`: a directory holding only a
  checkout is not a home, the vault and `~/.gitconfig` are genuinely shared, and
  `shutil.rmtree` refuses a symlink to a directory, so the same farm that keeps
  the vault readable makes a teardown aimed at `Path.home()/"obsidian"` raise
  instead of run. On for the three rungs that execute candidate test code (the
  suite, the base probe, the flake re-confirm) and off elsewhere, because the
  other rungs run scripts this repo controls and `tool_choice` deliberately
  works from the live tree. Fails open to the real home and puts the reading on
  the rung's data on the pass as well as the failure — a silent downgrade here
  is indistinguishable from the isolation working.

**The two fixes collide if the guard reads production through `Path.home()`,
and the collision is worse than the bug.** Under the new `HOME`,
`Path.home()/"lloyd"` IS the worktree, so such a guard fires on every gate run
and on nothing else: the loop stops running its tests. `_production_tree()`
reads the passwd entry, the one anchor `$HOME` cannot move.
`tests/test_live_tree_isolation.py::test_a_round_worktree_is_never_refused` is
that case; if it ever fails the gate cannot test anything.

The implement prompt carries the rule as contract (`never from ~/lloyd`, and
the `base_probe` detail is where "is this pre-existing?" is already answered) —
the prompt is what stops the attempt, the refusal is what stops the damage. It
cost +190 chars and the template's bound went 5.9k -> 6.1k, the third such raise
and recorded like the other two in
`tests/test_prompt_pacing_and_ordering.py::test_the_template_stays_bounded`.

Note what this does **not** claim to fix: a test that resolves a path from
neither anchor, and anything running outside a gate with a plain environment,
still reach production if they try. The refusal is the backstop for both.

### 4.4 The canary smoke asserts only what is deterministic

It requires: a `Bash` tool_start, a sentinel round-tripping through a real
tool result, a terminating `done` with non-empty content, and no error events.

It does **not** require the model to repeat the sentinel in prose. An early
version did, and it failed on a perfectly healthy build that answered "it
printed the string you asked for". Whether the model paraphrases is model
behaviour, not code correctness — gating on it would fail promotions at the
whim of sampling.

---

## 5. Isolation: one lever

The canary runs with `HOME=<round>/home` and its worktree at
`<round>/home/lloyd`. That makes `Path.home()/"lloyd"` and
`app.paths.LLOYD_HOME` the same directory, and `LLOYD_HOME.parent/"obsidian"`
and `$HOME/obsidian` the same directory.

One lever neutralizes the sessions dir, `autonomy-runs/`, the task registry,
`workers.db`, the vault paths, and `autonomy.py`'s `AUTONOMY_DIR` — which no
config key reaches (`autonomy.task_dir` in config.yaml is dead; nothing reads
it), so an empty scratch vault is what makes `recover_stuck_tasks()` a no-op.

The worktree must be a real directory, not a symlink: `app/paths.py` calls
`.resolve()`.

A config overlay (`LLOYD_CONFIG_OVERLAY`) handles what escapes the
filesystem: ports, `workers.enabled: false` (`workers/queue.py::claim_next`
does claim-by-UPDATE, so a second pool executes *real* jobs), and a
deliberately bogus `LLOYD_SUPERVISOR_SOCK` so a canary's startup hook
physically cannot reach the live supervisord and stop the live secondary vLLM.

`secondary_enabled` is deliberately **not** overridden — it must equal live so
that reconcile is a no-op by construction.

---

## 6. Landing

`scripts/automod/promote.py`, **in a detached session**, in this order:

0. **Refuse unless the candidate is the commit the gate judged, and unless
   nothing else is under observation.** With `automod.chamber` on, `round.land`
   first *waits* for an observed promotion to settle (§3.2d) and this refusal is
   the backstop; throughout, `land.running` tells the implement source's reaper
   the round is being landed, not abandoned. `land` passed the base along but the
   candidate HEAD was re-read from the worktree, so a commit made after a
   passing gate landed completely ungated. And a second landing simply
   overwrote `current.json`: the first promotion never settled, never advanced
   the LKG, and the new record's rollback target became a commit that had
   never survived a window — quietly breaking the one invariant the
   promoter/guardian split exists to provide.
1. **Write the rollback point and read it back.** Nothing is mutated until
   `current.json` round-trips. This is the strict analogue of the xfail at
   `tests/test_autoresearch_promotion.py:372`, which used to be an
   xfail: `snapshot_current_prompts` mkdirs unconditionally, never verified the
   copy, and promoted anyway — which is why 26 of 83 historical promotions have
   no rollback point. **That half is fixed** (2026-09-08): the function now
   raises when the copy yields no file, and the test is a real one. What the
   promoter here must not copy is the older shape, not the fixed one.
2. **Pause the worker pool, arm the drain, then** idle-gate on `/health.turns`
   — three consecutive quiet polls, resetting on any activity. The order is
   the fix described in §3.2: the **TTL'd drain flag** goes up *before* the
   first poll, not after the last, so a turn arriving between the last poll
   and the restart gets a 503 instead of being cancelled mid-flight and
   nothing new starts while the count is being taken. It is re-armed every 60s
   because its 180s TTL is shorter than the wait, and released on give-up. The
   TTL is mandatory: a promoter that dies here cannot wedge the endpoint, and
   the flag is in-memory so the restart clears it anyway. The pool is *paused*
   rather than merely drained, because the drain makes a dispatched worker job
   **fail** — each refusal counts an attempt, and three attempts poison the
   job — whereas a paused pool starts nothing and lets what is in flight
   finish. Only a pause this promoter set is released; a pause a human set is
   never touched. "Idle" counts `harness_runs` as well as the session queues —
   worker jobs call `run_query` directly and never enter a queue, so a
   ten-minute research job was invisible to the gate that exists to avoid
   killing it.
3. `git merge --ff-only`, swap the venv if one was built.
4. **Apply changed service definitions**, then restart MCP then backend, and
   **verify `/health.commit` and `boot_id` both changed** — the only proof the
   restart picked up new code rather than a stale process still serving. The
   restart lease is refreshed before each leg: it is 120s and each leg may
   legitimately spend 90s waiting on health, so one lease taken before the
   merge could expire mid-restart and hand the guardian its own deploy to
   judge.

**The caller must then stop talking.** The idle gate counts the calling turn
too, so an agent that lands and then polls `automod_status` in a loop is itself
the reason the backend never goes idle — the landing waits its full 15 minutes
and gives up. This is not a bug in the gate: the landing restarts the backend,
which ends that turn regardless. `automod_land` says so in its own result, and
the skill's step after "land it" is "end your turn". It only shows up when the
loop is driven by the agent rather than from a terminal, which is exactly the
path that had never been exercised.

**Landing a *definition* is not the same as landing code.** `spec.py` calls
the supervisor confs, the systemd units and the guardian "protected" — allowed
to change, provided the drill passes — but nothing ever *applied* such a
change. supervisord includes `conf.d` straight out of the repo and needs
`reread`/`update` to notice; systemd reads `~/.config/systemd/user`, so a unit
edit in the repo reached nothing at all; and the guardian runs from a pinned
snapshot re-staged only on a unit restart. A round could pass a full drill,
land, be marked healthy, and leave the running system on the old definition
indefinitely. `_apply_service_changes` closes all three, and
`round status` reports `unit_drift` from any other cause.

Any failure between 3 and 4 reverts inline rather than waiting for the
guardian's next tick.

**On the land train (§3.2i) steps 2 and 4 move to the flush.** A deferrable
landing does 0, 1 (the pending entry is its rollback point), the chase, the
squash and 3, and stops: no window. The flush later does 2 and 4 once for the
whole batch and writes the one record. An eager landing (a venv,
`agent-services/`) still does all of it inline, carrying the batch.

**The observation window starts at step 4, not step 1.** The idle gate may
legitimately wait fifteen minutes; a window started early would be mostly
spent before the build existed. The record is written in state `landing` with
null windows, and the guardian treats `landing` as nothing to observe.

---

## 7. The guardian

`agent-services/guardian/`. A **systemd `--user` unit**, not a supervisord
program, for three reasons:

- `agent-supervisord.service` sets `KillMode=control-group`, so every
  supervisord child dies when that unit restarts — exactly when a watchdog is
  needed.
- Its remediation set includes "restart supervisord". A child cannot restart
  its own supervisor and survive.
- supervisord parks a program in FATAL after `startretries` and never un-parks
  it; systemd `Restart=always` with `StartLimitIntervalSec=0` never gives up.

`StartLimitIntervalSec` belongs in `[Unit]`. In `[Service]` systemd silently
ignores it and applies the default 5-starts-in-10s limit, which would let the
watchdog rate-limit *itself* into a failed state.

**Stdlib only, on `/usr/bin/python3`**, and it never imports from `app/`. Not
even `app.supervisor_client`, which has no third-party imports *today* — the
moment it grows an `app.config` import, a malformed `config.yaml` would take
down the watchdog along with the patient. The AF_UNIX XML-RPC transport is
copied rather than imported, and the code says why.

### 7.1 What guards the guardian

`ExecStartPre` stages the source into a pinned snapshot at
`~/.local/state/lloyd-guardian/bin/` **only if** it compiles and passes its own
selftest. `ExecStart` runs the snapshot. So a SyntaxError written into the
guardian degrades to a *stale* watchdog, never to *no* watchdog. The unit
invokes staging with a leading `-`, so even a crash there cannot prevent a
start. Verified by breaking it on purpose.

Lloyd may edit the guardian. A diff touching it triggers the `drill` rung: the
drill boots a deliberately unbootable build under a canary, runs the
**candidate** guardian against it, and requires that the tree came back and
the service returned. A bound on *how* it changes, not a ban on changing it.

### 7.2 The liveness predicate

Deliberately inverts `app/supervisor_client.py::_health`, whose comment reads
"port being open is the strongest signal — trust it over supervisord state".
Correct for the Services page; catastrophic for a watchdog, because a FATAL
backend whose port is held by a zombie would read "healthy".
`tests/test_guardian_predicates.py` asserts the divergence side by side so it
stays deliberate.

Supervisord state is consulted **first**, FATAL is decisive, and a probe may
only *add* failure. It also catches the crash loop that never reaches FATAL —
`autorestart=true` plus too small a `startsecs` retries forever while every
sample says RUNNING — by watching distinct spawn timestamps rather than
`statename`.

**Probe failures are classified by cause**, and there are three causes, not
two. A refused connection means nothing is listening: 3 ticks. A timeout means
the socket accepted but the app was busy: 24 ticks (2 minutes). See §9 for why
that distinction cost a false rollback to learn.

The third is an **answered non-200**, and folding it into "refused" was wrong
in both directions. The aggregator returns 503 whenever *any* module is
degraded — a closed Thunderbird bridge — so three ticks of an ordinary,
expected condition read as a dead aggregator, while the careful
newly-degraded-since-LKG check sat *after* the down predicate and never got a
vote. The aggregator's 503 is now judged only by `mcp_degraded_is_fatal`, and that
verdict is itself **confirmed across 3 ticks** — it fires on a body reporting
zero tools, and an aggregator answering mid-restart parses to exactly that, so
one bad response would otherwise revert a promotion on its own. Every other
detector here confirms across ticks; this one used to be reached only via a
503, which hid how sharp it was. The backend's own 503 is real (a router that
failed to mount, a startup event that never completed) and gets its own much
wider budget: 36 ticks.

**A deliberate stop is not an outage.** `STOPPED` and `EXITED` shared a branch,
so when flap protection quarantined the backend — the guardian stopping it on
purpose — the watchdog then alerted about the outage it had itself created,
every 15 minutes for as long as the quarantine lasted. `STOPPED` while
promotions are halted is excused; `EXITED` never is.

### 7.3 Four invariants

1. **`HEAD == last-known-good` ⇒ never roll back.** Everything on fire with
   nothing promoted is infrastructure, not a bad change.
2. **No promotion under observation ⇒ never roll back.** HEAD legitimately
   differs from LKG most of the time — a human commit, a nightly job.
   Reverting then destroys work nobody asked the guardian to judge.
3. **Unreachable supervisord restarts the unit**, never reverts code.
4. **A promotion that is no longer in history is never re-reverted.** It may
   have been undone by hand, or by a promoter that failed after its merge and
   reverted itself inline. Its recorded `rollback_target` still points
   somewhere real, so every other check passes and the guardian would happily
   rewind the tree a second time, discarding whatever landed since. Absence
   from `git merge-base --is-ancestor` is the tell — and it is asked of the
   commit being **blamed**, not of the record sitting in `current.json`: a stale
   duplicate request naming an already-reverted commit otherwise passes every
   gate while a live observation record vouches for someone else, which is how
   one rollback could become two (#1358).

### 7.4 Rollback order

**Reset when HEAD is still the commit being *blamed*; revert in place when it is
not.** `reset --hard` to a promotion's parent is only correct while HEAD *is*
that promotion. Nightly jobs commit straight to live `main`, so an observation
window can legitimately close over work the loop never touched, and resetting past it
destroys commits nobody asked the guardian to judge. That is the 26-commit
incident one level down: there the wrong *target* was chosen, here the right
target is reached by the wrong *route*. When HEAD has moved on, the guardian
reverts exactly the blamed commit and leaves the rest standing; a conflict
has no safe automatic answer, so it escalates instead of guessing.

**Blame and observation are two different questions** (#1358, 2026-09-24). The
sentence above says *the commit being blamed*, and that word is load. Two things
reach `do_rollback`: a liveness trip, which blames the promotion under
observation because that is the only change it can see, and a written rollback
request, which names a commit the detached quality check measured and may blame a
promotion that **settled** hours earlier while a newer one sits under observation.
Until #1358 the requested commit was read only when `current.json` was absent, so
a request that arrived while a window was open had its blame dropped and the
observed promotion was punished for it: HEAD *was* the observed promotion, the
route was `reset`, and both changes left `main` while the ledger row named the one
nobody blamed (2026-09-21, `a802b979`→`1e219da9`, then `dbec85aa`→`edc8ec60`). So
now the blame is resolved once, at the top of `do_rollback`, and the route, the
is-ancestor guard, the denylist and the alert all read *that*; a request whose
commit differs from the observed promotion's also makes that record's
`rollback_target` unreachable, because it describes the tree before some *other*
promotion. A promotion under observation and not blamed has its window **closed
unjudged** (Alan's decision, 2026-09-23): not settled, so LKG does not advance to
it, and `rollback_succeeded.left_unjudged` names it beside the reverted commit.
Reopening it across the rollback's own restart would convict it of the rollback,
and every rollback this loop has performed so far has been a false positive.

**A batch is one rollback, and only the batch** (the land train, §3.2i;
`gstate.BATCH_SCHEMA = 2`). A record carrying `commits` (two or more, oldest
first, ending in its `commit`) resets to its `rollback_target` only when HEAD is
the newest batch commit **and** `rev-list target..HEAD` is exactly the batch
(`rollback.range_is_exactly`); anything else — a nightly commit between two
landings, which 127 of 379 first-parent commits in a week made routine, or HEAD
moved past the batch — reverts every batch commit in place, newest first
(`rollback.revert_commits`). A conflict there takes back the reverts it already
made, leaves the tree exactly as found, and escalates without a retry
(`RevertConflict`: the same commits conflict the same way every time). Batch
commits already gone from history are not reverted twice. Every batch commit is
denylisted by its own content (`entries`, else `git diff-tree`), the row carries
`commits` and `batch`, and `settled` is written once per commit. A request
blaming ONE commit of an observed batch (the detached regression check) is the
#1358 case: that commit is reverted, the batch's window closes unjudged. A
request may name a batch itself (`commits`), which is how the promoter hands
over a flush it could not undo. A record without `commits`, or with one, is
judged exactly as before — `tests/test_guardian_rollback.py` pins both.

**The pointer must not outlive the change it certifies.** `maybe_settle` used to
be the only writer of `last_known_good.json`, so a rollback that removed the
commit LKG named left the pointer aimed at a dead commit — and
`rollback_target(None)`, asked with no `current.json`, hands that dead commit to
the *next* rollback, which would reset `main` back onto it and discard everything
since (2026-09-21: `dbec85aa` settled 21:37:32Z, reverted 21:45:05Z; the pointer
was cleared by hand with `round bless` at 2026-09-22T01:12:49Z, row
`b444b25de`, note "blessed by hand"). `_rollback_once` now repoints
LKG at the commit it restored whenever the rollback took the certified commit out
of the restored history, and says so in the row as `lkg_repointed_from`.

Stop the writers *before* moving the floor. The agent is what writes into this
repo, and `git reset --hard` during an `Edit` produces a half-applied revert —
strictly worse than either version. This is why both supervisor confs gained
`stopasgroup`/`killasgroup`: without them a Bash tool's child outlives the
stop.

**"Writer" means a write fd on the code tree**, not a process whose cwd is
somewhere under the repo. The cwd predicate was true of every shell, editor
and agent on the box, so the drain was never empty: it burned its full 20s on
every rollback and logged 44 pids that meant nothing, hiding the one case it
existed to detect. Scoping matters as much as the fd check — supervisord holds
`logs/server.err` open for append forever, so an unscoped fd scan just swaps
one always-true predicate for another. Only `.git` and the code directories
count.

**The rescue must not outlive the watchdog.** `WATCHDOG=1` was sent once per
loop iteration and a rollback *is* one iteration: two blocking stops at
`stopwaitsecs=15`, a writer drain, and up to 90s waiting for the backend —
comfortably past `WatchdogSec=90`. systemd killed the guardian mid-rescue, it
restarted, the resume path began the same rollback again, and it was killed at
the same place: a loop that can never reach `BROKEN`, in exactly the situation
`BROKEN` exists to report. Every slow step now beats. Deliberately *not* from
a background thread — that would keep the watchdog satisfied for a genuinely
hung guardian, which is the whole thing `WatchdogSec` is there to catch.

The XML-RPC client timeout must exceed `stopwaitsecs`, or a blocking stop that
is working reports failure and the rollback proceeds without knowing whether
the writers are actually down.

Evidence is preserved before it is destroyed: HEAD is tagged
`guardian-broken-<ts>`, the dirty tree is stashed by name, untracked sources
are copied aside. A rollback that erases the bug guarantees you fix it twice.
This is not decorative — it is how `a6c0eba` and a set of in-flight nightly
changes were both recovered after the incident in §9.

`git clean` is **path-scoped**, never repo-root: the root holds `.env` and
`.venvs/`, gitignored and not replaceable. (The databases, sessions and logs
used to sit there too; since 2026-09-22 they live in `~/lloyd-data`,
[[data-home]].)

If rollback fails twice, services are left **stopped** and `BROKEN` is
written. With no human in the loop, an honestly-dead system is safer than an
autonomous agent running half-reverted code.

---

## 8. Detectors

| Trigger | Window | Source |
|---|---|---|
| crash / won't boot | ~2 min | supervisord state + `/health` |
| error-rate spike | ~15 min | `~/lloyd-data/logs/server.err` by byte offset |
| data damage | ~15 min | KG row count, vault note count (`vaultwatch.measure`, `.git/**` excluded) |
| retrieval-quality regression | nightly | paired `eval/run_eval.py` — retrieval only; no post-landing agent-loop axis (§8.1, §13) |

**Errors come from `server.err`, not `server.log`.** `server.py`'s
`basicConfig` writes to stderr, so all application logs land in `.err`;
`server.log` is uvicorn's access log and contains zero error-shaped lines. A
watchdog grepping it would find nothing forever.

Log reading is by `(device, inode, offset)`. Rotation is by rename, so the
inode changes and the predecessor's tail must be drained before starting the
new file. Signatures normalize digits, hex, paths and durations, so
`…1266 min old` and `…1300 min old` collapse to one.

Anything seen in ≥3 distinct hours over 7 days is **chronic** and can never
fire. Production has emitted `autonomy scheduler may be stalled` hourly for
days; a detector that counted it would revert on its first tick. Ten such
signatures were learned on first boot.

**The chronic set expires after 24 hours**, in the cache *and* in the process.
Learned once and kept forever, it describes the box as it was on the first
boot after the state dir was created — so every recurring error that starts
happening afterwards stays "novel" indefinitely and the next promotion is
reverted for a steady-state failure it had nothing to do with. That is the
same shape as the stale log cursor: a detector quietly judging a new commit by
old evidence.

**Data damage is the class `git reset` cannot undo.** The KG and the vault are
gitignored, so a change that deletes rows or notes boots fine, logs nothing,
passes every eval, and *survives* the revert. Two counts cover it.

### 8.1 Retrieval-quality regression: measured, not assumed

The autoresearch composite is **not** used. Three identical baseline runs
scored 0.719 / 0.542 / 0.624 — spread 0.177 against a 0.05 threshold, and 61
of 83 historical promotions were decided inside that noise.

`eval/run_eval.py` has no LLM in it. Measured here, five consecutive runs
against an unchanged vault produced **identical** values for every quality
metric (stdev 0.0000 for entity_hit_rate, entity_recall_avg,
fact_entity_recall_avg, ndcg10, mrr_doc, doc_hit_rate, doc_recall_avg). Only
`latency_ms_avg` moved, at 562ms stdev — and it is the one metric that cannot
take part in a paired comparison: the qmd daemon caches query embeddings, so the
same text returns 20-34x faster than unseen text (priced 2026-09-18 at production
width — pool 240, rerank on, a fresh query text per sample: 3,672-4,027 ms for a
fresh query against 119-183 ms for the identical repeat). A before/after latency delta therefore measures arm
order, not the change. So it is compared against a number instead of against the
other arm — the per-context absolute ceiling `LATENCY_BUDGET_MS` (4,800 ms for
the nightly `eval/run_eval.py` context, 1,600 ms for the pinned paired check) —
and an over-ceiling average is the named verdict `latency_over_budget`, written
into the `regression_check` event and `eval_last.json` as a report. It never
enters `regressed` or `reasons`, so it cannot request a rollback (§13).

So the eval contributes no noise — and the real confound is **vault drift**.
Cross-day baselines differ by up to 0.044 because the vault changed, not the
code. The check is therefore a **paired same-data A/B**: the last-known-good
commit is checked out into a scratch worktree and pointed at the *live* fact
tree and KG via `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB` — env vars that exist for
rebuilds, used here in reverse (old code, current data). Both arms run in the
same window, so drift cancels.

A missing noise file means "cannot evaluate", never "no regression".

**It runs after landing, not as a gate rung, and that is forced.** The canary
is deliberately data-isolated — `HOME` redirection gives a candidate build an
empty scratch vault, which is what stops a bad promotion touching live
sessions, `workers.db` and the autonomy dir. A quality eval run there scores
an empty corpus and measures nothing. The check has to see live data in a
separate worktree, which is exactly what the paired A/B does.

Three things had to be fixed before it could ever fire:

- **It keyed on `current.json`, which is deleted at settle.** A once-a-day job
  therefore found "no promotion under observation" essentially always — this
  source has zero runs in its entire history. The guardian now writes
  `last_settled.json` at settle, carrying the promotion's own parent.
- **It compared against the LKG commit, which by then *is* the promoted
  commit.** Both arms would have checked out the same code: a comparison
  structurally incapable of finding anything. The baseline is the parent.
- **An empty corpus was indistinguishable from a healthy one.** See below.

**Not every armed metric can see the graph.** Measured 2026-09-06 against an
empty `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB`: with the graph deleted entirely,
`mrr_doc` (0.468), `ndcg10` (0.563), `doc_hit_rate` (0.90) and every
per-category `mrr_doc` came back **identical** to the real run. Two reasons —
`RECALL_GRAPH_RERANK` is `False`, so the graph never reorders documents, and
the document leg queries the qmd daemon over an absolute URL that keeps
working regardless. Only `entity_hit_rate`, `entity_recall_avg` and
`fact_entity_recall_avg` moved, to zero.

**The document corpus is pinned, and it has two halves.** The comparison
already held the fact tree and the knowledge graph still. It now also serves
both arms from a frozen qmd snapshot (`scripts/automod/evalpin.py`: qmd keeps
its index in one SQLite file and takes `--index <name>`, so a `VACUUM INTO`
copy on a second port is a 1.5-second freeze) and points both at one
`LLOYD_CODE_ROOT`.

That second half is the surprising one. **This retriever greps the repository
it ships in** — `_grep_lloyd_code` searches `agent_mcp/`, `app/`, `scripts/`
and `workers/` under its own checkout — so the code under test is also part of
the corpus, and each arm was searching its own source. Pinning qmd alone did
not fix the false positive: the arms still differed by exactly -0.0060, and
the query `lloyd-vllm-rel` returned six different files per arm. With both
halves pinned the arms agree to 0.0000 on all seven metrics, and four repeat
runs move 0.0000.

**All seven are armed:** `entity_hit_rate`, `entity_recall_avg`,
`fact_entity_recall_avg`, `ndcg10`, `mrr_doc`, `doc_hit_rate`,
`doc_recall_avg`. Both the count and the names are pinned:
`tests/test_automod_doc_claims.py` fails if the doc states a set different
from `ARMED_METRICS`. The pin is a precondition:
`execute` refuses rather than falling back to the live daemon, which would be
the old broken comparison wearing the new name.

**It is not free.** Measured: one eval run is 82s wall and **128 seconds of
qmd CPU**, and a paired comparison is two runs plus a second embedding model.
The cost is fan-out rather than embedding — `_vault_recall` queries each of the
twelve vault segments separately, so a twenty-question run makes 240 qmd
requests. Fine once per promotion, which is what the dedup enforces. Not fine
in a loop: re-measuring the noise floor repeatedly during development loaded
the live daemon and slowed real retrieval for everything else on the box.

qmd itself is GPU-accelerated (3.6 GB resident, spiking to 93% on a vector
leg). An earlier revision of this section claimed it was CPU-only; that came
from piping `nvidia-smi --query-compute-apps` through `head`, which cut qmd off
at entry fourteen of nineteen.

**The armed set was cut down once, and the size it was cut to came from being
wrong.** The original set was chosen because five consecutive runs gave stdev
0.0000 for all of them — a real measurement of the wrong thing. It describes
repeatability inside one short window; the paired A/B runs its arms *minutes*
apart, and it cancels drift only in what `LLOYD_FACTS_ROOT` and `LLOYD_KG_DB`
redirect. The document leg came from neither, and before the pin it came from
no pin either: the live qmd daemon at an absolute
`http://localhost:8181/query`, over a vault being written continuously by
nightly jobs and session capture.

The first real run proved it. On a promotion whose entire diff was text inside
an inject string — incapable of touching retrieval — it reported `ndcg10`
0.5680 → 0.5620 and `mrr_doc` 0.4740 → 0.4680, both "beyond 3σ", and asked for
a rollback. Three back-to-back runs of identical code and data then gave
0.0000 spread on every entity metric and **0.0250 on `doc_recall_avg`**, eight
times its own tolerance. The document leg, unlike the fact side, was not stable
across the window this check spans — so the four document metrics were pulled
out of the armed set, which left every graph-to-document change unfalsifiable.

That conclusion was right about the symptom and wrong about the cause: the leg
was unstable because each arm measured a *different* corpus, not because
document retrieval cannot be measured. Pinning both halves (above) made the
four repeatable, and they are armed again — `evaluate` loops `ARMED_METRICS`
and appends a rollback reason for any drop past `SIGMA_MULTIPLIER × σ`, so a
`doc_recall_avg` delta is now the line that stops a promotion, not noise to be
ignored. The limit that survives is `latency_ms_avg`: it still has no tolerance
and cannot make a comparison regress — `regressed` and `reasons` remain the seven
quality metrics and new eval errors, which is the only channel to
`request_rollback` — but it is no longer merely recorded. Since #1129 it is
compared against an absolute per-context ceiling, `LATENCY_BUDGET_MS` (4,800 ms
nightly, 1,600 ms for the pinned paired check), and an over-ceiling average is
the named report `latency_over_budget` — §8.1.

**A query the daemon did not answer is not a score of zero.** On 2026-09-17
05:34Z the check reverted #1194 — a change to `scripts/memory/revert-suffix-
merges.py`, nothing on the retrieval path — for `doc_hit_rate` 1.00 → 0.95,
`ndcg10` −0.022, `mrr_doc` −0.012, `doc_recall_avg` −0.025 and
`fact_entity_recall_avg` −0.025, all "beyond 3σ = 0.003". Every one of those
deltas is exactly one query: `backlog-363` came back from the pinned qmd
daemon with **zero documents and no error** (its fact leg resolved fine), and
0.4307/20, 0.25/20, 0.5/20 and 0.5/20 are the five numbers above. The same
code answered it with twenty documents four hours later. The eval itself is
still deterministic — re-measured that day, five pinned trials agree to
0.0000 on all seven armed metrics — so the tolerance is not the fix; a floor
wide enough to absorb one dropped answer (0.05) would absorb a real one-query
regression too. Instead `_load_run` records which queries the document leg
answered with nothing (`empty_doc_queries`, off `result_summary.n_docs`), and
an arm with a *strict subset* of its queries unanswered is `regression_skipped`
("did not answer, cannot evaluate"), never a score. Every query empty stays a
score, because that is the one shape a change under test can produce: a
retriever that returns nothing regresses to zero and is reverted.
`measure_noise` drops such trials for the same reason.
`tests/test_automod_regression.py` pins both halves.

**A floor cannot be narrower than the number it grades** (#1352, 2026-09-22). The
threshold `evaluate` applies is `max(k·σ, 1/n_queries + 0.001)`, where `n_queries`
is the denominator BOTH arms report — not a constant. Both terms are arithmetic
about the score, not a budget for noise. Every armed metric is `avg` over the
arm's scored records, so one question's whole contribution is `1/n`, and for the
two hit-rates the per-query score IS 0/1, which makes that the smallest non-zero
move the metric has at all; the reported value is `round(mean, 3)`, so a further
rounding step is added because the difference of two reported values can overstate
one question by up to that step. At the 87-question set live on 2026-09-21 the
floor `3 × MIN_SIGMA` produced was 0.0030 against a quantum of 0.011494, and the
three `doc_hit_rate` values that evening — 0.529, 0.517, 0.506 — are 46/87, 45/87
and 44/87: one question flipping three times, each booked as a regression "beyond
3σ" and two of them spent reverting a commit. Any tolerance below `1/n_queries` on
a 0/1 metric is guaranteed to fire on a single question, which means the rung
could not express "no change" at all. The σ and the winning term are published per
metric in the check's record as `sigma`, `sigma_source` (`measured`, or
`min_sigma_floor` when the artifact carried none), `resolution`, `n_queries`,
`floor` and `floor_governed_by`, so the floor and the σ behind it are read side by
side instead of inferred from a constant. `MIN_SIGMA` survives as the σ term's
fallback, which is what the `min_sigma_floor` label reports.

**A stale floor is no verdict** (#1352, 2026-09-22). `queries_fingerprint` matched
against the live question set decides: a mismatched artifact records its deltas and
names them in the ledger reason, and does not request a rollback — including the
second look, which is skipped because its only purpose was to justify a request
that cannot be justified. The summary distinguishes itself (`STALE FLOOR … no
verdict`) from both a clean pass and a refused-on-second-look pass, for the
same reason `empty_fact_leg` does: a check that reports "no regression" while its own
flag says its floor did not apply is a clean bill it cannot issue, and that is
exactly the state every check on 2026-09-21 was in. What the flag used to be —
"provenance, not a gate" — is explained in §9. The paired σ the confirm arm screens
magnitude against comes from the artifact's FRESH-ranker bucket, since that arm
draws djev independently rather than replaying it.

**And the three fact-layer metrics are named for what they actually read.**
Two degradation drills against copies of the live store:

| Degradation | Armed metrics | Doc metrics |
|---|---|---|
| 70% of `facts_idx` rows deleted (205,689 → 61,707) | `entity_hit_rate` 0.60 → 0.55, `entity_recall_avg` 0.443 → 0.433 — **fires** | unchanged |
| 70% of **active edges** expired (4,029 → 1,209) | unchanged — **blind** | unchanged |

So they are `FACT_LAYER_METRICS`, not "graph sensitive". Edge quality has no
armed metric at all: a change that expires most of the edge set walks past
this check in silence, and that is a stated limit rather than a covered case.
Naming them for the graph would have been the same overclaim the detector
exists to prevent.

**And the section used to be named for an axis it cannot see.** This check was
called the behavioural detector while every metric in `ARMED_METRICS` scored a
retrieval query. No armed metric observes a tool call, a turn count or a model
decision — `eval/run_eval.py` issues no model request, so a change that dropped
`Grep` from the baseline tool set, moved the compaction thresholds or changed
turn accounting left all seven metrics unmoved and produced a clean report which
then became the promotion's quality baseline. Not "nothing observes the loop":
the gate runs a scored loop-side check (`prompt_surface`, §4) — but only
*pre*-landing, and only for the five path names `prompt_builder.py`,
`prefetch.py`, `SOUL.md`, `MEMORY.md`, `USER.md`. A loop-side change outside
those five has no check at either end, which is why §13 states it and why the
measurement artifact itself now carries the claim (`axis`: what it measured, and
the two axes it did not).

This is also why `eval/run_eval.py` now records **which corpus it scored**
(`corpus_ok`, the resolved paths, entity and edge counts) and refuses an empty
one unless `--allow-empty-corpus` is passed. An arm that scored an empty
corpus is not a low score, it is a measurement that did not happen, and the
worker treats it as "cannot evaluate".

The measurement is handed to the guardian in `eval_last.json` and folded into
the LKG record's long-empty `eval` slot at settle. Read, never written, by the
guardian: `last_known_good.json` has exactly one writer, and that is what
makes it mean *observed healthy in production*.

#### 8.1a Every promotion is measured, by a check a landing cannot kill

Added 2026-09-18. The day landings sped up (gates went from ~16 min to ~7) the
ledger said what the detector had really been doing: **8 of 17 promotions
measured, 0 of the last 4** — and the five checks that finished after 11:00
compared 0.0 with 0.0 on every document metric and recorded "no regression".
Seven causes, stacked, and none of them new that day. The last was found only
after the first six were fixed and a real check STILL timed out on every
question — it had been under all of them:

| # | what | consequence |
|---|---|---|
| 1 | `promote.wait_idle` counted agent turns (`/health.turns`); the check is `subprocess.run` evals on a thread | a landing restarted the backend under a running check |
| 2 | the pinned qmd daemon runs in its own process group (`process_group=0`, so `stop` can take the tree), and "still a child" was the whole guarantee it would not outlive its owner | supervisord's group kill took the check and left the daemon: one orphan held :8182 for 5 h 30 min and burned two CPU-hours |
| 3 | the pin's port check asked `127.0.0.1`; qmd binds `localhost`, which is `[::1]` here and nothing else | the check could not see a qmd daemon at all. "Refusing to compare against something I did not start" was never recorded once: a second daemon was started on top of the orphan and died on `EADDRINUSE`. If it died before the first start-up probe the skip read `pinned qmd exited immediately (rc=1); see <log>` — four times, each naming a log the caller had already deleted. If it took a second longer, the ORPHAN answered the probe and both arms ran against a daemon the check did not start: that is the five zero-against-zero checks. Which one a check got was a race |
| 4 | `workers.round_hold` keeps the source unclaimed while a round is in flight, and one always is | a check could only START in the seconds after a restart — the one moment guaranteed to be followed by another |
| 5 | it measured "the latest promotion" (`current.json`, else `last_settled.json`) | a promotion that landed while the previous check ran was measured by nobody |
| 6 | qmd keeps working on a request its client abandoned, so once ONE recall outlasts the retriever's 15 s client timeout every later one queues behind it: 116–160 s a query by the end of an arm | every query empty in BOTH arms, which "every query empty stays a score" turned into zero-against-zero: no regression |
| 7 | the pin RESTATED how production runs qmd (the published CLI, three CUDA variables) instead of reading it, and the restatement went stale the day after it was written: the daemon moved to the fork on 2026-09-07 with `QMD_RERANK_PARALLELISM=4` and a 1200-char rerank window. Invisible while a recall reranked 40 rows; #504 made it 240 | the ledger dates it to the hour: the paired check averaged **0.49–0.69 s** a question through 2026-09-14, **11.0–12.9 s** from the first check after #504, and 15.05 s — every recall at the client's timeout, nothing retrieved — from 2026-09-18 18:03Z, once anything else shared GPU 0. A 12 s average under a 15 s timeout also loses single questions from one arm or the other, and one of those was the false "regression" of 2026-09-17 05:34Z (`doc_hit_rate` 1.00 → 0.95), which `unanswered_doc_queries` then treated as a symptom. One snapshot, a fresh question per sample, one request at a time: **16.3–20.1 s** on the pin's build and settings, 14.3–15.7 s on the fork with the pin's settings, **4.6–5.5 s** on the fork with production's. It is the settings. `LATENCY_BUDGET_MS["paired_check"]` (14 s) was calibrated against the defect and was re-derived on 2026-09-24 (#1247) from the ledger, by one rule — twice the worst average either arm read in the current era, rounded up to 100 ms — over the `regression_check` rows whose pin names `agent-qmd-daemon.conf`, date-bounded three ways: readings before a54ccda (2026-09-19T01:22Z) are excluded as this row's defect; aa6bee8 (2026-09-20T05:07Z, `QMD_RERANK_CONTEXT_SIZE=2048` in the daemon conf) is a second break inside the corrected-pin series (median 4,305 ms before it, 4,257 after, both cross-encoder); and e7bb4280 + c6e79b45 (2026-09-21T19:45Z / 2026-09-22T00:50Z, djev ranks the recall and the check replays it per request) open the era the number comes from — 98 rows from 2026-09-22T04:08Z to 2026-09-24T13:46Z, graded arm median 161.5 / max 274.9 ms, fresh arm max 752.3 ms, so **1,600 ms**. The 2026-09-20 tail (11,561 / 11,868 / 16,256 ms on three healthy runs) is deliberately not cleared: the verdict is report-only, and the tail was never attributed. `nightly` stays 4,800 ms — the wipe left two nightly artifacts (766 and 541 ms), not a population |

What runs now:

| what | where | rule |
|---|---|---|
| the comparison runs detached | `automod_regression.run_pending`, spawned by `start_runner` (the pool job, now milliseconds) and by `promote._start_regression_runner` the moment a landing is verified | its own session (`state.spawn_detached`), like the gate and the landing: a restart cannot reach it, it holds no pool slot, and no landing waits for it. `regression.lock` (flock) makes it one at a time; log in `regression.log`. Entry point `python -m scripts.automod.regression_runner run|pending` — its own module, because `-m` on the source module executes it twice (the registry imports it first). A promotion that comes back "cannot evaluate" for the last time is announced once (`promote.announce`): the runner's log is a file nobody tails |
| one check per promotion | `pending_promotions` | read off the ledger: every `promoted` row of the last 24 h that was not rolled back, has no `regression_check`, and has not come back "cannot evaluate" `MAX_SKIPS_PER_COMMIT` (2) times. Oldest first; the queue is re-read after each check, so what lands meanwhile is picked up by the same runner; a regression stops it so the guardian acts first |
| each check names what it compares | `check_promotion(subject, stage)` | `commit` against ITS `parent`, BOTH from a scratch worktree — the current arm used to run the live tree, which is the promoted commit only until the next landing. Every `regression_skipped` row carries the `commit` it could not evaluate |
| the pin dies with its owner | `evalpin._die_with_parent` | `PR_SET_PDEATHSIG` via `preexec_fn`, libc resolved before the fork. `test_regression_runner` SIGKILLs a real owner and watches the child go — and, in the control, watches it survive without it |
| an orphan is reaped, not refused | `evalpin.reap_stale`, first thing in `__enter__` | found by what it IS — a process of this user serving this `--index` on this `--port` whose parent is init or the user systemd — not by whether the port answers, which is the check that could not see it. A pin with a live parent is a run in progress and is left; anything else on the port is still refused. `port_free` asks both loopbacks, once, and only a refusal means free |
| an answer counts only from the pin's own daemon | `PinnedCorpus.__enter__` | the start-up probe is believed only while the process it started is still alive |
| a pin that dies at start says why | `PinnedCorpus._abandon` | the daemon's last 300 characters go into the `PinError` (and so onto the ledger), the daemon is stopped and the 1 GB snapshot removed — `__exit__` never runs for a `with` whose `__enter__` raised |
| the pin serves production's retriever | `evalpin.production_daemon`, `pin_command` | argv and environment read from `agent-services/supervisor/conf.d/agent-qmd-daemon.conf`, on the pin's port and index. A conf it cannot read, or a CLI that is not there, is the published build and `source: "fallback: …"`; either way `pin.daemon {cli, source, settings}` rides on the `regression_check` row |
| nothing is timed against a pin that cannot keep up | `PinnedCorpus.warm_up`, `production_payload` | the recall production sends — its pool and collections read from `agent_mcp.vault`, pinned against `_qmd_daemon_search` by a test — until one answers inside 10 s, two thirds of the client's timeout (7–11 s cold, then 4–5 s, measured). A DIFFERENT question every try: qmd caches a rerank score per (query, chunk) in the index it serves, so a repeated question comes back in ~0.2 s and passes the slowest daemon on its second try. The first cut asked for 30 rows, took 3.4 s, and waved through a daemon that then took 18 s a recall. Never → `PinError` → "cannot evaluate" |
| what a killed check left is removed | `automod_regression.sweep_stale_scratch` | by the runner, under its lock: `automod-eval-*` checkouts (each a registered worktree of the LIVE repo — three were, that day) and `automod-pin-*` work dirs older than 2 h, which is older than any check can be |
| an eval may wait longer than a live turn | `vault._qmd_timeout`, `LLOYD_QMD_TIMEOUT_S` | 15 s for production, 60 s for an eval arm; read per call. An arm whose code predates the override keeps its 15 s |
| zero against zero is not "no regression" | `all_queries_empty(baseline)` | the BASELINE answering nothing is never the change — that code was live and answering — so it is `regression_skipped`. Only the CURRENT arm answering nothing stays a score, which is the one shape a change can produce |
| a regression has to reproduce | `check_promotion`, `_would_regress` | while the pinned corpus is still up, a current arm that would be reported as a regression is run a SECOND time (`automod-check-confirm`). Under a pinned corpus the armed metrics are deterministic, so a real regression comes back the same; one that does not is recorded `regressed: false` with `unconfirmed_reasons` — a finding about the instrument — and a second look that fails or loses a question is "cannot evaluate". Both of this check's own rollback requests were false positives (2026-09-07 ndcg −0.006; 2026-09-17 one question lost to the client's timeout), and the queue now reaches promotions up to a day old, where a revert is a commit on top of other people's work. Costs one arm, only when it matters |
| the idle gate sees every pool job | `promote.wait_idle` | with `turns` quiet it also asks `pool_in_flight()`; a job in flight under the promoter's pause is finite, and `idle_hard_max_wait_s` still bounds the whole wait |
| the job is exempt from the round hold | `workers.round_hold.exempt` | it needs no engine and takes milliseconds |
| and is offered only when it would do something | `automod_regression.enqueue_if_due`, `_runner_needed` | a promotion is owed a measurement AND no runner holds the lock; otherwise no queue row at all. Polled every 15 min it would write ninety-six "nothing to measure" rows a day over the one worth reading — a runner that had to be started from the pool. The ledger read is ~150 ms on 8 MB, so it leaves the event loop; fails open |
| a wedged check cannot hold the queue | `_arm_watchdog`, `CHECK_WATCHDOG_S` (75 min) | every long step has its own timeout except `git` and the snapshot, and a runner stuck in one would hold `regression.lock` for ever — nothing measured, nothing said. SIGALRM records a skip naming the commit (so it cannot wedge the queue more than twice), SIGTERMs the runner's own process group (what it was wedged IN — only when the runner leads that group, as `spawn_detached` makes it) and exits 3. A dead runner is recoverable by construction: the kernel drops the flock and ends the pin, `sweep_stale_scratch` takes the rest, the next landing starts another |
| the runner's log says each verdict as it lands | `run_pending` | the JSON rows `main` prints arrive when the whole queue is done — seventeen checks and two hours, the first time it ran in production |

Scorecard row 10 carries `regression_coverage` — promotions, how many were
measured, how many could not be evaluated — because a detector that is not
running reads exactly like one that finds nothing.
`tests/test_regression_runner.py`.

---

#### 8.1b djev in the loop: replay per request, two floors (2026-09-21)

The pinned corpus made the check deterministic, and #1336 undid that without
anyone noticing: djev began ranking every recall at 20:07Z, and djev does not
repeat itself (`architecture/djev.md` §8.2, label logprobs 1-3 nats apart on
identical requests). The first check after it found −0.009 NDCG between two
arms of identical retrieval code, and its confirm run came back clean. The next
two checks each rolled back a promotion that touched no retrieval code:
a802b979 (#763, counterfactual scoring), for doc_hit 0.529 → 0.517, one query
of 87; and dbec85aa (#939, the scheduler), measured while djev was being
restarted for an experiment, with recall latency 2.0 s against 0.63 s. Two
rollbacks in six hours halted promotions. The floor they were judged on was
measured on 2026-09-17: 20 queries, the cross-encoder, `stdev 0`. Every check
since had flagged it `noise_floor_stale`, and it was used anyway. That flag is now
a gate and the confirm arm screens magnitude against a published paired σ rather
than sign — both are §8.1, and the σ the confirm arm reads is the artifact's
**fresh-ranker** bucket (`metrics_fresh_ranker`), because `automod-check-confirm`
draws djev independently instead of replaying the anchor's answers.

- **Every arm runs under one djev replay file** (`app.djev.replay_env`,
  `LLOYD_DJEV_REPLAY`), anchored on the baseline. A rank request the baseline
  asked is answered with the baseline's answer, so identical code compares
  identically again. A request only the current arm asks (the change moved
  djev's input) is a fresh draw, and counted as one. The confirm arm is its own
  arm, so what the change moved is drawn again rather than replayed from the
  first look.
- **Two floors, chosen per check** (`ranker_reading`, `_floor_for`). `replayed`
  (no fresh draw) is judged on `metrics`, the pinned floor. `fresh` or
  `unknown` (an arm from before replay leaves no record) is judged on
  `metrics_fresh_ranker`, djev's own spread. `measure_noise` records both: trial
  0 plus replayed trials, and trial 0 plus fresh trials. The ledger row carries
  `ranker {arms, reading, floor}`.
- **An arm djev did not answer is a non-measurement** (`REPLAY_FAILURES`:
  unreachable, 5xx, malformed), in either arm, exactly as an arm the pinned
  daemon did not answer is. Those recalls fell back to the cross-encoder, so
  the arm measured a different ranker. A 4xx is a schema the client built
  wrong, which a change can do, so it is scored.
- **The recall's pool was not deterministic either, and replay is what showed
  it.** The first replayed noise trial still drew 19 of 162 djev requests
  fresh on identical code. All of them came from the grep leg
  (`_grep_lloyd_code`): rg prints matching files as its threads finish, and the
  leg kept the first 8, so 7 of 81 gold queries admitted different files run to
  run. It sorts before the cut now. That was live recall's behaviour too, not
  just the eval's (`tests/test_recall_first_stage.py`).
- **Re-measure the floors when the eval changes:**
  `python -m scripts.automod.regression_runner noise` takes `regression.lock`
  and writes `eval-noise.json`. **Anything that restarts or loads djev or the
  qmd daemon holds that lock too**; the second rollback was the author's own
  canvas experiment running under a check.
- `tests/test_automod_regression.py` (the check) and `tests/test_djev_replay.py`
  (the client) pin it.

#### 8.1c The holdout leg: a gain only the dev questions see (#1412, 2026-09-24)

Every retrieval change is selected against the same 86 dev questions (the
nightly, the trend audit, this check), so after enough rounds those questions
are training signal: pairing and floors bound noise, not overfitting. A second,
disjoint tranche of 27 hand-authored questions
(`eval/vault_recall_holdout_queries.yaml`) is scored by this check only.

- **Same pin, same trees, after the dev arms.** `_run_holdout_leg` runs a
  baseline and a current arm over the holdout file inside the dev arms' pinned
  corpus and replay file (arms `baseline_holdout` / `current_holdout`, labels
  that share no substring with the dev ones, because `_load_run` globs).
  ~2 × 27 recalls, ~20 s per arm on today's stack.
- **The record.** `regression_check` carries `delta_dev`, `delta_holdout` and
  `transfer_gap` per armed metric, each paired against the promotion's own
  parent, and `overfit_suspected`. A metric that gained on dev past the dev
  floor while losing on holdout past the holdout floor (`effective_floor` at
  each leg's own n; at n=27 the resolution term is ~0.038, one question) sets
  the flag, and the summary reads `OVERFIT SUSPECTED …` instead of `no
  regression`. **The flag never enters `reasons`**: the rollback decision is
  byte-for-byte what it was (pinned), and every rollback this check ever made
  was a false positive, so a new detector reports before it acts.
- **Reserved, not just split.** `eval/retrieval_holdout.py` writes the reserved
  ids and a `split_hash` to `eval/vault_recall_holdout_manifest.json`;
  verification recomputes it and compares the file's ids and content, and a
  refused manifest means the leg does not run. `run_eval.py` refuses the
  holdout corpus without `LLOYD_EVAL_HOLDOUT_LEG=1` (an env var, because the
  baseline arm runs the parent's `run_eval.py`) and under a `nightly*` label.
  Everything the leg records is a count or an aggregate — a failed holdout
  arm logs its return code, not its output — and the corpus guards report the
  holdout file by count. `tests/test_retrieval_holdout_split.py` pins that no
  other tracked file names the file or carries a reserved id.
- **Latency is exempt by construction.** `latency_over_budget` reads the dev
  arm only; the holdout arm's average is recorded report-only on the holdout
  block, so the paired-check ceiling needs no re-derivation for it.
- **Not in the nightly, on purpose.** The nightly writes per-query rows into
  `~/lloyd-data/eval/baselines/`, which its skill and the trend audit read; a
  holdout row there is the leak the leg exists to prevent. The transfer gap is
  a property of a *change*, and the nightly measures a state.
- **Its first reading** (`eval/measurements/retrieval-holdout-2026-09-24.md`):
  the holdout tranche is easier on the document leg than dev (doc hit 0.963
  vs 0.651), so it has little headroom to show a doc gain and plenty to show a
  loss — which is the direction the predicate reads. Kill switch:
  `LLOYD_AUTOMOD_HOLDOUT_LEG=0`.

## 9. Incident: the false-positive rollback, 2026-09-06

Worth recording in full, because it is the failure mode this design is most
prone to and no unit test could have found it.

```
11:29:18  autoresearch round starts 77 bench trials (7 variants × 11 tasks)
11:30:39  liveness failure: health probe failed 3 consecutive times
11:30:44  stop lloyd-mc:lloyd-backend: error: timed out
11:31:05  preserved: tag=guardian-broken-20260906_113039 stash=guardian-rollback-…
11:31:10  Rolled back a6c0ebae → fc253ffe
```

The backend was never down. `/health` is served by the same asyncio loop as
the agent's own background work, and an autoresearch round — scheduled
hourly — pushes 77 trials through it. Three missed 2-second probes and a
perfectly good promotion was reverted. Left unfixed it would have recurred
about once an hour, which is worse than having no watchdog at all.

**Two fixes came out of it.** Probes now classify *why* they failed, with
separate budgets for refused (3 ticks) and timeout (24 ticks), and the probe
timeout went 2s → 10s — that is how long a *loaded* event loop may take to
answer, not how long a healthy one does (0.37ms). Separately, the "stop:
error: timed out" line exposed a 5s XML-RPC client timeout against a 15s
`stopwaitsecs`, so a working stop reported failure.

**What went right.** The rollback machinery was flawless: it tagged the
commit, stashed the dirty tree by name, restored, restarted in order, and
verified the running commit changed. Both `a6c0eba` and a set of in-flight
nightly changes were recovered afterwards from exactly the artifacts it left
behind. The bug was entirely in the trigger, and the evidence preservation is
what made it a recoverable inconvenience rather than lost work.

**The general lesson:** a watchdog's false-positive rate matters more than its
detection rate. A missed detection costs one bad build; a false positive that
fires hourly costs trust in the whole mechanism, and a mechanism nobody trusts
gets disabled.

## 9.1 Incident: the second false positive, same day, 20:14

```
20:14:21  promoted 5cc8618a (backlog #396, directed graph traversal)
20:14:25  error_rate: novel signature x9: failed domain-research/research:
          ConnectError: All connection attempts failed
20:14:25  rolled back 5cc8618a → 9a0a1d84
```

Four seconds after landing. Three separate defects lined up, and each is now
fixed and pinned:

- **The log cursor only advanced during an observation window**, so the first
  tick of a new window read *everything since the last one*. The nine
  `ConnectError` lines were from 11:47–11:56 that morning, eight hours stale,
  produced by the earlier incident. Fixed by draining on every tick.
- **The rollback used the LKG pointer rather than the promotion's own
  recorded target.** LKG had sat at 14:24 all day because it advances only
  when a promotion *settles*, and the day's promotions had not. So reverting a
  promotion whose parent was six hours newer took the whole evening with it:
  **26 commits of human work**. `current.json` had recorded the correct target
  and nothing read it.
- **The errors were not even the promotion's fault.** They were worker jobs
  killed by the landing's own restart — the idle gate could not see them
  because workers never enter a session queue. The promotion was reverted for
  damage its own landing caused.

Every rollback this loop has ever performed has been a false positive. That is
the number to keep in view: the mechanism's failure mode is not missing a bad
build, it is inventing one.

---

## 10. What Lloyd may change

`scripts/automod/spec.py`. **Denied beats protected beats allowed**, and a run
spec cannot widen its own permissions by naming a denied path in
`writable_paths`.

A reverted change is denied **by content as well as by SHA**. The SHA alone
was never going to catch anything: a round that is re-cut produces a new SHA
for identical content and walks straight past the denylist. The stored tree
hash is `git ls-tree` over the paths the promotion touched, so the same change
re-derived under any SHA is caught. A hash git cannot produce is treated as
*not denied* — refusing to promote because git hiccuped is a worse failure
than the one this prevents.

- **denied** — `config.yaml` and `data/tool_overrides.yaml` (a round could
  disable Bash and Edit and lock itself out with no Python change at all — a
  soft brick no test would catch), `.env*`, `pytest.ini`, `.gitignore`, and
  under `web/` the build inputs — `package.json`, the lockfile,
  `node_modules`, `dist`, `vite.config.*`, `tsconfig*.json` — because the
  frontend rung builds the candidate against the live tree's install, which
  is only the candidate's dependency set if a round cannot change it. Also
  `.git/**` and `.venvs/**`: the object store and the interpreter a rollback
  has to still work with afterwards.
- **protected** — the gate itself, the guardian, the supervisor confs, and the
  health/restart path the rollback depends on. Allowed, but only with a
  passing drill.
- **allowed** — ordinary code.

The two files the `venv` rung installs from — `requirements.txt` and
`requirements.lock` — are allowed *only because* that rung exists: a btrfs
reflink clone of the venv plus a `uv pip install` of the delta, booted by the
canary. `--reflink=always`, not `auto` — auto degrades to a real 6GB copy
silently. `requirements-dev.txt` is allowed for the opposite reason (#1073): the
rung cannot see it. `spec.touches_requirements` keys on exactly those two names,
and when the rung does run it installs `requirements.lock` if it exists else
`requirements.txt`, so a package listed only in the dev file reaches neither a
candidate venv nor the lock — and that is the divergence #1073 settles: a solver
in `requirements.txt` alone is in the live venv and missing from every rebuilt
candidate, and the round that notices is the round sitting on a 1000-passed
floor. `SETUP.md`, where #1073 says the decision gets written down, joins
`README.md` and `CLAUDE.md` as an allowed root file. The decision is written
there now (#1378, landed by hand): `SETUP.md` Part 4's `lloyd` venv section names
`requirements-dev.txt` and its install command.

Why the refreeze had to change with it: the lock is a `pip freeze` snapshot, so
a solver installed out of band would be swept into the rebuild dependency set by
whoever refroze next. The regeneration command is written in two places —
`SETUP.md` Part 4 and `requirements.lock`'s own header comment — and
`requirements.txt` lines 4-5 send a reader to that header. Both copies now pass
a `--exclude` for every package `requirements-dev.txt` names, and the unfiltered
form appears nowhere (`tests/test_automod_doc_claims.py` pins both).

---

## 11. State

Everything under `~/.local/state/lloyd-automod/` — outside the repo, so
`git reset --hard` and `git clean -fdx` cannot reach it. (`_pipeline/` was
ruled out when it was gitignored but still inside the tree; it has lived in
`~/lloyd-data` since 2026-09-22, [[data-home]].)

| File | Written by | Purpose |
|---|---|---|
| `last_known_good.json` | guardian only | rollback target + floor + eval baseline |
| `current.json` | promoter only | the promotion under observation; a land-train flush's carries `commits` (§3.2i) |
| `pending_restart.json` | promoter only, under the lock | the land train: merged landings not yet running, oldest first; the guardian never reads it |
| `flush.running` | `round flush` / `flush_detached` | a flush in flight (pid); holds new rounds and the pool's round hold like a land marker |
| `last_settled.json` | guardian only | what last settled, and its parent — `current.json` is deleted at settle, so this is the only way to ask afterwards |
| `rollback_request.json` | anyone | a rollback asked of the guardian by a process that cannot perform one itself |
| `eval_last.json` | quality worker | the measurement the guardian folds into LKG at settle. Guardian reads it, never writes it |
| `promotions.jsonl` | both | append-only audit trail, fsynced, **raises** |
| `lock` | all | one round / promotion / rollback at a time |
| `pause` | promoter | maintenance lease, capped in the pinned policy |
| `denied.json` | guardian | anti-ping-pong |
| `broken/<ts>/` | guardian | preserved evidence |
| `BROKEN` | guardian | terminal state; services left stopped |
| `rounds/<id>/` | round, gate | `run_spec.yaml`, `gate.json`, the `gate.running` marker (§4.5b), and the detached `gate.log` / `land.log` |
| `clusters.json` | `backlog-cluster` | last night's clustering (§3.2c); `autotriage`'s group mode reads it, and its mtime is what "nightly" means |
| `cluster_judgments.jsonl` | cluster pass | the pair-judge's verdicts, cached by body hash so a re-run costs no LLM calls |
| `dedupe.jsonl` | `backlog_write_task` | every write-time merge decision, for tuning the threshold from data rather than from the three points measured when it landed |
| `review_backfill.jsonl` | `review_tools backfill` | the grader's verdict beside the author's. **Its own file on purpose**: a backfilled verdict measures the grader, and `implement_outcomes` must never read it as a verdict on the round |
| `scorecard.jsonl` | `round scorecard --record` | the trend, so a row survives the terminal it was printed in |
| `eval-noise.json` | regression worker | the measured noise floor (§8.1) |
| `arch_review.json` | `arch-review` worker | per-unit review cursor: verdict, `reviewed_commit`, attempts, the ids each unit filed (`workers/sources/arch_review.py:147`) |

The ledger deliberately does **not** reuse
`scripts.autoresearch.common.ledger_append`, whose contract is "best-effort,
never raises". Defensible for a research ledger; wrong for the audit record of
what code is running in production, where a silently dropped line means you
cannot reconstruct what landed. A test asserts the two behave differently so
nobody refactors them together.

---

## 12. Operating it

```bash
python -m scripts.automod.round status              # state + ledger + guardian
python -m scripts.automod.round bless               # record HEAD as last-known-good
python -m scripts.automod.round recover             # clear BROKEN/halted, start the stack
python -m scripts.automod.round restart --reason "…"  # pause, drain, restart mcp+backend under the lease
python -m scripts.automod.round restart --only agent-llm-primary --reason "…"  # the engine too, since 2026-09-15
python -m scripts.automod.round scorecard --since 7d  # is the loop earning its keep? (--record appends)
python -m scripts.automod.round cluster --write      # regroup the open board by hand (§3.2c; flags pass through)
python -m scripts.automod.review_tools calibrate    # the review grader against known verdicts (§4.5)
python -m scripts.automod.round flush [--now]       # the land train: restart once for what is pending (§3.2i)
python -m scripts.automod.rehearse --yes-i-mean-it  # prove rollback still works
python -m scripts.automod.rehearse --yes-i-mean-it --batch  # + the batch routes; before defer_restart flips
systemctl --user status lloyd-guardian
/usr/bin/python3 agent-services/guardian/guardian.py --selftest
journalctl --user -u lloyd-guardian -f
```

`bless` is how an LKG pointer comes to exist at all, and how it is put right
after a manual recovery — both were being done by hand with a note pasted into
the ledger, and a stranded pointer is what turned one false positive into 26
lost commits. It verifies against the **running** `/health.commit`, not the
working tree: `git rev-parse` proves the filesystem, only `/health` proves the
service.

**Turning the land train on** (§3.2i): land it with `defer_restart: false`,
restage the guardian (`systemctl --user restart lloyd-guardian`; staging runs
the candidate's selftest first), confirm `round status` →
`pending_restart.train.why` no longer names the staged guardian, run
`rehearse --yes-i-mean-it --batch` green, then set
`automod.landing.defer_restart: true` (read per landing, no restart needed).
To turn it off, set it false and `round flush` whatever is still pending;
`bless` refuses until nothing restart-needed waits.

`recover` is the other half of escalation. The guardian deliberately leaves
services stopped when it writes `BROKEN` — an honestly-dead system beats a
half-reverted one — and the documented recovery was "clear the flag", which
leaves the box down.

`status` also reports `unit_drift`: systemd reads `~/.config/systemd/user`, so
a unit edited in the repo and never installed is a change that looks landed
and does nothing.

`restart --only agent-llm-primary` (2026-09-15) gives the engine the same
lease, pool pause and drain as the two Lloyd programs, then a leg of its
own in `promote._restart_primary`: stop, wait for the process to reach a
stopped state, wait for `MemAvailable` to pass `PRIMARY_RAM_FLOOR_GIB`
(180) for up to ten minutes, refuse under `PRIMARY_RAM_ABORT_GIB` (150) and
leave the engine stopped, `supervisorctl reread` + `update` so an edited
`environment=` in `agent-llm-primary.conf` is read, start, and a
20-minute health wait that refreshes the lease. Two lessons from its first
use, both pinned in `tests/test_automod_promote.py`: vLLM's `/health` is a
bare 200 with no body, and `_get` read that as no answer, so the leg sat on
a serving engine for its whole budget; and the first floor (150/120) was
the old desktop's number — a 16 GiB qemu VM had joined the desktop,
`MemAvailable` read 57 GiB with the engine up, and the boot's own transient
got the unit oomd-killed at 23:52:46Z ([[infrastructure]] has the kill).
The engine booted from the new conf on autorestart, at 844,969 FP8 tokens
for `KV_CACHE_MEMORY_BYTES` 14.0 GiB, and everything else on the box
bounced once.

`scorecard` (`scripts/automod/scorecard.py`) is the loop's report card, read
off the ledger, the backlog's front matter and a week of `git log` — fourteen
rows (row 12, `arch review`, landed 2026-09-11; row 13, board net flow,
09-13; row 14, `autocode duty cycle`, 09-15: how much of the window had an
implement turn in flight and what each idle gap was waiting on —
`promotion_gap` (a gap that contains a promotion; the landing's own waits
lead the row, and the restart is not measured), abort, restart — for Alan's rule that a round runs 100% of the time; 88%
over the 24 h before the row existed, 97 min of it across six landings),
each null rather than 0% when
it has no denominator: acceptance hit
rate, the audit delta between the author's and the grader's `met` clauses,
review refusals (and how many were fixed in turn, re-offered, escalated),
spawn ratio per source (with merges, appended findings, expiries and the
open self-spawned count against its bound; `seam_only_refusals`), landed
rounds whose added lines a later non-`lloyd` commit removed within seven
days (row 5, redefined 2026-09-24: "shared a file with a human commit" read
88% in a week whose commits were 92% a person's), test-honesty findings, bookkeeping defects
(nameless deferrals, stranded landings, bare aborts), the regex-fallback
rate, throughput (gate time per round beside the median full gate, and the
red-tree passes and items), rollbacks, and grouping — clusters formed, duplicates
closed, folds, umbrellas landed and the members each one closed (§3.2c). The dashboard's `automod` section shows the 7-day row; `--record`
appends it to `scorecard.jsonl` in the state dir so the trend survives. Its
baseline, from the loop's first 4.3 days: 32% acceptance met, 7 items closed
against 148 filed (triage 4.1:1), 41% regex fallback, 3 rollbacks, all three
false positives.

Recovering from a rollback: the reverted commit is on the
`guardian-broken-<ts>` tag and your uncommitted work is in the matching named
stash. `git merge --ff-only guardian-broken-<ts>` restores it (clear the SHA
from `denied.json` first).

`automod.enabled` in config.yaml is the master switch and defaults to
**false**.

**Before enabling:** merge the outstanding PRs, run the drill, and confirm
`round status` shows a last-known-good that matches the running commit.

---

## 13. Known limits

- **A change that is correct, boots, and is quietly worse** in ways no eval
  measures. The guardian catches crashes, error spikes and data loss; taste is
  not automatable.
- **Post-landing detection fails open by construction.** The code is live and
  has already executed tool calls while you measure. Quality belongs in the
  gate, where it can be slow and fail closed.
- **Frontend changes are gated by the build, not by a probe.** `web/src/**`,
  `web/index.html` and `web/public/**` are allowed since 2026-09-07; rung
  `frontend` runs `tsc --noEmit` as a delta (the tree carried three
  pre-existing errors that day; an absolute bar would have been switched off
  within the hour) and a full `vite build`, both from the worktree against the
  live `node_modules`. After a landing that touched `web/`, the promoter
  checks the Vite dev server still answers. There is deliberately no guardian
  probe of :5173: a broken `src` change is a browser-side error the dev
  server serves with a 200, so a probe would measure the liveness of a
  process the change cannot kill and nothing the change can break.
- **`eval/baselines/`, the Thunderbird bridge, and `.env` are gitignored**, so
  they are absent from every worktree and clone. The bridge contributes ~40 of
  ~124 tools, which is why tool-count assertions exclude
  external-application modules — a flat floor made the suite pass only in the
  live checkout, and therefore made the gate unable to pass its own test rung.
- **The regression detector's noise floor is five runs on one machine on one
  day.** If it is noisier in practice it will fire spuriously.
  `test_the_recorded_noise_floor_is_what_the_code_expects` fails loudly if a
  re-measurement disagrees, but this is the piece to trust least. Re-measured
  2026-09-17 (five pinned trials, stdev 0.0000 on every armed metric); what
  the floor cannot cover is the daemon dropping an answer, which §8.1 handles
  as a non-measurement.
- **The data-damage counter counts notes, not `.git`.** Both copies of
  `count_vault_files` (promoter baseline, guardian live count) walked
  `.git/**` until 2026-09-17, and a repack of ~400 loose objects during the
  observation window read as "vault files dropped 6.7% (6075 → 5667)" with
  the note count unchanged — twice (2026-09-09, #537; 2026-09-17, #1206), the
  second tripping the two-rollbacks-in-6h halt. They now share
  `vaultwatch.measure`; `tests/test_guardian_vault_count.py` replays the
  repack.
- **`/health` shares an event loop with agent work.** §9's fix widens the
  budget; it does not remove the coupling. A genuinely wedged event loop and a
  very busy one still look alike for two minutes.
- **Worker failure rates are not a detector.** There was a CUSUM
  implementation and four policy constants for one, wired to nothing; they are
  gone rather than left looking operable. If it comes back it needs its own
  false-positive budget, measured.
- **The master switch is an interlock, not a sandbox.** `automod.enabled`
  now gates the CLI as well as the MCP tools, and worker jobs and subagents
  cannot call `automod_*` at all. None of that constrains anything holding the
  Bash tool, which can edit `config.yaml` directly. It stops accidents and
  stops a prompt-injected worker; it is not a boundary against a misaligned
  agent with a shell.
- **The quality check is one measurement per promotion**, on a corpus that
  drifts between promotions. It answers "did this commit make retrieval
  worse", not "is retrieval good".
- **Graph EDGE quality is not checked by anything.** Expiring 70% of the
  active edge set moved no metric the eval produces, armed or reported. The
  three fact-layer metrics read the fact tree and fact index, and the armed
  document metrics read the pinned document corpus, not the edges. Covering
  edges needs a metric that traverses them, and there isn't one.
- **Agent-LOOP behaviour is not covered after landing either**, and the check
  that used to be named *behavioural* is the reason the gap was invisible
  (§8.1). No armed metric observes a tool call, a turn count or a model
  decision — `eval/run_eval.py` issues no model request — so a change to tool
  choice, turn accounting, compaction or skill injection passes the nightly
  check by construction, and that green report is what the guardian folds into
  `last_known_good.json`'s `eval` slot as the promotion's quality baseline. Not
  "nothing observes the loop": the gate's `prompt_surface` rung scores tool
  choice, but only *pre*-landing and only when the diff names one of five paths
  (`prompt_builder.py`, `prefetch.py`, `SOUL.md`, `MEMORY.md`, `USER.md`), so a
  tool-set or compaction change has no loop-side check at either end. Arming a
  post-landing loop-side axis needs a measured noise band first — a loop-side
  measurement carries variance the retrieval eval contributes none of, so
  `MIN_SIGMA`-style floors would roll it back on itself. The `eval` slot now
  states its own coverage in an `axis` field (`axis.measures`,
  `axis.does_not_measure`) and `eval_for_recorded_commit` says whether the
  number even belongs to the commit sitting beside it.
- **The pinned corpus is a snapshot of a moving vault.** Both arms see the
  same documents, which is what makes them comparable, but two comparisons run
  a week apart are not comparable to each other. The check answers "did this
  commit make retrieval worse", never "is retrieval better than last month".
- **`POST /api/backlog/task-create` is still unauthenticated** on the tailnet
  when no client fingerprint is forwarded. The guardian files rollback tasks
  through it over loopback; `/api/automod/drain` was restricted because it can
  silence every user turn for ten minutes, this one can only create a note.

---

## Review log

- **2026-09-12 — `current`.** Checked every backticked path, symbol, config key,
  rung list, constant and cadence in this file against the tree at `e534e1c`,
  plus the live `/health`, `/api/workers/health` and `/api/autonomy/health`
  routes and six recent `gate.json` reports. The mechanism described is what
  runs. Corrected in place: the whole-tree pyflakes baseline (69 → 215
  measured, ~3× drift), the rung timings table, `messages._clamp_max_turns`
  (no such function — the clamp is `messages._turn_budget`), two stale rung
  *ordinals* (`drill` and `venv`, both mis-numbered since `frontend` and
  `review` joined the ladder), the §6 xfail claim (autoresearch's snapshot copy
  was fixed 2026-09-08), the scorecard row count (eleven → twelve; row 12 is
  `arch review`), and the §11 state table, which never listed
  `arch_review.json`. Filed: #919 (`gate.py`'s own module docstring still
  lists eight rungs and pre-`frontend` timings), #823 (autocode's
  `max_turns: 150` is unreachable under `agent.max_turns_ceiling: 120`).
