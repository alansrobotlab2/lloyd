---
segment: architecture
tags: [architecture, lloyd, automod]
type: reference
status: implemented
date: 2026-09-06
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
 propose ──► WORKTREE ──► GATE (8 rungs) ──► PROMOTER ──► live tree
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
python -m scripts.automod.round gate  SM_<id>           # 8 rungs, ~2.5 min
python -m scripts.automod.round land  SM_<id>           # idle-gated, verified
```

Or the same four steps as MCP tools: `automod_start`, `automod_gate`,
`automod_land`, `automod_status` (plus `automod_abort` and
`automod_rollback`). Every mutating tool refuses while `automod.enabled` is
false, which is the default.

### 3.2 Unattended

Two worker sources, off by default, chained so that implementation can never
start from an unverified premise:

- **`autotriage`** triages one open item per run, oldest first, and
  records a verdict with evidence. `stale` and `already_done` close the item;
  `confirmed` records an acceptance check and stops. It never opens a round.
  Items this loop filed itself are held out of its candidate pool while they
  are fresh — see "The pass may not eat what it files" below.
- **`autocode`** takes the oldest still-open `confirmed` item that
  has an acceptance check, and runs one round on it through the normal gate.
  It refuses while the loop is anything but free — disabled, halted, BROKEN,
  a promotion under observation, a rollback pending, a round open — and checks
  that twice, once before queueing and again before spending the turn. **One
  attempt per item.** The attempt is recorded before the turn starts, so a
  crash cannot put the item back on the pile; a second try is a human's call.

Both run **in a real session through `POST /api/message/stream`**, not through
`run_query` directly. That is the only turn path that attaches the Inner Voice
observer, and it is what leaves a transcript in the Inner Voice tab — the same
way the three hand-driven rounds ran. `run_prompt_in_session` in
`workers/sources/_common.py` is that path; `run_prompt_on_primary` is the
session-less one, and it must not be used for anything that judges or changes
this code.

Three things were measured to be in the way before this was safe to leave
alone, each now pinned by a test:

| | was | measured need | now |
|---|---|---|---|
| iteration budget | 30 | 45, 65, 76 on the three hand-driven triages | 90, per-request via `max_turns`, ceiling 120 |
| item body | 6,000 chars | next item 12,279; largest 21,300 | 30,000, cut from the end |
| observer | none | — | attached, transcript kept |

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
  `spawned-by-triage` or `spawned-by-autocode` out of the pool until it is
  `SPAWN_TRIAGE_MIN_AGE_DAYS` (30) old. Quarantine rather than exclusion: an
  item nobody implements really can go stale, and then the question triage
  asks is real again. The gate keys on those tags and **not** on `draft`,
  which is the status of most of a stale backlog — a rule that skipped drafts
  would switch the pass off rather than bound it.

  An exhausted queue therefore has two meanings. `backlog.triage_pool` returns
  the held count beside the candidates so the skip summary can say which one:
  "every open backlog item has been triaged" was true, and misleading, on a
  board of 122 where 106 were this loop's own drafts. `SPAWN_CAP` (3, both
  sources) bounds fan-out per run, with the remainder folded into a single
  "Further findings from…" item rather than dropped — #229's lesson still
  holds, and the answer to too many findings is one more item, not fewer
  findings. It is **recorded, not enforced**: the items are on disk before
  `SPAWNED:` is parsed, so unfiling them would destroy real work. Both event
  types carry `spawn_cap` and `spawned_over_cap`.
  `tests/test_backlog_spawn_loop.py` pins it, including the counterfactual —
  with the window set to zero the same run grows the queue.
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
the sha; `automod_vault_revert` is a plain `git revert`, also recorded. Both
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
3. **`reap_abandoned_rounds`**, the backstop, run from the implementer's
   scheduler tick: a round this source opened, still open, nothing under
   observation, its session idle, twenty minutes after the turn ended, is
   aborted with its branch kept and the item told where the work is. Not at
   turn end — that would have raced the rescue and thrown away 875 lines
   the gate then passed.

**Human-only.** Some paths the loop may never touch remain: `config.yaml`,
`data/**`, `.env*`, `pytest.ini`, `.gitignore`, and the frontend's build
inputs. A triage whose fix needs one records `confirmed` with an acceptance
that begins `human-only:`, and `select_confirmed` skips it — the alternative
was an implement round spent discovering it, which is what #278 cost before
`web/src` was allowed.

### 3.2a Status is the state machine

Two jobs, one board, and until 2026-09-09 neither job wrote to it. The
ledger decided everything (`backlog_triage` verdicts, `backlog_implement`
phases) and `status` decided only open-versus-done — so the board could not
tell you where anything was, and nine landed items sat as `up_next`.

```
draft ──autotriage: confirmed──▶ up_next ──autocode: round opens──▶ in_progress
  │                                 ▲                                        │
  └──autotriage: already_done/stale─┼──── attempt ended without a verdict ◀──┤
                          done ◀────┘◀────────── landed & met, or unnecessary ┘
```

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

### 3.2b The item is closed when the round says so, once the landing settles

The loop wrote three records about a landing — the promotion, the guardian's
`settled`, the turn's `finished` — and read none of them back to the item.
Nine settled landings, nine open items. `implemented_ids` kept them from
being re-picked, so the failure was invisible from inside the loop and
visible only as an open count that never went down.

An implement turn now ends the way a triage turn does: one more completion
under a grammar (`IMPLEMENT_OUTCOME_SCHEMA`) restating the result as
`{landed, acceptance: met|not_met|deferred, deferred_to, summary, spawned}`,
recorded on the `finished` event as `outcome`, with `outcome_error` beside it
so a finalizer that quietly stopped working does not look like one that is
working. `close_settled_items` runs on every implement poll: for each
settled promotion (or vault landing) whose item is still open and not yet
marked, it writes `automod_landed: <sha>` and an activity line, and sets
`status: done` **only when the round said `met`**.

Only the round judged the acceptance, and it is asked in a grammar rather
than read out of prose. Everything else is noted and left open: `deferred`
names the ids it waits on, `not_met` says so, and a round from before the
finalizer says "a human decides". A closed item is never re-triaged, which is
the whole reason not to guess — and the reason the prompt tells the model
that `met` on an unverified acceptance is the one claim the loop cannot
recover from.

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

Eight rungs, cheapest first, short-circuiting. **Every rung fails closed** —
`_rung` catches exceptions and records them as failures, because with no human
review tier a rung that errors and reads as "didn't fail" silently removes a
check.

The gate always runs **from the live tree** against the worktree, so a
candidate that weakens the gate is judged by the old gate.

| Rung | Typical | Catches |
|---|---|---|
| preflight | ~0s | dirty tree, moved base, merge commits, out-of-scope paths |
| static | ~2s | syntax errors, **import failures**, new pyflakes findings |
| tests | ~30-40s | the full suite, plus floors on collected AND passed |
| venv | 3s–5m | only when `requirements*` changed |
| canary_boot | ~2-30s | a build that will not start |
| canary_smoke | ~5-15s | a build that starts but cannot dispatch a tool |
| drill | ~18s | only when the diff touches the rollback path |

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

### 4.1 pyflakes is a diff, not a bar

The tree carries 69 pre-existing findings. An absolute rule would be switched
off within a day, so the rung compares changed files against the same files at
the merge base and fails only on *new* findings.

### 4.2 The collected-count floor is not enough on its own

`pytest -q` exits 0 if a round deletes the test that was failing. Under
auto-landing that is not hypothetical, so the rung asserts a minimum
collected count and refuses a diff that removes files under `tests/`.

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
a throwaway worktree, and classifies:

- every failure reproduces at base → `external_blocker: true`
- any failure is new → the round's own, and the detail names which

The rung still **fails** either way. A red tree is not a tree to land onto:
the promotion's observation window would open against a broken baseline, and
§8 is built on the assumption that errors after a landing are about the
landing. Blocking was always right; spending the item was not.
`backlog.implemented_ids` reads the flag off the ledger event — not off
`gate.json`, which lives in the round dir and is deleted with the worktree —
and does not count such a round as the attempt.

Three things that decide whether it works:

- **It probes files, not node ids.** Handed a node id that does not exist at
  base — a test the round just wrote — pytest exits `ERROR: not found:` and
  runs *nothing*, so one new test would hide every pre-existing failure beside
  it and a red tree would read as green. Found while building this; pinned by
  `test_a_test_the_round_added_does_not_hide_the_pre_existing_ones`.
- **It fails closed in every direction.** A worktree that will not create, a
  probe that times out, an unparseable summary — all return "nothing
  reproduces", which blames the round. Being wrong that way costs the status
  quo; being wrong the other way lands a change nobody checked.
- **The exemption is capped** (`EXTERNAL_RETRY_CAP`, 3). `select_confirmed`
  takes the oldest ready item, so an item re-offered without bound would be
  re-picked every round for as long as the tree stayed red, starving
  everything behind it. A tree red across four rounds is an incident nobody is
  handling, not a blip worth retrying.

Only the `tests` rung grants it. `preflight` failing on a dirty live tree is
also not the round's fault, but it is cheap and re-runnable, and widening an
exemption is how it becomes an open door.

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
  100 iterations unspent.
- **`infra`** — the turn never reported completion. `run_prompt_in_session`
  returns `stop_reason=None` when the stream closes without a `done` frame.
  #392 was recorded as its item's one attempt *one second* after starting, on
  a session holding a single user message, while the guardian was alerting
  that supervisord was unreachable. New runs record `infra_failed`; the read
  path also recognises the old shape, so history heals without a backfill —
  but only an **explicit null**, never a missing key, so a writer whose shape
  we do not know falls through to `spent`.
- **`external`** — §4.2b, now including preflight's two live-tree refusals.
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

### 4.3 Rungs that run candidate code run it against scratch state

Only the canary redirected `LLOYD_AUTOMOD_STATE`. The static, tests and venv
rungs executed candidate code against the **live** state directory, so a
candidate test that forgot its isolation fixture could write a real `BROKEN`
or `promotions-halted` flag, or append to the production audit trail — from
inside the gate whose entire contract is read-only judgment. `Gate._child_env`
points all three at a per-round scratch dir and sets `LLOYD_VOICE_ALERTS=0`.

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
   nothing else is under observation.** `land` passed the base along but the
   candidate HEAD was re-read from the worktree, so a commit made after a
   passing gate landed completely ungated. And a second landing simply
   overwrote `current.json`: the first promotion never settled, never advanced
   the LKG, and the new record's rollback target became a commit that had
   never survived a window — quietly breaking the one invariant the
   promoter/guardian split exists to provide.
1. **Write the rollback point and read it back.** Nothing is mutated until
   `current.json` round-trips. This is the strict analogue of the xfail at
   `tests/test_autoresearch_promotion.py:362`, where
   `snapshot_current_prompts` mkdirs unconditionally, never verifies the copy,
   and promotes anyway — which is why 26 of 83 historical promotions have no
   rollback point.
2. Idle-gate on `/health.turns` — three consecutive quiet polls, resetting on
   any activity — then set a **TTL'd drain flag** so a turn arriving between
   the last poll and the restart gets a 503 instead of being cancelled
   mid-flight. The TTL is mandatory: a promoter that dies here cannot wedge
   the endpoint, and the flag is in-memory so the restart clears it anyway.
   "Idle" counts `harness_runs` as well as the session queues — worker jobs
   call `run_query` directly and never enter a queue, so a ten-minute research
   job was invisible to the gate that exists to avoid killing it.
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

Lloyd may edit the guardian. A diff touching it triggers gate rung 6: the
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

### 7.3 Three invariants

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
   from `git merge-base --is-ancestor` is the tell.

### 7.4 Rollback order

**Reset when HEAD is still the promotion; revert in place when it is not.**
`reset --hard` to the promotion's parent is only correct while HEAD *is* the
promotion. Nightly jobs commit straight to live `main`, so a 15-minute window
can legitimately close over work the loop never touched, and resetting past it
destroys commits nobody asked the guardian to judge. That is the 26-commit
incident one level down: there the wrong *target* was chosen, here the right
target is reached by the wrong *route*. When HEAD has moved on, the guardian
reverts exactly the promoted commit and leaves the rest standing; a conflict
has no safe automatic answer, so it escalates instead of guessing.

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

`git clean` is **path-scoped**, never repo-root: the root holds `usage.db`,
`workers.db`, `.env` and `.venvs/`, all gitignored and none replaceable.

If rollback fails twice, services are left **stopped** and `BROKEN` is
written. With no human in the loop, an honestly-dead system is safer than an
autonomous agent running half-reverted code.

---

## 8. Detectors

| Trigger | Window | Source |
|---|---|---|
| crash / won't boot | ~2 min | supervisord state + `/health` |
| error-rate spike | ~15 min | `logs/server.err` by byte offset |
| data damage | ~15 min | KG row count, vault file count |
| behavioural regression | nightly | paired `eval/run_eval.py` |

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

### 8.1 Behavioural regression: measured, not assumed

The autoresearch composite is **not** used. Three identical baseline runs
scored 0.719 / 0.542 / 0.624 — spread 0.177 against a 0.05 threshold, and 61
of 83 historical promotions were decided inside that noise.

`eval/run_eval.py` has no LLM in it. Measured here, five consecutive runs
against an unchanged vault produced **identical** values for every quality
metric (stdev 0.0000 for entity_hit_rate, entity_recall_avg,
fact_entity_recall_avg, ndcg10, mrr_doc, doc_hit_rate, doc_recall_avg). Only
`latency_ms_avg` moved, at 562ms stdev, and it is never compared.

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
runs move 0.0000, so all seven are armed again. The pin is a precondition:
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

**Only three metrics were armed for a while, and that number came from being wrong.** The
original seven were chosen because five consecutive runs gave stdev 0.0000 for
all of them — a real measurement of the wrong thing. It describes repeatability
inside one short window; the paired A/B runs its arms *minutes* apart, and it
cancels drift only in what `LLOYD_FACTS_ROOT` and `LLOYD_KG_DB` redirect. The
document leg comes from neither: it queries the qmd daemon at an absolute
`http://localhost:8181/query`, over a vault being written continuously by
nightly jobs and session capture.

The first real run proved it. On a promotion whose entire diff was text inside
an inject string — incapable of touching retrieval — it reported `ndcg10`
0.5680 → 0.5620 and `mrr_doc` 0.4740 → 0.4680, both "beyond 3σ", and asked for
a rollback. Three back-to-back runs of identical code and data then gave
0.0000 spread on every entity metric and **0.0250 on `doc_recall_avg`**, eight
times its own tolerance. The doc side is not stable across the window this
check spans; the fact side is. The doc-side four are now reported and never
fire.

**And the armed three are named for what they actually read.** Two degradation
drills against copies of the live store:

| Degradation | Armed metrics | Doc metrics |
|---|---|---|
| 70% of `facts_idx` rows deleted (205,689 → 61,707) | `entity_hit_rate` 0.60 → 0.55, `entity_recall_avg` 0.443 → 0.433 — **fires** | unchanged |
| 70% of **active edges** expired (4,029 → 1,209) | unchanged — **blind** | unchanged |

So they are `FACT_LAYER_METRICS`, not "graph sensitive". Edge quality has no
armed metric at all: a change that expires most of the edge set walks past
this check in silence, and that is a stated limit rather than a covered case.
Naming them for the graph would have been the same overclaim the detector
exists to prevent.

This is also why `eval/run_eval.py` now records **which corpus it scored**
(`corpus_ok`, the resolved paths, entity and edge counts) and refuses an empty
one unless `--allow-empty-corpus` is passed. An arm that scored an empty
corpus is not a low score, it is a measurement that did not happen, and the
worker treats it as "cannot evaluate".

The measurement is handed to the guardian in `eval_last.json` and folded into
the LKG record's long-empty `eval` slot at settle. Read, never written, by the
guardian: `last_known_good.json` has exactly one writer, and that is what
makes it mean *observed healthy in production*.

---

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
  is only the candidate's dependency set if a round cannot change it.
- **protected** — the gate itself, the guardian, the supervisor confs, and the
  health/restart path the rollback depends on. Allowed, but only with a
  passing drill.
- **allowed** — ordinary code.

`requirements*` is allowed *only because* rung 3 exists: a btrfs reflink clone
of the venv plus a `uv pip install` of the delta, booted by the canary.
`--reflink=always`, not `auto` — auto degrades to a real 6GB copy silently.

---

## 11. State

Everything under `~/.local/state/lloyd-automod/` — outside the repo, so
`git reset --hard` and `git clean -fdx` cannot reach it. `_pipeline/` would
not do: gitignored but still inside the tree.

| File | Written by | Purpose |
|---|---|---|
| `last_known_good.json` | guardian only | rollback target + floor + eval baseline |
| `current.json` | promoter only | the promotion under observation |
| `last_settled.json` | guardian only | what last settled, and its parent — `current.json` is deleted at settle, so this is the only way to ask afterwards |
| `rollback_request.json` | anyone | a rollback asked of the guardian by a process that cannot perform one itself |
| `eval_last.json` | quality worker | the measurement the guardian folds into LKG at settle. Guardian reads it, never writes it |
| `promotions.jsonl` | both | append-only audit trail, fsynced, **raises** |
| `lock` | all | one round / promotion / rollback at a time |
| `pause` | promoter | maintenance lease, capped in the pinned policy |
| `denied.json` | guardian | anti-ping-pong |
| `broken/<ts>/` | guardian | preserved evidence |
| `BROKEN` | guardian | terminal state; services left stopped |

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
python -m scripts.automod.rehearse --yes-i-mean-it  # prove rollback still works
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

`recover` is the other half of escalation. The guardian deliberately leaves
services stopped when it writes `BROKEN` — an honestly-dead system beats a
half-reverted one — and the documented recovery was "clear the flag", which
leaves the box down.

`status` also reports `unit_drift`: systemd reads `~/.config/systemd/user`, so
a unit edited in the repo and never installed is a change that looks landed
and does nothing.

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
  re-measurement disagrees, but this is the piece to trust least.
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
  armed three read the fact tree and fact index. Covering edges needs a metric
  that traverses them, and there isn't one.
- **The pinned corpus is a snapshot of a moving vault.** Both arms see the
  same documents, which is what makes them comparable, but two comparisons run
  a week apart are not comparable to each other. The check answers "did this
  commit make retrieval worse", never "is retrieval better than last month".
- **`POST /api/backlog/task-create` is still unauthenticated** on the tailnet
  when no client fingerprint is forwarded. The guardian files rollback tasks
  through it over loopback; `/api/automod/drain` was restricted because it can
  silence every user turn for ten minutes, this one can only create a note.
