# Lloyd — Claude Code Context

## Project Overview

Lloyd is a fully local AI agent. It runs its own in-process agent loop (`app/harness/`) against a local vLLM server and exposes all tools through a unified MCP aggregator (`agent_mcp/`). The backend is FastAPI + SSE; the frontend is React (Vite).

- **Backend**: `server.py` (FastAPI, port 8080)
- **Frontend**: `web/` (Vite dev server, proxied through backend)
- **Config**: `config.yaml`
- **MCP aggregator**: `agent_mcp/main.py` (unified `Server("lloyd")` on `:8500/mcp`, Streamable HTTP)
- **Agent harness**: `app/harness/` (`run_query(messages, options)` — async generator)
- **Venv**: `.venvs/lloyd/bin/python`

## Setup

Rebuilding on a fresh OS: **[SETUP.md](SETUP.md)** is the authority — system
packages, the uv/bun/npm-global toolchain, all four venvs, supervisord + the
systemd unit, and what must be backed up first (several runtime assets are
untracked and not re-downloadable). `agent-services/setup/setup-all.sh --check`
reports what's missing without changing anything.

Secrets live in `.env` (gitignored) and reach `config.yaml` through `${VAR}`
placeholders that `app/config.py` expands at boot. Never put a literal secret in
`config.yaml` — it is tracked.

## Service Management

Lloyd runs **directly on the host** under supervisord (installed as the `agent-supervisord.service` systemd `--user` unit; supervisord itself is the uv tool at `~/.local/bin/supervisord`). There is no longer any distrobox container in the loop.

**To restart the backend and/or the aggregator, use the round CLI, not
supervisorctl:**

```bash
.venvs/lloyd/bin/python -m scripts.automod.round restart --reason "picked up gate.py"
.venvs/lloyd/bin/python -m scripts.automod.round restart --only lloyd-backend
```

It does what the promoter does for its own restarts and what this file used
to describe as five manual steps: takes the guardian's pause lease, pauses
the worker pool, drains the backend and waits for it to go idle, restarts
`lloyd-mcp` then `lloyd-backend` with a health wait per leg, and releases
all three. A bare `supervisorctl restart` is indistinguishable from a crash
to the guardian — on 2026-09-09 four deliberate restarts each fired "Service
down, but no promotion to revert" through every alert channel, toast and
voice included — and it kills whatever worker job is mid-flight, whose
connection errors then land in someone's observation window. It refuses
while a promotion is under observation (`--force` overrides) and records a
`restart` event on the ledger with the reason.

supervisorctl is still the tool for everything else (the frontend, the
engines, status), and is what the command above wraps:

```bash
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf restart lloyd-mc:lloyd-backend
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf restart lloyd-mc:lloyd-frontend
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf status
```

The process group is `lloyd-mc`, not `lloyd-backend` bare. Always use `lloyd-mc:lloyd-backend` and `lloyd-mc:lloyd-frontend`.

After editing `server.py`, `round restart --only lloyd-backend` for changes to take effect.  
After editing frontend files, Vite HMR usually picks up changes automatically (no restart needed).

**Never restart `agent-llm-primary` twice in quick succession.** `supervisorctl
stop` returns when the processes are signalled, not when the kernel has
reclaimed their memory, and that engine holds a **95.37 GiB** BF16 n-gram table
in *host* RAM. Start the next boot before the old one's pages are freed and two
of those coexist on a 251 GiB box. On 2026-09-08 an A/B sweep did exactly that
twice and `systemd-oomd` killed the **whole `agent-supervisord.service` unit** —
953 processes the first time, 793 the second, i.e. every service on the machine,
not merely the engine being restarted. Peak RSS was 230.3 GiB. Everything came
back on its own, but the arm under test was lost and the failure looked like the
config being tested rather than the restart cadence.

Wait for `MemAvailable` to come back above ~150 GiB between boots;
`agent-services/bin/flash-next-run-arm.sh` does this and refuses to start below
120 GiB. Note the blast radius is the *unit*, so the guardian (a separate
systemd unit, deliberately) survives it.

## Automod (self-modification)

Lloyd can change his own code through a gated loop with automatic rollback.
`automod` is the whole of it — the worktree/gate/promoter/guardian machinery,
its state dir, MCP tools, API routes and config block. Two worker sources run
inside it: `autotriage` judges draft backlog items and `autocode` implements
the confirmed ones. (Named `selfmod`, then `autoimplement`, until 2026-09-09.)
`architecture/automod.md` is the long version. The master switch is
`automod.enabled` in config.yaml and it defaults to **false**.

```bash
python -m scripts.automod.round status              # state + ledger + guardian
python -m scripts.automod.round start "goal"        # cuts a worktree
python -m scripts.automod.round gate  SM_<id>       # 8 rungs, ~2.5 min
python -m scripts.automod.round land  SM_<id>       # idle-gated, verified
python -m scripts.automod.round bless               # HEAD becomes last-known-good
python -m scripts.automod.round recover             # clear BROKEN, restart the stack
python -m scripts.automod.rehearse --yes-i-mean-it  # prove rollback still works
```

**Nothing inside the blast radius performs a rollback, and a landing runs
detached.** A rollback stops the backend and the aggregator, so code doing one
from inside either process issues the stop that kills its own caller and never
reaches `git reset` — the stack goes down and the tree does not move. They
write `rollback_request.json` and the guardian performs it. Likewise the
promoter restarts `lloyd-mcp`, which is where the MCP tool calling it lives,
so `automod_land` spawns it in a new session (`state.spawn_detached`) — a
process group signal cannot reach that. Before this, the CLI path worked only
because the Bash tool already spawns that way, so the loop worked when a human
drove it and would have failed the first time Lloyd did.

Things worth knowing before touching any of it:

- **The guardian is a systemd unit, not a supervisord program.**
  `agent-supervisord.service` sets `KillMode=control-group`, so a
  supervisord-managed watchdog dies exactly when it is needed. It is stdlib
  only, runs on `/usr/bin/python3`, and never imports from `app/` — it must not
  share a failure domain with what it guards.
- **It runs from a pinned snapshot** at `~/.local/state/lloyd-guardian/bin/`,
  staged only if the candidate compiles and passes its own selftest. Editing
  `agent-services/guardian/` does nothing until a staged copy proves itself, so
  a broken guardian degrades to a *stale* watchdog, never to none.
- **State lives at `~/.local/state/lloyd-automod/`**, outside the repo, because
  the guardian must read its rollback target while the repo is being rewritten.
- **`HEAD == last-known-good` never rolls back.** Everything broken with
  nothing promoted is infrastructure, not a bad change.
- **A rollback reverts in place when HEAD has moved past the promotion.**
  `reset --hard` to the parent is right only while HEAD *is* the promotion.
  Nightly jobs commit straight to live `main`, so a 15-minute window can close
  over work the loop never touched, and resetting past it destroys commits
  nobody asked the guardian to judge. The route is chosen per rollback; a
  conflicting revert escalates rather than guessing.
- **A rollback restores the promotion's own `rollback_target`, not the LKG.**
  `gstate.rollback_target(current)` prefers what `current.json` recorded at
  landing time, then its `parent`, and only then the LKG pointer. The LKG is
  a blunt fallback because it advances *only when a promotion settles*: two
  rollbacks in a row strand it wherever it last settled while HEAD keeps
  moving with ordinary human commits. On 2026-09-06 that turned one false
  positive into **26 discarded commits** — LKG had sat at 14:24 all day, so
  reverting a promotion whose parent was six hours newer took the whole
  evening with it. The promoter had written the correct target and nothing
  read it. The failure is self-reinforcing, which is what makes it worth a
  rule: every rollback that does not settle makes the next one wider.
- **The log cursor advances on every tick, not only while observing.**
  `Guardian.drain_logs()` runs at the top of `tick()`, above every early
  return, and `evaluate_errors` reads the buffer it fills. Reading used to
  live inside `evaluate_errors`, which only runs during an observation
  window — so between rounds the cursor stood still and the first tick of a
  new window read *everything since the last one*. That is what fired on
  2026-09-06: a healthy promotion reverted four seconds after landing, on
  nine `ConnectError` lines from 11:47–11:56 that morning. Errors are still
  *judged* only inside a window; what changed is that the tape always moves.
  The paused path additionally **discards** its buffer rather than skipping
  it, because the promoter holds that pause across its own supervisord
  restart and the window for that very deploy opens seconds later.

Errors are read from `logs/server.err`, never `server.log` — `basicConfig`
writes to stderr, so `server.log` is uvicorn's access log and holds zero
error-shaped lines.

- **A failing `tests` rung says the tree is red, not that the round made it
  red.** On 2026-09-08 three rounds aborted here on the same three failures
  no diff under test had written — two asserting the wall clock against a
  23→07 quiet-hours window (so they failed exactly when the unattended loop
  runs), one asserting that a deliberately untracked file had been checked
  out (so it failed in every worktree). `8138f1c` fixed them at 16:39 UTC,
  after all three, and nothing went back: `implemented_ids` counts any
  finished round as the item's one attempt, so #361, #370 and #376 were
  consumed by someone else's breakage. Each of those rounds had *proved* it
  predated them — in prose, in a report read once. `rung_tests` now re-runs
  the failing **files** at the round's base in a throwaway worktree and
  records `external_blocker: true` when every failure reproduces there. The
  rung still fails, because landing onto a red tree opens the guardian's
  observation window against a broken baseline; what changes is that the
  item keeps its attempt. Probing by *file* is load-bearing — handed a node
  id the round just added, pytest exits `ERROR: not found:` and runs
  nothing, so one new test would hide every pre-existing failure beside it.
  Fails closed everywhere (a probe that cannot run blames the round), capped
  at `EXTERNAL_RETRY_CAP` re-offers so a permanently red tree cannot starve
  the board, and granted by the `tests` rung only.
- **A round that never reached a verdict has not spent the item.**
  `implemented_ids` counted any finished round as the one attempt "whatever it
  did", and six of the loop's first seventeen attempts were spent by something
  that was never a judgment on the change: three on pre-existing test failures,
  #446 on the wall clock (fourteen seconds and one `automod_gate` call short of
  landing 757 lines), #447 on `preflight: live tree is dirty` from an unrelated
  uncommitted edit in production, #392 on a turn that never ran at all — the
  backend was down and it was recorded as an attempt one second after starting.
  `backlog.implement_outcomes` classifies each: `spent` closes the item, and
  `incomplete` / `infra` / `external` / `rolled_back` re-offer it, each capped
  because `select_confirmed` takes the *oldest* ready item and an uncapped
  re-offer starves the board. **A promotion is a verdict however the turn
  ended** and is checked first — #278 died at `max_turns` and the observer
  landed it anyway. A `rolled_back` re-offer is the one that changes a safety
  property: nothing joined the promotion (which carries the round id) to the
  rollback (which carries only the commit), and every rollback this loop has
  performed has been a false positive. The reason rides into the next round's
  prompt via `reoffer_reason`, because a re-offer is not a fresh start — the
  branch usually still holds the work.
- **A worker turn can see its wall clock now.** `app.deadline_anchor` is the
  single definition (autonomy delegates to it), and `/api/message/stream`
  accepts `deadline_seconds`, which `run_prompt_in_session` sets to the same
  timeout it enforces. Only a caller that enforces a clock sends one — a chat
  turn has none and must never be told it does.
- **The tree is shared, so the gate rebases and retests rather than
  refusing.** A human works on `main` while rounds are open. `preflight`
  used to refuse a moved HEAD ("abort and re-cut") and any uncommitted edit
  anywhere in production; `promote` refused both again at landing. Now
  preflight rebases the worktree onto live HEAD and the rest of the ladder
  judges the change *on top of what landed* — the only build that was ever
  going to be live. Only a conflict fails, naming the files, with the rebase
  aborted. The promoter chases the same way, twice (before the idle wait and
  inside the drain after it) by re-running the gate with the old base; a
  third miss is a `land_failed`, which is now a ledger verdict the backlog
  reads so the item keeps its attempt. A rebase moves the base in three
  places and all three must follow: `gate.json` (what `land` reads),
  `run_spec.yaml` (what the next `run_gate` reads — stale, it sweeps the
  human's commits into the round's diff), and the promotion record.
  Uncommitted live edits are tolerated when disjoint from the round's diff
  (`merge --ff-only` never touches them) and refused by name when they
  overlap. The promoter's inline rollback is `reset --hard` after writing
  dirt to `broken/<stamp>/dirty.patch`; it used to run for pre-merge failures
  too, which with tolerated dirt would have stashed a human's edits out from
  under their editor for a tree that had not moved. A `merged` flag gates it.
- **A turn that dies at its budget is not the end of the round.** The
  budget anchor (`<budget>` at 75%/90% of `max_turns`) tells the model to
  gate-and-land or abort while it still can; the observer's ambient
  follow-up is the first responder when it did not (that is what landed
  #278); `autocode.reap_abandoned_rounds` is the backstop, twenty
  minutes later, branch kept. Never abort a round at turn end.
- **The idle gate drains first, then waits.** `wait_idle` used to arm the
  drain only *after* three quiet polls — the one moment it is no longer
  needed. Against a worker pool that starts a research job every few minutes
  that is a lottery, and the first landing of the unattended era
  (SM_20260907_233449) lost it: 900 s watching `harness_runs` flicker, never
  drained. Now the drain is armed before the first poll, refreshed inside its
  TTL, and released on give-up.
- **The idle gate counts `harness_runs`, not just session queues.** Worker
  jobs call `run_query` directly and never enter a queue, so a ten-minute
  research job was invisible to the gate that exists to avoid killing it. The
  landing then restarted the backend underneath it, and the `ConnectError`
  lines it logged on the way down landed inside the observation window — so
  the promotion was reverted for damage its own landing caused. That is the
  2026-09-06 20:14 rollback exactly.
- **An answered non-200 is not a refused connection.** Three probe classes,
  three budgets: refused 3 ticks, timeout 24, http-error 36. The aggregator's
  503 (any degraded module, e.g. a closed Thunderbird) is excluded from the
  down predicate entirely and judged by `mcp_degraded_is_fatal` instead.
- **Every rollback this loop has performed has been a false positive.** The
  failure mode to design against is inventing a bad build, not missing one.

### Alerts: one fan-out, six channels

`agent-services/guardian/notify.py` is the **only** producer of user-facing
alerts. That is a recent property, not an accident: `lloyd-guardian-nag.service`
used to run its own inline `notify-send`, a second private definition of "tell
the human" that was structurally incapable of gaining any channel this module
grew. It now calls `nag.py`, which goes through the same `Notifier`. Anything
that wants to announce something goes through here.

Two entry points, and picking the wrong one is the trap:

- **`alert()`** — an incident. Fans out to all six: ledger, ALERT.md, journal,
  desktop toast, voice, vault note, plus a backlog task when critical.
- **`announce()`** — news, with no bookkeeping. Journal, toast and voice only.
  Used by a successful promotion and by the 15-minute nag. The nag is the
  reason `announce` takes a `level`: the state really is critical and should
  look it, but it has **already** been recorded, and re-announcing through
  `alert` would append a ledger row and file a fresh backlog task every 15
  minutes, burying the task the rollback filed under copies of itself.

Promotion *success* is announced too. Before, every notify-send in the tree
hung off a guardian alert, so a loop that rewrites the running system in the
background was silent whenever it worked — backwards, since the successful
landings are the ones nobody is watching a terminal for.

### The spoken channel

`speak.py` says the alert aloud in the same cloned voice as voice mode. It
lives in `agent-services/guardian/` because `guardian-stage.sh` stages
`guardian/*.py` and nothing else — a module the guardian imports must be in
that directory or it will not exist in the pinned snapshot.

- **It reports *dispatched*, not *heard*.** The unit watchdogs the loop at
  `WatchdogSec=90` against a 5s tick, so synthesis and playback happen in a
  detached child and `alert()` returns in milliseconds. What actually came out
  of the speaker is in `voice.log` in the guardian state dir.
- **The child runs the venv python, and that is deliberate.** The guardian's
  stdlib-only rule exists so the watchdog cannot be taken down by what it
  watches. This child runs *after* all five reliable channels have fired,
  nothing waits on its exit, and its failure cannot reach the loop — so
  spending the venv buys the presence EQ (scipy) at no cost to the property
  the rule protects. A wrecked venv costs a duller voice, never an alert.
- **Shaping degrades in tiers and says which one ran.** EQ needs scipy, the
  WSOLA speed needs only numpy, so a system-python fallback still fixes the
  pace. The tier is logged because the first cut of this module called
  `OutputShaper.enabled()` — it is a `@property` — and shipped *unshaped*
  audio while looking perfectly healthy. A silent downgrade is
  indistinguishable from success.
- **Suppression is on disk**, keyed by alert title, because the two producers
  are different processes: the daemon and the nag oneshot. In-memory dedupe
  (`guardian.py::_alert_seen`, `ALERT_REPEAT_SECONDS` = 900s) cannot see the
  nag. `policy.VOICE_REPEAT_SECONDS` is 3600s — a toast you have already seen
  costs a glance, a sentence you have already heard costs the whole sentence,
  and at 900s an unresolved incident would say the same thing aloud four times
  an hour indefinitely.
- **Quiet hours gate the clock, and only the sound.** `guardian.voice.quiet_hours`
  in config.yaml (23–07 by default) withholds speech; the toast, journal,
  ledger, vault note and backlog task all still fire, so nothing is lost — it
  is waiting in the morning. That is what makes it safe to default on.
  `allow_critical: true` lets a rollback wake you anyway. The window is
  checked **before** `should_speak`, which records as it decides: recording a
  quiet-hours drop would spend the hourly slot on an utterance nobody heard,
  and the 08:00 repeat of an 03:00 alert would then stay silent for the wrong
  reason. A window that wraps midnight is the normal shape, and `start == end`
  means *no* window rather than a full day of silence.
  It lives under `guardian.voice`, **not** `livekit.tts`, and the split is
  load-bearing: `livekit_worker` reads `livekit.tts`, and a voice conversation
  that went mute at 23:00 because an alert policy leaked into it would be a
  real bug. Only alerts are gated by the hour.
- **`LLOYD_VOICE_ALERTS=0`** keeps every other channel and drops only speech.
  `tests/conftest.py` sets it for every test — otherwise `pytest tests/` talks
  to the room from a process that outlives the test.
- **Voice sits below the `external` gate**, like the vault note and the
  backlog task: the drill runs a real guardian against a throwaway repo, and a
  rehearsal that announces a rollback out loud is indistinguishable from a
  production incident to anyone in the room.

`config.yaml`'s `livekit.tts` stays the single source for the voice.
`agent-services/bin/sync-voice-config.py` pushes it into the guardian's state
dir at stage time (the guardian has no yaml, and must not read the repo on a
critical path). If it never runs, `speak.py`'s built-in defaults still sound
right — the sync only stops the two drifting after a voice *change*.

### Unattended: triage, then implement

`autotriage` triages; `autocode` runs one round per `confirmed`
item with an acceptance check, behind every gate the loop enforces. Both are
off by default and both run **in a real session** via `run_prompt_in_session`,
which is the only way a worker turn gets Inner Voice and a transcript — never
`run_prompt_on_primary` for anything that judges or changes this code. Budget
exhaustion records `incomplete`, not a verdict; the item comes back once.

Two things the first unattended run (#229) taught, both pinned by tests:
a worker session is **never the user's session** — worker turns arrive
through the chat path, so "the last session to receive a user turn" was one
nobody reads, and the morning brief was delivered there.
`sessions_io.NON_USER_PLATFORMS` (`autonomy`, `worker`) is the one definition
of that, and `/inject` refuses such a session with 409 so no producer counts
it delivered. And a verdict's `ACCEPTANCE` is a contract only if
`backlog.acceptance_text` says so: the old template's `else: ->` was copied
verbatim, and `strip("-")` left a truthy `>`. And **a finding that lives only
in EVIDENCE is lost**: #229 said two claims "belong in two new items" and filed
none, in a turn told to write nothing. Both prompts now require filing whatever
the in-focus item does not cover, via `backlog_write_task`, and report it under
`SPAWNED:`; ids are verified on disk before the ledger links them.

**But the pass may not eat what it files.** That filing requirement met
`select_candidate`, which takes the oldest untriaged *open* item, and
`OPEN_STATUSES` includes `draft` — the status `backlog_write_task` writes. So
every item triage filed re-entered the queue it came out of. Over the loop's
first 48 hours: 40 triage runs closed 28 items and filed 78, a reproduction
number of **1.95**, or +46 open items a day at the then-cadence of one run per
30 minutes. The open board went **19 → 122**, and 110 of the 122 were the
loop's own output. R > 1 is the entire bug: the queue doubles rather than
drains, however good the verdicts are. Oldest-first ordering hid it, because
self-filed items sort to the back and the pass reads as healthy right up to
the moment the real backlog runs out — which was 6 items away when this
landed. `backlog.is_quarantined` holds a self-filed item (tagged
`spawned-by-triage` or `spawned-by-autocode`) out of the candidate pool until
it is `SPAWN_TRIAGE_MIN_AGE_DAYS` old. Quarantine, not exclusion: an item
nobody implements really can go stale, and then the question is real again.
The gate keys on those tags and **not** on `draft`, which is the status of
most of a stale backlog — a rule that skipped drafts would switch the pass
off rather than bound it.

An exhausted queue therefore has two meanings, and `triage_pool` returns the
held count so the skip summary can say which one it is. "Every open backlog
item has been triaged" was true, and misleading, on a board of 122 where 106
were this loop's own drafts.

**Status is the pipeline's state machine, and the ledger is its source of
truth.** Until 2026-09-09 the loop wrote `status` in exactly two places,
both `done`; a `confirmed` verdict left an item wherever it was, no round
ever set `in_progress`, and #353 landed while still `draft`. Now
`autotriage` reads **`draft`** only and its verdict moves the item —
`confirmed` → `up_next`, `already_done`/`stale` → `done`, anything else stays
`draft` (triaged, not for the loop). `autocode` reads **`up_next`** only,
sets `in_progress` the moment its turn starts, and the item ends `done`
(landed and settled with the acceptance `met`, or the round's outcome said
`unnecessary`) or back in `up_next` (external, incomplete, infra, rolled
back, reopened) — or back to `draft`, tagged `needs-human`, when its one
attempt is spent, because `up_next` means implement will take it and it will
not until a human reopens it; the tag comes off when a reopen moves it back. `backlog.desired_statuses` is the one table; `reconcile_statuses` runs
it on every implement poll and after every turn, so a human moving an item by
hand is honoured until the ledger next says otherwise, and untriaged items
parked in `up_next` — where nothing can pull them — go back to `draft`. The
first run was the migration. Kill switch: `workers.sources.autocode.status_pipeline`.

**But a status outside that vocabulary is invisible to the loop and open on
the board.** Five lists name it — the loop, the `backlog_*` tools, the
Mission Control writer, `dashboard._BACKLOG_CLOSED`, and `STATUSES` in the
React board — and all five agreed on the four words, so drift between them
was never the bug. None had a case for a value *outside* them, and the two
halves then disagreed in the worst direction: the dashboard counted such an
item as open work while `OPEN_STATUSES` could not see it at all. Nothing
could move it either — `set_status` and `reconcile_statuses` both reach
items through `open_items`, which filters *on status*, so the pass that
exists to correct a status is structurally unable to reach the one that is
wrong in this particular way. #287 (`review`) and #304 (`closed`) sat there
from April 2026 until 2026-09-09, stranded when the vocabulary was narrowed
to four words and nothing migrated what was already on disk.
`app/backlog_status.py` is the one definition now (stdlib-only in `app/`,
like `backlog_tags`, so the automod CLI does not pull `mcp` and
`httpx` in behind it), and `backlog.rescue_off_vocabulary` runs at the top
of every reconcile — before the main pass, so a rescued item is judged in
that same pass — walking by **board** rather than by status, since an Alfie
item with an unusual status is not this loop's to rewrite. The mapping is
deliberately lopsided: only words already known to be terminal (`closed`,
`cancelled`, `wontfix` — the legacy half of `_BACKLOG_CLOSED`, which is the
tree's only record that they ever meant anything) reach `done`, and
everything else becomes `draft`. Calling a word terminal when it is not
buries a live item where nothing will look again; calling it live when it is
not costs one triage run that closes it. `is_off_vocabulary` is
case-sensitive — `status: Done` strands exactly as `review` does, since no
reader lowercases — while an *absent* status is not off-vocabulary at all,
because every reader already defaults it to `draft`.

**A landed item is closed when its round said the acceptance was met — and
only then.** Nine promotions settled in the loop's first three days and not
one item was closed: `promote` wrote the commit, the guardian wrote
`settled`, `execute` wrote `finished`, and nothing joined the three back to
the item's status, so the loop's own finished work sat on the board as
`up_next` and `draft`. An implement turn now ends with a structured finalizer
(`IMPLEMENT_OUTCOME_SCHEMA`, mirroring triage's): `landed`, and `acceptance`
as `met` / `not_met` / `deferred` with the ids it waits on. `backlog.close_settled_items`
runs beside the reaper in the implement source's poll, joins
`settled → promoted → finished` (a vault round lands on its own `vault_land`,
with no window), writes the landing onto the item as `automod_landed: <sha>`
and an activity line, and closes it **only for `met`**. `deferred` and
`not_met` are noted and left open; a round with no outcome (everything
before this) is noted and left for a human. The asymmetry is deliberate: a
closed item is never re-triaged, so the prompt tells the model that `met` on
an acceptance it did not verify is the one claim the loop cannot recover
from, and `deferred` with an id is the honest answer for a check that needs
traffic or a nightly run (#520 → #618). Kill switches:
`workers.sources.autocode.close_on_settle` and `structured_outcome`
(carried in the queue payload like the budgets).

`SPAWN_CAP` (3, both sources) bounds fan-out per run; overflow goes into one
"Further findings from…" item rather than being dropped, since #229's lesson
still holds. It is **recorded, not enforced** — the items exist on disk before
`SPAWNED:` is parsed, so unfiling them would destroy real findings — and the
ledger carries `spawn_cap`/`spawned_over_cap` on both event types.
`tests/test_backlog_spawn_loop.py` pins all of it, including the
counterfactual: with the window set to zero the same run grows the queue.

The verdict's `SURFACE:` picks the implementer's route. `code` and `frontend`
run a worktree round through the gate — `web/src/**` is in scope since the
`frontend` rung (tsc delta + `vite build`) exists. `vault` runs
`scripts/automod/vault_round.py` (`automod_vault_land`): the vault is a live,
shared tree with no worktree, so the route is validate the named paths (front
matter; the real prompt/skill/task loaders for `skills/**`, `lloyd/**`,
`autonomy/**`), commit exactly those paths on `main`, revert on failure. A
`confirmed` whose fix needs a path the loop may never touch begins its
acceptance with `human-only:` and is skipped, not attempted.
`architecture/automod.md` §3.2.

### Development happens in ~/lloyd-sandbox

`/home/alansrobotlab/lloyd` is production: a saved file is a deploy. Non-trivial
work belongs in the `~/lloyd-sandbox` clone (remotes: `origin` = GitHub,
`live` = the production tree), pushed as a PR. The autonomous loop is the
exception — it cuts worktrees from live `main` and lands offline, because a PR
step in an auto-landing loop is either ceremony or a contradiction.


## Architecture

```
~/lloyd/
├── server.py            # FastAPI backend — all API endpoints + SSE bridge
├── config.yaml          # Model configs, MCP server list, agent settings
├── prompt_builder.py    # System prompt assembly (SOUL.md + memories + skills)
├── autonomy.py          # Task scheduler
├── usage_store.py       # SQLite usage tracking
│
├── app/session_titles.py # few-word session names (secondary model)
├── app/host_metrics.py  # CPU/RAM/disk/GPU for the dashboard
├── app/vllm_metrics.py  # vLLM /metrics scrape + rate derivation
├── app/routers/dashboard.py  # GET /api/dashboard (one aggregated snapshot)
│
├── app/harness/         # In-process agent loop (replaces claude-agent-sdk)
│   ├── __init__.py      # Exports: run_query, RunOptions, HookRegistry
│   ├── options.py       # RunOptions dataclass
│   ├── events.py        # NormalizedEvent TypedDict types
│   ├── client.py        # httpx SSE stream → vLLM /v1/chat/completions
│   ├── loop.py          # Agent loop: stream → tool dispatch → loop
│   ├── hooks.py         # HookRegistry (pre/post tool-use callbacks)
│   ├── mcp_pool.py      # Persistent SSE client to lloyd-mcp aggregator
│   ├── tool_schema.py   # MCP tools → OpenAI tool schema translation
│   └── errors.py        # ParseError, ToolDispatchError, MaxTurnsExceeded
│
├── agent_mcp/           # Unified MCP aggregator (Server("lloyd") on :8500/mcp)
│   ├── main.py          # Aggregates all modules; MCP SSE endpoint
│   ├── builtin_bash.py  # Bash tool (timeout, truncation)
│   ├── builtin_fs.py    # Read, Write, Edit, Grep, Glob tools
│   ├── builtin_task.py  # Task subagent (in-process, recursion cap = 1)
│   └── ...              # Domain modules: ambient, facts, vault, session, etc.
│
├── web/src/
│   ├── api.ts           # All API calls + TypeScript types
│   └── components/pages/
│       └── ToolsPage.tsx
│
├── .venvs/lloyd/        # Python venv (use this python for all lloyd scripts)
└── logs/                # server.log, server.err, frontend.log
```

## Agent Harness

`run_query(messages: list[dict], options: RunOptions) -> AsyncIterator[NormalizedEvent]`

Events yielded by type (constructors in `app/harness/events.py` are the
authority — these keys are not the OpenAI wire names):
- `system` — `{type, session_id, model}` — turn opened
- `text_delta` — `{type, text}` — streaming text chunk
- `thinking_delta` — `{type, text}` — reasoning content chunk
- `thinking_done` — `{type, text, duration_ms}` — reasoning phase
  complete. `duration_ms` spans the first reasoning chunk to the last,
  not the iteration's wall clock, which also covers prefill and the
  answer written afterwards. It reaches the chat's collapsed thinking
  panel as `reasoning_ms` on that phase's own `role="thinking"` row (see
  "The thinking trace"), so the header reads the same on reload as it did
  live — the event lands *after* that iteration's text, so the browser
  opens the row on the first *delta* and measures the timestamps itself
  until the real number arrives.
- `tool_call` — `{type, call_id, name, args_json, args_dict, summary}` — tool
  invocation. `summary` is the model's own one-liner for the transcript;
  it is absent from `args_json`/`args_dict` (see "Tool-call summaries").
- `tool_result` — `{type, call_id, name, content, is_error}` — tool result
- `assistant_message` — `{type, text, tool_calls, thinking, usage,
  duration_ms, iteration, finish_reason}` — one agent-loop iteration.
  `usage` and `duration_ms` are per-iteration, not per-turn.
- `result` — `{type, stop_reason, usage, num_turns, duration_ms,
  response_text}` — turn complete
- `stream_raw` — `{type, raw, error}` — raw SSE line on parse failure

**Mid-turn state (the position-0 rule)**: the system prompt is built once
per turn and inserted at index 0; the loop only ever appends. That keeps the
whole prompt prefix KV-cached across every iteration, so a 160k-token turn
re-prefills nothing. The cost is that anything rendered into the system
prompt — `<active_todos>`, the plan, the goal — is frozen at turn start.
**Never refresh the system prompt mid-turn**; re-anchor by appending instead
(`RunOptions.state_anchor`, mirroring `notification_drain`). A turn that
creates its own todo list would otherwise never see it again — see
`app/routers/messages.py::_build_state_anchor`.

**Preserved thinking**: assistant messages carry their reasoning back into
history under **both** `reasoning` and `reasoning_content`, bounded to
`harness.preserve_thinking_iterations` recent iterations. Qwen3.8-Flash-Next
renders it into each prior turn's `<think>` block; dropping it showed the
model turn after turn in which it had apparently thought nothing. A/B it with
`eval/run_preserve_thinking_eval.py` before changing the window.

The two spellings are not redundant — the engines disagree, and each one
ignores the other's field *silently*:

| Engine | Reads | Ignores |
|---|---|---|
| vLLM 0.28 (primary) | `reasoning` | `reasoning_content` |
| llama.cpp (secondary, Qwen3.6) | `reasoning_content` | `reasoning` |

vLLM accepts both on the wire but only populates the template from
`reasoning` (`entrypoints/chat_utils.py:2000`); Qwen3.6's own
`chat_template.jinja:91` reads `reasoning_content` and never looks at
`reasoning`. Sending one spelling preserves thinking on one engine and
quietly discards it on the other, which is the exact failure this mechanism
exists to prevent. `_prune_reasoning` must drop the pair together or the
token bound stops bounding anything. `tests/test_preserved_thinking.py`
pins both halves; llama.cpp's `POST /apply-template` will show you the
rendered prompt if you need to re-verify.

Scope is **intra-turn only**: history is rebuilt from the session JSON on each
user turn (`load_and_compact_session`), which does not carry per-iteration
reasoning, so the window resets at every turn boundary. That is where the cost
was anyway — the motivating turn ran 52 iterations inside one turn.

**An empty tool pool is the worst failure in the system.** `client.stream_chat`
omits `tools` from the request when the list is falsy, so vLLM never engages
the `qwen3_xml` tool parser. The model still reads its whole toolbox in the
system prompt, reasons its way to "call Bash", and then has no channel to emit
a tool call on. What comes out is an empty message, or the call written as
prose (`{"name":"Bash","input":...}` — an Anthropic shape that appears nowhere
in this repo), or invented tool *output*. Nothing in the stream says "no
tools"; it reads exactly like the model having forgotten how to use them, and
Inner Voice's only lever — injecting more text — cannot help, because the
intent was never missing, the capability was.

`MCPPool.open()` used to log a warning, `continue`, and set `_opened = True`
even when its only server failed discovery. `get_or_open_pool` caches
process-wide and short-circuits on `_opened`, so one transient error (the
aggregator restarting) pinned an empty pool for the life of the backend.
`open()` now raises `ToolDiscoveryError` when discovery yields nothing, which
makes `get_or_open_pool`'s **existing** eviction path fire so the next caller
re-discovers — the recovery already existed, nothing ever failed loudly enough
to trigger it. A *partial* failure still degrades gracefully; that is what the
`continue` is for. `run_query` refuses a turn whose pool advertised nothing.

Two things this cost on 2026-09-06, both invisible as tool failures:
a 30-minute chat where Lloyd narrated `sqlite3` commands instead of running
them, and four `domain-research` jobs killed at the 600s cap — a toolless
research job cannot research, so it spins until the timer. The same job
finished in 17s once tools came back. `tests/test_mcp_pool_discovery_failure.py`
pins it. The tell in the log is `no server claims tool '_BackgroundTaskDrain'`
firing right after `mcp_pool: failed to discover` — the drain shares the pool,
so it is the cheapest early warning that every turn has gone toolless.

**Stream stalls**: `harness.stream_chunk_timeout_seconds` bounds the gap
*between* SSE lines once the engine has started producing, raising
`StreamStalledError`. It deliberately does **not** bound time-to-first-line:
prefill emits no bytes, and the secondary runs llama.cpp with `--parallel 1`,
so a queued request legitimately sits silent for as long as the one ahead of
it. `client.stream_chat` sets httpx `read=None`, so without this a wedged
engine mid-generation hangs the turn until the client gives up. The key
existed from the start and was read by nothing until 2026-09-06.

**Two engines, two reasoning keys**: assistant messages carry reasoning back
as **both** `reasoning` and `reasoning_content`. vLLM populates the template
only from the former; Qwen3.6's own jinja (what llama.cpp applies) reads only
the latter. Sending one breaks preserved thinking on the other engine
silently. See `_assistant_message_for_history`.

**Tool naming**: Built-in tools (Bash, Read, Write, Edit, Grep, Glob, Task) are advertised to vLLM under bare names. This keeps session JSON, SOUL.md deny rules, and Inner Voice `pretooluse_deny` patterns working unchanged.

## Concurrent tool dispatch (read-only batches only)

`harness.parallel_tool_calls.enabled` lets one iteration's tool calls overlap.
A batch qualifies only when **every** call in it is annotated `readOnlyHint`
(or is a parse error, or ToolSearch). One `Bash`, `Edit`, `Write` or mutating
MCP tool makes the whole batch sequential, byte-for-byte the old path.

Read-only-only is not caution, it is the only classification available that is
not a guess. `mcp_pool._list_tools` now carries `annotations` through, so
qualification comes from the server's own hint rather than a second private
list of names — the pattern `agent_mcp/annotations.py` was written to replace.
A server that sets no hints qualifies nothing, which is that file's contract.
`Bash(cat …)` serialises its batch: classifying shell commands as read-only is
the guessing game the safety hook deliberately refuses to play.

Three phases, and each one exists for a reason:

- **Phase 1 runs in wire order and stays sequential.** `_pre_dispatch` covers
  the parse error, the disabled-tool gate, the ToolSearch intercept (which
  mutates the shared `LoadedToolSet`) and the hook deny. Every `tool_call`
  event is yielded here, which is why both frames reach the UI before the
  first result.
- **Phase 2 overlaps only `_execute_tool_call`,** under a
  `Semaphore(max_concurrency)`. No `TaskGroup`: it cancels its siblings on
  the first exception, and one tool failing is a `tool_result`, not a reason
  to abandon the batch. Results are yielded as they land — the frontend and
  `messages.py` key on `call_id` — and a `finally` cancels outstanding tasks
  if the generator is closed mid-batch.
- **Phase 3 writes history in wire order** regardless of who finished first,
  so the replayed conversation matches the assistant message's own
  `tool_calls` array.

Caption bookkeeping also runs in wire order (`_account_captions`): the ratchet
is about the *first miss*, and the first call to come back is arbitrary.

`MCPPool._invoke` takes a per-server lock on the stdio path — one
`ClientSession` over one pair of pipes, and two concurrent `call_tool`s
interleave their JSON-RPC frames. The HTTP path opens a session per call and
is unlocked. The lock map outlives `_reopen`.

Subagents read the same config keys (`builtin_task`), since a Task is the
fan-out case this exists for and constructs its own `RunOptions`.

Ships **off**. Soak checklist before flipping it: `mcp_pool:` warnings,
`[iv.observer] inject` placement in transcripts, and
`harness.empty_terminal_iteration` counts.

## Tools

Every tool lives inside an MCP server — built-ins (Bash/Read/Write/Edit/Grep/Glob/Task) live inside the lloyd-mcp aggregator. Tool enable/disable state:

- Server-level: `mcp_servers.<name>.enabled: false`
- Tool-level: `mcp_servers.<name>.disabled_tools: [tool_name, ...]` (use the bare tool name)

config.yaml holds the hand-edited defaults and is **read-only at boot**; UI toggles (`/api/tool-toggle`, `/api/tool-discovery`) persist to `data/tool_overrides.yaml` (gitignored), which is merged over config.yaml at load (`app/config.py:_merge_tool_overrides`). To change tool state by hand, edit config.yaml and check `data/tool_overrides.yaml` isn't shadowing the same key.

**The override file must stay untracked, and that is the whole reason it
exists.** `save_tool_overrides` replaced dumping the entire CONFIG back over
config.yaml on every toggle, because config.yaml is tracked and a tracked
file rewritten by a UI click leaves the live tree dirty — which
`scripts/automod/gate.py` and `promote.py` both refuse. Until 2026-09-07 the
override file was tracked too, so the escape hatch had the defect it was
built to avoid: one click on the Tools page dirtied the tree and silently
stopped the self-modification loop until someone hand-committed the result
(`4fb1ccd`, `2ef86c7` are that happening). Untracked alone is not enough —
`git status --porcelain` lists new files as well — so it needs the
`.gitignore` rule beside it. `tests/test_tool_overrides.py` pins both halves.

**A test about untracked state may not require that state to be present.**
The end-to-end test asserted `(ROOT / "data/tool_overrides.yaml").exists()`
with `ROOT` resolved from `__file__` — so in an automod worktree it demanded a
file that is, by design, never checked out. `data/` has no tracked contents at
all, so it failed for **every round from `d11ad8c` onward, whatever the diff
under test**, and the `tests` rung is a hard rung: three rounds aborted on it
in fifteen hours while filing fourteen new backlog items and landing nothing.
The loop's only drain was blocked by a test asserting that a deliberately
untracked file had been checked out. It now *writes* the file through the real
writer — `app.paths.LLOYD_HOME` resolves from `__file__` too, so the writer and
the test address the same tree in a worktree — and asserts git never sees it,
which is also the stronger check: the old form could only observe a file
somebody else had already written and reverted. The `.<pid>.tmp` sibling
`atomic_write_text` lands before renaming is now ignored and asserted too; a
write killed in between leaves a stray that dirties the tree exactly as the
tracked file used to, one filename over.

**And a gate rung must not depend on the wall clock.** Two
`tests/test_guardian_speak.py` tests asserted that an alert dispatches voice,
while `speak.dispatch` consults `in_quiet_hours` against the real clock and a
23→07 default window — so they failed whenever the gate ran overnight, which
is when the unattended loop runs. Rounds gated at 23:52 and 04:07 failed;
the same code at 08:13 did not. They pin the policy off explicitly now
(`_AWAKE`), as the tests that are *about* quiet hours already pinned it on.

Because a fresh clone has no override file, **config.yaml is the state a
rebuild boots into**, so it has to keep describing what is actually served.
It claimed `tool_search.enabled: true` for an unknown stretch while the
override served `false`. The merge now warns on a key whose override differs
from the tracked value — the `disabled_tools` half has warned since the
2026-09-04 `browser_screenshot` incident, but `harness.tool_search` was a
bare `.update()`, and `enabled` decides whether the model is handed all 131
tools or a baseline plus ToolSearch. Agreement stays silent: the Tools page
rewrites the whole block on every toggle, so warning on it would fire each
boot and stop meaning anything.

Disabled tools are enforced via `RunOptions.disallowed_tools` as `mcp__<server>__<tool>`. The harness's bare-name aliasing in `tool_schema.py` blocks both the bare and namespaced form at advertise + dispatch time, so disabling `Bash` via `mcp_servers.lloyd-mcp.disabled_tools: [Bash]` blocks the model from calling either `Bash` or `mcp__lloyd-mcp__Bash`.

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
  wrong answer. The baseline persists to `_pipeline/tsc/baseline.json`.
- **A whole-project run becomes a per-session answer** by keeping the
  previous run's per-file counts and reporting `run[f] - baseline[f]` only
  for the files *that session* edited. Somebody else's breakage is somebody
  else's news.
- **The root is derived from the edited path**, not from `LLOYD_HOME`: a
  automod round edits under `~/lloyd-work/…` and the live tree's tsc would
  say nothing about it. No `web/node_modules/.bin/tsc` under that root means
  no run *and no hint* — promising a check that cannot happen is worse than
  silence.
- **Delivery is a `DiagnosticsRecord`, not a `TaskRecord`.** That dataclass
  carries a `process` and a `log_fd`, and `format_notification`, `_task_row`
  and `list_active` are all specific to a background bash child. It rides the
  per-session drain queue and never enters `_records`, so `/state`'s task
  rows are untouched.
- **A `task:*` session's result goes to the parent.** Nothing reads a
  subagent's drain queue once its Task has returned
  (`_subagent_registry.parent_scope`).
- A failed or timed-out run is still reported, for the same reason the hint
  exists.

`/state.tsc` shows the last run and what is pending.

### The per-turn change ledger

`~/lloyd` is production: a saved file is a deploy. The self-modification loop
has a worktree, a gate and an automatic rollback; an ordinary chat turn that
edits three files had none of that, no line saying which three, and no undo.

`agent_mcp/_change_ledger.py` writes one record per (session, turn) under
`sessions/<sid>.changes/<turn_id>/` — an `index.json` naming every file the
turn wrote and a `<sha1(realpath)>.pre` beside it. The layout follows
`tool_result_spill.py` for the same reason: per-turn side data that has to
survive an aggregator restart and be findable from a session id alone.

- **First writer per realpath per turn wins.** The pre-image is what the file
  looked like when *this turn* first touched it, so a turn that edits one file
  ten times reverts to where it came in. `begin` loads the on-disk index on a
  miss, so an aggregator restart mid-turn does not restart the rule.
- **Revert refuses a file that moved since.** If the current sha does not match
  what the turn wrote, something else has written and restoring the pre-image
  would destroy *that* — the exact damage this exists to prevent. Refused by
  name and reported, never silently skipped. Same for a `create` whose file
  was edited afterwards.
- **`turn_id` and `call_id` ride in `_meta`**, like the session id and the
  caption, because `args` is validated against each tool's inputSchema and is
  what the repetition guard hashes. A caller with no `turn_id` gets no
  ledger, which used to mean every background run: since 2026-09-10 both
  background paths mint one, so an unattended run's writes finally leave
  pre-images and are revertable — see "Background runs are recorded". A bare
  `run_query` caller still has none.
- **A subagent's writes land on the parent's turn.** `ledger.scope()` redirects
  a `task:*` session through `_subagent_registry.parent_scope`; the entry
  carries `via_session` so the row can still say it came from a Task. The
  read-before-edit gate keeps using the raw `task:*` session — the subagent
  must still Read what it edits — and only attribution moves.
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
**queue item**) in `workers.db`, written `unknown` **before** dispatch —
the only record that survives a cancel — and settled `ok`/`error` after. A
second identical call in scope replays the stored result; an `unknown` one is
refused with instructions to read the state back; the ledger fails **open**
if the database is unusable. `harness.effect_ledger.enabled` is the switch.
The scope is `item:<source>:<id>` (`workers/pool.py::effect_scope_for`) and
never the grant scope, which is stable across every future run of a task and
would suppress a legitimate second effect forever.

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
  the automod gate is growing a review rung.

`unknown` rows are pruned on a longer clock than settled ones
(`unknown_retention_days`): they are the only record an effect may have
landed, and pruning them re-arms the duplicate. `tests/test_tool_effects.py`
pins all of it, including the probe.

### Read-before-edit and stale-file gates

`Edit` used to be exact-match against whatever is on disk right now, with no
record of whether this session had ever looked at the file. The failure worth
designing against is not the edit that fails — `old_string not found` announces
itself — it is the edit that **succeeds**: the model Reads a file, something
else rewrites it, the `old_string` still matches, and the edit silently reverts
the other writer. Nothing in the transcript, the tool result or the logs says
so.

`builtin_fs._read_records` is `session_id -> realpath -> (mtime_ns, size)`,
written by a successful Read, Write or Edit and consulted by `_gate_check`.

- **The Read's stat is taken *before* the file is opened.** A write landing in
  between then makes the recorded key older than the bytes returned, so a later
  Edit is refused as stale — the safe direction. Stat'ing afterwards would
  record the other writer's key and wave that Edit through.
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

Editing a binary file is now a normal error rather than a `UnicodeDecodeError`
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

- **Never absolute.** The tree carries ~69 tolerated pyflakes findings.
  Reporting them on every edit trains the model to skip the block, which is
  worse than not having one.
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
`isError` by sniffing a leading JSON object with an `"error"` key, so appending
to an error payload would break the JSON and make a lint finding read as a
failed edit. Nothing in this path may raise: `_append_diagnostics` swallows
everything, because an edit with no diagnostics beats an edit that failed
because the linter did.

**The reach of all of that is one file, because pyflakes is.** So a second
block, `<blast_radius>`, carries what a per-file linter structurally cannot:
when an edit changes a module-level interface — a `def`/`class`/assignment
name, a signature, a base, an assigned value, or a `return`/`yield` expression
— the edit result also names the inbound callers living in *other* files
(`symbol → path.py:88`, advisory, never an error). The store is
`agent_mcp/code_graph.py`, the one the `graph_*` tools read; the point is that
this half is **passive**, because the graph was always available and only ever
fired when the model remembered to ask. Three rules keep it worth reading, all
the same shape as the three above: only an *interface* change fires (a local
variable adds nothing), only the **pre**-image's symbols are queried — the
graph describes the tree before the edit, which is the right source for "who
calls this today", so a stale graph is correct here, a name that exists only
afterwards silently does not resolve, and a rename is caught from the old side
— and a symbol above `FANOUT_CEILING` call sites is dropped rather than listed,
because a helper with 62 callers produces a wall of text that gets skipped.
Measured on this tree: `kg_store.store` (62 sites) and `RunOptions` (70) are
both suppressed, which is the rail working, not failing.

`_append_diagnostics` runs on the event loop, not in the edit's worker thread,
so the graph half runs in a daemon thread the edit joins for `RAIL_BUDGET_S`
(90 ms) and then abandons. Abandoning is not waste: the thread finishes the
210 ms load and fills a per-root cache, so the next edit in that tree gets its
answer. `RAIL_MAX_SOURCE_BYTES` is 120 KB because the fingerprint pass costs
~0.28 ms/KB and would break the latency promise on its own above that; pyflakes
still runs on bigger files.

`harness.edit_diagnostics.python: false` removes the pyflakes block and
`.blast_radius: false` the cross-file one — separate switches, separate stores.
Neither key is in `config.yaml` yet, and adding one is human-only.

### Tool-call summaries

Every advertised tool carries one extra string parameter, `summary`: a
short phrase the model writes saying what the call is doing ("Reading
server.py", "Restarting the backend"). The collapsed tool bubble in the
chat and Inner Voice transcripts renders it as **`ToolName`** — summary,
which is the whole point — a wall of `Bash`, `Bash`, `Read`, `Bash` says
nothing about what a 50-iteration turn actually did.

It is display metadata riding in the one channel a tool call has —
its arguments — which makes *where it is removed* the whole design:

- **`tool_schema.add_summary_param`** injects it and returns *which tools
  got it*. That return value is load-bearing: `session_inject_context`
  already has a required top-level `summary` of its own (1 of the 129
  tools advertised today), and popping that one before dispatch would
  delete a real argument. Injection **replaces** the `parameters` object
  rather than mutating it — it arrives as the very `inputSchema` dict
  held in `MCPPool.discovered`, which is process-shared for the life of
  the pool, so an in-place write would make the *next* turn read `summary`
  back as the tool's own parameter, skip injection, and stop stripping.
- **`loop._commit_tool_calls`** lifts the value onto the tool call's
  `_summary` and pops it from `_args_dict` — and *only* from there. The
  two records of a call deliberately disagree: `_args_dict` is what
  reaches MCP, which validates against each tool's real inputSchema (a
  leaked `summary` is a dispatch error, not a spare field), while
  `arguments` is what gets replayed to the engine and is **the only
  record of this call the model will ever see again**.
  **Stripping the caption from `arguments` too is what broke the first
  cut of this**, and it broke it invisibly: session
  `20260907_184351_ivec8d` shows the first call of each tool name
  carrying a summary and every repeat carrying none — 5/5 vs 0/31. The
  schema said `required`; the model's own most recent example of that
  tool said otherwise, and the example won. A few-shot channel you are
  writing into cannot be edited for brevity. It costs ~10 tokens per
  historical tool call to keep, which is the price of the field working
  past its first use.
- **`summary` is injected first** in `properties` and in `required`.
  Property order is the order the schema is shown to the model and
  roughly the order it emits arguments in, so a caption placed after
  Bash's `command` is one written after a 40-line heredoc.
- `messages.py` persists it on the tool call (omitted when empty, so
  every pre-existing session reads the same), puts it on the
  `tool_start` SSE frame so the live bubble has it before the result
  lands, and prefers it over `tool_activity_detail` for the dashboard's
  live activity line.
- The transcript's expanded **Arguments** block hides the key, because
  that block shows what was *dispatched* and the header already shows
  the caption. It is dropped only when the header is rendering it, so a
  tool with a real `summary` parameter of its own still shows it there.

The Inner Voice observer reads the caption in two of its three tool-call
inputs, and the third exclusion is deliberate:

- **`build_assistant_message_summary`** renders
  `Bash — Checking root disk usage` per call instead of
  `['Bash','Bash','Bash']`. The observer's job is judging whether the
  primary is still on the user's request, and a wall of identical names
  is the least informative possible input for that.
  `observer_prompt._tool_call_labels` accepts all three shapes a caption
  arrives in — `_summary` (live harness event), `summary` (rebuilt from
  session JSON), and `summary` inside the raw `arguments` string.
- **`build_pretool_event_summary`** states it before the arguments:
  the caption is what the primary *said* it was doing and the arguments
  are what it actually did, so the two disagreeing is the signal. (This
  path is dormant while `pretool_llm_enabled: false`.)
- **`guards.tool_call_signature` must never see it.** `exact` is the
  full `key=value` rendering for every tool but Bash, so a caption in
  the args makes two byte-identical calls compare as different — and
  rewording is exactly what a looping model does. This is why
  `fire_pre_tool_use` carries the caption as its own `tool_summary` key
  rather than merging it into `tool_input`: `tool_input` is what safety
  matching and the repetition guard read, and it stays clean.
  `tests/test_tool_call_summaries.py` pins all three.

**No tool may ask for the caption twice.** `Bash` used to declare a
`description` argument — "Short human-readable description (informational
only)" — and `Task` a `description`, "Short label for the task
(informational)". Both restated, one key later in the same object, exactly
what the injected `summary` asks for, and a model answers that question
once. On 2026-09-07 it began answering into the wrong half: sessions
`20260907_235236_backlogs_a8fd` and `20260908_000804_backlogi_3828` emitted
`{"command": ..., "description": "check"}` for 49 consecutive Bash calls
with no `summary` on any of them, and the chat rendered 49 bare `Bash`
rows. **Nothing errored**, because `description` was a real Bash argument —
the caption was not dropped, it was filed where only a background task
would read it.

What makes an ambiguous schema expensive here is the ratchet described
above: `arguments` is replayed as history, so the first miss becomes the
model's own most recent example of calling that tool and the session locks
into it. Across the 16 sessions since the feature landed, every one whose
*first* Bash call carried a summary stayed above 95%; both that missed
stayed below 26%. One field decides a whole session, which is why the fix
is to delete the competing field rather than to reword it.

The two tools still need their label — a background-task row and a subagent
row are both read by a human later — so the caption now travels the way the
session id and the calling turn's model already do: in the request's
`_meta`, as `lloyd/summary`, lifted into `_task_registry.current_call_summary`
by `agent_mcp/main.py::call_tool`. It must not be handed back through `args`
instead: that is what the aggregator validates against each tool's real
inputSchema, and it is what the repetition guard hashes.
`tests/test_tool_call_summaries.py` pins that Bash and Task advertise no
second caption field, and that the caption reaches MCP through `_meta` only.

`harness.tool_call_summaries: false` removes the parameter from every
schema; the UI falls back to the bare tool name. Worth reaching for if a
model ever starts spending its tool-call budget on the caption.

### The thinking trace

The harness has always emitted one `thinking_done` per agent-loop
iteration. The router kept almost none of them: a single
`accumulated_thinking` buffer held the current phase, each new phase
**replaced** it (`messages.py`, not `+=`), and it only reached disk on an
iteration that produced both tool calls *and* non-empty text. A tool-only
iteration — the common shape — never flushed, so a forty-iteration turn
persisted exactly one reasoning phase, the last one, and the chat could
only ever show that. On the verification turn for this feature the first
phase was the one that chose both tool calls and wrote no text at all:
precisely what used to be discarded.

Each phase is now its own message entry, `role="thinking"`, built by
`app/routers/_messages_thinking.py`:

```json
{"id": "think_<turn>_<seq>", "role": "thinking", "content": [],
 "reasoning": "…", "reasoning_ms": 11800,
 "thinking": {"chars": 3201, "iteration": 7, "turn_id": "…"}}
```

- **Ordering is free, and that is why it is written on `thinking_done`.**
  The loop yields that event before `_commit_tool_calls` and before the
  `assistant_message` that flushes a text segment, so appending there
  lands the thought ahead of the tool and text rows it produced. No
  sorting logic; Inner Voice's timeline sorts on `timestamp` and slots it
  in for free.
- **The role is what keeps reasoning out of the transcripts, and it is
  the only thing that does.** Every producer generated from a session log
  branches on role first — the vault exporter and
  `_build_capture_transcript` in `app/post_capture.py`,
  `session_titles.build_transcript`, the `scripts/memory/*` renderers,
  `scripts/extract-trajectories.py`, `session_recall`'s corpus — and none
  has a `"thinking"` case. Measured, not assumed: change the role to
  `assistant` and six of the seven leak; leave the role alone and filling
  `content` leaks from none. `content` stays empty as a *second* layer,
  against a future producer that walks content without checking role. Do
  not read the empty content as the reason this works and conclude the
  role is free to change. `tests/test_thinking_trace_transcripts.py` pins
  both directions.
- **It never re-enters the prompt.** `thinking` is not one of
  compaction's conversation roles, so the rows are dropped before
  `_prepare_messages_for_harness` is reached. Preserved thinking
  (`loop._assistant_message_for_history`) is a separate, in-flight
  mechanism and is untouched.
- **A hard compaction discards the trace**, exactly as it already
  discards `subliminal` rows — the rewrite sets `data["messages"]` to the
  conversation-only set. The event log's `brain1.thinking_block_emitted`
  stays the durable record. Called out at that site so it does not read
  as a bug.
- **`accumulated_thinking` is cleared when a phase is flushed**, so what
  remains at the cancel and error paths is only a phase that streamed
  deltas and never reached `thinking_done` — which is exactly what those
  paths should still attach to their own message. Nothing is lost to a
  cancel mid-reasoning, and no phase is written twice.
- **`seq` rides on the SSE frame only when the trace is on.** That is how
  the browser knows a row is going to be persisted for this phase;
  without it, it withdraws its provisional row and falls back to hanging
  the reasoning off the assistant bubble. The kill switch therefore
  changes live rendering and reload rendering together rather than
  leaving them disagreeing.

The UI is one component — `ChatPanel`'s `ThinkingRow`, which the chat,
the right-hand chat sidebar and the Inner Voice timeline all mount, so
none of them needed its own work. The row must be handled **above**
`MessageRow`'s content guard: it has no content blocks and would be
dropped before it rendered. It opens on the first `thinking_delta` and
ticks locally, because `thinking_done` arrives after the iteration's
text and a row created there would sort below the answer it preceded;
`makeThinkingTracker` holds that bookkeeping in one place because the
file's two stream handlers are near-duplicates and drift between them is
the standing hazard there.

`harness.thinking_trace.enabled: false` restores the old behaviour.

## Code graph

**Vault half, held as a patch.** The `automod-change-own-code` skill's
blast-radius step lives at
`scripts/maintenance/vault-automod-skill-blast-radius.patch`, not in the vault,
until `code_graph` is actually deployed. The vault is a live shared tree with
no PR path, so editing it lands *immediately* — while the `graph_*` tools it
names only exist after lloyd-mcp restarts on the merged code. An edited skill
in that gap tells every automod round to call a tool that returns "Unknown
tool" and makes its own quality gate 4 unsatisfiable. Apply it after the
restart:

    git -C ~/obsidian apply scripts/maintenance/vault-automod-skill-blast-radius.patch

`tests/test_code_graph_doc_claims.py::test_automod_skill_maps_the_radius_between_opening_and_working`
fails with that command until it is applied. It carries `live_vault`, so the
automod gate (`-m "not live_vault"`) excludes it and no round is failed by a
vault someone else has not updated yet.


`agent_mcp/code_graph.py` answers "who calls this" and "what breaks if I
change it" from graphify's deterministic AST extraction of a tree
(`<root>/graphify-out/graph.json`, ~15 s to build, zero LLM calls). Six
tools: `graph_explain`, `graph_affected`, `graph_path`, `graph_hubs`,
`graph_status`, `graph_refresh`.

- **It is not a second MCP server, deliberately.** graphify ships
  `graphify-mcp` and mounting it would have been one config line, but Lloyd
  advertises every server's tools under bare names and `build_tool_list`
  raises on a cross-server collision; Task subagents pin
  `DEFAULT_LLOYD_MCP_SERVERS` and would never see it;
  `tests/test_mcp_layer.py` needs every configured server discoverable at
  test time, including inside a worktree where no second daemon is running;
  `agent-services/supervisor/**` is a protected automod path, so Lloyd could
  never repair the program running it; and graphify-mcp has no `affected`,
  which is the one query a change actually needs.
- **`root` is explicit and never inferred.** Nothing on disk links a chat
  session to an open round — `round_start` ledger rows carry no session id —
  so a "bound session's worktree" default would silently answer about the
  wrong checkout. Defaults to `LLOYD_HOME`; accepts an `SM_…` round id or an
  absolute path. When the answer is about the live tree and a worktree is
  open, the header says so.
- **Staleness is commit mismatch OR an uncommitted source file newer than
  `graph.json`.** The second rule is mandatory: inside a round HEAD does not
  move while the model edits, so a commit-only rule calls the graph fresh
  for exactly the window it is most wrong in. Auto-rebuilds from the dirty
  rule are debounced by `min_refresh_interval_s` (30 s); a debounced query
  still answers and says `STALE`.
- **`graphify-out/` must stay gitignored, unanchored.** A build inside a
  round dirties the tree, and both `scripts/automod/gate.py` and
  `promote.py` refuse a dirty tree — so an unignored build would abort the
  round on its own map. `*.json` at the top of `.gitignore` hid `graph.json`
  by accident; `GRAPH_REPORT.md`, `graph.html`, `.graphify_root` and the
  ~64k-file `cache/ast/**` were covered by nothing.
- **The graph is blind across process seams.** It is an AST extraction of
  one tree: there is no edge from `run_prompt_in_session` to `run_query`,
  because that call crosses HTTP, and none from `run_query` into a tool
  handler, because that crosses MCP. Keep Grep for string keys, route paths
  and config names.
- Ambiguity is an answer, not an error: `main` matches 88 nodes here, and a
  listing with ids lets the model pick where an error makes it guess again.
- There is no `enabled` flag. The kill switch is
  `mcp_servers.lloyd-mcp.disabled_tools`; an `enabled: false` that emptied
  `list_tools()` would break the annotation-staleness test.

## Resumable Task subagents

`Task` started from nothing on every call. That is right for a
fire-and-forget fan-out and wrong for the case that keeps recurring: a
subagent burns its budget mid-investigation, the caller reads the partial
answer, and the only way to ask a follow-up is to pay for the whole
investigation again — a fresh prompt, a cold KV cache, and no memory of the
forty tool results it just collected.

Every Task result now carries a `task_id`; passing it back with a follow-up
`prompt` continues that subagent.

- **`run_query` ignores `messages` when `chat_messages_handle` is non-empty**
  (`loop.py`). So a resume appends the follow-up to the *stored list* and
  passes that as the handle. Sending it as `messages` would drop it silently.
- **The stored run's identity wins**: `subagent_type`, profile, model,
  `base_url` and the `task:*` session id all come from the history.
  `current_parent_model` is deliberately not consulted — moving a
  half-finished conversation to another engine re-prefills all of it. The
  session id is reused so the continuation keeps its `tool_search`
  LoadedToolSet and its spill directory. `disallowed_tools` is the one thing
  merged live, so a tool switched off since the first run is honoured.
- **`task_id` is stable across continuations; `run_id` is not.** The
  dashboard shows one row per *run*, and a resume is a new run of the same
  task, linked by `continuation_of`.
- **Every exit path stores as well as closes.** One `_close` helper, because
  there are five exits and a run that closed its row without storing is a
  `task_id` the model was told about and cannot use. `CancelledError` stores
  and re-raises.
- **The sanitiser drops a trailing assistant message with unanswered tool
  calls.** That is the only invalid shape the loop can leave behind — a
  cancel or an exception between the stream ending and the dispatch
  completing — and replaying it makes every engine reject the request.
- Bounded and process-scoped: 8 tasks, 30 minutes, 3M chars. After an
  aggregator restart every id reads `unknown or evicted`, which is honest —
  the conversation is gone with the process. The three refusal reasons
  (`unknown or evicted`, `expired`, `still running`) are distinct because
  they call for different next moves.
- `SubagentRecord.to_dict` never exposes `chat_messages`; the dashboard row
  gets `task_id` and `continuation_of` only.

## Mission Control dashboard

The `dashboard` tab (first in the sidebar, desktop landing tab) polls one
aggregated endpoint, `GET /api/dashboard`, every 2s. It is deliberately a
single endpoint rather than one per panel: the page is open all day, and
eight requests per tick times however many tabs are open is real load on
a box whose job is holding a 262k-token KV cache steady.

Sections are gathered concurrently and **degrade independently** — a
wedged supervisord turns one panel into an error string and leaves the
rest live. A dashboard is most useful when something is broken, so it
must not be the second thing to break.

Where each section comes from:

| Section | Source |
|---|---|
| `host` | `app/host_metrics.py` — psutil + `nvidia-smi` (2s cache) |
| `vllm` | `app/vllm_metrics.py` — scrapes `<base_url>/metrics` per configured model |
| `primary` | `sessions_io.active_sessions_snapshot()` + `session_titles` |
| `recent` | the last chats to stop talking — bounded scan of `sessions/` |
| `agents` | **the lloyd-mcp process**, over loopback — see below |
| `services` | `app/supervisor_client.py` |
| `workers` | `workers.queue` + `workers.pool` — pool slots, per-source depth, recent runs |
| `autonomy` | `~/obsidian/autonomy/*.md` frontmatter + the pool's in-flight `scheduled-task` jobs |
| `backlog` | `~/obsidian/backlog/*.md` frontmatter |
| `usage` | `usage_store` |

Sections that walk the vault (`autonomy`, `backlog`) are TTL-cached for
10s — the backlog is 300+ markdown files and its status counts do not
change between 2-second polls. Live sections are never cached; they are
the point of the page.

`recent` is the third cached section and the one with a trap. A session
JSON carries its whole transcript (100 files, 7.5 MB today), so the scan
is bounded twice: only the newest `_RECENT_CANDIDATES` files by **mtime**
are opened, and the parse is cached for 10s. The mtime window is safe
only because mtime is never *earlier* than `last_active` — background
writers (the titler, post-session capture, TodoWrite) push a file's mtime
later than its last real message, so mtime can promote a stale chat but
never demote a fresh one out of the window. The rows are then sorted on
`last_active`, which is what `GET /api/sessions` sorts on too.

The live filter — dropping sessions with a running or queued turn, which
the panel beside it already shows — is applied **outside** that cache.
Caching it would leave a chat that just started reading as finished for
up to ten seconds. Cache the expensive scan, never the cheap freshness.

A section can also be *missing*, not just failed: a browser tab left open
across a backend restart polls the new build with the old snapshot shape,
and `section.error` on an undefined section throws inside render and
blanks the whole page — the one thing this design exists to prevent. Use
`sectionError(section)` from `api.ts`, not `section.error`.

**Overdue is not "next up."** `_autonomy` splits scheduled tasks on
`next_run` vs now and returns them as separate lists. Sorting them
together ascending and labelling the head "next up" is how a fleet whose
ticker is months behind renders as a healthy schedule — the most overdue
task lands exactly where the soonest one belongs. Likewise `completed`
is excluded from worker "open" counts (`_OPEN_STATES`): it dominates the
depth table and would bury the handful of items actually waiting.

**And overdue is not "held."** The clock is only one of five gates the
scheduler applies. A task also needs a skill, `up_next` status, no failure
cooldown, a satisfied `depends_on`, and the current hour inside its
`preferred_hours`. `autonomy.hold_reason` mirrors `_is_task_due`'s gates in
order and returns the first one that bites (`"paused"`, `"waiting on #42"`,
`"outside hours 00-04,23"`, `"no skill"`) or `None`; the panel calls a
past-due task **overdue** only when nothing holds it, and **held** otherwise.
Both `_autonomy` and `GET /api/autonomy/tasks` call that one function rather
than restating the gates — a second private definition of "due" is what this
fixed. On 2026-09-06 the dashboard showed six overdue while the scheduler
considered none of them late: four nightly jobs outside their window and two
paused. A nightly task is past due for the eighteen hours a day it is not
allowed to run, so the counter was never zero and therefore said nothing.
The `classifier` field reports `naive` when `autonomy` could not be imported
and every past-due task is being called overdue, because a downgrade that
looks like success is the failure this whole split exists to prevent. Note
that the dependency gate resolves `depends_on` by id and treats an
unresolvable id as *met*, so it must always be handed the **whole** board —
`/api/autonomy/tasks?status=up_next` classified against its own filtered list
would report every dependency satisfied.

**Front matter is bounded by its closing `---`, not by a byte count.**
`_frontmatter` reads in 4 KB chunks up to a 64 KB ceiling and stops at a
line-anchored `^---$`. The previous flat 3000-byte prefix silently dropped
five backlog items, and the selection was causal rather than random: an item
grows its `activity_log` precisely by being worked on, so the two it hid were
the two that were `in_progress` — the board reported zero. A cap that hides
whatever is most active is the worst possible reading of "bounded". Splitting
on bare `"---"` is the matching trap: it also fires inside quoted log prose
and truncates the block somewhere plausible. A block that parses to a list or
a string returns `{}`, since the caller's first move is `.get`.

**Subagents and background bash tasks live in the lloyd-mcp process, not
the backend.** The aggregator owns the `Task` tool and spawns
`Bash(run_in_background=true)` children, so the backend has no handle on
either. `agent_mcp/main.py` exposes `GET :8500/state` beside `/health`
and `app/routers/dashboard.py` reads it over loopback. Adding a new
agent-side live panel means extending that route, not the backend.

`background_tasks` carries `active` **and** `recent`. `list_active`
filters on `status == "running"`, so before that a background bash left
the dashboard the instant it exited — a task that died three seconds in
was indistinguishable from one that never started, which is the opposite
of what a background task most needs to report when nobody is watching
its terminal. `list_recent` is bounded by its limit rather than by
eviction: `_records` is kept whole so a later `get(task_id)` can still
hand the model an output path to Read. A finished row's `elapsed_s` is
measured against `finished_at`, not `now`, or a task that ran for two
seconds reads as hours old by evening.

**Workers are not in that panel.** The worker pool lives in the backend
(`workers.queue` + `workers.pool`, rendered by `WorkersPanel`), while
subagents and background bash live in the aggregator. A worker job whose
prompt calls `Task` does put subagent rows there — via
`workers/sources/_common.py::run_prompt_on_primary` — but anonymously:
nothing on the row says which worker source it came from.

`agent_mcp/_subagent_registry.py` opens a row **before** the Task run
loop starts — a `Task` blocks its caller for minutes, so a row created on
completion would only ever describe runs that no longer need watching.
Closing it is the subtle part: `finish` is idempotent and
first-writer-wins, so a blanket `finally: finish("cancelled")` runs
*before* the success path and silently stamps every completed run
cancelled. Each exit path closes the row with its own real status;
`tests/test_task_registry_wiring.py` pins that.

**Not every engine is vLLM.** The secondary slot (:8091) runs llama-server,
because a GGUF Q3 is the only build of Qwen3.6-35B-A3B that fits a 24 GB
3090 at the full 262144 window — unsloth's NVFP4 needs SM100+ and vLLM's
GGUF path does not cover this hybrid linear-attention MoE. It serves the
same OpenAI API, so the harness is unchanged, but it publishes `llamacpp:`
Prometheus names. `vllm_metrics._translate_llamacpp` renames them into the
vLLM vocabulary so one snapshot path and one dashboard card serve both.
Two things genuinely do not exist there and are reported as `None` rather
than `0`: live KV occupancy (no gauge) and TTFT (no per-request count).
A llama.cpp engine is `awake` whenever it is reachable — it has no
sleep-state gauge, and falling through to the vLLM check renders a healthy
engine "asleep". Its model name comes from a `/props` probe cached per
engine lifetime, since llama.cpp does not label its metrics.

**Counters vs. gauges.** vLLM exposes both. Gauges (`num_requests_running`,
`kv_cache_usage_perc`) are read straight. Counters
(`prompt_tokens_total`, `prefix_cache_hits_total`) are monotonic since
engine boot, and their absolute value says nothing useful, so
`vllm_metrics` keeps the previous scrape per engine and reports a rate.
A counter that goes backwards (engine restarted) yields `None`, never a
number — otherwise a restart renders as a one-second spike of the
engine's entire history. An unreachable engine drops its baseline for the
same reason.

### The Browser tab has one control, and four lists decide who can see it

The panel mirrors the agent's Chromium (`app/routers/browser.py` + the
frames `agent_mcp/browser.py` pushes). Its **URL bar** POSTs to
`/api/browser/navigate`, which the backend proxies to the aggregator's own
`/browser/navigate` route — Playwright runs in that process, the same seam
the dashboard crosses to read `/state`. Deliberately **not** an MCP call:
the user typing a URL is not the agent using a tool, and dispatching it as
one would write a `browser_navigate` into the transcript that the model
never made. The frame it pushes is tagged `url_bar` so the tab can say who
drove it. `navigate_from_ui` completes a scheme-less host to `https://`,
detecting the scheme by `://` rather than by the bare colon — the colon
alone reads the whole of `localhost:8080` as a scheme, and on this box that
is the first thing anybody types. `browser_navigate` stays strict, because
an agent omitting the scheme has made a mistake worth seeing.

Everything else on the page stays read-only, and not for want of a route:
the ref overlay has nothing to send, since the tool surface has no "click
pixel (x,y)" and the a11y tree is gone by render time.

### The SSRF guard that was never called

`_is_private_host` was defined in `agent_mcp/browser.py` on 2026-04-11 and
called from nowhere until 2026-09-09. Backlog #278's acceptance required that
"the SSRF host check at `agent_mcp/browser.py:50` is intact"; its triage
verdict called that line "the only network-shaped code" in the module; and
the test written to satisfy both asserted the predicate returned the right
booleans. All three statements were true and none was about a guard. **An
acceptance criterion that names a symbol at a line number certifies that the
symbol is still there.** `tests/test_browser_panel.py` now asserts the entry
points *invoke* it, which is the property that was missing.

The policy mirrors `http_tools.http_request` rather than `http_fetch`: block
the network the machine is on, allow the machine itself. Loopback stays
reachable because the agent browses Lloyd's own dashboard, and because an
injected prompt that wants loopback has Bash and the whole MCP surface
already — the browser is not the weak link there. The LAN is what this
closes: the router, the NAS, the printer.

**The check is on resolved addresses, not on the hostname string.** That is
what makes it more than a speed bump. `getaddrinfo` normalises every encoding
that beat the old regex list — `2130706433`, `0x7f000001` and
`::ffff:127.0.0.1` are all 127.0.0.1 to the resolver exactly as they are to
Chromium — and it classifies a name like `router.local` that no prefix match
can. `_resolve_addrs` is `lru_cache`d, which also blunts DNS rebinding by
holding the first answer.

**Route interception does not see redirects, and that is measured, not
assumed.** `route.continue_()` hands the request to Chromium's network stack,
which follows a 3xx internally and never re-enters interception. Instrumented
against a loopback server that 302s to this box's own LAN address, the
handler fires exactly once, for the first hop, while the redirected request
arrives only as a `request` event. So the interceptor covers a clicked link
and a subresource, and `_enforce_landing` covers the rest by asking where the
page actually *ended up* — which needs no enumeration of lanes, and so also
catches a meta-refresh and a `window.location`. It blanks the page on a hit,
because leaving it parked means the next `browser_snapshot` reads it and the
state mirror pushes a screenshot of it to Mission Control.

`_browser_snapshot`, `_browser_evaluate` and `_capture_state` are the three
places page content becomes model context or a human-visible frame, and each
one checks. `browser.block_private_hosts: false` or
`LLOYD_BROWSER_BLOCK_PRIVATE=0` turns it off.

**Four hand-written lists name the tabs**, and nothing made them agree until
`tests/test_mc_tab_parity.py`. `Page` in `web/src/components/Sidebar.tsx`
renders them; `mc_state.VALID_TABS` decides what the frontend may *report*;
`mission_control_ui._VALID_TABS` what the agent may ask for; `VALID_TABS` in
`useMcNavigationEvents.ts` what the frontend will *act on*. Drift is silent
in the worst direction: `browser` was in the union and in none of the three,
so a user sitting on that tab made `POST /api/mc/state` return 400 — and
`useMcStateSync` swallows the failure *after* recording the payload as sent.
The mirror kept serving whichever tab they came from, so `mc_get_state`
answered confidently and **wrongly** for as long as they stayed there. Not
"Lloyd doesn't know", which he could have said. `dashboard` was missing from
two of the three, the quieter half: the backend had carried a
`_summarize_dashboard` all along for a tab the agent was refused and the
frontend would have ignored. A tab absent from `_SUMMARIZERS` is the same
shape of quiet — `_summarize_tab` returns `{}` and the agent is told nothing
about where it just sent the user.

`_summarize_browser` must never carry `screenshot_b64` or `snapshot`: that
is ~124 KB of base64 plus 8 KB of accessibility tree, and the summary goes
into the model's context every time it moves the user to the tab. It reads
`browser_router.latest_frame_summary()`, which exists to make that the
default rather than a thing each caller remembers.

## Session titles and live activity

Every Mission Control surface that names a session — the chat history
list, the chat header, the dashboard's agent panel, the Inner Voice
picker — renders a few-word **title** rather than the timestamp id.
`app/session_titles.py` owns it end to end; the id survives as the
element's `title=` tooltip.

Titles are written by the **secondary** model
(`_sync_secondary_title`), fired and forgotten off turn completion
beside `_post_session_capture`. That slot is single-tenant
(llama.cpp `--parallel 1`) and agent turns already queue behind it, so
`should_title` re-titles on a **geometric** schedule — after the 1st
real user message, then the 3rd, the 9th, the 27th — recorded in
`title_at_count`. A per-turn title call would put a model call in that
queue for a label nobody asked to be refreshed.

`clean_title` is strict on purpose and returning `""` is a normal
outcome: a bad title is worse than none, because the id at least
identifies the row while `Here is a title for the conversation` just
looks like a bug. Consumers share one fallback chain —
`web/src/lib/sessionLabel.ts`, title → preview → id — so a session never
reads as two different sessions in two panels.

`title_for` caches on a **TTL, not on mtime**. The session JSON is
rewritten on every appended message, so an mtime-keyed cache would
re-parse a multi-megabyte transcript on every 2-second dashboard poll,
which is the exact cost the cache exists to avoid. `invalidate` closes
the staleness window when a title is written.

**Live activity** is the second half. `SessionTurn.activity`
(`{kind, label, detail, at}`) is stamped by the turn runner as it
streams — `starting` → `prefill` → `thinking`/`responding` → `tool` →
`working` — and surfaces through `active_sessions_snapshot`. "Busy" is
equally true of a turn prefilling 160k tokens, one four minutes into a
`Bash` build, and one wedged on a dead engine; this line is what tells
them apart. It is display state the loop never reads back, so writing it
is a no-op when nothing is running and a no-op again when the state is
unchanged (the text path calls it per token).

The snapshot itself stays **pure in-memory queue state** — it is also
the automod promoter's idle gate, and a disk read there would put the
filesystem in front of a restart decision. Titles are joined on in
`_primary_state`, off the loop via `asyncio.to_thread`.

Each row on that panel is a button that opens the session in the Inner
Voice tab, through the same `setPendingFocus` + `setCurrentTab` pair the
agent's `mc_navigate` uses — so `InnerVoicePage` never has to know who
asked. Two things that panel taught us:

- **A page that applies incoming focus must not race its own list
  fetch.** `loadSessions` used to read `selectedSession` out of its
  closure to decide whether to default to the newest session. On mount
  that closure captures `null`, the fetch resolves *after* the focus has
  been applied, and the stale `null` overwrites it — so every row on the
  dashboard opened the same chat. Use the functional updater
  (`setSelectedSession(prev => prev ?? list[0].session_id)`) and keep the
  callback's deps empty; anything else reintroduces the race.
- **The Inner Voice picker holds only IV-enabled sessions**, but focus
  can point anywhere. A `Select` whose value matches no option renders an
  empty trigger, so the picker carries an out-of-list selection in as its
  own option and names it from `/api/sessions/{id}/meta`.

## Model slots

`models.<alias>` in config.yaml is only the *endpoint*. Which model actually
answers there is decided by the supervisord program's `environment=MODEL=...`
and its start script — three places that can drift apart, and did on
2026-09-06 when an automod rollback reverted `agent-llm-secondary.conf` to a
launcher branch serving a 4B under the same alias and port as the 35B.

`models.<alias>.expect_model` is a case-insensitive substring checked against
what the engine reports (vLLM `/v1/models`.root, llama.cpp `/props`.model_path).
`app/model_identity.py` sweeps it at boot — detached, with retries, because a
cold 35B takes minutes to load — and logs ERROR on a mismatch.
`GET /api/models/identity?refresh=1` re-probes on demand. A slot with no
`expect_model` reports `unchecked`, so **update it whenever you swap a slot's
occupant** or the check is inert. `expect_kv_cache_dtype` /
`expect_kv_pool_tokens_min` are the same idea one level down — the right model
served the wrong way — see "Primary throughput" below.

Current occupants: primary `:8096` = Qwen3.8-Flash-Next (vLLM, GPU 1);
secondary `:8091` = Qwen3.6-35B-A3B UD-Q3_K_XL (llama.cpp, GPU 2, single-tenant
at ~21.7 of 24 GiB). Two venvs can serve the primary and
`start-qwen38-flash-next.sh` adapts to whichever `VLLM_VENV` names:
`vllm-qwen38-flash-next` (the PLE-offload-worker build, the script's own
fallback default) and `vllm-flash-next-main` (vLLM main, UVA offload — what
`agent-llm-primary.conf` serves since 2026-09-10, with `MAX_NUM_BATCHED_TOKENS=4096`
(see "Primary throughput") and `KV_CACHE_DTYPE=fp8` for a
×1.74 KV pool that also sidesteps the >200k-token QSA prefill cliff, at 0–17%
slower decode by text type because the drafter accepts fewer tokens through an
e4m3 cache — the step time itself is unchanged). SETUP.md
and the script's `KV_CACHE_DTYPE` comment carry the measurements; the pinned
table on the main venv makes the never-two-boots-in-quick-succession rule
above stricter, not looser. The secondary serialises (`--parallel 1`) because
llama.cpp divides `--ctx-size` across slots and the full 256K window was the
point — `secondary_models.py` post-session jobs and voice summaries queue
behind agent turns there.

**Subagents inherit the calling turn's model.** `subagents.<type>.model: ''`
means "whatever spawned me"; the harness ships it in the MCP request `_meta`
(`lloyd/model`, `lloyd/base_url`) since Task runs in the aggregator process
and has no other way to know. Pin an alias there to override. Empty
`base_url` resolves from `models:` for the chosen model — *not* from
`default_model_base_url()`, which always returns the primary's endpoint.

## Primary throughput: prefix misses, KV pressure, the KV gate

`architecture/vllm-throughput-mitigation.md` is the long version, with the
measurements.

The 09-09 "5 tok/s" chat was cold re-prefills. An agent loop re-submits its
whole 100-200k context every iteration, and when that prefix had been
evicted between iterations the engine prefilled it again, one 8,192-token
chunk per step, while every other request got one token per step. The FP8 KV
cutover (2026-09-10: a 692,263-token pool, 3200-token pages) is what fixed
it. Four things keep it fixed, and none of them touches the engine:

- **The fix is pinned and asserted.** `agent-llm-primary.conf`'s
  `environment=` carries `VLLM_VENV` and `KV_CACHE_DTYPE=fp8` — the
  launcher's own fallbacks are the BF16 worker build.
  `flash-next-bootfacts.sh` exits 1 when the last boot's engine config is not
  `kv_cache_dtype=fp8` or its pool is under 600,000 tokens (2 while the boot
  is still loading), and `models.primary.expect_kv_cache_dtype` /
  `expect_kv_pool_tokens_min` make the boot sweep read
  `vllm:cache_config_info` and log ERROR on a BF16 boot.
- **A miss is counted.** `app/prefix_miss.py`, called by all three usage
  writers (streaming chat, sync chat, `run_recorder`): an iteration ≥ 3 whose
  ≥100k prompt is less than half cached is a `brain1.prefix_miss` event, the
  turn's usage row carries `reprefill_tokens` and `prefix_misses`, and the
  dashboard's Tokens panel shows the last 24 h. Three rules, each one of the
  investigation's traps: iterations 1–2 never count; a turn that reads zero
  on every counted iteration is **unmeasured** (NULL), because that is
  exactly what the unparsed field looked like before `5531f21`; and only miss
  iterations sum into `reprefill_tokens` — a healthy iteration re-prefills
  its appended tail (5–7k), and sixty of those must not read as 300k.
- **It is announced, once.** A turn whose misses pass 100k while another
  request was running fires `announce()` on the guardian fan-out — journal
  and toast, voice off by config — once per turn and at most every 30
  minutes. A cold prefill on an otherwise idle engine hurts nobody and says
  nothing.
- **Long-lived jobs wait for room.** The pool's KV gate
  (`architecture/workers.md` §2): sources marked `LONG_LIVED` are not claimed
  while the primary's one-minute **median** KV is over 60%. The median, not
  the last sample, because the gauge lies during a cold prefill: a 200k prompt
  being built drives it from 0.20 to 0.96 and it falls to 0.50 the moment
  the prompt is in (~2.5x its resident footprint — the hybrid model's
  prefix-cache checkpoints). It fails open.

`app/engine_pressure.py` samples the primary's /metrics every 5 s for all
three readers — through a stateless parser, so the dashboard's rate baseline
is untouched — and the dashboard's KV meter carries its 5-minute p90 against
a 65% line. The gauge counts blocks *referenced by running requests*; a
paused turn's cached prefix sits in the free remainder, which is where it
has to survive until its next iteration.

The chunk budget is 4096, not vLLM's 8192 (`MAX_NUM_BATCHED_TOKENS` in the
program's `environment=`). A cold 200k prefill beside a chat costs it 333 ms
a step instead of 608, and no longer transiently references ~2.5x its KV —
the overshoot that evicted neighbours' prefixes — for 7% more prefill time.
2048 halved the step again for +27% and was not taken.

Two smaller changes rode along. The compaction wall moved,
`compaction.microcompact` 0.8/0.6 → 0.72/0.52 (≈151k → 109k of the 210k
threshold, same band width), and the in-turn pass reads those fractions now —
it ran on `RunOptions` defaults before. And the Inner Voice observer's calls
take the priority of the turn they watch, so a chat's observer runs at 0
instead of queueing at 1 behind every worker iteration.

`agent-services/bin/bench-admission-stall.py` is the durable reproducer
(`verify`: a warm loop and a cold re-admission through the production
accounting; `cold`: the chunk-budget A/B shape). Run it with nothing else on
:8096; it refuses a busy engine.

## Voice output

The cloned voice (`clone:dave_cullen`, config `livekit.tts`) is synthesised by
Qwen3-TTS at :8090 and then **shaped client-side** by
`agent-services/tts_shaping.py` before it reaches LiveKit. Two things the
server does not do, both fixed in the worker because the TTS tree is
gitignored and a rebuild would silently delete a fix made there:

- **The 12 Hz speech tokenizer rolls off above ~1.5 kHz.** Against the
  clone's own reference clip the output matches the real speaker to within
  0.5 dB below 1.5 kHz and is then 1.8–6.2 dB down all the way up. That
  missing presence band is the "he's in a broom closet" sound. Two high
  shelves (`livekit.tts.shaping.shelves`) put it back. It is a vocoder
  property, not a bad reference — built-in voices with no cloning measure the
  same, and re-cutting the reference from another source video did not move
  it. **Do not "fix" it by cutting 300 Hz**: that band already matches the
  reference to 0.1 dB, and cutting it trades hollow for thin. The trap is the
  metric — gate frames on total RMS and raising the highs swaps vowel frames
  for fricatives in your own measurement, which reads a +9 dB shelf as +24 dB.
  Gate on sub-1 kHz energy, which the correction cannot move.
- **`speed` is dropped by the server's streaming path.**
  `generate_voice_clone_streaming` has no such parameter while the
  non-streaming path applies `librosa.effects.time_stretch`, and voice mode
  always streams — so `livekit.tts.speed` was inert for the only path that
  uses it. `WsolaStretch`
  applies it in the worker, and the request now sends `speed: 1.0` so a future
  server-side implementation cannot stretch twice. WSOLA rather than a phase
  vocoder: a phase vocoder adds exactly the smeared quality the shelves exist
  to remove.

Pace is set by ear, not by matching the reference's words/second: the w/s
match puts `speed` at 0.70 and that is audibly too slow, because the reference
is one deliberate segment and the model places its pauses differently. Every
setting from 0.70 to 1.00 measures *faster* than the reference by w/s. Sweep a
single synthesis across settings rather than re-synthesising per setting, or
sampling noise reads as the effect of the knob.

An utterance also ends in `livekit.tts.tail_silence_ms` of silence and waits on
`AudioSource.wait_for_playout()`: `_stream_utterance` returns when audio is
*queued*, not played, so `on_utterance_end` fired early and a following
`interrupt()` → `clear_queue()` cut the last syllable off.

Both stages hold state across chunk boundaries and are reset per utterance —
an interrupt mid-stream must drain the shaper or the next utterance opens with
the tail of the one the user talked over. `tests/test_tts_output_shaping.py`
pins chunk-invariance, rate stability, and pitch preservation. Measurements and
listen files: `~/obsidian/projects/lloyd/voice/voice-source-dave-cullen.md`.

A voice change needs **`lloyd-agent-worker`** restarted (it reads
`livekit.tts` once at construction); `agent-livekit-server` is the SFU binary
and never reads TTS config. It also needs the guardian **re-staged**
(`systemctl --user restart lloyd-guardian`), because the same voice is used
for spoken alerts and `sync-voice-config.py` pushes `livekit.tts` across at
stage time — see "The spoken channel" above. Neither restart is required for
the voice to *work*, only for a change to reach that consumer.

## Background runs are recorded; watching them is opt-in

`architecture/background-runs.md` is the long version, with the measurements.

Three paths run an agent loop and until 2026-09-10 only one of them wrote
anything down.

| path | transcript | event log | Inner Voice | platform |
|---|---|---|---|---|
| `autonomy.run_task` | yes | yes | per-task | `autonomy` |
| `_common.run_prompt_on_primary` | yes | yes | never | `worker` |
| `_common.run_prompt_in_session` | yes | yes | per-source | `worker` |

The first two called `run_query` directly and kept the text, so the record of
what a scheduled task *did* was a 200-character summary in an
`autonomy-runs/*.md` file. On 2026-09-10 an autonomy task was the prime
suspect in a full vault wipe and could be neither confirmed nor cleared,
because the tool calls it made had never been written down anywhere.

**Two axes, and conflating them is the mistake.** Every background run is
**recorded** — cheap, universal, a few file appends. Being **observed** by the
Inner Voice critic is a separate opt-in, because the observer runs on the
PRIMARY at priority 1 and spends a goal-extraction call plus a critique per
turn, in front of whatever a human is typing.

- **`app/run_recorder.py` is a passthrough**, not a new owner of the stream:
  an async generator that persists each event and re-yields it unchanged, so
  the caller's loop, options and hooks are untouched. Routing the direct paths
  through `/api/message/stream` instead — a stale `lloyd-sandbox` branch tried
  it, do not merge — drops `run_task`'s #534 grant gate, which the chat
  endpoint does not install, and has to re-plumb `saw_tool_call`,
  `tool_errors` and the timeout partial over the wire.
- **Persistence is incremental and the final flush is shielded.** A run killed
  at its deadline is how the interesting ones end; nothing is buffered until
  the `result` event a killed run never emits.
- **Recording may never break the run.** Every step is wrapped: a bug in there
  costs the record, not the work.
- **`turn_id` is the load-bearing option**, not `session_id`. It switches on
  the per-turn change ledger, so an unattended run's file writes leave
  pre-images and can be reverted. It was the only path with no undo.
- **`app/transcript_entries.py` is the one shape.** `messages.py` built five
  assistant variants and three tool-pair variants inline across its streaming,
  cancel and error paths. Both writers call the builders now, and
  `tests/test_transcript_entries.py` pins that the same events produce
  identical entries.
- **`sessions_io.create_session` is the one writer** for every non-chat
  session. `new_worker_session` had its own copy and already disagreed — `id`
  where the chat path writes `session_id`, and no `last_active`, so a worker
  session sorted by file mtime while every chat sorted by its conversation.
  **Titles are set at creation**, so a background session never queues behind
  the single-tenant secondary for a label nobody asked to have refreshed.
- **The pool collects the sessions a claimed job created**
  (`current_run_sessions`)
  and writes them onto the run row, including on the timeout and exception
  branches — a timed-out run is the one most worth reading. Collected rather
  than returned by each source, because "the handler remembered to pass it
  back" is not a property worth depending on eleven times.
- **Retention is platform-aware**: a background transcript gzips at 30 days
  against a conversation's 90, same never-delete rule. An unreadable
  `platform` keeps the *longer* window — a few stale kilobytes beats losing a
  conversation from the history two months early.

Kill switch: `harness.background_recording.enabled`.

### One definition of "is a human reading this session"

Six readers hand-rolled `platform == "autonomy"` and none had learned about
`worker`. Worker turns arrive through the chat path, so the sessions the
machine ran for itself were in the user's chat history, in the recent-chats
panel, titled by the LLM titler, written into the daily note, fact-extracted
into the knowledge graph as things the user said, and recalled back by
`session_recall` as the user's own conversations. `sessions_io.is_user_session`
is the definition; `tests/test_session_platform_checks.py` greps for the
literal, because a seventh reader written next month is how this comes back.

- **The id shape is a fast path, and in the chat listings it decides.** A
  background id has four underscore-separated parts
  (`20260910_120001_autocode_9f2a`) and a chat's has three, so most of the
  directory is skipped without being opened — ~240 background sessions a day
  against ~14 chats, measured over the week to 2026-09-10. `/api/sessions` and
  `_scan_recent_sessions` skip a four-part id *unread*, which is safe only
  while nothing that creates a user session mints one;
  `tests/test_session_platform_checks.py` pins the three mints. The Background
  listing parses every four-part file and judges it by `platform`. Eight
  `…_autonomy_…` sessions from the sandbox experiment at 2026-09-09 15:38 are
  four-part and labelled `mission-control`, so they appear in neither.
  `_scan_recent_sessions` used to keep the newest 24 by mtime and *then* drop
  non-user rows; on 2026-09-10, before autonomy was recorded at all, 22 of the
  newest 24 were already background. It now stops at `_RECENT_KEPT` user rows
  or `_RECENT_CEILING` files opened.
- **Post-capture keeps the markdown export for every platform** but writes
  background exports to `_pipeline/vault-derived/sessions-background/`, outside
  the qmd watch: `qmd-watcher.sh` *embeds* `sessions/` on every change, and the
  ~70 session-backed worker transcripts a day that reach post-capture would
  each be an embedding job over the machine talking to itself, against ~14
  chats — drowning the corpus. The daily note, the secondary summary and fact
  extraction are for user sessions only. Only chat-path turns fire
  post-capture at all, so a direct-path run's record is its session JSON.
- `GET /api/background/sessions` is the other listing, and the **Background**
  tab is its only reader. A row opens in the Inner Voice reader through the
  same `setPendingFocus` + `setCurrentTab` pair `mc_navigate` uses.
- Its second half is `GET /api/workers/health`, the workers' answer to
  `/api/autonomy/health`. `/api/workers/status` reports what a source is
  *allowed* to do and how much is queued; nothing joined a source to its
  outcomes, so one failing every run looked exactly like one succeeding at
  every run. `fail_rate` is **null** over zero runs, never 0.0 — "0% failing"
  for a source that has never run is the reading the panel exists to prevent.

### The observer, when a job asks for one

- **Worker sources**: `workers.sources.<name>.inner_voice`, read through
  `_common.source_inner_voice`. `deep-research` passed `inner_voice=False` as
  a literal until this landed, which is to say it was not a setting; no
  session-backed source passes the argument now, because a per-source switch a
  caller can override with a literal reads as broken the one time somebody
  uses it. On: `autocode`, `autotriage`, `youtube-digest`. Off:
  `deep-research`. The key is set only on sources that can be observed at all
  — the observer is wired in `app/routers/messages.py` and nowhere else, so it
  means nothing on a `run_prompt_on_primary` source, and `/api/workers/health`
  reports it tri-state rather than inviting a knob nothing reads. UI-mutable
  through `data/tool_overrides.yaml`, like `workers.enabled`.
- **Autonomy**: per-task `inner_voice:` frontmatter beats `autonomy.inner_voice`
  in config.yaml; fleet default off. The key is in `_parse_task_file`'s
  `fallback_fields`, so a file that needed the degraded parser does not
  silently lose its opt-in.
- **The observer attaches to the direct path.** `attach_observer_for_turn`
  creates a registry only when none exists, so passing `run_task`'s own
  `task_hooks` keeps the #534 grant gate *and* adds the observer — losing the
  gate to gain the observer would be a bad trade made silently.
  `options.cancel_event` is wired so the one lever that matters for unattended
  work reaches the loop; the ambient and clarify callbacks are `None` because
  both exist to reach a human mid-turn and nobody is reading. It closes in a
  `finally`, and a failure to attach costs the second opinion, never the work.
- Importing a router helper into `autonomy.py` is a layering smell, accepted
  rather than relocated: the alternative is a second definition of "how a turn
  is watched".

### The grant gate belongs to the endpoint

#534's authority gate is a PreToolUse hook, so it exists only where a caller
installed it — and the two callers that did are the two that build their own
`HookRegistry`. Every session-backed worker posts to `/api/message/stream`,
which installs Inner Voice, the destructive-Bash safety hook and skill
dispatch and not that one. So the four turn paths that read untrusted text and
rewrite this repo were the ungated ones, while the two with their own registry
were covered twice.

The trigger is the **session's own platform**, so a caller cannot forget
because a caller is not asked. A payload `grant_scope` also arms it and wins
when present: `policy.current_scope` is a contextvar the pool binds around the
claimed job, correct in the pool's own task and gone across the loopback POST,
and it is the difference between a grant made to `autonomy-task:39` and one
made to whatever runs on that source next. `grant_create` is banned on those
turns, written back into the request body because the disallowed list is
re-read every harness iteration. All three registry-building sites in the
router arm it, including the ambient builder where it cannot fire today —
"the other endpoint is the ungated one" is the shape of the bug being closed.

Nothing a worker does today is denied by it: tier 1 is everything
unclassified, so `Bash`, `Edit`, `Write`, `backlog_write_task` and
`vault_write` all pass. **It is therefore not vault protection** — only
`app/harness/safety.py`'s destructive-Bash patterns stand between an
unattended turn and `rm -rf`.

## Workers: one queue, everything unasked

Scheduled autonomy tasks, research, session mining, backlog triage and the
automod round that implements a confirmed item all run through one SQLite
queue drained by `workers.slots` asyncio workers **inside the backend
process**. `architecture/workers.md` is the long version.

- **`priority ASC` — a lower number runs sooner.** Read backwards once
  already: `autocode` sat at 80, behind research jobs at 70 that
  arrive every few minutes, and the rarest, most valuable job in the pool had
  no path to a slot.
- **`max_inflight` is applied in the SQL, not to a window of rows.**
  `claim_next` used to select 50 and skip over-quota rows in Python, so fifty
  queued items from one saturated source hid every claimable row behind them
  and the pool read the queue as empty.
- **A raised failure retries; a returned `{"status": "failed"}` does not.**
  That is the whole difference between "infrastructure hiccuped" and "the job
  failed and re-running it will not help". Raising for the latter is what made
  one timed-out autonomy task re-run three times at 600 s before the
  scheduler's own cooldown was consulted.
- **`skipped` is a third outcome and must be said as a status.**
  `automod-regression` signalled "I could not measure anything" by returning
  `{"skipped": reason}` — a key where a status belongs — so all 22 of its runs
  read as successes with an empty summary. `pool.normalize_result` is the
  contract now, and it reads that shape.
- **Nothing in a source may block the event loop.** It is the loop that serves
  every HTTP request and streams every chat turn, so a `subprocess.run` inside
  `execute` does not slow the pool, it stops Lloyd answering.
  `automod_regression` ran two 900-second eval arms there.
  `tests/test_workers_pool.py` greps for the pattern.
- **An empty turn is a failed turn.** `run_prompt_on_primary` returns a
  `TurnResult`, not a string, because a turn that dies at `max_turns` yields
  no text and an empty string is indistinguishable from a short answer. It
  used to return just the text: **225 of the 498 notes under
  `pending-research/` have the body `(no response)`**, and the source
  responsible ticked each topic off on the way, so none can be retried. That
  source, `domain-research`, has since been retired — see
  `architecture/research-pipeline.md`.
- **Research runs off a registry now, not a checklist.** `research.db`
  (`app/research_store.py`) holds topics with a lifecycle; task #65 proposes
  into it through `research_propose` and the `deep-research` source drains it
  through the deep-dive skill. The checklist it replaced held 2,839 items of
  which 314 were unique, and reading it cost 1.02M tokens a night.
  **Retries live in the registry**, because the pool completes an in-band
  `failed` item and never retries it.
- **A session-backed turn's own timer must beat the pool's.**
  `run_prompt_in_session` bounds itself at `max_duration_seconds` minus 60 s
  and cancels the turn in the backend on expiry. If the pool's `wait_for`
  wins, it cancels the HTTP request — and the chat path deliberately keeps
  running when its client disconnects, so the turn is orphaned while the pool
  requeues and re-selects the same item.
- **`session-distill` distils a session once, quiet, and only if a user wrote
  it.** Its watermark moved with each new message, so an active chat re-
  qualified on every tick — one was distilled 44 times — and worker sessions
  were mined back in as observations about the user.
- **`workers.enabled` is UI-mutable, so it lives in the override file.**
  `POST /api/workers/enable` used to `yaml.dump(CONFIG)` over the tracked
  `config.yaml`, which would have written expanded secrets into the tree,
  flattened its comments, and dirtied it — stopping the automod loop. Same
  route as the Tools page now.

### A budget the model cannot see is a deadline it cannot meet

An autonomy run is bounded by wall clock — `asyncio.timeout(timeout_seconds)`
in `autonomy.run_task` — and until 2026-09-08 nothing told the model that clock
existed. The chat path has warned at 75%/90% of `max_turns` since
`_build_state_anchor` landed, but that counts *iterations*, which is not the
budget an autonomy task dies on, and `run_task` calls `run_query` directly and
passed no `state_anchor` at all.

The failure that produces is silent and looks like a stall. Task #80 runs
`validate_okf.py`, which takes **2.4 seconds**, on a 300 s budget. Three
consecutive runs on 2026-09-08 failed, all three recorded
`(no output before timeout)`, and the task auto-disabled at `max_retries`. It
had not hung: the run records show 39 completions and real findings in the
partial text. The model had the answer inside the first minute, kept
investigating — re-verifying counts against a frozen vault snapshot, diagnosing
a bubblewrap sandbox that does not exist (nothing in `agent_mcp/builtin_bash.py`
sandboxes anything) — and was killed without ever being asked to write it down.
#78 and #24 died the same way in the same window.

`_build_deadline_anchor(timeout)` is now passed as `RunOptions.state_anchor`,
firing once at 70% and once at 90% of the resolved budget. Three things about
it are load-bearing:

- **It is built from `timeout`, not `declared_timeout`.** The pool clamps the
  frontmatter value (`max_duration - _POOL_TIMEOUT_MARGIN`), so warning at 70%
  of the *declared* budget can land after the kill.
- **The two levels say different things.** At 70% it is "do not open new lines
  of investigation"; at 90% it is "stop calling tools and write the report from
  what you have". A run that spends its last seconds on one more tool call
  reports nothing at all, and a partial report beats a failed run.
- **Each level fires once.** The chat anchor's rule, for the same reason: a
  warning re-sent every iteration is one the model learns to skip.

Resolution is one iteration — a single tool call longer than the remaining
budget still overruns. That is the accepted limit; the failure being fixed is
twenty short iterations past the stopping point, not one long one.

The skills matter as much as the mechanism, because a budget only helps a task
that knows when it is done. All three had an unbounded step:

- **#80** named `okf_migrate.py --apply` as the repair path without saying "not
  from this task", and its last successful report *ended in a question*
  ("Want me to run the migrate pass?"). Nobody answers an autonomy report, so
  the next three runs went looking for their own permission to act.
- **#78** Step 2 asked the model to hand-filter "unreferenced notes" — but only
  711 of 4,448 vault files contain a wikilink at all, so the sweep returned
  ~3,600 files, 84% of the vault. Its Step 3 was `[! -f "$file" ]`, which is not
  valid shell: it raises `[!: command not found` every iteration and has never
  detected a missing skill.
- **#24** Step 1 ran `nightly_extraction.py` in the foreground. The Bash tool
  defaults to 120 s and caps at 600 s; that script's last seven real runs took
  545–1666 s, so the call *cannot* complete and the model improvises a `nohup`.
  On 2026-09-08 the improvised run finished fine 40 minutes later — the task had
  already timed out, so `files_processed=133 facts=4969 failed=1048` was reported
  to nobody. (Those 1048 were one incident, not 1048 problems: the extractor
  talks to the primary on :8096 and that engine restarted mid-run, so every
  remaining document raised `Connection refused`. Unhashed files retry.)

`tests/test_autonomy_budget_anchor.py` pins the anchor.

Pause and drain the pool before restarting the backend
(`POST /api/workers/pause`): a worker turn killed mid-flight logs connection
errors that land in the guardian's observation window and get blamed on
whatever just landed.

## Structured verdicts

Worker verdicts used to be parsed out of `VERDICT:` / `SURFACE:` lines by
regex. That works until a turn words it slightly differently, and then a
`confirmed` is recorded as `unverifiable` and an item is retired for a
formatting reason.

`RunOptions.final_schema` asks the loop for one extra completion after the
turn ends, restating its conclusion as a JSON object
(`app/harness/finalizer.py`). Three things decide whether it is worth having:

- **The extra request must send the identical `tools` array with
  `tool_choice: "none"`.** Qwen renders the tools array inside the system
  message, so dropping it diverges the rendered prompt at token 41 of 1536 —
  2.7% in — and everything after that is a cache miss. Measured in the
  production shape (four tools-bearing iterations on a 180k conversation, then
  one finalizer): keeping tools reuses 177,600 of 180,068 tokens and takes
  1.19 s; dropping them reuses nothing and takes **25.00 s**. 21x, for one
  object. `eval/measurements/finalizer-2026-09-08.md` has the runs, and the
  two ways to measure this wrongly — `cached_tokens` reads 0 for the first two
  requests of any prefix on this engine, and an alternating A/B ends up
  caching both shapes, which is not what production does.
- **It is skipped unless `stop_reason` is `stop`/`end_turn`.** Forcing a
  verdict out of a turn that died at `max_turns` recreates the failure
  `INCOMPLETE` was added to fix. The reason is reported as `structured_error`,
  so a caller can tell "the turn never reached a verdict" from "the model
  refused to produce one".
- **The regex stays.** The finalizer can be skipped or can fail, and a verdict
  pipeline with no fallback turns a transient engine error into a lost triage.
  `parse_verdict(text, structured)` prefers the object when its verdict is
  known and records `source`; the ledger carries `verdict_source` and
  `structured_error` so a finalizer that quietly stopped working does not look
  exactly like one that is working.

`TRIAGE_VERDICT_SCHEMA` is built from `VERDICTS`/`SURFACES` rather than
restated — one list, or a new verdict lands in the grammar and not the
validator. It carries **no `maxLength`**: that is enforced by the guided
decoder, so the model would stop mid-sentence at the limit rather than write
something shorter. The clamps stay in Python, after the fact.

The router honours `final_schema` only for a session whose platform is in
`sessions_io.NON_USER_PLATFORMS`. A chat turn that quietly ran a second
completion under a grammar would be paying tokens for something nobody reads.

Kill switch: `workers.sources.autotriage.structured_verdict`, carried in
the queue payload like the budgets so a queued item runs under the config that
was live when it was enqueued.

## YouTube channel digests: the script fetches, a session judges

Two channels are tracked for new agent and model techniques — AI Engineer
(`@aiDotEngineer`) and Discover AI (`@code4AI`). Each new video becomes a
vault note under `knowledge/youtube/<Channel>/` **and** an answer to one
question: does it hold anything that would improve Lloyd? When it does, a
draft backlog item is filed (tags `youtube-eval`, `<channel-key>`).

The split is the design. `scripts/youtube_channel_monitor.py` is the
deterministic half — the channel registry (`CHANNELS`), `seen.json` per
channel under `~/.local/share/<channel>/`, the upload listing, and
`--fetch`, which writes the transcript, metadata and link enrichment into a
bundle directory. `workers/sources/youtube_digest.py` is the judgement half:
one **real session per video** through `run_prompt_in_session`, Inner Voice
on, that reads the bundle, writes the note, evaluates it against
`eval/lloyd_profile.md` and files the draft. Before 2026-09-08 autonomy task
#75 did all of this inside the script with a direct POST to the model and
thinking off — nothing of it was a transcript anyone could read, which is
why it moved. Task #75 is paused; the old name `ai-engineer-monitor.py` is a
shim onto `--channel ai-engineer` and the script path (`--process-one`)
still works as an operator fallback.

- **The eval rule is Alan's:** an open-source framework, tool or model may
  be proposed for direct adoption; a commercial product is never adopted,
  only the aspects worth recreating locally are named; a paper becomes a
  bounded experiment. It is in the prompt, and `eval/lloyd_profile.md` is
  the rubric's picture of Lloyd — keep that file current when the stack
  changes, or the eval will propose what already exists.
- **Disk decides.** The note must exist at the path the source chose and
  carry the video's id; a `FILED: #n` claim is checked against
  `~/obsidian/backlog` before it is recorded. The verdict is stored on the
  `seen.json` row and projected into
  `~/obsidian/projects/lloyd/channel-eval/<channel>.md` after every run.
- **The script owns the retry.** The pool never retries an in-band
  `failed`, so the source reports through `--fail` and `_is_retry_eligible`
  decides when `--pending` offers the video again. A `DrainActive` leaves
  the row `fetched` and the bundle is reused next tick.
- **Tracked from a date means a floor.** `--since-days N` registers the
  window and records the oldest video in it as `floor_video_id`; the
  new-video walk stops there. Without it a channel is crawled back through
  its whole history one video at a time, which is what happened to AI
  Engineer (628 notes). `--since-days N --requeue` also puts completed
  videos in the window back through the session so they get the eval.
- **Transcripts are wrapped at 100 columns** because the Read tool pages by
  line, and a 40-minute talk arrives as one 60 kB line.
- The toolbox denies `Bash`, `Edit`, `Task`, the automod tools and every
  queue writer: a transcript is untrusted text. `Read`, `Write` and
  `backlog_write_task` stay because they are the job, and vault writes
  outside `knowledge/`, `backlog/` and the report directory are reported.

## Knowledge graph

Two layers, and the distinction matters:

- **Fact layer** — markdown, one dir per entity under
  `_pipeline/vault-derived/facts/<Entity>/<Entity>-<category>.md`. Human
  readable, editable, diffable.
- **Store** — `_pipeline/vault-derived/kg.sqlite`, behind `app/kg_store.py`.
  Edges, aliases, the entity registry and a fact index derived from the
  markdown.

**Nothing opens the store except `app.kg_store`.** Not a script, not a
router, not a test fixture. Before 2026-09 the same state lived in two JSON
blobs that six programs rewrote whole with no lock, which produced the
2026-08-22 wipe (12,131 edges) and the 2026-09-03 merge incident (151
entities fused against a 2-edge graph).

```python
from app.kg_store import store
s = store()
s.edges.add({"source": "Lloyd", "target": "vLLM", "type": "uses"}, origin="fact_relate")
s.aliases.resolve("vllm")     # -> "vLLM"
s.facts_idx.for_entity("Lloyd", category="state")
```

Rules worth not relearning:

- A store that will not open raises `StoreUnavailable`. Never return an empty
  graph on a read failure — a writer will persist that emptiness.
- Expire edges, never delete them. `rewrite_endpoint` returns `(old_id,
  new_id)` pairs so a merge is exactly revertable.
- `LLOYD_FACTS_ROOT` / `LLOYD_KG_DB` point the fact tree and the store
  elsewhere — that is how a rebuild extracts without touching the live one.
- The extraction corpus is an allow-list in
  `scripts/memory/next-gen-memory/pipeline_config.yaml`. Edit the config,
  not the walk.

`architecture/knowledge-graph.md` is the long version.

## Config Structure (config.yaml)

```yaml
model:
  default: primary

models:
  primary:
    alias: primary
    base_url: http://127.0.0.1:8096
    context_length: 262144
    expect_model: Qwen3.8-Flash-Next   # identity check; see below
    env:
      ANTHROPIC_BASE_URL: "http://127.0.0.1:8096"
      ANTHROPIC_API_KEY: "no-key-required"
      ANTHROPIC_CUSTOM_MODEL_OPTION: "primary"
      ANTHROPIC_CUSTOM_MODEL_OPTION_NAME: "Primary"

harness:
  stream_chunk_timeout_seconds: 60      # gap BETWEEN SSE lines, not TTFB
  todo_anchor_interval_iterations: 10   # re-append session.todos this often
  preserve_thinking_iterations: 6       # carry N iterations' reasoning back
  tool_search:            # progressive disclosure; baseline + ToolSearch
    enabled: true
    threshold_tools: 30
    baseline_tools: [Bash, Read, Edit, http_search, http_fetch, ...]

subagents:
  general-purpose:
    system_prompt: ""
    max_turns: 40
    disallowed_tools: []
    model: ''        # '' = inherit the calling turn's model
    base_url: ''     # '' = resolve from `models:` for the chosen model

mcp_servers:
  lloyd-mcp:
    type: streamable-http
    url: http://127.0.0.1:8500/mcp
    disabled_tools: []  # bare tool names, e.g. [Bash, browser_screenshot]

agent:
  max_turns: 60
  permission_mode: bypassPermissions

session_titles:
  enabled: true    # false => surfaces fall back to the preview text
```

## Development Notes

- Each turn reconstructs the full conversation from the persisted session JSON (`load_and_compact_session`) and sends it as an OpenAI-format `messages` list to vLLM.
- vLLM tool calling: `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`
- Session continuity: no `resume=` — history is rebuilt from `sessions/<id>.json` each turn.
- The `/api/message/stream` endpoint uses SSE. The frontend connects via `fetch` + `ReadableStream`.
