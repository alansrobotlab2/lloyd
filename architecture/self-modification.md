---
segment: architecture
tags: [architecture, lloyd, selfmod]
type: reference
status: implemented
date: 2026-09-06
---

# Self-modification with automatic rollback

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
 propose ──► WORKTREE ──► GATE (7 rungs) ──► PROMOTER ──► live tree
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
| Round | `scripts/selfmod/round.py` | its worktree |
| Gate | `scripts/selfmod/gate.py` | nothing (read-only judgment) |
| Promoter | `scripts/selfmod/promote.py` | live tree, `current.json` |
| Guardian | `agent-services/guardian/` | live tree, `last_known_good.json` |

**The promoter never advances last-known-good and the guardian never
promotes.** That split is what makes LKG mean *observed healthy in
production* rather than *passed a pre-flight*, and it guarantees a rollback
always targets a commit that already survived a full observation window.

**The round is a wrapper, not a code generator.** `selfmod_start` returns a
worktree path; Lloyd edits it with the ordinary Edit/Write/Bash tools and
commits. There is deliberately no `selfmod_write_code` tool — proposing a
change is just normal work that happens somewhere safe.

---

## 3. The process

### 3.1 For Lloyd (autonomous)

```bash
python -m scripts.selfmod.round start "make X faster"   # cuts a worktree
#   ...edit that worktree, commit inside it...
python -m scripts.selfmod.round gate  SM_<id>           # 7 rungs, ~2 min
python -m scripts.selfmod.round land  SM_<id>           # idle-gated, verified
```

Or the same four steps as MCP tools: `selfmod_start`, `selfmod_gate`,
`selfmod_land`, `selfmod_status` (plus `selfmod_abort` and
`selfmod_rollback`). Every mutating tool refuses while `selfmod.enabled` is
false, which is the default.

### 3.2 For humans (this repo's development)

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

## 4. The gate

Seven rungs, cheapest first, short-circuiting. **Every rung fails closed** —
`_rung` catches exceptions and records them as failures, because with no human
review tier a rung that errors and reads as "didn't fail" silently removes a
check.

The gate always runs **from the live tree** against the worktree, so a
candidate that weakens the gate is judged by the old gate.

| Rung | Typical | Catches |
|---|---|---|
| preflight | ~0s | dirty tree, moved base, merge commits, out-of-scope paths |
| static | ~2s | syntax errors, **import failures**, new pyflakes findings |
| tests | ~30-40s | the full suite, plus a collected-count floor |
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

### 4.2 The collected-count floor

`pytest -q` exits 0 if a round deletes the test that was failing. Under
auto-landing that is not hypothetical, so the rung also asserts a minimum
collected count and refuses a diff that removes files under `tests/`.

### 4.3 The canary smoke asserts only what is deterministic

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

`scripts/selfmod/promote.py`, in this order:

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
3. `git merge --ff-only`, swap the venv if one was built.
4. Restart MCP then backend, and **verify `/health.commit` and `boot_id` both
   changed** — the only proof the restart picked up new code rather than a
   stale process still serving.

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

**Probe failures are classified by cause.** A refused connection means nothing
is listening: 3 ticks. A timeout means the socket accepted but the app was
busy: 24 ticks (2 minutes). See §9 for why this distinction cost a false
rollback to learn.

### 7.3 Three invariants

1. **`HEAD == last-known-good` ⇒ never roll back.** Everything on fire with
   nothing promoted is infrastructure, not a bad change.
2. **No promotion under observation ⇒ never roll back.** HEAD legitimately
   differs from LKG most of the time — a human commit, a nightly job.
   Reverting then destroys work nobody asked the guardian to judge.
3. **Unreachable supervisord restarts the unit**, never reverts code.

### 7.4 Rollback order

Stop the writers *before* moving the floor. The agent is what writes into this
repo, and `git reset --hard` during an `Edit` produces a half-applied revert —
strictly worse than either version. This is why both supervisor confs gained
`stopasgroup`/`killasgroup`: without them a Bash tool's child outlives the
stop.

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

---

## 10. What Lloyd may change

`scripts/selfmod/spec.py`. **Denied beats protected beats allowed**, and a run
spec cannot widen its own permissions by naming a denied path in
`writable_paths`.

- **denied** — `config.yaml` and `data/tool_overrides.yaml` (a round could
  disable Bash and Edit and lock itself out with no Python change at all — a
  soft brick no test would catch), `.env*`, `pytest.ini`, `.gitignore`,
  `web/**` (the gate does not build the frontend).
- **protected** — the gate itself, the guardian, the supervisor confs, and the
  health/restart path the rollback depends on. Allowed, but only with a
  passing drill.
- **allowed** — ordinary code.

`requirements*` is allowed *only because* rung 3 exists: a btrfs reflink clone
of the venv plus a `uv pip install` of the delta, booted by the canary.
`--reflink=always`, not `auto` — auto degrades to a real 6GB copy silently.

---

## 11. State

Everything under `~/.local/state/lloyd-selfmod/` — outside the repo, so
`git reset --hard` and `git clean -fdx` cannot reach it. `_pipeline/` would
not do: gitignored but still inside the tree.

| File | Written by | Purpose |
|---|---|---|
| `last_known_good.json` | guardian only | rollback target + floor |
| `current.json` | promoter only | the promotion under observation |
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
python -m scripts.selfmod.round status              # state + ledger + guardian
python -m scripts.selfmod.rehearse --yes-i-mean-it  # prove rollback still works
systemctl --user status lloyd-guardian
/usr/bin/python3 agent-services/guardian/guardian.py --selftest
journalctl --user -u lloyd-guardian -f
```

Recovering from a rollback: the reverted commit is on the
`guardian-broken-<ts>` tag and your uncommitted work is in the matching named
stash. `git merge --ff-only guardian-broken-<ts>` restores it (clear the SHA
from `denied.json` first).

`selfmod.enabled` in config.yaml is the master switch and defaults to
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
- **Frontend changes** are out of scope; `web/**` is denied.
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
