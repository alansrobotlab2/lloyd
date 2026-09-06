---
segment: architecture
tags: [architecture, lloyd, selfmod]
type: reference
status: implemented
date: 2026-09-06
---

# Self-modification with automatic rollback

Lets Lloyd change his own code and, if anything goes wrong, fall back to the
previous build without a human.

## Why this needed machinery

The git working tree **is** production. `conf.d/lloyd-backend.conf` runs
`server.py` out of `/home/alansrobotlab/lloyd` — no build step, no release
artifact. A saved file is a deploy, pending only a restart.

Worse, the repair mechanism lives inside the thing that breaks. The agent
loop, the tool dispatcher and the autonomy scheduler all run in one process
(`server.py` starts the autonomy ticker on its startup event). If
`lloyd-backend` goes FATAL, nothing is left alive to fix it. Break
`agent_mcp/main.py` instead and the backend survives with no tools — it can
see the problem and cannot touch it.

That is why the existing autoresearch loop mutates only SOUL.md, MEMORY.md
and USER.md: a bad prompt makes Lloyd dumb, not dead. Code needed two things
prompts never did — a canary that proves the build boots, and a separate
process that reverts it when it doesn't.

## Shape

```
 propose ──► WORKTREE ──► GATE (7 rungs) ──► PROMOTER ──► live tree
                              │                  │
                              └─ fail ───────────┤
                                                 ▼
                                          GUARDIAN (systemd, stdlib)
                                       liveness · errors · data · regression
                                                 │
                                                 └──► rollback to last-known-good
```

**The round is a wrapper, not a code generator.** `selfmod_start` hands back
a worktree path; Lloyd edits it with the ordinary Edit/Write/Bash tools and
commits. There is deliberately no `selfmod_write_code` tool.

## The gate

`scripts/selfmod/gate.py`. Cheapest first, short-circuiting, **every rung
fails closed** — a rung that raises is recorded as failed, because with no
human review tier a rung read as "didn't fail" silently removes a check.

| Rung | Cost | What it catches |
|---|---|---|
| preflight | ~1s | dirty tree, moved base, merge commits, out-of-scope paths |
| static | ~8s | syntax errors, **import failures**, new pyflakes findings |
| tests | ~35s | the full suite, plus a collected-count floor |
| venv | 3s–5m | only when `requirements*` changed |
| canary boot | ~30s | a build that will not start |
| canary smoke | ~30s | a build that starts but cannot dispatch a tool |
| drill | ~90s | only when the diff touches the rollback path |

Two rungs carry most of the weight.

**static** runs `python -c "import server, agent_mcp.main"`. An import-time
failure is the number-one way a self-modification bricks the boot, and this
catches it in seconds without binding a port.

**canary boot** starts both services from the worktree on 18080/18500 under a
throwaway supervisord and asserts the backend resolved MCP to *its own* port.
That is what proves config-follows-code held for this candidate: a round that
broke `service_url()` would leave the canary talking to the live aggregator
and everything else would still look green.

`pytest` cannot substitute for either. It imports modules in its own process
and never starts `server.py`, so it stays green through a broken startup
event, a port collision, or an aggregator that registers no tools.

### pyflakes is a diff, not a bar

The tree carries 69 pre-existing findings. An absolute rule would be disabled
within a day, so the rung compares the changed files against the same files at
the merge base and fails only on *new* findings.

### The collected-count floor

`pytest -q` exits 0 if a round deletes the test that was failing. Under
auto-landing that is not hypothetical, so the rung also asserts a minimum
collected count and refuses a diff that removes files under `tests/`.

## Isolation: one lever

The canary runs with `HOME=<round>/home` and its worktree at
`<round>/home/lloyd`. That makes `Path.home()/"lloyd"` and
`app.paths.LLOYD_HOME` the same directory, and `LLOYD_HOME.parent/"obsidian"`
and `$HOME/obsidian` the same directory. One lever neutralizes the sessions
dir, `autonomy-runs/`, the task registry, `workers.db`, the vault paths, and
`autonomy.py`'s `AUTONOMY_DIR` — which no config key reaches (`autonomy.task_dir`
in config.yaml is dead; nothing reads it), so an empty scratch vault is what
makes `recover_stuck_tasks()` a no-op.

A config overlay (`LLOYD_CONFIG_OVERLAY`) handles what escapes the filesystem:
ports, `workers.enabled: false`, and a bogus `LLOYD_SUPERVISOR_SOCK` so a
canary's startup hook physically cannot reach the live supervisord and stop
the live secondary vLLM.

`secondary_enabled` is deliberately **not** overridden — it must equal live so
that reconcile is a no-op by construction.

## Landing

`scripts/selfmod/promote.py`, in this order:

1. **Write the rollback point and read it back.** Nothing is mutated until
   `current.json` round-trips. This is the strict analogue of the xfail at
   `tests/test_autoresearch_promotion.py:362`: `snapshot_current_prompts`
   mkdirs unconditionally, never verifies the copy, and promotes anyway —
   which is why 26 of 83 historical promotions have no rollback point.
2. Idle-gate on `/health.turns`, then set a **TTL'd drain flag** so a turn
   arriving between the last quiet poll and the restart gets a 503 instead of
   being cancelled mid-flight. The TTL is mandatory: a promoter that dies here
   cannot wedge the endpoint.
3. `git merge --ff-only`, swap the venv if one was built.
4. Restart MCP then backend, and **verify `/health.commit` and `boot_id`
   actually changed** — the only proof the restart picked up new code.

Any failure between 3 and 4 reverts inline rather than waiting for the
guardian's next tick.

**The promoter never advances last-known-good.** The guardian does, after the
observation window. So LKG means *observed healthy in production*, and a
rollback always targets a commit that already survived a full window.

## The guardian

`agent-services/guardian/`. A **systemd `--user` unit**, not a supervisord
program, for three reasons:

- `agent-supervisord.service` sets `KillMode=control-group`, so every
  supervisord child dies when that unit restarts — exactly when a watchdog is
  needed.
- Its remediation set includes "restart supervisord". A child cannot restart
  its own supervisor and survive.
- supervisord parks a program in FATAL after `startretries` and never un-parks
  it; systemd `Restart=always` with `StartLimitIntervalSec=0` never gives up.

**Stdlib only, on `/usr/bin/python3`**, and it never imports from `app/`. Not
even `app.supervisor_client`, which has no third-party imports *today* — the
moment it grows an `app.config` import, a malformed `config.yaml` would take
down the watchdog along with the patient. The AF_UNIX XML-RPC transport is
copied, not imported, and says so.

### What guards the guardian

`ExecStartPre` stages the source into a pinned snapshot at
`~/.local/state/lloyd-guardian/bin/` **only if** it compiles and passes its own
selftest. `ExecStart` runs the snapshot. So a SyntaxError Lloyd writes into
the guardian degrades to a *stale* watchdog, never to *no* watchdog. The unit
invokes staging with a leading `-`, so even a crash there cannot prevent a
start.

Lloyd may edit the guardian. A diff that touches it triggers gate rung 6: the
drill boots a deliberately unbootable build under a canary, runs the
**candidate** guardian against it, and requires that the tree came back and
the service returned. A bound on *how* it changes, not a ban on changing it.

### The liveness predicate

Deliberately inverts `app/supervisor_client.py::_health`, whose comment says
"port being open is the strongest signal — trust it over supervisord state".
Right for the Services page; catastrophic for a watchdog, because a FATAL
backend whose port is still held by a zombie would read "healthy".
`tests/test_guardian_predicates.py` asserts the divergence side by side.

Supervisord state is consulted **first**, FATAL is decisive, and a probe may
only *add* failure. It also catches the crash loop that never reaches FATAL —
`autorestart=true` plus too small a `startsecs` retries forever while every
sample says RUNNING — by watching distinct spawn timestamps.

Three invariants:

1. **`HEAD == last-known-good` ⇒ never roll back.** Everything on fire with
   nothing promoted is infrastructure, not a bad change.
2. **Unreachable supervisord restarts the unit**, never reverts code.
3. **A degraded MCP is a warning**, unless it serves zero tools or a module
   that was healthy at LKG has newly broken. Thunderbird being closed must not
   revert anything.

### Rollback order

Stop the writers *before* moving the floor: the agent is what writes into this
repo, and `git reset --hard` during an `Edit` produces a half-applied revert.
This is why both supervisor confs gained `stopasgroup`/`killasgroup` — without
them a Bash tool's child outlives the stop.

Evidence is preserved before it is destroyed: HEAD is tagged
`guardian-broken-<ts>`, the dirty tree is stashed by name, and untracked
sources are copied aside. A rollback that erases the bug guarantees you fix it
twice.

`git clean` is **path-scoped**, never repo-root: the root holds `usage.db`,
`workers.db`, `.env` and `.venvs/`, all gitignored and none replaceable.

If rollback fails twice, the services are left **stopped** and `BROKEN` is
written. With no human in the loop, an honestly-dead system is safer than an
autonomous agent running half-reverted code.

## Detectors

| Trigger | Window | Source |
|---|---|---|
| crash / won't boot | ~2 min | supervisord state + `/health` |
| error-rate spike | ~15 min | `logs/server.err` by byte offset |
| data damage | ~15 min | KG row count, vault file count |
| behavioural regression | nightly | paired `eval/run_eval.py` |

**Errors come from `server.err`, not `server.log`.** `server.py`'s
`basicConfig` writes to stderr, so all application logs land in `.err`;
`server.log` is uvicorn's access log and contains zero error-shaped lines.

Signatures normalize digits, hex, paths and durations, so
`…1266 min old` and `…1300 min old` collapse to one. Anything seen in ≥3
distinct hours over 7 days is **chronic** and can never fire — production has
emitted `autonomy scheduler may be stalled` hourly for days, and a detector
that counted it would revert on its first tick.

**Data damage is the class `git reset` cannot undo.** The KG and the vault are
gitignored, so a change that deletes rows or notes boots fine, logs nothing,
passes every eval, and *survives* the revert. Two counts cover it.

### Behavioural regression: what was measured

The autoresearch composite is **not** used. Three identical baseline runs
scored 0.719 / 0.542 / 0.624 — spread 0.177 against a 0.05 threshold, and 61
of 83 historical promotions were decided inside that noise.

`eval/run_eval.py` has no LLM in it. Measured here, five consecutive runs
against an unchanged vault gave **identical** values for every quality metric
(stdev 0.0000); only latency moved. So the eval contributes no noise.

The real confound is **vault drift** — cross-day baselines differ because the
vault changed, not the code. So the check is a **paired same-data A/B**: the
last-known-good commit is checked out into a scratch worktree and pointed at
the *live* fact tree and KG via `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB`, and both arms
run in the same window. Drift cancels.

A missing noise file means "cannot evaluate", never "no regression".

## What Lloyd may change

`scripts/selfmod/spec.py`. **Denied beats protected beats allowed**, and a run
spec cannot widen its own permissions.

- **denied** — `config.yaml`, `data/tool_overrides.yaml` (a round could disable
  Bash and Edit and lock itself out with no Python change at all), `.env*`,
  `pytest.ini`, `.gitignore`, `web/**`.
- **protected** — the gate, the guardian, the supervisor confs, and the
  health/restart path. Allowed, but only with a passing drill.
- **allowed** — ordinary code.

## Operating it

```bash
# state, ledger, guardian heartbeat
python -m scripts.selfmod.round status

# a round by hand
python -m scripts.selfmod.round start "make X faster"
#   ...edit the worktree it printed, commit inside it...
python -m scripts.selfmod.round gate  SM_<id>
python -m scripts.selfmod.round land  SM_<id>

# prove the guardian still rescues a broken build
python -m scripts.selfmod.rehearse --yes-i-mean-it

# guardian
systemctl --user status lloyd-guardian
/usr/bin/python3 agent-services/guardian/guardian.py --selftest
```

State lives at `~/.local/state/lloyd-selfmod/` — outside the repo, so
`git reset --hard` and `git clean -fdx` cannot reach it.

`selfmod.enabled` in config.yaml is the master switch and defaults to
**false**.

## Known limits

- **A change that is correct, boots, and is quietly worse** in ways no eval
  measures. The guardian catches crashes, error spikes and data loss; taste is
  not automatable.
- **Frontend changes.** `lloyd-frontend` is Vite HMR; the gate does not build
  the web app, and `web/**` is denied.
- **`eval/baselines/` and the Thunderbird bridge are gitignored**, so they are
  absent from every worktree. The bridge contributes ~40 of ~124 tools, which
  is why tool-count assertions exclude external-application modules.
- **Post-landing detection fails open by construction.** The code is live and
  has already executed tool calls while you measure. Quality belongs in the
  gate, where it can be slow and fail closed.
