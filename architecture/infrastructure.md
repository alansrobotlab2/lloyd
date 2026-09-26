---
segment: architecture
tags: [architecture, lloyd, infrastructure, supervisord, systemd]
type: reference
status: implemented
date: 2026-09-14
---

# Infrastructure

Everything runs **directly on the host** under one supervisord, itself a
systemd `--user` unit. There is no container in the loop: the distrobox
setup this doc used to describe was retired with the move to the in-process
harness. `SETUP.md` is the rebuild authority (packages, venvs, secrets,
what to back up); this is the running shape.

## Host

| Property | Value |
|---|---|
| OS | Omarchy (Arch Linux), kernel 7.1 |
| CPU / RAM | 32 threads / 251 GiB |
| Production tree | `~/lloyd` — a saved file is a deploy |
| Development clone | `~/lloyd-sandbox` (remotes `origin` = GitHub, `live` = `~/lloyd`), lands as a PR |
| Automod worktrees | `~/lloyd-work/SM_<stamp>/`, cut from live `main` per round |
| Vault | `~/obsidian` (Obsidian Sync, headless via `agent-obsidian-sync`) |
| Secrets | `.env` (gitignored), expanded into `config.yaml` `${VAR}` placeholders at boot |
| Python venvs | `.venvs/lloyd` (backend, aggregator, voice worker), `vllm-flash-next-main` (primary, served), `vllm-qwen38-flash-next` and `vllm-qwen3.8` (alternate primary builds), `qwen3-tts` |
| State outside the repo | `~/.local/state/lloyd-automod/` (ledger, rounds, clusters), `~/.local/state/lloyd-guardian/` (pinned guardian snapshot), `~/.local/share/<channel>/` (youtube `seen.json`) |

## GPUs

`nvidia-smi` indices. `/dev/nvidiaN` does **not** match these numbers, and
neither does the CUDA runtime left to itself — without
`CUDA_DEVICE_ORDER=PCI_BUS_ID` it orders devices by capability, which is why
every launcher that sets it calls it mandatory.

Every program that touches a GPU does set it, but **not in one place**, and
this doc claimed otherwise until 2026-09-11 ("in every program's
environment") — worth correcting rather than dropping, because the wrong
half is exactly the half you would check. Four programs set it in the
supervisord `environment=`: `agent-tts`, `lloyd-agent-worker`,
`agent-qmd-daemon`, `agent-qmd-watcher`. The three engine launchers export it
themselves instead — `start-qwen38-flash-next.sh:558` (with
`CUDA_VISIBLE_DEVICES=1`), `start-secondary.sh` (taking the conf's
`GPU="2"`; before either exec it reads the card through
`agent-services/bin/gpu-mem.sh` — see the preflight in `architecture/djev.md`) and
`start-qwen3-tts.sh:23`. **Neither LLM program names it in
`environment=` at all**, so a `supervisorctl` dump of the primary's
environment shows no pin and the pin is nonetheless there. `agent-tts` is the
one pinned twice, in both places, to the same device.

Read the launcher, not only its header: until 2026-09-24 (#1447)
`start-qwen3-tts.sh` said "(GPU 1)" on line 4 while exporting
`CUDA_VISIBLE_DEVICES=0` twenty lines down. The export was the correct half —
GPU 0 is the RTX 3090 this table puts TTS on — and the header now says so;
`tests/test_gpu_placement_comments.py` holds the two in step.

| GPU | Hardware | VRAM | Serves |
|---|---|---|---|
| 0 | RTX 3090 | 24 GB | Qwen3-TTS (`agent-tts`), the LiveKit voice worker, qmd embeddings (`agent-qmd-daemon`, `agent-qmd-watcher`) |
| 1 | RTX PRO 6000 Blackwell | 96 GB | the primary LLM, vLLM (`agent-llm-primary`) |
| 2 | RTX 3090 | 24 GB | djev, DiffusionGemma on vLLM (`agent-djev`) at `GPU_UTIL=0.97` since 2026-09-20 — or the secondary LLM, llama.cpp (`agent-llm-secondary`, ~21.7 GB) if that slot is re-armed; never both, see [[djev]] |

Board power limits are clamped at boot by the system unit
`nvidia-power-limit.service` (Xid 79 mitigation) — see [[vllm]] §2, which
absorbed that page on 2026-09-11.

## Model slots

A slot is an *endpoint* in `config.yaml` (`models.<alias>`). Which model
actually answers there is decided somewhere else entirely — the supervisord
program's `environment=` and its start script — and those places drift. On
2026-09-06 an automod rollback reverted the secondary's launcher and put a 4B
under the 35B's alias and port. Every slot therefore carries an identity
check, and those checks are the only reason this table can be trusted.

| Alias | Port | Engine | Model | Program |
|---|---|---|---|---|
| `primary` | 8096 | vLLM, venv `vllm-flash-next-main` | Qwen3.8-Flash-Next NVFP4, FP8 KV | `agent-llm-primary` |
| `secondary` | 8091 | llama.cpp `llama-server`, `--parallel 1` | Qwen3.6-35B-A3B UD-Q3_K_XL | `agent-llm-secondary` |
| — | 8090 | Qwen3-TTS (`.venvs/qwen3-tts`) | cloned voice `clone:dave_cullen` | `agent-tts` |

The primary's own configuration, tuning and benchmarks are [[vllm]]; what
follows is what holds for the slots as a set.

### The secondary is single-tenant by design

llama.cpp divides `--ctx-size` across slots and the full 262,144 window was
the point, so `--parallel 1`. Everything on the secondary queues: session
titles (`app/session_titles.py`, geometric schedule), post-session and voice
summaries (`app/secondary_models.py`), the cluster pair-judge, and any agent
turn routed there.

It also publishes a different Prometheus vocabulary.
`vllm_metrics._translate_llamacpp` renames it into vLLM's so one dashboard
card serves both, and three special cases hang off the same `is_llamacpp`
flag: KV occupancy and TTFT are reported **`None`, not `0`** — it publishes
neither, and a `0.0` renders a full cache as an empty one; a reachable
llama.cpp server is **awake** by definition, since it has no sleep-state gauge
and falling through to the vLLM check renders a healthy engine "asleep"
forever; and its model name comes from a `/props` probe cached per engine
lifetime, because it labels none of its metrics and the loaded GGUF cannot
change while the process lives.

### `secondary_enabled` outranks supervisord

`server.py::_sync_secondary_llm_state` reconciles the program against that
flag on every backend boot, so the conf's own `autostart=true` is not the
deciding vote — leaving the flag false silently stopped the secondary three
seconds after each backend restart. It reroutes the *callers* too:
`config.resolve_model_alias` rewrites `secondary` to `primary` while the flag
is false, logged once per name because it is otherwise undetectable. Inner
Voice's config said `model: secondary` from 2026-05-07 and ran on the primary
the whole time for exactly that reason; when `de893d7` flipped the flag on,
the observer silently moved to what the slot then held — a 4B — and nothing in
any log said so. Both paths stand down when `services.sync_secondary_llm` is
false, which is what stops an automod canary booted from a worktree from
reconciling the live engines against its own config.

### Identity and health

`models.<alias>.expect_model` is a case-insensitive substring checked against
vLLM `/v1/models` (`root`, `id`) or llama.cpp `/props` (`model_path`,
`model_alias`) by `app/model_identity.py` — at boot (detached, 6 attempts 15 s
apart) and on `GET /api/models/identity?refresh=1`. It retries only while
something is `unreachable`: the 35B takes minutes to page 17 GB onto a 3090,
while a `MISMATCH` is conclusive on the first look and waiting would only
delay the alarm. The KV cache carries its own second verdict beside it (`ok` /
`REGRESSION` / `unknown` / `unreachable`), because the right model can still be
served the wrong way — [[vllm]] §3.1. A slot with no `expect_model` reads
`unchecked`. A third, `image_status`, asks a slot whose `supports_vision` is
literally `true` to take one small image, because that flag outlives a boot
that dropped `LANGUAGE_MODEL_ONLY=0` (#1420, [[desktop]]).

**Update `expect_model` whenever a slot's occupant changes.** The sweep only
ever reports; restarting a slot is precisely the operation that would have
swapped the model in the first place.

Three venvs can serve the primary's model family, which is why `VLLM_VENV` is
pinned in the conf rather than left to the launcher: `vllm-flash-next-main`
(what the program names, and the only one with the fp8 QSA path),
`vllm-qwen38-flash-next` (the PLE-offload-worker build — and the launcher's
*own* fallback default — but a dropped `VLLM_VENV` no longer reaches BF16:
the conf still exports `KV_CACHE_DTYPE=fp8`, that build's QSA kernel has no
fp8 path, and the guard at `:526-531` exits 1. It boots BF16 only if both
variables go), and
`vllm-qwen3.8`, which belongs to the 27B revert target at
`start-qwen3.8-27b-nvfp4.sh`. `SETUP.md` Part 4 (venvs) and Part 9 (LLM
models) carry the builds and the weights.

## Process management

`agent-supervisord.service` (systemd `--user`, `KillMode=control-group`,
`ExecStartPre=cleanup-orphans.sh`) runs `~/.local/bin/supervisord` (a uv
tool, symlinked into `~/.local/share/uv/tools/supervisor/bin/`) with
`agent-services/supervisor/supervisord.conf` + `conf.d/`. Its control socket
is `/tmp/agent-supervisor.sock`; `LLOYD_SUPERVISOR_SOCK` points a caller
somewhere else, which is how a gate's canary drives its own supervisord
without touching the live one.

Because the kill mode is the whole control group, an OOM of one program takes
every program with it — not a theory: on 2026-09-08 an A/B sweep restarted
the primary twice before the kernel had reclaimed the previous engine's
95 GiB host-RAM table, and `systemd-oomd` killed the **unit**, 953 processes
the first time and 793 the second, at a peak RSS of 230.3 GiB. Everything
came back on its own and the arm under test was lost. That is why the
guardian is a separate unit ([[vllm]] §8 has the cadence rule).

A fourth kill, 2026-09-15 23:52:46Z, came from a *single* boot that went
through a memory floor: the first `round restart --only agent-llm-primary`
waited for `MemAvailable` to pass 150 GiB after the stop and started the
engine, and the boot's own transient took the slice to pressure. The
difference from the earlier kills was the desktop: a 16 GiB
`qemu-system-x86_64` VM (`macos-tahoe`) was running, so the box read only
57 GiB available with the engine up. The unit restarted on its own 30 s
later, every program bounced once, and the engine booted from the new
conf. The leg's floor is 180 GiB now, with a hard refusal under 150 —
memwatch snapshot `20260915_235555`.

It happened a third time on 2026-09-15 at 04:48:34Z (796 processes, 174.6 GiB,
no restart in progress), and nothing could say what had grown. The unit is
the one killed whoever causes the pressure: oomd watches `app.slice`
(`/usr/lib/systemd/user/app.slice.d/10-oomd.conf`), past 50% full pressure for
20 s (`/etc/systemd/oomd.conf.d/10-omarchy.conf`), and kills the descendant
reclaiming most — and the desktop's browser, editor and VMs share that slice.
So the guardian records evidence: `agent-services/guardian/memwatch.py` reads
the unit's, the slice's and the host's `memory.pressure` every tick, and once
any reaches 15% full avg10 writes a snapshot at most every 15 s to
`~/.local/state/lloyd-guardian/mem-pressure/` (newest 120) — the three
readings, `/proc/meminfo`, the unit's `memory.stat`, and the top 30 processes
by RSS with their anon/shmem split and cgroup. The 95 GiB n-gram table is
shmem that no process's RSS shows, which is why the cgroup and meminfo
figures ride along. `memwatch.py latest` prints the newest. A quiet tick
costs ~0.1 ms; a snapshot ~45 ms and 14 KB.

**A fifth kill, 2026-09-17 00:09:25Z (716 processes), is the one that produced
a bound rather than another floor.** An A/B sweep's arm ran the launcher after
its `MemAvailable` check passed at **198 GiB**; the unit was killed 129 seconds
later. memwatch caught the whole build-up, and it says the floor was never
capable of preventing it: eleven seconds before the kill the unit's own cgroup
held **226.1 GiB** of the box's 251 — `file` 201.3 GiB (the 170 GiB checkpoint
being read, charged to the cgroup that reads it), of which `shmem` 128.9 GiB
(the UVA PLE path maps the table *shared*, so it can only go to swap), `anon`
5.0 GiB — while `MemAvailable` had fallen from 198 to 79.5 GiB and unit full
pressure had gone 31.1% → 69.1% in 26 s, peaking at 81.25% against oomd's 50%
trigger. **One boot is sufficient on its own**; the 09-15 kill's qemu VM was a
contributor, not the mechanism, and reading it as the mechanism is what put two
more `MemAvailable` floors in the tree. The gauge cannot work here — the boot
consumes precisely what it measures.

**The fix is the slice, and the first fix was wrong.** The same evening
`agent-supervisord.service` got `MemoryHigh` (170G, then 150G) and
`ManagedOOMPreference=avoid`, still inside `app.slice`. It protected the
services and made every boot a desktop outage. Throttling at the cap is itself
memory pressure; it counted in `app.slice`, and `avoid` told oomd to take
something else — Chrome at 17:21:47, VS Code (501 processes) at 17:22:03, then
terminals, Thunderbird and voxtype. At 150G the cap was below a cold boot's
resident need as well, so the 17:46 boot livelocked in the throttle for 25
minutes while oomd killed something every ~15 s, the guardian included. The
journal from 09-08 to 17:09 on 09-17 holds five oomd kills, all of this unit;
from 17:21 to 18:14 it holds about sixty, none of it.

What holds is `Slice=lloyd.slice` on the unit (tracked at
`agent-services/systemd/`; the `~/.config/systemd/user/` entry is a symlink).
oomd monitors only `app.slice` (`oomctl dump` lists that one path), so the
stack's pressure is no longer the desktop's and oomd can choose neither. The
kernel OOM killer remains the backstop for real exhaustion and picks by size —
the engine. There is no `MemoryHigh` on the unit, deliberately: one that is
not measured against a cold boot's peak livelocks the boot, and inside
`app.slice` any cap at all converts the stack's pressure into the desktop's.
`memwatch.py` finds the unit in whichever slice it lives and reads `app.slice`
by name, because that is still the slice whose pressure kills.

| Program | Port | What | Start script |
|---|---|---|---|
| `lloyd-mc:lloyd-backend` | 8080 | FastAPI + SSE, the worker pool, the autonomy scheduler | `server.py` |
| `lloyd-mc:lloyd-frontend` | 5173 | Vite dev server (HTTPS). It is the front door, not the thing behind one: it proxies `/api` to the backend and `/livekit` to the SFU | `npm --prefix web run dev` |
| `lloyd-mc:lloyd-mcp` | 8500 | the MCP aggregator `Server("lloyd")`, Streamable HTTP at `/mcp`; `/state`, `/health`, `/changes` | `python -m agent_mcp.main` |
| `agent-llm-primary` | 8096 | vLLM, Qwen3.8-Flash-Next, FP8 KV | `bin/start-qwen38-flash-next.sh` |
| `agent-llm-secondary` | 8091 | llama-server, Qwen3.6-35B-A3B | `bin/start-secondary.sh` |
| `agent-tts` | 8090 | Qwen3-TTS API | `bin/start-qwen3-tts.sh` |
| `agent-livekit-server` | 7880 | LiveKit SFU; signal/WS bind `0.0.0.0` (`conf/livekit.yaml` `bind_addresses`) and it is the **advertised ICE address** (`rtc.node_ip`) that resolves to the Tailscale address at boot | `bin/start-livekit-server.sh` |
| `lloyd-agent-worker` | 8501 | the LiveKit voice agent: STT, TTS shaping, wake word (8501 is the loopback wake-miss diagnostic rig) | `agent-services/livekit_worker.py` |
| `agent-qmd-daemon` | 8181 | qmd vector search over the vault (the fork in `~/lloyd/qmd`) | `node qmd/dist/cli/qmd.js mcp --http` |
| `agent-qmd-watcher` | — | re-embeds on vault change | `scripts/qmd-watcher.sh` |
| `agent-obsidian-sync` | — | headless Obsidian Sync | `bin/start-obsidian-sync.sh` |

The process group is `lloyd-mc`; address the three as
`lloyd-mc:lloyd-backend` etc. — supervisord's XML-RPC name resolution
requires the qualified form, and `app/supervisor_client.py` derives the group
from supervisord's own `group` field rather than hardcoding it.

**Restart the backend, the aggregator or the primary engine through
`python -m scripts.automod.round restart`** (`--only <program>` for one leg;
`agent-llm-primary` since 2026-09-15, with a host-RAM wait, a conf reread
and a 20-minute boot wait of its own — [[automod]] §12), which
takes the guardian's pause lease, pauses the worker pool, drains the backend
and waits for it to go idle, restarts `lloyd-mcp` then `lloyd-backend` with a
health wait per leg, and releases all three. A bare `supervisorctl restart`
is indistinguishable from a crash to the guardian: on 2026-09-09 four
deliberate restarts each fired "Service down, but no promotion to revert"
through every channel — ledger, ALERT.md, journal, toast, voice, vault note —
and each one killed whatever worker job was mid-flight, whose connection
errors then landed in someone's observation window. It refuses while a
promotion is under observation (`force=True` overrides, because the guardian
is judging that build) and records a `restart` row on the ledger with the
reason. supervisorctl is still the tool for everything else, and is what that
command wraps. Never restart the primary twice in quick succession
([[vllm]] §8).

A leg it stopped is always started again. Ctrl-C between the stop and the
start is held until the program is back (2026-09-24, when one interrupt left
the backend STOPPED), any failure is a `restart_failed` ledger row naming what
was `left_stopped`, and the idle wait refuses at once on a backend supervisord
holds stopped (start it, or `--skip-idle`) instead of waiting out its budget.
The `agent-llm-primary` leg (moved from CLAUDE.md, 2026-09-25): it stops the
engine, waits on the host-RAM floor above for the previous boot's 95.37 GiB
BF16 n-gram table to be released, runs `supervisorctl reread` and `update` so
an edited `environment=` in `agent-llm-primary.conf` is picked up, starts it,
and waits up to 20 minutes for `/health` while refreshing the lease. Every
turn in flight dies with the engine and is re-offered as `infra`, so do it
once, with the reason on the ledger. The `MemAvailable` floors cannot prevent
a kill — a boot consumes what they measure — they only confirm the last
engine's table was released.

### `startsecs` is a tier, not a formality

supervisord marks a program RUNNING once `startsecs` elapses, and after that
a death is an *unexpected exit* — which `autorestart=true` retries forever,
never reaching FATAL. A threshold shorter than the real boot therefore turns
a boot failure into a silent crash loop that looks RUNNING most times you
sample it. Three programs are tiered against their own boots:

- `agent-llm-primary` **900 s**. A warm boot is 240–265 s, but a venv change
  empties the torch.compile / Triton cache and the first boot after one took
  775 s to health (measured 2026-09-10). At 300 s supervisord would have
  called that a crash and torn down a healthy-but-slow engine three times.
- `lloyd-backend` **25 s**. It binds its socket well before the startup hooks
  finish — the autonomy ticker, the worker pool, the file watcher — so at 5 s
  a failure at t=8 s read as an unexpected exit.
- `lloyd-mcp` **15 s**, which covers importing every tool module. It was
  unset (supervisord's default: 1 s) until an aggregator dying during import
  retried forever instead of parking in FATAL.

The cost is that `supervisorctl start` blocks for that long;
`supervisor_client.restart_process()` polls instead of blocking.

`stopasgroup`/`killasgroup` are set on `lloyd-backend` and `lloyd-mcp` (not
the frontend), and are load-bearing for rollback rather than tidiness:
without them a Bash tool's child outlives its parent's stop and can still be
writing into `~/lloyd` while the guardian runs `git reset --hard`. The
aggregator has the same problem one layer out — it spawns a headful Chromium
and a Node bridge for Thunderbird, and every restart used to orphan one of
each. Stop the hands before moving the floor.

## Outside supervisord

| Unit | Kind | What |
|---|---|---|
| `lloyd-guardian.service` | user service, `WatchdogSec=90` | the automod rollback watchdog over `lloyd-mc:lloyd-backend` and `lloyd-mc:lloyd-mcp` (not the frontend): stdlib-only, runs `/usr/bin/python3` from a pinned snapshot staged by `guardian-stage.sh`; deliberately not under supervisord ([[automod]]) |
| `lloyd-guardian-nag.timer` | user timer, every 15 min | re-announces an unresolved BROKEN state through `nag.py` → the same `Notifier` |
| `lloyd-qmd-cleanup.timer` | user timer, 04:45 | qmd orphaned-vector cleanup |
| `lloyd-graph-backup.timer` | user timer, 05:30 | `scripts/backup/backup-graph.sh` — the knowledge-graph store |
| `lloyd-data-snapshot.timer` | user timer, hourly | `scripts/backup/snapshot-data.sh` — read-only btrfs snapshot of `~/lloyd-data` into `~/.lloyd-data-snapshots` ([[data-home]]) |
| `lloyd-data-snapshot-prune.timer` | **system** timer, daily, root | `/usr/local/sbin/prune-data-snapshots.sh` — 48 hourly + 14 daily. Root because the user cannot delete a read-only snapshot, which is the point; installed by hand (`SETUP.md`), so it lives in `agent-services/systemd/system/` and is not symlinked |
| `thunderbird.service` | user service | Thunderbird itself, hosting the `thunderbird-mcp` extension and its bridge on `:8765` |
| `nvidia-power-limit.service` | system service, root scope | GPU power clamp, via `/usr/local/sbin/set-gpu-power-limit.sh`. Declares 275 / **450** / 275 W. GPU 1, the Xid-79 card, was raised to 450 W on 2026-09-17 (#1107). It had been set to 450 W by hand twice before that, and each boot reset it to the 400 W the unit declared at the time. A hand-set `-pl` still lasts only until the next boot, so compare the unit's `Environment=` against `nvidia-smi` |

Unit files live in `agent-services/systemd/` and `install-services.sh:36`
**symlinks** every `.service` and `.timer` in that directory into
`~/.config/systemd/user/`, so editing the repo copy is the deploy. `thunderbird.service`
and `voxtype.service` are tracked here and linked like every other unit (#1109,
fixed) — both are `WantedBy=graphical-session.target` (Thunderbird hosts the
`thunderbird-mcp` extension its 40 tools bridge over; voxtype is push-to-talk),
so `enable` runs them with the desktop, not via supervisord. Two units are still
gaps rather than design: the groundskeeper and graph-backup timers exist only as
plain untracked files under `~/.config/systemd/user/`, so a rebuild from
`SETUP.md` loses them. The root-only `nvidia-power-limit.service` is skipped
by that glob (`SYSTEM_SCOPE_UNITS` in `install-services.sh`, #1108) — it used
to be linked into the user manager, where it can never clamp anything; the copy
that runs is a hand install at `/etc/systemd/system/`, and
`scripts/service_health_check.py` (`check_deployed_copies`, category `deploy`,
in the default run) compares it and `/usr/local/sbin/set-gpu-power-limit.sh`
byte for byte against the tracked pair: `drift` or `missing`, never `ok` for
a copy it could not read (`tests/test_service_health_check_gpu_power_drift.py`).
One tracked unit used to sit outside the directory —
`agent-services/autonomy.service`, a dead Idler heartbeat whose
`WorkingDirectory` and venv had both been deleted for a year; it was never
installed by anything and was removed on 2026-09-24 (#1110).

Thunderbird runs as a user service because `agent_mcp/thunderbird.py` talks
to a live instance; a closed Thunderbird is the usual reason the aggregator
reports `degraded`.

## Alerts: one fan-out, six channels

`agent-services/guardian/notify.py` is the **only** producer of user-facing
alerts, and that is a property rather than an accident.
`lloyd-guardian-nag.service` used to run its own inline `notify-send` — a
second, private definition of "tell the human", structurally incapable of
gaining any channel this module grew, so the day speech was added it would
have been the one alert that stayed silent. It calls `nag.py` now, through
the same `Notifier`. Anything that wants to announce something goes here.

Two entry points, and picking the wrong one is the trap:

- **`alert()`** — an incident, fanned out to all six: the ledger
  (`promotions.jsonl`), `ALERT.md` in the guardian state dir, the journal
  (`systemd-cat -t lloyd-guardian`), a desktop toast, the spoken line, a
  vault note under `~/obsidian/memory/<date>.md`, plus a backlog task when
  the level is critical or a trigger is named. They fail differently on
  purpose: the ledger works when the network is down, the journal when the
  state dir is unreadable, the toast when nobody is looking at a browser,
  speech when nobody is looking at the *screen*, and the vault note tomorrow
  morning. The backlog task is the one that closes the loop — after a
  rollback Lloyd wakes on known-good code with a work item naming what was
  reverted and the `git cherry-pick` that restores it.
- **`announce()`** — news, with no bookkeeping: journal, toast and voice
  only. A successful promotion uses it, and so does the 15-minute nag. The
  nag is why `announce` takes a `level`: the state really is critical and
  should look it, but it has **already** been recorded, and re-announcing
  through `alert` would append a ledger row and file a fresh backlog task
  every 15 minutes, burying the task the rollback filed under copies of
  itself.

Promotion *success* is announced too. Before that, every notify-send in the
tree hung off a guardian alert, so a loop that rewrites the running system in
the background was silent whenever it worked — backwards, since the
successful landings are the ones nobody is watching a terminal for. Three
things outside the guardian reach the same fan-out by importing the pinned
snapshot: `scripts/automod/promote.py` (landings),
`workers/sources/autocode.py` (an item that needs a human) and
`app/prefix_miss.py` (a turn that re-prefilled past its threshold).

Two suppression windows, for two different costs.
`policy.ALERT_REPEAT_SECONDS` is 900 s and lives in memory
(`guardian.py::_alert_seen`); `policy.VOICE_REPEAT_SECONDS` is 3600 s and
lives **on disk**, because the two producers are different processes — the
daemon and the nag oneshot — and an in-memory dedupe cannot see the nag. A
toast you have already seen costs a glance; a sentence you have already heard
costs the whole sentence, and at 900 s an unresolved incident would say the
same thing aloud four times an hour indefinitely.

`external=False` leaves only the two channels scoped to the guardian's own
state dir, so the rollback drill cannot file real backlog tasks or write the
live daily note. It is not the only mute it needs: the toast and the journal
are reachable from *any* process with a session bus, and on 2026-09-07 every
gate run — each one executes the full suite — put fixture strings like
`Lloyd guardian: real rollback` on the screen and wrote `STILL BROKEN` to the
live journal at priority 2. `LLOYD_JOURNAL_ALERTS=0` and
`LLOYD_DESKTOP_ALERTS=0` close that, checked at dispatch time.

### The spoken channel

`speak.py` says the alert aloud in the same cloned voice as voice mode. It
lives in `agent-services/guardian/` because `guardian-stage.sh` stages
`guardian/*.py` and nothing else — a module the guardian imports must be in
that directory or it will not exist in the pinned snapshot.

- **It reports *dispatched*, not *heard*.** The unit watchdogs the loop at
  `WatchdogSec=90` against a 5 s tick, so synthesis and playback happen in a
  detached child and `alert()` returns in milliseconds. What actually came
  out of the speaker is in `voice.log` in the guardian state dir.
- **The child runs the venv python, and that is deliberate.** The
  stdlib-only rule exists so the watchdog cannot be taken down by what it
  watches; this child runs after the ledger, ALERT.md, the journal and the
  toast have already fired, nothing waits on its exit, and its failure cannot
  reach the loop — so spending the venv buys the presence EQ (scipy) at no
  cost to the property the rule protects. A wrecked venv costs a duller
  voice, never an alert. Shaping degrades in tiers — the EQ needs scipy, the
  WSOLA speed only numpy, so a system-python fallback still fixes the pace —
  and logs which tier ran, because a silent downgrade is indistinguishable
  from success. The first cut of this module called `OutputShaper.enabled()`
  — it is a `@property` — and shipped *unshaped* audio while looking
  perfectly healthy.
- **Quiet hours gate the clock, and only the sound.**
  `guardian.voice.quiet_hours` in config.yaml (enabled, 23→07,
  `allow_critical: false`) withholds speech; the toast, journal, ledger,
  vault note and backlog task all still fire, so nothing is lost — it is
  waiting in the morning. That is what makes it safe to default on. The
  window is checked **before** the repeat suppression, which records as it
  decides (`should_speak`): recording a quiet-hours drop would spend the
  hourly slot on an utterance nobody heard, and the 08:00 repeat of an 03:00
  alert would then stay silent for the wrong reason. `allow_critical: true`
  lets a rollback wake you anyway. A window that wraps midnight is the normal
  shape, and `start == end` means *no* window rather than a full day of
  silence. Only alerts are gated by the hour. The key sits under `guardian.voice` and **not**
  `livekit.tts`, and the split is load-bearing — `livekit_worker` reads the
  latter, and a voice conversation that went mute at 23:00 because an alert
  policy leaked into it would be a real bug.
- **`LLOYD_VOICE_ALERTS=0`** keeps every other channel and drops only speech.
  `tests/conftest.py` sets it for every test — otherwise `pytest tests/`
  talks to the room from a process that outlives the test.
- **Voice sits below the `external` gate**, like the vault note and the
  backlog task: the drill runs a real guardian against a throwaway repo, and
  a rehearsal that announces a rollback out loud is indistinguishable from a
  production incident to anyone in the room.

`config.yaml`'s `livekit.tts` stays the single source for the voice.
`agent-services/bin/sync-voice-config.py` pushes it into `voice.json` in the
guardian state dir at stage time, because the guardian has no yaml and must
not read the repo on a critical path. If it never runs, `speak.py`'s built-in
defaults still sound right — the sync only stops the two drifting after a
voice *change*. So a voice change needs the guardian re-staged
(`systemctl --user restart lloyd-guardian`) as well as the worker restarted
([[voice]]).

## Ports

| Port | Owner |
|---|---|
| 5173 | Vite (TLS) |
| 7880 / 7881 | LiveKit — signal HTTP and its TCP fallback (`conf/livekit.yaml`: `bind_addresses: [0.0.0.0]`, `rtc.tcp_port: 7881`) |
| 8080 | backend |
| 8090 | Qwen3-TTS |
| 8091 | secondary LLM |
| 8096 | primary LLM |
| 8181 | qmd — listening on `[::1]` only, so an IPv4-literal probe exits 7 against a healthy daemon (`scripts/service_health_check.py:51`) |
| 8500 | lloyd-mcp |
| 8765 | the Thunderbird extension's bridge |
| 18080 / 18500 | the automod **canary**: a candidate backend and aggregator booted from the round's worktree by the gate |

## Remote access

Vite terminates TLS with the local CA in `agent-services/cert/`:
`scripts/gen-cert.sh` creates the CA and server cert once (`--force`
regenerates and so invalidates every client cert signed by the old CA), and
`scripts/mint-client-cert.sh <device>` mints a per-device cert and records
its sha256 in `clients.json`. If a publicly-trusted Tailscale cert has been
provisioned for the MagicDNS name (`tailscale cert <name>`) Vite prefers it,
because it is trusted by every device with no CA to install; today none is on
disk, so the private server cert is what is served.

**mTLS itself was dropped on 2026-06-14.** iOS Chrome and every other
third-party iOS browser cannot present keychain identities for mutual TLS —
only Safari can — so Vite no longer requests or requires a client cert, and any
browser on the tailnet works. Tailscale is the front door rather than the
certificate, and since 2026-09-20 the backend *checks* that instead of assuming
it: `server.py::ApiPeerGate` refuses `/api/*` with 403 unless the
request's ASGI peer is loopback or inside `server.trusted_networks`
(`_trusted_networks`, default Tailscale's CGNAT range `100.64.0.0/10`), which is
the address a tailnet browser presents after Vite's `xfwd` rewrite. The peer is
the only evidence, and no header is consulted: uvicorn runs `proxy_headers` with
`forwarded_allow_ips` defaulting to loopback and Vite proxies `/api` with
`xfwd: true`, so a client-supplied `X-Forwarded-For` *rewrites* `request.client`
rather than proving anything (uvicorn scans that list right-to-left for the
first address it does not trust, so a client cannot prepend a trusted hop).
`/health` and `/health/deep` are the entire
pre-auth surface, as exact paths (`PRE_AUTH_PATHS`), because a watchdog has to
be able to ask "is it alive" from a peer the gate refuses; a watchdog asking
`/api/health` gets the gate, not a 404. A CORS preflight is let through from any
peer — a browser sends one with no custom headers, so refusing it breaks a
trusted client and proves nothing — but only a real preflight, recognised by
`Access-Control-Request-Method` the same way starlette's own CORS middleware
recognises one (`middlewares/cors.py:91`). A bare `OPTIONS`, which no browser
sends, is gated: passed through it reaches the router and its 405-vs-404 answer
would tell an untrusted peer which API paths exist.

`ApiPeerGate` is a plain ASGI middleware rather than
`@app.middleware("http")`, and that is load-bearing: `/api/*` is not HTTP-only.
`app/routers/lsp.py` serves `@router.websocket("/api/lsp/{language}")`, and each
accepted connection spawns a language-server subprocess rooted at a
caller-supplied `workspace`. The `http` decorator wraps its handler in
starlette's `BaseHTTPMiddleware`, whose `__call__` returns before dispatching
any scope that is not `http` (`starlette/middleware/base.py:101-104`), so an
HTTP-only gate leaves that upgrade path entirely unenforced — measured on #683's
first commit, with the HTTP gate in place a LAN peer's LSP socket was accepted
and reached the spawn while the same peer's `GET /api/sessions` was refused. The
gate now decides on `scope["client"]` for both `http` and `websocket` scopes and
answers a refused upgrade by closing before accepting
(`REFUSAL_CLOSE_CODE`), which is the ASGI form of a 403 handshake.

What is **not** closed: `config.yaml` still binds the backend to
`server.host: 0.0.0.0`, so the port is reachable from anywhere routable — what
arrives there now meets a 403 instead of session data. Binding to `127.0.0.1`
outright, which would make Vite's stated trust model true by construction since
it proxies `/api` to `http://localhost:8080`, is #683's open half and needs a
person: `config.yaml` is boot-read-only and tracked. A LAN-only client would
need `server.trusted_networks` widened to the LAN subnet, which is a number only
a person should choose.

The allowlist did not go away either, and the order matters when reading
`server.py::ApiPeerGate`. The network rule runs first; then, for a peer
already inside it, Vite's injected `x-client-fingerprint` (the peer cert's
sha256, with `x-client-cn` its CN) is checked against `clients.json`, re-read per
request so a revocation takes effect without a restart, and an unknown or revoked
fingerprint is refused. A fingerprint is not proof of possession — it is
copyable out of `clients.json` — so it can take access away but cannot buy
network reach an untrusted peer does not already have. Loopback
(`127.0.0.1`, `::1`) is trusted outright, because same-host callers — the
LiveKit worker, the autonomy ticker, every worker source, the self-mod promoter
— POST straight to `:8080` and never cross Vite's TLS layer. The self-mod drain
keeps its own loopback-only guard on top of all this
(`app/routers/automod.py::_is_loopback`), so arming the drain from a browser tab
stays impossible even from the tailnet.
`tests/test_api_client_gating.py` is the differential that pins every sentence
above.

LiveKit advertises the Tailscale address when Tailscale is up and falls back
to the default route. That resolution happens at boot and is never hardcoded:
LiveKit binds its UDP media ports (50000–50100) to `node_ip`, so a stale
address makes the bind fail silently and every ICE negotiation dies in
`wait_pc_connection timed out` before audio reaches the agent — which is
exactly what the tailnet reassigning the address did at the 08-22 migration.

## Scheduled work

There is no cron. Everything unasked runs through the worker pool in the
backend ([[workers]]): the autonomy fleet in `~/obsidian/autonomy/*.md`
([[autonomy]] for the mechanism, [[autonomy-jobs]] for what each job is for), backlog
triage and implementation ([[automod]]), research, digests and session
mining. The systemd timers above are the only wall-clock schedules.

## Review log

- **2026-09-20 — current for the access-boundary section.** #683 closed the
  hole the previous entry recorded as open, so the prose around it is rewritten
  instead of left describing a defect that no longer exists. Re-read, not
  inherited: `ApiPeerGate` — the gate is a plain-ASGI class since round 3, so
  the `@app.middleware("http")` handler `_require_client_cert` it replaced is
  gone from the tree — with `_cert_fingerprint`, `_is_trusted_peer`,
  `_trusted_networks`, `PRE_AUTH_PATHS` and `REFUSAL_CLOSE_CODE` in `server.py`;
  uvicorn 0.44.0 defaults
  `proxy_headers=True` with `forwarded_allow_ips="127.0.0.1"`, which is how
  Vite's `xfwd: true` reaches `request.client`; `config.yaml` still
  `server.host: 0.0.0.0`, so the bind half of #683 stays open and says so above.
  Every listener row and port table below is untouched by this round and still
  carries the 2026-09-14 date.

- **2026-09-14 — stale.** Checked every path, port, cadence, `startsecs`
  value and model/venv row against the tree and the live box; most held, and
  the corrections were small. Wrong: the fp8 fallback (a dropped `VLLM_VENV`
  now exits 1 at the launcher's QSA guard instead of booting BF16), the
  access boundary (the backend binds `0.0.0.0` and serves an unauthenticated
  `/api/*` to the LAN — #683), LiveKit's bind (signal is `0.0.0.0`;
  `rtc.node_ip` is only the *advertised* ICE address), the port table (no
   7881), and the unit-file exception list (four gaps, not two: untracked
   timers plus `thunderbird`/`voxtype` #1109 — now tracked and linked, so that
   one is closed — the installer linking the
   root-only power unit into the user manager #1108, and a dead tracked
  `agent-services/autonomy.service` #1110 — since removed). Recorded without changing: GPU 1
  at 450 W against a declared 400 W (#1107; the unit declares 450 W since
  2026-09-17).

## Related

[[vllm]], [[harness]], [[mission-control]], [[automod]], [[workers]],
[[voice]].
