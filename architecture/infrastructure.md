---
segment: architecture
tags: [architecture, lloyd, infrastructure, supervisord, systemd]
type: reference
status: implemented
date: 2026-09-11
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

`nvidia-smi` indices, pinned by `CUDA_DEVICE_ORDER=PCI_BUS_ID` in every
program's environment. `/dev/nvidiaN` does **not** match these numbers.

| GPU | Hardware | VRAM | Serves |
|---|---|---|---|
| 0 | RTX 3090 | 24 GB | Qwen3-TTS (`agent-tts`), the LiveKit voice worker, qmd embeddings (`agent-qmd-daemon`, `agent-qmd-watcher`) |
| 1 | RTX PRO 6000 Blackwell | 96 GB | the primary LLM, vLLM (`agent-llm-primary`) |
| 2 | RTX 3090 | 24 GB | the secondary LLM, llama.cpp (`agent-llm-secondary`), single-tenant at ~21.7 GB |

Board power limits are clamped at boot by the system unit
`nvidia-power-limit.service` (Xid 79 mitigation) — see
[[gpu-power-limit-persist]]. Model details are in [[engines]].

## Process management

`agent-supervisord.service` (systemd `--user`, `KillMode=control-group`,
`ExecStartPre=cleanup-orphans.sh`) runs `~/.local/bin/supervisord` (a uv
tool) with `agent-services/supervisor/supervisord.conf` + `conf.d/`.
Because the kill mode is the whole control group, an OOM of one program can
take every program with it — which is why the guardian is a separate unit.

| Program | Port | What | Start script |
|---|---|---|---|
| `lloyd-mc:lloyd-backend` | 8080 | FastAPI + SSE, the worker pool, the autonomy scheduler | `server.py` |
| `lloyd-mc:lloyd-frontend` | 5173 | Vite dev server (TLS, client certs), proxied through the backend | `npm --prefix web run dev` |
| `lloyd-mc:lloyd-mcp` | 8500 | the MCP aggregator `Server("lloyd")`, Streamable HTTP at `/mcp`; `/state`, `/health`, `/changes` | `python -m agent_mcp.main` |
| `agent-llm-primary` | 8096 | vLLM, Qwen3.8-Flash-Next, FP8 KV | `bin/start-qwen38-flash-next.sh` |
| `agent-llm-secondary` | 8091 | llama-server, Qwen3.6-35B-A3B | `bin/start-secondary.sh` |
| `agent-tts` | 8090 | Qwen3-TTS API | `bin/start-qwen3-tts.sh` |
| `agent-livekit-server` | 7880 | LiveKit SFU (binds the Tailscale IP when up) | `bin/start-livekit-server.sh` |
| `lloyd-agent-worker` | — | the LiveKit voice agent: STT, TTS shaping, wake word | `agent-services/livekit_worker.py` |
| `agent-qmd-daemon` | 8181 | qmd vector search over the vault (the fork in `~/lloyd/qmd`) | `node qmd/dist/cli/qmd.js mcp --http` |
| `agent-qmd-watcher` | — | re-embeds on vault change | `scripts/qmd-watcher.sh` |
| `agent-obsidian-sync` | — | headless Obsidian Sync | `bin/start-obsidian-sync.sh` |

The process group is `lloyd-mc`; address the three as
`lloyd-mc:lloyd-backend` etc. **Restart the backend or the aggregator through
`python -m scripts.automod.round restart`**, which pauses the worker pool,
takes the guardian's pause lease, drains, restarts with a health wait and
releases — a bare `supervisorctl restart` looks like a crash to the guardian
and kills whatever worker job is mid-flight. Never restart the primary twice
in quick succession ([[engines]]).

## Outside supervisord

| Unit | Kind | What |
|---|---|---|
| `lloyd-guardian.service` | user service, `WatchdogSec=90` | the automod rollback watchdog: stdlib-only, runs `/usr/bin/python3` from a pinned snapshot staged by `guardian-stage.sh`; deliberately not under supervisord ([[automod]]) |
| `lloyd-guardian-nag.timer` | user timer, every 15 min | re-announces an unresolved BROKEN state through `nag.py` → the same `Notifier` |
| `lloyd-qmd-cleanup.timer` | user timer, 04:45 | qmd orphaned-vector cleanup |
| `lloyd-groundskeeper-survey.timer` | user timer, 02:30 | `scripts/groundskeeper/groundskeeper-survey.py` ([[groundskeeper]]) |
| `lloyd-graph-backup.timer` | user timer, 05:30 | `scripts/backup/backup-graph.sh` — the knowledge-graph store |
| `nvidia-power-limit.service` | system service | GPU power clamp |

Unit files live in `agent-services/systemd/`; the two `lloyd-*` timers for
the groundskeeper and the graph backup exist only in
`~/.config/systemd/user/` today. Thunderbird runs as a user service because
`agent_mcp/thunderbird.py` talks to a live instance; a closed Thunderbird is
the usual reason the aggregator reports `degraded`.

## Ports

| Port | Owner |
|---|---|
| 5173 | Vite (TLS) |
| 7880 | LiveKit |
| 8080 | backend |
| 8090 | Qwen3-TTS |
| 8091 | secondary LLM |
| 8096 | primary LLM |
| 8181 | qmd |
| 8500 | lloyd-mcp |
| 18080 / 18500 | the automod **canary**: a candidate backend and aggregator booted from the round's worktree by the gate |

## Remote access

Vite terminates TLS with a local CA in `agent-services/cert/` and requires a
client certificate; the backend enforces a per-device allowlist behind the
`/api` proxy. `scripts/gen-cert.sh` creates the CA (`--force` invalidates
every device), `scripts/mint-client-cert.sh <device>` enrolls one. LiveKit
advertises the Tailscale address when Tailscale is up and falls back to the
default route.

## Scheduled work

There is no cron. Everything unasked runs through the worker pool in the
backend ([[workers]]): the autonomy fleet in `~/obsidian/autonomy/*.md`
([[autonomy-system]], the nightly chain in [[nightly-reflection]]), backlog
triage and implementation ([[automod]]), research, digests and session
mining. The systemd timers above are the only wall-clock schedules.

## Related

[[engines]], [[harness]], [[mission-control]], [[automod]], [[workers]],
[[voice]].
