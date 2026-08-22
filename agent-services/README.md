# agent-services — the service layer

Everything Lloyd runs as a long-lived process: the vLLM inference server, the
TTS server, LiveKit, qmd, Obsidian sync, and the model start scripts. All of it
runs **directly on the host** under supervisord.

For a full rebuild from a fresh OS, see **[../SETUP.md](../SETUP.md)**. This file
is the day-to-day reference.

> **History:** this tree used to be a standalone `~/Projects/lloyd-services`
> repo with its own systemd units and a `voice_mcp_server.py` / `voice_bridge.py`
> / `voice_tui.py` voice pipeline. Both are gone. Voice is now LiveKit-based
> (`livekit_worker.py` + `agent-tts`), and the systemd layer is a single
> supervisord unit. Old docs describing those paths are obsolete.

---

## Layout

```
agent-services/
├── supervisor/
│   ├── supervisord.conf     # socket, logfile, includes conf.d/*.conf
│   └── conf.d/*.conf        # one [program:...] per service + the lloyd-mc group
├── systemd/
│   └── agent-supervisord.service   # the single user unit; symlinked into ~/.config/systemd/user/
├── bin/                     # start-*.sh — one per model/service, plus benchmarks
├── setup/                   # setup-*.sh — venv builds and model downloads
├── scripts/
│   └── qmd-watcher.sh       # inotify on the vault → qmd update + embed
├── conf/                    # livekit.yaml template (rendered to .runtime with secrets)
├── cert/                    # mTLS CA + server cert + client bundles (gitignored)
├── llm/models/              # LLM weights (gitignored, ~311 GB)
├── models/                  # wake word ONNX (gitignored, custom-trained)
├── services/                # tts/, autonomy/, idle-worker/, thunderbird-mcp/
├── livekit_worker.py        # LiveKit voice agent (STT → harness → TTS)
└── logs/                    # per-service .log / .err, 10 MB rotation
```

---

## Services

Supervisord manages eleven programs. Three of them — the backend, frontend, and
MCP aggregator — are in the **`lloyd-mc` group**, so they must be addressed with
the group prefix (`lloyd-mc:lloyd-backend`, not `lloyd-backend`).

| Program | Port | Command | GPU | Autostart |
|---|---|---|---|---|
| `agent-llm-primary` | 8096 | `bin/start-qwen3.8-27b-nvfp4.sh` | 1 | yes |
| `agent-llm-secondary` | 8091 | `bin/start-gemma-4-e4b-nvfp4.sh` | — | **no** |
| `lloyd-mc:lloyd-backend` | 8080 | `.venvs/lloyd/bin/python server.py` | — | yes |
| `lloyd-mc:lloyd-frontend` | 5173 | `npm run dev` (Vite) | — | yes |
| `lloyd-mc:lloyd-mcp` | 8500 | `.venvs/lloyd/bin/python -m agent_mcp.main` | — | yes |
| `lloyd-agent-worker` | — | `.venvs/lloyd/bin/python livekit_worker.py` | — | yes |
| `agent-livekit-server` | 7880 | `bin/start-livekit-server.sh` | — | yes |
| `agent-tts` | 8090 | `bin/start-qwen3-tts.sh` | 0 | yes |
| `agent-qmd-daemon` | 8181 | `node .../qmd.js mcp --http --port 8181` | 0 | yes |
| `agent-qmd-watcher` | — | `scripts/qmd-watcher.sh` | 0 | yes |
| `agent-obsidian-sync` | — | `bin/start-obsidian-sync.sh` | — | yes |

`agent-llm-secondary` is `autostart=false` on purpose — `secondary_enabled: false`
in `config.yaml`. Start it by hand when you want the Inner Voice observer model.

---

## Operating it

supervisorctl is a uv tool and always needs `-c`. Alias it:

```bash
alias lsup='/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl \
  -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf'
```

```bash
lsup status
lsup restart lloyd-mc:lloyd-backend      # after editing server.py
lsup restart agent-llm-primary           # after editing a start script, or on a wedge
lsup tail -f agent-llm-primary stderr
lsup reread && lsup update               # after adding/editing conf.d/*.conf
```

The whole stack:

```bash
systemctl --user restart agent-supervisord.service
systemctl --user status  agent-supervisord.service
```

Frontend edits need no restart — Vite HMR picks them up.

> **Do not restart `lloyd-backend` or `lloyd-mcp` from inside a Lloyd agent
> session.** It kills the MCP client mid-RPC and leaves the process `STOPPED`.
> Recovering the `lloyd-mcp` case takes `lsup start lloyd-mc:lloyd-mcp` **and** a
> backend restart.

---

## Adding a service

1. Write the start script in `bin/`, `chmod +x`, and make it `exec` the real
   process so supervisord tracks the right PID.
2. Add `supervisor/conf.d/<name>.conf`. Supervisord runs with a **minimal
   environment** — spell out `environment=HOME=...,PATH=...` and any CUDA vars.
   Copy an existing conf; missing `PATH` is the usual cause of a service that
   works in your shell but not under supervisord.
3. If it binds a port, add that port to `bin/cleanup-orphans.sh` so a dirty
   shutdown doesn't block the next boot.
4. `lsup reread && lsup update`.

GPU-using services must set `CUDA_DEVICE_ORDER=PCI_BUS_ID` before
`CUDA_VISIBLE_DEVICES`. Without it the runtime orders by capability and index 1
(the Blackwell) is not where you think it is.

---

## Start scripts

`bin/` holds a start script per model. Only one LLM occupies the `:8096`
"primary slot" at a time; `agent-llm-primary.conf` names which one is live.
Currently `start-qwen3.8-27b-nvfp4.sh`.

Each script carries a long header comment covering that model's quantization,
speculative-decode setup, sampling defaults, and known traps. Read it before
changing flags — several encode hard-won failures (the `qwen3_xml` vs
`qwen3_coder` parser wedge, the missing-MTP-head hang, the gcc-15 requirement
for FlashInfer JIT).

Venvs are model-specific so one upgrade can't silently re-qualify another model:

| Venv | vLLM | Serves |
|---|---|---|
| `.venvs/vllm-experimental` | 0.23.1rc1.dev1218 | Qwen3.5/3.6, 35B, 122B |
| `.venvs/vllm-laguna` | 0.25.1 | Laguna S 2.1 + DFlash |
| `.venvs/vllm-qwen3.8` | nightly | Qwen3.8-27B-NVFP4 (live) |

Override without editing a start script:

```bash
VLLM_VENV=~/lloyd/.venvs/vllm-laguna bash bin/start-qwen3.8-27b-nvfp4.sh
```

---

## Setup scripts

`setup/` builds venvs and downloads models. They are idempotent and safe to
re-run. The ones that matter for a rebuild:

| Script | Does |
|---|---|
| `setup-vllm-qwen3.8.sh` | Builds `.venvs/vllm-qwen3.8`; pins via `vllm-qwen3.8.versions.txt` |
| `setup-qwen3.8-27b-nvfp4.sh` | Downloads the live primary model (22 GB) |
| `setup-vllm-experimental.sh` | Builds `.venvs/vllm-experimental` |
| `setup-qmd.sh` | Installs qmd via bun, creates collections, indexes, embeds |

`setup-all.sh`, `install-services.sh`, and the llama.cpp / Orpheus / CosyVoice
scripts target the retired systemd-and-llama.cpp stack. **Do not run them** —
`install-services.sh` in particular symlinks unit files that no longer exist
and prints start commands for services that were removed. Follow
[../SETUP.md](../SETUP.md) instead.

---

## Logs

```
agent-services/logs/<service>.log   # stdout
agent-services/logs/<service>.err   # stderr
agent-services/logs/supervisord.log # supervisord itself
```

The three `lloyd-mc` services log to `~/lloyd/logs/` instead
(`server.log`, `server.err`, `mcp.log`, `mcp.err`, `frontend.log`).

All are capped at 10 MB with rotation.
