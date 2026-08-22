# Lloyd — Bare-Metal Setup

Rebuild Lloyd on a freshly imaged host. Written against the live 2026-08-21
configuration: Arch Linux, three NVIDIA GPUs, everything running **directly on
the host** under supervisord. There is no container in the loop.

Read [Part 0](#part-0--before-you-wipe) before you reimage. Several things this
machine depends on exist **only on this disk** and are not in git.

---

## Part 0 — Before you wipe

`scripts/backup.sh` writes to `/home/alansrobotlab/backups`, which is **on the
same disk you are about to erase**. Copy the archive off-box, and grab the
items below that the daily backup does not cover.

> **Check the daily backup is not stale.** `backup.timer` ships `disabled` and
> has to be enabled by hand (Part 11). As of 2026-08-21 it was off and the newest
> archive in `~/backups` was from **2026-07-06** — 46 days old. Verify before you
> trust it:
>
> ```bash
> systemctl --user list-timers backup.timer
> ls -lt ~/backups/backup_*.tar.gz | head -1
> ```
>
> If it is stale, run `bash scripts/backup.sh` once before collecting.

Run the collector, then move its output to external storage:

```bash
bash scripts/pre-reimage-backup.sh /run/media/<you>/<external>/lloyd-preimage
```

What it captures, and why each matters:

| Item | Size | Why it can't be re-fetched |
|---|---|---|
| `~/obsidian/` (whole vault, incl. `.git` and `.obsidian`) | 89 MB | The vault has **no git remote** — local history dies with the disk. Obsidian Sync restores notes but not git history or plugin state. |
| `~/lloyd/.env` | tiny | Gitignored. LiveKit key/secret. |
| `~/lloyd/data/tool_overrides.yaml` | tiny | Gitignored UI tool toggles, merged over config.yaml at boot. |
| `~/lloyd/agent-services/services/tts/qwen3-tts/voice_library/profiles/cullen/` | 2.7 MB | The **`clone:cullen` voice** referenced by `config.yaml` → `livekit.tts.voice`. Untracked and not reproducible. |
| `~/lloyd/agent-services/cert/` | 84 KB | mTLS CA + server cert + minted client bundles. Regenerating the CA invalidates every enrolled device. |
| `~/lloyd/sessions/` | 725 MB | Conversation history. Gitignored. Optional but not recoverable. |
| `~/backups/backup_*.tar.gz` (latest + `.sha256`) | varies | The daily archive itself. |

Four things that **used** to live only on this disk are now tracked in the repo,
so they arrive with a `git clone` and need no backup: the custom-trained
wake-word ONNX models, the openwakeword base models, the vendored TTS repo's
local patch (`qwen3-tts-local.patch`), and the qmd collection definitions
(`agent-services/conf/qmd-index.yml`). Keep them that way — if you retrain a wake
word or edit the vendored TTS code, re-sync into the repo rather than relying on
a backup.

**Models are the big one.** `agent-services/llm/models/` is **311 GB** and
`agent-services/services/tts/qwen3-tts/models/` is another 4.3 GB. All of it is
re-downloadable from HuggingFace, but that is a many-hour restore. If you have
the space, copy `agent-services/llm/models/` to external storage — or at minimum
the one model the primary actually serves:
`unsloth-Qwen3.8-27B-NVFP4` (22 GB).

Also worth noting before the wipe:

- **Obsidian Sync is one-client-per-device.** After reimaging you re-run
  `ob login` + `ob sync-setup`. Do not let the desktop app's Sync plugin come
  back on — see [Part 7](#part-7--obsidian-vault--headless-sync).
- The vault's `knowledge_graph.db` / `vault.db` are rebuildable but slow.

---

## Part 1 — OS and system packages

Arch Linux. Install the toolchain Lloyd's services shell out to:

```bash
sudo pacman -S --needed \
  base-devel git curl rsync jq \
  nodejs npm \
  inotify-tools gettext iproute2 openssl \
  ffmpeg \
  cuda gcc15
```

Why these specific ones:

- **`inotify-tools`** — `agent-services/scripts/qmd-watcher.sh` calls
  `inotifywait` to reindex the vault on `.md` changes.
- **`gettext`** — `start-livekit-server.sh` uses `envsubst` to render
  `livekit.yaml.runtime` from the committed template.
- **`iproute2`** — `cleanup-orphans.sh` and `start-qwen3-tts.sh` use `ss` to
  find and clear stale port holders.
- **`gcc15`** — nvcc 13.x **cannot** compile against the gcc-16 libstdc++.
  FlashInfer JIT fails without it; the vLLM start script exports
  `NVCC_CCBIN=/usr/bin/g++-15`.
- **`openssl`** — `gen-livekit-secrets.sh` and `gen-cert.sh`.

### NVIDIA driver and CUDA

Current known-good: **driver 610.57.04, CUDA 13.3** at `/opt/cuda`.

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
/opt/cuda/bin/nvcc --version
```

Expected GPU layout — **PCI bus order matters** and every service pins it
explicitly with `CUDA_DEVICE_ORDER=PCI_BUS_ID`:

| Index | GPU | Arch | Used by |
|---|---|---|---|
| 0 | RTX 3090 (24 GB) | sm_86 | qmd daemon/watcher, Qwen3-TTS |
| 1 | RTX PRO 6000 Blackwell (96 GB) | sm_120a | vLLM primary |
| 2 | RTX 3090 (24 GB) | sm_86 | spare |

Without `CUDA_DEVICE_ORDER=PCI_BUS_ID` the runtime reorders by capability and
vLLM lands on a 3090, which cannot hold the model.

> Driver 610.x has a history of **Xid 79 "GPU fell off the bus"** on the RTX PRO
> 6000 on this box, requiring a reboot. If it recurs after reimaging, a downgrade
> to the 595 branch is the known mitigation.

---

## Part 2 — User-level toolchain

Lloyd deliberately keeps its runtimes out of `/usr`. Three separate package
roots are in play; getting these paths right matters because supervisord configs
hardcode them.

### uv (all Python venvs)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Installs to `~/.local/bin/uv`. Every venv here is uv-created against a
**uv-managed CPython 3.12**, not the system python (which is 3.14 and too new
for the dependency set):

```bash
uv python install 3.12
```

### bun (qmd only)

```bash
curl -fsSL https://bun.sh/install | bash
```

Installs to `~/.bun`. Only `@tobilu/qmd` lives here.

### npm global prefix

Node global installs go to `~/.npm-global`, not `/usr/lib/node_modules`:

```bash
npm config set prefix ~/.npm-global
```

Add to your shell rc:

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/.bun/bin:$PATH"
```

### supervisord (as a uv tool)

supervisord is **not** a pacman package here — it is a uv tool, and the systemd
unit references the uv tool path directly:

```bash
uv tool install supervisor
```

Verify both paths the unit and CLAUDE.md depend on:

```bash
ls -l ~/.local/bin/supervisord
ls -l ~/.local/share/uv/tools/supervisor/bin/supervisorctl
```

---

## Part 3 — Clone the repo and restore secrets

```bash
git clone https://github.com/alansrobotlab2/lloyd.git ~/lloyd
cd ~/lloyd
```

### `.env`

Restore from backup, or create it from the template:

```bash
cp .env.example .env
chmod 600 .env
bash scripts/gen-livekit-secrets.sh     # fills LIVEKIT_API_KEY / LIVEKIT_API_SECRET
```

`config.yaml` references these as `${LIVEKIT_API_KEY}` / `${LIVEKIT_API_SECRET}`
— `app/config.py` loads `.env` into the environment and expands `${VAR}`
placeholders throughout the config tree at boot. Never put a literal secret in
`config.yaml`; it is tracked in git.

The `ANTHROPIC_*` entries in `.env` point Claude Code at the local vLLM server
and need no secrets (`ANTHROPIC_API_KEY=no-key-required`).

### Runtime directories

Gitignored, so create them:

```bash
mkdir -p logs sessions event_logs data voice_profiles \
         _pipeline/vault-derived/facts _pipeline/research \
         agent-services/logs
```

Restore `data/tool_overrides.yaml` from backup if you have it. Without it, the
tool enable/disable state falls back to the `config.yaml` defaults, which is a
valid (if slightly different) starting point.

### mTLS certs

Restore `agent-services/cert/` from backup to keep enrolled devices working.
Only if you are starting fresh:

```bash
bash scripts/gen-cert.sh
# then re-enroll each device:
bash scripts/mint-client-cert.sh <device-name>
```

`gen-cert.sh --force` regenerates the CA and **invalidates every existing client
cert**. Don't use it on a restore.

---

## Part 4 — Python virtual environments

Four venvs under `~/lloyd/.venvs/`. All are Python 3.12 from uv.

### `lloyd` — backend, MCP aggregator, LiveKit voice worker

This is the one CLAUDE.md refers to as "the venv".

```bash
cd ~/lloyd
uv venv .venvs/lloyd --python 3.12
.venvs/lloyd/bin/python -m ensurepip
.venvs/lloyd/bin/pip install -r requirements.lock
.venvs/lloyd/bin/playwright install chromium
```

Install from **`requirements.lock`**, not `requirements.txt`. The lock is the
frozen 162-package snapshot; `requirements.txt` holds loose human-edited intent
and resolving it fresh will pull an incompatible `mcp` major. It carries the
voice stack too — `faster-whisper`, `ctranslate2`, `openwakeword`, `onnxruntime`,
`Resemblyzer`, `livekit`, and a CPU `torch` 2.11.

Only reach for `requirements.txt` when intentionally upgrading, then refreeze:

```bash
.venvs/lloyd/bin/python -m pip freeze > requirements.lock   # keep the header
```

### `vllm-qwen3.8` — the primary inference server

```bash
bash agent-services/setup/setup-vllm-qwen3.8.sh
```

Builds a dedicated bleeding-edge vLLM venv, pinned by
`agent-services/setup/vllm-qwen3.8.versions.txt`. The script pins vLLM and lets
vLLM drag in its own torch/flashinfer/transformers — do **not** pre-pin torch.
It verifies the result is a working CUDA 13 / SM120 stack rather than assuming it.

Two other model-specific venvs exist and are only needed if you run those models:

| Venv | vLLM | Serves |
|---|---|---|
| `vllm-experimental` | 0.23.1rc1.dev1218 | Qwen3.5/3.6 family, 35B, 122B |
| `vllm-laguna` | 0.25.1 | Laguna S 2.1 + DFlash draft |
| `vllm-qwen3.8` | nightly | **Qwen3.8-27B-NVFP4 (live primary)** |

`vllm-experimental` is built by `setup-vllm-experimental.sh`, pinned by
`setup/vllm-experimental.versions.txt`. **`vllm-laguna` has no setup script** — it was built by hand. If you need Laguna S 2.1 back, adapt
`setup-vllm-qwen3.8.sh` (pin vLLM 0.25.1) and see
`setup/setup-qwen3.8-27b-nvfp4.sh`'s sibling `bin/start-laguna-s-2.1-nvfp4-dflash.sh`
for the runtime flags.

### `qwen3-tts` — TTS API server

The upstream repo ships a `pyproject.toml` (with an `api` extra), **not** a
`requirements.txt`. It also needs a **CUDA 13 nightly torch** and a `flash_attn`
built against it — `config.yaml` in the TTS repo sets
`optimization.attention: flash_attention_2`.

Install from the frozen snapshot, which pins the exact nightly:

```bash
cd ~/lloyd
uv venv .venvs/qwen3-tts --python 3.12
.venvs/qwen3-tts/bin/python -m ensurepip

# torch first, from the cu130 nightly index — the pinned .dev build
.venvs/qwen3-tts/bin/pip install --pre torch torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu130

.venvs/qwen3-tts/bin/pip install -r agent-services/setup/qwen3-tts.versions.txt

# the vendored repo itself, editable, last
.venvs/qwen3-tts/bin/pip install -e \
  'agent-services/services/tts/qwen3-tts[api]' --no-deps
```

If `flash-attn` has to compile rather than fetch a wheel, it needs
`NVCC_CCBIN=/usr/bin/g++-15` like everything else here, and it is slow.
Falling back to `attention: sdpa` in the TTS `config.yaml` works and skips it.

See [Part 8](#part-8--qwen3-tts) — the source directory is a **vendored upstream
repo with local patches**, so restore it before building this venv.

### `whisper`, `voice-mode`

Legacy venvs from the pre-LiveKit voice stack. Nothing under supervisord
references them. Skip unless you are reviving that path.

---

## Part 5 — Frontend

```bash
cd ~/lloyd/web
npm install
```

Supervisord runs `npm run dev` (Vite dev server on `:5173`), proxied through the
backend — there is no production build step in the service path.

---

## Part 6 — qmd (vault search)

qmd is the retrieval backbone. Installed via bun, but **run by system node** —
the supervisord config invokes
`/usr/bin/node .../@tobilu/qmd/dist/cli/qmd.js`, not the bun shim.

```bash
bun install -g @tobilu/qmd
~/.bun/bin/qmd --version    # expect 2.0.1
```

### Collections

qmd reads `~/.config/qmd/index.yml`. A tracked reference copy lives in the repo,
so this does **not** depend on a backup:

```bash
mkdir -p ~/.config/qmd
cp agent-services/conf/qmd-index.yml ~/.config/qmd/index.yml
```

It defines the 14 collections (`facts`, `memory`, `knowledge`, `projects`,
`lloyd`, `personal`, `work`, `skills`, `people`, `subliminal`, `backlog`,
`autonomy`, `sessions`, `architecture`) all rooted under `~/obsidian`.

If you change collections later, re-sync the tracked copy:

```bash
cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml
```

Then build the index (needs the vault in place first — do [Part 7](#part-7--obsidian-vault--headless-sync) first):

```bash
qmd update      # FTS index
qmd embed       # vector embeddings — downloads the embedding model on first run
qmd status
```

A healthy index looks like ~9.4k documents / ~32k vectors at ~840 MB.

### Two known qmd traps

- **Orphaned vectors.** If `vault_search` gets slow (`vec=700ms` at 0% GPU and
  one pinned CPU core) or multi-hop recall collapses, the index has accumulated
  orphaned embedding chunks. Fix with `qmd cleanup`. This is a *quality* fix,
  not just speed.
- **GPU arch mismatch.** If `vault_search` returns **0 results**, qmd's
  `node-llama-cpp` CUDA binary was built without the target GPU's arch and the
  vector leg crashes, zeroing the fused lex+vec query. Rebuild `node-llama-cpp`
  for `86-real;120a-real` into the default directory. The current daemon sidesteps
  this by running `QMD_VEC_BACKEND=bit`.

`qmd status` also reports `AST Chunking: unavailable` because `web-tree-sitter`
isn't installed. That is cosmetic — it falls back to regex chunking.

---

## Part 7 — Obsidian vault + headless sync

The vault lives at `~/obsidian` and is Lloyd's long-term memory. It holds
`lloyd/SOUL.md`, `lloyd/MEMORY.md`, and `lloyd/USER.md`, which
`prompt_builder.py` assembles into the system prompt on every turn.

```bash
npm install -g obsidian-headless
ob login                                            # email + password + MFA
ob sync-setup --vault "<vault-name>" --path ~/obsidian   # + E2E encryption password
```

> **The desktop Obsidian app's Sync core plugin must stay disabled.** Obsidian
> supports one sync client per device; running desktop Sync and Headless Sync on
> the same vault causes data conflicts.

Restore the vault's `.git` from backup — it has **no remote**, and the autonomy
daemon commits to it continuously. Vault edits must land on `main`; the daemon
checks out `main` and will revert feature-branch work sitting on disk.

If `ob` fails on a native module error, `better-sqlite3` must match the host
node ABI. Rebuild it with **node-gyp** — `npm rebuild` silently no-ops on npm 12.

---

## Part 8 — Qwen3-TTS

`agent-services/services/tts/qwen3-tts/` is a checkout of
[groxaxo/Qwen3-TTS-Openai-Fastapi](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi)
with **local modifications that are not committed anywhere**. Only three
unrelated `services/tts/*.py` files are tracked in the Lloyd repo — the whole
qwen3-tts subtree is its own git repo and is not a submodule.

Restore order:

The local patch **is tracked in this repo** — only the cloned voice needs a
backup. Rebuild from scratch:

```bash
cd ~/lloyd/agent-services/services/tts
git clone https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git qwen3-tts
cd qwen3-tts
git checkout "$(cut -d' ' -f1 ../qwen3-tts-upstream-commit.txt)"   # 308fbd4
git apply ../qwen3-tts-local.patch
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

Then restore `voice_library/profiles/cullen/` from backup — that one is not
reproducible. Re-sync the patch if you change the vendored code:

```bash
git -C qwen3-tts diff > qwen3-tts-local.patch
```

`config.yaml` sets `livekit.tts.voice: clone:cullen`, which resolves against
`voice_library/profiles/cullen/`. Without it, TTS starts but every synthesis
request for that voice fails.

First boot compiles with `max-autotune` and takes ~75 s before the port opens —
`startsecs=10` in the supervisor config tolerates this because uvicorn binds
before compilation finishes.

---

## Part 9 — LLM models

The primary serves `unsloth/Qwen3.8-27B-NVFP4` on `:8096`.

```bash
bash agent-services/setup/setup-qwen3.8-27b-nvfp4.sh
```

Downloads to `agent-services/llm/models/unsloth-Qwen3.8-27B-NVFP4` (22 GB).

The script downloads via the `hf` CLI **from inside the vLLM venv**, so build
that venv first (Part 4). `huggingface_hub` comes in as a vLLM dependency and
provides `hf`; if the script still reports it missing, it prints the fix:

```bash
.venvs/vllm-qwen3.8/bin/python -m pip install -U 'huggingface_hub[cli]'
```

Two things the start script checks for, and why:

- **`config.json`** must exist or it refuses to boot with a pointer to the setup
  script.
- **`model_mtp.safetensors`** (~810 MB, 15 BF16 tensors) is the MTP speculative
  decode head. If it's missing the script drops `--speculative-config` and warns
  rather than wedging the engine. Verify it survived the download — an earlier
  Qwen3.6 re-quant declared `mtp_num_hidden_layers=1` while shipping **zero**
  `mtp.*` tensors, which hung vLLM at load. Check the safetensors index, don't
  trust the config.

Other models under `agent-services/llm/models/` (Laguna S 2.1, the Qwen3.5/3.6
family, Gemma 4) are optional — download only what you'll run. They share the
`:8096` "primary slot" and only one runs at a time.

---

## Part 10 — LiveKit

`start-livekit-server.sh` downloads the binary itself on first run:

```bash
# nothing to do — it fetches v1.11.0 to ~/.local/bin/livekit-server if missing
```

It reads `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` from `~/lloyd/.env` (it does
**not** `source` the file — other entries contain spaces and would fail to parse
as bash), renders `agent-services/conf/livekit.yaml` through `envsubst` into
`livekit.yaml.runtime` (chmod 600, gitignored), and execs the server on `:7880`.

If the secrets are missing it exits with a pointer to `gen-livekit-secrets.sh`.

### Wake word models

These four ONNX files **are tracked in this repo** and arrive with the clone —
nothing to do:

```
agent-services/models/wakeword/Lloyd.onnx        # custom-trained
agent-services/models/wakeword/Hey_Lloyd.onnx    # custom-trained
agent-services/models/openwakeword/melspectrogram.onnx
agent-services/models/openwakeword/embedding_model.onnx
```

They are force-added past `agent-services/.gitignore`'s `models/` rule, which is
unanchored and would otherwise also match the 311 GB `llm/models/` tree. If you
retrain a wake word, `git add -f` the new file — do **not** try to negate the
ignore rule.

Without these, `livekit.acoustic_wake.enabled: true` fails and voice falls back
to text wake-word matching only.

---

## Part 11 — supervisord and systemd

This is the layer that ties everything together. **One** systemd user unit runs
supervisord; supervisord runs all eleven services.

### Install the unit

The unit file is tracked at `agent-services/systemd/agent-supervisord.service`
and gets **symlinked** into place:

```bash
mkdir -p ~/.config/systemd/user
ln -sf ~/lloyd/agent-services/systemd/agent-supervisord.service \
       ~/.config/systemd/user/agent-supervisord.service
systemctl --user daemon-reload
```

### Enable lingering

Without this, supervisord dies when you log out and never starts at boot:

```bash
sudo loginctl enable-linger alansrobotlab
loginctl show-user alansrobotlab -p Linger   # expect Linger=yes
```

### Start it

```bash
systemctl --user enable --now agent-supervisord.service
systemctl --user status agent-supervisord.service
```

The unit runs `bin/cleanup-orphans.sh` as `ExecStartPre`, which kills stale
listeners on ports 8093/8094/8096/8097/8098/8099/8181/18789 so a dirty shutdown
doesn't cause "address already in use" on the next boot.

### How supervisord is wired

`agent-services/supervisor/supervisord.conf` sets the socket at
`/tmp/agent-supervisor.sock` and includes every file in
`agent-services/supervisor/conf.d/*.conf`. Each service is one `[program:...]`
block; `lloyd-mc.conf` defines the **group**:

```ini
[group:lloyd-mc]
programs=lloyd-backend,lloyd-frontend,lloyd-mcp
```

Because of that group, the three core services are addressed as
`lloyd-mc:lloyd-backend`, **not** bare `lloyd-backend`.

### supervisorctl

The binary is a uv tool and needs `-c`. Define an alias:

```bash
alias lsup='/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl \
  -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf'

lsup status
lsup restart lloyd-mc:lloyd-backend
lsup restart lloyd-mc:lloyd-frontend
```

After adding or editing a file in `conf.d/`:

```bash
lsup reread && lsup update
```

> **Never restart `lloyd-backend` or `lloyd-mcp` from inside a Lloyd agent
> session.** The restart kills the MCP client mid-RPC and leaves the process
> `STOPPED`. The `lloyd-mcp` variant is worse — every tool goes offline. Recovery
> needs both `lsup start lloyd-mc:lloyd-mcp` **and** a backend restart.

### Service inventory

| Service | Port | Runs | Autostart |
|---|---|---|---|
| `agent-llm-primary` | 8096 | `bin/start-qwen3.8-27b-nvfp4.sh` → vLLM on GPU 1 | yes |
| `agent-llm-secondary` | 8091 | `bin/start-gemma-4-e4b-nvfp4.sh` | **no** |
| `lloyd-mc:lloyd-backend` | 8080 | `.venvs/lloyd/bin/python server.py` | yes |
| `lloyd-mc:lloyd-frontend` | 5173 | `npm run dev` (Vite) | yes |
| `lloyd-mc:lloyd-mcp` | 8500 | `.venvs/lloyd/bin/python -m agent_mcp.main` | yes |
| `lloyd-agent-worker` | — | `.venvs/lloyd/bin/python agent-services/livekit_worker.py` | yes |
| `agent-livekit-server` | 7880 | `bin/start-livekit-server.sh` | yes |
| `agent-tts` | 8090 | `bin/start-qwen3-tts.sh` on GPU 0 | yes |
| `agent-qmd-daemon` | 8181 | `node .../qmd.js mcp --http` on GPU 0 | yes |
| `agent-qmd-watcher` | — | `scripts/qmd-watcher.sh` (inotify → reindex) | yes |
| `agent-obsidian-sync` | — | `bin/start-obsidian-sync.sh` (`ob sync --continuous`) | yes |

Environment is set **per-program in the conf file**, not inherited from your
shell — supervisord runs with a minimal environment. If a service can't find a
binary, the fix is almost always its `environment=...PATH=...` line.

### Optional timers

```bash
systemctl --user enable --now backup.timer   # scripts/backup.sh, daily 02:00
```

Note that `backup.sh` targets `~/backups` on the local disk and covers only
`~/obsidian` and `~/lloyd/scripts`. Consider pointing `BACKUP_BASE` at external
storage and widening `SOURCE_DIRS` to include the untracked items listed in
[Part 0](#part-0--before-you-wipe).

---

## Part 12 — Verify

```bash
lsup status          # all RUNNING except agent-llm-secondary (STOPPED by design)
```

Then check each layer:

```bash
# vLLM primary — should list "primary" and the model id
curl -s localhost:8096/v1/models | jq '.data[].id'

# Backend — /api/services also reports what the backend thinks is up
curl -s localhost:8080/api/models   | jq -r '.[]?.alias // empty' 2>/dev/null || curl -s localhost:8080/api/models
curl -s localhost:8080/api/services

# MCP aggregator — SSE endpoint should hold open
curl -sN --max-time 2 localhost:8500/sse | head -3

# qmd
qmd status
qmd query "test" -c obsidian -n 1

# TTS
curl -s localhost:8090/v1/voices | jq '.[0:3]'

# LiveKit
curl -s localhost:7880 && echo OK

# Frontend
curl -sI localhost:5173 | head -1
```

Speculative decode acceptance on the primary (expect ~70% at 3 tokens):

```bash
curl -s localhost:8096/metrics | grep vllm:spec_decode
```

Finally, exercise the whole path from the UI at `http://localhost:8080` — send a
message that forces a tool call, and confirm vault search returns results
(that proves qmd, the MCP aggregator, and the harness are all wired).

---

## Troubleshooting

**Primary wedges — 200 OK but no tokens.** Recurring. Confirm with
`grammar_matcher.cc:612` stop-token loop in
`agent-services/logs/agent-llm-primary.err` plus a hung POST. Fix:
`lsup restart agent-llm-primary`. If it recurs constantly, bisect by disabling
MTP (drop `--speculative-config` from the start script).

**Primary wedges on the stop-token path at boot.** Check `ps` for the tool
parser flag — it must be `--tool-call-parser qwen3_xml`. `qwen3_coder` wedges
this model family.

**`vault_search` returns 0 results.** qmd GPU arch mismatch — see
[Part 6](#part-6--qmd-vault-search).

**`vault_search` is slow.** Orphaned vectors — `qmd cleanup`.

**FlashInfer JIT compile fails.** `/usr/bin/g++-15` missing, or `NVCC_CCBIN`
not exported. nvcc 13.x cannot parse the gcc-16 libstdc++.

**vLLM loads on the wrong GPU.** `CUDA_DEVICE_ORDER=PCI_BUS_ID` missing.

**A tool is missing from the UI.** `config.yaml` is read-only at boot; UI toggles
persist to `data/tool_overrides.yaml`, which is merged over it
(`app/config.py:_merge_tool_overrides`). Check the override file isn't shadowing
the key you edited.

**Autonomy tasks silently stop running.** The scheduler drops tasks whose
frontmatter fails `yaml.safe_load` with no alert. Run
`scripts/autonomy/validate_tasks.py`.

**Backend `STOPPED` after an agent-initiated restart.** See the warning in
[Part 11](#part-11--supervisord-and-systemd).
