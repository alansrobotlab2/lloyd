#!/usr/bin/env bash
set -euo pipefail

# Master setup for the CURRENT Lloyd stack: host-direct, supervisord-managed.
#
# Every step is idempotent and safe to re-run. Steps that need a human (Obsidian
# login, vault restore) are detected and reported rather than attempted.
#
# Full prose walkthrough, including OS packages and the pre-reimage backup:
#   ../SETUP.md
#
# Usage:
#   bash setup/setup-all.sh                 # everything except the 22GB model
#   bash setup/setup-all.sh --with-models   # also download the primary model
#   bash setup/setup-all.sh --check         # report what's missing, change nothing
#
# NOTE: this replaced an older script that built a llama.cpp + Orpheus +
# CosyVoice stack behind per-service systemd units. None of that is live.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # ~/lloyd/agent-services
REPO="$(cd "$PROJECT_DIR/.." && pwd)"             # ~/lloyd
cd "$REPO"

WITH_MODELS=0
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --with-models) WITH_MODELS=1 ;;
        --check)       CHECK_ONLY=1 ;;
        -h|--help)     sed -n '3,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

step()  { echo; echo "─── $* ───"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ! $*"; }
todo()  { echo "  → $*"; MANUAL+=("$*"); }
MANUAL=()

run() {
    if (( CHECK_ONLY )); then
        echo "  (--check) would run: $*"
    else
        "$@"
    fi
}

echo "============================================"
echo "  Lloyd — setup"
echo "  repo:   $REPO"
echo "  mode:   $( ((CHECK_ONLY)) && echo 'check only' || echo 'apply' )"
echo "============================================"

# ── 1. Toolchain ────────────────────────────────────────────────────
step "1/9  Toolchain"
missing_cmd=0
for c in uv node npm git curl rsync openssl inotifywait envsubst ss; do
    if command -v "$c" >/dev/null 2>&1; then ok "$c"; else warn "$c MISSING"; missing_cmd=1; fi
done
[[ -x /opt/cuda/bin/nvcc ]] && ok "CUDA at /opt/cuda ($(/opt/cuda/bin/nvcc --version | grep -oP 'release \K[0-9.]+'))" \
                            || { warn "CUDA toolkit missing at /opt/cuda"; missing_cmd=1; }
[[ -x /usr/bin/g++-15 ]] && ok "g++-15 (FlashInfer JIT)" \
                         || warn "g++-15 missing — FlashInfer JIT will fail against gcc-16 libstdc++"
[[ -x "$HOME/.bun/bin/bun" ]] && ok "bun" || { warn "bun missing — needed for qmd"; missing_cmd=1; }
[[ -x "$HOME/.local/bin/supervisord" ]] && ok "supervisord" \
                                        || { warn "supervisord missing — run: uv tool install supervisor"; missing_cmd=1; }
if (( missing_cmd )); then
    echo
    echo "Install the missing pieces first — see SETUP.md Parts 1 and 2."
    exit 1
fi

# ── 2. Runtime directories ──────────────────────────────────────────
step "2/9  Runtime directories"
# Runtime data lives outside the tree since 2026-09-22 (architecture/data-home.md):
# its own btrfs subvolume, so it can be snapshotted and no delete aimed at the
# code reaches it. The marker is what app.paths requires before production boots.
DATA="${LLOYD_DATA:-$HOME/lloyd-data}"
if [[ -d "$DATA" ]]; then ok "$DATA"
elif [[ "$(findmnt -no FSTYPE -T "$HOME")" == btrfs ]]; then
    run btrfs subvolume create "$DATA"; ok "$DATA (btrfs subvolume created)"
else
    run mkdir -p "$DATA"; warn "$DATA created as a plain directory (not btrfs: no snapshots)"
fi
[[ -f "$DATA/.lloyd-data-root" ]] || run touch "$DATA/.lloyd-data-root"
for d in logs/services sessions event_logs data voice_profiles \
         _pipeline/vault-derived/facts _pipeline/research; do
    if [[ -d "$DATA/$d" ]]; then ok "$DATA/$d"; else run mkdir -p "$DATA/$d"; ok "$DATA/$d (created)"; fi
done

# ── 3. Secrets ──────────────────────────────────────────────────────
step "3/9  Secrets"
if [[ -f .env ]]; then
    ok ".env present"
else
    run cp .env.example .env
    run chmod 600 .env
    ok ".env created from .env.example"
fi
if grep -q '^LIVEKIT_API_KEY=.\+' .env 2>/dev/null; then
    ok "LiveKit credentials set"
else
    run bash scripts/gen-livekit-secrets.sh --keep
fi
if [[ -f agent-services/cert/ca.crt ]]; then
    ok "mTLS certs present"
else
    todo "No mTLS certs. Restore agent-services/cert/ from backup, or run: bash scripts/gen-cert.sh"
fi
[[ -f "$DATA/data/tool_overrides.yaml" ]] && ok "tool_overrides.yaml present" \
    || warn "$DATA/data/tool_overrides.yaml absent — tool state falls back to config.yaml defaults"

# ── 4. Python venvs ─────────────────────────────────────────────────
step "4/9  Python venvs"
if [[ -x .venvs/lloyd/bin/python ]]; then
    ok "lloyd venv ($(.venvs/lloyd/bin/python --version))"
else
    run uv venv .venvs/lloyd --python 3.12
    run .venvs/lloyd/bin/python -m ensurepip
    run .venvs/lloyd/bin/pip install -r requirements.lock
    run .venvs/lloyd/bin/playwright install chromium
    ok "lloyd venv built from requirements.lock"
fi

if [[ -x .venvs/vllm-qwen3.8/bin/python ]]; then
    ok "vllm-qwen3.8 venv"
else
    warn "vllm-qwen3.8 venv missing — building (this takes a while)"
    run bash agent-services/setup/setup-vllm-qwen3.8.sh
fi

if [[ -x .venvs/qwen3-tts/bin/python ]]; then
    ok "qwen3-tts venv"
else
    TTS_DIR="$PROJECT_DIR/services/tts/qwen3-tts"
    TTS_LOCK="$PROJECT_DIR/setup/qwen3-tts.versions.txt"
    if [[ -f "$TTS_DIR/pyproject.toml" && -f "$TTS_LOCK" ]]; then
        # torch/torchaudio are cu130 nightlies and flash_attn is built against
        # them, so install torch from the nightly index before the lock.
        run uv venv .venvs/qwen3-tts --python 3.12
        run .venvs/qwen3-tts/bin/python -m ensurepip
        run .venvs/qwen3-tts/bin/pip install --pre torch torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu130
        run .venvs/qwen3-tts/bin/pip install -r "$TTS_LOCK"
        run .venvs/qwen3-tts/bin/pip install -e "$TTS_DIR[api]" --no-deps
        (( CHECK_ONLY )) || ok "qwen3-tts venv built"
    else
        todo "qwen3-tts source missing at $TTS_DIR — restore it first (SETUP.md Part 8)"
    fi
fi

# ── 5. Frontend ─────────────────────────────────────────────────────
step "5/9  Frontend"
if [[ -d web/node_modules ]]; then
    ok "web/node_modules present"
else
    run npm --prefix web install
    ok "web dependencies installed"
fi

# ── 6. Vault ────────────────────────────────────────────────────────
step "6/9  Obsidian vault"
if [[ -d "$HOME/obsidian/lloyd" ]]; then
    ok "vault present at ~/obsidian"
    for f in SOUL.md MEMORY.md USER.md; do
        [[ -f "$HOME/obsidian/lloyd/$f" ]] && ok "lloyd/$f" || warn "lloyd/$f MISSING — system prompt will be incomplete"
    done
else
    todo "Vault missing at ~/obsidian. Restore from backup, then: ob login && ob sync-setup --vault <name> --path ~/obsidian"
fi
if [[ -x "$HOME/.npm-global/bin/ob" ]]; then
    ok "obsidian-headless installed"
else
    run npm install -g obsidian-headless
    todo "Run once, interactively: ob login  &&  ob sync-setup --vault <name> --path ~/obsidian"
fi

# ── 7. qmd ──────────────────────────────────────────────────────────
step "7/9  qmd search"
# ONE qmd on this machine: the fork in ~/lloyd/qmd. The daemon, the watcher,
# the nightly cleanup timer, the index-maintenance task and the automod eval pin
# all run its dist/cli/qmd.js, and the `qmd` on PATH is a symlink to its
# launcher. The published @tobilu/qmd used to be installed beside it (bun
# global) and the watcher and this script ran THAT one -- same version string,
# different commit, against an index the fork serves. SETUP.md Part 6.
QMD_FORK="$HOME/lloyd/qmd"
QMD_JS="$QMD_FORK/dist/cli/qmd.js"
if command -v bun >/dev/null 2>&1 && bun pm ls -g 2>/dev/null | grep -q "@tobilu/qmd"; then
    todo "Published qmd is still installed: bun remove -g @tobilu/qmd   (it shadows the fork on PATH)"
fi
if [[ -f "$QMD_JS" ]]; then
    ok "$(/usr/bin/node "$QMD_JS" --version 2>/dev/null || echo 'qmd fork (version unknown)')"
    if grep -q -- '-dirty' "$QMD_FORK/dist/cli/build-info.json" 2>/dev/null; then
        todo "qmd fork was built from a dirty tree: cd $QMD_FORK && git status, then npm run build"
    fi
elif [[ -d "$QMD_FORK/.git" ]]; then
    run bash -c "cd '$QMD_FORK' && bun install && npm run build"
else
    todo "Clone the fork: git clone https://github.com/alansrobotlab2/qmd.git $QMD_FORK && cd $QMD_FORK && git checkout lloyd && bun install && npm run build"
fi
if [[ -f "$QMD_JS" ]]; then
    if [[ "$(readlink -f "$HOME/.local/bin/qmd" 2>/dev/null)" == "$QMD_FORK/bin/qmd" ]]; then
        ok "qmd on PATH is the fork"
    else
        run ln -sfn "$QMD_FORK/bin/qmd" "$HOME/.local/bin/qmd"
    fi
fi
if [[ -f "$HOME/.config/qmd/index.yml" ]]; then
    ok "qmd collections configured"
    if [[ -d "$HOME/obsidian/lloyd" && -f "$QMD_JS" ]]; then
        run /usr/bin/node "$QMD_JS" update
        run /usr/bin/node "$QMD_JS" embed
        (( CHECK_ONLY )) || ok "index updated"
    fi
else
    todo "Restore ~/.config/qmd/index.yml from backup (defines the collections), then: qmd update && qmd embed"
fi

# ── 8. Models ───────────────────────────────────────────────────────
step "8/9  Models"
PRIMARY_MODEL="$PROJECT_DIR/llm/models/unsloth-Qwen3.8-27B-NVFP4"
if [[ -f "$PRIMARY_MODEL/config.json" ]]; then
    ok "primary model present"
    [[ -f "$PRIMARY_MODEL/model_mtp.safetensors" ]] && ok "MTP head present (speculative decode enabled)" \
        || warn "model_mtp.safetensors missing — vLLM will start WITHOUT speculative decode"
elif (( WITH_MODELS )); then
    run bash agent-services/setup/setup-qwen3.8-27b-nvfp4.sh
else
    todo "Primary model not downloaded (22 GB). Run: bash setup/setup-qwen3.8-27b-nvfp4.sh   (or re-run with --with-models)"
fi

for f in models/wakeword/Lloyd.onnx models/openwakeword/melspectrogram.onnx models/silero-vad/silero_vad.onnx; do
    [[ -f "$PROJECT_DIR/$f" ]] && ok "$f" \
        || todo "agent-services/$f missing — it is tracked; check out the repo again"
done

# Smart Turn, Parakeet, streaming ASR: ~1.1 GB, re-downloadable, untracked.
if bash "$PROJECT_DIR/setup/fetch-voice-models.sh" --check >/dev/null 2>&1; then
    ok "voice models (smart-turn, parakeet, nemo-streaming)"
else
    run bash "$PROJECT_DIR/setup/fetch-voice-models.sh"
fi

TTS_VOICE="$PROJECT_DIR/services/tts/qwen3-tts/voice_library/profiles/dave_cullen"
[[ -d "$TTS_VOICE" ]] && ok "clone:dave_cullen voice profile" \
    || todo "TTS voice profile 'dave_cullen' missing — restore voice_library/profiles/dave_cullen/ from backup (config.yaml references clone:dave_cullen; not reproducible)"

# ── 9. Services ─────────────────────────────────────────────────────
step "9/9  supervisord + systemd"
run bash "$PROJECT_DIR/setup/install-services.sh"

# Thunderbird MCP bridge — the gitignored Node bridge (mcp-bridge.cjs) that
# backs the aggregator's 40 email/calendar/contacts/to-do tools. Its directory
# is gitignored and `thunderbird.list_tools()` degrades to an EMPTY list — still
# "ok", no degraded_module, no alert — when the file is absent, so a fresh clone
# serves 117 tools instead of 157 with `git status` clean. Nothing else surfaces
# this. SETUP.md Part 13, "Thunderbird bridge".
TB_BRIDGE="$PROJECT_DIR/services/thunderbird-mcp/mcp-bridge.cjs"
if [[ ! -f "$TB_BRIDGE" ]]; then
    todo "Thunderbird MCP bridge missing — 40 email/calendar/contacts tools will be absent. Run: bash agent-services/setup/setup-thunderbird-mcp.sh"
elif ss -ltn 2>/dev/null | grep -qE '[:.]8765( |$)'; then
    ok "thunderbird-mcp bridge present, Thunderbird MCP extension live on :8765"
else
    warn "thunderbird-mcp bridge present but nothing on :8765 — Thunderbird or its MCP extension is down; those 40 tools will not be served (systemctl --user status thunderbird)"
fi

# thunderbird + voxtype systemd units (#1109) — tracked in agent-services/systemd/,
# linked by install-services.sh, WantedBy=graphical-session.target. They lost
# 40 tools + push-to-talk silently when they were untracked; the check keeps that
# from regressing on a rebuild.
for unit in thunderbird voxtype; do
    if [[ ! -e "$HOME/.config/systemd/user/$unit.service" ]]; then
        todo "$unit.service not installed — run: bash agent-services/setup/install-services.sh"
    elif [[ "$(systemctl --user is-enabled "$unit" 2>/dev/null)" != "enabled" ]]; then
        todo "$unit.service present but not enabled — run: systemctl --user enable $unit"
    else
        ok "$unit.service installed and enabled (graphical-session unit)"
    fi
done

echo
echo "============================================"
if (( ${#MANUAL[@]} )); then
    echo "  Setup incomplete — manual steps remain"
    echo "============================================"
    printf '  → %s\n' "${MANUAL[@]}"
    echo
    echo "Details for each: SETUP.md"
else
    echo "  Setup complete"
    echo "============================================"
    echo
    echo "Start everything:"
    echo "  systemctl --user enable --now agent-supervisord.service"
    echo
    echo "Then verify (SETUP.md Part 12):"
    echo "  $HOME/.local/share/uv/tools/supervisor/bin/supervisorctl \\"
    echo "    -c $PROJECT_DIR/supervisor/supervisord.conf status"
fi
