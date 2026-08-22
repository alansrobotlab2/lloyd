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
for d in logs sessions event_logs data voice_profiles \
         _pipeline/vault-derived/facts _pipeline/research agent-services/logs; do
    if [[ -d "$d" ]]; then ok "$d"; else run mkdir -p "$d"; ok "$d (created)"; fi
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
[[ -f data/tool_overrides.yaml ]] && ok "tool_overrides.yaml present" \
    || warn "data/tool_overrides.yaml absent — tool state falls back to config.yaml defaults"

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
if [[ -x "$HOME/.bun/bin/qmd" ]]; then
    ok "$("$HOME/.bun/bin/qmd" --version 2>/dev/null || echo 'qmd (version unknown)')"
else
    run bun install -g @tobilu/qmd
fi
if [[ -f "$HOME/.config/qmd/index.yml" ]]; then
    ok "qmd collections configured"
    if [[ -d "$HOME/obsidian/lloyd" ]]; then
        run "$HOME/.bun/bin/qmd" update
        run "$HOME/.bun/bin/qmd" embed
        (( CHECK_ONLY )) || ok "index updated"
    fi
else
    todo "Restore ~/.config/qmd/index.yml from backup (defines the 15 collections), then: qmd update && qmd embed"
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

for f in models/wakeword/Lloyd.onnx models/openwakeword/melspectrogram.onnx; do
    [[ -f "$PROJECT_DIR/$f" ]] && ok "$f" \
        || todo "agent-services/$f missing — restore from backup (custom-trained, not downloadable)"
done

TTS_VOICE="$PROJECT_DIR/services/tts/qwen3-tts/voice_library/profiles/cullen"
[[ -d "$TTS_VOICE" ]] && ok "clone:cullen voice profile" \
    || todo "TTS voice profile 'cullen' missing — restore voice_library/profiles/cullen/ (config.yaml references clone:cullen)"

# ── 9. Services ─────────────────────────────────────────────────────
step "9/9  supervisord + systemd"
run bash "$PROJECT_DIR/setup/install-services.sh"

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
