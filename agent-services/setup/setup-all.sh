#!/usr/bin/env bash
set -euo pipefail

# Master setup script — runs all individual setup scripts in order.
# Each step is idempotent and can be re-run safely.
#
# Usage:
#   bash setup/setup-all.sh          # Run everything
#   bash setup/setup-all.sh --skip-models  # Skip large model downloads

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SKIP_MODELS=false
if [[ "${1:-}" == "--skip-models" ]]; then
    SKIP_MODELS=true
    echo "Skipping model downloads (--skip-models)"
fi

echo "============================================"
echo "  Lloyd Services — Full Setup"
echo "============================================"
echo ""

# 1. Main project venv (voice pipeline + MCP tools)
echo "--- [1/5] Main project venv ---"
if [ ! -d ".venv" ]; then
    uv sync
    echo "Main venv created."
else
    echo "Main venv already exists. Run 'uv sync' to update."
fi
echo ""

# 2. LLM server (llama.cpp + Qwen model)
echo "--- [2/5] LLM Server ---"
bash setup/setup-llm.sh
echo ""

# 3. Orpheus TTS
echo "--- [3/5] Orpheus TTS ---"
bash setup/setup-orpheus.sh
echo ""

# 4. CosyVoice TTS (optional, large download)
echo "--- [4/5] CosyVoice TTS ---"
if [ "$SKIP_MODELS" = true ]; then
    echo "Skipped (--skip-models). Run manually: bash setup/setup-cosyvoice.sh"
else
    bash setup/setup-cosyvoice.sh
fi
echo ""

# 5. QMD search (bun + GGUF models)
echo "--- [5/6] QMD Search ---"
bash setup/setup-qmd.sh
echo ""

# 6. Install systemd services
echo "--- [6/6] Systemd Services ---"
bash setup/install-services.sh
echo ""

echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Start all services:"
echo "  systemctl --user start lloyd-llm lloyd-tts lloyd-voice-mode lloyd-voice-mcp lloyd-tool-mcp openclaw-gateway lloyd-qmd-daemon lloyd-qmd-watcher"
echo ""
echo "Check status:"
echo "  systemctl --user status lloyd-llm lloyd-tts lloyd-voice-mode lloyd-voice-mcp lloyd-tool-mcp openclaw-gateway lloyd-qmd-daemon lloyd-qmd-watcher"
