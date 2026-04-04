#!/usr/bin/env bash
# Setup venv for SGLang + Qwen3.5-122B-A10B-NVFP4 on RTX PRO 6000 Blackwell
set -euo pipefail

VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)/.venvs/sglang"

PYTHON="${PYTHON:-python3.11}"

if ! command -v "$PYTHON" &>/dev/null; then
  echo "ERROR: $PYTHON not found. SGLang requires Python 3.9–3.12."
  echo "Try: PYTHON=python3.12 ./scripts/setup-sglang.sh"
  exit 1
fi

echo "==> Creating venv at $VENV_DIR (using $PYTHON)"
"$PYTHON" -m venv "$VENV_DIR"

echo "==> Installing SGLang + dependencies"
"$VENV_DIR/bin/pip" install --upgrade pip

"$VENV_DIR/bin/pip" install "sglang[all]"
"$VENV_DIR/bin/pip" install "transformers==4.57.1" accelerate

# Required for safe FP4 GEMM backend (avoids race condition in cutlass path)
"$VENV_DIR/bin/pip" install nvidia-cudnn-cu13==9.19.1.2

echo ""
echo "==> Done. Venv at: $VENV_DIR"
echo "    Run with: ./scripts/run-sglang.sh"
