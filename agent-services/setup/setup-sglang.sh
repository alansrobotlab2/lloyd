#!/usr/bin/env bash
set -euo pipefail
#
# Rebuild the sglang venv with the latest stack:
#   sglang 0.5.11 + torch 2.11.0+cu130 + flashinfer-python 0.6.8.post1 (JIT)
#
# cu130 matches the CUDA 13.2 system toolchain.
# flashinfer-python uses apache-tvm-ffi JIT (no pre-built cu-specific wheel needed).
#
# Usage: bash agent-services/setup/setup-sglang.sh
#
# GPU target: either GPU (SM 12.0 Blackwell).

UV="${HOME}/.local/bin/uv"
VENV_DIR="${HOME}/lloyd/.venvs/sglang"
PYTHON_VER="3.11"

echo "==> Removing old venv at $VENV_DIR"
rm -rf "$VENV_DIR"

echo "==> Creating fresh venv (Python ${PYTHON_VER})"
"$UV" venv "$VENV_DIR" --python "$PYTHON_VER" --seed

UVPIP="$UV pip install --python $VENV_DIR/bin/python"

echo "==> Installing torch 2.11.0+cu130"
$UVPIP \
    torch==2.11.0+cu130 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

echo "==> Installing sglang 0.5.11 with all extras"
$UVPIP "sglang[all]==0.5.11" --prerelease=allow

echo "==> Verifying install"
"$VENV_DIR/bin/python" -c "
import sglang, torch
print(f'sglang {sglang.__version__}')
print(f'torch  {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name}  SM {p.major}.{p.minor}  {p.total_memory//1024**3} GiB')
"

echo "==> Done. venv at $VENV_DIR"
