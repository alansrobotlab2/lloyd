#!/bin/bash
# Setup script for ik_llama.cpp venv with multi-GPU CUDA support
# For Qwen3.5-397B across 2 GPUs

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venvs/ik_llama.cpp"

echo "=== Setting up ik_llama.cpp venv ==="

# Create venv if not exists
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# Activate and install packages
source "$VENV_DIR/bin/activate"

echo "Installing packages..."
pip install --upgrade pip
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90" pip install llama-cpp-python huggingface-hub --no-binary :all:

# Create wrapper script with proper CUDA paths
WRAPPER_SCRIPT="$PROJECT_DIR/bin/ik_llama.cpp-env.sh"
cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# Activate ik_llama.cpp venv with CUDA paths for multi-GPU support
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venvs/ik_llama.cpp"

export PATH="$VENV_DIR/bin:/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source "$VENV_DIR/bin/activate"
exec "$@"
EOF

chmod +x "$WRAPPER_SCRIPT"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To use the venv with CUDA:"
echo "  source $PROJECT_DIR/bin/ik_llama.cpp-env.sh"
echo ""
echo "Or use the wrapper script:"
echo "  $PROJECT_DIR/bin/ik_llama.cpp-env.sh bash"
echo ""
echo "For multi-GPU (2 GPUs):"
echo "  export CUDA_VISIBLE_DEVICES=0,1"
echo "  python -c \"import llama_cpp; print(llama_cpp.__version__)\""
echo ""
