#!/usr/bin/env bash
set -euo pipefail

# One-time setup for qmd (full-text + vector search CLI).
# Installs qmd via bun, creates the obsidian collection, downloads
# GGUF models, and generates vector embeddings on GPU.
#
# Prerequisites: bun (https://bun.sh)
#
# After running this, mem_search in tool_services.py is ready to use.
#
# Services installed:
#   lloyd-qmd-daemon.service  — HTTP MCP server on port 8181
#   lloyd-qmd-watcher.service — inotifywait auto-reindex on vault changes

QMD="$HOME/.bun/bin/qmd"
VAULT="$HOME/obsidian"

# CUDA runtime libs needed by node-llama-cpp inside distrobox
export LD_LIBRARY_PATH="/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "=== QMD Search Setup ==="

# 1. Install qmd
if command -v "$QMD" &>/dev/null; then
    echo "qmd already installed: $($QMD --version)"
else
    echo "Installing qmd via bun..."
    bun install -g @tobilu/qmd
    echo "Installed: $($QMD --version)"
fi

# 2. Create obsidian collection (idempotent — skips if exists)
if "$QMD" collection list 2>/dev/null | grep -q "obsidian"; then
    echo "Collection 'obsidian' already exists."
else
    if [ ! -d "$VAULT" ]; then
        echo "WARNING: Obsidian vault not found at $VAULT"
        echo "Create or symlink it, then re-run this script."
        exit 1
    fi
    echo "Creating 'obsidian' collection from $VAULT..."
    "$QMD" collection add "$VAULT" --name obsidian --mask "**/*.md"
fi

# 3. Index the collection
echo "Indexing..."
"$QMD" update

# 4. Generate vector embeddings (downloads embedding model on first run)
echo "Generating vector embeddings (GPU-accelerated)..."
"$QMD" embed

# 5. Trigger query expansion + reranker model downloads
echo "Downloading query expansion and reranker models..."
"$QMD" query "test" -c obsidian -n 1 --json >/dev/null 2>&1 || true

# 6. Patch qmd wrapper to pin GPU 0 (survives bun updates need re-patch)
QMD_SCRIPT="$HOME/.bun/install/global/node_modules/@tobilu/qmd/qmd"
if [ -f "$QMD_SCRIPT" ] && ! grep -q 'CUDA_VISIBLE_DEVICES' "$QMD_SCRIPT"; then
    echo "Patching qmd wrapper for GPU 0..."
    sed -i '/^exec "\$NODE"/i # Use GPU 0\nexport CUDA_DEVICE_ORDER=PCI_BUS_ID\nexport CUDA_VISIBLE_DEVICES=0\n' "$QMD_SCRIPT"
else
    echo "qmd wrapper already patched for GPU 0."
fi

# 7. Verify
echo ""
echo "Status:"
"$QMD" status

echo ""
echo "Done. QMD ready for GPU-accelerated search."
echo "Services to enable:"
echo "  systemctl --user enable --now lloyd-qmd-daemon lloyd-qmd-watcher"
