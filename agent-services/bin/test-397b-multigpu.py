#!/usr/bin/env python3
"""
Test loading Qwen3.5-397B-A17B-UD-IQ1_M with multi-GPU support.
Uses llama-cpp-python with tensor splitting across 2 GPUs.
"""

import os
import sys

# Add venv to path if not already activated
venv_path = os.path.expanduser("~/lloyd/agent-services/.venvs/ik_llama.cpp")
if os.path.exists(f"{venv_path}/bin/python"):
    sys.path.insert(0, f"{venv_path}/lib/python3.11/site-packages")

# Set CUDA env vars
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from llama_cpp import Llama

MODEL_DIR = os.path.expanduser("~/lloyd/agent-services/llm/models/Qwen3.5-397B-A17B-UD-IQ1_M")
MODEL_GGUF = os.path.join(MODEL_DIR, "Qwen3.5-397B-A17B-UD-IQ1_M-00001-of-00004.gguf")

# For multi-GPU with llama-cpp-python:
# - tensor_split: How to split weights across GPUs (sum should be 1.0)
# - n_gpu_layers: Number of layers to offload (999 = all layers)
# - n_ctx: Context window size

print("Loading model...")
print(f"Model: {MODEL_GGUF}")
print(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

try:
    llm = Llama(
        model_path=MODEL_GGUF,
        n_gpu_layers=999,  # Offload all layers to GPU
        tensor_split=[0.5, 0.5],  # Split evenly across 2 GPUs
        n_ctx=8192,  # Context window
        verbose=True,
    )
    
    print("\nModel loaded successfully!")
    print("\nTesting inference...")
    
    output = llm(
        "Hello, how are you?",
        max_tokens=50,
        stop=["\n"],
        echo=True,
    )
    
    print(f"\nResponse: {output['choices'][0]['text']}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
