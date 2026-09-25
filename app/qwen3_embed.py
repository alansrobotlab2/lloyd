"""Qwen3-Embedding-0.6B query embedder in plain torch, CPU (#1486).

The production venv has torch and `tokenizers` but neither `transformers` nor
`safetensors`, and the only other process holding this model — the qmd daemon —
exposes no embedding endpoint. So this is the model's forward pass written out
(28 decoder layers: RMSNorm, GQA attention with per-head q/k RMSNorm and RoPE,
SwiGLU MLP; last-token pooling, L2 normalised) over weights read straight out of
the `model.safetensors` file. It is checked against the transformers
implementation by `tests/test_entity_semantic_seeds.py::test_qwen3_embed_*`
when both are available (cosine ≥ 0.999).

One query is a few dozen tokens, so there is no padding, no batching and no KV
cache: a plain causal forward. Weights load once per process, lazily, on the
first call — nothing is loaded unless `retrieval.entity_seeding.semantic` is on.
"""
from __future__ import annotations

import glob
import json
import math
import os
import struct
import threading
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_lock = threading.Lock()
_model = None


def model_dir() -> Path:
    """`retrieval.entity_seeding.semantic.model_dir`, else the HF cache snapshot."""
    try:
        from app.config import CONFIG
        d = ((((CONFIG or {}).get("retrieval") or {}).get("entity_seeding") or {})
             .get("semantic") or {}).get("model_dir")
        if d:
            return Path(os.path.expanduser(str(d)))
    except Exception:  # noqa: BLE001
        pass
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = sorted(glob.glob(str(hub / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / "*")))
    if not snaps:
        raise FileNotFoundError(f"{MODEL_ID} is not in {hub}")
    return Path(snaps[-1])


_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def _read_safetensors(path: Path) -> dict:
    import torch
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        dt = getattr(torch, _DTYPES[meta["dtype"]])
        t = torch.frombuffer(bytearray(raw[base + s:base + e]), dtype=dt).reshape(meta["shape"])
        out[name] = t.to(torch.float32)
    return out


class _Qwen3:
    def __init__(self, d: Path):
        import torch
        from tokenizers import Tokenizer
        cfg = json.loads((d / "config.json").read_text())
        self.cfg = cfg
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        w = _read_safetensors(d / "model.safetensors")
        self.w = {k[len("model."):] if k.startswith("model.") else k: v for k, v in w.items()}
        self.L = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg["rope_theta"])
        self.torch = torch

    def _rms(self, x, w):
        t = self.torch
        return x * t.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def _rope(self, x, cos, sin):
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        rot = self.torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    def embed(self, text: str):
        t = self.torch
        ids = self.tok.encode(text).ids[:512]
        n = len(ids)
        with t.no_grad():
            x = self.w["embed_tokens.weight"][t.tensor(ids)]
            pos = t.arange(n, dtype=t.float32)
            inv = 1.0 / (self.theta ** (t.arange(0, self.hd, 2, dtype=t.float32) / self.hd))
            f = t.outer(pos, inv)
            emb = t.cat((f, f), dim=-1)
            cos, sin = emb.cos()[None], emb.sin()[None]
            mask = t.full((n, n), float("-inf")).triu(1)
            rep = self.nh // self.nkv
            for i in range(self.L):
                p = f"layers.{i}."
                h = self._rms(x, self.w[p + "input_layernorm.weight"])
                q = (h @ self.w[p + "self_attn.q_proj.weight"].T).view(n, self.nh, self.hd).transpose(0, 1)
                k = (h @ self.w[p + "self_attn.k_proj.weight"].T).view(n, self.nkv, self.hd).transpose(0, 1)
                v = (h @ self.w[p + "self_attn.v_proj.weight"].T).view(n, self.nkv, self.hd).transpose(0, 1)
                q = self._rms(q, self.w[p + "self_attn.q_norm.weight"])
                k = self._rms(k, self.w[p + "self_attn.k_norm.weight"])
                q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
                k = k.repeat_interleave(rep, dim=0)
                v = v.repeat_interleave(rep, dim=0)
                a = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd) + mask
                o = (a.softmax(-1) @ v).transpose(0, 1).reshape(n, self.nh * self.hd)
                x = x + o @ self.w[p + "self_attn.o_proj.weight"].T
                h = self._rms(x, self.w[p + "post_attention_layernorm.weight"])
                g = t.nn.functional.silu(h @ self.w[p + "mlp.gate_proj.weight"].T)
                u = h @ self.w[p + "mlp.up_proj.weight"].T
                x = x + (g * u) @ self.w[p + "mlp.down_proj.weight"].T
            x = self._rms(x, self.w["norm.weight"])
            v = x[-1]
            return (v / v.norm()).numpy()


def _get():
    global _model
    with _lock:
        if _model is None:
            _model = _Qwen3(model_dir())
        return _model


def embed_query(text: str):
    """Unit vector for `text` (the caller adds any instruction prefix)."""
    return _get().embed(text)
