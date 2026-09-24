"""Seeded-canvas determinism probe (#1361).

scripts/djev_determinism_probe.py sends a bare /v1/completions with no
diffusion_seed_canvas, so every request starts from torch.randint's canvas
(diffusion_gemma.py init_canvas) and the logprobs move with the RNG. Production
(structured_server.one_read) always seeds the canvas and reads once
(diffusion_read_only). This sends that shape: fixed prompt, fixed seeded canvas,
read-only, and compares the top-k logprobs at EVERY canvas position across
byte-identical requests, warm (same prompt, prefix cache) and cold (fresh
cache_salt). Also runs the unseeded shape as a control.

Prints one JSON line.
"""
import json
import random
import sys
import urllib.request
import uuid

URL = "http://127.0.0.1:8010/v1/completions"
W = 16
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PROMPT = "".join(
    f"determinism probe line {i:04d}: the quick brown fox jumps over the "
    f"lazy dog, and the kernel reads the same canvas twice.\n" for i in range(160))
rng = random.Random(42)
SEED_CANVAS = [rng.randrange(1000, 200000) for _ in range(W)]


def call(seeded: bool, salt: str | None):
    body = {"model": "djev", "prompt": PROMPT, "max_tokens": W, "logprobs": 5}
    if seeded:
        body["vllm_xargs"] = {"diffusion_seed_canvas": SEED_CANVAS, "diffusion_canvas_length": W,
                              "diffusion_max_steps": 1, "diffusion_read_only": True}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return [list(p.items()) for p in d["choices"][0]["logprobs"]["top_logprobs"]]


def delta(a, b):
    worst, changed = 0.0, False
    for pa, pb in zip(a, b):
        for (_, x), (_, y) in zip(pa, pb):
            worst = max(worst, abs(x - y))
        da = dict(pa)
        for t, y in pb:
            if t in da:
                worst = max(worst, abs(da[t] - y))
        changed = changed or [t for t, _ in pa] != [t for t, _ in pb]
    return worst, changed


out = {}
for seeded in (True, False):
    for regime in ("warm", "cold"):
        base, worst, changed = None, 0.0, False
        for i in range(RUNS):
            salt = f"seeded-probe-{uuid.uuid4().hex}" if regime == "cold" else None
            rows = call(seeded, salt)
            if base is None:
                base = rows
                continue
            d, c = delta(base, rows)
            worst, changed = max(worst, d), changed or c
        out[f"{'seeded' if seeded else 'unseeded'}_{regime}"] = {"max_nats": round(worst, 4), "topk_changed": changed,
                                                                  "positions": len(base)}
print(json.dumps(out))
