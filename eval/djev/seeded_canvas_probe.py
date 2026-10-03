"""Seeded-canvas determinism probe (#1361).

scripts/djev_determinism_probe.py used to send a bare /v1/completions with no
diffusion_seed_canvas, so every request started from torch.randint's canvas
(diffusion_gemma.py init_canvas) and the logprobs moved with the RNG. Production
(structured_server.one_read) always seeds the canvas and reads once
(diffusion_read_only). This sends that shape: fixed prompt, fixed seeded canvas,
read-only, and compares the top-k logprobs at EVERY canvas position across
byte-identical requests, warm (same prompt, prefix cache) and cold (fresh
cache_salt). Also runs the unseeded shape as a control.

Prints one JSON line.

#2116 moved the request shape into `scripts/djev_canvas_shape.py` — the graded
probe and this one now build the seeded read from the same code — and put this
file's own work under `main()`. Before that, importing this module fired live
requests to the engine on :8010 and printed them, which made the shared shape
impossible to import from anywhere a test runs.
"""
from __future__ import annotations

import json
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.djev_canvas_shape import (  # noqa: E402  (needs the ROOT bootstrap above)
    CANVAS_READ_POSITIONS, seed_canvas, seeded_read_xargs)

URL = "http://127.0.0.1:8010/v1/completions"
W = CANVAS_READ_POSITIONS
PROMPT = "".join(
    f"determinism probe line {i:04d}: the quick brown fox jumps over the "
    f"lazy dog, and the kernel reads the same canvas twice.\n" for i in range(160))
SEED_CANVAS = seed_canvas()


def call(seeded: bool, salt: str | None, *, url: str = URL,
         timeout: int = 60) -> list[list[tuple[str, float]]]:
    """One read: the top-k at every canvas position, in the order returned."""
    body: dict[str, Any] = {"model": "djev", "prompt": PROMPT,
                            "max_tokens": W, "logprobs": 5}
    if seeded:
        body["vllm_xargs"] = seeded_read_xargs(SEED_CANVAS, W)
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return [list(p.items()) for p in d["choices"][0]["logprobs"]["top_logprobs"]]


def delta(a: Sequence, b: Sequence) -> tuple[float, bool]:
    """(worst |delta| nats over every position, did any position's top-k change)."""
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


def main(runs: int, *, url: str = URL) -> dict[str, dict[str, Any]]:
    """Replay each shape in each cache regime and return the four worst deltas."""
    out: dict[str, dict[str, Any]] = {}
    for seeded in (True, False):
        for regime in ("warm", "cold"):
            base, worst, changed = None, 0.0, False
            for _ in range(runs):
                salt = f"seeded-probe-{uuid.uuid4().hex}" if regime == "cold" else None
                rows = call(seeded, salt, url=url)
                if base is None:
                    base = rows
                    continue
                d, c = delta(base, rows)
                worst, changed = max(worst, d), changed or c
            key = f"{'seeded' if seeded else 'unseeded'}_{regime}"
            out[key] = {"max_nats": round(worst, 4), "topk_changed": changed,
                        "positions": len(base)}
    return out


if __name__ == "__main__":
    print(json.dumps(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)))
