"""The one statement of the request shape a production djev read has (#2116).

djev is a diffusion model: a `/v1/completions` read starts from a canvas of
token ids, and with no `diffusion_seed_canvas` the engine draws that canvas from
`torch.randint` (`vllm .../models/diffusion_gemma.py::init_canvas`). Two
byte-identical requests then differ before a kernel runs, because the canvas
they condition on is different. Production never reads that way —
`structured_server.one_read` always seeds the canvas, asks for one step and sets
read-only, so the logprobs it scores come off a fixed input.

This module is that shape, extracted out of `eval/djev/seeded_canvas_probe.py`
so the graded instrument (`scripts/djev_determinism_probe.py`) and the eval
instrument send the same bytes rather than each keeping its own copy of four
engine arguments. #2116 is the item that extracted it: with the shape in only one
of the two probes, the graded one measured canvas RNG and exited non-zero on the
one boot whose production-shaped reads repeat.

DO NOT EDIT THE NUMBERS HERE CASUALLY. `CANVAS_SEED`, the `randrange` bounds and
`CANVAS_READ_POSITIONS` are the exact shape every 2026-09-24 seeded number in
`agent-services/bin/start-djev.sh`'s header and
`eval/djev/kernel_bisect_2026-09-24.md` was measured with; changing one restates
those rows.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

#: Positions in one production read. `structured_server.one_read` reads a whole
#: canvas at once, so `max_tokens` and the number of scored positions are this
#: number — not 1. A probe that scores only position 0 is scoring a sixth of what
#: production scores (#2116 clause 2).
CANVAS_READ_POSITIONS = 16

#: The seed the fixed canvas is drawn from. Any fixed seed makes the input
#: reproducible; this one is the seed the measurements above were taken with.
CANVAS_SEED = 42

#: Canvas token ids are drawn from this range, matching what the eval probe has
#: always drawn: a plausible vocabulary id rather than a small integer.
CANVAS_ID_RANGE = (1000, 200000)


def seed_canvas(positions: int = CANVAS_READ_POSITIONS,
                seed: int = CANVAS_SEED) -> list[int]:
    """The fixed canvas a seeded read conditions on: `positions` token ids.

    Deterministic by construction — `random.Random(seed)` is a private generator,
    so nothing here depends on process state, and two calls return the same list.
    """
    rng = random.Random(seed)
    low, high = CANVAS_ID_RANGE
    return [rng.randrange(low, high) for _ in range(positions)]


def seeded_read_xargs(canvas: Sequence[int],
                      positions: int = CANVAS_READ_POSITIONS) -> dict[str, Any]:
    """The `vllm_xargs` that turn a bare completion into a production read.

    `diffusion_max_steps: 1` with `diffusion_read_only: true` is how production
    reads a canvas it supplies: one step, no sampling, so the returned logprobs
    are a function of the seeded input alone. Passing the canvas length explicitly
    is what stops the engine's boot-level `--diffusion-config` (the shipped boot
    runs `canvas_length: 128`) from deciding how many positions come back.
    """
    return {
        "diffusion_seed_canvas": list(canvas),
        "diffusion_canvas_length": positions,
        "diffusion_max_steps": 1,
        "diffusion_read_only": True,
    }
