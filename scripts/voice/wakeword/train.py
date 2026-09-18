#!/usr/bin/env python
"""Train Lloyd's wake word with livekit-wakeword, end to end.

Run with the TRAINING venv's python, never `.venvs/lloyd` — the pipeline pulls
its own torch, and on 2026-09-17 a single install into the shared venv
replaced production's (SETUP.md). See architecture/voice.md "Retraining the
wake word" for the environment, the espeak-ng unpack and the evaluation gate.

    WW_ROOT=~/.cache/lloyd-wakeword \\
      $WW_ROOT/livekit-wakeword/.venv/bin/python scripts/voice/wakeword/train.py \\
      scripts/voice/wakeword/hey_lloyd.yaml [--from extract]

Drives the six stages through the package's Python API rather than its `run`
command: upstream 95448a7 made `run_extraction`'s `sess_options` required and
`run` still calls it without one, so `run` dies after augmentation (the
one-line fix is in livekit-wakeword-sess-options.patch, beside this file).
`--from` resumes at a stage; each stage's output is on disk under
`output_dir`, so nothing before it is redone.
"""
import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

STAGES = ["generate", "augment", "extract", "train", "export", "eval"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--from", dest="start", choices=STAGES, default="generate")
    args = ap.parse_args()
    root = os.environ.get("WW_ROOT")
    if not root:
        print("set WW_ROOT (e.g. ~/.cache/lloyd-wakeword)", file=sys.stderr)
        return 2

    from livekit.wakeword import (load_config, run_augment, run_eval, run_export,
                                  run_extraction, run_generate, run_train)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    text = Path(args.config).read_text().replace("${WW_ROOT}", str(Path(root).expanduser()))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(text)
    cfg = load_config(fh.name)

    steps = {
        "generate": lambda: run_generate(cfg),
        "augment": lambda: run_augment(cfg),
        "extract": lambda: run_extraction(cfg, None),
        "train": lambda: run_train(cfg),
        "export": lambda: run_export(cfg),
        "eval": lambda: run_eval(cfg, cfg.model_output_dir / f"{cfg.model_name}.onnx"),
    }
    for name in STAGES[STAGES.index(args.start):]:
        logging.info("=== stage: %s ===", name)
        result = steps[name]()
        if name == "eval":
            logging.info("eval: %s", result)
    logging.info("ALL STAGES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
