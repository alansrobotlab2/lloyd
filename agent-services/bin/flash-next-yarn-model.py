#!/usr/bin/env python3
"""Build a YaRN-extended shadow of the Qwen3.8-Flash-Next checkpoint.

    flash-next-yarn-model.py --factor 2.0 [--src DIR] [--dst DIR]

Writes DIR-yarn<factor>/ next to the checkpoint: every file symlinked to the
original except config.json, which is rewritten with the YaRN rope block the
model card prescribes (text_config.rope_parameters). Then boot with

    MODEL_DIR=<dst> MAX_MODEL_LEN=524288 bin/start-qwen38-flash-next.sh

WHY A SHADOW CONFIG AND NOT --hf-overrides
  vLLM applies a *dict* --hf-overrides to the target model only:
  SpeculativeConfig.compose_draft_hf_overrides forwards callables to the MTP
  draft's config and deliberately drops dicts ("target-specific key patches").
  This model's draft is a full QSA layer reading rope_parameters from the same
  text_config, so an --hf-overrides YaRN block leaves the drafter on plain
  RoPE with a 262,144 horizon while the target runs past it — speculation
  degrades past native and nothing in the log says so. A config.json reaches
  both. (The YaRN recipe that used to sit in start-qwen38-flash-next.sh's
  comments had the wrong shape as well: top level rather than text_config, and
  no mrope_section.)

WHAT YaRN COSTS
  Static: the scaling applies to every request, and the model card warns it
  "potentially impact[s] performance on shorter texts", recommending factor
  2.0 if 524k is the real need rather than 4.0. Only 12 of the 48 layers use
  RoPE, on a quarter of each head (partial_rotary_factor 0.25), so the hit is
  probably small — but it is unmeasured on this box. Keep it as an opt-in arm
  until it is.

WHAT IT CANNOT DO
  The KV pool bounds max_model_len: vLLM refuses a length one request cannot
  hold. On one 96 GiB card that is ~398k tokens in BF16 and ~692k in FP8
  (11.5 GiB pool, MTP on), so 1M is not reachable on one card at any factor;
  factor 2.0 (524,288) fits in FP8 with 1.3x concurrency.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

DEFAULT_SRC = pathlib.Path.home() / "lloyd/agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4"
NATIVE = 262144


def yarn_block(factor: float) -> dict:
    # Straight from the model card's vLLM/SGLang snippet, factor substituted.
    return {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "yarn",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "factor": factor,
        "original_max_position_embeddings": NATIVE,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factor", type=float, required=True, help="YaRN factor; 2.0 -> 524,288, 4.0 -> 1,048,576")
    ap.add_argument("--src", type=pathlib.Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=pathlib.Path, default=None)
    ap.add_argument("--force", action="store_true", help="rebuild an existing dst")
    args = ap.parse_args()

    src: pathlib.Path = args.src.resolve()
    if not (src / "config.json").is_file():
        sys.exit(f"no config.json under {src}")
    tag = f"yarn{args.factor:g}"
    dst: pathlib.Path = args.dst or src.with_name(f"{src.name}-{tag}")
    if dst.exists():
        if not args.force:
            sys.exit(f"{dst} exists; pass --force to rebuild")
        for p in dst.iterdir():
            p.unlink()
        dst.rmdir()
    dst.mkdir(parents=True)

    cfg = json.loads((src / "config.json").read_text())
    text = cfg.get("text_config")
    if not isinstance(text, dict):
        sys.exit("config.json has no text_config block; this is not the Flash-Next layout")
    native = int(text.get("max_position_embeddings", NATIVE))
    if native != NATIVE:
        print(f"note: text_config.max_position_embeddings is {native}, not {NATIVE}", file=sys.stderr)
    native_block = dict(text.get("rope_parameters") or {})
    text["rope_parameters"] = yarn_block(args.factor)
    # vLLM derives the allowed max_model_len for yarn as
    # original_max_position_embeddings * factor, so this field can stay native.
    #
    # The OUTER config must keep the native block. vLLM's Qwen4ExpConfig sets
    # self.rope_parameters = (explicit kwarg) or text_config.rope_parameters
    # BEFORE calling the transformers constructor, whose dataclass
    # __post_init__ then runs standardize_rope_params on the outer object; for
    # rope_type "yarn" that does
    #   rope_parameters.setdefault("original_max_position_embeddings",
    #                              self.max_position_embeddings)
    # evaluating self.max_position_embeddings eagerly even though the key is
    # present — and the outer config has no such attribute, nor can a
    # top-level config.json key become one before that hook runs:
    #   AttributeError: 'Qwen4ExpConfig' object has no attribute 'max_position_embeddings'
    # Found by booting the first shadow on 2026-09-10. So the shadow pins the
    # outer mirror to what it is today (the native block) via the explicit
    # kwarg, and only text_config carries yarn. Nothing in the text path reads
    # the outer block: the QSA attention and the MTP draft build their rope
    # from Qwen4ExpTextConfig, and max_model_len is derived from hf_text_config.
    if native_block.get("rope_type", "default") != "default":
        sys.exit(f"source config already has rope_type {native_block.get('rope_type')!r}; refusing to stack scalings")
    cfg["rope_parameters"] = native_block or {"rope_type": "default", "rope_theta": text.get("rope_theta", 10000000)}

    for entry in sorted(src.iterdir()):
        if entry.name == "config.json":
            continue
        os.symlink(entry, dst / entry.name)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    horizon = int(NATIVE * args.factor)
    print(f"wrote {dst}")
    print(f"  rope: yarn factor {args.factor:g}, original {NATIVE} -> horizon {horizon}")
    print(f"  boot: MODEL_DIR={dst} MAX_MODEL_LEN={horizon} bash agent-services/bin/start-qwen38-flash-next.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
