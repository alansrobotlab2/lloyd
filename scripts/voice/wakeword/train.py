#!/usr/bin/env python
"""Train one of Lloyd's wake words with livekit-wakeword, end to end.

Run with the TRAINING venv's python, never `.venvs/lloyd` — the pipeline pulls
its own torch, and on 2026-09-17 a single install into the shared venv
replaced production's (SETUP.md). See architecture/voice.md "Retraining the
wake word" for the environment, the espeak-ng unpack and the evaluation gate.

    WW_ROOT=~/.cache/lloyd-wakeword \\
      $WW_ROOT/livekit-wakeword/.venv/bin/python scripts/voice/wakeword/train.py \\
      scripts/voice/wakeword/hey_lloyd_v2.yaml [--from extract] [--to extras]

Drives the stages through the package's Python API rather than its `run`
command: upstream 95448a7 made `run_extraction`'s `sess_options` required and
`run` still calls it without one, so `run` dies after augmentation (the
one-line fix is in livekit-wakeword-sess-options.patch, beside this file).
`--from` resumes at a stage; each stage's output is on disk under
`output_dir`, so nothing before it is redone.

A config may carry a `lloyd:` block the package never sees (it is stripped
before `load_config`, whose schema would refuse it):

  shuffle_speakers: true
      Piper's generator walks speaker pairs in order — (0,0), (0,1), … — so
      25 000 clips blend only speakers 0–27 with the rest. Shuffled, every
      clip draws a random pair from all 904 LibriTTS speakers.
  extras: [{dir, kind, rounds?, augment?}, …]
      Audio from outside Piper, featurised and folded into the training
      arrays by the `extras` stage (after `extract`, before `train`):
        kind: positive          short clips of the phrase, end-aligned and
                                augmented `rounds` times like Piper's
        kind: negative          short clips, centre-padded, augmented
        kind: negative_stream   long audio (real room, Lloyd's own voice) cut
                                into every 16-embedding window the runtime
                                would see (160 ms hop); `augment: true` adds
                                one RIR + background copy of each file
      Every file must be 16 kHz mono (the helper scripts write them so).
      The package's own arrays are kept as `*.base.npy`, so re-running the
      stage never folds an extra in twice.
"""
import argparse
import itertools
import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

STAGES = ["generate", "augment", "extract", "extras", "train", "export", "eval"]
LOG = logging.getLogger("train")


class _ShuffledPairs:
    """itertools, except `product(range(n), range(n))` is a seeded permutation.

    Patched into `livekit.wakeword.data.piper.synthesis` only; every other
    `product` (the settings grid) is untouched. Seeded, so the generator's
    resume (which consumes the same iterator to skip done clips) still lines up.
    """

    def __init__(self, seed: int = 1234):
        self._seed = seed

    def __getattr__(self, name):
        return getattr(itertools, name)

    def product(self, *iters):
        if len(iters) == 2 and all(isinstance(i, range) for i in iters):
            pairs = list(itertools.product(*iters))
            random.Random(self._seed).shuffle(pairs)
            return iter(pairs)
        return itertools.product(*iters)


def _features_for_stream(audio, mel, emb, hop: int = 2) -> np.ndarray:
    e = emb.extract_embeddings(mel(audio.astype(np.float32)[None, :]))[0]
    if e.shape[0] < 16:
        return np.zeros((0, 16, 96), np.float32)
    return np.stack([e[i - 15:i + 1] for i in range(15, e.shape[0], hop)]).astype(np.float32)


def run_extras(cfg, extras: list[dict]) -> None:
    import soundfile as sf
    from livekit.wakeword.data.augment import AudioAugmentor, _augment_directory
    from livekit.wakeword.data.features import extract_features_from_directory
    from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
    from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path

    mel = MelSpectrogramFrontend(onnx_path=get_mel_model_path())
    emb = SpeechEmbedding(onnx_path=get_embedding_model_path())
    aug = AudioAugmentor(
        background_paths=[Path(p) for p in cfg.augmentation.background_paths],
        rir_paths=[Path(p) for p in cfg.augmentation.rir_paths],
    )
    random.seed(7)
    out: dict[str, list[np.ndarray]] = {"positive": [], "negative": []}
    work = cfg.model_output_dir / "extras"
    for ex in extras:
        src = Path(ex["dir"]).expanduser()
        kind = ex["kind"]
        wavs = sorted(src.glob("*.wav"))
        if not wavs:
            raise SystemExit(f"extras: no wavs in {src}")
        if kind in ("positive", "negative"):
            # The package's augmenter keys on clip_NNNNNN.wav names, so stage a copy.
            d = work / src.name
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            for i, w in enumerate(wavs):
                shutil.copy(w, d / f"clip_{i:06d}.wav")
            for r in range(int(ex.get("rounds", cfg.augmentation.rounds))):
                _augment_directory(d, aug, is_positive=(kind == "positive"), round_idx=r,
                                   target_duration_s=cfg.augmentation.clip_duration)
            feats = extract_features_from_directory(d, mel, emb)
            out[kind].append(feats)
        elif kind == "negative_stream":
            # One continuous stream, as the worker hears it: most room
            # utterances are shorter than the model's 1.28 s window, so cut
            # alone they would yield no window at all.
            rng = np.random.default_rng(5)
            parts = []
            for w in wavs:
                a, sr = sf.read(str(w), dtype="float32")
                if a.ndim > 1:
                    a = a[:, 0]
                if sr != 16000:
                    raise SystemExit(f"extras: {w} is {sr} Hz; write 16 kHz")
                parts += [rng.normal(0, 0.002, 4000).astype(np.float32), a]
            stream = np.concatenate(parts)
            chunks = [_features_for_stream(stream, mel, emb)]
            if ex.get("augment"):
                b = aug.mix_with_background(aug.apply_rir(stream, p=1.0), snr_db_range=(5.0, 20.0))
                chunks.append(_features_for_stream(b, mel, emb))
            feats = np.concatenate(chunks)
            out["negative"].append(feats)
        else:
            raise SystemExit(f"extras: unknown kind {kind!r}")
        LOG.info("extras: %s (%s) -> %s windows", src, kind, feats.shape[0])

    rng = np.random.default_rng(11)
    for kind, name in (("positive", "positive_features_train"),
                       ("negative", "negative_features_train")):
        path = cfg.model_output_dir / f"{name}.npy"
        base = cfg.model_output_dir / f"{name}.base.npy"
        if not base.exists():
            shutil.copy(path, base)
        arrs = [np.load(base)] + out[kind]
        merged = np.concatenate(arrs).astype(np.float32)
        # The trainer walks each array in order, so an unshuffled tail of
        # extras would arrive as one block every epoch.
        merged = merged[rng.permutation(merged.shape[0])]
        np.save(path, merged)
        LOG.info("extras: %s = %d base + %d extra", path.name, arrs[0].shape[0],
                 merged.shape[0] - arrs[0].shape[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--from", dest="start", choices=STAGES, default="generate")
    ap.add_argument("--to", dest="stop", choices=STAGES, default=STAGES[-1])
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
    raw = yaml.safe_load(text)
    lloyd = raw.pop("lloyd", None) or {}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(raw, fh)
    cfg = load_config(fh.name)

    # GPU 0 is shared with the live TTS and qmd. Piper's generator grew to
    # 7.3 GB of cached allocations over 30 000 clips at batch 8 (2026-09-24),
    # though a clip needs ~1 GB: cap the process so the allocator frees its
    # cache instead of taking the card.
    cap_gb = float(os.environ.get("LLOYD_WW_GPU_MEM_GB", "0") or 0)
    if cap_gb > 0:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb / total), 0)
            logging.info("GPU cache capped at %.1f GiB of %.1f", cap_gb, total)

    if lloyd.get("shuffle_speakers"):
        from livekit.wakeword.data.piper import synthesis
        synthesis.it = _ShuffledPairs()

    steps = {
        "generate": lambda: run_generate(cfg),
        "augment": lambda: run_augment(cfg),
        "extract": lambda: run_extraction(cfg, None),
        "extras": lambda: run_extras(cfg, lloyd.get("extras") or []),
        "train": lambda: run_train(cfg),
        "export": lambda: run_export(cfg),
        "eval": lambda: run_eval(cfg, cfg.model_output_dir / f"{cfg.model_name}.onnx"),
    }
    for name in STAGES[STAGES.index(args.start):STAGES.index(args.stop) + 1]:
        logging.info("=== stage: %s ===", name)
        result = steps[name]()
        if name == "eval":
            logging.info("eval: %s", result)
    logging.info("STAGES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
