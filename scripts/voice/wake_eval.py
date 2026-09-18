#!/usr/bin/env python
"""Compare wake-word model sets the way the worker runs them.

Every candidate is scored through `voice.wake.ContinuousWakeWord` — the
worker's own runtime, fed 10 ms at a time after two seconds of room tone — so
a model that only works in some other harness cannot pass here. A model set is
a directory of `.onnx` classifiers, exactly what `livekit.acoustic_wake.models_dir`
points at; the worker fires on the best score across them.

Corpora (none committed — the room audio is from the house):

  held-out TTS   ~/.cache/lloyd-voice-eval/wake-tts/{pos,neg}_<voice>_<n>.wav
                 Qwen3-TTS voices, a different engine from anything a model was
                 trained on. Positives split into PREFIXED ("hey / hi / okay /
                 hello Lloyd…") and BARE ("Lloyd.", "Lloyd, …"), because a
                 prefixed-only model is not supposed to fire on the second.
  real room      ~/.lloyd/ww_diag/utterances/*.wav minus the utterances the
                 log recorded as a wake — false accepts on the house's own audio.
  LibriSpeech    the ASR eval corpus, played as one continuous stream — false
                 accepts per hour of speech that never says "Lloyd".

    python scripts/voice/wake_eval.py agent-services/models/wakeword  <candidate-dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-services"))
sys.path.insert(0, str(ROOT / "scripts" / "voice"))

from replay import load_16k  # noqa: E402
from voice.wake import WakeWordFactory  # noqa: E402

TTS = Path.home() / ".cache" / "lloyd-voice-eval" / "wake-tts"
DIAG = Path.home() / ".lloyd" / "ww_diag"
LIBRI = Path.home() / ".cache" / "lloyd-voice-eval" / "librispeech_dummy_clean_validation.parquet"
ENGINE = ROOT / "agent-services" / "models" / "openwakeword"
BARE = {0, 4}          # "Lloyd." and "Lloyd, what time is it?"
SR = 16000


def room(seconds, rng):
    return rng.normal(0, 0.002, int(SR * seconds)).astype(np.float32)


def peak(factory, audio, rng):
    """Best score the stream reaches over this clip, and per-model bests."""
    w = factory.create()
    per: dict[str, float] = {}
    w._on_score = lambda name, score, cur: per.__setitem__(name, max(per.get(name, 0.0), score))
    stream = np.concatenate([room(2.0, rng), audio, room(1.0, rng)])
    for i in range(0, stream.size, 160):
        w.feed(stream[i:i + 160])
    return max(per.values(), default=0.0), per


def fires_in_stream(factory, clips, threshold, rng, gap=1.0):
    """Detections over one continuous stream of clips, as the worker would see
    a long stretch of audio — refractory window and all."""
    w = factory.create()
    w.threshold = threshold
    n = 0
    for c in clips:
        s = np.concatenate([room(gap, rng), c])
        for i in range(0, s.size, 160):
            n += len(w.feed(s[i:i + 160]))
    return n


def libri_clips():
    if not LIBRI.exists():
        return []
    import pyarrow.parquet as pq

    from asr_eval import decode_audio, to_16k

    return [to_16k(*decode_audio(r["audio"])) for r in pq.read_table(LIBRI).to_pylist()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dirs", nargs="+")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    args = ap.parse_args()

    pos, neg = [], []
    for p in sorted(TTS.glob("*.wav")):
        m = re.match(r"(pos|neg)_(.+)_(\d+)\.wav$", p.name)
        if m:
            (pos if m.group(1) == "pos" else neg).append((m.group(2), int(m.group(3)), load_16k(p)))
    fired = set()
    if (DIAG / "scores.jsonl").exists():
        fired = {json.loads(line)["utterance_id"] for line in (DIAG / "scores.jsonl").open()
                 if line.strip() and json.loads(line).get("ww_fired")}
    real = [load_16k(p) for p in sorted((DIAG / "utterances").glob("*.wav")) if p.stem not in fired]
    libri = libri_clips()
    real_h = sum(c.size for c in real) / SR / 3600
    libri_h = sum(c.size for c in libri) / SR / 3600
    n_pre = sum(1 for _, i, _ in pos if i not in BARE)
    n_bare = len(pos) - n_pre
    voices = sorted({v for v, _, _ in pos})
    print(f"held-out TTS: {len(voices)} voices, {n_pre} prefixed + {n_bare} bare positives, "
          f"{len(neg)} near-misses | real room: {len(real)} utts ({real_h:.2f} h) | "
          f"LibriSpeech: {len(libri)} clips ({libri_h * 60:.1f} min)\n")

    for d in args.model_dirs:
        f = WakeWordFactory(models_dir=d, engine_dir=ENGINE, threshold=0.5)
        f.validate()
        names = [Path(p).stem for p in f._paths]
        rng = np.random.default_rng(0)
        pos_s = [(v, i, *peak(f, a, rng)) for v, i, a in pos]
        neg_s = [(v, i, *peak(f, a, rng)) for v, i, a in neg]
        print(f"== {d}  ({', '.join(names)})")
        print(f"{'thresh':>7} {'prefixed':>10} {'bare':>8} {'near-miss FA':>13} "
              f"{'real FA':>9} {'libri FA/h':>11}")
        for th in args.thresholds:
            rp = sum(1 for _, i, s, _ in pos_s if i not in BARE and s >= th)
            rb = sum(1 for _, i, s, _ in pos_s if i in BARE and s >= th)
            fn = sum(1 for _, _, s, _ in neg_s if s >= th)
            fr = fires_in_stream(f, real, th, np.random.default_rng(1))
            fl = fires_in_stream(f, libri, th, np.random.default_rng(2), gap=0.3) if libri else 0
            print(f"{th:>7.2f} {rp:>5}/{n_pre:<4} {rb:>4}/{n_bare:<3} {fn:>8}/{len(neg):<4} "
                  f"{fr:>9} {fl / max(libri_h, 1e-9):>11.1f}")
        misses = [(v, i, round(s, 2)) for v, i, s, _ in pos_s if i not in BARE and s < 0.5]
        loud = [(v, i, round(s, 2)) for v, i, s, _ in neg_s if s >= 0.3]
        print(f"  prefixed misses at 0.5: {misses}")
        print(f"  near-misses scoring >= 0.3: {loud}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
