#!/usr/bin/env python
"""Synthesise wake-word clips in voices Piper does not have, on the CPU.

livekit-wakeword's Piper generator blends LibriTTS speakers; this adds two
other engines through sherpa-onnx (already in `.venvs/lloyd`, nothing to
install): Kokoro v1.0 (53 voices, 28 of them English, the rest accented
English) and Piper-VCTK (109 British-Isles speakers). Output is 16 kHz mono
int16 with the silence trimmed, ready for train.py's `lloyd.extras`.

    python scripts/voice/wakeword/synth_extras.py --out $WW_ROOT/extra/x/pos \\
        --phrase "hey lloyd" --phrase "hi lloyd" --per-phrase all

Models (sherpa-onnx `tts-models` release, unpacked under $WW_ROOT/extra-tts):
kokoro-multi-lang-v1_0, vits-piper-en_GB-vctk-medium.

`--per-phrase all` renders every voice x speed; an integer draws that many
random (voice, speed) pairs per phrase, for negatives where breadth of
phrasing matters more than every voice saying each one.
"""
from __future__ import annotations

import argparse
import os
import random
import wave
from pathlib import Path

import numpy as np

KOKORO_SPEEDS = (0.85, 1.0, 1.2)
VCTK_SPEEDS = (0.9, 1.15)


def _trim(a: np.ndarray, sr: int, thresh_db: float = -40.0, pad_s: float = 0.03) -> np.ndarray:
    if a.size == 0:
        return a
    frame = int(sr * 0.01)
    n = a.size // frame
    if n == 0:
        return a
    rms = np.sqrt(np.mean(a[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    on = np.where(db > thresh_db)[0]
    if on.size == 0:
        return a
    pad = int(sr * pad_s)
    return a[max(0, on[0] * frame - pad): min(a.size, (on[-1] + 1) * frame + pad)]


def _to16k(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return a
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(sr, 16000)
    return resample_poly(a, 16000 // g, sr // g).astype(np.float32)


def engines(root: Path, threads: int):
    import sherpa_onnx

    k = root / "kokoro-multi-lang-v1_0"
    kok = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(k / "model.onnx"), voices=str(k / "voices.bin"),
                tokens=str(k / "tokens.txt"), data_dir=str(k / "espeak-ng-data"),
                dict_dir=str(k / "dict"),
                lexicon=f"{k / 'lexicon-us-en.txt'},{k / 'lexicon-zh.txt'}"),
            num_threads=threads, provider="cpu")))
    v = root / "vits-piper-en_GB-vctk-medium"
    vctk = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(v / "en_GB-vctk-medium.onnx"), tokens=str(v / "tokens.txt"),
                data_dir=str(v / "espeak-ng-data"), noise_scale=0.8, noise_scale_w=0.9),
            num_threads=threads, provider="cpu")))
    combos = [("kokoro", kok, sid, sp) for sid in range(kok.num_speakers) for sp in KOKORO_SPEEDS]
    combos += [("vctk", vctk, sid, sp) for sid in range(vctk.num_speakers) for sp in VCTK_SPEEDS]
    return combos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--phrase", action="append", default=[])
    ap.add_argument("--phrases-file")
    ap.add_argument("--per-phrase", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--root", default=os.path.expanduser("~/.cache/lloyd-wakeword/extra-tts"))
    args = ap.parse_args()
    phrases = list(args.phrase)
    if args.phrases_file:
        phrases += [ln.strip() for ln in Path(args.phrases_file).read_text().splitlines() if ln.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    combos = engines(Path(args.root), args.threads)
    rng = random.Random(args.seed)
    n = 0
    for pi, phrase in enumerate(phrases):
        pick = combos if args.per_phrase == "all" else rng.sample(combos, int(args.per_phrase))
        for eng, tts, sid, sp in pick:
            path = out / f"{eng}_{sid:03d}_{sp:.2f}_{pi:03d}.wav"
            if path.exists():
                n += 1
                continue
            # Capitalised, punctuated text reads as speech rather than a word list.
            text = phrase[0].upper() + phrase[1:] + ("" if phrase[-1] in ".?!" else ".")
            g = tts.generate(text, sid=sid, speed=sp)
            a = _trim(_to16k(np.asarray(g.samples, np.float32), g.sample_rate), 16000)
            if a.size < 1600:
                continue
            a = a / (np.abs(a).max() + 1e-9) * rng.uniform(0.3, 0.9)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((a * 32767).astype(np.int16).tobytes())
            n += 1
        print(f"{phrase!r}: {len(pick)} renders (total {n})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
