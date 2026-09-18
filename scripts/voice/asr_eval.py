#!/usr/bin/env python
"""Word error rate and latency for each ASR backend, on a fixed corpus.

The reason this exists rather than a note in a doc: the engine is now a config
key, so "which one is better" is a question somebody will ask again the next
time a model is released, and the honest answer needs re-measuring rather than
remembering. It is also the only check that would catch a model directory
being half-downloaded — a truncated encoder still loads and still transcribes,
just badly.

Corpus: LibriSpeech validation.clean as parquet, which is read speech and so
flatters everything. Treat the absolute numbers as a floor and the *ordering*
as the result.

    python scripts/voice/asr_eval.py --corpus ~/.cache/lloyd-voice-eval/librispeech_dummy_clean_validation.parquet
    python scripts/voice/asr_eval.py --corpus … --backends parakeet whisper \
        --snr 10          # additive white noise, to rank robustness

Download the corpus once with:

    curl -sL -o ~/.cache/lloyd-voice-eval/librispeech_dummy_clean_validation.parquet \\
      https://huggingface.co/api/datasets/hf-internal-testing/librispeech_asr_dummy/parquet/clean/validation/0.parquet
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent-services"))

from voice import asr as voice_asr  # noqa: E402
from voice.resample import StreamResampler  # noqa: E402

_WORD = re.compile(r"[a-z']+")


def normalise(text: str) -> list[str]:
    """Lower-case, strip punctuation. Deliberately not a full text normaliser:
    LibriSpeech references are already upper-case words with no digits, so the
    usual number/abbreviation expansion has nothing to do here and would only
    add a second thing that can be wrong."""
    return _WORD.findall(text.lower().replace("-", " "))


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def decode_audio(blob: dict) -> tuple[np.ndarray, int]:
    """HF audio column: {'bytes': <wav/flac>, 'path': str}."""
    raw = blob["bytes"] if isinstance(blob, dict) else blob
    try:
        with wave.open(io.BytesIO(raw)) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return x.astype(np.float32) / 32768.0, sr
    except wave.Error:
        import soundfile as sf  # only needed for flac corpora

        x, sr = sf.read(io.BytesIO(raw), dtype="float32")
        return x, sr


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32)
    r = StreamResampler(sr)
    return np.concatenate([r.push(x, sr), r.flush()])


def add_noise(x: np.ndarray, snr_db: float, rng) -> np.ndarray:
    sig = float(np.mean(x * x)) or 1e-12
    noise_power = sig / (10 ** (snr_db / 10))
    return (x + rng.normal(0, np.sqrt(noise_power), x.size)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="parquet with audio + text")
    ap.add_argument("--backends", nargs="+", default=["parakeet", "whisper"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--snr", type=float, default=None,
                    help="add white noise at this SNR in dB")
    ap.add_argument("--models-root", default=None,
                    help="defaults to <repo>/agent-services/models")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    table = pq.read_table(args.corpus)
    rows = table.to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    root = Path(args.models_root or
                (Path(__file__).resolve().parents[2] / "agent-services" / "models"))
    rng = np.random.default_rng(0)

    clips = []
    for r in rows:
        x, sr = decode_audio(r["audio"])
        a = to_16k(x, sr)
        if args.snr is not None:
            a = add_noise(a, args.snr, rng)
        clips.append((a, r["text"]))
    total_audio = sum(a.size for a, _ in clips) / 16000
    label = "clean" if args.snr is None else f"SNR {args.snr:g} dB"
    print(f"{len(clips)} clips, {total_audio:.0f}s of audio, {label}\n")

    print(f"{'backend':<12} {'WER':>7} {'errors/words':>14} {'s/clip':>8} {'RTFx':>7}")
    for name in args.backends:
        cfg = {"backend": name}
        if name == "parakeet":
            cfg["model_dir"] = str(root / "parakeet-tdt-v3")
        rec = voice_asr.build_recognizer(cfg)
        rec.load()
        errs = ref_len = 0
        elapsed = 0.0
        for audio, ref in clips:
            t0 = time.perf_counter()
            hyp = rec.transcribe(audio).text
            elapsed += time.perf_counter() - t0
            r, h = normalise(ref), normalise(hyp)
            errs += edit_distance(r, h)
            ref_len += len(r)
        wer = errs / max(1, ref_len)
        print(f"{name:<12} {wer * 100:6.2f}% {errs:>6}/{ref_len:<7} "
              f"{elapsed / len(clips):7.3f} {total_audio / elapsed:6.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
