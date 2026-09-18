#!/usr/bin/env python
"""Replay recorded audio through the hearing pipeline, offline.

The reason this exists: the old pipeline could only be measured by reading
`logs/lloyd-agent-worker.log`, because none of it could be built without a
LiveKit room — and a log is what let 5 wake fires in 949 utterances go
unnoticed for three weeks. Everything in `agent-services/voice/` can now be
driven from a WAV, so a threshold change, a new wake model or a VAD setting is
a measurement rather than a guess.

Two subcommands:

  compare-wake   False accepts, old shape against new, on the diagnostic
                 corpus the worker writes (~/.lloyd/ww_diag/utterances/).
                 Old shape = reset the model, sweep a closed utterance. New
                 shape = one continuous stream. Utterances the log recorded as
                 a wake are excluded; nothing else in that corpus is one.

  run FILE...    Feed WAVs (a miss dump from /api/voice/ww_miss, a recording,
                 anything) through the full pipeline and print every event it
                 raises: wake, speech start, utterance with its transcript and
                 its Smart Turn verdict. `--lead` prepends room tone so the
                 wake word's one-time warmup is paid before the audio starts,
                 which is what a live stream looks like.

    python scripts/voice/replay.py compare-wake
    python scripts/voice/replay.py run ~/.lloyd/ww_diag/misses/*.wav --asr

2026-09-17, threshold 0.4, 496 non-wake utterances (0.17 h): the sweep false-
accepted 6 times, the stream 0 times.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-services"))

from voice.pipeline import HearingPipeline  # noqa: E402
from voice.resample import StreamResampler, to_int16  # noqa: E402
from voice.vad import SileroSegmenter  # noqa: E402
from voice.wake import FRAME_SAMPLES, WakeWordFactory  # noqa: E402

DIAG = Path.home() / ".lloyd" / "ww_diag"
MODELS = ROOT / "agent-services" / "models"


def load_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    r = StreamResampler(sr)
    return np.concatenate([r.push(x, sr), r.flush()])


def factory(threshold: float) -> WakeWordFactory:
    f = WakeWordFactory(models_dir=MODELS / "wakeword",
                        engine_dir=MODELS / "openwakeword", threshold=threshold)
    f.validate()
    return f


def room_tone(seconds: float, rng) -> np.ndarray:
    return rng.normal(0, 0.002, int(16000 * seconds)).astype(np.float32)


def compare_wake(args) -> int:
    scores = DIAG / "scores.jsonl"
    if not scores.exists():
        print(f"no diagnostic corpus at {DIAG}", file=sys.stderr)
        return 1
    fired = {json.loads(line)["utterance_id"] for line in scores.open()
             if line.strip() and json.loads(line).get("ww_fired")}
    files = [p for p in sorted((DIAG / "utterances").glob("*.wav"))
             if p.stem not in fired]
    clips = [load_16k(p) for p in files]
    hours = sum(c.size for c in clips) / 16000 / 3600
    rng = np.random.default_rng(0)
    print(f"{len(clips)} non-wake utterances, {hours:.2f} h of audio\n")
    print(f"{'threshold':>9}  {'sweep FA':>9}  {'stream FA':>9}")
    for th in args.thresholds:
        f = factory(th)
        swept = f.create()
        fa_old = 0
        for c in clips:
            swept._model.reset()
            best = 0.0
            for i in range(0, c.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
                scores_ = swept._model.predict(to_int16(c[i:i + FRAME_SAMPLES]))
                best = max(best, max(scores_.values()))
            fa_old += best >= th
        stream = f.create()
        fa_new = 0
        for c in clips:
            s = np.concatenate([room_tone(1.0, rng), c])
            for i in range(0, s.size, 160):
                fa_new += len(stream.feed(s[i:i + 160]))
        print(f"{th:>9.2f}  {fa_old:>4} {fa_old / hours:>4.0f}/h  "
              f"{fa_new:>4} {fa_new / hours:>4.0f}/h")
    return 0


def run(args) -> int:
    from voice import asr as voice_asr
    from voice.turn import SmartTurn

    f = factory(args.threshold)
    turn = SmartTurn(MODELS / "smart-turn" / "smart-turn-v3.2-cpu.onnx")
    rec = voice_asr.build_recognizer(
        {"backend": "parakeet", "model_dir": str(MODELS / "parakeet-tdt-v3")}
    ) if args.asr else None
    rng = np.random.default_rng(0)
    for path in args.files:
        audio = load_16k(Path(path))
        lead = room_tone(args.lead, rng)
        stream = np.concatenate([lead, audio, room_tone(1.0, rng)])
        pipe = HearingPipeline(16000, wake=f.create(),
                               segmenter=SileroSegmenter(threshold=args.vad))
        print(f"== {path}  ({audio.size / 16000:.2f}s)")
        for i in range(0, stream.size, 160):
            for ev in pipe.feed(stream[i:i + 160], 16000):
                t = (ev.sample - lead.size) / 16000
                if ev.kind == "wake":
                    print(f"  {t:6.2f}s  wake      {ev.detection.name} "
                          f"{ev.detection.score:.2f}")
                elif ev.kind == "speech":
                    print(f"  {t:6.2f}s  speech")
                elif ev.kind == "utterance":
                    u = ev.utterance
                    v = turn.predict(u.audio)
                    line = (f"  {t:6.2f}s  utterance {u.duration_s:.2f}s "
                            f"vad={u.max_prob:.2f} wake={'Y' if ev.wake else '-'} "
                            f"turn={'done' if v.complete else 'open'}:{v.probability:.2f}")
                    if rec is not None:
                        line += f"  {rec.transcribe(u.audio).text!r}"
                    print(line)
        peak = pipe.wake.take_peak() if pipe.wake else ("", 0.0)
        print(f"  peak wake score {peak[1]:.2f} ({peak[0] or '-'})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare-wake")
    c.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    r = sub.add_parser("run")
    r.add_argument("files", nargs="+")
    r.add_argument("--threshold", type=float, default=0.4)
    r.add_argument("--vad", type=float, default=0.45)
    r.add_argument("--lead", type=float, default=2.0)
    r.add_argument("--asr", action="store_true")
    args = ap.parse_args()
    return compare_wake(args) if args.cmd == "compare-wake" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
