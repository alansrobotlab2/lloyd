#!/usr/bin/env python
"""VAD bake-off: Silero (production) against FireRedVAD and TEN VAD.

Every candidate's per-frame speech probability is driven through the same
state machine as `voice.vad.SileroSegmenter` (onset at `threshold`, exit at
`threshold - 0.15`, close after `min_silence_ms` below it, `speech_pad_ms` of
pre-roll, `min_utterance_ms` floor), so the comparison is of the models, not of
their vendors' post-processing. The simulation's onset and close *decisions*
equal `SileroSegmenter`'s sample for sample (checked on a 305 s stream: same
segment count at 250 and 380 ms); audio bounds agree to within one frame.

The number that matters is the speech->silence decision time at an EQUAL
mid-utterance split rate. A lower threshold or a shorter `min_silence_ms`
always closes sooner and always splits more; a model only wins if it closes
sooner at the same number of splits, without more false segments.

Data (built by `build`):
  * LibriSpeech test-clean utterances (2-14 s, 400 of them), scaled to active
    speech rms 0.03, concatenated with 200-1500 ms gaps over room tone cut
    from the quietest frames of the real-room corpus. Ground truth is the
    clean signal's 10 ms energy within 35 dB of its 95th percentile, so onset,
    offset and every internal pause are known. `--noise-gain` adds more of the
    same room tone (4 -> ~18 dB SNR, 9 -> ~11 dB).
  * ESC-50 folds 1-2 minus the human non-speech classes (640 clips, 64 min)
    at the same level over room tone: every segment there is a false one.
  * The real-room diagnostic corpus (`app.ww_diag`), each clip with 0.6 s of
    room tone either side. `--asr` labels it with Parakeet: >= 2 words is
    speech (missed / split), no words is non-speech (false).

    python scripts/voice/vad_eval.py build --librispeech ~/data/LibriSpeech/test-clean \\
        --esc50 ~/data/ESC-50-master --out /tmp/vadbake --asr
    python scripts/voice/vad_eval.py run --data /tmp/vadbake \\
        --firered-dir /tmp/vadbake/firered-vad [--ten]

FireRedVAD (Apache-2.0) ships PyTorch weights only. `export-firered` turns the
official `Stream-VAD` checkpoint into one ONNX graph (feat [1,T,80] + a stacked
FSMN cache [8,1,128,19] -> probs, new cache) plus `cmvn.json`; it needs torch
and `fireredvad`, so run it from a scratch venv, never the lloyd one. The
Kaldi fbank here is numpy (povey window, pre-emphasis 0.97, 512-point FFT, 80
mel bins 20 Hz-Nyquist), equal to kaldi-native-fbank to 1.3e-4, and the ONNX
path streamed in 3-frame chunks equals the official torch model to 5e-4.

TEN VAD needs the `ten-vad` package, whose `libten_vad.so` links libc++ /
libc++abi — not on this host (conda-forge `libcxx` + `libcxxabi` on
LD_LIBRARY_PATH works). sherpa-onnx's TEN binding answers only a thresholded
boolean, which the hysteresis here cannot use.

`gate` runs the real SileroSegmenter + Smart Turn + the worker's hold over the
same streams and counts cutoffs inside an utterance per `min_silence_ms`:

    python scripts/voice/vad_eval.py gate --data /tmp/vadbake --turn-ms 20 \
        --asr-model-dir ~/lloyd/agent-services/models/parakeet-tdt-v3

2026-09-24 result: no candidate wins, and 250 ms raises mid-sentence cutoffs
7.8% -> 17.5%; see architecture/voice.md "Segmentation".
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-services"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from voice.resample import StreamResampler  # noqa: E402

SR = 16000
#: ESC-50 classes that are a person making a sound; a VAD firing on a laugh is
#: not a false alarm in the sense measured here.
HUMAN = {"crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
         "laughing", "brushing_teeth", "snoring", "drinking_sipping"}
DEFAULT_THRESHOLDS = {"silero": [0.45, 0.5], "firered": [0.5, 0.6, 0.7], "ten": [0.6, 0.7, 0.8]}
MIN_SILENCE = [150, 200, 250, 300, 380, 500]


# --- audio ---------------------------------------------------------------------

def load16(path: Path) -> np.ndarray:
    """Any WAV/FLAC as float32 16 kHz mono."""
    if path.suffix == ".wav":
        with wave.open(str(path)) as w:
            sr, ch = w.getframerate(), w.getnchannels()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    else:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="int16")
        if x.ndim > 1:
            x = x.mean(axis=1).astype(np.int16)
    if sr != SR:
        r = StreamResampler(sr)
        x = np.concatenate([r.push(x, sr), r.flush()])
    x = np.asarray(x)
    return x.astype(np.float32) / 32768 if x.dtype == np.int16 else x.astype(np.float32)


# --- models: each returns (probs, decide_at) --------------------------------------
# decide_at[k] is the stream sample at which frame k's probability exists.

class SileroProbs:
    frame, hop = 512, 512

    def __call__(self, x):
        from voice.vad import SileroOnnx
        m = SileroOnnx()
        n = x.size // 512
        p = np.array([m(x[i * 512:(i + 1) * 512]) for i in range(n)], np.float32)
        return p, 512 * (np.arange(n) + 1)


class KaldiFbank:
    """kaldi-native-fbank's defaults with dither 0 (FireRedVAD's front end)."""

    def __init__(self, bins: int = 80):
        n = 400
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))) ** 0.85
        mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
        ml, mh = mel(20.0), mel(SR / 2)
        delta = (mh - ml) / (bins + 1)
        m = mel(SR / 512 * np.arange(256))
        W = np.zeros((bins, 257))
        for b in range(bins):
            lo, c, hi = ml + b * delta, ml + (b + 1) * delta, ml + (b + 2) * delta
            w = np.maximum(0, np.minimum((m - lo) / (c - lo), (hi - m) / (hi - c)))
            w[(m <= lo) | (m >= hi)] = 0
            W[b, :256] = w
        self.W = W

    def __call__(self, x_int16_scale: np.ndarray) -> np.ndarray:
        x = np.asarray(x_int16_scale, np.float64)
        if x.size < 400:
            return np.zeros((0, 80), np.float32)
        nf = 1 + (x.size - 400) // 160
        fr = x[np.arange(400)[None, :] + 160 * np.arange(nf)[:, None]]
        fr = fr - fr.mean(axis=1, keepdims=True)
        pre = fr.copy()
        pre[:, 1:] -= 0.97 * fr[:, :-1]
        pre[:, 0] -= 0.97 * fr[:, 0]
        spec = np.abs(np.fft.rfft(pre * self.win, n=512)) ** 2
        return np.log(np.maximum(spec @ self.W.T, np.finfo(np.float32).eps)).astype(np.float32)


class FireRedProbs:
    frame, hop = 400, 160

    def __init__(self, model_dir: Path):
        import onnxruntime as ort
        c = json.loads((model_dir / "cmvn.json").read_text())
        self.mu = np.array(c["means"], np.float32)
        self.istd = np.array(c["inv_std"], np.float32)
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_dir / "firered_stream_vad.onnx"), o,
                                         providers=["CPUExecutionProvider"])
        self.fb = KaldiFbank()

    def __call__(self, x, chunk: int = 64):
        # The model is causal (lookback-only FSMN), so a whole-stream fbank run
        # in chunks with the cache carried is identical to frame-by-frame.
        f = ((self.fb(x * 32768) - self.mu) * self.istd).astype(np.float32)
        cache = np.zeros((8, 1, 128, 19), np.float32)
        out = []
        for i in range(0, len(f), chunk):
            p, cache = self.sess.run(None, {"feat": f[None, i:i + chunk], "cache": cache})
            out.append(p[0])
        p = np.concatenate(out).astype(np.float32)
        return p, 400 + 160 * np.arange(p.size)


class TenProbs:
    frame, hop = 256, 256

    def __call__(self, x):
        from ten_vad import TenVad
        v = TenVad(hop_size=256)
        xi = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        n = xi.size // 256
        p = np.array([v.process(np.ascontiguousarray(xi[i * 256:(i + 1) * 256]))[0] for i in range(n)],
                     np.float32)
        return p, 256 * (np.arange(n) + 1)


# --- the segmenter, on any frame rate ---------------------------------------------

def segment(p, d, frame, hop, thr, min_sil_ms, pad_ms=220, min_utt_ms=250, max_utt_ms=30000, hyst=0.15):
    ex = max(0.0, thr - hyst)
    min_sil, pad = min_sil_ms * SR // 1000, pad_ms * SR // 1000
    min_utt, max_utt = min_utt_ms * SR // 1000, max_utt_ms * SR // 1000
    out, trig, run, prev_end = [], False, 0, 0
    start = onset = 0
    for k in range(p.size):
        t = int(d[k])
        if not trig:
            if p[k] >= thr:
                trig, run, onset = True, 0, t
                start = max(prev_end, t - frame - pad)
            continue
        run = 0 if p[k] >= ex else run + hop
        closed, capped = run >= min_sil, (t - start) >= max_utt
        if not (closed or capped):
            continue
        end = t - max(0, run - pad) if closed else t
        seg = dict(start=start, end=end, onset=onset, close=t, capped=not closed)
        trig, prev_end = False, t
        if capped:
            trig, start, onset, run = True, end, t, 0
        if end - seg["start"] >= min_utt:
            out.append(seg)
    if trig and int(d[-1]) - start >= min_utt:  # the worker flushes on leave
        out.append(dict(start=start, end=int(d[-1]), onset=onset, close=int(d[-1]), capped=True))
    return out


# --- build --------------------------------------------------------------------------

def _room_tone(clips, seconds, rng):
    F = 512
    rms = []
    for x in clips:
        n = x.size // F
        rms.append(np.sqrt((x[: n * F].reshape(n, F) ** 2).mean(1)))
    thr = np.percentile(np.concatenate(rms), 15)
    pieces = []
    for x, r in zip(clips, rms):
        q, i = r < thr, 0
        while i < q.size:
            if not q[i]:
                i += 1
                continue
            j = i
            while j < q.size and q[j]:
                j += 1
            if j - i >= 8:
                pieces.append(x[i * F + F: j * F - F])
            i = j
    fade = 160
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    total = int(seconds * SR)
    y = np.zeros(total + 20 * SR, np.float32)
    pos = 0
    while pos < total:
        pc = pieces[rng.integers(len(pieces))].copy()
        if pc.size < 3 * fade:
            continue
        pc[:fade] *= ramp
        pc[-fade:] *= ramp[::-1]
        y[pos:pos + pc.size] += pc
        pos += pc.size - fade
    return y[:total], len(pieces)


def cmd_build(a):
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng, nrng = random.Random(1234), np.random.default_rng(1234)
    corpus = Path(a.corpus).expanduser() if a.corpus else None
    if corpus is None:
        from app.ww_diag import diag_dir
        corpus = diag_dir()
    wavs = sorted((corpus / "utterances").glob("*.wav"))
    clips = {f.stem: load16(f) for f in wavs}
    np.savez_compressed(out / "corpus.npz", **clips)
    bed, npieces = _room_tone(list(clips.values()), 20 * 60, nrng)
    np.save(out / "room_tone.npy", bed)
    print(f"corpus {len(clips)} clips; room tone from {npieces} quiet runs, rms {np.sqrt((bed ** 2).mean()):.4f}")
    if a.asr:
        import yaml
        from voice.asr import SherpaOfflineRecognizer
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["livekit"]["stt"]
        md = Path(a.asr_model_dir or cfg["model_dir"]).expanduser()
        rec = SherpaOfflineRecognizer(md if md.is_absolute() else ROOT / md, num_threads=4)
        labels = {k: rec.transcribe(x).text for k, x in clips.items()}
        (out / "corpus_asr.json").write_text(json.dumps(labels, indent=0))

    utts = []
    files = sorted(Path(a.librispeech).expanduser().glob("*/*/*.flac"))
    rng.shuffle(files)
    for f in files:
        x = load16(f)
        if not 2.0 <= x.size / SR <= 14.0:
            continue
        n = x.size // 160
        e = 10 * np.log10((x[: n * 160].reshape(n, 160) ** 2).mean(1) + 1e-12)
        m = e > np.percentile(e, 95) - 35
        on, off = int(np.argmax(m)), int(m.size - np.argmax(m[::-1]))
        pauses, i = [], on
        while i < off:
            if m[i]:
                i += 1
                continue
            j = i
            while j < off and not m[j]:
                j += 1
            pauses.append((j - i) * 10)
            i = j
        act = x[on * 160: off * 160][np.repeat(m[on:off], 160)]
        utts.append((f.stem, (x * 0.03 / np.sqrt((act ** 2).mean())).astype(np.float32), on * 160, off * 160, pauses))
        if len(utts) >= 400:
            break
    for s in range(10):
        pos, parts, gts = SR, [], []
        for uid, x, on, off, pauses in utts[s * 40:(s + 1) * 40]:
            gap = int(rng.uniform(0.2, 1.5) * SR)
            gts.append(dict(uid=uid, on=pos + on, off=pos + off, pauses_ms=pauses))
            parts.append((pos, x))
            pos += x.size + gap
        y = np.zeros(pos + SR, np.float32)
        for p0, x in parts:
            y[p0:p0 + x.size] += x
        o = int(nrng.integers(0, bed.size - y.size))
        np.save(out / f"ls_{s}.npy", y + bed[o:o + y.size])
        (out / f"ls_{s}.json").write_text(json.dumps(gts))
    allp = [p for u in utts for p in u[4]]
    print(f"librispeech {len(utts)} utterances; internal pauses >=200 ms {sum(p >= 200 for p in allp)}, "
          f">=250 {sum(p >= 250 for p in allp)}, >=380 {sum(p >= 380 for p in allp)}")

    esc = Path(a.esc50).expanduser()
    meta = [m for m in csv.DictReader(open(esc / "meta/esc50.csv"))
            if m["category"] not in HUMAN and m["fold"] in ("1", "2")]
    pos, parts, cats = SR, [], []
    for m in meta:
        x = load16(esc / "audio" / m["filename"])
        act = x[np.abs(x) > 0.1 * np.abs(x).max() + 1e-9]
        if act.size == 0:
            continue
        x = x * (0.03 / (np.sqrt((act ** 2).mean()) + 1e-9))
        parts.append((pos, x))
        cats.append(dict(cat=m["category"], start=pos, end=pos + x.size))
        pos += x.size + SR
    y = np.zeros(pos, np.float32)
    for p0, x in parts:
        y[p0:p0 + x.size] += x
    tone, _ = _room_tone(list(clips.values()), pos / SR, nrng)
    np.save(out / "esc.npy", y + tone)
    (out / "esc.json").write_text(json.dumps(cats))
    print(f"esc50 {len(parts)} non-speech clips, {pos / SR / 60:.0f} min")


# --- run ----------------------------------------------------------------------------

def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def score_streams(probs, data, model, thr, ms):
    r = dict(utts=0, missed=0, splits=0, merged=0, false=0, onset=[], offset=[])
    for s in range(10):
        gts = json.loads((data / f"ls_{s}.json").read_text())
        segs = segment(*probs[f"ls_{s}"], model.frame, model.hop, thr, ms)
        for i, g in enumerate(gts):
            r["utts"] += 1
            ov = [x for x in segs if x["start"] < g["off"] and x["end"] > g["on"]]
            if not ov:
                r["missed"] += 1
                continue
            r["splits"] += len(ov) - 1
            if ov[0]["onset"] >= g["on"] - 3200:
                r["onset"].append((ov[0]["onset"] - g["on"]) / SR * 1000)
            nxt = gts[i + 1]["on"] if i + 1 < len(gts) else None
            if nxt is None or ov[-1]["end"] <= nxt:
                if not ov[-1]["capped"]:
                    r["offset"].append((ov[-1]["close"] - g["off"]) / SR * 1000)
            elif nxt - g["off"] >= (ms + 300) * SR // 1000:
                r["merged"] += 1
        r["false"] += sum(1 for x in segs
                          if not any(x["start"] < g["off"] + 3200 and x["end"] > g["on"] - 3200 for g in gts))
    return r


def cmd_run(a):
    data = Path(a.data).expanduser()
    models = {"silero": SileroProbs()}
    if a.firered_dir:
        models["firered"] = FireRedProbs(Path(a.firered_dir).expanduser())
    if a.ten:
        models["ten"] = TenProbs()
    corpus = np.load(data / "corpus.npz")
    bed = np.load(data / "room_tone.npy")
    asr = json.loads((data / "corpus_asr.json").read_text()) if (data / "corpus_asr.json").exists() else None
    rng = np.random.default_rng(7)
    rows = []
    for name, model in models.items():
        probs, cpu = {}, 0.0
        for s in range(10):
            x = np.load(data / f"ls_{s}.npy")
            if a.noise_gain:
                o = int(rng.integers(0, bed.size - x.size))
                x = x + a.noise_gain * bed[o:o + x.size]
            t0 = time.process_time()
            probs[f"ls_{s}"] = model(x)
            cpu += time.process_time() - t0
        probs["esc"] = model(np.load(data / "esc.npy"))
        cprobs = {k: model(np.concatenate([bed[:9600], corpus[k], bed[9600:19200]])) for k in corpus.files}
        esc_cats = json.loads((data / "esc.json").read_text())
        esc_h = probs["esc"][1][-1] / SR / 3600
        for thr in (a.thresholds or DEFAULT_THRESHOLDS[name]):
            for ms in MIN_SILENCE:
                r = score_streams(probs, data, model, thr, ms)
                esc = segment(*probs["esc"], model.frame, model.hop, thr, ms)
                row = dict(model=name, thr=thr, ms=ms, utts=r["utts"], missed=r["missed"], splits=r["splits"],
                           merged=r["merged"], false_gaps=r["false"],
                           onset_p50=_pct(r["onset"], 50), offset_p50=_pct(r["offset"], 50),
                           offset_p90=_pct(r["offset"], 90), esc_false_per_h=len(esc) / esc_h,
                           esc_clips_hit=sum(1 for c in esc_cats
                                             if any(x["start"] < c["end"] and x["end"] > c["start"] for x in esc)),
                           cpu_ms_per_s=cpu / (sum(p[1][-1] for k, p in probs.items() if k != "esc") / SR) * 1000)
                if asr is not None:
                    n = {k: len(segment(*cprobs[k], model.frame, model.hop, thr, ms)) for k in corpus.files}
                    sp = [k for k, v in asr.items() if len(v.split()) >= 2]
                    ns = [k for k, v in asr.items() if not v.split()]
                    row.update(corpus_speech=len(sp), corpus_missed=sum(n[k] == 0 for k in sp),
                               corpus_extra_splits=sum(max(0, n[k] - 1) for k in sp),
                               corpus_nonspeech=len(ns), corpus_false_clips=sum(n[k] > 0 for k in ns))
                rows.append(row)
    hdr = ("model", "thr", "ms", "splits", "merged", "false_gaps", "onset_p50", "offset_p50", "offset_p90",
           "esc_false_per_h", "corpus_missed", "corpus_extra_splits", "corpus_false_clips")
    print(" ".join(f"{h:>10.10s}" for h in hdr))
    for r in rows:
        print(" ".join(f"{r.get(h, ''):>10.2f}" if isinstance(r.get(h), float) else f"{str(r.get(h, '')):>10.10s}"
                       for h in hdr))
    # The comparison that decides: offset lag at Silero's own split counts.
    base = {r["ms"]: r["splits"] for r in rows if r["model"] == "silero" and r["thr"] == 0.45}
    print("\noffset p50/p90 at the split count Silero 0.45 has at 380 ms and at 250 ms:")
    for key in sorted({(r["model"], r["thr"]) for r in rows}):
        g = sorted((r for r in rows if (r["model"], r["thr"]) == key), key=lambda r: r["splits"])
        sp = np.array([r["splits"] for r in g], float)
        cells = []
        for target in (base.get(380), base.get(250)):
            if target is None or not sp.min() <= target <= sp.max():
                cells.append("        n/a        ")
                continue
            f = lambda k: np.interp(target, sp, [r[k] for r in g])  # noqa: E731
            cells.append(f"ms {f('ms'):4.0f} {f('offset_p50'):4.0f}/{f('offset_p90'):4.0f}")
        print(f"  {key[0]:8s} {key[1]:.2f}  | {cells[0]} | {cells[1]}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


# --- gate: false cutoffs end to end, with Smart Turn holding ------------------------

HOLD_POLL_S = 0.125  #: `_flush_held_loop` polls every 250 ms; the mean wait past a deadline


def simulate_gate(x, gts, seg, smart_turn, hold_timeout_s=2.2, hold_max_s=12.0, turn_ms=None):
    """The worker's gate over one stream, on stream time.

    Mirrors `RoomBridge`: an utterance closes (the segmenter's cursor at that
    frame), anything held is glued in front (`_take_held`), Smart Turn judges
    the joined audio unless the close was a length cap, an unfinished verdict
    is held with a deadline `hold_timeout` after the verdict (`_hold`), and
    the flush loop releases a held turn at its deadline whatever the speaker is
    doing — it does not wait for a VAD that is mid-utterance. Returns commits
    as dicts: time (s), segs (list of (start, end) samples), how.
    """
    commits, held = [], None  # held: dict(audio=[...], segs=[...], deadline=s, first_close=s)
    for i in range(0, x.size - 511, 512):
        for u in seg.feed(x[i:i + 512]):
            close = seg.cursor / SR
            if held is not None and held["deadline"] <= close:
                commits.append(dict(time=held["deadline"] + HOLD_POLL_S, segs=held["segs"], how="timeout"))
                held = None
            audio = u.audio if held is None else np.concatenate(held["audio"] + [u.audio])
            segs = ([] if held is None else held["segs"]) + [(u.start_sample, u.end_sample)]
            held_s = 0.0 if held is None else sum(a.size for a in held["audio"]) / SR
            held = None
            if u.reason == "max_duration":
                commits.append(dict(time=close, segs=segs, how="cap"))
                continue
            v = smart_turn.predict(audio)
            # A fixed verdict cost when given: the wall time of a nice-19 run on
            # a shared box is load, not Smart Turn.
            t = close + (v.elapsed_ms if turn_ms is None else turn_ms) / 1000
            if not v.complete and held_s < hold_max_s:
                held = dict(audio=[audio], segs=segs, deadline=t + hold_timeout_s)
            else:
                commits.append(dict(time=t, segs=segs, how="complete"))
        if held is not None and held["deadline"] <= seg.cursor / SR:
            commits.append(dict(time=held["deadline"] + HOLD_POLL_S, segs=held["segs"], how="timeout"))
            held = None
    if held is not None:
        commits.append(dict(time=held["deadline"] + HOLD_POLL_S, segs=held["segs"], how="timeout"))
    return commits


def score_gate(commits, gts):
    """(a) false cutoffs: a commit whose last segment is not the last one over
    the ground-truth utterance it ends in — the sentence was cut. (b) true
    ends committed on their own. (c) true ends that were held first: released
    by the timeout, or glued onto the next utterance (two turns sent as one)."""
    all_segs = sorted({sg for c in commits for sg in c["segs"]})
    def utts_of(sg):
        return [j for j, g in enumerate(gts) if sg[0] < g["off"] and sg[1] > g["on"]]
    last_seg_of = {}
    for sg in all_segs:
        for j in utts_of(sg):
            last_seg_of[j] = max(last_seg_of.get(j, sg), sg)
    r = dict(utts=len(gts), missed=len(gts) - len(last_seg_of), false_cutoffs=0, spurious=0,
             true_end_complete=0, true_end_timeout=0, true_end_merged=0, latency=[], latency_timeout=[],
             cutoff_spans=[], cutoff_how=[], rescued=0)
    for c in commits:
        last = c["segs"][-1]
        us = utts_of(last)
        if not us:
            r["spurious"] += 1
            continue
        if last_seg_of[us[-1]] != last:
            r["false_cutoffs"] += 1
            r["cutoff_spans"].append((c["segs"][0][0], last[1]))
            r["cutoff_how"].append(c["how"])
        # a held segment glued to its continuation inside the same utterance
        r["rescued"] += sum(1 for a, b in zip(c["segs"], c["segs"][1:])
                            if set(utts_of(a)) & set(utts_of(b)))
    for j, sg in last_seg_of.items():
        c = next(c for c in commits if sg in c["segs"])
        lat = (c["time"] - gts[j]["off"] / SR) * 1000
        if c["segs"][-1] != sg:
            r["true_end_merged"] += 1
        elif c["how"] == "timeout":
            r["true_end_timeout"] += 1
            r["latency"].append(lat)
            r["latency_timeout"].append(lat)
        else:
            r["true_end_complete"] += 1
            r["latency"].append(lat)
    return r


def cmd_gate(a):
    from voice.turn import SmartTurn
    from voice.vad import SileroSegmenter
    data = Path(a.data).expanduser()
    st = SmartTurn(model_path=a.smart_turn, threshold=a.turn_threshold)
    st.load()
    probe = np.random.default_rng(0).normal(0, 0.03, 3 * SR).astype(np.float32)
    turn_ms = a.turn_ms if a.turn_ms is not None else float(np.median([st.predict(probe).elapsed_ms for _ in range(20)]))
    print(f"smart turn verdict cost used: {turn_ms:.1f} ms", flush=True)
    rec = None
    if a.asr_model_dir:
        from voice.asr import SherpaOfflineRecognizer
        rec = SherpaOfflineRecognizer(Path(a.asr_model_dir).expanduser(), num_threads=4)
    bed = np.load(data / "room_tone.npy")
    out = []
    for gain in a.noise_gains:
        rng = np.random.default_rng(7)
        streams = []
        for s in range(10):
            x = np.load(data / f"ls_{s}.npy")
            if gain:
                o = int(rng.integers(0, bed.size - x.size))
                x = x + gain * bed[o:o + x.size]
            streams.append((x, json.loads((data / f"ls_{s}.json").read_text())))
        for ms in a.min_silence:
            tot = None
            for x, gts in streams:
                seg = SileroSegmenter(threshold=a.vad_threshold, min_silence_ms=ms)
                r = score_gate(simulate_gate(x, gts, seg, st, a.hold_timeout_ms / 1000, turn_ms=turn_ms), gts)
                # Parakeet punctuates: a cut whose text ends in . ? ! fell on a
                # sentence boundary inside a LibriSpeech utterance (which holds
                # several sentences); anything else is a mid-sentence cutoff.
                r["cut_at_sentence_end"] = sum(
                    1 for (b, e) in r["cutoff_spans"]
                    if rec is not None and rec.transcribe(x[b:e]).text.strip().endswith((".", "?", "!")))
                r["cut_by_timeout"] = r["cutoff_how"].count("timeout")
                if tot is None:
                    tot = r
                else:
                    for k, v in r.items():
                        tot[k] = tot[k] + v
            n = tot["utts"]
            row = dict(noise_gain=gain, min_silence_ms=ms, utts=n, missed=tot["missed"],
                       false_cutoffs=tot["false_cutoffs"], false_cutoff_rate=tot["false_cutoffs"] / n,
                       cut_at_sentence_end=tot["cut_at_sentence_end"] if rec else None,
                       mid_sentence_cutoffs=(tot["false_cutoffs"] - tot["cut_at_sentence_end"]) if rec else None,
                       cut_by_timeout=tot["cut_by_timeout"], rescued_by_hold=tot["rescued"],
                       spurious=tot["spurious"], true_end_complete=tot["true_end_complete"],
                       true_end_timeout=tot["true_end_timeout"], true_end_merged=tot["true_end_merged"],
                       commit_p50=_pct(tot["latency"], 50), commit_p90=_pct(tot["latency"], 90),
                       commit_p50_complete_only=_pct([v for v in tot["latency"] if v not in tot["latency_timeout"]], 50))
            out.append(row)
            mid = row["mid_sentence_cutoffs"]
            print(f"gain {gain:>3} min_silence {ms:4d}: cutoffs inside an utterance {row['false_cutoffs']:3d} "
                  f"({100 * row['false_cutoff_rate']:.1f}% of {n}; mid-sentence "
                  f"{'-' if mid is None else f'{mid} = {100 * mid / n:.1f}%'}; by timeout {row['cut_by_timeout']}; "
                  f"splits rescued by a hold {row['rescued_by_hold']}) | true ends: complete {row['true_end_complete']:3d}, "
                  f"held->timeout {row['true_end_timeout']:3d}, merged into next {row['true_end_merged']:3d} | "
                  f"commit latency p50/p90 {row['commit_p50']:5.0f}/{row['commit_p90']:5.0f} ms "
                  f"(complete-only p50 {row['commit_p50_complete_only']:4.0f}) | spurious {row['spurious']} missed {row['missed']}",
                  flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


def cmd_export_firered(a):
    import torch
    from fireredvad.core.audio_feat import CMVN
    from fireredvad.core.detect_model import DetectModel
    src, out = Path(a.src).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    m = DetectModel.from_pretrained(str(src)).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, feat, cache):
            h, new = self.m.dfsmn(feat, caches=[cache[i] for i in range(8)])
            return torch.sigmoid(self.m.out(h)).squeeze(-1), torch.stack(new, 0)

    torch.onnx.export(Wrap(m), (torch.randn(1, 3, 80), torch.zeros(8, 1, 128, 19)),
                      str(out / "firered_stream_vad.onnx"), input_names=["feat", "cache"],
                      output_names=["probs", "new_cache"], dynamic_axes={"feat": {1: "T"}, "probs": {1: "T"}},
                      opset_version=17, dynamo=False)
    c = CMVN(str(src / "cmvn.ark"))
    (out / "cmvn.json").write_text(json.dumps({"means": [float(v) for v in c.means],
                                               "inv_std": [float(v) for v in c.inverse_std_variances]}))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--librispeech", required=True, help="LibriSpeech test-clean directory")
    b.add_argument("--esc50", required=True, help="ESC-50-master directory")
    b.add_argument("--corpus", help="ww_diag directory (default: app.ww_diag.diag_dir())")
    b.add_argument("--out", required=True)
    b.add_argument("--asr", action="store_true", help="label the corpus with Parakeet")
    b.add_argument("--asr-model-dir", help="default: livekit.stt.model_dir (untracked; a worktree has none)")
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--firered-dir")
    r.add_argument("--ten", action="store_true")
    r.add_argument("--thresholds", type=float, nargs="+")
    r.add_argument("--noise-gain", type=float, default=0.0, help="extra room tone, x the bed (4 ~ 18 dB SNR)")
    r.add_argument("--json")
    g = sub.add_parser("gate", help="false cutoffs through Smart Turn and the hold, per min_silence")
    g.add_argument("--data", required=True)
    g.add_argument("--smart-turn", default=str(ROOT / "agent-services/models/smart-turn/smart-turn-v3.2-cpu.onnx"))
    g.add_argument("--turn-threshold", type=float, default=0.5)
    g.add_argument("--vad-threshold", type=float, default=0.45)
    g.add_argument("--min-silence", type=int, nargs="+", default=[380, 250, 200])
    g.add_argument("--hold-timeout-ms", type=float, default=2200)
    g.add_argument("--noise-gains", type=float, nargs="+", default=[0.0, 4.0])
    g.add_argument("--turn-ms", type=float, help="fixed Smart Turn cost (default: median of 20 calls at start)")
    g.add_argument("--asr-model-dir", help="Parakeet dir: split cutoffs into sentence-end vs mid-sentence")
    g.add_argument("--json")
    e = sub.add_parser("export-firered")
    e.add_argument("src", help="FireRedVAD/Stream-VAD (model.pth.tar + cmvn.ark)")
    e.add_argument("out")
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "gate": cmd_gate, "export-firered": cmd_export_firered}[a.cmd](a)


if __name__ == "__main__":
    main()
