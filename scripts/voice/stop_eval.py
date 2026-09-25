#!/usr/bin/env python
"""Measure the stop word the way the worker will run it: only while Lloyd talks.

`lloyd_stop.onnx` is armed while Lloyd is speaking, so the audio it must not
fire on is mostly *his own voice* — and his replies say "stop", "wait" and
"hold on" mid-sentence ("the service will stop when…"). Every candidate is
scored through `voice.wake.build_stop_factory`'s detector, the worker's own
runtime, fed 10 ms at a time.

Corpora (built by `build`, none committed):

  held-out TTS   ~/.cache/lloyd-voice-eval/stop-tts/{pos,neg}_<voice>_<n>.wav
                 Qwen3-TTS x-vector clones — of the ten voices wake_eval uses,
                 LibriSpeech speaker 1272 and Alan (from the room corpus) — a
                 different engine from every training voice.
  Lloyd's voice  ~/.cache/lloyd-voice-eval/stop-tts/lloyd_<n>.wav, 30 replies
                 in `clone:dave_cullen`, disjoint from the 60 the model was
                 trained against (`build --train-out`). Played as one stream.
  real room      the ww_diag corpus (app.ww_diag) not logged as a wake (`--diag`);
                 the ODD half is held out, the even half was trained on.
  LibriSpeech    as wake_eval.py.

    python scripts/voice/stop_eval.py build [--train-out DIR]
    python scripts/voice/stop_eval.py run agent-services/models/wakeword/stop/lloyd_stop.onnx
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-services"))
sys.path.insert(0, str(ROOT / "scripts" / "voice"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SR = 16000
OUT = Path.home() / ".cache" / "lloyd-voice-eval" / "stop-tts"
WAKE_TTS = Path.home() / ".cache" / "lloyd-voice-eval" / "wake-tts"
def _default_diag() -> Path:
    """The wake-diagnostic corpus, where `app.ww_diag` says it lives."""
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.ww_diag import diag_dir
    return diag_dir()
TTS = "http://127.0.0.1:8090/v1"

POS = ["Stop.", "Lloyd, stop.", "Stop talking.", "Hold on.", "Wait.", "Okay, stop.",
       "Stop, that's wrong.", "Wait, wait."]
NEG = ["Don't stop now.", "Is there a bus stop near here?", "I can't wait for dinner.",
       "Hold onto the railing.", "The stock went up today.", "I stopped by the shop.",
       "What's on top of the list?", "Wake me up at seven."]

# Replies in Lloyd's register. Many use the words in-sentence: that is the risk.
LLOYD_EVAL = [
    "The backend restarted cleanly, and the worker pool is running again.",
    "You can stop the service with supervisorctl, but the round CLI is safer.",
    "I'll wait for the gate to finish before I land anything.",
    "Hold on to that branch name, you'll need it for the merge.",
    "The build didn't stop on the first error, it kept going to the end.",
    "Three items are waiting in the backlog, and one of them is high priority.",
    "It's sixty eight degrees outside, with a light breeze from the west.",
    "The bus stop on Fourth Street moved half a block north last week.",
    "I stopped the experiment early because the numbers weren't moving.",
    "If you want, I can wait until tomorrow morning and remind you then.",
    "The primary engine is healthy, and the KV cache is at forty percent.",
    "Nothing has changed since the last check, so there's nothing to stop.",
    "Please hold on while I pull up the calendar for next week.",
    "The download will stop automatically once the disk is full.",
    "Waiting on the review rung, which usually takes about ten minutes.",
    "That's a non stop flight, so you'll land at six fifteen.",
    "Sure. I've set a timer for twenty minutes.",
    "The test suite passed, all four hundred and twelve tests.",
    "Let me stop and check the logs before I guess at a cause.",
    "Your package is out for delivery and should arrive by five.",
    "We can hold on to the old model for a week, just in case.",
    "The script waits for the lock, then writes the file in one go.",
    "I couldn't find anything by that name in the vault.",
    "The meeting starts at two, and there's a stop at the pharmacy before that.",
    "Traffic is light, so it's about a twenty minute drive.",
    "Stopping the timer now. You were at eleven minutes.",
    "The last three rounds all landed without a rollback.",
    "I'll hold off on the restart until the pool is idle.",
    "Good news, the wait time at the clinic is under fifteen minutes.",
    "The model fired once on the room audio, at a score of point seven two.",
]
LLOYD_TRAIN = [
    "The gate passed on the second try, so the round is landing now.",
    "I stopped the worker pool before the restart, as usual.",
    "You'll have to wait about five minutes for the engine to boot.",
    "Hold on, I'm still reading the rest of the log file.",
    "There's a full stop missing at the end of that sentence.",
    "The regression check is waiting on the pinned daemon.",
    "It won't stop raining until late tomorrow afternoon.",
    "I can stop the download if you need the bandwidth.",
    "The queue is empty, so nothing is waiting to run.",
    "We waited too long on that item, it expired last week.",
    "Stop signs on that road were replaced with a roundabout.",
    "I'll hold the confirmation until there's room in the pool.",
    "The kettle should stop boiling in about a minute.",
    "The research job stopped at the ten minute cap.",
    "Your flight is on time, and the gate is B twelve.",
    "I don't think we should wait any longer on that decision.",
    "The script holds on to the file handle until it exits.",
    "Everything is up to date. Nothing to do right now.",
    "It stops at every station between here and downtown.",
    "The tests are waiting for the canary ports to free up.",
    "Here's the summary. Two items closed, one reopened, none spent.",
    "I've stopped the timer. You spent twenty six minutes on it.",
    "Wait times at the airport are about thirty minutes this morning.",
    "The service keeps running unless you stop it by hand.",
    "Hold on to the receipt in case the return window matters.",
    "I paused the pool and I'm waiting for the backend to go idle.",
    "The weather looks clear all weekend, highs in the seventies.",
    "That process stopped responding about an hour ago.",
    "If the build stops again, I'll bisect the last three commits.",
    "The worker waits on the lock and then retries once.",
    "I found two notes that mention that project, both from August.",
    "The nightly job stopped early because the primary was restarting.",
    "You asked me to wait, so I held the message until now.",
    "There are four stops on the delivery route this afternoon.",
    "The review is still waiting on the grader, about two minutes left.",
    "Stopping here, because the rest depends on your decision.",
    "I'll hold the landing until the observation window closes.",
    "The music will stop at eleven, when quiet hours begin.",
    "Wait until the snapshot finishes before you restore anything.",
    "It's a two stop trip on the red line, about ten minutes.",
    "Nothing stopped overnight, all services are green.",
    "Hold on to your questions, the answer is in the next section.",
    "I waited for the sync to finish, and it's all caught up.",
    "The alarm will stop after sixty seconds on its own.",
    "The loop can't stop a round that's already landing.",
    "It's waiting for your approval before it sends the email.",
    "Your appointment is at nine, with a stop at the bank first.",
    "I'd hold off on buying until the price drops again.",
    "The fan stops spinning when the card is under fifty degrees.",
    "The stock price closed a little higher than yesterday.",
    "The next train stops here in four minutes.",
    "I'll wait for the numbers before I make a recommendation.",
    "Your reminder is set for tomorrow at eight, stop by the post office.",
    "We held on to the old config, it's in the backup directory.",
    "The engine doesn't stop cleanly if you send it two signals.",
    "Okay. I'll stop there and pick it up in the morning.",
    "The waiting list for that class has three names on it.",
    "I stopped counting after the first hundred matches.",
    "Wait, that number looks wrong, let me check it again.",
    "Hold on a moment, the transcript is still coming in.",
]


def _post(path: str, body: dict) -> bytes:
    req = urllib.request.Request(TTS + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _write16k(path: Path, wav_bytes: bytes) -> None:
    import soundfile as sf
    from scipy.signal import resample_poly
    a, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def _refs() -> dict[str, bytes]:
    """One reference WAV per held-out voice, as bytes for the clone endpoint."""
    import soundfile as sf
    from replay import load_16k
    refs: dict[str, np.ndarray] = {}
    for p in sorted(WAKE_TTS.glob("*.wav")):
        m = re.match(r"(pos|neg)_(.+)_(\d+)\.wav$", p.name)
        if m and not m.group(2).startswith("clone"):
            refs.setdefault(m.group(2), []).append(load_16k(p))
    try:
        import pyarrow.parquet as pq
        from asr_eval import decode_audio, to_16k
        rows = pq.read_table(Path.home() / ".cache/lloyd-voice-eval/"
                             "librispeech_dummy_clean_validation.parquet").to_pylist()
        refs["libri1272"] = [to_16k(*decode_audio(r["audio"])) for r in rows[:4]]
    except Exception as e:  # pragma: no cover - optional corpus
        print(f"no LibriSpeech ref: {e}")
    diag = _default_diag()
    rows = [json.loads(ln) for ln in (diag / "scores.jsonl").open() if ln.strip()]
    long_ = sorted((r for r in rows if len((r.get("stt_text") or "").split()) >= 6
                    and (diag / "utterances" / f"{r['utterance_id']}.wav").exists()),
                   key=lambda r: -r["duration_s"])[:3]
    if long_:
        refs["alan"] = [load_16k(diag / "utterances" / f"{r['utterance_id']}.wav") for r in long_]
    out = {}
    for v, clips in refs.items():
        a = np.concatenate([np.asarray(c, np.float32) / (32768.0 if np.asarray(c).dtype == np.int16 else 1.0)
                            for c in clips])[: SR * 15]
        buf = io.BytesIO()
        sf.write(buf, a, SR, format="WAV", subtype="PCM_16")
        out[v] = buf.getvalue()
    return out


def build(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for v, ref in _refs().items():
        b64 = base64.b64encode(ref).decode()
        for kind, texts in (("pos", POS), ("neg", NEG)):
            for i, t in enumerate(texts):
                jobs.append((OUT / f"{kind}_{v}_{i}.wav", "/audio/voice-clone",
                             {"input": t, "ref_audio": b64, "x_vector_only_mode": True,
                              "response_format": "wav", "language": "English"}))
    for i, t in enumerate(LLOYD_EVAL):
        jobs.append((OUT / f"lloyd_{i:02d}.wav", "/audio/speech",
                     {"model": "qwen3-tts", "input": t, "voice": "clone:dave_cullen",
                      "response_format": "wav"}))
    if args.train_out:
        tr = Path(args.train_out).expanduser()
        tr.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(LLOYD_TRAIN):
            jobs.append((tr / f"lloyd_train_{i:02d}.wav", "/audio/speech",
                         {"model": "qwen3-tts", "input": t, "voice": "clone:dave_cullen",
                          "response_format": "wav"}))
    todo = [j for j in jobs if not j[0].exists()]
    print(f"{len(jobs)} clips, {len(todo)} to synthesise (sequential, {args.gap}s apart)")
    for n, (path, route, body) in enumerate(todo):
        try:
            _write16k(path, _post(route, body))
        except Exception as e:
            print(f"  {path.name}: {e}")
        if n % 20 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
        time.sleep(args.gap)
    return 0


def _room(diag: Path, half: str) -> list[np.ndarray]:
    from replay import load_16k
    fired = {json.loads(ln)["utterance_id"] for ln in (diag / "scores.jsonl").open()
             if ln.strip() and json.loads(ln).get("ww_fired")}
    paths = [p for p in sorted((diag / "utterances").glob("*.wav")) if p.stem not in fired]
    keep = {"all": lambda i: True, "even": lambda i: i % 2 == 0, "odd": lambda i: i % 2 == 1}[half]
    return [load_16k(p) for i, p in enumerate(paths) if keep(i)]


def _room_tone(seconds, rng):
    return rng.normal(0, 0.002, int(SR * seconds)).astype(np.float32)


def peak(factory, audio, rng) -> float:
    """Best score an ARMED detector reaches over this clip, fed 10 ms at a time
    after two seconds of room tone (the model's warmup, paid while disarmed)."""
    d = factory.create()
    lead = _room_tone(2.0, rng)
    for i in range(0, lead.size, 160):
        d.feed(lead[i:i + 160])
    d.arm()
    best = 0.0
    tail = np.concatenate([audio, _room_tone(1.0, rng)])
    for i in range(0, tail.size, 160):
        d.feed(tail[i:i + 160])
        best = max(best, d.score)
    return best


def fires_in_stream(factory, clips, threshold, rng, gap=1.0) -> int:
    """Detections over one continuous, armed stream of clips."""
    d = factory.create()
    d.threshold = threshold
    d.arm()
    n = 0
    for c in clips:
        s = np.concatenate([_room_tone(gap, rng), c])
        for i in range(0, s.size, 160):
            n += len(d.feed(s[i:i + 160]))
    return n


def run(args) -> int:
    from voice.wake import build_stop_factory
    from wake_eval import libri_clips
    from replay import load_16k

    pos, neg, own = [], [], []
    for p in sorted(OUT.glob("*.wav")):
        m = re.match(r"(pos|neg)_(.+)_(\d+)\.wav$", p.name)
        if m:
            (pos if m.group(1) == "pos" else neg).append((m.group(2), int(m.group(3)), load_16k(p)))
        elif p.name.startswith("lloyd_"):
            own.append(load_16k(p))
    real = _room(Path(args.diag).expanduser() if args.diag else _default_diag(), args.real_half)
    libri = libri_clips()
    own_s = sum(c.size for c in own) / SR
    libri_h = sum(c.size for c in libri) / SR / 3600
    voices = sorted({v for v, _, _ in pos})
    print(f"held-out: {len(voices)} voices ({', '.join(voices)}), {len(pos)} positives, "
          f"{len(neg)} near-misses | Lloyd's voice: {len(own)} replies ({own_s:.0f} s) | "
          f"real room ({args.real_half}): {len(real)} utts | LibriSpeech {libri_h * 60:.1f} min\n")
    for model in args.models:
        f = build_stop_factory({"engine_dir": str(ROOT / "agent-services/models/openwakeword"),
                                "stop": {"enabled": True, "model": model, "threshold": 0.5}})
        rng = np.random.default_rng(0)
        pos_s = [(v, i, peak(f, a, rng)) for v, i, a in pos]
        neg_s = [(v, i, peak(f, a, rng)) for v, i, a in neg]
        own_peak = [peak(f, a, rng) for a in own]
        print(f"== {model}")
        print(f"{'thresh':>7} {'recall':>9} {'near-miss':>10} {'own FA':>7} {'own peaks>=':>11} "
              f"{'room FA':>8} {'libri FA/h':>11}")
        for th in args.thresholds:
            rp = sum(1 for *_, s in pos_s if s >= th)
            fn = sum(1 for *_, s in neg_s if s >= th)
            fo = fires_in_stream(f, own, th, np.random.default_rng(3), gap=0.4)
            po = sum(1 for s in own_peak if s >= th)
            fr = fires_in_stream(f, real, th, np.random.default_rng(1))
            fl = fires_in_stream(f, libri, th, np.random.default_rng(2), gap=0.3) if libri else 0
            print(f"{th:>7.2f} {rp:>4}/{len(pos):<4} {fn:>4}/{len(neg):<4} {fo:>7} "
                  f"{po:>4}/{len(own):<5} {fr:>8} {fl / max(libri_h, 1e-9):>11.1f}")
        by_phrase = {}
        for _, i, s in pos_s:
            by_phrase.setdefault(POS[i], []).append(s)
        th = args.thresholds[len(args.thresholds) // 2]
        print(f"  recall by phrase @ {th}: " + ", ".join(
            f"{k!r} {sum(s >= th for s in v)}/{len(v)}" for k, v in by_phrase.items()))
        by_voice = {}
        for v, _, s in pos_s:
            by_voice.setdefault(v, []).append(s)
        print(f"  recall by voice @ {th}: " + ", ".join(
            f"{k} {sum(s >= th for s in v)}/{len(v)}" for k, v in by_voice.items()))
        loud = sorted(((round(s, 2), NEG[i], v) for v, i, s in neg_s if s >= 0.3), reverse=True)
        print(f"  near-misses >= 0.3: {loud[:12]}")
        top = sorted(((round(s, 2), LLOYD_EVAL[i][:50]) for i, s in enumerate(own_peak)), reverse=True)[:5]
        print(f"  Lloyd's highest: {top}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train-out")
    b.add_argument("--gap", type=float, default=0.5, help="seconds between TTS requests")
    r = sub.add_parser("run")
    r.add_argument("models", nargs="+")
    r.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    r.add_argument("--diag", default=None)
    r.add_argument("--real-half", choices=["all", "even", "odd"], default="odd")
    args = ap.parse_args()
    return build(args) if args.cmd == "build" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
