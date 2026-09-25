#!/usr/bin/env python
"""Streaming ASR bake-off: WER, partial lag, end-of-utterance, CPU cost.

`asr_eval.py` answers "which recogniser writes the best final transcript".
This answers the questions a streaming model is for, which that script cannot:
how soon a spoken word appears in the running hypothesis, whether a model's own
end-of-utterance token fires in the middle of a sentence, and what it costs to
keep one running beside the offline pass.

Every clip is fed in 20 ms frames — what the pipeline hands the recogniser —
against a simulated wall clock: a frame is available when its audio has been
spoken, and a result exists when the decode that consumed it returns
(`clock = max(clock, audio_time) + decode_seconds`). A model that decodes
slower than real time therefore falls behind here exactly as it would live.

The reference clock is Parakeet TDT 0.6B v3 (the production final
recogniser) on the same audio: its token timestamps and durations give each
word's end. So *partial lag* for a word = when that word first appears,
correctly, in the candidate's running hypothesis, minus when Parakeet says the
speaker finished saying it. It is measured only on words the candidate's final
transcript shares with Parakeet's (difflib alignment), so a model is not
credited for a word it gets wrong.

End of utterance (models whose tokens carry `<EOU>`), against Smart Turn v3.2
as production runs it (VAD silence `min_silence_ms`, then the classifier):

* latency — `<EOU>` time minus Parakeet's last word end, on each clip with a
  1 s tail of room-floor noise;
* false triggers — the same clips with a pause spliced in at the middle word
  boundary (`--pauses`, default 0.5 s and 1.0 s): a `<EOU>` during the pause,
  or Smart Turn calling the audio up to the pause (+ min_silence) complete.

    python scripts/voice/asr_stream_eval.py \\
        --corpus ~/.cache/lloyd-voice-eval/librispeech_test_clean_0.parquet --stride 13 \\
        --backend eou=agent-services/models/parakeet-realtime-eou-120m \\
        --backend nemotron160=path/to/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8 \\
        --room ~/lloyd-data/ww_diag --room-limit 150 --json out.json

`--backend moonshine-medium=moonshine:<model dir>` runs Moonshine v2 through
the `moonshine-voice` package when it is importable (it is not in the lloyd
venv; run the script from a venv that has it plus sherpa-onnx).

Corpus: LibriSpeech test-clean from the HF parquet mirror
(`https://huggingface.co/api/datasets/openslr/librispeech_asr/parquet/clean/test/0.parquet`,
2620 rows; `--stride 13` takes 202 of them, every speaker represented).
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "agent-services"))
sys.path.insert(0, str(HERE.parent))

from asr_eval import decode_audio, edit_distance, normalise, to_16k  # noqa: E402
from voice import asr as voice_asr  # noqa: E402

SR = 16000
FRAME = 320  # 20 ms
FLOOR_DBFS = -60.0


def floor_noise(n: int, rng) -> np.ndarray:
    """Room floor, not digital zero: a model trained on real audio can treat
    exact zeros as out of distribution, and no microphone produces them."""
    return rng.normal(0, 10 ** (FLOOR_DBFS / 20), n).astype(np.float32)


# --------------------------------------------------------------- reference
class ParakeetReference:
    """The production final recogniser, used for WER and as the word clock."""

    def __init__(self, model_dir: Path, threads: int = 4) -> None:
        self.rec = voice_asr.SherpaOfflineRecognizer(model_dir, num_threads=threads)
        self.rec.load()

    def run(self, audio: np.ndarray):
        t0 = time.perf_counter()
        s = self.rec._rec.create_stream()
        s.accept_waveform(SR, np.asarray(audio, dtype=np.float32))
        self.rec._rec.decode_stream(s)
        dt = time.perf_counter() - t0
        r = s.result
        words: list[tuple[str, float, float]] = []   # (word, start, end)
        durs = list(getattr(r, "durations", []) or [])
        for i, (tok, ts) in enumerate(zip(r.tokens, r.timestamps)):
            end = ts + (durs[i] if i < len(durs) and durs[i] > 0 else 0.08)
            if tok.startswith((" ", "▁")) or not words:
                for w in normalise(tok):
                    words.append((w, ts, end))
            else:
                piece = normalise(tok)
                if piece and words:
                    w, st, _ = words[-1]
                    words[-1] = (w + piece[0], st, end)
                    for extra in piece[1:]:
                        words.append((extra, ts, end))
                elif words:
                    w, st, _ = words[-1]
                    words[-1] = (w, st, end)
        return (r.text or "").strip(), words, dt


# --------------------------------------------------------------- candidates
class MoonshineStreaming:
    """Moonshine v2 streaming through the moonshine-voice package. Updated
    every `update_s` of audio — a transcription pass re-runs the decoder over
    the whole line, so updating per 20 ms frame would be a different (and far
    costlier) operating point than the SDK's own."""

    has_eou = False

    def __init__(self, model_dir: str, arch: str = "MEDIUM_STREAMING",
                 update_s: float = 0.1) -> None:
        from moonshine_voice.moonshine_api import ModelArch
        from moonshine_voice.transcriber import Transcriber

        self.tr = Transcriber(model_dir, getattr(ModelArch, arch), update_interval=1e9)
        self.update_s = update_s

    def create_stream(self):
        s = self.tr.create_stream(update_interval=1e9)
        s.start()
        return {"s": s, "fed": 0, "last": 0, "text": ""}

    @staticmethod
    def _text(tr) -> str:
        return " ".join(line.text.strip() for line in (tr.lines if tr else []) if line.text)

    def accept(self, st, audio):
        st["s"].add_audio(np.asarray(audio, dtype=np.float32).tolist(), SR)
        st["fed"] += len(audio)
        if st["fed"] - st["last"] >= self.update_s * SR:
            st["last"] = st["fed"]
            st["text"] = self._text(st["s"].update_transcription())
        return st["text"]

    def endpointed(self, st) -> bool:
        return False

    def finish(self, st) -> str:
        tr = st["s"].stop()
        return self._text(tr) or st["text"]


def build_backend(spec: str, threads: int):
    name, _, path = spec.partition("=")
    if path.startswith("moonshine:"):
        return name, MoonshineStreaming(path.split(":", 1)[1])
    precision, language = "int8", ""
    if "#" in path:                     # dir#en — a per-stream language hint
        path, language = path.split("#", 1)
    if path.endswith("@fp32"):
        path, precision = path[:-5], "fp32"
    r = voice_asr.SherpaStreamingRecognizer(path, num_threads=threads, precision=precision,
                                            language=language)
    r.load()
    return name, r


# --------------------------------------------------------------- one clip
def stream_clip(backend, audio: np.ndarray):
    """Feed in 20 ms frames on a simulated clock. Returns the history of
    (clock, audio_time, text, eou), the final text, and decode seconds."""
    st = backend.create_stream()
    clock = 0.0
    busy = 0.0
    hist: list[tuple[float, float, str, bool]] = []
    last = (None, False)
    for i in range(0, audio.size, FRAME):
        chunk = audio[i:i + FRAME]
        t_audio = (i + chunk.size) / SR
        t0 = time.perf_counter()
        text = backend.accept(st, chunk)
        eou = backend.endpointed(st)
        dt = time.perf_counter() - t0
        busy += dt
        clock = max(clock, t_audio) + dt
        if (text, eou) != last:
            hist.append((clock, t_audio, text, eou))
            last = (text, eou)
    t0 = time.perf_counter()
    final = backend.finish(st)
    busy += time.perf_counter() - t0
    return hist, final, busy


def word_lags(hist, final_words, ref_words, offset=0.0):
    """Seconds from each shared word's end (reference clock, shifted by the
    pre-roll `offset`) to its first correct appearance in a partial."""
    ref = [w for w, _, _ in ref_words]
    sm = difflib.SequenceMatcher(a=final_words, b=ref, autojunk=False)
    hist_words = [(clock, normalise(text)) for clock, _, text, _ in hist]
    lags = []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            j, r = blk.a + k, blk.b + k
            target = final_words[j]
            for clock, ws in hist_words:
                if len(ws) > j and ws[j] == target:
                    lags.append(clock - (ref_words[r][2] + offset))
                    break
    return lags


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--stride", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", action="append", default=[],
                    help="name=model_dir[@fp32][#lang] or name=moonshine:<dir>")
    ap.add_argument("--threads", type=int, default=2, help="streaming decode threads")
    ap.add_argument("--parakeet", default=str(HERE.parents[2] / "agent-services/models/parakeet-tdt-v3"))
    ap.add_argument("--smart-turn", default=str(HERE.parents[2] / "agent-services/models/smart-turn/smart-turn-v3.2-cpu.onnx"))
    ap.add_argument("--min-silence-ms", type=float, default=380)
    ap.add_argument("--hold-timeout-ms", type=float, default=2200)
    ap.add_argument("--pauses", type=float, nargs="*", default=[0.5, 1.0])
    ap.add_argument("--room", default=None, help="ww_diag dir (scores.jsonl + utterances/)")
    ap.add_argument("--room-limit", type=int, default=150)
    ap.add_argument("--room-min-words", type=int, default=4,
                    help="only room utterances Parakeet hears this many words in: most of "
                         "the corpus is sub-second fragments (\"Yeah.\", \"Mm hmm.\") where an "
                         "empty streaming result and a one-word offline guess would "
                         "dominate the disagreement rate")
    ap.add_argument("--room-only", action="store_true",
                    help="skip the LibriSpeech streaming pass (reference still loads)")
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--preroll", type=float, default=0.5,
                    help="seconds of room floor fed before each clip. A cache-aware "
                         "streaming encoder drops the first ~0.3 s of a stream, so a "
                         "clip whose speech starts at 0 loses its first word without it")
    ap.add_argument("--json", default=None)
    ap.add_argument("--ref-cache", default=None,
                    help="JSON cache of the Parakeet reference + Smart Turn results, so "
                         "several candidate runs over one corpus decode it once")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    rows = pq.read_table(args.corpus).to_pylist()[:: args.stride]
    if args.limit:
        rows = rows[: args.limit]
    clips = []
    for r in rows:
        x, sr = decode_audio(r["audio"])
        a = to_16k(x, sr)
        if args.snr is not None:
            p = float(np.mean(a * a)) / (10 ** (args.snr / 10))
            a = (a + rng.normal(0, np.sqrt(p), a.size)).astype(np.float32)
        clips.append((a, r["text"]))
    total_audio = sum(a.size for a, _ in clips) / SR
    print(f"{len(clips)} clips, {total_audio:.0f} s, stride {args.stride}"
          + (f", SNR {args.snr:g} dB" if args.snr is not None else ""), flush=True)

    cache_key = f"{args.corpus}|{args.stride}|{args.limit}|{args.snr}"
    cache = {}
    if args.ref_cache and Path(args.ref_cache).exists():
        cache = json.loads(Path(args.ref_cache).read_text())
        if cache.get("key") != cache_key:
            cache = {}
    if cache:
        refs = [[tuple(w) for w in ws] for ws in cache["refs"]]
        results = {k: v for k, v in cache["results"].items()}
    else:
        ref = ParakeetReference(Path(args.parakeet))
        refs = []
        errs = n = 0
        busy = 0.0
        for a, text in clips:
            hyp, words, dt = ref.run(a)
            busy += dt
            refs.append(words)
            r = normalise(text)
            errs += edit_distance(r, normalise(hyp))
            n += len(r)
        results = {"parakeet-tdt (offline, final)": {
            "wer": errs / n, "rtf": busy / total_audio, "threads": 4}}
    p = results["parakeet-tdt (offline, final)"]
    print(f"parakeet-tdt offline: WER {100 * p['wer']:.2f}%  RTF {p['rtf']:.3f}", flush=True)

    smart = None
    try:
        from voice.turn import SmartTurn

        smart = SmartTurn(args.smart_turn)
        smart.load()
    except Exception as e:  # faster_whisper absent, model absent
        print(f"smart-turn unavailable: {e}")
        smart = None

    # Smart Turn, once: it does not depend on the candidate.
    ms = int(args.min_silence_ms * SR / 1000)
    splices = []   # (clip index, cut sample)
    for idx, (a, _) in enumerate(clips):
        w = refs[idx]
        if len(w) < 8:
            continue
        k = len(w) // 2 - 1
        cut = int(((w[k][2] + w[k + 1][1]) / 2) * SR)
        if 0 < cut < a.size:
            splices.append((idx, cut))
    if "smart-turn v3.2 (+VAD)" in results:
        print(f"smart-turn (cached): {results['smart-turn v3.2 (+VAD)']}", flush=True)
    elif smart is not None:
        end_ok = mid_fire = 0
        st_ms = []
        for idx, (a, _) in enumerate(clips):
            last_end = int(refs[idx][-1][2] * SR) if refs[idx] else a.size
            v = smart.predict(np.concatenate([a[:last_end], floor_noise(ms, rng)]))
            end_ok += v.complete
            st_ms.append(v.elapsed_ms)
        for idx, cut in splices:
            a = clips[idx][0]
            mid_fire += smart.predict(np.concatenate([a[:cut], floor_noise(ms, rng)])).complete
        lat = [args.min_silence_ms + np.median(st_ms)] * end_ok + \
              [args.hold_timeout_ms] * (len(clips) - end_ok)
        results["smart-turn v3.2 (+VAD)"] = {
            "end_complete": end_ok / len(clips),
            "false_mid": mid_fire / max(1, len(splices)),
            "eou_latency_p50_ms": pct(lat, 50), "eou_latency_p90_ms": pct(lat, 90),
            "infer_ms_p50": float(np.median(st_ms))}
        print(f"smart-turn: end complete {end_ok}/{len(clips)}, mid-sentence complete "
              f"{mid_fire}/{len(splices)}, latency p50 {pct(lat, 50):.0f} ms "
              f"p90 {pct(lat, 90):.0f} ms (min_silence {args.min_silence_ms:g} + "
              f"{np.median(st_ms):.0f} ms inference; hold {args.hold_timeout_ms:g})", flush=True)

    if args.ref_cache and not cache:
        Path(args.ref_cache).write_text(json.dumps({"key": cache_key, "refs": refs,
                                                    "results": results}))

    # Room audio: the reference is Parakeet TDT run NOW on the saved wav, not
    # the row's `stt_text`. That field was written at capture time by whatever
    # recogniser and segmentation ran then, and on this corpus it often
    # describes different audio (a 0.8 s clip whose stt_text is 17 words).
    room = []
    if args.room:
        import soundfile as sf

        rd = Path(args.room).expanduser()
        paths = []
        for line in (rd / "scores.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            p = rd / "utterances" / f"{row.get('utterance_id')}.wav"
            if p.exists() and p not in paths:
                paths.append(p)
        room_ref = ParakeetReference(Path(args.parakeet))
        for p in paths:
            x, sr = sf.read(str(p), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            a = to_16k(x, sr)
            text, _, _ = room_ref.run(a)
            if len(normalise(text)) >= args.room_min_words:
                room.append((a, text))
        room = room[-args.room_limit:]
        del room_ref

    for spec in args.backend:
        name, be = build_backend(spec, args.threads)
        errs = n = 0
        busy = 0.0
        lags: list[float] = []
        eou_lat: list[float] = []
        eou_missing = eou_early = 0
        for idx, (a, text) in enumerate([] if args.room_only else clips):
            pre = floor_noise(int(args.preroll * SR), rng)
            audio = np.concatenate([pre, a, floor_noise(SR, rng)])
            hist, final, dt = stream_clip(be, audio)
            busy += dt
            fw = normalise(final)
            r = normalise(text)
            errs += edit_distance(r, fw)
            n += len(r)
            lags += word_lags(hist, fw, refs[idx], args.preroll)
            if getattr(be, "has_eou", False) and refs[idx]:
                last_end = refs[idx][-1][2] + args.preroll
                first = next((c for c, t_a, _, e in hist if e), None)
                if first is None:
                    eou_missing += 1
                elif first < last_end - 0.05:
                    eou_early += 1
                else:
                    eou_lat.append((first - last_end) * 1000)
        res = {"wer": errs / max(1, n),
               "rtf": busy / (total_audio + len(clips) * (1 + args.preroll)),
               "threads": args.threads,
               "lag_p50_ms": pct(lags, 50) * 1000, "lag_p90_ms": pct(lags, 90) * 1000,
               "lag_mean_ms": float(np.mean(lags)) * 1000 if lags else float("nan"),
               "words_timed": len(lags)}
        line = (f"{name}: WER {100 * res['wer']:.2f}%  RTF {res['rtf']:.3f} @{args.threads}t  "
                f"partial lag p50 {res['lag_p50_ms']:.0f} ms p90 {res['lag_p90_ms']:.0f} ms "
                f"({len(lags)} words)")
        if getattr(be, "has_eou", False) and not args.room_only:
            mid = 0
            for idx, cut in splices:
                a = clips[idx][0]
                for pause in args.pauses:
                    audio = np.concatenate([floor_noise(int(args.preroll * SR), rng), a[:cut],
                                            floor_noise(int(pause * SR), rng), a[cut:],
                                            floor_noise(SR, rng)])
                    hist, _, _ = stream_clip(be, audio)
                    resume = args.preroll + cut / SR + pause
                    mid += any(e and t_a <= resume for _, t_a, _, e in hist)
            trials = len(splices) * len(args.pauses)
            res.update({"eou_latency_p50_ms": pct(eou_lat, 50), "eou_latency_p90_ms": pct(eou_lat, 90),
                        "eou_missing": eou_missing / len(clips), "eou_early": eou_early / len(clips),
                        "false_mid": mid / max(1, trials), "false_mid_trials": trials})
            line += (f"\n    EOU latency p50 {res['eou_latency_p50_ms']:.0f} ms p90 "
                     f"{res['eou_latency_p90_ms']:.0f} ms; no EOU in 1 s tail {eou_missing}/{len(clips)}; "
                     f"early {eou_early}; mid-sentence EOU {mid}/{trials} (pauses {args.pauses})")
        if room:
            rerr = rn = 0
            rbusy = raud = 0.0
            for a, stt in room:
                raud += a.size / SR + args.preroll + 1
                _, final, dt = stream_clip(be, np.concatenate(
                    [floor_noise(int(args.preroll * SR), rng), a, floor_noise(SR, rng)]))
                rbusy += dt
                rw = normalise(stt)
                rerr += edit_distance(rw, normalise(final))
                rn += len(rw)
            res.update({"room_disagree": rerr / max(1, rn), "room_rtf": rbusy / max(raud, 1e-9),
                        "room_n": len(room)})
            line += (f"\n    room ({len(room)} utts): disagreement with Parakeet "
                     f"{100 * res['room_disagree']:.1f}%  RTF {res['room_rtf']:.3f}")
        results[name] = res
        print(line, flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
