#!/usr/bin/env python
"""TTS bake-off: latency, codec presence, intelligibility and clone similarity.

Two phases, because a candidate model lives in its own venv while the raters
(Parakeet, Resemblyzer) live in the lloyd venv:

    # 1. synthesise, in whatever venv can run the candidate (numpy is enough
    #    for the HTTP runner; a local runner brings its own deps)
    python scripts/voice/tts_bakeoff.py synth --name live_qwen3 \
        --runner openai --url http://127.0.0.1:8090 --voice clone:dave_cullen \
        --out <dir>
    python scripts/voice/tts_bakeoff.py synth --name cosyvoice3 \
        --runner path/to/runner.py:make_runner --out <dir> [--live-text]

    # 2. score every candidate directory against the clone's reference clip
    .venvs/lloyd/bin/python scripts/voice/tts_bakeoff.py score <dir>/* \
        --ref <profile>/ref.wav --parakeet-dir <models>/parakeet-tdt-v3

What each number means:

- **TTFB** — request (or first text token, for live text) to the first
  non-empty PCM chunk, p50/p90 over the sentence set.
- **TTFA from the LLM** (`--live-text`) — text arrives as LLM-sized tokens at
  `--tok-rate` tokens/s. The *clause* arm is today's path: the first clause
  `voice.speakable.ClauseStream` would cut is complete, then it is synthesised
  whole, so TTFA = time the clause-ending token arrives + that clause's TTFB.
  The *live* arm feeds the same paced tokens into a runner that accepts text
  incrementally and times the first audio from the first token.
- **RTF** — synthesis wall time / audio seconds (lower is faster).
- **Presence** — the method in `architecture/voice.md` "Presence": STFT
  frames gated on **sub-1 kHz** energy (within 30 dB of the loudest), band
  power normalised to the 100–1500 Hz band, output minus reference per band.
  Gating on total RMS instead swaps vowel frames for fricatives and inflates
  exactly the number being measured.
- **WER** — the synthesis transcribed by Parakeet greedy with no hotword list
  (the #1165 rater), against the input text.
- **Speaker sim** — Resemblyzer cosine (the worker's own speaker-id encoder)
  between the synthesis and the reference clip.

A local runner is a module exposing a factory (default `make_runner`) that
returns an object with `name`, `sample_rate`, `stream(text) -> iterator of
float32 numpy chunks`, optionally `stream_live(fragments) -> iterator of
chunks` (fragments is a blocking iterator of text pieces, exhausted when the
text is done), optional `vram_mib()` and `close()`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-services"))

#: Ten short conversational replies of the shape Lloyd speaks. The first five
#: are the listen set. Several open with a short clause, which is where
#: clause-at-a-time waits longest.
SENTENCES = [
    "Sure, I've added the dentist appointment to your calendar for Thursday at three.",
    "The build finished cleanly, and all four hundred tests passed on the first run.",
    "Honestly? I'd wait until the morning, because the backup is still running.",
    "Your package from the hardware store should arrive sometime tomorrow afternoon.",
    "It's sixty two degrees outside right now, with light rain expected after six.",
    "I checked the logs, and the engine restarted twice overnight without losing a turn.",
    "Okay, the living room lights are off and the thermostat is set to sixty eight.",
    "There are three new emails, but only the one from the school looks urgent.",
    "Give me a second to look that up, it's buried in last week's notes.",
    "That should do it, let me know if the printer gives you any more trouble.",
]
LISTEN = 5

BANDS = [(1500, 2500), (2500, 3500), (3500, 5000), (5000, 9000), (9000, 12000)]


# ── small audio helpers (numpy only) ────────────────────────────────────────

def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    pcm = np.clip(np.rint(np.asarray(x, np.float32) * 32767.0), -32768, 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.astype("<i2").tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: only 16-bit wav is supported")
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def pctl(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ── pacing text like an LLM ─────────────────────────────────────────────────

_TOK = re.compile(r"\s?[A-Za-z']{1,5}|\s?\d{1,3}|\s?[^\sA-Za-z\d]|\s+")


def llm_tokens(text: str) -> list[str]:
    """BPE-sized pieces (~4 chars): close enough to pace text like the LLM."""
    toks = _TOK.findall(text)
    assert "".join(toks) == text, "tokeniser must be lossless"
    return toks


def first_clause_cut(tokens: list[str]) -> tuple[int, str]:
    """(index of the token that completes the first clause, the clause) — what
    the worker's ClauseStream would send to TTS first."""
    from voice.speakable import ClauseStream

    cs = ClauseStream()
    for i, t in enumerate(tokens):
        out = cs.feed(t)
        if out:
            return i, out[0]
    rest = cs.flush()
    return len(tokens) - 1, (rest[0] if rest else "".join(tokens))


def paced(tokens: list[str], rate: float, t0: float) -> Iterator[str]:
    for i, t in enumerate(tokens):
        delay = t0 + i / rate - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        yield t


# ── runners ─────────────────────────────────────────────────────────────────

class OpenAIRunner:
    """Any OpenAI-style `/v1/audio/speech` that streams raw s16le PCM."""

    def __init__(self, url: str, voice: str, model: str = "qwen3-tts",
                 sample_rate: int = 24000, extra: dict | None = None):
        self.name = "openai"
        self.url = url.rstrip("/") + "/v1/audio/speech"
        self.voice, self.model, self.sample_rate = voice, model, sample_rate
        self.extra = extra or {}

    def stream(self, text: str) -> Iterator[np.ndarray]:
        body = {"model": self.model, "input": text, "voice": self.voice,
                "response_format": "pcm", "stream": True, "speed": 1.0, **self.extra}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        carry = b""
        with urllib.request.urlopen(req, timeout=300) as r:
            while True:
                b = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                if not b:
                    break
                b = carry + b
                n = len(b) // 2 * 2
                carry = b[n:]
                if n:
                    yield np.frombuffer(b[:n], "<i2").astype(np.float32) / 32768.0


def load_runner(spec: str, args) -> object:
    if spec == "openai":
        extra = json.loads(args.extra) if args.extra else None
        return OpenAIRunner(args.url, args.voice, args.model, args.sample_rate, extra)
    path, _, fn = spec.partition(":")
    mod_spec = importlib.util.spec_from_file_location("bake_runner", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return getattr(mod, fn or "make_runner")(**(json.loads(args.runner_args or "{}")))


def gpu_mib_of(pid: int) -> int | None:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        p, _, m = line.partition(",")
        if p.strip() == str(pid):
            return int(m.strip())
    return None


# ── synth phase ─────────────────────────────────────────────────────────────

def timed_stream(chunks: Iterator[np.ndarray], t0: float) -> tuple[np.ndarray, float | None, float]:
    first, parts = None, []
    for c in chunks:
        c = np.asarray(c, np.float32).reshape(-1)
        if c.size and first is None and np.any(np.abs(c) > 1e-4):
            first = time.perf_counter() - t0
        parts.append(c)
    total = time.perf_counter() - t0
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), first, total


def live_stream(runner, tokens: list[str], rate: float) -> tuple[np.ndarray, float | None, float]:
    """Feed paced tokens to `runner.stream_live` from a producer thread."""
    import queue

    q: queue.Queue = queue.Queue()
    t0 = time.perf_counter()

    def produce():
        for t in paced(tokens, rate, t0):
            q.put(t)
        q.put(None)

    def fragments():
        while True:
            t = q.get()
            if t is None:
                return
            yield t

    threading.Thread(target=produce, daemon=True).start()
    return timed_stream(runner.stream_live(fragments()), t0)


def shape(audio: np.ndarray, sr: int) -> np.ndarray:
    """What the worker does to the server's audio (presence shelves + WSOLA
    speed), from config.yaml's livekit.tts — production's shaped output."""
    import tts_shaping
    import yaml

    tts = ((yaml.safe_load((ROOT / "config.yaml").read_text()) or {})
           .get("livekit", {}).get("tts") or {})
    sh_cfg = tts.get("shaping") or {}
    sh = tts_shaping.OutputShaper(sr, speed=float(tts.get("speed", 1.0)),
                                  presence_eq=bool(sh_cfg.get("presence_eq", True)),
                                  shelves=sh_cfg.get("shelves"))
    pcm = np.clip(np.rint(audio * 32767), -32768, 32767).astype("<i2").tobytes()
    out = sh.process(pcm) + sh.flush()
    return np.frombuffer(out, "<i2").astype(np.float32) / 32768.0


def synth(args) -> int:
    runner = load_runner(args.runner, args)
    sr = int(runner.sample_rate)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    sents = SENTENCES[: args.n]
    rows = []
    shaper_dir = None
    if args.shape:
        shaper_dir = Path(args.out) / f"{args.name}_shaped"
        shaper_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(args.warmup):
        timed_stream(runner.stream("Warming up the voice."), time.perf_counter())
    for i, text in enumerate(sents):
        t0 = time.perf_counter()
        audio, ttfb, total = timed_stream(runner.stream(text), t0)
        dur = audio.size / sr
        write_wav(out / f"s{i:02d}.wav", audio, sr)
        if shaper_dir is not None:
            write_wav(shaper_dir / f"s{i:02d}.wav", shape(audio, sr), sr)
        row = {"i": i, "text": text, "ttfb_s": ttfb, "wall_s": total, "audio_s": dur,
               "rtf": (total / dur) if dur else None}
        if args.live_text:
            toks = llm_tokens(text)
            cut_i, clause = first_clause_cut(toks)
            t0 = time.perf_counter()
            _, c_ttfb, _ = timed_stream(runner.stream(clause), t0)
            row["clause"] = {"first_clause": clause, "cut_token": cut_i,
                             "text_wait_s": cut_i / args.tok_rate, "ttfb_s": c_ttfb,
                             "ttfa_s": (cut_i / args.tok_rate + c_ttfb) if c_ttfb else None}
            if getattr(runner, "stream_live", None):
                la, l_ttfa, l_total = live_stream(runner, toks, args.tok_rate)
                write_wav(out / f"s{i:02d}_live.wav", la, sr)
                row["live"] = {"ttfa_s": l_ttfa, "wall_s": l_total, "audio_s": la.size / sr,
                               "text_done_s": (len(toks) - 1) / args.tok_rate}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
    vram = runner.vram_mib() if hasattr(runner, "vram_mib") else None
    if vram is None and args.runner != "openai":
        vram = gpu_mib_of(__import__("os").getpid())
    meta = {"name": args.name, "runner": args.runner, "sample_rate": sr,
            "tok_rate": args.tok_rate, "vram_mib": vram, "rows": rows,
            "note": args.note}
    (out / "synth.json").write_text(json.dumps(meta, indent=1))
    if hasattr(runner, "close"):
        runner.close()
    print(f"wrote {out}")
    return 0


# ── score phase ─────────────────────────────────────────────────────────────

def band_profile(x: np.ndarray, sr: int, gate_db: float = 30.0) -> dict:
    """Mean band power (dB), normalised to 100–1500 Hz, over frames gated on
    sub-1 kHz energy. The gate is the part that matters: see module doc."""
    n = 2048 if sr > 30000 else 1024
    hop = n // 4
    if x.size < n:
        return {}
    win = np.hanning(n).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    low = spec[:, (f >= 80) & (f < 1000)].sum(axis=1)
    low_db = 10 * np.log10(low + 1e-12)
    keep = low_db > low_db.max() - gate_db
    p = spec[keep].mean(axis=0)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return 10 * np.log10(p[m].mean() + 1e-20) if m.any() else None

    ref = band(100, 1500)
    return {f"{lo}-{hi}": float(band(lo, hi) - ref) if (hi <= sr / 2 and band(lo, hi) is not None)
            else None for lo, hi in BANDS}


def presence_delta(cand: dict, ref: dict) -> dict:
    d = {k: (cand[k] - ref[k]) if cand.get(k) is not None and ref.get(k) is not None else None
         for k in ref}
    core = [d[k] for k in ("1500-2500", "2500-3500", "3500-5000", "5000-9000") if d.get(k) is not None]
    d["mean_1.5-9k"] = float(np.mean(core)) if core else None
    return d


def score(args) -> int:
    sys.path.insert(0, str(ROOT / "scripts" / "voice"))
    from asr_eval import normalise, edit_distance, to_16k
    from voice import asr as voice_asr

    rec = voice_asr.build_recognizer({"backend": "parakeet", "model_dir": args.parakeet_dir,
                                      "decoding_method": "greedy_search",
                                      "parakeet_hotwords": False, "hotwords": []})
    rec.load()
    enc = None
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        enc = VoiceEncoder(device="cpu")
    except Exception as e:  # noqa: BLE001 — similarity is optional
        print(f"resemblyzer unavailable: {e}")

    rx, rsr = read_wav(Path(args.ref))
    ref_bands = band_profile(rx, rsr)
    ref_emb = enc.embed_utterance(preprocess_wav(rx, source_sr=rsr)) if enc else None
    table = []
    def rate(d: Path, rows: list[dict], suffix: str) -> dict | None:
        """WER, similarity and presence over `sNN<suffix>.wav` in `d`."""
        errs = words = 0
        sims, bands, per = [], [], []
        for row in rows:
            p = d / f"s{row['i']:02d}{suffix}.wav"
            if not p.exists():
                continue
            x, sr = read_wav(p)
            if x.size == 0:
                continue
            hyp = normalise(rec.transcribe(to_16k(x, sr)).text)
            ref = normalise(row["text"])
            e = edit_distance(ref, hyp)
            r = {"i": row["i"], "wer": e / max(1, len(ref)), "hyp": " ".join(hyp)}
            errs, words = errs + e, words + len(ref)
            if enc is not None:
                emb = enc.embed_utterance(preprocess_wav(x, source_sr=sr))
                r["spk_sim"] = float(np.dot(emb, ref_emb))
                sims.append(r["spk_sim"])
            bands.append(band_profile(x, sr))
            per.append(r)
        if not per:
            return None
        cand = {k: float(np.mean([b[k] for b in bands if b.get(k) is not None]))
                if any(b.get(k) is not None for b in bands) else None for k in ref_bands}
        return {"wer": errs / max(1, words), "spk_sim": statistics.mean(sims) if sims else None,
                "presence": presence_delta(cand, ref_bands), "per_sentence": per}

    for d in map(Path, args.dirs):
        meta_p = d / "synth.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        rows = meta["rows"]
        live = [r.get("live", {}).get("ttfa_s") for r in rows]
        clause = [r.get("clause", {}).get("ttfa_s") for r in rows]
        timing = {
            "n": len(rows), "sample_rate": meta["sample_rate"],
            "ttfb_p50": pctl([r["ttfb_s"] for r in rows], .5),
            "ttfb_p90": pctl([r["ttfb_s"] for r in rows], .9),
            "rtf_p50": pctl([r["rtf"] for r in rows], .5),
            "clause_ttfa_p50": pctl(clause, .5), "clause_ttfa_p90": pctl(clause, .9),
            "live_ttfa_p50": pctl(live, .5), "live_ttfa_p90": pctl(live, .9),
            "vram_mib": meta.get("vram_mib"), "note": meta.get("note"),
        }
        whole = rate(d, rows, "")
        out = [{"name": meta["name"], **timing, **whole}] if whole else []
        streamed = rate(d, rows, "_live")
        if streamed:
            out.append({"name": meta["name"] + " (live text)", **timing, **streamed})
        (d / "score.json").write_text(json.dumps(out, indent=1))
        table += out
    print(render(table))
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
    return 0


def _f(v, fmt="{:.2f}"):
    return "—" if v is None else fmt.format(v)


def render(table: list[dict]) -> str:
    head = ("| candidate | TTFB p50/p90 s | RTF | clause TTFA p50 | live TTFA p50 | "
            "WER | spk sim | presence 1.5–9k dB (2.5–3.5k / 5–9k) | VRAM MiB |")
    lines = [head, "|" + "---|" * 9]
    for s in table:
        p = s["presence"]
        lines.append(
            f"| {s['name']} | {_f(s['ttfb_p50'])}/{_f(s['ttfb_p90'])} | {_f(s['rtf_p50'])} | "
            f"{_f(s['clause_ttfa_p50'])} | {_f(s['live_ttfa_p50'])} | {_f(s['wer'], '{:.1%}')} | "
            f"{_f(s['spk_sim'])} | {_f(p.get('mean_1.5-9k'), '{:+.1f}')} "
            f"({_f(p.get('2500-3500'), '{:+.1f}')} / {_f(p.get('5000-9000'), '{:+.1f}')}) | "
            f"{s['vram_mib'] if s['vram_mib'] is not None else '—'} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--name", required=True)
    s.add_argument("--runner", default="openai", help="'openai' or path.py[:factory]")
    s.add_argument("--runner-args", default=None, help="JSON kwargs for the factory")
    s.add_argument("--url", default="http://127.0.0.1:8090")
    s.add_argument("--voice", default="clone:dave_cullen")
    s.add_argument("--model", default="qwen3-tts")
    s.add_argument("--sample-rate", type=int, default=24000)
    s.add_argument("--extra", default=None, help="JSON merged into the request body")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=len(SENTENCES))
    s.add_argument("--warmup", type=int, default=1)
    s.add_argument("--live-text", action="store_true",
                   help="also measure TTFA with text arriving at --tok-rate")
    s.add_argument("--tok-rate", type=float, default=45.0)
    s.add_argument("--note", default=None)
    s.add_argument("--shape", action="store_true",
                   help="also write the worker-shaped audio to <name>_shaped/")
    c = sub.add_parser("score")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--ref", required=True, help="the clone's reference clip")
    c.add_argument("--parakeet-dir", default=str(ROOT / "agent-services/models/parakeet-tdt-v3"))
    c.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return synth(args) if args.cmd == "synth" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
