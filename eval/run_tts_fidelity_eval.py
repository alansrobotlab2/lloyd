#!/usr/bin/env python
"""Does the cloned voice say the words it is sent? A TTS -> ASR round trip (#1165).

The TTS health rule proves a waveform came out, not that it holds the words.
This synthesises a pinned corpus of Lloyd's real spoken surface
(`eval/tts_fidelity_corpus.yaml`: household names, tool names, an email, a
phone number, a date, ratios, counts, a path, a bullet reply) through the
production path — `voice.speakable.ClauseStream` cuts it into clauses, each
clause is synthesised as `clone:dave_cullen` and shaped the way the worker
shapes it — then transcribes the audio with Parakeet **greedy, no hotword
list**, so the rater cannot "hear" a name because it was told to expect one.

Two arms over the same audio path: `raw` (the text as written, which is what
production sends today) and `for_speech` (`agent-services/tts_text.py`). Each
item is synthesised k times, because synthesis is not deterministic.

Scored per bucket and arm: target-token accuracy (targets heard / targets
sent) and exact match (the whole transcript, spacing ignored), with n beside
every rate. The benign `control` bucket is reported on its own — if it fails,
the rater is the finding, not the voice. A bucket with n=0 is "no verdict",
never a pass. The result goes to `<data root>/eval/baselines/` as a dated JSON.

    .venvs/lloyd/bin/python eval/run_tts_fidelity_eval.py            # k=2, both arms
    .venvs/lloyd/bin/python eval/run_tts_fidelity_eval.py --k 3 --arms raw

Needs the TTS server on :8090 and the Parakeet model on disk. Measuring is a
human step: the result decides whether `livekit.tts.normalise_text` is turned on.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import re
import sys
import urllib.request
import wave
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-services"))
sys.path.insert(0, str(ROOT / "scripts" / "voice"))
# Appended, not inserted: `app/` would shadow agent-services modules otherwise.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from asr_eval import decode_audio, normalise, to_16k  # noqa: E402
import tts_text  # noqa: E402
from voice.speakable import ClauseStream  # noqa: E402

CORPUS = ROOT / "eval" / "tts_fidelity_corpus.yaml"
VOICE = "clone:dave_cullen"
ARMS = ("raw", "for_speech")
CONTROL = "control"

#: The rater. Greedy is production's plain decode; `parakeet_hotwords: False`
#: is what makes it unbiased — production biases Parakeet toward a hotword
#: list in beam search, so "no list" is a setting, not a default.
RATER_CFG = {
    "backend": "parakeet",
    "model_dir": str(ROOT / "agent-services" / "models" / "parakeet-tdt-v3"),
    "decoding_method": "greedy_search",
    "parakeet_hotwords": False,
    "hotwords": [],
}

Synth = Callable[[str], bytes]           # clause -> wav bytes
Transcribe = Callable[[bytes], str]      # wav bytes -> transcript

_NUM = re.compile(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?")


# ── corpus and scoring ──────────────────────────────────────────────────────

def load_corpus(path: Path = CORPUS) -> list[dict]:
    items = (yaml.safe_load(Path(path).read_text()) or {}).get("items") or []
    for it in items:
        if not it.get("id") or not it.get("bucket") or not it.get("text"):
            raise ValueError(f"corpus item missing id/bucket/text: {it!r}")
        if not it.get("targets"):
            raise ValueError(f"corpus item {it['id']} has no targets")
    return items


def clauses(text: str) -> list[str]:
    """What production would synthesise, clause by clause."""
    cs = ClauseStream()
    return cs.feed(text) + cs.flush()


def _spell_number(m: re.Match) -> str:
    s = m.group(0)
    if s[-2:] in ("st", "nd", "rd", "th"):
        s = s[:-2]
        return tts_text.number_words(s) if "." in s else tts_text.ordinal_words(int(s))
    if "." not in s and len(s) == 4 and 1100 <= int(s) <= 2099:
        return tts_text.year_words(int(s))
    return tts_text.number_words(s)


def tokens(text: str) -> list[str]:
    """Words for scoring. A transcript writes "2026" or "3.1x" as often as the
    words, and `asr_eval.normalise` keeps letters only, so both hypothesis and
    reference go through the same spelling first: `for_speech`, then any bare
    number as words."""
    return normalise(_NUM.sub(_spell_number, tts_text.for_speech(text)))


def contains_target(hyp: list[str], target: str) -> bool:
    """Any `|` alternative present as a word run — or as the same letters
    spaced differently ("QMD" against "Q M D")."""
    for alt in target.split("|"):
        want = tokens(alt)
        if not want:
            continue
        n = len(want)
        if any(hyp[i:i + n] == want for i in range(len(hyp) - n + 1)):
            return True
        compact = "".join(want)
        for i in range(len(hyp)):
            for j in range(i + 1, min(len(hyp), i + n + 4) + 1):
                if "".join(hyp[i:j]) == compact:
                    return True
    return False


def exact_match(hyp: list[str], reference: str) -> bool:
    ref = tokens(" ".join(clauses(reference)))
    return "".join(hyp) == "".join(ref)


def score_sample(item: dict, transcript: str) -> dict:
    hyp = tokens(transcript)
    hits = [t for t in item["targets"] if contains_target(hyp, t)]
    return {
        "transcript": transcript,
        "targets_hit": len(hits),
        "targets_total": len(item["targets"]),
        "missed": [t for t in item["targets"] if t not in hits],
        "exact": exact_match(hyp, item.get("say") or item["text"]),
    }


def arm_text(arm: str, clause: str) -> str:
    return tts_text.for_speech(clause) if arm == "for_speech" else clause


# ── the run ─────────────────────────────────────────────────────────────────

def run(items: list[dict], synth: Synth, transcribe: Transcribe, k: int = 2,
        arms=ARMS, join: Callable[[list[bytes]], bytes] | None = None) -> dict:
    """Synthesise every item k times per arm and score it. `synth` takes one
    clause, `transcribe` one whole item's audio (the clauses joined by `join`,
    `join_wavs` unless a caller stubs the audio)."""
    join = join or join_wavs
    if k < 1:
        raise ValueError("k must be >= 1")
    samples = []
    for arm in arms:
        for it in items:
            sent = [arm_text(arm, c) for c in clauses(it["text"])]
            for i in range(k):
                wav = join([synth(c) for c in sent])
                s = score_sample(it, transcribe(wav))
                samples.append({"id": it["id"], "bucket": it["bucket"], "arm": arm,
                                "sample": i, "sent": sent, **s})
    return summarise(samples, items, k, arms)


def _rate(hit: int, n: int) -> dict:
    return {"hit": hit, "n": n, "rate": (hit / n) if n else None}


def summarise(samples: list[dict], items: list[dict], k: int, arms) -> dict:
    buckets = sorted({it["bucket"] for it in items} - {CONTROL})
    out = {"k": k, "arms": {}}
    for arm in arms:
        rows = [s for s in samples if s["arm"] == arm]
        per = {}
        for b in buckets + [CONTROL]:
            br = [s for s in rows if s["bucket"] == b]
            per[b] = {
                "items": len({s["id"] for s in br}),
                "target": _rate(sum(s["targets_hit"] for s in br),
                                sum(s["targets_total"] for s in br)),
                "exact": _rate(sum(s["exact"] for s in br), len(br)),
            }
        test = [s for s in rows if s["bucket"] != CONTROL]
        out["arms"][arm] = {
            "buckets": {b: per[b] for b in buckets},
            "control": per[CONTROL],
            "overall": {
                "target": _rate(sum(s["targets_hit"] for s in test),
                                sum(s["targets_total"] for s in test)),
                "exact": _rate(sum(s["exact"] for s in test), len(test)),
            },
        }
    out["samples"] = samples
    return out


def fmt_rate(r: dict) -> str:
    if not r["n"]:
        return "no verdict (n=0)"
    return f"{r['rate'] * 100:5.1f}% ({r['hit']}/{r['n']})"


def report(result: dict) -> str:
    lines = [f"k={result['k']} per item; target = target tokens heard, "
             "exact = whole transcript"]
    for arm, a in result["arms"].items():
        lines.append(f"\n[{arm}]")
        lines.append(f"  {'bucket':<14} {'items':>5}  {'target':<22} exact")
        for b, r in a["buckets"].items():
            lines.append(f"  {b:<14} {r['items']:>5}  {fmt_rate(r['target']):<22} "
                         f"{fmt_rate(r['exact'])}")
        o = a["overall"]
        lines.append(f"  {'OVERALL':<14} {'':>5}  {fmt_rate(o['target']):<22} "
                     f"{fmt_rate(o['exact'])}")
        c = a["control"]
        lines.append(f"  control arm (benign, scored apart): items={c['items']}  "
                     f"target {fmt_rate(c['target'])}  exact {fmt_rate(c['exact'])}")
    return "\n".join(lines)


def default_out_dir() -> Path:
    from app.paths import EVAL_BASELINES_DIR
    return EVAL_BASELINES_DIR


def write_result(result: dict, out_dir: Path, meta: dict,
                 today: _dt.date | None = None) -> Path:
    day = (today or _dt.date.today()).isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tts_fidelity_{day}.json"
    path.write_text(json.dumps({"date": day, **meta, **result}, indent=1))
    return path


# ── live synthesis and rater (not used by the tests) ────────────────────────

def pcm_to_wav(pcm: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def join_wavs(wavs: list[bytes], gap_ms: int = 250) -> bytes:
    """Concatenate clause audio with the worker's tail silence between them."""
    parts, sr = [], 24000
    for w in wavs:
        x, sr = decode_audio({"bytes": w})
        parts += [x, np.zeros(int(sr * gap_ms / 1000), np.float32)]
    x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype("<i2").tobytes()
    return pcm_to_wav(pcm, sr)


def live_synth(tts_cfg: dict, voice: str = VOICE, shape: bool = True) -> Synth:
    import tts_shaping

    api = (tts_cfg.get("api_url") or "http://127.0.0.1:8090").rstrip("/")
    sr = int(tts_cfg.get("sample_rate", 24000))
    shaping = tts_cfg.get("shaping") or {}

    def synth(text: str) -> bytes:
        req = urllib.request.Request(
            f"{api}/v1/audio/speech",
            data=json.dumps({"model": tts_cfg.get("model", "qwen3-tts"), "input": text,
                             "voice": voice, "response_format": "pcm", "stream": True,
                             "speed": 1.0}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            pcm = r.read()
        if shape:
            # A fresh shaper per clause, as the worker resets its own per utterance.
            sh = tts_shaping.OutputShaper(
                sr, speed=float(tts_cfg.get("speed", 1.0)),
                presence_eq=bool(shaping.get("presence_eq", True)),
                shelves=shaping.get("shelves"))
            pcm = sh.process(pcm) + sh.flush()
        return pcm_to_wav(pcm, sr)

    return synth


def build_rater():
    from voice import asr as voice_asr
    return voice_asr.build_recognizer(dict(RATER_CFG))


def live_transcribe(rec) -> Transcribe:
    def transcribe(wav: bytes) -> str:
        x, sr = decode_audio({"bytes": wav})
        return rec.transcribe(to_16k(x, sr)).text
    return transcribe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--k", type=int, default=2, help="samples per item (>=2)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--no-shaping", action="store_true",
                    help="score the server's audio, not the worker's shaped output")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to <data root>/eval/baselines")
    args = ap.parse_args(argv)

    items = load_corpus(Path(args.corpus))
    tts_cfg = ((yaml.safe_load((ROOT / "config.yaml").read_text()) or {})
               .get("livekit", {}).get("tts") or {})
    rec = build_rater()
    rec.load()
    result = run(items, live_synth(tts_cfg, args.voice, not args.no_shaping),
                 live_transcribe(rec), k=args.k, arms=args.arms)
    print(report(result))
    meta = {"voice": args.voice, "rater": RATER_CFG, "shaped": not args.no_shaping,
            "speed": tts_cfg.get("speed"), "corpus": str(args.corpus)}
    path = write_result(result, Path(args.out_dir) if args.out_dir else default_out_dir(),
                        meta)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
