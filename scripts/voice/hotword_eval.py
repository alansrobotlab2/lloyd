#!/usr/bin/env python
"""Does biasing Parakeet toward Lloyd's names help, and what does it cost?

Hotword biasing is only available in beam search, and on this model beam
search alone is worse than greedy (2026-09-18, below), so three decoders are
compared: greedy (production), beam with the list (`hw@s`), and the
combination production would run — greedy unless the biased decode gained a
hotword greedy lacks (`pick@s`, `voice.asr.prefer_biased`). Four corpora:

  targets      ~/.cache/lloyd-voice-eval/hotwords/t_*.wav — sentences that
               contain a hotword ("Tell Lisa…", "…the backlog…"). Recall of
               the hotword, per word and overall, and WER.
  distractors  …/d_*.wav — sound-alikes that must NOT become a hotword
               ("the back lot", "Emily", "stomping", "Floyd"). False
               insertions, and WER.
  LibriSpeech  the ASR eval corpus, clean and at 10 dB — general WER, which a
               bias toward a few names must not move.
  real room    ~/.lloyd/ww_diag/utterances — no references, so the count is
               of transcripts that CHANGE against greedy, printed for reading.

Targets and distractors are Qwen3-TTS built-in voices, which the ASR model
was not trained on, made by `synth_hotword_corpus.py`. `manifest.json`
({path, kind: t|d, voice, text, hotword}) is the only contract; any list of
sentences will do.

    python scripts/voice/hotword_eval.py --scores 1.0 1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-services"))
sys.path.insert(0, str(ROOT / "scripts" / "voice"))

from asr_eval import add_noise, decode_audio, edit_distance, normalise, to_16k  # noqa: E402
from replay import load_16k  # noqa: E402
from voice import asr as voice_asr  # noqa: E402

CORPUS = Path.home() / ".cache" / "lloyd-voice-eval" / "hotwords"
LIBRI = Path.home() / ".cache" / "lloyd-voice-eval" / "librispeech_dummy_clean_validation.parquet"
DIAG = Path.home() / ".lloyd" / "ww_diag"
MODEL = ROOT / "agent-services" / "models" / "parakeet-tdt-v3"

# Spoken forms a hotword may legitimately come out as.
SPOKEN = {"gr00t": ["gr00t", "groot"]}


def has_hotword(hyp: str, hotword: str) -> bool:
    return any(voice_asr.contains_hotword(hyp, f) for f in SPOKEN.get(hotword, [hotword]))


def wer(pairs) -> float:
    errs = sum(edit_distance(normalise(r), normalise(h)) for r, h in pairs)
    return errs / max(1, sum(len(normalise(r)) for r, _ in pairs))


def config_hotwords() -> list[str]:
    """The list production biases toward: config plus the vault file, minus the
    wake words — built by the same function the worker calls."""
    lk = (yaml.safe_load((ROOT / "config.yaml").read_text()) or {}).get("livekit", {})
    wake = {t for w in (lk.get("wake") or {}).get("words") or [] for t in str(w).split()}
    return voice_asr.parakeet_hotwords(lk.get("stt") or {}, exclude=wake)


def decode_all(rec, clips, libri, real):
    t0n = [0.0, 0]
    def run(a):
        tr = rec.transcribe(a)
        t0n[0] += tr.latency_s; t0n[1] += 1
        return tr.text
    out = {"clips": [run(a) for _, a in clips],
           "libri": [run(a) for _, a, _ in libri],
           "noisy": [run(n) for _, _, n in libri],
           "real": [run(a) for _, a in real]}
    out["ms"] = 1000 * t0n[0] / max(1, t0n[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores", type=float, nargs="+", default=[1.0, 1.5])
    ap.add_argument("--hotwords", nargs="*", default=None,
                    help="override the list from config.yaml")
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--no-libri", action="store_true")
    args = ap.parse_args()

    hotwords = args.hotwords if args.hotwords is not None else config_hotwords()
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    clips = [(m, load_16k(Path(m["path"]))) for m in manifest]
    libri = []
    if LIBRI.exists() and not args.no_libri:
        import pyarrow.parquet as pq
        rng = np.random.default_rng(0)
        for r in pq.read_table(LIBRI).to_pylist():
            a = to_16k(*decode_audio(r["audio"]))
            libri.append((r["text"], a, add_noise(a, 10.0, rng)))
    real = []
    if (DIAG / "utterances").exists():
        real = [(p.stem, load_16k(p)) for p in sorted((DIAG / "utterances").glob("*.wav"))]

    print(f"hotwords ({len(hotwords)}): {', '.join(hotwords)}")
    print(f"targets {sum(m['kind'] == 't' for m, _ in clips)}  distractors "
          f"{sum(m['kind'] == 'd' for m, _ in clips)}  LibriSpeech {len(libri)}  real {len(real)}\n")

    def rec(**kw):
        r = voice_asr.SherpaOfflineRecognizer(MODEL, **kw)
        r.load()
        return r

    g = decode_all(rec(), clips, libri, real)
    arms = [("greedy", g, 0.0)]
    for s in args.scores:
        b = decode_all(rec(hotwords=hotwords, hotwords_score=s), clips, libri, real)
        pick = {k: [voice_asr.prefer_biased(p, q, hotwords) for p, q in zip(g[k], b[k])]
                for k in ("clips", "libri", "noisy", "real")}
        arms += [(f"hw@{s:g}", b, b["ms"]), (f"pick@{s:g}", pick, g["ms"] + b["ms"])]

    per_word = defaultdict(dict)
    print(f"{'arm':9} {'recall':>9} {'tgt WER':>8} {'false ins':>9} {'dis WER':>8} "
          f"{'libri':>7} {'@10dB':>7} {'real Δ':>7} {'ms/utt':>7}")
    details = {}
    for name, d, ms in arms:
        hits = ins = 0
        t_pairs, d_pairs, notes = [], [], []
        words = defaultdict(lambda: [0, 0])
        for (m, _), hyp in zip(clips, d["clips"]):
            if m["kind"] == "t":
                t_pairs.append((m["text"], hyp))
                ok = has_hotword(hyp, m["hotword"])
                hits += ok
                words[m["hotword"]][0] += ok
                words[m["hotword"]][1] += 1
                if not ok:
                    notes.append(f"    miss {m['hotword']:16} {m['voice']:7} {hyp!r}")
            else:
                d_pairs.append((m["text"], hyp))
                bad = [h for h in hotwords if has_hotword(hyp, h)]
                if bad:
                    ins += 1
                    notes.append(f"    FALSE {'/'.join(bad):15} {m['voice']:7} {hyp!r}")
        for w, (k, n) in words.items():
            per_word[w][name] = f"{k}/{n}"
        lc = wer([(ref, h) for (ref, _, _), h in zip(libri, d["libri"])]) if libri else None
        lw = wer([(ref, h) for (ref, _, _), h in zip(libri, d["noisy"])]) if libri else None
        changed = [(a, b) for a, b in zip(g["real"], d["real"]) if a != b]
        print(f"{name:9} {hits:>4}/{len(t_pairs):<4} {wer(t_pairs):>8.2%} {ins:>4}/{len(d_pairs):<4} "
              f"{wer(d_pairs):>8.2%} {'-' if lc is None else f'{lc:.2%}':>7} "
              f"{'-' if lw is None else f'{lw:.2%}':>7} {len(changed):>7} {ms:>7.0f}")
        details[name] = (notes, changed)

    names = [n for n, _, _ in arms]
    print("\nper hotword  " + "  ".join(f"{n:>8}" for n in names))
    for w in sorted(per_word):
        print(f"  {w:16} " + "  ".join(f"{per_word[w].get(n, '-'):>8}" for n in names))
    for name in names:
        if not name.startswith("pick"):
            continue
        notes, changed = details[name]
        print(f"\n{name}:")
        print("\n".join(notes[:args.show]))
        for a, b in changed[:args.show]:
            print(f"    real: {a!r}\n       -> {b!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
