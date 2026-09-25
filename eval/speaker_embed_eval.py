#!/usr/bin/env python3
"""Speaker-embedding bake-off for the voice worker (voice plan Phase 4).

Three steps, each re-runnable, all offline once the inputs are on disk:

  prepare  — cut the evaluation clips to 16 kHz mono and write a manifest:
             * LibriSpeech test-clean (40 speakers): 10 utterances each,
               one 1/2/3/4 s crop per utterance (the worker's utterance lengths);
             * the wake-diagnostic room corpus (`app.ww_diag.diag_dir()`, read-only):
               every utterance >= 1 s, capped at 8 s (spread is measured
               on the ones with a transcript, own-voice keep-rate on all);
             * renders of Lloyd's own cloned voice (a directory of TTS wavs):
               the first ``--enroll-renders`` are the enrolment set, the rest
               are cut into 1-4 s test segments, clean and "through the room"
               (band-limited 150-6000 Hz + noise at 15 dB SNR).
  embed    — embed every clip with one model; writes ``emb_<model>.npz``.
             Run under whichever python has that model's dependencies:
             resemblyzer / campplus / wespeaker-campplus-lm / campplus-sherpa in the
             lloyd venv,
             redimnet-b2 in a scratch venv with torch + torchaudio.
  score    — EER + same/different cosine distributions, own-voice rejection,
             the room corpus's spread, and threshold suggestions, per model.

    python eval/speaker_embed_eval.py prepare --work DIR --librispeech DIR/LibriSpeech/test-clean \
        --lloyd DIR/tts   # --room defaults to the live ww_diag corpus
    python eval/speaker_embed_eval.py embed --work DIR --model campplus --model-path X.onnx
    python eval/speaker_embed_eval.py score --work DIR

Results of the 2026-09-24 run are in architecture/voice.md ("Who is speaking").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000


# ── audio helpers ────────────────────────────────────────────────────────

def _read(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a, sr


def _to16k(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return a.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr, SR)
    return resample_poly(a, SR // g, sr // g).astype(np.float32)


def _write(path: Path, a: np.ndarray) -> None:
    import soundfile as sf
    sf.write(str(path), np.clip(a, -1, 1), SR, subtype="PCM_16")


def _roomify(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(4, [150, 6000], btype="band", fs=SR, output="sos")
    b = sosfilt(sos, a)
    p = float(np.mean(b ** 2)) + 1e-12
    noise = rng.standard_normal(b.size) * np.sqrt(p / (10 ** (15 / 10)))
    return (b + noise).astype(np.float32)


# ── prepare ──────────────────────────────────────────────────────────────

def prepare(args) -> None:
    work = Path(args.work)
    clips = work / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    manifest: list[dict] = []

    # LibriSpeech
    spk_dirs = sorted(p for p in Path(args.librispeech).iterdir() if p.is_dir())
    for spk in spk_dirs:
        flacs = sorted(glob.glob(str(spk / "*" / "*.flac")))
        rng.shuffle(flacs)
        n = 0
        for f in flacs:
            a, sr = _read(f)
            a = _to16k(a, sr)
            dur = [1.0, 2.0, 3.0, 4.0][n % 4]
            L = int(dur * SR)
            if a.size < L + SR // 2:
                continue
            off = int(rng.integers(0, a.size - L))
            cid = f"ls_{spk.name}_{n:02d}"
            _write(clips / f"{cid}.wav", a[off:off + L])
            manifest.append({"id": cid, "set": "libri", "spk": spk.name, "dur": dur})
            n += 1
            if n >= args.per_speaker:
                break

    # Room corpus (read-only source; metadata from scores.jsonl)
    if args.room:
        room = Path(args.room).expanduser()
    else:
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from app.ww_diag import diag_dir
        room = diag_dir()
    meta = {}
    for line in (room / "scores.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        meta[r.get("utterance_id")] = r
    for wav in sorted((room / "utterances").glob("*.wav")):
        r = meta.get(wav.stem, {})
        text = (r.get("stt_text") or "").strip()
        a, sr = _read(str(wav))
        a = _to16k(a, sr)
        dur = a.size / SR
        if dur < 1.0:
            continue
        a = a[: 8 * SR]
        cid = f"room_{wav.stem}"
        _write(clips / f"{cid}.wav", a)
        manifest.append({"id": cid, "set": "room", "spk": "room", "dur": round(min(dur, 8.0), 2),
                         "text": text[:120], "has_text": bool(text.split())})

    # Lloyd's cloned voice
    renders = sorted(glob.glob(str(Path(args.lloyd) / "*.wav")),
                     key=lambda p: int("".join(c for c in Path(p).stem if c.isdigit()) or 0))
    for i, f in enumerate(renders):
        a, sr = _read(f)
        a = _to16k(a, sr)
        if i < args.enroll_renders:
            cid = f"lloyd_enroll_{i:02d}"
            _write(clips / f"{cid}.wav", a)
            manifest.append({"id": cid, "set": "lloyd_enroll", "spk": "lloyd", "dur": round(a.size / SR, 2)})
            continue
        pos, k = 0, 0
        while True:
            dur = [1.0, 2.0, 3.0, 4.0][k % 4]
            L = int(dur * SR)
            if pos + L > a.size:
                break
            seg = a[pos:pos + L]
            for variant, x in (("clean", seg), ("room", _roomify(seg, rng))):
                cid = f"lloyd_{variant}_{i:02d}_{k:02d}"
                _write(clips / f"{cid}.wav", x)
                manifest.append({"id": cid, "set": f"lloyd_{variant}", "spk": "lloyd", "dur": dur})
            pos += SR // 2  # overlapping windows, 0.5 s hop
            k += 1

    (work / "manifest.json").write_text(json.dumps(manifest, indent=1))
    by = {}
    for m in manifest:
        by[m["set"]] = by.get(m["set"], 0) + 1
    print(json.dumps(by))


# ── embedders ────────────────────────────────────────────────────────────

def _embedder(name: str, model_path: str | None, threads: int):
    """``resemblyzer`` / ``campplus`` / ``wespeaker-campplus-lm`` go through the
    shipped code path (agent-services/speaker_id.py); ``campplus-sherpa`` is the
    same CAM++ file through sherpa-onnx's own extractor, kept to show why the
    shipped path does not use it."""
    if name in ("resemblyzer", "campplus", "wespeaker-campplus-lm"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-services"))
        import speaker_id as sid
        if name == "resemblyzer":
            enc = sid._ResemblyzerEncoder("cpu")
        else:
            enc = sid._CampplusEncoder(model_path, threads)
        return enc.embed
    if name == "campplus-sherpa":
        import sherpa_onnx
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path, num_threads=threads, debug=False, provider="cpu"))

        def f(a):
            s = ex.create_stream()
            s.accept_waveform(SR, a)
            s.input_finished()
            return np.asarray(ex.compute(s), dtype=np.float32)
        return f
    if name == "redimnet-b2":
        import torch
        torch.set_num_threads(threads)
        sys.path.insert(0, os.environ["REDIMNET_REPO"])
        from redimnet.model import ReDimNetWrap
        sd = torch.load(model_path, map_location="cpu")
        m = ReDimNetWrap(**sd["model_config"])
        m.load_state_dict(sd["state_dict"])
        m.eval()

        def f(a):
            with torch.inference_mode():
                return m(torch.from_numpy(a)[None]).numpy()[0]
        return f
    raise SystemExit(f"unknown model {name}")


def embed(args) -> None:
    work = Path(args.work)
    manifest = json.loads((work / "manifest.json").read_text())
    f = _embedder(args.model, args.model_path, args.threads)
    ids, embs, lat = [], [], {}
    # warm-up
    f(np.zeros(SR, dtype=np.float32) + 1e-3 * np.random.default_rng(0).standard_normal(SR).astype(np.float32))
    for m in manifest:
        a, _ = _read(str(work / "clips" / f"{m['id']}.wav"))
        t = time.perf_counter()
        e = np.asarray(f(a), dtype=np.float32)
        dt = time.perf_counter() - t
        if m["set"] == "libri":
            lat.setdefault(str(m["dur"]), []).append(dt)
        e = e / (np.linalg.norm(e) + 1e-9)
        ids.append(m["id"])
        embs.append(e)
    np.savez(work / f"emb_{args.model}.npz", ids=np.array(ids), emb=np.stack(embs))
    (work / f"lat_{args.model}.json").write_text(json.dumps(
        {k: {"median_ms": 1000 * float(np.median(v)), "p95_ms": 1000 * float(np.percentile(v, 95))}
         for k, v in lat.items()}))
    print(args.model, np.stack(embs).shape)


# ── scoring ──────────────────────────────────────────────────────────────

def _eer(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    """(EER, threshold at the EER) by a sweep over the observed scores."""
    ths = np.unique(np.concatenate([same, diff]))
    ss, ds = np.sort(same), np.sort(diff)
    frr = np.searchsorted(ss, ths, side="left") / ss.size          # same < t
    far = 1.0 - np.searchsorted(ds, ths, side="left") / ds.size    # diff >= t
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(ths[i])


def _pct(x, qs=(5, 50, 95)):
    return [round(float(np.percentile(x, q)), 3) for q in qs]


def score_one(work: Path, model: str) -> dict:
    manifest = {m["id"]: m for m in json.loads((work / "manifest.json").read_text())}
    z = np.load(work / f"emb_{model}.npz")
    ids = list(z["ids"])
    E = z["emb"]
    sets = np.array([manifest[i]["set"] for i in ids])
    spk = np.array([manifest[i]["spk"] for i in ids])
    dur = np.array([manifest[i]["dur"] for i in ids], dtype=float)
    out: dict = {"model": model, "dim": int(E.shape[1])}

    # pairwise trials on LibriSpeech (the anchor task: one clip vs one clip)
    L = np.where(sets == "libri")[0]
    S = E[L] @ E[L].T
    iu = np.triu_indices(L.size, 1)
    same_mask = (spk[L][:, None] == spk[L][None, :])[iu]
    sc = S[iu]
    eer, t_eer = _eer(sc[same_mask], sc[~same_mask])
    out["pair_eer"] = round(eer * 100, 2)
    out["pair_same_p5_50_95"] = _pct(sc[same_mask])
    out["pair_diff_p5_50_95"] = _pct(sc[~same_mask])
    # Threshold that keeps 95% of same-speaker pairs (anchor: a false reject drops a real follow-up)
    t_anchor = float(np.percentile(sc[same_mask], 5))
    out["anchor_t_frr5"] = round(t_anchor, 3)
    out["anchor_far_at_t"] = round(float(np.mean(sc[~same_mask] >= t_anchor)) * 100, 2)
    # EER by the shorter clip of the pair
    dmin = np.minimum(dur[L][:, None], dur[L][None, :])[iu]
    out["pair_eer_by_min_dur"] = {}
    for d in (1.0, 2.0, 3.0, 4.0):
        k = dmin == d
        if k.sum() and (k & same_mask).sum():
            out["pair_eer_by_min_dur"][str(d)] = round(_eer(sc[k & same_mask], sc[k & ~same_mask])[0] * 100, 2)

    # profile trials: 3-clip averaged enrolment vs the rest (identify())
    tgt, imp = [], []
    spks = sorted(set(spk[L]))
    profiles = {}
    tests = {}
    for s in spks:
        idx = L[spk[L] == s]
        p = E[idx[:3]].mean(0)
        profiles[s] = p / np.linalg.norm(p)
        tests[s] = idx[3:]
    for s in spks:
        for s2 in spks:
            v = E[tests[s2]] @ profiles[s]
            (tgt if s == s2 else imp).extend(v.tolist())
    tgt, imp = np.array(tgt), np.array(imp)
    out["profile_eer"] = round(_eer(tgt, imp)[0] * 100, 2)
    t_prof = float(np.percentile(imp, 99))  # FAR 1%
    out["profile_t_far1"] = round(t_prof, 3)
    out["profile_frr_at_t"] = round(float(np.mean(tgt < t_prof)) * 100, 2)
    out["profile_tgt_p5_50_95"] = _pct(tgt)
    out["profile_imp_p5_50_95"] = _pct(imp)

    # room corpus spread
    R = np.where(sets == "room")[0]
    RT = np.array([r for r in R if manifest[ids[r]].get("has_text")], dtype=int)
    if RT.size > 1:
        RS = E[RT] @ E[RT].T
        ru = np.triu_indices(RT.size, 1)
        out["room_n_all"] = int(R.size)
        out["room_n_speech"] = int(RT.size)
        out["room_pair_p5_50_95"] = _pct(RS[ru])
        c = E[RT].mean(0)
        c /= np.linalg.norm(c)
        out["room_to_centroid_p5_50_95"] = _pct(E[RT] @ c)

    # own-voice rejection
    En = np.where(sets == "lloyd_enroll")[0]
    if En.size and R.size:
        p = E[En].mean(0)
        p /= np.linalg.norm(p)
        room_s = E[R] @ p
        res = {}
        for variant in ("lloyd_clean", "lloyd_room"):
            V = np.where(sets == variant)[0]
            ls = E[V] @ p
            lo = float(np.percentile(ls, 5))       # 95% of Lloyd segments at or above
            hi = float(np.percentile(room_s, 95))  # 95% of room utterances below
            # best single threshold: maximise min(lloyd reject rate, room keep rate)
            cand = np.unique(np.concatenate([ls, room_s]))
            best = max(cand, key=lambda t: min(np.mean(ls >= t), np.mean(room_s < t)))
            res[variant] = {
                "n": int(V.size),
                "lloyd_p5_50_95": _pct(ls),
                "lloyd_p5": round(lo, 3),
                "room_p95": round(hi, 3),
                "margin": round(lo - hi, 3),
                "best_t": round(float(best), 3),
                "reject_at_best": round(float(np.mean(ls >= best)) * 100, 1),
                "keep_at_best": round(float(np.mean(room_s < best)) * 100, 1),
            }
        out["own_voice"] = res
        out["room_vs_lloyd_p5_50_95_max"] = _pct(room_s) + [round(float(room_s.max()), 3)]
        top = np.argsort(-room_s)[:5]
        out["room_top_vs_lloyd"] = [(ids[R[i]], round(float(room_s[i]), 3),
                                     manifest[ids[R[i]]].get("text", "")[:60]) for i in top]
    latf = work / f"lat_{model}.json"
    if latf.exists():
        out["latency_ms"] = json.loads(latf.read_text())
    return out


def score(args) -> None:
    work = Path(args.work)
    models = args.model or [p.stem[4:] for p in sorted(work.glob("emb_*.npz"))]
    results = [score_one(work, m) for m in models]
    (work / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--librispeech", required=True)
    p.add_argument("--room", default=None,
                   help="the room corpus; default app.ww_diag.diag_dir()")
    p.add_argument("--lloyd", required=True)
    p.add_argument("--per-speaker", type=int, default=10)
    p.add_argument("--enroll-renders", type=int, default=5)
    p.set_defaults(fn=prepare)
    e = sub.add_parser("embed")
    e.add_argument("--work", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--model-path")
    e.add_argument("--threads", type=int, default=2)
    e.set_defaults(fn=embed)
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--model", action="append")
    s.set_defaults(fn=score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
