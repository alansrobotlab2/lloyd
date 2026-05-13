#!/usr/bin/env python3
"""Replay captured wake-word utterances through openWakeWord and report
detection-rate vs threshold per client class.

Reads ~/.lloyd/ww_diag/:
  - scores.jsonl       : per-utterance metadata (incl. client_info if present)
  - labels.jsonl       : ground-truth labels written by /api/voice/ww_label
  - utterances/*.wav   : the audio for each segmented utterance
  - misses/*.wav,json  : raw-ring miss dumps (pre-VAD audio, not utterance-segmented)

Mirrors the worker's AcousticWakeWord._detect_blocking exactly:
  - polyphase resample to 16kHz
  - reset the model
  - sweep predict() in 1280-sample (80 ms) chunks
  - keep max score per model

Run:
  .venvs/lloyd/bin/python scripts/ww_replay.py
  .venvs/lloyd/bin/python scripts/ww_replay.py --details
  .venvs/lloyd/bin/python scripts/ww_replay.py --labels-only
  .venvs/lloyd/bin/python scripts/ww_replay.py --replay-misses
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from collections import defaultdict
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.signal import resample_poly

DIAG_DIR = Path("~/.lloyd/ww_diag").expanduser()
SCORES_PATH = DIAG_DIR / "scores.jsonl"
LABELS_PATH = DIAG_DIR / "labels.jsonl"
UTTERANCES_DIR = DIAG_DIR / "utterances"
MISSES_DIR = DIAG_DIR / "misses"

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

DEFAULT_THRESHOLDS = (0.50, 0.40, 0.30, 0.20, 0.15, 0.10)


def _load_config_paths() -> tuple[Path, Path, float]:
    """Read wake-word model paths + production threshold from config.yaml."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    aw = (cfg.get("livekit") or {}).get("acoustic_wake") or {}
    models_dir = REPO_ROOT / aw.get("models_dir", "agent-services/models/wakeword")
    engine_dir = REPO_ROOT / aw.get("engine_dir", "agent-services/models/openwakeword")
    threshold = float(aw.get("threshold", 0.4))
    return models_dir, engine_dir, threshold


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _build_models(model_paths: list[Path], engine_dir: Path, enable_speex_ns: bool):
    """Build an openWakeWord Model exactly the way the worker does, given an
    explicit list of .onnx paths."""
    from openwakeword.model import Model
    ww_paths = [str(p) for p in model_paths]
    if not ww_paths:
        raise SystemExit("no .onnx wake-word model paths provided")
    mel = engine_dir / "melspectrogram.onnx"
    emb = engine_dir / "embedding_model.onnx"
    if not mel.exists() or not emb.exists():
        raise SystemExit(f"missing engine files: {mel}, {emb}")
    return Model(
        wakeword_model_paths=ww_paths,
        melspec_onnx_model_path=str(mel),
        embedding_onnx_model_path=str(emb),
        enable_speex_noise_suppression=enable_speex_ns,
    )


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw != 2:
        raise ValueError(f"need 16-bit PCM, got {sw*8}-bit")
    samples = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return samples, sr


def _score_wav(model, samples: np.ndarray, sr: int) -> dict[str, float]:
    """Return {model_name: max_score} for one wav. Mirrors
    AcousticWakeWord._detect_blocking."""
    if sr != 16000:
        up, down = 16000, sr
        g = gcd(up, down)
        s = resample_poly(samples.astype(np.float32), up // g, down // g)
        s = np.clip(s, -32768, 32767).astype(np.int16)
    else:
        s = samples
    try:
        model.reset()
    except Exception:
        pass
    chunk = 1280  # 80 ms @ 16 kHz
    max_per: dict[str, float] = {n: 0.0 for n in model.models.keys()}
    i = 0
    while i + chunk <= len(s):
        scores = model.predict(s[i:i + chunk])
        for n, sc in scores.items():
            if sc > max_per[n]:
                max_per[n] = float(sc)
        i += chunk
    return max_per


def _client_class(rec: dict) -> str:
    info = rec.get("client_info") or {}
    if not info:
        return "unknown"
    base = "mobile" if info.get("isMobile") else "desktop"
    apm = "raw" if info.get("rawAudio") else "apm"
    return f"{base}/{apm}"


def _classify_dataset(records: list[dict], labels_by_utt: dict[str, dict]
                      ) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into (positive, negative, unlabeled) based on labels.jsonl."""
    pos, neg, unl = [], [], []
    for rec in records:
        utt = rec.get("utterance_id")
        lbl = labels_by_utt.get(utt)
        if lbl is None:
            unl.append(rec)
        elif lbl.get("said_wake_word"):
            pos.append(rec)
        else:
            neg.append(rec)
    return pos, neg, unl


def _format_table(rows: list[tuple], headers: tuple[str, ...]) -> str:
    """Tiny markdown-ish table formatter."""
    cols = [list(headers)] + [[str(c) for c in r] for r in rows]
    widths = [max(len(c[i]) for c in cols) for i in range(len(headers))]
    def fmt(row):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))
    out = [fmt(headers), fmt(["-" * w for w in widths])]
    for r in rows:
        out.append(fmt([str(c) for c in r]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", default=",".join(f"{t:.2f}" for t in DEFAULT_THRESHOLDS),
                    help="comma-separated thresholds to sweep")
    ap.add_argument("--labels-only", action="store_true",
                    help="only include utterances with a ground-truth label")
    ap.add_argument("--details", action="store_true",
                    help="dump per-utterance scoring rows")
    ap.add_argument("--replay-misses", action="store_true",
                    help="also score the raw miss-ring wavs in misses/")
    ap.add_argument("--ns-mode", choices=("off", "on", "both"), default="off",
                    help="speex noise suppression inside openWakeWord "
                         "(default 'off' matches the LiveKit worker; 'on'/'both' "
                         "require the speexdsp_ns extra)")
    ap.add_argument("--models", default=None,
                    help="comma-separated explicit .onnx paths (overrides "
                         "the config-derived models_dir; useful to A/B a "
                         "candidate model before deploying)")
    args = ap.parse_args(argv)

    thresholds = sorted({float(t) for t in args.thresholds.split(",")}, reverse=True)
    models_dir, engine_dir, prod_threshold = _load_config_paths()

    if args.models:
        model_paths = [Path(p).expanduser() for p in args.models.split(",")]
        for p in model_paths:
            if not p.exists():
                raise SystemExit(f"model file not found: {p}")
        print(f"using explicit model paths: {[str(p) for p in model_paths]}")
    else:
        model_paths = sorted(models_dir.glob("*.onnx"))
        if not model_paths:
            raise SystemExit(f"no .onnx files in {models_dir}")
        print(f"models_dir = {models_dir}")

    print(f"engine_dir = {engine_dir}")
    print(f"prod threshold = {prod_threshold}  (livekit.acoustic_wake.threshold)")
    print(f"sweeping = {thresholds}")
    print()

    score_records = _load_jsonl(SCORES_PATH)
    labels = _load_jsonl(LABELS_PATH)
    print(f"scores.jsonl  : {len(score_records):>5} utterance records")
    print(f"labels.jsonl  : {len(labels):>5} ground-truth labels")

    # Most-recent label wins per utterance_id.
    labels_by_utt: dict[str, dict] = {}
    for lbl in labels:
        u = lbl.get("utterance_id")
        if u:
            labels_by_utt[u] = lbl

    if args.labels_only:
        score_records = [r for r in score_records if r.get("utterance_id") in labels_by_utt]
        print(f"after --labels-only filter: {len(score_records)} records")

    # Load openWakeWord variants we need to replay.
    ns_variants = []
    if args.ns_mode in ("off", "both"):
        ns_variants.append(("ns_off", False))
    if args.ns_mode in ("on", "both"):
        ns_variants.append(("ns_on", True))
    print(f"loading {len(ns_variants)} model variant(s): {[v[0] for v in ns_variants]}")
    print()
    variants = {}
    for name, ns in ns_variants:
        try:
            variants[name] = _build_models(model_paths, engine_dir, ns)
        except ModuleNotFoundError as e:
            print(f"[warn] variant {name!r} unavailable: {e} "
                  f"(skipping; pip install openwakeword[full] to enable speex NS)",
                  file=sys.stderr)
    if not variants:
        raise SystemExit("no model variants loaded — aborting")

    # Replay every utterance wav.
    per_utt: list[dict] = []
    missing_wav = 0
    for rec in score_records:
        wav_str = rec.get("audio_path")
        if not wav_str:
            missing_wav += 1
            continue
        wav = Path(wav_str)
        if not wav.exists():
            missing_wav += 1
            continue
        try:
            samples, sr = _read_wav(wav)
        except Exception as e:
            print(f"[warn] could not read {wav}: {e}", file=sys.stderr)
            missing_wav += 1
            continue
        if samples.size == 0:
            continue
        scored: dict[str, dict[str, float]] = {}
        for vname, model in variants.items():
            scored[vname] = _score_wav(model, samples, sr)
        per_utt.append({
            "rec": rec,
            "scored": scored,
            "label": labels_by_utt.get(rec.get("utterance_id")),
            "client_class": _client_class(rec),
        })

    if missing_wav:
        print(f"[note] skipped {missing_wav} record(s) with missing/unreadable wavs", file=sys.stderr)
    print(f"replayed {len(per_utt)} utterance(s)")
    print()

    # Aggregate detection rate per (client_class, label, threshold, variant).
    pos, neg, unl = _classify_dataset([p["rec"] for p in per_utt], labels_by_utt)
    print(f"corpus split: positive={len(pos)}  negative={len(neg)}  unlabeled={len(unl)}")
    print()

    # Build aggregate tables.
    for vname in variants:
        # detection rate (recall) on labeled positives
        if pos:
            print(f"=== detection rate on labeled-positive utterances ({vname}) ===")
            rows = []
            classes = sorted({_client_class(r) for r in pos})
            for cls in classes:
                cls_n = sum(1 for r in pos if _client_class(r) == cls)
                row = [cls, f"{cls_n}"]
                for t in thresholds:
                    fired = sum(
                        1 for p in per_utt
                        if p["rec"] in pos
                        and _client_class(p["rec"]) == cls
                        and max(p["scored"][vname].values()) >= t
                    )
                    pct = 100.0 * fired / max(1, cls_n)
                    row.append(f"{fired}/{cls_n} ({pct:>4.0f}%)")
                rows.append(row)
            row = ["ALL", str(len(pos))]
            for t in thresholds:
                fired = sum(
                    1 for p in per_utt
                    if p["rec"] in pos and max(p["scored"][vname].values()) >= t
                )
                pct = 100.0 * fired / max(1, len(pos))
                row.append(f"{fired}/{len(pos)} ({pct:>4.0f}%)")
            rows.append(row)
            headers = ("source", "n") + tuple(f"thr={t:.2f}" for t in thresholds)
            print(_format_table(rows, headers))
            print()
        else:
            print(f"=== detection rate on labeled-positive utterances ({vname}) ===")
            print("no labeled-positive utterances yet — POST /api/voice/ww_label "
                  "with said_wake_word=true to populate")
            print()

        # false-fire rate on labeled negatives
        if neg:
            print(f"=== false-fire rate on labeled-negative utterances ({vname}) ===")
            rows = []
            classes = sorted({_client_class(r) for r in neg})
            for cls in classes:
                cls_n = sum(1 for r in neg if _client_class(r) == cls)
                row = [cls, f"{cls_n}"]
                for t in thresholds:
                    fired = sum(
                        1 for p in per_utt
                        if p["rec"] in neg
                        and _client_class(p["rec"]) == cls
                        and max(p["scored"][vname].values()) >= t
                    )
                    pct = 100.0 * fired / max(1, cls_n)
                    row.append(f"{fired}/{cls_n} ({pct:>4.0f}%)")
                rows.append(row)
            headers = ("source", "n") + tuple(f"thr={t:.2f}" for t in thresholds)
            print(_format_table(rows, headers))
            print()

        # how many unlabeled-firings at each threshold (a useful early-warning
        # signal for false-accepts while labels are still sparse)
        if unl:
            print(f"=== fire rate on UNLABELED utterances ({vname}) — useful before labels accumulate ===")
            rows = []
            classes = sorted({_client_class(r) for r in unl})
            for cls in classes:
                cls_n = sum(1 for r in unl if _client_class(r) == cls)
                row = [cls, f"{cls_n}"]
                for t in thresholds:
                    fired = sum(
                        1 for p in per_utt
                        if p["rec"] in unl
                        and _client_class(p["rec"]) == cls
                        and max(p["scored"][vname].values()) >= t
                    )
                    pct = 100.0 * fired / max(1, cls_n)
                    row.append(f"{fired}/{cls_n} ({pct:>4.0f}%)")
                rows.append(row)
            headers = ("source", "n") + tuple(f"thr={t:.2f}" for t in thresholds)
            print(_format_table(rows, headers))
            print()

    if args.details:
        print("=== per-utterance scoring ===")
        for p in per_utt:
            r = p["rec"]
            uid = r.get("utterance_id", "?")
            cls = p["client_class"]
            lbl = p["label"]
            lbl_s = ("+" if lbl and lbl.get("said_wake_word") else
                     "-" if lbl else "?")
            dur = r.get("duration_s")
            rmsp = r.get("rms_peak")
            text = (r.get("stt_text") or "")[:40]
            score_summary = "  ".join(
                f"{v}={max(p['scored'][v].values()):.3f}" for v in variants
            )
            print(f"  {uid} cls={cls:7s} lbl={lbl_s} dur={dur:>4.2f}s rms_p={rmsp:.2f}  "
                  f"{score_summary}  text={text!r}")

    if args.replay_misses:
        miss_wavs = sorted(MISSES_DIR.glob("*.wav"))
        if not miss_wavs:
            print("\n(no miss-ring wavs to replay)")
        else:
            print("\n=== miss-ring wavs (raw pre-VAD audio) ===")
            for vname, model in variants.items():
                print(f"-- variant={vname} --")
                rows = []
                for w in miss_wavs:
                    samples, sr = _read_wav(w)
                    if samples.size == 0:
                        continue
                    max_per = _score_wav(model, samples, sr)
                    rows.append([w.name, f"{max(max_per.values()):.3f}",
                                 "/".join(f"{n}={v:.2f}" for n, v in max_per.items())])
                print(_format_table(rows, ("file", "max", "per-model")))
                print()


if __name__ == "__main__":
    main()
