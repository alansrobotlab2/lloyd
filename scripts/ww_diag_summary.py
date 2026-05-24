#!/usr/bin/env python3
"""Bucket ~/.lloyd/ww_diag/scores.jsonl by client device + APM flags and
print rms vs ww-score stats per bucket.

Answers three questions for the cross-device wake-word debugging:
  1. Does rms_mean cluster differently per device? → gain disparity is real
  2. Does ww_score correlate with rms within a bucket? → loudness-bound
  3. Does APM-off (?raw_audio=1) move the needle on phone? → APM is the bad actor

Usage:
    python scripts/ww_diag_summary.py
    python scripts/ww_diag_summary.py --path ~/.lloyd/ww_diag/scores.jsonl
    python scripts/ww_diag_summary.py --since 2026-05-15
    python scripts/ww_diag_summary.py --min-voiced 0.1  # filter near-silence
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import statistics
import sys
from collections import defaultdict


DEFAULT_PATH = pathlib.Path("~/.lloyd/ww_diag/scores.jsonl").expanduser()


def classify_device(ua: str) -> str:
    if not ua:
        return "unknown"
    if re.search(r"iPhone|iPad|iPod", ua):
        return "ios"
    if "Android" in ua:
        return "android"
    if "Mobile" in ua or "Tablet" in ua:
        return "mobile-other"
    if "Macintosh" in ua:
        return "mac"
    if "Windows" in ua:
        return "windows"
    if "Linux" in ua:
        return "linux"
    return "other"


def bucket_key(rec: dict) -> tuple[str, str, str]:
    ci = rec.get("client_info") or {}
    device = classify_device(ci.get("userAgent", ""))
    raw_audio = ci.get("rawAudio")
    apm = "raw" if raw_audio else ("apm" if raw_audio is False else "?")
    # Mic-gain bucket: round to one decimal so 0.9 and 1.0 don't fragment.
    g = ci.get("micGain")
    gain = f"g={g:.1f}" if isinstance(g, (int, float)) else "g=?"
    return device, apm, gain


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation. Returns None for n<3 or zero variance."""
    n = len(xs)
    if n < 3:
        return None

    def rank(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def load(path: pathlib.Path, since: float | None, min_voiced: float) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("ww_ran"):
                continue
            if since is not None and rec.get("ts", 0) < since:
                continue
            if (rec.get("voiced_ratio") or 0) < min_voiced:
                continue
            out.append(rec)
    return out


def format_row(label: str, n: int, rms: list[float], scores: list[float],
               fired: int, near_misses: int, rho: float | None) -> str:
    rho_s = f"{rho:+.2f}" if rho is not None else "  n/a"
    return (
        f"{label:<32} "
        f"n={n:>4}  "
        f"rms p50={pct(rms, 0.5):.3f}  "
        f"p90={pct(rms, 0.9):.3f}  "
        f"score p50={pct(scores, 0.5):.3f}  "
        f"p90={pct(scores, 0.9):.3f}  "
        f"fire%={100*fired/n:5.1f}  "
        f"near%={100*near_misses/n:5.1f}  "
        f"ρ(rms,score)={rho_s}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", type=pathlib.Path, default=DEFAULT_PATH)
    ap.add_argument("--since", help="ISO date or epoch — drop earlier records")
    ap.add_argument("--min-voiced", type=float, default=0.05,
                    help="Drop utterances with voiced_ratio below this (default 0.05)")
    ap.add_argument("--near-miss-band", type=float, default=0.15,
                    help="Score within this distance below threshold counts as near-miss")
    args = ap.parse_args()

    since: float | None = None
    if args.since:
        try:
            since = float(args.since)
        except ValueError:
            since = dt.datetime.fromisoformat(args.since).timestamp()

    if not args.path.exists():
        print(f"no scores file at {args.path}", file=sys.stderr)
        return 1

    records = load(args.path, since, args.min_voiced)
    if not records:
        print("no qualifying utterances after filters", file=sys.stderr)
        return 1

    print(f"# {len(records)} utterances from {args.path}")
    print(f"# filter: voiced_ratio >= {args.min_voiced}"
          + (f", ts >= {since}" if since else ""))
    print()

    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        buckets[bucket_key(rec)].append(rec)

    header = (f"{'device / apm / gain':<32} {'n':>5}  "
              f"{'rms p50':>9}  {'p90':>5}  "
              f"{'score p50':>10}  {'p90':>5}  "
              f"{'fire%':>6}  {'near%':>6}  ρ(rms,score)")
    print(header)
    print("-" * len(header))

    rows = []
    for key, recs in buckets.items():
        rms = [r["rms_mean"] for r in recs]
        scores = [r["ww_score"] for r in recs]
        fired = sum(1 for r in recs if r["ww_fired"])
        # Use each utterance's own threshold — it may have changed over time.
        near = sum(
            1 for r in recs
            if not r["ww_fired"]
            and r["ww_score"] >= max(0.0, r["ww_threshold"] - args.near_miss_band)
        )
        rho = spearman(rms, scores)
        rows.append((key, len(recs), rms, scores, fired, near, rho))

    rows.sort(key=lambda row: -row[1])
    for key, n, rms, scores, fired, near, rho in rows:
        label = " / ".join(key)
        print(format_row(label, n, rms, scores, fired, near, rho))

    # Cross-bucket diagnosis hints.
    print()
    print("# diagnosis hints")
    by_device: dict[str, list[float]] = defaultdict(list)
    for key, recs in buckets.items():
        for r in recs:
            by_device[key[0]].append(r["rms_mean"])
    medians = {d: statistics.median(v) for d, v in by_device.items() if len(v) >= 5}
    if len(medians) >= 2:
        lo_d, lo_v = min(medians.items(), key=lambda kv: kv[1])
        hi_d, hi_v = max(medians.items(), key=lambda kv: kv[1])
        ratio = (hi_v / lo_v) if lo_v > 0 else float("inf")
        print(f"  rms ratio across devices: {hi_d}/{lo_d} = {ratio:.1f}x "
              f"(med {hi_v:.3f} vs {lo_v:.3f})")
        if ratio >= 3:
            print("  → real gain disparity. Per-utterance scalar normalize "
                  "before predict() is justified.")
        else:
            print("  → loudness looks comparable. If a device still misses, "
                  "the bad actor is APM/codec, not gain.")

    apm_pairs = defaultdict(lambda: {"apm": [], "raw": []})
    for key, recs in buckets.items():
        device, apm, _ = key
        if apm in ("apm", "raw"):
            apm_pairs[device][apm].extend(r["ww_score"] for r in recs)
    for device, pair in apm_pairs.items():
        if len(pair["apm"]) >= 5 and len(pair["raw"]) >= 5:
            d = statistics.median(pair["raw"]) - statistics.median(pair["apm"])
            print(f"  {device}: raw_audio shifts median ww_score by {d:+.3f} "
                  f"(raw n={len(pair['raw'])}, apm n={len(pair['apm'])})")
            if d >= 0.1:
                print(f"    → WebRTC APM is suppressing wake on {device}. "
                      "Default raw_audio=1 there.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
