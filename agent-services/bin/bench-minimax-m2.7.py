#!/usr/bin/env python3
"""Quick perf benchmark for MiniMax-M2.7 on llama.cpp. Measures TTFT, ITL,
decode tok/s across short / medium / long prompts. Each prompt run twice
(cold + warm prefix cache hit) so we see both sides.

Decode tok/s is computed from the server's authoritative completion_tokens
counter divided by the post-TTFT decode window.

Usage:
  bench-minimax-m2.7.py [LABEL] [--max-tokens N] [--runs N]

Options:
  LABEL          Tag for the SUMMARY line (default: "run")
  --max-tokens   Tokens to generate per prompt (default: 200). Use 1000+
                 for stable measurements that amortize per-step overhead.
  --runs         Repetitions per prompt to average across (default: 1).
                 Each repetition is itself a cold+warm pair.
"""
import argparse
import json
import sys
import time
import threading
import urllib.request
from urllib.error import URLError

BASE = "http://127.0.0.1:8098"
MODEL = "MiniMax-M2.7"
BACKEND = "llama.cpp"

PROMPTS = {
    "short_512":
        "Explain how Mixture-of-Experts routing works in large language models in exactly four sentences. " * 8,
    "medium_4k":
        "Below is a list of facts. After reading them, answer in one sentence: "
        "what is the second word of the third fact?\n\n" +
        "\n".join(f"Fact {i}: The capital of country number {i} is city {i*7 % 137}, "
                  f"founded in year {1000 + i*13}, with population {i*9173 % 999999}."
                  for i in range(1, 220)),
    "long_16k":
        "Read the following document carefully. At the end, answer in one sentence: "
        "what is the value of variable X_137?\n\n" +
        "\n".join(f"X_{i} = {(i * 31337) % 99991}; "
                  f"Y_{i} = sqrt({(i * 7919) % 99991}); "
                  f"Z_{i} = log({(i * 2017) % 99991})."
                  for i in range(1, 950)),
}


def run_prompt(label, prompt, max_tokens=200):
    """Stream a single completion, measure TTFT/ITL/tok/s."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    ttft = None
    chunk_times = []
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                usage = data.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens") or completion_tokens
                    prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                    pt_details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = pt_details.get("cached_tokens", 0) or 0
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                has_text = bool(
                    (delta.get("content") or "")
                    or (delta.get("reasoning_content") or "")
                    or (delta.get("reasoning") or "")
                )
                if has_text:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    chunk_times.append(now)
    except Exception as e:
        return {"label": label, "error": str(e)}

    elapsed = time.perf_counter() - start

    if not chunk_times or completion_tokens == 0:
        return {"label": label, "error": "no tokens received"}

    decode_window = elapsed - ttft if ttft else elapsed
    decode_tps = completion_tokens / decode_window if decode_window > 0 else 0
    overall_tps = completion_tokens / elapsed if elapsed > 0 else 0
    avg_itl_ms = (decode_window * 1000) / max(1, completion_tokens - 1)

    return {
        "label": label,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "chunks": len(chunk_times),
        "ttft_s": round(ttft, 3),
        "decode_tps": round(decode_tps, 1),
        "overall_tps": round(overall_tps, 1),
        "avg_itl_ms": round(avg_itl_ms, 1),
        "elapsed_s": round(elapsed, 2),
    }


def run_concurrent(n_concurrent, prompt, max_tokens, label_prefix="concurrent"):
    """Fire n_concurrent requests simultaneously, measure per-request and
    aggregate throughput."""
    results = [None] * n_concurrent

    def _worker(idx):
        results[idx] = run_prompt(f"{label_prefix}/{idx}", prompt, max_tokens)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_concurrent)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_elapsed = time.perf_counter() - wall_start

    ok = [r for r in results if r and "error" not in r]
    total_tokens = sum(r["completion_tokens"] for r in ok)
    aggregate_tps = total_tokens / wall_elapsed if wall_elapsed > 0 else 0
    per_request_tps = [r["decode_tps"] for r in ok]
    avg_per_req = sum(per_request_tps) / len(per_request_tps) if per_request_tps else 0

    return {
        "n_concurrent": n_concurrent,
        "completed": len(ok),
        "errors": n_concurrent - len(ok),
        "wall_s": round(wall_elapsed, 2),
        "total_tokens": total_tokens,
        "aggregate_tps": round(aggregate_tps, 1),
        "avg_per_request_tps": round(avg_per_req, 1),
        "per_request": ok,
    }


def main():
    global BASE, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default="run")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--base", default=BASE, help="API base URL")
    ap.add_argument("--model", default=MODEL, help="served-model-name")
    ap.add_argument("--backend", default=BACKEND, help="label for output header")
    args = ap.parse_args()

    BASE = args.base
    MODEL = args.model

    print(f"{args.backend} benchmark @ {BASE} model={MODEL} max_tokens={args.max_tokens} runs={args.runs}")
    print("=" * 78)

    results = []
    for rep in range(args.runs):
        rep_tag = f"r{rep}/" if args.runs > 1 else ""
        for label, prompt in PROMPTS.items():
            for run_label in ("cold", "warm"):
                r = run_prompt(f"{rep_tag}{label}/{run_label}", prompt, max_tokens=args.max_tokens)
                results.append(r)
                if "error" in r:
                    print(f"  {r['label']:<22} ERROR: {r['error']}")
                    continue
                print(f"  {r['label']:<22} "
                      f"prompt={r['prompt_tokens']:>6}t (cached={r['cached_tokens']:>6}t) "
                      f"out={r['completion_tokens']:>4}t/{r['chunks']:>3}c  "
                      f"TTFT={r['ttft_s']:>6.3f}s  "
                      f"decode={r['decode_tps']:>6.1f}t/s  "
                      f"ITL={r['avg_itl_ms']:>5.1f}ms")

    # Single-line summary
    label = args.label

    def _first_ttft(suffix: str):
        return next((r["ttft_s"] for r in results if r.get("label", "").endswith(suffix) and "ttft_s" in r), None)

    cold_short_ttft = _first_ttft("short_512/cold") or 0.0
    cold_med_ttft = _first_ttft("medium_4k/cold") or 0.0
    cold_long_ttft = _first_ttft("long_16k/cold") or 0.0
    decode_samples = [r["decode_tps"] for r in results if "decode_tps" in r]
    itl_samples = [r["avg_itl_ms"] for r in results if "avg_itl_ms" in r]
    avg_decode_tps = sum(decode_samples) / max(1, len(decode_samples))
    avg_itl = sum(itl_samples) / max(1, len(itl_samples))
    if len(decode_samples) >= 2:
        mean = avg_decode_tps
        var = sum((x - mean) ** 2 for x in decode_samples) / (len(decode_samples) - 1)
        decode_stddev = var ** 0.5
    else:
        decode_stddev = 0.0

    print()
    print(f"SUMMARY {label} | decode_tps={avg_decode_tps:.1f}±{decode_stddev:.1f} "
          f"itl_ms={avg_itl:.1f} "
          f"cold_ttft_short={cold_short_ttft:.3f}s cold_ttft_med={cold_med_ttft:.3f}s "
          f"cold_ttft_long={cold_long_ttft:.3f}s")

    # Summary table
    print()
    print("Decode tok/s (averaged across runs):")
    bucket: dict = {}
    for r in results:
        if "error" in r:
            continue
        parts = r["label"].split("/")
        size, run = parts[-2], parts[-1]
        bucket.setdefault((size, run), []).append(r)

    def _avg(vs, k):
        return sum(v[k] for v in vs) / len(vs)

    print(f"  {'size':<12} {'cold ttft':>12} {'cold tps':>12} {'warm ttft':>12} {'warm tps':>12}")
    for size in PROMPTS:
        c = bucket.get((size, "cold"))
        w = bucket.get((size, "warm"))
        cttft = f"{_avg(c, 'ttft_s'):.3f}s" if c else "—"
        ctps = f"{_avg(c, 'decode_tps'):.1f}" if c else "—"
        wttft = f"{_avg(w, 'ttft_s'):.3f}s" if w else "—"
        wtps = f"{_avg(w, 'decode_tps'):.1f}" if w else "—"
        print(f"  {size:<12} {cttft:>12} {ctps:>12} {wttft:>12} {wtps:>12}")


    # --- Concurrent request benchmark ---
    concurrent_prompt = (
        "Write a detailed explanation of how transformer attention mechanisms work, "
        "including multi-head attention, scaled dot-product attention, and their "
        "computational complexity. Be thorough."
    )
    print()
    print("=" * 78)
    print(f"Concurrent request benchmark (max_tokens={args.max_tokens})")
    print("=" * 78)
    for n in (1, 2, 4):
        cr = run_concurrent(n, concurrent_prompt, args.max_tokens, f"conc_{n}")
        print(f"  {n} concurrent:  "
              f"wall={cr['wall_s']:>6.2f}s  "
              f"total_tok={cr['total_tokens']:>5}  "
              f"aggregate={cr['aggregate_tps']:>6.1f}t/s  "
              f"per_req={cr['avg_per_request_tps']:>6.1f}t/s  "
              f"errors={cr['errors']}")


if __name__ == "__main__":
    main()
