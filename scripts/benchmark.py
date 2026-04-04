#!/usr/bin/env python3
"""
vLLM serving benchmark: throughput, TTFT, and ITL.

Sends N concurrent requests at max throughput and measures:
  - TTFT    (time to first token, via streaming)
  - ITL     (inter-token latency, mean across all tokens)
  - Tok/s   (output tokens per second, wall-clock)
  - Req/s   (requests per second)

Usage:
  python scripts/benchmark.py                          # all scenarios
  python scripts/benchmark.py --input 512 --output 256 --n 50
  python scripts/benchmark.py --tag fp8_e4m3           # label results
"""
import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "artificial intelligence machine learning neural network transformer "
    "attention mechanism gradient descent optimization loss function "
    "tensor matrix multiplication convolution pooling activation "
    "softmax normalization embedding tokenization vocabulary "
).split()


def make_prompt(num_tokens: int) -> str:
    """Generate a ~num_tokens prompt from a word pool."""
    # ~1.3 tokens per word for English
    words_needed = int(num_tokens / 1.3)
    rng = random.Random(num_tokens)
    return " ".join(rng.choices(WORDS, k=words_needed))


# ---------------------------------------------------------------------------
# Single streaming request
# ---------------------------------------------------------------------------

def bench_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int = 120,
) -> dict:
    """
    Send one streaming chat completion request and record timing.

    Returns dict with: ttft_s, total_s, output_tokens, error
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,       # sampling for realistic throughput
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t_start = time.perf_counter()
    ttft = None
    output_tokens = 0
    last_token_t = t_start
    token_gaps = []

    try:
        with requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t_start
                    else:
                        token_gaps.append(now - last_token_t)
                    last_token_t = now
                    output_tokens += 1  # approximate; 1 per chunk
    except Exception as e:
        return {"error": str(e), "ttft_s": None, "total_s": None, "output_tokens": 0, "token_gaps": []}

    total_s = time.perf_counter() - t_start
    return {
        "error": None,
        "ttft_s": ttft,
        "total_s": total_s,
        "output_tokens": output_tokens,
        "token_gaps": token_gaps,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def get_model(url: str) -> str:
    try:
        return requests.get(f"{url}/v1/models", timeout=5).json()["data"][0]["id"]
    except Exception:
        return "unknown"


def run_scenario(
    url: str,
    model: str,
    input_len: int,
    output_len: int,
    n: int,
    concurrency: int,
    label: str,
) -> dict:
    prompt = make_prompt(input_len)
    results = []

    print(f"\n  {label}: {n} reqs, {input_len}→{output_len} tok, concurrency={concurrency}")
    t_wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(bench_request, url, model, prompt, output_len)
            for _ in range(n)
        ]
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{n} done...", flush=True)

    t_wall = time.perf_counter() - t_wall_start

    good = [r for r in results if r["error"] is None and r["ttft_s"] is not None]
    errors = len(results) - len(good)

    if not good:
        print(f"  ERROR: all {n} requests failed")
        return {}

    ttfts = [r["ttft_s"] * 1000 for r in good]          # ms
    all_itls = []
    for r in good:
        all_itls.extend([g * 1000 for g in r["token_gaps"]])  # ms

    total_out_tokens = sum(r["output_tokens"] for r in good)
    tok_per_s = total_out_tokens / t_wall
    req_per_s = len(good) / t_wall

    def pct(lst, p):
        lst_s = sorted(lst)
        idx = int(len(lst_s) * p / 100)
        return round(lst_s[min(idx, len(lst_s)-1)], 2)

    summary = {
        "label": label,
        "input_len": input_len,
        "output_len": output_len,
        "n_requests": n,
        "n_success": len(good),
        "n_errors": errors,
        "wall_s": round(t_wall, 2),
        "req_per_s": round(req_per_s, 2),
        "tok_per_s": round(tok_per_s, 1),
        "ttft_ms": {
            "mean": round(statistics.mean(ttfts), 2),
            "p50":  pct(ttfts, 50),
            "p95":  pct(ttfts, 95),
            "p99":  pct(ttfts, 99),
        },
        "itl_ms": {
            "mean": round(statistics.mean(all_itls), 2) if all_itls else 0,
            "p50":  pct(all_itls, 50) if all_itls else 0,
            "p95":  pct(all_itls, 95) if all_itls else 0,
            "p99":  pct(all_itls, 99) if all_itls else 0,
        },
    }

    print(f"    req/s={summary['req_per_s']}  tok/s={summary['tok_per_s']}")
    print(f"    TTFT  mean={summary['ttft_ms']['mean']}ms  p50={summary['ttft_ms']['p50']}ms  p95={summary['ttft_ms']['p95']}ms")
    print(f"    ITL   mean={summary['itl_ms']['mean']}ms  p50={summary['itl_ms']['p50']}ms  p95={summary['itl_ms']['p95']}ms")
    if errors:
        print(f"    ERRORS: {errors}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCENARIOS = [
    # (label,             input_len, output_len, n,   concurrency)
    ("short  512→256",    512,       256,        80,  4),
    ("medium 4k→512",    4096,       512,        40,  4),
    ("long   16k→256",  16384,       256,        20,  4),
    ("long   32k→256",  32768,       256,        10,  4),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8096")
    parser.add_argument("--tag", default="turboquant_4bit")
    parser.add_argument("--input", type=int, default=None)
    parser.add_argument("--output", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    model = get_model(args.url)
    print(f"\n{'='*60}")
    print(f"Benchmark  — {args.tag}")
    print(f"  URL:     {args.url}")
    print(f"  Model:   {model}")
    print(f"  Start:   {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    if args.input is not None:
        scenarios = [(f"custom {args.input}→{args.output or 256}",
                      args.input, args.output or 256,
                      args.n or 40, args.concurrency)]
    else:
        scenarios = [(l, i, o, n, args.concurrency or c)
                     for l, i, o, n, c in SCENARIOS]

    all_results = []
    for label, input_len, output_len, n, concurrency in scenarios:
        r = run_scenario(args.url, model, input_len, output_len, n, concurrency, label)
        if r:
            r["tag"] = args.tag
            all_results.append(r)

    # Summary table
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.tag}")
    print(f"{'Scenario':<22} {'req/s':>7} {'tok/s':>8} {'TTFT p50':>10} {'TTFT p95':>10} {'ITL p50':>9} {'ITL p95':>9}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['label']:<22} "
            f"{r['req_per_s']:>7.2f} "
            f"{r['tok_per_s']:>8.1f} "
            f"{r['ttft_ms']['p50']:>9.1f}ms "
            f"{r['ttft_ms']['p95']:>9.1f}ms "
            f"{r['itl_ms']['p50']:>8.1f}ms "
            f"{r['itl_ms']['p95']:>8.1f}ms"
        )
    print(f"{'='*60}\n")

    # Save
    out_file = args.output_file or f"/home/alansrobotlab/lloyd/logs/benchmarks/bench_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
