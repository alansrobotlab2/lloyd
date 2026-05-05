#!/usr/bin/env python3
"""Concurrency benchmark for Qwen3.6-35B-A3B on vLLM (port 8096).

Launches N simultaneous streaming requests and measures:
  - Per-user decode tok/s  (quality-of-service metric)
  - System decode tok/s    (total tokens/s across all users — batching efficiency)
  - TTFT distribution       (how much does queueing hurt latency?)

Usage:
  python3 bin/bench-35b-concurrency.py [--users "1,2,3,4,6"] [--max-tokens 300] [--runs 2]
"""

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from urllib.error import URLError

BASE = "http://127.0.0.1:8096"
MODEL = "Qwen3.6-35B-A3B-nvfp4"
# Overridden by --base-url and --model flags below

# Different prompts per user slot so prefix cache doesn't collapse identical streams.
# All are sized ~130 tokens and should elicit ~100-200 token responses.
PROMPTS = [
    "Explain how transformer attention works in exactly four sentences. " * 8,
    "Describe the tradeoffs between MoE and dense language models in four sentences. " * 8,
    "What's the difference between greedy and beam-search decoding? Four sentences. " * 8,
    "Explain prompt caching in vLLM in exactly four sentences. " * 8,
    "Describe how KV-cache memory scales with context length. Four sentences. " * 8,
    "Explain speculative decoding and MTP in four sentences. " * 8,
    "How does prefill differ from decode in an LLM server? Four sentences. " * 8,
    "Describe how FP4 quantization differs from INT4. Four sentences. " * 8,
]


def run_one(user_id, prompt, max_tokens, start_barrier):
    """One streaming request. Waits on start_barrier so all users fire together."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    # Gate: wait for all threads to be ready
    start_barrier.wait()

    t_start = time.perf_counter()
    ttft = None
    last_token_time = t_start
    completion_tokens = None
    chunks = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line_b, buf = buf.split(b"\n", 1)
                    line = line_b.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    usage = data.get("usage")
                    if usage:
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    text = (delta.get("reasoning") or "") + (delta.get("content") or "")
                    if text:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t_start
                        last_token_time = now
                        chunks += 1
    except Exception as e:
        return {"user": user_id, "error": str(e)}

    total_elapsed = last_token_time - t_start
    if ttft is None or completion_tokens is None or completion_tokens < 2:
        return {"user": user_id, "error": "no usable tokens received"}

    decode_elapsed = total_elapsed - ttft
    decode_tps = (completion_tokens - 1) / decode_elapsed if decode_elapsed > 1e-3 else 0.0

    return {
        "user":              user_id,
        "completion_tokens": completion_tokens,
        "chunks":            chunks,
        "ttft_s":            round(ttft, 3),
        "decode_elapsed":    round(decode_elapsed, 3),
        "total_elapsed":     round(total_elapsed, 3),
        "decode_tps":        round(decode_tps, 1),
        "t_start":           t_start,
        "t_end":             last_token_time,
    }


def run_concurrency(n_users, max_tokens, label):
    """Launch n_users concurrent requests; return aggregated stats."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_users)]
    barrier = __import__("threading").Barrier(n_users)

    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n_users) as pool:
        futures = [pool.submit(run_one, i, prompts[i], max_tokens, barrier)
                   for i in range(n_users)]
        results = [f.result() for f in cf.as_completed(futures)]
    wall_elapsed = time.perf_counter() - wall_start

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not ok:
        return {"label": label, "n_users": n_users, "error": "all requests failed"}

    total_out_tokens = sum(r["completion_tokens"] for r in ok)
    # System tps = total output tokens produced / the window during which any request was decoding.
    # Compute from the earliest first-token to the latest last-token.
    first_tok = min(r["t_start"] + r["ttft_s"] for r in ok)
    last_tok  = max(r["t_end"] for r in ok)
    system_decode_window = last_tok - first_tok
    system_tps = total_out_tokens / system_decode_window if system_decode_window > 0 else 0

    per_user_tps = [r["decode_tps"] for r in ok]
    ttfts       = [r["ttft_s"] for r in ok]
    total_elapsed = [r["total_elapsed"] for r in ok]

    def _stats(vs):
        if not vs:
            return (0, 0, 0, 0)
        m = sum(vs) / len(vs)
        s = (sum((x - m) ** 2 for x in vs) / max(1, len(vs)-1)) ** 0.5
        return (round(min(vs), 2), round(m, 2), round(max(vs), 2), round(s, 2))

    return {
        "label":                   label,
        "n_users":                 n_users,
        "n_ok":                    len(ok),
        "n_err":                   len(errors),
        "wall_elapsed":            round(wall_elapsed, 2),
        "total_output_tokens":     total_out_tokens,
        "system_decode_tps":       round(system_tps, 1),
        "per_user_tps_min":        _stats(per_user_tps)[0],
        "per_user_tps_mean":       _stats(per_user_tps)[1],
        "per_user_tps_max":        _stats(per_user_tps)[2],
        "per_user_tps_stddev":     _stats(per_user_tps)[3],
        "ttft_min":                _stats(ttfts)[0],
        "ttft_mean":               _stats(ttfts)[1],
        "ttft_max":                _stats(ttfts)[2],
        "total_elapsed_min":       _stats(total_elapsed)[0],
        "total_elapsed_max":       _stats(total_elapsed)[2],
        "per_user_detail":         ok,
    }


def main():
    global BASE, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="1,2,3,4,6",
                    help="Comma-separated user counts to test")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--runs", type=int, default=2,
                    help="Repetitions per user count")
    ap.add_argument("--warmup", action="store_true", default=True,
                    help="Do a 1-user warmup before measuring")
    ap.add_argument("--base-url", default=BASE, help="Server base URL")
    ap.add_argument("--model", default=MODEL, help="Model name")
    args = ap.parse_args()
    BASE = args.base_url
    MODEL = args.model

    user_counts = [int(u) for u in args.users.split(",")]

    # Warmup
    if args.warmup:
        print("Warming up with 1 user...")
        run_concurrency(1, args.max_tokens, "warmup")
        print()

    print(f"Concurrency sweep @ {BASE}  model={MODEL}  max_tokens={args.max_tokens}  runs={args.runs}")
    print(f"User counts: {user_counts}")
    print("=" * 100)

    all_results = []
    for n in user_counts:
        print(f"\n--- {n} user{'s' if n != 1 else ''} ---")
        for rep in range(args.runs):
            r = run_concurrency(n, args.max_tokens, f"n{n}_r{rep}")
            if "error" in r:
                print(f"  run {rep}: ERROR {r['error']}")
                continue
            all_results.append(r)
            print(f"  run {rep}: system={r['system_decode_tps']} tps | "
                  f"per-user mean={r['per_user_tps_mean']}±{r['per_user_tps_stddev']} "
                  f"[min={r['per_user_tps_min']}, max={r['per_user_tps_max']}] | "
                  f"TTFT mean={r['ttft_mean']}s (max={r['ttft_max']}s) | "
                  f"wall={r['wall_elapsed']}s")

    # Aggregate per user count (average across runs)
    print()
    print("=" * 100)
    print("AGGREGATED (averaged across runs)")
    print("=" * 100)
    print(f"  {'users':>5}  {'sys tps':>10}  {'per-user tps':>18}  "
          f"{'ttft mean':>10}  {'ttft max':>10}  {'scaling':>10}")
    print("  " + "-" * 88)

    base_per_user = None
    base_sys = None
    for n in user_counts:
        runs = [r for r in all_results if r["n_users"] == n]
        if not runs:
            continue
        sys_tps       = sum(r["system_decode_tps"] for r in runs) / len(runs)
        per_user_mean = sum(r["per_user_tps_mean"] for r in runs) / len(runs)
        per_user_std  = sum(r["per_user_tps_stddev"] for r in runs) / len(runs)
        ttft_mean     = sum(r["ttft_mean"] for r in runs) / len(runs)
        ttft_max      = max(r["ttft_max"] for r in runs)

        if base_per_user is None:
            base_per_user = per_user_mean
            base_sys = sys_tps
        per_user_ratio = per_user_mean / base_per_user if base_per_user else 0
        sys_scaling = sys_tps / base_sys if base_sys else 0

        print(f"  {n:>5}  {sys_tps:>10.1f}  {per_user_mean:>8.1f} ±{per_user_std:<6.1f}  "
              f"{ttft_mean:>10.2f}  {ttft_max:>10.2f}  "
              f"sys={sys_scaling:>4.2f}x  user={per_user_ratio:.2f}x")


if __name__ == "__main__":
    main()
