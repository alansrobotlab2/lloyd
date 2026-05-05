#!/usr/bin/env python3
"""Benchmark for Qwen3.6-35B-A3B-NVFP4 on vLLM (port 8096).

Modes:
  Single run:  bench-35b-nvfp4.py [LABEL] [--max-tokens N] [--runs N]
  MTP sweep:   bench-35b-nvfp4.py --mtp-sweep [--mtp-min N] [--mtp-max N]
                 Restarts agent-llm-35b-nvfp4 via supervisorctl for each depth.
                 Sweeps both max_tokens=200 (short-form) and max_tokens=1000
                 (long-form/stable). Saves JSON to benchmarks/.

Decode tok/s is computed from the server's authoritative completion_tokens
divided by the post-TTFT decode window. Counting SSE chunks gives wrong (low)
numbers under MTP because each chunk packs multiple tokens.

Usage examples:
  # Quick sanity check at current MTP setting:
  python3 bin/bench-35b-nvfp4.py baseline --max-tokens 200 --runs 2

  # Full MTP sweep (takes ~30 min, restarts server per depth):
  python3 bin/bench-35b-nvfp4.py --mtp-sweep

  # Sweep just depths 3-6:
  python3 bin/bench-35b-nvfp4.py --mtp-sweep --mtp-min 3 --mtp-max 6
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from urllib.error import URLError

BASE = "http://127.0.0.1:8096"
MODEL = "Qwen3.6-35B-A3B-nvfp4"
# Can be overridden via --base-url and --model
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_SCRIPT = os.path.join(PROJECT_DIR, "bin", "start-35b-nvfp4.sh")
SUPERVISORCTL = ["supervisorctl", "-c",
                 os.path.join(PROJECT_DIR, "supervisor", "supervisord.conf")]
BENCH_DIR = os.path.join(PROJECT_DIR, "benchmarks")

# Three prompt sizes to exercise different KV-cache / prefill paths.
# Each is run cold then warm (tests prefix-cache hit on second call).
PROMPTS = {
    "short_512":
        "Explain how MoE expert routing works in transformer models in exactly four sentences. " * 8,
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


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def fetch_spec_metrics():
    """Pull cumulative spec-decode counters from /metrics."""
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=5) as r:
            text = r.read().decode()
    except URLError:
        return None
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for key in (
            "vllm:spec_decode_num_drafts_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
        ):
            if line.startswith(key + "{") or line.startswith(key + " "):
                m = re.search(r"\}\s*([\d.eE+-]+)", line) or re.search(r"\s+([\d.eE+-]+)\s*$", line)
                if m:
                    out[key.split(":")[1]] = float(m.group(1))
    return out


def spec_delta(before, after):
    if not before or not after:
        return None
    d_drafts   = after.get("spec_decode_num_drafts_total", 0)        - before.get("spec_decode_num_drafts_total", 0)
    d_draft_tok= after.get("spec_decode_num_draft_tokens_total", 0)  - before.get("spec_decode_num_draft_tokens_total", 0)
    d_acc_tok  = after.get("spec_decode_num_accepted_tokens_total", 0)- before.get("spec_decode_num_accepted_tokens_total", 0)
    if d_draft_tok == 0:
        return None
    return {
        "drafts":          d_drafts,
        "draft_tokens":    d_draft_tok,
        "accepted_tokens": d_acc_tok,
        "accept_rate":     d_acc_tok / d_draft_tok,
        "mean_accept_len": (d_acc_tok + d_drafts) / d_drafts if d_drafts > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Single prompt runner
# ---------------------------------------------------------------------------

def run_prompt(label, prompt, max_tokens=200):
    """Stream one completion; return timing dict."""
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

    t_start = time.perf_counter()
    ttft = None
    chunk_times = []
    completion_tokens = None
    prompt_tokens = None
    cached_tokens = None
    chunks = 0
    error = None

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
                        prompt_tokens     = usage.get("prompt_tokens", prompt_tokens)
                        cached_tokens     = usage.get("prompt_tokens_details", {}).get("cached_tokens", cached_tokens)

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    has_text = (
                        (delta.get("reasoning") or "") +
                        (delta.get("content") or "")
                    )
                    if has_text:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t_start
                        chunk_times.append(now)
                        chunks += 1
    except Exception as e:
        error = str(e)

    elapsed = time.perf_counter() - t_start

    if error and not chunk_times:
        return {"label": label, "error": error}

    # Decode window = from first token to last token
    if ttft is None or completion_tokens is None:
        return {"label": label, "error": "no tokens received"}

    decode_elapsed = elapsed - ttft
    decode_tps = (completion_tokens - 1) / decode_elapsed if decode_elapsed > 1e-3 and completion_tokens > 1 else 0.0
    overall_tps = completion_tokens / elapsed if elapsed > 0 else 0.0

    itls = [chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]
    avg_itl_ms = 1000 * sum(itls) / len(itls) if itls else 0.0

    return {
        "label":             label,
        "prompt_tokens":     prompt_tokens or 0,
        "cached_tokens":     cached_tokens or 0,
        "completion_tokens": completion_tokens,
        "chunks":            chunks,
        "ttft_s":            round(ttft, 3),
        "decode_tps":        round(decode_tps, 1),
        "overall_tps":       round(overall_tps, 1),
        "avg_itl_ms":        round(avg_itl_ms, 1),
        "elapsed_s":         round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Benchmark run (all prompts, cold + warm)
# ---------------------------------------------------------------------------

def run_benchmark(max_tokens=200, runs=1, verbose=True):
    if verbose:
        print(f"  Benchmarking max_tokens={max_tokens} runs={runs}")

    spec_before = fetch_spec_metrics()
    results = []

    for rep in range(runs):
        rep_tag = f"r{rep}/" if runs > 1 else ""
        for pname, prompt in PROMPTS.items():
            for run_tag in ("cold", "warm"):
                label = f"{rep_tag}{pname}/{run_tag}"
                r = run_prompt(label, prompt, max_tokens=max_tokens)
                results.append(r)
                if verbose:
                    if "error" in r:
                        print(f"    {r['label']:<24} ERROR: {r['error']}")
                    else:
                        print(f"    {r['label']:<24} "
                              f"prompt={r['prompt_tokens']:>6}t(cached={r['cached_tokens']:>6}t) "
                              f"out={r['completion_tokens']:>4}t/{r['chunks']:>3}c  "
                              f"TTFT={r['ttft_s']:>6.3f}s  "
                              f"decode={r['decode_tps']:>7.1f}t/s  "
                              f"ITL={r['avg_itl_ms']:>5.1f}ms")

    spec_after = fetch_spec_metrics()
    sd = spec_delta(spec_before, spec_after)

    # Aggregate — exclude runs with <25 completion tokens from the decode average
    # (medium/long prompts may answer in 1-2 sentences; those runs measure TTFT
    # but the post-TTFT decode window is too short for a stable tok/s reading).
    ok = [r for r in results if "error" not in r]
    decode_tps_all = [r["decode_tps"] for r in ok if r.get("completion_tokens", 0) >= 25]
    itl_all        = [r["avg_itl_ms"] for r in ok if r.get("completion_tokens", 0) >= 25]
    ttft_all       = [r["ttft_s"]     for r in ok]

    def _first(suffix):
        return next((r["ttft_s"] for r in ok if r["label"].endswith(suffix)), None)

    avg_decode = sum(decode_tps_all) / len(decode_tps_all) if decode_tps_all else 0
    stddev = (sum((x - avg_decode)**2 for x in decode_tps_all) / max(1, len(decode_tps_all)-1))**0.5
    avg_itl  = sum(itl_all) / len(itl_all) if itl_all else 0
    avg_ttft = sum(ttft_all) / len(ttft_all) if ttft_all else 0

    summary = {
        "max_tokens":         max_tokens,
        "runs":               runs,
        "num_prompts_ok":     len(ok),
        "avg_decode_tps":     round(avg_decode, 1),
        "stddev_decode_tps":  round(stddev, 1),
        "avg_itl_ms":         round(avg_itl, 1),
        "avg_ttft_s":         round(avg_ttft, 3),
        "cold_ttft_short":    _first("short_512/cold"),
        "cold_ttft_med":      _first("medium_4k/cold"),
        "cold_ttft_long":     _first("long_16k/cold"),
        "spec":               sd,
        "per_prompt":         results,
    }

    if verbose and sd:
        print(f"\n  Spec-decode (this run): accept_rate={sd['accept_rate']:.3f}  "
              f"mean_accept_len={sd['mean_accept_len']:.2f}  "
              f"accepted={sd['accepted_tokens']:.0f}/{sd['draft_tokens']:.0f}")

    return summary


# ---------------------------------------------------------------------------
# MTP sweep helpers
# ---------------------------------------------------------------------------

def get_current_mtp():
    with open(START_SCRIPT) as f:
        m = re.search(r'"num_speculative_tokens":\s*(\d+)', f.read())
    return int(m.group(1)) if m else None


def set_mtp(depth):
    with open(START_SCRIPT) as f:
        text = f.read()
    new = re.sub(r'"num_speculative_tokens":\s*\d+',
                 f'"num_speculative_tokens": {depth}', text)
    with open(START_SCRIPT, "w") as f:
        f.write(new)


def restart_and_wait(timeout=300):
    subprocess.run(SUPERVISORCTL + ["restart", "agent-llm-35b-nvfp4"],
                   check=True, capture_output=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
                if r.status == 200:
                    time.sleep(2)   # brief settle
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def print_sweep_table(sweep_results):
    """Print a comparison table across MTP depths."""
    print()
    print("=" * 90)
    print("MTP SWEEP RESULTS")
    print("=" * 90)

    col_w = 12
    header_fields = ["MTP", "tps200", "±", "tps1000", "±", "ITL200ms", "ITL1kms",
                     "TTFT200s", "accept%", "accept_len"]
    print("  " + "  ".join(f"{h:>{col_w}}" for h in header_fields))
    print("  " + "-" * (col_w * len(header_fields) + 2 * len(header_fields)))

    base_tps200 = None
    base_tps1k  = None

    for depth, s200, s1k in sweep_results:
        tps200  = s200["avg_decode_tps"] if s200 else None
        std200  = s200["stddev_decode_tps"] if s200 else None
        tps1k   = s1k["avg_decode_tps"] if s1k else None
        std1k   = s1k["stddev_decode_tps"] if s1k else None
        itl200  = s200["avg_itl_ms"] if s200 else None
        itl1k   = s1k["avg_itl_ms"] if s1k else None
        ttft200 = s200["avg_ttft_s"] if s200 else None
        spec    = (s200 or s1k or {}).get("spec")
        accept  = f"{spec['accept_rate']*100:.1f}%" if spec else "N/A"
        alen    = f"{spec['mean_accept_len']:.2f}" if spec else "N/A"

        if base_tps200 is None and tps200:
            base_tps200 = tps200
        if base_tps1k is None and tps1k:
            base_tps1k = tps1k

        row = [
            str(depth),
            f"{tps200:.1f}" if tps200 else "—",
            f"±{std200:.1f}" if std200 is not None else "—",
            f"{tps1k:.1f}" if tps1k else "—",
            f"±{std1k:.1f}" if std1k is not None else "—",
            f"{itl200:.1f}" if itl200 else "—",
            f"{itl1k:.1f}" if itl1k else "—",
            f"{ttft200:.3f}" if ttft200 else "—",
            accept,
            alen,
        ]
        print("  " + "  ".join(f"{v:>{col_w}}" for v in row))

    # Speedup row vs MTP 1
    if base_tps200 and len(sweep_results) > 1:
        print()
        print("  Speedup vs MTP 1:")
        for depth, s200, s1k in sweep_results:
            t200 = s200["avg_decode_tps"] if s200 else None
            t1k  = s1k["avg_decode_tps"]  if s1k  else None
            sp200 = f"{t200/base_tps200:.2f}x" if t200 else "—"
            sp1k  = f"{t1k/base_tps1k:.2f}x"  if (t1k and base_tps1k) else "—"
            print(f"    MTP {depth}: {sp200:>8} (200tok)  {sp1k:>8} (1000tok)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global BASE, MODEL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", nargs="?", default="run")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=1,
                    help="Repetitions per prompt (more = lower stddev)")
    ap.add_argument("--mtp-sweep", action="store_true",
                    help="Sweep MTP depths, restart server for each")
    ap.add_argument("--mtp-min", type=int, default=1)
    ap.add_argument("--mtp-max", type=int, default=6)
    ap.add_argument("--base-url", default=BASE, help="Server base URL")
    ap.add_argument("--model", default=MODEL, help="Model name")
    args = ap.parse_args()
    BASE = args.base_url
    MODEL = args.model

    if args.mtp_sweep:
        original_mtp = get_current_mtp()
        print(f"MTP sweep depths {args.mtp_min}..{args.mtp_max}  "
              f"(current: {original_mtp})")
        print(f"Server: {BASE}  Model: {MODEL}")
        print("Each depth: 2×short-form (200tok) + 2×long-form (1000tok) cold+warm")
        print()

        sweep_results = []
        try:
            for depth in range(args.mtp_min, args.mtp_max + 1):
                print(f"{'='*70}")
                print(f"  MTP {depth}")
                print(f"{'='*70}")

                if get_current_mtp() != depth:
                    set_mtp(depth)
                    print(f"  Restarting server with num_speculative_tokens={depth}...")
                    if not restart_and_wait():
                        print(f"  ERROR: server did not come up for MTP {depth}, skipping")
                        sweep_results.append((depth, None, None))
                        continue
                    print(f"  Server healthy.")
                else:
                    print(f"  Server already running at MTP {depth}.")

                # Short-form: 2 runs for fast comparison
                print(f"\n  --- 200-token pass (2 runs) ---")
                s200 = run_benchmark(max_tokens=200, runs=2, verbose=True)

                # Long-form: 2 runs for stable measurement
                print(f"\n  --- 1000-token pass (2 runs) ---")
                s1k = run_benchmark(max_tokens=1000, runs=2, verbose=True)

                sweep_results.append((depth, s200, s1k))

                accept_str = f"{s200['spec']['accept_rate']:.3f}" if s200.get('spec') else 'N/A'
                print(f"\n  MTP {depth}: {s200['avg_decode_tps']:.1f} t/s (200tok)  "
                      f"{s1k['avg_decode_tps']:.1f} t/s (1000tok)  "
                      f"accept_rate={accept_str}")

        finally:
            # Restore best depth or original
            best = max((s200["avg_decode_tps"], depth)
                       for depth, s200, _ in sweep_results if s200) if sweep_results else (0, original_mtp)
            best_depth = best[1]
            print(f"\nBest MTP by 200-tok decode: {best_depth} ({best[0]:.1f} t/s)")

            if best_depth != get_current_mtp():
                print(f"Setting MTP to {best_depth} and restarting...")
                set_mtp(best_depth)
                restart_and_wait()
                print("Done.")

            # Save results
            os.makedirs(BENCH_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(BENCH_DIR, f"35b-nvfp4-mtp-sweep-{ts}.json")
            with open(out_path, "w") as f:
                json.dump([{"depth": d, "s200": s2, "s1k": s1}
                           for d, s2, s1 in sweep_results], f, indent=2)
            print(f"Results saved to {out_path}")

        print_sweep_table(sweep_results)

    else:
        # Single run
        mtp = get_current_mtp()
        print(f"vLLM benchmark @ {BASE}  model={MODEL}  MTP={mtp}  "
              f"max_tokens={args.max_tokens}  runs={args.runs}")
        print("=" * 78)
        summary = run_benchmark(max_tokens=args.max_tokens, runs=args.runs, verbose=True)
        sd = summary.get("spec")
        accept_str = f"{sd['accept_rate']:.3f}" if sd else "NA"
        alen_str   = f"{sd['mean_accept_len']:.2f}" if sd else "NA"
        print()
        print(f"SUMMARY {args.label} mtp={mtp} | "
              f"decode_tps={summary['avg_decode_tps']:.1f}±{summary['stddev_decode_tps']:.1f} "
              f"itl_ms={summary['avg_itl_ms']:.1f} "
              f"cold_ttft_short={summary['cold_ttft_short']}s "
              f"cold_ttft_med={summary['cold_ttft_med']}s "
              f"cold_ttft_long={summary['cold_ttft_long']}s "
              f"accept_rate={accept_str} mean_accept_len={alen_str}")

        print()
        print("Decode tok/s per prompt size:")
        print(f"  {'size':<12} {'cold ttft':>12} {'cold tps':>12} {'warm ttft':>12} {'warm tps':>12}")
        bucket = {}
        for r in summary["per_prompt"]:
            if "error" in r:
                continue
            parts = r["label"].split("/")
            size, run = parts[-2], parts[-1]
            bucket.setdefault((size, run), []).append(r)

        def _avg(vs, k):
            return sum(v[k] for v in vs) / len(vs)

        for size in PROMPTS:
            c = bucket.get((size, "cold"))
            w = bucket.get((size, "warm"))
            cttft = f"{_avg(c, 'ttft_s'):.3f}s"  if c else "—"
            ctps  = f"{_avg(c, 'decode_tps'):.1f}" if c else "—"
            wttft = f"{_avg(w, 'ttft_s'):.3f}s"  if w else "—"
            wtps  = f"{_avg(w, 'decode_tps'):.1f}" if w else "—"
            print(f"  {size:<12} {cttft:>12} {ctps:>12} {wttft:>12} {wtps:>12}")


if __name__ == "__main__":
    main()
