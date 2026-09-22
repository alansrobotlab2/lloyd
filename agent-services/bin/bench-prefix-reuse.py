#!/usr/bin/env python
"""Measure decode throughput and cross-request prefix reuse on the primary slot.

Written 2026-09-06 to settle whether the qwen4_exp MTP boot warning
("prefix-cache reuse across requests will be disabled") actually bites. It does
not -- reuse measures 96.0% with MTP on. See the MTP_ENABLED block in
start-qwen38-flash-next.sh for the full result table.

Two traps this script exists to avoid:

  1. vllm:prefix_cache_hits_total / prefix_cache_queries_total is a token-level
     reuse rate ONLY while prefix_cache_queries_total == prompt_tokens_total
     (1:1 parity), so check that parity before reading the number. It is one
     curl: scrape /metrics and compare those two counters.

     Current builds satisfy it. Re-read 2026-09-22T05:47Z on a primary booted
     that same day (model Qwen3.8-Flash-Next-nvfp4): prefix_cache_queries_total
     148,645,180 against prompt_tokens_total 148,644,807 -- 373 tokens apart,
     1.0000025x -- and the djev engine equal outright, prefix_cache_queries_total
     6,636,812 == prompt_tokens_total 6,636,812. Earlier reads of the same
     series: equal outright on 2026-09-13 (both 2,050,518,768) and 423 apart at a
     3.03-billion-token lifetime (2026-09-21T21:47Z). The gap on every read is a
     few hundred tokens, never a multiple, so here the counter ratio IS a genuine
     cross-request token-level reuse rate and the Mission Control card that
     serves it as prefix_cache_hit_rate is telling the truth. This script's own
     probe carries the equality down to a single request: pass 2 of the run
     quoted in #605 logged d_queries 60,005 for that pass's own 60,005-token
     prompt, and d_hits 54,400 equalled that request's
     usage.prompt_tokens_details.cached_tokens 54,400. The 2026-09-13 probe read
     delta-queries over delta-prompt_tokens at 1.000 on all 8 of its windows,
     with delta-hits equal to that window's cached_tokens on each.

     The 11.2x belonged to the boot this script was written for: the 2026-09-06
     qwen4_exp MTP run (9a0a1d8, "proof the eagle-group warning is benign"),
     whose speculative-decode draft head adds a second (eagle) KV cache group.
     On that MTP boot the counter summed queries across every group, so one
     60k-token prompt logged 673,179 queries -- an 11.2x amplification -- and the
     script's note from that 2026-09-06 boot has the ratio reading ~68% while
     real cross-request reuse was zero. Whether the eagle draft group is the
     multiplier was never isolated; what is measured is that this build does not
     multiply.

     Two caveats survive either way. A counter window on a busy engine counts
     every request inside it, so d_hits / d_queries there is the fleet's reuse
     rate and not yours -- the loaded 2026-09-21 run read 90.4%, 91.1% and 92.2%
     on three passes whose own cached_tokens never moved off 54,400. And load
     evicts: pass 5 of that run read cached_tokens 0 after 1,349,507 tokens of
     other traffic crossed its window. Use the per-request
     usage.prompt_tokens_details.cached_tokens as the figure immune to everyone
     else's traffic (needs --enable-prompt-tokens-details, which the start script
     passes).

  2. Reuse needs TWO warm-up passes before it engages: passes 1-2 cache nothing,
     pass 3+ hits ~96% and runs ~18x faster. A 2-pass A/B reports 0% on both arms
     and proves nothing. Hence PASSES = 5 below; do not lower it.

Usage:
    .venvs/lloyd/bin/python agent-services/bin/bench-prefix-reuse.py <tag> [--json out.json]

Run it against an idle, freshly booted engine -- a long-running engine under load
measured 60.3 tok/s where a fresh one measured 181.9.
"""

import argparse
import json
import random
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:8096"
MODEL = "primary"
PASSES = 5          # >= 3, see trap 2 above
PROBE_WORDS = 12000  # ~60k tokens


def _metrics() -> str:
    return urllib.request.urlopen(f"{BASE}/metrics", timeout=10).read().decode()


def _gauge(name: str) -> float | None:
    for line in _metrics().splitlines():
        if line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return None


def _counters() -> dict[str, float]:
    out = {}
    for line in _metrics().splitlines():
        for key in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"):
            if line.startswith(key + "{"):
                out[key.split(":", 1)[1]] = float(line.rsplit(" ", 1)[1])
    return out


def _wait_idle(limit: int = 180) -> None:
    """Never benchmark against the agent's own traffic."""
    for _ in range(limit):
        try:
            if _gauge("vllm:num_requests_running") == 0.0:
                return
        except Exception:
            pass
        time.sleep(1)


def _post(payload: dict, timeout: int = 900) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()), time.perf_counter() - start


def measure_decode(runs: int = 3) -> float:
    rates = []
    for i in range(runs):
        _wait_idle(30)
        out, dt = _post({
            "model": MODEL,
            "prompt": "Count slowly from one to two hundred, one number per line.\n1\n",
            "max_tokens": 256, "temperature": 0.0, "ignore_eos": True,
        })
        n = out["usage"]["completion_tokens"]
        rates.append(n / dt)
        print(f"  decode run{i + 1}: {n:>4} tok in {dt:5.2f}s -> {n / dt:6.1f} tok/s")
    median = statistics.median(rates)
    print(f"  decode MEDIAN: {median:.1f} tok/s\n")
    return median


def measure_reuse() -> list[dict]:
    # Fixed seed: the same corpus on both arms of an A/B, and across runs.
    rnd = random.Random(99)
    vocab = [f"tok{i:04d}" for i in range(4000)]
    prompt = "[reuse-probe] " + " ".join(rnd.choice(vocab) for _ in range(PROBE_WORDS))

    print("  pass  wall     cached_tokens   d_queries   d_hits   hit%")
    prev = _counters()
    passes = []
    for i in range(PASSES):
        _wait_idle()
        out, dt = _post({"model": MODEL, "prompt": prompt,
                         "max_tokens": 1, "temperature": 0.0})
        _wait_idle()
        cur = _counters()
        d_q = cur["prefix_cache_queries_total"] - prev["prefix_cache_queries_total"]
        d_h = cur["prefix_cache_hits_total"] - prev["prefix_cache_hits_total"]
        prev = cur
        usage = out["usage"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        passes.append({"pass": i + 1, "wall_s": dt, "cached_tokens": cached,
                       "prompt_tokens": usage["prompt_tokens"],
                       "d_queries": d_q, "d_hits": d_h})
        print(f"  {i + 1:>4}  {dt:6.2f}s  {cached:>13,}   {d_q:>9,.0f}  {d_h:>7,.0f}  "
              f"{(100.0 * d_h / d_q if d_q else 0):5.1f}%")
    return passes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="label for this run, e.g. mtp-on")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    print(f"\n{'=' * 66}\n  PREFIX REUSE BENCH [{args.tag}]\n{'=' * 66}")
    _wait_idle()
    decode = measure_decode()
    passes = measure_reuse()

    warm = [p for p in passes if p["cached_tokens"] > 0]
    best = max((p["cached_tokens"] for p in passes), default=0)
    total = passes[0]["prompt_tokens"] if passes else 0
    reuse_pct = (100.0 * best / total) if total else 0.0
    print(f"\n  RESULT [{args.tag}]  decode={decode:.1f} tok/s  "
          f"reuse={reuse_pct:.1f}% ({best:,}/{total:,})  "
          f"cold={passes[0]['wall_s']:.2f}s  "
          f"warm={min((p['wall_s'] for p in warm), default=float('nan')):.2f}s")
    if not warm:
        print("  WARNING: no pass reused anything -- prefix caching may be broken,\n"
              "           or PASSES is too low (see trap 2 in the module docstring).")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"tag": args.tag, "decode_tok_s": decode,
                       "reuse_pct": reuse_pct, "passes": passes}, fh, indent=2)


if __name__ == "__main__":
    main()
