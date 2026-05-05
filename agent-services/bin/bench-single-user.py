#!/usr/bin/env python3
"""Sustained single-user decode throughput measurement.

Generates large responses (default 2000 tokens) across diverse prompts to isolate
decode throughput. Reports TTFT, decode tok/s per-run, and stats across runs.
"""
import argparse, json, sys, time, urllib.request

PROMPTS = [
    "Write a detailed 2000-word essay about the history of computing, from the abacus to modern neural networks. Be thorough and include specific milestones.",
    "Explain in comprehensive detail how the attention mechanism in transformers works, including query-key-value matrices, scaled dot-product attention, multi-head attention, and positional encodings. Use formulas and examples.",
    "Describe the evolution of programming languages from assembly to modern type-inferred functional languages. Discuss paradigms, memory safety, and compilation strategies in depth.",
    "Write a long technical walkthrough of how a modern GPU executes matrix multiplication, covering SIMT execution, memory hierarchies, tensor cores, and kernel fusion.",
]

def run(base, model, prompt, max_tokens, timeout):
    payload = json.dumps({
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload, headers={"Content-Type":"application/json"})
    t0 = time.perf_counter()
    ttft = None
    last = t0
    completion = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        while True:
            chunk = r.read(4096)
            if not chunk: break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode("utf-8","replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                try: d = json.loads(line[6:])
                except: continue
                if d.get("usage"):
                    completion = d["usage"].get("completion_tokens", completion)
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content") or delta.get("reasoning"):
                        now = time.perf_counter()
                        if ttft is None: ttft = now - t0
                        last = now
                        chunks += 1
    total = last - t0
    decode = total - (ttft or 0)
    comp = completion or chunks
    tps = (comp-1)/decode if decode > 0 else 0
    return {"ttft": ttft, "total": total, "decode": decode, "tokens": comp, "tps": tps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default="run")
    ap.add_argument("--base-url", default="http://127.0.0.1:8091")
    ap.add_argument("--model", default="gemma-4-26B-A4B-it-NVFP4")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    print(f"{args.label}: {args.model} @ {args.base_url}, max_tokens={args.max_tokens}, runs={args.runs}")
    print(f"{'#':>3} {'prompt':>3}  {'ttft':>7} {'tokens':>7} {'decode_s':>9} {'tok/s':>7}")
    results = []
    for i in range(args.runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        r = run(args.base_url, args.model, prompt, args.max_tokens, args.timeout)
        results.append(r)
        print(f"{i:>3} {i%len(PROMPTS):>3}  {r['ttft']:>7.3f} {r['tokens']:>7} {r['decode']:>9.2f} {r['tps']:>7.1f}")

    if results:
        tpss = [r['tps'] for r in results]
        ttfts = [r['ttft'] for r in results]
        mean_tps = sum(tpss)/len(tpss)
        stddev = (sum((x-mean_tps)**2 for x in tpss)/max(1,len(tpss)-1))**0.5
        print()
        print(f"SUMMARY {args.label} | tok/s={mean_tps:.1f}±{stddev:.1f} "
              f"(min={min(tpss):.1f}, max={max(tpss):.1f}) | "
              f"ttft={sum(ttfts)/len(ttfts):.3f}s avg")

if __name__ == "__main__":
    main()
