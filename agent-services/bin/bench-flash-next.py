#!/usr/bin/env python3
"""A/B harness for the Qwen3.8-Flash-Next primary slot.

One arm per engine boot. Measures the three things a config change on this
model can move, and records the conditions under which it measured them:

  decode   single-stream tok/s, greedy + ignore_eos so the token count is
           exact and the run cannot end early on an EOS.
  batch    aggregate tok/s at 4 and 8 concurrent streams.
  prefill  cold TTFT at two prompt depths, with unique random content so
           the prefix cache cannot serve any of it (asserted via
           usage.prompt_tokens_details.cached_tokens).
  mtp      accepted / drafted tokens over the run, from the engine's own
           spec_decode counters (a delta, not the since-boot absolute).

WHY THE CONTAMINATION MONITOR IS NOT OPTIONAL
A background thread samples vllm:num_requests_running every 0.5 s for the
whole run. On 2026-09-08 a "single-stream" measurement of this slot read
50-100 tok/s against a recorded 182 because the worker pool had started
jobs underneath it; nothing in the numbers said so. Any phase that saw more
requests running than it launched is reported `contaminated` and must be
discarded rather than compared. Pause and drain the pool first
(POST /api/workers/pause), or stop the backend outright.

Requests are sent at priority 1 to match worker traffic: the engine runs
--scheduling-policy priority, so a priority-0 bench preempts live work and
forces its 50k-token prefills to recompute.

Usage:
  bench-flash-next.py <arm-label> [--out results.json] [--quick]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8096"
MODEL = "primary"

# Deterministic corpus for cold-prefill prompts. Random *words* rather than
# random tokens: a token-id soup prefills at a different rate than prose and
# would not describe the real workload.
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee "
    "zulu granite copper marble velvet ember tundra harbor lantern meadow orbit prism "
    "quartz ridge saddle timber vortex willow zephyr anvil beacon cinder dial ferrous "
    "gantry hollow ingot jetty kiln loam mantle nexus oxide plinth quarry rivet sluice"
).split()

DECODE_PROMPT = "Count upward from one, writing each number as an English word, one per line."


def _get(path: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return r.read().decode()


def metrics() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        body = _get("/metrics")
    except Exception:
        return out
    for line in body.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    return out


def wait_ready(limit: float = 1800.0) -> float:
    """Block until the server answers and reports no work in flight."""
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            _get("/health", timeout=5)
            m = metrics()
            if m.get("vllm:num_requests_running", 1) == 0 and m.get("vllm:num_requests_waiting", 1) == 0:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit(f"engine not ready+idle within {limit:.0f}s")


class Monitor(threading.Thread):
    """Samples in-flight request count so a phase can prove it ran alone."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                body = _get("/metrics", timeout=2)
                run = wait = 0.0
                for line in body.splitlines():
                    if line.startswith("vllm:num_requests_running{"):
                        run += float(line.rsplit(" ", 1)[1])
                    elif line.startswith("vllm:num_requests_waiting{"):
                        wait += float(line.rsplit(" ", 1)[1])
                self.samples.append((time.time(), run + wait))
            except Exception:
                pass
            self._stop.wait(0.5)

    def stop(self) -> None:
        self._stop.set()

    def peak_between(self, t0: float, t1: float) -> float:
        vals = [v for ts, v in self.samples if t0 <= ts <= t1]
        return max(vals) if vals else 0.0


def stream(prompt: str, max_tokens: int, *, greedy: bool = True,
           ignore_eos: bool = True, timeout: float = 900.0) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0 if greedy else 0.7,
        "ignore_eos": ignore_eos,
        # Match live worker traffic so this never preempts real work.
        "priority": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    last = t0
    usage = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        while True:
            piece = r.read(8192)
            if not piece:
                break
            buf += piece
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                try:
                    d = json.loads(line[6:])
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        last = now
                        chunks += 1
    usage = usage or {}
    completion = usage.get("completion_tokens", chunks)
    decode_s = last - (t0 + (ttft or 0.0))
    return {
        "ttft_s": ttft,
        "wall_s": last - t0,
        "decode_s": decode_s,
        "tokens": completion,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        # tokens-1: the first token is attributed to TTFT, not to decode.
        "tps": (completion - 1) / decode_s if decode_s > 0 and completion > 1 else 0.0,
    }


def phase_decode(mon: Monitor, runs: int, max_tokens: int) -> dict:
    per = []
    t0 = time.time()
    for _ in range(runs):
        per.append(stream(DECODE_PROMPT, max_tokens))
    t1 = time.time()
    tps = [r["tps"] for r in per]
    return {
        "runs": runs,
        "max_tokens": max_tokens,
        "tps_median": statistics.median(tps),
        "tps_all": [round(x, 1) for x in tps],
        "ttft_ms_median": statistics.median(r["ttft_s"] for r in per) * 1000,
        "peak_inflight": mon.peak_between(t0, t1),
        "contaminated": mon.peak_between(t0, t1) > 1,
    }


def phase_batch(mon: Monitor, conc: int, max_tokens: int) -> dict:
    res: list[dict | None] = [None] * conc
    nonce = random.randrange(10 ** 9)

    def work(i: int) -> None:
        res[i] = stream(f"[{nonce}-{i}] {DECODE_PROMPT}", max_tokens)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
    t0 = time.time()
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0
    t1 = time.time()
    done = [r for r in res if r]
    total = sum(r["tokens"] for r in done)
    peak = mon.peak_between(t0, t1)
    return {
        "concurrency": conc,
        "max_tokens": max_tokens,
        "aggregate_tps": total / wall if wall > 0 else 0.0,
        "per_stream_mean_tps": statistics.mean(r["tps"] for r in done),
        "per_stream_min_tps": min(r["tps"] for r in done),
        "ttft_ms_mean": statistics.mean(r["ttft_s"] for r in done) * 1000,
        "wall_s": wall,
        "tokens": total,
        "peak_inflight": peak,
        "contaminated": peak > conc,
    }


def phase_prefill(mon: Monitor, target_words: int) -> dict:
    rnd = random.Random(random.randrange(10 ** 9))
    nonce = rnd.randrange(10 ** 9)
    text = " ".join(rnd.choice(WORDS) for _ in range(target_words))
    prompt = (
        f"[cold {nonce}] Below is a list of code words.\n{text}\n\n"
        "Reply with the single word OK."
    )
    t0 = time.time()
    r = stream(prompt, 8, ignore_eos=False)
    t1 = time.time()
    pt = r["prompt_tokens"] or 0
    peak = mon.peak_between(t0, t1)
    return {
        "prompt_tokens": pt,
        "cached_tokens": r["cached_tokens"],
        "ttft_s": r["ttft_s"],
        "prefill_tps": pt / r["ttft_s"] if r["ttft_s"] else 0.0,
        "peak_inflight": peak,
        # A cold prefill that served cached blocks is not a cold prefill.
        "contaminated": peak > 1 or bool(r["cached_tokens"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--out", default="/tmp/flash-next-arms.jsonl")
    ap.add_argument("--decode-runs", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    waited = wait_ready()
    print(f"[{args.label}] engine ready+idle after {waited:.0f}s", flush=True)

    mon = Monitor()
    mon.start()
    before = metrics()

    # Warmup. Three passes, not one: the Triton QSA kernels JIT on first use
    # (the boot log names them), and cross-request prefix reuse on this model
    # does not engage until the third pass, so a single warmup measures the
    # cold path and calls it steady state.
    for _ in range(3):
        stream(DECODE_PROMPT, 64)
    print(f"[{args.label}] warmup done", flush=True)

    out: dict = {"label": args.label, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out["decode"] = phase_decode(mon, args.decode_runs, args.decode_tokens)
    print(f"  decode   {out['decode']['tps_median']:7.1f} tok/s  {out['decode']['tps_all']}", flush=True)

    # 512, not 256. At 256 a stream spends ~2 s decoding behind a ~1-2 s TTFT,
    # so time-to-first-token is half the wall clock the aggregate divides by
    # and thread-start stagger shows up as engine throughput: the same config
    # measured 376, 363 and 314 tok/s at conc=8 on 2026-09-08. Doubling the
    # generation makes decode dominate the window being measured.
    out["batch"] = []
    for conc in ((4,) if args.quick else (4, 8)):
        b = phase_batch(mon, conc, 512)
        out["batch"].append(b)
        print(f"  conc={conc:<2}   {b['aggregate_tps']:7.1f} tok/s agg   "
              f"per-stream {b['per_stream_mean_tps']:6.1f}   ttft {b['ttft_ms_mean']:5.0f} ms", flush=True)

    # Two passes per depth, and the SECOND is the measurement. The QSA and
    # sparse-GQA Triton kernels JIT per shape (the boot log names them under
    # jit_monitor), and a config that has never booted before also compiles
    # cold -- A2 on 2026-09-08 spent 26.6 s in compilation where the incumbent
    # config spent 0.34 s, then measured its first 35k prefill 42% slower than
    # the incumbent and the difference was entirely warmup. Content is unique
    # per pass, so the prefix cache serves neither of them (asserted).
    out["prefill"] = []
    for words in ((24000,) if args.quick else (24000, 72000)):
        first = phase_prefill(mon, words)
        second = phase_prefill(mon, words)
        second["jit_pass_tps"] = first["prefill_tps"]
        out["prefill"].append(second)
        print(f"  prefill  {second['prompt_tokens']:>7} tok  {second['prefill_tps']:7.0f} tok/s  "
              f"ttft {second['ttft_s']:6.2f}s  cached={second['cached_tokens']}"
              f"   (first/JIT pass {first['prefill_tps']:.0f})", flush=True)

    after = metrics()
    drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - before.get("vllm:spec_decode_num_drafts_total", 0)
    draft_tok = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    out["mtp"] = {
        "drafts": drafts,
        "draft_tokens": draft_tok,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / draft_tok if draft_tok else None,
        # 1 free token per step + whatever the draft got right.
        "tokens_per_step": (accepted + drafts) / drafts if drafts else None,
    }
    if drafts:
        print(f"  mtp      {out['mtp']['tokens_per_step']:.2f} tok/step  "
              f"acceptance {out['mtp']['acceptance_rate']:.1%}", flush=True)

    mon.stop()
    flagged = [k for k in ("decode",) if out[k]["contaminated"]]
    flagged += [f"conc{b['concurrency']}" for b in out["batch"] if b["contaminated"]]
    flagged += [f"prefill{p['prompt_tokens']}" for p in out["prefill"] if p["contaminated"]]
    out["contaminated_phases"] = flagged
    if flagged:
        print(f"  !! CONTAMINATED: {flagged} — other traffic shared the engine; discard", flush=True)

    with open(args.out, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"[{args.label}] appended to {args.out}", flush=True)


if __name__ == "__main__":
    main()
