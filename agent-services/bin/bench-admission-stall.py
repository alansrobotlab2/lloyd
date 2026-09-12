#!/usr/bin/env python
"""Reproduce the 09-09 admission stall on the primary, and count it the way
production does.

Two shapes, both from 2026-09-10 (architecture/vllm.md):

  verify  A = a ~120k-token context decoding continuously. B = an agent loop
          re-admitting its own growing prompt: 12 warm iterations (lloyd-be's
          run 1), then the same loop re-admitted with its first message
          changed — a cold ~200k re-prefill at iteration 13 (run 3c) — and two
          more. B's usage goes through the harness's own client and
          `_merge_usage`, and every iteration through `app.prefix_miss`, which
          is exactly what production runs. So this is also the end-to-end
          check of Layer 1: a warm loop produces no prefix_miss, a cold
          re-admission produces one and one announcement.
  cold    Only the cold admission: A decoding, B one nonce-prefixed prompt,
          then the same B again with A stopped (its prefill time alone). The
          Layer 3 max_num_batched_tokens shape.

What it reports: A's inter-chunk gaps — one SSE chunk per engine step, so a
gap is a step as the neighbour feels it — during B's prefill, B's prefill
time, and /metrics every 2 s. Read gaps from here and KV climb from /metrics:
on this build prefill tokens are invisible to tok/step and to
prompt_tokens_total until the request completes.

Why this file exists: every reproducer the investigation used (lloyd-be's
stall_repro.py, lloyd-14's kvtest.py) lived in a session scratchpad and went
with it. This one is the durable driver.

Run it with nothing else on the engine — the worker pool paused and drained,
or the backend down. It waits for an idle engine and refuses to start
otherwise. Every request goes at priority 1, so a real chat still outranks it.

    .venvs/lloyd/bin/python agent-services/bin/bench-admission-stall.py verify
    .venvs/lloyd/bin/python agent-services/bin/bench-admission-stall.py cold --label mnbt4096
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import engine_pressure, prefix_miss, vllm_metrics  # noqa: E402
from app.harness.client import stream_chat  # noqa: E402
from app.harness.loop import _merge_usage  # noqa: E402

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8096")
MODEL = "primary"
PRIORITY = 1
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
DEFAULT_OUT = ROOT / "agent-services" / "logs" / "admission-bench.jsonl"

METRIC_KEYS = (
    "vllm:iteration_tokens_total_sum", "vllm:iteration_tokens_total_count",
    "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
    "vllm:num_requests_running", "vllm:num_preemptions_total",
    "vllm:kv_cache_usage_perc",
)

A_GLOBS = ("app/**/*.py",)
B_GLOBS = ("agent_mcp/**/*.py", "workers/**/*.py", "scripts/**/*.py",
           "architecture/*.md")
TOOL_GLOBS = ("tests/*.py",)


# ── prompts ───────────────────────────────────────────────────────────


def corpus(globs: tuple[str, ...], chars: int) -> str:
    """Deterministic text from this repo — code and docs, which is what
    production prompts are made of."""
    parts: list[str] = []
    total = 0
    for pattern in globs:
        for path in sorted(ROOT.glob(pattern)):
            if "node_modules" in path.parts or not path.is_file():
                continue
            try:
                body = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            chunk = f"\n### {path.relative_to(ROOT)}\n{body}"
            parts.append(chunk)
            total += len(chunk)
            if total >= chars:
                return "".join(parts)[:chars]
    text = "".join(parts)
    while len(text) < chars:     # a small tree: repeat rather than fall short
        text += text
    return text[:chars]


async def count_tokens(client: httpx.AsyncClient, text: str) -> int:
    r = await client.post(f"{BASE}/tokenize", json={"model": MODEL, "prompt": text},
                          timeout=300)
    r.raise_for_status()
    return int(r.json()["count"])


async def text_of(client, globs, target_tokens: int, head: str) -> tuple[str, int]:
    raw = corpus(globs, int(target_tokens * 4.5))
    n = await count_tokens(client, head + raw)
    text = head + raw[: int(len(raw) * target_tokens / max(n, 1))]
    return text, await count_tokens(client, text)


# ── the engine ────────────────────────────────────────────────────────


async def gauges(client) -> dict:
    text = (await client.get(f"{BASE}/metrics", timeout=5)).text
    parsed = vllm_metrics.parse_prometheus(text)
    return {k: vllm_metrics._sum_all(parsed, k) for k in METRIC_KEYS}


async def wait_idle(client, quiet_s: float = 5.0, limit_s: float = 600.0,
                    allow_running: int = 0) -> None:
    """Nothing else may be on the engine: a momentary zero is not idle
    (memory bench-primary-pause-pool-first), so require `quiet_s` of it.

    The implementation moved to `app.vllm_metrics.wait_idle` so the evals and
    the bench runner share one definition of idle; this keeps the script's
    `SystemExit` wording, which is what a human running it reads.
    """
    try:
        await vllm_metrics.wait_idle(
            BASE, quiet_s=quiet_s, limit_s=limit_s,
            allow_running=allow_running, client=client,
        )
    except TimeoutError as exc:
        raise SystemExit(str(exc)) from exc


async def metrics_loop(client, rows: list, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            rows.append({"t": time.monotonic(), **await gauges(client)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"t": time.monotonic(), "error": str(exc)})
        try:
            await asyncio.wait_for(stop.wait(), 2.0)
        except asyncio.TimeoutError:
            pass


async def one_request(messages: list[dict], *, max_tokens: int, extra: dict | None = None,
                      cancel: asyncio.Event | None = None,
                      chunks: list | None = None) -> dict:
    """One streamed completion through the harness's own client."""
    t0 = time.monotonic()
    first = None
    usage: dict = {}
    text: list[str] = []
    body = {**NO_THINK, "max_tokens": max_tokens, **(extra or {})}
    async for chunk in stream_chat(base_url=BASE, model=MODEL, messages=messages,
                                   tools=None, extra_body=body, cancel_event=cancel,
                                   timeout_s=1800, priority=PRIORITY):
        if u := chunk.get("usage"):
            usage = _merge_usage(usage, u)
        for choice in chunk.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                now = time.monotonic()
                first = first or now
                text.append(content)
                if chunks is not None:
                    chunks.append(now)
    return {"t0": t0, "first": first, "end": time.monotonic(), "usage": usage,
            "text": "".join(text)}


# ── arithmetic ────────────────────────────────────────────────────────


def gap_stats(chunk_times: list[float], w0: float, w1: float) -> dict:
    """Gaps between consecutive A chunks whose interval overlaps [w0, w1]."""
    gaps = [(b - a) * 1000 for a, b in zip(chunk_times, chunk_times[1:])
            if b > w0 and a < w1]
    if not gaps:
        return {"n": 0}
    gaps.sort()

    def q(p: float) -> float:
        return round(gaps[min(len(gaps) - 1, int(p * len(gaps)))], 1)

    return {"n": len(gaps), "p50_ms": q(0.50), "p90_ms": q(0.90), "p99_ms": q(0.99),
            "max_ms": round(gaps[-1], 1), "mean_ms": round(statistics.mean(gaps), 1),
            "over_500ms": sum(1 for g in gaps if g > 500)}


def b_row(it: int, r: dict, released: list) -> dict:
    inp, cached = prefix_miss.usage_numbers(r["usage"])
    return {"iteration": it, "input_tokens": inp, "cache_read": cached,
            "uncached": max(0, inp - cached),
            "ttft_s": round((r["first"] or r["end"]) - r["t0"], 2),
            "total_s": round(r["end"] - r["t0"], 2),
            "prefix_miss": bool(released)}


# ── the shapes ────────────────────────────────────────────────────────


async def warm_a(a_msgs: list[dict]) -> None:
    for _ in range(2):          # the old build engaged reuse only from pass 3
        await one_request(a_msgs, max_tokens=1)


def a_decoder(a_msgs: list[dict], chunks: list, cancel: asyncio.Event) -> asyncio.Task:
    return asyncio.create_task(one_request(
        a_msgs, max_tokens=24_000, extra={"ignore_eos": True}, cancel=cancel,
        chunks=chunks))


async def wait_for_chunks(chunks: list, n: int, limit_s: float = 120.0) -> None:
    t0 = time.monotonic()
    while len(chunks) < n and time.monotonic() - t0 < limit_s:
        await asyncio.sleep(0.1)


async def run_verify(args, client, a_msgs, b_text: str, tool_pool: str) -> dict:
    events: list = []
    tracker = prefix_miss.TurnMissTracker.for_turn(
        "bench_admission", "verify", label="admission bench")

    def log(name: str, data: dict) -> None:
        events.append({"event": name, **data})

    history = [{"role": "user", "content": b_text + "\n\nIn two sentences: "
                                                    "what does the code above do?"}]
    rows: list = []
    chunks: list = []
    cancel = asyncio.Event()
    await warm_a(a_msgs)
    a_task = a_decoder(a_msgs, chunks, cancel)
    await wait_for_chunks(chunks, 60)
    alone = (chunks[5] if len(chunks) > 5 else time.monotonic(), time.monotonic())

    windows = {}
    cursor = 0
    for it in range(1, args.iterations + 1):
        if it == args.cold_at:
            # The re-admission the 09-09 stall was made of: the same loop,
            # its prefix no longer in the engine's cache.
            history[0] = {**history[0], "content":
                          f"[run {random.getrandbits(64):016x}]\n" + history[0]["content"]}
        r = await one_request(history, max_tokens=args.answer_tokens)
        released = prefix_miss.record_iteration(
            tracker, it, r["usage"], duration_ms=int((r["end"] - r["t0"]) * 1000),
            log=log)
        rows.append(b_row(it, r, released))
        if it == args.cold_at:
            windows["cold_prefill"] = (r["t0"], r["first"] or r["end"])
        if it == 1:
            windows["first_prefill"] = (r["t0"], r["first"] or r["end"])
        if it == args.cold_at - 1:
            windows["warm_loop"] = (windows["first_prefill"][1], r["end"])
        history.append({"role": "assistant", "content": r["text"]})
        tool = tool_pool[cursor:cursor + 1600]
        cursor += 1600
        history.append({"role": "user", "content":
                        f"Tool result:\n{tool}\n\nOne more sentence on this."})

    cancel.set()
    await asyncio.gather(a_task, return_exceptions=True)
    if prefix_miss._tasks:
        await asyncio.gather(*list(prefix_miss._tasks), return_exceptions=True)
    return {
        "a_alone": gap_stats(chunks, *alone),
        "a_during_warm_loop": gap_stats(chunks, *windows.get("warm_loop", (0, 0))),
        "a_during_cold_prefill": gap_stats(chunks, *windows["cold_prefill"]),
        "b_cold_prefill_s": round(windows["cold_prefill"][1] - windows["cold_prefill"][0], 2),
        "b_iterations": rows,
        "tracker": {**tracker.summary(), "announced": tracker.announced,
                    "measured": tracker.measured},
        "events": events,
    }


async def run_cold(args, client, a_msgs, b_text: str) -> dict:
    chunks: list = []
    cancel = asyncio.Event()
    await warm_a(a_msgs)
    a_task = a_decoder(a_msgs, chunks, cancel)
    await wait_for_chunks(chunks, 60)
    alone = (chunks[5] if len(chunks) > 5 else time.monotonic(), time.monotonic())

    def cold_msgs() -> list[dict]:
        return [{"role": "user", "content": f"[run {random.getrandbits(64):016x}]\n"
                 + b_text + "\n\nIn one sentence: what does the code above do?"}]

    beside = await one_request(cold_msgs(), max_tokens=16)
    await asyncio.sleep(3)
    cancel.set()
    await asyncio.gather(a_task, return_exceptions=True)
    await wait_idle(client, quiet_s=2)
    alone_b = await one_request(cold_msgs(), max_tokens=16)
    w = (beside["t0"], beside["first"] or beside["end"])
    return {
        "a_alone": gap_stats(chunks, *alone),
        "a_during_cold_prefill": gap_stats(chunks, *w),
        "b_prefill_beside_a_s": round(w[1] - w[0], 2),
        "b_prefill_alone_s": round((alone_b["first"] or alone_b["end"]) - alone_b["t0"], 2),
        "b_input_tokens": prefix_miss.usage_numbers(beside["usage"])[0],
        "b_cache_read": prefix_miss.usage_numbers(beside["usage"])[1],
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shape", choices=("verify", "cold"))
    ap.add_argument("--label", default="")
    ap.add_argument("--a-tokens", type=int, default=120_000)
    ap.add_argument("--b-tokens", type=int, default=200_000)
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--cold-at", type=int, default=13)
    ap.add_argument("--answer-tokens", type=int, default=200)
    ap.add_argument("--no-announce", action="store_true",
                    help="decide, but do not toast (the decision is still recorded)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.no_announce:
        prefix_miss._announce = lambda *a, **k: {"suppressed": True}

    async with httpx.AsyncClient() as client:
        await wait_idle(client)
        run = f"{random.getrandbits(32):08x}"
        a_text, a_n = await text_of(client, A_GLOBS, args.a_tokens, f"[A {run}]\n")
        b_text, b_n = await text_of(client, B_GLOBS, args.b_tokens, f"[B {run}]\n")
        tool_pool = corpus(TOOL_GLOBS, 1600 * (args.iterations + 1))
        a_msgs = [{"role": "user", "content": a_text + "\n\nWrite a long, "
                   "detailed, numbered commentary on the code above. Keep going."}]

        # The pressure sampler the alert reads, at the dashboard's cadence.
        sampler = asyncio.create_task(engine_pressure._run(BASE, 2.0))
        rows: list = []
        stop = asyncio.Event()
        mtask = asyncio.create_task(metrics_loop(client, rows, stop))
        t0 = time.monotonic()
        try:
            if args.shape == "verify":
                result = await run_verify(args, client, a_msgs, b_text, tool_pool)
            else:
                result = await run_cold(args, client, a_msgs, b_text)
        finally:
            stop.set()
            await asyncio.gather(mtask, return_exceptions=True)
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
        preempt = [r.get("vllm:num_preemptions_total") for r in rows
                   if r.get("vllm:num_preemptions_total") is not None]
        kv = [r["vllm:kv_cache_usage_perc"] for r in rows
              if r.get("vllm:kv_cache_usage_perc") is not None]

    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "shape": args.shape,
        "label": args.label, "a_tokens": a_n, "b_tokens": b_n,
        "wall_s": round(time.monotonic() - t0, 1),
        "preemptions": (preempt[-1] - preempt[0]) if len(preempt) > 1 else None,
        "kv_max": max(kv) if kv else None,
        **result,
        "metrics": [{k.split(":", 1)[-1]: v for k, v in r.items()} for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    brief = {k: v for k, v in record.items() if k not in ("metrics", "events")}
    print(json.dumps(brief, indent=2, default=str))
    print(f"(full record appended to {args.out})")


if __name__ == "__main__":
    asyncio.run(main())
