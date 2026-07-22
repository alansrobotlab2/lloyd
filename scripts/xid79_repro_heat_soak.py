"""Sustained saturation load on primary vLLM — no gaps, max heat soak."""
import asyncio
import json
import time

import httpx

URL = "http://127.0.0.1:8096/v1/chat/completions"
MODEL = "Qwen3.6-27B-nvfp4-sakamakismile-mtp"
CONCURRENCY = 32
DURATION_S = 30 * 60
MAX_TOKENS = 3000

BODY = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Write a very long, detailed essay about the history of computing."}],
    "max_tokens": MAX_TOKENS,
    "ignore_eos": True,
    "temperature": 0.8,
}

stats = {"done": 0, "errors": 0, "tokens": 0}


async def worker(client: httpx.AsyncClient, deadline: float):
    while time.monotonic() < deadline:
        try:
            r = await client.post(URL, json=BODY, timeout=600)
            d = r.json()
            stats["done"] += 1
            stats["tokens"] += d.get("usage", {}).get("completion_tokens", 0)
        except Exception as e:
            stats["errors"] += 1
            print(f"[{time.strftime('%H:%M:%S')}] request error: {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(2)


async def main():
    deadline = time.monotonic() + DURATION_S
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(worker(client, deadline)) for _ in range(CONCURRENCY)]
        while time.monotonic() < deadline:
            await asyncio.sleep(60)
            print(f"[{time.strftime('%H:%M:%S')}] completed={stats['done']} errors={stats['errors']} tokens={stats['tokens']}", flush=True)
        for t in tasks:
            await t
    print(f"FINAL: completed={stats['done']} errors={stats['errors']} tokens={stats['tokens']}", flush=True)


asyncio.run(main())
