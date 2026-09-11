"""Primary-engine pressure, sampled in the background.

Three readers need the same two numbers — how full the primary's KV cache is,
and how many requests it is serving — and none of them can afford to ask the
engine at the moment it needs to know:

  * the worker pool's KV gate (`workers/pool.py`) decides on every claim,
    every two seconds per slot, and a claim must not wait on an HTTP call;
  * the prefix-miss alert (`app/prefix_miss.py`) asks whether anything else
    was running *during* an iteration that has already finished — a question
    about the past, which only a history can answer;
  * the dashboard's KV pressure gauge is a p90 over five minutes, and the
    dashboard only polls while somebody has the page open.

So one task scrapes the primary's /metrics every few seconds and keeps a
bounded ring of samples. It reads through `vllm_metrics.gauges_from_text`,
which is stateless on purpose: the dashboard's own reader keeps a per-engine
baseline for its rate math, and a second poller going through it would
quietly halve that window and make both readings wrong.

What `kv_cache_usage_perc` measures matters for reading it. It is the share
of blocks *referenced by running requests*; a cached prefix whose turn is
between iterations sits in the free pool, evictable, and does not count. So
the gauge is resident pressure — how much of the pool is spoken for right
now — and the free remainder is exactly where a paused turn's prefix has to
survive until its next iteration. That is the quantity the 09-09 stall
turned on (`architecture/vllm.md`).

Reads fail open. A stale or missing sample means "unknown", and each reader
treats unknown as "no pressure": the gate claims, the alert stays quiet, the
gauge reads null. A pressure signal that failed closed would turn an
unreachable /metrics endpoint into a stopped worker pool.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx

from app import vllm_metrics

logger = logging.getLogger("lloyd-server")

#: Seconds between scrapes. /metrics is a local in-memory render; 5 s puts
#: two or three samples inside any cold 100k+ prefill (~10 s at ~10k tok/s),
#: which is what the alert's "was anything else running" question needs.
DEFAULT_INTERVAL_S = 5.0
#: How much history to keep, and the window the dashboard's p90 is over.
DEFAULT_WINDOW_S = 300.0
#: A sample older than this is not a reading: six missed scrapes.
STALE_AFTER_S = 30.0
#: The dashboard's line. The plan's acceptance is KV p90 < 60% on a normal
#: day; the line sits a little above the gate's 60% so a gauge that is merely
#: at the gate does not read as an alarm.
DEFAULT_KV_WARN_LINE = 0.65


@dataclass(frozen=True)
class Sample:
    t: float                 # time.monotonic() at the scrape
    wall: float              # time.time(), for anything a human reads
    kv: float | None         # vllm:kv_cache_usage_perc, 0..1
    running: int | None      # vllm:num_requests_running
    waiting: int | None      # vllm:num_requests_waiting


_samples: deque[Sample] = deque()
_task: asyncio.Task | None = None
_base_url: str = ""
_last_error: str = ""


def _cfg() -> dict[str, Any]:
    try:
        from app.config import CONFIG
        return dict(CONFIG.get("engine_pressure") or {})
    except Exception:
        return {}


def window_s() -> float:
    return float(_cfg().get("window_seconds", DEFAULT_WINDOW_S))


def kv_warn_line() -> float:
    return float(_cfg().get("kv_warn_line", DEFAULT_KV_WARN_LINE))


def primary_alias() -> str:
    try:
        from app.config import CONFIG
        return str((CONFIG.get("model") or {}).get("default") or "primary")
    except Exception:
        return "primary"


def primary_base_url() -> str:
    try:
        from app.config import CONFIG
    except Exception:
        return ""
    cfg = (CONFIG.get("models") or {}).get(primary_alias()) or {}
    return str(cfg.get("base_url")
               or (cfg.get("env") or {}).get("ANTHROPIC_BASE_URL", "") or "")


# ── the ring ──────────────────────────────────────────────────────────


def record(sample: Sample, *, window: float | None = None) -> None:
    """Append one sample and drop what has aged out of the window."""
    _samples.append(sample)
    horizon = sample.t - (window if window is not None else window_s())
    while _samples and _samples[0].t < horizon:
        _samples.popleft()


def reset() -> None:
    """Forget every sample. For tests, and for a sampler pointed elsewhere."""
    global _last_error
    _samples.clear()
    _last_error = ""


def latest(max_age_s: float = STALE_AFTER_S, *,
           now: float | None = None) -> Sample | None:
    """The newest sample, or None when there is none fresh enough to trust."""
    if not _samples:
        return None
    s = _samples[-1]
    now = time.monotonic() if now is None else now
    return s if now - s.t <= max_age_s else None


def _since(t0: float) -> list[Sample]:
    return [s for s in _samples if s.t >= t0]


def kv_percentile(q: float, *, window: float | None = None,
                  now: float | None = None) -> float | None:
    """Nearest-rank percentile of KV usage over the window, or None if empty."""
    now = time.monotonic() if now is None else now
    w = window if window is not None else window_s()
    vals = sorted(s.kv for s in _since(now - w) if s.kv is not None)
    if not vals:
        return None
    rank = max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))
    return vals[rank]


def neighbours_during(t0: float, *, now: float | None = None) -> int | None:
    """How many OTHER requests the engine served between `t0` and now.

    The request asking was itself one of the running ones while it ran, so
    this is the peak `num_requests_running` inside the window minus one. None
    when no sample landed inside the window — the caller decides what unknown
    means.
    """
    now = time.monotonic() if now is None else now
    vals = [s.running for s in _samples
            if s.running is not None and t0 <= s.t <= now]
    if not vals:
        return None
    return max(0, max(vals) - 1)


def snapshot(*, now: float | None = None) -> dict[str, Any]:
    """What the dashboard renders beside the primary engine's KV meter."""
    now = time.monotonic() if now is None else now
    w = window_s()
    fresh = latest(now=now)
    return {
        "alias": primary_alias(),
        "base_url": _base_url,
        "sampling": _task is not None and not _task.done(),
        "window_s": w,
        "samples": len([s for s in _since(now - w) if s.kv is not None]),
        "kv_now": fresh.kv if fresh else None,
        # p50 beside p90 because the two disagree for a reason: a cold long
        # prefill references ~2.5x its resident footprint while it builds
        # (measured 2026-09-10: 0.20 -> 0.96 over a 200k prefill, 0.50 the
        # moment it lands), so the tail is prefills and the middle is residents.
        "kv_p50": kv_percentile(0.5, window=w, now=now),
        "kv_p90": kv_percentile(0.9, window=w, now=now),
        "kv_max": kv_percentile(1.0, window=w, now=now),
        "warn_line": kv_warn_line(),
        "stale": fresh is None,
        "error": _last_error or None,
    }


# ── the sampler ───────────────────────────────────────────────────────


async def scrape_once(base_url: str = "", *,
                      client: httpx.AsyncClient | None = None
                      ) -> dict[str, Any] | None:
    """One read of the primary's gauges, or None. Never raises."""
    global _last_error
    base = (base_url or _base_url or primary_base_url()).rstrip("/")
    if not base:
        return None
    try:
        if client is None:
            async with httpx.AsyncClient() as cli:
                resp = await cli.get(f"{base}/metrics",
                                     timeout=vllm_metrics.SCRAPE_TIMEOUT_S)
        else:
            resp = await client.get(f"{base}/metrics",
                                    timeout=vllm_metrics.SCRAPE_TIMEOUT_S)
        resp.raise_for_status()
        return vllm_metrics.gauges_from_text(resp.text)
    except Exception as exc:  # noqa: BLE001 — an engine being down is a state
        msg = f"{type(exc).__name__}: {exc}"
        if msg != _last_error:
            logger.info("engine_pressure: %s/metrics unreadable: %s", base, msg)
        _last_error = msg
        return None


async def _run(base_url: str, interval: float) -> None:
    global _last_error
    async with httpx.AsyncClient() as client:
        while True:
            g = await scrape_once(base_url, client=client)
            if g is not None:
                _last_error = ""
                record(Sample(time.monotonic(), time.time(),
                              g.get("kv_cache_usage"),
                              g.get("requests_running"),
                              g.get("requests_waiting")))
            await asyncio.sleep(interval)


async def start() -> None:
    """Startup hook. Idempotent; a no-op when disabled or unconfigured."""
    global _task, _base_url
    cfg = _cfg()
    if not cfg.get("enabled", True):
        logger.info("engine_pressure disabled in config — no KV gate input, "
                    "no p90 gauge, no neighbour count for the miss alert")
        return
    if _task is not None and not _task.done():
        return
    base = primary_base_url()
    if not base:
        logger.warning("engine_pressure: no base_url for %r — not sampling",
                       primary_alias())
        return
    _base_url = base
    interval = max(1.0, float(cfg.get("sample_interval_seconds",
                                      DEFAULT_INTERVAL_S)))
    _task = asyncio.create_task(_run(base, interval),
                                name="lloyd-engine-pressure")
    logger.info("engine_pressure: sampling %s/metrics every %.0fs", base, interval)


async def stop() -> None:
    """Shutdown hook."""
    global _task
    task, _task = _task, None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
