"""vLLM engine telemetry — scrape /metrics, parse, and derive rates.

Every model in `config.yaml:models` is a vLLM server exposing a Prometheus
text endpoint at `<base_url>/metrics`. This module turns that into the
snapshot the Mission Control dashboard renders.

Two kinds of number come out of a Prometheus scrape and they need very
different handling:

  * **Gauges** (`num_requests_running`, `kv_cache_usage_perc`) are the
    live value. Read and report.
  * **Counters** (`prompt_tokens_total`, `prefix_cache_hits_total`) are
    monotonic since engine boot. Their absolute value is close to
    meaningless on a dashboard — 437M prompt tokens tells you the box has
    been up a while, not what it is doing now. What matters is the
    *rate*, so we keep the previous scrape per engine and divide the
    delta by the elapsed wall time.

The previous-sample cache is process-local and in-memory. A backend
restart means the first scrape after boot reports `null` rates rather
than a fabricated spike — a counter reset (engine restart) is detected
the same way, by the counter going backwards, and also yields `null`.
Never report a negative or post-reset rate as though it were throughput.

Latency histograms (`time_to_first_token_seconds`) are exposed by
Prometheus as `_sum` / `_count` pairs. Those are also cumulative, so the
same delta treatment gives a *windowed* mean — the mean TTFT over the
last poll interval rather than the since-boot average, which is what an
operator watching a live dashboard actually wants.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable

import httpx

logger = logging.getLogger("lloyd-server")

# How long to wait on a single engine's /metrics. The endpoint is a local
# in-memory render; if it takes longer than this the engine is wedged and
# we would rather show it as unreachable than stall the whole dashboard.
SCRAPE_TIMEOUT_S = 2.5

# Counters we report as per-second rates.
_RATE_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:num_preemptions_total",
)

# Histogram `_sum`/`_count` pairs we report as a windowed mean.
_LATENCY_HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_time_per_output_token_seconds",
)

# Ratio counters: (name, hits_metric, queries_metric). Reported both
# since-boot and windowed.
_RATIOS = (
    ("prefix_cache", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"),
    (
        "spec_decode",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
    ),
)


# ── Prometheus text parsing ────────────────────────────────────────────


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus text exposition into {metric: [(labels, value)]}.

    Deliberately minimal: we only need `name{label="v",...} value` and
    bare `name value`. Comment lines (`# HELP`, `# TYPE`) are skipped.
    A malformed line is skipped rather than raising — a single bad line
    in a 900-line scrape must not blank the whole dashboard panel.
    """
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            # Split the trailing value off first: the label block can
            # contain spaces inside quoted values, so rsplit is safer
            # than a left-to-right split.
            head, _, raw_value = line.rpartition(" ")
            if not head:
                continue
            value = float(raw_value)
        except ValueError:
            continue
        if head.endswith("}") and "{" in head:
            name, _, label_blob = head.partition("{")
            labels = _parse_labels(label_blob[:-1])
        else:
            name, labels = head, {}
        out.setdefault(name.strip(), []).append((labels, value))
    return out


def _parse_labels(blob: str) -> dict[str, str]:
    """Parse `a="1",b="2"` into a dict. Unquoted/odd pairs are dropped."""
    labels: dict[str, str] = {}
    for part in _split_labels(blob):
        key, _, val = part.partition("=")
        key = key.strip()
        val = val.strip()
        if key and len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            labels[key] = val[1:-1]
    return labels


def _split_labels(blob: str) -> Iterable[str]:
    """Split on commas that are not inside a quoted value."""
    buf: list[str] = []
    in_quotes = False
    for ch in blob:
        if ch == '"':
            in_quotes = not in_quotes
        if ch == "," and not in_quotes:
            yield "".join(buf)
            buf = []
        else:
            buf.append(ch)
    if buf:
        yield "".join(buf)


def _first(
    parsed: dict[str, list[tuple[dict[str, str], float]]],
    name: str,
    **match: str,
) -> float | None:
    """Value of the first series for `name` matching every label in `match`."""
    for labels, value in parsed.get(name, []):
        if all(labels.get(k) == v for k, v in match.items()):
            return value
    return None


def _sum_all(
    parsed: dict[str, list[tuple[dict[str, str], float]]], name: str
) -> float | None:
    """Sum every series for `name` (used across engine shards)."""
    series = parsed.get(name)
    if not series:
        return None
    return sum(v for _labels, v in series)


# ── Rate derivation ────────────────────────────────────────────────────

# alias -> {metric_name: value, "_t": scrape_monotonic_time}
_previous: dict[str, dict[str, float]] = {}


def _rate(prev: dict[str, float] | None, curr: dict[str, float], key: str) -> float | None:
    """Per-second rate for a counter, or None when it can't be trusted.

    Returns None on the first scrape (no baseline) and on a counter
    reset (value went backwards — the engine restarted). Reporting a
    post-reset delta would render the engine's entire boot history as a
    single one-second spike.
    """
    if prev is None:
        return None
    before, after = prev.get(key), curr.get(key)
    if before is None or after is None:
        return None
    elapsed = curr["_t"] - prev["_t"]
    if elapsed <= 0:
        return None
    delta = after - before
    if delta < 0:
        return None
    return delta / elapsed


def _windowed_mean(
    prev: dict[str, float] | None, curr: dict[str, float], histogram: str
) -> float | None:
    """Mean of a Prometheus histogram over the last poll interval."""
    if prev is None:
        return None
    sum_key, count_key = f"{histogram}_sum", f"{histogram}_count"
    d_sum = _delta(prev, curr, sum_key)
    d_count = _delta(prev, curr, count_key)
    if d_sum is None or d_count is None or d_count <= 0:
        return None
    return d_sum / d_count


def _delta(prev: dict[str, float], curr: dict[str, float], key: str) -> float | None:
    before, after = prev.get(key), curr.get(key)
    if before is None or after is None:
        return None
    delta = after - before
    return None if delta < 0 else delta


def _ratio(hits: float | None, queries: float | None) -> float | None:
    if hits is None or queries is None or queries <= 0:
        return None
    return hits / queries


# ── Snapshot assembly ──────────────────────────────────────────────────


def _snapshot_from_text(alias: str, text: str) -> dict[str, Any]:
    """Turn one engine's raw /metrics body into a dashboard snapshot."""
    parsed = parse_prometheus(text)

    # Model name comes off any labelled series — every vLLM metric carries
    # it, so we take it from the first one we find rather than requiring a
    # specific metric to be present.
    model_name = ""
    for series in parsed.values():
        for labels, _v in series:
            if labels.get("model_name"):
                model_name = labels["model_name"]
                break
        if model_name:
            break

    # Flatten the counters we track into a scalar map for delta math.
    curr: dict[str, float] = {"_t": time.monotonic()}
    tracked = [
        *_RATE_COUNTERS,
        *(f"{h}_sum" for h in _LATENCY_HISTOGRAMS),
        *(f"{h}_count" for h in _LATENCY_HISTOGRAMS),
        *(m for _n, a, b in _RATIOS for m in (a, b)),
    ]
    for metric in tracked:
        value = _sum_all(parsed, metric)
        if value is not None:
            curr[metric] = value

    prev = _previous.get(alias)
    _previous[alias] = curr

    running = _sum_all(parsed, "vllm:num_requests_running")
    waiting = _sum_all(parsed, "vllm:num_requests_waiting")

    waiting_by_reason = {
        labels.get("reason", "?"): value
        for labels, value in parsed.get("vllm:num_requests_waiting_by_reason", [])
        if value
    }
    finished_by_reason = {
        labels.get("finished_reason", "?"): value
        for labels, value in parsed.get("vllm:request_success_total", [])
    }

    snapshot: dict[str, Any] = {
        "alias": alias,
        "reachable": True,
        "model_name": model_name,
        "awake": _first(parsed, "vllm:engine_sleep_state", sleep_state="awake") == 1.0,
        # Live gauges — the "what is it doing right now" row.
        "requests_running": int(running) if running is not None else None,
        "requests_waiting": int(waiting) if waiting is not None else None,
        "requests_waiting_by_reason": {k: int(v) for k, v in waiting_by_reason.items()},
        "kv_cache_usage": _sum_all(parsed, "vllm:kv_cache_usage_perc"),
        # Throughput over the last poll interval.
        "prompt_tokens_per_s": _rate(prev, curr, "vllm:prompt_tokens_total"),
        "generation_tokens_per_s": _rate(prev, curr, "vllm:generation_tokens_total"),
        "preemptions_per_s": _rate(prev, curr, "vllm:num_preemptions_total"),
        # Latency over the last poll interval.
        "ttft_s": _windowed_mean(prev, curr, "vllm:time_to_first_token_seconds"),
        "itl_s": _windowed_mean(prev, curr, "vllm:inter_token_latency_seconds"),
        # Since-boot totals worth keeping as context for the rates.
        "prompt_tokens_total": curr.get("vllm:prompt_tokens_total"),
        "generation_tokens_total": curr.get("vllm:generation_tokens_total"),
        "preemptions_total": curr.get("vllm:num_preemptions_total"),
        "finished_by_reason": {k: int(v) for k, v in finished_by_reason.items()},
    }

    # Cache/acceptance ratios, both lifetime and windowed. The windowed
    # one is what moves during a turn; the lifetime one is the baseline.
    for name, hits_metric, queries_metric in _RATIOS:
        snapshot[f"{name}_hit_rate"] = _ratio(
            curr.get(hits_metric), curr.get(queries_metric)
        )
        snapshot[f"{name}_hit_rate_recent"] = (
            _ratio(_delta(prev, curr, hits_metric), _delta(prev, curr, queries_metric))
            if prev is not None
            else None
        )

    return snapshot


async def _scrape_one(client: httpx.AsyncClient, alias: str, base_url: str) -> dict[str, Any]:
    """Fetch and parse one engine, degrading to `reachable: false`.

    An engine being down is a normal dashboard state (the secondary is
    stopped whenever `secondary_enabled` is false), not an error — the
    panel shows it greyed rather than the whole request failing.
    """
    url = base_url.rstrip("/") + "/metrics"
    try:
        resp = await client.get(url, timeout=SCRAPE_TIMEOUT_S)
        resp.raise_for_status()
    except Exception as exc:
        # Drop any stale baseline: when the engine comes back its counters
        # will have reset, and a rate computed against the pre-outage
        # sample would be nonsense.
        _previous.pop(alias, None)
        return {
            "alias": alias,
            "reachable": False,
            "base_url": base_url,
            "error": f"{type(exc).__name__}: {exc}",
        }
    snapshot = _snapshot_from_text(alias, resp.text)
    snapshot["base_url"] = base_url
    return snapshot


async def collect(engines: dict[str, str]) -> list[dict[str, Any]]:
    """Scrape every engine concurrently. `engines` is {alias: base_url}."""
    if not engines:
        return []
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_scrape_one(client, alias, url) for alias, url in engines.items()),
            return_exceptions=True,
        )
    out: list[dict[str, Any]] = []
    for (alias, base_url), result in zip(engines.items(), results):
        if isinstance(result, BaseException):
            logger.warning("vllm scrape for %s raised: %s", alias, result)
            out.append({
                "alias": alias,
                "reachable": False,
                "base_url": base_url,
                "error": str(result),
            })
        else:
            out.append(result)
    return out


def configured_engines() -> dict[str, str]:
    """{alias: base_url} for every model defined in config.yaml."""
    from app.config import CONFIG

    engines: dict[str, str] = {}
    for alias, cfg in (CONFIG.get("models") or {}).items():
        base = (cfg or {}).get("base_url") or (cfg or {}).get("env", {}).get(
            "ANTHROPIC_BASE_URL", ""
        )
        if base:
            engines[alias] = base
    return engines
