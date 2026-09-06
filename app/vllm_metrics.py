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


# ── llama.cpp ──────────────────────────────────────────────────────────
#
# Not every engine on this box is vLLM. The secondary slot runs
# llama-server (Qwen3.6-35B-A3B UD-Q3_K_XL: a GGUF Q3 is the only build
# of it that fits a 24 GB card at the full 262144 window). llama.cpp
# exposes the same kind of Prometheus endpoint under a `llamacpp:`
# prefix with its own names, so rather than fork the snapshot logic we
# rename its series into the vLLM vocabulary and let one code path do
# the delta math. The dashboard renders one card shape either way.
#
# One source metric can feed two targets, hence the tuple values:
#
#   * llama.cpp has no latency *histograms*, but
#     `tokens_predicted_seconds_total / tokens_predicted_total` is
#     exactly a sum/count pair over generated tokens, which is what
#     `_windowed_mean` wants — so it becomes inter-token latency
#     directly. There is no equivalent for TTFT (no per-request count),
#     so `ttft_s` stays None rather than being faked from prefill time.
#
#   * `prompt_tokens_cached_total / prompt_tokens_total` is the prefix
#     cache hit rate, in tokens, which is the same quantity vLLM's
#     prefix_cache_hits/queries pair reports.
_LLAMACPP_TO_VLLM: dict[str, tuple[str, ...]] = {
    "llamacpp:requests_processing": ("vllm:num_requests_running",),
    "llamacpp:requests_deferred": ("vllm:num_requests_waiting",),
    "llamacpp:prompt_tokens_total": (
        "vllm:prompt_tokens_total",
        "vllm:prefix_cache_queries_total",
    ),
    "llamacpp:prompt_tokens_cached_total": (
        "vllm:prompt_tokens_cached_total",
        "vllm:prefix_cache_hits_total",
    ),
    "llamacpp:tokens_predicted_total": (
        "vllm:generation_tokens_total",
        "vllm:inter_token_latency_seconds_count",
    ),
    "llamacpp:tokens_predicted_seconds_total": (
        "vllm:inter_token_latency_seconds_sum",
    ),
    "llamacpp:spec_decode_num_accepted_tokens_total": (
        "vllm:spec_decode_num_accepted_tokens_total",
    ),
    "llamacpp:spec_decode_num_draft_tokens_total": (
        "vllm:spec_decode_num_draft_tokens_total",
    ),
}

# alias -> human label for a llama.cpp engine, resolved once from /props.
# llama.cpp does not label its metrics with a model name the way vLLM
# does, and the loaded file cannot change while the process lives, so
# this is fetched on the first successful scrape and dropped when the
# engine goes unreachable (same discipline as the counter baselines).
_llamacpp_model_name: dict[str, str] = {}


def _translate_llamacpp(
    parsed: dict[str, list[tuple[dict[str, str], float]]],
) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Rename llama.cpp series into the vLLM names the snapshot expects."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for src, targets in _LLAMACPP_TO_VLLM.items():
        series = parsed.get(src)
        if series is None:
            continue
        for target in targets:
            out.setdefault(target, []).extend(series)
    return out


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

    # llama.cpp speaks the same protocol under a different vocabulary.
    # Translate first so everything below is engine-agnostic.
    is_llamacpp = any(k.startswith("llamacpp:") for k in parsed)
    if is_llamacpp:
        parsed = _translate_llamacpp(parsed)

    # Model name comes off any labelled series — every vLLM metric carries
    # it, so we take it from the first one we find rather than requiring a
    # specific metric to be present. llama.cpp labels nothing, so its name
    # comes from the /props probe cached by `_scrape_one`.
    model_name = _llamacpp_model_name.get(alias, "") if is_llamacpp else ""
    for series in parsed.values():
        if model_name:
            break
        for labels, _v in series:
            if labels.get("model_name"):
                model_name = labels["model_name"]
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
        "engine": "llama.cpp" if is_llamacpp else "vllm",
        "model_name": model_name,
        # llama.cpp has no sleep/wake state — a reachable server is awake.
        # Without this it would fail the vLLM gauge check and render as
        # "asleep" forever.
        "awake": (
            True
            if is_llamacpp
            else _first(parsed, "vllm:engine_sleep_state", sleep_state="awake") == 1.0
        ),
        # Live gauges — the "what is it doing right now" row.
        "requests_running": int(running) if running is not None else None,
        "requests_waiting": int(waiting) if waiting is not None else None,
        "requests_waiting_by_reason": {k: int(v) for k, v in waiting_by_reason.items()},
        # llama.cpp exposes no live KV-occupancy gauge, so this is None
        # there. Reporting 0.0 would render a full cache as an empty one.
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
        _llamacpp_model_name.pop(alias, None)
        return {
            "alias": alias,
            "reachable": False,
            "base_url": base_url,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # llama.cpp does not carry a model name on its metrics. Resolve it once
    # per engine lifetime from /props — the loaded GGUF cannot change while
    # the process lives, so this costs one extra local GET per restart, not
    # one per 2-second poll.
    if "llamacpp:" in resp.text and alias not in _llamacpp_model_name:
        _llamacpp_model_name[alias] = await _llamacpp_props_name(
            client, base_url
        )

    snapshot = _snapshot_from_text(alias, resp.text)
    snapshot["base_url"] = base_url
    return snapshot


async def _llamacpp_props_name(client: httpx.AsyncClient, base_url: str) -> str:
    """Loaded GGUF's filename from llama-server /props, or "" if unavailable.

    Best-effort by design: the model name is a label on a dashboard card,
    so a failure here must not cost us the metrics we already scraped.
    """
    try:
        resp = await client.get(
            base_url.rstrip("/") + "/props", timeout=SCRAPE_TIMEOUT_S
        )
        resp.raise_for_status()
        path = (resp.json() or {}).get("model_path") or ""
        return path.rsplit("/", 1)[-1].removesuffix(".gguf")
    except Exception as exc:
        logger.debug("llama.cpp /props probe for %s failed: %s", base_url, exc)
        return ""


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
