"""The primary-engine pressure sampler (`app/engine_pressure.py`).

Three readers share it — the pool's KV gate, the prefix-miss alert and the
dashboard's KV p90 — and all three depend on the same two properties: a
stale or missing reading is None (each reader then fails open), and reading
through it never disturbs the dashboard's own rate baseline.
"""

from __future__ import annotations

import asyncio

import pytest

from app import engine_pressure as ep
from app import vllm_metrics

METRICS = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.425
vllm:prompt_tokens_total{engine="0",model_name="m"} 123456.0
vllm:cache_config_info{_block_size_resolved="True",block_size="3200",cache_dtype="fp8",enable_prefix_caching="True",engine="0",kv_cache_dtype_skip_layers="[]",kv_cache_size_tokens="692263",mamba_page_size_padded="None"} 1.0
"""


@pytest.fixture(autouse=True)
def _clean():
    ep.reset()
    yield
    ep.reset()


def test_gauges_come_out_of_one_metrics_body():
    assert vllm_metrics.gauges_from_text(METRICS) == {
        "kv_cache_usage": 0.425, "requests_running": 2, "requests_waiting": 1}


def test_reading_gauges_leaves_the_dashboards_rate_baseline_alone():
    """The dashboard's reader keeps a per-engine baseline for its rates; a
    second poller through it would halve that window."""
    vllm_metrics._previous.clear()
    vllm_metrics.gauges_from_text(METRICS)
    assert vllm_metrics._previous == {}


def test_the_cache_config_says_what_the_kv_cache_is():
    assert vllm_metrics.cache_config_from_text(METRICS) == {
        "cache_dtype": "fp8", "kv_cache_size_tokens": 692263, "block_size": 3200}


def test_an_engine_that_publishes_no_cache_config_reads_none():
    assert vllm_metrics.cache_config_from_text(
        "llamacpp:requests_processing 1\n") is None


def _s(t: float, kv: float | None = 0.5, running: int | None = 1) -> ep.Sample:
    return ep.Sample(t, t, kv, running, 0)


def test_a_stale_sample_is_no_reading():
    ep.record(_s(100.0), window=1_000)
    assert ep.latest(now=110.0) is not None
    assert ep.latest(now=100.0 + ep.STALE_AFTER_S + 1) is None


def test_the_ring_forgets_what_leaves_the_window():
    ep.record(_s(0.0), window=60)
    ep.record(_s(100.0), window=60)
    assert ep.kv_percentile(1.0, window=1_000, now=100.0) == 0.5
    assert len(ep._samples) == 1


def test_p90_is_nearest_rank_over_the_window():
    for i in range(1, 11):
        ep.record(_s(float(i), kv=i / 10), window=1_000)
    assert ep.kv_percentile(0.9, window=1_000, now=10.0) == pytest.approx(0.9)
    assert ep.kv_percentile(1.0, window=1_000, now=10.0) == pytest.approx(1.0)
    # A window that reaches back only three samples.
    assert ep.kv_percentile(1.0, window=2.5, now=10.0) == pytest.approx(1.0)
    assert ep.kv_percentile(0.0, window=2.5, now=10.0) == pytest.approx(0.8)
    assert ep.kv_percentile(0.9, window=1_000, now=5_000.0) is None


def test_neighbours_subtract_the_request_that_is_asking():
    for t, running in [(10.0, 1), (15.0, 3), (20.0, 2), (40.0, 7)]:
        ep.record(_s(t, running=running), window=1_000)
    assert ep.neighbours_during(9.0, now=21.0) == 2      # peak 3, minus itself
    assert ep.neighbours_during(21.0, now=30.0) is None  # no sample inside
    assert ep.neighbours_during(0.0, now=12.0) == 0


def test_the_snapshot_says_stale_rather_than_zero():
    snap = ep.snapshot(now=1_000.0)
    assert snap["stale"] is True
    assert snap["kv_now"] is None and snap["kv_p90"] is None
    assert snap["warn_line"] == pytest.approx(ep.DEFAULT_KV_WARN_LINE)


def test_a_scrape_of_a_dead_port_is_none_and_never_raises():
    assert asyncio.run(ep.scrape_once("http://127.0.0.1:9")) is None
    assert ep.snapshot()["error"]
