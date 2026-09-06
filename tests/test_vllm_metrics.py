"""vLLM telemetry parsing and rate derivation.

The scraping is trivial; the delta math is not, and it is the part that
silently lies when wrong. A counter reset rendered as a rate turns an
engine's entire boot history into a one-second spike, and a rate computed
against a stale pre-outage sample does the same. Both are pinned here.
"""

from __future__ import annotations

import pytest

from app import vllm_metrics as vm


@pytest.fixture(autouse=True)
def _clear_baselines():
    """Each test starts with no previous sample — the module cache is
    process-global and would otherwise leak between tests."""
    vm._previous.clear()
    yield
    vm._previous.clear()


# ── Parsing ────────────────────────────────────────────────────────────


def test_parses_labels_values_and_skips_comments():
    text = """
# HELP vllm:num_requests_running Number of requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen"} 0.42
bare_metric 7
"""
    parsed = vm.parse_prometheus(text)
    assert vm._first(parsed, "vllm:num_requests_running", engine="0") == 3.0
    assert parsed["vllm:kv_cache_usage_perc"][0][0]["model_name"] == "Qwen"
    assert parsed["bare_metric"] == [({}, 7.0)]


def test_label_values_may_contain_commas_and_spaces():
    """The reason label is prose in vLLM 0.28 — splitting naively on ','
    would shear it in half and lose the series."""
    text = (
        'vllm:num_requests_waiting_by_reason'
        '{engine="0",reason="waiting for capacity, deferred",model_name="Q"} 2.0\n'
    )
    parsed = vm.parse_prometheus(text)
    labels, value = parsed["vllm:num_requests_waiting_by_reason"][0]
    assert labels["reason"] == "waiting for capacity, deferred"
    assert labels["model_name"] == "Q"
    assert value == 2.0


def test_malformed_lines_are_skipped_not_raised():
    text = "good_metric 1.0\nthis line has no value\nanother{ 2.0\nalso_good 2.0\n"
    parsed = vm.parse_prometheus(text)
    assert parsed["good_metric"] == [({}, 1.0)]
    assert parsed["also_good"] == [({}, 2.0)]


def test_sum_all_adds_series_across_engine_shards():
    text = (
        'm{engine="0"} 10\n'
        'm{engine="1"} 5\n'
    )
    assert vm._sum_all(vm.parse_prometheus(text), "m") == 15.0


# ── Rate derivation ────────────────────────────────────────────────────


def _sample(t: float, **counters: float) -> dict[str, float]:
    return {"_t": t, **counters}


def test_rate_is_none_without_a_baseline():
    """First scrape after boot has nothing to diff against. Reporting the
    raw counter as a rate would claim 437M tokens in one second."""
    curr = _sample(100.0, **{"vllm:prompt_tokens_total": 1000.0})
    assert vm._rate(None, curr, "vllm:prompt_tokens_total") is None


def test_rate_divides_delta_by_elapsed_time():
    prev = _sample(100.0, **{"vllm:generation_tokens_total": 1000.0})
    curr = _sample(102.0, **{"vllm:generation_tokens_total": 1600.0})
    assert vm._rate(prev, curr, "vllm:generation_tokens_total") == 300.0


def test_rate_is_none_when_a_counter_goes_backwards():
    """An engine restart resets counters to zero. The delta is negative;
    anything else here would render the reset as a throughput spike."""
    prev = _sample(100.0, **{"vllm:prompt_tokens_total": 5_000_000.0})
    curr = _sample(102.0, **{"vllm:prompt_tokens_total": 12.0})
    assert vm._rate(prev, curr, "vllm:prompt_tokens_total") is None


def test_rate_is_none_when_no_time_elapsed():
    prev = _sample(100.0, **{"m": 1.0})
    curr = _sample(100.0, **{"m": 5.0})
    assert vm._rate(prev, curr, "m") is None


# ── Histograms ─────────────────────────────────────────────────────────


def test_windowed_mean_uses_the_interval_not_the_lifetime_average():
    """Since-boot mean TTFT is a number that barely moves. The operator
    wants the mean over the last poll, which is a different figure."""
    prev = _sample(100.0, **{
        "vllm:time_to_first_token_seconds_sum": 1000.0,
        "vllm:time_to_first_token_seconds_count": 1000.0,   # lifetime mean 1.0s
    })
    curr = _sample(102.0, **{
        "vllm:time_to_first_token_seconds_sum": 1010.0,
        "vllm:time_to_first_token_seconds_count": 1005.0,   # 10s over 5 reqs
    })
    assert vm._windowed_mean(prev, curr, "vllm:time_to_first_token_seconds") == 2.0


def test_windowed_mean_is_none_when_no_requests_completed():
    """Zero new observations is not "0 seconds latency" — it is no data."""
    prev = _sample(100.0, **{"h_sum": 10.0, "h_count": 5.0})
    curr = _sample(102.0, **{"h_sum": 10.0, "h_count": 5.0})
    assert vm._windowed_mean(prev, curr, "h") is None


def test_ratio_guards_against_a_zero_denominator():
    assert vm._ratio(5.0, 0.0) is None
    assert vm._ratio(None, 10.0) is None
    assert vm._ratio(5.0, 10.0) == 0.5


# ── Snapshot assembly ──────────────────────────────────────────────────


_METRICS = """
vllm:num_requests_running{{engine="0",model_name="Qwen"}} {running}
vllm:num_requests_waiting{{engine="0",model_name="Qwen"}} 0.0
vllm:engine_sleep_state{{engine="0",model_name="Qwen",sleep_state="awake"}} 1.0
vllm:kv_cache_usage_perc{{engine="0",model_name="Qwen"}} 0.25
vllm:prompt_tokens_total{{engine="0",model_name="Qwen"}} {prompt}
vllm:generation_tokens_total{{engine="0",model_name="Qwen"}} {gen}
vllm:prefix_cache_queries_total{{engine="0",model_name="Qwen"}} 1000.0
vllm:prefix_cache_hits_total{{engine="0",model_name="Qwen"}} 800.0
"""


def _text(running=1.0, prompt=1000.0, gen=500.0) -> str:
    return _METRICS.format(running=running, prompt=prompt, gen=gen)


def test_first_snapshot_reports_gauges_but_no_rates():
    snap = vm._snapshot_from_text("primary", _text())
    assert snap["reachable"] is True
    assert snap["model_name"] == "Qwen"
    assert snap["awake"] is True
    assert snap["requests_running"] == 1
    assert snap["kv_cache_usage"] == 0.25
    # Lifetime ratio is available immediately; the rates are not.
    assert snap["prefix_cache_hit_rate"] == 0.8
    assert snap["prompt_tokens_per_s"] is None
    assert snap["generation_tokens_per_s"] is None
    assert snap["prefix_cache_hit_rate_recent"] is None


def test_second_snapshot_derives_rates_from_the_first():
    vm._snapshot_from_text("primary", _text(prompt=1000.0, gen=500.0))
    # Force a known elapsed window rather than sleeping.
    vm._previous["primary"]["_t"] -= 2.0
    snap = vm._snapshot_from_text("primary", _text(prompt=1200.0, gen=600.0))
    assert snap["prompt_tokens_per_s"] == pytest.approx(100.0, rel=0.02)
    assert snap["generation_tokens_per_s"] == pytest.approx(50.0, rel=0.02)


def test_each_engine_keeps_its_own_baseline():
    """Two engines share the module cache; a shared key would compute the
    secondary's rate against the primary's counters."""
    vm._snapshot_from_text("primary", _text(prompt=1000.0))
    vm._snapshot_from_text("secondary", _text(prompt=50.0))
    assert vm._previous["primary"]["vllm:prompt_tokens_total"] == 1000.0
    assert vm._previous["secondary"]["vllm:prompt_tokens_total"] == 50.0


@pytest.mark.asyncio
async def test_unreachable_engine_degrades_and_drops_its_baseline():
    """An offline engine is a normal state (the secondary is stopped
    whenever secondary_enabled is false), and its stale baseline must go:
    when it returns, its counters have reset."""
    import httpx

    vm._snapshot_from_text("primary", _text())
    assert "primary" in vm._previous

    class _Failing:
        async def get(self, *_a, **_kw):
            raise httpx.ConnectError("connection refused")

    result = await vm._scrape_one(_Failing(), "primary", "http://127.0.0.1:9999")
    assert result["reachable"] is False
    assert "ConnectError" in result["error"]
    assert "primary" not in vm._previous


@pytest.mark.asyncio
async def test_collect_returns_empty_for_no_configured_engines():
    assert await vm.collect({}) == []
