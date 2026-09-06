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
    vm._llamacpp_model_name.clear()
    yield
    vm._previous.clear()
    vm._llamacpp_model_name.clear()


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


# ── llama.cpp engines ──────────────────────────────────────────────────
#
# The secondary slot is llama-server, not vLLM (Qwen3.6-35B-A3B only fits
# a 24 GB card as a GGUF Q3). Its /metrics speaks the same protocol with
# different names, and the dashboard renders one card shape for both, so
# the translation has to land in exactly the vLLM snapshot keys.


def _llamacpp_text(*, prompt=1000.0, cached=800.0, predicted=200.0,
                   predicted_s=4.0, processing=1.0, deferred=2.0):
    return f"""
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total {prompt}
llamacpp:prompt_tokens_cached_total {cached}
llamacpp:tokens_predicted_total {predicted}
llamacpp:tokens_predicted_seconds_total {predicted_s}
llamacpp:requests_processing {processing}
llamacpp:requests_deferred {deferred}
llamacpp:n_decode_total 55.0
"""


def test_llamacpp_metrics_map_into_the_vllm_snapshot_shape():
    snap = vm._snapshot_from_text("secondary", _llamacpp_text())
    assert snap["engine"] == "llama.cpp"
    assert snap["reachable"] is True
    assert snap["requests_running"] == 1
    assert snap["requests_waiting"] == 2
    assert snap["prompt_tokens_total"] == 1000.0
    assert snap["generation_tokens_total"] == 200.0


def test_llamacpp_engine_is_awake_when_reachable():
    """llama.cpp has no sleep/wake gauge. Falling through to the vLLM
    check would compare None == 1.0 and render a live engine 'asleep'."""
    assert vm._snapshot_from_text("secondary", _llamacpp_text())["awake"] is True


def test_vllm_engines_still_report_themselves_as_vllm():
    assert vm._snapshot_from_text("primary", _text())["engine"] == "vllm"


def test_llamacpp_reports_no_kv_usage_or_ttft_rather_than_zero():
    """Neither is derivable from llama.cpp's endpoint. Reporting 0.0 would
    draw a saturated cache as an empty one and an unknown TTFT as instant."""
    vm._snapshot_from_text("secondary", _llamacpp_text())
    snap = vm._snapshot_from_text("secondary", _llamacpp_text(prompt=2000.0))
    assert snap["kv_cache_usage"] is None
    assert snap["ttft_s"] is None


def test_llamacpp_inter_token_latency_from_seconds_over_tokens():
    """`tokens_predicted_seconds_total / tokens_predicted_total` is a
    sum/count pair over generated tokens — exactly what ITL means."""
    vm._snapshot_from_text("secondary", _llamacpp_text(predicted=200.0, predicted_s=4.0))
    snap = vm._snapshot_from_text(
        "secondary", _llamacpp_text(predicted=300.0, predicted_s=6.0)
    )
    # 2.0s of decode over 100 new tokens = 20ms/token.
    assert snap["itl_s"] == pytest.approx(0.02)


def test_llamacpp_prefix_cache_hit_rate_from_cached_prompt_tokens():
    snap = vm._snapshot_from_text("secondary", _llamacpp_text(prompt=1000.0, cached=800.0))
    assert snap["prefix_cache_hit_rate"] == pytest.approx(0.8)


def test_llamacpp_throughput_is_a_rate_not_a_since_boot_total():
    vm._snapshot_from_text("secondary", _llamacpp_text(prompt=1000.0))
    vm._previous["secondary"]["_t"] -= 2.0
    snap = vm._snapshot_from_text("secondary", _llamacpp_text(prompt=1400.0))
    assert snap["prompt_tokens_per_s"] == pytest.approx(200.0, rel=0.02)


def test_llamacpp_counter_reset_yields_none_not_a_spike():
    vm._snapshot_from_text("secondary", _llamacpp_text(prompt=100000.0))
    snap = vm._snapshot_from_text("secondary", _llamacpp_text(prompt=10.0))
    assert snap["prompt_tokens_per_s"] is None


@pytest.mark.asyncio
async def test_llamacpp_model_name_probed_once_from_props():
    """The loaded GGUF cannot change while the process lives, so /props is
    worth one GET per engine lifetime — not one per 2-second poll."""
    calls = []

    class _Client:
        async def get(self, url, **_kw):
            calls.append(url)
            if url.endswith("/props"):
                return _Resp(json_body={
                    "model_path": "/x/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
                })
            return _Resp(text=_llamacpp_text())

    client = _Client()
    first = await vm._scrape_one(client, "secondary", "http://127.0.0.1:8091")
    assert first["model_name"] == "Qwen3.6-35B-A3B-UD-Q3_K_XL"

    second = await vm._scrape_one(client, "secondary", "http://127.0.0.1:8091")
    assert second["model_name"] == "Qwen3.6-35B-A3B-UD-Q3_K_XL"
    assert calls.count("http://127.0.0.1:8091/props") == 1


@pytest.mark.asyncio
async def test_llamacpp_props_failure_does_not_lose_the_scrape():
    class _Client:
        async def get(self, url, **_kw):
            if url.endswith("/props"):
                raise RuntimeError("no /props on this build")
            return _Resp(text=_llamacpp_text())

    snap = await vm._scrape_one(_Client(), "secondary", "http://127.0.0.1:8091")
    assert snap["reachable"] is True
    assert snap["model_name"] == ""
    assert snap["requests_running"] == 1


@pytest.mark.asyncio
async def test_unreachable_llamacpp_drops_its_cached_model_name():
    """A restart can load a different GGUF, so the label must not outlive
    the process it described."""
    import httpx

    vm._llamacpp_model_name["secondary"] = "Qwen3.6-35B-A3B-UD-Q3_K_XL"

    class _Failing:
        async def get(self, *_a, **_kw):
            raise httpx.ConnectError("connection refused")

    await vm._scrape_one(_Failing(), "secondary", "http://127.0.0.1:8091")
    assert "secondary" not in vm._llamacpp_model_name


class _Resp:
    def __init__(self, *, text="", json_body=None):
        self.text = text
        self.status_code = 200
        self._json = json_body

    def raise_for_status(self):
        return None

    def json(self):
        return self._json
