"""The primary must serve its KV cache the way the stall fix left it.

The 2026-09-10 cutover to FP8 KV — a 692,263-token pool where BF16 held
398,175, with 3200-token pages — is what fixed the 09-09 stall. A boot that
lands back on BF16 (a reverted conf, a dropped KV_CACHE_DTYPE, the old venv)
answers every request correctly, so the identity check alone would call it
healthy. `expect_kv_cache_dtype` / `expect_kv_pool_tokens_min` read the
engine's own `vllm:cache_config_info` and make that boot a REGRESSION.
"""

from __future__ import annotations

import logging

import pytest

from app import model_identity as mi

EXPECT = {"expect_kv_cache_dtype": "fp8", "expect_kv_pool_tokens_min": 600_000}
FP8 = {"cache_dtype": "fp8", "kv_cache_size_tokens": 692_263, "block_size": 3200}
BF16 = {"cache_dtype": "auto", "kv_cache_size_tokens": 398_175, "block_size": 1600}


@pytest.fixture(autouse=True)
def _clear_cache():
    mi.LAST_RESULT.clear()
    yield
    mi.LAST_RESULT.clear()


def test_the_production_cache_is_ok():
    status, detail = mi.judge_kv(EXPECT, FP8, True)
    assert status == "ok"
    assert "692,263" in detail


def test_the_bf16_boot_is_a_regression_on_both_counts():
    status, detail = mi.judge_kv(EXPECT, BF16, True)
    assert status == "REGRESSION"
    assert "cache_dtype=auto" in detail and "398,175" in detail


def test_a_named_fp8_variant_counts_as_fp8():
    assert mi.judge_kv(EXPECT, {**FP8, "cache_dtype": "fp8_e4m3"}, True)[0] == "ok"


def test_nothing_declared_is_unchecked():
    assert mi.judge_kv({}, BF16, True) == ("unchecked", "")


def test_down_and_silent_are_different_answers():
    assert mi.judge_kv(EXPECT, None, False)[0] == "unreachable"
    assert mi.judge_kv(EXPECT, None, True)[0] == "unknown"


@pytest.mark.asyncio
async def test_the_boot_sweep_logs_a_regression_at_error(monkeypatch, caplog):
    monkeypatch.setattr("app.config.MODEL_CONFIGS", {"primary": {
        "base_url": "http://x:8096", "expect_model": "Qwen3.8-Flash-Next",
        **EXPECT,
    }}, raising=False)

    async def _served(base_url, *, timeout=5.0):
        return "/models/Inferact-Qwen3.8-Flash-Next-NVFP4 primary"

    async def _kv(base_url, *, timeout=5.0):
        return BF16

    monkeypatch.setattr(mi, "probe_served_model", _served)
    monkeypatch.setattr(mi, "probe_kv_config", _kv)
    with caplog.at_level(logging.INFO, logger="lloyd-server"):
        rows = await mi.verify_models_with_retry(attempts=1)
    assert rows[0]["status"] == "ok"                 # the right model...
    assert rows[0]["kv_status"] == "REGRESSION"      # ...served the old way
    assert any(r.levelno == logging.ERROR and "KV cache REGRESSION" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_a_slot_without_expectations_is_not_probed(monkeypatch):
    monkeypatch.setattr("app.config.MODEL_CONFIGS", {"secondary": {
        "base_url": "http://x:8091", "expect_model": "Qwen3.6-35B-A3B"}},
        raising=False)

    async def _served(base_url, *, timeout=5.0):
        return "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"

    async def _kv(base_url, *, timeout=5.0):
        raise AssertionError("probed a slot that declared nothing")

    monkeypatch.setattr(mi, "probe_served_model", _served)
    monkeypatch.setattr(mi, "probe_kv_config", _kv)
    rows = await mi.verify_models()
    assert rows[0]["kv_status"] == "unchecked"
