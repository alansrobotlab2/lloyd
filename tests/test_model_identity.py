"""A slot must serve the model config says it serves.

The 2026-09-06 near-miss: a `git reset` from the autoimplement/guardian
machinery reverted `agent-llm-secondary.conf` to `MODEL=qwen35`, a
launcher branch that serves Qwen3.5-**4B** on vLLM under the same alias
and the same port as the 35B. The running process was never restarted so
the 35B survived, but any restart in that window would have answered to
`secondary` with a model an order of magnitude smaller — and every
caller would have gone on logging `model: secondary` as though nothing
had happened.

`server.py::_sync_secondary_llm_state` only ever decided whether the
secondary should be running, never what it is.
"""

from __future__ import annotations

import pytest

from app import model_identity


@pytest.fixture(autouse=True)
def _clear_cache():
    model_identity.LAST_RESULT.clear()
    yield
    model_identity.LAST_RESULT.clear()


def _configs(monkeypatch, cfg: dict) -> None:
    monkeypatch.setattr("app.config.MODEL_CONFIGS", cfg, raising=False)


def _served(monkeypatch, mapping: dict[str, str]) -> None:
    async def _probe(base_url, *, timeout=5.0):
        return mapping.get(base_url, "")

    monkeypatch.setattr(model_identity, "probe_served_model", _probe)


@pytest.mark.asyncio
async def test_matching_model_is_ok(monkeypatch):
    _configs(monkeypatch, {"secondary": {
        "base_url": "http://x:8091", "expect_model": "Qwen3.6-35B-A3B",
    }})
    _served(monkeypatch, {
        "http://x:8091": "/models/unsloth-Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
    })
    rows = await model_identity.verify_models()
    assert rows[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_the_reverted_launcher_is_a_mismatch(monkeypatch):
    """The exact drift: alias `secondary` on :8091, but it's the 4B."""
    _configs(monkeypatch, {"secondary": {
        "base_url": "http://x:8091", "expect_model": "Qwen3.6-35B-A3B",
    }})
    _served(monkeypatch, {"http://x:8091": "/models/Qwen-Qwen3.5-4B secondary"})
    rows = await model_identity.verify_models()
    assert rows[0]["status"] == "MISMATCH"
    # The served string must survive into the row — a mismatch that
    # doesn't say what it found sends you back to the same investigation.
    assert "Qwen3.5-4B" in rows[0]["served"]


@pytest.mark.asyncio
async def test_match_is_case_insensitive(monkeypatch):
    _configs(monkeypatch, {"p": {
        "base_url": "http://x:8096", "expect_model": "qwen3.8-flash-next",
    }})
    _served(monkeypatch, {"http://x:8096": "/models/Inferact-Qwen3.8-Flash-Next-NVFP4"})
    rows = await model_identity.verify_models()
    assert rows[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_stopped_engine_is_unreachable_not_a_mismatch(monkeypatch):
    """A stopped secondary is a normal state, not a config error."""
    _configs(monkeypatch, {"secondary": {
        "base_url": "http://x:8091", "expect_model": "Qwen3.6-35B-A3B",
    }})
    _served(monkeypatch, {})
    rows = await model_identity.verify_models()
    assert rows[0]["status"] == "unreachable"


@pytest.mark.asyncio
async def test_slot_without_expect_model_is_unchecked(monkeypatch):
    """Opt-in: an undeclared slot must not manufacture an alarm."""
    _configs(monkeypatch, {"third": {"base_url": "http://x:9000"}})
    _served(monkeypatch, {"http://x:9000": "whatever"})
    rows = await model_identity.verify_models()
    assert rows[0]["status"] == "unchecked"


@pytest.mark.asyncio
async def test_result_is_cached_for_the_endpoint(monkeypatch):
    _configs(monkeypatch, {"secondary": {
        "base_url": "http://x:8091", "expect_model": "Qwen3.6-35B-A3B",
    }})
    _served(monkeypatch, {"http://x:8091": "Qwen3.6-35B-A3B"})
    await model_identity.verify_models()
    assert model_identity.LAST_RESULT["secondary"]["status"] == "ok"


@pytest.mark.asyncio
async def test_retry_waits_out_a_loading_engine(monkeypatch):
    """The 35B pages 17 GB onto a 3090; one probe at boot proves nothing."""
    calls = {"n": 0}

    _configs(monkeypatch, {"secondary": {
        "base_url": "http://x:8091", "expect_model": "Qwen3.6-35B-A3B",
    }})

    async def _probe(base_url, *, timeout=5.0):
        calls["n"] += 1
        return "" if calls["n"] < 3 else "Qwen3.6-35B-A3B"

    monkeypatch.setattr(model_identity, "probe_served_model", _probe)
    rows = await model_identity.verify_models_with_retry(
        attempts=5, delay_seconds=0.0
    )
    assert rows[0]["status"] == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_does_not_wait_out_a_mismatch(monkeypatch):
    """A mismatch is conclusive on the first look; don't delay the alarm."""
    calls = {"n": 0}

    _configs(monkeypatch, {"secondary": {
        "base_url": "http://x:8091", "expect_model": "Qwen3.6-35B-A3B",
    }})

    async def _probe(base_url, *, timeout=5.0):
        calls["n"] += 1
        return "Qwen-Qwen3.5-4B"

    monkeypatch.setattr(model_identity, "probe_served_model", _probe)
    rows = await model_identity.verify_models_with_retry(
        attempts=5, delay_seconds=0.0
    )
    assert rows[0]["status"] == "MISMATCH"
    assert calls["n"] == 1


def test_live_config_declares_both_slots():
    """Without `expect_model` the check is inert for that slot."""
    from app.config import CONFIG

    models = CONFIG.get("models") or {}
    for alias in ("primary", "secondary"):
        assert (models.get(alias) or {}).get("expect_model"), (
            f"models.{alias}.expect_model is missing; identity drift there "
            f"would go unnoticed"
        )
