"""`GET /api/dashboard` carries each engine's prefix-cache hit rate as a fraction.

`DashboardPage.tsx` renders `prefix_cache_hit_rate` through `pct()`, which
multiplies by 100, so the route has to hand it the fraction `vllm_metrics`
computed — 0.929 for a 92.9 % hit. A percentage (92.9) would render as
9290 %, and llama.cpp's cached-over-processed quotient (929 / 71 = 13.1) as
1310 %, which is #1083's "961%" card. The unit tests in
`tests/test_vllm_metrics.py` pin the ratio; this pins that the route passes
it through unchanged, from a scraped /metrics body to the JSON the card reads.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import engine_pressure, vllm_metrics
from app.routers import dashboard as dash

# 929 of 1000 prompt tokens served from cache on both engines.
VLLM_BODY = """\
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen3.8-Flash-Next"} 929.0
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen3.8-Flash-Next"} 1000.0
vllm:prompt_tokens_total{engine="0",model_name="Qwen3.8-Flash-Next"} 1000.0
vllm:num_requests_running{engine="0",model_name="Qwen3.8-Flash-Next"} 0.0
"""

# llama.cpp's prompt_tokens_total EXCLUDES cached tokens: 71 processed + 929
# cached is the same 1000-token total, and 929/71 is the 13.1 the card must
# never see.
LLAMACPP_BODY = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed, excluding cached tokens.
llamacpp:prompt_tokens_total 71
llamacpp:prompt_tokens_cached_total 929
llamacpp:requests_processing 0
"""

ENGINES = {"primary": "http://primary.test", "secondary": "http://llama.test"}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/props":
        return httpx.Response(200, json={"model_path": "/m/Qwen3.6-35B.gguf"})
    body = VLLM_BODY if request.url.host == "primary.test" else LLAMACPP_BODY
    return httpx.Response(200, text=body)


@pytest.fixture
def client(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        vllm_metrics.httpx, "AsyncClient",
        lambda *a, **kw: real_client(transport=httpx.MockTransport(_handler)))
    monkeypatch.setattr(vllm_metrics, "configured_engines", lambda: dict(ENGINES))
    monkeypatch.setattr(vllm_metrics, "_previous", {})
    monkeypatch.setattr(vllm_metrics, "_llamacpp_model_name", {})
    monkeypatch.setattr(engine_pressure, "snapshot", lambda: {"alias": "primary"})
    # Every other section is irrelevant here and some read live state; each
    # is replaced with an empty answer so the route is exercised alone.
    for name in ("_network", "_recent_sessions", "_services", "_workers",
                 "_autonomy", "_backlog", "_automod", "_usage"):
        monkeypatch.setattr(dash, name, lambda: {})

    async def _empty():
        return {}
    for name in ("_primary_state", "_agent_state"):
        monkeypatch.setattr(dash, name, _empty)

    async def _host():
        return {}
    monkeypatch.setattr(dash.host_metrics, "collect", _host)

    app = FastAPI()
    app.include_router(dash.router)
    return TestClient(app)


def test_each_engine_row_carries_the_hit_rate_as_a_fraction(client):
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    rows = {row["alias"]: row for row in resp.json()["vllm"]}
    assert set(rows) == set(ENGINES)
    for alias, row in rows.items():
        assert row["reachable"] is True, row
        rate = row["prefix_cache_hit_rate"]
        assert rate == pytest.approx(0.929), (alias, rate)
        # The card multiplies by 100; anything above 1 renders over 100 %.
        assert 0.0 <= rate <= 1.0
    assert rows["secondary"]["engine"] == "llama.cpp"
    assert rows["primary"]["engine"] == "vllm"
