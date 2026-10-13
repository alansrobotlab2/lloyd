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

from app import engine_pressure, usage_store, vllm_metrics
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


def _build_client(monkeypatch, keep: tuple[str, ...] = ()):
    """A client for `/api/dashboard` with every section stubbed but `keep`.

    The other sections read live state — the sessions directory, supervisord,
    the worker queue — so the route is exercised alone. `_usage` is left real by
    the one node that needs it and reads whatever `usage_store.DB_PATH` the test
    moved.
    """
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        vllm_metrics.httpx, "AsyncClient",
        lambda *a, **kw: real_client(transport=httpx.MockTransport(_handler)))
    monkeypatch.setattr(vllm_metrics, "configured_engines", lambda: dict(ENGINES))
    monkeypatch.setattr(vllm_metrics, "_previous", {})
    monkeypatch.setattr(vllm_metrics, "_llamacpp_model_name", {})
    monkeypatch.setattr(engine_pressure, "snapshot", lambda: {"alias": "primary"})
    for name in ("_network", "_recent_sessions", "_services", "_workers",
                 "_autonomy", "_backlog", "_automod", "_usage"):
        if name not in keep:
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


@pytest.fixture
def client(monkeypatch):
    return _build_client(monkeypatch)


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


#: Real bench-family session ids (#2663): the trial runner's `_bench_` and the
#: bench-mining worker's `_benchmine_`, beside a minted web-chat id.
_BENCH_ID = "20260924_100311_bench_67d5"
_BENCH_MINE_ID = "20260922_183424_benchmine_6d65"
_CHAT_ID = "20260924_100311_a1b2c3"


@pytest.fixture
def usage_client(monkeypatch):
    """`/api/dashboard` with its `_usage` section reading a real store."""
    return _build_client(monkeypatch, keep=("_usage",))


def test_the_stop_reason_block_publishes_the_split_under_its_existing_key(
        usage_client, tmp_path, monkeypatch):
    """`stop_reasons_24h` keeps the shape `StopReasonStrip` draws, and now answers
    the production question on the same row (#2663 clause 5).

    This is the seam the conflation actually reached a human through:
    `usage_store.stop_reason_breakdown` → `_usage()` → this key →
    `DashboardPage.tsx`'s `<StopReasonStrip rows={usage.stop_reasons_24h} />`.
    One row per `stop_reason` is what keeps the strip from duplicating an entry
    per kind, and `turns` is the bar's denominator — so the split has to ride
    beside those keys, not replace them.
    """
    monkeypatch.setattr(usage_store, "DB_PATH", tmp_path / "usage.db")
    for sid, wrapped in ((_BENCH_ID, 1), (_BENCH_MINE_ID, 0), (_CHAT_ID, 1)):
        usage_store.record_usage(session_id=sid, model="primary",
                                 input_tokens=10, output_tokens=5,
                                 stop_reason="max_turns", wrapped_up=wrapped)

    resp = usage_client.get("/api/dashboard")
    assert resp.status_code == 200
    rows = resp.json()["usage"]["stop_reasons_24h"]
    assert isinstance(rows, list) and rows, "an empty block renders no strip at all"
    assert [row["stop_reason"] for row in rows] == ["max_turns"]
    for row in rows:
        assert {"stop_reason", "turns", "wrapped_up"} <= set(row), row
    assert (rows[0]["turns"], rows[0]["wrapped_up"]) == (3, 2)
    # The number the item could not get before, straight off the published row.
    assert (rows[0]["bench_turns"], rows[0]["production_turns"]) == (2, 1)
    assert (rows[0]["bench_wrapped_up"],
            rows[0]["production_wrapped_up"]) == (1, 1)
