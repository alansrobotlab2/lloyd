"""Harness telemetry (review 2026-09-24, P11), driven through the real loop.

TTFT, per-request time and the cache ratio ride on `assistant_message`; a tool
result carries its MCP wall time and, when it failed, a class; an overflow
recovery is one countable `harness.overflow_recovered`. Each is asserted on
what `run_query` actually yields over the replay seams, not on its source.
"""
from __future__ import annotations

import asyncio
import time

from app.harness import loop as L
from app.harness import telemetry
from app.harness.errors import ContextOverflowError
from app.harness.options import RunOptions
from app.harness.tests import _replay as R


def _opts(**kw):
    kw.setdefault("max_turns", 6)
    kw.setdefault("tool_search_enabled", False)
    return RunOptions(model="m", **kw)


class _SlowFirstChunk(R.ReplayEngine):
    """Sleeps before the first chunk of each request: a prefill."""

    def __init__(self, steps, delay_s: float):
        super().__init__(steps)
        self.delay_s = delay_s

    async def _gen(self, step):
        await asyncio.sleep(self.delay_s)
        async for chunk in super()._gen(step):
            yield chunk


class _Silent(R.ReplayEngine):
    """A request that ends without a single chunk."""

    async def _gen(self, step):
        return
        yield  # pragma: no cover - makes this an async generator


async def test_ttft_is_measured_from_the_request_not_from_relief(monkeypatch):
    """A slow pre-request relief pass lands in the iteration's duration and
    never in its TTFT, which starts when the request goes out."""
    relief_s = 0.3
    prefill_s = 0.05

    def _slow_relief(*_a, **_kw):
        time.sleep(relief_s)
        return {"freed_tokens": 0, "rungs": []}

    monkeypatch.setattr(L, "_relieve_context", _slow_relief)
    engine = _SlowFirstChunk([
        R.Step(tool_calls=[R.tool_call("c1", "Read", "a")]),
        R.Step(text="done"),
    ], delay_s=prefill_s)
    R.install(monkeypatch, engine, R.ReplayPool())
    # Any measured headroom is below this, so iteration 2 relieves first.
    out = await R.drive(_opts(context_relief_min_completion_tokens=10**9))

    second = R.of_type(out, "assistant_message")[1]
    assert second["duration_ms"] >= relief_s * 1000
    assert second["ttft_ms"] is not None
    assert prefill_s * 1000 * 0.8 <= second["ttft_ms"] < relief_s * 1000
    assert second["request_ms"] >= second["ttft_ms"]
    assert second["request_ms"] < relief_s * 1000


async def test_ttft_is_none_when_no_chunk_arrived(monkeypatch):
    R.install(monkeypatch, _Silent([R.Step(text="x")]), R.ReplayPool())
    out = await R.drive(_opts())
    asst = R.of_type(out, "assistant_message")[0]
    assert asst["ttft_ms"] is None
    assert asst["request_ms"] is not None
    # No usage chunk either: no prompt size, so no ratio.
    assert asst["cache_ratio"] is None


async def test_cache_ratio_is_cache_read_over_input(monkeypatch):
    engine = R.ReplayEngine([R.Step(text="done", usage={
        "prompt_tokens": 1000, "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 750},
    })])
    R.install(monkeypatch, engine, R.ReplayPool())
    out = await R.drive(_opts())
    assert R.of_type(out, "assistant_message")[0]["cache_ratio"] == 0.75


async def test_tool_result_duration_covers_the_mcp_call(monkeypatch):
    engine = R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Read", "a"),
                           R.tool_call("c2", "Bash", "b", command="x")]),
        R.Step(text="done"),
    ])
    pool = R.ReplayPool(answers={"c2": {"content": "MCP error calling Bash: nope",
                                        "is_error": True}},
                        delay_by_call_id={"c1": 0.15})
    R.install(monkeypatch, engine, pool)
    out = await R.drive(_opts())
    results = {e["call_id"]: e for e in R.of_type(out, "tool_result")}
    assert results["c1"]["duration_ms"] >= 150 * 0.8
    assert "error_class" not in results["c1"]
    # No `timing` from this pool: the handshake is absent, not zero.
    assert "handshake_ms" not in results["c1"]
    assert results["c2"]["error_class"] == "mcp_error"
    assert "duration_ms" in results["c2"]


async def test_handshake_is_lifted_from_the_pools_timing(monkeypatch):
    engine = R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Read", "a")]),
        R.Step(text="done"),
    ])
    pool = R.ReplayPool(answers={"c1": {"content": "ok", "is_error": False,
                                        "timing": {"handshake_ms": 12}}})
    R.install(monkeypatch, engine, pool)
    out = await R.drive(_opts())
    assert R.of_type(out, "tool_result")[0]["handshake_ms"] == 12


async def test_overflow_recovery_emits_one_event(monkeypatch):
    overflow = ContextOverflowError("prompt too long", requested_input_tokens=300_000)
    engine = R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Read", "a")]),
        R.Step(raise_=overflow),
        R.Step(text="done"),
    ])
    R.install(monkeypatch, engine, R.ReplayPool())
    logged: list[tuple[str, dict]] = []
    real = telemetry.log_harness_event

    def _spy(session_id, event, data, *, turn_id=None):
        logged.append((event, data))
        real(session_id, event, data, turn_id=turn_id)

    monkeypatch.setattr(L, "_log_harness_event", _spy)
    counts = telemetry.bind_event_counts()
    out = await R.drive(_opts())

    recovered = [d for e, d in logged if e == "harness.overflow_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["attempt"] == 1
    assert recovered[0]["requested_input_tokens"] == 300_000
    assert counts.get("harness.overflow_recovered") == 1
    assert R.of_type(out, "result")[0]["stop_reason"] == "stop"
