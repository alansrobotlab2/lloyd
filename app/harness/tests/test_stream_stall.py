"""`harness.stream_chunk_timeout_seconds` bounds mid-generation silence.

The key `stream_chunk_timeout_seconds: 60` sat in config.yaml since the
harness was written and was read by nothing — a repo-wide grep found the
config line and no consumer. Meanwhile `client.stream_chat` builds
`httpx.Timeout(timeout_s, read=None, ...)`, so the read is unbounded and
a wedged engine mid-generation blocks the turn until the client gives up.

The deadline applies only BETWEEN lines once data has started flowing.
Time-to-first-line must stay unbounded: prefill emits no bytes at all,
and the secondary slot runs llama.cpp with `--parallel 1`, so a queued
request legitimately sits silent for as long as the one ahead of it.
Bounding that would kill healthy turns on a busy slot — the opposite of
the fix.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.harness.client import stream_chat
from app.harness.errors import StreamStalledError


class _FakeResponse:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(self, lines: list[tuple[float, str]], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code
        self.request = None

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for delay, line in self._lines:
            if delay:
                await asyncio.sleep(delay)
            yield line


class _FakeStream:
    def __init__(self, resp: _FakeResponse):
        self._resp = resp

    async def __aenter__(self) -> _FakeResponse:
        return self._resp

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeClient:
    def __init__(self, resp: _FakeResponse):
        self._resp = resp

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def stream(self, *_a, **_kw) -> _FakeStream:
        return _FakeStream(self._resp)


def _install(monkeypatch, lines: list[tuple[float, str]]) -> None:
    resp = _FakeResponse(lines)
    monkeypatch.setattr(
        "app.harness.client.httpx.AsyncClient", lambda **_kw: _FakeClient(resp)
    )


def _chunk(text: str) -> str:
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": text}, "index": 0}]}
    )


async def _drain(**kw) -> list[dict[str, Any]]:
    return [c async for c in stream_chat(
        base_url="http://127.0.0.1:9",
        model="secondary",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        extra_body=None,
        cancel_event=None,
        timeout_s=30.0,
        **kw,
    )]


def test_gap_after_first_line_raises(monkeypatch):
    """Engine emits one chunk, then goes quiet past the deadline."""
    _install(monkeypatch, [
        (0.0, _chunk("hello")),
        (0.30, _chunk(" world")),
    ])
    with pytest.raises(StreamStalledError) as exc:
        asyncio.run(_drain(chunk_timeout_s=0.05))
    # The forensic detail matters: "stalled after 1 line" distinguishes a
    # wedged engine from one that never started.
    assert exc.value.lines_seen >= 1
    assert exc.value.timeout_s == 0.05


def test_slow_first_line_is_not_a_stall(monkeypatch):
    """A long silence BEFORE any data is prefill or queueing, not a wedge.

    This is the case that makes the secondary usable: `--parallel 1`
    serialises, so an agent turn queued behind a post-session capture
    sees nothing for as long as that job runs.
    """
    _install(monkeypatch, [
        (0.30, _chunk("late")),
        (0.0, "data: [DONE]"),
    ])
    chunks = asyncio.run(_drain(chunk_timeout_s=0.05))
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"]["content"] == "late"


def test_zero_timeout_disables_the_check(monkeypatch):
    """0 means "no deadline" — the pre-existing unbounded behaviour."""
    _install(monkeypatch, [
        (0.0, _chunk("a")),
        (0.20, _chunk("b")),
        (0.0, "data: [DONE]"),
    ])
    chunks = asyncio.run(_drain(chunk_timeout_s=0.0))
    assert len(chunks) == 2


def test_default_is_disabled(monkeypatch):
    """Callers that never pass the kwarg keep the old semantics."""
    _install(monkeypatch, [
        (0.0, _chunk("a")),
        (0.20, _chunk("b")),
        (0.0, "data: [DONE]"),
    ])
    chunks = asyncio.run(_drain())
    assert len(chunks) == 2


def test_config_key_reaches_run_options():
    """The config key is actually plumbed, which was the whole defect."""
    from app.harness.options import RunOptions
    from app.mcp_discovery import _get_harness_kwargs

    kwargs = _get_harness_kwargs()
    assert "stream_chunk_timeout_s" in kwargs, (
        "harness.stream_chunk_timeout_seconds is not being read from config"
    )
    opts = RunOptions(model="primary", **kwargs)
    assert opts.stream_chunk_timeout_s > 0
