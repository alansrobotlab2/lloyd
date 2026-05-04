"""Integration test: ToolSearch progressive-disclosure flow inside run_query.

Asserts the end-to-end loop behavior without touching live vLLM or MCP:
  - Disabled mode sends the full catalog (regression guard for today's behavior).
  - Enabled mode first turn sends only baseline + ToolSearch.
  - When the model calls ToolSearch, dispatch is intercepted locally
    (no MCP round-trip), the matched tools are marked loaded, and the
    next turn's ``tools=`` includes them.
  - Catalog reminder system message is injected exactly once.
  - Defensive intercept guides the model back when it calls an
    unloaded tool that isn't in the visible set.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.harness import tool_search_cache
from app.harness.loop import run_query
from app.harness.options import RunOptions
from app.harness.tool_search import TOOLSEARCH_TOOL_NAME


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePool:
    """Mimics MCPPool's ``discovered`` and ``call_tool`` surface."""

    def __init__(self, server_name: str, tools: list[dict]):
        self._discovered = [(server_name, tools)]
        self.call_log: list[tuple[str, dict]] = []

    @property
    def discovered(self):
        return self._discovered

    async def call_tool(self, name: str, args: dict):
        self.call_log.append((name, args))
        return {"content": f"FAKE_RESULT[{name}]", "is_error": False}


def _mcp_tool(name: str, description: str = "test tool") -> dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": {}},
    }


class _StreamScript:
    """Replays a scripted sequence of vLLM responses, one per turn.

    Each entry is a (text, tool_calls) tuple. ``tool_calls`` is a list of
    ``{id, name, arguments}`` dicts; finish_reason is auto-set to
    ``tool_calls`` when non-empty, ``stop`` otherwise.
    """

    def __init__(self, turns: list[tuple[str, list[dict[str, Any]]]]):
        self.turns = turns
        self.captured_tools: list[list[dict]] = []
        self.captured_messages: list[list[dict]] = []

    def __call__(self, **kwargs):
        idx = len(self.captured_tools)
        # Snapshot the per-call args BEFORE returning the generator.
        self.captured_tools.append(list(kwargs.get("tools") or []))
        self.captured_messages.append(
            [dict(m) for m in (kwargs.get("messages") or [])]
        )
        text, tool_calls = self.turns[idx]
        return self._gen(text, tool_calls)

    async def _gen(self, text: str, tool_calls: list[dict]):
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments") or {}),
                            },
                        }],
                    },
                }],
            }
        finish = "tool_calls" if tool_calls else "stop"
        yield {"choices": [{"delta": {}, "finish_reason": finish}]}
        yield {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


def _patch_pool(monkeypatch, pool):
    async def _build_pool(_options):
        return pool
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)


def _make_catalog(n: int) -> list[dict]:
    """N MCP-shape tools — Bash + n-1 unique domain tools."""
    tools = [_mcp_tool("Bash", "shell")]
    for i in range(n - 1):
        tools.append(_mcp_tool(f"domain_tool_{i:03d}", f"description {i}"))
    return tools


async def _drain(messages, options):
    out = []
    async for evt in run_query(messages, options):
        out.append(evt)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disabled_mode_sends_full_catalog(monkeypatch):
    """Regression guard: with tool_search_enabled=False, the harness sends
    every catalog tool every turn (today's behavior)."""
    pool = _FakePool("lloyd-mcp", _make_catalog(50))
    _patch_pool(monkeypatch, pool)
    script = _StreamScript([("done", [])])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    opts = RunOptions(
        model="primary",
        tool_search_enabled=False,
        session_id="test-disabled",
    )
    asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))

    sent = script.captured_tools[0]
    names = {t["function"]["name"] for t in sent}
    # Full catalog — no ToolSearch added.
    assert "Bash" in names
    assert "mcp__lloyd-mcp__domain_tool_000" in names
    assert TOOLSEARCH_TOOL_NAME not in names
    assert len(names) == 50

    # No catalog reminder injected.
    msgs = script.captured_messages[0]
    assert not any("deferred tools" in (m.get("content") or "") for m in msgs)


def test_enabled_first_turn_sends_only_baseline_plus_toolsearch(monkeypatch):
    pool = _FakePool("lloyd-mcp", _make_catalog(50))
    _patch_pool(monkeypatch, pool)
    script = _StreamScript([("acknowledged", [])])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    opts = RunOptions(
        model="primary",
        tool_search_enabled=True,
        tool_search_threshold_tools=30,
        session_id="test-enabled-first",
    )
    asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))

    sent = script.captured_tools[0]
    names = {t["function"]["name"] for t in sent}
    assert TOOLSEARCH_TOOL_NAME in names
    assert "Bash" in names
    # Domain tools are NOT advertised yet.
    assert not any(n.startswith("mcp__lloyd-mcp__domain_tool_") for n in names)


def test_threshold_below_count_keeps_full_catalog(monkeypatch):
    """If catalog is small enough, tool_search stays inactive even when enabled."""
    pool = _FakePool("lloyd-mcp", _make_catalog(5))
    _patch_pool(monkeypatch, pool)
    script = _StreamScript([("k", [])])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    opts = RunOptions(
        model="primary",
        tool_search_enabled=True,
        tool_search_threshold_tools=30,
        session_id="test-below-threshold",
    )
    asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))

    sent = script.captured_tools[0]
    names = {t["function"]["name"] for t in sent}
    assert TOOLSEARCH_TOOL_NAME not in names


def test_toolsearch_call_loads_matched_tools_for_next_turn(monkeypatch):
    """The end-to-end happy path: model calls ToolSearch, harness intercepts
    locally, returns a <functions> block, and the matched tool appears in
    the *next* turn's ``tools=`` array. The MCP pool is never asked to
    dispatch ToolSearch."""
    pool = _FakePool("lloyd-mcp", _make_catalog(50))
    _patch_pool(monkeypatch, pool)

    script = _StreamScript([
        # Turn 1: model calls ToolSearch
        ("", [{
            "id": "call_a",
            "name": TOOLSEARCH_TOOL_NAME,
            "arguments": {"query": "select:mcp__lloyd-mcp__domain_tool_007", "max_results": 5},
        }]),
        # Turn 2: model calls the now-loaded tool
        ("", [{
            "id": "call_b",
            "name": "mcp__lloyd-mcp__domain_tool_007",
            "arguments": {},
        }]),
        # Turn 3: model finishes
        ("ok", []),
    ])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    opts = RunOptions(
        model="primary",
        tool_search_enabled=True,
        tool_search_threshold_tools=30,
        session_id="test-toolsearch-load",
    )
    events = asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))

    # Turn 2 sees the loaded tool in tools=
    turn2_names = {t["function"]["name"] for t in script.captured_tools[1]}
    assert "mcp__lloyd-mcp__domain_tool_007" in turn2_names
    assert TOOLSEARCH_TOOL_NAME in turn2_names

    # Turn 1's tool_result for ToolSearch was synthesized by the harness,
    # not by the MCP pool.
    assert all(name != TOOLSEARCH_TOOL_NAME for name, _ in pool.call_log)
    # The MCP pool DID get the second tool call.
    assert ("mcp__lloyd-mcp__domain_tool_007", {}) in pool.call_log

    # The synthesized ToolSearch result is a tool_result event with a
    # <functions> block in its content.
    ts_results = [
        e for e in events
        if e["type"] == "tool_result" and e["name"] == TOOLSEARCH_TOOL_NAME
    ]
    assert len(ts_results) == 1
    assert "<function>" in ts_results[0]["content"]
    assert "domain_tool_007" in ts_results[0]["content"]


def test_catalog_reminder_injected_exactly_once(monkeypatch):
    pool = _FakePool("lloyd-mcp", _make_catalog(50))
    _patch_pool(monkeypatch, pool)
    script = _StreamScript([
        ("", [{"id": "c1", "name": "Bash", "arguments": {"command": "ls"}}]),
        ("done", []),
    ])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    opts = RunOptions(
        model="primary",
        tool_search_enabled=True,
        tool_search_threshold_tools=30,
        session_id="test-reminder-once",
    )
    asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))

    # Both turns: the catalog reminder is appended into the leading system
    # message (vLLM's chat template only allows one system message at the
    # start; a separate role:system addition is rejected). Marker must
    # appear exactly once across the whole message list — no duplication
    # across iterations.
    marker = "lloyd-toolsearch-catalog-reminder"
    for turn_msgs in script.captured_messages:
        hits = sum((m.get("content") or "").count(marker) for m in turn_msgs)
        assert hits == 1, f"expected exactly one catalog reminder, got {hits}"
        # And it lives inside a system message at position 0, not as a
        # standalone follow-up message.
        first = turn_msgs[0]
        assert first["role"] == "system"
        assert marker in first["content"]
        # No second system message anywhere in the list.
        assert sum(1 for m in turn_msgs if m.get("role") == "system") == 1


def test_unloaded_tool_call_returns_guidance(monkeypatch):
    """Defensive intercept: vLLM dispatches a tool not in visible_tools.
    The harness should return a guidance tool_result without hitting MCP."""
    pool = _FakePool("lloyd-mcp", _make_catalog(50))
    _patch_pool(monkeypatch, pool)
    script = _StreamScript([
        # Model calls a domain tool without first using ToolSearch.
        ("", [{
            "id": "call_x",
            "name": "mcp__lloyd-mcp__domain_tool_005",
            "arguments": {},
        }]),
        ("acknowledged", []),
    ])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    opts = RunOptions(
        model="primary",
        tool_search_enabled=True,
        tool_search_threshold_tools=30,
        session_id="test-unloaded-intercept",
    )
    events = asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))

    # MCP was NOT called for the unloaded tool.
    assert pool.call_log == []
    # A tool_result with is_error=True and ToolSearch guidance was emitted.
    results = [e for e in events if e["type"] == "tool_result"]
    assert results
    assert results[0]["is_error"]
    assert "ToolSearch" in results[0]["content"]
    assert "select:mcp__lloyd-mcp__domain_tool_005" in results[0]["content"]


def test_session_scoped_loaded_set_persists_across_run_query(monkeypatch):
    """Two consecutive run_query calls with the same session_id should share
    the loaded set — the model only needs to ToolSearch once per session."""
    pool = _FakePool("lloyd-mcp", _make_catalog(50))
    _patch_pool(monkeypatch, pool)

    # First run_query: model loads domain_tool_002.
    script_a = _StreamScript([
        ("", [{
            "id": "c_a",
            "name": TOOLSEARCH_TOOL_NAME,
            "arguments": {"query": "select:mcp__lloyd-mcp__domain_tool_002"},
        }]),
        ("done a", []),
    ])
    monkeypatch.setattr("app.harness.loop.stream_chat", script_a)

    opts = RunOptions(
        model="primary",
        tool_search_enabled=True,
        tool_search_threshold_tools=30,
        session_id="persist-sess",
    )
    asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))

    # Second run_query, same session_id: domain_tool_002 should already be
    # in tools= on the FIRST turn this time.
    script_b = _StreamScript([("done b", [])])
    monkeypatch.setattr("app.harness.loop.stream_chat", script_b)
    asyncio.run(_drain([{"role": "user", "content": "again"}], opts))

    first_turn_names = {t["function"]["name"] for t in script_b.captured_tools[0]}
    assert "mcp__lloyd-mcp__domain_tool_002" in first_turn_names


def test_toolsearch_in_disallowed_falls_back_to_full_catalog(monkeypatch):
    """If ToolSearch itself is disabled, the harness falls back to today's
    behavior (advertise everything) rather than locking the model out."""
    pool = _FakePool("lloyd-mcp", _make_catalog(50))
    _patch_pool(monkeypatch, pool)
    script = _StreamScript([("ok", [])])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    opts = RunOptions(
        model="primary",
        tool_search_enabled=True,
        tool_search_threshold_tools=30,
        disallowed_tools=[TOOLSEARCH_TOOL_NAME],
        session_id="test-disallowed-toolsearch",
    )
    asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))

    sent = script.captured_tools[0]
    names = {t["function"]["name"] for t in sent}
    assert TOOLSEARCH_TOOL_NAME not in names
    # Full domain catalog is advertised again.
    assert "mcp__lloyd-mcp__domain_tool_010" in names
