"""The `research_*` MCP tools — the surface a generator and a human share.

The registry is written from two processes, and this module is the aggregator's
half. What is pinned here is mostly about what a tool must NOT do: never raise
out of `call_tool`, never open the database inside `list_tools`, never claim a
topic from a chat turn, and never advertise a second field competing with the
harness's injected `summary`.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from agent_mcp import research as T
from app import research_store as R


@pytest.fixture
def registry(tmp_path):
    """A registry of this test's own, pointed at by the module default."""
    store = R.configure(tmp_path / "research.db")
    yield store
    R.reset()


async def call(name: str, args: dict | None = None):
    result = await T.call_tool(name, args or {})
    return result, json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# The advertised schemas
# ---------------------------------------------------------------------------


async def test_the_five_tools_are_advertised():
    names = [t.name for t in await T.list_tools()]
    assert names == ["research_propose", "research_next", "research_complete",
                     "research_list", "research_stats"]


async def test_no_tool_asks_for_the_caption_twice():
    """`add_summary_param` injects a `summary` on every advertised tool.

    A tool declaring its own `summary` opts out of that injection silently, and
    one declaring a `description` argument asks the same question one key
    later — which is how 49 consecutive Bash calls came to answer into the
    wrong field while nothing errored.
    """
    for tool in await T.list_tools():
        props = tool.input_schema.get("properties", {})
        assert "summary" not in props, tool.name
        assert "description" not in props, tool.name


async def test_every_parameter_is_documented_and_the_tools_explain_themselves():
    for tool in await T.list_tools():
        assert len(tool.description) >= 60, tool.name
        assert "additionalProperties" not in tool.input_schema, tool.name
        for key, spec in tool.input_schema.get("properties", {}).items():
            assert spec.get("description"), f"{tool.name}.{key}"


async def test_the_advertised_statuses_match_the_store():
    """The enum is a literal so `list_tools` cannot raise; this is what keeps
    the literal honest."""
    assert tuple(T._STATUSES) == tuple(R.STATUSES)
    tools = {t.name: t for t in await T.list_tools()}
    enum = tools["research_list"].input_schema["properties"]["status"]["enum"]
    assert tuple(enum) == tuple(R.STATUSES)


async def test_listing_tools_never_touches_the_database(monkeypatch):
    """A module whose `list_tools` raises is recorded `ok: False`, and that
    turns the whole aggregator's /health into a 503."""
    def explode():
        raise AssertionError("list_tools opened the store")
    monkeypatch.setattr(R, "store", explode)
    assert len(await T.list_tools()) == 5


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


async def test_proposing_a_topic_queues_it(registry):
    result, payload = await call("research_propose", {
        "topic": "WSOLA time-stretching for streaming TTS with no speed control",
        "domain": "voice-tts", "signal": "daily-note"})
    assert result.is_error is False
    assert payload["created"] is True and payload["status"] == "queued"
    assert "queued as #" in payload["message"]
    assert registry.get(payload["id"])["signal"] == "daily-note"


async def test_proposing_the_same_topic_twice_says_so_instead_of_duplicating(registry):
    await call("research_propose", {"topic": "Speculative decoding for agent loops"})
    result, payload = await call("research_propose",
                                 {"topic": "speculative decoding for agent loops"})
    assert result.is_error is False, "a known topic is not an error"
    assert payload["created"] is False
    assert "already known" in payload["message"]
    assert registry.stats()["total"] == 1


async def test_a_proposal_carries_similar_topics_back(registry):
    await call("research_propose", {
        "topic": "Real-time SLAM integration with VLA policies during manipulation"})
    _, payload = await call("research_propose", {
        "topic": "Real-time SLAM integration with VLA policies for manipulation tasks"})
    assert payload["similar"], "the generator needs this to avoid a reword"
    assert payload["similar"][0]["status"] == "queued"


async def test_a_proposal_with_no_topic_is_an_error(registry):
    result, payload = await call("research_propose", {})
    assert result.is_error is True
    assert "topic is required" in payload["error"]


async def test_a_topic_that_normalises_to_nothing_is_an_error(registry):
    result, payload = await call("research_propose", {"topic": "!!!"})
    assert result.is_error is True and "error" in payload


async def test_the_proposer_defaults_to_the_calling_session(registry, monkeypatch):
    monkeypatch.setattr(T, "get_bound_session", lambda: "20260908_abcdef")
    _, payload = await call("research_propose", {"topic": "a topic from a chat turn"})
    assert registry.get(payload["id"])["proposed_by"] == "20260908_abcdef"


async def test_an_explicit_proposer_wins_over_the_session(registry):
    _, payload = await call("research_propose",
                            {"topic": "a topic the user asked for", "proposed_by": "human"})
    assert registry.get(payload["id"])["proposed_by"] == "human"


async def test_the_depth_cap_is_reported_not_raised(registry):
    for i in range(R.MAX_QUEUED):
        registry.propose(f"a distinct queued topic numbered {i}")
    result, payload = await call("research_propose", {"topic": "one topic too many"})
    assert result.is_error is False, "a full queue is a state, not a failure"
    assert payload["created"] is False and payload["id"] is None
    assert "already queued" in payload["message"]


# ---------------------------------------------------------------------------
# next / complete / list / stats
# ---------------------------------------------------------------------------


async def test_next_peeks_without_claiming(registry):
    """A chat turn that peeks and wanders off must not strand the topic."""
    topic = registry.propose("a topic the worker should still get")["id"]
    _, payload = await call("research_next", {"n": 5})
    assert [t["id"] for t in payload["topics"]] == [topic]
    assert registry.get(topic)["status"] == "queued", "peeking claimed it"


async def test_completing_a_topic_records_the_outcome(registry):
    topic = registry.propose("a topic researched by hand")["id"]
    registry.claim(topic, by="human")
    result, payload = await call("research_complete", {
        "topic_id": topic, "status": "written",
        "artifact_path": "/home/x/knowledge/research/a.md", "note": "facts=6"})
    assert result.is_error is False
    assert payload["topic"]["status"] == "written"
    assert payload["topic"]["artifact_path"].endswith("a.md")


async def test_nothing_found_is_a_recordable_outcome(registry):
    topic = registry.propose("a topic with nothing behind it")["id"]
    registry.claim(topic, by="human")
    _, payload = await call("research_complete", {
        "topic_id": topic, "status": "nothing_found", "note": "three searches, nothing"})
    assert payload["topic"]["status"] == "nothing_found"
    assert registry.recent(days=1)[0]["status"] == "nothing_found"


async def test_completing_with_a_bad_status_is_an_error(registry):
    topic = registry.propose("a topic")["id"]
    result, payload = await call("research_complete",
                                 {"topic_id": topic, "status": "done"})
    assert result.is_error is True and "written|nothing_found" in payload["error"]


async def test_completing_without_an_id_is_an_error(registry):
    result, payload = await call("research_complete", {"status": "written"})
    assert result.is_error is True and "topic_id is required" in payload["error"]


async def test_a_settled_topic_is_not_rewritten_through_the_tool(registry):
    topic = registry.propose("a settled topic")["id"]
    registry.claim(topic, by="human")
    await call("research_complete", {"topic_id": topic, "status": "nothing_found"})
    result, payload = await call("research_complete",
                                 {"topic_id": topic, "status": "written"})
    assert result.is_error is True and "already" in payload["error"]


async def test_listing_filters_by_status_and_recency(registry):
    written = registry.propose("a written topic")["id"]
    registry.claim(written, by="w")
    registry.finish(written, "written", artifact_path="/tmp/a.md")
    registry.propose("a queued topic")

    _, payload = await call("research_list", {"status": "written"})
    assert payload["count"] == 1 and payload["topics"][0]["topic"] == "a written topic"
    _, payload = await call("research_list", {"since_days": 30})
    assert payload["count"] == 2


async def test_stats_answers_without_arguments(registry):
    registry.propose("a queued topic")
    _, payload = await call("research_stats")
    assert payload["queued"] == 1 and payload["done_today"] == 0


# ---------------------------------------------------------------------------
# Failure posture
# ---------------------------------------------------------------------------


async def test_an_unknown_tool_is_an_error_not_an_exception():
    result, payload = await call("research_nonexistent")
    assert result.is_error is True and "unknown tool" in payload["error"]


async def test_an_unreadable_registry_reports_rather_than_raising(monkeypatch, registry):
    def unavailable():
        raise R.StoreUnavailable("disk is gone")
    monkeypatch.setattr(T, "_store", unavailable)
    result, payload = await call("research_stats")
    assert result.is_error is True
    assert "registry unavailable" in payload["error"]


async def test_a_busy_registry_says_so(monkeypatch, registry):
    def busy():
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(T, "_store", busy)
    result, payload = await call("research_list")
    assert result.is_error is True and "busy" in payload["error"]


async def test_an_unexpected_failure_still_returns_a_payload(monkeypatch, registry):
    def boom():
        raise RuntimeError("something new")
    monkeypatch.setattr(T, "_store", boom)
    result, payload = await call("research_stats")
    assert result.is_error is True and "RuntimeError" in payload["error"]


def test_store_calls_are_hopped_off_the_event_loop():
    """The aggregator awaits `call_tool` on the loop that serves every session,
    and every handler here opens SQLite."""
    import inspect
    src = inspect.getsource(T.call_tool)
    assert "asyncio.to_thread" in src
