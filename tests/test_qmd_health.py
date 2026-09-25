"""A recall answered without its rerank is counted, logged and announced -- once.

The daemon half is qmd/test/rerank-fallback.test.ts; this is the client half."""
from __future__ import annotations

import json
import time

import pytest

from app import qmd_health


@pytest.fixture(autouse=True)
def _fresh():
    qmd_health._reset_for_tests()
    yield
    qmd_health._reset_for_tests()


def _wait(pred, seconds=2.0):
    end = time.time() + seconds
    while time.time() < end and not pred():
        time.sleep(0.01)
    return pred()


def test_a_ranked_answer_is_counted_and_rings_nothing():
    rang = []
    assert qmd_health.note_response(True, {"reranked": True, "rerankFallback": None, "ms": 41, "phases": {"rerank": 30}},
                                    announce=lambda *a: rang.append(a)) is False
    s = qmd_health.stats()
    assert (s["responses"], s["reranked"], s["rerank_fallbacks"], s["last_ms"]) == (1, 1, 0, 41)
    time.sleep(0.05)
    assert rang == []


def test_a_fallback_is_counted_and_announced_once_per_cooldown():
    rang = []
    meta = {"reranked": False, "rerankFallback": "A context size of 4096 is too large for the available VRAM"}
    t0 = 1_000_000.0
    assert qmd_health.note_response(True, meta, announce=lambda *a: rang.append(a), now=t0) is True
    assert qmd_health.note_response(True, meta, announce=lambda *a: rang.append(a), now=t0 + 60) is True
    assert _wait(lambda: len(rang) >= 1)
    time.sleep(0.05)
    assert len(rang) == 1, "a bad evening is one toast, not one per recall"
    assert "too large for the available VRAM" in rang[0][1]
    assert qmd_health.note_response(True, meta, announce=lambda *a: rang.append(a),
                                    now=t0 + qmd_health.ANNOUNCE_COOLDOWN_SECONDS + 1) is True
    assert _wait(lambda: len(rang) == 2)
    s = qmd_health.stats()
    assert s["rerank_fallbacks"] == 3 and "VRAM" in s["last_fallback_reason"]


def test_unreranked_is_only_a_fallback_when_a_rerank_was_asked_for():
    rang = []
    # prefetch sends rerank: false; the daemon reports reranked: null
    assert qmd_health.note_response(False, {"reranked": None}, announce=lambda *a: rang.append(a)) is False
    # and a false that nobody asked for is not an alarm either
    assert qmd_health.note_response(False, {"reranked": False}, announce=lambda *a: rang.append(a)) is False
    time.sleep(0.05)
    assert rang == [] and qmd_health.stats()["rerank_fallbacks"] == 0


def test_a_daemon_that_sends_no_meta_is_tolerated_and_counted():
    assert qmd_health.note_response(True, None) is False
    assert qmd_health.stats()["without_meta"] == 1


def test_the_client_reports_what_the_daemon_said(monkeypatch):
    """`_qmd_post` is the one door every qmd request leaves through."""
    from agent_mcp import vault as V

    class Resp:
        def __init__(self, body): self._b = json.dumps(body).encode()
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    body = {"results": [{"file": "qmd://memory/a.md", "title": "a", "snippet": "s", "score": 0.5}],
            "meta": {"reranked": False, "rerankFallback": "no VRAM", "ms": 300, "phases": {}}}
    monkeypatch.setattr(V.urllib.request, "urlopen", lambda *a, **k: Resp(body))
    rows = V._qmd_post({"searches": [], "rerank": True})
    assert [r["file"] for r in rows] == ["qmd://memory/a.md"], "a degraded answer is still an answer"
    assert qmd_health.stats()["rerank_fallbacks"] == 1


# ── #1498: memory search and backlog dedupe go through the same door ─────────

# Captured at import, before any fixture runs: `tests/conftest.py`'s autouse
# `_isolate_backlog_dedupe` replaces `semantic_candidates` with a stub for every
# test so none reaches the live daemon. The real function is what these tests
# are about; `urlopen` is stubbed below instead, so the daemon is still never
# called.
from agent_mcp import backlog_similar as _SIM  # noqa: E402

_REAL_SEMANTIC_CANDIDATES = _SIM.semantic_candidates

_DEGRADED = {"reranked": False, "rerankFallback": "no VRAM", "ms": 300, "phases": {}}


class _Resp:
    def __init__(self, body): self._b = json.dumps(body).encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _through_the_door(monkeypatch, body):
    """Stub the transport under `qmd_query`; record what reached the wire and
    what went through the door. A caller that opened its own request would
    appear in the first list and not the second."""
    from agent_mcp import vault as V
    wire, door = [], []
    real_query = V.qmd_query

    def urlopen(req, *a, **k):
        wire.append(json.loads(req.data))
        return _Resp(body)

    def counted(payload, **kw):
        door.append(payload)
        return real_query(payload, **kw)

    monkeypatch.setattr(V.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(V, "qmd_query", counted)
    return wire, door


def test_neither_caller_opens_its_own_request_to_the_daemon():
    """Clause 1 by structure: no urllib and no daemon URL in either module."""
    from pathlib import Path

    import agent_mcp.backlog_similar as sim
    import app.routers.memory as mem
    for mod in (sim, mem):
        src = Path(mod.__file__).read_text()
        assert "urlopen" not in src and "urllib.request" not in src, mod.__name__
        assert "QMD_DAEMON_URL" not in src and "8181" not in src, mod.__name__
        assert "qmd_query" in src, f"{mod.__name__} no longer goes through the door"


def test_backlog_dedupe_goes_through_the_door_and_its_fallback_is_counted(monkeypatch):
    body = {"results": [{"file": "qmd://backlog/1498-some-item.md", "score": 0.81}],
            "meta": _DEGRADED}
    wire, door = _through_the_door(monkeypatch, body)
    rows = _REAL_SEMANTIC_CANDIDATES("memory search - bypasses the door", limit=4, timeout=2.0)
    assert rows == [{"id": 1498, "score": 0.81}], "a degraded answer is still an answer"
    assert len(door) == 1 and wire == door, "the dedupe opened its own request"
    # Its payload as it was: vec only, the backlog collection, rerank asked for.
    assert door[0] == {"searches": [{"type": "vec", "query": "memory search bypasses the door"}],
                       "collections": ["backlog"], "limit": 4, "rerank": True}
    assert qmd_health.stats()["rerank_fallbacks"] == 1


def test_memory_search_goes_through_the_door_and_its_fallback_is_counted(monkeypatch):
    import asyncio

    from app.routers import memory as mem
    body = {"results": [{"file": "qmd://people/Ali%20Behrouz.md", "title": "Ali", "score": 0.5,
                         "snippet": "s", "summary": "sum"}],
            "meta": _DEGRADED}
    wire, door = _through_the_door(monkeypatch, body)
    out = json.loads(asyncio.run(mem.memory_search(q="who is ali", limit=3, scope="people")).body)
    assert out["results"] == [{"path": "qmd://people/Ali Behrouz.md", "title": "Ali",
                               "score": 0.5, "snippet": "s", "summary": "sum"}]
    assert len(door) == 1 and wire == door, "memory search opened its own request"
    # Its payload as it was: no `rerank` key, which qmd reads as "rerank"
    # (it skips only on `rerank: false`) — so an unreranked reply IS a fallback.
    assert door[0] == {"searches": [{"type": "lex", "query": "who is ali"},
                                    {"type": "vec", "query": "who is ali"}],
                       "limit": 3, "collections": ["people"]}
    assert qmd_health.stats()["rerank_fallbacks"] == 1


def test_an_explicit_no_rerank_is_still_not_a_fallback(monkeypatch):
    """The widened reading (absent key = asked) must not turn prefetch's
    `rerank: false` into an alarm."""
    from agent_mcp import vault as V
    monkeypatch.setattr(V.urllib.request, "urlopen",
                        lambda *a, **k: _Resp({"results": [], "meta": _DEGRADED}))
    V.qmd_query({"searches": [], "rerank": False})
    assert qmd_health.stats()["rerank_fallbacks"] == 0
