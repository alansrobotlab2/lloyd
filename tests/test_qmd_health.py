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
