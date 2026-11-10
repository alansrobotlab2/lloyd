"""A retrieval leg that fails is named, not rendered as "nothing matched".

Before 2026-11-10 `_search_vault` turned a `QmdUnavailable` into `[]` with no
log, a leg that raised was swallowed by `_prefetch_run`, and a crash after the
drains returned a 500 with the ambient queue and next-session notes consumed.
"""
from __future__ import annotations

import pytest

from app import prefetch
from agent_mcp.vault import QmdUnavailable

_Q = "what did we decide about the guardian rollback ledger policy"


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(prefetch, "_search_skills_for_turn", lambda *a: [])
    monkeypatch.setattr(prefetch, "_search_facts", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_recent_sessions", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_backlog_refs", lambda t: [])
    monkeypatch.setattr(prefetch, "_format_ide_state", lambda: "")
    monkeypatch.setattr(prefetch, "_leg_last_logged", {})


def _down(*a, **kw):
    raise QmdUnavailable("connection refused")


def test_search_vault_still_returns_empty_but_records_the_failure(monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _down)
    res, fail = prefetch._tracked(prefetch._search_vault, _Q)
    assert res == []
    assert fail == "QmdUnavailable"


def test_a_qmd_outage_is_said_in_the_block(quiet, monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _down)
    out = prefetch.prefetch_context(_Q, session_id="test-leg-outage", plan_mode=False)
    assert "<retrieval_unavailable>" in out
    assert "vault search (QmdUnavailable)" in out
    assert out.endswith("\n\n" + _Q)


def test_a_raising_leg_is_named(quiet, monkeypatch):
    def _boom(t):
        raise RuntimeError("facts tree unreadable")
    monkeypatch.setattr(prefetch, "_search_facts", _boom)
    monkeypatch.setattr(prefetch, "_qmd_daemon_search",
                        lambda *a, **kw: [{"file": "qmd://obsidian/a.md", "title": "A",
                                           "score": 0.9, "snippet": "rollback ledger"}])
    out = prefetch.prefetch_context(_Q, session_id="test-leg-raise", plan_mode=False)
    assert "entity facts (RuntimeError)" in out
    assert "vault search" not in out.split("<retrieval_unavailable>")[1], (
        "the vault answered, so it must not be reported unavailable")


def test_a_healthy_turn_says_nothing_about_availability(quiet, monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", lambda *a, **kw: [])
    out = prefetch.prefetch_context(_Q, session_id="test-leg-healthy", plan_mode=False)
    assert "<retrieval_unavailable>" not in out


def test_a_crash_after_the_drains_puts_the_signals_back(monkeypatch, tmp_path):
    from app import sessions_io
    from app.sessions_io import AmbientPrefetchEntry

    sid = "test-leg-crash"
    sessions_io.clear_ambient_prefetch(sid)
    import time as _t
    sessions_io.enqueue_ambient_prefetch(sid, AmbientPrefetchEntry(
        source="calendar", summary="standup in 10 minutes", enqueued_at=_t.time()))

    def _explode(*a, **kw):
        raise RuntimeError("renderer bug")
    monkeypatch.setattr(prefetch, "_prefetch_run", _explode)
    monkeypatch.setattr(prefetch, "peek_next_session_notes", lambda: [], raising=False)

    out = prefetch.prefetch_context(_Q, session_id=sid, plan_mode=False)
    assert out == _Q, "a crash in the budgeted half must not fail the turn"
    left = sessions_io.peek_ambient_prefetch(sid)
    assert [e.summary for e in left] == ["standup in 10 minutes"], (
        "the drained ambient entry was lost with the crash")
    sessions_io.clear_ambient_prefetch(sid)
