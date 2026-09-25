"""#1485: chat transcripts and autonomy runs as recall floors; session_recall on qmd.

Both ship OFF (`RECALL_EPISODIC_FLOORS`, `SESSION_RECALL_BACKEND`); these tests
pin that the off state is today's request byte for byte, what the on state asks
for, and that #1511's echo rule applies on both paths. Hermetic: `_qmd_post` and
`_qmd_daemon_search` are stubbed.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mcp import session as session_mod  # noqa: E402
from agent_mcp import transcript_self_hit as tsh  # noqa: E402
from agent_mcp import vault as vault_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _no_note_reads(monkeypatch):
    monkeypatch.setattr(tsh, "_sessions_root", lambda: None)


def test_off_is_todays_shape():
    assert vault_mod.RECALL_EPISODIC_FLOORS is False
    for ranker in ("djev", "qmd"):
        shape = vault_mod.recall_doc_leg_shape(ranker)
        assert "extra" not in shape
        assert "sessions" not in shape["floor"] and "autonomy-runs" not in shape["floor"]
    assert vault_mod.recall_doc_leg_shape("djev")["candidateLimit"] == vault_mod.RECALL_DJEV_HEAD


def test_on_names_both_collections_and_still_fits_one_canvas(monkeypatch):
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", True)
    shape = vault_mod.recall_doc_leg_shape("djev")
    assert shape["extra"] == ["sessions"]
    assert shape["floor"]["sessions"] == 1
    # head + every floor x two search legs must fit the rows djev ranks in one read
    assert shape["candidateLimit"] + 2 * sum(shape["floor"].values()) <= vault_mod.RECALL_DJEV_POOL
    assert vault_mod.recall_doc_leg_shape("qmd")["extra"] == ["sessions"]


def _capture_post(monkeypatch, rows):
    sent = {}

    def _post(payload):
        sent.update(payload)
        return rows
    monkeypatch.setattr(vault_mod, "_qmd_post", _post)
    return sent


def test_extra_collections_join_the_request_and_survive_the_fold(monkeypatch):
    rows = [{"file": "qmd://sessions/2026-09-20/20260920_101010_ivaaaa.md", "score": 0.9},
            {"file": "qmd://autonomy-runs/39/run_1.md", "score": 0.8},
            {"file": "qmd://knowledge/x.md", "score": 0.7},
            {"file": "qmd://agents/y.md", "score": 0.6}]
    sent = _capture_post(monkeypatch, rows)
    out = vault_mod._qmd_daemon_search("q words here", 32, vault_mod.VAULT_SEGMENTS,
                                       extra_collections=["sessions", "autonomy-runs"],
                                       floor={"sessions": 1, "autonomy-runs": 1})
    assert sent["collections"][-2:] == ["sessions", "autonomy-runs"]
    assert sent["collectionFloor"] == {"sessions": 1, "autonomy-runs": 1}
    files = [r["file"] for r in out]
    assert "qmd://sessions/2026-09-20/20260920_101010_ivaaaa.md" in files
    assert "qmd://autonomy-runs/39/run_1.md" in files
    assert "qmd://agents/y.md" not in files


def test_without_extras_the_fold_still_drops_them(monkeypatch):
    rows = [{"file": "qmd://sessions/2026-09-20/a.md", "score": 0.9},
            {"file": "qmd://knowledge/x.md", "score": 0.7}]
    sent = _capture_post(monkeypatch, rows)
    out = vault_mod._qmd_daemon_search("q words here", 32, vault_mod.VAULT_SEGMENTS)
    assert "sessions" not in sent["collections"]
    assert [r["file"] for r in out] == ["qmd://knowledge/x.md"]


def test_extras_are_ignored_on_a_restricted_search(monkeypatch):
    sent = _capture_post(monkeypatch, [])
    vault_mod._qmd_daemon_search("q words here", 5, ["knowledge"], extra_collections=["sessions"])
    assert sent["collections"] == ["knowledge"]


PROMPT = "please walk me through how the qmd daemon picks its embedding model again"


def test_recall_drops_an_echoing_transcript_and_the_callers_own(monkeypatch):
    from agent_mcp import _task_registry
    docs = [{"path": "sessions/2026-09-20/20260920_101010_ivaaaa.md", "snippet": f"user: {PROMPT}"},
            {"path": "sessions/2026-09-21/20260921_101010_ivbbbb.md", "snippet": "unrelated chat"},
            {"path": "sessions/2026-09-22/20260922_101010_ivcccc.md", "snippet": "user: other"},
            {"path": "knowledge/qmd.md", "snippet": f"user: {PROMPT}"}]
    tok = _task_registry.current_session_id.set("20260922_101010_ivcccc")
    try:
        kept = vault_mod._drop_recall_self_hits(PROMPT, docs)
    finally:
        _task_registry.current_session_id.reset(tok)
    assert [d["path"] for d in kept] == ["sessions/2026-09-21/20260921_101010_ivbbbb.md",
                                         "knowledge/qmd.md"]


def test_recall_filter_is_a_noop_without_transcripts():
    docs = [{"path": "knowledge/qmd.md", "snippet": "x"}]
    assert vault_mod._drop_recall_self_hits(PROMPT, docs) is docs


# ── session_recall on qmd ───────────────────────────────────────────────────

def _day(offset: int) -> str:
    return (datetime.datetime.now() - datetime.timedelta(days=offset)).strftime("%Y%m%d")


def test_session_recall_qmd_keeps_the_shape_window_and_echo_rule(monkeypatch):
    recent, old = _day(1), _day(40)
    rows = [
        {"file": f"qmd://sessions/x/{recent}_101010_ivaaaa.md", "score": 0.9,
         "snippet": "we changed the embedding model to qwen3 in index.yml"},
        {"file": f"qmd://sessions/x/{old}_101010_ivbbbb.md", "score": 0.8, "snippet": "old"},
        {"file": f"qmd://sessions/x/{recent}_111111_ivcccc.md", "score": 0.7,
         "snippet": f"user: {PROMPT}"},
        {"file": "qmd://knowledge/k.md", "score": 0.6, "snippet": "not a session"},
    ]
    seen = {}

    def _search(query, limit, collections, **kw):
        seen.update(collections=collections, limit=limit, **kw)
        return rows
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", _search)
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "qmd")
    out = session_mod._session_recall({"query": PROMPT, "days": 7, "limit": 5})
    assert seen["collections"] == ["sessions"] and seen["limit"] == 5 * session_mod._SESSION_QMD_OVERFETCH
    assert out["backend"] == "qmd"
    assert [s["session_id"] for s in out["sessions"]] == [f"{recent}_101010_ivaaaa"]
    s = out["sessions"][0]
    assert set(s) >= {"session_id", "created_at", "preview", "snippets", "match_score"}
    assert s["created_at"].startswith(f"{recent[:4]}-{recent[4:6]}-{recent[6:]}T10:10:10")


def test_session_recall_qmd_falls_back_to_the_scorer_on_an_outage(monkeypatch):
    def _down(*a, **k):
        raise vault_mod.QmdUnavailable("down")
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", _down)
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "qmd")
    monkeypatch.setattr(session_mod, "_load_session_index", lambda max_days=14: {})
    out = session_mod._session_recall({"query": "anything at all", "days": 7})
    assert "backend" not in out and out["sessions"] == []


def test_session_recall_default_backend_is_the_token_scorer():
    assert session_mod.SESSION_RECALL_BACKEND == "tokens"
