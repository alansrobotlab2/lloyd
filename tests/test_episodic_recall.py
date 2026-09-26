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


def test_off_is_todays_shape(monkeypatch):
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", False)
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


def test_recall_keeps_only_chat_transcripts(monkeypatch):
    docs = [{"path": "sessions/2026-09-20/20260920_101010_ivaaaa.md", "snippet": "a chat"},
            {"path": "sessions/2026-09-20/20260920_101010_youtubed_1a2b.md", "snippet": "a digest"},
            {"path": "sessions/2026-09-07/e2e_selfmod294_1788815463.md", "snippet": "fixture"},
            {"path": "knowledge/qmd.md", "snippet": "doc"}]
    kept = [d["path"] for d in vault_mod._drop_recall_self_hits("unrelated words here", docs)]
    assert kept == ["sessions/2026-09-20/20260920_101010_ivaaaa.md", "knowledge/qmd.md"]
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_CHAT_ONLY", False)
    assert len(vault_mod._drop_recall_self_hits("unrelated words here", docs)) == 4


def test_session_recall_qmd_skips_background_exports(monkeypatch):
    recent = _day(1)
    rows = [{"file": f"qmd://sessions/x/{recent}_101010_autocode_9f2a.md", "score": 0.9, "snippet": "bg"},
            {"file": f"qmd://sessions/x/{recent}_101010_ivaaaa.md", "score": 0.8, "snippet": "chat"}]
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", lambda *a, **k: rows)
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "qmd")
    out = session_mod._session_recall({"query": "what did we say", "days": 7, "limit": 5})
    assert [s["session_id"] for s in out["sessions"]] == [f"{recent}_101010_ivaaaa"]


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


def _config(monkeypatch, retrieval: dict):
    from app import config as cfg
    monkeypatch.setattr(cfg, "CONFIG", {**(cfg.CONFIG or {}), "retrieval": retrieval})


def test_session_recall_backend_reads_config(monkeypatch):
    assert session_mod.SESSION_RECALL_BACKEND is None      # no override ships
    _config(monkeypatch, {})
    assert session_mod.session_recall_backend() == "tokens"
    _config(monkeypatch, {"session_recall": {"backend": "qmd"}})
    assert session_mod.session_recall_backend() == "qmd"
    _config(monkeypatch, {"session_recall": {"backend": "bogus"}})
    assert session_mod.session_recall_backend() == "tokens"
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "tokens")
    _config(monkeypatch, {"session_recall": {"backend": "qmd"}})
    assert session_mod.session_recall_backend() == "tokens"


def test_episodic_floors_read_config_and_the_kill_env(monkeypatch):
    assert vault_mod.RECALL_EPISODIC_FLOORS is None        # no override ships
    monkeypatch.delenv("LLOYD_RECALL_EPISODIC", raising=False)
    _config(monkeypatch, {})
    assert vault_mod.recall_episodic_floors() is False
    assert "extra" not in vault_mod.recall_doc_leg_shape("djev")
    _config(monkeypatch, {"recall": {"episodic_floors": True}})
    assert vault_mod.recall_episodic_floors() is True
    assert vault_mod.recall_doc_leg_shape("djev")["extra"] == ["sessions"]
    monkeypatch.setenv("LLOYD_RECALL_EPISODIC", "0")
    assert vault_mod.recall_episodic_floors() is False


def test_the_tracked_config_matches_the_verdict():
    """config.yaml carries both keys, so a rebuild boots into what is served."""
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    r = raw["retrieval"]
    assert isinstance(r["recall"]["episodic_floors"], bool)
    assert r["session_recall"]["backend"] in ("tokens", "qmd")


# ── the regression pin sends what production sends ──────────────────────────

def test_production_payload_mirrors_the_floor(monkeypatch):
    from scripts.automod import evalpin
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", False)
    off = evalpin.production_payload("what did we decide")
    assert "sessions" not in off["collections"]
    assert "sessions" not in off.get("collectionFloor", {})
    monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", True)
    on = evalpin.production_payload("what did we decide")
    assert on["collections"] == list(vault_mod.VAULT_SEGMENTS) + ["sessions"]
    assert on["collectionFloor"]["sessions"] == 1
    monkeypatch.setattr(vault_mod, "recall_reranker", lambda: "djev")
    djev = evalpin.production_payload("what did we decide")
    assert djev["candidateLimit"] == vault_mod.RECALL_EPISODIC_DJEV_HEAD
    assert djev["collectionFloor"]["sessions"] == 1 and djev["rerank"] is False


def test_production_payload_is_the_doc_legs_request(monkeypatch):
    """Byte-level: the pin's payload equals what `_qmd_daemon_search` posts for
    the recall doc leg, flag on and off (lex/vec query text aside)."""
    from scripts.automod import evalpin
    for on in (False, True):
        monkeypatch.setattr(vault_mod, "RECALL_EPISODIC_FLOORS", on)
        sent = _capture_post(monkeypatch, [])
        shape = vault_mod.recall_doc_leg_shape()
        vault_mod._qmd_daemon_search("what did we decide", shape["limit"], vault_mod.VAULT_SEGMENTS,
                                     skip_rerank=not shape["rerank"], exact_pool=shape["limit"],
                                     candidate_limit=shape["candidateLimit"], floor=shape["floor"],
                                     lex_mode=shape.get("lexMode"),
                                     extra_collections=shape.get("extra"))
        pin = evalpin.production_payload("what did we decide")
        for k in ("limit", "candidateLimit", "collections", "rerank", "fusion",
                  "collectionFloor", "lexMode"):
            assert pin.get(k) == sent.get(k), (on, k)


def test_pin_refuses_a_snapshot_missing_a_requested_collection():
    from scripts.automod import evalpin
    prov = {"collections": {"knowledge": 10, "sessions": 0}}
    assert evalpin.missing_collections(prov, ["knowledge", "sessions"]) == ["sessions"]
    assert evalpin.missing_collections({"collections": None}, ["sessions"]) == []


def test_snapshot_collections_counts_active_documents(tmp_path):
    import sqlite3
    from scripts.automod import evalpin
    con = sqlite3.connect(tmp_path / "i.sqlite")
    con.execute("create table documents (collection text, path text, active int)")
    con.executemany("insert into documents values (?,?,?)",
                    [("sessions", "a", 1), ("sessions", "b", 0), ("knowledge", "c", 1)])
    assert evalpin.snapshot_collections(con) == {"sessions": 1, "knowledge": 1}


def test_pin_config_report(tmp_path):
    from scripts.automod import evalpin
    (tmp_path / "index.yml").write_text(
        "collections:\n  sessions: {path: /x, pattern: '**/*.md'}\nmodels:\n  embed: E\n")
    (tmp_path / "evalpin.yml").write_text("collections: {}\nmodels:\n  embed: E\n")
    r = evalpin.pin_config_report("evalpin", config_dir=tmp_path)
    assert r["embed_match"] and not r["collections_match"]
    assert r["missing_collections"] == ["sessions"]
