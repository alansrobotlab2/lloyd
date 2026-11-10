"""`session_read` opens a past conversation the recall tools only point at.

Before it, a recall hit was a session id plus three 300-char snippets and a
`sessions/<date>/<id>.md` path that `vault_read` resolved under an empty vault
directory — nothing could return the rest of what was said.
"""
from __future__ import annotations

import asyncio
import gzip
import json

import pytest

from agent_mcp import annotations, session as session_mod, vault

SID = "20261109_181624_abc123"


def _msg(role, text="", **extra):
    return {"role": role, "content": [{"type": "text", "text": text}], **extra}


def _session(n_turns: int = 3, reply: str = "reply") -> dict:
    msgs = []
    for i in range(n_turns):
        msgs.append(_msg("user", f"question {i}"))
        msgs.append(_msg("subliminal", "<context>injected</context>"))
        msgs.append(_msg("thinking", "", reasoning="private thoughts"))
        msgs.append(_msg("assistant", "", tool_calls=[
            {"type": "function", "function": {"name": "Bash", "arguments": "{}"}}]))
        msgs.append(_msg("tool", "TOOL OUTPUT SHOULD NOT APPEAR"))
        msgs.append(_msg("assistant", f"{reply} {i}"))
    return {"session_id": SID, "title": "a chat", "created_at": "2026-11-09T18:16:24",
            "platform": "mission-control", "messages": msgs}


@pytest.fixture()
def sessions(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", d)
    return d


def test_it_returns_the_conversation_and_nothing_else(sessions):
    (sessions / f"{SID}.json").write_text(json.dumps(_session()))
    out = session_mod._session_read({"session_id": SID})
    assert out["total_turns"] == 3 and not out["truncated"]
    t0 = out["turns"][0]
    assert t0 == {"turn": 0, "user": "question 0", "lloyd": "reply 0", "tools": ["Bash"]}
    blob = json.dumps(out)
    for leaked in ("injected", "private thoughts", "TOOL OUTPUT"):
        assert leaked not in blob


def test_a_turn_range_and_a_gzipped_file(sessions):
    with gzip.open(sessions / f"{SID}.json.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_session(5)))
    out = session_mod._session_read({"session_id": SID, "start_turn": 2, "end_turn": 3})
    assert [t["turn"] for t in out["turns"]] == [2, 3]


def test_the_cap_pages_with_next_start_turn(sessions, monkeypatch):
    monkeypatch.setattr(session_mod, "SESSION_READ_MAX_CHARS", 200)
    (sessions / f"{SID}.json").write_text(json.dumps(_session(6, reply="x" * 60)))
    out = session_mod._session_read({"session_id": SID})
    assert out["truncated"] and out["next_start_turn"] == len(out["turns"])
    more = session_mod._session_read({"session_id": SID, "start_turn": out["next_start_turn"]})
    assert more["turns"][0]["turn"] == out["next_start_turn"]


def test_an_export_path_is_accepted_and_traversal_is_refused(sessions):
    (sessions / f"{SID}.json").write_text(json.dumps(_session(1)))
    out = session_mod._session_read({"session_id": f"sessions/2026-11-09/{SID}.md"})
    assert out["session_id"] == SID and out["turns"]
    for bad in ("../secrets", "a/b", ".hidden"):
        assert session_mod._session_read({"session_id": bad}).get("code") == "INVALID_PARAM"
    assert session_mod._session_read({"session_id": "20991231_000000_none"})["code"] == "NOT_FOUND"


def test_it_is_registered_read_only():
    names = [t.name for t in asyncio.run(session_mod.list_tools())]
    assert "session_read" in names
    assert "session_read" in annotations.READ_ONLY


def test_vault_read_on_a_session_export_points_at_session_read(monkeypatch, tmp_path):
    monkeypatch.setattr(vault, "VAULT", tmp_path)
    out = vault._vault_read({"path": f"sessions/2026-11-09/{SID}.md"})
    assert out["code"] == "NOT_FOUND"
    assert "session_read" in out["error"] and SID in out["error"]


def test_a_recall_session_hit_carries_its_id_for_the_model():
    res = vault._model_facing_recall({"documents": [
        {"path": f"sessions/2026-11-09/{SID}.md", "title": "", "snippet": "x", "score": 1},
        {"path": "knowledge/a.md", "title": "", "snippet": "y", "score": 0.5}]})
    hit, other = res["documents"]
    assert hit["session_id"] == SID and hit["open_with"] == "session_read"
    assert "session_id" not in other
