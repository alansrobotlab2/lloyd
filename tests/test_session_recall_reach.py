"""`session_recall` finds the right chats, cheaply, and never the caller's own.

Pinned (2026-11-10 review):
  * a background run is skipped on its NAME, before a byte of it is read — they
    outnumbered chats ~200:1 in the window and each cold index parsed them all;
  * a retention-swept `<id>.json.gz` chat is still searchable;
  * a platform nobody classified (`canary`) is not a conversation (allow-list);
  * a query token matches at the start of a word, so `ai` no longer hits `said`;
  * the calling session is never recalled back to itself;
  * the handler runs off the aggregator's event loop.
"""
from __future__ import annotations

import asyncio
import datetime
import gzip
import json
import threading
from pathlib import Path

import pytest

from agent_mcp import _task_registry
from agent_mcp import session as session_mod

TODAY = datetime.datetime.now().strftime("%Y%m%d")


def _data(sid: str, text: str, **extra) -> dict:
    d = {"session_id": sid, "created_at": "2026-11-10T12:00:00", "model": "primary",
         "preview": text[:40], "message_count": 2,
         "messages": [{"role": "user", "content": [{"type": "text", "text": text}]},
                      {"role": "assistant", "content": [{"type": "text", "text": "noted, " + text}]}]}
    d.update(extra)
    return d


def _write(d: Path, sid: str, text: str, gz: bool = False, **extra) -> None:
    body = json.dumps(_data(sid, text, **extra))
    if gz:
        with gzip.open(d / f"{sid}.json.gz", "wt", encoding="utf-8") as fh:
            fh.write(body)
    else:
        (d / f"{sid}.json").write_text(body, encoding="utf-8")


@pytest.fixture()
def sessions(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", d)
    monkeypatch.setattr(session_mod, "_session_index_cache", None)
    monkeypatch.setattr(session_mod, "SESSION_RECALL_BACKEND", "tokens")
    return d


def _ids(out: dict) -> list[str]:
    return [s["session_id"] for s in out["sessions"]]


def test_a_background_run_is_skipped_without_being_read(sessions, monkeypatch):
    _write(sessions, f"{TODAY}_120000_chat1", "the gearbox backlash measurement")
    _write(sessions, f"{TODAY}_120001_autocode_ab12", "the gearbox backlash measurement",
           platform="worker")
    read: list[str] = []
    real = session_mod._read_session_json
    monkeypatch.setattr(session_mod, "_read_session_json",
                        lambda p: read.append(p.name) or real(p))
    out = session_mod._session_recall({"query": "gearbox backlash"})
    assert _ids(out) == [f"{TODAY}_120000_chat1"]
    assert read == [f"{TODAY}_120000_chat1.json"]


def test_a_gzipped_chat_is_still_found(sessions):
    _write(sessions, f"{TODAY}_090000_old", "the servo calibration offsets", gz=True)
    out = session_mod._session_recall({"query": "servo calibration"})
    assert _ids(out) == [f"{TODAY}_090000_old"]


def test_an_unclassified_platform_is_not_a_conversation(sessions):
    _write(sessions, f"{TODAY}_100000_smoke", "the gearbox backlash", platform="canary")
    assert _ids(session_mod._session_recall({"query": "gearbox backlash"})) == []


def test_a_token_matches_at_a_word_start_only(sessions):
    _write(sessions, f"{TODAY}_110000_said", "she said it again and again, plainly")
    _write(sessions, f"{TODAY}_110001_ai", "the AI model rollbacks were fine")
    out = session_mod._session_recall({"query": "ai"})
    assert _ids(out) == [f"{TODAY}_110001_ai"]
    # Inflections still match: the start of the word is anchored, not the end.
    out = session_mod._session_recall({"query": "rollback"})
    assert _ids(out) == [f"{TODAY}_110001_ai"]


def test_the_calling_session_is_not_recalled_to_itself(sessions):
    own = f"{TODAY}_130000_me"
    _write(sessions, own, "what did we decide about the dryer vent")
    _write(sessions, f"{TODAY}_080000_earlier", "we decided the dryer vent goes left")
    tok = _task_registry.current_session_id.set(own)
    try:
        out = session_mod._session_recall({"query": "dryer vent"})
    finally:
        _task_registry.current_session_id.reset(tok)
    assert _ids(out) == [f"{TODAY}_080000_earlier"]


def test_the_handler_runs_off_the_event_loop(sessions, monkeypatch):
    seen: list = []
    monkeypatch.setattr(session_mod, "_session_recall",
                        lambda a: seen.append(threading.current_thread()) or {"sessions": []})

    async def go():
        await session_mod.call_tool("session_recall", {"query": "x"})
        return threading.current_thread()

    loop_thread = asyncio.run(go())
    assert seen and seen[0] is not loop_thread
