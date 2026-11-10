"""A chat's qmd export follows the conversation, not just its first turn.

Before 2026-11-10 `_export_session_markdown` ran only inside
`_capture_summary_once`, which the `captured` latch limits to turn 1, so the
sessions collection held each chat's opening and nothing after it (47 of 106
exports with a surviving JSON were short; 163 of 301 user turns unsearchable).
"""
import json

import pytest

from app import post_capture


SID = "20261110_101112_abc123"


def _session(turns: list[tuple[str, str]], captured: bool) -> dict:
    msgs = []
    for u, a in turns:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return {"session_id": SID, "platform": "mission-control",
            "created_at": "2026-11-10T10:11:12", "captured": captured,
            "messages": msgs}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions-json"
    sessions.mkdir()
    chat = tmp_path / "sessions"
    monkeypatch.setattr(post_capture, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(post_capture, "VAULT_SESSIONS_DIR", chat)
    monkeypatch.setattr(post_capture, "VAULT_BACKGROUND_SESSIONS_DIR", tmp_path / "bg")

    async def _no_facts(_sid):
        return 0
    monkeypatch.setattr(post_capture, "_extract_facts_past_watermark", _no_facts)
    return sessions, chat


@pytest.mark.anyio
async def test_a_captured_chat_is_re_exported_with_its_later_turns(corpus):
    sessions, chat = corpus
    first = [("which port does the aggregator listen on?", "It listens on 8500, under /mcp.")]
    (sessions / f"{SID}.json").write_text(json.dumps(_session(first, captured=False)))
    post_capture._export_session_markdown(SID, _session(first, captured=False))

    later = first + [("and what restarts the backend safely?",
                      "The round CLI restart, which drains the worker pool first.")]
    (sessions / f"{SID}.json").write_text(json.dumps(_session(later, captured=True)))
    await post_capture._post_session_capture(SID)

    [out] = list(chat.rglob("*.md"))
    text = out.read_text()
    assert "what restarts the backend safely" in text, (
        "a turn after the capture latch never reached the export qmd indexes")


def test_an_unchanged_export_is_not_rewritten(corpus, tmp_path):
    _sessions, chat = corpus
    data = _session([("which port does the aggregator listen on?",
                      "It listens on 8500, under /mcp.")], captured=True)
    path = post_capture._export_session_markdown(SID, data)
    before = path.stat().st_mtime_ns
    import os
    os.utime(path, ns=(before - 10_000_000_000, before - 10_000_000_000))
    stamped = path.stat().st_mtime_ns
    post_capture._export_session_markdown(SID, data)
    assert path.stat().st_mtime_ns == stamped, (
        "an identical re-export rewrote the file and would wake qmd-watcher")
