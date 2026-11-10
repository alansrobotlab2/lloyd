"""Three things about what a turn searches with and what it is handed (2026-11-10).

* a voice turn searches without its `[Speaker]: ` tag, while the model still
  reads it (the tag cut merged doc_hit 0.210 -> 0.148 on the 81-query set);
* `<ide_state>` — the human's open files — reaches chat turns, not workers;
* a `<now>` clock line opens the turn tail and is split off as tail, not text.
"""
from __future__ import annotations

import pytest

from app import prefetch, prompt_layout
from app.routers import voice
from app.routers._messages_subliminal import _split_subliminal


def test_speaker_tags_are_stripped_from_every_line():
    assert voice.strip_speaker_tags("[Alan]: what port is qmd on") == "what port is qmd on"
    assert voice.strip_speaker_tags("[Alan]: first\n[Alan]: second") == "first\nsecond"
    assert voice.strip_speaker_tags("no tag [here]: stays") == "no tag [here]: stays"


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(prefetch, "_search_skills_for_turn", lambda *a: [])
    monkeypatch.setattr(prefetch, "_search_facts", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_recent_sessions", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_backlog_refs", lambda t: [])


def test_query_text_is_searched_and_the_message_is_shown(quiet, monkeypatch):
    seen: list[str] = []

    def _daemon(query, *a, **kw):
        seen.append(query)
        return [{"file": "qmd://obsidian/a.md", "title": "A", "score": 0.9,
                 "snippet": "qmd listens on 8181"}]
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _daemon)
    monkeypatch.setattr(prefetch, "_format_ide_state", lambda: "")
    msg = "[Alan]: which port does the qmd daemon listen on"
    out = prefetch.prefetch_context(msg, session_id="t-voice-q", plan_mode=False,
                                    query_text=voice.strip_speaker_tags(msg))
    assert out.endswith("\n\n" + msg), "the model must still read who spoke"
    assert seen and not any("alan" in q.lower() for q in seen), seen


def test_ide_state_reaches_a_chat_turn_but_not_a_worker(quiet, monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", lambda *a, **kw: [])
    monkeypatch.setattr(prefetch, "_format_ide_state",
                        lambda: "<ide_state>\n  visible_file: app/prefetch.py\n</ide_state>")
    q = "explain what the budgeted half of prefetch does"
    chat = prefetch.prefetch_context(q, session_id="20261110_101112_abc123", plan_mode=False)
    worker = prefetch.prefetch_context(q, session_id="20261110_101112_autocode_0a1b",
                                       plan_mode=False)
    assert "<ide_state>" in chat
    assert "<ide_state>" not in worker


def test_clock_line_opens_the_tail_and_is_split_off(monkeypatch):
    monkeypatch.setattr(prompt_layout, "clock_enabled", lambda: True)
    tail = prompt_layout.turn_tail()
    assert tail.startswith("<now>") and tail.endswith("</now>")
    text = "what did we land yesterday?"
    sent = prompt_layout.append_turn_tail(text, tail)
    _prefix, got_tail = _split_subliminal(sent, text)
    assert got_tail == tail


def test_clock_off_leaves_the_tail_empty(monkeypatch):
    monkeypatch.setattr(prompt_layout, "clock_enabled", lambda: False)
    monkeypatch.setattr(prompt_layout._cfg.__globals__["logger"], "disabled", False)
    from app import prompt_builder as pb
    if pb.session_state_layout() == "user_tail":
        pytest.skip("session state rides in the tail on this config")
    assert prompt_layout.turn_tail() == ""


def test_a_session_hit_names_its_id_and_session_read():
    out = prefetch._format_context(
        [], [], [{"title": "20261109_181624_ivd095", "score": 0.8,
                  "file": "qmd://sessions/2026-11-09/20261109_181624_ivd095.md",
                  "snippet": "we decided to keep djev on GPU 2"}], [])
    assert "session_id: 20261109_181624_ivd095" in out
    assert "session_read" in out
    assert "file: sessions/" not in out, "vault_read cannot open that path"


def test_the_manifest_digest_does_not_move_with_the_clock():
    text = "block\n\nhello"
    for minute in ("10:56", "10:57"):
        sent = prompt_layout.append_turn_tail(text, f"<now>Tue 2026-11-10 {minute} PST</now>")
        assert prompt_layout.without_clock(sent) == text
    assert prompt_layout.without_clock(text) == text
