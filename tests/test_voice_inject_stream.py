"""`/api/voice/inject`: the stream the worker speaks from, and the two things a
spoken turn asks of the model.

A turn's events queue has exactly one reader. Until 2026-09-17 a spoken turn
had none — its events piled up unread while the worker re-fetched the whole
transcript every 500 ms and waited for the reply to finish. With `stream: true`
the caller that enqueued the turn reads that queue, as `/api/message/stream`
does for a typed turn.

The reminder is what replaced the secondary model's after-the-fact rewrite for
speech, and where it goes matters as much as what it says: in the text the
MODEL sees (the tail of the prompt, so the cached prefix is untouched), never
in the text the chat SHOWS, which must stay exactly what was said.
"""
import asyncio
import json

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.routers.voice as voice  # noqa: E402


@pytest.fixture
def harness(tmp_path, monkeypatch):
    captured = {}

    async def fake_enqueue(session_id, turn, consumer_factory=None):
        captured["turn"] = turn

        async def play():
            await turn.events.put({"event": "text_delta",
                                   "data": {"text": "It is noon."}})
            await turn.events.put({"event": "done", "data": {"response": "It is noon."}})
            await turn.events.put(None)

        asyncio.get_running_loop().create_task(play())

    async def fake_prefetch(text, **kw):
        return text

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(voice, "enqueue_turn", fake_enqueue)
    monkeypatch.setattr(voice, "prefetch_context_async", fake_prefetch)
    monkeypatch.setattr(voice, "build_system_prompt", lambda **kw: "SYS")
    monkeypatch.setattr(voice, "_save_session_meta", noop)
    monkeypatch.setattr(voice, "set_last_user_session", lambda s: None)
    monkeypatch.setattr(voice, "_get_mcp_servers", lambda: {})
    monkeypatch.setattr(voice, "_get_disallowed_tools", lambda **kw: [])
    monkeypatch.setattr(voice, "_get_harness_kwargs", lambda: {})
    monkeypatch.setattr(voice, "_resolve_model_name", lambda m: "primary")
    monkeypatch.setattr(voice, "_get_model_env", lambda m: {})
    monkeypatch.setattr(voice, "SESSIONS_DIR", tmp_path)

    def client(voice_turn=None):
        monkeypatch.setattr(voice, "CONFIG",
                            {"livekit": {"voice_turn": voice_turn or {}}})
        app = FastAPI()
        app.include_router(voice.router)
        return TestClient(app)

    return client, captured


def _frames(body: str):
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines.get("event"), json.loads(lines.get("data", "{}"))))
    return out


def test_stream_true_returns_the_turns_own_events_headed_by_its_id(harness):
    client, captured = harness
    r = client().post("/api/voice/inject",
                      json={"text": "what time is it", "session_key": "s1",
                            "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = _frames(r.text)
    assert frames[0] == ("voice_turn", {"session_id": "s1",
                                        "turn_id": captured["turn"].turn_id})
    assert [f[0] for f in frames[1:]] == ["text_delta", "done"]
    assert frames[1][1]["text"] == "It is noon."


def test_without_stream_the_answer_is_the_json_it_always_was(harness):
    client, captured = harness
    r = client().post("/api/voice/inject",
                      json={"text": "hi", "session_key": "s1"})
    assert r.json() == {"success": True, "session_id": "s1",
                        "turn_id": captured["turn"].turn_id}


def test_the_model_is_told_it_is_speaking_and_the_chat_is_not(harness):
    client, captured = harness
    client().post("/api/voice/inject", json={"text": "hello", "session_key": "s1"})
    payload = captured["turn"].payload
    assert payload["text"] == "hello", "the chat shows what was said, nothing more"
    assert payload["prefetched_text"].startswith(voice.VOICE_TURN_REMINDER)
    assert payload["prefetched_text"].endswith("hello")


def test_the_reminder_can_be_switched_off(harness):
    client, captured = harness
    client({"reminder": False}).post("/api/voice/inject",
                                     json={"text": "hello", "session_key": "s1"})
    assert captured["turn"].payload["prefetched_text"] == "hello"


@pytest.mark.parametrize("setting,expected", [
    ("off", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("on", {}),
    (None, {}),
])
def test_thinking_follows_config(harness, setting, expected):
    client, captured = harness
    cfg = {} if setting is None else {"thinking": setting}
    client(cfg).post("/api/voice/inject", json={"text": "hi", "session_key": "s1"})
    assert captured["turn"].payload["options"].extra_body == expected


# ── /api/voice/prewarm ───────────────────────────────────────────────────

@pytest.fixture
def prewarm(harness, monkeypatch, tmp_path):
    client, _ = harness
    voice._PREWARM_LAST.clear()
    runs = []

    async def fake_run_query(messages, options):
        runs.append((messages, options))
        if False:
            yield None

    import app.harness as harness_mod
    monkeypatch.setattr(harness_mod, "run_query", fake_run_query)
    return client, runs


def _drain_tasks():
    """The endpoint answers first and prefills in a background task; give it
    a turn of the loop TestClient runs."""
    import time
    time.sleep(0.2)


def test_prewarm_sends_the_turns_own_prefix_for_one_token(prewarm, tmp_path, monkeypatch):
    client, runs = prewarm
    (tmp_path / "s1.json").write_text(json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "earlier"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    ]}))
    r = client().post("/api/voice/prewarm", json={"session_key": "s1"})
    assert r.json() == {"started": True}
    _drain_tasks()
    assert len(runs) == 1
    messages, options = runs[0]
    assert messages[-1] == {"role": "user", "content": "."}
    assert messages[0]["role"] == "user" and messages[1]["role"] == "assistant"
    assert options.extra_body["max_tokens"] == 1
    assert options.system_prompt == "SYS", "the same system prompt the turn sends"
    assert options.session_id == "s1" and options.priority == 0


def test_prewarm_is_debounced_per_session(prewarm):
    client, runs = prewarm
    c = client()
    assert c.post("/api/voice/prewarm", json={"session_key": "s2"}).json()["started"]
    assert c.post("/api/voice/prewarm", json={"session_key": "s2"}).json() == {
        "started": False, "reason": "recent"}
    assert c.post("/api/voice/prewarm", json={"session_key": "s3"}).json()["started"]


def test_prewarm_has_a_switch(prewarm):
    client, runs = prewarm
    r = client({"prewarm": False}).post("/api/voice/prewarm", json={"session_key": "s4"})
    assert r.json() == {"started": False, "reason": "disabled"}


def test_prewarm_skips_a_session_that_will_summarize(prewarm, tmp_path, monkeypatch):
    """Over the compaction threshold the turn summarizes, so its prefix will
    differ — and a prewarm must not spend a summarization call on a wake that
    may have been a false alarm. It loads with `truncate`, which never calls a
    model, and stands down if that had to drop anything."""
    client, runs = prewarm
    (tmp_path / "s5.json").write_text(json.dumps({"messages": []}))
    seen = {}

    async def fake_load(path, model="", system_prompt="", *, mode_override=None):
        seen["mode"] = mode_override
        return {"history": [], "truncated": True}

    import app.compaction as compaction
    monkeypatch.setattr(compaction, "load_and_compact_session", fake_load)
    client().post("/api/voice/prewarm", json={"session_key": "s5"})
    _drain_tasks()
    assert seen["mode"] == "truncate"
    assert runs == []
