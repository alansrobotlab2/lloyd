"""Speaking a voice turn's reply while it streams.

The old worker waited for the whole answer, then for the secondary model to
rewrite it, then for a 500 ms transcript poll to notice it. These pin what the
streaming path has to get right instead — each one a way it could go wrong
without anything raising:

  * clauses spoken in the order written, the first one before the reply ends;
  * the turn registered before its text can be persisted, or the session
    poller (which still speaks typed turns) says the reply a second time;
  * a filler only for silence, never on top of an answer;
  * nothing more spoken once the listener has interrupted;
  * a backend that predates streaming answered with JSON, and left to the
    poller rather than treated as an empty reply.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

livekit_worker = pytest.importorskip("livekit_worker")


def _sse(*events):
    lines = []
    for name, data in events:
        lines += [f"event: {name}", f"data: {json.dumps(data)}", ""]
    return lines


class _Resp:
    def __init__(self, lines, content_type="text/event-stream", status=200,
                 gate=None, open_delay=0.0):
        self._lines = lines
        self.headers = {"content-type": content_type}
        self.status_code = status
        self._gate = gate
        self._open_delay = open_delay

    async def __aenter__(self):
        # The backend runs prefetch before it answers, so the headers
        # themselves can be late.
        await asyncio.sleep(self._open_delay)
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            if self._gate is not None and line == "GATE":
                await self._gate()
                continue
            yield line
            await asyncio.sleep(0)

    async def aread(self):
        return b"{}"


class _HTTP:
    def __init__(self, resp):
        self.resp = resp
        self.sent = None

    def stream(self, method, url, json=None, timeout=None):
        self.sent = json
        return self.resp


class _TTS:
    """Records what would be said, and honours the generation counter."""

    is_speaking = False
    is_paused = False
    is_idle = True

    def __init__(self):
        self.said = []
        self.generation = 0
        self.thinking = 0

    async def speak(self, text, timeline=None):
        self.said.append(text)

    def interrupt(self):
        self.generation += 1
        return 0

    def silence_s(self, now=None):
        return 0.0

    def start_thinking(self):
        self.thinking += 1

    async def stop_thinking(self):
        pass


def _bridge(resp, **voice_turn):
    async def make():
        b = livekit_worker.RoomBridge(
            "lloyd-20260917_000000_vtst",
            {"room_prefix": "lloyd-", "voice_turn": voice_turn},
            stt=None, vad_cfg={}, http_client=_HTTP(resp))
        b.tts = _TTS()
        return b
    return asyncio.run(make())


def _run(bridge, payload=None):
    asyncio.run(bridge._speak_voice_turn(payload or {"text": "hi",
                                                     "session_key": "s"}))
    return bridge.tts.said


def test_the_reply_is_spoken_clause_by_clause_in_order():
    lines = _sse(("voice_turn", {"turn_id": "t1"}),
                 ("text_delta", {"text": "It is seven "}),
                 ("text_delta", {"text": "thirty. Your meeting "}),
                 ("text_delta", {"text": "is at eight."}),
                 ("done", {"response": "…"}))
    b = _bridge(_Resp(lines), filler={"enabled": False})
    said = _run(b)
    assert said == ["It is seven thirty.", "Your meeting is at eight."]
    assert b.http.sent["stream"] is True


def test_the_turn_is_registered_so_the_poller_skips_its_rows():
    lines = _sse(("voice_turn", {"turn_id": "abc"}),
                 ("text_delta", {"text": "Done."}), ("done", {}))
    b = _bridge(_Resp(lines), filler={"enabled": False})
    _run(b)
    assert "abc" in b._streamed_turns


def test_text_before_a_tool_call_is_said_before_the_tool_runs():
    lines = _sse(("voice_turn", {"turn_id": "t"}),
                 ("text_delta", {"text": "Let me check the calendar"}),
                 ("tool_start", {"name": "calendar_list"}),
                 ("text_delta", {"text": "You are free after four."}),
                 ("done", {}))
    b = _bridge(_Resp(lines), filler={"enabled": True, "after_seconds": 60})
    said = _run(b)
    assert said == ["Let me check the calendar", "You are free after four."]


def test_a_tool_call_before_any_words_gets_one_filler():
    lines = _sse(("voice_turn", {"turn_id": "t"}),
                 ("tool_start", {"name": "a"}),
                 ("tool_start", {"name": "b"}),
                 ("text_delta", {"text": "All three builds passed."}),
                 ("done", {}))
    b = _bridge(_Resp(lines), filler={"enabled": True, "after_seconds": 60,
                                      "phrases": ["One moment."]})
    assert _run(b) == ["One moment.", "All three builds passed."]


def test_a_long_silence_gets_a_filler_but_an_answer_never_does():
    async def slow():
        await asyncio.sleep(0.25)

    lines = (_sse(("voice_turn", {"turn_id": "t"})) + ["GATE"] +
             _sse(("text_delta", {"text": "Here it is."}), ("done", {})))
    b = _bridge(_Resp(lines, gate=slow),
                filler={"enabled": True, "after_seconds": 0.1,
                        "phrases": ["One moment."]})
    assert _run(b) == ["One moment.", "Here it is."]

    quick = _sse(("voice_turn", {"turn_id": "t"}),
                 ("text_delta", {"text": "Here it is."}), ("done", {}))
    b = _bridge(_Resp(quick), filler={"enabled": True, "after_seconds": 0.1})
    assert _run(b) == ["Here it is."]


def test_nothing_more_is_spoken_after_an_interrupt():
    holder = {}

    async def interrupt():
        holder["b"].tts.interrupt()

    lines = (_sse(("voice_turn", {"turn_id": "t"}),
                  ("text_delta", {"text": "First sentence here. "})) + ["GATE"] +
             _sse(("text_delta", {"text": "Second sentence here."}), ("done", {})))
    b = _bridge(_Resp(lines, gate=interrupt), filler={"enabled": False})
    holder["b"] = b
    assert _run(b) == ["First sentence here."]


def test_a_runaway_reply_is_capped_with_a_pointer_to_the_chat():
    text = "This is a sentence of moderate length. " * 40
    lines = _sse(("voice_turn", {"turn_id": "t"}),
                 ("text_delta", {"text": text}), ("done", {}))
    b = _bridge(_Resp(lines), filler={"enabled": False}, max_spoken_chars=200)
    said = _run(b)
    assert said[-1] == "The rest is in the chat."
    assert sum(len(c) for c in said[:-1]) <= 200


def test_skipped_code_is_pointed_at():
    lines = _sse(("voice_turn", {"turn_id": "t"}),
                 ("text_delta", {"text": "Here is the fix.\n```\nx = 1\n```\n"}),
                 ("done", {}))
    b = _bridge(_Resp(lines), filler={"enabled": False})
    assert _run(b) == ["Here is the fix.", "I've put the code in the chat."]


def test_the_filler_clock_starts_at_the_question_not_at_the_headers():
    # 2026-09-18: the clock started only once the stream opened, so time the
    # backend spent before answering was silence nobody filled.
    quick = _sse(("voice_turn", {"turn_id": "t"}),
                 ("text_delta", {"text": "Here it is."}), ("done", {}))
    b = _bridge(_Resp(quick, open_delay=0.25),
                filler={"enabled": True, "after_seconds": 0.1,
                        "phrases": ["One moment."]})
    assert _run(b) == ["One moment.", "Here it is."]


def test_a_backend_without_streaming_is_left_to_the_poller():
    b = _bridge(_Resp([], content_type="application/json"),
                filler={"enabled": False})
    assert _run(b) == []
    assert not b._streamed_turns


def test_the_sse_parser_reads_named_events_and_skips_junk():
    async def collect():
        resp = _Resp(["event: a", 'data: {"x": 1}', "", "data: not json", "",
                      ": comment", "event: b", 'data: {"y": 2}', ""])
        return [e async for e in livekit_worker._iter_sse(resp)]

    assert asyncio.run(collect()) == [("a", {"x": 1}), ("message", {}),
                                      ("b", {"y": 2})]
