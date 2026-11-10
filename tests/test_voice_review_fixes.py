"""The 2026-11-09 voice review: each defect it found, as a test that failed first.

The review's diagnosis of why these escaped was the fakes: a source whose
`clear_queue` only counted, a TTS with no drain, a timeline built by hand. So
these tests drive the REAL `TTSStreamer` (its drain loop, its pause/resume,
its HTTP stream) over a source that really drops what `clear_queue` clears, and
the real `_handle_utterance_event → _speak_voice_turn → ReplySpeaker` path.

  Phase 0  the latency row times the answer, not the filler; a released hold
           is dated from when the speaker stopped; engine facts reach the row.
  C1       a false barge-in's resume replays the frames it took back.
  C2       `TTSStreamer.close()` returns while a clause is mid-stream.
  C3       "heard" is what had played when the reply paused.
  C4       a failed clause is not logged as heard, and is retried once on
           the fallback TTS.
  C5/C6    "stop, that's all" leaves the window closed; an ambient reply never
           reopens it; Lloyd's last words are the last ones heard.
  C7/C8    a barge-in cancel names its turn; a typed turn read aloud is never
           cancelled by a word in the room.
  C9       Lloyd's own voice is dropped before the hold, never held.
  C10      nothing slow runs on the event loop: hearing construction, the
           diagnostic WAV, the hearing thread's join, the transcript poll.
  C11      a backend error is spoken, and a stalled reply apologises and lets go.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

livekit_worker = pytest.importorskip("livekit_worker")
from voice.asr import Transcript  # noqa: E402
from voice.pipeline import HearingEvent  # noqa: E402
from voice.timeline import STAGES, TurnTimeline  # noqa: E402
from voice.turn import TurnVerdict  # noqa: E402
from voice.vad import Utterance  # noqa: E402

SR = 24000
FRAME = SR // 10            # 100 ms
WORDS = ["lloyd", "hey lloyd"]
ROOM = "lloyd-20261109_000000_rvw"


# ── doubles that behave like the real things ──────────────────────────────

class _DropSource:
    """An AudioSource whose `clear_queue` really drops what was queued."""

    queued_duration = 0.0

    def __init__(self):
        self.live: list[bytes] = []
        self.dropped: list[bytes] = []

    async def capture_frame(self, frame):
        self.live.append(bytes(frame.data))

    def clear_queue(self):
        self.dropped += self.live
        self.live = []

    async def wait_for_playout(self):
        return None


class _TTSResp:
    def __init__(self, chunks=(), status=200, first_delay=0.0, hang=False):
        self.status_code = status
        self._chunks = list(chunks)
        self._first_delay = first_delay
        self._hang = hang

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aread(self):
        return b"engine error"

    async def aiter_bytes(self):
        if self._first_delay:
            await asyncio.sleep(self._first_delay)
        for c in self._chunks:
            yield c
            await asyncio.sleep(0)
        if self._hang:
            await asyncio.sleep(3600)


class _TTSHTTP:
    """`stream()` answers per base URL; records every request."""

    def __init__(self, by_url):
        self.by_url = by_url
        self.calls: list[str] = []

    def stream(self, method, url, json=None, **kw):
        base = url.split("/v1/")[0]
        self.calls.append(base)
        make = self.by_url[base]
        return make() if callable(make) else make

    async def aclose(self):
        pass


def _pcm(frames=1):
    return (np.full(FRAME * frames, 1000, dtype=np.int16)).tobytes()


def _streamer(by_url=None, **cfg):
    conf = {"api_url": "http://primary", "sample_rate": SR, "tail_silence_ms": 0,
            "shaping": {"presence_eq": False}}
    conf.update(cfg)
    st = livekit_worker.TTSStreamer(conf, room=None)
    st.source = _DropSource()
    st.track = object()                       # published: ensure_published is a no-op
    st._http = _TTSHTTP(by_url or {"http://primary": lambda: _TTSResp([_pcm()])})
    st._worker_task = asyncio.get_running_loop().create_task(st._drain())
    return st


async def _until(pred, timeout=3.0):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.01)


class _STT:
    name = "fake"

    def __init__(self, texts):
        self._texts = list(texts)

    def transcribe(self, audio):
        return Transcript(self._texts.pop(0) if self._texts else "", 0.01, "fake")


class _SSE:
    def __init__(self, events, *, raise_after=None):
        self._events = events
        self.headers = {"content-type": "text/event-stream"}
        self.status_code = 200
        self._raise_after = raise_after

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aread(self):
        return b""

    async def aiter_lines(self):
        for item in self._events:
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
                continue
            name, data = item
            for line in (f"event: {name}", f"data: {json.dumps(data)}", ""):
                yield line
        if self._raise_after is not None:
            raise self._raise_after


class _Backend:
    """The backend as the worker sees it: inject stream, queue, cancel, posts."""

    def __init__(self, events=(), current_turn=None):
        self.events = list(events)
        self.current_turn = current_turn
        self.posts: list[tuple[str, dict]] = []
        self.cancels: list[dict] = []
        self.gets: list[str] = []
        self.raise_after = None

    def stream(self, method, url, json=None, timeout=None):
        return _SSE(self.events, raise_after=self.raise_after)

    async def post(self, url, json=None, timeout=None):
        if url.endswith("/cancel"):
            self.cancels.append(json or {})
        else:
            self.posts.append((url, json))
        return type("R", (), {"status_code": 200, "text": ""})()

    async def get(self, url, timeout=None):
        self.gets.append(url)
        cur = {"turn_id": self.current_turn} if self.current_turn else None
        return type("R", (), {"status_code": 200,
                              "json": lambda self_: {"current": cur, "messages": []}})()


def _bridge(texts=(), backend=None, **extra):
    cfg = {"room_prefix": "lloyd-",
           "wake": {"words": WORDS, "continuation_seconds": 6.0},
           "turn_detection": {"hold_timeout_ms": 60000},
           "conversation": {"enabled": True, "idle_close_s": 90, "addressee": "off"},
           "barge_in": {"enabled": False}}
    cfg.update(extra)
    return livekit_worker.RoomBridge(ROOM, cfg, stt=_STT(texts), vad_cfg={},
                                     http_client=backend or _Backend())


def _utt(seconds=0.6, reason="silence"):
    audio = np.zeros(int(seconds * 16000), dtype=np.float32)
    return Utterance(audio=audio, start_sample=0, end_sample=audio.size,
                     max_prob=0.9, reason=reason)


async def _settle(b):
    for _ in range(300):
        pending = [t for t in list(b._utterance_tasks) + list(b._turn_tasks) if not t.done()]
        if not pending:
            return
        await asyncio.sleep(0.02)


# ── Phase 0: the row times the answer ──────────────────────────────────────

def test_a_filler_is_first_sound_and_the_row_times_the_answer(monkeypatch):
    """The production path, end to end, with the filler firing before the model
    writes anything: the row's `eos_to_answer` is the answer's playout, its
    `eos_to_audio` the filler's, and no chain delta is negative (19 of the first
    28 production rows had a negative `first_tts_byte`)."""
    rows = []
    monkeypatch.setattr(livekit_worker.voice_turns, "append_turn",
                        lambda row, **kw: rows.append(row) or True)
    backend = _Backend([
        ("voice_turn", {"turn_id": "t1", "queued_behind": False, "setup_ms": 420,
                        "prewarm": {"result": "ok", "age_s": 9.0, "trigger": "speech"}}),
        ("session", {}),
        0.8,                                    # the model thinks; the filler fires
        ("turn_stats", {"prompt_tokens": 81000, "cached_tokens": 80500,
                        "ttft_ms": 340, "queue_wait_s": 0.0}),
        ("text_delta", {"text": "It is seven. "}),
        ("done", {}),
    ])

    async def go():
        b = _bridge(["What time is it?"], backend=backend,
                    voice_turn={"filler": {"after_seconds": 0.1}, "thinking_sound": False})
        b.tts = _streamer()
        b.wake.extend("user-a")
        b.wake.open_conversation("user-a")
        await b._handle_utterance_event(HearingEvent("utterance", utterance=_utt()), "user-a")
        await _settle(b)
        await _until(lambda: rows, timeout=5.0)
        return b
    asyncio.run(go())
    row = rows[0]
    assert row["v"] == 2 and row["label"] == "voice"
    assert row["asides"] == 1, "One moment."
    assert row["eos_to_audio"] < row["eos_to_answer"]
    assert row["eos_to_answer"] - row["eos_to_audio"] > 0.3
    assert all(dt >= 0 for dt in row["stages"].values()), row["stages"]
    assert sum(row["stages"].values()) == pytest.approx(row["eos_to_answer"])
    assert set(row["stages"]) >= {"first_answer_byte", "first_answer_played"}
    assert row["engine"]["ttft_ms"] == 340 and row["engine"]["setup_ms"] == 420
    assert row["engine"]["prewarm"]["result"] == "ok"


def test_a_released_hold_is_dated_from_when_the_speaker_stopped(monkeypatch):
    """Held turns stamped `speech_end` at release, hiding the hold wait."""
    seen = []

    class _Unfinished:
        threshold = 0.5
        streaming = False

        def predict(self, audio):
            return TurnVerdict(complete=False, probability=0.1, elapsed_ms=1.0)

    async def go():
        b = _bridge(["So I was thinking"], voice_turn={"filler": {"enabled": False}})
        b.smart_turn = _Unfinished()
        b.tts = type("T", (), {"is_speaking": False, "is_paused": False,
                               "last_reply_end": None})()
        b.wake.extend("user-a")
        b.wake.open_conversation("user-a")

        async def capture(payload, tl):
            seen.append((payload, tl))
        b._speak_voice_turn = capture
        ev = HearingEvent("utterance", utterance=_utt())
        await b._handle_utterance_event(ev, "user-a")
        assert "user-a" in b._held
        await asyncio.sleep(0.3)               # the hold waits for a continuation
        b.stt._texts = ["So I was thinking"]
        b._release_held("user-a", "hold_timeout")
        await _settle(b)
        return ev.at
    closed = asyncio.run(go())
    tl = seen[0][1]
    assert tl.marks["speech_end"] == pytest.approx(closed - 0.38, abs=0.01)
    assert tl.facts["held_s"] >= 0.25
    assert tl.label == "voice"


# ── C1: resume replays what pause took back ───────────────────────────────

def test_resume_after_a_false_barge_in_replays_the_frames_it_took_back():
    async def go():
        st = _streamer()
        for i in range(5):
            await st._push_frame(bytes([i + 1]) * (2 * FRAME), FRAME)
        assert st.pause()
        taken = [pcm for pcm, _ in st._resume_frames]
        assert len(taken) == 5 and st.source.live == []
        assert await st.resume()
        return st, taken
    st, taken = asyncio.run(go())
    assert st.source.live == taken, "the replayed frames must survive in the source"
    assert not st.is_paused and st._resume_frames == []


# ── C2: close() mid-clause ────────────────────────────────────────────────

@pytest.mark.parametrize("where", ["streaming", "paused"])
def test_close_returns_while_a_clause_is_mid_stream(where):
    """Not `wait_for(close())`: its own timeout cancel is a second cancel, and
    that one frees the drain — which is how the first version of this test
    passed on the broken code. The worker manager awaits close() with no
    timeout, so close() itself has to return."""
    async def go():
        if where == "streaming":
            st = _streamer({"http://primary": lambda: _TTSResp([_pcm()], hang=True)})
        else:
            st = _streamer({"http://primary": lambda: _TTSResp([_pcm(30)])})
            st.pause()                     # the next push blocks on the pause
        await st.speak("A long sentence that never finishes.")
        await asyncio.sleep(0.2)
        closing = asyncio.create_task(st.close())
        done, _ = await asyncio.wait({closing}, timeout=1.0)
        stuck = closing not in done
        if stuck:                          # free it, so the test itself ends
            st._worker_task.cancel()
            await asyncio.wait({closing}, timeout=1.0)
        return st, stuck
    st, stuck = asyncio.run(go())
    assert not stuck, "close() hung on a drain that swallowed its cancel"
    assert st._worker_task.done()


# ── C3: heard at the pause, and the clause in flight is unheard ───────────

def test_heard_is_what_had_played_when_the_reply_paused():
    backend = _Backend(current_turn="t-old")

    async def go():
        b = _bridge(backend=backend, barge_in={"enabled": True})
        b._streamed_turns.append("t-old")
        b._speaking_turn = "t-old"
        st = _streamer()
        st._speaking.set()
        now = time.monotonic()
        st._clause_log = [("It is seven.", now - 0.5), ("Your meeting is at eight.", now - 0.1)]
        st.paused_at = now - 0.3
        st._unpaused.clear()
        b.tts = st
        await b._commit_barge_in("user-a", "test")
        return b
    b = asyncio.run(go())
    (url, note), = [p for p in backend.posts if p[0].endswith("/interrupted")]
    assert note["heard"] == "It is seven."
    assert note["unheard_chars"] >= len("Your meeting is at eight.")


def test_the_clause_being_synthesised_counts_as_unheard():
    async def go():
        st = _streamer({"http://primary": lambda: _TTSResp([_pcm()], hang=True)})
        await st.speak("This clause is still streaming.")
        await _until(lambda: st.source.live)
        unheard = st.unheard_chars()
        await st.close()
        return unheard
    assert asyncio.run(go()) >= len("This clause is still streaming.")


# ── C4: a failed clause, and the fallback ─────────────────────────────────

def test_a_tts_error_is_retried_once_on_the_fallback_and_then_heard():
    async def go():
        st = _streamer({"http://primary": lambda: _TTSResp(status=500),
                        "http://fallback": lambda: _TTSResp([_pcm(2)])},
                       fallback_url="http://fallback")
        await st.speak("Hello there.")
        await _until(lambda: not st.is_speaking and st._queue.empty() and st._http.calls)
        await _until(lambda: st.last_reply_end is not None)
        return st
    st = asyncio.run(go())
    assert st._http.calls == ["http://primary", "http://fallback"]
    assert len(st.source.live) >= 2


def test_a_primary_with_no_first_byte_falls_back_in_time():
    async def go():
        st = _streamer({"http://primary": lambda: _TTSResp([_pcm()], first_delay=10.0),
                        "http://fallback": lambda: _TTSResp([_pcm()])},
                       fallback_url="http://fallback", first_byte_timeout_s=0.2)
        t0 = time.monotonic()
        await st.speak("Hello there.")
        await _until(lambda: st.last_reply_end is not None)
        return st, time.monotonic() - t0
    st, took = asyncio.run(go())
    assert st._http.calls == ["http://primary", "http://fallback"]
    assert took < 2.0


def test_a_clause_neither_server_spoke_is_not_logged_as_heard():
    async def go():
        st = _streamer({"http://primary": lambda: _TTSResp(status=500),
                        "http://fallback": lambda: _TTSResp(status=503)},
                       fallback_url="http://fallback")
        log = []
        orig = st._drain
        await st.speak("Nobody heard this.")
        await _until(lambda: st.last_reply_end is not None)
        log = list(st._clause_log)
        return log, st
    log, st = asyncio.run(go())
    assert log == [], "a clause that produced no audio was never heard"
    assert st.heard_text() == ""


# ── C5 / C6: the window after a closer, and whose reply reopens it ────────

def test_stop_thats_all_mid_reply_leaves_the_window_closed():
    backend = _Backend(current_turn="t-old")

    async def go():
        b = _bridge(["Stop, that's all."], backend=backend, barge_in={"enabled": True})
        st = _streamer()
        st._speaking.set()
        b.tts = st
        b._streamed_turns.append("t-old")
        b._speaking_turn = b._active_turn = "t-old"
        b.wake.extend("user-a")
        b.wake.open_conversation("user-a")
        await b._handle_utterance_event(HearingEvent("utterance", utterance=_utt()), "user-a")
        b._reply_kind = "voice"
        b._on_tts_utterance_end()            # the interrupted reply's drain ends
        return b
    b = asyncio.run(go())
    assert not b.wake.in_continuation() and not b.wake.in_conversation()
    assert backend.cancels == [{"turn_id": "t-old"}]


def test_an_ambient_reply_never_reopens_a_closed_window():
    async def make():
        return _bridge()
    b = asyncio.run(make())
    b._schedule_wake_state_publish = lambda: None
    b.wake.extend("user-a")
    b.wake._until = 0.0                       # the follow-up window expired
    b._reply_kind = "typed:ambient"
    b._on_tts_utterance_end()
    assert not b.wake.in_continuation()
    b._reply_kind = "voice"                   # a spoken turn's end still does
    b._on_tts_utterance_end()
    assert b.wake.in_continuation()


def test_lloyds_last_words_are_the_last_ones_synthesised_not_queued():
    async def go():
        b = _bridge()
        st = _streamer({"http://primary": lambda: _TTSResp([_pcm()], first_delay=0.3)},
                       )
        st.on_clause = b._on_clause_spoken
        b.tts = st
        b._last_agent_text = "Before."
        rs = livekit_worker.ReplySpeaker(b, turn_id="t1", label="voice", extras=False)
        await rs.say("Do you want the details?")
        queued = b._last_agent_text
        await _until(lambda: st.last_reply_end is not None)
        return queued, b._last_agent_text
    queued, heard = asyncio.run(go())
    assert queued == "Before.", "a queued clause is not yet something Lloyd said"
    assert heard == "Do you want the details?"


# ── C7 / C8: the cancel names its turn; typed turns are never cancelled ───

def test_the_cancel_names_the_turn_it_decided_on():
    backend = _Backend(current_turn="t-old")

    async def go():
        b = _bridge(backend=backend)
        b._streamed_turns.append("t-old")
        await b._cancel_turn("t-old", any_spoken=True)
    asyncio.run(go())
    assert backend.cancels == [{"turn_id": "t-old"}]


def test_a_typed_turn_being_read_aloud_stops_speaking_but_is_not_cancelled():
    backend = _Backend(current_turn="t-typed")

    async def go():
        b = _bridge(backend=backend, barge_in={"enabled": True})
        st = _streamer()
        st._speaking.set()
        b.tts = st
        b._streamed_turns.append("t-typed")
        b._tap_turns["t-typed"] = object()
        rs = livekit_worker.ReplySpeaker(b, turn_id="t-typed", label="typed:user",
                                         extras=False)
        await rs.say("Here is what I found.")   # a typed turn being read aloud
        await b._commit_barge_in("user-a", "words")
        return b
    b = asyncio.run(go())
    assert backend.cancels == [], "a word in the room never cancels a typed turn"
    assert b.tts.generation >= 1, "but the speech stops"


def test_the_backend_cancel_is_a_no_op_for_a_turn_that_is_not_running(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import sessions

    ev = asyncio.Event()
    monkeypatch.setattr(sessions, "get_cancel_event", lambda sid: ev)
    monkeypatch.setattr(sessions, "get_queue_state",
                        lambda sid: {"current": {"turn_id": "t-typed"}})
    app = FastAPI()
    app.include_router(sessions.router)
    client = TestClient(app)
    r = client.post("/api/sessions/s1/cancel", json={"turn_id": "t-spoken"})
    assert r.json()["cancelled"] is False and not ev.is_set()
    r = client.post("/api/sessions/s1/cancel", json={"turn_id": "t-typed"})
    assert r.json()["cancelled"] is True and ev.is_set()
    ev.clear()
    r = client.post("/api/sessions/s1/cancel")           # the browser's: unconditional
    assert r.json()["cancelled"] is True and ev.is_set()


# ── C9: own voice, before the hold ────────────────────────────────────────

def test_own_voice_is_dropped_before_the_hold_and_never_interrupts():
    class _Unfinished:
        threshold = 0.5
        streaming = False

        def predict(self, audio):
            return TurnVerdict(complete=False, probability=0.1, elapsed_ms=1.0)

    class _Spk:
        def identify(self, samples, sr):
            return "lloyd-voice", 0.9, np.ones(4) / 2

        def list_profiles(self):
            return [{"name": "lloyd-voice"}]

    class _PausedTTS:
        is_speaking = True
        is_paused = True
        generation = 0
        paused_at = None
        last_reply_end = None
        interrupts = resumes = 0

        def heard_text(self, at=None):
            return ""

        def unheard_chars(self, at=None):
            return 0

        def interrupt(self):
            self.interrupts += 1
            return 0

        async def resume(self):
            self.resumes += 1
            return True

    async def go():
        b = _bridge(["and that is how the build went"], barge_in={"enabled": True})
        b.smart_turn = _Unfinished()
        b.speaker_id = _Spk()
        b.tts = _PausedTTS()
        b.wake.extend("user-a")
        b.wake.open_conversation("user-a")
        b._barge = {"identity": "user-a", "paused_at": time.monotonic(), "silence_at": None}
        await b._handle_utterance_event(HearingEvent("utterance", utterance=_utt()), "user-a")
        await asyncio.sleep(0.05)
        return b
    b = asyncio.run(go())
    assert "user-a" not in b._held, "an echo is never held to be glued to a real turn"
    assert b.tts.interrupts == 0, "and never interrupts Lloyd"
    assert b.tts.resumes == 1, "the pause it caused is resolved as false"


# ── C10: nothing slow on the event loop ───────────────────────────────────

async def _loop_stall_during(coro_factory):
    """The longest gap between 10 ms ticks while `coro_factory()` runs."""
    worst = 0.0
    stop = False

    async def ticker():
        nonlocal worst
        last = time.monotonic()
        while not stop:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            worst = max(worst, now - last)
            last = now
    t = asyncio.create_task(ticker())
    await asyncio.sleep(0.02)
    await coro_factory()
    stop = True
    await t
    return worst


def test_the_diagnostic_recording_does_not_block_the_loop():
    class _SlowCapture:
        def record_utterance(self, **kw):
            time.sleep(0.3)

    async def go():
        b = _bridge()
        b.wake_capture = _SlowCapture()
        ev = HearingEvent("utterance", utterance=_utt())
        return await _loop_stall_during(
            lambda: b._record_diag_async("user-a", _utt().audio, ev, "hi", 0.1, True))
    assert asyncio.run(go()) < 0.15


def test_building_a_hearing_pipeline_does_not_block_the_loop():
    class _SlowWake:
        threshold = 0.5

        def create(self):
            time.sleep(1.0)
            return None

    async def go():
        b = _bridge()
        b.wake_factory = _SlowWake()
        stall = await _loop_stall_during(lambda: b._build_hearing_async("user-a", 48000))
        for ht in b._hearing.values():
            ht.stop()
        return stall
    # On the loop the stall is the whole 1.0 s; off it, the loop's own jitter —
    # which under a loaded `-n` run reached 0.15 s, hence the wide margin.
    assert asyncio.run(go()) < 0.5


def test_the_utterance_rotation_keeps_the_newest_without_rescanning(tmp_path):
    cap = livekit_worker.WakeMissCapture(diag_dir=tmp_path)
    cap.MAX_UTTERANCE_FILES = 3
    for i in range(6):
        cap._save_utterance_wav(f"u{i}", np.zeros(160, dtype=np.int16), 16000)
    kept = sorted(p.stem for p in (tmp_path / "utterances").glob("*.wav"))
    assert kept == ["u3", "u4", "u5"]


def test_the_transcript_poll_is_idle_while_the_tap_carries_the_session():
    backend = _Backend()

    async def go():
        b = _bridge(backend=backend)
        b.SESSION_POLL_INTERVAL = 0.02
        b._tap_connected = True
        task = asyncio.create_task(b._poll_session_messages())
        await asyncio.sleep(0.2)
        task.cancel()
        return b
    asyncio.run(go())
    assert [u for u in backend.gets if "/api/messages/" in u] == []


# ── C11: errors spoken, stalls apologised for ─────────────────────────────

class _SpeakTTS:
    is_paused = False
    is_speaking = False

    def __init__(self):
        self.said = []
        self.generation = 0

    @property
    def is_idle(self):
        return True

    async def speak(self, text, timeline=None, aside=False):
        self.said.append(text)

    def silence_s(self, now=None):
        return 0.0

    def start_thinking(self):
        pass

    async def stop_thinking(self):
        pass


def test_a_backend_error_is_spoken():
    backend = _Backend([("voice_turn", {"turn_id": "t1"}), ("session", {}),
                        ("error", {"detail": "engine unavailable"})])

    async def go():
        b = _bridge(backend=backend, voice_turn={"filler": {"enabled": False}})
        b.tts = _SpeakTTS()
        await b._speak_voice_turn({"text": "hi", "session_key": "s"})
        return b
    b = asyncio.run(go())
    assert b.tts.said and "sorry" in b.tts.said[-1].lower()


def test_a_dropped_stream_is_spoken():
    backend = _Backend([("voice_turn", {"turn_id": "t1"}), ("session", {}),
                        ("text_delta", {"text": "Let me look"})])
    backend.raise_after = ConnectionError("backend went away")

    async def go():
        b = _bridge(backend=backend, voice_turn={"filler": {"enabled": False}})
        b.tts = _SpeakTTS()
        await b._speak_voice_turn({"text": "hi", "session_key": "s"})
        return b
    b = asyncio.run(go())
    assert any("sorry" in s.lower() for s in b.tts.said)
    assert b._active_turn is None


def test_a_reply_that_stalls_mid_sentence_apologises_and_lets_go():
    backend = _Backend([("voice_turn", {"turn_id": "t1"}), ("session", {}),
                        ("text_delta", {"text": "The build is "}), 1.0, ("done", {})])

    async def go():
        b = _bridge(backend=backend, voice_turn={"filler": {"enabled": False},
                                                 "stall_after_seconds": 0.3})
        b.tts = _SpeakTTS()
        task = asyncio.create_task(b._speak_voice_turn({"text": "hi", "session_key": "s"}))
        await asyncio.sleep(0.8)
        active = b._active_turn
        said = list(b.tts.said)
        await task
        return active, said
    active, said = asyncio.run(go())
    assert any("sorry" in s.lower() for s in said)
    assert active is None, "a stalled turn no longer blocks the next request as a barge-in"


# ── small ones ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, stop, close", [
    ("Stop, that's all.", True, True),
    ("Wait, never mind.", True, True),
    ("Okay, stop. Thanks.", True, True),
    ("Stop, what about tomorrow?", False, False),
    ("That's all I needed to know about the build.", False, False),
])
def test_a_stop_and_a_closer_said_together_are_both(text, stop, close):
    """"Stop, that's all." was neither, and went to the model as a request."""
    from voice import conversation as conv
    assert conv.is_stopper(text) is stop
    assert conv.is_closer(text) is close


def test_no_is_a_whole_sentence_not_an_abbreviation():
    from voice.speakable import ClauseStream
    cs = ClauseStream()
    out = cs.feed("No. That would delete the vault. ")
    assert out[0] == "No."


def test_the_turn_threshold_defaults_to_the_backends_own():
    from voice import turn
    assert turn.default_threshold("turn-1-mini") == pytest.approx(0.06)
    assert turn.default_threshold("smart-turn") == pytest.approx(0.5)


def test_the_orphan_cleanup_names_no_retired_service():
    text = (ROOT / "agent-services" / "bin" / "cleanup-orphans.sh").read_text()
    assert "tool_services" not in text and "voice_services" not in text


# ── Phase 2(c): a superseded spoken turn lets go at once ──────────────────

def test_a_superseded_queued_turn_ends_its_reply_quietly():
    """`/api/voice/inject` drops a spoken turn still queued when a newer
    request supersedes it; its stream says `superseded` and ends. Nothing of
    it is spoken — no apology either: nothing failed."""
    backend = _Backend([("voice_turn", {"turn_id": "t1", "queued_behind": True}),
                        ("superseded", {"turn_id": "t1"}),
                        ("text_delta", {"text": "never said."}), ("done", {})])

    async def go():
        b = _bridge(backend=backend, voice_turn={"filler": {"enabled": False},
                                                 "queued_notice": {"enabled": False}})
        b.tts = _SpeakTTS()
        await b._speak_voice_turn({"text": "hi", "session_key": "s"})
        return b
    b = asyncio.run(go())
    assert not any("never said" in s for s in b.tts.said)
    assert not any("sorry" in s.lower() for s in b.tts.said)
    assert b._active_turn is None


def test_inner_voice_judgment_does_not_hold_a_cancelled_turn():
    """2026-10-31 16:19:33: a spoken turn was cancelled and still ran 6.6 s —
    its terminal Inner Voice review awaited the observer LLM before looking at
    the cancel. The review now stops waiting when the turn is cancelled."""
    import time as _t
    from types import SimpleNamespace

    from app.inner_voice import observer

    async def slow_judgment():
        await asyncio.sleep(5.0)
        return observer.ObserverDecision(action="inject", content="late")

    async def go():
        state = SimpleNamespace(cancel_event=asyncio.Event())
        asyncio.get_running_loop().call_later(0.1, state.cancel_event.set)
        t0 = _t.monotonic()
        d = await observer._judge_unless_cancelled(state, slow_judgment())
        return d, _t.monotonic() - t0

    d, took = asyncio.run(go())
    assert took < 1.0 and d.action == "noop"

    async def fast():
        state = SimpleNamespace(cancel_event=asyncio.Event())

        async def judged():
            return observer.ObserverDecision(action="inject", content="x")
        return await observer._judge_unless_cancelled(state, judged())
    assert asyncio.run(fast()).action == "inject"


def test_inject_drops_queued_spoken_turns_and_carries_their_words(monkeypatch):
    """The 2026-10-31 sequence: "Yeah, he found it." queued behind a cancelled
    turn, then "so anyways…" superseded both. The queued spoken turn is
    dropped (its reader told `superseded`), its words ride the new request,
    and a queued TYPED turn is left alone."""
    from datetime import datetime

    from app import sessions_io as sio
    from app import voice_tap
    from app.routers import voice

    async def go():
        sid = "sup-1"
        q = sio._get_or_create_queue(sid)
        spoken = sio.SessionTurn(turn_id="sp-q", source="user",
                                 payload={"text": "[Alan]: Yeah, he found it."},
                                 enqueued_at=datetime.now())
        typed = sio.SessionTurn(turn_id="ty-q", source="user",
                                payload={"text": "typed thing"}, enqueued_at=datetime.now())
        voice_tap.mark_voice_turn("sp-q")
        q.pending_user.extend([spoken, typed])
        cancelled, earlier = await voice._supersede_spoken_turn(sid, {})
        events = []
        while not spoken.events.empty():
            events.append(spoken.events.get_nowait())
        return cancelled, earlier, [t.turn_id for t in q.pending_user], events

    cancelled, earlier, left, events = asyncio.run(go())
    assert cancelled is None, "nothing was running"
    assert earlier == ["[Alan]: Yeah, he found it."]
    assert left == ["ty-q"]
    assert events == [{"event": "superseded", "data": {"turn_id": "sp-q"}}, None]


# ── Phase 3.2: asides play from a prerendered cache (#1163) ───────────────

def test_an_aside_plays_from_the_phrase_cache_without_a_synthesis():
    """A filler is said because nothing has been heard for 2.5 s; it must not
    then wait on a TTS round trip. Rendered once at room start, it is pushed
    from memory, marks only the first-sound stages, and an answer clause with
    the same words is still synthesised live."""
    async def go():
        st = _streamer({"http://primary": lambda: _TTSResp([_pcm(3)])})
        assert await st.prerender(["One moment."]) == 1
        calls_after_render = len(st._http.calls)
        tl = TurnTimeline("voice")
        await st.speak("One moment.", timeline=tl, aside=True)
        await _until(lambda: len(st.source.live) >= 3)
        assert len(st._http.calls) == calls_after_render, "no synthesis for a cached aside"
        assert "first_tts_byte" in tl.marks and "first_answer_byte" not in tl.marks
        await st.speak("One moment.", timeline=tl)          # an answer: live
        await _until(lambda: len(st._http.calls) == calls_after_render + 1)
        await st.close()
    asyncio.run(go())


def test_a_phrase_that_will_not_render_is_spoken_live():
    async def go():
        st = _streamer({"http://primary": lambda: _TTSResp(status=500)})
        assert await st.prerender(["One moment."]) == 0
        await st.close()
    asyncio.run(go())


# ── Phase 3.1: speculative turn start, shadow ─────────────────────────────

class _ProbModel:
    """Silero stand-in: speech for the first `speech` frames, then silence."""

    def __init__(self, speech):
        self.n, self.speech = 0, speech

    def __call__(self, frame):
        self.n += 1
        return 0.9 if self.n <= self.speech else 0.0

    def reset_states(self):
        self.n = 0


def test_the_segmenter_reports_a_tentative_close_before_the_real_one():
    from voice.vad import FRAME_SAMPLES, SileroSegmenter
    seg = SileroSegmenter(model=_ProbModel(speech=40), min_silence_ms=380,
                          tentative_silence_ms=200)
    out = []
    tentative_at = close_at = None
    for i in range(80):
        utts = seg.feed(np.zeros(FRAME_SAMPLES, dtype=np.float32))
        t = seg.take_tentatives()
        if t and tentative_at is None:
            tentative_at, tent = i, t[0]
        if utts and close_at is None:
            close_at, out = i, utts
    assert tentative_at is not None and close_at is not None
    assert tentative_at < close_at, "the guess comes before the close"
    assert tent.start_sample == out[0].start_sample and tent.reason == "tentative"
    off = SileroSegmenter(model=_ProbModel(speech=40), min_silence_ms=380)
    for _ in range(80):
        off.feed(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert off.take_tentatives() == [], "0 = off"


def test_the_shadow_verdict_compares_the_guess_with_the_turn():
    async def go():
        b = _bridge()
        tl = TurnTimeline("voice")
        utt = _utt()
        b._spec["user-a"] = (utt.start_sample, utt.end_sample, "What time is it?", time.monotonic())
        b._spec_verdict("user-a", utt, "what time is it", time.monotonic() + 0.2, tl)
        hit = tl.facts["spec"]
        b._spec["user-a"] = (utt.start_sample, utt.end_sample, "What time", time.monotonic())
        tl2 = TurnTimeline("voice")
        b._spec_verdict("user-a", utt, "what time is it", time.monotonic(), tl2)
        return hit, tl2.facts["spec"]
    hit, miss = asyncio.run(go())
    assert hit["match"] is True and hit["asr_saved_ms"] >= 150
    assert miss["match"] is False


def test_first_turn_stats_go_out_once_and_only_for_a_spoken_turn():
    """The message path reports a spoken turn's first model request to the worker, once."""
    import asyncio
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    from app import voice_tap

    sent = []

    async def emit(turn, event, data):
        sent.append((turn.turn_id, event, data))

    t0 = datetime(2026, 11, 9, 12, 0, 0)
    spoken = SimpleNamespace(turn_id="vt-stats-1", enqueued_at=t0,
                             started_at=t0 + timedelta(seconds=1.25))
    typed = SimpleNamespace(turn_id="typed-stats-1", enqueued_at=t0, started_at=t0)
    voice_tap.mark_voice_turn(spoken.turn_id)
    evt = {"ttft_ms": 640, "request_ms": 2100}
    usage = {"input_tokens": 81000, "cache_read": 80500}

    async def go():
        await voice_tap.emit_first_turn_stats(typed, evt, usage, emit)
        await voice_tap.emit_first_turn_stats(spoken, evt, usage, emit)
        await voice_tap.emit_first_turn_stats(spoken, {"ttft_ms": 9}, usage, emit)

    asyncio.run(go())
    assert sent == [("vt-stats-1", "turn_stats", {
        "prompt_tokens": 81000, "cached_tokens": 80500, "ttft_ms": 640,
        "request_ms": 2100, "queue_wait_s": 1.25})]
