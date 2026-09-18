"""The gate between a closed utterance and an injected turn.

Until 2026-09-17 the gate could only be observed through the worker's log.
These drive `RoomBridge._handle_utterance_event` with a scripted recogniser and
a fake inject endpoint, so every decision it makes is a test rather than a
line somebody has to go and read:

  * the transcript is a second wake path — measured at 40 of 48 synthetic wake
    phrases against 17-23 for the acoustic models alone, with no false accepts
    on near-misses ("Floyd came over", "alloy wheels") or on 496 real
    utterances of room audio;
  * a bare wake word opens the window and injects nothing, and the follow-up
    then needs no wake word;
  * a sentence Smart Turn calls unfinished is held and sent as ONE turn with
    what follows, not as two half-questions.
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

livekit_worker = pytest.importorskip("livekit_worker")
from voice.asr import Transcript  # noqa: E402
from voice.pipeline import HearingEvent  # noqa: E402
from voice.turn import TurnVerdict  # noqa: E402
from voice.vad import Utterance  # noqa: E402
from voice.wake import WakeDetection  # noqa: E402

WORDS = ["lloyd", "hey lloyd", "hi lloyd", "okay lloyd", "hello lloyd"]


class _STT:
    name = "fake"

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return Transcript(self._texts.pop(0) if self._texts else "", 0.01, "fake")


class _Turn:
    def __init__(self, verdicts):
        self._v = list(verdicts)

    def predict(self, audio):
        p = self._v.pop(0) if self._v else 1.0
        return TurnVerdict(p >= 0.5, p, 1.0)


class _HTTP:
    def __init__(self):
        self.posts = []

    async def post(self, url, json=None, timeout=None):
        self.posts.append(json)
        return type("R", (), {"status_code": 200, "text": ""})()


def _bridge(texts, turn=None, text_fallback=True):
    async def make():
        b = livekit_worker.RoomBridge(
            "lloyd-20260917_000000_gate",
            {"room_prefix": "lloyd-",
             "wake": {"words": WORDS, "continuation_seconds": 6.0,
                      "text_fallback": text_fallback},
             "turn_detection": {"hold_timeout_ms": 60000}},
            stt=_STT(texts), vad_cfg={}, http_client=_HTTP(),
            smart_turn=turn)
        return b
    return asyncio.run(make())


def _utt(seconds=1.0):
    return Utterance(audio=np.zeros(int(16000 * seconds), dtype=np.float32),
                     start_sample=0, end_sample=int(16000 * seconds),
                     max_prob=0.9, reason="silence")


def _hear(bridge, *, wake=False, identity="user-a"):
    det = WakeDetection("Hey_Lloyd", 0.8, 100, 0.0) if wake else None
    ev = HearingEvent("utterance", utterance=_utt(), wake=det)

    async def go():
        if wake:
            bridge._on_wake(HearingEvent("wake", detection=det), identity)
        await bridge._handle_utterance_event(ev, identity)
    asyncio.run(go())
    return [p["text"] for p in bridge.http.posts if "text" in p]  # injects, not prewarms


def test_a_transcript_that_opens_with_the_wake_phrase_wakes_lloyd():
    b = _bridge(["Hey Lloyd, what time is it?"])
    assert _hear(b) == ["what time is it?"]


@pytest.mark.parametrize("text", [
    "Floyd came over yesterday.",
    "The alloy wheels are shiny.",
    "So I told Lloyd about it.",   # the name, not an address
    "Lloyd's car is outside.",
])
def test_a_near_miss_does_not(text):
    b = _bridge([text])
    assert _hear(b) == []


def test_the_text_path_has_a_kill_switch():
    b = _bridge(["Hey Lloyd, what time is it?"], text_fallback=False)
    assert _hear(b) == []
    assert b.stt.calls == 0, "with the path off, idle audio is not transcribed"


def test_an_acoustic_wake_strips_the_word_from_the_request():
    b = _bridge(["Hey Lloyd, turn on the lights."])
    assert _hear(b, wake=True) == ["turn on the lights."]


def test_a_bare_wake_opens_the_window_and_the_follow_up_needs_no_wake_word():
    b = _bridge(["Hey Lloyd.", "What time is it?"])
    assert _hear(b) == []                      # bare: nothing injected
    assert b.wake.in_continuation()
    assert _hear(b) == ["What time is it?"]


def test_the_window_belongs_to_whoever_woke_it():
    b = _bridge(["Hey Lloyd.", "What time is it?"])
    _hear(b, identity="user-a")
    assert _hear(b, identity="user-b") == []


def test_an_unfinished_sentence_is_held_and_sent_with_what_follows():
    """In the window, a 380 ms pause closes an utterance. Smart Turn calls the
    first half unfinished, so it waits and the two halves become one turn."""
    b = _bridge(["Hey Lloyd.",
                 "I was wondering if you could tell me about the weather tomorrow"],
                turn=_Turn([0.02, 0.97]))
    _hear(b)                                    # bare wake, window open
    assert _hear(b) == []                       # first half: held
    assert b._held.get("user-a"), "the unfinished half must be kept"
    posts = _hear(b)                            # second half: joined, sent
    assert posts == ["I was wondering if you could tell me about the weather tomorrow"]
    assert b.stt.calls == 2, "the joined audio is transcribed once, as one turn"


@pytest.mark.parametrize("text,tail", [
    ("Uh, hey Lloyd, stop.", "stop."),
    ("Um hi Lloyd what's up", "what's up"),
    ("Okay Lloyd.", ""),
    ("So I told Lloyd", None),          # one filler, then not a wake phrase
    ("Uh uh hey Lloyd", None),          # only ONE filler is allowed
])
def test_the_matcher_allows_one_leading_filler(text, tail):
    assert livekit_worker._strip_wake_word(text, sorted(WORDS, key=len, reverse=True)) == tail


def test_the_speaker_encoder_is_handed_int16():
    """SpeakerIdentifier divides by 32768 itself. The pipeline's float32
    [-1, 1] audio passed straight through is a 90 dB attenuation — invisible
    with no profile enrolled, wrong the day one is."""
    seen = {}

    class _Spk:
        def identify(self, samples, sr):
            seen["dtype"] = samples.dtype
            seen["peak"] = int(np.abs(samples).max())
            return "Unknown", 0.0, np.ones(4)

    b = _bridge([])
    b.speaker_id = _Spk()
    audio = (np.sin(np.linspace(0, 100, 16000)) * 0.5).astype(np.float32)
    asyncio.run(b._embed_async(audio, 16000))
    assert seen["dtype"] == np.int16
    assert seen["peak"] > 10000, "the level must survive the conversion"


def test_an_acoustic_wake_asks_the_backend_to_prewarm():
    b = _bridge([])

    async def go():
        det = WakeDetection("Hey_Lloyd", 0.8, 100, 0.0)
        b._on_wake(HearingEvent("wake", detection=det), "user-a")
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert {"session_key": "20260917_000000_gate"} in b.http.posts


class _SpeakingTTS:
    def __init__(self, speaking=True):
        self.is_speaking = speaking
        self.interrupts = 0

    def interrupt(self):
        self.interrupts += 1
        return 0


@pytest.mark.parametrize("enabled,speaking,who,expected", [
    (True, True, "user-a", 1),     # the window's owner talks over Lloyd: stop
    (True, True, "user-b", 0),     # somebody else (a TV, a guest): do not
    (True, False, "user-a", 0),    # nothing to interrupt
    (False, True, "user-a", 0),    # off by default: needs client-side AEC
])
def test_barge_in_is_gated_on_the_switch_the_speaker_and_the_speech(
        enabled, speaking, who, expected):
    async def make():
        return livekit_worker.RoomBridge(
            "lloyd-20260917_000000_barge",
            {"room_prefix": "lloyd-", "wake": {"words": WORDS},
             "barge_in": {"enabled": enabled}},
            stt=_STT([]), vad_cfg={}, http_client=_HTTP())
    b = asyncio.run(make())
    b.tts = _SpeakingTTS(speaking)
    b.wake.extend("user-a")
    b._on_speech_start(who)
    assert b.tts.interrupts == expected
