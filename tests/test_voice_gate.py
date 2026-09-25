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
import logging
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
        self.urls = []
        self.cancels = []
        self.current_turn = "turn-running"

    async def post(self, url, json=None, timeout=None):
        self.urls.append(url)
        if url.endswith("/cancel"):
            self.cancels.append(self.current_turn)
        else:
            self.posts.append(json)
        return type("R", (), {"status_code": 200, "text": ""})()

    async def get(self, url, timeout=None):
        cur = {"turn_id": self.current_turn}
        return type("R", (), {"status_code": 200,
                              "json": lambda self_: {"current": cur}})()


class _TTS:
    """Enough of TTSStreamer for the gate: nothing is playing."""
    is_speaking = False
    is_paused = False
    generation = 0
    reply_started_at = None
    last_reply_end = None

    def heard_text(self):
        return ""

    def unheard_chars(self):
        return 0

    def interrupt(self):
        self.generation += 1
        return 0


def _bridge(texts, turn=None, text_fallback=True, conversation=False, **extra):
    async def make():
        cfg = {"room_prefix": "lloyd-",
               "wake": {"words": WORDS, "continuation_seconds": 6.0,
                        "text_fallback": text_fallback},
               "turn_detection": {"hold_timeout_ms": 60000},
               "conversation": {"enabled": conversation, "addressee": "off"}}
        cfg.update(extra)
        b = livekit_worker.RoomBridge(
            "lloyd-20260917_000000_gate", cfg,
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
    "Eloid",                       # a mis-heard wake word, no acoustic fire
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


@pytest.mark.parametrize("text", ["Go!", "Go! Go!"])
def test_a_short_command_on_a_fired_wake_is_injected_whole(text):
    """A short command whose wake word the ASR did not spell. The acoustic model
    heard the wake word inside this utterance, so a transcript too short for the
    matcher to strip is a command, not a mis-heard name. Before this the branch
    had no path to an injection at all: it opened the window and returned,
    discarding the word. Both texts are cases from the retained diag corpus
    (`Go!` fired at 0.44, `Go! Go!` at 0.79), and both were commands said to
    Lloyd. ("Lloyd, stop" heard as "Stop" was the third case; since
    2026-09-24 it is a stop word, below.)"""
    b = _bridge([text])
    assert _hear(b, wake=True) == [text]


def test_lloyd_stop_with_nothing_running_closes_rather_than_asks():
    """"Lloyd, stop" heard as "Stop" with nothing to stop is a closer: it ends
    the conversation instead of sending the model a one-word turn to answer."""
    b = _bridge(["Stop"], conversation=True)
    assert _hear(b, wake=True) == []
    assert not b.wake.in_conversation()


def test_lloyd_stop_with_a_turn_in_flight_cancels_it_and_injects_nothing():
    """With a spoken turn still running (tools, nothing said yet), "stop" is
    what it sounds like: that turn is cancelled — and not a new "Stop" turn
    queued behind the one it was meant to stop."""
    b = _bridge(["Stop"], conversation=True)
    b._active_turn = "turn-running"
    b.tts = _TTS()
    _hear(b, wake=True)
    assert [p for p in b.http.posts if "text" in p] == []
    assert b.http.cancels == ["turn-running"]


def test_the_acoustic_bare_wake_word_still_injects_nothing():
    """The legitimate case survives: a cleanly-spelled wake word strips to an
    empty tail and takes the bare-wake arm, which opens the window and injects
    nothing — even with the detection fired inside the same utterance."""
    b = _bridge(["Hey Lloyd."])
    assert _hear(b, wake=True) == []
    assert b.wake.in_continuation()


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
    first half unfinished, so it waits and the two halves become one turn.

    Smart Turn and the recogniser run side by side since 2026-09-24, so the
    held half is transcribed too — one ~60 ms CPU pass, spent to take Smart
    Turn off the critical path of every turn — but what is SENT is the joined
    audio's transcript, once."""
    b = _bridge(["Hey Lloyd.",
                 "I was wondering if",
                 "I was wondering if you could tell me about the weather tomorrow"],
                turn=_Turn([0.02, 0.97]))
    _hear(b)                                    # bare wake, window open
    assert _hear(b) == []                       # first half: held
    assert b._held.get("user-a"), "the unfinished half must be kept"
    posts = _hear(b)                            # second half: joined, sent
    assert posts == ["I was wondering if you could tell me about the weather tomorrow"]
    assert b.stt.calls == 3, "held half + the joined turn (+ the bare wake)"


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
    is_paused = False
    reply_started_at = None
    generation = 0

    def __init__(self, speaking=True):
        self.is_speaking = speaking
        self.interrupts = 0
        self.pauses = 0

    def interrupt(self):
        self.interrupts += 1
        return 0

    def pause(self):
        self.pauses += 1
        self.is_paused = True
        return True


class _Hearing:
    """A hearing thread whose segmenter says the speaker is still talking."""

    def __init__(self, in_speech=True):
        self.pipeline = type("P", (), {"segmenter": type("S", (), {"in_speech": in_speech})()})()


@pytest.mark.parametrize("enabled,speaking,who,expected", [
    (True, True, "user-a", 1),     # the window's owner talks over Lloyd: pause
    (True, True, "user-b", 0),     # somebody else (a TV, a guest): do not
    (True, False, "user-a", 0),    # nothing to interrupt
    (False, True, "user-a", 0),    # the switch
])
def test_barge_in_is_gated_on_the_switch_the_speaker_and_the_speech(
        enabled, speaking, who, expected):
    """Since 2026-09-24 a barge-in PAUSES first (after `min_duration_s` of
    sustained speech) and the utterance that follows decides; nothing is
    interrupted on the rising edge alone. tests/test_voice_duplex.py has the
    rest of the state machine."""
    async def go():
        b = livekit_worker.RoomBridge(
            "lloyd-20260917_000000_barge",
            {"room_prefix": "lloyd-", "wake": {"words": WORDS},
             "barge_in": {"enabled": enabled, "min_duration_s": 0.01,
                          "warmup_s": 0.0}},
            stt=_STT([]), vad_cfg={}, http_client=_HTTP())
        b.tts = _SpeakingTTS(speaking)
        b._hearing[who] = _Hearing(True)
        b.wake.extend("user-a")
        b._on_speech_start(who)
        await asyncio.sleep(0.05)
        return b
    b = asyncio.run(go())
    assert b.tts.pauses == expected
    assert b.tts.interrupts == 0, "an edge alone never interrupts"


@pytest.mark.parametrize("text,injected", [
    # Mentions: the model fired, the transcript names Lloyd mid-sentence.
    # "Did Lloyd finish the report?" is the one the retrained model scored
    # 0.93 on — "did Lloyd" sounds like "hi Lloyd".
    ("Did Lloyd finish the report?", []),
    ("I was telling Lloyd about it earlier.", []),
    ("Lloyd's car is parked outside.", []),
    # Addresses: first or last.
    ("What time is it, Lloyd?", ["What time is it, Lloyd?"]),
    # Misheard: no name in the transcript at all, so the acoustic model
    # stays the authority on whether it was said.
    ("Louida, what time is it?", ["Louida, what time is it?"]),
])
def test_an_acoustic_wake_on_a_mention_is_not_a_request(text, injected):
    b = _bridge([text])
    assert _hear(b, wake=True) == injected


def _log_bodies(caplog, needle):
    """Worker log messages naming `needle`, with the `[room] ` prefix removed."""
    bodies = []
    for record in caplog.records:
        msg = record.getMessage()
        if needle in msg:
            bodies.append(msg.split("] ", 1)[1] if msg.startswith("[") else msg)
    return bodies


def test_a_short_transcript_with_no_acoustic_fire_is_still_dropped(caplog):
    """The other half of the branch, which must keep discarding. With no
    detection inside the utterance the wake came from the transcript alone, so
    a short transcript the matcher cannot strip is a mistranscribed bare wake
    word, not a command — and the line it logs must not be confusable with the
    legitimate bare-wake line, which shares its opening words today.

    Reached the way it happens in a room: a half-sentence held from an earlier
    utterance is glued in front of this one, so the joined audio is
    re-transcribed and the transcript that reaches the gate is not the one that
    woke it.
    """
    b = _bridge(["Hey Lloyd", "Eloid"])
    b._hold("user-a", np.zeros(16000 // 4, dtype=np.float32))
    with caplog.at_level(logging.INFO, logger="lloyd-agent-worker"):
        assert _hear(b) == []
    bodies = _log_bodies(caplog, "Eloid")
    assert any("no acoustic fire" in line for line in bodies), bodies
    assert not any(line.startswith("bare wake-word") for line in bodies), bodies


def test_a_mis_heard_wake_word_on_a_fired_wake_is_the_accepted_cost():
    """The trade-off this fix buys, named so it cannot be silently widened.
    "Eloid" with no detection is a near-miss and stays dropped; with the
    acoustic model having fired inside the utterance there is no way to tell it
    from "Stop", so it is injected as a question. A confidence floor above the
    0.4 firing threshold is the human decision that would change this."""
    b = _bridge(["Eloid"])
    assert _hear(b, wake=True) == ["Eloid"]


def test_the_mention_check_reads_the_name_not_the_phrases():
    names = sorted(WORDS, key=len, reverse=True)
    assert livekit_worker._mentions_wake_name("ask Lloyd later", names)
    assert not livekit_worker._mentions_wake_name("thanks Lloyd", names)
    assert not livekit_worker._mentions_wake_name("Lloyd, stop", names)
    assert not livekit_worker._mentions_wake_name("the alloy wheels", names)
