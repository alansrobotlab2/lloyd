"""The voice loop as a conversation (2026-09-24): measured, duplex, wake-word-free.

What each part has to get right, as tests rather than log lines:

  * the latency breakdown adds up and finds the silence in a tool turn;
  * a backchannel ('Yeah.' while Lloyd talks) is not a turn — unless he just
    asked a question, when it is the answer;
  * a conversation stays open past the follow-up window, and there an
    utterance reaches the model only when the addressee judge says it was
    meant for Lloyd; a judge that does not answer is a drop;
  * a closer ends the conversation; Lloyd's own voice is never a person;
  * pause/resume loses nothing, and what was heard is what finished playing;
  * a barge-in pauses on sustained speech and the words decide: silence or a
    backchannel resumes, real words stop the reply, tell the backend what was
    heard and cancel the turn being spoken — and only that one;
  * a tool turn says what it is doing, and a turn stuck behind another says so;
  * Smart Turn, the recogniser and the embedding run at the same time;
  * the backend's session tap carries typed turns and never a spoken one.
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
from voice import conversation as conv  # noqa: E402
from voice.addressee import Verdict  # noqa: E402
from voice.asr import Transcript  # noqa: E402
from voice.pipeline import HearingEvent  # noqa: E402
from voice.thinking import ThinkingSound  # noqa: E402
from voice.timeline import TurnTimeline  # noqa: E402
from voice.turn import TurnVerdict  # noqa: E402
from voice.vad import Utterance  # noqa: E402

WORDS = ["lloyd", "hey lloyd", "hi lloyd"]


# ── doubles ────────────────────────────────────────────────────────────────

class _STT:
    name = "fake"

    def __init__(self, texts, delay=0.0):
        self._texts = list(texts)
        self.calls = 0
        self.delay = delay

    def transcribe(self, audio):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return Transcript(self._texts.pop(0) if self._texts else "", 0.01, "fake")


class _HTTP:
    def __init__(self, current_turn="t-old"):
        self.posts = []
        self.cancels = []
        self.interrupted = []
        self.current_turn = current_turn

    async def post(self, url, json=None, timeout=None):
        if url.endswith("/cancel"):
            self.cancels.append(self.current_turn)
        elif url.endswith("/api/voice/interrupted"):
            self.interrupted.append(json)
        else:
            self.posts.append(json)
        return type("R", (), {"status_code": 200, "text": ""})()

    async def get(self, url, timeout=None):
        cur = {"turn_id": self.current_turn}
        return type("R", (), {"status_code": 200,
                              "json": lambda self_: {"current": cur}})()


class _TTS:
    def __init__(self, speaking=False, heard="It is seven.", last_end=None):
        self.is_speaking = speaking
        self.is_paused = False
        self.generation = 0
        self.reply_started_at = None
        self.last_reply_end = last_end
        self._heard = heard
        self.pauses = self.resumes = self.interrupts = 0

    def heard_text(self):
        return self._heard

    def unheard_chars(self):
        return 40

    def interrupt(self):
        self.interrupts += 1
        self.generation += 1
        self.is_speaking = False
        return 0

    def pause(self):
        if self.is_paused:
            return False
        self.pauses += 1
        self.is_paused = True
        return True

    async def resume(self):
        if not self.is_paused:
            return False
        self.resumes += 1
        self.is_paused = False
        return True


class _Judge:
    def __init__(self, p, mode="enforce"):
        self.p = p
        self.mode = mode
        self.asked = []

    @property
    def enforcing(self):
        return self.mode == "enforce"

    @property
    def active(self):
        return True

    async def judge(self, utterance, last_agent, since, speaker, mentions):
        self.asked.append((utterance, last_agent))
        if self.p is None:
            return None
        return Verdict(self.p >= 0.7, self.p, 5.0)


def _bridge(texts, *, conversation=True, barge=False, stt_delay=0.0, **extra):
    async def make():
        cfg = {"room_prefix": "lloyd-",
               "wake": {"words": WORDS, "continuation_seconds": 6.0},
               "turn_detection": {"hold_timeout_ms": 60000},
               "conversation": {"enabled": conversation, "idle_close_s": 90,
                                "addressee": "off"},
               "barge_in": {"enabled": barge, "min_duration_s": 0.01,
                            "warmup_s": 0.0, "false_interruption_timeout_s": 0.2}}
        cfg.update(extra)
        return livekit_worker.RoomBridge(
            "lloyd-20260924_000000_dplx", cfg,
            stt=_STT(texts, stt_delay), vad_cfg={}, http_client=_HTTP())
    return asyncio.run(make())


def _utt(seconds=0.6):
    audio = np.zeros(int(seconds * 16000), dtype=np.float32)
    return Utterance(audio=audio, start_sample=0, end_sample=audio.size, max_prob=0.9,
                     reason="silence")


def _hear(b, identity="user-a"):
    ev = HearingEvent("utterance", utterance=_utt())
    asyncio.run(b._handle_utterance_event(ev, identity))
    return [p["text"] for p in b.http.posts if p and "text" in p]


def _open_conversation_past_follow_up(b, identity="user-a"):
    """A conversation that is open, with the short follow-up window closed."""
    b.wake.extend(identity)
    b.wake.open_conversation(identity)
    b.wake._until = 0.0


# ── the latency breakdown ──────────────────────────────────────────────────

def test_the_breakdown_is_the_stages_in_order_and_sums_to_the_total():
    tl = TurnTimeline()
    base = 100.0
    for i, stage in enumerate(["speech_end", "vad_close", "asr_done",
                               "inject_sent", "first_delta", "first_clause"]):
        tl.mark(stage, base + i * 0.1)
    tl.audio_pushed(base + 0.7, base + 0.75, 0.1)
    parts = tl.contributions()
    assert [p[0] for p in parts] == ["vad_close", "asr_done", "inject_sent",
                                     "first_delta", "first_clause",
                                     "first_pushed", "first_played"]
    assert sum(dt for _, dt in parts) == pytest.approx(tl.total())
    assert tl.total() == pytest.approx(0.75)
    assert "eos→audio=0.75s" in tl.summary()


def test_the_parallel_stages_are_measured_from_the_close_not_from_each_other():
    tl = TurnTimeline()
    tl.mark("vad_close", 10.0)
    tl.mark("asr_done", 10.8)          # the slow one
    tl.mark("embed_done", 10.1)
    tl.mark("turn_verdict", 10.2)
    tl.mark("inject_sent", 10.85)
    parts = dict(tl.contributions())
    assert parts["asr_done"] == pytest.approx(0.8)
    assert parts["embed_done"] == pytest.approx(0.1)
    assert parts["inject_sent"] == pytest.approx(0.05), "from the LAST parallel stage"
    assert all(dt >= 0 for dt in parts.values())
    assert "asr_done@800" in tl.summary()


def test_marks_are_first_writer_wins():
    tl = TurnTimeline()
    tl.mark("first_tts_byte", 1.0)
    tl.mark("first_tts_byte", 2.0)
    assert tl.marks["first_tts_byte"] == 1.0


def test_the_longest_silence_between_spoken_audio_is_recorded():
    """The 2026-09-24 tool turn: a filler, then 44 s of nothing."""
    tl = TurnTimeline()
    tl.audio_pushed(10.0, 10.0, 0.7)          # "One moment."
    tl.audio_pushed(55.4, 55.4, 1.0)          # the answer, 44.7 s later
    tl.audio_pushed(56.4, 56.4, 1.0)          # back to back: no gap
    assert tl.max_gap_s == pytest.approx(44.7)
    assert tl.gaps == [pytest.approx(44.7)]


# ── the vocabulary ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["Yeah.", "Okay.", "Mm.", "Mm-hmm.", "Thank you.",
                                  "Yeah, okay.", "Crazy.", "Uh-huh", "Got it."])
def test_acknowledgements_are_backchannels(text):
    assert conv.is_backchannel(text)


@pytest.mark.parametrize("text", ["Yeah, check the other one.", "What?",
                                  "Okay, now restart it.", "No.", "Stop."])
def test_requests_are_not(text):
    assert not conv.is_backchannel(text)


def test_after_a_question_yeah_is_an_answer_and_mm_is_still_not():
    assert not conv.is_backchannel("Yeah.", after_question=True)
    assert not conv.is_backchannel("Sure.", after_question=True)
    assert conv.is_backchannel("Mm-hmm.", after_question=True)


@pytest.mark.parametrize("text,closer", [
    ("That's all.", True), ("Thanks, Lloyd.", True), ("Never mind.", True),
    ("Stop.", True), ("That's all, Lloyd.", True), ("I'm done.", True),
    ("That's all, thanks.", True), ("Okay, never mind, Lloyd.", True),
    ("Thanks.", False),
    ("Stop the build.", False), ("Done.", False), ("That's all the files?", False),
])
def test_closers_are_whole_utterances(text, closer):
    assert conv.is_closer(text) is closer


def test_stoppers_include_the_interrupt_words_that_are_not_closers():
    assert conv.is_stopper("Wait.") and not conv.is_closer("Wait.")
    assert conv.is_stopper("Lloyd, stop.")
    assert not conv.is_stopper("Wait for the build to finish.")


@pytest.mark.parametrize("caption,spoken", [
    ("Checking supervisor status", "Checking supervisor status now."),
    ("Reading server.py", ""),                        # a path and one word
    ("grep for enqueue_turn in app/", ""),            # lower-case tool jargon
    ("Searching the vault for kernel notes", "Searching the vault for kernel notes."),
    ("Restarting the backend", "Restarting the backend now."),
    ("Looking up 8f3a2c1d in the ledger", "Looking up in the ledger."),
])
def test_captions_become_short_spoken_progress(caption, spoken):
    assert conv.caption_to_speech(caption) == spoken


def test_a_reply_ending_in_a_question_is_one():
    assert conv.ends_with_question("Want me to restart it?")
    assert conv.ends_with_question('Should I file "that"?"')
    assert not conv.ends_with_question("It restarted.")


def test_the_thinking_sound_stays_under_the_half_duplex_threshold():
    """VoiceRoom mutes the mic while Lloyd's track is above 0.02; the working
    sound must not trip that for a whole tool turn."""
    ts = ThinkingSound(24000)
    assert 0.002 < ts.rms < 0.012
    frame = ts.next_frame(2400)
    assert len(frame) == 4800
    peak = np.abs(np.frombuffer(frame * 20, dtype=np.int16)).max() / 32768
    assert peak <= 0.021


# ── backchannels in the gate ───────────────────────────────────────────────

def test_a_backchannel_in_the_window_is_not_a_turn_and_keeps_it_open():
    """The 19:17 'Yeah.' was injected and answered with 59 s of speech."""
    b = _bridge(["Yeah."])
    b.wake.extend("user-a")
    before = b.wake._until
    time.sleep(0.01)
    assert _hear(b) == []
    assert b.wake._until > before, "a backchannel extends the window"


def test_after_a_question_the_same_word_is_the_answer():
    b = _bridge(["Yeah."])
    b.wake.extend("user-a")
    b._last_agent_text = "Want me to restart it?"
    assert _hear(b) == ["Yeah."]


def test_outside_any_window_a_backchannel_is_still_just_dropped():
    b = _bridge(["Yeah."], conversation=False)
    assert _hear(b) == []


# ── the conversation ───────────────────────────────────────────────────────

def test_past_the_follow_up_window_an_addressed_utterance_is_a_turn():
    b = _bridge(["And what about tomorrow?"])
    b.addressee = _Judge(0.98)
    b._last_agent_text = "It's sixty-eight degrees and sunny."
    _open_conversation_past_follow_up(b)
    assert _hear(b) == ["And what about tomorrow?"]
    assert b.addressee.asked == [("And what about tomorrow?",
                                  "It's sixty-eight degrees and sunny.")]


def test_past_the_follow_up_window_speech_to_someone_else_is_dropped():
    b = _bridge(["Honey, can you grab the milk?"])
    b.addressee = _Judge(0.001)
    _open_conversation_past_follow_up(b)
    assert _hear(b) == []


def test_a_judge_that_does_not_answer_is_a_drop_there():
    """What the gate did before conversation mode existed — so a djev outage
    costs only what the mode added."""
    b = _bridge(["What time is it?"])
    b.addressee = _Judge(None)
    _open_conversation_past_follow_up(b)
    assert _hear(b) == []


def test_in_shadow_the_judge_changes_nothing():
    b = _bridge(["What time is it?"])
    b.addressee = _Judge(0.99, mode="shadow")
    _open_conversation_past_follow_up(b)
    assert _hear(b) == []


def test_inside_the_follow_up_window_the_judge_is_not_consulted_first():
    b = _bridge(["And tomorrow?"])
    b.addressee = _Judge(0.0001)
    b.wake.extend("user-a")
    b.wake.open_conversation("user-a")
    assert _hear(b) == ["And tomorrow?"]


def test_a_conversation_belongs_to_whoever_opened_it():
    b = _bridge(["What time is it?"])
    b.addressee = _Judge(0.99)
    _open_conversation_past_follow_up(b, "user-a")
    assert _hear(b, identity="user-b") == []


def test_once_someone_is_enrolled_the_conversation_needs_an_enrolled_voice():
    class _Spk:
        def __init__(self, name):
            self.name = name

        def list_profiles(self):
            return [{"name": "Alan"}, {"name": "lloyd-voice"}]

        def identify(self, samples, sr):
            return self.name, 0.9, np.ones(4) / 2

    for name, expected in (("Alan", ["What time is it?"]), ("Unknown", [])):
        b = _bridge(["What time is it?"])
        b.addressee = _Judge(0.99)
        b.speaker_id = _Spk(name)
        _open_conversation_past_follow_up(b)
        assert _hear(b) == expected, name
        if expected:
            assert b.http.posts[-1]["speaker"] == "Alan"


def test_a_closer_ends_the_conversation():
    b = _bridge(["That's all, thanks."])
    b.wake.extend("user-a")
    b.wake.open_conversation("user-a")
    assert _hear(b) == []
    assert not b.wake.in_conversation() and not b.wake.in_continuation()


def test_the_speaker_leaving_ends_it():
    b = _bridge([])
    b.wake.extend("user-a")
    b.wake.open_conversation("user-a")
    b._on_participant_disconnected(type("P", (), {"identity": "user-a"})())
    assert not b.wake.in_conversation()


def test_a_reply_that_asks_holds_the_window_open_longer():
    b = _bridge([])
    b.wake.extend("user-a")
    b._last_agent_text = "Should I restart it?"
    b._schedule_wake_state_publish = lambda: None
    b._on_tts_utterance_end()
    assert b.wake.remaining_s() > 15.0


def test_lloyds_own_voice_is_never_a_person():
    class _Spk:
        def list_profiles(self):
            return [{"name": "lloyd-voice"}]

        def identify(self, samples, sr):
            return "lloyd-voice", 0.92, np.ones(4) / 2

    b = _bridge(["The build finished with two warnings."])
    b.speaker_id = _Spk()
    b.tts = _TTS(speaking=True)
    b.wake.extend("user-a")
    b.wake.open_conversation("user-a")
    assert _hear(b) == []


def test_a_hold_that_timed_out_is_not_held_again():
    """Released by the timeout precisely because Smart Turn said "unfinished";
    asking again re-held it forever — the e2e barge-in looped 2.49 s of speech
    for 45 s, and the question never reached the model."""
    class _Unfinished:
        def __init__(self):
            self.calls = 0

        def predict(self, audio):
            self.calls += 1
            return TurnVerdict(False, 0.14, 1.0)

    b = _bridge(["Actually, what is two plus two?"])
    b.smart_turn = _Unfinished()
    b.wake.extend("user-a")
    utt = _utt()
    utt = Utterance(audio=utt.audio, start_sample=0, end_sample=utt.audio.size,
                    max_prob=1.0, reason="hold_timeout")
    asyncio.run(b._handle_utterance_event(HearingEvent("utterance", utterance=utt), "user-a"))
    assert b.smart_turn.calls == 0
    assert [p["text"] for p in b.http.posts if p and "text" in p] == \
        ["Actually, what is two plus two?"]


def test_a_hold_is_not_released_while_the_speaker_is_still_talking():
    """The deadline passing mid-utterance used to release the first half as a
    turn and cut the speaker off; now it waits for the utterance to close."""
    b = _bridge(["half"])
    b.wake.extend("user-a")
    b._hold("user-a", np.zeros(16000, dtype=np.float32))
    b._held_deadline["user-a"] = time.monotonic() - 1.0
    b._hearing["user-a"] = _Hearing(True)

    async def go():
        t = asyncio.create_task(b._flush_held_loop())
        await asyncio.sleep(0.4)
        still = "user-a" in b._held
        b._hearing["user-a"] = _Hearing(False)          # they stopped
        await asyncio.sleep(0.7)
        t.cancel()
        return still, "user-a" in b._held
    still_held, held_after = asyncio.run(go())
    assert still_held, "not released while in speech"
    assert not held_after, "released once they stopped"


# ── the stages run together ────────────────────────────────────────────────

def test_smart_turn_asr_and_the_embedding_overlap():
    class _Turn:
        def predict(self, audio):
            time.sleep(0.2)
            return TurnVerdict(True, 0.99, 200.0)

    class _Spk:
        def list_profiles(self):
            return []

        def identify(self, samples, sr):
            time.sleep(0.2)
            return "Unknown", 0.0, np.ones(4) / 2

    b = _bridge(["And what about tomorrow?"], stt_delay=0.2)
    b.smart_turn = _Turn()
    b.speaker_id = _Spk()
    b.wake.extend("user-a")
    t0 = time.monotonic()
    assert _hear(b) == ["And what about tomorrow?"]
    assert time.monotonic() - t0 < 0.45, "three 0.2 s stages, run side by side"


# ── pause / resume / heard ─────────────────────────────────────────────────

class _Source:
    queued_duration = 0.0

    def __init__(self):
        self.frames = []
        self.clears = 0

    async def capture_frame(self, frame):
        self.frames.append(bytes(frame.data))

    def clear_queue(self):
        self.clears += 1

    async def wait_for_playout(self):
        return None


def _streamer():
    st = livekit_worker.TTSStreamer({"sample_rate": 24000,
                                     "shaping": {"presence_eq": False}}, room=None)
    st.source = _Source()
    return st


def test_pause_takes_back_what_had_not_played_and_resume_replays_it_first():
    async def go():
        st = _streamer()
        n = 2400
        for i in range(5):                     # 0.5 s of audio, all unplayed
            await st._push_frame(bytes([i]) * (2 * n), n)
        assert st.pause()
        assert st.is_paused and st.source.clears == 1
        replay = [pcm for pcm, _ in st._resume_frames]
        assert len(replay) == 5
        pushed_before = len(st.source.frames)
        assert await st.resume()
        assert st.source.frames[pushed_before:] == replay
        assert not st.is_paused
    asyncio.run(go())


def test_a_push_blocked_on_the_pause_waits_and_interrupt_releases_it():
    async def go():
        st = _streamer()
        st.pause()
        task = asyncio.create_task(st._push_frame(b"\0" * 4800, 2400))
        await asyncio.sleep(0.05)
        assert not task.done(), "a paused reply holds its next frame"
        st.interrupt()
        await asyncio.sleep(0.05)
        assert task.done() and not st.is_paused
    asyncio.run(go())


def test_heard_text_is_the_clauses_whose_playout_finished():
    st = _streamer()
    now = time.monotonic()
    st._clause_log = [("It is seven.", now - 1.0), ("Your meeting is at eight.", now + 2.0)]
    assert st.heard_text() == "It is seven."
    assert st.unheard_chars() == len("Your meeting is at eight.")


# ── barge-in ───────────────────────────────────────────────────────────────

class _Hearing:
    def __init__(self, in_speech=True):
        self.pipeline = type("P", (), {"segmenter": type("S", (), {"in_speech": in_speech})()})()


def _barge_bridge(texts):
    b = _bridge(texts, barge=True)
    b.tts = _TTS(speaking=True)
    b._speaking_turn = "t-old"
    b._hearing["user-a"] = _Hearing(True)
    b.wake.extend("user-a")
    b.wake.open_conversation("user-a")
    return b


def test_sustained_speech_pauses_and_silence_resumes():
    b = _barge_bridge([])

    async def go():
        b._on_speech_start("user-a")
        await asyncio.sleep(0.05)
        assert b.tts.pauses == 1 and b._barge
        watchdog = asyncio.create_task(b._barge_watchdog())
        b._on_speech_end("user-a")
        await asyncio.sleep(0.4)
        watchdog.cancel()
        return b
    b = asyncio.run(go())
    assert b.tts.resumes == 1 and b.tts.interrupts == 0 and b._barge is None


def test_the_warm_up_ignores_speech_as_a_reply_starts():
    b = _barge_bridge([])

    async def go():
        b._barge_warmup_s = 3.0
        b.tts.reply_started_at = time.monotonic()
        b._on_speech_start("user-a")
        await asyncio.sleep(0.05)
        return b
    assert asyncio.run(go()).tts.pauses == 0


def test_a_backchannel_over_lloyd_resumes_him():
    b = _barge_bridge(["Yeah."])

    async def go():
        b._on_speech_start("user-a")
        await asyncio.sleep(0.05)
        await b._handle_utterance_event(HearingEvent("utterance", utterance=_utt()), "user-a")
        await asyncio.sleep(0.05)
        return b
    b = asyncio.run(go())
    assert b.tts.resumes == 1 and b.tts.interrupts == 0
    assert [p for p in b.http.posts if p and "text" in p] == []


def test_a_real_request_over_lloyd_stops_him_reports_what_was_heard_and_asks():
    b = _barge_bridge(["No, the other one."])

    async def go():
        b._stream_replies = False
        b._on_speech_start("user-a")
        await asyncio.sleep(0.05)
        await b._handle_utterance_event(HearingEvent("utterance", utterance=_utt()), "user-a")
        return b
    b = asyncio.run(go())
    assert b.tts.interrupts == 1
    assert b.http.interrupted == [{"session_key": "20260924_000000_dplx",
                                   "heard": "It is seven.", "unheard_chars": 40}]
    assert b.http.cancels == ["t-old"]
    assert [p["text"] for p in b.http.posts if p and "text" in p] == ["No, the other one."]


def test_a_barge_in_never_cancels_a_turn_that_is_not_the_one_being_spoken():
    b = _barge_bridge(["No, the other one."])

    async def go():
        b._stream_replies = False
        b.http.current_turn = "t-newer"
        await b._handle_utterance_event(HearingEvent("utterance", utterance=_utt()), "user-a")
        return b
    assert asyncio.run(go()).http.cancels == []


def test_stop_while_speaking_stops_and_injects_nothing():
    b = _barge_bridge(["Stop."])

    async def go():
        await b._handle_utterance_event(HearingEvent("utterance", utterance=_utt()), "user-a")
        return b
    b = asyncio.run(go())
    assert b.tts.interrupts == 1 and b.http.cancels == ["t-old"]
    assert [p for p in b.http.posts if p and "text" in p] == []


# ── what a spoken turn says while it works ─────────────────────────────────

class _SpeakTTS:
    is_paused = False

    def __init__(self):
        self.said = []
        self.generation = 0
        self.is_speaking = False
        self.thinking = 0
        self._silence = 0.0

    @property
    def is_idle(self):
        return not self.is_speaking

    async def speak(self, text, timeline=None):
        self.said.append(text)
        self._silence = 0.0          # the listener just heard something

    def silence_s(self, now=None):
        return self._silence

    def start_thinking(self):
        self.thinking += 1

    async def stop_thinking(self):
        pass


def _speaker(**voice_turn):
    b = _bridge([], voice_turn=voice_turn)
    b.tts = _SpeakTTS()
    return b, livekit_worker.ReplySpeaker(b, label="voice", extras=True)


def test_a_long_tool_turn_speaks_the_tools_own_captions():
    b, rs = _speaker(progress={"every_seconds": 0.1}, filler={"after_seconds": 99})

    async def go():
        rs.start()
        await rs.on_event("voice_turn", {"turn_id": "t1"})
        await rs.on_event("session", {})
        await rs.on_event("tool_start", {"summary": "Checking supervisor status"})
        assert b.tts.said == ["One moment."], "the first tool call, before any words"
        b.tts._silence = 5.0                   # ...and then nothing, for a while
        rs.last_progress_at -= 1.0
        await asyncio.sleep(0.8)             # the ticker runs every 0.25 s
        await rs.finish()
        return b
    b = asyncio.run(go())
    assert "Checking supervisor status now." in b.tts.said
    assert b.tts.thinking >= 1, "the working sound fills the rest"


def test_a_turn_queued_behind_another_says_so_instead_of_a_filler():
    b, rs = _speaker(queued_notice={"after_seconds": 0.05},
                     filler={"after_seconds": 0.05})

    async def go():
        rs.start()
        await rs.on_event("voice_turn", {"turn_id": "t1", "queued_behind": True})
        await asyncio.sleep(0.4)
        await rs.finish()
        return b
    assert asyncio.run(go()).tts.said == ["Still finishing the last thing."]


def test_a_typed_turn_through_the_tap_has_no_fillers():
    b = _bridge([], voice_turn={"filler": {"after_seconds": 0.01}})

    async def go():
        b.tts = _SpeakTTS()
        b.room = type("R", (), {"remote_participants": {"user-a": object()}})()
        b._schedule_wake_state_publish = lambda: None
        await b._on_tap_event("session", {"turn_id": "typed1", "source": "user"})
        await asyncio.sleep(0.1)
        await b._on_tap_event("text_delta", {"turn_id": "typed1", "text": "It is noon. "})
        await b._on_tap_event("done", {"turn_id": "typed1"})
        return b
    b = asyncio.run(go())
    assert b.tts.said == ["It is noon."]
    assert b.wake.in_conversation(), "typing to Lloyd with the room open opens one"


# ── the backend half ───────────────────────────────────────────────────────

def test_the_tap_carries_typed_turns_and_never_a_spoken_one():
    from app import voice_tap
    voice_tap.reset_for_tests()

    async def go():
        q = voice_tap.open_tap("s1")
        voice_tap.mark_voice_turn("spoken")
        voice_tap.publish("s1", "text_delta", {"turn_id": "spoken", "text": "x"})
        voice_tap.publish("s1", "text_delta", {"turn_id": "typed", "text": "y"})
        voice_tap.publish("s1", "thinking_delta", {"turn_id": "typed", "text": "z"})
        voice_tap.publish("s2", "text_delta", {"turn_id": "other", "text": "w"})
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        voice_tap.close_tap("s1", q)
        return out
    out = asyncio.run(go())
    assert out == [{"event": "text_delta", "data": {"turn_id": "typed", "text": "y"}}]
    assert not voice_tap.has_listener("s1")


def test_emit_fans_out_to_the_tap_without_touching_the_turns_own_queue():
    from app import voice_tap
    from app.routers._messages_harness_adapter import _emit
    from app.sessions_io import SessionTurn
    from datetime import datetime
    voice_tap.reset_for_tests()

    async def go():
        q = voice_tap.open_tap("s1")
        turn = SessionTurn(turn_id="t1", source="user", payload={},
                           enqueued_at=datetime.now(), session_id="s1")
        await _emit(turn, "text_delta", {"text": "hi"})
        return turn.events.get_nowait(), q.get_nowait()
    own, tapped = asyncio.run(go())
    assert own["data"]["text"] == tapped["data"]["text"] == "hi"
    voice_tap.reset_for_tests()


def test_an_interruption_note_rides_the_next_turn_once():
    from app import voice_tap
    voice_tap.reset_for_tests()
    voice_tap.record_interruption("s1", "It is seven.", 40)
    note = voice_tap.take_heard_note("s1")
    assert note and "It is seven." in note and "interrupted" in note
    assert voice_tap.take_heard_note("s1") is None


def test_a_typed_turn_is_told_about_the_voice_room_only_while_one_listens():
    from app import voice_tap
    from app.routers import voice as voice_router
    voice_tap.reset_for_tests()
    assert voice_router.voice_room_prefix("s1") == ""
    q = voice_tap.open_tap("s1")
    voice_tap.record_interruption("s1", "It is seven.")
    prefix = voice_router.voice_room_prefix("s1")
    assert "It is seven." in prefix and "voice room is open" in prefix
    voice_tap.close_tap("s1", q)
    voice_tap.reset_for_tests()


def test_the_label_rig_takes_an_addressed_label(tmp_path):
    cap = livekit_worker.WakeMissCapture(diag_dir=tmp_path)
    cap.record_label(utterance_id="abc", miss_ts=None, said_wake_word=None,
                     note=None, addressed="backchannel")
    rec = json.loads((tmp_path / "labels.jsonl").read_text().splitlines()[-1])
    assert rec["addressed"] == "backchannel" and rec["said_wake_word"] is None
    with pytest.raises(ValueError):
        cap.record_label(utterance_id="abc", miss_ts=None, said_wake_word=None,
                         note=None, addressed="maybe")


# ── the acoustic stop word, wired ──────────────────────────────────────────

class _StopDet:
    def __init__(self, fire=False):
        self.armed = False
        self.fire = fire
        self.fed = 0

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False

    def feed(self, audio):
        from voice.wake import WakeDetection
        self.fed += 1
        return [WakeDetection("lloyd_stop", 0.9, 0, 0.0)] if self.fire and self.armed else []


def test_the_pipeline_feeds_the_stop_word_and_reports_it():
    from voice.pipeline import HearingPipeline
    det = _StopDet(fire=True)
    p = HearingPipeline(in_rate=16000, stop=det)
    frame = np.zeros(1600, dtype=np.int16)
    assert [e.kind for e in p.feed(frame, 16000) if e.kind == "stop"] == []
    det.arm()
    assert [e.kind for e in p.feed(frame, 16000) if e.kind == "stop"] == ["stop"]
    assert det.fed == 2, "fed while disarmed too, so it is warm when armed"


def test_it_is_armed_exactly_while_lloyd_speaks():
    b = _bridge([])
    det = _StopDet()
    b._hearing["user-a"] = type("H", (), {"pipeline": type("P", (), {"stop": det})()})()
    b.tts = _TTS(speaking=True)
    b._sync_stop_arming()
    assert det.armed
    b.tts.is_paused = True
    b._sync_stop_arming()
    assert not det.armed, "a paused reply is already waiting on the speaker"
    b.tts.is_paused = False
    b.tts.is_speaking = False
    b._sync_stop_arming()
    assert not det.armed


def test_a_stop_word_stops_lloyd_for_his_speaker_only():
    from voice.wake import WakeDetection
    for who, stopped in (("user-a", 1), ("user-b", 0)):
        b = _barge_bridge([])

        async def go():
            b._on_stop_word(HearingEvent("stop", detection=WakeDetection("lloyd_stop", 0.8, 0, 0.0)), who)
            await asyncio.sleep(0.05)
        asyncio.run(go())
        assert b.tts.interrupts == stopped, who
        assert b.http.cancels == (["t-old"] if stopped else []), who


# ── the latency row: kept, not just logged (#2273) ──────────────────────────
#
# The breakdown above used to exist only as one LOG.info and a grep in
# scripts/voice/e2e_voice.py, so every latency number this project ever quoted came
# from a synthetic rig. These are the tests for the other half: the same turn leaves
# a machine-readable row on disk, and the row's failures are never the reason a
# reply stops being spoken.

from app import voice_turns  # noqa: E402  the SAME module the worker writes through
from voice.timeline import STAGES  # noqa: E402  the order the row must be in


def _rows(store):
    """The rows in `store`, malformed lines dropped — the reader's own view of it."""
    return [row for _number, row in voice_turns.read_rows(store)]


def test_the_row_shape_carries_every_stage_and_the_turn_in_both_clocks():
    tl = TurnTimeline("voice")
    base = 100.0
    for i, stage in enumerate(["speech_end", "vad_close", "asr_done",
                               "inject_sent", "first_delta", "first_clause"]):
        tl.mark(stage, base + i * 0.1)
    tl.audio_pushed(base + 0.7, base + 0.75, 0.1)     # the reply starts playing 750 ms in
    tl.audio_pushed(base + 2.0, base + 2.0, 0.1)      # then the next frame goes out cold
    # ...which is the tool-turn silence the gap figure exists to catch: 1.15 s between
    # the end of the first frame's audio (base + 0.85) and the start of the second.
    # `first_pushed`/`first_played` stay the FIRST frame's marks, so this does not
    # move the total — the gap is the second thing the row carries about the audio.

    row = voice_turns.turn_row(tl, room="lloyd-x", epoch=1759732200.5)
    assert row["label"] == "voice"
    # In STAGES order, restricted to the marks this turn has. `contributions()` omits
    # the first recorded stage for want of an earlier one to measure from; a row that
    # omitted it too would read as though `speech_end` never happened.
    assert list(row["stages"]) == ["speech_end", "vad_close", "asr_done",
                                   "inject_sent", "first_delta", "first_clause",
                                   "first_pushed", "first_played"]
    assert row["stages"]["speech_end"] == 0.0, "the origin costs nothing"
    assert row["stages"]["vad_close"] == pytest.approx(0.1)
    assert row["stages"]["first_pushed"] == pytest.approx(0.2)
    assert row["stages"]["first_played"] == pytest.approx(0.05)
    assert sum(row["stages"].values()) == pytest.approx(0.75) == pytest.approx(
        row["eos_to_audio"]), "the stages are a decomposition of the total, not a second measurement"
    assert row["max_gap_s"] == pytest.approx(1.15)
    # The wall clock is one instant spelled twice: the sortable half a join needs,
    # and the half a human can read with its timezone written on it.
    assert row["epoch"] == pytest.approx(1759732200.5)
    assert row["at"].endswith("Z")
    from datetime import datetime
    assert datetime.fromisoformat(
        row["at"].replace("Z", "+00:00")).timestamp() == pytest.approx(
        row["epoch"]), "at and epoch must never be two different measurements"


def test_eos_to_audio_is_null_and_never_zero_when_either_end_is_missing():
    # A typed turn spoken out loud has no end of speech; a turn cancelled before any
    # audio has no first played. A 0.0 there would be counted by the trend as the
    # fastest turn on the box, so `None` is the only acceptable answer.
    typed = TurnTimeline("typed:user")
    typed.mark("first_clause", 5.0)
    typed.audio_pushed(5.1, 5.2, 0.5)
    row = voice_turns.turn_row(typed, room="lloyd-x", epoch=0.0)
    assert row["eos_to_audio"] is None
    assert row["stages"]["first_clause"] == 0.0, "the origin of a partial turn is still its first mark"
    assert row["stages"]["first_pushed"] == pytest.approx(0.1)
    assert row["max_gap_s"] == 0.0, "one frame cannot contain a silence in it"

    cancelled = TurnTimeline("voice")
    cancelled.mark("speech_end", 10.0)
    cancelled.mark("vad_close", 10.1)
    partial = voice_turns.turn_row(cancelled, room="lloyd-x", epoch=0.0)
    assert partial["eos_to_audio"] is None
    assert partial["stages"] == {"speech_end": 0.0, "vad_close": pytest.approx(0.1)}


def test_a_row_the_worker_stamps_lands_on_the_instant_it_was_built():
    # The worker passes no epoch, so the stamp is the clock at the
    # moment the row is made. `+00:00` spelled as `Z` is what keeps a later reader
    # from reading a naive local time as UTC (the 2026-09-21 class rule).
    import time as _time
    from datetime import datetime, timezone
    before = _time.time()
    row = voice_turns.turn_row(TurnTimeline("voice"), room="lloyd-x")
    after = _time.time()
    assert before <= row["epoch"] <= after
    assert row["at"] == datetime.fromtimestamp(
        row["epoch"], timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z")


def test_the_rows_measurement_half_is_the_timelines_own_as_dict():
    """Clause 1's home: `as_dict` is on `TurnTimeline`, and the store copies it verbatim.

    The alternative this deliberately is not: the store restating the stage order, or
    computing deltas from `marks` itself. Either would put the row's numbers where the
    marks are not, and a stage added to `STAGES` would then drop out of every row's
    ordering silently — the trend would read a missing stage rather than a new one. So
    the order asserted here is `STAGES` itself, filtered to what this turn reached, and
    the store's row is checked to carry the timeline's dict unchanged.
    """
    tl = TurnTimeline("voice")
    tl.mark("speech_end", 10.0)
    tl.mark("vad_close", 10.2)
    tl.audio_pushed(10.6, 10.7, 0.4)

    payload = tl.as_dict(epoch=1759732200.5)
    assert sorted(payload) == ["at", "eos_to_audio", "epoch", "label", "max_gap_s",
                               "stages"], "the keys are the documented six, no more"
    assert payload["label"] == "voice"
    assert list(payload["stages"]) == [s for s in STAGES if s in tl.marks], (
        f"as_dict ordered {list(payload['stages'])}, STAGES orders "
        f"{[s for s in STAGES if s in tl.marks]}")
    assert payload["stages"]["speech_end"] == 0.0, "the origin is recorded, not omitted"
    assert payload["at"].endswith("Z") and payload["epoch"] == 1759732200.5

    row = voice_turns.turn_row(tl, turn_id="abc123def456", room="lloyd-1",
                               epoch=1759732200.5)
    for key, value in payload.items():
        assert row[key] == value, f"the store changed the timeline's own {key!r}"
    assert row["turn_id"] == "abc123def456" and row["room"] == "lloyd-1"


def test_the_row_shape_is_read_off_the_real_timeline():
    # The row's measurement half comes from the timeline, not from anything the store
    # invents: the keys are the documented set, the stage numbers are the timeline's own
    # contributions plus its origin, and `eos_to_audio`/`max_gap_s` are the same two
    # numbers `summary()` prints. If the store started naming a stage it had not measured,
    # or a bound keyed on a field that never appears, this is what notices.
    tl = TurnTimeline("voice")
    tl.mark("speech_end", 10.0)
    tl.mark("first_clause", 11.0)
    tl.audio_pushed(11.4, 11.5, 0.3)
    row = voice_turns.turn_row(tl, turn_id="abc123def456", room="lloyd-1",
                               interrupted=True, queued_behind=True, tools_ran=True,
                               spoken_chars=42, epoch=1759732200.5)
    assert row.pop("at") == "2025-10-06T06:30:00.500Z"
    assert row == {
        "v": voice_turns.SCHEMA_VERSION, "turn_id": "abc123def456",
        "room": "lloyd-1", "interrupted": True, "queued_behind": True,
        "tools_ran": True, "spoken_chars": 42, "capped": False, "label": "voice",
        "epoch": 1759732200.5,
        # Monotonic differences are exact to the float and approximated to the reader.
        "stages": {"speech_end": 0.0, "first_clause": pytest.approx(1.0),
                   "first_pushed": pytest.approx(0.4),
                   "first_played": pytest.approx(0.1)},
        "eos_to_audio": pytest.approx(1.5), "max_gap_s": 0.0}


def test_a_completed_spoken_turn_appends_exactly_one_row(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))

    b, rs = _speaker()

    async def go():
        rs.start()
        await rs.on_event("voice_turn", {"turn_id": "t7"})
        await rs.on_event("session", {})
        await rs.on_event("text_delta", {"text": "It is noon. "})
        await rs.finish()
        await asyncio.sleep(0.05)   # the record task runs once the reply stops playing
        return b
    b = asyncio.run(go())

    assert b.tts.said == ["It is noon."], "the reply was spoken"
    store = voice_turns.turns_path()
    assert store == tmp_path / "voice" / "turns.jsonl", \
        "the store is under the data root, never the checkout"
    assert store.is_file(), "the writer creates the store's directory on the first row"
    assert store.read_text().count("\n") == 1, "one JSON object per line"
    rows = _rows(store)
    assert len(rows) == 1, f"one spoken turn, one row: {rows}"
    row = rows[0]
    assert row["v"] == voice_turns.SCHEMA_VERSION
    # The join to the session's turn record: the id the backend minted in the
    # `voice_turn` head, with the room beside it. Not invented for the row.
    assert row["turn_id"] == "t7"
    assert row["room"] == b.room_name
    assert row["label"] == "voice"
    assert row["interrupted"] is False
    assert row["queued_behind"] is False
    assert row["tools_ran"] is False
    assert row["spoken_chars"] == len("It is noon.")
    assert row["capped"] is False
    assert row["stages"]["voice_turn"] == 0.0, "this turn's first mark is its origin"
    assert "first_clause" in row["stages"]
    assert "asr_done" not in row["stages"], \
        "a turn that never went through the microphone has no ASR stage to report"
    assert row["eos_to_audio"] is None, "and no end of speech to measure from"


def test_a_turn_that_ran_a_tool_is_recorded_as_running_tools(tmp_path, monkeypatch):
    # The gap during tool work is the stage a #1163-class fix claims to move, so the
    # row has to say which turns did that work. `tool_phase` cannot answer it: the
    # first text delta clears that flag, and by the time the row is built it is False.
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))

    b, rs = _speaker(filler={"after_seconds": 99})

    async def go():
        await rs.on_event("voice_turn", {"turn_id": "t8", "queued_behind": True})
        await rs.on_event("tool_start", {"summary": "Checking supervisor status"})
        await rs.on_event("text_delta", {"text": "Supervisor is up. "})
        await rs.finish()
        await asyncio.sleep(0.05)
        return b
    asyncio.run(go())

    row = _rows(voice_turns.turns_path())[0]
    assert row["tools_ran"] is True
    assert row["queued_behind"] is True, "running behind another turn is its own explanation"
    assert row["turn_id"] == "t8"


def test_a_reply_a_barge_in_cancelled_still_leaves_one_row(tmp_path, monkeypatch):
    # The cancelled turn is what the missing-stage rate is for: a store that only
    # kept the turns that replied would compute every percentile over its own
    # survivors and call the result a trend.
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))

    b, rs = _speaker()

    async def go():
        rs.start()
        await rs.on_event("voice_turn", {"turn_id": "t9"})
        await rs.on_event("text_delta", {"text": "The supervisor is. "})
        b.tts.generation += 1        # a barge-in took the generation: this reply is over
        await rs.finish()
        await asyncio.sleep(0.05)
    asyncio.run(go())

    rows = _rows(voice_turns.turns_path())
    assert len(rows) == 1
    assert rows[0]["interrupted"] is True


def test_a_store_that_raises_still_leaves_the_reply_spoken(tmp_path, monkeypatch):
    # Clause 3. The row is a measurement of a turn, never a participant in one: the
    # failure has to die here rather than travel out of an asyncio task nobody is
    # awaiting, where it would become an unretrieved exception and no turn at all.
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))

    def boom(row):
        raise RuntimeError("the disk is gone")
    monkeypatch.setattr(voice_turns, "append_turn", boom)

    b, rs = _speaker()

    async def go():
        rs.start()
        await rs.on_event("voice_turn", {"turn_id": "t10"})
        await rs.on_event("text_delta", {"text": "It is noon. "})
        await rs.finish()
        await asyncio.sleep(0.05)
        return b
    b = asyncio.run(go())                      # must not raise
    assert b.tts.said == ["It is noon."], "every clause still spoke"
    assert not voice_turns.turns_path().exists(), "and nothing was half-written"


def test_a_store_path_that_is_not_a_directory_is_swallowed_too(tmp_path, monkeypatch):
    # The other shape of the same clause: the write itself fails, not a monkeypatch.
    # `mkdir` raises FileExistsError on a file in the way, and `append_turn` owns the
    # swallow so no caller has to remember it.
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))
    (tmp_path / "voice").write_text("not a directory")
    assert voice_turns.append_turn({"v": 1, "epoch": time.time()}) is False


def test_the_latency_log_line_still_fires_beside_the_row(tmp_path, monkeypatch, caplog):
    # e2e_voice.py's latency_lines() greps `[latency] ` out of a canary rig's
    # worker.log. The row is additive: if this line ever goes, the rig goes blind and
    # nothing in this file would notice.
    import logging
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))

    b, rs = _speaker()

    async def go():
        rs.start()
        await rs.on_event("voice_turn", {"turn_id": "t11"})
        await rs.on_event("text_delta", {"text": "It is noon. "})
        await rs.finish()
        await asyncio.sleep(0.05)
    with caplog.at_level(logging.INFO, logger="lloyd-agent-worker"):
        asyncio.run(go())

    assert "[latency]" in caplog.text, "the grep consumer this row has to coexist with"
    assert len(_rows(voice_turns.turns_path())) == 1, "and the kept half, in the same breath"
