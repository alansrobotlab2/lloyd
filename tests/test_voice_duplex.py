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
