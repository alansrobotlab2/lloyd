"""The hearing pipeline: one audio stream in, decisions out.

This object exists so the hearing path can be replayed without a LiveKit room.
The 2026-09-17 review could only measure the old inline pipeline by parsing a
log file, because nothing in it could be constructed offline — which is also
why nobody noticed for three weeks that the wake word had fired five times.

The clock is the thing it really owns. The wake word, the segmenter and the
caller all need to agree on *where* in the stream something happened, and they
consume audio at three different frame sizes (1280, 512, and whatever LiveKit
sends). Every index here is 16 kHz samples since the stream opened.

Ordering matters and is deliberate: the wake word is fed **before** the
segmenter, and a wake event is emitted the moment it fires. The old worker
could not report a wake until the VAD had closed the utterance and Whisper had
run, so "Listening" appeared roughly half a second after the user stopped
talking. Now it appears while they are still saying the word.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .resample import StreamResampler
from .vad import SileroSegmenter, Utterance
from .wake import ContinuousWakeWord, WakeDetection

LOG = logging.getLogger("lloyd-agent-worker.pipeline")

#: Audio handed to a new partials stream ahead of the frame that opened it.
STREAM_PREROLL_S = 0.5

SAMPLE_RATE = 16000

#: How far after an utterance's end a wake detection may still be attributed to
#: it. openWakeWord stamps a fire at the end of the 80 ms frame that crossed
#: the threshold and needs ~1 s of context to reach it, so a short "hey Lloyd"
#: can be closed by the VAD a frame or two before the score peaks.
WAKE_ATTACH_GRACE = int(0.7 * SAMPLE_RATE)

#: A detection older than this is dropped rather than attached to whatever
#: comes next — without it, a wake that produced no utterance would silently
#: open the window on an unrelated sentence a minute later.
WAKE_MAX_AGE = int(3.0 * SAMPLE_RATE)


@dataclass
class HearingEvent:
    """Something the gate may want to act on.

    kind:
      ``wake``      — the wake word fired. `detection` is set. Emitted
                      immediately, before any transcript exists.
      ``speech``    — a rising edge into speech. `sample` is set. This is the
                      barge-in signal.
      ``silence``   — the falling edge after it: the segmenter left speech.
                      A paused reply waits this long, plus its false-
                      interruption timeout, for the utterance to decide.
      ``utterance`` — a closed span of speech. `utterance` is set, and `wake`
                      carries the detection that fell inside it, if any.
      ``partial``   — a running hypothesis from the streaming recogniser.
    """

    kind: str
    sample: int = 0
    at: float = field(default_factory=time.monotonic)
    detection: Optional[WakeDetection] = None
    utterance: Optional[Utterance] = None
    wake: Optional[WakeDetection] = None
    text: str = ""


class HearingPipeline:
    """Owns one participant's audio. Not thread-safe; feed it from one place.

    Every component below holds per-stream state — Silero's recurrent state,
    openWakeWord's feature window, the resampler's filter history — so a
    pipeline is per audio stream and never shared between participants.
    """

    def __init__(
        self,
        in_rate: int,
        wake: Optional[ContinuousWakeWord] = None,
        segmenter: Optional[SileroSegmenter] = None,
        streaming: Optional[Any] = None,
    ) -> None:
        self.resampler = StreamResampler(in_rate)
        self.wake = wake
        self.segmenter = segmenter if segmenter is not None else SileroSegmenter()
        self.streaming = streaming
        self._stream_handle = None
        self._pending: list[WakeDetection] = []
        self._was_speaking = False
        self._partial = ""
        # The last STREAM_PREROLL_S of audio, handed to a new partials stream
        # before the frame that opened it. A cache-aware streaming encoder
        # drops the first ~0.3 s of a stream, and the stream only opens once
        # the VAD has already heard speech, so without this every caption lost
        # its first word ("what is the weather" -> "the weather", measured
        # 2026-09-24 on tests/fixtures/voice/complete_16k.wav).
        self._ring: list[np.ndarray] = []
        self._ring_len = 0

    @property
    def cursor(self) -> int:
        return self.segmenter.cursor

    def close(self) -> Optional[Utterance]:
        """Flush a half-spoken utterance when the participant leaves."""
        return self.segmenter.flush()

    def feed(self, frame: np.ndarray, in_rate: int) -> list[HearingEvent]:
        events: list[HearingEvent] = []
        audio = self.resampler.push(frame, in_rate)
        if audio.size == 0:
            return events

        # 1. Wake word first, so a fire is reported at the moment it happens
        #    rather than after the VAD and the ASR have had their turn.
        if self.wake is not None:
            for det in self.wake.feed(audio):
                self._pending.append(det)
                events.append(
                    HearingEvent("wake", sample=det.sample_index, detection=det)
                )

        # 2. Segmentation.
        utterances = self.segmenter.feed(audio)

        # A rising edge into speech is the barge-in signal, and it has to be
        # reported before the utterance closes or it is useless for that.
        speaking = self.segmenter.in_speech
        if speaking and not self._was_speaking:
            events.append(HearingEvent("speech", sample=self.segmenter.cursor))
            # The peak a drop reports must describe THIS utterance. Carried
            # over, the rest of an earlier "hey Lloyd" (which keeps scoring
            # after its fire, inside the refractory window) was reported
            # against the next, unaddressed sentence as a 0.94.
            if self.wake is not None:
                self.wake.take_peak()
        elif self._was_speaking and not speaking:
            events.append(HearingEvent("silence", sample=self.segmenter.cursor))
        self._was_speaking = speaking

        # 3. Partial hypotheses, only while someone is talking.
        if self.streaming is not None and speaking:
            try:
                feed = audio
                if self._stream_handle is None:
                    self._stream_handle = self.streaming.create_stream()
                    if self._ring:
                        feed = np.concatenate(self._ring + [audio])
                text = self.streaming.accept(self._stream_handle, feed)
                if text and text != self._partial:
                    self._partial = text
                    events.append(HearingEvent("partial", text=text))
            except Exception as e:
                LOG.warning("streaming asr failed: %s — disabling partials", e)
                self.streaming = None

        if self.streaming is not None:
            self._ring.append(audio)
            self._ring_len += audio.size
            cap = int(STREAM_PREROLL_S * 16000)
            while self._ring and self._ring_len - self._ring[0].size >= cap:
                self._ring_len -= self._ring.pop(0).size

        for utt in utterances:
            det = self._claim_wake(utt)
            events.append(HearingEvent("utterance", utterance=utt, wake=det,
                                       sample=utt.end_sample))
            if self.streaming is not None and self._stream_handle is not None:
                try:
                    self.streaming.reset(self._stream_handle)
                except Exception:
                    self._stream_handle = None
                self._partial = ""
        return events

    def _claim_wake(self, utt: Utterance) -> Optional[WakeDetection]:
        """Attach a pending detection to this utterance, and expire stale ones.

        Newest-first, because when somebody says the wake word twice the second
        one is the one they meant.
        """
        cursor = self.segmenter.cursor
        self._pending = [
            d for d in self._pending if cursor - d.sample_index <= WAKE_MAX_AGE
        ]
        for i in range(len(self._pending) - 1, -1, -1):
            if utt.contains(self._pending[i].sample_index, grace=WAKE_ATTACH_GRACE):
                return self._pending.pop(i)
        return None
