"""Lloyd's hearing pipeline — the pieces between a LiveKit audio frame and a
transcript the gate can judge.

Everything here is deliberately free of LiveKit and of `livekit_worker`, so the
parts can be replayed offline against recorded audio. That is the whole point:
the 2026-09-17 review could measure what the old inline pipeline did only by
reading a log, because none of it could be constructed without a room.

The stages, in the order a sample passes through them:

    StreamResampler   48 kHz int16 frames -> a continuous 16 kHz float32 stream
    ContinuousWakeWord   fed every frame, never reset mid-stream
    SileroSegmenter      speech/silence, emitting utterances with sample offsets
    SmartTurn            "was that a finished thought?" on a closed utterance
    asr.Recognizer       the transcript, once something has decided to ask

`HearingPipeline` in `pipeline.py` is the one object that owns all five and
keeps their sample clocks in step.
"""

from .resample import StreamResampler, to_float32, to_int16
from .wake import ContinuousWakeWord, WakeDetection, WakeWordFactory
from .vad import SileroSegmenter, Utterance
from .turn import SmartTurn, TurnVerdict
from .pipeline import HearingPipeline, HearingEvent

__all__ = [
    "StreamResampler",
    "to_float32",
    "to_int16",
    "ContinuousWakeWord",
    "WakeWordFactory",
    "WakeDetection",
    "SileroSegmenter",
    "Utterance",
    "SmartTurn",
    "TurnVerdict",
    "HearingPipeline",
    "HearingEvent",
]
