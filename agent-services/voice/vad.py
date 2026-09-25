"""Silero VAD in place of the energy threshold.

The energy VAD was the root cause of the 2026-09-17 finding. Its `speech_rms`
was 0.025 while every wake-word utterance in the diagnostic corpus measured
`rms_mean` between 0.013 and 0.017 — so normal speaking volume at a normal
distance sat *below* the speech threshold. Only the loudest syllables crossed
it, which cut single sentences into 0.7-1.4 s fragments with the quiet parts
missing, and Whisper returned `''` for most of them. The `min_voiced_ratio`
check then dropped another 208 as "not speech".

Tuning the threshold cannot fix that, because the quantity is wrong: RMS
measures loudness, and the thing being asked is whether the sound is *speech*.
A whisper close to the mic and a fan at the same level are the same number.
Silero answers the actual question — 309k parameters, 0.14 ms per 32 ms frame
on CPU here, so it is free at this scale.

The segmenter keeps `speech_pad_ms` of audio from before the onset, which is
what the old `LEAD_IN_MS` ring was for. It does *not* reproduce
`min_voiced_ratio`: Silero's own probability is the voicing measure, and a
second one layered on top is what dropped a fifth of the corpus.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .resample import FrameChunker

LOG = logging.getLogger("lloyd-agent-worker.vad")

#: Silero v5/v6 accept exactly 512 samples at 16 kHz (32 ms). Anything else
#: raises inside the ONNX session.
FRAME_SAMPLES = 512
SAMPLE_RATE = 16000


@dataclass
class Utterance:
    """A closed span of speech, with its place in the stream."""

    audio: np.ndarray  #: float32 at 16 kHz, including the pads
    start_sample: int  #: index of `audio[0]` in the stream
    end_sample: int
    max_prob: float
    reason: str  #: "silence" | "max_duration"

    @property
    def duration_s(self) -> float:
        return self.audio.size / SAMPLE_RATE

    def contains(self, sample_index: int, grace: int = 0) -> bool:
        """Whether a stream position falls inside this utterance.

        `grace` extends the window forward, because a wake detection is stamped
        at the end of the 80 ms frame that crossed the threshold and openWakeWord
        needs roughly a second of context — so the fire can land slightly after
        the VAD has already closed a short "hey Lloyd".
        """
        return self.start_sample - grace <= sample_index <= self.end_sample + grace


@dataclass
class _State:
    triggered: bool = False
    silence_run: int = 0
    buf: list[np.ndarray] = field(default_factory=list)
    buf_len: int = 0
    start_sample: int = 0
    max_prob: float = 0.0


class SileroSegmenter:
    """Streaming speech segmentation.

    Feed 16 kHz float32; get back closed utterances. Not thread-safe: the
    underlying ONNX model carries per-stream recurrent state, so build one per
    audio stream.
    """

    def __init__(
        self,
        model=None,
        threshold: float = 0.5,
        min_silence_ms: int = 380,
        speech_pad_ms: int = 220,
        min_utterance_ms: int = 250,
        max_utterance_ms: int = 30000,
    ) -> None:
        self._model = model if model is not None else load_vad_model()
        self.threshold = float(threshold)
        #: Silero's own hysteresis: leaving speech needs a clearly lower score
        #: than entering it, so a brief dip mid-word does not close the turn.
        self.exit_threshold = max(0.0, self.threshold - 0.15)
        self._min_silence = int(min_silence_ms / 1000 * SAMPLE_RATE)
        self._pad = int(speech_pad_ms / 1000 * SAMPLE_RATE)
        self._min_utt = int(min_utterance_ms / 1000 * SAMPLE_RATE)
        self._max_utt = int(max_utterance_ms / 1000 * SAMPLE_RATE)
        self._chunker = FrameChunker(FRAME_SAMPLES)
        self._cursor = 0
        self._st = _State()
        self._preroll: list[np.ndarray] = []
        self._preroll_len = 0
        self._last_prob = 0.0

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def in_speech(self) -> bool:
        return self._st.triggered

    @property
    def last_prob(self) -> float:
        return self._last_prob

    def reset(self) -> None:
        self._model.reset_states()
        self._st = _State()
        self._preroll, self._preroll_len = [], 0
        self._chunker.reset()

    def _prob(self, frame: np.ndarray) -> float:
        return self._model(frame)

    def feed(self, samples_16k: np.ndarray) -> list[Utterance]:
        out: list[Utterance] = []
        for frame in self._chunker.push(samples_16k):
            self._cursor += FRAME_SAMPLES
            try:
                p = self._prob(np.ascontiguousarray(frame, dtype=np.float32))
            except Exception as e:
                LOG.warning("silero predict failed: %s", e)
                p = 0.0
            self._last_prob = p
            st = self._st

            if not st.triggered:
                self._preroll.append(frame)
                self._preroll_len += frame.size
                while self._preroll_len - self._preroll[0].size >= self._pad:
                    self._preroll_len -= self._preroll[0].size
                    self._preroll.pop(0)
                if p >= self.threshold:
                    st.triggered = True
                    st.buf = list(self._preroll)
                    st.buf_len = self._preroll_len
                    # The pre-roll already ends with this frame, so the
                    # buffer starts `preroll_len` before the cursor — not a
                    # frame earlier, which labelled every utterance 32 ms
                    # ahead of its audio (and the first one negative).
                    st.start_sample = self._cursor - self._preroll_len
                    st.max_prob = p
                    st.silence_run = 0
                    self._preroll, self._preroll_len = [], 0
                continue

            st.buf.append(frame)
            st.buf_len += frame.size
            st.max_prob = max(st.max_prob, p)
            if p >= self.exit_threshold:
                st.silence_run = 0
            else:
                st.silence_run += frame.size

            closed = st.silence_run >= self._min_silence
            capped = st.buf_len >= self._max_utt
            if not (closed or capped):
                continue

            audio = np.concatenate(st.buf) if st.buf else np.zeros(0, dtype=np.float32)
            reason = "silence" if closed else "max_duration"
            # The trailing silence that closed the turn is kept only up to the
            # pad; the rest is what the model needs to *decide*, not part of
            # what was said, and handing it to the ASR only invites a
            # hallucinated token.
            if closed and st.silence_run > self._pad:
                trim = st.silence_run - self._pad
                if trim < audio.size:
                    audio = audio[: audio.size - trim]
            utt = Utterance(
                audio=audio,
                start_sample=st.start_sample,
                end_sample=st.start_sample + audio.size,
                max_prob=st.max_prob,
                reason=reason,
            )
            self._st = _State()
            if capped:
                # A capped utterance is mid-sentence by definition, so the next
                # one continues immediately rather than waiting for a fresh
                # onset — otherwise the first word after the cap is lost.
                self._st.triggered = True
                self._st.start_sample = utt.end_sample
                self._st.max_prob = p
            if audio.size >= self._min_utt:
                out.append(utt)
            else:
                LOG.debug(
                    "vad: dropped %.2fs utterance below min (%.2fs)",
                    audio.size / SAMPLE_RATE, self._min_utt / SAMPLE_RATE,
                )
        return out

    def flush(self) -> Optional[Utterance]:
        """Close whatever is open — used when a participant leaves mid-sentence
        so the last thing they said is not silently discarded."""
        st = self._st
        if not st.triggered or not st.buf:
            return None
        audio = np.concatenate(st.buf)
        self._st = _State()
        if audio.size < self._min_utt:
            return None
        return Utterance(
            audio=audio,
            start_sample=st.start_sample,
            end_sample=st.start_sample + audio.size,
            max_prob=st.max_prob,
            reason="flush",
        )


#: Silero VAD v6 (MIT), vendored beside the wake-word models so a rebuild
#: cannot leave the worker deaf for want of a download.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "silero-vad" / "silero_vad.onnx"
_CONTEXT = 64  # samples of the previous frame the model is given, at 16 kHz


class SileroOnnx:
    """Silero VAD called through onnxruntime directly, with no torch.

    The `silero-vad` package does the same thing through a thin torch wrapper,
    and its install requirement on `torchaudio` pinned torch: installing it on
    2026-09-17 moved `.venvs/lloyd` from torch 2.11.0 (CUDA 13) to 2.9.1 (CUDA
    12) under every service in that venv. This is the wrapper's arithmetic —
    a (2, 1, 128) recurrent state and the last 64 samples of the previous frame
    prepended to each 512 — checked equal to the package's output to 1e-6
    before the package was removed (`test_silero_onnx_matches_the_reference`).
    """

    def __init__(self, path: Path = MODEL_PATH) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset_states()

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context,
                            np.asarray(frame, dtype=np.float32).reshape(1, -1)], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -_CONTEXT:]
        return float(out.reshape(-1)[0])


def load_vad_model() -> SileroOnnx:
    """Load Silero VAD. One per stream — it carries recurrent state."""
    return SileroOnnx()


def build_segmenter(cfg: dict) -> SileroSegmenter:
    """From the `livekit.vad` block."""
    return SileroSegmenter(
        threshold=float(cfg.get("threshold", 0.5)),
        min_silence_ms=int(cfg.get("min_silence_ms", 380)),
        speech_pad_ms=int(cfg.get("speech_pad_ms", 220)),
        min_utterance_ms=int(cfg.get("min_utterance_ms", 250)),
        max_utterance_ms=int(cfg.get("max_utterance_ms", 30000)),
    )
