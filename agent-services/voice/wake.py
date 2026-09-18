"""openWakeWord, fed continuously instead of swept per utterance.

The old `AcousticWakeWord` waited for the energy VAD to close an utterance,
called `Model.reset()`, and swept `predict()` across it. Three things were
wrong with that, and together they are why 949 utterances produced 5 fires:

1. **It could only fire after the VAD had already decided.** The wake word was
   therefore gated on the very component that was dropping the audio — an
   utterance the segmenter never emitted, or emitted as a fragment, could not
   be woken on at all.
2. **`reset()` did not do what the call assumed.** In openwakeword 0.4.0 it
   clears only `prediction_buffer`; the mel and embedding buffers inside
   `AudioFeatures` keep the *previous* utterance, so each sweep began with 400 ms
   of someone else's audio in the feature window. 0.6.0 added
   `self.preprocessor.reset()`, which is the fix — and which makes resetting
   per utterance worse, not better, because of (3).
3. **`predict()` zeroes its first 5 frames after a reset.** That is 400 ms of
   guaranteed-zero score at the start of every sweep, against a 200 ms lead-in.
   The wake word was being discarded precisely where it was spoken.

Feeding continuously fixes all three: the model keeps one rolling feature
window over the whole conversation, the 400 ms warmup is paid once when a
participant joins, and detection happens at the moment the word is said rather
than 500 ms after the speaker stops.

**One model per audio stream.** `Model` holds mutable buffers and is not
thread-safe; the old code shared a single instance across every room and called
it from `asyncio.to_thread`, so two concurrent utterances interleaved their
feature windows. `WakeWordFactory` validates the files once and hands each
stream its own.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .resample import FrameChunker, to_int16

LOG = logging.getLogger("lloyd-agent-worker.wake")

# openWakeWord's native frame: 1280 samples = 80 ms at 16 kHz.
FRAME_SAMPLES = 1280

# `predict()` returns 0.0 for the first 5 frames after a reset, so a freshly
# built model is deaf for this long. Paid once per participant, not per
# utterance — which is the entire point of feeding continuously.
WARMUP_FRAMES = 5


@dataclass(frozen=True)
class WakeDetection:
    """A wake word, and where in the stream it was heard."""

    name: str
    score: float
    #: Index (16 kHz samples since the stream opened) of the END of the frame
    #: that crossed the threshold. The word itself spans roughly the preceding
    #: second, which is why the gate matches against an utterance's whole span
    #: rather than against this instant.
    sample_index: int
    at: float


class ContinuousWakeWord:
    """One openWakeWord model, fed the stream frame by frame.

    Not thread-safe, by design — construct one per audio stream and feed it
    from one place.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        refractory_ms: int = 1500,
        on_score: Optional[Callable[[str, float, int], None]] = None,
    ) -> None:
        self._model = model
        self.threshold = float(threshold)
        self._refractory_samples = int(refractory_ms / 1000.0 * 16000)
        self._chunker = FrameChunker(FRAME_SAMPLES)
        self._cursor = 0
        self._frames_seen = 0
        self._last_fire_sample = -(10 ** 9)
        self._on_score = on_score
        #: Highest score seen since the last `take_peak()`, for diagnostics —
        #: a miss is only debuggable if the near-misses were recorded.
        self._peak = 0.0
        self._peak_name = ""

    @property
    def warm(self) -> bool:
        return self._frames_seen >= WARMUP_FRAMES

    def take_peak(self) -> tuple[str, float]:
        """Read and clear the running peak score."""
        out = (self._peak_name, self._peak)
        self._peak, self._peak_name = 0.0, ""
        return out

    def feed(self, samples_16k: np.ndarray) -> list[WakeDetection]:
        """Push 16 kHz audio (float32 or int16). Returns any detections."""
        hits: list[WakeDetection] = []
        for frame in self._chunker.push(samples_16k):
            self._cursor += FRAME_SAMPLES
            self._frames_seen += 1
            try:
                scores = self._model.predict(to_int16(frame))
            except Exception as e:  # a wake word that raises must not kill audio
                LOG.warning("wake predict failed: %s", e)
                continue
            if not scores:
                continue
            name = max(scores, key=lambda k: scores[k])
            score = float(scores[name])
            if score > self._peak:
                self._peak, self._peak_name = score, name
            if self._on_score is not None:
                try:
                    self._on_score(name, score, self._cursor)
                except Exception:
                    pass
            if score < self.threshold:
                continue
            if self._cursor - self._last_fire_sample < self._refractory_samples:
                # Still inside the previous fire's shadow. openWakeWord scores
                # stay high for several frames after the word, so without this
                # one "hey Lloyd" opens the window three times and the diag log
                # reads as three separate wakes.
                continue
            self._last_fire_sample = self._cursor
            hits.append(WakeDetection(name, score, self._cursor, time.monotonic()))
        return hits

    @property
    def cursor(self) -> int:
        """16 kHz samples consumed since the stream opened."""
        return self._cursor


class WakeWordFactory:
    """Validates the model files once, then mints a model per stream.

    Also absorbs the openwakeword 0.4 -> 0.6 constructor change. 0.6.0 renamed
    `wakeword_model_paths` to `wakeword_models`, moved the engine paths into
    `**kwargs` under different names, and defaults `inference_framework` to
    tflite — which is not installed here, and which would otherwise surface as
    a confusing import error rather than "you upgraded the library".
    """

    def __init__(
        self,
        models_dir: str | Path,
        engine_dir: str | Path,
        threshold: float = 0.5,
        refractory_ms: int = 1500,
        verifier_models: Optional[dict] = None,
        verifier_threshold: float = 0.1,
    ) -> None:
        self.models_dir = Path(models_dir).expanduser()
        self.engine_dir = Path(engine_dir).expanduser()
        self.threshold = float(threshold)
        self.refractory_ms = int(refractory_ms)
        self.verifier_models = dict(verifier_models or {})
        self.verifier_threshold = float(verifier_threshold)
        self._paths: list[str] = []
        self._mel = self.engine_dir / "melspectrogram.onnx"
        self._emb = self.engine_dir / "embedding_model.onnx"

    def validate(self) -> None:
        """Raise if the model files are not where config says. Called once at
        startup so a typo is a boot error rather than a silently deaf worker."""
        self._paths = sorted(str(p) for p in self.models_dir.glob("*.onnx"))
        if not self._paths:
            raise RuntimeError(f"no wake-word .onnx models in {self.models_dir}")
        for p in (self._mel, self._emb):
            if not p.exists():
                raise RuntimeError(f"missing openWakeWord engine file: {p}")

    def _build_model(self):
        from openwakeword.model import Model

        if not self._paths:
            self.validate()
        common = {}
        if self.verifier_models:
            common["custom_verifier_models"] = self.verifier_models
            common["custom_verifier_threshold"] = self.verifier_threshold
        try:  # openwakeword >= 0.5
            return Model(
                wakeword_models=self._paths,
                inference_framework="onnx",
                melspec_model_path=str(self._mel),
                embedding_model_path=str(self._emb),
                **common,
            )
        except TypeError:  # openwakeword 0.4.x
            return Model(
                wakeword_model_paths=self._paths,
                melspec_onnx_model_path=str(self._mel),
                embedding_onnx_model_path=str(self._emb),
                **common,
            )

    def create(
        self, on_score: Optional[Callable[[str, float, int], None]] = None
    ) -> ContinuousWakeWord:
        t0 = time.monotonic()
        model = self._build_model()
        LOG.info(
            "openWakeWord model built in %.2fs (%s) threshold=%.2f",
            time.monotonic() - t0,
            ", ".join(model.models.keys()),
            self.threshold,
        )
        return ContinuousWakeWord(
            model,
            threshold=self.threshold,
            refractory_ms=self.refractory_ms,
            on_score=on_score,
        )


def build_factory(cfg: dict) -> Optional[WakeWordFactory]:
    """From the `livekit.acoustic_wake` block, or None when disabled.

    Unlike the old `_build_acoustic_wake_word`, a *misconfigured* factory is an
    error rather than a warning. Falling back to text matching was how the
    worker stayed silently deaf: `_strip_wake_word` cannot fire on a transcript
    that Whisper returned as an empty string, which is what a quiet "hey Lloyd"
    reliably produces.
    """
    if not cfg.get("enabled", True):
        LOG.info("acoustic wake-word disabled in config")
        return None
    f = WakeWordFactory(
        models_dir=cfg.get("models_dir", "agent-services/models/wakeword"),
        engine_dir=cfg.get("engine_dir", "agent-services/models/openwakeword"),
        threshold=float(cfg.get("threshold", 0.5)),
        refractory_ms=int(cfg.get("refractory_ms", 1500)),
        verifier_models=cfg.get("verifier_models") or {},
        verifier_threshold=float(cfg.get("verifier_threshold", 0.1)),
    )
    f.validate()
    return f
