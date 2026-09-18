"""Smart Turn v3 — "was that a finished thought?", from the waveform.

A VAD can only answer "has the sound stopped". Every pause in the middle of a
sentence looks identical to the end of one, so the silence threshold is a
straight trade: short enough to feel responsive, and it cuts people off
mid-sentence; long enough not to, and every turn costs that much dead air.
The old worker sat at 500 ms and did both badly.

Smart Turn is a Whisper-tiny encoder with a linear head over the last 8 s of
audio, trained to predict whether the speaker finished. That turns the trade
into a decision: close the turn after a short silence when the model says the
thought is complete, and wait out a long one when it does not. BSD-2-Clause,
8 MB, ~10-25 ms on CPU here.

Preprocessing has to match pipecat's `inference.py` exactly, because the head
is a linear probe on a frozen encoder and a mel that is off by a normalisation
constant is simply a different input. Specifically: keep the **last** 8 s, zero
-pad to 128000 samples, apply zero-mean unit-variance over the *real* samples
only, re-zero the padding, then Whisper's own log-mel. `FeatureExtractor` from
faster-whisper is that mel to the line — same filters, same `log10`, same
`max(x, x.max() - 8)`, same `(x + 4) / 4` — so the mel is not reimplemented
here, only the waveform normalisation transformers does around it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger("lloyd-agent-worker.turn")

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE
MEL_FRAMES = 800


@dataclass(frozen=True)
class TurnVerdict:
    complete: bool
    probability: float
    elapsed_ms: float
    #: False when the model could not be consulted at all, in which case
    #: `complete` is True — see `SmartTurn.predict`.
    ran: bool = True


class SmartTurn:
    """Semantic end-of-turn classifier. Thread-safe: ONNX Runtime sessions are,
    and the feature extractor holds no per-call state."""

    def __init__(self, model_path: str | Path, threshold: float = 0.5) -> None:
        self.model_path = Path(model_path).expanduser()
        self.threshold = float(threshold)
        self._session = None
        self._fe = None
        self._input_name = "input_features"

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from faster_whisper.feature_extractor import FeatureExtractor

        if not self.model_path.exists():
            raise RuntimeError(f"smart-turn model not found: {self.model_path}")
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.monotonic()
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._fe = FeatureExtractor(chunk_length=WINDOW_SECONDS)
        LOG.info("smart-turn loaded in %.2fs from %s",
                 time.monotonic() - t0, self.model_path.name)

    def _features(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size > WINDOW_SAMPLES:
            x = x[-WINDOW_SAMPLES:]
        n = x.size
        padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        padded[:n] = x
        if n > 0:
            # transformers' zero_mean_unit_var_norm: statistics from the real
            # samples, applied to the whole vector, padding re-zeroed after.
            mean = padded[:n].mean()
            var = padded[:n].var()
            padded = (padded - mean) / np.sqrt(var + 1e-7)
            padded[n:] = 0.0
        mel = self._fe(padded, padding=0)
        if mel.shape[-1] > MEL_FRAMES:
            mel = mel[..., :MEL_FRAMES]
        return mel.astype(np.float32)[None, ...]

    def predict(self, audio_16k: np.ndarray) -> TurnVerdict:
        """Probability that the speaker finished their turn.

        **Fails open.** A model that will not load, or an audio buffer it
        rejects, returns `complete=True` so the turn is committed on the VAD
        silence alone. The failure mode of the alternative is that Lloyd stops
        answering entirely, which is far worse than answering slightly early.
        """
        t0 = time.monotonic()
        try:
            self.load()
            feats = self._features(audio_16k)
            out = self._session.run(None, {self._input_name: feats})
            p = float(np.asarray(out[0]).reshape(-1)[0])
            if not (0.0 <= p <= 1.0):
                # The exported graph names its output `logits`; v3.x applies the
                # sigmoid inside the graph, so this is a guard against a future
                # export that does not, never a routine path.
                p = 1.0 / (1.0 + float(np.exp(-p)))
        except Exception as e:
            LOG.warning("smart-turn predict failed (%s) — treating turn as complete", e)
            return TurnVerdict(True, 1.0, (time.monotonic() - t0) * 1000, ran=False)
        return TurnVerdict(
            complete=p >= self.threshold,
            probability=p,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )


def build_smart_turn(cfg: dict) -> Optional[SmartTurn]:
    """From the `livekit.turn_detection` block, or None when disabled."""
    if not cfg.get("enabled", False):
        return None
    st = SmartTurn(
        model_path=cfg.get(
            "model_path", "agent-services/models/smart-turn/smart-turn-v3.2-cpu.onnx"
        ),
        threshold=float(cfg.get("threshold", 0.5)),
    )
    st.load()
    return st
