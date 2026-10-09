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


# --- turn-1-mini ----------------------------------------------------------------

#: One 160 ms step of 16 kHz audio: the ONNX graph's fixed input length.
T1M_STEP = 2560
#: One 20 ms frame: the graph answers one probability per frame, 8 per step.
T1M_FRAME = 320


class Turn1Mini:
    """p99lab's turn-1-mini — a streaming end-of-turn model, audio only.

    5.3M parameters, Apache-2.0, 2 causal convolutions and 6 transformer
    layers with 1.2 s of causal attention; the 4 kHz low-pass it was trained
    on and the log-mel are inside the ONNX graph, so raw 16 kHz audio goes
    in. It answers once per 160 ms step with eight P(end of turn) values, one
    per 20 ms frame, and carries its state between steps, so a live stream
    costs one step per 160 ms (~2 ms on one core here) whatever its length.

    On LiveKit's eot-bench (real human-to-agent conversation, English) it cuts
    the speaker off 22.6% of the time at a 300 ms decision budget against
    Smart Turn v3.2's 35.2%, and its scores are *not* calibrated: the
    benchmark's 300 ms operating point used a threshold of 0.06. Calibrate
    `threshold` on this house's audio (`scripts/voice/vad_eval.py gate`).

    `predict(audio)` keeps `SmartTurn`'s contract — a whole buffer in, one
    verdict out — by running the buffer from a zero state and reading the
    score at its last real frame; `stream()` is the per-stream object the
    worker would feed frame by frame. Fails open like `SmartTurn`.
    """

    streaming = True

    def __init__(self, model_path: str | Path, threshold: float = 0.5) -> None:
        self.model_path = Path(model_path).expanduser()
        self.threshold = float(threshold)
        self._session = None
        self._state_inputs: list = []
        self._outputs: list[str] = []

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        if not self.model_path.exists():
            raise RuntimeError(f"turn-1-mini model not found: {self.model_path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        t0 = time.monotonic()
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._state_inputs = [i for i in self._session.get_inputs() if i.name != "audio"]
        self._outputs = [o.name for o in self._session.get_outputs()]
        assert self._outputs[0] == "probs", self._outputs
        LOG.info("turn-1-mini loaded in %.2fs from %s",
                 time.monotonic() - t0, self.model_path.name)

    def zero_state(self) -> dict:
        return {i.name: np.zeros(i.shape, dtype=np.float32) for i in self._state_inputs}

    def step(self, chunk: np.ndarray, state: dict) -> tuple[np.ndarray, dict]:
        """One 160 ms step: (8 probabilities, oldest first; the new state)."""
        out = self._session.run(self._outputs,
                                {"audio": chunk.reshape(1, T1M_STEP).astype(np.float32), **state})
        new = {n.replace("new_state_", "state_"): v for n, v in zip(self._outputs[1:], out[1:])}
        return np.asarray(out[0]).reshape(-1), new

    def score(self, audio_16k: np.ndarray) -> float:
        """P(end of turn) at the last real frame of `audio_16k`, from a zero
        state — the benchmarked read-out, over the buffer the gate holds."""
        self.load()
        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        state = self.zero_state()
        probs = None
        full = (x.size // T1M_STEP) * T1M_STEP
        for i in range(0, full, T1M_STEP):
            probs, state = self.step(x[i:i + T1M_STEP], state)
        rem = x.size - full
        if rem > 0:
            # The graph's own rule for a moment between step boundaries: pad
            # the step with zeros and run it on a copy of the state, reading
            # the score at the last frame that holds real audio.
            chunk = np.zeros(T1M_STEP, dtype=np.float32)
            chunk[:rem] = x[full:]
            probs, _ = self.step(chunk, state)
            last = min(7, (rem + T1M_FRAME - 1) // T1M_FRAME - 1)
            return float(probs[last])
        if probs is None:
            return 0.0
        return float(probs[-1])

    def predict(self, audio_16k: np.ndarray) -> TurnVerdict:
        t0 = time.monotonic()
        try:
            p = self.score(audio_16k)
        except Exception as e:
            LOG.warning("turn-1-mini predict failed (%s) — treating turn as complete", e)
            return TurnVerdict(True, 1.0, (time.monotonic() - t0) * 1000, ran=False)
        return TurnVerdict(complete=p >= self.threshold, probability=p,
                           elapsed_ms=(time.monotonic() - t0) * 1000)

    def stream(self) -> "Turn1MiniStream":
        self.load()
        return Turn1MiniStream(self)


class Turn1MiniStream:
    """One audio stream's state. `push` takes any length of 16 kHz float audio
    and returns one probability per finished 20 ms frame; `latest` is the
    most recent one, and `reset` starts a new stream."""

    def __init__(self, model: Turn1Mini) -> None:
        self._m = model
        self.reset()

    def reset(self) -> None:
        self._state = self._m.zero_state()
        self._buf = np.zeros(0, dtype=np.float32)
        self.latest = 0.0

    def push(self, audio_16k: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, np.asarray(audio_16k, dtype=np.float32).reshape(-1)])
        out = []
        while self._buf.size >= T1M_STEP:
            probs, self._state = self._m.step(self._buf[:T1M_STEP], self._state)
            self._buf = self._buf[T1M_STEP:]
            out.append(probs)
        if out:
            probs = np.concatenate(out)
            self.latest = float(probs[-1])
            return probs
        return np.zeros(0, dtype=np.float32)

    def peek(self) -> float:
        """P(end of turn) at the most recent frame pushed, including audio
        still short of a full step: the graph's own rule — pad the step with
        zeros and run it on a copy of the state, which is then discarded."""
        rem = self._buf.size
        if rem < T1M_FRAME:
            return self.latest
        chunk = np.zeros(T1M_STEP, dtype=np.float32)
        chunk[:rem] = self._buf
        probs, _ = self._m.step(chunk, self._state)
        return float(probs[min(7, rem // T1M_FRAME - 1)])


TURN_BACKENDS = ("smart-turn", "turn-1-mini")
DEFAULT_TURN1MINI = "agent-services/models/turn-1-mini/onnx/turn-1-mini.step.onnx"


def build_turn_detector(cfg: dict):
    """From the `livekit.turn_detection` block: `backend` picks the model
    (`smart-turn`, the default, or `turn-1-mini`); None when disabled."""
    if not cfg.get("enabled", False):
        return None
    backend = str(cfg.get("backend", "smart-turn"))
    if backend == "turn-1-mini":
        det = Turn1Mini(model_path=cfg.get("model_path", DEFAULT_TURN1MINI),
                        threshold=float(cfg.get("threshold", 0.5)))
        det.load()
        return det
    if backend != "smart-turn":
        raise ValueError(f"turn_detection.backend must be one of {TURN_BACKENDS}, not {backend!r}")
    return build_smart_turn(cfg)
