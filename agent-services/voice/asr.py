"""Speech recognition, behind one interface so the engine is a config key.

Two things the 2026-09-17 measurements said about `base.en` on this path:

* **It was slower on shorter audio.** Clips under 1 s averaged 2.76 s to
  transcribe and clips over 8 s averaged 0.57 s. The cause is faster-whisper's
  temperature fallback: the tightened `no_speech_threshold` / `log_prob_threshold`
  / `compression_ratio_threshold` that were added to stop hallucinations fail
  most often on short clips, and each failure re-decodes the whole clip at the
  next temperature. Up to six passes, and the gate the thresholds implement is
  exactly what a short utterance trips.
* **It was wrong.** `base.en` is the second-smallest Whisper, and the corpus is
  full of confident nonsense for near-silence.

Parakeet TDT 0.6B v3 through sherpa-onnx answers both: 6.3% WER against
`base.en`'s ~12%, and 0.06 s for a 1 s clip against 2.76 s — measured on this
box, int8, four CPU threads, no GPU. It is also a transducer, so it has no
temperature fallback to trip and no long-form decoder to hallucinate with.

Whisper stays as `backend: whisper` because it is the one engine that ships
hotword biasing on this path. That matters much less than it did: hotwords were
carrying the wake word, and the wake word is now decided acoustically.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

LOG = logging.getLogger("lloyd-agent-worker.asr")

SAMPLE_RATE = 16000


@dataclass
class Transcript:
    text: str
    latency_s: float
    backend: str


def looks_repetitive(text: str) -> bool:
    """Whisper's near-silence signature: a 1- or 2-word phrase four or more
    times in a row ("Thank you. Thank you. Thank you. Thank you.").

    Kept engine-agnostic and applied to every backend. Transducers do not
    hallucinate this way, but they do stutter on looped audio, and the check is
    cheap enough that exempting one engine would only invite the question.
    """
    if not text:
        return False
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < 4:
        return False
    for ngram in (1, 2):
        if len(words) < ngram * 4:
            continue
        run = 1
        prev = tuple(words[:ngram])
        for i in range(ngram, len(words) - ngram + 1, ngram):
            cur = tuple(words[i:i + ngram])
            if cur == prev:
                run += 1
                if run >= 4:
                    return True
            else:
                run = 1
            prev = cur
    return False


class Recognizer(Protocol):
    name: str

    def load(self) -> None: ...

    def transcribe(self, audio_16k: np.ndarray) -> Transcript: ...


class SherpaOfflineRecognizer:
    """NeMo transducer (Parakeet TDT v3) via sherpa-onnx.

    Audio must already be 16 kHz float32. sherpa will happily resample, but it
    builds a fresh resampler per stream and logs a paragraph about it each time
    — and the pipeline has already done the conversion once for everybody.
    """

    name = "parakeet"

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 4,
        decoding_method: str = "greedy_search",
        model_type: str = "nemo_transducer",
        provider: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.num_threads = int(num_threads)
        self.decoding_method = decoding_method
        self.model_type = model_type
        self.provider = provider
        self._rec = None

    def _pick(self, *names: str) -> str:
        for n in names:
            p = self.model_dir / n
            if p.exists():
                return str(p)
        raise RuntimeError(f"{self.model_dir}: none of {names} present")

    def load(self) -> None:
        if self._rec is not None:
            return
        import sherpa_onnx

        t0 = time.monotonic()
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self._pick("encoder.int8.onnx", "encoder.onnx"),
            decoder=self._pick("decoder.int8.onnx", "decoder.onnx"),
            joiner=self._pick("joiner.int8.onnx", "joiner.onnx"),
            tokens=self._pick("tokens.txt"),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method=self.decoding_method,
            model_type=self.model_type,
            provider=self.provider,
        )
        LOG.info("parakeet loaded in %.1fs from %s", time.monotonic() - t0, self.model_dir)

    def transcribe(self, audio_16k: np.ndarray) -> Transcript:
        self.load()
        t0 = time.monotonic()
        stream = self._rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.asarray(audio_16k, dtype=np.float32))
        self._rec.decode_stream(stream)
        text = (stream.result.text or "").strip()
        if looks_repetitive(text):
            text = ""
        return Transcript(text, time.monotonic() - t0, self.name)


class WhisperRecognizer:
    """faster-whisper, with the temperature fallback under a config key.

    `temperature_fallback: false` (the default) pins `temperature=0.0`, which
    is what turns a 2.76 s decode into a 0.62 s one. The fallback exists to
    rescue a bad decode by retrying hotter; on utterance-length audio it mostly
    just pays six times for the same answer.
    """

    name = "whisper"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self._model = None
        self.hotwords: Optional[str] = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        model = self.cfg.get("model", "base.en")
        device = self.cfg.get("device", "cpu")
        compute = self.cfg.get("compute_type", "int8")
        t0 = time.monotonic()
        self._model = WhisperModel(model, device=device, compute_type=compute)
        LOG.info("faster-whisper %s loaded in %.1fs", model, time.monotonic() - t0)

    def transcribe(self, audio_16k: np.ndarray) -> Transcript:
        self.load()
        t0 = time.monotonic()
        kwargs = dict(
            language=self.cfg.get("language", "en"),
            beam_size=int(self.cfg.get("beam_size", 1)),
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=float(self.cfg.get("no_speech_threshold", 0.7)),
            log_prob_threshold=float(self.cfg.get("log_prob_threshold", -0.8)),
            compression_ratio_threshold=float(
                self.cfg.get("compression_ratio_threshold", 2.0)
            ),
            without_timestamps=bool(self.cfg.get("without_timestamps", True)),
        )
        if not self.cfg.get("temperature_fallback", False):
            kwargs["temperature"] = 0.0
        if self.hotwords:
            kwargs["hotwords"] = self.hotwords
        segments, _ = self._model.transcribe(
            np.asarray(audio_16k, dtype=np.float32), **kwargs
        )
        text = "".join(s.text for s in segments).strip()
        if looks_repetitive(text):
            text = ""
        return Transcript(text, time.monotonic() - t0, self.name)


class SherpaStreamingRecognizer:
    """Cache-aware streaming FastConformer CTC, for partial hypotheses.

    Not the final transcript: CTC greedy at a 480 ms chunk is measurably worse
    than the offline transducer, and the offline pass costs ~60 ms anyway. This
    exists so there is something to show — and eventually something to start
    generating from — while the speaker is still talking.

    One `OnlineStream` per audio stream; the recognizer object itself is shared.
    """

    name = "nemo-streaming"

    def __init__(self, model_dir: str | Path, num_threads: int = 2,
                 provider: str = "cpu") -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.num_threads = int(num_threads)
        self.provider = provider
        self._rec = None

    def load(self) -> None:
        if self._rec is not None:
            return
        import sherpa_onnx

        model = self.model_dir / "model.onnx"
        tokens = self.model_dir / "tokens.txt"
        if not model.exists() or not tokens.exists():
            raise RuntimeError(f"streaming model files missing in {self.model_dir}")
        t0 = time.monotonic()
        self._rec = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
            model=str(model),
            tokens=str(tokens),
            num_threads=self.num_threads,
            provider=self.provider,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
        )
        LOG.info("nemo streaming loaded in %.1fs", time.monotonic() - t0)

    def create_stream(self):
        self.load()
        return self._rec.create_stream()

    def accept(self, stream, audio_16k: np.ndarray) -> str:
        """Feed audio, decode what is ready, return the running hypothesis."""
        stream.accept_waveform(SAMPLE_RATE, np.asarray(audio_16k, dtype=np.float32))
        while self._rec.is_ready(stream):
            self._rec.decode_stream(stream)
        return (self._rec.get_result(stream) or "").strip()

    def reset(self, stream) -> None:
        self._rec.reset(stream)


def build_recognizer(stt_cfg: dict) -> Recognizer:
    """From the `livekit.stt` block. Unknown backend is an error, not a silent
    fallback — a typo that quietly reinstates the slow engine is exactly the
    kind of drift this review found."""
    backend = (stt_cfg.get("backend") or "parakeet").strip().lower()
    if backend in ("parakeet", "sherpa", "nemo", "sherpa-offline"):
        return SherpaOfflineRecognizer(
            model_dir=stt_cfg.get("model_dir", "agent-services/models/parakeet-tdt-v3"),
            num_threads=int(stt_cfg.get("num_threads", 4)),
            decoding_method=stt_cfg.get("decoding_method", "greedy_search"),
            model_type=stt_cfg.get("model_type", "nemo_transducer"),
            provider=stt_cfg.get("provider", "cpu"),
        )
    if backend == "whisper":
        return WhisperRecognizer(stt_cfg)
    raise RuntimeError(
        f"livekit.stt.backend: unknown backend {backend!r} "
        "(expected 'parakeet' or 'whisper')"
    )


def build_streaming_recognizer(cfg: dict) -> Optional[SherpaStreamingRecognizer]:
    """From `livekit.stt.streaming`, or None when off."""
    sc = (cfg or {}).get("streaming") or {}
    if not sc.get("enabled", False):
        return None
    r = SherpaStreamingRecognizer(
        model_dir=sc.get("model_dir", "agent-services/models/nemo-streaming-480ms"),
        num_threads=int(sc.get("num_threads", 2)),
        provider=sc.get("provider", "cpu"),
    )
    r.load()
    return r
