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

Whisper stays as `backend: whisper`, the fallback. Parakeet gained the hotword
list on 2026-09-18 (`HotwordRecognizer`), after the first real session heard
"our Lloyd backlog" as "back block" — through a second, biased decode that is
consulted only for the words on the list.
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


def read_hotwords(path: str | Path | None) -> list[str]:
    """One name per line from a markdown hotwords file (optional YAML front
    matter; blank, `#` and `-` lines skipped). Empty on any error — a missing
    list costs the biasing, never the recogniser.

    File format (matching the user's ~/obsidian/hotwords.md):
      ---
      title: Hotwords
      ---
      Lloyd
      Alan
    """
    if not path:
        return []
    p = Path(path).expanduser()
    try:
        text = p.read_text()
    except FileNotFoundError:
        return []
    except Exception as e:
        LOG.warning("hotwords: failed to read %s: %s", p, e)
        return []
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end >= 0:
            text = text[end + 4:]
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        names.append(line)
    return names


def hotword_variants(words) -> list[str]:
    """Each word as written, plus a capitalised copy of a lower-case one.

    Parakeet's tokenizer is cased — "backlog" and "Backlog" are different
    pieces — and the model capitalises a sentence's first word, so a
    lower-case term with no capital form is never biased at the start of a
    sentence. Names are left alone: nobody says "alan"."""
    out: list[str] = []
    for w in words:
        w = " ".join(str(w).split())
        if not w:
            continue
        for v in (w, w[0].upper() + w[1:]):
            if v not in out:
                out.append(v)
    return out


def _words(text: str) -> list[str]:
    # "Stompy's" is Stompy. Digits kept, so "gr00t" can match itself.
    return [w[:-2] if w.endswith("'s") else w
            for w in re.findall(r"[a-z0-9']+", (text or "").lower())]


def contains_hotword(text: str, hotword: str) -> bool:
    """Whole-word, case-insensitive; a phrase must appear contiguously."""
    t, h = _words(text), _words(hotword)
    return bool(h) and any(t[i:i + len(h)] == h for i in range(len(t) - len(h) + 1))


def prefer_biased(plain: str, biased: str, hotwords) -> str:
    """The biased transcript only when it gained a hotword the plain one
    lacks; the plain one otherwise.

    Beam search — the only decoder sherpa biases — is measurably worse than
    greedy on this model before any list is applied, so the list's decode is
    used only where the list could have been the point of it."""
    gained = any(contains_hotword(biased, h) and not contains_hotword(plain, h)
                 for h in hotwords)
    return biased if gained else plain


def bpe_vocab_from_tokens(tokens_txt: str | Path) -> str:
    """A sentencepiece-style `piece<TAB>score` vocab rebuilt from sherpa's
    `tokens.txt`, which is all the Parakeet export ships.

    sherpa-onnx needs it to turn a hotword into model pieces itself — handed
    pieces directly, it splits `▁back` at the `▁` and refuses the rest. A BPE
    vocab lists pieces in merge order, so `-id` as the score reproduces the
    merge priority; specials (`<unk>`, `<pad>`, `<|…|>`) score 0."""
    lines = []
    for raw in Path(tokens_txt).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        piece, idx = raw.rsplit(" ", 1) if " " in raw.strip() else (raw, "0")
        score = 0 if piece.startswith("<") else -int(idx)
        lines.append(f"{piece}\t{score}\n")
    return "".join(lines)


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
        hotwords=(),
        hotwords_score: float = 1.5,
        max_active_paths: int = 4,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.num_threads = int(num_threads)
        self.decoding_method = decoding_method
        self.model_type = model_type
        self.provider = provider
        self.hotwords = hotword_variants(hotwords or ())
        self.hotwords_score = float(hotwords_score)
        self.max_active_paths = int(max_active_paths)
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
        kw = dict(
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
        if not self.hotwords:
            self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw)
        else:
            # Biasing only exists in beam search. sherpa reads both files
            # during construction, so they live exactly that long.
            import tempfile

            with tempfile.TemporaryDirectory(prefix="lloyd-hotwords-") as d:
                vocab = Path(d) / "bpe.vocab"
                vocab.write_text(bpe_vocab_from_tokens(kw["tokens"]), encoding="utf-8")
                hw = Path(d) / "hotwords.txt"
                hw.write_text("\n".join(self.hotwords) + "\n", encoding="utf-8")
                kw.update(
                    decoding_method="modified_beam_search",
                    max_active_paths=self.max_active_paths,
                    modeling_unit="bpe",
                    bpe_vocab=str(vocab),
                    hotwords_file=str(hw),
                    hotwords_score=self.hotwords_score,
                )
                self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw)
        LOG.info("parakeet loaded in %.1fs from %s (%s)", time.monotonic() - t0, self.model_dir,
                 f"{len(self.hotwords)} hotwords @ {self.hotwords_score:g}" if self.hotwords
                 else kw["decoding_method"])

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


class HotwordRecognizer:
    """Greedy Parakeet, with a hotword-biased beam decode consulted beside it.

    sherpa biases only in beam search, and on this model beam search alone is
    worse than greedy — LibriSpeech 4.78% against 3.83%, and it changes 188 of
    500 real-room transcripts, mostly silence turned into "Mm.". So both run
    (the biased one on its own thread) and `prefer_biased` keeps greedy's text
    unless the biased decode gained a hotword. 2026-09-18, the configured 10
    at score 1.0 (`scripts/voice/hotword_eval.py`): hotword recall 87 → 106 of
    132 on held-out voices, 0 false insertions in 72 sound-alikes, LibriSpeech
    3.83% / 9.30% at 10 dB unchanged, and 1 of 500 real transcripts changed —
    "our Lloyd back block" → "our Lloyd backlog". ~+100 ms and one more copy
    of the model (~1 GB) are the price.
    """

    name = "parakeet"

    def __init__(self, plain: SherpaOfflineRecognizer, biased: SherpaOfflineRecognizer) -> None:
        self.plain = plain
        self.biased = biased
        self.hotwords = biased.hotwords
        self._pool = None

    def load(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.plain.load()
        self.biased.load()
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr-hotwords")

    def transcribe(self, audio_16k: np.ndarray) -> Transcript:
        self.load()
        t0 = time.monotonic()
        fut = self._pool.submit(self.biased.transcribe, audio_16k)
        plain = self.plain.transcribe(audio_16k)
        try:
            biased = fut.result()
        except Exception as e:
            # The bias is an improvement, never a dependency.
            LOG.warning("hotword decode failed, using greedy: %s", e)
            return plain
        text = prefer_biased(plain.text, biased.text, self.hotwords)
        backend = "parakeet+hw" if text != plain.text else self.name
        return Transcript(text, time.monotonic() - t0, backend)


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


def parakeet_hotwords(stt_cfg: dict, exclude=()) -> list[str]:
    """The list Parakeet is biased toward: the hotwords file plus
    `stt.hotwords`, minus `exclude` (case-insensitive), deduplicated.

    The worker excludes the wake names. Biased toward "Lloyd", the decoder
    heard "Floyd came over yesterday" as "Lloyd came over yesterday" — a
    transcript that opens with the wake word, which the transcript wake path
    would have answered."""
    if not stt_cfg.get("parakeet_hotwords", True):
        return []
    drop = {str(w).lower() for w in exclude}
    out: list[str] = []
    for w in read_hotwords(stt_cfg.get("hotwords_file")) + list(stt_cfg.get("hotwords") or []):
        w = " ".join(str(w).split())
        if w and w.lower() not in drop and w not in out:
            out.append(w)
    return out


def build_recognizer(stt_cfg: dict, exclude_hotwords=()) -> Recognizer:
    """From the `livekit.stt` block. Unknown backend is an error, not a silent
    fallback — a typo that quietly reinstates the slow engine is exactly the
    kind of drift this review found."""
    backend = (stt_cfg.get("backend") or "parakeet").strip().lower()
    if backend in ("parakeet", "sherpa", "nemo", "sherpa-offline"):
        common = dict(
            model_dir=stt_cfg.get("model_dir", "agent-services/models/parakeet-tdt-v3"),
            num_threads=int(stt_cfg.get("num_threads", 4)),
            model_type=stt_cfg.get("model_type", "nemo_transducer"),
            provider=stt_cfg.get("provider", "cpu"),
        )
        plain = SherpaOfflineRecognizer(
            decoding_method=stt_cfg.get("decoding_method", "greedy_search"), **common)
        hotwords = parakeet_hotwords(stt_cfg, exclude_hotwords)
        if not hotwords:
            return plain
        return HotwordRecognizer(plain, SherpaOfflineRecognizer(
            hotwords=hotwords, hotwords_score=float(stt_cfg.get("hotwords_score", 1.0)),
            **common))
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
