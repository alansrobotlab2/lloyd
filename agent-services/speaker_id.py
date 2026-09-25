"""Speaker identification + anchor matching for the LiveKit worker.

Two embedding backends, chosen by ``livekit.voiceprint.backend``:

  * ``campplus`` (default since 2026-09-24) — 3D-Speaker's CAM++ trained on
    VoxCeleb (``3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx``, 512-d,
    Apache-2.0), run through onnxruntime with the Kaldi fbank front end below.
  * ``resemblyzer`` — the GE2E d-vector this module started with (256-d).

Measured 2026-09-24 (``eval/speaker_embed_eval.py``; numbers in
architecture/voice.md "Who is speaking"): on 1-4 s LibriSpeech crops CAM++
halves Resemblyzer's equal-error rate (4.7% vs 9.7% clip-to-clip), and it is
the only one of the two whose own-voice margin — Lloyd's TTS clone against the
room corpus, +0.31 vs +0.07 — is wide enough to gate barge-in on.

**The published CAM++ export is patched as it loads.** CAM++'s context-aware
masking pools 100-frame segments with ``avg_pool1d(ceil_mode=True)``; PyTorch
averages a trailing partial segment over the frames it has, but the ONNX export
(the file sherpa-onnx publishes) sets ``count_include_pad=1`` on all 52
``AveragePool`` nodes, so onnxruntime divides that segment by 100. The
embedding then collapses every time an utterance crosses a 2 s boundary: a
3.0 s prefix of a 4 s LibriSpeech clip scored 0.14 against the whole clip, and
over 1-4 s crops the equal-error rate was 38% — worse than Resemblyzer, and
exactly as bad through sherpa-onnx's own extractor. ``_patch_count_include_pad``
flips those attributes to 0 in the model bytes (same length, so a plain byte
swap) before onnxruntime sees them; the file on disk stays the published one.
``kaldi_fbank`` replaces sherpa-onnx's front end and is checked against
``torchaudio.compliance.kaldi.fbank`` (max |diff| 3e-4 on log-mel).

Two roles:
  1. **Enrolled-profile recognition** — ``<name>.<backend>.npy`` files in
     ``profiles_dir``. ``identify(audio)`` returns the best-matching profile
     name + cosine score, or ``(unknown_label, best_score)`` below ``threshold``.
  2. **Anchor matching for wake-word continuation** — the caller stashes the
     wake-word utterance's embedding via ``extract_embedding()`` and compares
     follow-ups against it (``anchor_threshold`` lives with the caller).

Profiles are tagged by backend because cosines between two models' vectors are
meaningless (and the dims differ: 512 vs 256). A profile of another backend is
ignored, never compared; an untagged ``<name>.npy`` is a legacy Resemblyzer
profile and is read only under the ``resemblyzer`` backend.

All embeddings are unit-norm, so cosine == dot. Input is int16 PCM at any rate.
The encoder is loaded lazily (``_ensure_encoder``) — the worker calls it at
startup so the first wake does not pay for it.
"""
from __future__ import annotations

import logging
from math import gcd
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


LOG = logging.getLogger("lloyd-agent-worker.speaker_id")

SR = 16000
BACKENDS = ("campplus", "resemblyzer")
DEFAULT_BACKEND = "campplus"
_MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATHS = {
    "campplus": _MODELS_DIR / "campplus" / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
}
# Below this CAM++ has too few frames to pool (and the worker never embeds
# anything this short — utterances are >= ~0.5 s).
MIN_SAMPLES = SR // 4


# ── Kaldi fbank (numpy) ─────────────────────────────────────────────────

_FBANK_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _fbank_tables(num_mel_bins: int, n_fft: int, frame_len: int,
                  sr: int, low_freq: float) -> tuple[np.ndarray, np.ndarray]:
    key = (num_mel_bins, n_fft, frame_len, sr, low_freq)
    hit = _FBANK_CACHE.get(key)
    if hit is not None:
        return hit
    n = np.arange(frame_len, dtype=np.float64)
    window = np.power(0.5 - 0.5 * np.cos(2 * np.pi * n / (frame_len - 1)), 0.85)  # povey
    half = n_fft // 2
    mel_lo, mel_hi = _mel(low_freq), _mel(sr / 2.0)
    delta = (mel_hi - mel_lo) / (num_mel_bins + 1)
    m = np.arange(num_mel_bins, dtype=np.float64)[:, None]
    left, center, right = mel_lo + m * delta, mel_lo + (m + 1) * delta, mel_lo + (m + 2) * delta
    fft_mel = _mel(np.arange(half) * sr / n_fft)[None, :]
    up = (fft_mel - left) / (center - left)
    down = (right - fft_mel) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    banks = np.pad(banks, ((0, 0), (0, 1)))  # Kaldi leaves the Nyquist bin out
    tables = (window, banks.T.copy())
    _FBANK_CACHE[key] = tables
    return tables


def kaldi_fbank(wav: np.ndarray, sample_rate: int = SR, num_mel_bins: int = 80) -> np.ndarray:
    """Kaldi ``compute-fbank-feats`` defaults (25 ms / 10 ms, povey window,
    pre-emphasis 0.97, DC removal, snip_edges, low_freq 20, no dither), as
    ``torchaudio.compliance.kaldi.fbank`` computes them. ``(frames, bins)``."""
    frame_len, shift = int(sample_rate * 0.025), int(sample_rate * 0.010)
    n_fft = 1 << (frame_len - 1).bit_length()
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size < frame_len:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    n_frames = 1 + (x.size - frame_len) // shift
    idx = np.arange(frame_len)[None, :] + shift * np.arange(n_frames)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - 0.97 * prev
    window, banks = _fbank_tables(num_mel_bins, n_fft, frame_len, sample_rate, 20.0)
    power = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)) ** 2
    mel = power @ banks
    return np.log(np.maximum(mel, np.finfo(np.float32).eps)).astype(np.float32)


# ── Audio prep ──────────────────────────────────────────────────────────

def _to_float16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """int16 (or float32 in [-1, 1]) at any rate → float32 mono at 16 kHz."""
    a = np.asarray(audio)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    else:
        a = a.astype(np.float32)
    sample_rate = int(sample_rate)
    if sample_rate != SR:
        from scipy.signal import resample_poly
        g = gcd(sample_rate, SR)
        a = resample_poly(a, SR // g, sample_rate // g).astype(np.float32)
    return a


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("degenerate embedding")
    return v / n


# ── Backends ────────────────────────────────────────────────────────────

# AttributeProto{name="count_include_pad", i=1, type=INT}: field 1 (name),
# field 3 varint (i), field 20 varint (type). Only the `i` byte changes.
_CIP_ON = b"\x0a\x11count_include_pad\x18\x01\xa0\x01\x02"
_CIP_OFF = b"\x0a\x11count_include_pad\x18\x00\xa0\x01\x02"


def _patch_count_include_pad(model: bytes) -> tuple[bytes, int]:
    """Set every ``count_include_pad=1`` to 0 (see module docstring). With
    ``pads == 0`` that is PyTorch's ``ceil_mode`` semantics exactly."""
    n = model.count(_CIP_ON)
    return (model.replace(_CIP_ON, _CIP_OFF) if n else model), n


class _CampplusEncoder:
    name = "campplus"
    dim = 512

    def __init__(self, model_path: str | Path, num_threads: int = 2) -> None:
        import onnxruntime as ort
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"CAM++ model not found at {path} — bash agent-services/setup/fetch-voice-models.sh")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(num_threads))
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        model, self.patched_pools = _patch_count_include_pad(path.read_bytes())
        self._sess = ort.InferenceSession(model, sess_options=so,
                                          providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        meta = self._sess.get_modelmeta().custom_metadata_map or {}
        # 3D-Speaker exports want [-1, 1] samples, WeSpeaker ones int16-scaled.
        # After mean normalisation only the log floor can tell them apart, but
        # honour the model's own declaration rather than guess.
        self._scale = 1.0 if str(meta.get("normalize_samples", "1")) == "1" else 32768.0

    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        if wav16k.size < MIN_SAMPLES:
            raise ValueError(f"utterance too short to embed ({wav16k.size / SR:.2f} s)")
        feats = kaldi_fbank(wav16k * self._scale)
        feats = feats - feats.mean(axis=0, keepdims=True)
        out = self._sess.run(None, {self._input: feats[None]})[0]
        return _unit(out[0])


class _ResemblyzerEncoder:
    name = "resemblyzer"
    dim = 256

    def __init__(self, device: str = "cpu") -> None:
        from resemblyzer import VoiceEncoder
        self._enc = VoiceEncoder(device=device)

    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        from resemblyzer import preprocess_wav
        # preprocess_wav trims silence (webrtcvad) before the d-vector.
        return _unit(self._enc.embed_utterance(preprocess_wav(wav16k, source_sr=SR)))


_ENCODER_DIMS = {"campplus": _CampplusEncoder.dim, "resemblyzer": _ResemblyzerEncoder.dim}


def _parse_profile_stem(stem: str) -> tuple[str, str]:
    """``alan.campplus`` → (alan, campplus); ``alan`` → (alan, resemblyzer),
    the legacy untagged format, which only Resemblyzer ever wrote."""
    name, dot, tag = stem.rpartition(".")
    if dot and tag in BACKENDS and name:
        return name, tag
    return stem, "resemblyzer"


def _valid_name(name: str) -> bool:
    return bool(name) and name.replace("-", "").replace("_", "").isalnum()


class SpeakerIdentifier:
    def __init__(
        self,
        profiles_dir: str | Path,
        threshold: float = 0.5,
        unknown_label: str = "Unknown",
        device: str = "cpu",
        backend: str = DEFAULT_BACKEND,
        model_path: str | Path | None = None,
        num_threads: int = 2,
    ) -> None:
        backend = (backend or DEFAULT_BACKEND).strip().lower()
        if backend not in BACKENDS:
            raise ValueError(f"unknown voiceprint backend {backend!r} (one of {BACKENDS})")
        self.backend = backend
        self.profiles_dir = Path(profiles_dir).expanduser()
        self.threshold = float(threshold)
        self.unknown_label = unknown_label
        self.device = device
        self.model_path = (Path(model_path).expanduser() if model_path
                           else DEFAULT_MODEL_PATHS.get(backend))
        self.num_threads = int(num_threads)
        self._encoder = None  # lazy
        self._profiles: dict[str, np.ndarray] = {}
        self._paths: dict[str, Path] = {}
        self._load_profiles()

    # ── Lazy encoder ────────────────────────────────────────────────────
    def _ensure_encoder(self):
        if self._encoder is None:
            if self.backend == "campplus":
                self._encoder = _CampplusEncoder(self.model_path, self.num_threads)
            else:
                self._encoder = _ResemblyzerEncoder(self.device)
            LOG.info("voice encoder loaded: %s (%d-d)", self.backend, self._encoder.dim)
        return self._encoder

    @property
    def embedding_dim(self) -> int:
        return _ENCODER_DIMS[self.backend]

    # ── Profile store ───────────────────────────────────────────────────
    def _profile_path(self, name: str) -> Path:
        return self.profiles_dir / f"{name}.{self.backend}.npy"

    def _load_profiles(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        ignored: list[str] = []
        # Tagged files sort after a legacy `<name>.npy` of the same name only
        # by accident of spelling, so resolve precedence explicitly.
        for path in sorted(self.profiles_dir.glob("*.npy"), key=lambda p: (p.stem.count("."), p.name)):
            name, tag = _parse_profile_stem(path.stem)
            if tag != self.backend:
                ignored.append(path.name)
                continue
            try:
                emb = np.load(path).astype(np.float32).reshape(-1)
            except Exception as e:
                LOG.warning("failed to load profile %s: %s", path, e)
                continue
            if emb.shape[0] != self.embedding_dim:
                LOG.warning("profile %s is %d-d, backend %s is %d-d — ignored",
                            path.name, emb.shape[0], self.backend, self.embedding_dim)
                continue
            # Later (tagged) files overwrite an untagged legacy one.
            self._profiles[name] = emb
            self._paths[name] = path
        if self._profiles:
            LOG.info("loaded %d %s voice profile(s): %s",
                     len(self._profiles), self.backend, sorted(self._profiles))
        if ignored:
            LOG.info("ignored %d voice profile(s) of another backend: %s",
                     len(ignored), ignored)

    def reload(self) -> None:
        """Re-scan profiles_dir. Called after enrollment/delete via API."""
        self._profiles = {}
        self._paths = {}
        self._load_profiles()

    @property
    def has_profiles(self) -> bool:
        return bool(self._profiles)

    def list_profiles(self) -> list[dict]:
        out = []
        for name, emb in sorted(self._profiles.items()):
            out.append({
                "name": name,
                "embedding_dim": int(emb.shape[0]),
                "backend": self.backend,
                "path": str(self._paths.get(name, self._profile_path(name))),
            })
        return out

    # ── Embedding ───────────────────────────────────────────────────────
    def extract_embedding(self, audio_int16: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed an int16 PCM utterance → unit-norm vector (512-d CAM++,
        256-d Resemblyzer)."""
        wav = _to_float16k(audio_int16, sample_rate)
        return self._ensure_encoder().embed(wav)

    def embed_many(self, clips: Iterable, sample_rate: int = SR) -> np.ndarray:
        """Embed several clips → ``(n, dim)``. Each clip is an int16 array at
        ``sample_rate`` or an ``(audio, sample_rate)`` pair."""
        out = []
        for clip in clips:
            if isinstance(clip, tuple):
                audio, sr = clip
            else:
                audio, sr = clip, sample_rate
            out.append(self.extract_embedding(audio, sr))
        if not out:
            raise ValueError("no clips to embed")
        return np.stack(out)

    # ── Identification ──────────────────────────────────────────────────
    def score_profiles(self, emb: np.ndarray) -> dict[str, float]:
        """Cosine of ``emb`` against every loaded profile of this backend."""
        return {name: float(np.dot(emb, p)) for name, p in self._profiles.items()}

    def identify(self, audio_int16: np.ndarray, sample_rate: int) -> tuple[str, float, Optional[np.ndarray]]:
        """Returns (name, score, embedding). `name` is `unknown_label` when
        no profile clears `threshold` OR when no profiles are enrolled.
        Embedding is always returned (None only on encoder failure)."""
        try:
            emb = self.extract_embedding(audio_int16, sample_rate)
        except Exception as e:
            LOG.warning("embedding extraction failed: %s", e)
            return self.unknown_label, 0.0, None
        if not self._profiles:
            return self.unknown_label, 0.0, emb
        best_name, best_score = max(self.score_profiles(emb).items(), key=lambda kv: kv[1])
        if best_score < self.threshold:
            return self.unknown_label, best_score, emb
        return best_name, best_score, emb

    # ── Enrollment ──────────────────────────────────────────────────────
    def _save(self, name: str, emb: np.ndarray) -> str:
        out_path = self._profile_path(name)
        np.save(out_path, emb.astype(np.float32))
        self._profiles[name] = emb
        self._paths[name] = out_path
        LOG.info("enrolled %s profile %r → %s", self.backend, name, out_path)
        return str(out_path)

    def enroll(self, name: str, audio_int16: np.ndarray, sample_rate: int) -> str:
        """Save one clip's embedding under `name`. Returns the profile path."""
        return self.enroll_reference(name, [audio_int16], sample_rate)

    def enroll_reference(self, name: str, clips: Sequence, sample_rate: int = SR) -> str:
        """Save the renormalised mean of several clips' embeddings under
        `name` — Phase 2 enrols Lloyd's own TTS voice as ``lloyd-voice`` from a
        handful of renders this way. Clips as in ``embed_many``."""
        if not _valid_name(name):
            raise ValueError(f"invalid profile name: {name!r} (alphanumeric/-/_ only)")
        embs = self.embed_many(clips, sample_rate)
        return self._save(name, _unit(embs.mean(axis=0)))

    def delete_profile(self, name: str) -> bool:
        """Remove `name` under every backend (and a legacy untagged file), so
        switching backends never resurrects a profile somebody deleted."""
        if not _valid_name(name):
            return False
        removed = False
        for path in [self.profiles_dir / f"{name}.npy"] + [
                self.profiles_dir / f"{name}.{b}.npy" for b in BACKENDS]:
            if path.exists():
                path.unlink()
                removed = True
        if removed:
            self._profiles.pop(name, None)
            self._paths.pop(name, None)
            LOG.info("deleted profile %r", name)
        return removed
