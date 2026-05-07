"""Speaker identification + anchor matching for the LiveKit worker.

Ported from the legacy `voice_pipeline.py:SpeakerIdentifier`. Uses
Resemblyzer (256-d d-vector embeddings, unit-norm so cosine == dot).

Two roles:
  1. **Enrolled-profile recognition** — `*.npy` files in `profiles_dir`.
     `identify(audio)` returns the best-matching profile name + cosine score,
     or `(unknown_label, best_score)` if the score is below `threshold`.
  2. **Anchor matching for wake-word continuation** — caller stashes the
     wake-word utterance's embedding via `extract_embedding()` and compares
     follow-up utterances against it. The legacy heuristic accepted a
     follow-up if EITHER it matched the same enrolled profile OR cosine
     vs the anchor cleared a (typically lower) threshold — so unenrolled
     speakers can still hold a continuation as long as the anchor stays
     consistent.

Embeddings are computed at 16 kHz mono float32. Resemblyzer's
`preprocess_wav` resamples + trims silence; pass the utterance's actual
sample rate as `source_sr`.

The encoder is loaded lazily on first use — keeping a 5–10 MB
torch.load() out of import time so worker startup stays snappy.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np


LOG = logging.getLogger("lloyd-agent-worker.speaker_id")


class SpeakerIdentifier:
    def __init__(
        self,
        profiles_dir: str | Path,
        threshold: float = 0.75,
        unknown_label: str = "Unknown",
        device: str = "cpu",
    ) -> None:
        self.profiles_dir = Path(profiles_dir).expanduser()
        self.threshold = float(threshold)
        self.unknown_label = unknown_label
        self.device = device
        self._encoder = None  # lazy
        self._profiles: dict[str, np.ndarray] = {}
        self._load_profiles()

    # ── Lazy encoder ────────────────────────────────────────────────────
    def _ensure_encoder(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder(device=self.device)
            LOG.info("voice encoder loaded on %s", self.device)
        return self._encoder

    # ── Profile store ───────────────────────────────────────────────────
    def _load_profiles(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.profiles_dir.glob("*.npy")):
            try:
                self._profiles[path.stem] = np.load(path)
            except Exception as e:
                LOG.warning("failed to load profile %s: %s", path, e)
        if self._profiles:
            LOG.info("loaded %d voice profile(s): %s",
                     len(self._profiles), sorted(self._profiles))

    def reload(self) -> None:
        """Re-scan profiles_dir. Called after enrollment/delete via API."""
        self._profiles = {}
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
                "path": str(self.profiles_dir / f"{name}.npy"),
            })
        return out

    # ── Embedding ───────────────────────────────────────────────────────
    def extract_embedding(self, audio_int16: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed an int16 PCM utterance → unit-norm 256-d vector."""
        from resemblyzer import preprocess_wav
        wav = audio_int16.astype(np.float32) / 32768.0
        # preprocess_wav resamples to 16 kHz internally and trims silence.
        wav = preprocess_wav(wav, source_sr=int(sample_rate))
        return self._ensure_encoder().embed_utterance(wav)

    # ── Identification ──────────────────────────────────────────────────
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
        best_name = self.unknown_label
        best_score = -1.0
        for name, profile in self._profiles.items():
            score = float(np.dot(emb, profile))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score < self.threshold:
            return self.unknown_label, best_score, emb
        return best_name, best_score, emb

    # ── Enrollment ──────────────────────────────────────────────────────
    def enroll(self, name: str, audio_int16: np.ndarray, sample_rate: int) -> str:
        """Save an embedding under `name`. Returns the profile path."""
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"invalid profile name: {name!r} (alphanumeric/-/_ only)")
        emb = self.extract_embedding(audio_int16, sample_rate)
        out_path = self.profiles_dir / f"{name}.npy"
        np.save(out_path, emb)
        self._profiles[name] = emb
        LOG.info("enrolled profile %r → %s", name, out_path)
        return str(out_path)

    def delete_profile(self, name: str) -> bool:
        path = self.profiles_dir / f"{name}.npy"
        if path.exists():
            path.unlink()
            self._profiles.pop(name, None)
            LOG.info("deleted profile %r", name)
            return True
        return False
