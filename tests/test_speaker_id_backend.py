"""The speaker encoder behind `livekit.voiceprint.backend` (voice plan Phase 4).

CAM++ replaced Resemblyzer on 2026-09-24 (eval/speaker_embed_eval.py; numbers
in architecture/voice.md "Who is speaking"). What these pin:

  * the backend is chosen by config and an unknown one is refused;
  * profiles are tagged `<name>.<backend>.npy`, so a backend switch never
    compares a 512-d CAM++ vector with a 256-d Resemblyzer one — another
    backend's profile, and a legacy untagged Resemblyzer `.npy`, are ignored
    rather than crashing or matching;
  * `enroll_reference` averages several clips into one unit-norm profile;
  * the numpy Kaldi fbank equals torchaudio's (a stored reference);
  * the published CAM++ export's `count_include_pad=1` is patched as it loads —
    unpatched, every embedding collapses once an utterance crosses a 2 s
    segment boundary (the four 2.6 s fixture clips all score ~0.9 against each
    other) — and on the real model the same speaker outscores a different one.

The model-backed tests skip when `agent-services/models/campplus/` has not been
fetched (`bash agent-services/setup/fetch-voice-models.sh`).

Fixture clips: LibriSpeech test-clean speakers 6930 and 8230, 2.6 s crops
(CC BY 4.0, Panayotov et al., openslr.org/12).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

import speaker_id as sid  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "voice" / "speaker"
MODEL = sid.DEFAULT_MODEL_PATHS["campplus"]
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason=f"CAM++ model not fetched ({MODEL})")


def _clip(name: str) -> np.ndarray:
    sf = pytest.importorskip("soundfile")
    a, sr = sf.read(str(FIX / name), dtype="int16")
    assert sr == 16000
    return a


class _FakeEncoder:
    """Deterministic stand-in: the embedding is a function of the audio's
    first sample, so tests about the profile store never load a model."""

    def __init__(self, dim):
        self.dim = dim

    def embed(self, wav):
        rng = np.random.default_rng(int(abs(wav[0]) * 1e4) % 2**32)
        return sid._unit(rng.standard_normal(self.dim))


def _ident(tmp_path, backend="campplus", **kw):
    s = sid.SpeakerIdentifier(tmp_path, backend=backend, **kw)
    s._encoder = _FakeEncoder(s.embedding_dim)
    return s


# ── backend selection ───────────────────────────────────────────────────

def test_default_backend_is_campplus_and_config_says_so():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    vp = cfg["livekit"]["voiceprint"]
    assert sid.DEFAULT_BACKEND == "campplus"
    assert vp["backend"] == "campplus"
    # The thresholds are per model: Resemblyzer's 0.75/0.4 on CAM++'s scale
    # would name almost nobody and let almost anybody continue.
    assert vp["profile_threshold"] < 0.6
    assert vp["anchor_threshold"] < vp["profile_threshold"]


def test_backend_selection_and_dims(tmp_path):
    assert sid.SpeakerIdentifier(tmp_path, backend="campplus").embedding_dim == 512
    assert sid.SpeakerIdentifier(tmp_path, backend="Resemblyzer").embedding_dim == 256
    with pytest.raises(ValueError):
        sid.SpeakerIdentifier(tmp_path, backend="ecapa")


def test_a_missing_model_fails_at_load_not_at_construction(tmp_path):
    s = sid.SpeakerIdentifier(tmp_path, backend="campplus", model_path=tmp_path / "absent.onnx")
    with pytest.raises(FileNotFoundError):
        s._ensure_encoder()
    # identify() degrades to "no embedding", the worker's identity-only path.
    name, score, emb = s.identify(np.zeros(16000, dtype=np.int16), 16000)
    assert (name, score, emb) == ("Unknown", 0.0, None)


# ── profile tagging ─────────────────────────────────────────────────────

def test_enroll_writes_a_backend_tagged_profile(tmp_path):
    s = _ident(tmp_path)
    path = s.enroll("alan", np.full(16000, 1000, dtype=np.int16), 16000)
    assert Path(path).name == "alan.campplus.npy"
    assert np.load(path).shape == (512,)
    assert s.list_profiles() == [{"name": "alan", "embedding_dim": 512,
                                  "backend": "campplus", "path": path}]


def test_legacy_and_other_backend_profiles_are_ignored(tmp_path):
    np.save(tmp_path / "alan.npy", sid._unit(np.ones(256)))              # legacy Resemblyzer
    np.save(tmp_path / "kid.resemblyzer.npy", sid._unit(np.ones(256)))   # tagged Resemblyzer
    np.save(tmp_path / "bad.campplus.npy", sid._unit(np.ones(256)))      # mis-sized
    s = _ident(tmp_path)
    assert not s.has_profiles
    name, score, emb = s.identify(np.full(16000, 500, dtype=np.int16), 16000)
    assert name == "Unknown" and emb.shape == (512,)


def test_the_legacy_file_still_loads_under_resemblyzer(tmp_path):
    np.save(tmp_path / "alan.npy", sid._unit(np.ones(256)))
    np.save(tmp_path / "kid.campplus.npy", sid._unit(np.ones(512)))
    s = _ident(tmp_path, backend="resemblyzer")
    assert [p["name"] for p in s.list_profiles()] == ["alan"]
    # A tagged file beats the untagged one of the same name.
    np.save(tmp_path / "alan.resemblyzer.npy", -sid._unit(np.ones(256)))
    s.reload()
    assert s.list_profiles()[0]["path"].endswith("alan.resemblyzer.npy")


def test_delete_removes_every_backends_copy(tmp_path):
    for f in ("alan.npy", "alan.resemblyzer.npy", "alan.campplus.npy"):
        np.save(tmp_path / f, np.zeros(4))
    s = _ident(tmp_path)
    assert s.delete_profile("alan")
    assert not list(tmp_path.glob("alan*"))
    assert not s.delete_profile("alan")
    assert not s.delete_profile("../etc")


def test_enroll_reference_averages_clips_into_one_unit_profile(tmp_path):
    s = _ident(tmp_path)
    clips = [np.full(16000, v, dtype=np.int16) for v in (100, 200, 300)]
    embs = s.embed_many(clips, 16000)
    assert embs.shape == (3, 512)
    path = s.enroll_reference("lloyd-voice", clips, 16000)
    prof = np.load(path)
    assert Path(path).name == "lloyd-voice.campplus.npy"
    np.testing.assert_allclose(np.linalg.norm(prof), 1.0, rtol=1e-5)
    np.testing.assert_allclose(prof, sid._unit(embs.mean(0)), rtol=1e-5)
    # (audio, rate) pairs are accepted too, and resampled.
    assert s.embed_many([(np.full(48000, 100, dtype=np.int16), 48000)]).shape == (1, 512)
    with pytest.raises(ValueError):
        s.enroll_reference("bad name!", clips)


def test_identify_names_the_best_profile_above_threshold(tmp_path):
    s = _ident(tmp_path, threshold=0.5)
    a = np.full(16000, 1000, dtype=np.int16)
    s.enroll("alan", a, 16000)
    assert s.identify(a, 16000)[0] == "alan"
    assert s.identify(np.full(16000, 2000, dtype=np.int16), 16000)[0] == "Unknown"


# ── front end ───────────────────────────────────────────────────────────

def test_kaldi_fbank_matches_torchaudio():
    """Reference: torchaudio.compliance.kaldi.fbank(num_mel_bins=80,
    dither=0) on the same clip, first 40 frames (torchaudio 2.11)."""
    ref = np.load(FIX / "ls_6930_0_fbank_head.npy")
    wav = _clip("ls_6930_0.flac").astype(np.float32) / 32768.0
    mine = sid.kaldi_fbank(wav)
    assert mine.shape == (258, 80)
    assert np.abs(mine[:40] - ref).max() < 2e-3


def test_the_avgpool_patch_is_a_same_length_byte_swap():
    blob = b"xx" + sid._CIP_ON + b"yy" + sid._CIP_ON
    out, n = sid._patch_count_include_pad(blob)
    assert n == 2 and len(out) == len(blob) and sid._CIP_ON not in out
    assert sid._patch_count_include_pad(b"nothing")[1] == 0


# ── the real model ──────────────────────────────────────────────────────

@needs_model
def test_campplus_same_speaker_outscores_different_speaker(tmp_path):
    s = sid.SpeakerIdentifier(tmp_path, backend="campplus")
    assert s._ensure_encoder().patched_pools == 52
    names = ["ls_6930_0.flac", "ls_6930_1.flac", "ls_8230_0.flac", "ls_8230_1.flac"]
    E = s.embed_many([_clip(n) for n in names], 16000)
    assert E.shape == (4, 512)
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, rtol=1e-5)
    C = E @ E.T
    same = min(C[0, 1], C[2, 3])
    diff = max(C[0, 2], C[0, 3], C[1, 2], C[1, 3])
    # Measured 0.77 / 0.07 patched; unpatched all four are ~0.85-0.96.
    assert same > 0.6 and diff < 0.3, C.round(2)


@needs_model
def test_campplus_does_not_drift_across_a_segment_boundary(tmp_path):
    """The unpatched export's failure: 30 ms more of the same speech moved the
    cosine to the shorter clip from 1.00 to ~0.6 at every 2 s boundary."""
    s = sid.SpeakerIdentifier(tmp_path, backend="campplus")
    a = _clip("ls_6930_0.flac")
    base = s.extract_embedding(a[:32000], 16000)          # 198 frames
    longer = s.extract_embedding(a[:32000 + 480], 16000)  # past the boundary
    assert float(base @ longer) > 0.9


@needs_model
def test_campplus_identify_end_to_end(tmp_path):
    s = sid.SpeakerIdentifier(tmp_path, backend="campplus", threshold=0.40)
    s.enroll("sixnine", _clip("ls_6930_0.flac"), 16000)
    s.enroll("eighttwo", _clip("ls_8230_0.flac"), 16000)
    assert s.identify(_clip("ls_6930_1.flac"), 16000)[0] == "sixnine"
    assert s.identify(_clip("ls_8230_1.flac"), 16000)[0] == "eighttwo"
    # A reloaded identifier reads what enroll wrote.
    s2 = sid.SpeakerIdentifier(tmp_path, backend="campplus", threshold=0.40)
    assert sorted(p["name"] for p in s2.list_profiles()) == ["eighttwo", "sixnine"]
    assert not sid.SpeakerIdentifier(tmp_path, backend="resemblyzer").has_profiles
