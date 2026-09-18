"""Biasing Parakeet toward the names and terms Lloyd hears.

On 2026-09-18, the first real session after the Parakeet switch, "how many
items are in our Lloyd backlog" came out as "back block". Whisper had a
hotword list (`~/obsidian/hotwords.md`) and Parakeet was never given one.
sherpa-onnx can bias a transducer, but only in beam search and only if it can
turn each hotword into model pieces itself — handed pieces directly it splits
`▁back` at the `▁` and silently skips the word. These pin the parts that make
the list actually reach the decoder.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

from voice import asr  # noqa: E402

MODEL = ROOT / "agent-services" / "models" / "parakeet-tdt-v3"


def test_the_hotwords_file_is_read_past_front_matter_and_markup(tmp_path):
    p = tmp_path / "hotwords.md"
    p.write_text("---\ntitle: Hotwords\ntags:\n- family\n---\nAlan\n\n# a comment\n"
                 "- a bullet\n  Lisa  \ngr00t\n")
    assert asr.read_hotwords(p) == ["Alan", "Lisa", "gr00t"]


def test_a_missing_or_unset_file_is_an_empty_list_not_an_error(tmp_path):
    assert asr.read_hotwords(tmp_path / "absent.md") == []
    assert asr.read_hotwords(None) == []
    assert asr.read_hotwords("") == []


def test_a_lower_case_term_is_also_biased_at_the_start_of_a_sentence():
    # The tokenizer is cased and the model capitalises a first word, so
    # "backlog" alone would never help "Backlog is…".
    assert asr.hotword_variants(["backlog", "Alan", "Mission  Control"]) == [
        "backlog", "Backlog", "Alan", "Mission Control"]
    assert asr.hotword_variants(["", "  ", "Alan", "Alan"]) == ["Alan"]


def test_the_vocab_is_the_tokens_in_merge_order_with_specials_at_zero(tmp_path):
    t = tmp_path / "tokens.txt"
    t.write_text("<unk> 0\n<|nospeech|> 1\n▁the 2\ns 3\n▁back 4\n")
    lines = asr.bpe_vocab_from_tokens(t).splitlines()
    assert lines == ["<unk>\t0", "<|nospeech|>\t0", "▁the\t-2", "s\t-3", "▁back\t-4"]


def test_config_hotwords_reach_parakeet_and_the_switch_turns_them_off(tmp_path):
    p = tmp_path / "hw.md"
    p.write_text("Alan\nLloyd\n")
    cfg = {"backend": "parakeet", "hotwords_file": str(p), "hotwords": ["backlog", "Alan"],
           "hotwords_score": 2.0}
    r = asr.build_recognizer(cfg, exclude_hotwords={"lloyd", "hey"})
    assert isinstance(r, asr.HotwordRecognizer)
    assert r.plain.decoding_method == "greedy_search" and not r.plain.hotwords
    assert r.hotwords == ["Alan", "backlog", "Backlog"]
    assert r.biased.hotwords_score == 2.0
    off = asr.build_recognizer(dict(cfg, parakeet_hotwords=False))
    assert isinstance(off, asr.SherpaOfflineRecognizer) and off.hotwords == []


def test_the_wake_name_is_never_a_hotword(tmp_path):
    # Biased toward "Lloyd", "Floyd came over yesterday" came out as a
    # transcript that opens with the wake word.
    p = tmp_path / "hw.md"
    p.write_text("Lloyd\nAlan\n")
    assert asr.parakeet_hotwords({"hotwords_file": str(p)}, exclude={"lloyd"}) == ["Alan"]


def test_the_biased_decode_is_used_only_when_it_gained_a_hotword():
    hw = ["backlog", "Alan", "Inner Voice"]
    pick = asr.prefer_biased
    assert pick("our Lloyd back block", "our Lloyd backlog", hw) == "our Lloyd backlog"
    assert pick("Ayla's meeting", "Alan's meeting", hw) == "Alan's meeting"
    assert pick("turn on in a voice", "turn on Inner Voice", hw) == "turn on Inner Voice"
    # Anything else beam search does differently is not the list's doing.
    assert pick("", "Mm.", hw) == ""
    assert pick("the back lot", "the back law", hw) == "the back lot"
    # Greedy already had it: greedy's text wins.
    assert pick("the backlog is long", "the backlog's long", hw) == "the backlog is long"
    # A word inside another word is not the word.
    assert not asr.contains_hotword("backlogged", "backlog")


@pytest.mark.skipif(not (MODEL / "tokens.txt").exists(), reason="Parakeet model not fetched")
def test_every_configured_hotword_encodes_against_the_real_vocab(tmp_path, capfd):
    """sherpa reports a hotword it cannot encode on stderr and skips it — the
    recogniser still loads. So the check is that stderr says nothing."""
    import yaml

    stt = yaml.safe_load((ROOT / "config.yaml").read_text())["livekit"]["stt"]
    words = list(stt.get("hotwords") or []) + ["Alan", "Lisa", "Emilio", "Alfie",
                                               "Stompy", "Gracie", "gr00t"]
    r = asr.SherpaOfflineRecognizer(MODEL, hotwords=words, num_threads=1)
    r.load()
    err = capfd.readouterr().err
    assert "failed to encode" not in err and "Cannot find ID" not in err, err


@pytest.mark.skipif(not (MODEL / "tokens.txt").exists(), reason="Parakeet model not fetched")
def test_a_listed_term_greedy_cannot_hear_comes_through():
    """LiveKit was 0 of 6 for greedy and 6 of 6 biased across the held-out
    voices. The fixture is one of those clips (Qwen3-TTS, voice Ryan)."""
    import wave

    import numpy as np

    with wave.open(str(ROOT / "tests" / "fixtures" / "voice" / "livekit_server_16k.wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    r = asr.build_recognizer({"backend": "parakeet", "model_dir": str(MODEL),
                              "hotwords": ["LiveKit"], "hotwords_score": 1.0})
    plain = r.plain.transcribe(audio).text
    got = r.transcribe(audio)
    assert not asr.contains_hotword(plain, "LiveKit"), plain
    assert asr.contains_hotword(got.text, "LiveKit"), got.text
    assert got.backend == "parakeet+hw"
