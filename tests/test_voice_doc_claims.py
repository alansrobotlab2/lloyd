"""What the tracked prose may claim about the TTS server.

`tests/test_qwen3_tts_launcher.py` pins what `start-qwen3-tts.sh` does. This
file pins what the docs are allowed to claim about it, because the docs are where
#1446 came from: the 2026-09-24 architecture review wrote "The knob is not
exported by the launch script, so hand-running `start-qwen3-tts.sh` is the lazy
path and reproduces that incident" into the TTS-server section and filed the
item from that same sentence (commit `391f03f4`, whose review log records
"the eager-load knob lives only in `agent-tts.conf`"). A stale negative in an
architecture doc re-files its own item — the next reader sees a gap and files
it, and a gap list is a to-do list — so once the launcher defaults the knob the
negative has to go, and the two assertions here are the two directions of that:
the stale claim must be gone, and its replacement must still carry the
measurements that justify the default, so this file cannot be satisfied by
deleting the paragraph.

#2300 extends the same rule to a second claim that was live in three files at
once: that a built-in voice "still works" and "just triggers the swap on the
request that asks for one" (`architecture/voice.md`), that falling back to one is
a thing to do by checking `/v1/voices` (`SETUP.md`), and that `Ryan` "still works
as a fallback" (`config.yaml`). No swap exists —
`_customvoice_model_key` appears nowhere in the vendored `optimized_backend.py`,
and both generation paths load `self._base_model_key()` — so the default
`1.7B-Base` served no built-in speaker at all and answered one with a 500 (or, on
`stream:true`, a 200 with zero bytes). Those three sentences are the nodes below,
each with the same positive control as #1446's: the measurements and the
load-bearing ordering rules have to survive the correction.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "voice.md"
LAUNCHER = ROOT / "agent-services" / "bin" / "start-qwen3-tts.sh"
SETUP = ROOT / "SETUP.md"
CONFIG = ROOT / "config.yaml"


def _para(marker: str) -> str:
    """The paragraph of the doc that opens with `marker`, or fail the test.

    Scoped to one paragraph on purpose: the whole-file assertions below pass a
    sentence that has been moved into a footnote as happily as one that is still
    the doc's live claim about how TTS boots.
    """
    text = DOC.read_text()
    start = text.find(marker)
    assert start >= 0, f"architecture/voice.md no longer has the {marker!r} paragraph"
    end = text.find("\n\n", start)
    return text[start:end if end > 0 else len(text)]



# ── clause 4: the stale negative is gone ─────────────────────────────


def test_the_doc_no_longer_says_the_knob_is_unexported():
    """The sentence that filed #1446. It is false the moment the launcher
    defaults the knob, and it is the shape of claim that reads as an open gap
    rather than as history."""
    text = DOC.read_text()
    assert "not exported by the launch script" not in text, \
        "voice.md still claims the launcher does not set TTS_LAZY_LOAD"
    assert not re.search(r"hand-running[^.]*lazy", text), (
        "voice.md still describes a hand-run as the lazy path: "
        f"{re.findall(r'.{0,60}hand-running[^.]*lazy.{0,40}', text)}")


def test_a_superseded_claim_in_the_review_log_says_so():
    """The review log is history, so the entry that filed this item keeps its
    wording — but only annotated. An unannotated "lives only in `agent-tts.conf`"
    is the sentence a future arch review quotes back as an open gap. The note has
    to sit next to the claim it retires: a correction three sections away is a
    claim and a counter-claim, and the reader who finds only the first one files
    the item again."""
    lines = DOC.read_text().splitlines()
    hits = [i for i, ln in enumerate(lines) if "lives only in `agent-tts.conf`" in ln]
    assert hits, "positive control: the filed-item entry this test exists to check is gone"
    for i in hits:
        window = "\n".join(lines[max(0, i - 6):i + 7])
        assert "#1446 landed" in window, \
            f"voice.md:{i + 1} states the pre-fix location with no note beside it"


# ── clause 4: the replacement states the real arrangement ───────────


def test_the_doc_says_the_launcher_defaults_it_and_the_conf_overrides():
    para = _para("**Eager loading")
    # The heading sentence is the claim itself, not a passing mention.
    assert re.search(r"Eager loading is the launcher.s default", para), \
        "the section does not attribute the default to the launcher"
    assert "start-qwen3-tts.sh" in para, "the launcher is not named"
    assert "${TTS_LAZY_LOAD:-false}" in para, \
        "the doc does not state the default, nor the `:-` form that keeps it a default"
    assert "agent-tts.conf" in para and "environment=" in para, \
        "the supervisor conf's override route is not named"
    assert "override" in para, "nothing calls the conf's value an override"


def test_the_replacement_still_carries_the_measurement_that_justifies_it():
    """Positive control. Without it the two tests above are satisfiable by
    gutting the paragraph, which is how the 2026-09-19 numbers got lost once
    already: the section predated them."""
    para = _para("**Eager loading")
    assert "2026-09-19" in para, "the incident is no longer dated"
    assert "4 min 6 s" in para, "the measured cold first synthesis is no longer stated"
    assert "read=120.0" in para, "the serial read timeout is no longer stated"
    assert re.search(r"discarded rather than delayed", para), \
        "the doc no longer says a cold first utterance is lost, not late"


def test_the_doc_describes_the_launcher_that_actually_exists():
    """The seam between the prose and the script. The expansion is read out of
    the doc rather than written down again here, so the two files cannot agree on
    a stale value: the literal is pinned once, in
    `tests/test_qwen3_tts_launcher.py::test_the_default_is_a_default_and_not_a_hard_export`,
    in the file a change to it would have to touch anyway."""
    quoted = set(re.findall(r"\$\{TTS_LAZY_LOAD:-[^}]+\}", _para("**Eager loading")))
    assert quoted, "the section no longer quotes the launcher's expansion verbatim"
    launcher = LAUNCHER.read_text()
    for token in quoted:
        assert token in launcher, f"voice.md quotes {token}, which the launcher lacks"


# ── #2300: no tracked prose may promise a built-in voice the default model cannot serve ──
# `_para` above is the existing paragraph scoper; these nodes reuse it rather than
# adding a second reader of the same file.

def _setup_section(marker: str) -> str:
    """The SETUP.md passage from the bold heading `marker` to the next bold heading.

    Scoped to a section rather than a paragraph because the fallback advice is a
    heading, prose and two code fences, and the stale claim spans all three.
    """
    text = SETUP.read_text(encoding="utf-8")
    i = text.find(marker)
    assert i >= 0, f"marker not found in SETUP.md: {marker!r}"
    j = text.find("\n**", i + len(marker))
    return text[i:j if j > 0 else len(text)]


def _livekit_voice_comment() -> tuple[str, str]:
    """The comment block above `livekit.tts.voice` and the voice line itself.

    Returned separately so a node can prove the *value* is untouched while only its
    comment changed — the item rules the model/voice choice out of scope, so a
    rewrite that also moved `voice:` would be a scope breach this assertion catches.
    """
    lines = CONFIG.read_text(encoding="utf-8").splitlines()
    at = next(i for i, ln in enumerate(lines)
              if ln.strip().startswith("voice: clone:dave_cullen"))
    start = at
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    return "\n".join(lines[start:at]), lines[at]


def test_voice_md_no_longer_promises_a_swap_for_a_builtin_voice():
    """The claim that the default model serves built-ins, corrected without losing its numbers."""
    para = _para("**Two models, and which one is default is a latency decision.**")
    assert "Built-in voices still work" not in para, (
        "voice.md still says a built-in voice works; the vendored server carries no "
        "model-swap machinery, so that promise is what sent a reader to `Ryan`")
    assert "just trigger the swap" not in para
    for required in ("no swap", "_customvoice_model_key", "_base_model_key()",
                     "400", "/v1/voices", "#2300"):
        assert required in para, f"correction lost {required!r}: {para}"
    # Positive control: the correction has to keep the facts that justify the default,
    # so it cannot be satisfied by amputing the paragraph.
    for measurement in ("~20 s measured", "1.7B-Base", "_base_model_key()"):
        assert measurement in para, f"{measurement!r} lost from the paragraph"
    assert "must stay the first" in para, (
        "the load-bearing ordering rule left with the false claim")


def test_setup_md_no_longer_offers_a_builtin_voice_as_a_fallback():
    """SETUP.md's built-in-fallback section now states what a Base model refuses."""
    section = _setup_section("**A built-in voice is not served by the default model")
    for stale in ("Male built-ins are", "a bad voice still returns HTTP 200",
                  "Falling back to a built-in voice"):
        assert stale not in section, f"SETUP.md still says {stale!r}"
    for required in ("HTTP 400", "stream:true", "zero bytes", "clone:dave_cullen",
                     "CustomVoice"):
        assert required in section, f"corrected section lost {required!r}"
    assert "curl -s localhost:8090/v1/voices" in section, (
        "the section must still tell a reader to ask the running service")
    assert "~20 s" in section, (
        "the cost of switching to CustomVoice belongs beside the instruction to do it")


def test_config_yaml_voice_comment_no_longer_promises_a_fallback():
    """The voice-mode comment states the Base refusal, and the voice value is untouched."""
    comment, voice_line = _livekit_voice_comment()
    for stale in ("still works as a fallback", "swaps models on the next request"):
        assert stale not in comment, f"config.yaml comment still says {stale!r}"
    for required in ("no model swap", "400", "CustomVoice", "#2300"):
        assert required in comment, f"corrected comment lost {required!r}: {comment}"
    # The item rules the voice/model choice out of scope: comment-only, by construction.
    assert voice_line.strip() == "voice: clone:dave_cullen", voice_line
