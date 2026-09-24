"""What `architecture/voice.md` may say about the TTS eager-load knob.

`tests/test_qwen3_tts_launcher.py` pins what `start-qwen3-tts.sh` does. This
file pins what the doc is allowed to claim about it, because the doc is where
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
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "voice.md"
LAUNCHER = ROOT / "agent-services" / "bin" / "start-qwen3-tts.sh"


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
