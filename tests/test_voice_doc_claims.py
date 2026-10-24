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

A fourth site of the same claim — `architecture/voice.md`'s own "The cloned
voice" section — was corrected on 2026-10-24 by `2e9fe880` and had no node at
all until the section-scoped ones at the end of this file. It is not a
`_para` job: the retired advice was a sentence *and* the fenced `curl` under it,
and `_para` stops at the blank line above that fence, so a paragraph-scoped node
passes on the very regression it names. Its negative control is the struck
wording quoted verbatim as a module constant, so the checks are proved against
the real regression rather than against a summary of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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


# ── #2300's fourth site: voice.md's own "### The cloned voice" section ──
#
# The nodes above never named this section: before these were added, a search of
# this file for "escape hatch", "cloned voice" and "invalid_voice" hit nothing,
# while the corrected prose sat in the file the whole time. `_para` is the wrong
# scoper here rather than a shorter one — its first `\n\n` is the blank line
# above the fenced `curl`, and the half of the retired advice that actually sends
# a reader to a refused speaker was the request body inside that fence.

CLONED_VOICE_HEADING = "### The cloned voice"
CLONED_VOICE_MARKER = "A built-in voice is **not** an escape hatch"

#: What the section may never say again, matched on the flat view (`_flat`).
_STALE_CLONED_VOICE_CLAIMS = (
    "Falling back to a built-in voice",
    "real escape hatch",
    "check `/v1/voices`, not the source",
)

#: The paragraph and fence `2e9fe880` struck, verbatim out of
#: `git show 2e9fe880^:architecture/voice.md`. This is the negative control: the
#: checks below take a string, so the constant is what proves they can fail.
RETIRED_CLONED_VOICE_SECTION = r"""Falling back to a built-in voice is a real escape hatch, but **check
`/v1/voices`, not the source** — and verify with a round trip, since a bad voice
name still returns HTTP 200:

```bash
curl -s localhost:8090/v1/voices | jq -r '.voices[].id'
curl -s localhost:8090/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"qwen3-tts","voice":"Ryan","input":"test"}' \
  -o /tmp/v.wav -w '%{http_code} %{size_download}\n'
```
"""

_QUOTED_200_CLAIM = "still returns HTTP 200"


def _flat(text: str) -> str:
    """The text with every run of whitespace collapsed to one space.

    The doc wraps prose at ~80 columns, so on disk the retired instruction is split
    between its bold `check` and the quoted route it names, and the retired sentence
    is split between `a bad voice name` and `still returns HTTP 200`. A raw substring
    test misses both — and would keep missing them if somebody only re-wrapped the
    stale claim. The flat view is the claim; the line breaks are the markup.
    """
    return re.sub(r"\s+", " ", text)


def _cloned_voice_section_in(text: str) -> str:
    """The `### The cloned voice` section of a voice.md text, heading to next heading.

    A section, not a paragraph: the scope has to reach the fenced `curl` block at
    the end of it. Both failure messages name the heading, because a re-titled
    section and a reworded marker are the same accident — the correction is still
    on the page and nothing points at it any more.
    """
    i = text.find("\n" + CLONED_VOICE_HEADING + "\n")
    assert i >= 0, (
        f"architecture/voice.md has no {CLONED_VOICE_HEADING!r} heading, so nothing scopes "
        f"the #2300 built-in-voice correction any more ({CLONED_VOICE_MARKER!r} has to be "
        "guarded by a node, not just left in the file)")
    start = i + 1
    j = text.find("\n### ", start + len(CLONED_VOICE_HEADING))
    section = text[start:j if j > 0 else len(text)]
    assert CLONED_VOICE_MARKER in section, (
        f"architecture/voice.md {CLONED_VOICE_HEADING} no longer states "
        f"{CLONED_VOICE_MARKER!r} — the sentence that replaced the built-in escape hatch "
        "has left the section it belongs to")
    return section


def _cloned_voice_section() -> str:
    return _cloned_voice_section_in(DOC.read_text(encoding="utf-8"))


def _assert_no_retired_phrasing(section: str) -> None:
    """Refuse the three sentences `2e9fe880` struck, wherever in the section they sit."""
    flat = _flat(section)
    for stale in _STALE_CLONED_VOICE_CLAIMS:
        assert stale not in flat, (
            f"the {CLONED_VOICE_HEADING} section promises a built-in voice again: {stale!r}")


def _assert_no_builtin_voice_asked(section: str) -> None:
    """Refuse a request body that asks for a speaker the served model will not speak.

    Matched structurally rather than as the literal `"voice":"Ryan"` because what a
    reader copies out of this section is the curl: a built-in name in a fence is an
    instruction to ask the served Base model for a speaker it refuses with HTTP 400
    `invalid_voice` (#2300), whatever the prose above it now says. The whole section
    is scanned, so the fence below the marker paragraph is inside the check — which
    is the whole reason this node is not a `_para` node.
    """
    asked = [v for v in re.findall(r'"voice"\s*:\s*"([^"]*)"', _flat(section))
             if not v.startswith("clone:")]
    assert not asked, (
        f'the {CLONED_VOICE_HEADING} section asks a non-clone voice again: {asked} — the '
        'served model is Base and refuses a built-in name with HTTP 400 invalid_voice '
        '(#2300), which is the `"voice":"Ryan"` request this node exists to keep out')


def _assert_no_builtin_escape_hatch(section: str) -> None:
    """Both halves of the struck advice: the promise in the prose and the name in the curl."""
    _assert_no_retired_phrasing(section)
    _assert_no_builtin_voice_asked(section)


def _assert_the_base_refusal_facts_survive(section: str) -> None:
    """Positive control: the correction's own evidence has to stay in the section.

    Without it the guard is satisfiable by deleting the paragraph, which is how the
    #1446 measurements got lost once already.
    """
    flat = _flat(section)
    for required in ("HTTP 400", "invalid_voice", "404", "/v1/audio/voices", "#2300"):
        assert required in flat, (
            f"the {CLONED_VOICE_HEADING} section lost {required!r}: what a built-in name "
            "is answered with has to stay beside the instruction not to ask for one")
    assert re.search(r"bare HTTP 200[^.]*genuine evidence", flat), (
        "the section no longer says a bare HTTP 200 is genuine evidence — that is the "
        "consequence of the 400/404 pair, and the reason the retired advice is retired")


def _assert_200_claim_is_only_a_quotation(section: str) -> None:
    """Allow `still returns HTTP 200` in the section only inside a double quote.

    The landed correction keeps the phrase on purpose, quoted, as the claim it is
    retiring, so `assert PHRASE not in section` would be red against the fixed
    prose. What may not come back is the unquoted sentence the retired paragraph
    ran. Both directions are judged on the flat view so that a line break after
    the opening quote is not read as an unquoted claim, and the sentence check is
    run on the raw text with `\\s+` so the wrapped form is refused too.
    """
    flat = _flat(section)
    for m in re.finditer(re.escape(_QUOTED_200_CLAIM), flat):
        assert m.start() > 0 and flat[m.start() - 1] == '"', (
            f'{_QUOTED_200_CLAIM!r} is stated as a live claim again, not a quotation: '
            f'...{flat[max(0, m.start() - 70):m.end() + 10]}...')
    assert not re.search(r"a bad voice name\s+still returns HTTP 200", section), (
        "the retired sentence 'a bad voice name still returns HTTP 200' is back: the "
        "server answers a bad built-in name with HTTP 400 invalid_voice and a missing "
        "clone profile with 404 (#2300), so a bare 200 is evidence now")


def test_voice_md_cloned_voice_section_no_longer_offers_a_builtin_escape_hatch():
    """The fourth #2300 site: voice.md's cloned-voice section, its fence included."""
    section = _cloned_voice_section()
    _assert_no_builtin_escape_hatch(section)
    _assert_the_base_refusal_facts_survive(section)
    _assert_200_claim_is_only_a_quotation(section)


def test_the_cloned_voice_scoper_fails_loudly_when_the_section_moves():
    """The scoper's own two failure modes, both naming the heading.

    Built by rewriting the live doc in memory: a renamed heading and a reworded
    marker are how the correction stops being guarded while still reading well.
    """
    text = DOC.read_text(encoding="utf-8")
    with pytest.raises(AssertionError) as heading:
        _cloned_voice_section_in(text.replace(CLONED_VOICE_HEADING, "### Voice assets"))
    assert CLONED_VOICE_HEADING in str(heading.value), str(heading.value)
    with pytest.raises(AssertionError) as marker:
        _cloned_voice_section_in(text.replace(CLONED_VOICE_MARKER, "Built-ins are not served"))
    assert CLONED_VOICE_HEADING in str(marker.value), str(marker.value)


def test_the_retired_escape_hatch_section_still_trips_every_check():
    """Negative control, pinned in this file: the refused wording is still refusably there.

    `RETIRED_CLONED_VOICE_SECTION` is the struck paragraph and fence verbatim, so
    the checks are proved against the regression rather than a paraphrase of it —
    and the positive control on the constant is what stops the control being
    softened into something harmless, which is how a control stops being one.
    """
    flat = _flat(RETIRED_CLONED_VOICE_SECTION)
    for stale in _STALE_CLONED_VOICE_CLAIMS:
        assert stale in flat, f"the retired constant no longer holds {stale!r}"
    assert '"voice":"Ryan"' in flat, "the retired constant lost the built-in curl it is for"
    assert "invalid_voice" not in flat, "the constant has drifted into the corrected wording"

    with pytest.raises(AssertionError) as stale_claim:
        _assert_no_builtin_escape_hatch(RETIRED_CLONED_VOICE_SECTION)
    assert "Falling back to a built-in voice" in str(stale_claim.value)
    with pytest.raises(AssertionError) as lost_facts:
        _assert_the_base_refusal_facts_survive(RETIRED_CLONED_VOICE_SECTION)
    assert "HTTP 400" in str(lost_facts.value)
    with pytest.raises(AssertionError) as unquoted:
        _assert_200_claim_is_only_a_quotation(RETIRED_CLONED_VOICE_SECTION)
    assert _QUOTED_200_CLAIM in str(unquoted.value)

    # The scope control, for the one thing `_para` could never see: keep today's
    # corrected section and swap in only the retired request body. A node scoped to
    # the marker paragraph passes that; this fails it.
    fence = RETIRED_CLONED_VOICE_SECTION[RETIRED_CLONED_VOICE_SECTION.index("```bash"):]
    with pytest.raises(AssertionError) as fenced:
        _assert_no_builtin_voice_asked(_cloned_voice_section() + "\n" + fence)
    assert "Ryan" in str(fenced.value), str(fenced.value)
