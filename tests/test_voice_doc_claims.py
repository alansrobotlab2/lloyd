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

#3254 is the same correction one layer down: "ask the running service which voices
it can speak" had hardened into one literal command, `curl -s localhost:8090/v1/voices`,
and voice stopped running on :8090 in `d9ff1c0f` (2026-10-09). The intent — ask the
running service — survived; the command did not, because the route differs per engine
(:8093's list route is `/v1/audio/voices` and it 404s `/v1/voices`) and the two servers
keep *separate* voice stores. So the doc's own capability check was probing the fallback
and reading clean while the room spoke through a different server. The nodes at the end
of this file pin the command to `livekit.tts.api_url`, pin every surviving `:8090`
voice-list command to an explicit "fallback" label, and pin the store-separation
sentence — the one that says why the label matters. The literal that kept the stale
command alive was this file's own assertion, which is why these read structure
(fenced block, `## ` section, sentence) rather than one hard-coded command line.
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
    # #3254: this assertion used to read `assert "curl -s localhost:8090/v1/voices" in
    # section`, which is what kept the stale command in the tree — the intent is "ask the
    # running service", and since d9ff1c0f the running service is livekit.tts.api_url.
    assert "curl -s localhost:8093/v1/audio/voices" in section, (
        "the section must still tell a reader to ask the running service — and the "
        "service voice runs on is :8093 (`livekit.tts.api_url`), whose list route is "
        "/v1/audio/voices; :8090's /v1/voices is a different server with a different "
        "voice store (#3254)")
    assert "livekit.tts.api_url" in section, (
        "the engine has to be named by the config key that selects it, not by a port "
        "the reader has to go and look up")
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


# ── #3254: "ask the server which voices it can speak" has to name the engine ──
#
# Voice mode has spoken through `livekit.tts.api_url` — `http://127.0.0.1:8093`,
# vLLM-Omni, `agent-tts-omni` — since `d9ff1c0f` (2026-10-09). Measured on the box
# 2026-10-24: `:8093/v1/voices` is 404 (no such route), `:8093/v1/audio/voices` is
# `{"voices":["clone:dave_cullen","default"],…}` with `.voices` a list of strings, and
# `:8090/v1/voices` is 200 with `.voices[]` as objects. Both servers answer, but omni
# keeps its clones as `.safetensors` under `SPEAKER_SAMPLES_DIR` while the vendored
# server reads `voice_library/profiles/`, so the fallback printing a voice says nothing
# about the room speaking it. These nodes therefore read *structure* — fenced block,
# `## ` section, sentence — rather than one hard-coded command line: a literal command
# in an assertion is exactly how the stale `:8090` one survived being wrong for a year.

ROOM_ENGINE = "localhost:8093"        # whatever livekit.tts.api_url selects; today :8093
FALLBACK_ENGINE = "localhost:8090"    # the vendored agent-tts, kept running as fallback

_VOICES_ROUTE = re.compile(r"/v1/(?:audio/)?voices")
_ENGINE_NAMED = re.compile(r"809\d|vendored|omni|fallback|api_url", re.IGNORECASE)

#: How far a route claim may reach backwards for the engine it means, in characters of
#: the flat comment. Measured, not guessed: the retired claim sits 92 characters from
#: its engine and must trip, the fixed comment's widest claim sits 54 and must pass,
#: and the node below asserts all three numbers.
_ROUTE_ENGINE_WINDOW = 60

#: The run #3254 replaced, verbatim and flat from the pre-round `config.yaml` comment.
#: It names the vendored server earlier in the same run and still leaves the route the
#: reader acts on unqualified — which is the ambiguity this node exists to refuse.
RETIRED_VOICE_COMMENT_CLAIM = (
    "there is no model swap in the vendored server, so a built-in name is refused with "
    "HTTP 400 before synthesis (#2300), and `/v1/voices` on this server lists only "
    "`clone:` profiles.")

#: The same claim as fixed. The control pair pins the window: without this form the
#: node would be refusing any mention of a route, which nobody can satisfy.
LABELLED_VOICE_COMMENT_CLAIM = (
    "the vendored server (:8090, the fallback) answers `/v1/voices` with `clone:` "
    "profiles only")


def _setup_heading_section(heading: str) -> str:
    """The SETUP.md passage from a `## ` heading to the next one, or fail the test.

    Scoped by `## ` rather than by `_setup_section`'s bold headings because Part 12's
    body is one long fenced block, and bold-scoped reading would stop at bold text
    inside the code rather than at the end of the part.
    """
    text = SETUP.read_text(encoding="utf-8")
    i = text.find("\n" + heading + "\n")
    assert i >= 0, f"SETUP.md has no {heading!r} heading"
    start = i + 1
    j = text.find("\n## ", start + len(heading))
    return text[start:j if j > 0 else len(text)]


def _voice_list_label_windows(text: str, span: int = 6) -> list[tuple[int, str]]:
    """Each `:8090/v1/voices` command in a SETUP.md text, with the label window around it.

    The window is the command line, the run of comment lines directly above it and the
    line below — what a reader's eye is on while copying one command. Two rejected
    scopers, both measured: a whole-file grep for "fallback" passes a stale command
    three screens from the word, and pairing ``` markers into fenced blocks is wrong on
    SETUP.md as it stands (an odd number of markers precedes the first stale command, so
    the blocks come out inverted and the command lands in none of them — a node built on
    that reader was vacuous against the very line it was written for).
    """
    lines = text.splitlines()
    out = []
    for n, ln in enumerate(lines):
        if f"{FALLBACK_ENGINE}/v1/voices" not in ln:
            continue
        start = n
        while start > 0 and lines[start - 1].strip().startswith("#") and n - start < span:
            start -= 1
        out.append((n + 1, "\n".join(lines[start:n + 2])))
    return out


def _assert_voice_list_windows_are_labelled(text: str) -> None:
    """Refuse a `:8090/v1/voices` command whose window never says it is the fallback."""
    for line, window in _voice_list_label_windows(text):
        assert "fallback" in window.lower(), (
            f"SETUP.md:{line} documents a {FALLBACK_ENGINE}/v1/voices command with no "
            "engine label beside it: voice speaks through livekit.tts.api_url (:8093, "
            "route /v1/audio/voices) and the two engines keep separate voice stores, so "
            "an unlabelled :8090 check reads clean while the room cannot speak the "
            f"voice (#3254). Window read: {window!r}")


def _assert_voices_routes_are_engine_labelled(comment: str) -> None:
    """Refuse a voices-route claim with no engine named within `_ROUTE_ENGINE_WINDOW`.

    Windowed rather than sentence-scoped, and measured both ways: the pre-#3254 comment
    *does* name the vendored server, two clauses earlier, so a sentence-level rule was
    satisfied by a word the reader never reaches — and a clause-level rule (split on
    ``,``/``:``/``.``) is satisfied by nothing, since the fixed wording qualifies its
    route across a comma. A window is the honest shape of "the engine has to be named
    where the route is named", and its edges are pinned by the two controls in the node.
    """
    flat = _flat(comment)
    for m in _VOICES_ROUTE.finditer(flat):
        window = flat[max(0, m.start() - _ROUTE_ENGINE_WINDOW):m.end() + _ROUTE_ENGINE_WINDOW]
        assert _ENGINE_NAMED.search(window), (
            f"a comment above livekit.tts.voice names a route with no engine within "
            f"{_ROUTE_ENGINE_WINDOW} characters: ...{window.strip()!r}... That key's own "
            "api_url is :8093, and :8093 has no /v1/voices route at all, so an unlabelled "
            "route claim sends a reader to a 404 and lets them read the silence as the "
            "room's capability (#3254)")


def test_no_setup_voice_list_command_can_be_read_as_the_room_s_engine():
    """Clause 3: each documented `:8090/v1/voices` command is labelled the fallback.

    The label has to be beside the command, not merely somewhere in SETUP.md: a reader
    pastes one line, and the line's only context is the comment above it.
    """
    text = SETUP.read_text(encoding="utf-8")
    _assert_voice_list_windows_are_labelled(text)
    windows = _voice_list_label_windows(text)
    assert windows, (
        "no :8090 voice-list command is documented anywhere, so this node has nothing to "
        "check — delete it together with the commands it guards")
    # Control, in the file rather than a manual paste: the same real windows with the
    # label word taken out have to trip it. That proves the reader is looking at the
    # documented commands and that the word "fallback" is what satisfies it.
    joined = "\n".join(w for _, w in windows)
    delabelled = re.sub(r"(?i)fallback", "the other box", joined)
    assert delabelled != joined, "no window carried a fallback label to remove"
    with pytest.raises(AssertionError) as unlabelled:
        _assert_voice_list_windows_are_labelled(delabelled)
    assert "#3254" in str(unlabelled.value), str(unlabelled.value)
    assert f"{FALLBACK_ENGINE}/v1/voices" in str(unlabelled.value)


def test_part_12_verify_asks_the_engine_voice_runs_on_and_slices_nothing():
    """Clause 1: the fresh-install TTS check probes the configured engine's real route."""
    section = _setup_heading_section("## Part 12 — Verify")
    assert f"curl -s {ROOM_ENGINE}/v1/audio/voices | jq -r '.voices[]'" in section, (
        "Part 12 is the check a fresh install runs last; pointed at :8090's /v1/voices it "
        "prints the fallback's store, and omni's list route is /v1/audio/voices (#3254)")
    # Comment lines are exempt: naming the broken form to explain it is the point of the
    # comment. What may not come back is a runnable command that slices the response.
    commands = "\n".join(ln for ln in section.splitlines() if not ln.strip().startswith("#"))
    assert not re.search(r"\.\[\d*:\d*\]", commands), (
        "a range slice is back on a runnable Part 12 command: a voices response is an "
        "object, so `.[0:3]` printed `jq: error (at <stdin>:0): Cannot index object with "
        "object` and exit 5 — a reader cannot tell a dead engine from a jq misuse")
    # The regex above is only worth having if it still matches the line it replaced.
    assert re.search(r"\.\[\d*:\d*\]", "curl -s localhost:8090/v1/voices | jq '.[0:3]'")


def test_setup_md_says_the_two_tts_engines_keep_separate_voice_stores():
    """Clause 4: both stores are named, and the round trip that proves a voice is named."""
    section = _setup_section("**A built-in voice is not served by the default model")
    flat = _flat(section)
    for required in ("separate voice stores", "SPEAKER_SAMPLES_DIR",
                     "~/lloyd-data/tts/speakers", "start-qwen3-tts-omni.sh",
                     "voice_library/profiles/", "/v1/audio/voices",
                     "scripts/voice/register_tts_voice.py"):
        assert required in flat, f"the engine story lost {required!r}"
    assert "round trip" in flat, (
        "the sentence naming register_tts_voice.py has to say what it proves: registering "
        "a voice is only evidence on the engine that will be asked to speak it")
    # The seam the sentence crosses: the prose names a script and a default port, so the
    # script has to exist and still default to the engine the prose claims it proves.
    registrar = (ROOT / "scripts" / "voice" / "register_tts_voice.py")
    assert registrar.is_file(), f"the section names {registrar} as the round trip; it is gone"
    reg = registrar.read_text(encoding="utf-8")
    assert 'default="http://127.0.0.1:8093"' in reg, (
        "the section says register_tts_voice.py proves a voice on the room's engine, so a "
        "default that is no longer :8093 makes that sentence false")
    assert "/v1/audio/voices" in reg, (
        "the registrar no longer posts the omni list route the section says it posts")


def _engine_separation(text: str) -> int:
    """How far the widest route claim in `text` sits from an engine named before it.

    Max over every voices-route mention, so the number is the one that decides whether
    `_assert_voices_routes_are_engine_labelled` fires. `len(text) + 1` when a route has
    no engine named before it at all, so the caller can compare in one direction.
    """
    flat = _flat(text)
    routes = list(_VOICES_ROUTE.finditer(flat))
    assert routes, f"no voices route in {text!r}"
    worst = 0
    for route in routes:
        named = [m.start() for m in _ENGINE_NAMED.finditer(flat[:route.start()])]
        worst = max(worst, len(flat) + 1 if not named else route.start() - max(named))
    return worst


def test_config_yaml_voice_comment_names_the_engine_behind_each_voices_route():
    """Clause 5: every route claim in the loaded comment says which server it means."""
    comment, voice_line = _livekit_voice_comment()
    _assert_voices_routes_are_engine_labelled(comment)
    # Both edges of the window, on text held in this file: the fixed claim has to pass
    # or the rule refuses every mention of a route, and the retired one has to trip or
    # the rule refuses nothing.
    _assert_voices_routes_are_engine_labelled(LABELLED_VOICE_COMMENT_CLAIM)
    with pytest.raises(AssertionError) as ambiguous:
        _assert_voices_routes_are_engine_labelled(RETIRED_VOICE_COMMENT_CLAIM)
    assert "#3254" in str(ambiguous.value), str(ambiguous.value)
    # And the distances the window constant quotes, re-measured here so they cannot rot
    # into a claim nobody checked — including the margin the live comment runs with.
    assert _engine_separation(LABELLED_VOICE_COMMENT_CLAIM) == 19, \
        _engine_separation(LABELLED_VOICE_COMMENT_CLAIM)
    assert _engine_separation(RETIRED_VOICE_COMMENT_CLAIM) == 92, \
        _engine_separation(RETIRED_VOICE_COMMENT_CLAIM)
    assert _engine_separation(comment) == 54 < _ROUTE_ENGINE_WINDOW, \
        _engine_separation(comment)
    # Comment-only by construction: the value line is byte-identical, indentation and
    # all, so the node cannot be satisfied by moving the voice to dodge the comment.
    assert voice_line == "    voice: clone:dave_cullen", repr(voice_line)
