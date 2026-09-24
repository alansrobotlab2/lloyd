"""`tts_text.for_speech`: what the cloned voice is sent instead of raw text (#1165).

Three properties, each a way the pass could go wrong without anything raising:

  * the hard spoken forms come out as words — `qmd` as "Q M D", an email with
    no `@`, a phone number, an ISO date, a ratio and a `~/…/foo.md` path with
    no slash left in it;
  * it cannot be bypassed — a short plain reply streamed through the voice
    turn reaches the synthesis queue already expanded, because the pass sits
    in `TTSStreamer.speak()`, which every spoken path ends in;
  * a benign sentence comes back byte-identical, so the pass cannot regress
    the round-trip corpus's control arm.

The flag that turns it on in production (`livekit.tts.normalise_text`) stays
off until the corpus has been measured against the live voice.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

import tts_text  # noqa: E402
from tts_text import for_speech  # noqa: E402

CORPUS = ROOT / "eval" / "tts_fidelity_corpus.yaml"
_SLASH_PATH = re.compile(r"(?:~/|/)\w")


def _controls() -> list[str]:
    items = yaml.safe_load(CORPUS.read_text())["items"]
    return [it["text"] for it in items if it["bucket"] == "control"]


# ── the expansions ──────────────────────────────────────────────────────────

def test_qmd_is_spelled_out():
    assert for_speech("The qmd daemon is up.") == "The Q M D daemon is up."
    assert for_speech("QMD answered.") == "Q M D answered."


def test_the_lexicon_expands_tool_names_and_leaves_lookalikes():
    out = for_speech("vLLM and supervisord behind LiveKit and FastAPI; gr00t.")
    assert out == "V L L M and supervisor D behind Live Kit and Fast A P I; Groot."
    # Word boundaries: a lexicon token inside a longer word is not that token.
    assert for_speech("The qmdx tool") == "The qmdx tool"


def test_an_email_is_said_without_an_at_sign():
    out = for_speech("Mail alan.smith@example.com today.")
    assert out == "Mail alan dot smith at example dot com today."
    assert "@" not in out


def test_a_phone_number_is_said_digit_by_digit_in_groups():
    assert for_speech("Call 555-867-5309.") == \
        "Call five five five, eight six seven, five three zero nine."
    assert for_speech("Or (415) 555-0100 instead.") == \
        "Or four one five, five five five, zero one zero zero instead."
    assert re.search(r"\d", for_speech("+1 415 555 0100")) is None


def test_an_iso_date_is_said_as_a_date():
    assert for_speech("It landed on 2026-09-15.") == \
        "It landed on September fifteenth, twenty twenty-six."
    assert for_speech("2005-01-01") == "January first, two thousand five"
    # Not a calendar date: left alone rather than guessed at.
    assert for_speech("build 2026-13-45") == "build 2026-13-45"


def test_a_ratio_and_an_approximate_count_are_said_as_words():
    assert for_speech("It is 3.1x faster.") == "It is three point one times faster."
    assert for_speech("The prompt is ~23.6k tokens.") == \
        "The prompt is about twenty-three point six thousand tokens."
    assert for_speech("~40 left") == "about 40 left"


def test_a_home_path_is_said_without_slashes():
    out = for_speech("I saved it to ~/obsidian/knowledge/foo.md.")
    assert out == "I saved it to home folder, obsidian, knowledge, foo dot md."
    assert _SLASH_PATH.search(out) is None and "~" not in out
    assert _SLASH_PATH.search(for_speech("See /home/alan/lloyd/app/paths.py now")) is None
    # "and/or" is not a path.
    assert for_speech("tea and/or coffee") == "tea and/or coffee"


def test_one_sentence_with_every_hard_form_leaves_no_at_and_no_path():
    out = for_speech("qmd says mail alan@example.com or call 555-867-5309 "
                     "by 2026-09-15; it is 3.1x faster, see ~/obsidian/knowledge/foo.md")
    assert "@" not in out
    assert _SLASH_PATH.search(out) is None
    assert re.search(r"\d", out) is None
    assert out.startswith("Q M D says")


def test_number_words():
    assert tts_text.int_words(0) == "zero"
    assert tts_text.int_words(23600) == "twenty-three thousand six hundred"
    assert tts_text.number_words("0.25") == "zero point two five"
    assert tts_text.ordinal_words(22) == "twenty-second"
    assert tts_text.ordinal_words(30) == "thirtieth"
    assert tts_text.year_words(1999) == "nineteen ninety-nine"
    assert tts_text.year_words(1905) == "nineteen oh five"


# ── the control arm cannot move ─────────────────────────────────────────────

def test_the_corpus_has_at_least_fifteen_plain_control_sentences():
    controls = _controls()
    assert len(controls) >= 15
    for s in controls:
        assert not re.search(r"\d", s), s
        # No proper nouns: nothing capitalised but the first word and "I".
        words = re.findall(r"[A-Za-z']+", s)
        assert all(w[0].islower() or w == "I" for w in words[1:]), s


@pytest.mark.parametrize("sentence", _controls())
def test_a_benign_control_sentence_passes_byte_identical(sentence):
    out = for_speech(sentence)
    assert out == sentence
    assert out.encode("utf-8") == sentence.encode("utf-8")


# ── it cannot be bypassed ───────────────────────────────────────────────────

livekit_worker = pytest.importorskip("livekit_worker")


class _Resp:
    def __init__(self, lines):
        self._lines = lines
        self.headers = {"content-type": "text/event-stream"}
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line
            await asyncio.sleep(0)

    async def aread(self):
        return b"{}"


class _HTTP:
    def __init__(self, resp):
        self.resp = resp

    def stream(self, method, url, json=None, timeout=None):
        return self.resp


def _sse(*events):
    lines = []
    for name, data in events:
        lines += [f"event: {name}", f"data: {json.dumps(data)}", ""]
    return lines


def _trivial_max_chars() -> int:
    """`app.secondary_models._TRIVIAL_MAX_CHARS`, read off the source rather
    than imported: that module pulls the whole backend config in behind it."""
    src = (ROOT / "app" / "secondary_models.py").read_text()
    return int(re.search(r"^_TRIVIAL_MAX_CHARS\s*=\s*(\d+)", src, re.M).group(1))


REPLY = "Sure, the qmd daemon is back. Mail alan@example.com if it drops again."


def _queued_through_voice_turn(tts_cfg: dict) -> list[str]:
    """Stream REPLY through the real voice-turn clause path into a real
    TTSStreamer, and return what reached its synthesis queue."""
    async def go():
        bridge = livekit_worker.RoomBridge(
            "lloyd-20260924_000000_ftst",
            {"room_prefix": "lloyd-", "voice_turn": {"filler": {"enabled": False}}},
            stt=None, vad_cfg={}, http_client=_HTTP(_Resp(_sse(
                ("voice_turn", {"turn_id": "t"}),
                ("text_delta", {"text": REPLY[:30]}),
                ("text_delta", {"text": REPLY[30:]}),
                ("done", {})))))
        tts = livekit_worker.TTSStreamer(tts_cfg, room=None)

        async def published():
            return None
        tts.ensure_published = published
        bridge.tts = tts
        await bridge._speak_voice_turn({"text": "hi", "session_key": "s"})
        queued = []
        while not tts._queue.empty():
            queued.append(tts._queue.get_nowait()[1])
        await tts.close()
        return queued
    return asyncio.run(go())


def test_a_short_plain_reply_reaches_the_synthesis_queue_expanded():
    assert len(REPLY) <= _trivial_max_chars()
    queued = _queued_through_voice_turn({"normalise_text": True})
    assert queued, "nothing reached the synthesis queue"
    spoken = " ".join(queued)
    assert "Q M D" in spoken and "qmd" not in spoken
    assert "alan at example dot com" in spoken
    assert "@" not in spoken


def test_the_pass_is_off_until_measured():
    """`normalise_text` defaults off: with no key, today's text goes out as is."""
    spoken = " ".join(_queued_through_voice_turn({}))
    assert "qmd" in spoken and "alan@example.com" in spoken
