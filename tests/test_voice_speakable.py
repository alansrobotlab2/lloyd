"""Streamed reply text -> clauses the TTS can say.

Every test feeds the text in awkward slices, because that is how it arrives: a
model's deltas split words, markdown markers and code fences at arbitrary
points, and a segmenter that only works on whole lines would pass every test
written against whole strings and fail on the first real reply.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-services"))

speakable = pytest.importorskip("voice.speakable")
ClauseStream = speakable.ClauseStream


def _run(text, step=3, **kw):
    cs = ClauseStream(**kw)
    out = []
    for i in range(0, len(text), step):
        out += cs.feed(text[i:i + step])
    out += cs.flush()
    return out, cs


@pytest.mark.parametrize("step", [1, 3, 7, 50])
def test_sentences_come_out_whole_whatever_the_slicing(step):
    out, _ = _run("It is seven thirty. Your meeting starts at eight, "
                  "and the traffic is light. Anything else?", step=step)
    assert " ".join(out) == ("It is seven thirty. Your meeting starts at eight, "
                             "and the traffic is light. Anything else?")
    assert out[0] == "It is seven thirty."


def test_the_first_clause_is_released_before_the_sentence_ends():
    """The listener is waiting on the first sound, so a long opening sentence
    is cut at its first comma rather than held for its full stop."""
    cs = ClauseStream(first_soft_chars=40)
    got = cs.feed("I checked the calendar for the whole of next week, "
                  "and there is")
    assert got == ["I checked the calendar for the whole of next week,"]


def test_a_decimal_and_an_abbreviation_do_not_end_a_sentence():
    # Soft split off: this is about where a SENTENCE ends, and the first-clause
    # comma cut would otherwise fire at "today," first.
    out, _ = _run("The value is 3.5 today, e.g. higher than yesterday. Done now.",
                  first_soft_chars=10_000)
    assert out[0] == "The value is 3.5 today, e.g. higher than yesterday."


def test_a_code_block_is_skipped_whole_and_noted():
    text = ("Here is the fix.\n```python\ndef f():\n    return 1\n```\n"
            "That should work now.")
    out, cs = _run(text, step=2)
    assert out == ["Here is the fix.", "That should work now."]
    assert cs.skipped_code
    assert not any("def" in c or "return" in c for c in out)


def test_everything_after_a_divider_is_for_the_chat_only():
    text = "The build is green. The log is in the chat.\n---\nhttps://x/y\n| a | b |\n"
    out, cs = _run(text, step=4)
    assert out == ["The build is green.", "The log is in the chat."]
    assert cs.stopped


def test_a_bullet_list_is_spoken_as_sentences_without_its_markers():
    text = "Three things:\n- the backend is up\n- the engine is warm\n1. and the queue is empty\n"
    out, _ = _run(text, step=5)
    joined = " ".join(out)
    assert "-" not in joined and "1." not in joined
    assert "the backend is up" in joined and "the queue is empty" in joined


def test_a_table_is_dropped():
    out, _ = _run("Totals below.\n| host | ram |\n|---|---|\n| a | 1 |\nThat is all.\n", step=3)
    assert out == ["Totals below.", "That is all."]


def test_inline_markdown_is_unwrapped_and_urls_are_not_read_out():
    out, _ = _run("See **the** [docs](https://example.com/x) at https://a.b/c "
                  "and `config.yaml` for *details*.")
    assert out == ["See the docs at and config.yaml for details."]


def test_snake_case_survives_the_italic_rule():
    out, _ = _run("Set max_inflight to two.")
    assert out == ["Set max_inflight to two."]


def test_a_short_clause_is_joined_to_the_next_rather_than_spoken_alone():
    out, _ = _run("Right. Yes. Ok. The engine restarted at nine and is healthy.",
                  min_chars=24)
    assert out[0] == "Right."            # the first is never held
    assert "Yes. Ok. The engine" in out[1]


def test_an_endless_sentence_is_cut_at_the_ceiling():
    out, _ = _run("word " * 200, max_chars=120)
    assert all(len(c) <= 125 for c in out)
    assert len(out) > 3


def test_nothing_is_lost_at_flush():
    cs = ClauseStream()
    cs.feed("And one more thing without a full stop")
    assert "without a full stop" in " ".join(cs.flush())
