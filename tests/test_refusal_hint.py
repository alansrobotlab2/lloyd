"""The refusal paragraph in the system prompt (`prompt_builder._refusal_hint`).

Pinned: it is present on a worker turn and on a chat turn, it is inserted as
one paragraph and moves nothing else, and it says the two things a detour
needs to hear — do not retry, do not respell — plus that a named remedy is the
way forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import prompt_builder as PB  # noqa: E402


def test_the_paragraph_is_in_every_platforms_prompt():
    for platform in ("", "worker", "autonomy"):
        text = PB.build_system_prompt(include_skills_index=False, memories_text="",
                                      platform=platform)
        assert PB._refusal_hint() in text, platform


def test_the_paragraph_says_what_a_detour_needs_to_hear():
    hint = PB._refusal_hint()
    assert hint.startswith("Refused tool calls:")
    assert "Tool call denied" in hint, "it names the exact prefix the aggregator and hook use"
    assert "Do not retry" in hint
    assert "another tool or another spelling" in hint
    assert "find -delete" in hint and "rm -rf" in hint
    assert "a grant to mint" in hint
    assert len(hint) < 900, "a paragraph, not a policy document"


def test_it_is_one_paragraph_and_nothing_else_moves():
    text = PB.build_system_prompt(include_skills_index=False, memories_text="")
    without = text.replace("\n\n" + PB._refusal_hint(), "", 1)
    assert PB._refusal_hint() not in without
    assert without.count("\n\n") == text.count("\n\n") - 1
