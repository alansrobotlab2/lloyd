"""observer_prompt._format_subliminal_context — unit tests.

The observer gets the same prefetched block the primary saw, capped. The
cap must not throw away the small sections (facts, vault, ide) that sit
after a long skill body.

Run:
  .venvs/lloyd/bin/python -m pytest tests/unit/test_observer_prompt_subliminal.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))

from app.inner_voice import observer_prompt as op  # noqa: E402


def _block(skill_chars: int) -> str:
    skill_body = "FIRST-LINE purpose of the skill\n" + ("step step step\n" * (skill_chars // 15)) + "LAST-LINE done when X"
    return (
        "<context>\n"
        f'<skill name="big-skill" score="9.0">\n{skill_body}\n</skill>\n'
        "<backlog-refs>\n- [Task #300] \"Track vllm\" — open\n</backlog-refs>\n"
        "<facts>\n- [QMD] lex is AND-only (confidence: 1.0)\n</facts>\n"
        "<vault-context>\n- **2026-04-14 Daily Notes** (score: 1.00): servo notes\n</vault-context>\n"
        "<ide_state>\n  visible_file: prefetch.py\n</ide_state>\n"
        "</context>"
    )


def test_small_sections_survive_a_long_skill_body():
    out = op._format_subliminal_context(_block(6000))
    for marker in ("<backlog-refs>", "<facts>", "<vault-context>", "<ide_state>", "visible_file: prefetch.py"):
        assert marker in out, marker
    # skill purpose and closing criteria both kept, middle trimmed
    assert "FIRST-LINE purpose" in out and "LAST-LINE done when X" in out
    assert "chars trimmed" in out
    assert len(out) <= op._SUBLIMINAL_PROMPT_CHAR_CAP + 200  # header + marker overhead


def test_short_block_is_untouched():
    small = _block(200)
    out = op._format_subliminal_context(small)
    assert small in out
    assert "trimmed" not in out


def test_empty_is_empty():
    assert op._format_subliminal_context("") == ""
    assert op._format_subliminal_context("   \n") == ""
    assert op._format_subliminal_context(None) == ""
