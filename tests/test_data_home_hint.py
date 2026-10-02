"""The runtime-data paragraph in the system prompt (`prompt_builder._data_home_hint`).

The platform paragraph named the checkout (`Home:`) and nothing else, and on 2026-10-02
a nightly task went looking for its run history there: `sqlite3 workers.db` from the
tree created an empty database that alerted hourly. Pinned: every platform's prompt
names the data root, the store and table that turn was looking for, and the rule —
built from `app.paths`, so a relocated root moves the sentence with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import prompt_builder as PB  # noqa: E402
from app.paths import DATA_ROOT, LLOYD_HOME  # noqa: E402


def test_the_paragraph_is_in_every_platforms_prompt():
    for platform in ("", "worker", "autonomy"):
        text = PB.build_system_prompt(include_skills_index=False, memories_text="",
                                      platform=platform)
        assert PB._data_home_hint() in text, platform


def test_it_names_the_root_the_store_and_the_rule():
    hint = PB._data_home_hint()
    assert hint.startswith("Runtime data:")
    assert str(DATA_ROOT) in hint and str(LLOYD_HOME) in hint
    assert "workers.db" in hint and "`runs`" in hint
    assert "code only" in hint
    assert "creates an empty database" in hint
    assert len(hint) < 800, "a paragraph, not a policy document"


def test_it_sits_directly_after_the_line_that_names_the_checkout():
    text = PB.build_system_prompt(include_skills_index=False, memories_text="")
    home = text.index("Platform: Lloyd (local harness).")
    assert home < text.index(PB._data_home_hint())
    between = text[home:text.index(PB._data_home_hint())]
    assert between.count("\n\n") == 1, "one paragraph break, nothing in between"
