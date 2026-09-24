"""The skill-library count `eval/lloyd_profile.md` puts in front of the intake judge.

The youtube-digest session reads this profile before deciding whether a video
holds an idea worth filing, and nothing at read time can override a number in
it. It said "~275" skills for weeks while the loader saw 189 (#807), and three
backlog items inherited the figure. Three live counts exist and differ — loaded
by `_iter_skills`, top-level dirs, `SKILL.md` files on disk — so the sentence
must name which one it states, and the guard recomputes the loaded count rather
than comparing against a literal: nightly jobs add and archive skills, and an
equality test would be a permanent flake while a band still catches a 1.45x
drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_mcp.skills import _iter_skills

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "eval" / "lloyd_profile.md"

#: How far the stated count may sit from the live one before the sentence is
#: stale. Ordinary churn is a few skills; the drift this caught was 1.45x.
TOLERANCE = 0.15


def _skills_sentence() -> str:
    """The vault bullet's parenthetical about skills, flattened to one line."""
    text = " ".join(PROFILE.read_text().split())
    m = re.search(r"skills \(`SKILL\.md` per skill;(.*?)\), knowledge notes", text)
    assert m, "the vault bullet no longer carries its skills parenthetical"
    return m.group(1)


def _stated_loaded_count(sentence: str) -> int:
    m = re.search(r"(\d+) loaded", sentence)
    assert m, f"no '<n> loaded' count in the skills sentence: {sentence!r}"
    return int(m.group(1))


def test_no_275_skill_count_survives():
    assert "275" not in PROFILE.read_text()


def test_sentence_names_its_derivation_and_separates_the_disk_counts():
    sentence = _skills_sentence()
    assert "_iter_skills" in sentence, "the loaded count must name how it was derived"
    assert ".archived/" in sentence and "on disk" in sentence, (
        "the on-disk / .archived/ figure must be stated apart from the loaded one"
    )
    _stated_loaded_count(sentence)
    assert re.search(r"\d+ top-level skill dirs", sentence) and re.search(
        r"\d+ `SKILL\.md` files", sentence
    ), "the on-disk counts must be stated as numbers of their own"


@pytest.mark.live_vault
def test_stated_count_is_within_band_of_the_live_loader():
    live = len(list(_iter_skills()))
    if live == 0:
        pytest.skip("no skills visible to the loader (vault not present)")
    stated = _stated_loaded_count(_skills_sentence())
    assert abs(stated - live) <= TOLERANCE * live, (
        f"eval/lloyd_profile.md says {stated} skills loaded; _iter_skills() "
        f"returns {live} — refresh the sentence"
    )


@pytest.mark.parametrize("stated,live,ok", [(187, 189, True), (275, 189, False),
                                             (160, 189, False)])
def test_the_band_is_a_band_not_an_equality(stated, live, ok):
    assert (abs(stated - live) <= TOLERANCE * live) is ok
