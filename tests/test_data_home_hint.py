"""The runtime-data paragraph in the system prompt (`prompt_builder._data_home_hint`).

The platform paragraph named the checkout (`Home:`) and nothing else, and on 2026-10-02
a nightly task went looking for its run history there: `sqlite3 workers.db` from the
tree created an empty database that alerted hourly. Pinned: every platform's prompt
names the data root, the store and table that turn was looking for, and the rule —
built from `app.paths`, so a relocated root moves the sentence with it.

The paragraph answers one more question now (#3095), in the other direction. Having
been told that everything Lloyd writes goes under the data root, an autotriage session
ran `git worktree add --detach ~/lloyd-data/triage-3087/check HEAD` to get a tree the
test suite would accept, and its own clean-up 55 seconds later tripped the guardian's
data tripwire critical — 3,347 files vanishing from a watched root halts promotions
and pauses the worker pool. So the paragraph that says where writes belong has to say
where a THROWAWAY CHECKOUT belongs, and that route is a declared prompt marker so a
session's `PROMPT_MARKERS` row bears witness to it rather than the code merely
containing it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import prompt_builder as PB  # noqa: E402
from app.paths import DATA_ROOT, LLOYD_HOME, SCRATCH_CHECKOUT_ROOT  # noqa: E402


def test_the_paragraph_is_in_every_platforms_prompt():
    for platform in ("", "worker", "autonomy"):
        text = PB.build_system_prompt(include_skills_index=False, memories_text="",
                                      platform=platform)
        assert PB._data_home_hint() in text, platform


def _prose_only(hint: str) -> str:
    """The paragraph's English, with every interpolated root masked to a fixed stub.

    WHY THE CEILING BELOW IS NOT ON THE RAW STRING. The paragraph interpolates
    three roots, and their widths are properties of the machine, not of the
    prompt: a gate run resolves `DATA_ROOT`, `LLOYD_HOME` and the scratch root
    under the round's own home (`scripts/automod/worktree.py::round_data_root`),
    where this paragraph measures 885 characters against 780 here. The same
    paragraph masked measures 603 in BOTH layouts, which is the only figure that
    says something about the prose. A raw-length bound is therefore an instrument
    reading its own probe — it failed the #3095 gate on a run whose sentence was
    word-for-word the one that passed locally, because the round's paths happen to
    be 105 characters longer.
    """
    out = hint
    # Longest root first: `LLOYD_HOME` is a prefix of `DATA_ROOT` in a round's
    # layout (`…/home/lloyd` inside `…/home/lloyd-data`), so masking the short one
    # first would shred the long one and leave a stub count that proves nothing.
    for real, stub in sorted(((str(DATA_ROOT), "<DATA>"),
                              (str(LLOYD_HOME), "<HOME>"),
                              (str(SCRATCH_CHECKOUT_ROOT), "<SCRATCH>")),
                             key=lambda pair: -len(pair[0])):
        out = out.replace(real, stub)
    return out


def test_it_names_the_root_the_store_and_the_rule():
    hint = PB._data_home_hint()
    assert hint.startswith("Runtime data:")
    assert str(DATA_ROOT) in hint and str(LLOYD_HOME) in hint
    assert "workers.db" in hint and "`runs`" in hint
    assert "code only" in hint
    assert "creates an empty database" in hint
    prose = _prose_only(hint)
    assert "<DATA>" in prose and "<HOME>" in prose and "<SCRATCH>" in prose, (
        "the mask has to have masked: an assertion on a length that never "
        "interpolated anything would pass on any string this short")
    assert len(prose) < 700, (
        "a paragraph, not a policy document. 700 is the old raw ceiling of 800 "
        "restated on the prose alone: this paragraph's English measured 603 masked "
        "with #3095's sentence naming the throwaway-checkout root in it, so the "
        "room added is that sentence, and the ceiling did not move to fit it.")


def test_it_names_where_a_throwaway_checkout_goes():
    """#3095: the paragraph sent every write to the data root and said nothing about
    a scratch git checkout, so a triage session put one there and its own cleanup
    tripped the tripwire. The route has to be spelled, not implied."""
    hint = PB._data_home_hint()
    assert "throwaway git checkout" in hint
    assert str(SCRATCH_CHECKOUT_ROOT) in hint, (
        "the sanctioned root, spelled — the same value `agent_mcp/_bash_tree_strays.py` "
        "prints on the call that ignored it, so the two can never disagree")
    assert "never the data root" in hint
    assert "#3095" in hint, "the sentence says which incident it exists for"
    assert str(SCRATCH_CHECKOUT_ROOT) != str(DATA_ROOT), (
        "and the two roots are genuinely different places: `DATA_ROOT` follows the "
        "tree that imports it, a scratch checkout has one home")


def test_the_scratch_route_is_a_declared_prompt_marker():
    """The route is prose, and prose is only in the prompt if the assembly kept it.
    Declared in `PROMPT_MARKERS`, so `journal_prompt_markers` records `present` or
    `absent` for it on every assembly and a session that never got the sentence is
    readable from its own row rather than inferred from a model's behaviour."""
    assert "scratch-checkout-home" in PB.PROMPT_MARKERS
    needle = PB.PROMPT_MARKERS["scratch-checkout-home"]
    assert needle in PB._data_home_hint()
    for platform in ("", "worker", "autonomy"):
        text = PB.build_system_prompt(include_skills_index=False, memories_text="",
                                      platform=platform)
        assert needle in text, platform


def test_it_sits_directly_after_the_line_that_names_the_checkout():
    text = PB.build_system_prompt(include_skills_index=False, memories_text="")
    home = text.index("Platform: Lloyd (local harness).")
    assert home < text.index(PB._data_home_hint())
    between = text[home:text.index(PB._data_home_hint())]
    assert between.count("\n\n") == 1, "one paragraph break, nothing in between"
