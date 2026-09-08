"""A selfmod round's worktree path must not make every call look like a repeat.

Why this file exists
--------------------
Round SM_20260908_165950 (backlog #377) took five deterministic repetition
injects across its 34 minutes. All five were false. Every Bash call in a
selfmod round opens `cd /home/<user>/lloyd-work/SM_<round>/home/lloyd &&`, and
`sm_20260908_165950` survives `_strip_ambient` (it has underscores), clears the
12-character identifier floor, and reaches 18 characters — so `_is_distinctive`
treats it as a specific symbol the primary is chasing and lets it carry a near
match on its own. Four of the five injects named it first.

The cost was not the noise. `interventions_used` is capped, the fifth inject
exhausted it (`inner_voice.deterministic_budget_exhausted`, 17:26:42), and the
round reached `selfmod_gate` and `selfmod_land` ninety seconds later with no
guard left. Every implement round runs in a worktree, so this fires in every
round: it is structural.

Two fixes, and the tests below pin both:

* `_strip_cd_prefix` — the working-directory preamble is not the subject of the
  call, so it is dropped before identifiers are extracted.
* `ubiquitous_identifiers` — an identifier carried by every call in the ring
  discriminates between none of them, so it cannot carry a match either. Three
  details are load-bearing: it is measured over the whole ring rather than the
  comparison window (within a window a chased symbol looks identical to an
  ambient one — `test_a_chased_symbol_is_not_treated_as_ambient`); the observer
  passes the full ring because after a fire it compares only the calls since
  its last one; and it reads `all_idents`, the pre-`cd`-strip view, or a term
  that reaches the ring by two different idioms reads as ambient in neither.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.inner_voice import guards  # noqa: E402
from fixture_iv_selfmod_worktree import WORKTREE_BASH_CALLS  # noqa: E402

WORKTREE = "/home/alansrobotlab/lloyd-work/SM_20260908_165950/home/lloyd"


def _sig(command: str):
    return guards.tool_call_signature("Bash", {"command": command})


def _replay(calls, ring_cap: int = 16) -> list[int]:
    """Model the pretool hook exactly: full ring, baseline instead of clearing.

    The observer keeps `recent_tool_calls` whole and compares only the calls
    made since it last fired (`repetition_baseline`). A replay that cleared
    the ring would hand the guard a three-call history right after a fire and
    measure a different thing entirely — which is the bug this file exists for.
    """
    ring: list = []
    seen = 0
    baseline = 0
    fired: list[tuple[int, tuple[str, ...]]] = []
    for idx, command in calls:
        ring.append(_sig(command))
        seen += 1
        if len(ring) > ring_cap:
            del ring[:-ring_cap]
        since = seen - baseline
        comparable = ring[-since:] if since > 0 else []
        verdict = guards.repetition_verdict(
            comparable, ambient=guards.ubiquitous_identifiers(ring),
        )
        if verdict is not None:
            fired.append((idx, verdict.shared_terms))
            baseline = seen
    return fired


# ── the round that produced the false fires ─────────────────────────────────

def test_no_fire_is_attributable_to_the_worktree_id():
    """The precise claim: the round id can no longer carry a match.

    It was the first-named term in four of the five real injects, and the sole
    term in the fifth. Asserting on the *terms* rather than the fire count is
    what makes this a test of the fix and not of the fixture.
    """
    for idx, terms in _replay(WORKTREE_BASH_CALLS):
        assert "sm_20260908_165950" not in terms, (
            f"call {idx} still matched on the worktree id: {terms}"
        )


def test_the_real_round_drops_from_five_false_fires_to_one_real_one():
    """Five injects, all false, exhausting the budget as the round reached its gate.

    What survives is call 77, and it is a different animal: iterations 74, 75
    and 77 really do re-run pyflakes and pytest over the same five files, and
    the verdict names `prompt_builder` / `test_prompt_surface_budget` — symbols
    those commands actually share. Whether that deserves an inject is the
    guard's ordinary business. It is not this bug, which was five fires that
    named a directory.
    """
    fired = _replay(WORKTREE_BASH_CALLS)
    assert [idx for idx, _ in fired] == [77], fired
    assert "prompt_builder" in dict(fired)[77]


def test_the_round_id_is_not_an_identifier_of_the_call():
    """It was the shared term four of the five injects named first."""
    sig = _sig(f"cd {WORKTREE} && sed -n '255,420p' prompt_builder.py")
    assert "sm_20260908_165950" not in sig.idents
    assert "prompt_builder" in sig.idents


def test_the_cd_preamble_still_shows_in_the_preview():
    """`exact` and `preview` keep the whole command — only idents are stripped."""
    command = f"cd {WORKTREE} && ls"
    sig = _sig(command)
    assert sig.exact == command
    assert "lloyd-work" in sig.preview


def test_chained_cd_hops_are_all_stripped():
    sig = _sig("cd /tmp/aaa_bbb_ccc && cd /tmp/ddd_eee_fff && grep -n target_symbol x.py")
    assert "aaa_bbb_ccc" not in sig.idents and "ddd_eee_fff" not in sig.idents
    assert "target_symbol" in sig.idents


def test_a_quoted_cd_path_is_stripped():
    sig = _sig("cd '/tmp/with space/aaa_bbb_ccc' && grep -n target_symbol x.py")
    assert "aaa_bbb_ccc" not in sig.idents


def test_a_semicolon_hop_is_stripped_too():
    sig = _sig("cd /tmp/aaa_bbb_ccc ; grep -n target_symbol x.py")
    assert "aaa_bbb_ccc" not in sig.idents


def test_cd_without_a_following_command_is_left_alone():
    """`cd somewhere` on its own IS the call; there is nothing else to be about."""
    sig = _sig("cd /tmp/aaa_bbb_ccc")
    assert "aaa_bbb_ccc" in sig.idents


# ── the guard still has to work ─────────────────────────────────────────────

def test_a_chased_symbol_is_not_treated_as_ambient():
    """The failure mode of the first cut of this fix.

    Computed over the comparison window alone, a symbol the primary is chasing
    is in every compared call and reads exactly like the worktree id — so the
    ambient filter would strip it and the guard would go silent precisely when
    it should fire. Ambient is measured over the whole ring for this reason.
    """
    ring = [_sig(f"cd {WORKTREE} && grep -rn 'iv_inject_queue' --include='*.py' {flag}")
            for flag in ("app/", "app/ | head", "app/ -l", "app/ -c", "app/ --color")]
    verdict = guards.repetition_verdict(ring)
    assert verdict is not None, "the guard stopped catching a real reformulation loop"
    assert "iv_inject_queue" in verdict.shared_terms


def test_verbatim_repeats_inside_a_worktree_still_fire():
    """Stripping the preamble must not make identical calls compare as different."""
    once = f"cd {WORKTREE} && pytest -q tests/test_thing.py"
    verdict = guards.repetition_verdict([_sig(once)] * 4)
    assert verdict is not None and verdict.exact


def test_ambient_needs_more_history_than_the_comparison_window():
    """Under the floor, nothing is called ambient — too little to generalise from."""
    sigs = [_sig(f"cd {WORKTREE} && grep -n shared_symbol_here f{i}.py") for i in range(3)]
    assert guards.ubiquitous_identifiers(sigs) == frozenset()


def test_a_term_in_every_call_of_a_long_ring_is_ambient():
    sigs = [_sig(f"grep -n unrelated_thing_{i} ambient_marker_token.py") for i in range(10)]
    assert "ambient_marker_token" in guards.ubiquitous_identifiers(sigs)
