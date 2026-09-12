"""What a round is told before it starts, and which round runs next.

Three things, all measured on the 2026-09-11 rounds:

  * **What a round reads before its first tool call.** The template is ~10k
    chars, the item body was capped at 30k and an umbrella's members at 24k —
    about 13k tokens of a 262k window, spent before anything happened. Three
    rounds died at the context wall.
  * **No sense of its own clock.** 875 spent 18 minutes before `automod_start`
    and hit the wall at iteration 50-odd with 44 minutes left.
  * **Which item goes next.** A re-offer whose branch still holds the work is
    a fix cycle; a fresh umbrella is an hour and two review attempts. Five of
    the first sat behind the second.
"""

from __future__ import annotations

import pytest

from scripts.automod import backlog as B
from workers.sources import autocode as I
from workers.sources import autotriage as M


def _prompt(**over):
    kw = dict(item_id=9, status="up_next", priority="high", name="n", body="b",
              triaged_ago="today", surface="code", check="c", evidence="e",
              acceptance="a", clauses="    1. a", spawn_cap=I.SPAWN_CAP,
              max_turns=I.DEFAULT_MAX_TURNS, reoffer="", members="",
              human_clauses="")
    kw.update(over)
    return I.PROMPT.format(**kw)


# ---------------------------------------------------------------------------
# what the round reads before it starts
# ---------------------------------------------------------------------------

def test_the_template_stays_bounded():
    """Not a style rule: the template is half of what a round reads before
    its first tool call, and the other half is the item.

    10.5k chars is ~2.6k tokens. It is deliberately not lower — the pacing
    block, the `human_paths` rule and the external-failure guidance all
    earned their place by a round dying without them — but it must not drift
    upward unnoticed, which is how it reached 11.8k.
    """
    assert len(I.PROMPT) < 10_500, f"{len(I.PROMPT)} chars"


def test_the_item_body_is_capped_well_under_the_old_thirty_thousand():
    import inspect

    src = inspect.getsource(I)
    assert "body=candidate.body[:12_000]" in src
    assert "body[:30_000]" not in src


def test_the_members_block_is_capped_too():
    """An umbrella brought 30k of body plus 24k of members."""
    import inspect

    src = inspect.getsource(I._members_block)
    assert "12_000" in src
    assert "24_000" not in src


# ---------------------------------------------------------------------------
# pacing
# ---------------------------------------------------------------------------

def test_the_prompt_carries_the_turns_it_actually_has():
    text = _prompt(max_turns=150)
    assert "150 iterations" in text


def test_the_pacing_block_names_the_three_marks():
    text = _prompt()
    assert "automod_start` by iteration 6" in text
    assert "iteration 25" in text
    assert "minute 30" in text
    assert "minute 40" in text


def test_pacing_says_commit_before_every_gate():
    """A gate you have not committed for is answered from the ledger without
    a review — a wasted one of the round's two attempts.
    """
    text = " ".join(_prompt().split())
    assert "Commit before every gate" in text
    assert "answered from the ledger" in text


def test_pacing_says_an_anchor_is_not_advice():
    text = " ".join(_prompt().split())
    assert "<context>" in text and "<budget>" in text
    assert "it is not advice" in text


def test_the_eval_commands_are_gone_from_the_prompt():
    text = _prompt()
    assert "run_tool_choice_eval.py" not in text
    assert "compare_tool_choice.py" not in text
    assert "prompt_surface" in text, "the model must know the check still runs"


def test_the_external_failure_guidance_is_stated():
    """Round 866-a ended its turn on a grader 503 caused by a SIBLING round's
    landing, and was reaped 30 minutes later with the work intact.
    """
    text = " ".join(_prompt().split())
    assert "outside your diff" in text
    assert "do not abort on it" in text.lower()


def test_human_paths_is_stated_with_its_prohibition():
    text = " ".join(_prompt().split())
    assert "human_paths" in text
    assert "git add -f" in text


def test_every_phrase_the_other_tests_pin_survived_the_trim():
    """The trim's one rule. Each of these is somewhere a property was lost
    once and a test was written to stop it happening again.
    """
    text = " ".join(_prompt().split())
    for phrase in (
        "backlog_write_task", "SPAWNED", "Seams.", "automod_gate_wait",
        "automod_amend_clause", "Do not edit, commit or run anything in the",
        "A deferral that names no id is recorded as `not_met`", "per clause",
        "closed automatically", "never re-triaged",
        "do not leave them only in your report",
        "`unnecessary` means the work is not needed after all",
    ):
        assert phrase in text, phrase


# ---------------------------------------------------------------------------
# triage clause rules
# ---------------------------------------------------------------------------

def test_triage_requires_jointly_satisfiable_clauses():
    """#875 carried a clause fixing `n` and a clause requiring a floor only a
    different `n` reaches. No diff could satisfy both, so the round could only
    be refused.
    """
    text = " ".join(M.PROMPT.split())
    assert "jointly satisfiable" in text
    assert "trade-off is ONE clause" in text


def test_triage_requires_a_purpose_clause_when_a_gate_is_weakened():
    text = " ".join(M.PROMPT.split())
    assert "purpose clause" in text
    assert "what that gate exists to catch" in text


@pytest.mark.parametrize("clause,is_post_landing", [
    ("a day of real traffic shows no regression", True),
    ("24 hours of traffic with no errors", True),
    ("confirmed after it has landed", True),
    ("post-landing the dashboard shows the gate engaged", True),
    ("once it is live the counter is non-zero", True),
    ("the next nightly run reports zero", True),
    # ...and the ones a gate CAN judge must not be moved.
    ("`workers/pool.py::_round_hold_held` returns the held sources", False),
    ("a test asserts the traffic shaper drops the frame", False),
    ("the function is covered by tests/test_x.py::test_y", False),
    ("", False),
])
def test_the_post_landing_backstop_is_narrow(clause, is_post_landing):
    """A false positive moves a gradeable clause out of the graded contract,
    which is worse than a false negative — the review rung's own
    `post_landing` verdict catches what this misses.
    """
    graded, later = B.split_post_landing_clauses([clause] if clause else [])
    assert bool(later) is is_post_landing, (clause, graded, later)


def test_the_backstop_keeps_the_rest_of_the_contract():
    graded, later = B.split_post_landing_clauses([
        "the function returns the held sources",
        "a day of real traffic shows no regression",
        "a test pins the empty case",
    ])
    assert len(graded) == 2
    assert len(later) == 1


# ---------------------------------------------------------------------------
# re-offer ordering
# ---------------------------------------------------------------------------

def test_a_first_re_offer_with_a_kept_branch_is_near(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"event": "backlog_implement", "phase": "started", "item_id": 7, '
        '"round_id": "SM_A"}\n')
    from scripts.automod import worktree as W
    monkeypatch.setattr(W, "branch_exists", lambda repo, br: br == "automod/SM_A")
    got = B.first_reoffer_with_a_branch(ledger, {7: ("review_retry", "x")})
    assert got == {7}


def test_a_second_re_offer_is_not_near(monkeypatch, tmp_path):
    """It has already had its fix cycle. Letting it keep jumping the queue is
    the monopoly the oldest-first ordering exists to prevent.
    """
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"event": "backlog_implement", "phase": "started", "item_id": 7, '
        '"round_id": "SM_A"}\n'
        '{"event": "backlog_implement", "phase": "started", "item_id": 7, '
        '"round_id": "SM_B"}\n')
    from scripts.automod import worktree as W
    monkeypatch.setattr(W, "branch_exists", lambda repo, br: True)
    assert B.first_reoffer_with_a_branch(ledger, {7: ("review_retry", "x")}) == set()


def test_a_re_offer_whose_branch_is_gone_is_not_near(monkeypatch, tmp_path):
    """Without the branch there is no work to resume, so it is a fresh hour
    like anything else."""
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"event": "backlog_implement", "phase": "started", "item_id": 7, '
        '"round_id": "SM_A"}\n')
    from scripts.automod import worktree as W
    monkeypatch.setattr(W, "branch_exists", lambda repo, br: False)
    assert B.first_reoffer_with_a_branch(ledger, {7: ("review_retry", "x")}) == set()


def test_a_spent_item_is_never_near(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"event": "backlog_implement", "phase": "started", "item_id": 7, '
        '"round_id": "SM_A"}\n')
    from scripts.automod import worktree as W
    monkeypatch.setattr(W, "branch_exists", lambda repo, br: True)
    assert B.first_reoffer_with_a_branch(ledger, {7: ("spent", "x")}) == set()


def test_a_git_failure_does_not_break_selection(monkeypatch, tmp_path):
    """Ordering is not correctness: a repo that cannot be read costs a place
    in the queue, never a round.
    """
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"event": "backlog_implement", "phase": "started", "item_id": 7, '
        '"round_id": "SM_A"}\n')
    from scripts.automod import worktree as W

    def _boom(*a, **k):
        raise OSError("no git here")
    monkeypatch.setattr(W, "branch_exists", _boom)
    assert B.first_reoffer_with_a_branch(ledger, {7: ("review_retry", "x")}) == set()
