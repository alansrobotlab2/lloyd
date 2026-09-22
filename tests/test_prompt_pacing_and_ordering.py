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
from pathlib import Path

from scripts.automod import backlog as B
from workers.sources import autocode as I
from workers.sources import autotriage as M


def _prompt(**over):
    kw = dict(item_id=9, status="up_next", priority="high", name="n", body="b",
              triaged_ago="today", surface="code", check="c", evidence="e",
              acceptance="a", clauses="    1. a", spawn_cap=I.SPAWN_CAP,
              max_turns=I.DEFAULT_MAX_TURNS, reoffer="", members="",
              human_clauses="", surface_rules="", gate_minutes=16, first_gate_by=23)
    kw.update(over)
    return I.PROMPT.format(**kw)


# ---------------------------------------------------------------------------
# what the round reads before it starts
# ---------------------------------------------------------------------------

def test_the_template_stays_bounded():
    """Not a style rule: the template is half of what a round reads before
    its first tool call, and the other half is the item.

    Under 5.5k chars since cut 4 of senses-not-supervision moved the
    procedure into the vault skill `automod-change-own-code`. What is left is
    the contract: the item, the clauses, the findings and blocker rules,
    pacing, and the finalizer's outcome vocabulary. Procedure the model can
    read once belongs in a file it reads; procedure in a prompt is re-sent
    every iteration and re-derived by every re-offer. 11.8k -> 10.3k -> 4.9k.

    5.5k -> 5.9k on 2026-09-18, for two things that are contract and cannot
    live in the skill. Pacing (+190): the gate's MEASURED minutes, "commit and
    gate, however late", and "never the whole suite" — four rounds that day
    were lost to a clock the old literal marks misdescribed, and 22 turns
    launched the full suite 31 times beside a gate that runs it. And one
    sentence of outcome vocabulary (+127): a clause is judged on the change,
    not on whether it has been promoted — two turns reported every clause
    `not_met` for a finished, gated change. What `landed` means went into the
    finalizer's own prompt and `automod_land`'s result, which cost nothing here.

    5.9k -> 6.1k on 2026-09-22 (+190), for the other half of "never the whole
    suite": never from `~/lloyd`. The 09-18 raise bought the rule and a round
    broke the machine anyway by obeying it in the wrong tree — its gate `tests`
    rung had failed with 24 errors and it re-ran the suite against production to
    ask whether they were pre-existing, which deleted the tree. Contract, not
    procedure, on the same test as the 09-18 line: a re-offered round re-derives
    from this template, and the question it went looking for an answer to is one
    the `tests` rung already answers, so the prompt has to say where that answer
    is. `tests/conftest.py` refuses the run as well — the prompt is what stops
    the attempt, the refusal is what stops the damage.
    """
    assert len(I.PROMPT) < 6_100, f"{len(I.PROMPT)} chars"


SKILL_PATH = Path.home() / "obsidian" / "skills" / "automod-change-own-code" / "SKILL.md"


def _skill_text() -> str:
    return " ".join(SKILL_PATH.read_text(encoding="utf-8").split())


def test_the_prompt_says_what_the_review_grades():
    """Only the vault skill said it, and a round that skipped that section
    learned the grader's rules one refusal at a time: seams (decisive in 4 of
    the grader era's first 21 refusals), suite-level evidence (#860's clause
    8, three times), test-honesty findings — mostly about a test's own prose
    — which led 15 of the 21, and the two-attempt cap that made each lesson
    cost."""
    text = " ".join(_prompt().split())
    assert "process boundary your change crosses" in text
    assert "a suite run cited as `tests/ -k <expr>`" in text
    assert "docstrings, names, assert messages" in text
    assert "two review attempts per round" in text


def test_the_prompt_points_at_the_skill():
    text = _prompt()
    assert "automod-change-own-code" in text
    assert "this message is the contract, not the procedure" in " ".join(text.split())


@pytest.mark.live_vault
def test_the_procedure_lives_in_the_skill():
    """Every rule the prompt stopped restating is in the file it points at.
    Reads the live vault, so it runs where the vault is and is excluded from
    the automod gate like the other `live_vault` tests.
    """
    text = _skill_text()
    for phrase in (
        "Seams.",
        "Do not edit, commit or run anything in the worktree while a gate runs",
        "outside your diff",
        "do not abort on it",
        "prompt_surface",
        "automod_amend_clause",
        "post_landing",
        "is work, not a blocker",
        "automod_vault_land",
        "web/src/**",
        "human_paths",
        "git add -f",
    ):
        assert phrase in text, phrase


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
    text = " ".join(_prompt(gate_minutes=17, first_gate_by=22).split())
    assert "automod_start` by iteration 6" in text
    assert "iteration 25" in text
    # The third mark is measured now, not written down: "minute 30" and "after
    # minute 40 start nothing you cannot gate" described a five-minute gate.
    assert "A full gate takes about 17 minutes now (measured)" in text
    assert "the first `automod_gate` by minute 22" in text
    assert "minute 30" not in text and "minute 40" not in text


def test_the_gate_mark_is_measured_off_the_ledger(tmp_path, monkeypatch):
    """`first_gate_by` leaves room for the slow gate, a refusal and the re-gate
    after it. On 2026-09-18's numbers — median 967 s, p90 1235 s, a 59-minute
    turn — that is minute 23, and the four rounds the clock cost that day
    opened their gates at minutes 37, 40, 42 and 45."""
    from scripts.automod import state as S
    from workers.sources import _common as C
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", ledger)
    monkeypatch.setattr(C, "turn_timeout_for", lambda source, default=3600.0: 3540.0)
    assert B.gate_duration_stats(ledger) == {"n": 0}
    assert I._pacing_marks() == {"gate_minutes": I.DEFAULT_GATE_MINUTES, "first_gate_by": 30}

    def gate(rid, t0, seconds, *, tests="5830 passed", review=True):
        rows = [("preflight", 0.2, "3 file(s) in scope"), ("tests", seconds * 0.55, tests)]
        if review:
            rows.append(("review", seconds * 0.45, "review: 4 met of 4 clause(s)"))
        rows.append(("drill", 0.0, "not required"))
        t = t0
        for rung, secs, detail in rows:
            t += secs
            S.append_event({"event": "gate", "round_id": rid, "rung": rung, "ok": True,
                            "seconds": secs, "detail": detail, "ts": t}, path=ledger)

    for n, secs in enumerate([900, 960, 975, 1000, 1235] * 4):
        gate(f"SM_{n}", 1_000_000 + n * 5000, secs)
    # Not full gates, so not in the figure: a re-gate that re-ran two test
    # files, a reused suite, and a run that never reached the review.
    gate("SM_partial", 2_000_000, 120, tests="pytest (partial, 2 changed test file(s)…): 20 passed")
    gate("SM_reused", 2_010_000, 60, tests="REUSED (nothing committed since)")
    gate("SM_noreview", 2_020_000, 30, review=False)
    stats = B.gate_duration_stats(ledger)
    assert stats["n"] == 20 and 955 <= stats["median_s"] <= 985 and stats["p90_s"] >= 1200
    marks = I._pacing_marks()
    assert marks == {"gate_minutes": 16, "first_gate_by": 23}, marks
    # Faster gates move the mark later, never past the old advice; slower ones
    # move it earlier, never below ten.
    for n in range(20):
        gate(f"SM_fast{n}", 3_000_000 + n * 5000, 300)
    assert I._pacing_marks() == {"gate_minutes": 5, "first_gate_by": 30}
    for n in range(20):
        gate(f"SM_slow{n}", 4_000_000 + n * 5000, 2400)
    assert I._pacing_marks()["first_gate_by"] == 10


def test_pacing_says_gate_late_rather_than_not_at_all_and_never_the_whole_suite():
    text = " ".join(_prompt().split())
    assert "commit and gate, however late" in text
    assert "a gate that passes after your turn ends is landed by the loop" in text
    assert "When a gate passes, `automod_land` at once" in text
    assert "Run the tests you changed, never the whole suite" in text


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
        "backlog_write_task", "SPAWNED", "item_id=9",
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
