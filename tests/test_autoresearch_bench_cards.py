"""#4650 clause 5: the durable-write cards keep a `max_tool_calls` bound, at a number the
measured trials can meet.

Why this file exists
--------------------
The three `writes:`-scoped bench cards each declare a `max_tool_calls` objective check, and
all three declared a bound below what the trials actually cost. bench_031 declared 4 against
per-trial `tool_call_count` measurements of 5, 14, 14, 14, 15, 15, 15, 15, 16 and 18 — the
check could not pass on any trial the round ever ran, so it was a permanent 0 on the card's
objective layer and the card's ceiling looked like a model failure when it was a card bug.
The obvious wrong fix is to delete the check; the item names that explicitly ("instead of
deleting the check"), and a card with no bound stops measuring efficiency at all.

The bound is therefore RE-DECLARED from measurement. The figures below are not invented and
not quoted from a prior handoff: they are every non-null `tool_call_count` on a `per_trial`
row for these three cards in the live ledger, read with a grep because
`~/lloyd-data/_pipeline/research/ledger.jsonl` is 37 MB / 32,918 rows and over
`WITNESS_MAX_BYTES`, so ordering it whole is forbidden by #4650 clause 6:

    grep '"task_id": "bench_0[23][901]_durable_write' \\
        ~/lloyd-data/_pipeline/research/ledger.jsonl \\
      | python3 -c "import sys,json;[print(json.loads(l)['task_id'], \\
            json.loads(l).get('tool_call_count')) for l in sys.stdin]" \\
      | sort

Re-read that way on 2026-11-27: bench_029 → 3,4,5,5,6,6,7,8,8 (9 trials, max 8);
bench_030 → 2,2,2,2,2,2,2,2,3 (9 trials, max 3); bench_031 → 5,14,14,14,15,15,15,15,16,18
(9 trials, max 18). A bound is a ceiling on waste, so it is set at the observed maximum:
above it a trial is doing something the bench has never seen it need, which is the only
claim a bound can honestly make without a measurement nobody took.

These nodes read the REAL corpus at `load_config().paths.bench_dir`, the way
`tests/test_autoresearch_judge_compare.py` does, because the artifact under test is the card
file itself — a stubbed bench directory here would pin nothing about the file the round loads.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.autoresearch import judge
from scripts.autoresearch.common import load_bench_tasks, load_config

#: Measured maximum `tool_call_count` per card, from the grep above (9 admitted writes
#: trials each, rounds `R_20261122_153643` .. `R_20261127_235240`). Keyed by card id.
MEASURED_MAX_CALLS = {
    "bench_029_durable_write_vault_note": 8,
    "bench_030_durable_write_kg_row": 3,
    "bench_031_durable_write_backlog_card": 18,
}

CARDS = sorted(MEASURED_MAX_CALLS)


def _tasks() -> dict[str, dict]:
    cfg = load_config()
    tasks = load_bench_tasks(cfg.paths.bench_dir)
    assert tasks, f"the live bench at {cfg.paths.bench_dir} read as empty"
    return {t["id"]: t for t in tasks}


def _check(task: dict) -> dict:
    checks = [c for c in (task.get("objective_checks") or [])
              if c.get("type") == "max_tool_calls"]
    assert len(checks) == 1, (
        f"{task['id']} declares {len(checks)} `max_tool_calls` checks; the clause requires "
        "exactly one re-declared bound, and zero is the deletion this clause forbids")
    return checks[0]


@pytest.mark.parametrize("card_id", CARDS)
def test_the_card_keeps_a_bound_the_measured_trials_can_meet(card_id):
    """Clause 5 both ways: the check is STILL DECLARED, and its value is ≥ the largest call
    count a real admitted trial of that card ever recorded.

    The `is not None` half is the anti-deletion half and it is checked first, because the two
    halves fail differently: a deleted check makes this node pass nothing at all, which is how
    a bound quietly disappears from a corpus with no reviewer reading the diff line by line.
    """
    task = _tasks()[card_id]
    check = _check(task)
    bound = int(check["value"])
    assert bound >= MEASURED_MAX_CALLS[card_id], (
        f"{card_id} bounds the trial at {bound} calls but the measured maximum is "
        f"{MEASURED_MAX_CALLS[card_id]}, so the check fails on every trial and the card's "
        "objective layer cannot reach 1")


@pytest.mark.parametrize("card_id", CARDS)
def test_a_trial_at_the_measured_maximum_passes_and_one_call_over_it_does_not(card_id):
    """The re-declared bound is a live check, not a formality: at the measured maximum it
    scores True, and one call beyond the bound it scores False.

    Without the second half a fix could have declared the bound as 999 and passed the node
    above. The `+1` case is what keeps the check able to fail, which is the whole reason it is
    worth keeping instead of deleting.
    """
    task = _tasks()[card_id]
    check = _check(task)
    bound = int(check["value"])
    measured = MEASURED_MAX_CALLS[card_id]

    def trace(n: int) -> dict:
        return {"tool_calls": [{"name": "Read", "args": {}} for _ in range(n)],
                "denied_calls": [], "final_text": "done", "messages": []}

    assert judge._match_check(check, trace(measured)) is True
    assert judge._match_check(check, trace(bound)) is True
    assert judge._match_check(check, trace(bound + 1)) is False


def test_the_three_cards_still_declare_their_write_tool_check():
    """The objective layer the reward is published beside has to be passable as a whole.

    Each card declares `tool_called` with the legacy `mcp__<server>__<tool>` spelling and the
    agent loop records the bare name; #4650's judge half makes that pair resolve, and this
    node checks the pairing on the real card files rather than on a copy, so a card that
    re-spelled its value to dodge the judge — the other available wrong fix — fails here.
    """
    tasks = _tasks()
    for card_id in CARDS:
        tool_check = next((c for c in tasks[card_id].get("objective_checks") or []
                           if c.get("type") == "tool_called"), None)
        assert tool_check is not None, f"{card_id} lost its tool_called check"
        declared = str(tool_check["value"])
        assert declared.startswith("mcp__lloyd-mcp__"), (
            f"{card_id} declares {declared!r}: re-spelling the declaration to dodge the "
            "prefix mismatch would leave the corpus inconsistent and is not the fix")
        bare = declared[len("mcp__lloyd-mcp__"):]
        assert judge._match_check(tool_check, {
            "tool_calls": [{"name": bare, "args": {}}], "denied_calls": [],
            "final_text": "done", "messages": []}) is True
        # …and the file the card sits in is the file the bound was edited into, so a test
        # that read a stale copy of the corpus could not see the clause at all.
        src = tasks[card_id].get("_source")
        if src:
            assert Path(src).name == f"{card_id}.md"
