"""#2019: the success-gated cost advantage, and the ranking that cannot be bought.

The CLM paper's gate, used as a selection rule with no model trained: the efficiency
advantage of a trial is `mean(cost | success) - cost` and a failed trial gets exactly
0, because rewarding cheapness alone rewards deleting the context the task needed.
"""
from __future__ import annotations

from scripts.autoresearch import cost

C = cost.REPREFILL_FIELD


def _row(variant, task, c, *, objective=1.0, status="success"):
    return {"variant_id": variant, "task_id": task, "trace_status": status,
            "objective_score": objective, C: c}


def test_the_advantage_is_gated_on_success_and_names_the_condition():
    """Clause 3. Successful: mean over successful trials of the same task minus this
    trial's cost. Failed: exactly 0. Infrastructure-failed: in neither the mean nor
    the zeroing — a timeout is not a task failure."""
    rows = [_row("A", "t1", 1000), _row("B", "t1", 3000),
            _row("C", "t1", 10, objective=0.5),                     # cheap and wrong
            _row("D", "t1", 99999, status="timeout", objective=None),
            _row("A", "t2", 500), _row("B", "t2", 500)]
    rec = cost.round_cost_records(rows)
    for r in rec.values():
        assert r["success_condition"] == cost.SUCCESS_CONDITION
        assert r["currency"] == cost.REPREFILL_FIELD
    adv = {v: {t["task_id"]: t["advantage"] for t in rec[v]["per_trial"]} for v in rec}
    assert adv["A"] == {"t1": 1000.0, "t2": 0.0}, "mean(1000, 3000) - 1000"
    assert adv["B"] == {"t1": -1000.0, "t2": 0.0}
    assert adv["C"] == {"t1": 0.0}, "a failed trial is zeroed, however cheap"
    assert adv["D"] == {"t1": None}, "an infra failure is not zeroed: it is excluded"

    # C's cost 10 and D's 99999 are in no successful-trial mean.
    assert rec["A"]["cost_mean_success"] == 750.0 and rec["B"]["cost_mean_success"] == 1750.0
    assert rec["C"]["cost_mean_success"] is None and rec["C"]["failures"] == 1
    assert rec["D"]["infra_excluded"] == 1 and rec["D"]["failures"] == 0
    assert rec["D"]["cost_total"] is None and rec["D"]["cost_mean_valid"] is None
    assert rec["D"]["advantage_total"] == 0.0 and rec["D"]["successes"] == 0
    # A failed trial's cost is still a valid measurement of what the variant spent.
    assert rec["C"]["cost_mean_valid"] == 10.0 and rec["C"]["cost_total"] == 10


def test_an_unmeasured_cost_is_counted_and_never_priced_at_zero():
    rows = [_row("A", "t1", None), _row("B", "t1", 2000)]
    rec = cost.round_cost_records(rows)
    assert rec["A"]["unmeasured"] == 1 and rec["A"]["cost_mean_valid"] is None
    assert rec["A"]["per_trial"][0]["advantage"] is None
    assert rec["B"]["per_trial"][0]["advantage"] == 0.0, "the mean is over measured successes"


def test_a_cheaper_and_wrong_variant_cannot_outrank_the_correct_expensive_one():
    """Clause 4, the #703-shaped fire test: inject the record a cost-only rule would
    promote — a tenth of the cost, wrong on every task — and rank the round."""
    rows = [_row("V_correct", "t1", 40000), _row("V_correct", "t2", 40000),
            _row("V_cheap_wrong", "t1", 4000, objective=0.0),
            _row("V_cheap_wrong", "t2", 4000, objective=0.5)]
    rec = cost.round_cost_records(rows)
    assert rec["V_cheap_wrong"]["advantage_total"] == 0.0
    assert rec["V_cheap_wrong"]["cost_mean_valid"] < rec["V_correct"]["cost_mean_valid"]
    assert cost.rank(rec) == ["V_correct", "V_cheap_wrong"]

    lines = cost.report_lines(rec)
    table = [l for l in lines if l.startswith("| ") and "`V_" in l]
    assert table[0].startswith("| 1 | `V_correct`") and table[1].startswith("| 2 | `V_cheap_wrong`")
    assert "emit-only" in "\n".join(lines) and cost.SUCCESS_CONDITION in "\n".join(lines)

    # The direction the gate exists for, shown able to move: among two CORRECT
    # variants the cheaper one ranks first, so the column is not decoration.
    both = cost.round_cost_records([_row("V_dear", "t1", 40000), _row("V_lean", "t1", 10000),
                                    _row("V_cheap_wrong", "t1", 100, objective=0.0)])
    assert cost.rank(both) == ["V_lean", "V_dear", "V_cheap_wrong"]
    assert both["V_lean"]["advantage_total"] == 15000.0
    assert cost.report_lines({}) == []
