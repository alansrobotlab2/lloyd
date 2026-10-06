"""#2019: the success-gated cost advantage, and the ranking that cannot be bought.

The CLM paper's gate, used as a selection rule with no model trained: the efficiency
advantage of a trial is `mean(cost | success) - cost` and a failed trial gets exactly
0, because rewarding cheapness alone rewards deleting the context the task needed.

#2299 bounds the same block to the population it measures. Only an arm in
`cost.USAGE_BEARING_HARNESSES` writes the usage rows the currency is summed from, so a
round that benches both arms has a cost figure for one of them — on the live ledger, 160
of 5,593 trial rows, all of them sdk — and an unpriced trial's advantage contribution is
a bare 0.0, the number a measured break-even also writes. The nodes below `test_the_cost_block_names_the_harnesses_it_measured`
pin that a figure cannot stand unlabeled, that an absent one says why, and that a
phantom 0.0 cannot separate two variants priced on different arms.
"""
from __future__ import annotations

from scripts.autoresearch import cost

C = cost.REPREFILL_FIELD

#: A score shaped like `judge_trace`'s, for the nodes that build rows through the real
#: ledger writer instead of `_row`.
_SCORE = {"composite_score": 0.9, "objective_score": 1.0, "rubric_overall": 0.9,
          "safety_critical": False, "safety_passed": True}


def _row(variant, task, c, *, objective=1.0, status="success"):
    return {"variant_id": variant, "task_id": task, "trace_status": status,
            "objective_score": objective, C: c}


def _hrow(variant, task, c, harness, *, objective=1.0, status="success"):
    """A trial row that names its arm, as both ledger writers' rows do (#2299)."""
    return {**_row(variant, task, c, objective=objective, status=status),
            "harness": harness}


def _trace(variant, task, harness, session="20261001_053224_bench_7a06"):
    """A trace shaped the way `run()` hands one to `trial_ledger_row`."""
    return {"variant_id": variant, "task_id": task, "status": "success",
            "harness": harness, "session_id": session, "turns": 1, "tool_calls": [],
            "denied_calls": [], "duration_seconds": 1.0}


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


# ── #2299: the block is bounded to the population it measured ───────────────────────

def test_the_cost_block_names_the_harnesses_it_measured(monkeypatch):
    """Clause 1. A cost figure cannot stand unlabeled over an sdk-only subset.

    The rows come from `run_round.trial_ledger_row`, the writer `run()` calls for every
    trial, so the harness the block names is the one the ledger row carries rather than
    one this file invented, and the decision block is the one that reaches ledger.jsonl.
    The shape is the live one: one sdk trial with a usage row, three direct trials
    without one.
    """
    from scripts.autoresearch import run_round

    monkeypatch.setattr(cost, "usage_rows_for",
                        lambda sid: [{"input_tokens": 42861, "cache_read": 0}])
    rows = [run_round.trial_ledger_row("R_2299", _trace("V_mixed", "t1", "sdk"), _SCORE),
            *[run_round.trial_ledger_row("R_2299",
                                         _trace("V_mixed", f"t{n}", "direct"), _SCORE)
              for n in (2, 3, 4)]]
    assert [r[C] for r in rows] == [42861, None, None, None], "the writer prices only the sdk arm"

    rec = cost.round_cost_records(rows)["V_mixed"]
    assert (rec["trials"], rec["measured"], rec["unmeasured"]) == (4, 1, 3)
    assert rec["measured_on"] == ["sdk"] and rec["advantage_on"] == ["sdk"]

    block = cost.decision_cost_fields(rec)["cost"]
    assert block["measured_on"] == ["sdk"] and block["unmeasured"] == 3
    keys = list(block)
    assert keys.index("measured_on") == keys.index("unmeasured") + 1, keys
    assert {"advantage_on", "cost_reason"} <= set(keys), keys
    assert block["cost_mean_success"] == 42861.0, "the figure is labelled, never moved"
    assert cost.decision_cost_fields(None) == {}, "no record still means no block"


def test_a_variant_with_nothing_priced_reports_none_and_says_why():
    """Clause 2. None plus the reason, never a bare 0, and a partial figure names the
    arm the unmeasured trials ran on."""
    rows = [_hrow("V_direct", "t1", None, "direct"),
            _hrow("V_direct", "t2", None, "direct", objective=0.0),
            _hrow("V_direct", "t3", None, "direct", status="timeout"),
            _hrow("V_one", "t1", 54058, "sdk"),
            *[_hrow("V_one", f"t{n}", None, "direct") for n in (2, 3)]]
    rec = cost.round_cost_records(rows)

    d = rec["V_direct"]
    assert d["measured"] == 0 and d["trials"] == 3 and d["measured_on"] == []
    assert d["cost_total"] is None and d["cost_mean_valid"] is None
    assert d["cost_mean_success"] is None, "None, never 0"
    assert d["advantage_on"] == []
    reason = d["cost_reason"]
    assert reason and "None" in reason and "never 0" in reason
    assert "2 unmeasured trial(s) on direct" in reason, reason
    assert "records no usage row" in reason, "the reason names the harness without usage rows"
    assert "1 infrastructure-failed trial(s)" in reason, reason
    # The empty sum stays a number; what stops it reading as a measured break-even is the
    # reason beside it, and `rank` never reads it — pinned in the disjoint-set node below.
    assert d["advantage_total"] == 0.0 and "break-even" in reason

    p = rec["V_one"]
    assert p["measured"] == 1 and p["unmeasured"] == 2 and p["measured_on"] == ["sdk"]
    assert p["cost_reason"].startswith("cost priced for 1 of 3 trial(s)"), p["cost_reason"]
    assert "2 unmeasured trial(s) on direct" in p["cost_reason"], p["cost_reason"]
    assert cost.decision_cost_fields(p)["cost"]["cost_reason"] == p["cost_reason"]

    clean = cost.round_cost_records([_hrow("V_clean", "t1", 1000, "sdk")])["V_clean"]
    assert clean["cost_reason"] is None, "a fully measured variant has nothing to explain"


def test_rank_never_separates_harness_disjoint_measured_sets_on_advantage():
    """Clause 3. The advantage values below are stacked so the two orders differ: an
    advantage sort puts the direct-priced variant first, and it must not get there."""
    rows = [_hrow("V_b", "t1", 1000, "direct"), _hrow("V_a", "t1", 9000, "sdk")]
    rec = cost.round_cost_records(rows)
    assert rec["V_a"]["advantage_on"] == ["sdk"]
    assert rec["V_b"]["advantage_on"] == ["direct"]
    assert rec["V_b"]["advantage_total"] == 4000.0 > rec["V_a"]["advantage_total"]
    assert cost.rank(rec) == ["V_a", "V_b"], (
        "successes tie and the measured sets are disjoint by harness: advantage_total "
        "may not separate them, so it falls through to the id")

    # Successes decide even when the advantage points the other way.
    more = cost.round_cost_records([_hrow("V_b", "t1", 1000, "direct"),
                                    _hrow("V_a", "t1", 9000, "sdk"),
                                    _hrow("V_a", "t2", 9000, "sdk")])
    assert more["V_b"]["advantage_total"] > more["V_a"]["advantage_total"]
    assert cost.rank(more) == ["V_a", "V_b"], "two successes beat any advantage"

    # The leg still ranks on advantage where the sets share a harness: the column is not
    # decoration, which is the whole content of #2019.
    same = cost.round_cost_records([_hrow("V_lean", "t1", 10000, "sdk"),
                                    _hrow("V_dear", "t1", 40000, "sdk")])
    assert cost.rank(same) == ["V_lean", "V_dear"]

    # The live misordering (#20261006_164456): one priced sdk trial of 23 leaves a
    # variant with a phantom 0.0 that slotted it between two variants that were priced.
    phantom = cost.round_cost_records([_hrow("V_dear", "t1", 54058, "sdk"),
                                       _hrow("V_lean", "t1", 42861, "sdk"),
                                       _hrow("V_ghost", "t1", None, "direct")])
    assert phantom["V_ghost"]["advantage_total"] == 0.0 and phantom["V_ghost"]["measured"] == 0
    assert phantom["V_dear"]["advantage_total"] == -5598.5
    assert phantom["V_lean"]["advantage_total"] == 5598.5
    assert cost.rank(phantom) == ["V_lean", "V_dear", "V_ghost"], (
        "an unmeasured variant does not get the break-even it was never priced for")


def test_the_round_report_names_the_measured_population():
    """Clause 4. The mean-cost columns must not read as all-trial cost."""
    rec = cost.round_cost_records([_hrow("V_mixed", "t1", 42861, "sdk"),
                                   *[_hrow("V_mixed", f"t{n}", None, "direct")
                                     for n in (2, 3)]])
    lines = cost.report_lines(rec)
    text = "\n".join(lines)
    assert "Measured population: 1 of 3 trial(s) carry a cost, on `sdk`" in text, text
    assert "not every trial" in text
    row = [l for l in lines if l.startswith("| 1 ")][0]
    assert "`sdk`" in row and "*none*" not in row, row
    assert "emit-only" in text and cost.SUCCESS_CONDITION in text, "still emit-only"

    blind = cost.round_cost_records([_hrow("V_direct", "t1", None, "direct")])
    blind_text = "\n".join(cost.report_lines(blind))
    assert "Measured population: 0 of 1 trial(s) carry a cost, on no harness" in blind_text
    assert "No advantage comparison spans this round" in blind_text
    assert "*none*" in [l for l in cost.report_lines(blind) if l.startswith("| 1 ")][0]
