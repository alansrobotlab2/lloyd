"""#2019: the success-gated cost advantage, and the ranking that cannot be bought.

The CLM paper's gate, used as a selection rule with no model trained: the efficiency
advantage of a trial is `mean(cost | success) - cost` and a failed trial gets exactly
0, because rewarding cheapness alone rewards deleting the context the task needed.

#2299 bounds the same block to the population it measures. Only an arm in
`cost.USAGE_BEARING_HARNESSES` can be priced at all, so a round that benches an arm
outside it has no cost figure for those trials — on the live ledger when the tuple held
one arm, 160 of 5,593 trial rows were priced, all of them sdk — and an unpriced trial's
advantage contribution is a bare 0.0, the number a measured break-even also writes. The
nodes below `test_the_cost_block_names_the_harnesses_it_measured` pin that a figure
cannot stand unlabeled, that an absent one says why, and that a phantom 0.0 cannot
separate two variants priced on different arms.

#2390 grows that tuple to both bench arms, which changes what the block is bounded to
and what keeps the ranking honest. Each arm is priced by its own route — an agent turn
from `usage.db`, a direct trial from the engine usage block on its trace — so the two
unpriced causes have to stay two findings rather than one sentence, and the advantage
group is now keyed on (task, harness) because the two routes measure two different
quantities and a mean over both belongs to neither. The nodes below
`test_the_priced_set_is_the_one_tuple_and_the_two_causes_stay_told_apart` pin that pair.
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
    assert [r[C] for r in rows] == [42861, None, None, None], (
        "the direct traces carry no engine usage block, so nothing prices them")

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
    assert "engine reported no prompt_tokens_details" in reason, (
        "#2390: the direct arm is priced from its trace now, so an unpriced direct "
        "trial is missing the engine's usage block, not a usage row it never wrote")
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
    """Clause 3. Two variants priced on different arms are not compared on advantage.

    Since #2390 keyed the group on (task, harness), a pair like this one cannot even
    *have* differing advantage totals: each trial is the only member of its own
    (task, arm) group, so both sit at 0.0. Before the split the same two rows stood 4000
    apart, because the direct trial was being scored against an average that included the
    sdk price — which is the phantom gap this node was written to refuse, now refused one
    stage earlier by the grouping. What the guard still bites on is the case below where a
    variant carries a nonzero advantage from a shared arm and an unpriced one does not.
    """
    rows = [_hrow("V_b", "t1", 1000, "direct"), _hrow("V_a", "t1", 9000, "sdk")]
    rec = cost.round_cost_records(rows)
    assert rec["V_a"]["advantage_on"] == ["sdk"]
    assert rec["V_b"]["advantage_on"] == ["direct"]
    assert rec["V_a"]["advantage_total"] == 0.0 == rec["V_b"]["advantage_total"], (
        "each trial is its group's only member: neither is scored against the other arm")
    assert cost.rank(rec) == ["V_a", "V_b"], (
        "successes tie and the measured sets are disjoint by harness: advantage_total "
        "may not separate them, so it falls through to the id")

    # Successes decide, whatever the advantage column says about the two arms.
    more = cost.round_cost_records([_hrow("V_b", "t1", 1000, "direct"),
                                    _hrow("V_a", "t1", 9000, "sdk"),
                                    _hrow("V_a", "t2", 9000, "sdk")])
    assert more["V_b"]["advantage_total"] == 0.0 and more["V_a"]["advantage_total"] == 0.0
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


# ── #2390: the direct arm priced from the engine's own usage block ────────────────────

def _direct_trace(variant, task, *, prompt=6324, cached=4096, session=None):
    """A direct trace shaped as `bench_runner._run_one_sync` hands one over: the arm
    stamped by the runner itself, the counts folded by `add_usage`, and no recorded
    session — a bare `/v1/chat/completions` call writes no usage row anywhere (#1879).
    `session=` exists only to prove a pasted one is never joined."""
    trace = {"variant_id": variant, "task_id": task, "status": "success",
             "harness": "direct", "turns": 1, "tool_calls": [], "denied_calls": [],
             "duration_seconds": 1.0, "prompt_tokens": prompt, "cached_tokens": cached}
    if session is not None:
        trace["session_id"] = session
    return trace


def test_a_direct_trial_is_priced_from_the_engine_usage_block_on_its_trace(monkeypatch):
    """#2390 clause 2, through the real row writer.

    `prompt_tokens - cached_tokens` off the trace, and no store read: this arm writes no
    usage row, so a lookup here would attach some other trial's rows to this trial's
    price. The session id is pasted onto the trace deliberately to prove the route does
    not reach for it, and stays None on the row because no join produced the figure.
    """
    from scripts.autoresearch import run_round

    def _no_store(session):
        raise AssertionError(f"pricing a direct trial must not read usage.db ({session})")

    monkeypatch.setattr(cost, "usage_rows_for", _no_store)
    row = run_round.trial_ledger_row(
        "R_2390", _direct_trace("V_d", "t1", session="20261001_053224_bench_7a06"), _SCORE)
    assert row[C] == 2228 and type(row[C]) is int, "6324 - 4096, as an integer"
    assert row[cost.SESSION_FIELD] is None
    assert row["cached_tokens"] == 4096, (
        "the price and the discount it was taken at travel on the same row, so a later "
        "reader can tell a discounted figure from a raw one")


def test_a_direct_trial_is_priced_at_its_prompt_when_the_engine_reported_nothing_cached():
    """A measured zero discount is a real measurement, not a missing one.

    The engine at the bench endpoint reports `cached_tokens: 0` on a warm repeat of an
    identical prompt, so `prompt_tokens - 0` is this arm's ordinary figure — the raw
    prompt cost, which is a price. The distinction clause 2 turns on is between a zero
    the engine reported and a block it never sent, and only the fold in
    `bench_runner.add_usage` keeps them apart.
    """
    assert cost.trace_reprefill_cost(_direct_trace("V", "t", cached=0)) == 6324
    assert cost.trace_reprefill_cost(_direct_trace("V", "t", cached=None)) is None


def test_a_direct_trace_that_cannot_be_discounted_is_never_priced_at_its_prompt():
    """#2390 clause 2's "it never prices prompt_tokens alone".

    A vLLM without `--enable-prompt-tokens-details` sends `prompt_tokens` and nothing
    else, which is the shape of `cached=None` below. Charging the full prompt there would
    silently price every trial in every round at a figure that includes prefixes the
    engine may have served from cache — the exact error the currency was introduced to
    avoid — so a partial measurement reads None, the same rule `reprefill_cost` applies
    per usage row.
    """
    assert cost.trace_reprefill_cost(_direct_trace("V", "t", cached=None)) is None
    assert cost.trace_reprefill_cost(_direct_trace("V", "t", prompt=None)) is None, (
        "a cached count with no prompt to discount is no price either")
    assert cost.trace_reprefill_cost(_direct_trace("V", "t", prompt=None,
                                                   cached=None)) is None
    assert cost.trace_reprefill_cost({}) is None


def test_the_priced_set_is_the_one_tuple_and_the_two_causes_stay_told_apart(monkeypatch):
    """#2390 clause 3: both arms inside, both causes still distinguishable, one spelling.

    An unpriced direct trial and an unpriced sdk trial were distinguishable when only
    one arm had a route; growing the tuple must not blur them, because the two findings
    have different owners — a missing usage row is a hole in one round's accounting, an
    absent `prompt_tokens_details` is the endpoint's configuration and applies to every
    round until someone changes it.
    """
    from pathlib import Path

    assert set(cost.USAGE_BEARING_HARNESSES) == {"sdk", "direct"}

    rec = cost.round_cost_records([_hrow("V_direct", "t1", None, "direct"),
                                   _hrow("V_sdk", "t1", None, "sdk")])
    direct_reason, sdk_reason = rec["V_direct"]["cost_reason"], rec["V_sdk"]["cost_reason"]
    assert "engine reported no prompt_tokens_details" in direct_reason, direct_reason
    assert "no usage row" not in direct_reason, "the direct arm never writes one to miss"
    assert "no usage row for its recorded session" in sdk_reason, sdk_reason
    assert "prompt_tokens_details" not in sdk_reason, "nor is the store route the sdk hole"

    # The reason is read off the priced set, not spelled per arm: shrinking the tuple
    # re-labels the direct arm's finding with no other edit, which is what "no second
    # spelling" is a test of.
    monkeypatch.setattr(cost, "USAGE_BEARING_HARNESSES", ("sdk",))
    shrunk = cost.round_cost_records([_hrow("V_direct", "t1", None, "direct")])
    assert cost.NO_ROUTE_REASON in shrunk["V_direct"]["cost_reason"], shrunk["V_direct"]
    assert "prompt_tokens_details" not in shrunk["V_direct"]["cost_reason"]

    # And the mechanism behind it: no per-arm comparison anywhere in the module, and one
    # line only where the two constants define the priced set.
    src = Path(cost.__file__).read_text(encoding="utf-8")
    assert 'harness == "' not in src and "harness == '" not in src
    arm_lines = [ln.strip() for ln in src.splitlines()
                 if '"sdk"' in ln or '"direct"' in ln]
    assert arm_lines, "the priced set must name its arms somewhere"
    assert all(ln.startswith(("USAGE_BEARING_HARNESSES =", "TRACE_PRICED_ARMS ="))
               for ln in arm_lines), arm_lines


def test_the_advantage_group_is_one_task_on_one_arm_and_never_pools_the_arms():
    """#2390 clause 4. The two routes price different things, so they never share a mean.

    One task, both arms, two variants: the sdk group holds 1000 and 9000 (mean 5000), the
    direct group holds one trial of 8000. Pooling them would score every trial against
    6000 — a number that is neither arm's mean — and would hand V_a 3000 where its own
    arms' means give 4000 + 0. The direct trial's 0 is the honest reading: it is the only
    trial in its group, so it is a break-even against itself rather than against the
    other arm's prices.
    """
    rows = [_hrow("V_a", "t1", 1000, "sdk"), _hrow("V_b", "t1", 9000, "sdk"),
            _hrow("V_a", "t1", 8000, "direct")]
    rec = cost.round_cost_records(rows)
    assert rec["V_a"]["advantage_total"] == 4000.0, "5000-1000 on sdk, 8000-8000 on direct"
    assert rec["V_b"]["advantage_total"] == -4000.0, "5000-9000, its only trial's arm"
    assert rec["V_a"]["advantage_on"] == ["direct", "sdk"], "both groups contributed"
    assert rec["V_a"]["cost_mean_success"] == 4500.0, (
        "a variant's own mean always spanned both arms; only the comparison group splits")
    for v in rec.values():
        assert v["currency"] == cost.REPREFILL_FIELD


def test_rank_separates_on_advantage_only_where_two_variants_share_a_priced_arm():
    """#2390 clause 4's ranking half, in the direction the rule must still allow.

    Both variants have two successes, so the tie-break is the advantage column: `V_z`
    measured on both arms and `V_a` on the sdk arm only share `sdk`, so they may be
    separated, and the names are chosen so the advantage order is the reverse of the id
    order — falling through to the id would rank them the other way and fail here.
    """
    both = cost.round_cost_records([_hrow("V_a", "t1", 40000, "sdk"),
                                    _hrow("V_a", "t2", 40000, "sdk"),
                                    _hrow("V_z", "t1", 10000, "sdk"),
                                    _hrow("V_z", "t1", 30000, "direct")])
    assert set(both["V_z"]["advantage_on"]) == {"direct", "sdk"}
    assert set(both["V_a"]["advantage_on"]) == {"sdk"}
    assert both["V_z"]["advantage_total"] == 15000.0 > both["V_a"]["advantage_total"]
    assert cost.rank(both) == ["V_z", "V_a"], (
        "they share the sdk arm, so advantage decides even against the id order; the "
        "direct trial adds 0 of its own and never dilutes the sdk comparison")

    # The other half of the clause: two variants with no arm in common are not compared
    # on advantage at all. Split per arm each is its group's only trial, so both sit at a
    # break-even; pooled into one group they would stand 17,500 apart on the strength of
    # two currencies, and that phantom gap is what the guard exists to refuse.
    disjoint = cost.round_cost_records([_hrow("V_a", "t1", 40000, "sdk"),
                                        _hrow("V_x", "t1", 5000, "direct")])
    assert set(disjoint["V_a"]["advantage_on"]) == {"sdk"}
    assert set(disjoint["V_x"]["advantage_on"]) == {"direct"}
    assert disjoint["V_a"]["advantage_total"] == 0.0 == disjoint["V_x"]["advantage_total"]
    assert cost.rank(disjoint) == ["V_a", "V_x"], "successes tie and no arm is shared"
