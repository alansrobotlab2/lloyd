"""Re-prefill cost per trial, and a success-gated cost advantage per variant (#2019).

The promotion gate has a quality term and no cost term, and every surface that needed
one worked around it by fiat (a compute-matched baseline, a matched token ceiling).
This module adds the currency and nothing else: it is EMIT-ONLY. Nothing here is read
by `promote.evaluate_promotion`, and `tests/test_autoresearch_promotion.py` pins that
the verdict is the same with these fields present or absent.

**The currency** is re-prefilled prompt tokens: a trial's `input_tokens - cache_read`,
summed over the usage rows of its recorded session. Not seconds (an estimate, and the
reason #731's replay "settled nothing"), not total tokens (which prices a cached prefix
the engine never recomputed), and not `usage.reprefill_tokens`, which read 0 on every
bench row probed on 2026-10-01 including rows that re-prefilled 42.5k tokens. A count
that is missing is None, never 0: an unmeasured trial is not a free one.

**The gate** is the CLM paper's: the efficiency advantage of a rollout is computed only
over the successful ones — `mean(cost | success) - cost` within the group of trials of
one task across the round's variants — and a failed rollout gets exactly 0. Rewarding
cheapness without that gate rewards deleting the context the task needed. An
infrastructure-failed trial (`trace_status != "success"`) is in neither the mean nor
the zeroing: a timeout is not a task failure.

**The measured population (#2299)** is narrower than the round, and says so. Only a
harness in `USAGE_ROW_HARNESSES` writes the usage rows this currency is summed from, so
a trial on any other harness is unmeasured by construction, and a round that benches
one arm on each has a cost figure for exactly one of them. Every per-variant block
therefore carries the harnesses that comprise its measured set beside its `unmeasured`
count, plus a `cost_reason` saying why an absent figure is absent; `rank()` refuses to
let `advantage_total` separate two variants whose measured sets are harness-disjoint,
because an unmeasured trial's advantage contribution is a bare 0.0 and a break-even is
indistinguishable from it. Bounding the block to what it measured is emit-only work
too: it labels the figure, it never changes it.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

#: Per-trial ledger keys. `recorded_session_id` is the join key: usage.db keys a bench
#: row by the recorded session (`20261001_053224_bench_7a06`), not by the trial id
#: (`bench_<variant>_<task>_<hex>`), so without it no usage row reaches its variant.
SESSION_FIELD = "recorded_session_id"
REPREFILL_FIELD = "reprefill_cost_tokens"
COST_LEDGER_KEYS = (SESSION_FIELD, REPREFILL_FIELD)

#: What "successful" means for the gate, in the record's own words.
SUCCESS_CONDITION = "trace_status == 'success' and objective_score >= 1.0"
SUCCESS, FAILED, INFRA = "success", "failed", "infra_failed"

#: The harnesses whose trials can have a cost at all: only one whose trace carries a
#: recorded session has usage rows for `usage_rows_for` to sum, and `cost_ledger_fields`
#: resolves a session id on exactly those — so every other arm is unmeasurable by
#: construction, and the measured population of a round that benches both is the arm
#: listed here. When the direct arm ever gets engine-usage capture (the other half of
#: #2299), this tuple is the single place that grows, and the emitted reasons follow it.
USAGE_BEARING_HARNESSES = ("sdk",)

#: What a trial row with no `harness` key says it ran on. Both ledger writers always
#: stamp one (`run_round.trial_ledger_row` defaults to `direct`), so this label can only
#: ever appear on a hand-built row — which is the point: an unnamed arm must not silently
#: stand for a population nobody recorded.
UNKNOWN_HARNESS = "unrecorded"


def reprefill_cost(usage_rows: Iterable[dict[str, Any]] | None) -> int | None:
    """`input_tokens - cache_read` summed over a trial's usage rows, or None.

    None when there are no rows, or when ANY row is missing either count — a partial
    sum would understate the cost of exactly the trials whose accounting broke.
    """
    rows = list(usage_rows or [])
    if not rows:
        return None
    total = 0
    for row in rows:
        inp, cached = row.get("input_tokens"), row.get("cache_read")
        if not _is_count(inp) or not _is_count(cached):
            return None
        total += inp - cached
    return total


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def usage_rows_for(session_id: str) -> list[dict[str, Any]]:
    """The usage rows one recorded session wrote. Read-only; never raises."""
    try:
        from app import usage_store
        return usage_store.read_session_rows_readonly(
            usage_store.DB_PATH, session_id, ["input_tokens", "cache_read"])
    except Exception:  # noqa: BLE001 — a cost that cannot be read is None, not a crash
        return []


def cost_ledger_fields(trace: dict[str, Any],
                       lookup: Callable[[str], list[dict[str, Any]]] | None = None
                       ) -> dict[str, Any]:
    """The two per-trial keys, for both ledger writers.

    A trace on a harness outside `USAGE_BEARING_HARNESSES` has no recorded session (the
    direct arm is one `/v1/chat/completions` call, no usage row), so both read None
    there — the same honest None `tool_search_enabled` carries on that arm. This is the
    writer half of the measured population #2299 names: the None it writes here is what
    `round_cost_records` later reports as an unmeasured trial on that harness, so the
    two must read one constant, not two spellings of `"sdk"`.
    """
    sid = (trace.get("session_id") if trace.get("harness") in USAGE_BEARING_HARNESSES
           else None)
    sid = str(sid) if sid else None
    cost = reprefill_cost((lookup or usage_rows_for)(sid)) if sid else None
    return {SESSION_FIELD: sid, REPREFILL_FIELD: cost}


def trial_outcome(row: dict[str, Any]) -> str:
    if row.get("trace_status") != "success":
        return INFRA
    score = row.get("objective_score")
    return SUCCESS if isinstance(score, (int, float)) and score >= 1.0 else FAILED


def _harness_of(row: dict[str, Any]) -> str:
    """The arm one trial row says it ran on, in the form the cost block names it."""
    harness = row.get("harness")
    return str(harness) if harness else UNKNOWN_HARNESS


def _why_unmeasured(harness: str) -> str:
    """Why a trial on this harness carries no cost, in the mechanism and not the count.

    The two ways a trial goes unmeasured are different facts: an arm outside
    `USAGE_BEARING_HARNESSES` never writes a usage row, so its trials are unpriced in
    every round forever; one on an arm that does write them but arrives without a row is
    a hole in that round's accounting, and the kind of hole worth reading about.
    """
    if harness in USAGE_BEARING_HARNESSES:
        return "no usage row for its recorded session"
    return "the harness records no usage row, so its trials cannot be priced"


def _cost_reason(trials: int, measured: int, unmeasured: int, infra_excluded: int,
                 unmeasured_by_harness: dict[str, int]) -> str | None:
    """Why this variant's cost figure is absent or partial, naming the population.

    None only when every trial that could be priced was priced. Otherwise the reason is
    emitted because a None cost is read as "cost nothing" by the next reader — the exact
    inversion this block exists to prevent, and the one the 2.9%-measured live ledger
    makes easy: with `direct` unpriced by construction, a variant benched on both arms
    has one sdk number standing in for twenty-something trials.
    """
    if measured and not unmeasured:
        return None
    parts = [f"{n} unmeasured trial(s) on {h}: {_why_unmeasured(h)}"
             for h, n in sorted(unmeasured_by_harness.items())]
    if infra_excluded:
        parts.append(f"{infra_excluded} infrastructure-failed trial(s) in neither the "
                     "mean nor the zeroing")
    head = (f"cost priced for {measured} of {trials} trial(s)" if measured else
            f"no trial was priced out of {trials}, so the cost fields are None and "
            "never 0, and advantage_total is the empty sum rather than a measured "
            "break-even")
    return head + (": " + "; ".join(parts) if parts else "")


def round_cost_records(trial_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-variant cost record over one round's per-trial ledger rows.

    The advantage group is one task across every variant that ran it, so two variants
    benched under different cache profiles get different numbers; within a single
    variant the advantages of its own successes would always sum to zero.

    #2299 bounds every figure to the population that produced it: `measured_on` names the
    harnesses the cost means are summed over, `advantage_on` the harnesses whose *priced
    successes* `advantage_total` actually compares, and `cost_reason` says in words why an
    absent figure is absent. `advantage_on` is not redundant with `measured_on` — a
    variant whose only priced trial failed has an `advantage_total` of exactly 0.0 from
    the zeroing, and one whose trials were never priced has the same number from the
    empty sum, which is precisely the indistinguishable pair `rank` refuses to sort on.
    """
    rows = [r for r in trial_rows if r.get("variant_id") is not None]
    success_costs: dict[Any, list[int]] = {}
    for r in rows:
        if trial_outcome(r) == SUCCESS and _is_count(r.get(REPREFILL_FIELD)):
            success_costs.setdefault(r.get("task_id"), []).append(r[REPREFILL_FIELD])
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        rec = out.setdefault(str(r["variant_id"]), {
            "success_condition": SUCCESS_CONDITION, "currency": REPREFILL_FIELD,
            "trials": 0, "successes": 0, "failures": 0, "infra_excluded": 0,
            "measured": 0, "unmeasured": 0, "measured_on": [], "advantage_on": [],
            "cost_reason": None, "advantage_total": 0.0,
            "per_trial": [], "_valid": [], "_success": [], "_measured_h": set(),
            "_advantage_h": set(), "_unmeasured_h": {}})
        outcome, cost = trial_outcome(r), r.get(REPREFILL_FIELD)
        harness = _harness_of(r)
        cost = cost if _is_count(cost) else None
        rec["trials"] += 1
        advantage: float | None
        if outcome == INFRA:
            rec["infra_excluded"] += 1
            advantage = None
        else:
            if cost is None:
                rec["unmeasured"] += 1
                counts = rec["_unmeasured_h"]
                counts[harness] = counts.get(harness, 0) + 1
            else:
                rec["measured"] += 1
                rec["_valid"].append(cost)
                rec["_measured_h"].add(harness)
            if outcome == FAILED:
                rec["failures"] += 1
                advantage = 0.0
            else:
                rec["successes"] += 1
                group = success_costs.get(r.get("task_id")) or []
                if cost is None or not group:
                    advantage = None
                else:
                    rec["_success"].append(cost)
                    rec["_advantage_h"].add(harness)
                    advantage = round(sum(group) / len(group) - cost, 2)
            rec["advantage_total"] += advantage or 0.0
        rec["per_trial"].append({"task_id": r.get("task_id"), "outcome": outcome,
                                 REPREFILL_FIELD: cost, "advantage": advantage})
    for rec in out.values():
        valid, success = rec.pop("_valid"), rec.pop("_success")
        unmeasured_by_harness = rec.pop("_unmeasured_h")
        rec["measured_on"] = sorted(rec.pop("_measured_h"))
        rec["advantage_on"] = sorted(rec.pop("_advantage_h"))
        rec["cost_reason"] = _cost_reason(rec["trials"], rec["measured"],
                                          rec["unmeasured"], rec["infra_excluded"],
                                          unmeasured_by_harness)
        rec["cost_total"] = sum(valid) if valid else None
        rec["cost_mean_valid"] = round(sum(valid) / len(valid), 2) if valid else None
        rec["cost_mean_success"] = round(sum(success) / len(success), 2) if success else None
        rec["advantage_total"] = round(rec["advantage_total"], 2)
    return out


def _advantage_basis(records: dict[str, dict[str, Any]]) -> frozenset[str]:
    """The harnesses every priced advantage in the round shares, or none.

    #2299: `advantage_total` is a comparison, and a comparison needs one currency. Where
    no harness is common to all priced variants — an sdk-priced set against a
    direct-priced one, which is the disjoint case — there is no such currency, and the
    empty set says so rather than picking whichever arm happens to sort first by name.
    """
    priced = [frozenset(r.get("advantage_on") or ()) for r in records.values()
              if r.get("advantage_on")]
    return frozenset.intersection(*priced) if priced else frozenset()


def _cost_slot(record: dict[str, Any],
               basis: frozenset[str]) -> tuple[int, float]:
    """Where this variant sits in the cost leg of the ranking, and the number it may use.

    A variant whose advantage was priced on the round's shared currency is ranked by it.
    One whose advantage was not — nothing priced at all, or priced on an arm disjoint
    from the basis — is placed below those that were, at equal successes, with its own
    `advantage_total` never read: that number is an empty sum or a single-arm artifact,
    and letting it float the unmeasured variant above the measured-and-found-dear one is
    the misordering #2299 was filed on. Nothing above this slot can be reached by a cost
    figure, so successes still decide first and the id breaks an unpriced tie.
    """
    if basis and basis <= frozenset(record.get("advantage_on") or ()):
        return (0, -record["advantage_total"])
    return (1, 0.0)


def rank(records: dict[str, dict[str, Any]]) -> list[str]:
    """Variants, best first: successes, then the gated advantage, then id.

    Correctness orders before cost, and a failed trial's advantage is 0, so a variant
    that is cheaper and wrong can never sit above one that is right and expensive.

    #2299 bounds the cost leg to one currency: `advantage_total` separates two variants
    only where their measured sets share a harness (`_advantage_basis`). Two variants
    whose measured sets are disjoint by harness are never separated by their advantage
    values — the ranking falls to successes, and below that to the id — because an
    unmeasured trial contributes a bare 0.0 to that sum, which is the same number a
    measured break-even writes.
    """
    basis = _advantage_basis(records)
    return sorted(records, key=lambda v: (-records[v]["successes"],
                                          _cost_slot(records[v], basis), v))


def _population_line(records: dict[str, dict[str, Any]]) -> str:
    """The round report's sentence on what its mean-cost columns are an average *of*.

    Without it the table reads as all-trial cost, which over the live ledger is a figure
    for under three per cent of the trials — every one of them on the single harness that
    writes usage rows.
    """
    trials = sum(int(r.get("trials", 0)) for r in records.values())
    priced = sum(int(r.get("measured", 0)) for r in records.values())
    on = sorted({h for r in records.values() for h in (r.get("measured_on") or ())})
    where = "/".join(f"`{h}`" for h in on) if on else "no harness"
    line = (f"Measured population: {priced} of {trials} trial(s) carry a cost, on "
            f"{where}. The mean-cost columns are that population's, not every trial's — "
            "the `measured on` cell is the arm that row's figure was priced on, and "
            "`unmeasured` counts the trials missing from it.")
    if not _advantage_basis(records):
        line += (" No advantage comparison spans this round: no harness is common to "
                 "every priced variant, so the ranking below orders by successes and then "
                 "id, never by `advantage_total`.")
    return line


def report_lines(records: dict[str, dict[str, Any]]) -> list[str]:
    """The round report's cost section. Emit-only, and it says so.

    #2299: the section opens by naming the measured harness population, and each row
    carries the arm its own cost figure came from, so neither the header nor a single
    variant's mean can be read as a whole-trial cost.
    """
    if not records:
        return []
    lines = ["", "## Cost (#2019, emit-only)",
             f"Currency: `{REPREFILL_FIELD}` = input_tokens - cache_read over each trial's "
             "recorded session. Advantage = mean(cost | success, same task) - cost for a "
             f"successful trial, 0 for a failed one; success is `{SUCCESS_CONDITION}`. "
             "No promotion leg reads this section.",
             _population_line(records),
             "", "| rank | variant | successes | advantage | mean cost (success) | "
             "mean cost (valid) | measured on | unmeasured | infra excluded |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, vid in enumerate(rank(records), 1):
        r = records[vid]
        lines.append(f"| {n} | `{vid}` | {r['successes']}/{r['trials']} | "
                     f"{r['advantage_total']} | {_show(r['cost_mean_success'])} | "
                     f"{_show(r['cost_mean_valid'])} | {_measured_on_cell(r)} | "
                     f"{r['unmeasured']} | {r['infra_excluded']} |")
    return lines


def _measured_on_cell(record: dict[str, Any]) -> str:
    """The row's own measured arm, or the words that say there is none."""
    on = record.get("measured_on") or []
    return "/".join(f"`{h}`" for h in on) if on else "*none*"


def _show(value: Any) -> str:
    return "null" if value is None else str(value)


def decision_cost_fields(record: dict[str, Any] | None) -> dict[str, Any]:
    """The per-variant summary a decision row carries; `{}` when there is none.

    #2299: `measured_on`, `advantage_on` and `cost_reason` ride beside `unmeasured`
    because this block is the only copy a reader months later has. Without them a
    decision row can carry `cost_mean_success: 42861.0` over 23 trials of which one was
    priced, and nothing on the row says which harness that one trial ran on or why the
    other 22 have no figure — so the number reads as the variant's cost position and is
    not one.
    """
    if not record:
        return {}
    return {"cost": {k: record.get(k) for k in (
        "currency", "success_condition", "trials", "successes", "failures",
        "infra_excluded", "measured", "unmeasured", "measured_on", "advantage_on",
        "cost_reason", "cost_total", "cost_mean_valid", "cost_mean_success",
        "advantage_total")}}
