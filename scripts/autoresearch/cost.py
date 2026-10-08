"""Re-prefill cost per trial, and a success-gated cost advantage per variant (#2019).

The promotion gate has a quality term and no cost term, and every surface that needed
one worked around it by fiat (a compute-matched baseline, a matched token ceiling).
This module adds the currency and nothing else: it is EMIT-ONLY. Nothing here is read
by `promote.evaluate_promotion`, and `tests/test_autoresearch_promotion.py` pins that
the verdict is the same with these fields present or absent.

**The currency** is re-prefilled prompt tokens: the prompt tokens the engine had to
compute rather than read from cache. Where that comes from depends on the arm, and only
on the arm (`_route_of`): an agent turn's is its `input_tokens - cache_read` summed over
the usage rows of its recorded session, and a direct trial's is `prompt_tokens -
cached_tokens` off the engine's own usage block that `bench_runner.add_usage` folded onto
its trace (#2390). Not seconds (an estimate, and the reason #731's replay "settled
nothing"), not total tokens (which prices a cached prefix the engine never recomputed),
and not `usage.reprefill_tokens`, which read 0 on every bench row probed on 2026-10-01
including rows that re-prefilled 42.5k tokens. A count that is missing is None, never 0:
an unmeasured trial is not a free one, and neither route ever prices a prompt it could
not also discount — a figure with no cached count beside it is None, not `prompt_tokens`.

**The gate** is the CLM paper's: the efficiency advantage of a rollout is computed only
over the successful ones — `mean(cost | success) - cost` within the group of trials of
one task *on one arm* across the round's variants — and a failed rollout gets exactly 0.
Rewarding cheapness without that gate rewards deleting the context the task needed. An
infrastructure-failed trial (`trace_status != "success"`) is in neither the mean nor
the zeroing: a timeout is not a task failure. The group is per (task, harness) (#2390)
because the two routes price different things: an sdk figure discounts a prefix `usage.db`
saw served, a direct one discounts what the engine itself reported, and a mean over both
is the average of two currencies.

**The measured population (#2299, grown by #2390)** is narrower than the round, and says
so. Only a harness in `USAGE_BEARING_HARNESSES` has a route that can price it, so a trial
on any other arm is unmeasured by construction, and an arm whose route found nothing on
the trial is unmeasured by this round's accounting. Every per-variant block therefore
carries the harnesses that comprise its measured set beside its `unmeasured` count, plus a
`cost_reason` saying why an absent figure is absent — in the words of the route that
should have priced it, which is what keeps "no usage row arrived" and "the engine sent no
`prompt_tokens_details`" two different findings; `rank()` refuses to let `advantage_total`
separate two variants whose measured sets are harness-disjoint, because an unmeasured
trial's advantage contribution is a bare 0.0 and a break-even is indistinguishable from
it. Bounding the block to what it measured is emit-only work too: it labels the figure, it
never changes it.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from .bench_runner import CACHED_TOKENS_FIELD, HARNESS as DIRECT_HARNESS

#: Per-trial ledger keys. `recorded_session_id` is the join key: usage.db keys a bench
#: row by the recorded session (`20261001_053224_bench_7a06`), not by the trial id
#: (`bench_<variant>_<task>_<hex>`), so without it no usage row reaches its variant. It
#: is None on an arm priced off its own trace, because no store row was joined to
#: produce the figure that row carries — the field names the join, not a provenance.
SESSION_FIELD = "recorded_session_id"
REPREFILL_FIELD = "reprefill_cost_tokens"
COST_LEDGER_KEYS = (SESSION_FIELD, REPREFILL_FIELD)

#: What "successful" means for the gate, in the record's own words.
SUCCESS_CONDITION = "trace_status == 'success' and objective_score >= 1.0"
SUCCESS, FAILED, INFRA = "success", "failed", "infra_failed"

#: The arms whose trials can have a cost at all, and one of the two constants that
#: constitute the priced set (`TRACE_PRICED_ARMS` is the other): `_route_of` answers both
#: the pricing and the emitted reason from those two and nothing else, so a reason can
#: never disagree about which arms are priced and growing the set moves the reasons with
#: it. There is no per-arm comparison of the harness name anywhere in this module — prose
#: about a route may name the arm it prices, but only these two constants decide it. Both
#: arms the currency now prices are here: the agent turn, whose usage rows
#: `usage_rows_for` sums, and the direct bench trial, whose price is the engine's own
#: usage block off its trace (#2390, the other half of #2299). An arm outside the set has
#: no route at all and is unmeasurable by construction.
USAGE_BEARING_HARNESSES = ("sdk", "direct")

#: The arms priced from the engine usage block on their own trace rather than from
#: usage.db. Membership is not a second list of arms to maintain: it is the direct
#: runner's own name for its arm, because what makes that arm trace-priced is that its
#: runner is a bare `/v1/chat/completions` call which writes no usage row anywhere
#: (#1879). Every other arm in the tuple above is an agent turn whose harness writes
#: the rows the currency sums, so a newly added arm defaults to that route.
TRACE_PRICED_ARMS = (DIRECT_HARNESS,)

#: The two routes, named by what they read. `_route_of` picks one, `cost_ledger_fields`
#: prices through it and `_why_unmeasured` says in words what its absence means, so the
#: cause a reader is told is the cause the writer actually looked for.
USAGE_ROWS_ROUTE = "usage rows of its recorded session"
ENGINE_USAGE_ROUTE = "engine usage block on its own trace"

#: Why a trial priced by this route came home with nothing — kept distinct because the
#: two are different findings: a missing usage row is a hole in one round's accounting,
#: while an engine that never sends `prompt_tokens_details` leaves every trial on that
#: arm unpriced in every round until the endpoint is reconfigured.
UNPRICED_REASON = {
    USAGE_ROWS_ROUTE: "no usage row for its recorded session",
    ENGINE_USAGE_ROUTE: "engine reported no prompt_tokens_details, so its trace "
                        "carries no prompt count to price",
}
NO_ROUTE_REASON = "no pricing route is registered for this arm, so its trials cannot be priced"

#: What a trial row with no `harness` key says it ran on. Both ledger writers always
#: stamp one (`bench_runner._run_one_sync` stamps its own `HARNESS` onto the trace, and
#: `run_round.trial_ledger_row` still defaults a hand-built one to the same name), so
#: this label can only ever appear on a hand-built row — which is the point: an unnamed
#: arm must not silently stand for a population nobody recorded.
UNKNOWN_HARNESS = "unrecorded"


def _route_of(harness: str) -> str | None:
    """The route that prices one arm, or None for an arm nothing can price.

    One function answers it, for the writer and for the reason beside its None. Read
    through `USAGE_BEARING_HARNESSES` and `TRACE_PRICED_ARMS` rather than by a per-arm
    `if`, which is how #2299's two arms became two spellings of one test.
    """
    if harness not in USAGE_BEARING_HARNESSES:
        return None
    return ENGINE_USAGE_ROUTE if harness in TRACE_PRICED_ARMS else USAGE_ROWS_ROUTE


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


def trace_reprefill_cost(trace: dict[str, Any]) -> int | None:
    """`prompt_tokens - cached_tokens` off the trial's own trace, or None (#2390).

    Both counts come from one engine usage block, and both are required: a price from
    `prompt_tokens` alone would charge the trial for a prefix the engine served from
    cache, which is the one thing this currency exists not to do, and it is why a trace
    whose engine sent `prompt_tokens` but no `prompt_tokens_details` (the shape of a
    vLLM without `--enable-prompt-tokens-details`) reads None rather than its raw prompt
    count. The same rule `reprefill_cost` applies per row: a partial measurement is a
    missing one, never a cheaper one.

    `cached_tokens` is the count `bench_runner.add_usage` summed over the trial's calls,
    so on a multi-call trial this discounts every cached prefix it got, not one.
    """
    prompt, cached = trace.get("prompt_tokens"), trace.get(CACHED_TOKENS_FIELD)
    if not _is_count(prompt) or not _is_count(cached):
        return None
    return prompt - cached


def _price_via_usage_rows(trace: dict[str, Any],
                          lookup: Callable[[str], list[dict[str, Any]]],
                          ) -> tuple[str | None, int | None]:
    """(recorded session, cost summed over its usage rows): the agent-turn route."""
    sid = trace.get("session_id")
    sid = str(sid) if sid else None
    return sid, (reprefill_cost(lookup(sid)) if sid else None)


def _price_via_engine_usage(trace: dict[str, Any],
                            lookup: Callable[[str], list[dict[str, Any]]],
                            ) -> tuple[str | None, int | None]:
    """(None, cost off the engine usage block on the trace): the direct route (#2390).

    Never calls `lookup`, and so never reads a usage row: this arm's trial wrote none,
    and a `session_id` pasted onto one of these traces (a direct trial holds no session
    at any level of its chain, #1879) would join to rows some other trial wrote. Its
    `recorded_session_id` therefore stays None even when a price was found — that field
    names a join, and this figure came no such way.
    """
    return None, trace_reprefill_cost(trace)


#: One pricer per route, so the sentence in `UNPRICED_REASON` and the arithmetic here
#: are two readings of the same route and cannot drift apart.
_PRICERS = {USAGE_ROWS_ROUTE: _price_via_usage_rows,
            ENGINE_USAGE_ROUTE: _price_via_engine_usage}


def cost_ledger_fields(trace: dict[str, Any],
                       lookup: Callable[[str], list[dict[str, Any]]] | None = None
                       ) -> dict[str, Any]:
    """The two per-trial keys, for both ledger writers.

    A trace on an arm outside `USAGE_BEARING_HARNESSES` has no route at all, so both
    keys read None — the same honest None `tool_search_enabled` carries there. Inside
    it, `_route_of` picks the one route that arm has: sum the usage rows of its recorded
    session, or read the engine usage block the runner folded onto the trace itself.

    This is the writer half of the measured population #2299 names: the None it writes
    here is what `round_cost_records` later reports as an unmeasured trial on that
    harness, and the reason it reports is the same route's own sentence, so writer and
    reason read one mechanism and neither spells an arm out.

    The arm is read off the trace and never guessed: both runners stamp their own name
    there (`bench_runner.HARNESS`, `bench_runner_sdk.HARNESS`), so a trace from a real
    trial always says which route prices it, and a hand-built trace that names none is
    unpriced rather than priced by whichever route the reader expected.
    """
    harness = trace.get("harness")
    price = _PRICERS.get(_route_of(str(harness))) if harness else None
    if price is None:
        return {SESSION_FIELD: None, REPREFILL_FIELD: None}
    sid, cost = price(trace, lookup or usage_rows_for)
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

    The cause is the sentence of the route that should have priced the arm — read
    through `_route_of`, the same lookup the writer priced by — because the ways a trial
    goes unmeasured are different facts worth telling apart: an agent-turn trial with no
    price is missing a usage row this round's accounting should have written, a direct
    trial is missing the engine's own `prompt_tokens_details`, and a trial on an arm with
    no route was never going to be priced anywhere. Before #2399 the first sentence was
    the only one any arm could get, which is what made an unpriced direct arm read as a
    hole in the round rather than as the arm the currency did not reach.
    """
    route = _route_of(harness)
    return NO_ROUTE_REASON if route is None else UNPRICED_REASON[route]


def _cost_reason(trials: int, measured: int, unmeasured: int, infra_excluded: int,
                 unmeasured_by_harness: dict[str, int]) -> str | None:
    """Why this variant's cost figure is absent or partial, naming the population.

    None only when every trial that could be priced was priced. Otherwise the reason is
    emitted because a None cost is read as "cost nothing" by the next reader — the exact
    inversion this block exists to prevent, and the one a ledger with an unpriced arm
    makes easy, where one priced trial of twenty stands in for the whole variant. Each
    arm in the list gets its own clause, in the words of the route that failed it
    (`_why_unmeasured`), so an arm the currency still cannot reach and an arm whose
    engine sent no usage block are never reported as the same finding.
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

    The advantage group is one task on one arm across every variant that ran it, so two
    variants benched under different cache profiles get different numbers; within a
    single variant the advantages of its own successes would always sum to zero.

    The arm is in the key because of #2390, and not before it: now that both arms carry a
    price, a group keyed on the task alone would average an sdk figure discounted by
    `usage.db`'s `cache_read` with a direct figure discounted by the engine's own
    `cached_tokens` — two measurements of two different things standing in one mean, and
    every trial's advantage in that round measured against a number that is neither
    arm's. `_advantage_basis`/`_cost_slot` then stop the ranking comparing across the
    arms; this stops the arithmetic from mixing them in the first place.

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
            success_costs.setdefault((r.get("task_id"), _harness_of(r)), []) \
                .append(r[REPREFILL_FIELD])
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
                group = success_costs.get((r.get("task_id"), harness)) or []
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
             f"Currency: `{REPREFILL_FIELD}` = the prompt tokens the engine had to "
             "re-compute: `input_tokens - cache_read` summed over the usage rows of a "
             "turn's recorded session, or `prompt_tokens - cached_tokens` off the "
             "engine's own usage block on a direct trial's trace (#2390). Advantage = "
             "mean(cost | success, same task on the same arm) - cost for a "
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
