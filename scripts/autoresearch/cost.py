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

    A direct-arm trace has no recorded session (one `/v1/chat/completions` call, no
    usage row), so both read None there — the same honest None `tool_search_enabled`
    carries on that arm.
    """
    sid = trace.get("session_id") if trace.get("harness") == "sdk" else None
    sid = str(sid) if sid else None
    cost = reprefill_cost((lookup or usage_rows_for)(sid)) if sid else None
    return {SESSION_FIELD: sid, REPREFILL_FIELD: cost}


def trial_outcome(row: dict[str, Any]) -> str:
    if row.get("trace_status") != "success":
        return INFRA
    score = row.get("objective_score")
    return SUCCESS if isinstance(score, (int, float)) and score >= 1.0 else FAILED


def round_cost_records(trial_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-variant cost record over one round's per-trial ledger rows.

    The advantage group is one task across every variant that ran it, so two variants
    benched under different cache profiles get different numbers; within a single
    variant the advantages of its own successes would always sum to zero.
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
            "unmeasured": 0, "_valid": [], "_success": [], "advantage_total": 0.0,
            "per_trial": []})
        outcome, cost = trial_outcome(r), r.get(REPREFILL_FIELD)
        cost = cost if _is_count(cost) else None
        rec["trials"] += 1
        advantage: float | None
        if outcome == INFRA:
            rec["infra_excluded"] += 1
            advantage = None
        else:
            if cost is None:
                rec["unmeasured"] += 1
            else:
                rec["_valid"].append(cost)
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
                    advantage = round(sum(group) / len(group) - cost, 2)
            rec["advantage_total"] += advantage or 0.0
        rec["per_trial"].append({"task_id": r.get("task_id"), "outcome": outcome,
                                 REPREFILL_FIELD: cost, "advantage": advantage})
    for rec in out.values():
        valid, success = rec.pop("_valid"), rec.pop("_success")
        rec["cost_total"] = sum(valid) if valid else None
        rec["cost_mean_valid"] = round(sum(valid) / len(valid), 2) if valid else None
        rec["cost_mean_success"] = round(sum(success) / len(success), 2) if success else None
        rec["advantage_total"] = round(rec["advantage_total"], 2)
    return out


def rank(records: dict[str, dict[str, Any]]) -> list[str]:
    """Variants, best first: successes, then the gated advantage, then id.

    Correctness orders before cost, and a failed trial's advantage is 0, so a variant
    that is cheaper and wrong can never sit above one that is right and expensive.
    """
    return sorted(records, key=lambda v: (-records[v]["successes"],
                                          -records[v]["advantage_total"], v))


def report_lines(records: dict[str, dict[str, Any]]) -> list[str]:
    """The round report's cost section. Emit-only, and it says so."""
    if not records:
        return []
    lines = ["", "## Cost (#2019, emit-only)",
             f"Currency: `{REPREFILL_FIELD}` = input_tokens - cache_read over each trial's "
             "recorded session. Advantage = mean(cost | success, same task) - cost for a "
             f"successful trial, 0 for a failed one; success is `{SUCCESS_CONDITION}`. "
             "No promotion leg reads this section.",
             "", "| rank | variant | successes | advantage | mean cost (success) | "
             "mean cost (valid) | unmeasured | infra excluded |",
             "|---|---|---|---|---|---|---|---|"]
    for n, vid in enumerate(rank(records), 1):
        r = records[vid]
        lines.append(f"| {n} | `{vid}` | {r['successes']}/{r['trials']} | "
                     f"{r['advantage_total']} | {_show(r['cost_mean_success'])} | "
                     f"{_show(r['cost_mean_valid'])} | {r['unmeasured']} | "
                     f"{r['infra_excluded']} |")
    return lines


def _show(value: Any) -> str:
    return "null" if value is None else str(value)


def decision_cost_fields(record: dict[str, Any] | None) -> dict[str, Any]:
    """The per-variant summary a decision row carries; `{}` when there is none."""
    if not record:
        return {}
    return {"cost": {k: record.get(k) for k in (
        "currency", "success_condition", "trials", "successes", "failures",
        "infra_excluded", "unmeasured", "cost_total", "cost_mean_valid",
        "cost_mean_success", "advantage_total")}}
