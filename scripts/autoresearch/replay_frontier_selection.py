#!/usr/bin/env python3
"""#595 — replay frontier (strict-dominance) selection over the autoresearch ledger.

What question this answers
--------------------------
The gate decides on a strict-win fraction over the targeted slice. A win fraction is
a fraction of tasks that MOVED, so on a bench whose tasks sit at the floor (0.000 on
both sides) or the ceiling (0.975/1.0 on both sides) it has a low ceiling by
construction, and it cannot see the shape that matters: a variant that improved
several tasks and regressed on NONE. This script counts exactly that shape —
variants the loop refused which strictly dominate their round's baseline — over the
whole ledger, from the per-trial rows alone.

Two things it deliberately does not do
--------------------------------------
* **It adds no compute.** Every number comes from the per-task `composite_score` and
  the per-task `safety_critical`/`safety_passed` already in `ledger.jsonl`, and the
  dominance condition is `promote.slice_metrics(...)["dominates"]` — the live
  predicate, imported, not a re-derivation. A replay with its own arithmetic would be
  measuring the replay.
* **It does not print a verdict over a corpus it never opened.** A missing or empty
  ledger raises instead of printing `0 dominating across 0 rounds`, because that line
  over a path that does not exist reads exactly like "the loop did nothing" — the same
  false-zero class as a grep against a directory that isn't there.

The census is keyed on the RECORDED refusals, because "refused" is what the loop did;
only dominance is recomputed. The safety veto is hard: a variant whose safety-critical
task's probe failed is NOT counted however it scores, so the census can never be read
as "the frontier would have promoted an unsafe prompt".

Leg attribution: the tie question is #1060's, not this one's
-----------------------------------------------------------
The printed attribution separates two remedies that overlap. Of the refusals whose
reason starts `insufficient_win_fraction`, how many had a whole-bench mean delta at or
above the loaded `min_composite_delta` — those are reachable by #1060's tie semantics
alone (counting a tie as a win), since they already cleared the delta floor and were
stopped only by the fraction. Of those, how many strictly dominate — only a frontier
condition can accept a variant whose gain came with a regression. A person should pick
one remedy, not both; these three numbers are what makes that a scope call rather than
a guess.

Reproducing a round
-------------------
`--round R_20260908_181458` re-derives that round's decision list from its own rows
and prints the refusal/promotion counts; `--expect-refusals 7 --expect-promotions 0`
turns the print into an assertion (exit 1 on a mismatch) so the reproduction is
checkable rather than remembered. `--through 2026-09-13T23:59:59Z` freezes the census
at a date: the totals move whenever the loop runs another round, and a published
figure has to stay answerable after the ledger grows.

Usage
-----
    .venvs/lloyd/bin/python -m scripts.autoresearch.replay_frontier_selection
    ... replay_frontier_selection --through 2026-09-13T23:59:59Z
    ... replay_frontier_selection --round R_20260908_181458 \
            --expect-refusals 7 --expect-promotions 0
    .venvs/lloyd/bin/python -m scripts.autoresearch.replay_frontier_selection --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.autoresearch import promote
from scripts.autoresearch.promote import REFUSAL_WIN_FRACTION as WIN_FRACTION_PREFIX
from scripts.autoresearch.common import AutoresearchConfig, load_config
from scripts.autoresearch.judge import aggregate_variant

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The path named by `--help` when nothing else holds a ledger.
DEFAULT_LEDGER = REPO_ROOT / "_pipeline" / "research" / "ledger.jsonl"
LEDGER_RELPATH = Path("_pipeline") / "research" / "ledger.jsonl"


def default_ledger() -> Path:
    """The real ledger, resolved lazily so importing this module touches no disk.

    `_pipeline/` is gitignored, so an automod worktree has no ledger and a census run
    from inside a round would resolve to a path that does not exist. Search order: the
    tree named by `LLOYD_HOME`, this module's own checkout, then `~/lloyd` — the first
    that actually holds the file. Lazy rather than module-level, because an
    import-time `exists()` is a filesystem probe inside every import of this module,
    including the gate's.
    """
    for root in (os.environ.get("LLOYD_HOME", ""), str(REPO_ROOT),
                 str(Path.home() / "lloyd")):
        if not root:
            continue
        candidate = Path(root).expanduser() / LEDGER_RELPATH
        if candidate.exists():
            return candidate
    return DEFAULT_LEDGER

#: The loop names its per-round baseline run `BASELINE` (`run_round.run()`), and a
#: round with none is skipped rather than dominated against nothing.
BASELINE_PREFIX = "BASE"

#: The census keys on the win-fraction refusal prefix the gate writes. Imported from
#: `promote`, never re-spelled here: the string lives with the decision that emits it,
#: so a census key cannot drift from the reason text and silently count nothing.

#: The cutoff the triage census (24 dominating refusals across 22 rounds, 551 of 552
#: win-fraction refusals clearing the delta floor) was taken under. Kept so the
#: recorded figure can be re-answered without re-deriving the date from prose.
TRIAGE_CUTOFF = "2026-09-13T23:59:59Z"

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every parseable row in `path`. A malformed line is skipped and counted by the
    caller's `skipped` tally in the header, never silently dropped: this file is
    appended to by six programs and an occasional torn line is normal."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


#: The per-trial ledger keys `aggregate_variant` reads off its score half. A ledger
#: row is a flattened (task, score) pair — `run_round.run()` writes it straight from
#: the same structures the aggregator consumed — so the census rehydrates them and
#: calls the real function instead of reimplementing its rankable exclusion and its
#: safety conjunction, which are exactly the semantics the veto depends on.
_SCORE_KEYS = ("composite_score", "objective_score", "rubric_overall",
               "safety_critical", "safety_passed", "rankable", "objective_excluded",
               "not_rankable_reason")


#: Trial keys the writer sometimes omits, with the value the live loop writes when it
#: does include them. `not_rankable` and `safety_veto_did_not_run` are derived from
#: these, so a row that leaves `rankable` out must not be read as unmeasured — that
#: would drop the task from the mean and quietly change the frontier's answer.
_SCORE_DEFAULTS: dict[str, Any] = {"rankable": True, "objective_excluded": [],
                                   "not_rankable_reason": None,
                                   "objective_check_count": 0,
                                   "objective_score": 0.0, "rubric_overall": None}


def _summary(rows: list[dict[str, Any]], variant_id: str) -> dict[str, Any]:
    """A per-variant summary through the same aggregator the live gate's input came
    from, so the veto and the rankable exclusion are the live ones, not a paraphrase.

    Rehydrating rather than rebuilding the summary by hand: the aggregator's own
    semantics — a not-rankable trial out of the mean, an unmeasured safety veto out of
    the conjunction — are precisely the semantics the frontier depends on, and a
    census that reimplemented them could report a dominating variant the gate would
    never have accepted."""
    pairs = [({"id": r.get("task_id"), "category": r.get("task_category", "unknown")},
              {**_SCORE_DEFAULTS, **{k: r[k] for k in _SCORE_KEYS if k in r}})
             for r in rows]
    return aggregate_variant(variant_id, pairs)


def _baseline_of(variants: dict[str, list[dict[str, Any]]]) -> tuple[str, dict] | None:
    """The round's baseline run, by the name the loop gives it — never "the variant
    with the highest mean", which would let the census pick the winner to compare
    winners against."""
    named = sorted(v for v in variants if v.startswith(BASELINE_PREFIX))
    if not named:
        return None
    pick = next((v for v in named if v.startswith("BASELINE")), named[0])
    return pick, _summary(variants[pick], pick)


def replay(ledger_path: Path, *, through: str | None = None,
           cfg: AutoresearchConfig | None = None) -> dict[str, Any]:
    """Recompute per-round refusal and dominance counts from `ledger_path`.

    Raises FileNotFoundError on a missing or empty ledger — see the module docstring.
    """
    if not ledger_path.exists():
        raise FileNotFoundError(
            f"no autoresearch ledger at {ledger_path} — the census would print "
            f"0 dominating across 0 rounds, which is not the same answer")
    rows = read_jsonl(ledger_path)
    if not rows:
        raise FileNotFoundError(
            f"autoresearch ledger {ledger_path} holds no rows — refusing to report "
            f"an empty census as a null result")

    trials = [r for r in rows if r.get("event") in (None, "") and r.get("task_id")]
    decisions = [r for r in rows if r.get("event") == "decision"]
    if not trials:
        raise FileNotFoundError(
            f"autoresearch ledger {ledger_path} has {len(rows)} rows and no per-trial "
            f"rows — nothing to compute dominance from")

    by_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in trials:
        by_variant[(str(r["round_id"]), str(r["variant_id"]))].append(r)

    rounds: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for (rid, vid), rr in by_variant.items():
        rounds[rid][vid] = rr

    dec_map: dict[tuple[str, str], dict[str, Any]] = {}
    for d in decisions:
        dec_map[(str(d["round_id"]), str(d["variant_id"]))] = d

    out_rounds: list[dict[str, Any]] = []
    skipped_no_baseline = 0
    undecidable = 0
    attribution = {"win_fraction_refusals": 0, "win_fraction_refusals_delta_clear": 0,
                   "win_fraction_refusals_dominating": 0, "dominating_refused": 0,
                   "dominating_by_other_reason": 0, "safety_vetoed_no_regression": 0}

    for rid in sorted(rounds):
        variants = rounds[rid]
        base = _baseline_of(variants)
        if base is None:
            skipped_no_baseline += 1
            continue
        base_vid, base_summ = base
        others = {vid: _summary(rows, vid)
                  for vid, rows in variants.items() if vid != base_vid}
        split = promote.derive_split(base_summ, *others.values())
        floor = (cfg.promotion_min_composite_delta if cfg else 0.05)

        refused: list[str] = []
        promotions: list[str] = []
        dominating: list[str] = []
        iwf_dominating = 0
        for vid in sorted(variants):
            if vid == base_vid:
                continue
            dec = dec_map.get((rid, vid))
            if dec is not None and through is not None and str(dec.get("created_at")) > through:
                continue        # frozen census: this decision is outside the window
            summ = others[vid]

            if dec is None:
                # Scored, never gated: the round died before its decision loop reached
                # this variant. It is NOT a refusal — nothing refused it — and
                # counting it as one would let the census claim credit for reclaiming
                # variants the selector never had a chance to keep: the 11 variants
                # this ledger holds without a decision row all sit in the 2026-09-04
                # to 09-06 window with truncated trial rows, and one of them would
                # otherwise have been reported as a dominating variant the loop threw
                # away. Counted in the header as `variants_without_decision_rows` and
                # left out of every classification below. A variant that reached the
                # gate and was accepted-but-not-landed is likewise not a refusal: the
                # gate did not refuse it.
                undecidable += 1
                if any(r.get("promoted") for r in variants[vid]):
                    promotions.append(vid)
                continue
            reason = str(dec.get("reason") or "")
            if dec.get("promoted"):
                promotions.append(vid)
                continue
            if dec.get("should_promote"):
                # Accepted and not landed (`roundup` mode, or the vault write
                # declined). Not a refusal — the gate did not refuse it.
                continue
            refused.append(vid)

            m = promote.slice_metrics(base_summ, summ, split)
            # Hard veto first: `dominates` is False whenever a safety-critical task's
            # probe failed, so a frontier — and this census — cannot count one.
            if m["dominates"]:
                dominating.append(vid)
                if reason.startswith(WIN_FRACTION_PREFIX):
                    iwf_dominating += 1
            if (m["better_ids"] and not m["worse_ids"] and m["safety_failed_ids"]
                    and not (dec or {}).get("promoted")):
                attribution["safety_vetoed_no_regression"] += 1
            if reason.startswith(WIN_FRACTION_PREFIX):
                attribution["win_fraction_refusals"] += 1
                mean_delta = ((sum(p["composite_score"] for p in summ["per_task"])
                               - sum(p["composite_score"] for p in base_summ["per_task"]))
                              / len(base_summ["per_task"]))
                if mean_delta >= floor:
                    attribution["win_fraction_refusals_delta_clear"] += 1

        attribution["dominating_by_other_reason"] += (
            len(dominating) - iwf_dominating)
        attribution["win_fraction_refusals_dominating"] += iwf_dominating
        out_rounds.append({
            "round_id": rid, "baseline_variant_id": base_vid,
            "variants": len(variants) - 1,
            "refused": len(refused), "promoted": len(promotions),
            "dominating_refused": len(dominating),
            "dominating_variant_ids": dominating,
            "promoted_variant_ids": promotions,
            "reason": "",
        })
        attribution["dominating_refused"] += len(dominating)

    return {
        "ledger": str(ledger_path),
        "through": through,
        "counts": {"rows": len(rows), "trial_rows": len(trials),
                   # Both, because a `--through` that matches no decision is a wrong
                   # argument and not a null result, and the only way to tell them
                   # apart is to print the denominator the window selected. Trial rows
                   # carry no timestamp of their own, so the window is a decision
                   # window: the round set cannot shrink, only the classifications.
                   "decision_rows": len(decisions),
                   "decision_rows_in_window": sum(
                       1 for d in decisions
                       if not (through and str(d.get("created_at") or "") > through))},
        "population": {"rounds": len(out_rounds),
                       "rounds_skipped_no_baseline": skipped_no_baseline,
                       "variants_without_decision_rows": undecidable},
        "thresholds": ({"min_composite_delta": cfg.promotion_min_composite_delta,
                        "min_bench_win_fraction": cfg.promotion_min_win_fraction,
                        "require_safety_pass": cfg.promotion_require_safety_pass}
                       if cfg else {}),
        "rounds": out_rounds,
        "attribution": attribution,
        "totals": {
            "refused": sum(r["refused"] for r in out_rounds),
            "promoted": sum(r["promoted"] for r in out_rounds),
            "dominating_refused": sum(r["dominating_refused"] for r in out_rounds),
            "rounds_with_dominating": sum(1 for r in out_rounds if r["dominating_refused"]),
        },
    }


def print_report(out: dict[str, Any]) -> None:
    c, pop, tot, attr = out["counts"], out["population"], out["totals"], out["attribution"]
    print(f"Ledger: {out['ledger']}")
    print(f"Rows: {c['rows']} read, {c['trial_rows']} per-trial, {c['decision_rows']} decision")
    thr = out["thresholds"]
    if thr:
        print(f"Thresholds (loaded, unchanged by this script): "
              f"min_composite_delta={thr['min_composite_delta']}, "
              f"min_bench_win_fraction={thr['min_bench_win_fraction']}, "
              f"require_safety_pass={thr['require_safety_pass']}")
    if out["through"]:
        print(f"Cutoff: decisions created after {out['through']} excluded")
    if out.get("through"):
        print(f"Decision rows inside the window: "
              f"{out['counts']['decision_rows_in_window']} of "
              f"{out['counts']['decision_rows']} in the file")
    print(f"Population: {pop['rounds']} rounds with a baseline run "
          f"({pop['rounds_skipped_no_baseline']} skipped for having none), "
          f"{pop['variants_without_decision_rows']} variants with no decision row "
          f"counted as refusals")
    print("")
    print("Per-round census — refused variants that strictly dominate that round's")
    print("baseline (better on at least one scored task, worse on none, every scored")
    print("task covered by both sides). A failed safety-critical probe is a HARD VETO:")
    print("such a variant is not counted, so no line here says a frontier would have")
    print("promoted an unsafe prompt.")
    print("Dominance, coverage and the veto are recomputed from per-task rows alone;")
    print("the refusal reason is taken as recorded, so this is a census of what the")
    print("selector threw away, not a re-derivation of why it refused. For a")
    print("reason-level recompute there is `replay_promotion_gate.py` — which")
    print("reproduces 1 of the 10 recorded reason strings it claims to, and is why")
    print("this script exists.")
    print(f"{'round_id':<22} {'cand':>5} {'refused':>8} {'promo':>6} {'dominating':>11}  dominating_variant_ids")
    for r in out["rounds"]:
        print(f"{r['round_id']:<22} {r['variants']:>5} {r['refused']:>8} "
              f"{r['promoted']:>6} {r['dominating_refused']:>11}  "
              f"{', '.join(r['dominating_variant_ids']) or '-'}")
    print(f"{'TOTAL':<22} {'':>5} {tot['refused']:>8} {tot['promoted']:>6} "
          f"{tot['dominating_refused']:>11}  in {tot['rounds_with_dominating']} of "
          f"{pop['rounds']} rounds")
    print("")
    print("Leg attribution — #1060's tie question separated from #595's frontier question")
    floor = (out["thresholds"].get("min_composite_delta", 0.05) if out["thresholds"] else 0.05)
    print(f"  refusals whose reason starts `{WIN_FRACTION_PREFIX}`: "
          f"{attr['win_fraction_refusals']}")
    print(f"  of those {attr['win_fraction_refusals']} refusals, cleared the "
          f"+{floor} mean-delta floor: "
          f"{attr['win_fraction_refusals_delta_clear']}   (#1060's tie semantics alone "
          f"could have accepted exactly these; the rest were refused on arithmetic the "
          f"floor had already blessed)")
    print(f"  strictly dominating among those refusals: {attr['win_fraction_refusals_dominating']}   "
          f"(nothing regressed on any scored task — a frontier is the only way to accept them)")
    print(f"  dominating-but-refused in total: {attr['dominating_refused']}   "
          f"(across every refusal reason, not only this leg)")
    print(f"  withheld by the safety veto despite no regression: "
          f"{attr['safety_vetoed_no_regression']}   "
          f"(scored clean, veto failed — never counted above)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("What question this answers")[0].strip())
    ap.add_argument("--ledger", type=Path, default=None,
                    help="ledger.jsonl to replay (default: the live checkout's, "
                         f"absent one: {DEFAULT_LEDGER})")
    ap.add_argument("--through", default=None,
                    help=f"exclude decisions created after this ISO date "
                         f"(triage used {TRIAGE_CUTOFF})")
    ap.add_argument("--round", dest="round_id", default=None,
                    help="print one round's recomputed decision counts and check them")
    ap.add_argument("--expect-refusals", type=int, default=None)
    ap.add_argument("--expect-promotions", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = ap.parse_args(argv)

    out = replay(args.ledger or default_ledger(), through=args.through,
                 cfg=load_config())
    if args.through and not out["counts"]["decision_rows_in_window"]:
        # A census over no decisions prints `0 dominating across 0 rounds`, which is
        # the shape of a null result and the shape of a wrong date. They are different
        # answers, and a script that prints the first for the second is how a
        # re-runnable number stops being re-runnable.
        print(f"MISMATCH: --through {args.through} selected 0 of "
              f"{out['counts']['decision_rows']} decision rows in {out['ledger']} — "
              f"the census would report 0 dominating across 0 rounds")
        return 1

    if args.round_id:
        match = next((r for r in out["rounds"] if r["round_id"] == args.round_id), None)
        if match is None:
            print(f"MISMATCH: {args.round_id} has no baseline run, or no scored "
                  f"variants under one, in {out['ledger']}")
            return 1
        print(f"{args.round_id}: baseline `{match['baseline_variant_id']}` — "
              f"{match['refused']} refused / {match['promoted']} promoted / "
              f"{match['dominating_refused']} dominating")
        target = f"round {args.round_id}"
        refused_now, promoted_now = match["refused"], match["promoted"]
    else:
        target = "this window"
        refused_now, promoted_now = out["totals"]["refused"], out["totals"]["promoted"]

    # The census is only worth its printed numbers if a wrong one fails the run. Same
    # discipline `replay_promotion_gate.py` reached after its headline drifted away
    # from its own script: the expectation is compared against what was RECORDED.
    bad: list[str] = []
    if args.expect_refusals is not None and refused_now != args.expect_refusals:
        bad.append(f"{target} refused {refused_now}, expected {args.expect_refusals}")
    if args.expect_promotions is not None and promoted_now != args.expect_promotions:
        bad.append(f"{target} promoted {promoted_now}, expected {args.expect_promotions}")
    if bad:
        print("MISMATCH: " + "; ".join(bad))
        return 1

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print_report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
