"""Replay every recorded autoresearch promotion under the #549 gate. Zero compute:
the ledger already carries one row per (round, variant, task) with `task_category`,
`composite_score`, `safety_critical` and `safety_passed`, so the two-condition rule
can be run retrospectively over decisions that already happened.

    python -m scripts.autoresearch.replay_promotion_gate            # whole corpus
    python -m scripts.autoresearch.replay_promotion_gate --by-task  # which tasks vetoed

Prints the flip count, and beside every raw delta the headroom and normalized gain
HarnessOpt-Bench insists on — `+0.05` over a 0.50 seed captures 10% of what was
available, over a 0.85 seed 33%, and a decision that reports only the sign of the
delta cannot tell those apart. Exits non-zero if the named 2026-09-08 variant, the
one that landed a contract at 64% gate stack / 28% prohibition lines, is NOT
refused by the replayed gate.

Two things this cannot claim, and says so in its own output: a round whose baseline
rows are missing is skipped, not counted as a refusal; and the rotation is re-derived
from the round id rather than read from a `bench_split.json` that was never written
for those rounds, so the slices here are the rule as it would run, not the rule as
it ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import bench_split, promote
from .common import AutoresearchConfig, load_config

# The variant #549 names: "add negative constraints to fix the shape benchmarks",
# promoted 2026-09-08T16:57:08Z. Identified by measurement, not by the item's
# prose — its SOUL.md overlay measures 64.36% gate stack / 27.96% prohibition
# lines, which is the 64%/28% the item and MEMORY.md record.
NAMED_VARIANT = "V_20260908_165359_f45720"


def _load_rows(ledger_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        rows.append(entry)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a `judge.aggregate_variant`-shaped summary from per-task ledger rows.

    A row carrying `rubric_excluded: true` is out of `mean_composite` and out of
    `per_task`, exactly as `judge.aggregate_variant` leaves that trial out (#646).
    The replay's entire claim is that it re-decides the evidence the round decided
    on: such a row's `composite_score` is arithmetic on a number the judge never
    gave, so folding it back in re-adds the phantom 0.5 the round excluded, and a
    replay of a post-#646 round could then reach a different verdict from the one
    that actually promoted. Rows written before #646 carry no such key, so the
    historical corpus — the flip count and the named-variant refusal — is
    bit-identical under this rule.

    The safety conjunction still covers excluded rows. Disarming the veto on a
    rubric outage is the failure mode `aggregate_variant` exists to remove, and the
    replay must not become the one place that reintroduces it: a critical row whose
    probe failed fails `safety_passed` whether or not its own trial was scored.
    """
    per: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    safety = True
    for r in rows:
        tid = r.get("task_id")
        score = r.get("composite_score")
        if not tid or not isinstance(score, (int, float)):
            continue
        if r.get("safety_critical") and r.get("safety_passed") is False:
            safety = False
        if r.get("rubric_excluded"):
            excluded.append(str(tid))
            continue
        per[str(tid)] = {"task_id": str(tid), "composite_score": float(score),
                         "category": str(r.get("task_category") or "unknown")}
    scores = [p["composite_score"] for p in per.values()]
    return {"mean_composite": (sum(scores) / len(scores)) if scores else 0.0,
            "safety_passed": safety, "task_count": len(per) + len(excluded),
            "rubric_excluded": len(excluded),
            "rubric_excluded_tasks": sorted(excluded),
            "per_task": sorted(per.values(), key=lambda p: p["task_id"])}


def _contract_refusals(variant_dir: Path) -> list[str]:
    """The same guard `promote` runs, against the overlay as it was stored."""
    soul = variant_dir / "SOUL.md"
    if not soul.exists():
        return []
    try:
        import prompt_surface
    except ImportError:
        return []
    def _text(name: str) -> str | None:
        p = variant_dir / name
        return p.read_text(encoding="utf-8") if p.exists() else None

    # Three surfaces because `promote` now checks three, and this function's entire
    # claim is that it runs *that* guard. A replay still reading two would print a
    # historical verdict computed under a guard that no longer exists and present the
    # difference as a finding — the reporting/enforcement drift #1069 is the record
    # of, in the opposite direction.
    return list(prompt_surface.check_contract(
        soul.read_text(encoding="utf-8"),
        _text("MEMORY.md"),
        _text("USER.md")))


def replay(cfg: AutoresearchConfig, ledger_path: Path,
           variants_dir: Path) -> list[dict[str, Any]]:
    rows = _load_rows(ledger_path)
    by_round: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        rid, vid = r.get("round_id"), r.get("variant_id")
        if rid and vid:
            by_round[rid][str(vid)].append(r)

    decisions: list[dict[str, Any]] = []
    for rid, variants in sorted(by_round.items()):
        promoted = sorted({str(r.get("variant_id")) for r in rows
                           if r.get("round_id") == rid and r.get("promoted") is True})
        if not promoted:
            continue
        baseline_ids = [v for v in variants if v.startswith("BASE")]
        if not baseline_ids:
            decisions.append({"round_id": rid, "variant_id": promoted[0],
                              "skipped": "no baseline rows in the ledger"})
            continue
        base = _summary(variants[baseline_ids[0]])
        for vid in promoted:
            var = _summary(variants.get(vid, []))
            if not var["per_task"]:
                decisions.append({"round_id": rid, "variant_id": vid,
                                  "skipped": "no per-task rows for the promoted variant"})
                continue
            split = bench_split.compute_split(
                [{"id": p["task_id"], "category": p["category"]} for p in base["per_task"]],
                rid)
            should, reason = promote.evaluate_promotion(cfg, base, var, split=split)
            m = promote.slice_metrics(base, var, split)
            refusals = _contract_refusals(variants_dir / vid)
            # `promote()` applies the contract guard after the score gate, so the
            # replayed decision is the whole accept path, not one predicate.
            refused_by = ("score_gate" if not should
                          else ("contract_guard" if refusals else "none"))
            delta = m["targeted_variant"] - m["targeted_baseline"]
            decisions.append({
                "round_id": rid, "variant_id": vid, "skipped": None,
                "old_reason": str(next((r.get("reason") for r in rows
                                        if r.get("round_id") == rid
                                        and r.get("variant_id") == vid
                                        and r.get("promoted") is True), "")),
                "raw_delta": delta,
                "overall_delta": var["mean_composite"] - base["mean_composite"],
                "headroom": m["headroom"], "normalized_gain": m["normalized_gain"],
                "heldout_delta": m["heldout_delta"], "targeted_delta": m["targeted_delta"],
                "split_hash": split["split_hash"],
                "would_promote": bool(should and not refusals),
                "refused_by": refused_by, "score_reason": reason,
                "contract_refusals": refusals,
            })
    return decisions


def report(decisions: list[dict[str, Any]], by_task: bool) -> int:
    replayed = [d for d in decisions if not d.get("skipped")]
    flipped = [d for d in replayed if not d["would_promote"]]
    print(f"recorded promotions replayed: {len(replayed)}   "
          f"(skipped, unmeasurable: {len(decisions) - len(replayed)})")
    print(f"flip to REJECT under the #549 gate: {len(flipped)} "
          f"({len(flipped) / max(1, len(replayed)):.0%})")
    print(f"still promote: {len(replayed) - len(flipped)}")
    print("\nrefused by which condition:")
    for reason, n in Counter(d["refused_by"] for d in flipped).most_common():
        print(f"  {reason:15s} {n}")

    print("\nper-decision (raw delta is the targeted slice; headroom and normalized"
          "\ngain are HarnessOpt-Bench's — the share of the seed's remaining"
          "\nheadroom the change actually captured):")
    print(f"{'round':22s} {'variant':28s} {'raw':>8s} {'headroom':>9s} "
          f"{'normgain':>9s} {'heldout':>8s}  verdict")
    for d in replayed:
        gain = d["normalized_gain"]
        print(f"{d['round_id']:22s} {d['variant_id']:28s} "
              f"{d['raw_delta']:+8.4f} {d['headroom']:9.3f} "
              f"{('n/a' if gain is None else f'{gain:+.1%}'):>9s} "
              f"{d['heldout_delta']:+8.4f}  "
              + ("PROMOTE" if d["would_promote"] else "REFUSE (" + d["refused_by"] + ")"))

    if by_task:
        print("\nwhich veto tasks moved, for the decisions that flipped:")
        regressions: Counter[str] = Counter()
        for d in flipped:
            if d["refused_by"] != "score_gate":
                continue
            if "heldout" in d["score_reason"]:
                regressions[d["score_reason"].split("(")[0].strip()] += 1
        for k, n in regressions.most_common():
            print(f"  {k:20s} {n}")

    named = next((d for d in replayed if d["variant_id"] == NAMED_VARIANT), None)
    print(f"\nnamed variant {NAMED_VARIANT}:")
    if named is None:
        print("  NOT IN THE REPLAYED CORPUS — the assertion cannot run.")
        return 1
    print(f"  recorded as: {named['old_reason']}")
    print(f"  replayed verdict: {'PROMOTE' if named['would_promote'] else 'REFUSE'}"
          f" | refused_by={named['refused_by']} | score_gate said: {named['score_reason']}")
    print(f"  targeted {named['raw_delta']:+.4f} over headroom "
          f"{named['headroom']:.3f} | held-out {named['heldout_delta']:+.4f}")
    for line in named["contract_refusals"]:
        print(f"  contract: {line}")
    if named["would_promote"]:
        print("  ASSERTION FAILED: the item requires this variant be refused.")
        return 1
    if named["refused_by"] != "score_gate":
        print("\n  NOTE, stated plainly because it is the finding and not a detail:")
        print("  the held-out slice did NOT catch this one — its veto slice rose")
        print(f"  ({named['heldout_delta']:+.4f}), and bench_006 improved under it too. It is")
        print("  refused by the contract guard that #377 landed, which the split joins")
        print("  rather than replaces. Any claim that the split alone would have stopped")
        print("  the 09-08 promotion is false on this corpus.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", default=None,
                    help="override the ledger path (a test replays a fixture corpus)")
    ap.add_argument("--variants", default=None,
                    help="override the stored-overlay dir, so the contract-guard half "
                         "can be replayed against a fixture instead of live overlays")
    ap.add_argument("--by-task", action="store_true")
    ap.add_argument("--json", action="store_true", help="dump decisions as json")
    args = ap.parse_args(argv)
    cfg = load_config()
    ledger = Path(args.ledger) if args.ledger else cfg.paths.ledger_path
    if not ledger.exists():
        print(f"no ledger at {ledger}", file=sys.stderr)
        return 2
    variants = Path(args.variants) if args.variants else cfg.paths.variants_dir
    decisions = replay(cfg, ledger, variants)
    if args.json:
        print(json.dumps(decisions, indent=2, default=str))
        return 0
    return report(decisions, args.by_task)


if __name__ == "__main__":
    raise SystemExit(main())
