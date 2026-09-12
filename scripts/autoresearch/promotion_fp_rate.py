"""Measure the false-positive rate of autoresearch promotion at the live gate.

Backlog #428. The promotion gate (:mod:`scripts.autoresearch.promote`) compares a
candidate variant's mean bench composite against its own round's baseline mean
and promotes on ``delta >= min_composite_delta`` **and** ``win_fraction >=
min_bench_win_fraction`` (live: ``0.05`` / ``0.5``). The threshold has been
``0.5`` since 2026-05-23 and 65 promotions had landed, but the false-positive
rate had never been measured, so the keep-0.5 / raise / revert decision that
backlog #352 and #367 existed to make could not be made.

Why not the clean controls
    Backlog #428's method named two noise controls. Neither exists on this
    machine, and :data:`UNAVAILABLE_CONTROLS` records the evidence rather than
    leaving the substitution unauditable:

    * *same-trace re-scoring* — ``judge.py`` scores a ``(task, trace)`` pair, and
      no trace text is persisted anywhere under ``_pipeline/research/``
      (``rounds/R_*/`` holds only ``run_spec.yaml``). "Re-scoring" would mean
      re-running the bench, which cannot reproduce the promoted run's outputs.
    * *a same-day control round* — the ``autoresearch`` worker source is
      ``enabled: false`` (off since 2026-09-08, stays off until backlog #506
      closes), so no new round of any kind can be produced.

What is used instead
    One control is available on disk, and it is the one that matters.

    1. **The null distribution of the drop statistic itself.** For a round ``r``
       let ``bm(r)`` be the ``baseline mean composite`` recorded in
       ``rounds/R_<r>.md``, and define

           D(r) = bm(r) - mean(bm of the next ``window`` non-degenerate rounds)

       A positive ``D`` means measured performance later fell below the level the
       agent had at ``r``. Fresh bench prompts are generated every round (backlog
       #324, ``knowledge/evaluation/autoresearch-baseline-stability.md``: 0%
       cross-round baseline overlap), so a round-to-round fall of this size is
       ordinary: the floor is the ``1 - alpha`` percentile of ``D`` over the
       *null* rounds — every round in the window that was neither a promoted
       round nor a collapse (``bm == 0``, an infrastructure failure rather than a
       measurement).

    There is deliberately no second, matched control group. The rows that look
    like one — ``should_promote: true`` with ``promoted`` false — turn out to be
    *runner-up variants inside rounds that did promote something*: all 16 distinct
    rounds carrying them are promoted rounds. They therefore cannot show what a
    beyond-noise drop looks like "with no promotion anywhere", and any control of
    that shape would be a restatement of the null. :func:`unlanded_pass_rounds`
    returns them with that caveat attached rather than dressed up as a control.

    A promoted round counts as a false positive **only** when ``D(r) > floor``.

The run is anchored to a declared cutoff round, so the measurement stays
reproducible after the loop is re-armed: re-running over the frozen window must
reproduce every number in the note, and later rounds cannot silently change it.

Nothing here edits the threshold. The keep / raise / revert call is human-only
(``config.yaml`` is on the self-modification loop's never-touch list); this
module only reports which band the number lands in.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LEDGER = REPO_ROOT / "_pipeline" / "research" / "ledger.jsonl"
DEFAULT_ROUNDS_DIR = REPO_ROOT / "_pipeline" / "research" / "rounds"

#: Backlog #352's decision bands, carried verbatim so no reader re-invents them.
BANDS = (
    (0.20, "keep 0.5"),
    (0.40, "consider raising to 0.575"),
    (float("inf"), "revert to 0.6"),
)

#: The two controls backlog #428's method step 2 named, with their evidence.
UNAVAILABLE_CONTROLS = {
    "same_trace_rescore": (
        "impossible: judge.py:175 judge_trace(task, trace) needs the model's "
        "output trace and no trace text is persisted. rounds/R_*/ holds only "
        "run_spec.yaml; variants/*/ holds only variant.json plus the overlay "
        "prompt files. Re-scoring would mean re-running the bench, which cannot "
        "reproduce the promoted run's outputs."
    ),
    "same_day_control_round": (
        "impossible today: the autoresearch worker source is enabled: false "
        "(config.yaml:723-724, off since 2026-09-08, stays off until backlog "
        "#506 closes), so no control round can be produced."
    ),
}

BASELINE_RE = re.compile(r"^- baseline mean composite:\s*([0-9.]+)", re.M)
PROMOTE_LINE_RE = re.compile(r"^-\s*`(?P<vid>[^`]+)`:\s*PROMOTE.*delta=(?P<delta>[+-][0-9.]+)")


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile of ``values`` (numpy-free; small populations)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty population")
    if len(ordered) == 1:
        return ordered[0]
    pos = p * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _rows(ledger_path: Path):
    with ledger_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def decision_rows(ledger_path: Path) -> list[dict]:
    """``event == "decision"`` ledger rows — the gate's verdicts."""
    return [r for r in _rows(ledger_path) if r.get("event") == "decision"]


def rounds_by_predicate(rows: list[dict], predicate) -> dict[str, list[str]]:
    """``{round_id: [variant_id, ...]}`` over decision rows matching ``predicate``.

    The promoted set — ``event == "decision"`` and a truthy ``promoted`` — is the
    measurement's denominator; the control set is the same shape with
    ``should_promote`` true and ``promoted`` falsy.
    """
    out: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if predicate(row):
            rid = row.get("round_id")
            if rid:
                out[rid].append(row.get("variant_id") or "")
    return dict(out)


def promoted_rounds(ledger_path: Path) -> dict[str, list[str]]:
    return rounds_by_predicate(decision_rows(ledger_path), lambda r: bool(r.get("promoted")))


def unlanded_pass_rounds(ledger_path: Path) -> dict[str, list[str]]:
    """Rounds carrying a variant that passed the gate but was not the one applied.

    Whether they *are* a control is data, not assertion: it holds only if none of
    these rounds also promoted something. On the live ledger all 16 do, so the
    answer there is no — see ``unlanded_gate_passes`` in the report, and the
    fixture/live pair in ``tests/test_promotion_fp_rate.py`` that pins both sides.
    """
    return rounds_by_predicate(
        decision_rows(ledger_path),
        lambda r: bool(r.get("should_promote")) and not r.get("promoted"),
    )


def baseline_means(rounds_dir: Path) -> dict[str, float]:
    """``{round_id: baseline mean composite}`` from every ``rounds/R_*.md``."""
    out: dict[str, float] = {}
    for path in sorted(rounds_dir.glob("R_*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = BASELINE_RE.search(text)
        if match:
            out[path.stem] = float(match.group(1))
    return out


def baseline_means_from_ledger(table: dict[tuple[str, str], dict[str, float]]) -> dict[str, float]:
    """``{round_id: mean composite of that round's BASELINE_* variant}``.

    Built from the per-task ``composite_score`` rows, so the noise floor and the
    denominator come from one store. A round whose baseline row-set is missing is
    simply absent from the result, which is what makes a missing round visible in the
    counts rather than silently absorbed.
    """
    out: dict[str, float] = {}
    for (rid, vid), scores in table.items():
        if not vid.startswith("BASELINE") or not scores:
            continue
        out[rid] = statistics.fmean(scores.values())
    return out


def recorded_deltas(rounds_dir: Path) -> dict[tuple[str, str], float]:
    """``{(round_id, variant_id): delta}`` from each round's PROMOTE line.

    The gate's own number, transcribed into the round report. Cross-checked
    against the ledger's paired per-task rows by :func:`paired_deltas`.
    """
    out: dict[tuple[str, str], float] = {}
    for path in sorted(rounds_dir.glob("R_*.md")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = PROMOTE_LINE_RE.match(line)
            if match:
                out[(path.stem, match.group("vid"))] = float(match.group("delta"))
    return out


def per_task_rows(ledger_path: Path) -> list[dict]:
    """Per-task evaluation rows — the shape carrying ``composite_score``.

    Excludes ``event`` rows (decisions / round specs), which carry no score.
    """
    return [
        row
        for row in _rows(ledger_path)
        if not row.get("event") and row.get("composite_score") is not None
    ]


def score_table(rows: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """``{(round_id, variant_id): {task_id: composite_score}}``."""
    table: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        table[(row["round_id"], row["variant_id"])][row["task_id"]] = row["composite_score"]
    return table


def paired_deltas(
    table: dict[tuple[str, str], dict[str, float]],
    rounds: set[str],
    promoted: dict[str, list[str]],
):
    """Per-variant delta measured against its own round's baseline, same tasks.

    Returns ``(deltas_by_round_variant, winner_null_deltas)``. A *winner null*
    entry is the best variant delta in a round that was **not** promoted — the
    matched comparator for "what does a round's winner look like when nothing was
    landed". Variants that were themselves promoted are kept out of the
    per-variant null. Used only for the gain-side check; the false-positive rate
    itself is the drop test.
    """
    by_round: dict[str, list[str]] = defaultdict(list)
    for rid, vid in table:
        by_round[rid].append(vid)
    deltas: dict[tuple[str, str], float] = {}
    winner_null: list[float] = []
    for rid in by_round:
        if rid not in rounds:
            continue
        baselines = [v for v in by_round[rid] if v.startswith("BASELINE")]
        if not baselines:
            continue
        base = table[(rid, baselines[0])]
        round_deltas = []
        for vid in by_round[rid]:
            if vid == baselines[0]:
                continue
            scores = table[(rid, vid)]
            common = set(scores) & set(base)
            if len(common) < 2:
                continue
            delta = statistics.fmean(scores[t] for t in common) - statistics.fmean(
                base[t] for t in common
            )
            deltas[(rid, vid)] = delta
            round_deltas.append(delta)
        if round_deltas and rid not in promoted:
            winner_null.append(max(round_deltas))
    return deltas, winner_null


def band_for(rate: float) -> str:
    for cutoff, label in BANDS:
        if rate < cutoff:
            return label
    return BANDS[-1][1]  # pragma: no cover - the last band is open-ended


def measure(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    rounds_dir: Path = DEFAULT_ROUNDS_DIR,
    data_cutoff: str | None = None,
    window: int = 3,
    alpha: float = 0.05,
    alt_floors: dict[str, float] | None = None,
) -> dict:
    """Recompute the whole measurement. Every number in the #428 note is one of these."""
    decisions = decision_rows(ledger_path)
    promoted = rounds_by_predicate(decisions, lambda r: bool(r.get("promoted")))
    unlanded_passes = rounds_by_predicate(
        decisions, lambda r: bool(r.get("should_promote")) and not r.get("promoted")
    )
    rows = per_task_rows(ledger_path)
    table = score_table(rows)
    # The floor's input is the ledger's own per-task composite_score rows: a round's
    # baseline mean composite is the mean of its BASELINE_* variant's task scores. The
    # round reports are the cross-check, not the source — they exist for 351 of the 354
    # rounds, so reading them instead would silently drop three rounds from the window.
    bm_all = baseline_means_from_ledger(table)
    reported = baseline_means(rounds_dir)
    compared = sorted(set(bm_all) & set(reported))
    report_deltas = [abs(bm_all[rid] - reported[rid]) for rid in compared]
    cutoff = data_cutoff or max(bm_all)

    rounds = [rid for rid in sorted(bm_all) if rid <= cutoff]
    collapsed = {rid for rid in rounds if bm_all[rid] <= 0.0}
    sequence = [rid for rid in rounds if rid not in collapsed]
    index = {rid: i for i, rid in enumerate(sequence)}

    def drop(rid: str) -> float | None:
        position = index.get(rid)
        if position is None:
            return None
        following = sequence[position + 1 : position + 1 + window]
        if not following:
            return None
        return bm_all[rid] - statistics.fmean(bm_all[other] for other in following)

    null_values = sorted(
        d for d in (drop(rid) for rid in sequence if rid not in promoted) if d is not None
    )
    if not null_values:
        raise ValueError(
            f"no null population in the window ending {cutoff}: every round with a baseline "
            f"mean was promoted, or the window has no rounds after its rounds. "
            f"The floor is measured against rounds nothing was landed in, so it cannot be "
            f"estimated here — widen --data-cutoff or --window rather than assuming a floor."
        )
    floor = percentile(null_values, 1 - alpha)

    def score_group(members: dict[str, list[str]]) -> list[dict]:
        records = []
        for rid in sorted(members):
            if rid not in bm_all or rid > cutoff:
                continue
            value = drop(rid)
            records.append(
                {
                    "round_id": rid,
                    "variants": sorted(members[rid]),
                    "baseline_mean_composite": bm_all[rid],
                    "control_rounds_used": (
                        max(0, min(window, len(sequence) - index[rid] - 1)) if rid in index else 0
                    ),
                    "drop": None if value is None else round(value, 4),
                    "is_fp": value is not None and value > floor,
                }
            )
        return records

    promoted_records = score_group(promoted)
    scored = [r for r in promoted_records if r["drop"] is not None]

    def summarise(records: list[dict], denominator: int, all_records: list[dict] | None = None) -> dict:  # noqa: E501
        hits = [r["round_id"] for r in records if r["is_fp"]]
        rate = len(hits) / denominator if denominator else 0.0
        return {
            "rounds_total": len(all_records if all_records is not None else records),
            "measurable_rounds": len(records),
            "fp_count": len(hits),
            "fp_rate": round(rate, 4),
            "fp_rate_fraction": f"{len(hits)}/{denominator}",
            "fp_rounds": hits,
            "band": band_for(rate),
            "drops": {r["round_id"]: r["drop"] for r in records},
        }

    denominator = len(promoted_records)
    primary = summarise(scored, denominator, promoted_records)
    drops = [r["drop"] for r in scored]

    sensitivities = {}
    for name, value in (alt_floors or {}).items():
        hits = [r["round_id"] for r in scored if r["drop"] > value]
        rate = len(hits) / denominator if denominator else 0.0
        sensitivities[name] = {
            "floor": round(value, 4),
            "fp_count": len(hits),
            "fp_rate": round(rate, 4),
            "fp_rate_fraction": f"{len(hits)}/{denominator}",
            "band": band_for(rate),
            "rounds": hits,
        }

    paired, best_null = paired_deltas(table, set(rounds), promoted)
    deltas = recorded_deltas(rounds_dir)
    gain = []
    for record in promoted_records:
        for vid in record["variants"]:
            gain.append(
                {
                    "round_id": record["round_id"],
                    "variant_id": vid,
                    "recorded_delta": deltas.get((record["round_id"], vid)),
                    "paired_delta": (
                        round(paired[(record["round_id"], vid)], 4)
                        if (record["round_id"], vid) in paired
                        else None
                    ),
                }
            )
    best_null_floor = percentile(best_null, 1 - alpha) if best_null else None
    inside = [
        g["round_id"]
        for g in gain
        if g.get("paired_delta") is not None
        and best_null_floor is not None
        and g["paired_delta"] <= best_null_floor
    ]

    return {
        "schema": "autoresearch-promotion-fp/1",
        "threshold_win_fraction": 0.5,
        "data_cutoff_round": cutoff,
        "window_rounds": window,
        "alpha": alpha,
        "ledger_rows": sum(1 for line in ledger_path.open() if line.strip()),
        "decision_rows": len(decisions),
        "should_promote_rows": sum(1 for r in decisions if r.get("should_promote")),
        "per_task_score_rows": len(rows),
        "distinct_task_ids": len({r["task_id"] for r in rows}),
        "rounds_with_baseline_mean_in_window": len(rounds),
        "report_crosscheck": {
            "source_of_record": "ledger.jsonl BASELINE_* per-task composite_score rows",
            "compared_against": "rounds/R_*.md '- baseline mean composite:' lines",
            "rounds_compared": len(compared),
            "rounds_with_ledger_baseline_but_no_report": len(set(bm_all) - set(reported)),
            "max_abs_delta": round(max(report_deltas), 6) if report_deltas else None,
            "tolerance": 1e-3,
            "mismatches": sum(1 for d in report_deltas if d > 1e-3),
        },
        "rounds_excluded_zero_baseline": len(collapsed),
        "denominator": denominator,
        "floor": round(floor, 4),
        "floor_definition": (
            f"{round((1 - alpha) * 100)}th percentile of the drop statistic over the "
            f"null (non-promoted, non-collapsed) rounds in the window; each round's "
            f"baseline mean is the mean of its BASELINE_* per-task composite_score rows"
        ),
        "null_population": {
            "n": len(null_values),
            "mean": round(statistics.fmean(null_values), 4),
            "median": round(statistics.median(null_values), 4),
            "std": round(statistics.pstdev(null_values), 4),
            "mad_std": round(
                1.4826 * statistics.median([abs(d - statistics.median(null_values)) for d in null_values]),
                4,
            ),
            "p95": round(floor, 4),
            "max": round(max(null_values), 4),
        },
        "promoted_drops": {
            "mean": round(statistics.fmean(drops), 4),
            "median": round(statistics.median(drops), 4),
            "max": round(max(drops), 4),
            "count_positive": sum(1 for d in drops if d > 0),
            "count_positive_fraction": f"{sum(1 for d in drops if d > 0)}/{denominator}",
        },
        "fp_count": primary["fp_count"],
        "fp_rate": primary["fp_rate"],
        "fp_rate_fraction": primary["fp_rate_fraction"],
        "fp_rounds": primary["fp_rounds"],
        "band": primary["band"],
        "expected_false_alarms_at_alpha": round(alpha * denominator, 2),
        "unlanded_gate_passes": {
            "rounds": sorted(unlanded_passes),
            "rows": sum(1 for d in decisions if d.get("should_promote") and not d.get("promoted")),
            "also_promoted_rounds": len(set(unlanded_passes) & set(promoted)),
            "is_a_no_landing_control": not (set(unlanded_passes) & set(promoted)),
            "why": (
                "every distinct round carrying an unlanded gate-pass is itself a "
                "promoted round, so these are runner-up variants, not promotion-free "
                "rounds — they cannot calibrate a no-promotion drop rate"
            ),
        },
        "sensitivities": sensitivities,
        "gain_side": {
            "note": (
                "not the false-positive rate: whether each promotion's claimed gain "
                "beat the same-round winner-matched null"
            ),
            "promoted_median_recorded_delta": round(
                statistics.median([g["recorded_delta"] for g in gain if g.get("recorded_delta") is not None]),
                4,
            )
            if gain
            else None,
            "winner_null": {
                "n": len(best_null),
                "median": round(statistics.median(best_null), 4) if best_null else None,
                "p95": round(best_null_floor, 4) if best_null_floor is not None else None,
            },
            "rounds_at_or_below_winner_null_p95": len(inside),
            "rounds_at_or_below_fraction": (
                f"{len(inside)}/{denominator}" if denominator else None
            ),
            "per_round": gain,
        },
        "rounds": promoted_records,
        "unavailable_controls": UNAVAILABLE_CONTROLS,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure autoresearch promotion false-positive rate")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--rounds-dir", type=Path, default=DEFAULT_ROUNDS_DIR)
    ap.add_argument("--data-cutoff", default=None, help="last round id to include")
    ap.add_argument("--window", type=int, default=3, help="control rounds taken after each round")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alt-floor", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here too")
    args = ap.parse_args(argv)

    alt = {}
    for item in args.alt_floor:
        name, _, value = item.partition("=")
        alt[name] = float(value)

    result = measure(
        ledger_path=args.ledger,
        rounds_dir=args.rounds_dir,
        data_cutoff=args.data_cutoff,
        window=args.window,
        alpha=args.alpha,
        alt_floors=alt,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
