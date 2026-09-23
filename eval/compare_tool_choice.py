#!/usr/bin/env python3
"""Compare two tool-choice eval runs against each metric's own noise floor.

Why this exists
---------------
`run_tool_choice_eval.py` writes a JSON baseline per run and prints nothing a
reader can compare. On 2026-09-08 the round that rewrote 66% of the operating
contract ran it, then spent four iterations guessing the file's shape —
`summary` came back `{}` on the first three probes because the script looked
for the wrong key — and landed without ever comparing against the 0.90
`correct_rate` recorded on 09-04. A number with no prior is not a regression
guard; it is a number.

So this does the comparison and exits non-zero when something moved the wrong
way. No model, no vault, no network: it reads two JSON files.

Why the decision is per-metric and measured
-------------------------------------------
The first version decided against one hand-picked constant,
`TOLERANCE = 0.051`, on the reasoning that one query of twenty is 0.05. That
constant sat *below* the instrument's own run-to-run spread, so it flagged
movements a re-run of the identical tree also produces: two runs 82 s apart on
an unmodified tree (backlog #691, 2026-09-09) moved `correct_rate` by 0.050 and
`http_tool_first_rate` by 0.071, and reproduced exactly on 2026-09-13
(`875noise-a` / `875noise-b`, 75 s apart). A gate that fires on its own noise
teaches the implementer to argue their way past it, which is what round
SM_20260908_165950 had to do.

So the threshold is now derived, per metric, from two things this file can
actually see: the binomial spread of a proportion at that metric's own
denominator (`correct_rate` is over 20 queries, `http_tool_first_rate` over the
14 public-web ones, `control_correct_rate` over the 6 controls — a rate over 6
trials is roughly 1.8x noisier than one over 20), and the largest same-tree
spread on record in `noise_floor_tool_choice.yaml`, which
`--measure-floor` writes from labelled a/b run pairs. Every floor is printed
beside the delta it decided against, so a reader can see which side of the
noise a movement is on rather than which side of a constant.

The cost of that honesty is stated in the output: at n=20 this instrument can
only certify movements larger than about 0.14. Raising n is the way to certify
smaller ones; lowering the tolerance is not.

Exit codes
----------
0  no metric moved further than its own noise floor
1  a treated metric moved the wrong way beyond its floor  -> the change regressed
2  there was nothing to compare                          -> not a pass
3  the CONTROL set moved beyond its floor -> instrument failure. The control
   rows (localhost, structured-API) are ones where Bash is the right answer and
   the prompt surface cannot reach them, so a movement there is the instrument
   reading something else — engine state, tool surface, a different tree. The
   comparison certified nothing; re-run both sides. Exit 3 is NOT a regression
   and must not be argued past as one.

Usage:
    python eval/compare_tool_choice.py                     # newest vs the one before
    python eval/compare_tool_choice.py --label soul377-post
    python eval/compare_tool_choice.py --current a.json --baseline b.json
    python eval/compare_tool_choice.py --measure-floor     # rewrite the floor record
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from app.paths import EVAL_BASELINES_DIR  # noqa: E402

BASELINE_DIR = EVAL_BASELINES_DIR / "tool-choice"
FLOOR_RECORD = HERE / "noise_floor_tool_choice.yaml"

# Metric -> whether higher is better. `shelled_to_web_rate` and
# `no_tool_call_rate` are failures being counted, so lower wins; the rest are
# successes. Getting this backwards would report a regression as an
# improvement, which is worse than not comparing at all.
METRICS: dict[str, bool] = {
    "correct_rate": True,
    "http_tool_first_rate": True,
    "control_correct_rate": True,
    "shelled_to_web_rate": False,
    "no_tool_call_rate": False,
}

# Metrics over the control rows. A change to the prompt surface cannot move
# these, so a beyond-floor movement here is the instrument, not the change.
CONTROL_METRICS = frozenset({"control_correct_rate"})

# How wide a band around the mean counts as "not a movement". 2σ is ~95% of a
# normal approximation to the binomial: with the floor set here, a delta this
# small or smaller has had a chance to appear on an unchanged tree.
BINOMIAL_SIGMAS = 2.0


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def overall(run: dict[str, Any]) -> dict[str, Any]:
    """The summary block, tolerating the shape drift between versions."""
    summary = run.get("summary") or {}
    if isinstance(summary, dict) and summary.get("overall"):
        return summary["overall"]
    return summary if isinstance(summary, dict) else {}


def summary_block(run: dict[str, Any]) -> dict[str, Any]:
    s = run.get("summary")
    return s if isinstance(s, dict) else {}


def runs_by_recency(directory: Path | None = None) -> list[Path]:
    directory = directory or BASELINE_DIR
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


# ── the noise floor ─────────────────────────────────────────────────────────

def denominators(run: dict[str, Any]) -> dict[str, int]:
    """What each rate in this run was divided by.

    Read from the run's own per-query records rather than a constant, because
    `--only` runs change the split and because `http_tool_first_rate` is never
    over the same denominator as `correct_rate`. When a run carries no records
    (a summary-only artifact) the split is unknown and every metric is priced
    over the whole set; `compare` says so, since an unknown denominator makes
    the control floor too narrow, not too wide.
    """
    recs = run.get("records")
    n_all = run.get("n_queries") or (len(recs) if isinstance(recs, list) else 0)
    n_all = int(n_all or 0)
    if isinstance(recs, list) and recs:
        n_web = sum(1 for r in recs if (r.get("scoring") or {}).get("is_web_category"))
        n_ctl = len(recs) - n_web
        split_known = True
    else:
        n_web = n_ctl = n_all
        split_known = False
    return {
        "correct_rate": n_all,
        "http_tool_first_rate": n_web,
        "shelled_to_web_rate": n_web,
        "no_tool_call_rate": n_all,
        "control_correct_rate": n_ctl,
        "__split_known__": split_known,
    }


def binomial_sigma(p: float, n: int) -> float:
    """1σ of a proportion measured over `n` binary trials.

    A rate pinned at 0 or 1 cannot price its own variance — the sample sd is
    zero there, which would set the floor to nothing and let a single flipped
    query read as a movement. So p is clamped to the nearest interior value the
    denominator can express (1/n … 1-1/n) before the variance is taken.
    """
    if n < 2:
        return 1.0
    p_eff = min(max(float(p), 1.0 / n), 1.0 - 1.0 / n)
    return math.sqrt(p_eff * (1.0 - p_eff) / n)


def observed_spreads(record: dict[str, Any] | None) -> dict[str, float]:
    """Per-metric largest same-tree spread from the floor record, or {}."""
    if not isinstance(record, dict):
        return {}
    out: dict[str, float] = {}
    for metric, block in (record.get("metrics") or {}).items():
        v = (block or {}).get("observed_spread")
        if isinstance(v, (int, float)):
            out[metric] = float(v)
    return out


def load_floor_record(path: Path | str | None = None) -> dict[str, Any] | None:
    """The measured-spread record, or None. Malformed or absent is the same answer."""
    path = Path(path) if path else FLOOR_RECORD
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None


def noise_floor(metric: str, base: float, cur: float, n: int,
                observed: dict[str, float]) -> tuple[float, str]:
    """(floor, how_it_was_derived) — the threshold this delta is decided against.

    Never below `BINOMIAL_SIGMAS` σ of the rate at its own denominator, and
    never below the spread two runs of an unchanged tree were actually measured
    to produce. The larger of the two wins: the binomial term says what a
    coin-flip instrument of that size can do, the measured term says what THIS
    engine did.
    """
    if n < 2:
        return 1.0, "no usable denominator — nothing is certifiable at this n"
    sigma = max(binomial_sigma(base, n), binomial_sigma(cur, n))
    floor = BINOMIAL_SIGMAS * sigma
    # The label names BOTH terms even when one is not binding. "Floor 0.160"
    # with no measured spread beside it is the unauditable verdict #691 was
    # filed about wearing a different hat: a reader has to be able to see which
    # term decided, and what the other one said.
    obs = observed.get(metric)
    parts = [f"binom {BINOMIAL_SIGMAS:g}σ @{n}={floor:.3f}"]
    if obs is not None:
        parts.append(f"measured same-tree spread={obs:.3f}")
        floor = max(floor, float(obs))
    return floor, " / ".join(parts)


def certifiable_floor(floors: dict[str, float]) -> tuple[str, float] | None:
    """The smallest movement this comparison could certify, and its metric."""
    if not floors:
        return None
    metric = min(floors, key=lambda m: floors[m])
    return metric, floors[metric]


# ── measuring the floor from same-tree run pairs ────────────────────────────

def same_tree_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """`<stem>-a-<ts>.json` / `<stem>-b-<ts>.json` pairs in `directory`.

    The convention `--label noise-a` then `--label noise-b` produces on an
    unmodified tree. A pair only counts when the two runs' recorded `config`
    blocks agree — the model, the tool surface and the system-prompt size are
    what would have had to stay still for the spread to be noise.
    """
    groups: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.name[: -len(".json")]
        label = stem.rsplit("-", 2)[0]           # strip the -YYYYMMDD-HHMMSS
        for arm in ("a", "b"):
            if label.endswith(f"-{arm}"):
                groups.setdefault(label[: -len(f"-{arm}")], {}).setdefault(arm, []).append(path)
    pairs: list[Path | tuple] = []
    for stem, arms in sorted(groups.items()):
        if not (arms.get("a") and arms.get("b")):
            continue
        a, b = arms["a"][-1], arms["b"][-1]       # newest of each arm
        try:
            ca, cb = load(a).get("config"), load(b).get("config")
        except (OSError, json.JSONDecodeError):
            continue
        if ca and ca == cb:
            pairs.append((a, b))
    return pairs


def measure_floor(directory: Path, out_path: Path) -> dict[str, Any]:
    """Write the floor record from every same-tree pair on disk. Returns it."""
    pairs = same_tree_pairs(directory)
    per_metric: dict[str, float] = {}
    pair_rows = []
    for a, b in pairs:
        ra, rb = load(a), load(b)
        oa, ob = overall(ra), overall(rb)
        if (oa.get("errors") or 0) or (ob.get("errors") or 0):
            continue          # an errored run measures the harness, not the model
        spread: dict[str, float] = {}
        for metric in METRICS:
            x, y = oa.get(metric), ob.get(metric)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                d = round(abs(y - x), 3)
                spread[metric] = max(spread.get(metric, 0.0), d)
                per_metric[metric] = max(per_metric.get(metric, 0.0), d)
        pair_rows.append({"a": a.name, "b": b.name,
                          "n_queries": ra.get("n_queries"), "spread": spread})

    record = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "eval/compare_tool_choice.py --measure-floor",
        "note": (
            "Largest per-metric movement two runs of the SAME unmodified tree "
            "produced, from labelled a/b run pairs under eval/baselines/"
            "tool-choice with identical recorded config. Backlog #691 measured "
            "the same pair on 2026-09-09 82 s apart (correct_rate +0.050, "
            "http_tool_first_rate +0.071); the 875noise pair on 2026-09-13 "
            "reproduced both magnitudes at the opposite sign. These are floors, "
            "not budgets: compare_tool_choice decides against max(this, 2σ "
            "binomial at the metric's own denominator)."
        ),
        "binomial_sigmas": BINOMIAL_SIGMAS,
        "pairs": pair_rows,
        "metrics": {m: {"observed_spread": per_metric[m]} for m in sorted(per_metric)},
    }
    out_path.write_text(yaml.safe_dump(record, sort_keys=False, width=100),
                        encoding="utf-8")
    return record


# ── the comparison ──────────────────────────────────────────────────────────

class Comparison(NamedTuple):
    """What a comparison of two runs knows, not just what it concluded.

    `floors` is per-metric `{floor, src, n}` — the threshold, how it was
    derived, and the denominator it was priced over — because the whole point
    of #691 is that a verdict without its threshold is unauditable.
    `failures` is the control set moving: not a regression, an instrument that
    just told you nothing.
    """
    regressions: list[str]
    failures: list[str]
    lines: list[str]
    floors: dict[str, dict[str, Any]]
    n_queries: int
    split_known: bool


def compare(current: dict[str, Any], baseline: dict[str, Any],
            observed: dict[str, float] | None = None) -> Comparison:
    """The comparison: the table, its two verdict lists, and the floors.

    `observed` is the per-metric measured same-tree spread; when omitted it is
    read from `FLOOR_RECORD`, and a missing record simply means the floors are
    the binomial term alone — which the table labels as such.
    """
    if observed is None:
        observed = observed_spreads(load_floor_record())
    cur, base = overall(current), overall(baseline)
    denoms = denominators(current) if current.get("records") else denominators(baseline)
    regressions: list[str] = []
    failures: list[str] = []
    lines: list[str] = []
    floors: dict[str, dict[str, Any]] = {}

    for metric, higher_is_better in METRICS.items():
        if metric not in cur and metric not in base:
            continue
        c = cur.get(metric)
        b = base.get(metric)
        if not isinstance(c, (int, float)) or not isinstance(b, (int, float)):
            lines.append(f"  {metric:<24} {b!r} -> {c!r}  (not comparable)")
            continue
        n = int(denoms.get(metric) or 0)
        delta = c - b
        floor, src = noise_floor(metric, b, c, n, observed)
        floors[metric] = {"floor": floor, "src": src, "n": n}
        # A control metric is decided on |delta|: the rows it covers have one
        # fixed right answer, so the direction of an impossible move says
        # nothing and its size says everything.
        if metric in CONTROL_METRICS:
            beyond = abs(delta) > floor
        else:
            beyond = delta > floor if not higher_is_better else delta < -floor
        if beyond:
            mark = "INSTRUMENT" if metric in CONTROL_METRICS else "REGRESSED"
            row = (f"{metric} {b:.3f} -> {c:.3f} ({delta:+.3f}) "
                   f"beyond floor {floor:.3f}")
            (failures if metric in CONTROL_METRICS else regressions).append(row)
        else:
            mark = "ok"
        lines.append(f"  {metric:<24} {b:.3f} -> {c:.3f}  ({delta:+.3f})  "
                     f"floor {floor:.3f} ({src})  {mark}")

    # An eval that errored measured nothing, whatever its rates say.
    errors = cur.get("errors")
    if isinstance(errors, int) and errors:
        regressions.append(f"{errors} quer(y/ies) errored — the run did not measure cleanly")

    n_q = current.get("n_queries") or baseline.get("n_queries") or 0
    return Comparison(regressions=regressions, failures=failures, lines=lines,
                      floors=floors, n_queries=int(n_q or 0),
                      split_known=bool(denoms.get("__split_known__")))


def _floor_report(result: Comparison, observed: dict[str, float],
                  record_measured: bool) -> list[str]:
    """The 'what this instrument can actually certify' block (clauses 1 and 3)."""
    out: list[str] = []
    floors = {m: block["floor"] for m, block in result.floors.items()}
    n = result.n_queries
    cert = certifiable_floor(floors)
    if cert:
        metric, floor = cert
        out.append(f"smallest movement this comparison can certify: {floor:+.3f} "
                   f"(floor of `{metric}`)" + (f", n={n}" if n else ""))
    # The one-flip guarantee is per-metric, because each rate has its own
    # denominator: one flipped query is 1/20 on `correct_rate` but 1/6 on the
    # controls. It is asserted, not asserted-to-be-true — any metric whose floor
    # came out narrower than its own single query is named, because that floor is
    # the defect this item exists to remove and the report must not hide it.
    if result.floors:
        narrow = {m: b for m, b in result.floors.items()
                  if b["n"] and b["floor"] < 1.0 / int(b["n"]) - 1e-9}
        one_flip = "; ".join(
            f"{m}: one flip is 1/{b['n']}={1.0 / int(b['n']):.3f}, floor {b['floor']:.3f}"
            for m, b in sorted(result.floors.items()) if b["n"])
        if one_flip:
            out.append(f"  a single flipped query is never a movement — {one_flip}")
        if narrow:
            out.append(f"  DEFECT: {', '.join(sorted(narrow))}: floor is narrower than one "
                       "query of its own set, so a single flip WOULD read as a movement.")
    if not observed:
        out.append("  floor source: UNMEASURED — no noise_floor_tool_choice.yaml, so every "
                   "floor is binomial 2σ alone. `--measure-floor` fixes that.")
    else:
        named = sum(1 for m, b in result.floors.items()
                    if f"measured same-tree spread={b['floor']:.3f}" in str(b["src"]))
        out.append(f"  floor source: every floor is max(binomial {BINOMIAL_SIGMAS:g}σ at its own "
                   f"denominator, measured same-tree spread); {named} of "
                   f"{len(result.floors)} compared metric(s) were decided by the measured "
                   f"spread, the rest by the binomial term "
                   f"({FLOOR_RECORD.name}, {len(observed)} metric(s) on record).")
    out.append("  a movement smaller than a floor is NOT evidence of anything: to certify "
               "smaller ones raise the query count, never the tolerance.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", help="compare the newest run with this label "
                                    "against the newest run without it")
    ap.add_argument("--current", help="explicit path to the run under test")
    ap.add_argument("--baseline", help="explicit path to the prior run")
    ap.add_argument("--baselines", help=f"directory of run artifacts (default {BASELINE_DIR})")
    ap.add_argument("--measure-floor", action="store_true",
                    help="rewrite the noise floor record from same-tree a/b run pairs and exit")
    ap.add_argument("--floor-record", help=f"where the floor record lives (default {FLOOR_RECORD})")
    args = ap.parse_args(argv)

    directory = Path(args.baselines) if args.baselines else BASELINE_DIR
    floor_path = Path(args.floor_record) if args.floor_record else FLOOR_RECORD

    if args.measure_floor:
        record = measure_floor(directory, floor_path)
        print(f"[info] measured floor from {len(record['pairs'])} same-tree pair(s) "
              f"in {directory} -> {floor_path}")
        for metric, block in record["metrics"].items():
            print(f"  {metric:<24} observed spread {block['observed_spread']:.3f}")
        if not record["pairs"]:
            print("[warn] no `<stem>-a`/`<stem>-b` pairs with matching config: the record "
                  "is empty and every floor will read as binomial-only")
            return 2
        return 0

    record = load_floor_record(floor_path)
    observed = observed_spreads(record)

    if args.current and args.baseline:
        cur_path, base_path = Path(args.current), Path(args.baseline)
    else:
        runs = runs_by_recency(directory)
        if len(runs) < 2:
            print(f"[skip] need two runs under {directory}, found {len(runs)}. "
                  "Nothing to compare — this is not a pass.")
            return 2
        if args.label:
            match = [p for p in runs if p.name.startswith(args.label)]
            if not match:
                print(f"[error] no run under {directory} labelled {args.label!r}")
                return 2
            cur_path = match[0]
            prior = [p for p in runs if not p.name.startswith(args.label)]
            if not prior:
                print(f"[skip] every run is labelled {args.label!r}; no prior to compare against.")
                return 2
            base_path = prior[0]
        else:
            cur_path, base_path = runs[0], runs[1]

    current, baseline = load(cur_path), load(base_path)
    result = compare(current, baseline, observed)

    print(f"current : {cur_path.name}  ({current.get('n_queries', '?')} queries)")
    print(f"baseline: {base_path.name}  ({baseline.get('n_queries', '?')} queries)")
    if current.get("n_queries") != baseline.get("n_queries"):
        print("  note: query counts differ — the two runs are not the same test")
    if not result.split_known:
        print("  note: neither run carries per-query records, so the web/control split is "
              "unknown and every floor is priced over the whole set — the control "
              "floor is too NARROW, which makes an instrument failure harder to see")
    print("\n".join(result.lines))
    print("\n".join([""] + _floor_report(result, observed, bool(record))))

    if result.failures:
        print("\nINSTRUMENT FAILURE (exit 3): " + "; ".join(result.failures))
        print("  The control rows (localhost, structured-API) are ones where Bash is right "
              "and the prompt surface cannot reach them. A beyond-floor move there is the "
              "instrument reading engine or tool-surface state, not the change: this "
              "comparison certified nothing. Re-run both sides before reading anything "
              "into it, and do not treat 3 as a regression to argue past.")
        if result.regressions:
            print("  Also recorded as regressions, but void until re-measured: "
                  + "; ".join(result.regressions))
        return 3
    if result.regressions:
        print("\nREGRESSED: " + "; ".join(result.regressions))
        return 1
    print("\nno movement beyond the per-metric noise floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
