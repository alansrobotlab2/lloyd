#!/usr/bin/env python3
"""Re-decide past retrieval-eval verdicts against a 95 % interval (#696 clause 4).

Why this exists and what it refuses to do
----------------------------------------
`skills/retrieval-eval/SKILL.md` §"Step 3" and its nightly report have called any
metric that moved more than 0.05 a finding since the skill was written, and
`scripts/eval_trend_stats.py` (#608) established that the 20-query eval cannot
support a 0.05 call — the 09-09 run's "third consecutive night of entity-side
decline. Not noise." is one query of twenty. This script measures that gap
instead of arguing about it: it replays the baselines already on disk and counts
how many of the movements the old rule would have called out a 95 % interval
calls *indistinguishable*.

It is offline by construction, which is the whole reason its answer is
trustworthy: no eval is re-run, no LLM, no network, no daemon — every input is a
`records[]` array `eval/run_eval.py` already wrote. A backtest that re-ran the
eval would be measuring today's vault against yesterday's code.

Two interval forms, chosen by the data, never by preference
----------------------------------------------------------
* **paired** (`eval.stats.paired_bootstrap_ci`) — only when the two runs share
  the same query ids AND the same `corpus` provenance block. The eval is
  deterministic (stdev 0.0000 measured over five repeats,
  `workers/sources/automod_regression.py:14-18`), so on a truly identical corpus
  the only variance is which queries a change moved.
* **independent** (`eval.stats.independent_bootstrap_ci`) — whenever the ids or
  the provenance differ, or a run has no `corpus` block at all. `drift unknown`
  is NOT `drift zero`: an audit that treats an unmeasured corpus as an unchanged
  one is the failure mode #608 was filed to remove, and quoting a paired interval
  across a corpus shift manufactures the precision this item exists to retire.

The split is printed, so a headline count can be read for what it is rather than
as a single verdict about every comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

try:
    from eval import stats as evstats
except ImportError:  # pragma: no cover - script-dir invocation
    import stats as evstats

#: the trigger the skill's Step 3 used to fire on, quoted from
#: `skills/retrieval-eval/SKILL.md` §Step 3 before this item changed it.
MOVE_THRESHOLD = 0.05

#: metric -> the `scoring` key it averages. Same seven as
#: `eval/run_eval.py:CI_METRICS`; kept local so this file still runs against a
#: baseline written before the ci95 block existed.
METRICS = {
    "entity_hit_rate": "entity_hit",
    "doc_hit_rate": "doc_hit",
    "entity_recall_avg": "entity_recall",
    "doc_recall_avg": "doc_recall",
    "mrr_doc": "rr_doc",
    "ndcg10": "ndcg10",
    "fact_entity_recall_avg": "fact_entity_recall",
}

#: The metrics `skills/retrieval-eval` §Step 3 names as entity-side; reported
#: separately because a headline over all seven is a wider claim than the rule
#: that actually authored the verdicts.
ENTITY_SIDE = ("entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg")

_FAMILY = re.compile(r"^(?P<base>.*?)(?P<stamp>(?:-\d{8})+(?:-\d{6})?)$")

#: The families whose comparisons became verdicts somebody read. `nightly` is
#: autonomy task #82's report — the surface that actually authors regression
#: claims — and the `*-check` families are the self-mod/autoimplement gate arms,
#: which a promotion decision was made on. Everything else on disk is a
#: one-off experiment arm (`qmdpin-*`, `ab-alpha-*`, `automod-noise-*`,
#: `guard-*`): consecutive arms of someone's A/B probe never produced a nightly
#: line, so counting them as "prior verdicts" would inflate M with comparisons
#: no report ever made. They are still counted, under their own heading, because
#: the number of times a 0.05 movement fired on an unchanged corpus is itself a
#: datum — `--families all` makes them the headline instead.
AUTHORITATIVE = ("nightly", "automod-check", "selfmod-check", "autoimplement-check")


def family_of(name: str) -> str:
    """`nightly-20260904-20260904-060324.json` and
    `nightly-20260913-20260913-060251.json` are one family.

    The filename is `{label}-{YYYYMMDD}-{YYYYMMDD}-{HHMMSS}` (`eval/run_eval.py`),
    and the nightly job passes the day IN the label — so stripping only the
    trailing timestamp leaves every nightly its own family and the backtest
    compares zero pairs of the one series that matters. Measured, not assumed:
    that was the first version of this function, and it reported 0 nightly
    transitions over 17 nightly baselines.
    """
    stem = name[:-5] if name.endswith(".json") else name
    m = _FAMILY.match(stem)
    return m.group("base") if (m and m.group("stamp")) else stem


def is_authoritative(family: str) -> bool:
    return any(family == a or family.startswith(a + "-") for a in AUTHORITATIVE)



class Run:
    """One baseline artifact, reduced to what a verdict was made from."""

    def __init__(self, path: Path):
        doc = json.loads(path.read_text())
        self.path = path
        self.label = str(doc.get("label") or path.stem)
        self.ran_at = str(doc.get("ran_at") or "")
        self.records = {r["id"]: (r.get("scoring") or {})
                        for r in (doc.get("records") or []) if r.get("id")}
        corpus = doc.get("corpus")
        self.corpus = (json.dumps(corpus, sort_keys=True)
                       if isinstance(corpus, dict) and corpus else None)

    @property
    def family(self) -> str:
        return family_of(self.path.name)

    @property
    def authoritative(self) -> bool:
        """Did a comparison in this family ever become a line a person read?"""
        return is_authoritative(self.family)

    def values(self, metric: str) -> dict[str, float]:
        key = METRICS[metric]
        out = {}
        for qid, scoring in self.records.items():
            v = scoring.get(key)
            if v is None:
                continue
            out[qid] = 1.0 if v else 0.0 if metric.endswith("_rate") else float(v)
        return out

    @property
    def usable(self) -> bool:
        return len(self.records) >= 2


def load_runs(baselines: Path) -> list[Run]:
    runs = []
    for p in sorted(baselines.glob("*.json")):
        try:
            r = Run(p)
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            continue
        if r.usable:
            runs.append(r)
    runs.sort(key=lambda r: (r.family, r.ran_at or r.path.name))
    return runs


def corpus_state(prev: Run, cur: Run) -> str:
    """`paired` / `unpaired-ids` / `unpaired-corpus` / `unpaired-drift-unknown`."""
    if prev.corpus is None or cur.corpus is None:
        return "unpaired-drift-unknown"
    if set(prev.records) != set(cur.records):
        return "unpaired-ids"
    if prev.corpus != cur.corpus:
        return "unpaired-corpus"
    return "paired"


def transitions(runs: list[Run]) -> list[tuple[Run, Run]]:
    """Consecutive runs *within* a family, oldest first.

    `nightly` vs `automod-check` vs `selfmod-check` never compare to each other:
    a nightly and a gate arm have different corpora by construction, and
    cross-family pairs would produce intervals that mean nothing.
    """
    out = []
    by_family: dict[str, list[Run]] = {}
    for r in runs:
        by_family.setdefault(r.family, []).append(r)
    for fam in sorted(by_family):
        seq = by_family[fam]
        out.extend((a, b) for a, b in zip(seq, seq[1:]))
    return out


def redecide(prev: Run, cur: Run) -> list[dict]:
    """Every metric the old rule would call out, and what an interval says."""
    state = corpus_state(prev, cur)
    paired = state == "paired"
    shared = sorted(set(prev.records) & set(cur.records))
    out = []
    for metric in METRICS:
        a_all, b_all = prev.values(metric), cur.values(metric)
        shared_m = [q for q in shared if q in a_all and q in b_all]
        if not shared_m:
            continue
        a = [a_all[q] for q in shared_m]
        b = [b_all[q] for q in shared_m]
        delta = (sum(b) / len(b)) - (sum(a) / len(a))
        if abs(delta) <= MOVE_THRESHOLD:
            continue
        res = (evstats.paired_bootstrap_ci(a, b) if paired
               else evstats.independent_bootstrap_ci(a, b))
        out.append({
            "prev": prev.path.name, "cur": cur.path.name, "family": cur.family,
            "authoritative": cur.authoritative,
            "metric": metric, "entity_side": metric in ENTITY_SIDE,
            "pairing": state, "n": len(shared_m),
            "prev_rate": round(sum(a) / len(a), 4), "cur_rate": round(sum(b) / len(b), 4),
            "delta": round(delta, 4),
            "lo": round(res["lo"], 4), "hi": round(res["hi"], 4),
            "indistinguishable": not res["significant"],
        })
    return out


def _tally(decisions: list[dict]) -> dict:
    n = len(decisions)
    ind = sum(1 for d in decisions if d["indistinguishable"])
    paired = [d for d in decisions if d["pairing"] == "paired"]
    return {
        "prior_verdicts": n,
        "indistinguishable": ind,
        "distinguished": n - ind,
        "entity_side_verdicts": sum(1 for d in decisions if d["entity_side"]),
        "entity_side_indistinguishable": sum(1 for d in decisions
                                             if d["entity_side"] and d["indistinguishable"]),
        "paired_verdicts": len(paired),
        "paired_indistinguishable": sum(1 for d in paired if d["indistinguishable"]),
    }


def summarise(auth: list[dict], all_decisions: list[dict], runs: list[Run],
              pairs: list[tuple[Run, Run]]) -> dict:
    ran = [r.ran_at for r in runs if r.ran_at]
    auth_runs = [r for r in runs if r.authoritative]
    auth_ran = [r.ran_at for r in auth_runs if r.ran_at]
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "move_threshold": MOVE_THRESHOLD,
        "confidence": 0.95,
        "seed": evstats.SEED,
        "n_resamples": evstats.N_RESAMPLES,
        "window": {
            "first_run": min(auth_ran) if auth_ran else None,
            "last_run": max(auth_ran) if auth_ran else None,
            "n_baselines": len(auth_runs),
            "families": sorted({r.family for r in auth_runs}),
            "transitions": sum(1 for a, b in pairs
                               if a.authoritative and a.family == b.family),
            "all_families_first_run": min(ran) if ran else None,
            "all_families_last_run": max(ran) if ran else None,
            "n_baselines_all_families": len(runs),
        },
        "totals": _tally(auth),
        "totals_all_families": _tally(all_decisions),
        "by_metric": {
            m: {"verdicts": sum(1 for d in auth if d["metric"] == m),
                "indistinguishable": sum(1 for d in auth
                                         if d["metric"] == m and d["indistinguishable"])}
            for m in METRICS},
        "by_pairing": {
            s: {"verdicts": sum(1 for d in auth if d["pairing"] == s),
                "indistinguishable": sum(1 for d in auth
                                         if d["pairing"] == s and d["indistinguishable"])}
            for s in sorted({d["pairing"] for d in auth})},
        "by_family": {
            f: {"verdicts": sum(1 for d in auth if d["family"] == f),
                "indistinguishable": sum(1 for d in auth
                                         if d["family"] == f and d["indistinguishable"])}
            for f in sorted({d["family"] for d in auth})},
        "decisions": auth,
    }


def default_baselines_dir() -> Path:
    """The gitignored `eval/baselines` this backtest replays.

    `.gitignore:95` excludes the whole directory, so an automod worktree has an
    empty one and a backtest reading its own tree would report zero prior
    verdicts and look like a clean result. Same resolution
    `scripts/eval_trend_stats.py:default_baselines_dir` uses for the same reason:
    `LLOYD_ROOT` wins, else this checkout when it actually has nightly files,
    else the live checkout.
    """
    override = os.environ.get("LLOYD_ROOT")
    if override:
        return Path(override).expanduser() / "eval" / "baselines"
    own = ROOT / "eval" / "baselines"
    if any(own.glob("nightly-*.json")):
        return own
    # Off the account home: a gate's `HOME=<round>/home` makes `~/lloyd` the
    # worktree whose empty baselines sent us here (`app.paths.LIVE_CHECKOUT`).
    from app.paths import LIVE_CHECKOUT
    return LIVE_CHECKOUT / "eval" / "baselines"


def render(rep: dict) -> str:
    t, w = rep["totals"], rep["window"]
    all_t = rep["totals_all_families"]
    lines = [
        "Retrieval-eval CI backtest (#696): what a 95 % interval would have said",
        f"  corpus window : {w['first_run']} .. {w['last_run']}",
        f"                  {w['n_baselines']} baselines,"
        f" {w['transitions']} consecutive same-family transitions",
        f"                  families: {', '.join(w['families'])}",
        f"  rule replayed : |delta| > {rep['move_threshold']} on a metric,"
        " the Step-3 callout rule skills/retrieval-eval shipped with",
        f"  interval      : {int(rep['confidence'] * 100)} % bootstrap,"
        f" {rep['n_resamples']} resamples, seed {rep['seed']};"
        " paired iff query ids AND corpus provenance match",
        "",
        f"  {t['indistinguishable']} of {t['prior_verdicts']} prior"
        f" {rep['move_threshold']}-movement verdicts would have been called"
        " INDISTINGUISHABLE by a 95 % CI",
        f"    entity-side only  : {t['entity_side_indistinguishable']} of"
        f" {t['entity_side_verdicts']}",
        f"    paired form only  : {t['paired_indistinguishable']} of"
        f" {t['paired_verdicts']}",
        f"    every label on disk: {all_t['indistinguishable']} of"
        f" {all_t['prior_verdicts']}"
        f"  ({w['n_baselines_all_families']} baselines incl. one-off experiment arms)",
        "",
    ]
    for state, counts in sorted(rep["by_pairing"].items()):
        lines.append(f"  {state:<24} {counts['indistinguishable']:>3} indistinguishable"
                     f" / {counts['verdicts']:<3} verdicts")
    lines.append("")
    for family, counts in rep["by_family"].items():
        lines.append(f"  {family:<24} {counts['indistinguishable']:>3} indistinguishable"
                     f" / {counts['verdicts']:<3} verdicts")
    lines.append("")
    for metric, counts in rep["by_metric"].items():
        if counts["verdicts"]:
            lines.append(f"  {metric:<24} {counts['indistinguishable']:>3} of"
                         f" {counts['verdicts']:<3} indistinguishable")
    lines.append("")
    lines.append("  the verdicts an interval WOULD still have called out:")
    shown = 0
    for d in rep["decisions"]:
        if d["indistinguishable"]:
            continue
        shown += 1
        lines.append(
            f"    {d['metric']:<22} {d['prev_rate']:.3f} -> {d['cur_rate']:.3f}"
            f"  delta {d['delta']:+.3f}  ci [{d['lo']:+.3f},{d['hi']:+.3f}]"
            f"  n={d['n']:<3d} {d['pairing']:<20} {d['family']}")
    if not shown:
        lines.append("    (none — at this n the eval decided nothing at all)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baselines", default=None,
                    help="directory of baseline JSON (default: the live gitignored dir)")
    ap.add_argument("--out", default=None,
                    help="write the full report as JSON here (runtime state; not committed)")
    ap.add_argument("--families", choices=("authoritative", "all"), default="authoritative",
                    help="which labels count as prior verdicts (default: the ones that"
                         " authored report lines: nightly + the *-check gate arms)")
    ap.add_argument("--quiet", action="store_true", help="print only the headline count")
    args = ap.parse_args(argv)

    baselines = Path(args.baselines).expanduser() if args.baselines else default_baselines_dir()
    if not baselines.is_dir():
        print(f"[error] no baseline directory at {baselines}", file=sys.stderr)
        return 2
    runs = load_runs(baselines)
    if not runs:
        # An empty directory is not a clean result — name the directory that was
        # read, so nobody reads "0 of 0" as "the gates were never fooled". The
        # gitignored-baselines-in-a-worktree trap is what `default_baselines_dir`
        # exists to avoid; this message is its second line of defence.
        print(f"[error] no usable baselines in {baselines}", file=sys.stderr)
        return 2
    pairs = transitions(runs)
    decisions = [d for pair in pairs for d in redecide(*pair)]
    if args.families == "authoritative":
        headline = [d for d in decisions if d["authoritative"]]
        if not any(r.authoritative for r in runs):
            # Zero *decisions* is a legitimate answer — it means nothing moved
            # past the threshold. Zero authoritative BASELINES means the
            # directory held only experiment arms, which is a wrong --baselines,
            # not a finding, and must not print as "0 of 0".
            print(f"[error] no nightly/*-check baselines in {baselines}; rerun with "
                  "--families all to include every label on disk", file=sys.stderr)
            return 2
    else:
        headline = decisions
    rep = summarise(headline, decisions, runs, pairs)
    rep["baselines_dir"] = str(baselines)
    rep["families_mode"] = args.families
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, sort_keys=False) + "\n")
        print(f"[info] wrote {out}")
    if args.quiet:
        t = rep["totals"]
        print(f"{t['indistinguishable']} of {t['prior_verdicts']}")
    else:
        print(render(rep))
        print(f"\n[info] baselines: {baselines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
