#!/usr/bin/env python3
"""
eval/run_skill_match_eval.py — recall of the prefetch skill matcher (#557).

Scores `eval/skill_match_queries.yaml` — real user turns labelled with the
skill(s) that turn's job belongs to — against the production lexical matcher:
`prefetch._search_skills` as prefetch calls it, token overlap over
name/description/tags with the #311 metadata-hit gate and the
SKILL_THRESHOLD_* floors. That function decides injection today, and it is
imported here rather than reimplemented, so this number moves when it does.

Reported:

  recall@5            |expected ∩ top5| / |expected|, averaged over the turns
                      that have an expected skill
  precision@5         |expected ∩ top5| / 5 over the same turns (classic P@5)
  no_match_rate       fraction of labelled turns where the matcher returned
                      NOTHING at all — the population that gets "No skills
                      matched automatically"
  expected_empty_named  over the expected-empty turns: how often the matcher
                      names a skill where nothing should be named (the #311
                      precision control)

The set was built on branch automod/SM_20260911_055006 together with an
embedding "geometric" arm. Alan's 2026-09-14 ruling kept the labelled set and
this scorer and dropped the arm: at its shipped 0.62 floor it scored recall@5
0.6863 against lexical 0.7451, so the +0.147 it was landed on was a floor
artifact. Nothing here imports `agent_mcp.skill_cards`, and nothing may.

No model, no GPU, no daemon. It reads the skill inventory the matcher reads.

    .venvs/lloyd/bin/python eval/run_skill_match_eval.py                   # score
    .venvs/lloyd/bin/python eval/run_skill_match_eval.py --write-baseline  # record
    .venvs/lloyd/bin/python eval/run_skill_match_eval.py --compare         # diff

The baseline is runtime data, so it lives under `app.paths.EVAL_BASELINES_DIR`
(`skill_match_baseline.json`), never in the tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERIES = ROOT / "eval" / "skill_match_queries.yaml"
BASELINE_NAME = "skill_match_baseline.json"
K = 5
#: A lexical recall@5 move larger than this against the recorded baseline is
#: reported as a regression by --compare.
DRIFT_TOLERANCE = 0.05

Arm = Callable[[str], list[str]]


def baseline_path() -> Path:
    from app.paths import EVAL_BASELINES_DIR
    return EVAL_BASELINES_DIR / BASELINE_NAME


def lexical_top5(turn: str) -> list[str]:
    """The production lexical selection, imported rather than reimplemented."""
    import prefetch
    # The injectable set, as before #435 widened `_search_skills` to every offer.
    scored = prefetch._injectable_skills(
        prefetch._search_skills(prefetch._query_tokens(turn)))
    return [skill["name"] for _score, skill in scored[:K]]


def pseudo_query_arm(path: str | Path, weight: float = 3.0,
                     mode: str = "union") -> Arm:
    """#1490 arm (a): the production matcher with `skills.pseudo_queries` on.

    The scorer code is production's (`agent_mcp.skills._score_skill` via
    `prefetch._search_skills`); only the index it reads is handed in, so the
    arm needs no config edit and cannot leak into another arm.
    """
    return with_pseudo_queries(lexical_top5, path, weight, mode)


def with_pseudo_queries(arm: Arm, path: str | Path | None, weight: float = 3.0,
                        mode: str = "union") -> Arm:
    """Run `arm` with prefetch reading this pseudo-query index; `None` = as is."""
    if not path:
        return arm
    import prefetch
    from agent_mcp import skills as S
    path = Path(path)

    def index() -> dict:
        key, table = S._load_pseudo_queries(path)
        return {"weight": float(weight), "mode": mode, "path": path,
                "key": key, "table": table}

    def wrapped(turn: str) -> list[str]:
        saved = prefetch.pseudo_query_index
        prefetch.pseudo_query_index = index
        try:
            return arm(turn)
        finally:
            prefetch.pseudo_query_index = saved
    return wrapped


#: djev latencies (ms) and failures seen by the rerank arm, for the report.
RERANK_STATS: dict = {"latency_ms": [], "failed": 0, "calls": 0}


def skill_candidate_text(skill: dict) -> str:
    """What the reranker reads per skill: name, description, then the body."""
    desc = " ".join(str(skill.get("description") or "").split())
    return f"{skill.get('name')}: {desc}\n{skill.get('body') or ''}"


def djev_rerank_arm(k: int = 16, chars: int = 400, pool: str = "injectable") -> Arm:
    """#1490 arm (b): djev listwise over the lexical top-`k` skills' bodies.

    `pool="injectable"` reorders only what the injection gate already lets
    through (so the gate, and the expected-empty control, are untouched);
    `pool="offers"` reranks the top-`k` offers above `SKILL_REPORT_FLOOR` and
    keeps the ones the gate passes, in djev's order. A djev that does not
    answer falls back to the lexical order and is counted.
    """
    import time as _t

    import prefetch
    from app import djev

    def arm(turn: str) -> list[str]:
        scored = prefetch._search_skills(prefetch._query_tokens(turn))
        injectable = prefetch._injectable_skills(scored)
        gate = {id(sk) for _, sk in injectable}
        cands = (injectable if pool == "injectable" else scored)[:k]
        if len(cands) <= 1:
            return [sk["name"] for _, sk in cands if id(sk) in gate][:K]
        RERANK_STATS["calls"] += 1
        t0 = _t.perf_counter()
        rows = djev.rank(turn[:1500], [skill_candidate_text(sk) for _, sk in cands],
                         chars=chars, seam="skill_rerank_eval", max_n=max(k, 2))
        RERANK_STATS["latency_ms"].append((_t.perf_counter() - t0) * 1000)
        if rows is None:
            RERANK_STATS["failed"] += 1
            order = cands
        else:
            order = [cands[r["index"]] for r in rows]
        return [sk["name"] for _, sk in order if id(sk) in gate][:K]
    return arm


def paired(base: dict, arm: dict, metric: str) -> dict | None:
    """Paired bootstrap of `arm - base` on one per-turn metric (eval/stats.py)."""
    from eval.stats import paired_bootstrap_ci
    a = {t["id"]: t[metric] for t in base["turns"] if metric in t}
    b = {t["id"]: t[metric] for t in arm["turns"] if metric in t}
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    ci = paired_bootstrap_ci([a[i] for i in ids], [b[i] for i in ids])
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ci.items()}


def score(records: Iterable[dict], arm: Arm, k: int = K) -> dict:
    """Score one arm over labelled records. Pure: the arm is the only input
    that reaches outside, which is what lets a fixture pin the definitions."""
    recall_sum = 0.0
    precision_sum = 0.0
    labelled = no_match = hit = named = empties = hit1 = 0
    rr_sum = 0.0
    per_turn = []
    for rec in records:
        expected = set(rec.get("expected_skills") or [])
        got = list(arm(rec["turn"]))[:k]
        row = {"id": rec["id"], "returned": got}
        per_turn.append(row)
        if expected:
            labelled += 1
            hits = len(expected & set(got))
            recall_sum += hits / len(expected)
            precision_sum += hits / k
            hit += bool(hits)
            no_match += not got
            first = bool(got) and got[0] in expected
            hit1 += first
            rr = next((1.0 / (i + 1) for i, g in enumerate(got) if g in expected), 0.0)
            rr_sum += rr
            row.update({"recall": hits / len(expected), "hit1": float(first), "rr": rr})
        elif rec.get("expected_empty"):
            empties += 1
            named += bool(got)
            row["named"] = float(bool(got))

    def rate(n: float, d: int):
        return round(n / d, 4) if d else None

    return {
        "labelled_turns": labelled,
        "expected_empty_turns": empties,
        "recall@5": rate(recall_sum, labelled),
        "precision@5": rate(precision_sum, labelled),
        "hit@5": rate(hit, labelled),
        # hit@1: the skill that gets its full body injected is a right one (#1490).
        "hit@1": rate(hit1, labelled),
        "mrr@5": rate(rr_sum, labelled),
        "no_match_rate": rate(no_match, labelled),
        "no_match_count": no_match,
        "expected_empty_named_rate": rate(named, empties),
        "expected_empty_named_count": named,
        "turns": per_turn,
    }


def load_records(path: Path = QUERIES) -> list[dict]:
    return yaml.safe_load(path.read_text())["records"]


def active_skill_names() -> set[str]:
    from agent_mcp.skills import _iter_skills
    return {s["name"] for s in _iter_skills()}


def check_labels(records: Iterable[dict], active: set[str]) -> list[str]:
    """An expected skill that no longer exists must fail loudly, not drift."""
    return [f"{rec['id']}: expected skill {name!r} is not an active skill"
            for rec in records
            for name in rec.get("expected_skills") or []
            if name not in active]


def build_report(records: list[dict], lexical: dict) -> dict:
    return {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # tests/test_eval_scorer.py requires every baseline to say whether it
        # measured production's system and to carry the vault-leg knobs. No
        # model is involved and the matcher is imported with its own untouched
        # thresholds, so the flag is true and the vault knobs do not apply.
        "matches_production_defaults": True,
        "graph_rerank": "not_applicable: scores the skill matcher, not the vault leg",
        "rerank_alpha": "not_applicable: scores the skill matcher, not the vault leg",
        "graph_top_k": "not_applicable: scores the skill matcher, not the vault leg",
        "graph_hops": "not_applicable: scores the skill matcher, not the vault leg",
        "matches_production_defaults_note": (
            "lexical arm = prefetch._search_skills with production SKILL_THRESHOLD_*; "
            "no model, no GPU, no config override"),
        "set": str(QUERIES.relative_to(ROOT)),
        "records": len(records),
        "k": K,
        "arms": {"lexical": lexical},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--json", default="", help="also write the full report here")
    ap.add_argument("--write-baseline", action="store_true",
                    help=f"record the report as EVAL_BASELINES_DIR/{BASELINE_NAME}")
    ap.add_argument("--compare", action="store_true",
                    help="diff lexical recall@5 against the recorded baseline")
    ap.add_argument("--pseudo-queries", default="",
                    help="#1490: also score arm (a) against this pseudo-query cache")
    ap.add_argument("--pq-weight", default="3.0",
                    help="comma-separated weights, one pseudo-query arm each")
    ap.add_argument("--pq-mode", default="union", choices=("union", "max"))
    ap.add_argument("--rerank", default="", choices=("", "djev"),
                    help="#1490: also score arm (b), djev listwise over skill bodies")
    ap.add_argument("--rerank-k", type=int, default=16)
    ap.add_argument("--rerank-chars", type=int, default=400)
    ap.add_argument("--rerank-pool", default="injectable", choices=("injectable", "offers"))
    args = ap.parse_args(argv)

    records = load_records()
    active = active_skill_names()
    bad = check_labels(records, active)
    if bad:
        print("STALE LABELS (skill retired/renamed since the set was built):",
              file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 2
    print(f"{len(records)} records, {len(active)} active skills")

    lex = score(records, lexical_top5)
    report = build_report(records, lex)

    def line(name: str, r: dict) -> str:
        return (f"{name} recall@5={r['recall@5']}  precision@5={r['precision@5']}  "
                f"hit@5={r['hit@5']}  hit@1={r['hit@1']}  mrr@5={r['mrr@5']}  "
                f"no_match_rate={r['no_match_rate']} "
                f"({r['no_match_count']}/{r['labelled_turns']})  "
                f"expected_empty_named={r['expected_empty_named_count']}/"
                f"{r['expected_empty_turns']}")
    print(line("lexical", lex))

    extra: dict[str, dict] = {}
    if args.pseudo_queries:
        for w in [float(x) for x in args.pq_weight.split(",") if x.strip()]:
            extra[f"pq_{args.pq_mode}_w{w:g}"] = score(
                records, pseudo_query_arm(args.pseudo_queries, w, args.pq_mode))
    if args.rerank == "djev":
        # With --pseudo-queries the rerank sits on top of the FIRST weight's arm.
        pq_w = float(args.pq_weight.split(",")[0])
        on_pq = f"_on_pq_w{pq_w:g}" if args.pseudo_queries else ""
        extra[f"djev_{args.rerank_pool}_k{args.rerank_k}_c{args.rerank_chars}{on_pq}"] = score(
            records, with_pseudo_queries(
                djev_rerank_arm(k=args.rerank_k, chars=args.rerank_chars,
                                pool=args.rerank_pool),
                args.pseudo_queries or None, pq_w, args.pq_mode))
    for name, r in extra.items():
        print(line(name, r))
        r["vs_lexical"] = {m: paired(lex, r, m) for m in ("recall", "hit1", "rr", "named")}
        for m, ci in r["vs_lexical"].items():
            if ci:
                print(f"    {m:6s} diff={ci['diff']:+.4f} 95% CI [{ci['lo']:+.4f}, "
                      f"{ci['hi']:+.4f}] p={ci['p']:.3f} n={ci['n']}")
        report["arms"][name] = r
    if RERANK_STATS["calls"]:
        lat = sorted(RERANK_STATS["latency_ms"])
        report["rerank_latency_ms"] = {
            "calls": RERANK_STATS["calls"], "failed": RERANK_STATS["failed"],
            "p50": round(lat[len(lat) // 2], 1), "p90": round(lat[int(len(lat) * 0.9)], 1),
            "max": round(lat[-1], 1)}
        print(f"djev rerank latency: {report['rerank_latency_ms']}")

    rc = 0
    base_file = baseline_path()
    if args.compare:
        if not base_file.exists():
            print(f"no baseline at {base_file}", file=sys.stderr)
        else:
            base = json.loads(base_file.read_text())
            b = base["arms"]["lexical"]["recall@5"]
            n = lex["recall@5"]
            drift = round(n - b, 4)
            report["baseline_comparison"] = {
                "baseline_file": str(base_file),
                "baseline_measured_at": base.get("measured_at"),
                "lexical_recall5_baseline": b,
                "lexical_recall5_now": n,
                "drift": drift,
            }
            print(f"vs baseline ({base.get('measured_at')}): lexical recall@5 "
                  f"{b} -> {n} ({drift:+})")
            if abs(drift) > DRIFT_TOLERANCE:
                print(f"REGRESSION: lexical recall@5 moved more than "
                      f"{DRIFT_TOLERANCE} against the baseline", file=sys.stderr)
                rc = 1

    if args.write_baseline:
        base_file.parent.mkdir(parents=True, exist_ok=True)
        base_file.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {base_file}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
