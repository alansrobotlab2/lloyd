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
    scored = prefetch._search_skills(prefetch._query_tokens(turn))
    return [skill["name"] for _score, skill in scored[:K]]


def score(records: Iterable[dict], arm: Arm, k: int = K) -> dict:
    """Score one arm over labelled records. Pure: the arm is the only input
    that reaches outside, which is what lets a fixture pin the definitions."""
    recall_sum = 0.0
    precision_sum = 0.0
    labelled = no_match = hit = named = empties = 0
    per_turn = []
    for rec in records:
        expected = set(rec.get("expected_skills") or [])
        got = list(arm(rec["turn"]))[:k]
        per_turn.append({"id": rec["id"], "returned": got})
        if expected:
            labelled += 1
            hits = len(expected & set(got))
            recall_sum += hits / len(expected)
            precision_sum += hits / k
            hit += bool(hits)
            no_match += not got
        elif rec.get("expected_empty"):
            empties += 1
            named += bool(got)

    def rate(n: float, d: int):
        return round(n / d, 4) if d else None

    return {
        "labelled_turns": labelled,
        "expected_empty_turns": empties,
        "recall@5": rate(recall_sum, labelled),
        "precision@5": rate(precision_sum, labelled),
        "hit@5": rate(hit, labelled),
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
    print(f"lexical recall@5={lex['recall@5']}  precision@5={lex['precision@5']}  "
          f"hit@5={lex['hit@5']}  no_match_rate={lex['no_match_rate']} "
          f"({lex['no_match_count']}/{lex['labelled_turns']})  "
          f"expected_empty_named={lex['expected_empty_named_count']}/"
          f"{lex['expected_empty_turns']}")

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
