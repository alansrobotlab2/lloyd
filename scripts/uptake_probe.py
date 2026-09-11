#!/usr/bin/env python3
"""Run the #552 uptake probe: mine disputes, grade the grader, emit the table.

    .venvs/lloyd/bin/python -m scripts.uptake_probe                 # full run
    .venvs/lloyd/bin/python -m scripts.uptake_probe --eval-only     # classifier only
    .venvs/lloyd/bin/python -m scripts.uptake_probe --days 45 --dry-run

The order is the item's order and it is not reorderable: step 2 says **stop
here if precision < 0.70**, so the labeled evaluation runs first and a table is
never written on top of a classifier that cannot tell a correction from a new
request. Below the floor this exits 3 and emits only the classifier report —
that report is the honest artifact, and there is deliberately no uptake table
for a run that failed the gate.

Two costs worth knowing before scheduling it nightly: every candidate is one
request to the secondary engine (roughly a second each at `max_tokens=16`), and
the labeled eval is bounded by the committed corpus, not by traffic, so its
runtime does not grow with usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import uptake  # noqa: E402


def build_corpus(days: int = 30) -> list[uptake.Turn]:
    return uptake.human_turns(days=days)


def _corpus_index(days: int = 900) -> dict[str, uptake.Turn]:
    """Every human turn, keyed by `turn_id`.

    Deliberately ignores the attribution window: a label committed in September
    must resolve to the same turn in December, and resolving labels through the
    rolling 30-day window would silently drop them and quietly re-weight the
    precision figure the acceptance criterion is quoted on.

    An empty transcript store raises rather than returning `{}`. Zero turns is
    not "no disputes"; it is "the measurement did not run", and the second
    reading is the one that has repeatedly gotten this repo into trouble.
    """
    index = {t.turn_id: t for t in uptake.human_turns(days=days)}
    if not index:
        raise RuntimeError(
            f"no human-authored turns found under {uptake.lloyd_root()}/sessions — "
            "refusing to report an uptake result over an empty corpus")
    return index


def _classify_cached(turn: uptake.Turn, cache: dict[str, Any]) -> Any:
    tid = turn.turn_id
    if tid not in cache:
        cache[tid] = uptake.classify_dispute(turn.prev_assistant, turn.user_text)
    return cache[tid]


def run_classifier_eval(cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score the classifier against the hand-labeled corpus.

    Returns the metrics plus the per-item detail, so a reader can open any
    disagreement in the transcript instead of taking the number. `engine` says
    how many items the secondary slot refused to answer — those are excluded
    from both classes, which is why `n` and `n_labeled` differ when the engine
    was flapping.
    """
    cache = {} if cache is None else cache
    labels = uptake.load_labels()
    index = _corpus_index()
    per_item: list[dict[str, Any]] = []
    labels_used = 0
    missing: list[str] = []

    for item in labels:
        turn = index.get(item["turn_id"])
        if turn is None:
            missing.append(item["turn_id"])
            continue
        verdict = _classify_cached(turn, cache)
        if verdict is None:
            per_item.append({**item, "predicted": None, "scored": False})
            continue
        labels_used += 1
        per_item.append({
            **item, "predicted": 1 if verdict else 0,
            "scored": True,
            "correct": (1 if verdict else 0) == int(item["label"]),
        })

    scored = [p for p in per_item if p["scored"]]
    metrics = uptake.precision_recall([int(p["label"]) for p in scored],
                                      [int(p["predicted"]) for p in scored])
    metrics["n_labeled"] = len(labels)
    metrics["n_scored"] = len(scored)
    metrics["n_unanswered"] = len(per_item) - len(scored)
    metrics["n_labels_unresolvable"] = len(missing)
    # Both ends of the matrix are gated: an always-NOT grader scores precision
    # 1.00 on this corpus and would attribute no disputes to anything.
    passed = bool(
        metrics["measured"]
        and (metrics["precision"] or 0) >= uptake.PRECISION_FLOOR
        and (metrics["recall"] or 0) >= uptake.RECALL_FLOOR
    )
    return {
        "metrics": metrics,
        "passed": passed,
        "floors": {"precision": uptake.PRECISION_FLOOR, "recall": uptake.RECALL_FLOOR},
        "engine": uptake.SECONDARY_MODEL,
        # The prompt's exemplars were authored against this corpus's *shapes*
        # (none of its turns are quoted in it), so this is an in-sample estimate
        # and must travel with that label rather than read as a clean holdout.
        "prompt_tuned_on_labels": True,
        "labels_file": str(uptake.labels_path() or ""),
        "unresolvable_turn_ids": missing[:20],
        "per_item": per_item,
    }


def run(days: int, cache: dict[str, Any]) -> tuple[dict[str, Any], list[uptake.Turn]]:
    """Classify the window's candidates and join them to the entries in force."""
    turns = build_corpus(days=days)
    candidates = uptake.candidate_disputes(turns)
    flags = {t.ordinal: _classify_cached(t, cache) for t in candidates}
    table = uptake.build_uptake_table(
        turns=turns,
        dispute_flags=flags,
        memory_entries=uptake.memory_entries(),
        skills_read=uptake.skills_read_by_session(),
        active_skills=uptake.active_skill_names(),
    )
    table["corpus"]["candidates_screened"] = len(candidates)
    table["corpus"]["note"] = (
        "Only cue-screened turns are sent to the classifier, so dispute counts "
        "are a lower bound on the window; the screen's own recall is measured "
        "against the hand labels in the classifier block."
    )
    return table, candidates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=30, help="attribution window (default 30)")
    ap.add_argument("--eval-only", action="store_true", help="grade the classifier and stop")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write the table")
    ap.add_argument("--out-dir", default=None, help="default <repo>/eval/uptake")
    ap.add_argument("--ignore-precision-floor", action="store_true",
                    help="emit a table even below the 0.70 floor (audit only)")
    args = ap.parse_args(argv)

    cache: dict[str, Any] = {}
    report = run_classifier_eval(cache)
    m = report["metrics"]
    prec = m["precision"]
    print(f"classifier: precision={prec if prec is None else round(prec, 3)} "
          f"recall={m['recall'] if m['recall'] is None else round(m['recall'], 3)} "
          f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']} "
          f"n_positives={m['n_positives']} scored={m['n_scored']}/{m['n_labeled']}")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO / "eval" / "uptake")
    classifier_block = {
        "engine": report["engine"], "labels_file": report["labels_file"],
        "precision": m["precision"], "recall": m["recall"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        "n_positives": m["n_positives"], "n_scored": m["n_scored"],
        "n_labeled": m["n_labeled"], "n_unanswered": m["n_unanswered"],
        "passed": report["passed"], "floors": report["floors"],
        "prompt_tuned_on_labels": report["prompt_tuned_on_labels"],
        "note": "recall ~0.5 means dispute counts in this table are a LOWER BOUND, "
                "not a census",
    }

    if report["passed"]:
        gate = uptake.retrieval_gate()
        classifier_block["retrieval_gate"] = {
            k: gate[k] for k in ("nights", "latest", "hardcoded_gate",
                                 "hardcoded_gate_would_have_failed_nights") if k in gate
        }
        for metric in ("doc_hit_rate", "ndcg10"):
            if metric in gate:
                classifier_block["retrieval_gate"][metric] = gate[metric]

    if args.eval_only:
        return 0 if report["passed"] else 3

    if not report["passed"]:
        print(f"STOP: precision below the {uptake.PRECISION_FLOOR} floor — no uptake "
              f"table written. Fix the classifier before attributing disputes.",
              file=sys.stderr)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "classifier-report.json").write_text(
                json.dumps({"classifier": classifier_block,
                            "note": "emitted because the run did not clear the floor; "
                                    "no uptake table exists for this run"}, indent=2) + "\n")
        return 3

    table, candidates = run(args.days, cache)
    stores = uptake.store_sizes()
    if args.dry_run:
        print(json.dumps({"table": table, "stores": stores}, indent=2, default=str)[:6000])
        return 0

    path = uptake.write_table(
        table, out_dir=out_dir, classifier=classifier_block, stores=stores,
        extra={"glossary": {
            "dispute_rate": "disputes / present_in_turns. An UPPER BOUND for "
                            "always-in-force memory entries, since those are in every prompt.",
            "weighted_disputes": "dispute counts discounted by lexical overlap between the "
                                 "entry text and the correction — the usable signal for memory "
                                 "entries, and a proxy for the embedding weight the item "
                                 "prescribes.",
            "presence_source": "which evidence says the entry was in force. Evidence-bound "
                              "for skills and notes; always_in_force for USER.md/MEMORY.md.",
        }})
    print(f"wrote {path} ({len(table['entries'])} entries, "
          f"{table['corpus']['disputes']} disputes over {table['corpus']['turns']} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
