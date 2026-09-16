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


def collect_turns(days: int = 30) -> list[uptake.Turn]:
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


#: The shape of "this metric does not exist", as opposed to a metric that exists
#: and is zero. `{}` would be indistinguishable from a crash, and `None` in a
#: precision field is already what "no positives" means downstream.
_UNMEASURED: dict[str, Any] = {"measured": False, "precision": None, "recall": None,
                               "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                               "n": 0, "n_positives": 0}


def _zero_shot_pass(turn_ids: set[str], index: dict[str, uptake.Turn],
                    label_of: dict[str, int],
                    transport: Any = None) -> dict[str, Any]:
    """Re-grade the holdout turns with the few-shot block removed.

    The exemplars were written against this corpus, so every precision figure
    measured WITH them is in-sample no matter how the labels are split. Removing
    them is the only variant that cannot have read the answers, and it is what the
    stop condition is actually gated on. Cost is one request per holdout turn
    (~40 requests, ~6s against the secondary slot), which is why it runs once per
    probe rather than per candidate.

    An engine that will not answer stays `measured: False` — an unavailable grader
    must not be reported as a precision of 0.0 (which would fail the gate for the
    wrong reason) nor as 1.0 (which would pass it for no reason at all).
    """
    labs: list[int] = []
    preds: list[int] = []
    unanswered = 0
    for tid in sorted(turn_ids):
        turn = index.get(tid)
        if turn is None:
            continue
        verdict, _ = uptake.classify_dispute_raw(
            turn.prev_assistant, turn.user_text, examples=False, transport=transport)
        label = label_of.get(tid)
        if label is None:
            continue
        if verdict is None:
            unanswered += 1
            preds.append(0)
        else:
            preds.append(1 if verdict else 0)
        labs.append(label)
    if not labs or sum(labs) == 0:
        out = dict(_UNMEASURED)
        out["unanswered"] = unanswered
        return out
    out = uptake.precision_recall(labs, preds)
    out["unanswered"] = unanswered
    return out


def _classify_cached(turn: uptake.Turn, cache: dict[str, Any]) -> Any:
    tid = turn.turn_id
    if tid not in cache:
        cache[tid] = uptake.classify_dispute(turn.prev_assistant, turn.user_text)
    return cache[tid]


#: How many labeled turns a replay must re-ask before "no mismatch" means
#: something. Fewer and the sentence reads as "we happened to look at the six
#: turns that agree".
REPLAY_MIN_CHECKED = 6


def verify_replay(transport: Any = None, *,
                  min_checked: int = REPLAY_MIN_CHECKED) -> dict[str, Any]:
    """Re-ask a sample of labeled turns and compare against the committed `engine_raw`.

    `test_recorded_engine_replies_reproduce_the_measured_precision` proves the
    recorded matrix reproduces *itself*, which says nothing about whether the
    engine still answers that way. The engine runs temperature 0 on a pinned slot,
    so a mismatch here is the model or the prompt having moved — which is exactly
    when a committed precision figure stops being re-quotable and the artifact
    needs saying so, not quietly re-reading as current.

    This lives in the probe and not in the test suite. A reproducibility check
    that asks the live model is a property of the measurement run, which is awake
    by definition there; inside the suite it turns a hermetic replay claim into a
    dependency on which GGUF happens to be loaded, and a round that caught the
    slot asleep could not tell a moved model from an offline box. `transport` is
    injectable so the *comparison* is still pinned by a test that never POSTs.

    A sample that cannot be re-asked is reported as `measured: False`, never as a
    pass: zero re-asks and zero mismatches are the same two numbers, and the
    second one is what a green row is built on.
    """
    labels = uptake.load_labels()
    index = _corpus_index()
    checked = 0
    mismatched: list[dict[str, Any]] = []
    unanswered: list[str] = []
    for item in labels:
        raw = item.get("engine_raw")
        turn = index.get(item["turn_id"])
        if raw is None or turn is None:
            continue
        if checked >= min_checked:
            break
        verdict, fresh = uptake.classify_dispute_raw(
            turn.prev_assistant, turn.user_text, transport=transport)
        want = uptake._parse_verdict(raw)
        if verdict is None:
            unanswered.append(item["turn_id"])
        elif verdict != want:
            mismatched.append({"turn_id": item["turn_id"],
                               "recorded": raw, "now": fresh})
        checked += 1
    measured = checked >= min_checked
    return {
        "measured": measured,
        "checked": checked,
        "min_checked": min_checked,
        "mismatched": mismatched,
        "unanswered": unanswered,
        # `ok` is the only field a reader should act on: measured AND no
        # disagreement AND nothing left unanswered. A replay over an engine that
        # refused half the sample is not a confirmation.
        "ok": bool(measured and not mismatched and not unanswered),
        "note": (
            f"{checked} labeled turns re-asked against their committed engine_raw, "
            f"{len(mismatched)} moved, {len(unanswered)} unanswered"
            if measured else
            f"only {checked} of {min_checked} labeled turns could be re-asked, so the "
            "committed engine_raw replies are UNVERIFIED against the current engine"
        ),
    }


def _repo_relative(path: str | Path) -> str:
    """Paths inside the checkout are emitted relative to it.

    The classifier report is committed as evidence, and an absolute path baked
    at run time outlives the tree it points into: the first artifact recorded
    `labels_file` inside `~/lloyd-work/SM_20260911_031441/…`, a worktree that
    `automod_abort` deletes. An evidence pointer that resolves to nothing is
    worse than no pointer — it looks auditable and is not. Paths outside the
    checkout (an override dir) stay absolute, because relative-to-what would be
    a guess.
    """
    p = Path(path)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def run_classifier_eval(cache: dict[str, Any] | None = None, *,
                        record_raw: bool = False) -> dict[str, Any]:
    """Score the classifier against the hand-labeled corpus.

    Returns the metrics plus the per-item detail, so a reader can open any
    disagreement in the transcript instead of taking the number. `engine` says
    how many items the secondary slot refused to answer — those are excluded
    from both classes, which is why `n` and `n_labeled` differ when the engine
    was flapping.

    `record_raw=True` additionally stores the engine's literal reply per item
    (bypassing the cache, which holds verdicts only). That is what lets the
    measurement be replayed offline: the suite re-parses recorded replies
    through `app.uptake._parse_verdict` and recomputes precision, so the
    committed number has a check that does not depend on the engine being awake,
    and does not trust the fixture's own `predicted` field.
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
        if record_raw:
            verdict, raw = uptake.classify_dispute_raw(
                turn.prev_assistant, turn.user_text)
        else:
            verdict, raw = _classify_cached(turn, cache), ""
        if verdict is None:
            per_item.append({**item, "predicted": None, "scored": False,
                             **({"engine_raw": raw} if record_raw else {})})
            continue
        labels_used += 1
        per_item.append({
            **item, "predicted": 1 if verdict else 0,
            "scored": True,
            "correct": (1 if verdict else 0) == int(item["label"]),
            **({"engine_raw": raw} if record_raw else {}),
        })

    scored = [p for p in per_item if p["scored"]]
    metrics = uptake.precision_recall([int(p["label"]) for p in scored],
                                      [int(p["predicted"]) for p in scored])
    metrics["n_labeled"] = len(labels)
    metrics["n_scored"] = len(scored)
    metrics["n_unanswered"] = len(per_item) - len(scored)
    metrics["n_labels_unresolvable"] = len(missing)

    # --- label integrity ----------------------------------------------------
    # `labeled_by` is a string the labeler wrote, so "hand-labeled" proves nothing
    # on its own. Re-anchor every label to the transcript and refuse to score a
    # corpus that does not resolve.
    label_check = uptake.validate_labels(labels, index)

    # --- holdout: contamination measured, not asserted ----------------------
    split = uptake.holdout_split(labels)
    hold_ids = {i["turn_id"] for i in split["holdout"]}
    hold_scored = [p for p in scored if p["turn_id"] in hold_ids]
    holdout_metrics = uptake.precision_recall(
        [int(p["label"]) for p in hold_scored],
        [int(p["predicted"]) for p in hold_scored]) if hold_scored else dict(_UNMEASURED)
    holdout_metrics.update({
        "n_holdout": split["n_holdout"], "n_dev_contaminated": split["n_dev"],
        "n_positives_holdout": split["n_holdout_positives"],
    })

    # --- zero-shot: the few-shot block taken away ---------------------------
    # The exemplars encode this corpus's shapes, so precision with them is an
    # in-sample number. Without this pass the artifact could only say so.
    zero_shot = _zero_shot_pass(hold_ids, index,
                                {i["turn_id"]: int(i["label"]) for i in labels})

    # --- the deployed pipeline, screen included -----------------------------
    # Parallel arrays over the RESOLVED labels only, and built from one list: the
    # verdicts come from `per_item`, which skips labels whose turn no longer
    # resolves, so zipping them against `labels` would misalign every row after the
    # first unresolvable one and silently move disputes between classes.
    screened_ids = {t.turn_id for t in uptake.candidate_disputes(list(index.values()))}
    resolved = [i for i in labels if i["turn_id"] in index]
    verdict_by_id = {p["turn_id"]: (p["predicted"] if p.get("scored") else None)
                     for p in per_item}
    pipeline = uptake.pipeline_confusion(
        [int(i["label"]) for i in resolved],
        [1 if i["turn_id"] in screened_ids else 0 for i in resolved],
        [verdict_by_id.get(i["turn_id"]) for i in resolved],
    )

    # Both ends of the matrix are gated: an always-NOT grader scores precision
    # 1.00 on this corpus and would attribute no disputes to anything. The stop
    # condition is `app.uptake.measurement_clears_floors`, shared with the tests —
    # an inline copy of the gate here would leave the gate itself untested.
    gate_report = {
        "labels_ok": label_check["ok"],
        "classifier": metrics, "holdout": holdout_metrics, "zero_shot": zero_shot,
        "pipeline": pipeline,
    }
    passed = uptake.measurement_clears_floors(gate_report)
    return {
        "metrics": metrics,
        "passed": passed,
        "labels_check": label_check,
        "labels_ok": label_check["ok"],
        "holdout": holdout_metrics,
        "zero_shot": zero_shot,
        "pipeline": pipeline,
        "split_note": split["note"],
        "floors": {"precision": uptake.PRECISION_FLOOR, "recall": uptake.RECALL_FLOOR},
        "engine": uptake.SECONDARY_MODEL,
        # The prompt's exemplars were authored against this corpus's shapes and
        # some quote a labeled turn's own words, so THIS block is an in-sample
        # estimate. It is not the whole measurement: `holdout` re-scores the turns
        # the prompt does not quote and `zero_shot` re-scores them with the block
        # removed, and `app.uptake.measurement_clears_floors` will not pass unless
        # all three clear. A reader who quotes only this number is quoting the one
        # that flatters the prompt.
        "prompt_tuned_on_labels": True,
        "labels_file": _repo_relative(uptake.labels_path() or ""),
        "unresolvable_turn_ids": missing[:20],
        # The disagreements, by turn id. Counts alone let a report say
        # "precision 1.00, fp 0" and be unverifiable by anything shipped with it;
        # with the ids listed, a reader (or a test) can recompute the matrix from
        # the labels and open every disagreement in the transcript. The full
        # per-item detail stays out of the committed table on purpose: it would
        # quote ~46 labeled user turns, including Alan's corrections, into git.
        "disagreements": {
            "false_positive": [p["turn_id"] for p in scored
                               if int(p["label"]) == 0 and int(p["predicted"]) == 1],
            "false_negative": [p["turn_id"] for p in scored
                               if int(p["label"]) == 1 and int(p["predicted"]) == 0],
        },
        "per_item": per_item,
    }


def run(days: int, cache: dict[str, Any]) -> tuple[dict[str, Any], list[uptake.Turn]]:
    """Classify the window's candidates and join them to the entries in force."""
    turns = collect_turns(days=days)
    candidates = uptake.candidate_disputes(turns)
    # Verdicts are keyed on `turn_id`, never on `ordinal`. `Turn.ordinal` restarts
    # at 1 in every session — the live 30-day window is 220 turns over 132
    # sessions occupying ordinals 1-10 — so an ordinal-keyed dict collapses 24
    # verdicts into 9 keys, and the join then charges every session that has a
    # turn at that position with someone else's dispute. `build_uptake_table`
    # refuses keys that are not turn_ids of the turns it was handed.
    flags = {t.turn_id: _classify_cached(t, cache) for t in candidates}
    tally: dict[str, dict[str, int]] = {}
    table = uptake.build_uptake_table(
        turns=turns,
        dispute_flags=flags,
        memory_entries=uptake.memory_entries(tally=tally),
        skills_read=uptake.skills_read_by_session(),
        active_skills=uptake.active_skill_names(),
        memory_tally=tally,
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
    ho, zs, pl = report["holdout"], report["zero_shot"], report["pipeline"]

    def _r(v: Any) -> Any:
        return None if v is None else round(v, 3)

    print(f"classifier(in-sample, deployed prompt): precision={_r(m['precision'])} "
          f"recall={_r(m['recall'])} tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']} "
          f"n_positives={m['n_positives']} scored={m['n_scored']}/{m['n_labeled']}")
    print(f"holdout(prompt does not quote it):      precision={_r(ho.get('precision'))} "
          f"recall={_r(ho.get('recall'))} n={ho.get('n', 0)} "
          f"positives={ho.get('n_positives', 0)} contaminated={ho.get('n_dev_contaminated', 0)}")
    print(f"zero-shot(few-shot block removed):      precision={_r(zs.get('precision'))} "
          f"recall={_r(zs.get('recall'))} n={zs.get('n', 0)} "
          f"positives={zs.get('n_positives', 0)}")
    print(f"deployed pipeline(screen + classifier): precision={_r(pl['precision'])} "
          f"recall={_r(pl['recall'])} screened={pl.get('screened')} "
          f"tp={pl['tp']} fp={pl['fp']} fn={pl['fn']} tn={pl['tn']}")
    print(f"labels re-anchored to transcripts: ok={report['labels_ok']} "
          f"({report['labels_check']['n_resolved']}/{report['labels_check']['n']} resolved, "
          f"{len(report['labels_check']['excerpt_mismatch_turn_ids'])} mismatched)")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO / "eval" / "uptake")
    classifier_block = {
        "engine": report["engine"], "labels_file": report["labels_file"],
        "precision": m["precision"], "recall": m["recall"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        "n_positives": m["n_positives"], "n_scored": m["n_scored"],
        "n_labeled": m["n_labeled"], "n_unanswered": m["n_unanswered"],
        "passed": report["passed"], "floors": report["floors"],
        "prompt_tuned_on_labels": report["prompt_tuned_on_labels"],
        # The three numbers a reader must compare before trusting the two above.
        "holdout": {k: v for k, v in ho.items()},
        "zero_shot": {k: v for k, v in zs.items()},
        "deployed_pipeline": {k: v for k, v in pl.items()},
        "labels_check": report["labels_check"],
        "split_note": report["split_note"],
        "disagreements": report.get("disagreements", {}),
        # Derived, never typed: a hard-coded "recall ~0.5" in a template is a
        # number that gets re-quoted forever and was wrong the first time the
        # model moved. This clause is the reason a dispute count must be read as
        # a lower bound, so it has to describe *this* run — and it has to describe
        # the run DEPLOYED, not the model in isolation. Only cue-screened turns
        # reach the classifier, so the loss that bounds the table is the pipeline's
        # recall (measured 0.43 on the holdout, against the classifier's 0.52);
        # quoting the classifier's number here understated the loss by a third.
        "note": (
            f"deployed pipeline recall {pl['recall']:.2f} (classifier alone "
            f"{m['recall']:.2f}) means dispute counts in this table are a LOWER "
            "BOUND, not a census"
            if m["recall"] is not None and pl["recall"] is not None else
            "recall unmeasurable (no labeled positives answered), so dispute "
            "counts here mean nothing"
        ),
    }

    if report["passed"]:
        replay = verify_replay()
        classifier_block["replay"] = replay
        print(f"replay(committed engine_raw re-asked):  measured={replay['measured']} "
              f"checked={replay['checked']} moved={len(replay['mismatched'])} "
              f"unanswered={len(replay['unanswered'])} ok={replay['ok']}")
        if not replay["ok"]:
            # Loud, but not a stop: this run has just measured precision against
            # the engine it has now, so the NEW number is valid. What a mismatch
            # invalidates is the figure in the previously committed artifact, and
            # the reader has to be told that from the artifact that replaces it.
            print(f"WARNING: {replay['note']}", file=sys.stderr)
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
        which = ("labels did not re-resolve to their transcripts"
                 if not report["labels_ok"] else
                 "deployed classifier below a floor"
                 if not uptake.classifier_clears_floors(m) else
                 "holdout and/or zero-shot precision below "
                 f"{uptake.PRECISION_FLOOR} — the in-sample number is not trusted")
        print(f"STOP: {which}. No uptake table written. Fix the classifier before "
              f"attributing disputes.", file=sys.stderr)
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
