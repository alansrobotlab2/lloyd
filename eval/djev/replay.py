#!/usr/bin/env python3
"""Replay recorded decisions through djev and measure how well it separates them.

Five corpora already on disk, no live wiring needed for any of them:

    edges      _pipeline/memory-graph/classified-v4-batch.jsonl      35,802
    entities   _pipeline/memory-graph/semantic-verdicts.jsonl         1,040
    dedupe     ~/.local/state/lloyd-automod/dedupe.jsonl              1,305
    reverted   _pipeline/memory-graph/entity-merges-reverted-*.json     151
    clusters   ~/.local/state/lloyd-automod/cluster_judgments.jsonl     609

WHAT IS AND IS NOT GROUND TRUTH
-------------------------------
`reverted` is the only true ground truth in the set: 151 merges a human
inspected and reverted on 2026-09-03, so every pair in it is a verified
DIFFERENT. Everything else is a model's opinion — `entities` and `clusters`
were written by judges, `edges` by a v4 classifier prompt, `dedupe` by a
reranker score and a Jaccard floor. Those measure AGREEMENT, never
correctness, and this tool says so on every report rather than letting a
0.83 AUC against another model read as accuracy.

WHY AUC AND A RELIABILITY CURVE, NEVER BARE ACCURACY
-----------------------------------------------------
Two reasons, both measured on 2026-09-20.

Accuracy needs a threshold and there is no meaningful fixed one: across four
framings of the same question the optimal cutoff was 0.39, 0.03, 0.30 and
0.01, and `acc@0.5` ranged 0.50-0.80 on a set where AUC barely moved
(0.72-0.83). AUC is threshold-free, which is exactly why it is the headline
here and `acc@0.5` appears only as the number that proves the point.

And most of these corpora are single-class. Of 1,309 clause verdicts in one
week, 1,258 were `met` — "always say met" scores 96% and beats any classifier
you could build. `--balance` samples an equal number per class and the report
states the class counts either way, because an accuracy figure over a 96/4
split is a statement about the split.

THE ORDER-SWAP PROBE
--------------------
`--swap-probe` runs every pair a second time with the two items exchanged and
reports |P_A - P_B|. It measured mean 0.324 / max 0.851 on 40 pairs, which is
the single finding that rules out a fixed threshold and makes option order
part of the schema hash. Run it whenever a schema's wording changes.

    eval/djev/replay.py --corpus clusters --limit 200 --balance --swap-probe
    eval/djev/replay.py --corpus reverted --limit 151
    eval/djev/replay.py --floors          # the label_mass distribution per schema
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent.parent
sys.path.insert(0, str(LLOYD_HOME))

# Muted for the same reason `run_eval.py` mutes it, and before any import that
# could reach a seam: a replay is not production traffic and must not land in
# the distribution the floors are read off.
os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")

from app import djev  # noqa: E402
from app.paths import PIPELINE_DIR, VAULT_FACTS_ROOT  # noqa: E402
from eval.djev import schemas  # noqa: E402

#: The recorded corpora. `_pipeline/` is runtime data under the data root, so it
#: exists only in the root that produced it — a worktree's or the sandbox's root
#: has an empty one and every corpus here silently reads as "no rows", which
#: is indistinguishable from a corpus that ran out. `LLOYD_DJEV_CORPUS_ROOT`
#: is how a run outside the live tree names the tree that HOLDS the data, and
#: `--corpus` prints where it looked when it finds nothing.
PIPELINE = Path(os.environ.get("LLOYD_DJEV_CORPUS_ROOT",
                               str(PIPELINE_DIR / "memory-graph")))
AUTOMOD_STATE = Path.home() / ".local" / "state" / "lloyd-automod"
SHADOW_LOG = Path.home() / ".local" / "state" / "lloyd-djev" / "shadow.jsonl"
BACKLOG_DIR = Path.home() / "obsidian" / "backlog"

#: A replayed pair is one read; at ~45 ms warm plus prefill this is minutes,
#: not hours. The bound exists because 35,802 rows is not a thing to run by
#: accident.
DEFAULT_LIMIT = 200


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based AUC (Mann-Whitney U), ties averaged.

    Threshold-free by construction, which is the whole reason it is the
    headline: a metric that needed a cutoff would be measuring the cutoff.
    `None` when either class is empty — an AUC over one class is not a small
    number, it is undefined, and returning 0.5 there would read as "no
    signal" for what is actually "no measurement".
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def best_threshold(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """`(threshold, accuracy)` at the cutoff that maximises accuracy.

    Reported beside `acc@0.5` so the gap between them is visible. On the
    2026-09-20 set that gap was 0.50 -> 0.75 for one framing: the classifier
    was fine and the cutoff was the problem.
    """
    best = (0.5, 0.0)
    for t in sorted(set(scores)) or [0.5]:
        acc = sum(1 for s, y in zip(scores, labels) if (s >= t) == bool(y)) / len(labels)
        if acc > best[1]:
            best = (t, acc)
    return best


def reliability(scores: list[float], labels: list[int], bins: int = 10) -> list[dict]:
    """Predicted probability against observed rate, per bin.

    The curve is what says whether a probability MEANS anything, which an AUC
    cannot: a model can rank perfectly and still put 0.99 on everything.
    """
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores)
               if (lo <= s < hi) or (b == bins - 1 and s == 1.0)]
        if not idx:
            continue
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx),
                    "mean_p": round(sum(scores[i] for i in idx) / len(idx), 3),
                    "observed": round(sum(labels[i] for i in idx) / len(idx), 3)})
    return out


# ---------------------------------------------------------------------------
# Corpora -> (label, item A, item B) triples
# ---------------------------------------------------------------------------

def _jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _definition(name: str) -> str:
    """The gate's OWN reader, imported rather than reimplemented.

    A second private "what is this entity's definition" is how a replay comes
    to measure a different question from the seam it is calibrating. The first
    cut of this file had one, and it silently parsed 0 of 302 overviews that
    the real reader parses 288 of — so the 151-pair ground-truth run asked
    djev to judge from NAME SHAPE alone, which is exactly the signal
    `SemanticGate.verdict` refuses to answer from.
    """
    try:
        sys.path.insert(0, str(LLOYD_HOME / "scripts" / "memory"))
        from entity_semantic_gate import entity_definition
        return entity_definition(name, VAULT_FACTS_ROOT)
    except Exception:  # noqa: BLE001
        return ""


def corpus_entities(limit: int) -> list[dict]:
    """`semantic-verdicts.jsonl` — the gate's own SAME/REVIEW decisions.

    Rows written while the secondary existed carry two judges; the label is
    the gate's DECISION, so this measures agreement with a rule that requires
    unanimity, not with either judge alone.
    """
    rows = []
    for rec in _jsonl(PIPELINE / "semantic-verdicts.jsonl"):
        a, b = rec.get("a"), rec.get("b")
        da, db = rec.get("def_a") or "", rec.get("def_b") or ""
        if not (a and b and da and db):
            continue
        rows.append({"label": 1 if rec.get("decision") == "SAME" else 0,
                     "a_title": a, "a_body": da, "b_title": b, "b_body": db,
                     "meta": {"a": a, "b": b, "decision": rec.get("decision")}})
        if len(rows) >= limit * 4:
            break
    return rows


def corpus_reverted(limit: int) -> list[dict]:
    """The 151 verified-bad merges. Every pair is a true DIFFERENT.

    Single-class by construction, so AUC is undefined here and the report
    says so. What this corpus answers is the only question that matters for
    the entity seam: of 151 pairs a human REVERTED, how many would djev have
    called the same thing? A schema that cannot separate these is not a
    candidate whatever it scores elsewhere.
    """
    path = next(iter(sorted(PIPELINE.glob("entity-merges-reverted-*.json"))), None)
    if path is None:
        return []
    try:
        plan = json.loads(path.read_text(encoding="utf-8")).get("plan") or []
    except Exception:  # noqa: BLE001
        return []
    rows, no_definition = [], 0
    for entry in plan[:limit * 2]:
        variant, canonical = entry.get("variant"), entry.get("canonical")
        if not (variant and canonical):
            continue
        da, db = _definition(variant), _definition(canonical)
        # The gate's own rule, and it belongs here for the gate's own reason:
        # "asked about a name with nothing on file, the judges answer from
        # the name's shape — the very signal this gate exists to distrust".
        # Replaying a definition-less pair measures the name-shape heuristic
        # and reports it as djev's accuracy. These entities were merged in
        # September 2026 and some directories have not regenerated since.
        if not (da and db):
            no_definition += 1
            continue
        rows.append({"label": 0,   # verified DIFFERENT
                     "a_title": variant, "a_body": da,
                     "b_title": canonical, "b_body": db,
                     "meta": {"variant": variant, "canonical": canonical}})
    if rows:
        rows[0].setdefault("_corpus_note", {})
        rows[0]["_corpus_note"] = {"skipped_no_definition": no_definition}
    return rows


def corpus_clusters(limit: int) -> list[dict]:
    """`cluster_judgments.jsonl` — 609 pair verdicts from the retired 35B
    pair-judge.

    The label is RELATED-or-SAME against DISTINCT, because that is the split
    the corpus actually carries: 382 `distinct`, 221 `related`, 6 `same`, and
    no `duplicate` verdict at any row. The first cut of this loader asked for
    `duplicate` vs `distinct` and got 12 usable pairs out of 609 — six `same`
    rows against six sampled `distinct` ones — then reported AUC 1.000, which
    is what a corpus of twelve looks like when it agrees with you. The class
    counts are on every report for exactly that reason.
    """
    rows = []
    for rec in _jsonl(AUTOMOD_STATE / "cluster_judgments.jsonl"):
        verdict = str(rec.get("verdict") or "")
        if verdict not in ("related", "same", "distinct"):
            continue
        a, b = _backlog_head(rec.get("a")), _backlog_head(rec.get("b"))
        if not (a and b):
            continue
        rows.append({"label": 0 if verdict == "distinct" else 1,
                     "a_title": a["title"], "a_body": a["text"],
                     "b_title": b["title"], "b_body": b["text"],
                     "meta": {"a": rec.get("a"), "b": rec.get("b"),
                              "verdict": verdict,
                              "judge_confidence": rec.get("confidence")}})
        if len(rows) >= limit * 4:
            break
    return rows


def corpus_dedupe(limit: int) -> list[dict]:
    """`dedupe.jsonl` — every write-time decision, with its top candidates.

    The label is what the rule DID: `merged` is positive, `created` with
    candidates present is negative. Rows with no candidates at all carry no
    decision to replay and are skipped.
    """
    rows = []
    for rec in _jsonl(AUTOMOD_STATE / "dedupe.jsonl"):
        top = rec.get("top") or ([{"id": rec.get("into")}] if rec.get("into") else [])
        if not top:
            continue
        cand = _backlog_head(top[0].get("id"))
        if not cand:
            continue
        name = str(rec.get("name") or "")
        if not name:
            continue
        rows.append({"label": 1 if rec.get("action") == "merged" else 0,
                     "a_title": name, "a_body": "",
                     "b_title": f"#{cand['id']} {cand['title']}",
                     "b_body": cand["text"],
                     "meta": {"action": rec.get("action"), "into": rec.get("into"),
                              "score": top[0].get("score"),
                              "lexical": top[0].get("lexical")}})
        if len(rows) >= limit * 4:
            break
    return rows


_BACKLOG_ID_RE = re.compile(r"^(\d+)[-_]")


def _backlog_head(item_id) -> dict | None:
    try:
        iid = int(item_id)
    except (TypeError, ValueError):
        return None
    for path in BACKLOG_DIR.glob(f"{iid}-*.md"):
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:4096]
        except OSError:
            return None
        body = head.split("---", 2)[-1] if head.startswith("---") else head
        m = re.search(r"^#\s+(.+)$", body, re.M)
        title = m.group(1).strip() if m else path.stem
        text = (body[m.end():] if m else body).strip()[:800]
        return {"id": iid, "title": title, "text": text}
    return None


def corpus_edges(limit: int) -> list[dict]:
    """`classified-v4-batch.jsonl` — 35,802 edge-type decisions.

    Not a pair schema: this one is a 10-way `choice` scored as "did djev pick
    the same type the v4 classifier did". Reported as accuracy and a
    per-class breakdown rather than AUC, because AUC over a 10-way label is
    not one number.
    """
    rows = []
    options = list(schemas.EDGES.spec["edge_type"]["criteria"])
    for rec in _jsonl(PIPELINE / "classified-v4-batch.jsonl"):
        new_type = str(rec.get("new_type") or "")
        quote = str(rec.get("reason_quote") or "")
        if new_type not in options or not quote:
            continue
        rows.append({"label": new_type,
                     "a_title": str(rec.get("resolved_src") or rec.get("source") or ""),
                     "b_title": str(rec.get("resolved_tgt") or rec.get("target") or ""),
                     "quote": quote,
                     "meta": {"original_type": rec.get("original_type"),
                              "confidence": rec.get("confidence")}})
        if len(rows) >= limit * 4:
            break
    return rows


CORPORA: dict[str, tuple[Callable[[int], list[dict]], str, str]] = {
    "entities": (corpus_entities, "entity", "a judge's SAME/REVIEW — agreement, not truth"),
    "reverted": (corpus_reverted, "entity", "151 human-verified bad merges — TRUE GROUND TRUTH"),
    "clusters": (corpus_clusters, "clusters", "a retired 35B pair-judge — agreement, not truth"),
    "dedupe": (corpus_dedupe, "dedupe", "what the reranker+Jaccard rule did — agreement, not truth"),
    "edges": (corpus_edges, "edges", "a v4 classifier prompt — agreement, not truth"),
}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _positive_name(schema: schemas.Schema) -> str:
    """The option that means "yes" for this schema.

    The LAST key, because every pair schema here lists the negative first —
    that framing measured AUC 0.833 against 0.795 for the reverse. Reading it
    off the spec rather than naming it keeps the two facts in one place.
    """
    q = next(iter(schema.spec.values()))
    return list(q["criteria"])[-1]


def replay_pairs(rows: list[dict], schema: schemas.Schema, *,
                 swap: bool = False, batch: int = 8) -> list[dict]:
    """Score each pair, `batch` pairs per canvas.

    Batched because a decision is ~33 ms fixed plus ~1.3 ms per extra
    question, so eight pairs in one read costs barely more than one — and
    8 is inside the window where `label_mass` measured 0.997-1.000. It never
    approaches the 32-question canvas split.
    """
    q_template = next(iter(schema.spec.values()))
    positive = _positive_name(schema)
    out: list[dict] = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        parts, questions = [], {}
        for i, r in enumerate(chunk):
            a_t, a_b, b_t, b_b = r["a_title"], r["a_body"], r["b_title"], r["b_body"]
            if swap:
                a_t, a_b, b_t, b_b = b_t, b_b, a_t, a_b
            parts.append(f"=== pair {i} ===\n" + schemas.pair_state(a_t, a_b, b_t, b_b))
            questions[f"p{i}"] = {
                "type": q_template["type"],
                "instructions": q_template["instructions"] + f" (This is pair {i}.)",
                "criteria": dict(q_template["criteria"]),
            }
        answers = djev.ask_sync("\n\n".join(parts), questions, seam="replay",
                                floor=schema.label_mass_floor, timeout=120.0)
        if answers is None:
            for r in chunk:
                out.append({**r, "score": None, "label_mass": None})
            continue
        for i, r in enumerate(chunk):
            a = answers.get(f"p{i}")
            if a is None:
                out.append({**r, "score": None, "label_mass": None})
                continue
            out.append({**r, "score": float(a.probabilities.get(positive, 0.0)),
                        "label_mass": a.label_mass,
                        "argmax_is_label": a.argmax_is_label,
                        "low_trust": a.low_trust,
                        "chosen": a.label})
    return out


def replay_edges(rows: list[dict], schema: schemas.Schema, *,
                 batch: int = 8) -> list[dict]:
    q_template = schema.spec["edge_type"]
    out: list[dict] = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        parts, questions = [], {}
        for i, r in enumerate(chunk):
            parts.append(f"=== relation {i} ===\nSource: {r['a_title']}\n"
                         f"Target: {r['b_title']}\nQuote: {r['quote'][:400]}")
            questions[f"e{i}"] = {
                "type": q_template["type"],
                "instructions": q_template["instructions"] + f" (This is relation {i}.)",
                "criteria": dict(q_template["criteria"]),
            }
        answers = djev.ask_sync("\n\n".join(parts), questions, seam="replay",
                                floor=schema.label_mass_floor, timeout=120.0)
        if answers is None:
            out.extend({**r, "chosen": None, "label_mass": None} for r in chunk)
            continue
        for i, r in enumerate(chunk):
            a = answers.get(f"e{i}")
            out.append({**r, "chosen": (a.value if a else None),
                        "confidence": (a.confidence if a else None),
                        "label_mass": (a.label_mass if a else None),
                        "low_trust": bool(a and a.low_trust)})
    return out


def balance(rows: list[dict], rng: random.Random) -> list[dict]:
    """An equal count per class.

    Unbalanced accuracy on these corpora is a statement about the split: of
    1,309 clause verdicts in one week 1,258 were `met`, so "always say met"
    scores 96%. Balancing is how a number here becomes about the model.
    """
    by_class: dict[Any, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["label"], []).append(r)
    if len(by_class) < 2:
        return rows
    n = min(len(v) for v in by_class.values())
    out: list[dict] = []
    for v in by_class.values():
        out.extend(rng.sample(v, n))
    rng.shuffle(out)
    return out


def report_binary(name: str, provenance: str, schema: schemas.Schema,
                  scored: list[dict], swapped: list[dict] | None) -> dict:
    usable = [r for r in scored if r.get("score") is not None]
    scores = [r["score"] for r in usable]
    labels = [int(r["label"]) for r in usable]
    masses = [r["label_mass"] for r in usable if r.get("label_mass") is not None]
    pos, neg = sum(labels), len(labels) - sum(labels)
    a = auc(scores, labels) if usable else None
    thr, acc = best_threshold(scores, labels) if usable else (None, None)
    note = next((r.get("_corpus_note") for r in scored if r.get("_corpus_note")), None)
    rep: dict[str, Any] = {
        "corpus": name,
        "provenance": provenance,
        "corpus_note": note,
        "ground_truth": name == "reverted",
        "schema": schema.name,
        "schema_hash": schema.hash,
        "n": len(usable),
        "n_unanswered": len(scored) - len(usable),
        "class_counts": {"positive": pos, "negative": neg},
        "auc": round(a, 3) if a is not None else None,
        "auc_note": (None if a is not None else
                     "undefined: this corpus has only one class, which is the "
                     "point of it — it answers 'how many of these would djev "
                     "have got wrong', not 'how well does it separate'"),
        "acc_at_0.5": (round(sum(1 for s, y in zip(scores, labels)
                                 if (s >= 0.5) == bool(y)) / len(labels), 3)
                       if usable else None),
        "best_threshold": round(thr, 3) if thr is not None else None,
        "acc_at_best": round(acc, 3) if acc is not None else None,
        "mean_p_positive": (round(statistics.fmean([s for s, y in zip(scores, labels) if y]), 3)
                            if pos else None),
        "mean_p_negative": (round(statistics.fmean([s for s, y in zip(scores, labels) if not y]), 3)
                            if neg else None),
        "label_mass": _dist(masses),
        # How often this schema's own floor fired on its own corpus. A floor
        # that fires on a tenth of normal traffic is not a trust flag, it is
        # a second opinion nobody asked for.
        "label_mass_floor": schema.label_mass_floor,
        "low_trust_rate": (round(sum(1 for r in usable if r.get("low_trust"))
                                 / len(usable), 3) if usable else None),
        "reliability": reliability(scores, labels) if usable else [],
    }
    if swapped:
        deltas = [abs(a_["score"] - b_["score"])
                  for a_, b_ in zip(scored, swapped)
                  if a_.get("score") is not None and b_.get("score") is not None]
        rep["order_swap"] = {
            "n": len(deltas),
            "mean_abs_delta": round(statistics.fmean(deltas), 3) if deltas else None,
            "max_abs_delta": round(max(deltas), 3) if deltas else None,
            "note": "the same pairs with A and B exchanged. A large delta is "
                    "why the option order is part of the schema hash and why "
                    "no fixed cutoff survives a rewording.",
        }
    return rep


def report_edges(scored: list[dict], schema: schemas.Schema) -> dict:
    usable = [r for r in scored if r.get("chosen")]
    hits = sum(1 for r in usable if r["chosen"] == r["label"])
    per_class: dict[str, dict] = {}
    for r in usable:
        row = per_class.setdefault(r["label"], {"n": 0, "hit": 0})
        row["n"] += 1
        row["hit"] += int(r["chosen"] == r["label"])
    # The majority-class floor. Any accuracy at or under it is a statement
    # about the corpus, not about the model.
    counts: dict[str, int] = {}
    for r in usable:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    majority = max(counts.values()) / len(usable) if usable else None
    return {
        "corpus": "edges",
        "provenance": CORPORA["edges"][2],
        "ground_truth": False,
        "schema": schema.name,
        "schema_hash": schema.hash,
        "n": len(usable),
        "n_unanswered": len(scored) - len(usable),
        "agreement": round(hits / len(usable), 3) if usable else None,
        "majority_class_baseline": round(majority, 3) if majority else None,
        "per_class": {k: {**v, "rate": round(v["hit"] / v["n"], 3)}
                      for k, v in sorted(per_class.items())},
        "label_mass": _dist([r["label_mass"] for r in usable
                             if r.get("label_mass") is not None]),
        "label_mass_floor": schema.label_mass_floor,
        "low_trust_rate": (round(sum(1 for r in usable if r.get("low_trust"))
                                 / len(usable), 3) if usable else None),
    }


def _dist(values: list[float]) -> dict | None:
    """The distribution a floor gets read off. Percentiles rather than a mean:
    a floor is about the bottom of the distribution and a mean says nothing
    about it."""
    if not values:
        return None
    s = sorted(values)

    def pct(p: float) -> float:
        return round(s[min(len(s) - 1, int(p * len(s)))], 4)

    return {"n": len(s), "min": round(s[0], 4), "p01": pct(0.01),
            "p05": pct(0.05), "p10": pct(0.10), "p50": pct(0.50),
            "mean": round(statistics.fmean(s), 4), "max": round(s[-1], 4)}


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------

def floors_from_shadow(path: Path = SHADOW_LOG) -> dict:
    """The `label_mass` distribution per seam, from the shadow log.

    This is what step 6 of the plan sets the floors from. Until the log has
    rows, `eval/djev/replay.py --corpus <c>` over a recorded corpus of the
    SAME schema is the honest stand-in and its provenance is recorded as such
    — a floor read off a corpus is a measurement, a floor chosen because 0.5
    is a round number is not.
    """
    per_seam: dict[str, list[float]] = {}
    if not path.exists():
        return {"log": str(path), "rows": 0, "seams": {}}
    rows = 0
    for rec in _jsonl(path):
        rows += 1
        d = rec.get("djev") or {}
        for a in (d.get("answers") or {}).values():
            m = a.get("label_mass")
            if isinstance(m, (int, float)):
                per_seam.setdefault(str(rec.get("seam") or "-"), []).append(float(m))
    return {"log": str(path), "rows": rows,
            "seams": {k: _dist(v) for k, v in per_seam.items()}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=[*CORPORA, "all"])
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"Decisions to replay (default {DEFAULT_LIMIT})")
    ap.add_argument("--balance", action="store_true",
                    help="Equal count per class — see the docstring on why an "
                         "unbalanced accuracy describes the split")
    ap.add_argument("--swap-probe", action="store_true",
                    help="Also run every pair with A and B exchanged and "
                         "report |P_A - P_B|")
    ap.add_argument("--batch", type=int, default=8,
                    help="Decisions per canvas (default 8; never near the "
                         "32-question chunk split)")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--floors", action="store_true",
                    help="Print the label_mass distribution per seam from the "
                         "shadow log and exit")
    ap.add_argument("--out", default="", help="Append the report as JSON here")
    args = ap.parse_args()

    if args.floors:
        print(json.dumps(floors_from_shadow(), indent=2))
        return 0
    if not args.corpus:
        ap.error("--corpus is required (or --floors)")
    if not djev.reachable():
        print(f"djev is not answering on {djev.structured_url()}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    wanted = list(CORPORA) if args.corpus == "all" else [args.corpus]
    reports = []
    for name in wanted:
        loader, schema_name, provenance = CORPORA[name]
        schema = schemas.get(schema_name)
        schema.check()
        rows = loader(args.limit)
        if not rows:
            print(f"{name}: no usable rows under {PIPELINE} / {AUTOMOD_STATE}. "
                  f"`_pipeline/` is gitignored derived data — set "
                  f"LLOYD_DJEV_CORPUS_ROOT to the tree that holds it when "
                  f"running from a clone.", file=sys.stderr)
            continue
        if args.balance:
            rows = balance(rows, rng)
        rng.shuffle(rows)
        rows = rows[:args.limit]
        t0 = time.perf_counter()
        if name == "edges":
            scored = replay_edges(rows, schema, batch=args.batch)
            rep = report_edges(scored, schema)
        else:
            scored = replay_pairs(rows, schema, batch=args.batch)
            swapped = (replay_pairs(rows, schema, swap=True, batch=args.batch)
                       if args.swap_probe else None)
            rep = report_binary(name, provenance, schema, scored, swapped)
        rep["elapsed_s"] = round(time.perf_counter() - t0, 1)
        reports.append(rep)
        print(json.dumps(rep, indent=2))

    if args.out and reports:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            for rep in reports:
                fh.write(json.dumps({"ts": time.time(), **rep}) + "\n")
        print(f"recorded to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
