#!/usr/bin/env python3
"""Self-question retrieval arm (#1164): retrieve against LLM-drafted recall questions.

The idea (Tolan's memory system, AI Engineer 2026-09-15): do not retrieve only
against the user's words — have a cheap model draft the questions the message
*implies* ("which engine did we move to FP8?") and retrieve against those too,
so an oblique reference becomes retrievable text. This module is the offline
measurement of that idea against Lloyd's own gold set, and nothing else: it
touches no production path, and the drafted questions reach only the artifact
it writes.

One invocation, per query, in this order (interleaved so drift in the daemon,
djev or the engine lands on every arm alike):

  control      the raw query through `_vault_recall`, exactly what the nightly
               scores (same knobs as `eval/run_eval.py`'s defaults)
  self_question  the model drafts <= MAX_QUESTIONS questions; one recall per
               question; reciprocal-rank fusion of [raw] + per-question lists
  topics       the same fusion over the topic phrases the LIVE focus extractor
               drafts (`app.secondary_models._sync_secondary_focus_extraction`,
               the expansion production already ships in prefetch) — the
               honest control for "questions" vs "any LLM expansion"
  control_b    the raw query again: the run's own noise floor, since djev
               ranks the recall and does not promise to repeat itself

Fusion is budget-equal: every fused list is cut to the longest list that went
into it, i.e. to what ONE recall returns. An unbounded union would win
`doc_recall` / `entity_recall` by length alone, which is not retrieval quality.
The price is that a raw-query document can fall off the tail; how many did is
recorded per query (`raw_docs_displaced`) rather than hidden.

Scoring is `eval.run_eval._score`, the nightly's own scorer, with the RAW
query's seeds on every arm: a seed is a string lifted from the query text, so
crediting an arm with its drafted questions' seeds would score the drafter
for naming an entity, not retrieval for returning one.

Usage (primary engine and the retrieval stack are both loaded — take both locks):
    flock <scratch>/primary.lock flock <scratch>/retrieval.lock \\
      .venvs/lloyd/bin/python eval/self_question.py --out eval/measurements/self-question-<date>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# Same reason and same position as run_eval.py: the djev shadow seam lives
# inside `_vault_recall`, and an eval must never write production-shaped rows.
os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")

import yaml  # noqa: E402

#: The item's own cap, and the fan-out the latency is priced at.
MAX_QUESTIONS = 3
#: Reciprocal-rank-fusion constant (Cormack et al.); the conventional value.
RRF_K = 60
#: The arms, in the order a run executes them per query.
ARMS = ("control", "self_question", "topics", "control_b")
#: Per-query scoring fields compared arm-vs-control, with the nightly's names.
METRICS = {
    "doc_hit_rate": "doc_hit",
    "mrr_doc": "rr_doc",
    "ndcg10": "ndcg10",
    "doc_recall_avg": "doc_recall",
    "entity_hit_rate": "entity_hit",
    "entity_recall_avg": "entity_recall",
    "fact_entity_recall_avg": "fact_entity_recall",
}

DRAFT_SYSTEM = (
    "You help a memory search system for Lloyd, a local AI agent, and Alan, "
    "the person who built and uses it. Given a message (and any recent "
    "conversation), write up to 3 short questions that the memory search "
    "should answer before replying. Name the specific systems, projects, "
    "files, people, decisions or backlog items the message refers to or only "
    "implies. One question per line. No numbering, no preamble."
)


# ── drafting ─────────────────────────────────────────────────────────────────

def parse_questions(text: str) -> list[str]:
    """One question per non-empty line, list markers stripped, deduplicated."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        q = line.strip().lstrip("-*•0123456789.) ").strip()
        if len(q) < 4 or q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
    return out


def llm_draft(query: str, context: str = "", *, timeout: float = 30.0) -> str:
    """Draft recall questions on the engine the `secondary` alias resolves to.

    `secondary_enabled: false` sends it to the primary, which is where a live
    arm would run today too — so the latency measured is the deployable one.
    Thinking off, greedy, short: the drafts are a query rewrite, not an answer.
    """
    from app.secondary_models import _endpoint
    url, model = _endpoint("self_question")
    user = f"Recent conversation:\n{context}\n\nMessage: {query}" if context else f"Message: {query}"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": DRAFT_SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 160,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"] or ""


def topic_draft(query: str, context: str = "") -> list[str]:
    """The expansion production already ships: the prefetch focus extractor."""
    from app.secondary_models import _sync_secondary_focus_extraction
    text = f"{context}\n\n{query}" if context else query
    return list(_sync_secondary_focus_extraction(text))


def draft_questions(query: str, context: str = "", *,
                    drafter: Callable[[str, str], object] = llm_draft) -> list[str]:
    """At most MAX_QUESTIONS drafts, whatever the drafter returned.

    A drafter may return raw text (parsed one per line) or a list. The cap is
    applied HERE, after the model, because a model asked for three returns ten
    often enough that a prompt is not a bound.
    """
    got = drafter(query, context)
    qs = parse_questions(got) if isinstance(got, str) else [str(q).strip() for q in got or [] if str(q).strip()]
    return qs[:MAX_QUESTIONS]


# ── fusion ───────────────────────────────────────────────────────────────────

def _fact_key(f: dict) -> tuple:
    return (str(f.get("entity") or "").lower(), str(f.get("id") or f.get("fact") or ""))


def rrf_fuse(lists: list[list[dict]], key: Callable[[dict], object], *,
             k: int = RRF_K, cap: int | None = None) -> list[dict]:
    """Reciprocal-rank fusion; first occurrence of an item is the one kept.

    `cap` defaults to the longest input list, so the fused list is never longer
    than what one retrieval returns (see the module docstring for why).
    Ties break on first appearance, with the raw list first — so with no
    question lists at all the output IS the raw list.
    """
    scores: dict = {}
    first: dict = {}
    order: dict = {}
    for li, lst in enumerate(lists):
        for rank, item in enumerate(lst or [], start=1):
            kk = key(item)
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (k + rank)
            if kk not in first:
                first[kk] = item
                order[kk] = (li, rank)
    ranked = sorted(scores, key=lambda kk: (-scores[kk], order[kk]))
    if cap is None:
        cap = max((len(lst or []) for lst in lists), default=0)
    return [first[kk] for kk in ranked[:cap]]


def fuse_results(raw: dict, extra: list[dict]) -> dict:
    """Fuse a raw-query recall with per-question recalls, list by list."""
    allr = [raw] + list(extra)

    def lists(field):
        return [r.get(field) or [] for r in allr]

    out = {
        "documents": rrf_fuse(lists("documents"), lambda d: str(d.get("path") or "")),
        "facts": rrf_fuse(lists("facts"), _fact_key),
        "graph_expanded_facts": rrf_fuse(lists("graph_expanded_facts"), _fact_key),
        "graph_neighbors_used": rrf_fuse(lists("graph_neighbors_used"),
                                         lambda n: str(n.get("entity") or "").lower()),
    }
    fused_paths = {str(d.get("path") or "") for d in out["documents"]}
    out["raw_docs_displaced"] = sum(1 for d in raw.get("documents") or []
                                    if str(d.get("path") or "") not in fused_paths)
    return out


def expansion_arm(raw: dict, drafts: list[str], recall: Callable[[str], dict]) -> tuple[dict, list[float]]:
    """One recall per draft (exactly one), fused with the raw result.

    Returns (fused_result, per-draft recall latencies in ms). A draft whose
    recall raised contributes nothing rather than failing the arm: the raw list
    is always in the fusion, so the arm degrades to the control, not to zero.
    """
    extra, lat = [], []
    for q in drafts:
        t0 = time.perf_counter()
        try:
            extra.append(recall(q) or {})
        except Exception:  # noqa: BLE001 - an errored draft is an empty list
            extra.append({})
        lat.append((time.perf_counter() - t0) * 1000)
    return fuse_results(raw, extra), lat


# ── the run ──────────────────────────────────────────────────────────────────

def _summ(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1),
            "p50": round(xs[len(xs) // 2], 1),
            "p95": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1)}


def run(queries: list[dict], *, recall: Callable[[str], dict],
        q_drafter=llm_draft, t_drafter=topic_draft,
        seeds_for: Callable[[str], list[str]] = lambda q: [],
        score: Callable[[dict, dict, list[str]], dict] | None = None,
        progress: Callable[[str], None] = lambda s: None) -> list[dict]:
    """Score every arm on every query. Pure over its injected callables."""
    if score is None:
        from eval.run_eval import _score as score  # noqa: N813
    recs = []
    for i, spec in enumerate(queries):
        query = spec.get("query") or ""
        if not query:
            continue
        context = spec.get("context") or ""
        seeds = seeds_for(query)
        rec = {"id": spec.get("id"), "category": spec.get("category"), "query": query,
               "arms": {}}

        t0 = time.perf_counter()
        raw = recall(query) or {}
        raw_ms = (time.perf_counter() - t0) * 1000
        rec["arms"]["control"] = {"scoring": score(spec, raw, seeds), "latency_ms": round(raw_ms, 1)}

        for arm, drafter in (("self_question", q_drafter), ("topics", t_drafter)):
            t0 = time.perf_counter()
            try:
                drafts = draft_questions(query, context, drafter=drafter)
                derr = None
            except Exception as e:  # noqa: BLE001
                drafts, derr = [], f"{type(e).__name__}: {e}"
            draft_ms = (time.perf_counter() - t0) * 1000
            fused, lat = expansion_arm(raw, drafts, recall)
            rec["arms"][arm] = {
                "drafts": drafts, "draft_error": derr,
                "scoring": score(spec, fused, seeds),
                "raw_docs_displaced": fused["raw_docs_displaced"],
                "draft_ms": round(draft_ms, 1),
                "recall_ms": [round(x, 1) for x in lat],
                # What the arm ADDS on top of the raw recall it fuses with:
                # sequential (as measured here) and the parallel lower bound
                # (draft, then every question recall at once).
                "added_ms_sequential": round(draft_ms + sum(lat), 1),
                "added_ms_parallel": round(draft_ms + (max(lat) if lat else 0.0), 1),
            }

        t0 = time.perf_counter()
        raw_b = recall(query) or {}
        rec["arms"]["control_b"] = {"scoring": score(spec, raw_b, seeds),
                                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        recs.append(rec)
        progress(f"[{i + 1}/{len(queries)}] {rec['id']}: "
                 f"q={len(rec['arms']['self_question']['drafts'])} "
                 f"t={len(rec['arms']['topics']['drafts'])}")
    return recs


def _num(v):
    if v is None:
        return None
    return 1.0 if v is True else 0.0 if v is False else float(v)


def compare(recs: list[dict], *, n_resamples: int = 2000) -> dict:
    """Per-arm means and paired deltas vs `control`, with paired bootstrap CIs."""
    from eval import stats as evstats
    out: dict = {"n_queries": len(recs), "arms": {}, "paired_vs_control": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for metric, field in METRICS.items():
            vals = [_num(r["arms"][arm]["scoring"].get(field)) for r in recs]
            vals = [v for v in vals if v is not None]
            out["arms"][arm][metric] = round(statistics.fmean(vals), 4) if vals else None
    for arm in ARMS[1:]:
        out["paired_vs_control"][arm] = {}
        for metric, field in METRICS.items():
            a, b = [], []
            for r in recs:
                x = _num(r["arms"]["control"]["scoring"].get(field))
                y = _num(r["arms"][arm]["scoring"].get(field))
                if x is not None and y is not None:
                    a.append(x)
                    b.append(y)
            if not a:
                out["paired_vs_control"][arm][metric] = None
                continue
            ci = evstats.paired_bootstrap_ci(a, b, n_resamples=n_resamples)
            out["paired_vs_control"][arm][metric] = {
                "delta": round(ci["diff"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                "p": round(ci["p"], 4), "n": ci["n"], "significant": ci["significant"],
                "wins": sum(1 for x, y in zip(a, b) if y > x),
                "losses": sum(1 for x, y in zip(a, b) if y < x),
            }
    lat = {"control_ms": _summ([r["arms"]["control"]["latency_ms"] for r in recs])}
    for arm in ("self_question", "topics"):
        lat[arm] = {
            "draft_ms": _summ([r["arms"][arm]["draft_ms"] for r in recs]),
            "added_ms_sequential": _summ([r["arms"][arm]["added_ms_sequential"] for r in recs]),
            "added_ms_parallel": _summ([r["arms"][arm]["added_ms_parallel"] for r in recs]),
            "mean_drafts": round(statistics.fmean(len(r["arms"][arm]["drafts"]) for r in recs), 2) if recs else None,
            "draft_errors": sum(1 for r in recs if r["arms"][arm].get("draft_error")),
            "raw_docs_displaced_mean": round(statistics.fmean(r["arms"][arm]["raw_docs_displaced"] for r in recs), 2) if recs else None,
        }
    out["latency"] = lat
    # The arm's own latency, judged against the context it ran in: an arm run
    # is a paired check, never folded into the nightly's average (#1164 cl. 5).
    try:
        from workers.sources.automod_regression import CONTEXT_PAIRED_CHECK, over_budget
        ctrl = lat["control_ms"]["mean"] or 0.0
        added = lat["self_question"]["added_ms_sequential"]["mean"] or 0.0
        out["latency_budget"] = over_budget({"latency_ms_avg": ctrl + added}, CONTEXT_PAIRED_CHECK)
    except Exception:  # noqa: BLE001 - reporting only
        out["latency_budget"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queries", default=str(HERE / "vault_recall_queries.yaml"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True, help="artifact path stem (.json is written)")
    args = ap.parse_args()

    from agent_mcp.facts import _extract_entities_from_query
    from agent_mcp.vault import RECALL_SEED_TOP_K, _vault_recall
    from eval.run_eval import _corpus_provenance

    queries = (yaml.safe_load(Path(args.queries).read_text()) or {}).get("queries") or []
    if args.max_queries:
        queries = queries[: args.max_queries]

    def recall(q: str) -> dict:
        # The nightly's own knobs: run_eval's signature defaults ARE the
        # production constants, with expand_graph on as the eval measures it.
        return _vault_recall({"query": q, "limit": args.limit, "expand_graph": True},
                             seed_top_k=RECALL_SEED_TOP_K)

    def seeds_for(q: str) -> list[str]:
        return [e for e, _ in (_extract_entities_from_query(q) or [])[:RECALL_SEED_TOP_K]]

    t0 = time.time()
    recs = run(queries, recall=recall, seeds_for=seeds_for,
               progress=lambda s: print(s, flush=True))
    summary = compare(recs)
    out = {
        "item": 1164,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "queries_file": args.queries,
        "limit": args.limit,
        "max_questions": MAX_QUESTIONS,
        "rrf_k": RRF_K,
        "corpus": _corpus_provenance(),
        "summary": summary,
        "records": recs,
    }
    path = Path(args.out).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("arms", "latency", "latency_budget")}, indent=1))
    for arm, rows in summary["paired_vs_control"].items():
        for m, v in rows.items():
            if v:
                print(f"{arm:14s} {m:24s} d={v['delta']:+.4f} ci={v['ci95']} p={v['p']} W/L={v['wins']}/{v['losses']}")
    print(f"[info] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
