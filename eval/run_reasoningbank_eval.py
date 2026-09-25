#!/usr/bin/env python3
"""Offline eval of the autocode ReasoningBank (#1489).

A live A/B of rounds with and without the bank is the real test and a human's
call (the pool is paused). This asks the question that can be answered from the
ledger alone: **had the bank been on, would the lessons injected into a round
that the review rung later refused have named the cause of that refusal?**

- Held-out set: every round that started at or after ``--cutoff`` whose first
  blocking review refused at least one clause (or a test) with an informative
  grader note. One row per round.
- The bank each held-out round sees is rebuilt from the ledger, restricted to
  items dated strictly before THAT round's start (so nothing from its own or any
  later review), pruned as production would prune at that moment, and — in the
  headline arm — without the round's own item's earlier rounds (those already
  reach the prompt through the re-offer banner).
- Arms: ``sim`` (production: TF-IDF item similarity, distinct causes), ``prior``
  (the k most frequent causes in the bank so far — a static paragraph needing no
  retrieval), ``random`` (k random distinct-cause items), ``sim_same`` (``sim``
  with the item's own earlier rounds allowed).
- Relevance: an LLM judge on the primary (thinking off, temperature 0), asked
  whether following the item would have directly addressed one of the refusals;
  a hand-audited subsample checks the judge. A deterministic proxy (the item's
  cause equals a refusal's classified cause) is reported beside it.

Artifacts go to ``~/lloyd-data/eval/1489/`` (judge cache, per-pair rows, the
summary). Run the judge under the shared primary lock::

    flock -s -w 7200 ~/.local/state/lloyd-automod/primary.lock \\
        .venvs/lloyd/bin/python eval/run_reasoningbank_eval.py --judge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from calendar import timegm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.stats import paired_bootstrap_ci, wilson_ci  # noqa: E402
from scripts.automod import reasoning_bank as RB  # noqa: E402

LEDGER = Path.home() / ".local/state/lloyd-automod/promotions.jsonl"
OUT = Path.home() / "lloyd-data/eval/1489"
BASE = "http://127.0.0.1:8096"
ARMS = ("sim", "prior", "random", "sim_same")

JUDGE_SYSTEM = (
    "You audit an automated code-review loop. A coding agent's change was refused by a "
    "reviewer for the reasons listed. Before the agent started, it was shown one 'lesson' "
    "distilled from earlier refusals on other work. Decide whether that lesson names the "
    "cause of at least one of THESE refusals: following the lesson's advice would have "
    "directly addressed the same failure mechanism (for example an assertion that cannot "
    "fail, a clause half left undelivered, a check that needs a live service, a test that "
    "never reaches production code). Generic good practice that does not bear on these "
    "specific refusals is NOT relevant. Answer with exactly one word: YES or NO.")


def load_events(path: Path) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def iso_ts(s: str) -> float:
    return float(timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))


def heldout(rounds: dict, cutoff: float) -> list[dict]:
    rows = []
    for r in rounds.values():
        if r["start_ts"] < cutoff or r["item_id"] is None:
            continue
        for rev in r["reviews"]:
            refs = RB._refusals(rev)
            if refs:
                rows.append({"round_id": r["round_id"], "item_id": r["item_id"],
                             "start_ts": r["start_ts"], "goal": r["goal"],
                             "clauses": r["clauses"], "refusals": refs,
                             "causes": sorted({RB.classify_cause(x["text"]) for x in refs})})
                break
    rows.sort(key=lambda d: d["start_ts"])
    return rows


def pool_at(bank: list[dict], q: dict, *, same_item: bool) -> list[dict]:
    """What production would hold at the round's start: dated before it, pruned
    as of then (pruning AFTER the date filter, or a later fix supersedes an
    earlier refusal from the future)."""
    visible = [d for d in bank if d["ts"] < q["start_ts"]
               and (same_item or d["item_id"] != q["item_id"])]
    return RB.prune(visible, now=q["start_ts"])


def arm_items(arm: str, bank: list[dict], q: dict, k: int, seed: int) -> list[dict]:
    query = RB.item_query(q["goal"], q["clauses"])
    if arm in ("sim", "sim_same"):
        pool = pool_at(bank, q, same_item=(arm == "sim_same"))
        return RB.retrieve(pool, query, k=k)
    pool = pool_at(bank, q, same_item=False)
    if arm == "prior":
        return RB.prior_items(pool, k=k)
    if arm == "random":
        rng = random.Random(f"{seed}:{q['round_id']}")
        shuffled = pool[:]
        rng.shuffle(shuffled)
        out, seen = [], set()
        for d in shuffled:
            if d["cause"] not in seen:
                seen.add(d["cause"])
                out.append(d)
            if len(out) >= k:
                break
        return out
    raise ValueError(arm)


def refusal_text(q: dict) -> str:
    lines = []
    for x in q["refusals"]:
        head = f"clause {x['clause']} ({x['verdict']})" if x["clause"] else "blocking test finding"
        lines.append(f"- {head}: {RB._clean(x['text'], 700)}")
    return "\n".join(lines)


def pair_key(q: dict, d: dict) -> str:
    return hashlib.sha1(f"{q['round_id']}|{d['id']}|v1".encode()).hexdigest()


def judge_pair(client, q: dict, d: dict) -> bool:
    user = (f"Refusals:\n{refusal_text(q)}\n\nLesson shown before the work:\n"
            f"{RB.render_item(d)}\n\nDoes the lesson name the cause of at least one of "
            "these refusals? YES or NO.")
    body = {"model": "primary", "temperature": 0, "max_tokens": 4,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": user}]}
    for attempt in range(3):
        try:
            r = client.post(f"{BASE}/v1/chat/completions", json=body, timeout=120)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            return text.strip().upper().startswith("YES")
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(5)
    return False


def count_tokens(client, text: str) -> int:
    if not text:
        return 0
    try:
        r = client.post(f"{BASE}/tokenize", json={"model": "primary", "prompt": text}, timeout=30)
        r.raise_for_status()
        return int(r.json()["count"])
    except Exception:  # noqa: BLE001
        return round(len(text) / 3.6)


def ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-09-17T00:00:00Z")
    ap.add_argument("--k", type=int, default=RB.DEFAULT_K)
    ap.add_argument("--seed", type=int, default=1489)
    ap.add_argument("--judge", action="store_true", help="call the primary for relevance")
    ap.add_argument("--audit", type=int, default=0, help="print N judged pairs to hand-audit")
    ap.add_argument("--ledger", default=str(LEDGER))
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events(Path(args.ledger))
    rounds = RB.rounds_from_events(events)
    bank = RB.build_bank(events)
    qs = heldout(rounds, iso_ts(args.cutoff))

    cache_path = OUT / "judge_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            d = json.loads(line)
            cache[d["key"]] = d["relevant"]

    client = None
    if args.judge:
        import httpx
        client = httpx.Client()

    rows, tokens = [], {a: [] for a in ARMS}
    for q in qs:
        for arm in ARMS:
            items = arm_items(arm, bank, q, args.k, args.seed)
            block = RB.render_block(items)
            if client is not None and arm in ("sim", "prior"):
                tokens[arm].append(count_tokens(client, block))
            for d in items:
                key = pair_key(q, d)
                if key not in cache and client is not None:
                    cache[key] = judge_pair(client, q, d)
                    with open(cache_path, "a") as fh:
                        fh.write(json.dumps({"key": key, "round_id": q["round_id"],
                                             "bank_id": d["id"], "relevant": cache[key]}) + "\n")
                rows.append({"arm": arm, "round_id": q["round_id"], "item_id": q["item_id"],
                             "bank_id": d["id"], "bank_item": d["item_id"],
                             "bank_cause": d["cause"], "score": d.get("score"),
                             "refusal_causes": q["causes"],
                             "tag_match": d["cause"] in q["causes"] and d["cause"] != "other",
                             "judge": cache.get(key)})
            if not items:
                rows.append({"arm": arm, "round_id": q["round_id"], "item_id": q["item_id"],
                             "bank_id": None, "judge": None, "tag_match": False})

    (OUT / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    summary = {"cutoff": args.cutoff, "k": args.k, "n_heldout_rounds": len(qs),
               "bank_items_total": len(bank),
               "bank_items_before_cutoff": sum(1 for d in bank if d["ts"] < iso_ts(args.cutoff)),
               "heldout_cause_mix": {}, "arms": {}}
    from collections import Counter
    summary["heldout_cause_mix"] = dict(Counter(c for q in qs for c in q["causes"]).most_common())
    per_query = {}
    for arm in ARMS:
        ar = [r for r in rows if r["arm"] == arm and r.get("bank_id")]
        judged = [r for r in ar if r["judge"] is not None]
        hits_j = {q["round_id"]: 0 for q in qs}
        hits_t = {q["round_id"]: 0 for q in qs}
        for r in ar:
            if r["judge"]:
                hits_j[r["round_id"]] = 1
            if r["tag_match"]:
                hits_t[r["round_id"]] = 1
        per_query[arm] = ([hits_j[q["round_id"]] for q in qs], [hits_t[q["round_id"]] for q in qs])
        summary["arms"][arm] = {
            "injected_items": len(ar),
            "rounds_with_items": len({r["round_id"] for r in ar}),
            "precision_judge": ci(sum(1 for r in judged if r["judge"]), len(judged)),
            "recall_judge": ci(sum(hits_j.values()), len(qs)) if judged else None,
            "precision_tag": ci(sum(1 for r in ar if r["tag_match"]), len(ar)),
            "recall_tag": ci(sum(hits_t.values()), len(qs)),
        }
        if tokens[arm]:
            t = sorted(tokens[arm])
            summary["arms"][arm]["block_tokens"] = {
                "mean": round(sum(t) / len(t), 1), "median": t[len(t) // 2],
                "max": t[-1], "n": len(t)}
    for other in ("prior", "random"):
        for idx, name in ((0, "judge"), (1, "tag")):
            a, b = per_query[other][idx], per_query["sim"][idx]
            if name == "judge" and not any(r["judge"] is not None for r in rows):
                continue
            summary[f"recall_{name}_sim_minus_{other}"] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in paired_bootstrap_ci(a, b).items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.audit:
        rng = random.Random(args.seed)
        judged = [r for r in rows if r["judge"] is not None and r["arm"] in ("sim", "prior")]
        sample = rng.sample(judged, min(args.audit, len(judged)))
        qmap = {q["round_id"]: q for q in qs}
        bmap = {d["id"]: d for d in bank}
        audit = []
        for i, r in enumerate(sample):
            q, d = qmap[r["round_id"]], bmap[r["bank_id"]]
            audit.append({"n": i, "round_id": r["round_id"], "bank_id": r["bank_id"],
                          "judge": r["judge"], "refusals": refusal_text(q),
                          "lesson": RB.render_item(d)})
        (OUT / "audit_sample.json").write_text(json.dumps(audit, indent=2))
        print(f"audit sample: {OUT / 'audit_sample.json'} ({len(audit)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
