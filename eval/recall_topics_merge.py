#!/usr/bin/env python3
"""#1456: the topics merge on the `vault_recall` path, paired against today's recall.

Two passes, both on a pinned corpus (a frozen qmd snapshot on its own port, the
regression runner's `PinnedCorpus`) with the live fact tree and store read-only:

  quality   per query, interleaved: `control` (merge off, i.e. production),
            `facts` (merge "facts"), `full` (merge "full"), `control_b` (off
            again: the run's noise floor). Under djev replay anchored on
            `control`, so every raw recall inside an arm gets the very answer
            control got and the only difference left is the merge itself.
            `--leg dev` scores the 86-query gold set; `--leg holdout` scores the
            reserved #1412 tranche and writes AGGREGATES ONLY (no id, no
            per-query row), the reserve rule `eval/retrieval_holdout.py` states.
  latency   the same arms with replay OFF, so each arm pays a fresh djev read,
            which is what production would pay. Reported per arm as wall ms and
            as added ms over the control recall of the same query.

The arms call the real `agent_mcp.vault._vault_recall` with the mode forced, so
what is measured is the code that would ship. Scoring is the nightly's own
`eval.run_eval._score`, with the raw query's seeds on every arm.

Locks (the caller takes them): regression.lock exclusive (the pin), and
primary.lock — shared for `quality`, exclusive for `latency` (the drafter is a
primary call and its timing is reported).

    flock -x ~/.local/state/lloyd-automod/regression.lock \\
      flock -s ~/.local/state/lloyd-automod/primary.lock \\
      .venvs/lloyd/bin/python eval/recall_topics_merge.py --pass quality --leg dev --out <stem>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

ARMS = ("control", "facts", "full", "control_b")
MODES = {"control": "off", "facts": "facts", "full": "full", "control_b": "off"}
METRICS = {
    "doc_hit_rate": "doc_hit",
    "mrr_doc": "rr_doc",
    "ndcg10": "ndcg10",
    "doc_recall_avg": "doc_recall",
    "entity_hit_rate": "entity_hit",
    "entity_recall_avg": "entity_recall",
    "fact_entity_recall_avg": "fact_entity_recall",
}


def _num(v):
    if v is None:
        return None
    return 1.0 if v is True else 0.0 if v is False else float(v)


def _summ(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1),
            "p50": round(xs[len(xs) // 2], 1),
            "p95": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1)}


def compare(recs: list[dict], arms=ARMS, *, n_resamples: int = 2000) -> dict:
    from eval import stats as evstats
    out: dict = {"n_queries": len(recs), "means": {}, "paired_vs_control": {}}
    for arm in arms:
        out["means"][arm] = {}
        for metric, field in METRICS.items():
            vals = [_num(r["arms"][arm]["scoring"].get(field)) for r in recs]
            vals = [v for v in vals if v is not None]
            out["means"][arm][metric] = round(statistics.fmean(vals), 4) if vals else None
    for arm in arms[1:]:
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
                continue
            ci = evstats.paired_bootstrap_ci(a, b, n_resamples=n_resamples)
            out["paired_vs_control"][arm][metric] = {
                "delta": round(ci["diff"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                "p": round(ci["p"], 4), "n": ci["n"], "significant": ci["significant"],
                "wins": sum(1 for x, y in zip(a, b) if y > x),
                "losses": sum(1 for x, y in zip(a, b) if y < x)}
    lat = {}
    for arm in arms:
        lat[arm] = {"wall_ms": _summ([r["arms"][arm]["ms"] for r in recs])}
        if arm != "control":
            lat[arm]["added_ms_vs_control"] = _summ(
                [r["arms"][arm]["ms"] - r["arms"]["control"]["ms"] for r in recs])
    out["latency"] = lat
    out["topics"] = {
        "mean_n": round(statistics.fmean(len(r.get("topics") or []) for r in recs), 2) if recs else None,
        "queries_with_none": sum(1 for r in recs if not r.get("topics")),
    }
    return out


def child(args) -> int:
    """Runs inside the pinned environment (overlay already in env)."""
    import yaml
    from agent_mcp import vault
    from agent_mcp.facts import _extract_entities_from_query
    from app import djev
    from eval.run_eval import _corpus_provenance, _score

    queries = (yaml.safe_load(Path(args.queries).read_text()) or {}).get("queries") or []
    if args.max_queries:
        queries = queries[: args.max_queries]
    replay = args.pass_ == "quality"
    mode_box = {"mode": "off"}
    vault.recall_topics_merge_mode = lambda: mode_box["mode"]  # the one knob the arms move

    def seeds_for(q):
        return [e for e, _ in (_extract_entities_from_query(q) or [])[:vault.RECALL_SEED_TOP_K]]

    recs = []
    for i, spec in enumerate(queries):
        q = spec.get("query") or ""
        if not q:
            continue
        seeds = seeds_for(q)
        rec = {"id": spec.get("id"), "arms": {}}
        for arm in ARMS:
            mode_box["mode"] = MODES[arm]
            if replay:
                os.environ[djev.REPLAY_ARM_ENV] = arm
                os.environ[djev.REPLAY_ANCHOR_ENV] = "control"
            t0 = time.perf_counter()
            res = vault._vault_recall({"query": q, "limit": args.limit, "expand_graph": True},
                                      seed_top_k=vault.RECALL_SEED_TOP_K)
            ms = (time.perf_counter() - t0) * 1000
            if "error" in res:
                rec["arms"][arm] = {"scoring": {}, "ms": ms, "error": str(res.get("error"))[:200]}
                continue
            rec["arms"][arm] = {"scoring": _score(spec, res, seeds), "ms": round(ms, 1)}
            tm = res.get("topics_merge")
            if tm and arm == "facts":
                rec["topics"] = tm.get("topics")
        recs.append(rec)
        if args.leg == "dev":
            print(f"[{i + 1}/{len(queries)}] {rec['id']} topics={rec.get('topics')}", flush=True)
        else:
            print(f"[{i + 1}/{len(queries)}]", flush=True)
    summary = compare(recs)
    summary["errors"] = {arm: sum(1 for r in recs if r["arms"][arm].get("error")) for arm in ARMS}
    out = {"item": 1456, "pass": args.pass_, "leg": args.leg,
           "ran_at": datetime.now(timezone.utc).isoformat(), "replay": replay,
           "limit": args.limit, "corpus": _corpus_provenance(), "summary": summary}
    if replay:
        out["djev_replay"] = djev.replay_stats(os.environ.get(djev.REPLAY_ENV))
    if args.leg == "dev":
        out["records"] = recs          # the holdout leg writes aggregates only (#1412)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    return 0


def parent(args) -> int:
    from scripts.automod.evalpin import PinnedCorpus
    from app import djev
    work = Path(tempfile.mkdtemp(prefix="topics-merge-", dir=args.workdir))
    out = Path(args.out).with_suffix(f".{args.pass_}.{args.leg}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PinnedCorpus(work, name="topicsmerge", port=args.port) as pin:
            pin.warm_up()
            env = pin.env_for(code_root=ROOT)
            env["PYTHONPATH"] = str(ROOT)
            # The fact half of the corpus, pinned too: a copy of the fact tree and
            # a VACUUM INTO of the store, never the live files.
            env["LLOYD_FACTS_ROOT"] = str(Path(args.fact_pin) / "facts")
            env["LLOYD_KG_DB"] = str(Path(args.fact_pin) / "kg.sqlite")
            if args.pass_ == "quality":
                env.update(djev.replay_env(work / "replay.sqlite", "control", "control"))
            if args.leg == "holdout":
                from eval import retrieval_holdout
                env[retrieval_holdout.HOLDOUT_LEG_ENV] = "1"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
                   "--pass", args.pass_, "--leg", args.leg, "--queries", args.queries,
                   "--limit", str(args.limit), "--max-queries", str(args.max_queries),
                   "--out", str(out)]
            r = subprocess.run(cmd, cwd=str(ROOT), env=env)
            if r.returncode:
                return r.returncode
            data = json.loads(out.read_text())
            data["pin"] = {k: v for k, v in pin.provenance.items() if k != "daemon"}
            out.write_text(json.dumps(data, indent=1, default=str))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    s = json.loads(out.read_text())["summary"]
    print(json.dumps({"means": s["means"], "latency": s["latency"], "topics": s["topics"],
                      "errors": s.get("errors")}, indent=1))
    for arm, rows in s["paired_vs_control"].items():
        for m, v in rows.items():
            print(f"{arm:10s} {m:24s} d={v['delta']:+.4f} ci={v['ci95']} p={v['p']} "
                  f"W/L={v['wins']}/{v['losses']} n={v['n']}")
    print(f"[info] wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pass", dest="pass_", choices=("quality", "latency"), required=True)
    ap.add_argument("--leg", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--queries", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--port", type=int, default=8183)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--fact-pin", required=False,
                    default=str(Path.home() / "lloyd-data/eval/pin-2026-09-25"),
                    help="directory holding facts/ and kg.sqlite copies")
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args(argv)
    if args.queries is None:
        from eval import retrieval_holdout
        args.queries = str(retrieval_holdout.HOLDOUT_QUERIES if args.leg == "holdout"
                           else retrieval_holdout.DEV_QUERIES)
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    sys.exit(main())
