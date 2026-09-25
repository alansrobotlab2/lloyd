"""GPU 2 bake-off: djev (with and without thinking) against the decider models.

One harness, three of Lloyd's own call shapes, every arm sent byte-identical
requests, one request at a time (djev runs MAX_SEQS=1, so production is serial):

  recall     the vault recall ranking. The 87-query gold set's candidate pools
             are captured ONCE (`pools`) through the production recall path
             with the ranker stubbed out, so every arm ranks the same ≤32 rows;
             the request is `app.djev`'s own rank body (160-char candidates,
             four score levels, samples 1) and the ordering is production's
             (expected level, descending, stable). Scored with the recall
             eval's `_score` on the top 20, as production cuts.
  jevbench   JevBench's 231 public items (easy 48, standard 72, hard 111),
             requests built by JevBench's TypeSafe adapter and scored with its
             own `score_task`, so the numbers mean what the leaderboard's do.
  addressee  the voice gate's 46 labelled cases (scripts/voice/addressee_eval.py),
             state and question from `voice.addressee`, no samples (as sent),
             judged at the configured 0.7.

`think` (djev only) is a top-level request field: the model writes up to N
tokens of thought, once per request, and the read conditions on it. The
decider server ignores it. Nothing in production sends it today.

    python eval/djev/bakeoff.py pools  --out DIR/pools.json
    python eval/djev/bakeoff.py run    --arm djev-t0 --url http://127.0.0.1:8011 \\
                                       --pools DIR/pools.json --jevbench JEVBENCH_CLONE --out DIR
    python eval/djev/bakeoff.py report DIR

Run it from the production tree (cwd ~/lloyd): `pools` calls the live qmd
daemon through `agent_mcp.vault`, and the recall scorer reads the gold set.
Hold `regression.lock` for the whole bake-off: it borrows djev and GPU 2.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-services"))

RECALL_CHARS = 160      # vault.RECALL_DJEV_CHARS
RECALL_POOL = 32        # vault.RECALL_DJEV_POOL
RECALL_LIMIT = 20       # the recall's default output cut
ADDRESSEE_THRESHOLD = 0.7
TIERS = {"easy.jsonl": "easy", "original.jsonl": "standard", "hard.jsonl": "hard"}


# ── transport ─────────────────────────────────────────────────────────

def post(url: str, body: dict, timeout: float) -> tuple[int, dict | None, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + "/v1/systemone", data=data,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
            return r.status, out, (time.perf_counter() - t0) * 1e3
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:300]}, (time.perf_counter() - t0) * 1e3
    except Exception as e:  # noqa: BLE001 — a timeout is a result, recorded
        return 0, {"error": f"{type(e).__name__}: {e}"}, (time.perf_counter() - t0) * 1e3


def with_think(body: dict, think: int) -> dict:
    return {**body, "think": think} if think else body


def thought_of(resp: dict | None) -> dict:
    t = ((resp or {}).get("diagnostics") or {}).get("thought") or {}
    return {k: t.get(k) for k in ("tokens", "closed", "ms")} if t else {}


# ── pools ─────────────────────────────────────────────────────────────

def cmd_pools(args) -> int:
    import yaml
    from agent_mcp import vault
    import eval.run_eval as RE

    spec = yaml.safe_load((ROOT / "eval" / "vault_recall_queries.yaml").read_text())
    captured: dict[str, list[dict]] = {}

    def capture(documents, query):
        captured[query] = [{"path": d.get("path"), "title": d.get("title"),
                            "snippet": d.get("snippet"), "text": vault._djev_doc_text(d)}
                           for d in documents[:RECALL_POOL]]
        return documents  # fusion order: the ranker is what is being compared

    vault._djev_rank_recall = capture
    pools = []
    for q in spec["queries"]:
        captured.clear()
        RE._vault_recall({"query": q["query"], "limit": RECALL_LIMIT}, seed_top_k=RE.RECALL_SEED_TOP_K)
        pool = captured.get(q["query"])
        if pool is None:
            print(f"  {q['id']}: no pool (the recall did not reach the ranker)", file=sys.stderr)
            continue
        pools.append({"id": q["id"], "query": q["query"], "spec": q, "pool": pool})
    Path(args.out).write_text(json.dumps({"captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          "pools": pools}, indent=1))
    sizes = [len(p["pool"]) for p in pools]
    print(f"{len(pools)} pools, rows p50 {statistics.median(sizes)} min {min(sizes)} max {max(sizes)} -> {args.out}")
    return 0


# ── tasks ─────────────────────────────────────────────────────────────

def run_recall(args, emit):
    from app import djev
    import eval.run_eval as RE
    pools = json.loads(Path(args.pools).read_text())["pools"]
    if args.limit:
        pools = pools[: args.limit]
    for i, p in enumerate(pools):
        texts = [c["text"] for c in p["pool"]]
        body = djev._body(djev.rank_state(p["query"], texts, chars=RECALL_CHARS),
                          djev.rank_questions(texts), samples=1, instructions=None, seed=None)
        status, resp, ms = post(args.url, with_think(body, args.think), args.timeout)
        answers = (resp or {}).get("answers") or {}
        scores = []
        for k in range(len(texts)):
            a = answers.get(f"c{k}")
            scores.append(float(a["score"]) if isinstance(a, dict) and "score" in a else None)
        ok = status == 200 and all(s is not None for s in scores)
        chunks = ((resp or {}).get("diagnostics") or {}).get("chunks") or []
        order = sorted(range(len(texts)), key=lambda k: -(scores[k] or 0.0)) if ok else list(range(len(texts)))
        docs = [{"path": p["pool"][k]["path"]} for k in order][:RECALL_LIMIT]
        m = RE._score(p["spec"], {"documents": docs, "facts": []}, [])
        emit({"task": "recall", "id": p["id"], "ok": ok, "status": status, "ms": ms,
              "n": len(texts), "chunks": len(chunks), "thought": thought_of(resp),
              "doc_hit": m.get("doc_hit"), "rr": m.get("rr_doc"), "ndcg10": m.get("ndcg10"),
              "error": None if ok else (resp or {}).get("error")})
        if args.progress and i % 10 == 0:
            print(f"  recall {i + 1}/{len(pools)} {ms:.0f} ms", flush=True)


def run_jevbench(args, emit):
    sys.path.insert(0, args.jevbench)
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.metrics import brier_score
    from jevbench.scoring import score_task
    from jevbench.tasks import load_jsonl

    class Adapter(TypeSafeAdapter):
        def build_request(self, task):
            return with_think(super().build_request(task), args.think)

    ad = Adapter(endpoint=args.url, model="djev", key_env=None, timeout_s=args.timeout)
    for fname, tier in TIERS.items():
        tasks = load_jsonl(str(Path(args.jevbench) / "datasets" / "public" / fname))
        if args.limit:
            tasks = tasks[: args.limit]
        for i, t in enumerate(tasks):
            r = ad.run(t)
            s = score_task(r.probs, t) if r.ok else {}
            clean = s.get("probs") if s.get("valid") else None
            emit({"task": "jevbench", "tier": tier, "id": t.id, "family": t.family,
                  "type": t.question.get("type"), "ok": r.ok, "valid": bool(s.get("valid")),
                  "ms": (r.latency_s or 0) * 1e3, "correct": s.get("correct"),
                  "confidence": max(clean.values()) if clean else None,
                  "brier": brier_score(clean, str(t.expected), [str(x) for x in t.labels]) if clean and t.expected is not None else None,
                  "thought": thought_of(r.raw if isinstance(r.raw, dict) else None),
                  "error": r.error})
            if args.progress and i % 20 == 0:
                print(f"  jevbench {tier} {i + 1}/{len(tasks)} {(r.latency_s or 0) * 1e3:.0f} ms", flush=True)


def run_addressee(args, emit):
    sys.path.insert(0, str(ROOT / "scripts" / "voice"))
    from voice import addressee as A
    import addressee_eval as AE
    cases = AE.CASES[: args.limit] if args.limit else AE.CASES
    for i, (label, last, since, utt, name) in enumerate(cases):
        state = A.state_text(utt, last, since, None, name)
        body = {"model": "djev", "state": state, "questions": A.QUESTION}
        status, resp, ms = post(args.url, with_think(body, args.think), args.timeout)
        a = ((resp or {}).get("answers") or {}).get("addressed")
        p = float(a["noul"]) if status == 200 and isinstance(a, dict) and "noul" in a else None
        emit({"task": "addressee", "id": i, "label": label, "p": p, "ok": p is not None, "ms": ms,
              "thought": thought_of(resp), "error": None if p is not None else (resp or {}).get("error")})


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.arm}.jsonl"
    tasks = args.tasks.split(",")
    with path.open("a") as fh:
        def emit(row):
            fh.write(json.dumps({"arm": args.arm, "think": args.think, **row}) + "\n")
            fh.flush()
        # Warm-up, discarded: one request of each shape, so a first-request
        # JIT or graph capture is not charged to the first item.
        if not args.no_warmup:
            print(f"[{args.arm}] warm-up", flush=True)
            post(args.url, with_think({"model": "djev", "state": "warm-up", "questions": {
                "q": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}}}}, args.think), args.timeout)
        for t in tasks:
            t0 = time.time()
            print(f"[{args.arm}] {t}", flush=True)
            {"recall": run_recall, "jevbench": run_jevbench, "addressee": run_addressee}[t](args, emit)
            print(f"[{args.arm}] {t} done in {time.time() - t0:.0f} s", flush=True)
    return 0


# ── report ────────────────────────────────────────────────────────────

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def paired_ci(a: dict, b: dict, n_boot=4000, seed=7):
    """Bootstrap 95% interval of mean(b - a) over shared ids."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 5:
        return None
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return (sum(d) / len(d), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)])


def cmd_report(args) -> int:
    rows = [json.loads(ln) for f in sorted(Path(args.dir).glob("*.jsonl")) for ln in f.read_text().splitlines() if ln.strip()]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    base = args.baseline if args.baseline in arms else (arms[0] if arms else None)
    by = lambda arm, task: [r for r in rows if r["arm"] == arm and r["task"] == task]  # noqa: E731
    fmt = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    out = []

    out.append("## Recall ranking (87-query gold set, identical pools, top 20)\n")
    out.append("| arm | answered | doc_hit | MRR | NDCG@10 | MRR vs " + str(base) + " [95% CI] | p50 ms | p95 ms | thought tokens p50 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    b_rr = {r["id"]: r["rr"] for r in by(base, "recall") if r["rr"] is not None}
    for a in arms:
        rs = by(a, "recall")
        if not rs:
            continue
        rr = {r["id"]: r["rr"] for r in rs if r["rr"] is not None}
        ci = paired_ci(b_rr, rr) if a != base else None
        ci_s = "" if a == base else ("—" if ci is None else f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
        ok = sum(r["ok"] for r in rs)
        out.append(f"| {a} | {ok}/{len(rs)} | {fmt(mean([r['doc_hit'] for r in rs]))} | {fmt(mean(rr.values()))} | "
                   f"{fmt(mean([r['ndcg10'] for r in rs]))} | {ci_s} | {fmt(pct([r['ms'] for r in rs], .5), 0)} | "
                   f"{fmt(pct([r['ms'] for r in rs], .95), 0)} | {fmt(pct([(r['thought'] or {}).get('tokens') for r in rs], .5), 0)} |")

    out.append("\n## JevBench public items (JevBench's own scoring)\n")
    out.append("| arm | answered | easy | standard | hard | all | hard vs " + str(base) + " [95% CI] | Brier | p50 ms | p95 ms |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    b_hard = {r["id"]: float(bool(r["correct"])) for r in by(base, "jevbench") if r["tier"] == "hard" and r["ok"]}
    for a in arms:
        rs = by(a, "jevbench")
        if not rs:
            continue
        acc = lambda tier: mean([float(bool(r["correct"])) for r in rs if r["tier"] == tier])  # noqa: E731
        hard = {r["id"]: float(bool(r["correct"])) for r in rs if r["tier"] == "hard" and r["ok"]}
        ci = paired_ci(b_hard, hard) if a != base else None
        ci_s = "" if a == base else ("—" if ci is None else f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
        out.append(f"| {a} | {sum(r['ok'] for r in rs)}/{len(rs)} | {fmt(acc('easy'))} | {fmt(acc('standard'))} | {fmt(acc('hard'))} | "
                   f"{fmt(mean([float(bool(r['correct'])) for r in rs]))} | {ci_s} | {fmt(mean([r['brier'] for r in rs]))} | "
                   f"{fmt(pct([r['ms'] for r in rs], .5), 0)} | {fmt(pct([r['ms'] for r in rs], .95), 0)} |")

    out.append(f"\n## Voice addressee (46 labelled cases, threshold {ADDRESSEE_THRESHOLD})\n")
    out.append("| arm | answered | accuracy | false accepts | missed | AUC | p50 ms | p95 ms |")
    out.append("|---|---|---|---|---|---|---|---|")
    for a in arms:
        rs = [r for r in by(a, "addressee")]
        if not rs:
            continue
        ans = [r for r in rs if r["p"] is not None]
        fa = sum(1 for r in ans if not r["label"] and r["p"] >= ADDRESSEE_THRESHOLD)
        miss = sum(1 for r in ans if r["label"] and r["p"] < ADDRESSEE_THRESHOLD)
        accn = sum(1 for r in ans if (r["p"] >= ADDRESSEE_THRESHOLD) == r["label"])
        out.append(f"| {a} | {len(ans)}/{len(rs)} | {fmt(accn / len(ans) if ans else None)} | {fa} | {miss} | "
                   f"{fmt(auc([r['p'] for r in ans if r['label']], [r['p'] for r in ans if not r['label']]))} | "
                   f"{fmt(pct([r['ms'] for r in rs], .5), 0)} | {fmt(pct([r['ms'] for r in rs], .95), 0)} |")
    text = "\n".join(out)
    print(text)
    if args.write:
        Path(args.write).write_text(text + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pools")
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--think", type=int, default=0)
    r.add_argument("--tasks", default="recall,jevbench,addressee")
    r.add_argument("--pools", required=True)
    r.add_argument("--jevbench", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--no-warmup", action="store_true")
    r.add_argument("--progress", action="store_true")
    s = sub.add_parser("report")
    s.add_argument("dir")
    s.add_argument("--baseline", default="djev-t0")
    s.add_argument("--write", default=None)
    args = ap.parse_args()
    return {"pools": cmd_pools, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
