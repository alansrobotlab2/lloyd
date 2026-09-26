#!/usr/bin/env python3
"""#1491: answer-bearing sentences as djev's ranked row, paired against today's row.

Today (`agent_mcp.vault._djev_doc_text`, #1467) djev ranks `title + the stripped
qmd snippet`, cut at 160 characters. qmd's snippet is a ~4-line window chosen by
where the match sits, not by whether it answers. This arm reads the ~1,200-char
window around that match from the note on disk (qmd's own rerank window width,
`QMD_RERANK_WINDOW_CHARS`), splits it into sentences, keeps the sentences that
carry the most query terms and hands djev `title + those sentences in document
order` at the SAME 160 characters. A Provence-style pruner, lexical, with no
model: if the selection does not move the ranking, a learned one is not worth
a GPU-0 slot either.

Arms per query, interleaved, replay OFF (the arm changes djev's input, so each
row is a fresh read; `control_b` repeats production and is the run's noise):
`control`, `answer_bearing`, `control_b`. Documents, facts and the pool are
identical across arms — only the row djev reads differs, so doc_hit /
doc_recall can only move through what falls off the 20-row cut.

The same selection also prices the model-facing half: the snippet a recall
returns (`chars_snippet`) against the pruned window (`chars_pruned`).

    flock -x ~/.local/state/lloyd-automod/regression.lock \\
      .venvs/lloyd/bin/python eval/answer_bearing_rows.py --out <stem>
"""
from __future__ import annotations

import argparse
import json
import re
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

ARMS = ("control", "answer_bearing", "control_b")
WINDOW_CHARS = 1200
ROW_CHARS = 160
_WORD = re.compile(r"[a-z0-9][a-z0-9_\-\.]{2,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NOISE = re.compile(r"^[\s#>*\-|`:]+|[\s|`]+$")


def query_terms(query: str, stop: set[str]) -> set[str]:
    return {w.strip(".-") for w in _WORD.findall(query.lower())} - stop - {""}


def sentences(text: str) -> list[str]:
    out = []
    for s in _SENT_SPLIT.split(text or ""):
        s = _NOISE.sub("", s).strip()
        if len(s) >= 12:
            out.append(s)
    return out


def select(sents: list[str], terms: set[str], budget: int) -> str:
    """The sentences carrying the most distinct query terms, in document order,
    until `budget` characters. With no term anywhere the window's head stands in."""
    scored = []
    for i, s in enumerate(sents):
        toks = {w.strip(".-") for w in _WORD.findall(s.lower())}
        scored.append((len(terms & toks), i, s))
    if not scored:
        return ""
    if max(sc for sc, _, _ in scored) == 0:
        return " ".join(s for _, _, s in scored)[:budget]
    keep, used = [], 0
    for sc, i, s in sorted(scored, key=lambda t: (-t[0], t[1])):
        if sc == 0 or used >= budget:
            break
        keep.append((i, s))
        used += len(s) + 1
    return " ".join(s for _, s in sorted(keep))[:budget]


def window_text(path: str, line: int | None, resolve) -> str | None:
    """~WINDOW_CHARS of the note around `line` (1-based), or its head."""
    f = resolve(path)
    if f is None:
        return None
    try:
        lines = f.read_text(errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return ""
    at = max(0, min(len(lines) - 1, (line or 1) - 1))
    lo, hi = at, at + 1
    size = len(lines[at])
    while size < WINDOW_CHARS and (lo > 0 or hi < len(lines)):
        if hi < len(lines):
            size += len(lines[hi]) + 1
            hi += 1
        if lo > 0 and size < WINDOW_CHARS:
            lo -= 1
            size += len(lines[lo]) + 1
    return "\n".join(lines[lo:hi])[:WINDOW_CHARS * 2]


def child(args) -> int:
    import yaml
    from agent_mcp import vault
    # The topics merge (#1456) is its own arm with its own primary call; off here.
    vault.recall_topics_merge_mode = lambda: "off"
    from agent_mcp._shared import _QUERY_STOPWORDS
    from agent_mcp.facts import _extract_entities_from_query
    from eval.recall_topics_merge import METRICS, _num, _summ
    from eval import stats as evstats
    from eval.run_eval import _corpus_provenance, _score

    queries = (yaml.safe_load(Path(args.queries).read_text()) or {}).get("queries") or []
    if args.max_queries:
        queries = queries[: args.max_queries]
    stop = {w.lower() for w in _QUERY_STOPWORDS}
    production_row = vault._djev_doc_text
    box = {"arm": "control", "query": "", "rows": []}

    def arm_row(doc: dict) -> str:
        if box["arm"] != "answer_bearing":
            return production_row(doc)
        title = str(doc.get("title") or doc.get("path") or "")
        snippet, line = vault.strip_qmd_snippet(str(doc.get("snippet") or ""))
        win = window_text(str(doc.get("path") or ""), line, vault._resolve_case_insensitive)
        budget = max(0, ROW_CHARS - len(title) - 1)
        picked = select(sentences(win), query_terms(box["query"], stop), budget) if win else ""
        box["rows"].append({"resolved": win is not None, "chars_snippet": len(snippet),
                            "chars_window": len(win or ""),
                            "chars_pruned_600": len(select(sentences(win or ""),
                                                           query_terms(box["query"], stop), 600))})
        body = picked or snippet
        return f"{title}\n{body}" if body else title

    vault._djev_doc_text = arm_row

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
            box.update(arm=arm, query=q)
            t0 = time.perf_counter()
            res = vault._vault_recall({"query": q, "limit": args.limit, "expand_graph": True},
                                      seed_top_k=vault.RECALL_SEED_TOP_K)
            ms = (time.perf_counter() - t0) * 1000
            if "error" in res:
                rec["arms"][arm] = {"scoring": {}, "ms": ms, "error": str(res["error"])[:200]}
                continue
            rec["arms"][arm] = {"scoring": _score(spec, res, seeds), "ms": round(ms, 1)}
        recs.append(rec)
        print(f"[{i + 1}/{len(queries)}] {rec['id']}", flush=True)

    out_s: dict = {"n_queries": len(recs), "means": {}, "paired_vs_control": {}, "latency": {}}
    for arm in ARMS:
        out_s["means"][arm] = {}
        for metric, field in METRICS.items():
            vals = [_num(r["arms"][arm]["scoring"].get(field)) for r in recs]
            vals = [v for v in vals if v is not None]
            out_s["means"][arm][metric] = round(statistics.fmean(vals), 4) if vals else None
        out_s["latency"][arm] = _summ([r["arms"][arm]["ms"] for r in recs])
    for arm in ARMS[1:]:
        out_s["paired_vs_control"][arm] = {}
        for metric, field in METRICS.items():
            a = [_num(r["arms"]["control"]["scoring"].get(field)) for r in recs]
            b = [_num(r["arms"][arm]["scoring"].get(field)) for r in recs]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if not pairs:
                continue
            ci = evstats.paired_bootstrap_ci([x for x, _ in pairs], [y for _, y in pairs])
            out_s["paired_vs_control"][arm][metric] = {
                "delta": round(ci["diff"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                "p": round(ci["p"], 4), "n": ci["n"],
                "wins": sum(1 for x, y in pairs if y > x),
                "losses": sum(1 for x, y in pairs if y < x)}
    rows = box["rows"]
    out_s["rows"] = {
        "n": len(rows), "resolved": sum(1 for r in rows if r["resolved"]),
        "chars_snippet": _summ([r["chars_snippet"] for r in rows]),
        "chars_window": _summ([r["chars_window"] for r in rows if r["resolved"]]),
        "chars_pruned_600": _summ([r["chars_pruned_600"] for r in rows if r["resolved"]]),
    }
    out_s["errors"] = {arm: sum(1 for r in recs if r["arms"][arm].get("error")) for arm in ARMS}
    out = {"item": 1491, "ran_at": datetime.now(timezone.utc).isoformat(),
           "window_chars": WINDOW_CHARS, "row_chars": ROW_CHARS,
           "corpus": _corpus_provenance(), "summary": out_s, "records": recs}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    return 0


def parent(args) -> int:
    from scripts.automod.evalpin import PinnedCorpus
    work = Path(tempfile.mkdtemp(prefix="abrows-"))
    out = Path(args.out).with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PinnedCorpus(work, name="topicsmerge", port=args.port) as pin:
            pin.warm_up()
            env = pin.env_for(code_root=ROOT)
            env["PYTHONPATH"] = str(ROOT)
            env["LLOYD_FACTS_ROOT"] = str(Path(args.fact_pin) / "facts")
            env["LLOYD_KG_DB"] = str(Path(args.fact_pin) / "kg.sqlite")
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child",
                                "--queries", args.queries, "--limit", str(args.limit),
                                "--max-queries", str(args.max_queries), "--out", str(out)],
                               cwd=str(ROOT), env=env)
            if r.returncode:
                return r.returncode
    finally:
        shutil.rmtree(work, ignore_errors=True)
    s = json.loads(out.read_text())["summary"]
    print(json.dumps({k: s[k] for k in ("means", "latency", "rows", "errors")}, indent=1))
    for arm, rows in s["paired_vs_control"].items():
        for m, v in rows.items():
            print(f"{arm:15s} {m:24s} d={v['delta']:+.4f} ci={v['ci95']} p={v['p']} "
                  f"W/L={v['wins']}/{v['losses']} n={v['n']}")
    print(f"[info] wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queries", default=str(HERE / "vault_recall_queries.yaml"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--port", type=int, default=8183)
    ap.add_argument("--fact-pin", default=str(Path.home() / "lloyd-data/eval/pin-2026-09-25"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args(argv)
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    sys.exit(main())
