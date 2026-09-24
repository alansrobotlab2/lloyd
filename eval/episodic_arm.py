#!/usr/bin/env python3
"""The episodic raw-transcript retrieval arm, measured offline (#675).

The question: do the exported chat transcripts — raw turns, no fact or entity
extraction — hold answers the production recall path misses, and at what
latency and token cost? The mechanism under test is the adjacency window: a
retrieved transcript is cut down to the turn that matched plus ``k`` turns
either side, packed to a token budget, and ``k=0`` is the control.

Three pieces, each small enough to pin in a test:

* **Turns** (`parse_turns`). The unit is the markdown `app/post_capture.py`
  exports: a ``user:`` or ``lloyd:`` line opens a turn; ``tool_call:`` lines and
  ``→ [OK]`` / ``→ [ERROR]`` tool-result lines (``[ERR]`` accepted too) attach
  to the turn they follow, and continuation lines to whatever block they
  continue. `include_kinds` decides which of those reach the scored text, and
  the run records the set it used.
* **Expansion** (`expand`). The matched turn, then its neighbours nearest-first
  (before, after, before-2, …) while each whole turn fits the budget. The first
  neighbour that does not fit ends the expansion, so an episode is always a
  contiguous window; a matched turn larger than the budget is cut to it.
* **Gold** (`labels_for`, `score_query`). An episode is gold if its text holds
  any ``expect_entities`` name or ``expect_docs`` path (the full path, or its
  file stem). A query whose labels appear in NO turn of the corpus has no gold
  episode at all; it is counted and left out of the arm's rates rather than
  scored as a miss, because no retriever could have answered it from here.

This is a text-containment scorer, which is more lenient than the nightly's
entity leg (that one asks whether the *fact layer* returned the entity). So the
same scorer is also run over the documents the production recall returned
(`--baseline` records' ``doc_paths_top10``, paragraphs as turns) — that control
is what makes "the episodic arm adds an answer" mean "adds one the vault path's
own text did not carry", not "a looser ruler found more".

Retrieval itself is `eval/episodic_arm_search.mjs`: qmd's hybrid search over a
scratch index of the corpus, with the daemon's own models.

    .venvs/lloyd/bin/python eval/episodic_arm.py \\
        --baseline ~/lloyd-data/eval/baselines/nightly-<date>.json \\
        --corpus-dir <dir of exported session markdown> --work-dir <scratch>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
if str(LLOYD_HOME) not in sys.path:
    sys.path.insert(0, str(LLOYD_HOME))

try:
    from eval import stats as evstats
except ImportError:  # pragma: no cover - script-dir invocation
    import stats as evstats

OUT_DIR = HERE / "episodic-arm"
KS = (0, 2, 5)
# Per-episode budget, in estimated tokens. 1,024 is roughly what one prefetch
# document costs today; ten of them is the ~10k a recall's document block is.
BUDGET_TOKENS = 1024
LIMIT = 10
ALL_KINDS = ("user", "lloyd", "tool_call", "tool_result")
PROSE_KINDS = ("user", "lloyd")
# Text that means the transcript is ABOUT the retrieval eval, not about the
# subject of the query: a turn that printed the gold set or a run's scores
# contains every label by construction. Episodes carrying one are counted and,
# in the sensitivity row, dropped from the ranking.
EVAL_MARKERS = ("vault_recall_queries", "run_eval", "entity_hit", "expect_entities",
                "expect_docs", "ndcg10", "doc_hit_rate")

_TURN_RE = re.compile(r"^(user|lloyd): ?")
_CALL_RE = re.compile(r"^tool_call: ?")
_RESULT_RE = re.compile(r"^\s*→ \[(OK|ERR|ERROR)\] ?")


def est_tokens(text: str) -> int:
    """Estimated tokens: ceil(chars / 4). Declared, not measured per model."""
    return math.ceil(len(text) / 4)


def _norm(s: str) -> str:
    """The nightly scorer's normalisation (eval/run_eval.py `_norm`)."""
    return str(s or "").lower().replace("-", "_")


@dataclass
class Turn:
    kind: str                 # "user", "lloyd" or "preamble"
    start: int                # char offset of the turn's first line
    end: int                  # char offset one past its last line
    parts: list = field(default_factory=list)   # [(kind, text)]

    def text(self, include_kinds=ALL_KINDS) -> str:
        return "\n".join(t for k, t in self.parts if k in include_kinds)


def parse_turns(doc: str) -> list[Turn]:
    """Split one exported transcript into turns. See the module docstring."""
    turns: list[Turn] = []
    pos = 0
    in_header = True
    for line in doc.splitlines(keepends=True):
        start, pos = pos, pos + len(line)
        body = line.rstrip("\n")
        if in_header and (body.startswith("# ") or not body.strip()):
            continue
        in_header = False
        m = _TURN_RE.match(body)
        if m:
            turns.append(Turn(kind=m.group(1), start=start, end=pos,
                              parts=[(m.group(1), body)]))
            continue
        if _CALL_RE.match(body):
            kind = "tool_call"
        elif _RESULT_RE.match(body):
            kind = "tool_result"
        else:
            kind = None           # a continuation of the block above
        if not turns:
            turns.append(Turn(kind="preamble", start=start, end=pos, parts=[]))
        cur = turns[-1]
        cur.end = pos
        if kind is None:
            if cur.parts:
                k, t = cur.parts[-1]
                cur.parts[-1] = (k, t + "\n" + body)
            else:
                cur.parts.append(("tool_result", body))
        else:
            cur.parts.append((kind, body))
    return turns


def _qtokens(query: str) -> set[str]:
    return {w for w in re.findall(r"\w+", query.lower()) if len(w) >= 3}


def matched_turn(turns: list[Turn], pos: int, span: int, query: str,
                 include_kinds=ALL_KINDS) -> int:
    """The turn a hit is anchored on: among the turns overlapping qmd's best
    chunk ``[pos, pos+span)``, the one sharing most query terms (earliest on a
    tie); the turn containing ``pos`` when none overlaps."""
    if not turns:
        return -1
    lo, hi = pos, pos + max(span, 1)
    cand = [i for i, t in enumerate(turns) if t.start < hi and t.end > lo]
    if not cand:
        cand = [min(range(len(turns)), key=lambda i: abs(turns[i].start - pos))]
    q = _qtokens(query)
    return max(cand, key=lambda i: (len(q & _qtokens(turns[i].text(include_kinds))), -i))


@dataclass
class Episode:
    turn_indices: list
    text: str
    tokens: int
    truncated: bool


def expand(turns: list[Turn], idx: int, k: int, budget: int = BUDGET_TOKENS,
           include_kinds=ALL_KINDS) -> Episode:
    """The matched turn plus up to ``k`` turns each side, clamped at the file's
    ends, nearest-first, whole turns only, within ``budget`` tokens."""
    texts = [t.text(include_kinds) for t in turns]
    first = texts[idx]
    truncated = False
    if est_tokens(first) > budget:
        first, truncated = first[: budget * 4], True
    chosen = {idx: first}
    used = est_tokens(first)
    for d in range(1, k + 1):
        if truncated:
            break
        for j in (idx - d, idx + d):
            if not 0 <= j < len(turns):
                continue
            cost = est_tokens(texts[j])
            if used + cost > budget:
                truncated = True
                break
            chosen[j] = texts[j]
            used += cost
    order = sorted(chosen)
    text = "\n".join(chosen[i] for i in order if chosen[i])
    return Episode(turn_indices=order, text=text, tokens=est_tokens(text),
                   truncated=truncated)


def labels_for(spec: dict) -> tuple[list[str], list[list[str]]]:
    """(entity labels, doc labels) for one gold query. A doc label is the list
    of strings any of which satisfies it: the path and, when it has one of at
    least 8 characters, its file stem — a transcript names a file far more
    often than it spells its vault path."""
    ents = [str(e) for e in (spec.get("expect_entities") or []) if str(e).strip()]
    docs = []
    for d in (spec.get("expect_docs") or []):
        d = str(d).strip()
        if not d:
            continue
        forms = [d]
        stem = Path(d).stem if d.endswith(".md") else Path(d).name
        if len(stem) >= 8 and stem != d:
            forms.append(stem)
        docs.append(forms)
    return ents, docs


def _has(label: str, text_n: str) -> bool:
    return _norm(label) in text_n


def gold_flags(spec: dict, text: str) -> dict:
    ents, docs = labels_for(spec)
    t = _norm(text)
    ent_hit = [e for e in ents if _has(e, t)]
    doc_hit = [forms[0] for forms in docs if any(_has(f, t) for f in forms)]
    return {"entities": ent_hit, "docs": doc_hit, "gold": bool(ent_hit or doc_hit)}


def is_contaminated(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in EVAL_MARKERS)


def _ndcg(rel: list[int], k: int = 10) -> float:
    rel = rel[:k]
    n = sum(rel)
    if not n:
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    return dcg / sum(1.0 / math.log2(i + 2) for i in range(n))


def score_query(spec: dict, episodes: list[Episode]) -> dict:
    """The nightly's six metrics over a ranked list of episodes, by text
    containment. ``rr_doc`` / ``ndcg10`` rank gold EPISODES (the clause-3
    definition); IDCG counts the gold found, as run_eval's `_ndcg_at_k` does."""
    ents, docs = labels_for(spec)
    flags = [gold_flags(spec, ep.text) for ep in episodes]
    got_e = {e for f in flags for e in f["entities"]}
    got_d = {d for f in flags for d in f["docs"]}
    rel = [1 if f["gold"] else 0 for f in flags]
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    return {
        "entity_hit": bool(got_e),
        "doc_hit": bool(got_d),
        "entity_recall": (len(got_e) / len(ents)) if ents else None,
        "doc_recall": (len(got_d) / len(docs)) if docs else None,
        "rr_doc": round(1.0 / first, 4) if first else 0.0,
        "ndcg10": round(_ndcg(rel), 4),
        "first_gold_rank": first,
        "entities_matched": sorted(got_e),
        "docs_matched": sorted(got_d),
    }


def corpus_presence(spec: dict, corpus_texts_n: list[str]) -> dict:
    """Which labels appear in ANY turn of the corpus (normalised texts)."""
    ents, docs = labels_for(spec)
    e_in = [e for e in ents if any(_norm(e) in t for t in corpus_texts_n)]
    d_in = [forms[0] for forms in docs
            if any(_norm(f) in t for f in forms for t in corpus_texts_n)]
    return {"entities": e_in, "docs": d_in, "any": bool(e_in or d_in)}


METRICS = ("entity_hit", "doc_hit", "entity_recall", "doc_recall", "rr_doc", "ndcg10")


def aggregate(scores: list[dict]) -> dict:
    """Means over the queries given, with the nightly's CI kinds (Wilson for
    the two rates, bootstrap for the averages)."""
    out = {"n": len(scores)}
    for m in METRICS:
        vals = [s[m] for s in scores if s.get(m) is not None]
        vals = [float(v) for v in vals]
        if not vals:
            out[m] = None
            continue
        out[m] = round(sum(vals) / len(vals), 3)
        if m in ("entity_hit", "doc_hit"):
            ci = evstats.wilson_ci(int(sum(vals)), len(vals))
        else:
            b = evstats.bootstrap_mean_ci(vals)
            ci = (b["lo"], b["hi"]) if b["lo"] is not None else None
        out[m + "_ci95"] = [round(ci[0], 3), round(ci[1], 3)] if ci else None
    return out


# ── the vault-doc control: same scorer over what production recall returned ──

def paragraph_turns(doc: str) -> list[Turn]:
    """A vault document cut into blank-line paragraphs, as `Turn`s, so the
    control goes through the same `matched_turn` / `expand` as the arm."""
    turns, pos = [], 0
    for block in re.split(r"(\n\s*\n)", doc):
        if block.strip() and not re.fullmatch(r"\n\s*\n", block):
            turns.append(Turn(kind="lloyd", start=pos, end=pos + len(block),
                              parts=[("lloyd", block.strip())]))
        pos += len(block)
    return turns


def resolve_doc(rel: str, roots: list[Path]) -> Path | None:
    for r in roots:
        p = r / rel
        if p.is_file():
            return p
    return None


# ── the run ──────────────────────────────────────────────────────────────────

def _search(spec_queries: list[dict], args, work: Path) -> dict:
    spec = {
        "qmd_dist": str(args.qmd_dist),
        "db_path": str(work / "episodic.sqlite"),
        "corpus_dir": str(Path(args.corpus_dir).resolve()),
        "models": args.models,
        "limit": args.limit,
        "candidate_limit": 40,
        "queries": spec_queries,
    }
    sp, out = work / "search_spec.json", work / "search_out.json"
    sp.write_text(json.dumps(spec))
    env = dict(os.environ)
    env.update(args.qmd_env)
    subprocess.run(["node", str(HERE / "episodic_arm_search.mjs"), str(sp), str(out)],
                   check=True, env=env)
    return json.loads(out.read_text())


def _qmd_env_from_conf(conf: Path) -> dict:
    """The daemon's QMD_*/CUDA_* environment, read from its supervisor conf so
    the scratch store ranks the way production does."""
    try:
        raw = next(l for l in conf.read_text().splitlines() if l.startswith("environment="))
    except (OSError, StopIteration):
        return {}
    pairs = re.findall(r'(\w+)="([^"]*)"', raw)
    return {k: v for k, v in pairs if k.startswith(("QMD_", "CUDA_", "LD_LIBRARY_PATH"))}


def _qmd_dist_from_conf(conf: Path) -> Path:
    """The SDK entry of the build the daemon serves, read off its `command=`
    (`.../dist/cli/qmd.js` -> `.../dist/index.js`). The fork is not checked
    out in a worktree, so the tree's own `qmd/` is only the fallback."""
    try:
        cmd = next(l for l in conf.read_text().splitlines() if l.startswith("command="))
        cli = next(a for a in cmd.split() if a.endswith("/dist/cli/qmd.js"))
        return Path(cli).parent.parent / "index.js"
    except (OSError, StopIteration):
        from app.paths import LIVE_CHECKOUT
        return LIVE_CHECKOUT / "qmd" / "dist" / "index.js"


def _qmd_models(index_yml: Path) -> dict:
    import yaml
    return (yaml.safe_load(index_yml.read_text()) or {}).get("models") or {}


def run(args) -> dict:
    import yaml
    from agent_mcp.vault import _qmd_sanitize, _qmd_strip_stopwords

    baseline = json.loads(Path(args.baseline).read_text())
    base_by_id = {r["id"]: r for r in baseline["records"]}
    gold = yaml.safe_load(Path(args.queries).read_text())["queries"]
    specs = [q for q in gold if q.get("id") in base_by_id]

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    qs = []
    for q in specs:
        s = _qmd_strip_stopwords(_qmd_sanitize(q["query"]))
        qs.append({"id": q["id"], "lex": s, "vec": s})
    if args.search_json:
        search = json.loads(Path(args.search_json).read_text())
    else:
        search = _search(qs, args, work)
    hits_by_id = {r["id"]: r for r in search["rows"]}

    corpus_dir = Path(args.corpus_dir)
    files = sorted(corpus_dir.rglob("*.md"))
    parsed = {str(f.relative_to(corpus_dir)): parse_turns(f.read_text(errors="replace"))
              for f in files}
    kinds = tuple(args.kinds)
    corpus_n = [_norm(t.text(kinds)) for ts in parsed.values() for t in ts]
    dates = sorted(p.split("/")[0] for p in parsed)
    vault_roots = [Path(p) for p in args.doc_roots]

    records = []
    for q in specs:
        qid = q["id"]
        base = base_by_id[qid]
        row = hits_by_id.get(qid) or {"hits": [], "latency_ms": None, "error": "no row"}
        presence = corpus_presence(q, corpus_n)
        per_k, ctrl_k = {}, {}
        for k in args.ks:
            eps, contaminated = [], 0
            for h in row["hits"][: args.limit]:
                turns = parsed.get(h["path"].split("/", 1)[-1]) or parsed.get(h["path"])
                if not turns:
                    continue
                i = matched_turn(turns, h["best_chunk_pos"], h["best_chunk_len"],
                                 q["query"], kinds)
                ep = expand(turns, i, k, args.budget, kinds)
                contaminated += is_contaminated(ep.text)
                eps.append(ep)
            sc = score_query(q, eps)
            clean = score_query(q, [e for e in eps if not is_contaminated(e.text)])
            sc.update({
                "n_episodes": len(eps),
                "tokens_total": sum(e.tokens for e in eps),
                "tokens_top3": sum(e.tokens for e in eps[:3]),
                "turns_mean": round(statistics.mean(len(e.turn_indices) for e in eps), 2) if eps else 0,
                "n_truncated": sum(e.truncated for e in eps),
                "n_contaminated": contaminated,
                "clean": {m: clean[m] for m in METRICS},
            })
            per_k[str(k)] = sc
            # The control: the same scorer over the documents production returned.
            deps = []
            for rel in base["result_summary"].get("doc_paths_top10", [])[: args.limit]:
                p = resolve_doc(rel, vault_roots)
                if not p:
                    continue
                turns = paragraph_turns(p.read_text(errors="replace"))
                if not turns:
                    continue
                i = matched_turn(turns, 0, len(p.read_text(errors="replace")), q["query"])
                deps.append(expand(turns, i, k, args.budget))
            cs = score_query(q, deps)
            cs["tokens_total"] = sum(e.tokens for e in deps)
            ctrl_k[str(k)] = cs
        records.append({
            "id": qid, "query": q["query"], "category": q.get("category"),
            "expected": {"entities": q.get("expect_entities") or [],
                         "docs": q.get("expect_docs") or []},
            "corpus_presence": presence,
            "zero_gold": not presence["any"],
            "latency_ms": row.get("latency_ms"), "error": row.get("error"),
            "hits": [h["path"] for h in row["hits"][: args.limit]],
            "arm": per_k,
            "vault_doc_text_control": ctrl_k,
            "baseline": {m: base["scoring"].get(m) for m in
                         ("entity_hit", "doc_hit", "entity_recall", "doc_recall",
                          "rr_doc", "ndcg10", "fact_entity_recall")},
        })
    return summarize(records, baseline, args, search, len(files), dates, kinds)


def _union_hit(recs, k, key="entity_hit"):
    return [bool(r["baseline"][key]) or bool(r["arm"][k][key]) for r in recs]


def summarize(records, baseline, args, search, n_files, dates, kinds) -> dict:
    ks = [str(k) for k in args.ks]
    eligible = [r for r in records if not r["zero_gold"]]
    lat = sorted(r["latency_ms"] for r in records if r["latency_ms"] is not None)
    summary = {"n_queries": len(records), "zero_gold_count": len(records) - len(eligible),
               "zero_gold_ids": [r["id"] for r in records if r["zero_gold"]],
               "n_eligible": len(eligible)}
    summary["baseline_all"] = aggregate([r["baseline"] for r in records])
    summary["baseline_eligible"] = aggregate([r["baseline"] for r in eligible])
    for k in ks:
        summary[f"arm_k{k}"] = aggregate([r["arm"][k] for r in eligible])
        summary[f"arm_k{k}_clean"] = aggregate([r["arm"][k]["clean"] for r in eligible])
        summary[f"control_k{k}"] = aggregate([r["vault_doc_text_control"][k] for r in eligible])
        summary[f"tokens_k{k}"] = {
            "per_query_top10_mean": round(statistics.mean(r["arm"][k]["tokens_total"] for r in records), 1),
            "per_query_top3_mean": round(statistics.mean(r["arm"][k]["tokens_top3"] for r in records), 1),
            "turns_per_episode_mean": round(statistics.mean(r["arm"][k]["turns_mean"] for r in records), 2),
            "control_top10_mean": round(statistics.mean(r["vault_doc_text_control"][k]["tokens_total"] for r in records), 1),
        }
    # The split the item asks for: queries the production path misses.
    miss = [r for r in records if not r["baseline"]["entity_hit"]]
    split = []
    for r in miss:
        any_k = any(r["arm"][k]["entity_hit"] for k in ks)
        any_k_clean = any(r["arm"][k]["clean"]["entity_hit"] for k in ks)
        split.append({"id": r["id"], "category": r["category"],
                      "entity_in_corpus": bool(r["corpus_presence"]["entities"]),
                      "arm_surfaced_any_k": any_k,
                      "arm_surfaced_any_k_clean": any_k_clean,
                      "by_k": {k: r["arm"][k]["entity_hit"] for k in ks},
                      "control_surfaced_any_k": any(r["vault_doc_text_control"][k]["entity_hit"] for k in ks)})
    n_miss = len(split)
    ans = sum(s["arm_surfaced_any_k"] for s in split)
    ans_clean = sum(s["arm_surfaced_any_k_clean"] for s in split)
    ans_new = sum(s["arm_surfaced_any_k_clean"] and not s["control_surfaced_any_k"] for s in split)
    present = sum(s["entity_in_corpus"] for s in split)
    summary["miss_split"] = {
        "n_miss": n_miss, "entity_in_corpus": present,
        "arm_answers": ans, "arm_answers_fraction": round(ans / n_miss, 3) if n_miss else None,
        "arm_answers_clean": ans_clean,
        "arm_answers_clean_fraction": round(ans_clean / n_miss, 3) if n_miss else None,
        "arm_answers_clean_ci95": [round(x, 3) for x in evstats.wilson_ci(ans_clean, n_miss)] if n_miss else None,
        "arm_answers_clean_not_in_control": ans_new,
        "queries": split,
    }
    for k in ks:
        summary[f"union_entity_hit_k{k}"] = round(
            sum(_union_hit(records, k)) / len(records), 3)
    summary["latency_ms"] = {
        "p50": lat[len(lat) // 2] if lat else None,
        "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
        "mean": round(statistics.mean(lat), 1) if lat else None,
        "warmup_ms": search.get("warmup_ms"), "index_ms": search.get("index_ms"),
        "baseline_recall_mean": baseline.get("summary", {}).get("overall", {}).get("latency_ms_avg"),
    }
    return {
        "item": 675,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "ks": list(args.ks), "budget_tokens": args.budget,
            "token_estimate": "ceil(chars/4)", "limit": args.limit,
            "include_kinds": list(kinds),
            "turn_rule": "user:/lloyd: open a turn; tool_call: and → [OK]/[ERROR] lines "
                         "attach to the turn they follow",
            "expansion_rule": "nearest-first (before, after), whole turns, first misfit ends it",
            "gold_rule": "episode text contains an expect_entities name or an expect_docs "
                         "path/stem (normalised substring)",
            "retrieval": "qmd structured search, lex+vec, lexMode or, rerank on, candidateLimit 40",
            "models": args.models, "qmd_env": args.qmd_env,
            "eval_markers": list(EVAL_MARKERS),
        },
        "corpus": {"source": args.corpus_source, "files": n_files,
                   "first_date": dates[0] if dates else None,
                   "last_date": dates[-1] if dates else None,
                   "update": search.get("update"), "embed": search.get("embed")},
        "baseline": {"artifact": Path(args.baseline).name,
                     "ran_at": baseline.get("ran_at"),
                     "overall": {k: v for k, v in baseline.get("summary", {}).get("overall", {}).items()
                                 if k in ("n_queries", "entity_hit_rate", "doc_hit_rate",
                                          "entity_recall_avg", "doc_recall_avg", "mrr_doc",
                                          "ndcg10", "fact_entity_recall_avg", "latency_ms_avg")},
                     "nightly_20260909_quoted": NIGHTLY_20260909},
        "summary": summary,
        "records": records,
    }


# The item's comparison target. Its artifact did not survive the 2026-09-22
# data-home wipe, so only the overall figures the item quotes exist; they were
# 20 queries, before the #1319 re-base to 87, and are not comparable per query.
NIGHTLY_20260909 = {"n_queries": 20, "entity_hit_rate": 0.50, "entity_recall_avg": 0.40,
                    "fact_entity_recall_avg": 0.375, "doc_hit_rate": 0.85,
                    "doc_recall_avg": 0.55, "mrr_doc": 0.447, "ndcg10": 0.536,
                    "latency_ms_avg": 1009}


def _f(v):
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render_markdown(res: dict) -> str:
    s, c = res["summary"], res["config"]
    ks = [str(k) for k in c["ks"]]
    L = [f"# Episodic raw-transcript arm vs the recall path (#675)", "",
         f"Run {res['ran_at']}. Baseline `{res['baseline']['artifact']}` "
         f"(ran {res['baseline']['ran_at']}), re-derived today; the item's "
         f"nightly-20260909 figures are quoted beside it (artifact lost in the 09-22 wipe).", "",
         "## Setup", "",
         f"- Corpus: {res['corpus']['files']} exported chat transcripts, "
         f"{res['corpus']['first_date']} → {res['corpus']['last_date']} ({res['corpus']['source']}).",
         f"- Turn kinds included: {', '.join(c['include_kinds'])}. {c['turn_rule']}.",
         f"- Expansion: k ∈ {{{', '.join(ks)}}}, {c['budget_tokens']} tokens per episode "
         f"({c['token_estimate']}), {c['expansion_rule']}.",
         f"- Gold: {c['gold_rule']}.",
         f"- Retrieval: {c['retrieval']}; top {c['limit']} episodes.",
         f"- Queries: {s['n_queries']} (the baseline's scored set); zero-gold "
         f"(no label in any corpus turn, excluded from arm rates): **{s['zero_gold_count']}**; "
         f"eligible: {s['n_eligible']}.", "",
         "## Headline (eligible queries unless marked)", "",
         "| arm | n | entity_hit | doc_hit | entity_recall | doc_recall | MRR | nDCG@10 |",
         "|---|---|---|---|---|---|---|---|"]

    def row(name, a):
        L.append(f"| {name} | {a['n']} | " + " | ".join(_f(a.get(m)) for m in METRICS) + " |")
    q = res["baseline"]["nightly_20260909_quoted"]
    L.append(f"| nightly-20260909 (quoted, 20 q) | 20 | {q['entity_hit_rate']} | {q['doc_hit_rate']} | "
             f"{q['entity_recall_avg']} | {q['doc_recall_avg']} | {q['mrr_doc']} | {q['ndcg10']} |")
    row("baseline today, all", s["baseline_all"])
    row("baseline today, eligible", s["baseline_eligible"])
    for k in ks:
        row(f"episodic k={k}", s[f"arm_k{k}"])
        row(f"episodic k={k}, eval-talk dropped", s[f"arm_k{k}_clean"])
        row(f"vault-doc text control k={k}", s[f"control_k{k}"])
    L += ["", "The baseline's entity metrics come from the fact layer; the episodic and "
          "control rows use text containment. Compare episodic against the control row "
          "at the same k, not against the baseline row. The baseline's MRR/nDCG rank "
          "gold *documents*; the arm's rank gold *episodes*.", "",
          "95% CIs (Wilson for hit rates, bootstrap for means):", ""]
    for k in ks:
        a, cl, ct = s[f"arm_k{k}"], s[f"arm_k{k}_clean"], s[f"control_k{k}"]
        L.append(f"- k={k}: episodic entity_hit {_f(a['entity_hit'])} {a.get('entity_hit_ci95')}, "
                 f"eval-talk dropped {_f(cl['entity_hit'])} {cl.get('entity_hit_ci95')}, "
                 f"control {_f(ct['entity_hit'])} {ct.get('entity_hit_ci95')}")
    m = s["miss_split"]
    L += ["", "## The miss subset (baseline entity_hit = 0)", "",
          f"{m['n_miss']} queries. Expected entity present in any corpus turn: "
          f"{m['entity_in_corpus']}. Arm surfaced it at any k: {m['arm_answers']} "
          f"({_f(m['arm_answers_fraction'])}); with eval-talk episodes dropped: "
          f"**{m['arm_answers_clean']} ({_f(m['arm_answers_clean_fraction'])}, 95% CI "
          f"{m['arm_answers_clean_ci95']})**; of those, not also carried by the vault "
          f"docs' own text: {m['arm_answers_clean_not_in_control']}.", "",
          "| query | category | in corpus | arm any k | arm any k (clean) | "
          + " | ".join(f"k={k}" for k in ks) + " | vault-doc text |",
          "|---|---|---|---|---|" + "---|" * len(ks) + "---|"]
    yn = lambda b: "yes" if b else "no"
    for x in m["queries"]:
        L.append(f"| {x['id']} | {x['category']} | {yn(x['entity_in_corpus'])} | "
                 f"{yn(x['arm_surfaced_any_k'])} | {yn(x['arm_surfaced_any_k_clean'])} | "
                 + " | ".join(yn(x['by_k'][k]) for k in ks)
                 + f" | {yn(x['control_surfaced_any_k'])} |")
    lat = s["latency_ms"]
    L += ["", "## Cost", "",
          f"- Episodic search latency (scratch store, same models, rerank on): p50 "
          f"{lat['p50']} ms, p95 {lat['p95']} ms, mean {lat['mean']} ms "
          f"(baseline recall mean {lat['baseline_recall_mean']} ms).", ""]
    for k in ks:
        t = s[f"tokens_k{k}"]
        L.append(f"- k={k}: {t['per_query_top10_mean']} tokens per query for 10 episodes "
                 f"({t['per_query_top3_mean']} for the top 3), {t['turns_per_episode_mean']} "
                 f"turns per episode; the vault-doc control costs {t['control_top10_mean']}.")
    L += ["", "## Per query", "",
          "| query | zero-gold | base e_hit | base d_hit | base e_rec | base d_rec | base RR | base nDCG | "
          + " | ".join(f"k={k} e_hit/d_hit/e_rec/d_rec/RR/nDCG" for k in ks) + " |",
          "|---|---|---|---|---|---|---|---|" + "---|" * len(ks)]
    for r in res["records"]:
        b = r["baseline"]
        cells = []
        for k in ks:
            a = r["arm"][k]
            cells.append("/".join(_f(a[mm]) if not isinstance(a[mm], bool) else ("1" if a[mm] else "0")
                                  for mm in METRICS))
        L.append(f"| {r['id']} | {yn(r['zero_gold'])} | "
                 + " | ".join(_f(b[mm]) if not isinstance(b[mm], bool) else ("1" if b[mm] else "0")
                              for mm in METRICS)
                 + " | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", required=True, help="a run_eval.py nightly JSON")
    ap.add_argument("--queries", default=str(HERE / "vault_recall_queries.yaml"))
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--corpus-source", default="unspecified")
    ap.add_argument("--work-dir", required=True, help="scratch: the index and search I/O")
    ap.add_argument("--search-json", help="reuse a previous search output")
    ap.add_argument("--qmd-dist", help="qmd SDK entry (default: the daemon's build)")
    ap.add_argument("--qmd-index-yml", default=str(Path.home() / ".config" / "qmd" / "index.yml"))
    ap.add_argument("--qmd-conf", default=str(LLOYD_HOME / "agent-services" / "supervisor"
                                              / "conf.d" / "agent-qmd-daemon.conf"))
    ap.add_argument("--doc-roots", nargs="+", default=None,
                    help="where the baseline's doc paths resolve (vault, then code root)")
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--budget", type=int, default=BUDGET_TOKENS)
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument("--kinds", nargs="+", default=list(ALL_KINDS), choices=ALL_KINDS)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--label", default="results")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from app.paths import VAULT_ROOT
    args.doc_roots = args.doc_roots or [str(VAULT_ROOT), str(LLOYD_HOME)]
    args.models = _qmd_models(Path(args.qmd_index_yml))
    args.qmd_env = _qmd_env_from_conf(Path(args.qmd_conf))
    args.qmd_dist = args.qmd_dist or str(_qmd_dist_from_conf(Path(args.qmd_conf)))
    t0 = time.time()
    res = run(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / f"{args.label}.md").write_text(render_markdown(res))
    print(f"wrote {out}/{args.label}.{{json,md}} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
