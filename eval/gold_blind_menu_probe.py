#!/usr/bin/env python3
"""Probe three gold-blind candidate-menu builders for the label-agreement ceiling (#1937).

`eval/label_agreement_ceiling.py` builds each query's candidate menu by ranking the
namespace on query-token overlap and keeping the top 40. Replayed over the live pools
on 2026-10-01 that order offered 0.468 of entity gold at cap 40, 0.649 at cap 320 and
1.000 only uncapped: the golds it hides share no token with their query, so no bounded
cap reaches them and the lever is the builder, not the cap.

This script measures candidates for a replacement and changes nothing. Three builders,
each a pure function of (query, pool, its own gold-free side input):

  alias     — token overlap over each name AND the alias surfaces that route to it
              (`kg_store` aliases), so a query that says "the Pi" can reach
              "Raspberry Pi 5";
  embed     — cosine between the query and each candidate name under the local CPU
              embedder (`app.qwen3_embed.embed_query`);
  expansion — token overlap after the query is widened with its own seeds / linked
              note titles (what retrieval SEEDING derives from the query — never what
              retrieval returned).

Every builder keeps the three properties the shipped one documents: a deterministic
total order, a NARROWING of the pool it was handed (it can return nothing the pool
does not hold), and gold-blindness — no builder takes a gold argument, and the gold is
read only afterwards, to count where it fell. No labeler is called and no HTTP request
is made; the embedder is the one model involved and it runs in-process on CPU.

The namespace and the corpus are ARGUMENTS. `probe()` takes them as values; the CLI
takes them as files, and only `--from-live` opens the store (read-only) to fetch them.

    python eval/gold_blind_menu_probe.py --from-live --caps 40,60           # alias + expansion
    python eval/gold_blind_menu_probe.py --from-live --caps 40,60 --embed   # + embedder, minutes of CPU

The decision rule is #1937's: deploy a builder only if it reaches entity offered >= 0.80
at cap <= 60; otherwise record the three fractions and close.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import eval.label_agreement_ceiling as lac  # noqa: E402
import eval.run_eval as ev  # noqa: E402

BASELINE = "token-overlap"          # the shipped builder, as the control row
BUILDERS = ("alias", "embed", "expansion")
DEFAULT_CAPS = (40, 60)
DEPLOY_BAR = 0.80
DEPLOY_MAX_CAP = 60


# ── the builders: (query, pool, side input) -> the pool in a total order ─────

def _overlap_order(tokens: list[str], pool: list[str],
                   haystack: Callable[[str], str]) -> list[str]:
    """`lac._ranked`'s order with the haystack swapped: (-overlap, length, name)."""
    scored = []
    for name in pool:
        hay = haystack(name)
        scored.append((-sum(1 for t in tokens if t in hay), len(name), name))
    scored.sort()
    return [n for _s, _l, n in scored]


def rank_token_overlap(query: str, pool: list[str]) -> list[str]:
    """The shipped builder, unchanged — the control every other row is read against."""
    return lac._ranked(query, pool)


def rank_alias(query: str, pool: list[str], aliases: dict[str, str]) -> list[str]:
    """Token overlap over each name plus the alias surfaces that route to it.

    `aliases` is surface -> canonical. Only surfaces whose canonical is a pool member
    contribute, and what is returned is the pool's own names: an alias widens what a
    name can be FOUND by, it never becomes a candidate.
    """
    members = set(pool)
    surfaces: dict[str, list[str]] = {}
    for surface, canonical in sorted(aliases.items()):
        if canonical in members and surface != canonical:
            surfaces.setdefault(canonical, []).append(surface)
    return _overlap_order(
        lac.query_tokens(query), pool,
        lambda name: " ".join(ev._norm(s) for s in [name, *surfaces.get(name, [])]))


def rank_expansion(query: str, pool: list[str], expansion: Iterable[str]) -> list[str]:
    """Token overlap after widening the query with its own seeds / linked note titles."""
    tokens = lac.query_tokens(" ".join([str(query), *map(str, expansion or [])]))
    return _overlap_order(tokens, pool, ev._norm)


def rank_embed(query_vec, pool: list[str], name_vecs: dict) -> list[str]:
    """Cosine query -> candidate name, ties broken by (length, name).

    The vectors are the side input, so the order is a pure function of them: the
    embedder is called once per distinct string by `embed_all`, never in here.
    Scores are rounded before sorting so float noise cannot reorder a tie between
    runs. A pool member with no vector sorts last rather than disappearing.
    """
    import numpy as np

    have = [n for n in pool if n in name_vecs]
    scores: dict[str, float] = {}
    if have:
        mat = np.asarray([name_vecs[n] for n in have], dtype=np.float64)
        q = np.asarray(query_vec, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0)
        cos = (mat @ q) / np.where(norms == 0.0, 1.0, norms)
        scores = {n: round(float(c), 9) for n, c in zip(have, cos)}
    scored = [(-scores[n] if n in scores else 2.0, len(n), n) for n in pool]
    scored.sort()
    return [n for _s, _l, n in scored]


def menu(ordered: list[str], cap: int) -> list[str]:
    """The narrowing itself: the first `cap` of an order. No shuffle — the shipped
    builder shuffles for the labeler's benefit, and nothing here is shown to one."""
    return ordered[:max(0, int(cap))]


def embed_all(texts: Iterable[str], embed_fn: Callable[[str], object],
              cache: dict | None = None) -> dict:
    """One embedding per distinct string, in sorted order so a re-run asks identically.

    `cache` (text -> vector) is consulted first and filled as it goes, so an
    interrupted run over the live pools does not pay for the same 19,000 strings
    twice; the CLI persists it with `--embed-cache`.
    """
    import numpy as np

    cache = cache if cache is not None else {}
    out: dict = {}
    for text in sorted(set(texts)):
        if text not in cache:
            vec = embed_fn(text)
            if hasattr(vec, "detach"):
                vec = vec.detach().cpu().numpy()
            cache[text] = np.asarray(vec, dtype=np.float32).reshape(-1)
        out[text] = cache[text]
    return out


def load_embed_cache(path: str) -> dict:
    import numpy as np

    p = Path(path)
    if not p.exists():
        return {}
    with np.load(p, allow_pickle=False) as z:
        return dict(zip(json.loads(str(z["texts"])), z["vecs"]))


def save_embed_cache(path: str, cache: dict) -> None:
    import numpy as np

    texts = sorted(cache)
    if not texts:
        return
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, texts=np.asarray(json.dumps(texts)),
             vecs=np.stack([np.asarray(cache[t], dtype=np.float32) for t in texts]))
    tmp.replace(path)


# ── the measurement ──────────────────────────────────────────────────────────

_LEGS = (("entity", "expect_entities", lac.entity_label_satisfied),
         ("doc", "expect_docs", lac.doc_label_satisfied))


def probe(*, entity_names: list[str], vault_paths: list[str], corpus: list[dict],
          caps: Iterable[int] = DEFAULT_CAPS, aliases: dict[str, str] | None = None,
          expansions: dict[str, list[str]] | None = None,
          embed_fn: Callable[[str], object] | None = None,
          builders: Iterable[str] = BUILDERS, embed_cache: dict | None = None) -> dict:
    """Offered fraction per builder, per leg, per cap. Reads gold only to count.

    A builder whose side input was not supplied is reported as `skipped` with the
    reason, never as a row of zeros: `embed` with no `embed_fn`, `alias` with no alias
    map, `expansion` with no expansions. The doc leg has no aliases, so `alias` on it
    is the control order and says so.
    """
    caps = [int(c) for c in caps]
    wanted = [b for b in builders if b in BUILDERS]
    pools = {"entity": list(entity_names), "doc": list(vault_paths)}
    skipped: dict[str, str] = {}
    if "alias" in wanted and not aliases:
        skipped["alias"] = "no alias map supplied"
    if "expansion" in wanted and not expansions:
        skipped["expansion"] = "no query expansions supplied"
    if "embed" in wanted and embed_fn is None:
        skipped["embed"] = "no embedder supplied (pass --embed)"
    active = [BASELINE] + [b for b in wanted if b not in skipped]

    vecs: dict[str, list[float]] = {}
    if "embed" in active:
        texts = set(pools["entity"]) | set(pools["doc"]) | {str(q.get("query") or "")
                                                             for q in corpus}
        vecs = embed_all(texts, embed_fn, embed_cache)

    hits = {b: {leg: {c: 0 for c in caps} for leg, _k, _s in _LEGS} for b in active}
    totals = {leg: 0 for leg, _k, _s in _LEGS}
    for q in corpus:
        qid, text = str(q.get("id") or ""), str(q.get("query") or "")
        for leg, key, satisfied in _LEGS:
            golds = list(q.get(key) or [])
            if not golds:
                continue
            totals[leg] += len(golds)
            pool = pools[leg]
            for b in active:
                if b == BASELINE:
                    ordered = rank_token_overlap(text, pool)
                elif b == "alias":
                    ordered = rank_alias(text, pool, aliases if leg == "entity" else {})
                elif b == "expansion":
                    ordered = rank_expansion(text, pool, (expansions or {}).get(qid) or [])
                else:
                    ordered = rank_embed(vecs[text], pool, vecs)
                for cap in caps:
                    shown = menu(ordered, cap)
                    hits[b][leg][cap] += sum(
                        1 for g in golds if any(satisfied(g, [c]) for c in shown))
    rows = {b: {leg: {str(c): {"offered": hits[b][leg][c], "labels": totals[leg],
                               "fraction": (round(hits[b][leg][c] / totals[leg], 4)
                                            if totals[leg] else None)}
                      for c in caps} for leg, _k, _s in _LEGS} for b in active}
    return {"caps": caps, "labels": totals, "builders": rows, "skipped": skipped,
            "namespace": {"entity": len(pools["entity"]), "doc": len(pools["doc"])},
            "queries": len(corpus), "gold_aware": False, "labeler_calls": 0}


def verdict(result: dict) -> str:
    """#1937's rule over the printed numbers: entity offered >= 0.80 at a cap <= 60."""
    best = None
    for name, legs in result["builders"].items():
        if name == BASELINE:
            continue
        for cap, cell in legs["entity"].items():
            if int(cap) <= DEPLOY_MAX_CAP and cell["fraction"] is not None:
                if best is None or cell["fraction"] > best[2]:
                    best = (name, cap, cell["fraction"])
    if best is None:
        return ("no builder measured at a cap <= "
                f"{DEPLOY_MAX_CAP}: nothing to rule on")
    name, cap, frac = best
    if frac >= DEPLOY_BAR:
        return (f"{name} reaches entity offered {frac} at cap {cap} (bar {DEPLOY_BAR} at "
                f"cap <= {DEPLOY_MAX_CAP}): a deploy candidate — deploying re-bases the "
                "ceiling, so it and its re-run belong in one commit")
    return (f"no builder reaches the bar: best is {name} at entity offered {frac}, cap "
            f"{cap} (bar {DEPLOY_BAR} at cap <= {DEPLOY_MAX_CAP}) — tried, did not work "
            "out; cap 40 and the all-gold veto stay as they are")


def print_report(result: dict) -> None:
    ns = result["namespace"]
    print(f"gold-blind menu probe: {result['queries']} queries; namespace entity="
          f"{ns['entity']} doc={ns['doc']}; labeler calls: {result['labeler_calls']}")
    for leg in ("entity", "doc"):
        print(f"\n{leg} gold offered ({result['labels'][leg]} labels)")
        print("  " + f"{'builder':<16}" + "".join(f"{'cap ' + str(c):>18}"
                                                  for c in result["caps"]))
        for name, legs in result["builders"].items():
            cells = "".join(
                f"{str(cell['fraction']) + ' (' + str(cell['offered']) + '/' + str(cell['labels']) + ')':>18}"
                for cell in legs[leg].values())
            print(f"  {name:<16}{cells}")
    for name, why in result["skipped"].items():
        print(f"\nskipped {name}: {why}")
    print(f"\nverdict: {verdict(result)}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expansions_from_baseline(path: str) -> dict[str, list[str]]:
    """query id -> the seeds a nightly baseline recorded for that query.

    `records[].seeds_extracted` is what seeding derived from the query text before any
    retrieval ran, so it is a function of the query — not of what retrieval returned.
    """
    out: dict[str, list[str]] = {}
    for rec in (_load_json(path).get("records") or []):
        seeds = rec.get("seeds_extracted")
        if rec.get("id") is not None and isinstance(seeds, list):
            out[str(rec["id"])] = [str(s.get("name") if isinstance(s, dict) else s)
                                   for s in seeds if s]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(lac.CORPUS_PATH))
    ap.add_argument("--entity-names", help="JSON list of entity names")
    ap.add_argument("--doc-paths", help="JSON list of vault .md paths")
    ap.add_argument("--aliases", help="JSON object: alias surface -> canonical name")
    ap.add_argument("--expansions", help="JSON object: query id -> list of seed strings")
    ap.add_argument("--expansions-from-baseline",
                    help="a nightly-*.json; uses records[].seeds_extracted")
    ap.add_argument("--from-live", action="store_true",
                    help="read the namespace and aliases from the live store (read-only) "
                         "for whichever of the three files was not given")
    ap.add_argument("--caps", default=",".join(map(str, DEFAULT_CAPS)))
    ap.add_argument("--embed", action="store_true",
                    help="run the embedding builder (CPU, minutes over the live pools)")
    ap.add_argument("--embed-cache",
                    help="an .npz the embeddings are read from and saved to")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    names = _load_json(args.entity_names) if args.entity_names else None
    paths = _load_json(args.doc_paths) if args.doc_paths else None
    aliases = _load_json(args.aliases) if args.aliases else None
    if args.from_live:
        names = names if names is not None else lac.entity_name_table()
        paths = paths if paths is not None else lac.vault_markdown_paths()
        if aliases is None:
            from app import kg_store
            aliases = kg_store.store().aliases.all()
    if names is None or paths is None:
        ap.error("give --entity-names and --doc-paths, or --from-live")
    expansions = (_load_json(args.expansions) if args.expansions
                  else expansions_from_baseline(args.expansions_from_baseline)
                  if args.expansions_from_baseline else None)
    embed_fn = None
    if args.embed:
        from app.qwen3_embed import embed_query
        embed_fn = embed_query
    cache = load_embed_cache(args.embed_cache) if args.embed_cache else None
    try:
        result = probe(entity_names=names, vault_paths=paths,
                       corpus=lac.load_corpus(args.corpus),
                       caps=[int(c) for c in args.caps.split(",") if c.strip()],
                       aliases=aliases, expansions=expansions, embed_fn=embed_fn,
                       embed_cache=cache)
    finally:
        if args.embed_cache and cache:
            save_embed_cache(args.embed_cache, cache)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
