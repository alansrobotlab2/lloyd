#!/usr/bin/env python3
"""Complement probe: does the warp retrieve complements, or anything off-axis? (#3642 step 3)

Three arms of the same measurement on the same gold, in one artifact:

``warp-off``
    the stored seed vector, unwarped.
``warp-on``
    the seed with the Stage-1 neighbour residual injected (``complement_warp``).
``control-on``
    the seed with a random unit vector injected at the same cap.

The comparison that decides anything is warp-on against CONTROL-on, not against
warp-off. MAGE's paper concedes its residuals "are not guaranteed to contain only
complementary semantics" and the repo ships no ablation separating "the residual
helped" from "any orthogonal perturbation at that cap helped", so a warp that
beats warp-off by the same margin the random direction does has injected noise,
and this script says so out loud rather than tuning it away.

Circular-gold guard. The warp consumes KG edges; a probe whose gold is also KG
edges measures itself. Active complement-class edges are therefore split by id
into a FIT half (the neighbours the warp is fitted on) and a GOLD half (the pairs
scored), and both id sets plus their intersection are printed — the intersection
must be empty by construction, and is reported whatever it turns out to be. The
split is by id parity, so it is reproducible and nobody has to trust a hash.

Judged on the DOC axis: the complement's own indexed chunks are the targets, in a
deterministic candidate pool drawn from the scored store. The entity axis is not
used here — ``_pipeline/vault-derived/entity-vectors/vectors.npy`` is 36 days and
~4,200 entities behind the store with nothing scheduling a rebuild, which is the
open question the item lists as owed, not something this round may settle.

    python3 scripts/retrieval/complement_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import complement_harvest_probe as H  # noqa: E402
import complement_warp as W  # noqa: E402
import qmd_vectors as Q  # noqa: E402

#: How many top ranks count as retrieving the complement.
DEFAULT_K = 10
#: Candidate pool size: the chunk vectors the seed is ranked against.
DEFAULT_POOL = 1500
#: Cap on scored gold pairs, so the probe's cost is bounded and stated.
DEFAULT_GOLD_CAP = 150


def ordered_edges(store, types) -> list[tuple[int, str, str]]:
    """Active `types` edges as (id, source, target), ordered by id."""
    marks = ",".join("?" * len(types))
    return [(int(i), s, t) for i, s, t in store.conn.execute(
        f"select id, source, target from edges where expired_at is null "
        f"and type in ({marks}) order by id", tuple(types))]


def split_by_id_parity(rows: list[tuple[int, str, str]]
                       ) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """(fit, gold): every other edge id, so the halves cannot overlap.

    Parity, not a hash and not a random draw: the claim "the warp was not fitted
    on the edges that define the gold" has to be checkable by a reader who has the
    id list and nothing else.
    """
    return rows[0::2], rows[1::2]


def gold_violations(gold: list[dict], resolved: dict) -> list[dict]:
    """Gold pairs the probe must refuse to score, whatever produced them.

    A pair whose endpoints resolve to a shared note is not a complement question: the
    target's chunks ARE the seed's chunks, so every arm ranks it first and all three
    arms report 1.0 — a rate that looks like a win and measures an identity
    (#3642 clause 5). :func:`main` removes these from the gold side with
    :func:`drop_shared_note` before the cosine window is applied, so under a normal
    run this returns nothing; it returning anything means a gold set reached the
    scorer from somewhere the filter does not live, and the run then refuses to print
    a rate rather than publish one computed on a pair that cannot fail. The
    ``--show-same-note-gold`` diagnostic mode deliberately keeps the pairs so this is
    the code that names them.
    """
    return [{"edge_id": g["edge_id"], "source": g["source"], "target": g["target"],
             "shared_notes": H.shared_notes(resolved, g["source"], g["target"])}
            for g in gold if H.shared_notes(resolved, g["source"], g["target"])]


def drop_shared_note(rows, resolved) -> tuple[list, int]:
    """Partition edge rows by whether the two endpoints resolve to a shared note.

    Returns ``(kept, dropped_count)``. The shared-note case is a fact about the vault's
    entity linking, not about the edge: two entities that are really one article's two
    headings both resolve to that article's chunks, so their cosine is a consequence of
    that identity and a complement probe over them has no way to come back wrong. The
    pair is not evidence of anything, so it is excluded from the fit side and the gold
    side alike (:func:`main`) rather than scored (#3642 clause 5).
    """
    kept = [r for r in rows if not H.shared_notes(resolved, r[1], r[2])]
    return kept, len(rows) - len(kept)


def substitute_neighbours(store) -> dict[str, set[str]]:
    """entity -> entities joined to it by a substitute-class ACTIVE edge.

    The strong set's typed half. These edges are a DIFFERENT type set from the
    complement gold (``supersedes``/``competes_with``/``derived_from`` against
    ``uses``/``depends_on``/``part_of``) and no gold pair is ever one of them, so
    they may come from either half of the parity split without touching the
    disjointness this probe claims: the printed intersection is of COMPLEMENT edge
    ids, which is the only evidence the fit side consumes.
    """
    out: dict[str, set[str]] = {}
    rows = store.conn.execute(
        "select source, target from edges where expired_at is null and type in ("
        + ",".join("?" * len(H.SUBSTITUTE_TYPES)) + ")",
        tuple(H.SUBSTITUTE_TYPES)).fetchall()
    for src, tgt in rows:
        out.setdefault(src, set()).add(tgt)
        out.setdefault(tgt, set()).add(src)
    return out


def strong_set(unit_fn, seed_path: str, seed: np.ndarray,
               pool: list[tuple[str, int, int]], M: np.ndarray,
               sub_entities: set[str], entity_chunks_fn, path_of,
               *, max_rows: int = 256, min_cos: float = H.MIN_SUBSTITUTE_COS,
               ) -> tuple[np.ndarray, list[str]]:
    """The Stage-1 strong set for one seed: its SUBSTITUTES, as unit vectors.

    Two routes, unioned, both named in #3642's design ("the strong set is the
    entities joined by alias-like / ``supersedes`` / ``competes_with`` / high-cosine
    edges"):

    * **typed** — every chunk of every entity joined to the seed by a
      substitute-class edge (``sub_entities``);
    * **high-cosine** — every chunk in the candidate pool at cosine >=
      ``min_cos`` (0.70) from the seed. The pool is the set the ranking actually
      runs over, so these are precisely the seed's COMPETITORS in the measurement:
      the subspace the conceptor has to remove is the one that is already beating
      the complement.

    Chunks of the seed's OWN note are excluded from the high-cosine half, and the
    whole set is capped at ``max_rows``.

    The exclusion of the seed note is load-bearing. The first shipped version built
    the strong set from "the seed note's own remaining chunks" (``docs[note][1:257]``),
    which is not a substitute set at all: two chunks of one note measured 0.877 apart
    on the live store, so those rows are near-duplicates of the seed. And it was
    usually EMPTY, because 9,022 of the 9,411 indexed notes are single-chunk —
    ``[1:257]`` of a one-element list is ``[]`` — which left ``_conceptor_removal``
    with a stack of one and the gap-preserving constraint short-circuiting at
    ``strong_u.shape[0] < 2``, so the arm reported as the Stage-1 warp had neither
    its conceptor nor its disturbance limit running.

    Returns the (rows, paths) actually used.
    """
    paths: list[str] = []
    rows: list[np.ndarray] = []

    def note_of(where) -> str:
        return path_of.get(where, where[0])

    def add(where, vec) -> None:
        if len(rows) >= max_rows:
            return
        rows.append(vec)
        paths.append(note_of(where))

    for ent in sorted(sub_entities):
        for where in entity_chunks_fn(ent)[:4]:
            v = unit_fn(where)
            if v is not None:
                add(where, v)
    if M.shape[0]:
        sims = M @ seed
        order = np.argsort(-sims, kind="stable")
        for i in order[:max_rows * 2]:
            if len(rows) >= max_rows:
                break
            where = pool[int(i)]
            if note_of(where) == seed_path:
                continue                       # the seed's own note is not its substitute
            if float(sims[int(i)]) < min_cos:
                break                          # `order` is descending: nothing later passes
            add(where, M[int(i)])
    return (np.stack(rows) if rows else np.zeros((0, seed.size))), paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default=None,
                    help="qmd scored index; default $QMD_INDEX or ~/.cache/qmd/index.sqlite")
    ap.add_argument("--data-root", default=None,
                    help="redirect the data root instead of letting app.data_root resolve it")
    ap.add_argument("--out", default=None, help="artifact path")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
    ap.add_argument("--gold-cap", type=int, default=DEFAULT_GOLD_CAP)
    ap.add_argument("--min-complement-cos", type=float, default=H.MIN_COMPLEMENT_COS)
    ap.add_argument("--max-complement-cos", type=float, default=H.MAX_COMPLEMENT_COS)
    ap.add_argument("--min-substitute-cos", type=float, default=H.MIN_SUBSTITUTE_COS,
                    help="cosine at which a pool chunk counts as a SUBSTITUTE of the "
                         "seed and so joins the strong set (the high-cosine half)")
    ap.add_argument("--arms", default=",".join(W.ARMS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-same-note-gold", action="store_true",
                    help="diagnostic: keep the shared-note gold pairs the run would "
                         "otherwise drop, so the refusal names them one per line. It "
                         "exits 3 and prints no rate — that IS the point.")
    args = ap.parse_args(argv)

    t0 = time.time()
    root, kg_path, how = H.resolve_roots(args.data_root)
    print("== complement probe (#3642 step 3) ==")
    print(f"data root read:      {root}  (via {how})")
    print(f"kg store:            {kg_path}")
    store = H.KGStore(kg_path)
    conn = Q.connect(Q.store_path(args.index))
    dim = Q.dimension(conn)
    model = Q.embedder(conn)
    docs = Q.doc_chunks(conn)
    leaves = Q.leaf_index(docs)
    titles = Q.note_titles(conn)
    diag = Q.mean_pairwise_cosine(conn, dim)
    fingerprint = Q.fingerprint(conn)
    print(f"embedder:            {model}")
    print(f"vector dim:          {dim}")
    print(f"store fingerprint:   {fingerprint}")
    print(f"mean pairwise cosine (first {diag['sample_size']} scored vectors): "
          f"{diag['mean_pairwise_cosine']}")

    cache: dict[tuple[str, int, int], np.ndarray] = {}

    def unit(where) -> np.ndarray | None:
        v = cache.get(where)
        if v is None:
            v = Q.read_vector(conn, where, dim)
            if v is None or not np.all(np.isfinite(v)) or float(np.linalg.norm(v)) <= 0:
                cache[where] = np.zeros(dim)
            else:
                cache[where] = v / float(np.linalg.norm(v))
        out = cache[where]
        return None if float(np.linalg.norm(out)) <= 0 else out

    def entity_chunks(name: str) -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        for path in resolved.get(name, ()):
            out.extend(docs.get(path, ())[:8])
        return out

    sources = H.entity_source_docs(store)
    resolved = H.resolve_to_index(sources, leaves, titles)

    rows = ordered_edges(store, H.COMPLEMENT_TYPES)
    fit_rows, gold_rows = split_by_id_parity(rows)
    # A pair whose endpoints resolve to a shared note is dropped from BOTH sides
    # before either is used. Fitting the warp on one teaches it nothing (the two
    # vectors are means over the same chunks, so their "complement" direction is the
    # identity), and scoring on one asks which chunk is nearest to itself and always
    # finds it — a 1.00 on every arm that reads as a win. Both counts are printed,
    # never quoted here: the size of the fit-side drop and of the gold-side drop is
    # the entity-linking coverage finding this item owes, and a number carried in a
    # comment is one nobody re-derives (#3719's 25,872-versus-26,031 and #3764's
    # moved fixture are both that failure).
    fit_rows, fit_same_note = drop_shared_note(fit_rows, resolved)
    if args.show_same_note_gold:
        # Diagnostic mode: count the gold-side pairs but keep them, so the violation
        # check below can name them one per line. It cannot produce a rate — that is
        # the point, and the reason the mode exits non-zero. The count printed means
        # the same thing in both modes: how many gold pairs are this shape.
        _, gold_same_note = drop_shared_note(gold_rows, resolved)
    else:
        gold_rows, gold_same_note = drop_shared_note(gold_rows, resolved)
    fit_ids = [r[0] for r in fit_rows]
    gold_ids = [r[0] for r in gold_rows]
    overlap = sorted(set(fit_ids) & set(gold_ids))
    print(f"\nactive complement-class edges: {len(rows)} "
          f"({', '.join(H.COMPLEMENT_TYPES)})")
    print(f"pairs dropped, endpoints resolve to the same note: "
          f"fit side {fit_same_note}, gold side {gold_same_note}")
    print(f"fit-side edge ids  ({len(fit_ids)}):  {fit_ids[:20]}"
          f"{' ...' if len(fit_ids) > 20 else ''}")
    print(f"gold-side edge ids ({len(gold_ids)}):  {gold_ids[:20]}"
          f"{' ...' if len(gold_ids) > 20 else ''}")
    print(f"intersection of the two edge-id sets: {overlap if overlap else '(empty)'}")

    def cos_pair(a: str, b: str):
        va = [unit(w) for w in entity_chunks(a)[:4]]
        vb = [unit(w) for w in entity_chunks(b)[:4]]
        va = [v for v in va if v is not None]
        vb = [v for v in vb if v is not None]
        if not va or not vb:
            return None
        ma = np.mean(va, axis=0)
        mb = np.mean(vb, axis=0)
        return Q.cosine(ma, mb)

    gold: list[dict] = []
    for eid, src, tgt in gold_rows:
        c = cos_pair(src, tgt)
        if c is None or not (args.min_complement_cos <= c < args.max_complement_cos):
            continue
        gold.append({"edge_id": eid, "source": src, "target": tgt, "cosine": round(c, 6)})
        if len(gold) >= args.gold_cap:
            break
    print(f"\ngold pairs surviving the cosine window "
          f"[{args.min_complement_cos}, {args.max_complement_cos}): {len(gold)}")
    # In diagnostic mode the check runs over the gold rows BEFORE the cosine window,
    # since a shared-note pair is refused on its shape, not on where its cosine fell;
    # normally `gold` is already the stripped, windowed set and must come back empty.
    checked = gold if not args.show_same_note_gold else [
        {"edge_id": r[0], "source": r[1], "target": r[2]} for r in gold_rows]
    viol = gold_violations(checked, resolved)
    if viol:
        # Unreachable through this loop, and the reason it stays: a gold set that
        # arrives from anywhere else must not be able to reach the scorer.
        for v in viol[:10]:
            print(f"REFUSED gold pair edge {v['edge_id']}: {v['source']} / {v['target']} "
                  f"share {v['shared_notes']}", file=sys.stderr)
        print("REFUSED: the gold set contains a pair whose endpoints resolve to the "
              "same note, so no rate is reported (#3642 clause 5).", file=sys.stderr)
        return 3
    if not gold:
        print("STOP RULE: no gold pair survives, so there is nothing to score. "
              "The harvest probe's pair count is the deliverable (#3642 step 1).")

    fit_neighbours: dict[str, set[str]] = {}
    for _eid, src, tgt in fit_rows:
        fit_neighbours.setdefault(src, set()).add(tgt)
        fit_neighbours.setdefault(tgt, set()).add(src)
    sub_nb = substitute_neighbours(store)
    strong_sizes: list[int] = []
    pairs_with_no_strong = 0

    all_coords = sorted({w for lst in docs.values() for w in lst})
    stride = max(1, len(all_coords) // max(1, args.pool))
    pool = all_coords[::stride][:args.pool]
    pool_vecs = [unit(w) for w in pool]
    keep = [i for i, v in enumerate(pool_vecs) if v is not None]
    pool = [pool[i] for i in keep]
    M = np.stack([pool_vecs[i] for i in keep]) if keep else np.zeros((0, dim))
    pool_owner = {coord: i for i, coord in enumerate(pool)}
    # vec0 stores an id of the form `<file-hash>_<chunk-seq>`, so a coordinate's FIRST
    # element is not a note path — this map is the only thing that turns a coordinate
    # back into the document it came from, which both the seed-note exclusion and the
    # reported `strong_paths` need. Reading `where[0]` as a path was the defect this
    # round was opened for: `docs` is keyed by path, so `docs.get(where[0])` came back
    # empty for EVERY seed — the strong set was empty by construction on a multi-chunk
    # note just as on a single-chunk one.
    path_of = {coord: doc for doc, coords in docs.items() for coord in coords}
    print(f"candidate pool:      {len(pool)} chunk vectors "
          f"(stride {stride} over {len(all_coords)} indexed chunks)")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    scores = {a: {"hit": 0, "recip": 0.0, "n": 0} for a in arms}
    per_pair: list[dict] = []
    for g in gold:
        seed_coords = entity_chunks(g["source"])
        if not seed_coords:
            continue
        seed_where = seed_coords[0]
        seed = unit(seed_where)
        if seed is None:
            continue
        seed_path = path_of.get(seed_where, seed_where[0])
        strong, strong_paths = strong_set(
            unit, seed_path, seed, pool, M,
            sub_nb.get(g["source"], set()), entity_chunks, path_of,
            min_cos=args.min_substitute_cos)
        strong_sizes.append(int(strong.shape[0]))
        if strong.shape[0] == 0:
            pairs_with_no_strong += 1
        weak: list[np.ndarray] = []
        for nb in sorted(fit_neighbours.get(g["source"], ())):
            for w in entity_chunks(nb)[:4]:
                v = unit(w)
                if v is None:
                    continue
                c = float(v @ seed)
                if args.min_complement_cos <= c < args.max_complement_cos:
                    weak.append(v)
            if len(weak) >= 256:
                break
        target_ranks = [pool_owner[w] for w in entity_chunks(g["target"]) if w in pool_owner]
        if not target_ranks:
            continue
        row: dict = {"edge_id": g["edge_id"], "source": g["source"], "target": g["target"],
                     "seed_chunk": seed_where[0], "seed_note": seed_path,
                     "fit_neighbours": len(weak),
                     "strong_neighbours": int(strong.shape[0]),
                     "strong_paths": sorted(set(strong_paths))[:8],
                     "arms": {}}
        for arm in arms:
            v = W.apply_arm(arm, seed, strong, weak, key=f"{seed_where[0]}|{g['source']}")
            n = float(np.linalg.norm(v))
            sims = M @ (v / n) if n > 0 else M @ seed
            order = np.argsort(-sims, kind="stable")
            rank_of = np.empty(len(order), dtype=int)
            rank_of[order] = np.arange(len(order))
            rank = int(min(rank_of[t] for t in target_ranks))
            hit = rank < args.k
            scores[arm]["n"] += 1
            scores[arm]["hit"] += int(hit)
            scores[arm]["recip"] += 1.0 / (rank + 1)
            row["arms"][arm] = {"rank": rank, "hit": hit,
                               "injection": round(float(np.linalg.norm(v - seed))
                                                  / max(1e-12, n), 6)}
        per_pair.append(row)

    strong_summary = {
        "rule": (f"typed substitute edge ({'/'.join(H.SUBSTITUTE_TYPES)}), or pool chunk "
                 f"at cosine >= {args.min_substitute_cos} from the seed; the seed's own "
                 f"note excluded; capped at 256 rows"),
        "pairs_scored": len(strong_sizes),
        "min": min(strong_sizes) if strong_sizes else 0,
        "median": (int(sorted(strong_sizes)[len(strong_sizes) // 2]) if strong_sizes else 0),
        "max": max(strong_sizes) if strong_sizes else 0,
        "pairs_with_empty_strong_set": pairs_with_no_strong,
    }
    print(f"strong set (substitutes): {strong_summary['min']}/"
          f"{strong_summary['median']}/{strong_summary['max']} rows per seed "
          f"(min/median/max), empty for {pairs_with_no_strong} of "
          f"{strong_summary['pairs_scored']} seeds — {strong_summary['rule']}")

    rates = {a: (round(scores[a]["hit"] / scores[a]["n"], 6) if scores[a]["n"] else None)
             for a in arms}
    mrr = {a: (round(scores[a]["recip"] / scores[a]["n"], 6) if scores[a]["n"] else None)
           for a in arms}
    print(f"\ncomplement_hit_rate@{args.k} over {scores[arms[0]]['n'] if arms else 0} "
          f"gold pairs, doc axis:")
    for a in arms:
        print(f"  {a:8s} complement_hit_rate={rates[a]}  mrr={mrr[a]}")
    verdict = "undetermined"
    if W.ARM_WARP in arms and W.ARM_CONTROL in arms and rates.get(W.ARM_WARP) is not None:
        gap = rates[W.ARM_WARP] - rates[W.ARM_CONTROL]
        verdict = "null" if abs(gap) < 0.02 else ("warp-ahead" if gap > 0 else "control-ahead")
        print(f"\ncontrol comparison (the acceptance, not the baseline comparison): "
              f"warp-on minus control-on = {gap:+.4f} -> {verdict}")
        if verdict == "null":
            print("The transform injects an orthogonal perturbation whose benefit is "
                  "indistinguishable from a random one at the same cap. That IS the "
                  "finding; it is not to be tuned away (#3642 step 3).")

    report = {
        "kind": "complement_probe",
        "item": 3642,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "data_root": str(root),
        "data_root_via": how,
        "kg_store": str(kg_path),
        "qmd_index": str(Q.store_path(args.index)),
        "embedder": model,
        "vector_dim": dim,
        "store_fingerprint": fingerprint,
        "cosine_diagnostics": diag,
        "strong_set": strong_summary,
        "k": args.k,
        "pool_size": len(pool),
        "gold_cap": args.gold_cap,
        "arms": arms,
        "edge_split": {"fit_edge_ids": fit_ids, "gold_edge_ids": gold_ids,
                       "intersection": overlap,
                       "complement_types": list(H.COMPLEMENT_TYPES)},
        "cos_cut": {"min_complement_cos": args.min_complement_cos,
                    "max_complement_cos": args.max_complement_cos,
                    "min_substitute_cos": args.min_substitute_cos},
        "gold_pairs_scored": scores[arms[0]]["n"] if arms else 0,
        "gold_pairs_endpoints_shared_note_excluded": gold_same_note,
        "fit_pairs_endpoints_shared_note_excluded": fit_same_note,
        "complement_hit_rate": rates,
        "complement_mrr": mrr,
        "control_verdict": verdict,
        "warp_minus_control": (None if W.ARM_WARP not in arms or W.ARM_CONTROL not in arms
                               or rates.get(W.ARM_WARP) is None or rates.get(W.ARM_CONTROL) is None
                               else round(rates[W.ARM_WARP] - rates[W.ARM_CONTROL], 6)),
        "per_pair": per_pair[:400],
        "knobs": W.WARP_KNOBS.__dict__,
    }
    out = Path(args.out) if args.out else (
        H.baselines_dir_for(root) / f"complement-probe-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.json")
    if args.dry_run:
        print(f"\nartifact:            {out}  (not written: --dry-run)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nartifact:            {out}")
    print(f"elapsed:             {report['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
