#!/usr/bin/env python3
"""Harvest probe: does the live KG carry complements the vector leg can see? (#3642 step 1)

MAGE's Stage 1 needs neighbour evidence: for a seed, a *substitute* set to
project out and a *complement* set to keep. In Lloyd the only source of either is
the KG, and the per-turn retrieval legs never touch it (``app/prefetch.py``
imports ``_qmd_daemon_search`` and no ``kg_store``), so before any geometry is
changed this probe answers the coverage question the geometry depends on: how
many complement-typed edges have BOTH endpoints present in qmd's scored index,
and how many seeds have at least one.

The item's own stop rule is the deliverable here: fewer than ~50 usable pairs and
edge coverage — not vector geometry — is the blocker, so the pair count and the
edge-type histogram are printed first and written to a JSON artifact whether or
not anything downstream is ever run.

The KG is resolved through ``app.data_root.production_data_root()`` +
``kg_store_for_root()`` and never through a repo-relative path: the store lives
under the data root (``_pipeline/vault-derived/kg.sqlite``, gitignored), so a
checkout that is not the live tree — an automod worktree, which is where #577's
round could not adjudicate its own clause — must still report live counts rather
than the zeros of a tree that has no data directory. ``app.data_root`` applies
the same order itself (``$LLOYD_DATA``, a sibling ``.lloyd-data``, then the
production root under ``$HOME`` or ``/home/alansrobotlab/lloyd-data``) and
publishes which one won.

Run it as one command, with no service restart and nothing written to the store:

    python3 scripts/retrieval/complement_harvest_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from app.data_root import (  # noqa: E402
    DATA_ROOT_ENV, kg_store_for_root, production_data_root,
    resolve_data_root_for_tree)
from app.kg_store import KGStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qmd_vectors as Q  # noqa: E402

#: The complement classes, by their weights in ``agent_mcp/retrieval.py``
#: EDGE_TYPE_WEIGHTS: `uses` 1.0, `depends_on` 1.0, `part_of` 0.85. These are the
#: relations MAGE would call complements — "what goes with this" — and they are
#: the only types this probe counts as usable complement pairs.
COMPLEMENT_TYPES = ("uses", "depends_on", "part_of")

#: The substitute classes MAGE's strong set is built from: interchangeable or
#: successive rather than complementary. `alias`-spelling edges are not counted
#: because the live graph has no `alias` edge TYPE — aliases live on entity rows
#: (``entities.aliases``), which is a finding this probe prints rather than hides.
SUBSTITUTE_TYPES = ("supersedes", "competes_with", "derived_from")

#: The weak-set floor, the repo's shipped minimum weak similarity.
MIN_COMPLEMENT_COS = 0.20

#: A complement pair must sit BELOW this cosine. Set above the corpus mean
#: pairwise cosine measured on the live store at the time of writing (0.406 over
#: the first 200 scored vectors — this probe prints the current figure every run,
#: so read that rather than trusting it) and well below the cosine of two
#: neighbouring chunks of one note (0.877 measured), so it admits pairs that are
#: not substitutes without admitting the whole space.
MAX_COMPLEMENT_COS = 0.55

#: Substitute pairs are the strong-set analogue: typed AND in the upper tail.
MIN_SUBSTITUTE_COS = 0.70


def resolve_roots(cli_root: str | None) -> tuple[Path, Path, str]:
    """The data root this probe reads, and the KG store inside it.

    Precedence and fallback, in ``app.data_root``'s own order, never a path built
    from the script's location:

    1. ``--data-root`` — a deliberate override, honoured EXACTLY. A caller who
       named a root that holds no store wants the error that names it, not the
       live root answering for it, so the fallback below never fires here.
    2. ``$LLOYD_DATA`` through ``resolve_data_root_for_tree()``, which is also
       what a named tree answers with when it is NOT a worktree and carries its
       own marker.
    3. Otherwise ``production_data_root()`` — the live tree's store, which exists
       whether or not the checkout this was invoked from has a data directory of
       its own. Clause 2's whole point.

    The fallback fires on EITHER failure shape, and the second is the common one:
    ``resolve_data_root_for_tree()`` returns ``data_root_for_tree(lloyd_home)``
    for a tree that IS a worktree (``app/data_root.py``:90, whose docstring says
    "a worktree therefore gets its own root on purpose"), so an automod round's
    tree resolves to ``<worktree>/.lloyd-data``, which does not exist and never
    raises — it just has no ``_pipeline/vault-derived/kg.sqlite``, because that
    directory is gitignored. A probe that stopped at the returned root would read
    an empty graph and report 0 complement pairs, which is the #577 failure shape
    clause 2 exists to prevent: zeros that look like a thin harvest instead of the
    missing store they are. So a resolved-and-not-explicitly-named root with no
    store at it falls through to the production root, and the printed ``via`` says
    which root it abandoned and why.

    ``DataRootMissing`` from the resolver is the other shape — an unresolvable
    root — and is handled the same way, naming the exception it recovered from.
    """
    if cli_root:
        root = Path(cli_root).expanduser()
        return root, kg_store_for_root(root), "--data-root (exact, no fallback)"
    env_named = bool(os.environ.get(DATA_ROOT_ENV))
    try:
        root = resolve_data_root_for_tree()
        how = "app.data_root.resolve_data_root_for_tree()"
    except Exception as exc:                        # noqa: BLE001 — see clause text
        root = production_data_root()
        how = (f"app.data_root.production_data_root() "
               f"(resolve_data_root_for_tree raised {type(exc).__name__})")
    if not env_named and not kg_store_for_root(root).is_file():
        abandoned = root
        root = production_data_root()
        how = (f"app.data_root.production_data_root() (the root "
               f"resolve_data_root_for_tree named, {abandoned}, holds no KG store)")
    return root, kg_store_for_root(root), how


def baselines_dir_for(root: Path) -> Path:
    """``<root>/eval/baselines`` — the shape ``app.paths.EVAL_BASELINES_DIR``
    publishes (``DATA_ROOT/eval/baselines``), spelled from the resolved root rather
    than imported so the artifact sits beside the store this probe read instead of
    beside whatever root ``app.paths`` happened to bind at import time.
    """
    return root / "eval" / "baselines"


def edge_histogram(store: KGStore) -> dict[str, int]:
    """Active edge counts by type, most common first."""
    rows = store.conn.execute(
        "select type, count(*) from edges where expired_at is null group by type "
        "order by count(*) desc, type").fetchall()
    return {t: int(n) for t, n in rows}


def entity_source_docs(store: KGStore) -> dict[str, set[str]]:
    """entity -> the documents its edges and facts were extracted from.

    The only link this probe needs between an entity and a note: ``edges.source_doc``
    is a vault-relative path and no table maps an entity to a document directly.
    Both endpoint columns are walked, because an entity that is only ever an edge
    target has no row in the ``source`` sweep and would resolve to nothing — which
    reads as a thin harvest and would be blamed on the wrong thing. Facts are not
    consulted: a pair is an edge, so every endpoint of a counted pair already has
    an edge, and a facts join would only add entities no pair can contain.
    """
    out: dict[str, set[str]] = {}
    for sql in (
            "select source, source_doc from edges where expired_at is null "
            "and source_doc is not null and source_doc != ''",
            "select target, source_doc from edges where expired_at is null "
            "and source_doc is not null and source_doc != ''",
    ):
        try:
            rows = store.conn.execute(sql).fetchall()
        except Exception as exc:                    # noqa: BLE001 — schema drift
            print(f"  (skipped one source query: {exc})", file=sys.stderr)
            continue
        for name, doc in rows:
            if name and doc:
                out.setdefault(name, set()).add(doc)
    return out


def resolve_to_index(sources: dict[str, set[str]],
                     leaves: dict[str, list[str]],
                     titles: dict[str, list[str]],
                     ) -> dict[str, list[str]]:
    """entity -> indexed relative note paths: title/stem first, source doc second.

    Titles first because that join is per-ENDPOINT, which is what a pair needs:
    ``edges.source_doc`` is the note an EDGE was extracted from and therefore names
    one document shared by both endpoints, so a source-doc-only resolution hands
    every pair a shared vector component and inflates exactly the cosines this
    probe is supposed to discriminate. ``app/entity_linker.py`` already uses the
    name rule for entity vectors ("entity name == file stem or H1 title"), so this
    is the same contract, not a new one.

    Source documents remain as the fallback for entities with no matching note.
    ``edges.source_doc`` is vault-relative (``knowledge/foo.md``) while
    ``files.path`` in the code collection is ABSOLUTE (``/home/.../lloyd/app/x.py``
    — 1,120 such documents at the time of writing), so the fallback joins on the
    basename, which is the only form both spellings agree on, and an ambiguous
    leaf resolves to its whole candidate set rather than being silently narrowed.
    """
    out: dict[str, list[str]] = {}
    for name, srcs in sources.items():
        paths: set[str] = set(titles.get(Q.norm_name(name), ()))
        if not paths:
            for doc in srcs:
                paths.update(leaves.get(Path(str(doc)).name, ()))
        if paths:
            out[name] = sorted(paths)
    return out


def entity_matrix(conn, resolved: dict[str, list[str]],
                  docs: dict[str, list[tuple[str, int, int]]], dim: int,
                  max_chunks: int = 8, max_entities: int = 4000,
                  ) -> tuple[dict[str, np.ndarray], dict]:
    """entity -> mean of its note chunks' unit vectors, plus what was skipped.

    One read per chunk, cached across entities, so the cost is bounded by the
    number of distinct indexed notes rather than by the number of edges.

    ``max_entities <= 0`` means NO cap. The default is uncapped because a cap here
    silently shrinks the harvest the STOP RULE is read from: on the live stores a
    cap of 4,000 of 13,054 resolved entities reported 62 usable pairs and 66 seeds
    where the whole population has 737 and 593, and 62 < the rule's 50 by a smaller
    margin than a reviewer would trust. A truncated run must be a CHOICE.
    """
    cache: dict[tuple[str, int, int], np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    names = sorted(resolved)
    capped = max_entities is not None and max_entities > 0
    truncated = capped and len(names) > max_entities
    for name in (names[:max_entities] if capped else names):
        rows: list[np.ndarray] = []
        for path in resolved[name]:
            for where in docs.get(path, ())[:max_chunks]:
                v = cache.get(where)
                if v is None:
                    v = Q.read_vector(conn, where, dim)
                    if v is None or not np.all(np.isfinite(v)):
                        v = np.zeros(dim)
                    cache[where] = v
                nrm = float(np.linalg.norm(v))
                if nrm > 0:
                    rows.append(v / nrm)
        if rows:
            out[name] = np.mean(rows, axis=0)
    stats = {"entities_resolved": len(names), "entities_vectorised": len(out),
             "entities_skipped_over_cap": max(0, len(names) - max_entities),
             "truncated": bool(truncated), "chunk_vectors_read": len(cache),
             "max_entities": max_entities, "max_chunks_per_note": max_chunks}
    return out, stats


def cos(vectors: dict[str, np.ndarray], a: str, b: str):
    va, vb = vectors.get(a), vectors.get(b)
    if va is None or vb is None:
        return None
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na <= 0 or nb <= 0:
        return None
    return float(va @ vb / (na * nb))


def shared_notes(resolved: dict, a: str, b: str) -> list[str]:
    """Note paths BOTH entities resolve to; empty means the pair is admissible.

    #3642 measured 894 complement pairs whose two endpoints resolve to the same
    note: one document wearing two entity names. Such a pair is not evidence of a
    relation at all — the two vectors are means over the same chunks, so their
    cosine is a consequence of that identity and sits at or near 1.0 no matter what
    the edge claims, and a probe scored on one is asking which chunk is nearest to
    itself and always finds it. So the pair is dropped where it is COUNTED, not
    after a rate is computed: an identity cosine left in the sweep also pollutes
    the distribution the cosine band is cut on, which is how a same-note pair could
    move the band as well as the count.
    """
    return sorted(set(resolved.get(a, ())) & set(resolved.get(b, ())))


def pair_sweep(store: KGStore, vectors: dict[str, np.ndarray], types,
               lo: float, hi: float, resolved: dict | None = None,
               upper: bool = False):
    """Active `types` edges with both endpoints vectorised and cosine in range.

    Returns ``(pairs, seen, same_note)`` where a pair is ``(edge_id, source,
    target, cosine)``, ``seen`` counts the edges that had both endpoints vectorised
    regardless of where their cosine fell — the denominator the coverage number
    needs — and ``same_note`` counts the pairs dropped by :func:`shared_notes`
    before the cosine band was even consulted, which is why it is a number worth
    publishing: it is the size of the defect, not of the geometry.
    """
    pairs, seen, same_note = [], 0, 0
    for etype in types:
        rows = store.conn.execute(
            "select id, source, target from edges where expired_at is null "
            "and type = ? order by id", (etype,)).fetchall()
        for eid, src, tgt in rows:
            c = cos(vectors, src, tgt)
            if c is None:
                continue
            seen += 1
            if resolved is not None and shared_notes(resolved, src, tgt):
                same_note += 1
                continue
            # upper=True is the upper-tail mode: keep cos >= lo, with no ceiling
            # below 1.0. Comparing against hi=1.01 there selects zero pairs.
            if (c >= lo if upper else lo <= c < hi):
                pairs.append({"edge_id": int(eid), "source": src, "target": tgt,
                              "type": etype, "cosine": round(c, 6)})
    return pairs, seen, same_note


def degree_histogram(pairs) -> dict[str, int]:
    deg: Counter = Counter()
    for p in pairs:
        deg[p["source"]] += 1
        deg[p["target"]] += 1
    out: Counter = Counter()
    for n in deg.values():
        out[min(n, 10)] += 1
    return {("10+" if k == 10 else str(k)): int(v) for k, v in sorted(out.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default=None,
                    help="qmd scored index; default $QMD_INDEX or ~/.cache/qmd/index.sqlite")
    ap.add_argument("--data-root", default=None,
                    help="redirect the data root instead of letting app.data_root resolve it "
                         "(for a sandbox run only; a production run leaves this unset)")
    ap.add_argument("--out", default=None,
                    help="artifact path; default <resolved root>/eval/baselines/"
                         "complement-harvest-<utc>.json")
    ap.add_argument("--max-complement-cos", type=float, default=MAX_COMPLEMENT_COS)
    ap.add_argument("--min-complement-cos", type=float, default=MIN_COMPLEMENT_COS)
    # Uncapped by default, measured: the whole resolved population costs 18 s on the
    # live stores (13,054 entities, 9,411 distinct notes, one cached vector read per
    # chunk), while a cap of 4,000 silently cut the harvest from 737 usable pairs and
    # 593 seeds to 62 and 66 — which would have printed the STOP RULE and reported
    # edge coverage as the blocker when the blocker was the probe's own default.
    # `0` means no cap; pass a positive number for a deliberately bounded run.
    ap.add_argument("--max-entities", type=int, default=0,
                    help="cap the entities vectorised for the sweep (0 = all of them)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print everything and resolve every path, write nothing")
    args = ap.parse_args(argv)

    t0 = time.time()
    root, kg_path, how = resolve_roots(args.data_root)
    print("== complement harvest probe (#3642 step 1) ==")
    print(f"data root read:      {root}  (via {how})")
    print(f"kg store:            {kg_path}")
    if not Path(kg_path).is_file():
        # Reachable only for an explicitly named root: with no `--data-root` and no
        # `$LLOYD_DATA`, `resolve_roots` has already fallen through to
        # `production_data_root()`, which is the difference between this message and
        # the zeros #577's round mistook for an empty graph.
        print(f"BLOCKER: no KG store at {kg_path}\n"
              f"         the root was named ({how}); this probe does not invent one, "
              f"and it does not silently read a different root than the one you named",
              file=sys.stderr)
        return 2
    store = KGStore(kg_path)

    index = Q.store_path(args.index)
    print(f"qmd scored index:    {index}")
    conn = Q.connect(index)
    dim = Q.dimension(conn)
    model = Q.embedder(conn)
    docs = Q.doc_chunks(conn)
    leaves = Q.leaf_index(docs)
    titles = Q.note_titles(conn)
    diag = Q.mean_pairwise_cosine(conn, dim)
    print(f"embedder:            {model}")
    print(f"vector dim:          {dim}")
    print(f"indexed notes:       {len(docs)}  (chunks: {sum(len(v) for v in docs.values())})")
    print(f"mean pairwise cosine (first {diag['sample_size']} scored vectors): "
          f"{diag['mean_pairwise_cosine']}")

    hist = edge_histogram(store)
    active = sum(hist.values())
    print(f"\nactive edge type histogram ({active} active edges):")
    for t, n in hist.items():
        mark = " <- complement" if t in COMPLEMENT_TYPES else (
            " <- substitute" if t in SUBSTITUTE_TYPES else "")
        print(f"  {t:16s} {n:7d}{mark}")

    sources = entity_source_docs(store)
    resolved = resolve_to_index(sources, leaves, titles)
    vectors, vstats = entity_matrix(conn, resolved, docs, dim,
                                   max_entities=args.max_entities)
    print(f"\nentities with a source document: {len(sources)}")
    print(f"  resolving to an indexed note:   {vstats['entities_resolved']}")
    cap_note = f" (capped at {vstats['max_entities']})" if vstats["truncated"] else ""
    print(f"  vectorised for the sweep:       {vstats['entities_vectorised']}{cap_note}")

    # `resolved` goes into the sweep so a pair whose endpoints are the SAME note is
    # excluded from the usable count rather than reported as one (#3642 clause 1:
    # usable is defined over disjoint note-path endpoints, and live that is 894 of
    # the pairs the raw cosine band would otherwise hand over).
    comp, comp_seen, comp_same_note = pair_sweep(
        store, vectors, COMPLEMENT_TYPES,
        args.min_complement_cos, args.max_complement_cos, resolved=resolved)
    subs, subs_seen, subs_same_note = pair_sweep(
        store, vectors, SUBSTITUTE_TYPES, MIN_SUBSTITUTE_COS, 1.01,
        resolved=resolved, upper=True)
    seeds = sorted({p["source"] for p in comp} | {p["target"] for p in comp})
    print(f"\ncomplement-class active edges with both endpoints vectorised: {comp_seen}")
    print(f"  excluded, endpoints resolve to the same note:               "
          f"{comp_same_note}  (substitute class: {subs_same_note})")
    print(f"USABLE complement pairs (cosine in "
          f"[{args.min_complement_cos}, {args.max_complement_cos})): {len(comp)}")
    print(f"SEEDS with >= 1 usable complement pair:                    {len(seeds)}")
    print(f"substitute-class pairs (typed, cosine >= {MIN_SUBSTITUTE_COS}):  {len(subs)}")
    print(f"usable-pair degree histogram (pairs per entity): {degree_histogram(comp)}")
    if len(comp) < 50:
        print("\nSTOP RULE: fewer than 50 usable complement pairs, so edge coverage — "
              "not vector geometry — is the blocker. Record this number and close "
              "the geometry line (#3642 step 1).")

    report = {
        "kind": "complement_harvest_probe",
        "item": 3642,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "data_root": str(root),
        "data_root_via": how,
        "kg_store": str(kg_path),
        "qmd_index": str(index),
        "embedder": model,
        "vector_dim": dim,
        "store_fingerprint": Q.fingerprint(conn),
        "cosine_diagnostics": diag,
        "active_edge_total": active,
        "active_edge_histogram": hist,
        "complement_types": list(COMPLEMENT_TYPES),
        "substitute_types": list(SUBSTITUTE_TYPES),
        "cos_cut": {"min_complement_cos": args.min_complement_cos,
                    "max_complement_cos": args.max_complement_cos,
                    "min_substitute_cos": MIN_SUBSTITUTE_COS},
        "entities_with_source_doc": len(sources),
        "entity_resolution": vstats,
        "complement_edges_both_endpoints_vectorised": comp_seen,
        "substitute_edges_both_endpoints_vectorised": subs_seen,
        "complement_pairs_endpoints_shared_note_excluded": comp_same_note,
        "substitute_pairs_endpoints_shared_note_excluded": subs_same_note,
        "usable_complement_pairs": len(comp),
        "seeds_with_usable_complement_pair": len(seeds),
        "usable_pair_degree_histogram": degree_histogram(comp),
        "usable_complement_pairs_below_stop_rule": len(comp) < 50,
        "substitute_pairs": len(subs),
        "pairs_sample": comp[:50],
        "substitutes_sample": subs[:50],
    }
    out = Path(args.out) if args.out else (
        baselines_dir_for(root) / f"complement-harvest-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.json")
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
