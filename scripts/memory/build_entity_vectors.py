#!/usr/bin/env python3
"""Build the entity vectors semantic seeding reads (#1486).

One vector per entity row in the store, over the text

    <name> (<kind>): <top-3 facts by confidence>          (≤ 600 chars)

— the store has no `definition` filled in (0 of 11,990 rows on 2026-09-25), so
an entity's highest-confidence facts stand in for one. Embedded with
Qwen3-Embedding-0.6B (`app.qwen3_embed`, the same forward pass that embeds the
query, so the two sides cannot drift apart) and written as `names.json` +
`vectors.npy` into `app.paths.ENTITY_VECTORS_DIR` (or `--out`).

CPU, one text at a time: ~0.2 s each, so a full build of ~12k entities is
~40 minutes. Rebuild after a KG rebuild; a stale index only costs recall of
entities created since, because `semantic_candidates` intersects with the live
rankable directory set — and of entities whose FACT TEXT has moved on, which is the
half that sentence left out and what #3907 measured: a 2026-09-26 artifact served 42
days later at 71.2% coverage of the then-16,822 live entities, with 19 index names no
longer live entities at all.

Usage:
  build_entity_vectors.py [--out DIR] [--facts 3] [--limit N]
  build_entity_vectors.py --status [--out DIR]

`--status` reports what the artifact under `--out` IS — rows, age against
`app.entity_linker.MAX_INDEX_AGE_DAYS`, live coverage, provenance — and writes
nothing, loads no weights. Exit codes: 0 inside the bound, 1 STALE, 2 no artifact,
3 the live store is unreadable so coverage cannot be stated. (The build path returns
0 on writing rows; it has no other code, and an EMPTY build is not special-cased
here — `entity_texts` on an empty store writes a 0-row artifact, which the consumer
reads as an index of nothing rather than the "no semantic leg" fallback a MISSING
artifact gives. That asymmetry is owed, not fixed here.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

#: What `--status` exits with. Named because the skill's job gate reads the code, not
#: the prose: `entity-vectors-rebuild` declares STALE as the thing a scheduled run has
#: to be able to say out loud, and a job that can only print cannot fail.
EXIT_FRESH, EXIT_STALE, EXIT_ABSENT, EXIT_STORE = 0, 1, 2, 3


def status(out: Path) -> int:
    """Print the artifact's state; exit nonzero past the age bound.

    The arithmetic is `app.entity_linker.artifact_status` — the SAME function the
    loader calls to write its one log line per cold load — deliberately, because two
    implementations of `coverage` is how an artifact reads green in a log and stale in
    a terminal. The skill calls this file the contract between the job and the thing
    the job produces; a shared computation is what makes that true rather than
    aspirational.

    Missing `build.json` is reported as ABSENT provenance, never assumed: this builder
    has written the file since its only commit `dbfde750` (#1486), so an artifact
    without one came from a build that died before its last line, and which store its
    11,990 rows were derived from is unrecoverable — which is exactly the state the
    live artifact was in when #3907 was filed.
    """
    from app import entity_linker as EL

    out = Path(out)
    st = EL.artifact_status(out)
    if st is None:
        print(f"STATUS: ABSENT — no entity-vector artifact under {out.parent}")
        print("        (names.json and/or vectors.npy missing; the semantic leg really "
              "is inert here: `semantic_candidates` returns [] and recall is lexical)")
        print("rows: n/a\nbuilt: n/a\nlive entities: n/a\ncoverage: n/a")
        return EXIT_ABSENT

    print(f"STATUS: {'STALE' if st['stale'] else 'FRESH'} — {out}")
    print(f"rows: {st['rows']}" + (f" x {st['dim']}" if st["dim"] else ""))
    # The timestamp is `vectors.npy`'s FILESYSTEM mtime — the artifact's own bytes and,
    # while build.json is absent, the only age evidence that exists at all. Labelled so
    # it is never read as build.json's `built_at`, which is a claim by the process that
    # wrote the file rather than a fact about the file.
    print(f"built: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(_mtime(out)))}"
          f"  age {st['age_days']:.2f} days  (bound {st['max_age_days']} days)"
          "  [vectors.npy mtime]")
    bj = st["build_json"]
    if bj:
        print(f"provenance: build.json built_at={bj.get('built_at')} n={bj.get('n')} "
              f"model={bj.get('model')} store={bj.get('store')}")
    else:
        print("provenance: build.json ABSENT — which store this artifact was built from "
              "is unrecoverable; rebuild to record it")
    # The build-time denominator is copied from build.json or not printed at all. An
    # earlier draft here derived `rows - names_not_live` when the file was missing and
    # labelled the result "derived" — the same class of error #3907 struck from the
    # complement probe: a figure computed against a population its writer never
    # measured. Absent provenance means coverage-at-build-time is UNRECOVERABLE, and
    # saying that IS the reading.
    if isinstance(bj, dict):
        print(f"  built against: {bj.get('n', '?')} entities"
              + (f"  store={bj.get('store')}" if bj.get("store") else "")
              + "  (build.json's own record)")
    if st["coverage_pct"] is None:
        print("live entities: UNKNOWN — the store is unreadable, so coverage cannot be "
              "stated. rows and age above still stand.")
        print("coverage: n/a")
        return EXIT_STORE
    print(f"live entities: {st['live_entities']}")
    print(f"coverage: {st['coverage_pct']}% ({st['covered_entities']} of "
          f"{st['live_entities']} live entities have a vector; "
          f"{st['live_entities'] - st['covered_entities']} cannot be seeded semantically; "
          f"{st['names_not_live']} index names are no longer live entities)")
    if st["stale"]:
        print(f"        {st['age_days']:.1f} days is past the "
              f"{st['max_age_days']}-day bound `MAX_INDEX_AGE_DAYS` "
              f"(app/entity_linker.py). Rebuild, or retire the leg in config.yaml — "
              f"what is NOT allowed is leaving it enabled and unbounded.")
        return EXIT_STALE
    return EXIT_FRESH


def _mtime(out: Path) -> float:
    return Path(out).stat().st_mtime


def entity_texts(n_facts: int = 3) -> list[tuple[str, str]]:
    from app.kg_store import store
    st = store()
    facts: dict[str, list[str]] = {}
    for r in st._query(
            "SELECT entity, fact FROM facts_idx WHERE expired_at IS NULL AND fact IS NOT NULL "
            "ORDER BY entity, confidence DESC"):
        lst = facts.setdefault(r["entity"], [])
        if len(lst) < n_facts:
            lst.append(str(r["fact"]).strip())
    out = []
    for r in st._query("SELECT name, kind FROM entities"):
        name, kind = r["name"], r["kind"] or ""
        body = " ".join(facts.get(name, []))
        out.append((name, f"{name} ({kind}): {body}"[:600] if n_facts else f"{name} ({kind})"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--facts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", action="store_true",
                    help="report the artifact under --out (rows, age, live coverage, "
                         "provenance) and exit without building: 0 fresh, 1 stale, "
                         "2 no artifact, 3 store unreadable. No weights, no writes.")
    args = ap.parse_args(argv)
    import numpy as np
    from app import paths
    out = args.out or Path(paths.ENTITY_VECTORS_DIR)
    if args.status:
        return status(out)
    from app.qwen3_embed import embed_query
    out.mkdir(parents=True, exist_ok=True)
    rows = entity_texts(args.facts)
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    vecs = []
    for i, (_n, text) in enumerate(rows):
        vecs.append(embed_query(text))
        if i % 500 == 0:
            print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    mat = np.asarray(vecs, dtype=np.float32)
    tmp = out / "vectors.tmp.npy"
    np.save(tmp, mat)
    (out / "names.json").write_text(json.dumps([n for n, _ in rows], ensure_ascii=False))
    tmp.replace(out / "vectors.npy")
    (out / "build.json").write_text(json.dumps({
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n": len(rows),
        "facts_per_entity": args.facts, "store": str(paths.VAULT_KG_DB),
        "model": "Qwen/Qwen3-Embedding-0.6B", "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"wrote {len(rows)} vectors to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
