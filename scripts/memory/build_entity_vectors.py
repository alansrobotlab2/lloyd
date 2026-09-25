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
rankable directory set.

Usage:
  build_entity_vectors.py [--out DIR] [--facts 3] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))


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
    args = ap.parse_args(argv)
    import numpy as np
    from app import paths
    from app.qwen3_embed import embed_query
    out = args.out or Path(paths.ENTITY_VECTORS_DIR)
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
