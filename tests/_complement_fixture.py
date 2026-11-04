"""Fixtures for the #3642 probes: a tiny scored index and a tiny KG.

Both replicas hold exactly the columns ``scripts/retrieval/qmd_vectors.py`` and
the two probes read, under the real table names, so a probe exercising them
exercises the same SQL it will run against the live stores. They are deliberately
NOT full copies of qmd's or the KG's schema: a fixture that duplicates a schema
drifts from it, and the point here is the join and the decode, not the catalogue.

The scored index replica keeps qmd's vec0 storage shape that matters —
``vectors_vec_rowids(id, chunk_id, chunk_offset)`` pointing into a BLOB in
``vectors_vec_vector_chunks00``, with ``id = <file-hash>_<chunk-seq>`` — and
stores one vector per chunk row, which the same ``substr(vectors, off*dim*4+1,
dim*4)`` decode reads either way. The ``float[dim]`` declaration lives in a
comment inside the DDL because that is how :func:`qmd_vectors.dimension` finds
the dimension on the real store too: out of ``sqlite_master``, with no loadable
extension required.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path

REPLICA_NOTE = "fixture replica of the columns scripts/retrieval/qmd_vectors.py reads"


def write_qmd_index(path: str | Path, dim: int, notes: dict[str, list[list[float]]],
                    *, model: str = "fixture:Test-Embedding-0.6B.gguf") -> Path:
    """Create a read-only-readable scored index over ``notes``.

    ``notes`` maps a collection-relative path (the shape qmd stores for vault
    collections) to that note's chunk vectors.
    """
    p = Path(path)
    conn = sqlite3.connect(p)
    conn.executescript(f"""
        create table files (id integer primary key, path text);
        create table documents (id integer primary key, collection text, path text,
                                title text, hash text, file_id integer, active integer);
        create table content_vectors (hash text, seq integer, model text, command text,
                                      total_chunks integer, best_chunk integer,
                                      embedded_at integer);
        /* the vec0 declaration is what `dimension` parses; a real store gets it
           from sqlite-vec, a fixture only needs the float[dim] text on record */
        create table vectors_vec (/* vec0 embedding float[{dim}] partition 1 */
                                  id text, embedding text);
        create table vectors_vec_rowids (id text, rowid integer, chunk_id integer,
                                         chunk_offset integer);
        create table vectors_vec_vector_chunks00 (rowid integer primary key,
                                                  vectors blob);
    """)
    rowid = 0
    for path_rel, chunks in sorted(notes.items()):
        digest = hashlib.sha256(path_rel.encode("utf-8")).hexdigest()
        conn.execute("insert into files (path) values (?)", (f"/fixture/{path_rel}",))
        conn.execute(
            "insert into documents (collection, path, title, hash, file_id, active) "
            "values (?,?,?,?,?,1)",
            (path_rel.split("/", 1)[0], path_rel, Path(path_rel).stem, digest,
             conn.execute("select last_insert_rowid()").fetchone()[0]))
        for seq, vec in enumerate(chunks):
            assert len(vec) == dim, f"fixture vector {seq} of {path_rel} is not {dim}-dim"
            blob = struct.pack(f"{dim}f", *[float(x) for x in vec])
            rowid += 1
            conn.execute("insert into vectors_vec_rowids (id, rowid, chunk_id, chunk_offset)"
                         " values (?,?,?,0)", (f"{digest}_{seq}", rowid, rowid))
            conn.execute("insert into vectors_vec_vector_chunks00 (rowid, vectors) "
                         "values (?,?)", (rowid, blob))
            conn.execute("insert into content_vectors (hash, seq, model, command, "
                         "total_chunks, best_chunk, embedded_at) values (?,?,?,?,?,?,?)",
                         (digest, seq, model, "fixture", len(chunks), 0, 0))
    conn.commit()
    conn.close()
    return p


def write_kg(path: str | Path, edges: list[tuple[str, str, str, str]],
             *, db_path: str | Path | None = None) -> Path:
    """Create a KG store with ``edges`` = [(source, target, type, source_doc)].

    Ids are assigned 1-based in the given order, so a test that splits edges by
    id can state the split it expects. Built through ``app.kg_store.KGStore`` so
    the schema is the production one rather than a copy that could drift.
    """
    from app.kg_store import KGStore

    p = Path(db_path or path)
    store = KGStore(p)
    for i, (src, tgt, etype, doc) in enumerate(edges, start=1):
        store.conn.execute(
            "insert into edges (id, source, target, type, created_at, source_doc) "
            "values (?,?,?,?,?,?)",
            (i, src, tgt, etype, "2026-11-04T00:00:00Z", doc))
    store.conn.commit()
    store.conn.close()
    return p
