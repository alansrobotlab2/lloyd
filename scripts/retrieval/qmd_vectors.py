"""Read-only access to qmd's scored vector store (#3642).

The scored space Lloyd retrieves against lives in qmd's sqlite index
(``~/.cache/qmd/index.sqlite``, or ``$QMD_INDEX``), which the daemon writes and
nobody else is allowed to write. Everything here opens it ``mode=ro`` and reads
four of its tables:

``documents`` / ``files``
    one row per indexed chunk-source. ``files.path`` is the on-disk path — an
    ABSOLUTE path for the code collection (``lloyd``) and a collection-relative
    one for every vault collection, so matching a vault note is
    ``f.path = <rel> or f.path like '%/<leaf>'``.
``content_vectors``
    ``hash`` (sha256 of the file), ``seq`` (0-based chunk index), ``model``. The
    vec0 row id qmd uses is ``hash_seq``, which is what this module keys on.
``vectors_vec_rowids`` / ``vectors_vec_vector_chunks00``
    the vec0 shadow tables: the rowid table gives ``(chunk_id, chunk_offset)``,
    the chunk table packs ``dim * 4`` bytes per vector into one BLOB. Reading a
    vector is therefore one ``substr(vectors, offset*dim*4+1, dim*4)`` — 4 KiB,
    not the ~4 MiB whole-chunk BLOB. Going through the shadow tables rather than
    the ``vec0`` virtual table means no loadable-extension is needed, which is
    why the probe runs under a bare ``python3``.

The dimension is parsed from the ``vec0`` declaration in ``sqlite_master``
(``... float[1024] ...``) rather than assumed, and the embedder name comes from
``content_vectors.model`` rather than from the yaml, so an artifact states what
the store actually holds. See ``complement_warp`` for why an artifact that names
no embedder is the one that orphaned #577's whole measurement.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import numpy as np

#: Default location of the daemon's scored index. Overridable so a test can
#: point at a fixture instead of at the live store.
DEFAULT_STORE = Path.home() / ".cache" / "qmd" / "index.sqlite"


def store_path(cli: str | None = None) -> Path:
    """The store to read: ``--index``, else ``$QMD_INDEX``, else the default."""
    if cli:
        return Path(cli).expanduser()
    env = os.environ.get("QMD_INDEX")
    return Path(env).expanduser() if env else DEFAULT_STORE


def connect(path: Path) -> sqlite3.Connection:
    """Open the scored index read-only. Never creates: an absent store is a
    message, not an empty database (#1516-shaped: a created-empty store reads
    as 'zero vectors' rather than 'wrong path')."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"qmd scored index not found at {p}; point --index or $QMD_INDEX at it")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def dimension(conn: sqlite3.Connection) -> int:
    """The vector dimension, from the vec0 declaration in the schema."""
    for (sql,) in conn.execute(
            "select sql from sqlite_master where type='table' and sql like '%vec0%'"):
        m = re.search(r"float\[(\d+)\]", sql or "")
        if m:
            return int(m.group(1))
    raise ValueError("no vec0 virtual table with a float[N] declaration in this store")


def embedder(conn: sqlite3.Connection) -> str:
    """The model that produced the stored vectors, as the store records it.

    Refuses to guess when the store holds more than one: a mixed space is not one
    space, and a per-entity transform over it would be averaging two geometries.
    """
    rows = [r[0] for r in conn.execute(
        "select distinct model from content_vectors order by 1")]
    if not rows:
        raise ValueError("content_vectors is empty — nothing to read")
    if len(rows) > 1:
        raise ValueError(f"store holds {len(rows)} embedders: {rows}")
    return rows[0]


def fingerprint(conn: sqlite3.Connection) -> str:
    """A cheap content hash of the scored store: row counts, id extremes, size.

    Named in the warp manifest so an artifact can be re-checked against the store
    it was measured on. Deliberately not a byte hash of the file: the daemon
    rewrites it constantly, and a manifest that invalidates on every keystroke
    invalidates nothing useful.
    """
    chunks = conn.execute("select count(*) from content_vectors").fetchone()[0]
    vecs = conn.execute("select count(*) from vectors_vec_rowids").fetchone()[0]
    first = conn.execute("select min(id) from vectors_vec_rowids").fetchone()[0]
    try:
        size = conn.execute("select size from pragma_database_size").fetchone()[0]
    except sqlite3.Error:              # pragma unavailable on some builds
        size = -1
    return (f"content_vectors={chunks};vectors={vecs};min_id={first};bytes={size}")


def doc_chunks(conn: sqlite3.Connection) -> dict[str, list[tuple[str, int, int]]]:
    """Every active indexed note: vault-relative path -> its vec0 row ids.

    Each entry is ``(vec_id, chunk_id, chunk_offset)`` in chunk order, i.e. the
    coordinates :func:`read_vector` needs. A path in more than one collection is
    kept under both of its relative paths.
    """
    out: dict[str, list[tuple[str, int, int]]] = {}
    rows = conn.execute(
        """select d.path, cv.hash || '_' || cv.seq, vr.chunk_id, vr.chunk_offset
           from documents d
           join content_vectors cv on cv.hash = d.hash
           join vectors_vec_rowids vr on vr.id = cv.hash || '_' || cv.seq
           where d.active = 1
           order by d.path, cv.seq""")
    for path, vec_id, chunk_id, offset in rows:
        out.setdefault(path, []).append((vec_id, int(chunk_id), int(offset)))
    return out


def leaf_index(docs: dict[str, list[tuple[str, int, int]]]) -> dict[str, list[str]]:
    """basename -> the relative paths that end with it.

    Two collections can hold the same leaf name, so this is one-to-many and the
    caller decides. It exists because an entity's KG source document is the
    vault-relative note path while ``files.path`` in the code collection is
    absolute — a leaf match is the only join that survives both.
    """
    out: dict[str, list[str]] = {}
    for path in docs:
        out.setdefault(Path(path).name, []).append(path)
    return out


def read_vector(conn: sqlite3.Connection, where: tuple[str, int, int],
                dim: int) -> np.ndarray | None:
    """Decode one stored vector: ``(vec_id, chunk_id, chunk_offset)`` -> float64.

    ``substr`` over the chunk BLOB, so the read is ``dim * 4`` bytes rather than
    the whole ~4 MiB chunk.
    """
    vec_id, chunk_id, offset = where
    raw = conn.execute(
        "select substr(vectors, ?, ?) from vectors_vec_vector_chunks00 where rowid = ?",
        (int(offset) * dim * 4 + 1, dim * 4, chunk_id)).fetchone()
    if not raw or not raw[0]:
        return None
    return np.frombuffer(raw[0], dtype=np.float32).astype(np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def sample_vectors(conn: sqlite3.Connection, dim: int, n: int = 200,
                   ) -> tuple[np.ndarray, int]:
    """Up to ``n`` stored vectors, in deterministic vec0-rowid order.

    Returns the (unit-normalised) matrix and the number of rows that could not
    be decoded. Deterministic because the sample has to be re-runnable: an
    artifact quoting a mean cosine is quoting this ordering.
    """
    rows = conn.execute(
        "select id, chunk_id, chunk_offset from vectors_vec_rowids order by rowid limit ?",
        (n,)).fetchall()
    vecs, skipped = [], 0
    for vec_id, chunk_id, offset in rows:
        v = read_vector(conn, (vec_id, int(chunk_id), int(offset)), dim)
        if v is None or not np.all(np.isfinite(v)):
            skipped += 1
            continue
        nrm = float(np.linalg.norm(v))
        if nrm <= 0.0:
            skipped += 1
            continue
        vecs.append(v / nrm)
    return (np.asarray(vecs, dtype=np.float64).reshape(-1, dim), skipped)


def mean_pairwise_cosine(conn: sqlite3.Connection, dim: int, n: int = 200
                         ) -> dict:
    """The measured mean cosine over a deterministic sample of the scored space.

    This is the number that says how anisotropic the space is, which is what the
    complement cosine cut has to be read against — measured here rather than
    quoted from an older item, because the embedder changed underneath the last
    one (#577).
    """
    mat, skipped = sample_vectors(conn, dim, n)
    k = mat.shape[0]
    if k < 2:
        return {"sample_size": k, "skipped": skipped, "mean_pairwise_cosine": None,
                "dim": dim}
    sims = mat @ mat.T
    iu = np.triu_indices(k, 1)
    return {"sample_size": k, "skipped": skipped, "dim": dim,
            "mean_pairwise_cosine": round(float(sims[iu].mean()), 6),
            "max_pairwise_cosine": round(float(sims[iu].max()), 6)}


def norm_name(s: str) -> str:
    """The spelling both sides of the entity/name join agree on."""
    return str(s).strip().lower().replace(" ", "-")


def note_titles(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Normalised title and path stem -> the relative paths carrying them.

    This is the join ``app/entity_linker.py`` documents for entity vectors —
    "entity name == file stem or H1 title" — and it is the ONLY per-endpoint join
    the KG offers: ``edges.source_doc`` names the note an EDGE was extracted from,
    which is one document shared by both endpoints, so resolving an entity through
    it alone gives a pair a shared component and biases every complement cosine
    upward. Titles come first; source documents are the fallback.
    """
    out: dict[str, list[str]] = {}
    for title, path in conn.execute(
            "select distinct title, path from documents where active = 1"):
        for key in {norm_name(title or ""), norm_name(Path(path).stem)}:
            if key:
                out.setdefault(key, set()).add(path)
    return {k: sorted(v) for k, v in out.items()}
