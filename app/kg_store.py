"""The knowledge-graph store: edges, aliases, entity registry, fact index.

One SQLite file (`app.paths.VAULT_KG_DB`, WAL mode) behind one module. Until
2026-09 these lived in two JSON blobs under the facts root that six programs
in three processes rewrote whole, with no lock and one atomic writer between
them. That produced the 08-22 wipe and the 09-03 suffix-merge incident, and
it meant a crash mid-write could hand the next reader a truncated file that
looked exactly like an empty graph.

Rules:
  * Nothing outside this module opens the database.
  * Every write is a transaction; `transaction()` nests.
  * Every edge and alias says where it came from (`origin`) and when.
  * The markdown fact files stay the human-readable fact layer. `facts_idx`
    is derived from them and can always be rebuilt with `facts_idx.reindex()`.

Process-level caches (adjacency, degree, alias map) key on `version()` —
SQLite's `PRAGMA data_version`, which changes whenever *another* connection
commits, combined with a local serial that this connection bumps on its own
commits. That replaces the mtime checks the JSON readers used.

Usage:
    from app.kg_store import store
    s = store()
    s.edges.add({"source": "Lloyd", "target": "vLLM", "type": "uses",
                 "provenance": "STATED", "origin": "fact_relate"})
    s.aliases.set("vllm", "vLLM", kind="case", origin="sweep")
    s.resolve("VLLM")  # -> "vLLM"
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import yaml

from app.paths import VAULT_FACTS_ROOT, VAULT_KG_DB

try:  # ~10x faster than the pure-Python loader; all input is our own files
    from yaml import CSafeLoader as _YamlLoader  # type: ignore
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _YamlLoader  # type: ignore

logger = logging.getLogger("app.kg_store")

SCHEMA_VERSION = 1

# Alias kinds — how the surface form differs from the canonical.
ALIAS_KINDS = ("self", "case", "punct", "suffix", "semantic", "manual")
# Where a row came from. Free-form is allowed, these are the known writers.
ORIGINS = ("extractor", "sweep", "semantic", "fact_add", "fact_relate", "seed",
           "classifier", "conversation", "revert", "migration", "manual", "legacy")

EDGE_COLUMNS = (
    "id", "source", "target", "type", "confidence", "provenance", "created_at",
    "expired_at", "expired_reason", "source_doc", "evidence", "classifier_model",
    "classifier_meta", "superseded_edge_id", "origin",
)
# Legacy edge fields that have no column of their own ride along in `extra`.
_EDGE_EXTRA_FIELDS = ("reason", "superseded_edge", "invalid_at")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    name        TEXT PRIMARY KEY,
    name_lc     TEXT NOT NULL,
    kind        TEXT,
    definition  TEXT,
    source_hash TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entities_lc ON entities(name_lc);

CREATE TABLE IF NOT EXISTS aliases (
    surface     TEXT PRIMARY KEY,
    surface_lc  TEXT NOT NULL,
    canonical   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    origin      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    report_path TEXT
);
CREATE INDEX IF NOT EXISTS aliases_lc ON aliases(surface_lc);
CREATE INDEX IF NOT EXISTS aliases_canonical ON aliases(canonical);

CREATE TABLE IF NOT EXISTS edges (
    id                 INTEGER PRIMARY KEY,
    source             TEXT NOT NULL,
    target             TEXT NOT NULL,
    type               TEXT NOT NULL,
    confidence         REAL NOT NULL DEFAULT 0.5,
    provenance         TEXT,
    created_at         TEXT NOT NULL,
    expired_at         TEXT,
    expired_reason     TEXT,
    source_doc         TEXT,
    evidence           TEXT,
    classifier_model   TEXT,
    classifier_meta    TEXT,
    superseded_edge_id INTEGER,
    origin             TEXT,
    extra              TEXT
);
CREATE INDEX IF NOT EXISTS edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS edges_expired ON edges(expired_at);
CREATE UNIQUE INDEX IF NOT EXISTS edges_active_unique
    ON edges(source, target, type) WHERE expired_at IS NULL;

CREATE TABLE IF NOT EXISTS facts_idx (
    entity      TEXT NOT NULL,
    category    TEXT NOT NULL,
    fact_id     TEXT,
    text_hash   TEXT NOT NULL,
    fact        TEXT,
    confidence  REAL,
    created_at  TEXT,
    valid_at    TEXT,
    source_doc  TEXT,
    source_hash TEXT,
    provenance  TEXT,
    expired_at  TEXT,
    invalid_at  TEXT,
    file_path   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_entity ON facts_idx(entity);
CREATE INDEX IF NOT EXISTS facts_file ON facts_idx(file_path);
CREATE INDEX IF NOT EXISTS facts_source_doc ON facts_idx(source_doc);
"""


class StoreUnavailable(RuntimeError):
    """The store exists but cannot be opened or read.

    Deliberately not "return an empty graph": an unreadable store must never
    look like an empty one to a writer. Read paths may catch this and degrade.
    """


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def _jsonable(v: Any) -> Any:
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    return v


# ── Store ────────────────────────────────────────────────────────────────────

class KGStore:
    """One SQLite file. Thread-safe: every operation runs under one RLock on
    one long-lived connection, so `PRAGMA data_version` means what the cache
    needs it to mean (see module docstring)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._serial = 0           # bumps on our own commits
        self._txn_depth = 0
        self._cache: dict[str, tuple[tuple[int, int], Any]] = {}
        self.edges = _Edges(self)
        self.aliases = _Aliases(self)
        self.entities = _Entities(self)
        self.facts_idx = _FactsIdx(self)
        self._open()

    # ── connection ──────────────────────────────────────────────────────
    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.path), timeout=30.0, isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._conn = conn
            self._init_schema()
        except sqlite3.Error as exc:
            raise StoreUnavailable(f"cannot open {self.path}: {exc}") from exc

    def _init_schema(self) -> None:
        # executescript() commits on its own, so it must not run inside
        # transaction(). Every statement is IF NOT EXISTS; a crash between
        # two of them just means the next open finishes the job.
        with self._lock:
            self._conn.executescript(_SCHEMA)
        with self.transaction() as c:
            c.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
                      "ON CONFLICT(key) DO NOTHING", (str(SCHEMA_VERSION),))
            # Additive migrations go here, `queue.py:_init_db` style:
            #   cols = {r[1] for r in c.execute("PRAGMA table_info(edges)")}
            #   if "new_col" not in cols: ALTER TABLE edges ADD COLUMN new_col TEXT

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StoreUnavailable(f"{self.path} is closed")
        return self._conn

    # ── transactions ────────────────────────────────────────────────────
    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """`BEGIN IMMEDIATE` … `COMMIT`, nestable. The outermost level owns
        the commit; any exception rolls the whole thing back."""
        with self._lock:
            outer = self._txn_depth == 0
            if outer:
                try:
                    self.conn.execute("BEGIN IMMEDIATE")
                except sqlite3.Error as exc:
                    raise StoreUnavailable(f"cannot begin transaction on {self.path}: {exc}") from exc
            self._txn_depth += 1
            try:
                yield self.conn
            except BaseException:
                self._txn_depth -= 1
                if outer:
                    self.conn.execute("ROLLBACK")
                raise
            else:
                self._txn_depth -= 1
                if outer:
                    self.conn.execute("COMMIT")
                    self._serial += 1
                    self._cache.clear()

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                return self.conn.execute(sql, params).fetchall()
            except sqlite3.DatabaseError as exc:
                raise StoreUnavailable(f"read failed on {self.path}: {exc}") from exc

    # ── versioning / cache ──────────────────────────────────────────────
    def version(self) -> tuple[int, int]:
        """(data_version, local_serial). Changes on any commit from any process."""
        with self._lock:
            try:
                dv = self.conn.execute("PRAGMA data_version").fetchone()[0]
            except sqlite3.DatabaseError as exc:
                raise StoreUnavailable(f"read failed on {self.path}: {exc}") from exc
            return (int(dv), self._serial)

    def cached(self, key: str, build):
        """Memoise `build()` under `key` until the store version moves."""
        with self._lock:
            v = self.version()
            hit = self._cache.get(key)
            if hit is not None and hit[0] == v:
                return hit[1]
            value = build()
            self._cache[key] = (v, value)
            return value

    def invalidate_caches(self) -> None:
        with self._lock:
            self._cache.clear()

    # ── meta ────────────────────────────────────────────────────────────
    def meta_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        rows = self._query("SELECT value FROM meta WHERE key=?", (key,))
        return rows[0]["value"] if rows else default

    def meta_set(self, key: str, value: Any) -> None:
        with self.transaction() as c:
            c.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, str(value)))

    # ── resolution ──────────────────────────────────────────────────────
    def resolve(self, name: str) -> Optional[str]:
        """alias table → entity registry (exact, then case-insensitive)."""
        name = (name or "").strip()
        if not name:
            return None
        hit = self.aliases.resolve(name)
        if hit is not None:
            return hit
        return self.entities.lookup(name)

    # ── export / import / backup ────────────────────────────────────────
    def export_json(self, dest_dir: Path | str) -> dict[str, Path]:
        """Write `_relationships.json` and `entity-aliases.json` in the legacy
        format under `dest_dir`. Self-identities are included in the alias
        file (the legacy readers expected them). Returns the paths."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        from app.atomic_io import atomic_write_text
        edges = [self.edges._row_to_legacy(r) for r in self.edges.all_rows()]
        rel = dest / "_relationships.json"
        atomic_write_text(rel, json.dumps({"schema_version": 1, "edges": edges}, indent=2,
                                          ensure_ascii=False), fsync=True)
        alias_map = self.aliases.all()
        for name in self.entities.all():
            alias_map.setdefault(name, name)
        al = dest / "entity-aliases.json"
        atomic_write_text(al, json.dumps(alias_map, indent=2, sort_keys=True,
                                         ensure_ascii=False), fsync=True)
        return {"relationships": rel, "aliases": al}

    def import_json(self, rel_path: Optional[Path | str] = None,
                    alias_path: Optional[Path | str] = None,
                    *, classify_alias=None, origin: str = "migration") -> dict[str, int]:
        """Load the legacy JSON files. Idempotent on edges (an identical
        (source, target, type, created_at) row is not inserted twice) and on
        aliases (upsert). `classify_alias(surface, canonical) -> kind` names the
        alias kind; default is a shape guess."""
        stats = {"edges": 0, "edges_skipped": 0, "aliases": 0, "entities": 0}
        with self.transaction():
            if rel_path and Path(rel_path).exists():
                data = json.loads(Path(rel_path).read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("edges"), list):
                    raise StoreUnavailable(f"{rel_path} has no `edges` list")
                existing = {
                    (r["source"], r["target"], r["type"], r["created_at"])
                    for r in self._query("SELECT source, target, type, created_at FROM edges")
                }
                for e in data["edges"]:
                    key = (e.get("source"), e.get("target"), e.get("type"), e.get("created_at"))
                    if key in existing or not all(key[:3]):
                        stats["edges_skipped"] += 1
                        continue
                    self.edges._insert_raw(e, origin=e.get("origin") or origin, allow_dup_active=True)
                    existing.add(key)
                    stats["edges"] += 1
            if alias_path and Path(alias_path).exists():
                raw = json.loads(Path(alias_path).read_text(encoding="utf-8"))
                for surface, canonical in raw.items():
                    if not surface or not canonical:
                        continue
                    if surface == canonical:
                        if self.entities.register(canonical) is not None:
                            stats["entities"] += 1
                        continue
                    kind = classify_alias(surface, canonical) if classify_alias else _guess_kind(surface, canonical)
                    self.aliases.set(surface, canonical, kind=kind, origin=origin)
                    stats["aliases"] += 1
        return stats

    def backup(self, dest_path: Path | str) -> Path:
        """Consistent copy via the SQLite backup API — safe under writers."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        with self._lock:
            out = sqlite3.connect(str(tmp))
            try:
                self.conn.backup(out)
            finally:
                out.close()
        os.replace(tmp, dest)
        return dest

    def integrity_check(self) -> str:
        return self._query("PRAGMA integrity_check")[0][0]

    def stats(self) -> dict[str, int]:
        return {
            "edges_total": self.edges.count(active_only=False),
            "edges_active": self.edges.count(active_only=True),
            "aliases": self.aliases.count(),
            "entities": self.entities.count(),
            "facts": self.facts_idx.count(),
        }


_ALIAS_TOKEN_RE = __import__("re").compile(r"[a-z0-9]+")


def alias_kind(surface: str, canonical: str) -> str:
    """How a surface form differs from its canonical.

    The one implementation — entity_naming, _shared and the migration all
    call this, so an alias written by the extractor is classified the same
    way as one written by the sweep.

    Tokenises on non-alphanumerics rather than whitespace, so `vllm-engine`
    against `vLLM` reads as a suffix difference, not an unrelated semantic
    merge.
    """
    if surface == canonical:
        return "self"
    if surface.lower() == canonical.lower():
        return "case"
    st = _ALIAS_TOKEN_RE.findall(surface.lower())
    ct = _ALIAS_TOKEN_RE.findall(canonical.lower())
    if st == ct:
        return "punct"
    if st and ct and (st[:len(ct)] == ct or ct[:len(st)] == st):
        return "suffix"
    return "semantic"


# Back-compat name used by import_json's default.
_guess_kind = alias_kind


# ── Edges ────────────────────────────────────────────────────────────────────

class _Edges:
    def __init__(self, store: KGStore):
        self._s = store

    # ── row shaping ─────────────────────────────────────────────────────
    @staticmethod
    def _row_to_dict(r: sqlite3.Row) -> dict:
        d = {k: r[k] for k in EDGE_COLUMNS}
        if d.get("classifier_meta"):
            try:
                d["classifier_meta"] = json.loads(d["classifier_meta"])
            except (TypeError, ValueError):
                pass
        extra = r["extra"]
        if extra:
            try:
                d.update(json.loads(extra))
            except (TypeError, ValueError):
                pass
        return d

    @classmethod
    def _row_to_legacy(cls, r: sqlite3.Row) -> dict:
        """Today's `_relationships.json` shape plus `id`/`origin`/`evidence`."""
        d = cls._row_to_dict(r)
        return {k: v for k, v in d.items() if v is not None or k in
                ("expired_at", "source_doc", "confidence", "provenance", "created_at")}

    def _insert_raw(self, e: dict, *, origin: str, allow_dup_active: bool = False) -> int:
        extra = {k: e[k] for k in _EDGE_EXTRA_FIELDS if e.get(k) is not None}
        for k, v in e.items():
            if k not in EDGE_COLUMNS and k not in _EDGE_EXTRA_FIELDS and k != "extra" and v is not None:
                extra[k] = v
        meta = e.get("classifier_meta")
        if isinstance(meta, (dict, list)):
            meta = json.dumps(meta, ensure_ascii=False)
        expired_at = e.get("expired_at") or None
        params = (
            e["source"], e["target"], e["type"],
            float(e.get("confidence", 0.5) or 0.0),
            e.get("provenance"), e.get("created_at") or _now(),
            expired_at, e.get("expired_reason"),
            e.get("source_doc"), e.get("evidence"),
            e.get("classifier_model"), meta,
            e.get("superseded_edge_id"), origin,
            json.dumps(extra, ensure_ascii=False, default=str) if extra else None,
        )
        sql = ("INSERT INTO edges(source, target, type, confidence, provenance, created_at, "
               "expired_at, expired_reason, source_doc, evidence, classifier_model, "
               "classifier_meta, superseded_edge_id, origin, extra) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
        c = self._s.conn
        try:
            cur = c.execute(sql, params)
        except sqlite3.IntegrityError:
            if not allow_dup_active or expired_at is not None:
                raise
            # Legacy data can hold two active rows for one key. Keep the
            # first as active and bring this one in expired, so nothing
            # is lost and the unique index holds from here on.
            params = params[:6] + (e.get("created_at") or _now(),
                                   "migration: duplicate active edge") + params[8:]
            cur = c.execute(sql, params)
        return int(cur.lastrowid)

    # ── reads ───────────────────────────────────────────────────────────
    def by_id(self, edge_id: int) -> Optional[dict]:
        rows = self._s._query("SELECT * FROM edges WHERE id=?", (edge_id,))
        return self._row_to_dict(rows[0]) if rows else None

    def all_rows(self, include_expired: bool = True) -> list[sqlite3.Row]:
        sql = "SELECT * FROM edges" + ("" if include_expired else " WHERE expired_at IS NULL")
        return self._s._query(sql + " ORDER BY id")

    def all(self, include_expired: bool = True) -> list[dict]:
        return [self._row_to_dict(r) for r in self.all_rows(include_expired)]

    def active(self, source: Optional[str] = None, target: Optional[str] = None,
               types: Optional[Iterable[str]] = None, *, either: Optional[str] = None,
               min_confidence: float = 0.0) -> list[dict]:
        """Active edges, optionally filtered. `either=name` matches source OR target."""
        where = ["expired_at IS NULL"]
        params: list[Any] = []
        if source is not None:
            where.append("source=?"); params.append(source)
        if target is not None:
            where.append("target=?"); params.append(target)
        if either is not None:
            where.append("(source=? OR target=?)"); params.extend([either, either])
        if types:
            tl = list(types)
            where.append("type IN (%s)" % ",".join("?" * len(tl))); params.extend(tl)
        if min_confidence > 0:
            where.append("confidence >= ?"); params.append(float(min_confidence))
        rows = self._s._query("SELECT * FROM edges WHERE " + " AND ".join(where) + " ORDER BY id",
                              tuple(params))
        return [self._row_to_dict(r) for r in rows]

    def count(self, active_only: bool = True) -> int:
        sql = "SELECT COUNT(*) FROM edges" + (" WHERE expired_at IS NULL" if active_only else "")
        return int(self._s._query(sql)[0][0])

    def find_active(self, source: str, target: str, type_: str) -> Optional[dict]:
        rows = self._s._query(
            "SELECT * FROM edges WHERE source=? AND target=? AND type=? AND expired_at IS NULL",
            (source, target, type_))
        return self._row_to_dict(rows[0]) if rows else None

    def adjacency(self, min_confidence: float = 0.0) -> dict[str, list[dict]]:
        """node → [edge dict] over active edges; every edge listed under both
        endpoints. Cached until the store version moves; the confidence
        filter is applied on the cached structure."""
        def build():
            adj: dict[str, list[dict]] = {}
            for r in self._s._query(
                "SELECT id, source, target, type, confidence, provenance, source_doc "
                "FROM edges WHERE expired_at IS NULL"
            ):
                e = {"id": r["id"], "source": r["source"], "target": r["target"],
                     "type": r["type"], "confidence": float(r["confidence"] or 0.0),
                     "provenance": r["provenance"], "source_doc": r["source_doc"]}
                adj.setdefault(e["source"], []).append(e)
                if e["target"] != e["source"]:
                    adj.setdefault(e["target"], []).append(e)
            return adj
        adj = self._s.cached("adjacency", build)
        if min_confidence <= 0:
            return adj
        return {n: [e for e in es if e["confidence"] >= min_confidence] for n, es in adj.items()}

    def degree(self) -> dict[str, int]:
        """Entity → active-edge count. Cached until the store version moves."""
        def build():
            counts: dict[str, int] = {}
            for r in self._s._query(
                "SELECT source, target FROM edges WHERE expired_at IS NULL"):
                counts[r["source"]] = counts.get(r["source"], 0) + 1
                counts[r["target"]] = counts.get(r["target"], 0) + 1
            return counts
        return self._s.cached("degree", build)

    def degree_ci(self) -> dict[str, int]:
        """Lowercased entity → summed degree, for callers that key on lowercase."""
        def build():
            out: dict[str, int] = {}
            for name, n in self.degree().items():
                k = name.lower()
                out[k] = out.get(k, 0) + n
            return out
        return self._s.cached("degree_ci", build)

    def nodes(self) -> set[str]:
        return set(self.degree().keys())

    # ── writes ──────────────────────────────────────────────────────────
    def add(self, edge: dict, *, origin: Optional[str] = None) -> int:
        """Insert an active edge. Returns the id of the new row, or of the
        existing active (source, target, type) row when one is already there.
        Self-loops are refused."""
        src, tgt, typ = (edge.get("source") or "").strip(), (edge.get("target") or "").strip(), (edge.get("type") or "").strip()
        if not src or not tgt or not typ:
            raise ValueError("edge needs source, target and type")
        if src == tgt:
            raise ValueError(f"refusing self-loop edge on {src!r}")
        e = dict(edge, source=src, target=tgt, type=typ)
        e["expired_at"] = None
        with self._s.transaction():
            existing = self.find_active(src, tgt, typ)
            if existing is not None:
                return int(existing["id"])
            return self._insert_raw(e, origin=origin or e.get("origin") or "unknown")

    def expire(self, edge_id: int, reason: str, at: Optional[str] = None) -> bool:
        with self._s.transaction() as c:
            cur = c.execute(
                "UPDATE edges SET expired_at=?, expired_reason=? WHERE id=? AND expired_at IS NULL",
                (at or _now(), reason, edge_id))
            return cur.rowcount > 0

    def reactivate(self, edge_id: int) -> bool:
        """Undo `expire`. Any *other* active row on the same key is expired
        first so the unique index keeps holding."""
        with self._s.transaction() as c:
            row = self.by_id(edge_id)
            if row is None or row["expired_at"] is None:
                return False
            other = self.find_active(row["source"], row["target"], row["type"])
            if other is not None and other["id"] != edge_id:
                self.expire(other["id"], f"reactivated edge {edge_id} takes this key")
            c.execute("UPDATE edges SET expired_at=NULL, expired_reason=NULL WHERE id=?", (edge_id,))
            return True

    def retype(self, old_id: int, new_edge: dict, *, origin: str, reason: str = "retyped") -> int:
        """Expire `old_id` and add `new_edge` with `superseded_edge_id=old_id`,
        in one transaction. Any other active edge on the same (source, target)
        pair — whatever its type — is expired too, so one pair has one typed
        relation at a time."""
        with self._s.transaction():
            old = self.by_id(old_id)
            if old is None:
                raise ValueError(f"edge {old_id} not found")
            src = new_edge.get("source") or old["source"]
            tgt = new_edge.get("target") or old["target"]
            self.expire(old_id, reason)
            for other in self.active(source=src, target=tgt):
                if other["id"] != old_id:
                    self.expire(other["id"], f"{reason}: pair re-typed as {new_edge.get('type')}")
            e = dict(new_edge, source=src, target=tgt, superseded_edge_id=old_id)
            e.setdefault("source_doc", old.get("source_doc"))
            e.setdefault("created_at", _now())
            e["expired_at"] = None
            return self._insert_raw(e, origin=origin)

    def rewrite_endpoint(self, old_name: str, new_name: str, *, origin: str,
                         reason: Optional[str] = None) -> list[tuple[int, int]]:
        """Merge helper: every active edge touching `old_name` is expired and
        re-added with `new_name` in its place. History survives; returns
        [(old_id, new_id)] so a revert can be exact. When the rewritten edge
        already exists active, new_id is that existing row's id."""
        reason = reason or f"merge: {old_name!r} -> {new_name!r}"
        pairs: list[tuple[int, int]] = []
        with self._s.transaction():
            for e in self.active(either=old_name):
                src = new_name if e["source"] == old_name else e["source"]
                tgt = new_name if e["target"] == old_name else e["target"]
                self.expire(e["id"], reason)
                if src == tgt:
                    # A merge that folds an edge onto itself has nothing to keep.
                    continue
                existing = self.find_active(src, tgt, e["type"])
                if existing is not None:
                    # Keep the earlier created_at and the higher confidence, as
                    # the JSON-era sweep did on rewrite-induced collisions.
                    self._s.conn.execute(
                        "UPDATE edges SET confidence=MAX(confidence, ?), "
                        "created_at=MIN(created_at, ?) WHERE id=?",
                        (e["confidence"], e["created_at"], existing["id"]))
                    pairs.append((e["id"], existing["id"]))
                    continue
                new = {k: v for k, v in e.items() if k not in ("id", "expired_at", "expired_reason")}
                new.update(source=src, target=tgt, superseded_edge_id=e["id"])
                new_id = self._insert_raw(new, origin=origin)
                pairs.append((e["id"], new_id))
        return pairs

    def revert_rewrites(self, pairs: Iterable[tuple[int, int]], *, reason: str) -> int:
        """Exact inverse of `rewrite_endpoint`: expire new_id, reactivate old_id."""
        n = 0
        with self._s.transaction():
            for old_id, new_id in pairs:
                new = self.by_id(new_id)
                if new is not None and new.get("superseded_edge_id") == old_id:
                    self.expire(new_id, reason)
                if self.reactivate(old_id):
                    n += 1
        return n


# ── Aliases ──────────────────────────────────────────────────────────────────

class _Aliases:
    def __init__(self, store: KGStore):
        self._s = store

    def resolve(self, name: str) -> Optional[str]:
        """Case-insensitive surface → canonical. Exact-case match wins."""
        name = (name or "").strip()
        if not name:
            return None
        rows = self._s._query(
            "SELECT surface, canonical FROM aliases WHERE surface_lc=? ORDER BY created_at",
            (name.lower(),))
        if not rows:
            return None
        for r in rows:
            if r["surface"] == name:
                return r["canonical"]
        return rows[0]["canonical"]

    def set(self, surface: str, canonical: str, *, kind: str, origin: str,
            report_path: Optional[str] = None) -> None:
        surface, canonical = surface.strip(), canonical.strip()
        if not surface or not canonical:
            raise ValueError("alias needs surface and canonical")
        if surface == canonical:
            # A self-identity is an entity registration, not an alias.
            self._s.entities.register(canonical)
            return
        if kind not in ALIAS_KINDS:
            raise ValueError(f"unknown alias kind {kind!r}")
        with self._s.transaction() as c:
            c.execute(
                "INSERT INTO aliases(surface, surface_lc, canonical, kind, origin, created_at, report_path) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(surface) DO UPDATE SET "
                "canonical=excluded.canonical, kind=excluded.kind, origin=excluded.origin, "
                "created_at=excluded.created_at, report_path=excluded.report_path",
                (surface, surface.lower(), canonical, kind, origin, _now(), report_path))

    def remove(self, surface: str) -> bool:
        with self._s.transaction() as c:
            return c.execute("DELETE FROM aliases WHERE surface=?", (surface,)).rowcount > 0

    def remove_where(self, *, canonical: str, surface_lc: Optional[str] = None) -> list[str]:
        """Drop aliases routing to `canonical` (optionally only one surface,
        case-insensitively). Returns the surfaces removed — for revert reports."""
        with self._s.transaction() as c:
            if surface_lc is not None:
                rows = c.execute("SELECT surface FROM aliases WHERE canonical=? AND surface_lc=?",
                                 (canonical, surface_lc.lower())).fetchall()
                c.execute("DELETE FROM aliases WHERE canonical=? AND surface_lc=?",
                          (canonical, surface_lc.lower()))
            else:
                rows = c.execute("SELECT surface FROM aliases WHERE canonical=?", (canonical,)).fetchall()
                c.execute("DELETE FROM aliases WHERE canonical=?", (canonical,))
            return [r["surface"] for r in rows]

    def for_canonical(self, canonical: str) -> list[dict]:
        return [dict(r) for r in self._s._query(
            "SELECT surface, kind, origin, created_at, report_path FROM aliases WHERE canonical=? ORDER BY surface",
            (canonical,))]

    def all(self) -> dict[str, str]:
        """surface → canonical (no self-identities; see entities.all())."""
        return {r["surface"]: r["canonical"] for r in
                self._s._query("SELECT surface, canonical FROM aliases ORDER BY surface")}

    def all_lower(self) -> dict[str, str]:
        """lowercased surface → canonical, plus every entity mapped to itself.
        Cached until the store version moves — this is the hot lookup map."""
        def build():
            out: dict[str, str] = {}
            for r in self._s._query("SELECT surface_lc, canonical FROM aliases ORDER BY created_at"):
                out.setdefault(r["surface_lc"], r["canonical"])
            for r in self._s._query("SELECT name_lc, name FROM entities ORDER BY created_at"):
                out.setdefault(r["name_lc"], r["name"])
            return out
        return self._s.cached("alias_map_lower", build)

    def rows(self) -> list[dict]:
        return [dict(r) for r in self._s._query("SELECT * FROM aliases ORDER BY surface")]

    def count(self) -> int:
        return int(self._s._query("SELECT COUNT(*) FROM aliases")[0][0])


# ── Entities ─────────────────────────────────────────────────────────────────

class _Entities:
    def __init__(self, store: KGStore):
        self._s = store

    def register(self, name: str, *, kind: Optional[str] = None,
                 definition: Optional[str] = None, source_hash: Optional[str] = None) -> Optional[str]:
        """Insert `name` if missing (exact case). Returns the name when a row
        was created, None when it already existed. Never merges cases: two
        directories that differ only by case are two rows, as on disk."""
        name = (name or "").strip()
        if not name:
            return None
        with self._s.transaction() as c:
            cur = c.execute(
                "INSERT INTO entities(name, name_lc, kind, definition, source_hash, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(name) DO NOTHING",
                (name, name.lower(), kind, definition, source_hash, _now(), _now()))
            if cur.rowcount == 0:
                if kind or definition or source_hash:
                    c.execute(
                        "UPDATE entities SET kind=COALESCE(?, kind), definition=COALESCE(?, definition), "
                        "source_hash=COALESCE(?, source_hash), updated_at=? WHERE name=?",
                        (kind, definition, source_hash, _now(), name))
                return None
            return name

    def exists(self, name: str) -> bool:
        return bool(self._s._query("SELECT 1 FROM entities WHERE name=?", (name,)))

    def lookup(self, name: str) -> Optional[str]:
        """Exact-case match first, then the earliest-registered case variant."""
        name = (name or "").strip()
        if not name:
            return None
        rows = self._s._query("SELECT name FROM entities WHERE name_lc=? ORDER BY created_at, name",
                              (name.lower(),))
        if not rows:
            return None
        for r in rows:
            if r["name"] == name:
                return name
        return rows[0]["name"]

    def get(self, name: str) -> Optional[dict]:
        rows = self._s._query("SELECT * FROM entities WHERE name=?", (name,))
        return dict(rows[0]) if rows else None

    def rename(self, old: str, new: str) -> bool:
        with self._s.transaction() as c:
            row = self.get(old)
            if row is None:
                return False
            c.execute("DELETE FROM entities WHERE name=?", (old,))
            c.execute(
                "INSERT INTO entities(name, name_lc, kind, definition, source_hash, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET updated_at=excluded.updated_at",
                (new, new.lower(), row["kind"], row["definition"], row["source_hash"],
                 row["created_at"], _now()))
            return True

    def remove(self, name: str) -> bool:
        with self._s.transaction() as c:
            return c.execute("DELETE FROM entities WHERE name=?", (name,)).rowcount > 0

    def all(self) -> list[str]:
        return [r["name"] for r in self._s._query("SELECT name FROM entities ORDER BY name")]

    def rows(self) -> list[dict]:
        return [dict(r) for r in self._s._query("SELECT * FROM entities ORDER BY name")]

    def count(self) -> int:
        return int(self._s._query("SELECT COUNT(*) FROM entities")[0][0])


# ── Fact index ───────────────────────────────────────────────────────────────

def parse_fact_file(path: Path) -> tuple[dict, list[dict]]:
    """(frontmatter, facts) for one markdown fact file. Corrupt or non-fact
    files come back as ({}, []); the caller decides what that means."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}, []
    if not text.startswith("---"):
        return {}, []
    end = text.find("\n---", 3)
    if end == -1:
        return {}, []
    try:
        fm = yaml.load(text[3:end], Loader=_YamlLoader) or {}
    except Exception:
        return {}, []
    if not isinstance(fm, dict):
        return {}, []
    facts = fm.get("facts")
    if not isinstance(facts, list):
        return fm, []
    return fm, [f for f in facts if isinstance(f, dict)]


class _FactsIdx:
    def __init__(self, store: KGStore):
        self._s = store

    # ── build ───────────────────────────────────────────────────────────
    def _rows_for_file(self, path: Path, root: Path) -> list[tuple]:
        fm, facts = parse_fact_file(path)
        if not facts:
            return []
        entity = str(fm.get("entity") or path.parent.name)
        category = str(fm.get("category") or (path.stem.rsplit("-", 1)[-1] if "-" in path.stem else "general"))
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        rows = []
        for f in facts:
            text = str(f.get("fact") or "")
            if not text.strip():
                continue
            rows.append((
                str(f.get("entity") or entity), str(f.get("category") or category),
                f.get("id"), _text_hash(text), text,
                _float_or_none(f.get("confidence")),
                _jsonable(f.get("created_at")), _jsonable(f.get("valid_at")),
                f.get("source_doc"), f.get("source_hash"), f.get("provenance"),
                _jsonable(f.get("expired_at")), _jsonable(f.get("invalid_at")), rel,
            ))
        return rows

    _INSERT = ("INSERT INTO facts_idx(entity, category, fact_id, text_hash, fact, confidence, "
               "created_at, valid_at, source_doc, source_hash, provenance, expired_at, invalid_at, file_path) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)")

    def reindex(self, paths: Optional[Iterable[Path]] = None, root: Optional[Path] = None,
                *, register_entities: bool = True) -> dict[str, int]:
        """Rebuild the index from the markdown files. With `paths`, only those
        files are re-read (rows for each path are replaced); without, the
        whole tree under `root` is walked and the table replaced. Entity dirs
        seen are registered."""
        root = Path(root or VAULT_FACTS_ROOT)
        stats = {"files": 0, "facts": 0, "entities_registered": 0}
        if paths is None:
            # Sorted so a rebuild indexes in the same order every time: the
            # row order is what `for_entity` returns, and a diff of two
            # reindexes should be empty when nothing changed.
            files = sorted(p for d in sorted(root.iterdir())
                           if d.is_dir() and not d.name.startswith((".", "_"))
                           for p in d.glob("*.md")) if root.exists() else []
            full = True
        else:
            files = [Path(p) for p in paths]
            full = False
        with self._s.transaction() as c:
            if full:
                c.execute("DELETE FROM facts_idx")
            seen_dirs: set[str] = set()
            for p in files:
                try:
                    rel = str(p.relative_to(root))
                except ValueError:
                    rel = str(p)
                if not full:
                    c.execute("DELETE FROM facts_idx WHERE file_path=?", (rel,))
                if not p.exists():
                    continue
                rows = self._rows_for_file(p, root)
                if rows:
                    c.executemany(self._INSERT, rows)
                stats["files"] += 1
                stats["facts"] += len(rows)
                seen_dirs.add(p.parent.name)
            if register_entities:
                for name in seen_dirs:
                    if self._s.entities.register(name) is not None:
                        stats["entities_registered"] += 1
            if full:
                c.execute("INSERT INTO meta(key, value) VALUES ('last_reindex', ?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (_now(),))
        return stats

    def update_file(self, path: Path, root: Optional[Path] = None) -> int:
        """Re-read one fact file after a write. Returns rows indexed."""
        return self.reindex([path], root=root)["facts"]

    # ── reads ───────────────────────────────────────────────────────────
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {k: r[k] for k in r.keys()}

    def for_entity(self, name: str, category: Optional[str] = None, as_of: Optional[str] = None,
                   include_expired: bool = False) -> list[dict]:
        where, params = ["entity=?"], [name]
        if category:
            where.append("category=?"); params.append(category)
        if not include_expired:
            if as_of:
                where.append("(valid_at IS NULL OR valid_at <= ?)"); params.append(as_of)
                where.append("(expired_at IS NULL OR expired_at > ?)"); params.append(as_of)
                where.append("(invalid_at IS NULL OR invalid_at > ?)"); params.append(as_of)
            else:
                where.append("expired_at IS NULL AND invalid_at IS NULL")
        rows = self._s._query("SELECT * FROM facts_idx WHERE " + " AND ".join(where) + " ORDER BY rowid",
                              tuple(params))
        return [self._row(r) for r in rows]

    def count(self, entity: Optional[str] = None, active_only: bool = False) -> int:
        sql, params = "SELECT COUNT(*) FROM facts_idx", []
        where = []
        if entity is not None:
            where.append("entity=?"); params.append(entity)
        if active_only:
            where.append("expired_at IS NULL AND invalid_at IS NULL")
        if where:
            sql += " WHERE " + " AND ".join(where)
        return int(self._s._query(sql, tuple(params))[0][0])

    def entity_fact_counts(self, active_only: bool = True) -> dict[str, int]:
        """entity → fact count. Cached until the store version moves."""
        key = "fact_counts_active" if active_only else "fact_counts_all"
        def build():
            sql = "SELECT entity, COUNT(*) AS n FROM facts_idx"
            if active_only:
                sql += " WHERE expired_at IS NULL AND invalid_at IS NULL"
            sql += " GROUP BY entity"
            return {r["entity"]: int(r["n"]) for r in self._s._query(sql)}
        return self._s.cached(key, build)

    def categories_for(self, entity: str) -> list[str]:
        return [r["category"] for r in self._s._query(
            "SELECT DISTINCT category FROM facts_idx WHERE entity=? ORDER BY category", (entity,))]

    def entities_with_category(self, category: str) -> set[str]:
        return {r["entity"] for r in self._s._query(
            "SELECT DISTINCT entity FROM facts_idx WHERE category=?", (category,))}

    def search(self, q: str, limit: int = 50) -> list[dict]:
        like = f"%{q.lower()}%"
        rows = self._s._query(
            "SELECT * FROM facts_idx WHERE LOWER(fact) LIKE ? OR LOWER(entity) LIKE ? LIMIT ?",
            (like, like, int(limit)))
        return [self._row(r) for r in rows]

    def last_reindex(self) -> Optional[str]:
        return self._s.meta_get("last_reindex")


def _float_or_none(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ── Process default ──────────────────────────────────────────────────────────

_default_lock = threading.Lock()
_default: Optional[KGStore] = None
_default_path: Path = VAULT_KG_DB


def store() -> KGStore:
    """The process-wide store for `app.paths.VAULT_KG_DB` (or whatever
    `configure()` pointed it at). Opened lazily on first use."""
    global _default
    with _default_lock:
        if _default is None or _default.path != _default_path:
            if _default is not None:
                _default.close()
            _default = KGStore(_default_path)
        return _default


def configure(path: Path | str) -> KGStore:
    """Point the process default at `path` (tests, rebuilds). Returns it."""
    global _default_path, _default
    with _default_lock:
        _default_path = Path(path)
        if _default is not None:
            _default.close()
            _default = None
    return store()


def reset() -> None:
    """Close the default store and forget it (next `store()` reopens)."""
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
            _default = None
