"""Query → entity linking beyond the lexical extractor (#1486).

`agent_mcp.retrieval.extract_entities_from_query` ranks entity names by
substring and token overlap. That cannot reach a query that paraphrases its
subject — "the job that keeps rewriting the code it itself runs in" is about
`automod` and shares no word with it — and 25 of the gold set's 66
entity-labelled queries are anchorless for exactly that reason (#1502's
canonical count).

This module owns the two switches under `retrieval.entity_seeding`:

``alias_surfaces``
    Match alias SURFACES as names, scoring their canonical (retrieval.py).
``semantic``
    Embedding retrieval over entity ``name (kind): top facts`` vectors, unioned
    with the lexical seeds (retrieval.py appends; never substitutes).

Both default OFF. The vectors are built offline by
``scripts/memory/build_entity_vectors.py`` into ``paths.ENTITY_VECTORS_DIR``;
the query embedder is pluggable (``set_embedder``) so tests stub it.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Optional

DEFAULTS = {
    "alias_surfaces": False,
    "semantic": {"enabled": False, "k": 3, "min_score": 0.0},
}

#: The instruction Qwen3-Embedding expects on the QUERY side only. Changing it
#: changes every query vector; the entity side is embedded without one.
QUERY_INSTRUCTION = ("Instruct: Given a question about a personal knowledge vault, "
                     "retrieve the entity it is about\nQuery: ")


#: Set to "0" and both switches read off whatever config says. `tests/conftest.py`
#: sets it for every test, so the recall tests keep the lexical path they were
#: written against (subprocesses inherit it); it is also an operator kill switch.
KILL_ENV = "LLOYD_ENTITY_SEEDING"


def seeding_config() -> dict:
    """`retrieval.entity_seeding` over DEFAULTS. Fails closed (all off)."""
    cfg = {"alias_surfaces": DEFAULTS["alias_surfaces"], "semantic": dict(DEFAULTS["semantic"])}
    import os
    if os.environ.get(KILL_ENV, "").strip() == "0":
        return cfg
    try:
        from app.config import CONFIG
        block = (((CONFIG or {}).get("retrieval") or {}).get("entity_seeding") or {})
    except Exception:  # noqa: BLE001
        return cfg
    if not isinstance(block, dict):
        return cfg
    if "alias_surfaces" in block:
        cfg["alias_surfaces"] = bool(block["alias_surfaces"])
    sem = block.get("semantic")
    if isinstance(sem, dict):
        for k in DEFAULTS["semantic"]:
            if k in sem:
                cfg["semantic"][k] = sem[k]
        cfg["semantic"]["enabled"] = bool(cfg["semantic"]["enabled"])
        cfg["semantic"]["k"] = max(0, int(cfg["semantic"]["k"]))
        cfg["semantic"]["min_score"] = float(cfg["semantic"]["min_score"])
    return cfg


# ── the entity vector index ─────────────────────────────────────────────────

_lock = threading.Lock()
_index: Optional[tuple] = None          # (dir, mtime, names, matrix)
_embedder: Optional[Callable[[str], "object"]] = None


def vectors_dir() -> Path:
    from app import paths
    return Path(paths.ENTITY_VECTORS_DIR)


def _load_index():
    import numpy as np
    d = vectors_dir()
    names_p, mat_p = d / "names.json", d / "vectors.npy"
    if not names_p.exists() or not mat_p.exists():
        return None
    mtime = mat_p.stat().st_mtime
    global _index
    with _lock:
        if _index is not None and _index[0] == d and _index[1] == mtime:
            return _index
        names = json.loads(names_p.read_text())
        mat = np.load(mat_p, mmap_mode="r")
        if len(names) != mat.shape[0]:
            return None
        _index = (d, mtime, names, np.asarray(mat, dtype=np.float32))
        return _index


def set_embedder(fn: Optional[Callable[[str], "object"]]) -> None:
    """Install the query embedder: `fn(text) -> unit vector`. None restores the default."""
    global _embedder
    _embedder = fn


def _embed_query(text: str):
    fn = _embedder
    if fn is None:
        from app.qwen3_embed import embed_query as fn  # lazy: loads weights on first use
    return fn(QUERY_INSTRUCTION + text)


def semantic_candidates(query: str, k: int, *, allowed: Optional[set] = None,
                        min_score: float = 0.0) -> list[tuple[str, float]]:
    """Top-`k` entities by cosine to the query, restricted to `allowed` names.

    `[]` on any failure (no index, no embedder, a bad vector): the semantic leg
    is additive, so its absence must read as "no extra seeds", never as an error
    that costs the lexical ones.
    """
    if k <= 0 or not (query or "").strip():
        return []
    try:
        import numpy as np
        idx = _load_index()
        if idx is None:
            return []
        _d, _m, names, mat = idx
        q = np.asarray(_embed_query(query), dtype=np.float32).reshape(-1)
        if q.shape[0] != mat.shape[1]:
            return []
        n = float(np.linalg.norm(q)) or 1.0
        sims = mat @ (q / n)
        order = np.argsort(-sims)
        out = []
        for i in order:
            s = float(sims[i])
            if s < min_score:
                break
            name = names[int(i)]
            if allowed is not None and name.lower() not in allowed:
                continue
            out.append((name, s))
            if len(out) >= k:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


def reset_cache() -> None:
    global _index
    with _lock:
        _index = None
