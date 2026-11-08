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

#: How old the served entity-vector artifact may get before it is called STALE.
#: 10.0 days = the weekly rebuild period the `entity-vectors-rebuild` skill declares
#: plus its retry headroom (the skill's job runs with 6h/12h/24h backoff, so one
#: missed week still reads as fresh rather than flipping on a Tuesday morning).
#:
#: It is a REPORTING bound and the ONLY thing `--status` exits nonzero on. It never
#: gates a seed and never will: the semantic leg is additive over the lexical seeds
#: (#1486 measured +0.163 entity_hit on a paired pinned corpus), so an index that is
#: merely old still retrieves, and refusing to use it would silently subtract a
#: measured gain on a threshold nobody validated against recall. What the bound is
#: for is the failure #3907 recorded: a (11990,1024) artifact with mtime
#: 2026-09-26T00:22Z sat behind `semantic.enabled: true` for 42 days while the
#: loader read that mtime for cache invalidation only, so nothing anywhere could
#: distinguish "recently rebuilt" from "never rebuilt".
#:
#: 43 days of a weekly job is 6 missed runs. Two things that look like a number but
#: are not: `coverage` is reported without one, because a live entity with no vector
#: costs recall only for queries about IT — unmeasurable per query — whereas an index
#: nobody has refreshed for a month is a schedule that died; and the loader's log line
#: does not consult this bound at all, so a load never changes behaviour.
#: `scripts/memory/build_entity_vectors.py --status` is the reader; the value lives
#: here because the loader reads the artifact and must not import a script to know
#: how old it is.
MAX_INDEX_AGE_DAYS = 10.0



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
_index_status: Optional[dict] = None    # what `index_status()` returns; see below
_embedder: Optional[Callable[[str], "object"]] = None


def vectors_dir() -> Path:
    from app import paths
    return Path(paths.ENTITY_VECTORS_DIR)


def _live_entity_names() -> Optional[set]:
    """Every name in the live `entities` table, or None when the store is unreadable.

    Read once per cold index load, never per query: the whole point of #3907 is that
    coverage is the number nobody could see, and it is only visible against the store
    the artifact was NOT built from. A failure here is reported as `None` and the
    status line prints `coverage: unknown (store unreadable)` — the leg still seeds.
    """
    try:
        from app import kg_store
        return {str(r["name"]) for r in kg_store.store()._query("SELECT name FROM entities")}
    except Exception:  # noqa: BLE001
        return None


def index_status() -> Optional[dict]:
    """What the semantic leg is READING right now: rows, age, live coverage, provenance.

    `None` before the first load, so a caller cannot mistake "no index" for "fresh
    index" — the two are different answers and #3907's artifact made them
    indistinguishable. Populated by `_load_index` on a COLD load only (one line per
    process per artifact, not one per `vault_recall`), and refreshed when the
    artifact's mtime changes, which is how a rebuild shows up without a restart.

    Keys: `dir`, `rows`, `dim`, `age_days`, `built_at`, `max_age_days`, `stale`,
    `live_entities`,
    `covered_entities`, `coverage_pct`, `names_not_live`, `build_json` (the dict the
    artifact ships, or `None` when the file is absent — which is #3907's finding: the
    builder has written `build.json` since its only commit `dbfde750`, so an artifact
    without one has no recoverable record of which store it came from).

    Nothing here consults `MAX_INDEX_AGE_DAYS` to change behaviour: `stale` is the
    reported comparison against the bound, and no caller may use it to drop a seed.
    """
    return _index_status


def artifact_status(d, *, names: Optional[list] = None,
                    mtime: Optional[float] = None) -> Optional[dict]:
    """Everything knowable about the artifact in `d`, from the bytes and the store.

    `None` when there is no artifact to report on (either file missing) — never a dict
    of zeroes, because a directory that exists with nothing in it must not read as a
    fresh-but-empty index. The builder's `--status` calls THIS function rather than
    reimplementing the arithmetic, so the number a human reads in a terminal and the
    number the loader logs are computed in one place and cannot drift apart.

    `names`/`mtime` may be handed in by a caller that already read them (the loader
    has both), so reporting never costs a second parse of `names.json`.
    """
    import json as _json
    import time
    d = Path(d)
    names_p, mat_p = d / "names.json", d / "vectors.npy"
    if not names_p.exists() or not mat_p.exists():
        return None
    if names is None:
        names = _json.loads(names_p.read_text())
    if mtime is None:
        mtime = mat_p.stat().st_mtime
    age_days = max(0.0, (time.time() - mtime) / 86400.0)
    st: dict = {
        "dir": str(d), "rows": len(names), "dim": None,
        "age_days": round(age_days, 2), "max_age_days": MAX_INDEX_AGE_DAYS,
        "stale": age_days > MAX_INDEX_AGE_DAYS,
        "live_entities": None, "covered_entities": None, "coverage_pct": None,
        "names_not_live": None, "build_json": None,
    }
    try:
        import numpy as np
        st["dim"] = int(np.load(mat_p, mmap_mode="r").shape[1])
    except Exception:  # noqa: BLE001
        pass
    try:
        st["build_json"] = _json.loads((d / "build.json").read_text())
    except Exception:  # noqa: BLE001
        st["build_json"] = None
    live = _live_entity_names()
    if live:
        idx = {str(n) for n in names}
        covered = idx & live
        st["live_entities"] = len(live)
        st["covered_entities"] = len(covered)
        st["coverage_pct"] = round(100.0 * len(covered) / len(live), 1)
        st["names_not_live"] = len(idx - live)
    return st


def status_line(st: dict) -> str:
    """The one-line rendering both the loader's log and `--status` print from."""
    prov = ("build.json built_at=%s model=%s" % (
        st["build_json"].get("built_at"), st["build_json"].get("model"))
        if st["build_json"] else "build.json ABSENT (provenance unrecoverable)")
    cov = (f"coverage {st['coverage_pct']}% ({st['covered_entities']} of "
           f"{st['live_entities']} live entities, {st['names_not_live']} index names "
           f"not live)" if st["coverage_pct"] is not None
           else "coverage unknown (store unreadable)")
    return (f"{st['rows']} rows x {st['dim']}, {st['age_days']:.1f} days old "
            f"(bound {st['max_age_days']}), {cov}, {prov}"
            f"{' — STALE' if st['stale'] else ''}")


def _report_index(d: Path, mtime: float, names: list) -> dict:
    """Compute the status and say it out loud, once per load. Age from the bytes.

    The log line is the fix for the actual defect: `app/entity_linker` read
    `mat_p.stat().st_mtime` only to invalidate its cache, so a 42-day-old artifact
    logged nothing and served silently forever. Reported through the module logger at
    INFO, on the same channel every other retrieval fact is published on.
    """
    import logging
    st = artifact_status(d, names=names, mtime=mtime)
    if st is None:
        # The artifact went away between the existence check above and here. No report
        # is printed and `index_status()` stays honest about having none; the loaded
        # matrix in `_index` still serves the query, which is the whole no-fail-closed
        # rule. A later load retries the report.
        return None
    logging.getLogger(__name__).info("entity-vectors index loaded: %s", status_line(st))
    return st



def _load_index():
    import numpy as np
    d = vectors_dir()
    names_p, mat_p = d / "names.json", d / "vectors.npy"
    if not names_p.exists() or not mat_p.exists():
        return None
    mtime = mat_p.stat().st_mtime
    global _index, _index_status
    with _lock:
        if _index is not None and _index[0] == d and _index[1] == mtime:
            return _index
        names = json.loads(names_p.read_text())
        mat = np.load(mat_p, mmap_mode="r")
        if len(names) != mat.shape[0]:
            return None
        _index = (d, mtime, names, np.asarray(mat, dtype=np.float32))
        # AFTER the cache is set, and inside the lock, so exactly one load reports even
        # when several threads race the first query. A rebuild changes the mtime, which
        # is the same fact the cache invalidates on, so the line is per artifact rather
        # than per process — the fresh index is announced, not assumed.
        _index_status = _report_index(d, mtime, names)
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
    """Drop the loaded index AND its reported status.

    Both halves go together on purpose: `index_status()` reading the previous
    artifact's figures after a rebuild was pointed at a different directory would be
    the same misreading #3907 is about, one cache-invalidation bug instead of one
    missing bound. `None` means "this process has not read an index", which is a
    different answer from "the index is fresh".
    """
    global _index, _index_status
    with _lock:
        _index = None
        _index_status = None
