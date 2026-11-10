"""Embedding fusion for the chat skill matcher (2026-11-09 skills review, finding 4).

The lexical scorer (`agent_mcp.skills._score_skill`) weights a skill's directory
name x3, so retrieval quality had come to depend on naming: two skills were renamed
in 2026-09 purely to win the token overlap. This leg adds what the name cannot carry,
a cosine between the user's message and each skill's `name: description`, embedded by
`app.qwen3_embed` (Qwen3-Embedding-0.6B on CPU), and uses it to RE-ORDER the lexical
offers — never to admit one:

  * the pool is the lexical top-`k` offers, so a skill with no name/description/tag
    overlap (#311's failure) cannot enter, and the expected-empty control cannot move;
  * scores stay lexical, so `SKILL_THRESHOLD_FIRST` / `_SECOND`, the rendered
    `score=` and the telemetry rows mean what they meant; only the order changes,
    which is what decides the one skill whose full body is injected.

Measured on `eval/skill_match_queries.yaml` (`eval/run_skill_match_eval.py
--embed-fusion`, 51 labelled + 24 expected-empty turns, 208 live skills):
weight 2 lifts hit@1 0.510 -> 0.686 and recall@5 0.677 -> 0.755, expected-empty named
15/24 both; the gain survives three instruction wordings and a split-half check
(weight chosen on one half, +0.16 / +0.19 hit@1 on the held-out half). Recorded in
`eval/measurements/skill-embed-fusion-2026-11-09.md`. This is not the 2026-09-14
"geometric" arm, which replaced the lexical score with similarity at a 0.62 floor.

Costs, and how each is bounded:
  * ~150 ms CPU per query: one dedicated thread, a `wait_ms` deadline inside the
    skills leg (under the 300 ms prefetch budget); late means lexical order, never a
    dropped skills leg;
  * ~2.5 GB RSS for fp32 weights, loaded lazily on the first chat turn (that turn
    and any turn before the skill vectors exist answer in lexical order);
  * chat sessions only — worker prompts are boilerplate the evidence never covered,
    and they are 97% of prefetch traffic.

`skills.embed_fusion` in config.yaml: `enabled`, `weight` (2.0), `k` (16),
`wait_ms` (200). Every failure is lexical order.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

logger = logging.getLogger(__name__)

DEFAULTS = {"enabled": False, "weight": 2.0, "k": 16, "wait_ms": 200}
INSTRUCT = ("Instruct: Given a user's request to an assistant, retrieve the skill "
            "whose procedure handles it\nQuery: ")
#: Intra-op threads for the embedder. torch's default is every core, which on this
#: box would let one chat turn's 150 ms embed contend with gate xdist workers.
TORCH_THREADS = 4

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skill-embed")
_vec_lock = threading.Lock()
_vectors: dict[str, tuple[str, object]] = {}     # name -> (text hash, unit vector)
_building = threading.Event()
#: After a failed build (no model on disk, say), no retry before this monotonic
#: instant — otherwise every chat turn would retry and log the same warning.
RETRY_AFTER_FAILURE_S = 3600.0
_retry_after = 0.0


def settings() -> dict | None:
    """`skills.embed_fusion` merged over DEFAULTS, or None when off or unreadable."""
    try:
        from app.config import CONFIG

        cfg = (CONFIG.get("skills") or {}).get("embed_fusion") or {}
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
        return None
    out = dict(DEFAULTS, enabled=True)
    for key, cast in (("weight", float), ("k", int), ("wait_ms", int)):
        try:
            if key in cfg:
                out[key] = cast(cfg[key])
        except (TypeError, ValueError):
            pass
    return out


def embed_text(skill: dict) -> str:
    """What a skill is embedded as: its index line (name, description)."""
    desc = " ".join(str(skill.get("description") or "").split())
    return f"{str(skill.get('name') or '').replace('-', ' ')}: {desc}"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _embed(text: str):
    import numpy as np
    import torch

    if torch.get_num_threads() > TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)
    from app.qwen3_embed import embed_query

    return np.asarray(embed_query(text), dtype="float32")


def _cache_path():
    from app.paths import SKILL_EMBED_CACHE_PATH

    return SKILL_EMBED_CACHE_PATH


def _build_vectors(skills: list[dict]) -> None:
    """Background: fill `_vectors` for every skill whose text changed, persist."""
    import numpy as np

    try:
        try:
            disk = json.loads(_cache_path().read_text())
        except (OSError, ValueError):
            disk = {}
        dirty = False
        for sk in skills:
            text = embed_text(sk)
            h = _hash(text)
            name = sk.get("name") or ""
            with _vec_lock:
                have = _vectors.get(name)
            if have and have[0] == h:
                continue
            if h not in disk:
                disk[h] = [round(float(x), 6) for x in _embed(text)]
                dirty = True
            with _vec_lock:
                _vectors[name] = (h, np.asarray(disk[h], dtype="float32"))
        if dirty:
            path = _cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            live = {_hash(embed_text(sk)) for sk in skills}
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({h: v for h, v in disk.items() if h in live}))
            os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 — a missing vector is lexical order
        global _retry_after
        import time

        _retry_after = time.monotonic() + RETRY_AFTER_FAILURE_S
        logger.warning("skill embed: vector build failed (%s); lexical order for the "
                       "next %d s", exc, int(RETRY_AFTER_FAILURE_S))
    finally:
        _building.clear()


def _vectors_for(skills: list[dict]) -> dict | None:
    """`{name: vector}` when every skill's vector is current; else start a build."""
    out = {}
    stale = False
    with _vec_lock:
        for sk in skills:
            have = _vectors.get(sk.get("name") or "")
            if not have or have[0] != _hash(embed_text(sk)):
                stale = True
                break
            out[sk["name"]] = have[1]
    if not stale:
        return out
    import time

    if not _building.is_set() and time.monotonic() >= _retry_after:
        _building.set()
        _executor.submit(_build_vectors, list(skills))
    return None


def rerank(text: str, scored: list[tuple[float, dict]], skills: list[dict],
           cfg: dict | None = None) -> list[tuple[float, dict]]:
    """`scored` with its top `k` re-ordered by lexical + weight x 10 x cosine.

    Scores are returned unchanged (lexical); only the order of the top `k` moves,
    and the tail after it keeps its place. Any failure, a missing vector or a late
    embed returns `scored` as given.
    """
    cfg = cfg or settings()
    if not cfg or len(scored) < 2:
        return scored
    try:
        vecs = _vectors_for(skills)
        if vecs is None:
            return scored
        fut = _executor.submit(_embed, INSTRUCT + text[:1500])
        q = fut.result(timeout=cfg["wait_ms"] / 1000.0)
    except FutureTimeout:
        logger.debug("skill embed: query embed missed %d ms; lexical order", cfg["wait_ms"])
        return scored
    except Exception as exc:  # noqa: BLE001
        logger.warning("skill embed: rerank failed (%s); lexical order", exc)
        return scored
    k = max(2, int(cfg["k"]))
    head, tail = scored[:k], scored[k:]
    w = float(cfg["weight"]) * 10.0

    def fused(pair):
        vec = vecs.get(pair[1].get("name") or "")
        return pair[0] + (w * float(vec @ q) if vec is not None else 0.0)

    return sorted(head, key=fused, reverse=True) + tail


def _reset_for_tests() -> None:
    global _retry_after
    _retry_after = 0.0
    with _vec_lock:
        _vectors.clear()
    _building.clear()
