"""Work sources — each registers into SOURCE_REGISTRY.

A source is a module exposing:
  - NAME: str                   — unique source identifier
  - DEFAULT_PRIORITY: int       — 0..100 (lower = sooner)
  - async enqueue_if_due(queue, src_cfg) -> None
  - async execute(item: QueueItem) -> dict
      dict may contain: summary, artifact_path, response, task_id
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import CONFIG

logger = logging.getLogger("lloyd-workers.sources")


SOURCE_REGISTRY: dict[str, Any] = {}


def register(source) -> None:
    name = getattr(source, "NAME", None)
    if not name:
        raise ValueError("Source must define NAME")
    SOURCE_REGISTRY[name] = source
    logger.info("Registered work source: %s", name)


def get_sources_config() -> dict[str, dict]:
    """Re-read per-source config from config.yaml each call.

    CONFIG is mutated in place by /api/config/save — we want live updates.
    """
    return CONFIG.get("workers", {}).get("sources", {}) or {}


# Import and register all sources. Each module calls register() at import-time.
# KG pipeline steps (data-pipeline, conversation-relation-linking,
# entity-resolution-sweep) run as regular scheduled-task entries in
# ~/obsidian/autonomy/, chained via depends_on — no dedicated source.
from workers.sources import scheduled_task as _scheduled_task  # noqa: E402,F401
from workers.sources import autoresearch as _autoresearch  # noqa: E402,F401
from workers.sources import selfmod_regression as _selfmod_regression  # noqa: E402,F401
from workers.sources import backlog_selfmod as _backlog_selfmod  # noqa: E402,F401
from workers.sources import gap_fill as _gap_fill  # noqa: E402,F401
from workers.sources import session_distill as _session_distill  # noqa: E402,F401
from workers.sources import domain_research as _domain_research  # noqa: E402,F401
from workers.sources import bench_mine as _bench_mine  # noqa: E402,F401

register(_scheduled_task)
register(_autoresearch)
register(_selfmod_regression)
register(_backlog_selfmod)
register(_gap_fill)
register(_session_distill)
register(_domain_research)
register(_bench_mine)
