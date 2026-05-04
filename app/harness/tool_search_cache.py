"""Session-scoped cache for ``LoadedToolSet`` instances.

Lloyd's frontend opens a fresh ``run_query`` for every user message, but
conceptually one *session* spans many turns. We want the model to discover
a tool once via ToolSearch and have it stay loaded for the rest of the
conversation — otherwise a long thread re-pays the search round-trip every
turn.

This module is a process-wide ``session_id → LoadedToolSet`` map. The cache
is invalidated automatically when the catalog signature changes (e.g. config
reload added/removed tools), in which case the loaded set is dropped and
the model starts fresh.

Subagents (Task tool) get their own keyed entries — different
``disallowed_tools`` profiles must not share state.
"""

from __future__ import annotations

import asyncio
from typing import Iterable

from app.harness.tool_search import LoadedToolSet, catalog_signature

_CACHE: dict[str, LoadedToolSet] = {}
_LOCK = asyncio.Lock()


async def get_or_create(
    session_id: str,
    *,
    catalog: list[dict],
    baseline: Iterable[str],
    enabled: bool,
) -> LoadedToolSet:
    """Return the session's LoadedToolSet, creating or invalidating as needed.

    Invalidation rule: if a cached entry exists but its ``catalog_signature``
    differs from the new catalog, the loaded set is dropped and a fresh
    LoadedToolSet replaces it. The model has to re-discover tools — accepted
    cost for the rare case of a config-time catalog change mid-session.

    A blank ``session_id`` is treated as un-cacheable (returns a fresh
    LoadedToolSet without touching the cache). This keeps callers without
    session correlation safe — they just don't get persistence across calls.
    """
    sig = catalog_signature(catalog)
    baseline_set = set(baseline)

    if not session_id:
        return LoadedToolSet(
            catalog=catalog,
            baseline=baseline_set,
            enabled=enabled,
            catalog_signature=sig,
        )

    async with _LOCK:
        cached = _CACHE.get(session_id)
        if cached is None or cached.catalog_signature != sig:
            cached = LoadedToolSet(
                catalog=catalog,
                baseline=baseline_set,
                enabled=enabled,
                catalog_signature=sig,
            )
            _CACHE[session_id] = cached
        else:
            # Same catalog — refresh the live references in case enabled
            # toggled or baseline expanded between calls (e.g. user edited
            # config). The accumulated ``loaded`` set is preserved.
            cached.catalog = catalog
            cached.baseline = baseline_set
            cached.enabled = enabled
        return cached


async def drop(session_id: str) -> None:
    """Forget a session's loaded set. Called on session deletion."""
    async with _LOCK:
        _CACHE.pop(session_id, None)


async def clear() -> None:
    """Drop everything. Test helper."""
    async with _LOCK:
        _CACHE.clear()
