#!/usr/bin/env python3
"""
agent_mcp/memory.py — backward-compat shim (Task #340 PR 5).

The original 1,684-line module has been split into:

    agent_mcp/facts.py    — fact_* + relationships + graph (10 tools)
    agent_mcp/vault.py    — vault_* + qmd hybrid search    (5 tools)
    agent_mcp/session.py  — memory_* + session_recall      (5 tools)

This file remains as a thin re-export so external callers (prefetch.py,
app/post_capture.py, the characterization tests) can still import names
from agent_mcp.memory without rewriting their imports. New code SHOULD
import from agent_mcp.facts / .vault / .session directly.

main.py no longer registers this module — it dispatches to facts, vault,
and session individually.
"""

# ── Re-exports from agent_mcp._shared ────────────────────────────────────────
from agent_mcp._shared import (  # noqa: F401
    VAULT,
    FACTS_ROOT,
    ALIASES_PATH,
    _ENTITY_STOPWORDS,
    _SCORING_STOPWORDS,
    _QUERY_STOPWORDS,
    _token_overlap,
    _levenshtein,
    _fuzzy_entity_match,
    _parse_fact_frontmatter,
    _write_fact_frontmatter,
    _find_entity_dir,
    _load_aliases,
    _save_aliases,
    _get_entity_dirs_cached,
    _invalidate_entity_dirs_cache,
    _resolve_entity,
)

# ── Re-exports from agent_mcp.facts ──────────────────────────────────────────
from agent_mcp.facts import (  # noqa: F401
    # tool handlers
    _fact_get,
    _fact_add,
    _fact_profile,
    _fact_check,
    _fact_resolve,
    _fact_invalidate,
    _fact_relate,
    _fact_relationships,
    _fact_path,
    _fact_neighbors,
    # internal helpers
    _generate_fact_id,
    _get_facts_sync,
    _detect_contradictions_sync,
    _get_entity_edge_counts,
    _extract_entities_from_query,
    _load_relationships,
    _save_relationships,
    _invalidate_relationships_cache,
    _graph_expand_entities,
    _graph_weighted_neighbors,
    _fact_query_tokens,
    _fact_blob,
    _fact_matches_tokens,
    _fact_score,
    # constants
    EDGE_TYPE_WEIGHTS,
    RELATIONSHIPS_PATH,
    FACT_GODNODE_THRESHOLD,
    FACT_RANK_CAP_SEED,
    FACT_RANK_CAP_GRAPH,
    _OPPOSING_PAIRS,
    _TASK_ID_RE,
    _FACT_QUERY_STOPWORDS,
)

# ── Re-exports from agent_mcp.vault ──────────────────────────────────────────
from agent_mcp.vault import (  # noqa: F401
    # tool handlers
    _vault_get,
    _vault_write,
    _vault_overview,
    _vault_search,
    _vault_recall,
    # internal helpers
    _qmd_sanitize,
    _qmd_strip_stopwords,
    _qmd_log,
    _qmd_post,
    _qmd_daemon_search,
    _qmd_subprocess_search,
    _consolidate_results,
    _rrf_fuse,
    _run_vault_search,
    _resolve_case_insensitive,
    _audit_write,
    # constants
    AUDIT_LOG_DIR,
    AUDIT_LOG_FILE,
    QMD_BIN,
    QMD_DAEMON_URL,
    VAULT_SEGMENTS,
    VAULT_EXCLUDE_DIRS,
    VAULT_EXCLUDE_FILES,
    CONSOLIDATION_ENDPOINT,
    CONSOLIDATION_MODEL,
    CONSOLIDATION_MIN_RESULTS,
    CONSOLIDATION_TIMEOUT,
    CONSOLIDATION_SYSTEM_PROMPT,
)

# ── Re-exports from agent_mcp.session ────────────────────────────────────────
from agent_mcp.session import (  # noqa: F401
    # tool handlers
    _memory_read,
    _memory_add,
    _memory_replace,
    _memory_remove,
    _session_recall,
    # internal helpers
    _check_injection,
    _extract_msg_text,
    _load_session_index,
    _score_session,
    # constants
    MEMORIES_ROOT,
    MEMORY_FILES,
    _INJECTION_PATTERNS,
    SESSIONS_DIR,
)

# ── Aggregated list_tools (for backward-compat tests) ────────────────────────
# main.py no longer routes through here; this exists so existing callers
# that do `await memory.list_tools()` continue to see the full 20-tool set.
from agent_mcp import facts as _facts_mod
from agent_mcp import vault as _vault_mod
from agent_mcp import session as _session_mod


async def list_tools():
    """Aggregated tool list across facts/vault/session for backward compat."""
    out = []
    for mod in (_facts_mod, _vault_mod, _session_mod):
        out.extend(await mod.list_tools())
    return out
