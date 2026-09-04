"""Shared helpers for agent_mcp knowledge-graph and vault servers.

Extracted from agent_mcp/memory.py as part of Task #340 PR 1. No behavior
change — pure module split for maintainability.

Contents:
    - Path constants (VAULT, FACTS_ROOT, ALIASES_PATH)
    - Stopword sets (_ENTITY_STOPWORDS, _SCORING_STOPWORDS, _QUERY_STOPWORDS,
      _FACT_QUERY_STOPWORDS, _SKILLS_QUERY_STOPWORDS)
    - Pure helpers (_token_overlap, _levenshtein, _fuzzy_entity_match)
    - Fact frontmatter helpers (_parse_fact_frontmatter, _write_fact_frontmatter)
    - Entity resolution (_find_entity_dir, _load_aliases, _save_aliases,
      _get_entity_dirs_cached, _resolve_entity) — aliases live in
      app.kg_store since the 2026-09 migration; the two helpers are
      store-backed shims kept for their callers' shape.
    - Cache invalidation (_invalidate_entity_dirs_cache)

Anything that touches the relationships index, qmd daemon, vault audit log,
session memory injection patterns, or fact ranking stays in its owning module
(facts/vault/session after PR 5).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import yaml
from mcp.types import TextContent

# ── Path constants ───────────────────────────────────────────────────────────

from app.paths import (
    VAULT_ROOT as VAULT,
    VAULT_FACTS_ROOT as FACTS_ROOT,
    VAULT_FACTS_ALIASES as ALIASES_PATH,
)
from app.atomic_io import atomic_write_text  # noqa: F401  (re-export for agent_mcp modules)


# ── Result types & dispatch helpers (#340 PR 4) ──────────────────────────────
#
# All handlers in agent_mcp.facts / .vault / .session return a Python dict.
# The per-module call_tool dispatcher wraps each return in TextContent via
# _wrap() — json.dumps happens exactly once per call, not 3+ times per
# handler.
#
# Shape policy:
#   - Success: handler-specific data, NOT wrapped in an envelope. The agent
#     reads tool output as text; an extra {"ok": true, "data": {...}} layer
#     adds visual noise on every successful call. Pre-existing success
#     shapes ({"facts": [...]}, {"path": ..., "text": ...}, etc.) are
#     preserved verbatim.
#
#   - Error: standardized via _err() — always {"error": str, "code": str,
#     ...extra}. Extra keys preserve pre-existing companion fields like
#     {"facts": []} that callers may expect on the error path.
#
# Why no Result envelope:
#   The original audit (#340 background) proposed
#   {"ok": bool, "data": ..., "error": ..., "code": ...}. PR 4's callsite
#   audit found zero programmatic consumers (prefetch.py uses internal
#   helpers below the json layer; post_capture.py discards the return).
#   The only "consumer" is the LLM agent reading the JSON visually, where
#   verbosity is a tax. We get the type-safety, single-dumps, and code-
#   field wins from this approach without taxing every successful call.

class Result(TypedDict, total=False):
    """Type contract for MCP handler returns.

    All handlers in facts.py / vault.py / session.py return Result dicts
    (Python dicts). The dispatcher serializes via _wrap() exactly once.

    Two valid shapes:
      Success: arbitrary handler-specific keys (no envelope).
      Error:   {"error": str, "code": str, ...extra}

    Use _err() to construct errors so the shape stays consistent.
    """
    error: str
    code: str


class ErrorCode:
    """Standardized error-code constants for handler returns.

    Use these instead of inline strings so callers can branch on the code
    without parsing the human-readable error message.
    """
    MISSING_PARAM = "MISSING_PARAM"        # Required parameter omitted/empty
    INVALID_PARAM = "INVALID_PARAM"        # Parameter present but malformed
    NOT_FOUND = "NOT_FOUND"                # Entity/file/resource doesn't exist
    PATH_ESCAPE = "PATH_ESCAPE"            # Path traversal outside vault
    INJECTION = "INJECTION"                # Prompt-injection guardrail tripped
    NO_MATCH = "NO_MATCH"                  # Substring not found in target
    INTERNAL = "INTERNAL"                  # Caught exception during handler
    UNKNOWN_TOOL = "UNKNOWN_TOOL"          # Dispatcher could not route name


def _err(message: str, code: str = ErrorCode.INTERNAL, **extra: Any) -> dict:
    """Construct a standardized error result.

    Example:
        return _err("entity is required", ErrorCode.MISSING_PARAM, facts=[])

    The order is fixed: error first, code second, then any caller-supplied
    extras (e.g. empty list/dict companions that pre-existing callers may
    expect on the error path).

    Messages are capped at 500 chars: exception text is deliberately shown
    to the model (it enables self-correction) but many handlers pass raw
    str(exc), and a deep traceback or bad-input echo shouldn't eat context
    budget. Intentional error messages are far shorter than the cap.
    """
    if len(message) > 500:
        message = message[:500] + " …[truncated]"
    return {"error": message, "code": code, **extra}


def _wrap(result: dict) -> list[TextContent]:
    """Serialize a handler return to a TextContent envelope.

    Exactly one json.dumps per tool call. default=str handles datetimes
    and other non-JSON-native types that may slip through (e.g. event_date
    values from YAML-parsed fact frontmatter).
    """
    return [TextContent(type="text", text=json.dumps(result, default=str))]


# ── Session binding & HTTP clients ───────────────────────────────────────────

def get_bound_session() -> str:
    """Session id the aggregator bound for the current tool call ('' if none).

    main.py strips `_session_id` from tool arguments and sets it in the
    _task_registry contextvar; every session-aware tool should read it
    through this helper rather than touching the contextvar directly.
    """
    from agent_mcp import _task_registry
    return _task_registry.current_session_id.get()


def make_http_client(timeout: float | None = 10.0, **kwargs) -> "httpx.AsyncClient":
    """Standard async HTTP client for agent_mcp modules.

    One construction point so cross-cutting concerns (proxies, retries,
    default headers) have a single home when they arrive.
    """
    import httpx
    return httpx.AsyncClient(timeout=timeout, **kwargs)


def make_sync_http_client(timeout: float | None = 10.0, **kwargs) -> "httpx.Client":
    """Sync twin of make_http_client for non-async call paths."""
    import httpx
    return httpx.Client(timeout=timeout, **kwargs)


# ── Resilient frontmatter parsing ────────────────────────────────────────────
#
# The agent writes its own task/backlog frontmatter, so malformed YAML is a
# routine input, not an edge case. A hard-drop on parse failure dormant-killed
# 34/40 autonomy tasks on 2026-05-28 (see project_autonomy_silent_task_drop).
# Every reader of agent-written frontmatter must degrade gracefully; this is
# the one parser that does.

def _fold_orphaned_tag_items(fm_text: str) -> str:
    """Repair the recurring `tags:` corruption: a block→inline replacement of
    the tags field that orphans the pre-existing block-list items, e.g.
        tags: [38-foo, autonomy, pipeline]
        - nightly
        - reflection
    Folds the orphan `- item` lines back into the inline list."""
    lines = fm_text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)tags:\s*\[(.*)\]\s*$", lines[i])
        if m:
            indent, inside = m.group(1), m.group(2)
            items = [x.strip().strip("'\"") for x in inside.split(",") if x.strip()]
            j = i + 1
            while j < len(lines) and re.match(r"^\s*-\s+\S", lines[j]):
                it = lines[j].strip()[1:].strip().strip("'\"").strip("[]").strip().strip("'\"")
                if it and it not in items:
                    items.append(it)
                j += 1
            if j > i + 1:  # there were orphan items → repair
                out.append(f"{indent}tags: [{', '.join(items)}]")
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def parse_frontmatter_text(
    fm_text: str,
    *,
    fallback_fields: tuple[str, ...] = (),
    log_label: str = "frontmatter",
) -> dict:
    """Parse the YAML between frontmatter fences with graduated recovery.

    1. Plain yaml.safe_load.
    2. Retry after repairing the known orphaned-tags corruption.
    3. Regex-extract `fallback_fields` line by line, marking the result with
       `_yaml_broken: True` so writers know the file needs normalizing.

    Never raises and never returns None — a record may come back degraded,
    but it can't silently vanish from a listing or the scheduler.
    """
    logger = logging.getLogger("agent_mcp.shared")
    first_err: Exception | None = None
    try:
        fm = yaml.safe_load(fm_text)
        if isinstance(fm, dict):
            return fm
    except Exception as e:
        first_err = e

    repaired = _fold_orphaned_tag_items(fm_text)
    if repaired != fm_text:
        try:
            fm = yaml.safe_load(repaired)
            if isinstance(fm, dict):
                logger.warning("[%s] recovered corrupted frontmatter (%s)", log_label, first_err)
                return fm
        except Exception:
            pass

    logger.warning(
        "[%s] YAML frontmatter parse failed (%s); falling back to regex extraction",
        log_label, first_err or "frontmatter is not a mapping",
    )
    fm = {"_yaml_broken": True}
    for field in fallback_fields:
        m = re.search(rf"^{re.escape(field)}:\s*(.+)$", fm_text, re.MULTILINE)
        if not m:
            continue
        raw = m.group(1).strip()
        try:
            # Recover scalar types ("38" → int, "[a, b]" → list) where the
            # individual line is still valid YAML.
            fm[field] = yaml.safe_load(raw)
        except Exception:
            fm[field] = raw
    return fm


# ── Stopword sets ────────────────────────────────────────────────────────────

_ENTITY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "how", "when", "where", "why",
}

# Scoring stopwords — broader than _ENTITY_STOPWORDS. "task" and "backlog" as
# bare tokens are noise for the scorer (the task-ID regex handles numeric
# references explicitly).
_SCORING_STOPWORDS = _ENTITY_STOPWORDS | {
    "task", "backlog", "item", "issue", "ticket",
}

# Query-phrasing stopwords (#327) — superset of _ENTITY_STOPWORDS used ONLY
# by qmd lex cleanup (FTS5 BM25 leg). DO NOT use for entity scoring, fact
# ranking, or anywhere a short query token could be a legitimate match target.
_QUERY_STOPWORDS = _ENTITY_STOPWORDS | {
    # Conversational pronouns / self-reference
    "me", "us", "our", "ours", "my", "mine", "your", "yours",
    "myself", "yourself", "ourselves", "them", "their", "theirs",
    # Question-framing verbs (content-empty when used as prefix/filler)
    "walk", "tell", "show", "describe", "explain", "let", "lets",
    "give", "ask", "share", "summarize",
    # Third-person-s question framing verbs (#332). Restricted to -s forms
    # so root-verb uses in content queries remain searchable.
    "happens", "sends", "occurs", "goes",
    # Discourse fillers / intensifiers
    "just", "also", "really", "actually", "basically", "pretty",
    "please", "kindly", "maybe", "probably", "perhaps",
    # NOTE deliberately NOT stripped — can be meaningful in architectural
    # content: enable, handle, return, work, function, operate, build,
    # ship, shipped, run, start, stop, fail, failed.
}

# Fact-side query stopwords (#322 fact ranking). Used by
# agent_mcp.facts._fact_query_tokens to keep fact ranking responsive to
# query-token overlap without aggressive trimming. Lightweight on purpose:
# aggressive stopword removal hurts when the query is short
# ("how does fact_path work"). Kept as a frozenset because it's hot-path.
_FACT_QUERY_STOPWORDS = frozenset({
    "what", "how", "when", "where", "why", "who", "which",
    "the", "and", "for", "with", "from", "into", "over", "under",
    "does", "did", "do", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "can", "could", "will", "would", "should", "may",
    "might", "must", "shall",
    "this", "that", "these", "those", "it", "its", "them", "they", "their",
    "our", "ours", "my", "mine", "your", "yours", "his", "her", "hers",
    "we", "you", "us", "me", "i",
    "about", "also", "just", "than", "then", "there", "here", "through",
    "during", "between", "among", "across", "within", "without", "after",
    "before", "while", "until", "since",
    "of", "in", "on", "to", "at", "by", "as", "or", "nor", "not", "so",
    "if", "else", "but", "yet", "though", "although",
    "tell", "show", "explain", "describe", "walk", "give", "get", "use",
    "using", "used", "make", "made", "see", "seen", "know", "need",
})

# Skill-matching query stopwords. Used by agent_mcp.skills._query_tokens
# to filter conversational fillers from the *query* side of skill scoring
# (skill body side is unfiltered — the asymmetry is deliberate). Per
# the original docstring in skills.py: a query like "lets dig into 311"
# should NOT fire skills just because "lets", "dig", "into" appear in
# arbitrary skill bodies.
#
# Broader than _QUERY_STOPWORDS — includes many more conversational
# tokens (ok, yeah, sure, getting, looked, made, etc.). Distinct purpose,
# distinct shape. Kept as a separate constant rather than aliased.
_SKILLS_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "his", "her",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "between",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "just", "also", "very", "really", "quite", "too", "much",
    "ok", "okay", "yeah", "yes", "nah", "sure", "right",
    "lets", "let", "go", "going", "get", "got", "getting",
    "want", "wants", "wanted", "know", "knows", "knew",
    "think", "thinks", "thought", "look", "looking", "looked",
    "take", "takes", "took", "make", "makes", "made",
    "now", "some", "any", "all", "each", "every", "both",
    "up", "out", "over", "down", "off", "away",
    "here", "there", "thing", "things", "stuff",
    "left", "done", "next", "back", "ready", "still", "already",
    "tell", "show", "give", "put", "run", "running", "ran",
    "come", "came", "see", "saw", "seen", "say", "said",
    "try", "tried", "use", "used", "using",
    "start", "started", "stop", "stopped", "keep", "kept",
    "set", "well", "good",
    "bit", "lot", "way", "something", "anything", "everything",
    "like", "first", "last", "new", "old", "one", "two",
    "dig", "really",
}


# ── Pure helpers ─────────────────────────────────────────────────────────────

def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of word tokens, case-insensitive."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _levenshtein(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _fuzzy_entity_match(name: str, candidates: list[str], threshold: float = 0.85) -> Optional[str]:
    """Fuzzy match an entity name against known entities.

    History: previously used threshold=0.7 with a substring-match boost to 0.8,
    which poisoned the alias table with entries like
      'agent prompt constraint' -> 'agent'
      'autonomy-system.md'      -> 'System'
    because any short canonical name containing a token of the query got the
    substring boost. See backlog #310 Tier 4.

    Changes:
    - Threshold bumped 0.7 → 0.85 (tight similarity required).
    - Substring boost removed entirely — Levenshtein + token overlap drive
      the match.
    - Additional guard: block matches where the length ratio is extreme
      (>2×), which reliably indicates a short canonical swallowing a long
      specific name.
    """
    name_lower = name.lower().strip()
    best_match: Optional[str] = None
    best_score = 0.0
    for candidate in candidates:
        cand_lower = candidate.lower().strip()
        if name_lower == cand_lower:
            return candidate
        # Block extreme length asymmetry — "agent prompt constraint" shouldn't
        # match "agent".
        if name_lower and cand_lower:
            len_ratio = max(len(name_lower), len(cand_lower)) / min(
                len(name_lower), len(cand_lower)
            )
            if len_ratio > 2.0:
                continue
        overlap = _token_overlap(name_lower, cand_lower)
        max_len = max(len(name_lower), len(cand_lower))
        lev_score = 1.0 - (_levenshtein(name_lower, cand_lower) / max_len) if max_len > 0 else 0.0
        combined = 0.4 * overlap + 0.6 * lev_score
        if combined > best_score:
            best_score = combined
            best_match = candidate
    return best_match if best_score >= threshold else None


# ── Fact frontmatter helpers ────────────────────────────────────────────────

def _parse_fact_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from the head of a markdown string.

    Returns {} for any malformed input (no leading ---, no closing ---,
    or YAML parse error). Logs YAML errors so bad files are visible without
    crashing the caller — a single corrupt fact file should not kill an
    entity-wide read. See backlog #343/#348 for the bug class this protects
    against.
    """
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    try:
        return yaml.safe_load(content[3:end]) or {}
    except yaml.YAMLError as e:
        # Fail-soft: log and skip. Caller gets an empty dict (treated as
        # "no facts in this file") rather than an exception that aborts
        # the whole entity read.
        logging.getLogger(__name__).warning(
            "fact frontmatter YAML parse failed; skipping file: %s", e
        )
        return {}


def _write_fact_frontmatter(data: dict) -> str:
    """Emit YAML frontmatter block with leading and trailing --- markers."""
    return f"---\n{yaml.dump(data, default_flow_style=False, sort_keys=False)}---\n"


# ── Entity directory cache ──────────────────────────────────────────────────

_entity_dirs_cache: Optional[tuple[float, list[str]]] = None
# Lowercased name → on-disk name, rebuilt whenever _entity_dirs_cache is.
_entity_dirs_index: Optional[dict[str, str]] = None
# st_mtime_ns of FACTS_ROOT at the last real scan; lets TTL expiry skip the
# 64k-entry rescan when no entity dir was added or removed.
_entity_dirs_mtime: Optional[int] = None
_ENTITY_DIRS_TTL = 60


def _get_entity_dirs_cached() -> list[str]:
    """List entity directory names under FACTS_ROOT, cached for 60s.

    On TTL expiry the directory's own mtime is checked before rescanning: a
    directory's mtime changes precisely when entries are added or removed,
    which is exactly the set this cache tracks. Rescanning 64k entries and
    rebuilding the match index costs ~500ms, so without this probe one query
    per minute paid that toll even when nothing had changed. Now the common
    case is a single stat().
    """
    global _entity_dirs_cache, _entity_dirs_index, _entity_dirs_mtime
    now = time.monotonic()
    if _entity_dirs_cache is not None and (now - _entity_dirs_cache[0]) < _ENTITY_DIRS_TTL:
        return _entity_dirs_cache[1]
    if not FACTS_ROOT.exists():
        _entity_dirs_cache = (now, [])
        _entity_dirs_index = {}
        _entity_dirs_mtime = None
        return []
    try:
        root_mtime = FACTS_ROOT.stat().st_mtime_ns
    except OSError:
        root_mtime = None
    if (_entity_dirs_cache is not None and _entity_dirs_index is not None
            and root_mtime is not None and root_mtime == _entity_dirs_mtime):
        # Nothing added or removed — renew the lease, skip the rescan.
        _entity_dirs_cache = (now, _entity_dirs_cache[1])
        return _entity_dirs_cache[1]
    _entity_dirs_mtime = root_mtime
    names = [d.name for d in FACTS_ROOT.iterdir() if d.is_dir()]
    _entity_dirs_cache = (now, names)
    # Built in the same pass so the O(1) lookup below never triggers its own
    # scan. FIRST occurrence wins, because `names` preserves iterdir() order
    # and the scan this replaces returned the first case-insensitive match.
    # A dict comprehension would give last-wins and silently resolve the 4
    # case-colliding dirs (OpenClaw/openclaw, Schema/schema, ...) to the other
    # directory — worth 0.005 MRR when it changed which facts got loaded.
    idx: dict[str, str] = {}
    for n in names:
        idx.setdefault(n.lower(), n)
    _entity_dirs_index = idx
    return names


def _get_entity_dirs_index() -> dict[str, str]:
    """Lowercased-name → on-disk-name map for FACTS_ROOT, cached for 60s."""
    _get_entity_dirs_cached()  # refreshes both cache and index when stale
    return _entity_dirs_index or {}


def _invalidate_entity_dirs_cache() -> None:
    """Clear the entity-dir cache. Call after creating a new entity dir."""
    global _entity_dirs_cache, _entity_dirs_index, _entity_dirs_mtime
    _entity_dirs_cache = None
    _entity_dirs_index = None
    _entity_dirs_mtime = None


# ── Entity resolution ───────────────────────────────────────────────────────

def _find_entity_dir(entity: str) -> Optional[Path]:
    """Find an entity directory under FACTS_ROOT, case-insensitive.

    O(1) via the cached lowercase index. This used to walk
    ``FACTS_ROOT.iterdir()`` with an ``is_dir()`` stat per entry on EVERY
    call — ~30 calls per `vault_recall`, so ~1.9M stat syscalls per query
    once the corpus reached 64k entity dirs. That was ~200ms at the ~2,700
    entities this code was written against and ~5s by 2026-08-06; it was the
    dominant cost in retrieval latency (#380).

    The exact-path fallback preserves the old contract: a directory created
    after the last cache refresh is still found immediately, so writers that
    forget `_invalidate_entity_dirs_cache()` don't silently 404.
    """
    if not FACTS_ROOT.exists():
        return None
    hit = _get_entity_dirs_index().get(entity.lower())
    if hit is not None:
        return FACTS_ROOT / hit
    # Cache miss: the dir may have been created since the last refresh.
    # An exact-name probe is one stat, versus a full 64k-entry rescan.
    direct = FACTS_ROOT / entity
    if direct.is_dir():
        return direct
    return None


def _load_aliases() -> dict:
    """The alias map as `{surface: canonical}`, self-identities included.

    Reads app.kg_store. Kept in the flat shape its callers expect; the
    authoritative lookup is `_resolve_entity`, which asks the store directly
    rather than materialising this dict.
    """
    from app.kg_store import StoreUnavailable, store
    try:
        st = store()
        out = st.aliases.all()
        for name in st.entities.all():
            out.setdefault(name, name)
        return out
    except StoreUnavailable:
        return {}


def _save_aliases(aliases: dict) -> None:
    """Write a `{surface: canonical}` map into the store.

    Additive: entries not in `aliases` are left alone. Nothing in Lloyd
    rewrites the whole table any more — the sweep and the extractor set one
    alias at a time, which is what made the whole-file rewrite races
    (2026-08-22, 2026-09-03) possible in the first place.
    """
    from app.kg_store import alias_kind, store
    st = store()
    with st.transaction():
        for surface, canonical in (aliases or {}).items():
            if not surface or not canonical:
                continue
            if surface == canonical:
                st.entities.register(canonical)
            else:
                st.aliases.set(surface, canonical, kind=alias_kind(surface, canonical),
                               origin="legacy")


def _resolve_entity(name: str, *, mode: Literal["read", "write"]) -> tuple[str, bool]:
    """Resolve an entity name to its canonical form.

    Returns (canonical_name, is_new). `is_new` is True when no resolution
    happened (the caller's input is returned verbatim).

    Resolution order depends on mode:

    Read mode (mode="read"):
        1. Exact case-insensitive directory match
        2. Alias table lookup (original case or lowercase)
        3. Fuzzy match against known entities (in-memory only)
        Returns the caller's input verbatim if nothing resolves.
        NO DISK WRITES — pure read-side resolution.

    Write mode (mode="write"):
        1. Exact case-insensitive directory match
        2. Alias table lookup (original case or lowercase)
        Returns the caller's input verbatim if nothing resolves.
        FUZZY MATCHING IS DISABLED. The caller writes to the exact name
        they specified — this is the fix for the silent fuzzy-merge bug.

    Why mode is required (and kw-only):
        Task #340 PR 3 introduces this parameter to fix a silent data-
        corruption bug: previously fuzzy match ran regardless of
        auto_create, so fact_add(entity="Lloyd", ...) could silently
        land on "lloyd-mc" if the fuzzy threshold passed. Reads also
        wrote to the alias table as a side effect.

        Forcing mode= keyword-only and required makes every callsite
        explicit about whether it expects existing data (read) or is
        creating something new (write). No silent default.

    Mode="write" no longer creates the entity directory — that's still
    the caller's responsibility (see _fact_add). It also no longer
    pre-registers an alias mapping, because the caller's input IS the
    canonical name.
    """
    name = name.strip()
    if not name:
        return name, True

    # 1. Exact directory match (both modes).
    entity_dir = _find_entity_dir(name)
    if entity_dir:
        return entity_dir.name, False

    # 2. Alias table lookup (both modes). One indexed query against the
    #    store, case-insensitive with exact case preferred — the JSON era
    #    materialised a 19,914-entry dict on every miss to do the same thing.
    from app.kg_store import StoreUnavailable, store
    try:
        canonical = store().aliases.resolve(name)
    except StoreUnavailable:
        canonical = None
    if canonical and _find_entity_dir(canonical):
        return canonical, False

    # 3. Fuzzy match — read mode ONLY, in-memory, no persistence.
    if mode == "read":
        known_entities = _get_entity_dirs_cached()
        fuzzy_match = _fuzzy_entity_match(name, known_entities)
        if fuzzy_match:
            return fuzzy_match, False

    # Nothing resolved — return verbatim, mark as new.
    return name, True
