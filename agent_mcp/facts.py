#!/usr/bin/env python3
"""
Lloyd MCP Server: Facts — knowledge-graph facts and typed relationships.

Tools:
    fact_get, fact_add, fact_resolve, fact_resolve_apply, fact_invalidate,
    fact_relate, fact_relationships  (7 tools)

`fact_profile` and `fact_check` were retired on 2026-09-23: `fact_get` took the
profile's per-category cap and `query` ranking, and `fact_check` was
`fact_resolve` under a second name. `_fact_check` stays as a function, the
contradiction detector's direct entry point for tests.

`fact_path` and `fact_neighbors` were deleted from this module on 2026-09-24
(#1077, on the #877 precedent for an advertised-but-never-called surface):
zero calls across every stored session, while the writer `fact_add` has
hundreds. Nothing is hidden behind a flag — no non-test caller existed for
either, so both handlers, their caps and their registrations are gone. What
the removed walk owned that survives is the confidence floor, now taken by
`fact_relationships` as `min_confidence`. What does NOT survive is a
walk-time-bounded expansion: `agent_mcp.retrieval.graph_weighted_neighbors`
bounds nothing during its walk (`top_k` sliced only at the end), so whoever
wires a graph arm into per-turn retrieval (#1025) re-implements the caps.

Data root: app.paths.VAULT_FACTS_ROOT
    (currently ~/lloyd-data/_pipeline/vault-derived/facts/)
Edge graph, aliases, entity registry, fact index: app.kg_store

Split out of agent_mcp/memory.py as part of Task #340 PR 5. Owns the
fact tool handlers and the entity-keyed fact files
(<FACTS_ROOT>/<Entity>/<Entity>-<category>.md).

The retrieval core (entity extraction, relationships index + cache,
graph expansion, fact ranking) lives in agent_mcp.retrieval — shared
with vault.py via public names. The underscore aliases imported below
keep this module's call sites and external consumers stable.
"""

import datetime
import re
from pathlib import Path

from mcp.types import Tool

try:
    from app.entity_naming import looks_like_junk_entity as _is_junk_entity
except Exception:  # pragma: no cover - defensive import
    def _is_junk_entity(name: str) -> bool:
        return False

from app.entity_kind import derive_kind as _derive_kind
from app.entity_naming import (
    SCHEMA_TYPES as _SCHEMA_TYPES,
    gate_entity_name as _gate_entity_name,
    normalize_declared_type as _normalize_declared_type,
    schema_identity as _schema_identity,
)
from agent_mcp._shared import (
    FACTS_ROOT,
    ErrorCode,
    atomic_write_text,
    _err,
    _find_entity_dir,
    _invalidate_entity_dirs_cache,
    _parse_fact_frontmatter,
    _resolve_entity,
    _token_overlap,
    _wrap,
    _write_fact_frontmatter,
)
from app.atomic_io import locked_file
from app.fact_ids import assign_ids as _assign_fact_ids, category_prefix, next_fact_id
from app.kg_store import (
    EDGE_TYPES, StoreUnavailable, canonical_edge_type, text_hash as _text_hash, store as _store,
)
from agent_mcp.retrieval import (  # noqa: F401  (re-exported compat names)
    EDGE_TYPE_WEIGHTS,
    RelationshipsCorrupt,
    FACT_GODNODE_THRESHOLD,
    FACT_RANK_CAP_GRAPH,
    FACT_RANK_CAP_SEED,
    extract_entities_from_query as _extract_entities_from_query,
    fact_matches_tokens as _fact_matches_tokens,
    fact_query_tokens as _fact_query_tokens,
    fact_score as _fact_score,
    get_entity_edge_counts as _get_entity_edge_counts,
    get_facts_sync as _get_facts_sync,
    graph_expand_entities as _graph_expand_entities,
    graph_weighted_neighbors as _graph_weighted_neighbors,
    invalidate_relationships_cache as _invalidate_relationships_cache,
    fact_source_file as _fact_source_file,
    load_relationships as _load_relationships,
)

# ── Constants ────────────────────────────────────────────────────────────────

# Used by _detect_contradictions_sync, through _opposing_reason (#701).
#
# Each term is matched as a WHOLE WORD, which is what makes this list safe to
# read at face value. Two of these pairs are self-collapsing as bare
# substrings — `"active" in "inactive"` and `"supported" in "unsupported"` are
# true by construction — so before #701 any two facts that both said
# "unsupported", or both said "inactive", were classified as disagreeing. That
# is the exact class the module docstring blames for 32,857 false positives on
# `Lloyd`, and `REQUIRE_OPPOSING_TERMS` in `agent_mcp/fact_improvement.py`
# makes this the one trigger allowed to authorise an expiry. Measured on the
# 2026-09-15 nightly: 2 of 215 pairs fired `opposing_terms`, and both were
# false — `opposing_terms:success/failure` paired a 43% success rate with a
# sentence about the *cause* of failures.
#
# Keeping the self-collapsing pairs is deliberate: bounded matching does not
# blunt a real opposition, so "is active" vs "is inactive" still fires while
# "inactive" vs "inactive" does not.
_OPPOSING_PAIRS = [
    ("yes", "no"), ("true", "false"), ("enabled", "disabled"),
    ("active", "inactive"), ("supported", "unsupported"),
    ("working", "broken"), ("success", "failure"),
]

# Compiled `\b<term>\b`, one per term. `\b` on both sides is the boundary bare
# `in` lacked: it is what stops "no" matching inside *another*, *notable*,
# *denote*, and stops "supported" matching inside "unsupported".
_OPPOSING_TERM_RE: dict[str, re.Pattern] = {}


def _opposing_term_re(term: str) -> re.Pattern:
    """Whole-word matcher for one opposing term, compiled once per term."""
    rx = _OPPOSING_TERM_RE.get(term)
    if rx is None:
        rx = _OPPOSING_TERM_RE[term] = re.compile(rf"\b{re.escape(term)}\b")
    return rx


def _opposing_reason(t1: str, t2: str) -> str | None:
    """`opposing_terms:<a>/<b>` if the two lowercased texts oppose by term.

    Whole-word matching only (#701). Bare containment, which is what this
    function replaced, matched a term inside its own opposite and inside
    unrelated longer words, so the pairs it named were pairs phrased alike —
    and the reason string it produced was the one thing downstream
    (`REQUIRE_OPPOSING_TERMS`) trusted as evidence of disagreement.

    Returns the matched pair rather than a boolean: the caller stores it as the
    pair's `reason`, which `fact_improvement.plan_entity` then carries into the
    action's reason. A reviewer who sees `opposing_terms:working/broken` can
    reject that specific pair; a bare "opposing terms" cannot be rejected.
    """
    for a, b in _OPPOSING_PAIRS:
        ra, rb = _opposing_term_re(a), _opposing_term_re(b)
        if (ra.search(t1) and rb.search(t2)) or (rb.search(t1) and ra.search(t2)):
            return f"opposing_terms:{a}/{b}"
    return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _reindex_files(paths) -> None:
    """Re-read written fact files into the store's index.

    Every writer of a fact file owes the index this call: the markdown is
    the fact layer, but `facts_idx` is what the router, `fact_get` and
    the health report actually read. A write that skips it leaves the two
    disagreeing until the next full reindex.
    """
    try:
        _store().facts_idx.reindex(list(paths), root=FACTS_ROOT)
    except StoreUnavailable:
        pass          # markdown is written; `kg reindex` rebuilds the index


# ── write-time duplicate refusal (#499) ──────────────────────────────────────
#
# One session event used to reach the store twice with slightly different
# wording, and a second time verbatim, because every writer of a fact file
# checked only the file it was about to append to. `remember` had a verbatim
# check; `_fact_add`, the tool and `app/post_capture.py` did not — so the one
# guard that existed was bypassed by the highest-volume writer. The measured
# cost: 5,591 same-entity duplicate groups in the live store on 2026-09-12,
# 4,839 of them cross-file AND cross-category, which a file-scoped check
# structurally cannot see.
#
# The key is `(entity, text_hash)` against `facts_idx` — `text_hash` because
# `fact_id` is a per-file counter that restarts in every file, so it cannot
# identify a claim. Refusing at the write is also the whole point: #499
# records that retiring the lower-confidence copy of a pair after the fact
# moved `fact_entity_recall` 0.35 → 0.30. Nothing here expires anything.

def _store_copy_of(entity, fact_text):
    """The `facts_idx` row already holding `fact_text` for `entity`, any category.

    None when there is none, and also None when the store cannot be read —
    the caller falls back to the file-scoped check, so an unreadable store
    narrows the guard instead of disabling it.
    """
    try:
        return _store().facts_idx.find_duplicate(entity, fact_text)
    except StoreUnavailable:
        return None


def _in_file_copy_of(existing, fact_text, fact_file, entity, category):
    """The entry in this one file that already carries `fact_text`, or None.

    The fallback refusal while the store is unreadable: the guard degrades to
    the narrow same-file check it had before #499 rather than to appending.
    Reported in the same shape as an index row so a caller sees one thing
    either way.
    """
    h = _text_hash(fact_text)
    for f in existing:
        if not isinstance(f, dict):
            continue
        if _text_hash(str(f.get("fact") or "")) != h:
            continue
        if f.get("expired_at") or f.get("invalid_at"):
            continue          # a superseded copy does not refuse a new claim
        try:
            rel = str(fact_file.relative_to(FACTS_ROOT))
        except ValueError:
            rel = str(fact_file)
        return {"entity": entity, "category": category, "fact_id": f.get("id"),
                "text_hash": h, "file_path": rel}
    return None


def _duplicate_refusal(entity, category, dup) -> dict:
    """What `fact_add` returns when the entity already carries the text.

    Success, and nothing written: the claim is in the store, it just does not
    need a second row. Deliberately not an error — every caller here treats a
    non-success `fact_add` as a lost fact and retries or reports a drop, which
    is how a duplicate check turns into a data-loss warning on every retry.
    """
    return {"success": True, "skipped": True, "duplicate": True,
            "entity": entity, "category": category, "duplicate_of": dup,
            "reason": "duplicate: this entity already carries that fact "
                      "verbatim (same entity, same text_hash); nothing was appended"}


def _generate_fact_id(category: str, existing_ids=()) -> str:
    """The next `<prefix>-NNN` for this file.

    Was `f"{category[:4]}-{uuid4().hex[:4]}"`, which produced a second,
    incompatible ID scheme in the same files the extractor numbered
    sequentially. One scheme now — see app.fact_ids.
    """
    return next_fact_id(existing_ids, category_prefix(category))


# ── one fact, one file (#874) ────────────────────────────────────────────────
#
# `fact_resolve(auto_resolve=true)` used to collect the losers' ids into a dict
# and then walk every `*.md` in the entity directory marking any entry whose
# `id` matched. A fact id is a per-file counter (`app.fact_ids.next_fact_id`),
# so that walk reaches every fact in the entity sharing the loser's number —
# measured on the live `Assistant` entity 2026-09-08: 2 planned actions
# invalidated 25 facts, 29 active down to 4. `apply_action` in
# `agent_mcp/fact_improvement.py` works around it by aiming a substring at one
# stored text and halting an entity when one action reports more than one
# expiry, which made the loop safe and left `fact_resolve` as the live footgun.
#
# Both go through `_apply_fact_marks` now, keyed on (file, id) and refusing a
# mark whose file is unknown. So a condemned fact that was read without a file
# attribution cannot be marked at all — the failure lands closed, where the old
# code failed open and retired the neighbours.

def fact_identity(fact: dict):
    """The key naming exactly one stored fact: `(source_file, id)`.

    Never the id alone — ids collide across an entity's category files. Never
    the text either: `facts_idx`'s dedupe key `(entity, text_hash)` (#499) is a
    key for *refusing a write*, and ingestion can legitimately record one
    sentence twice, in which case it is two facts with two lifetimes.

    None for a fact read without `with_source_file`, which is every display and
    ranking reader. A caller that means to act has to read it that way first,
    and that read is the point: it is what makes the handle addressable.
    """
    if not isinstance(fact, dict) or not fact.get("id"):
        return None
    source = str(fact.get("source_file") or "").strip()
    return (source, str(fact["id"])) if source else None


def _apply_fact_marks(marks: dict, files_to_scan, *, field: str, stamp: str,
                      reason_field: str, text_matches: dict | None = None,
                      stop_after_first: bool = False) -> dict:
    """Mark the facts named by `marks` {(file_name, fact_id): reason}. One route.

    Both fact writers — `fact_resolve`, `fact_invalidate` and the improve
    loop's `apply_action` — reached the same frontmatter through their own loop,
    and the difference between the loops was the bug: one matched on
    `f.get("id")` across every file of an entity, so one condemned fact marked
    every fact that shared its id. Fact ids are a per-file counter
    (`app.fact_ids.next_fact_id`), so a mark now names the file the fact is in,
    and the one loop honours that scope.

    `text_matches` is {lowercase substring: (kind, reason)} for the caller whose
    handle is a phrase rather than a fact: the improve loop plans against a read
    view and is not always given an id it can carry back. It is matched against
    fact text only — never body prose, the rule `_fact_invalidate` already
    enforced — and `stop_after_first` ends the scan at the first hit, so a
    single action cannot mark two facts that happen to contain the same
    sentence.

    Returns the facts it marked, each with the file it came from, plus
    `unapplied` (a requested mark that found nothing) and `already_marked` (a
    candidate that already had `field` set). A caller that planned N actions and
    applied fewer has to be able to say so: the planned count is not evidence of
    the applied one.
    """
    matched: list[dict] = []
    unapplied: list[dict] = []
    already_marked: list[dict] = []
    touched: list[Path] = []
    wanted = dict(marks)
    by_text = dict(text_matches or {})
    seen_files: set[str] = set()

    for fact_file in files_to_scan:
        source = _fact_source_file(Path(fact_file))
        seen_files.add(source)
        keys_here = {k: v for k, v in wanted.items() if k[0] == source}
        if not keys_here and not by_text:
            continue
        if not Path(fact_file).exists():
            for key in keys_here:
                unapplied.append({"id": key[1], "why": keys_here[key],
                                  "reason": f"file absent: {source}"})
            continue
        # The lock covers the whole read-modify-write, as it does in `_fact_add`
        # and `_fact_invalidate`: four extractor threads write these same files,
        # and a writer that reads outside the lock and writes inside it drops
        # whatever a concurrent `fact_add` added between the two. Advisory, so it
        # only excludes writers that take it — which is why every writer of a
        # fact file has to take it (`app.atomic_io.locked_file`).
        with locked_file(fact_file):
            content = Path(fact_file).read_text(encoding="utf-8")
            frontmatter = _parse_fact_frontmatter(content)
            if "facts" not in frontmatter:
                continue
            changed = False
            for f in frontmatter["facts"]:
                if not isinstance(f, dict):
                    continue
                want, how = None, "identity"
                if f.get("id") and (source, f["id"]) in keys_here:
                    want = keys_here[(source, f["id"])]
                else:
                    ftext = (f.get("fact") or "").lower()
                    for sub, (kind, why) in by_text.items():
                        if sub and sub in ftext and not ftext.startswith(("note:", "> ")):
                            want = why or "manual invalidation"
                            how = kind
                            break
                if want is None:
                    continue
                if f.get(field):
                    already_marked.append({"id": f.get("id"), "file": source,
                                           "fact": str(f.get("fact") or "")[:80]})
                    if stop_after_first:
                        break
                    continue
                f[field] = stamp
                f[reason_field] = want
                changed = True
                matched.append({"id": f.get("id"), "file": source,
                                "fact": str(f.get("fact") or "")[:80], "how": how})
                if stop_after_first:
                    break
            if changed:
                body_start = content.find("---", 3)
                body = content[body_start + 3:] if body_start != -1 else ""
                atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)
                touched.append(Path(fact_file))

    hit = {(m["file"], m["id"]) for m in matched}
    already = {(a["file"], a["id"]) for a in already_marked}
    for key, why in wanted.items():
        if key not in hit and key not in already:
            unapplied.append({"id": key[1], "why": why,
                              "reason": "no active fact with that id in that file"})
    if by_text and not matched and not already_marked:
        unapplied.append({"id": None, "why": "; ".join(sorted(by_text))[:120],
                          "reason": "no fact text in the scanned files matched"})

    if touched:
        _reindex_files(touched)
    return {"marked": len(matched), "matched_facts": matched,
            "unapplied": unapplied, "already_marked": already_marked,
            "files_touched": [str(t) for t in touched],
            "files_scanned": sorted(seen_files)}


def _detect_contradictions_sync(entity: str, category: str = None, *,
                                facts: list = None) -> dict:
    """Pairwise contradiction scan. O(n²) in the entity's fact count.

    Refused above FACT_GODNODE_THRESHOLD facts. `Lloyd` has 5,489, which is
    15 million `_token_overlap` comparisons — measured at 113 seconds
    through MCP, and it produced 32,857 "contradictions", almost all of them
    the high-overlap heuristic firing on two facts that are merely phrased
    alike. Narrow with `category` to scan a slice.

    Pass `facts` to judge a list that was already read. The caller that does is
    `fact_improvement.plan_entity`, which read the entity with
    `with_source_file=True`; the pairs come back holding those same dicts, so
    the loser the caller judged is a loser it can write, file attribution
    included. The tool path reads for itself and behaves exactly as before.
    """
    if facts is None:
        facts = _get_facts_sync(entity, category).get("facts", [])
    if len(facts) > FACT_GODNODE_THRESHOLD:
        return {"entity": entity, "category": category, "contradictions": [],
                "checked": len(facts), "refused": True,
                "hint": (f"{entity} has {len(facts)} facts"
                         + (f" in category {category}" if category else "")
                         + f"; the pairwise scan is refused above "
                         f"{FACT_GODNODE_THRESHOLD} because it is O(n²) and the "
                         "overlap heuristic yields mostly false positives at that "
                         "size. Pass a narrower `category`.")}
    contradictions = []
    for i, f1 in enumerate(facts):
        for f2 in facts[i + 1:]:
            t1, t2 = f1.get("fact", "").lower(), f2.get("fact", "").lower()
            reason = _opposing_reason(t1, t2)
            if not reason and _token_overlap(t1, t2) > 0.6:
                reason = "high_overlap_potential_update"
            if reason:
                contradictions.append({"fact1": f1, "fact2": f2, "reason": reason})
    return {"entity": entity, "category": category, "contradictions": contradictions, "checked": len(facts)}


# ── Tool handlers ────────────────────────────────────────────────────────────

def _fact_get(params: dict) -> dict:
    """An entity's facts, capped per category and optionally ranked by `query`.

    This absorbed `fact_profile` (2026-09-23), which existed only to add the
    cap and the ranking: uncapped, a read of a hub entity returned every fact
    it had — `Lloyd` alone carries 5,489 — straight into the model's context.
    Each category keeps `limit_per_category` facts (default
    `FACT_RANK_CAP_SEED`; 0 means no cap), the most relevant to `query` when
    one is given and the most recent otherwise, and a capped category is named
    in `truncated_categories` so the cut is never silent.
    """
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM, facts=[])
    category = params.get("category") or None
    as_of = params.get("as_of") or None
    include_expired = bool(params.get("include_expired", False))
    query = (params.get("query") or "").strip()
    try:
        cap = int(params.get("limit_per_category", FACT_RANK_CAP_SEED))
    except (TypeError, ValueError):
        return _err("limit_per_category must be an integer", ErrorCode.INVALID_PARAM, facts=[])
    try:
        result = _get_facts_sync(entity, category, as_of=as_of,
                                 include_expired=include_expired)
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, facts=[])
    facts = result.get("facts") or []
    if not facts or (cap <= 0 and not query):
        return result
    by_cat: dict = {}
    for fact in facts:
        by_cat.setdefault(fact.get("category", "general"), []).append(fact)
    tokens = _fact_query_tokens(query) if query else []
    truncated: dict = {}
    kept: list = []
    for cat, cat_facts in by_cat.items():
        if tokens:
            cat_facts.sort(key=lambda f: (-_fact_score(f, tokens),
                                          str(f.get("created_at") or "")))
        else:
            cat_facts.sort(key=lambda f: str(f.get("created_at") or ""), reverse=True)
        if cap > 0 and len(cat_facts) > cap:
            truncated[cat] = len(cat_facts)
            cat_facts = cat_facts[:cap]
        kept.extend(cat_facts)
    result = dict(result)
    result["facts"] = kept
    result["fact_count"] = len(facts)
    if truncated:
        result["truncated_categories"] = truncated
        result["hint"] = (
            f"{entity} has {len(facts)} facts; each category is capped at {cap}. "
            "Pass `query` to rank by relevance, `category` to read one slice, "
            "or limit_per_category=0 for everything.")
    return result


def _writes_enabled() -> bool:
    """`knowledge_graph.write_enabled` in config.yaml.

    The rebuild sets this false while it extracts into a parallel tree: a
    fact added in a chat turn during that window would land in a directory
    about to be renamed to `facts-quarantine-<ts>`.
    """
    try:
        from app.config import CONFIG
        return bool(CONFIG.get("knowledge_graph", {}).get("write_enabled", True))
    except Exception:
        return True


def _fact_add(params: dict) -> dict:
    if not _writes_enabled():
        return _err(
            "fact writes are disabled (config.yaml knowledge_graph.write_enabled "
            "= false). A knowledge-graph rebuild is in progress; a fact added now "
            "would land in a tree about to be replaced. Say the fact in the "
            "conversation and it can be added when the rebuild lands.",
            ErrorCode.INTERNAL,
        )
    raw_entity = params.get("entity", "").strip()
    category = params.get("category", "").strip()
    fact_text = params.get("fact", "").strip()
    if not raw_entity or not category or not fact_text:
        return _err("entity, category, and fact are required", ErrorCode.MISSING_PARAM)
    source_doc = params.get("source_doc")
    if _is_junk_entity(raw_entity, source_doc):
        return _err(
            f"'{raw_entity}' looks like a filename, a code fragment or a pipeline "
            "run, not an entity; use a concept/project/person name",
            ErrorCode.INVALID_PARAM,
        )
    declared_raw = params.get("entity_type")
    declared = None
    if declared_raw not in (None, ""):
        # Refused before anything is written: a type the schema does not carry
        # would otherwise be dropped and the entity typed by guesswork (#758).
        declared = _normalize_declared_type(declared_raw)
        if declared is None:
            return _err(f"entity_type {declared_raw!r} is not a declared type; use one "
                        f"of {', '.join(_SCHEMA_TYPES)}", ErrorCode.INVALID_PARAM)
    confidence = float(params.get("confidence", 0.9))
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        # mode="write" — exact + alias only, no fuzzy match. The fact lands
        # on the literal name the caller specified. (#340 PR 3 — fixes the
        # silent fuzzy-merge data-corruption bug.)
        entity, is_new = _resolve_entity(raw_entity, mode="write")
        # A new name, or a declared one, is minted through the extractor's gate
        # so the row gets a kind: this path registered untyped and was where
        # nearly every `kind IS NULL` entity came from (#758). A caller with no
        # entity_type is still a declaration that the thing exists, so the kind
        # falls back to the shape rule rather than to a refusal. The gate asks
        # the schema first, so a declared alias files under its canonical.
        kind = declared or _derive_kind(raw_entity, source_doc)
        if is_new or _schema_identity(raw_entity):
            gated, verdict = _gate_entity_name(raw_entity, declared_type=kind,
                                               source_doc=source_doc)
            if not gated:
                return _err(f"'{raw_entity}' is not filed as an entity (gate verdict "
                            f"{verdict}); use a concept/project/person name",
                            ErrorCode.INVALID_PARAM)
            entity = gated
        entity_dir = _find_entity_dir(entity)
        if not entity_dir:
            entity_dir = FACTS_ROOT / entity
            entity_dir.mkdir(parents=True, exist_ok=True)
        fact_file = entity_dir / f"{entity}-{category}.md"
        provenance = params.get("provenance", "STATED")
        if provenance not in ("STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"):
            provenance = "STATED"

        # The lock covers read-modify-write. Four extractor threads write
        # these same files; without it whichever finishes last wins and the
        # other's facts are gone.
        with locked_file(fact_file):
            if fact_file.exists():
                raw = fact_file.read_text(encoding="utf-8")
                frontmatter = _parse_fact_frontmatter(raw)
                if not frontmatter and raw.strip():
                    # A file that exists, is non-empty, and will not parse is
                    # corrupt. Writing here would replace an entity's whole
                    # history with this one fact.
                    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
                    quarantine = fact_file.with_name(f"{fact_file.name}.corrupt-{stamp}")
                    fact_file.rename(quarantine)
                    return _err(
                        f"{fact_file.name} is corrupt and was quarantined as "
                        f"{quarantine.name}; nothing was overwritten. Retry to "
                        "start a fresh file.",
                        ErrorCode.INTERNAL,
                    )
            else:
                frontmatter = {}
            if not frontmatter:
                frontmatter = {"type": "facts", "entity": entity, "category": category, "facts": []}
            existing = frontmatter.setdefault("facts", [])
            # Refused here, inside the lock, so two concurrent writers adding
            # the same claim cannot both pass an earlier check and both append.
            dup = _store_copy_of(entity, fact_text) or _in_file_copy_of(
                existing, fact_text, fact_file, entity, category)
            if dup:
                return _duplicate_refusal(entity, category, dup)
            fact_id = _generate_fact_id(category, [f.get("id") for f in existing if isinstance(f, dict)])
            new_fact = {"fact": fact_text, "confidence": confidence, "category": category,
                        "id": fact_id, "created_at": now_iso, "valid_at": params.get("valid_at"),
                        "invalid_at": None, "expired_at": None, "provenance": provenance,
                        "source_doc": source_doc}
            existing.append(new_fact)
            _assign_fact_ids(existing, category)
            frontmatter["last_updated"] = now_iso
            body = (f"\n# {entity} - {category}\n\n**Entity:** {entity}\n"
                    f"**Category:** {category}\n**Fact Count:** {len(existing)}\n")
            atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)
        _invalidate_entity_dirs_cache()
        # The store learns about the entity and the new fact here rather than
        # waiting for the nightly reindex, so a fact added in chat is visible
        # to the router and to `facts_idx` immediately.
        # `skipped` is on both outcomes, so an absent key is never how a
        # caller learns the difference between the two.
        result: dict = {"success": True, "skipped": False, "fact_id": fact_id,
                        "entity": entity, "category": category}
        try:
            st = _store()
            # Only fills a registry that lags the tree; the gate above has
            # already registered a new name with its kind. The reindex must not
            # register the directory name itself: that was a second, untyped
            # mint of the same entity on the same write (#758).
            st.entities.register(entity, kind=None if st.entities.exists(entity)
                                 else (declared or _derive_kind(entity, source_doc)))
            st.facts_idx.update_file(fact_file, root=FACTS_ROOT, register_entities=False)
        except StoreUnavailable as exc:
            # The markdown write already succeeded and is the fact layer; the
            # index is derived and `kg reindex` rebuilds it. Say so, don't fail.
            result["warning"] = f"fact written; store index not updated ({exc})"
        if entity != raw_entity:
            result["resolved_from"] = raw_entity
        return result
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_check(params: dict) -> dict:
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM, contradictions=[], checked=0)
    try:
        result = _detect_contradictions_sync(entity, params.get("category"))
        if result.get("refused"):
            return _err(result["hint"], ErrorCode.INVALID_PARAM,
                        contradictions=[], checked=result["checked"])
        return result
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, contradictions=[], checked=0)


def _resolve_scan(params: dict) -> dict:
    """Scan one entity for contradictory pairs, with each fact file-attributed.

    One read shared by both verbs, so the report a caller acts on and the write
    it triggers cannot disagree about which pairs exist. Returns an `_err`
    payload — including the god-node refusal — or `{"entity", "contradictions"}`.

    Refuses outright on an entity above FACT_GODNODE_THRESHOLD facts, where the
    pairwise scan is O(n²) and the overlap heuristic produces mostly false
    positives. The scan runs for the write too: an unbounded scan is exactly as
    dangerous from a verb that marks as from one that reports.
    """
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM)
    try:
        # Read with a file attribution, because a loser selected by id alone is
        # a loser that cannot be written safely — see `fact_identity`. This read
        # returns copies, so the cache's own entries stay unwritten.
        judged = _get_facts_sync(entity, params.get("category"),
                                 with_source_file=True).get("facts", [])
        detection = _detect_contradictions_sync(entity, params.get("category"),
                                                facts=judged)
        if detection.get("refused"):
            return _err(detection["hint"], ErrorCode.INVALID_PARAM,
                        resolved=0, remaining=0)
        return {"entity": entity,
                "contradictions": detection.get("contradictions", [])}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, contradictions=[], checked=0)


def _fact_resolve(params: dict) -> dict:
    """Report the contradictory pairs on an entity. Marks nothing.

    It used to take `auto_resolve`, and the marking sat behind it in the same
    handler — so the tool whose name is a read was annotated read-only and drew
    none of the refusals that classification is supposed to carry (plan mode,
    the bench/eval sandbox, a sessionless call, a transport-drop retry). The
    write is `fact_resolve_apply` now (#1326); a caller that still passes
    `auto_resolve` gets the report and is pointed at the verb that acts.

    A same-confidence pair is left alone in both verbs — there is no basis to
    pick a winner, and the detector fires on `_token_overlap > 0.6`, which is
    two facts phrased similarly, not two facts that disagree. That is why the
    write is a separate call a caller has to name.
    """
    scanned = _resolve_scan(params)
    if "error" in scanned:
        return scanned
    contradictions = scanned["contradictions"]
    return {"entity": scanned["entity"], "resolved": 0,
            "contradictions": contradictions[:20],
            "remaining": len(contradictions),
            "hint": ("Reporting only. Call fact_resolve_apply(entity=...) to mark "
                     "the lower-confidence side invalid, or use fact_invalidate "
                     "to expire a specific fact.")}


def _fact_resolve_apply(params: dict) -> dict:
    """Mark the lower-confidence side of each contradictory pair `invalid_at`.

    Sets `invalid_at` only, never `expired_at`: expired is "was true, no longer
    is"; invalid is "should not have been recorded". Same-confidence pairs are
    skipped, and a loser is only marked when it carries a file attribution — a
    fact id is a per-file counter, so selecting by id alone is what made one
    call invalidate 25 facts to change 2 (#874).

    A writer by classification, which is the point of splitting it out: the name
    appears in none of the annotation tables, so plan mode and a bench session
    refuse it, a sessionless call is refused, and the pool will not re-send it
    after a dropped transport.
    """
    scanned = _resolve_scan(params)
    if "error" in scanned:
        return scanned
    entity = scanned["entity"]
    contradictions = scanned["contradictions"]
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        entity_dir = _find_entity_dir(entity)
        marks: dict[tuple, str] = {}
        unattributed = 0
        for contradiction in contradictions:
            f1, f2 = contradiction.get("fact1", {}), contradiction.get("fact2", {})
            c1, c2 = f1.get("confidence", 0.5), f2.get("confidence", 0.5)
            if c1 == c2:
                continue     # no basis to pick a winner
            loser = f2 if c1 > c2 else f1
            key = fact_identity(loser)
            if key is None:
                unattributed += 1
                continue
            marks[key] = (f"fact_resolve_apply: "
                          f"{contradiction.get('reason', 'contradiction')}")
        resolved = 0
        applied: dict = {"marked": 0, "matched_facts": [], "applied": 0,
                         "unapplied": [], "files_touched": []}
        if entity_dir and (marks or unattributed):
            applied = _apply_fact_marks(
                marks, list(entity_dir.glob("*.md")),
                field="invalid_at", stamp=now_iso, reason_field="invalid_reason")
            resolved = applied["marked"]
        unresolved_pairs = len(contradictions) - resolved - unattributed
        out = {"entity": entity, "resolved": resolved,
               "remaining": max(unresolved_pairs, 0),
               # Which facts, in which files. A count without the list cannot be
               # audited, and this change exists because a count was trusted.
               "facts": applied["matched_facts"]}
        if applied["unapplied"] or unattributed:
            # Say what could not be marked instead of quietly marking less. A
            # caller that reads `resolved` alone would otherwise report fewer
            # contradictions than it saw and never learn why.
            out["unapplied"] = [
                *({"id": None, "why": "", "reason":
                   "fact read without a file attribution; re-read it with "
                   "with_source_file before acting"}
                  for _ in range(unattributed)),
                *applied["unapplied"]]
        return out
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_invalidate(params: dict) -> dict:
    """Expire facts that are no longer current (were true, now outdated)."""
    entity = (params.get("entity") or "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM, expired_count=0)
    category = params.get("category") or None
    fact_substring = (params.get("fact_substring") or "").strip().lower()
    # An unscoped call would expire every fact the entity has: naming an entity
    # is not a decision about its whole history. This refusal and the default
    # date below are what `forget` added on top of this tool, folded in when
    # `forget` was retired (2026-09-23).
    if not fact_substring and not category:
        return _err("fact_invalidate needs a scope: pass `fact_substring` (text of "
                    "the fact to expire) or `category`. Refusing to expire every "
                    "fact an entity has on the strength of naming it.",
                    ErrorCode.MISSING_PARAM, expired_count=0)
    ended = (params.get("ended") or "").strip() or datetime.datetime.now(
        datetime.timezone.utc).date().isoformat()
    reason = (params.get("reason") or "").strip()
    try:
        resolved, _ = _resolve_entity(entity, mode="read")
        entity_dir = _find_entity_dir(resolved)
        if not entity_dir:
            return _err(f"Entity not found: {entity}", ErrorCode.NOT_FOUND, expired_count=0)
        expired_count = 0
        matched_facts = []
        files_to_scan = []
        if category:
            fact_file = entity_dir / f"{resolved}-{category}.md"
            if not fact_file.exists():
                fact_file = entity_dir / f"{entity}-{category}.md"
            if fact_file.exists():
                files_to_scan.append(fact_file)
        else:
            files_to_scan = list(entity_dir.glob("*.md"))
        touched: list = []
        for fact_file in files_to_scan:
            with locked_file(fact_file):
                content = fact_file.read_text(encoding="utf-8")
                frontmatter = _parse_fact_frontmatter(content)
                if "facts" not in frontmatter:
                    continue
                changed = False
                for f in frontmatter["facts"]:
                    if f.get("expired_at") or f.get("invalid_at"):
                        continue
                    if fact_substring and fact_substring not in f.get("fact", "").lower():
                        continue
                    f["expired_at"] = ended
                    if reason:
                        f["expired_reason"] = reason
                    changed = True
                    expired_count += 1
                    matched_facts.append({"id": f.get("id"), "fact": f.get("fact", "")[:80]})
                if changed:
                    body_start = content.find("---", 3)
                    body = content[body_start + 3:] if body_start != -1 else ""
                    atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)
                    touched.append(fact_file)
        # The index is what the router and vault_recall read. Without this an
        # expired fact went on being served as current until the next full
        # reindex — the markdown said expired, the Memory page said active.
        if touched:
            _reindex_files(touched)
        return {"success": True, "entity": resolved, "expired_count": expired_count, "matched_facts": matched_facts}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, expired_count=0)


def _fact_relate(params: dict) -> dict:
    """Add a typed relationship edge between two entities."""
    if not _writes_enabled():
        return _err(
            "edge writes are disabled (config.yaml knowledge_graph.write_enabled "
            "= false) while a knowledge-graph rebuild is in progress.",
            ErrorCode.INTERNAL,
        )
    source = params.get("source", "").strip()
    target = params.get("target", "").strip()
    rel_type = params.get("type", "").strip()
    if not source or not target or not rel_type:
        return _err("source, target, and type are required", ErrorCode.MISSING_PARAM)
    # A closed vocabulary (#546): a free-form type here became a count-1 type
    # in the store on every novel call. Checked in canonical spelling, which is
    # what the store writes, so `depends-on` is accepted as `depends_on`.
    rel_type = canonical_edge_type(rel_type)
    if rel_type not in EDGE_TYPES:
        return _err(
            f"type {params.get('type')!r} is not an edge type; use one of: "
            + ", ".join(sorted(EDGE_TYPES)),
            ErrorCode.INVALID_PARAM,
        )
    confidence = float(params.get("confidence", 0.9))
    provenance = params.get("provenance", "STATED")
    if provenance not in ("STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"):
        provenance = "STATED"
    try:
        # mode="write" so edges land on the literal names the caller
        # specified, not fuzzy-matched neighbours. (#340 PR 3.)
        src_resolved, _ = _resolve_entity(source, mode="write")
        tgt_resolved, _ = _resolve_entity(target, mode="write")
        if src_resolved == tgt_resolved:
            return _err(
                f"source and target both resolve to {src_resolved!r}",
                ErrorCode.INVALID_PARAM,
            )
        st = _store()
        existing = st.edges.find_active(src_resolved, tgt_resolved, rel_type)
        if existing is not None:
            return {"success": True, "action": "already_exists", "edge_id": existing["id"],
                    "source": src_resolved, "target": tgt_resolved, "type": rel_type}
        edge_id = st.edges.add({
            "source": src_resolved, "target": tgt_resolved, "type": rel_type,
            "confidence": confidence, "provenance": provenance,
            "source_doc": params.get("source_doc"),
            "evidence": params.get("evidence"),
        }, origin="fact_relate")
        return {"success": True, "action": "created", "edge_id": edge_id,
                "source": src_resolved, "target": tgt_resolved, "type": rel_type}
    except StoreUnavailable as exc:
        # Writing over an unreadable graph is how 6,539 edges become 1.
        return _err(f"edge store is unreadable, refusing to write: {exc}", ErrorCode.INTERNAL)
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_relationships(params: dict) -> dict:
    """Get all relationships for an entity (inbound + outbound)."""
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM, edges=[])
    direction = params.get("direction", "both")
    rel_type = params.get("type") or None
    # The confidence floor `fact_neighbors` applied during its walk (#1077).
    # 0.0 is `edges.active`'s own default, so a call that passes nothing gets
    # byte-for-byte the edges it got before the removed tool existed.
    min_confidence = float(params.get("min_confidence", 0.0))
    try:
        resolved, _ = _resolve_entity(entity, mode="read")
        st = _store()
        types = [rel_type] if rel_type else None
        if direction == "out":
            edges = st.edges.active(source=resolved, types=types,
                                    min_confidence=min_confidence)
        elif direction == "in":
            edges = st.edges.active(target=resolved, types=types,
                                    min_confidence=min_confidence)
        else:
            edges = st.edges.active(either=resolved, types=types,
                                    min_confidence=min_confidence)
        return {"entity": resolved, "edges": edges, "count": len(edges)}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, edges=[])


# ── Graph traversal (used by vault.py for vault_recall) ──────────────────────

# ── MCP registration ─────────────────────────────────────────────────────────

async def list_tools():
    return [
        Tool(name="fact_get", description=f"Use to read what is known about one entity; for a question across documents and facts use vault_recall instead. Returns the entity's facts, at most {FACT_RANK_CAP_SEED} per category (most recent, or most relevant to `query`); a capped category is named in truncated_categories.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string", "description": "Entity name; resolved through the alias table, so a near-miss usually still lands"}, "category": {"type": "string", "description": "Fact category (e.g. state, identity, preference) — one markdown file per entity/category"}, "query": {"type": "string", "description": "Rank each category by relevance to this text instead of recency"}, "limit_per_category": {"type": "integer", "description": f"Facts kept per category (default {FACT_RANK_CAP_SEED}; 0 = no cap)"}, "as_of": {"type": "string", "description": "ISO date — return facts valid at this point in time"}, "include_expired": {"type": "boolean", "description": "If true, include expired/invalidated facts"}}, "required": ["entity"]}),
        Tool(name="fact_add", description="Add a structured fact for a named entity and category. Writes a line to the entity's markdown fact file and indexes it; use one clear sentence per call rather than a paragraph. An entity that already carries that text verbatim is refused: the result reports success with skipped=true and the surviving copy in `duplicate_of`, and nothing is written, whatever category was asked for.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string", "description": "Entity name; resolved through the alias table, so a near-miss usually still lands"}, "category": {"type": "string", "description": "Fact category (e.g. state, identity, preference) — one markdown file per entity/category"}, "fact": {"type": "string", "description": "The fact, as one self-contained sentence that will still make sense read on its own"}, "confidence": {"type": "number", "description": "0.0-1.0 belief in the fact (default 0.9); the weaker side loses a contradiction"}, "valid_at": {"type": "string", "description": "ISO date the fact started being true (default: today)"}, "provenance": {"type": "string", "enum": ["STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"], "description": "How the fact was derived (default: STATED)"}, "source_doc": {"type": "string", "description": "Vault path this fact came from, for provenance"}, "entity_type": {"type": "string", "enum": list(_SCHEMA_TYPES), "description": "What kind of thing a NEW entity is; ignored for one that already exists (default: derived from the name)"}}, "required": ["entity", "category", "fact"]}),
        Tool(name="fact_resolve", description=f"Report contradictions between an entity's facts. Reports only — it marks nothing; `fact_resolve_apply` is the call that marks. Refused above {FACT_GODNODE_THRESHOLD} facts; pass `category` to scan a slice.", inputSchema={
            "type": "object", "properties": {
                "entity": {"type": "string", "description": "Entity name; resolved through the alias table, so a near-miss usually still lands"},
                "category": {"type": "string", "description": "Scan one category instead of the whole entity"},
            }, "required": ["entity"]}),
        Tool(name="fact_resolve_apply", description=f"Mark the lower-confidence side of each contradictory pair on an entity `invalid_at` (never expired), and report which facts in which files it marked. Refused above {FACT_GODNODE_THRESHOLD} facts; pass `category` to act on a slice. The write half of `fact_resolve`, split out so the read stays a read: this name is classified as a writer, so plan mode and a bench or eval session refuse it.", inputSchema={
            "type": "object", "properties": {
                "entity": {"type": "string", "description": "Entity name; resolved through the alias table, so a near-miss usually still lands"},
                "category": {"type": "string", "description": "Apply within one category instead of the whole entity"},
            }, "required": ["entity"]}),
        Tool(name="fact_invalidate", description="Expire facts that are no longer current by setting expired_at. Needs a scope, `fact_substring` or `category`: an unscoped call is refused rather than expiring every fact the entity has.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string", "description": "Entity name; resolved through the alias table, so a near-miss usually still lands"}, "category": {"type": "string", "description": "Fact category (e.g. state, identity, preference) — one markdown file per entity/category"}, "fact_substring": {"type": "string", "description": "Match facts containing this text"}, "ended": {"type": "string", "description": "ISO date when the fact stopped being true (default: today)"}, "reason": {"type": "string", "description": "Why the fact was expired"}}, "required": ["entity"]}),
        Tool(name="fact_relate", description="Add a typed relationship edge between two entities in the knowledge graph. Edges are expired rather than deleted, so a wrong edge is recoverable.", inputSchema={
            "type": "object", "properties": {"source": {"type": "string", "description": "Entity the edge points from (alias-resolved)"}, "target": {"type": "string", "description": "Entity the edge points to (alias-resolved)"}, "type": {"type": "string", "description": "Relationship type (e.g. built_on, uses, part_of, related_to)"}, "confidence": {"type": "number", "description": "0.0-1.0 belief in the edge (default 0.9)"}, "provenance": {"type": "string", "enum": ["STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"], "description": "How the edge was derived (default: STATED)"}, "source_doc": {"type": "string", "description": "Vault path this edge came from, for provenance"}}, "required": ["source", "target", "type"]}),
        Tool(name="fact_relationships", description="Use to list the typed edges one entity sits on (inbound + outbound); for what is known about an entity use fact_get instead, and for a question across documents and facts use vault_recall. `min_confidence` drops weaker edges (default 0.0 — every edge, whatever its confidence).", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string", "description": "Entity name; resolved through the alias table, so a near-miss usually still lands"}, "direction": {"type": "string", "enum": ["in", "out", "both"], "description": "Edge direction to return (default both)"}, "type": {"type": "string", "description": "Only return edges of this relationship type"}, "min_confidence": {"type": "number", "description": "Drop edges below this confidence (default 0.0 — keep all of them)"}}, "required": ["entity"]}),
    ]


async def call_tool(name: str, arguments: dict):
    handlers = {
        "fact_get": _fact_get, "fact_add": _fact_add,
        "fact_resolve": _fact_resolve,
        "fact_resolve_apply": _fact_resolve_apply,
        "fact_invalidate": _fact_invalidate,
        "fact_relate": _fact_relate, "fact_relationships": _fact_relationships,
    }
    handler = handlers.get(name)
    if handler:
        return _wrap(handler(arguments))
    return _wrap(_err(f"Unknown tool: {name}", ErrorCode.UNKNOWN_TOOL))
