#!/usr/bin/env python3
"""
Lloyd MCP Server: memory operations — one verb per cognitive operation.

Backlog #376, Priority 2. The memory surface had grown to 19 registered tools
(`fact_*` ×10, `vault_*` ×5, `memory_*` ×4) at four different abstraction
levels, and two earlier consolidation efforts (#174 "17→7", #340 the split)
moved the number *up*. Cognee's four-operation API — `remember`, `recall`,
`forget`, `improve` — is the counter-proposal, and it is the right shape for a
reason that has nothing to do with tidiness: 19 verbs at 3 levels means the
agent has to know which tool family owns a given act, and "the agent does not
know what it does not know" (architecture/gaps.md) is exactly that failure.

Four verbs is also not four fewer tools. These are **routers, not replacements**:

    remember → fact_add          (with a dedupe check the tool does not do)
    recall   → vault_recall      (documents + facts + graph, in one call)
    forget   → fact_invalidate   (refuses an unscoped blanket expire)
    improve  → fact_improvement.run_improvement  (the #376 feedback loop)

Deliberately *not* aliases: each one adds the one guard its underlying tool
lacks. `forget` without a scope is the same class of foot-gun as
`fact_resolve(auto_resolve=True)` used to be, so it refuses; `remember` of a
fact the entity already has returns `skipped` instead of appending a duplicate
— the exact defect behind 8,102 duplicate rows in `facts_idx`, which is
`text_hash`-measured, not `fact_id`-measured (fact_id is a per-file counter).

The 19 stay registered and unchanged. A caller that knows it wants
`fact_neighbors(min_confidence=…)` should keep calling it; these are for the
call that just wants to remember, recall, forget, or improve.
"""

from __future__ import annotations

import datetime
import re

from mcp.types import Tool

from agent_mcp import fact_improvement as improvement
from agent_mcp._shared import ErrorCode, _err, _wrap
from agent_mcp.facts import _fact_add, _fact_invalidate
from agent_mcp.retrieval import get_facts_sync as _get_facts_sync
from agent_mcp.vault import _vault_recall

# A remembered fact is the same claim as an existing one when their
# whitespace/case/punctuation-normalised forms match. Deliberately cheap and
# exact-ish: near-duplicates are the extractor's problem, not this router's.
_NORMALISE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalise(text: str) -> str:
    return _NORMALISE_RE.sub("", " ".join(str(text or "").lower().split()))


# ── remember ─────────────────────────────────────────────────────────────────

def remember(params: dict) -> dict:
    """Record one fact about one entity, unless the entity already has it.

    The dedupe check reads through `_get_facts_sync`, i.e. the same current-only
    view recall serves, so a fact that is recalled is a fact that will not be
    re-added. Expired/invalid facts do not count as coverage: re-stating a
    superseded claim is a new claim, and it should be recorded.
    """
    entity = str(params.get("entity") or "").strip()
    fact_text = str(params.get("fact") or "").strip()
    category = str(params.get("category") or "").strip()
    if not entity or not fact_text:
        return _err("entity and fact are required", ErrorCode.MISSING_PARAM)
    if not category:
        return _err("category is required — one of state, identity, preference, "
                    "usage, correction, or a new one; it becomes the file name",
                    ErrorCode.MISSING_PARAM)
    wanted = _normalise(fact_text)
    existing = _get_facts_sync(entity, category).get("facts") or []
    if any(_normalise(f.get("fact", "")) == wanted for f in existing):
        return {"success": True, "skipped": True, "entity": entity,
                "category": category,
                "reason": "this entity already carries that fact verbatim"}
    result = _fact_add(params)
    if result.get("error"):
        return result
    result["skipped"] = False
    return result


# ── recall ───────────────────────────────────────────────────────────────────

def recall(params: dict) -> dict:
    """Answer from memory: documents, entity facts and graph neighbours.

    Pure delegation to `vault_recall`, with `grep_code` defaulted off — the
    underlying tool also greps the lloyd checkout for code hits, which is
    almost never what "recall" means and adds latency to every query. Pass
    `grep_code: true` to get it back.
    """
    args = dict(params or {})
    args.setdefault("grep_code", False)
    return _vault_recall(args)


# ── forget ───────────────────────────────────────────────────────────────────

def forget(params: dict) -> dict:
    """Expire a fact that is no longer true, scoped to a match or a category.

    An unscoped `forget(entity=…)` would expire every fact the entity has, so
    it is refused: naming an entity is not a decision about its whole history.
    Pass `match` for one fact, or `category` for a slice.
    """
    entity = str(params.get("entity") or "").strip()
    match = str(params.get("match") or "").strip()
    category = str(params.get("category") or "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM)
    if not match and not category:
        return _err("forget needs a scope: pass `match` (a substring of the fact "
                    "to expire) or `category`. Refusing to expire every fact an "
                    "entity has on the strength of naming it.",
                    ErrorCode.MISSING_PARAM, expired_count=0)
    return _fact_invalidate({
        "entity": entity,
        "category": category or None,
        "fact_substring": match,
        "ended": params.get("ended") or datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "reason": params.get("reason") or "forget(): user or agent judged it no longer true",
    })


# ── improve ──────────────────────────────────────────────────────────────────

def improve(params: dict) -> dict:
    """Run one improvement pass over fact quality. Dry-run by default.

    Thin translation of tool arguments onto `fact_improvement.run_improvement`;
    the reasoning about what counts as evidence lives there.
    """
    params = params or {}
    sources = tuple(params.get("sources") or ("corrections", "drift"))
    return improvement.run_improvement(
        apply=bool(params.get("apply", False)),
        sources=sources,
        entities=params.get("entities") or None,
        days=int(params.get("days", improvement.DRIFT_WINDOW_DAYS)),
        limit=int(params.get("limit", 40)),
        report_eval=bool(params.get("report_eval", False)),
        eval_limit=int(params.get("eval_limit", 20)),
    )


# ── MCP registration ─────────────────────────────────────────────────────────

async def list_tools():
    return [
        Tool(name="remember", description="Record one fact about one entity through a single entry point; returns skipped=true when that entity already carries the same claim verbatim, so a repeated memory pass cannot inflate the fact count.", inputSchema={
            "type": "object", "properties": {
                "entity": {"type": "string", "description": "Entity the fact is about; resolved through the alias table"},
                "category": {"type": "string", "description": "Fact category (state, identity, preference, usage, correction…) — one markdown file per entity/category"},
                "fact": {"type": "string", "description": "The fact, as one self-contained sentence that still makes sense read alone"},
                "confidence": {"type": "number", "description": "0.0-1.0 belief in the fact (default 0.9); the weaker side loses a contradiction"},
                "provenance": {"type": "string", "enum": ["STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"], "description": "How the fact was derived (default STATED)"},
                "source_doc": {"type": "string", "description": "Vault path this fact came from, for provenance"},
            }, "required": ["entity", "category", "fact"]}),
        Tool(name="recall", description="Answer from memory in one call: vault documents, entity facts and graph-neighbour facts together. This is the read half of the memory surface; the lower-level fact_*/vault_* tools stay available when you need their parameters.", inputSchema={
            "type": "object", "properties": {
                "query": {"type": "string", "description": "Natural-language query; entities named in it are resolved and their facts returned alongside documents"},
                "limit": {"type": "integer", "description": "Documents to return (default 20)"},
                "expand_graph": {"type": "boolean", "description": "Also return facts from graph neighbours (default false)"},
                "graph_rerank": {"type": "boolean", "description": "Re-rank documents by graph votes (default: production setting)"},
                "demote_daily_logs": {"type": "boolean", "description": "Down-weight daily notes (default true)"},
                "grep_code": {"type": "boolean", "description": "Also grep the lloyd checkout for code hits (default false here; vault_recall defaults it on)"},
            }, "required": ["query"]}),
        Tool(name="forget", description="Expire a fact that is no longer true. Needs a scope — a substring match or a category — and refuses to expire every fact an entity has just because it was named.", inputSchema={
            "type": "object", "properties": {
                "entity": {"type": "string", "description": "Entity whose fact should be expired (alias-resolved)"},
                "match": {"type": "string", "description": "Case-insensitive substring identifying the fact to expire"},
                "category": {"type": "string", "description": "Expire the facts in this category instead of matching one"},
                "ended": {"type": "string", "description": "ISO date the fact stopped being true (default: now)"},
                "reason": {"type": "string", "description": "Why it is being forgotten; recorded on the fact"},
            }, "required": ["entity"]}),
        Tool(name="improve", description="Run one feedback pass over fact quality: read corrections and recent-write drift, pair contradictions, and expire or invalidate claims an independent reason condemns. Dry-run unless apply=true; reports before/after active-fact counts.", inputSchema={
            "type": "object", "properties": {
                "apply": {"type": "boolean", "description": "Actually change facts (default false — plan and report only)"},
                "sources": {"type": "array", "items": {"type": "string"}, "description": "Signal sources to read: corrections, drift (default both)"},
                "entities": {"type": "array", "items": {"type": "string"}, "description": "Improve only these entities instead of reading the signal sources"},
                "days": {"type": "integer", "description": "Drift window in days for entity selection (default 3)"},
                "limit": {"type": "integer", "description": "Maximum entities to examine (default 40)"},
                "report_eval": {"type": "boolean", "description": "Also score fact_entity_recall from eval/run_eval.py — slow, one query per second"},
            }}),
    ]


async def call_tool(name: str, arguments: dict):
    handlers = {"remember": remember, "recall": recall,
                "forget": forget, "improve": improve}
    handler = handlers.get(name)
    if handler:
        return _wrap(handler(arguments))
    return _wrap(_err(f"Unknown tool: {name}", ErrorCode.UNKNOWN_TOOL))
