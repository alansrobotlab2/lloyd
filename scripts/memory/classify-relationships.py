#!/usr/bin/env python3
"""Typed-edge relation classifier — v1. RETIRED as a driver, kept for its helpers.

`classify-v4-batch.py` is the runner; it and `classify-relationships-v4.py`
import this module's fact-snippet loading, entity-dir resolution, prompt
scaffolding and defaults. `main()` was removed on 2026-09-04 — running v1
against today's graph would write JSON nothing applies.

Original purpose (backlog #294 Phase 1B): read the edge graph, find `mentions`
edges — the ~97.8% dominant type produced by substring matching — and ask a
local LLM to reclassify each into a richer typed vocabulary. v4 does that now,
with quote grounding and a direction check.

"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR
RELATIONSHIPS_FILE = FACTS_DIR / "_relationships.json"
DEFAULT_OUTPUT = Path.home() / "lloyd" / "_pipeline" / "memory-graph" / "classified.jsonl"
DEFAULT_ENDPOINT = "http://127.0.0.1:8096/v1/chat/completions"
DEFAULT_MODEL = "primary"
DEFAULT_MAX_CTX_CHARS = 1500
DEFAULT_TIMEOUT_SEC = 60

# Vocabulary the classifier is allowed to emit. Any type not in this set is
# coerced back to "mentions" with low confidence. Keep in sync with
# agent_mcp/memory.py EDGE_TYPE_WEIGHTS.
VOCABULARY = [
    "uses",
    "depends_on",
    "implements",
    "supersedes",
    "part_of",
    "created_by",
    "discusses",
    "competes_with",
    "related_to",
    "mentions",
]

SYSTEM_PROMPT = (
    "You are a relation classifier for an entity knowledge graph. "
    "You emit strict JSON and nothing else."
)

USER_PROMPT_TEMPLATE = """Classify the strongest semantic relationship from SOURCE to TARGET.

Vocabulary (pick ONE; definitions are directional: SOURCE [verb] TARGET):
- uses: SOURCE actively invokes, calls, or leverages TARGET as a tool/library/service to accomplish its function.
    YES: "Lloyd uses Claude Code" (Lloyd spawns claude CLI).
    NO:  "SOURCE writes output to a path inside TARGET" — storage location is not usage.
- depends_on: SOURCE cannot function without TARGET (stronger than "uses"). Explicit runtime or build dependency.
- implements: SOURCE is a concrete realization or instance of TARGET's spec/interface/concept.
    NO:  Do NOT use when SOURCE and TARGET appear to be aliases or alternate names for the same thing — use "related_to" instead.
- supersedes: SOURCE explicitly REPLACES, OBSOLETES, or is the successor to TARGET as a temporal replacement.
    REQUIRED: the evidence must contain language like "replaces", "obsoletes", "deprecated by", "successor to", "new version of", or "takes over from".
    NO:  "modernizes", "manages", "reduces dependency on", "integrates with", "is compatible with" — these are NOT supersedes.
    NO:  Entity-name confusion (e.g., SOURCE supersedes a different-named TARGET that's actually the same product family but unrelated).
- part_of: SOURCE is a subsystem, component, sub-task, or sub-folder INSIDE TARGET.
    Direction check: if the evidence says TARGET contains/includes SOURCE, emit part_of. If SOURCE contains TARGET, the direction is reversed — emit "mentions" instead of flipping the verb.
- created_by: SOURCE was built, authored, or produced by TARGET (TARGET is the creator/author/company).
    YES: "LangGraph → LangChain" (company created the framework).
    NO:  "Roman → Author" — 'Author' is a role, not an entity that created Roman. Use "related_to" for role/is-a relations.
    NO:  "Lloyd → extract-transcript.py" — if Lloyd created the script, the direction is reversed; emit "mentions".
- discusses: SOURCE is a document/note/paper/conversation whose subject is TARGET.
- competes_with: SOURCE and TARGET are ALTERNATIVE solutions for the same purpose or use-case.
    NO:  A product and its maker (e.g., "Claude competes_with Anthropic" — Claude IS Anthropic's product; use "created_by" in reversed direction or "mentions").
    NO:  Unrelated entities that happen to co-occur in text.
- related_to: Semantically connected but none of the above fits clearly. Use for is-a/role relations, aliases, ambiguous-direction, or unusual pairings.
- mentions: TARGET appears only incidentally in SOURCE's text; no strong semantic link. Also use when direction is reversed for an asymmetric relation.

Return strict JSON with no prose, no markdown:
{{"type": "<one of the vocabulary>", "confidence": <0.0-1.0>, "reason": "<8-25 words>"}}

Rules:
- **Direction check FIRST.** For asymmetric verbs (uses, created_by, part_of, implements, supersedes), confirm the relation runs SOURCE→TARGET before emitting the verb. If the evidence describes TARGET acting on SOURCE (or containing SOURCE), return "mentions" rather than emitting the verb with reversed direction. Do not rely on surface word order in the context.
- Pick the MOST SPECIFIC type that fits. Only use "mentions" if nothing stronger applies OR direction is wrong.
- Confidence 0.9+ only when the evidence is explicit. 0.5-0.8 for inferred. <0.5 for weak.
- If CONTEXT is labeled "(reverse direction: ...)", the text is from TARGET mentioning SOURCE. Re-verify direction carefully; if you can't confirm SOURCE→TARGET, emit "mentions".
- If CONTEXT says "no fact text available", classify from entity names alone with confidence ≤ 0.4 — prefer "mentions" or "related_to" over stronger types.
- Reason must cite what in the context supports the choice AND explicitly name SOURCE and TARGET to verify direction (one sentence, 8-25 words).

SOURCE: {source}
TARGET: {target}

CONTEXT:
{context}
"""


def _load_relationships() -> dict:
    """Read view of the edge graph, kept for v4's imports of this module.

    The v1 driver is retired; v4 reads the store directly. This shim exists
    so an old call site cannot silently read a stale JSON file.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from app.kg_store import store
    return {"edges": store().edges.all()}


_DIR_CACHE: dict[str, str | None] = {}


def _resolve_entity_dir(name: str) -> Path | None:
    """Resolve an entity name to its facts directory.

    Handles common drift: exact match → case-insensitive match. Returns None
    if no directory exists. Cached per-process since FACTS_DIR listing is
    stable within a run.
    """
    if name in _DIR_CACHE:
        cached = _DIR_CACHE[name]
        return Path(cached) if cached else None

    exact = FACTS_DIR / name
    if exact.is_dir():
        _DIR_CACHE[name] = str(exact)
        return exact

    # Build the lowercased index lazily once per run
    if "_IDX_" not in _DIR_CACHE:
        idx = {
            d.name.lower(): d.name
            for d in FACTS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        }
        # encode as json string so it survives the str|None dict type
        _DIR_CACHE["_IDX_"] = json.dumps(idx)

    idx = json.loads(_DIR_CACHE["_IDX_"])
    real = idx.get(name.lower())
    if real:
        resolved = FACTS_DIR / real
        _DIR_CACHE[name] = str(resolved)
        return resolved

    _DIR_CACHE[name] = None
    return None


#: Parsed active-fact list per resolved entity directory. The classifier only
#: reads the fact tree -- the sole writer in this checkout
#: (`kg-pipeline.py:258-294`) does an open/write/replace on a path it builds
#: itself and never goes through anything here -- and an autonomy run is a new
#: process each time, so a run this long cannot observe a change that this cache
#: would hide.
#:
#: Bounded, and the tally uses `sys.getsizeof` because that is the instrument
#: that can be checked: `len(fact)` counts *code points*, while CPython stores 1,
#: 2 or 4 bytes per character depending on the string's highest code point, so a
#: length-based tally understates resident memory by up to 4x and is no RSS
#: proxy at all. What is tallied is the cached value only -- the fact strings
#: plus the `list` holding them -- so it excludes the frontmatter text that was
#: parsed to produce them, and it estimates live strings rather than peak RSS.
#:
#: The entry cap and the byte cap are separate bounds, each pinned by its own
#: test in tests/test_classify_entity_facts.py. The entry cap alone would be
#: unbounded in bytes exactly where this cost is (hub entities are expensive
#: *because* of how many facts one directory holds); a byte cap alone would let
#: a run over hundreds of thousands of tiny directories hold unbounded keys.
_FACTS_CACHE: "OrderedDict[str, list[str]]" = OrderedDict()
#: The 512 most-fact-populated directories of the live tree hold 17.6 MiB of
#: cached fact strings between them (whole-tree `sys.getsizeof` sweep, 2026-09-19:
#: 26,107 directories, 315,918 active facts, 44.0 MiB in total; largest single
#: directory `Lloyd` at 1.0 MiB). A run only touches directories its candidate
#: queue names, so the entry cap is the bound that binds on today's tree.
_FACTS_CACHE_MAX_ENTRIES = 512
#: `sys.getsizeof` of an empty `list`; added per cached value so the list object
#: is charged and not only its contents.
_FACTS_CACHE_SLOT = 56
#: 64 MiB is above the 44.0 MiB the *entire* live tree needs, so this cap cannot
#: bind on today's data: it is the tail guard for a tree that grows by an order
#: of magnitude, on a box that also pages a 95 GiB model.
_FACTS_CACHE_MAX_BYTES = 64 * 1024 * 1024
_FACTS_CACHE_BYTES = 0


def _facts_cache_bytes(facts: list[str]) -> int:
    """Resident bytes of one cached value, as `sys.getsizeof` measures them."""
    return sum(map(sys.getsizeof, facts)) + _FACTS_CACHE_SLOT


def _facts_cache_store(key: str, facts: list[str]) -> None:
    """Memoize one directory's parsed facts, evicting oldest-first on both caps.

    Keeps at least one entry whatever its size: dropping the last one would let
    a single oversized entity directory disable the cache outright.
    """
    global _FACTS_CACHE_BYTES
    _FACTS_CACHE[key] = facts
    _FACTS_CACHE_BYTES += _facts_cache_bytes(facts)
    while len(_FACTS_CACHE) > _FACTS_CACHE_MAX_ENTRIES or (
            _FACTS_CACHE_BYTES > _FACTS_CACHE_MAX_BYTES and len(_FACTS_CACHE) > 1):
        _, dropped = _FACTS_CACHE.popitem(last=False)
        _FACTS_CACHE_BYTES -= _facts_cache_bytes(dropped)


def _read_entity_facts(entity_dir: Path) -> list[str]:
    """Read all active fact texts from an entity directory.

    Parsed once per directory per process (see `_FACTS_CACHE`), and hands back a
    copy so a caller that mutates the list it was given cannot change what every
    later edge in the run sees — the one caller truncates and filters it, and the
    strings it produces are hashed into the resume key.
    """
    key = str(entity_dir)
    if key in _FACTS_CACHE:
        return list(_FACTS_CACHE[key])

    facts: list[str] = []
    # Resolved per directory, not at import: which loader exists is a property
    # of the yaml install, and reading it here is what lets the fallback be
    # exercised without restarting the process.
    fm_loader = getattr(yaml, "CSafeLoader", None)
    if not entity_dir.is_dir():
        _facts_cache_store(key, facts)
        return list(facts)
    for fname in os.listdir(entity_dir):
        if not fname.endswith(".md"):
            continue
        try:
            content = (entity_dir / fname).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not content.startswith("---"):
            continue
        parts = content.split("---", 2)
        if len(parts) < 3:
            continue
        try:
            # CSafeLoader is libyaml's C scanner. The pure-Python SafeLoader
            # that `yaml.safe_load` binds measured 2,473 ms over facts/vLLM's
            # 22 files / 2.90 MB against 370 ms for the same bytes here (6.7x),
            # and both parse the same fact list — pinned by
            # tests/test_classify_entity_facts.py.
            fm = yaml.load(parts[1], Loader=fm_loader) if fm_loader is not None \
                else yaml.safe_load(parts[1])
        except Exception:
            continue
        if not isinstance(fm, dict):
            continue
        for fact in fm.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            if fact.get("expired_at") or fact.get("invalid_at"):
                continue
            text = str(fact.get("fact", "")).strip()
            if text:
                facts.append(text)
    _facts_cache_store(key, facts)
    # A copy, so the memoized list is never handed out live.
    return list(facts)


def _build_pattern(name: str) -> re.Pattern:
    """Flexible match on entity name.

    Uses word boundaries when the name starts/ends alphanumeric; falls back
    to plain substring match otherwise (entity names can contain punctuation
    where `\\b` doesn't fire reliably, e.g. "AGENTS.md - Reviewer Workspace").
    """
    escaped = re.escape(name)
    prefix = r"\b" if name[:1].isalnum() else ""
    suffix = r"\b" if name[-1:].isalnum() else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


def _fmt_snippets(pool: list[str], max_chars: int) -> str:
    collected: list[str] = []
    budget = max_chars
    for text in pool:
        if len(text) + 3 > budget:
            if budget > 20:
                collected.append(text[: budget - 3] + "...")
            break
        collected.append(text)
        budget -= len(text) + 3
        if budget <= 0:
            break
    if not collected:
        return ""
    return "\n- " + "\n- ".join(collected)


def _load_fact_snippets(source: str, target: str, max_chars: int) -> str:
    """Pull fact-text context relating source and target.

    Strategy (stops at first non-empty result):
      1. Source entity's facts where target name appears (forward direction)
      2. Source entity's first few facts (general context about source)
      3. Target entity's facts where source name appears (reverse direction) —
         labeled so the classifier knows the direction is flipped
      4. Empty → classifier falls back to names alone
    """
    target_pat = _build_pattern(target)
    source_pat = _build_pattern(source)

    src_dir = _resolve_entity_dir(source)
    if src_dir is not None:
        src_facts = _read_entity_facts(src_dir)
        direct = [t for t in src_facts if target_pat.search(t)]
        if direct:
            return _fmt_snippets(direct[:5], max_chars)
        if src_facts:
            snippet = _fmt_snippets(src_facts[:3], max_chars)
            if snippet:
                return "(general context about source; target not directly mentioned)\n" + snippet

    tgt_dir = _resolve_entity_dir(target)
    if tgt_dir is not None:
        tgt_facts = _read_entity_facts(tgt_dir)
        reverse = [t for t in tgt_facts if source_pat.search(t)]
        if reverse:
            return "(reverse direction: TARGET's fact text where SOURCE is mentioned)\n" + _fmt_snippets(reverse[:5], max_chars)

    return ""


def _build_prompt(source: str, target: str, context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        source=source,
        target=target,
        context=context or "(no fact text available for either entity)",
    )


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction. LLM may wrap in fences or prose."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Strip code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # First balanced-brace object
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _normalize_classification(raw: dict, fallback_reason: str) -> dict:
    """Coerce classifier output into the expected shape + vocabulary."""
    etype = str(raw.get("type", "")).strip().lower()
    if etype not in VOCABULARY:
        return {"type": "mentions", "confidence": 0.3, "reason": f"classifier returned out-of-vocab '{etype}'"}
    try:
        conf = float(raw.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.5
    reason = str(raw.get("reason", "")).strip() or fallback_reason
    if len(reason) > 400:
        reason = reason[:397] + "..."
    return {"type": etype, "confidence": conf, "reason": reason}


def _call_llm(endpoint: str, model: str, prompt: str, timeout: int) -> dict | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        # vLLM runs with --scheduling-policy priority (lower = sooner). Interactive
        # chat sends 0 and autonomy runs send 1; batch classification is the
        # lowest-value traffic on the box and must yield to both. Without this it
        # competed at the default and starved the fleet: a 3,787-edge batch at
        # concurrency 4 pushed task #70 — normally a 21-44s run — past its 300s
        # timeout on the first cycle after the rebuild started.
        "priority": 2,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"[llm] HTTP {e.code}: {e.reason}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[llm] request failed: {e!r}", file=sys.stderr)
        return None
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        print(f"[llm] unexpected response shape: {str(data)[:200]}", file=sys.stderr)
        return None
    parsed = _extract_json(content)
    if parsed is None:
        print(f"[llm] unparseable: {content[:200]}", file=sys.stderr)
        return None
    return parsed


def _edge_key(edge: dict) -> tuple:
    return (edge.get("source", ""), edge.get("target", ""), edge.get("type", ""))


def _already_classified(output_path: Path) -> set:
    """Read existing classified.jsonl and return edge-keys already processed."""
    if not output_path.exists():
        return set()
    seen = set()
    try:
        with output_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                seen.add((rec["source"], rec["target"], rec["original_type"]))
    except Exception as exc:
        print(f"[warn] couldn't parse existing {output_path}: {exc!r}", file=sys.stderr)
    return seen


# main() removed 2026-09-04. The v1 driver is retired — `classify-v4-batch.py`
# is the runner and reads the store directly. This module survives only for the
# helpers v4 imports from it (`_load_fact_snippets`, `_resolve_entity_dir`, the
# prompt scaffolding and the defaults). Running v1 against today's graph would
# write JSON that nothing applies.
