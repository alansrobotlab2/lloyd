#!/usr/bin/env python3
"""v4 classifier: entity-type hints + two-step direction verification + hallucination gate.

Design targets from #294 v3 post-mortem (2026-04-21):
- 20%+ direction errors in v2 on asymmetric verbs (uses/part_of/implements/created_by)
- 10-15% hallucinations (model asserts world-knowledge facts not in context)
- 5-10% signal collapse (wrong entity-type assumptions — person vs system vs role)

Pipeline per edge:
  1. Pre-filter: skip self-edges, near-alias pairs → force "mentions" low-conf
  2. Derive entity type hints heuristically (FILE / SKILL / PERSON / ROLE / TASK / SYSTEM)
  3. LLM call 1: classify relation + quote-grounded reason
  4. Hallucination gate: verify reason's quoted phrase is actually in context
  5. LLM call 2 (only for asymmetric verbs): independent direction check
  6. Resolve: if direction check disagrees, downgrade or flip

Output JSONL adds fields:
- src_type_hint, tgt_type_hint — the derived entity types
- reason_quote — the backticked phrase from reason (if any)
- quote_verified — True if phrase found in context
- direction_check — {"verdict": "confirmed"|"reversed"|"unclear", "raw": "..."}
- verdict_adjustment — "none"|"downgraded"|"flipped"

Usage:
  .venvs/lloyd/bin/python scripts/memory/classify-relationships-v4.py --sample 20 --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse helpers from v2 classifier
CLASSIFIER_V2 = Path.home() / "lloyd" / "scripts" / "memory" / "classify-relationships.py"
_spec = importlib.util.spec_from_file_location("classifier_v2", str(CLASSIFIER_V2))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)

VOCABULARY = _v2.VOCABULARY
DEFAULT_ENDPOINT = _v2.DEFAULT_ENDPOINT
DEFAULT_MODEL = _v2.DEFAULT_MODEL
DEFAULT_MAX_CTX_CHARS = _v2.DEFAULT_MAX_CTX_CHARS
DEFAULT_TIMEOUT_SEC = _v2.DEFAULT_TIMEOUT_SEC
DEFAULT_OUTPUT = Path.home() / "lloyd" / "_pipeline" / "memory-graph" / "classified-v4.jsonl"

ASYMMETRIC_VERBS = {"uses", "depends_on", "implements", "supersedes", "part_of", "created_by", "discusses"}

# Asymmetric verbs that make no sense when one endpoint is a generic ROLE.
# A role is not a specific agent that can create/implement/supersede a specific
# artifact. "Roman created_by Author", "EvoSkills created_by researcher" etc.
# should collapse to `related_to` (the person PLAYS the role).
ROLE_BLOCKED_VERBS = {"created_by", "implements", "supersedes", "part_of", "depends_on", "uses", "discusses"}

# ---------------------------------------------------------------------------
# Entity alias table (loaded once at module import)
# ---------------------------------------------------------------------------

ALIASES_PATH = Path.home() / "obsidian" / "facts" / "entity-aliases.json"


def _load_aliases() -> dict[str, str]:
    """Load alias → canonical mapping from entity-aliases.json."""
    if not ALIASES_PATH.exists():
        return {}
    try:
        return json.loads(ALIASES_PATH.read_text("utf-8"))
    except Exception as exc:
        print(f"[alias] failed to load {ALIASES_PATH}: {exc}", file=sys.stderr)
        return {}


_ALIASES: dict[str, str] = _load_aliases()


def resolve_canonical(name: str) -> str:
    """Return canonical name via alias table; fall back to `name` unchanged.

    Tries exact match first, then lowercased match. If no mapping exists,
    returns the input unchanged.
    """
    if not _ALIASES:
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    low = name.lower()
    if low in _ALIASES:
        return _ALIASES[low]
    return name


# Patch v2's directory resolver so `_load_fact_snippets` picks up aliases
# transparently — saves us duplicating its fact-reading pipeline.
_original_v2_resolve = _v2._resolve_entity_dir


def _resolve_entity_dir_with_aliases(name):
    canonical = resolve_canonical(name)
    return _original_v2_resolve(canonical)


_v2._resolve_entity_dir = _resolve_entity_dir_with_aliases


# ---------------------------------------------------------------------------
# Entity type heuristics
# ---------------------------------------------------------------------------

ROLE_TOKENS = {
    "author", "researcher", "user", "developer", "admin", "admin user",
    "engineer", "designer", "scientist", "lead", "owner", "maintainer",
    "contributor", "reviewer", "tester", "operator", "editor", "student",
    "professor", "writer", "curator", "librarian", "architect",
}

CONCEPT_TOKENS = {
    "knowledge", "research", "concept", "idea", "field", "topic",
    "pattern", "framework", "methodology", "approach", "technique",
    "metric", "benchmark", "category", "paradigm",
}

FILE_EXTENSIONS = (".md", ".py", ".ts", ".tsx", ".js", ".json", ".yaml", ".yml",
                   ".toml", ".txt", ".sh", ".cjs", ".mjs", ".html", ".css",
                   ".cfg", ".ini", ".log", ".jsonl", ".csv")


def derive_entity_type(name: str) -> str:
    """Heuristic entity type classifier.

    Returns one of: FILE, TASK, PERSON, ROLE, CONCEPT, SKILL, SYSTEM, ENTITY.
    No LLM call — pure string rules.
    """
    n = name.strip()
    low = n.lower()

    # TASK: "Autonomy Task #33", "backlog_item_235", "run_39a_..."
    if re.search(r"(task|item|run)[ _#-]?\d", low):
        return "TASK"
    if re.match(r"^\d{8}[_-]\d{6}", n):  # timestamp-like ids
        return "TASK"

    # FILE: has extension or path separator
    if any(low.endswith(ext) for ext in FILE_EXTENSIONS):
        return "FILE"
    if "/" in n and not n.startswith("http"):
        return "FILE"

    # SKILL: kebab-case all-lowercase short name
    if re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)+", n):
        return "SKILL"

    # PERSON: "First Last" or "First M. Last" — capitalized tokens, 2-4 words,
    # no punctuation beyond periods, no digits
    if re.fullmatch(r"[A-Z][a-z]+(\s+[A-Z]\.?)?(\s+[A-Z][a-z]+){1,2}", n) and not any(c.isdigit() for c in n):
        return "PERSON"

    # ROLE: lowercase noun in role token list
    if low in ROLE_TOKENS:
        return "ROLE"

    # CONCEPT: common concept token
    if low in CONCEPT_TOKENS:
        return "CONCEPT"

    # SYSTEM: CamelCase multiword, or ends in "System"/"Agent"/"SDK"/"Service"
    if low.endswith(("system", "agent", "sdk", "service", "engine",
                     "pipeline", "loop", "orchestrator", "controller")):
        return "SYSTEM"
    if re.search(r"[A-Z][a-z]+[A-Z]", n):  # CamelCase
        return "SYSTEM"
    if " " in n and n[0].isupper():
        return "SYSTEM"

    return "ENTITY"


# ---------------------------------------------------------------------------
# Alias detection
# ---------------------------------------------------------------------------

def normalize_for_alias(name: str) -> str:
    """Aggressive normalization for alias detection."""
    n = name.lower().strip()
    # Remove common suffixes
    for suffix in (" system", " agent", " sdk", " service", " pipeline", " loop"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    # Strip punctuation, collapse whitespace + separators
    n = re.sub(r"[-_./\s]+", "", n)
    return n


def is_probable_alias(source: str, target: str) -> bool:
    """True if source and target look like aliases of the same entity.

    First checks the alias table (post-Tier-1 sweep): if both names resolve
    to the same canonical, they're aliases. Falls back to string heuristics
    for names not in the table.
    """
    if source == target:
        return True
    # Table-based: both resolve to same canonical
    s_canon = resolve_canonical(source)
    t_canon = resolve_canonical(target)
    if s_canon == t_canon:
        return True
    # String heuristic fallback
    ns = normalize_for_alias(source)
    nt = normalize_for_alias(target)
    if ns == nt:
        return True
    # Plural-y variants: "Skill" vs "Skills"
    if ns.rstrip("s") == nt.rstrip("s") and min(len(ns), len(nt)) > 3:
        return True
    return False


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V4 = (
    "You are a relation classifier for an entity knowledge graph. "
    "You emit strict JSON and nothing else."
)

USER_PROMPT_V4 = """Classify the strongest semantic relationship from SOURCE to TARGET.

Vocabulary (pick ONE; definitions are directional: SOURCE [verb] TARGET):
- uses: SOURCE actively invokes/calls/leverages TARGET as a tool/library/service.
- depends_on: SOURCE cannot function without TARGET. Explicit runtime/build dependency.
- implements: SOURCE is a concrete realization of TARGET's spec/interface.
- supersedes: SOURCE explicitly REPLACES/OBSOLETES TARGET. Required language: "replaces", "obsoletes", "deprecated by", "successor to", "new version of".
- part_of: SOURCE is a subsystem/component/sub-folder INSIDE TARGET.
- created_by: SOURCE was built/authored/produced BY TARGET. TARGET is the creator.
- discusses: SOURCE is a document/note/paper whose subject is TARGET.
- competes_with: SOURCE and TARGET are alternative solutions for the same use-case.
- related_to: semantically connected but none of the above fits. Use for is-a, role, alias, ambiguous.
- mentions: TARGET appears incidentally. No strong semantic link.

ENTITY TYPE HINTS (heuristic, may be wrong — use as a prior):
  SOURCE type: {src_type}
  TARGET type: {tgt_type}

Type → relation priors:
- PERSON → ROLE: use related_to (e.g. "Joscha Bach → researcher"), NOT implements.
- SOURCE → PERSON: if the context describes creation, use created_by with TARGET as creator.
- FILE → SYSTEM: if the file path suggests it lives inside the system, part_of is correct.
- SYSTEM → SKILL: if skill is in system's skills directory, part_of (skill is INSIDE system).
- TASK → SYSTEM: part_of (task is within system) unless task uses system as tool.
- Any → CONCEPT: usually discusses or related_to, rarely uses/implements.

DIRECTION RULE (hard constraint):
Before choosing an asymmetric verb (uses, depends_on, implements, part_of, created_by, supersedes),
you MUST state which entity is the ACTOR and which is the RECIPIENT of the action. If the
evidence shows TARGET acting on SOURCE (or containing SOURCE, creating SOURCE, etc.), the
direction is REVERSED — emit "mentions" instead of flipping the verb.

HALLUCINATION GUARD:
Your `reason` MUST include a short phrase from the CONTEXT in backticks (`like this`).
The phrase must be a real substring of the context below. If the context is empty or you
cannot ground the relation in the context, emit "mentions" with confidence ≤ 0.4.

Return strict JSON, no prose, no markdown:
{{"type": "<vocab>", "confidence": <0.0-1.0>, "reason": "<8-25 words with `quoted phrase`>"}}

SOURCE: {source}
TARGET: {target}

CONTEXT:
{context}
"""

DIRECTION_CHECK_PROMPT = """You are verifying the direction of a relation. Respond with JSON only.

Given:
  ENTITY_A = "{source}"
  ENTITY_B = "{target}"
  RELATION = "{relation}"
  INITIAL CLAIM: ENTITY_A {relation_verb} ENTITY_B

Question: based on the context below, is the claim correct, or is the direction reversed?

Semantic check:
- uses / depends_on: Does A actively invoke B? (A is the actor, B is the tool)
- part_of: Is A INSIDE B? (B contains A — not the other way around)
- implements: Is A a concrete realization of B's spec? (B is the abstract/spec)
- created_by: Was A created by B? (B is the creator/author)
- supersedes: Did A replace B? (B is the old thing)
- discusses: Is A a document whose subject is B? (A is the document, B is the topic)

Return strict JSON:
{{"verdict": "confirmed"|"reversed"|"unclear", "reason": "<8-20 words>"}}

- "confirmed" = the initial direction is correct (A → B)
- "reversed" = direction should be B → A instead
- "unclear" = context doesn't establish either direction

CONTEXT:
{context}
"""


# ---------------------------------------------------------------------------
# Hallucination gate
# ---------------------------------------------------------------------------

BACKTICK_RE = re.compile(r"`([^`]+)`")


def extract_reason_quote(reason: str) -> str | None:
    """Extract the backticked phrase from reason text. Returns None if absent."""
    m = BACKTICK_RE.search(reason)
    if not m:
        return None
    return m.group(1).strip()


def verify_quote_in_context(quote: str, context: str) -> bool:
    """Case-insensitive substring check. Fuzzy on whitespace."""
    if not quote or not context:
        return False
    # Normalize whitespace in both
    q = re.sub(r"\s+", " ", quote.strip()).lower()
    c = re.sub(r"\s+", " ", context).lower()
    return q in c


# ---------------------------------------------------------------------------
# Relation → verb for direction prompt
# ---------------------------------------------------------------------------

RELATION_VERB = {
    "uses": "uses",
    "depends_on": "depends on",
    "implements": "implements",
    "supersedes": "supersedes (replaces)",
    "part_of": "is part of",
    "created_by": "was created by",
    "discusses": "discusses (is about)",
}


# ---------------------------------------------------------------------------
# Main classification pipeline
# ---------------------------------------------------------------------------

def classify_edge_v4(
    source: str,
    target: str,
    context: str,
    endpoint: str,
    model: str,
    timeout: int,
    skip_direction_check: bool = False,
) -> dict:
    """Full v4 classification pipeline for one edge.

    Returns a dict with the standard classifier fields plus v4 instrumentation.
    """
    src_type = derive_entity_type(source)
    tgt_type = derive_entity_type(target)

    resolved_src = resolve_canonical(source)
    resolved_tgt = resolve_canonical(target)

    out = {
        "source": source,
        "target": target,
        "resolved_src": resolved_src,
        "resolved_tgt": resolved_tgt,
        "src_type_hint": src_type,
        "tgt_type_hint": tgt_type,
    }

    # Pre-filter 1: self-edges
    if source == target:
        out.update({
            "type": "mentions",
            "confidence": 0.1,
            "reason": "self-edge pre-filtered (source == target)",
            "reason_quote": None,
            "quote_verified": False,
            "direction_check": None,
            "verdict_adjustment": "pre_filter_self_edge",
        })
        return out

    # Pre-filter 2: near-aliases
    if is_probable_alias(source, target):
        out.update({
            "type": "related_to",
            "confidence": 0.4,
            "reason": f"alias pair pre-filtered — `{source}` and `{target}` normalize to same entity",
            "reason_quote": None,
            "quote_verified": False,
            "direction_check": None,
            "verdict_adjustment": "pre_filter_alias",
        })
        return out

    # Step 1: LLM classification with type hints + quote requirement
    prompt = USER_PROMPT_V4.format(
        source=source,
        target=target,
        src_type=src_type,
        tgt_type=tgt_type,
        context=context or "(no fact text available for either entity)",
    )
    raw = _call_llm_v4(endpoint, model, SYSTEM_PROMPT_V4, prompt, timeout)
    if raw is None:
        out.update({
            "type": "mentions",
            "confidence": 0.1,
            "reason": "classifier call failed",
            "reason_quote": None,
            "quote_verified": False,
            "direction_check": None,
            "verdict_adjustment": "call_failed",
        })
        return out

    norm = _v2._normalize_classification(raw, fallback_reason="v4 classifier normalized")
    relation = norm["type"]

    # Step 2: Hallucination gate — require reason to quote context
    quote = extract_reason_quote(norm["reason"])
    quote_ok = verify_quote_in_context(quote, context) if quote else False

    if not context:
        # No context to ground against — trust model at lowered confidence
        norm["confidence"] = min(norm["confidence"], 0.4)
    elif quote is None:
        # Model didn't quote → downgrade
        norm["confidence"] = min(norm["confidence"], 0.55)
    elif not quote_ok:
        # Model invented a quote → harder downgrade + force weaker relation
        norm["confidence"] = min(norm["confidence"], 0.45)
        if relation in ASYMMETRIC_VERBS:
            # Likely hallucination on strong relation → soften to mentions
            norm["type"] = "mentions"
            norm["reason"] = f"[hallucination-gate] quoted phrase `{quote}` not in context; downgraded from {relation}"
            relation = "mentions"

    # Step 2b: Role gate — asymmetric verb with a ROLE on either endpoint
    # is a type error. Soften to related_to before direction-checking.
    role_gate_fired = False
    if relation in ROLE_BLOCKED_VERBS and (src_type == "ROLE" or tgt_type == "ROLE"):
        role_gate_fired = True
        role_side = "source" if src_type == "ROLE" else "target"
        norm["type"] = "related_to"
        norm["confidence"] = min(norm["confidence"], 0.55)
        norm["reason"] = (
            f"[role-gate] {role_side} is generic ROLE; softened {relation} → related_to"
        )
        relation = "related_to"

    # Step 3: Direction check for asymmetric verbs
    direction_check = None
    verdict_adjustment = "role_gate" if role_gate_fired else "none"
    if not skip_direction_check and relation in RELATION_VERB and context:
        dir_prompt = DIRECTION_CHECK_PROMPT.format(
            source=source,
            target=target,
            relation=relation,
            relation_verb=RELATION_VERB[relation],
            context=context,
        )
        dir_raw = _call_llm_v4(endpoint, model, SYSTEM_PROMPT_V4, dir_prompt, timeout)
        if dir_raw is not None:
            verdict = str(dir_raw.get("verdict", "unclear")).strip().lower()
            reason = str(dir_raw.get("reason", ""))[:200]
            direction_check = {"verdict": verdict, "reason": reason}

            if verdict == "reversed":
                # Downgrade to mentions — we don't swap automatically since
                # the reverse edge may not exist as a separate record
                verdict_adjustment = "downgraded_reversed"
                norm["type"] = "mentions"
                norm["confidence"] = min(norm["confidence"], 0.5)
                norm["reason"] = f"[dir-check] direction reversed: {reason}"
            elif verdict == "unclear":
                verdict_adjustment = "downgraded_unclear"
                norm["confidence"] = min(norm["confidence"], 0.55)

    out.update({
        "type": norm["type"],
        "confidence": norm["confidence"],
        "reason": norm["reason"],
        "reason_quote": quote,
        "quote_verified": quote_ok,
        "direction_check": direction_check,
        "verdict_adjustment": verdict_adjustment,
    })
    return out


def _call_llm_v4(endpoint: str, model: str, system: str, prompt: str, timeout: int) -> dict | None:
    """Wrapper around v2's _call_llm that accepts a custom system prompt."""
    import json as _json
    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        endpoint,
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
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
    return _v2._extract_json(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--all-types", action="store_true")
    p.add_argument("--only-types", default=None)
    p.add_argument("--max-ctx-chars", type=int, default=DEFAULT_MAX_CTX_CHARS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    p.add_argument("--no-direction-check", action="store_true",
                   help="Skip step 3 (direction verification call). Useful for A/B test.")
    args = p.parse_args()

    if args.all_types and args.only_types:
        print("[error] --all-types and --only-types are mutually exclusive", file=sys.stderr)
        return 2

    data = _v2._load_relationships()
    all_edges = data.get("edges", [])
    if args.only_types:
        only_set = {t.strip() for t in args.only_types.split(",") if t.strip()}
        candidates = [e for e in all_edges if not e.get("expired_at") and e.get("type") in only_set]
    elif args.all_types:
        candidates = [e for e in all_edges if not e.get("expired_at")]
    else:
        candidates = [e for e in all_edges if not e.get("expired_at") and e.get("type") == "mentions"]

    print(f"[info] {len(candidates)} candidate edges")
    if args.sample:
        candidates = candidates[: args.sample]
    print(f"[info] will classify {len(candidates)} edges")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_fh = None
    if not args.dry_run:
        out_fh = args.output.open("a")

    t_start = time.perf_counter()
    ok = 0
    fail = 0
    vocab_counts: dict[str, int] = {t: 0 for t in VOCABULARY}
    adjust_counts: dict[str, int] = {}

    for i, edge in enumerate(candidates, 1):
        source = edge["source"]
        target = edge["target"]
        original_type = edge.get("type", "")
        context = _v2._load_fact_snippets(source, target, args.max_ctx_chars)

        t0 = time.perf_counter()
        result = classify_edge_v4(
            source, target, context,
            args.endpoint, args.model, args.timeout,
            skip_direction_check=args.no_direction_check,
        )
        dt_ms = (time.perf_counter() - t0) * 1000

        if result.get("type") not in VOCABULARY:
            fail += 1
            continue

        vocab_counts[result["type"]] += 1
        ok += 1
        adj = result.get("verdict_adjustment", "none")
        adjust_counts[adj] = adjust_counts.get(adj, 0) + 1

        record = {
            "source": source,
            "target": target,
            "resolved_src": result.get("resolved_src", source),
            "resolved_tgt": result.get("resolved_tgt", target),
            "original_type": original_type,
            "new_type": result["type"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "src_type_hint": result["src_type_hint"],
            "tgt_type_hint": result["tgt_type_hint"],
            "reason_quote": result["reason_quote"],
            "quote_verified": result["quote_verified"],
            "direction_check": result["direction_check"],
            "verdict_adjustment": adj,
            "classified_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "prompt_version": "v4",
        }
        if out_fh is not None:
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

        flag = ""
        if adj != "none":
            flag = f"[{adj}]"
        print(
            f"  [{i}/{len(candidates)}] {result['type']:<12} c={result['confidence']:.2f} "
            f"{dt_ms:5.0f}ms {source[:22]!r:<26}->{target[:22]!r:<26} {flag}"
        )

    if out_fh is not None:
        out_fh.close()

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 72)
    print(f"Classified: {ok} ok, {fail} failed in {elapsed:.1f}s")
    if ok > 0:
        print(f"Mean latency: {elapsed / ok * 1000:.0f} ms/edge (includes 2x LLM calls when direction check fires)")
    print("\nVocab distribution:")
    for t in VOCABULARY:
        c = vocab_counts[t]
        if c == 0:
            continue
        pct = (c / ok * 100) if ok else 0
        print(f"  {c:>4}  ({pct:5.1f}%)  {t}")
    print("\nVerdict adjustments:")
    for a, c in sorted(adjust_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:>4}  {a}")

    if not args.dry_run:
        print(f"\nWrote -> {args.output}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
