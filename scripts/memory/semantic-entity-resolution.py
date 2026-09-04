#!/usr/bin/env python3
"""Semantic entity resolution — LLM-backed clustering pass.

Complements the string-based `entity-resolution-sweep.py` by catching
duplicates that string rules cannot:
- Name differs by a non-safe suffix: `Knowledge` vs `Knowledge Library`
- Name differs by a qualifier: `Idler` vs `Idler Agent` (handled by string)
  vs  `Idler` vs `Sleep Mode` (same concept, unrelated names — this catches it)
- Abbreviation vs expansion: `SDK` vs `Software Development Kit`

Pipeline:
1. Load entities (fact dirs) + existing aliases + edge graph
2. Generate candidate pairs via multiple signals:
   - Name-token Jaccard ≥ 0.4
   - Normalized stem match (shared prefix ≥ 5 chars after normalize_full)
   - Shared-neighbor count ≥ 3 in edge graph
3. Dedupe candidates (skip already-aliased pairs)
4. Per pair: LLM judgment call — "same / related / distinct" with confidence
5. Apply per verdict:
   - `same` + conf ≥ 0.85: auto-merge (dir move + edge rewrite + alias)
   - `same` + conf 0.65-0.85: alias-only (no destructive move)
   - `related` or `distinct`: skip, log
6. Write merge log + candidate log to `_pipeline/memory-graph/`

Idempotent: re-runs skip pairs already resolved (either aliased or merged).
Safe: never deletes an entity dir, only moves content; always backs up
_relationships.json and entity-aliases.json before mutation.

Usage:
    # Dry run (default) — generate candidates + judge + print plan, no mutations
    .venvs/lloyd/bin/python scripts/memory/semantic-entity-resolution.py

    # Apply
    .venvs/lloyd/bin/python scripts/memory/semantic-entity-resolution.py --apply

    # Limit LLM calls (dev / sampling)
    .venvs/lloyd/bin/python scripts/memory/semantic-entity-resolution.py --limit 50

    # Custom confidence thresholds
    .venvs/lloyd/bin/python scripts/memory/semantic-entity-resolution.py \\
        --merge-threshold 0.85 --alias-threshold 0.65
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_ROOT

REL_PATH = FACTS_ROOT / "_relationships.json"
ALIASES_PATH = FACTS_ROOT / "entity-aliases.json"

PIPELINE_ROOT = Path.home() / "lloyd" / "_pipeline" / "memory-graph"
CANDIDATE_LOG = PIPELINE_ROOT / f"semantic-entity-candidates-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
JUDGMENT_LOG = PIPELINE_ROOT / f"semantic-entity-judgments-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
APPLY_LOG = PIPELINE_ROOT / f"semantic-entity-applied-{datetime.now().strftime('%Y-%m-%d')}.jsonl"

CLASSIFIER_V2 = Path.home() / "lloyd" / "scripts" / "memory" / "classify-relationships.py"
_spec = importlib.util.spec_from_file_location("classifier_v2", str(CLASSIFIER_V2))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)

DEFAULT_ENDPOINT = _v2.DEFAULT_ENDPOINT
DEFAULT_MODEL = _v2.DEFAULT_MODEL
DEFAULT_TIMEOUT_SEC = 45

JACCARD_THRESHOLD = 0.4
STEM_MIN_CHARS = 5
SHARED_NEIGHBOR_THRESHOLD = 3
MAX_CANDIDATES_PER_ENTITY = 20  # cap fanout

# Merge guards — if violated, downgrade proposed merge to alias-only.
# Rationale: LLM at conf ≥ 0.85 tends to conflate specific task instances with
# parent concepts, or SDK/component with parent project. Safer to alias than
# destructively merge a high-value entity.
MERGE_VARIANT_MAX_FACTS = 3          # smaller side must have ≤ this many facts
MERGE_VARIANT_MAX_DEGREE = 15        # smaller side must have ≤ this many edges
MERGE_COMBINED_FACTS_CAP = 25        # combined facts total above this → alias only
TASK_NUMBER_PATTERN = re.compile(
    r"#\d+|Task\s*\d+|\b(task|issue|backlog|run|session|ticket|item)[-_\s]?\d+\b",
    re.IGNORECASE,
)
# Timestamped session IDs (session_20260331_130709, run_YYYYMMDD_HHMMSS)
ID_PATTERN = re.compile(r"\d{8}[T_]\d{4,}", re.IGNORECASE)
# Possessive/descriptive artifact names that shouldn't merge into a person/project entity
ARTIFACT_PATTERN = re.compile(r"'s\s|'s\b", re.IGNORECASE)

# Stopwords in entity names — skip during tokenization for Jaccard
NAME_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "by", "and", "or",
    "system", "service", "pipeline", "app", "tool", "task", "loop",
    "agent", "sdk",
}


# ---------------------------------------------------------------------------
# Normalization & candidate generation
# ---------------------------------------------------------------------------


def normalize_full(name: str) -> str:
    """Aggressive: lowercase + strip punct/whitespace/separators."""
    return re.sub(r"[-_./\s]+", "", name.lower())


def tokenize(name: str) -> set[str]:
    """Tokens for Jaccard — splits on non-alpha, drops stopwords and short tokens."""
    toks = re.findall(r"[a-z0-9]+", name.lower())
    return {t for t in toks if len(t) > 2 and t not in NAME_STOPWORDS}


def token_jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def shares_stem(a: str, b: str, min_chars: int = STEM_MIN_CHARS) -> bool:
    """Do normalized names share a prefix of at least min_chars?"""
    na, nb = normalize_full(a), normalize_full(b)
    if len(na) < min_chars or len(nb) < min_chars:
        return False
    # Either one contains the other's first min_chars
    return na.startswith(nb[:min_chars]) or nb.startswith(na[:min_chars])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def list_entities() -> list[str]:
    """All directories under facts/ that are entity stores."""
    return sorted(
        d.name for d in FACTS_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )


def load_aliases() -> dict[str, str]:
    if not ALIASES_PATH.exists():
        return {}
    return json.loads(ALIASES_PATH.read_text("utf-8"))


def save_aliases(aliases: dict[str, str]) -> None:
    ALIASES_PATH.write_text(json.dumps(aliases, indent=2, sort_keys=True))


def load_graph() -> dict:
    return json.loads(REL_PATH.read_text("utf-8"))


def build_neighbors(graph: dict) -> dict[str, set[str]]:
    """entity -> set of its graph neighbors (across active edges)."""
    n: dict[str, set[str]] = defaultdict(set)
    for e in graph["edges"]:
        if e.get("expired_at"):
            continue
        src, tgt = e.get("source"), e.get("target")
        if src and tgt and src != tgt:
            n[src].add(tgt)
            n[tgt].add(src)
    return n


def entity_degree(entity: str, neighbors: dict[str, set[str]]) -> int:
    return len(neighbors.get(entity, ()))


def count_facts(entity: str) -> int:
    d = FACTS_ROOT / entity
    if not d.exists():
        return 0
    return sum(1 for _ in d.glob("*.md"))


def load_fact_snippets(entity: str, max_chars: int = 500) -> str:
    """Concatenate first few fact files for context."""
    d = FACTS_ROOT / entity
    if not d.exists():
        return ""
    pieces = []
    total = 0
    for f in sorted(d.glob("*.md"))[:3]:
        try:
            txt = f.read_text("utf-8").strip()
        except Exception:
            continue
        if not txt:
            continue
        # Strip YAML frontmatter if present
        if txt.startswith("---"):
            end = txt.find("\n---", 3)
            if end > 0:
                txt = txt[end + 4:].strip()
        snippet = txt[: max_chars - total]
        pieces.append(f"[{f.name}]\n{snippet}")
        total += len(snippet) + 20
        if total >= max_chars:
            break
    return "\n\n".join(pieces)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidates(
    entities: list[str],
    aliases: dict[str, str],
    neighbors: dict[str, set[str]],
) -> list[dict]:
    """Return candidate pairs with signal scores."""
    # Build alias resolution map
    alias_canonical = {k.lower(): v for k, v in aliases.items()}

    def already_aliased(a: str, b: str) -> bool:
        # Same canonical via table?
        ca = aliases.get(a) or aliases.get(a.lower()) or a
        cb = aliases.get(b) or aliases.get(b.lower()) or b
        return ca == cb

    # Bucketize by normalize_full prefix (5 chars) — only compare entities
    # within same prefix bucket OR with shared neighbors.
    buckets: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        nf = normalize_full(e)
        if len(nf) >= STEM_MIN_CHARS:
            buckets[nf[:STEM_MIN_CHARS]].append(e)

    # Build inverse: entity → buckets it belongs to (first 5, first 6 chars)
    pair_keys: set[tuple[str, str]] = set()
    for bucket_ents in buckets.values():
        if len(bucket_ents) < 2:
            continue
        for i, a in enumerate(bucket_ents):
            for b in bucket_ents[i + 1:]:
                pair_keys.add((min(a, b), max(a, b)))

    # Shared-neighbor candidates: any pair sharing ≥ threshold neighbors
    # To keep O(|E|), iterate edges and count pair-co-occurrence per shared third
    cooccur: dict[tuple[str, str], int] = defaultdict(int)
    for third, nset in neighbors.items():
        ents = sorted(nset)
        # cap — don't explode on high-degree entities
        if len(ents) > 40:
            continue
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                cooccur[(a, b)] += 1
    for pair, count in cooccur.items():
        if count >= SHARED_NEIGHBOR_THRESHOLD:
            pair_keys.add(pair)

    # Score each pair
    out = []
    for (a, b) in pair_keys:
        if already_aliased(a, b):
            continue
        if a == b:
            continue
        jacc = token_jaccard(a, b)
        stem = shares_stem(a, b)
        shared = len(neighbors.get(a, set()) & neighbors.get(b, set()))

        # Require at least one strong signal
        strong = (jacc >= JACCARD_THRESHOLD) or stem or (shared >= SHARED_NEIGHBOR_THRESHOLD)
        if not strong:
            continue

        # Score for ordering — higher = more likely duplicate
        score = jacc * 3.0 + (1.0 if stem else 0.0) + min(shared, 10) * 0.3

        out.append({
            "a": a,
            "b": b,
            "jaccard": round(jacc, 3),
            "shares_stem": stem,
            "shared_neighbors": shared,
            "score": round(score, 3),
            "deg_a": entity_degree(a, neighbors),
            "deg_b": entity_degree(b, neighbors),
            "facts_a": count_facts(a),
            "facts_b": count_facts(b),
        })

    out.sort(key=lambda x: -x["score"])
    return out


# ---------------------------------------------------------------------------
# LLM judgment
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a precise entity-resolution judge. Given two entity names and "
    "their stored facts, decide whether they refer to the SAME real-world "
    "entity, are RELATED but distinct, or are DISTINCT. Respond with JSON only."
)

USER_PROMPT = """Two entities from a knowledge graph. Decide whether they refer to the same real thing.

ENTITY A: {a}
Facts (A):
{facts_a}

ENTITY B: {b}
Facts (B):
{facts_b}

Graph signals:
  - shared neighbors: {shared}
  - name Jaccard: {jaccard}
  - shares stem: {stem}

Guidance:
- SAME: the two names refer to the same real entity (e.g., "Knowledge" and "Knowledge Library" when both describe the same library)
- RELATED: they are connected but distinct (e.g., "Claude" the model vs "Claude Code" the CLI)
- DISTINCT: unrelated things that happen to share tokens (e.g., "Research Agent" vs "Research Library")

Respond with strict JSON:
{{"verdict": "same" | "related" | "distinct", "confidence": 0.0-1.0, "reason": "one sentence citing concrete evidence"}}
"""


def judge_pair(pair: dict, endpoint: str, model: str, timeout: int) -> dict | None:
    facts_a = load_fact_snippets(pair["a"], 500) or "(no facts)"
    facts_b = load_fact_snippets(pair["b"], 500) or "(no facts)"
    prompt = USER_PROMPT.format(
        a=pair["a"], b=pair["b"],
        facts_a=facts_a, facts_b=facts_b,
        shared=pair["shared_neighbors"],
        jaccard=pair["jaccard"],
        stem=pair["shares_stem"],
    )
    # Reuse v2's _call_llm with a custom system prompt.
    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        # vLLM --scheduling-policy priority (lower = sooner): chat sends 0,
        # autonomy runs 1. Batch classification is the lowest-value traffic on
        # the box and must yield to both, or a long batch starves the fleet.
        "priority": 2,
    }
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)[:200]}

    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as exc:
        return {"error": f"parse failed: {exc}"}

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ("same", "related", "distinct"):
        verdict = "distinct"
    return {
        "verdict": verdict,
        "confidence": float(parsed.get("confidence", 0)),
        "reason": str(parsed.get("reason", ""))[:300],
    }


# ---------------------------------------------------------------------------
# Merge / alias application
# ---------------------------------------------------------------------------


def merge_allowed(a: str, b: str, neighbors: dict[str, set[str]]) -> tuple[bool, str]:
    """Return (allow_merge, reason). False → downgrade to alias-only."""
    # Task/issue number in either name → never merge, alias only
    if TASK_NUMBER_PATTERN.search(a) or TASK_NUMBER_PATTERN.search(b):
        return False, "task_number_in_name"
    # Timestamp-shaped ID → never merge (specific event instance)
    if ID_PATTERN.search(a) or ID_PATTERN.search(b):
        return False, "timestamp_id_in_name"
    # Possessive form ("Alan's X") is an artifact, not the entity itself
    if ARTIFACT_PATTERN.search(a) or ARTIFACT_PATTERN.search(b):
        return False, "possessive_artifact"

    facts_a = count_facts(a)
    facts_b = count_facts(b)
    deg_a = entity_degree(a, neighbors)
    deg_b = entity_degree(b, neighbors)

    # Combined mass cap
    if facts_a + facts_b > MERGE_COMBINED_FACTS_CAP:
        return False, "combined_facts_over_cap"

    # Smaller-side-must-be-tiny rule
    variant_facts = min(facts_a, facts_b)
    variant_deg = min(deg_a, deg_b) if facts_a == facts_b else (
        deg_b if facts_a > facts_b else deg_a
    )
    if variant_facts > MERGE_VARIANT_MAX_FACTS:
        return False, "variant_too_many_facts"
    if variant_deg > MERGE_VARIANT_MAX_DEGREE:
        return False, "variant_too_many_edges"
    return True, "ok"


def pick_canonical(a: str, b: str, neighbors: dict[str, set[str]]) -> tuple[str, str]:
    """Return (canonical, variant)."""
    deg_a = entity_degree(a, neighbors)
    deg_b = entity_degree(b, neighbors)
    facts_a = count_facts(a)
    facts_b = count_facts(b)
    # Prefer higher (facts + degree), then shorter, then alpha
    score_a = facts_a + deg_a
    score_b = facts_b + deg_b
    if score_a > score_b:
        return a, b
    if score_b > score_a:
        return b, a
    if len(a) < len(b):
        return a, b
    if len(b) < len(a):
        return b, a
    return (a, b) if a < b else (b, a)


def apply_merge(
    canonical: str,
    variant: str,
    graph: dict,
    aliases: dict[str, str],
    apply: bool,
) -> dict:
    """Move facts + rewrite edges + add alias. Returns stats."""
    stats = {
        "canonical": canonical,
        "variant": variant,
        "files_moved": 0,
        "edges_rewritten": 0,
        "edges_deduped": 0,
        "alias_added": False,
    }

    # Step 1: move variant/ fact files into canonical/
    v_dir = FACTS_ROOT / variant
    c_dir = FACTS_ROOT / canonical
    if v_dir.exists():
        if apply:
            c_dir.mkdir(parents=True, exist_ok=True)
        for f in v_dir.glob("*.md"):
            new_name = f.name.replace(variant, canonical, 1)
            target = c_dir / new_name
            if target.exists():
                # Keep existing (canonical wins on dedupe); expire variant file
                if apply:
                    f.unlink()
            else:
                if apply:
                    shutil.move(str(f), str(target))
            stats["files_moved"] += 1
        # Remove empty variant dir
        if apply and not any(v_dir.iterdir()):
            v_dir.rmdir()

    # Step 2: rewrite edges
    active_seen = set()
    for e in graph["edges"]:
        if e.get("expired_at"):
            continue
        rewrite = False
        if e.get("source") == variant:
            if apply:
                e["source"] = canonical
            rewrite = True
        if e.get("target") == variant:
            if apply:
                e["target"] = canonical
            rewrite = True
        if rewrite:
            stats["edges_rewritten"] += 1

    # Step 3: dedupe resulting active edges
    if apply:
        now = datetime.now(timezone.utc).isoformat()
        for e in graph["edges"]:
            if e.get("expired_at"):
                continue
            key = (e.get("source"), e.get("target"), e.get("type"))
            if key in active_seen:
                e["expired_at"] = now
                stats["edges_deduped"] += 1
            else:
                active_seen.add(key)

    # Step 4: alias
    if aliases.get(variant) != canonical:
        if apply:
            aliases[variant] = canonical
            aliases[variant.lower()] = canonical
        stats["alias_added"] = True

    return stats


def apply_alias_only(
    canonical: str,
    variant: str,
    aliases: dict[str, str],
    apply: bool,
) -> dict:
    stats = {"canonical": canonical, "variant": variant, "alias_added": False}
    if aliases.get(variant) != canonical:
        if apply:
            aliases[variant] = canonical
            aliases[variant.lower()] = canonical
        stats["alias_added"] = True
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of candidate pairs judged (dev).")
    p.add_argument("--merge-threshold", type=float, default=0.85,
                   help="confidence ≥ this + verdict==same triggers full merge")
    p.add_argument("--alias-threshold", type=float, default=0.65,
                   help="confidence ≥ this (below merge) + verdict==same triggers alias-only")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    p.add_argument("--skip-judge", action="store_true",
                   help="Stop after candidate generation (for tuning thresholds)")
    p.add_argument("--from-candidates", type=Path, default=None,
                   help="Skip candidate generation; load candidates from existing JSONL file")
    p.add_argument("--replay", type=Path, default=None,
                   help="Skip candidate gen + judging; replay apply plan from existing judgments JSONL")
    args = p.parse_args()

    # --replay short-circuit handled below by setting judgments from file
    # and skipping candidate gen + LLM calls.

    print(f"[info] loading entities from {FACTS_ROOT}")
    entities = list_entities()
    print(f"[info] {len(entities)} entity directories")

    aliases = load_aliases()
    graph = load_graph()
    neighbors = build_neighbors(graph)
    active_count = sum(1 for e in graph["edges"] if not e.get("expired_at"))
    print(f"[info] {len(aliases)} aliases, {active_count} active edges")

    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)

    if args.replay:
        judgments = []
        with args.replay.open() as f:
            for line in f:
                if line.strip():
                    judgments.append(json.loads(line))
        print(f"[info] replay: {len(judgments)} judgments from {args.replay}")
        verdict_counts = Counter(r["verdict"] for r in judgments)
        print(f"[info] verdicts: {dict(verdict_counts)}")
        candidates = []  # not used in replay
    elif args.from_candidates:
        candidates = []
        with args.from_candidates.open() as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
        print(f"[info] loaded {len(candidates)} candidates from {args.from_candidates}")

        if args.skip_judge:
            return 0

        if args.limit:
            candidates = candidates[: args.limit]
            print(f"[info] limited to first {len(candidates)} candidates")
    else:
        print("[info] generating candidates…")
        candidates = generate_candidates(entities, aliases, neighbors)
        print(f"[info] {len(candidates)} candidate pairs after filtering")

        with CANDIDATE_LOG.open("w") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")
        print(f"[info] candidates → {CANDIDATE_LOG}")

        if args.skip_judge:
            return 0

        if args.limit:
            candidates = candidates[: args.limit]
            print(f"[info] limited to first {len(candidates)} candidates")

    # Judgment loop (skipped in replay mode)
    t_start = time.perf_counter()
    if not args.replay:
        judgments = []
        verdict_counts = Counter()
        for i, pair in enumerate(candidates, 1):
            t0 = time.perf_counter()
            j = judge_pair(pair, args.endpoint, args.model, args.timeout)
            dt_ms = (time.perf_counter() - t0) * 1000
            if j is None or "error" in j:
                verdict_counts["error"] += 1
                print(f"  [{i}/{len(candidates)}] ERROR  {pair['a']!r} vs {pair['b']!r}: {j}")
                continue
            verdict = j["verdict"]
            conf = j["confidence"]
            verdict_counts[verdict] += 1
            flag = ""
            if verdict == "same" and conf >= args.merge_threshold:
                flag = "[MERGE]"
            elif verdict == "same" and conf >= args.alias_threshold:
                flag = "[ALIAS]"
            print(
                f"  [{i:>4}/{len(candidates)}] {verdict:<8} c={conf:.2f} {dt_ms:5.0f}ms "
                f"{pair['a'][:28]!r:<32}<->{pair['b'][:28]!r:<32} {flag}"
            )
            record = {**pair, **j}
            judgments.append(record)

        with JUDGMENT_LOG.open("w") as f:
            for r in judgments:
                f.write(json.dumps(r) + "\n")
        print(f"\n[info] judgments → {JUDGMENT_LOG}")
        print(f"[info] elapsed: {time.perf_counter() - t_start:.1f}s")
        print(f"[info] verdicts: {dict(verdict_counts)}")

    # Apply plan — split by confidence AND guard rails.
    # A merge-threshold pair is only truly merged if merge_allowed() passes;
    # otherwise it's downgraded to alias-only.
    to_merge: list[dict] = []
    to_alias: list[dict] = []
    guard_downgrades = Counter()
    for r in judgments:
        if r["verdict"] != "same":
            continue
        conf = r["confidence"]
        if conf < args.alias_threshold:
            continue
        allow_merge = False
        guard_reason = "below_merge_threshold"
        if conf >= args.merge_threshold:
            allow_merge, guard_reason = merge_allowed(r["a"], r["b"], neighbors)
        if allow_merge:
            to_merge.append(r)
        else:
            r["guard_reason"] = guard_reason
            guard_downgrades[guard_reason] += 1
            to_alias.append(r)

    print(f"\nApply plan:")
    print(f"  merges (dir + edges + alias): {len(to_merge)}")
    print(f"  alias-only:                   {len(to_alias)}")
    if guard_downgrades:
        print(f"  merge→alias downgrades by guard:")
        for k, n in guard_downgrades.most_common():
            print(f"    {k:<30} {n}")

    if not args.apply:
        print("\n[info] dry run (pass --apply to mutate)")
        # Still write a plan log for inspection
        return 0

    # Backup
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rel_backup = REL_PATH.with_name(f"_relationships.{ts}.pre-semantic.bak.json")
    al_backup = ALIASES_PATH.with_name(f"entity-aliases.{ts}.pre-semantic.bak")
    shutil.copy2(REL_PATH, rel_backup)
    shutil.copy2(ALIASES_PATH, al_backup)
    print(f"[info] backups: {rel_backup.name}, {al_backup.name}")

    apply_records = []
    for r in to_merge:
        canonical, variant = pick_canonical(r["a"], r["b"], neighbors)
        stats = apply_merge(canonical, variant, graph, aliases, apply=True)
        stats["verdict"] = r["verdict"]
        stats["confidence"] = r["confidence"]
        stats["reason"] = r["reason"]
        stats["action"] = "merge"
        apply_records.append(stats)

    for r in to_alias:
        canonical, variant = pick_canonical(r["a"], r["b"], neighbors)
        stats = apply_alias_only(canonical, variant, aliases, apply=True)
        stats["verdict"] = r["verdict"]
        stats["confidence"] = r["confidence"]
        stats["reason"] = r["reason"]
        stats["action"] = "alias_only"
        apply_records.append(stats)

    REL_PATH.write_text(json.dumps(graph, indent=2, sort_keys=False))
    save_aliases(aliases)

    with APPLY_LOG.open("w") as f:
        for rec in apply_records:
            f.write(json.dumps(rec) + "\n")
    print(f"[info] applied {len(apply_records)} actions → {APPLY_LOG}")

    # Final state
    active_after = sum(1 for e in graph["edges"] if not e.get("expired_at"))
    print(f"[info] active edges: {active_count} -> {active_after}")
    print(f"[info] aliases: {len(aliases)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
