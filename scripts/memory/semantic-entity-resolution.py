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

    # Selection floor: candidates scoring below this are never judged.
    # Default 4.0 is where the weekly `--limit 2000` slice already lands;
    # 0 reopens the tail (which is 98.8 % of the pool — budget accordingly).
    .venvs/lloyd/bin/python scripts/memory/semantic-entity-resolution.py --min-score 0

    # Custom confidence thresholds
    .venvs/lloyd/bin/python scripts/memory/semantic-entity-resolution.py \\
        --merge-threshold 0.85 --alias-threshold 0.65
"""
from __future__ import annotations

import argparse
import hashlib
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

from app.kg_store import store as _kg_store  # noqa: E402

PIPELINE_ROOT = Path.home() / "lloyd" / "_pipeline" / "memory-graph"
CANDIDATE_LOG = PIPELINE_ROOT / f"semantic-entity-candidates-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
JUDGMENT_LOG = PIPELINE_ROOT / f"semantic-entity-judgments-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
PROPOSAL_LOG = PIPELINE_ROOT / f"semantic-proposals-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
# The accumulated record, deduped by (canonical, variant). `semantic-proposals-
# latest.jsonl` — the ONLY path the sweep reads — points at THIS file, not at one
# run's dated file. This pass is propose-only, so a proposal the sweep had not
# reached vanished whenever the next run overwrote its dated file (#744).
PROPOSAL_CUMULATIVE = PIPELINE_ROOT / "semantic-proposals-cumulative.jsonl"
PROPOSAL_LATEST = PIPELINE_ROOT / "semantic-proposals-latest.jsonl"
# Verdicts keyed on the pair plus a hash of both definitions, the same shape
# `entity_semantic_gate.SemanticGate._key` uses. This pass writes no aliases, so
# a pair stays in the pool after it is judged and every weekly run would
# re-judge what it had already settled; with this it only pays for pairs whose
# definitions actually changed.
VERDICT_CACHE = PIPELINE_ROOT / "semantic-verdicts-pairs.jsonl"

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

# Selection floor (#879). Measured on the live store the same way a run measures
# it: 567,123 candidate pairs, of which 5,235 score ≥ 4.0 — so 98.8 % of the pool
# is tail the run must not be charged for. `--limit 2000` cut at exactly 4.0
# before the floor existed, which means the line was already being drawn, just
# not stated, and a run whose head drained would have walked into the tail. The
# floor makes it a knob and makes the remaining work countable; `--min-score 0`
# reopens the tail on purpose.
DEFAULT_MIN_SCORE = 4.0

# Name shapes that are a note filename or a date-prefixed note rather than an
# entity (#729). Measured 2026-09-11 over the pre-filter pool: 48,771 pairs
# (8.61 %) had a `.md`-suffixed side, 19,421 (3.43 %) a date-shaped one, and
# 2,348 of the 6,863 pairs above the floor (34.2 %) were one of the two —
# `out.sort` put them at the very top, so they dominated every `--limit` slice
# and 40 of the 221 proposals the sweep was handed. A fresh generation with this
# filter drops 50,547 such pairs and yields 0. #537 attacks the same shapes on
# the write side; this is the read-side triage that stops paying for them
# meanwhile.
MD_SUFFIX = ".md"
DATE_NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")

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


def is_artifact_name(name: str) -> bool:
    """Is this 'entity' actually a note filename or a date-prefixed note?

    Extraction writes an entity for the note it came from, so the store holds
    both `knowledge-library` and `knowledge-library.md`, and both
    `2026-09-08-daily-note` and its sibling. A pair of those is not a duplicate
    claim about the world — it is one thing and a rendering of it — so judging
    it is a wasted LLM call and merging it is a wrong alias.
    """
    n = (name or "").strip()
    return n.endswith(MD_SUFFIX) or bool(DATE_NAME_PATTERN.match(n))


def is_artifact_pair(a: str, b: str) -> bool:
    """Either side artifact-shaped → drop at generation, before any LLM call.

    Dropping on EITHER side (not only the exact `X` vs `X.md` shape) is what the
    acceptance check measures, and it is the honest rule: an entity whose name is
    a filename should not be in a resolution pool at all.
    """
    return is_artifact_name(a) or is_artifact_name(b)


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
    """Lowercased surface → canonical, from the store. Read-only: this pass
    proposes, the sweep merges (see the skill; `--apply` retired 2026-09-04)."""
    return _kg_store().aliases.all_lower()


def load_graph() -> dict:
    return {"edges": _kg_store().edges.all()}


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
    dropped_artifacts = 0
    for (a, b) in pair_keys:
        if already_aliased(a, b):
            continue
        if a == b:
            continue
        # #729: artifact name shapes are dropped HERE, not judged. They also used
        # to sort to the very top (a `.md` twin is a near-perfect Jaccard match),
        # which is how they came to dominate every `--limit` slice.
        if is_artifact_pair(a, b):
            dropped_artifacts += 1
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
    if dropped_artifacts:
        print(f"[info] dropped {dropped_artifacts} artifact-shaped pair(s) "
              f"({MD_SUFFIX}-suffixed or date-named) before the judge")
    return out


def select_candidates(
    candidates: list[dict],
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    """What this run may judge, and how much eligible work is left.

    Two rules, both before the limit, so `--limit N` cuts the eligible head and a
    run can never spend budget below the floor:

    - the score floor — `--min-score`, default 4.0;
    - the artifact-name filter, re-applied here. Generation already drops those
      pairs, but `--from-candidates` loads a file written by an older run, and
      re-judging a stale pool is exactly how the budget goes back to being spent
      on `.md` twins.

    Returns `(selected, above_floor_total)`; `above_floor_total - len(selected)`
    is the eligible backlog left for later runs.
    """
    eligible = [c for c in candidates
                if float(c.get("score", 0.0)) >= min_score
                and not is_artifact_pair(str(c.get("a") or ""), str(c.get("b") or ""))]
    return (eligible[:limit] if limit else eligible), len(eligible)


def run_summary(newly_judged: int, from_cache: int, above_floor_total: int | None,
                selected: int, min_score: float) -> str:
    """One line: judged newly, judged free from cache, and eligible pairs left.

    Without the third number a run read as an unbounded backlog over a
    566,170-pair pool when what is actually left is a countable set just above
    the floor (#730). `None` means the run had no pool to measure — `--replay`.
    """
    remaining = ("unknown" if above_floor_total is None
                 else max(above_floor_total - selected, 0))
    return (f"[summary] newly_judged={newly_judged} from_cache={from_cache} "
            f"above_floor_remaining={remaining} (min_score={min_score})")


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


def _definition(entity: str) -> str:
    """The entity's own definition line, or its first facts as a stand-in."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import entity_semantic_gate
        return entity_semantic_gate.entity_definition(entity, FACTS_ROOT) or ""
    except Exception:
        return load_fact_snippets(entity, 300)


def _cache_key(a: str, b: str, da: str, db: str) -> str:
    """Pair + definition hash. Re-judged only when a definition changes."""
    lo, hi = sorted((a, b))
    da, db = (da, db) if lo == a else (db, da)
    return hashlib.sha256(f"{lo}\x00{hi}\x00{da}\x00{db}".encode("utf-8")).hexdigest()[:32]


def load_verdict_cache(path: Path = VERDICT_CACHE) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("key"):
                cache[rec["key"]] = rec
        except json.JSONDecodeError:
            continue
    return cache


def append_verdict(rec: dict, path: Path = VERDICT_CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


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


# ---------------------------------------------------------------------------
# Proposal emission — the durable record the sweep reads
# ---------------------------------------------------------------------------


def _proposal_key(rec: dict) -> tuple[str, str]:
    return (str(rec.get("canonical") or ""), str(rec.get("variant") or ""))


def filter_artifact_proposals(proposals: list[dict]) -> list[dict]:
    """#729 on the output side: never hand the sweep an artifact-named row.

    Generation already drops these, so a nonzero count here means proposals came
    in through `--from-candidates` or `--replay`, which bypass candidate
    generation — and the sweep would have acted on them.
    """
    kept: list[dict] = []
    dropped = 0
    for p in proposals:
        if is_artifact_pair(*_proposal_key(p)):
            dropped += 1
            continue
        kept.append(p)
    if dropped:
        print(f"[info] withheld {dropped} artifact-shaped proposal(s) from the record")
    return kept


def load_cumulative_proposals(path: Path) -> dict[tuple[str, str], dict]:
    """The accumulated record as (canonical, variant) → most recent proposal."""
    out: dict[tuple[str, str], dict] = {}
    path = Path(path)
    if not path.exists():
        return out
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = _proposal_key(rec)
        if all(key):
            out[key] = rec
    return out


def merge_proposals(existing: dict[tuple[str, str], dict],
                    new: list[dict], now_iso: str) -> list[dict]:
    """Union by (canonical, variant) — a re-stated proposal refreshes, never duplicates.

    `first_seen_at` is the run that surfaced it and `last_seen_at` the most
    recent: that gap is what tells a reader the same proposal has been shown to
    the sweep for six weeks without being acted on.
    """
    merged = dict(existing)
    for rec in new:
        key = _proposal_key(rec)
        if not all(key):
            continue
        rec = dict(rec)
        prior = merged.get(key) or {}
        rec["first_seen_at"] = (prior.get("first_seen_at") or prior.get("proposed_at")
                               or rec.get("proposed_at") or now_iso)
        rec["last_seen_at"] = rec.get("proposed_at") or now_iso
        merged[key] = rec
    return sorted(merged.values(),
                  key=lambda r: (-float(r.get("confidence") or 0), _proposal_key(r)))


def emit_proposals(proposals: list[dict], *, run_log: Path, cumulative: Path,
                   latest: Path, now_iso: str | None = None) -> dict:
    """Write this run's file, refresh the cumulative record, point `latest` at it.

    `latest` used to be re-pointed at the run's dated file, which is how an
    unconsumed proposal disappeared without a trace (#744). It now names the
    cumulative record, so the sweep's single loader path sees every proposal that
    is still open. Returns counts for the run report.
    """
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    run_log, cumulative, latest = Path(run_log), Path(cumulative), Path(latest)
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("w") as f:
        for rec in proposals:
            f.write(json.dumps(rec) + "\n")
    merged = merge_proposals(load_cumulative_proposals(cumulative), proposals, now_iso)
    tmp = cumulative.with_name(cumulative.name + ".tmp")
    with tmp.open("w") as f:
        for rec in merged:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(cumulative)
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(cumulative.name)
    except OSError:
        pass
    return {"run": len(proposals), "cumulative": len(merged)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
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
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                   help=f"candidate score floor for selection "
                        f"(default {DEFAULT_MIN_SCORE}; 0 reopens the tail)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

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

    cache_hits = 0
    newly_judged = 0
    above_floor_total: int | None = 0
    candidates: list[dict] = []

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
        above_floor_total = None  # a replay has no pool to measure
    elif args.from_candidates:
        pool = []
        with args.from_candidates.open() as f:
            for line in f:
                if line.strip():
                    pool.append(json.loads(line))
        print(f"[info] loaded {len(pool)} candidates from {args.from_candidates}")

        if args.skip_judge:
            return 0

        candidates, above_floor_total = select_candidates(pool, args.min_score, args.limit)
        print(f"[info] {above_floor_total} of them at or above min_score="
              f"{args.min_score}; judging {len(candidates)} of those")
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

        candidates, above_floor_total = select_candidates(candidates, args.min_score, args.limit)
        print(f"[info] {above_floor_total} of them at or above min_score="
              f"{args.min_score}; judging {len(candidates)} of those")

    # Judgment loop (skipped in replay mode)
    t_start = time.perf_counter()
    if not args.replay:
        judgments = []
        verdict_counts = Counter()
        cache = load_verdict_cache()
        for i, pair in enumerate(candidates, 1):
            t0 = time.perf_counter()
            da, db = _definition(pair["a"]), _definition(pair["b"])
            key = _cache_key(pair["a"], pair["b"], da, db)
            cached = cache.get(key)
            if cached is not None:
                cache_hits += 1
                judgments.append({**pair, **{k: v for k, v in cached.items() if k != "key"},
                                  "cached": True})
                verdict_counts[cached.get("verdict", "?")] += 1
                continue
            j = judge_pair(pair, args.endpoint, args.model, args.timeout)
            dt_ms = (time.perf_counter() - t0) * 1000
            if j is None or "error" in j:
                verdict_counts["error"] += 1
                print(f"  [{i}/{len(candidates)}] ERROR  {pair['a']!r} vs {pair['b']!r}: {j}")
                continue
            append_verdict({"key": key, "a": pair["a"], "b": pair["b"],
                            "judged_at": datetime.now(timezone.utc).isoformat(), **j})
            cache[key] = j
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
            newly_judged += 1

        with JUDGMENT_LOG.open("w") as f:
            for r in judgments:
                f.write(json.dumps(r) + "\n")
        print(f"\n[info] judgments → {JUDGMENT_LOG}")
        print(f"[info] elapsed: {time.perf_counter() - t_start:.1f}s")
        print(f"[info] verdicts: {dict(verdict_counts)}  ({cache_hits} from cache)")

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

    print("\nProposal summary:")
    print(f"  merge candidates:  {len(to_merge)}")
    print(f"  alias-only:        {len(to_alias)}")
    if guard_downgrades:
        print("  merge→alias downgrades by guard:")
        for k, n in guard_downgrades.most_common():
            print(f"    {k:<30} {n}")

    # This pass proposes; `entity-resolution-sweep.py --apply` merges. Its
    # own apply path was retired on 2026-09-04 because it bypassed the
    # sweep's degraded-graph gate, its invocation ledger, its fact retagging
    # and its revert path — the four things that made the 2026-09-03
    # 151-merge mistake recoverable.
    proposals = []
    for r, action in [(x, "merge") for x in to_merge] + [(x, "alias_only") for x in to_alias]:
        canonical, variant = pick_canonical(r["a"], r["b"], neighbors)
        proposals.append({
            "canonical": canonical, "variant": variant, "action": action,
            "verdict": r["verdict"], "confidence": r["confidence"],
            "reason": r.get("reason", ""), "guard_reason": r.get("guard_reason"),
            "cached": r.get("cached", False),
            "proposed_at": datetime.now(timezone.utc).isoformat(),
        })
    # No artifact-named row may reach the sweep from any path, including
    # `--from-candidates` / `--replay`, which skip candidate generation (#729).
    proposals = filter_artifact_proposals(proposals)
    proposals.sort(key=lambda p: -p["confidence"])
    counts = emit_proposals(proposals, run_log=PROPOSAL_LOG,
                            cumulative=PROPOSAL_CUMULATIVE, latest=PROPOSAL_LATEST)
    print(f"[info] {counts['run']} proposals → {PROPOSAL_LOG}")
    print(f"[info] {counts['cumulative']} total in the accumulated record; "
          f"{PROPOSAL_LATEST.name} → {PROPOSAL_CUMULATIVE.name}, so a proposal the "
          f"sweep has not reached cannot be lost by the next run")
    print("[info] no changes made. The sweep reads these as review input; "
          "run `entity-resolution-sweep.py` to see them in its plan.")
    print(run_summary(newly_judged, cache_hits, above_floor_total,
                      len(candidates), args.min_score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
