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
# entity_semantic_gate lives beside this file, not under the repo root above,
# and `_definition` imports it lazily. Add the sibling dir ONCE, guarded: the
# insert used to sit inside `_definition`, one duplicate entry per call, ~4000
# a weekly run (#745).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(1, _HERE)
from app.paths import VAULT_FACTS_ROOT as FACTS_ROOT
from app.paths import PIPELINE_DIR  # noqa: E402

from app.kg_store import store as _kg_store  # noqa: E402

PIPELINE_ROOT = PIPELINE_DIR / "memory-graph"
# The UTC date, like every `judged_at`/`proposed_at` written into these files.
# A naive local date named the 09-09 run's judgments `…-2026-09-08.jsonl` on a
# Pacific box, so the skill's dated success check failed on any run crossing
# local midnight (#1176). The skill checks `date -u +%F` to match.
RUN_DATE = datetime.now(timezone.utc).strftime('%Y-%m-%d')
CANDIDATE_LOG = PIPELINE_ROOT / f"semantic-entity-candidates-{RUN_DATE}.jsonl"
JUDGMENT_LOG = PIPELINE_ROOT / f"semantic-entity-judgments-{RUN_DATE}.jsonl"
PROPOSAL_LOG = PIPELINE_ROOT / f"semantic-proposals-{RUN_DATE}.jsonl"
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

CLASSIFIER_V2 = Path(__file__).resolve().parent / "classify-relationships.py"  # this tree's, never the live one
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
# Role nouns a suffixed instance hangs off a bare concept (#1175): `Browser` vs
# `Browser Tool`, `OpenClaw Cron` vs `OpenClaw Cron System`, `alfie_vr` vs
# `Alfie VR System`. A `same` verdict at merge confidence on such a pair is the
# conflation this guard list exists for even with no task number and no
# possessive in either name: one row is a recurring system, the other one of
# its task instances. `pick_canonical` breaks a facts+degree tie on the SHORTER
# name, so a merge here silently adopts whichever row happens to be terser —
# usually the bare concept absorbing the instance, and the reverse whenever the
# instance row carries more facts.
# Measured on the 2026-09-16 run's own proposals (183 merge rows): 39 (21 %)
# have this shape with `guard_reason: null`; 6 of those carry a role noun on the
# short side as well and are genuine aliases (`Tool MCP` vs `Tool MCP Service`),
# so 33 are guardable. Reproduced on 2026-09-08 at 41 of 156 (26 %).
MERGE_ROLE_TOKENS = {
    "system", "service", "agent", "sdk", "pipeline", "task", "loop", "app",
    "tool", "toolkit", "framework", "module", "component", "version",
}

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
    cache: dict[str, dict] | None = None,
) -> tuple[list[dict], int, int]:
    """What this run may judge, and how much eligible work is left.

    Three rules, all applied BEFORE the budget is spent, so that `--limit N` buys
    N NEW judgments:

    - the score floor — `--min-score`, default 4.0;
    - the artifact-name filter, re-applied here. Generation already drops those
      pairs, but `--from-candidates` loads a file written by an older run, and
      re-judging a stale pool is exactly how the budget goes back to being spent
      on `.md` twins.
    - the verdict cache, consulted *during* selection whenever a `limit` is set —
      `cache` is the loaded cache, or `None` for an empty one. It used to be
      consulted only inside the judgment loop, where a hit `continue`d while still
      occupying one of the N slots, so every already-judged pair at the head of the
      pool consumed budget, the slice never advanced, and the backlog drained at the
      rate definitions happened to change rather than at the rate the limit claims
      (#535).

    Returns `(selected, above_floor_total, cached_skipped)`. Eligible pairs left
    for later runs is `above_floor_total - len(selected) - cached_skipped`:
    `cached_skipped` comes off because those pairs are already judged, not still
    queued. Each selected pair carries the `_cache_key` computed here, so the
    judgment loop never reads either definition a second time.

    With no `limit` the cache is not consulted at all: there is no budget to
    protect, and dropping cached pairs from an unlimited slice would silently
    remove their verdicts from this run's proposal record.
    """
    eligible = [c for c in candidates
                if float(c.get("score", 0.0)) >= min_score
                and not is_artifact_pair(str(c.get("a") or ""), str(c.get("b") or ""))]
    if not limit:
        return eligible, len(eligible), 0

    cache = cache or {}  # one path whether a cache was supplied or not
    read_definition = _definition  # resolved per call, so a caller can stub it
    memo: dict[str, str] = {}

    def _def(name: str) -> str:
        # One definition read per entity, not per pair: a definition is a file
        # read, and the same names recur all the way down the list. Measured
        # 2026-09-15T23:20Z over the live 2026-09-08 pool: the 2,580 pairs a
        # `--limit 2000` selection walks (2,000 new + 580 cached) name 1,643
        # distinct entities, so this memo costs 1,643 reads where one per pair
        # side costs 5,160. Re-measure by counting `_definition` calls across
        # one selection — against the LIVE facts root. `app.paths` resolves
        # VAULT_FACTS_ROOT through LLOYD_DATA, which the gate and conftest point
        # at a scratch root with no facts in it; there every definition is ""
        # and the count is a count of nothing.
        if name not in memo:
            memo[name] = read_definition(name)
        return memo[name]

    selected: list[dict] = []
    taken_keys: set[str] = set()
    cached_skipped = 0
    for cand in eligible:
        if len(selected) >= limit:
            break
        a, b = str(cand.get("a") or ""), str(cand.get("b") or "")
        key = _cache_key(a, b, _def(a), _def(b))
        if key in cache or key in taken_keys:
            # `taken_keys` covers a pool that names the same pair twice, which is
            # likewise not new work (the 2026-09-08 pool has 0 such rows).
            cached_skipped += 1
            continue
        taken_keys.add(key)
        selected.append({**cand, "_cache_key": key})
    return selected, len(eligible), cached_skipped


def above_floor_remaining(above_floor_total: int | None, selected: int,
                          cached_skipped: int = 0) -> int | None:
    """Eligible pairs still queued after this run, or None when there was no pool.

    `None` is a replay, which never touches the pool: the summary prints that as
    `unknown` instead of inventing a backlog (#730). `cached_skipped` is not
    backlog — the verdict cache already holds those pairs — so it comes off here
    as well as out of `from_cache` (#535).
    """
    if above_floor_total is None:
        return None
    return max(above_floor_total - selected - cached_skipped, 0)


def run_summary(newly_judged: int, from_cache: int, above_floor_total: int | None,
                selected: int, min_score: float, cached_skipped: int = 0,
                compact: dict | None = None) -> str:
    """One line: judged newly, judged free from cache, and eligible pairs left.

    Without the third number a run read as an unbounded backlog over a
    566,170-pair pool when what is actually left is a countable set just above
    the floor (#730). `None` means the run had no pool to measure — `--replay`.
    `cached_skipped` is the count of pairs selection passed over as already
    judged, or as a pair the pool named twice (#535): they are served from cache,
    so `from_cache` includes them, and they are not backlog, so the remainder
    subtracts them. `compact` is `compact_verdict_cache`'s result, so the
    store's real size is on the same line as the cache hit rate it explains
    (#746); `None` on a replay, which judges nothing and compacts nothing.
    """
    remaining = above_floor_remaining(above_floor_total, selected, cached_skipped)
    line = (f"[summary] newly_judged={newly_judged} from_cache={from_cache} "
            f"above_floor_remaining={'unknown' if remaining is None else remaining} "
            f"(min_score={min_score})")
    if compact is not None:
        line += f" compact: kept {compact['kept']} dropped {compact['dropped']}"
    return line


def report_selection(above_floor_total: int, selected: int, cached_skipped: int,
                     limit: int | None, min_score: float) -> None:
    """The slice line: what was eligible, what this run will pay for, what it skipped.

    With `--limit` the line has to count UNcached pairs, since that is what the
    budget bought. `limited to first N candidates` over a list that had already
    been judged could not tell a new pair from a spent one, which is what hid
    #535: the run printed a full slice and bought 1,420 judgments with a
    2,000-judgment budget.
    """
    if limit:
        print(f"[info] {above_floor_total} candidates at or above min_score={min_score}; "
              f"limited to first {selected} uncached candidates (--limit {limit} = new "
              f"judgments; {cached_skipped} already-judged-or-duplicate pairs "
              f"skipped, not charged to the budget)")
    else:
        print(f"[info] {above_floor_total} of them at or above min_score={min_score}; "
              f"judging {selected} of those")


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
        import entity_semantic_gate
        return entity_semantic_gate.entity_definition(entity, FACTS_ROOT) or ""
    except Exception:
        return load_fact_snippets(entity, 300)


# The definition line comes from `<entity>-overview.md`, which the nightly
# extractor rewrites; keyed on the raw text, 0 of 40 verdicts written on
# 2026-09-23 still matched their key a day later (#728). Case, spacing and the
# long tail of the line are what the rewrite churns, so the key sees none of
# them; a substantively different definition still moves it.
DEFINITION_KEY_CHARS = 200


def canonical_definition(text: str) -> str:
    """The part of a definition the verdict-cache key is allowed to see."""
    return " ".join((text or "").lower().split())[:DEFINITION_KEY_CHARS]


def definition_hash(text: str) -> str:
    """Short hash of the canonical definition, stored on the verdict record so
    a later miss can say which side's definition moved."""
    return hashlib.sha256(canonical_definition(text).encode("utf-8")).hexdigest()[:8]


def _cache_key(a: str, b: str, da: str, db: str) -> str:
    """Pair + canonical definitions. Re-judged only when a definition changes."""
    lo, hi = sorted((a, b))
    da, db = (da, db) if lo == a else (db, da)
    da, db = canonical_definition(da), canonical_definition(db)
    return hashlib.sha256(f"{lo}\x00{hi}\x00{da}\x00{db}".encode("utf-8")).hexdigest()[:32]


def verdicts_by_pair(cache: dict[str, dict]) -> dict[frozenset, dict]:
    """The newest record per name pair, whatever its key — what a key miss is
    attributed against."""
    by_pair: dict[frozenset, dict] = {}
    for rec in cache.values():
        if rec.get("a") and rec.get("b"):
            by_pair[frozenset((rec["a"], rec["b"]))] = rec
    return by_pair


def attribute_miss(prior: dict | None, a: str, b: str, hash_a: str, hash_b: str) -> str:
    """Why a pair is being judged again.

    `new_pair` — never judged; `pre_728_record` — judged under the raw-text
    key, which carried no hashes; otherwise which definition(s) changed.
    A `reworded` verdict cannot arise: a rewording the canonical form absorbs
    is a cache hit, so it never reaches this function.
    """
    if prior is None:
        return "new_pair"
    prior_hashes = {prior.get("a"): prior.get("def_a"), prior.get("b"): prior.get("def_b")}
    if not (prior_hashes.get(a) and prior_hashes.get(b)):
        return "pre_728_record"
    changed = [name for name, now in ((a, hash_a), (b, hash_b)) if prior_hashes[name] != now]
    if len(changed) == 2:
        return "both_changed"
    return "a_changed" if changed == [a] else "b_changed"


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


def _sweep_prune_old_backups():
    """The sweep's `prune_old_backups`, loaded on demand (hyphenated filename).

    One definition of the `.bak` discipline, not a second copy of it. Loaded
    only when a compaction has written a backup, so the weekly run does not
    import the sweep (and its store) just to judge pairs.
    """
    path = Path(__file__).resolve().parent / "entity-resolution-sweep.py"
    spec = importlib.util.spec_from_file_location("entity_resolution_sweep", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prune_old_backups


def compact_verdict_cache(path: Path = VERDICT_CACHE, keep: int = 5) -> dict:
    """Rewrite the verdict cache keeping the newest row per unordered pair (#746).

    `append_verdict` only ever appends and `load_verdict_cache` reads the whole
    file, so since #535 made the slice advance the store grew by up to
    `--limit` rows a week with nothing giving space back — and a pair whose
    definition changed was appended again under a new key while the row under
    the old key stayed forever. The file is append-only, so file order is
    judgment order and the LAST row for a pair is the newest; the kept rows are
    written back in their original order, which keeps that true across
    compactions.

    Keyed on the pair, never on the definition hash: #728 (a key that hashes
    rewritten prose) is open, and a compaction that dropped rows whose
    recomputed key no longer matched would delete verdicts that are only stale
    because of that churn. A row that no key can ever hit again is dropped only
    when a newer verdict for the same pair exists. Called once at the end of a
    real run, never from the read path — a loader must not rewrite its input.

    A rewrite copies the file to `<name>.<stamp>.bak` first and prunes to
    `keep` backups the way the sweep does; nothing to drop means nothing is
    rewritten and no backup is taken. Returns ``{"kept", "dropped", "backup"}``.
    """
    if not path.exists():
        return {"kept": 0, "dropped": 0, "backup": None}
    rows: list[tuple[str, dict]] = []
    unreadable = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            unreadable += 1
            continue
        if not (isinstance(rec, dict) and rec.get("key") and rec.get("a") and rec.get("b")):
            unreadable += 1
            continue
        rows.append((line, rec))
    newest: dict[frozenset, int] = {}
    for i, (_line, rec) in enumerate(rows):
        newest[frozenset((rec["a"], rec["b"]))] = i
    survivors = set(newest.values())
    dropped = (len(rows) - len(survivors)) + unreadable
    if dropped == 0:
        return {"kept": len(rows), "dropped": 0, "backup": None}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for i, (line, _rec) in enumerate(rows)
                           if i in survivors), encoding="utf-8")
    tmp.replace(path)
    _sweep_prune_old_backups()(path, keep=keep)
    return {"kept": len(survivors), "dropped": dropped, "backup": backup}


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


def name_tokens(name: str) -> set[str]:
    """Every alphanumeric token, ROLE WORDS INCLUDED.

    Deliberately NOT `tokenize`: that one drops `NAME_STOPWORDS` — which is
    exactly where `system`/`tool`/`task` live — because Jaccard must not be
    diluted by them. This guard reads those words, so it needs its own
    tokenizer; sharing `tokenize` would make both names tokenize identically
    and the guard could never see the suffix it exists to catch.
    """
    return set(re.findall(r"[a-z0-9]+", (name or "").lower()))


def is_suffix_asymmetric(a: str, b: str) -> bool:
    """Is one name just the other plus role nouns? (#1175)

    Token-set containment, not `str.find`: the item's examples cross separators
    (`alfie_vr` vs `Alfie VR System`, `trajectory-extraction` vs
    `trajectory-extraction-system`), which no substring test normalises. Both
    orientations are tested, so the answer never depends on which name the
    proposal lists first; equal token sets are not a proper containment, so an
    identical pair is not asymmetric.
    """
    ta, tb = name_tokens(a), name_tokens(b)
    for small, big in ((ta, tb), (tb, ta)):
        if not small or not small < big:
            continue
        # The short side carries a role noun too → the suffix is part of BOTH
        # names' identity (`Tool MCP` vs `Tool MCP Service`, `Claude SDK Hooks`
        # vs `Claude Agent SDK Hooks`): a genuine alias, and the six rows the
        # non-fire clause of #1175 exists to protect.
        if small & MERGE_ROLE_TOKENS:
            continue
        if (big - small) <= MERGE_ROLE_TOKENS:
            return True
    return False


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
    # Bare concept vs its role-suffixed instance (#1175): `Browser` vs
    # `Browser Tool`. The two NAMES are the only evidence — no number, no
    # possessive, and facts/edges on both sides can be tiny — so this sits with
    # the other name-shape guards and ahead of the store reads below, which
    # makes the downgrade independent of what the facts root holds today.
    if is_suffix_asymmetric(a, b):
        return False, "suffix_asymmetry"

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


def accumulated_record_lost(cumulative: Path, existing: dict, run_log: Path) -> Path | None:
    """The one sound signal that the accumulated record was wiped (#1410).

    Every run writes its dated `semantic-proposals-<date>.jsonl` beside the
    cumulative file, so a dated file with rows next to a cumulative that is
    absent or holds none can only mean the record was lost: the first-ever run
    has no dated file, and a healthy record is at least as old as the newest
    one. Judged BEFORE this run writes its own dated file, which is excluded
    by name in case it already exists from an earlier run today — after the
    write, a wipe and a first run look identical. Returns the dated file that
    proves the loss, or None.
    """
    if cumulative.exists() and existing:
        return None
    dated = sorted(p for p in cumulative.parent.glob("semantic-proposals-????-??-??.jsonl")
                   if p.name != run_log.name and p.stat().st_size > 0)
    return dated[-1] if dated else None


def emit_proposals(proposals: list[dict], *, run_log: Path, cumulative: Path,
                   latest: Path, now_iso: str | None = None) -> dict:
    """Write this run's file, refresh the cumulative record, point `latest` at it.

    `latest` used to be re-pointed at the run's dated file, which is how an
    unconsumed proposal disappeared without a trace (#744). It now names the
    cumulative record, so the sweep's single loader path sees every proposal that
    is still open. Returns counts for the run report.

    A cumulative record that is missing is merged into as an empty one and the
    result is a healthy-looking total; on 2026-09-23 the data-root move had
    taken the record with it and the run reported 4 total as if that were the
    history (#1410). The merge still goes ahead — the run's proposals are
    real — but `reset` in the counts and the warning are what let the run
    report, and a human, know to restore from a snapshot rather than trust
    the total.
    """
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    run_log, cumulative, latest = Path(run_log), Path(cumulative), Path(latest)
    run_log.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cumulative_proposals(cumulative)
    lost_proof = accumulated_record_lost(cumulative, existing, run_log)
    if lost_proof is not None:
        print(f"[warn] accumulated record {cumulative} is absent or empty while "
              f"{lost_proof.name} holds earlier proposals — rebuilding it from "
              f"{len(existing)} rows; restore the record from a data snapshot "
              f"before the sweep acts on a partial one")
    with run_log.open("w") as f:
        for rec in proposals:
            f.write(json.dumps(rec) + "\n")
    merged = merge_proposals(existing, proposals, now_iso)
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
    return {"run": len(proposals), "cumulative": len(merged),
            "reset": lost_proof is not None, "reset_from": len(existing)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Budget of NEW judgments: takes the first N eligible "
                        "candidates whose verdict-cache key is absent, so cached "
                        "pairs do not consume the budget (#535).")
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
    # Budget bookkeeping for `--limit`: pairs passed over because the verdict
    # cache already holds them, and the cache itself, loaded once here so
    # selection and the judgment loop read the same snapshot.
    cached_skipped = 0
    cache: dict[str, dict] | None = None

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

        if args.limit:
            cache = load_verdict_cache()
        candidates, above_floor_total, cached_skipped = select_candidates(
            pool, args.min_score, args.limit, cache)
        report_selection(above_floor_total, len(candidates), cached_skipped,
                         args.limit, args.min_score)
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
            cache = load_verdict_cache()
        candidates, above_floor_total, cached_skipped = select_candidates(
            candidates, args.min_score, args.limit, cache)
        report_selection(above_floor_total, len(candidates), cached_skipped,
                         args.limit, args.min_score)

    # Judgment loop (skipped in replay mode)
    t_start = time.perf_counter()
    compact: dict | None = None
    if not args.replay:
        judgments = []
        verdict_counts = Counter()
        if cache is None:
            cache = load_verdict_cache()
        miss_reasons = Counter()
        by_pair = verdicts_by_pair(cache)
        for i, pair in enumerate(candidates, 1):
            t0 = time.perf_counter()
            # The key selection computed, when it computed one — re-deriving it
            # would read both definitions again for every pair.
            key = pair.pop("_cache_key", None)
            da = db = None
            if key is None:
                da, db = _definition(pair["a"]), _definition(pair["b"])
                key = _cache_key(pair["a"], pair["b"], da, db)
            cached = cache.get(key)
            if cached is not None:
                cache_hits += 1
                judgments.append({**pair, **{k: v for k, v in cached.items() if k != "key"},
                                  "cached": True})
                verdict_counts[cached.get("verdict", "?")] += 1
                continue
            if da is None:
                # Selection keyed this pair; the miss attribution needs the
                # definitions themselves (memoised, so this is not a re-read).
                da, db = _definition(pair["a"]), _definition(pair["b"])
            hash_a, hash_b = definition_hash(da), definition_hash(db)
            miss_reasons[attribute_miss(by_pair.get(frozenset((pair["a"], pair["b"]))),
                                        pair["a"], pair["b"], hash_a, hash_b)] += 1
            j = judge_pair(pair, args.endpoint, args.model, args.timeout)
            dt_ms = (time.perf_counter() - t0) * 1000
            if j is None or "error" in j:
                verdict_counts["error"] += 1
                print(f"  [{i}/{len(candidates)}] ERROR  {pair['a']!r} vs {pair['b']!r}: {j}")
                continue
            append_verdict({"key": key, "a": pair["a"], "b": pair["b"],
                            "def_a": hash_a, "def_b": hash_b,
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
        # A miss on a pair judged before is the cache decaying, and the report
        # says on which side the definition moved (#728).
        print(f"[info] cache misses by reason: {dict(miss_reasons)}")
        # After the appends, outside the loader: one row per pair from here on.
        compact = compact_verdict_cache()
        if compact["backup"] is not None:
            print(f"[info] verdict cache compacted: kept {compact['kept']} "
                  f"dropped {compact['dropped']} (backup {compact['backup'].name})")

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
    if counts.get("reset"):
        # Not an exit status: task #67 would go `failed` for a condition a
        # snapshot restore fixes. The run record has to say it, though — the
        # "total" line above reads healthy at 4 and at 4,000.
        print(f"[warn] the accumulated record was rebuilt from zero this run: that "
              f"total is this run's {counts['run']} proposals, not the history "
              f"(#1410). Restore {PROPOSAL_CUMULATIVE.name} from ~/.lloyd-data-snapshots "
              f"(scripts/backup/restore-data.sh) and re-run.")
    print("[info] no changes made. The sweep reads these as review input; "
          "run `entity-resolution-sweep.py` to see them in its plan.")
    # #535's three reports, in the words the item uses, printed beside #879's
    # `[summary]` line rather than replacing it. `cached` is every pair this run
    # did NOT pay an LLM call for — passed over while filling the slice plus any
    # hit inside the loop — and the split beside it is what shows the budget went
    # to new work instead of to pairs already judged.
    remaining = above_floor_remaining(above_floor_total, len(candidates), cached_skipped)
    print(f"[info] run: judged={newly_judged} cached={cache_hits + cached_skipped} "
          f"({cached_skipped} skipped while filling the slice, {cache_hits} served "
          f"in-loop) remaining-above-floor="
          f"{'unknown' if remaining is None else remaining}")
    print(run_summary(newly_judged, cache_hits + cached_skipped, above_floor_total,
                      len(candidates), args.min_score, cached_skipped,
                      compact=compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
