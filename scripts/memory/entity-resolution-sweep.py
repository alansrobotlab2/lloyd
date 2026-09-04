#!/usr/bin/env python3
"""
Entity Resolution Sweep — Tier 1 (mechanical) for backlog #310.

Reads the live knowledge graph and entity directories, clusters entities by
normalized name, classifies each cluster as auto-mergeable (CASE / PUNCT /
SUFFIX) or ambiguous (LOOP / RESEARCH / OTHER), and either prints a plan
(dry-run) or applies the merges.

Apply mode:
  1. Backs up the store (SQLite backup API — consistent under writers).
  2. In ONE transaction, for each SAFE merge:
       a. Adds  {variant: canonical}  to the alias table.
       b. Rewrites edges through `edges.rewrite_endpoint`: every active edge
          touching the variant is expired and re-added on the canonical, and
          the (old_id, new_id) pairs are recorded so a revert is exact.
     Either every merge in the run lands or none does — the JSON era wrote
     aliases and edges as two separate whole-file rewrites, and a crash
     between them left the tree half-merged (2026-09-03).
  3. Moves fact files from variant dir into canonical dir (rename prefix),
     retags them, removes the empty variant dir.

Ambiguous clusters are dumped to a review JSONL for Tier 2 hand-review.

Usage:
  # dry-run (default):
  python entity-resolution-sweep.py

  # apply Tier 1:
  python entity-resolution-sweep.py --apply

  # also drop inherited alias entries that are pipeline noise:
  python entity-resolution-sweep.py --apply --rebuild-aliases
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# Ensure app/ is importable when running this script standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────

from app.paths import VAULT_FACTS_ROOT as FACTS_ROOT, VAULT_KG_DB
from app.entity_naming import looks_like_junk_entity
from app.kg_store import KGStore
from app.atomic_io import atomic_write_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _invocation import invocation_ledger  # noqa: E402

OUT_DIR = Path.home() / "lloyd" / "_pipeline" / "memory-graph"
BASELINE_PATH = OUT_DIR / "graph-baseline.json"
# --apply refuses when the graph holds less than this fraction of the largest
# active-edge count ever recorded here, unless --allow-degraded. On 2026-09-03
# an apply ran against a 2-edge graph (baseline 7,260): every entity had degree
# 0, so every suffix pair passed the "a variant has 0 degree" shortcut, and 151
# semantic conflations were merged with no check at all.
DEGRADED_FRACTION = 0.5
ALL_TIERS = ("CASE", "PUNCT", "SUFFIX_SAFE")

# Tier → the alias `kind` the store records, so a later reader can tell a
# safe case-fold from a judged semantic merge without re-deriving it.
TIER_ALIAS_KIND = {"CASE": "case", "PUNCT": "punct", "SUFFIX_SAFE": "suffix",
                   "SUFFIX_AMBIGUOUS": "suffix", "IDENTICAL": "case", "OTHER": "semantic"}

# ── Normalization ────────────────────────────────────────────────────────────

STOP_SUFFIX_TOKENS_SAFE = {
    # Stripping these suffixes CLUSTERS candidates; it does not decide identity.
    # `Intel Pipeline` vs `Intel`, `Fact System` vs `FACT` (a robotics action
    # tokenizer), `Alfie pipeline` vs `Alfie` (the robot) all cluster here and are
    # all different things. A SUFFIX_SAFE cluster merges only when the semantic
    # gate (entity_semantic_gate.py) says every judge agrees the definitions
    # describe the same thing.
    "system",
    "agent",
    "sdk",
    "service",
    "pipeline",
    "app",
}
STOP_SUFFIX_TOKENS_AMBIGUOUS = {
    # These are often legitimately distinct sub-entities.
    "loop",
    "research",
    "tool",
    "task",
    "bot",
}


def tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def normalize_full(name: str) -> str:
    """Most aggressive normalization: lowercase, strip non-alnum, drop ALL stop suffixes.

    Used for *clustering* — grouping candidates that might be the same thing.
    Not used for semantic equivalence decisions.
    Never strips the last remaining token — a single-word entity whose name
    IS a suffix token (e.g. 'agent', 'System', 'SDK') must not normalize to empty.
    """
    toks = tokens(name)
    while len(toks) > 1 and (toks[-1] in STOP_SUFFIX_TOKENS_SAFE or toks[-1] in STOP_SUFFIX_TOKENS_AMBIGUOUS):
        toks.pop()
    return "".join(toks)


def normalize_case(name: str) -> str:
    return name.lower()


def normalize_punct(name: str) -> str:
    return "".join(tokens(name))


def safe_suffix_forms(name: str) -> list[str]:
    """Return all forms reachable by stripping ALL consecutive safe suffix tokens
    from the end of the name.

    This handles compound safe suffixes like "OpenClaw Agent SDK" → {openclawagentsdk,
    openclawagent, openclaw}. The full normalization (normalize_full) already strips
    ALL suffix tokens (safe + ambiguous), so stripping just the safe ones from the
    end is the correct boundary: it produces the bare entity name which can then
    match against the bare-name variant.
    """
    toks = tokens(name)
    forms = ["".join(toks)]
    while toks and toks[-1] in STOP_SUFFIX_TOKENS_SAFE:
        toks = toks[:-1]
        forms.append("".join(toks))
    return [f for f in forms if f]


# ── Cluster classification ───────────────────────────────────────────────────


def classify_pair(a: str, b: str) -> tuple[str, str]:
    """
    Classify how `a` and `b` relate.

    Returns (tier, reason) where tier ∈ {CASE, PUNCT, SUFFIX_SAFE,
    SUFFIX_AMBIGUOUS, OTHER}.
    """
    if a == b:
        return ("IDENTICAL", "identical")
    if normalize_case(a) == normalize_case(b):
        return ("CASE", "case-only difference")
    if normalize_punct(a) == normalize_punct(b):
        return ("PUNCT", "punctuation/separator difference")
    a_forms = set(safe_suffix_forms(a))
    b_forms = set(safe_suffix_forms(b))
    if a_forms & b_forms:
        return ("SUFFIX_SAFE", "match after stripping one safe suffix (System/Agent/SDK/…)")
    if normalize_full(a) == normalize_full(b):
        return ("SUFFIX_AMBIGUOUS", "match only after stripping Loop/Research/Tool/…")
    return ("OTHER", "unclear")


def cluster_tier(variants: list[str]) -> str:
    """Take the MOST conservative tier across all pairs."""
    tiers = set()
    for i, a in enumerate(variants):
        for b in variants[i + 1 :]:
            t, _ = classify_pair(a, b)
            tiers.add(t)
    # If every pair is CASE/PUNCT/SUFFIX_SAFE → SAFE
    # If any pair is SUFFIX_AMBIGUOUS → AMBIGUOUS
    # If any pair is OTHER → OTHER (shouldn't happen after clustering by norm_full)
    if "OTHER" in tiers:
        return "OTHER"
    if "SUFFIX_AMBIGUOUS" in tiers:
        return "AMBIGUOUS"
    return "SAFE"


# ── Canonical selection ──────────────────────────────────────────────────────


def _has_safe_suffix(name: str) -> bool:
    toks = tokens(name)
    return bool(toks and toks[-1] in STOP_SUFFIX_TOKENS_SAFE)


_SLUG_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)+$")


def _is_slug(name: str) -> bool:
    """`nightly-reflection`, `worker_queue`: all-lowercase, joined by - or _."""
    return bool(_SLUG_RE.fullmatch(name))


def pick_canonical(variants: list[str], degrees: dict[str, int], existing_dirs: set[str]) -> str:
    """
    Pick the canonical name for a cluster.

    Priority:
      1. Highest degree — the name the graph already uses most.
      2. Prefer variant whose directory already exists.
      3. Prefer a readable title over a slug (`Nightly Reflection` over
         `nightly-reflection`); the survivor is what people and the extractor
         see. The previous first rule preferred the BARE NOUN over any suffixed
         form, which is how `Alfie pipeline` was absorbed into `Alfie` and
         `Intel Pipeline System` into `Intel`.
      4. Shorter name.
      5. Alphabetical (stable).
    """

    def key(v: str) -> tuple:
        return (
            -degrees.get(v, 0),
            0 if v in existing_dirs else 1,
            1 if _is_slug(v) else 0,
            len(v),
            v,
        )

    return sorted(variants, key=key)[0]


# ── Auto-merge decision ──────────────────────────────────────────────────────
#
# Rules:
#   CASE  / PUNCT  → auto-merge UNLESS cluster's total degree > HIGH_VALUE_GATE
#                    (reserves Tier-3 hand-review for highly-connected entities
#                    like lloyd/Lloyd, openclaw/OpenClaw where a wrong merge
#                    damages many edges at once).
#   SUFFIX_SAFE    → auto-merge if canonical ≥ 2× max_other_degree, OR variant
#                    degree ≤ 5 (small leftover cleanup). Otherwise hand-review.
#   SUFFIX_AMBIG.  → always hand-review.

HIGH_VALUE_GATE = 150  # total cluster degree above which CASE/PUNCT → review
SUFFIX_SAFE_RATIO = 1.3  # canonical:max-other ratio for SUFFIX_SAFE auto-merge
SUFFIX_SAFE_SMALL_VARIANT = 5


def decide_merge(
    tier: str,
    canonical: str,
    variants: list[str],
    degrees: dict[str, int],
) -> tuple[bool, str]:
    """Return (auto_merge_ok, reason) given the cluster's tier and degrees.

    Ratio is computed as max/second-max across the cluster (not canonical/other),
    so canonical-pick heuristics (e.g. bare-noun preference) don't interfere
    with the imbalance measurement.
    """
    if len(variants) < 2:
        return (True, "single variant")

    sorted_degs = sorted((degrees.get(v, 0) for v in variants), reverse=True)
    top_deg = sorted_degs[0]
    second_deg = sorted_degs[1]
    smallest_deg = sorted_degs[-1]
    total = sum(sorted_degs)

    if tier in ("CASE", "PUNCT", "IDENTICAL"):
        # When one variant has 0 degree (ghost entity), the high-value gate
        # is meaningless — there's no risk of merging distinct high-degree
        # entities. Only the high-degree variant's edges are at stake, and
        # they map 1:1 to the canonical.
        if total > HIGH_VALUE_GATE and smallest_deg > 0:
            return (
                False,
                f"high-value cluster (total degree {total} > {HIGH_VALUE_GATE}) — hand-review",
            )
        return (True, f"{tier} merge, total degree {total} (smallest={smallest_deg})")

    if tier == "SUFFIX_SAFE":
        # Never on name shape alone. The old "a variant has 0 degree" shortcut
        # is what fired 151 times against the empty graph on 2026-09-03. The
        # semantic gate in build_plan is the only path to an auto-merge here.
        return (False, "SUFFIX_SAFE — requires the semantic gate (definitions must agree)")

    if tier == "SUFFIX_AMBIGUOUS":
        # Suffix-ambiguous clusters (match only after stripping Loop/Research/Tool/…)
        # always go to hand-review per skill spec — no auto-merge exceptions.
        return (False, "SUFFIX_AMBIG — always hand-review")

    if tier == "OTHER":
        return (False, f"OTHER — always hand-review")

    if tier == "SAFE":
        # SAFE tier: every pair is CASE/PUNCT/SUFFIX_SAFE.
        # Auto-merge unless high-value gate.
        if total > HIGH_VALUE_GATE:
            return (
                False,
                f"SAFE high-value cluster (total degree {total} > {HIGH_VALUE_GATE}) — hand-review",
            )
        return (True, f"SAFE merge, total degree {total}")

    return (False, f"unclassified tier {tier}")


# ── Plan generation ──────────────────────────────────────────────────────────


def build_plan(edges: list[dict], existing_dirs: set[str],
               gate=None, allowed_tiers=None) -> dict:
    """Compute clusters and classify into a merge plan.

    `gate` is an entity_semantic_gate.SemanticGate (or any object with a
    `verdict(a, b) -> {"decision": ...}`); without one, SUFFIX_SAFE clusters go
    to review. `allowed_tiers` restricts which tiers may auto-merge.
    """
    allowed_tiers = set(allowed_tiers or ALL_TIERS)
    gate_stats = {"asked": 0, "same": 0, "review": 0}
    # degree = appearances as source or target in active edges
    degrees: dict[str, int] = collections.Counter()
    for e in edges:
        degrees[e["source"]] += 1
        degrees[e["target"]] += 1

    # Include existing directory names so entities with zero active-edge degree
    # (e.g. "OpenClaw SDK", "Voice Mode System") are still discovered and can be
    # merged into their canonical partners.
    entities = [e for e in (set(degrees.keys()) | existing_dirs) if not looks_like_junk_entity(e)]

    # Five-stage clustering:
    #   Stage A: cluster by normalize_case (case-only) — these are always SAFE_CASE.
    #            Process first so case-only variants don't get absorbed into
    #            larger suffix-ambiguous clusters (e.g. "agent" vs "Agent" shouldn't
    #            merge with "Agent Loop" into one AMBIGUOUS cluster).
    #   Stage B: cluster by normalize_punct (case+punct) — these are SAFE_PUNCT.
    #            Process second so pure case/punct variants (e.g. "Claude Code" vs
    #            "claude-code") are merged before being absorbed into suffix-ambiguous
    #            clusters via normalize_full connected components.
    #   Stage C: cluster by normalize_full (suffix-stripped) to surface candidates.
    #   Stage D: merge candidate sets from C, deduplicate, then classify tier.

    # Stage A: case-only clusters (processed independently, always SAFE_CASE)
    clusters_by_case: dict[str, list[str]] = collections.defaultdict(list)
    for ent in entities:
        key = normalize_case(ent)
        if key:
            clusters_by_case[key].append(ent)

    case_only_clusters = [
        sorted(v) for v in clusters_by_case.values() if len(v) > 1
    ]

    # Track which entities are already in case-only clusters
    case_clustered: set[str] = set()
    for variants in case_only_clusters:
        case_clustered.update(variants)

    # Stage B: punct-only clusters (processed independently, always SAFE_PUNCT)
    # These are entities that differ only by punctuation/separators (space, hyphen,
    # underscore) but are NOT case-only. Process them before suffix clustering so
    # they don't get absorbed into suffix-ambiguous groups.
    remaining_after_case = [e for e in entities if e not in case_clustered]
    clusters_by_punct: dict[str, list[str]] = collections.defaultdict(list)
    for ent in remaining_after_case:
        key = normalize_punct(ent)
        if key:
            clusters_by_punct[key].append(ent)

    punct_only_clusters: list[list[str]] = []
    for variants in clusters_by_punct.values():
        if len(variants) < 2:
            continue
        # A cluster is "punct-only" if every pair differs only by punctuation
        # (normalize_punct matches but normalize_case does not).
        is_punct_only = all(
            normalize_punct(a) == normalize_punct(b)
            and normalize_case(a) != normalize_case(b)
            for i, a in enumerate(variants)
            for b in variants[i + 1 :]
        )
        if is_punct_only:
            punct_only_clusters.append(sorted(variants))

    # Track which entities are in punct-only clusters
    punct_clustered: set[str] = set()
    for variants in punct_only_clusters:
        punct_clustered.update(variants)

    # Stage C: suffix clustering for remaining entities
    remaining = [e for e in remaining_after_case if e not in punct_clustered]

    clusters_by_full: dict[str, list[str]] = collections.defaultdict(list)
    for ent in remaining:
        key = normalize_full(ent)
        if key:
            clusters_by_full[key].append(ent)

    # Build connected components from normalize_full candidate edges
    ent_to_candidates: dict[str, set[str]] = collections.defaultdict(set)
    for group in clusters_by_full.values():
        if len(group) > 1:
            for ent in group:
                for other in group:
                    ent_to_candidates[ent].add(other)

    visited = set()
    suffix_clusters: list[list[str]] = []
    for ent in remaining:
        if ent in visited or ent not in ent_to_candidates:
            continue
        # BFS to find connected component
        component: list[str] = []
        queue = [ent]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in ent_to_candidates.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(component) > 1:
            suffix_clusters.append(component)

    # Combine: case-only and punct-only clusters are always SAFE,
    # suffix clusters are classified
    dupe_clusters = case_only_clusters + punct_only_clusters + suffix_clusters

    safe_merges: list[dict] = []
    ambiguous: list[dict] = []
    skipped: list[dict] = []

    for variants in dupe_clusters:
        # Case-only clusters are always SAFE_CASE — they were pre-separated
        # in build_plan to avoid absorption into suffix-ambiguous clusters.
        is_case_only = all(
            normalize_case(a) == normalize_case(b)
            for i, a in enumerate(variants)
            for b in variants[i + 1 :]
        )
        if is_case_only:
            cluster_worst = "CASE"
        else:
            # Punct-only clusters are always SAFE_PUNCT — they were pre-separated
            # in build_plan to avoid absorption into suffix-ambiguous clusters.
            is_punct_only = all(
                normalize_punct(a) == normalize_punct(b)
                and normalize_case(a) != normalize_case(b)
                for i, a in enumerate(variants)
                for b in variants[i + 1 :]
            )
            if is_punct_only:
                cluster_worst = "PUNCT"
            else:
                # Classify cluster-wide tier: take the most conservative pairwise tier.
                pairwise_tiers = set()
                for i, a in enumerate(variants):
                    for b in variants[i + 1 :]:
                        t, _ = classify_pair(a, b)
                        pairwise_tiers.add(t)
                if "OTHER" in pairwise_tiers:
                    cluster_worst = "OTHER"
                elif "SUFFIX_AMBIGUOUS" in pairwise_tiers:
                    cluster_worst = "SUFFIX_AMBIGUOUS"
                elif "SUFFIX_SAFE" in pairwise_tiers:
                    cluster_worst = "SUFFIX_SAFE"
                elif "PUNCT" in pairwise_tiers:
                    cluster_worst = "PUNCT"
                elif "CASE" in pairwise_tiers:
                    cluster_worst = "CASE"
                else:
                    cluster_worst = "IDENTICAL"

        canonical = pick_canonical(variants, degrees, existing_dirs)
        variant_degs = sorted(
            [(v, degrees.get(v, 0)) for v in variants], key=lambda x: -x[1]
        )
        auto_ok, decide_reason = decide_merge(cluster_worst, canonical, variants, degrees)

        gate_info = None
        if cluster_worst == "SUFFIX_SAFE" and gate is not None:
            gate_info = {}
            for v in variants:
                if v == canonical:
                    continue
                verdict = gate.verdict(v, canonical)
                gate_stats["asked"] += 1
                gate_info[v] = {"decision": verdict.get("decision"),
                                "judges": verdict.get("judges", {})}
            if gate_info and all(g["decision"] == "SAME" for g in gate_info.values()):
                auto_ok, decide_reason = True, "SUFFIX_JUDGED — every judge: SAME"
                gate_stats["same"] += 1
            else:
                auto_ok, decide_reason = False, "SUFFIX_SAFE — semantic gate: review"
                gate_stats["review"] += 1

        if auto_ok and cluster_worst not in allowed_tiers:
            auto_ok, decide_reason = False, f"tier {cluster_worst} excluded by --tiers"

        norm_key = normalize_punct(canonical)
        base = {
            "norm_key": norm_key,
            "canonical": canonical,
            "variants": variant_degs,
            "tier": cluster_worst,
            "decision": decide_reason,
        }
        if gate_info is not None:
            base["gate"] = gate_info

        if auto_ok:
            merges = []
            for v, d in variant_degs:
                if v == canonical:
                    continue
                subtier, subreason = classify_pair(v, canonical)
                merges.append(
                    {"variant": v, "degree": d, "subtier": subtier, "reason": subreason}
                )
            safe_merges.append({**base, "merges": merges})
        elif cluster_worst == "OTHER":
            skipped.append(base)
        else:
            ambiguous.append(base)

    return {
        "entity_count": len(entities),
        "active_edges": len(edges),
        "clusters_analyzed": len(dupe_clusters),
        "safe_clusters": len(safe_merges),
        "ambiguous_clusters": len(ambiguous),
        "skipped_clusters": len(skipped),
        "safe_merges": safe_merges,
        "ambiguous": ambiguous,
        "skipped": skipped,
        "existing_dirs_count": len(existing_dirs),
        "all_entities": sorted(entities),
        "gate_stats": gate_stats,
    }


# ── Apply ────────────────────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        fm = {}
    return fm, parts[2].lstrip("\n")


def _dump_frontmatter(fm: dict, body: str) -> str:
    ytxt = yaml.dump(fm, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return f"---\n{ytxt}---\n\n{body}"


def _merge_facts_lists(a: list, b: list) -> list:
    """Dedup by fact text. Prefer higher confidence, tiebreak by most-recent created_at."""
    seen: dict[str, dict] = {}
    for fact in list(a) + list(b):
        text = (fact.get("fact") or "").strip().lower()
        if not text:
            continue
        existing = seen.get(text)
        if existing is None:
            seen[text] = fact
            continue
        try:
            fact_conf = float(fact.get("confidence", 0))
        except (TypeError, ValueError):
            fact_conf = 0.0
        try:
            existing_conf = float(existing.get("confidence", 0))
        except (TypeError, ValueError):
            existing_conf = 0.0
        if fact_conf > existing_conf:
            seen[text] = fact
        elif fact_conf == existing_conf:
            if fact.get("created_at", "") > existing.get("created_at", ""):
                seen[text] = fact
    return list(seen.values())


def retag_fact_file(path: Path, variant: str, canonical: str) -> int:
    """After a merge, facts must carry the canonical's name. Rewrites the
    file-level `entity:` and every fact tagged with the variant, and stamps
    `merged_from: <variant>` on each retagged fact so the merge stays
    revertable (revert-suffix-merges.py matches on either field).

    Without this, every legitimate merge reads as cross-entity contamination
    to kg_hygiene.py and would be undone by a revert. Returns facts retagged.
    """
    try:
        fm, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    except OSError:
        return 0
    if not fm:
        return 0
    changed = 0
    if normalize_punct(str(fm.get("entity") or "")) == normalize_punct(variant):
        fm["entity"] = canonical
        changed += 1
    for fact in fm.get("facts") or []:
        if isinstance(fact, dict) and normalize_punct(str(fact.get("entity") or "")) == normalize_punct(variant):
            fact["entity"] = canonical
            fact.setdefault("merged_from", variant)
            changed += 1
    if not changed:
        return 0
    category = str(fm.get("category") or "")
    if fm.get("type") != "overview" and category:
        body = (f"\n# {canonical} - {category}\n\n**Entity:** {canonical}\n"
                f"**Category:** {category}\n**Fact Count:** {len(fm.get('facts') or [])}\n")
    atomic_write_text(path, _dump_frontmatter(fm, body))
    return changed


def _merge_fact_file_into(src: Path, dst: Path) -> None:
    """Merge src fact-file content into existing dst.

    For type=overview files: keep dst (newer canonical), discard src.
    For type=facts files: merge `facts:` arrays, dedup by text, write back.
    Caller is responsible for unlinking src after this returns.
    """
    src_fm, _ = _parse_frontmatter(src.read_text(encoding="utf-8"))
    dst_text = dst.read_text(encoding="utf-8")
    dst_fm, dst_body = _parse_frontmatter(dst_text)
    if dst_fm.get("type") == "overview" or src_fm.get("type") == "overview":
        return
    merged = _merge_facts_lists(dst_fm.get("facts") or [], src_fm.get("facts") or [])
    if not merged:
        return
    dst_fm["facts"] = merged
    dst_fm["last_updated"] = dt.datetime.now().isoformat()
    entity = dst_fm.get("entity", "")
    category = dst_fm.get("category", "")
    if entity and category:
        body = (
            f"\n# {entity} - {category}\n\n"
            f"**Entity:** {entity}\n"
            f"**Category:** {category}\n"
            f"**Fact Count:** {len(merged)}\n"
        )
    else:
        body = dst_body
    atomic_write_text(dst, _dump_frontmatter(dst_fm, body))


def load_semantic_proposals(out_dir: Path) -> list[dict]:
    """#67's latest judged pairs, as review input for this plan.

    Task #67 writes `semantic-proposals-latest.jsonl` and stops; it has no
    apply path any more. Missing or unreadable is normal (it runs weekly).
    """
    path = out_dir / "semantic-proposals-latest.jsonl"
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    except Exception as exc:
        print(f"  [proposals] unreadable ({exc}); ignoring")
        return []
    out.sort(key=lambda r: -float(r.get("confidence") or 0))
    return out


def prune_old_backups(path: Path, keep: int = 3, pattern: str | None = None) -> int:
    """Keep the newest `keep` backups beside `path`; delete older ones.

    Pre-2026-08-30, backups accumulated unbounded (246 files / 314 MB in
    ~8 days, mostly incident-doc runNNN-pre.bak snapshots). Safe no-op on
    any glob/stat error — rotation must never break the sweep.
    """
    try:
        baks = sorted(
            path.parent.glob(pattern or (path.name + "*.bak")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        pruned = 0
        for old in baks[keep:]:
            old.unlink(missing_ok=True)
            pruned += 1
        return pruned
    except Exception:
        return 0


def apply_merges(
    plan: dict,
    st,
    facts_root: Path,
    rebuild_aliases: bool,
    existing_dirs: set[str] | None = None,
    entities: list[str] | None = None,
) -> dict:
    """Execute all SAFE merges in the plan against the store and the fact tree.

    The alias writes and every edge rewrite happen in ONE store transaction:
    a crash or a kill in the middle leaves the graph exactly as it was, not
    half-merged. The fact-file moves follow, after the transaction commits,
    because the filesystem cannot join it — and that is the safe order: the
    alias table already routes new facts to the survivor, so a crash between
    the two costs a re-run, not a corrupted tree.

    Returns a report dict. `edge_rewrites` maps each variant to the list of
    (old_edge_id, new_edge_id) pairs, which `revert-suffix-merges.py
    --fix-edges` inverts exactly.
    """
    variant_to_canonical: dict[str, str] = {}
    variant_tier: dict[str, str] = {}
    for cluster in plan["safe_merges"]:
        canonical = cluster["canonical"]
        for m in cluster["merges"]:
            variant_to_canonical[m["variant"]] = canonical
            variant_tier[m["variant"]] = m.get("subtier") or cluster.get("tier") or "OTHER"

    edge_rewrites: dict[str, list[tuple[int, int]]] = {}
    alias_writes = 0
    with st.transaction():
        # 1. Aliases first, inside the same transaction — the extractor
        #    consults them, and they must never survive a rolled-back merge.
        for variant, canonical in variant_to_canonical.items():
            kind = TIER_ALIAS_KIND.get(variant_tier.get(variant, ""), "semantic")
            st.aliases.set(variant, canonical, kind=kind, origin="sweep")
            st.entities.register(canonical)
            alias_writes += 1
        if rebuild_aliases:
            alias_writes += _prune_noise_aliases(st, existing_dirs)
        # 2. Edges. rewrite_endpoint expires each active edge and re-adds it
        #    on the canonical, so the pre-merge graph stays readable.
        for variant, canonical in variant_to_canonical.items():
            pairs = st.edges.rewrite_endpoint(variant, canonical, origin="sweep")
            if pairs:
                edge_rewrites[variant] = pairs

    rewrite_count = sum(len(v) for v in edge_rewrites.values())

    # ── Move fact files
    dir_ops: list[dict] = []
    for variant, canonical in variant_to_canonical.items():
        vdir = facts_root / variant
        cdir = facts_root / canonical
        if not vdir.exists():
            dir_ops.append(
                {"variant": variant, "canonical": canonical, "action": "skip_no_variant_dir"}
            )
            continue
        cdir.mkdir(parents=True, exist_ok=True)
        moved = 0
        # Try multiple possible old-prefix forms — entity names in filenames
        # might use space, hyphen, or underscore separators.
        possible_prefixes = [
            variant + "-",
            variant.replace(" ", "-") + "-",
            variant.replace(" ", "_") + "-",
            variant.lower() + "-",
            variant.lower().replace(" ", "-") + "-",
        ]
        # Dedupe while preserving order
        seen_pfx = set()
        possible_prefixes = [p for p in possible_prefixes if not (p in seen_pfx or seen_pfx.add(p))]

        new_prefix = canonical + "-"
        touched: list[Path] = []
        for f in list(vdir.iterdir()):
            if not f.is_file():
                continue
            new_name = f.name
            for old_prefix in possible_prefixes:
                if f.name.startswith(old_prefix):
                    new_name = new_prefix + f.name[len(old_prefix):]
                    break
            dest = cdir / new_name
            if dest.exists():
                # Merge YAML facts list instead of creating _dup{N} sidecar.
                # Sidecars accumulated 499 files of cleanup debt (see
                # _pipeline/memory-graph/reconcile_dup_files.py).
                _merge_fact_file_into(f, dest)
                f.unlink()
            else:
                shutil.move(str(f), str(dest))
            retag_fact_file(dest, variant, canonical)
            touched.append(dest)
            moved += 1
        # Move nested subdirs (writer pattern: <variant>/<variant>-experiment.md).
        # The file loop above skips non-files, so without this a non-empty
        # variant dir survives the merge, rmdir fails silently, and the next
        # sweep re-detects the same SAFE merge (2026-08-18 experiments/
        # recurrence #5 left a nested autoresearch fact dir behind).
        for d in list(vdir.iterdir()):
            if not d.is_dir():
                continue
            dest = cdir / d.name
            if dest.exists():
                # Name collision: merge file-by-file, never overwrite
                for inner in list(d.iterdir()):
                    if not inner.is_file():
                        continue
                    dest_file = dest / inner.name
                    if dest_file.exists():
                        _merge_fact_file_into(inner, dest_file)
                        inner.unlink()
                    else:
                        shutil.move(str(inner), str(dest_file))
                    touched.append(dest_file)
                try:
                    d.rmdir()
                except OSError:
                    pass
                moved += 1
            else:
                shutil.move(str(d), str(dest))
                touched.extend(dest.glob("*.md"))
                moved += 1
        # Remove variant dir if empty
        removed = False
        try:
            vdir.rmdir()
            removed = True
        except OSError:
            pass
        # The index follows the files: without this the merged facts would
        # still be indexed under the variant until the next full reindex.
        try:
            st.facts_idx.reindex(touched, root=facts_root)
            st.entities.remove(variant) if removed else None
        except Exception as exc:  # index is derived; never fail a merge on it
            print(f"    [warn] index update for {variant!r} failed: {exc}")
        dir_ops.append(
            {
                "variant": variant,
                "canonical": canonical,
                "files_moved": moved,
                "removed_dir": removed,
            }
        )

    return {
        "rewritten_edges": rewrite_count,
        "edge_rewrites": {k: [list(p) for p in v] for k, v in edge_rewrites.items()},
        "alias_writes": alias_writes,
        "dir_operations": dir_ops,
        "variant_to_canonical": variant_to_canonical,
    }


def _prune_noise_aliases(st, existing_dirs: set[str] | None) -> int:
    """Drop inherited alias rows that are pipeline noise (--rebuild-aliases).

    Noise is an entry whose surface and canonical collapse to the same
    `normalize_full` by suffix stripping alone. Case-only and punct-only
    variants are legitimate and stay. Entries whose canonical has no fact
    directory are dropped too, unless the canonical is still an edge endpoint.
    """
    if existing_dirs is None:
        existing_dirs = set()
    live = existing_dirs | st.edges.nodes()
    removed = 0
    for row in st.aliases.rows():
        k, v = row["surface"], row["canonical"]
        if _is_alias_noise(k, v) or (live and v not in live):
            st.aliases.remove(k)
            removed += 1
    return removed


def _is_alias_noise(k: str, v: str) -> bool:
    """Same rule the JSON-era `compute_aliases` applied to the inherited table:
    identical after suffix stripping AND differing by more than case."""
    if normalize_full(k) != normalize_full(v):
        return False
    tk, tv = tokens(k), tokens(v)
    if len(tk) != len(tv):
        return True
    return not all(a.lower() == b.lower() for a, b in zip(tk, tv))


# ── Output formatters ────────────────────────────────────────────────────────


def print_plan(plan: dict) -> None:
    print(f"== Entity Resolution Sweep — Plan ==")
    print(f"  Entities:        {plan['entity_count']}")
    print(f"  Active edges:    {plan['active_edges']}")
    print(f"  Dupe clusters:   {plan['clusters_analyzed']}")
    print(f"    SAFE (Tier 1): {plan['safe_clusters']}")
    print(f"    AMBIGUOUS:     {plan['ambiguous_clusters']}")
    print(f"    SKIPPED:       {plan['skipped_clusters']}")
    print()

    if plan["safe_merges"]:
        print("── SAFE merges (auto-apply with --apply) ──")
        for c in sorted(
            plan["safe_merges"], key=lambda x: -max(v[1] for v in x["variants"])
        ):
            print(
                f"  [{c['norm_key']}] canonical = {c['canonical']!r}  "
                f"tier={c['tier']}  ({c['decision']})"
            )
            for m in c["merges"]:
                print(
                    f"      {m['variant']!r} (d={m['degree']}) → {c['canonical']!r}"
                    f"   [{m['subtier']}: {m['reason']}]"
                )
        print()

    if plan["ambiguous"]:
        print("── AMBIGUOUS (Tier 2/3 hand-review) ──")
        for c in sorted(
            plan["ambiguous"], key=lambda x: -max(v[1] for v in x["variants"])
        ):
            variants_str = ", ".join(f"{v!r}(d={d})" for v, d in c["variants"])
            print(f"  [{c['norm_key']}] tier={c['tier']}  {variants_str}")
            print(f"      → {c['decision']}")
        print()


def write_plan_jsonl(plan: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for cluster in plan["safe_merges"]:
            f.write(json.dumps({"status": "SAFE", **cluster}) + "\n")
        for cluster in plan["ambiguous"]:
            f.write(json.dumps({"status": "AMBIGUOUS", **cluster}) + "\n")
        for cluster in plan["skipped"]:
            f.write(json.dumps({"status": "SKIPPED", **cluster}) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def load_baseline(path: Path | None = None) -> int:
    path = path or BASELINE_PATH
    try:
        return int(json.loads(path.read_text()).get("active_edges", 0))
    except Exception:
        return 0


def update_baseline(active: int, path: Path | None = None) -> int:
    """Record the largest active-edge count seen; returns the baseline in force."""
    path = path or BASELINE_PATH
    current = load_baseline(path)
    if active > current:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"active_edges": active,
                                    "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2))
        return active
    return current


def degraded_reason(active: int, baseline: int, fraction: float = DEGRADED_FRACTION) -> str | None:
    """Why --apply must refuse, or None."""
    if baseline <= 0:
        return None
    if active < baseline * fraction:
        return (f"graph is degraded: {active:,} active edges is below {fraction:.0%} of the "
                f"recorded baseline {baseline:,}. Every entity looks disconnected on a broken "
                f"graph and the merge heuristics stop meaning anything. Restore the graph, or "
                f"pass --allow-degraded if you have reviewed the plan by hand.")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Apply SAFE merges")
    ap.add_argument("--tiers", default=",".join(ALL_TIERS),
                    help="Tiers allowed to auto-merge (default: all; SUFFIX_SAFE still needs the gate)")
    ap.add_argument("--no-gate", action="store_true",
                    help="Skip the semantic gate: every SUFFIX_SAFE cluster goes to review")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="Apply even when active edges are far below the recorded baseline")
    ap.add_argument(
        "--rebuild-aliases",
        action="store_true",
        help="Also drop inherited alias rows that are pipeline noise. Requires --apply.",
    )
    ap.add_argument("--db", default=str(VAULT_KG_DB), help="knowledge-graph store")
    ap.add_argument("--facts-dir", default=str(FACTS_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    facts_root = Path(args.facts_dir)
    out_dir = Path(args.out_dir)

    st = KGStore(Path(args.db))
    active_edges = st.edges.active()

    existing_dirs = (
        {d.name for d in facts_root.iterdir() if d.is_dir()} if facts_root.exists() else set()
    )

    allowed_tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    gate = None
    if not args.no_gate:
        try:
            from entity_semantic_gate import SemanticGate
            gate = SemanticGate(facts_root)
        except Exception as e:  # no judge reachable → suffix clusters go to review
            print(f"  [gate] unavailable ({type(e).__name__}: {e}); SUFFIX_SAFE → review")
    plan = build_plan(active_edges, existing_dirs, gate=gate, allowed_tiers=allowed_tiers)
    plan["tiers_allowed"] = sorted(allowed_tiers)

    baseline_path = out_dir / "graph-baseline.json"   # lives with the plans/reports it guards
    baseline = update_baseline(len(active_edges), baseline_path)
    print(f"  Store:           {args.db}")
    print(f"  Baseline:        {baseline:,} active edges (now {len(active_edges):,})")
    gs = plan.get("gate_stats") or {}
    if gs.get("asked"):
        print(f"  Semantic gate:   {gs['asked']} suffix pairs judged — {gs['same']} clusters SAME, {gs['review']} to review")

    # Proposals from #67's weekly judge, if it has run. They are review input,
    # never an auto-merge: #67 lost its apply path on 2026-09-04.
    proposals = load_semantic_proposals(out_dir)
    if proposals:
        plan["semantic_proposals"] = proposals[:200]
        print(f"  #67 proposals:   {len(proposals)} pairs awaiting review")

    # Emit plan. Timestamped: the old `entity-merges-<date>.jsonl` was opened in
    # write mode by every dry-run, so a second run on the same day overwrote the
    # plan an earlier --apply had executed — the only record of what that apply
    # was shown. `-latest` is a convenience pointer for readers.
    print_plan(plan)
    run_ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    plan_out = out_dir / f"entity-merges-{args.date}-{run_ts}.jsonl"
    write_plan_jsonl(plan, plan_out)
    latest = out_dir / "entity-merges-latest.jsonl"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(plan_out.name)
    except OSError:
        pass
    print(f"Plan written: {plan_out}")

    if not args.apply:
        print()
        print(f"(dry-run — pass --apply to execute {plan['safe_clusters']} SAFE merges)")
        return 0

    reason = degraded_reason(len(active_edges), baseline)
    if reason and not args.allow_degraded:
        print(f"\nREFUSING --apply: {reason}")
        return 3

    # Apply
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    print()
    print(f"== Applying merges (timestamp: {ts}) ==")

    backup_dir = out_dir / "store-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    store_bak = st.backup(backup_dir / f"kg-sweep-{ts}.sqlite")
    prune_old_backups(store_bak, keep=5, pattern="kg-sweep-*.sqlite")
    print(f"  Backed up: {store_bak}")

    before = st.stats()
    report = apply_merges(plan, st, facts_root, args.rebuild_aliases,
                          existing_dirs=existing_dirs, entities=plan.get("all_entities", []))
    after = st.stats()

    # Report
    print(f"  Edges rewritten:   {report['rewritten_edges']}")
    print(f"  Alias writes:      {report['alias_writes']}")
    print(f"  Dir operations:    {len(report['dir_operations'])}")
    for op in report["dir_operations"]:
        if op.get("action") == "skip_no_variant_dir":
            print(f"    {op['variant']!r}: no dir, skipped")
        else:
            print(
                f"    {op['variant']!r} → {op['canonical']!r}: "
                f"moved {op['files_moved']} files, removed_dir={op['removed_dir']}"
            )
    print(f"  Store: {before} → {after}")
    print()

    # Save an apply report
    apply_out = out_dir / f"entity-merges-applied-{args.date}-{ts}.json"
    apply_out.parent.mkdir(parents=True, exist_ok=True)
    report_to_save = {k: v for k, v in report.items() if k != "aliases"}
    report_to_save["plan_file"] = str(plan_out)
    report_to_save["tiers_allowed"] = sorted(allowed_tiers)
    report_to_save["gate_stats"] = plan.get("gate_stats")
    report_to_save["baseline_active_edges"] = baseline
    report_to_save["store_backup"] = str(store_bak)
    report_to_save["store_before"], report_to_save["store_after"] = before, after
    report_to_save["ledger"] = invocation_ledger()   # who ran this — see _invocation.py
    with apply_out.open("w") as f:
        json.dump(report_to_save, f, indent=2, ensure_ascii=False)
    print(f"  Report: {apply_out}")

    st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
