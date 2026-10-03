#!/usr/bin/env python3
"""
Entity Resolution Sweep — Tier 1 (mechanical) for backlog #310.

Reads the live knowledge graph and entity directories, clusters entities by
normalized name, classifies each cluster as auto-mergeable (CASE / PUNCT /
SUFFIX) or ambiguous (LOOP / RESEARCH / OTHER), and either prints a plan
(dry-run) or applies the merges.

Apply mode — TWO steps with a killable gap, not one bounded operation:
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
     retags them, removes the empty variant dir. The filesystem cannot join
     the transaction, so this half runs AFTER it: it is a second, unbounded
     step, and a kill in it leaves aliases routing to the canonical while some
     of the variant's fact files are still under the variant. What makes that
     survivable is a journal, not atomicity — each variant's outcome is written
     into the apply report as it completes (`report_status` stays "started"
     until the whole run is done, which is what stops #1538's audit reading a
     partial run as a finished one), and the half can be replayed alone with
     `--resume <that report>`. Plain `--apply` is NOT the retry: a second apply
     records an EMPTY edge trail for an already-rewritten variant, because
     `rewrite_endpoint` walks only ACTIVE edges and the first apply expired
     them, and `revert-suffix-merges.py --fix-edges` takes its pairs from the
     one report it is handed — so the first run's ids go un-inverted and it
     falls back to its prose heuristic.

Ambiguous clusters are dumped to a review JSONL for Tier 2 hand-review.

Usage:
  # dry-run (default):
  python entity-resolution-sweep.py

  # apply Tier 1:
  python entity-resolution-sweep.py --apply

  # also drop inherited alias entries that are pipeline noise:
  python entity-resolution-sweep.py --apply --rebuild-aliases

  # finish an apply that died between the transaction and the moves (#1558):
  python entity-resolution-sweep.py --resume <entity-merges-applied-*.json>
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# Ensure app/ is importable when running this script standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────

from app.paths import PIPELINE_DIR, VAULT_FACTS_ROOT as FACTS_ROOT, VAULT_KG_DB
from app.entity_naming import looks_like_junk_entity
# The two sub-rules behind `looks_like_junk_entity` that mean "this string is a
# code artifact", imported bare rather than via the composite predicate: an alias
# SURFACE may be dotted, parenthesised or task-shaped and still be a real routing
# row. See `_is_exhaust_surface`.
from app.entity_naming import _CODE_FILE_RE, _CODE_CALL_RE, _CODE_CALL_CONTENT_RE
from app.fact_ids import dedupe_ids
from app.frontmatter import split_frontmatter   # #2138: one anchored closing-fence rule
from app.kg_store import KGStore
from app.atomic_io import atomic_write_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _invocation import invocation_ledger  # noqa: E402

OUT_DIR = PIPELINE_DIR / "memory-graph"
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


# Punctuation that separates or decorates a name without changing what it
# names: spaces, hyphens, dots, brackets, quotes, an ellipsis.
SEPARATOR_CHARS = frozenset(" \t-_./\\()[]{},:;'\"`\u2019\u2018\u201c\u201d\u2026")


def symbol_residue(name: str) -> str:
    """The characters of a name that are neither ASCII alphanumerics nor
    separators — `++`, `#`, `@`, `^`, `+`, `τ²`.

    The PUNCT tier reads every non-alphanumeric as noise, and on 2026-09-16
    its first automatic apply merged `C` and `C#` into `C++`, `pass^k` into
    `pass@k`, `τ²-bench` into `Bench` and `BrowseComp+` into `BrowseComp`:
    names whose meaning lives in the symbol the tier discards. Two names
    with different residues are different names whatever their letters say,
    and no mechanical tier may merge them; they go to hand review.
    """
    return "".join(sorted(ch for ch in name.lower()
                          if not re.fullmatch(r"[a-z0-9]", ch) and ch not in SEPARATOR_CHARS))


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
    if symbol_residue(a) != symbol_residue(b) and (
            normalize_full(a) == normalize_full(b)
            or set(safe_suffix_forms(a)) & set(safe_suffix_forms(b))):
        # Same letters, different symbols: C / C++ / C#. Never mechanical, and
        # not for the semantic gate either — a judge shown two one-line
        # definitions of "C" and "C++" is a coin toss on a merge that moves
        # every fact. Always a person.
        return ("SUFFIX_AMBIGUOUS", "same letters, different symbols (C / C++ / C#) — always hand-review")
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
    clusters_by_punct: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for ent in remaining_after_case:
        # Keyed on the symbol residue too, so `C`, `C++` and `C#` never share
        # a punct cluster (see symbol_residue).
        key = normalize_punct(ent)
        if key:
            clusters_by_punct[(key, symbol_residue(ent))].append(ent)

    punct_only_clusters: list[list[str]] = []
    for variants in clusters_by_punct.values():
        if len(variants) < 2:
            continue
        # A cluster is "punct-only" if every pair differs only by punctuation
        # (normalize_punct matches but normalize_case does not).
        is_punct_only = all(
            normalize_punct(a) == normalize_punct(b)
            and symbol_residue(a) == symbol_residue(b)
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
                and symbol_residue(a) == symbol_residue(b)
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
    """`(front matter, body)` for a fact file, cut at the ANCHORED closing fence.

    #1400 gave the extractor `app.frontmatter.split_frontmatter`; this is that
    same rule reaching the sweep (#2138). The rule is "the first *line* that is
    exactly `---`", and the difference is not cosmetic, because a fact's own text
    can hold a fence. `yaml.dump` writes a multi-line fact as a single-quoted
    scalar whose continuation lines are INDENTED, so a fact whose prose carries a
    `---` — a fenced line, or the live corpus's `` `---segment:` `` — survives
    this module's own re-dump and then defeats its next read. What the old
    unanchored three-way split on that fence did is worse than an error: the
    truncated slice PARSES, returning just `{type, entity, category, facts}` with
    the half-cut fact still in the list and `source_doc`/`last_updated` gone. So
    neither `if not fm` in `retag_fact_file` nor `if not merged` in
    `_merge_fact_file_into` fires, and the writer re-dumps the truncated dict
    over the file — 4 facts in, 2 out on the live
    `Zero-width assertion regex bug/Zero-width assertion regex bug-skill.md`,
    6 in / 4 out on `False Absence Guard Pattern/…-state.md`.

    A non-dict load (a list, a bare string) comes back as `{}` too: every caller
    below reads `fm.get(...)`, and this is the shape
    `revert-suffix-merges.py::_read` has always returned.
    """
    split = split_frontmatter(text)
    if split is None:
        return {}, text
    try:
        fm = yaml.safe_load(split[0]) or {}
    except Exception:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), split[1].lstrip("\n")


def _frontmatter_unreadable(text: str) -> bool:
    """True for a file this sweep must not write back, because it cannot read it.

    The pair to the anchored read. `_parse_frontmatter` answers `{}` both for a
    file with no front matter and for one whose YAML will not load, and the apply
    path used to treat those as "nothing to do here" while still MOVING the file
    and — when the destination existed — MERGING into it and UNLINKING the source
    (`_merge_fact_file_into(f, dest); f.unlink()`). Writing a file whose dict is
    missing keys is one loss; deleting the copy you never read is another, and
    neither leaves a witness. So a file that CLAIMS front matter (a `---` first
    line) and cannot be parsed — no anchored closing fence, or YAML that will not
    load — is left exactly where it is and named in the apply report.

    False for a file that does not open with a fence: that is not a fact file, and
    moving it is a rename which costs no facts.
    """
    if not text.startswith("---"):
        return False
    split = split_frontmatter(text)
    if split is None:
        return True
    try:
        fm = yaml.safe_load(split[0])
    except Exception:
        return True
    return fm is not None and not isinstance(fm, dict)


def _unreadable_paths(*paths: Path | None) -> list[Path]:
    """Which of `paths` this sweep cannot round-trip (and so must not write or move).

    An `OSError` counts: a file that cannot be read is exactly as unsafe to
    overwrite or unlink as one whose YAML is broken.
    """
    out: list[Path] = []
    for p in paths:
        if p is None:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            out.append(p)
            continue
        if _frontmatter_unreadable(text):
            out.append(p)
    return out


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
    # Both sides numbered their facts from 1, so the concatenation holds each
    # ID twice. Dedup above is by fact TEXT and cannot see it.
    dedupe_ids(merged, dst_fm.get("category"))
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


SEEN_PROPOSALS_NAME = "semantic-proposals-seen.jsonl"


def load_seen_proposals(path: Path) -> set[tuple[str, str]]:
    """The (canonical, variant) keys this sweep has already surfaced in a plan.

    Its own state, appended by `surface_semantic_proposals` below. #67 now
    accumulates proposals across runs instead of rewriting one run's file, so
    without this ledger the sweep could not tell a proposal it has been shown
    every 15 minutes for a week from one that just arrived.
    """
    seen: set[tuple[str, str]] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return seen
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (str(rec.get("canonical") or ""), str(rec.get("variant") or ""))
        if all(key):
            seen.add(key)
    return seen


def _append_seen_proposals(path: Path, fresh: list[dict], now: str | None = None) -> int:
    """Append newly-surfaced keys. Append-only: a ledger that is rewritten can be
    lost, and this one is the only answer to 'what have I never looked at'."""
    if not fresh:
        return 0
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = now or dt.datetime.now().isoformat()
        with path.open("a", encoding="utf-8") as f:
            for rec in fresh:
                f.write(json.dumps({
                    "canonical": rec.get("canonical"),
                    "variant": rec.get("variant"),
                    "confidence": rec.get("confidence"),
                    "evaluated_at": stamp,
                }) + "\n")
    except OSError as exc:
        print(f"  [proposals] seen-ledger unwritable ({exc}); counts will repeat")
        return 0
    return len(fresh)


def surface_semantic_proposals(out_dir: Path, seen_path: Path | None = None,
                               now: str | None = None) -> tuple[list[dict], int]:
    """Load #67's proposals for this plan, report how many are new here, ledger them.

    Surfaced-in-a-plan counts as evaluated: the plan file is the review artifact,
    and dispositions (applied / rejected) are the apply ledger's business, not
    this one's. Returns `(proposals, never_evaluated)` and prints the plan line.
    """
    proposals = load_semantic_proposals(out_dir)
    if not proposals:
        return [], 0
    seen_path = Path(seen_path) if seen_path else Path(out_dir) / SEEN_PROPOSALS_NAME
    seen = load_seen_proposals(seen_path)
    keyed = [(r, (str(r.get("canonical") or ""), str(r.get("variant") or "")))
             for r in proposals]
    # A row missing either half of the key cannot be tracked, so it is not
    # counted as new either — otherwise the count would never reach 0.
    trackable = [(r, k) for r, k in keyed if all(k)]
    never = [r for r, k in trackable if k not in seen]
    _append_seen_proposals(seen_path, never, now)
    print(f"  #67 proposals:   {len(proposals)} pairs awaiting review, "
          f"{len(never)} never evaluated")
    return proposals, len(never)


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
    report_path: str | None = None,
) -> dict:
    """Execute all SAFE merges in the plan against the store and the fact tree.

    `report_path` is the apply report this run is about to write. Every alias
    row the apply creates carries it, so a later reader can answer "which run
    said this surface means that entity, and under what gate" without re-deriving
    it from mtimes. Before #475 the apply passed no `report_path` at all, so
    `SELECT COUNT(*) FROM aliases WHERE report_path IS NOT NULL` stayed 0 even
    after a clean apply and the backfill could never be attributed.

    The alias writes and every edge rewrite happen in ONE store transaction:
    a crash or a kill in the middle leaves the graph exactly as it was, not
    half-merged. The fact-file moves follow, after the transaction commits,
    because the filesystem cannot join it — which makes the apply TWO steps with
    a killable gap, not one bounded operation. A kill in that gap DOES leave a
    half-merge: the aliases already route to the canonical while some of the
    variant's fact files are still under the variant's own dir.

    What makes that gap survivable is a journal, not atomicity. Each variant's
    dir outcome is written into `report_path` as it completes, so a killed run
    leaves a report that names the dirs already moved, and `--resume <that
    report>` finishes the rest without touching the store half again (#1558).
    The sentence this replaces — "a crash between the two costs a re-run, not a
    corrupted tree" — held only if the re-run could tell what had already moved.
    It could not: `dir_ops` was in-memory until the final report, which a killed
    run never writes, and re-applying the same plan silently loses the first
    run's edge trail because `rewrite_endpoint` only walks ACTIVE edges, which
    the first apply had already expired.

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
            st.aliases.set(variant, canonical, kind=kind, origin="sweep",
                           report_path=report_path)
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

    # ── Move fact files: the half the transaction cannot cover (#1558) ──────
    dir_ops = _move_fact_dirs(
        variant_to_canonical, st, facts_root, report_path=report_path,
        extra={"variant_to_canonical": variant_to_canonical,
               "edge_rewrites": {k: [list(p) for p in v] for k, v in edge_rewrites.items()}})

    return {
        "rewritten_edges": rewrite_count,
        "edge_rewrites": {k: [list(p) for p in v] for k, v in edge_rewrites.items()},
        "alias_writes": alias_writes,
        "dir_operations": dir_ops,
        "variant_to_canonical": variant_to_canonical,
    }


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write `obj` to `path` through a dot-prefixed tmp file and `os.replace`.

    The file this writes is the recovery input for a half-finished merge, so it
    has to parse for the whole time it exists: a truncated report is worse than
    a missing one, because the next reader cannot tell a partial journal from a
    complete one and `--resume` would trust it. Same shape as the final report
    write in `main()`, which is why the tmp name is dot-prefixed — the apply
    report glob in `memory_disposition_audit._sweep_applied_reports` is
    `entity-merges-applied-*.json`, and a stray tmp must not match it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _journal_dir_progress(report_path: str | None, dir_ops: list[dict], extra: dict) -> None:
    """Re-write the claimed apply report with the dir outcomes earned so far.

    Read-modify-write of the file `claim_report` put down, keeping its `run_id`,
    `ledger` and plan pointers: the two facts a reader needs are which run this
    is and which dirs have moved. `report_status` stays `"started"` and
    `applied_clusters` stays None while the run is in progress — a report that
    counted merges it had not finished is the exact lie #1538 exists to catch —
    and the whole file goes through tmp + `os.replace`, so a second kill cannot
    leave a half-written journal behind.
    """
    if not report_path:
        return
    path = Path(report_path)
    doc: dict = {}
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}                     # a stub we cannot read is still ours to rewrite
    doc.setdefault("report_status", "started")
    doc.setdefault("applied_clusters", None)
    doc.update(extra)
    doc["dir_operations"] = dir_ops
    _atomic_write_json(path, doc)


def _move_fact_dirs(variant_to_canonical: dict[str, str], st, facts_root: Path, *,
                    report_path: str | None = None, carried: list[dict] = (),
                    skip: set[str] = frozenset(), extra: dict | None = None) -> list[dict]:
    """Move each variant's fact files into its canonical dir, journaling as it goes.

    The loop body is the apply's original one — same prefix matching, same
    `_merge_fact_file_into`/`retag_fact_file` rules, same `facts_idx` retirement of
    the paths a merge moved away (#996), same `entities.remove`. Only three things
    are new (#1558): the entries a previous run already finished are carried in as
    `carried`, the variants they name are in `skip` so a resume never re-applies a
    finished move, and every finished variant — including one that had no dir to
    move — is stamped `done` and written to `report_path` before the next one
    starts. That journal is what makes the post-commit window survivable:
    `apply_merges`' old promise that "a crash between the two costs a re-run" was
    only true if a re-run could tell what had already moved, and `dir_ops` used to
    live in memory until a report a killed run never writes.
    """
    dir_ops: list[dict] = list(carried)
    # Journal the store half BEFORE the first move. A kill inside the very first
    # variant then leaves a report whose dir list holds only the carried entries —
    # correctly "nothing finished here" — but it carries the pairs and the edge ids,
    # so --resume has input and revert-suffix-merges.py --fix-edges can still name
    # the edges it has to undo.
    _journal_dir_progress(report_path, dir_ops, dict(extra or {}))
    for variant, canonical in variant_to_canonical.items():
        if variant in skip:
            continue                        # journaled done by the run that died
        vdir = facts_root / variant
        cdir = facts_root / canonical
        if not vdir.exists():
            # A variant with no fact dir is finished work, not pending work: it gets
            # the same `done` stamp, so a resume never revisits it.
            dir_ops.append(
                {"variant": variant, "canonical": canonical, "action": "skip_no_variant_dir",
                 "done": True}
            )
            _journal_dir_progress(report_path, dir_ops, dict(extra or {}))
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
        touched: list[Path] = []    # where each file is NOW, re-read by the index
        # Where each file USED TO BE. `facts_idx` keys rows on `file_path`, so a
        # move that only re-reads the destination leaves the source row live
        # (`expired_at IS NULL AND invalid_at IS NULL`) pointing at a path that
        # no longer exists — and it still carries the variant's name, so every
        # live-row coverage/fragmentation query keeps counting an entity that has
        # genuinely been merged away (#996: 1,095 live rows across the 115 SAFE
        # merges that have a variant dir on disk). The index is derived from the
        # markdown, so retiring a row here costs nothing that a reindex of the
        # restored file cannot put back — which is exactly what
        # `revert-suffix-merges.py` does on a revert.
        retired: list[Path] = []
        # #2138: files this run refused to touch, by full path, so the apply
        # report can name them instead of a later reader finding them still here.
        skipped_unreadable: list[str] = []
        for f in list(vdir.iterdir()):
            if not f.is_file():
                continue
            new_name = f.name
            for old_prefix in possible_prefixes:
                if f.name.startswith(old_prefix):
                    new_name = new_prefix + f.name[len(old_prefix):]
                    break
            dest = cdir / new_name
            # Both sides are checked because the merge writes the destination and
            # then unlinks this file: an unreadable DESTINATION would be re-dumped
            # from a truncated dict, and an unreadable SOURCE contributes nothing
            # to the merge and is deleted anyway. Neither is a reason to lose a
            # fact, so the pair stays exactly where it is and says so.
            unreadable = _unreadable_paths(f, dest if dest.exists() else None)
            if unreadable:
                for p in unreadable:
                    print(f"    [skip] unparseable front matter, left untouched: {p}")
                skipped_unreadable.extend(str(p) for p in unreadable)
                continue
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
            retired.append(f)
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
                    bad = _unreadable_paths(inner, dest_file if dest_file.exists() else None)
                    if bad:
                        for p in bad:
                            print(f"    [skip] unparseable front matter, left untouched: {p}")
                        skipped_unreadable.extend(str(p) for p in bad)
                        continue
                    if dest_file.exists():
                        _merge_fact_file_into(inner, dest_file)
                        inner.unlink()
                    else:
                        shutil.move(str(inner), str(dest_file))
                    retag_fact_file(dest_file, variant, canonical)
                    touched.append(dest_file)
                    retired.append(inner)
                try:
                    d.rmdir()
                except OSError:
                    pass
                moved += 1
            else:
                src_mds = sorted(d.glob("*.md"))        # their paths BEFORE the move
                shutil.move(str(d), str(dest))
                dest_mds = sorted(dest.glob("*.md"))
                for p in dest_mds:                      # the file loop's retag, here too
                    retag_fact_file(p, variant, canonical)
                touched.extend(dest_mds)
                retired.extend(src_mds)
                moved += 1
        # Remove variant dir if empty
        removed = False
        try:
            vdir.rmdir()
            removed = True
        except OSError:
            pass
        # The index follows the files — in both directions. `touched` re-reads
        # each destination so the merged facts are counted under the canonical;
        # `retired` names each path this merge just moved away, and `reindex`
        # drops a path's rows before checking the file exists, so a source that
        # is gone has its rows retired rather than left live under the variant
        # (#996). Without the second half the variant keeps counting forever:
        # nothing else ever re-reads a path it did not itself name, and no
        # scheduled job does a full reindex.
        try:
            st.facts_idx.reindex([*touched, *retired], root=facts_root)
            st.entities.remove(variant) if removed else None
        except Exception as exc:  # index is derived; never fail a merge on it
            print(f"    [warn] index update for {variant!r} failed: {exc}")
        entry = {
            "variant": variant,
            "canonical": canonical,
            "files_moved": moved,
            "removed_dir": removed,
            "done": True,
        }
        if skipped_unreadable:
            # `done` is still True — every file this run COULD read has moved,
            # and a resume must not re-run the merge to chase a file it is
            # deliberately leaving alone. The refusal is the extra field, and it
            # is journaled with the variant, so it survives a kill in this window.
            entry["skipped_unparseable"] = skipped_unreadable
        dir_ops.append(entry)
        # One journal step per finished variant: the most a kill can cost is the
        # variant it landed inside, and a resume re-does exactly that one.
        _journal_dir_progress(report_path, dir_ops, dict(extra or {}))
    return dir_ops


def resume_merges(report: dict, st, facts_root: Path, report_path: str | None = None) -> dict:
    """Finish the fact-move half of an apply that died mid-window.

    Reads the pairs out of the report a killed run claimed and replays ONLY the
    `dir_operations` not yet stamped done. It writes no alias and no edge row:
    the store half of the original run committed, and re-selecting a plan here
    would merge new pairs into a report that claims to describe one run. The
    original run's `edge_rewrites` are carried into the returned report rather
    than re-derived, because `rewrite_endpoint` records only ACTIVE edges and
    the first apply already expired them — re-deriving would hand
    `revert-suffix-merges.py --fix-edges` an empty trail, and it picks its mode
    from the one report it is handed, so an empty trail silently degrades exact
    by-id inversion to the `_own_prose_edges` heuristic.
    """
    v2c = {k: v for k, v in (report.get("variant_to_canonical") or {}).items()}
    ops = [op for op in (report.get("dir_operations") or []) if isinstance(op, dict)]
    carried = [op for op in ops if op.get("done")]
    edge_rewrites = report.get("edge_rewrites") or {}
    dir_ops = _move_fact_dirs(
        v2c, st, facts_root, report_path=report_path, carried=carried,
        skip={str(op.get("variant")) for op in carried},
        extra={"variant_to_canonical": v2c, "edge_rewrites": edge_rewrites})
    return {
        "rewritten_edges": sum(len(v) for v in edge_rewrites.values()),
        "edge_rewrites": edge_rewrites,          # run 1's pairs, untouched
        "alias_writes": 0,                        # nothing here writes an alias
        "dir_operations": dir_ops,
        "variant_to_canonical": v2c,
    }


def _prune_noise_aliases(st, existing_dirs: set[str] | None) -> int:
    """Drop inherited alias rows that are pipeline noise (--rebuild-aliases).

    Three kinds of noise, and only the second one is new here:
      * an entry whose surface and canonical collapse to the same
        `normalize_full` by suffix stripping alone. Case-only and punct-only
        variants are legitimate and stay.
      * a surface `looks_like_junk_entity` refuses — a filename, a code-call
        fragment, a bookkeeping file. `build_plan` already refuses to put such a
        a code filename or a call fragment. `build_plan` already refuses to put
        such a name inside a cluster (:332), and the 2026-09-03 migration carried
        rows in from a JSON table that predates that filter — a row the plan never
        looks at is a row nothing else ever prunes. See `_is_exhaust_surface` for
        why this is narrower than `looks_like_junk_entity`, and why being broader
        here would have cost coverage instead of gaining it.
      * an entry whose canonical has no fact directory and is not an edge
        endpoint.
    """
    if existing_dirs is None:
        existing_dirs = set()
    live = existing_dirs | st.edges.nodes()
    removed = 0
    for row in st.aliases.rows():
        k, v = row["surface"], row["canonical"]
        if _is_exhaust_surface(k) or _is_alias_noise(k, v) or (live and v not in live):
            st.aliases.remove(k)
            removed += 1
    return removed


def _is_exhaust_surface(surface: str) -> bool:
    """True only for a surface that is unambiguously this pipeline's own exhaust: a
    code filename (`server.py`, `kg_store.py`) or a call fragment (`query()`,
    `models.load_lora_adapter()`).

    Deliberately NOT `looks_like_junk_entity`, which this module already imports
    for the cluster path. That predicate answers "may this become an ENTITY?" and
    is tuned high-precision for creation; an alias surface is a different object —
    a spelling that routes to a canonical. Measured against the live store on
    2026-09-12, filtering surfaces by the whole predicate would have deleted 263 of
    3,919 rows in a run whose stated purpose is to raise coverage, and the sample
    shows they are not exhaust: `.openclaw → OpenClaw`, `.openclaw config →
    OpenClaw Config`, `Autonomy task #24 → Autonomy Task #24`, `Neo et al.
    (Interpreting Vision Grounding in VLMs) → …`. Those are working routing rows,
    so removing them LOWERS alias coverage — the one way clause 2 could be failed by
    the fix itself. These two shapes remove 0 of today's 3,919 rows and still catch
    the exhaust the migration can carry.
    """
    s = (surface or "").strip()
    if _CODE_FILE_RE.match(s):
        return True
    m = _CODE_CALL_RE.search(s)
    return bool(m) and (not m.group(1) or bool(_CODE_CALL_CONTENT_RE.search(m.group(1))))


def _is_alias_noise(k: str, v: str) -> bool:
    """Same rule the JSON-era `compute_aliases` applied to the inherited table:
    identical after suffix stripping AND differing by more than case."""
    if normalize_full(k) != normalize_full(v):
        return False
    tk, tv = tokens(k), tokens(v)
    if len(tk) != len(tv):
        return True
    return not all(a.lower() == b.lower() for a, b in zip(tk, tv))


def claim_report(path: Path, plan_out: Path, date: str, ts: str) -> None:
    """Put the apply report's NAME on disk before the store transaction runs.

    Every alias row the apply writes carries this path as its provenance. Until
    now the path was only *computed* before the apply and the file written after
    it — so a run killed between the commit and the write left alias rows pointing
    at a report that never existed, and the disposition audit could say nothing
    about a merge that genuinely happened except `report_missing`. Claiming the
    name first means a row's pointer always resolves.

    The stub deliberately claims no decision: `applied_clusters` is None, not 0.
    Zero would read as "the plan was evaluated and nothing was safe to merge",
    which is a verdict, and a run that did not finish does not have one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "report_status": "started",
        "date": date,
        "timestamp": ts,
        "plan_file": str(plan_out),
        "applied_clusters": None,
        "applied_merges": None,
        "ledger": invocation_ledger(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def degraded_reason(active: int, baseline: int, fraction: float = DEGRADED_FRACTION,
                    *, measured: bool = True) -> str | None:
    """Why --apply must refuse, or None.

    An empty store refuses whatever the baseline file says (#1557). Before this,
    the guard opened with `if baseline <= 0: return None`, so a missing or
    unparseable `graph-baseline.json` disarmed it for the whole run — and
    `update_baseline` is max-only and bootstraps `baseline := active` three lines
    before the guard is consulted, so `degraded_reason(active, active)` can never
    fire and the freshly written floor is whatever the degraded count happened to
    be. The 2026-09-22 apply is that shape on disk:
    `entity-merges-applied-2026-09-22-20260922T204831Z.json` carries
    `baseline_active_edges: 0`, `safety.degraded_graph: "not degraded"`, 30 applied
    merges and `rewritten_edges: 0`, against a 77,824-byte store backup where the
    three following nights' backups are 78 MB, 78 MB and 87 MB. A store with no
    active edges is the 2026-09-03 condition in its limit case — every entity at
    degree 0, every variant indistinguishable from every other — and it is a fact
    about the store, so it does not need the baseline file to be knowable.

    `measured` says whether a floor was on disk before this run wrote one. It
    only changes the wording: the refusal is the same either way, and a caller
    that cannot know still gets the honest message.
    """
    if active <= 0:
        floor = ("and no baseline was on disk before this run, so there was no recorded "
                 "count to compare against either" if not measured else
                 f"and the recorded baseline of {baseline:,} active edges says nothing "
                 "about a store that now holds none")
        return (f"graph is empty or unmeasurable: {active:,} active edges. Every entity "
                f"looks disconnected and every name looks like a variant of every other, "
                f"which is the shape that fused 151 entities on 2026-09-03 and moved 30 "
                f"fact dirs on 2026-09-22 while rewriting 0 edges, {floor}. Point --db at "
                f"the real store and restore the graph, or pass --allow-degraded if you "
                f"have reviewed the plan by hand.")
    if baseline <= 0:
        # No floor recorded and a store that does hold edges: nothing to compare
        # against, so nothing to refuse — but this is `unmeasured`, not healthy,
        # and safety_record labels it as such rather than "not degraded".
        return None
    if active < baseline * fraction:
        return (f"graph is degraded: {active:,} active edges is below {fraction:.0%} of the "
                f"recorded baseline {baseline:,}. Every entity looks disconnected on a broken "
                f"graph and the merge heuristics stop meaning anything. Restore the graph, or "
                f"pass --allow-degraded if you have reviewed the plan by hand.")
    return None


def safety_record(no_gate: bool, allow_degraded: bool, gate, plan: dict,
                  degraded: str | None, *, active_edges: int | None = None,
                  measured: bool = True) -> dict:
    """What the apply report must say about the checks that actually ran.

    The 2026-09-03 apply that fused 151 distinct entities is hard to audit
    precisely because nothing on disk recorded that it ran against a 2-edge
    graph: the report carried counts but not the state of the two switches that
    were built to stop it. This is that answer, written from the flags as
    parsed, so a later reader never has to reconstruct it from a shell history
    line — and `--no-gate` / `--allow-degraded` are visible as `true` when they
    were used instead of merely absent.

    `degraded_graph` must also never read `"not degraded"` for a run the guard
    could not measure (#1557). The 2026-09-22 report put `baseline_active_edges:
    0` and `store_before.edges_active: 0` on either side of `"degraded_graph":
    "not degraded"` — three facts that are only legible to a reader who already
    knows 0 means "unmeasured" — because with no baseline file the guard returned
    `None`, which this function rendered as a clean verdict. So `measured` is
    stated, and the store's own active-edge count is carried beside it as
    `active_edges`: a guarded apply, a bypassed one and an unmeasured one are
    distinguishable from the `safety` block alone.

    Both new parameters are keyword-only and default to the *measured* case, so
    the existing positional callers and the unit calls that pin the gate verdict
    keep their meaning: a caller that says nothing is a caller that measured.
    """
    gs = plan.get("gate_stats") or {}
    if no_gate:
        verdict = "skipped: --no-gate — every SUFFIX_SAFE cluster went to review"
    elif gate is None:
        verdict = ("unavailable: the semantic gate could not be constructed — "
                   "every SUFFIX_SAFE cluster went to review")
    else:
        verdict = (f"ran: {gs.get('asked', 0)} suffix pairs judged, "
                   f"{gs.get('same', 0)} SAME, {gs.get('review', 0)} to review")
    if degraded and allow_degraded:
        degraded_note = "bypassed: --allow-degraded"
    elif degraded:
        # Defensive, and unreachable from `main` today: main returns 3 before it
        # applies when the graph is degraded and --allow-degraded is absent, so no
        # shipped caller reaches this with a degradation it went on to ignore. It
        # stays because the alternative is falling through to "not degraded", which
        # would have the report deny a degradation the caller could see. Anyone
        # looking for the caller: there is none, by design — this is the branch
        # that keeps a future caller from lying.
        degraded_note = "degraded (would have refused)"
    elif not measured:
        degraded_note = ("unmeasured: no graph baseline was on disk before this run, so the "
                         "degraded-graph guard had no floor to compare against — the count "
                         "below is the store's own, not a verdict about it")
    else:
        degraded_note = "not degraded"
    return {"no_gate": bool(no_gate), "allow_degraded": bool(allow_degraded),
            "gate_verdict": verdict, "degraded_graph": degraded_note,
            "active_edges": active_edges, "baseline_measured": bool(measured)}


def _resume_apply(args, st, facts_root: Path) -> int:
    """`--resume <apply report>`: finish a killed apply's file-move half.

    Deliberately skips three things the fresh-apply path runs. It does not build
    a plan or run the gate: this run merges nothing, and selecting new pairs would
    fold them into a report that claims to describe one run. It does not enforce
    the degraded-graph guard either — that guard stops a run from *making* a mess
    on a broken graph, and refusing a recovery there would leave a half-merge
    stuck, which is the failure this path exists to close; it prints the guard's
    reason instead, so the operator still sees the ground underfoot. And `--apply`
    is ignored, because there is nothing here to authorise: the merges were already
    committed by the run that claimed the report.
    """
    report_path = Path(args.resume)
    if not report_path.is_file():
        print(f"REFUSING --resume: no apply report at {report_path}")
        return 2
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"REFUSING --resume: {report_path} does not parse ({exc})")
        return 2
    v2c = report.get("variant_to_canonical") or {}
    if not v2c:
        print(f"REFUSING --resume: {report_path} records no variant_to_canonical, so no merge "
              "was ever committed for it to finish. A stub with none means the run died inside "
              "the store transaction, which rolled back on its own.")
        return 2
    done = {str(op.get("variant")) for op in (report.get("dir_operations") or [])
            if isinstance(op, dict) and op.get("done")}
    pending = [v for v in v2c if v not in done]
    if not pending and report.get("report_status") == "complete":
        print(f"== {report_path.name}: already complete, {len(v2c)} variants journaled done ==")
        return 0

    print(f"== Resuming apply {report_path.name} (claimed {report.get('timestamp', '?')}, "
          f"{len(pending)} of {len(v2c)} dirs pending) ==")
    if args.apply:
        print("  [resume] --apply ignored: this finishes the claimed report, it does not merge")
    baseline_floor = load_baseline(Path(args.out_dir) / "graph-baseline.json")
    guard = degraded_reason(len(st.edges.active()), baseline_floor)
    print("  [resume] graph, read-only: "
          + (guard or ("healthy" if baseline_floor > 0 else
                       "unmeasured (no baseline on disk) — read is not a risk, so it cannot "
                       "refuse a recovery"))
          )

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    backup_dir = Path(args.out_dir) / "store-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    store_bak = st.backup(backup_dir / f"kg-sweep-resume-{ts}.sqlite")
    print(f"  Backed up store: {store_bak}")

    before = st.stats()
    result = resume_merges(report, st, facts_root, str(report_path))
    after = st.stats()
    for op in result["dir_operations"]:
        if op["variant"] not in pending:
            continue                                # journaled done by the run that died
        if op.get("action") == "skip_no_variant_dir":
            print(f"  {op['variant']} → {op['canonical']}: no dir on disk, nothing to move")
        else:
            print(f"  {op['variant']} → {op['canonical']}: {op['files_moved']} file(s) moved, "
                  f"removed_dir={op['removed_dir']}")
        if op.get("skipped_unparseable"):
            print(f"  {op['variant']} → {op['canonical']}: SKIPPED as unparseable, left in place: "
                  + ", ".join(op["skipped_unparseable"]))
    print(f"  Store: entities {before['entities']}→{after['entities']}, "
          f"aliases {before['aliases']}→{after['aliases']}, "
          f"edges {before['edges_active']}→{after['edges_active']} "
          "(0 alias/edge writes expected: the first run committed them)")

    finished = dict(report)                      # every field of the claiming run survives
    finished.update(result)                      # dir list complete now, run 1's ids carried
    finished["report_status"] = "complete"
    finished["applied_merges"] = len(v2c)
    finished["applied_clusters"] = report.get("applied_clusters")   # run 1 never claimed a count
    finished["store_backup"] = str(store_bak)
    finished["store_before"], finished["store_after"] = before, after
    finished["resumed"] = {
        "pending_variants": pending,
        "ledger": invocation_ledger(),
        "note": "the aliases and edge rewrites were committed by the run that claimed this "
                "report; this run only moved the fact files that report had not journaled done, "
                "and wrote no alias or edge row",
    }
    _atomic_write_json(report_path, finished)
    print(f"  Report: {report_path}")
    st.close()
    return 0


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
    ap.add_argument("--gate-cache", default=None,
                    help="semantic-gate verdict cache to read/write. Default: the pipeline's "
                         "own cache. Pass a private path (a tmp dir) for a hermetic run — "
                         "without this, a test that runs the CLI with the gate on silently "
                         "reads production verdicts it never earned.")
    ap.add_argument("--db", default=str(VAULT_KG_DB), help="knowledge-graph store")
    ap.add_argument("--facts-dir", default=str(FACTS_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--resume", metavar="APPLY_REPORT", default=None,
                    help="finish the fact-file half of an apply that died after its store "
                         "transaction committed: replay only the dir_operations that report has "
                         "not journaled done, write no alias and no edge, and keep the original "
                         "run's (old_edge_id, new_edge_id) pairs so revert stays exact by id")
    args = ap.parse_args()

    facts_root = Path(args.facts_dir)
    out_dir = Path(args.out_dir)

    st = KGStore(Path(args.db))
    if args.resume:
        return _resume_apply(args, st, facts_root)
    active_edges = st.edges.active()

    existing_dirs = (
        {d.name for d in facts_root.iterdir() if d.is_dir()} if facts_root.exists() else set()
    )

    allowed_tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    gate = None
    if not args.no_gate:
        try:
            from entity_semantic_gate import SemanticGate
            # Unset means the gate's own default (the pipeline cache); set means
            # this run reads and writes only that file.
            gate = (SemanticGate(facts_root) if not args.gate_cache
                    else SemanticGate(facts_root, cache_path=Path(args.gate_cache)))
        except Exception as e:  # no judge reachable → suffix clusters go to review
            print(f"  [gate] unavailable ({type(e).__name__}: {e}); SUFFIX_SAFE → review")
    plan = build_plan(active_edges, existing_dirs, gate=gate, allowed_tiers=allowed_tiers)
    plan["tiers_allowed"] = sorted(allowed_tiers)

    baseline_path = out_dir / "graph-baseline.json"   # lives with the plans/reports it guards
    # Whether the guard has a floor is only knowable BEFORE `update_baseline` runs:
    # it bootstraps `baseline := active` when the file is missing or unparseable, so
    # afterwards a lost baseline is indistinguishable from a graph that legitimately
    # sits at its own count. That lost distinction is what let 2026-09-22 apply 30
    # merges while the report said "not degraded" (#1557).
    baseline_measured = baseline_path.is_file() and load_baseline(baseline_path) > 0
    baseline = update_baseline(len(active_edges), baseline_path)
    print(f"  Store:           {args.db}")
    print(f"  Baseline:        {baseline:,} active edges (now {len(active_edges):,})"
          + ("" if baseline_measured else "  [no baseline on disk before this run — guard unmeasured]"))
    gs = plan.get("gate_stats") or {}
    if gs.get("asked"):
        print(f"  Semantic gate:   {gs['asked']} suffix pairs judged — {gs['same']} clusters SAME, {gs['review']} to review")

    # Proposals from #67's weekly judge, if it has run. They are review input,
    # never an auto-merge: #67 lost its apply path on 2026-09-04. The
    # never-evaluated count comes from this sweep's own ledger — #67 accumulates
    # proposals across runs now, so a proposal shown here every 15 minutes for a
    # week and one that just arrived would otherwise look identical (#744).
    proposals, never_evaluated = surface_semantic_proposals(out_dir)
    if proposals:
        plan["semantic_proposals"] = proposals[:200]
        plan["semantic_proposals_never_evaluated"] = never_evaluated

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

    reason = degraded_reason(len(active_edges), baseline, measured=baseline_measured)
    if reason and not args.allow_degraded:
        print(f"\nREFUSING --apply: {reason}")
        return 3

    # Apply
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    # Claimed before the apply, not just named: the alias rows this run writes
    # carry this path as their provenance, so the FILE has to exist before they
    # can point at it, or a kill mid-run leaves provenance that resolves to
    # nothing (#475, `claim_report`).
    apply_out = out_dir / f"entity-merges-applied-{args.date}-{ts}.json"
    claim_report(apply_out, plan_out, args.date, ts)
    print()
    print(f"== Applying merges (timestamp: {ts}) ==")

    backup_dir = out_dir / "store-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    store_bak = st.backup(backup_dir / f"kg-sweep-{ts}.sqlite")
    prune_old_backups(store_bak, keep=5, pattern="kg-sweep-*.sqlite")
    print(f"  Backed up: {store_bak}")

    before = st.stats()
    report = apply_merges(plan, st, facts_root, args.rebuild_aliases,
                          existing_dirs=existing_dirs, entities=plan.get("all_entities", []),
                          report_path=str(apply_out))
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
        if op.get("skipped_unparseable"):
            print(f"    {op['variant']!r}: SKIPPED as unparseable, left in place: "
                  + ", ".join(op["skipped_unparseable"]))
    print(f"  Store: {before} → {after}")
    print()

    # Save an apply report
    apply_out.parent.mkdir(parents=True, exist_ok=True)
    report_to_save = {k: v for k, v in report.items() if k != "aliases"}
    # Replaces the `started` stub `claim_report` wrote before the transaction:
    # a reader who finds `started` on disk knows the run was killed, and knows no
    # counts were ever claimed by it.
    report_to_save["report_status"] = "complete"
    report_to_save["plan_file"] = str(plan_out)
    report_to_save["tiers_allowed"] = sorted(allowed_tiers)
    report_to_save["gate_stats"] = plan.get("gate_stats")
    # What this run actually did and under what checks: a reader must be able
    # to tell a guarded apply from a bypassed one from the report alone.
    report_to_save["applied_clusters"] = plan.get("safe_clusters", len(plan["safe_merges"]))
    report_to_save["applied_merges"] = len(report["variant_to_canonical"])
    report_to_save["safety"] = safety_record(
        args.no_gate, args.allow_degraded, gate, plan, reason,
        active_edges=len(active_edges), measured=baseline_measured)
    report_to_save["alias_provenance"] = {"origin": "sweep", "report_path": str(apply_out)}
    report_to_save["baseline_active_edges"] = baseline
    report_to_save["store_backup"] = str(store_bak)
    report_to_save["store_before"], report_to_save["store_after"] = before, after
    report_to_save["ledger"] = invocation_ledger()   # who ran this — see _invocation.py
    # Atomic replace, not `open("w")`: opening the claimed path in write mode
    # truncates the `started` stub first, so a run that dies part-way through the
    # dump leaves alias rows pointing at a half-written file that no longer parses.
    # That is the same defect `claim_report` was written to close, one window later.
    # Via tmp + os.replace the pointer resolves to the stub or to the complete
    # report, and never to something in between.
    # Dot-prefixed so neither this tool's nor the disposition audit's
    # `entity-merges-applied-*.json` glob can ever mistake a leftover tmp — the
    # shape a mid-write death leaves behind — for an apply report.
    tmp_report = apply_out.with_name(f".{apply_out.name}.tmp")
    with tmp_report.open("w") as f:
        json.dump(report_to_save, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_report, apply_out)
    print(f"  Report: {apply_out}")

    st.close()
    return 0


def _flush_djev_shadow() -> None:
    """Give the shadow recorder its chance before this process exits.

    The recorder's worker is a DAEMON thread, so a script takes its queue with
    it on the way out and this seam — ~302 clusters a day, the only one with
    real ground truth behind it — would be the one that never recorded
    anything. The aggregator does the same from `main.lifespan`; a script has
    to ask.
    """
    try:
        from app import djev_shadow
        left = djev_shadow.flush(timeout=30.0)
        if left:
            print(f"  [djev] {left} shadow rows unsent at exit "
                  f"(counted as dropped_at_shutdown)")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _flush_djev_shadow()
