#!/usr/bin/env python3
"""
Entity Resolution Sweep — Tier 1 (mechanical) for backlog #310.

Reads the live knowledge graph and entity directories, clusters entities by
normalized name, classifies each cluster as auto-mergeable (CASE / PUNCT /
SUFFIX) or ambiguous (LOOP / RESEARCH / OTHER), and either prints a plan
(dry-run) or applies the merges.

Apply mode:
  1. Backs up  _relationships.json  and  entity-aliases.json  with timestamp.
  2. For each SAFE merge:
       a. Moves fact files from variant dir into canonical dir (rename prefix).
       b. Rewrites edges: source/target variant → canonical; dedupes.
       c. Adds  {variant_lower: canonical}  to entity-aliases.json.
       d. Removes empty variant dir.
  3. Writes updated files.

Ambiguous clusters are dumped to a review JSONL for Tier 2 hand-review.

Usage:
  # dry-run (default):
  python entity-resolution-sweep.py

  # apply Tier 1:
  python entity-resolution-sweep.py --apply

  # also rewrite entity-aliases.json from scratch (drops fuzzy-match poison):
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

from app.paths import VAULT_FACTS_ROOT as FACTS_ROOT, VAULT_FACTS_ALIASES as ALIASES_PATH

REL_PATH = FACTS_ROOT / "_relationships.json"
OUT_DIR = Path.home() / "lloyd" / "_pipeline" / "memory-graph"

# ── Normalization ────────────────────────────────────────────────────────────

STOP_SUFFIX_TOKENS_SAFE = {
    # These suffixes are safe to strip for merge: X vs "X System" is basically
    # always the same thing.
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


def pick_canonical(variants: list[str], degrees: dict[str, int], existing_dirs: set[str]) -> str:
    """
    Pick the canonical name for a cluster.

    Priority:
      1. Prefer variant without a SAFE suffix (bare noun beats "X System", "X Agent").
         This matches the task's stated pattern (Idler Agent → idler,
         Data Pipeline System → Data Pipeline).
      2. Highest degree.
      3. Prefer variant whose directory already exists.
      4. Shorter name.
      5. Alphabetical (stable).
    """

    def key(v: str) -> tuple:
        return (
            1 if _has_safe_suffix(v) else 0,
            -degrees.get(v, 0),
            0 if v in existing_dirs else 1,
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
        if smallest_deg == 0:
            return (True, "SUFFIX_SAFE, a variant has 0 degree")
        if second_deg <= SUFFIX_SAFE_SMALL_VARIANT:
            return (True, f"SUFFIX_SAFE, second-largest variant is small ({second_deg})")
        ratio = top_deg / second_deg if second_deg else float("inf")
        if ratio >= SUFFIX_SAFE_RATIO:
            return (True, f"SUFFIX_SAFE, imbalance {ratio:.1f}× >= {SUFFIX_SAFE_RATIO}")
        return (False, f"SUFFIX_SAFE but imbalance {ratio:.1f}× < {SUFFIX_SAFE_RATIO}")

    if tier == "SUFFIX_AMBIGUOUS":
        # Generally hand-review, but allow when there's a clear bare-name
        # dominant (ratio >= threshold or second-largest is small).
        if smallest_deg == 0:
            return (True, "SUFFIX_AMBIG, a variant has 0 degree")
        if second_deg <= SUFFIX_SAFE_SMALL_VARIANT:
            return (True, f"SUFFIX_AMBIG, second-largest variant is small ({second_deg})")
        ratio = top_deg / second_deg if second_deg else float("inf")
        if ratio >= SUFFIX_SAFE_RATIO:
            return (True, f"SUFFIX_AMBIG, imbalance {ratio:.1f}× >= {SUFFIX_SAFE_RATIO}")
        return (False, f"SUFFIX_AMBIG, imbalance {ratio:.1f}× < {SUFFIX_SAFE_RATIO} — hand-review")

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


def build_plan(edges: list[dict], existing_dirs: set[str]) -> dict:
    """Compute clusters and classify into a merge plan."""
    # degree = appearances as source or target in active edges
    degrees: dict[str, int] = collections.Counter()
    for e in edges:
        degrees[e["source"]] += 1
        degrees[e["target"]] += 1

    # Include existing directory names so entities with zero active-edge degree
    # (e.g. "OpenClaw SDK", "Voice Mode System") are still discovered and can be
    # merged into their canonical partners.
    entities = list(set(degrees.keys()) | existing_dirs)

    # Three-stage clustering:
    #   Stage A: cluster by normalize_full (suffix-stripped) to surface candidates.
    #   Stage B: cluster by normalize_punct (case+only) to catch space/hyphen/underscore variants
    #            that diverge under suffix stripping (e.g. "Auto Research" vs "Autoresearch").
    #   Stage C: merge candidate sets, deduplicate, then classify tier.
    clusters_by_full: dict[str, list[str]] = collections.defaultdict(list)
    for ent in entities:
        key = normalize_full(ent)
        if key:
            clusters_by_full[key].append(ent)

    clusters_by_punct: dict[str, list[str]] = collections.defaultdict(list)
    for ent in entities:
        key = normalize_punct(ent)
        if key:
            clusters_by_punct[key].append(ent)

    # Merge candidate sets: entities that collide by either method go together
    ent_to_candidates: dict[str, set[str]] = collections.defaultdict(set)
    for group in clusters_by_full.values():
        if len(group) > 1:
            for ent in group:
                for other in group:
                    ent_to_candidates[ent].add(other)
    for group in clusters_by_punct.values():
        if len(group) > 1:
            for ent in group:
                for other in group:
                    ent_to_candidates[ent].add(other)

    # Build connected components from candidate edges
    visited = set()
    dupe_clusters: list[list[str]] = []
    for ent in entities:
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
            dupe_clusters.append(component)

    safe_merges: list[dict] = []
    ambiguous: list[dict] = []
    skipped: list[dict] = []

    for variants in dupe_clusters:
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

        norm_key = normalize_punct(canonical)
        base = {
            "norm_key": norm_key,
            "canonical": canonical,
            "variants": variant_degs,
            "tier": cluster_worst,
            "decision": decide_reason,
        }

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
        if float(fact.get("confidence", 0)) > float(existing.get("confidence", 0)):
            seen[text] = fact
        elif float(fact.get("confidence", 0)) == float(existing.get("confidence", 0)):
            if fact.get("created_at", "") > existing.get("created_at", ""):
                seen[text] = fact
    return list(seen.values())


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
    dst.write_text(_dump_frontmatter(dst_fm, body), encoding="utf-8")


def backup_file(path: Path, timestamp: str) -> Path:
    if not path.exists():
        return path
    bak = path.with_suffix(path.suffix + f".{timestamp}.bak")
    shutil.copy2(path, bak)
    return bak


def apply_merges(
    plan: dict,
    rel_data: dict,
    aliases: dict[str, str],
    facts_root: Path,
    rebuild_aliases: bool,
) -> dict:
    """
    Execute all SAFE merges in the plan. Mutates rel_data and aliases in place.

    Returns a report dict.
    """
    edges = rel_data["edges"]

    # Build a fast mapping: variant → canonical for this session's merges
    variant_to_canonical: dict[str, str] = {}
    dir_moves: list[tuple[Path, Path]] = []
    for cluster in plan["safe_merges"]:
        canonical = cluster["canonical"]
        for m in cluster["merges"]:
            variant_to_canonical[m["variant"]] = canonical

    # ── Rewrite edges
    # We need to dedupe after rewriting: (source, target, type) should be unique
    # but we preserve both active and expired edges. Only merge across ACTIVE.
    seen_active: dict[tuple, dict] = {}
    new_edges = []
    rewrite_count = 0
    active_dedupe_count = 0

    for e in edges:
        src = variant_to_canonical.get(e["source"], e["source"])
        tgt = variant_to_canonical.get(e["target"], e["target"])
        if src != e["source"] or tgt != e["target"]:
            rewrite_count += 1
        e_new = dict(e)
        e_new["source"] = src
        e_new["target"] = tgt

        active = e_new.get("expired_at") is None
        if active:
            key = (src, tgt, e_new.get("type"))
            if key in seen_active:
                # Merge: prefer higher confidence; keep earliest created_at
                prev = seen_active[key]
                prev["confidence"] = max(
                    prev.get("confidence", 0.0), e_new.get("confidence", 0.0)
                )
                if e_new.get("created_at", "") < prev.get("created_at", ""):
                    prev["created_at"] = e_new["created_at"]
                active_dedupe_count += 1
                continue
            seen_active[key] = e_new
        new_edges.append(e_new)

    rel_data["edges"] = new_edges

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
            moved += 1
        # Remove variant dir if empty
        removed = False
        try:
            vdir.rmdir()
            removed = True
        except OSError:
            pass
        dir_ops.append(
            {
                "variant": variant,
                "canonical": canonical,
                "files_moved": moved,
                "removed_dir": removed,
            }
        )

    # ── Update alias table
    # Always filter out self-referential noise entries on each apply run.
    # Per skill guardrail: any entry where normalize_full(alias) ==
    # normalize_full(canonical) is pipeline noise.
    new_aliases = {
        k: v for k, v in aliases.items()
        if k.strip().lower() != v.strip().lower()
    }

    for variant, canonical in variant_to_canonical.items():
        new_aliases[variant.lower()] = canonical
        # Also ensure exact case variants resolve
        new_aliases[variant] = canonical

    # Canonicals should self-resolve (identity) — these are useful for the
    # lookup path (alias_lower -> canonical) so we keep them, but only for
    # entities actually involved in this sweep's merges.
    for canonical in set(variant_to_canonical.values()):
        new_aliases[canonical.lower()] = canonical

    return {
        "rewritten_edges": rewrite_count,
        "active_dedupe_count": active_dedupe_count,
        "dir_operations": dir_ops,
        "variant_to_canonical": variant_to_canonical,
        "aliases_final_count": len(new_aliases),
        "aliases": new_aliases,
    }


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Apply SAFE merges")
    ap.add_argument(
        "--rebuild-aliases",
        action="store_true",
        help="Rewrite entity-aliases.json from scratch (drop existing entries). Requires --apply.",
    )
    ap.add_argument("--relationships", default=str(REL_PATH))
    ap.add_argument("--facts-dir", default=str(FACTS_ROOT))
    ap.add_argument("--aliases", default=str(ALIASES_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    rel_path = Path(args.relationships)
    facts_root = Path(args.facts_dir)
    aliases_path = Path(args.aliases)
    out_dir = Path(args.out_dir)

    # Load data
    with rel_path.open() as f:
        rel_data = json.load(f)
    active_edges = [e for e in rel_data["edges"] if e.get("expired_at") is None]

    existing_dirs = (
        {d.name for d in facts_root.iterdir() if d.is_dir()} if facts_root.exists() else set()
    )

    aliases: dict[str, str] = {}
    if aliases_path.exists():
        with aliases_path.open() as f:
            aliases = json.load(f)

    plan = build_plan(active_edges, existing_dirs)

    # Emit plan
    print_plan(plan)
    plan_out = out_dir / f"entity-merges-{args.date}.jsonl"
    write_plan_jsonl(plan, plan_out)
    print(f"Plan written: {plan_out}")

    if not args.apply:
        print()
        print(f"(dry-run — pass --apply to execute {plan['safe_clusters']} SAFE merges)")
        return 0

    # Apply
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    print()
    print(f"== Applying merges (timestamp: {ts}) ==")

    rel_bak = backup_file(rel_path, ts)
    alias_bak = backup_file(aliases_path, ts) if aliases_path.exists() else None
    print(f"  Backed up: {rel_bak}")
    if alias_bak:
        print(f"  Backed up: {alias_bak}")

    report = apply_merges(plan, rel_data, aliases, facts_root, args.rebuild_aliases)

    # Persist
    with rel_path.open("w") as f:
        json.dump(rel_data, f, indent=2, ensure_ascii=False)
    with aliases_path.open("w") as f:
        json.dump(report["aliases"], f, indent=2, sort_keys=True, ensure_ascii=False)

    # Report
    print(f"  Edges rewritten:   {report['rewritten_edges']}")
    print(f"  Active dedupes:    {report['active_dedupe_count']}")
    print(f"  Dir operations:    {len(report['dir_operations'])}")
    for op in report["dir_operations"]:
        if op.get("action") == "skip_no_variant_dir":
            print(f"    {op['variant']!r}: no dir, skipped")
        else:
            print(
                f"    {op['variant']!r} → {op['canonical']!r}: "
                f"moved {op['files_moved']} files, removed_dir={op['removed_dir']}"
            )
    print(f"  Alias table size:  {report['aliases_final_count']}")
    print()

    # Save an apply report
    apply_out = out_dir / f"entity-merges-applied-{args.date}-{ts}.json"
    apply_out.parent.mkdir(parents=True, exist_ok=True)
    # Strip the giant aliases dict from the written report, keep just what changed
    report_to_save = {k: v for k, v in report.items() if k != "aliases"}
    with apply_out.open("w") as f:
        json.dump(report_to_save, f, indent=2, ensure_ascii=False)
    print(f"  Report: {apply_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
