#!/usr/bin/env python3
"""
Knowledge Health Report Generator

Analyzes the fact store and relationship graph to produce a markdown
health report. Designed to be run by the nightly cron system.

Inputs:
  - Facts dir + relationships index resolved via app.paths.VAULT_FACTS_ROOT
    (currently ~/lloyd/_pipeline/vault-derived/facts/)

Output:
  - ~/lloyd/_pipeline/reflection/knowledge-health-YYYY-MM-DD.md
"""

import argparse
import json
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR, VAULT_KG_DB
from app.kg_store import StoreUnavailable, store as _kg_store

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "_pipeline" / "reflection"

# Rows rendered per long section. The report is read by a human and by the
# morning briefing; a 23,564-row table is neither. The count is always given
# in full, only the listing is capped.
SECTION_ROW_CAP = 50

GOD_ENTITY_THRESHOLD = 20
THIN_ENTITY_MAX_FACTS = 2
STALE_DAYS_THRESHOLD = 60

# Questions, not a listing — 20 is plenty for the section a research worker
# reads out of `## Suggested Research Questions`.
RESEARCH_QUESTION_COUNT = 20

# Below this many names that survive the artifact filter there is no ranking to
# report, and the section says so instead of filling the slot with whatever the
# extractor left behind (#954). 10 = half the section: fewer than that and the
# questions would be dominated by whatever passed, not by what is thinnest.
MIN_USABLE_RESEARCH_QUESTIONS = 10

# Entity names that are an extraction artifact rather than a thing anyone can
# research. On 2026-09-19 every one of the 20 questions in the live report
# matched one of these: 20 of 20 were `#NNN` backlog ids that had been minted as
# entities (`#160`, `#165`, `#222` …), each of which resolves to an existing
# `~/obsidian/backlog/<id>-*.md`. `#` is byte 0x23 — the lowest printable
# character — so under the old alphabetical-by-load order they were *always* at
# the top of the section. Stopping the minting is #743; this filter is what
# makes the section useless-to-useful before that gate lands and harmless after
# it.
EXTRACTION_ARTIFACT_NAME_RES = (
    re.compile(r"^#\d"),                # '#441' — a backlog id, not an entity
    re.compile(r"^--"),                 # '--continue' — a CLI flag
    re.compile(r"\.md$"),               # '03-linear-representation-hypothesis.md' — a filename
    re.compile(r"^\d{4}-\d{2}-\d{2}"),  # '2026-09-11' — a date
)


def is_extraction_artifact_name(name: str) -> bool:
    """True when an entity name is a shape the extractor minted, not a concept.

    Only the questions are filtered by this. The `## Thin Entities` table is not:
    it is the surface on which a regrowth in `#NNN` minting becomes visible, so
    hiding those rows would destroy the very signal #743 needs.
    """
    return any(pattern.search(name) for pattern in EXTRACTION_ARTIFACT_NAME_RES)


def thin_entity_rank(item) -> tuple:
    """Order thin entities by how thin they are, then by recency, then by name.

    Every thin entity has the same `active_facts` (the section selects
    `< THIN_ENTITY_MAX_FACTS`, so it is 0 or 1), which made the old key
    `x[1]["active_facts"]` a total tie: the sort is stable, `load_entities` fills
    its dict from `sorted(facts_dir.iterdir())`, and the surviving order — and so
    the whole questions section — was alphabetical by entity directory name
    (#954). `latest_created` is the fix: it is computed from facts already
    loaded, so the tie now breaks on "newest fact first" with no extra I/O. An
    entity with no dated fact sorts after every dated one, and the name is the
    final key so the order never depends on dict insertion order.
    """
    name, stats = item
    newest = stats.get("latest_created")
    return (stats["active_facts"], -(newest.timestamp() if newest else float("-inf")), name)


def parse_frontmatter(file_path: Path) -> dict | None:
    """Parse YAML frontmatter from a markdown file.

    Returns the parsed dict or None if no valid frontmatter is found.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if not text.startswith("---"):
        return None

    end = text.find("---", 3)
    if end == -1:
        return None

    frontmatter_text = text[3:end]
    try:
        return yaml.safe_load(frontmatter_text)
    except yaml.YAMLError:
        return None


def is_fact_active(fact: dict) -> bool:
    """A fact is active if it has no expired_at and no invalid_at set."""
    expired = fact.get("expired_at")
    invalid = fact.get("invalid_at")
    return not expired and not invalid


def load_entities(facts_dir: Path) -> dict:
    """Load all entities and their facts from the facts directory.

    Returns a dict mapping entity name -> list of dicts, each with:
      { "category": str, "facts": list[dict], "file": Path }
    """
    entities: dict[str, list[dict]] = defaultdict(list)

    if not facts_dir.is_dir():
        print(f"Warning: facts directory not found: {facts_dir}", file=sys.stderr)
        return entities

    for entity_dir in sorted(facts_dir.iterdir()):
        if not entity_dir.is_dir():
            continue
        # Skip hidden/internal directories
        if entity_dir.name.startswith(".") or entity_dir.name.startswith("_"):
            continue

        entity_name = entity_dir.name

        for md_file in sorted(entity_dir.glob("*.md")):
            fm = parse_frontmatter(md_file)
            if not fm or fm.get("type") != "facts":
                continue

            category = fm.get("category", "unknown")
            facts_list = fm.get("facts", [])
            if not isinstance(facts_list, list):
                facts_list = []

            entities[entity_name].append({
                "category": category,
                "facts": facts_list,
                "file": md_file,
            })

    return entities


def load_relationships(_unused: Path | None = None) -> list[dict]:
    """Every edge in the store, expired ones included (the report counts both).

    Raises `StoreUnavailable` if the store cannot be read — this report is a
    monitor, and a monitor that silently reports zero edges when it cannot
    see the graph is worse than one that fails.
    """
    return _kg_store().edges.all()


def is_edge_active(edge: dict) -> bool:
    """An edge is active if it has no expired_at set."""
    return not edge.get("expired_at")


def compute_entity_stats(entities: dict) -> dict:
    """Compute per-entity aggregate statistics."""
    stats = {}
    for entity_name, category_entries in entities.items():
        all_facts = []
        categories = set()
        for entry in category_entries:
            all_facts.extend(entry["facts"])
            categories.add(entry["category"])

        active = [f for f in all_facts if is_fact_active(f)]
        expired = [f for f in all_facts if not is_fact_active(f)]
        # The newest `created_at` on the entity — when this entity was last
        # touched. A max, not the first one found, and None when no fact carries
        # a parseable date. Thin entities are all tied on `active_facts`, so this
        # is the field that gives the ranking in `thin_entity_rank` anything to
        # break a tie with (#954).
        fact_dates = [
            parse_date(fact.get("created_at"))
            for fact in all_facts if isinstance(fact, dict)
        ]
        dated = [d for d in fact_dates if d]

        stats[entity_name] = {
            "total_facts": len(all_facts),
            "active_facts": len(active),
            "expired_facts": len(expired),
            "latest_created": max(dated) if dated else None,
            "categories": sorted(categories),
            "facts": all_facts,
            "category_entries": category_entries,
        }
    return stats


def compute_relationship_stats(edges: list[dict], entities: dict) -> dict:
    """Compute relationship statistics.

    Returns a dict with:
      - entity_edge_counts: {entity_name: count_of_active_edges}
      - type_distribution: {type: count}
      - active_count: int
      - expired_count: int
      - entities_in_graph: set of entity names appearing in edges
    """
    entity_edge_counts: dict[str, int] = defaultdict(int)
    type_distribution: dict[str, int] = defaultdict(int)
    active_count = 0
    expired_count = 0
    entities_in_graph: set[str] = set()

    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        edge_type = edge.get("type", "unknown")

        entities_in_graph.add(source)
        entities_in_graph.add(target)

        if is_edge_active(edge):
            active_count += 1
            entity_edge_counts[source] += 1
            entity_edge_counts[target] += 1
            type_distribution[edge_type] += 1
        else:
            expired_count += 1

    return {
        "entity_edge_counts": dict(entity_edge_counts),
        "type_distribution": dict(type_distribution),
        "active_count": active_count,
        "expired_count": expired_count,
        "entities_in_graph": entities_in_graph,
    }


def parse_date(date_str: str | None) -> datetime | None:
    """Parse an ISO-format date string into a datetime object."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(str(date_str))
        # Ensure timezone-aware for comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def stale_age_reference(fact: dict) -> datetime | None:
    """The date a fact's age is measured from, or None when it has no usable one.

    `created_at` is when the row was recorded; `event_date` is when the claim was
    true in the world. The oldest usable date wins, so a fact recorded last week
    about an event 200 days ago ages at ~200 days. Reading only `created_at` was
    the #841 defect: the 2026-09-03 rebuild stamped it on roughly a third of the
    store and on nothing older than the rebuild itself, so the 60-day threshold
    was unreachable by construction and the detector returned 0 on every run.

    `valid_at` RE-BASES the clock rather than exempting the fact: a claim
    re-affirmed inside the threshold is young again, but one whose `valid_at` is
    itself older than the threshold stays reportable. The old code skipped every
    fact carrying `valid_at`, which made the field a permanent amnesty — 19,916
    active facts on 2026-09-16 were invisible for that reason alone.

    A re-affirmation can only move the clock FORWARD: `created_at`/`event_date`
    alone decide the base, and `valid_at` overrides it only when it is the newer
    of the two. A fact recorded last week that asserts something which became
    valid 250 days ago is a week old, not 250.
    """
    dates = [d for d in (parse_date(fact.get("created_at")),
                         parse_date(fact.get("event_date"))) if d]
    if not dates:
        return None
    reference = min(dates)
    valid_at = parse_date(fact.get("valid_at"))
    if valid_at and valid_at > reference:
        return valid_at
    return reference


def find_stale_facts(entities: dict, now: datetime, threshold_days: int) -> list[dict]:
    """Active facts whose oldest usable date is older than threshold_days.

    A fact carrying no usable date is NOT returned: it is not fresh, it is
    unmeasured. `stale_coverage` counts those so the report can name them
    instead of printing a clean verdict over an unseen share of the store.
    """
    stale = []
    for entity_name, category_entries in entities.items():
        for entry in category_entries:
            category = entry["category"]
            for fact in entry["facts"]:
                if not is_fact_active(fact):
                    continue

                reference = stale_age_reference(fact)
                if reference is None:
                    continue

                age = (now - reference).days
                if age >= threshold_days:
                    fact_text = str(fact.get("fact", ""))
                    preview = fact_text[:60] + ("..." if len(fact_text) > 60 else "")
                    stale.append({
                        "entity": entity_name,
                        "category": category,
                        "preview": preview,
                        "age_days": age,
                    })
    return stale


def stale_coverage(entities: dict) -> tuple[int, int]:
    """Return (active facts with no usable date, active facts) — the blind share.

    The denominator the stale check cannot evaluate. A monitor that reports
    "No stale facts found" while 45% of the store carries no date is reporting a
    verdict on an input it could not read, which is the #841 defect class; this
    is the number that makes that state visible instead of clean. Counted over
    active facts, the same population `find_stale_facts` walks.
    """
    unevaluable = 0
    active_total = 0
    for _entity_name, category_entries in entities.items():
        for entry in category_entries:
            for fact in entry["facts"]:
                if not is_fact_active(fact):
                    continue
                active_total += 1
                if stale_age_reference(fact) is None:
                    unevaluable += 1
    return unevaluable, active_total


def compute_hygiene(entities: dict, now: datetime, regrowth_days: int = 7) -> dict:
    """Contamination, near-duplicate clusters and regrowth.

    Delegates to `kg_hygiene.snapshot`, which is the measured definition of
    all three. This module had its own re-implementation of each — same
    intent, different code — so the report and `kg_health --json` could
    disagree about the same tree and there was no way to tell which was
    right. The shapes the report renders are kept.
    """
    import importlib.util
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import kg_hygiene  # noqa: E402

    root = _facts_root_for(entities)
    # Each of these walks the tree parsing YAML. Call contamination ONCE and
    # derive both the counts and the detail from it — `snapshot` would run it
    # again internally, and at 60,622 files that is a minute of pure re-read.
    detail = kg_hygiene.contamination(root)
    c = {k: v for k, v in detail.items() if k != "items"}
    n = kg_hygiene.near_duplicates(root)
    r = kg_hygiene.regrowth(root, regrowth_days)
    contaminated = [(item["dir"], tag, slot["facts"])
                    for item in detail["items"]
                    for tag, slot in item["foreign"].items()]

    sweep = kg_hygiene.sweep()
    regrown: list[tuple[str, str, str]] = []
    for name in r.get("samples", []):
        older = [o for o in entities
                 if o != name
                 and sweep.normalize_full(o) == sweep.normalize_full(name)]
        if older:
            regrown.append((name, older[0], sweep.classify_pair(name, older[0])[0]))

    return {
        "contaminated": contaminated,
        "contaminated_dirs": c["dirs"],
        "foreign_facts": c["foreign_facts"],
        "near_dup_clusters": n["clusters"],
        "near_dup_dirs": n["dirs"],
        "near_dup_tiers": n["by_tier"],
        "regrown": regrown,
        "new_dirs": r["new_dirs"],
        "regrowth_days": r["days"],
        "provenance": kg_hygiene.provenance_coverage(root),
    }


def _facts_root_for(entities: dict) -> Path:
    """The tree the loaded entities came from."""
    for files in entities.values():
        for entry in files:
            return Path(entry["file"]).parent.parent
    return FACTS_DIR


def fact_duplicate_stats() -> dict:
    """Exact-duplicate FACT totals, read from `facts_idx.text_hash`.

    Nothing else in this report measures a duplicated fact. The line that reads
    "Near-duplicate name clusters" comes from `kg_hygiene.near_duplicates`,
    which clusters entity-name DIRECTORIES (`_clusters(root)`, on `d.name`) — so
    a report could print zero near-duplicates while the store held thousands of
    rows whose text was identical (#499: 11,775 duplicate rows measured
    2026-09-12). This is the fact-level number the trend is read on.

    Counted over every indexed row, expired and invalid included: a retired row
    is still a row ingestion wrote, and #499's guard forbids expiring anything,
    so retirement is not in this denominator either way.

    An unreadable store reports `unavailable` with the reason, never a 0 —
    a check that prints a verdict for an input it could not read is the exact
    defect class #499 is catalogued under.
    """
    try:
        return {"unavailable": False, **_kg_store().facts_idx.exact_duplicate_stats()}
    except StoreUnavailable as e:
        return {"unavailable": True, "reason": str(e)}


def _fact_duplicate_cell(d: dict) -> str:
    """One table cell: the total, with the denominator it is a fraction of."""
    if d.get("unavailable"):
        return f"not measured: {d.get('reason', 'store unavailable')}"
    return (f"{d['same_entity_redundant_rows']} redundant of {d['rows']} rows "
            f"(entities with an exact twin: {d['entities_with_exact_dupes']}; "
            f"distinct texts: {d['distinct_texts']})")


def generate_report(
    entity_stats: dict,
    rel_stats: dict,
    edges: list[dict],
    stale_facts: list[dict],
    now: datetime,
    hygiene: dict | None = None,
    fact_dups: dict | None = None,
    stale_unevaluable: tuple[int, int] | None = None,
) -> str:
    """Generate the markdown health report.

    `stale_unevaluable` is the `(n, m)` pair from `stale_coverage`: how many of
    the m active facts carry no date the stale check can age them from. Omitting
    it means the caller did not measure it, and the section says so rather than
    claiming the store is clean.
    """
    lines: list[str] = []
    date_str = now.strftime("%Y-%m-%d")

    lines.append(f"# Knowledge Health Report - {date_str}")
    lines.append("")

    # --- Section 1: Summary Stats ---
    total_entities = len(entity_stats)
    total_facts = sum(s["total_facts"] for s in entity_stats.values())
    active_facts = sum(s["active_facts"] for s in entity_stats.values())
    expired_facts = sum(s["expired_facts"] for s in entity_stats.values())
    active_rels = rel_stats["active_count"]
    expired_rels = rel_stats["expired_count"]
    total_rels = active_rels + expired_rels

    lines.append("## Summary Stats")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total entities | {total_entities} |")
    lines.append(f"| Total facts | {total_facts} |")
    lines.append(f"| Active facts | {active_facts} |")
    lines.append(f"| Expired facts | {expired_facts} |")
    # Fact-level duplicates, from `facts_idx.text_hash`. Kept apart from the
    # Hygiene section's "Near-duplicate name clusters" line on purpose: that one
    # counts entity-name directories, this one counts rows whose text is the
    # same claim twice. See `fact_duplicate_stats`.
    lines.append(f"| Exact-duplicate fact rows (facts_idx.text_hash) | "
                 f"{_fact_duplicate_cell(fact_dups or {'unavailable': True, 'reason': 'not computed'})} |")
    lines.append(f"| Total relationships | {total_rels} |")
    lines.append(f"| Active relationships | {active_rels} |")
    lines.append(f"| Expired relationships | {expired_rels} |")
    lines.append("")

    # --- Section 2: God Entities ---
    god_entities = [
        (name, s)
        for name, s in entity_stats.items()
        if s["total_facts"] > GOD_ENTITY_THRESHOLD
    ]
    god_entities.sort(key=lambda x: x[1]["total_facts"], reverse=True)

    lines.append("## God Entities")
    lines.append("")
    lines.append(f"Entities with >{GOD_ENTITY_THRESHOLD} facts. May need splitting or are heavily documented.")
    lines.append("")

    if god_entities:
        lines.append(f"| Entity | Fact Count | Categories |  <!-- top {SECTION_ROW_CAP} -->")
        lines.append("|--------|-----------|------------|")
        for name, s in god_entities[:SECTION_ROW_CAP]:
            cats = ", ".join(s["categories"])
            lines.append(f"| {name} | {s['total_facts']} | {cats} |")
        if len(god_entities) > SECTION_ROW_CAP:
            lines.append(f"| … | *{len(god_entities) - SECTION_ROW_CAP} more* | |")
    else:
        lines.append("*No god entities found.*")
    lines.append("")

    # --- Section 3: Thin Entities ---
    edge_counts = rel_stats["entity_edge_counts"]
    thin_entities = [
        (name, s)
        for name, s in entity_stats.items()
        if s["active_facts"] < THIN_ENTITY_MAX_FACTS
        and edge_counts.get(name, 0) == 0
    ]
    # Not `key=active_facts`: that key was a total tie across every thin entity,
    # which left the section ordered by entity name and rendered the same 20
    # ASCII-first names — all of them `#NNN` backlog ids — every single night
    # (#954). See `thin_entity_rank` for what the tie now breaks on.
    thin_entities.sort(key=thin_entity_rank)

    lines.append("## Thin Entities")
    lines.append("")
    lines.append(f"Entities with <{THIN_ENTITY_MAX_FACTS} active facts and 0 active relationships. Knowledge gaps.")
    lines.append("")

    if thin_entities:
        lines.append(f"**{len(thin_entities):,}** in total; the {min(SECTION_ROW_CAP, len(thin_entities))} thinnest:")
        lines.append("")
        lines.append("| Entity | Active Facts |")
        lines.append("|--------|-------------|")
        for name, s in thin_entities[:SECTION_ROW_CAP]:
            lines.append(f"| {name} | {s['active_facts']} |")
        if len(thin_entities) > SECTION_ROW_CAP:
            lines.append(f"| … | *{len(thin_entities) - SECTION_ROW_CAP:,} more* |")
    else:
        lines.append("*No thin entities found.*")
    lines.append("")

    # --- Section 4: Orphan Entities ---
    entities_in_graph = rel_stats["entities_in_graph"]
    orphan_entities = [
        (name, s)
        for name, s in entity_stats.items()
        if name not in entities_in_graph
        and s["total_facts"] > 0
    ]
    orphan_entities.sort(key=lambda x: x[1]["total_facts"], reverse=True)

    lines.append("## Orphan Entities")
    lines.append("")
    lines.append("Entities with facts but zero relationships. Candidates for relationship wiring.")
    lines.append("")

    if orphan_entities:
        lines.append(f"**{len(orphan_entities):,}** in total; the {min(SECTION_ROW_CAP, len(orphan_entities))} largest:")
        lines.append("")
        lines.append("| Entity | Fact Count |")
        lines.append("|--------|-----------|")
        for name, s in orphan_entities[:SECTION_ROW_CAP]:
            lines.append(f"| {name} | {s['total_facts']} |")
        if len(orphan_entities) > SECTION_ROW_CAP:
            lines.append(f"| … | *{len(orphan_entities) - SECTION_ROW_CAP:,} more* |")
    else:
        lines.append("*No orphan entities found.*")
    lines.append("")

    # --- Section 5: Relationship Type Distribution ---
    type_dist = rel_stats["type_distribution"]

    lines.append("## Relationship Type Distribution")
    lines.append("")

    if type_dist:
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for edge_type, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {edge_type} | {count} |")
    else:
        lines.append("*No relationships found.*")
    lines.append("")

    # --- Section 6: Stale Facts ---
    lines.append("## Stale Facts")
    lines.append("")
    lines.append(f"Active facts whose oldest usable date — `created_at` or `event_date`, "
                 f"re-based by `valid_at` — is older than {STALE_DAYS_THRESHOLD} days.")
    lines.append("")

    # The clean verdict is earned only when both numbers are zero. An unmeasured
    # or undatable share is a finding in its own right, not a pass (#841).
    if stale_unevaluable is None:
        lines.append("STALE_UNEVALUABLE: coverage not measured, so no clean verdict is available "
                     "from this section.")
        lines.append("")
    else:
        n_unevaluable, n_active = stale_unevaluable
        if n_unevaluable:
            pct = round(100.0 * n_unevaluable / n_active, 1) if n_active else 100.0
            lines.append(f"STALE_UNEVALUABLE: {n_unevaluable:,} of {n_active:,} facts carry no usable "
                         f"date ({pct}%). These are unmeasured, not fresh.")
            lines.append("")

    if stale_facts:
        stale_sorted = sorted(stale_facts, key=lambda x: x["age_days"], reverse=True)
        lines.append(f"**{len(stale_sorted):,}** in total; the {min(SECTION_ROW_CAP, len(stale_sorted))} oldest:")
        lines.append("")
        lines.append("| Entity | Category | Fact Preview | Age (days) |")
        lines.append("|--------|----------|-------------|-----------|")
        for sf in stale_sorted[:SECTION_ROW_CAP]:
            # Escape pipe characters in preview text
            preview = sf["preview"].replace("|", "\\|")
            lines.append(f"| {sf['entity']} | {sf['category']} | {preview} | {sf['age_days']} |")
        if len(stale_sorted) > SECTION_ROW_CAP:
            lines.append(f"| … | | *{len(stale_sorted) - SECTION_ROW_CAP:,} more* | |")
    elif stale_unevaluable is not None and stale_unevaluable[0] == 0:
        # Reached only with no stale facts AND nothing left unevaluable: every
        # active fact was aged and none crossed the threshold. That is the whole
        # of what this verdict is allowed to claim.
        lines.append("*No stale facts found.*")
    lines.append("")

    # --- Section 7: Suggested Research Questions ---
    # --- Hygiene ---
    if hygiene:
        lines.append("## Hygiene")
        lines.append("")
        lines.append("| Metric | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| Contaminated entity dirs (facts tagged with another entity) | {hygiene['contaminated_dirs']} |")
        lines.append(f"| Foreign facts | {hygiene['foreign_facts']} |")
        lines.append(f"| Near-duplicate name clusters | {hygiene['near_dup_clusters']} ({hygiene['near_dup_dirs']} dirs; {hygiene['near_dup_tiers']}) |")
        lines.append(f"| Near-duplicates born in the last {hygiene['regrowth_days']} days | {len(hygiene['regrown'])} of {hygiene['new_dirs']} new dirs |")
        lines.append("")
        if hygiene["contaminated"]:
            lines.append("**Contamination must be 0.** A rise means an entity merge fused two different things; "
                         "revert it with `scripts/memory/revert-suffix-merges.py`. Worst offenders:")
            lines.append("")
            for name, tag, n in sorted(hygiene["contaminated"], key=lambda x: -x[2])[:10]:
                lines.append(f"- `{name}` holds {n} fact(s) tagged `{tag}`")
            lines.append("")
        if hygiene["regrown"]:
            lines.append("Recent near-duplicates (extraction coined a name next to an existing one):")
            lines.append("")
            for n, o, tier in hygiene["regrown"][:10]:
                lines.append(f"- `{n}` next to `{o}` ({tier})")
            lines.append("")

    lines.append("## Suggested Research Questions")
    lines.append("")

    if not thin_entities:
        lines.append("*No thin entities to generate questions for.*")
    else:
        usable = [(name, s) for name, s in thin_entities
                  if not is_extraction_artifact_name(name)]
        artifact_count = len(thin_entities) - len(usable)
        if len(usable) < MIN_USABLE_RESEARCH_QUESTIONS:
            # A missing usable input is a finding, not a pass — the same rule the
            # stale section applies to an undatable store (#906 clause 11). Printing
            # the survivors anyway would hand `research-queue-generator` a section
            # that reads as a ranking and is not one.
            lines.append(f"RESEARCH_QUESTIONS_UNEVALUABLE: {artifact_count} of "
                         f"{len(thin_entities)} thin entities carry an artifact-shaped name")
            lines.append("")
            lines.append(f"*Only {len(usable)} of {len(thin_entities):,} thin entities carry a name "
                         f"that survives the artifact filter, below the "
                         f"{MIN_USABLE_RESEARCH_QUESTIONS} needed to call this a ranking. The "
                         f"population is still listed in full above, and the minting that "
                         f"produced these names is owned by #743.*")
        else:
            for name, s in usable[:RESEARCH_QUESTION_COUNT]:
                lines.append(f"- What does **{name}** relate to?")
                lines.append(f"- Is **{name}** still relevant?")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated at {now.isoformat()} by knowledge-health-report.py*")
    lines.append("")

    return "\n".join(lines)


# A monitor that reports success when it cannot see the thing it monitors is
# worse than no monitor. These are the states that mean "stop and look".
EXIT_OK = 0
EXIT_ALARM = 2


def _alarms(store_stats: dict | None, hygiene: dict, duplicate_id_files: int,
            baseline: int) -> list[str]:
    """Conditions that make this run an alarm rather than a report."""
    out = []
    if store_stats is None:
        out.append("the knowledge-graph store could not be read")
        return out
    active = store_stats.get("edges_active", 0)
    if baseline > 0 and active < baseline * 0.5:
        out.append(f"active edges {active:,} is below 50% of the baseline {baseline:,}")
    if hygiene.get("contaminated_dirs"):
        out.append(f"{hygiene['contaminated_dirs']} directories hold facts about another entity "
                   f"({hygiene['foreign_facts']} facts) — a merge went wrong")
    if duplicate_id_files:
        out.append(f"{duplicate_id_files} fact files carry duplicate fact IDs")
    return out


def _duplicate_id_files(entities: dict) -> int:
    """Fact files where one ID names two facts.

    43% of files were in this state on 2026-09-03 because the extractor
    restarted its numbering each run. Anything that addresses a fact by ID
    then acts on whichever it finds first.
    """
    bad = 0
    for files in entities.values():
        for entry in files:
            ids = [f.get("id") for f in entry["facts"] if isinstance(f, dict) and f.get("id")]
            if len(ids) != len(set(ids)):
                bad += 1
    return bad


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Knowledge Health Report from the fact store and relationship graph."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the report (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--facts-dir",
        type=Path,
        default=FACTS_DIR,
        help=f"Facts directory (default: {FACTS_DIR})",
    )
    parser.add_argument(
        "--no-alarm-exit", action="store_true",
        help="Always exit 0, even on an alarm condition (for ad-hoc runs)",
    )

    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # Load data
    print(f"Loading entities from {args.facts_dir} ...")
    entities = load_entities(args.facts_dir)
    print(f"  Found {len(entities)} entities")

    store_stats = None
    try:
        edges = load_relationships()
        store_stats = _kg_store().stats()
        print(f"  Store: {store_stats}")
    except StoreUnavailable as exc:
        edges = []
        print(f"  STORE UNAVAILABLE: {exc}", file=sys.stderr)
    print(f"  Found {len(edges)} edges")

    # Compute stats
    entity_stats = compute_entity_stats(entities)
    rel_stats = compute_relationship_stats(edges, entities)
    stale_facts = find_stale_facts(entities, now, STALE_DAYS_THRESHOLD)
    stale_unevaluable = stale_coverage(entities)

    hygiene = compute_hygiene(entities, now)
    fact_dups = fact_duplicate_stats()
    dup_id_files = _duplicate_id_files(entities)

    # Generate report
    report = generate_report(entity_stats, rel_stats, edges, stale_facts, now, hygiene,
                             fact_dups=fact_dups, stale_unevaluable=stale_unevaluable)

    # Write output
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"knowledge-health-{date_str}.md"
    output_file.write_text(report, encoding="utf-8")

    print(f"Report written to {output_file}")
    print(f"  Entities: {len(entity_stats)}")
    print(f"  God entities: {sum(1 for s in entity_stats.values() if s['total_facts'] > GOD_ENTITY_THRESHOLD)}")
    print(f"  Thin entities: {sum(1 for name, s in entity_stats.items() if s['active_facts'] < THIN_ENTITY_MAX_FACTS and rel_stats['entity_edge_counts'].get(name, 0) == 0)}")
    print(f"  Orphan entities: {sum(1 for name in entity_stats if name not in rel_stats['entities_in_graph'] and entity_stats[name]['total_facts'] > 0)}")
    print(f"  Stale facts: {len(stale_facts)}")
    # On its own line because the count above is only a verdict on the dated
    # share; this is the share it could not age (#841).
    print(f"  Stale unevaluable: {stale_unevaluable[0]:,} of {stale_unevaluable[1]:,} active facts "
          f"carry no usable date")
    print(f"  Contaminated dirs: {hygiene['contaminated_dirs']} ({hygiene['foreign_facts']} foreign facts)")
    print(f"  Near-dup clusters: {hygiene['near_dup_clusters']}; regrown in {hygiene['regrowth_days']}d: {len(hygiene['regrown'])}")
    # On its own line, right under the name-cluster line, because the two are
    # different measurements and the #499 trend is read from this one.
    print(f"  Exact-duplicate fact rows (facts_idx.text_hash): {_fact_duplicate_cell(fact_dups)}")
    print(f"  Files with duplicate fact IDs: {dup_id_files}")
    pv = hygiene.get("provenance") or {}
    if "both_pct" in pv:
        print(f"  Provenance coverage: {pv['both_pct']}% of {pv['facts']:,} facts")

    baseline = 0
    try:
        baseline_path = Path.home() / "lloyd" / "_pipeline" / "memory-graph" / "graph-baseline.json"
        baseline = int(json.loads(baseline_path.read_text())["active_edges"])
    except Exception:
        pass

    alarms = _alarms(store_stats, hygiene, dup_id_files, baseline)
    if alarms:
        print("\nALARM:", file=sys.stderr)
        for a in alarms:
            print(f"  - {a}", file=sys.stderr)
        _alert(alarms, output_file)
        if not args.no_alarm_exit:
            return EXIT_ALARM
    return EXIT_OK


def _alert(alarms: list[str], report_path: Path) -> None:
    """Post the alarm to Discord. Best effort — a failed notification must not
    change the exit code, which is the signal the scheduler reads."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        import asyncio
        from app.discord_notify import _discord_notify_task_complete
        body = ("Knowledge-graph health alarm:\n"
                + "\n".join(f"• {a}" for a in alarms)
                + f"\n\nReport: {report_path}")
        asyncio.run(_discord_notify_task_complete(60, "Knowledge Health Report", body))
    except Exception as exc:
        print(f"[alert] could not notify: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
