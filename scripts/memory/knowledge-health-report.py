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

        stats[entity_name] = {
            "total_facts": len(all_facts),
            "active_facts": len(active),
            "expired_facts": len(expired),
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


def find_stale_facts(entities: dict, now: datetime, threshold_days: int) -> list[dict]:
    """Find facts older than threshold_days with no valid_at update."""
    stale = []
    for entity_name, category_entries in entities.items():
        for entry in category_entries:
            category = entry["category"]
            for fact in entry["facts"]:
                if not is_fact_active(fact):
                    continue

                created = parse_date(fact.get("created_at"))
                if not created:
                    continue

                valid_at = fact.get("valid_at")
                if valid_at:
                    continue

                age = (now - created).days
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


def generate_report(
    entity_stats: dict,
    rel_stats: dict,
    edges: list[dict],
    stale_facts: list[dict],
    now: datetime,
    hygiene: dict | None = None,
) -> str:
    """Generate the markdown health report."""
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
    thin_entities.sort(key=lambda x: x[1]["active_facts"])

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
    lines.append(f"Active facts with `created_at` older than {STALE_DAYS_THRESHOLD} days and no `valid_at` update.")
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
    else:
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

    if thin_entities:
        for name, s in thin_entities[:20]:  # questions, not a listing — 20 is plenty
            lines.append(f"- What does **{name}** relate to?")
            lines.append(f"- Is **{name}** still relevant?")
    else:
        lines.append("*No thin entities to generate questions for.*")
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

    hygiene = compute_hygiene(entities, now)
    dup_id_files = _duplicate_id_files(entities)

    # Generate report
    report = generate_report(entity_stats, rel_stats, edges, stale_facts, now, hygiene)

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
    print(f"  Contaminated dirs: {hygiene['contaminated_dirs']} ({hygiene['foreign_facts']} foreign facts)")
    print(f"  Near-dup clusters: {hygiene['near_dup_clusters']}; regrown in {hygiene['regrowth_days']}d: {len(hygiene['regrown'])}")
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
