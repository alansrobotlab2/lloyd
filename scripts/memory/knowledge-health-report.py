#!/usr/bin/env python3
"""
Knowledge Health Report Generator

Analyzes the fact store and relationship graph to produce a markdown
health report. Designed to be run by the nightly cron system.

Inputs:
  - Facts dir + relationships index resolved via app.paths.VAULT_FACTS_ROOT
    (currently ~/lloyd-data/_pipeline/vault-derived/facts/)

Output:
  - ~/lloyd-data/_pipeline/reflection/knowledge-health-YYYY-MM-DD.md
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import PIPELINE_DIR, VAULT_FACTS_ROOT as FACTS_DIR, VAULT_KG_DB
from app.kg_store import EDGE_TYPES, StoreUnavailable, store as _kg_store
from scripts.reflection_archive import copy_gaps as _copy_gaps, reports_in as _reports_in

DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "reflection"

# Rows rendered per long section. The report is read by a human and by the
# morning briefing; a 23,564-row table is neither. The count is always given
# in full, only the listing is capped.
SECTION_ROW_CAP = 50

GOD_ENTITY_THRESHOLD = 20
THIN_ENTITY_MAX_FACTS = 2
STALE_DAYS_THRESHOLD = 60
#: Past how many days a stale fact stops being a review candidate.
#:
#: `STALE_DAYS_THRESHOLD` decides *whether* a fact is stale; this decides whether a
#: stale fact is worth a reader's row. Measured live 2026-09-27 over the 15,948 active
#: stale facts the section reports: median 148 days, 88.1% within a year, 114 facts
#: (0.71%) older than ten years. Sorting by age descending therefore spent every one of
#: the 50 rows on that 0.71% — top row `Dual Process Theory lineage includes William
#: James's distinction`, 49,942 days, a historical fact with a real `event_date` that no
#: review would invalidate — while the 14,053-fact band just past the threshold, where
#: the backlog actually sits, was never listed at all. So the table lists the oldest
#: *reviewable* facts and the rest stay visible as a count (#1543).
REVIEWABLE_AGE_DAYS = 365
#: Upper edge in days of each age band, ascending; the last band is open above its own
#: edge. The band below the first edge is `STALE_DAYS_THRESHOLD` to that edge, since
#: nothing younger reaches this section. `REVIEWABLE_AGE_DAYS` must be one of these
#: edges or the printed distribution would not show the cut the table selects on —
#: pinned by `test_the_reviewable_band_is_one_of_the_printed_band_edges`.
AGE_BAND_BOUNDS_DAYS = (90, 180, 365, 730, 3650)
# Edge-type cardinality (#546): a type used fewer than this many times is a
# one-off, and one type holding this share of active edges is a catch-all
# absorbing relations that should have been typed.
#: What makes an under-floor type a defect (#1820) is its name not being in
#: `app.kg_store.EDGE_TYPES`, imported here so the report and the writer cannot
#: disagree about what a real relation is. Two writers can persist a name that
#: `_Edges.add` only spell-checks — `conversation_relations.py:1119` (the Stage-2
#: classifier's proposed type) and `kg_rebuild.py:597` (a migration payload) — because
#: the vocabulary refusal lives in `_fact_relate` alone (`agent_mcp/facts.py:1025`), and
#: nothing else catches it. A name that IS in the set is a legitimate rare use, and is
#: named on the line without failing it: the 2026-09-29 store measured 38,069 active
#: edges across 12 types with nothing outside `EDGE_TYPES`, so a count-based floor was
#: firing on approved types alone.
EDGE_TYPE_MIN_USES = 5
#: The ceiling on one type's share of active edges. It was 0.5 until #1658, and 0.5
#: is not a bound that anything can act on: `mentions` sat at 69-92% for every night
#: measured (09-23 92.2%, 09-24 83.9%, 09-25 84.6%, 09-26 76.3%, 09-27 69.4-69.8%)
#: while autonomy task #74 retyped ~2.5k of those edges a night, so the line said
#: FAIL nightly about a number that was falling — a constant verdict carries no
#: information, and the reader stops reading. What IS informative is the two things
#: this replaces it with: a level that is alarming on its own (0.8), and a share
#: that is RISING against the previous night's measurement. Under those rules 09-24
#: and 09-25 still FAIL (>0.8) and 09-26 and 09-27 PASS, so this is a looser bound
#: with a sharper trigger, not a switch to always-green.
EDGE_TYPE_MAX_SHARE = 0.8
#: What the floor rules on since #1820: the store's own type vocabulary, imported from
#: `app.kg_store` so the report and the writer cannot disagree about what a real
#: relation is. An under-floor type is a defect when its name is not in here —
#: `conversation_relations.py:1119` (the classifier's proposed type) and
#: `kg_rebuild.py:597` (a migration payload) can persist a name that
#: `_Edges.add` only spell-checks, because the vocabulary refusal lives in
#: `_fact_relate` alone (`agent_mcp/facts.py:1025`) — and a legitimate rare use when it
#: is. The 2026-09-29 store measured 38,069 active edges across 12 types with nothing
#: outside this set, which is why the floor was firing on approved types only.
#: How far back to look for the previous night's share. Seven because the report is
#: nightly and three nights of history is enough to catch a store that stopped being
#: written; beyond a week the "previous night" is not a night anymore.
EDGE_TYPE_SHARE_HISTORY_NIGHTS = 7
#: Where `kg_health.py` writes its snapshots. The share trend is read from there
#: rather than from the report's own previous run because the report overwrites its
#: dated file and keeps no share history, and because the snapshots run 3-5 times a
#: day — which is precisely why the comparison below is DAY-aligned: `mentions` grew
#: between CONSECUTIVE snapshots every hour of 2026-09-27 (+149, +152, +152, +84,
#: +68), so a snapshot-to-snapshot trend would FAIL on every single run.
EDGE_TYPE_METRICS_DIR = PIPELINE_DIR / "metrics"

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
#
# The two shapes #954 never covered, added by #1542. The extractor mints a test
# node id and a dotted config key as entities (`test_iv_loop_guards.pytest_…`,
# `inner_voice.todo_stewardship.enabled`), and a bare line number too (`255` was
# row 40 of the 2026-09-26 thin table); all three dominated the thin list. The
# dotted leg keys on an underscore beside the dot, not on the dot itself, because
# `jail.nix` — a nix flake, a real tool, row 27 of the same table — is dotted with
# no underscore and is a researchable thing. Whitespace stops that leg: a code
# path is one token, so a name with a space in it is prose and is left alone.
EXTRACTION_ARTIFACT_NAME_RES = (
    re.compile(r"^#\d"),                # '#441' — a backlog id, not an entity
    re.compile(r"^--"),                 # '--continue' — a CLI flag
    re.compile(r"\.md$"),               # '03-linear-representation-hypothesis.md' — a filename
    re.compile(r"^\d{4}-\d{2}-\d{2}"),  # '2026-09-11' — a date
    re.compile(r"[^.\s]*_[^.\s]*\.|\.[^.\s]*_"),  # 'guards._strip_cd_prefix', 'a.b_c.enabled'
    re.compile(r"^\d+$"),               # '255' — a bare number, not an entity
)


def is_extraction_artifact_name(name: str) -> bool:
    """True when an entity name is a shape the extractor minted, not a concept.

    The shapes are the four mints #954 named plus the two #1542 adds — an
    underscore-bearing dotted path and a bare number — which is why the section's
    verdict line can call everything it rejects "artifact-shaped".

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


def stale_age_bands(stale_facts: list[dict]) -> list[tuple[str, int, float]]:
    """[(band label, count, percent of the total)] over the AGE_BAND_BOUNDS_DAYS ladder.

    The shape a bare total hides (#1543): of the 15,948 stale facts measured live on
    2026-09-27, 88.1% were inside a year and 0.71% older than ten, so "the 50 oldest"
    sampled that tail and none of the backlog. The first band opens AT
    `STALE_DAYS_THRESHOLD`, not one day past it, because `find_stale_facts` selects with
    `age >= threshold_days` — an exactly-threshold-old fact is stale, and a floor of
    `THRESH + 1` dropped 78 of them from the live line while the total beside it still
    counted them. Pinned by
    `test_the_printed_bands_cover_the_total_without_gaps_or_overlap`.
    """
    total = len(stale_facts)
    ages = [sf["age_days"] for sf in stale_facts]
    out: list[tuple[str, int, float]] = []
    low = STALE_DAYS_THRESHOLD
    for high in AGE_BAND_BOUNDS_DAYS:
        count = sum(1 for a in ages if low <= a <= high)
        out.append((f"{low:,}-{high:,}", count,
                    round(count / total * 100, 1) if total else 0.0))
        low = high + 1
    # Closed band per bound, then one open band above the largest: the loop above
    # leaves no gap, so the shares always sum to 100% of the total.
    count = sum(1 for a in ages if a > AGE_BAND_BOUNDS_DAYS[-1])
    out.append((f">{AGE_BAND_BOUNDS_DAYS[-1]:,}", count,
                round(count / total * 100, 1) if total else 0.0))
    return out


def _stale_band_line(stale_facts: list[dict]) -> str:
    """The distribution as one markdown line: every band, its count and its share."""
    return "Age bands (days): " + " | ".join(
        f"{label}: {count:,} ({share}%)"
        for label, count, share in stale_age_bands(stale_facts))


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


#: The key a resolved contradiction leaves on the loser
#: (`agent_mcp.facts._CONTRADICTION_TRACE`), spelled exactly as the canonical edge
#: type it stands for (`app.kg_store.EDGE_TYPES`), so this count and the edge table
#: are counting one relation (#1596).
CONTRADICTION_TRACE_KEY = "conflicts_with"


def contradiction_trace_coverage(entities: dict) -> tuple[int, int]:
    """Return (fact records carrying a `conflicts_with` trace, all fact records).

    The trace shipped in #1544 with nothing counting it, so "is the number rising?"
    was answered by a hand-run grep over ~32k files and the instrument that would
    answer it nightly did not exist (#1596). This is that measurement.

    Over EVERY fact record, active or retired, and that is deliberate: a trace only
    ever exists on the loser of a pair, which the same write has just marked
    `invalid_at`. A counter filtered through `is_fact_active` would report 0 over a
    corpus made entirely of resolutions, which is the #841 class — a verdict printed
    for an input the check silently excluded. `is_fact_active` is not called here.

    Report-only by construction: `_alarms()` takes `(store_stats, hygiene,
    duplicate_id_files, baseline)` and none of those is this, so the line can rise
    from 0 to 10,000 without touching the exit code the scheduler reads. It measures
    whether resolutions are happening at all; it does not judge them.
    """
    traced = 0
    total = 0
    for _entity_name, category_entries in entities.items():
        for entry in category_entries:
            for fact in entry["facts"]:
                if not isinstance(fact, dict):
                    continue
                total += 1
                if fact.get(CONTRADICTION_TRACE_KEY):
                    traced += 1
    return traced, total


def _trace_coverage_cell(tc: tuple[int, int] | None) -> str:
    """The row's cell: the count beside its denominator, zero printed as zero.

    `0 of 118,343 fact records` is a measurement; a blank or an omitted row is what
    a zero would otherwise read as, and the whole point of counting a store-wide
    zero is that it is the state a first real resolution moves.
    """
    if tc is None:
        return "not measured"
    traced, total = tc
    return f"{traced:,} of {total:,} fact records"


def compute_hygiene(entities: dict, now: datetime, regrowth_days: int = 7,
                    baseline_path=None) -> dict:
    """Contamination, near-duplicate clusters and regrowth.

    Delegates to `kg_hygiene`, which is the measured definition of all three.
    This module had its own re-implementation of each — same intent, different
    code — so the report and `kg_health --json` could disagree about the same
    tree and there was no way to tell which was right. The shapes the report
    renders are kept, including `regrowth_line`: the one phrase kg_hygiene
    formats for every human surface, counts AND the baseline they are a delta
    against, so this row can no longer print `1 of 12027 new dirs` with the
    12,027 being the whole store (#1535).
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
    r = kg_hygiene.regrowth(root, regrowth_days, baseline_path=baseline_path)
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
        # None when no baseline exists, which is the point: the alternative was
        # the store's own size wearing a growth number (#1535).
        "new_dirs": r["new_dirs"],
        "regrowth_days": r["days"],
        "regrowth_line": kg_hygiene.describe(r),
        "baseline_at": r.get("baseline_at"),
        "dirs_at_baseline": r.get("dirs_at_baseline"),
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


def _provenance_cell(pv: dict | None) -> str:
    """The Provenance row. `both_pct` alone cannot tell "no date" from "no
    source", so the components ride with it; an unmeasured store says so rather
    than printing `None%` (the STALE_UNEVALUABLE convention, #841)."""
    if not pv or "both_pct" not in pv:
        why = (pv or {}).get("error")
        return "not measured" + (f" ({why})" if why else "")
    return (f"{pv['both_pct']}% of {pv['facts']:,} facts "
            f"(created_at {pv['created_at_pct']}%, source_doc {pv['source_doc_pct']}%)")


def generate_report(
    entity_stats: dict,
    rel_stats: dict,
    edges: list[dict],
    stale_facts: list[dict],
    now: datetime,
    hygiene: dict | None = None,
    fact_dups: dict | None = None,
    stale_unevaluable: tuple[int, int] | None = None,
    duplicate_id_files: int | None = None,
    copy_gaps: list | None = None,
    trace_coverage: tuple[int, int] | None = None,
) -> str:
    """Generate the markdown health report.

    `duplicate_id_files` is `main()`'s `_duplicate_id_files` count. It and the
    hygiene dict's `provenance` go into the Hygiene table because the dated
    file is what the briefing and the sweep read; both used to be printed after
    the file was written, so neither reached any reader (#1289).

    `stale_unevaluable` is the `(n, m)` pair from `stale_coverage`: how many of
    the m active facts carry no date the stale check can age them from. Omitting
    it means the caller did not measure it, and the section says so rather than
    claiming the store is clean.

    `copy_gaps` is `reflection_archive.copy_gaps` over the reflection directory:
    a report naming an archive copy the directory lacks is a lost cycle (#1227).
    None means not measured, and the section says that too.

    `trace_coverage` is the `(n, m)` pair from `contradiction_trace_coverage`: how
    many of the m fact records carry a `conflicts_with` resolution trace (#1596).
    It rides in the Hygiene table because it is a corpus measurement with a real
    zero, not a verdict: the row prints `0 of 118,343 fact records` rather than
    disappearing, since the zero is the finding. None means the caller did not
    measure it.
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
    # The one input the line still takes comes from data the report can already read:
    # the previous night's share, from the kg_health snapshot series, DAY-aligned. The
    # floor's other input since #1820 is `EDGE_TYPES`, imported from the store itself.
    lines.append(edge_type_cardinality(type_dist,
                                       previous=previous_dated_share(now)))
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
        within = [sf for sf in stale_sorted if sf["age_days"] <= REVIEWABLE_AGE_DAYS]
        over_band = len(stale_sorted) - len(within)
        # Total and median on one line: 15,948 alone reads as a mountain, and 15,948
        # with a median of 148 days says one dense band just past the threshold
        # (#1543). The `g` format prints a whole-day median as 148, not 148.0.
        lines.append(f"**{len(stale_sorted):,}** in total; median **"
                     f"{statistics.median([sf['age_days'] for sf in stale_sorted]):g} "
                     f"days** old.")
        lines.append(_stale_band_line(stale_sorted))
        if within:
            lines.append(f"the {min(SECTION_ROW_CAP, len(within))} oldest within "
                         f"{REVIEWABLE_AGE_DAYS:,} days — {len(within):,} of "
                         f"{len(stale_sorted):,} are reviewable; the {over_band:,} older "
                         f"than that are counted in the bands above, not listed:")
            listed = within[:SECTION_ROW_CAP]
        else:
            # An empty table one line after a five-figure count would read as "nothing
            # stale", so the oldest rows still print with the reason said out loud.
            lines.append(f"none of the {len(stale_sorted):,} is within the "
                         f"{REVIEWABLE_AGE_DAYS:,}-day reviewable band, so the "
                         f"{min(SECTION_ROW_CAP, len(stale_sorted))} oldest overall are "
                         f"listed below — every one is historical rather than a review "
                         f"candidate.")
            listed = stale_sorted[:SECTION_ROW_CAP]
        lines.append("")
        lines.append("| Entity | Category | Fact Preview | Age (days) |")
        lines.append("|--------|----------|-------------|-----------|")
        for sf in listed:
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
        # kg_hygiene's own phrase, reference and all. The row this replaces said
        # "born in the last 7 days | 51 of 12027 new dirs", where 12,027 was the
        # entire store and the window covered it because a rebuild had re-dated
        # every file (#1535). `.get` because a caller that hand-builds a hygiene
        # dict (the stale-facts tests do) must not turn a missing reference into
        # a bare number.
        lines.append(f"| Near-duplicate dirs coined since the baseline | {hygiene.get('regrowth_line') or 'not measured: no baseline was named'} |")
        lines.append(f"| Provenance coverage (created_at and source_doc) | {_provenance_cell(hygiene.get('provenance'))} |")
        dup_cell = ("not measured" if duplicate_id_files is None
                    else str(duplicate_id_files))
        lines.append(f"| Files with duplicate fact IDs | {dup_cell} |")
        # The count of resolved contradictions the corpus actually holds. Kept in
        # this table rather than Summary Stats because it belongs beside the other
        # two corpus-wide fact counts, and because Summary Stats is read as the
        # store's size while this is read as its activity (#1596). It is the number
        # `fact_resolve_apply`'s `traces_written` reports per call and nobody
        # totalled: 0 on the live corpus as of 2026-09-28, and a rising line is the
        # first evidence that the resolve path is being used at all.
        lines.append(f"| Fact records carrying a `conflicts_with` resolution trace | "
                     f"{_trace_coverage_cell(trace_coverage)} |")
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

    # --- Reflection retention ---
    # The one automated check that a nightly `-latest` overwrite kept its dated
    # copy; before it, a lost cycle surfaced only to a human's `grep -c` (#436).
    lines.append("## Reflection Retention")
    lines.append("")
    if copy_gaps is None:
        lines.append("*Not measured: the reflection directory was not checked for missing archive copies.*")
    elif not copy_gaps:
        lines.append("Reports naming an archive copy that is missing: 0 — no cycle is missing a copy.")
    else:
        lines.append(f"Reports naming an archive copy that is missing: {len(copy_gaps)}")
        lines.append("")
        lines.append("| Report | Missing copy |")
        lines.append("|--------|--------------|")
        for gap in copy_gaps[:SECTION_ROW_CAP]:
            lines.append(f"| `{gap.report.name}` | `{gap.copy}` |")
        if len(copy_gaps) > SECTION_ROW_CAP:
            lines.append(f"| … | *{len(copy_gaps) - SECTION_ROW_CAP:,} more* |")
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


def _parse_edge_ts(value) -> datetime | None:
    """A `created_at` string as a UTC datetime, or None when it cannot be stated.

    The store writes `2026-09-26T05:11:43Z`; the snapshots write
    `2026-09-28T11:33:56.877016+00:00`, and a snapshot name carries no offset at all.
    A value that parses to None is treated as UNKNOWABLE rather than as epoch-zero:
    an unparseable first-seen must not silently buy a grace, and must not silently
    lose one either — it is reported as no history, which is a different claim.
    """
    if not value:
        return None
    text = str(value).strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        out = datetime.fromisoformat(text)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out


def previous_dated_share(now: datetime | None = None,
                         metrics_dir: Path | None = None) -> dict | None:
    """The last night's dominant-type share, DAY-aligned, or None if none exists.

    `{"date": "2026-09-27", "share": 0.698}` for the most recent snapshot whose
    UTC date is strictly before `now`'s date, taking that date's LAST snapshot.

    Day-aligned is not a detail. `kg_health` runs 3-5 times a day, and `mentions`
    grew between CONSECUTIVE snapshots in every hour measured on 2026-09-27 (+149,
    +152, +152, +84, +68): a snapshot-to-snapshot trend would report "rising" on
    every single nightly run and FAIL nightly for the same reason the old 50% limit
    did. Comparing only across a date boundary asks the question the ruling is about
    — did the share come down overnight, against #74's ~2.5k-a-night retype.

    Returns None when no snapshot in `EDGE_TYPE_SHARE_HISTORY_NIGHTS` carries a
    readable date and `edges.by_type`, and the caller says so on the line rather
    than defaulting the missing comparison to "not rising".
    """
    moment = now or datetime.now(timezone.utc)
    root = metrics_dir or EDGE_TYPE_METRICS_DIR
    if not root.exists():
        return None
    today = moment.astimezone(timezone.utc).date()
    floor_date = today - timedelta(days=EDGE_TYPE_SHARE_HISTORY_NIGHTS)
    best: tuple | None = None
    for path in sorted(root.glob("kg-health-*.json")):
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stamp = _parse_edge_ts(snap.get("captured_at")) or _parse_edge_ts(
            path.stem.replace("kg-health-", "").replace("T", "T"))
        if stamp is None:
            continue
        stamp = stamp.astimezone(timezone.utc)
        by_type = ((snap.get("edges") or {}).get("by_type") or {})
        total = sum(int(v) for v in by_type.values())
        if not total:
            continue
        day = stamp.date()
        if day >= today or day < floor_date:
            continue
        share = max(int(v) for v in by_type.values()) / total
        key = (day, stamp)
        if best is None or key > best[0]:
            best = (key, {"date": day.isoformat(), "share": share})
    return best[1] if best else None


def edge_type_cardinality(type_dist: dict[str, int], *,
                          previous: dict | None = None) -> str:
    """The `Edge-type cardinality: PASS|FAIL` verdict line (#546, #1658, #1820).

    Report output only, never an alarm: the store as measured holds a dominant
    type and one-off types on most nights, and alarming on that pages about the
    graph's shape rather than about the store being broken. The two conditions are
    attributed separately on the FAIL line so a reader sees which one drifted.

    Two conditions, each stated as a fact about the graph that can be checked:

    * The 5-use floor fails on VOCABULARY (#1820). Under it, a type outside
      `EDGE_TYPES` FAILs the line at any count and any age, named with its count; a
      type inside `EDGE_TYPES` is named on the line and never contributes to the
      verdict. Age decided this split until today and could not: `fact_relate` refuses
      a type outside the vocabulary, so a canonical singleton is by construction a
      legitimate first use, while the off-vocabulary name a Stage-2 classifier or a
      rebuild payload can persist is the defect the floor exists to catch. Measured on
      2026-09-29: all 38,069 active edges carry canonical types, so the only two types
      the floor was failing — `conflicts_with (1)`, `describes (1)` — were approved
      types, and the grace that exempted them expired on 2026-10-10, at which point the
      line fails again on those same two singletons.
    * The share condition is a TREND, not a level: FAIL when the leader exceeds
      `EDGE_TYPE_MAX_SHARE` or when it is higher than the previous night's
      day-aligned share. A constant FAIL teaches nothing; a level that is falling
      while the line says FAIL nightly trains the reader to skip the section.
      #1820 changed none of this half.

    The floor is a floor and not a vocabulary check: an off-vocabulary type AT or above
    5 uses is not named by this line, and neither is a canonical one. `tests/
    test_edge_type_vocabulary.py` asserts the DECLARED vocabularies agree with
    `EDGE_TYPES`; nothing asserts the stored rows do, and that gap is owed a ruling
    (#1820). `EDGE_TYPE_MAX_SHARE` still catches a runaway catch-all at 0.8 on its own,
    a rise still FAILs, and the measured dominant type, share and active-edge total are
    printed on every PASS, so the bound quoted in the text is the one that can fail the
    line.
    """
    total = sum(type_dist.values())
    if not total:
        return "Edge-type cardinality: PASS — no active edges"

    # Canonical types are sorted with the rest and named on the line; only a name
    # outside the vocabulary may contribute to the verdict.
    under = sorted(t for t, n in type_dist.items() if n < EDGE_TYPE_MIN_USES)
    below = [t for t in under if t not in EDGE_TYPES]

    rare = ", ".join(f"{t} ({type_dist[t]})" for t in under) or "none"
    dominant, top = max(type_dist.items(), key=lambda kv: kv[1])
    share = top / total

    failing = []
    if below:
        failing.append(f"types below the {EDGE_TYPE_MIN_USES}-use floor "
                       f"({', '.join(f'{t} ({type_dist[t]})' for t in below)})")

    # The trend bound needs `previous`; without it only the level can fail, and the
    # line says the comparison was not available rather than implying it passed.
    if share > EDGE_TYPE_MAX_SHARE:
        failing.append(f"dominant type share ({dominant} {share:.1%} > "
                       f"{EDGE_TYPE_MAX_SHARE:.0%})")
    elif previous and share > previous["share"]:
        failing.append(f"dominant type share ({dominant} {share:.1%} > previous "
                       f"night {previous['date']} {previous['share']:.1%})")

    verdict = "FAIL" if failing else "PASS"
    limit = f"ceiling {EDGE_TYPE_MAX_SHARE:.0%}"
    if previous:
        limit += f", previous night {previous['date']} {previous['share']:.1%}"
    else:
        limit += ", no previous night's share available"
    return (f"Edge-type cardinality: {verdict} — types under {EDGE_TYPE_MIN_USES} "
            f"uses: {rare}; dominant type {dominant} is {share:.1%} of "
            f"{total:,} active edges ({limit}) — conditions failing: "
            f"{', '.join(failing) or 'none'}\n")


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
    # Over the entities already parsed above — no second walk of the tree.
    trace_coverage = contradiction_trace_coverage(entities)
    reflection_dir = DEFAULT_OUTPUT_DIR
    gaps = _copy_gaps(reflection_dir, _reports_in(reflection_dir))

    # Generate report
    report = generate_report(entity_stats, rel_stats, edges, stale_facts, now, hygiene,
                             fact_dups=fact_dups, stale_unevaluable=stale_unevaluable,
                             duplicate_id_files=dup_id_files, copy_gaps=gaps,
                             trace_coverage=trace_coverage)

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
    # Not "regrown in 7d": with #1535 the count is a delta against the stored
    # entity-dir baseline, so the stdout line prints kg_hygiene's own phrase,
    # which carries the date that delta runs from.
    print(f"  Near-dup clusters: {hygiene['near_dup_clusters']}; regrowth: "
          f"{hygiene.get('regrowth_line') or 'not measured: no baseline was named'}")
    # On its own line, right under the name-cluster line, because the two are
    # different measurements and the #499 trend is read from this one.
    print(f"  Exact-duplicate fact rows (facts_idx.text_hash): {_fact_duplicate_cell(fact_dups)}")
    print(f"  Files with duplicate fact IDs: {dup_id_files}")
    # The same measurement twice, deliberately: this stdout line and the report's
    # section are the two places a human reads the verdict, and they must not be
    # computed from different inputs — the report's copy takes the previous night's
    # trend, so a bare call here would print a second, laxer verdict on the run.
    stdout_cardinality = edge_type_cardinality(
        rel_stats["type_distribution"],
        previous=previous_dated_share(now))
    print(f"  {stdout_cardinality}")
    print(f"  Reflection copies named but missing: {len(gaps)}")
    pv = hygiene.get("provenance") or {}
    if "both_pct" in pv:
        print(f"  Provenance coverage: {pv['both_pct']}% of {pv['facts']:,} facts")

    baseline = 0
    try:
        baseline_path = PIPELINE_DIR / "memory-graph" / "graph-baseline.json"
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
