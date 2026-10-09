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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import PIPELINE_DIR, VAULT_FACTS_ROOT as FACTS_DIR, VAULT_KG_DB
from app.kg_store import (EDGE_TYPES, StoreUnavailable, canonical_edge_type,
                          store as _kg_store)
from scripts.reflection_archive import (copy_gaps as _copy_gaps,
                                        pattern_staleness as _pattern_staleness,
                                        reports_in as _reports_in)

DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "reflection"

# Rows rendered per long section. The report is read by a human and by the
# morning briefing; a 23,564-row table is neither. The count is always given
# in full, only the listing is capped.
SECTION_ROW_CAP = 50

GOD_ENTITY_THRESHOLD = 20
THIN_ENTITY_MAX_FACTS = 2
STALE_DAYS_THRESHOLD = 60
#: Share of ACTIVE records that have to carry one `created_at` calendar date before
#: that date stops being a recording date and is a re-derivation stamp (#2179).
#:
#: Measured live 2026-10-04 over the 130,115 active fact records: 99,707 of them
#: (76.6%) carry `created_at: 2026-09-23`, one day, and the whole field spans 11 days
#: (`2026-09-23 .. 2026-10-04`) — `clean_facts_directory()`
#: (`scripts/memory/next-gen-memory/nightly_extraction.py:185`) wipes the entity tree
#: and `fact_extractor.py:551` restamps every re-derived row with the run's date. A
#: date two thirds of the store shares says when the store was rebuilt, not when the
#: claim was recorded, so a fact whose ONLY date is that stamp has no age signal.
STALE_DOMINANT_DATE_SHARE = 0.25
#: How many records must share that date before the guard arms, beside the share.
#:
#: The share alone cannot carry it: every fixture in
#: `tests/test_knowledge_health_stale_facts.py` is 1-3 facts, where the single
#: `created_at` present is trivially 100% of the store, and a share-only guard would
#: file those fixtures' facts as undated — `test_old_valid_at_is_still_reported`
#: (one record, `created_at` 250 days old) among them. The live cohort is 99,707
#: records, so a floor two orders of magnitude below that arms on the real store and
#: stays silent on a hand-written corpus.
STALE_DOMINANT_DATE_FLOOR = 1_000
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
#: What makes a type a defect (#1820, #1931) is its name not being in
#: `app.kg_store.EDGE_TYPES`, imported here so the report and the writer cannot
#: disagree about what a real relation is — judged on `canonical_edge_type(t)`,
#: because this report's histogram counts the RAW row type and a pre-#1161
#: `related-to` is an approved type's old spelling, not a defect. Two writers can
#: persist a name that `_Edges.add` only spell-checks — `conversation_relations.py:1119`
#: (the Stage-2 classifier's proposed type) and `kg_rebuild.py:597` (a migration
#: payload) — because the vocabulary refusal lives in `_fact_relate` alone
#: (`agent_mcp/facts.py:1025`), and nothing else catches it. Since #1931 the count
#: gates nothing about that verdict (an off-vocabulary name FAILs at any count;
#: measured 2026-09-30, `ships` at exactly 5 uses printed PASS until then) — this
#: floor only chooses WHICH failing condition names it. A name that IS in the set
#: is a legitimate rare use, and is named on the line without failing it: the
#: 2026-09-30 store measured 38,551 active edges across 13 types with nothing
#: outside `EDGE_TYPES` (witness: vault `backlog/data/kg-health-2026-09-30T222931Z.json`),
#: so a count-based floor was firing on approved types alone.
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
    verdict line can call everything it rejects "artifact-shaped". #2000 adds
    one that is not a pattern: an all-lowercase name of several words
    (`is_lowercase_prose_fragment`).

    Only the questions are filtered by this. The `## Thin Entities` table is not:
    it is the surface on which a regrowth in `#NNN` minting becomes visible, so
    hiding those rows would destroy the very signal #743 needs.
    """
    return (any(pattern.search(name) for pattern in EXTRACTION_ARTIFACT_NAME_RES)
            or is_lowercase_prose_fragment(name))


def is_lowercase_prose_fragment(name: str) -> bool:
    """True for a name of two or more words with no capital letter in it (#2000).

    The six shapes above are all punctuation- or digit-shaped, so a noun phrase
    the extractor clipped out of a sentence passed them as a researchable
    concept: `cartpole camera presets task` (from an Isaac Lab PR note) headed
    the 2026-10-01 section twice, because `thin_entity_rank` puts the newest
    fact first and the newest thin entities are whatever last night's
    extraction mis-minted. The rule is structural, a shape and not a list of
    names: a named thing of several words is written with a capital somewhere
    (`Isaac Lab`, `NVIDIA Jetson`, `DiffusionGemma Technical Report`), a
    clipped phrase is not.

    Two things it deliberately does not do. A single lowercase token is left
    alone, since `certifi` and `gh-pages` are real and nothing in a name alone
    tells them from `unrelated`. And "the name is a substring of its own fact"
    is not used: measured on the same section it matched 11 of 20 entities,
    real ones included, and missed this rule's own witness. Known cost: a real
    subsystem written all in lowercase (`djev shadow-mode score logs`) loses
    its two questions and keeps its `## Thin Entities` row.
    """
    return len(name.split()) >= 2 and name == name.lower() and any(c.isalpha() for c in name)


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


def _created_at_day(fact: dict) -> str | None:
    """The `created_at` calendar date as `YYYY-MM-DD`, or None when there is none."""
    recorded = parse_date(fact.get("created_at"))
    return recorded.date().isoformat() if recorded else None


def record_age_days(fact: dict, now: datetime) -> "int | None":
    """How old the RECORD is: days since its own `created_at`, and nothing else. The
    other axis from `stale_age_reference` (#2475).

    The claim-age axis ages on the OLDEST usable date, so a fact written last week about a
    video published two years ago is stale the night it lands — which is how that section
    reached 24,234 with no record having aged in place. This axis asks what a reader of that
    number means when they hear "stale": has this row been sitting here unrevisited past the
    window. It is measured from `created_at` alone — no `event_date`, and no `valid_at`
    re-base, because either one would move the record's age with the claim's and there would
    be one axis again, which is the thing being fixed.

    `None` when `created_at` is absent or unparseable, NOT zero. Zero reads as "this record
    is new", a claim about the record that cannot be made; the absence is what
    `stale_age_split` counts as a third state and the section prints, for the same reason
    #841's fact with a missing `created_at` is never silently aged at today.
    """
    created = parse_date(fact.get("created_at"))
    if created is None:
        return None
    return (now - created).days


def _has_event_date(fact: dict) -> bool:
    """True when `event_date` gives the fact an age of its own.

    `parse_date` rather than `is not None`, so an `event_date` that will not parse
    (absent, `""`, the literal string `"null"`) counts as ABSENT: it carries no age
    signal either, and letting such a value disqualify a fact from the stamp cohort
    would leave that fact aging off the stamp and flipping on the cohort's day
    anyway — the blindness this guard exists to name.
    """
    return parse_date(fact.get("event_date")) is not None


def stale_stamp_cohort(entities: dict) -> dict | None:
    """The facts dated by nothing but a `created_at` the bulk of the store shares.

    Returns `{date, date_records, cohort, active_total, flip_date}`, or None when no
    `created_at` date clears both `STALE_DOMINANT_DATE_SHARE` and
    `STALE_DOMINANT_DATE_FLOOR` — which is every hand-written corpus, and was every
    corpus before a rebuild. `cohort` counts ACTIVE records with no `event_date` whose
    `created_at` falls on `date`, the dominant date: the most-carried one, earliest on
    a tie. `flip_date` is `date` + `STALE_DAYS_THRESHOLD`.

    This is the #841 survivor condition, and the opposite case. #841 fixed aging on a
    MISSING `created_at`; here the field is present on effectively every record and
    means the re-derivation date. Measured live 2026-10-04: the oldest `created_at` in
    the whole active store is 11 days old, so `min(created_at, event_date)` can only
    cross 60 days through `event_date` — of the 20,825 facts the section reports, 0
    have a null `event_date`, and 80,673 active facts have no age signal but the stamp.
    `stale_coverage` reported `(2, 130115)` over them, because the stamp parses as a
    perfectly usable date. Those 80,673 do not become reviewable one by one; they
    become stale together on 2026-11-22, and no nightly pass will ever watch them age,
    because the next full re-derivation resets the field to that run's date.

    A date the store shares is provenance loss, not freshness, so the cohort is
    reported as undated and given the day it would flip. Which facts are NOT in it:
    anything carrying a real `event_date`, which keeps aging from that date exactly as
    #841 established, and every fact on a date below either gate.
    """
    day_counts: dict[str, int] = {}
    undated_counts: dict[str, int] = {}
    active_total = 0
    for _entity_name, category_entries in entities.items():
        for entry in category_entries:
            for fact in entry["facts"]:
                if not is_fact_active(fact):
                    continue
                active_total += 1
                day = _created_at_day(fact)
                if day is None:
                    continue
                day_counts[day] = day_counts.get(day, 0) + 1
                if not _has_event_date(fact):
                    undated_counts[day] = undated_counts.get(day, 0) + 1
    if not day_counts or not active_total:
        return None
    # Most-carried date wins; a tie goes to the earlier date, since the older stamp is
    # the one whose cohort has been aging (unseen) longest.
    stamp = min(day_counts, key=lambda d: (-day_counts[d], d))
    carried = day_counts[stamp]
    if carried < STALE_DOMINANT_DATE_FLOOR:
        return None
    if carried / active_total <= STALE_DOMINANT_DATE_SHARE:
        return None
    return {
        "date": stamp,
        "date_records": carried,
        "cohort": undated_counts.get(stamp, 0),
        "active_total": active_total,
        "flip_date": (date.fromisoformat(stamp)
                      + timedelta(days=STALE_DAYS_THRESHOLD)).isoformat(),
    }


def _in_stamp_cohort(fact: dict, cohort: dict | None) -> bool:
    """Whether `fact` is one of `cohort`'s stamp-dated, event-date-less records."""
    return (cohort is not None and not _has_event_date(fact)
            and _created_at_day(fact) == cohort["date"])


def find_stale_facts(entities: dict, now: datetime, threshold_days: int) -> list[dict]:
    """Active facts whose oldest usable date is older than threshold_days.

    A fact carrying no usable date is NOT returned: it is not fresh, it is
    unmeasured. `stale_coverage` counts those so the report can name them
    instead of printing a clean verdict over an unseen share of the store.

    Neither is a fact in the stamp cohort (#2179): its only date is a `created_at`
    `STALE_DOMINANT_DATE_SHARE` of the store shares, which measures the rebuild, not
    the claim. Returning it would dress provenance loss up as a review backlog of
    80,673 rows appearing in one night; the cohort is reported on its own line with
    the day it flips, and counted unevaluable by `stale_coverage`.
    """
    cohort = stale_stamp_cohort(entities)
    stale = []
    for entity_name, category_entries in entities.items():
        for entry in category_entries:
            category = entry["category"]
            for fact in entry["facts"]:
                if not is_fact_active(fact):
                    continue
                if _in_stamp_cohort(fact, cohort):
                    continue

                reference = stale_age_reference(fact)
                if reference is None:
                    continue

                age = (now - reference).days
                if age >= threshold_days:
                    fact_text = str(fact.get("fact", ""))
                    preview = fact_text[:60] + ("..." if len(fact_text) > 60 else "")
                    # `age_days` is the CLAIM's age (oldest usable date, as the section
                    # header says); `record_age` is the RECORD's, from its own `created_at`
                    # alone, and None when there is no parseable one (#2475). Both travel on
                    # the row because the section's entire point is that they diverge, and a
                    # reader holding one stale row has to be able to say which of the two it
                    # is stale in — which is not derivable from `age_days` plus `created_at`
                    # without re-implementing `record_age_days`' no-`valid_at`-re-base rule a
                    # second time, somewhere else, where it can drift.
                    stale.append({
                        "entity": entity_name,
                        "category": category,
                        "preview": preview,
                        "age_days": age,
                        "record_age": record_age_days(fact, now),
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


#: The two axes, spelled once. The section header, the split line and this file's tests all
#: quote these strings, so the words the label uses and the words the counts use cannot drift
#: apart — which matters because the failure being fixed was a reader taking the header's
#: "oldest usable date" and the total beneath it as one instruction about the store (#2475).
STALE_CLAIM_AGE_LABEL = "claim age (content)"
STALE_RECORD_AGE_LABEL = "record age (the record's own `created_at`)"

#: The three states the split counts stale rows into. The renderer, the section's own sum
#: check and this file's tests read these names, so a fourth state cannot be added in one
#: place and silently missed in the other two.
STALE_ARRIVED_KEY, STALE_ROTTED_KEY, STALE_RECORD_UNSEEN_KEY = (
    "arrived_stale", "record_aged", "record_age_unseen")


def stale_age_split(stale_facts: list[dict]) -> dict:
    """{arrived already stale / the record itself is older / record age unseen} over the
    SAME rows `stale_age_bands` spreads, summing to the number of rows it is given.

    Why this exists and is not just a nicer label on the total (#2475): the headline aged
    21,341 → 24,234 in four days and read as a standing instruction to review 24k facts.
    Measured through the shipped selectors, every one of those 24,234 rows had a
    `created_at` inside the 60-day window and an `event_date` far outside it — the number was
    tracking how much feed content got digested overnight, and the rot component was exactly
    zero. Split, the two halves say different things, and only one of them is about the store
    going bad.

    `record_age_unseen` is the third state `record_age_days` returns None for: a row whose
    own record age cannot be established is neither new nor old, and folding it into either
    count would be an unsupported claim about it. It prints beside the other two for the same
    reason #2179 prints the stamp cohort above this section instead of folding it in.
    """
    arrived = rotted = unseen = 0
    for sf in stale_facts:
        record_age = sf.get("record_age")
        if record_age is None:
            unseen += 1
        elif record_age >= STALE_DAYS_THRESHOLD:
            rotted += 1
        else:
            arrived += 1
    return {STALE_ARRIVED_KEY: arrived, STALE_ROTTED_KEY: rotted,
            STALE_RECORD_UNSEEN_KEY: unseen}


def _stale_age_split_line(stale_facts: list[dict]) -> str:
    """`STALE_AGE_SPLIT: 24,234 arrived already stale, 0 the record itself is older, …`.

    `arrived + rotted + unseen == len(stale_facts)` is asserted, not assumed: the whole value
    of the line is that the halves sum to the total the section just printed (#2475 clause 2),
    and a split that quietly dropped or double-counted a row would be WORSE than the
    undifferentiated number it replaced, because it would look audited.

    The trailing clause is the sentence that keeps the total from being read as a backlog. A
    bare count rots into `stale: 24,234` in someone's summary within a week, which is how the
    number became a standing instruction to review 24k facts in the first place.
    """
    split = stale_age_split(stale_facts)
    total = sum(split.values())
    assert total == len(stale_facts), (
        f"the split counts {total} of {len(stale_facts)} stale rows, so the numbers under "
        "it cannot be read against the section total")
    return (
        f"STALE_AGE_SPLIT: {split[STALE_ARRIVED_KEY]:,} arrived already stale — counted by "
        f"{STALE_CLAIM_AGE_LABEL}, recorded inside the {STALE_DAYS_THRESHOLD}-day window —, "
        f"{split[STALE_ROTTED_KEY]:,} where {STALE_RECORD_AGE_LABEL} itself passes "
        f"{STALE_DAYS_THRESHOLD} days (the rot signal), "
        f"{split[STALE_RECORD_UNSEEN_KEY]:,} with no parseable `created_at` to date the "
        "record by. The first count tracks INGEST volume, not rot; the second is the one "
        "that says knowledge is going stale.")


def stale_coverage(entities: dict) -> tuple[int, int]:
    """Return (active facts with no usable date, active facts) — the blind share.

    The denominator the stale check cannot evaluate. A monitor that reports
    "No stale facts found" while 45% of the store carries no date is reporting a
    verdict on an input it could not read, which is the #841 defect class; this
    is the number that makes that state visible instead of clean. Counted over
    active facts, the same population `find_stale_facts` walks.

    Two populations land in `n`, and both are facts this check cannot age: the ones
    with no date at all, and the stamp cohort (`stale_stamp_cohort`, #2179) whose only
    date is a `created_at` the bulk of the store shares. The second is why the live
    store no longer returns `(2, 130115)`: it returns the 80,673 stamp-dated facts as
    well, because a date two thirds of the corpus shares measures the rebuild and tells
    nothing about the claim. `stale_stamp_cohort`'s own count is the difference, so the
    two are separable by a reader — and the section prints it, so they are separated
    before anyone has to subtract.
    """
    cohort = stale_stamp_cohort(entities)
    unevaluable = 0
    active_total = 0
    for _entity_name, category_entries in entities.items():
        for entry in category_entries:
            for fact in entry["facts"]:
                if not is_fact_active(fact):
                    continue
                active_total += 1
                if stale_age_reference(fact) is None or _in_stamp_cohort(fact, cohort):
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
            f"distinct texts: {d['distinct_texts']}){_same_source_clause(d)}")


def _same_source_clause(d: dict) -> str:
    """The second denominator of the fact-duplicate cell (#1942).

    The exact-twin number beside it reads 0 on a store where two thirds of the
    active rows re-assert something their own source document already said
    about that entity, because every one of those rows is worded differently.
    So the count is printed with the ACTIVE-row denominator it was measured on
    (never `rows`, which includes retired ones) and named for what it is: a
    population at risk. One document supports many distinct claims about one
    entity; the share that is one claim twice was measured on recent writes
    only (#1487, about a third), not on this row set.
    """
    if "same_source_redundant_rows" not in d or "active_rows" not in d:
        return ""
    n, active = d["same_source_redundant_rows"], d["active_rows"]
    pct = f"{100.0 * n / active:.1f}%" if active else "no active rows"
    return (f"; same-source: {n} of {active} active rows ({pct}) share an entity and "
            f"a source document with an earlier row, in "
            f"{d['same_source_paraphrase_groups']} groups: a population at risk, "
            f"not counted as duplicates")


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
    stamp_cohort: dict | None = None,
    pattern_staleness: list | None = None,
) -> str:
    """Generate the markdown health report.

    `duplicate_id_files` is `main()`'s `_duplicate_id_files` count. It and the
    hygiene dict's `provenance` go into the Hygiene table because the dated
    file is what the briefing and the sweep read; both used to be printed after
    the file was written, so neither reached any reader (#1289).

    `stale_unevaluable` is the `(n, m)` pair from `stale_coverage`: how many of
    the m active facts carry no date the stale check can age them from. Omitting
    it means the caller did not measure it, and the section says so rather than
    claiming the store is clean. Since #2179 its `n` holds the stamp cohort as well
    as the dateless facts, which is why the pair and `stamp_cohort` are passed
    together: the cohort's own count is printed beside them, so the two are never
    read as two separate losses.

    `stamp_cohort` is `stale_stamp_cohort`'s dict (#2179): the facts whose only date
    is a `created_at` more than `STALE_DOMINANT_DATE_SHARE` of the store shares. The
    section names them on their own line — stamp date, count, share of active records,
    and the day they all cross the threshold at once — because a store where two thirds
    of the corpus shares one `created_at` has lost those facts' age signal, and neither
    "fresh" nor a 80,673-row stale backlog is a true report of that. None means the
    caller did not measure it. No line prints when no date dominated the corpus, and
    none when a dominated date's records all carry an `event_date` of their own: an
    empty cohort has nothing to name.

    `copy_gaps` is `reflection_archive.copy_gaps` over the reflection directory:
    a report naming an archive copy the directory lacks is a lost cycle (#1227).
    None means not measured, and the section says that too.

    `pattern_staleness` is `reflection_archive.pattern_staleness` over the same
    directory (#2413): a §2e pattern file whose newest dated archive copy predates
    the newest `knowledge-write-*` report. It is the answer to the shape `copy_gaps`
    cannot see — a run that skipped the archive step names no copy in its report, so
    no pointer goes missing and the pointer check is silent while the file on disk is
    a cycle behind. Task 39's run of 2026-10-07 did exactly that and reported
    `status: success`. None means not measured.

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
    # The header IS the label on the headline number, so it names which of the two ages it
    # is and which one beside it is the rot signal (#2475 clause 3). Before the split this
    # line described the rule and nothing else, and the total under it — 24,234, +13.5% in
    # four days — was read as a standing instruction to review 24k facts when every row in
    # it had been recorded inside the window. Stating the semantics was not enough; the
    # number needs its axis named where a skimmer reads.
    lines.append(f"Active facts whose oldest usable date — `created_at` or `event_date`, "
                 f"re-based by `valid_at` — is older than {STALE_DAYS_THRESHOLD} days.")
    lines.append("")
    lines.append(f"**{STALE_CLAIM_AGE_LABEL}** (the total below) and **{STALE_RECORD_AGE_LABEL}** "
                 f"(the count beside it) are two different ages of the same rows: the first is "
                 "how old the fact's CONTENT is, so a fact written last week about a two-year-old "
                 "video lands stale and this total moves with feed volume; the second is how old "
                 "the RECORD is, aged from its own `created_at` alone, and it is the rot signal — "
                 "this is the only number here that says knowledge is going stale.")
    lines.append("")
    # Always printed, including at a zero total: a line that vanishes with the stale count
    # cannot distinguish "no stale facts" from "no split shipped", and 0 rot is exactly the
    # reading worth having when the total is large.
    lines.append(_stale_age_split_line(stale_facts))
    lines.append("")

    # The stamp cohort first, because it is the reason the two numbers under it are not
    # the whole store (#2179). Printed above the unevaluable line, which now counts these
    # facts too, so the reader sees which part of that n is a rebuild's date.
    if stamp_cohort and stamp_cohort["cohort"]:
        cohort_pct = (round(100.0 * stamp_cohort["cohort"] / stamp_cohort["active_total"], 1)
                      if stamp_cohort["active_total"] else 100.0)
        lines.append(
            f"STALE_STAMP_COHORT: {stamp_cohort['cohort']:,} facts ({cohort_pct}% of "
            f"{stamp_cohort['active_total']:,} active) carry no date but "
            f"`created_at: {stamp_cohort['date']}`, a calendar date "
            f"{stamp_cohort['date_records']:,} of them share — a re-derivation stamp, "
            "not a recording date. Undated, not fresh: counted in the unevaluable line "
            f"below, and every one crosses the {STALE_DAYS_THRESHOLD}-day threshold on "
            f"{stamp_cohort['flip_date']}.")
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
    elif (stale_unevaluable is not None and stale_unevaluable[0] == 0
          and not (stamp_cohort and stamp_cohort["cohort"])):
        # Reached only with no stale facts AND nothing left unevaluable: every
        # active fact was aged and none crossed the threshold. That is the whole
        # of what this verdict is allowed to claim. The cohort is tested here as
        # well as through `stale_unevaluable`, which already counts it (#2179), so a
        # caller that measured the cohort but passed a dateless-only coverage pair —
        # or no pair at all, which takes the branch above — still cannot call a store
        # whose dates are all one rebuild stamp clean.
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

    # The same retention question, asked of the directory rather than of the
    # report's own pointer (#2413). A run that skipped §2e's `cp` names no copy, so
    # the rows above are empty and the pattern file is still a cycle behind.
    lines.append("### §2e pattern files behind the newest report")
    lines.append("")
    if pattern_staleness is None:
        lines.append("*Not measured: the pattern files were not checked for a stale archive.*")
    elif not pattern_staleness:
        lines.append("Pattern files behind the newest report: 0 — every §2e archive is current.")
    else:
        lines.append(f"Pattern files behind the newest report: {len(pattern_staleness)}")
        lines.append("")
        lines.append("| Pattern file | Newest dated copy | Newest report |")
        lines.append("|--------------|-------------------|---------------|")
        for stale in pattern_staleness[:SECTION_ROW_CAP]:
            lines.append(f"| `{stale.stem}.md` | `{stale.newest_copy or 'none on disk'}` "
                         f"| `{stale.report}` |")
        if len(pattern_staleness) > SECTION_ROW_CAP:
            lines.append(f"| … | *{len(pattern_staleness) - SECTION_ROW_CAP:,} more* |")
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

    Conditions, each stated as a fact about the graph that can be checked:

    * VOCABULARY decides the verdict on types (#1820, #1931): a name outside
      `EDGE_TYPES` FAILs the line AT ANY COUNT, named with its count. The floor
      chooses only WHICH failing condition names it — under it, #1820's
      `types below the {EDGE_TYPE_MIN_USES}-use floor (…)` (wording byte-identical
      from #1544 on); at or above it, #1931's `types outside the edge-type
      vocabulary (…)`. Before #1931 the count was tested first, so a junk name an
      above-floor count of uses — the shape of a classifier that mints one
      repeatedly — printed PASS (measured 2026-09-30: `ships` at exactly 5 uses);
      the store's own refusal (`agent_mcp/facts.py:1025`) covers `fact_relate`
      only, and the writers that can persist such a name
      (`conversation_relations.py:1119`, `kg_rebuild.py:597`) do not consult it.
    * Membership is judged on `canonical_edge_type(t)`, the store's spelling fold
      matching the one `kg_health.py` applies to its snapshot: this report's
      histogram counts the RAW row type, so without the fold a pre-#1161
      `related-to` — an approved relation's old spelling — would FAIL as
      off-vocabulary, and the check would fire on spelling instead of on names.
    * A name inside `EDGE_TYPES` never contributes to the verdict — `fact_relate`
      refuses anything else, so a canonical singleton is by construction a
      legitimate first use — and every type under the floor, canonical or not, is
      named in the `types under {EDGE_TYPE_MIN_USES} uses:` column as ever.
    * The share condition is a TREND, not a level: FAIL when the leader exceeds
      `EDGE_TYPE_MAX_SHARE` or when it is higher than the previous night's
      day-aligned share. A constant FAIL teaches nothing; a level that is falling
      while the line says FAIL nightly trains the reader to skip the section.
      Neither #1820 nor #1931 changed this half.

    Measured 2026-09-30 over the witness committed as vault
    `backlog/data/kg-health-2026-09-30T222931Z.json`: all 38,551 active edges
    carry names inside `EDGE_TYPES` across 13 types, and the only under-floor
    types are the approved `conflicts_with (1)`, `derived_from (3)` and
    `describes (1)` — the pure vocabulary test moves no live line. `tests/
    test_edge_type_vocabulary.py` asserts the DECLARED vocabularies agree with
    `EDGE_TYPES`, and this line is where the STORED rows are checked against it.
    `EDGE_TYPE_MAX_SHARE` still catches a runaway catch-all at 0.8 on its own, a
    rise still FAILs, and the measured dominant type, share and active-edge total
    are printed on every PASS, so the bound quoted in the text is the one that can
    fail the line.
    """
    total = sum(type_dist.values())
    if not total:
        return "Edge-type cardinality: PASS — no active edges"

    # Canonical types are sorted with the rest and named on the line; only a name
    # outside the vocabulary may contribute to the verdict (#1820), and since #1931
    # at ANY count — the floor no longer gates the vocabulary condition, it only
    # chooses which failing condition names an off-vocabulary type. The fold is the
    # store's own (`canonical_edge_type`, as `kg_health.py` does before counting),
    # because this histogram holds raw row spellings: membership is a question about
    # the relation, not about the hyphens.
    under = sorted(t for t, n in type_dist.items() if n < EDGE_TYPE_MIN_USES)
    off_vocab = [t for t in type_dist if canonical_edge_type(t) not in EDGE_TYPES]
    below = sorted(t for t in off_vocab if type_dist[t] < EDGE_TYPE_MIN_USES)
    at_or_above = sorted(t for t in off_vocab if type_dist[t] >= EDGE_TYPE_MIN_USES)

    rare = ", ".join(f"{t} ({type_dist[t]})" for t in under) or "none"
    dominant, top = max(type_dist.items(), key=lambda kv: kv[1])
    share = top / total

    failing = []
    if below:
        failing.append(f"types below the {EDGE_TYPE_MIN_USES}-use floor "
                       f"({', '.join(f'{t} ({type_dist[t]})' for t in below)})")
    if at_or_above:
        failing.append("types outside the edge-type vocabulary "
                       f"({', '.join(f'{t} ({type_dist[t]})' for t in at_or_above)})")

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
    # Named again for the section's own line: `stale_unevaluable`'s n already carries
    # these facts (#2179), and printing the count that makes up the difference is what
    # keeps the two from being read as two separate losses. Like the two calls above,
    # this walks the parsed dict, not the tree.
    stamp_cohort = stale_stamp_cohort(entities)

    hygiene = compute_hygiene(entities, now)
    fact_dups = fact_duplicate_stats()
    dup_id_files = _duplicate_id_files(entities)
    # Over the entities already parsed above — no second walk of the tree.
    trace_coverage = contradiction_trace_coverage(entities)
    reflection_dir = DEFAULT_OUTPUT_DIR
    gaps = _copy_gaps(reflection_dir, _reports_in(reflection_dir))
    # The directory listing only — no second read of the report bodies, which the
    # line above already did for the pointer check (#2413).
    stale_patterns = _pattern_staleness(reflection_dir)

    # Generate report
    report = generate_report(entity_stats, rel_stats, edges, stale_facts, now, hygiene,
                             fact_dups=fact_dups, stale_unevaluable=stale_unevaluable,
                             duplicate_id_files=dup_id_files, copy_gaps=gaps,
                             trace_coverage=trace_coverage,
                             stamp_cohort=stamp_cohort,
                             pattern_staleness=stale_patterns)

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
    # The count above is CLAIM age and moves with ingest; the split under it is what says
    # whether any of it is rot (#2475). Printed on the console too, because a job's stdout is
    # what an operator pastes into a summary — and a number quoted from here with no axis on
    # it is exactly how 24,234 came to read as a review backlog.
    print(f"  Stale facts: {len(stale_facts)}")
    _split = stale_age_split(stale_facts)
    print(f"    of which: {_split[STALE_ARRIVED_KEY]:,} arrived already stale (ingest), "
          f"{_split[STALE_ROTTED_KEY]:,} the record itself is older (rot), "
          f"{_split[STALE_RECORD_UNSEEN_KEY]:,} no parseable created_at")
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
