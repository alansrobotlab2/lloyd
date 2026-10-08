#!/usr/bin/env python3
"""Key skill-catalog overlap on resolved KG entities, not on description tokens (#2440).

The overlap number anyone acts on today is improvised per run. `skill_lint.find_duplicates`
scores `difflib` ratios over `(name, description)` and its own comment records why that is
weak; the nightly screen #83 Stage 5 runs is prose in a runbook with no scorer and no
threshold committed, and it has produced three different readings of one corpus in two
nights: the 2026-10-07 pass scored `ml-paper-writing` vs `research-paper-writing` at 0.03
and called the pair a name look-alike, the 2026-10-08 pass scored the same pair 0.327-0.358
and filed it as #2409, which is 2,607 lines of two skills for one procedure. #2409's own
note says it plainly: "The two numbers are not comparable and one of the two readings is
wrong." A score that moves when the wording moves is not an instrument.

This module commits the instrument instead of the prose. Two scores are computed and
printed for every pair, side by side, always:

  * `desc_jaccard` — the incumbent. Word-set Jaccard over the two front-matter
    descriptions, the quantity #83 Stage 5 thresholded at 0.34 and #2409 scored. Named so
    the comparison is falsifiable rather than "the old way"; it is reported, never gated.
  * `entity_jaccard` — the new key. Each SKILL.md body is projected through the knowledge
    graph onto a set of CANONICAL entity names (surface forms resolved through
    `app.kg_store`'s alias table and entity registry), and the pair is scored by Jaccard
    over those sets. Body, not description — which is what makes the second stability
    property real rather than tautological: re-word a description and the incumbent moves
    while the entity key cannot.

Three rules make the entity key mean "about the same thing" rather than "cites the same
manual pages", and they are the whole of the mechanism:

  1. Resolution goes through the alias map, so `Neo4j`, `neo4j` and a registered surface
     like `Agent Craft` reach one canonical name. This is the talk's hinge — without a
     canonical representation of a thing, every judgement downstream inherits the
     ambiguity — and the alias table is the data Lloyd already maintains for it.
  2. Harness-tool and filesystem-path surface forms are SUPPRESSED. `Bash`, `Read`, `Write`
     and friends are registered entities in this graph, and a skill that says "call `Read`
     first" is not about the Read tool any more than a Python manual is about its semicolons.
     Unsuppressed, the shared boilerplate of 195 runbooks would be the largest signal in the
     corpus — triage measured exactly that, a naive longest-match pass putting the one
     adjudicated real pair 8th of 12 at 0.405, inside the 0.32-0.53 band of the `github-*`
     look-alikes it exists to separate.
  3. A pair with nothing to compare is `unknown`, never `not duplicates`. Coverage here is
     genuinely thin and says so: `source_doc LIKE 'skills/%'` contributes 1 fact row and 0
     active edges, so no SKILL.md has `mentions` evidence and this projection is a fresh
     deterministic name/alias pass over skill text rather than a read of stored edges. The
     zero-coverage count is printed first, before any pair, because it bounds every number
     below it.

Deliberately absent, each for a stated reason:

  * No embedding. #557 measured the geometric arm below lexical (recall@5 0.6863 against
    0.7451) at the shipped floor and Alan ruled on 2026-09-13 to keep the asset and drop the
    arm. This is a different instrument, not that one revived: deterministic and alias
    resolved, with no similarity space to recalibrate.
  * No threshold on the entity score that gates anything. `ENTITY_OVERLAP_FLAG` only orders
    the printout; acceptance for this detector is a ranking against a labelled fixture,
    because entity overlap is not procedural identity — the six `github-*` skills legitimately
    share entities and differ in job, which is why they are the fixture's negatives.
  * No write anywhere but the one report file. No merge, no quarantine, no `status:` change:
    #83 and #58 own the merge decision, exactly as #2409's route did.
  * No community/coherence column. It is owed-item 2, and it needs a hand-audit of the top
    flagged skills against the four known-intentional sibling clusters before a number from
    it means anything.

Usage::

    python3 scripts/skill_entity_overlap.py report                       # live corpus
    python3 scripts/skill_entity_overlap.py report --skills-root DIR \\
        --store kg.sqlite --pairs eval/skill_entity_overlap_pairs.json \\
        --out /tmp/report.json

Exit is 0 whenever a report was written, including over a fixture whose positives rank last:
this is a measurement, and a measurement that exits non-zero because it did not flatter its
author is a measurement that gets switched off.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import kg_store                                        # noqa: E402
from app.paths import PIPELINE_DIR                              # noqa: E402
from eval import dup_detect                                     # noqa: E402

SCHEMA = 1

#: The default home of the ledger, beside the other nightly skill artifacts under
#: `_pipeline/skills/`. Overridable with `--out`; nothing else is written.
REPORT_RELPATH = "skills/entity-overlap.json"

#: The incumbent, named. `dup_detect.jaccard` is the house Jaccard (0.0 when either side is
#: empty, so a blank description can never read as a 1.0); the token set is the lower-cased
#: word set of the front-matter `description`, which is the quantity #83 Stage 5 thresholded
#: at 0.34 and #2409 quoted as `desc=0.327`.
INCUMBENT_NAME = "desc_jaccard"
INCUMBENT_THRESHOLD = 0.34

#: Sorts the printed pair list only. Not a decision: nothing in this module blocks,
#: quarantines or merges, and no exit status depends on it.
ENTITY_OVERLAP_FLAG = 0.25

#: Below this many canonical entities a pair is `unknown`. Two skills resolving to three
#: entities each are not comparable, and reporting a quotient of two 3-element sets as a
#: measurement is how a thin corpus gets believed. Named rather than implied because the
#: verdict string says `thin`, so a reader can ask what the floor was.
MIN_ENTITIES_TO_SCORE = 5

#: How many unscored pairs the report enumerates. The count above is the figure anyone quotes;
#: the sample is ordered by combined set size so the pairs nearest to becoming measurable — the
#: ones a re-run could score — are the ones on show.
UNKNOWN_PAIR_SAMPLE = 20

#: Longest-match bounds. Five words is the longest registered entity surface in the store
#: worth matching in prose; below four characters a "surface" is a substring of an ordinary
#: word and the pass starts resolving fragments.
MAX_SURFACE_WORDS = 5
MIN_SURFACE_CHARS = 4

#: Entity surfaces that name the harness rather than the skill's subject. The test in
#: `tests/test_skill_entity_overlap.py` walks every `name=` in `agent_mcp/builtin_*.py` and
#: fails if one of them is missing here, so this list cannot quietly fall behind a new tool —
#: which is the failure mode that would make it useless: suppression that stops covering the
#: tools is suppression that silently measures boilerplate again. The names are also listed
#: in their canonical-entity spelling because that is how the graph registers them: the live
#: store carries `Bash`, `Read`, `Write`, `Edit`, `Grep` and `Glob` as entities, so an
#: unsuppressed pass credits every runbook that says "call `Read` first" with an opinion
#: about Read.
HARNESS_TOOL_SURFACES = frozenset({
    "Read", "Write", "Edit", "Glob", "Grep", "Bash", "Task", "TodoWrite", "Scratchpad",
    "SetGoal", "ClearGoal", "EnterPlanMode", "ExitPlanMode", "Skill", "SlashCommand",
    "AskUserQuestion", "WebFetch", "WebSearch", "NotebookEdit", "KillShell", "Agent",
    # Served like every other tool and named like one, so it belongs here even with its
    # leading underscore: the check that keeps this list honest scans the tool modules that
    # define them and does not guess about case, and this name came out of that scan.
    "_BackgroundTaskDrain",
    "browser_navigate", "browser_snapshot", "browser_click", "browser_fill",
    "browser_press", "browser_tabs", "browser_screenshot", "browser_evaluate",
    "browser_wait", "browser_scroll", "browser_select", "browser_drag", "browser_cookies",
    "fact_get", "fact_add", "fact_relate", "fact_relationships", "fact_resolve",
    "vault_read", "vault_write", "vault_search", "vault_recall", "vault_overview",
    "memory_read", "memory_add", "memory_replace", "memory_remove", "session_recall",
    "graph_explain", "graph_affected", "graph_path", "graph_hubs", "graph_status",
    "graph_refresh", "djev_decide", "djev_rank", "djev_status", "skills_search",
    "skills_read", "research_propose", "research_next", "research_complete",
    "research_list", "research_stats", "backlog_tasks", "backlog_get_task",
    "backlog_write_task", "backlog_boards", "autonomy_tasks", "autonomy_get_task",
    "autonomy_write_task", "autonomy_run_task", "automod_start", "automod_gate",
    "automod_land", "automod_status", "http_request", "http_fetch", "http_search",
    "email_send", "email_search", "email_read", "calendar_events", "calendar_create",
    "tasks_create", "tasks_list", "contacts_search", "contacts_get", "desktop_capture",
    "desktop_act", "session_inject_context", "ambient_decide",
})

#: Characters that make a match a filesystem path rather than a name. A surface with one of
#: these on either side is a path segment — `~/obsidian/skills/Bash/SKILL.md`,
#: `skills/bash-runbook/` — and the projection credits the SUBJECT of the skill, not the
#: directory it happens to be filed in.
_PATH_EDGE = "/\\~"

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'+._-]*")
_DESC_TOKEN = re.compile(r"[a-z0-9][a-z0-9'+._-]*")
_FM = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _norm(text: str) -> str:
    """Fold a surface to its lookup key: lower-case, inner whitespace collapsed."""
    return " ".join((text or "").lower().split())


def surface_map(store) -> dict[str, str]:
    """`surface_lc -> canonical`, over every registered entity and every alias row.

    Aliases win an exact-key tie: an alias row exists because someone recorded that a
    surface form MEANS another entity, which is a stronger claim than a registered name
    that happens to be spelled the same way.
    """
    out: dict[str, str] = {}
    for name in store.entities.all():
        key = _norm(name)
        if key:
            out.setdefault(key, name)
    for surface, canonical in store.aliases.all_lower().items():
        key = _norm(surface)
        if key and canonical:
            out[key] = canonical
    return out


def is_tool_surface(surface: str) -> bool:
    """Whether a surface form names a harness tool rather than a subject."""
    return _norm(surface) in {_norm(t) for t in HARNESS_TOOL_SURFACES}


def _path_adjacent(text: str, start: int, end: int) -> bool:
    """Whether the span at [start:end] sits inside a filesystem path."""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return before in _PATH_EDGE or after in _PATH_EDGE


def alias_surfaces(surfaces: dict[str, str]) -> frozenset[str]:
    """The lookup keys that came from an ALIAS row rather than a bare entity name.

    A registered alias is a spelling someone recorded as a NAME — `vllm` routed to `vLLM` —
    which is why the lowercase-prose screen below exempts it while still refusing the
    ordinary English word that happens to be registered as an entity.
    """
    return frozenset(k for k, v in surfaces.items() if k != _norm(v))


def project(text: str, surfaces: dict[str, str], *,
            alias_keys: frozenset[str] = frozenset(),
            suppress_tools: bool = True,
            suppress_lowercase: bool = True) -> dict:
    """Project one skill text onto its resolved canonical entity set.

    Longest match wins, left to right, and a span rejected by a screen advances one token so
    a shorter surface inside it can still be seen. Counters are returned rather than
    discarded: the suppressed counts are how a reader tells a skill with real coverage from
    one whose only matches were boilerplate, which is the difference between a thin corpus
    and an empty one.
    """
    tokens = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text or "")]
    entities: set[str] = set()
    hits = {"resolved": 0, "tool": 0, "path": 0, "short": 0, "lowercase": 0}
    # `suppress_lowercase` is a measurement knob, not a feature: the number of prose matches
    # the screen drops is the difference between "the catalogue resolves to 12 entities" and
    # "it resolves to 12 entities plus every ordinary noun in it", and a reader has to be able
    # to ask which one they are looking at.

    i = 0
    n = len(tokens)
    while i < n:
        matched = 0
        for length in range(min(MAX_SURFACE_WORDS, n - i), 0, -1):
            span_tokens = tokens[i:i + length]
            surface = " ".join(t[0] for t in span_tokens)
            key = _norm(surface)
            if key not in surfaces:
                continue
            start, end = span_tokens[0][1], span_tokens[-1][2]
            if len(key) < MIN_SURFACE_CHARS:
                hits["short"] += 1
                break
            if _path_adjacent(text, start, end):
                hits["path"] += 1
                break
            if suppress_tools and is_tool_surface(surface):
                hits["tool"] += 1
                break
            # A single word written lower-case in running prose is an activity, not the
            # proper-noun-shaped thing the registry names: "the session restarted" is not a
            # claim about an entity called `session`. Multi-word surfaces and registered
            # alias forms are excluded because those spellings are how a name arrives.
            if (suppress_lowercase and length == 1 and surface[:1].islower()
                    and key not in alias_keys):
                hits["lowercase"] += 1
                break
            entities.add(surfaces[key])
            hits["resolved"] += 1
            matched = length
            break
        i += max(matched, 1)
    return {"entities": entities, "suppressed": dict(hits)}


def strip_front_matter(text: str) -> str:
    m = _FM.match(text or "")
    return text[m.end():] if m else (text or "")


def description_of(text: str) -> str:
    """The front-matter `description` value, or "" — read without a YAML dependency."""
    m = _FM.match(text or "")
    if not m:
        return ""
    desc: list[str] = []
    inside = False
    for line in m.group(0).splitlines()[1:]:
        if not inside:
            if line.lower().startswith("description:"):
                inside = True
                rest = line.split(":", 1)[1].strip()
                if rest and rest not in (">", ">", "|", "|-", ">-"):
                    desc.append(rest.strip("'\""))
            continue
        if line[:1] in (" ", "\t"):
            desc.append(line.strip())
        else:
            break
    return " ".join(desc).strip("'\"")


def desc_jaccard(a: str, b: str) -> float:
    """The incumbent's score: Jaccard over the lower-cased word sets of two descriptions.

    Word SET, in the order the nightly screen that produced the 0.34 figure and the
    `desc=0.327` quoted in #2409 used it: set-membership over the tokens, no shingles and no
    word order, which is exactly why it moves when someone re-orders a sentence.
    """
    return dup_detect.jaccard(set(_DESC_TOKEN.findall((a or "").lower())),
                              set(_DESC_TOKEN.findall((b or "").lower())))


def corpus(roots: Sequence[Path] | None = None) -> list[dict]:
    """Every advertised skill, with its body, description and canonical entity set.

    Through `agent_mcp.skills.iter_active_skills`, which is the one definition of
    "advertised" every surface shares (#1294): a dot-prefixed directory is the archive, a
    quarantined `status:` is retired in place, a name in two roots belongs to the first. A
    caller passing `roots` gets exactly that walker over its own tree, which is what makes
    this measurable without touching the vault.
    """
    from agent_mcp.skills import iter_active_skills
    out = []
    for skill in iter_active_skills(roots=list(roots) if roots else None):
        try:
            text = skill.skill_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.append({"name": skill.name,
                    "description": str((skill.frontmatter or {}).get("description")
                                       or description_of(text)),
                    "body": strip_front_matter(text)})
    return out


def size_histogram(rows: Iterable[dict]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for row in rows:
        key = str(min(row["n_entities"], 50))
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def verdict_row(name_a: str, name_b: str, ents_a: set[str], ents_b: set[str],
                desc_a: str, desc_b: str) -> dict:
    """One pair scored both ways, with the `unknown` rail decided BEFORE any quotient.

    `unknown` is a verdict, not a zero, and it is decided before the score is read at all:
    a pair whose either side resolved nothing has no denominator, and printing `not
    duplicates` for it would let a coverage hole read as a clean catalogue — which is the
    failure #2440 exists to stop, since a skill that resolves nothing is the *common* case
    on this corpus rather than the rare one. `unknown_reason` names which hole it is
    (`no_coverage`, or `thin` below the stated floor), and both say the same thing about
    every pair they cover: this run has no reading, not a negative one.

    `not_duplicates` is reserved for the one case that is genuinely measured and genuinely
    empty: both sides resolved, both are over the floor, and the two sets share no entity at
    all. It is not a tuned cut-off — the item rules acceptance as a ranking, not a threshold,
    and nothing in this run gates on it — and the rail that matters is that an unscored pair
    can never reach it.
    """
    if not ents_a or not ents_b:
        reason = "no_coverage"
    elif len(ents_a) < MIN_ENTITIES_TO_SCORE or len(ents_b) < MIN_ENTITIES_TO_SCORE:
        reason = "thin"
    else:
        reason = ""
    score = dup_detect.jaccard(ents_a, ents_b)
    verdict = ("unknown" if reason else
               "not_duplicates" if score == 0.0 else "overlap")
    return {"a": name_a, "b": name_b,
            "n_a": len(ents_a), "n_b": len(ents_b),
            "entity_score": round(score, 4),
            "incumbent_name": INCUMBENT_NAME,
            "incumbent_score": round(desc_jaccard(desc_a, desc_b), 4),
            "verdict": verdict,
            "unknown_reason": reason}


def fixture_rows(path: Path) -> list[dict]:
    """Score the committed fixture from the bytes committed with it.

    The row's two entity sets and two descriptions are *in* the fixture, so the numbers this
    prints are the ones the adjudication was made against on 2026-10-08 and nothing about
    them depends on which store, which checkout or which day the run happens in. That is a
    deliberate split: the labels are hand-adjudicated evidence about a dated corpus, and an
    eval that quietly re-derived its own inputs would be measuring whatever the graph
    happens to hold tonight while quoting a 2026-10-08 decision.

    What is re-derived at run time is `fixture_drift`'s business, printed beside the table and
    never folded into a score.

    A row missing its recorded set is `unknown` with reason `missing_entity_set` and no score
    at all — never a zero — because a pair this file could not read is not evidence that two
    skills differ, and reporting it as one is the failure #2440 is about.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for pair in data["pairs"]:
        name_a, name_b = pair["a"], pair["b"]
        sets = pair.get("entity_sets") or {}
        a_list, b_list = sets.get("a"), sets.get("b")
        desc_a = (pair.get("descriptions") or {}).get("a") or ""
        desc_b = (pair.get("descriptions") or {}).get("b") or ""
        row = verdict_row(name_a, name_b, frozenset(a_list or ()),
                          frozenset(b_list or ()), desc_a, desc_b)
        row.update({"label": pair["label"], "label_reason": pair["label_reason"],
                    "source": pair.get("source", "")})
        if a_list is None or b_list is None:
            row["verdict"] = "unknown"
            row["entity_score"] = None
            row["unknown_reason"] = "missing_entity_set"
        rows.append(row)
    return rows


def fixture_drift(path: Path, texts: dict[str, str], surfaces: dict[str, str], *,
                  alias_keys: frozenset[str] = frozenset()) -> list[str]:
    """Which fixture rows a live corpus would now project differently, and why.

    The scored numbers are frozen by design; this is the honest counterpart, and the only
    thing standing between a dated fixture and a silently rotting one. The archived copy of
    a quarantined skill has to be passed in by the caller: #2440's shipped positive is scored
    on `research-paper-writing` as it read before `17e447d8` archived it in place, and
    `iter_active_skills()` stopped yielding it on the day the fold landed.

    A drift is a report, not a repair. The projection moving because a body was edited is
    legitimate, and the response is to re-read the pair and re-date the row — not to let the
    fixture follow the corpus while keeping the old labels.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    drift = []
    for pair in data["pairs"]:
        for side, name in (("a", pair["a"]), ("b", pair["b"])):
            want = set((pair.get("entity_sets") or {}).get(side) or ())
            if name not in texts:
                drift.append(f'{"|".join((pair["a"], pair["b"]))}:{side} absent_from_corpus')
            else:
                got = set(project(texts[name], surfaces, alias_keys=alias_keys)["entities"])
                if got != want:
                    drift.append(
                        f'{"|".join((pair["a"], pair["b"]))}:{side} '
                        f'+{len(got - want)} -{len(want - got)}')
    return sorted(drift)


def ranking(rows: list[dict]) -> list[dict]:
    """Fixture rows ordered by the entity score, highest first, ties by pair name.

    `unknown` rows sort last rather than by their stored 0.0, because a 0.0 that came from an
    absent file and a 0.0 that came from two disjoint entity sets are different facts and
    only one of them is a measurement.
    """
    def key(row):
        return (row["verdict"] == "unknown",
                -row["entity_score"] if row["verdict"] != "unknown" else 0.0,
                row["a"], row["b"])
    return sorted(rows, key=key)


def ranking_verdict(rows: list[dict]) -> tuple[bool, str]:
    """Whether every labelled duplicate outranks every labelled non-duplicate.

    A ranking claim, not a threshold: the reported answer is the position of the positives.
    Unknown rows are excluded from the comparison and their count printed beside it, so a
    pass over a thin corpus says `2 of 13 pairs measured` rather than silently passing on
    the strength of rows it could not read.
    """
    measured = [r for r in rows if r["verdict"] != "unknown"]
    positives = [r for r in measured if r["label"] == "duplicate"]
    negatives = [r for r in measured if r["label"] != "duplicate"]
    if not positives or not negatives:
        return (False, f"RANKING: not judged ({len(positives)} positive and "
                       f"{len(negatives)} negative pair(s) were measurable of "
                       f"{len(rows)} rows)")
    worst_positive = min(r["entity_score"] for r in positives)
    best_negative = max(r["entity_score"] for r in negatives)
    passed = worst_positive > best_negative
    return (passed,
            f"RANKING: {'PASS' if passed else 'FAIL'} — {len(positives)} positive and "
            f"{len(negatives)} negative pair(s) measured of {len(rows)} rows; "
            f"lowest positive {worst_positive:.4f}, highest negative {best_negative:.4f} "
            f"(not a gate; the incumbent puts the best negative at "
            f"{max(r['incumbent_score'] for r in negatives):.4f})")


def build_report(*, roots: Sequence[Path] | None, store, fixture_path: Path | None,
                 archived_roots: Sequence[Path] = ()) -> dict:
    surfaces = surface_map(store)
    alias_keys = alias_surfaces(surfaces)
    rows = corpus(roots)
    for row in rows:
        proj = project(row["body"], surfaces, alias_keys=alias_keys)
        row["entities"] = sorted(proj["entities"])
        row["n_entities"] = len(row["entities"])
        row["suppressed"] = proj["suppressed"]
    by_name = {r["name"]: r for r in rows}
    ent = {r["name"]: set(r["entities"]) for r in rows}
    # Every pair is emitted, including the ones with no reading. Dropping an unscored pair
    # silently is the same error one step earlier as calling it clean: on the live catalogue the
    # unscored majority IS the coverage finding the item asks to be printed nightly, and a list
    # holding only the measurable pairs reads as "nothing else in the catalogue overlaps". The
    # unscored rows are counted and sampled rather than enumerated — 194 advertised skills is
    # 18,721 pairs, and a report nobody reads is a log.
    pairs, unscored = [], []
    names = sorted(by_name)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = verdict_row(a, b, ent[a], ent[b],
                              by_name[a]["description"], by_name[b]["description"])
            (unscored if row["verdict"] == "unknown" else pairs).append(row)
    pairs.sort(key=lambda r: (-r["entity_score"], r["a"], r["b"]))
    unscored.sort(key=lambda r: (-(r["n_a"] + r["n_b"]), r["a"], r["b"]))
    report = {
        "schema": SCHEMA,
        "corpus_roots": [str(p) for p in (roots or [])] or None,
        "n_skills": len(rows),
        "zero_entity_skills": sum(1 for r in rows if not r["n_entities"]),
        "below_floor_skills": sum(1 for r in rows
                                 if 0 < r["n_entities"] < MIN_ENTITIES_TO_SCORE),
        "min_entities_to_score": MIN_ENTITIES_TO_SCORE,
        "entity_set_size_histogram": size_histogram(rows),
        "suppressed_totals": {
            k: sum(r["suppressed"][k] for r in rows)
            for k in ("tool", "path", "lowercase", "short")},
        "incumbent": {"name": INCUMBENT_NAME, "threshold_quoted_by_incumbent":
                      INCUMBENT_THRESHOLD},
        "entity_overlap_flag": ENTITY_OVERLAP_FLAG,
        "top_pairs": pairs[:40],
        "n_pairs_scored": len(pairs),
        "n_pairs_unknown": len(unscored),
        "pairs_unknown_by_reason": {
            k: sum(1 for r in unscored if r["unknown_reason"] == k)
            for k in ("no_coverage", "thin")},
        "unknown_pairs_sample": unscored[:UNKNOWN_PAIR_SAMPLE],
        "skills": [{"name": r["name"], "n_entities": r["n_entities"],
                    "entities": r["entities"]} for r in
                   sorted(rows, key=lambda r: r["name"])],
    }
    if fixture_path is not None and fixture_path.is_file():
        frows = fixture_rows(fixture_path)
        ok, note = ranking_verdict(frows)
        report["fixture"] = {"path": str(fixture_path), "rows": ranking(frows),
                             "incumbent": INCUMBENT_NAME,
                             "ranking_first_positives": int(ok), "verdict": note}
        # The anti-rot counterpart to scoring the recorded sets: re-project the bytes on disk
        # — the advertised tree AND the archived one, because a skill quarantined in place by
        # its own fold is exactly what the fixture has to reach — and name every side whose
        # set has moved. Printed, never gated: drift is a signal to re-read the pair and
        # re-date the row, not a result.
        live = {row["name"]: row["body"] for row in rows}
        for root in archived_roots:
            if not root.is_dir():
                continue
            for manifest in sorted(root.glob("*/SKILL.md")):
                text = manifest.read_text(encoding="utf-8", errors="ignore")
                live.setdefault(manifest.parent.name, strip_front_matter(text))
        report["fixture"]["drift"] = fixture_drift(fixture_path, live, surfaces,
                                                    alias_keys=alias_keys)
    return report


def render(report: dict) -> str:
    """The stdout shape. Deterministic by construction: no wall-clock anywhere in it."""
    lines = [
        f"schema {report['schema']}",
        f"skills_scanned {report['n_skills']}",
        f"zero_entity_skills {report['zero_entity_skills']}",
        f"below_floor_skills {report['below_floor_skills']} "
        f"(floor={report['min_entities_to_score']})",
        f"entity_set_size_histogram {report['entity_set_size_histogram']}",
        f"suppressed_matches {report['suppressed_totals']}",
        f"pairs_scored {report['n_pairs_scored']} "
        f"(both sides >= floor; unknown pairs are not counted here)",
        f"pairs_unknown {report['n_pairs_unknown']} {report['pairs_unknown_by_reason']}",
    ]
    if report.get("top_pairs"):
        lines.append("top_pairs_by_entity_score:")
        for row in report["top_pairs"][:20]:
            lines.append(
                f"  {row['a']} | {row['b']} :: entity={row['entity_score']:.4f} "
                f"{row['incumbent_name']}={row['incumbent_score']:.4f} "
                f"sizes={row['n_a']}/{row['n_b']} verdict={row['verdict']}"
                + (f" [{row['unknown_reason']}]" if row["unknown_reason"] else ""))
    fixture = report.get("fixture")
    if fixture:
        lines.append(f"fixture {fixture['path']} ({len(fixture['rows'])} pairs):")
        for rank, row in enumerate(fixture["rows"], 1):
            lines.append(
                f"  {rank}. {row['a']} | {row['b']} :: label={row['label']} "
                # A row with no recorded set has no score to print, and printing `None` or a
                # fake 0.0000 would both look like a measurement. The dash and the reason
                # together are the finding: this run has no reading of the pair.
                f"entity={row['entity_score'] if row['entity_score'] is not None else '-'} "
                f"{row['incumbent_name']}={row['incumbent_score']:.4f} "
                f"sizes={row['n_a']}/{row['n_b']} verdict={row['verdict']}"
                + (f" [{row['unknown_reason']}]" if row["unknown_reason"] else ""))
        lines.append(fixture["verdict"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="skill_entity_overlap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="project the catalogue and score the fixture")
    rep.add_argument("--skills-root", action="append", type=Path,
                     help="a skills directory to walk instead of the configured roots "
                          "(repeatable); the walker is still iter_active_skills")
    rep.add_argument("--store", type=Path,
                     help="a knowledge-graph database to resolve against; it must exist, "
                          "because reading counts off a store that is not there is the "
                          "false-clean this module's own coverage number would inherit")
    rep.add_argument("--pairs", type=Path, default=Path("eval/skill_entity_overlap_pairs.json"))
    rep.add_argument("--out", type=Path, help="report path (default _pipeline location)")
    rep.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd != "report":
        return 2
    if args.store is not None:
        if not args.store.is_file():
            sys.stderr.write(f"STORE_UNAVAILABLE {args.store} :: no database there; "
                             "this pass will not invent one and report 0 entities\n")
            return 2
        kg_store.configure(args.store)
    try:
        store = kg_store.store()
    except kg_store.StoreUnavailable as exc:
        sys.stderr.write(f"STORE_UNAVAILABLE {exc}\n")
        return 2
    out = args.out if args.out is not None else PIPELINE_DIR / REPORT_RELPATH
    try:
        report = build_report(roots=args.skills_root, store=store,
                              fixture_path=args.pairs)
    except OSError as exc:
        sys.stderr.write(f"REPORT_FAILED {exc}\n")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(render(report))
    sys.stderr.write(f"report_written {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
