#!/usr/bin/env python3
"""One dated census of the durable-artifact funnel, by stage (#2507).

The loop currently measures its own output as *production* — skills created,
merged and archived per night, facts extracted, items filed. Singh et al.'s
"Agent Plasticity" study (2026-10-06) is the reason that number carries almost no
information: writing an artifact and acquiring a capability are different events,
and the paper separates them into four independently measurable links — the
artifact exists, a later instance finds and uses it, its assumption fits the new
situation, behaviour improves on held-out work. Their per-model headline is a
*stage prevalence*, and for the sub-frontier models in that study the dominant
mode is "artifact available, not used" (98 % of GPT-5.6 Luna's failures), not
"artifact too narrowly written". This box runs a sub-frontier engine, so the
prediction is checkable here rather than importable.

The legs already exist as four separate instruments that each publish a number in
its own artifact and never meet: #711 scores the matcher offline (the surfaced
link alone, away from real traffic), #435 counts a skill offered against a skill
loaded, #552 measures presence-plus-dispute per durable entry, #2378 prices
automod per landed round. This module joins them into one dated artifact so a
single read can answer the only question the four cannot: *which link loses the
most*.

    python -m scripts.funnel_census                 # default 30-day window
    python -m scripts.funnel_census --days 14

v1 measures three links and names two. Stages in funnel order:

  produced   an artifact on disk that the live loader serves (#1294's
             `iter_active_skills` is the definition; a `status:` in the
             quarantine set was written and then pulled from circulation, which
             is a produce-leg loss and nothing else in the funnel can see it)
  offered    an artifact in that cohort that the matcher put forward at least
             once inside the window
  surfaced   an offer that actually rendered into a turn's prompt — `landed`
             on a `prefetch.skill_match` row, plus the `skill.embedded` rows on
             the chat routes, which are the bytes that arrived
  used       NOT MEASURED. Named, with its denominator, never proxied (#2507's
             acceptance: "names the missing stage rather than proxying it").
             `cannot_see.used_surrogate` holds the predicate v1 considered and
             the reason it is not evidence.
  benefited  NOT MEASURED: the store is read where `app/uptake.py` WRITES it
             (`app.uptake.UPTAKE_ARTIFACT_DIR`, three tracked tables today), and
             the newest one is named with its age and its citability keys — but
             the leg still has no denominator, because joining its dispute
             attribution into a per-artifact funnel needs the per-turn key the
             `used` leg has no predicate for. The stage prints a refusal with
             that 0 beside its name rather than a rate, and refuses by name
             rather than citing a table older than `UPTAKE_TABLE_MAX_AGE_DAYS`
             or one stamped `engine_rerouted` / `audit_only` / `measured: false`.

Two further rulings v1 takes and states rather than settles: the cohort is
SKILL.md only (a USER.md/MEMORY.md line sits in every prompt and therefore
attributes everything to nothing — the crux #552 hit), and the cohort is the chat
route only, because the per-turn predicate that would one day read `used` can
only ever see a chat transcript. The uncapped `autonomy_task` / `worker_prompt`
embed routes are counted, named and excluded.

Nothing here acts. v1 gates nothing, prunes nothing and archives nothing on these
counts, and per-skill counts on a one-owner box are too thin for that anyway.

The artifact lands under `app.paths.DATA_ROOT`, never the checkout — #2378's
ruling on in-tree runtime data, enforced by `resolve_out_dir` rather than by
convention: a path inside the code tree is a refusal that exits 2.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import paths                                    # noqa: E402
from app.skill_telemetry import (                        # noqa: E402
    DEFAULT_DAYS, SKILL_MATCH_EVENT, skill_injection_counts)
from eval.stats import wilson_ci                         # noqa: E402

__all__ = ["ARTIFACT_PRODUCING_SOURCES", "COST_SQL_TEMPLATE", "CHAT_EMBED_ROUTES",
           "OUT_OF_COHORT_EMBED_ROUTES", "CensusWriteRefused", "build_census",
           "main", "render_text", "resolve_out_dir"]

#: Funnel order. `stages` is emitted in this order and the report prints it in
#: this order; a stage missing from the list would be a stage nobody reads.
FUNNEL_ORDER = ("produced", "offered", "surfaced", "used", "benefited")

#: The `skill.embedded` routes that mean "these bytes entered a chat turn's
#: prompt" (`app/prefetch.py` writes both: the full body and the runner-up's
#: excerpt). Anything not in this set and not in `OUT_OF_COHORT_EMBED_ROUTES` is
#: published as an unclassified route rather than silently bucketed.
CHAT_EMBED_ROUTES = ("prefetch", "prefetch_excerpt")

#: Routes outside the v1 cohort, counted and named. Both splice a whole SKILL.md
#: into a run prompt with no cap (`app/skill_embed.UNCAPPED_ROUTES`) and neither
#: produces a transcript a per-run `used` predicate could read, so including
#: them in the denominator would claim a funnel leg the census cannot finish.
#: #2507 ruled it: chat-only UNTIL such a per-run predicate exists, so folding them
#: in needs that predicate first — not another ruling, which is why the share of
#: them prints as a cohort boundary line and never as a stage.
OUT_OF_COHORT_EMBED_ROUTES = ("autonomy_task", "worker_prompt", "dispatch")

#: Sources whose job WRITES a durable artifact. `scheduled-task` is the one that
#: matters: it is the queue source the autonomy jobs actually run under — the
#: nightly reflection, skills-management, consolidation and knowledge-write runs
#: that produce skills, memory lines, facts and notes. `autocode` rounds land
#: code and occasionally a skill; `autotriage`, `owed-check`, `bench-mine` and
#: `autoresearch` file items, rulings and notes. The list is explicit rather
#: than `SELECT DISTINCT source` because the census must be able to say what it
#: priced in: an unfiltered sum over `runs` would fold in canaries and queue
#: maintenance and call it the price of learning. Sources that write vault notes
#: but are not on this list (`session-distill`, `youtube-digest`,
#: `deep-research`) are published under `cost.excluded_sources` with their hours
#: so the exclusion is a named line, not a hole.
ARTIFACT_PRODUCING_SOURCES = ("scheduled-task", "autocode", "autotriage",
                              "owed-check", "bench-mine", "autoresearch")

#: Literal-valued on purpose: the acceptance is that the string in the artifact
#: REPRODUCES the numbers when a reader pastes it, so it carries its own source
#: list and window instead of placeholders the reader would have to invent. All
#: interpolated values are module constants or a generated ISO stamp.
COST_SQL_TEMPLATE = (
    "SELECT source, COUNT(*) AS runs, ROUND(SUM(duration_seconds) / 3600.0, 4) "
    "AS gpu_hours FROM runs WHERE source IN ({sources}) "
    "AND started_at >= '{since}' GROUP BY source ORDER BY gpu_hours DESC")

#: Why the price is hours. `usage.cost_usd` is 0.0 on every row of this box, so
#: the study's percentage-points-per-$1,000 denominator has no dollar form here;
#: the paper's own point is that an unpriced gain is not a ranking. The word
#: below is spelled out because the artifact asserts no dollar figure anywhere.
NO_DOLLAR_REASON = ("no dollar denominator exists on this box: usage.cost_usd is "
                    "0.0 on every row, so the loop is priced in GPU-hours "
                    "(wall-clock duration_seconds / 3600, the same proxy the "
                    "dashboard and #2378 print)")

_EVENT_SUFFIX = ".events.jsonl"


class CensusWriteRefused(Exception):
    """The census was asked to write where it may not (inside the code tree)."""


# ── stage shapes ──────────────────────────────────────────────────────────────

def _measured(stage: str, unit: str, k: int, n: int, numerator: str,
              denominator: str, **extra: Any) -> dict:
    """A stage that was measured: `k/n` plus its Wilson interval, or a refusal.

    A zero denominator is never a rate. `wilson_ci(k, 0)` is `(nan, nan)`, which
    is not JSON at all, and printing `0/0 = 0` would render "nothing to count"
    exactly like "counted everything and failed" (#2507 clause 2). So `n == 0`
    returns the refusal shape with the 0 named as its denominator.
    """
    if n <= 0:
        return _refused(stage, "empty_denominator", 0,
                        f"{denominator} — the window held none", unit=unit)
    if k > n:
        # Cannot happen with the predicates below; would mean a numerator that
        # counts rows the denominator never admitted, which is a join bug.
        raise AssertionError(f"stage {stage}: k={k} exceeds n={n}")
    lo, hi = wilson_ci(k, n)
    out: dict[str, Any] = {
        "stage": stage, "measured": True, "unit": unit,
        "k": k, "n": n, "rate": round(k / n, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "numerator": numerator, "denominator": denominator,
        "loss": {"k": n - k, "n": n, "share": round((n - k) / n, 6),
                 "ci95": [round(wilson_ci(n - k, n)[0], 6),
                          round(wilson_ci(n - k, n)[1], 6)]},
    }
    out.update(extra)
    return out


def _refused(stage: str, refusal: str, denominator: int, detail: str,
             *, unit: str | None = None, **extra: Any) -> dict:
    """A stage v1 cannot measure: a NAME, its denominator, and no rate.

    Deliberately carries no `k`, `n`, `rate` or `ci95` keys, so a later reader
    cannot mistake an unmeasured link for a measured zero.
    """
    out: dict[str, Any] = {
        "stage": stage, "measured": False, "refusal": refusal,
        "refusal_label": refusal.replace("_", " "),
        "denominator": denominator, "detail": detail,
    }
    if unit:
        out["unit"] = unit
    out.update(extra)
    return out


# ── stage 1: produced ─────────────────────────────────────────────────────────

def _skill_dirs(roots: Iterable[Path]) -> list[Path]:
    """Every skill directory on disk, quarantine included, first root wins.

    The same walk `iter_active_skills` performs minus its retirement filter, so
    the produced denominator is "artifacts the loop has written and left on
    disk" and the loss is "written but not in circulation". Dot-prefixed roots
    are the archive by #1294's rule 1 and are not artifacts.
    """
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in seen:
                continue
            if not (entry / "SKILL.md").is_file():
                continue
            seen.add(entry.name)
            out.append(entry)
    return out


def _default_roots() -> list[Path]:
    """`agent_mcp.skills.SKILLS_DIRS` — the one definition of where skills live.

    Imported lazily because that module pulls in `mcp`: a census of files should
    not die on a missing MCP install, and when the roots cannot be resolved the
    leg is reported ABSENT rather than proxied by a glob of a guessed directory.
    """
    from agent_mcp.skills import SKILLS_DIRS
    return list(SKILLS_DIRS)


def _cohort_names(roots: list[Path] | None) -> set[str]:
    """The artifact names the live loader serves, from the roots being censused.

    Names, not a count: the offered and surfaced legs are computed by
    intersecting this set with the names the offer rows carry, and a count cannot
    be intersected.
    """
    from agent_mcp.skills import iter_active_skills
    return {s.name for s in iter_active_skills(roots=list(roots) if roots
                                               else _default_roots())}


def produced_stage(roots: list[Path] | None = None) -> dict:
    """Stage 1 — artifacts on disk vs artifacts the live loader serves.

    The denominator is every skill directory left on disk under a skill root, the
    quarantine included; the numerator is the subset `iter_active_skills` (#1294,
    the one definition of a live skill) actually serves. An artifact that was
    written and then retired in place by its `status:` is therefore a
    produce-leg loss, which is the only place in the funnel that can see it — the
    offer rows for a retired skill stop arriving, and downstream that reads as an
    artifact nobody needed.

    `roots` defaults to `agent_mcp.skills.SKILLS_DIRS`, imported lazily so a
    census of files does not die on a missing MCP install; unresolvable roots are
    reported as an absent leg, never proxied by a glob of a guessed directory.
    """
    try:
        roots = list(roots) if roots else _default_roots()
    except Exception as exc:                                # noqa: BLE001
        return _refused("produced", "skill_roots_unresolvable", 0,
                        f"`agent_mcp.skills.SKILLS_DIRS` could not be "
                        f"imported ({type(exc).__name__}: {exc}); the leg is "
                        f"reported absent rather than globbed from a guess")
    on_disk = _skill_dirs(roots)
    names = {d.name for d in on_disk}
    try:
        live = _cohort_names(roots)
    except Exception as exc:                                    # noqa: BLE001
        return _refused("produced", "skill_loader_unimportable", len(on_disk),
                        f"`iter_active_skills` failed ({type(exc).__name__}: "
                        f"{exc}); {len(on_disk)} directories were counted but "
                        f"none could be checked for circulation")
    served = live & names
    missing = sorted(names - served)
    return _measured(
        "produced", "artifact", len(served), len(on_disk),
        "skill directories the live loader serves (`iter_active_skills`, #1294)",
        "directories under a skill root carrying a `SKILL.md`, quarantine included",
        roots=[str(r) for r in roots],
        loss_label="written but out of circulation",
        # Two walks, one corpus: a name the loader serves that the walk did not
        # find is the two disagreeing, and it is published, never reconciled.
        walk_disagreement=sorted(live - names)[:20],
        not_in_circulation=missing[:20],
        not_in_circulation_more=max(0, len(missing) - 20))


# ── the event-side scan ───────────────────────────────────────────────────────

def _scan_embed_rows(root: Path, since: datetime, until: datetime) -> dict:
    """`skill.embedded` rows in the window by route, plus the injected-body split.

    A second pass over the store, and it has to exist: `skill_injection_counts`
    is the mandated reader for the offer/`landed` numbers (clause 3) and does not
    expose `injected_body` or the embed routes at all, which are the two things
    `cannot_see` needs. The window is the one the telemetry reader returned
    rather than a fresh `now()`, so the two scans cannot disagree by seconds.

    `landed=True, injected_body=False` is published, not bucketed, and it is not a
    stage: #2507 ruled excerpt-only a RENDER MODE of `surfaced`, because
    `app/prefetch.py:733-734` derives both flags from one `mode` (`landed` is
    "a mode was chosen", `injected_body` is "that mode was `body`"). So the split
    stays inside the surfaced count and prints beside it as a share; folding these
    rows into `surfaced_not_used` would count one event twice, and giving them a
    stage of their own would make the funnel's denominator depend on a renderer's
    budget decision.
    """
    per_route: dict[str, int] = {}
    per_route_skills: dict[str, set[str]] = {}
    landed_body = landed_no_body = 0
    for path in sorted(Path(root).glob(f"*{_EVENT_SUFFIX}")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            data = row.get("data")
            event, stamp = row.get("event"), row.get("ts")
            if not isinstance(data, dict) or not isinstance(stamp, str):
                continue
            try:
                when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)   # one clock, as every
            elif when < since or when > until:             # reader here reads it
                continue
            if event == "skill.embedded":
                route = str(data.get("route") or "unset")
                per_route[route] = per_route.get(route, 0) + 1
                name = data.get("skill")
                if isinstance(name, str) and name:
                    per_route_skills.setdefault(route, set()).add(name)
            elif event == SKILL_MATCH_EVENT and data.get("landed") is True:
                if data.get("injected_body") is True:
                    landed_body += 1
                else:
                    landed_no_body += 1
    return {"rows_by_route": per_route,
            # Names are capped: the out-of-cohort set is what a reader of the
            # cohort boundary has to read, and 200 names is more than that needs.
            # #2507 settled which routes are out of cohort; the names are here to
            # show what sits behind the count, not to decide it. The COUNTS above
            # are uncapped and are what every rate here
            # divides by — a cap on a published name list never touches a
            # denominator.
            "skills_by_route": {r: sorted(v)[:40] for r, v in per_route_skills.items()},
            "skills_by_route_distinct": {r: len(v) for r, v in per_route_skills.items()},
            "landed_rows": {"with_injected_body": landed_body,
                            "without_injected_body": landed_no_body},
            "phantom_warning": (
                "a `<skill>` tag quoted in prose is booked by "
                "`skill_dispatch._walk_deliveries` as everything after it, so "
                "some names here (`'…'`, `'{rule.skill}'`) are quoted prose, not "
                "artifacts — see the re-derive block in "
                "`app/skill_embed.py:56-78`. Cohort intersection is what "
                "filters them out of every rate, which is why no rate here is "
                "computed from these name lists alone")}


def _route_buckets(per_route: dict[str, int]) -> dict:
    """Publish every route the store holds, bucketed by the cohort ruling."""
    known = set(CHAT_EMBED_ROUTES) | set(OUT_OF_COHORT_EMBED_ROUTES)
    return {
        "in_cohort": {r: per_route[r] for r in CHAT_EMBED_ROUTES if r in per_route},
        "excluded_named": {r: per_route[r] for r in OUT_OF_COHORT_EMBED_ROUTES
                           if r in per_route},
        "unclassified": {r: c for r, c in per_route.items() if r not in known},
    }


def surface_stages(cohort: set[str] | None, event_root: Path, days: int) -> tuple[dict, dict, dict]:
    """Stages 2 and 3 — offered, then surfaced — from the events that exist.

    `skill_injection_counts` is the reader, not `app.event_log.read_events`:
    that one is the paginated UI reader (a 200-row default page) and a window
    over a busy session is thousands of rows, so building a count on it reports
    the tail and silently drops the rest — the failure
    `tests/test_skill_injection_telemetry.py` already pins for #435, pinned again
    here because a new caller is a new chance to make the old mistake.

    Returns `(offered, surfaced, detail)`.
    """
    tele = skill_injection_counts(event_root, days)
    skills: dict[str, dict[str, float]] = tele["skills"]
    offers = sum(int(v["offers"]) for v in skills.values())
    landed = sum(int(v["loaded"]) for v in skills.values())
    ignored = sum(int(v["ignored"]) for v in skills.values())
    since = datetime.fromisoformat(tele["since"])
    until = datetime.fromisoformat(tele["until"])
    embed = _scan_embed_rows(event_root, since, until)
    chat_names = {n for r in CHAT_EMBED_ROUTES
                  for n in embed["skills_by_route"].get(r, ())}
    out_names = {n for r in OUT_OF_COHORT_EMBED_ROUTES
                 for n in embed["skills_by_route"].get(r, ())}

    if cohort is None:
        offered = _refused("offered", "no_cohort", 0,
                           "the produced leg has no cohort to divide by, so "
                           "there is no denominator for artifacts that were "
                           "matched at least once")
        reached = None
    else:
        offered_names = {name for name, v in skills.items() if int(v["offers"]) > 0}
        offered = _measured(
            "offered", "artifact", len(offered_names & cohort), len(cohort),
            "cohort artifacts with at least one `prefetch.skill_match` offer row "
            "in the window",
            f"the produced cohort: artifacts the live loader serves "
            f"({len(cohort)})",
            loss_label="never offered in the window",
            offered_not_in_cohort=sorted(set(offered_names) - cohort)[:20])
        reached = sorted(({name for name, v in skills.items() if int(v["loaded"]) > 0}
                          | chat_names) & cohort)

    if offers == 0:
        surfaced = _refused(
            "surfaced", "no_offer_rows_in_window", 0,
            f"zero `{SKILL_MATCH_EVENT}` rows landed inside the window "
            f"({tele['note']}); nothing downstream of an offer is countable",
            unit="offer row")
    else:
        surfaced = _measured(
            "surfaced", "offer row", landed, offers,
            f"`{SKILL_MATCH_EVENT}` rows with `landed: true` — the skill body "
            f"rendered into that turn's prompt",
            f"`{SKILL_MATCH_EVENT}` offer rows in the window",
            loss_label="offers that never landed",
            artifacts_reaching_this_stage=(len(reached) if reached is not None else None))

    detail = {
        "telemetry": {k: tele[k] for k in ("window_days", "since", "until",
                                           "first_event", "events", "sessions",
                                           "skipped", "reads", "no_telemetry")},
        "skills_offered": len(skills),
        "skills_loaded": sum(1 for v in skills.values() if int(v["loaded"]) > 0),
        "pulled_by_skills_read": tele["loaded_by_read"],
        "offer_rows_ignored": ignored,
        "embed_rows": embed,
        "embed_routes": _route_buckets(embed["rows_by_route"]),
        "artifacts_reaching_surfaced": reached,
        "out_of_cohort_artifacts": sorted(out_names - (set(reached or ()))) [:20],
    }
    return offered, surfaced, detail


# ── stages 4 and 5: the links v1 names instead of measuring ───────────────────

def used_stage(reached: list[str] | None) -> dict:
    """`used` — NOT MEASURED. The denominator is what it would divide.

    Naming the denominator is the point: an unmeasured link printed without one
    reads as "zero artifacts were used", which is a claim about behaviour and
    false; printed with one it reads as "N artifacts reached a prompt and nothing
    here can say what happened next", which is the truth of it.
    """
    return _refused(
        "used", "no_mechanical_use_signal", len(reached) if reached is not None else 0,
        "no deterministic predicate over the traces on disk separates a turn "
        "that followed a skill body from one that merely carried it, and #630 is "
        "the reason v1 does not invent one: a presence-only scorer there read an "
        "echoed constraint list as 99 % compliance, so a body sitting in a "
        "prompt is not evidence of use. See `cannot_see.used_surrogate`.",
        unit="artifact",
        surrogate_not_taken=(
            "grep the session JSON for a tool call naming a tool the skill's own "
            "body names, after the injected block — rejected for v1 because a "
            "SKILL.md declares no tool set, so the predicate would be a guess "
            "per skill, and because the `skills_read` pull count that IS "
            "mechanical measures reaching for an artifact, not applying it"))


#: The uptake table's filename pattern, spelled here rather than imported:
#: `app/uptake.py` documents the shape (`uptake-YYYY-MM-DD.json`) but publishes no
#: glob constant of its own, and `scripts/memory_revision_candidates.py` hard-codes
#: the same pattern for the same reason.
UPTAKE_TABLE_GLOB = "uptake-*.json"

#: How old an uptake table may be before this stage refuses to speak from it.
#: Deliberately NOT imported from `scripts/eval_instrument_index.py`, which also
#: holds 30 for this store but publishes it only inside an `Instrument` literal
#: for its own staleness column and owns nothing a reader may rely on; and
#: deliberately not `SPILL_MAX_AGE_DAYS` from the retention sweep, which ages a
#: completely different store. 30 is the window the probe itself runs over, so a
#: table older than this describes a corpus the probe no longer reads. A table
#: past it is named and refused, never cited as a benefited-leg denominator.
UPTAKE_TABLE_MAX_AGE_DAYS = 30


#: The three depths an uptake table's keys can live at, in the order the reader
#: walks them. Not invented here: the instrument's own schema test spells the
#: same walk (`tests/test_uptake.py::
#: test_the_citability_rule_asks_about_the_keys_uptake_emits`, whose
#: `instrument_writes` is the positive control), because the writer puts
#: `audit_only` at table level (via `write_table`'s `extra`), `engine_rerouted`
#: in the `classifier` block (`scripts/uptake_probe.py`'s `classifier_block`),
#: and `measured` only inside that block's sub-blocks — never beside the others.
#: A reader that looks only at the top level therefore reports two of the three
#: keys absent on EVERY real table, which is the difference between "the file
#: does not say" and "the file says it was scored by a rerouted engine".
UPTAKE_TABLE_SUBBLOCKS: tuple[str, ...] = ("metrics", "holdout", "zero_shot")


def _uptake_key_lookup(doc: dict, key: str) -> tuple[bool, Any]:
    """Is this key stated in the table, and with what value? Depth-aware.

    Returns `(stated, value)`; `stated` is False only when no level carries it.
    """
    if key in doc:
        return True, doc[key]
    block = doc.get("classifier")
    if not isinstance(block, dict):
        return False, None
    if key in block:
        return True, block[key]
    for sub in UPTAKE_TABLE_SUBBLOCKS:
        inner = block.get(sub)
        if isinstance(inner, dict) and key in inner:
            return True, inner[key]
    return False, None


def _uptake_citability_states(doc: dict) -> dict[str, str]:
    """Each citability key's state in one table: `absent` / `true` / `false`.

    `absent` is its OWN state and never `citable`: the three committed tables
    carry none of the keys at all — they predate #1850, which introduced
    `engine_rerouted` — while `app/uptake.py`'s own glossary reads
    "`audit_only` ABSENT means this table cleared every floor and is citable".
    That reading is sound for a table the current writer stamped and unsound for
    one it never stamped at all, so this stage reports what the file does or does
    not say and decides nothing about citability from an absent key. "Says
    nothing" is decided only after all three depths are read, so an absent
    `engine_rerouted` here means the FILE omits it, not that this reader missed
    it.
    """
    from scripts.uptake_citability import CITABILITY_KEYS
    out: dict[str, str] = {}
    for key in CITABILITY_KEYS:
        stated, value = _uptake_key_lookup(doc, key)
        out[key] = "absent" if not stated else ("true" if value is True else "false")
    return out


def _uptake_table_read(path: Path) -> tuple[dict | None, str]:
    """A table's bytes, and why they could not be read. Never raises."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(doc, dict):
        return None, f"top-level JSON is a {type(doc).__name__}, not an object"
    return doc, ""


def _uptake_table_age_days(doc: dict, now: datetime) -> tuple[int | None, str]:
    """The table's age in whole days, measured on its OWN `probe_timestamp`.

    `probe_timestamp` is the only age field a table has: there is no top-level
    `date`, and the files' mtimes are checkout mtimes — the retention sweep does
    not `copystat` what it archives, so every table here reads the day it was
    cloned. A missing or unparseable stamp is not age 0: the second element names
    it, and the caller refuses rather than treating an undated table as fresh.
    """
    raw = doc.get("probe_timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return None, "probe_timestamp is absent"
    try:
        stamp = datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        return None, f"probe_timestamp {raw!r} is not ISO-8601 ({exc})"
    if stamp.tzinfo is None:
        # A naive stamp in a machine-facing payload is read as UTC by every later
        # reader (the class rule of 2026-09-21), so it is read as UTC here too.
        stamp = stamp.replace(tzinfo=timezone.utc)
    return max(0, (now - stamp).days), ""


def benefited_stage(uptake_dir: Path | None = None,
                    *, now: datetime | None = None) -> dict:
    """`benefited` — NOT MEASURED, and its denominator is genuinely 0.

    The store is resolved through `app.uptake.UPTAKE_ARTIFACT_DIR`, the constant
    the probe's own `write_table` outputs to — NOT through a re-spelled
    `DATA_ROOT / "eval" / "uptake"`, which is the path this stage held until
    #2733 and which nothing on this box has ever written. The attribute is read
    off the module at CALL time (`uptake.UPTAKE_ARTIFACT_DIR`, not a value
    imported once) for the same reason `scripts/memory_revision_candidates.py`
    spells it that way: it is what a fixture repoints. #2597's rule is that an
    instrument is read only where its own writer declares it, and this store is
    the single eval instrument that lives inside the code tree
    (`git ls-files eval/uptake`) rather than under the data root.

    The newest table is named by filename, aged on its own `probe_timestamp` and
    published with each citability key's state, so "the store is absent" can only
    ever be said of a directory that really holds none.

    Naming the store still cannot produce a rate, and the question of whether to
    buy a table at all is settled rather than pending: #2507 ruled on 2026-10-15
    NOT to pay the secondary engine to attempt the leg, and the measurement
    forecloses it rather than the cost — `config.yaml:38` carries
    `secondary_enabled: false`, so the scorer reroutes away from the engine this
    instrument names, and `scripts/uptake_probe.stamp_engine` stamps what such a
    run scored uncitable: `measured: false` on every block, `passed: false` on the
    run. Two gaps stand behind that one: `scripts/uptake_probe.py` exits 3 and
    deliberately writes nothing below precision 0.70, and joining a table's
    disputes into a per-artifact denominator needs the per-turn key the `used` leg
    has no predicate for. So the leg ships as the named gap. Every branch of this
    function returns `measured: false` with `denominator: 0`.
    """
    from app import uptake as _uptake
    uptake = Path(uptake_dir) if uptake_dir else Path(_uptake.UPTAKE_ARTIFACT_DIR)
    now = now or datetime.now(timezone.utc)
    detail: dict[str, Any] = {"uptake_store": str(uptake)}
    if not uptake.is_dir():
        return _refused(
            "benefited", "no_uptake_table", 0,
            f"`{uptake}` does not exist — it is the directory "
            f"`app.uptake.UPTAKE_ARTIFACT_DIR` names, so the probe has written no "
            f"table here: it exits 3 rather than write one below precision 0.70, "
            f"which leaves no dispute signal to divide by. Buying a table is ruled "
            f"out rather than open: #2507 ruled on 2026-10-15 not to pay the "
            f"secondary engine, because with `secondary_enabled: false` "
            f"(`config.yaml:38`) the scorer reroutes and "
            f"`scripts/uptake_probe.stamp_engine` stamps what such a run scored "
            f"uncitable (`measured: false` on every block, `passed: false` on the "
            f"run), and the join needs the per-turn key the `used` leg has no "
            f"predicate for. The leg ships as the named gap, not a step here",
            unit="artifact", **detail)
    names = sorted(p.name for p in uptake.glob(UPTAKE_TABLE_GLOB))
    if not names:
        return _refused(
            "benefited", "no_uptake_table", 0,
            f"`{uptake}` exists but holds no `{UPTAKE_TABLE_GLOB}` table: "
            f"`scripts/uptake_probe.py` exits 3 rather than write one below "
            f"precision 0.70, so there is no dispute signal to divide by. Buying a "
            f"table is ruled out rather than open: #2507 ruled on 2026-10-15 not "
            f"to pay the secondary engine, because with `secondary_enabled: false` "
            f"(`config.yaml:38`) the scorer reroutes and "
            f"`scripts/uptake_probe.stamp_engine` stamps what such a run scored "
            f"uncitable (`measured: false` on every block, `passed: false` on the "
            f"run), and the join needs the per-turn key the `used` leg has no "
            f"predicate for. The leg ships as the named gap, not a step here",
            unit="artifact", **detail)
    newest = uptake / names[-1]
    detail["uptake_table"] = newest.name
    doc, read_err = _uptake_table_read(newest)
    if doc is None:
        return _refused(
            "benefited", "uptake_table_unreadable", 0,
            f"`{newest}` — the newest table in `{uptake}` by filename — could not "
            f"be read as a table ({read_err}), so its age and citability are "
            f"unknown and nothing may be spoken from it",
            unit="artifact", **detail)
    detail["uptake_citability"] = _uptake_citability_states(doc)
    age_days, age_err = _uptake_table_age_days(doc, now)
    detail["uptake_table_age_days"] = age_days
    reasons: list[str] = []
    if age_err:
        reasons.append(f"its `probe_timestamp` is unreadable ({age_err})")
    elif age_days > UPTAKE_TABLE_MAX_AGE_DAYS:
        reasons.append(f"it is {age_days} days old, over the "
                       f"{UPTAKE_TABLE_MAX_AGE_DAYS}-day ceiling this stage will "
                       f"speak from")
    states = detail["uptake_citability"]
    if states["engine_rerouted"] == "true":
        reasons.append("`engine_rerouted` is true")
    if states["audit_only"] == "true":
        reasons.append("`audit_only` is true, so every dispute count in it is an "
                       "unvalidated classifier's guess")
    if states["measured"] == "false":
        reasons.append("`measured` is false")
    if reasons:
        return _refused(
            "benefited", "uptake_table_uncitable", 0,
            f"the newest uptake table `{newest.name}` in `{uptake}` is not a "
            "table this census may speak from: " + "; ".join(reasons)
            + ". Citability keys as the file states them: "
            + ", ".join(f"{k}={v}" for k, v in states.items())
            + " (an absent key is that file's own state, not a cleared floor). "
            "Naming the store is as far as v1 goes; the leg stays named rather "
            "than proxied",
            unit="artifact", **detail)
    return _refused(
        "benefited", "uptake_table_not_read_in_v1", 0,
        f"`{uptake}` holds its newest table `{newest.name}` "
        f"({age_days} days old; citability keys as the file states them: "
        + ", ".join(f"{k}={v}" for k, v in states.items()) + "), but joining its "
        "dispute attribution into a per-artifact funnel needs the per-turn key "
        "the `used` leg has no predicate for, so the denominator stays 0 and the "
        "leg stays named rather than proxied",
        unit="artifact", **detail)


# ── the priced line ───────────────────────────────────────────────────────────

def cost_block(workers_db: Path, since: str, reached: list[str] | None,
               sources: Iterable[str] = ARTIFACT_PRODUCING_SOURCES) -> dict:
    """GPU-hours spent by the artifact-writing jobs, per artifact that reached use.

    One store, one query, and the query is embedded because the acceptance is
    that the artifact reproduces: `started_at` is compared as TEXT, which is
    correct for exactly as long as every writer keeps `app.event_log`'s ISO+00:00
    shape, and the printed `first_run`/`last_run` are the evidence that the rows
    this window selected really are placed where the window says.

    `duration_seconds` is wall-clock and includes gate-rung subprocess time, so
    this is a proxy and is labelled one. It is never summed with a ledger
    duration field: #2378 found `rung_wait_s` and `waited_s` measure different
    spans and reading them together double-counts the wait.
    """
    src_list = list(sources)
    sql = COST_SQL_TEMPLATE.format(
        sources=", ".join(f"'{s}'" for s in src_list), since=since)
    out: dict[str, Any] = {
        "store": str(workers_db), "sql": sql, "sources": src_list,
        "window_start": since,
        "gpu_hours_is": ("wall-clock run duration from workers.db `runs."
                         "duration_seconds` / 3600 — a proxy, since a GPU-hours "
                         "meter is not kept per source"),
        "dollar_figure": NO_DOLLAR_REASON,
    }
    try:
        con = sqlite3.connect(f"file:{workers_db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        out["refusal"] = f"store_unreadable: {exc}"
        return out
    try:
        rows = [list(r) for r in con.execute(sql)]
        span = con.execute(
            "SELECT MIN(started_at), MAX(started_at), COUNT(*) FROM runs").fetchone()
    except sqlite3.Error as exc:        # a missing `runs` table is a leg, not a crash
        out["refusal"] = f"query_failed: {exc}"
        return out
    finally:
        con.close()
    total_rows = sum(int(r[1]) for r in rows)
    hours = round(sum(float(r[2] or 0.0) for r in rows), 4)
    out.update({
        "per_source": [{"source": r[0], "runs": int(r[1]),
                        "gpu_hours": round(float(r[2] or 0.0), 4)} for r in rows],
        "runs": total_rows, "gpu_hours": hours,
        "store_span_first_run": span[0], "store_span_last_run": span[1],
        "store_span_rows": int(span[2] or 0),
    })
    reached_count = len(reached) if reached is not None else 0
    out["reached_use"] = {
        "count": reached_count,
        "taken_at_stage": "surfaced",
        "why": "the `used` leg is unmeasured, so the divisor is the artifacts "
               "that reached a prompt in the window — an UPPER bound on artifacts "
               "reached and applied, which makes the hours-per-artifact figure "
               "below a LOWER bound on the true price",
    }
    if reached_count and hours:
        out["gpu_hours_per_artifact_reaching_use"] = round(hours / reached_count, 4)
    else:
        out["gpu_hours_per_artifact_reaching_use"] = None
        out["ratio_refusal"] = ("no reached-use artifacts or no hours in the "
                                "window: the denominator is empty, so no rate is "
                                "printed (0 would read as a free loop)")
    return out


def excluded_sources(workers_db: Path, since: str) -> list[dict]:
    """Sources that write durable artifacts but are NOT priced in v1, with hours.

    A named exclusion, so the next reader can widen the list from a number
    instead of a guess. Best-effort: an unreadable store yields `[]`, because
    this block is a footnote to the priced line and not the priced line.
    """
    named = ("session-distill", "youtube-digest", "deep-research", "board-steward")
    sql = COST_SQL_TEMPLATE.format(
        sources=", ".join(f"'{s}'" for s in named), since=since)
    try:
        con = sqlite3.connect(f"file:{workers_db}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = list(con.execute(sql))
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    return [{"source": r[0], "runs": int(r[1]),
             "gpu_hours": round(float(r[2] or 0.0), 4)} for r in rows]


# ── joining it ────────────────────────────────────────────────────────────────

def loss_shares(stages: list[dict]) -> dict[str, dict]:
    """Each measured stage's loss, keyed by stage, over its own denominator."""
    return {s["stage"]: dict(s["loss"], unit=s["unit"],
                             label=s.get("loss_label", "not reaching this stage"))
            for s in stages if s.get("measured") and s.get("loss")}


def dominant_loss(shares: dict[str, dict]) -> dict | None:
    """The one stage that loses the most, or None when nothing was measured.

    Loss shares are fractions *within each stage's own denominator*, so they are
    compared as shares and reported with the k/n that produced them — the k/n
    travels with the name so no reader can take 16169/21542 for an artifact
    count. Ties break on the larger denominator, then on funnel order, so the
    line is a function of the data and not of dict order.
    """
    order = {name: i for i, name in enumerate(FUNNEL_ORDER)}
    best = None
    for stage, loss in shares.items():
        key = (-loss["share"], -loss["n"], order.get(stage, 99))
        if best is None or key < best[0]:
            best = (key, stage, loss)
    if best is None:
        return None
    _, stage, loss = best
    return {"stage": stage, "k": loss["k"], "n": loss["n"],
            "share": loss["share"], "denominator_unit": loss["unit"],
            "loss_label": loss["label"]}


def cannot_see(used: dict) -> dict:
    """What this census structurally cannot see, written into what it prints."""
    return {
        "used_surrogate": {
            "predicate_v1_considered": (
                "for a turn whose prompt carried a skill body, a later tool call "
                "in that same turn naming one of that skill's tools, over "
                "`~/lloyd-data/sessions/*.json`"),
            "blind_spot": (
                "a body sitting in a prompt is not evidence the model followed "
                "it — #630 scored an echoed constraint list as 99 % compliance "
                "— and the predicate's own half is worse: SKILL.md declares no "
                "tool set, so which calls count would be authored per skill and "
                "would read a coincidental call as compliance. `surfaced` "
                "therefore never implies `benefited`"),
            "not_measured_because": used["detail"],
        },
        "retention_policy_bounding_the_window": {
            "sessions_store": (
                "the retention sweep gzips `~/lloyd-data/sessions/*.json` once a "
                "conversation session is 90 days inactive and a background one "
                "30 days (`SESSION_ARCHIVE_AGE_DAYS` / "
                "`BACKGROUND_SESSION_ARCHIVE_AGE_DAYS`), so a window deeper than "
                "that is bounded by archives, not by `*.json`. This census does "
                "not read transcripts, so its own window is not truncated by "
                "that; the `used` predicate it declines would be"),
            "event_logs": (
                "`~/lloyd-data/event_logs/` is in none of the retention sweep's "
                "seventeen stores, so offer rows are not aged out — the window's "
                "depth here is the day the emitter first shipped "
                "(`window.measured_from`), not a sweep's doing"),
            "survivorship": (
                "an artifact produced near the end of the window had less "
                "opportunity to be offered and surfaced than one produced at the "
                "start; no stage corrects for that, so the surfaced and offered "
                "shares are pessimistic for recent artifacts"),
        },
        "correct_but_never_applicable": (
            "an artifact that is correct and simply never came up in the window "
            "is indistinguishable here from one that was applicable and ignored; "
            "`offered` loss cannot tell those two apart, which is why nothing "
            "prunes on it"),
        "landed_without_injected_body": (
            "counted inside `surfaced` and published as a share of it "
            "(`surfaced excerpt-only: k/n`), because #2507 ruled it a RENDER MODE "
            "of surfaced and not a stage: `app/prefetch.py:733-734` sets both "
            "flags from ONE `mode` (`landed` is `mode is not None`, `injected_body` "
            "is `mode == \"body\"`), so a row with no body is an offer that landed "
            "in the cheaper of two render modes, not an event between two stages. "
            "The split lives in `surface_detail.embed_rows.landed_rows`"),
        "artifact_classes_not_censused": (
            "USER.md / MEMORY.md lines, KG facts and knowledge notes: v1 is "
            "SKILL.md-only, and the reason is settled rather than pending — no "
            "per-turn predicate exists that could read a fact or a note the way a "
            "hash check reads a skill body, and those classes are already measured "
            "by the nightly retrieval baseline (`eval/run_eval.py`, which scores "
            "entity and doc hits over `vault_recall`), so a funnel over them here "
            "would duplicate an instrument rather than extend one. A "
            "USER.md/MEMORY.md line also sits in every prompt and so attributes "
            "every outcome to everything in force, which is the dilution #552 hit"),
    }


def build_census(days: int = DEFAULT_DAYS, *, event_root: Path | None = None,
                 workers_db: Path | None = None,
                 skills_roots: list[Path] | None = None,
                 uptake_dir: Path | None = None,
                 now: datetime | None = None) -> dict:
    """Assemble the whole census. Every argument exists for a test or a job."""
    from app import paths as _paths
    event_root = Path(event_root) if event_root else _paths.EVENT_LOGS_DIR
    workers_db = Path(workers_db) if workers_db else _paths.WORKERS_DB
    now = now or datetime.now(timezone.utc)

    produced = produced_stage(skills_roots)
    # The join's key: the served artifact NAMES, so the offer rows can be
    # intersected with the cohort. `None` (rather than the empty set) when the
    # produce leg itself failed, because "the cohort is empty" is a measurement
    # and "there is no cohort" is an absent leg — the two print differently.
    cohort = _cohort_names(skills_roots) if produced.get("measured") else None
    offered, surfaced, detail = surface_stages(cohort, event_root, days)
    used = used_stage(detail.get("artifacts_reaching_surfaced"))
    # `now` is passed on so the table's age is measured on the SAME clock as the
    # window: a census assembled at a pinned instant must not age its uptake
    # table against a later wall clock.
    benefited = benefited_stage(uptake_dir=uptake_dir, now=now)
    stages = [s for s in (produced, offered, surfaced, used, benefited)]
    shares = loss_shares(stages)
    since = detail["telemetry"]["since"]
    cost = cost_block(workers_db, since, detail.get("artifacts_reaching_surfaced"))
    cost["excluded_sources"] = excluded_sources(workers_db, since)
    until = detail["telemetry"]["until"]
    return {
        "artifact": "funnel-census", "version": 1, "item": "#2507",
        "measured_at": now.isoformat(timespec="seconds"),
        # Which tree's stores these numbers came from. `app.paths` anchors a
        # worktree's data inside the worktree, so a census run from a round
        # measures an empty partial store and reads exactly like a quiet week;
        # the artifact says which tree it looked at instead of leaving the
        # reader to guess.
        "tree": _paths.describe_tree(),
        "artifact_class": "SKILL.md",
        "window": {
            "days_asked": days, "since": since, "until": until,
            "measured_from": detail["telemetry"]["first_event"],
            "note": (f"asked {days}d; the offer store reaches back to "
                     f"{detail['telemetry']['first_event']} and holds "
                     f"{detail['telemetry']['events']} "
                     f"{SKILL_MATCH_EVENT} rows in the window"),
        },
        "stages": stages,
        "loss_shares": shares,
        "dominant_loss_stage": dominant_loss(shares),
        "surface_detail": detail,
        "cost": cost,
        "cannot_see": cannot_see(used),
        "acts_on": ("nothing: v1 gates, prunes and archives nothing on these "
                    "counts, and per-skill counts on a one-owner box are too "
                    "thin for it anyway"),
    }


# ── the printed report ────────────────────────────────────────────────────────

def render_text(census: dict) -> str:
    """The report, ending in the line the census exists for.

    Exactly one line may begin `dominant loss stage: ` and it names one stage
    with its k/n — no runners-up beside it, because a line that lists candidates
    is the number the reader remembers and the candidates are what the reader
    would act on.

    Two further lines print shares the census had already computed but never
    showed (#2732): `surfaced excerpt-only: k/n`, the render-mode split inside the
    surfaced count, and `out-of-cohort embed rows: k/n`, the cohort boundary. Both
    are ratios BESIDE a stage. Neither may enter `stages[]` or `loss_shares`, and
    neither can reach `dominant_loss_stage`, which is derived from
    `loss_shares(stages)` — so a reader who wants the funnel's arithmetic unchanged
    can check that in the same artifact rather than trust this sentence.
    """
    w = census["window"]
    lines = [
        f"durable-artifact funnel census {census['item']} — artifact class "
        f"{census['artifact_class']} — measured {census['measured_at']}",
        f"tree: {census['tree']}",
        f"window: asked {w['days_asked']}d, {w['since']} -> {w['until']} "
        f"(measured from {w['measured_from']})",
        w["note"],
    ]
    for st in census["stages"]:
        if st.get("measured"):
            lo, hi = st["ci95"]
            lines.append(f"stage {st['stage']}: {st['k']}/{st['n']} = "
                         f"{st['rate']:.3f} (95% CI {lo:.3f}-{hi:.3f}) "
                         f"[{st['unit']}]")
        else:
            lines.append(f"stage {st['stage']}: REFUSAL {st['refusal']} "
                         f"(denominator {st['denominator']})")
    for stage, loss in census["loss_shares"].items():
        note = f"  <- {loss['label']}" if loss.get("label") else ""
        lines.append(f"loss share {stage}: {loss['k']}/{loss['n']} = "
                     f"{loss['share'] * 100:.1f}%{note}")
    # ── Two shares the census already computed and never printed (#2732) ──────
    # Both are ratios BESIDE a stage, never stages: `stages[]` is the five funnel
    # legs and `dominant_loss_stage` is drawn from `loss_shares(stages)`, so a line
    # appended here cannot move either, and a reader who wanted it to read as a leg
    # can check that against `stages[]` in the same artifact.
    #
    # The first is the render-mode split inside `surfaced` that #2507 ruled NOT a
    # stage: over the live 30-day window it measured 3,141 of 6,297 landed rows on
    # 2026-10-15 — 49.9%, and 49.9% in every measurement since, though the counts
    # drift by the minute. Half of everything that surfaced arrived as an excerpt,
    # which the old report counted and then hid. The second is the cohort boundary:
    # 734 of 7,042 embed rows in the same window arrived on an out-of-cohort route
    # (10.4%), and printing it is what keeps the denominator question from being
    # answered by anyone who forgets to ask it.
    #
    # Both k and n are read out of the published dict, never recomputed here: a
    # number restated in two places is two numbers, and the JSON artifact is the
    # one a later report parses.
    embed = census["surface_detail"]["embed_rows"]
    landed = embed.get("landed_rows") or {}
    l_wo = int(landed.get("without_injected_body") or 0)
    l_n = l_wo + int(landed.get("with_injected_body") or 0)
    lines.append(
        f"surfaced excerpt-only: {l_wo}/{l_n} of landed rows (a render mode of "
        f"surfaced, not a stage)" if l_n else
        "surfaced excerpt-only: REFUSAL - the embed-row scan published no "
        "`landed_rows` (telemetry off, or no landed row in the window), so there "
        "is no render-mode split to divide")
    per_route = embed.get("rows_by_route") or {}
    ex_n = sum(int(per_route.get(r) or 0) for r in OUT_OF_COHORT_EMBED_ROUTES)
    emb_n = sum(int(v) for v in per_route.values())
    lines.append(
        f"out-of-cohort embed rows: {ex_n}/{emb_n} - the cohort boundary, not a "
        f"funnel stage ({'/'.join(OUT_OF_COHORT_EMBED_ROUTES)} splice a body into "
        f"a run prompt with no transcript a per-run `used` predicate could read)"
        if emb_n else
        "out-of-cohort embed rows: REFUSAL - the embed-row scan published no "
        "`rows_by_route` (telemetry off, or no `skill.embedded` row in the "
        "window), so the cohort boundary has no denominator")
    cost = census["cost"]
    if "gpu_hours" in cost:
        lines.append(f"cost: {cost['gpu_hours']} GPU-hours over {cost['runs']} runs "
                     f"from sources {', '.join(cost['sources'])} "
                     f"(window started_at >= {cost['window_start']})")
        for row in cost["per_source"]:
            lines.append(f"  {row['source']}: {row['runs']} runs, "
                         f"{row['gpu_hours']} h")
        ru = cost["reached_use"]
        lines.append(f"reached use: {ru['count']} artifacts, counted at the "
                     f"{ru['taken_at_stage']} stage (the used leg is unmeasured)")
        rate = cost.get("gpu_hours_per_artifact_reaching_use")
        lines.append(
            f"price: {rate} GPU-hours per artifact reaching use (wall-clock "
            f"proxy; {NO_DOLLAR_REASON})" if rate is not None
            else f"price: REFUSAL {cost.get('ratio_refusal')}")
        lines.append(f"cost sql: {cost['sql']}")
    else:
        lines.append(f"cost: REFUSAL {cost.get('refusal')}")
    dom = census["dominant_loss_stage"]
    if dom:
        lines.append(f"dominant loss stage: {dom['stage']} ({dom['k']}/{dom['n']})")
    else:
        lines.append("dominant loss stage: UNMEASURED (no stage had a denominator)")
    lines.append(f"cannot see: {', '.join(sorted(census['cannot_see']))}")
    return "\n".join(lines)


# ── where it lands ────────────────────────────────────────────────────────────

#: The census's home relative to the data root, used by `resolve_out_dir` as the
#: default and named so a reader that only wants to know WHERE the dated censuses
#: are resolves it through this module. #2597's eval instrument index is that
#: reader: it may only report an instrument at the location its own writer
#: declares, and until now `eval/funnel` existed only inside this function body.
#: A tuple of segments, not a joined Path, because it is relative to a root the
#: caller supplies.
FUNNEL_ARTIFACT_SUBDIR: tuple[str, ...] = ("eval", "funnel")


def resolve_out_dir(cli_value: str | None = None,
                    *, data_root: Path | None = None,
                    code_root: Path | None = None) -> Path:
    """The artifact's home: under the data root, and never inside the checkout.

    #2378's ruling on runtime data is a rule, so it is enforced here rather than
    left to the caller: a resolved path that sits inside the code tree raises
    `CensusWriteRefused`, and `main` exits 2 on it. `~/lloyd` is the running
    system; a census that wrote into it would be committed by the next round that
    happened to run `git add -A`.
    """
    base = Path(data_root if data_root else paths.DATA_ROOT)
    out = Path(cli_value) if cli_value else base.joinpath(*FUNNEL_ARTIFACT_SUBDIR)
    out = out.expanduser().resolve()
    forbidden = Path(code_root if code_root else paths.LLOYD_HOME).resolve()
    if out == forbidden or forbidden in out.parents:
        raise CensusWriteRefused(
            f"refusing to write the census inside the code tree ({forbidden}); "
            f"runtime data belongs under the data root")
    return out


def write_artifact(census: dict, out_dir: Path) -> Path:
    """Write the census as `funnel-<date>.json` and return the path.

    `artifact_path` is stamped in BEFORE the bytes are written, so the artifact
    names itself: a reader holding one file out of a month of them can tell which
    tree and which window it came from without the directory layout.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(census["measured_at"]).strftime("%Y-%m-%d")
    path = out_dir / f"funnel-{stamp}.json"
    census["artifact_path"] = str(path)
    # allow_nan=False: a stage that produced a nan would otherwise serialise as
    # the literal `NaN`, which is not JSON and breaks every later reader.
    text = json.dumps(census, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.funnel_census",
        description="Census the durable-artifact funnel by stage (#2507).")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"window in days ending now (default {DEFAULT_DAYS})")
    ap.add_argument("--out-dir", default=None,
                    help="where the dated artifact goes (default "
                         "<data root>/eval/funnel; a path inside the checkout is "
                         "refused)")
    ap.add_argument("--event-root", default=None,
                    help="directory of <session>.events.jsonl (default "
                         "app.paths.EVENT_LOGS_DIR)")
    ap.add_argument("--workers-db", default=None,
                    help="workers.db for the priced line (default "
                         "app.paths.WORKERS_DB)")
    ap.add_argument("--skills-root", action="append", default=None,
                    metavar="DIR", help="skill root to census; repeatable "
                                        "(default agent_mcp.skills.SKILLS_DIRS)")
    ap.add_argument("--uptake-dir", default=None, metavar="DIR",
                    help="uptake table store for the `benefited` stage (default "
                         "app.uptake.UPTAKE_ARTIFACT_DIR — the directory the "
                         "probe's own write_table outputs to, which is inside the "
                         "code tree, NOT under the data root)")
    args = ap.parse_args(argv)

    try:
        out_dir = resolve_out_dir(args.out_dir)
    except CensusWriteRefused as exc:
        print(f"funnel census REFUSED: {exc}")
        return 2
    census = build_census(args.days, event_root=Path(args.event_root) if args.event_root
                          else None, workers_db=Path(args.workers_db)
                          if args.workers_db else None,
                          skills_roots=[Path(p) for p in args.skills_root]
                          if args.skills_root else None,
                          uptake_dir=Path(args.uptake_dir) if args.uptake_dir
                          else None)
    path = write_artifact(census, out_dir)
    print(render_text(census))
    print(f"artifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
