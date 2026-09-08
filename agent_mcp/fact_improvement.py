#!/usr/bin/env python3
"""
Lloyd MCP Server: fact improvement — the feedback loop over the knowledge graph.

Backlog #376, Priority 1. Cognee ships an `improve` operation that learns from
feedback to refine memory quality; until now Lloyd had the *writers*
(`fact_invalidate`, `fact_resolve`) and no caller that knew when to use them.
`fact_resolve(auto_resolve=…)` had zero automated callers, and the one place
that claimed to learn from feedback — dream consolidation — is dead text (#463).

Why this is a module and not just a cron line over `fact_resolve`
----------------------------------------------------------------
The detector is not a verdict. `_detect_contradictions_sync` fires when two
facts of the same entity share >0.6 token overlap, which is *two facts phrased
alike*, not necessarily two facts that disagree — measured on `Lloyd`: 32,857
"contradictions" from 5,489 facts. That heuristic is exactly why
`fact_resolve`'s `auto_resolve` stopped defaulting to true. So a loop that
auto-resolved whatever the detector returned would be a fact-deletion machine
with extra steps.

Every action here therefore needs an independent reason *on top of* the
detector's pairing:

  confidence    the two sides disagree on confidence → `fact_resolve`'s case
  created_at    one was written ≥ MIN_STALE_GAP_DAYS after the other, so the
                older one is the one a later write superseded → expire it
  same day, same confidence → no basis. Reported, never acted on.

Writers are the existing tools, called as functions. This module never edits a
fact file: three writers of `expired_at` would be the same drift bug
`fact_profile` and the router already had (agent_mcp/facts.py:71-82), and
`fact_*` are the documented owners.

Dry-run by default. `apply=True` is opt-in, and each run writes a record under
`_pipeline/improvement/` with before/after active-fact counts so a run can be
audited or its reasoning re-read afterwards.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path

from app.paths import LLOYD_HOME, VAULT_FACTS_ROOT
from agent_mcp.facts import _detect_contradictions_sync, _fact_invalidate
from agent_mcp.retrieval import get_facts_sync as _get_facts_sync
from app.kg_store import StoreUnavailable, store as _store

logger = logging.getLogger("lloyd.improvement")

# Overridable module-level names: tests point these at a temp tree.
FACTS_ROOT = VAULT_FACTS_ROOT
CORRECTIONS_PATH = Path.home() / "obsidian" / "memory" / "corrections.md"
RECORD_DIR = LLOYD_HOME / "_pipeline" / "improvement"

# Entity dirs whose mtime is inside this window count as "a writer has been
# here recently, and that fresh claim is the one most likely to already be
# stale". Three days: the nightly extractor runs daily, so one cycle of slack.
DRIFT_WINDOW_DAYS = 3
# The minimum age gap before "written later" is evidence of "superseded".
# Same-day pairs are re-phrasings, not corrections — that is the class that
# produced the 32,857 false positives.
MIN_STALE_GAP_DAYS = 1.0
# Act only on pairs the detector flagged for *opposing terms* (working/broken,
# enabled/disabled, true/false…). The detector's other trigger is
# `_token_overlap > 0.6` alone, which labels two facts that say nearly the same
# thing a contradiction — on this tree that is the dominant mode, and picking
# one of a near-duplicate pair to expire is a dedupe decision made by whichever
# fact happened to carry a lower confidence number. Measured: acting on those
# pairs is what moved `fact_entity_recall` 0.35 -> 0.30, i.e. it deleted useful
# facts. Near-duplicate pairs are reported, never acted on.
REQUIRE_OPPOSING_TERMS = True
# Cap per entity: a god-node's contradiction list is noise, and expiring 20
# facts because a heuristic shrugged is not an improvement.
MAX_ACTIONS_PER_ENTITY = 5
# Cap per run across all entities.
MAX_ACTIONS_PER_RUN = 25
# Substring length used to aim `fact_invalidate` at one fact. The tool matches
# on a case-insensitive substring, so a 60-char prefix is specific in practice.
_SUBSTRING_LEN = 60

_SECTION_RE = re.compile(r"^#{1,4}\s+(.+)$", re.M)
# Entity-shaped token in a corrections heading: capitalized word-run that is
# also a known entity, checked against the store — a heading alone is a guess.
_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9][A-Za-z0-9._-]{1,}\b")
_DATE_IN_HEAD_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b|\b[A-Z][a-z]{2,8}\s+\d{1,2}\b")


# ── signal sources ───────────────────────────────────────────────────────────

def _known_entities() -> dict[str, str]:
    """lowercased entity name → canonical, from the entity registry.

    An empty map means the store is unreadable; callers must then decline to
    act rather than fall back to guessing from the heading's own capitalisation
    — that is how you end up expiring facts about a word.
    """
    try:
        return {name.lower(): name for name in _store().entities.all()}
    except Exception:  # noqa: BLE001 - a store that cannot answer must not become a guess
        return {}


def read_correction_signals(limit: int = 25) -> list[dict]:
    """Entities named in the user's own corrections log.

    `~/obsidian/memory/corrections.md` is a record of things Lloyd got wrong
    and was told so. It is consumed today only by the behaviour/prompt loops
    (nightly-reflection-signals, nightly-prompt-audit, nightly-behavior-test,
    scripts/autonomy/self_improve.py) — nothing routes it into *fact* quality.
    This is that route.

    An entry is credited to an entity only when a token in its heading is a
    registered entity name. Dates and prose are stripped first, so
    "2026-09-08 — TTS service status" yields `TTS` and nothing else does.
    """
    try:
        text = CORRECTIONS_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    known = _known_entities()
    if not known:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for match in _SECTION_RE.finditer(text):
        heading = _DATE_IN_HEAD_RE.sub(" ", match.group(1))
        for token in _TOKEN_RE.findall(heading):
            canonical = known.get(token.lower())
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            out.append({"entity": canonical, "source": "corrections",
                        "evidence": match.group(1).strip()[:200]})
            break
        if len(out) >= limit:
            break
    return out


def read_drift_signals(days: int = DRIFT_WINDOW_DAYS, limit: int = 50) -> list[dict]:
    """Entities with a fact file written inside `days`.

    This is the only automatic quality signal the store itself emits: a fact
    written this week is a claim about a world that has moved since. It is not
    a verdict either — it selects *where to look*, and the contradiction
    pairing plus the `created_at` ordering decides whether anything is stale.

    Fact-file mtimes, not `facts_idx.created_at`: the index records when a
    fact was *written into the graph*, and the whole tree was rebuilt on
    09-03, so every backfilled row carries a rebuild date. Directory mtimes
    would be wrong too — they only move when a file is added or removed, and
    the nightly extractor rewrites in place. The tree walk is 0.4 s over
    23,625 entity dirs on the live box.
    """
    cutoff = datetime.datetime.now().timestamp() - days * 86400
    try:
        entries = list(FACTS_ROOT.iterdir())
    except OSError:
        return []
    out: list[dict] = []
    for child in entries:
        try:
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            newest = max((p.stat().st_mtime for p in child.glob("*.md")), default=0)
            if newest < cutoff:
                continue
        except OSError:
            continue
        out.append({"entity": child.name, "source": "drift",
                    "evidence": f"fact file written {datetime.datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}"})
    out.sort(key=lambda s: s["entity"])
    return out[:limit]


def collect_signals(sources=("corrections", "drift"), days: int = DRIFT_WINDOW_DAYS,
                    limit: int = 40) -> list[dict]:
    """Union of the enabled sources, deduped by entity, corrections first.

    Corrections outrank drift because a user saying "that was wrong" is a
    better reason to look than a file being new.
    """
    wanted = set(sources)
    out: list[dict] = []
    seen: set[str] = set()
    if "corrections" in wanted:
        out.extend(read_correction_signals(limit=limit))
    if "drift" in wanted:
        out.extend(read_drift_signals(days=days, limit=limit))
    deduped = []
    for sig in out:
        if sig["entity"] in seen:
            continue
        seen.add(sig["entity"])
        deduped.append(sig)
        if len(deduped) >= limit:
            break
    return deduped


# ── the plan ─────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Whitespace/case/punctuation-collapsed fact text, for uniqueness counts."""
    return " ".join(str(text or "").lower().split())


def _iso(value) -> datetime.datetime | None:
    """Parse a fact timestamp. Values arrive as ISO strings or dates, and the
    tree holds both naive and offset-bearing stamps."""
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    try:
        parsed = datetime.datetime.fromisoformat(str(value).strip().rstrip("Z") + (
            "+00:00" if str(value).strip().endswith("Z") else ""))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def _loser_by_age(f1: dict, f2: dict) -> tuple[dict, dict, str] | None:
    """Return (loser, winner, reason) when write order is evidence."""
    t1, t2 = _iso(f1.get("created_at")), _iso(f2.get("created_at"))
    if not t1 or not t2:
        return None
    gap = abs((t2 - t1).total_seconds()) / 86400
    if gap < MIN_STALE_GAP_DAYS:
        return None
    loser, winner = (f1, f2) if t1 < t2 else (f2, f1)
    return loser, winner, (
        f"contradiction detector paired it with {winner.get('fact', '')[:70]!r}, "
        f"written {gap:.1f} days later; created_at order makes the older claim the superseded one")


def plan_entity(entity: str, max_actions: int = MAX_ACTIONS_PER_ENTITY) -> dict:
    """Decide what — if anything — to change about one entity's facts.

    Reads through the same `_get_facts_sync` the recall path uses, so the plan
    is made on the facts a query would actually have been answered with.
    """
    detection = _detect_contradictions_sync(entity)
    if detection.get("refused"):
        return {"entity": entity, "refused": True, "checked": detection.get("checked", 0),
                "contradictions": 0, "actions": [],
                "before_active": _active_count(entity),
                "skipped_reason": detection.get("hint", "entity too large to scan")}

    contradictions = detection.get("contradictions", [])
    if REQUIRE_OPPOSING_TERMS:
        actable = [c for c in contradictions
                   if str(c.get("reason", "")).startswith("opposing_terms")]
    else:
        actable = list(contradictions)
    actions: list[dict] = []
    planned_against: set[str] = set()
    for item in actable:
        if len(actions) >= max_actions:
            break
        f1 = item.get("fact1") or {}
        f2 = item.get("fact2") or {}
        if not f1.get("fact") or not f2.get("fact"):
            continue
        c1 = float(f1.get("confidence") or 0.5)
        c2 = float(f2.get("confidence") or 0.5)
        if c1 != c2:
            loser, winner = (f2, f1) if c1 > c2 else (f1, f2)
            kind = "confidence"
            reason = (f"contradiction detector paired it with "
                      f"{winner.get('fact', '')[:70]!r}; confidence "
                      f"{loser.get('confidence')} < {winner.get('confidence')}")
        else:
            ordered = _loser_by_age(f1, f2)
            if ordered is None:
                continue          # equal confidence, no age basis → leave it
            loser, winner, reason = ordered
            kind = "superseded"
        action = {"kind": kind, "entity": entity,
                  "category": loser.get("category"),
                  "loser_fact": loser.get("fact", ""), "loser_id": loser.get("id"),
                  "reason": reason}
        # Dedupe on the fact text, not the id: ids are per-file counters, so
        # `fact-001` in three category files is three different facts and one
        # condemned claim must not silence the other two.
        key = (action["category"], _normalise(action["loser_fact"]))
        if key in planned_against:
            continue
        planned_against.add(key)
        actions.append(action)

    return {"entity": entity, "refused": False, "checked": detection.get("checked", 0),
            "contradictions": len(contradictions),
            # Reported so the near-duplicate class stays visible: it is the
            # auto-capture noise this loop declines to delete, and the reason
            # the metric does not move.
            "near_duplicates": len(contradictions) - len(actable),
            "actions": actions, "before_active": _active_count(entity)}


# ── applying a plan (through the existing writers) ───────────────────────────

def _active_count(entity: str | None = None) -> int:
    """Active facts in the index. -1 when the store cannot answer, so a
    missing count is never mistaken for a zero."""
    try:
        return _store().facts_idx.count(entity=entity, active_only=True)
    except Exception:  # noqa: BLE001 - StoreUnavailable and everything the driver raises
        return -1


def _aim_substring(entity: str, action: dict) -> str | None:
    """A substring that identifies exactly one current fact, or None.

    `fact_invalidate` matches case-insensitively over a category file, so a
    prefix is only safe once counted. If the full fact text still matches more
    than one stored fact, the two are indistinguishable to the writer and the
    action is dropped — a plan may not take out a fact it did not condemn, and
    on this tree that is not hypothetical: fact ids are per-file counters
    (`Assistant` has 19 facts sharing id `fact-001`), which is the same
    collateral-damage shape that made `fact_resolve(auto_resolve=true)`
    invalidate 25 facts to change 2.
    """
    target = _normalise(action["loser_fact"])
    if not target:
        return None
    facts = _get_facts_sync(entity, action.get("category")).get("facts") or []
    lowered = [(f.get("fact") or "", (f.get("fact") or "")) for f in facts]
    for length in (_SUBSTRING_LEN, 120, len(action["loser_fact"])):
        probe = action["loser_fact"][:length]
        if not probe.strip():
            continue
        hits = [t for t, _ in lowered if probe.lower() in t.lower()]
        if len(hits) == 1:
            return probe
    return None


def apply_action(action: dict, now_iso: str) -> dict:
    """Expire one condemned fact through `fact_invalidate`.

    Both evidence classes go through the same writer on purpose. The
    confidence class is semantically `fact_resolve`'s — mark the weaker side
    `invalid_at` rather than `expired_at` — but `fact_resolve(auto_resolve=true)`
    selects losers by fact *id*, and ids repeat across an entity's category
    files, so it invalidates every fact sharing the loser's id: measured on the
    live `Assistant` entity, 2 planned actions became 25 invalidated facts
    (29 active → 4). Until ids are entity-unique, a scoped expire is the only
    writer this loop can trust with its blast radius. The `reason` string keeps
    the distinction, so a later pass can re-tag them.
    """
    aim = _aim_substring(action["entity"], action)
    if aim is None:
        return {"expired_count": 0, "skipped": "no unique match for the condemned fact"}
    return _fact_invalidate({
        "entity": action["entity"],
        "category": action.get("category"),
        "fact_substring": aim,
        "ended": now_iso,
        "reason": f"improve(#376) {action['kind']}: {action['reason']}",
    })


def _fact_entity_recall(limit: int = 20) -> float | None:
    """Score the live fact tree with the eval's own scorer, on production knobs.

    `eval/run_eval.py` is a script, not a package, so it is loaded by path. Two
    traps, both documented in that file and both hit here on the first try:

      * `run_eval()`'s **signature** defaults are the pre-#322 configuration
        (`graph_rerank=False`, `rerank_alpha=0.5`) — it is the *argparse*
        defaults that are imported from `agent_mcp.vault`. Calling
        `run_eval(queries)` directly measures a configuration nothing serves,
        which is the blind spot the 2026-09-03 review closed. So the production
        knobs are passed explicitly and cross-checked.
      * the corpus underneath must be readable. Measured rather than assumed:
        a failure to score returns None, not 0.0.

    Returns None when it could not be measured, so "did not run" can never be
    read as "measured zero".
    """
    try:
        import importlib.util

        import yaml
        from agent_mcp import vault

        script = LLOYD_HOME / "eval" / "run_eval.py"
        spec = importlib.util.spec_from_file_location("lloyd_run_eval", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        queries = (yaml.safe_load(
            (LLOYD_HOME / "eval" / "vault_recall_queries.yaml").read_text()) or {}).get("queries") or []
        knobs = {"expand_graph": True,
                 "graph_rerank": vault.RECALL_GRAPH_RERANK,
                 "rerank_alpha": vault.RECALL_RERANK_ALPHA,
                 "graph_top_k": vault.RECALL_GRAPH_TOP_K,
                 "graph_hops": vault.RECALL_GRAPH_HOPS}
        records = module.run_eval(queries[:limit], limit=limit, **knobs)
        summary = module.summarize(records)
        if summary["overall"]["errors"]:
            logger.warning("improve: %d eval queries errored; not reporting the metric",
                           summary["overall"]["errors"])
            return None
        return round(summary["overall"]["fact_entity_recall_avg"], 4)
    except Exception as exc:  # noqa: BLE001 - a metric that cannot run is not a zero
        logger.warning("improve: fact_entity_recall could not be measured: %s", exc)
        return None


# ── the operation ────────────────────────────────────────────────────────────

def run_improvement(apply: bool = False, sources=("corrections", "drift"),
                    entities=None, days: int = DRIFT_WINDOW_DAYS,
                    limit: int = 40, report_eval: bool = False,
                    eval_limit: int = 20, record: bool = True,
                    max_actions: int = MAX_ACTIONS_PER_RUN) -> dict:
    """One improvement pass. Dry-run unless `apply=True`.

    Reports `fact_entity_recall` when asked: #376's own acceptance bar is that
    a change which cannot move that metric is not this feature, so the number
    has to come out of the operation, not out of a human remembering to run
    the eval afterwards.
    """
    started = datetime.datetime.now(datetime.timezone.utc)
    now_iso = started.isoformat()
    before = _active_count()

    if entities:
        signals = [{"entity": e, "source": "explicit", "evidence": "named by caller"}
                   for e in entities]
    else:
        signals = collect_signals(sources=sources, days=days, limit=limit)

    planned = taken = 0
    per_entity: list[dict] = []
    budget = max_actions
    for signal in signals:
        entity = signal["entity"]
        try:
            plan = plan_entity(entity)
        except Exception as exc:  # noqa: BLE001 - one bad entity must not stop the run
            per_entity.append({"entity": entity, "error": str(exc)})
            continue
        entry = {"entity": entity, "signal": signal["source"],
                 "contradictions": plan["contradictions"],
                 "near_duplicates": plan.get("near_duplicates", 0),
                 "refused": plan["refused"],
                 "planned": len(plan["actions"]), "taken": 0, "actions": []}
        planned += len(plan["actions"])
        if apply:
            for action in plan["actions"]:
                if budget <= 0:
                    entry["stopped"] = "run action budget"
                    break
                result = apply_action(action, now_iso)
                changed = int(result.get("expired_count") or 0)
                # One action, at most one fact: `apply_action` only ever aims a
                # substring that matches a single stored fact. Anything larger
                # here means the writer over-reached, so the run reports it and
                # stops rather than continuing to delete.
                if changed > 1:
                    entry["actions"].append({
                        "kind": action["kind"], "loser_fact": action["loser_fact"][:90],
                        "reason": action["reason"], "applied": False, "changed": changed,
                        "error": f"writer expired {changed} facts for one action; run halted for this entity"})
                    entry["stopped"] = "blast radius exceeded one fact per action"
                    budget = 0
                    break
                taken += changed
                budget -= 1
                entry["actions"].append({
                    "kind": action["kind"], "loser_fact": action["loser_fact"][:90],
                    "reason": action["reason"], "applied": changed > 0,
                    "changed": changed, "error": result.get("error"),
                    "skipped": result.get("skipped"),
                })
            entry["taken"] = sum(a["changed"] for a in entry["actions"] if a.get("applied"))
        else:
            # A plan-mode pass that reports only a count is not a plan. List
            # what it would do, with the reason, so an operator can read the
            # judgement before trusting it with `apply`.
            entry["actions"] = [{"kind": a["kind"], "loser_fact": a["loser_fact"][:90],
                                 "reason": a["reason"], "planned": True, "applied": False}
                                for a in plan["actions"]]
        entry["before_active"] = plan.get("before_active", -1)
        entry["after_active"] = _active_count(entity) if apply else entry["before_active"]
        per_entity.append(entry)

    after = _active_count()
    record_obj = {
        "ran_at": now_iso,
        "apply": bool(apply),
        "sources": list(sources) if not entities else ["explicit"],
        "days": days,
        "signals": len(signals),
        "entities": [s["entity"] for s in signals],
        "actions_planned": planned,
        "actions_taken": taken,
        "before_active": before,
        "after_active": after,
        "delta_active": (after - before) if after >= 0 and before >= 0 else None,
        "per_entity": per_entity,
        "fact_entity_recall": _fact_entity_recall(eval_limit) if report_eval else None,
        "corrections_path": str(CORRECTIONS_PATH),
    }
    if record:
        try:
            RECORD_DIR.mkdir(parents=True, exist_ok=True)
            path = RECORD_DIR / f"{started:%Y%m%d-%H%M%S}-{'apply' if apply else 'dryrun'}.json"
            path.write_text(json.dumps(record_obj, indent=2, default=str), encoding="utf-8")
            record_obj["record_path"] = str(path)
        except OSError as exc:
            record_obj["record_error"] = str(exc)

    logger.info(
        "improve(%s): signals=%d planned=%d taken=%d active_facts %d -> %d%s",
        "apply" if apply else "dry-run", len(signals), planned, taken, before, after,
        "" if record_obj.get("fact_entity_recall") is None
        else f" fact_entity_recall={record_obj['fact_entity_recall']}")
    return record_obj
