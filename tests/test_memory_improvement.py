"""Backlog #376 — the `improve` feedback loop and the unified memory surface.

Two things are pinned here, and both were absent before this change:

1. **An `improve`-equivalent.** A consumer that reads a *real* feedback
   signal, treats the existing contradiction detector as *evidence* rather
   than a verdict, and acts only through the two existing fact writers
   (`fact_resolve(auto_resolve=…)` / `fact_invalidate`) while recording
   before/after active-fact counts. It is dry-run by default.

   The detector alone is not a verdict: `_token_overlap > 0.6` fires on two
   facts phrased alike. That is exactly why `fact_resolve`'s `auto_resolve`
   stopped defaulting to true, and why every action this loop takes must
   carry an independent reason — the user contested the entity, or the
   `created_at` ordering says which fact is the current one.

2. **A unified entry point.** `remember` / `recall` / `forget` over the
   19 memory-family tools, so there is one verb per cognitive operation and
   the long-tail tools stay available as the escape hatch.

Run: .venvs/lloyd/bin/python -m pytest tests/test_memory_improvement.py
"""
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import fact_improvement as fi          # noqa: E402
from agent_mcp import memory_ops                       # noqa: E402
from app import kg_store                               # noqa: E402


# ── fixture: a fact tree + store + vault, all in tmp_path ────────────────────

@pytest.fixture
def world(tmp_path, monkeypatch):
    """Temp facts tree, KG store and vault root, wired into every reader."""
    import agent_mcp._shared as shared
    from agent_mcp import retrieval

    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    vault_root = tmp_path / "vault"
    (vault_root / "memory").mkdir(parents=True)

    monkeypatch.setattr(shared, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts_root)
    from agent_mcp import facts as facts_mod
    monkeypatch.setattr(facts_mod, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(fi, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(fi, "CORRECTIONS_PATH", vault_root / "memory" / "corrections.md")
    monkeypatch.setattr(fi, "RECORD_DIR", tmp_path / "records")
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None

    st = kg_store.configure(tmp_path / "kg.sqlite")
    yield facts_root, st, vault_root
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None


def _write_facts(root, entity, category, facts):
    """One fact file with explicit per-fact fields (mirrors the real shape)."""
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    prepared = []
    for i, f in enumerate(facts, start=1):
        prepared.append({
            "fact": f["fact"], "confidence": f.get("confidence", 0.9),
            "category": category, "id": f"{category[:4]}-{i:03d}",
            "created_at": f["created_at"], "valid_at": f["created_at"],
            "invalid_at": None, "expired_at": None, "provenance": "STATED",
            "source_doc": None,
        })
    fm = {"type": "facts", "entity": entity, "category": category, "facts": prepared}
    (d / f"{entity}-{category}.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n", encoding="utf-8")


def _active(st, entity=None):
    return st.facts_idx.count(entity=entity, active_only=True)


def _reindex(st, root):
    st.facts_idx.reindex(root=root)


def _days_ago(n, hours=0):
    return (datetime.now(timezone.utc) - timedelta(days=n, hours=hours)).isoformat()


# ── 1. signals: a real source, not an invented one ───────────────────────────

def test_correction_signal_names_the_entity_the_user_contested(world):
    """`memory/corrections.md` entries are the user's own corrections; the
    entity named in the entry is the one worth re-checking."""
    facts_root, st, vault_root = world
    # An entity with no facts has nothing to improve, and a heading that names
    # an unknown word must not become a fact edit — so TTS has to exist first.
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    (vault_root / "memory" / "corrections.md").write_text(
        "# Corrections Log\n\n"
        "## 2026-09-08 07:00 PDT — TTS service status\n"
        "**Correction:** TTS is fixed; the built-in voices were returning 500s.\n\n"
        "## 2026-09-07 09:00 PDT — plain prose with no entity\n"
        "**Correction:** nothing entity-shaped here.\n",
        encoding="utf-8")
    found = fi.read_correction_signals()
    assert [s["entity"] for s in found] == ["TTS"], found
    assert found[0]["source"] == "corrections"


def test_drift_signal_finds_the_tree_a_writer_touched_today(world):
    """A fact file written in the last N days is a claim that may already be
    stale — it is the only automatic feedback source the store itself offers."""
    import os
    facts_root, st, _ = world
    _write_facts(facts_root, "FRESH", "state",
                 [{"fact": "FRESH is running on the box.", "created_at": _days_ago(40)},
                  {"fact": "FRESH runs two workers.", "created_at": _days_ago(2)}])
    _write_facts(facts_root, "STALE", "state",
                 [{"fact": "STALE is not running.", "created_at": _days_ago(60)}])
    _reindex(st, facts_root)
    old = datetime.now() - timedelta(days=30)
    stale_file = facts_root / "STALE" / "STALE-state.md"
    os.utime(stale_file, (old.timestamp(), old.timestamp()))
    os.utime(facts_root / "STALE", (old.timestamp(), old.timestamp()))
    found = fi.read_drift_signals(days=3)
    assert [s["entity"] for s in found] == ["FRESH"], found
    assert found[0]["source"] == "drift"


def test_collect_signals_dedupes_by_entity(world):
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    (vault_root / "memory" / "corrections.md").write_text(
        "## TTS regression\n**Correction:** TTS broke again.\n", encoding="utf-8")
    signals = fi.collect_signals(sources=("corrections", "drift"))
    assert sum(1 for s in signals if s["entity"] == "TTS") == 1


# ── 2. the loop: evidence + reason, dry-run by default ───────────────────────

def test_equal_confidence_contradiction_needs_a_time_order_reason(world):
    """Detector says "these two disagree", confidences tie → the tie is broken
    by created_at, and only when the loser is clearly older."""
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30), "confidence": 0.9},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
         "created_at": _days_ago(2), "confidence": 0.9},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("TTS")
    assert plan["contradictions"] >= 1, plan
    assert len(plan["actions"]) == 1
    action = plan["actions"][0]
    assert action["kind"] == "superseded"
    assert action["loser_fact"].startswith("TTS built-in voices are working")
    assert "created" in action["reason"]           # states *why* this side lost
    assert plan["before_active"] == 2


def test_unequal_confidence_is_planned_as_the_confidence_class(world):
    """A low-confidence claim contradicted by a high-confidence one is a
    separate evidence class: the loser is the weak one regardless of which was
    written first."""
    facts_root, st, _ = world
    _write_facts(facts_root, "ASR", "state", [
        {"fact": "ASR wake word detection is working end to end.",
         "created_at": _days_ago(2), "confidence": 0.95},
        {"fact": "ASR wake word detection is broken.",
         "created_at": _days_ago(4), "confidence": 0.4},
    ])
    _reindex(st, facts_root)
    action = fi.plan_entity("ASR")["actions"][0]
    assert action["kind"] == "confidence"
    assert action["loser_fact"].startswith("ASR wake word detection is broken")


def test_one_action_cannot_touch_a_fact_it_did_not_condemn(world):
    """The collateral-damage class this loop exists next to: fact ids are
    per-file counters, so `fact-001` names several facts in one entity.
    `fact_resolve(auto_resolve=true)` selects losers by id and invalidated 25
    facts to change 2 on the live `Assistant` entity. An improve action may
    expire only the fact it names."""
    facts_root, st, _ = world
    loser = "Lloyd built-in voices are working and returning 200 OK."
    _write_facts(facts_root, "Lloyd", "state", [
        {"fact": loser, "created_at": _days_ago(30)},
        {"fact": "Lloyd built-in voices are broken and returning 500 errors.",
         "created_at": _days_ago(2)},
    ])
    _write_facts(facts_root, "Lloyd", "usage", [
        {"fact": "Lloyd is used for nightly reflection runs.", "created_at": _days_ago(20)}])
    _write_facts(facts_root, "Lloyd", "preference", [
        {"fact": "Lloyd is preferred over the older assistant.", "created_at": _days_ago(20)}])
    _reindex(st, facts_root)
    # Both files carry a fact with id `state-001`/`usage-001` etc. — the point is
    # the plan condemns exactly one claim and exactly one fact must go.
    assert _active(st, "Lloyd") == 4
    rec = fi.run_improvement(apply=True, entities=["Lloyd"])
    assert rec["actions_taken"] == 1, rec["per_entity"]
    assert _active(st, "Lloyd") == 3
    survivors = [f["fact"] for f in
                 fi._get_facts_sync("Lloyd").get("facts", [])]
    assert loser not in survivors
    assert any("nightly reflection" in f for f in survivors), survivors
    assert any("broken and returning 500" in f for f in survivors), survivors
    assert any("preferred over the older" in f for f in survivors), survivors


def test_same_day_equal_confidence_pair_is_left_alone(world):
    """No basis to pick a winner: same confidence, written an hour apart. This
    is the false-positive class that made `auto_resolve` default to false."""
    facts_root, st, _ = world
    _write_facts(facts_root, "QMD", "state", [
        {"fact": "QMD index is current and queryable.", "created_at": _days_ago(1, 2)},
        {"fact": "QMD index is stale and not queryable.", "created_at": _days_ago(1, 1)},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("QMD")
    assert plan["contradictions"] >= 1
    assert plan["actions"] == []


def test_a_near_duplicate_pair_is_reported_not_deleted(world):
    """The detector's second trigger is token overlap alone — two facts that
    say nearly the same thing. Acting on those is where an `improve` loop eats
    useful facts (measured: 0.35 -> 0.30), so the pair is reported and left
    alone even when the confidences differ."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Transcripts", "usage", [
        {"fact": "Transcripts were extracted from the video with the VTT parser.",
         "created_at": _days_ago(20), "confidence": 0.9},
        {"fact": "Transcripts were extracted from the video with the parser.",
         "created_at": _days_ago(19), "confidence": 0.6},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Transcripts")
    assert plan["contradictions"] >= 1, plan
    assert plan["near_duplicates"] >= 1
    assert plan["actions"] == []
    rec = fi.run_improvement(apply=True, entities=["Transcripts"])
    assert rec["actions_taken"] == 0
    assert _active(st) == 2


def test_godnode_entity_is_refused_not_rescanned(world):
    """The detector refuses above FACT_GODNODE_THRESHOLD; the loop inherits the
    refusal instead of paying O(n²) for noise."""
    facts_root, st, _ = world
    from agent_mcp.retrieval import FACT_GODNODE_THRESHOLD
    facts = [{"fact": f"QMD detail number {i} differs from {i + 1}.",
              "created_at": _days_ago(20)} for i in range(FACT_GODNODE_THRESHOLD + 1)]
    _write_facts(facts_root, "QMD", "state", facts)
    _reindex(st, facts_root)
    plan = fi.plan_entity("QMD")
    assert plan.get("refused") is True
    assert plan["actions"] == []


def test_run_is_dry_run_by_default_and_acts_on_request(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
         "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    _reindex(st, facts_root)

    dry = fi.run_improvement(sources=("drift",), days=3)
    assert dry["apply"] is False
    assert dry["actions_planned"] == 1 and dry["actions_taken"] == 0
    assert _active(st) == 2, "dry run must not touch a single fact"
    text = (facts_root / "TTS" / "TTS-state.md").read_text(encoding="utf-8")
    assert "expired_at: null" in text
    # A plan-mode pass that reports only a count is not a plan: the list of
    # what it would do, with the reason, is the whole point of running it.
    listed = dry["per_entity"][0]["actions"]
    assert len(listed) == 1 and listed[0]["planned"] is True and listed[0]["applied"] is False
    assert listed[0]["loser_fact"].startswith("TTS built-in voices are working")
    assert "created" in listed[0]["reason"]

    wet = fi.run_improvement(apply=True, sources=("drift",), days=3)
    assert wet["apply"] is True
    assert wet["actions_taken"] == 1
    assert wet["before_active"] == 2 and wet["after_active"] == 1


def test_run_record_carries_before_after_counts_and_is_persisted(world):
    facts_root, st, tmp_vault = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
         "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    rec = fi.run_improvement(apply=True, sources=("drift",), days=3)
    for key in ("before_active", "after_active", "delta_active", "signals",
                "actions_planned", "actions_taken", "per_entity", "fact_entity_recall"):
        assert key in rec, key
    assert rec["delta_active"] == rec["after_active"] - rec["before_active"] == -1
    written = list((fi.RECORD_DIR).glob("*.json"))
    assert written, "the run must be recorded, not just returned"
    on_disk = json.loads(written[0].read_text(encoding="utf-8"))
    assert on_disk["before_active"] == 2 and on_disk["after_active"] == 1


def test_run_reports_the_metric_it_claims_to_move(world, monkeypatch):
    """The acceptance bar for #376: a change that cannot move
    `fact_entity_recall` is not this feature — so the run carries the number."""
    facts_root, st, _ = world
    seen = {}

    def _fake_report(limit):
        seen["limit"] = limit
        return 0.5

    monkeypatch.setattr(fi, "_fact_entity_recall", _fake_report)
    rec = fi.run_improvement(sources=("drift",), days=3, report_eval=True, eval_limit=7)
    assert rec["fact_entity_recall"] == 0.5
    assert seen["limit"] == 7
    skipped = fi.run_improvement(sources=("drift",), days=3)
    assert skipped.get("fact_entity_recall") is None, "off by default"


def test_run_with_no_signals_changes_nothing(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "OLD", "state",
                 [{"fact": "OLD is disabled.", "created_at": _days_ago(90)}])
    _reindex(st, facts_root)
    import os
    old = datetime.now() - timedelta(days=40)
    target = facts_root / "OLD" / "OLD-state.md"
    os.utime(target, (old.timestamp(), old.timestamp()))
    os.utime(facts_root / "OLD", (old.timestamp(), old.timestamp()))
    rec = fi.run_improvement(apply=True, sources=("drift", "corrections"), days=2)
    assert rec["signals"] == 0 and rec["entities"] == []
    assert rec["actions_taken"] == 0
    assert _active(st) == 1


# ── 3. unified surface: remember / recall / forget ───────────────────────────

def _tool_names(mod):
    return [t.name for t in asyncio.run(mod.list_tools())]


def test_unified_verbs_are_registered_tools():
    """The acceptance grep: `def remember|def recall|def forget` must hit."""
    import inspect
    src = inspect.getsource(memory_ops)
    for verb in ("def remember", "def recall", "def forget"):
        assert verb in src, verb
    assert set(_tool_names(memory_ops)) == {"remember", "recall", "forget", "improve"}


def test_improve_is_registered_on_the_fact_side():
    assert "improve" in _tool_names(memory_ops)


def test_unified_verbs_are_dispatchable_and_unique():
    from agent_mcp import main as M
    names = [t.name for t in asyncio.run(M.list_tools())]
    assert len(names) == len(set(names)), "duplicate tool registration"
    for verb in ("remember", "recall", "forget", "improve"):
        assert verb in names, verb
    assert set(M._dispatch) >= {"remember", "recall", "forget", "improve"}


def test_remember_adds_a_fact_once(world):
    facts_root, st, _ = world
    res = memory_ops.remember({"entity": "Bernie", "category": "state",
                               "fact": "Bernie uses a mecanum drive."})
    assert res.get("success") is True, res
    assert _active(st, "Bernie") == 1
    again = memory_ops.remember({"entity": "Bernie", "category": "state",
                                 "fact": "Bernie uses a mecanum drive."})
    assert again.get("skipped") is True
    assert _active(st, "Bernie") == 1, "remember must not duplicate a fact"


def test_remember_requires_an_entity_and_a_fact(world):
    res = memory_ops.remember({"entity": "Bernie", "category": "state"})
    assert res.get("error"), res


def test_recall_returns_documents_and_facts(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "Bernie", "state",
                 [{"fact": "Bernie uses a mecanum drive.", "created_at": _days_ago(3)}])
    _reindex(st, facts_root)
    res = memory_ops.recall({"query": "Bernie drive", "limit": 5, "grep_code": False})
    assert "documents" in res and "facts" in res
    assert any("mecanum" in f.get("fact", "") for f in res["facts"]), res["facts"]


def test_recall_refuses_an_empty_query(world):
    res = memory_ops.recall({"query": "   "})
    assert res.get("error")


def test_forget_expires_only_facts_that_match(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "Bernie", "state", [
        {"fact": "Bernie uses a mecanum drive.", "created_at": _days_ago(20)},
        {"fact": "Bernie has a 5-lb Olympic plate mount.", "created_at": _days_ago(20)},
    ])
    _reindex(st, facts_root)
    res = memory_ops.forget({"entity": "Bernie", "match": "mecanum"})
    assert res.get("expired_count") == 1, res
    assert _active(st, "Bernie") == 1


def test_forget_refuses_to_blank_an_entity(world):
    """A bare `forget(entity=…)` is a blanket delete over every fact the entity
    has. Refuse it — the same reason `fact_resolve` stopped defaulting to
    auto_resolve."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Bernie", "state",
                 [{"fact": "Bernie uses a mecanum drive.", "created_at": _days_ago(20)}])
    _reindex(st, facts_root)
    res = memory_ops.forget({"entity": "Bernie"})
    assert res.get("error"), res
    assert _active(st, "Bernie") == 1


def test_improve_tool_defaults_to_dry_run(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
         "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    res = memory_ops.improve({"sources": ["drift"], "days": 3})
    assert res["apply"] is False and res["actions_taken"] == 0
    assert _active(st) == 2


def test_annotation_tables_classify_the_new_verbs():
    """An unclassified actuator is a plan-mode hole; a mislabelled one is a
    badge lie. Both are pinned by the existing suite, so name them here."""
    from agent_mcp import annotations as A
    assert "recall" in A.READ_ONLY
    for verb in ("remember", "forget", "improve"):
        assert verb not in A.READ_ONLY, verb
    assert {"forget", "improve"} <= A.DESTRUCTIVE
    assert A.DESTRUCTIVE & A.READ_ONLY == frozenset()


def test_new_tools_carry_descriptions_and_documented_parameters():
    """Same hygiene bar the whole surface is held to (test_mcp_layer)."""
    thin, undocumented = [], []
    for tool in asyncio.run(memory_ops.list_tools()):
        if len(tool.description or "") < 60:
            thin.append(tool.name)
        for pname, spec in ((tool.input_schema or {}).get("properties") or {}).items():
            if not (spec.get("description") or "").strip():
                undocumented.append(f"{tool.name}.{pname}")
    assert thin == []
    assert undocumented == []
