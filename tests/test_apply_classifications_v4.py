"""#1246 clause 4: apply-classifications-v4 does not re-type a pair whose
verdict is already live.

The extractor used to mint a `mentions` row beside an active classifier-typed
edge on the same (source, target) (the guard is `_index_and_link`'s, pinned in
`tests/test_fact_extractor.py`), and that row put the pair back through
`retype`: the live `uses` was expired and a `uses` inserted again, 3-29% of
each nightly apply's output. Such a record is now counted as `already_typed`
and gets no retype in either mode. Under `--apply` the redundant `mentions`
row is retired on its own, so the pair converges to the one active relation
`retype` guarantees instead of being counted again on every run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.kg_store import KGStore  # noqa: E402


def _load():
    path = ROOT / "scripts" / "memory" / "apply-classifications-v4.py"
    spec = importlib.util.spec_from_file_location("apply_classifications_v4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


apply_mod = _load()


@pytest.fixture
def store(tmp_path):
    """A tmp store holding one pair the classifier already typed (with the
    extractor's redundant mentions row beside it) and one bare mentions pair."""
    db = tmp_path / "kg.sqlite"
    st = KGStore(db)
    typed_id = st.edges.add({"source": "Lloyd", "target": "vLLM", "type": "uses",
                             "confidence": 0.9, "provenance": "EXTRACTED_CLASSIFIER_V4"},
                            origin="classifier")
    redundant_id = st.edges.add({"source": "Lloyd", "target": "vLLM", "type": "mentions",
                                 "confidence": 0.8, "provenance": "EXTRACTED"},
                                origin="extractor")
    bare_id = st.edges.add({"source": "Lloyd", "target": "Isaac Lab", "type": "mentions",
                            "confidence": 0.8, "provenance": "EXTRACTED"},
                           origin="extractor")
    st.close()
    classified = tmp_path / "classified"
    classified.mkdir()
    rows = [
        {"source": "Lloyd", "target": "vLLM", "new_type": "uses", "confidence": 0.95,
         "classified_at": "2026-09-24T00:00:00+00:00", "reason": "already live"},
        {"source": "Lloyd", "target": "Isaac Lab", "new_type": "uses", "confidence": 0.95,
         "classified_at": "2026-09-24T00:00:00+00:00", "reason": "real upgrade"},
    ]
    (classified / "classified-v4-batch.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return {"db": db, "classified": classified, "typed_id": typed_id,
            "redundant_id": redundant_id, "bare_id": bare_id}


def _run(monkeypatch, store, *flags):
    monkeypatch.setattr(sys, "argv", ["apply-classifications-v4.py",
                                      "--classified-dir", str(store["classified"]),
                                      "--db", str(store["db"]), *flags])
    return apply_mod.main()


def _stat(out, name):
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == name:
            return int(parts[1])
    raise AssertionError(f"no `{name}` row in the plan:\n{out}")


def test_dry_run_reports_the_already_typed_pair_and_writes_nothing(
        monkeypatch, store, capsys):
    assert _run(monkeypatch, store, "--dry-run") == 0
    out = capsys.readouterr().out
    assert _stat(out, "already_typed") == 1, out
    assert _stat(out, "upgrades") == 1, "the bare pair is still a real upgrade"

    st = KGStore(store["db"])
    on_pair = {(e["id"], e["type"]) for e in st.edges.active(source="Lloyd", target="vLLM")}
    assert on_pair == {(store["typed_id"], "uses"), (store["redundant_id"], "mentions")}, (
        "a dry run wrote something")
    assert st.edges.by_id(store["bare_id"])["expired_at"] is None
    st.close()


def test_apply_keeps_the_live_verdict_and_retires_only_the_redundant_row(
        monkeypatch, store, capsys):
    """Against the pre-fix tree the typed `uses` is expired with reason
    `... pair re-typed as uses` and a new `uses` row takes its id's place —
    the churn the item measured. Now the typed edge is untouched, the
    redundant mentions row is retired with a reason that says why, and the
    bare pair is upgraded as before."""
    assert _run(monkeypatch, store, "--apply") == 0
    out = capsys.readouterr().out
    assert _stat(out, "already_typed") == 1, out

    st = KGStore(store["db"])
    typed = st.edges.by_id(store["typed_id"])
    assert typed["expired_at"] is None and typed["type"] == "uses"
    on_pair = st.edges.active(source="Lloyd", target="vLLM")
    assert [e["id"] for e in on_pair] == [store["typed_id"]], on_pair
    redundant = st.edges.by_id(store["redundant_id"])
    assert redundant["expired_at"] is not None
    assert "already typed as uses" in redundant["expired_reason"], redundant
    assert "re-typed" not in (typed.get("expired_reason") or "")

    bare = st.edges.by_id(store["bare_id"])
    assert bare["expired_at"] is not None, "the bare pair still gets its upgrade"
    upgraded = st.edges.active(source="Lloyd", target="Isaac Lab")
    assert [e["type"] for e in upgraded] == ["uses"]
    assert upgraded[0]["superseded_edge_id"] == store["bare_id"]
    st.close()


# ── #1453: a pair that carries a human or inferred edge is not this pass's to re-type ──
#
# The v4 guard rail ("human-authored `STATED`, `INFERRED` and already-classifier-
# judged edges are never re-typed by this pass") lived only in the candidate
# filters — `ELIGIBLE_PROVENANCES` and the `type == "mentions"` selection — while
# `EdgeSet.retype` expires *every* other active edge on the pair whatever its
# provenance (`app/kg_store.py`, `pair re-typed as`). Nothing asserted the rail on
# the write path, so one `fact_relate` landing a typed edge on a pair that still
# holds an un-judged `mentions` row was enough: 29,299 pairs sat in that state on
# 2026-09-24, with 0 of them carrying a second active edge yet (live damage zero,
# exposure one write away).
#
# The fix is a plan lane, not a change to `retype`: #925 forbids relaxing the
# one-active-relation-per-pair invariant, and
# `tests/test_kg_store.py::test_retype_keeps_history_and_collapses_the_pair`
# asserts that a retype *does* expire a co-resident `STATED` stray edge. So the
# pass now refuses to issue the retype instead of softening the sweep.

@pytest.fixture
def guarded_store(tmp_path):
    """A tmp store with one pair per lane the v4 apply distinguishes.

    `Lloyd → Obsidian` holds an eligible `EXTRACTED` `mentions` row AND a
    hand-authored `STATED` `uses` row, and the classifier's verdict for that
    pair is `uses` — the violation in its cleanest form: before #1453 the apply
    re-typed the mentions row, the collateral sweep expired the human row with
    reason `v4 reclassified mentions → uses: pair re-typed as uses`, and the
    relation on the pair did not change at all. `Lloyd → Home Assistant` is the
    same shape with an `INFERRED` co-resident. `vLLM → CUDA` carries a prior
    classifier verdict of a *different* type (a re-judgment, which the pass
    still supersedes) and `Lloyd → vLLM` one of the *same* type, so the
    classifier's own lanes stay pinned beside the new one.
    """
    db = tmp_path / "kg.sqlite"
    st = KGStore(db)
    ids: dict[str, int] = {}
    ids["ob_mentions"] = st.edges.add(
        {"source": "Lloyd", "target": "Obsidian", "type": "mentions",
         "confidence": 0.8, "provenance": "EXTRACTED"}, origin="extractor")
    ids["ob_stated"] = st.edges.add(
        {"source": "Lloyd", "target": "Obsidian", "type": "uses",
         "confidence": 0.99, "provenance": "STATED"}, origin="stated")
    ids["ha_mentions"] = st.edges.add(
        {"source": "Lloyd", "target": "Home Assistant", "type": "mentions",
         "confidence": 0.8, "provenance": "EXTRACTED"}, origin="extractor")
    ids["ha_inferred"] = st.edges.add(
        {"source": "Lloyd", "target": "Home Assistant", "type": "depends_on",
         "confidence": 0.7, "provenance": "INFERRED"}, origin="inferred")
    ids["bare"] = st.edges.add(
        {"source": "Lloyd", "target": "Isaac Lab", "type": "mentions",
         "confidence": 0.8, "provenance": "EXTRACTED"}, origin="extractor")
    ids["cuda_mentions"] = st.edges.add(
        {"source": "vLLM", "target": "CUDA", "type": "mentions",
         "confidence": 0.8, "provenance": "EXTRACTED"}, origin="extractor")
    ids["cuda_classifier"] = st.edges.add(
        {"source": "vLLM", "target": "CUDA", "type": "uses",
         "confidence": 0.9, "provenance": "EXTRACTED_CLASSIFIER_V4"}, origin="classifier")
    ids["vllm_mentions"] = st.edges.add(
        {"source": "Lloyd", "target": "vLLM", "type": "mentions",
         "confidence": 0.8, "provenance": "EXTRACTED"}, origin="extractor")
    ids["vllm_classifier"] = st.edges.add(
        {"source": "Lloyd", "target": "vLLM", "type": "uses",
         "confidence": 0.9, "provenance": "EXTRACTED_CLASSIFIER_V4"}, origin="classifier")
    st.close()

    classified = tmp_path / "classified"
    classified.mkdir()
    rows = [
        {"source": "Lloyd", "target": "Obsidian", "new_type": "uses",
         "confidence": 0.95, "classified_at": "2026-09-24T00:00:00+00:00",
         "reason": "same type as the human edge"},
        {"source": "Lloyd", "target": "Home Assistant", "new_type": "depends_on",
         "confidence": 0.95, "classified_at": "2026-09-24T00:00:00+00:00",
         "reason": "same type as the inferred edge"},
        {"source": "Lloyd", "target": "Isaac Lab", "new_type": "uses",
         "confidence": 0.95, "classified_at": "2026-09-24T00:00:00+00:00",
         "reason": "real upgrade"},
        {"source": "vLLM", "target": "CUDA", "new_type": "depends_on",
         "confidence": 0.95, "classified_at": "2026-09-24T00:00:00+00:00",
         "reason": "classifier re-judges its own earlier verb"},
        {"source": "Lloyd", "target": "vLLM", "new_type": "uses",
         "confidence": 0.95, "classified_at": "2026-09-24T00:00:00+00:00",
         "reason": "verdict already live"},
    ]
    (classified / "classified-v4-batch.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return {"db": db, "classified": classified, "ids": ids}


def test_a_pair_holding_a_stated_or_inferred_edge_is_its_own_plan_lane(
        monkeypatch, guarded_store, capsys):
    """The lane exists and is counted; nothing else's count moves.

    Against the pre-fix tree `protected_pair` is absent from the plan entirely
    and the two guarded pairs are booked as `upgrades` — 4 instead of 2 —
    because the only co-resident index the planner reads
    (`_build_live_typed_index`) filters on `provenance.startswith(
    "EXTRACTED_CLASSIFIER")`, so a `STATED` or `INFERRED` row is invisible to
    it and the pair falls through to `retype`.
    """
    assert _run(monkeypatch, guarded_store, "--dry-run") == 0
    out = capsys.readouterr().out
    assert _stat(out, "protected_pair") == 2, out
    assert _stat(out, "upgrades") == 2, (
        "only the bare pair and the classifier's own re-judgment upgrade")
    assert _stat(out, "already_typed") == 1, out
    assert _stat(out, "total_records") == 5, out
    # Existing lane names are unchanged, so the nightly report still parses.
    for key in ("below_threshold", "still_mentions", "no_eligible_edge",
                "duplicate_pair"):
        assert _stat(out, key) == 0, out

    st = KGStore(guarded_store["db"])
    assert st.edges.by_id(guarded_store["ids"]["ob_stated"])["expired_at"] is None
    assert st.edges.by_id(guarded_store["ids"]["bare"])["expired_at"] is None, (
        "a dry run wrote something")
    st.close()


def test_apply_leaves_the_stated_and_inferred_co_residents_active(
        monkeypatch, guarded_store, capsys):
    """#1453's acceptance check: no retype for a guarded pair, both rows live.

    The pre-fix run expires `ob_stated` and `ha_inferred` with a
    `pair re-typed as` reason and leaves a single classifier row on each pair;
    the human/inferred statement is gone and the relation is unchanged.
    """
    assert _run(monkeypatch, guarded_store, "--apply") == 0
    out = capsys.readouterr().out
    assert _stat(out, "protected_pair") == 2, out

    st = KGStore(guarded_store["db"])
    ids = guarded_store["ids"]
    for pair, held, mentions in (("Obsidian", ids["ob_stated"], ids["ob_mentions"]),
                                 ("Home Assistant", ids["ha_inferred"], ids["ha_mentions"])):
        human = st.edges.by_id(held)
        assert human["expired_at"] is None, f"{pair}: the human row was expelled"
        row = st.edges.by_id(mentions)
        assert row["expired_at"] is None, f"{pair}: the mentions row was re-typed"
        assert "re-typed" not in (human.get("expired_reason") or ""), human
        assert "re-typed" not in (row.get("expired_reason") or ""), row
        on_pair = sorted(e["id"] for e in st.edges.active(source="Lloyd", target=pair))
        assert on_pair == sorted([held, mentions]), (
            f"{pair}: expected exactly the two pre-existing rows, got {on_pair}")
    st.close()


def test_apply_still_upgrades_a_bare_mentions_pair(
        monkeypatch, guarded_store, capsys):
    """The guard is a carve-out, not a freeze: a pair whose only active edge
    is an eligible `mentions` row is upgraded as before, counted in `upgrades`,
    and the new row points back at the expired one."""
    assert _run(monkeypatch, guarded_store, "--apply") == 0
    assert _stat(capsys.readouterr().out, "upgrades") == 2, "both real upgrades land"

    st = KGStore(guarded_store["db"])
    bare = st.edges.by_id(guarded_store["ids"]["bare"])
    assert bare["expired_at"] is not None, bare
    upgraded = st.edges.active(source="Lloyd", target="Isaac Lab")
    assert [e["type"] for e in upgraded] == ["uses"], upgraded
    assert upgraded[0]["superseded_edge_id"] == guarded_store["ids"]["bare"]
    assert upgraded[0]["provenance"] == "EXTRACTED_CLASSIFIER_V4"
    st.close()


def test_apply_still_supersedes_the_classifier_s_own_co_resident_edges(
        monkeypatch, guarded_store, capsys):
    """`EXTRACTED_CLASSIFIER*` co-residents keep both of their lanes.

    `vLLM → CUDA` carries a prior classifier verdict of a *different* type, so
    it is still an upgrade and `retype`'s sweep still expires that stale verb —
    one active relation per pair, which #925 requires and
    `test_retype_keeps_history_and_collapses_the_pair` pins for `STATED` too.
    `Lloyd → vLLM` carries the *same* type, so it counts `already_typed` and
    only the redundant `mentions` row is retired.
    """
    assert _run(monkeypatch, guarded_store, "--apply") == 0
    out = capsys.readouterr().out
    assert _stat(out, "already_typed") == 1, out
    assert _stat(out, "upgrades") == 2, out

    st = KGStore(guarded_store["db"])
    ids = guarded_store["ids"]

    stale = st.edges.by_id(ids["cuda_classifier"])
    assert stale["expired_at"] is not None, "a classifier's own stale verb must still move"
    assert "pair re-typed as depends_on" in (stale["expired_reason"] or ""), stale
    cuda = st.edges.active(source="vLLM", target="CUDA")
    assert [e["type"] for e in cuda] == ["depends_on"], cuda
    assert cuda[0]["superseded_edge_id"] == ids["cuda_mentions"]

    live = st.edges.by_id(ids["vllm_classifier"])
    assert live["expired_at"] is None, live
    redundant = st.edges.by_id(ids["vllm_mentions"])
    assert redundant["expired_at"] is not None, redundant
    assert "already typed as uses" in redundant["expired_reason"], redundant
    st.close()
