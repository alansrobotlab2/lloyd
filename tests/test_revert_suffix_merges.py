"""revert-suffix-merges.py — put a wrongly merged variant's facts back."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from app.kg_store import KGStore  # noqa: E402
_spec = importlib.util.spec_from_file_location("revert_suffix_merges", ROOT / "scripts/memory/revert-suffix-merges.py")
rv = importlib.util.module_from_spec(_spec); sys.modules["revert_suffix_merges"] = rv; _spec.loader.exec_module(rv)


def _write(path: Path, fm: dict, body: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n{body}")


def _facts(entity, cat, items):
    return {"type": "facts", "entity": entity, "category": cat,
            "facts": [{"entity": e, "fact": t, "confidence": 0.9, "category": cat} for e, t in items]}


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "facts"
    V, C = "Intel Pipeline System", "Intel"
    # whole file renamed into Intel/ by the merge: still tagged as the pipeline
    _write(root / C / "Intel-goal.md", _facts(V, "goal", [(V, "Scan ArXiv nightly."), (V, "Score Hacker News.")]))
    # merged file: two Intel facts + one pipeline fact
    _write(root / C / "Intel-state.md", _facts(C, "state", [(C, "Intel undercuts Nvidia on price."),
                                                             (V, "Runs at 03:00 on the primary model."),
                                                             (C, "Intel shipped the Pro B70 GPU.")]))
    # Intel's own overview survived; the pipeline's was discarded by the merge
    _write(root / C / "Intel-overview.md", {"type": "overview", "entity": C, "category": "overview",
                                           "definition": "Intel is a semiconductor company."}, "# Summary\n")
    # an unrelated CASE merge in the same report must be untouched
    _write(root / "PolaRiS" / "PolaRiS-state.md", _facts("PolaRiS", "state", [("PolaRiS", "A benchmark.")]))
    st = KGStore(tmp_path / "kg.sqlite")
    st.entities.register(C); st.entities.register("PolaRiS")
    st.aliases.set("intel pipeline system", C, kind="suffix", origin="sweep")
    st.aliases.set("Intel Pipeline System", C, kind="suffix", origin="sweep")
    st.aliases.set("polaris", "PolaRiS", kind="case", origin="sweep")
    st.edges.add({"source": C, "target": "ArXiv", "type": "mentions", "provenance": "EXTRACTED",
                  "created_at": "2026-09-03T22:41:09+00:00"}, origin="seed")
    st.edges.add({"source": C, "target": "Nvidia", "type": "competes_with",
                  "provenance": "EXTRACTED_CLASSIFIER_V4",
                  "created_at": "2026-09-03T22:41:20+00:00"}, origin="classifier")
    st.edges.add({"source": C, "target": "Old Thing", "type": "mentions", "provenance": "EXTRACTED",
                  "created_at": "2026-05-01T00:00:00+00:00"}, origin="seed")
    report = {"variant_to_canonical": {V: C, "Polaris": "PolaRiS"},
              "ledger": {"timestamp": "2026-09-03T12:33:14+00:00"}}
    yield root, st, report, V, C
    st.close()


def test_plan_selects_only_the_suffix_tier_and_classifies_files(world):
    root, st, report, V, C = world
    ops = rv.plan_revert(report, {"SUFFIX_SAFE"}, root)
    assert [o["variant"] for o in ops] == [V]
    actions = {f["file"]: f["action"] for f in ops[0]["files"]}
    assert actions == {"Intel-goal.md": "move_whole", "Intel-state.md": "split"}
    assert ops[0]["lost_overview"] is True


def test_dry_run_changes_nothing(world):
    root, st, report, V, C = world
    ops = rv.plan_revert(report, {"SUFFIX_SAFE"}, root)
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*.md"))
    res = rv.execute(ops, root, st, apply=False)
    assert len(res["file_ops"]) == 2
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*.md")) == before
    assert st.aliases.resolve("intel pipeline system") == C


def test_apply_restores_the_variant_and_cleans_aliases(world):
    root, st, report, V, C = world
    ops = rv.plan_revert(report, {"SUFFIX_SAFE"}, root)
    res = rv.execute(ops, root, st, apply=True)

    goal = root / V / f"{V}-goal.md"
    assert goal.exists() and not (root / C / "Intel-goal.md").exists()
    gfm = yaml.safe_load(goal.read_text().split("---")[1])
    assert gfm["entity"] == V and all(f["entity"] == V for f in gfm["facts"]) and len(gfm["facts"]) == 2

    state_c = yaml.safe_load((root / C / "Intel-state.md").read_text().split("---")[1])
    assert [f["fact"] for f in state_c["facts"]] == ["Intel undercuts Nvidia on price.", "Intel shipped the Pro B70 GPU."]
    state_v = yaml.safe_load((root / V / f"{V}-state.md").read_text().split("---")[1])
    assert [f["fact"] for f in state_v["facts"]] == ["Runs at 03:00 on the primary model."]
    assert "**Entity:** Intel" in (root / C / "Intel-state.md").read_text()

    assert (root / C / "Intel-overview.md").exists()            # canonical keeps its own
    assert (root / "PolaRiS" / "PolaRiS-state.md").exists()      # CASE merge untouched

    assert st.aliases.resolve("intel pipeline system") is None
    assert st.entities.exists(V)                                 # variant is itself again
    assert st.aliases.resolve("polaris") == "PolaRiS"             # the CASE merge is untouched
    assert res["touched_canonicals"] == [C]


def test_split_merges_into_an_already_recreated_variant_dir(world):
    root, st, report, V, C = world
    _write(root / V / f"{V}-state.md", _facts(V, "state", [(V, "Recreated by the extractor yesterday.")]))
    ops = rv.plan_revert(report, {"SUFFIX_SAFE"}, root)
    rv.execute(ops, root, st, apply=True)
    fm = yaml.safe_load((root / V / f"{V}-state.md").read_text().split("---")[1])
    assert sorted(f["fact"] for f in fm["facts"]) == ["Recreated by the extractor yesterday.",
                                                      "Runs at 03:00 on the primary model."]


def test_fix_edges_falls_back_to_prose_for_a_report_with_no_id_trail(world):
    """Apply reports written before the store carry no `edge_rewrites`."""
    root, st, report, V, C = world
    ops = rv.plan_revert(report, {"SUFFIX_SAFE"}, root)
    rv.execute(ops, root, st, apply=True)
    out = rv.fix_edges(st, report, [C], root, since="2026-09-03T00:00:00", apply=True)
    assert out["mode"] == "heuristic"
    assert [e["target"] for e in out["expired"]] == ["ArXiv"]     # named only in the moved pipeline facts
    edges = {e["target"]: e for e in st.edges.all()}
    assert edges["ArXiv"]["expired_at"] and "revert-suffix-merges" in edges["ArXiv"]["expired_reason"]
    assert edges["Nvidia"]["expired_at"] is None                    # still named in Intel's own facts
    assert edges["Old Thing"]["expired_at"] is None                 # predates the apply window


def test_fix_edges_is_exact_when_the_report_carries_rewrite_ids(world):
    """The id trail restores the pre-merge graph precisely — no prose guessing."""
    root, st, report, V, C = world
    st.entities.register(V)
    keep = st.edges.add({"source": V, "target": "ArXiv", "type": "mentions",
                         "provenance": "EXTRACTED"}, origin="seed")
    pairs = st.edges.rewrite_endpoint(V, C, origin="sweep")
    assert pairs and st.edges.by_id(keep)["expired_at"]
    report["edge_rewrites"] = {V: [list(p) for p in pairs]}

    out = rv.fix_edges(st, report, [C], root, since="2026-09-03T00:00:00", apply=True)
    assert out["mode"] == "exact" and out["reverted"] == len(pairs)
    assert st.edges.by_id(keep)["expired_at"] is None            # the original is live again
    assert {(e["source"], e["target"]) for e in st.edges.active(either=V)} == {(V, "ArXiv")}
    # everything the merge did not touch is untouched
    assert st.edges.find_active(C, "Nvidia", "competes_with")


def test_variant_filename_prefix_restoration():
    assert rv._variant_filename("Intel-goal.md", "Intel", "Intel Pipeline System") == "Intel Pipeline System-goal.md"
    assert rv._variant_filename("Intel Pipeline System-goal.md", "Intel", "Intel Pipeline System") == "Intel Pipeline System-goal.md"
    assert rv._variant_filename("notes.md", "Intel", "X") == "X-notes.md"


def test_revert_recognises_facts_retagged_by_a_merge(world):
    """A merge now rewrites facts to the canonical and stamps merged_from; the
    revert must still find them by that stamp and drop it on the way back."""
    root, st, report, V, C = world
    _write(root / C / "Intel-preference.md",
           {"type": "facts", "entity": C, "category": "preference",
            "facts": [{"entity": C, "fact": "prefers ArXiv first", "merged_from": V},
                      {"entity": C, "fact": "Intel's own preference"}]})
    ops = rv.plan_revert(report, {"SUFFIX_SAFE"}, root)
    assert {f["file"]: f["action"] for f in ops[0]["files"]}["Intel-preference.md"] == "split"
    rv.execute(ops, root, st, apply=True)
    back = yaml.safe_load((root / V / f"{V}-preference.md").read_text().split("---")[1])
    assert [f["fact"] for f in back["facts"]] == ["prefers ArXiv first"]
    assert back["facts"][0]["entity"] == V and "merged_from" not in back["facts"][0]
    kept = yaml.safe_load((root / C / "Intel-preference.md").read_text().split("---")[1])
    assert [f["fact"] for f in kept["facts"]] == ["Intel's own preference"]
