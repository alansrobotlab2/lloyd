"""revert-suffix-merges.py — put a wrongly merged variant's facts back."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from app.kg_store import KGStore, parse_fact_file  # noqa: E402
_spec = importlib.util.spec_from_file_location("revert_suffix_merges", ROOT / "scripts/memory/revert-suffix-merges.py")
rv = importlib.util.module_from_spec(_spec); sys.modules["revert_suffix_merges"] = rv; _spec.loader.exec_module(rv)


def _write(path: Path, fm: dict, body: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n{body}")


def _facts(entity, cat, items):
    """`items` are `(fact_tag, text)`, or `(fact_tag, text, extra_frontmatter)`
    for a fact a merge retagged — `{"merged_from": <variant>}`."""
    facts = []
    for e, t, *extra in items:
        facts.append({"entity": e, "fact": t, "confidence": 0.9, "category": cat, **(extra[0] if extra else {})})
    return {"type": "facts", "entity": entity, "category": cat, "facts": facts}


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


# --- a canonical that absorbed TWO variants -----------------------------------
# The sweep picks merge candidates by `normalize_punct` equality, so `C`, `C#`
# and `C++` are one family and one apply can fold both variants into `C++`.
# Membership in a revert has to compare the surface itself: normalize them and
# every fact in the canonical looks like the variant being reverted.

AMBIGUOUS = {"SUFFIX_AMBIGUOUS"}          # classify_pair("C", "C++")[0]


@pytest.fixture
def two_variants(tmp_path):
    """`C` and `C#` both merged into `C++`, as the 2026-09-16 apply left them."""
    root = tmp_path / "facts"
    C, CS, CPP = "C", "C#", "C++"
    _write(root / CPP / f"{CPP}-relationship.md", _facts(CPP, "relationship", [
        (CPP, "C one", {"merged_from": C}),                   # C's fact, retagged by the merge
        (CPP, "Cs is the preview of C++", {"merged_from": CS}),  # C#'s fact
        (CPP, "C++ added move semantics in C++11", {}),       # C++'s own
        (CPP, "C++ modules shipped in C++20", {}),            # C++'s own
    ]))
    _write(root / CPP / f"{CPP}-state.md", _facts(CPP, "state", [
        (CPP, "Cs is standardised in C++26", {"merged_from": CS}),   # only C#'s fact
    ]))
    _write(root / CPP / f"{CPP}-overview.md", {"type": "overview", "entity": CPP, "category": "overview",
                                              "definition": "C++ is a systems language."}, "# Summary\n")
    st = KGStore(tmp_path / "kg.sqlite")
    st.entities.register(CPP)
    st.aliases.set(C, CPP, kind="punct", origin="sweep")
    st.aliases.set(CS, CPP, kind="punct", origin="sweep")
    report = {"variant_to_canonical": {C: CPP, CS: CPP},
              "ledger": {"timestamp": "2026-09-16T05:40:12+00:00"}}
    yield root, st, report, C, CS, CPP
    st.close()


def _one_pair(report, variant):
    """The report as one revert pass sees it: just this variant's merge."""
    return {"variant_to_canonical": {variant: report["variant_to_canonical"][variant]},
            "ledger": report["ledger"]}


def _fm(path):
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---")[1])


def test_reverting_one_variant_splits_the_shared_canonical_file(two_variants):
    """Clause 1: `C++-relationship.md` mixes three entities, so it must be
    `split`, and `C` gets back exactly the one fact tagged `merged_from: C`."""
    root, st, report, C, CS, CPP = two_variants
    ops = rv.plan_revert(_one_pair(report, C), AMBIGUOUS, root)
    assert [o["variant"] for o in ops] == [C]
    assert ops[0]["files"] == [{"file": f"{CPP}-relationship.md",
                                "action": "split", "facts": 1, "remaining": 3}]
    rv.execute(ops, root, st, apply=True)
    back = _fm(root / C / f"{C}-relationship.md")
    assert [f["fact"] for f in back["facts"]] == ["C one"]
    assert back["facts"][0]["entity"] == C and "merged_from" not in back["facts"][0]


def test_the_revert_leaves_the_canonical_its_own_and_the_other_variant_facts(two_variants):
    """Clause 2: reverting `C` must not touch `C++`'s own facts, must leave C#'s
    fact still tagged `merged_from: C#`, and must not move `C++-state.md`, which
    holds none of C's facts at all."""
    root, st, report, C, CS, CPP = two_variants
    ops = rv.plan_revert(_one_pair(report, C), AMBIGUOUS, root)
    assert [f["file"] for f in ops[0]["files"]] == [f"{CPP}-relationship.md"]   # state/overview unplanned
    rv.execute(ops, root, st, apply=True)
    kept = _fm(root / CPP / f"{CPP}-relationship.md")
    assert kept["entity"] == CPP
    assert [(f["fact"], f.get("merged_from")) for f in kept["facts"]] == [
        ("Cs is the preview of C++", CS), ("C++ added move semantics in C++11", None),
        ("C++ modules shipped in C++20", None)]
    state = _fm(root / CPP / f"{CPP}-state.md")                    # untouched, still in C++/
    assert [f.get("merged_from") for f in state["facts"]] == [CS]
    assert (root / CPP / f"{CPP}-overview.md").exists()            # C++ keeps its own overview
    assert not (root / C / f"{C}-state.md").exists()


def test_the_second_revert_pass_returns_the_other_variant(two_variants):
    """Clause 3: after C's pass, reverting `C#` on the same tree moves both of
    C#'s facts out and leaves `C++` with only its own two."""
    root, st, report, C, CS, CPP = two_variants
    rv.execute(rv.plan_revert(_one_pair(report, C), AMBIGUOUS, root), root, st, apply=True)
    ops = rv.plan_revert(_one_pair(report, CS), AMBIGUOUS, root)
    assert {f["file"]: f["action"] for f in ops[0]["files"]} == {
        f"{CPP}-relationship.md": "split", f"{CPP}-state.md": "move_whole"}
    rv.execute(ops, root, st, apply=True)
    assert sorted(f["fact"] for f in _fm(root / CS / f"{CS}-relationship.md")["facts"]) == \
        ["Cs is the preview of C++"]
    assert [f["fact"] for f in _fm(root / CS / f"{CS}-state.md")["facts"]] == ["Cs is standardised in C++26"]
    left = _fm(root / CPP / f"{CPP}-relationship.md")
    assert [f["fact"] for f in left["facts"]] == ["C++ added move semantics in C++11",
                                                  "C++ modules shipped in C++20"]
    assert all(f["entity"] == CPP for f in left["facts"])
    assert not (root / CPP / f"{CPP}-state.md").exists()           # emptied by the whole-file move
    assert {str(p.relative_to(root)) for p in root.rglob("*.md")} == {
        "C#/C#-relationship.md", "C#/C#-state.md", "C/C-relationship.md",
        "C++/C++-overview.md", "C++/C++-relationship.md"}


def test_a_canonical_file_with_none_of_the_variant_facts_is_not_planned(tmp_path):
    """Clause 4: `RT-X` normalizes equal to `RTX`, but every fact in
    `RTX/RTX-state.md` is tagged RTX and none was merged from RT-X, so the
    revert has nothing to take and must leave the file where it is."""
    root = tmp_path / "facts"
    _write(root / "RTX" / "RTX-state.md", _facts("RTX", "state", [
        ("RTX", "RTX renders a 3DGS scene at 60 fps on the 3090."),
        ("RTX", "The RTX PRO 6000 is the Blackwell card."),
        ("RTX", "RTX needs driver 570 or newer for the 50-series."),
    ]))
    st = KGStore(tmp_path / "kg.sqlite")
    st.entities.register("RTX")
    st.aliases.set("RT-X", "RTX", kind="punct", origin="sweep")
    ops = rv.plan_revert({"variant_to_canonical": {"RT-X": "RTX"}}, {"PUNCT"}, root)
    assert [o["files"] for o in ops] == [[]]
    rv.execute(ops, root, st, apply=True)
    fm = _fm(root / "RTX" / "RTX-state.md")
    assert fm["entity"] == "RTX" and len(fm["facts"]) == 3
    assert all(f["entity"] == "RTX" for f in fm["facts"])
    assert not (root / "RT-X" / "RT-X-state.md").exists()
    assert not (root / "RT-X").exists()
    st.close()


def test_membership_compares_the_surface_exactly_not_normalized(two_variants):
    """Clause 5: `normalize_punct` drops every non-alphanumeric, so it calls
    `C++` and `C` the same name — which is how a revert of `C` moved `C++`'s
    whole file and un-aliased `C#` on the way past it."""
    assert rv._belongs({"entity": "C++", "fact": "x"}, "C++", "C") is False
    assert rv._belongs({"entity": "PolaRiS", "fact": "x", "merged_from": "Polaris"}, "PolaRiS", "Polaris") is True
    assert rv._same("C++", "C") is False and rv._same("PolaRiS", "Polaris") is True
    root, st, report, C, CS, CPP = two_variants
    res = rv.execute(rv.plan_revert(_one_pair(report, C), AMBIGUOUS, root), root, st, apply=True)
    assert res["alias_ops"] == [{"remove": C, "was": CPP}, {"add": C}]
    assert st.aliases.resolve("c") is None
    assert st.aliases.resolve("c#") == CPP                          # C# is still routed to C++


# ── a killed apply, resumed, still reverts exactly by id (#1558) ──────────────
#
# Every test above hands `fix_edges` a report that was finished on purpose. This
# one is about the report a KILLED run leaves: the store half committed, the fact
# moves did not, and the only recovery is `entity-resolution-sweep.py --resume
# <that report>`. If the resume re-derives the edge trail instead of carrying the
# original one, the finished report shows 0 pairs — `rewrite_endpoint` walks only
# ACTIVE edges and the first apply expired them — and `fix_edges` then picks its
# mode from the one report it is handed and spends its single shot on the prose
# heuristic. An unrevertable merge that reports itself reverted is the failure.

SWEEP = ROOT / "scripts/memory/entity-resolution-sweep.py"


def _sweep_module():
    """Load the sweep by path, the way `kg_hygiene` does, to drive an apply that
    dies at a chosen line. A subprocess cannot be interrupted mid-function."""
    spec = importlib.util.spec_from_file_location("entity_resolution_sweep", SWEEP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["entity_resolution_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cli(script, root, db, out, *extra):
    return subprocess.run([sys.executable, str(script), "--facts-dir", str(root),
                           "--db", str(db), "--out-dir", str(out), *extra],
                          capture_output=True, text=True, timeout=180)


def _reverted_record(out: Path) -> dict:
    recs = sorted(out.glob("entity-merges-reverted-*.json"))
    assert recs, "the revert wrote no record, so it never ran"
    return json.loads(recs[-1].read_text())


def test_a_resumed_apply_still_reverts_exactly_by_id(tmp_path, monkeypatch):
    """Clause 3: `--resume` writes no alias and no edge for the pairs the killed
    run already committed, carries that run's `(old_edge_id, new_edge_id)` pairs
    into the finished report, and leaves
    `revert-suffix-merges.py --applied <that report> --apply --fix-edges` in
    `exact` mode with pairs > 0.

    Two variants share one canonical so the kill has to land between them: one
    dir journaled done, one left pending.
    """
    root, db = tmp_path / "facts", tmp_path / "kg.sqlite"
    out = tmp_path / "out"; out.mkdir()
    canonical = "Rerank Reranker"
    v_upper, v_lower = "RERANK RERANKER", "rerank reranker"
    st = KGStore(db)
    (root / canonical).mkdir(parents=True)
    _write(root / canonical / "Rerank Reranker-state.md",
           _facts(canonical, "state", [(canonical, "Reranker v2 ships Thursday.")]))
    # Both variants are CASE forms of the canonical, which is the merge #48 makes
    # daily and needs no semantic gate. A bare `Rerank` beside
    # `Rerank Reranker` would be a SUFFIX pair and would never merge at all.
    for variant, target in ((v_upper, "Ray"), (v_lower, "Triton")):
        _write(root / variant / f"{variant}-state.md",
               _facts(variant, "state", [(variant, f"{variant} serves 12 req/s.")]))
        st.entities.register(canonical); st.entities.register(variant)
        st.edges.add({"source": canonical, "target": target, "type": "mentions"}, origin="seed")
        st.edges.add({"source": variant, "target": target, "type": "mentions"}, origin="seed")
    st.facts_idx.reindex([p for p in root.rglob("*.md")], root=root)

    ers = _sweep_module()
    dirs = {d.name for d in root.iterdir()}
    plan = ers.build_plan(st.edges.active(), dirs, allowed_tiers=["CASE", "PUNCT"])
    merged = {m["variant"] for c in plan["safe_merges"] for m in c["merges"]}
    assert merged == {v_upper, v_lower}, f"need both mechanical variants merged: {merged}"

    report = out / "entity-merges-applied-2026-09-26-20260926T000000Z.json"
    ers.claim_report(report, out / "entity-merges-latest.jsonl", "2026-09-26", "20260926T000000Z")
    real, seen = ers.retag_fact_file, [0]

    def dying(path, old, new):
        seen[0] += 1
        if seen[0] > 1:
            raise KeyboardInterrupt("killed in the fact-move window")
        return real(path, old, new)

    monkeypatch.setattr(ers, "retag_fact_file", dying)
    with pytest.raises(KeyboardInterrupt):
        ers.apply_merges(plan, st, root, rebuild_aliases=False, existing_dirs=dirs,
                         report_path=str(report))
    st.close()

    stub = json.loads(report.read_text())
    journaled = [op["variant"] for op in stub["dir_operations"] if op.get("done")]
    assert journaled in ([v_upper], [v_lower]), \
        f"exactly one variant had finished before the kill: {stub['dir_operations']}"
    assert sum(len(v) for v in stub["edge_rewrites"].values()) == 2, \
        "both variants' edges committed in the one transaction, so the journal holds both trails"

    r = _cli(SWEEP, root, db, out, "--resume", str(report))
    assert r.returncode == 0, r.stdout + r.stderr
    finished = json.loads(report.read_text())
    assert finished["report_status"] == "complete"
    assert finished["alias_writes"] == 0, "a resume must write no alias"
    assert finished["store_before"]["aliases"] == finished["store_after"]["aliases"] \
        and finished["store_before"]["edges_active"] == finished["store_after"]["edges_active"], \
        "the resume changed the store: it is the file half only"
    assert sum(len(v) for v in finished["edge_rewrites"].values()) == 2, \
        ("the finished report lost run 1's ids; a re-derived trail is EMPTY here because "
         "rewrite_endpoint walks only ACTIVE edges and run 1 expired them")
    assert not (root / v_upper).exists() and not (root / v_lower).exists(), \
        "the resume left a variant dir behind"

    # `--tiers CASE`, not ALL: `plan_revert` selects each pair by
    # `classify_pair(variant, canonical)`, and both variants here are CASE forms of
    # their canonical — an ALL would plan zero ops and never reach `fix_edges`.
    r = _cli(ROOT / "scripts/memory/revert-suffix-merges.py", root, db, out,
             "--applied", str(report), "--tiers", "CASE", "--apply", "--fix-edges")
    assert r.returncode == 0, r.stdout + r.stderr
    edges = _reverted_record(out)["edges"]
    assert edges["mode"] == "exact", f"fell back to the prose heuristic: {edges}"
    assert edges["pairs"] == 2 and edges["reverted"] == 2, edges

    st = KGStore(db)
    try:
        live = {(e["source"], e["target"]) for e in st.edges.active()}
    finally:
        st.close()
    assert (canonical, "Ray") in live and (v_lower, "Triton") in live, \
        "the merged edges are not live again, so the revert did not invert them by id"


# ── a fact whose own text carries a fence (#2138) ────────────────────────────
#
# `yaml.dump` renders a multi-line fact as a single-quoted scalar with INDENTED
# continuation lines, and prose like the live corpus's `` `---segment:` `` keeps a
# fence inline. Either way the file's own `---` sits after the opening fence, so
# `_read`'s old `text.split("---", 2)` cut the block in half — and the truncated
# slice still LOADS, as `{type, entity, category, facts}` with the half-cut fact in
# the list and `source_doc` gone. This module writes what `_read` returns and then
# unlinks the source, so a revert of a `---`-bearing file deleted the facts it had
# never read: 4 facts in, 2 out, no witness.


def test_revert_moves_a_fence_bearing_file_with_every_fact_intact(world):   # clause 4
    root, st, report, V, C = world
    src_doc = "knowledge/software/fact-file-fence-handling.md"
    texts = ["Scan ArXiv nightly.",
             "The digest is cut at `---segment:` when the insert is anchored on the fence line.",
             "Score Hacker News nightly, then re-dump:\n---\nand a naive cut loses what follows.",
             "Anchored closing fences survive every writer."]
    fm = {"type": "facts", "entity": V, "category": "goal",
          "facts": [{"id": f"goal-00{i}", "entity": V, "fact": t, "confidence": 0.9,
                     "category": "goal", "created_at": f"2026-09-01T04:00:0{i}+00:00",
                     "source_doc": src_doc} for i, t in enumerate(texts, 1)],
          "source_doc": src_doc}
    goal = root / C / "Intel-goal.md"
    _write(goal, fm, "# Intel Pipeline System - goal\n")

    ops = rv.plan_revert(report, {"SUFFIX_SAFE"}, root)
    assert {f["file"]: f["action"] for f in ops[0]["files"]}["Intel-goal.md"] == "move_whole"
    rv.execute(ops, root, st, apply=True)

    dest = root / V / f"{V}-goal.md"
    assert dest.exists() and not goal.exists(), "the whole-file move did not happen"
    out_fm, out_body = rv._read(dest)
    assert [f["fact"] for f in out_fm["facts"]] == texts, \
        "the revert wrote back only what its read could see"
    assert [f["fact"] for f in parse_fact_file(dest)[1]] == texts, \
        "the store's own reader must see the same four facts the revert wrote"
    assert out_fm["source_doc"] == src_doc, "file-level provenance was never in a truncated read"
    assert all(f.get("created_at") and f.get("source_doc") == src_doc for f in out_fm["facts"]), \
        "per-fact provenance has to survive the re-dump, not just the fact text"
    assert out_body.startswith("# Intel Pipeline System - goal"), repr(out_body)
