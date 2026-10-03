"""entity-resolution-sweep.py — the rules that stop a name-shape match from
becoming a merge, and the guards around --apply."""
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from app.kg_store import KGStore, parse_fact_file  # noqa: E402
SWEEP = ROOT / "scripts" / "memory" / "entity-resolution-sweep.py"
_spec = importlib.util.spec_from_file_location("ers_test", SWEEP)
ers = importlib.util.module_from_spec(_spec); sys.modules["ers_test"] = ers; _spec.loader.exec_module(ers)


class FakeGate:
    def __init__(self, answers):        # {(variant, canonical): "SAME"|"REVIEW"}
        self.answers, self.asked = answers, []
    def verdict(self, a, b):
        self.asked.append((a, b))
        return {"decision": self.answers.get((a, b), "REVIEW"), "judges": {"fake": {"verdict": "x", "reason": "r"}}}


# ── canonical selection ──────────────────────────────────────────────────────

def test_canonical_prefers_degree_then_title_over_slug():
    deg = {"Nightly Reflection": 9, "nightly-reflection": 2}
    assert ers.pick_canonical(list(deg), deg, set(deg)) == "Nightly Reflection"
    deg = {"Nightly Reflection": 3, "nightly-reflection": 3}
    assert ers.pick_canonical(list(deg), deg, set(deg)) == "Nightly Reflection"

def test_bare_noun_is_no_longer_preferred_over_the_suffixed_form():
    # the old rule 1 absorbed `Alfie pipeline` into `Alfie` regardless of use
    deg = {"Alfie": 1, "Alfie pipeline": 6}
    assert ers.pick_canonical(list(deg), deg, set(deg)) == "Alfie pipeline"

def test_is_slug():
    assert ers._is_slug("nightly-reflection") and ers._is_slug("worker_queue")
    assert not ers._is_slug("Nightly Reflection") and not ers._is_slug("vLLM") and not ers._is_slug("alfie")


# ── decisions ────────────────────────────────────────────────────────────────

def test_suffix_never_auto_merges_on_shape_even_with_zero_degree():
    ok, why = ers.decide_merge("SUFFIX_SAFE", "Intel", ["Intel", "Intel Pipeline"], {"Intel": 0, "Intel Pipeline": 0})
    assert ok is False and "semantic gate" in why

def test_case_and_punct_still_auto_merge():
    assert ers.decide_merge("CASE", "vLLM", ["vLLM", "vllm"], {"vLLM": 3, "vllm": 0})[0] is True
    assert ers.decide_merge("PUNCT", "SWE-Bench", ["SWE-Bench", "SweBench"], {})[0] is True


def _edges(pairs):
    return [{"source": s, "target": t, "type": "mentions", "expired_at": None} for s, t in pairs]

def test_build_plan_routes_suffix_through_the_gate():
    dirs = {"Intel", "Intel Pipeline", "Morning Briefing", "Morning Briefing System", "vLLM", "vllm"}
    gate = FakeGate({("Morning Briefing System", "Morning Briefing"): "SAME",
                     ("Intel Pipeline", "Intel"): "REVIEW"})
    plan = ers.build_plan(_edges([("Intel", "Nvidia"), ("Morning Briefing", "Alan")]), dirs, gate=gate)
    safe = {c["canonical"]: c for c in plan["safe_merges"]}
    amb = {c["canonical"]: c for c in plan["ambiguous"]}
    assert "Morning Briefing" in safe and safe["Morning Briefing"]["decision"].startswith("SUFFIX_JUDGED")
    assert safe["Morning Briefing"]["gate"]["Morning Briefing System"]["decision"] == "SAME"
    assert "Intel" in amb and "review" in amb["Intel"]["decision"]
    assert "vLLM" in safe and safe["vLLM"]["tier"] == "CASE"
    assert plan["gate_stats"] == {"asked": 2, "same": 1, "review": 1}
    assert sorted(gate.asked) == [("Intel Pipeline", "Intel"), ("Morning Briefing System", "Morning Briefing")]

def test_without_a_gate_every_suffix_cluster_is_review():
    plan = ers.build_plan([], {"Intel", "Intel Pipeline", "vLLM", "vllm"}, gate=None)
    assert [c["canonical"] for c in plan["safe_merges"]] == ["vLLM"]
    assert [c["tier"] for c in plan["ambiguous"]] == ["SUFFIX_SAFE"]

def test_tiers_filter_demotes_excluded_tiers():
    plan = ers.build_plan([], {"vLLM", "vllm", "SWE-Bench", "SweBench"}, allowed_tiers={"CASE"})
    assert [c["tier"] for c in plan["safe_merges"]] == ["CASE"]
    assert "excluded by --tiers" in plan["ambiguous"][0]["decision"]

def test_junk_entities_never_enter_a_cluster():
    plan = ers.build_plan([], {"server.py", "Server.PY", "vLLM", "vllm"})
    assert [c["canonical"] for c in plan["safe_merges"]] == ["vLLM"]


# ── aliases: an approved suffix merge must be routable ───────────────────────

def test_apply_writes_the_approved_suffix_alias(tmp_path):
    """A merge this plan approved gets its alias whatever its tier. Suffix
    aliases used to be dropped as `noise` even for merges the sweep had just
    performed, so the extractor recreated the variant on its next pass."""
    st = KGStore(tmp_path / "kg.sqlite")
    plan = {"safe_merges": [{"canonical": "Morning Briefing", "tier": "SUFFIX_SAFE",
                             "merges": [{"variant": "Morning Briefing System", "subtier": "SUFFIX_SAFE"}]}]}
    ers.apply_merges(plan, st, tmp_path / "facts", rebuild_aliases=False, existing_dirs=set())
    assert st.aliases.resolve("morning briefing system") == "Morning Briefing"
    assert st.aliases.resolve("Morning Briefing System") == "Morning Briefing"
    assert st.aliases.for_canonical("Morning Briefing")[0]["kind"] == "suffix"
    st.close()


def test_rebuild_aliases_prunes_inherited_suffix_noise(tmp_path):
    st = KGStore(tmp_path / "kg.sqlite")
    st.entities.register("legacy noise"); st.entities.register("Kept"); st.entities.register("KEPT")
    st.aliases.set("legacy noise system", "legacy noise", kind="suffix", origin="legacy")
    st.aliases.set("KEPT", "Kept", kind="case", origin="legacy")
    ers.apply_merges({"safe_merges": []}, st, tmp_path / "facts", rebuild_aliases=True,
                     existing_dirs={"legacy noise", "Kept"})
    assert st.aliases.resolve("legacy noise system") is None   # suffix-only difference → noise
    assert st.aliases.resolve("KEPT") == "Kept"                # case-only variants are legitimate
    st.close()


def test_is_alias_noise_rule():
    assert ers._is_alias_noise("legacy noise system", "legacy noise")
    assert not ers._is_alias_noise("VLLM", "vLLM")
    assert not ers._is_alias_noise("swe-bench", "SWE Bench")


# ── baseline guard ───────────────────────────────────────────────────────────

def test_degraded_reason():
    assert ers.degraded_reason(2, 7260) is not None
    assert ers.degraded_reason(3789, 7260) is None
    # #1557 replaced `assert ers.degraded_reason(5, 0) is None  # no baseline yet →
    # nothing to compare`. That assertion pinned the hole: with `baseline <= 0`
    # returning None first, a missing or unparseable graph-baseline.json disarmed
    # the guard for the whole run — and `update_baseline` bootstraps
    # `baseline := active` before the guard is consulted, so the run never learns.
    # 2026-09-22 is that run on disk: baseline_active_edges 0, 30 merges applied,
    # 0 edges rewritten, safety reading "not degraded".
    # An empty store refuses with no baseline to measure against, and the message
    # names the graph as empty/unmeasurable rather than calling it healthy.
    reason = ers.degraded_reason(0, 0)
    assert reason is not None, "a 0-edge store must never be measurable as healthy"
    assert "empty or unmeasurable" in reason
    assert "--allow-degraded" in reason
    # The refusal is a fact about the store, not about the file: it stands with a
    # baseline on disk too, and it does not stand just because the floor is gone.
    assert ers.degraded_reason(0, 50_837) is not None
    assert ers.degraded_reason(5, 0) is None       # a non-empty store, no floor: labelled, not refused


def test_safety_record_labels_an_unmeasured_guard_apart_from_a_clean_one():
    """`not degraded` and `unmeasured` were the same string on the report that
    moved 30 fact dirs (#1557); they are now different labels, and the store's own
    count rides with them."""
    measured = ers.safety_record(True, False, None, {}, None, active_edges=50_908)
    assert measured["degraded_graph"] == "not degraded"
    assert measured["active_edges"] == 50_908
    assert measured["baseline_measured"] is True
    unmeasured = ers.safety_record(True, False, None, {}, None,
                                   active_edges=0, measured=False)
    assert unmeasured["degraded_graph"] != "not degraded"
    assert unmeasured["degraded_graph"].startswith("unmeasured")
    assert unmeasured["active_edges"] == 0
    assert unmeasured["baseline_measured"] is False
    # an explicit override outranks the label: a bypass is never reported as a
    # guard that simply could not see
    bypassed = ers.safety_record(True, True, None, {}, "graph is empty or unmeasurable",
                                 active_edges=0, measured=False)
    assert bypassed["degraded_graph"] == "bypassed: --allow-degraded"

def test_update_baseline_keeps_the_max(tmp_path):
    p = tmp_path / "b.json"
    assert ers.update_baseline(100, p) == 100
    assert ers.update_baseline(50, p) == 100
    assert ers.load_baseline(p) == 100


# ── end to end on a temp tree ────────────────────────────────────────────────

def _tree(tmp_path, edges: int = 2):
    """The mergeable fixture tree; `edges=0` is the 2026-09-22 store — the same
    entities and fact dirs, no graph under them, so every degree is 0."""
    root = tmp_path / "facts"
    for name in ("vLLM", "vllm", "Intel", "Intel Pipeline"):
        d = root / name; d.mkdir(parents=True)
        fm = {"type": "facts", "entity": name, "category": "state",
              "facts": [{"entity": name, "fact": f"{name} exists.", "confidence": 0.9, "category": "state"}]}
        (d / f"{name}-state.md").write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name} - state\n")
    db = tmp_path / "kg.sqlite"
    st = KGStore(db)
    for n in ("vLLM", "vllm", "Intel", "Intel Pipeline"):
        st.entities.register(n)
    if edges >= 1:
        st.edges.add({"source": "vLLM", "target": "Ray", "type": "mentions"}, origin="test")
    if edges >= 2:
        st.edges.add({"source": "vllm", "target": "Ray", "type": "mentions"}, origin="test")
    st.close()
    return root, db

def _run(root, db, out, *extra):
    cmd = [sys.executable, str(SWEEP), "--facts-dir", str(root), "--db", str(db),
           "--out-dir", str(out), "--no-gate", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

def test_apply_refuses_on_a_degraded_graph(tmp_path):
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    (out / "graph-baseline.json").write_text(json.dumps({"active_edges": 10000}))
    r = _run(root, db, out, "--apply")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUSING --apply" in r.stdout
    assert (root / "vllm").exists()                       # nothing moved
    st = KGStore(db)
    assert st.aliases.resolve("vllm") is None             # nothing rewritten
    st.close()
    # the override is explicit
    r2 = _run(root, db, out, "--apply", "--allow-degraded")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert not (root / "vllm").exists()

def test_apply_refuses_a_store_with_no_active_edges_even_with_no_baseline(tmp_path):
    """#1557, the shape that already fired. `entity-merges-applied-2026-09-22-
    20260922T204831Z.json` records `store_before.edges_active: 0` and no
    `graph-baseline.json` on disk, and the guard answered `baseline <= 0 → None`:
    30 fact dirs moved, 0 edges rewritten, and the report's own `safety` block said
    `"not degraded"`. The store's own count refuses now, so losing the baseline
    file can no longer disarm the check — the 09-22 run came from
    `--apply --tiers CASE,PUNCT` under `SUPERVISOR_PROCESS_NAME=lloyd-mcp` with no
    `--allow-degraded` in its argv, which is what makes an automated refusal the
    behaviour that matters here."""
    root, db = _tree(tmp_path, edges=0); out = tmp_path / "out"; out.mkdir()
    assert not (out / "graph-baseline.json").exists(), "the fixture seeded the very file that went missing"

    r = _run(root, db, out, "--apply")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUSING --apply" in r.stdout
    assert "empty or unmeasurable" in r.stdout, r.stdout
    assert "[no baseline on disk before this run" in r.stdout, r.stdout
    assert (root / "vllm").exists()                          # no fact dir moved
    assert (root / "Intel Pipeline").exists()
    assert not list(out.glob("entity-merges-applied-*.json"))   # nothing claimed an apply
    st = KGStore(db)
    assert st.aliases.resolve("vllm") is None                 # nothing rewritten
    st.close()

    # the override stays explicit — and says which store it overrode for
    r2 = _run(root, db, out, "--apply", "--allow-degraded")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    safety = _applied_report(out)["safety"]
    assert safety["active_edges"] == 0, safety
    assert safety["baseline_measured"] is False, safety
    assert safety["degraded_graph"] == "bypassed: --allow-degraded", safety

def test_a_store_with_edges_and_no_baseline_file_still_applies_and_says_unmeasured(tmp_path):
    """The other half of #1557: no floor is not the same finding as an empty store.
    Two active edges and no `graph-baseline.json` must still complete — every
    fixture apply in this file runs that way, so an over-broad refusal would fail
    them all — and the report must own up to the guard not having measured."""
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    assert not (out / "graph-baseline.json").exists()
    r = _run(root, db, out, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (root / "vllm").exists()
    safety = _applied_report(out)["safety"]
    assert safety["degraded_graph"].startswith("unmeasured"), safety
    assert safety["baseline_measured"] is False, safety
    assert safety["active_edges"] == 2, safety
    # and the floor it had no opinion about is now recorded for the next run
    assert json.loads((out / "graph-baseline.json").read_text())["active_edges"] == 2

def test_apply_writes_aliases_and_stamps_the_ledger(tmp_path):
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    r = _run(root, db, out, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (root / "vllm").exists() and (root / "vLLM" / "vLLM-state.md").exists()
    assert (root / "Intel Pipeline").exists()                          # suffix cluster untouched
    st = KGStore(db)
    assert st.aliases.resolve("vllm") == "vLLM"
    assert st.aliases.for_canonical("vLLM")[0]["kind"] == "case"
    st.close()
    reports = list(out.glob("entity-merges-applied-*.json")); assert len(reports) == 1
    rep = json.loads(reports[0].read_text())
    assert rep["ledger"]["argv"] and rep["ledger"]["pid"] and "cwd" in rep["ledger"]
    assert rep["tiers_allowed"] == ["CASE", "PUNCT", "SUFFIX_SAFE"]
    assert Path(rep["store_backup"]).exists()
    assert json.loads((out / "graph-baseline.json").read_text())["active_edges"] == 2
    # the plan carries the suffix cluster as review, with the reason
    plans = [q for q in out.glob("entity-merges-*.jsonl") if not q.is_symlink()]
    assert len(plans) == 1 and (out / "entity-merges-latest.jsonl").resolve() == plans[0].resolve()
    assert rep["plan_file"] == str(plans[0])
    plan_lines = [json.loads(l) for l in plans[0].read_text().splitlines() if l.strip()]
    amb = [c for c in plan_lines if c["status"] == "AMBIGUOUS"]
    assert amb and "semantic gate" in amb[0]["decision"]


# ── the facts_idx rows a merge leaves behind (#996) ─────────────────────────
#
# `apply_merges` moves the variant's fact files into the canonical dir and then
# reindexes the paths it touched. The index is derived from the markdown, so the
# one thing that must follow a move is the row that named the path BEFORE the
# move: while it stays live (`expired_at IS NULL AND invalid_at IS NULL`) any
# coverage/fragmentation query that filters on live rows still counts a variant
# that has genuinely been merged away.

def _applied_index(tmp_path):
    """The `_tree` harness, indexed BEFORE the merge, with one nested variant
    subdir, run through one `--no-gate --apply`.

    Two things the plain harness lacks and this needs:

    * an index that predates the apply — nothing in the sweep indexes first, and
      on the live store the indexer has already run, so the rows the merge has
      to retire are already there;
    * `<variant>/<sub>/<file>.md` — the nested variant dir the merge moves at its
      nested-subdir loop. A full reindex walks one level per entity dir, so this
      file is indexed by path instead, which is how such a row reaches the live
      index at all (an incremental writer indexes the paths it is named).

    Returns (root, db, live rows before the apply, live Intel rows before it).
    """
    root, db = _tree(tmp_path)
    nested = root / "vllm" / "experiments"
    nested.mkdir(parents=True)
    fm = {"type": "facts", "entity": "vllm", "category": "experiment",
          "facts": [{"entity": "vllm", "fact": "vllm scanned a run.",
                     "confidence": 0.9, "category": "experiment"}]}
    (nested / "scan.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# vllm - experiment\n")
    st = KGStore(db)
    st.facts_idx.reindex(root=root)                            # before any sweep runs
    st.facts_idx.update_file(nested / "scan.md", root=root)    # as a writer would
    before, intel_before = _live_rows(st), _intel_rows(st)
    st.close()
    out = tmp_path / "out"; out.mkdir()
    r = _run(root, db, out, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    return root, db, before, intel_before


def _live_rows(st):
    """(entity, file_path) for every live facts_idx row."""
    return {(r["entity"], r["file_path"]) for r in st.conn.execute(
        "SELECT entity, file_path FROM facts_idx "
        "WHERE expired_at IS NULL AND invalid_at IS NULL")}


def test_apply_leaves_no_live_index_row_pointing_at_a_deleted_path(tmp_path):  # clause 1
    root, db, _, _ = _applied_index(tmp_path)
    st = KGStore(db)
    orphans = sorted((r["entity"], r["file_path"]) for r in st.conn.execute(
        "SELECT entity, file_path FROM facts_idx "
        "WHERE expired_at IS NULL AND invalid_at IS NULL")
        if not (root / r["file_path"]).exists())
    st.close()
    assert orphans == []


def test_apply_retires_the_variant_but_keeps_the_facts_counted(tmp_path):  # clause 2
    root, db, _, _ = _applied_index(tmp_path)
    st = KGStore(db)
    rows = _live_rows(st)
    st.close()
    assert sum(1 for e, _ in rows if e == "vLLM") >= 1          # retirement, not emptying
    assert sum(1 for e, _ in rows if e == "vllm") == 0


def _intel_rows(st):
    """Full live row tuples for the two entity dirs the plan never merged
    ('Intel' and 'Intel Pipeline'), so a fix that over-reaches past the paths
    this merge moved — retiring by entity name, or wiping the table — fails
    here on content and not just on presence."""
    return sorted(tuple(r) for r in st.conn.execute(
        "SELECT entity, file_path, category, fact FROM facts_idx "
        "WHERE expired_at IS NULL AND invalid_at IS NULL "
        "AND file_path LIKE 'Intel%'").fetchall())


def test_apply_does_not_touch_the_rows_of_an_unmerged_entity(tmp_path):  # clause 3
    root, db, before, intel_before = _applied_index(tmp_path)
    assert ("Intel Pipeline", "Intel Pipeline/Intel Pipeline-state.md") in before
    st = KGStore(db)
    intel_after = _intel_rows(st)
    st.close()
    assert len(intel_before) == 2, intel_before                  # both live going in
    assert intel_after == intel_before                           # entity AND file_path unchanged


def test_apply_retires_nested_variant_subdir_rows_without_keeping_the_variant(tmp_path):  # clause 4
    root, db, before, _ = _applied_index(tmp_path)
    assert ("vllm", "vllm/experiments/scan.md") in before         # the row the loop must retire
    st = KGStore(db)
    rows = _live_rows(st)
    st.close()
    assert ("vllm", "vllm/experiments/scan.md") not in rows
    assert ("vLLM", "vLLM/experiments/scan.md") in rows           # moved, and renamed in the index
    assert (root / "vLLM/experiments/scan.md").exists()
    assert not [p for e, p in rows if e == "vllm"]


def test_apply_rewrites_edges_and_records_revertable_pairs(tmp_path):
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    r = _run(root, db, out, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(next(out.glob("entity-merges-applied-*.json")).read_text())
    pairs = rep["edge_rewrites"]["vllm"]
    assert len(pairs) == 1
    st = KGStore(db)
    # the variant's edge is expired and folded onto the canonical's existing one
    assert st.edges.active(either="vllm") == []
    assert len(st.edges.active(either="vLLM")) == 1
    old_id, new_id = pairs[0]
    assert st.edges.by_id(old_id)["expired_at"] and "merge" in st.edges.by_id(old_id)["expired_reason"]
    # history survives: the pre-merge edge is still readable
    assert st.edges.by_id(old_id)["source"] == "vllm"
    st.close()


def test_apply_is_one_transaction(tmp_path, monkeypatch):
    """A failure partway through the merge leaves the store untouched."""
    import importlib.util as _iu
    spec = _iu.spec_from_file_location("ers_txn", SWEEP)
    mod = _iu.module_from_spec(spec); sys.modules["ers_txn"] = mod; spec.loader.exec_module(mod)
    root, db = _tree(tmp_path)
    st = KGStore(db)
    plan = mod.build_plan(st.edges.active(), {d.name for d in root.iterdir()}, allowed_tiers=["CASE"])
    boom = [0]
    def explode(*a, **k):
        boom[0] += 1
        raise RuntimeError("kill -9 equivalent")
    monkeypatch.setattr(st.edges, "rewrite_endpoint", explode)
    with pytest.raises(RuntimeError):
        mod.apply_merges(plan, st, root, False, existing_dirs={d.name for d in root.iterdir()})
    assert boom[0] == 1
    assert st.aliases.resolve("vllm") is None       # the alias write rolled back too
    assert len(st.edges.active(either="vllm")) == 1
    assert (root / "vllm").exists()                 # files never moved
    st.close()


# ── merged facts carry the canonical's tag, and remember where they came from ─

def test_retag_fact_file_rewrites_entity_and_stamps_origin(tmp_path):
    f = tmp_path / "Inner Voice-state.md"
    fm = {"type": "facts", "entity": "Inner Voice System", "category": "state",
          "facts": [{"entity": "Inner Voice System", "fact": "two-brain critique"},
                    {"entity": "Inner Voice", "fact": "already canonical"}]}
    f.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# old body\n")
    assert ers.retag_fact_file(f, "Inner Voice System", "Inner Voice") == 2
    out = yaml.safe_load(f.read_text().split("---")[1])
    assert out["entity"] == "Inner Voice"
    assert out["facts"][0]["entity"] == "Inner Voice" and out["facts"][0]["merged_from"] == "Inner Voice System"
    assert "merged_from" not in out["facts"][1]
    assert "**Entity:** Inner Voice" in f.read_text()
    assert ers.retag_fact_file(f, "Inner Voice System", "Inner Voice") == 0   # idempotent

def test_apply_leaves_no_contamination_behind(tmp_path):
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    r = _run(root, db, out, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    merged = yaml.safe_load((root / "vLLM" / "vLLM-state.md").read_text().split("---")[1])
    tags = {f["entity"] for f in merged["facts"]}
    assert tags == {"vLLM"}, tags
    assert any(f.get("merged_from") == "vllm" for f in merged["facts"])
    sys.path.insert(0, str(ROOT / "scripts" / "memory")); import kg_hygiene
    assert kg_hygiene.contamination(root)["dirs"] == 0


def test_two_dry_runs_on_one_day_keep_both_plans(tmp_path):
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    assert _run(root, db, out).returncode == 0
    import time; time.sleep(1.1)
    assert _run(root, db, out).returncode == 0
    plans = [q for q in out.glob("entity-merges-*.jsonl") if not q.is_symlink()]
    assert len(plans) == 2, "a second dry-run must not overwrite the first plan"


# ── #475: an apply has to be attributable, and the guard has to survive it ───
#
# #475's acceptance is checked with SQL against the alias table, and two of its
# clauses asked for things the apply path could not produce however many clean
# applies ran: rows carried no `report_path`, and the report carried counts but
# not the state of the two switches built to stop a 2026-09-03 repeat.

VOICE = {"Voice-Loop", "Voice Pipeline", "voice"}


def _applied_report(out: Path) -> dict:
    reports = sorted(out.glob("entity-merges-applied-*.json"))
    assert reports, "the apply wrote no report"
    return json.loads(reports[-1].read_text())


def test_apply_stamps_the_apply_report_into_every_alias_row(tmp_path):
    """"Which run said this surface means that entity" must be answerable from
    the store itself. `Aliases.set` has always taken a `report_path`; the sweep's
    apply call site never passed one, so
    `SELECT COUNT(*) FROM aliases WHERE report_path IS NOT NULL` stayed 0 even
    after a perfect apply — #475 clause 1 could never go green."""
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    r = _run(root, db, out, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    st = KGStore(db)
    try:
        row = next((a for a in st.aliases.for_canonical("vLLM") if a["surface"] == "vllm"), None)
    finally:
        st.close()
    assert row is not None and row["origin"] == "sweep"
    assert row["report_path"], "an apply-origin alias row with no report behind it"
    rep = Path(row["report_path"])
    assert rep.is_file(), f"alias provenance points at a report that does not exist: {rep}"
    assert json.loads(rep.read_text())["alias_provenance"]["report_path"] == str(rep)


def test_apply_report_records_the_switches_and_what_it_applied(tmp_path):
    """The 2026-09-03 apply fused 151 entities and left no record that it ran
    with the protections off. The report now has to say so itself, because a
    reader who cannot tell a guarded apply from a bypassed one cannot audit it."""
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    (out / "graph-baseline.json").write_text(json.dumps({"active_edges": 10_000}))
    r = _run(root, db, out, "--apply", "--allow-degraded")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = _applied_report(out)
    assert rep["safety"]["allow_degraded"] is True
    assert rep["safety"]["degraded_graph"] == "bypassed: --allow-degraded"
    assert rep["safety"]["no_gate"] is True          # _run always passes --no-gate
    assert rep["safety"]["gate_verdict"].startswith("skipped: --no-gate")
    assert rep["applied_clusters"] >= 1, rep["applied_clusters"]
    assert rep["applied_merges"] == len(rep["variant_to_canonical"]) >= 1
    assert rep["alias_provenance"]["origin"] == "sweep"


def test_the_backfilled_aliases_exclude_the_permanent_noise_shapes(tmp_path):
    """`--rebuild-aliases` removes the exhaust the 09-03 migration carried in, and
    leaves every routing row that actually works. Both halves are asserted because
    the first version of the prune failed only the second one: filtering alias
    surfaces by `looks_like_junk_entity` measured 263 deletions on the live store,
    including `.openclaw → OpenClaw` and `Autonomy task #24 → Autonomy Task #24` —
    real rows, whose removal would have lowered the very coverage ratio #475 exists
    to raise. The boundary that survived is `_is_exhaust_surface`: a code filename
    or a call fragment, and nothing that merely looks unusual.

    Seeded, not merely observed. The first version of this test inspected only the
    rows the apply had just written, and could not fail: `_tree` holds
    `vLLM/vllm/Intel/Intel Pipeline` (none exhaust-shaped) and `build_plan` filters
    junk out upstream at :332, so the apply set was clean by construction. The rows
    that matter are the ones already in the store when a run starts — 3,919 of them
    came from the JSON table, and `--rebuild-aliases` is the only path that touches
    them.
    """
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    exhaust = ("server.py", "kg_store.py", "query()")     # this pipeline's own debris
    working = (".openclaw",                               # dotted, but a real name
               "Neo et al. (Interpreting Vision Grounding in VLMs)")
    st = KGStore(db)
    try:
        for s in exhaust + working:          # inherited migration rows
            st.aliases.set(s, "vLLM", kind="semantic", origin="migration")
        assert all(st.aliases.resolve(s) == "vLLM" for s in exhaust + working), "seed did not land"
    finally:
        st.close()

    assert _run(root, db, out, "--apply", "--rebuild-aliases").returncode == 0

    st = KGStore(db)
    try:
        rows = list(st.aliases.rows())
    finally:
        st.close()
    surfaces = {r["surface"] for r in rows}
    assert [r["surface"] for r in rows if r["origin"] == "sweep"], "the apply wrote no rows to check"
    assert not (surfaces & set(exhaust)), f"--rebuild-aliases kept {sorted(surfaces & set(exhaust))}"
    assert set(working) <= surfaces, f"the prune ate working routing rows: {set(working) - surfaces}"
    assert not [s for s in surfaces if ers._is_exhaust_surface(s)], sorted(surfaces)


def test_a_clean_apply_does_not_report_itself_as_bypassed(tmp_path):
    """The other half: `safety` must not read as bypassed when nothing was.

    Seeded with a baseline on disk (2 active edges, the store's own count) so this
    actually exercises the name it carries. Without the file the run is not clean,
    it is unmeasured — #1557 — and `test_a_store_with_edges_and_no_baseline_file_
    still_applies_and_says_unmeasured` is the test for that case."""
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    (out / "graph-baseline.json").write_text(json.dumps({"active_edges": 2}))
    assert _run(root, db, out, "--apply").returncode == 0
    safety = _applied_report(out)["safety"]
    assert safety["allow_degraded"] is False
    assert safety["degraded_graph"] == "not degraded"
    assert safety["baseline_measured"] is True
    assert safety["active_edges"] == 2


def test_a_kill_between_the_transaction_and_the_report_leaves_no_dangling_pointer(tmp_path, monkeypatch):
    """A run can die anywhere. If it dies after the store transaction but before
    the report is written, every alias row it committed names a report that was
    never created — and the disposition audit's only word for that is
    `report_missing`, on a merge that genuinely happened. `claim_report` puts the
    file down first; this simulates the kill at that exact seam.

    In-process on purpose: the point is the ORDER inside main(), which a
    subprocess run cannot interrupt mid-way. `report_path` is still what the rows
    carry, so the assertion reads the store, not the log.
    """
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    real_dump = json.dump

    def dying_dump(obj, fh, *a, **kw):        # dies writing the FINAL report, nothing earlier
        if "entity-merges-applied" in getattr(fh, "name", ""):
            raise OSError("simulated kill after the store transaction")
        return real_dump(obj, fh, *a, **kw)

    monkeypatch.setattr(sys, "argv", [str(SWEEP), "--facts-dir", str(root), "--db", str(db),
                                     "--out-dir", str(out), "--no-gate", "--apply"])
    monkeypatch.setattr(json, "dump", dying_dump)
    with pytest.raises(OSError):
        ers.main()
    monkeypatch.setattr(json, "dump", real_dump)

    st = KGStore(db)
    try:
        rows = [r for r in st.aliases.rows() if r["origin"] == "sweep"]
    finally:
        st.close()
    assert rows, "the apply wrote no rows, so the seam was never exercised"
    for r in rows:
        assert r["report_path"], f"{r['surface']!r} committed with no provenance at all"
        p = Path(r["report_path"])
        assert p.is_file(), f"alias provenance points at nothing: {p}"
        claimed = json.loads(p.read_text())
        assert claimed["report_status"] == "started", "a killed run must not claim completion"
        assert claimed["applied_clusters"] is None, "a killed run must not claim a count"


def test_the_gate_cache_path_is_plumbed_to_the_gate(tmp_path, monkeypatch):
    """The gate reads a cache under `_pipeline/memory-graph` by default. A test, or
    a dry-run on someone's laptop, that consults it is reading verdicts the real
    pipeline earned — and a verdict cache written by a run nobody authorized is
    worse: it feeds future applies. `--gate-cache` exists so a run can be hermetic;
    this pins the flag actually reaching the constructor."""
    seen = {}

    class GateSpy:
        def __init__(self, root, cache_path=None):
            seen["root"], seen["cache_path"] = root, cache_path

        def verdict(self, a, b):
            return {"decision": "REVIEW", "judges": {}}

    fake = types.ModuleType("entity_semantic_gate")
    fake.SemanticGate = GateSpy
    monkeypatch.setitem(sys.modules, "entity_semantic_gate", fake)
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    cache = tmp_path / "hermetic-verdicts.jsonl"
    monkeypatch.setattr(sys, "argv", [str(SWEEP), "--facts-dir", str(root), "--db", str(db),
                                     "--out-dir", str(out), "--apply",
                                     "--gate-cache", str(cache)])
    assert ers.main() == 0
    assert seen["cache_path"] == cache, "the flag did not reach the gate"


def test_the_default_gate_cache_is_left_alone_when_a_run_names_its_own(tmp_path, monkeypatch):
    """The other direction: unset must stay the pipeline's own cache, or the flag
    silently breaks the nightly run it was added to protect."""
    seen = {}

    class GateSpy:
        def __init__(self, root, cache_path=None):
            seen["cache_path"] = cache_path

        def verdict(self, a, b):
            return {"decision": "REVIEW", "judges": {}}

    fake = types.ModuleType("entity_semantic_gate")
    fake.SemanticGate = GateSpy
    monkeypatch.setitem(sys.modules, "entity_semantic_gate", fake)
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    monkeypatch.setattr(sys, "argv", [str(SWEEP), "--facts-dir", str(root), "--db", str(db),
                                     "--out-dir", str(out), "--apply"])
    assert ers.main() == 0
    assert seen["cache_path"] is None        # the gate's own default, untouched


def test_an_apply_with_the_gate_on_reports_itself_as_not_bypassed(tmp_path):
    """`_run` hardcodes --no-gate, so every other end-to-end test here can only
    ever produce `no_gate: true`. The state the 09-03 apply needed to be caught in
    — a real run, gate up, saying so — has to be produced by the CLI too, or the
    report's most important field is only ever exercised in its bypassed form.
    --tiers excludes the suffix tier, so the gate is constructed and consulted
    nowhere: no judge is asked, no HTTP. `--gate-cache` keeps even the read off the
    production verdict cache — a run that reads verdicts the live pipeline earned
    is not measuring the gate, it is replaying it."""
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    r = subprocess.run([sys.executable, str(SWEEP), "--facts-dir", str(root), "--db", str(db),
                        "--out-dir", str(out), "--apply", "--tiers", "PUNCT,CASE",
                        "--gate-cache", str(tmp_path / "hermetic-verdicts.jsonl")],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    safety = _applied_report(out)["safety"]
    assert safety["no_gate"] is False, safety
    assert safety["gate_verdict"].startswith("ran"), safety
    assert safety["allow_degraded"] is False


def test_safety_record_names_the_gate_verdict_whichever_way_the_gate_ran():
    """`gate_verdict` is the one line a reviewer reads to know whether the suffix
    tier was judged at all, so each way the gate can go has to read differently."""
    g = ers.safety_record(True, False, None, {}, None)
    assert g["no_gate"] is True and g["gate_verdict"].startswith("skipped: --no-gate")
    g = ers.safety_record(False, False, None, {}, None)
    assert g["gate_verdict"].startswith("unavailable")
    assert g["degraded_graph"] == "not degraded" and g["allow_degraded"] is False
    g = ers.safety_record(False, False, object(),
                          {"gate_stats": {"asked": 4, "same": 2, "review": 2}}, None)
    assert g["gate_verdict"] == "ran: 4 suffix pairs judged, 2 SAME, 2 to review"
    g = ers.safety_record(False, True, object(), {}, "graph is degraded")
    assert g["allow_degraded"] is True
    assert g["degraded_graph"] == "bypassed: --allow-degraded"


# ── the 06f0e41 guard, pinned against the apply path itself ──────────────────

def test_the_name_shape_guard_still_refuses_the_09_03_shape():
    """06f0e41 removed the 0-degree shortcut: `X`, `X Loop` and `X Pipeline`
    share a normalization, but only the gate may say they are one system. This is
    the exact condition that fused 151 entities — every variant at degree 0."""
    assert ers.cluster_tier(["Voice-Loop", "Voice Pipeline", "voice"]) == "AMBIGUOUS"
    assert (ers.normalize_full("Voice-Loop") == ers.normalize_full("Voice Pipeline")
            == ers.normalize_full("voice") == "voice")
    # `X Loop` is not even shape-safe: the ambiguous suffix keeps it out of SAFE.
    assert ers.cluster_tier(["Voice-Loop", "voice"]) == "AMBIGUOUS"
    # `X` inside `X Pipeline` IS shape-safe, and still refuses without a gate —
    # this is the shortcut that fused 151 entities, at their exact 0-degree shape.
    ok, why = ers.decide_merge("SUFFIX_SAFE", "Voice Pipeline", ["Voice Pipeline", "voice"],
                              {"Voice Pipeline": 0, "voice": 0})
    assert ok is False and "gate" in why, why


def test_an_apply_leaves_voice_loop_voice_pipeline_and_voice_separate(tmp_path):
    """#475 clause 4: the backfill must not fuse them. Run through the real
    apply, not just the classifier, because the guard's whole point is what ends
    up in the alias table."""
    root = tmp_path / "facts"; root.mkdir()
    for name in sorted(VOICE):
        d = root / name; d.mkdir()
        fm = {"type": "facts", "entity": name, "category": "state",
              "facts": [{"entity": name, "fact": f"{name} exists.", "confidence": 0.9,
                         "category": "state"}]}
        (d / f"{name}-state.md").write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name} - state\n")
    db, out = tmp_path / "kg.sqlite", tmp_path / "out"; out.mkdir()
    existing = {d.name for d in root.iterdir() if d.is_dir()}
    st = KGStore(db)
    try:
        for name in sorted(VOICE):
            st.entities.register(name)
        # No edges: on 2026-09-03 every entity looked disconnected, which is the
        # condition the old 0-degree shortcut merged under. And a gate that says
        # SAME to everything: without it `gate=None` alone would keep every
        # suffix cluster out of the plan and the test would pass no matter how
        # badly the name-shape guard were broken.
        class Permissive:
            def verdict(self, a, b):
                return {"decision": "SAME", "judges": {}}

        assert ers.build_plan([], {"Voice Pipeline", "voice"}, gate=Permissive())["safe_clusters"] == 1, \
            "a permissive gate must be able to fuse `X` with `X Pipeline`, or this test proves nothing"
        plan = ers.build_plan([], existing, gate=Permissive())
        merged = {mm["variant"] for c in plan["safe_merges"] for mm in c.get("merges", [])}
        assert not (merged & VOICE), f"the plan would fuse {merged & VOICE}"
        ers.apply_merges(plan, st, root, rebuild_aliases=True, existing_dirs=existing,
                         entities=plan.get("all_entities", []),
                         report_path=str(out / "applied.json"))
        rows = [r for r in st.aliases.rows() if r["surface"] in VOICE]
        assert rows == [], f"the backfill wrote an alias for {rows}"
        # Every reader goes through `entity_naming.normalize`, which is
        # `store().resolve(name) or name`. All three must come back unchanged.
        # `resolve` returns the name itself or None; `normalize(name) == name` is
        # the claim every reader actually depends on, and it is what fails if any
        # surface of one of these becomes an alias for another. The `rows == []`
        # assertion above is what proves nothing mapped them at all.
        from app import entity_naming, kg_store
        kg_store.configure(db)
        try:
            for name in sorted(VOICE):
                assert st.resolve(name) in (None, name)
                assert entity_naming.normalize(name) == name
        finally:
            kg_store.reset()
    finally:
        st.close()


# ── symbols are meaning, not punctuation ─────────────────────────────────────

def test_symbol_bearing_names_are_never_a_mechanical_merge():
    """The first automatic CASE/PUNCT apply (2026-09-16) merged `C` and `C#`
    into `C++`, `pass^k` into `pass@k`, `τ²-bench` into `Bench` and
    `BrowseComp+` into `BrowseComp`. A name whose symbols differ is a
    different name; it goes to hand review, never through the semantic gate."""
    for a, b in [("C", "C++"), ("C#", "C++"), ("C# SDK", "C++ SDK"), ("pass^k", "pass@k"),
                 ("τ²-bench", "Bench"), ("BrowseComp+", "BrowseComp"), ("Office Q&A", "Office QA")]:
        tier, why = ers.classify_pair(a, b)
        assert tier == "SUFFIX_AMBIGUOUS" and "symbols" in why, (a, b, tier)
    # Separators still merge mechanically.
    for a, b in [("SWE-Bench", "SweBench"), ("Pierluca D'Oro", "Pierluca Doro"),
                 ("Amodio et al. (2007)", "Amodio et al. 2007"), ("Brando...", "Brando"),
                 ("Context forking", "Context Forking")]:
        assert ers.classify_pair(a, b)[0] in ("PUNCT", "CASE"), (a, b)
    assert ers.symbol_residue("C++") == "++" and ers.symbol_residue("τ²-bench") == "²τ"
    assert ers.symbol_residue("Amodio et al. (2007)") == ""


def test_build_plan_keeps_c_cpp_and_csharp_apart():
    dirs = {"C", "C++", "C#", "pass^k", "pass@k", "Context forking", "Context Forking"}
    plan = ers.build_plan(_edges([("C++", "Alan"), ("pass@k", "Alan")]), dirs, gate=None)
    safe_variants = {v for c in plan["safe_merges"] for v, _deg in c["variants"]}
    assert not ({"C", "C#", "C++", "pass^k", "pass@k"} & safe_variants), plan["safe_merges"]
    assert {"Context forking", "Context Forking"} <= safe_variants
    reviewed = {c["canonical"]: {v for v, _deg in c["variants"]} for c in plan["ambiguous"]}
    assert reviewed["C++"] == {"C", "C++", "C#"} and reviewed["pass@k"] == {"pass^k", "pass@k"}
    assert all(c["tier"] == "SUFFIX_AMBIGUOUS" for c in plan["ambiguous"])


# ── the fact-move half is resumable (#1558) ──────────────────────────────────
#
# The apply has two halves and a killable gap between them: aliases and every
# edge rewrite commit in one store transaction, and the fact files move AFTER it,
# because the filesystem cannot join that transaction. What makes the gap
# survivable is a journal — each variant's dir outcome written into the claimed
# apply report as it completes — plus `--resume <that report>` to finish what the
# kill left pending. These tests kill a run inside the window in-process, because
# a subprocess cannot be interrupted at a chosen line, and then finish it with the
# real CLI flag.

def _pair_facts(entity: str, texts: list[str]) -> str:
    fm = {"type": "facts", "entity": entity, "category": "state",
          "facts": [{"entity": entity, "fact": t, "confidence": 0.9, "category": "state"}
                    for t in texts]}
    return f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity} - state\n"


def _two_pair_tree(tmp_path):
    """Facts and a store holding TWO case pairs, each variant dir with TWO fact
    files: two variants so a kill can land *between* them, two files each so a
    kill can land *inside* one. The canonical of each pair carries two edges and
    the variant one, which is what makes degree — not a coin-flip — pick it.

    The fact files are indexed (`facts_idx.reindex`) because that is how a live
    tree arrives: the apply retags only a file its own index attributes to the
    variant, so in an unindexed fixture `retag_fact_file` is never reached and a
    kill patched onto it silently never fires.
    """
    root = tmp_path / "facts"
    db = tmp_path / "kg.sqlite"
    st = KGStore(db)
    for canonical, variant in (("vLLM", "vllm"), ("Gemma", "gemma")):
        (root / canonical).mkdir(parents=True)
        (root / canonical / f"{canonical}-state.md").write_text(
            _pair_facts(canonical, [f"{canonical} serves 40 req/s."]))
        (root / variant).mkdir(parents=True)
        (root / variant / f"{variant}-state.md").write_text(
            _pair_facts(variant, [f"{variant} restarts nightly at 04:00."]))
        (root / variant / f"{variant}-goal.md").write_text(
            _pair_facts(variant, [f"{variant} aims to halve cold start."]))
        st.entities.register(canonical); st.entities.register(variant)
        st.edges.add({"source": canonical, "target": "Ray", "type": "mentions"}, origin="test")
        st.edges.add({"source": canonical, "target": "Triton", "type": "mentions"}, origin="test")
        st.edges.add({"source": variant, "target": "Triton", "type": "mentions"}, origin="test")
    st.facts_idx.reindex([p for p in root.rglob("*.md")], root=root)
    st.close()
    return root, db


def _killed_apply(tmp_path, monkeypatch, die_after_retags: int):
    """Run `apply_merges` until it dies `die_after_retags` fact FILES into the
    move half — 2 lands between the two variants, 1 lands inside the first one.
    Returns (root, db, out_dir, claimed report path)."""
    root, db = _two_pair_tree(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    st = KGStore(db)
    dirs = {d.name for d in root.iterdir()}
    plan = ers.build_plan(st.edges.active(), dirs, allowed_tiers=["CASE"])
    assert len(plan["safe_merges"]) == 2, plan["safe_merges"]
    report = out / "entity-merges-applied-2026-09-26-20260926T000000Z.json"
    ers.claim_report(report, out / "entity-merges-latest.jsonl", "2026-09-26", "20260926T000000Z")
    real, seen = ers.retag_fact_file, [0]

    def dying(path, old, new):
        seen[0] += 1
        if seen[0] > die_after_retags:
            raise KeyboardInterrupt("killed in the fact-move window")
        return real(path, old, new)

    monkeypatch.setattr(ers, "retag_fact_file", dying)
    with pytest.raises(KeyboardInterrupt):
        ers.apply_merges(plan, st, root, rebuild_aliases=False, existing_dirs=dirs,
                         report_path=str(report))
    st.close()
    return root, db, out, report


def _fact_texts(root: Path) -> list[str]:
    """Every fact text on the tree, path by path — the census that catches a
    resume applying the same move twice."""
    return [f["fact"] for p in sorted(root.rglob("*.md"))
            for f in (yaml.safe_load(p.read_text().split("---")[1]) or {}).get("facts") or []]


def test_a_kill_in_the_move_window_journals_what_already_moved(tmp_path, monkeypatch):
    """Clause 1: a run interrupted after the store commit leaves a report that
    still PARSES, still says `started` and claims no count, and already lists
    every `dir_operations` entry it finished — so a later reader, and `--resume`,
    can tell the moved dirs from the untouched ones. Before #1558 `dir_ops` lived
    only in memory and reached disk in the final report a killed run never
    writes, so the stub said nothing about a merge that had half happened."""
    root, db, out, report = _killed_apply(tmp_path, monkeypatch, die_after_retags=2)
    doc = json.loads(report.read_text())              # an unparseable stub fails the test here
    assert doc["report_status"] == "started", "a killed run must not claim completion"
    assert doc["applied_clusters"] is None, "a killed run must not claim a count"
    done = [op for op in doc["dir_operations"] if op.get("done")]
    assert len(done) == 1, f"exactly one variant had finished: {doc['dir_operations']}"
    assert not (root / done[0]["variant"]).exists(), "the journal claims a dir still on disk"
    left = set(doc["variant_to_canonical"]) - {op["variant"] for op in done}
    assert len(left) == 1 and (root / left.pop()).exists(), \
        "the unfinished variant is still on disk and must not be in the journal"
    assert doc["edge_rewrites"], "the store half committed, so its ids belong in the journal"


def test_resume_finishes_only_the_dirs_the_kill_left_pending(tmp_path, monkeypatch):
    """Clause 2: `--resume <apply report>` replays only the `dir_operations` not
    yet recorded as done. This kills INSIDE the first variant — after one of its
    two files had moved, so nothing was journaled — and the resume must complete
    both variants: no file left under either variant dir, and every fact counted
    exactly once under its canonical (a re-applied move would show up as a
    duplicate text or a `_dup` sidecar)."""
    root, db, out, report = _killed_apply(tmp_path, monkeypatch, die_after_retags=1)
    stub = json.loads(report.read_text())
    assert [op for op in stub["dir_operations"] if op.get("done")] == [], \
        "the kill landed inside the first variant, so nothing may be journaled done"

    r = _run(root, db, out, "--resume", str(report))
    assert r.returncode == 0, r.stdout + r.stderr
    done = json.loads(report.read_text())
    assert done["report_status"] == "complete"
    assert {op["variant"] for op in done["dir_operations"] if op.get("done")} \
        == set(done["variant_to_canonical"]), done["dir_operations"]
    for variant in done["variant_to_canonical"]:
        vdir = root / variant
        assert not vdir.exists() or not list(vdir.rglob("*")), f"{variant} still holds files"
    assert not list(root.rglob("*_dup*")), "the resume wrote a sidecar instead of merging"
    assert sorted(_fact_texts(root)) == sorted([
        "vLLM serves 40 req/s.", "vllm restarts nightly at 04:00.",
        "vllm aims to halve cold start.", "Gemma serves 40 req/s.",
        "gemma restarts nightly at 04:00.", "gemma aims to halve cold start.",
    ]), "each moved fact must be counted exactly once under the canonical"


# ── a fact file whose OWN text carries a fence (#2138) ───────────────────────
#
# Two shapes, both produced by the fact writers themselves and both present in the
# live corpus (`Zero-width assertion regex bug/…-skill.md` reads 4 facts anchored /
# 2 naive, `False Absence Guard Pattern/…-state.md` 6 / 4):
#
#   * `yaml.dump` renders a multi-line fact as a single-quoted scalar whose
#     continuation lines are INDENTED, so a fact quoting a fence line keeps it —
#     indented, the only place YAML allows one inside a value;
#   * a fact quoting `` `---segment:` `` mid-sentence keeps the fence inline.
#
# Both sit after the opening `---`, so `text.split("---", 2)` cut the block in
# half there. What made that a data-loss bug rather than a parse error is that the
# truncated slice LOADS: it answers `{type, entity, category, facts}` with the
# half-cut fact still in the list and `source_doc`/`last_updated` gone. Every
# guard on this module's write path is an `if not fm`, so none of them fired, and
# the writer re-dumped the truncated dict over the file — 4 facts in, 2 out.

FENCE_ENTITY = "Fence Fusion Guard"
FENCE_SRC = "knowledge/software/fact-file-fence-handling.md"
FENCE_FACTS = [
    ("stat-001", 1, "The extractor anchors its insert on the opening fence line only.", 1.0),
    ("stat-002", 2, "Anchored with `$`, the insert lands inside `---segment:` and swallows the next key.", 0.9),
    ("stat-003", 3, "A re-dump reproduces the hazard line:\n---\nand a cut at the first fence loses the tail.", 0.9),
    ("stat-004", 4, "The anchored closing fence is the only rule that survives a re-dump.", 0.8),
]


def _fenced_fm(entity: str = FENCE_ENTITY, category: str = "state",
               facts: list | None = None) -> dict:
    """Front matter for a fact file whose facts carry a fence in their own text."""
    return {
        "type": "facts", "entity": entity, "category": category,
        "facts": [{"id": fid, "entity": entity, "fact": text, "confidence": conf,
                   "category": category, "provenance": "EXTRACTED",
                   "created_at": f"2026-09-23T05:18:39.0620{n:02d}+00:00",
                   "source_doc": FENCE_SRC}
                  for fid, n, text, conf in (facts or FENCE_FACTS)],
        "source_doc": FENCE_SRC, "last_updated": "2026-09-30T05:29:51.000000",
    }


def _write_fence_file(dir_: Path, fm: dict, entity: str = FENCE_ENTITY,
                      category: str = "state") -> Path:
    """Write it the way the fact writers do: `yaml.dump` between two bare fences."""
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / f"{entity}-{category}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False, allow_unicode=True)}---\n\n"
                 f"# {entity} - {category}\n", encoding="utf-8")
    return p


def test_parse_frontmatter_finds_every_fact_the_store_reader_finds(tmp_path):   # clause 1
    p = _write_fence_file(tmp_path / FENCE_ENTITY, _fenced_fm())
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert any(ln.strip() == "---" and ln.startswith(" ") for ln in lines), \
        "fixture drift: no fact's own fence line survives inside its scalar any more"
    assert any("---" in ln and not ln.strip().startswith("---")
               for ln in lines), "fixture drift: no fact carries a fence inside its own prose"
    # The naive rule this replaces, recorded as the witness of what the file defeats:
    # the truncated slice PARSES, returning 2 of the 4 facts, the second one cut
    # mid-sentence, and no `source_doc`. `if not fm` cannot see that file.
    naive = yaml.safe_load(text.split("---", 2)[1]) or {}
    naive_texts = [f["fact"] for f in naive.get("facts") or []]
    assert len(naive_texts) == 2, f"fixture drift: the naive read no longer truncates: {naive_texts}"
    assert naive_texts[0] == FENCE_FACTS[0][2], naive_texts[0]
    assert FENCE_FACTS[1][2].startswith(naive_texts[1]), naive_texts[1]
    assert "source_doc" not in naive, naive

    fm, body = ers._parse_frontmatter(text)
    assert fm["facts"] == parse_fact_file(p)[1], \
        "the sweep's reader and the store's reader must name the same facts for one file"
    assert [f["id"] for f in fm["facts"]] == ["stat-001", "stat-002", "stat-003", "stat-004"]
    assert fm["source_doc"] == FENCE_SRC, "the keys sitting after the inner fence must come back"
    assert fm["last_updated"] == "2026-09-30T05:29:51.000000"
    assert body.startswith(f"# {FENCE_ENTITY} - state"), repr(body)


def test_merging_into_a_fence_bearing_destination_loses_no_fact(tmp_path):      # clause 2
    dst = _write_fence_file(tmp_path / FENCE_ENTITY, _fenced_fm())
    before = [f["fact"] for f in parse_fact_file(dst)[1]]
    assert len(before) == 4, before
    variant = f"{FENCE_ENTITY} System"
    src = _write_fence_file(tmp_path / variant,
                            _fenced_fm(entity=variant, facts=[
                                ("stat-001", 1, "A variant file: the insert is anchored on the newline, not the fence.", 0.9),
                                ("stat-002", 2, "A re-dump keeps a quoted fence inside the scalar, indented.", 0.8)]),
                            entity=variant)

    ers._merge_fact_file_into(src, dst)

    after = [f["fact"] for f in parse_fact_file(dst)[1]]   # anchored reader, post-write
    assert [t for t in before if t not in after] == [], "a destination fact vanished in the re-dump"
    assert len(after) == 6, f"4 destination facts plus the source's 2, got {len(after)}: {after}"
    fm = parse_fact_file(dst)[0]
    assert fm["entity"] == FENCE_ENTITY and fm["category"] == "state", fm
    assert fm["source_doc"] == FENCE_SRC, \
        "the file-level provenance key survived only in a read that never saw it"
    assert all(f.get("created_at") and f.get("source_doc") for f in parse_fact_file(dst)[1])


def test_apply_refuses_to_write_or_unlink_an_unparseable_fact_file(tmp_path):   # clause 3
    """A front matter with no closing fence cannot be read, so the apply must not
    write it and must not move it. Both files on the merge path are unparseable
    here: the variant's, which the loop used to merge-and-unlink, and the
    canonical's, which it used to re-dump from a dict it never parsed."""
    root, db = _tree(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    src = root / "vllm" / "vllm-state.md"            # the name this loop maps onto the dest below
    dst = root / "vLLM" / "vLLM-state.md"
    src.write_text("---\ntype: facts\nentity: vllm\ncategory: state\nfacts:\n"
                   "- entity: vllm\n  fact: front matter that never closes, so no rule can read it\n",
                   encoding="utf-8")
    dst.write_text("---\ntype: facts\nentity: vLLM\ncategory: state\nfacts: [\n", encoding="utf-8")
    src_bytes, dst_bytes = src.read_bytes(), dst.read_bytes()

    r = _run(root, db, out, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert src.exists(), "the merge unlinked a source file this run had never read"
    assert src.read_bytes() == src_bytes, "an unreadable source was rewritten"
    assert dst.read_bytes() == dst_bytes, "an unreadable destination was re-dumped from nothing"
    assert str(src) in r.stdout and str(dst) in r.stdout, \
        f"the apply output must name each skipped path:\n{r.stdout}"

    rep = json.loads(next(out.glob("entity-merges-applied-*.json")).read_text())
    entry = [op for op in rep["dir_operations"] if op["variant"] == "vllm"]
    assert len(entry) == 1, rep["dir_operations"]
    assert sorted(entry[0].get("skipped_unparseable") or []) == sorted([str(src), str(dst)]), \
        "the refusal has to survive into the apply report, not just the terminal"
    assert entry[0]["files_moved"] == 0
