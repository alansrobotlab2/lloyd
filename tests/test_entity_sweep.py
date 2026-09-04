"""entity-resolution-sweep.py — the rules that stop a name-shape match from
becoming a merge, and the guards around --apply."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from app.kg_store import KGStore  # noqa: E402
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
    assert ers.degraded_reason(5, 0) is None          # no baseline yet → nothing to compare

def test_update_baseline_keeps_the_max(tmp_path):
    p = tmp_path / "b.json"
    assert ers.update_baseline(100, p) == 100
    assert ers.update_baseline(50, p) == 100
    assert ers.load_baseline(p) == 100


# ── end to end on a temp tree ────────────────────────────────────────────────

def _tree(tmp_path):
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
    st.edges.add({"source": "vLLM", "target": "Ray", "type": "mentions"}, origin="test")
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
