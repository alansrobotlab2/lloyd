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

def test_compute_aliases_keeps_the_approved_suffix_alias():
    plan = {"safe_merges": [{"canonical": "Morning Briefing",
                             "merges": [{"variant": "Morning Briefing System"}]}]}
    al = ers.compute_aliases(plan, {"legacy noise system": "legacy noise", "Kept": "Kept"},
                             Path("/nonexistent"), rebuild_aliases=False,
                             existing_dirs={"Morning Briefing", "Kept"})
    assert al["morning briefing system"] == "Morning Briefing"
    assert al["Morning Briefing System"] == "Morning Briefing"
    assert al["Morning Briefing"] == "Morning Briefing"
    assert "legacy noise system" not in al        # inherited suffix noise is still pruned
    assert al["Kept"] == "Kept"


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
    rel = root / "_relationships.json"
    rel.write_text(json.dumps({"edges": [{"source": "vLLM", "target": "Ray", "type": "mentions", "expired_at": None},
                                         {"source": "vllm", "target": "Ray", "type": "mentions", "expired_at": None}]}))
    al = root / "entity-aliases.json"; al.write_text(json.dumps({n: n for n in ("vLLM", "vllm", "Intel", "Intel Pipeline")}))
    return root, rel, al

def _run(root, rel, al, out, *extra):
    cmd = [sys.executable, str(SWEEP), "--facts-dir", str(root), "--relationships", str(rel),
           "--aliases", str(al), "--out-dir", str(out), "--no-gate", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

def test_apply_refuses_on_a_degraded_graph(tmp_path):
    root, rel, al = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    (out / "graph-baseline.json").write_text(json.dumps({"active_edges": 10000}))
    r = _run(root, rel, al, out, "--apply")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUSING --apply" in r.stdout
    assert (root / "vllm").exists()                       # nothing moved
    assert json.loads(al.read_text())["vllm"] == "vllm"   # nothing rewritten
    # the override is explicit
    r2 = _run(root, rel, al, out, "--apply", "--allow-degraded")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert not (root / "vllm").exists()

def test_apply_writes_aliases_and_stamps_the_ledger(tmp_path):
    root, rel, al = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    r = _run(root, rel, al, out, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (root / "vllm").exists() and (root / "vLLM" / "vLLM-state.md").exists()
    assert (root / "Intel Pipeline").exists()                          # suffix cluster untouched
    aliases = json.loads(al.read_text()); assert aliases["vllm"] == "vLLM"
    reports = list(out.glob("entity-merges-applied-*.json")); assert len(reports) == 1
    rep = json.loads(reports[0].read_text())
    assert rep["ledger"]["argv"] and rep["ledger"]["pid"] and "cwd" in rep["ledger"]
    assert rep["tiers_allowed"] == ["CASE", "PUNCT", "SUFFIX_SAFE"]
    assert json.loads((out / "graph-baseline.json").read_text())["active_edges"] == 2
    # the plan carries the suffix cluster as review, with the reason
    plan_lines = [json.loads(l) for l in (out / list(out.glob("entity-merges-*.jsonl"))[0].name).read_text().splitlines() if l.strip()]
    amb = [c for c in plan_lines if c["status"] == "AMBIGUOUS"]
    assert amb and "semantic gate" in amb[0]["decision"]
