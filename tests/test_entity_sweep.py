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
    """The alias set a backfill writes has to be entity-shaped: a surface the
    extractor is forbidden to treat as an entity (a filename, a bare code
    identifier) regrows the variants a merge just removed, which is the 08-26 shape
    rule (`e235e5a`) and the reason the 09-03 merge undid itself. Run through the
    real apply, because it is the apply that writes these rows."""
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    assert _run(root, db, out, "--apply", "--rebuild-aliases").returncode == 0
    st = KGStore(db)
    try:
        applied = [r["surface"] for r in st.aliases.rows() if r["origin"] == "sweep"]
    finally:
        st.close()
    assert applied, "the apply wrote no alias rows to check"
    assert not [s for s in applied if ers.looks_like_junk_entity(s)], applied


def test_a_clean_apply_does_not_report_itself_as_bypassed(tmp_path):
    """The other half: `safety` must not read as bypassed when nothing was."""
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    assert _run(root, db, out, "--apply").returncode == 0
    safety = _applied_report(out)["safety"]
    assert safety["allow_degraded"] is False
    assert safety["degraded_graph"] == "not degraded"


def test_an_apply_with_the_gate_on_reports_itself_as_not_bypassed(tmp_path):
    """`_run` hardcodes --no-gate, so every other end-to-end test here can only
    ever produce `no_gate: true`. The state the 09-03 apply needed to be caught in
    — a real run, gate up, saying so — has to be produced by the CLI too, or the
    report's most important field is only ever exercised in its bypassed form.
    --tiers excludes the suffix tier, so the gate is constructed and consulted
    nowhere: no judge is asked, no HTTP."""
    root, db = _tree(tmp_path); out = tmp_path / "out"; out.mkdir()
    r = subprocess.run([sys.executable, str(SWEEP), "--facts-dir", str(root), "--db", str(db),
                        "--out-dir", str(out), "--apply", "--tiers", "PUNCT,CASE"],
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
