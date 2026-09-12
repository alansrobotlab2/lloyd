"""entity_merge_disposition.py — the audit that answers "who decided about this
reverted merge, and when" (#475 clause 5).

The 2026-09-03 revert left 151 variant→canonical pairs with no owner. Nine days
later no command on the machine could say which of them a later apply had redone
and which nobody had ever looked at again, which is how alias coverage fell from
16.4 % to 15.26 % with nobody touching it. These tests run the script as the CLI
the nightly operator runs, across the same process boundary the sweep's own
tests use — the audit is worth nothing if it only works in-process.
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from app.kg_store import KGStore  # noqa: E402
import entity_merge_disposition as emd  # noqa: E402

AUDIT = ROOT / "scripts" / "memory" / "entity_merge_disposition.py"
SWEEP = ROOT / "scripts" / "memory" / "entity-resolution-sweep.py"


def _graph(tmp_path, pairs, aliases=(), plan_clusters=(), report_plan_file=True):
    """out-dir + store shaped like a real `_pipeline/memory-graph`, minus live data."""
    out = tmp_path / "mg"; out.mkdir()
    db = tmp_path / "kg.sqlite"
    art = out / "entity-merges-reverted-20260903T174108Z.json"
    art.write_text(json.dumps({"tiers": ["SUFFIX_SAFE"],
                              "plan": [{"variant": v, "canonical": c} for v, c in pairs]}))
    report = out / "entity-merges-applied-2026-09-12-120000Z.json"
    plan = out / "entity-merges-2026-09-12-115900Z.jsonl"
    if plan_clusters:
        plan.write_text("".join(json.dumps(c) + "\n" for c in plan_clusters))
    report.write_text(json.dumps({"applied_clusters": len(aliases),
                                 "plan_file": str(plan) if report_plan_file else None}))
    st = KGStore(db)
    for surface, canonical, origin in aliases:
        st.aliases.set(surface, canonical, kind="punct", origin=origin,
                       report_path=str(report) if origin == "sweep" else None)
    st.close()
    return out, db, art


def _audit(db, out, dest, *extra):
    return subprocess.run(
        [sys.executable, str(AUDIT), "--db", str(db), "--out-dir", str(out),
         "--out", str(dest), *extra], capture_output=True, text=True, timeout=120)


PAIRS = [("vllm", "vLLM"), ("Multi-agent pipeline", "multi-agent"),
         ("vLLM pipeline", "vLLM")]


def test_reverted_pairs_reads_the_plan_the_revert_recorded(tmp_path):
    out, db, art = _graph(tmp_path, PAIRS)
    pairs = emd.reverted_pairs(json.loads(art.read_text()))
    assert pairs == [{"variant": v, "canonical": c} for v, c in PAIRS]


def test_the_audit_splits_the_reverted_pairs_three_ways(tmp_path):
    """One pair re-applied by a gated run, one pair a run looked at and refused,
    one pair nobody has touched since — reported as three counts, not as a
    coverage percentage somebody has to interpret."""
    out, db, _ = _graph(
        tmp_path, PAIRS,
        aliases=[("vllm", "vLLM", "sweep")],
        plan_clusters=[{"status": "SKIPPED", "canonical": "multi-agent",
                        "variants": [["Multi-agent pipeline", 0], ["multi-agent", 1]],
                        "decision": "SUFFIX_SAFE — semantic gate: review"}])
    dest = out / "disposition.json"
    r = _audit(db, out, dest)
    assert r.returncode == 1, r.stdout + r.stderr     # something is still unaccounted
    d = json.loads(dest.read_text())
    assert d["counts"] == {"applied": 1, "declined": 1, "unaccounted": 1}, d["counts"]
    assert d["pairs"] == 3 and Path(d["artifact"]).is_file()
    unaccounted = [e for e in d["entries"] if e["status"] == "unaccounted"]
    assert [e["variant"] for e in unaccounted] == ["vLLM pipeline"]
    assert d["entries"][0]["detail"]["report_path"]                  # provenance recorded
    assert "UNACCOUNTED: 1" in r.stdout, r.stdout


def test_an_inherited_alias_row_is_not_a_disposition(tmp_path):
    """The fragmentation case #475 is about: a carry-over row from the 09-03
    migration maps the pair, and it proves nothing about a gated apply. Counting
    it as applied would have shown this item green while the store was unchanged."""
    out, db, _ = _graph(tmp_path, PAIRS, aliases=[("vllm", "vLLM", "migration")])
    dest = out / "disposition.json"
    assert _audit(db, out, dest).returncode == 1
    d = json.loads(dest.read_text())
    assert d["counts"] == {"applied": 0, "declined": 0, "unaccounted": 3}, d["counts"]
    assert d["entries"][0]["detail"]["inherited_alias"] == {"canonical": "vLLM",
                                                            "origin": "migration"}


def test_a_revert_row_is_not_an_apply_either(tmp_path):
    """"Dispositioned" must not be satisfiable by the 09-03 revert writing the old
    mapping back — that is the state the audit starts from, not a decision."""
    out, db, _ = _graph(tmp_path, [("vllm", "vLLM")], aliases=[("vllm", "vLLM", "revert")])
    dest = out / "disposition.json"
    assert _audit(db, out, dest).returncode == 1
    assert json.loads(dest.read_text())["counts"]["applied"] == 0


def test_a_legacy_origin_row_is_not_an_apply_either(tmp_path):
    """`agent_mcp/_shared.py:640` writes aliases with origin="legacy" and no
    `report_path`, so a row it wrote names neither a run nor a gate. While
    INHERITED_ORIGINS omitted "legacy", such a row fell to the applied branch on
    the strength of a string being absent from a set — the audit would have
    reported #475's clause 5 green off an MCP-side write nobody authorized, with
    only a `provenance_missing` note against it. The rule: an origin counts as an
    apply only if the code that writes it also writes a report."""
    out, db, _ = _graph(tmp_path, [("vllm", "vLLM")], aliases=[("vllm", "vLLM", "legacy")])
    dest = out / "disposition.json"
    assert _audit(db, out, dest).returncode == 1
    d = json.loads(dest.read_text())
    assert d["counts"] == {"applied": 0, "declined": 0, "unaccounted": 1}, d["counts"]
    assert d["entries"][0]["detail"]["inherited_alias"]["origin"] == "legacy"


def test_an_apply_report_written_by_the_sweep_is_the_evidence_the_audit_reads(tmp_path):
    """The seam between the two programs, with no fixture in the middle.

    The sweep commits alias rows naming a report it is about to write, and the
    audit, in a separate process, resolves that path. Either side can change the
    name, the directory, or the tmp-file convention and the other will not see it:
    the audit's whole applied verdict is `Path(report).is_file()`. So run the real
    `--apply`, then the real audit over the same out-dir, and require the audit to
    accept the report the sweep left. Everything upstream of this test is a fixture
    someone wrote by hand; this is the only one where one program consumes the
    other's artifact.
    """
    facts = tmp_path / "facts"
    for name in ("vLLM", "vllm", "Intel", "Intel Pipeline"):
        d = facts / name; d.mkdir(parents=True)
        fm = {"type": "facts", "entity": name, "category": "state",
              "facts": [{"entity": name, "fact": f"{name} exists.", "confidence": 0.9,
                         "category": "state"}]}
        (d / f"{name}-state.md").write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name} - state\n")
    out = tmp_path / "mg"; out.mkdir()
    db = tmp_path / "kg.sqlite"
    out.joinpath("entity-merges-reverted-20260903T174108Z.json").write_text(json.dumps(
        {"tiers": ["CASE"], "plan": [{"variant": "vllm", "canonical": "vLLM"}]}))
    st = KGStore(db)
    for n in ("vLLM", "vllm", "Intel", "Intel Pipeline"):
        st.entities.register(n)
    st.edges.add({"source": "vLLM", "target": "Ray", "type": "mentions"}, origin="test")
    st.edges.add({"source": "vllm", "target": "Ray", "type": "mentions"}, origin="test")
    st.close()

    applied = subprocess.run(
        [sys.executable, str(SWEEP), "--facts-dir", str(facts), "--db", str(db),
         "--out-dir", str(out), "--no-gate", "--apply"],
        capture_output=True, text=True, timeout=180)
    assert applied.returncode == 0, applied.stdout + applied.stderr

    dest = out / "disposition.json"
    audit = _audit(db, out, dest)
    assert audit.returncode == 0, audit.stdout + audit.stderr   # the pair is accounted for
    d = json.loads(dest.read_text())
    assert d["counts"] == {"applied": 1, "declined": 0, "unaccounted": 0}, d["counts"]
    detail = d["entries"][0]["detail"]
    assert "provenance_missing" not in detail and "report_missing" not in detail, detail
    assert Path(detail["report_path"]).is_file()                # resolves across processes
    report = json.loads(Path(detail["report_path"]).read_text())
    assert report["report_status"] == "complete"
    assert report["applied_clusters"] >= 1
    # the tmp name the sweep writes through must never read as a second report
    assert len(list(out.glob("entity-merges-applied-*.json"))) == 1


def test_the_audit_is_green_only_when_every_pair_is_dispositioned(tmp_path):
    out, db, _ = _graph(
        tmp_path, PAIRS,
        aliases=[("vllm", "vLLM", "sweep"), ("vLLM pipeline", "vLLM", "sweep")],
        plan_clusters=[{"status": "AMBIGUOUS", "canonical": "multi-agent",
                        "variants": [["Multi-agent pipeline", 0], ["multi-agent", 1]],
                        "decision": "hand-review"}])
    dest = out / "disposition.json"
    r = _audit(db, out, dest)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(dest.read_text())
    assert d["counts"] == {"applied": 2, "declined": 1, "unaccounted": 0}
    assert "UNACCOUNTED: 0" in r.stdout


def test_an_absent_store_is_refused_not_audited_as_empty(tmp_path):
    """sqlite creates a file on open, so an audit pointed at a typo'd or
    not-yet-built path printed `applied: 0 / UNACCOUNTED: 151` about a store with
    nothing in it — a verdict indistinguishable from the true one, from an
    instrument whose entire purpose is attribution."""
    out, _db, _art = _graph(tmp_path, PAIRS)
    missing = tmp_path / "typo" / "kg.sqlite"
    dest = out / "should-not-exist.json"
    r = _audit(missing, out, dest)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "refusing to audit against an empty one" in r.stderr, r.stderr
    assert not missing.exists() and not missing.parent.exists()
    assert not dest.exists()


SWEEP = ROOT / "scripts" / "memory" / "entity-resolution-sweep.py"


def test_the_two_tools_default_to_the_same_report_directory():
    """The audit discovers which runs dispositioned a pair by globbing the sweep's
    own report directory and reading each report's `plan_file`. Two anchors — the
    sweep home-anchored, the audit checkout-anchored — and every applied pair reads
    as unaccounted from a worktree or canary run."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ers_out_dir", str(SWEEP))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert emd.OUT_DIR == mod.OUT_DIR == mod.BASELINE_PATH.parent


def test_a_real_sweep_apply_is_the_disposition_the_audit_reports(tmp_path):
    """The seam between the two processes: one run writes the alias row and the
    report it names, a separate later process reads the store and the report
    directory to answer "who applied this pair". Neither half can fake that join,
    and it is the only test that the stamped report_path is findable by whoever has
    to answer the question in production."""
    root = tmp_path / "facts"; root.mkdir()
    for name in ("vLLM", "vllm"):
        d = root / name; d.mkdir()
        fm = {"type": "facts", "entity": name, "category": "state",
              "facts": [{"entity": name, "fact": f"{name} exists.", "confidence": 0.9,
                         "category": "state"}]}
        (d / f"{name}-state.md").write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name}\n")
    db, out = tmp_path / "kg.sqlite", tmp_path / "mg"; out.mkdir()
    (out / "entity-merges-reverted-20260903T174108Z.json").write_text(
        json.dumps({"plan": [{"variant": "vllm", "canonical": "vLLM"}]}))
    r = subprocess.run([sys.executable, str(SWEEP), "--facts-dir", str(root), "--db", str(db),
                        "--out-dir", str(out), "--no-gate", "--apply"],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    a = _audit(db, out, out / "disposition.json")
    assert a.returncode == 0, a.stdout + a.stderr
    d = json.loads((out / "disposition.json").read_text())
    assert d["counts"] == {"applied": 1, "declined": 0, "unaccounted": 0}, d["counts"]
    assert d["entries"][0]["detail"]["origin"] == "sweep"
    assert Path(d["entries"][0]["detail"]["report_path"]).is_file()


def test_a_run_that_never_applied_proves_nothing_about_the_reverted_pairs(tmp_path):
    """Declination evidence comes from plans a run actually wrote, so the same
    plan text sitting in the directory unnamed by any apply report must not
    quietly turn unaccounted pairs into declined ones."""
    out, db, _ = _graph(tmp_path, [("Multi-agent pipeline", "multi-agent")],
                        plan_clusters=[{"status": "SKIPPED", "canonical": "multi-agent",
                                        "variants": [["Multi-agent pipeline", 0]],
                                        "decision": "hand-review"}],
                        report_plan_file=False)
    dest = out / "disposition.json"
    assert _audit(db, out, dest).returncode == 1
    assert json.loads(dest.read_text())["counts"]["declined"] == 0
    # ...and pointing the audit at that same plan explicitly does count.
    dest2 = out / "d2.json"
    _audit(db, out, dest2, "--plan", str(out / "entity-merges-2026-09-12-115900Z.jsonl"))
    assert json.loads(dest2.read_text())["counts"]["declined"] == 1
