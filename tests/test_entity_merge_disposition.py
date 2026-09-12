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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from app.kg_store import KGStore  # noqa: E402
import entity_merge_disposition as emd  # noqa: E402

AUDIT = ROOT / "scripts" / "memory" / "entity_merge_disposition.py"


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
