"""`apply-classifications-v4.py --apply` reaches its fixed point in ONE invocation (#1257).

The skill said the apply step was idempotent, and a converged store is — but
one pass is not the fixed point. `_plan` reads eligibility off an index built
before the pass: when a pair holds an active eligible `mentions` edge in both
directions and the input carries one orientation's `related_to` record, pass 1
retypes the record's own direction and only pass 2 — whose index no longer
holds it — reaches the reversed-fold branch and retypes the other. Measured on
2026-09-20 over a copy of the live store: 145 → 11 → 0. Task #74 ran that as a
single `--apply` that printed "applied 145" and exited 0, so the second-pass plan
was never seen by anything but the next day's run.

The seeded stores below reproduce the mechanism with three edges. The script is
driven through its own `main()` (argv, `--db`, `--classified-dir`) so the test
exercises the pass loop and its report lines, not a re-implementation of them.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts" / "memory" / "apply-classifications-v4.py"

from app.kg_store import KGStore  # noqa: E402


@pytest.fixture(scope="module")
def applier():
    spec = importlib.util.spec_from_file_location("apply_v4_under_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["apply_v4_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _mentions(src: str, tgt: str) -> dict:
    return {"source": src, "target": tgt, "type": "mentions",
            "provenance": "EXTRACTED", "confidence": 0.5}


def _record(src: str, tgt: str, new_type: str = "related_to") -> dict:
    return {"source": src, "target": tgt, "new_type": new_type, "confidence": 0.9,
            "classified_at": "2026-09-24T00:00:00+00:00", "reason": "seeded",
            "direction_check": "correct", "verdict_adjustment": "none"}


@pytest.fixture
def seeded(tmp_path, monkeypatch, applier):
    """A store and a records dir; returns `run(*flags) -> (rc, stdout)`."""
    db = tmp_path / "kg.sqlite"
    classified = tmp_path / "classified"
    classified.mkdir()

    def seed(edges: list[dict], records: list[dict]) -> KGStore:
        st = KGStore(db)
        for e in edges:
            st.edges.add(e, origin="test")
        st.close()
        with (classified / "classified-v4-test.jsonl").open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        return KGStore(db)

    def run(*flags: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", [
            "apply-classifications-v4.py", "--classified-dir", str(classified),
            "--db", str(db), *flags])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = applier.main()
        return rc, buf.getvalue()

    return seed, run


def _active_eligible_mentions(st: KGStore) -> set[tuple[str, str]]:
    return {(e["source"], e["target"]) for e in st.edges.active(types=["mentions"])
            if e.get("provenance") == "EXTRACTED"}


# --- clause 1: one invocation, no eligible mentions edge left on either direction


def test_one_invocation_clears_both_directions_of_a_two_way_pair(seeded):
    seed, run = seeded
    st = seed([_mentions("A", "B"), _mentions("B", "A")], [_record("A", "B")])
    assert _active_eligible_mentions(st) == {("A", "B"), ("B", "A")}

    rc, out = run("--apply")
    assert rc == 0, out
    assert _active_eligible_mentions(st) == set(), (
        "a pass-2 plan was left behind: " + out)
    typed = {(e["source"], e["target"]) for e in st.edges.active(types=["related_to"])}
    assert typed == {("A", "B"), ("B", "A")}


def test_a_single_pass_is_not_the_fixed_point(seeded, applier, monkeypatch):
    """The counterfactual the loop exists for: with one pass, pass 2's plan is
    exactly the reversed direction, which the old script never saw."""
    seed, run = seeded
    st = seed([_mentions("A", "B"), _mentions("B", "A")], [_record("A", "B")])
    monkeypatch.setattr(applier, "MAX_PASSES", 1)
    rc, out = run("--apply")
    assert rc == 0
    assert _active_eligible_mentions(st) == {("B", "A")}


# --- clause 2: each pass reported, and the final line says the last pass planned zero


def test_the_report_names_every_pass_and_ends_on_a_zero_plan(seeded):
    seed, run = seeded
    seed([_mentions("A", "B"), _mentions("B", "A")], [_record("A", "B")])
    rc, out = run("--apply")
    assert rc == 0
    assert "v4 reclassification plan (pass 1):" in out
    assert "v4 reclassification plan (pass 2):" in out
    assert "v4 reclassification plan (pass 3):" in out
    assert "[info] pass 1: applied 1 retypes" in out
    assert "[info] pass 2: applied 1 retypes" in out
    last = [ln for ln in out.splitlines() if ln.strip()][-1]
    assert "converged: passes=3 upgrades_per_pass=[1, 1, 0] applied=2" in last, last
    assert last.endswith("last pass planned 0 upgrades"), last


def test_a_converged_store_reports_one_zero_pass(seeded):
    seed, run = seeded
    seed([_mentions("A", "B")], [_record("A", "B")])
    rc, out = run("--apply")
    assert rc == 0
    rc2, out2 = run("--apply")
    assert rc2 == 0
    last = [ln for ln in out2.splitlines() if ln.strip()][-1]
    assert "passes=1 upgrades_per_pass=[0] applied=0" in last, last
    assert "[info] no changes to apply" in out2


def test_a_dry_run_shows_the_first_pass_only_and_says_so(seeded):
    seed, run = seeded
    st = seed([_mentions("A", "B"), _mentions("B", "A")], [_record("A", "B")])
    rc, out = run("--dry-run")
    assert rc == 0
    assert "(pass 2)" not in out
    assert "pass 1 plans 1 upgrades; --apply repeats until a pass plans zero" in out
    assert _active_eligible_mentions(st) == {("A", "B"), ("B", "A")}


# --- clause 3: bounded, exit 0 at the cap, residual reported


def test_the_cap_stops_without_error_and_reports_the_residual(seeded, applier, monkeypatch):
    seed, run = seeded
    seed([_mentions("A", "B"), _mentions("B", "A")], [_record("A", "B")])
    monkeypatch.setattr(applier, "MAX_PASSES", 1)
    rc, out = run("--apply")
    assert rc == 0
    assert "converged" not in out
    warn = [ln for ln in out.splitlines() if ln.startswith("[warn] pass cap reached")]
    assert warn, out
    assert "passes=1 upgrades_per_pass=[1] applied=1; 1 upgrades still planned" in warn[0]


def test_the_shipped_cap_is_at_most_four(applier):
    assert 1 <= applier.MAX_PASSES <= 4


# --- clause 4: no pass expires an active non-mentions edge


def test_a_typed_edge_on_the_reverse_direction_survives_every_pass(seeded):
    """1,475 such rows were live on 2026-09-20: a prior v4 verdict on (D, C)
    whose reverse (C, D) is an eligible `mentions` edge. The loop must reach
    its fixed point without touching it — that is #1246's seam, not this one."""
    seed, run = seeded
    typed = {"source": "D", "target": "C", "type": "uses",
             "provenance": "EXTRACTED_CLASSIFIER_V4", "confidence": 0.8}
    st = seed([_mentions("C", "D"), typed, _mentions("A", "B"), _mentions("B", "A")],
              [_record("C", "D"), _record("A", "B")])
    before = {e["id"]: e for e in st.edges.active() if e["type"] != "mentions"}
    assert len(before) == 1

    rc, out = run("--apply")
    assert rc == 0, out
    for edge_id, was in before.items():
        now = st.edges.by_id(edge_id)
        assert now is not None and now["expired_at"] is None, (edge_id, was, now)
    assert st.edges.find_active("D", "C", "uses") is not None
    assert _active_eligible_mentions(st) == set()
