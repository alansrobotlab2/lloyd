"""#557: the skill-matcher recall eval, ported without its embedding arm.

The labelled set and scorer came from branch automod/SM_20260911_055006
(0c5f61c). Alan's 2026-09-14 ruling kept them and dropped the geometric arm,
so these tests pin three things: the metric definitions (a hand-computed
fixture, so a change to what recall@5 means fails here), the corpus contract
#711 filters, and the arm's absence.

The `--compare` nodes at the bottom pin #4611: the graded drift is scored over
only the turn ids the baseline and today's set share, so adding labelled turns
can never read as matcher drift again. On 2026-11-26 `--compare` exited 1
claiming `-0.0572` while the move on the 51 ids the two sets shared was
`-0.0196` — every byte of the headline gap came from three new labelled ids
that score 0.0.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "eval" / "run_skill_match_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_skill_match_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIXTURE = [
    {"id": "f1", "turn": "t1", "expected_skills": ["a", "b"]},
    {"id": "f2", "turn": "t2", "expected_skills": ["c"]},
    {"id": "f3", "turn": "t3", "expected_skills": ["d"]},
    {"id": "f4", "turn": "t4", "expected_skills": [], "expected_empty": True},
    {"id": "f5", "turn": "t5", "expected_skills": [], "expected_empty": True},
]
RETURNED = {
    "t1": ["a", "x", "y"],                 # 1 of 2 expected
    "t2": [],                              # nothing at all: a no-match
    "t3": ["d", "e", "f", "g", "h", "i"],  # 6 returned; only the top 5 count
    "t4": ["z"],                           # names a skill on an empty turn
    "t5": [],
}


def test_metrics_on_a_hand_computed_fixture():
    ev = _load()
    out = ev.score(FIXTURE, lambda turn: RETURNED[turn])
    assert out["labelled_turns"] == 3
    assert out["expected_empty_turns"] == 2
    assert out["recall@5"] == pytest.approx(0.5)          # (0.5 + 0 + 1) / 3
    assert out["precision@5"] == pytest.approx(0.1333)    # (1/5 + 0 + 1/5) / 3
    assert out["hit@5"] == pytest.approx(0.6667)          # f1, f3
    assert out["no_match_rate"] == pytest.approx(0.3333)  # f2
    assert out["no_match_count"] == 1
    assert out["expected_empty_named_rate"] == pytest.approx(0.5)
    assert out["expected_empty_named_count"] == 1
    assert [t["returned"] for t in out["turns"]][2] == ["d", "e", "f", "g", "h"]


def test_an_empty_population_reports_none_not_zero():
    ev = _load()
    out = ev.score([FIXTURE[3]], lambda turn: [])
    assert out["recall@5"] is None and out["no_match_rate"] is None
    assert out["expected_empty_named_rate"] == 0.0


def test_the_labelled_set_meets_its_contract():
    ev = _load()
    records = ev.load_records()
    assert len(records) >= 40
    assert all(r.get("source") for r in records), "every record carries provenance"
    assert sum(1 for r in records if r.get("expected_empty")) >= 10
    assert len({r["id"] for r in records}) == len(records)


def test_a_stale_label_is_named():
    ev = _load()
    bad = ev.check_labels(FIXTURE, {"a", "b", "c"})
    assert bad == ["f3: expected skill 'd' is not an active skill"]


def test_the_baseline_lives_under_the_data_root():
    from app.paths import EVAL_BASELINES_DIR
    ev = _load()
    assert ev.baseline_path() == EVAL_BASELINES_DIR / "skill_match_baseline.json"
    rep = ev.build_report(FIXTURE, ev.score(FIXTURE, lambda t: RETURNED[t]))
    assert rep["matches_production_defaults"] is True
    assert {"graph_rerank", "rerank_alpha", "graph_top_k", "graph_hops"} <= set(rep)
    assert set(rep["arms"]) == {"lexical"}


def test_the_embedding_arm_did_not_land():
    assert not (ROOT / "agent_mcp" / "skill_cards.py").exists()
    assert not (ROOT / "scripts" / "setup_skill_cards.py").exists()
    imported = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    assert not [m for m in imported if "skill_cards" in m]


def test_the_scorer_imports_the_production_matcher(monkeypatch):
    """The lexical arm is prefetch's own function, not a copy of it."""
    from app import prefetch
    ev = _load()
    seen = []
    monkeypatch.setattr(prefetch, "_search_skills",
                        lambda toks: seen.append(toks) or [
                            (prefetch.SKILL_THRESHOLD_FIRST, {"name": "k"}),
                            (prefetch.SKILL_THRESHOLD_FIRST - 0.1, {"name": "offered"})])
    # An offer below the injection gate is not a lexical pick (#435 widened
    # `_search_skills` to the whole offer set).
    assert ev.lexical_top5("restart the backend please") == ["k"]
    assert seen and isinstance(seen[0], set)


def test_the_gold_set_carries_queries_whose_skill_the_pruned_arm_cuts_away():
    """#4027 clause 4: the gold set can actually populate the deferred-task control.

    `split_by_advertisement` derives the tail by diffing the two arms' advertised name sets, so a
    gold set whose every labelled skill survives the cut would report `tail_n: 0` and print the
    surviving-skill half as though it were the whole claim — the false green the item names. This
    node asks the rendered prompts who was cut (never a hardcoded skill, which would change
    population when the 30-day rank table moves) and requires at least three labelled records whose
    expected skill production advertises WITHOUT a description and the pruned arm drops entirely:
    a names-only skill is exactly the one a cut is supposed to cost, because it was already the
    weakest signal on the page.
    """
    import importlib.util as ilu
    import yaml
    root = Path(__file__).resolve().parent.parent
    spec = ilu.spec_from_file_location("rpe_4027", str(root / "eval" / "run_prefetch_cost_eval.py"))
    ev = ilu.module_from_spec(spec)
    spec.loader.exec_module(ev)

    pruned = ev.advertised_skill_names(ev.ARM_INJECTED_PRUNED)
    block = ev.advertised_index_block(ev.ARM_INJECTED)
    prod = ev.advertised_skill_names(ev.ARM_INJECTED)
    described = {ln.strip()[2:].split(" — ")[0] for ln in block.splitlines() if " — " in ln}
    assert prod and len(pruned) < len(prod), (len(prod), len(pruned))

    gold = yaml.safe_load((root / "eval" / "skill_match_queries.yaml").read_text())
    tail = [r["id"] for r in gold["records"]
            if any(s in prod and s not in described and s not in pruned
                   for s in (r.get("expected_skills") or []))]
    assert len(tail) >= 3, (
        f"only {len(tail)} labelled records name a skill the pruned arm cuts away ({tail}); "
        "the pruned arm would then be scored on surviving queries alone and its headline is the "
        "green this clause forbids")


# --- #4611: --compare grades drift on the shared turn ids, not on two
# --- different denominators.
#
# The fixture reproduces 2026-11-26 in miniature. The baseline scored two
# labelled turns (c1 hits at full recall, c2 returns nothing) for a recorded
# recall@5 of 0.5. Since then the matcher has NOT moved: on the same two ids it
# returns the same rankings. The only change is that the set gained one labelled
# turn, c3, which the matcher scores 0.0. Headline recall@5 therefore falls
# 0.5 -> 0.3333 (-0.1667, past DRIFT_TOLERANCE = 0.05) while the shared-set drift
# is exactly 0.0. The old compare block subtracted those two headlines and exited
# 1 on this fixture; that is the false REGRESSION #4611 removes.

CMP_RECORDS = [
    {"id": "c1", "turn": "t1", "expected_skills": ["a"]},
    {"id": "c2", "turn": "t2", "expected_skills": ["b"]},
    {"id": "c3", "turn": "t3", "expected_skills": ["d"]},   # added since the baseline
    {"id": "c4", "turn": "t4", "expected_skills": [], "expected_empty": True},
]
CMP_NOW = {"t1": ["a"], "t2": [], "t3": ["e"], "t4": []}
CMP_BASELINE = {
    "measured_at": "2026-09-24T19:29:49Z",
    "arms": {"lexical": {
        "labelled_turns": 2,
        "expected_empty_turns": 1,
        "recall@5": 0.5,
        # The September baseline records only id + returned, no per-turn metric.
        "turns": [{"id": "c1", "returned": ["a"]},
                  {"id": "c2", "returned": []},
                  {"id": "c4", "returned": []}],
    }},
}


def _run_compare(tmp_path, monkeypatch, records, returned, baseline):
    """Run the real CLI main() over a fixture: rc plus what it printed.

    The scorer, the label check, the active-skill inventory and the baseline file
    are all that main() reaches outside of, so patching exactly those four keeps
    the node hermetic while still crossing the boundary that matters — the exit
    code and stdout a human or a scheduled job reads.
    """
    ev = _load()
    base_file = tmp_path / "skill_match_baseline.json"
    base_file.write_text(json.dumps(baseline))
    monkeypatch.setattr(ev, "load_records", lambda *a, **k: records)
    monkeypatch.setattr(ev, "active_skill_names", lambda: {"a", "b", "c", "d", "e"})
    monkeypatch.setattr(ev, "lexical_top5", lambda turn: returned[turn])
    monkeypatch.setattr(ev, "baseline_path", lambda: base_file)
    return ev, ev.main(["--compare"])


def test_the_graded_drift_is_scored_on_only_the_turn_ids_both_sets_share(tmp_path, monkeypatch, capsys):
    """Clause 1: the figure --compare grades is recall@5 over the shared turn ids.

    c3 exists only in today's set and scores 0.0, so it belongs in neither side of
    the graded subtraction. The graded drift here is 0.0 while the headline gap is
    -0.1667: pinning both numbers is what stops a later edit from comparing
    denominators again.
    """
    ev, rc = _run_compare(tmp_path, monkeypatch, CMP_RECORDS, CMP_NOW, CMP_BASELINE)
    out = capsys.readouterr().out
    cmp_block = ev.baseline_drift(CMP_BASELINE, ev.score(CMP_RECORDS, lambda t: CMP_NOW[t]),
                                  CMP_RECORDS)
    assert cmp_block["shared_labelled_turns"] == 2, "c1 and c2 only — c3 is not in the baseline"
    assert cmp_block["shared_recall5_baseline"] == pytest.approx(0.5)
    assert cmp_block["shared_recall5_now"] == pytest.approx(0.5)
    assert cmp_block["graded_drift"] == pytest.approx(0.0)
    assert cmp_block["headline_drift"] == pytest.approx(-0.1667)
    assert rc == 0
    assert "graded" in out.lower(), out
    assert "+0.0000" in out, f"the graded figure is the one printed: {out}"


def test_label_growth_alone_does_not_read_as_a_regression(tmp_path, monkeypatch, capsys):
    """Clause 2: shared drift inside tolerance, headline past it -> exit 0, both headlines printed."""
    ev, rc = _run_compare(tmp_path, monkeypatch, CMP_RECORDS, CMP_NOW, CMP_BASELINE)
    captured = capsys.readouterr()
    assert rc == 0, f"label growth is not matcher drift: {captured.err}"
    assert "REGRESSION" not in captured.err, captured.err
    # Both headline figures still print, so the artifact shows what a naive read sees.
    assert "0.5" in captured.out and "0.3333" in captured.out, captured.out
    assert "0.1667" in captured.out, f"the headline drift is still stated: {captured.out}"
    assert ev.DRIFT_TOLERANCE == 0.05


def test_the_baseline_comparison_records_both_denominators(tmp_path, monkeypatch):
    """Clause 3: the artifact states which two sets were compared.

    2 labelled turns on the baseline side, 3 today, 2 shared — recorded as data,
    so a reader of the JSON never has to guess which denominator a figure came
    from (the guess that made the -0.0572 look like a loss).
    """
    ev = _load()
    lex = ev.score(CMP_RECORDS, lambda t: CMP_NOW[t])
    cmp_block = ev.baseline_drift(CMP_BASELINE, lex, CMP_RECORDS)
    assert cmp_block["baseline_labelled_turns"] == 2
    assert cmp_block["current_labelled_turns"] == 3
    assert cmp_block["shared_labelled_turns"] == 2
    assert cmp_block["comparable"] is True
    report = ev.build_report(CMP_RECORDS, lex)
    report["baseline_comparison"] = cmp_block
    assert report["baseline_comparison"]["baseline_labelled_turns"] == 2
    assert report["baseline_comparison"]["current_labelled_turns"] == 3


def test_a_baseline_sharing_no_turn_id_is_reported_not_comparable(tmp_path, monkeypatch, capsys):
    """Clause 4: nothing in common -> not comparable, exit 0, never a REGRESSION claim."""
    disjoint = json.loads(json.dumps(CMP_BASELINE))
    disjoint["arms"]["lexical"]["turns"] = [{"id": "smq900", "returned": ["a"]}]
    ev, rc = _run_compare(tmp_path, monkeypatch, CMP_RECORDS, CMP_NOW, disjoint)
    captured = capsys.readouterr()
    assert rc == 0, f"an incomparable baseline must not fail the run: {captured.err}"
    assert "REGRESSION" not in captured.err, captured.err
    assert "not comparable" in (captured.out + captured.err).lower()
    cmp_block = ev.baseline_drift(disjoint, ev.score(CMP_RECORDS, lambda t: CMP_NOW[t]),
                                 CMP_RECORDS)
    assert cmp_block["comparable"] is False
    assert cmp_block["graded_drift"] is None
    assert cmp_block["shared_labelled_turns"] == 0


def test_a_real_matcher_move_on_the_shared_set_still_exits_1(tmp_path, monkeypatch, capsys):
    """The guard still guards: fix the denominator, do not delete the check.

    Same ids, and the matcher now finds c2's skill where the baseline found
    nothing: shared recall@5 0.5 -> 1.0, +0.5 past the 0.05 tolerance.
    """
    moved = dict(CMP_NOW, t2=["b"])
    ev, rc = _run_compare(tmp_path, monkeypatch, CMP_RECORDS, moved, CMP_BASELINE)
    captured = capsys.readouterr()
    assert rc == 1, f"a 0.5 move on the shared set is a real regression: {captured.out}"
    assert "REGRESSION" in captured.err, captured.err
    cmp_block = ev.baseline_drift(CMP_BASELINE, ev.score(CMP_RECORDS, lambda t: moved[t]),
                                 CMP_RECORDS)
    assert cmp_block["graded_drift"] == pytest.approx(0.5)


def test_a_baseline_without_per_turn_rows_is_not_comparable(tmp_path, monkeypatch):
    """A trimmed baseline that records no turn ids gives no shared set, not a headline subtraction.

    Its headline was computed over a set this round cannot enumerate, so grading
    against it is the exact mismatch #4611 removes.
    """
    trimmed = json.loads(json.dumps(CMP_BASELINE))
    trimmed["arms"]["lexical"].pop("turns")
    ev = _load()
    cmp_block = ev.baseline_drift(trimmed, ev.score(CMP_RECORDS, lambda t: CMP_NOW[t]),
                                 CMP_RECORDS)
    assert cmp_block["comparable"] is False
    assert cmp_block["graded_drift"] is None
    assert cmp_block["reason"]
