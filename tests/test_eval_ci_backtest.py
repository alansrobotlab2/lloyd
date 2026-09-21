"""`eval/ci_backtest.py` — replaying past eval verdicts against an interval (#696).

Clause 4: the backtest runs offline over the baselines that actually exist, and
its output is a count, not a mood. The tests here build synthetic baseline
directories in `tmp_path` rather than reading the live one, because the live
corpus is gitignored runtime state that changes nightly — a test asserting a
number from it could not fail for the right reason. The one test that does read
the live directory asserts only the *shape* of its report and that it names its
window.

The fixture corpus is 20 queries, matching the eval's real size through
2026-09-20, so the intervals a test sees are the intervals the nightly report
actually had.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.ci_backtest as bt  # noqa: E402
import eval.run_eval as ev  # noqa: E402

PYTEST_SELF = ROOT / ".venvs" / "lloyd" / "bin" / "python"
SCRIPT = ROOT / "eval" / "ci_backtest.py"


def _baseline(path: Path, label: str, ran_at: str, ids: list[str],
              hits: dict[str, dict], corpus: dict | None = None) -> None:
    """Write one artifact in exactly the shape `eval/run_eval.py` writes."""
    overall = ev.summarize([
        {"id": q, "category": "single", "latency_ms": 50.0, "error": None,
         "scoring": dict({"entity_hit": False, "doc_hit": False, "entity_recall": 0.0,
                          "doc_recall": 0.0, "rr_doc": 0.0, "ndcg10": 0.0,
                          "fact_entity_recall": 0.0, "first_doc_rank": None},
                         **hits.get(q, {}))}
        for q in ids])["overall"]
    doc = {"label": label, "ran_at": ran_at, "limit": len(ids), "records": [
        {"id": q, "category": "single", "latency_ms": 50.0, "error": None,
         "scoring": dict({"entity_hit": False, "doc_hit": False, "entity_recall": 0.0,
                          "doc_recall": 0.0, "rr_doc": 0.0, "ndcg10": 0.0,
                          "fact_entity_recall": 0.0, "first_doc_rank": None},
                         **hits.get(q, {}))}
        for q in ids],
        "summary": {"overall": overall}}
    if corpus is not None:
        doc["corpus"] = corpus
    path.write_text(json.dumps(doc))


IDS = [f"q{i:02d}" for i in range(20)]
CORPUS = {"sqlite": {"entities": 100, "edges_active": 200, "path": "/x/kg.sqlite",
                     "facts": 300}}


def _all_hit(over: dict[str, dict] | None = None) -> dict[str, dict]:
    """Every query scores perfect; `over` patches the named fields per query.

    The patch MERGES: a fixture that replaced the whole scoring dict would leave
    the untouched legs at their False/0.0 defaults, and a test written to flip
    one metric would silently move all seven — which is exactly the sort of
    fixture bug that makes a statistic test pass for the wrong reason.
    """
    hits = {q: {"entity_hit": True, "doc_hit": True, "entity_recall": 1.0,
                "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 1.0,
                "fact_entity_recall": 1.0, "first_doc_rank": 1} for q in IDS}
    for qid, patch in (over or {}).items():
        hits[qid] = {**hits[qid], **patch}
    return hits


# ── the pairing decision is made by the data, not by preference ──────────────

def test_the_backtest_pairs_only_when_ids_and_corpus_provenance_both_match(tmp_path):
    """Quoting a paired interval across a corpus change is the false precision
    #608 exists to prevent, so the gate is ids AND the provenance block."""
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    c = tmp_path / "nightly-20260903-20260903-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(c, "nightly", "2026-09-03T06:00:00+00:00", IDS, _all_hit(),
              dict(CORPUS, sqlite=dict(CORPUS["sqlite"], facts=1)))
    runs = bt.load_runs(tmp_path)
    assert [bt.corpus_state(*pair) for pair in bt.transitions(runs)] == [
        "paired",               # identical ids, identical provenance
        "unpaired-corpus",      # facts moved: the corpus is not the same corpus
    ]


def test_a_baseline_with_no_provenance_block_can_never_be_paired(tmp_path):
    """Baselines written before 2026-09-12 record no `corpus` at all. Absence of
    evidence of a change is not evidence of no change."""
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), None)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS, _all_hit(), None)
    runs = bt.load_runs(tmp_path)
    pairs = bt.transitions(runs)
    assert [bt.corpus_state(p, c) for p, c in pairs] == ["unpaired-drift-unknown"]


def test_a_changed_query_set_is_unpaired_even_with_an_identical_corpus(tmp_path):
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS[:-1], _all_hit(), CORPUS)
    pairs = bt.transitions(bt.load_runs(tmp_path))
    assert [bt.corpus_state(p, c) for p, c in pairs] == ["unpaired-ids"]


# ── the two verdicts the tool has to tell apart ──────────────────────────────

def test_three_queries_flipping_is_reported_as_indistinguishable(tmp_path):
    """The 09-09 callout: three entity-side queries flipping is a 0.15 movement,
    past the old 0.05 trigger, and still nothing a 20-query eval can support."""
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS,
              _all_hit({q: {"entity_hit": False, "entity_recall": 0.0,
                            "fact_entity_recall": 0.0} for q in IDS[:3]}), CORPUS)
    decisions = [d for pair in bt.transitions(bt.load_runs(tmp_path))
                 for d in bt.redecide(*pair)]
    picked = [d for d in decisions if d["metric"] == "entity_hit_rate"]
    assert len(picked) == 1, picked          # the old rule DID call it out
    d = picked[0]
    assert d["delta"] == -0.15 and abs(d["delta"]) > bt.MOVE_THRESHOLD
    assert d["indistinguishable"] is True
    # The paired bracket on three flips is [-0.15, 0.0]: its upper bound lands
    # exactly on zero, which does not exclude zero. That is the same verdict as
    # straddling it, and it is why the rule is "the interval excludes zero", not
    # "the interval is strictly on one side".
    assert d["lo"] < 0 and d["hi"] >= 0, d


def test_the_paired_form_starts_deciding_at_four_flipped_queries(tmp_path):
    """The paired design's power edge, measured on this fixture rather than
    claimed: three flipped queries of twenty stays inside the interval, four
    falls outside it (P(no flip drawn) = 0.8**20 = 1.2 %, under the 2.5 % tail).

    This is the clause that keeps the tool from being a blinder. #696's own Risks
    section allows that wider intervals could deadlock a gate; the answer the
    item gives is the paired design, and this is what that design is actually
    worth — a threshold of four queries on a pinned corpus, where the unpaired
    rule needs more than ten.
    """
    for n_flips, expect_indistinguishable in ((3, True), (4, False)):
        tmp = tmp_path / f"n{n_flips}"
        tmp.mkdir()
        _baseline(tmp / "nightly-20260901-20260901-060000.json", "nightly",
                  "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
        _baseline(tmp / "nightly-20260902-20260902-060000.json", "nightly",
                  "2026-09-02T06:00:00+00:00", IDS,
                  _all_hit({q: {"entity_hit": False} for q in IDS[:n_flips]}), CORPUS)
        picked = [d for pair in bt.transitions(bt.load_runs(tmp))
                  for d in bt.redecide(*pair) if d["metric"] == "entity_hit_rate"]
        assert len(picked) == 1, (n_flips, picked)
        assert picked[0]["pairing"] == "paired"
        assert picked[0]["indistinguishable"] is expect_indistinguishable, (n_flips, picked[0])


def test_a_metric_collapsing_to_zero_is_still_called_out(tmp_path):
    """The other half of the clause, and the one that keeps the tool honest: a
    real one-query-collapse-shaped regression (every doc score gone) must NOT be
    drowned by the wider intervals, or this change would be a blinder, not a
    bound."""
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS,
              _all_hit({q: {"doc_hit": False, "doc_recall": 0.0, "rr_doc": 0.0,
                            "ndcg10": 0.0} for q in IDS}), CORPUS)
    decisions = [d for pair in bt.transitions(bt.load_runs(tmp_path))
                 for d in bt.redecide(*pair)]
    doc = [d for d in decisions if d["metric"] == "doc_hit_rate"]
    assert len(doc) == 1 and doc[0]["indistinguishable"] is False, doc
    assert doc[0]["hi"] < 0, doc[0]


def test_metrics_the_old_rule_would_not_have_called_out_are_not_counted(tmp_path):
    """The backtest replays the rule that was in force; counting sub-threshold
    drift as a 'prior verdict' would inflate M and flatter the finding."""
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS,
              _all_hit({IDS[0]: {"ndcg10": 0.5}}), CORPUS)   # delta -0.025
    decisions = [d for pair in bt.transitions(bt.load_runs(tmp_path))
                 for d in bt.redecide(*pair)]
    assert decisions == []


# ── families: what counts as a verdict somebody acted on ────────────────────

def test_the_nightly_series_is_one_family_even_though_its_label_carries_two_dates(tmp_path):
    """The nightly job passes the day IN the label, so stripping only the clock
    would make every night its own family and the backtest would silently replay
    zero pairs of the one series the report is built from."""
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    runs = bt.load_runs(tmp_path)
    assert {r.family for r in runs} == {"nightly"}
    assert len(bt.transitions(runs)) == 1


def test_a_one_off_experiment_arm_is_not_counted_as_a_prior_verdict(tmp_path):
    """Consecutive arms of someone's A/B probe never became a report line, so a
    headline that counted them would overstate what the rule actually did."""
    a = tmp_path / "qmdpin-win1200-20260915-090000.json"
    b = tmp_path / "qmdpin-win1200-20260915-091200.json"
    _baseline(a, "qmdpin-win1200", "2026-09-15T09:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "qmdpin-win1200", "2026-09-15T09:12:00+00:00", IDS,
              _all_hit({q: {"entity_hit": False} for q in IDS[:10]}), CORPUS)
    pairs = bt.transitions(bt.load_runs(tmp_path))
    decisions = [d for pair in pairs for d in bt.redecide(*pair)]
    assert decisions, "the probe arm should still produce decisions to report"
    assert all(d["authoritative"] is False for d in decisions)


# ── the CLI: it names a count, and it names the window it covered ────────────

def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args, capture_output=True, text=True,
        timeout=300, env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
                          "LLOYD_ROOT": str(ROOT),
                          "LLOYD_VOICE_ALERTS": "0", "PYTHONDONTWRITEBYTECODE": "1"})


def test_the_cli_output_names_the_count_and_the_window_it_covered(tmp_path):
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS,
              _all_hit({q: {"entity_hit": False} for q in IDS[:3]}), CORPUS)
    proc = _run(["--baselines", str(tmp_path)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "1 of 1" in out, out                      # N of M, the clause's shape
    assert "indistinguishable" in out.lower(), out
    assert "2026-09-01" in out and "2026-09-02" in out, out   # the stated window
    assert "paired" in out, out


def test_quiet_mode_prints_only_the_count(tmp_path):
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    proc = _run(["--baselines", str(tmp_path), "--quiet"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "0 of 0", proc.stdout


def test_an_empty_baseline_directory_is_an_error_not_a_zero_verdict(tmp_path):
    """`0 of 0` in a worktree with no gitignored baselines would read exactly
    like 'the rule never fooled anyone', which is the opposite of true."""
    proc = _run(["--baselines", str(tmp_path), "--quiet"])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert str(tmp_path) in (proc.stdout + proc.stderr)


def test_the_backtest_is_offline_and_reproducible(tmp_path):
    """No LLM, no network, no daemon: every input is a records[] array already on
    disk. Proven by environment — the only variables the child gets are PATH,
    HOME, LLOYD_ROOT and the alert opt-out, so a stray recall call to qmd or a
    fact-store read cannot succeed and cannot be needed. Reproducible by
    comparison: two runs emit byte-identical decision lists, which is also what
    the fixed bootstrap seed buys."""
    a = tmp_path / "nightly-20260901-20260901-060000.json"
    b = tmp_path / "nightly-20260902-20260902-060000.json"
    _baseline(a, "nightly", "2026-09-01T06:00:00+00:00", IDS, _all_hit(), CORPUS)
    _baseline(b, "nightly", "2026-09-02T06:00:00+00:00", IDS,
              _all_hit({q: {"entity_hit": False, "entity_recall": 0.0, "ndcg10": 0.0,
                            "doc_hit": False, "doc_recall": 0.0,
                            "fact_entity_recall": 0.0} for q in IDS[:5]}), CORPUS)
    first, second = _run(["--baselines", str(tmp_path)]), _run(["--baselines", str(tmp_path)])
    assert first.returncode == 0 and second.returncode == 0
    strip = lambda s: [ln for ln in s.splitlines() if "generated" not in ln]
    assert strip(first.stdout) == strip(second.stdout)
