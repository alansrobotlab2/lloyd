"""The printed retrieval-eval summary carries its interval, n, or "no verdict".

Clause 3 of #696. The rule being pinned is narrow and is the reason the item
exists: a retrieval-eval number may not leave this file as a bare point
estimate, and a metric with nothing in its denominator may not print as a
score at all. `fact_entity_recall_avg` already had that rule on the artifact
side (#1250/#1260 — a null and a 0.000 must not print the same); this is the
same rule for the interval and for the other six metrics.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_VENV = ROOT / ".venvs" / "lloyd" / "bin" / "python"
PY = _VENV if _VENV.exists() else Path(sys.executable)
SCRIPT = ROOT / "eval" / "run_eval.py"

sys.path.insert(0, str(ROOT))
# run_eval.py writes its record under the data root the child inherits.
from app.paths import EVAL_BASELINES_DIR as BASELINES  # noqa: E402
import eval.run_eval as ev  # noqa: E402

#: printed label -> overall-metric key, in the order the summary prints them
PRINTED = (
    ("MRR", "mrr_doc"),
    ("NDCG10", "ndcg10"),
    ("doc_hit", "doc_hit_rate"),
    ("doc_recall", "doc_recall_avg"),
    ("entity_hit", "entity_hit_rate"),
    ("entity_recall", "entity_recall_avg"),
    ("fact_entity_recall", "fact_entity_recall_avg"),
)


def _rec(i, hit=True, doc_hit=True, val=0.5):
    return {"id": f"p{i}", "category": "single", "latency_ms": 100.0, "error": None,
            "scoring": {"entity_hit": hit, "doc_hit": doc_hit, "entity_recall": val,
                        "doc_recall": val, "rr_doc": val, "ndcg10": val,
                        "fact_entity_recall": val, "first_doc_rank": 1}}


def _lines(text: str) -> dict:
    """Map each printed metric label to its line, from the `Overall:` block."""
    out = {}
    for label, _metric in PRINTED:
        hits = [ln for ln in text.splitlines() if ln.strip().startswith(label + " ")]
        assert hits, f"{label} was not printed:\n{text}"
        out[label] = hits[0]
    return out


def test_every_overall_metric_line_carries_its_interval_and_its_n(capsys):
    records = [_rec(i, hit=(i % 2 == 0), doc_hit=(i % 3 != 0), val=(i % 5) / 4)
               for i in range(20)]
    summary = ev.summarize(records)
    ev.print_table(records, summary)
    lines = _lines(capsys.readouterr().out)
    for label, metric in PRINTED:
        line = lines[label]
        entry = summary["overall"]["ci95"][metric]
        assert f"n={entry['n']}" in line, (label, line)
        assert f"{entry['ci'][0]:.3f}" in line and f"{entry['ci'][1]:.3f}" in line, (label, line)
        # The rate and its bracket on ONE line, rate first: a number copied off
        # this page without its interval is the reading this clause prevents.
        # Rate, then bracket, then n — in that order. No `and`-guarded
        # expression: a lower bound of exactly 0.0 short-circuits to the float
        # and raises TypeError instead of failing.
        assert line.index(f"{entry['ci'][0]:.3f}") < line.index("n="), (label, line)


def test_a_zero_denominator_prints_no_verdict_and_never_a_pass(capsys):
    """The clause's sharp edge: an unscored metric must not print as 0.00 with a
    zero-width bracket. That is a pass-shaped line about nothing."""
    records = [_rec(i) for i in range(20)]
    for r in records:
        r["scoring"]["fact_entity_recall"] = None
    summary = ev.summarize(records)
    ev.print_table(records, summary)
    lines = _lines(capsys.readouterr().out)
    fact = lines["fact_entity_recall"]
    assert "no verdict" in fact, fact
    assert "n=0" in fact, fact
    assert "0.000" not in fact, fact
    # The only bracket on the line is the words "no verdict" — no numeric bound,
    # because there is no measurement for a bound to describe.
    assert not re.search(r"\[\s*\d", fact), fact
    # The six metrics that WERE scored still show brackets — the rule is not a
    # blanket suppression of intervals.
    assert "[" in lines["entity_hit"], lines["entity_hit"]


def test_a_one_query_run_reports_no_bracket_on_its_mean_metrics(capsys):
    """A percentile bootstrap over one observation returns that observation with
    zero width. Printing `[0.500,0.500] n=1` would claim a perfectly measured
    mean, so the mean metrics print no verdict while the hit rates — which are
    defined on one trial — keep their (very wide) Wilson interval."""
    records = [_rec(0)]
    summary = ev.summarize(records)
    ev.print_table(records, summary)
    lines = _lines(capsys.readouterr().out)
    assert "no verdict" in lines["MRR"] and "n=1" in lines["MRR"], lines["MRR"]
    assert "[" in lines["entity_hit"] and "n=1" in lines["entity_hit"], lines["entity_hit"]


def test_a_summary_with_no_ci95_block_at_all_says_so_rather_than_looking_bounded():
    """An older summary dict (or one hand-built by a caller) must not read as
    evidence. `[no interval]` is a different claim from `[0.5,0.5]`."""
    assert ev._fmt_ci("mrr_doc", {"mrr_doc": 0.5}) == "  [no interval] n=?"
    assert ev._fmt_ci("mrr_doc", {"ci95": {"mrr_doc": {"ci": [0.4, 0.6], "n": 20}}}) \
        == "  [0.400,0.600] n=20"


# ── end to end: the real writer, the real page ───────────────────────────────

def _queries_file(tmp_path: Path) -> Path:
    p = tmp_path / "queries.yaml"
    p.write_text(
        "queries:\n"
        "  - id: ci-probe\n"
        "    query: what is lloyd\n"
        "    category: single\n"
        "    expect_entities: [Lloyd]\n"
        "    expect_docs: [lloyd]\n"
    )
    return p


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """One query against an empty fact tree and a freshly created store.

    Same isolation `test_eval_corpus_guard.py` uses: the env overrides are what
    make 'empty corpus' mean this directory rather than the live vault, and the
    store has to be created because the reader refuses an absent database
    (#1236). One scored query is the fixture on purpose: it is the case where a
    metric's denominator is too small to bound at all, which is what clause 3 is
    about, and it is also the cheapest honest run the script can do.
    """
    facts = tmp_path / "facts"
    facts.mkdir(exist_ok=True)
    db = tmp_path / "kg.sqlite"
    env = dict(os.environ)
    env["LLOYD_FACTS_ROOT"] = str(facts)
    env["LLOYD_KG_DB"] = str(db)
    env["LLOYD_VOICE_ALERTS"] = "0"
    if not db.exists():
        subprocess.run(
            [str(PY), "-c",
             "import sys; from app.kg_store import KGStore; KGStore(sys.argv[1]).close()",
             str(db)],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
            check=True)
    return subprocess.run(
        [str(PY), str(SCRIPT), "--queries", str(_queries_file(tmp_path)), *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)


def test_a_real_run_prints_intervals_and_writes_them_into_the_baseline(tmp_path):
    label = "pytest-ci-reporting"
    for f in BASELINES.glob(f"{label}-*.json"):
        f.unlink()
    try:
        proc = _run(tmp_path, "--label", label, "--allow-empty-corpus")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        lines = _lines(proc.stdout)
        for label_, _metric in PRINTED:
            assert ("n=" in lines[label_]), (label_, lines[label_])
            assert ("[" in lines[label_]) or ("no verdict" in lines[label_]), \
                (label_, lines[label_])
        # One query scored means every MEAN metric has a single observation and
        # so no bracket, and the Wilson bounds on the two hit rates are the
        # whole [0,1] range bar one tail. Nothing here is allowed to look
        # precise, and on this fixture nothing does.
        assert "no verdict" in lines["fact_entity_recall"], lines["fact_entity_recall"]
        assert "no verdict" in lines["MRR"], lines["MRR"]
        assert lines["entity_hit"].count("[") == 1, lines["entity_hit"]

        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        overall = json.loads(written[0].read_text())["summary"]["overall"]
        assert set(m for _l, m in PRINTED) <= set(overall["ci95"])
        assert overall["ci95"]["entity_hit_rate"]["n"] == 1
        # One query against an empty corpus, and it missed: Wilson on k=0, n=1 is
        # [0.0, 0.7935]. The rate prints 0.00 and the interval reaches up to 0.79
        # — the clause in miniature. A run this small cannot support either
        # direction, which is exactly what the old bare-rate line hid.
        assert overall["ci95"]["entity_hit_rate"]["ci"] == [0.0, 0.7935], \
            overall["ci95"]["entity_hit_rate"]
        assert overall["ci95"]["fact_entity_recall_avg"]["ci"] == [None, None]
    finally:
        for f in BASELINES.glob(f"{label}-*.json"):
            f.unlink()


# ── the consumer across the process boundary ─────────────────────────────────
# `summary.overall` is written here and READ by another process: the self-mod
# regression rung loads it out of eval/baselines/*.json
# (`workers/sources/automod_regression.py:328`) and feeds it to `_accumulate`
# (which builds the noise floor every tolerance comes from) and `evaluate`
# (which decides whether a commit regresses). That reader ALSO opens pre-#696
# baselines that have no `ci95` key at all, and `ci95` is a nested dict sitting
# among numbers — the exact shape a `.values()`-style consumer mishandles.
# Review rung finding on the first gate: no test crossed this seam. Two shapes,
# one reader, tested against the real `summarize` output rather than a fixture
# shaped like it.

def _real_overall(entity_hits: list[int]) -> dict:
    """A `summary.overall` from the real `summarize`, so the payload under test
    has every key the real one has — including the ones this round did not
    think of."""
    records = [_rec(i) for i in range(len(entity_hits))]
    for r, h in zip(records, entity_hits):
        r["scoring"]["entity_hit"] = bool(h)
    return ev.summarize(records)["overall"]


def test_the_regression_reader_tolerates_ci95_and_its_absence_alike():
    from workers.sources import automod_regression as reg

    modern = _real_overall([1] * 10 + [0] * 10)
    assert "ci95" in modern, "nothing to test if the writer stopped emitting it"
    legacy = {k: v for k, v in modern.items() if k != "ci95"}   # a pre-#696 file

    samples: dict = {}
    reg._accumulate(samples, modern)
    reg._accumulate(samples, legacy)
    assert "ci95" not in samples, sorted(samples)
    assert samples["entity_hit_rate"] == [modern["entity_hit_rate"]] * 2

    noise = {"metrics": {m: {"stdev": 0.0} for m in reg.ARMED_METRICS}}
    assert reg.evaluate(modern, dict(modern), noise)[0] is False
    assert reg.evaluate(legacy, dict(legacy), noise)[0] is False

    # The armed comparison still fires on a real drop carried in the new payload:
    # the interval is additive, it did not replace the gate.
    regressed, reasons, _ = reg.evaluate(
        _real_overall([1] * 4 + [0] * 16), _real_overall([1] * 20), noise)
    assert regressed and any("entity_hit_rate" in r for r in reasons), reasons


def test_a_baseline_the_real_writer_produced_loads_in_the_real_reader(tmp_path):
    """Bytes, not fixtures: `eval/run_eval.py` runs as a subprocess, writes a
    baseline carrying `ci95`, and `_load_arm` reads that file off disk."""
    import tempfile

    from workers.sources import automod_regression as reg

    label = "ci95seam"
    queries = tmp_path / "q.yaml"
    queries.write_text("queries:\n"
                       "  - id: seam-probe\n    query: what is lloyd\n"
                       "    category: single\n    expect_entities: [Lloyd]\n"
                       "    expect_docs: [lloyd]\n")
    with tempfile.TemporaryDirectory() as store, tempfile.TemporaryDirectory() as facts:
        db = Path(store) / "kg.sqlite"
        env = dict(os.environ, PYTHONPATH=str(ROOT), LLOYD_KG_DB=str(db),
                   LLOYD_FACTS_ROOT=facts, LLOYD_VOICE_ALERTS="0")
        subprocess.run([sys.executable, "-c",
                        f"from app.kg_store import KGStore; KGStore({str(db)!r}).close()"],
                       cwd=ROOT, env=env, capture_output=True, check=True)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "run_eval.py"),
             "--queries", str(queries), "--label", label, "--allow-empty-corpus"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"*{label}*.json"))
        assert len(written) == 1, written
        try:
            blob = json.loads(written[0].read_text())
            assert "ci95" in blob["summary"]["overall"]
            arm = reg._load_run(BASELINES, label)
            assert arm is not None
            assert arm["overall"]["entity_hit_rate"] == \
                blob["summary"]["overall"]["entity_hit_rate"]
            assert arm["corpus_ok"] == blob["corpus_ok"]
            assert "ci95" not in reg.ARMED_METRICS
        finally:
            for f in written:
                f.unlink()


# ═════════════════════════════════════════════════════════════════════════════
# The duplicate-slot stage is report-only (#2271, clause 4)
#
# With the stage ON by default, a fixture run's `doc_hit_rate`, `mrr_doc`,
# `ndcg10`, `doc_recall_avg` — and their `<metric>_normalized` /
# `<metric>_ceiling_kind` companions — must EQUAL the same run's values with the
# stage switched off. The stage reads slots and never reorders, drops or re-scores
# one; this is the test that fails the day someone lets it.
# ═════════════════════════════════════════════════════════════════════════════

_DUP_SHARED = (
    "The nightly retrieval eval scores whether the gold document is inside the "
    "returned top K and how high it sits, and when those two figures disagree no "
    "arm of this harness has ever named the cause. Duplicate slots are one "
    "candidate mechanism for that divergence, because a twin occupies a slot that "
    "could have held the document the question actually needed, and the metric "
    "measures exactly what that share of the returned set costs a run."
)
_DUP_OTHER = (
    "Wake-word audio depends on the pulse media binding far more than it depends "
    "on the acoustic models, and an outage blamed on a model is usually a capture "
    "device that moved between cards. The binding is read directly, by name, "
    "before anyone is permitted to say a word about the models themselves or "
    "about which of them was loaded into the process at the time."
)
_DUP_THIRD = (
    "Nightly writers append knowledge notes with no principled forgetting rule, so "
    "the corpus grows sideways instead of upward and one finding arrives again "
    "under a slightly different title some weeks later. A number attached to that "
    "growth is precisely what a retention proposal would need before anybody "
    "deleted a file, and its absence is why nobody has written that proposal."
)
_DUP_PATHS = ["knowledge/gold-note.md", "knowledge/gold-note-twin.md",
              "knowledge/wake-word.md", "knowledge/nightly-writers.md"]
_DUP_TEXTS = {"knowledge/gold-note.md": _DUP_SHARED,
              "knowledge/gold-note-twin.md": _DUP_SHARED,
              "knowledge/wake-word.md": _DUP_OTHER,
              "knowledge/nightly-writers.md": _DUP_THIRD}


def _dup_note(body, stamp):
    return f"---\ntimestamp: '{stamp}'\n---\n\n{body}\n"


def _dup_scoring(monkeypatch, *, report_dups: bool):
    """One fixture query scored through `_score`, slot content served from a dict.

    `report_dups=False` is what `--no-dup-metrics` produces: `dup_metrics=None`.
    """
    stamps = {"knowledge/gold-note.md": "2026-01-01T00:00:00",
              "knowledge/gold-note-twin.md": "2026-06-01T00:00:00",
              "knowledge/wake-word.md": "2026-02-01T00:00:00",
              "knowledge/nightly-writers.md": "2026-03-01T00:00:00"}
    real = ev._slot_text
    monkeypatch.setattr(ev, "_slot_text",
                        lambda p: (_dup_note(_DUP_TEXTS[p], stamps[p])
                                   if p in _DUP_TEXTS else None))
    try:
        spec = {"id": "dup-ci", "query": "duplicate slots",
                "category": "adversarial", "expect_entities": ["Retrieval Eval"],
                "expect_docs": ["gold-note"],
                "retrieval_result": {
                    "documents": [{"path": p, "citation": p, "snippet": "s"}
                                  for p in _DUP_PATHS],
                    "entities": ["Retrieval Eval"], "facts": [],
                    "graph_facts": [], "vault_facts": []}}
        res = spec["retrieval_result"]
        return ev._score(spec, res,
                         dup_metrics=(ev._dup_metrics_for_paths(_DUP_PATHS)
                                      if report_dups else None))
    finally:
        monkeypatch.setattr(ev, "_slot_text", real)


def _dup_overall(scoring):
    rec = {"id": "dup-ci", "category": "adversarial", "latency_ms": 5.0,
           "query": "duplicate slots", "seeds_extracted": [],
           "expected": {"docs": ["gold-note"], "entities": ["Retrieval Eval"]},
           "result_summary": {"n_docs": len(_DUP_PATHS),
                              "doc_paths_top10": list(_DUP_PATHS)},
           "scoring": scoring, "error": None}
    return ev.summarize([rec])["overall"]


def test_the_duplicate_slot_stage_changes_no_scored_metric(monkeypatch):
    """Clause 4: ON and OFF give the same five scores AND the same companions."""
    on, off = (_dup_scoring(monkeypatch, report_dups=b) for b in (True, False))
    for key in ("doc_hit", "rr_doc", "ndcg10", "doc_recall", "first_doc_rank",
                "entity_hit", "entity_recall", "fact_entity_recall"):
        assert on[key] == off[key], f"{key} moved when the stage was switched on"

    o_on, o_off = _dup_overall(on), _dup_overall(off)
    for metric in ("doc_hit_rate", "mrr_doc", "ndcg10", "doc_recall_avg"):
        assert o_on[metric] == o_off[metric], metric
        assert (o_on[f"{metric}_normalized"], o_on[f"{metric}_ceiling_kind"]) \
            == (o_off[f"{metric}_normalized"], o_off[f"{metric}_ceiling_kind"]), metric
        assert o_on["ci95"][metric] == o_off["ci95"][metric], metric
    # the stage's own numbers are the only thing that differs
    assert o_on["dup_slot_share"] > 0.0
    assert o_off["dup_slot_share"] == 0.0, (
        "stage off reports 0.0 over ZERO slots compared and keeps the key present, "
        "so a night is never missing a number and never reads twin-free by accident")
    assert on["dup_slot_n"] > 0 and off["dup_slot_n"] == 0


def test_the_stage_cannot_reorder_or_drop_a_slot(monkeypatch):
    """The mechanism behind clause 4, checked on the list it is handed.

    `_dup_metrics_for_paths` takes a list and returns a dict — it has no way to
    hand back a different ordering. This asserts the caller's list is untouched and
    that nothing readable yields `dup_slot_n: 0` (no verdict) rather than a 0.0
    share that would read as a twin-free corpus.
    """
    real = ev._slot_text
    monkeypatch.setattr(ev, "_slot_text", lambda p: None)
    try:
        paths = list(_DUP_PATHS)
        snapshot = list(paths)
        out = ev._dup_metrics_for_paths(paths)
        assert paths == snapshot, "the stage mutated the caller's slot list"
        assert out["dup_slot_n"] == 0
        assert out["dup_slot_unreadable"] == len(_DUP_PATHS)
    finally:
        monkeypatch.setattr(ev, "_slot_text", real)
