"""#580 — the durable-write judge calibration harness.

Six of the item's clauses are process claims ("a shipped script prints it",
"the two judges differ only in X", "nothing in the live write path reads a judge
score"), and each is pinned here against either the shipped artifacts under
``eval/durable_write_judge/`` or a small crafted corpus. The crafted corpora are
**test fixtures**, never corpus samples: the corpus itself is real repaired Lloyd
output and its provenance is a clause of its own (test 1).

The judging pass itself (100 secondary-engine calls) is deliberately not run
here — see ``eval/durable_write_judge/README.md``.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.durable_write_judge.build_corpus as bc  # noqa: E402
import eval.durable_write_judge.judge as jd  # noqa: E402
import eval.durable_write_judge.retrieve as rt  # noqa: E402
import eval.durable_write_judge.score as sc  # noqa: E402

HERE = ROOT / "eval" / "durable_write_judge"
CORPUS = HERE / "corpus.jsonl"


def load_shipped_corpus() -> list[dict]:
    assert CORPUS.exists(), f"corpus artifact missing: {CORPUS}"
    return sc.load_jsonl(CORPUS)


def fixture_sample(sid: str, label: str, text: str, classes=(), evidence=None,
                   path=None) -> dict:
    """A crafted sample for retrieval/scoring tests — fixture data, not corpus data."""
    key = "pre_repair_text" if label == "bad" else "accepted_text"
    return {
        "id": sid, "label": label, "seq": int(sid.split("-")[1]),
        "vault_path": path or f"knowledge/youtube/CH/{sid}.md",
        "defect_classes": list(classes),
        "correction_evidence": evidence or {},
        "recovered_from": f"git show 0a1b2c3d^{'' if label == 'good' else '^'}:"
                          f"{path or f'knowledge/youtube/CH/{sid}.md'}",
        key: text,
    }


@pytest.fixture
def mini_corpus() -> list[dict]:
    return [
        fixture_sample("bad-001", "bad",
                       "Router notes at https://github.com/acme/quantum-router all day.",
                       ["invented_url"], {"invented_url": ["https://github.com/acme/quantum-router"]}),
        fixture_sample("bad-002", "bad",
                       "The transcript cuts off before the solutions, so numbers are absent.",
                       ["false_cutoff_claim"], {"false_cutoff_claim": ["cuts off"]}),
        fixture_sample("bad-003", "bad",
                       "Router latency improved a lot. Nothing quantitative at all here.",
                       ["omitted_numerical_data"], {"omitted_numerical_data": ["p50 12 ms"]}),
        fixture_sample("good-001", "good",
                       "The router's p50 latency was 12 ms at 4 000 rps, quoted from the talk.",
                       path="knowledge/youtube/CH/router-metrics.md"),
        fixture_sample("good-002", "good",
                       "Ontology kernel paper, arXiv 2608.22974, dynamic ontology over a schema graph.",
                       path="knowledge/youtube/CH/ontology-kernel.md"),
        fixture_sample("good-003", "good",
                       "Kubernetes namespace deletion and Postgres wire protocol parsing below HTTP.",
                       path="knowledge/youtube/CH/claw-patrol.md"),
        # A second invented_url sample in a *different* file: without it the only
        # same-class cases are bad-001's own duplicates, which the twin guard
        # excludes, and selection quality could only ever read False.
        fixture_sample("bad-004", "bad",
                       "Quantum router paper at https://github.com/acme/ghost-router with benchmarks.",
                       ["invented_url"], {"invented_url": ["https://github.com/acme/ghost-router"]},
                       path="knowledge/youtube/CH/quantum-router-paper.md"),
    ]


# --- clause 1: the corpus artifact and the script that prints its counts -------

def test_counts_printed_per_class_and_every_bad_carries_a_recovery_object():
    samples = load_shipped_corpus()
    out = bc.format_counts(samples).splitlines()
    bad = sum(1 for s in samples if s["label"] == "bad")
    good = sum(1 for s in samples if s["label"] == "good")
    assert out[0] == f"corpus samples={len(samples)} bad={bad} good={good}"
    for name in bc.DEFECT_CLASSES:
        assert f"class {name} n=" in "\n".join(out), f"{name} absent from the count block"
    assert bad >= 15, "clause 1 targets >=15 bad; the shipped corpus must state n, not pad"
    assert good >= 30, "clause 1 targets >=30 matched good"

    for s in samples:
        assert s["label"] in ("bad", "good")
        assert re.match(r"git show [0-9a-f]{7,40}\^?:\S+", s["recovered_from"]), s["id"]
        assert bc.sample_text(s).strip(), f"{s['id']} carries no text to judge"
        if s["label"] == "bad":
            assert s["defect_classes"], f"{s['id']} is bad with no named defect class"
            assert set(s["defect_classes"]) <= set(bc.DEFECT_CLASSES), s["id"]
            assert "pre_repair_text" in s, f"{s['id']} missing its pre-repair text"
            assert s["correction_evidence"].keys() <= set(s["defect_classes"]), s["id"]
            assert "^:" in s["recovered_from"], f"{s['id']} not recovered pre-repair"
        else:
            assert "accepted_text" in s, f"{s['id']} missing its accepted text"
            assert s["defect_classes"] == []


def test_builder_labels_only_from_a_visible_repair_signature(tmp_path):
    url_note = "Title\nSee https://github.com/acme/x and https://github.com/acme/y\n"
    classes = bc.detect_defect_classes(url_note, "Title\nNo link survives the repair.\n")
    assert classes["invented_url"] == ["https://github.com/acme/x",
                                       "https://github.com/acme/y"]
    # The same URL kept by the repair is not a defect: no signature, no label.
    assert "invented_url" not in bc.detect_defect_classes(url_note, url_note + " extra\n")
    assert bc.detect_defect_classes(url_note, url_note) == {}

    cutoff = "The transcript cut off before the solutions.\n"
    assert "false_cutoff_claim" in bc.detect_defect_classes(
        cutoff, "The Q&A runs to the end of the recording.\n")
    assert "false_cutoff_claim" not in bc.detect_defect_classes(
        cutoff, cutoff + "More detail added later.\n")

    bare, measured = ("The system got faster and cheaper.\n",
                      "The system hit 4 000 rps at 12 ms p50, saving 30%.\n"
                      "Cost fell by $12k a month.\nCache hit rate was 82%.\n")
    assert "omitted_numerical_data" in bc.detect_defect_classes(bare, measured)
    assert "omitted_numerical_data" not in bc.detect_defect_classes(measured, measured)

    assert "missing_section" in bc.detect_defect_classes(
        "## Summary\nbody\n", "## Summary\nbody\n\n## Q&A\naudience asked X\n")
    assert "missing_section" not in bc.detect_defect_classes(
        "## Summary\nbody\n## Q&A\nx\n", "## Summary\nlonger body\n## Q&A\nx\n")

    fm_pre = "---\ntitle: Wrong Name\nspeaker: A\n---\nbody\n"
    fm_post = "---\ntitle: Right Name\nspeaker: A\n---\nbody\n"
    named = bc.detect_defect_classes(fm_pre, fm_post)
    assert named["wrong_metadata"] == ["title: 'Wrong Name' -> 'Right Name'"]
    assert bc.detect_defect_classes(fm_pre, fm_pre) == {}

    # --out writes the artifact; --counts re-reads it without touching git.
    out = tmp_path / "corpus.jsonl"
    bc.main(["--vault", str(tmp_path), "--out", str(out)])
    written = sc.load_jsonl(out)
    assert bc.format_counts(written) == bc.format_counts([])
    assert subprocess.run(
        [sys.executable, str(HERE / "build_corpus.py"), "--counts", str(out)],
        capture_output=True, text=True).stdout.startswith("corpus samples=0")


# --- clause 2: the judges differ in example selection, and nothing else -------

def test_the_two_judges_differ_only_in_example_selection(mini_corpus):
    pool = mini_corpus * 3  # > EXAMPLE_K candidates, as in the shipped corpus
    seen_differences = 0
    for s in pool:
        if s["label"] != "bad":
            continue
        a_ex = jd.random_examples(pool, s)
        b_ex = jd.retrieved_examples(pool, s)
        assert len(a_ex) == len(b_ex) == jd.EXAMPLE_K == 5, s["id"]
        pa = jd.build_prompt(s, a_ex)
        pb = jd.build_prompt(s, b_ex)
        assert pa.head == pb.head, "rubric preamble must be byte-identical"
        assert pa.tail == pb.tail, "the judged sample block must be byte-identical"
        assert pa.example_count == pb.example_count == 5
        assert {e["id"] for e in a_ex} != {e["id"] for e in b_ex}, s["id"]
        seen_differences += 1
    assert seen_differences >= 3
    assert jd.ENGINE_MODEL == "secondary" and "8091" in jd.ENGINE_URL, (
        "one model for both judges, on the secondary engine")
    assert jd.SEED == 580  # Judge A's draw is reproducible, not re-rolled per run


def test_random_draw_is_seeded_and_never_shows_the_sample_its_own_file(mini_corpus):
    pool = mini_corpus * 3
    s = pool[0]
    ids = {e["id"] for e in jd.random_examples(pool, s)}
    assert s["id"] not in ids
    assert {e["vault_path"] for e in jd.random_examples(pool, s)} <= {
        e["vault_path"] for e in pool if e["vault_path"] != s["vault_path"]}
    assert jd.random_examples(pool, s) == jd.random_examples(pool, s)


def test_retrieval_never_hands_the_sample_its_own_file_back(mini_corpus):
    pool = mini_corpus * 3
    index = rt.build_index(pool)
    s = next(x for x in pool if x["id"] == "bad-003")
    retrieved = rt.top_k(index, pool, s, k=5)
    assert len(retrieved) == 5
    ids = {r["id"] for r in retrieved}
    assert s["id"] not in ids
    leaked = {r["id"] for r in retrieved if r["vault_path"] == s["vault_path"]}
    assert not leaked, f"same-file twin retrieved: {leaked} — that is the label leaking"
    # Same text, same file, different id: the twin is the case that must be excluded.
    twin = fixture_sample("bad-099", "bad", bc.sample_text(s), ["omitted_numerical_data"],
                          path=s["vault_path"])
    pool2 = pool + [twin]
    assert twin["id"] not in {r["id"] for r in rt.top_k(rt.build_index(pool2), pool2, s, k=5)}
    assert rt.excluded_ids(pool2, s) == {s["id"], twin["id"]}


# --- clause 3: the reported pair, including the silent-pass rate --------------

def test_silent_pass_rate_is_accepted_and_bad_over_all_accepted(mini_corpus):
    # 4 bad: one flagged, three accepted. 3 good: one flagged, two accepted.
    verdicts_a = {"bad-001": "flag", "bad-002": "accept", "bad-003": "accept",
                  "bad-004": "accept",
                  "good-001": "flag", "good-002": "accept", "good-003": "accept"}
    assert set(verdicts_a) == {s["id"] for s in mini_corpus}
    m = sc.metrics(mini_corpus, verdicts_a)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 3, 1, 2)
    assert m["recall_on_bad"] == round(100 * 1 / 4, 1)
    assert m["false_positive_rate"] == round(100 * 1 / 3, 1)
    assert m["silent_pass_rate"] == round(100 * 3 / 5, 1), (
        "silent pass = accepted samples whose label is bad, over everything accepted")
    assert m["accepted"] == 5 and m["unjudged"] == 0

    # An engine error is not an accept and not a flag: it is reported, not scored.
    degraded = dict(verdicts_a, **{"bad-001": "error", "good-001": "unparsed"})
    d = sc.metrics(mini_corpus, degraded)
    assert d["unjudged"] == 2
    assert (d["tp"], d["fn"], d["fp"], d["tn"]) == (0, 3, 0, 2)
    assert d["silent_pass_rate"] == round(100 * 3 / 5, 1)


def test_a_missing_verdict_is_counted_as_unjudged_not_as_an_accept(mini_corpus):
    m = sc.metrics(mini_corpus, {"bad-001": "flag", "good-001": "accept"})
    assert m["unjudged"] == 5 and m["accepted"] == 1
    assert m["silent_pass_rate"] == 0.0 and m["recall_on_bad"] == 100.0


def test_a_failed_engine_call_is_an_error_row_and_never_a_silent_accept(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine unreachable")

    monkeypatch.setattr(jd, "call_engine", boom)
    row = jd.judge_one("b", [{"id": "bad-001", "label": "bad", "vault_path": "p",
                              "defect_classes": ["invented_url"], "seq": 1,
                              "pre_repair_text": "x"}],
                       {"id": "bad-001", "label": "bad", "vault_path": "p",
                        "defect_classes": ["invented_url"], "seq": 1,
                        "pre_repair_text": "x"}, jd.ENGINE_URL, jd.ENGINE_MODEL)
    assert row["verdict"] == "error", row
    assert row["verdict"] != "accept"
    assert jd.parse_verdict("")["verdict"] == "unparsed"
    assert jd.parse_verdict("no object here at all")["verdict"] == "unparsed"


def test_parse_verdict_takes_the_object_out_of_a_reasoning_reply():
    reasoning = ("Here is my thinking:\n1. Analyze the note. The URL looks plausible but "
                 "the transcript never named it, so it may be invented.\n\n"
                 '{"verdict": "flag", "defect_class": "invented_url", '
                 '"reason": "URL absent from the source"}')
    assert jd.parse_verdict(reasoning) == {
        "verdict": "flag", "defect_class": "invented_url",
        "reason": "URL absent from the source"}
    assert jd.parse_verdict('{"verdict": "maybe", "defect_class": "none"}')["verdict"] == "unparsed"


# --- clause 4: per-class recall, printed for every class ----------------------

def test_report_names_every_class_and_reports_zero_recall_rows(mini_corpus):
    verdicts = {"bad-001": "flag", "bad-002": "flag", "bad-003": "accept",
                "good-001": "accept", "good-002": "accept", "good-003": "accept"}
    rows = sc.per_class_recall(mini_corpus, verdicts)
    by_class = {r["class"]: r for r in rows}
    assert set(by_class) == {"invented_url", "false_cutoff_claim",
                             "omitted_numerical_data"}
    assert by_class["invented_url"]["recall"] == 100.0
    assert by_class["omitted_numerical_data"] == {
        "class": "omitted_numerical_data", "n": 1, "n_judged": 1, "caught": 0,
        "recall": 0.0}, "a class nobody catches is the deliverable, not a failure"


def test_shipped_report_prints_per_class_recall_for_every_corpus_class(tmp_path, mini_corpus):
    raw_a = tmp_path / "judge_raw_a.jsonl"
    raw_b = tmp_path / "judge_raw_b.jsonl"
    raw_a.write_text("\n".join(json.dumps({"sample_id": s["id"], "judge": "a",
                                           "verdict": "accept"}) for s in mini_corpus))
    raw_b.write_text("\n".join(json.dumps({"sample_id": s["id"], "judge": "b",
                                           "verdict": "accept"}) for s in mini_corpus))
    report = sc.render(mini_corpus, {"a": raw_a, "b": raw_b})
    for name in sc.class_counts(mini_corpus):
        assert f"| `{name}` |" in report, f"per-class table missing {name}"
    assert "| 0 | 0.0 |" in report, "an all-accept judge must print recall 0.0"
    assert "silent-pass rate" in report


# --- clause 5: selection quality, separate from judgement quality -------------

def test_selection_quality_reads_the_labels_not_the_verdict(mini_corpus):
    pool = mini_corpus * 3
    index = rt.build_index(pool)
    s = next(x for x in pool if x["id"] == "bad-001")  # invented_url, shares "router"
    retrieved = rt.top_k(index, pool, s, k=5)
    assert rt.same_class_retrieved(s, retrieved) is True, retrieved
    good = next(x for x in pool if x["label"] == "good")
    assert rt.same_class_retrieved(good, retrieved) is None, (
        "a good sample has no class to match; it must not count either way")

    # A sample whose nearest neighbours are all other classes scores False, not True.
    orphan = fixture_sample("bad-004", "bad",
                            "Zebra flibflabs the gizmo quux with wombler power and zzz.",
                            ["wrong_metadata"], path="knowledge/youtube/CH/orphan.md")
    pool2 = pool + [orphan]
    assert rt.same_class_retrieved(
        orphan, [r for r in rt.top_k(rt.build_index(pool2), pool2, orphan, k=5)
                 if r["id"] != orphan["id"]]) is False

    # And it cannot be moved by the judge's own answer: the metric is a function
    # of the corpus and the retriever only, so it survives a re-run unchanged and
    # never consults a verdict file.
    import inspect
    assert "verdict" not in inspect.signature(rt.same_class_retrieved).parameters
    assert "verdict" not in inspect.signature(sc.selection_quality).parameters
    assert sc.selection_quality(pool) == sc.selection_quality(pool) == sc.selection_quality(pool, k=5)


def test_report_keeps_selection_quality_apart_from_judgement_quality(tmp_path, mini_corpus):
    raw_a = tmp_path / "judge_raw_a.jsonl"
    raw_b = tmp_path / "judge_raw_b.jsonl"
    for path, judge in ((raw_a, "a"), (raw_b, "b")):
        path.write_text("\n".join(json.dumps({
            "sample_id": s["id"], "judge": judge,
            "verdict": "flag" if (judge == "b" and s["label"] == "bad") else "accept",
            "example_ids": []}) for s in mini_corpus))
    report = sc.render(mini_corpus, {"a": raw_a, "b": raw_b})
    head, _, tail = report.partition("## Selection quality")
    assert tail, "the report must have a selection-quality section of its own"
    assert "silent-pass rate" in head, "judgement metrics live before it, not inside it"
    sq = sc.selection_quality(mini_corpus)
    assert f"{sq['same_class_rate']}%" in tail
    assert "same-file twin" in tail and "twin_leak_rate" not in report
    for name in sc.JUDGE_NAMES:
        assert name in head
    assert "Which cases moved the score" in report


# --- clause 6: offline only, no live write path reads a judge score -----------

def test_the_judging_harness_is_not_wired_into_any_write_path():
    roots = ["workers", "app", "scripts", "server.py", "prefetch.py",
             "prompt_builder.py", "autonomy.py"]
    needles = ("durable_write_judge", "build_corpus", "judge_one",
               "durable_write_judge.judge")
    hits = []
    for root in roots:
        target = ROOT / root
        if not target.exists():
            continue
        grep = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.md", "-e",
             needles[0], "-e", needles[1], "-e", needles[2], *([str(target)])],
            capture_output=True, text=True)
        hits += [ln for ln in grep.stdout.splitlines() if ln.strip()]
    assert not hits, "the judging module must not be imported by a live path:\n" + "\n".join(hits)

    vault = Path.home() / "obsidian"
    for segment in ("autonomy", "skills"):
        seg = vault / segment
        if not seg.is_dir():
            continue  # no vault on this machine; the repo half above still pins it
        grep = subprocess.run(["grep", "-rn", "-e", needles[0], str(seg)],
                              capture_output=True, text=True)
        assert not grep.stdout.strip(), (
            f"a {segment} definition lands writes and must not name the judge:\n"
            + grep.stdout)


def test_shipped_artifacts_exist_and_their_counts_are_the_ones_reported():
    for name in ("corpus.jsonl", "judge_raw_a.jsonl", "judge_raw_b.jsonl",
                 "report.md", "README.md"):
        assert (HERE / name).exists(), f"{name} must be shipped for #580 to be checkable"
    samples = load_shipped_corpus()
    report = (HERE / "report.md").read_text()
    counts = sc.class_counts(samples)
    bad = sum(1 for s in samples if s["label"] == "bad")
    good = sum(1 for s in samples if s["label"] == "good")
    assert f"bad: **{bad}** / good: **{good}**" in report

    # Recompute Judge A's row from the raw verdicts, independently of sc.metrics,
    # and require the shipped report to carry exactly those numbers.
    verdicts = {r["sample_id"]: r["verdict"]
                for r in sc.load_jsonl(HERE / "judge_raw_a.jsonl")}
    fn = sum(1 for s in samples if s["label"] == "bad" and verdicts.get(s["id"]) == "accept")
    tn = sum(1 for s in samples if s["label"] == "good" and verdicts.get(s["id"]) == "accept")
    tp = sum(1 for s in samples if s["label"] == "bad" and verdicts.get(s["id"]) == "flag")
    fp = sum(1 for s in samples if s["label"] == "good" and verdicts.get(s["id"]) == "flag")
    judged_bad, accepted = tp + fn, fn + tn
    recall = round(100.0 * tp / judged_bad, 1) if judged_bad else None
    fpr = round(100.0 * fp / (fp + tn), 1) if (fp + tn) else None
    silent = round(100.0 * fn / accepted, 1) if accepted else None
    assert f"| **{recall}** | {fpr} | **{silent}** |" in report, (
        f"report disagrees with raws: recall={recall} fpr={fpr} silent={silent}")
    assert counts and all(f"| `{c}` | {n}" in report for c, n in counts.items())


def test_no_judged_sample_was_its_own_retrieved_example_in_the_shipped_run():
    samples = load_shipped_corpus()
    by_id = {s["id"]: s for s in samples}
    index = rt.build_index(samples)
    for row in sc.load_jsonl(HERE / "judge_raw_b.jsonl"):
        sample = by_id[row["sample_id"]]
        assert sample["id"] not in row["example_ids"], row["sample_id"]
        twins = {s["id"] for s in samples if s["vault_path"] == sample["vault_path"]}
        assert not (twins & set(row["example_ids"])), row["sample_id"]
        assert len(row["example_ids"]) == jd.EXAMPLE_K
        assert set(row["example_ids"]) == {
            r["id"] for r in rt.top_k(index, samples, sample, k=jd.EXAMPLE_K)}, (
            "the shipped run must be reproducible from the shipped corpus")
    for row in sc.load_jsonl(HERE / "judge_raw_a.jsonl"):
        assert len(row["example_ids"]) == jd.EXAMPLE_K
        assert row["sample_id"] not in row["example_ids"]
