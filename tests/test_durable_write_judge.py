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


def test_the_report_carries_the_self_preference_pointer_beside_the_published_rate():
    """Clause 5 (#2465): #2254's stop-rule result is written where the figure is read.

    The probe happened, measured no self-preference effect, and left no trace here: the
    whole record was `~/lloyd-data/eval/owedcheck-2254/REPORT.md`, outside the checkout,
    so a reader of the 32.4 % line — which two other files cite by line number as that
    figure's source — had no way to know the confound they may be about to invoke had
    been tested and closed. Hence two demands on the line: it sits within a few lines of
    the number, and it carries the finding in its own words, because a bare path into
    `~/lloyd-data/` is a pointer a reader of a clone cannot follow.

    The position is asserted from the table row outward rather than hard-coded, with one
    exception: the Judge A row must stay line 32. Two shipped files print
    `eval/durable_write_judge/report.md:32` as a citation, and inserting a paragraph
    ABOVE the table would renumber it and make both citations silently false — a true
    citation that has quietly become a wrong line number is exactly the kind of claim no
    later test would catch.
    """
    rep = (HERE / "report.md").read_text()
    lines = rep.splitlines()
    a_row = next(i for i, l in enumerate(lines)
                 if "32.4" in l and "Judge A" in l and l.startswith("|"))
    assert a_row + 1 == 32, (
        f"the Judge A row is line {a_row + 1}; the probe's files cite `report.md:32` "
        "from two places, so it has to stay line 32")

    window = "\n".join(lines[a_row + 1:a_row + 6])
    assert "self-preference" in window.lower(), (
        f"nothing about the confound appears beside the figure: {window[:300]}")
    assert "2026-10-09" in window, "the pointer must date the probe"
    assert "`primary`" in window, (
        "and name the engine that answered, since 32.4 % came off a different one")
    assert "negative" in window.lower(), "and say which way it came back"
    assert "NOT reproduce" in window, (
        "in the words the finding actually supports, not just the word 'negative'")
    assert "~/lloyd-data/eval/owedcheck-2254/REPORT.md" in window, (
        "and name where the per-arm figures live")
    assert "65.0 % for" in window and "75.0 %" in window, (
        "the two pass rates the negative rests on, so the claim is checkable without "
        "opening a file that is not in this checkout")
    assert "32.4 % stands as a calibration figure" in window, (
        "and what the reader is to conclude about the row above it")
    assert rep == sc.render(load_shipped_corpus(),
                            {"a": HERE / "judge_raw_a.jsonl",
                             "b": HERE / "judge_raw_b.jsonl"}), (
        "the pointer must be part of what score.py renders, not a hand edit beside it")


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
    roots = ["workers", "app", "scripts", "server.py"]
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

    # Recompute EACH judge's whole row from its own raw verdicts, independently of
    # sc.metrics, and require the shipped report to carry exactly those numbers.
    # Re-deriving only Judge A's row would leave B's recall, false-positive rate and
    # silent-pass rate as literal prose that a wrong number still satisfies — and
    # clause 3 asks for one of each *per judge*.
    for judge in ("a", "b"):
        verdicts = {r["sample_id"]: r["verdict"]
                    for r in sc.load_jsonl(HERE / f"judge_raw_{judge}.jsonl")}
        fn = sum(1 for s in samples if s["label"] == "bad" and verdicts.get(s["id"]) == "accept")
        tn = sum(1 for s in samples if s["label"] == "good" and verdicts.get(s["id"]) == "accept")
        tp = sum(1 for s in samples if s["label"] == "bad" and verdicts.get(s["id"]) == "flag")
        fp = sum(1 for s in samples if s["label"] == "good" and verdicts.get(s["id"]) == "flag")
        judged_bad, judged_good, accepted = tp + fn, fp + tn, fn + tn
        unjudged = len(samples) - judged_bad - judged_good
        recall = round(100.0 * tp / judged_bad, 1) if judged_bad else None
        fpr = round(100.0 * fp / judged_good, 1) if judged_good else None
        silent = round(100.0 * fn / accepted, 1) if accepted else None
        # The row shape moves with #2823's clause 5, which demands the label on the
        # table row itself: every recomputed number stays in its own cell, and the
        # silent-pass cell additionally carries the UNRATIFIED-DENOMINATOR label.
        assert (f"| {sc.JUDGE_NAMES[judge]} | {judged_bad}/{judged_good} "
                f"(unjudged {unjudged}) | **{recall}** | {fpr} | **{silent}** "
                "UNRATIFIED-DENOMINATOR |") in report, (
            f"report disagrees with judge_raw_{judge}.jsonl: "
            f"recall={recall} fpr={fpr} silent={silent}")
    assert counts and all(f"| `{c}` | {n}" in report for c, n in counts.items())
    # Every other figure the report states — the per-class recall table, the
    # selection-quality percentages, the moved-case rows — is re-derived the same
    # way: the shipped artifact must be exactly what score.py renders from these
    # two raw files and this corpus, so no reported number is hand-written.
    assert report == sc.render(samples, {"a": HERE / "judge_raw_a.jsonl",
                                        "b": HERE / "judge_raw_b.jsonl"}), (
        "report.md is not what score.py renders from the shipped corpus and raws")


def test_the_examples_every_shipped_row_showed_reproduce_from_the_shipped_corpus():
    """Clause 2 is a claim about the *shipped run*, not only about the functions.

    Every one of the 100 shipped rows must show exactly five examples, none from
    the judged sample's own file, and the same five the shipped code would pick
    again: Judge B's by recomputing the BM25 top-k, Judge A's by recomputing the
    seeded draw. Half-pinning this (B only) would leave the README's claim that
    re-running ``judge.py`` reproduces the same ``example_ids`` untrue for A, and
    A's five examples are the comparison's control arm.
    """
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
        sample = by_id[row["sample_id"]]
        assert len(row["example_ids"]) == jd.EXAMPLE_K
        assert row["sample_id"] not in row["example_ids"]
        twins = {s["id"] for s in samples if s["vault_path"] == sample["vault_path"]}
        assert not (twins & set(row["example_ids"])), row["sample_id"]
        assert set(row["example_ids"]) == {
            e["id"] for e in jd.random_examples(samples, sample)}, (
            "Judge A's five examples are a seeded draw (SEED=580); re-drawing them "
            "must give the shipped five, or A is not a reproducible control")


def _fixture_raws(tmp_path, corpus, verdict):
    """Both judges agree on `verdict` for every sample: enough to render."""
    raws = {}
    for name in ("a", "b"):
        path = tmp_path / f"judge_raw_{name}.jsonl"
        path.write_text("\n".join(
            json.dumps({"sample_id": s["id"], "judge": name, "verdict": verdict})
            for s in corpus) + "\n")
        raws[name] = path
    return raws


def test_a_single_example_class_is_named_by_name_when_one_exists(tmp_path, mini_corpus):
    raws = _fixture_raws(tmp_path, mini_corpus, "accept")
    line = next(ln for ln in sc.render(mini_corpus, raws).splitlines()
                if "single example" in ln)
    assert "`false_cutoff_claim`" in line, (
        "this fixture has exactly one bad false_cutoff_claim sample, so the report "
        "must name it: its recall row is one sample")
    assert "`invented_url`" not in line, "invented_url has 2 bad examples here"


def test_a_corpus_whose_thinnest_class_has_two_examples_makes_no_thin_claim(tmp_path):
    """The shipped corpus's thinnest class is `false_cutoff_claim` at n=2, so the
    sentence is conditional on the corpus and not a fixed line about a named class."""
    pair = [
        fixture_sample("bad-001", "bad", "first transcript cut-off note here.",
                       ["false_cutoff_claim"], {"false_cutoff_claim": ["x"]},
                       path="knowledge/youtube/CH/cutoff-one.md"),
        fixture_sample("bad-002", "bad", "second transcript cut-off note here.",
                       ["false_cutoff_claim"], {"false_cutoff_claim": ["y"]},
                       path="knowledge/youtube/CH/cutoff-two.md"),
        fixture_sample("good-001", "good", "Kubernetes pods drain cleanly here.",
                       path="knowledge/youtube/CH/pods-drain.md"),
    ]
    text = sc.render(pair, _fixture_raws(tmp_path, pair, "accept"))
    assert "single example" not in text


def test_the_shipped_report_makes_no_single_example_claim():
    """Reads the landed artifact, not a fixture: the shipped corpus's thinnest class
    has 2 examples, so the report may not assert a class of one."""
    assert "Classes with a single example" not in (HERE / "report.md").read_text()


def test_every_bad_row_carries_class_text_and_a_reconstructible_git_object():
    """Clause 1 is per bad sample, and the git object must be derivable from the
    row's own commit and path rather than merely shaped like one."""
    for s in load_shipped_corpus():
        if s["label"] != "bad":
            continue
        assert s["defect_classes"], s["id"]
        assert s["pre_repair_text"].strip(), s["id"]
        assert s["recovered_from"] == (
            f'git show {s["repair_commit"]}^:{s["vault_path"]}'), s["id"]


def test_the_readme_measured_section_agrees_with_the_shipped_artifacts():
    """The README's headline numbers and its test citations must be re-measurable.

    Two drifts this pins, both of them real here. (1) The Measured section quoted a
    retrieval class-match rate and a twin-leak rate from an earlier corpus build
    (36.8 % and 68.4 %) while the shipped corpus measures 75.0 % and 0 % — a summary
    that no longer matches the table it summarises is the defect this item exists to
    measure, one file up. (2) The README cites tests by name, and a rename in this
    file leaves the citation pointing at nothing, which reads as satisfied to the
    next reader. So every figure in the Measured section must equal what score.py
    re-derives from the shipped corpus and raws, and every ``test_...`` name quoted
    anywhere in the README must be a test defined in this file.
    """
    readme = (HERE / "README.md").read_text()
    samples = load_shipped_corpus()

    measured = readme.split("## Measured", 1)[1]
    vs = {j: sc.verdicts_by_id(sc.load_jsonl(HERE / f"judge_raw_{j}.jsonl"))
          for j in ("a", "b")}
    ma, mb = sc.metrics(samples, vs["a"]), sc.metrics(samples, vs["b"])
    sq = sc.selection_quality(samples)
    for value in (ma["recall_on_bad"], ma["false_positive_rate"],
                  ma["silent_pass_rate"], mb["recall_on_bad"],
                  mb["false_positive_rate"], mb["silent_pass_rate"]):
        assert f"{abs(value)} %" in measured, (
            f"README Measured section does not state {value} %")
    margin = round(mb["recall_on_bad"] - ma["recall_on_bad"], 1)
    assert f"{abs(margin)} points" in measured, (
        f"README Measured section does not state the {margin}-point margin")
    assert f"**{sq['same_class_rate']} %**" in measured, sq["same_class_rate"]
    assert f"**{int(sq['twin_leak_rate'])} % same-file twin leakage**" in measured

    counts = bc.class_counts(samples)
    n_bad = sum(1 for s in samples if s["label"] == "bad")
    assert f"{n_bad} bad across {len(counts)} classes" in readme
    assert f"the smallest class at n = {min(counts.values())}" in readme

    defined = set(re.findall(r"^def (test_\w+)", Path(__file__).read_text(), re.M))
    cited = set(re.findall(r"`(test_\w+)`", readme))
    assert cited, "the README cites no tests, so the citation half is vacuous"
    assert not (cited - defined), (
        f"README cites tests that do not exist in this file: {sorted(cited - defined)}")


# --- #2823: the bounded, delegated ratification of the labels --------------------
#
# #2639's owed-check ruled on these labels on 2026-10-16 and the ruling is dated and
# scoped; what it did not do was reach the three places that describe them. Until the
# nodes below, nothing in this file mentioned ratification at all (grep `ratif` here
# returned 0 hits), which is why the sentence had been free to rot. These tests pin
# the WORDING of a settled ruling, not a judgement: a future round that re-opens the
# ruling changes the ruling's note first, and these assertions with it.

#: The phrases the ruling itself uses. Each has to be legible in the emitted text.
RATIF_MUST = ("RATIFIED as gold", "#2639 owed-check", "2026-10-16",
              "on Alan's delegation", "refused as gold", "absence-of-repair")

#: Every spelling under which the text could hand the ratification back to Alan, who
#: explicitly did not ratify it: "the text must not claim otherwise".
RATIF_MUST_NOT = ("pending Alan", "unratified by Alan", "would have to ratify",
                  "Alan is the domain expert", "ratified by Alan", "until Alan",
                  "until that ratification exists")


def shipped_report_lines() -> list[str]:
    """The committed artifact's lines, 1-based citation via ``lines[n - 1]``."""
    return (HERE / "report.md").read_text().splitlines()


def shipped_render() -> list[str]:
    """What score.py renders from the shipped corpus and both raw verdict files."""
    return sc.render(load_shipped_corpus(),
                     {"a": HERE / "judge_raw_a.jsonl",
                      "b": HERE / "judge_raw_b.jsonl"}).splitlines()


def readme_ratif_paragraph() -> str:
    readme = (HERE / "README.md").read_text()
    start = readme.index("Labels are rule-derived")
    return readme[start:readme.index("\n\n", start)]


def test_the_status_sentence_states_which_label_half_is_gold_and_who_said_so():
    """Clause 1 (#2823): the emitted status line carries the ruling, not the old ask.

    The 20 `bad` labels are gold because each re-derives from a repair diff plus the
    repairer's own defect word; the 30 `good` labels are not gold at all, because all
    that warrants them is that no repair exists. Both halves and both counts are
    asserted, and the counts are interpolated from the shipped corpus — a sentence that
    names 20 and 30 by hand goes stale the first time `build_corpus.py` is re-run.
    """
    samples = load_shipped_corpus()
    n_bad = sum(1 for s in samples if s["label"] == "bad")
    n_good = sum(1 for s in samples if s["label"] == "good")
    assert (n_bad, n_good) == (20, 30), "the ruling is scoped to these counts"
    status = shipped_render()[2]
    assert status.startswith("**Status: offline experiment."), (
        "the ratification belongs in the status sentence the report opens with, not in "
        "a paragraph added below it — an added paragraph moves the line citations")
    for needle in RATIF_MUST:
        assert needle in status, f"status line omits {needle!r}: {status}"
    assert f"the {n_bad} `bad` labels are RATIFIED as gold" in status
    assert f"the {n_good} `good` labels are refused as gold" in status
    assert "recall-on-bad is quotable and silent-pass is not" in status
    for banned in RATIF_MUST_NOT:
        assert banned not in status, f"status line still says {banned!r}"


def test_no_shipped_line_names_alan_as_the_ratifier_or_the_pending_one():
    """Clause 2 (#2823): in this eval Alan holds one role only — the one who delegated.

    The rule is mechanical because the misattribution is easy to write by accident: any
    occurrence of the name must be followed by ``'s delegation``, so "pending Alan's
    ratification", "unratified by Alan" and "Alan is the domain expert who would have to
    ratify them" all fail here, while the delegation phrase passes. It is checked on all
    three surfaces the ruling had to reach: the status line, the Caveats bullet, and the
    README's provenance paragraph. The grep is then run as the item's own command over
    the whole directory, because without ``--exclude-dir=__pycache__`` that command has
    matched the stale compiled ``score.cpython-312.pyc`` and could never reach 0 hits.
    """
    report = shipped_report_lines()
    targets = {"report.md line 3": report[2],
               "report.md line 113": report[112],
               "README.md provenance paragraph": readme_ratif_paragraph()}
    for where, text in targets.items():
        assert "on Alan's delegation" in text, f"{where} lost the delegation: {text}"
        for tail in re.findall(r"Alan(.{0,15})", text):
            assert tail.startswith("'s delegation"), (
                f"{where} names Alan in another role: 'Alan{tail}'")
        for banned in RATIF_MUST_NOT:
            assert banned not in text, f"{where} credits Alan: {banned!r}"

    grep = subprocess.run(
        ["grep", "-rn", "--exclude-dir=__pycache__", "pending Alan", str(HERE)],
        capture_output=True, text=True)
    assert grep.returncode == 1 and not grep.stdout.strip(), (
        "the item's own acceptance grep still returns hits:\n" + grep.stdout)


def test_the_one_rater_caveat_carries_the_ruling_without_moving_two_cited_lines():
    """Clause 3 (#2823): the second stale claim lived in Caveats, and both lines cited.

    The self-preference probe's own ``report.py`` prints
    ``eval/durable_write_judge/report.md:113`` twice as the source of the `good`-half
    caveat, and ``report.md:32`` as the source of Judge A's published figure (pinned in
    ``test_the_report_carries_the_self_preference_pointer_beside_the_published_rate``).
    So each rewritten sentence must stay exactly ONE emitted ``lines`` element: a
    rewrite that wrapped to a second paragraph would renumber both citations and leave
    two true claims pointing at the wrong numbers, which no test outside this file
    would notice.
    """
    lines = shipped_report_lines()
    bullet = lines[112]
    assert bullet.startswith("- **One rater.**"), bullet
    for needle in RATIF_MUST:
        assert needle in bullet, f"the caveat omits {needle!r}: {bullet}"
    assert "Alan is the domain expert" not in bullet
    assert "would have to ratify" not in bullet
    assert "inter-rater reliability is unavailable" in bullet, (
        "the honest half of the old sentence survives: no second rater was ever run, "
        "and a Gemma-family second rater is inter-rater evidence, not ratification")
    assert "(`correction_evidence`)" in bullet, "and the route to re-judge it survives"

    rendered = shipped_render()
    assert rendered[2] == lines[2] and rendered[112] == bullet, (
        "both rewritten sentences are emitted, not hand-written beside the renderer")
    a_row = next(i for i, ln in enumerate(lines)
                 if "32.4" in ln and "Judge A" in ln and ln.startswith("|"))
    assert a_row + 1 == 32, (
        f"the Judge A row moved to line {a_row + 1}: the self-preference probe cites "
        "`report.md:32` from its arms.py and report.py")
    assert lines.index(bullet) + 1 == 113, (
        "the `good`-half caveat must stay line 113 — report.py cites that number twice")


def test_the_score_cli_regenerates_the_committed_report_byte_for_byte(tmp_path):
    """Clause 4 (#2823), across the one process boundary this change crosses.

    ``report.md`` is a product, so the ratification line is code: triage ran the CLI on
    the shipped corpus and both raws and got the committed bytes back unchanged, which
    is why a hand edit of the report alone would have been deleted by the next scoring
    run. The check here is the CLI, not another in-process ``render`` call, because what
    a reader trusts is the file ``python eval/durable_write_judge/score.py`` writes.
    """
    out = tmp_path / "report-regenerated.md"
    proc = subprocess.run([sys.executable, str(HERE / "score.py"), "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"score.py exited {proc.returncode}: {proc.stderr[-400:]}"
    committed = (HERE / "report.md").read_bytes()
    assert out.read_bytes() == committed, (
        "the committed report.md is not what score.py writes — the new provenance text "
        "would be pasted prose that the next scoring run silently reverts")
    text = committed.decode()
    assert "on Alan's delegation" in text and "UNRATIFIED-DENOMINATOR" in text, (
        "the regenerated bytes must be the ones carrying the ruling")
    assert "(UNRATIFIED-DENOMINATOR)" in proc.stdout, (
        "the CLI's own silent-pass echo is a printed figure too")


def test_every_printed_silent_pass_figure_is_labelled_and_no_recall_figure_is(tmp_path,
                                                                             mini_corpus):
    """Clause 5 (#2823): the label sits on each figure, not in a paragraph below it.

    Silent-pass's denominator is *all accepted*, so it rests entirely on the 30 `good`
    labels the ruling refused as gold, while recall-on-bad reads only the 20 ratified
    ones — the two columns of one table are therefore not equally quotable, and the
    report has to say so on the row. Recall must NOT carry the label either: an
    undifferentiated warning on every number would leave the quotable figure unreadable
    and the unquotable one merely scary. Run on a fixture corpus as well as the shipped
    one, because the label is a property of the renderer, not of these verdicts.
    """
    def check(lines: list[str], where: str) -> None:
        rows = [ln for ln in lines if ln.startswith("| Judge ")]
        assert len(rows) == 2, f"{where}: both result rows expected, got {rows}"
        for row in rows:
            cells = row.split("|")
            assert "UNRATIFIED-DENOMINATOR" in cells[-2], (
                f"{where}: the silent-pass cell is unlabelled: {row}")
            assert "UNRATIFIED-DENOMINATOR" not in cells[3], (
                f"{where}: recall-on-bad must not borrow the label: {row}")
        header = next(ln for ln in lines if ln.startswith("| judge |"))
        hcells = header.split("|")
        assert "UNRATIFIED-DENOMINATOR" in hcells[-2], f"{where}: {header}"
        assert "recall-on-bad" in hcells[3], (
            f"{where}: the header changed columns: {header}")
        bullets = [ln for ln in lines if ln.startswith("- Judge ") and "silent pass" in ln]
        assert len(bullets) == 2, f"{where}: per-judge bullets missing: {bullets}"
        for bullet in bullets:
            assert "UNRATIFIED-DENOMINATOR" in bullet, (
                f"{where}: bullet prints the rate unlabelled: {bullet}")
            assert "UNRATIFIED-DENOMINATOR" not in bullet.split("recall ")[-1], (
                f"{where}: the recall clause of the same bullet stays unlabelled: {bullet}")

    check(sc.render(mini_corpus, _fixture_raws(tmp_path, mini_corpus, "accept")).splitlines(),
          "fixture render")
    check(shipped_report_lines(), "shipped report.md")


def test_the_readme_states_the_delegated_ruling_and_keeps_its_measured_figures():
    """Clause 5's README half (#2823): the hand-written provenance sentence agreed with
    the stale one, so it had to be rewritten from the same ruling and not from the
    report. ``test_the_readme_measured_section_agrees_with_the_shipped_artifacts`` keeps
    the numbers themselves pinned; this node keeps only the provenance wording.
    """
    paragraph = readme_ratif_paragraph()
    for needle in RATIF_MUST:
        assert needle in paragraph, f"README omits {needle!r}: {paragraph}"
    assert "UNRATIFIED-DENOMINATOR" in paragraph, (
        "and names the label the report prints the silent-pass figures under")
    for banned in RATIF_MUST_NOT:
        assert banned not in paragraph, f"README still says {banned!r}"
