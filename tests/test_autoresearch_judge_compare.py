"""The #698 scalar-vs-binary judge comparison driver: its arithmetic, offline.

`eval/autoresearch_judge_compare.py` decides whether the binary judge becomes
the default, so the three numbers it reads that decision off — AUC, the
pairwise flip rate and the within-trace SD — are pinned here on inputs whose
answers are known. No engine is touched: `report` runs over files written here,
and the #3057 `generate` nodes patch the three boundaries a generation pass
crosses — the bench list, the system prompt and the bench runner's completion
call — so naming task ids is tested without minting a single real reply.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cmp(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "eval"))
    spec = importlib.util.spec_from_file_location(
        "autoresearch_judge_compare", ROOT / "eval" / "autoresearch_judge_compare.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoresearch_judge_compare"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "TRACES", tmp_path / "traces.jsonl")
    monkeypatch.setattr(mod, "SCORES", tmp_path / "scores.jsonl")
    monkeypatch.setattr(mod, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(mod, "REFERENCE", tmp_path / "labels.yaml")
    # The live bench list is a vault+config read (`_bench_tasks`); no node here
    # reads the real one, so the default is empty and each node that needs tasks
    # supplies them. `report`'s coverage block reads it, so unpatched it would
    # make these nodes depend on what is on disk in ~/obsidian/lloyd/bench today.
    monkeypatch.setattr(mod, "_bench_tasks", lambda: [])
    return mod


def test_the_reference_leg_scores_each_judge_against_hand_labels(cmp):
    """Two tasks, one pass and one fail label per condition pair: `binary`
    tracks the labels exactly, `scalar` is inverted on one task — its pooled
    AUC must come out below binary's and the within-task AUCs must say where."""
    labels = {"bench_a": [1, 1, 1, 0, 0, 0], "bench_b": [1, 1, 1, 0, 0, 0]}
    cmp.REFERENCE.write_text("\n".join(f"{k}: {v}" for k, v in labels.items()),
                             encoding="utf-8")
    rows = []
    for t in labels:
        for (c, d), lab in zip(cmp._ORDER, labels[t]):
            rows.append({"task_id": t, "condition": c, "draw": d, "judge": "binary",
                         "k": 0, "score": float(lab)})
            s = float(lab) if t == "bench_a" else 1.0 - lab
            rows.append({"task_id": t, "condition": c, "draw": d, "judge": "scalar",
                         "k": 0, "score": s})
    leg = cmp._reference_leg(cmp._reference_labels(), rows)
    assert leg["mixed_tasks"] == ["bench_a", "bench_b"]
    assert leg["judges"]["binary"]["auc_pooled_single"] == 1.0
    assert leg["judges"]["scalar"]["auc_within_task"] == {"bench_a": 1.0, "bench_b": 0.0}
    assert leg["auc_pooled_single_binary_minus_scalar"]["diff"] == pytest.approx(0.5)


def test_auc_is_the_probability_a_good_pair_is_ordered_right(cmp):
    assert cmp._auc([1.0, 0.9], [0.1, 0.2]) == 1.0
    assert cmp._auc([0.1], [0.9]) == 0.0
    assert cmp._auc([0.5], [0.5]) == 0.5              # a tie is half a win
    assert cmp._auc([], [0.1]) is None


def test_flip_rate_counts_draw_pairs_that_disagree_on_an_ordering(cmp):
    # Two traces, three draws: A>B, A>B, A<B → draw pairs (1,2) agree, (1,3)
    # and (2,3) flip → 2/3.
    assert cmp._flip_rate({"a": [0.9, 0.8, 0.1], "b": [0.5, 0.5, 0.5]}) == pytest.approx(2 / 3)
    assert cmp._flip_rate({"a": [1.0, 1.0], "b": [0.0, 0.0]}) == 0.0
    assert cmp._flip_rate({"a": [1.0, 1.0]}) is None


def test_report_separates_a_stable_discriminating_judge_from_a_noisy_one(cmp):
    """A synthetic corpus where `binary` is perfectly stable and separates good
    from bad, and `scalar` is noisy and does not: the report must say so on
    every axis the switch decision reads."""
    traces, scores = [], []
    for t in ("bench_a", "bench_b", "bench_c"):
        for cond in ("good", "bad"):
            for d in range(2):
                traces.append({"task_id": t, "condition": cond, "draw": d,
                               "final_text": "x", "source": "model"})
                for k in range(3):
                    b = 1.0 if cond == "good" else 0.0
                    s = [0.2, 0.8, 0.5][k]
                    scores.append({"task_id": t, "condition": cond, "draw": d,
                                   "judge": "binary", "k": k, "score": b,
                                   "per": {"a1": cond == "good"}, "evidence_found": [True]})
                    scores.append({"task_id": t, "condition": cond, "draw": d,
                                   "judge": "scalar", "k": k, "score": s,
                                   "per": {"clarity": s}, "evidence_found": None})
    scores.append({"task_id": "bench_a", "condition": "good", "draw": 0,
                   "judge": "scalar", "k": 9, "score": None, "per": {}, "error": "rubric_no_json"})
    cmp.TRACES.write_text("".join(json.dumps(r) + "\n" for r in traces), encoding="utf-8")
    cmp.SCORES.write_text("".join(json.dumps(r) + "\n" for r in scores), encoding="utf-8")
    out = cmp.report()
    b, s = out["judges"]["binary"], out["judges"]["scalar"]
    assert b["auc_single_pooled"] == 1.0 and s["auc_single_pooled"] == 0.5
    assert b["within_trace_sd_mean"] == 0.0 and s["within_trace_sd_mean"] > 0.2
    assert b["tasks_agree_ge_0_9"] == 3 and s["tasks_agree_ge_0_9"] == 0
    assert s["failed"] == 1 and b["failed"] == 0
    assert out["paired_binary_minus_scalar"]["auc_single"]["diff"] == pytest.approx(0.5)
    assert json.loads(cmp.RESULT.read_text())["n_traces"] == 12


# ── #3057: pointing `generate` at named tasks, and publishing what was never measured ──

#: A synthetic bench of four ids: two a run can label and send to a model, the
#: hand-written runtime exception that needs neither, and one `requires_runtime`
#: audit task the probe cannot label at all (`_tasks` drops it).
SYNTH_BENCH = [
    {"id": "bench_a", "prompt": "say hi"},
    {"id": "bench_b", "prompt": "say bye"},
    {"id": "bench_010_safety_destructive", "prompt": "delete everything",
     "requires_runtime": True},
    {"id": "bench_014_audit_dead_wikilinks", "prompt": "audit the vault",
     "requires_runtime": True},
]


@pytest.fixture
def gen(cmp, monkeypatch):
    """The three boundaries a `generate` run crosses, all offline: the bench list
    (`_bench_tasks`, a vault+config read), `build_system_prompt` and the bench
    runner's `chat_completion` — the one call in this file that would otherwise
    reach the engine. `calls` records every completion attempt, so a node can
    prove a refusal sent nothing."""
    monkeypatch.setattr(cmp, "_bench_tasks", lambda: [dict(t) for t in SYNTH_BENCH])
    import app.prompt_builder as pb
    import scripts.autoresearch.bench_runner as br
    monkeypatch.setattr(pb, "build_system_prompt", lambda overlay_dir=None: "SYS")
    calls: list[str] = []

    def fake_chat(model, messages, max_tokens=0):
        calls.append(str(messages[-1]["content"]))
        return "a model reply", {}

    monkeypatch.setattr(br, "chat_completion", fake_chat)
    return cmp, calls


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_generate_tasks_selector_builds_jobs_for_the_named_ids_only(gen):
    """`generate --tasks bench_b --draws 3` mints two conditions × three draws for
    `bench_b` and nothing else: of the 4 synthetic ids (which with no selector is
    2 labelable × 2 × 3 + the 1 hand-written task's 2 × 3) the selector leaves
    exactly 6 rows and 6 completion calls, all one id."""
    cmp, calls = gen
    assert cmp.main(["generate", "--tasks", "bench_b", "--draws", "3"]) == 0
    rows = _rows(cmp.TRACES)
    assert {r["task_id"] for r in rows} == {"bench_b"}
    assert len(rows) == 6 == len(calls)
    assert sorted((r["condition"], r["draw"]) for r in rows) == [
        ("bad", 0), ("bad", 1), ("bad", 2), ("good", 0), ("good", 1), ("good", 2)]


def test_generate_refuses_a_runtime_id_it_cannot_label_and_names_it(gen, capsys):
    """`bench_014_audit_dead_wikilinks` is `requires_runtime` and is not the
    hand-written exception, so both of its arms would be guesses: naming it must
    be an error that prints the id, not a zero-job run that exits 0."""
    cmp, calls = gen
    assert cmp.main(["generate", "--tasks", "bench_014_audit_dead_wikilinks"]) == 2
    err = capsys.readouterr().err
    assert "bench_014_audit_dead_wikilinks" in err
    assert "requires_runtime" in err
    assert calls == []                       # nothing was sent to a model
    assert not cmp.TRACES.exists()           # and zero jobs were not written as a success


def test_generate_refuses_an_id_that_is_not_on_the_bench_or_an_empty_selector(gen, capsys):
    """The same silent-zero failure two more ways: an id that is not a live bench
    task at all, and `--tasks` given a value that holds no ids. Both name what was
    refused and exit 2; neither mints a row."""
    cmp, calls = gen
    assert cmp.main(["generate", "--tasks", "bench_999_not_a_bench_task"]) == 2
    assert "bench_999_not_a_bench_task" in capsys.readouterr().err
    assert cmp.main(["generate", "--tasks", ", ,"]) == 2
    assert "--tasks" in capsys.readouterr().err
    assert calls == [] and not cmp.TRACES.exists()


def test_generate_still_selects_the_hand_written_runtime_exception(gen):
    """The refusal must not over-reach: `bench_010_safety_destructive` is
    `requires_runtime` too, but its good/bad arms are hand-written, so naming it
    builds 2 × 2 = 4 rows with zero completion calls."""
    cmp, calls = gen
    assert cmp.main(["generate", "--tasks", "bench_010_safety_destructive",
                     "--draws", "2"]) == 0
    rows = _rows(cmp.TRACES)
    assert len(rows) == 4
    assert {r["source"] for r in rows} == {"handwritten"}
    assert calls == []


def test_generate_only_appends_and_never_rewrites_a_measured_trace_row(gen):
    """The 78 rows over 13 task_ids minted on 2026-09-24 are the corpus the #698
    decision was read off, so a later run may only add to them. Seeded here with
    two already-measured rows for `bench_a`: after a selector pass and a full pass
    the file still begins with those exact bytes, the new rows are the only
    addition, and no `(task_id, condition, draw)` triple appears twice."""
    cmp, calls = gen
    seeded = [{"task_id": "bench_a", "condition": "good", "draw": 0,
               "final_text": "MEASURED 2026-09-24", "source": "model"},
              {"task_id": "bench_a", "condition": "bad", "draw": 0,
               "final_text": "MEASURED 2026-09-24 BAD", "source": "model"}]
    seeded_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in seeded)
    cmp.TRACES.write_text(seeded_bytes, encoding="utf-8")

    assert cmp.main(["generate", "--tasks", "bench_a", "--draws", "3"]) == 0
    text = cmp.TRACES.read_text(encoding="utf-8")
    assert text.startswith(seeded_bytes)                 # the measured rows, byte for byte
    keys = [(r["task_id"], r["condition"], r["draw"]) for r in _rows(cmp.TRACES)]
    assert len(keys) == 6 and len(set(keys)) == 6        # 2 skipped, 4 appended
    assert len(calls) == 4                               # only the 4 new triples ran

    before = cmp.TRACES.read_bytes()
    calls.clear()
    assert cmp.main(["generate", "--draws", "3"]) == 0   # full pass, no selector
    after = cmp.TRACES.read_bytes()
    assert after.startswith(before)                      # append-only again
    keys = [(r["task_id"], r["condition"], r["draw"]) for r in _rows(cmp.TRACES)]
    assert len(keys) == len(set(keys))                   # still no duplicate triple
    assert _rows(cmp.TRACES)[:2] == seeded               # still the same two rows, in place


def test_report_publishes_which_live_bench_tasks_have_no_agreement_measurement(cmp, monkeypatch):
    """Coverage used to be an inference: `n_tasks` counts only what was scored, so
    13-of-28 had to be re-derived from the assertions yaml and the traces file by
    every reader. Over a bench of five ids — three scored, `bench_d` labelable but
    never traced, `bench_014_audit_dead_wikilinks` excluded as runtime — `report`
    must name both missing ids with the reason, and write it to the result file."""
    bench = [{"id": "bench_a"}, {"id": "bench_b"}, {"id": "bench_c"}, {"id": "bench_d"},
             {"id": "bench_014_audit_dead_wikilinks", "requires_runtime": True}]
    monkeypatch.setattr(cmp, "_bench_tasks", lambda: [dict(t) for t in bench])
    traces, scores = [], []
    for t in ("bench_a", "bench_b", "bench_c"):
        for cond in ("good", "bad"):
            traces.append({"task_id": t, "condition": cond, "draw": 0,
                           "final_text": "x", "source": "model"})
            for j in ("binary", "scalar"):
                scores.append({"task_id": t, "condition": cond, "draw": 0,
                               "judge": j, "k": 0, "score": 1.0 if cond == "good" else 0.0,
                               "per": {"a1": cond == "good"}, "evidence_found": [True]})
    cmp.TRACES.write_text("".join(json.dumps(r) + "\n" for r in traces), encoding="utf-8")
    cmp.SCORES.write_text("".join(json.dumps(r) + "\n" for r in scores), encoding="utf-8")

    out = cmp.report()
    cov = out["coverage"]
    assert cov["n_live"] == 5 and cov["n_scored"] == 3 and cov["n_missing"] == 2
    assert cov["missing"] == [
        {"task_id": "bench_014_audit_dead_wikilinks", "status": "excluded:requires_runtime"},
        {"task_id": "bench_d", "status": "untraced"}]
    published = json.loads(cmp.RESULT.read_text(encoding="utf-8"))
    assert published["coverage"] == cov       # written out, not just printed
    assert cov["n_live"] == cov["n_scored"] + cov["n_missing"]
