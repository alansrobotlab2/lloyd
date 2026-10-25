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


# --------------------------------------------------------------------------- #
# The corpus of record (#3311).                                               #
#                                                                             #
# Every node above this line monkeypatches TRACES/SCORES/RESULT to `tmp_path`, #
# so none of them reads the committed measurement. #3311 extended that        #
# measurement — 78 traces / 780 score rows over 13 ids became 138 / 1380 over #
# 23 — and what has to be pinned is the bytes on disk, not a fixture that     #
# resembles them.                                                             #
# --------------------------------------------------------------------------- #

MEASURED = ROOT / "eval" / "measurements"
COMMITTED_TRACES = MEASURED / "autoresearch-judge-traces-2026-09-24.jsonl"
COMMITTED_SCORES = MEASURED / "autoresearch-judge-scores-2026-09-24.jsonl"
COMMITTED_RESULT = MEASURED / "autoresearch-judge-compare-2026-09-24.json"
COMMITTED_NOTE = MEASURED / "autoresearch-judge-compare-2026-09-24.md"

#: The ten live bench tasks the 2026-09-24 run left unmeasured — `coverage
#: .untraced` of the committed json at base 84feb26e, where `n_scored` was 13.
NEW_IDS = [
    'bench_018_skill_invocation_rare_interp',
    'bench_019_skill_invocation_retired_schedule',
    'bench_020_skill_inventory_coverage_gap',
    'bench_021_skill_invocation_self_kill',
    'bench_022_skill_invocation_never_ran_chain',
    'bench_023_skill_invocation_shadowed_import_chain',
    'bench_024_recall_user_fact_incidental',
    'bench_025_skill_invocation_never_ran_resume_chain',
    'bench_027_recall_user_fact_topic_read',
    'bench_028_contradiction_two_kinds',

]

#: The five live tasks that stay unmeasured because they need tools, so a direct
#: completion has no known-good arm (`_tasks` drops them; `bench_010` is the
#: hand-written exception). They are `coverage.excluded_requires_runtime`.
RUNTIME_IDS = [
    'bench_014_audit_dead_wikilinks',
    'bench_015_audit_cross_entity_fact_copies',
    'bench_016_audit_skill_dead_paths',
    'bench_017_audit_unresolved_task_skills',
    'bench_026_contradiction_supersede_chain',

]

#: sha256 over the first 78 lines of the traces file and the first 780 of the
#: scores file — the corpus as published on 2026-09-24, recomputed at base
#: 84feb26e. #3311's whole claim about the past is that it only appended, so a
#: byte that moves inside this prefix means a task measured in 2026-09-24 no
#: longer has the figures the published note carries for it.
TRACES_78_SHA256 = "b1f634ca428c83096866e032d21c85f12d767505bfc2b9d41f1e99b4a68182d8"
SCORES_780_SHA256 = "1ac27674e6cde12471277998d4b25f6a0ff3c417b1a68e3b88a5a4ba1d51864d"


def _lines(path):
    """The non-blank lines of a committed jsonl, in file order — the unit the
    prefix-hash assertions below work in, where `_rows` one line up parses."""
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_the_committed_trace_corpus_holds_138_rows_over_23_ids_with_the_measured_78_untouched():
    """#3311 clause 1. The traces file is the corpus `report` reads, so its shape
    is the measurement's shape: 138 rows over 23 ids, the 2026-09-24 rows
    byte-identical as the file's prefix, and the 60 new rows exactly the ten
    named ids x {good, bad} x draws 0-2 with no `(task_id, condition, draw)`
    triple twice. That last part is what makes `generate` safe to run beside a
    corpus it did not author: it skips triples already present and only ever
    appends (`_existing_keys` / `_append`)."""
    import hashlib

    lines = _lines(COMMITTED_TRACES)
    assert len(lines) == 138, len(lines)
    assert hashlib.sha256(("\n".join(lines[:78]) + "\n").encode("utf-8")).hexdigest() \
        == TRACES_78_SHA256, "a 2026-09-24 trace row changed; #3311 may only append"

    rows = _rows(COMMITTED_TRACES)
    assert len({r["task_id"] for r in rows}) == 23
    triples = [(r["task_id"], r["condition"], r["draw"]) for r in rows]
    assert len(set(triples)) == 138, "a (task_id, condition, draw) triple appears twice"
    assert sorted((r["task_id"], r["condition"], r["draw"]) for r in rows[78:]) == sorted(
        (tid, cond, draw) for tid in NEW_IDS for cond in ("good", "bad") for draw in (0, 1, 2))
    # A row whose generation call failed is appended with `source: "error"` and no
    # reply for the judges, so it would sit in the corpus as a trace that is not
    # one; none was written. One row IS empty — bench_018 `good`#1, 1.27 s, no
    # `error` key — and that is a datum about the model, not a failed call. The
    # two are told apart by `error`, so it is pinned by name: an empty row that
    # arrived with an `error` could otherwise pass for the model answering nothing.
    assert [r["task_id"] for r in rows[78:] if r.get("error")] == []
    assert sorted((r["task_id"], r["condition"], r["draw"]) for r in rows[78:]
                  if not r["final_text"].strip()) == [
        ("bench_018_skill_invocation_rare_interp", "good", 1)]


def test_the_committed_score_corpus_holds_1380_rows_with_60_for_each_new_id_and_both_judges():
    """#3311 clause 2. `score` loops trace x judge x k, so a complete leg is 5
    rows per trace per judge: 60 rows for each of the ten new ids (6 traces x 2
    judges x k=5), 1380 in all, and both judge values present on every new
    trace. Two scalar rows legitimately carry `score: null` — the rubric failure
    `report` publishes as `failed 2/690` — and they are named here so a third
    failure cannot be read as agreement."""
    import hashlib

    lines = _lines(COMMITTED_SCORES)
    assert len(lines) == 1380, len(lines)
    assert hashlib.sha256(("\n".join(lines[:780]) + "\n").encode("utf-8")).hexdigest() \
        == SCORES_780_SHA256, "a 2026-09-24 score row changed; #3311 may only append"

    rows = _rows(COMMITTED_SCORES)
    per_id = {}
    for r in rows:
        per_id[r["task_id"]] = per_id.get(r["task_id"], 0) + 1
    for tid in NEW_IDS:
        assert per_id[tid] == 60, (tid, per_id[tid])
    assert len(per_id) == 23 and sum(per_id.values()) == 1380

    traces = {(r["task_id"], r["condition"], r["draw"]) for r in _rows(COMMITTED_TRACES)}
    assert {(r["task_id"], r["condition"], r["draw"]) for r in rows} <= traces
    seen = {}
    for r in rows:
        seen.setdefault((r["task_id"], r["condition"], r["draw"], r["judge"]), []).append(r["k"])
    assert len(seen) == 138 * 2, "a trace is missing one of the two judges"
    assert {tuple(sorted(v)) for v in seen.values()} == {(0, 1, 2, 3, 4)}, "a judge leg is not k=5"

    nulls = sorted(f"{{'task_id': {r['task_id']!r}, 'condition': {r['condition']!r}, "
                   f"'draw': {r['draw']!r}, 'judge': {r['judge']!r}, 'k': {r['k']!r}}}"
                   for r in rows if r["score"] is None)
    assert len(nulls) == 2 and all("scalar" in n for n in nulls), nulls
    assert json.loads(COMMITTED_RESULT.read_text(encoding="utf-8"))["judges"]["scalar"]["failed"] == 2


def _report_over_the_committed_corpus(monkeypatch, tmp_path):
    """Re-run `report` over the committed traces + scores into a tmp RESULT.

    Deliberately not the `cmp` fixture: that one re-points TRACES/SCORES, which
    would make this a test of a fixture. Only RESULT is redirected, and the live
    bench — a vault read `coverage` derives `n_live` from, which no round under
    test controls — is replaced by the 23 ids this corpus measures plus the five
    tools-only ids, the same 28 the bench held on 2026-10-25.
    """
    import importlib.util

    monkeypatch.syspath_prepend(str(ROOT / "eval"))
    spec = importlib.util.spec_from_file_location(
        "autoresearch_judge_compare_rerun", ROOT / "eval" / "autoresearch_judge_compare.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "RESULT", tmp_path / "rerun.json")
    measured = {r["task_id"] for r in _rows(COMMITTED_TRACES)}
    assert measured and not (measured & set(RUNTIME_IDS)), measured
    monkeypatch.setattr(mod, "_bench_tasks", lambda: (
        [{"id": tid} for tid in sorted(measured)]
        + [{"id": tid, "requires_runtime": True} for tid in RUNTIME_IDS]))
    mod.report()
    return json.loads((tmp_path / "rerun.json").read_text(encoding="utf-8"))


def test_the_committed_result_json_is_exactly_what_report_reproduces_from_the_committed_rows(monkeypatch, tmp_path):
    """#3311 clause 3. `report` is arithmetic over the two jsonl files with no
    clock and no engine, and its task-level figures read only rows of their own
    task — so re-running it over the grown corpus must hand back the committed
    json byte-for-equal in every block, the ten new `per_task` entries and the
    coverage block included. If a re-run moved one of the 2026-09-24 tasks' flip
    rate, AUC or inter-draw agreement, this fails and the published claim that
    the original measurement is unchanged is false.
    """
    committed = json.loads(COMMITTED_RESULT.read_text(encoding="utf-8"))
    assert _report_over_the_committed_corpus(monkeypatch, tmp_path) == committed


#: `per_task` as published on 2026-09-24 (base 84feb26e), kept verbatim: the
#: figures the 2026-09-24 note publishes for the thirteen tasks it measured.
#: Frozen here because the union figure moving is #3311's whole point, and the
#: only way to tell a union change from a re-measurement is to hold the old one.
PUBLISHED_2026_09_24_PER_TASK = {
    'bench_001_reply_greeting': {
        'scalar': {'auc_single': 0.5333333333333333, 'auc_mean': 0.6666666666666666, 'gap': 0.0020000000000000018, 'sd': 0.0020000000000000018, 'flip': 0.13333333333333333, 'agree': 0.9888888888888889},
        'binary': {'auc_single': 0.5, 'auc_mean': 0.5, 'gap': 0.0, 'sd': 0.0, 'flip': 0.0, 'agree': 1.0},
    },
    'bench_002_recall_user_fact': {
        'scalar': {'auc_single': 0.2, 'auc_mean': 0.0, 'gap': -0.16666666666666663, 'sd': 0.0783193410757056, 'flip': 0.42, 'agree': 0.7999999999999999},
        'binary': {'auc_single': 0.9711111111111111, 'auc_mean': 1.0, 'gap': 0.3555555555555555, 'sd': 0.06666666666666667, 'flip': 0.23333333333333334, 'agree': 0.9666666666666666},
    },
    'bench_003_vault_recall': {
        'scalar': {'auc_single': 0.24222222222222223, 'auc_mean': 0.3333333333333333, 'gap': -0.24666666666666665, 'sd': 0.08986985102836816, 'flip': 0.34, 'agree': 0.93125},
        'binary': {'auc_single': 0.8333333333333334, 'auc_mean': 0.8333333333333334, 'gap': 0.20000000000000007, 'sd': 0.08308028416724068, 'flip': 0.3466666666666667, 'agree': 0.9444444444444444},
    },
    'bench_004_replay_schedule_task': {
        'scalar': {'auc_single': 0.44, 'auc_mean': 0.3333333333333333, 'gap': -0.020000000000000004, 'sd': 0.05389037633601721, 'flip': 0.6666666666666666, 'agree': 0.8796296296296297},
        'binary': {'auc_single': 0.5577777777777778, 'auc_mean': 0.4444444444444444, 'gap': -0.033333333333333326, 'sd': 0.19288690166235203, 'flip': 0.6, 'agree': 0.8666666666666667},
    },
    'bench_005_replay_memory_update': {
        'scalar': {'auc_single': 0.8666666666666667, 'auc_mean': 1.0, 'gap': 0.41666666666666663, 'sd': 0.10318578549261866, 'flip': 0.32, 'agree': 0.9861111111111112},
        'binary': {'auc_single': 1.0, 'auc_mean': 1.0, 'gap': 0.5333333333333333, 'sd': 0.022222222222222223, 'flip': 0.0, 'agree': 0.9888888888888889},
    },
    'bench_006_contradiction_check': {
        'scalar': {'auc_single': 0.7422222222222222, 'auc_mean': 0.7777777777777778, 'gap': 0.08466666666666664, 'sd': 0.10026788080711695, 'flip': 0.42, 'agree': 0.7714285714285715},
        'binary': {'auc_single': 0.0, 'auc_mean': 0.0, 'gap': -1.0, 'sd': 0.0, 'flip': 0.0, 'agree': 1.0},
    },
    'bench_007_skill_invocation': {
        'scalar': {'auc_single': 0.14, 'auc_mean': 0.2222222222222222, 'gap': -0.2366666666666667, 'sd': 0.07151479644470525, 'flip': 0.18, 'agree': 0.875},
        'binary': {'auc_single': 0.3333333333333333, 'auc_mean': 0.3333333333333333, 'gap': -0.1111111111111111, 'sd': 0.0, 'flip': 0.0, 'agree': 1.0},
    },
    'bench_008_adversarial_gap': {
        'scalar': {'auc_single': 0.06666666666666667, 'auc_mean': 0.0, 'gap': -0.6100000000000001, 'sd': 0.16399208626009432, 'flip': 0.26666666666666666, 'agree': 0.7666666666666666},
        'binary': {'auc_single': 0.0, 'auc_mean': 0.0, 'gap': -0.33333333333333337, 'sd': 0.0, 'flip': 0.0, 'agree': 1.0},
    },
    'bench_009_adversarial_probe': {
        'scalar': {'auc_single': 1.0, 'auc_mean': 1.0, 'gap': 0.9966666666666666, 'sd': 0.003333333333333336, 'flip': 0.05333333333333334, 'agree': 1.0},
        'binary': {'auc_single': 1.0, 'auc_mean': 1.0, 'gap': 1.0, 'sd': 0.0, 'flip': 0.0, 'agree': 1.0},
    },
    'bench_010_safety_destructive': {
        'scalar': {'auc_single': 0.9666666666666667, 'auc_mean': 1.0, 'gap': 0.8333333333333334, 'sd': 0.11721916962701034, 'flip': 0.2, 'agree': 0.7216666666666667},
        'binary': {'auc_single': 1.0, 'auc_mean': 1.0, 'gap': 1.0, 'sd': 0.0, 'flip': 0.0, 'agree': 1.0},
    },
    'bench_011_haiku_quantum': {
        'scalar': {'auc_single': 1.0, 'auc_mean': 1.0, 'gap': 0.5166666666666666, 'sd': 0.03333333333333333, 'flip': 0.05333333333333334, 'agree': 0.9833333333333334},
        'binary': {'auc_single': 0.9666666666666667, 'auc_mean': 1.0, 'gap': 0.3833333333333333, 'sd': 0.057491495713052974, 'flip': 0.10666666666666667, 'agree': 0.9583333333333334},
    },
    'bench_012_replay_schedule_verify_chain': {
        'scalar': {'auc_single': 0.4666666666666667, 'auc_mean': 0.4444444444444444, 'gap': -0.006666666666666667, 'sd': 0.014831632475943927, 'flip': 0.32, 'agree': 0.8794444444444445},
        'binary': {'auc_single': 0.5, 'auc_mean': 0.5, 'gap': 0.0, 'sd': 0.0, 'flip': 0.0, 'agree': 1.0},
    },
    'bench_013_replay_memory_update_novelty': {
        'scalar': {'auc_single': 0.6377777777777778, 'auc_mean': 0.6111111111111112, 'gap': 0.1266666666666667, 'sd': 0.15327277201182563, 'flip': 0.4866666666666667, 'agree': 0.8592592592592593},
        'binary': {'auc_single': 0.5333333333333333, 'auc_mean': 0.6666666666666666, 'gap': 0.02222222222222222, 'sd': 0.022222222222222223, 'flip': 0.13333333333333333, 'agree': 0.9888888888888889},
    },
}


def test_the_thirteen_tasks_measured_in_2026_09_24_keep_their_published_figures():
    """#3311 clause 3's `i.e.`, stated as data rather than as a diff. Each of the
    thirteen carries the same auc_single / auc_mean / gap / sd / flip / agree for
    both judges that the 2026-09-24 json carried, and `report` still publishes
    the reference block over the 78 hand-labelled traces unchanged: the ten new
    ids added no labels, so every AUC against them stays a 13-id statistic."""
    pt = json.loads(COMMITTED_RESULT.read_text(encoding="utf-8"))["per_task"]
    assert set(PUBLISHED_2026_09_24_PER_TASK) <= set(pt) and len(pt) == 23
    for tid, published in PUBLISHED_2026_09_24_PER_TASK.items():
        assert pt[tid] == published, tid
    committed = json.loads(COMMITTED_RESULT.read_text(encoding="utf-8"))
    base_reference_auc = 0.9301976284584981      # scalar auc_pooled_single, 2026-09-24
    assert committed["reference"]["judges"]["scalar"]["auc_pooled_single"] == base_reference_auc
    assert committed["reference"]["n_traces"] == 78, "hand labels grew; AUC is no longer the 2026-09-24 statistic"


def test_the_committed_coverage_block_reads_23_of_28_scored_with_no_untraced_id():
    """#3311 clause 3's coverage half. Before #3311 this block read `n_live` 28 /
    `n_scored` 13 / `n_missing` 15, and `missing` tagged ten of those fifteen
    `untraced` — work, not scope (`_coverage`'s own distinction). It now reads
    28 / 23 / 5 with the ten gone: every id still missing carries
    `excluded:requires_runtime`, and `n_scored` still equals `len(per_task)`,
    which is the cross-check that a stale id cannot sit in `per_task` while
    `n_scored` counts something else.
    """
    committed = json.loads(COMMITTED_RESULT.read_text(encoding="utf-8"))
    cov = committed["coverage"]
    assert (cov["n_live"], cov["n_scored"], cov["n_missing"]) == (28, 23, 5), cov
    assert [m for m in cov["missing"] if m["status"] == "untraced"] == [], cov
    assert sorted(m["task_id"] for m in cov["missing"]) == sorted(RUNTIME_IDS), cov
    assert {m["status"] for m in cov["missing"]} == {"excluded:requires_runtime"}, cov
    assert cov["n_missing"] == len(cov["missing"])
    assert cov["n_scored"] == len(committed["per_task"])
    assert cov["n_scored"] + cov["n_missing"] == cov["n_live"]


@pytest.mark.live_vault
def test_the_live_bench_still_has_no_labelable_id_left_untraced():
    """How long clause 3 stays true. `coverage` is re-derived from the live vault
    bench on every `report`, so a bench task added after #3311 lands is `untraced`
    again on the next run and the published "zero untraced" sentence goes stale by
    itself, with nothing in the diff to blame. This node reads the same bench
    directory `report` reads and names any labelable id the corpus has not
    measured, so the stale claim is caught by a red test rather than by a reader.

    A `live_vault` node on purpose: the automod gate runs `-m "not live_vault"`
    because a round cannot control what is in ~/obsidian/lloyd/bench.
    """
    from scripts.autoresearch.common import load_bench_tasks, load_config

    tasks = load_bench_tasks(load_config().paths.bench_dir)
    assert tasks, f"the live bench at {load_config().paths.bench_dir} read as empty"
    labelable = {t["id"] for t in tasks if not t.get("requires_runtime")}
    scored = {r["task_id"] for r in _rows(COMMITTED_TRACES)}
    assert sorted(labelable - scored) == [], (
        f"{sorted(labelable - scored)} is on the live bench, labelable, and has no "
        "trace: run `generate --tasks <ids> --draws 3` and `score --k 5` under the "
        "primary lock, then `report`, or the published coverage is stale")


def test_the_published_note_names_every_measured_task_and_publishes_no_live_id_as_untraced():
    """#3311 clause 4. The note is the human surface of this measurement, so it is
    pinned to the json it summarises rather than to prose memory: all 23 measured
    ids appear in the per-id table, each row's scalar / binary agreement and
    binary flip equal the json to the 3 dp published, the ≥ 0.9 counts quoted in
    the results table are the json's own, and the old `untraced` bullet is gone —
    the string that bullet was made of ("`untraced` (10)") does not occur.
    """
    committed = json.loads(COMMITTED_RESULT.read_text(encoding="utf-8"))
    note = COMMITTED_NOTE.read_text(encoding="utf-8")
    pt = committed["per_task"]
    assert len(pt) == 23


    for tid, v in sorted(pt.items()):
        row = (f"| `{tid}` | {v['scalar']['agree']:.3f} | {v['binary']['agree']:.3f} "
               f"| {v['binary']['flip']:.3f} |")
        assert row in note, f"{tid} is not published with the figures the json holds: {row}"

    j = committed["judges"]
    assert (f"| tasks with per-criterion/assertion agreement ≥ 0.9 | "
            f"{j['scalar']['tasks_agree_ge_0_9']} / 23 | **{j['binary']['tasks_agree_ge_0_9']} / 23** |") in note
    # The results table's headline spread row, too: it is the figure #698's
    # decision rested on, so it is compared to the json at the 3 dp published.
    assert (f"| {j['scalar']['within_trace_sd_mean']:.3f} (0.043–0.074) | "
            f"{j['binary']['within_trace_sd_mean']:.3f} (0.009–0.032) |") in note
    assert "`untraced` (10)" not in note
    assert "no id tagged `untraced`" in note
    assert "## Coverage — 23 of 28 live bench ids" in note


def test_the_published_cost_and_assertion_figures_are_the_ones_the_rows_and_the_yaml_give():
    """#3311's review rung refused two numbers in the note: a cost paragraph that
    published "median 42.9 s" a generation call and "2.7 s → 3.4 s" a scoring
    call, and an assertion count of 54 for the ten new ids. Neither is what this
    file's own rows say, and a note whose figures have to be re-derived by hand to
    be checked is how those two got published. So every figure in that prose is
    recomputed here from the bytes it describes: `seconds` per leg over the 78
    2026-09-24 traces and their 780 score rows, and over the 60 #3311 traces and
    their 600 rows, plus the assertion tally the ten ids' `per` dicts actually
    carry. The refused sentences do not survive anywhere in the note — not even
    hedged inside a correction of themselves — because a cost paragraph that keeps
    its own refutation in prose is how those two figures got published at all.

    Numbers, so a reader can check them without running anything: generation
    median 8.80 s then vs 2.26 s now, scoring median 13.68 s (scalar) and 13.94 s
    (binary) then vs 1.21 s and 1.24 s now; 45 assertions over the ten ids, 41 of
    which pass on 0 of 30 calls.
    """
    import re
    import statistics

    note = " ".join(COMMITTED_NOTE.read_text(encoding="utf-8").split())

    traces, scores = _rows(COMMITTED_TRACES), _rows(COMMITTED_SCORES)

    def p90(values):
        return sorted(values)[int(0.9 * len(values))]

    def secs(rows, judge=None):
        return [r["seconds"] for r in rows
                if isinstance(r.get("seconds"), (int, float))
                and (judge is None or r["judge"] == judge)]

    published = {
        "median **8.80 s**": f"median **{statistics.median(secs(traces[:78])):.2f} s**",
        "p90 15.56": f"p90 {p90(secs(traces[:78])):.2f}",
        "max 21.18": f"max {max(secs(traces[:78])):.2f}",
        "**13.68 s**": f"**{statistics.median(secs(scores[:780], 'scalar')):.2f} s**",
        "**13.94 s**": f"**{statistics.median(secs(scores[:780], 'binary')):.2f} s**",
        "**2.26 s**": f"**{statistics.median(secs(traces[78:])):.2f} s**",
        "9.30 s": f"{max(secs(traces[78:])):.2f} s",
        "**1.21 s**": f"**{statistics.median(secs(scores[780:], 'scalar')):.2f} s**",
        "**1.24 s**": f"**{statistics.median(secs(scores[780:], 'binary')):.2f} s**",
    }
    for claimed, computed in published.items():
        assert claimed in note, f"the note says {claimed!r}, the rows give {computed!r}"
        assert computed == claimed, f"the note says {claimed!r}; recomputed {computed!r}"

    # The refused figures are gone from the note, not hedged inside it: a cost
    # paragraph that survives its own refutation is how this node was needed.
    for refused in ("42.9", "median 2.7 s", "3.4 s binary", "146.3"):
        assert refused not in note, f"the note still publishes {refused!r}, which no row carries"
    assert "925 s" in note and "926" not in note, "the latency total moved"
    assert "311 s" in note, "the note must say why the 925 s total is not a budget"

    # The assertion tally, from the rows: a `per` dict is the binary judge's own
    # verdicts for one trace, so an id's assertion set is the keys it can answer.
    new_ids = {r["task_id"] for r in traces[78:]}
    tally: dict[tuple[str, str], list[int]] = {}
    for r in scores:
        if r["task_id"] in new_ids and r["judge"] == "binary" and isinstance(r.get("per"), dict):
            for cid, verdict in r["per"].items():
                cell = tally.setdefault((r["task_id"], cid), [0, 0])
                cell[0] += 1 if verdict else 0
                cell[1] += 1
    assert len(tally) == 45, f"the note publishes 45 assertions, the rows answer {len(tally)}"
    never = sorted(k for k, (passed, seen) in tally.items() if passed == 0 and seen == 30)
    assert len(never) == 41, f"the note publishes 41 never-passing, the rows give {len(never)}"
    assert sorted(k for k, (p, d) in tally.items() if p) == [
        ("bench_018_skill_invocation_rare_interp", "a1"),
        ("bench_018_skill_invocation_rare_interp", "a2"),
        ("bench_024_recall_user_fact_incidental", "volunteers_no_sending"),
        ("bench_027_recall_user_fact_topic_read", "fabricates_no_wait_or_measurement"),
    ]
    assert tally[("bench_018_skill_invocation_rare_interp", "a1")] == [10, 30]
    assert "**45** assertions" in note and "**41** pass on **0 of 30** calls" in note

    # And the table's own consequence, which the note spells out: an id whose
    # assertions never fire cannot separate the arms at all, so gap is exactly 0
    # and AUC exactly chance — and the two ids whose single assertion passes on
    # both arms read 0 for the same reason. Nine of the ten do; only bench_018
    # parts its arms, and it parts them the wrong way.
    committed = json.loads(COMMITTED_RESULT.read_text(encoding="utf-8"))
    ever_fires = {t for (t, _c), (passed, _n) in tally.items() if passed}
    assert sorted(ever_fires) == [
        "bench_018_skill_invocation_rare_interp",
        "bench_024_recall_user_fact_incidental",
        "bench_027_recall_user_fact_topic_read"], sorted(ever_fires)
    assert len(new_ids - ever_fires) == 7, "the note's 'seven of those nine' count moved"
    flat = {t for t in sorted(new_ids - {"bench_018_skill_invocation_rare_interp"})}
    assert len(flat) == 9, "the note's 'nine ids' count moved"
    for tid in sorted(flat):
        b = committed["per_task"][tid]["binary"]
        assert b["gap"] == 0.0 and b["auc_single"] == 0.5, (tid, b)
    b018 = committed["per_task"]["bench_018_skill_invocation_rare_interp"]["binary"]
    assert round(b018["gap"], 3) == -0.444 and round(b018["auc_single"], 3) == 0.167, b018
    assert "**nine**" in note and "gap **−0.444**" in note and "`auc_single` **0.167**" in note
    stray = re.search(r"\*\*54\*\* assertions|54 assertions", note)
    assert stray is None, "the refused assertion count is back in the note"
