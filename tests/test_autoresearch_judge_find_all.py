"""`find_all` — grading an answer set, not an answer (#647).

A "find all X" bench task used to have only `contains`/`regex` to grade it, so
padding a reply with guesses could only raise the objective fraction and a reply
naming one of N defects scored the same as one naming all N. These tests drive
`judge_trace` with synthetic traces: every classification comes from the task's
declared deterministic verifier, the rubric endpoint is never consulted for it,
five phrasings of one defect are one hit, padding lowers the composite, and a
zero-item control cannot be passed by always reporting one finding.
"""
from __future__ import annotations

import pytest

from scripts.autoresearch import judge

GOLD = ["skills/old-name/skill.md", "autonomy/42-renamed.md", "lloyd/bench/gone.md"]


def task(gold=GOLD, value="gold_member", key="path", **extra):
    check = {"type": "find_all", "value": value, "gold_items": list(gold),
             "dedupe_key": key, **extra}
    return {"id": "bench_900_find_all", "objective_checks": [check]}


def trace(*items, prose="Findings:"):
    body = "\n".join(f"- {i}" for i in items)
    return {"status": "success", "final_text": f"{prose}\n{body}".strip(),
            "tool_calls": []}


@pytest.fixture
def no_rubric(monkeypatch):
    """The answer-set path must not touch the rubric model at all."""
    def _boom(*a, **k):
        raise AssertionError("find_all classification called the rubric endpoint")
    monkeypatch.setattr(judge, "_call_rubric_llm", _boom)


@pytest.fixture
def fixed_rubric(monkeypatch):
    """judge_trace with a rubric that always answers 0.6, so composite moves only
    with the objective half."""
    calls = []

    def _fixed(prompt, model="primary", timeout=180):
        calls.append(prompt)
        return '{"overall": 0.6}'
    monkeypatch.setattr(judge, "_call_rubric_llm", _fixed)
    return calls


def grade(t, tr):
    return judge.grade_answer_set(t["objective_checks"][0], tr)


# ── clause 1: per-item table from the declared verifier, no rubric call ──────

def test_every_submitted_item_is_classified_by_the_verifier(no_rubric):
    tr = trace("broken link to skills/old-name/SKILL.md in index",
               "not a real path: skills/never-existed.md",
               "autonomy/42-renamed.md is referenced by task #7")
    score, results = judge._score_objective(task(), tr)
    table = results[0]["answer_set"]["items"]
    assert [r["verdict"] for r in table] == ["hit", "miss", "hit"]
    assert results[0]["answer_set"]["gold"] == [
        {"key": GOLD[0], "found": True}, {"key": GOLD[1], "found": True},
        {"key": GOLD[2], "found": False}]
    assert score == pytest.approx((2 / 3) * (2 / 3), abs=1e-4)


def test_the_verifier_is_the_one_the_task_names(no_rubric, monkeypatch):
    seen = []

    def spy(item, key, check):
        seen.append(key)
        return True
    monkeypatch.setitem(judge.VERIFIERS, "spy", spy)
    graded = grade(task(value="spy"), trace("a/b.md", "c/d.md"))
    assert seen == ["a/b.md", "c/d.md"]
    assert [r["verdict"] for r in graded["items"]] == ["untracked", "untracked"]
    assert graded["untracked"] == ["a/b.md", "c/d.md"]


def test_an_unknown_verifier_fails_closed_without_raising(no_rubric):
    graded = grade(task(value="ask_the_llm"), trace(GOLD[0]))
    assert graded["passed"] is False and graded["error"] == "unknown verifier"
    assert (graded["precision"], graded["recall"], graded["f1"]) == (0.0, 0.0, 0.0)


def test_judge_trace_never_calls_the_rubric_on_a_safety_miss_either(no_rubric):
    t = task()
    t["safety_critical"] = True
    out = judge.judge_trace(t, trace("wrong/thing.md"))
    assert out["composite_score"] == 0.0
    assert out["precision"] == 0.0 and out["answer_sets"][0]["items"][0]["verdict"] == "miss"


# ── clause 2: uniquified on the key, not the wording ─────────────────────────

def test_five_phrasings_of_one_defect_are_one_hit_against_five_submissions(no_rubric):
    tr = trace("skills/old-name/skill.md is dead",
               "Dead link: `skills/old-name/SKILL.md`",
               "the file skills/old-name/SKILL.md no longer exists",
               "SKILLS/OLD-NAME/SKILL.MD (renamed)",
               "see skills/old-name/skill.md.")
    graded = grade(task(), tr)
    verdicts = [r["verdict"] for r in graded["items"]]
    assert verdicts == ["hit", "duplicate", "duplicate", "duplicate", "duplicate"]
    assert all(r.get("duplicate_of") == 0 for r in graded["items"][1:])
    assert graded["submitted"] == 5
    assert graded["precision"] == pytest.approx(0.2)
    assert graded["recall"] == pytest.approx(1 / 3, abs=1e-4)


def test_two_distinct_defects_count_two_hits(no_rubric):
    graded = grade(task(), trace(f"stale ref {GOLD[0]}", f"stale ref {GOLD[1]}"))
    assert [r["verdict"] for r in graded["items"]] == ["hit", "hit"]
    assert graded["precision"] == 1.0
    assert graded["recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_text_hash_key_is_the_fact_stores_own(no_rubric):
    from app.kg_store import text_hash
    claim = "Lloyd uses vLLM"
    t = task(gold=[text_hash(claim)], key="text_hash")
    graded = grade(t, trace(claim, "  lloyd uses vllm  ", "Lloyd uses llama.cpp"))
    assert [r["verdict"] for r in graded["items"]] == ["hit", "duplicate", "miss"]


# ── clause 3: numeric P/R/F1 on the trial record; empty cases defined ────────

def test_trial_record_carries_precision_recall_f1(fixed_rubric):
    out = judge.judge_trace(task(), trace(GOLD[0], GOLD[1], "nope/x.md"))
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert len(out["answer_sets"]) == 1


def test_a_task_without_find_all_gains_no_answer_set_keys(fixed_rubric):
    out = judge.judge_trace({"objective_checks": [{"type": "contains", "value": "x"}]},
                            trace("x"))
    assert "precision" not in out and "answer_sets" not in out


def test_zero_submissions_against_real_gold_is_zero_not_a_default(no_rubric):
    graded = grade(task(), trace(prose="I looked and found nothing."))
    assert graded["submitted"] == 0
    assert (graded["precision"], graded["recall"], graded["f1"]) == (0.0, 0.0, 0.0)
    assert graded["passed"] is False


def test_zero_gold_and_zero_submissions_is_the_correct_empty_answer(no_rubric):
    graded = grade(task(gold=[]), trace(prose="None found."))
    assert (graded["precision"], graded["recall"], graded["f1"]) == (1.0, 1.0, 1.0)
    assert graded["passed"] is True


# ── clause 4: padding loses ──────────────────────────────────────────────────

def test_padding_with_wrong_extras_lowers_the_composite(fixed_rubric):
    clean = judge.judge_trace(task(), trace(GOLD[0], GOLD[1]))
    padded = judge.judge_trace(task(), trace(GOLD[0], GOLD[1], "guess/one.md",
                                             "guess/two.md", "guess/three.md"))
    assert padded["composite_score"] < clean["composite_score"]
    assert padded["recall"] == clean["recall"]


def test_contains_grading_cannot_see_the_same_padding(fixed_rubric):
    """The counterfactual: the pre-#647 way to grade this task rewards the pad
    exactly as much as the clean set, which is the hole find_all closes."""
    t = {"objective_checks": [{"type": "contains", "value": GOLD[0]},
                              {"type": "contains", "value": GOLD[1]}]}
    clean = judge.judge_trace(t, trace(GOLD[0], GOLD[1]))
    padded = judge.judge_trace(t, trace(GOLD[0], GOLD[1], "guess/one.md"))
    assert padded["composite_score"] == clean["composite_score"]


def test_stopping_at_the_first_item_also_loses(fixed_rubric):
    one = judge.judge_trace(task(), trace(GOLD[0]))
    all3 = judge.judge_trace(task(), trace(*GOLD))
    assert one["composite_score"] < all3["composite_score"]
    assert all3["objective_score"] == 1.0


# ── clause 5: the zero-item control discriminates ────────────────────────────

def test_zero_item_control_none_found_beats_one_finding(fixed_rubric):
    control = task(gold=[])
    none = judge.judge_trace(control, trace(prose="No stale references found."))
    one = judge.judge_trace(control, trace("skills/old-name/SKILL.md"))
    assert none["composite_score"] > one["composite_score"]
    assert none["objective_score"] == 1.0 and one["objective_score"] == 0.0


def test_zero_item_control_cannot_be_passed_by_a_verifier_that_accepts_anything(fixed_rubric):
    """Even when the finding verifies (an untracked item), a zero-item task's
    recall is 0 once anything is submitted: the gold says the answer is none,
    and the untracked item is surfaced for a human rather than credited."""
    control = task(gold=[], value="witness_regex", witness_regex=r"\.md$")
    out = judge.judge_trace(control, trace("skills/old-name/SKILL.md"))
    assert out["objective_score"] == 0.0
    assert out["answer_sets"][0]["untracked"] == ["skills/old-name/SKILL.md"]


# ── extraction ───────────────────────────────────────────────────────────────

def test_numbered_lists_and_a_custom_item_regex(no_rubric):
    tr = {"status": "success", "final_text": f"1. {GOLD[0]}\n2) {GOLD[1]}"}
    assert grade(task(), tr)["submitted"] == 2
    tr = {"status": "success", "final_text": f"FOUND: {GOLD[0]}; FOUND: {GOLD[2]}"}
    graded = grade(task(item_regex=r"FOUND: (\S+?)(?:;|$)"), tr)
    assert [r["verdict"] for r in graded["items"]] == ["hit", "hit"]
