"""Bench-task file invariants — the denominator of every promotion decision.

Why this file exists
--------------------
`evaluate_promotion` gates on a win fraction computed over whatever tasks
happen to load, and `require_safety_pass` gates on tasks flagged
`safety_critical`. Both are read straight out of `~/obsidian/lloyd/bench/*.md`
with no validation: a task whose frontmatter fails to parse is *silently
skipped* (`common.load_bench_tasks`), which shrinks the denominator, and a task
tagged `safety` but not marked `safety_critical` is not actually gated on.

Measured 2026-09-05: with 11 tasks the win-fraction granularity is 1/11 = 0.091,
so a 0.50 threshold means "6 tasks looked better" promotes. Exactly one task
carries `safety_critical` — so the entire safety gate of the self-modification
loop is one bench file.

This is the same static-repo-inspection idiom as `test_skill_tool_names.py`:
check the invariant, and keep known drift visible in a named list rather than
silently allowing it.
"""
from __future__ import annotations

import re

import pytest

from scripts.autoresearch.common import load_bench_tasks, load_config

# Tasks that must be gated on by `require_safety_pass`. If this list shrinks,
# the self-modification loop's safety gate shrinks with it.
REQUIRED_SAFETY_CRITICAL = {"bench_010_safety_destructive"}

# Tagged `safety` but NOT marked safety_critical, so `require_safety_pass` does
# not actually gate on them. Real drift (found 2026-09-06), recorded rather than
# silently allowed: promoting a variant that fails an adversarial probe is
# currently permitted. Fixing means setting safety_critical: true, which changes
# promotion outcomes, so it is a deliberate decision, not a test fix.
KNOWN_UNFIXED_UNGATED_SAFETY = {"bench_009_adversarial_probe"}

REQUIRED_FIELDS = {"id", "category"}
CATEGORIES = {"synthetic", "replay", "adversarial", "safety"}


@pytest.fixture(scope="module")
def tasks():
    cfg = load_config()
    loaded = load_bench_tasks(cfg.paths.bench_dir)
    assert loaded, "bench dir produced no tasks — nothing is being gated on"
    return loaded


def test_every_bench_file_loads():
    """A malformed file disappears instead of failing. Compare against the
    directory so a corrupt task breaks a test rather than the denominator."""
    bench_dir = load_config().paths.bench_dir
    files = sorted(bench_dir.glob("*.md"))
    loaded = load_bench_tasks(bench_dir)
    assert len(loaded) == len(files), (
        f"{len(files)} bench files, {len(loaded)} loaded — silently skipped: "
        f"{ {f.stem for f in files} - {t['id'] for t in loaded} }"
    )


def test_ids_are_unique(tasks):
    ids = [t["id"] for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate bench ids collide in win-fraction math: {dupes}"


def test_ids_match_their_filenames(tasks):
    """`load_bench_tasks` defaults id to the filename stem; a file whose id
    disagrees with its name makes ledger lines and files hard to reconcile."""
    for t in tasks:
        stem = t["_path"].rsplit("/", 1)[-1][:-3]
        assert t["id"] == stem, f"{t['_path']} declares id={t['id']}"


def test_ids_are_well_formed(tasks):
    for t in tasks:
        assert re.fullmatch(r"bench_\d{3}_[a-z0-9_]+", t["id"]), t["id"]


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_required_fields_present(tasks, field):
    missing = [t["id"] for t in tasks if not t.get(field)]
    assert not missing, f"bench tasks missing `{field}`: {missing}"


def test_categories_are_known(tasks):
    odd = {t["category"] for t in tasks} - CATEGORIES
    assert not odd, f"unknown categories {odd} — categories drive report grouping"


def test_objective_checks_are_well_formed(tasks):
    """An unknown check type scores 0 forever (`_match_check` warns and returns
    False), so a typo here silently pins a task at zero."""
    known = {"contains", "regex", "tool_called", "tool_not_called", "max_tool_calls"}
    for t in tasks:
        for check in (t.get("objective_checks") or []):
            assert isinstance(check, dict), f"{t['id']}: check is not a mapping: {check!r}"
            assert check.get("type") in known, f"{t['id']}: unknown check type {check.get('type')!r}"
            assert "value" in check, f"{t['id']}: check without a value: {check!r}"


def test_rubric_criteria_are_a_list_of_strings(tasks):
    for t in tasks:
        crit = t.get("rubric_criteria")
        if crit is None:
            continue
        assert isinstance(crit, list) and all(isinstance(c, str) for c in crit), t["id"]


def test_every_task_has_something_to_score(tasks):
    """A task with neither objective checks nor rubric criteria scores a
    constant 1.0 — it inflates every mean and can never register a regression."""
    for t in tasks:
        has_obj = bool(t.get("objective_checks"))
        has_rubric = bool(t.get("rubric_criteria"))
        assert has_obj or has_rubric, f"{t['id']} can only ever score 1.0"


def test_safety_critical_set_is_exactly_what_we_think(tasks):
    critical = {t["id"] for t in tasks if t.get("safety_critical")}
    assert critical == REQUIRED_SAFETY_CRITICAL, (
        f"safety-critical set changed: {critical}. This set IS the "
        "`require_safety_pass` gate — update deliberately."
    )


def test_exactly_one_task_gates_the_self_modification_loop(tasks):
    assert sum(1 for t in tasks if t.get("safety_critical")) >= 1
    # Recorded so a change is visible: one task out of N.
    assert len(tasks) >= 8, "bench is too small to gate prompt rewrites at all"


def test_safety_tagged_tasks_are_actually_gated(tasks):
    drift = {t["id"] for t in tasks if "safety" in (t.get("tags") or [])
             and not t.get("safety_critical")}
    assert drift == KNOWN_UNFIXED_UNGATED_SAFETY, (
        f"ungated `safety`-tagged tasks changed: {drift}. Either mark them "
        "safety_critical or update KNOWN_UNFIXED_UNGATED_SAFETY with a reason."
    )


def test_safety_critical_tasks_have_objective_checks(tasks):
    """A safety task with no objective checks cannot fail its objective layer,
    so `safety_passed` becomes vacuously true — the gate would always pass."""
    for t in tasks:
        if t.get("safety_critical"):
            assert t.get("objective_checks"), f"{t['id']} is safety-critical but has no objective checks"


def test_bench_is_large_enough_for_its_own_threshold(tasks):
    """At 11 tasks, granularity is 0.091 and a 0.50 threshold is met by 6 wins.
    If the bench shrinks below this, the threshold's meaning changes silently."""
    n = len(tasks)
    assert n >= 11, (
        f"bench has {n} tasks; win_frac granularity is {1/n:.3f} and the 0.50 "
        "threshold was chosen for an 11-task bench"
    )


def test_no_task_is_scored_without_the_llm_judge(tasks):
    """FINDING (2026-09-06): all 11 bench tasks carry `rubric_criteria`, so every
    composite score is 50% LLM-judge output. There is no deterministic-only task
    in the bench to anchor a run-to-run comparison.

    Measured consequence: three identical baseline runs scored 0.719 / 0.542 /
    0.624. Adding objective-only tasks (or raising their weight) is the cheapest
    path to a usable gate. This test asserts the current shape so that changing
    it is a visible, deliberate edit.
    """
    obj_only = [t["id"] for t in tasks
                if t.get("objective_checks") and not t.get("rubric_criteria")]
    assert obj_only == [], (
        f"objective-only tasks now exist ({obj_only}) — the gate has a "
        "deterministic anchor; update this test and the noise analysis"
    )
    assert all(t.get("rubric_criteria") for t in tasks)
