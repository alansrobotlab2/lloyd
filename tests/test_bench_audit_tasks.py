"""The four set-shaped audit tasks in the live bench (#647, clause 6).

`find_all` grades an answer set only as well as its task's gold set and
verifier, and "verifier bugs are silent" is the item's own risk: a
deterministic grader that is wrong is trusted. So each task's hand-checked
positive and negative case — written in its body — is run here through the
same `judge.grade_answer_set` a round scores with, against the task file as it
sits in `~/obsidian/lloyd/bench/`, and the gold sets that are pure file facts
are re-derived from disk: label rot shows up as a failing test, not as a model
that suddenly "missed" a defect that was fixed.

`live_vault`, because every assertion reads the live vault (or `~/lloyd-data`)
and no round under test controls either. The duplicate-fact task's gold lives in
the fact store and is re-derived through `app.kg_store`, never here: conftest
points the data root at a scratch directory, and a test must not open the
production store.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from scripts.autoresearch.common import load_bench_tasks
from scripts.autoresearch.judge import DEDUPE_KEYS, VERIFIERS, grade_answer_set

pytestmark = pytest.mark.live_vault

HOME = Path.home()
BENCH = HOME / "obsidian" / "lloyd" / "bench"
VAULT = HOME / "obsidian"

AUDIT = ("bench_014_audit_dead_wikilinks", "bench_015_audit_cross_entity_fact_copies",
         "bench_016_audit_skill_dead_paths", "bench_017_audit_unresolved_task_skills")

#: (positive, negative) per task: the hand-checked cases each body records.
CASES = {
    "bench_014_audit_dead_wikilinks": (
        "- [[knowledge/hardware/pcie-bar1-p2p-gpu-offload-check]] — "
        "projects/ai/research/2026-09-07-freetoken-moe-expert-residency-vs-cutoff-2x3090.md",
        "- [[REPORT-2026-09-20]] — projects/lloyd/secondary-routing-eval/trend.md"),
    "bench_015_audit_cross_entity_fact_copies": (
        "- Hook execution logic lives entirely in the Claude Code CLI binary.",
        "- The Claude Code CLI is responsible for spawning individual MCP stdio server processes."),
    "bench_016_audit_skill_dead_paths": (
        "- ~/lloyd/scripts/memory/extract-session-log.py — historical-knowledge-refresh",
        "- /home/alansrobotlab/lloyd/lloyd/inner-voice/system-prompt.md — file-path-resolution"),
    "bench_017_audit_unresolved_task_skills": (
        "Every task's skill_name resolves.",
        "- 24-data-pipeline.md — autonomy-data-pipeline"),
}

#: A prompt that tells the model how many items exist — including that the
#: answer may be zero — lets it tune on the number (Brumley; #647 step 3).
COUNT_HINT = re.compile(
    r"\b(?:\d+|one|two|three|four|five|several|a few|no|zero|none|at least|exactly)\s+"
    r"(?:missing|dead|broken|findings?|items?|links?|facts?|paths?|defects?|tasks?|copies)\b"
    r"|\bmay be (?:zero|none|empty)\b|\bif (?:there are )?(?:none|any)\b",
    re.IGNORECASE)


@pytest.fixture(scope="module")
def tasks():
    if not BENCH.is_dir():
        pytest.skip("no live bench directory")
    loaded = {t["id"]: t for t in load_bench_tasks(BENCH)}
    missing = set(AUDIT) - set(loaded)
    assert not missing, f"audit tasks missing from the bench: {sorted(missing)}"
    return loaded


def _check(task: dict) -> dict:
    checks = [c for c in task["objective_checks"] if c.get("type") == "find_all"]
    assert len(checks) == 1, task["id"]
    return checks[0]


@pytest.mark.parametrize("task_id", AUDIT)
def test_each_declares_find_all_with_a_known_deterministic_verifier(tasks, task_id):
    check = _check(tasks[task_id])
    assert check["value"] in VERIFIERS
    assert (check.get("dedupe_key") or "normalized") in DEDUPE_KEYS
    assert isinstance(check.get("gold_items"), list)
    assert tasks[task_id].get("requires_runtime") is True, "a tool-less arm cannot audit disk"


@pytest.mark.parametrize("task_id", AUDIT)
def test_no_prompt_states_or_implies_an_item_count(tasks, task_id):
    prompt = tasks[task_id]["prompt"]
    assert not COUNT_HINT.search(prompt), COUNT_HINT.search(prompt).group(0)


@pytest.mark.parametrize("task_id", AUDIT)
def test_the_hand_checked_cases_grade_as_recorded(tasks, task_id):
    check = _check(tasks[task_id])
    positive, negative = CASES[task_id]
    pos = grade_answer_set(check, {"final_text": positive})
    neg = grade_answer_set(check, {"final_text": negative})
    if check["gold_items"]:
        assert [r["verdict"] for r in pos["items"]] == ["hit"]
        assert pos["precision"] == 1.0 and pos["recall"] > 0
    else:  # zero-item control: the empty answer is the positive
        assert pos["submitted"] == 0 and pos["passed"]
    assert [r["verdict"] for r in neg["items"]] == ["miss"]
    assert neg["recall"] == 0.0 and not neg["passed"]


@pytest.mark.parametrize("task_id", AUDIT)
def test_the_full_gold_answer_passes_and_padding_costs_precision(tasks, task_id):
    """Also proves every gold key is reachable through the task's own
    item_regex and dedupe key — a gold item no answer can produce is dead."""
    check = _check(tasks[task_id])
    answers = {
        "bench_014_audit_dead_wikilinks": [f"- [[{g}]] — x.md" for g in check["gold_items"]],
        "bench_015_audit_cross_entity_fact_copies": [
            "- The Claude Code CLI handles race conditions between simultaneous MCP tool "
            "calls via its internal JSON-RPC router.",
            "- Hook execution logic lives entirely in the Claude Code CLI binary.",
            "- CLI timeouts show a system message and execution continues."],
        "bench_016_audit_skill_dead_paths": [f"- ~/lloyd/{g} — s" for g in check["gold_items"]],
        "bench_017_audit_unresolved_task_skills": [],
    }[task_id]
    full = grade_answer_set(check, {"final_text": "\n".join(answers)})
    assert full["passed"], full
    padded = grade_answer_set(check, {"final_text": "\n".join(answers + [CASES[task_id][1]])})
    assert padded["precision"] < 1.0 and not padded["passed"]


def test_a_commentary_bullet_that_names_a_file_is_not_a_finding(tasks):
    """The first live trial of the zero-item control said "zero findings" and
    then explained itself in bullets, one naming `74-kg-mention-classifier.md`;
    a bullet-anywhere item_regex counted that as a submission and failed a
    correct answer. Only a bullet that *leads* with the answer token counts."""
    reply = ("**No broken skill references found — zero findings.**\n"
             "- No numbered task names a skill through any other key. The sole hit "
             "is `74-kg-mention-classifier.md` line 31, `skill_path: ''`.\n")
    g = grade_answer_set(_check(tasks["bench_017_audit_unresolved_task_skills"]),
                         {"final_text": reply})
    assert g["submitted"] == 0 and g["passed"]
    note = "- The link over-qualifies: see [[secondary-routing-eval-instrument-unlanded]]."
    assert grade_answer_set(_check(tasks["bench_014_audit_dead_wikilinks"]),
                            {"final_text": note})["submitted"] == 0


# ── label rot: re-derive the file-fact gold sets ─────────────────────────────

def test_dead_wikilink_gold_still_matches_the_vault(tasks):
    files = [p for p in VAULT.rglob("*") if p.is_file() and ".git" not in p.parts]
    stems = {p.stem.lower() for p in files} | {p.name.lower() for p in files}
    rels = ({str(p.relative_to(VAULT).with_suffix("")).lower() for p in files}
            | {str(p.relative_to(VAULT)).lower() for p in files})
    link = re.compile(r"(?<!!)\[\[([^\]\|#\n]+)")
    dead = set()
    for p in (VAULT / "projects").rglob("*.md"):
        text = re.sub(r"`[^`\n]*`", "", re.sub(r"```.*?```", "", p.read_text(errors="ignore"),
                                               flags=re.S))
        for m in link.finditer(text):
            t = m.group(1).strip().lower()
            if not (t in rels or ("/" not in t and t in stems)):
                dead.add(t)
    assert dead == set(_check(tasks["bench_014_audit_dead_wikilinks"])["gold_items"])


def test_skill_dead_path_gold_still_matches_the_skills(tasks):
    skip_wrong_examples = {"lloyd/inner-voice/system-prompt.md"}
    ref = re.compile(r"(?:~|/home/alansrobotlab)/lloyd/([A-Za-z0-9_./-]*[A-Za-z0-9_-])")
    dead = set()
    for p in (VAULT / "skills").rglob("SKILL.md"):
        if ".archived" in p.parts:
            continue
        for m in ref.finditer(p.read_text(errors="ignore")):
            rel = m.group(1)
            if "." not in Path(rel).name or rel in skip_wrong_examples:
                continue  # directories and the presented-as-wrong example
            if not (HOME / "lloyd" / rel).exists():
                dead.add(rel.lower())
    assert dead == set(_check(tasks["bench_016_audit_skill_dead_paths"])["gold_items"])


def test_the_zero_item_control_is_still_zero(tasks):
    unresolved = []
    for p in (VAULT / "autonomy").glob("*.md"):
        if not re.match(r"\d+-", p.name):
            continue
        fm = yaml.safe_load(p.read_text().split("---\n", 2)[1]) or {}
        slug = str(fm.get("skill_name") or "").strip()
        if slug and not (VAULT / "skills" / slug / "SKILL.md").is_file():
            unresolved.append(p.name)
    assert unresolved == [], "the control now has an answer; its gold must be relabelled"
    assert _check(tasks["bench_017_audit_unresolved_task_skills"])["gold_items"] == []
