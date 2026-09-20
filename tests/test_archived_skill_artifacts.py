"""Artifacts that still describe a skill backlog #900 archived, pinned so they die.

Backlog #900 moved five unbound skills to `status: archived` on 2026-09-19 (vault
commit `978aa34b`). That status is what pulls a skill out of the system prompt's
skills index: an archived skill is not advertised and `skills_read` refuses it.
Everything written for the pre-archival world kept asserting, grading and documenting
as though those skills were live. Three such artifacts were found, and each is pinned
here by a rule rather than by a diff:

1. `EXPECTED_VAULT_WRITE_WRITERS` in `tests/test_skill_reflection_archive.py` recorded
   an exemption for two of them. That set is *discovered* by scanning live skills and
   asserted equal to the recorded dict, so once a skill is archived the equality is
   unachievable — the scan can never produce it again. The node carrying it is marked
   `live_vault`, and `scripts/automod/gate.py:335` and `:1081` both run
   `-m "not live_vault"`, so the checkout was red while every hard rung reported the
   file green (11 passed, 10 deselected). That is how it survived five days.
2. `lloyd/bench/bench_007_skill_invocation.md` told the LLM judge to reward an answer
   naming `groundskeeper-loop` — archived, unadvertised, unreadable via `skills_read`.
   The autoresearch bench was scoring "recommends a procedure the tool refuses" as
   better `skill_awareness`.
3. `architecture/morning-briefing.md` documented a live 7:00 AM `morning-briefing` job
   end to end — announce-mode delivery (a mechanism #900 proved has never existed for
   autonomy tasks), a `SKILL.md` that has never existed in this vault, and an output
   file that exists nowhere.

The rule, in one line: **an artifact may not depend on a skill the index does not
advertise.** A retired skill cannot hold an exemption, cannot be the correct answer,
and cannot be a live job.

Why these nodes are NOT marked `live_vault`
-------------------------------------------
They all read the live vault, which is precisely what that marker is for, and they
carry it nowhere. The reason is failure mode (1) itself: a marked assertion about
vault content is only ever *reported* — the hard rung deselects it, nothing enforces
it, and the tree stays red until a human happens to run the file. These three have to
be enforced, so they are unmarked, and that is paid for with the coupling it brings: a
nightly rewrite of one of these files reddens the next round's `tests` rung. Each
assertion is therefore structural — does this text name a skill the index no longer
advertises, is this claim spent — rather than a quote of prose that will legitimately
drift. The same trade is already struck 54 other times in `tests/`: `test_bench_split`
and `test_bench_invariants` read the same live bench dir unmarked.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import prompt_builder as pb
from agent_mcp.skills import _parse_frontmatter
from scripts.autoresearch.common import load_bench_tasks, load_config

ROOT = Path(__file__).resolve().parents[1]

# The same two roots `tests/test_skill_reflection_archive.py:149` scans. Deliberately
# NOT `prompt_builder._CANON_SKILLS_DIRS`: inside an automod worktree `app.paths`
# re-anchors to the round's tree, so those constants point at
# `<worktree>/home/obsidian/skills` — which does not exist — and
# `_load_skills_index()` returns None. A guard built on them would read nothing and
# pass. The claims here are about the live vault's skills, so the live paths are named
# outright and the scan has a floor below which it reports that it measured nothing.
SKILL_ROOTS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]
VAULT = Path.home() / "obsidian"

#: Skills whose *input prompt* still names a retired skill, and which are deliberately
#: left alone. Re-pointing `bench_007_skill_invocation`'s prompt from
#: `groundskeeper-loop` to `groundskeeper-survey` would break comparability with the
#: round scores already recorded against that task, so the prompt stays and only the
#: grading prose was corrected (backlog #1270). When the prompt is re-pointed, empty
#: this in the same change — `test_a_retired_name_survives_only_in_a_bench_prompt`
#: fails on drift in either direction.
KNOWN_RETIREDS_STILL_IN_A_PROMPT = {"bench_007_skill_invocation": "groundskeeper-loop"}


def _skill_files() -> list[Path]:
    """Every SKILL.md on disk in a root that exists, from the real roots."""
    found: list[Path] = []
    for root in SKILL_ROOTS:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            skill_file = entry / "SKILL.md"
            if entry.is_dir() and skill_file.is_file():
                found.append(skill_file)
    assert found, (
        f"no SKILL.md under any of {SKILL_ROOTS} — retired-vs-advertised is decided "
        "by these files, so a scan that read nothing must not be taken for a clean "
        "vault. The live vault skills root going missing is the outage to report."
    )
    return found


def _split_by_advertisement(skill_files: list[Path]) -> tuple[set[str], set[str]]:
    """Partition into (advertised, retired) using the loader's own predicate.

    `prompt_builder._is_quarantined_skill` is the exact function
    `_load_skills_index()` applies, so both sides of every assertion below come off one
    measurement. The alternative — re-deriving "retired" from a `status:` regex here —
    would be a second opinion the loader is free to contradict, which is the shape of
    the self-rewriting-baseline defect catalogued in
    `knowledge/software/guardian-data-damage-false-trip.md`.
    """
    advertised: set[str] = set()
    retired: set[str] = set()
    for skill_file in skill_files:
        (retired if pb._is_quarantined_skill(skill_file) else advertised).add(
            skill_file.parent.name
        )
    assert advertised, "no skill is advertised, so 'is it advertised' discriminates nothing"
    overlap = advertised & retired
    assert not overlap, f"skills both advertised and quarantined: {overlap}"
    return advertised, retired


@pytest.fixture(scope="module")
def advertised() -> set[str]:
    return _split_by_advertisement(_skill_files())[0]


@pytest.fixture(scope="module")
def retired() -> dict[str, str]:
    """Retired skill name -> its recorded `status`, the status only for the message."""
    names = _split_by_advertisement(_skill_files())[1]
    out: dict[str, str] = {}
    for name in names:
        for root in SKILL_ROOTS:
            skill_file = root / name / "SKILL.md"
            if skill_file.is_file():
                fm, _ = _parse_frontmatter(skill_file.read_text(encoding="utf-8", errors="replace"))
                out[name] = str(fm.get("status") or "")
                break
    return out


@pytest.fixture(scope="module")
def bench_tasks() -> list[dict]:
    cfg = load_config()
    tasks = load_bench_tasks(cfg.paths.bench_dir)
    assert tasks, (
        f"bench dir {cfg.paths.bench_dir} produced no tasks — a bench check that found "
        "no grading prose to judge found nothing, not a clean bench"
    )
    return tasks


# --- 1. The recorded exemption ------------------------------------------------------
def test_no_recorded_reflection_writer_expectation_names_a_retired_skill(retired):
    """Clause 1 of #1270, widened into the recurrence guard the fix was missing.

    #900 archived the skills; nothing but a human reading the file could notice that
    the recorded exempt set had to shrink with them, and the node whose equality
    assert enforces that is the one the gate deselects. So the rule that a retired
    skill holds no exemption now has a node of its own.

    Both recorded writer sets are checked because both are asserted-equal to a scan:
    `EXPECTED_BOUND_WRITERS` at the tier-1 discovery, `EXPECTED_VAULT_WRITE_WRITERS` at
    the PATH_ESCAPE exemption.
    """
    from tests.test_skill_reflection_archive import (
        EXPECTED_BOUND_WRITERS,
        EXPECTED_VAULT_WRITE_WRITERS,
    )

    assert EXPECTED_BOUND_WRITERS and EXPECTED_VAULT_WRITE_WRITERS, (
        "both recorded writer sets are empty, so asserting that neither names a "
        "retired skill is vacuous — delete this test if the rule ever goes away"
    )
    stale = {
        name: retired[name]
        for name in set(EXPECTED_BOUND_WRITERS) | set(EXPECTED_VAULT_WRITE_WRITERS)
        if name in retired
    }
    assert not stale, (
        f"reflection-writer expectations still carry retired skills: {stale}. Drop the "
        "entry: a retired skill has no instruction left to exempt, and the discovery "
        "scan cannot produce it (`agent_mcp/skills.py:69` returns None for a "
        "quarantined status), so the equality assert in "
        "test_skill_reflection_archive.py can never pass while it is listed."
    )
    # Positive control that this guard has teeth today: the two skills whose entries
    # #1270 deleted are still archived. Without it, `not stale` would also be satisfied
    # by their reappearing as active skills.
    assert {"nightly-prompt-audit", "nightly-behavior-test"} <= set(retired), (
        f"the two #900 skills this pin exists for are no longer archived "
        f"(retired now: {sorted(retired)}). If they were un-archived, their entries "
        "belong back in EXPECTED_VAULT_WRITE_WRITERS — do not delete this check"
    )


# --- 2. The bench grading prose -----------------------------------------------------
def test_no_bench_task_grades_a_retired_skill(bench_tasks, retired):
    """A bench body is the rubric the judge reads as "the right answer".

    `load_bench_tasks` puts the markdown below the frontmatter into `_body`, and that
    prose is what `skill_awareness` is scored against. Naming a skill the index does
    not advertise there rewards recommending a procedure `skills_read` refuses — and
    `objective_checks` uses a substring, so nothing else in the task would object.
    """
    assert retired, (
        "no skill is archived right now, so a check for retired names in grading prose "
        "can only pass — say so rather than reporting a clean bench"
    )
    offenders = {
        task["id"]: sorted(name for name in retired if name in (task.get("_body") or ""))
        for task in bench_tasks
        if any(name in (task.get("_body") or "") for name in retired)
    }
    assert not offenders, (
        f"bench grading prose names retired skills: {offenders}. Re-point it at the "
        "advertised skill that covers the same ground."
    )


def test_bench_007_grades_an_advertised_skill(bench_tasks, advertised):
    """Clause 2 as written: the grading prose names `groundskeeper-survey`, not
    `groundskeeper-loop`, and the skill it names is in the advertised set.

    The acceptance check reads lines 21-23 of the bench file, which is its grading
    prose; asserting off the parsed body instead means the clause cannot be satisfied
    by a coincidental line position, and the check survives the prose being reflowed.
    The third assertion is what makes this a boundary check rather than a grep:
    "advertised" is a property of what `prompt_builder` builds into the system prompt,
    which is the thing the bench claims to be testing.
    """
    by_id = {task["id"]: task for task in bench_tasks}
    task = by_id.get("bench_007_skill_invocation")
    assert task, "bench_007_skill_invocation is not loading, so nothing grades skill invocation"
    body = task["_body"]
    assert body, "bench_007 has no grading prose to check"
    assert "groundskeeper-loop" not in body, (
        "the grading prose rewards naming an archived skill again"
    )
    assert "groundskeeper-survey" in body, (
        "the grading prose does not name the skill #1270 re-pointed it at"
    )
    assert "groundskeeper-survey" in advertised, (
        "`groundskeeper-survey` is no longer advertised, so the bench is back to "
        "rewarding a recommendation the index cannot support — re-point at a live skill"
    )


def test_a_retired_name_survives_only_in_a_bench_prompt(bench_tasks, retired):
    """The one place a retired name may remain is the task's *input*, never its rubric.

    `prompt` is what the variant is asked; naming a retired skill there is the trap the
    task is about, and the correct answer is to name the advertised one. It is pinned
    to a recorded set rather than merely permitted, so a second retired name quietly
    joining some prompt, and the recorded prompt being re-pointed without this file
    being told, both surface as a diff instead of a shrug.
    """
    found: dict[str, object] = {}
    for task in bench_tasks:
        hits = sorted(name for name in retired if name in str(task.get("prompt") or ""))
        if hits:
            found[task["id"]] = hits[0] if len(hits) == 1 else hits
    assert found == KNOWN_RETIREDS_STILL_IN_A_PROMPT, (
        f"retired names in bench prompts changed: found {found}, recorded "
        f"{KNOWN_RETIREDS_STILL_IN_A_PROMPT}. A new one means a task is asking for a "
        "skill nothing advertises; a removed one means the prompt was re-pointed, so "
        "empty the recorded drift in the same change."
    )


# --- 3 + 4. The architecture doc ----------------------------------------------------
#: The claims `architecture/morning-briefing.md` made about a job that does not run.
#: `announce mode` names a delivery mechanism #900 proved has never existed for
#: autonomy tasks; the rest name a skill file, an output path, and a schedule slot that
#: exist nowhere. Patterns rather than line numbers, so re-asserting a claim in new
#: prose is caught, not just an untouched line.
RETIRED_CLAIMS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[Aa]nnounce mode"),
    re.compile(r"morning-briefing-latest"),
    re.compile(r"skills/morning-briefing/SKILL\.md"),
    re.compile(r"7:00 AM"),
)

#: What makes a matching line spent: both halves of the treatment commit `978aa34b`
#: gave this doc's own lines 71-72 — the text struck, and a word saying it is retired.
#: A strike alone reads as decoration beside an un-struck row; a marker alone leaves a
#: reader unable to tell what was retracted.
SPEND_MARKERS = ("Retired", "retired")

MORNING_BRIEFING_DOC = VAULT / "architecture" / "morning-briefing.md"


def test_morning_briefing_doc_retires_every_claim_about_the_job():
    """Clause 3: no unstruck, non-retired match survives in the doc.

    Measured on the live vault 2026-09-20: before vault commit `4cece7f` six lines
    matched and none carried a strike; after it the same six match and all six are
    struck *and* marked — `:22` and `:30` (announce mode), `:32` (the dead SKILL.md
    link), `:57` (the output file), plus the schedule's `:26`/`:31`. The claims stay
    legible rather than deleted, which is what #900's two struck rows did.
    """
    assert MORNING_BRIEFING_DOC.is_file(), f"{MORNING_BRIEFING_DOC} is gone"
    lines = MORNING_BRIEFING_DOC.read_text(encoding="utf-8").splitlines()
    matched = [
        (number, line)
        for number, line in enumerate(lines, start=1)
        if any(pattern.search(line) for pattern in RETIRED_CLAIMS)
    ]
    assert matched, (
        "no line of the doc names a retired claim any more, so this check has nothing "
        "to spend. Keep the retired text struck rather than deleting it — the point of "
        "#900's treatment is that the gap stays readable, and a doc that simply omits "
        "the job is what let this one go wrong in the first place."
    )
    spent = {
        number: line
        for number, line in matched
        if "~~" in line and any(marker in line for marker in SPEND_MARKERS)
    }
    unspent = [f":{number}: {line[:100]}" for number, line in matched if number not in spent]
    assert not unspent, (
        f"{len(unspent)} line(s) still assert a retired morning-briefing claim: "
        + "; ".join(unspent)
    )
    # The floor is today's measured count of spent lines, so the doc cannot shed its
    # retirement markers and pass by saying less.
    assert len(spent) >= 6, (
        f"only {len(spent)} retired claims are marked; six were retired on 2026-09-20"
    )


def test_morning_briefing_doc_names_the_job_that_carries_the_name_today(advertised):
    """Clause 4: the gap points at what actually runs, so a reader gets a successor.

    `morning-briefing` names a job that never ran and a skill directory that has never
    existed in this vault; `morning-brief-and-triage` is the advertised skill bound to
    autonomy #68 (`every-15min`, `model: secondary`, `status: draft`). The last
    assertion is the one with teeth: if someone ever creates and advertises
    `skills/morning-briefing/`, every struck claim this doc carries becomes a lie, and
    that is a doc rewrite — not this test being deleted.
    """
    text = MORNING_BRIEFING_DOC.read_text(encoding="utf-8")
    assert "68-morning-brief-triage" in text, (
        "the doc no longer names the autonomy task that carries this name today"
    )
    assert "morning-brief-and-triage" in text, (
        "the doc no longer names the skill #68 is actually bound to"
    )
    assert "morning-brief-and-triage" in advertised, (
        "`morning-brief-and-triage` is what the doc points a reader at, so it had "
        "better be advertised"
    )
    assert "morning-briefing" not in advertised, (
        "`skills/morning-briefing/` is now an advertised skill, which makes this doc's "
        "struck claims about a job that never existed false — re-point the doc at the "
        "live job instead of striking it"
    )
