"""Task #95's rendered prompt must run the drill and nothing else (#2153 clause 5).

An autonomy task's markdown body looks like a specification and reaches no run:
`_build_task_prompt` (`app/autonomy.py:2487`) assembles the silent-run hint, the
task header, the `skill_name` SKILL.md and the front-matter `description` — never
the body. #463 caught the first instance (#47's `Fact Store Consolidation` phase,
instructed in a body, present in 0 of 4 run records), and
`~/obsidian/autonomy/90-corpus-shape-trend.md:62` states the rule in the file
itself. So the only version of this task a worker can obey is the two carriers
this file renders, and a step that exists only below the `---` is a step that
will never happen.

The fixtures are byte-for-byte copies of the live pair (`cmp` clean at
2026-10-04T03:35Z). That is deliberate: the gate runs `-m "not live_vault"`
(`pytest.ini:10-13`), so a clause about a vault file is only enforceable by the
loop if the bytes it grades are also in the repo — and the `live_vault` node at
the bottom is what stops the copy from drifting away from what dispatches.

What is pinned is the part that has to be true for the drill to produce a median
series at all: the command is exact, the exit code is reported rather than
inferred, `Edit`/`Write` are forbidden, and a `round_hold` refusal (exit 2) reads
as a valid outcome. That last one is the difference between a daily measurement
that survives a busy box and one whose red days get ignored — this box runs
self-mod rounds often enough that a refusal is a normal day, and a control that
really regressed (exit 1) is the day the task exists for.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app import autonomy, mitigation_state
from app import paths as app_paths

ROOT = Path(__file__).resolve().parents[1]
TASK_FIXTURE = ROOT / "tests/fixtures/mitigation_drill/task-95.md"
SKILL_FIXTURE = ROOT / "tests/fixtures/mitigation_drill/SKILL.md"

LIVE_TASK = Path.home() / "obsidian/autonomy/95-mitigation-drill.md"
LIVE_SKILL = Path.home() / "obsidian/skills/mitigation-drill/SKILL.md"

#: The command the item names, verbatim. `; echo "EXIT=$?"` follows it in both
#: carriers: the drill exits non-zero on purpose (2 = refused, 1 = a control
#: failed) and an absorbed exit code is the only way the runner learns it.
DRILL_COMMAND = "cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.mitigation_drill"

#: Lines that live in the body and nowhere else. Their absence from the prompt is
#: the delivery claim; their presence in the file is the control that makes that
#: absence mean "not delivered" rather than "not written".
BODY_ONLY_PROSE = ("Why exit 2 is a report and not a failure",
                   "Verification after landing")


def _render(task_path: Path, skill_path: Path) -> tuple[dict, str]:
    """Parse and render one pair through the real loader and the real builder."""
    task = autonomy._parse_task_file(task_path)
    assert task, f"{task_path} does not parse as an autonomy task"
    assert skill_path.is_file(), f"{skill_path} is missing; the run would get no skill"
    return task, autonomy._build_task_prompt(task, skill_path.read_text(encoding="utf-8"))


def _checks(task: dict) -> dict[str, list]:
    """`objective_checks` as `{type: [values]}`."""
    out: dict[str, list] = {}
    for check in ((task.get("acceptance") or {}).get("objective_checks")) or []:
        out.setdefault(str(check.get("type")), []).append(check.get("value"))
    return out


def test_the_rendered_prompt_runs_the_drill_and_reports_the_exit_code():
    """#2153 clause 5, against the bytes the repo carries.

    Rendered rather than grepped: `_build_task_prompt` is the boundary between a
    file a person edits and a prompt a model obeys, and each of these properties
    could hold in the file and still fail to arrive — which is exactly how #47's
    contradiction-resolution phase went unexecuted for its whole life.
    """
    task, prompt = _render(TASK_FIXTURE, SKILL_FIXTURE)

    assert DRILL_COMMAND in prompt, (
        "the drill command moved out of the delivered prompt: the run would then "
        "be told to report a measurement it never took"
    )
    assert 'echo "EXIT=$?"' in prompt, (
        "without the absorbed exit code the runner cannot tell exit 2 (refused, "
        "fine) from exit 1 (a control regressed), which is the whole report"
    )
    assert "Bash" in prompt, "the check below grades `tool_called: Bash`"
    assert "VERBATIM" in prompt and "verbatim" in prompt, (
        "the JSON is the record; a paraphrased run leaves the median unreadable"
    )
    assert "mitigation drill: exit <N>" in prompt, (
        "the summary line the acceptance regex grades is no longer spelled out"
    )
    assert "Never call Edit or Write" in prompt, (
        "the read-only rule must reach the runner, not just the reviewer: the "
        "drill script is the only writer of its state file"
    )


def test_the_prompt_names_a_round_hold_refusal_as_valid_outcome_not_a_failure():
    """#2153 clause 5's exit-2 half, and the reason it is worth a node.

    `scripts/mitigation_drill.py:268-270` returns 2 when `round_hold` is engaged,
    and on this box a round or a landing holds the pool often enough that a daily
    job which called that a failure would be red most weeks. A red the task
    itself says to ignore is how the one red that matters — exit 1, a control
    that stopped stopping — gets ignored too.
    """
    task, prompt = _render(TASK_FIXTURE, SKILL_FIXTURE)

    assert "round_hold" in prompt, (
        "the refusal is no longer named by the flag that causes it, so a runner "
        "reading `refused` has nothing to check the report against"
    )
    assert "Exit 2" in prompt and "VALID outcome" in prompt, (
        "exit 2 is no longer labelled a valid outcome in the delivered prompt"
    )
    for phrase in ("not a failure", "not a retry"):
        assert phrase in prompt, f"{phrase!r} is gone from the delivered prompt"
    assert re.search(r"[Ee]xit 1", prompt), (
        "exit 1 must still be named as the real finding beside exit 2's "
        "innocence; the two only mean anything against each other"
    )


def test_each_carrier_alone_carries_the_command_and_the_read_only_rule():
    """#2153 clause 5's two carriers, graded separately.

    `_build_task_prompt` concatenates the SKILL.md and the description, so a
    union-only assertion cannot tell which of the two holds an instruction, and
    the one that does not is free to rot. Mutation-probed: deleting
    `; echo "EXIT=$?"` from `SKILL.md` alone left every union assertion green,
    because the description still carried it. That matters because the two files
    are not edited together — the SKILL.md is the file a round rewrites when the
    drill's behaviour changes, the front-matter `description` is a YAML scalar a
    person re-wraps to fit the task card, and a protocol that survives in exactly
    one of them is one careless edit from being in neither. Each carrier
    therefore has to stand on its own for what makes the run safe and legible:
    the exact command, the exit code surfaced rather than absorbed, the
    Edit/Write prohibition, a `round_hold` refusal labelled valid, exit 1 named as
    the finding, and the summary line the task's own `regex` check matches.

    (A missing skill does not degrade to a description-only prompt:
    `app/autonomy.py:4009-4013` returns `Skill not found` and the run never dispatches.
    This node is not insurance against that; it is insurance against a carrier
    quietly ceasing to carry.)
    """
    task, _ = _render(TASK_FIXTURE, SKILL_FIXTURE)
    description = str(task.get("description") or "")
    skill = SKILL_FIXTURE.read_text(encoding="utf-8")

    for carrier, text in (("SKILL.md", skill), ("description", description)):
        assert DRILL_COMMAND in text, f"{carrier} lost the drill command"
        assert 'echo "EXIT=$?"' in text, (
            f"{carrier} no longer surfaces the exit code, so a run cannot tell a "
            "refused drill (exit 2) from a control that regressed (exit 1)"
        )
        assert re.search(r"Edit or Write", text), (
            f"{carrier} no longer forbids Edit/Write, which is what the "
            "task's own `tool_not_called` checks then grade a run against"
        )
        assert re.search(r"round_hold", text) and re.search(
            r"valid outcome", text, re.IGNORECASE), (
            f"{carrier} no longer labels a round_hold refusal a valid outcome"
        )
        assert re.search(r"exit\s+1", text, re.IGNORECASE), (
            f"{carrier} no longer names exit 1 as the real finding beside the "
            "refusal's innocence"
        )
        # The summary line is what the task's own `regex` check grades, and the
        # prompt the run receives is either this file alone (a skill that failed
        # to load) or this file plus the other, so the shape has to survive in
        # whichever one is in force.
        assert "mitigation drill: exit <N>" in text, (
            f"{carrier} no longer spells the summary line the acceptance regex "
            "matches: a run obeying it would fail its own check"
        )


def test_the_objective_checks_grade_the_report_the_prompt_asks_for():
    """The grading seam: the regex in the task's own front matter has to accept
    the line the prompt instructs and reject a paraphrase of it.

    A `regex` check nobody can satisfy fails every good run, and one so loose it
    matches prose passes every bad one. Both halves are asserted against the
    pattern as parsed from the fixture, so the pattern itself is what is tested.
    """
    task, _ = _render(TASK_FIXTURE, SKILL_FIXTURE)
    checks = _checks(task)

    assert checks.get("tool_called") == ["Bash"], checks
    assert set(checks.get("tool_not_called") or []) == {"Edit", "Write"}, checks
    limits = [int(v) for v in checks.get("max_tool_calls") or []]
    assert limits and limits[0] <= 4, (
        f"max_tool_calls={limits}: this task is one command and one report, and a "
        "generous ceiling is what lets a run go editing"
    )
    patterns = [v for v in checks.get("regex") or [] if v]
    assert len(patterns) == 1, checks
    graded = re.compile(patterns[0])

    reported = graded.search(
        "mitigation drill: exit 2 | session_cancel in-flight - | pool_pause dispatch-only"
    )
    assert reported, (
        "the acceptance regex rejects the summary line the prompt instructs, so "
        "every refused day grades as a failure"
    )
    assert graded.search("mitigation drill: exit 0 | session_cancel in-flight "
                         "0.021 | pool_pause dispatch-only")
    assert not graded.search("the drill seems to have run fine today, all good"), (
        "the regex accepts a paraphrase with no exit code in it, which is the "
        "report shape this task was written to forbid"
    )


def test_the_body_stays_documentation_and_reaches_no_run():
    """The seam itself, on this task's own bytes.

    `tests/test_autonomy_task_prompt_claims.py` pins `_build_task_prompt`'s
    mechanism generally; this pins that #95's body actually holds instruction-like
    prose that a reader could mistake for the run's brief, so the caveat at the
    top of the file is load-bearing rather than decorative.
    """
    task, prompt = _render(TASK_FIXTURE, SKILL_FIXTURE)
    body = str(task.get("body") or "")

    for line in BODY_ONLY_PROSE:
        assert line in body, (
            f"{line!r} is gone from the body, so its absence from the prompt "
            "below proves only that nothing was written"
        )
        assert line not in prompt, (
            f"{line!r} reached the delivered prompt from the body, which means "
            "the renderer changed under #463's rule"
        )


def test_the_skill_names_the_history_cap_the_code_keeps():
    """Prose-to-code seam: the skill promises a median over a number of readings,
    and `mitigation_state.HISTORY_CAP` is the only thing that decides whether a
    five-day-old reading is still in that median.
    """
    _, prompt = _render(TASK_FIXTURE, SKILL_FIXTURE)

    named = {int(n) for n in re.findall(r"(\d+) readings?", SKILL_FIXTURE.read_text())}
    assert named == {mitigation_state.HISTORY_CAP}, (
        f"the skill promises a median over {sorted(named)} readings while "
        f"`mitigation_state.HISTORY_CAP` is {mitigation_state.HISTORY_CAP}; the "
        "number in the instruction is what an operator reads into the route"
    )
    state_name = app_paths.MITIGATION_DRILL_STATE.name
    assert state_name in prompt, (
        f"the skill no longer names {state_name}, the file the drill writes and "
        "the file the run is told never to edit"
    )


@pytest.mark.live_vault
def test_the_live_task_and_skill_are_the_bytes_this_file_grades():
    """#2153 clause 5, against the vault that dispatches.

    `@live_vault` because `~/obsidian` is not a tree this round controls
    (`pytest.ini:10-13`); run it from a worktree with

        ~/lloyd/.venvs/lloyd/bin/python -m pytest \\
            tests/test_mitigation_drill_task.py::test_the_live_task_and_skill_are_the_bytes_this_file_grades -q -m ""

    The fixtures above are gate-enforced; this node is what makes them the same
    bytes the scheduler globs (`app/autonomy.py:101`) rather than a copy that has
    quietly diverged from it.
    """
    assert autonomy._find_task_file(95) == LIVE_TASK, (
        "the scheduler's own lookup does not resolve id 95 to this file, so the "
        "task is documentation rather than a scheduled job"
    )
    assert LIVE_TASK.read_bytes() == TASK_FIXTURE.read_bytes(), (
        "the live task file differs from the bytes this file grades"
    )
    assert LIVE_SKILL.read_bytes() == SKILL_FIXTURE.read_bytes(), (
        "the live SKILL.md differs from the bytes this file grades"
    )

    live_task, live_prompt = _render(LIVE_TASK, LIVE_SKILL)
    assert live_prompt == _render(TASK_FIXTURE, SKILL_FIXTURE)[1]
    assert str(live_task.get("frequency")) == "daily"
    assert autonomy._frequency_interval_seconds(live_task) == 86_400.0, (
        "the rendered task is not a daily one, so the median series the item "
        "promises has no schedule behind it"
    )
    assert str(live_task.get("skill_name")) == "mitigation-drill", (
        "the task's skill_name no longer resolves to the skill whose body is in "
        "the prompt: a run would be dispatched with no instructions"
    )
    assert (Path.home() / "obsidian" / "skills"
            / str(live_task.get("skill_name")) / "SKILL.md").is_file()
