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

The fixtures are copies of the live pair, and the two halves are graded
differently on purpose. `SKILL.md` is still byte-for-byte, and the gate runs
`-m "not live_vault"` (`pytest.ini:14-19`), so a clause about a vault file is
only enforceable by the loop if the bytes it grades are also in the repo. The
TASK file is graded parsed-and-rendered and never as whole-file bytes, because
dispatch writes into `~/obsidian/autonomy/95-mitigation-drill.md` itself on
every run — `_update_task_field` (`app/autonomy.py:174`) stamps its front matter
and `_append_activity_log` (`app/autonomy.py:324`) appends a `## Activity Log`
entry to its body, and the YAML re-serialisation re-wraps the description's line
breaks. Byte equality against a copy is therefore unsatisfiable one dispatch
after the task is armed, however correct the task is (#2229). What the
`live_vault` nodes at the bottom grade of the task instead is the value a run
receives: the rendered prompt, the front matter minus the keys the scheduler
owns, and the body above the log.

The keys the live copy is allowed to carry that this one does not are not listed
here: they are read off the call sites of the engine's one front-matter writer,
because that writer is what stamps them. Listing them would leave the exclusion
one new stamp wide of red — which is the defect #2229 was filed against — while
reading them off a function every runtime write has to go through to reach the
vault at all is a guarantee the set cannot silently fall behind the engine.

What is pinned is the part that has to be true for the drill to produce a median
series at all: the command is exact, the exit code is reported rather than
inferred, `Edit`/`Write` are forbidden, and a `round_hold` refusal (exit 2) reads
as a valid outcome. That last one is the difference between a daily measurement
that survives a busy box and one whose red days get ignored — this box runs
self-mod rounds often enough that a refusal is a normal day, and a control that
really regressed (exit 1) is the day the task exists for.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest
import yaml

from app import autonomy, mitigation_state
from app import paths as app_paths

# Imported by name for two reasons, both load-bearing. `_update_task_field` is
# the symbol whose CALL SITES the key derivation below reads, so an attribute
# access spelled `autonomy._update_task_field(...)` would not be visible to a
# reader, or to a grep, looking for what this file binds the derivation to; it
# binds the same object either way. `ACTIVITY_LOG_HEADING` is imported rather
# than typed out so the split below the run log cannot fall behind the string the
# appender writes.
from app.autonomy import ACTIVITY_LOG_HEADING, _update_task_field

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

    `scripts/mitigation_drill.py`'s `main()` returns 2 when `round_hold` is
    engaged and no `--wait-free-window` was given — the branch this task's
    verbatim command takes — and on this box a round or a landing holds the pool
    often enough that a daily job which called that a failure would be red most
    weeks. A red the task itself says to ignore is how the one red that matters —
    exit 1, a control that stopped stopping — gets ignored too.

    (Cited by name, not `file:line`: #2333 added the wait branch to `main()` and
    any number written here would have moved with it.)
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


# ── fixture vs live, graded by value ─────────────────────────────────────────
# The five nodes above grade the repo's own copy; the `live_vault` nodes below
# grade the vault the scheduler globs (`app/autonomy.py:101`). What makes the
# copy a copy is not its bytes. Dispatch writes into
# `~/obsidian/autonomy/95-mitigation-drill.md` on every run:
# `_update_task_field` (`app/autonomy.py:174`) re-serialises its front matter,
# which re-stamps the engine's own keys and re-wraps the description's line
# breaks, and `_append_activity_log` (`app/autonomy.py:324`) appends a run line
# under `ACTIVITY_LOG_HEADING`. #2229 is the record of what that does to a
# byte comparison: it failed at `At index 581 diff: b'\n' != b' '` one daily run
# after task #95 was armed, and would have failed again on the next one however
# correct the task was. So SKILL.md is still graded as bytes — no dispatch path
# writes a skill — and the task is graded as the value a run receives.


def _engine_written_task_keys() -> frozenset[str]:
    """Every front-matter key the scheduler owns, read off the one writer.

    Derived, never listed. The key set is an open set owned by `app/autonomy.py`:
    today it stamps `updated` in one call, `last_attempt`/`last_run`/`next_run`
    in others and `failure_count`/`infra_failure_count`/`infra_rest_until` on the
    failure paths, and the next release may add one. A hand-written exclusion
    list is therefore one new stamp away from the red #2229 was filed for, while
    the call sites of `_update_task_field` — the only function that rewrites a
    task's front matter, and private to the module that stamps it — cover a key
    the day it is written. The bound is that writer's privacy: only
    `app/autonomy.py` calls it (the test files that do are excluded by reading
    that module's own source), so this walk is one module deep and a stamp
    written somewhere else would be a second funnel, which is itself the bug.
    """
    module = ast.parse(inspect.getsource(autonomy))

    def _literal_keys(node: ast.AST) -> set[str]:
        return {key.value for sub in ast.walk(node) if isinstance(sub, ast.Dict)
                for key in sub.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)}

    keys: set[str] = set()
    splatted: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            callee = node.func
            spelling = (callee.id if isinstance(callee, ast.Name)
                        else callee.attr if isinstance(callee, ast.Attribute)
                        else "")
            if spelling != _update_task_field.__name__:
                continue
            for kw in node.keywords:
                if kw.arg:
                    keys.add(kw.arg)                       # `updated=now_iso`
                elif isinstance(kw.value, ast.Name):
                    splatted.add(kw.value.id)              # `**fields`
                else:
                    keys |= _literal_keys(kw.value)        # `**({"next_run": x} …)`
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if node.value is None:
                    continue
                if isinstance(target, ast.Name) and target.id in splatted:
                    keys |= _literal_keys(node.value)      # `fields: dict = {…}`
                elif (isinstance(target, ast.Subscript)
                      and isinstance(target.value, ast.Name)
                      and target.value.id in splatted
                      and isinstance(target.slice, ast.Constant)
                      and isinstance(target.slice.value, str)):
                    keys.add(target.slice.value)           # `fields["next_run"] = …`
    return frozenset(keys)


#: The task-file keys dispatch is entitled to change under a copy's feet. This is
#: the whole exclusion, and `test_the_engine_key_set_is_read_off_the_scheduler`
#: is what proves it cannot be widened to swallow a graded field or emptied by
#: refactor.
ENGINE_WRITTEN_KEYS = _engine_written_task_keys()


def _graded_task_view(path: Path) -> dict:
    """What the task file is to dispatch: parsed, minus what the engine owns.

    `_parse_task_file` is the loader dispatch itself uses, so the YAML re-wrap a
    dispatch performs is already gone at this point — the values are equal even
    when the bytes are not. `_path` is the loader's own record of where it read,
    and is asserted against the argument before it goes: a comparison of two
    files must not be graded on which of the two filenames got remembered. The
    body is cut at `ACTIVITY_LOG_HEADING` (the constant the appender writes, not
    a copy of the string) because that section is written one line per run, and
    normalised for whitespace because the live copy keeps a blank line the fixture
    has never had in front of the heading.
    """
    parsed = dict(autonomy._parse_task_file(path) or {})
    assert parsed, f"{path} does not parse as an autonomy task"
    assert parsed.get("_path") == str(path), (
        f"_parse_task_file reported {parsed.get('_path')!r}, not {str(path)!r}: "
        "the loader changed and this view is no longer a view of this file")
    parsed.pop("_path")
    lines = str(parsed.pop("body", "")).split("\n")
    at = next((i for i, line in enumerate(lines)
               if line.strip().lower() == ACTIVITY_LOG_HEADING.lower()), None)
    docs = lines if at is None else lines[:at]
    return {
        "fields": {k: v for k, v in parsed.items() if k not in ENGINE_WRITTEN_KEYS},
        "docs": " ".join(" ".join(docs).split()),
    }


def _fixture_front_and_body() -> tuple[dict, str]:
    """The fixture's front matter and body, parsed, for re-serialising a copy."""
    parsed = autonomy._parse_task_file(TASK_FIXTURE)
    assert parsed, f"{TASK_FIXTURE} does not parse as an autonomy task"
    front = {k: v for k, v in parsed.items() if k not in ("_path", "body")}
    return front, str(parsed.get("body") or "")


def _write_front_matter(dst_dir: Path, front: dict, body: str, *,
                        name: str, width: int = 80) -> Path:
    """Write one task file from a parsed front matter, at a chosen wrap column."""
    path = dst_dir / name
    dumped = yaml.dump(front, default_flow_style=False, allow_unicode=True,
                       width=width)
    path.write_text(f"---\n{dumped}---\n{body}", encoding="utf-8")
    return path


def _assert_live_task_matches(task_copy: Path = TASK_FIXTURE,
                              skill_copy: Path = SKILL_FIXTURE) -> None:
    """The one fixture-vs-live comparison, so every node here grades one thing.

    Three graded surfaces and one byte surface. The rendered prompt is what a run
    actually receives, so a description or SKILL.md that no longer says what the
    drill does is caught here whatever the front matter looks like. The graded
    fields are everything else the file decides — `frequency`, `skill_name`,
    `timeout_seconds`, the `acceptance` regex that grades a run — two of which,
    `frequency` and `skill_name`, reach no prompt at all (`_build_task_prompt`,
    `app/autonomy.py:2487`, renders the id, the name, the skill and the
    description) and so are caught nowhere else. The body above the Activity Log
    is documentation and is graded as documentation: same words, any wrapping.
    Every divergence found is named in one message, so one run reports all of
    what moved rather than only the first.
    """
    assert LIVE_SKILL.read_bytes() == skill_copy.read_bytes(), (
        "the live SKILL.md differs from the bytes this file grades: skills are "
        "authored, never dispatched-into, so bytes remain the bar for this half")

    live_task, live_prompt = _render(LIVE_TASK, LIVE_SKILL)
    copy_task, copy_prompt = _render(task_copy, skill_copy)
    assert live_task and copy_task

    problems: list[str] = []
    if live_prompt != copy_prompt:
        problems.append("the prompt a run of this task would receive differs")
    live_fields = _graded_task_view(LIVE_TASK)["fields"]
    copy_fields = _graded_task_view(task_copy)["fields"]
    for key in sorted(set(live_fields) | set(copy_fields)):
        if live_fields.get(key) != copy_fields.get(key):
            problems.append(f"{key}: live={live_fields.get(key)!r} "
                            f"copy={copy_fields.get(key)!r}")
    live_docs = _graded_task_view(LIVE_TASK)["docs"]
    copy_docs = _graded_task_view(task_copy)["docs"]
    if live_docs != copy_docs:
        problems.append("the body above the Activity Log differs")

    assert not problems, (
        "the live task differs from the copy this file grades, on the values a "
        "run acts on that the scheduler does not stamp: "
        + "; ".join(problem[:120] for problem in problems))


@pytest.mark.live_vault
def test_the_live_task_and_skill_are_the_job_this_file_grades():
    """#2153 clause 5 against the vault that dispatches; #2229 clauses 1 and 4.

    This node is `test_the_live_task_and_skill_are_the_bytes_this_file_grades`,
    renamed by #2229: the byte comparison it made of the live TASK file is the
    defect, not a bar to be met, and no bytes satisfy it once task #95 has run —
    the scheduler stamps four front-matter keys, re-wraps the description and
    appends an Activity Log line to that very file on every dispatch. The
    SKILL.md byte comparison is retained, inside the helper, and so is every
    schedule assertion the node used to make.

    `@live_vault` because `~/obsidian` is not a tree this round controls
    (`pytest.ini:14-19`); run the live nodes of this file from a worktree with

        ~/lloyd/.venvs/lloyd/bin/python -m pytest \\
            tests/test_mitigation_drill_task.py -q -m ""

    Clause 4 is why the first line is not the comparison: `_find_task_file` is
    the scheduler's own lookup, so passing it is what makes this a job rather
    than documentation, and `frequency` and `skill_name` are what make "a daily
    job that receives its instructions" more than a file that happens to sit in
    `~/obsidian/autonomy/`.
    """
    assert autonomy._find_task_file(95) == LIVE_TASK, (
        "the scheduler's own lookup does not resolve id 95 to this file, so the "
        "task is documentation rather than a scheduled job"
    )
    _assert_live_task_matches()

    live_task, _ = _render(LIVE_TASK, LIVE_SKILL)
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


@pytest.mark.live_vault
@pytest.mark.parametrize("field,new_value", [
    ("frequency", "weekly"),
    ("skill_name", "nightly-vault-maintenance"),
    ("description", "Run the drill, then call Edit on mitigation_drill.json."),
], ids=["frequency", "skill_name", "description"])
def test_a_copy_that_diverges_on_one_graded_value_is_caught(tmp_path, field,
                                                            new_value):
    """#2229 clause 2: taking the byte bar off the task did not take the guard.

    Each case copies the fixture into `tmp_path`, checks the untouched copy still
    matches the live task — so the raise below can only be about the mutation,
    never about an already-divergent pair — changes exactly one value, and
    requires the comparison to raise and to name the key it caught. The three
    values are the three ways a copy can stop describing the job: `frequency` and
    `skill_name` are graded by the field comparison alone (neither reaches
    `_build_task_prompt`), and the `description` is graded by the rendered prompt
    as well, which is the seam a run actually crosses.
    """
    front, body = _fixture_front_and_body()
    _assert_live_task_matches(_write_front_matter(tmp_path, front, body,
                                                  name="pristine.md"))

    mutated = dict(front)
    mutated[field] = new_value
    with pytest.raises(AssertionError) as raised:
        _assert_live_task_matches(_write_front_matter(tmp_path, mutated, body,
                                                      name="mutated.md"))
    message = str(raised.value)
    assert message.startswith("the live task differs"), message
    assert f"{field}: live=" in message, message


@pytest.mark.live_vault
def test_a_re_wrapped_or_stamped_copy_still_matches_the_live_task(tmp_path):
    """#2229 clause 3: the two ways the copy may differ and still be the copy.

    The first half is the re-wrap a dispatch itself performs. The same front
    matter is serialised at two column widths, so the description breaks at
    different characters while parsing to the same text — `wide` and `narrow` are
    not byte-equal to each other, which is the control that this is a test about
    bytes and not a no-op — and both have to match the live task, which carries a
    description wrapped at a third width again.

    The second half is the stamp. `infra_rest_until` is a key the engine writes
    and neither file has ever carried, which is the shape of the future the
    exclusion exists for: a key nobody has listed. Its presence on the copy side
    alone must not read as a divergence, and its absence from the graded view is
    asserted rather than assumed.
    """
    front, body = _fixture_front_and_body()

    wide = _write_front_matter(tmp_path, front, body, name="wide.md", width=200)
    narrow = _write_front_matter(tmp_path, front, body, name="narrow.md", width=40)
    assert wide.read_bytes() != narrow.read_bytes(), (
        "the two serialisations came out byte-identical, so the re-wrap half of "
        "this node would grade nothing")
    wide_view, narrow_view = _graded_task_view(wide), _graded_task_view(narrow)
    assert wide_view["fields"]["description"] == narrow_view["fields"]["description"], (
        "the two wraps parse to different descriptions, so this is not the "
        "whitespace-only re-wrap a dispatch performs")
    _assert_live_task_matches(wide)
    _assert_live_task_matches(narrow)

    stamp = "infra_rest_until"
    assert stamp in ENGINE_WRITTEN_KEYS, (
        f"{stamp} is no longer derived as engine-written, so the exclusion this "
        "node is about is not the one the derivation produces")
    assert stamp not in _graded_task_view(LIVE_TASK)["fields"], (
        f"the live task now carries {stamp} itself, which makes the unseen-key "
        "case here no longer unseen — pick a key neither file has")

    stamped = dict(front)
    stamped[stamp] = "2026-10-06T13:00:00+00:00"
    stamped_path = _write_front_matter(tmp_path, stamped, body, name="stamped.md")
    assert stamp in autonomy._parse_task_file(stamped_path), (
        "the stamp did not survive serialisation, so the node graded a copy that "
        "carries no extra key at all")
    _assert_live_task_matches(stamped_path)


def test_the_engine_key_set_is_read_off_the_scheduler():
    """The exclusion is only safe while it tracks the engine, so that is graded.

    #2229's whole argument is that the excluded set has to be derived rather than
    maintained, which is worth nothing if the derivation quietly stops working:
    an empty or shrunken set is a byte comparison by another name, and an
    over-wide one is a fixture that can never diverge. Both directions are
    asserted. The floor names the stamps the scheduler writes today, so a
    refactor that renames the writer or moves the stamps off it turns this node
    red instead of silently tolerating the stamp — and the ceiling pins the keys
    this file grades inside the same set, so the exclusion cannot be widened into
    covering the thing it is compared against.
    """
    assert ENGINE_WRITTEN_KEYS, "the derivation found no writer at all"
    assert {"last_attempt", "last_run", "next_run", "updated",
            "failure_count"} <= ENGINE_WRITTEN_KEYS, (
        f"the derivation reads only {sorted(ENGINE_WRITTEN_KEYS)}; the live task "
        "file carries the stamps named above, written through "
        "`_update_task_field`, and dropping one here puts the byte-level red back"
    )
    graded_by_this_file = {"description", "name", "skill_name", "frequency",
                           "acceptance", "timeout_seconds", "id"}
    assert not graded_by_this_file & ENGINE_WRITTEN_KEYS, (
        f"the writer now stamps {sorted(graded_by_this_file & ENGINE_WRITTEN_KEYS)}, "
        "which this file grades: excluding it would make the fixture-vs-live "
        "comparison unable to fail, so grade it somewhere or name it here")
