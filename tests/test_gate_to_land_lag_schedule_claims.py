"""#2858: the schedule that publishes the gate-pass→promote lag figure.

`scripts/maintenance/gate_to_land_lag.py` shipped in `66ac3fc7` (#2759) with a
`--record` flag and no caller. #2858 files the caller — autonomy task 101 plus
`skills/gate-to-land-lag-daily-record/SKILL.md` — and both live on the vault, which a
round lands through `automod_vault_land` (vault commit `e66bc6e1`). What this file pins is
the contract those two bytes carry, because nothing else in the repo reads them: the
scheduler reads `~/obsidian/autonomy/` and the worker reads the SKILL.md, so a silently
wrong flag sentence or a dropped guard regex shows up only as a journal that stopped
accruing rows.

Where each clause of #2858 is pinned — all five live in THIS file, one node each. No other
test module names these two vault files; the corpus guards that glob
`~/obsidian/autonomy/*.md` read task 101 the way they read every other task and pin nothing
about it (`grep -rln gate-to-land-lag-daily-record tests/` answers this file alone):

| clause | node |
|---|---|
| 1 the task file loads with the fields the scheduler dispatches on | `test_task_101_loads_with_every_dispatch_field_clause_1_names_besides_the_railed_one` |
| 2 the recorder command pinned unmodified, plus what a flagless run does | `test_task_101_pins_the_recorder_command_unmodified_and_says_a_flagless_run_appends_nothing` |
| 3 the guard regex, the tool rails, and who writes the line | `test_task_101_objective_checks_carry_the_composed_guard_line_and_the_tool_rails` |
| 4 the journal is append-only and silence is a failed run | `test_task_101_forbids_rewriting_the_journal_and_names_the_failure_phrase` |
| 5 the companion skill names what each flag breaks | `test_the_companion_skill_names_what_each_recorder_flag_breaks` |

`test_the_guard_line_is_not_script_output_so_the_run_must_compose_it` is the premise under
clauses 3 and 4 and the only node here that needs no vault, so it is the one the gate's own
`-m "not live_vault"` run exercises.

Two shapes worth stating before the nodes. The nodes that read the two vault files carry
`live_vault` (pytest.ini:14): the gate runs `-m "not live_vault"` under an isolated HOME
with no vault at all, so an unmarked node here would be reading the developer's private
files or failing for their absence — both are the trap that file documents. The last node
reads only this repo and needs no mark, and it is the one that states the premise: the
guarded summary line is NOT script output, so the run composes it.

Clause 1 of #2858 asks the task file to carry `status: up_next`. It carries
`status: draft`, and the #724 dispatch-field rail is why — an unattended create that would
be dispatchable on landing is refused by `create_arms_a_task` (app/harness/policy.py:458)
on all three lanes onto these bytes, empirically on this very write
(`~/lloyd-data/safety/denials.jsonl`, row `"at": "2026-10-16T20:50:54+00:00"`, reason
`create with status: up_next`). The one-field arm is #2865, so clause 1 is deferred on
that item and NO node here asserts `up_next`: a test that pinned the rail's refusal as a
failure would be red on every run until a person armed it, and a test that accepted either
value would pin nothing. What the first node does pin is the reason the flip is one field:
every other dispatch-affecting field clause 1 names is present and loadable, so arming
needs no second change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.paths import VAULT_ROOT

live_vault = pytest.mark.live_vault

TASK_PATH = VAULT_ROOT / "autonomy" / "101-gate-to-land-lag-daily-record.md"
SKILL_PATH = VAULT_ROOT / "skills" / "gate-to-land-lag-daily-record" / "SKILL.md"
RECORDER = Path(__file__).resolve().parent.parent / "scripts" / "maintenance" / "gate_to_land_lag.py"

#: The command clause 2 pins, byte for byte. No flag but `--record`, no pipe, no
#: second command — the whole point is that one Bash call appends the row.
RECORDED_COMMAND = ("cd ~/lloyd && .venvs/lloyd/bin/python "
                    "scripts/maintenance/gate_to_land_lag.py --record")

#: Clause 3's guard regex, in the spelling the task file's `acceptance` block stores.
GUARD_PATTERN = r"gate-to-land-lag: rows=\d+ n=\d+ p50=\S+"

#: What `render()` really prints at the top of a run — the string the guard regex can
#: never match, which is the proof the summary line is the worker's, not the script's.
RENDER_HEADLINE = ("gate-pass→promote lag, from the promotions ledger and its archives "
                   "(#2759)")

JOURNAL = "~/.local/state/lloyd-automod/gate_to_land_lag.jsonl"


def _task() -> dict:
    """Task 101 through the scheduler's OWN parser, not a local regex.

    `app.autonomy._parse_task_file` is the reader the dispatch loop uses on the
    `~/obsidian/autonomy/*.md` glob, so a file that parses here and nowhere else is
    exactly the failure #2858 exists to prevent.
    """
    from app.autonomy import _parse_task_file

    if not TASK_PATH.parent.is_dir():
        # A machine with no vault checkout at all is a fact about the machine, so it
        # skips. THIS FILE MISSING is the failure the schedule can actually suffer —
        # the scheduler's glob would just stop seeing the job and nothing downstream
        # would notice — so it fails, and does not report green.
        pytest.skip(f"no vault autonomy dir at {TASK_PATH.parent}")
    assert TASK_PATH.exists(), (
        f"{TASK_PATH.name} is gone from {TASK_PATH.parent}: autonomy task 101 has left "
        f"the scheduler's glob, so `gate_to_land_lag.py --record` has no caller again")
    task = _parse_task_file(TASK_PATH)
    assert task is not None, f"{TASK_PATH.name} did not parse at all"
    assert not task.get("_yaml_broken"), (
        f"{TASK_PATH.name} came back degraded (`_yaml_broken`): the regex fallback can "
        f"recover scalars but not the nested `acceptance` block, so the run would be "
        f"dispatched with no objective_checks and nothing downstream would notice")
    return task


@live_vault
def test_task_101_loads_with_every_dispatch_field_clause_1_names_besides_the_railed_one():
    """Clause 1's fields, read back through the loader that dispatches on them.

    `id: 101`, `agent_id: worker`, `model: primary`, `frequency: daily`,
    `skill_name: gate-to-land-lag-daily-record`, and a `next_run` set — with
    `status: draft` as the one field the #724 rail would not let an unattended round
    write (#2865 is the flip). `frequency: daily` is asserted with `runs_per_day`
    ABSENT because `_frequency_interval_seconds` (app/autonomy.py:1212-1224) reads
    `runs_per_day` first and divides 86400 by it, ignoring the word `daily` beside it —
    which is how #30 ran three times a day for 47 days while declaring `daily` (#2445).
    """
    task = _task()
    assert task.get("id") == 101, f"id is {task.get('id')!r}, not 101"
    assert task.get("agent_id") == "worker", f"agent_id is {task.get('agent_id')!r}"
    assert task.get("model") == "primary", f"model is {task.get('model')!r}"
    assert task.get("frequency") == "daily", f"frequency is {task.get('frequency')!r}"
    assert not task.get("runs_per_day"), (
        f"runs_per_day is {task.get('runs_per_day')!r}: it overrides `frequency: daily` "
        f"in _frequency_interval_seconds, so this job would not run daily")
    assert task.get("skill_name") == "gate-to-land-lag-daily-record"
    assert (task.get("next_run") or "").strip(), (
        "next_run is unset: an armed task with no next_run waits for a backfill "
        "nobody owes it")
    assert task.get("status") in ("draft", "up_next"), (
        f"status is {task.get('status')!r}. `draft` is the file as #2858 landed it — the "
        f"#724 rail refuses an unattended create carrying the dispatching status — and "
        f"`up_next` is what #2865's one-field flip writes, so both are correct and "
        f"anything else (`paused`, or no status at all) means the job is dark: "
        f"DISPATCH_STOPPING_STATUSES is (\"draft\", \"paused\") at app/autonomy.py:351")
    assert task.get("timeout_seconds") == 600


@live_vault
def test_task_101_pins_the_recorder_command_unmodified_and_says_a_flagless_run_appends_nothing():
    """Clause 2: one Bash call, this command, nothing added — and why the flag is the job.

    The command is asserted as a contiguous substring because a run that inserts a pipe,
    a second flag or a second copy of the script appends a different row or none, and the
    exit code still says 0 either way.
    """
    desc = _task()["description"]
    assert RECORDED_COMMAND in desc, (
        "the description no longer pins the recorder command byte for byte:\n"
        f"  {RECORDED_COMMAND}")
    pinned = desc.split(RECORDED_COMMAND, 1)[1].lstrip()
    assert not pinned.startswith(("|", "&&", ";", "2>", ">", "--")), (
        "the command is followed by shell decoration in the same breath it is pinned; "
        "the contract is ONE unmodified call")
    low = desc.lower()
    assert "without the flag" in low and "exits 0" in low, (
        "the description must say that a flagless run still exits 0 — the reason the "
        "exit code cannot be the check")
    assert "nothing" in low and "append" in low, (
        "the description must say a flagless run APPENDS NOTHING")
    assert RECORDER.exists(), f"{RECORDER} is gone; the pinned command names it"
    script = RECORDER.read_text(encoding="utf-8")
    assert '"--record"' in script, "--record is no longer a flag of the pinned script"
    assert 'RECORD_NAME = "gate_to_land_lag.jsonl"' in script, (
        "the recorder's artefact name changed; the journal path this task verifies and "
        "#2858's checks are written against it")


@live_vault
def test_task_101_objective_checks_carry_the_composed_guard_line_and_the_tool_rails():
    """Clause 3: the regex bites, and the description says the run writes the line itself.

    The regex is compiled and tried against a well-formed composed line and against what
    the script's own renderer prints, so a future edit cannot quietly point the check at
    text the recorder emits (it emits none) or at a line missing a figure.
    """
    task = _task()
    acceptance = task.get("acceptance") or {}
    checks = acceptance.get("objective_checks") or []
    kinds = [(c.get("type"), str(c.get("value"))) for c in checks]
    regexes = [v for t, v in kinds if t == "regex"]
    assert GUARD_PATTERN in regexes, (
        f"objective_checks lost the guard regex {GUARD_PATTERN!r}; it holds "
        f"{'landed-outcomes' if any('landed-outcomes' in r for r in regexes) else regexes}")
    guard = re.search(f"^({GUARD_PATTERN})$", "gate-to-land-lag: rows=7 n=1173 p50=19.3175",
                      re.M)
    assert guard, "the pinned regex does not match a well-formed summary line"
    assert re.search(GUARD_PATTERN, RENDER_HEADLINE) is None, (
        "the guard now matches the renderer's own headline, which would pass a run that "
        "composed nothing")
    assert re.search(GUARD_PATTERN, "gate-to-land-lag: rows=1 n=1173") is None, (
        "the guard accepts a line with no p50: a run that verified the row but reported "
        "no figure would pass")
    for want in (("tool_not_called", "Edit"), ("tool_not_called", "Write"),
                 ("tool_called", "Bash")):
        assert want in kinds, f"objective_checks lost {want[0]}: {want[1]} — {kinds}"
    low = task["description"].lower()
    assert "wc -l" in low and "tail -1" in low, (
        "the description must name where each figure comes from: rows= off `wc -l`, "
        "n= and p50= off `tail -1` of the journal")
    assert "compose" in low, (
        "the description must say the summary line is composed by the worker, because "
        "the script never prints it")


@live_vault
def test_task_101_forbids_rewriting_the_journal_and_names_the_failure_phrase():
    """Clause 4: the journal is append-only history, and silence is a failed run.

    `failed because` is checked as the literal string, not a paraphrase:
    `_detect_silent_failures` (app/autonomy.py:31) fires on that phrase in the final
    prose, and the runner's own turn ends 0 whatever the script did — so without the
    literal phrase a journal that stopped accruing rows is a green row on the fleet board.
    """
    low = _task()["description"].lower()
    assert "never edit" in low and JOURNAL.split("/")[-1] in low, (
        "the description no longer forbids editing the journal by name")
    for banned in ("truncate", "re-sort", "dedupe"):
        assert banned in low, f"the ban on {banned!r} was dropped"
    assert "appending a second row is correct" in low, (
        "a same-day rerun appends a second row and that is correct; without this line a "
        "future run 'fixes' the duplicate by rewriting history")
    assert "failed because" in _task()["description"], "the literal phrase is gone"
    assert "recorded" in low and "recorded →" in _task()["description"], (
        "the receipt line the run looks for is `recorded →`; without naming it a run "
        "cannot tell an append from a flagless report")
    assert "ends 0" in low, (
        "the description must say the runner's turn ends 0 whatever the script did")


@live_vault
def test_the_companion_skill_names_what_each_recorder_flag_breaks():
    """Clause 5: the SKILL.md the run is actually handed, and the flags it must not add.

    `skill_name` is load-bearing: the worker's prompt is built from the SKILL.md it
    names, so a slug that no longer matches the directory means the run is handed a
    different protocol while this file's pins still pass.
    """
    if not SKILL_PATH.exists():
        pytest.skip(f"no skill at {SKILL_PATH}")
    skill = SKILL_PATH.read_text(encoding="utf-8")
    task = _task()
    slug = task.get("skill_name")
    assert SKILL_PATH.parent.name == slug, (
        f"task 101 names skill {slug!r} but the file lives at "
        f"{SKILL_PATH.parent.name}/SKILL.md — the run would be handed no protocol")
    assert f"name: {slug}" in skill.split("---")[1], (
        "the SKILL.md front matter no longer carries the matching `name:`")
    for flag in ("--record", "--state-dir", "--json"):
        assert flag in skill, f"the skill stopped naming {flag}, so a run can add it"
    state_dir = skill.split("--state-dir", 1)[1][:400]
    assert "appends" in state_dir or "file it appends" in state_dir, (
        "--state-dir must be explained by what it breaks: it moves the file the recorder "
        "appends OFF the live state dir, so the live series gains no row")
    json_part = skill.split("--json", 1)[1][:400]
    assert "report" in json_part or "printed" in json_part, (
        "--json must be explained by what it breaks: it replaces the printed report with "
        "the raw row, so the lines the report contract quotes are gone")
    assert "recorded →" in skill, (
        "the skill must name the append's own receipt line `recorded →`: without it a "
        "run cannot tell an appended row from a flagless report that exits 0")
    assert JOURNAL in skill, (
        f"the skill must name the artefact {JOURNAL} — the ban on rewriting it and the "
        f"`wc -l`/`tail -1` verification are both written against that path")


def test_the_guard_line_is_not_script_output_so_the_run_must_compose_it():
    """The premise under clauses 3 and 4, pinned against this repo alone.

    Measured for triage on 2026-10-16: a flagless run piped into
    `grep -cE 'gate-to-land-lag: rows=.* n=.* p50='` returned **0**. Pinned here rather
    than asserted only in prose because the day the recorder learns to print that line is
    the day the composed figure becomes a duplicate of it — and that day is a change to
    this file, not a silent drift in a journal.
    """
    script = RECORDER.read_text(encoding="utf-8")
    assert "gate-to-land-lag:" not in script, (
        "the recorder now prints the guarded summary line itself; the task file and "
        "SKILL.md say the worker composes it, so either the pins or the script moves")
    assert GUARD_PATTERN not in script
    assert RENDER_HEADLINE.split(",")[0] in script, (
        f"render() no longer heads its report with {RENDER_HEADLINE.split(',')[0]!r}; "
        f"the negative controls above are written against that headline")
