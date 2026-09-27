"""`skill_lint`'s DEAD_SYSTEMD_UNIT category: a command must name a unit that exists.

Backlog **#1580**. `groundskeeper-survey/SKILL.md` told the agent to run
`systemctl --user status lloyd-groundskeeper-survey.service` and
`systemctl --user list-timers lloyd-groundskeeper-survey.timer` in a Bash block,
two lines under its own banner saying the survey was retired on 2026-09-23 (#1012),
which is when both units were deleted. The step cannot work: the unit answer is
`Unit lloyd-groundskeeper-survey.service could not be found.` and the timer listing
is empty. The measured damage is in the run record, not the unit file —
`~/lloyd-data/autonomy-runs/36/` holds 73 records for that task, and reading three of
them (2026-09-09, 2026-09-11, 2026-09-19) every one contains a `could not be found`
the agent then narrated as part of a stale-queue incident report.

Nothing in the weekly lint could see it. The eight categories at base all check
shapes a description or body can take, and `MISSING_SCRIPT` resolves repo *scripts*
against the tree with `_SCRIPT_PATH_RE`, which needs a `~/lloyd/` anchor **and** a
`.py`/`.sh`/`.js`/`.ts`/`.mjs` extension — a bare `lloyd-…​.timer` matches neither.
Confirmed by absence at base: `git grep -c "systemd\\|systemctl\\|\\.timer\\|\\.service"
-- scripts/skill_lint.py` → no file matched, against the positive control
`git grep -c MISSING_SCRIPT -- scripts/skill_lint.py` → 7. So #70, weekly, passed
over the skill clean, and the one thing it reported about it was a clipped
description (`~/obsidian/autonomy/skill-lint-report.json`, 2026-09-26).

The category is ON, which is only admissible because of how it decides what is a
claim: only a `systemctl` invocation is one. `agent-services/systemd/` and
`~/.config/systemd/user/` hold no fewer than 25 template instances
(`getty@getty.service`, `dbus-org.freedesktop.resolve1.service`, …) that are not
unit *definitions*, so resolving a cited name against installed file names alone
would call `lloyd-guardian@2.service` missing while `lloyd-guardian@.service` sits in
the repo. And the retired skill, correctly rewritten, must still *name* the units it
lost — a check that greps unit-shaped tokens would flag the retirement text and need
a ledger to silence it. `test_a_prose_retirement_mention_of_the_same_unit_stays_clean`
is the positive control that pins both properties; the same suppression logic as
`PHANTOM_TOOL`'s, which earns the category its `yes` row in `CATEGORY_TRUST`.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "skill_lint.py"

_SPEC = importlib.util.spec_from_file_location("skill_lint", SCRIPT)
assert _SPEC and _SPEC.loader
sl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sl)

#: The skills on this box whose commands name a unit no unit root has, measured at
#: this commit with `~/lloyd/.venvs/lloyd/bin/python -m pytest
#: tests/test_skill_dead_units.py -q -s` printing the live result. Three, not one:
#: the item predicted `groundskeeper-survey` (fixed by this round's vault commit, so
#: it is now legitimately clean), `voice-mode` and `local-llm-gotchas`, and the check
#: also caught `qmd-collection-management`, whose `systemctl --user restart
#: openclaw-gateway.service` names a unit absent from `agent-services/systemd/`,
#: `~/.config/systemd/user/` AND `/etc/systemd/system/` — while
#: `stale-process-cleanup/SKILL.md:68` asserts that exact unit is what "host systemd"
#: manages. The set is pinned exact, and exact in both directions, because the item's
#: stated failure mode for this category is being "tuned down to zero": a check that
#: silently stops naming its corpus is the bug, not the finding.
EXPECTED_DEAD_UNIT_SKILLS = {"local-llm-gotchas", "qmd-collection-management",
                             "voice-mode"}

#: A unit that is genuinely absent on this box — verified with
#: `find ~/lloyd/agent-services/systemd ~/.config/systemd/user /etc/systemd/system
#: -name 'lloyd-no-such-unit*'` → no output — so the live corpus can be checked for
#: its absence without inventing a name that might land.
ABSENT_UNIT = "lloyd-no-such-unit.service"


def _write_skill(root: Path, slug: str, body: str) -> Path:
    """One active skill at `<root>/<slug>/SKILL.md`, the way the loader wants it."""
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        "---\n"
        f"name: {slug}\n"
        "description: Fixture skill for the dead-systemd-unit check.\n"
        "status: active\n"
        "---\n"
        f"# {slug}\n\n{body}\n",
        encoding="utf-8",
    )
    return d


def _records(skills_root: Path):
    from agent_mcp.skills import iter_active_skills
    return list(iter_active_skills(roots=[skills_root]))


def _lint(tmp_path: Path, slug: str, body: str, *, installed=(), extra_roots=()):
    """Lint one fixture skill against a unit corpus holding `installed`."""
    _write_skill(tmp_path / "skills", slug, body)
    corpus = tmp_path / "units"
    corpus.mkdir(parents=True, exist_ok=True)
    for name in installed:
        (corpus / name).write_text("[Unit]\n", encoding="utf-8")
    roots = [corpus, *extra_roots]
    return _lint_roots(_records(tmp_path / "skills"), roots)


def _lint_roots(records, roots):
    original = sl.default_unit_roots
    sl.default_unit_roots = lambda: list(roots)
    try:
        return sl.lint(skill_records=records)
    finally:
        sl.default_unit_roots = original


def _findings(result, slug: str) -> list[dict]:
    for item in result["dead_unit"]:
        if item["name"] == slug:
            return item["units"]
    return []


def _lines(skills_root: Path, slug: str) -> list[str]:
    return (skills_root / slug / "SKILL.md").read_text(encoding="utf-8").splitlines()


def _each_line_names_a_systemctl_call(hits, skills_root: Path, slug: str):
    """Every reported line number points at a line that really holds the call.

    The number is what a fixer is sent to, so it is asserted against the file rather
    than against a constant: a `line_no` of 0, or one counted from the match instead
    of the file, fails here even if the unit set is right.
    """
    lines = _lines(skills_root, slug)
    for hit in hits:
        assert 1 <= hit["line_no"] <= len(lines), hit
        line = lines[hit["line_no"] - 1]
        assert "systemctl" in line and hit["unit"] in line, hit


def test_a_command_naming_an_uninstalled_unit_is_reported(tmp_path):
    """Clause 4. `systemctl` against a unit in neither root is a finding, in both
    forms a command actually takes on this box.

    Fenced and inline: `voice-mode` prescribes a fenced
    `systemctl --user restart lloyd-voice-mode.service`, and
    `qmd-collection-management` prescribes `systemctl --user restart
    openclaw-gateway.service` inside an inline backtick span in a numbered step. A
    check that read only fenced blocks would report one of the two live offenders
    and read as working.

    `start` and `stop` both count. Half of `local-llm-gotchas`' four hits are
    `systemctl --user stop lloyd-vllm.service`; a verbs list that names only
    `start`/`restart` would catch the restart and miss the shutdown advice, so the
    second unit here is reached exclusively by `stop`.
    """
    body = (
        "Restart it:\n\n"
        "```\n"
        "systemctl --user start lloyd-gone-a.service\n"
        "systemctl restart lloyd-gone-b.timer\n"
        "```\n\n"
        "or from a step: `systemctl --user stop lloyd-gone-a.service` and re-check.\n"
    )
    result = _lint(tmp_path, "dead-unit", body)
    hits = _findings(result, "dead-unit")
    assert sorted(h["unit"] for h in hits) == [
        "lloyd-gone-a.service", "lloyd-gone-a.service", "lloyd-gone-b.timer",
    ], hits
    _each_line_names_a_systemctl_call(hits, tmp_path / "skills", "dead-unit")
    # The inline span is a second finding for the same unit, not a deduped one: the
    # fenced restart and the step's stop are two places the body tells someone to
    # touch a unit that is not there.
    assert len({h["line_no"] for h in hits}) == 3, hits


def test_the_category_appears_in_the_report_category_table(tmp_path):
    """Clause 4's second half. `tests/test_skill_lint_report_trust.py:56` pins
    `set(rows) == set(sl.CATEGORY_TRUST)` against a hand-built `_result()` that has no
    `dead_unit` key at all, so it proves the row's *membership* and nothing about the
    row this run produced. Here the result comes from linting the fixture skill, so
    the count, the one-line action and the trust cell are all graded together.

    `desc_missing` is counted off `result["description_missing"]["missing"]`, so the
    key is supplied: `lint()` always builds it, and dropping it here would raise
    rather than test anything.
    """
    body = "```\nsystemctl --user start lloyd-gone-a.service\n```\n"
    _write_skill(tmp_path / "skills", "dead-unit", body)
    corpus = tmp_path / "units"
    corpus.mkdir(parents=True, exist_ok=True)
    original = sl.default_unit_roots
    sl.default_unit_roots = lambda: [corpus]
    try:
        result = sl.lint(skill_records=_records(tmp_path / "skills"))
        report = sl.render_report(result)
    finally:
        sl.default_unit_roots = original

    # Split the way `tests/test_skill_lint_report_trust.py:46` does, because the row
    # carries its one-line meaning inside the category cell —
    # `| DEAD_SYSTEMD_UNIT (runs systemctl against a unit in neither unit root) | **1**
    # | action | trust |` — so a pattern anchored on the bare category name followed by
    # a pipe matches nothing and would read as absence.
    rows = [p for p in (line.split("|") for line in report.splitlines())
            if len(p) == 6 and p[1].strip().startswith(sl.DEAD_UNIT_CATEGORY)]
    assert len(rows) == 1, f"the category is not in the table:\n{report[:800]}"
    category, count, action, trust = (p.strip() for p in rows[0][1:5])
    assert category == f"{sl.DEAD_UNIT_CATEGORY} (runs systemctl against a unit in " \
                       f"neither unit root)", category
    assert count == "**1**", count
    assert "supervisorctl" in action, action
    assert trust == sl.trust_cell(sl.DEAD_UNIT_CATEGORY), trust

    # The section a fixer reads: it names the skill, the unit and the line. And the
    # report must not also claim the library is clean, which is the same sentence the
    # pre-#1308 build printed on 29 findings.
    assert f"## {sl.DEAD_UNIT_CATEGORY}" in report
    assert "`lloyd-gone-a.service`" in report and "`dead-unit`" in report
    assert "No findings in any category" not in report


def test_a_prose_retirement_mention_of_the_same_unit_stays_clean(tmp_path):
    """Clause 5, and the reason the category may stay ON.

    The rewritten `groundskeeper-survey/SKILL.md` names
    `lloyd-groundskeeper-survey.service` and `lloyd-groundskeeper-survey.timer` in
    prose, states they were removed in #1012, and says to expect them absent. Naming a
    dead unit while describing its removal is the correct thing for a retired skill to
    do; the same file with those names moved into a command becomes the bug again, and
    only the command can be executed. Both halves are asserted on one body, so the
    control cannot pass by the unit name being absent from the fixture.

    A bare `systemctl` with no unit is clean too — that is the form
    `stale-process-cleanup/SKILL.md:59-60` uses when it tells the agent NOT to use
    systemctl, and a check that fired on it would fire on the correct advice.
    """
    prose = (
        "**Retired 2026-09-23 (#1012).**\n\n"
        "The units `lloyd-gone-a.service` and `lloyd-gone-b.timer` were removed in\n"
        "#1012, deleted not disabled, so a status query for either reports that it\n"
        "could not be found and a timer listing comes back empty. Do not run\n"
        "systemctl at all; `supervisorctl` is the way.\n"
    )
    clean = _lint(tmp_path, "retired-skill", prose, installed=["lloyd-kept.service"])
    assert clean["dead_unit"] == [], clean["dead_unit"]

    prescribing = prose + "\n```\nsystemctl --user start lloyd-gone-a.service\n```\n"
    _write_skill(tmp_path / "skills", "retired-skill", prescribing)
    result = _lint_roots(_records(tmp_path / "skills"), [tmp_path / "units"])
    hits = _findings(result, "retired-skill")
    assert [h["unit"] for h in hits] == ["lloyd-gone-a.service"], hits
    _each_line_names_a_systemctl_call(hits, tmp_path / "skills", "retired-skill")


def test_a_command_that_does_not_run_systemctl_is_not_a_finding(tmp_path):
    """Clause 4's scope: the finding is a systemctl step, not an unfamiliar name.

    The control is the `systemctl` requirement itself, so the fixture names its units
    inside command contexts: `ls` inspects a path without claiming the unit is
    loadable, and `install` is the repair the report's own action column recommends
    ("restart the real supervisor (supervisorctl) or install the unit"). Firing on
    those would report the remedy alongside the fault, and a report that condemns its
    own advice gets ignored — which is how an advisory bucket dies.
    """
    body = (
        "Check, then install it:\n\n"
        "```\n"
        "ls -l ~/.config/systemd/user/lloyd-gone-c.service\n"
        "install -m 644 files/lloyd-gone-d.timer ~/.config/systemd/user/\n"
        "```\n\n"
        "The unit file is in `files/`; nothing is running yet.\n"
    )
    result = _lint(tmp_path, "not-a-finding", body)
    assert _findings(result, "not-a-finding") == [], _findings(result, "not-a-finding")


def test_a_unit_the_box_really_has_is_never_reported(tmp_path):
    """Clause 4's other side: the two roots are resolved, not merely mentioned.

    `lloyd-vault-backup.timer` is tracked under `agent-services/systemd/` and
    `agent-supervisord.service` is installed under `~/.config/systemd/user/`, so a
    command naming either is correct writing and must stay out of the report. This is
    also the node that fails if the roots are dropped and a corpus fixture list takes
    over — the check would then be naming units from a list rather than resolving
    them, which is the `KNOWN_ABSENT_SCRIPTS` shape again.
    """
    body = ("```\n"
            "systemctl --user restart lloyd-vault-backup.timer\n"
            "systemctl --user status agent-supervisord.service\n"
            "```\n")
    _write_skill(tmp_path / "skills", "live-unit", body)
    result = _lint_roots(_records(tmp_path / "skills"), sl.default_unit_roots())
    assert result["dead_unit"] == [], result["dead_unit"]
    # And the same body with one dead name added reports only the dead one, so the
    # clean result above comes from the roots resolving, not from the scan skipping.
    dead = body.replace("lloyd-vault-backup.timer", "lloyd-gone-e.timer")
    _write_skill(tmp_path / "skills", "live-unit", dead)
    result = _lint_roots(_records(tmp_path / "skills"), sl.default_unit_roots())
    assert [h["unit"] for i in result["dead_unit"] for h in i["units"]] == [
        "lloyd-gone-e.timer"], result["dead_unit"]


def test_an_installed_unit_and_a_template_instance_are_not_reported(tmp_path):
    """The other side of precision, on the rule rather than on the prose.

    `lloyd-guardian@2.service` is not a file anywhere: `agent-services/systemd/`
    holds `lloyd-guardian.service`, and the 25 `*|*@*` names in
    `~/.config/systemd/user/` (`getty@getty.service`,
    `dbus-org.freedesktop.resolve1.service`) are runtime instances, not definitions.
    Resolving an instance to its template is what keeps `systemctl --user restart
    lloyd-guardian@2.service` — a command that works on this box — out of the report;
    a check that matched names literally would open with a false positive on a unit
    that is installed, and a false positive is what gets a category switched off.
    """
    body = (
        "```\n"
        "systemctl --user restart lloyd-guardian@2.service\n"
        "systemctl --user status lloyd-kept.timer\n"
        "```\n"
    )
    result = _lint(tmp_path, "live-unit", body,
                   installed=["lloyd-guardian@.service", "lloyd-kept.timer"])
    assert result["dead_unit"] == [], result["dead_unit"]


def test_a_unit_in_a_repo_subdirectory_counts_as_tracked(tmp_path):
    """`agent-services/systemd/system/lloyd-data-snapshot-prune.timer` is two levels
    down, and `tests/test_automod_hardening.py:752` compares the same tree
    recursively, so the corpus walk has to recurse or that real unit reads as dead.
    """
    nested = tmp_path / "nested"
    (nested / "system").mkdir(parents=True)
    (nested / "system" / "lloyd-nested.timer").write_text("[Unit]\n", encoding="utf-8")
    body = "```\nsystemctl --user start lloyd-nested.timer\n```\n"
    _write_skill(tmp_path / "skills", "nested-unit", body)
    corpus = tmp_path / "units"
    corpus.mkdir(parents=True, exist_ok=True)
    result = _lint_roots(_records(tmp_path / "skills"), [corpus, nested])
    assert result["dead_unit"] == [], result["dead_unit"]


def test_the_live_skills_the_check_names_are_the_expected_three():
    """Both clause halves against the real corpus, and the anti-tuning pin.

    Read-only: `lint(skill_records=…)` walks records and never reaches
    `REPORT_PATH.write_text`, which only `main()` calls — unlike
    `scripts/skill_lint.py:1332`, a vault write, so this node commits nothing to
    `~/obsidian`.

    Three directions, each failing differently. `groundskeeper-survey` must be absent,
    which is the rewritten body holding clauses 1 and 2 against the check that named it
    before the rewrite. The exact set must equal `EXPECTED_DEAD_UNIT_SKILLS`, the
    anti-tuning pin: an empty `dead_unit` list reads as a clean library and is just as
    well produced by a check narrowed until it names nothing, so carrying the measured
    set is what says the check did not stay that way. And `ABSENT_UNIT` must be absent
    — the positive control on the assertion itself, proving the scan resolves files
    rather than matching a list.

    That a skill is *named* here says nothing about whether it is advertised:
    `tests/test_archived_skill_artifacts.py::test_bench_007_grades_an_advertised_skill`
    pins `status: active` and advertised-ness for the retired skill, and it is the
    reason the fix is a body rewrite rather than an archive.
    """
    from agent_mcp.skills import iter_active_skills

    result = sl.lint(skill_records=list(iter_active_skills()))
    named = {item["name"] for item in result["dead_unit"]}
    assert named == EXPECTED_DEAD_UNIT_SKILLS, named
    assert "groundskeeper-survey" not in named
    hits = [h for i in result["dead_unit"] for h in i["units"]]
    assert hits, (
        "the live corpus yielded nothing at all, so the equality above could be one "
        "empty set matching another — a check that never fires gets a green run")
    assert ABSENT_UNIT not in {h["unit"] for h in hits}


# ── the skill this category was filed against ──────────────────────────────────
#
# Clauses 1 and 2 are about one file in the vault, so they are pinned here rather
# than in `tests/test_archived_skill_artifacts.py`: that file owns whether the skill
# is *advertised* (clause 3, `:215`), and the two questions need different markers —
# the retention node reads the derived skills index, these read the body. Unmarked, so
# the gate runs them: `test_the_live_skill_library_has_no_injection_findings`
# (`tests/test_skill_lint_gates.py:278`) is the precedent for a live-library node that
# the gate executes, and a clause pinned on a marker the gate deselects is pinned on
# no hard rung at all.

GROUNDSKEEPER_SKILL = Path.home() / "obsidian" / "skills" / "groundskeeper-survey" / "SKILL.md"

#: The two units `201901b4` (#1012) deleted. Naming them in prose is correct; naming
#: one inside a command is the bug — which is exactly what `DEAD_SYSTEMD_UNIT` decides
#: between, so these two clauses and that category share one mechanism instead of
#: restating each other.
REMOVED_UNITS = ("lloyd-groundskeeper-survey.service", "lloyd-groundskeeper-survey.timer")

#: A schedule sentence is admissible only in the past tense. Each marker is a form the
#: rewritten body really uses — `used to run`, `were removed in #1012`, `was retired
#: on 2026-09-23`, `is off and the queue is no longer rebuilt`, and the 2026-09-22
#: report that "the nightly timer ran fine" — and each is judged per sentence, because
#: the paragraph that failed is one sentence long.
PAST_MARKERS = ("used to", "were removed", "was retired", "retired on", "no longer",
                " ran ")

#: Present-tense forms the body must never take again, checked beside the sentence
#: rule rather than instead of it: the first catches a schedule claim that also
#: mentions the retirement, this one catches a reworded claim that dropped the word
#: "nightly" ("the timer is scheduled for 02:30").
PRESENT_SCHEDULE_FORMS = (r"\bruns nightly\b", r"\bruns at 02:30\b",
                          r"\bis run nightly\b", r"\bis scheduled[^.]{0,30}02:30\b",
                          r"\bthe timer runs\b", r"\bfires nightly\b")


def _command_texts(body: str) -> list[str]:
    """The body's command contexts — the same surfaces the lint category reads.

    Reusing the production reader rather than writing a second fence parser is the
    point: if `command_contexts` ever stopped tracking a form, a body hiding a
    systemctl call in that form would read as clean to the check and to this file.
    """
    return [text for _, text in sl.command_contexts(body)]


def _flat(body: str) -> str:
    """Prose with hard wraps collapsed. The vault file wraps near 80 columns, so
    "reach for the" and "timer status" land on different lines and a raw substring
    match finds nothing — the same reason `tests/test_autonomy_jobs_doc_claims.py:108`
    flattens before it matches a sentence."""
    return re.sub(r"\s+", " ", body)


def _schedule_claims(body: str) -> list[str]:
    """Sentences in `body` that assert a live schedule. Empty is the fixed state."""
    flat = _flat(body)
    claims = [s for s in re.split(r"(?<=[.!?]) +", flat)
              if "nightly" in s.lower()
              and not any(m in s.lower() for m in PAST_MARKERS)]
    claims += [f"present form: {f}" for f in PRESENT_SCHEDULE_FORMS
               if re.search(f, flat, re.I)]
    return claims


def _runnable_unit_commands(body: str) -> list[str]:
    """Command contexts in `body` that run systemctl against one of the removed units."""
    return [text for text in _command_texts(body)
            if "systemctl" in text and any(u in text for u in REMOVED_UNITS)]


#: The two paragraphs as they stood when the item was filed, verbatim from
#: `c3bbe7c^:skills/groundskeeper-survey/SKILL.md` (the vault commit that rewrote
#: them). Both clause predicates are run over this text and must FIRE on it: a clause
#: pinned only on the current file cannot tell a fixed body from a predicate that
#: matches nothing at all.
PRE_FIX_SKILL_BODY = """**Retired 2026-09-23 (#1012).** The nightly survey timer is off and the queue is
no longer rebuilt; autonomy #36 is archived.

**You do not run the survey.** The scan walks the whole vault plus a ~63,000-file
facts tree and takes about 40 minutes, which is far longer than any tool call
here can wait. It runs nightly at 02:30 under the systemd user timer
`lloyd-groundskeeper-survey.timer`. Attempting to run it inline is what made
this task fail ~70 times a week.

3. The age is expected since the retirement. Only reach for the timer status if
   asked.
```
Bash("systemctl --user status lloyd-groundskeeper-survey.service --no-pager -n 20; systemctl --user list-timers lloyd-groundskeeper-survey.timer --no-pager")
```
"""


def test_the_retired_skill_no_longer_claims_a_schedule_that_runs():
    """Clause 1. The paragraph under the banner asserted a live nightly schedule.

    #1012 deleted both units on 2026-09-23, so that sentence was not merely stale
    prose: told the queue is rebuilt nightly at 02:30, a runner looks for the timer
    that rebuilds it, reads `could not be found`, and files an incident. The retired
    state is now the only thing the paragraph asserts, checked positively as well as
    negatively — the two units are still named with their removal, so the fix cannot
    pass by deleting the contradiction whole.
    """
    body = GROUNDSKEEPER_SKILL.read_text(encoding="utf-8")

    assert _schedule_claims(body) == [], _schedule_claims(body)
    assert "nightly" in _flat(body).lower(), (
        "the body never mentions the nightly survey, so the clause above is vacuous")
    assert "#1012" in body, "the body never names what retired the survey"
    assert all(unit in body for unit in REMOVED_UNITS), (
        "the body stopped naming the two units #1012 deleted, so a reader cannot tell "
        "which schedule is gone — a retirement has to be specific")


def test_no_command_in_the_retired_skill_names_a_unit_the_box_does_not_have():
    """Clause 2. The Step 3 block ran `systemctl` against both deleted units.

    Two halves, both asserted. No command context — fenced or inline — may run
    systemctl against either unit (prose may, and does); and Step 3 has to say the
    units were removed and that their absence is the expected answer, or an agent
    that asks anyway reads `could not be found` as a fresh incident.
    """
    body = GROUNDSKEEPER_SKILL.read_text(encoding="utf-8")
    assert _runnable_unit_commands(body) == [], _runnable_unit_commands(body)
    systemctl_calls = [t for t in _command_texts(body) if "systemctl" in t]
    assert systemctl_calls == [], f"a systemctl command is still runnable: {systemctl_calls}"

    flat = _flat(body)
    anchor = "only reach for the timer status if asked"
    assert anchor in flat, "Step 3's timer clause moved; the assertions below prove nothing"
    step3 = flat[flat.index(anchor):]
    assert "removed in #1012" in step3, step3[:300]
    assert "deleted, not disabled" in step3, (
        "Step 3 must say the units were deleted, not disabled — `disabled` would leave "
        "a runner looking for a unit to enable")
    assert "could not be found" in step3, (
        "Step 3 never says what querying either unit actually answers, so the absence "
        "reads as a failure rather than as the retirement")
    assert "expect that absence" in step3.lower(), (
        "Step 3 never tells the runner the absence is the expected answer, which is the "
        "half that stops a `could not be found` being reported as an incident")


def test_the_two_skill_predicates_fire_on_the_text_the_item_was_filed_against():
    """Positive control on both clause predicates, against the pre-#1580 bytes.

    The vault file is one fixed input, and a negative assertion on one fixed input is
    the shape that passes forever whether or not it can fail. Running the same
    predicates over the text of `c3bbe7c^` — the commit before the rewrite — is what
    shows they are looking for something: the schedule predicate names the "It runs
    nightly at 02:30" sentence, and the command predicate names the Bash block.
    """
    claims = _schedule_claims(PRE_FIX_SKILL_BODY)
    assert any("It runs nightly at 02:30" in c for c in claims), claims
    commands = _runnable_unit_commands(PRE_FIX_SKILL_BODY)
    assert len(commands) == 1 and "lloyd-groundskeeper-survey.service" in commands[0], commands
