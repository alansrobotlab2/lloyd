"""The corpus gate for a command that re-opens its own string (#4419).

`scripts/skill_lint.py` is advisory by construction, and being advisory is why six
categories produced no enforcement for a year. So the shape #4270 fixed is guarded
twice, the way `INJECTION_PATTERN` (#677), `MISSING_SCRIPT` (#874) and
`DEAD_SYSTEMD_UNIT` (#1580) are: the lint finds it, THIS file fails the build.

The defect itself is not a wrong filter. `queue-health-check` carried a `jq` format
string that defaulted a field with `// "n/a"` written INSIDE its own `\\(...)`
interpolation, and a jq interpolation is evaluated inside the string it is typed
into — so the quote before `n/a` closes that string and the program never parses.
What the reader of that skill saw was the endpoint failing, and a defect was filed
against the endpoint (#4261). #4270 rewrote the command on the vault's main
(bind the default once, `("n/a") as $na | … "\\(.a // $na)"`) and that fix is
verified on disk; nothing in `~/lloyd` kept it, so one hand-retyped filter in one
published skill re-opens the shape and re-authors the phantom bug report.

Why the predicate is per-interpolation rather than per-line, and this file is the
place that claim gets spent. The wide version — a `"` anywhere after a `\\(` — is
wrong on the file the rule was filed against: a jq format line is a quoted string
that CONTAINS interpolations, and the quotes of that string are legitimate, so the
naive rule fires 4 lines in each of `queue-health-check`'s two files.
`test_the_naive_per_line_rule_would_fire_on_the_fixed_skill` measures those 4 so the
narrowness is not folklore. The other failure direction is narrower still and just
as real: the `// "` spelling a hand `awk` matched is ONE surface form, so it misses
`// 'x'` and `// (…)` — `test_a_nested_default_expression_is_reported` spends that.

Scope this check does NOT claim, and the tests below keep honest: the scan is per
command line, so an interpolation that closes on a later line is unseen, and a
`\\(` with no balancing paren on its line is a regex capture group rather than a
broken literal (`test_an_unbalanced_opening_is_not_reported`). A green suite here
means no published skill re-opens a string inside its own interpolation, not that
every command in the library parses.

Seam: like `tests/test_skill_lint_gates.py`, this is a surface that reads the whole
live corpus through `agent_mcp.skills.iter_active_skills` with `scripts/skill_lint.py`
loaded by path — two processes that never call each other at runtime (nightly task
#70 shells out to the script; pytest imports the file with `spec_from_file_location`),
so the code graph sees no edge and a rename here looks local to it and is not.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_skill_lint(tag: str):
    """Load `scripts/skill_lint.py` by path — it is not an importable module.

    The nightly invocation is `python …/scripts/skill_lint.py`, which puts
    `scripts/` on `sys.path` and not the repo root; same reasoning as
    `tests/test_skill_lint_gates.py` and `tests/test_skill_dead_units.py`.
    """
    spec = importlib.util.spec_from_file_location(
        tag, ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LINT = _load_skill_lint("skill_lint_under_command_shape_gate")

SKILLS_DIR = Path.home() / "obsidian" / "skills"

#: The two files #4270 rewrote. `queue-health-check` is a spilled skill, and the
#: broken filter existed in BOTH — the index file and the step spilled beside it —
#: which is why the scan reads the folder and why both are named here.
QHC_FILES = ("skills/queue-health-check/SKILL.md",
             "skills/queue-health-check/step3b-wait-history.md")

#: The #4270 shape as one command line: `| "\(.a // "n/a")"`. The default operand's
#: quotes are inside the interpolation, so they close the enclosing jq string.
OFFENDING_CMD = r"""  | jq -r '.sources[] | "\(.a // "n/a")"'"""

#: #4270's actual repair, in the same shape: the default is bound above the filter
#: and the interpolation names a variable, so it carries no quote of its own.
FIXED_CMD = r"""  | jq -r '("n/a") as $na | ("none in the window") as $nw | .sources[] \
           | "\(.name): rate \(.health.fail_rate // $na) · last failure \(.health.last_failure // $nw)"'"""

#: A `grep` BRE capture group: a `\(` that is not an interpolation at all. The
#: `"` is inside it and there is no closing paren on the line, which is the shape
#: the per-line rule would fire on and this one cannot see (correctly).
BRE_CMD = r"""  | grep -o '"\(.\{60\}' access.log"""


def _fenced(command: str) -> str:
    """`command` inside a ```bash fence, which is what makes it a command context.

    `command_contexts` reads fenced lines and inline backtick spans and NOTHING
    else, so a bare command line handed to the predicate as prose is correctly not
    a finding — every raw-predicate fixture has to be fenced or it is testing the
    walker's absence of a match rather than the rule.
    """
    return ("```bash\n"
            'curl -s "localhost:8080/api/workers/health?days=7" \\\n'
            f"{command}\n"
            "```\n")


_FRONT = """---
name: {name}
description: Use this skill to read the worker queue health endpoint.
tags: [demo]
status: active
---
"""


def _skill(root: Path, name: str, command: str, *, sibling: bool = False) -> Path:
    """One published-shaped skill in a temp skills root, `command` inside a fence.

    Real front matter, because `lint()` skips the body checks for a skill whose
    front matter does not parse — a fixture that is DEAD would pass every "must be
    clean" assertion for the wrong reason. With `sibling=True` the fence goes into
    the spilled step file and SKILL.md carries no command at all, so a finding can
    only come from reading the folder.
    """
    slug = root / name
    slug.mkdir(parents=True, exist_ok=True)
    if sibling:
        (slug / "SKILL.md").write_text(
            _FRONT.format(name=name)
            + f"# SKILL: {name}\n\n## Step 1 — ask the endpoint\n\n"
              "See `step-detail.md`.\n", encoding="utf-8")
        (slug / "step-detail.md").write_text(
            f"# Step 1 — ask the endpoint\n\n{_fenced(command)}", encoding="utf-8")
    else:
        (slug / "SKILL.md").write_text(
            _FRONT.format(name=name)
            + f"# SKILL: {name}\n\n## Step 1 — ask the endpoint\n\n"
              + _fenced(command), encoding="utf-8")
    return slug


def _records(*roots: Path):
    from agent_mcp.skills import iter_active_skills

    recs = []
    for root in roots:
        recs.extend(iter_active_skills(roots=[root]))
    return recs


def _corpus_findings(records) -> list[dict]:
    """The gate's own predicate: the findings the library must not carry.

    One function so the live assertion and the throwaway assertion cannot disagree
    about what they measured, and so the temp-root test below fails the SAME
    assertion the corpus test passes.
    """
    return LINT.lint(skill_records=list(records))["bad_interpolation"]


@pytest.fixture(scope="module")
def live_records():
    """The corpus `iter_active_skills` owns — the population the gate is over.

    This is the population the clause measures, and it is NOT the same size as the
    hand `ls */*.md` count in #4419's triage body: the loader excludes archived and
    quarantined skill directories. Whether those belong in the gate is an open ruling
    on #4419, so nothing here asserts a file count — the population floor is in
    SKILLS, which is what `lint()` walks.
    """
    from agent_mcp.skills import iter_active_skills

    return list(iter_active_skills())


@pytest.fixture(scope="module")
def live_scan(live_records) -> dict:
    """One lint pass over the live corpus, shared by every test in this file.

    `lint()` runs `find_duplicates`, quadratic in the number of live skills, so
    re-running it per test would multiply the most expensive call in the file.
    """
    return LINT.lint(skill_records=live_records)


# ── clause 1: the shape fires, once, and names the skill ─────────────────────

def test_the_reopened_literal_in_a_fence_is_reported(tmp_path):
    """`| "\\(.a // "n/a")"` inside a ``` fence is exactly one finding, and it
    names the skill that published it.

    Both the raw predicate and the full `lint()` loop are asserted, because the
    clause is about the report's row: a check that fires on a string but is never
    called from `lint()` pins nothing at all.
    """
    name = _skill(tmp_path, "shape-offender", OFFENDING_CMD).name
    hits = LINT.check_bad_interpolations(_fenced(OFFENDING_CMD))
    assert len(hits) == 1, hits
    assert hits[0]["interpolation"] == r'\(.a // "n/a")', hits
    assert hits[0]["line"] == OFFENDING_CMD.strip(), hits

    findings = _corpus_findings(_records(tmp_path))
    assert [f["name"] for f in findings] == [name], findings
    assert len(findings[0]["interpolations"]) == 1, findings
    assert findings[0]["path"].endswith("shape-offender/SKILL.md"), findings


def test_a_spilled_sibling_carrying_the_shape_is_reported(tmp_path):
    """The shape has to be caught in the step file too, not only in SKILL.md.

    #4270's filter lived in `SKILL.md` AND in the `step3b-wait-history.md` spilled
    beside it — the agent reads the sibling, so a scan of the index file alone
    would call that skill clean while the broken command is one `Read` away.
    """
    name = _skill(tmp_path, "shape-spill", OFFENDING_CMD, sibling=True).name
    findings = _corpus_findings(_records(tmp_path))
    assert [f["name"] for f in findings] == [name], findings
    assert LINT.skill_folder_text(
        tmp_path / name,
        (tmp_path / name / "SKILL.md").read_text(encoding="utf-8")
    ).count(r'\(.a // "n/a")') == 1


def test_a_nested_default_expression_is_reported():
    """Balance, not the first `)`, is what makes the rule catch the defect at all.

    `"\\((.a + .b) // "x")"` is one interpolation whose own body re-opens the
    string. Stopping at the first `)` would return the body `.a`, find no quote, and
    report a broken command as clean — the false-negative half of the same mistake
    the per-line rule makes in the other direction.
    """
    cmd = r"""  | jq -r '.a[] | "\((.x + .y) // "n/a")"'"""
    hits = LINT.check_bad_interpolations(_fenced(cmd))
    assert len(hits) == 1, hits
    assert hits[0]["interpolation"] == r'\((.x + .y) // "n/a")', hits


# ── clause 2: the fix and the legitimate format lines stay clean ─────────────

def test_the_bound_var_fix_is_clean():
    """#4270's repair must not be a finding — that is what makes it the repair.

    If this goes red, the bucket cannot tell the fix from the defect, and the
    library's only correct example of an interpolated `jq` format string would be
    unshippable. The fixture is checked against the live file too, so the "fixed
    form" here is the form actually on disk and not an invented one.
    """
    assert LINT.check_bad_interpolations(FIXED_CMD + "\n") == []
    live = (SKILLS_DIR / "queue-health-check" / "SKILL.md").read_text(
        encoding="utf-8", errors="replace")
    assert '("n/a") as $na' in live, (
        "the bound-variable fix #4270 landed must still be the published form")


def test_queue_health_checks_own_format_lines_are_clean():
    """The four interpolated format lines per file in #4270's skill: zero findings.

    These are the lines the wide rule would fire on, so this is the noise-floor
    assertion in concrete form: 4 `\\(`-bearing command contexts in each of the two
    files, every one of them legitimate because their quotes sit OUTSIDE the
    interpolation spans.
    """
    for rel in QHC_FILES:
        text = (SKILLS_DIR.parent / rel).read_text(encoding="utf-8", errors="replace")
        ctxs = [c for _, c in LINT.command_contexts(text) if LINT._INTERP_OPEN in c]
        assert len(ctxs) == 4, f"{rel}: {len(ctxs)} interpolation-bearing lines"
        assert all('"' in c for c in ctxs), f"{rel}: expected quoted format strings"
        assert LINT.check_bad_interpolations(text) == [], rel


def test_the_naive_per_line_rule_would_fire_on_the_fixed_skill():
    """Why the bucket is per-interpolation: measured, not asserted in a comment.

    `\\(.*"` over the same command contexts matches 4 lines in each of
    #4270's files — the rule #4419 explicitly rejects would call the FIXED skill
    broken, and a bucket wrong on the file that motivated it is a bucket someone
    turns off. This test exists so the next person who reaches for a one-line regex
    finds the measurement already here.
    """
    naive = re.compile(r"\\\(.*\"")
    for rel in QHC_FILES:
        text = (SKILLS_DIR.parent / rel).read_text(encoding="utf-8", errors="replace")
        matches = [c for _, c in LINT.command_contexts(text) if naive.search(c)]
        assert len(matches) == 4, f"{rel}: {len(matches)} naive matches"


# ── clause 3: the noise floor over the corpus is measured ────────────────────

def test_the_live_corpus_carries_no_reopened_literal(live_records, live_scan):
    """Zero findings over the corpus `iter_active_skills` yields, measured now.

    Three assertions, and the third is the one that makes the first mean anything:
    the population must be a library rather than an empty walk, and the corpus must
    actually contain interpolations for the scan to have passed over. On
    2026-11-25 the corpus is 208 skills carrying 53 `\\(` openings inside command
    contexts, all balanced, none re-opening a string; the floors below are set well
    under those figures so the node says 'the scan had work to do' without rotting
    the first time a skill gains a filter.
    """
    assert len(live_records) > 150, (
        f"only {len(live_records)} active skills — the gate is vacuous, not green")
    findings = live_scan["bad_interpolation"]
    assert findings == [], (
        f"a published skill re-opens a string inside its own interpolation: "
        f"{findings}. Bind the default above the filter as a jq variable.")

    openings = 0
    for record in live_records:
        for md in sorted(record.directory.glob("*.md")):
            text = md.read_text(encoding="utf-8", errors="replace")
            for _, ctx in LINT.command_contexts(text):
                at = ctx.find(LINT._INTERP_OPEN)
                while at != -1:
                    openings += 1
                    span = LINT._interp_span(ctx, at + 1)
                    at = ctx.find(LINT._INTERP_OPEN,
                                  (span[1] + 1) if span else at + 2)
    assert openings >= 40, (
        f"only {openings} interpolations reached across the corpus: the scan is not "
        "seeing command contexts, so a zero here proves nothing")


# ── clause 4: the corpus assertion can fail ──────────────────────────────────

def test_the_corpus_assertion_fails_on_a_temp_root_holding_one_offender(tmp_path):
    """The same predicate, fed a throwaway corpus with one offender, must bite.

    Without this, `test_the_live_corpus_carries_no_reopened_literal` could be green
    because the check stopped matching: the check that proves a zero is a measurement
    is the same check producing a one on a planted line. It uses
    `iter_active_skills(roots=[tmp_path])` and the shared `_corpus_findings`
    predicate, so it exercises the walk and the assertion, not a private copy.
    """
    name = _skill(tmp_path, "shape-control", OFFENDING_CMD).name
    records = _records(tmp_path)
    assert [r.name for r in records] == [name], (
        f"the temp root must be walkable as a skills dir: {[r.name for r in records]}")
    findings = _corpus_findings(records)
    assert [f["name"] for f in findings] == [name], findings


def test_the_finding_reaches_the_report_with_its_line(tmp_path):
    """A fired count is only actionable if the report prints WHICH line.

    #4270's whole cost was a reader who could not tell a broken quote from a dead
    endpoint, so the repair has to be identifiable from the row. Rendered off the
    temp corpus through the real `render_report`, not a hand-built payload.
    """
    _skill(tmp_path, "shape-reported", OFFENDING_CMD)
    result = LINT.lint(skill_records=_records(tmp_path))
    report = LINT.render_report(result)
    assert f"## {LINT.BAD_INTERPOLATION_CATEGORY} — 1 skill(s)" in report, report
    assert "| `shape-reported` | `\\(.a // \"n/a\")`" in report.replace(
        "\\|", "|"), report
    assert "## ✅ Clean" not in report, (
        "a fired count must not leave the report saying all skills pass")


# ── the precision price: what this check does NOT claim ──────────────────────

def test_an_unbalanced_opening_is_not_reported():
    """A `\\(` that never closes on its line is a regex group, not a claim.

    `grep -o '"\\(…'` is a BRE capture group with a quote inside it and no `)` on
    the line; flagging it would put the lint's authority on the shape against the
    hundred `sed`/`grep` patterns in the library. This node is the scope price
    written as a test: silence here is deliberate, not a matcher that broke.
    """
    assert LINT.check_bad_interpolations(_fenced(BRE_CMD)) == []


def test_prose_without_a_code_span_is_not_a_command_context():
    """Narrating the shape is not publishing it; naming it in code is.

    `command_contexts` reads fenced lines AND inline backtick spans, which is the
    width #4419's clause asks for ("a fenced/inline command"), and the width has a
    price this node records rather than hides: a skill that documents this defect by
    putting `\\(.name // "n/a")` in backticks is publishing a command context that
    carries it, and it fires. Dead-unit precision rests on the same width
    (`qmd-collection-management:92` lives inside one span), so narrowing to fences
    would make the two command-context categories disagree about what a command is —
    #1294's one-walker rule. What stays clean is prose: the sentence in backticks
    fires, the sentence without them does not.
    """
    prose = ('The broken form defaulted a field inside its own interpolation, '
             'which closes the jq string early.\n')
    assert LINT.check_bad_interpolations(prose) == []

    in_backticks = 'The broken form was `\\(.name // "n/a")`.\n'
    hits = LINT.check_bad_interpolations(in_backticks)
    assert [h["interpolation"] for h in hits] == [r'\(.name // "n/a")'], hits

    fenced = "```bash\n" + OFFENDING_CMD + "\n```\n"
    assert len(LINT.check_bad_interpolations(fenced)) == 1
