"""Tests for `scripts/autonomy/validate_tasks.py`, the autonomy-task linter.

The linter is a CLI and nothing imports it: ripgrep over the checkout finds
`validate_tasks` only in prose (SETUP.md, architecture/autonomy*.md), no
`.github/workflows/`, no other test. So every test here runs the script as a
subprocess — the exit code is part of the contract (0 clean, 1 unparseable, 2
structural problems under `--strict`) and cannot be pinned from inside the
process.

What #811 added is the class these tests exist for: a field's value must
*resolve*, not merely be present. Three values pass the old linter and fail at
dispatch — a `depends_on` whose upstream is parked, a `model:` no engine serves,
and a `skill_name` with no SKILL.md behind it.
"""
from __future__ import annotations

import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "autonomy" / "validate_tasks.py"

# Two keys, one of which declares an alias that is not its own key — so a test
# can tell "keys are accepted" apart from "aliases are accepted" apart from
# "anything goes". Mirrors the real config.yaml, which has only primary and
# secondary and is what `model: eco` fails against.
CONFIG = """models:
  primary:
    alias: primary
  secondary:
    alias: fast
"""


class Linter:
    """A fixture dir set with a task file per `add()` call."""

    def __init__(self, tmp_path: Path, *, config: str | None = CONFIG):
        self.autonomy_dir = tmp_path / "autonomy"
        self.autonomy_dir.mkdir()
        self.skills_dir = tmp_path / "skills"
        (self.skills_dir / "heartbeat").mkdir(parents=True)
        (self.skills_dir / "heartbeat" / "SKILL.md").write_text("# heartbeat\n")
        self.config_path = tmp_path / "config.yaml"
        self.config_path.write_text(config if config is not None else "models: {}\n")

    def add(self, filename: str, **fields) -> None:
        body = "\n".join(f"{k}: {v}" for k, v in fields.items())
        (self.autonomy_dir / filename).write_text(f"---\n{body}\n---\n\nbody\n")

    def run(self, *args: str) -> tuple[int, str]:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--autonomy-dir", str(self.autonomy_dir),
             "--skills-dir", str(self.skills_dir),
             "--config", str(self.config_path), *args],
            capture_output=True, text=True, timeout=60,
        )
        return proc.returncode, proc.stdout + proc.stderr


# ── Clause 1/2: a depends_on whose upstream parses but is parked ─────────────

@pytest.mark.parametrize("upstream_status", ["paused", "draft", "failed", "done"])
def test_a_dependency_on_a_parked_upstream_names_file_upstream_and_status(tmp_path, upstream_status):
    """`2-y.md: depends_on 1 but #1 is status=paused` — all three facts in one line.

    #870 made dispatch *see* a paused upstream, so the linter is the only layer
    that says a chain is wired to something that will never run.
    """
    lint = Linter(tmp_path)
    lint.add("1-x.md", id=1, name="x", status=upstream_status, frequency="daily",
             skill_name="heartbeat")
    lint.add("2-y.md", id=2, name="y", status="up_next", frequency="daily",
             depends_on=1, skill_name="heartbeat")

    code, out = lint.run()

    assert code == 0, "a structural warning is only a failure under --strict"
    assert f"2-y.md: depends_on 1 but #1 is status={upstream_status}" in out


def test_a_parked_upstream_fails_the_strict_rung(tmp_path):
    """The premise's own check: paused upstream, strict, must not exit clean."""
    lint = Linter(tmp_path)
    lint.add("1-x.md", id=1, name="x", status="paused", frequency="daily",
             skill_name="heartbeat")
    lint.add("2-y.md", id=2, name="y", status="up_next", frequency="daily",
             depends_on=1, skill_name="heartbeat")

    assert lint.run("--strict")[0] == 2


@pytest.mark.parametrize("upstream_status", ["up_next", "in_progress"])
def test_a_healthy_chain_exits_clean_with_no_dependency_warning(tmp_path, upstream_status):
    """The check must not fire on a chain that can run, or it is just noise."""
    lint = Linter(tmp_path)
    lint.add("1-x.md", id=1, name="x", status=upstream_status, frequency="daily",
             skill_name="heartbeat")
    lint.add("2-y.md", id=2, name="y", status="up_next", frequency="daily",
             depends_on=1, skill_name="heartbeat")

    plain_code, plain_out = lint.run()
    strict_code, strict_out = lint.run("--strict")

    assert "depends_on" not in plain_out
    assert plain_code == 0
    assert strict_code == 0, strict_out
    assert "structural warning" not in strict_out


def test_an_upstream_with_no_status_at_all_is_parked(tmp_path):
    """A missing status is not `up_next`: `_is_task_due` reports 'no status' and skips."""
    lint = Linter(tmp_path)
    lint.add("1-x.md", id=1, name="x", frequency="daily", skill_name="heartbeat")
    lint.add("2-y.md", id=2, name="y", status="up_next", frequency="daily",
             depends_on=1, skill_name="heartbeat")

    assert "2-y.md: depends_on 1 but #1 is status=unset" in lint.run()[1]


# ── Clause 3: the pre-existing not-found warning stays its own class ─────────

def test_an_absent_upstream_keeps_the_not_found_warning(tmp_path):
    """An id with no file stays a *membership* warning, worded as it always was."""
    lint = Linter(tmp_path)
    lint.add("4-w.md", id=4, name="w", status="up_next", frequency="daily",
             depends_on=99, skill_name="heartbeat")

    out = lint.run()[1]

    assert "4-w.md: depends_on '99' not found among parseable tasks" in out
    assert "is status=" not in out, "the two classes must not share wording"


def test_an_absent_upstream_and_a_parked_one_are_reported_separately(tmp_path):
    """One dir, both defects: two lines, each naming its own cause."""
    lint = Linter(tmp_path)
    lint.add("1-x.md", id=1, name="x", status="paused", frequency="daily",
             skill_name="heartbeat")
    lint.add("2-y.md", id=2, name="y", status="up_next", frequency="daily",
             depends_on=1, skill_name="heartbeat")
    lint.add("4-w.md", id=4, name="w", status="up_next", frequency="daily",
             depends_on=99, skill_name="heartbeat")

    out = lint.run()[1]

    assert "2-y.md: depends_on 1 but #1 is status=paused" in out
    assert "4-w.md: depends_on '99' not found among parseable tasks" in out
    assert "2 structural warning" in out


# ── Clause 4: model: against the models: keys and aliases ───────────────────

def test_an_unknown_model_names_the_file_and_the_value(tmp_path):
    """`model: eco` reaches the primary's endpoint under a name it does not serve."""
    lint = Linter(tmp_path)
    lint.add("85-e.md", id=85, name="e", status="draft", frequency="daily",
             skill_name="heartbeat", model="eco")

    code, out = lint.run()

    assert code == 0
    assert "85-e.md: model 'eco' is not a models: key or alias" in out
    assert "primary, secondary" in out, "the line has to say what IS valid"
    assert lint.run("--strict")[0] == 2


@pytest.mark.parametrize("model", ["primary", "secondary", "fast"])
def test_a_declared_key_or_alias_is_accepted(tmp_path, model):
    """`_get_model_env` matches a key then any alias; both must pass the linter.

    The config spells both slot switches ON. `CONFIG` alone carries no
    `secondary_enabled` key, and `app.llm_slots.is_enabled` answers False for a flag
    that is absent — the same rule that makes a missing flag a real retirement on the
    box — so under the bare fixture the #2632 arm would report `model: secondary` as a
    switched-off slot and this node would fail for a slot verdict it never set out to
    make. Its property is name acceptance, and a slot that is booted is the config
    where name acceptance is the only question left.
    """
    lint = Linter(tmp_path, config=_slot_config(True, True))
    lint.add("7-m.md", id=7, name="m", status="up_next", frequency="daily",
             skill_name="heartbeat", model=model)

    code, out = lint.run("--strict")

    assert "is not a models: key or alias" not in out
    assert "unchecked" not in out
    assert code == 0, out


def test_an_unreadable_config_warns_instead_of_passing(tmp_path):
    """No models: mapping means no verdict — which must never print as a clean pass."""
    lint = Linter(tmp_path, config="worker:\n  enabled: true\n")
    lint.add("7-m.md", id=7, name="m", status="up_next", frequency="daily",
             skill_name="heartbeat", model="eco")

    code, out = lint.run()

    assert "model 'eco' unchecked" in out
    assert lint.run("--strict")[0] == 2, "an unchecked value is not a passing value"


# ── Clause 5: skill_name / skill_path must resolve to a SKILL.md ─────────────

def test_an_unresolvable_skill_name_names_the_file_and_the_skill(tmp_path):
    """Dispatch fails this with 'Skill not found'; the linter sees it first."""
    lint = Linter(tmp_path)
    lint.add("9-s.md", id=9, name="s", status="up_next", frequency="daily",
             skill_name="totally-not-a-real-skill")

    code, out = lint.run()

    assert code == 0
    assert "9-s.md: skill_name 'totally-not-a-real-skill' resolves to no SKILL.md" in out
    assert str(lint.skills_dir) in out, "the line names the dir it looked in"


def test_a_skill_name_resolving_in_the_pointed_dir_is_accepted(tmp_path):
    """--skills-dir is what makes the check hermetic; a slug in it is clean."""
    lint = Linter(tmp_path)
    (lint.skills_dir / "nightly-thing").mkdir()
    (lint.skills_dir / "nightly-thing" / "SKILL.md").write_text("# nightly\n")
    lint.add("9-s.md", id=9, name="s", status="up_next", frequency="daily",
             skill_name="nightly-thing")

    code, out = lint.run("--strict")

    assert "resolves to no SKILL.md" not in out
    assert code == 0, out


def test_a_path_valued_skill_reference_must_be_a_file_that_exists(tmp_path):
    """`_load_skill_content` takes a path branch and never falls back to a slug."""
    lint = Linter(tmp_path)
    lint.add("9-s.md", id=9, name="s", status="up_next", frequency="daily",
             skill_name="~/nowhere/secondary_routing_eval.py")

    assert "9-s.md: skill_name '~/nowhere/secondary_routing_eval.py' " \
           "is not a file that exists" in lint.run()[1]


def test_an_unresolvable_skill_path_warns_too(tmp_path):
    """skill_path is the other key that carries a skill reference (#827's shape)."""
    lint = Linter(tmp_path)
    lint.add("9-s.md", id=9, name="s", status="up_next", frequency="daily",
             skill_path="gone-skill")

    assert "9-s.md: skill_path 'gone-skill' resolves to no SKILL.md" in lint.run()[1]


def test_the_no_skill_present_warning_is_unchanged(tmp_path):
    """The presence warning #811 must not disturb: same text, still fires alone."""
    lint = Linter(tmp_path)
    lint.add("9-s.md", id=9, name="s", status="up_next", frequency="daily")

    code, out = lint.run()

    assert "9-s.md: runnable but has no skill_name/skill_path" in out
    assert "resolves to no SKILL.md" not in out, "nothing names a skill to resolve"
    assert "1 structural warning" in out
    assert code == 0


# ── The seam: the linter must agree with what dispatch actually accepts ───────
#
# The linter runs in a shell; the thing it is predicting is `autonomy` deciding
# whether a task can run. A test that only exercises the linter against itself
# would pin a rule that quietly drifts from the loader — so these call the real
# dispatch functions over the same fixture and assert the two verdicts coincide.

# Each entry carries an `env:` block, because `_get_model_env` returns the env
# dict and an env-less entry would be indistinguishable from an unmatched one.
SEAM_CONFIG = """models:
  primary:
    alias: primary
    env:
      ANTHROPIC_BASE_URL: http://127.0.0.1:8096
  secondary:
    alias: fast
    env:
      ANTHROPIC_BASE_URL: http://127.0.0.1:8091
"""


@pytest.mark.parametrize("model", ["primary", "secondary", "fast", "eco"])
def test_the_model_check_accepts_exactly_what_the_dispatch_loader_resolves(
        tmp_path, monkeypatch, model):
    """`_get_model_env(name)` finds an env iff the linter's accepted set has `name`.

    That function answers an unmatched name with `{}` and no raise — which is how
    `model: eco` reached the primary's endpoint under a name it does not serve.
    If its matching rule ever moves, this is what says the linter moved with it.
    """
    from app import autonomy
    from scripts.autonomy import validate_tasks as vt

    (tmp_path / "config.yaml").write_text(SEAM_CONFIG)
    monkeypatch.setattr(autonomy, "LLOYD_HOME", tmp_path)

    assert bool(autonomy._get_model_env(model)) is (model in vt.model_names(tmp_path / "config.yaml"))


def test_the_skill_check_resolves_exactly_what_the_dispatch_loader_loads(
        tmp_path, monkeypatch):
    """`skill_resolves` is True iff `_load_skill_content` returns content.

    Covers both branches of the loader: the path branch (`/` or `.md`), which
    never falls back to a slug, and the slug branch against a skills root.
    """
    from app import autonomy
    from scripts.autonomy import validate_tasks as vt

    vault = tmp_path / "obsidian"
    slug_dir = vault / "skills" / "nightly-seam"
    slug_dir.mkdir(parents=True)
    (slug_dir / "SKILL.md").write_text("# nightly seam\n")
    real_path = tmp_path / "eval" / "secondary_routing_eval.py"
    real_path.parent.mkdir()
    real_path.write_text("print('here')\n")

    real_path_cls = autonomy.Path

    class _VaultHome(real_path_cls):  # Path.home() is how the loader finds skills
        @classmethod
        def home(cls):
            return tmp_path

    monkeypatch.setattr(autonomy, "Path", _VaultHome)

    cases = {
        "nightly-seam": True,                      # slug that exists in the skills root
        "no-such-slug-xyz": False,                 # slug branch miss
        str(real_path): True,                      # path branch hit
        str(tmp_path / "gone.py"): False,          # path branch miss
    }
    skills_root = vault / "skills"
    for value, expected in cases.items():
        assert (autonomy._load_skill_content(value) is not None) is expected, value
        assert vt.skill_resolves(value, [skills_root]) is expected, value


# ── #1577: a `requires_slot:` naming a slot the box has switched off ──────────
#
# `requires_slot` names a supervisord program, and two of them are optional with a
# switch in config.yaml. `secondary_enabled: false` has been in force since
# 2026-09-20 (`551e9044`, "Retire the secondary slot and put DiffusionGemma (djev)
# on GPU 2"), yet #85 kept declaring `requires_slot: agent-llm-secondary` for a
# week and this linter printed a clean pass, because name resolution cannot see a
# flag: `secondary` is still a key of `models:`, so both `model: primary` and
# `model: secondary` resolve however the flag is set. The eval refused itself
# (`EXIT_SLOT_DISABLED = 7`) and the arm guard refuses a write (#1555); the static
# report is the half that was missing, and it is what would have made the
# retirement an obvious decision instead of a nightly 404-shaped mystery.

#: The programs the check can be told about: the retired slot, the other optional
#: slot, and a program with no switch at all.
SLOT_PROGRAMS = ["agent-llm-primary", "agent-llm-secondary", "agent-djev"]


def _slot_config(secondary_enabled: bool, djev_enabled: bool) -> str:
    """The `--config` yaml with both slot switches spelled out, so the two can be
    set independently. `djev.enabled` is the other real slot
    (`app.llm_slots._SLOTS`), and a linter that hardcoded the secondary's rule has
    somewhere to be wrong in the grid below rather than passing on the one slot it
    happened to know about."""
    return (CONFIG
            + f"secondary_enabled: {'true' if secondary_enabled else 'false'}\n"
            + f"djev:\n  enabled: {'true' if djev_enabled else 'false'}\n")


def test_a_switched_off_slot_names_the_file_the_program_and_the_flag(tmp_path):
    """One line, three facts: which file, which program, which key to change.

    "disabled" is not actionable. "`secondary_enabled` is false in config.yaml" is
    the sentence that ends the search, which is why `llm_slots` carries the flag
    string rather than a boolean.
    """
    lint = Linter(tmp_path, config=_slot_config(False, False))
    lint.add("85-r.md", id=85, name="r", status="draft", frequency="daily",
             skill_name="heartbeat", requires_slot="agent-llm-secondary")

    code, out = lint.run()

    assert code == 0, "a structural warning is a failure only under --strict"
    line = next(ln for ln in out.splitlines() if "requires_slot" in ln)
    # Warnings print indented under their header; the file name is what leads the
    # text itself, and it is what tells a reader which task file to open.
    assert line.lstrip().startswith("85-r.md:"), line
    assert "agent-llm-secondary" in line
    assert "secondary_enabled" in line, "the flag a person has to flip is the point"
    assert "switched-off" in line
    assert lint.run("--strict")[0] == 2, "the state that retired #85 must fail the strict rung"


@pytest.mark.parametrize("secondary_enabled,djev_enabled",
                         [(False, False), (True, True), (False, True), (True, False)],
                         ids=["off/off", "on/on", "off/on", "on/off"])
def test_the_slot_verdict_is_app_llm_slots_verdict_for_every_program(
        tmp_path, secondary_enabled, djev_enabled):
    """Three programs under both flags: the linter speaks iff `is_enabled` is False.

    `agent-llm-primary` has no switch, and `app.llm_slots`'s rule that a program
    which is not an optional slot is always enabled is what keeps this from becoming
    a second service registry — it must stay silent whatever the flags say. The two
    split cells (`off/on`, `on/off`) are the ones a table copy dies on: with both
    flags moved together, a check that read `secondary_enabled` for *every* program
    would agree with the module by accident. All three declarations go through one
    run, so the expectation for each is read off the line its own file is named on.
    """
    from app import llm_slots

    config = _slot_config(secondary_enabled, djev_enabled)
    parsed = yaml.safe_load(config)
    lint = Linter(tmp_path, config=config)
    files = {program: f"{n}-r.md" for n, program in enumerate(SLOT_PROGRAMS, start=1)}
    for program, fname in files.items():
        lint.add(fname, id=int(fname.split("-")[0]), name="r", status="up_next",
                 frequency="daily", skill_name="heartbeat", requires_slot=program)

    code, out = lint.run("--strict")
    lines = [ln for ln in out.splitlines() if "requires_slot" in ln]

    for program, fname in files.items():
        mine = [ln for ln in lines if ln.lstrip().startswith(fname + ":")]
        assert len(mine) <= 1, f"{fname} reported twice: {mine}"
        reported = bool(mine)
        enabled = llm_slots.is_enabled(program, parsed)
        assert reported is not enabled, (
            f"{program} with secondary_enabled={secondary_enabled}, "
            f"djev.enabled={djev_enabled}: llm_slots.is_enabled answers {enabled}, "
            f"the linter reported {mine or 'nothing'}")
        if reported:
            flag = llm_slots.slot_flag(program)
            assert flag in mine[0], (
                f"the line names a flag of its own, not the module's {flag!r}: {mine[0]}")

    anything_off = any(not llm_slots.is_enabled(p, parsed) for p in SLOT_PROGRAMS)
    assert code == (2 if anything_off else 0), out


def test_an_unreadable_config_leaves_a_slot_declaration_unchecked(tmp_path):
    """No verdict is not a switched-off verdict: `is_enabled` says False for a flag
    that is merely absent, so an unparseable config would otherwise invent a
    retired slot out of a broken file."""
    lint = Linter(tmp_path, config="models: [this never closes\n")
    lint.add("9-r.md", id=9, name="r", status="up_next", frequency="daily",
             skill_name="heartbeat", requires_slot="agent-llm-secondary")

    out = lint.run()[1]

    assert "requires_slot 'agent-llm-secondary' unchecked" in out
    assert "switched-off" not in out


# ── #2445: a `frequency:` word beside a `runs_per_day:` that contradicts it ───
#
# Task #30 carried `frequency: daily` (line 25 of its file) and `runs_per_day: 3`
# (line 39). `_frequency_interval_seconds` (app/autonomy.py:1227-1238) tries
# `runs_per_day` FIRST, so the task resolved 86400/3 = 28800 s and dispatched
# three times a day — measured as exactly 3 runs on every one of the eight UTC
# days 2026-10-01..2026-10-08 in `~/lloyd-data/workers.db`. The linter passed it:
# the whole frequency check sat behind `(not rpd or _is_nullish(rpd))`
# (validate_tasks.py:322), so a file carrying BOTH fields never had the two
# compared, and the run over 38 live files printed "✅ all task files parse and
# pass structural checks" with 0 warnings.


def test_a_frequency_word_contradicted_by_runs_per_day_names_the_file(tmp_path):
    """The warning has to carry both intervals, because which one wins is the finding.

    A reader who is told only "`daily` and 3 disagree" cannot tell whether the job
    is running too often or too seldom; the line says the scheduler dispatches at
    28800 s, which is the 8-hour cadence the item was filed over.
    """
    lint = Linter(tmp_path)
    lint.add("30-i.md", id=30, name="intel", status="up_next",
             frequency="daily", runs_per_day=3, skill_name="heartbeat")

    code, out = lint.run()

    assert code == 0, "a structural warning is only a failure under --strict"
    # Warnings are indented under the "structural warning(s)" header, so the
    # match is on the substring the other checks in this file use.
    line = next((l for l in out.splitlines() if "30-i.md:" in l), "")
    assert "30-i.md: frequency 'daily'" in line, \
        f"no frequency warning for this file: {out!r}"
    assert "runs_per_day" in line and " 3" in line, line
    assert "86400" in line and "28800" in line, (
        "the line must name the period the word declares AND the one runs_per_day "
        f"resolves to, or it cannot say which way the job is wrong: {line!r}")
    assert "runs_per_day wins" in line, (
        f"the line must say which field the scheduler obeyed: {line!r}")


def test_a_contradicted_frequency_fails_the_strict_rung(tmp_path):
    """`--strict` is the rung a job that runs 3x more than declared belongs on."""
    lint = Linter(tmp_path)
    lint.add("30-i.md", id=30, name="intel", status="up_next",
             frequency="daily", runs_per_day=3, skill_name="heartbeat")

    code, out = lint.run("--strict")

    assert code == 2, f"expected a strict failure, got exit {code}: {out!r}"
    assert "30-i.md" in out


@pytest.mark.parametrize("frequency,runs_per_day", [
    ("daily", 1),            # 14 board files, exactly: 86400/1 == 86400
    ("weekly", 0.14),        # #47/#54/#67: 617142.9 vs 604800, 2.04% — two decimals of 1/7
    ("hourly", 24),          # 3600 == 3600
    ("every-15min", 96),     # 900 == 900
    ("6x-daily", 6),         # #24: the label is OUTSIDE the map on purpose
    ("weekly", 0),           # a zero rpd is skipped by the scheduler, so the word rules
    ("daily", "three"),      # a non-number likewise
])
def test_an_agreeing_pair_or_a_label_outside_the_map_stays_silent(
        tmp_path, frequency, runs_per_day):
    """The other 17 board files must gain no warning, and #24's idiom must stay legal.

    `weekly` + `0.14` is the case an exact comparison would get wrong: three files
    carry it and it is 2.04% off 604800 s only because 0.14 is the two-decimal
    rounding of 1/7. #30's `daily` + `3` was 66.67% off, which is a contradiction
    and not a rounding artefact.
    """
    lint = Linter(tmp_path)
    lint.add("7-x.md", id=7, name="x", status="up_next",
             frequency=frequency, runs_per_day=runs_per_day,
             skill_name="heartbeat")

    code, out = lint.run("--strict")

    assert code == 0, f"{frequency}+{runs_per_day} warned: {out!r}"
    assert "7-x.md" not in out


def test_the_seventeen_board_files_that_agree_still_lint_clean(tmp_path):
    """The whole surviving board in one run, not one pair at a time.

    The 17 files that carry both fields, as measured on `~/obsidian/autonomy`
    on 2026-10-08: thirteen `daily`+1, three `weekly`+0.14, one `6x-daily`+6.
    Per-pair tests can each pass while a linter that warns on *any* pair still
    floods the real board; this run is the one that says they would not.
    """
    lint = Linter(tmp_path)
    board = [(f"{n}-x.md", "daily", 1) for n in
             (35, 51, 53, 65, 74, 76, 81, 82, 86, 90, 93, 94, 95)]
    board += [(f"{n}-x.md", "weekly", 0.14) for n in (47, 54, 67)]
    board += [("24-x.md", "6x-daily", 6)]
    assert len(board) == 17, board
    for name, frequency, runs_per_day in board:
        lint.add(name, id=name.split("-")[0], name="x", status="up_next",
                 frequency=frequency, runs_per_day=runs_per_day,
                 skill_name="heartbeat")

    code, out = lint.run("--strict")

    assert code == 0, f"a board file that agrees with itself warned: {out!r}"
    assert "Scanned 17 task files" in out, out


def test_a_label_outside_the_map_with_no_runs_per_day_still_warns(tmp_path):
    """#815's check is untouched by widening the gate around it.

    That warning is the reason #24's `runs_per_day: 6` going missing would be
    caught rather than parking a nightly pipeline in silence, so it must survive
    the branch rewrite that added the disagreement check beside it.
    """
    lint = Linter(tmp_path)
    lint.add("24-x.md", id=24, name="pipeline", status="up_next",
             frequency="6x-daily", skill_name="heartbeat")

    code, out = lint.run()

    assert code == 0
    assert "24-x.md: frequency '6x-daily' is not one of" in out, out
    assert "never dispatch it" in out, out


# ── The pre-existing contract, pinned so the new checks cannot quietly widen ──

def test_an_unparseable_file_still_exits_1_and_outranks_warnings(tmp_path):
    """Exit 1 (scheduler-invisible) beats exit 2, strict or not."""
    lint = Linter(tmp_path)
    # No leading '---' block at all, which is what _parse reports as
    # "no frontmatter" and the scheduler then drops silently.
    (lint.autonomy_dir / "1-bad.md").write_text("id: 1\nname: bad\n")
    lint.add("2-y.md", id=2, name="y", status="up_next", frequency="daily",
             depends_on=99, skill_name="heartbeat")

    code, out = lint.run("--strict")

    assert code == 1
    assert "INVISIBLE to the scheduler" in out
    assert "1-bad.md" in out


def test_notes_in_the_dir_are_not_linted(tmp_path):
    """Only NN-name.md files are task files; the scheduler ignores the rest."""
    lint = Linter(tmp_path)
    (lint.autonomy_dir / "meta-analysis-2026-06-03.md").write_text(
        "---\nid: 999\nstatus: paused\ndepends_on: 1\n---\nnotes\n")

    code, out = lint.run("--strict")

    assert "Scanned 0 task files" in out
    assert code == 0, out


# ── #2632: a `model:` naming a slot the box has switched off ─────────────────
#
# The `requires_slot:` arm above (#1577) guards a field NO task file declares: on
# 2026-10-12 `grep -ln requires_slot ~/obsidian/autonomy/*.md` returned nothing at
# all, while the one field that does carry a slot name — `model:` — was invisible to
# it. #68 `morning-brief-triage` has declared `model: secondary` since it was created,
# and `secondary_enabled: false` has been in force since `551e9044` (2026-09-20, "Retire
# the secondary slot and put DiffusionGemma (djev) on GPU 2"), so the check that exists
# could not see the one case it was written for, and the check that could not see this
# one was guarding nothing.
#
# The two slot guards both key on `requires_slot:` and skip `model:` for the same
# reason: `slot_arm_block` (app/autonomy.py:5706-5710) returns None for a task that
# declares none, and #1577's lint read that field alone. That is not an oversight to
# "fix" here — arming #68 is correctly *not* refused, and
# tests/test_autonomy_slot_arm_guard.py:113 pins exactly that
# (`assert A.slot_arm_block({"id": 68}, "up_next") is None`) and stays green. What was
# missing is the static report: a value that routes every run to the primary through
# `resolve_model_alias` (app/config.py:500-507) while the task file says otherwise is a
# stale line, and the linter is where a stale line gets said out loud.

#: `models:` alias → the supervisord program whose switch governs it. Restated here as
#: the test's own expectation, not imported: the split cells in the grid below die on
#: exactly this relation, and a test that imported the map would agree with a linter
#: that read the wrong flag for the wrong alias. `app/supervisor_client.py:72`
#: `_MODEL_ROW_ALIASES` is the tree's one alias↔row map and the linter imports it.
ALIAS_PROGRAM = {"primary": "agent-llm-primary", "secondary": "agent-llm-secondary"}

live_vault = pytest.mark.live_vault


def test_a_model_naming_a_switched_off_slot_names_the_file_the_value_and_the_flag(tmp_path):
    """One line, three facts: which file, which value, which key to change.

    This is the #68 case exactly — a `draft` task file, a slot nobody boots, and a
    name check that resolves it as clean because `secondary` is a `models:` key
    whatever the flag says. The retired #85 case above needed `requires_slot:` to be
    declared; this one needs only the field every task file already has.
    """
    lint = Linter(tmp_path, config=_slot_config(False, False))
    lint.add("68-b.md", id=68, name="b", status="draft", frequency="daily",
             skill_name="heartbeat", model="secondary")

    code, out = lint.run()

    assert code == 0, "a structural warning is a failure only under --strict"
    line = next(ln for ln in out.splitlines() if "switched-off" in ln)
    assert line.lstrip().startswith("68-b.md:"), line
    assert "model 'secondary'" in line, line
    assert "secondary_enabled" in line, "the flag a person has to flip is the point"
    assert lint.run("--strict")[0] == 2, (
        "the state #68 sat in for three weeks must fail the strict rung")


@pytest.mark.parametrize("secondary_enabled,djev_enabled",
                         [(False, False), (True, True), (False, True), (True, False)],
                         ids=["off/off", "on/on", "off/on", "on/off"])
def test_the_model_verdict_is_app_llm_slots_verdict_for_every_alias(
        tmp_path, secondary_enabled, djev_enabled):
    """Both aliases under both flags: the linter speaks iff the alias's slot is off.

    The four-way grid is the one the #1577 check already runs, and it is the shape a
    hardcoded rule dies on: `model: primary` names a program with no switch, so
    `app.llm_slots`'s rule that a non-optional program is always enabled has to keep
    it silent in every cell — including `off/*`, where a check that consulted
    `secondary_enabled` for every alias would report the primary as retired. The two
    split cells say the same about the other direction: a linter that flagged
    `secondary` whenever `djev.enabled` disagreed would go red in `off/on`.

    Both declarations go through one run, so each file's expectation is read off the
    line its own file is named on.
    """
    from app import llm_slots

    config = _slot_config(secondary_enabled, djev_enabled)
    parsed = yaml.safe_load(config)
    lint = Linter(tmp_path, config=config)
    files = {alias: f"{n}-m.md" for n, alias in enumerate(ALIAS_PROGRAM, start=1)}
    for alias, fname in files.items():
        lint.add(fname, id=int(fname.split("-")[0]), name="m", status="up_next",
                 frequency="daily", skill_name="heartbeat", model=alias)

    code, out = lint.run("--strict")
    lines = [ln for ln in out.splitlines() if "switched-off" in ln]

    for alias, fname in files.items():
        mine = [ln for ln in lines if ln.lstrip().startswith(fname + ":")]
        assert len(mine) <= 1, f"{fname} reported twice: {mine}"
        program = ALIAS_PROGRAM[alias]
        enabled = llm_slots.is_enabled(program, parsed)
        assert bool(mine) is not enabled, (
            f"model '{alias}' (program {program}) with secondary_enabled="
            f"{secondary_enabled}, djev.enabled={djev_enabled}: llm_slots.is_enabled "
            f"answers {enabled}, the linter reported {mine or 'nothing'}")
        if mine:
            flag = llm_slots.slot_flag(program)
            assert flag is not None, (
                f"{program} is an optional slot with no flag to name — the line "
                f"cannot cite one: {mine[0]}")
            assert flag in mine[0], (
                f"the line names a flag of its own, not the module's {flag!r}: "
                f"{mine[0]}")

    any_off = any(not llm_slots.is_enabled(ALIAS_PROGRAM[a], parsed) for a in files)
    assert code == (2 if any_off else 0), out


@pytest.mark.parametrize("bad_config", [
    pytest.param("models: [this never closes\n", id="unparseable"),
    pytest.param("secondary_enabled: true\n", id="no-models-block"),
])
def test_an_unreadable_or_models_less_config_leaves_the_model_check_unchecked(
        tmp_path, bad_config):
    """Clause 4, the `model:` half of the shape #1577 established.

    `is_enabled` answers False for a flag that is merely absent, so a config that
    cannot be read would otherwise print a retired-slot verdict nobody configured.
    The sentence has to say `unchecked`, and must not say `switched-off` — which is
    the same rule the `requires_slot:` arm applies, reached through the same
    `config_mapping` read rather than a second copy of it.

    Two bad shapes, because they fail differently. `unparseable` is YAML
    `yaml.safe_load` raises on, so NEITHER leg can answer and neither does.
    `no-models-block` is worse for a reader: the slot flags ARE readable and say the
    secondary engine is booted, so a check that conflated the two legs would print
    "all clear" about an alias that exists in no `models:` table at all. The name leg
    prints `unchecked` and the slot leg stays silent — that asymmetry is the point.
    """
    lint = Linter(tmp_path, config=bad_config)
    lint.add("68-b.md", id=68, name="b", status="draft", frequency="daily",
             skill_name="heartbeat", model="secondary")

    out = lint.run()[1]

    assert "model 'secondary' unchecked" in out, out
    assert "switched-off" not in out, out
    assert lint.run("--strict")[0] == 2, "an unchecked value is not a passing value"


def test_the_alias_to_program_map_is_the_service_registrys():
    """The linter must derive the slot from `supervisor_client`, not from a copy.

    `_MODEL_ROW_ALIASES` is where the tree says which model alias an engine row
    tracks. If that constant is renamed or reshaped, the import in the linter falls
    back to None and every `model:` line starts reporting `unchecked` — loud, but
    nobody runs this linter on a schedule the way CI runs pytest, so the drift needs
    a node that goes red here.
    """
    from app.supervisor_client import _MODEL_ROW_ALIASES
    from scripts.autonomy import validate_tasks as vt

    registry_map = {alias: program
                    for program, alias in _MODEL_ROW_ALIASES.items()}
    assert vt.MODEL_ROW_ALIASES == registry_map, (
        f"the linter maps model aliases to programs as {vt.MODEL_ROW_ALIASES} while "
        f"app/supervisor_client.py says {registry_map}; the map is imported rather "
        f"than restated precisely so the two cannot disagree")
    assert set(ALIAS_PROGRAM) <= set(registry_map), (
        f"this file expects {sorted(ALIAS_PROGRAM)} and the registry no longer "
        f"covers them: {sorted(registry_map)}")


#: §#68 of `architecture/autonomy-jobs.md` states the fleet's `model:` census in prose.
#: The real-board node below takes its expectation out of that sentence rather than from a
#: number typed into an assertion, because the typed number is what broke: the fleet gained
#: #98 `retag-untagged-facts` on 2026-10-14 and the node went red over a sentence in another
#: file nobody was editing — a failure whose fix is not the assertion the reader is shown.
#: Read from the doc, the same event names the parted sentence instead, and retyping the doc
#: is then the entire repair. This is the same bargain the doc's own header states for every
#: count it types: dated, and pinned to something re-derivable.
DOC = ROOT / "architecture" / "autonomy-jobs.md"

#: The census sentence, spelled the way the doc spells it since #2892: a FLOOR. The grep
#: command is part of the pattern on purpose: it is what fixes the number to the measurement
#: it reports, so a stray "`primary` 12 times" elsewhere in the document cannot be mistaken
#: for the census. The exact spelling — `answers `primary` 43 times`, no floor words — is
#: deliberately NOT matched any more: retyping that number was the whole repair three days
#: running (#2747 on 2026-10-15, #2814 on 2026-10-16, #2882 on 2026-10-17), so a doc that
#: has drifted back to an exact count is the defect and must raise, not parse.
DOC_CENSUS = re.compile(
    r"`grep -h '\^model:' ~/obsidian/autonomy/\*\.md`\s+"
    r"(?P<sentence>answers\s+`(?P<value>\w+)`\s+at\s+least\s+(?P<count>\d+)\s+times?"
    r"\s+or\s+beyond)")

#: How the doc spells the measurement the census reports, in backticks.
CENSUS_CMD = "`grep -h '^model:' ~/obsidian/autonomy/*.md`"


def census_prose(count: int, value: str = "primary") -> str:
    """One census sentence in the doc's own floor spelling, for the parser's own tests."""
    return f"{CENSUS_CMD} answers `{value}` at least {count} times or beyond"


def doc_model_census(doc_text: str) -> tuple[int, str]:
    """The FLOOR §#68 puts on the fleet's `model:` census, with the sentence that states it.

    The int is a MINIMUM, not the fleet's size: since #2892 the sentence reads "at least N …
    or beyond", so a fleet that grows past N satisfies it and only a fleet that drops under
    N contradicts the doc. That is the whole point of the floor — under the exact spelling
    this replaced, every task anyone filed parted the sentence and the repair was a retype.

    Raises rather than defaulting. A doc reworded past this shape has stopped being
    pinned to the fleet, and the node that depends on the pin must say so — the silent
    alternative is a node that quietly returns to green while measuring nothing against
    the sentence, which is the blind spot this replaced. Two sentences quoting different
    numbers raise for the same reason: the doc would be holding two censuses.
    """
    hits = list(DOC_CENSUS.finditer(doc_text))
    if not hits:
        raise AssertionError(
            "architecture/autonomy-jobs.md no longer states its `model:` census in the "
            "floor shape `answers `primary` at least N times or beyond`, so the doc and "
            "the fleet are no longer joined — restate the sentence in that shape, with the "
            "date it was measured, rather than dropping the join. An exact `answers "
            "`primary` N times` is NOT the shape: retyping that number was the entire "
            "repair on 2026-10-15 (#2747), 2026-10-16 (#2814) and 2026-10-17 (#2882)")
    numbers = {int(h.group("count")) for h in hits}
    if len(numbers) > 1:
        raise AssertionError(
            f"the doc states {len(hits)} `model:` censuses disagreeing with each other "
            f"{sorted(numbers)}; the node cannot tell which one the fleet is meant to match")
    return int(hits[0].group("count")), " ".join(hits[0].group("sentence").split())


def census_below_floor_message(fleet_count: int, floor: int, sentence: str) -> str:
    """What the real-board node says when the fleet drops UNDER §#68's floor — pinned,
    because it is the whole product of that failure: a reader with nothing but the
    assertion has to see the sentence, both numbers, which file to edit, and the fact that
    a floor moves only when tasks are RETIRED, never when they arrive.
    """
    return (
        f"{fleet_count} task files declare a `model:`; §#68 of "
        f"architecture/autonomy-jobs.md floors the census at {floor}, so the fleet has "
        f"fallen under the minimum the doc states — that happens only if task files were "
        f"retired, so confirm the retirement was intended before lowering the number. Its "
        f"sentence reads: {sentence!r}. Re-count with `grep -h '^model:' "
        f"~/obsidian/autonomy/*.md | wc -l`, retype that sentence with the new minimum and "
        f"the date it was measured; this assertion needs no edit, which is the point of "
        f"reading the number out of the doc.")


def fleet_reaches_census_floor(declared: list[str], floor: int, sentence: str) -> None:
    """The join between §#68's stated floor and a measured fleet: the fleet must REACH the
    minimum, never equal it.

    A function rather than an ``assert len(declared) >= stated`` inline in the real-board
    node, because that node carries ``@live_vault`` and the gate deselects it
    (``pytest.ini:14-15``): an inline comparison could only be pinned by reading the node's
    source, which cannot tell ``>=`` from ``==`` in any way a failing run demonstrates. Here
    both directions run at every gate —
    ``test_the_floor_join_lets_a_grown_fleet_pass_and_a_shrunken_one_fail``.
    """
    assert len(declared) >= floor, census_below_floor_message(
        len(declared), floor, sentence)


@live_vault
def test_the_real_board_lints_clean_now_that_the_model_slot_check_is_live(tmp_path):
    """Clause 3's real-directory leg: 0 warnings, exit 0, over `~/obsidian/autonomy`.

    The grid above proves the rule on fixtures; this proves the fleet satisfies it,
    which is the half that could only be true after the vault half of #2632 landed
    (`autonomy/68-morning-brief-triage.md` `model: secondary` → `primary`, vault
    commit c467ed53). Before that flip this node was the whole point: the one file
    naming the retired slot was #68, and the new check would have failed the strict
    rung over a task nobody runs.

    `@live_vault` for the standing reason (`pytest.ini:10-13`): the vault is written
    by nightly jobs and by `autonomy_write_task`, so the automod gate deselects this
    and the fixture-backed nodes carry the rule there. Run it with

        ~/lloyd/.venvs/lloyd/bin/python -m pytest \\
            tests/test_validate_tasks_linter.py -q -k real_board_lints_clean

    The config is the checkout's own (`DEFAULT_CONFIG`, two levels above the script),
    which is the config a scheduled run would read; only `--autonomy-dir`/`--skills-dir`
    defaults are relied on, so no path is hardcoded here.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict"],
        capture_output=True, text=True, timeout=180,
    )
    out = proc.stdout + proc.stderr

    assert "structural warning" not in out, out
    assert "unchecked" not in out, out
    assert "✅ all task files parse and pass structural checks" in out, out
    assert proc.returncode == 0, out

    # §#68 of architecture/autonomy-jobs.md FLOORS this census in prose, so the expected
    # minimum is read out of that sentence and never typed here: the doc-claims suite cannot
    # see the vault's front matter, and this is the only node that reads both the directory
    # and the flags the sentence is about. Typed as an equality it went stale on 2026-10-14,
    # then again on 2026-10-16 and 2026-10-17; read as a floor, filing a task can never
    # redden this node — only retiring one can, and then the message says so.
    from scripts.autonomy import validate_tasks as vt
    declared = [ln.strip() for p in vt.DEFAULT_DIR.glob("*.md")
                for ln in p.read_text(errors="replace").splitlines()
                if ln.startswith("model:")]
    floor, sentence = doc_model_census(DOC.read_text(encoding="utf-8"))
    fleet_reaches_census_floor(declared, floor, sentence)
    assert set(declared) == {"model: primary"}, (
        f"a task file names an engine other than the primary: {sorted(set(declared))} "
        f"— §#68 states that every `model:` line in the declaring fleet reads `primary`, "
        f"the census having first been taken the day #2632 landed the check on 2026-10-13")


# ── #2747: the doc's census is a joined claim, not a number two files agree on ──
#
# `architecture/autonomy-jobs.md` §#68 says `grep -h '^model:' ~/obsidian/autonomy/*.md`
# answers `primary` at least N times or beyond. The live node above reads that FLOOR out of
# the sentence, and the nodes here are the gate-run half: they run in every checkout
# (`live_vault` is deselected by `pytest.ini:14-15`, so the real-board node is exercised only
# by #83's nightly Pre-Flight and by hand), and they pin the parser, the floor shape, the
# join's two directions, the failure text, and the absence of a hand-typed expectation.
# #2892 moved the sentence from an exact count to a floor after the same number had been
# retyped on three consecutive days — #2747 39→40, #2814 40→41, #2882 41→43 — which is the
# event that proves a typed ceiling on a growing fleet is unenforceable where no gate runs.


def test_the_census_expectation_is_parsed_out_of_the_doc_sentence():
    """Clauses 1 and 2's gate-run half: the expectation is a parse of a FLOOR, and it is dated.

    Two synthetic sentences quoting different minima are parsed to those two numbers, so the
    value demonstrably comes from the text and not from a constant the parser happens to
    agree with. Then the real doc is read: it must hold exactly one census, it must state
    `primary` (the value #2632 settled the fleet on), it must be FLOOR-spelled — "at least",
    a number, "or beyond", and the word "minimum" beside it, so a reader is not left to infer
    that the count is not a count — the measurement command must still sit immediately in
    front of it, and an ISO date must sit inside the 200 characters before the number (the
    doc's own header rule is that "every count the doc does type carries the date it was
    true", and an undated floor is a claim no later reader can tell stale from correct).

    The NUMBER is deliberately not pinned here, only its shape. A typed count in a node that
    runs at the gate is the same #2747 defect one file over: the count is the fleet's, the
    fleet grows, and `pytest.ini:14-15` deselects the one node that may lawfully compare it
    to the directory. So this node proves the doc still floors the census at some positive
    size on the primary, and only `@live_vault`
    `test_the_real_board_lints_clean_now_that_the_model_slot_check_is_live` asks whether the
    fleet reaches that size.
    """
    assert doc_model_census(census_prose(40)) == (
        40, "answers `primary` at least 40 times or beyond")
    assert doc_model_census(f"…prose… {census_prose(41)} …prose…") == (
        41, "answers `primary` at least 41 times or beyond")
    assert doc_model_census(
        f"{CENSUS_CMD} answers `primary` at least 1 time or beyond\n") == (
        1, "answers `primary` at least 1 time or beyond"), (
        "the parser's plural allowance has to hold at one, or a fleet that shrank to a "
        "single declaring task file would read as a broken doc")

    text = DOC.read_text(encoding="utf-8")
    stated, sentence = doc_model_census(text)
    assert stated > 0, (
        f"the census floor parsed to {stated}; a zero would state no floor at all")
    assert re.fullmatch(r"answers `primary` at least \d+ times? or beyond", sentence), (
        f"§#68 must state the census as a floor in the shape answers `primary` at least "
        f"<digits> times or beyond — the value is `primary` because every task file "
        f"declaring a `model:` reads `primary` — and it reads {sentence!r}")
    hit = DOC_CENSUS.search(text)
    window = text[max(0, hit.start() - 200):hit.end()]
    assert re.search(r"20\d\d-\d\d-\d\d", window), (
        f"the census sentence carries no date within 200 characters before the count: "
        f"…{window[-160:]!r} — an undated count is exactly the sentence that rotted here")
    assert text[hit.start() + len(CENSUS_CMD):hit.start("sentence")].strip() == "", (
        f"the measurement command no longer immediately precedes the census number: "
        f"{text[hit.start():hit.start('sentence')]!r} — the command is what fixes the "
        f"number to the measurement it reports")
    tail = text[hit.end():hit.end() + 160]
    assert "minimum" in tail, (
        f"the floor is not labelled in words: nothing within 160 characters after "
        f"{sentence!r} says the number is a minimum — {tail!r}. A bare 'at least N' reads "
        f"as a count to the next editor, which is how it got retyped instead of left alone")


def test_the_floor_join_lets_a_grown_fleet_pass_and_a_shrunken_one_fail():
    """Clause 3's gate-run leg: the join is ``>=``, demonstrated in both directions.

    The real-board node that runs this against `~/obsidian/autonomy` is deselected at the
    gate, so without this node the difference between ``>=`` and ``==`` would rest on a
    source-text regex in the rail below. Here a fleet of 41 against a floor of 40 passes —
    the exact event that reddened the old equality three times — and a fleet of 39 against
    the same floor fails naming the floor, the count, the sentence and the file.
    """
    sentence = "answers `primary` at least 40 times or beyond"
    fleet_reaches_census_floor(["model: primary"] * 41, 40, sentence)
    fleet_reaches_census_floor(["model: primary"] * 40, 40, sentence)
    with pytest.raises(AssertionError) as shrunk:
        fleet_reaches_census_floor(["model: primary"] * 39, 40, sentence)
    msg = str(shrunk.value)
    assert "39 task files declare a `model:`" in msg, msg
    assert "floors the census at 40" in msg, msg
    assert sentence in msg, msg


def test_a_reworded_or_doubled_census_breaks_the_join_loudly():
    """The parser may not degrade quietly, which is how the old pin went blind.

    A doc reworded past the floor shape used to leave the node asserting a number nobody had
    looked at; here it raises, and says which join broke. Two sentences quoting different
    numbers raise too — a doc holding two censuses cannot tell the node which minimum the
    fleet is meant to reach, and picking one at random is the same silent wrong answer.

    The two spellings #2892 retired raise as well. An exact `answers `primary` 43 times` is
    no longer the shape: parsing it would let the doc drift back to the sentence that had to
    be retyped on 2026-10-15, 2026-10-16 and 2026-10-17 and call that green. A floor with no
    "or beyond" is #1520's ceiling defect in floor clothing — true today, false the day the
    next task is filed — so it is refused for the same reason the numbering note refuses one.
    """
    with pytest.raises(AssertionError) as gone:
        doc_model_census("the fleet has forty model lines in it\n")
    assert "no longer states" in str(gone.value), str(gone.value)

    with pytest.raises(AssertionError) as exact:
        doc_model_census(f"{CENSUS_CMD} answers `primary` 43 times\n")
    assert "no longer states" in str(exact.value), str(exact.value)

    with pytest.raises(AssertionError) as capped:
        doc_model_census(f"{CENSUS_CMD} answers `primary` at least 43 times\n")
    assert "no longer states" in str(capped.value), str(capped.value)

    with pytest.raises(AssertionError) as doubled:
        doc_model_census(
            f"{census_prose(40)} … and elsewhere {census_prose(41)}\n")
    assert "disagreeing" in str(doubled.value), str(doubled.value)

    # A doc that repeats the *same* floor is dull, not broken: no raise.
    assert doc_model_census(
        f"{census_prose(40)} … {census_prose(40)}\n") == (
        40, "answers `primary` at least 40 times or beyond")

    # A stray "`primary` 12 times" with no measurement in front of it is not the census:
    # the command prefix is what joins the number to the thing it counts.
    with pytest.raises(AssertionError):
        doc_model_census("the model table lists `primary` 12 times in all\n")


def test_the_census_floor_is_the_only_one_in_architecture():
    """Clause 1's "stays the only census" leg, widened past §#68's own file.

    #1520 found its range bound duplicated in `index.md`, where "each of the 32 scheduled
    jobs" rotted inside three days while the note being repaired stayed green. A second
    census anywhere under `architecture/` is the same defect in another file, so this walks
    the directory: any sentence in either the retired exact spelling or the floor spelling
    must be the one §#68 sentence the parser already found.
    """
    doc_text = DOC.read_text(encoding="utf-8")
    stated, sentence = doc_model_census(doc_text)
    census_files = sorted(p.relative_to(ROOT).as_posix()
                          for p in (ROOT / "architecture").rglob("*.md")
                          if DOC_CENSUS.search(p.read_text(encoding="utf-8")))
    assert census_files == ["architecture/autonomy-jobs.md"], (
        f"a `model:` census in the floor spelling turned up in {census_files}; §#68 is the "
        f"only place the fleet's `model:` census belongs — a copy in another file is "
        f"re-typed by whoever notices it rotting, which is the #2747 loop again")
    # Wraps are formatting, not meaning: the doc breaks near 88 columns, so this leg reads
    # the flattened prose, exactly as `test_autonomy_jobs_doc_claims._flat` does for the
    # numbering note. An exact-string match against raw lines would fail on a rewrap.
    flat = re.sub(r"\s+", " ", doc_text)
    exact = re.findall(r"answers `primary` (?:at least )?\d+ times?", flat)
    assert len(exact) == 1 and exact[0] == f"answers `primary` at least {stated} times", (
        f"architecture/autonomy-jobs.md carries {len(exact)} census-shaped counts "
        f"{exact}, of which the parser found one floor of {stated}: {sentence!r}")


#: A census number typed into source, in EITHER spelling: the exact count the doc used until
#: #2892 and the floor it uses now. The floor form has to match too — a rail that matches
#: only the retired spelling reads an empty set the moment the doc stops typing an exact
#: count, and the ban below then passes over nothing while looking green.
TYPED_CENSUS = re.compile(
    r"answers\s+`primary`\s+(?:at\s+least\s+)?\d+\s+times?")


def typed_census_on_doc_reads(node_src: str) -> list[str]:
    """Lines of one test node that compare a TYPED census number against the real doc.

    The distinction this rail has to get right, because both spellings are in the file:
    `doc_model_census(census_prose(40)) == (40, "answers `primary` at least 40 times or
    beyond")` is the parser measured against prose the node itself wrote, and is fine. What
    must never come back is a typed number beside a value bound from `DOC.read_text` — that
    number is the fleet's, and the fleet grows, which is precisely the assertion that went
    red on 2026-10-14 (#2747), 2026-10-16 (#2814) and 2026-10-17 (#2882) over a file nobody
    was editing.
    """
    doc_names = set(re.findall(r"(\w+)\s*=\s*DOC\.read_text\b", node_src))
    for m in re.finditer(
            r"(\w+)(?:\s*,\s*(\w+))?\s*=\s*doc_model_census\(([^)]*)\)", node_src):
        arg = m.group(3).strip()
        if "DOC.read_text" in arg or arg in doc_names:
            doc_names.update(n for n in m.groups()[:2] if n)
    return [ln for ln in node_src.splitlines()
            if TYPED_CENSUS.search(ln)
            and any(re.search(rf"\b{re.escape(n)}\b", ln) for n in doc_names)]


#: One node in the shape the rail exists to refuse, and one in the shape it must keep
#: allowing. They sit at module level rather than inline in the rail node because the sweep
#: reads test NODES, and a fixture written inside the node it is meant to prove would be
#: read back as an offence by the very rule it demonstrates (it is: it was).
#: One node in the shape the rail exists to refuse, and one in the shape it must keep
#: allowing. They sit at module level rather than inline in the rail node because the sweep
#: reads test NODES, and a fixture written inside the node it is meant to prove would be
#: read back as an offence by the very rule it demonstrates (it is: it was). Both are written
#: in the doc's CURRENT floor spelling: a fixture left in the retired exact spelling would
#: still pass the detector check while proving nothing about the sentence the doc now holds.
OFFENDER_SRC = (
    'def test_bench_typed_count():\n'
    '    text = DOC.read_text(encoding="utf-8")\n'
    '    stated, sentence = doc_model_census(text)\n'
    '    assert sentence == "answers `primary` at least 40 times or beyond"\n')

ALLOWED_SRC = (
    'def test_bench_synthetic_prose():\n'
    '    assert doc_model_census(census_prose(40)) == (40, '
    '"answers `primary` at least 40 times or beyond")\n')

#: The same pair in the spelling #2892 retired, so the rail's reach across BOTH is shown:
#: widening TYPED_CENSUS for the floor must not have dropped the exact count, which is the
#: shape a rewording can slide back into.
OFFENDER_SRC_EXACT = (
    'def test_bench_typed_count():\n'
    '    text = DOC.read_text(encoding="utf-8")\n'
    '    stated, sentence = doc_model_census(text)\n'
    '    assert sentence == "answers `primary` 40 times"\n')


def test_the_real_board_node_holds_no_typed_census_and_names_the_floor_when_the_fleet_drops():
    """Clauses 3 and 4's regression rail, run at the gate where the live node is not.

    The first assertions are the ones that keep this fix from being undone by the next
    convenient edit: the real-board node must derive its expectation from the doc, must hand
    it to `fleet_reaches_census_floor` rather than comparing inline, must not compare the
    census with `==` at all, and must still assert the exact engine property. A typed number
    back in that node means it is failing on a fleet that grew rather than on a sentence the
    doc owns.

    Then the failure text, which is the only thing a reader gets at 03:00 in a nightly log:
    both numbers, the sentence verbatim, the command that re-counts, and the file to retype.

    The last leg generalises the first across the whole file, because #2814 found the same
    typed number living in a SECOND node — `test_the_census_expectation_is_parsed_out_of_the_doc_sentence`
    — which carries no `live_vault` mark and so runs at every gate: retyping the doc for a
    grown fleet turned that green node red. A typed census number compared against the doc is
    refused wherever the node is not marked `live_vault`, so only the node whose whole job is
    to measure the fleet can ever be reddened by the fleet. Since #2892 that ban reads BOTH
    spellings, because the doc's sentence no longer types an exact count and a rail matched
    to the retired spelling alone would sweep an empty set and report it as clean.
    """
    src = inspect.getsource(
        test_the_real_board_lints_clean_now_that_the_model_slot_check_is_live)
    assert "doc_model_census(DOC.read_text" in src, (
        "the real-board node stopped deriving its expectation from the doc")
    assert "fleet_reaches_census_floor(declared, floor, sentence)" in src, (
        "the real-board node no longer runs the census through the floor join, so the "
        "comparison is back inline where only a deselected node can see it")
    assert not re.search(r"len\(declared\)\s*==", src), (
        "an equality against the census count is back in the real-board node; equality is "
        "the shape that went stale on 2026-10-14, 2026-10-16 and 2026-10-17 when the fleet "
        "grew past the number the doc typed")
    assert re.search(r"set\(declared\)\s*==\s*\{\"model: primary\"\}", src), (
        "the exact-property leg left the real-board node: the floor says how many task "
        "files declare an engine, only this assertion says they all name the primary")

    parted = census_below_floor_message(
        39, 40, "answers `primary` at least 40 times or beyond")
    assert "39 task files declare a `model:`" in parted, parted
    assert "floors the census at 40" in parted, parted
    assert "'answers `primary` at least 40 times or beyond'" in parted, (
        f"the message must quote the sentence the fleet fell under, got {parted!r}")
    assert "grep -h '^model:' ~/obsidian/autonomy/*.md | wc -l" in parted, parted
    assert "retype" in parted and "no edit" in parted, (
        f"the reader must be told which file to retype and that this assertion is not it: "
        f"{parted!r}")

    # No node that the gate runs may compare a typed census number with the doc. The controls
    # first: the detector has to catch the shape in both spellings, and must NOT catch the
    # parser's own synthetic legs, or the rail would forbid the very tests that replaced the
    # pin — and a rail that only caught the retired exact spelling would be green on vacuity.
    caught = typed_census_on_doc_reads(OFFENDER_SRC)
    assert len(caught) == 1 and "at least 40 times" in caught[0], (
        f"the detector missed a typed floor compared against the doc: {caught}")
    caught_exact = typed_census_on_doc_reads(OFFENDER_SRC_EXACT)
    assert len(caught_exact) == 1 and "40 times" in caught_exact[0], (
        f"the detector lost the retired exact spelling when it widened to the floor: "
        f"{caught_exact}")
    assert typed_census_on_doc_reads(ALLOWED_SRC) == [], (
        "the detector must still allow parsing prose the node wrote itself — the number "
        "is only a stale claim when it is compared with the doc being read")

    src = Path(__file__).read_text(encoding="utf-8")
    offenders: dict[str, list[str]] = {}
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        decs = " ".join(
            ast.get_source_segment(src, d) or "" for d in node.decorator_list)
        if "live_vault" in decs:
            continue
        body = ast.get_source_segment(src, node) or ""
        if hits := typed_census_on_doc_reads(body):
            offenders[node.name] = hits
    assert not offenders, (
        f"a typed census count is compared against the real doc in "
        f"{sorted(offenders)} — nodes the gate runs, since none is marked `live_vault`. "
        f"The count is the fleet's and the fleet grows: §#68's number went stale that way "
        f"twice. Assert the census's SHAPE here and leave the count to "
        f"`test_the_real_board_lints_clean_now_that_the_model_slot_check_is_live`: "
        f"{offenders}")


# ── #2895: DOC_MEMBERSHIP — the linter reads the doc's two membership tables ────
#
# The fleet's own membership was never one of this linter's classes, so a task file could
# land that the doc names nowhere and every check above would pass on it. Four rounds were
# spent on that gap from the wrong end (#1102, #2630/#2631, #2813, #2881): the only node that
# reconciled the two tables against the directory carries `@live_vault`, which the automod gate
# deselects (`scripts/automod/gate.py:641`), so the first thing in a day to read both sides was
# allowed to be a red suite.
#
# These nodes stay with the file's own convention — the linter is a CLI and nothing imports it,
# so every fact here comes out of a subprocess and an exit code. The parser is
# `scripts/autonomy/jobs_membership.py`, the same one both task-file writers reach on a create
# (`tests/test_autonomy_task_membership_at_filing.py`), which is the point of extracting it: the
# linter cannot hold a laxer rule than filing does.
#
# `--jobs-doc` is passed explicitly in every seeded node. Without it the check is SKIPPED for a
# non-default `--autonomy-dir`, by design — see
# `test_without_the_flag_a_fixture_directory_is_not_graded_against_the_shared_doc`.


def _membership_doc(tmp_path: Path, *, functions=(), tiers=()) -> Path:
    """A two-table membership doc in the live doc's shape.

    Which cell the ids sit in is the detail that decides whether this fixture means anything:
    the live rows are `| [Ingest](#ingest-68-30-53) | #68 #30 #53 | … |` and `| Writes durable
    state unattended | #30 #53 … |`, so the ids are in the SECOND cell of both tables and
    `jobs_membership.table_rows` reads `cells[1]`. A fixture that spelled them into cell 1
    parses as a table naming nobody — and "names nobody" is indistinguishable from "in sync" for
    any node that checks only one direction, which is the exact blind spot this class grew in.
    """
    headers = ("| Function | Jobs | What it is for |", "| Tier | Jobs | |")
    parts = []
    for hdr, ids in zip(headers, (functions, tiers)):
        rows = "".join(f"| Group {i} | #{i} | note |\n" for i in ids)
        parts.append(f"{hdr}\n|---|---|---|\n{rows}")
    d = tmp_path / "autonomy-jobs.md"
    d.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return d


def _membership_run(tmp_path, lint_args: "Linter", doc: Path, *extra: str):
    return lint_args.run("--jobs-doc", str(doc), *extra)


def test_doc_membership_names_a_task_file_neither_table_lists(tmp_path):
    """The #2881 shape, from the linter's side: `41-….md` on disk, #41 named nowhere.

    Both table names appear in the line because the fix is one row in EACH, and a warning that
    names one sends a reader to add that one and leave the other wrong — #2881's third instance
    was precisely a job in the functions table and absent from the tier table. Naming the id is
    the other half of the value: "membership drift" without `#41` sends a person to diff the
    whole fleet by hand.
    """
    lint = Linter(tmp_path)
    lint.add("40-x.md", id=40, name="x", status="draft", frequency="daily",
             skill_name="heartbeat")
    lint.add("41-y.md", id=41, name="y", status="draft", frequency="daily",
             skill_name="heartbeat")
    doc = _membership_doc(tmp_path, functions=(40,), tiers=(40,))

    rc, out = _membership_run(tmp_path, lint, doc)
    hits = [ln for ln in out.splitlines() if "DOC_MEMBERSHIP" in ln]
    assert len(hits) == 2, out
    assert all("#41" in ln for ln in hits), hits
    assert any("functions table" in ln for ln in hits) and any(
        "write-authority" in ln for ln in hits), hits
    assert rc == 0, f"a doc that lags the fleet must cost a run nothing, got exit {rc}"


def test_doc_membership_names_a_row_with_no_task_file_behind_it(tmp_path):
    """The opposite direction, which is the one a filing check cannot catch by definition: the
    doc claims a job the fleet does not have (#87 #88 #89, and the three paper digests of #2630).

    The ghost rule discharges an id whose OWN row says Retired, read off the row rather than
    from a typed list — the case that forced the exemption is #36, whose membership row at
    `architecture/autonomy-jobs.md:933` says **Retired 2026-09-23** while its file sits in
    `_archived/`. The negative control below is the half that keeps the exemption honest: prose
    about somebody else's retirement must not discharge anything, or the exemption is an
    allowlist with extra steps.
    """
    lint = Linter(tmp_path)
    lint.add("40-x.md", id=40, name="x", status="draft", frequency="daily",
             skill_name="heartbeat")
    doc = _membership_doc(tmp_path, functions=(40, 41), tiers=(40,))

    rc, out = _membership_run(tmp_path, lint, doc)
    hits = [ln for ln in out.splitlines() if "DOC_MEMBERSHIP" in ln and "#41" in ln]
    assert len(hits) == 1, out
    assert "no task file" in hits[0] and "no Retired row" in hits[0], hits[0]

    # The discharging row's shape is #36's, at `architecture/autonomy-jobs.md:933`:
    # `| #36 | daily | the vault | **Retired 2026-09-23 (#1012)** — … |`. It starts with the ID,
    # which is what `RETIRED_MEMBERSHIP_ROW`'s `^\| #(\d+) \|` anchors on, and it sits outside
    # both membership tables — a row in a table this check reads would also NAME the id, and the
    # test would then be grading a doc that lists #41 rather than one that discharges it.
    discharged = tmp_path / "discharged.md"
    discharged.write_text(
        doc.read_text(encoding="utf-8")
        + "\n| #41 | daily | the vault | **Retired 2026-10-17** — merged into #40 |\n",
        encoding="utf-8")
    assert not [ln for ln in _membership_run(tmp_path, lint, discharged)[1].splitlines()
                if "DOC_MEMBERSHIP" in ln and "#41" in ln], (
        "a row that discharges its own id was still reported as a phantom")

    # The same words, and the same id, in a line that is not #41's own row: not a discharge.
    # Without this the exemption is an allowlist you can buy with any sentence in the doc.
    prose = tmp_path / "prose.md"
    prose.write_text(
        doc.read_text(encoding="utf-8")
        + "\nGroup 41, whose predecessor Retired in 2026, absorbs it.\n",
        encoding="utf-8")
    assert [ln for ln in _membership_run(tmp_path, lint, prose)[1].splitlines()
            if "DOC_MEMBERSHIP" in ln and "#41" in ln], (
        "prose mentioning a retirement discharged an id whose row does not say retired")


def test_doc_membership_is_the_directory_not_a_registration(tmp_path):
    """No allowlist, and no way to file a task quietly.

    This class turned on a hand-typed roster that nobody updated, so the property that decides
    whether the fix holds is this one: a file that appears in the directory is asked about on
    the NEXT run with no edit to any list, in any file, anywhere. Anything that needs
    registering is the same roster under a better name.
    """
    lint = Linter(tmp_path)
    lint.add("40-x.md", id=40, name="x", status="draft", frequency="daily",
             skill_name="heartbeat")
    doc = _membership_doc(tmp_path, functions=(40,), tiers=(40,))
    assert not [ln for ln in _membership_run(tmp_path, lint, doc)[1].splitlines()
                if "DOC_MEMBERSHIP" in ln], "a clean pair reported drift"

    lint.add("41-y.md", id=41, name="y", status="draft", frequency="daily",
             skill_name="heartbeat")
    hits = [ln for ln in _membership_run(tmp_path, lint, doc)[1].splitlines()
            if "DOC_MEMBERSHIP" in ln]
    assert len(hits) == 2 and all("#41" in ln for ln in hits), hits

    # A name that is not `NN-*.md` is not a fleet member, so the fleet's own `_config.md` and
    # any note dropped into the directory cannot manufacture a violation — and, symmetrically,
    # cannot hide one: `_config.md` beside a real unlisted id leaves exactly the two hits.
    (lint.autonomy_dir / "_config.md").write_text("---\ntype: note\n---\n\nx\n",
                                                  encoding="utf-8")
    (lint.autonomy_dir / "scratch.md").write_text("# scratch\n", encoding="utf-8")
    hits = [ln for ln in _membership_run(tmp_path, lint, doc)[1].splitlines()
            if "DOC_MEMBERSHIP" in ln]
    assert len(hits) == 2 and all("#41" in ln for ln in hits), hits


def test_doc_membership_reports_an_unreadable_doc_instead_of_passing(tmp_path):
    """An empty parse and a clean doc must be tellable apart from the output alone.

    `jobs_membership.table_rows` raises rather than returning no rows for exactly this reason,
    and the class surfaces both failure shapes as one reported line: a doc that is not on disk,
    and a doc whose two table headers have been reworded so the tables cannot be located. Both
    say `unchecked`, which no reader can mistake for a pass, and both are warnings, so
    `--strict` fails on them.
    """
    lint = Linter(tmp_path)
    lint.add("40-x.md", id=40, name="x", status="draft", frequency="daily",
             skill_name="heartbeat")

    rc, out = _membership_run(tmp_path, lint, tmp_path / "not-here.md")
    hits = [ln for ln in out.splitlines() if "DOC_MEMBERSHIP" in ln]
    assert len(hits) == 1 and "unchecked" in hits[0], out
    assert rc == 0
    assert _membership_run(tmp_path, lint, tmp_path / "not-here.md", "--strict")[0] == 2, (
        "an unchecked membership check is outside --strict, so it can pass silently")

    headerless = tmp_path / "reworded.md"
    headerless.write_text("# A doc whose tables were reworded\n\nnothing here\n",
                          encoding="utf-8")
    rc, out = _membership_run(tmp_path, lint, headerless)
    hits = [ln for ln in out.splitlines() if "DOC_MEMBERSHIP" in ln]
    assert len(hits) == 1 and "unchecked" in hits[0], out


def test_without_the_flag_a_fixture_directory_is_not_graded_against_the_shared_doc(tmp_path):
    """The scoping rule, pinned because it is the one thing here that could lie.

    With no `--jobs-doc` the class runs only when `--autonomy-dir` is the default fleet
    directory. Two reasons, and the second is the one a round can be burned by: `jobs_membership`
    resolves the doc relative to the tree that is RUNNING, so grading the real fleet against a
    checkout that is a day behind would report every job filed since that checkout as unlisted —
    a measurement of the worktree, not of the fleet. A test passing `--autonomy-dir tmp` is
    asking about one file's fields and has no opinion about a shared doc it did not name.
    """
    lint = Linter(tmp_path)
    lint.add("40-x.md", id=40, name="x", status="draft", frequency="daily",
             skill_name="heartbeat")

    rc, out = lint.run()
    assert "DOC_MEMBERSHIP" not in out, (
        "a redirected fixture directory was graded against a doc the caller never named")
    assert rc == 0
    # And it is not simply dead code: naming the doc turns the same directory into a report.
    doc = _membership_doc(tmp_path, functions=(), tiers=())
    assert any("DOC_MEMBERSHIP" in ln for ln in lint.run("--jobs-doc", str(doc))[1].splitlines())


def test_doc_membership_leaves_every_other_class_and_every_exit_code_alone(tmp_path):
    """The bound the item states outright, measured as exit codes.

    Four facts from this linter's own contract, each re-checked on a directory that ALSO carries
    membership drift, which is the only way to show the new class did not disturb the others: an
    unparseable file is exit 1, `--strict` turns structural warnings into exit 2, a bad
    `depends_on` is still named by file and upstream, and a clean run is exit 0. The one
    intentional difference is that DOC_MEMBERSHIP joins `warnings`, so it is exit 0 by itself and
    exit 2 under `--strict` — the doc is prose, and prose lagging the scheduler has never been a
    reason to fail the thing that dispatches jobs.
    """
    lint = Linter(tmp_path)
    lint.add("40-x.md", id=40, name="x", status="draft", frequency="daily",
             skill_name="heartbeat")
    doc = _membership_doc(tmp_path, functions=(40,), tiers=(40,))

    assert _membership_run(tmp_path, lint, doc)[0] == 0, "a clean case stopped being clean"

    lint.add("41-y.md", id=41, name="y", status="draft", frequency="daily",
             skill_name="heartbeat")
    assert _membership_run(tmp_path, lint, doc)[0] == 0
    assert _membership_run(tmp_path, lint, doc, "--strict")[0] == 2

    # A real structural violation keeps its own class, its own wording and its own code with
    # membership drift present beside it: #815's frequency vocabulary, unchanged.
    lint.add("42-z.md", id=42, name="z", status="up_next", frequency="hourly-on-the-hour",
             skill_name="heartbeat")
    rc, out = _membership_run(tmp_path, lint, doc)
    assert "hourly-on-the-hour" in out and rc == 0, out

    (lint.autonomy_dir / "43-broken.md").write_text(
        "---\ntype: autonomy\nid: [unclosed\n---\n\nx\n", encoding="utf-8")
    rc, out = _membership_run(tmp_path, lint, doc)
    assert rc == 1, f"DOC_MEMBERSHIP changed the unparseable exit code: {rc}\n{out}"
    assert "UNPARSEABLE" in out or "unparseable" in out, out


@pytest.mark.live_vault
def test_the_live_fleet_and_the_live_doc_reconcile_through_the_new_class():
    """What this round can honestly say about the real tree: the class is silent on today's
    fleet, and it is silent because #2881 hand-fixed both tables on 2026-10-17, not because the
    check skipped.

    Marked `@live_vault` for the reason the same file's own reconciling nodes are
    (`tests/test_autonomy_jobs_doc_claims.py:3349`, `:3377`): it reads `~/obsidian/autonomy`,
    which no round under test controls, so the gate deselects it and it runs under a bare
    `pytest tests/` and in the nightly. The seeded nodes above are what the gate sees — which is
    the shape this item exists to install.

    The fleet size is READ off the run and compared against the directory the run names, not
    typed into this file. The first draft asserted `"Scanned 43 task files" in out`, and 43 was
    simply how many files the fleet held on the morning #2881 hand-fixed the two tables
    (`f1cd276e`): the next task filed — WITH its two doc rows just as much as without — reddened
    this node, so it was a maintenance tripwire dressed as a check and could not tell "the fleet
    grew" apart from "the linter stopped counting". Counting the directory behind the scanner is
    the claim that survives a filing: it says the scanner saw the whole fleet, and it stays true
    whatever the fleet is. The `DOC_MEMBERSHIP` silence below is the part about the doc.
    """
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True,
                          timeout=180, cwd=str(ROOT))
    out = proc.stdout + proc.stderr
    scanned = re.search(r"^Scanned (\d+) task files in (.*)$", out, re.MULTILINE)
    assert scanned, f"the run printed no scan line this node can count:\n{out[:600]}"
    printed, scanned_dir = int(scanned.group(1)), Path(scanned.group(2).strip())
    # The same glob the scanner uses (`validate_tasks.py`: `*.md` whose name starts with
    # digits-and-a-dash), so this compares two readings of one rule and not a guess at it.
    on_disk = [p for p in scanned_dir.glob("*.md") if re.match(r"\d+-", p.name)]
    assert printed > 0, f"the directory the linter scanned came back empty: {scanned_dir}"
    assert printed == len(on_disk), (
        f"the linter counted {printed} task files while {len(on_disk)} numbered files sit in "
        f"{scanned_dir}: the count on the report is not the fleet, so the silence below proves "
        f"nothing about a fleet the scanner did not see\n{out[:400]}")
    hits = [ln for ln in out.splitlines() if "DOC_MEMBERSHIP" in ln]
    assert hits == [], "the live fleet and the live doc disagree:\n" + "\n".join(hits)
