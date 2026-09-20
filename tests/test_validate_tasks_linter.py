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

import subprocess
import sys
from pathlib import Path

import pytest

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
    """`_get_model_env` matches a key then any alias; both must pass the linter."""
    lint = Linter(tmp_path)
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
    import autonomy
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
    import autonomy
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
