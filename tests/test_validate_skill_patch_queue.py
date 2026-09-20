"""Tests for `scripts/validate_skill_patch_queue.py`, the skill-patch queue checker.

The queue (`~/lloyd/_pipeline/skills/proposed/`) is written and read by agents
following prose — `nightly-skill-consolidation` Phase 3.1 writes entries,
`nightly-skills-management` Stage 6 selects on each file's parsed `applied:` key.
Nothing parsed it: `grep -rn "skills/proposed" --include=*.py` over the checkout
returned 0 hits at triage. So from 2026-09-08 to 2026-09-20 four of the seven
queue files were unparseable and every pass still reported selecting on a parsed
key — a consumer that cannot parse its input and does not fail.

The checker is a CLI and nothing imports it in production, so every test here
runs it as a subprocess: the exit code is the deliverable (0 clean, 1 defective
files named, 2 no verdict) and cannot be pinned from inside the process. Each
test writes its own fixture dir — `_pipeline/` is gitignored and any run can
mutate it, so the live queue can pin nothing.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "validate_skill_patch_queue.py"

# Phase 3.1 of `nightly-skill-consolidation` — the shape the queue's own writer is
# told to emit, with its placeholders filled the way a real pass fills them (a
# plain scalar may not *start* with `{`, so `target_skill: {skill-name}` parses
# only once the name is in there). Clause 3 requires the checker to exit 0 over
# exactly this, so writer and checker must agree.
PHASE_3_1_TEMPLATE = """---
type: patch
target_skill: bash-timeout
source_candidates: [candidate-bash-timeout-seq-2-20260919.md]
confidence: 0.9
sessions: 12
generated: 2026-09-20T09:40:00Z
applied: false
applied_by: null
---

# Proposed Patch: bash-timeout
"""

# The shape one of the four broken files actually had: a quoted scalar, then a
# comma, then a `#` comment. YAML reads the comma as a flow-sequence separator
# inside a block mapping, so the whole front matter is one ParserError.
COMMA_BEFORE_COMMENT = """---
type: patch
target_skill: bash-timeout
generated: "2026-09-08T04:40:00Z",  # ← applied: false
applied: false
---

# Proposed Patch: bash-timeout
"""

# Stage 6 step 4's own prohibition: `held_at:` / `held_reason:` appear twice
# apiece in two files the 2026-09-19 pass appended to instead of replacing.
DUPLICATE_HELD_KEYS = """---
type: patch
target_skill: automod-change-own-code
applied: false
held_at: 2026-09-19T04:55:00Z
held_reason: below floor on 2026-09-19
held_at: 2026-09-19T05:23:00Z
held_reason: re-held on 2026-09-19
---

# Proposed Patch: automod-change-own-code
"""

NO_APPLIED = """---
type: patch
target_skill: file-read-error-handling
generated: 2026-09-16T09:40:00Z
---

# Proposed Patch: file-read-error-handling
"""

# A queue entry a real pass leaves behind after applying: `applied: true` plus
# the bookkeeping keys, including a plain scalar carrying em dashes and
# semicolons. Clean, so it must not be reported.
APPLIED_REAL_WORLD = """---
type: patch
target_skill: file-mutation-safety
confidence: 0.9
sessions: 27
generated: 2026-09-11T09:00:00Z
applied: true
landed: true
blocked_on: none — cleared by task-83 on 2026-09-20; the retry is no longer the right action
---

# Applied Patch: the Edit read-refusal class
"""

NO_FRONT_MATTER = "# Proposed Patch: nothing\n\nbody text, no `---` fence anywhere.\n"


class Queue:
    """A fixture queue dir with one `add()` call per queue file."""

    def __init__(self, tmp_path: Path):
        self.dir = tmp_path / "proposed"
        self.dir.mkdir()

    def add(self, filename: str, text: str) -> Path:
        path = self.dir / filename
        path.write_text(text, encoding="utf-8")
        return path

    def run(self, *args: str) -> tuple[int, str]:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--dir", str(self.dir), *args],
            capture_output=True, text=True, timeout=60,
        )
        return proc.returncode, proc.stdout + proc.stderr


def reported_defect_files(out: str) -> set[str]:
    """The file names named on `DEFECT` lines — what the operator has to act on."""
    return {Path(line.split(" ", 1)[1].split(": ", 1)[0]).name
            for line in out.splitlines() if line.startswith("DEFECT ")}


# ── Clause 1: front matter that does not parse stops the stage ────────────────

def test_a_quoted_scalar_followed_by_a_comma_before_a_comment_exits_nonzero(tmp_path):
    """`generated: "…Z",  # ← applied: false` — the shape that hid 4 files for 11 days."""
    queue = Queue(tmp_path)
    queue.add("patch-bash-timeout-2026-09-08.md", COMMA_BEFORE_COMMENT)
    queue.add("patch-clean-2026-09-16.md", PHASE_3_1_TEMPLATE)

    code, out = queue.run()

    assert code == 1, out
    assert "patch-bash-timeout-2026-09-08.md" in out
    assert "does not parse" in out
    assert str(queue.dir / "patch-bash-timeout-2026-09-08.md") in out, \
        "the line names the file by path, not just its stem"
    assert "patch-clean-2026-09-16.md" not in out, "only the broken file is named"


def test_a_parse_failure_reports_the_parsers_own_diagnosis(tmp_path):
    """The message must be actionable — PyYAML's 'expected <block end>, but found'."""
    queue = Queue(tmp_path)
    queue.add("patch-broken-2026-09-08.md", COMMA_BEFORE_COMMENT)

    out = queue.run()[1]

    assert "expected <block end>, but found ','" in out, out
    assert "front-matter line" in out, "the line number is how a hand edit gets found"


def test_a_file_with_no_front_matter_block_is_a_defect_too(tmp_path):
    """A body-only file parses to nothing at all, which reads as 'not applied' forever."""
    queue = Queue(tmp_path)
    queue.add("patch-fenceless-2026-09-12.md", NO_FRONT_MATTER)

    code, out = queue.run()

    assert code == 1, out
    assert "patch-fenceless-2026-09-12.md" in out
    assert "no front-matter block" in out


# ── Clause 2: a repeated key, which safe_load resolves silently ───────────────

def test_a_repeated_held_at_key_exits_nonzero_and_names_the_file(tmp_path):
    """Two `held_at:` lines is one front matter, not two verdicts."""
    queue = Queue(tmp_path)
    queue.add("patch-automod-change-own-code-2026-09-17.md", DUPLICATE_HELD_KEYS)
    queue.add("patch-clean-2026-09-16.md", PHASE_3_1_TEMPLATE)

    code, out = queue.run()

    assert code == 1, out
    assert "patch-automod-change-own-code-2026-09-17.md" in out
    assert "duplicate key 'held_at'" in out
    assert "duplicate key 'held_reason'" in out


def test_the_duplicate_is_detected_where_yaml_safe_load_sees_nothing(tmp_path):
    """The checker cannot be `safe_load` + a catch: last-wins means no exception exists.

    Safe-load the same bytes and it answers one `held_at`, whichever it chose;
    the checker's verdict has to come from walking the mapping node's keys.
    """
    from scripts import validate_skill_patch_queue as vq

    fm = DUPLICATE_HELD_KEYS[3:DUPLICATE_HELD_KEYS.find("\n---", 3)]
    assert sum(1 for line in fm.splitlines() if line.startswith("held_at: ")) == 2
    merged = yaml.safe_load(fm)
    assert list(merged).count("held_at") == 1, \
        "safe_load merges the repetition into one key, so there is no exception to catch"
    assert merged["held_at"] == datetime(2026, 9, 19, 5, 23, tzinfo=timezone.utc), \
        "and the survivor is whichever the parser chose, not the verdict that is current"

    duplicates = vq.duplicate_key_errors(fm)
    assert any("held_at" in d for d in duplicates), duplicates
    assert any("held_reason" in d for d in duplicates), duplicates


def test_a_repeated_key_nested_in_the_front_matter_is_also_detected(tmp_path):
    """The rule is one mapping, one key — not one top level, one key."""
    queue = Queue(tmp_path)
    queue.add("patch-nested-2026-09-18.md", """---
type: patch
applied: false
reviewers:
  - name: task-83
    note: first
    note: second
---

# body
""")

    code, out = queue.run()

    assert code == 1, out
    assert "patch-nested-2026-09-18.md" in out
    assert "duplicate key 'note'" in out


def test_the_same_key_in_two_distinct_mappings_is_not_a_repetition(tmp_path):
    """The rule is one mapping, one key — so two mappings may repeat between them.

    Soundness half of the nested rule: a checker that rolled keys up across nesting
    would refuse a queue for a shape it is allowed to hold, and the runbook's route
    around a too-strict check is to stop running the check.
    """
    queue = Queue(tmp_path)
    queue.add("patch-siblings-2026-09-18.md", """---
type: patch
applied: false
before:
  held_at: 2026-09-19T04:55:00Z
after:
  held_at: 2026-09-19T05:23:00Z
---

# body
""")

    code, out = queue.run()

    assert code == 0, out


# ── Clause 3: no `applied:` key fails; the writer's template passes ───────────

def test_a_parsed_mapping_with_no_applied_key_exits_nonzero_and_names_the_file(tmp_path):
    """Stage 6 treats a missing key as `false` and writes it — the checker names it."""
    queue = Queue(tmp_path)
    queue.add("patch-file-read-error-handling-2026-09-16.md", NO_APPLIED)

    code, out = queue.run()

    assert code == 1, out
    assert "patch-file-read-error-handling-2026-09-16.md" in out
    assert "no `applied:` key" in out


def test_a_queue_holding_only_the_phase_3_1_template_exits_zero(tmp_path):
    """The checker must pass its own writer's output, or it is noise that gets ignored."""
    queue = Queue(tmp_path)
    queue.add("patch-foo-2026-09-20.md", PHASE_3_1_TEMPLATE)

    code, out = queue.run()

    assert code == 0, out
    assert "files: 1" in out
    assert "defects: 0" in out


def test_a_fully_applied_real_world_entry_exits_zero(tmp_path):
    """`applied: true` with landed/blocked_on bookkeeping is clean, not a defect."""
    queue = Queue(tmp_path)
    queue.add("patch-file-mutation-safety-2026-09-11.md", APPLIED_REAL_WORLD)

    code, out = queue.run()

    assert code == 0, out
    assert "DEFECT" not in out


def test_an_empty_queue_is_a_clean_verdict_with_its_denominator(tmp_path):
    """Drained queue: exit 0, but the count still prints — a bare 'nothing to do' is not."""
    queue = Queue(tmp_path)

    code, out = queue.run()

    assert code == 0, out
    assert "files: 0" in out


def test_a_missing_queue_dir_is_no_verdict(tmp_path):
    """Exit 2, not 0: a directory that isn't there cannot pass a check over its contents."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(tmp_path / "no-queue-here")],
        capture_output=True, text=True, timeout=60,
    )

    assert proc.returncode == 2
    assert "QUEUE DIR MISSING" in proc.stdout + proc.stderr


# ── The seam: the checker's verdict against what Stage 6 actually does ────────
#
# Stage 6's selection is `yaml.safe_load(front matter)` and then a read of
# `applied:`. The checker runs before it, so a false alarm would make the runbook
# route around the checker and a missed file restores the silent skip that lost
# 11 days. The contract is therefore one-directional, and these tests say which
# way: every file the checker passes must be one safe_load can select on (no
# false alarms), and the duplicate-key class is the file safe_load *does* select
# on and the checker must still refuse — the whole reason the checker does not
# just call safe_load and catch.

ALL_SHAPES = {
    "patch-template-2026-09-20.md": PHASE_3_1_TEMPLATE,
    "patch-applied-2026-09-11.md": APPLIED_REAL_WORLD,
    "patch-broken-2026-09-08.md": COMMA_BEFORE_COMMENT,
    "patch-duplicate-2026-09-19.md": DUPLICATE_HELD_KEYS,
    "patch-noapplied-2026-09-16.md": NO_APPLIED,
    "patch-fenceless-2026-09-12.md": NO_FRONT_MATTER,
}


def consumer_selects(text: str) -> bool:
    """Stage 6's own selection: parse the leading block, read `applied:` out of it."""
    if not text.startswith("---"):
        return False
    block = text[3:text.find("\n---", 3)]
    if block == text[3:]:  # no closing fence
        return False
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        return False
    return isinstance(data, dict) and "applied" in data


@pytest.mark.parametrize("filename", sorted(ALL_SHAPES))
def test_a_file_the_checker_passes_is_one_stage_6_can_select_on(tmp_path, filename):
    """Soundness: exit 0 over a dir holding this file ⇒ safe_load yields `applied:`."""
    queue = Queue(tmp_path)
    queue.add(filename, ALL_SHAPES[filename])

    code, out = queue.run()

    if code == 0:
        assert consumer_selects(ALL_SHAPES[filename]), \
            f"a clean verdict on {filename} while the consumer cannot read it\n{out}"
    else:
        assert filename in reported_defect_files(out), out


def test_one_dir_of_every_shape_names_exactly_the_four_unusable_files(tmp_path):
    """Denominator plus the offending names, in one report — no silent skip left."""
    queue = Queue(tmp_path)
    for filename, text in ALL_SHAPES.items():
        queue.add(filename, text)

    code, out = queue.run()

    assert code == 1, out
    assert "files: 6" in out
    assert "defects: 4" in out
    assert reported_defect_files(out) == {
        "patch-broken-2026-09-08.md",
        "patch-duplicate-2026-09-19.md",
        "patch-noapplied-2026-09-16.md",
        "patch-fenceless-2026-09-12.md",
    }, out


def test_the_duplicate_key_file_is_one_stage_6_still_selects_on(tmp_path):
    """The added strictness: safe_load finds `applied: false` in it and moves on.

    That is the state the 2026-09-19 pass left behind — a decision recorded
    twice, selected on as if recorded once. Only the node-walking detector stops
    it, so this is the pair of assertions that says the checker is not a
    `safe_load` wrapper.
    """
    queue = Queue(tmp_path)
    queue.add("patch-duplicate-2026-09-19.md", DUPLICATE_HELD_KEYS)

    assert consumer_selects(DUPLICATE_HELD_KEYS), \
        "the consumer must be able to select on it — that is the defect, not the escape"
    code, out = queue.run()
    assert code == 1, out
    assert "duplicate key 'held_at'" in out


# ── Clauses 4 & 5: the two runbooks that own the queue must require the checker ─
#
# The checker has no production importer — its only caller is an agent following a
# runbook — so the prose IS the integration point, and a step nobody runs is the
# state that lost 11 days. These read the live vault (no round under test controls
# it), hence `live_vault`, which `gate.py`'s `-m "not live_vault"` deselects; the
# run that cites them must say so.

MGMT = Path.home() / "obsidian" / "skills" / "nightly-skills-management" / "SKILL.md"
CONSOLIDATION = Path.home() / "obsidian" / "skills" / "nightly-skill-consolidation" / "SKILL.md"

INVOCATION = ("~/lloyd/.venvs/lloyd/bin/python "
              "~/lloyd/scripts/validate_skill_patch_queue.py")


def _plain(text: str) -> str:
    """Markdown emphasis stripped and whitespace collapsed.

    Runbook prose is bolded and re-wrapped on every edit, so `**stops this\n   stage**`
    is the same instruction as `stops this stage` and a raw substring check would fail
    on formatting. Every phrase asserted below is checked against this form; ordering
    assertions use it too, so a re-wrap cannot invent or erase a sequence.
    """
    return " ".join(text.replace("**", "").split())


def _section(path: Path, start: str, end: str) -> str:
    body = path.read_text(encoding="utf-8")
    return _plain(body[body.index(start):body.index(end, body.index(start))])


@pytest.fixture(scope="module")
def stage_6() -> str:
    assert MGMT.exists(), f"{MGMT} is absent: the runbook clause 4 binds is not here"
    # Stage 6 is the runbook's last stage; `## Output` is what bounds it.
    return _section(MGMT, "## Stage 6: Drain the Skill Patch Queue", "\n## Output")


@pytest.fixture(scope="module")
def phase_4() -> str:
    assert CONSOLIDATION.exists(), \
        f"{CONSOLIDATION} is absent: the runbook clause 5 binds is not here"
    return _section(CONSOLIDATION, "## Phase 4: Hand Proposals to #83", "## Phase 5:")


@pytest.mark.live_vault
def test_stage_6_runs_the_checker_before_it_selects(stage_6):
    """Clause 4: the gate precedes selection, so a bad file stops the stage."""
    assert "validate_skill_patch_queue.py" in stage_6
    assert INVOCATION in stage_6

    # Ordering is the whole clause: a check after selection cannot stop it.
    assert stage_6.index("validate_skill_patch_queue.py") < stage_6.index("Then select:"), stage_6
    assert stage_6.startswith("## Stage 6: Drain the Skill Patch Queue")
    assert "1. Validate the queue, then select." in stage_6, \
        "the checker is step 1 itself, not a note beside a selection step"


@pytest.mark.live_vault
def test_stage_6_requires_reporting_the_exit_status_and_named_files(stage_6):
    """The report is the only audit trail: `/_pipeline/` is gitignored."""
    assert "exit status" in stage_6
    assert "DEFECT" in stage_6, "names the files the operator has to repair"
    assert "stops this stage" in stage_6, "non-zero is a stop, not a warning"
    assert "failed" in stage_6, "an unreparable file reports a failed stage, never a partial queue"


@pytest.mark.live_vault
def test_phase_4_requires_the_checker_on_the_run_own_writes(phase_4):
    """Clause 5: the writer is the only party that can fix its own malformed file cheaply."""
    assert "validate_skill_patch_queue.py" in phase_4
    assert INVOCATION in phase_4
    assert "exit status" in phase_4
    assert "repair" in phase_4, "non-zero routes to fixing the file, not to reporting it anyway"
    assert phase_4.index("validate_skill_patch_queue.py") < phase_4.index(
        "2. Mark the source candidates"), "the check precedes the hand-off bookkeeping"


@pytest.mark.live_vault
def test_the_two_runbooks_name_one_command(stage_6, phase_4):
    """Two files, one rule — the divergence class this pair has already been caught in."""
    assert stage_6.count(INVOCATION) == 1, stage_6
    assert phase_4.count(INVOCATION) == 1, phase_4
    assert "proposed/" in stage_6 and "proposed/" in phase_4
