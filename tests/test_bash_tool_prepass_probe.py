"""The Bash-tool prepass oracle (#3215): `scripts/bash_tool_prepass_probe.py`.

Four behaviours are graded here, and the first two are the reason the script exists.

* **The end-to-end line, from a real `_bash` spawn.** The `Bash/logic` key carries 29
  rows in `~/lloyd-data/_pipeline/skills/reviews/verdicts.jsonl` (518 rows total, all
  29 `rejected_unverifiable`, re-counted 2026-10-24 by filtering that file on
  `pattern_key == "Bash/logic"`), 27 of them opening `RE-CONFIRMED (step 0 SKIP)`.
  Every one of them re-executed an `evidence_cmd` that greps `candidate-bash-logic-*.md`
  files — a falsifier that runs no braced reference, so it can never contradict the
  "no protocol failure exists" reason it re-confirms. This test runs one, through
  `agent_mcp.builtin_bash._bash`, and pins the reading this box gives: `prepass=fired`
  at rc 0. It runs the shipped script as a subprocess from an unrelated cwd, because
  the oracle leg is a command another job pastes into `skill_verdicts.py record
  --evidence-cmd`, not a function this process can call.
* **That `absent` is printable.** An oracle that can only say `fired` is the same
  instrument as the grep it replaces. The counter-example is pinned against a
  scripted `_bash` rather than waited for: no box can be made to lose the pass on
  demand. It is asserted to read `absent` and NOT `fired`, so a refactor cannot
  quietly collapse the two states.
* **That the reference is never spelled.** A dollar-brace literal anywhere in the
  probe's own bytes is consumed by the pass being measured whenever the file is
  authored through the Bash tool, leaving a probe asserting about a string that never
  reached disk. Banned as an assertion, and the runtime-built value is pinned to the
  braced array subscript the note reproduces.
* **That a probe which produced no reading cannot be filed as a reading.** `_bash`
  answering with one of its JSON errors, a shell killed by a signal, and a command
  whose positive control did not run all exit 2 with an empty stdout. Silent
  consumption is the opposite case and must read `fired`: the note measures a name the
  tool's environment happens to define being substituted with nothing printed at all,
  so "no warning line" is not evidence of no pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import bash_tool_prepass_probe as probe
from scripts.bash_tool_prepass_probe import (ARR_NAME, NAMES_NONE, NAMES_SILENT,
                                             SENTINEL, LINE_RE, MARKER, evaluate,
                                             format_reading, parse_names, probe_command)

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "bash_tool_prepass_probe.py"

# What `_bash` returns on this box, transcribed from the 2026-10-24 measurement: the
# emitter's line first (both of its classes print before the command's first line of
# output), then the shell's own line — braced expansion empty, unbraced control
# expanded.
FIRED_TEXT = ("Invalid environment variable name evaluates to an empty string: arr[0]\n"
              f"{MARKER}=[] [{SENTINEL}]\n")
# What it would return with no pass standing between `_bash` and the shell: bash owns
# the array and expands the braced form itself.
ABSENT_TEXT = f"{MARKER}=[{SENTINEL}] [{SENTINEL}]\n"
# The note's real two-name aggregate, whose `", "` spacing a naive `names=` would
# split into two tokens and so break the printed line's own format.
AGGREGATE_TEXT = ("Invalid environment variable name evaluates to an empty string: "
                  f"PIPESTATUS[0], arr[0]\n{MARKER}=[] [{SENTINEL}]\n")
SILENT_TEXT = f"{MARKER}=[] [{SENTINEL}]\n"
DOLLAR_BRACE = chr(36) + "{"


def _stub(text: str):
    """An awaitable stand-in for `_bash`, so `main()`'s transitions are testable."""
    async def _fake(args):
        return text
    return _fake


def _run_probe(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PROBE_PATH)], cwd=str(cwd),
                          capture_output=True, text=True, timeout=180)


def test_probe_line_is_one_fired_line_from_any_cwd(tmp_path: Path) -> None:
    """Clauses 1 and 2 together: one line, in the stored shape, fired at rc 0 here.

    `tmp_path` is the "any cwd" half. The script puts its own repo root on `sys.path`
    rather than inheriting one, and `_bash` takes its start directory from the session
    record rather than from a `cd` in the command string, so a job running it from a
    scratch directory gets the same reading as one running it from the checkout. The
    worktree anchor warning `app.paths` writes is on stderr and stays out of the
    assertion on purpose: stdout is the artifact.
    """
    result = _run_probe(tmp_path)
    assert result.returncode == 0, f"probe exited {result.returncode}: {result.stderr}"
    lines = result.stdout.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    assert LINE_RE.fullmatch(lines[0]), f"line {lines[0]!r} is outside the pinned format"
    assert lines[0].startswith("prepass=fired"), (
        f"this box resolves braced references before the shell sees them, so the oracle "
        f"must print fired; got {lines[0]!r} (stderr: {result.stderr.strip()!r})")
    assert lines[0].endswith(" rc=0"), (
        "the ledger keeps rc-0 falsifiers, so the probe's own command must exit 0")


def test_absent_is_printable_and_is_not_fired() -> None:
    """The leg the stored verdict never had: evidence that can hold a counter-example.

    Scripted rather than waited for, and the whole point of the item: a falsifier
    whose only negative state is "the command broke" is the defect #3215 is fixing.
    """
    reading = evaluate(ABSENT_TEXT)
    assert reading.state == "absent", reading
    assert reading.rc == 0
    assert format_reading(reading) == f"prepass=absent names={NAMES_NONE} rc=0"
    assert LINE_RE.fullmatch(format_reading(reading))


def test_fired_reading_carries_the_name_the_pass_printed() -> None:
    reading = evaluate(FIRED_TEXT)
    assert reading.state == "fired", reading
    assert reading.names == (ARR_NAME,)
    assert format_reading(reading) == f"prepass=fired names={ARR_NAME} rc=0"


def test_aggregate_names_stay_one_whitespace_free_token() -> None:
    """`names=\\S+` has to survive the emitter's own `", "`-joined aggregate."""
    assert parse_names(AGGREGATE_TEXT) == ("PIPESTATUS[0]", ARR_NAME)
    line = format_reading(evaluate(AGGREGATE_TEXT))
    assert line == f"prepass=fired names=PIPESTATUS[0],{ARR_NAME} rc=0"
    assert LINE_RE.fullmatch(line), line


def test_silent_consumption_reads_fired_not_absent() -> None:
    """No warning line is not evidence that no pass ran.

    The note measures a name the tool's own environment defines being substituted with
    nothing printed at all. Here the marker is empty while bash's control expanded,
    which bash alone cannot do to an array the same command declared.
    """
    reading = evaluate(SILENT_TEXT)
    assert reading.state == "fired", reading
    assert format_reading(reading) == f"prepass=fired names={NAMES_SILENT} rc=0"


def test_rc_is_read_off_the_tools_suffix_not_from_quoted_output() -> None:
    """A command that *prints* an exit-code-looking tail is not an exited command."""
    quoted = ("Invalid environment variable name evaluates to an empty string: arr[0]\n"
              f"{MARKER}=[] [{SENTINEL}] which quoted [exit code: 7] mid-line\n")
    assert evaluate(quoted).rc == 0
    assert evaluate(FIRED_TEXT + "[exit code: 7]").rc == 7


@pytest.mark.parametrize("bash_text,why", [
    ('{"error": "failed to spawn shell: [Errno 2] no such file or directory"}',
     "the no-shell state"),
    ("(no output)\n[exit code: 1]", "the no-marker state"),
    (f"{MARKER}=[] [{SENTINEL}]\n[exit code: -9]", "the signal-kill state"),
    (f"{MARKER}=[]\n", "the lost-control state"),
], ids=["json-error", "no-marker", "signal-kill", "control-missing"])
def test_no_reading_exits_two_and_prints_no_line(bash_text: str, why: str,
                                                 monkeypatch, capsys) -> None:
    """A broken probe must never be fileable as `absent` — that is the false green.

    Driven through `main()` with `_bash` stubbed, because these are the states a
    healthy box will not produce on command.
    """
    import agent_mcp.builtin_bash as builtin_bash

    monkeypatch.setattr(builtin_bash, "_bash", _stub(bash_text))
    assert probe.main() == probe.NO_READING_EXIT, why
    captured = capsys.readouterr()
    assert captured.out == "", f"a no-reading run prints nothing on stdout: {captured.out!r}"
    assert captured.err.strip(), f"{why}: the reason belongs on stderr"


def test_main_prints_the_line_and_exits_zero_whichever_way_it_reads(monkeypatch,
                                                                    capsys) -> None:
    """Both readings exit 0 on purpose: this is a report, not a guard.

    The stored falsifier has to stay runnable whichever way the mechanism goes, and
    `record` keys its refusal off the command's rc (#2052) — an `absent` that exited
    non-zero would be recorded as a broken ledger row precisely on the day the
    mechanism it measures stopped existing.
    """
    import agent_mcp.builtin_bash as builtin_bash

    monkeypatch.setattr(builtin_bash, "_bash", _stub(FIRED_TEXT))
    assert probe.main() == 0
    assert capsys.readouterr().out == f"prepass=fired names={ARR_NAME} rc=0\n"

    monkeypatch.setattr(builtin_bash, "_bash", _stub(ABSENT_TEXT))
    assert probe.main() == 0
    assert capsys.readouterr().out == f"prepass=absent names={NAMES_NONE} rc=0\n"


def test_the_reference_is_built_at_runtime_and_never_spelled() -> None:
    """A spelled literal is eaten before the file lands, so the ban is the test."""
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert probe.BRACED_REF == chr(36) + "{arr[0]}", (
        "the oracle must reproduce the failing form the note measures: a braced array "
        "subscript, which the pass cannot parse and therefore cannot expand")
    assert DOLLAR_BRACE not in source, (
        "the probe's own bytes carry a dollar-brace literal, which the substitution "
        "pass consumes the moment the file is authored through the Bash tool")


def test_the_measurement_is_taken_at_the_tool_boundary_not_at_a_shell() -> None:
    """A bare `bash -c` measures bash, and bash was never the suspect (`/bin/sh -> bash`).

    So the oracle reaches the pass through `agent_mcp.builtin_bash._bash` and spawns no
    shell of its own — otherwise a future edit could "fix" a red test by dropping the
    very layer under measurement, which is how the grep-in-the-ledger got written.
    """
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert "from agent_mcp.builtin_bash import _bash" in source
    assert '_bash({"command": probe_command()})' in source
    assert "import subprocess" not in source
    assert "os.system" not in source
    assert "create_subprocess" not in source


def test_command_carries_its_own_positive_control() -> None:
    """Without the unbraced leg, an empty `[]` is also just an unset array."""
    command = probe_command()
    assert f"arr=({SENTINEL})" in command, command
    assert probe.UNBRACED_REF in command, command
    assert probe.BRACED_REF in command, command
    assert f"{MARKER}=[" in command, command
