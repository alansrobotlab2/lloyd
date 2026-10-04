"""#2162: the opt-in forced-preemption step the flash-next arm route calls.

`agent-services/bin/flash-next-preempt-step.sh` is a separate executable for one
reason: the assertions are about what ONE CHILD PROCESS IS HANDED, and the arm
route that calls it spends four minutes restarting the primary to get anywhere
near them. So the nodes below run the step itself against a stub tree — a fake
`.venvs/lloyd/bin/python` that records its argv and replays a canned probe
outcome — and check the route's *position* in `flash-next-run-arm.sh` as text,
which is the whole of what a shell sequence can prove about it.

The behaviours, in the order the clauses state them: opt-in both halves; called
once, after the boot guard and the #1625 canary verdict persistence and before
the `SKIP_BENCH=1` exit; the driver's idle refusal reported as one non-fatal
note with the step exiting 0; prompt length as the only size knob; and a
not-reached run as a stated result rather than an error.

What is deliberately NOT here: any run of the real probe, the real engine or the
real arm script. `PREEMPT_ARM` is a load driver — 8 concurrent prompts — and a
test that armed it would be the busy engine the probe's guard refuses.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STEP_REL = "agent-services/bin/flash-next-preempt-step.sh"
ARM_REL = "agent-services/bin/flash-next-run-arm.sh"
PROBE_REL = "eval/engine_output_probe.py"

#: The whole of what the arm route adds: one line, handing the step nothing but
#: the arm's label. Deliberately WITHOUT the interpreter, so the node below has to
#: read what runs it rather than match a string holding its own answer. It must
#: name `bash`: the arm route that makes this call is itself mode 644 in git and is
#: run through bash (tests/test_flash_next_launcher.py invokes it exactly that way),
#: and an interpreter-prefixed call never asks what the exec bit on the target
#: says. That independence is worth having because the caller's `||` note absorbs
#: any status the SHELL returns for a failed exec — 126, 127 — and a note is also
#: what a step that ran and did not cross leaves behind, so the two would be
#: indistinguishable. Named as a constant because the arm's prose mentions the
#: step's path too, and counting occurrences in prose would grade the comment.
CALL = '"$ROOT/agent-services/bin/flash-next-preempt-step.sh" "$LABEL"'

#: The `||` the call has to travel with. Proved by behaviour rather than here:
#: tests/test_flash_next_launcher.py runs the arm script around a fake tree with
#: no bins in it, so an unguarded call there is rc 127, and five of its
#: `test_the_canary_*` nodes read that rc as the arm's verdict. This file holds
#: the word for what that rc would mean; they hold the fact.
GUARD = '|| echo "preemption load: step did not run (rc=$?); non-fatal'

#: The probe's own default `--load-prompt-words`, which the step copies rather
#: than re-states as its own number. Pinned against the parser below.
PROBE_DEFAULT_WORDS = "20000"

#: The one line `preempt` prints on a run that completed, from main()'s print at
#: eval/engine_output_probe.py:809-811. The step parses fields out of exactly
#: this shape, so a stub that invented its own wording would test a parser no
#: probe ever feeds.
NOT_REACHED = ("wrote /tmp/x/preempt_20261004T080000+0000_arm-t1.json: "
               "preemptions 0.0 -> 0.0, preemptions_reached=false, peak kv 0.157")
REACHED = ("wrote /tmp/x/preempt_20261004T080000+0000_arm-t1.json: "
           "preemptions 0.0 -> 1.0, preemptions_reached=true, peak kv 0.982")
#: The idle guard's refusal, worded exactly as the driver raises and prints it:
#: `raise ProbeRefused(f"engine busy: num_requests_running={running:g}, ...")` at
#: eval/engine_output_probe.py:639-641, rendered by the handler at :815-816 as
#: `engine_output_probe: cannot decide — <that>`. The step quotes the child's own
#: line into its note rather than re-describing it, so a stub that invented its
#: own wording would test a sentence no probe ever prints.
BUSY_REFUSAL = ("engine_output_probe: cannot decide — engine busy: "
                "num_requests_running=3, num_requests_waiting=1; refusing to load it")

#: The variables bash itself puts into a child's exported environment, so the
#: assertion below can tell them from ones the step exported. Measured on this
#: box, not assumed: the difference between the exported keys the stub sees and
#: the ones `_run` passed in is exactly these three — `_` (bash's
#: last-argument marker), `SHLVL` and `PWD` (this file runs the step with
#: `cwd=ROOT`, and a nested bash exports both for its children). Three names,
#: fixed and named: an assertion against them still says something about
#: everything else the child can see.
BASH_OWN_ENV_KEYS = {"_", "PWD", "SHLVL"}

#: Everything the step is allowed to hand the probe. A flag added here is a new
#: route to the driver's idle check or to its load shape, and clause 3 is the
#: clause that says neither exists.
ALLOWED_PROBE_ARGS = {"preempt", "--out-dir", "--label", "--load-prompt-words"}


def _stub_root(tmp_path: Path, *, stdout: str = "", stderr: str = "", rc: int = 0) -> Path:
    """A tree whose `.venvs/lloyd/bin/python` records its argv and replays a result.

    Nothing under this root is real, so the step's ROOT seam is what keeps every
    node below from launching a load at the box's own engine.

    The tree is `tmp_path/tree` and the arm's runtime root is `tmp_path/data`
    (set by `_run`), deliberately disjoint: an assertion that the record's
    directory is not under the repo is only worth making if the two can be told
    apart, which they cannot be when both are `tmp_path`.
    """
    root = tmp_path / "tree"
    bindir = root / ".venvs" / "lloyd" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / PROBE_REL).write_text("# stub: the step passes a path, never imports it\n")
    (bindir / "stdout.txt").write_text(stdout)
    (bindir / "stderr.txt").write_text(stderr)
    (bindir / "rc").write_text(str(rc))
    stub = bindir / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'here="$(cd "$(dirname "$0")" && pwd)"\n'
        '{ echo "@call $0"; printf \'%s\\n\' "$@"; '
        'echo "@env $(compgen -e | sort | tr \'\\n\' \' \')"; } >> "${ARM_CALLS:?}"\n'
        '[ -s "$here/stdout.txt" ] && cat "$here/stdout.txt"\n'
        '[ -s "$here/stderr.txt" ] && cat "$here/stderr.txt" >&2\n'
        '[ -s "$here/rc" ] && exit "$(cat "$here/rc")"\n'
        "exit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    return root


def _run(root: Path, tmp_path: Path, *, preempt: str | None = None,
         words: str | None = None, out_dir: str | None = None) -> subprocess.CompletedProcess:
    """Run the real step against `root`, with only these knobs exported."""
    calls = tmp_path / "calls.txt"
    calls.write_text("")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "ARM_TEST_ROOT": str(root), "ARM_CALLS": str(calls),
           "LLOYD_DATA": str(tmp_path / "data")}
    if preempt is not None:
        env["PREEMPT_ARM"] = preempt
    if words is not None:
        env["PREEMPT_LOAD_PROMPT_WORDS"] = words
    if out_dir is not None:
        env["PREEMPT_DIR"] = out_dir
    proc = subprocess.run(["bash", str(ROOT / STEP_REL), "t1"], cwd=str(ROOT),
                          env=env, capture_output=True, text=True, timeout=60)
    proc.arm_calls = calls.read_text()          # type: ignore[attr-defined]
    proc.arm_env = env                          # type: ignore[attr-defined]
    proc.arm_env_keys = set(env)                # type: ignore[attr-defined]
    return proc


def _call_lines(proc) -> list[list[str]]:
    """One list per probe invocation: `[<interpreter>, <argv...>]`.

    The stub writes `@call $0` as a header line and one line per argument, so
    the interpreter the step chose is argv[0] and not a fact the test has to
    take on faith.
    """
    calls: list[list[str]] = []
    for line in proc.arm_calls.splitlines():
        if line.startswith("@call"):
            head = line.split(maxsplit=1)
            assert len(head) == 2, f"the stub recorded no $0: {line!r}"
            calls.append([head[1]])
        elif line.startswith("@env "):
            continue                      # the child's env keys, read by _env_keys
        elif calls:
            calls[-1].append(line)
    return calls


def _env_keys(proc) -> set[str]:
    """Names exported into the probe's environment, as the child itself saw them.

    Clause 3's `or env` half cannot be read off argv: a step that exported
    something past the arm's own knobs would look clean to an argv assertion. The
    stub is bash, so `compgen -e` lists exactly what the child had exported.
    """
    for line in proc.arm_calls.splitlines():
        if line.startswith("@env "):
            return set(line[5:].split())
    return set()


def _argv(proc) -> list[str]:
    """The argv of the single recorded call, interpreter path included."""
    calls = _call_lines(proc)
    assert len(calls) == 1, f"expected exactly one probe call, got {len(calls)}: {proc.arm_calls!r}"
    return calls[0]


def _notes(proc) -> list[str]:
    """The outcome lines: the step prints exactly one, whatever happened."""
    return [ln for ln in proc.stdout.splitlines() if ln.startswith("preemption load:")]


# ── clause 1: one preempt call through the canary's interpreter, or none ─────

def test_preempt_arm_drives_exactly_one_preempt_call_via_the_venv_python(tmp_path):
    root = _stub_root(tmp_path, stdout=NOT_REACHED)
    proc = _run(root, tmp_path, preempt="1", out_dir=str(tmp_path / "rec"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    argv = _argv(proc)
    assert argv[0] == str(root / ".venvs" / "lloyd" / "bin" / "python"), \
        "the step must use the interpreter under $ROOT, the one the canary block uses"
    assert argv[1:] == [str(root / PROBE_REL), "preempt", "--out-dir", str(tmp_path / "rec"),
                        "--label", "arm-t1", "--load-prompt-words", PROBE_DEFAULT_WORDS]


@pytest.mark.parametrize("preempt", [None, "0"])
def test_the_step_is_inert_when_preempt_arm_is_not_1(tmp_path, preempt):
    root = _stub_root(tmp_path, stdout=NOT_REACHED)
    proc = _run(root, tmp_path, preempt=preempt)
    assert proc.returncode == 0
    assert _call_lines(proc) == [], "an unarmed arm must issue no probe call at all"
    assert _notes(proc) == [], "and no outcome to report, since the route calls it every arm"


# ── clause 2: one call site, after the boot guard, before the SKIP_BENCH exit ─

def test_the_arm_route_calls_the_step_once_between_the_boot_guard_and_skip_bench():
    arm = (ROOT / ARM_REL).read_text(encoding="utf-8")
    assert arm.count(CALL) == 1, "the route must invoke the step exactly once"
    at = arm.index(CALL)
    # Its own statement, and run by the interpreter rather than by the exec bit:
    # this repo's own arm script is mode 644 in git, so a direct-exec call would
    # make the step's presence depend on a permission the arm script lacks.
    assert arm[arm.rindex("\n", 0, at) + 1:at] == "bash ", \
        "the call is its own statement, run by bash rather than by the exec bit"
    # After BOTH of the boot guard's `exit 2`s: each abort echo is followed by
    # the exit that ends the arm, and the step sits after the later of them, so
    # an arm whose engine did not come back never reaches a load.
    for echo in ('echo "!! ABORT $LABEL: engine initialised $BOOTS times in this window."',
                 'echo "!! ABORT $LABEL: engine reported a startup failure in this window."'):
        guard = arm.index(echo)
        refuse = arm.index("exit 2", guard)
        assert refuse < at, f"the step sits above a boot-guard exit ({echo[:40]}…)"
    # After the #1625 canary verdict persistence (#2163), so the canary's
    # decision is already on disk whatever the load does.
    assert arm.index("verdict NOT persisted to $CANARY_LOG") < at
    # Before the SKIP_BENCH=1 exit, so an admission arm — which is how a sweep
    # that might reach a preemption actually runs — still gets its load step.
    assert at < arm.index('if [[ "${SKIP_BENCH:-0}" == "1" ]]')
    assert at < arm.index('"$ROOT/.venvs/lloyd/bin/python" "$ROOT/agent-services/bin/bench-flash-next.py"')


def test_the_steps_absence_cannot_be_mistaken_for_the_arms_verdict():
    # The step always exits 0, so the only status that can come back from this
    # call is the shell failing to execute the file: a tree that predates it, a
    # checkout mid-copy, or the launcher suite's own fake root. This script runs
    # `set -uo pipefail` without -e, and the call sits where the last statement
    # above the SKIP_BENCH exit would otherwise BE the script's exit code — so
    # "the step wasn't there" would report itself as "the arm failed", which is
    # a wrong verdict about an arm whose engine is up and serving. The guard is
    # what makes it a note instead; the rc 127 the launcher's `test_the_canary_*`
    # nodes produced before it was exactly this.
    arm = (ROOT / ARM_REL).read_text(encoding="utf-8")
    assert "set -uo pipefail" in arm
    at = arm.index(CALL)
    after = arm[at + len(CALL):at + len(CALL) + 200]
    assert after.lstrip().startswith("\\") and GUARD in after, \
        "the call must be guarded, or its rc becomes the arm's"
    assert "continuing to bench" in after


# ── clause 3: rc=2 is one non-fatal note, and nothing bypasses the guard ─────

def test_the_drivers_idle_refusal_becomes_one_non_fatal_note_and_the_step_exits_zero(tmp_path):
    root = _stub_root(tmp_path, stderr=BUSY_REFUSAL, rc=2)
    proc = _run(root, tmp_path, preempt="1")
    assert proc.returncode == 0, "an arm must reach bench whatever the load step heard"
    notes = _notes(proc)
    assert len(notes) == 1, proc.stdout
    assert "engine busy" in notes[0], notes[0]
    assert "non-fatal" in notes[0] and "bench" in notes[0], notes[0]
    assert "wrote" not in proc.stdout, "a refused run wrote no record; say no more than that"


def test_the_step_passes_nothing_that_reaches_the_idle_guard(tmp_path):
    root = _stub_root(tmp_path, stdout=NOT_REACHED)
    proc = _run(root, tmp_path, preempt="1")
    argv = _argv(proc)
    flags = [a for a in argv[2:] if a.startswith("--")]
    assert set(flags) <= ALLOWED_PROBE_ARGS, flags
    # The `or env` half, which argv cannot show: whatever the step exported, the
    # child can say. Nothing past what `_run` itself handed the step, and nothing
    # whose name is a way past a guard. A step that added
    # `export LLOYD_PROBE_IGNORE_IDLE=1` above the call would pass an argv-only
    # assertion and fail this one.
    # bash's own bookkeeping aside (BASH_OWN_ENV_KEYS, measured above), every
    # name the child can see came from the arm — nothing the step added.
    extra = _env_keys(proc) - set(proc.arm_env) - BASH_OWN_ENV_KEYS
    assert not extra, f"the step exported keys of its own: {sorted(extra)}"
    step_src = (ROOT / STEP_REL).read_text(encoding="utf-8")
    assert not re.search(r"^\s*export ", step_src, re.M), \
        "the step is a reader of the arm's environment, not a writer into it"
    assert not [k for k in _env_keys(proc) if re.search(r"(?i)idle|busy|force|bypass|override", k)]
    src = (ROOT / PROBE_REL).read_text(encoding="utf-8")
    # The guard the clause says to KEEP, still where the item says it is, still
    # raising on a non-idle engine — and with no bypass flag on the parser for
    # the step to have found.
    assert "raise ProbeRefused(" in src
    assert "num_requests_running" in src and "num_requests_waiting" in src
    assert "--allow-busy" not in src and "--force" not in src


def test_an_instrument_failure_is_a_note_and_not_an_abort(tmp_path):
    root = _stub_root(tmp_path, stdout="Traceback (most recent call last): ...", rc=1)
    proc = _run(root, tmp_path, preempt="1")
    assert proc.returncode == 0
    notes = _notes(proc)
    assert len(notes) == 1 and "rc=1" in notes[0] and "non-fatal" in notes[0], notes


# ── clause 4: prompt length is the only size knob ────────────────────────────

def test_prompt_words_default_to_the_probes_own_and_are_the_only_size_forwarded(tmp_path):
    root = _stub_root(tmp_path, stdout=NOT_REACHED)
    argv = _argv(_run(root, tmp_path, preempt="1"))
    assert argv[argv.index("--load-prompt-words") + 1] == PROBE_DEFAULT_WORDS
    assert "--load-requests" not in argv, "the 8 sequences sharing the pool do not move"
    probe = (ROOT / PROBE_REL).read_text(encoding="utf-8")
    assert f'"--load-prompt-words", type=int, default={PROBE_DEFAULT_WORDS}' in probe
    assert '"--load-requests", type=int, default=8' in probe


def test_the_words_knob_escalates_and_no_pool_or_sequence_value_moves_anywhere(tmp_path):
    root = _stub_root(tmp_path, stdout=NOT_REACHED)
    argv = _argv(_run(root, tmp_path, preempt="1", words="24000"))
    assert argv[argv.index("--load-prompt-words") + 1] == "24000"
    for rel in (STEP_REL, ARM_REL):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            code = line.split(" #")[0]
            assert "MAX_NUM_SEQS=" not in code, f"{rel}: {line}"
            assert "KV_CACHE_MEMORY_BYTES=" not in code, f"{rel}: {line}"
    # The route forwards only the label: no size, no pool, no sequence count.
    arm = (ROOT / ARM_REL).read_text(encoding="utf-8")
    call = arm[arm.index(STEP_REL):].splitlines()[0]
    assert "--load-requests" not in call and "MAX_NUM_SEQS" not in call


# ── clause 5: not crossing the line is a stated result, with its numbers ─────

def test_a_run_that_did_not_cross_prints_the_record_and_its_numbers_and_succeeds(tmp_path):
    root = _stub_root(tmp_path, stdout=NOT_REACHED)
    proc = _run(root, tmp_path, preempt="1")
    assert proc.returncode == 0
    notes = _notes(proc)
    assert len(notes) == 1, proc.stdout
    note = notes[0]
    assert "/tmp/x/preempt_20261004T080000+0000_arm-t1.json" in note, note
    assert "0.0 -> 0.0" in note, "the before→after pair is the result; print it"
    assert "0.157" in note, "peak kv is recorded on both outcomes"
    assert "not an error" in note and "PREEMPT_LOAD_PROMPT_WORDS" in note, note


def test_a_reached_preemption_names_the_record_and_who_commits_it(tmp_path):
    root = _stub_root(tmp_path, stdout=REACHED)
    proc = _run(root, tmp_path, preempt="1")
    assert proc.returncode == 0
    notes = _notes(proc)
    assert len(notes) == 1, proc.stdout
    assert "0.0 -> 1.0" in notes[0] and "eval/engine_output/preempt/" in notes[0], notes[0]
    assert "load block" in notes[0], "the note must send the reader to the load block first"


def test_the_record_lands_in_the_runtime_dir_and_never_in_the_repo(tmp_path):
    # The step has no route for dirtying a tracked path: with no PREEMPT_DIR it
    # writes under $LLOYD_DATA, and nothing it does lands in $ROOT/eval.
    root = _stub_root(tmp_path, stdout=NOT_REACHED)
    argv = _argv(_run(root, tmp_path, preempt="1"))
    out = argv[argv.index("--out-dir") + 1]
    assert out == str(tmp_path / "data" / "logs" / "services" / "flash-next-arm" / "preempt")
    assert not out.startswith(str(root))


def test_the_probe_still_states_a_non_crossing_run_as_a_result():
    # The rail clause 5 says stays green is `test_preempt_not_reached_is_a_stated_result`
    # in tests/test_engine_output_probe.py, which this diff does not touch. This
    # node pins the half the step depends on: a completed preempt run writes its
    # record and exits 0 with `preemptions_reached` lowercased into the line the
    # step parses, and rc 2 is reserved for a refusal, not for "did not cross".
    src = (ROOT / PROBE_REL).read_text(encoding="utf-8")
    assert '"preemptions_reached": reached,' in src
    assert "preemptions_reached={str(record['preemptions_reached']).lower()}" in src
    # rindex, not index: an earlier `except ProbeRefused` closes the `compare`
    # branch at :682, and slicing to it would leave a window that is empty.
    tail = src[src.index('if args.cmd == "preempt":'):src.rindex("except ProbeRefused")]
    assert "path.write_text(" in tail and "return 0" in tail
    assert "raise ProbeRefused" in tail, "the only exit-2 route out of a completed run"
