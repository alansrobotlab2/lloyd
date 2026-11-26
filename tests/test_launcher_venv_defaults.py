"""#4145 — a launcher's `VLLM_VENV` route has to end somewhere a human can act on.

#3136 closed this for the primary slot's own launcher: `start-qwen38-flash-next.sh`
now defaults to an installed venv and its guard names two rebuilds. What it left
behind is the same defect in every file its checks did not read — `test_flash_next_-
launcher.py` pins that one script's default, and the venv-installed scan in
`test_vllm_doc_claims.py` read two prose scopes and no table. So
`start-secondary.sh` still defaulted to the deleted worker build and its venv route
ended at `vLLM venv not found` with nothing underneath it, and
`start-qwen3.8-27b-nvfp4.sh` still offered a Laguna fallback that nothing in
`setup/` can build.

This is risky rather than untidy for #3136's own reason: a `VLLM_VENV` line is
edited mid-incident, and `agent-llm-primary.conf:26-30` records a wrong rollback
instruction standing for ten days because nothing reads one until it is needed.

What is NOT pinned here is "the default is installed". Both arms are latent, not
broken: the secondary runs `MODEL=qwen36`, which execs `llama-server` before the venv
guard, with `autostart=false` and `secondary_enabled: false`; the 27B script is not
what the conf runs. Their weights are off-box too — `agent-services/llm/models/`
holds only the Flash-Next and DiffusionGemma checkpoints — so a default repointed at
an installed `vllm-flash-next-main*` build would boot a venv qualified for a
different model family at a moment when the reader believes they are recovering this
one. That is a worse failure than a refusal, so the defaults stay put, and the two
owed checks on #4145 (a weights download plus a boot measurement, and whether these
arms should be retired at all) decide whether they ever move. The invariant this file
holds instead: a route that cannot boot says what would make it bootable, and a
fallback never names an off-box venv as if it were a recovery.

`agent-services/README.md` is graded here too. It was out of reach until #4156
(`841da495`) added it to the gate's `ALLOWED_GLOBS` (`spec.classify` -> allowed);
#4512 then corrected its program table and venv table and restored the three nodes
#4145 wrote for them (`c3b16e22`, which never reached `main`): every `agent-llm-*`
row against its conf's `command=` and `GPU=`, the row set against the `[program:]`
units in `conf.d/`, the primary-slot paragraph against the conf's occupant and its
revert target, and the venv table under the same absent-mark rule as SETUP.md's.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENT_SERVICES = ROOT / "agent-services"
BIN = AGENT_SERVICES / "bin"
SETUP_DIR = AGENT_SERVICES / "setup"
CONF_D = AGENT_SERVICES / "supervisor" / "conf.d"
AGENT_README = AGENT_SERVICES / "README.md"

# The live tree the paths in these files point into. `ROOT` is the checkout, which
# in a gate worktree has no `.venvs` at all, so it is never the thing to check
# against — the same trap `test_vllm_doc_claims.py` names for `VENV_ROOT`.
VENV_ROOT = Path("/home/alansrobotlab/lloyd/.venvs")

SECONDARY = BIN / "start-secondary.sh"
NVFP4_27B = BIN / "start-qwen3.8-27b-nvfp4.sh"
FLASH_NEXT = BIN / "start-qwen38-flash-next.sh"
# The one script allowed to name the deleted Laguna venv: it is that script's own
# model, and its guard prints the `uv …` command that would rebuild it.
LAGUNA_LAUNCHER = BIN / "start-laguna-s-2.1-nvfp4-dflash.sh"

# One copy of the "that venv is not here" vocabulary, from the file that owns the
# doc-side scan, so the launcher rule and the doc rule cannot drift apart.
from tests.test_vllm_doc_claims import (  # noqa: E402
    _bad_venv_lines,
    _venv_table_lines,
)

# Every vLLM launcher that reaches a venv guard: the two this item names plus the
# primary's, which is the file #3136 already fixed — kept in the scan so a future
# edit to it cannot quietly reintroduce what this one closed.
VLLM_LAUNCHERS = [SECONDARY, NVFP4_27B, FLASH_NEXT]

# What each launcher defaults `VLLM_VENV` to. These values are the DECISION, not a
# snapshot: neither may move to a `vllm-flash-next-main*` build until the owed
# weights-and-boot measurement on #4145 says one is qualified for that model.
DEFAULTS = {
    SECONDARY.name: "$HOME/lloyd/.venvs/vllm-qwen38-flash-next",
    NVFP4_27B.name: "$HOME/lloyd/.venvs/vllm-qwen3.8",
    FLASH_NEXT.name: "$HOME/lloyd/.venvs/vllm-flash-next-main",
}

DEFAULT_LINE = re.compile(r'^VLLM_VENV="\$\{VLLM_VENV:-(?P<venv>.+)\}"$', re.M)
GUARD_OPEN = 'if [[ ! -x "$VLLM_VENV/bin/python" ]]; then'
# The rebuild a refusal may name. `setup/` is where this tree puts a build, so a
# name out of that set is only as good as the script that carries it.
SETUP_SCRIPT = re.compile(r"(setup-vllm-[A-Za-z0-9._-]+\.sh)")
# A `VLLM_VENV=<path> <command>` example line, in a comment or a doc's code fence —
# the shape a reader copies without reading. It is an invocation, which is what tells
# it apart from a launcher's own `VLLM_VENV="${VLLM_VENV:-…}"` default: that line is
# the decision clause 5 pins, and it is allowed to name a venv the box does not have
# precisely because clause 1 makes its refusal say so.
FALLBACK_EXAMPLE = re.compile(r"^\s*(?:#+\s*)?VLLM_VENV=(?!\$\{VLLM_VENV)(\S+)\s+\S+")
# A supervisor program row in the README's program table: `| `agent-llm-<x>` | port |
# `bin/<script>` | gpu | autostart |`, or `| `lloyd-mc:lloyd-backend` | …` for the
# grouped three. Columns are positional, which is what lets the GPU column be read
# back out without a table parser.
PROGRAM_ROW = re.compile(r"^\|\s*`((?:lloyd-mc:)?(?:agent|lloyd)-[a-z0-9-]+)`\s*\|")
# The README's own count of the table, in words: "Supervisord manages twelve programs".
PROGRAM_COUNT = re.compile(r"Supervisord manages (\w+) programs")
_WORDS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16}


def _text(path: Path) -> str:
    assert path.is_file(), f"{path} is not in the tree"
    return path.read_text(encoding="utf-8")


def _venv_guard_block(launcher: Path) -> str:
    """The body of the launcher's venv-exists refusal, comment lines included."""
    lines = _text(launcher).splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(GUARD_OPEN)]
    assert starts, f"{launcher.name} has no {GUARD_OPEN!r} guard — the venv route " \
                   "is no longer guarded, so this file's premise needs re-reading"
    body: list[str] = []
    for ln in lines[starts[0] + 1:]:
        if ln == "fi":
            break
        body.append(ln)
    assert body, f"{launcher.name}'s venv guard has no body"
    return "\n".join(body)


def _program_table_rows() -> dict:
    """agent-services/README.md's supervisor program table, keyed by program name
    with any `lloyd-mc:` group prefix dropped, so a key is a `[program:]` name."""
    out: dict[str, str] = {}
    for ln in _text(AGENT_README).splitlines():
        m = PROGRAM_ROW.match(ln)
        if m:
            name = m.group(1).split(":", 1)[-1]
            assert name not in out, \
                f"the program table has two `{name}` rows, so which one a " \
                "reader trusts is a coin flip"
            out[name] = ln
    return out


def _conf_programs() -> set[str]:
    """Every `[program:<name>]` section in `conf.d/*.conf` — what supervisord
    actually manages. `lloyd-mc.conf` is a `[group:]` with no program of its own."""
    out: set[str] = set()
    for conf in CONF_D.glob("*.conf"):
        out.update(re.findall(r"^\[program:([^\]]+)\]", _text(conf), re.M))
    return out


def _conf_field(pattern: str) -> str:
    """A field out of the live primary conf — the only statement of who holds the
    slot, which is why the README is graded against it and not against itself."""
    conf = _text(CONF_D / "agent-llm-primary.conf")
    m = re.search(pattern, conf, re.M)
    assert m, f"agent-llm-primary.conf no longer matches {pattern!r}"
    return Path(m.group(1)).name


def _example_lines() -> list[tuple[str, str]]:
    """Every `VLLM_VENV=<path> <command>` example in the vLLM launchers' own comment
    blocks and in agent-services/README.md — the lines a reader pastes without
    reading."""
    out: list[tuple[str, str]] = []
    for path in VLLM_LAUNCHERS + [AGENT_README]:
        for ln in _text(path).splitlines():
            if FALLBACK_EXAMPLE.match(ln):
                out.append((path.name, ln))
    return out


# ── clause 1: a venv refusal carries its own way out ────────────────────────────

@pytest.mark.parametrize("launcher", VLLM_LAUNCHERS, ids=lambda p: p.name)
def test_the_venv_guard_names_a_rebuild_that_exists(launcher: Path):
    """#3136's shape, applied past the primary. `start-secondary.sh` used to print
    `vLLM venv not found at $VLLM_VENV` and `exit 1`, and the only `Build it first`
    line in that file belongs to the `LLAMA_SERVER` check above, not to this venv."""
    block = _venv_guard_block(launcher)
    assert "not found" in block, \
        f"{launcher.name}'s guard no longer says the venv is missing, so this node " \
        "is grading something else"
    named = sorted(set(SETUP_SCRIPT.findall(block)))
    assert named, (
        f"{launcher.name}'s venv guard ends at a bare refusal: it names no "
        "`setup-vllm-*.sh` to rebuild with, which is exactly the dead end #3136 "
        "closed for the primary slot")
    missing = [n for n in named if not (SETUP_DIR / n).is_file()]
    assert missing == [], (
        f"{launcher.name}'s guard tells the reader to run {missing}, and nothing at "
        f"{SETUP_DIR} carries those scripts — an instruction to rebuild with a "
        "script that is not in the tree is the same dead end in a different font")
    assert re.search(r"^\s*exit 1\s*$", block, re.M), \
        f"{launcher.name}'s venv guard no longer refuses with exit 1"


@pytest.mark.parametrize("launcher", VLLM_LAUNCHERS, ids=lambda p: p.name)
def test_the_refusal_lines_rebuild_something_when_bash_runs_them(launcher: Path, tmp_path):
    """The refusal has to survive being printed, not just being written down: a
    `$PROJECT_DIR` that expanded empty, or a path built from a variable the guard
    never sets, would leave the reader a sentence with no command in it. So the
    launcher's own `echo` lines are lifted verbatim into a script, run, and what came
    out is checked — no launcher is executed, because the harness refuses to hand-run
    one from a background session ("that script boots a program supervisorctl already
    owns"), and a `subprocess` in a test is the same service action through a
    different surface. What is executed here is only the lines that print."""
    block = _venv_guard_block(launcher)
    printed = [ln for ln in block.splitlines() if ln.lstrip().startswith("echo ")]
    assert printed, f"{launcher.name}'s guard prints nothing at all"
    script = tmp_path / "guard-echoes.sh"
    # Only the lines that print, and no trailing `exit`: the exit status is then
    # bash's own verdict on the extracted text, not a constant this node typed in.
    # Under `set -u` an echo that expands a variable the guard never sets dies with
    # status 1 and nothing reaches the reader, which is the failure the assertions
    # below are for; a hardcoded `exit 1` would have hidden it.
    script.write_text(
        "set -euo pipefail\n"
        f'PROJECT_DIR="{SETUP_DIR.parent}"\n'
        'MODEL="qwen35"\n'
        'VLLM_VENV="/nonexistent/.venvs/absent"\n'
        + "\n".join(printed) + "\n", encoding="utf-8")
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          timeout=30, cwd=str(tmp_path))
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"{launcher.name}'s refusal lines did not run to completion under bash "
        f"(status {proc.returncode}): an echo here expanded something the guard does "
        f"not define, so the reader gets a partial sentence:\n{out}")
    named = sorted(set(SETUP_SCRIPT.findall(out)))
    assert named, (
        f"{launcher.name}'s refusal lines printed no rebuild command once bash "
        f"expanded them — this is what a human would have read:\n{out}")
    missing = [n for n in named if not (SETUP_DIR / n).is_file()]
    assert missing == [], \
        f"{launcher.name}'s printed refusal names {missing}, absent from {SETUP_DIR}"
    assert "nonexistent" in out or "/absent" in out, \
        f"the refusal never echoed the path it checked, so the reader cannot tell " \
        f"which venv it is complaining about:\n{out}"


# ── clause 2: an example is a route, so it has to be one ────────────────────────

def test_no_venv_example_names_a_directory_that_is_not_on_the_box():
    """A `VLLM_VENV=…/.venvs/…` line in a launcher's comment block is pasted without
    being read, mid-incident, by someone who believes the path is a recovery. Each one
    names a venv that is installed, or says on that same line that it is not — the
    rule `test_vllm_doc_claims.py` already holds for the docs' prose and tables,
    extended to the launchers."""
    examples = _example_lines()
    assert examples, "no VLLM_VENV example survives in the launchers, so this node " \
                     "has stopped guarding anything"
    bad = [(where, *row) for where, ln in examples
           for row in _bad_venv_lines([ln])]
    assert bad == [], (
        f"these examples name a venv that is not installed without saying so: {bad} "
        f"— the box holds {sorted(p.name for p in VENV_ROOT.iterdir())}")


def test_the_deleted_laguna_venv_is_no_longer_offered_as_a_fallback():
    """The specific line #4145's acceptance check greps for. `start-qwen3.8-27b-nvfp4.sh`
    offered `VLLM_VENV=~/lloyd/.venvs/vllm-laguna` as its no-edit fallback. That venv is
    uninstalled and no `setup/` script builds it. Measured at this commit,
    `git grep -n venvs/vllm-laguna HEAD -- agent-services/setup/` returns two lines and
    neither one is a build: `setup-qwen3.8-27b-nvfp4.sh:44` lists the path in a
    `VENV_CANDIDATES` probe that skips any candidate without `bin/python`, and
    `setup-vllm-qwen3.8.sh:300` PRINTS the same dead fallback in its closing
    `echo` — so that script's own success message still tells the operator to
    recover a 27B with a Laguna venv that is not on the box. It is
    `unlisted` in the gate's `ALLOWED_GLOBS` (`spec.classify` → unlisted), so this
    round cannot reach it; the line is recorded on #4145 instead. The only route that
    would actually build Laguna is the inline
    `uv venv … vllm==0.25.1 --torch-backend=cu130` inside
    `start-laguna-s-2.1-nvfp4-dflash.sh:79`'s own guard.

    So the rule is not "no launcher may say the word": the Laguna launcher names its own
    venv in its own guard and that is correct. It is that no launcher may point at it as
    somebody else's fallback. The scan is every `*.sh` in `bin/`, not the two files this
    round happened to edit — a third script left pointing at a deleted venv is the same
    defect, and `git grep -l vllm-laguna HEAD -- agent-services/bin/` returning anything
    but that one script is what that state looks like."""
    offenders = sorted(p.name for p in BIN.glob("*.sh")
                       if "vllm-laguna" in _text(p))
    assert offenders == [LAGUNA_LAUNCHER.name], \
        f"{offenders} name vllm-laguna; only {LAGUNA_LAUNCHER.name}, whose model IS "\
        "Laguna S 2.1, may — it is not on this box and no setup/ script builds it, so "\
        "in any other script the name is a dead route"
    # agent-services/README.md is the other surface a reader copies from: it carried
    # `VLLM_VENV=~/lloyd/.venvs/vllm-laguna bash bin/start-qwen3.8-27b-nvfp4.sh` as its
    # override example and a `vllm-laguna` venv row, both gone since #4512. Not even as
    # history: a dead path quoted in a doc is still a path somebody `cd`s to.
    assert "vllm-laguna" not in _text(AGENT_README), \
        "agent-services/README.md names vllm-laguna again, which is not on this box " \
        "and which no setup/ script builds"


# ── #4145 clause 3, landed by #4512: the README says what the confs run ─────────
#
# These three nodes were written for #4145 (`c3b16e22`), run red against the base
# README, and withheld while the file was outside the gate's `ALLOWED_GLOBS`. #4156
# (`841da495`) put it in the writable set; #4512 restored them with the README fix.

def test_every_llm_program_row_names_the_script_its_conf_runs():
    """The README's program table is what a reader reaches for before touching
    supervisorctl, and it was wrong on both LLM rows: the primary row said
    `start-qwen3.8-27b-nvfp4.sh` while `command=` ran `start-qwen38-flash-next.sh`,
    and the secondary row said `start-gemma-4-e4b-nvfp4.sh` while its conf runs
    `start-secondary.sh`.

    The GPU column is pinned too where a conf declares one, because that column is
    how a reader decides whether booting a slot evicts something: the secondary row
    said `—` while its conf pins `GPU="2"`, which is the card djev runs on.

    And the row SET is pinned against `conf.d/`: the table had eleven rows, one of
    them `agent-tts` (retired to `supervisor/retired/` on 2026-11-10), while
    `agent-djev` and `agent-tts-omni` had none — so the sentence above the table
    counted eleven programs against twelve `[program:]` units. `lloyd-mc.conf` is a
    `[group:]` with no program of its own and gets no row."""
    rows = _program_table_rows()
    programs = _conf_programs()
    assert set(rows) == programs, (
        f"agent-services/README.md's program table rows {sorted(rows)} are not the "
        f"[program:] units in {CONF_D}: missing {sorted(programs - set(rows))}, "
        f"extra {sorted(set(rows) - programs)}")
    m = PROGRAM_COUNT.search(_text(AGENT_README))
    assert m, "the README no longer says how many programs supervisord manages"
    stated = _WORDS.get(m.group(1).lower(), m.group(1))
    assert stated == len(programs), (
        f"the README says supervisord manages {m.group(1)} programs; conf.d/ "
        f"declares {len(programs)}")

    confs = sorted(CONF_D.glob("agent-llm-*.conf"))
    assert len(confs) >= 2, f"only {confs} — this node's premise is at least two slots"
    for conf in confs:
        program = conf.stem
        assert program in rows, \
            f"agent-services/README.md's program table has no `{program}` row, so " \
            "there is nothing here to compare the conf against"
        text = _text(conf)
        m = re.search(r"^command=/bin/bash\s+(\S+)", text, re.M)
        assert m, f"{conf.name} has no `command=/bin/bash …` line to check against"
        script = Path(m.group(1)).name
        assert f"bin/{script}" in rows[program], \
            f"{conf.name} runs {script}, but the `{program}` row says {rows[program]}"
        gpu = re.search(r'GPU="(\d+)"', text)
        if gpu:
            cols = [c.strip() for c in rows[program].strip().strip("|").split("|")]
            assert cols[3] == gpu.group(1), \
                f"the `{program}` row puts the slot on GPU {cols[3]!r} while " \
                f"{conf.name} pins GPU {gpu.group(1)!r}"


def test_the_readme_paragraph_names_the_occupant_and_the_revert_target():
    """The prose half of the same correction: the Start scripts paragraph asserted
    the 27B script was the live occupant ("Currently `start-qwen3.8-27b-nvfp4.sh`").
    It is the conf's named *revert target*, which is a different thing to say and the
    thing a reader swapping `command=` needs to see labelled."""
    script = _conf_field(r"^command=/bin/bash\s+(\S+)")
    assert (BIN / script).is_file(), \
        f"agent-llm-primary.conf runs {script}, which is not in {BIN}"
    readme = _text(AGENT_README)

    slots = [p for p in readme.split("\n\n") if "primary slot" in p]
    assert len(slots) == 1, \
        f"the 'primary slot' paragraph moved or doubled up: {len(slots)} paragraphs"
    assert script in slots[0], \
        f"the primary-slot paragraph does not name {script} as the occupant"

    # The 27B script stays in that paragraph — as the conf's named revert target,
    # which is what it is now — so the correction is a re-label and not a deletion of
    # the one name a revert would reach for.
    target = _conf_field(r"; Previous occupant \(revert target\):\n;+\s*(\S+)")
    assert target != script, "the conf's revert target is its own command, so this " \
                             "paragraph has nothing left to disambiguate"
    assert target in slots[0], \
        f"the paragraph no longer names {target}, the conf's revert target, so a " \
        "reader swapping `command=` has lost the pointer to what to swap back to"
    assert "revert target" in slots[0], \
        f"the paragraph names {target} without labelling it the revert target, so it " \
        "still reads as a second occupant"
    assert f"Currently `{target}`" not in slots[0], \
        f"the paragraph presents the revert target {target} as the live occupant"


def test_the_readme_venv_table_marks_every_venv_that_is_absent():
    """Same machinery `test_vllm_doc_claims.py` applies to SETUP.md, pointed at the
    table that used to list three venvs as though all three were on the box (and one
    of them as `live`)."""
    rows = _venv_table_lines(AGENT_README)
    assert rows, "agent-services/README.md has no venv table for this to read"
    bad = _bad_venv_lines(rows)
    assert bad == [], (
        f"these README rows name a venv that is not installed without saying so on "
        f"the row: {bad} — the box holds "
        f"{sorted(p.name for p in VENV_ROOT.iterdir())}")


# ── clause 5's other half: the defaults themselves stay where they are ──────────

@pytest.mark.parametrize("launcher", VLLM_LAUNCHERS, ids=lambda p: p.name)
def test_the_launcher_default_is_the_one_the_docs_describe(launcher: Path):
    """The default is reached only when the conf's `environment=` pin is dropped —
    `agent-llm-primary.conf:77` pins `VLLM_VENV=…-1110`, so today's boot never sees
    this line. It is the line a dropped variable falls to, which is why it is pinned
    rather than left to drift: repointing it at an installed build of another family
    needs the boot measurement #4145 owes, not a tidy grep."""
    m = DEFAULT_LINE.search(_text(launcher))
    assert m, \
        f"{launcher.name} no longer defaults VLLM_VENV with a `${{VLLM_VENV:-…}}` line"
    default = m.group("venv")
    assert default == DEFAULTS[launcher.name], (
        f"{launcher.name}'s default moved to {default}. If an installed build has "
        "now been MEASURED against this model family, update DEFAULTS with the "
        "measurement and the arm that produced it; if not, #4145's owed check is "
        "still open and this default is not ready to move")


@pytest.mark.parametrize("launcher", [SECONDARY, NVFP4_27B],
                         ids=lambda p: p.name)
def test_a_latent_arm_does_not_default_at_another_models_build(launcher: Path):
    """The half of #4145's acceptance that is a refusal to "fix" the grep by
    repointing a default. `vllm-flash-next-main*` is the FP8 Flash-Next build,
    qualified by measurement for the primary's checkpoint and for nothing else here —
    the secondary's arms are Qwen3.5-4B and LFM2.5-2.6B and the 27B is NVFP4, and
    none of those three checkpoints is on this box. Pointing a launcher at a venv
    that boots but was never qualified for its model turns a loud refusal into a
    served model nobody asked for, so it waits on the boot measurement #4145 owes.

    `start-qwen38-flash-next.sh` is deliberately NOT in this scan: its default is
    that build because it is that build's model, which is #3136's fix, pinned in
    `test_flash_next_launcher.py::test_the_venv_default_is_an_installed_directory`."""
    m = DEFAULT_LINE.search(_text(launcher))
    assert m, f"{launcher.name} no longer defaults VLLM_VENV with an explicit fallback"
    default = m.group("venv")
    assert "vllm-flash-next-main" not in default, (
        f"{launcher.name} now defaults to {default} — a Flash-Next build. Qualifying "
        "it for this arm is a boot measurement, not a rename; #4145 owes it and says "
        "what to do once it exists")
