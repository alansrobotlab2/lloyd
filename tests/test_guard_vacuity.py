"""Pins the teeth of `scripts/maintenance/guard_vacuity.py` (#636).

The harness answers one question — does an action-gating guard ever green-light
an input it cannot refuse? A harness that reports "0 % vacuous" *by construction*
is the same defect it audits (a check that cannot fail is not a check), so this
file does two things:

1. Plants fixture guards known to be vacuous and known to discriminate, and
   asserts the harness scores them the opposite ways. The scoring rule is
   exercised independently of every production guard, so a 0 % report can only
   come from the measurement, never from the instrument.
2. Pins the report's contract: one line per guard per reachable class with the
   real verdict beside the always-safe mutant's, every guard scored LIVE or
   VACUOUS, the rate line last and recomputable, and the #559 scope call derived
   from the table rather than asserted.

Offline by design: the harness builds its own synthetic AUTONOMY_DIR and
VAULT_PATH and pins its clock, so nothing here reads the live vault — safe under
`-m "not live_vault"`.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "maintenance" / "guard_vacuity.py"


def _load():
    spec = importlib.util.spec_from_file_location("guard_vacuity", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves its own annotations through sys.modules, so the module
    # has to be registered before its body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gv = _load()

RATE_RE = re.compile(r"VACUITY RATE: (\d+)/(\d+) = (\d+)% \((\d{4}-\d{2}-\d{2})\)")
HEAD_RE = re.compile(r"^GUARD \d+  (\S+)  (\S+) → (.*)$")
# The per-class unit of the report: clause 1's "one line per guard per class".
# Each line carries that guard's verdict and the always-safe mutant's side by side.
# The rate is per guard, so it is recomputed over the GUARD blocks and their
# SCORE lines, not over these — see test_the_rate_line_is_recomputable_from_the_lines.
OBSERVE_PREFIX = "    OBSERVE "
CLASS_RE = re.compile(
    r"^    OBSERVE (?P<site>\S+) (?P<symbol>\S+) class=(?P<cls>\S+): "
    r"guard=(?P<guard>PASS|BLOCK) mutant=(?P<mutant>PASS|BLOCK) -> (?P<status>\S+)"
    r"(?: \([^)]*owed (?P<owed>PASS|BLOCK)\))?")


def _capture():
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                          text=True, timeout=180, env=dict(os.environ), cwd=ROOT)
    # Clause 1 is "exits 0 … and prints". Pinned at the one place the script is
    # spawned for a clause check, so a run that printed a header and then died
    # cannot satisfy a stdout-only assertion anywhere in this file.
    assert proc.returncode == 0, f"rc={proc.returncode} stderr={proc.stderr[-2000:]}"
    assert proc.stdout.strip(), proc.stderr[-2000:]
    return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
            "lines": proc.stdout.splitlines()}


def _probed():
    """A private probed inventory, for the tests that rewrite verdicts in place.
    A shared fixture would leak a planted vacuity into everything after them."""
    inventory = gv.build_inventory()
    for g in inventory:
        g.probe()
    return inventory


@pytest.fixture(scope="module")
def report():
    """Two full runs of the script as an operator invokes it. Repeating it is
    what proves the report measures a fixed tree and not its environment."""
    first, second = _capture(), _capture()
    assert first["rc"] == 0, first["stdout"][-1500:] + first["stderr"][-1500:]
    first["again"] = second
    return first


@pytest.fixture(scope="module")
def guards():
    return _probed()


def _fixture(site="fixture/planted.py:1", symbol="always_safe", classes=(),
             scope_for=""):
    """A Guard holding planted classes; no production predicate is involved.

    Rows are (name, expect, real_verdict, mutant_verdict[, status_class]), and
    results are pre-computed the way `probe()` computes them, so only the
    scoring rule and the rendering are under test.
    """
    rows = []
    for row in classes:
        name, expect, real, mutant = row[:4]
        rows.append({"class": name, "expect": expect, "real": real, "mutant": mutant,
                     "diff": real != mutant, "note": "",
                     "status_class": row[4] if len(row) > 4 else False,
                     "status": ("witnessed" if real != mutant else
                                ("silent-ok" if expect == gv.PASS else "DEAD-BRANCH"))})
    # A planted guard lives at a synthetic site, so `site=` is handed in and
    # `_site` is never consulted (that is `__post_init__`'s one skip path). The
    # file/needle pair exists only to keep the field set honest: no production
    # guard may pass `site=` at all — see
    # test_no_inventory_site_is_pinned_to_a_line_number.
    g = gv.Guard(site_file=site.rpartition(":")[0], site_needle=symbol, site=site,
                 symbol=symbol, action="dispatch",
                 real=lambda p: gv.PASS, mutant=lambda p: gv.PASS,
                 classes=[], scope_for=scope_for)
    g.results = rows
    return g


def _verdicts(line):
    """Match on one OBSERVE line, with named groups site / symbol / cls / guard /
    mutant / status / owed. None for any other line. A class whose driver raised
    prints no `class=` pair of PASS/BLOCK verdicts and so does not match — an
    errored class is not a pair of verdicts, and this file asserts nothing about
    one beyond that it is printed and counted."""
    return CLASS_RE.match(line)


def _last_line(report):
    return [ln for ln in report["stdout"].splitlines() if ln.strip()][-1]


# ── clause 5: the harness cannot report 0 % by construction ─────────────────

def test_planted_guard_that_never_objects_is_scored_vacuous():
    """Every class either owes no refusal, or owes one and gets the mutant's
    answer anyway. The harness must call that VACUOUS, not 'it passed'."""
    g = _fixture(classes=[("healthy", gv.PASS, gv.PASS, gv.PASS),
                          ("missing_id", gv.BLOCK, gv.PASS, gv.PASS),
                          ("paused", gv.BLOCK, gv.PASS, gv.PASS),
                          ("draft", gv.BLOCK, gv.PASS, gv.PASS)])
    assert g.score == gv.VACUOUS, g.results


def test_planted_guard_that_discriminates_on_one_class_is_scored_live():
    g = _fixture(classes=[("healthy", gv.PASS, gv.PASS, gv.PASS),
                          ("missing_id", gv.BLOCK, gv.BLOCK, gv.PASS),
                          ("paused", gv.BLOCK, gv.PASS, gv.PASS)])
    assert g.score == gv.LIVE, g.results


def test_a_guard_with_no_class_list_cannot_be_scored_live():
    """Clause 3's tail, through the real code path rather than a planted result
    set: `probe()` fills in one unexercised row, which discriminates nothing."""
    g = gv.Guard(site_file="fixture/empty.py", site_needle="no_classes",
                 site="fixture/empty.py:1", symbol="no_classes", action="dispatch",
                 real=lambda p: gv.BLOCK, mutant=lambda p: gv.PASS, classes=[])
    g.probe()
    assert g.score == gv.VACUOUS, g.results
    assert g.results[0]["status"] == "unexercised"


def test_score_is_read_from_the_diffs_and_not_hardcoded():
    """Flip one class from discriminating to not, and the verdict flips with it.
    This is the only thing standing between the report and a vacuous 0 %."""
    g = _fixture(classes=[("x", gv.BLOCK, gv.BLOCK, gv.PASS)])
    assert g.score == gv.LIVE
    g.results[0]["mutant"] = gv.BLOCK
    g.results[0]["diff"] = False
    assert g.score == gv.VACUOUS


def test_rate_line_never_reports_a_nonzero_count_as_zero_percent():
    assert gv.rate_line(0, 13, "2026-01-01") == "VACUITY RATE: 0/13 = 0% (2026-01-01)"
    assert gv.rate_line(1, 150, "2026-01-01") == "VACUITY RATE: 1/150 = 1% (2026-01-01)"
    assert gv.rate_line(1, 13, "2026-01-01") == "VACUITY RATE: 1/13 = 8% (2026-01-01)"


def test_a_class_whose_driver_raises_is_counted_and_cannot_make_its_guard_live():
    """A driver that throws is a class nobody measured. It must still be on the
    page and still be in the denominator — swallowing it would print a rate over
    a hole in the measurement, which is the same silence this harness audits."""
    def boom(payload):
        raise RuntimeError("the driver itself broke")
    g = gv.Guard(site_file="fixture/raising.py", site_needle="raises",
                 site="fixture/raising.py:1", symbol="raises", action="dispatch",
                 real=boom, mutant=lambda p: gv.PASS,
                 classes=[gv.InputClass("one", gv.BLOCK, {}, ""),
                          gv.InputClass("two", gv.BLOCK, {}, "")])
    g.probe()
    assert [r["status"] for r in g.results] == ["error", "error"]
    assert g.score == gv.VACUOUS, "an unmeasured class is not evidence of a witness"
    assert g.observed().count("OBSERVE") == 2
    for line in g.observed().splitlines():
        assert "driver raised RuntimeError" in line, line


def test_the_error_status_is_explained_in_the_legend(report):
    """The legend names every status a line can carry; an undocumented one reads
    as a rendering glitch and gets skipped by whoever is triaging the report."""
    head = report["stdout"].split("GUARD 1", 1)[0]
    for status in ("witnessed", "silent-ok", "DEAD-BRANCH", "error"):
        assert status in head, status


def test_the_per_class_rationale_is_not_folded_into_the_observe_line():
    """One OBSERVE line per guard per class is the count the rate is recomputed
    from. A class note appended to the line would keep it one line but make the
    line's own fields depend on prose, so notes print separately and never
        inside an OBSERVE line."""
    g = _fixture(classes=[("healthy", gv.PASS, gv.PASS, gv.PASS)])
    g.results[0]["note"] = "nothing to refuse here"
    assert "note" not in g.observed()
    assert "nothing to refuse here" in g.rationale()
    report = _capture()
    assert not [ln for ln in report["lines"]
                if ln.startswith(OBSERVE_PREFIX) and "# " in ln]


def test_a_broken_probe_exits_nonzero_instead_of_printing_a_rate(monkeypatch, capsys):
    """A rate computed from a half-loaded inventory reads exactly like a clean
    audit — the failure mode this whole item is about."""
    def boom():
        raise RuntimeError("synthetic: a guard module failed to load")
    monkeypatch.setattr(gv, "build_inventory", boom)
    assert gv.run([]) == 1
    captured = capsys.readouterr()
    assert "PROBE FAILED" in captured.err
    assert "VACUITY RATE" not in captured.out


# ── clause 1: exits 0 fast, offline, one line per guard per class ───────────

def test_script_exits_0_in_under_sixty_seconds(report):
    assert report["rc"] == 0, report["stderr"][-1500:]


def test_makes_no_network_call_and_no_llm_call():
    """Mechanical, not a reading test: no URL literal, no client import, no
    socket, no HTTP library anywhere in the harness."""
    src = SCRIPT.read_text()
    assert not re.search(r"https?://", src), "the harness grew a URL"
    for banned in ("requests", "urllib", "httpx", "openai", "anthropic", "socket"):
        assert banned not in src, banned


def test_prints_one_line_per_guard_per_class_with_both_verdicts(report, guards):
    """Clause 1's shape, pinned as a count: the number of OBSERVE lines in the
    report equals the number of guard×class pairs in the inventory, so a class
    that silently stopped printing cannot shrink the report unnoticed."""
    printed = [ln for ln in report["lines"] if ln.startswith(OBSERVE_PREFIX)]
    assert len(printed) == sum(len(g.results) for g in guards), len(printed)
    for ln in report["lines"]:
        m = _verdicts(ln)
        if not m:
            continue
        owed, real, mutant = m["owed"], m["guard"], m["mutant"]
        assert owed in (gv.PASS, gv.BLOCK), ln
        assert real in (gv.PASS, gv.BLOCK) and mutant in (gv.PASS, gv.BLOCK), ln


def test_a_probed_class_prints_its_real_verdict_beside_its_mutant_verdict(report, guards):
    """The pairing is the deliverable: each class's row carries that guard's real
    verdict and the mutant's, not two columns assembled independently."""
    for g in guards:
        for r in g.results:
            rows = [ln for ln in report["lines"]
                    if (m := _verdicts(ln)) and m["site"] == g.site
                    and m["cls"] == r["class"]]
            assert any(f"guard={r['real']}" in ln and f"mutant={r['mutant']}" in ln
                       for ln in rows), (g.site, r["class"], r["real"], r["mutant"])


def test_the_report_carries_no_chatter_from_the_modules_under_test(report):
    """Two probed scripts log at import time. Forwarded, the report would be a
    transcript nobody can pin — and one line quoted a $HOME-derived path, which
    broke the determinism assertion below."""
    assert "Task #" not in report["stdout"]
    assert "/tmp/" not in report["stdout"]


# The re-exec seam. `python scripts/maintenance/guard_vacuity.py` is clause 1's
# literal command, and the `autonomy` import pulls in `mcp.types`, which a plain
# interpreter on this box does not have — so the script self-re-execs into one
# that can. That re-exec is the one process boundary clause 1 crosses, and it is
# tested with the first interpreter genuinely unable to import the services, not
# with a skip that passes when the block is absent.
_BLOCKER = """
import os
import sys


class _BlockMcp:
    def find_spec(self, name, path=None, target=None):
        if name == 'mcp' or name.startswith('mcp.'):
            raise ModuleNotFoundError('No module named ' + repr(name) +
                                      ' (test blocker)')
        return None


def _log(line):
    with open(os.environ['GUARD_VACUITY_BLOCK_LOG'], 'a') as fh:
        fh.write(line + chr(10))


if os.path.basename(sys.argv[0]) == 'guard_vacuity.py':
    if os.environ.get('GUARD_VACUITY_REEXEC'):
        # Stood down on purpose: the whole point of the re-exec is to land in an
        # interpreter that CAN import the services.
        _log('PASSED_THROUGH')
    else:
        sys.meta_path.insert(0, _BlockMcp())
        try:
            import mcp.types  # noqa: F401  positive control, in-process
            _log('NOT_BLOCKED')
        except ModuleNotFoundError:
            _log('BLOCKED')
"""


def _forced_block_run(tmp_path):
    """Run the script with `mcp` genuinely unavailable to the first interpreter.

    The first interpreter is a symlink to this pytest process's own interpreter,
    invoked through that symlink: the script compares candidate interpreters
    against `sys.executable` by path, so a distinct path is what makes it a
    different interpreter to re-exec *away* from, exactly as `python` is on this
    box. Symlinking (rather than pointing at `/usr/bin/python`) keeps the test
    about the seam instead of about which builds happen to be installed: it would
    pass unchanged on a box where `/usr/bin/python` grew an `mcp`, and it fails
    here if the re-exec block is deleted.

    Returns (CompletedProcess, blocker log lines). The log is the positive
    control: a `BLOCKED` line means the block really applied to the process that
    started the run, so a pass cannot come from the block having silently not
    applied — which is the shape of defect this whole item exists to catch.
    """
    pkg = tmp_path / "gvblock"
    pkg.mkdir()
    (pkg / "sitecustomize.py").write_text(_BLOCKER)
    blocked_python = tmp_path / "python-without-mcp"
    blocked_python.symlink_to(sys.executable)
    log = tmp_path / "block.log"
    env = {k: v for k, v in os.environ.items() if k != "GUARD_VACUITY_REEXEC"}
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(pkg) + ((os.pathsep + existing) if existing else "")
    env["GUARD_VACUITY_BLOCK_LOG"] = str(log)
    proc = subprocess.run([str(blocked_python), str(SCRIPT)], capture_output=True,
                          text=True, timeout=180, env=env, cwd=ROOT)
    lines = log.read_text().split() if log.exists() else []
    return proc, lines


def test_the_re_exec_survives_a_first_interpreter_that_cannot_import_the_services(tmp_path):
    """The seam crossed: `mcp` blocked in the first interpreter, and the report
    must still come out exit 0 with its rate line."""
    proc, lines = _forced_block_run(tmp_path)
    assert "BLOCKED" in lines, (
        "the blocker never fired, so this run proved nothing about the re-exec: "
        f"log={lines}")
    assert proc.returncode == 0, f"rc={proc.returncode} stderr={proc.stderr[-2000:]}"
    assert proc.stdout.rstrip().startswith("guard-vacuity")
    assert RATE_RE.search(proc.stdout), "no VACUITY RATE line after the re-exec"


def test_the_re_exec_lands_where_the_services_import(tmp_path):
    """The other side of the seam: the re-exec'd process ran with the block
    lifted. A re-exec back into the same blocked environment would have to die
    loudly to fail this, and the rate line would be absent."""
    proc, lines = _forced_block_run(tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "PASSED_THROUGH" in lines, (
        f"no re-exec'd process ran with the blocker stood down: log={lines}")
    assert lines.index("BLOCKED") < lines.index("PASSED_THROUGH"), lines
    assert RATE_RE.search(proc.stdout)


def test_the_blocker_does_not_poison_the_candidate_probe(tmp_path):
    """Scope guard on the blocker, so the two tests above cannot pass by accident.
    The script probes candidate interpreters with `python -c`; poisoning that too
    would make every interpreter look broken and the run would report a re-exec
    that never happened. `import mcp.types` under `-c` with the same PYTHONPATH is
    the direct check, and it fails if the blocker ever keys on something other than
    the script's own argv[0]."""
    pkg = tmp_path / "gvblock2"
    pkg.mkdir()
    (pkg / "sitecustomize.py").write_text(_BLOCKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pkg)
    env["GUARD_VACUITY_BLOCK_LOG"] = str(tmp_path / "unused.log")
    probe = subprocess.run([sys.executable, "-c", "import mcp.types; print('ok')"],
                           capture_output=True, text=True, timeout=120, env=env)
    assert probe.returncode == 0, probe.stderr[-1000:]
    assert "ok" in probe.stdout


def _scratch_worktrees(tmp_path):
    """A main worktree that owns `.venvs/lloyd/bin/python`, plus a real LINKED
    worktree off it that owns none — the layout an automod round actually runs in.

    Built with real `git worktree add` rather than a mock of git's answer, because
    the thing under test is precisely what git answers for `--git-common-dir` from a
    linked checkout. A two-file scratch repo, not a checkout of this one: the
    derivation is git's, so it is the same derivation for any repository, and this
    keeps the fixture to one commit and one worktree instead of a whole copy.
    """
    main = tmp_path / "main-wt"
    main.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=main, capture_output=True,
                              text=True, timeout=120)

    assert git("init", "-q", ".").returncode == 0
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (main / "keep.txt").write_text("x\n")
    assert git("add", "keep.txt").returncode == 0
    assert git("commit", "-qm", "init").returncode == 0
    venv = main / ".venvs/lloyd/bin/python"
    venv.parent.mkdir(parents=True)
    # A wrapper, not a symlink: a symlinked venv python resolves through to the
    # base build and loses the venv's site-packages, which is the very trap
    # `_services_importable` exists to catch — a symlink here would make the
    # candidate genuinely un-importable and the test would fail on the fixture.
    venv.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
    venv.chmod(0o755)
    linked = tmp_path / "linked-wt"
    add = git("worktree", "add", "--detach", str(linked), "HEAD")
    assert add.returncode == 0, add.stderr[-500:]
    assert not (linked / ".venvs").exists(), "a linked worktree must have no venv of its own"
    return main, linked, venv


def test_a_linked_worktree_finds_the_main_worktrees_venv_through_git(tmp_path, monkeypatch):
    """The named seam: an automod worktree has no `.venvs` of its own, so
    `ROOT/.venvs/...` cannot be the whole answer and the `--git-common-dir`
    fallback is what makes the script runnable from a round. Answered by real git
    against a real linked worktree; the only thing redirected is which tree the
    script believes it is standing in."""
    main, linked, venv = _scratch_worktrees(tmp_path)
    monkeypatch.setattr(gv, "ROOT", linked)
    cands = list(gv._venv_candidates())
    assert cands[0] == linked / ".venvs/lloyd/bin/python", cands
    assert venv in cands, (
        f"the git-derived fallback did not resolve to {venv}: {cands}")
    assert (main / ".git").exists(), "fixture lost its common dir"


def test_the_re_exec_target_is_the_git_derived_venv_not_the_live_ones(tmp_path, monkeypatch):
    """And the target the re-exec actually gets handed. The first interpreter is
    genuinely unable to import the services — `/bin/false`, probed for real, not a
    mock of the probe — and `os.execv` is recorded because it replaces the process,
    so the recorded path is the only observable side of the seam. It must be the
    scratch main worktree's venv: had the `--git-common-dir` fallback not resolved,
    the next candidate would have been THIS repo's venv, and the assertion would
    fail on a path that exists and works.

    A recorded execv RETURNS, so control reaches the harness's last line — the
    refusal that exits 2 rather than print a rate it could not have measured. In
    production that line is unreachable once a candidate answers; here it is the
    only way to see the loop ended, so the exit code is asserted as part of the
    seam rather than treated as noise."""
    main, linked, venv = _scratch_worktrees(tmp_path)
    monkeypatch.delenv("GUARD_VACUITY_REEXEC", raising=False)
    monkeypatch.setattr(gv, "ROOT", linked)
    monkeypatch.setattr(gv.sys, "executable", "/bin/false")
    seen = []
    # Rebind the harness's OWN `os` name to a shim rather than patching `os.execv`
    # on the shared module: `subprocess` resolves `os.execv` at spawn time inside
    # the child, so patching the real one makes every candidate interpreter look
    # un-importable and the test would fail for a reason that has nothing to do
    # with the seam. `environ` is a real dict copy, so the marker write is still
    # exercised; it just cannot escape into the test process's own environment.
    shim = types.SimpleNamespace(environ=dict(os.environ),
                                 execv=lambda path, argv: seen.append((path, argv)))
    shim.environ.pop("GUARD_VACUITY_REEXEC", None)
    monkeypatch.setattr(gv, "os", shim)
    # sys.argv honestly, not pytest's. The harness re-execs only when argv names its
    # own script — the same guard that stops it chasing an `argv[1]` pointing at some
    # other script, which is what makes the loop terminate. Under pytest argv[1] is
    # THIS file, so leaving argv alone exercises that guard's refusal path instead of
    # the candidate walk, and the test fails because the harness correctly declined
    # to re-exec into a test file.
    monkeypatch.setattr(gv.sys, "argv", [str(SCRIPT)])
    with pytest.raises(SystemExit) as raised:
        gv._ensure_project_interpreter()
    assert raised.value.code == 2, (
        f"fell through to exit {raised.value.code!r}: a candidate walk that found "
        "nothing must refuse loudly, not report a rate")
    assert seen, (
        "no re-exec was attempted: with /bin/false as the first interpreter the "
        f"candidate list was {[str(p) for p in gv._venv_candidates()]}")
    path, argv = seen[0]
    assert Path(path) == venv, f"re-exec'd {path}, not the git-derived {venv}"
    assert argv[0] == str(venv) and argv[1] == str(SCRIPT), argv


def test_the_report_fixture_pins_the_exit_code_it_is_named_for():
    """Clause 1 is `exits 0 … and prints`. The shared fixture asserts the exit
    code, so every clause test that reads the report inherits it — a run that
    printed a header and then died fails here, not twelve lines later."""
    report = _capture()
    assert report["rc"] == 0


def test_the_script_under_a_foreign_interpreter_still_produces_a_report():
    """Clause 1 spells the command `python`, and on this box that is a different
    build with no `mcp` — hence the self-re-exec. Without it, the clause's own
    invocation is the thing that breaks the clause. Kept as the end-to-end run of
    the literal command the clause names, wherever this box's `python` points."""
    foreign = shutil.which("python")
    if not foreign:
        pytest.fail("clause 1's command is `python scripts/maintenance/guard_vacuity.py`; "
                    "no `python` on PATH, so the clause cannot be run as written")
    env = {k: v for k, v in os.environ.items() if k != "GUARD_VACUITY_REEXEC"}
    proc = subprocess.run([foreign, str(SCRIPT)], capture_output=True,
                          text=True, timeout=180, env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.rstrip().startswith("guard-vacuity")
    assert RATE_RE.search(proc.stdout)


def test_repeated_runs_are_identical_apart_from_the_probe_stamp(report):
    """Two runs of the same tree must agree on everything that is a measurement of
    the code, and the assertion is only as good as its list of what legitimately
    moves. Three things do: the probe stamp and the rate's date (a run either side
    of midnight re-dates itself, and the rate line carries the date clause 4 asks
    for), and the live corpus size `kg_rebuild._corpus_size()` reads so a dead
    branch can be told from a branch that is dead *right now* — that count is a
    vault read, so a nightly write landing between the two runs moves exactly that
    one digit-run and nothing else. Scrubbing it is naming the known-varying
    input, which is what makes the remaining bytes a determinism claim rather than
    a race; leaving it in would flake every clause test that reads `report`."""
    def scrub(text):
        text = re.sub(r"(repo HEAD: [0-9a-f]+ ).*", r"\1<stamp>", text)
        text = re.sub(r"live corpus size at probe time: \S+",
                      "live corpus size at probe time: <n>", text)
        return re.sub(r"\d{4}-\d{2}-\d{2}[T ]?[\d:]*Z?", "<date>", text)

    assert scrub(report["again"]["stdout"]) == scrub(report["stdout"])
    # Positive control on the scrub itself: it only passes because the corpus line
    # is scrubbed, so prove the line was present to be scrubbed — an unscrubbed
    # diff of the two raw runs would otherwise be the only thing that noticed its
    # absence, and a scrub that quietly matches nothing always "passes".
    assert "live corpus size at probe time:" in report["stdout"]


# ── clause 2: >=10 guards, as file:line → action, the item's sites present ──

def test_inventory_names_at_least_ten_guards_as_file_line_arrow_action(report):
    heads = [HEAD_RE.match(ln) for ln in report["lines"] if ln.startswith("GUARD ")]
    unparsed = [ln for ln in report["lines"]
                if ln.startswith("GUARD ") and not HEAD_RE.match(ln)]
    assert not unparsed, unparsed[:2]
    assert len(heads) >= 10, len(heads)
    for m in heads:
        site, symbol, action = m.groups()
        assert re.fullmatch(r"[\w./-]+:\d+", site), site
        assert symbol.strip() and action.split("   (")[0].strip(), (symbol, action)


def test_every_site_names_a_file_that_exists_and_a_line_it_has(guards):
    """A site nobody can open is how the item's own proving command failed
    (`scripts/selfmod/gate.py`). The harness must never print one."""
    for g in guards:
        path, _, line_no = g.site.rpartition(":")
        target = ROOT / path
        assert path and line_no.isdigit() and target.is_file(), g.site
        assert int(line_no) <= len(target.read_text().splitlines()), g.site


@pytest.mark.parametrize("needle", [
    "autonomy.py",                                    # the dependency gate
    "scripts/automod/gate.py",                        # the pytest floors
    "scripts/memory/entity-resolution-sweep.py",      # merge tier + baseline
    "scripts/memory/kg_rebuild.py",                   # the coverage gate
    "scripts/service_health_check.py",                # RUNNING-without-progress
    "scripts/autoresearch/promote.py",                # delta + win fraction
])
def test_a_surface_the_item_named_is_covered(needle, guards):
    assert any(g.site.startswith(needle + ":") for g in guards), needle


@pytest.mark.parametrize("symbol", ["_is_dependency_met", "decide_merge",
                                    "degraded_reason", "_supervisor_verdict"])
def test_a_symbol_the_item_named_by_name_is_in_the_inventory(symbol, guards):
    assert symbol in {g.symbol for g in guards}


def test_the_corrected_gate_path_is_used_not_the_item_mistake(report):
    """#636's own proving command named `scripts/selfmod/gate.py`, which does not
    exist; the report must carry the real path and never the wrong one."""
    assert "scripts/automod/gate.py:" in report["stdout"]
    assert "scripts/selfmod/" not in report["stdout"]


def test_each_clause_spelling_of_a_moved_site_is_printed_and_resolves(report, guards):
    """#636's clauses cite `gate.py:54/60`, `kg_rebuild.py:63-64` and
    `service_health_check.py:114`. HEAD moved two of them, so the report prints
    both the live site and the clause's spelling; both must be real lines, and
    the live site must name the code it claims to gate."""
    for clause, file, needle in [
        # `PYTEST` not `PYTEST_MIN`: gate.py:63 is the third floor, the sibling
        # the clause did not name, and its line must resolve too.
        ("gate.py:54/60", "scripts/automod/gate.py", "PYTEST"),
        ("kg_rebuild.py:63-64", "scripts/memory/kg_rebuild.py", "coverage_pct"),
        ("service_health_check.py:114", "scripts/service_health_check.py", "RUNNING"),
    ]:
        assert clause in report["stdout"], clause
        body = (ROOT / file).read_text().splitlines()
        cited = [g for g in guards if g.site.startswith(file + ":")]
        assert cited, file
        for g in cited:
            live = int(g.site.rsplit(":", 1)[1])
            line = body[live - 1]
            # Admissible two ways: the line itself holds the code the guard gates,
            # or the site is the `def` of the named symbol (as
            # service_health_check.py:112 is, where the decisive comparison is
            # three lines down at :128 and the clause quoted the comment above).
            assert needle in line or line.lstrip().startswith(f"def {g.symbol}("), (
                g.site, line)
        for quoted in re.findall(rf"{re.escape(Path(file).name)}:(\d+)", report["stdout"]):
            assert 1 <= int(quoted) <= len(body), (file, quoted, len(body))


def test_no_inventory_site_is_pinned_to_a_line_number(guards):
    """Clause 2's `no line numbers are pinned`, checked the only way it can be:
    every printed site must equal what a fresh lookup of the guard's own name
    returns. A site literal baked into the inventory drifts the day the file
    above it changes: two of the thirteen did not name the guard printed beside
    them when this round started — the `kg_rebuild` coverage floors, declared in
    the opposite order to the pair item #636 spells, so each cited a line holding
    the other floor's key — and a stale number still reads as a citation."""
    for g in guards:
        assert g.site_file and g.site_needle, g.site
        assert g.site == gv._site(g.site_file, g.site_needle), g.site
        assert g.site.startswith(g.site_file + ":"), (g.site, g.site_file)


def test_a_site_moves_when_the_file_under_it_shifts(tmp_path):
    """The same lookup, run against a copy of the file with lines prepended,
    prints the shifted line — proof the number is read out of the checkout and
    not out of the source of the harness. Falsified by replacing `_site` with a
    returned constant: the assertion below fails on the mismatch."""
    rel = "scripts/automod/gate.py"
    body = (ROOT / rel).read_text().splitlines()
    shifted = ["# prepended by a later commit"] * 7 + body
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text("\n".join(shifted) + "\n")

    before = gv._site(rel, "PYTEST_MIN_PASSED")
    after = gv._site(rel, "PYTEST_MIN_PASSED", root=tmp_path)
    assert int(after.rsplit(":", 1)[1]) == int(before.rsplit(":", 1)[1]) + 7, (before,
                                                                              after)
    assert "PYTEST_MIN_PASSED" in (tmp_path / rel).read_text().splitlines()[
        int(after.rsplit(":", 1)[1]) - 1]


def test_a_renamed_guard_makes_the_report_refuse_rather_than_cite_a_stale_line():
    """The failure this is for: a guard that moves or is renamed must not leave a
    plausible `file:line` behind pointing at unrelated code. That is the same
    defect the harness is auditing — a verdict that cannot be wrong because it no
    longer reads the thing it claims to read."""
    with pytest.raises(gv.SiteNotFound) as raised:
        gv._site("scripts/automod/gate.py", "PYTEST_MIN_NEVER_EXISTED")
    assert "no line names" in str(raised.value)


def test_a_moved_guard_name_stops_the_report_instead_of_printing_a_rate(monkeypatch,
                                                                        capsys):
    """Same refusal, one level up: the lookup runs inside `build_inventory`, which
    `run()` already guards, so a moved symbol exits 1 with no rate line — never a
    rate measured over an inventory whose sites no longer resolve."""
    def refuse(rel, needle, *, root=None):
        raise gv.SiteNotFound(f"{rel}: no line names {needle!r}")
    monkeypatch.setattr(gv, "_site", refuse)
    assert gv.run([]) == 1
    captured = capsys.readouterr()
    assert "PROBE FAILED" in captured.err and "no line names" in captured.err
    assert "VACUITY RATE" not in captured.out


def test_every_inventory_line_carries_a_gated_action(guards):
    """Clause 2's `→ gated action` half: a green verdict must name what it lets
    happen, and the actions must span more than one kind of action."""
    actions = {g.action for g in guards}
    assert "dispatch" in actions, actions
    assert len(actions) >= 3, actions
    assert all(g.action.strip() for g in guards)


# ── clause 3: every line scored, VACUOUS only when nothing discriminates ────

def test_every_guard_is_scored_live_or_vacuous_in_the_text(report, guards):
    scores = re.findall(r"^    SCORE: (\S+)", report["stdout"], re.MULTILINE)
    assert len(scores) == len(guards), (len(scores), len(guards))
    assert set(scores) <= {gv.LIVE, gv.VACUOUS}, set(scores)
    assert set(scores) == {gv.LIVE, gv.VACUOUS}, "both verdicts appear in a real run"


def test_vacuous_is_emitted_only_when_no_class_discriminates(guards):
    for g in guards:
        assert g.score == (gv.VACUOUS if not any(r["diff"] for r in g.results)
                           else gv.LIVE), g.site
    vacuous = [g for g in guards if g.score == gv.VACUOUS]
    assert vacuous, ("with zero VACUOUS lines in the real inventory this rule would be "
                     "unpinnable here; the planted fixtures above are the other half")
    for g in vacuous:
        assert not any(r["diff"] for r in g.results), g.site


def test_a_vacuous_guard_is_printed_with_its_verdict_and_every_class(report, guards):
    for g in [g for g in guards if g.score == gv.VACUOUS]:
        head = report["stdout"].split(f"{g.site}  {g.symbol}", 1)
        assert len(head) > 1, g.site
        body, score = head[1].split("SCORE:", 1)
        assert score.strip().startswith(gv.VACUOUS), g.site
        for r in g.results:
            assert f"class={r['class']}" in body, (g.site, r["class"])


def test_every_dead_branch_is_listed_under_its_own_heading(report, guards):
    dead = [(g, r) for g in guards for r in g.results if r["status"] == "DEAD-BRANCH"]
    assert dead, "no dead branch anywhere would make this assertion vacuous"
    body = report["stdout"].split("DEAD BRANCHES", 1)[1].split("VACUITY RATE")[0]
    for g, r in dead:
        assert g.site in body and g.symbol in body and r["class"] in body, (g.site,
                                                                            r["class"])


def test_a_witnessed_class_is_not_reported_as_a_dead_branch(report, guards):
    """The mirror of the test above: the section is the DEAD-BRANCH set, not
    every class where the guard refused something."""
    body = report["stdout"].split("DEAD BRANCHES", 1)[1].split("VACUITY RATE")[0]
    for g in guards:
        for r in g.results:
            if r["diff"]:
                assert f"{g.symbol} class={r['class']}\n" not in body, (g.site,
                                                                        r["class"])


# ── clause 4: the rate line is last and recomputable ───────────────────────

def test_rate_line_is_the_last_line_and_well_formed(report):
    assert RATE_RE.fullmatch(_last_line(report)), _last_line(report)


def test_rate_line_recomputes_from_the_lines_above_it(report, guards):
    n, N, pct, date = RATE_RE.match(_last_line(report)).groups()
    n, N, pct = int(n), int(N), int(pct)
    assert N == len(guards), (N, len(guards))
    assert n == sum(1 for g in guards if g.score == gv.VACUOUS), n
    assert pct == (max(round(100.0 * n / N), 1) if n else 0), (pct, n, N)
    assert date in report["stdout"], date


def test_the_industrial_comparison_is_attributed_and_caveated(report):
    """Deliverable 1 asks for the comparison; an unattributed or uncaveated 20 %
    is the kind of number that gets quoted as a target."""
    assert "[BBER01]" in report["stdout"] and "[KV03]" in report["stdout"]
    assert "per GUARD" in report["stdout"]


# ── clause 6: the #559 scope call comes out of the table ───────────────────

def test_scope_section_tabulates_every_status_class(report, guards):
    start = report["stdout"].index("\n#559 SCOPE\n")
    end = report["stdout"].index("SCOPE CALL:", start)
    section = report["stdout"][start:end]
    status_rows = [(g, r) for g in guards for r in g.results if r["status_class"]]
    assert status_rows, "no status/state class tagged anywhere: the section is vacuous"
    assert len({g.site for g, _ in status_rows}) >= 2, "one guard is not a table"
    for g, r in status_rows:
        assert f"class={r['class']}" in section, (g.site, r["class"])
    assert "in-triad=" in section and "vacuous=" in section and "guard-fired=" in section
    assert "#559 asked about this one directly" in section, "the triad's own gate is marked"


def test_scope_call_is_present_and_its_verdict_matches_the_measurement(report, guards):
    """The printed call is the rendering of `scope_call(guards, …)` for whatever the
    rows say, so the assertion follows the branch instead of naming one.

    It used to begin `assert beyond or triad_hit` — that is, it required the
    measurement to have found something vacuous somewhere in the triad. That is not
    a property of the report: it is a claim about the code under audit, and it went
    red the day #558's fail-closed fix landed underneath it and every triad class
    came back LIVE. A test that fails because the system got safer reports a broken
    change, which is the same false alarm this whole item exists to stop. What is
    invariant is that the sentence is generated from the rows: asserted below by
    comparing the printed text against freshly regenerated lines, and by
    `test_the_scope_call_credits_only_what_the_rows_show` for the branches the live
    tree does not currently take."""
    beyond, triad_hit = gv.scope_beyond(guards)
    printed = report["stdout"].split("\n#559 SCOPE\n", 1)[1]
    lines = gv.scope_call(guards, beyond, triad_hit)
    assert lines, "the scope call rendered nothing"
    assert lines[0].startswith("    #559 SCOPE CALL: "), lines[0]
    for line in lines:
        assert line in printed, line[:120]
    if beyond:
        assert "outside" in lines[0], lines[0]
        for b in beyond:
            assert b in printed, b[:160]
    else:
        assert "no vacuous status/state class outside {missing-id, paused, draft}" \
            in lines[0], lines[0]
        # Nothing outside the triad may be called vacuous in the call's own text.
        assert not any(b in "\n".join(lines) for b in beyond)


HIT_NOT_TOLERATED = "did find vacuous inside the triad"


def test_the_scope_call_credits_only_what_the_rows_show():
    """Clause 5's teeth, on the branches the live tree is not currently on.

    The shipped call once ended 'What survives is the DEAD-BRANCH one alone —
    #558's open decision'. That was measured on 2026-09-19, went false within hours
    when #558's fail-closed fix landed underneath it, and stayed in the file: the
    table printed `0 vacuous class(es) … inside the triad: none` and the call went
    on naming a surviving dead branch a few lines lower. A call that credits a class
    the table does not show is the vacuous green verdict this item audits, wearing a
    sentence instead of a guard. Each fixture below is what the call must say, and
    must not say, for a table in that shape.
    """
    def call_text(rows_by_guard):
        guards = []
        for i, (site, rows) in enumerate(rows_by_guard):
            guards.append(_fixture(site=site, symbol=f"gate_{i}",
                                   classes=rows, scope_for="#559"))
        for g in guards:
            for r in g.results:
                r["status_class"] = True
        beyond, hit = gv.scope_beyond(guards)
        return "\n".join(gv.scope_call(guards, beyond, hit)), beyond, hit, guards

    # (a) triad measured, one class dead: the dead one is named, and the count of
    # classes actually fed is quoted rather than the size of the triad.
    text, beyond, hit, _ = call_text([
        ("fixtures/a.py:1", [("missing-id", gv.BLOCK, gv.PASS, gv.PASS),
                             ("paused", gv.BLOCK, gv.BLOCK, gv.PASS)]),
    ])
    assert hit and not beyond
    assert gv.HIT_MARKER in text
    assert "missing-id=DEAD-BRANCH" in text
    assert "paused=LIVE" in text
    # The count is what was fed, not the size of the triad: a call that said "all
    # three measured" with `draft` never in front of a mutant would be exactly the
    # kind of sentence this function exists to prevent.
    assert "2 of 3 triad classes were fed" in text
    assert "draft=not fed to this harness" in text

    # (b) triad measured, nothing dead (the shape #558's fix left behind): no
    # surviving dead branch may be named, and the call says it measured nothing.
    text, beyond, hit, _ = call_text([
        ("fixtures/a.py:1", [("missing-id", gv.BLOCK, gv.BLOCK, gv.PASS),
                             ("paused", gv.BLOCK, gv.BLOCK, gv.PASS),
                             ("draft", gv.BLOCK, gv.BLOCK, gv.PASS)]),
    ])
    assert not hit and not beyond
    assert gv.CLEAN_MARKER in text
    assert HIT_NOT_TOLERATED not in text
    assert "open decision" not in text
    assert "no measured target" in text
    assert "What survives" not in text

    # (c) nothing fed at all: the call must refuse to make a scope claim either way.
    text, beyond, hit, _ = call_text([("fixtures/a.py:1", [("up_next", gv.PASS,
                                                           gv.PASS, gv.PASS)])])
    assert not hit and not beyond
    assert gv.NOT_FED_MARKER in text
    assert "not fed to this harness" in text
    assert "NOT measured by this run" in text
    assert gv.CLEAN_MARKER not in text and gv.HIT_MARKER not in text
    assert "no vacuous status/state class outside {missing-id, paused, draft}" in text, \
        "the count is still zero — what is refused is the scope claim, not the arithmetic"

    # (d) a class outside the triad comes back vacuous: the enumeration earned its
    # keep, and the call must say keep-the-solver, not shrink-it.
    text, beyond, hit, _ = call_text([
        ("fixtures/a.py:1", [("missing-id", gv.BLOCK, gv.BLOCK, gv.PASS),
                             ("backoff", gv.BLOCK, gv.PASS, gv.PASS)]),
    ])
    assert len(beyond) == 1 and not hit
    assert "1 vacuous status/state class(es) outside" in text
    assert "class=backoff" in text
    assert "Keep #559's full Z3 + reference-model + 10k-differential scope" in text
    assert "no vacuous status/state class outside" not in text


def test_scope_call_moves_when_a_class_outside_the_triad_becomes_vacuous():
    """Plant a vacuity on a class #558 never named: the measurement must name it,
    which is what makes the printed call a finding rather than a sentence."""
    guards = _probed()
    target = next((g for g in guards if any(r["status_class"]
                                            and r["class"] not in gv.TRIAD
                                            and not r["diff"] for r in g.results)), None)
    assert target is not None, "no non-triad status class to plant on"
    row = next(r for r in target.results
               if r["status_class"] and r["class"] not in gv.TRIAD)
    row["status"], row["diff"] = "DEAD-BRANCH", False
    beyond, _ = gv.scope_beyond(guards)
    assert any(target.site in b and row["class"] in b for b in beyond), beyond


def test_scope_helper_ignores_classes_that_are_not_status_classes():
    """The SCOPE section is about status/state classes. A dead branch on some
    other axis (a coverage number, a suite count) must not move the #559 call."""
    guards = _probed()
    planted = [(g, r) for g in guards for r in g.results
               if not r["status_class"] and r["status"] == "DEAD-BRANCH"]
    assert planted, "no non-status dead branch to test with"
    before = gv.scope_beyond(guards)
    for g, r in planted:
        r["status"], r["diff"] = "witnessed", True
    assert gv.scope_beyond(guards) == before


def test_json_output_agrees_with_the_text_report(report, guards):
    """One measurement, two renderings: a reader who counts printed lines and a
    reader who parses --json must land on the same number and the same words."""
    proc = subprocess.run([sys.executable, str(SCRIPT), "--json"],
                          capture_output=True, text=True, timeout=180,
                          env=dict(os.environ), cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.loads(proc.stdout)
    assert data["guards"] == len(guards)
    assert data["vacuous"] == sum(1 for g in guards if g.score == gv.VACUOUS)
    assert data["vacuity_rate_line"] in report["stdout"]
    assert data["vacuity_rate_line"].endswith(f"({data['date']})")
    assert data["scope_beyond_triad"] == gv.scope_beyond(guards)[0]
    assert data["dead_branches"], data["dead_branches"]
    for finding in data["findings"]:
        assert finding in report["stdout"], finding[:160]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
