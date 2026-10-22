"""The rails that make `tests/test_path_sandbox.py` unable to read green while saying
nothing (#3133).

`test_path_sandbox.py` is the substrate's only functional proof, and it had three ways
to report all-pass while pinning nothing: an import-time `skipif` on the probe, a
`pytest.skip` inside the corpus loop, and a blindness pin with nothing denied beside it.
All three are LATENT on this box — `/usr/bin/bwrap` and `/usr/bin/perl` both exist and
`_path_sandbox.available()` is `(True, '')` — so the file passing here is not evidence
that these rails work. Every node below FORCES a box state on a child `pytest` process
and asserts what that run reported, the way #1691 forces the frontend pins to be
missing rather than trusting that they are present:

    ~/lloyd/.venvs/lloyd/bin/python -m pytest tests/test_path_sandbox_skip_accounting.py -q -rs

A child process is the only seam that answers the question. "All-pass" is a property of
a RUN — an exit code plus a set of node outcomes — and `pytest.main` in-process shares
this interpreter's already-imported modules, its cached substrate probe and its
collection, so it cannot report "this run would have exited 0". Each node runs the
SHIPPED pin file, reads its exit code and its per-node outcomes out of a junit report
(`-v`'s own lines are unusable here: a parametrized node id like the corpus's is longer
than a non-tty terminal's 80 columns and pytest folds it), and the forcing is a `-p`
plugin written to a scratch directory and loaded into the child before it imports the
pin file — no test-only backdoor is added to the pin file itself, and the scenario's
state never enters this repo's tree.

The four states forced:

  * `_PROBE_PLUGIN` — the substrate probe reports unavailable with a real-looking
    reason. The box the item names ("on a box whose bwrap probe fails").
  * `_NO_PERL_PLUGIN` — `shutil.which` finds everything except `perl`. The missing
    vector dependency.
  * `_NO_BINDS_PLUGIN` — the profile reports itself enforcing while emitting no
    read-only bind. The failure the corpus exists to catch, planted on the far side of
    the seam so the vectors' verdicts are what is under test.
  * a silent `check_bash_command` — the mutation that the blindness pin's positive
    control exists to redden on, which is #1666's instrument (a deny-list that cannot
    go red is not a deny-list) applied to this node.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import substrate_accounting as SA                       # noqa: E402
from app.harness.safety import check_bash_command       # noqa: E402

PIN = "tests/test_path_sandbox.py"
#: A whole run of the pin file drives ~40 commands through real bwrap, so tens of
#: seconds; a timeout is an exception here, never a pass.
TIMEOUT_S = 600

#: The reason the forced probe gives, in the real function's own shape:
#: `_probe()` returns `f"bwrap exited {rc}: {stderr[:200]}"`, which is the branch a
#: userns hiccup takes. Asserted against, so the rail is proven on the text a broken
#: box actually produces rather than on a placeholder.
FORCED_PROBE_ERROR = ("bwrap exited 1: bwrap: creating new namespace failed: "
                      "Operation not permitted")

_PROBE_PLUGIN = f'''"""Make the FIRST `available()` call answer unavailable, later ones the truth.

The scenario verbatim: the item is about a probe that "fails once at collection time",
and the pin file's marker is the first caller. Replacing the function outright would leak
into `test_status_reports_enforcing_or_fallback`, which captures the `P.available` it
finds as its own "real" one and would go red for the forcing rather than for the code — a
rail that fails because of its own harness is a rail the next editor deletes.
"""
from agent_mcp import _path_sandbox as P

_real = P.available
_state = {{"first": True}}


def _forced():
    if _state["first"]:
        _state["first"] = False
        return (False, {FORCED_PROBE_ERROR!r})
    return _real()


P.available = _forced
'''

_NO_PERL_PLUGIN = (
    '"""Make `perl` the only program `shutil.which` cannot find.\n\n'
    "`agent_mcp._path_sandbox.bwrap_path()` resolves `bwrap` through the same call, so "
    "the stub has to pass everything else through or it would take the substrate away "
    'as well and the run would be testing clause 1 instead of clause 2."""\n'
    "import shutil\n\n"
    "_real_which = shutil.which\n\n\n"
    "def _which(cmd, *args, **kwargs):\n"
    '    if str(cmd).rsplit("/", 1)[-1] == "perl":\n'
    "        return None\n"
    "    return _real_which(cmd, *args, **kwargs)\n\n\n"
    "shutil.which = _which\n")

_NO_BINDS_PLUGIN = (
    '"""A profile that builds and binds nothing read-only, for one import-time answer.\n\n'
    "The failure the corpus exists to catch (#3133 clause 2), planted on the far side of "
    "the seam so the vectors\' verdicts are what is under test.\n\n"
    "`ro_bind_argv` is stubbed to bind nothing. `available()` then needs a one-shot True "
    "as well, and the reason is the probe doing its job: a profile that lets a protected "
    "write through is precisely what `_run_probe` exists to notice, so under such a "
    "profile the probe reports the substrate unavailable and the markers skip the whole "
    "corpus — the vectors would never run and the scenario would test nothing. Only the "
    "FIRST call is answered True, which is the marker\'s at import; every later answer is "
    "the probe\'s own. So `wrap()` discovers the hole at each vector, runs the command "
    "bare exactly as it does for a broken namespace, the write goes through, and the "
    "vector fails. `test_status_reports_enforcing_or_fallback`, which captures the "
    "`P.available` it finds and hands it back afterwards, then sees a function that "
    'answers faithfully from here on, and fails only if /state misreports the substrate' \
    '"""\n'
    "from agent_mcp import _path_sandbox as P\n\n"
    "P.ro_bind_argv = lambda entries=None: []\n"
    "_real = P.available\n"
    "_first = [True]\n\n\n"
    "def _available():\n"
    "    global _first\n"
    "    if _first[0]:\n"
    "        _first[0] = False\n"
    "        return (True, '')\n"
    "    return _real()\n\n\n"
    "P.available = _available\n")

_SILENT_MATCHER_PLUGIN = (
    '"""Make the string predicate return None for every command, as a mutation would.\n\n'
    "This is what the blindness pin's positive control exists to catch: twelve "
    "spellings asserted to return None are also what a dead deny-list returns. "
    "Patches the module attribute, which is where the pin node looks it up at call "
    'time."""\n'
    "import app.harness.safety as S\n\n\n"
    "def _blind(command, cwd=None):\n"
    "    return None\n\n\n"
    "S.check_bash_command = _blind\n")

_PLUGINS = {"lloyd_force_probe_off": _PROBE_PLUGIN,
            "lloyd_force_no_perl": _NO_PERL_PLUGIN,
            "lloyd_force_no_binds": _NO_BINDS_PLUGIN,
            "lloyd_force_silent_matcher": _SILENT_MATCHER_PLUGIN}


@pytest.fixture(scope="module")
def pin():
    """The pin file itself, so the corpus, the vector ids and the gate names are read
    off it rather than typed out here.

    #1691's helper makes the same choice for the same reason: a count written into the
    rail that checks the count is the stale-number class that `scripts/automod/gate.py`
    keeps reporting on itself (#1448).
    """
    spec = importlib.util.spec_from_file_location("lloyd_substrate_pin", ROOT / PIN)
    assert spec and spec.loader, f"cannot load {PIN}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _child_env(tmp_path: Path, *, declared: bool) -> dict:
    """The child's environment: this process's, minus three poisons, plus the scenario.

    * `LLOYD_PATH_SANDBOX` is dropped because the parent pytest may itself be running
      INSIDE the substrate — a round's Bash call is sandboxed, and the profile exports
      the marker — and an inherited marker makes the pin file's fail-open node
      (`test_bash_still_runs_when_the_substrate_cannot_build`) see `ran=enforced2` and
      fail for a reason that has nothing to do with anything here. That trap was walked
      into during this round, which is why it is named rather than assumed.
    * `SA.DECLARED_ENV` is dropped unless THIS scenario asks for the declaration: a node
      that inherited the ambient value would be asserting the round it happened to run
      in, not the code. #1691's helper strips its own declaration for the same reason.
    * `PYTEST_ADDOPTS` is dropped so a caller's flags cannot change what a rail
      measured, and `PYTHONPATH` gains the scratch directory holding the plugins, which
      is what the `-p` names refer to.
    """
    env = dict(os.environ)
    env.pop("LLOYD_PATH_SANDBOX", None)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop(SA.DECLARED_ENV, None)
    if declared:
        env[SA.DECLARED_ENV] = "1"
    paths = [str(tmp_path)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run_pin(tmp_path: Path, *plugins: str, declared: bool = False,
             keep: str = ""):
    """Run the shipped pin file in a child pytest.

    `(completed, outcomes, report_text)` where `outcomes` is
    `{node name: (outcome, detail)}` and `detail` is the skip reason or the failure text.
    """
    for name in plugins:
        src = _PLUGINS.get(name)
        assert src, f"no such forced-state plugin: {name}"
        (tmp_path / f"{name}.py").write_text(src)
    xml = tmp_path / "report.xml"
    cmd = [sys.executable, "-m", "pytest", PIN, "-q", "-p", "no:cacheprovider"]
    for name in plugins:
        cmd += ["-p", name]
    if keep:
        cmd += ["-k", keep]
    cmd += [f"--junitxml={xml}"]
    done = subprocess.run(cmd, cwd=str(ROOT),
                          env=_child_env(tmp_path, declared=declared),
                          capture_output=True, text=True, timeout=TIMEOUT_S,
                          check=False)
    outcomes = _outcomes(xml)
    assert outcomes, \
        f"the child run reported no node outcomes (exit {done.returncode}); the rail " \
        f"measured nothing:\n{(done.stdout + done.stderr)[-1500:]}"
    return done, outcomes, done.stdout + done.stderr


def _outcomes(xml_path: Path) -> "dict[str, tuple[str, str]]":
    """`{node name: (outcome, detail)}` out of the child's junit report."""
    if not xml_path.exists():
        return {}
    out: dict[str, tuple[str, str]] = {}
    for case in ET.parse(str(xml_path)).getroot().iter("testcase"):
        outcome, detail = "passed", ""
        for child in case:
            if child.tag == "skipped":
                outcome, detail = "skipped", child.get("message") or ""
            elif child.tag in ("failure", "error"):
                outcome = "failed"
                detail = f"{child.get('message') or ''}\n{child.text or ''}"
        out[case.get("name") or "?"] = (outcome, detail)
    return out


def _corpus(outcomes: "dict[str, tuple[str, str]]", pin,
            name: str) -> "dict[str, tuple[str, str]]":
    """The corpus nodes only, keyed by their parametrize id."""
    stem = getattr(pin, name).__name__
    return {node.split("[", 1)[1].rstrip("]"): rest
            for node, rest in outcomes.items() if node.startswith(stem + "[")}


CORPUS_NODE = "test_a_corpus_vector_is_refused_and_its_bytes_are_unchanged"
BLINDNESS_NODE = ("test_the_string_predicate_is_still_blind_to_the_shapes_the_"
                  "substrate_refuses")


def _perl_id(pin) -> str:
    """The parametrize id of the one vector that names a dependency, read off CORPUS."""
    rows = [row for row in pin.CORPUS if len(row) > 3 and row[3] == "perl"]
    assert len(rows) == 1, \
        f"{len(rows)} vectors name a `perl` dependency; the forcing plugin lies about " \
        "exactly one program, so it needs exactly one vector to be about that program"
    return pin._vector_id(rows[0])


# ── clause 1: a mass substrate skip is a loud non-pass ─────────────────────

def test_an_unavailable_substrate_probe_is_a_failure_naming_the_count(tmp_path, pin):
    """Probe off → the run exits non-zero and names how many substrate nodes did not run.

    Before this change the same forced run ended `18 passed, 10 skipped`, exit 0: the
    substrate's whole functional proof gone, with nothing between it and a green board
    but a suite-wide skip budget with nine skips of headroom and no floor at all on the
    gate's partial-run branch.
    """
    done, outcomes, text = _run_pin(tmp_path, "lloyd_force_probe_off")
    assert done.returncode != 0, \
        "a run in which not one substrate node executed exited 0 — the silent-green " \
        f"path #3133 exists to close:\n{text[-1500:]}"
    m = re.search(rf"{SA.FINDING}: (\d+) of (\d+) substrate nodes", text)
    assert m, \
        f"the run does not name {SA.FINDING} with its counts:\n{text[-1500:]}"
    pending, total = int(m.group(1)), int(m.group(2))
    skipped = sorted(n for n, (o, _d) in outcomes.items()
                     if o == "skipped" and n.split("[")[0] in pin.SUBSTRATE.gated)
    assert pending >= 10, \
        f"the finding names {pending} substrate nodes; the file gates ten functions, " \
        "so a run in which none of them executed cannot name fewer"
    assert pending == len(skipped), \
        f"the finding names {pending} substrate nodes while {len(skipped)} of this " \
        f"run's substrate nodes skipped — the count is not read off the run: {skipped}"
    assert total >= pending, \
        f"the finding's denominator {total} is below the {pending} nodes it names"
    assert outcomes[SA.GATE_NODE][0] == "failed", \
        f"{SA.GATE_NODE} reported {outcomes[SA.GATE_NODE][0]}, not failed: a mass skip " \
        "has to fail THROUGH this node, not warn beside it"
    assert "did not run" in text, \
        f"the failure does not say the nodes did not run: " \
        f"{[ln for ln in text.splitlines() if SA.FINDING in ln][:3]}"
    # …and each skip's own reason carries the probe's explanation (#3133 clause 4's
    # "a single substrate gate whose skip reason carries the probe's own explanation",
    # witnessed on the box where it actually bites).
    unexplained = {n: d[:140] for n, (_o, d) in outcomes.items()
                   if n.split("[")[0] in pin.SUBSTRATE.gated
                   and FORCED_PROBE_ERROR not in d}
    assert not unexplained, \
        f"{len(unexplained)} substrate nodes skipped without the probe's explanation in " \
        f"the reason: {unexplained}"


def test_a_declared_unavailable_substrate_is_the_only_quiet_skip(tmp_path, pin):
    """The declaration seam is the ONLY thing that turns that run quiet (#3133 clause 1).

    Mirrors `FRONTEND_PINS_ENV` (`scripts/automod/gate.py:189-195`): a runner that has
    decided this box is meant to lack a namespace says so, and gets skips it can read.
    The declared route is a SKIP, never a pass — a pass would claim the substrate nodes
    executed, which is the sentence the whole item exists to make unwritable.
    """
    done, outcomes, text = _run_pin(tmp_path, "lloyd_force_probe_off", declared=True)
    assert done.returncode == 0, \
        f"a declared run went red anyway:\n{text[-1500:]}"
    assert outcomes[SA.GATE_NODE][0] == "skipped", \
        f"{SA.GATE_NODE} reported {outcomes[SA.GATE_NODE][0]}; the declared route is a " \
        "skip, not a pass"
    assert SA.DECLARED_ENV in outcomes[SA.GATE_NODE][1], \
        f"the skip does not name the declaration that bought it: " \
        f"{outcomes[SA.GATE_NODE][1][:300]}"
    assert SA.FINDING not in text, \
        "the declared skip still printed the finding: the declaration is meant to be " \
        "the difference between the loud and the quiet state, not a cosmetic one"


def test_the_declaration_seam_is_named_by_exactly_its_own_three_files():
    """No fourth file may set `LLOYD_PATH_SUBSTRATE_AVAILABLE`.

    This rail is the whole safety of the seam: a mass skip goes quiet only for a runner
    that DECLARED it, so a runner added casually — a CI job, a gate branch, a local
    script — would re-open the hole with every node in this file still passing. The
    positive control discipline applies to the grep itself: `git grep` exits 1 on zero
    hits, which is asserted here, so a pattern that matched nothing cannot masquerade as
    "clean".
    """
    listed = subprocess.run(["git", "-C", str(ROOT), "grep", "-l", SA.DECLARED_ENV],
                            capture_output=True, text=True, check=False)
    assert listed.returncode == 0, \
        f"`git grep {SA.DECLARED_ENV}` itself found nothing ({listed.returncode}): " \
        f"{listed.stderr[:200]} — a zero-hit grep proves only that the pattern or the " \
        "path failed"
    files = sorted(ln.strip() for ln in listed.stdout.splitlines() if ln.strip())
    assert PIN in files, \
        f"{PIN} names {SA.DECLARED_ENV} and did not appear in `git grep`'s output, so " \
        f"this rail is not reading the tree it thinks it is reading: {files}"
    allowed = {"tests/substrate_accounting.py", PIN,
               "tests/test_path_sandbox_skip_accounting.py"}
    assert set(files) <= allowed, \
        f"{SA.DECLARED_ENV} is named by {sorted(set(files) - allowed)} beyond the " \
        "helper, the pin file and this rail. A file naming it is a runner that can " \
        "declare the substrate away, which is a decision someone records, not a " \
        "default something inherits. (`git grep` reads tracked files, so an untracked " \
        "scratch script setting it is exactly how this hole reopens unseen — that gap " \
        "is stated rather than papered over, because `--untracked` would make the rail " \
        "answer to whatever a stranger has lying in their worktree.)"


# ── clause 2: one missing dependency costs one vector, not the corpus ──────

def test_the_corpus_is_fifteen_nodes_not_one(tmp_path, pin):
    """Each vector is its own node — measured on collection, not on the source.

    `git grep -n parametrize -- tests/test_path_sandbox.py` returning nothing was the
    triage's proof of the bug; this is the proof of the fix. A collected count is used
    rather than the presence of a decorator because a `parametrize` whose `ids=` collide
    still greps well and still collapses vectors into one another.
    """
    assert len(pin.CORPUS) == 15, \
        f"the corpus holds {len(pin.CORPUS)} vectors; the clause is stated over 15, so " \
        "this rail and the corpus have to move together and deliberately"
    ids = [pin._vector_id(row) for row in pin.CORPUS]
    assert len(set(ids)) == len(ids), \
        f"two vectors collide on a parametrize id: {sorted(ids)}"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", PIN, "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT), env=_child_env(tmp_path, declared=False),
        capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    collected = [ln.strip().split("::", 1)[1]
                 for ln in (done.stdout + done.stderr).splitlines()
                 if ln.strip().startswith(f"{PIN}::")]
    assert collected, "collection named no nodes at all, so this rail measured nothing"
    corpus = sorted(n for n in collected if n.startswith(f"{CORPUS_NODE}["))
    assert len(corpus) == len(pin.CORPUS), \
        f"{len(corpus)} corpus nodes collected for {len(pin.CORPUS)} vectors: " \
        f"{corpus}"
    assert sorted(n[len(CORPUS_NODE) + 1:-1] for n in corpus) == sorted(ids), \
        f"the collected ids are not the corpus's: {corpus}"


def test_one_missing_program_skips_one_vector_and_keeps_fourteen_verdicts(tmp_path,
                                                                         pin):
    """`perl` absent: exactly that vector skips, the other fourteen still report PASSED.

    The old shape reported SKIPPED for the whole corpus on a box missing that one
    program, discarding the fourteen verdicts the loop had already collected.
    """
    perl = _perl_id(pin)
    done, outcomes, text = _run_pin(tmp_path, "lloyd_force_no_perl")
    vectors = _corpus(outcomes, pin, CORPUS_NODE)
    assert len(vectors) == len(pin.CORPUS), \
        f"{len(vectors)} corpus nodes in the run: {sorted(vectors)}"
    skipped = sorted(n for n, (o, _d) in vectors.items() if o == "skipped")
    assert skipped == [perl], \
        f"the missing `perl` cost {skipped}; it is the only program this run removed, " \
        f"so it may cost only the vector that names it ({perl})"
    assert "perl" in vectors[perl][1], \
        f"the one skip does not name the program it needs: {vectors[perl][1][:200]}"
    passed = sorted(n for n, (o, _d) in vectors.items() if o == "passed")
    assert len(passed) == len(pin.CORPUS) - 1, \
        f"{len(passed)} of the {len(pin.CORPUS) - 1} vectors that need nothing reported " \
        f"PASSED; a vector lost to another reason is a verdict lost the same way: " \
        f"{ {n: o for n, (o, _d) in vectors.items() if o != 'passed'} }"
    assert done.returncode == 0, \
        f"every vector either passed or declared-skipped, yet the run failed:\n" \
        f"{text[-1500:]}"


def test_a_write_that_goes_through_fails_even_beside_the_one_skipped_vector(tmp_path,
                                                                           pin):
    """The half the old loop destroyed: a write-through AND a dependency skip in one run.

    Pre-fix, the loop reached the perl vector, skipped, and pytest overwrote the node's
    recorded outcome with SKIPPED — the standalone repro of that shape reported
    `3 skipped`, exit 0, with "the substrate let the write through" already in the
    failures list. The vectors' own failure text is asserted, so this cannot be
    satisfied by nodes going red for an unrelated reason.
    """
    perl = _perl_id(pin)
    done, outcomes, text = _run_pin(tmp_path, "lloyd_force_no_perl",
                                    "lloyd_force_no_binds")
    assert done.returncode != 0, \
        "a substrate that bound nothing and let the vectors write exited 0:\n" \
        f"{text[-1500:]}"
    vectors = _corpus(outcomes, pin, CORPUS_NODE)
    assert len(vectors) == len(pin.CORPUS), f"{len(vectors)} corpus nodes in the run"
    failed = sorted(n for n, (o, _d) in vectors.items() if o == "failed")
    skipped = sorted(n for n, (o, _d) in vectors.items() if o == "skipped")
    passed = sorted(n for n, (o, _d) in vectors.items() if o == "passed")
    assert skipped == [perl], \
        f"the skips are {skipped}: the dependency skip has to survive as the one skip " \
        "while every other vector's verdict is kept"
    assert not passed, \
        f"with no read-only bind at all, these vectors still reported a refusal: {passed}"
    assert len(failed) >= len(pin.CORPUS) - 2, \
        f"with the binds removed only {len(failed)} vectors failed: the per-vector " \
        f"nodes are not reporting the hole: { {n: o for n, (o, _d) in vectors.items()} }"
    assert any("the substrate let the write through" in vectors[n][1] for n in failed), \
        "vectors went red but none of them said the write went through, so they are " \
        f"failing for something else: {[vectors[n][1][:200] for n in failed[:3]]}"


# ── clause 3: the blindness pin cannot be satisfied by a silent matcher ────

def test_a_matcher_that_denies_nothing_reddens_the_blindness_pin(tmp_path):
    """`check_bash_command` forced to return None for everything → the pin goes RED.

    The literal statement of clause 3, and the instrument #1666 added for the bench
    deny-list: a pin that cannot go red is not a pin. Only the blindness node is run, so
    the mutation is confined to the node whose claim it contradicts.
    """
    done, outcomes, text = _run_pin(tmp_path, "lloyd_force_silent_matcher",
                                    keep=BLINDNESS_NODE)
    assert outcomes[BLINDNESS_NODE][0] == "failed", \
        f"with the matcher returning None for every command the blindness node reported " \
        f"{outcomes[BLINDNESS_NODE][0]}: the pin still proves only that the function " \
        f"under test never answers\n{text[-1200:]}"
    assert "DENIED" in outcomes[BLINDNESS_NODE][1], \
        f"the blindness node went red without naming its control, so the failure does " \
        f"not tell the next reader which half broke: " \
        f"{outcomes[BLINDNESS_NODE][1][:400]}"
    assert done.returncode != 0, \
        f"the blindness node failed and the run still exited {done.returncode}"


def test_the_blindness_pin_asserts_its_control_before_the_none_loop(pin):
    """The control is INSIDE the blindness node, ahead of the twelve, and denies today.

    Structural, and paired with the mutation node above: this one says the control is in
    the place a reader of the pin sees it and cannot mistake for one of the twelve, and
    that the two commands it names are genuinely denied by the real matcher. The
    behavioural half — that they are LOAD-BEARING — is
    `test_a_matcher_that_denies_nothing_reddens_the_blindness_pin`, because a control
    asserted after the loop, or one the matcher is also blind to, would leave this file
    exactly as blind as it was.
    """
    src = (ROOT / PIN).read_text()
    node = [n for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name == BLINDNESS_NODE]
    assert node, f"{BLINDNESS_NODE} is not a module-level function of {PIN}"
    body = ast.get_source_segment(src, node[0]) or ""
    assert "POSITIVE CONTROL" in body, \
        f"{BLINDNESS_NODE} asserts no positive control at all"
    assert body.index("POSITIVE CONTROL") < body.index("assert not still_blind"), \
        "the control is asserted after the None-loop: read top-down, a control that " \
        "comes last is a footnote"
    assert body.count("is None") >= 1, "the None assertions are gone from the node"

    home = os.path.expanduser("~")
    for cmd in (f"rm -rf {home}", f"rm -rf {home}/obsidian"):
        reason = check_bash_command(cmd, cwd="/tmp")
        assert reason is not None, \
            f"the control {cmd!r} is NOT denied by the real matcher (it returns None), " \
            "so it is decoration and cannot hold the blindness pin up"


# ── clause 4: one gate, and its reason is the probe's own words ────────────

def test_the_pin_file_has_one_substrate_gate_and_no_dead_marker(pin):
    """`_HAS_BWRAP` / `needs_bwrap` are gone and exactly one gate is left (#3133 clause 4).

    Static on purpose: the dead marker was dead on every box, so the rail against it
    must not itself be gated on the substrate the marker claimed to detect.
    """
    src = (ROOT / PIN).read_text()
    tree = ast.parse(src)
    assigned = {t.id for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert not assigned & {"_HAS_BWRAP", "needs_bwrap"}, \
        f"the dead `shutil.which` marker is back " \
        f"({sorted(assigned & {'_HAS_BWRAP', 'needs_bwrap'})}): a marker no node " \
        "applies is decoration that reads as a second guard, and the box state it " \
        "claims to detect is not the one the file keys on"

    gates = [n for n in tree.body if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call)
             and "skipif" in ast.unparse(n.value)]
    assert len(gates) == 1, \
        f"{len(gates)} module-level skipif markers: one file, one substrate gate, and " \
        f"a second gate is a second box state to explain: " \
        f"{[ast.unparse(g.targets[0]) for g in gates]}"
    gate = gates[0]
    assert ast.unparse(gate.targets[0]) == SA.SUBSTRATE_GATE, \
        f"the surviving gate is named {ast.unparse(gate.targets[0])}, which is the name " \
        f"the ledger counts decorated nodes by ({SA.SUBSTRATE_GATE})"
    marker_src = ast.unparse(gate.value)
    assert "cannot build a namespace here" in marker_src, \
        f"the gate's reason does not name the substrate probe: {marker_src}"
    assert "_SUBSTRATE_WHY" in marker_src, \
        "the gate's reason is not the value the probe wrote, so the skip would print " \
        f"a reason the kernel never gave: {marker_src}"
    assert marker_src.count("_SUBSTRATE_OK") == 1, \
        "the marker calls the probe more than once, so the skip condition and the " \
        f"printed reason can be two answers 60 s apart: {marker_src}"
    probed = [n for n in tree.body if isinstance(n, ast.Assign)
              and "available()" in ast.unparse(n.value)]
    assert len(probed) == 1, \
        f"{len(probed)} module-level calls to `available()`: the file probes the kernel " \
        "more than once at import, so two nodes can key on two answers of the same " \
        "60-second cache"


def test_the_ledgers_voice_is_the_last_node_of_the_pin_file(pin):
    """The reporting node runs after the nodes it counts, and its position is a test.

    pytest runs a module in source order: move the ledger node above the gated functions
    and a perfectly healthy run reports "all of them did not run". A rail that fires
    only when everything worked is a rail the next editor deletes in annoyance, so the
    order is pinned rather than commented.
    """
    src = (ROOT / PIN).read_text()
    tree = ast.parse(src)
    names = [n.name for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    assert names, f"no test functions found in {PIN} at all"
    assert names[-1] == SA.GATE_NODE, \
        f"the last test function of {PIN} is {names[-1]}, so {SA.GATE_NODE} reports " \
        "before the substrate nodes have entered their bodies — it has to stay last"
    gate = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == SA.GATE_NODE][0]
    assert not gate.decorator_list, \
        f"{SA.GATE_NODE} gained a decorator; if it can be skipped the ledger has no " \
        "voice, which is the whole of #3133"
