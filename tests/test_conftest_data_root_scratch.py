"""#4239 — the suite's substitute `LLOYD_DATA` may never land inside the live data root.

`tests/conftest.py::_data_root_to_scratch` hands every pytest run a scratch data root
so no fixture writes into `~/lloyd-data`, and it refuses a caller who *names* the live
root. What it never checked is where `tempfile.mkdtemp` put the replacement — and
`mkdtemp` honours `$TMPDIR`. A run launched with
`TMPDIR=/home/alansrobotlab/lloyd-data/tmp/triage-4236` therefore created
`lloyd-test-data-w0xh1qbg/workers.db` (7 tables, 0 data rows) four directories deep
INSIDE the root the guard exists to keep the suite out of, and the guardian's
`data_root_shadows` check filed it as a database shadowing the populated
`/home/alansrobotlab/lloyd-data/workers.db` (32,935,936 bytes) — high priority, with a
remedy telling a person to consider deleting a test fixture. Its youngest rail is
`INERT_MIN_AGE_SECONDS = 600.0` (`agent-services/guardian/datawatch.py:334`), so any
suite run with a data-root `TMPDIR` re-creates the alert ten minutes later. The
`atexit` teardown that should have taken the tree with it did not run because the
pytest process died, and nothing owns `~/lloyd-data/tmp`: `git grep -n
"lloyd-data/tmp"` returns nothing, and the retention sweep's tmp leg runs over
`tempfile.gettempdir()`.

So the fix is at the chooser, not at the residue.
`_test_scratch_dir` walks the same candidate list `tempfile` itself consults —
`TMPDIR`, `TEMP`, `TMP`, then the platform defaults — and takes the first that is a
usable directory not under the live root, so the class cannot be re-created. When
`TMPDIR` is already outside the live root nothing changes, which is the property the
gate depends on: its children get a deliberately SHORT `TMPDIR`
(`scripts/automod/gate.py::MAX_CHILD_TMPDIR` = 48) because Chromium's singleton socket
lands 45 bytes past it, so moving the scratch root somewhere else on pain of a long
socket path would break every browser test.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

#: The prefix `_data_root_to_scratch` gives the run's substitute data root, and the
#: name the alert was filed on. Named here so a node can ask "did any survive?".
DATA_SCRATCH_PREFIX = "lloyd-test-data-"


def _conftest():
    """The real conftest, loaded under its own name so patching it is local."""
    spec = importlib.util.spec_from_file_location(
        "lloyd_data_scratch_conftest", REPO / "tests" / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _under(candidate: Path, root: Path) -> bool:
    """Is `candidate` `root` or below it? Both sides resolved, or a symlinked
    `/tmp` answers the question differently on the two halves of the comparison."""
    return candidate.resolve() == root.resolve() or \
        candidate.resolve().is_relative_to(root.resolve())


@pytest.fixture
def swept():
    """Directories a node created for real, removed again on the way out.

    `_test_scratch_dir` calls `mkdtemp`, which creates. A node that only asserted
    where the answer pointed would still leave the directory behind, and the whole
    point of the item is a scratch tree nobody owned.
    """
    created: list[Path] = []
    yield created
    for path in created:
        shutil.rmtree(path, ignore_errors=True)


# ── clause 1: the chooser never answers a path inside the live root ─────────

def test_the_scratch_helper_skips_a_tmpdir_inside_the_live_root(tmp_path, monkeypatch,
                                                                swept):
    """All three temp-dir variables aimed inside the live root: the answer is outside it.

    This is the shape the alert was filed on — `TMPDIR` four directories under the
    data root — with `TEMP` and `TMP` pinned too, so the node cannot pass by the
    helper reading only one variable and a fall-through passing for a decision.
    """
    c = _conftest()
    live = tmp_path / "lloyd-data"
    inner = live / "tmp" / "triage-4236"
    inner.mkdir(parents=True)
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(inner))

    got = c._test_scratch_dir(DATA_SCRATCH_PREFIX, live.resolve())
    swept.append(got)

    assert got.is_dir(), f"{got} was never created"
    assert got.name.startswith(DATA_SCRATCH_PREFIX), got
    assert not _under(got, live), f"the substitute root sits inside the live root: {got}"


def test_the_scratch_helper_takes_the_next_candidate_rather_than_a_hardcoded_one(
        tmp_path, monkeypatch, swept):
    """`TMPDIR` inside the live root, `TEMP` outside it: the helper follows
    `tempfile`'s own order and lands under `TEMP`.

    The list order is the whole of `tempfile._candidate_tempdir_list`, so a helper
    that jumped straight to `/tmp` would ignore a caller who had already moved the
    run's temp tree — and on a box where `/tmp` is the inode-starved tmpfs of
    2026-09-22 memory, that is the difference between a working run and an ENOSPC.
    """
    c = _conftest()
    live = tmp_path / "lloyd-data"
    inner = live / "tmp" / "triage-4236"
    outside = tmp_path / "short-tmp"
    inner.mkdir(parents=True)
    outside.mkdir()
    monkeypatch.setenv("TMPDIR", str(inner))
    monkeypatch.setenv("TEMP", str(outside))

    got = c._test_scratch_dir(DATA_SCRATCH_PREFIX, live.resolve())
    swept.append(got)

    assert got.parent == outside.resolve(), f"did not fall through to TEMP: {got}"


def test_a_live_root_that_holds_every_candidate_is_refused_not_written_into(
        tmp_path, monkeypatch):
    """The absolute half of "never answers a path inside the live data root".

    With `live` at the filesystem root every candidate is inside it, and the only
    honest answer is a refusal that names the variable to move. Silence here would
    mean a guard that reports success while having written into the thing it
    protects — the failure this item is about. Unreachable on this box, where the
    live root is `$HOME/lloyd-data`; reachable in exactly one way, by an account
    whose data root IS the filesystem root.
    """
    c = _conftest()
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    with pytest.raises(pytest.UsageError) as got:
        c._test_scratch_dir(DATA_SCRATCH_PREFIX, Path("/"))
    assert "TMPDIR" in str(got.value), "the refusal must name the variable to move"


def test_an_unusable_tmpdir_is_skipped_the_way_tempfile_skips_it(tmp_path, monkeypatch,
                                                                 swept):
    """A `TMPDIR` that does not exist is not a place to create anything.

    `tempfile.gettempdir` ignores a candidate that is not a usable directory and
    keeps walking, and the helper consults the same list, so it has to make the same
    judgement — otherwise a stale `TMPDIR` turns every suite run into an `OSError`
    from `mkdtemp`.
    """
    c = _conftest()
    live = tmp_path / "lloyd-data"
    live.mkdir()
    monkeypatch.setenv("TMPDIR", str(live / "does-not-exist"))
    monkeypatch.delenv("TEMP", raising=False)
    monkeypatch.delenv("TMP", raising=False)

    got = c._test_scratch_dir(DATA_SCRATCH_PREFIX, live.resolve())
    swept.append(got)

    assert got.is_dir(), got
    assert not _under(got, live), got


# ── clause 2: a TMPDIR outside the live root still holds the root ───────────

def test_a_tmpdir_outside_the_live_root_still_holds_the_substitute_root(
        tmp_path, monkeypatch, swept):
    """Nothing moves when the run's temp dir is already elsewhere.

    The gate hands every child a deliberately short `TMPDIR`
    (`scripts/automod/gate.py::MAX_CHILD_TMPDIR` = 48, because Chromium's singleton
    socket is 45 bytes past it), and the suite's scratch root lives under that. If
    this node starts failing, the substitute root has moved out of the short tree
    and the browser tests are about to die on `sun_path`.
    """
    c = _conftest()
    live = tmp_path / "lloyd-data"
    short = tmp_path / ".t" / "05431072c3"      # the gate's own layout
    live.mkdir()
    short.mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(short))

    got = c._test_scratch_dir(DATA_SCRATCH_PREFIX, live.resolve())
    swept.append(got)

    assert got.parent == short.resolve(), f"left TMPDIR: {got}"
    # TMPDIR + "/" + prefix + mkdtemp's 8-character suffix, and nothing else: the
    # 48-byte budget a gate child's TMPDIR is capped at has to cover it.
    assert len(str(got)) <= len(str(short)) + 1 + len(DATA_SCRATCH_PREFIX) + 8, \
        f"the scratch path grew past the short-temp budget: {got}"


def test_a_live_root_that_cannot_be_named_still_uses_tmpdir(tmp_path, monkeypatch,
                                                            swept):
    """`None` from the live-root lookup means "nothing to refuse", not "nowhere to go".

    `_live_data_root` answers `None` when the passwd entry cannot be read, the same
    shape as `_production_tree`. A guard that read that as an emergency would
    refuse every run on a box with an odd account, which is the worse failure.
    """
    c = _conftest()
    short = tmp_path / "short-tmp"
    short.mkdir()
    monkeypatch.setenv("TMPDIR", str(short))

    got = c._test_scratch_dir(DATA_SCRATCH_PREFIX, None)
    swept.append(got)

    assert got.parent == short.resolve(), got


# ── clause 3: still exported as LLOYD_DATA, still registered for removal ────

def test_the_chosen_root_is_exported_as_lloyd_data_and_registered_for_removal(
        tmp_path, monkeypatch, swept):
    """The two things the caller does with the helper's answer, unchanged.

    `LLOYD_DATA` is what `app.paths.DATA_ROOT` resolves from at import, so an
    answer that is never exported isolates nothing; the `atexit`
    `shutil.rmtree(..., ignore_errors=True)` is what stops the tree outliving the
    run. Both are asserted on the registration and the environment, not on a
    comment saying they happen.
    """
    c = _conftest()
    live = tmp_path / "lloyd-data"
    inner = live / "tmp" / "triage-4236"
    outside = tmp_path / "short-tmp"
    inner.mkdir(parents=True)
    outside.mkdir()
    monkeypatch.setattr(c, "_live_data_root", lambda: live.resolve())
    monkeypatch.setenv("TMPDIR", str(inner))
    monkeypatch.setenv("TEMP", str(outside))
    monkeypatch.delenv("LLOYD_DATA", raising=False)
    registered: list[tuple] = []
    monkeypatch.setattr(c, "atexit",
                        type("FakeAtExit", (), {
                            "register": staticmethod(
                                lambda fn, *a, **kw: registered.append((fn, a, kw)))}))

    c._data_root_to_scratch()

    chosen = Path(os.environ["LLOYD_DATA"])
    swept.append(chosen)
    assert chosen.is_dir(), os.environ["LLOYD_DATA"]
    assert not _under(chosen, live), f"inside the live root: {chosen}"
    assert registered == [(shutil.rmtree, (chosen,), {"ignore_errors": True})], \
        f"the substitute root is not registered for removal: {registered}"


def test_data_root_to_scratch_never_leaves_the_substitute_inside_the_live_root(
        tmp_path, monkeypatch, swept):
    """Clause 1 through the function the suite actually calls, not just its helper.

    A helper that is correct and a caller that ignores it is the shape of this bug:
    the guard was right about the root it refuses and blind about the root it chose.
    """
    c = _conftest()
    live = tmp_path / "lloyd-data"
    inner = live / "tmp" / "triage-4236"
    outside = tmp_path / "short-tmp"
    inner.mkdir(parents=True)
    outside.mkdir()
    monkeypatch.setattr(c, "_live_data_root", lambda: live.resolve())
    monkeypatch.setenv("TMPDIR", str(inner))
    monkeypatch.setenv("TEMP", str(outside))
    monkeypatch.delenv("LLOYD_DATA", raising=False)

    c._data_root_to_scratch()

    chosen = Path(os.environ["LLOYD_DATA"])
    swept.append(chosen)
    assert chosen.is_dir()
    assert not _under(chosen, live), chosen
    assert not list(live.rglob(f"{DATA_SCRATCH_PREFIX}*")), \
        "a substitute data root survived inside the live data root"


# ── the refusal and the passthrough the refactor must not lose ──────────────

def test_a_caller_that_named_the_live_data_root_is_still_refused(tmp_path, monkeypatch):
    """The pre-existing refusal survives the change, message and all.

    `LLOYD_DATA=<live>` without the opt-in has to stay a `pytest.UsageError` that
    names the opt-in — a guard whose way past it is undocumented gets worked around
    by deleting the guard, and this refusal is the only thing standing between a
    hand-run suite and the machine's sessions and `workers.db`.
    """
    c = _conftest()
    live = tmp_path / "lloyd-data"
    live.mkdir()
    monkeypatch.setattr(c, "_live_data_root", lambda: live.resolve())
    monkeypatch.setenv("LLOYD_DATA", str(live))
    monkeypatch.delenv(c.LIVE_TREE_OPT_IN, raising=False)

    with pytest.raises(pytest.UsageError) as got:
        c._data_root_to_scratch()
    assert c.LIVE_TREE_OPT_IN in str(got.value), str(got.value)


def test_a_caller_that_chose_its_own_scratch_root_keeps_it(tmp_path, monkeypatch):
    """The gate names the round's data root, and conftest must not second-guess it.

    Only the LIVE root is refused; a scratch root of the caller's — which is what
    `gate._child_env` sets for every rung — is kept verbatim, and no directory is
    created. Redirecting it would move the gate's own baseline out from under it.
    """
    c = _conftest()
    live = tmp_path / "lloyd-data"
    chosen = tmp_path / "round-data"
    live.mkdir()
    chosen.mkdir()
    monkeypatch.setattr(c, "_live_data_root", lambda: live.resolve())
    monkeypatch.setenv("LLOYD_DATA", str(chosen))

    c._data_root_to_scratch()

    assert os.environ["LLOYD_DATA"] == str(chosen)
    assert list(chosen.iterdir()) == [], "the guard created files in the caller's root"


def test_the_guard_is_wired_at_conftest_import():
    """Defined and never called is the shape of the bug it fixes.

    Same node shape as `tests/test_live_tree_isolation.py::test_the_refusal_is_wired_at_conftest_import`:
    everything here depends on `LLOYD_DATA` being set BEFORE a test module imports
    `app.paths`, and an import-time call is the only place that is.
    """
    src = (REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "\n_data_root_to_scratch()\n" in src, \
        "the guard must run at conftest import, not merely be defined"


# ── across the process boundary: a real pytest run, a data-root TMPDIR ──────

def test_a_real_pytest_run_with_tmpdir_in_the_live_root_leaves_no_scratch_there(tmp_path):
    """The item's acceptance check, run for real in a child process.

    A staged tree holding a copy of THIS conftest, whose live-root lookup the node
    aims at a fake `~/lloyd-data` under `tmp_path`, and `TMPDIR` pointed inside that
    fake root — the exact state that left
    `tmp/triage-4236/lloyd-test-data-w0xh1qbg/workers.db` on the machine. The child's
    own node asserts the `LLOYD_DATA` its conftest chose resolves outside the fake
    live root; this node then asserts nothing carrying the data-scratch prefix
    survived anywhere under it.

    The lookup is re-pointed by rewriting the copied source, the way
    `tests/test_live_tree_isolation.py::test_the_refusal_fires_in_a_real_pytest_run`
    re-points `ROOT`, because the account anchor cannot be moved any other way from
    here: a stub `pwd` module on `PYTHONPATH` was tried first and does not work —
    `import pwd` inside `_live_data_root` answers whatever the interpreter already
    loaded, and by the time conftest runs something else has imported the real one.
    The rewrite is asserted rather than assumed, so a conftest whose lookup no longer
    reads like this fails the node instead of silently aiming the child at the real
    data root.

    The child then kills itself with `os._exit`, which is not theatre: on the machine
    the residue was left because the pytest process DIED and the `atexit`
    `shutil.rmtree` never ran. A child that exited cleanly would remove its own
    scratch root and pass this node with or without the fix — measured, the
    pre-change conftest staged the same way and left nothing behind, which is exactly
    how the guard looked green right up until the run that was killed. So the node
    asserts on the path the child recorded before it died as well as on the residue,
    and neither claim rests on the other.

    `_default_state_dirs_to_scratch` still writes its own `lloyd-test-state-*` under
    a data-root `TMPDIR` — the same hole, a different store, recorded on #4239 rather
    than widened here — so the assertion is on the prefix this item names.
    """
    live = tmp_path / "fake-home" / "lloyd-data"
    inner_tmp = live / "tmp" / "triage-4236"
    stage = tmp_path / "stage"
    (stage / "tests").mkdir(parents=True)
    inner_tmp.mkdir(parents=True)
    src = (REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
    anchor = "def _live_data_root() -> Path | None:"
    assert src.count(anchor) == 1, \
        "the staged run re-points the live-root lookup by name; that name changed"
    src = src.replace(anchor, anchor + "\n"
                                  "    return Path(os.environ[\"FAKE_LIVE_DATA_ROOT\"]).resolve()\n\n\n"
                                  "def _live_data_root_unpatched() -> Path | None:")
    (stage / "tests" / "conftest.py").write_text(src, encoding="utf-8")
    record = tmp_path / "chosen.json"
    (stage / "tests" / "test_probe.py").write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n\n\n"
        "def test_lloyd_data_is_outside_the_live_root():\n"
        "    live = Path(os.environ['FAKE_LIVE_DATA_ROOT']).resolve()\n"
        "    chosen = Path(os.environ['LLOYD_DATA']).resolve()\n"
        "    Path(os.environ['CHOSEN_RECORD']).write_text(json.dumps(str(chosen)))\n"
        "    assert chosen != live, chosen\n"
        "    assert not chosen.is_relative_to(live), chosen\n"
        "    os._exit(0)  # skip atexit: the residue this item is about needs a run "
        "                # that died, not one that tidied up\n",
        encoding="utf-8")

    env = {**os.environ,
           "TMPDIR": str(inner_tmp),
           # This checkout, because the conftest under test has autouse fixtures that
           # import `scripts.automod.state`, and the staged tree is deliberately not a
           # checkout — without this the child errors before its node can answer.
           "PYTHONPATH": str(REPO),
           "FAKE_LIVE_DATA_ROOT": str(live),
           "CHOSEN_RECORD": str(record)}
    env.pop("LLOYD_DATA", None)
    env.pop("LLOYD_ALLOW_LIVE_TREE_TESTS", None)
    # The production shape of the incident: TMPDIR set, the other two temp
    # variables not, so the helper's platform defaults are what it falls back on.
    env.pop("TEMP", None)
    env.pop("TMP", None)
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_probe.py"],
                          cwd=stage, env=env, capture_output=True, text=True,
                          timeout=300)
    out = done.stdout + done.stderr

    # The child exits via `os._exit`, so it prints no summary and its exit status
    # proves nothing; the record it wrote is the witness that its node ran, and the
    # two claims below are read off that record and off the filesystem.
    assert record.is_file(), f"the child never reached its node: {out}"
    chosen = Path(json.loads(record.read_text(encoding="utf-8")))
    try:
        assert not _under(chosen, live), f"the child's substitute root: {chosen}"
        survivors = sorted(p.name for p in live.rglob(f"{DATA_SCRATCH_PREFIX}*"))
        assert survivors == [], \
            f"the run left its substitute data root in the live root: {survivors}"
    finally:
        # The child died on purpose, so ITS teardown never ran and the root it did
        # make — outside the live root, which is the point — is this node's to take.
        shutil.rmtree(chosen, ignore_errors=True)
