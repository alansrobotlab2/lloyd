"""#824 — importing `eval/run_prefetch_eval.py` must not blind the rest of the process.

The script hushed its noisy ``import prefetch`` with a bare
``logging.disable(logging.CRITICAL)`` executed at import time (arrived in
``f5789b71``, 2026-09-03). ``logging.disable`` writes ``Logger.manager.disable``:
process-global, consulted before *any* handler runs, and nothing in the tree ever
restored it. So from that import onward every ``caplog`` assertion in the same
pytest process came back empty -- at base ``ed12261e`` that is 29 files under
``tests/`` using ``caplog`` across 63 lines reading ``caplog.records``/``.text``/
``.messages``; 26 of the 29 call ``at_level``/``set_level`` somewhere, which is
pytest's per-test un-disable and the only reason most of them survive. Which is how
#562 lost a round to a red it did not cause.

What is pinned here:

* clause 1 -- importing the module leaves global logging undisabled, and the
  silencing still covers the noisy import it was written for;
* clause 1 again across the other loader the repo uses -- ``importlib.
  util.spec_from_file_location`` by path, which is how
  ``tests/test_tool_choice_eval_cost.py`` actually pulls the script into a test
  process today and therefore the copy of this hazard that is live in the suite;
* clause 3 -- ``main()`` is quiet while it runs and neutral afterwards, on the
  return path and on the raise path.

Clause 2 (a ``caplog`` test that imports the script first still captures the
``tool_effects: suppressed duplicate`` line with its arguments) lives with the
effect ledger it guards, in ``tests/test_tool_effects.py``.

A note on why "quiet" is measured as *emitted or not* rather than by reading the
flag back: a handler at level 0 sees a record only if the logging machinery was
allowed to create it, so an invisible record proves the quiet worked and a visible
one proves it did not. Every test here carries that control beside its assertion,
because an empty list is also what a broken collector returns.
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "eval" / "run_prefetch_eval.py"
HIT = {"file": "qmd://knowledge/example-note.md"}


class _Collector(logging.Handler):
    """Every record that actually reaches a handler, at any level."""

    def __init__(self) -> None:
        super().__init__(level=0)
        self.emitted: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.emitted.append(record)


def _messages(collector: _Collector) -> list[str]:
    return [r.getMessage() for r in collector.emitted]


@pytest.fixture
def collector(monkeypatch):
    """A level-0 handler on the root logger, with root permissive.

    ``logging.disable`` drops a record before propagation, so what this collects is
    exactly what the process was permitted to say. ``lloyd.prefetch`` declares no
    level of its own, so pinning root is what makes a stub's WARNING reliably
    emit no matter what an earlier test left on the root logger.
    """
    root = logging.getLogger()
    handler = _Collector()
    root.addHandler(handler)
    monkeypatch.setattr(root, "level", min(root.level, logging.DEBUG))
    try:
        yield handler
    finally:
        root.removeHandler(handler)


def _forget_script() -> None:
    """Make the next import of the script really execute it.

    Both halves are needed: ``sys.modules`` and the parent package's attribute.
    ``from eval import run_prefetch_eval`` is answered off the parent's attribute
    when it is set, so popping only ``sys.modules`` hands back the old module and
    the module body -- the quiet window this file is about -- never runs again.
    """
    sys.modules.pop("eval.run_prefetch_eval", None)
    pkg = sys.modules.get("eval")
    if pkg is not None:
        pkg.__dict__.pop("run_prefetch_eval", None)


def _load_script() -> object:
    """Re-execute the script as ``eval.run_prefetch_eval`` and hand it back."""
    _forget_script()
    return importlib.import_module("eval.run_prefetch_eval")


@pytest.fixture
def script(collector):
    """A freshly imported script module plus the collector, global level restored.

    Depends on ``collector`` so the collector is installed before the import and
    can see anything that import lets through.
    """
    prev = logging.Logger.manager.disable
    try:
        yield _load_script(), collector
    finally:
        logging.disable(prev)


# ── clause 1: importing the script leaves logging enabled ───────────────────


def test_importing_the_script_leaves_global_logging_undisabled(collector):
    """`from eval import run_prefetch_eval` must not silence the process.

    Before the fix this asserted nothing about a flag nobody could see: the value
    here is the one the acceptance names, and the record after it is the positive
    control that says `NOTSET` really means "a WARNING gets out".
    """
    prev = logging.Logger.manager.disable
    try:
        _forget_script()
        from eval import run_prefetch_eval  # the literal import clause 1 names

        assert Path(run_prefetch_eval.__file__).resolve() == SCRIPT, (
            "`eval.run_prefetch_eval` resolved to "
            f"{run_prefetch_eval.__file__}, not {SCRIPT}: the assertions below "
            "would be about some other module")
        assert logging.Logger.manager.disable == logging.NOTSET, (
            "importing eval.run_prefetch_eval left Logger.manager.disable at "
            f"{logging.Logger.manager.disable}; every later caplog assertion in "
            "this process would read 'no record was logged'")

        logging.getLogger("lloyd.test824.after_import").warning("visible again")
        assert "visible again" in _messages(collector), (
            "manager.disable reads NOTSET but nothing reached a handler: the "
            "collector, not the flag, is the witness")
    finally:
        logging.disable(prev)


def test_loading_the_script_by_path_leaves_global_logging_undisabled(collector):
    """The seam that fires inside the suite today, not only the import clause 1 names.

    `tests/test_tool_choice_eval_cost.py::_load_prefetch_runner` never says
    ``eval.run_prefetch_eval``: it execs the file by path as a module named
    ``run_prefetch_eval`` through ``importlib.util.spec_from_file_location``. Under
    the old import-time disable that was the trigger live in the committed suite --
    and with the gate's 8 xdist workers it poisoned whichever worker drew that one
    file, which is a flake whose assignment is a file-to-worker hash rather than a
    standing red. Both loaders execute the same module body, so both are pinned.
    """
    prev = logging.Logger.manager.disable
    try:
        sys.modules.pop("run_prefetch_eval", None)
        spec = importlib.util.spec_from_file_location(
            "run_prefetch_eval", str(SCRIPT))
        assert spec is not None and spec.loader is not None, (
            f"{SCRIPT} is not loadable by path, so this test would prove nothing")
        module = importlib.util.module_from_spec(spec)
        sys.modules["run_prefetch_eval"] = module
        spec.loader.exec_module(module)

        assert Path(module.__file__).resolve() == SCRIPT, (
            f"the by-path load resolved to {module.__file__}, not {SCRIPT}")
        assert logging.Logger.manager.disable == logging.NOTSET, (
            "executing the script by path left Logger.manager.disable at "
            f"{logging.Logger.manager.disable}; every later caplog assertion in "
            "that xdist worker reads 'no record was logged'")

        logging.getLogger("lloyd.test824.by_path").warning("visible by path")
        assert "visible by path" in _messages(collector), (
            "manager.disable reads NOTSET but the by-path load still silenced "
            "this process: the collector, not the flag, is the witness")
    finally:
        sys.modules.pop("run_prefetch_eval", None)
        logging.disable(prev)


def test_the_noisy_import_it_was_written_for_is_still_hushed():
    """Deleting the disable would satisfy clause 1 and hand the nightly a wall of
    `prefetch` INFO, so the silence is pinned in a fresh interpreter over a
    stand-in `prefetch` whose *import* logs -- the one thing the quiet exists for.

    Run as a subprocess because it is exactly that: a process that imports the
    script once, with the real `prefetch` (and its vector and qmd legs) never
    touched. Both lines are the clause: silent during the import, and disabled
    nothing afterwards.
    """
    prog = """
import importlib.abc, importlib.util, logging, sys, types

seen = []


class _Collector(logging.Handler):
    def emit(self, record):
        seen.append(record.getMessage())


root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(_Collector(level=0))


class _NoisyLoader(importlib.abc.Loader):
    def create_module(self, spec):
        logging.getLogger("lloyd.prefetch").warning("IMPORT NOISE")
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        pass


class _OnlyPrefetch:
    def find_spec(self, name, path=None, target=None):
        if name != "prefetch":
            return None
        return importlib.util.spec_from_loader("prefetch", _NoisyLoader())


sys.meta_path.insert(0, _OnlyPrefetch())
sys.path.insert(0, sys.argv[1])
from eval import run_prefetch_eval  # noqa: F401

print("DISABLE", logging.Logger.manager.disable)
print("NOISE_SEEN", any("IMPORT NOISE" in m for m in seen))
"""
    proc = subprocess.run(
        [sys.executable, "-c", prog, str(REPO)],
        cwd=str(REPO), capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"probe failed: {proc.stderr[-800:]}"
    assert "NOISE_SEEN False" in proc.stdout, (
        "the script's own `import prefetch` is no longer quiet -- the run this "
        f"probe models said: {proc.stdout.strip()!r}")
    assert "DISABLE 0" in proc.stdout, (
        "importing the script left the process's logging disabled "
        f"(manager.disable): {proc.stdout.strip()!r}")


def test_the_quiet_window_hushes_its_block_and_restores_the_previous_level(script):
    """The primitive behind both clauses: quiet inside, exactly as found outside.

    Pinned on records rather than on the flag, so it fails both ways a fix can go
    wrong -- a window that stops silencing, and a window whose quiet outlives it.
    """
    mod, collector = script
    prev = logging.Logger.manager.disable
    noise = logging.getLogger("lloyd.prefetch")
    try:
        with mod._quiet_logging():
            noise.warning("inside the window")
            assert logging.Logger.manager.disable == logging.CRITICAL, (
                "inside the window manager.disable must be CRITICAL, got "
                f"{logging.Logger.manager.disable}")
        assert "inside the window" not in _messages(collector), (
            "a WARNING raised inside the window reached a handler: the quiet is "
            "not covering its own block")
    finally:
        logging.disable(prev)

    noise.warning("outside the window")
    assert "outside the window" in _messages(collector), (
        "after the window a WARNING must be emitted again; if it is not, the quiet "
        "outlived its block, which is exactly the #824 defect")


# ── clause 3: main() is quiet during the run, neutral after ─────────────────


def _one_query(tmp_path) -> Path:
    queries = tmp_path / "queries.yaml"
    queries.write_text(
        "queries:\n"
        "  - id: q-example\n"
        "    query: the example note\n"
        "    category: knowledge\n"
        "    expect_docs: [knowledge/example-note.md]\n",
        encoding="utf-8")
    return queries


def _stub_legs(monkeypatch, *, calls: list, noise: bool = True):
    """Stub the legs `run()` calls, and make the lex leg log like the real one.

    `prefetch` logs at INFO from inside these legs, which is the noise the script
    is right to hush; the stub keeps that behaviour so "quiet" is tested against a
    leg that actually talks, not a silent one.
    """
    import prefetch

    def lex(query, focus, deadline=None):
        calls.append(query)
        if noise:
            logging.getLogger("lloyd.prefetch").warning("PREFETCH LEG NOISE")
        return [dict(HIT)]

    monkeypatch.setattr(prefetch, "_search_vault_lex", lex)
    monkeypatch.setattr(prefetch, "_merge_vault_results",
                        lambda lex_hits, hybrid_hits: list(lex_hits))
    return prefetch


def test_main_emits_nothing_below_error_and_leaves_logging_as_it_found_it(
        script, monkeypatch, tmp_path):
    """Clause 3, return path: quiet through the run, neutral the moment it ends."""
    mod, collector = script
    calls: list = []
    _stub_legs(monkeypatch, calls=calls)

    baselines = tmp_path / "baselines"
    baselines.mkdir()
    monkeypatch.setattr("app.paths.EVAL_BASELINES_DIR", baselines)
    monkeypatch.setattr(sys, "argv", [
        "run_prefetch_eval.py", "--queries", str(_one_query(tmp_path)),
        "--label", "pinned-824", "--skip-hybrid"])

    before = logging.Logger.manager.disable
    assert mod.main() == 0

    assert calls, "the eval never reached the stubbed leg, so 'no noise' proves nothing"
    artifact = sorted(baselines.glob("*.json"))
    assert len(artifact) == 1, f"expected one artifact, got {[p.name for p in artifact]}"
    artifact_json = json.loads(artifact[0].read_text())
    assert artifact_json["records"][0]["lex"]["doc_hit"] is True, (
        "the stubbed leg was called but its hit never reached the artifact, so the "
        "run did not really go through it")

    assert logging.Logger.manager.disable == before, (
        "main() returned with the process's logging disabled at "
        f"{logging.Logger.manager.disable}: the nightly would leave every later "
        "run in the process unable to log")
    assert "PREFETCH LEG NOISE" not in _messages(collector), (
        f"the run logged {calls.__len__()} time(s) while it was meant to be quiet")

    logging.getLogger("lloyd.prefetch").warning("PREFETCH LEG NOISE")
    assert "PREFETCH LEG NOISE" in _messages(collector), (
        "positive control: this logger's records reach the collector at all, so "
        "the silence above was the quiet, not a deaf handler")


def test_a_run_that_raises_still_leaves_the_disable_level_as_it_was(
        script, monkeypatch, tmp_path):
    """Clause 3, raise path.

    The restore has to be on the way out of a failure too: this script runs
    unattended over a corpus that can be missing, and a `finally`-less quiet turns
    one crashed nightly into a process that silently logs nothing ever again.
    """
    mod, collector = script
    calls: list = []
    import prefetch

    def boom(query, focus, deadline=None):
        calls.append(query)
        logging.getLogger("lloyd.prefetch").warning("NOISE BEFORE THE RAISE")
        raise RuntimeError("lex leg died")

    monkeypatch.setattr(prefetch, "_search_vault_lex", boom)
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    monkeypatch.setattr("app.paths.EVAL_BASELINES_DIR", baselines)
    monkeypatch.setattr(sys, "argv", [
        "run_prefetch_eval.py", "--queries", str(_one_query(tmp_path)),
        "--label", "pinned-824-raise", "--skip-hybrid"])

    before = logging.Logger.manager.disable
    with pytest.raises(RuntimeError, match="lex leg died"):
        mod.main()

    assert calls, "the leg was never called, so nothing was on the raise path"
    assert "NOISE BEFORE THE RAISE" not in _messages(collector), (
        "the failing leg logged while main() was running: the quiet has to cover "
        "the run that ends in an exception, not only the one that succeeds")
    assert logging.Logger.manager.disable == before, (
        "main() raised and left Logger.manager.disable at "
        f"{logging.Logger.manager.disable}")

    logging.getLogger("lloyd.prefetch").warning("NOISE BEFORE THE RAISE")
    assert "NOISE BEFORE THE RAISE" in _messages(collector), (
        "positive control: the collector sees this logger, so the silence above "
        "was the quiet and not a deaf handler")


def test_the_quiet_window_never_makes_logging_lounder_than_a_caller_asked():
    """`max(prev, level)` is the whole reason this is safe to import: entering the
    window can only be quieter, so a caller that had deliberately disabled INFO is
    not un-disabled by running this script -- and is put back exactly as it was."""
    prev = logging.Logger.manager.disable
    try:
        mod = _load_script()
        logging.disable(logging.ERROR)
        with mod._quiet_logging(logging.CRITICAL):
            inside = logging.Logger.manager.disable
            assert inside == logging.CRITICAL, (
                "inside the window the level must be CRITICAL (quieter than the "
                f"caller's ERROR), got {inside}")
        assert logging.Logger.manager.disable == logging.ERROR, (
            "leaving the window must restore the caller's own level (ERROR), not "
            f"NOTSET: {logging.Logger.manager.disable}")
    finally:
        logging.disable(prev)
