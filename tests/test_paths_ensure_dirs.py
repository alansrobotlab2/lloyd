"""Importing `app.paths` lays down nothing; the boots create the directories.

`app/paths.py` used to run `SESSIONS_DIR.mkdir(parents=True, exist_ok=True)` at
module level (#712). An import-time `mkdir` means `pytest` writes directories
into the tree it merely *collects* from — and during a self-modification gate the
full-suite rung imports the round's own checkout, so containers appeared in a
tree no code had run in, and a reader that asked whether a `_pipeline` container
existed took that for a populated store. Backlog #525 exists to kill exactly this
shape: a guard reading its own missing (or here, newly materialised) input as a
verdict.

What this pins, one test per acceptance clause:

* importing the module from a copy in an empty directory has no filesystem
  effect in or beside that directory, and still resolves `LLOYD_HOME` to it;
* `ensure_dirs()` creates the runtime-state directories, and is idempotent;
* both boot processes create the dirs before their first writer — the backend by
  running `server.app`'s own startup-hook list through the framework's lifespan
  protocol with every other hook stubbed, the aggregator by running its real
  `lifespan` coroutine the same way;
* the session-meta writer still creates `SESSIONS_DIR` when it is absent, so
  moving the mkdir off the import strands no writer.

Every test that claims a directory got created aims the module's directories at a
`tmp_path` it has confirmed empty, because the suite runs under
`tests/conftest.py`'s scratch `LLOYD_DATA`, which already exists: an `is_dir()`
against the real data root cannot fail here, and an assertion that cannot fail is
not evidence.

`LLOYD_HOME` stays code-anchored throughout: the canary depends on a worktree
resolving its own empty state, and only the side effect moved.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import app.paths as paths
import server

ROOT = Path(__file__).resolve().parent.parent

#: The five directories `ensure_dirs()` owns. Named here rather than derived
#: from the module, so a change to what it creates fails this file instead of
#: being confirmed by it.
STATE_DIRS = ("SESSIONS_DIR", "AUTONOMY_RUNS_DIR", "TASKS_DIR", "LOGS_DIR",
              "SCREENSHOTS_DIR")


def _checkout_copy(dest: Path) -> Path:
    """A copy of the two stdlib-only modules as an `app` package inside `dest`.

    `dest` is a pytest `tmp_path`, so it is not any checkout on this machine: the
    data-root rule's third branch applies and `DATA_ROOT` lands inside it, which
    is what makes "importing created something" observable at all.
    """
    pkg = dest / "app"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for mod in ("paths.py", "data_root.py"):
        shutil.copyfile(ROOT / "app" / mod, pkg / mod)
    return dest


def _listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _child_env() -> dict[str, str]:
    """The suite's environment minus every override that would move the copy's
    data root somewhere other than inside the copied tree."""
    return {k: v for k, v in os.environ.items()
            if k not in ("LLOYD_DATA", "PYTHONPATH", "LLOYD_KG_DB",
                         "LLOYD_FACTS_ROOT", "LLOYD_RESEARCH_DB")}


def _load_copy(tree: Path, monkeypatch):
    """Import the copied `paths` as its own module, in this process.

    Its `from app.data_root import ...` resolves to the already-imported real
    `app.data_root`, which is the same stdlib-only source with the same
    passwd-derived constants — what varies is only `__file__`, and therefore
    `LLOYD_HOME`.
    """
    _checkout_copy(tree)
    monkeypatch.delenv("LLOYD_DATA", raising=False)
    spec = importlib.util.spec_from_file_location(
        f"paths_copy_{tree.name}", tree / "app" / "paths.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aim_dirs_at(module, root: Path, monkeypatch) -> None:
    """Point one module's five runtime-state directories under `root`.

    Used by every test that claims a directory was created by a boot path. The
    gate runs the suite with `LLOYD_DATA` on a scratch root that already exists,
    so without this the creation asserts could not fail; `root` starts empty. The
    layout mirrors the real one — `screenshots` under `logs`, `tasks` under
    `_pipeline` — so what gets created has the production shape.
    """
    aimed = {"SESSIONS_DIR": root / "sessions",
             "AUTONOMY_RUNS_DIR": root / "autonomy-runs",
             "TASKS_DIR": root / "_pipeline" / "tasks",
             "LOGS_DIR": root / "logs",
             "SCREENSHOTS_DIR": root / "logs" / "screenshots"}
    for name, target in aimed.items():
        assert target.name == getattr(module, name).name, (
            f"{name} no longer points where this map assumes: {getattr(module, name)}")
        monkeypatch.setattr(module, name, target)
    # `ensure_dirs()` walks `RUNTIME_STATE_DIRS`, built once at import, so the
    # tuple has to move with the five names: patching the names alone would leave
    # the function creating the production directories instead, and the probes
    # below would report a defect that is not there.
    monkeypatch.setattr(module, "RUNTIME_STATE_DIRS",
                        tuple(aimed[name] for name in STATE_DIRS))


# ── clause 1: the import itself is side-effect free ──────────────────────────

def test_importing_app_paths_creates_nothing_in_or_beside_the_tree_it_came_from(tmp_path):
    """A fresh interpreter importing the module from a copy in an empty
    directory: the directory must be byte-for-byte what it was before.

    Run in a subprocess because the claim is about a *first* import in a process
    that has never seen `app.paths` — in here, the conftest scratch root and the
    already-imported package would both be in the way. `-B` keeps the
    interpreter's own `__pycache__` out of the listing: the claim is about what
    the module body does, not what the import machinery caches.
    """
    tree = _checkout_copy(tmp_path)
    before = _listing(tree)

    out = subprocess.run(
        [sys.executable, "-B", "-c",
         "import app.paths as p; print(p.LLOYD_HOME); print(p.DATA_ROOT)"],
        cwd=str(tree), env=_child_env(), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr

    home, data_root = out.stdout.split()
    # Positive control: the copy we placed there is the module that answered, and
    # its data root is inside it — so the listing below has somewhere to look.
    assert Path(home) == tree.resolve(), f"imported the wrong tree: {home}"
    assert Path(data_root) == tree.resolve() / ".lloyd-data"
    assert _listing(tree) == before, (
        f"importing app.paths created {_listing(tree)[:-len(before)]} in"
        f" {tree} — an import-time mkdir writes into any tree that imports it,"
        " including a checkout the run only collected from")


# ── clause 2: ensure_dirs() creates them, and twice is fine ──────────────────

def test_ensure_dirs_creates_the_runtime_state_dirs_under_its_own_tree(tmp_path, monkeypatch):
    mod = _load_copy(tmp_path / "checkout", monkeypatch)

    assert mod.LLOYD_HOME == (tmp_path / "checkout").resolve()
    for name in STATE_DIRS:
        assert not getattr(mod, name).exists(), f"{name} already existed after import"

    # What `ensure_dirs()` creates is that tuple, so membership is checkable per
    # name. It matters for `LOGS_DIR` above all: `SCREENSHOTS_DIR` is its child and
    # `mkdir(parents=True)` would make `logs` on the way there, so no filesystem
    # probe can tell the two apart and the tuple is the only surface on which
    # dropping `LOGS_DIR` is visible at all.
    assert sorted(str(d.relative_to(mod.DATA_ROOT)) for d in mod.RUNTIME_STATE_DIRS) == [
        "_pipeline/tasks", "autonomy-runs", "logs", "logs/screenshots", "sessions"], (
        "RUNTIME_STATE_DIRS no longer names the five directories the writers use")

    mod.ensure_dirs()

    for name in STATE_DIRS:
        directory = getattr(mod, name)
        assert directory.is_dir(), f"{name} was not created at {directory}"
        assert directory.is_relative_to(mod.LLOYD_HOME), directory
    # And nothing else is created: exactly those five, plus `_pipeline`, the parent
    # `TASKS_DIR` needs.
    assert _listing(mod.DATA_ROOT) == [
        "_pipeline", "_pipeline/tasks", "autonomy-runs", "logs", "logs/screenshots",
        "sessions"]

    mod.ensure_dirs()  # idempotent: a second boot, or a test, raises nothing


# ── clause 3: both boot processes call it before anything writes ─────────────

def _backend_boot_sightings(monkeypatch) -> list[tuple[str, bool]]:
    """Run `server.app`'s OWN startup-hook list, and report what each hook saw.

    Returns `(hook name, were the five dirs already there?)` per hook, in the
    order the framework actually invoked them. That is how the ordering is
    measured: every hook except `paths.ensure_dirs` is replaced, in its own slot
    of the live list, by a recorder that first looks at the filesystem and then
    notes its own name — so `ensure_dirs` runs UNMODIFIED, in the slot `server.py`
    registered it in, and what is under test is what the hooks behind it can see.
    The replacements exist because the ticker, the worker pool and the LLM-slot
    reconciliation behind them claim jobs and talk to supervisorctl: they are the
    one thing here that must not run.

    The framework is real, not paraphrased. FastAPI keeps `on_event("startup")`
    handlers in `app.router.on_startup`, and its `_DefaultLifespan.__aenter__`
    calls `router._startup()`, which walks that list in registration order and
    awaits a coroutine handler or calls a plain one. `TestClient` drives exactly
    that ASGI lifespan protocol, so both the order and the fact that a plain
    zero-argument callable is accepted in this list are executed here rather than
    read off the source.
    """
    router = server.app.router
    production = list(router.on_startup)
    # By qualified name, not identity: another test file `importlib.reload`s
    # app.paths, which rebinds `paths.ensure_dirs` to a new function object while
    # server.py still holds the one it registered (same module dict, so the old
    # object reads the same globals).
    ours = [h for h in production if getattr(h, "__module__", "") == "app.paths"
            and getattr(h, "__name__", "") == "ensure_dirs"]
    assert ours, (
        "server.py does not register paths.ensure_dirs as a startup hook at all")
    ensure_dirs = ours[0]
    assert production[0] is ensure_dirs, (
        f"the dirs hook sits at index {production.index(ensure_dirs)} of the"
        " backend's registration order, not first")

    sightings: list[tuple[str, bool]] = []

    def _created_yet() -> bool:
        return all(getattr(paths, name).is_dir() for name in STATE_DIRS)

    def in_place(hook):
        name = getattr(hook, "__name__", repr(hook))
        if hook is ensure_dirs:
            return hook  # the real callable, unmodified, in its real slot

        def recorder(*args, **kwargs):
            sightings.append((name, _created_yet()))

        # A coroutine handler is awaited and a plain one called; a recorder of the
        # wrong kind would change what the framework does with the slot.
        if inspect.iscoroutinefunction(hook):
            async def async_recorder(*args, **kwargs):
                sightings.append((name, _created_yet()))
            async_recorder.__name__ = name
            return async_recorder
        recorder.__name__ = name
        return recorder

    monkeypatch.setattr(router, "on_startup", [in_place(h) for h in production])
    monkeypatch.setattr(router, "on_shutdown", [])
    with TestClient(server.app):
        pass
    return sightings


def test_the_backend_boot_creates_the_dirs_before_the_ticker_and_the_pool(tmp_path, monkeypatch):
    """The backend creates the runtime-state dirs at the moment boot does, ahead of
    both hooks that write on their first tick (a run record under
    `AUTONOMY_RUNS_DIR`, a row in `workers.db`)."""
    _aim_dirs_at(paths, tmp_path, monkeypatch)
    assert list(tmp_path.iterdir()) == [], "the probe directory must start empty"

    sightings = _backend_boot_sightings(monkeypatch)

    names = [name for name, _ in sightings]
    assert names[0] == "start_autonomy_ticker", names
    assert names[1] == "start_worker_pool", names
    for name, created in sightings[:2]:
        assert created, (
            f"{name} is a backend writer that starts on its first tick and it saw"
            f" no directories: {sightings[:2]}")
    assert all(created for _, created in sightings), sightings
    for name in STATE_DIRS:
        assert getattr(paths, name).is_dir(), (
            f"{name} missing after the backend's startup list ran: the hook is"
            " registered but does not create it")


def test_the_sighting_probe_would_notice_a_late_registration(tmp_path, monkeypatch):
    """Positive control on the test above: the probe reports False when the dirs
    hook sits behind a writer, so those assertions cannot pass vacuously.

    Same framework, same probe, three hooks with the dir-creator LAST — the exact
    shape the clause forbids. The first hook has to see the directories missing.
    """
    _aim_dirs_at(paths, tmp_path, monkeypatch)

    sightings: list[tuple[str, bool]] = []

    def writer(*args, **kwargs):
        sightings.append(("writer", all(getattr(paths, n).is_dir() for n in STATE_DIRS)))

    def late(*args, **kwargs):
        paths.ensure_dirs()
        sightings.append(("ensure_dirs", True))

    router = server.app.router
    monkeypatch.setattr(router, "on_startup", [writer, writer, late])
    monkeypatch.setattr(router, "on_shutdown", [])
    with TestClient(server.app):
        pass

    assert [name for name, _ in sightings] == ["writer", "writer", "ensure_dirs"], sightings
    assert [created for _, created in sightings] == [False, False, True], (
        "the probe saw the directories already present in front of a writer that"
        " started before them, so it cannot detect the ordering defect")


async def test_the_mcp_lifespan_creates_the_dirs_before_its_first_writer(tmp_path, monkeypatch):
    """The aggregator is a second process (`python -m agent_mcp.main`), so the
    backend's hook covers nothing there. Runs the real `lifespan` coroutine with
    its external work stubbed, and records what ran in what order."""
    import agent_mcp.main as M

    _aim_dirs_at(paths, tmp_path, monkeypatch)
    assert list(tmp_path.iterdir()) == [], "the probe directory must start empty"
    order: list[str] = []
    real_ensure_dirs = paths.ensure_dirs

    def spy_ensure_dirs() -> None:
        real_ensure_dirs()
        order.append("ensure_dirs")

    async def _start_bot(*args, **kwargs):
        order.append("start_bot")

    async def _noop(*args, **kwargs):
        return None

    def _prune(*args, **kwargs):
        order.append("prune")

    monkeypatch.setattr(paths, "ensure_dirs", spy_ensure_dirs)
    monkeypatch.setattr(M.discord_bot, "start_bot_task", _start_bot)
    monkeypatch.setattr(M.discord_bot, "stop_bot", _noop)
    monkeypatch.setattr(M._tsc_runner, "warm_baseline", _noop)
    monkeypatch.setattr(M._tsc_runner, "shutdown", _noop)
    monkeypatch.setattr(M._change_ledger, "prune", _prune)
    monkeypatch.setattr(M._task_registry, "terminate_all", _noop)
    monkeypatch.setattr(M, "MODULES", [])

    async with M.lifespan(None):
        pass

    # `ensure_dirs` ahead of the bot task it is written ahead of, and ahead of the
    # ledger prune that stats `sessions/*.changes/`; and by the time the lifespan
    # has run, the directories it is there to create really exist.
    assert order == ["ensure_dirs", "start_bot", "prune"], order
    for name in STATE_DIRS:
        assert getattr(paths, name).is_dir(), (
            f"{name} missing after the aggregator's lifespan ran: the call is in"
            " the body but creates nothing")


# ── clause 4: the session-meta writer still creates the directory ────────────

async def test_the_session_meta_writer_creates_sessions_dir_when_it_is_absent(tmp_path,
                                                                             monkeypatch):
    """The writer that mints a transcript must not depend on some other boot step
    having made the directory first — `create_session` already covers itself, and
    this is the other half of the chat path."""
    from app import sessions_io

    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", sessions_dir)
    session_id = "20260924_000000_pathtest_abcd"

    assert not sessions_dir.exists()
    await sessions_io._save_session_meta(session_id, "claude-test", "a preview")

    meta = sessions_dir / f"{session_id}.json"
    assert meta.is_file(), f"{meta} was not written with {sessions_dir} absent"
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["session_id"] == session_id
    assert data["platform"] == "worker"  # a four-part id is a machine run (#1064)
    assert data["message_count"] == 1
