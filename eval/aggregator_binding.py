"""Where the egress window's aggregator child gets to find `agent_mcp` (#3169).

The ON arm of `eval/run_egress_arms.sh`'s window is served by an aggregator THIS process
starts: `DEFAULT_MCP_CMD` in `eval/egress_arms.py` is `[sys.executable, "-m",
"agent_mcp.main"]`, and `agent_mcp` is a plain package of this checkout installed in no
interpreter — `~/lloyd/.venvs/lloyd/lib/python3.12/site-packages` holds no `agent_mcp` and
no `.pth` pointing at the checkout — so it resolves ONLY through a cwd or a PYTHONPATH
naming the tree. `eval/egress_arms.py` widens its own `sys.path` and never hands that
widening to the child, which is why an operator's run worked and the service's did not:
`lloyd-egress-window.service` sets no `WorkingDirectory=`, the fire began with the child's
cwd at `/`, and 2026-10-19 06:27:12 the child died of `ModuleNotFoundError: No module named
'agent_mcp'` — after the window had already spent 16.652 s of wall clock pausing the pool
and draining in-flight work, and with zero rows written.

Two consumers, deliberately: `eval/egress_arms.py` uses these to bind the real spawn, and
`eval/run_egress_arms.sh` imports this module ALONE to pre-flight the import before it
pauses anything. That second reader is why this file imports nothing but the standard
library and nothing from the repo: `egress_arms` pulls `persistence_arms` and `workers.pool`
at module level, so a check phrased as `import egress_arms` could not answer the question in
a partial tree — it would fail for a reason that has nothing to do with the aggregator. The
binding is one function so the pre-flight and the spawn cannot drift apart.

Four nodes in `tests/test_run_egress_arms.py` pin this file, one per claim in it:
`test_the_binding_the_window_hands_the_child_is_what_makes_the_import_resolve` (the dict, a
real `import agent_mcp.main` under it, and the same argv unbound from `cwd="/"` failing — the
negative control), `test_the_aggregator_child_imports_agent_mcp_from_a_foreign_executor_cwd`
(a real window, launched from a cwd outside the checkout, whose child answers the import
question about itself), `test_an_unimportable_tree_is_refused_before_the_pool_is_paused` (the
pre-flight, over two trees that differ by one file), and
`test_the_import_pre_flight_never_asks_the_question_of_the_wrappers_cwd` (the trap the shape
exists to avoid). `tests/test_egress_window_unit.py`'s
`test_widening_the_childs_environment_brought_no_arm_flag_and_no_working_directory` pins the
arm-scope rule and, by AST, that this module loads nothing from the repo.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: The checkout this driver runs from, resolved through the SYMLINK. Same derivation as
#: `LLOYD_HOME` in `eval/egress_arms.py`; a round's worktree copy is self-consistent by
#: construction, which is what keeps a node run from a worktree from asserting a stale path.
LLOYD_HOME = Path(__file__).resolve().parent.parent

#: The module `DEFAULT_MCP_CMD` in `eval/egress_arms.py` asks the child to run. Named here,
#: and imported THERE, so the wrapper's pre-flight and the driver's argv cannot disagree
#: about what to look for. An operator's `LLOYD_MCP_CMD` override reaches the check as an
#: explicit `cmd=`, and a command with no `-m` in it is not a module import to check.
DEFAULT_MCP_MODULE = "agent_mcp.main"


def default_mcp_cmd() -> "list[str]":
    """The child command with no override in play — `mcp_cmd()`'s fallback in
    `eval/egress_arms.py`, spelled the same way so this module needs nothing from there.

    `tests/test_run_egress_arms.py::test_the_binding_the_window_hands_the_child_is_what_makes_the_import_resolve`
    asserts this list is byte-identical to `egress_arms.DEFAULT_MCP_CMD`: the duplication is
    the price of a pre-flight that must run in a tree whose driver cannot start, and the
    node is what keeps the price from becoming a drift.
    """
    return [sys.executable, "-m", DEFAULT_MCP_MODULE]

#: How many lines of a failed probe's tail a refusal repeats. The 2026-10-19 fire's own
#: `ModuleNotFoundError` is four lines, so four is enough to see the whole cause of the
#: common case without burying the refusal under a traceback.
AGG_PROBE_TAIL_LINES = 4

#: How many lines of a failed probe's tail a refusal repeats. The 2026-10-19 fire's own
#: cause was three lines (`Error while finding module specification for
#: 'agent_mcp.main'`, the ModuleNotFoundError, and the interpreter's `sys.path` note); the
#: tail is what turns `aggregator exited 1` into a diagnosis, and more than this is the
#: child's whole log, which is already in the journal next to the refusal.
AGG_PROBE_TAIL_LINES = 4


def aggregator_binding(env: dict[str, str]) -> dict[str, object]:
    """The `create_subprocess_exec` kwargs that bind an aggregator child to the checkout:
    a `cwd` of the repo root, and an `env` whose `PYTHONPATH` names that root first.

    One function, because the question "can this child import what it is about to import?"
    has to be asked of THIS pair, and there are two wrong answers that are easy to give. The
    one that fired on 2026-10-19 06:27:12 asked it of nothing: the child was spawned with no
    `cwd=`, so it inherited the executor's — `lloyd-egress-window.service` sets no
    `WorkingDirectory=`, so that was wherever systemd put the wrapper — and
    `DEFAULT_MCP_CMD`'s `-m agent_mcp.main` died as `ModuleNotFoundError: No module named
    'agent_mcp'` (journalctl: exit 1, after 16.652 s of wall clock already spent on the
    pause and the drain). The "fix" that asks the import question from wherever the wrapper
    happens to stand is worse, because it then refuses every systemd fire forever on a tree
    that imports perfectly — a permanent exit 2 that looks like a clean refusal.

    So both halves derive from `LLOYD_HOME`, this file's own resolved parent-of-parent, and
    never from the process that launched the window: the child's cwd IS the checkout, and
    its `PYTHONPATH` gains that same root at the front. The entry is what makes it work at
    all — `agent_mcp` is a plain package in the checkout, installed in no interpreter here
    (`~/lloyd/.venvs/lloyd/lib/python3.12/site-packages/` holds `distutils-precedence.pth`
    and `_virtualenv.pth` and nothing named `agent_mcp`), so a path entry is its ONLY
    resolution route. The parent's own widening — `LLOYD_HOME` and `LLOYD_HOME/eval`
    inserted into `sys.path` at the top of this file — changes THIS process only and crosses
    no boundary, which is exactly why the driver's own `from agent_mcp.egress import …` was
    fine while its child could not find the package.

    The input mapping is never mutated: the caller's arm entries (`LLOYD_EGRESS_ENFORCE`,
    `LLOYD_MCP_PORT`, `LLOYD_DATA`) pass through untouched, `PYTHONPATH` is the only key
    this adds, and the window's own environment is not the child's environment.
    """
    root = str(LLOYD_HOME)
    widened = dict(env)
    parts = [p for p in (widened.get("PYTHONPATH") or "").split(os.pathsep) if p]
    if root not in parts:
        widened["PYTHONPATH"] = os.pathsep.join([root, *parts])
    return {"cwd": root, "env": widened}


def import_needed_by(argv: list[str]) -> "str | None":
    """The module this argv makes the interpreter import, or None if it imports none.

    `-m pkg.mod` is answered by the interpreter's own `sys.path`, so argv of that shape
    cannot even start unless `pkg.mod` resolves. An absolute script argv — the shape
    `eval/persistence_arms.py:119` ships for its own arm, and the shape every stub in
    `tests/test_run_egress_arms.py` uses — resolves by path and needs no package at all, so
    a pre-flight that checked `agent_mcp` against that launch would be testing a thing that
    is not about to happen.
    """
    for i, tok in enumerate(argv[:-1]):
        if tok == "-m":
            return argv[i + 1]
    return None


def aggregator_import_check(cmd: "list[str] | None" = None,
                            env: "dict[str, str] | None" = None, *,
                            timeout_s: float = 120.0) -> tuple[int, str]:
    """Ask a fresh interpreter, in the child's cwd and with the child's environment, to
    import the module the child's own argv asks for. Returns `(returncode, stderr tail)`.

    `-c "import agent_mcp.main"` and `-m agent_mcp.main` take the same two inputs —
    `sys.path`, which the interpreter seeds from the cwd and `PYTHONPATH` — so this is the
    same question the fork asks, asked one moment earlier and for a measured 1.4 s
    (`time PYTHONPATH=<root> python -c 'import agent_mcp.main'` from `/`), against the same
    interpreter (`argv[0]`, which is `sys.executable` for the shipped command).
    """
    argv = list(cmd or default_mcp_cmd())
    target = import_needed_by(argv)
    if target is None:
        return 0, ""
    bound = aggregator_binding(dict(os.environ if env is None else env))
    try:
        done = subprocess.run([argv[0], "-c", f"import {target}"],
                              capture_output=True, text=True, timeout=timeout_s,
                              **bound)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 2, f"{type(exc).__name__}: {exc}"
    tail = "\n".join((done.stderr or done.stdout or "").strip()
                     .splitlines()[-AGG_PROBE_TAIL_LINES:])
    return int(done.returncode), tail


def aggregator_preflight(cmd: "list[str] | None" = None) -> int:
    """`eval/run_egress_arms.sh`'s gate, in Python so the shell cannot invent its own
    idea of what the child's environment is. Returns 0 to proceed, 2 to refuse.

    Exit 2 is the code the window already refuses with (its own `AggregatorRefused` path
    and the unit's `status=2`), so the fire that was already a refusal stays the same kind
    of refusal — earlier, and without the pool pause in front of it.
    """
    rc, tail = aggregator_import_check(cmd)
    if rc:
        argv = list(cmd or default_mcp_cmd())
        print(f"{Path(__file__).name}: the enforce-ON arm's aggregator cannot import "
              f"{import_needed_by(argv)!r} in the environment it is about to be given "
              f"(cwd={LLOYD_HOME})", file=sys.stderr)
        for line in tail.splitlines():
            print(f"  | {line}", file=sys.stderr)
    return 2 if rc else 0
