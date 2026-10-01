"""Which guards does each production dispatch path arm? Derived, not listed (#1963).

`architecture/guard-coverage.md` publishes a guard-by-path matrix as prose, and
its own review found that prose wrong in the way a call-site grep is always
wrong: `app/routers/turn_options.py` contains no `install_outbound_content_gate(`
call, so a reader running the page's command concluded the interactive builder
does not arm the outbound gate — while `install_default_safety_hook` arms it from
inside its own body. `outbound_content.stale_gate_arm_points` already reads
bodies, but for one guard and only from tests.

This is the same derivation over every hook-installed guard:

- **Rows** are the files `outbound_content.dispatch_registry_sites_files` finds:
  those that build a turn a sender tool is reachable from. One denominator, the
  one the gate's own roster is measured against.
- **An arm** is a call path, read from syntax: a call in that file to an
  installer, or to an installer-shaped function (`install_*` / `_install_*`)
  whose body reaches one, however many bodies deep. Calls written *inside* an
  installer's own definition belong to whoever calls that installer, not to the
  file the definition happens to live in.
- A path is credited with exactly what its calls reach. A file that calls only
  the floor gets the floor and whatever the floor's body arms, and nothing else.

What it does not say: whether the arm is conditional. `install_policy_hook` is
called only where a grant scope exists and the action reviewer only on a worker
stream turn; a cell here means "this path can arm it", which is the question a
missing cell answers and the one the page's matrix was getting wrong. Guards
that refuse from inside a tool handler (`_injection_probe`, `session.py`,
`egress.py`) are not hook-installed and are out of this table by construction.

Stdlib and `outbound_content`'s own AST helpers only; no turn is built.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from app.harness import outbound_content as _oc

#: Guard → the installer names that count as arming it. The action reviewer has
#: two: the harness installer and the router's platform-gated wrapper around it.
GUARD_INSTALLERS: dict[str, tuple[str, ...]] = {
    "safety": ("install_default_safety_hook",),
    "policy": ("install_policy_hook",),
    "outbound_content": ("install_outbound_content_gate",),
    "action_review": ("_install_action_review", "install_action_review_hook"),
}
GUARDS: tuple[str, ...] = tuple(GUARD_INSTALLERS)

#: What makes a function an installer worth reading the body of. Following every
#: call by bare name would credit a path with whatever any same-named function
#: anywhere happens to do; installers are named as installers on this tree.
_INSTALLER_NAME = re.compile(r"^_?install_[a-z0-9_]+$")


def _base(root: str | Path | None) -> Path:
    return Path(str(root)) if root else Path(__file__).resolve().parents[2]


def _installer_calls(node: ast.AST) -> set[str]:
    return {name for n in ast.walk(node) if isinstance(n, ast.Call)
            for name in (_oc._attr(n.func),) if _INSTALLER_NAME.match(name)}


def _is_installer_def(node: ast.AST) -> bool:
    return (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and bool(_INSTALLER_NAME.match(node.name)))


def installer_bodies(root: str | Path | None = None) -> dict[str, set[str]]:
    """Installer name → the installer names its body calls, tree-wide."""
    out: dict[str, set[str]] = {}
    for rel, path in _oc._package_files(_base(root)):
        tree = _oc._parse(rel, path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if _is_installer_def(node):
                out.setdefault(node.name, set()).update(
                    _installer_calls(node) - {node.name})
    return out


def _reach(name: str, bodies: dict[str, set[str]], seen: set[str] | None = None) -> set[str]:
    seen = seen if seen is not None else set()
    if name in seen:
        return seen
    seen.add(name)
    for callee in bodies.get(name, ()):
        _reach(callee, bodies, seen)
    return seen


def _calls_outside_installer_defs(tree: ast.Module) -> set[str]:
    """Installer calls in a file, minus those inside an installer's own body."""
    inside: set[int] = set()
    for node in ast.walk(tree):
        if _is_installer_def(node):
            inside.update(id(n) for n in ast.walk(node) if isinstance(n, ast.Call))
    return {name for n in ast.walk(tree)
            if isinstance(n, ast.Call) and id(n) not in inside
            for name in (_oc._attr(n.func),) if _INSTALLER_NAME.match(name)}


def guard_arm_matrix(root: str | Path | None = None) -> dict[str, dict[str, bool]]:
    """`{dispatch file: {guard: armed}}` for every sender-reachable turn builder."""
    base = _base(root)
    bodies = installer_bodies(base)
    matrix: dict[str, dict[str, bool]] = {}
    for rel in sorted(_oc.dispatch_registry_sites_files(base)):
        tree = _oc._parse(rel, base / rel)
        reached: set[str] = set()
        if tree is not None:
            for name in _calls_outside_installer_defs(tree):
                reached |= _reach(name, bodies)
        matrix[rel] = {guard: bool(reached & set(names))
                       for guard, names in GUARD_INSTALLERS.items()}
    return matrix


def render(matrix: dict[str, dict[str, bool]]) -> str:
    """One row per dispatch path, naming the guards it arms."""
    lines = ["| dispatch path | " + " | ".join(GUARDS) + " | arms |",
             "|---|" + "---|" * (len(GUARDS) + 1)]
    for rel, cells in matrix.items():
        armed = [g for g in GUARDS if cells.get(g)]
        lines.append(f"| `{rel}` | "
                     + " | ".join("yes" if cells.get(g) else "—" for g in GUARDS)
                     + f" | {', '.join(armed) if armed else 'none'} |")
    return "\n".join(lines)
