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
import dataclasses
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


# ── The capability envelope, on the same roster machinery (#2269) ────────────
#
# The reason this lives here and not in a second file is the reason #1963
# exists at all: a fact about a dispatch path is only worth anything if the
# denominator is derived, so that the path nobody enumerated is the one the
# report names. A worker source's capability envelope is the same kind of fact —
# one per source, checkable per run — and this module already had the mechanism
# to enumerate the sources and the shape for reporting a divergence. A source
# that reaches a durable-external tool outside its declaration is therefore
# reported the way an unarmed registry is today, rather than joining the set of
# things that read as fine because nobody enumerated them (#1828).
#
# Unlike the guard columns this is not purely textual: an envelope is a set of
# tool names, so `app.harness.capabilities` is imported here (which imports the
# committed tool roster and the policy tiers, and nothing that opens a pool).
# The denominator — which sources exist at all — stays AST-derived, which is
# where the drift value is.

#: A worker source whose turn has no declared capability set. Printed as a
#: column, not failed on: the item's own risk rule is one source per round, so
#: a source the loop has not narrowed yet is expected, and a fleet-wide red on
#: day one would only train everyone to ignore the report.
FINDING_UNDECLARED_ENVELOPE = "worker-source-declares-no-capability-set"

#: A source whose whole input is text somebody else wrote — a transcript off the
#: internet, a fetched page, another agent's session file — reaching something
#: durable outside itself. That pairing is the one the envelope exists to break,
#: so for these sources the column becomes a finding.
FINDING_INGEST_REACHES_DURABLE = "untrusted-ingest-source-reaches-a-durable-external-tool"


def worker_source_names(root: str | Path | None = None) -> set[str]:
    """Every source name the roster imports, read from the AST.

    Not `import workers.sources`: the value is that a module added to that
    import list appears in the denominator with nobody remembering to say so,
    and pulling the package in would execute fifteen module bodies — each of
    which opens config, the pool's registration and the backlog — to learn a
    list of names.
    """
    base = _base(root)
    tree = _oc._parse("workers/sources/__init__.py",
                      base / "workers/sources/__init__.py")
    if tree is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "workers.sources":
            for a in node.names:
                rel = f"workers/sources/{a.name}.py"
                mod = _oc._parse(rel, base / rel)
                if mod is None:
                    continue
                for n in ast.walk(mod):
                    if (isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "NAME"
                                    for t in n.targets)
                            and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, str)):
                        names.add(n.value.value)
    # The stem scan, which the docstring has always promised and which this
    # change found missing: `workers/sources/failure_ledger.py` defines
    # `NAME = fl.SOURCE`, an attribute rather than a literal, so the AST walk
    # above cannot read its name and the roster's own fifteenth source was about
    # to be a row this report never printed. That is the identical defect the
    # counting half of this file was written for, one level up — a denominator
    # that silently drops a member reports a clean fleet. Stems normalise
    # underscore-to-hyphen because that is the convention every source's `NAME`
    # follows; `test_the_capability_report_enumerates_the_source_roster` asserts
    # this set against the live registry, so if the convention breaks the test
    # names the source instead of the report losing it.
    src = base / "workers/sources"
    if src.is_dir():
        names |= {p.stem.replace("_", "-") for p in src.glob("*.py")
                  if not p.name.startswith("_")}
    return names


def capability_matrix(root: str | Path | None = None) -> dict[str, dict]:
    """`{source: {declared, n_allowed, durable_reach, untrusted_ingest}}`.

    For a source with a declared set `durable_reach` is exact: the envelope *is*
    the reachable set, so any tier-2/3 name inside it is one that turn may
    reach. For a source with none, both cells are `None` rather than a guessed
    zero — its reach is the deny list less whatever `config.yaml` disables,
    which this module does not read. An empty cell would be the exact lie
    `_counting`/`_completeness` were written for.
    """
    from app.harness.capabilities import (SOURCE_CAPABILITIES,
                                          UNTRUSTED_INGEST_SOURCES,
                                          durable_external, envelope_for)
    out: dict[str, dict] = {}
    for name in sorted(worker_source_names(root)):
        declared = name in SOURCE_CAPABILITIES
        env = envelope_for(name) if declared else None
        out[name] = {
            "declared": declared,
            "n_allowed": None if env is None else len(env),
            "durable_reach": None if env is None
            else sorted(durable_external(env)),
            "untrusted_ingest": name in UNTRUSTED_INGEST_SOURCES,
        }
    return out


@dataclasses.dataclass(frozen=True)
class CapabilityFinding:
    """One source whose capability roster disagrees with the tree."""
    file: str
    target: str
    reason: str
    extra: str = ""


def capability_findings(root: str | Path | None = None) -> list[CapabilityFinding]:
    """Where the capability declarations and the source roster have come apart."""
    from app.harness.capabilities import SOURCE_CAPABILITIES
    found = capability_matrix(root)
    findings = []
    for name, cells in found.items():
        if not cells["declared"]:
            findings.append(CapabilityFinding(
                "workers/sources", name, FINDING_UNDECLARED_ENVELOPE,
                "no declared capability set: what its turn may reach is "
                "whatever the deny list forgot to name"))
        elif cells["untrusted_ingest"] and cells["durable_reach"]:
            findings.append(CapabilityFinding(
                "workers/sources", name, FINDING_INGEST_REACHES_DURABLE,
                f"declared envelope reaches {', '.join(cells['durable_reach'])} "
                "on a turn whose whole input is untrusted text"))
    # The other direction of the same drift: a set that reads as protecting a
    # job the roster no longer registers.
    for name in sorted(set(SOURCE_CAPABILITIES) - set(found)):
        findings.append(CapabilityFinding(
            "workers/sources", name, FINDING_UNDECLARED_ENVELOPE,
            "a capability set is declared for a source the roster does not "
            "register"))
    return findings


def render_capabilities(matrix: dict[str, dict]) -> str:
    """One row per worker source: its envelope, and what durable reach is in it."""
    lines = ["| worker source | declared | tools allowed | durable-external "
             "reach | untrusted ingest |", "|---|---|---|---|---|"]
    for name, c in matrix.items():
        lines.append(
            f"| `{name}` | {'yes' if c['declared'] else 'no'} "
            f"| {'—' if c['n_allowed'] is None else c['n_allowed']} "
            f"| {'—' if c['n_allowed'] is None else (', '.join(c['durable_reach']) or 'none')} "
            f"| {'yes' if c['untrusted_ingest'] else 'no'} |")
    return "\n".join(lines)
