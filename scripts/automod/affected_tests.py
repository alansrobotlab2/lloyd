"""Affected-test selection, in shadow: what a selected run would have missed.

`architecture/testing.md` ("Affected-test selection") orders the work: shadow
first, promote only on zero misses over a few hundred rounds. This module is
step 1. The gate's `tests` rung still runs the full suite; afterwards
`shadow()` computes the files a selection would have run and records which of
the rung's red nodes fall outside them. **Nothing here can change a verdict**:
`gate.py` attaches it at its foot through a wrapper that swallows every error.

The selection, deliberately conservative:

- **A full run** whenever a changed path is anything other than a Python file
  under the tree or a Markdown doc: `config.yaml`, any `conftest.py`, a
  `tests/_*.py` helper, `pytest.ini`, a requirements or lock file, a YAML, JSON,
  shell or frontend file. The AST import graph cannot see who reads those.
- **A changed test file** selects itself.
- **A changed Python module** selects every test file that imports it, directly
  or through other repo modules (reverse closure over `import` / `from` edges
  read by AST, relative imports resolved).
- **Any changed path** also selects every test file whose source mentions the
  path as text — the citation rails read docs and sources by path, and 2,214
  tests are rails (`testing.md`, "The citation rails, counted"). A `.py` path
  is also looked for as its dotted module name, which catches a monkeypatch by
  dotted string.

Two selections are recorded side by side, because the first measurement
(2026-11-23) showed the transitive one barely selects: the repo's import graph
is so connected that a change to `scripts/automod/sweep.py`, `autotriage.py` or
`app/harness/loop.py` reaches about 650 of 930 test files. `direct` keeps only
the test files that import the changed module themselves or name the path —
cheap enough to matter, and exactly the selection whose misses need counting.

What it records (`test_selection_shadow` ledger row, and `selection_shadow` in
the rung's data): whether the selection would have been a full run and why, how
many test files each selection kept of how many, and — the numbers that decide
promotion — `missed` / `missed_direct`: the red nodes the change is blamed for
(failed, not reproduced at the base, not flaky) whose file that selection left
out.

Mode `automod.gate.test_selection`: `shadow` (ships) or `off`. A round may not
change it (`spec.GATE_SELECTION_DENIED_KEYS`). There is no `select` mode: that
is a separate change, made only on the evidence this one collects.
"""
from __future__ import annotations

import ast
import re
import time
from pathlib import Path

MODES = ("off", "shadow")
EVENT = "test_selection_shadow"

#: Where pytest collects (`pytest.ini` `testpaths`), and the packages whose
#: modules can be imported by a test. Anything else under the tree is not read.
TEST_ROOTS = ("tests", "app/harness/tests", "scripts")
PACKAGES = ("app", "agent_mcp", "scripts", "workers", "eval", "tests")

_FULL_RUN_NAMES = ("conftest.py", "pytest.ini", "config.yaml", "setup.cfg",
                   "pyproject.toml", "requirements.txt", "requirements.lock", "uv.lock")


def mode(value) -> str:
    """`off` only when said; anything else — a typo included — is `shadow`,
    which changes no verdict."""
    if value is False or str(value).strip().lower() in ("off", "false", "0", "no"):
        return "off"
    return "shadow"


def configured_mode() -> str:
    try:
        from app.config import load_config
        gate = ((load_config() or {}).get("automod") or {}).get("gate") or {}
        return mode(gate.get("test_selection", "shadow"))
    except Exception:  # noqa: BLE001 — a measurement, never a reason to fail
        return "shadow"


def is_test_file(rel: str) -> bool:
    p = Path(rel)
    return (p.suffix == ".py" and p.name.startswith("test_")
            and any(rel == r or rel.startswith(r + "/") for r in TEST_ROOTS))


def full_run_reason(changed: list[str]) -> str:
    """Why the selection would have been the whole suite, or "" when it would not."""
    for rel in changed:
        p = Path(rel)
        if p.name in _FULL_RUN_NAMES or p.name.startswith("requirements"):
            return f"{rel}: a file every test may read"
        if p.suffix == ".py" and p.name.startswith("_") and rel.startswith("tests/"):
            return f"{rel}: a shared test helper"
        if p.suffix not in (".py", ".md"):
            return f"{rel}: not Python or Markdown, invisible to the import graph"
        if p.suffix == ".py" and not rel.startswith(PACKAGES):
            return f"{rel}: Python outside the importable packages"
    return ""


def module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _python_files(root: Path) -> list[str]:
    out: list[str] = []
    for pkg in PACKAGES:
        base = root / pkg
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            if "/node_modules/" in rel or "/.venv" in rel or "/__pycache__/" in rel:
                continue
            out.append(rel)
    return sorted(out)


def _imports(rel: str, source: str, known: set[str]) -> set[str]:
    """Repo modules `rel` imports, resolved against the set of known modules.
    `from a.b import c` counts `a.b.c` when that is a module, else `a.b`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    here = module_name(rel).split(".")
    is_pkg = rel.endswith("__init__.py")
    found: set[str] = set()

    def add(name: str) -> None:
        while name:
            if name in known:
                found.add(name)
                return
            name = name.rpartition(".")[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = here if is_pkg else here[:-1]
                pkg = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                base = ".".join(pkg + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            for a in node.names:
                full = f"{base}.{a.name}" if base else a.name
                if full in known:
                    found.add(full)
                else:
                    add(base)
    found.discard(module_name(rel))
    return found


def select(root: Path, changed: list[str]) -> dict:
    """`{full_run, reason, selected, test_files, seconds}`; `selected` is a sorted
    list of test files (empty on a full run, where every file is selected)."""
    started = time.monotonic()
    files = _python_files(root)
    tests = [f for f in files if is_test_file(f)]
    reason = full_run_reason(changed)
    if reason:
        return {"full_run": True, "reason": reason, "selected": [], "direct": [],
                "test_files": len(tests), "seconds": round(time.monotonic() - started, 2)}
    sources: dict[str, str] = {}
    for rel in files:
        try:
            sources[rel] = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            sources[rel] = ""
    by_module = {module_name(f): f for f in files}
    known = set(by_module)
    importers: dict[str, set[str]] = {}
    for rel in files:
        for mod in _imports(rel, sources[rel], known):
            importers.setdefault(mod, set()).add(rel)

    selected: set[str] = set()
    direct: set[str] = set()
    for rel in changed:
        if is_test_file(rel) and rel in sources:
            selected.add(rel)
            direct.add(rel)
        needles = [rel]
        if rel.endswith(".py"):
            mod = module_name(rel)
            needles.append(mod)
            direct |= {f for f in importers.get(mod, ()) if is_test_file(f)}
            seen, todo = {rel}, [mod]
            while todo:
                for importer in importers.get(todo.pop(), ()):
                    if importer not in seen:
                        seen.add(importer)
                        todo.append(module_name(importer))
            selected |= {f for f in seen if is_test_file(f)}
        pattern = re.compile("|".join(re.escape(n) for n in needles))
        mentions = {t for t in tests if pattern.search(sources.get(t, ""))}
        selected |= mentions
        direct |= mentions
    return {"full_run": False, "reason": "", "selected": sorted(selected),
            "direct": sorted(direct),
            "test_files": len(tests), "seconds": round(time.monotonic() - started, 2)}


def blamed_nodes(data: dict) -> list[str]:
    """The red nodes the change is blamed for: failed, not reproduced at the
    base, not flaky. A pre-existing failure is not the change's to miss."""
    failed = list(data.get("failed_node_ids") or [])
    not_ours = set(data.get("pre_existing_failures") or []) | set(data.get("flaky_node_ids") or [])
    return [n for n in failed if n not in not_ours]


def shadow(root: Path, changed: list[str], data: dict) -> dict:
    sel = select(root, list(changed))
    blamed = blamed_nodes(data or {})
    def outside(files: list[str]) -> list[str]:
        if sel["full_run"]:
            return []
        chosen = set(files)
        return [n for n in blamed if n.split("::", 1)[0] not in chosen]
    return {"full_run": sel["full_run"], "reason": sel["reason"],
            "selected_files": len(sel["selected"]), "direct_files": len(sel["direct"]),
            "test_files": sel["test_files"], "blamed": len(blamed),
            "missed": outside(sel["selected"]), "missed_direct": outside(sel["direct"]),
            "seconds": sel["seconds"], "changed": len(changed)}


def attach(gate_cls) -> None:
    """Wrap `gate_cls.rung_tests` so a finished rung is shadowed. The wrapper
    returns exactly what the rung returned, whatever happens inside it."""
    original = gate_cls.rung_tests
    if getattr(original, "_selection_shadow", False):
        return

    def rung_tests(self, only=None):
        result = original(self, only)
        try:
            _record(self, only, result)
        except Exception as exc:  # noqa: BLE001 — a shadow never costs a rung
            print(f"[warn] test selection shadow: {type(exc).__name__}: {exc}")
        return result

    rung_tests._selection_shadow = True  # type: ignore[attr-defined]
    rung_tests.__doc__ = original.__doc__
    gate_cls.rung_tests = rung_tests


def _record(gate, only, result) -> None:
    if only is not None or configured_mode() == "off":
        return  # the rung already ran a selection of its own (`_tests_delta_only`)
    ok, _detail, data = result
    if not isinstance(data, dict) or data.get("skipped") is True:
        return
    changed = list(getattr(getattr(gate, "report", None), "changed_paths", None) or [])
    if not changed:
        return
    row = shadow(Path(gate.worktree), changed, data)
    data["selection_shadow"] = row
    from scripts.automod import state as S
    S.append_event({"event": EVENT, "round_id": getattr(gate, "round_id", ""),
                    "rung_ok": bool(ok), **row})


def report(events: list[dict]) -> dict:
    """The promotion question over recorded rows: how many shadowed rungs, how
    many would have selected, how many tests they would have skipped on
    average, and every miss."""
    rows = [e for e in events if e.get("event") == EVENT]
    selecting = [r for r in rows if not r.get("full_run")]
    red = [r for r in selecting if r.get("blamed")]

    def mean_share(key: str):
        vals = [r[key] / r["test_files"] for r in selecting if r.get("test_files") and key in r]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {"rungs": len(rows), "selecting": len(selecting),
            "selecting_with_blamed_red": len(red),
            "transitive": {"mean_selected_share": mean_share("selected_files"),
                           "misses": [(r.get("round_id"), n) for r in rows
                                      for n in r.get("missed") or []]},
            "direct": {"mean_selected_share": mean_share("direct_files"),
                       "misses": [(r.get("round_id"), n) for r in rows
                                  for n in r.get("missed_direct") or []]}}


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Affected-test selection: shadow report or a dry run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report", help="read the ledger's shadow rows")
    p = sub.add_parser("select", help="what a selection would run for these paths")
    p.add_argument("paths", nargs="+")
    p.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        from scripts.automod import state as S
        print(json.dumps(report(list(S.read_events())), indent=2))
        return 0
    sel = select(Path(args.root).resolve(), args.paths)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in sel.items()}, indent=2))
    for f in sel["selected"]:
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
