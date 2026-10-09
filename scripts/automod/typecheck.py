"""A pyright type-check delta over what a round changed (backlog #2450).

The gate has the agent half of validation and none of the analyser half. Its
`static` rung is compileall, an import smoke and a **pyflakes** delta, and
pyflakes is name-resolution and unused-import level only. The edit-time rail
(#528) lists the callers of a changed symbol and its own Risks section names
the hole this module fills: *"Textual/AST blast radius is not type checking: a
bool→dataclass return change produces no finding anywhere in pyflakes or the
graph; only the caller list is reported, and judging it is still on the model."*
Nothing in the loop type-checked a diff until now.

The class is not hypothetical. D13 (`9595ddc3`) deleted the `env` field from
`RunOptions` while `eval/run_prefetch_cost_eval.py` went on passing it; pyright
sees that break on the caller's own line —

    error reportCallIssue: No parameter named "env"

— and reproducing it takes `git archive 725aa7f0^` plus one command (item #2450,
step 2). It is the shape every behavioural rung misses until a test happens to
exercise the caller.

Three rules, taken from the rungs this one is modelled on:

  * **A delta, never absolute.** The tree carries 1,837 pyright findings in
    Lloyd's own 316 files (measured 2026-10-09, pyright 1.1.409, `basic` mode).
    An absolute bar fails every round forever, so what is reported is the
    multiset difference between the round's head and the *same run* at its base
    commit — the discipline `rung_static` applies to pyflakes and `rung_frontend`
    to tsc.
  * **A multiset keyed on rule + symbol + file, never on position.** Two
    identical findings on two lines are two breaks; one finding that only moved
    down the file is not a finding at all (#1210 is the pyflakes version of
    that bug). `app.lint_findings.parse_pyright` owns the key.
  * **Over the changed files plus their inbound callers.** A signature break
    lives in the caller, so a file set drawn from the diff alone would never
    see it. Which files those are is #528's own discipline — module-level
    symbols whose interface moved — imported rather than copied, so the rail
    and the gate cannot disagree about what "changed" means.

Not here, deliberately:

  * A `pyrightconfig.json`. Measured without one: `--pythonpath` resolves the
    venv, and 1.1.409's CLI has no flag for `typeCheckingMode` or any `report*`
    rule, so cutting the rule set needs a config file. That file is a new
    tracked root path and whether a round may edit it is owed-check clause 3 on
    #2450 — not this round's call. Until then the rule set is whatever `basic`
    mode ships, and the rung is observe-only precisely because nobody has yet
    measured how often that set is wrong.
  * The vendored fork. 68,003 of the tree's 69,840 findings are under
    :data:`FORK_PREFIX` (97% of the output, and most of a 330-second whole-tree
    run). It is excluded from the file set by rule, not by luck: it is a real
    inbound caller of tracked modules.
  * Any block. This is a record during the soak (#679's precedent). The flag
    rate over real rounds is what decides whether it may ever refuse one, and
    #2450's acceptance says a zero flag rate closes the item rather than
    promoting the rung.

Never raises: a checker that cannot run returns `status="unevaluated"` with the
reason, because a check that did not run must not be read as a clean one.

Run it by hand over any round — see :func:`main` for the exact invocation,
which takes the base tree as an argument rather than assuming one.
"""

from __future__ import annotations

import ast
import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agent_mcp._edit_diagnostics import changed_symbols as _changed_symbols
from app import lint_findings
from scripts.automod import worktree as W

#: The vendored vLLM fork: untracked, not ours, and the source of 68,003 of the
#: tree's 69,840 pyright findings — 627 of them in one of its test files.
FORK_PREFIX = "agent-services/llm/djev-vllm-fork/"

EVALUATED = "evaluated"
UNEVALUATED = "unevaluated"
SKIPPED = "skipped"

#: Ceiling on files per invocation. pyright is a Node process doing
#: whole-program analysis, and the rung budget this ladder documents is ~8 s;
#: 2 s was one file (measured 2026-10-09). Changed files are always in; past
#: the ceiling it is callers that get dropped, and the count is recorded rather
#: than silently lost.
MAX_FILES = 120

#: How many findings the record names. Compact, like the vet's `labels`, so a
#: rung on every round cannot bloat the ledger.
MAX_LABELS = 20
MAX_NEW_LISTED = 40


def in_fork(path: str) -> bool:
    p = (path or "").replace("\\", "/").lstrip("./")
    return p == FORK_PREFIX.rstrip("/") or p.startswith(FORK_PREFIX)


def module_of(path: str) -> str:
    """`app/harness/options.py` → `app.harness.options`; a package's `__init__` → the package.

    Derived from the path alone, not from `__init__.py` being present, because
    a namespace package is legal and a scratch repo has no packages at all.
    """
    stem = path[:-3] if path.endswith(".py") else path
    parts = [p for p in stem.replace("\\", "/").split("/") if p]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(module: str, path: str) -> list[str]:
    """The packages `module` sits in, innermost first, for relative imports.

    `app/harness/options.py` → `["app.harness", "app", ""]`, so `from . import x`
    resolves at index 0 and `from .. import x` at index 1. A file at the tree
    root has one entry, the empty string, which is the root package.
    """
    parts = module.split(".")
    if not path.endswith("__init__.py"):
        parts = parts[:-1]                    # a module is not its own package
    out = [".".join(parts[:n]) for n in range(len(parts), 0, -1)]
    out.append("")                            # above the top level is the root
    return out


def imported_names(source: str, importer_path: str) -> set[str]:
    """Every dotted name this file imports, resolved enough to match an edge.

    `from runopts import RunOptions` yields `runopts` *and* `runopts.RunOptions`
    (the latter is how `from app.harness import options` names the changed
    module itself). Relative imports are resolved against the importer's own
    package, which is how a sibling inside `app/` reaches a changed module.
    """
    try:
        tree = ast.parse(source)
    except Exception:
        return set()
    me = module_of(importer_path)
    packages = _package_of(me, importer_path)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                owner = packages[node.level - 1] if node.level - 1 < len(packages) else ""
                base = f"{owner}.{base}" if owner and base else (owner or base)
            if base:
                out.add(base)
                for alias in node.names:
                    out.add(f"{base}.{alias.name}")
    return out


def imports_a_changed(imported: set[str], modules: set[str]) -> bool:
    """An edge exists when an import names a changed module or lies under it.

    The reverse (`import app` while `app.harness.options` changed) is
    deliberately not an edge: it would make every importer of a top-level
    package a caller of every module inside it, which is the whole tree.
    """
    for imp in imported:
        for mod in modules:
            if imp == mod or imp.startswith(mod + "."):
                return True
    return False


def tracked_py(root: Path) -> list[str] | None:
    """Tracked `.py` paths in `root`, fork excluded — or None if git could not say.

    `git ls-files` rather than a walk: it is the tracked set the round can
    touch, and a walk would come back through `.venvs`, `web/node_modules`, the
    qmd fork and `.git`.
    """
    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
                           capture_output=True, timeout=120, check=False)
    except Exception:  # noqa: BLE001 — the caller reports it as unevaluated
        return None
    if r.returncode != 0:
        return None
    return [p for p in (x.decode("utf-8", "replace") for x in r.stdout.split(b"\0") if x)
            if p and not in_fork(p)]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def changed_module_symbols(head_tree: Path, base_tree: Path,
                           changed_py: list[str]) -> dict[str, set[str]]:
    """`{module: names whose module-level interface moved}` for this round.

    #528's fingerprinting, asked of the base/head pair instead of an edit's
    pre/post image. A name added by the round is not here — and cannot be a
    caller-break, because nothing at base could have imported it.
    """
    out: dict[str, set[str]] = {}
    for rel in changed_py:
        head_src = _read(head_tree / rel)
        if not head_src:
            continue
        base_src = _read(base_tree / rel) if (base_tree / rel).exists() else ""
        for name in _changed_symbols(base_src, head_src):
            top = name.split(".")[0]
            if top:
                out.setdefault(module_of(rel), set()).add(top)
    return out


def caller_files(head_tree: Path, changed_py: list[str], tracked: list[str],
                 modules: dict[str, set[str]]) -> list[str]:
    """Tracked files outside the diff that import a changed module *and* name one
    of its changed symbols.

    Both halves are needed: the import edge alone would drag in every file that
    touches a popular module (and the run is whole-program, so each one costs
    analysis), and the name alone would fire on an unrelated identifier. The
    pairing is a conservative narrowing — over-inclusion costs runtime only,
    since pyright itself decides whether the caller actually breaks.
    """
    if not modules:
        return []
    names = {n for s in modules.values() for n in s}
    targets = set(modules)
    changed = set(changed_py)
    out: list[str] = []
    for rel in tracked:
        if rel in changed or in_fork(rel):
            continue
        text = _read(head_tree / rel)
        if not text or not any(n in text for n in names):
            continue
        if imports_a_changed(imported_names(text, rel), targets):
            out.append(rel)
    return sorted(out)


def _run_pyright(tree: Path, files: list[str], *, python: Path, pyright: Path,
                 timeout: float) -> tuple[Counter | None, str, dict]:
    """One pyright run over `files` in `tree`. Returns (findings, error, stats)."""
    if not files:
        return Counter(), "", {"seconds": 0.0}
    started = time.monotonic()
    cmd = [str(pyright), "--pythonpath", str(python), "--outputjson", *files]
    try:
        r = subprocess.run(cmd, cwd=str(tree), env=lint_findings.node_env(),
                           capture_output=True, text=True, timeout=timeout,
                           check=False)
    except Exception as exc:  # noqa: BLE001 — reported as unevaluated, never as clean
        return None, f"pyright did not run ({type(exc).__name__}: {exc})", {}
    stats: dict[str, object] = {"seconds": round(time.monotonic() - started, 2)}
    payload = lint_findings.pyright_payload(r.stdout)
    unusable = lint_findings.parse_pyright_unusable(r.stdout)
    if unusable:
        tail = (unusable or (r.stderr or "").strip())[:200]
        return None, f"pyright gave no JSON (exit {r.returncode}): {tail}", stats
    if payload:
        summary = payload.get("summary") or {}
        stats["files_analyzed"] = summary.get("filesAnalyzed")
        stats["version"] = payload.get("version")
    return lint_findings.parse_pyright(r.stdout, root=tree), "", stats


def _entries(delta: Counter) -> list[dict]:
    """The delta as sorted, named entries — file, rule, message, count."""
    out = []
    for key, n in delta.items():
        f, rule, message = lint_findings.split_pyright_key(key)
        out.append({"file": f, "rule": rule, "message": message, "count": n})
    out.sort(key=lambda d: (d["file"], d["rule"], d["message"]))
    return out[:MAX_NEW_LISTED]


@dataclass
class TypeCheckResult:
    """What the checker concluded — including whether it got to conclude anything."""

    status: str = EVALUATED
    reason: str = ""
    files: list[str] = field(default_factory=list)
    caller_files: list[str] = field(default_factory=list)
    new: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    totals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {"status": self.status, "files": list(self.files),
                     "caller_files": list(self.caller_files),
                     "new": list(self.new), "counts": dict(self.counts),
                     "labels": list(self.labels), "totals": dict(self.totals)}
        if self.reason:
            out["reason"] = self.reason
        return out


def _unevaluated(reason: str, totals: dict | None = None) -> TypeCheckResult:
    """A check that could not complete, which is never the same as a clean one."""
    return TypeCheckResult(status=UNEVALUATED, reason=str(reason)[:400],
                           totals=totals or {})


def check_round(*, base: str, worktree: Path, base_tree: Path | None,
                changed: list[str], python: Path, pyright: Path,
                timeout: float = 120.0) -> TypeCheckResult:
    """The pyright delta for one round: run at head, run at base, subtract.

    `base_tree` is a checkout of `base` that the caller made and owns — the
    run needs a whole tree, not the changed files alone, because a type checker
    resolves a caller's imports through the modules the round changed. (That is
    also why this cannot reuse `rung_static`'s scratch-dir-of-blobs trick:
    blobs without their package make every import unresolved, and the delta
    then measures the scratch directory.)

    Never raises. Everything that can go wrong comes back as
    `status="unevaluated"` with the reason, which is what keeps an observe-only
    rung from ever failing a round while still telling the truth about whether
    it ran.
    """
    changed_py = sorted({p for p in changed
                         if p.endswith(".py") and not in_fork(p)})
    if not changed_py:
        return TypeCheckResult(status=SKIPPED, reason="no python changed",
                               totals={"files": 0, "callers": 0})
    if not base_tree or not Path(base_tree).is_dir():
        return _unevaluated(f"no checkout of {str(base)[:12]} to compare against",
                            {"changed_py": len(changed_py)})
    tracked = tracked_py(Path(worktree))
    if tracked is None:
        return _unevaluated("git ls-files failed; the caller set is unknown",
                            {"changed_py": len(changed_py)})

    modules = changed_module_symbols(Path(worktree), Path(base_tree), changed_py)
    callers = caller_files(Path(worktree), changed_py, tracked, modules)

    files = sorted(set(changed_py) | set(callers))
    omitted = 0
    if len(files) > MAX_FILES:
        room = max(MAX_FILES - len(changed_py), 0)
        files = sorted(set(changed_py) | set(callers[:room]))
        omitted = len(set(changed_py) | set(callers)) - len(files)
    checked_callers = [c for c in callers if c in files]

    # Named `_f` and not `head`/`base` because `base` is this function's own
    # SHA parameter. Rebinding it to the base multiset is the shadowing bug this
    # rung exists to catch, and the rung reported it here first, on the round
    # that adds it: the first delta over this file's own round was 5 findings
    # over 7 files, 3 of them the shadow (`Operator "-" not supported for types
    # "Counter[Unknown]" and "str"`, `Cannot access attribute "values" for class
    # "str"`, one `reportAssignmentType`) and 2 the untyped `stats` dict.
    head_f, head_err, head_stats = _run_pyright(Path(worktree), files, python=python,
                                             pyright=pyright, timeout=timeout)
    if head_f is None:
        return _unevaluated(f"head run: {head_err}",
                            {"changed_py": len(changed_py), "files": len(files),
                             "callers": len(callers)})
    # The same file set at base, minus what the round added there.
    base_files = [f for f in files if (Path(base_tree) / f).exists()]
    base_f, base_err, base_stats = _run_pyright(Path(base_tree), base_files,
                                              python=python, pyright=pyright,
                                              timeout=timeout)
    if base_f is None:
        return _unevaluated(f"base run: {base_err}",
                            {"changed_py": len(changed_py), "files": len(files),
                             "callers": len(callers), **{f"head_{k}": v for k, v in
                                                         head_stats.items()}})

    delta = head_f - base_f
    counts: Counter = Counter()
    for key, n in delta.items():
        counts[lint_findings.split_pyright_key(key)[1]] += n
    totals = {
        "changed_py": len(changed_py),
        "files": len(files),
        "callers": len(checked_callers),
        "callers_omitted": len(callers) - len(checked_callers),
        "files_omitted": omitted,
        "head_findings": sum(head_f.values()),
        "base_findings": sum(base_f.values()),
        "new_findings": sum(delta.values()),
        "head_seconds": head_stats.get("seconds", 0.0),
        "base_seconds": base_stats.get("seconds", 0.0),
        "seconds": round(float(head_stats.get("seconds", 0.0))
                         + float(base_stats.get("seconds", 0.0)), 2),
    }
    if head_stats.get("files_analyzed") is not None:
        totals["files_analyzed"] = head_stats["files_analyzed"]
    if head_stats.get("version"):
        totals["pyright_version"] = head_stats["version"]
    labels = sorted({f"{r}:{f}" for f, r, _m in
                     (lint_findings.split_pyright_key(k) for k in delta)})
    return TypeCheckResult(
        status=EVALUATED,
        files=files,
        caller_files=checked_callers,
        new=_entries(delta),
        counts=dict(counts),
        labels=labels[:MAX_LABELS],
        totals=totals,
    )


def main(argv: list[str]) -> int:
    """Report one round's pyright delta by hand. Prints JSON; always exits 0.

        git -C ~/lloyd worktree add --detach /tmp/base-<sha> <base-ref>
        .venvs/lloyd/bin/python -m scripts.automod.typecheck \
            <worktree> <base-ref> /tmp/base-<sha>

    The base tree is an argument and not something this makes: the gate owns
    scratch paths, and a second writer of detached checkouts in the live repo
    is how a round's cleanup ends up removing somebody else's. Passing the
    worktree as its own base tree is refused outright — it is a delta against
    itself, which is a guaranteed empty answer that reads as a clean tree.
    """
    if len(argv) < 3:
        print(__doc__)
        return 2
    wt, base, base_tree = Path(argv[0]), argv[1], Path(argv[2])
    if wt.resolve() == base_tree.resolve():
        print("refusing: the base tree IS the worktree, so the delta would be "
              "empty whatever the round broke")
        return 2
    live = Path.home() / "lloyd"
    res = check_round(
        base=base, worktree=wt, base_tree=base_tree, changed=_diff(wt, base),
        python=live / ".venvs" / "lloyd" / "bin" / "python",
        pyright=live / ".venvs" / "lloyd" / "bin" / "pyright")
    print(json.dumps(res.to_dict(), indent=2, sort_keys=True))
    return 0


def _diff(worktree: Path, base: str) -> list[str]:
    return W.changed_paths(worktree, base)


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(main(sys.argv[1:]))
