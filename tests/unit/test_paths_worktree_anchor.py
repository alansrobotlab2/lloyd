"""app/paths.py — the tree the paths anchor to has to announce itself (#733).

`LLOYD_HOME` is derived from the location of the imported module. That is right
for a running service and quietly wrong for a script a self-modification round
runs out of its own worktree: the worktree's `sessions/` exists and is empty, so
a scan of it returns clean, plausible and false — the #529 replay printed
`no session file matching ... in /…/SM_20260910_004029/home/lloyd/sessions` for a
500,934-byte file that is in the live checkout, and an hour later reported
`cumulative_prompt_tokens: null` for an arm. The shape is the one this repo has a
rule about: a guard that reads its own missing input reports OK.

So this pins *legibility*, not retargeting. `IS_WORKTREE` names the tree and the
import says so out loud; every path constant keeps resolving inside the imported
tree exactly as before, because state following the code is what the canary boot
depends on (`scripts/automod/worktree.py:1-10`: the `<round>/home/lloyd` layout
exists precisely so `app.paths.LLOYD_HOME` and `HOME=<round>/home` are the same
directory).

Every case imports `app.paths` in a FRESH SUBPROCESS. `IS_WORKTREE` and the
import-time warning are decided once, while the module body runs, and this
session has already imported the real module — an in-process re-import could
show neither, and a `caplog` assertion would capture nothing. The subprocess is
the seam; the probe wires its handler to the root logger before the import.

The trees are synthesised under `tmp_path` from the real `app/paths.py` source,
because the discriminator is on disk and nothing else: a linked git worktree
keeps `.git` as a FILE (`gitdir: …`), the main checkout keeps it as a DIRECTORY.
No case imports the live `~/lloyd` checkout — inside a round that would import
the pre-fix module sitting in the live tree and fail the gate on its own round.

Run:
  /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/unit/test_paths_worktree_anchor.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PATHS_PY = REPO / "app" / "paths.py"
PKG_INIT = REPO / "app" / "__init__.py"
LIVE_CHECKOUT = Path.home() / "lloyd"

#: Prints one `RESULT<json>` line describing what the import decided. The
#: handler goes on the root logger BEFORE the import, because the record is
#: emitted while the module body runs.
_PROBE = '''"""Import app.paths and report what it announced."""
import json
import logging


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


_collector = _Collect()
_root = logging.getLogger()
_root.addHandler(_collector)
_root.setLevel(logging.DEBUG)

import app.paths as paths  # noqa: E402  after the handler is wired

print("RESULT" + json.dumps({
    "module": paths.__file__,
    "LLOYD_HOME": str(paths.LLOYD_HOME),
    "IS_WORKTREE": paths.IS_WORKTREE,
    "IS_WORKTREE_TYPE": type(paths.IS_WORKTREE).__name__,
    "SESSIONS_DIR": str(paths.SESSIONS_DIR),
    "AUTONOMY_RUNS_DIR": str(paths.AUTONOMY_RUNS_DIR),
    "VAULT_DERIVED_ROOT": str(paths.VAULT_DERIVED_ROOT),
    "describe_tree": paths.describe_tree(),
    "warnings": [r.getMessage() for r in _collector.records
                 if r.levelno == logging.WARNING],
}))
'''


def _probe(tree: Path, *, probe_in: Path | None = None) -> dict:
    """Import `app.paths` with `tree` on the path and report what it announced.

    `probe_in` keeps the probe script itself out of `tree` when `tree` is a real
    checkout; the module under test still comes from `tree` via `PYTHONPATH`.
    """
    tree = tree.resolve()
    script_dir = (probe_in or tree).resolve()
    script = script_dir / "_probe_paths_tree.py"
    script.write_text(_PROBE)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    # Env overrides would repoint the derived dirs and the probe would then be
    # measuring the override, not the tree.
    for var in ("LLOYD_FACTS_ROOT", "LLOYD_KG_DB", "LLOYD_RESEARCH_DB", "LLOYD_ROOT",
                "LLOYD_DATA"):
        env.pop(var, None)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(tree), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"importing app.paths from {tree} failed (rc={proc.returncode}):\n"
        f"--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}"
    )
    hits = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert hits, f"probe printed no RESULT line:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    out = json.loads(hits[0][len("RESULT"):])
    # Positive control: without this every assertion below could pass trivially
    # against whichever app.paths happened to win sys.path.
    assert Path(out["module"]).resolve().is_relative_to(tree), (
        f"the probe imported {out['module']}, not the app/paths.py in {tree}"
    )
    return out


def _make_tree(root: Path, *, gitfile: bool) -> Path:
    """A tree holding the module under test, `.git` as a file or a directory.

    The `gitfile` form is what `git worktree add` leaves behind; the main
    checkout has a directory. The *content* of the git metadata is never read,
    only its form — that is the whole discriminator, and it is why this needs no
    live-root computation and no `$HOME` preference.
    """
    tree = root / ("linked-worktree" if gitfile else "main-checkout")
    (tree / "app").mkdir(parents=True)
    shutil.copyfile(PKG_INIT, tree / "app" / "__init__.py")
    shutil.copyfile(PATHS_PY, tree / "app" / "paths.py")
    if gitfile:
        (tree / ".git").write_text("gitdir: /nonexistent/.git/worktrees/lloyd9\n")
    else:
        (tree / ".git").mkdir()
        (tree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return tree


@pytest.fixture()
def worktree_tree(tmp_path: Path) -> dict:
    tree = _make_tree(tmp_path, gitfile=True)
    return {"tree": tree, "out": _probe(tree)}


@pytest.fixture()
def main_tree(tmp_path: Path) -> dict:
    tree = _make_tree(tmp_path, gitfile=False)
    return {"tree": tree, "out": _probe(tree)}


# ------------------------------------------------------- clause 1: the boolean --

def test_is_worktree_is_true_for_a_linked_worktree(worktree_tree):
    """`.git` as a gitfile is a linked worktree, and the module says so."""
    out = worktree_tree["out"]
    assert out["IS_WORKTREE"] is True, (
        f"a linked worktree must report IS_WORKTREE True; got {out['IS_WORKTREE']!r} "
        f"({out['IS_WORKTREE_TYPE']}) from {worktree_tree['tree']}"
    )


def test_is_worktree_is_false_for_the_main_checkout(main_tree):
    """The main checkout's `.git` is a directory, and is not a worktree."""
    out = main_tree["out"]
    assert out["IS_WORKTREE"] is False, (
        f"the main checkout must report IS_WORKTREE False; got {out['IS_WORKTREE']!r} "
        f"({out['IS_WORKTREE_TYPE']}) from {main_tree['tree']}"
    )


# ---------------------------------------- clause 2: name the tree, do not move --

def test_every_path_still_resolves_inside_the_worktree(worktree_tree):
    """The fix names the tree; it does not retarget it toward `$HOME/lloyd`.

    The forbidden shape is making `app.paths` prefer the live checkout: the
    canary boot imports its own tree on purpose, and state following the code is
    what stops a canary claiming the live `workers.db` jobs and writing into the
    live `sessions/`. So these constants are asserted to stay under the worktree
    AND not under the live checkout.
    """
    tree = worktree_tree["tree"]
    out = worktree_tree["out"]
    resolved = {
        "LLOYD_HOME": out["LLOYD_HOME"],
        "SESSIONS_DIR": out["SESSIONS_DIR"],
        "AUTONOMY_RUNS_DIR": out["AUTONOMY_RUNS_DIR"],
        "VAULT_DERIVED_ROOT": out["VAULT_DERIVED_ROOT"],
    }
    for name, value in resolved.items():
        assert value == str(tree / {
            "LLOYD_HOME": "",
            # The worktree's own data root (app.paths rule 3), never production's.
            "SESSIONS_DIR": ".lloyd-data/sessions",
            "AUTONOMY_RUNS_DIR": ".lloyd-data/autonomy-runs",
            "VAULT_DERIVED_ROOT": ".lloyd-data/_pipeline/vault-derived",
        }[name]).rstrip("/"), f"{name} moved off the imported tree: {value}"

    live = str(LIVE_CHECKOUT.resolve())
    for name, value in resolved.items():
        assert value != live and not value.startswith(live + os.sep), (
            f"{name} was retargeted toward the live checkout ({value}); the anchor "
            "must stay on the imported tree and only announce itself"
        )


# ---------------------------------------------------- clause 3: the loud import --

def test_a_worktree_import_warns_once_naming_its_root(worktree_tree):
    """Exactly one WARNING-level record, and it carries the absolute root."""
    tree = worktree_tree["tree"]
    warnings = worktree_tree["out"]["warnings"]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING-level record from a worktree import, got "
        f"{len(warnings)}: {warnings}"
    )
    assert str(tree) in warnings[0], (
        f"the warning must name the tree it is about; it said: {warnings[0]!r}"
    )


def test_a_main_checkout_import_emits_no_warning(main_tree):
    """Quiet where nothing is wrong, or the warning stops being news."""
    assert main_tree["out"]["warnings"] == [], (
        f"the main checkout import must be silent, got: {main_tree['out']['warnings']}"
    )


# ------------------------------------------- clause 4: a line for an artifact --

def _tree_word(text: str) -> str:
    """Pull the `tree=<kind>` token out of describe_tree()'s line."""
    match = re.search(r"\btree=([A-Za-z-]+)", text or "")
    assert match, f"describe_tree() must carry a `tree=<kind>` token, got {text!r}"
    return match.group(1)


def test_describe_tree_names_the_worktree_root(worktree_tree):
    """An aggregate artifact can carry the tree it was measured on."""
    text = worktree_tree["out"]["describe_tree"]
    assert str(worktree_tree["tree"]) in text, f"root missing: {text!r}"
    assert _tree_word(text) == "worktree", f"wrong kind word: {text!r}"


def test_describe_tree_says_live_from_a_main_checkout(main_tree):
    text = main_tree["out"]["describe_tree"]
    assert str(main_tree["tree"]) in text, f"root missing: {text!r}"
    assert _tree_word(text) == "live", f"wrong kind word: {text!r}"


# ------------------------------------- the reproduction, on the tree under test --

def test_the_checkout_under_test_announces_itself(tmp_path: Path):
    """#733's one-line reproduction, run against the tree running this suite.

    `cd <tree> && python -c "from app.paths import SESSIONS_DIR"` used to print a
    bare path and a count with nothing saying which tree it was. Here it
    announces itself if and only if that tree is a linked worktree — so in a
    round the warning is present, and in a main checkout the import stays quiet.
    """
    out = _probe(REPO, probe_in=tmp_path)
    assert out["LLOYD_HOME"] == str(REPO), (
        f"the tree under test must anchor to itself: {out['LLOYD_HOME']} != {REPO}"
    )
    is_worktree = (REPO / ".git").is_file()
    assert out["IS_WORKTREE"] is is_worktree, (
        f"IS_WORKTREE={out['IS_WORKTREE']!r} disagrees with the tree on disk: "
        f"{REPO / '.git'} is {'a gitfile' if is_worktree else 'not a gitfile'}"
    )
    warnings = out["warnings"]
    assert bool(warnings) is is_worktree, (
        f"expected a warning exactly when the tree is a worktree "
        f"(is_worktree={is_worktree}), got {warnings}"
    )
    for message in warnings:
        assert str(REPO) in message, f"the warning must name its own tree: {message!r}"
    assert _tree_word(out["describe_tree"]) == ("worktree" if is_worktree else "live")
    assert str(REPO) in out["describe_tree"]
