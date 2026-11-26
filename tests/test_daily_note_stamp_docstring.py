"""#4279: the guardian `_stamp` docstring cites a file that exists.

It used to give `timestamp` a `str | None` type "per" a `scripts/vault/` schema module
that does not exist in this tree and has no deletion commit (the name is spelt in two
halves below, so this file is not itself a hit for clause 1's `git grep`). The
on-disk definer of the OKF `timestamp:` value is `derive_timestamp` in
`scripts/vault/okf_migrate.py`; that file types only its `_to_iso` helper `str | None`,
so the key type is not attributed to it either. `_stamp`'s three return statements are pinned byte-for-byte: this was a
docstring fix, and #3998's ruling (2026-11-18, declined) keeps the key naive-local.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "agent-services" / "guardian" / "daily_note.py"
OKF_MIGRATE = ROOT / "scripts" / "vault" / "okf_migrate.py"

DEAD = "okf_" + "schema"

_RETURNS = (
    'return now.strftime("%Y-%m-%dT%H:%M:%S")',
    'return now.isoformat() + "T00:00:00"',
    "return str(now)",
)


def _stamp_node() -> ast.FunctionDef:
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_stamp")


def test_no_python_file_cites_the_missing_schema_module():
    assert not (ROOT / "scripts" / "vault" / f"{DEAD}.py").exists()
    out = subprocess.run(["git", "grep", "-n", DEAD, "--", "*.py"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 1, f"{DEAD} is still cited:\n{out.stdout}{out.stderr}"


def test_the_stamp_docstring_names_the_real_timestamp_definer():
    doc = " ".join((ast.get_docstring(_stamp_node()) or "").split())
    assert "scripts/vault/okf_migrate.py" in doc and "derive_timestamp" in doc, doc
    assert "str | None" not in doc, (
        "the docstring attributes a `str | None` key type to okf_migrate.py, which types "
        "only its `_to_iso` helper that way")
    tree = ast.parse(OKF_MIGRATE.read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.FunctionDef) and n.name == "derive_timestamp"
               for n in tree.body), "okf_migrate.py no longer defines derive_timestamp"


def test_the_stamp_returns_are_unchanged():
    src = BUILDER.read_text(encoding="utf-8")
    node = _stamp_node()
    returns = [ast.get_source_segment(src, n).strip()
               for n in sorted((n for n in ast.walk(node) if isinstance(n, ast.Return)),
                               key=lambda n: n.lineno)]
    assert tuple(returns) == _RETURNS, returns
