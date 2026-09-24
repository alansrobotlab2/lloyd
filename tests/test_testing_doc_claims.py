"""What `architecture/testing.md` may say about where live data sits.

The doc explains the suite's two kinds of test, and its live-data paragraph
names the runtime roots those tests read. On 2026-09-22 runtime data left the
code tree for `~/lloyd-data` (`6426668b`); the sweep that followed said no doc
still put data in the tree, and this one did — `sessions/`, `_pipeline/`,
`eval/baselines/` and the KG store, tree-relative, for two days until an
arch-review pass caught it (#1439). Nothing pinned a line of the doc, so the
guard is the missing half, not the doc.

Two claims are pinned:

  * every root the live-data paragraph names is written under `~/lloyd-data/`
    and, read that way, is the `app.paths` constant under `DATA_ROOT` — never
    under `LLOYD_HOME`. The extractor is fed a tree-relative variant too, so the
    guard fails on the drift itself and not only on a missing paragraph;
  * the helpers the doc lists are the public functions `tests/_live_data.py`
    defines, compared by name only — the doc abbreviates signatures (it omits
    `noun=` and `what=` defaults), so a signature match would be red on a
    correct doc.

Fully synthetic: it reads repo files and `app.paths` names, never the data
root's contents, so it passes in a worktree where `~/lloyd-data` holds nothing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from app import paths as P

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "testing.md"
LIVE_DATA = ROOT / "tests" / "_live_data.py"

DATA_PREFIX = "~/lloyd-data/"

#: Each runtime root the live-data paragraph must name, and the `app.paths`
#: constant it has to be.
ROOTS = {
    "sessions/": P.SESSIONS_DIR,
    "_pipeline/": P.PIPELINE_DIR,
    "eval/baselines/": P.EVAL_BASELINES_DIR,
    "_pipeline/vault-derived/kg.sqlite": P.VAULT_KG_DB_DEFAULT,
}

_RUNTIME_HEADS = ("sessions", "_pipeline", "eval/baselines")


def live_data_paragraph(text: str) -> str:
    m = re.search(r"\*\*Live-data\*\* tests read.*?(?:\n\s*\n|\Z)", text, re.S)
    assert m, "architecture/testing.md has no **Live-data** paragraph"
    return m.group(0)


def root_claims(paragraph: str) -> tuple[dict[str, str], list[str]]:
    """({relative root: backticked spec}, [violations]) for the paragraph.

    A backticked path counts when, after an optional `~/lloyd-data/` or
    `~/lloyd/` prefix, it starts with a runtime root. Anything not written
    under `~/lloyd-data/` is a violation: it names data in the code tree.
    """
    found: dict[str, str] = {}
    violations: list[str] = []
    for spec in re.findall(r"`([^`\s]+)`", paragraph):
        rel = spec
        for prefix in (DATA_PREFIX, "~/lloyd/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
        if not rel.startswith(_RUNTIME_HEADS):
            continue
        if not spec.startswith(DATA_PREFIX):
            violations.append(spec)
            continue
        found[rel] = spec
    return found, violations


def test_every_named_root_is_written_under_the_data_root():
    found, violations = root_claims(live_data_paragraph(DOC.read_text()))
    assert not violations, f"tree-relative data roots in testing.md: {violations}"
    assert set(ROOTS) <= set(found), f"missing roots: {set(ROOTS) - set(found)}"


def test_each_root_resolves_to_its_paths_constant_under_data_root_not_the_tree():
    found, _ = root_claims(live_data_paragraph(DOC.read_text()))
    for rel, constant in ROOTS.items():
        assert rel in found
        resolved = P.DATA_ROOT / rel.rstrip("/")
        assert resolved == constant, f"{found[rel]} -> {resolved}, app.paths says {constant}"
        assert resolved.is_relative_to(P.DATA_ROOT)
        assert not resolved.is_relative_to(P.LLOYD_HOME), (
            f"{found[rel]} resolves inside the code tree ({P.LLOYD_HOME})"
        )


def test_a_tree_relative_root_is_reported_as_a_violation():
    """The 2026-09-22 form, through the same extractor."""
    good = live_data_paragraph(DOC.read_text())
    for rel in ROOTS:
        rotted = good.replace(f"`{DATA_PREFIX}{rel}`", f"`{rel}`")
        assert rotted != good, f"the paragraph no longer names {rel}"
        _, violations = root_claims(rotted)
        assert rel in violations
    _, violations = root_claims(good.replace(f"`{DATA_PREFIX}sessions/`", "`~/lloyd/sessions/`"))
    assert "~/lloyd/sessions/" in violations


def _documented_helpers(text: str) -> set[str]:
    section = text.split("## The rule for live data", 1)[1]
    block = re.search(r"```python\n(.*?)```", section, re.S)
    assert block, "the live-data rule section lost its python block"
    return set(re.findall(r"^(\w+)\(", block.group(1), re.M))


def _defined_helpers() -> set[str]:
    tree = ast.parse(LIVE_DATA.read_text())
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not n.name.startswith("_")}


def test_the_documented_helpers_are_the_defined_ones_by_name():
    documented = _documented_helpers(DOC.read_text())
    assert documented, "no helper names parsed from the doc"
    assert documented == _defined_helpers()
