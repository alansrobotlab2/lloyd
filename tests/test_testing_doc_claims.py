"""What `architecture/testing.md` may say about where live data sits.

The doc explains the suite's two kinds of test, and its live-data paragraph
names the runtime roots those tests read. On 2026-09-22 runtime data left the
code tree for `~/lloyd-data` (`6426668b`); the sweep that followed said no doc
still put data in the tree, and this one did — `sessions/`, `_pipeline/`,
`eval/baselines/` and the KG store, tree-relative, for two days until an
arch-review pass caught it (#1439). Nothing pinned a line of the doc, so the
guard is the missing half, not the doc.

Three claims are pinned:

  * every root the live-data paragraph names is written under `~/lloyd-data/`
    and, read that way, is the `app.paths` constant under `DATA_ROOT` — never
    under `LLOYD_HOME`. The extractor is fed a tree-relative variant too, so the
    guard fails on the drift itself and not only on a missing paragraph;
  * the helpers the doc lists are the public functions `tests/_live_data.py`
    defines, compared by name only — the doc abbreviates signatures (it omits
    `noun=` and `what=` defaults), so a signature match would be red on a
    correct doc;
  * the 2026-10-07 ruling in the production-tree section is still there and still
    true of the code (#2361): the refusal is absolute, `live_vault` has no
    sanctioned in-tree path, and scheduled vault-reading runs go through a
    detached on-disk worktree. That one resolves the guard's symbol and the
    opt-in's name against `tests/conftest.py` by AST instead of matching only
    prose, because the ruling it records had lived sole-sourced in one
    `owed_settled` entry on a CLOSED backlog item — the surface nobody greps —
    and the cites for it in this repo's own docs are how a reader would next look
    for it and find nothing.

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


# ── the live_vault ruling (#2361) ────────────────────────────────────────────
#
# The 2026-10-07 ruling on #1537, answered on #2189's owed entry: the
# production-tree refusal stays absolute. It was recorded in exactly one place —
# one `owed_settled` entry on a CLOSED backlog item, which is not a surface any
# later reader greps — while the doc that states the guard says nothing about
# what a run that needs a `live_vault` test is supposed to do. So the ruling went
# into `architecture/testing.md`, and this is the half that keeps it honest: the
# paragraph names a guard symbol and an opt-in variable, so both are resolved
# against `tests/conftest.py` by AST rather than trusted as prose.

PROD_TREE_SECTION = "## Never run the suite against the production tree"
GUARD_SYMBOL = "_refuse_the_production_tree"
CONFTEST = ROOT / "tests" / "conftest.py"

#: A dated sub-section inside the production-tree section. The date is required
#: by the shape, not by taste: an undated ruling paragraph cannot later be told
#: apart from the incident narrative around it, and these two sections are both
#: narratives about the same tree being deleted.
RULING = re.compile(r"^### (\d{4}-\d{2}-\d{2}):.*?$", re.M)

#: What the paragraph has to say, and the pattern that says it. Tolerant of
#: wording (a rephrase must not go red) and anchored on the claim itself (an
#: omission must). Each key is what a reader of the doc needs to learn.
CLAIMS = {
    "the refusal does not depend on which tests were selected":
        re.compile(r"not conditional on (?:the )?selection|"
                   r"(?:un)?conditional on (?:the )?selection"),
    "narrowing the run buys nothing":
        re.compile(r"(?:naming one file|one file)[^\n]{0,120}"
                   r"(?:buys nothing|changes nothing|is the same fixtures)|"
                   r"(?:same fixtures with the same teardowns)"),
    "no in-tree path exists or is planned for live_vault":
        re.compile(r"no (?:such )?(?:read-only )?in[- ]tree (?:way|path)[^\n]{0,60}"
                   r"(?:exists|planned)|(?:in-tree [a-z ]{0,20})?"
                   r"(?:none|nothing) is planned", re.I),
    "a vault-reading run reaches the suite from a detached on-disk worktree":
        re.compile(r"detached on-disk worktree|detached[^\n]{0,40}worktree"
                   r"[^\n]{0,40}~/lloyd-work/"),
}

#: An assertion that an in-tree carve-out is granted. The doc must not contain
#: it, and the ruling's whole content is that it cannot: the sentence below is
#: what a later edit that quietly reverses the ruling would look like.
CARVE_OUT_GRANTED = re.compile(
    r"(in-tree|inside `~/lloyd`)[^\n]{0,60}\b(?:is|are|was|were)\b"
    r"[^\n]{0,30}\b(?:permitted|sanctioned|allowed|granted)\b", re.I)

#: Line citations of the rotting kind. `tests/conftest.py:283` is true for one
#: commit; the seven cites in skills/nightly-skills-management/SKILL.md are what
#: one looks like after the file moves under them.
LINE_CITE = re.compile(r"conftest\.py:\d+")


def _section(text: str) -> str:
    """The production-tree section, up to the next level-two heading."""
    start = text.index(PROD_TREE_SECTION) + len(PROD_TREE_SECTION)
    nxt = re.search(r"^## ", text[start:], re.M)
    return text[start:start + (nxt.start() if nxt else len(text) - start)]


def ruling_paragraph(text: str) -> str:
    """The dated ruling sub-section inside it, or a failure naming what's missing.

    Raises rather than returns empty, so a doc that lost the paragraph reads as
    the premise coming back and not as a pattern that matched nothing.
    """
    section = _section(text)
    m = RULING.search(section)
    assert m, (
        "the production-tree section has no dated (### YYYY-MM-DD) sub-section, so "
        "the ruling that the guard is absolute is again recorded only in a closed "
        "backlog item's owed entry — the state #2361 exists to end"
    )
    end = re.search(r"^## ", section[m.end():], re.M)
    para = section[m.start():m.end() + (end.start() if end else len(section) - m.end())]
    assert "live_vault" in para, (
        "the dated sub-section at testing.md does not mention live_vault, so it is "
        "some other decision and not the ruling #1537 was owed"
    )
    return para


def test_the_live_vault_ruling_is_dated_and_says_all_four_things():
    para = ruling_paragraph(DOC.read_text())
    missing = [claim for claim, pat in CLAIMS.items() if not pat.search(para)]
    assert not missing, (
        f"the ruling paragraph no longer states: {'; '.join(missing)}. It is the "
        "answer to #1537 and the only durable surface for it"
    )
    assert LINE_CITE.findall(para) == [], (
        f"line citations back in the ruling ({LINE_CITE.findall(para)}) — cite the "
        "symbol; conftest.py line numbers are how every cite in this repo's docs "
        "went stale, most recently seven of them in one SKILL.md paragraph"
    )


def test_the_ruling_names_the_guard_and_the_opt_in_the_tree_actually_defines():
    """Clause 2's cross-boundary half: the paragraph's names, checked against AST.

    Prose that says `_refuse_the_production_tree` is worth nothing if the guard
    was renamed, moved, or made conditional on the very selection the paragraph
    says is irrelevant. So the symbol has to be a real top-level def in
    `tests/conftest.py`, called unconditionally at module import (a bare
    top-level call, not one inside an `if` or a fixture), and its opt-in variable
    has to hold the string the paragraph quotes.
    """
    para = ruling_paragraph(DOC.read_text())
    assert GUARD_SYMBOL in para, "the paragraph dropped the guard's symbol"
    assert "LLOYD_ALLOW_LIVE_TREE_TESTS" in para, (
        "the paragraph dropped the opt-in's name, so a reader cannot tell what "
        "not to set"
    )
    assert "~/lloyd-work/" in para, "the paragraph dropped the route path"

    tree = ast.parse(CONFTEST.read_text())
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert GUARD_SYMBOL in defs, (
        f"{GUARD_SYMBOL} is no longer a top-level def in tests/conftest.py — the "
        "doc names a guard that is not there"
    )
    called = [n for n in tree.body
              if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Name)
              and n.value.func.id == GUARD_SYMBOL]
    assert called, (
        f"{GUARD_SYMBOL} is not called at module level any more, so the refusal is "
        "now conditional somewhere — the paragraph's first claim is false and the "
        "paragraph, not this node, is what has to change"
    )
    assigned = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id] = n.value.value
    assert assigned.get("LIVE_TREE_OPT_IN") == "LLOYD_ALLOW_LIVE_TREE_TESTS", (
        f"conftest's opt-in constant is {assigned.get('LIVE_TREE_OPT_IN')!r}, not "
        "what the doc tells a human to set"
    )


def test_the_ruling_guard_fails_on_the_pre_fix_doc_and_on_a_granted_carve_out():
    """Clause 3: both directions of non-emptiness, on the shipped text.

    The pre-fix half is the section exactly as `architecture/testing.md` shipped
    before #2361 — guard paragraph, worktree block, the human-only sentence, and
    no ruling. The granted half is that same paragraph with the ruling's answer
    flipped from "no such in-tree path exists, and none is planned" to the
    permission the ruling refuses, which is the edit a well-meaning later round
    would make if the guard only checked that a date was present.
    """
    body = _section(DOC.read_text())
    pre_fix_doc = (PROD_TREE_SECTION + body[:RULING.search(body).start()]
                   + "\n## Next section\n")
    assert "live_vault" not in body[:RULING.search(body).start()], (
        "the pre-fix slice now contains live_vault, so this control is checking a "
        "slice that already states the ruling"
    )
    try:
        ruling_paragraph(pre_fix_doc)
    except AssertionError as e:
        assert "#2361" in str(e), f"failed for the wrong reason: {e}"
    else:
        raise AssertionError("the guard accepted a section with no ruling paragraph")

    para = ruling_paragraph(DOC.read_text())
    granted = para.replace("no such in-tree path exists, and none is planned",
                           "a read-only in-tree path is permitted")
    assert granted != para, "the ruling's answer was rephrased; update this control"
    assert CARVE_OUT_GRANTED.search(granted), (
        "the carve-out pattern does not catch the flipped ruling, so the doc could "
        "grant an in-tree path and this node would still be green"
    )
    assert not CARVE_OUT_GRANTED.search(para), (
        "the shipped paragraph already reads as granting an in-tree carve-out"
    )
    assert CLAIMS["no in-tree path exists or is planned for live_vault"].search(para)
    assert not CLAIMS["no in-tree path exists or is planned for live_vault"].search(
        "A read-only in-tree path is permitted for live_vault runs.")
