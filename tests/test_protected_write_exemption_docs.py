"""#3563 — the write deny-set has one exemption, and the tree says so.

Three comments and two architecture rows asserted that a sanctioned in-process
writer depended on a `ContextVar` lift over the write deny-set. The lift had no
caller and never had one — all eight commits that ever touched it were read and
contain none — and the writer named in the prose does not exist either: the
paths the nightly reflection and the loaded-memory writers land are outside the
deny-set, and the one deny-set file that does get written by a sanctioned route
(`~/obsidian/lloyd/SOUL.md`) goes through `automod_vault_land`, which validates
the path against the prompt-surface contract and commits it with a `git`
subprocess without ever calling `write_deny_reason`.

So the mechanism is deleted, and this file is the rail that keeps the *story*
deleted with it. The prose is what makes a deleted guard dangerous again: a
future reader who finds "lifted only by the allow-listed contextvar" in a doc
row will not re-check whether the contextvar exists, which is precisely how an
escape hatch with no owner, no reason anyone reads back and no ledger row sat
beside a journalled refusal for weeks. The architecture doc-claims checker
cannot grade this — `tests/test_architecture_coverage_doc_claims.py` states that
a claim is graded only where the cited module has a dispatch chain, and
`allow_protected_writes` appears there as a *symbol* citation, not a command —
so the doc text is asserted here instead of riding ungraded.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Where the deleted mechanism was named. A sweep of the whole tree would be
#: noise (`tests/` keeps the name deliberately: it is the spoof key in
#: `test_no_tool_argument_lifts_the_denial`, a model can send any key it likes,
#: and the point of that node is that reading none of them is the property), and
#: these three are everywhere the mechanism was claimed to *work*.
SWEEP_ROOTS = ("app", "agent_mcp", "architecture")

#: Every name the lift went by. Not one of them may survive outside `tests/`,
#: because a name in a comment is a promise, and a promise about an exemption
#: nobody can audit is the bug — #3563 is about the promise as much as the code.
FORBIDDEN = (
    "allow_protected_writes",
    "grant_protected_writes",
    "release_protected_writes",
    "protected_writes_granted",
    "_write_grant",
)

_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".sh", ".conf", ".toml"}


def _swept() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in SWEEP_ROOTS:
        base = REPO / root
        assert base.is_dir(), f"{root} is gone, so the sweep below proves nothing"
        out.extend(p for p in sorted(base.rglob("*"))
                   if p.is_file() and p.suffix in _SUFFIXES
                   and "__pycache__" not in p.parts)
    assert len(out) > 200, (
        f"the sweep found {len(out)} files under {SWEEP_ROOTS}; a pattern "
        "searched over nothing is a clean result about nothing")
    return out


@pytest.mark.parametrize("name", FORBIDDEN)
def test_no_non_test_file_still_names_the_deleted_lift(name):
    """The acceptance check, as a test rather than as a grep someone runs once.

    The name is what matters, not the code: with the definitions gone the
    behaviour cannot regress by itself, but a comment that still promises a lift
    is a reader's invitation to add one back — and it is the exact prose that
    hid the fact that no writer needed it.
    """
    hits = []
    for path in _swept():
        text = path.read_text(encoding="utf-8", errors="replace")
        if name in text:
            line = next(i for i, ln in enumerate(text.splitlines(), 1) if name in ln)
            hits.append(f"{path.relative_to(REPO)}:{line}")
    assert not hits, (
        f"`{name}` was deleted with the lift, so nothing outside tests/ may "
        f"still assert a writer depends on it: {hits}")


def test_the_docs_name_the_exempt_route_and_its_mechanism():
    """"Exempt by design" is not an explanation, so the rows have to say which
    lane is exempt and by what mechanism: a `git` commit behind a validator.

    Both rows are pinned for the route *and* the mechanism word, because naming
    `automod_vault_land` alone would let a reader conclude the tool holds a
    permission inside the guard. It does not: it never calls the predicate, so
    deleting the lift could not break it.
    """
    rows = {
        "authority-surfaces.md": (REPO / "architecture" / "authority-surfaces.md").read_text(),
        "tools.md": (REPO / "architecture" / "tools.md").read_text(),
    }
    for doc, text in rows.items():
        assert "automod_vault_land" in text, f"{doc} does not name the exempt route"
        i = text.index("automod_vault_land")
        window = text[max(0, i - 700): i + 700]
        assert re.search(r"\bgit\b", window), f"{doc}: no commit mechanism near the route"
        assert re.search(r"validat", window), (
            f"{doc}: the exemption is not tied to validation, which is the only "
            "thing that distinguishes it from a bypass")


def test_authority_surfaces_no_longer_calls_the_exemption_a_design_choice():
    """Row 3's old words were "`automod_vault_land` is the one exempt lane, by
    design". "By design" reads as a decision that was made and can be revisited;
    the truth is that the exemption is a *mechanism* that does not pass through
    the guard at all — and the design sentence was standing where the actual
    design (no in-process lift, ever) belonged."""
    text = (REPO / "architecture" / "authority-surfaces.md").read_text()
    row = next(ln for ln in text.splitlines()
               if ln.startswith("| 3 |"))
    assert "by design" not in row, row
    assert "automod_vault_land" in row and "CONTRACT_PATHS" in row, row
    assert "Nothing inside this process lifts the write deny set" in row, (
        "the row must state the positive property, not merely drop the old claim")


def test_the_grants_row_denies_that_a_minted_grant_reaches_the_deny_set():
    """Row 8 closed with "the protected-write escape in layer 3 is the same idea
    scoped to one call" — the sentence that told a reader a human-minted grant
    and a deny-set exemption were one mechanism with two scopes. They are not,
    and `grant_create` has never had a scope that lifts a deny-set path: the row
    now says so, in those terms, next to the row that explains the route."""
    text = (REPO / "architecture" / "authority-surfaces.md").read_text()
    row = next(ln for ln in text.splitlines() if ln.startswith("| 8 |"))
    assert "no" in row.lower() and "deny-set" in row, row
    assert "automod_vault_land" in row, (
        "the row must point at the route rather than only deny the mechanism")
