"""Every claim `architecture/backlog.md` makes about who writes a status move.

The item that produced `app/backlog_move.py` was filed *because* that document
said "Every writer appends one line to the `activity_log`" and "Every status move
is attributed in the activity log and carries its reason" while the Mission
Control route appended nothing at all. The prose was not wrong when written; it
was wrong for the writer that had accumulated since. A claim about who writes is
exactly the kind that rots silently, and clause 5 of #1023 is a claim about the
prose, so the prose gets a test like the ones already kept for the automod,
dashboard, code-graph and session docs.

The assertions are written against the two sections rather than the whole file.
A `substring in document` check passes the moment the sentence is quoted
somewhere — say in a paragraph about how it used to be false — which is not the
claim clause 5 asks for. Each check therefore names the heading it must live
under, and fails if the section moved or lost the sentence.

The last section checks the *code* prose of `scripts/automod/backlog.py` instead
of this document, for the same reason and with the same failure behind it:
#2069 found that module telling a reader that an error path was "pinned" by a
test which no revision of this repository ever contained, so the sentence read as
a guarantee while the behaviour it promised had never run. Prose that points at a
test is a claim, and a claim with no check behind it rots silently — so now a
citation in that file's comments and docstrings is only allowed if it resolves.
"""

from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "architecture" / "backlog.md"
BACKLOG_PY = ROOT / "scripts" / "automod" / "backlog.py"
TESTS = ROOT / "tests"


def _section(heading: str) -> str:
    """One `## ` section's text, up to the next heading of any level."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index(heading)
    rest = text[start + len(heading):]
    end = len(rest)
    for marker in ("\n## ", "\n### "):
        found = rest.find(marker)
        if found != -1:
            end = min(end, found)
    section = rest[:end]
    # A section that sliced empty would make every `in` assertion below pass on
    # an empty string, which is the failure mode of a doc test, not a passing one.
    assert len(section) > 200, f"{heading!r} sliced to {len(section)} chars"
    return section


def test_the_activity_log_section_names_both_writers_and_the_one_recorder():
    """The section says a status move is recorded by a shared function, and says
    which two writers call it.

    "`_apply_status` writes `old → new: why`" was the sentence that made the loop
    sound like the only writer; the route is now named beside it.
    """
    section = _section("## The activity log")

    assert "app/backlog_move.py" in section, "the recorder is not named where it is claimed"
    assert "record_status_move" in section
    assert "_apply_status" in section
    assert "Mission Control" in section, "the section still names one writer"
    assert "task-update" in section, "the route must be named by its path, not by prose"
    assert "completed" in section, "the close stamp is the load-bearing half of the claim"


def test_the_activity_log_section_no_longer_crediting_the_loop_with_the_format():
    """The attribution sentence names the shared recorder, not `_apply_status`.

    Pinning the *absence* of the old wording is the only way to test "the prose no
    longer implies the loop's writer is the only one": the positive claim could be
    satisfied while the parenthetical still named one function as the writer.
    """
    section = _section("## The activity log")

    assert "`_apply_status` writes `old → new: why`" not in section
    assert "The single writer" not in section
    # And what replaced it must be a claim about both, not an omission.
    assert "Both" in section or "both writers" in section.lower()


def test_the_access_rules_record_a_mission_control_move_with_its_reason_and_stamp():
    """Access rules is where a reader looks to learn what a move costs.

    Clause 5 asks for the reason *and* the `completed` stamp to be stated there:
    the reason is what makes the audit trail auditable, and the stamp is what the
    board's 7-day done window reads first.
    """
    section = _section("## Access rules")

    assert "Every status move is attributed" in section, "the rule was reworded away"
    assert "Mission Control" in section
    assert "completed" in section
    assert "record_status_move" in section, "the rule must name where the rule lives"


def test_a_save_that_moves_nothing_is_documented_as_recording_nothing():
    """The no-move case is in the doc because it is in the code.

    `TaskModal` posts the status it already has on every save. A reader of the
    doc who did not learn that would expect an activity line per click, and a
    future writer who reads only the doc would not know the quiet case exists.
    """
    section = _section("## Access rules")

    assert "TaskModal" in section
    assert "no move" in section.lower()


def test_the_clock_section_grandfathers_the_legacy_rows_instead_of_asking():
    """The cut-over paragraph states the ruling, names its date, and stops routing
    the reader to a decision nobody owes (#2183 clause 4).

    It read "Pre-existing rows are grandfathered — nobody rewrote ~1,000 live board
    files; whether to backfill them is an open `needs-human` decision on #1517",
    which was false twice over by the day it was last read: #1517's owed entry has
    been `owed_settled` since 2026-10-04T12:45:36 with the ruling "Grandfather;
    never backfill", and `needs-human` was itself retired as a destination on
    2026-09-27 — as this same document says in its own status table. The absence
    half is load-bearing here, not decoration: the paragraph must say *why* the
    migration was refused, because a reason is what stops a future run attempting
    it, where a bare "we did not" only reports a preference. Both halves are
    checked against the section folded onto one line, since a claim about what the
    prose says must not depend on where it happens to wrap.
    """
    section = " ".join(_section("## One clock, and where the old one stopped").split())

    assert "grandfathered permanently" in section, "the ruling is not stated as final"
    assert "never backfilled" in section, "the ruling's own word is 'never backfill'"
    assert "2026-10-04" in section, "the ruling carries no date, so it reads as pending"
    assert "no per-row clock provenance" in section, (
        "the refusal reason is absent, and the paragraph is advice rather than a "
        "prohibition against rewriting ~1,000 live board files"
    )
    assert "completed:" in section and "1,027" in section, (
        "the population a uniform shift would double-move is not named; both figures "
        "are the owed-check ruling's own"
    )
    assert "needs-human" not in section, (
        "the paragraph still points at a status this document retired on 2026-09-27"
    )
    assert "whether to backfill" not in section, "the choice is still posed as open"
    assert not re.search(r"\bopen\b", section), (
        "something in the section still calls a settled ruling an open question"
    )


# ── prose that cites a test: the citation has to resolve (#2069) ─────────
#
# `_split_frontmatter`'s docstring said the error path was "pinned" by a test. It
# named one that had never existed — `git log --all -S` on that name returns only
# the commit that wrote the sentence — so the sentence certified a guarantee
# nobody was keeping while the `except yaml.YAMLError` it described had never run
# in a test. A citation is a pointer, and a pointer is worth exactly what happens
# when you follow it.

_BACKTICK = re.compile(r"`([^`]+)`")
_TESTISH = re.compile(r"(?:^|[/\s:.])test_")


def _cited_test_witnesses(path: Path) -> list[tuple[int, str]]:
    """`(line, citation)` for every back-quoted test reference in a file's prose.

    Comments and strings only, taken with `tokenize` rather than a line scan: a
    citation lives in prose, and the alternative is reading a `test_…` name in
    live code (a fixture name, a parametrize id) as a claim about a witness.
    Multi-line strings count, which is what makes a module docstring's citation
    checked too.
    """
    out: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        for span in _BACKTICK.findall(tok.string):
            span = " ".join(span.split())     # a citation wrapped over lines
            if _TESTISH.search(span):
                out.append((tok.start[0], span))
    return out


def _test_node_names() -> set[str]:
    """Every `def test_…` defined under `tests/`, from the source, not the runner.

    Collected by regex rather than `pytest --collect-only` because the runner
    refuses to start in the production tree (`tests/conftest.py`'s
    `_refuse_the_production_tree`), and a citation check that only works inside a
    worktree is a citation check that is not run.
    """
    names: set[str] = set()
    for p in sorted(TESTS.rglob("*.py")):
        names |= set(re.findall(r"^def (test_[A-Za-z0-9_]+)\(",
                                p.read_text(encoding="utf-8", errors="replace"), re.M))
    return names


def _resolves(citation: str, nodes: set[str]) -> bool:
    """Does a citation point at something real?

    Three shapes the prose uses: a node name on its own, a test module path, and
    the two joined by `::`. Each is checked against the tree it points at.
    """
    # Both halves get stripped: prose wraps, and `...py::\n    test_x` is the same
    # citation as `...py::test_x` once the token's whitespace is collapsed.
    head, sep, node = (p.strip() for p in citation.partition("::"))
    if "/" in head or head.endswith(".py"):
        target = ROOT / head.rstrip(":0123456789")
        if not target.is_file():
            return False
        if sep and _TESTISH.match(node):
            body = target.read_text(encoding="utf-8", errors="replace")
            return bool(re.search(rf"^def {re.escape(node)}\(", body, re.M))
        return True
    return head in nodes


def test_every_test_the_backlog_prose_cites_exists():
    """No sentence in `scripts/automod/backlog.py` may cite a witness that is not there.

    The denominator is asserted before the sweep, because a scan that finds
    nothing reports a clean file: 15 citations live in this module today, so an
    empty list means the extractor stopped seeing them, not that the prose got
    honest. This is the same "a check whose denominator can be zero is not a
    check" that #1287's own safety reading wrote down.
    """
    citations = _cited_test_witnesses(BACKLOG_PY)
    assert len(citations) >= 5, (
        f"{BACKLOG_PY.name} yielded {len(citations)} test citations; the prose "
        f"carries 15 (comments and docstrings both), so the extractor is reading "
        f"the wrong thing and a dead name would pass as resolved")
    nodes = _test_node_names()
    assert nodes, "no test node found under tests/, so nothing here could resolve"
    dead = [(line, cite) for line, cite in citations
            if not _resolves(cite, nodes)]
    assert not dead, (
        f"{len(dead)} of {len(citations)} test citations in {BACKLOG_PY.name} "
        f"point at nothing: {dead[:4]}. A docstring that says a behaviour is "
        f"'pinned by `test_x`' is a claim that the reader stops looking — which "
        f"is how #2069's unpinned error path read as covered for a fortnight.")


def test_split_frontmatter_s_error_path_citation_points_at_a_test_that_runs():
    """The one sentence #2069 was filed about, checked by name rather than in aggregate.

    It cites the witness for "one malformed item costs one item, never the walk",
    so the citation must be a node the suite actually contains. Asserting the
    positive (this function's docstring cites N nodes, all defined) rather than
    the negative of one retired name: the retired name is only the instance, and
    a check written around it would miss the next one.
    """
    from scripts.automod import backlog as B

    doc = B._split_frontmatter.__doc__ or ""
    cited = [c for c in _BACKTICK.findall(doc) if _TESTISH.search(c)]
    cited = [" ".join(c.split()) for c in cited]
    nodes = _test_node_names()
    node_cited = [c for c in cited if c in nodes]
    assert node_cited, (
        f"`_split_frontmatter`'s docstring cites no defined test node (cited: "
        f"{cited}). #2069 exists because this docstring's citation was a name no "
        f"revision of this repo ever had; 'no citation' is not the fix, a real "
        f"witness is.")
    dead = [c for c in cited if not _resolves(c, nodes)]
    assert not dead, f"dead citations in the docstring: {dead}"


def test_nothing_left_in_the_tree_names_the_front_matter_witness_that_never_was():
    """The aggregate check above's literal twin: the retired name is gone tree-wide.

    Written as a `git grep` because that is the check the item recorded, and a
    test that reproduces the accepted command is a test a reader can re-run by
    hand. The pattern carries a bracket class (`test_[m]alformed…`) so that this
    file is not itself the hit the search reports — the same reason a `ps | grep`
    spells its target with a class.

    A 0-hit grep is only evidence with a positive control beside it, so the same
    instrument is run on a name that must be found first. Without that, a broken
    cwd, a git that refused, or a pattern that matches nothing anywhere would all
    report this as a pass.
    """
    def search(pattern: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "grep", "-n", "-e", pattern, "--", "."],
                              cwd=str(ROOT), capture_output=True, text=True,
                              timeout=120)

    control = search("test_unterminated_front_matter_degrades_the_same")
    assert control.returncode == 0, (
        f"the positive control found nothing, so the instrument, not the tree, is "
        f"the problem: rc={control.returncode} "
        f"cwd={ROOT} stderr={control.stderr[:200]!r}")

    gone = search("test_[m]alformed_front_matter_behaves_the_same")
    assert gone.returncode == 1, (
        f"the retired witness name is still cited (rc={gone.returncode}): "
        f"{gone.stdout[:400]!r} {gone.stderr[:200]!r}")


def test_no_reader_label_in_the_loader_suite_carries_a_line_number():
    """`file.py:886` in a printed label is a citation with nothing re-reading it.

    All three reader labels in `tests/test_dashboard_yaml_loader.py` carried a
    line number at #2069's triage and all three were off (886 against a def at
    1490, 150 against 230, 118 against 112). The sibling test in that file checks
    the labels it prints; this one checks the source, so a re-added `:NNN` fails
    here even if the reader-side assert never sees it.
    """
    src = (TESTS / "test_dashboard_yaml_loader.py").read_text(encoding="utf-8")
    offenders = re.findall(r'"[\w.]+\([^"]*\.(?:py|tsx|ts):\d+\)"', src)
    assert not offenders, (
        f"reader labels carrying line numbers: {offenders[:3]} — a line number in "
        f"a string literal moves with the code and the string does not")


def test_every_backlog_module_the_prose_blames_actually_defines_that_symbol():
    """A doc naming `app/x.py::thing` where `thing` is not defined is the same
    defect in the other direction — and this one is cheap to check.
    """
    from app import backlog_move, backlog_tags

    assert callable(backlog_move.record_status_move)
    assert backlog_tags.NEEDS_HUMAN_TAG == "needs-human"
    # The loop's writer still re-exports the tag it moved, which is how
    # `cluster.py`'s `B.NEEDS_HUMAN_TAG` and every existing test still resolve it.
    from scripts.automod import backlog as loop_module
    assert loop_module.NEEDS_HUMAN_TAG == backlog_tags.NEEDS_HUMAN_TAG
