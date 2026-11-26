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

import ast
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
    """Every `def test_…` or `async def test_…` under `tests/`, from the source, not the runner.

    Collected by regex rather than `pytest --collect-only` because the runner
    refuses to start in the production tree (`tests/conftest.py`'s
    `_refuse_the_production_tree`), and a citation check that only works inside a
    worktree is a citation check that is not run.

    The `async` arm is load-bearing and was missing until #3725 tripped it:
    `scripts/automod/backlog.py` cites
    `tests/test_dashboard_cold_render.py::test_one_cold_dashboard_cycle_beats_the_budget`,
    an `async def` that pytest collects (27 tests in that file), and this scan read it as
    dead. An extractor that sees only synchronous definitions makes an asynchronous witness
    look fabricated, so the prose gets rewritten away from the node that really covers it —
    the same wrong denominator this file's own denominator assertion worries about. Both
    this set and `_resolves`' per-file check had the blind spot.
    """
    names: set[str] = set()
    for p in sorted(TESTS.rglob("*.py")):
        names |= set(re.findall(r"^(?:async )?def (test_[A-Za-z0-9_]+)\(",
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
            return bool(re.search(rf"^(?:async )?def {re.escape(node)}\(", body, re.M))
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


# ── #2658: who writes `needs-human`, and where a debt actually goes ──────────
#
# The tag table said "No code adds it". One place does, and the two prose sites
# beside the owed mechanism said the opposite of what that mechanism is:
# `record_human_paths` and its caller in the landing route both told a reader that
# recording a path tags the item `needs-human` and holds it open, when the function
# writes a `path` entry on the item's `owed` list and does neither. A sentence that
# names the wrong destination is worse than no sentence: the reader stops looking,
# and here they would have stopped looking at the one tag the 2026-09-27 ruling left
# standing.


def _tag_table_row(tag: str) -> str:
    """One row of the tags table, by its tag cell, folded onto one line.

    A whole-section check would pass with the wording sitting in another row, and
    the row is the claim: a reader learns who writes a tag from the row for it.
    """
    rows = [ln for ln in DOC.read_text(encoding="utf-8").splitlines()
            if ln.startswith(f"| `{tag}` |")]
    assert len(rows) == 1, f"expected exactly one table row for {tag!r}, found {len(rows)}"
    return " ".join(rows[0].split())


def _grep_source(pathspec: str, pattern: str) -> subprocess.CompletedProcess:
    """`git grep` over one pathspec, as a reader holding this file could run it."""
    return subprocess.run(["git", "grep", "-n", "-e", pattern, "--", pathspec],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=120)


def test_the_needs_human_row_names_the_writer_that_remains():
    """The row states the ruling AND the one writer the loop keeps — clause 5.

    "No code adds it" was false: `workers/sources/autocode.py` tags an umbrella it
    closes `unnecessary`/`rejected` while its `members` are still folded, and
    `_apply_status` subtracts a tag only when the caller names it in `remove_tags`,
    so that tag survives its own close. A reader who believed the row would treat
    the writer the ruling deliberately kept as a second bug to remove.
    """
    row = _tag_table_row("needs-human")

    assert "Retired as a destination (2026-09-27)" in row, "the ruling itself was dropped"
    assert "No code adds it" not in row, "the row still denies that any code adds the tag"
    assert "workers/sources/autocode.py" in row, "the one live writer is not named"
    assert "members" in row, "the row does not say the condition that writer fires on"
    assert "owed" in row and "owed-check" in row, "the surviving destination is not named"
    assert "#2658" not in row, "the row still points a reader at the item that fixed it"


def test_the_human_paths_prose_no_longer_promises_a_tag_or_a_hold():
    """What `record_human_paths` says about its own debt, and what its caller says.

    The docstring is read off the function so a re-worded or deleted docstring
    fails rather than passing on stale bytes; the caller's half is a comment, so it
    is checked with the same `git grep` a reader can re-run, with a positive control
    on `record_human_paths`, the call that comment sits above.
    """
    from scripts.automod import backlog as B

    doc = " ".join((B.record_human_paths.__doc__ or "").split())
    assert doc, "record_human_paths has no docstring to check"
    assert "needs-human" not in doc, "the docstring still names the retired tag"
    assert "`path` entry" in doc and "`owed` list" in doc, (
        "the real destination of the debt is not stated")
    assert "never a tag" in doc, "the docstring does not rule out the retired destination"

    control = _grep_source("workers/sources/autocode.py", "record_human_paths")
    assert control.returncode == 0, (
        f"the positive control found nothing, so the instrument, not the tree, is the "
        f"problem: rc={control.returncode} cwd={ROOT} stderr={control.stderr[:200]!r}")
    gone = _grep_source("workers/sources/autocode.py", "needs-human")
    assert gone.returncode == 1, (
        f"the landing route still narrates the retired tag (rc={gone.returncode}): "
        f"{gone.stdout[:400]!r}")


# ── #3410: the corpus figures in `app/routers/backlog.py` must be dated ──────
#
# The module opens with "1,137 files / 11 MB … one YAML frontmatter pass over the
# corpus costs 1.78 s", and three comments deeper it repeats "parse up to 11 MB of
# YAML", "9 MB across 1,137 rows" and "1.78 s over 1,137 files that is ~1.6 ms a
# file" with no date at all. The corpus tripled: measured 2026-10-26 it is 3,356
# files / 71 MB and the pass costs 16.0 s, so every one of those reads as a live
# figure while being between 2.9x and 9x low — and a reader sizing a board
# endpoint, or the weekly board census that walks the whole corpus, calibrates
# against it. #1566's `owed_settled` evidence had already measured the same pass at
# 7.00 s on 2026-10-04 and nobody re-read the header.
#
# A dated figure cannot rot silently, because the next measurement is compared to
# its date rather than to the reader's expectation, so the rule pinned here is
# per-line and mechanical: a line that puts a backlog corpus size in front of a
# reader — a file/row count, or a total in MB — carries a measurement date on that
# same line, or states the parse-to-I/O ratio instead. Two absolute seconds whose
# gap silently widens is the shape that failed here, so the ratio is the form that
# is allowed to stand undated. A response's byte total (what `/tasks` shipped) is a
# different quantity from what is on disk, but it rots the same way and the rule
# catches it by the same `MB` match, so those lines are dated too.


ROUTE = ROOT / "app" / "routers" / "backlog.py"

# How this module states the size of the corpus it reads: a count of its files (or
# of the rows those files become), or a total in MB. Deliberately narrow: the
# failure being pinned is an *undated corpus size*, not any number in a comment,
# and a pattern wide enough to catch "two visible lines" would be a pattern some
# future reader has to widen with an exemption — which is how a date rule dies.
_SIZE_CLAIM = re.compile(
    r"\b\d[\d,]{2,}\s+(?:[\w-]+\s+)?(?:files|rows)\b"   # "1,137 files", "1,129 board files"
    r"|\b\d+(?:\.\d+)?\s*MB\b"                          # "11 MB", "71 MB"
)
_MEASUREMENT_DATE = re.compile(r"2026-\d{2}-\d{2}")
_RATIO_FORM = re.compile(r"\b\d+(?:\.\d+)?x\b|orders of magnitude")


def _undated_size_lines(src: str) -> list[str]:
    """Lines stating a corpus size with neither a date nor the ratio form."""
    return [ln.strip() for ln in src.splitlines()
            if _SIZE_CLAIM.search(ln)
            and not _MEASUREMENT_DATE.search(ln)
            and not _RATIO_FORM.search(ln)]


def test_no_corpus_figure_in_the_backlog_route_reads_as_todays_board_without_a_date():
    """Clause 1. Every corpus-size line in the route is dated or is the ratio.

    The three sites #3410 names — "~1.6 ms a file" beside `_FM_CACHE_LOCK`,
    "parse up to 11 MB of YAML" above the boards route, and "9 MB across 1,137
    rows" in the detail route's docstring — are the ones that mislead a reader
    sizing the route *today*, because an undated number is read as current state.
    """
    offenders = _undated_size_lines(ROUTE.read_text(encoding="utf-8"))
    assert not offenders, (
        f"{len(offenders)} corpus-size line(s) in app/routers/backlog.py state a "
        f"file count or an MB total with no date and no ratio on the same line, so "
        f"they read as the board the routes face now: {offenders[:4]}. The corpus "
        f"grew from 1,137 files to 3,356 in six weeks; date the figure or say "
        f"'parsing dominates I/O by roughly two orders of magnitude' instead.")


def test_the_backlog_route_size_detector_can_actually_fail():
    """A 0-hit scan is only evidence with a positive control beside it.

    Taken from the module's own dated lines so the control is the real string: the
    date is deleted from one line that carries a size claim, and the detector must
    then report it. Without this, a pattern that never matches anything — a typo in
    the alternation, a file that moved and read as empty — would report clause 1 as
    satisfied forever.
    """
    lines = ROUTE.read_text(encoding="utf-8").splitlines()
    dated_size_lines = [ln for ln in lines
                        if _SIZE_CLAIM.search(ln) and _MEASUREMENT_DATE.search(ln)]
    assert dated_size_lines, (
        "no line of app/routers/backlog.py carries both a corpus size and a date, "
        "so there is nothing here to control against — clause 1's own fix (one "
        "dated re-measurement) is missing, not merely un-detected")
    doctored = "\n".join(
        _MEASUREMENT_DATE.sub("", ln) if ln == dated_size_lines[0] else ln
        for ln in lines)
    assert _undated_size_lines(doctored), (
        f"the detector missed a size claim with its date stripped from "
        f"{dated_size_lines[0]!r}: the check cannot fail, so it certifies nothing")


def test_the_route_carries_one_dated_re_measurement_the_next_rot_is_measured_against():
    """Clause 2. One line, in the module docstring, that drift is measured against.

    It has to carry the date, the count and BOTH seconds together — a reader who
    learns "16 s" but not "3,356 files on 2026-10-26" has a number and no way to
    tell whether it is still true, which is the exact state this item was filed
    about. And it must quote no per-file cost as an absolute: mean file size
    doubled between the two measurements (9.7 KB to 21.2 KB), so 1.6 ms a file
    became 4.77 ms a file and restating either re-arms the same rot one corpus
    later.
    """
    src = ROUTE.read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or ""
    assert doc, "app/routers/backlog.py has no module docstring to carry the figure"
    calibrated = [ln.strip() for ln in doc.splitlines()
                  if all(tok in ln for tok in ("2026-10-26", "3,356", "71 MB",
                                               "16.0 s", "0.14 s"))]
    assert len(calibrated) == 1, (
        f"expected exactly one re-measurement line in the module docstring carrying "
        f"the date, the file count and both seconds; found {len(calibrated)}. "
        f"2026-10-26: 3,356 files / 71 MB, front-matter pass 16.0 s, raw read "
        f"0.14 s.")

    per_file = [ln.strip() for ln in src.splitlines()
                if re.search(r"ms\s*(?:a|per|/)\s*(?:file|row)", ln)]
    assert not per_file, (
        f"a per-file cost is back as an absolute: {per_file[:2]} — mean file size "
        f"doubled between 2026-09-16 and 2026-10-26, which is why 1.6 ms a file "
        f"became 4.77 ms a file; only the ratio held")


def test_the_recorded_acceptance_grep_over_the_route_returns_only_dated_lines():
    """The item's own check, run as the reader would run it.

    `-E` because the recorded pattern is an alternation; `git grep` because that is
    the form a reader holding this file can re-run, and the same instrument is
    pointed at a string that must still be found so a wrong pathspec cannot report
    the pass.
    """
    hits = subprocess.run(
        ["git", "grep", "-n", "-E", "-e", "1,137|11 MB|1137",
         "--", "app/routers/backlog.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert hits.returncode == 0, (
        f"the acceptance pattern found nothing at all (rc={hits.returncode}, "
        f"cwd={ROOT}), so either the module no longer names its corpus or the "
        f"instrument is broken: stderr={hits.stderr[:200]!r}")
    lines = [ln for ln in hits.stdout.splitlines() if ln.strip()]
    assert lines, "the pattern matched but returned no lines"
    undated = [ln for ln in lines if "2026-" not in ln]
    assert not undated, (
        f"{len(undated)} of {len(lines)} hits carry no measurement date, so they "
        f"read as the live corpus: {undated[:4]}")


# ── #4115: the `board-steward-disagreement` row and the producer it describes ───
#
# The tag-table row said `workers/board_disagreement.py` files one `draft` item per
# disagreement, on a ledger-provenance test, with a four-name kind list and an
# intake cap of five items a tick under a constant named in the same breath. None
# of it was ever written: `git log -S` over the filer's history for the cap's name
# and for the provenance label both come back empty, and the cap appears in zero
# tracked Python files — so the row was not stale against a refactor, it described
# a producer that never existed. The live filer triggers on a *measurement*
# (`below_floor`, owed-check's flip bar) and cannot accumulate at all: it merges
# into the one opener, which is why 18 filings carried this tag between 2026-10-27
# and 2026-11-13 and none of them was open.
#
# What the false sentence was worth is the reason for this section. The row ended
# "…does not bound them → filed.", and the architecture review of `doc:backlog`
# that refreshed the page on 2026-11-12 read it, re-derived an unbounded-expiry
# defect from constants that do not exist, and filed #4111 — closed `stale` 25
# minutes later. Closing the item without rewriting the row re-arms the loop: the
# next review of the same page reads the same row. `_tag_table_row` already folded
# this very row for the `needs-human` pass and never looked inside it, which is how
# four invented identifiers survived a suite whose whole subject is prose that
# blames code. So each claim the row now makes is checked against the filer whose
# behaviour it states, not against this file's memory of it.

FILER = ROOT / "workers" / "board_disagreement.py"

#: The identifiers the row blamed filing on. None is a filer constant or a kind
#: the code writes — `MAX_NEW_ITEMS` is in no Python file and the provenance label
#: lives only in unrelated test names — so a row carrying one points a reader at
#: code that never existed.
INVENTED_FILER_NAMES = ("no_ledger", "unapplied", "ambiguous", "unreadable",
                        "MAX_NEW_ITEMS", "5 a tick")

#: Every filer symbol the rewritten row names. Each has to resolve in the module it
#: is blamed in: the defect this section exists for is a doc naming code.
LIVE_FILER_SYMBOLS = ("TAG", "ROWS_MAX", "FLIP_MIN_JUDGED", "FLIP_MIN_RATE",
                      "below_floor", "reverted_moves", "file_or_update",
                      "openers", "_merge", "_create")


def _filer_src() -> str:
    src = FILER.read_text(encoding="utf-8")
    assert len(src) > 2000, f"the disagreement filer read back as {len(src)} chars"
    return src


def _filer_func(name: str) -> str:
    """One function's source from the filer.

    A row that says a gate lives in a named function is a claim about that
    function's body; checked against the whole file, the claim would pass with the
    gate moved anywhere else in the module.
    """
    src = _filer_src()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            assert seg, f"{name} parsed but has no source segment"
            return seg
    raise AssertionError(f"{name} is not defined in workers/board_disagreement.py")


def _filer_const(name: str):
    """A module-level literal constant of the filer, read from the AST."""
    for node in ast.parse(_filer_src()).body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level constant in the filer")


def _git_grep(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "grep", "-n", *args], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=120)


def test_the_board_steward_row_blames_no_name_the_filer_never_had():
    """Clause 1: the row carries none of the invented names, and names only code.

    Two directions, because a deletion would satisfy the first alone: the banned
    names are gone, and every filer symbol the row does carry is defined in that
    file. The old row would have passed a pure-absence check by being deleted.
    """
    row = _tag_table_row("board-steward-disagreement")
    offenders = [n for n in INVENTED_FILER_NAMES if n in row]
    assert not offenders, (
        f"the tag row still names {offenders} — `git log -S` over "
        f"workers/board_disagreement.py is empty for every one of them, so the row "
        f"is describing a producer that was never written, not a refactored one")

    # The item's own check, run as a reader runs it: `-E` because the recorded
    # pattern is an alternation, and aimed at a control string second so a wrong
    # pathspec cannot report the pass.
    gone = _git_grep("-E", "-e", "|".join(INVENTED_FILER_NAMES),
                     "--", "architecture/backlog.md")
    assert gone.returncode == 1, (
        f"the invented-name pattern still matches the page "
        f"(rc={gone.returncode}): {gone.stdout[:400]!r}")
    control = _git_grep("-e", "board_steward.py", "--", "architecture/backlog.md")
    assert control.returncode == 0, (
        f"the same instrument over the same page found its control string nowhere "
        f"(rc={control.returncode}), so the absence above proves nothing")

    src = _filer_src()
    missing = [s for s in LIVE_FILER_SYMBOLS
               if not re.search(rf"\b{re.escape(s)}\b", src)]
    assert not missing, (
        f"the row names {missing} as filer code and the module does not define "
        f"them — the exact defect #4115 removed")


def test_the_board_steward_row_states_the_measured_floor_that_gates_filing():
    """Clause 2: filing is gated by a measured window, and the row says whose bar.

    The two figures are owed-check's flip bar (#3089 asked, #3329 ruled
    2026-10-25) restated in the filer as `FLIP_MIN_JUDGED` / `FLIP_MIN_RATE`; the
    row's numbers are compared to those constants rather than to literals written
    twice, so a re-ruled floor fails here instead of leaving the page explaining a
    trigger that no longer fires.
    """
    row = _tag_table_row("board-steward-disagreement")
    assert "end of that same tick" in row, "the row no longer says when the filer runs"
    assert "rolling-7-day" in row, "the window the trigger reads is not named"
    assert "owed-check" in row, "the owner of the bar is not named"
    assert "below_floor" in row, "the row does not name where the gate is enforced"
    assert "judged >= 50" in row and "rate >= 0.90" in row, (
        "the flip floor is missing its figures: the accepted bar is judged >= 50 "
        "AND rate >= 0.90 over the 7-day window")

    m_judged = re.search(r"judged >= (\d+)", row)
    m_rate = re.search(r"rate >= (\d+\.?\d*)", row)
    assert m_judged and m_rate, "the floor figures are not in the row at all"
    assert int(m_judged.group(1)) == _filer_const("FLIP_MIN_JUDGED"), (
        f"the row says judged >= {m_judged.group(1)} but the filer's "
        f"FLIP_MIN_JUDGED is {_filer_const('FLIP_MIN_JUDGED')}")
    assert float(m_rate.group(1)) == _filer_const("FLIP_MIN_RATE"), (
        f"the row says rate >= {m_rate.group(1)} but the filer's FLIP_MIN_RATE is "
        f"{_filer_const('FLIP_MIN_RATE')}")

    body = _filer_func("file_or_update")
    assert "below_floor(" in body, (
        "the row blames `below_floor` for the trigger, but the filing entry point "
        "never calls it — the gate is somewhere the reader will not look")
    assert "return None" in body.split("below_floor(")[1][:200], (
        "`file_or_update` calls `below_floor` but does not stop filing on its "
        "verdict, so the row overstates the gate")


def test_the_board_steward_row_bounds_the_producer_at_one_open_item():
    """Clause 3: at most one open item, and `ROWS_MAX` is rows, not items.

    The bound is what makes the old "unbounded accumulation" reading wrong, so it
    is checked on both sides: the row states the merge design, and `file_or_update`
    still implements it — openers collected by tag, nothing filed beside an
    `in_progress` one, otherwise `_merge` into the first.
    """
    row = _tag_table_row("board-steward-disagreement")
    assert "one open item" in row, "the accumulation bound is gone from the row"
    assert "openers" in row and "in_progress" in row and "_merge" in row, (
        "the bound is asserted without the mechanism that produces it")
    assert "ROWS_MAX" in row, "the row dropped the real cap"
    assert "rows per item" in row and "not items per tick" in row, (
        "ROWS_MAX is not stated as rows per item — the quantity it bounds is the "
        "one thing a reader sizing the producer needs")
    assert "one `draft` item per disagreement" not in row, (
        "the row still files per disagreement, which is the claim that made #4111 "
        "look like unbounded growth")

    body = _filer_func("file_or_update")
    assert "openers" in body and "in_progress" in body, (
        "the filer no longer skips filing beside an item a round is working")
    assert "_merge(openers[0]" in body, (
        "the filer no longer merges into the existing opener, so more than one "
        "open item can exist and the row's bound is false")
    assert _filer_const("ROWS_MAX") == 40, (
        f"ROWS_MAX is {_filer_const('ROWS_MAX')}, not the 40 the row states")
    assert "ROWS_MAX" in _filer_func("disagreement_rows"), (
        "ROWS_MAX no longer caps the rows one filing writes, so the row names the "
        "wrong bound")


def test_the_board_steward_row_states_the_gauge_undercount_instead_of_an_item():
    """Clause 4: the residual is documentation, and the true tag sentence stays.

    The old row drew a false inference from a true fact: these items carry no
    `spawned-by-*` tag, and from that the page concluded the 30-day cap does not
    bound them and filed a defect. The true consequence is narrower — the tag is
    in neither `loop_spawn_tags()` nor `quarantine_tags()`, so the scorecard's open
    self-spawned gauge never counts them — and it is read here through the real
    readers, which is the only way the row's claim can be more than a sentence.
    """
    row = _tag_table_row("board-steward-disagreement")
    assert "→ filed" not in row, "the row still ends on the imperative that re-files it"
    assert "#4111" not in row, "the row points a reader at the item instead of the residual"
    assert "spawned-by-*" in row, "the one true tag sentence was dropped with the false inference"
    assert "loop_spawn_tags()" in row and "quarantine_tags()" in row, (
        "the residual is not stated as the two readers that miss the tag")
    assert "scorecard" in row, "the gauge the undercount lands on is not named"

    from scripts.automod import backlog as B
    tag = _filer_const("TAG")
    assert tag == "board-steward-disagreement", (
        f"the filer's tag moved to {tag!r}; the row and this test describe the old one")
    assert tag not in B.quarantine_tags(), (
        "the tag joined quarantine, so these items are held out of triage and the "
        "row's account of why triage closed them is stale")
    assert tag not in B.loop_spawn_tags(), (
        "the tag joined the expiry-and-gauge set, so the ≤ 1-item undercount the "
        "row documents no longer exists")
    assert B.loop_spawn_tag([tag, "swept"]) == "", (
        "the prefix rule now matches a disagreement item, which is the reading the "
        "row says is false")
    gauge = _git_grep("-e", "loop_spawn_tag(", "--", "scripts/automod/scorecard.py")
    assert gauge.returncode == 0, (
        f"the scorecard no longer asks `loop_spawn_tag` (rc={gauge.returncode}), so "
        f"the row names a gauge that is not there")

    arrow = _git_grep("-e", "→ filed", "--", "architecture/backlog.md")
    table_rows = [ln for ln in arrow.stdout.splitlines() if ln.split(":", 2)[-1]
                  .lstrip().startswith("| `")]
    assert not table_rows, (
        f"a tag-table row still tells the next review this was filed: {table_rows[:2]}")


# The dated refresh entry that filed #4111 off this page. The label, not the item:
# the entry is a record of what the 2026-11-12 review believed, and the thing a
# future review re-derives is the label "that expiry hole" pointing at an open
# defect. The fix is to date the refutation into it, not to delete the history.
_STALE_HOLE_ANCHOR = "#4111 (that expiry hole"


def _refresh_entries() -> list[str]:
    """Each `- ` list entry of the page's refresh log, folded onto one line.

    Folded because the entry wraps: an assertion against a single source line would
    fail on a re-flow that changed nothing a reader cares about.
    """
    lines = DOC.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("- "):
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("- "):
                j += 1
            out.append(" ".join(" ".join(lines[i:j]).split()))
            i = j
        else:
            i += 1
    return out


def test_the_refresh_narration_records_the_refutation_of_that_expiry_hole():
    """Clause 5: the 2026-11-12 narration is annotated, not erased.

    #4111 closed `stale` on 2026-11-13, 25 minutes after the review that minted it
    from this page, because the accumulation it named cannot happen. The entry still
    has to say what that review saw — it is dated history — and has to say beside it
    that the claim was refuted and which item survived as the actual defect.
    """
    hits = [e for e in _refresh_entries() if _STALE_HOLE_ANCHOR in e]
    assert len(hits) == 1, (
        f"expected exactly one refresh entry naming {_STALE_HOLE_ANCHOR!r}, found "
        f"{len(hits)} — the clause says the line is not deleted, and a second copy "
        f"would be the same claim standing in two places")
    entry = hits[0]
    assert "that expiry hole" in entry, "the dated claim was deleted rather than annotated"
    assert "refuted" in entry, "the refutation is not recorded beside the claim"
    assert "2026-11-13" in entry, "the refutation is undated, so it reads as opinion"
    assert "stale" in entry, (
        "the entry does not say how #4111 closed, so a reader cannot see the "
        "closure was a refutation and not a fix")
    assert "#4115" in entry, "the surviving doc defect is not named as the successor"
    assert "one open item" in entry, (
        "the annotation does not say why the accumulation cannot happen, which is "
        "the half a future review needs to stop re-deriving it")


def _route_table_row(route: str) -> str:
    """One row of the `## HTTP routes` table, by its route cell, folded onto one line."""
    rows = [ln for ln in _section("## HTTP routes").splitlines()
            if ln.startswith(f"| `{route}` |")]
    assert len(rows) == 1, f"expected exactly one table row for {route!r}, found {len(rows)}"
    return " ".join(rows[0].split())


def test_the_page_states_both_guardian_writes_and_how_the_close_is_booked():
    """The guardian writes the board twice — a filing and an all-clear close — and the
    close arrives over the board route, so it is booked as a Mission Control edit.

    Row-scoped for the table, section-scoped for the prose, and cross-checked against
    the code with `git grep`: renaming the guardian's constants or the route's reason
    turns this red rather than leaving the page describing a seam that moved. The new
    prose names code, never an item number — a closed item is not an owner.
    """
    update_row = _route_table_row("POST /api/backlog/task-update")
    assert "guardian" in update_row, f"the task-update row names no guardian write: {update_row}"
    assert "The UI's create — and the guardian's" in _route_table_row(
        "POST /api/backlog/task-create")

    routes = " ".join(_section("## HTTP routes").split())
    assert "notify.py::_backlog_task" in routes
    at = routes.index("The guardian's second write is the close.")
    close = routes[at:routes.index("that is an edit, never a close.)", at)]
    for needle in ("notify.py::_post_board_close", "`task-update`", '"status": "done"',
                   "BACKLOG_UPDATE_PATH", "BACKLOG_CLOSED_STATUS"):
        assert needle in close, f"the close paragraph lost {needle!r}: {close}"

    access = " ".join(_section("## Access rules").split())
    at = access.index("The board route's reason is a constant")
    booked = access[at:access.index("not to this page.", at)]
    assert "`set from Mission Control`" in booked, booked
    assert "guardian all-clear close" in booked and "Mission Control edit" in booked, booked
    # Beside, not instead of, the attribution rule the older node pins.
    assert "Every status move is attributed" in access

    for prose in (close, booked):
        assert not re.search(r"#\d+", prose), f"the new prose cites an item number: {prose}"

    for path, pattern in (
            ("agent-services/guardian/notify.py",
             'BACKLOG_UPDATE_PATH = "/api/backlog/task-update"'),
            ("agent-services/guardian/notify.py", 'BACKLOG_CLOSED_STATUS = "done"'),
            ("agent-services/guardian/notify.py", "def _post_board_close("),
            ("app/routers/backlog.py", '"set from Mission Control"')):
        hit = _grep_source(path, pattern)
        assert hit.returncode == 0 and hit.stdout.strip(), (
            f"{pattern!r} is gone from {path}; the page describes a seam that moved")
