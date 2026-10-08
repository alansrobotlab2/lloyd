"""#1699 pins the four architecture docs that made the unreviewed areas review
units to the tree they inventory.

`workers/sources/arch_review.py::doc_slugs` globs top-level
`architecture/*.md`, so a doc is what turns an area into something the
cadence reads at all — an undocumented area cannot be reached by `groups:`,
because `parse_groups` only accepts a `<doc-slug>:<section>` pair inside a doc
that already exists. Four areas had no doc: the eval/bench surface, the context
window, write authority across surfaces, and the browser side panel.

A doc that inventories a set rots in one direction only: the set moves and the
prose does not. `tests/test_djev_doc_claims.py` and
`tests/test_desktop_doc_claims.py` are the precedent for pinning a doc to the
tree, and the same rule applies here — every assertion in this file extracts
something from the doc and compares it to the tree, so the failure it catches
is drift, not a reworded sentence. Three things each extraction has to prove
about itself, because a regex that matched nothing looks exactly like a doc
with nothing wrong:

  * the extracted set is asserted non-empty (an arms table that stopped
    parsing would otherwise pass on an empty intersection);
  * the check is shown to be able to fail, on text this file writes;
  * paths resolve against the working tree, not `git ls-files` — a doc may cite
    a git-ignored runtime file (`data/tool_overrides.yaml`) that is real on this
    box and absent from every worktree, and a tracked-files check reports it as
    missing. `$LLOYD_DATA/...` and `~/...` resolve under the data root and
    the home directory, because the scored store moved there on 2026-09-22
    (`architecture/data-home.md`) and a repo-rooted check would call it absent.

A *command* is a citation of the same kind, and it needed an extractor of its
own: `_CITATION` only fires on a span carrying a slash, so `run-fixtures-eval` —
the verb `architecture/measurement.md` once asserted for
`app/harness/supply_chain.py`, which never dispatched it — extracted as nothing,
and the check above stayed green while an operator following the doc got a usage
dump and exit 1 (#1792). The command half is at the bottom of this file and owes
the same three proofs.

**Two rulings about that command half, shipped here because they are the reason
it is shaped as it is (#1890).**

*It grades every top-level `architecture/*.md`, not the four docs of #1699.*
`NEW_DOCS` stays what it always was — the four docs whose *path* assertions,
index rows and "does not cover" sections this file pins — but a wrong *verb*
fails the same way wherever it is written: `workers/sources/arch_review.py`
makes every doc a review unit, and a command claim in one of the other thirty
sends an operator to the same usage dump and exit 1. Grading four of thirty-four
docs was prevention that protected 2 of the corpus's 15 command spans; the other
13 went ungraded, and the one report the corpus-wide sweep produced
(`harness.md:1573`) was in an ungraded doc. So the command sweep reads the glob.
The cost is that a newly written doc joins the surface the moment it lands, which
is the point: `test_the_command_sweep_covers_every_architecture_doc` names the
count so a glob that stopped finding the corpus cannot pass quietly.

*A module with no dispatch chain is still skipped, and that is not an exemption.*
Nothing beside a library or a flag-only script can be a subcommand, and reporting
a bare identifier there would read a symbol citation as a broken command —
`architecture/authority-surfaces.md:34` brackets `check_bash_command` beside
`app/harness/safety.py` for exactly that reason. The skip is keyed on the module
having no chain, never on which doc or which module is being read, so it is not a
list of names: `test_a_module_with_no_dispatch_chain_is_not_graded` pins it, and
`test_the_widened_sweep_is_shown_able_to_fail` pins the other side.
"""

from __future__ import annotations

import ast
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architecture"
EVAL = ROOT / "eval"
HOME = Path.home()

#: The four docs this item added. Named explicitly rather than glob-fresh:
#: a new doc elsewhere must not silently join the surface this file grades.
NEW_DOCS = ("measurement.md", "context-window.md", "authority-surfaces.md",
            "browser-side-panel.md")

#: How many top-level architecture docs exist. The command sweep reads the glob,
#: not this number; the number is the vacuity guard, because a `glob("*.md")`
#: that found no corpus — a moved directory, a wrong `ROOT` — would otherwise let
#: a sweep over "every doc" pass by sweeping nothing. A doc landing here moves
#: the number, and that is the sweep admitting its new subject, which is the
#: #1890 ruling: every doc is graded, so every doc has to be in the surface.
ARCH_DOC_SURFACE = 35   # + guard-coverage.md (#1948, 2026-10-01)

#: The heading every one of the four carries, in the words clause 3 asks for.
NOT_COVERED = "What this doc does not cover"

#: `eval/` directories that are artifacts, not arms. `__pycache__` is here
#: because a byte-compiled directory appears under the tree the moment anything
#: imports from `eval/`, and an inventory test that scored it would fail on the
#: reader's own machine rather than on the doc.
NON_ARMS = {"baselines", "measurements", "__pycache__"}


def _text(slug: str) -> str:
    return (ARCH / slug).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body of one `## <heading>` section, up to the next `##`."""
    m = re.search(rf"^## {re.escape(heading)}.*?$(.*?)(?=^## |\Z)",
                  text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading}' section"
    return m.group(1)


# --------------------------------------------------------------------------- #
# The arms inventory (measurement.md §The arms)
# --------------------------------------------------------------------------- #

#: A row of the arms table: first cell is the arm name in backticks.
_ARMS_ROW = re.compile(r"^\|\s*`([a-z0-9._-]+)`\s*\|", re.MULTILINE)


def eval_arms_on_disk() -> set[str]:
    """Same walk the doc claims to mirror: `eval/*` directories, artifacts out."""
    return {p.name for p in EVAL.iterdir()
            if p.is_dir() and p.name not in NON_ARMS}


def arms_named_by(text: str) -> set[str]:
    return set(_ARMS_ROW.findall(_section(text, "The arms")))


def test_the_arm_scan_finds_the_directories_it_claims_to_mirror():
    """Positive control on the scanner itself: `eval/` really does hold a dozen
    arm-shaped directories, so a set-equality below means parity and not that
    `NON_ARMS` swallowed the lot."""
    arms = eval_arms_on_disk()
    assert len(arms) >= 10, f"only {sorted(arms)} — the eval tree is not the tree this doc describes"
    assert not (arms & NON_ARMS)


def test_measurement_doc_names_exactly_the_eval_arms_that_exist():
    named = arms_named_by(_text("measurement.md"))
    assert named, "no arm rows parsed from measurement.md §The arms"
    on_disk = eval_arms_on_disk()
    assert named == on_disk, (
        f"measurement.md §The arms disagrees with eval/*: named-but-absent "
        f"{sorted(named - on_disk)}, on-disk-and-unlisted {sorted(on_disk - named)}")


def test_the_arm_check_sees_an_arm_that_went_unlisted(tmp_path):
    """Negative control: drop one row out of a two-row table and the checker
    must name that arm, so an empty diff above means the table matches."""
    text = "## The arms\n\n| Arm | Scores |\n|---|---|\n| `alpha_arm` | x |\n"
    named = arms_named_by(text)
    assert named == {"alpha_arm"}
    assert named - {"alpha_arm", "beta_arm"} == set()
    assert {"alpha_arm", "beta_arm"} - named == {"beta_arm"}


# --------------------------------------------------------------------------- #
# Cited paths and wikilinks
# --------------------------------------------------------------------------- #

_EXT = r"(?:py|md|ts|tsx|json|jsonl|yaml|yml|js|mjs|css|html|db|sql|txt|sh|service)"
#: A citation is a path inside a markdown code span, matched against the WHOLE
#: span, so prose is never mistaken for an inventory entry. Paths outside
#: backticks are deliberately not extracted: a doc's running text contains
#: `keep/raise/revert` and `met/not_met`, and a directory-shaped match against
#: those would report invented missing files — a check that fails on nothing
#: real is a check a later reader disables. Inside a span, a citation is
#: `[$LLOYD_DATA/ | ~/] a/b/c.ext | a/b/c/ [:line]`. A bare `stats.py` is not a
#: citation: it names a file without claiming where it lives. Neither is a glob
#: (`web/src/**`) or a URL — both are rules and routes, not paths on disk.
_CODE_SPAN = re.compile(r"`([^`\n]+)`")

_CITATION = re.compile(
    r"\A(?P<pre>\$LLOYD_DATA/|~/)?"
    r"(?P<path>(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+\.(?:" + _EXT + r")"   # a file
    r"|(?:[A-Za-z0-9_.\-]+/)+)"                                             # or a directory
    r"(?::(?P<line>\d+))?\Z")


def _resolve(prefix: str | None, rel: str) -> Path:
    if prefix == "$LLOYD_DATA/":
        return HOME / "lloyd-data" / rel
    if prefix == "~/":
        return HOME / rel
    return ROOT / rel


def cited_paths(text: str) -> list[tuple[Path, int | None]]:
    """Every code-span citation in `text`, resolved. Repo-relative against the
    working tree — deliberately NOT `git ls-files`, because a citation has to
    resolve where the reader is standing and some real ones are untracked;
    `$LLOYD_DATA/...` against `~/lloyd-data`, the root the mover moved them to;
    `~/...` against home."""
    out = []
    for span in _CODE_SPAN.findall(text):
        m = _CITATION.match(span.strip())
        if not m or "/" not in m.group("path"):
            continue
        out.append((_resolve(m.group("pre"), m.group("path")),
                    int(m.group("line")) if m.group("line") else None))
    return out


def git_ignored(paths) -> set[str]:
    """Which of `paths` git reports as ignored, by pattern (empty set on error).

    Why the resolution check honours a gitignore hit: a worktree contains only
    what git carries, so a citation to a path `.gitignore` matches can never
    resolve there — and a doc that says out loud that the file is untracked and
    exists only on this box is describing the machine correctly. Before this, the
    only tree the check could pass in was the live checkout, so the node failed
    every round that touched this file from a worktree, and the failure read like
    the round's own.
    The trade, priced: a citation to a *fictional* path that happens to fall under
    an ignore pattern is no longer reported. It is reported by nothing else either
    (git has no record of such a file in any tree), so the exemption gives up a
    case the check could not win anyway, and the fiction case stays covered by
    `test_the_path_extractor_is_what_the_check_depends_on`.
    """
    rels = []
    for p in paths:
        try:
            rels.append(str(Path(p).relative_to(ROOT)))
        except ValueError:
            continue
    if not rels:
        return set()
    proc = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--stdin"],
                          input="\n".join(rels), capture_output=True, text=True)
    # rc 0 = at least one path ignored, rc 1 = none. Both are clean answers.
    if proc.returncode not in (0, 1):
        return set()
    return {ROOT / line for line in proc.stdout.split("\n") if line.strip()}


def unresolved_citations(cited) -> list:
    """Cited paths that do not exist and are not git-ignored."""
    ignored = git_ignored([p for p, _ in cited if not Path(p).exists()])
    return [p for p, _ in cited if not Path(p).exists() and Path(p) not in ignored]


@pytest.mark.parametrize("slug", NEW_DOCS)
def test_every_path_a_new_doc_cites_resolves_in_the_working_tree(slug):
    cited = cited_paths(_text(slug))
    assert cited, f"{slug} cites no path at all — the doc has stopped being checkable"
    missing = [str(p) for p in unresolved_citations(cited)]
    too_short = [f"{p}:{n}" for p, n in cited
                 if p.exists() and n is not None
                 and len(p.read_text(encoding="utf-8", errors="replace").splitlines()) < n]
    assert not missing, f"{slug} cites paths that are not on disk: {missing}"
    assert not too_short, f"{slug} cites line numbers past the end of the file: {too_short}"


def test_the_path_extractor_is_what_the_check_depends_on():
    """Negative control over deliberately wrong text: one real path, one
    invented one, one line number past the end. If the extractor could not tell
    these apart, `test_every_path_a_new_doc_cites_resolves_in_the_working_tree`
    would pass on a doc full of fiction."""
    real = "app/paths.py"
    text = ("see `app/paths/no_such_module_really.py`, then `$LLOYD_DATA/eval/nope/`, "
            f"then `{real}:100` and `{real}:999999`, written over "
            "https://example.com/app/paths.py, with keep/raise/revert decisions and "
            "a bare `stats.py` and a rule over `web/src/**`")
    cited = cited_paths(text)
    paths = [p for p, _ in cited]
    assert ROOT / real in paths, "the extractor missed a path that is plainly there"
    assert HOME / "lloyd-data/eval/nope" in paths, (
        "a $LLOYD_DATA citation was not extracted, so the data-root half of the "
        "resolution check has nothing to resolve")
    missing = {str(p) for p in paths if not p.exists()}
    assert any("no_such_module_really" in m for m in missing), (
        "an invented path slipped through the extractor, so the resolution check "
        "above cannot fail on a doc full of fiction")
    assert any(str(p).endswith("eval/nope") for p in paths if not p.exists()), (
        "an invented $LLOYD_DATA directory resolved, or was never extracted")
    joined = " ".join(missing)
    assert "example.com" not in joined, (
        "a URL was extracted as a repo path, which would fail a doc for citing "
        "somebody else's file")
    assert "keep" not in joined and "met" not in joined, (
        "prose was read as a path: `keep/raise/revert` is a decision, not a "
        "directory, and a doc that fails on it is failing on nothing")
    assert not any(p == ROOT / "web/src" for p in paths), (
        "a glob was extracted as a path, so a rule over a tree reads as an "
        "inventory entry")
    anchored = sorted(n for p, n in cited if p == ROOT / real and n)
    assert anchored == [100, 999999], "line anchors were not parsed beside their paths"
    assert len((ROOT / real).read_text(encoding="utf-8").splitlines()) < 999999


def test_each_new_doc_is_linked_from_the_index_as_its_own_row():
    """Clause 2's positive half, pinned in the file this diff adds.

    `tests/test_architecture_index_parity.py` owns set equality over the whole
    corpus in both directions, and its link-names-a-doc node is red at this
    round's base for an unrelated reason: `4a6cdd54` removed
    `architecture/recall-research-2026-09-24.md` from git and left its
    `index.md` row standing, which item #1704 owns. This node asserts only what
    THIS diff contributes — each new slug is a cell in an index table row, so a
    reader reaches a new area by following the index rather than by knowing the
    filename. A table row and not any old mention: `index.md`'s own preamble
    tells a reader to go to a table rather than add a list item.
    """
    index = (ARCH / "index.md").read_text(encoding="utf-8")
    cells = {c.strip("[]() ")
             for line in index.splitlines() if line.startswith("|")
             for c in line.split("|") if c.strip()}
    for slug in (s[:-3] for s in NEW_DOCS):
        assert slug in cells, (
            f"[[{slug}]] is not a cell in an index.md table row — the doc is on "
            "disk but nothing routes to it, so it stays unreviewed in practice")


_WIKILINK = re.compile(r"\[\[([a-z0-9-]+)(?:\|[^\]]*)?\]\]")


@pytest.mark.parametrize("slug", NEW_DOCS)
def test_every_wikilink_in_a_new_doc_names_a_top_level_architecture_doc(slug):
    links = _WIKILINK.findall(_text(slug))
    assert links, f"{slug} wikilinks nothing, so it is unreachable from its neighbours"
    dangling = [s for s in set(links) if not (ARCH / f"{s}.md").exists()]
    assert not dangling, f"{slug} links [[slug]]s with no architecture/<slug>.md: {dangling}"


# --------------------------------------------------------------------------- #
# The four docs, as review units
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("slug", NEW_DOCS)
def test_each_new_doc_says_what_it_does_not_cover(slug):
    """Clause 3: an inventory is not a verdict. `architecture/arch-review.md`
    opens with the failure mode of a doc that reads like an all-clear, and a
    section that exists but is empty is the same failure wearing a heading."""
    raw = _section(_text(slug), NOT_COVERED)
    body = " ".join(raw.split())
    bullets = [ln for ln in raw.splitlines() if ln.startswith("- ")]
    assert len(bullets) >= 4, (
        f"{slug} §{NOT_COVERED} names {len(bullets)} boundary as bullets. The "
        "boundary of an area is a list of what the NEXT reader has to go "
        "elsewhere for — all four docs carry at least 4, so a section trimmed to "
        "one line is boilerplate wearing the heading, not a boundary")
    assert len(body) >= 400, (
        f"{slug} §{NOT_COVERED} has {len(body)} chars of prose, under the 400 "
        "floor — a heading with a stub under it reads as an all-clear, which is "
        "the failure this section exists to prevent")


@pytest.mark.parametrize("slug", NEW_DOCS)
def test_each_new_doc_is_a_review_unit_the_moment_it_lands(slug):
    """The whole reason for the four docs: `doc_slugs` globs the directory, so
    landing one is what makes the cadence able to read it. Pinned here rather
    than watched later because the glob is the mechanism, and a rename or a
    move into a subdirectory would quietly un-make the unit."""
    from workers.sources.arch_review import doc_slugs
    assert (ARCH / slug).exists()
    assert slug.removesuffix(".md") in set(doc_slugs(ROOT))


# ── #1763: the citation that left main red, and the green it could buy by erasure ──


def test_the_identity_file_is_still_cited_and_lands_outside_the_repo():
    """Row 1 of `authority-surfaces.md` is the vault identity file, and the only
    spelling of it this resolver can resolve is the one rooted at the vault.

    `_resolve` has three branches (line 138): `$LLOYD_DATA/` to the data root, `~/`
    to the home directory, everything else to the source tree. The file header says
    why the second exists — a repo-rooted check calls the moved store absent — and
    three architecture docs spell vault paths with that prefix (`measurement.md`,
    `vault-protection.md`, `workers-jobs.md` all cite `~/obsidian/lloyd/bench`).
    Row 1 spelled the identity file `lloyd/SOUL.md`: no prefix, contains a slash, so
    `_resolve` took the default branch and asked the source tree for
    `<repo>/lloyd/SOUL.md`, which exists nowhere — `git ls-files lloyd/` is empty,
    and the deny-set entry the row is describing is spelled
    `~/obsidian/lloyd/SOUL.md` in the code (`app/harness/protected_paths.py:154`).
    That is main's red node, and the fix is the row's root, not a checker that stops
    looking.

    Asserted in three parts, because deleting the citation is the other way to get
    a green here and a doc that cites nothing is an all-clear:

    - the row still names the file;
    - `cited_paths` — the same extraction the graded check runs — lands a path under
      the home directory for it, and none under the source tree;
    - the gate's prompt-surface trigger really does key on the bare name, which is
      the half of the row that needs no root and so is not a claim about paths.
    """
    text = _text("authority-surfaces.md")
    assert "SOUL.md" in text, (
        "row 1 no longer cites the identity file at all: the checker is green "
        "because it has nothing left to check")

    cited = cited_paths(text)
    rooted = HOME / "obsidian" / "lloyd" / "SOUL.md"
    hits = [p for p, _line in cited if p.name == "SOUL.md"]
    assert hits == [rooted], (
        f"the identity-file citation resolves to {[str(h) for h in hits]}, not to "
        f"{rooted} — a repo-relative spelling is what left main red")
    assert rooted.exists(), (
        f"{rooted} is not on disk, so the graded check would be red again for a "
        "reason this node's own assertion cannot tell apart from the old one")

    # The control that says the checker still rejects an unrooted vault path, so
    # this node cannot be satisfied by a resolver that stopped caring.
    unrooted = cited_paths("see `lloyd/SOUL.md` for the classes")
    assert unrooted == [(ROOT / "lloyd" / "SOUL.md", None)], unrooted
    assert not unrooted[0][0].exists(), (
        "the repo-relative spelling resolved, so the red node this item files "
        "would no longer be red — the positive control has stopped controlling")

    from scripts.automod.gate import Gate
    assert "SOUL.md" in Gate.PROMPT_SURFACE_VAULT, Gate.PROMPT_SURFACE_VAULT


# ── #1763 clause 2: green by fixing, not by hiding ─────────────────────────────
#
# The named node failed at base `eb889136`, and a red node stops being red three
# ways without anything being fixed: mark it out of the gate's mark expression,
# mark it `skip`/`skipif`/`xfail`, or delete it. This file's graded check is the one
# an author can defuse from inside the file, so clause 2 is pinned over the seam the
# gate actually crosses — `scripts/automod/gate.py` builds an argv and a child
# interpreter runs it (`TESTS_MARK_EXPR` at `scripts/automod/gate.py:326`) — rather
# than by reading decorators off the source.

THIS_FILE = Path(__file__).resolve().relative_to(ROOT).as_posix()

#: The node #1763 files, spelled with its parametrisation. Named exactly, because
#: "the check is green" is equally true of a check that is no longer collected.
NAMED_NODE = ("test_every_path_a_new_doc_cites_resolves_in_the_working_tree"
              "[authority-surfaces.md]")

#: Anchored to a decorator at the start of a line, or to a call of the runtime
#: form. A substring scan over the source is not a check: it matches this file's
#: own prose about the markers it forbids, which is exactly what an author who
#: wanted to hide a node would write.
_HIDE_RE = re.compile(
    r"^\s*@pytest\.mark\.(?:live_vault|fault_injection|skip|skipif|xfail)\b"
    r"|\bpytest\.(?:skip|skipif|xfail)\s*\("
    r"|\bpytest\.param\s*\("
    r"|\bskip(?:if)?\s*=\s*[\"']",
    re.M)


def _pytest(extra: list[str]) -> subprocess.CompletedProcess:
    """Run pytest in a child interpreter, the way the gate's rung does."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_the_gate_selection_drops_nothing_and_the_named_node_passes():
    """Clause 2, across the process boundary: pytest's own selection and pytest's
    own per-node verdicts, not a reading of the source.

    One `-v` run under the gate's mark expression answers every half:

    - `--collect-only` with and without `-m "not live_vault and not
      fault_injection"` collect the same nodes, so nothing in this file was moved
      out of the gate's reach;
    - the run reports one **PASSED line** per node collected less this one, so a
      skipped or xfailed node cannot hide inside a passing exit code;
    - the node #1763 files — `test_every_path_a_new_doc_cites_resolves_in_the_working_tree`
      `[authority-surfaces.md]` — is among them and is reported PASSED **by
      name**, which is the verdict this item was filed for;
    - exit code 0, and no `skipped`/`xfailed` anywhere in the output.

    This node is the one exclusion, and it is a recursion guard, not a dodge: a node
    that runs its own file runs itself again — measured at 298 live pytest processes
    before it was bounded. The arithmetic carries that honestly, and every count
    below is derived from collection rather than quoted from a triage — the numbers
    this paragraph used to carry (23 collected, 22 reported) had been stale since
    #1792 added to the file, and a stale count here is a count nobody can use to
    notice a node going missing. This node's own verdict is what the gate's
    full-suite run records with no exclusion in it.
    """
    from scripts.automod.gate import TESTS_MARK_EXPR

    unfiltered = [ln for ln in (_pytest(["--collect-only", "-q", THIS_FILE]).stdout
                                .splitlines()) if "::" in ln]
    gated = [ln for ln in (_pytest(["--collect-only", "-q", "-m", TESTS_MARK_EXPR,
                                    THIS_FILE]).stdout.splitlines()) if "::" in ln]
    assert gated == unfiltered, (
        f"the gate's selection ({TESTS_MARK_EXPR!r}) deselected "
        f"{len(unfiltered) - len(gated)} of this file's nodes: "
        f"{sorted(set(unfiltered) - set(gated))}")
    assert unfiltered, "collection found no nodes at all — the check is empty"
    assert any(NAMED_NODE in node for node in gated), (
        f"{NAMED_NODE} is no longer collected: the red node was removed rather "
        "than fixed")

    src = (ROOT / THIS_FILE).read_text(encoding="utf-8")
    hidden = _HIDE_RE.findall(src)
    assert hidden == [], (
        f"this file carries a marker that takes a node off the hard rung: {hidden}")

    this_node = (f"{THIS_FILE}::"
                 "test_the_gate_selection_drops_nothing_and_the_named_node_passes")
    ran = _pytest(["-v", "-m", TESTS_MARK_EXPR, "--deselect", this_node, THIS_FILE])
    out = ran.stdout + ran.stderr
    assert ran.returncode == 0, out[-1500:]
    low = out.lower()
    assert "skipped" not in low and "xfail" not in low, out[-1500:]
    assert f"{len(gated) - 1} passed" in out, out[-400:]

    passed = [ln for ln in out.splitlines() if " PASSED" in ln]
    assert len(passed) == len(gated) - 1, (
        f"{len(gated)} nodes collected, {len(gated) - 1} run, only {len(passed)} "
        f"report PASSED — a verdict is missing, not passing: "
        + "\n".join(ln for ln in out.splitlines()
                    if "::" in ln and " PASSED" not in ln))
    named = [ln for ln in passed if NAMED_NODE in ln]
    assert len(named) == 1, (
        f"{NAMED_NODE} is not reported PASSED by name, so the node this item was "
        f"filed for has no recorded verdict: {named}")


# ── #1760 clause 4: the deny-set rows cite symbols, not moved line numbers ────
#
# Row 3 of `authority-surfaces.md` pinned `PROTECTED_WRITE_ROOTS` to
# `app/harness/protected_paths.py:148` and `write_deny_reason` to
# `app/harness/protected_paths.py:195`, and the "Protected." bullet pinned the
# vault-root agreement to `app/paths.py:109`. All three of those numbers were
# already wrong or were going to be: the two in row 3 sat a constant-definition
# away from their real lines, and the `app/paths.py` one had been copied there
# from a comment in the very module it describes, which is how a rotting pointer
# reaches the doc layer — `app/paths.py:11` was the number the code's own comment
# carried, and line 11 is prose about `HOME=<round>/home`. A symbol citation cannot
# drift that quietly: `app.paths.VAULT_ROOT` is either in the module or it is not.
#
# Scope is deliberate and is asserted, not assumed. The doc still cites line
# numbers elsewhere (`app/harness/safety.py:372`, `agent_mcp/builtin_fs.py:238`,
# `agent_mcp/vault.py:1608`), and this guard must leave them alone — a doc-wide
# ban would be a different rule than the one clause 4 states, and would be
# satisfied by deleting citations rather than replacing them.

#: Matches `app/paths.py:<n>` and `<…>protected_paths.py:<n>` alike, which is
#: exactly the set clause 4 names: the vault-root pointer, and the two pointers
#: that stand in for `PROTECTED_WRITE_ROOTS` and `write_deny_reason`.
PATHS_LINE_CITATION = re.compile(r"paths\.py:\d+")

#: Any line citation at all, for the control that the doc still cites lines.
ANY_LINE_CITATION = re.compile(r"[\w/.-]+\.py:\d+")

#: The three citations as they stood at base `a820d20b`, kept as the fixture the
#: reader has to catch.
STALE_ROW3_CITATIONS = (
    "`PROTECTED_WRITE_ROOTS` at `app/harness/protected_paths.py:148`, "
    "`write_deny_reason` at `app/harness/protected_paths.py:195`")
STALE_VAULT_CITATION = "`app/paths.py:109` is where `VAULT_ROOT` agrees with it."


def _row3_and_protected_bullets(text: str) -> str:
    """Row 3 of the ladder table plus the "Protected." bullet — the two places
    clause 4 names, extracted so the assertion is about them and not about an
    incidental line elsewhere in the file."""
    row = re.search(r"^\| 3 \|.*$", text, re.MULTILINE)
    bullet = re.search(r"^- \*\*\"Protected\.\"\*\*.*?(?=^- \*\*|\n## )",
                       text, re.MULTILINE | re.DOTALL)
    assert row and bullet, (
        "row 3 or the \"Protected.\" bullet is missing or renamed, so the section "
        "clause 4 governs is no longer there to be checked")
    return row.group(0) + "\n" + bullet.group(0)


def test_authority_surfaces_cites_the_denied_paths_by_symbol_not_line():
    """Clause 4: no `paths.py:<line>` citation survives in the doc, and the two
    sections that carried the three still name all three symbols.

    The second half is what stops "no citations" being won by deletion: row 3 and
    the bullet have to keep saying `PROTECTED_WRITE_ROOTS`, `write_deny_reason` and
    `app.paths.VAULT_ROOT`, so the only green is one that still tells the reader
    which module decides. The reader is then shown to catch each pre-fix spelling,
    and to leave the doc's other line citations alone — the scope of the clause,
    not an accident of the pattern.
    """
    text = _text("authority-surfaces.md")
    assert PATHS_LINE_CITATION.findall(text) == [], PATHS_LINE_CITATION.findall(text)

    scope = _row3_and_protected_bullets(text)
    for symbol in ("PROTECTED_WRITE_ROOTS", "write_deny_reason", "app.paths.VAULT_ROOT"):
        assert symbol in scope, (
            f"{symbol} is no longer named where clause 4 says it must be cited by "
            "symbol — a doc that cites nothing is an all-clear, not a fix")

    stale = STALE_ROW3_CITATIONS + " … " + STALE_VAULT_CITATION
    assert PATHS_LINE_CITATION.findall(stale) == ["paths.py:148", "paths.py:195",
                                                  "paths.py:109"], (
        "the reader does not see the citations this round removed, so the assertion "
        "above is an empty pattern over an empty corpus")
    others = [c for c in ANY_LINE_CITATION.findall(text)
              if not c.endswith("paths.py:" + c.split(":")[-1])]
    assert len(others) >= 3, (
        f"the doc cites no other line numbers ({others}), so this guard could not "
        "tell a scoped rule from a doc-wide ban")
    assert any("safety.py:" in c for c in others) and any("vault.py:" in c for c in others), (
        f"the untouched citations this round left alone are missing: {others}")


#: The fixture the exemption exists for: a path `.gitignore` keeps out of every
#: tree, which docs cite as real on this box. It was the extension's
#: `manifest.json` until #1701 tracked it; `data/tool_overrides.yaml` is the same
#: shape (ignored by its own rule, cited by `architecture/data-home.md` and
#: others, present only in the live checkout).
IGNORED_CITATION = ROOT / "data/tool_overrides.yaml"
FICTION_CITATION = ROOT / "app/paths/no_such_module_really.py"
OUTSIDE_CITATION = HOME / "lloyd-data/eval/nope"


def test_an_ignored_citation_is_exempt_and_a_fictional_one_is_not():
    """The exemption control, in both directions, because an always-empty
    `unresolved_citations` and a correctly-empty one look identical in the report.

    Three citations, three verdicts: the git-ignored runtime file is exempt, a
    fabricated module under no ignore pattern is reported, and a path outside the
    repo (`$LLOYD_DATA`-style, which is what the extractor resolves against home)
    is reported rather than silently skipped — the exemption is about git's
    coverage of *this* tree, not about paths it cannot see. Then the two
    preconditions of the exemption are themselves asserted, so the node cannot
    rot into a no-op: if `.gitignore` stops matching that file, or starts
    matching the fabricated path, the exemption is no longer what is making the
    main node green and this node says so.
    """
    cited = [(IGNORED_CITATION, None), (FICTION_CITATION, None), (OUTSIDE_CITATION, None)]
    unresolved = [Path(p) for p in unresolved_citations(cited)]
    assert IGNORED_CITATION not in unresolved, (
        "the git-ignored citation was still reported, so a doc that tells "
        "the truth about an untracked file cannot pass from a worktree")
    assert FICTION_CITATION in unresolved, "a fabricated repo path slipped through"
    assert OUTSIDE_CITATION in unresolved, (
        "a path outside the repo was exempted; the exemption is for git's blind "
        "spots inside the tree, not for paths git never addresses")
    assert IGNORED_CITATION in git_ignored([IGNORED_CITATION]), (
        f"{IGNORED_CITATION} is no longer git-ignored, so the exemption is dead code "
        "and the main node is passing for the wrong reason")
    assert FICTION_CITATION not in git_ignored([FICTION_CITATION]), (
        "the fabricated path is git-ignored, which makes it a useless negative control")


# ── #1787: the compaction-recall runner, the presets its flag takes, and the
# ── second sense of "arm" they live in ──────────────────────────────────────
#
# `architecture/context-window.md` delegated two "ships off until compared"
# `compaction` flags to `architecture/measurement.md` as the doc that owned the
# comparison, and measurement.md named none of them: at base `d00f56d0` greps for
# `summary_persisted`, `memory_flush`, `persist_summary` and `summary_legacy` over
# it returned 0 hits, and the only "compact" line in the file was the tracked
# baseline-filename list under §Two roots — a pin, not an arm. What made the
# pointer worse than empty was that measurement.md's own §The arms defines an arm
# as "a directory under `eval/`" and holds that table set-equal to `eval/*`, while
# `summary_persisted` and `memory_flush` are `--arms` values of one runner. So the
# row the item first suggested is the one shape that cannot work: the set-equality
# node above would grade it as named-but-absent. The content had to be its own
# `## ` section, and these nodes are what stop that section from being a paragraph
# nobody can falsify — the preset list is read off the runner's `ARMS` dict, and
# the compared column off the baseline artifacts it writes.
#
# One thing the measurement changed about the contract. Clause 2 asked for the two
# flags to be recorded as "the uncompared pair", and they are not:
# `eval/baselines/compaction-summary-arms-2026-09-25.json` carries twelve kept rows
# per preset and a `paired_vs_summary_legacy` block, and
# `eval/measurements/compaction-summary-arms-2026-09-25.md` rules both flags stay
# off. Writing "uncompared" would have made this doc the third place a stale claim
# is copied to, so the section says what was measured and what is still unmeasured
# (D2's cross-turn reuse question, which no preset in `ARMS` can ask). The numbers
# below are what makes that reading re-measurable rather than a nicer sentence.

#: The runner whose `--arms` values the new section inventories. Cited in
#: backticks, never as a wikilink: `[[compaction-recall]]` would have to be a
#: top-level architecture doc for
#: `test_every_wikilink_in_a_new_doc_names_a_top_level_architecture_doc` to
#: tolerate it, and it is not a doc at all.
RUNNER_REL = "eval/run_compaction_recall_eval.py"

#: The config-table cell #1787 rewrote, kept verbatim from base `d00f56d0` so the
#: node that forbids it is known to match something.
OLD_CONTEXT_WINDOW_ROW = (
    "| `compaction.persist_summary` | false | fold vs regenerate, one or the "
    "other. Off until the `summary_persisted` arm of "
    "`eval/run_compaction_recall_eval.py` is compared |")

#: The disclosure clause 5 removes, as it stood at base `d00f56d0`. Compared
#: whitespace-flattened, because the doc wrapped it over three lines.
OLD_DISCLOSURE = ("Neither is named in [[measurement]], which is filed #1787; "
                  "until that lands, the runner is the only place the "
                  "comparison is defined.")

#: The three summary presets, in the order the doc scores them.
SUMMARY_PRESETS = ("summary_legacy", "summary_persisted", "memory_flush")

#: The run that answered the two flags, and its write-up. Both are tracked, and
#: both are cited by the section, so a reader gets the artifact from the prose.
SUMMARY_RUN = EVAL / "baselines/compaction-summary-arms-2026-09-25.json"
SUMMARY_WRITEUP = EVAL / "measurements/compaction-summary-arms-2026-09-25.md"

_H2 = re.compile(r"^## (.+)$", re.M)

#: A row of the preset table: first cell a preset in backticks, second cell the
#: compared marker plus the kept-row count the doc claims for it.
_PRESET_ROW = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|\s*(yes|no)"
                         r"(?:\s*\((\d+)(?: of (\d+))? kept\))?\s*\|", re.M)

#: `[[measurement]] §Section`, as the two delegates spell it. The capture stops at
#: a colon so a reference may name a section with a sub-clause in its heading; the
#: heading match is by unique prefix, which is what a hand-written §-ref is.
_MEASUREMENT_REF = re.compile(r"\[\[measurement\]\] §([A-Za-z][^,.;)\n]*)")


def _flat(text: str) -> str:
    """Text with every run of whitespace collapsed to one space. The doc is
    hand-wrapped at ~78 columns, so any sentence long enough to matter contains a
    newline, and a fixture asserted against the raw bytes fails for where the
    author broke the line."""
    return " ".join(text.split())


def _h2_sections(text: str) -> dict[str, str]:
    """Every top-level `## ` section of a doc, in file order. `### ` is not a
    boundary here, which is the whole reason #1787's content could not be a `### `
    child of §The arms: `_section()`'s window, and so the graded table's window,
    runs to the next `## `."""
    hits = list(_H2.finditer(text))
    return {m.group(1).strip():
            text[m.end(): (hits[i + 1].start() if i + 1 < len(hits) else len(text))]
            for i, m in enumerate(hits)}


def recall_section(text: str) -> tuple[str, str]:
    """The one `## ` section that cites the runner — the section #1787 added."""
    hits = [(head, body) for head, body in _h2_sections(text).items()
            if f"`{RUNNER_REL}`" in body]
    assert len(hits) == 1, (
        f"{RUNNER_REL} is cited by {len(hits)} `## ` sections of measurement.md; "
        "the nodes below need exactly one to grade (a second citation belongs in "
        "the section that already has it)")
    return hits[0]


def runner_presets() -> list[str]:
    """The keys of the runner's `ARMS` dict, read by executing the module: what
    `--arms` actually accepts, so the doc is checked against the parser and not
    against a transcription of the list in prose."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_arms_probe", ROOT / RUNNER_REL)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module builds `@dataclass` types at import and
    # `dataclasses` resolves the declaring module through `sys.modules`. Without
    # this line the probe dies on `AttributeError: 'NoneType' object has no
    # attribute '__dict__'` and says nothing about the presets.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return list(module.ARMS)
    finally:
        sys.modules.pop(spec.name, None)


def baseline_rows() -> dict[str, tuple[int, int]]:
    """`preset -> (rows, kept)` over every tracked `compaction*.json`. This is the
    same definition the artifacts use for themselves: a row counts when its
    `status` is `ok`, which is what the per-arm `kept` in their own `summary`
    block adds up to."""
    rows: collections.Counter = collections.Counter()
    kept: collections.Counter = collections.Counter()
    for path in sorted((EVAL / "baselines").glob("compaction*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")).get("rows", []):
            rows[row["arm"]] += 1
            if (row.get("status") or "ok") == "ok" and not row.get("dropped"):
                kept[row["arm"]] += 1
    return {arm: (rows[arm], kept[arm]) for arm in rows}


def preset_rows(body: str) -> dict[str, tuple[str, int | None, int | None]]:
    """The preset table as `name -> (yes|no, kept count, row count)`."""
    return {name: (mark, int(kept) if kept else None, int(total) if total else None)
            for name, mark, kept, total in _PRESET_ROW.findall(body)}


def _heading_for(ref: str, headings: set[str]) -> str:
    """The one heading a `§ref` points at — matched by unique prefix, since the
    delegates write the short form of a heading that carries a sub-clause."""
    hits = sorted(h for h in headings if h == ref or h.startswith(ref))
    assert len(hits) == 1, (
        f"§{ref!r} matches {len(hits)} headings of measurement.md "
        f"({hits}); a delegate that could mean two sections means neither")
    return hits[0]


def test_the_runner_presets_are_a_section_of_their_own_and_match_the_dict():
    """Clauses 1 and 4: the presets live in a `## ` section that is not §The arms,
    they are exactly what `--arms` accepts, and adding them did not touch the
    directory table.

    The set-equality against `runner_presets()` is what makes the section a
    document and not a guess: add a preset to `ARMS` without a row, or a row
    without a preset, and this node names the difference — the same hold the arms
    table has on directories. The directory table is then re-graded inside its own
    §window rather than trusted, because this diff is the one that put a second
    backtick-first table into the file. Note which window: `_section()` ends at the
    next `## `, which is why the comparison below is scoped to it. Whole-doc
    `arms_named_by()` now also sees the preset rows, so it is the wrong denominator
    for a doc that carries both senses, and `COVERED` does not list
    `measurement.md` for that reason.
    """
    text = _text("measurement.md")
    head, body = recall_section(text)
    assert head != "The arms" and "`--arms`" in body
    assert f"`{RUNNER_REL}`" in body

    rows = preset_rows(body)
    assert rows, "no preset rows parsed — the table stopped being parseable"
    assert set(rows) == set(runner_presets()), (
        f"measurement.md's preset table and {RUNNER_REL}'s ARMS dict disagree: "
        f"doc-only {sorted(set(rows) - set(runner_presets()))}, "
        f"dict-only {sorted(set(runner_presets()) - set(rows))}")
    for preset in SUMMARY_PRESETS:
        assert preset in rows, f"{preset} has no row in the preset table"

    # Clause 4: the graded inventory, in its own window, still names exactly the
    # directories — no row leaked into it, and none was edited.
    # `_ARMS_ROW` over the window, not `arms_named_by()`: that helper takes a
    # whole doc and re-parses §The arms out of it itself (`arms_named_by`
    # line ~89), which is right for its own node and wrong for a doc that now
    # carries a second backtick-first table.
    window = _section(text, "The arms")
    named = set(_ARMS_ROW.findall(window))
    assert f"`{RUNNER_REL}`" not in window, (
        "the runner citation moved inside §The arms, so the window the graded "
        "node parses is no longer only the directory table")
    assert named == eval_arms_on_disk()
    assert not (set(rows) & named), (
        "a preset name also reads as an eval/ directory arm, so the two senses "
        f"have collided: {sorted(set(rows) & named)}")


def test_the_compared_column_is_the_baselines_and_the_quoted_numbers_are_theirs():
    """Clause 2, measured twice: the yes/no column against every tracked
    `compaction*.json`, and each figure the section quotes against the artifact it
    came from.

    Re-deriving the numbers rather than the sentence is the point, because this is
    where the item and the tree disagreed. The clause described `summary_persisted`
    and `memory_flush` as "the uncompared pair"; the artifacts give both twelve
    kept rows and a paired comparison against `summary_legacy`, and the write-up
    ruled both flags stay off. So the section states the comparison and the ruling,
    and this node is what makes a later reader re-measure it instead of re-copying
    either claim — including the claim in this docstring.
    """
    _, body = recall_section(_text("measurement.md"))
    flat = _flat(body)
    rows = preset_rows(body)
    measured = baseline_rows()
    assert measured, "no compaction baselines on disk to compare the column with"
    for preset, (mark, kept, total) in sorted(rows.items()):
        have_rows, have_kept = measured.get(preset, (0, 0))
        assert mark == ("yes" if have_kept else "no"), (
            f"{preset}: the doc says {mark!r}, the tracked baselines kept "
            f"{have_kept} of {have_rows} rows")
        if mark == "yes":
            assert kept == have_kept, (
                f"{preset}: the doc quotes {kept} kept rows, the artifacts have "
                f"{have_kept}")
            if total is not None:
                assert total == have_rows, (
                    f"{preset}: the doc says {total} rows, the artifacts have "
                    f"{have_rows}")
    # `sidecar` joined the set on 2026-10-07 with #2349, which built the arm and
    # could not run it: the figure it exists to produce needs the real primary
    # engine, so no `compaction*.json` has ever held a `sidecar` row. The rail
    # above re-derives every `no` from the tracked artifacts, so this literal is
    # the only thing that can be stale here — and the row it names says so too.
    assert {name for name, (mark, _k, _t) in rows.items() if mark == "no"} == {
        "self_record", "observation", "production_self_record",
        "production_observation", "rung4", "rung4_lossy", "rung4_self_record",
        "sidecar"}, (
        "the never-run set changed shape; the doc's 'not yet run anywhere' "
        "sentence and the baselines have to be read together again")

    # What is scored against what — the thing clause 2 asks a reader to be able
    # to tell — named in the prose and present in the artifact.
    assert SUMMARY_PRESETS[0] in flat and "baseline" in flat.lower()
    assert "`paired_vs_summary_legacy`" in body, (
        "the section no longer names the block that says what is scored against "
        "what")
    run = json.loads(SUMMARY_RUN.read_text(encoding="utf-8"))
    paired = run["paired_vs_summary_legacy"]
    assert set(paired) == set(SUMMARY_PRESETS[1:]), (
        "the artifact no longer pairs exactly the two summary presets against "
        f"summary_legacy: {sorted(paired)}")
    for arm, want in {"summary_legacy": (11, 12), "summary_persisted": (8, 12),
                      "memory_flush": (12, 12)}.items():
        s = run["summary"][arm]
        assert (s["distinctive"]["k"], s["distinctive"]["n"], s["kept"],
                s["dropped"], s["errors"]) == (*want, want[1], 0, 0), (arm, s)
    assert ("twelve kept rows per summary preset, no drops, no errors,"
            " paired against `summary_legacy`") in flat, flat[:400]

    # The figures the prose quotes, each against the field it came from.
    persist = paired["summary_persisted"]
    assert round(persist["distinctive_hit"]["diff"], 2) == -0.25, persist["distinctive_hit"]
    assert "-0.25" in flat and not persist["distinctive_hit"]["significant"]
    assert not persist["distinctive_hit"]["significant"], persist["distinctive_hit"]
    stall = persist["turn_start_wall_s"]
    assert round(stall["diff"]) == 42 and stall["significant"], stall
    assert (round(stall["lo"]), round(stall["hi"])) == (30, 53), stall
    assert "+42 s" in flat and "+30 to +53" in flat
    medians = {a: run["summary"][a]["median_ttft_first_s"] for a in SUMMARY_PRESETS}
    assert round(medians["summary_persisted"], 1) == 11.7, medians
    assert round(medians["summary_legacy"], 1) == 4.1, medians
    assert "11.7 s against legacy's 4.1 s" in flat, medians
    flush_pair = paired["memory_flush"]
    assert round(flush_pair["distinctive_hit"]["diff"], 2) == 0.08, flush_pair
    assert "+0.08" in flat
    rows_ = run["rows"]
    saw = sum(1 for r in rows_ if r["arm"] == "memory_flush"
              and r["flush"]["planted_in_history"])
    saved = sum(1 for r in rows_ if r["arm"] == "memory_flush"
                and r["flush"]["planted_saved"]["distinctive"])
    assert (saw, saved) == (8, 1), (saw, saved)
    assert ("saw the planted fact in the bound history 8 times and wrote the "
            "distinctive one down once") in flat, (saw, saved)

    # And the ruling the two flags rest on belongs to the write-up, quoted here.
    assert SUMMARY_WRITEUP.exists()
    assert "keep both flags off" in flat.lower()
    assert "keep `compaction.persist_summary: false`" in SUMMARY_WRITEUP.read_text(
        encoding="utf-8")

    # The control that these comparisons could fail: one row claiming "no" for a
    # preset with kept rows, one claiming a count no artifact has.
    false_rows = preset_rows("| Preset | Kept rows | What it is |\n|---|---|---|\n"
                             "| `summary_persisted` | no | invented row |\n"
                             "| `none` | yes (99 kept) | invented count |\n")
    assert any(mark == "no" and measured.get(name, (0, 0))[1]
               for name, (mark, _k, _t) in false_rows.items()), (
        "the yes/no comparison cannot see a false 'no', so clause 2 is not being "
        "measured by it")
    assert any(mark == "yes" and kept != measured.get(name, (0, 0))[1]
               for name, (mark, kept, _t) in false_rows.items()), (
        "the kept-count comparison cannot see an inflated count")


def test_the_two_senses_of_arm_are_separated_in_the_prose_and_the_tree():
    """Clause 3: the section says these are `--arms` values inside one runner and
    not the directories §The arms inventories — and shows it twice, because the
    prose alone is the thing that rotted once already.

    The tree half is what makes the sentence checkable rather than stylistic: no
    preset may name a directory under `eval/`, so the two senses stay disjoint by
    construction, and a future run that adds `eval/summary_persisted/` has to
    reconcile the two tables instead of letting a reader confuse them. The
    runner's own `--arms` default is read beside it, so "the threshold family is
    the default" is the parser's default and not the doc's.
    """
    _, body = recall_section(_text("measurement.md"))
    flat = _flat(body).lower().replace("**", "")
    assert "`--arms`" in body
    assert "an `--arms` value gets no row in §the arms" in flat, (
        "the rule separating the two senses is gone from the section, which is "
        "the conflation #1787 was filed over")
    for sense in ("directory", "directories"):
        assert sense in flat, (
            f"the section never uses the word {sense!r}, so it cannot be saying "
            "these presets are not that")

    presets = set(runner_presets())
    dirs = eval_arms_on_disk()
    assert not (presets & dirs), (
        f"a preset shares a name with an eval/ directory: {sorted(presets & dirs)}")
    for arm in ("none", "production", "tool_clear", "memory_flush"):
        assert not (EVAL / arm).exists(), (
            f"{RUNNER_REL} preset {arm!r} is also an eval/{arm} directory, so the "
            "two tables now name one thing two ways")

    src = (ROOT / RUNNER_REL).read_text(encoding="utf-8")
    default = re.search(r'--arms",\s*default="([^"]+)"', src)
    assert default, "the runner's --arms default is no longer parseable"
    listed = default.group(1).split(",")
    assert listed == ["none", "production", "tool_clear", "raised", "trigger90"], listed
    assert all(f"`{name}`" in body for name in listed), (
        "the default family the section describes is not the flag's default")
    assert "`closed_book`" in body and "`memory_eval`" in body, (
        "the section's worked example of the two senses colliding is gone: "
        "`memory_eval` is a directory arm whose runner takes --arms values, which "
        "is why the distinction is stated rather than assumed")
    assert "closed_book" in (EVAL / "run_memory_eval.py").read_text(encoding="utf-8")

    # Control: the disjointness assertion has teeth. `memory_eval` really is a
    # directory, so pretending it is a preset must trip the check above.
    assert (EVAL / "memory_eval").is_dir()
    assert (presets | {"memory_eval"}) & dirs, (
        "injecting a real directory name into the preset set did not trip the "
        "disjointness check, so nothing here is holding the two senses apart")


def test_context_window_delegates_to_the_section_and_says_nothing_unmeasured():
    """Clause 5: `context-window.md` no longer says the answer is absent, and every
    delegate of its own to [[measurement]] lands on a section that is there.

    Four moves, because a removed sentence and a dangling pointer are both ways to
    finish this item wrongly. The two removed spellings are asserted absent against
    the text kept above, so neither assertion is an empty pattern; every
    `[[measurement]] §…` reference in that doc has to match one heading of
    measurement.md by unique prefix — that drift is what made the original pointer
    wrong, so it is pinned for all of them, not only the new ones; the delegate
    paragraph is asserted to name the run's date and the fact that it was measured;
    and the fixtures are asserted to still contain what clause 5 removed.
    """
    cw = _text("context-window.md")
    cflat = _flat(cw)
    assert _flat(OLD_DISCLOSURE) not in cflat, (
        "the disclosure that measurement.md names neither arm is back, and "
        "measurement.md now carries the section it was disclaiming")
    assert "is compared" not in cflat, (
        "a compaction flag is described as waiting on a comparison again; the "
        "comparison ran on 2026-09-25 and ruled both flags stay off")
    assert "summary_persisted" in cflat and "memory_flush" in cflat, (
        "the doc no longer names the two presets at all — an emptied pointer is "
        "not a fixed one")

    refs = _MEASUREMENT_REF.findall(cw)
    assert refs, "context-window.md cites [[measurement]] with no named section"
    heads = set(_h2_sections(_text("measurement.md")))
    for ref in refs:
        _heading_for(ref.strip(), heads)

    paras = [q for q in cw.split("\n\n")
             if "`compaction.memory_flush`" in q and "app/memory_flush.py" in q]
    assert len(paras) == 1, (
        f"{len(paras)} paragraphs cover the memory_flush delegate, so clause 5 "
        "has nothing specific to grade")
    joined = _flat(paras[0])
    assert "measured against `summary_legacy` on 2026-09-25" in joined, joined
    assert "[[measurement]]" in joined and "§" in joined, joined

    # The fixtures still carry what was removed, so the two absences above are not
    # passing because the patterns went stale.
    assert OLD_CONTEXT_WINDOW_ROW.endswith("is compared |")
    assert "is compared" in OLD_CONTEXT_WINDOW_ROW
    assert "Neither is named in" in OLD_DISCLOSURE
    assert not _MEASUREMENT_REF.search(OLD_DISCLOSURE), (
        "the old disclosure carried a §-reference, so the reference check above "
        "would have passed on the text clause 5 removes — it is no longer a "
        "control")


#: The heading #2027 landed its prefix-break verdict under — inside this existing
#: doc, not in a new file. A new top-level doc would have had to move
#: `ARCH_DOC_SURFACE` above, which is the sweep's vacuity guard, and a measurement
#: write-up is not the place to change what the sweep grades.
REWRITE_COST_HEADING = "What a rewrite costs, priced per mechanism"


def test_the_rewrite_cost_verdict_names_its_break_sites_and_its_command():
    """Clause 4: the verdict is prose in the doc, and every number in it resolves.

    Four halves, all graded on the one slice `_section` cuts, so nothing here can
    pass on text the reader never meets:

    * both front-break sites, each with the code it quotes — `app/compaction.py:505`
      building the compacted conversation with the summary at position 0, and the
      oldest-first clearing rule that puts microcompaction's break near the front;
    * the command that reproduces the arm table, as a `sql` fence answering BOTH
      freeing paths — a predicate reading only `turn_start.*` is what made this
      item's first draft report 235 turns instead of 360, and it fails silently;
    * the headline stated as NOT size-controlled, asserted by walking every
      occurrence of the word rather than by matching one sentence;
    * the figures quoted from `app/usage_store.py`'s own constants — the window, the
      row count and the total — so the prose and the module cannot drift apart
      without one of them going red.
    """
    from app import usage_store

    sec = _section(_text("context-window.md"), REWRITE_COST_HEADING)
    flat = _flat(sec)
    # The slice is the priced section, not an empty match: the arm table's own
    # headline ratio is inside it.
    assert "51.8" in flat, f"the section slice holds no measured ratio:\n{sec[:200]}"

    # (a) the front-break sites, named as sites rather than described.
    assert "`app/compaction.py:505` is `new_convo: list[dict] = " \
        "[CS.summary_message(record)]`" in flat, flat[:400]
    assert "app/harness/microcompact.py:11" in sec, flat[:400]
    assert "Clearing is oldest-first by design" in flat, (
        "the microcompact site is cited without the rule that makes it a "
        "front break, so the citation prices nothing")

    # (b) the reproducing command, with both freeing paths in its arm predicate.
    fence = re.search(r"```sql\n(.*?)```", sec, re.DOTALL)
    assert fence, "no sql fence in the section, so the arm table cannot be re-run"
    sql = fence.group(1)
    assert "$.turn_start.tokens_freed" in sql and "$.relief_tokens_freed" in sql, (
        "the printed arm predicate reads only one of the two freeing paths, which "
        "is the mistake that filed this item with weaker numbers")
    assert "reprefill_tokens is not null" in sql, sql
    # Positive control for the assertion above: the single-path spelling this item's
    # retracted draft used is contained in the block, so the two-path check above is
    # grading a real extension of that shape and not a pattern nothing matches.
    assert "when coalesce(json_extract(compaction,'$.turn_start.tokens_freed'),0)>0" \
        in sql, sql
    # The window is quoted literally, BOTH bounds, inside the fence itself. Matching
    # the closing minute against the whole section would not do: the prose names that
    # minute anyway ("at 2026-10-01T16:28Z"), so a fence carrying only `ts>=` passes
    # while the command printed above the arm table answers every row from 2026-09-24
    # on — a strict superset of the 3,868-row / 112,908,568-token window the table
    # beside it is pinned to, and the discrepancy grows by a day's rows per day. The
    # bound is the same one `reprefill_attribution` closes its own row scan with.
    assert f"ts>='{usage_store.REPREFILL_WITNESS_SINCE}'" in sql, sql
    assert f"ts<'{usage_store.REPREFILL_WITNESS_UNTIL}'" in sql, (
        "the fence carries no upper bound, so the command above the arm table cannot "
        "return the table it is printed above on any re-run after the window closes:\n"
        + sql)
    # Control: the half-closed spelling is one character from the fenced one, so the
    # assertion above bites rather than matching a shape nothing else could write.
    assert f"ts>='{usage_store.REPREFILL_WITNESS_SINCE}'" in sql
    assert f"ts>='{usage_store.REPREFILL_WITNESS_UNTIL}'" not in sql
    assert "USAGE_DB" in sec and "app/paths.py" in sec, (
        "the command names no database to run against")

    # (c) the headline is not presented as size-controlled. The invariant is on the
    # CLAIM, not on the word: the section also speaks of "the paired size-controlled
    # ratio" as something owed, which is an honest sentence and must stay.
    NEGATED = "That 51.8× is not size-controlled, and is stated as such."
    assert NEGATED in flat, (
        f"the section never states the headline as {NEGATED!r}:\n{flat[:400]}")
    assert "size-controlled" in flat, (
        "the section stopped using the words at all, and an absent claim is not a "
        "caveat — clause 4 asks for the headline stated with its caveat")
    affirmative = re.compile(r"\bis\s+size-controlled")
    assert not affirmative.search(sec), (
        "the section states the headline ratio as size-controlled: "
        f"…{sec[affirmative.search(sec).start() - 80:affirmative.search(sec).end() + 40]}…")
    # Positive control: the same regex fires on the negated sentence with its `not`
    # removed, so the assertion above is not searching a shape that cannot occur.
    assert affirmative.search(NEGATED.replace("not ", "")), (
        "the affirmative pattern matches even the doc's own caveat stripped of its "
        "'not', so it was never going to catch a re-write")

    # (d) the constants the replay route is driven by are the ones the prose quotes.
    assert usage_store.REPREFILL_WITNESS in sec, sec[:400]
    assert f"{usage_store.REPREFILL_WITNESS_ROWS:,} rows" in flat
    assert f"{usage_store.REPREFILL_WITNESS_TOTAL:,}" in flat
    assert "REPREFILL_WITNESS_SINCE" in sec and "REPREFILL_WITNESS_UNTIL" in sec, (
        "the section no longer names the two constants the replay window is cut "
        "from, so its figures cannot be re-derived")
    assert usage_store.REPREFILL_WITNESS_UNTIL[:16] in sec, (
        "the minute the prose says it measured at is no longer the minute the "
        "replay window closes at, so the command and the published split are two "
        "different windows")
    # The bucket the trade is priced at, row and all — the same (n, Σ) pair
    # `test_the_extract_replays_to_its_own_published_split` re-derives from the
    # committed witness bytes, kept here as the sentence a reader quotes.
    assert "| `turn_start:microcompact` | 236 | 68,444,409 |" in sec, flat[:400]


# ────────────────────────────────────────────────────────── command spellings
#
# The half above proves a doc's *path* resolves. It cannot see whether a doc's
# *command* resolves: `_CITATION` only fires on a span with a slash in it, so
# `run-fixtures-eval` — a verb `architecture/measurement.md` once asserted for
# `app/harness/supply_chain.py`, which the module never dispatched — extracted
# as nothing and the check stayed silent while an operator following the doc
# got a usage dump and exit 1 (#1792). This is the same failure the path half
# guards, one layer up: a citation that resolves to no code.
#
# A claim is graded only where the target module has a dispatch chain to grade
# it against. A module that dispatches no subcommand at all (a library, or a
# script with only flags) cannot be mis-cited as having one, and the alternative
# — reporting a bare identifier beside such a module — reads a symbol citation
# as a broken command. Every top-level doc is swept (#1890), so this scoping,
# not a list of docs or modules, is what keeps the half quiet: of the corpus's 15
# command spans, 3 sit beside a module with no chain and are skipped on that
# property alone — see the finding appended to #1792 and the two rulings in this
# file's docstring.

_COMMAND_SPAN = re.compile(r"`([^`\n]+)`")
# `.py` path with an optional `:NNN` anchor — the same shape `cited_paths` takes.
_PATH_SCRIPT = re.compile(r"\A(?P<mod>(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_\-]+\.py)"
                          r"(?::\d+)?\Z")
# `python -m` takes a dotted module, which has no slash and so never reaches
# `cited_paths` at all.
_DOTTED_MODULE = re.compile(r"\A(?P<mod>[A-Za-z_][A-Za-z0-9_]*"
                            r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\Z")
_INTERPRETER = re.compile(r"(?:^|/)python\d*(?:\.\d+)?\Z")
# A subcommand, as this tree spells them: lowercase words joined by hyphens
# (`scan`, `write-baseline`, `board-decisions`). An underscore is excluded on
# purpose: the only subcommand strings in the repo carrying one are private
# helpers (`add_parser("_child")`), while every snake_case span a doc brackets
# after a module is a *symbol* citation — `check_bash_command`,
# `allow_protected_writes`, `check_deployed_copies`.
_COMMAND_WORD = re.compile(r"\A[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
# A bracket opened right after a module citation is the doc's own way of saying
# "and these are its commands" — `supply_chain.py` (`fixtures`,
# `scan --write-baseline`). Prose inside the bracket ends the list.
_LIST_GAP = re.compile(r"\A[\s,]*\Z")


def _script_module(tok):
    """The repo file a `.py` path token names, or None."""
    m = _PATH_SCRIPT.match(tok)
    if not m:
        return None
    p = ROOT / m.group("mod")
    return p if p.is_file() else None


def _span_module_and_rest(span):
    """(module file, the tokens after it) when a span names a module on disk.

    Two shapes reach a module: `[interpreter] path/to/script.py …` and
    `[interpreter] -m dotted.module …`, with the interpreter allowed anywhere in
    front (`.venvs/lloyd/bin/python -m …`). The module must be the first token
    after the interpreter and `-m`, so a span that merely *starts* with a cited
    path (`server.py's USAGE constant`, `web/src/app/page.tsx:1095`) has nothing
    left that can be a subcommand and is not a claim.
    """
    toks = span.split()
    i = 0
    while i < len(toks) and _INTERPRETER.search(toks[i]):
        i += 1
    module = None
    if i < len(toks) and toks[i] == "-m" and i + 1 < len(toks):
        dotted = _DOTTED_MODULE.match(toks[i + 1])
        if dotted:
            p = ROOT / (dotted.group("mod").replace(".", "/") + ".py")
            module = p if p.is_file() else None
        i += 2
    elif i < len(toks) and (module := _script_module(toks[i])) is not None:
        i += 1
    return module, toks[i:]


def _only_flags(rest):
    """True when the leftover tokens are flags and the values they were given."""
    i = 0
    while i < len(rest):
        if not rest[i].startswith("-"):
            return False
        i += 1
        if i < len(rest) and not rest[i].startswith("-"):
            i += 1                    # a value sitting right after its flag
    return True


def command_claims(text):
    """Every code span in *text* that claims a subcommand of a module: (line,
    span, module path, verb).

    An inline claim is a span carrying the module and the verb together; a list
    claim is a verb span sitting inside a bracket opened immediately after a
    module citation on the same line. A module cited without a verb is recorded
    only so the list form has something to hang on — nothing here is decided
    about a module that is not on disk.
    """
    claims = []
    for ln, line in enumerate(text.splitlines(), 1):
        cited = None                  # module named to the left, on this line
        listing = None                # module whose bracketed list we are in
        opened = False                # the list was opened by this span's gap
        prev_end = None
        for m in _COMMAND_SPAN.finditer(line):
            span = m.group(1)
            gap = line[prev_end:m.start()] if prev_end is not None else ""
            if listing is None and cited is not None and re.match(r"\A\s*[\(\[]", gap):
                listing, opened = cited, True
            module, rest = _span_module_and_rest(span)
            if module is not None:
                if rest and _COMMAND_WORD.match(rest[0]):
                    claims.append((ln, span, module, rest[0]))
                cited, listing, opened, prev_end = module, None, False, m.end()
                continue
            if listing is not None:
                if ((opened or _LIST_GAP.match(gap)) and rest
                        and _COMMAND_WORD.match(rest[0]) and _only_flags(rest[1:])):
                    claims.append((ln, span, listing, rest[0]))
                    opened, prev_end = False, m.end()
                    continue
                listing = None        # prose in the bracket ends the list
            # Neither a command nor a verb inside someone else's list: the
            # current module is gone, so a later bracket on this line cannot
            # borrow it.
            cited, opened, prev_end = None, False, m.end()
    return claims


# --------------------------------------------------------------------------- #
# The dispatch chain, read by parsing — never by importing. `app.harness.
# supply_chain` and `workers.sources.automod_regression` pull in enough of the
# machine that importing them from a gate worktree is its own hazard, and this
# file runs there (#1721). AST parsing has the same answer with no side effects.
# --------------------------------------------------------------------------- #

_USES_A_COMMAND = {"command", "cmd", "subcommand", "verb", "action"}
_USAGE_VARIABLES = {"USAGE", "CLI", "HELP", "USAGE_TEXT", "EPILOG"}
#: The argument vector itself, under either spelling this tree uses: a bare
#: `argv` for a `main(argv=None)` helper, or `sys.argv` in a script that reads
#: it directly.
_ARGV_NAME = "argv"


def _is_argv(node):
    """Is this expression the argument vector — `argv`, `sys.argv`?"""
    if isinstance(node, ast.Name):
        return node.id.lower() == _ARGV_NAME
    if isinstance(node, ast.Attribute):
        return node.attr.lower() == _ARGV_NAME
    return False


def _dispatch_slot(node):
    """Is this expression the thing a subcommand gets compared against?

    Two ways a subscript is one, and they name different halves of it: the
    *index* being command-named is a lookup table (`DISPATCH[command]`), while
    the *base* being the argument vector is a stdlib-only script reading its own
    positional arguments (`agent-services/rpc/lloyd_rpc.py:243`,
    `if len(argv) < 2 or argv[0] not in ("call", "map")`). Only the index case
    was read, so the RPC client's chain harvested the one verb that also turned
    up in a `verb == "call"` comparison and reported `map`, which
    `architecture/harness.md:1573` correctly cites, as invented (#1890). A
    positional integer index is required: `sys.argv[1:]` is a slice, not a slot.
    """
    if isinstance(node, ast.Name):
        return node.id.lower() in _USES_A_COMMAND
    if isinstance(node, ast.Attribute):
        return node.attr.lower() in _USES_A_COMMAND
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        if str(node.slice.value).lower() in _USES_A_COMMAND:
            return True
        return _is_argv(node.value) and isinstance(node.slice.value, int)
    return False


def _verb_like(value) -> bool:
    """Is this compared constant a subcommand, as opposed to a flag or prose?

    The span grammar already refuses a flag on the doc side (`_COMMAND_WORD`
    needs a leading lowercase letter, which is why `-h, --help` never enters a
    chain from a usage string). The same grammar has to decide the code side, or
    harvesting `sys.argv` turns a flag check into a whole dispatch chain:
    `agent-services/services/idle-worker/check-github-releases.py:184` compares
    `sys.argv[1] == "--init"`, and read naively that module's chain would be one
    flag — grading every verb anyone ever cites against it, and reporting a doc
    that cited `scan` beside a module that takes no subcommand at all. With the
    flag dropped that module has no chain and is skipped, which is the sentence
    `undispatched_claims` already applies to a library (#1890 clause 3).
    """
    return isinstance(value, str) and bool(_COMMAND_WORD.match(value))


class _Chain(ast.NodeVisitor):
    """Collects every subcommand a module can dispatch to, and what it forwards.

    Four spellings cover this tree: argparse `add_parser("x")`, argparse
    `add_argument("command", choices=(…))`, a comparison against a
    command-shaped name *or a positional of `argv`/`sys.argv`*
    (`if command == "gate"`, `if argv[0] not in ("call", "map")`), and a `match`
    over one. Plus the `USAGE` string the module prints when it does not
    recognise a verb, whose leading word per line is a verb it does. A compared
    constant counts only if `_verb_like` says so: a flag is not a subcommand.
    """

    def __init__(self):
        self.verbs = set()
        self.usage = ""
        self.imports = []
        self.forwards_main = False

    @staticmethod
    def _strings(node):
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        return []

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id.upper() in _USAGE_VARIABLES:
                v = node.value
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    self.usage += v.value + "\n"
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "main":
            self.forwards_main = True
        elif isinstance(f, ast.Name) and f.id == "main":
            self.forwards_main = True
        if isinstance(f, ast.Attribute) and f.attr == "add_parser" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                self.verbs.add(first.value)
        if isinstance(f, ast.Attribute) and f.attr == "add_argument" and node.args:
            first = node.args[0]
            positional = (isinstance(first, ast.Constant)
                          and isinstance(first.value, str)
                          and not first.value.startswith("-"))
            if positional:
                for kw in node.keywords:
                    if kw.arg == "choices":
                        self.verbs.update(self._strings(kw.value))
        self.generic_visit(node)

    def visit_Compare(self, node):
        sides = [node.left, *node.comparators]
        if any(_dispatch_slot(x) for x in sides):
            for x in sides:
                if isinstance(x, ast.Constant) and _verb_like(x.value):
                    self.verbs.add(x.value)
            for op, other in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    self.verbs.update(v for v in self._strings(other) if _verb_like(v))
        self.generic_visit(node)

    def visit_Match(self, node):
        if _dispatch_slot(node.subject):
            for case in node.cases:
                stack = [case.pattern]
                while stack:
                    pat = stack.pop()
                    if (isinstance(pat, ast.MatchValue)
                            and isinstance(pat.value, ast.Constant)
                            and isinstance(pat.value.value, str)):
                        self.verbs.add(pat.value.value)
                    for field in getattr(pat, "_fields", ()):
                        child = getattr(pat, field)
                        for c in (child if isinstance(child, list) else [child]):
                            if isinstance(c, ast.pattern):
                                stack.append(c)
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports.extend(a.name for a in node.names)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.extend(f"{node.module}.{a.name}" for a in node.names)


def _usage_words(usage):
    """Verbs named by a module's usage string.

    A command entry is an indented line sitting in the block's least-indented
    column, and its first word is the verb — which is how the one usage block in
    this tree that lists verbs at all formats them:
    `app/harness/supply_chain.py:1772-1784` puts `scan`, `deps`, `provenance` and
    `fixtures` at two spaces and their descriptions at eight. Everything else is
    skipped: the unindented `usage:` synopsis, whose first word is not a verb;
    the descriptions wrapped deeper than the entries, whose first words are prose
    — :1782 and :1783 resume under `fixtures` beginning `the`, and read as
    entries they would teach the checker that `the` is a subcommand, waving
    through any doc that said so; and a `--flag`, which is a flag and not a
    subcommand. A module that spells its verbs only here dispatches them exactly
    as if the code compared them.
    """
    entries = [ln for ln in usage.splitlines() if ln[:1].isspace() and ln.strip()]
    if not entries:
        return set()
    column = min(len(ln) - len(ln.lstrip()) for ln in entries)
    words = set()
    for ln in entries:
        if len(ln) - len(ln.lstrip()) != column:
            continue
        first = ln.split()[0]
        if _COMMAND_WORD.match(first):
            words.add(first)
    return words


_chain_cache = {}


def _chain_of(path):
    """(verbs defined here, module it forwards to) for a repo file."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), None
    c = _Chain()
    c.visit(tree)
    verbs = c.verbs | _usage_words(c.usage)
    forward = None
    if not verbs and c.forwards_main:
        # A forwarder (scripts/automod/regression_runner.py is 16 lines and 701
        # bytes) names no verb itself. It is followed only when it dispatches
        # nothing locally and calls a `main`, which is what a forwarder is; the
        # deepest import name it holds is the module, not the package it sits
        # in, so a shim is never confused with its own `__init__`.
        named = sorted({i for i in c.imports if not i.startswith("_")},
                       key=lambda n: (n.split(".")[-1] == path.stem,
                                       n.count(".")),
                       reverse=True)
        for name in named:
            p = ROOT / (name.replace(".", "/") + ".py")
            if p.is_file() and p != path:
                forward = p
                break
    return verbs, forward


def dispatch_verbs(path, _seen=frozenset()):
    """Every subcommand *path* can be invoked with, following one forwarding
    import when the file itself dispatches nothing."""
    if path in _chain_cache:
        verbs, forward = _chain_cache[path]
    else:
        verbs, forward = _chain_of(path)
        _chain_cache[path] = (verbs, forward)
    if verbs or forward is None or forward in _seen:
        return verbs
    return dispatch_verbs(forward, _seen | {path})


def undispatched_claims(text, label="<doc>"):
    """Report lines for the command claims in *text* whose verb the module they
    name does not dispatch.

    A module with no dispatch chain at all is not graded against: a library or
    a flag-only script cannot be mis-cited with a subcommand, and reporting a
    bare identifier beside one would read a symbol citation as a broken
    command.
    """
    reports = []
    for ln, span, module, verb in command_claims(text):
        known = dispatch_verbs(module)
        if not known:
            continue
        if verb not in known:
            reports.append(
                f"{label}:{ln} cites `{span}`, but `{verb}` is not a subcommand "
                f"of {module.relative_to(ROOT)} — its dispatch chain is "
                f"{sorted(known)}, so an operator following the doc gets the "
                f"module's usage dump and exit 1")
    return reports


def test_the_command_sweep_covers_every_architecture_doc():
    """Clause 1 (#1890): the graded surface is the corpus, not the four #1699 docs.

    `NEW_DOCS` protected 2 of the corpus's 15 command spans, both in
    `measurement.md`; the other 13 sat in 7 docs no node read, and the report the
    corpus-wide sweep produced was one of them. The sweep now reads
    `ARCH.glob("*.md")`, the same glob `workers/sources/arch_review.py::doc_slugs`
    uses to decide what is a review unit at all — a doc cannot be reviewed by a
    cadence it is not in, and it cannot be graded by a check that is not in it
    either. `ARCH_DOC_SURFACE` is what stops "the sweep reads the glob" from
    meaning "the sweep reads whatever it happens to find".
    """
    docs = sorted(ARCH.glob("*.md"))
    assert len(docs) == ARCH_DOC_SURFACE, (
        f"the glob found {len(docs)} top-level architecture docs, want "
        f"{ARCH_DOC_SURFACE}. The sweep grades the glob's output, so a count that "
        f"moved means either a doc landed (update the constant and the sweep now "
        f"grades it too) or this file is reading the wrong tree")

    graded = []
    for p in docs:
        text = p.read_text(encoding="utf-8", errors="replace")
        graded += [(p.name, ln, verb) for ln, span, module, verb
                   in command_claims(text) if dispatch_verbs(module)]

    assert graded, ("the command sweep graded nothing across "
                    f"{len(docs)} docs about this machine, which means the "
                    "extractor or the resolver went stale — the same vacuity the "
                    "path half pins at its own extractor")
    # The widening itself: claims outside the four docs are being graded, not
    # skipped. Anything that narrowed the sweep back — a per-doc exemption, a
    # re-introduced `for slug in NEW_DOCS` — empties this set.
    outside = sorted({g for g in graded if g[0] not in NEW_DOCS})
    assert outside, (
        "every graded command claim came from the four #1699 docs, so the other "
        f"{ARCH_DOC_SURFACE - len(NEW_DOCS)} docs are grading nothing")

    # What it graded, pinned so the aggregate above cannot be satisfied by one
    # lucky span: both verbs the supply-chain row cites (#1792's pins, kept).
    assert ("measurement.md", 73, "fixtures") in graded, graded
    assert ("measurement.md", 73, "scan") in graded, graded


def test_no_architecture_doc_cites_a_command_its_module_does_not_dispatch():
    """Clause 4 (#1890): the corpus decides clean, and nothing is excused by name.

    Zero reports is only worth something if the sweep is the one that graded
    twelve claims to get there, so this node pins the shape of the pass alongside
    it: which docs contributed a graded claim, and which claims were skipped —
    the latter because the skip is the one place a false negative can hide, and
    `undispatched_claims` excuses a claim on exactly one property, that the module
    cited has no dispatch chain at all. A fourth name entering `excused` means a
    citation stopped being graded and nobody read why; a name leaving `graded_by`
    means a doc's claims went ungraded, which is what an exemption list looks
    like from the inside.
    """
    docs = sorted(ARCH.glob("*.md"))
    assert len(docs) == ARCH_DOC_SURFACE, (
        f"{len(docs)} top-level docs, want {ARCH_DOC_SURFACE} — see the sweep node")

    bad, graded_by, excused = [], set(), []
    for p in docs:
        text = p.read_text(encoding="utf-8", errors="replace")
        bad += undispatched_claims(text, p.name)
        for ln, span, module, verb in command_claims(text):
            if dispatch_verbs(module):
                graded_by.add(p.name)
            else:
                excused.append((p.name, verb))

    assert not bad, "\n".join(bad)
    assert graded_by == {"automod.md", "harness.md", "infrastructure.md",
                         "measurement.md", "research-pipeline.md", "voice.md"}, (
        f"graded claims came from {sorted(graded_by)}; harness.md among them is "
        f"the point of #1890 — its `lloyd_rpc.py` row was the report the widening "
        f"existed to settle")
    assert sorted(excused) == [("djev.md", "rerank"),
                               ("harness.md", "structure"),
                               ("mission-control.md", "route")], (
        "a command claim was skipped, or stopped being skipped, and the only "
        "legitimate reason is the cited module having no dispatch chain: "
        "check whether that module grew one, not whether the doc is on a list")


def test_an_argv_positional_test_is_read_as_a_dispatch_chain():
    """Clause 2 (#1890): `argv[0] not in ("call", "map")` IS the dispatch chain.

    `agent-services/rpc/lloyd_rpc.py` is stdlib-only and takes no argparse: it
    checks its first positional argument against a tuple at :243 and branches on
    `verb == "call"` at :261. Reading only the second spelling is why its chain
    came back as `{'call'}` and `architecture/harness.md:1573`, which correctly
    cites `call` and `map` with `concurrency`, was reported as inventing `map` —
    a doc graded wrong by the reader, the exact mistake #1792 was filed to stop.
    The synthetic half is the `argv-positional` and `sys-argv-positional` rows of
    `_DISPATCH_SOURCES`, run by `test_each_dispatch_spelling_yields_its_verbs`.
    """
    rpc = ROOT / "agent-services/rpc/lloyd_rpc.py"
    assert dispatch_verbs(rpc) == {"call", "map"}, (
        f"lloyd_rpc.py harvested {sorted(dispatch_verbs(rpc))}; its usage dump at "
        f":240 prints `call` and `map` and its guard is "
        f"`argv[0] not in (\"call\", \"map\")`, so a chain missing either one "
        f"reports a correct doc as invented")

    # The doc line that was reported: graded against the corpus's own extractor,
    # this is the report #1890 removes.
    assert undispatched_claims(_text("harness.md"), "harness.md") == [], (
        "harness.md reports again: " + "; ".join(
            undispatched_claims(_text("harness.md"), "harness.md")))


def test_a_flag_shaped_token_is_never_harvested_as_a_verb():
    """Clause 3 (#1890): harvesting `argv` must not turn a flag into a subcommand.

    `agent-services/services/idle-worker/check-github-releases.py:184` reads
    `if len(sys.argv) > 1 and sys.argv[1] == "--init"`, and naively that module's
    entire dispatch chain is one flag. Grading against a one-flag chain is worse
    than grading nothing: every verb anyone ever brackets beside that module is
    then reported as invented, and the doc that said `scan` beside a script that
    takes `--init` or nothing would be sent off to fix a line that is right. With
    the flag dropped the module has no chain, and the sentence
    `undispatched_claims` already applies to a library covers it.
    """
    assert not _verb_like("--init") and not _verb_like("-i"), (
        "a flag-shaped token was accepted as a subcommand, which is the harvest "
        "this node exists to refuse")
    assert _verb_like("board-pass") and not _verb_like("check_deployed_copies"), (
        "_verb_like stopped being the same grammar the doc side uses: hyphen-joined "
        "lowercase verbs in, snake_case symbols and flags out")

    flag_only = ROOT / "agent-services/services/idle-worker/check-github-releases.py"
    assert dispatch_verbs(flag_only) == set(), (
        f"check-github-releases.py harvested {sorted(dispatch_verbs(flag_only))}; "
        f"it compares only `sys.argv[1] == \"--init\"`, so it takes no subcommand "
        f"and must be skipped rather than graded against one flag")

    # Positive control: the verbs beside it ARE extracted, so the empty report
    # below is the chain-less skip and not the extractor going blind.
    line = ("`agent-services/services/idle-worker/check-github-releases.py` "
            "(`scan`, `board-pass`)\n")
    assert [c[3] for c in command_claims(line)] == ["scan", "board-pass"], (
        "the extraction itself found nothing, so the skip proves nothing")
    assert undispatched_claims(line) == [], undispatched_claims(line)


def test_the_widened_sweep_is_shown_able_to_fail(tmp_path):
    """Clause 5 (#1890): wider must not mean weaker — a doc nobody has ever
    heard of still gets reported on the first run.

    The tmp doc is outside `NEW_DOCS` and outside `ARCH/` entirely, so this is
    the widened function being shown to decide something the corpus cannot: the
    invented verb is reported, the verb the module really dispatches is not, and
    neither verdict depends on the doc's name. It is the same call the sweep
    makes, on text no pin covers.
    """
    doc = tmp_path / "a-doc-that-does-not-exist-yet.md"
    assert doc.name not in NEW_DOCS and not (ARCH / doc.name).exists(), (
        "the fixture doc joined the graded corpus, so this is not an unseen doc")

    doc.write_text("Run `python -m app.harness.supply_chain run-fixtures-eval` "
                   "to refresh it.\n", encoding="utf-8")
    reports = undispatched_claims(doc.read_text(encoding="utf-8"), doc.name)
    assert len(reports) == 1, reports
    assert "run-fixtures-eval" in reports[0] \
        and "app/harness/supply_chain.py" in reports[0], reports[0]

    doc.write_text("Run `python -m app/harness.supply_chain fixtures --limit 5` "
                   "to refresh it.\n", encoding="utf-8")
    assert undispatched_claims(doc.read_text(encoding="utf-8"), doc.name) == [], (
        "a verb the module really dispatches was reported in a doc the sweep has "
        "no pin for, so the widened surface now reports correct docs")


def test_an_invented_subcommand_is_reported_and_a_real_one_is_not():
    """Clause 1: a verb absent from the dispatch chain is reported; the verbs
    the doc really cites are not.

    Both halves run over text this test writes, so neither can be satisfied by
    the corpus going quiet. `run-fixtures-eval` is the verb
    `architecture/measurement.md:72` asserted before commit `d00f56d0` corrected
    it by prose; `app/harness/supply_chain.py` dispatches `scan`, `deps`,
    `provenance` and `fixtures` and falls through anything else to a usage dump
    and exit 1 (`main`, app/harness/supply_chain.py:1787-1830).
    """
    module = ROOT / "app/harness/supply_chain.py"
    row = ("| `supply-chain` | whether the scan ran | `app/harness/supply_chain.py` "
           "(`fixtures`, `scan --write-baseline`) |\n")
    invented = ("| `supply-chain` | whether the scan ran | `app/harness/supply_chain.py` "
                "(`run-fixtures-eval`) |\n")
    inline = "Run `python -m app.harness.supply_chain fixtures` to measure it.\n"

    known = dispatch_verbs(module)
    assert known == {"scan", "deps", "provenance", "fixtures"}, (
        f"the chain harvested {sorted(known)} — the four verbs this module "
        f"dispatches are those, and anything else came in through a usage "
        f"string's description text rather than its command column")

    real = [(ln, s, v) for ln, s, m, v in command_claims(row)]
    assert real == [(1, "fixtures", "fixtures"), (1, "scan --write-baseline", "scan")], real
    assert undispatched_claims(row) == [], (
        "the verbs the doc really cites were reported: "
        + "; ".join(undispatched_claims(row)))

    claim = command_claims(invented)
    assert claim == [(1, "run-fixtures-eval", module, "run-fixtures-eval")], claim
    assert "run-fixtures-eval" not in known, (
        "the invented verb resolved against the real dispatch chain, so the "
        "report below proves nothing")
    reports = undispatched_claims(invented, "measurement.md")
    assert len(reports) == 1, reports
    assert "run-fixtures-eval" in reports[0] and "app/harness/supply_chain.py" \
        in reports[0], reports[0]

    # The inline form resolves the same chain, and reports the same way.
    assert command_claims(inline) == [(1, inline.split("`")[1], module, "fixtures")]
    assert undispatched_claims(inline) == []
    inline_bad = "Run `python -m app.harness.supply_chain run-fixtures-eval` now.\n"
    assert len(undispatched_claims(inline_bad)) == 1, undispatched_claims(inline_bad)
    assert undispatched_claims(
        "Run `app/harness/supply_chain.py` to measure it.\n") == [], (
        "a module cited with no verb was read as a command claim")


def test_a_forwarding_shim_resolves_to_the_module_it_forwards_to():
    """Clause 2: `regression_runner noise` passes through the 16-line shim.

    `scripts/automod/regression_runner.py` is 16 lines and 701 bytes: it imports
    `workers.sources.automod_regression` and calls its `main`. The verb lives at
    `workers/sources/automod_regression.py:2226`
    (`add_argument("command", choices=("run", "pending", "latest", "noise"))`)
    and `:2233` (`if args.command == "noise":`), so a check that read only the
    file the span names would report a correct doc as invented — the exact
    mistake #1792's triage names at `architecture/automod.md:3602`. The control
    below proves the hop is not a free pass.
    """
    shim = ROOT / "scripts/automod/regression_runner.py"
    target = ROOT / "workers/sources/automod_regression.py"
    assert shim.is_file() and target.is_file()
    assert len(shim.read_text(encoding="utf-8").splitlines()) == 16, (
        "the shim grew dispatch of its own, so this node no longer proves the "
        "hop and needs re-scoping")

    span = "python -m scripts.automod.regression_runner noise"
    claims = command_claims(f"Run `{span}` first.\n")
    assert claims == [(1, span, shim, "noise")], claims
    assert "noise" in dispatch_verbs(shim), (
        "`noise` did not resolve through the forwarder, so every doc citing the "
        "shim would be reported as inventing it")
    assert not _chain_of(shim)[0], "the shim dispatches locally now; the hop is unused"
    assert dispatch_verbs(shim) == dispatch_verbs(target), (
        "resolving through the shim produced a different chain than the target's "
        "own, so the verb set is not the one the CLI accepts")

    assert undispatched_claims(f"Run `{span}` first.\n") == [], (
        "the doc line the triage names as correct was reported as invented")

    # The hop is not a free pass: a verb the target does not dispatch is still
    # reported, and reported against the module the doc named.
    bogus = "`python -m scripts.automod.regression_runner regression-runner-bogus`\n"
    reports = undispatched_claims(bogus)
    assert len(reports) == 1, reports
    assert "regression-runner-bogus" in reports[0]
    assert "scripts/automod/regression_runner.py" in reports[0], reports[0]


#: Every spelling of "these are my subcommands" this tree uses, and the verbs
#: each is supposed to yield. A doc may cite a module that dispatches by any of
#: them; a resolver that read only one would report the others as invented,
#: which is the same mistake in the other direction.
_DISPATCH_SOURCES = {
    "add_parser": ('import argparse\n'
                   'sp = ap.add_subparsers()\n'
                   'sp.add_parser("audit")\n'
                   'sp.add_parser("re-audit")\n', {"audit", "re-audit"}),
    "positional-choices": ('ap.add_argument("command", choices=("run", "noise"))\n'
                           'ap.add_argument("--format", choices=("text", "json"))\n',
                           {"run", "noise"}),
    "comparison": ('if command == "scan":\n    pass\n'
                   'elif command != "deps":\n    pass\n', {"scan", "deps"}),
    "membership": ('if cmd in ("grade", "compare"):\n    pass\n',
                   {"grade", "compare"}),
    "match-statement": ('match args.command:\n'
                        '    case "prepare":\n        pass\n'
                        '    case _:\n        pass\n', {"prepare"}),
    # A stdlib-only script with no argparse: the verb is whichever positional of
    # the argument vector it likes, and both `argv` and `sys.argv` spellings occur
    # in this tree (`agent-services/rpc/lloyd_rpc.py:243` is the real one).
    "argv-positional": ('if len(argv) < 2 or argv[0] not in ("call", "map"):\n'
                        '    return 2\n', {"call", "map"}),
    "sys-argv-positional": ('if sys.argv[1] == "board-pass":\n    pass\n'
                            'elif sys.argv[2] in ("flush", "land"):\n    pass\n',
                            {"board-pass", "flush", "land"}),
    # The same subscript shape, compared against a flag: a module whose only
    # argv comparison is `--init` takes no subcommand, so its chain is empty and
    # the module is skipped — not graded against one flag. (`agent-services/
    # services/idle-worker/check-github-releases.py:184` is the real one.)
    "argv-flag-only": ('if len(sys.argv) > 1 and sys.argv[1] == "--init":\n'
                       '    initialize_state()\n', set()),
    # The `-h, --help` line is what argparse itself puts in the command column:
    # a flag standing where a verb would be, and not a subcommand.
    "usage-string": ('USAGE = """usage: thing.py <verb> [flags]\n'
                     '\n'
                     'commands:\n'
                     '  scan [--write-baseline]\n'
                     '  deps\n'
                     '  -h, --help  show this help message and exit\n'
                     '"""\n'
                     'if command not in USAGE:\n    print(USAGE)\n', {"scan", "deps"}),
}


def test_each_dispatch_spelling_yields_its_verbs(tmp_path):
    """The resolver reads the chain however the module spells it, and reads only
    the chain: a flag's `choices` are values, not subcommands, and a `--flag`
    sitting in the usage block's command column is not a verb either.

    Each module here is one the test writes, so a spelling that stopped being
    harvested fails on its own line rather than on a doc three months from now.
    """
    for name, (src, expected) in _DISPATCH_SOURCES.items():
        p = tmp_path / f"{name}.py"
        p.write_text(src, encoding="utf-8")
        verbs, forward = _chain_of(p)
        assert verbs == expected, (
            f"{name}: harvested {sorted(verbs)}, want {sorted(expected)}")
        assert forward is None, f"{name}: a module with verbs was read as a forwarder"
    flag_only = tmp_path / "flag-only.py"
    flag_only.write_text('ap.add_argument("--format", choices=("text", "json"))\n',
                         encoding="utf-8")
    assert _chain_of(flag_only)[0] == set(), (
        "a flag's values were harvested as subcommands, so a doc citing "
        "`--format json` would be graded as if `json` were a verb")


def test_a_module_with_no_dispatch_chain_is_not_graded():
    """The one thing that keeps this check off the back of a library: a module
    that dispatches nothing has no chain to be wrong about, so nothing beside it
    is claimed as a command.

    `app/paths.py` resolves data roots and takes no subcommand at all. A doc may
    bracket a function name beside it — `architecture/authority-surfaces.md:34`
    brackets `check_bash_command` beside `app/harness/safety.py` for exactly this
    reason — and the check that graded those as broken commands would be the
    second tool on this item's list of things that report a doc wrong when it is
    right.
    """
    paths = ROOT / "app/paths.py"
    assert dispatch_verbs(paths) == set(), (
        "app/paths.py has a dispatch chain now, so this node's example needs a "
        "module that genuinely takes no subcommand")
    line = "`app/paths.py` (`data`, `roots`)\n"
    assert [c[3] for c in command_claims(line)] == ["data", "roots"], (
        "the verbs were not even extracted, so the skip below proves nothing")
    assert undispatched_claims(line) == [], undispatched_claims(line)


def test_the_command_extractor_is_what_the_check_depends_on(tmp_path):
    """Clause 3: extraction is proven before its verdicts are trusted, the way
    `test_the_path_extractor_is_what_the_check_depends_on` does for paths.

    The positive half reads the real docs and fails the moment the span grammar
    stops matching what they actually contain, or a file moves out from under a
    claim; the negative half runs over text this test writes, so it cannot be
    satisfied by the corpus going quiet.
    """
    def claims_of(path):
        text = path.read_text(encoding="utf-8", errors="replace")
        return {(path.name, ln, v) for ln, s, m, v in command_claims(text)
                if dispatch_verbs(m)}

    graded = set().union(*(claims_of(ARCH / slug) for slug in NEW_DOCS))
    assert graded, ("the command extractor found no dispatchable command in the "
                    "four graded docs, so the check over them is vacuous — the "
                    "span grammar or the module spellings went stale")
    corpus = set().union(*(claims_of(p) for p in sorted(ARCH.glob("*.md"))))
    assert graded <= corpus, (
        "the graded docs' claims vanished inside a corpus-wide run of the same "
        "extractor, which means line numbers shifted under them")
    assert ("measurement.md", 73, "fixtures") in corpus, corpus

    tmp = tmp_path / "doc.md"
    tmp.write_text("no commands here, just `app/paths.py` and `--dry-run`\n",
                   encoding="utf-8")
    assert claims_of(tmp) == set(), (
        "the extractor reported a command in a file with none, so a non-empty "
        "aggregate would not have caught it going blind")

    tmp.write_text(
        "Inline: `python -m app.harness.supply_chain run-fixtures-eval`.\n"
        "Listed: `app/harness/supply_chain.py` (`run-fixtures-eval`).\n",
        encoding="utf-8")
    found = command_claims(tmp.read_text(encoding="utf-8"))
    assert [c[3] for c in found] == ["run-fixtures-eval", "run-fixtures-eval"], found
    assert all(c[2] == ROOT / "app/harness/supply_chain.py" for c in found), found
    assert all("run-fixtures-eval" not in dispatch_verbs(c[2]) for c in found)


def test_prose_and_fragment_flags_are_never_read_as_commands():
    """Clause 4: the non-commands are not claimed, and the graded docs pass
    unchanged.

    `keep/raise/revert` is a decision; `---` and `^---$` are separators and
    regexes; `--apply`, `--user`, `--dry-run` are flags cited out of band;
    `--hf-overrides` and `--limit-mm-per-prompt {"image": 20}` carry JSON
    values; `python3` is an interpreter, `npx vite build …` is not a module of
    this tree. All of these are spans in the corpus — the triage counted 215
    command-shaped ones across the 34 top-level docs — and none may reach the
    report.
    """
    not_commands = [
        "keep/raise/revert decisions per job class",
        "---",
        "^---$",
        "|---|---|---|",
        "--apply",
        "--user",
        "--dry-run",
        "--parallel 1",
        "--hf-overrides",
        '--limit-mm-per-prompt {"image": 20}',
        "python3",
        "npx vite build -c vite.chrome.config.ts --watch",
        "--help",
        "--arms",
        "paired(rows, base=\"summary_legacy\")",
        "kickoff = false",
        "POST /api/sessions/create",
        "server.py's own `except Exception` is a bare",
        "app/harness/safety.py",
        "app/paths.py:16",
        "eval/run_compaction_recall_eval.py --arms none",
    ]
    for span in not_commands:
        assert command_claims(f"x `{span}` y\n") == [], span
        assert command_claims(f"| a | `app/paths.py` | `{span}` |\n") == [], span

    # Four lines, each isolating one rule of the extractor. Break a rule and
    # the line that needs it goes red; that is the only reason they are here.
    #
    # A snake_case span in a module's own bracket is a symbol, not a verb, and
    # a span that is not a verb ends the list — so the `scan` behind it is not
    # read either. The corpus does this for real:
    # `architecture/infrastructure.md:343` brackets `check_deployed_copies`
    # beside `scripts/service_health_check.py`, and
    # `architecture/authority-surfaces.md:34` brackets `check_bash_command`.
    symbol = "`app/harness/supply_chain.py` (`check_deployed_copies`, `scan`)\n"
    assert command_claims(symbol) == [], command_claims(symbol)
    assert undispatched_claims(symbol) == [], undispatched_claims(symbol)

    # A bracket that does not open on the module is a parenthetical, not the
    # module's list of commands.
    elsewhere = "`app/harness/supply_chain.py` runs the scan (`scan-baseline`)\n"
    assert command_claims(elsewhere) == [], command_claims(elsewhere)

    # A verb-shaped word with ordinary prose behind it is prose that began on
    # the wrong foot, not `verb --flag value`.
    chatty = "`app/harness/supply_chain.py` (`scan-baseline the tree`)\n"
    assert command_claims(chatty) == [], command_claims(chatty)

    # A span does not go looking for a module; the module is the token the verb
    # follows, or every sentence naming a file would claim a subcommand.
    looked_up = "`see app/harness/supply_chain.py scan-baseline for the arm`\n"
    assert command_claims(looked_up) == [], command_claims(looked_up)

    # And the four graded docs, unchanged.
    for slug in NEW_DOCS:
        assert undispatched_claims(_text(slug), slug) == [], slug


# ── #1965: the egress posture's owner on guard-coverage.md is a live item ─────

def _guard_coverage_lines() -> list[str]:
    return (ARCH / "guard-coverage.md").read_text(encoding="utf-8").splitlines()


def test_guard_coverage_cites_the_live_owner_of_the_egress_posture():
    """#1960 closed; a closed item is not an owner. The two ownership citations
    name #1965, the §3 row says where the posture is published, and the page's
    changelog sentence that records #1960 being *filed* is history and stays."""
    lines = _guard_coverage_lines()
    row = [ln for ln in lines if ln.startswith("| `agent_mcp/egress.py` |")]
    assert len(row) == 1, row
    assert "#1965" in row[0] and "#1960" not in row[0], row[0]
    for needle in ("egress.network_report()", "policy.enforce", "GET /api/dashboard",
                   "network.policy", "enforcing", "recording only"):
        assert needle in row[0], f"the egress row no longer names {needle!r}"

    unarmed = [ln for ln in lines if "destination table (unarmed" in ln]
    assert len(unarmed) == 1 and "#1965" in unarmed[0] and "#1960" not in unarmed[0]

    still_1960 = [ln for ln in lines if "#1960" in ln]
    assert len(still_1960) == 1 and "Filed #1959, #1960" in still_1960[0], still_1960

    from agent_mcp import egress
    import inspect
    assert '"enforce": enforce_on()' in inspect.getsource(egress.network_report)


# ── #2024: the file-gated safety-state table names real edges and real events ─

def test_guard_coverage_tabulates_every_file_gated_flag_with_both_edges():
    """One row per flag, and each event the row names is the constant the code
    appends — so a renamed event or a dropped row fails here, not in a reader."""
    import sys

    from scripts.automod import state as S
    sys.path.insert(0, str(ROOT / "agent-services" / "guardian"))
    import gstate

    text = (ARCH / "guard-coverage.md").read_text(encoding="utf-8")
    start = text.index("## File-gated safety state")
    section = text[start:text.index("\n## ", start + 1)]
    rows = {ln.split("|")[1].strip().strip("`"): ln
            for ln in section.splitlines()
            if ln.startswith("| `")}
    assert set(rows) == {"BROKEN", "pause", "promotions-halted",
                         "rollback_request.json"}, sorted(rows)
    for flag, path in (("BROKEN", S.BROKEN_PATH), ("pause", S.PAUSE_PATH),
                       ("promotions-halted", S.HALTED_PATH),
                       ("rollback_request.json", S.ROLLBACK_REQUEST_PATH)):
        assert path.name == flag, f"{flag} is not the file the code gates on"
        assert len([c for c in rows[flag].split("|") if c.strip()]) == 4, (
            f"the {flag} row lost a column (readers, create edge, remove edge)")

    expected = {
        "BROKEN": (gstate.BROKEN_SET_EVENT, S.BROKEN_CLEAR_EVENT),
        "pause": (S.PAUSE_SET_EVENT, S.PAUSE_CLEAR_EVENT),
        "promotions-halted": (S.HALT_SET_EVENT, S.HALT_CLEAR_EVENT),
        "rollback_request.json": ("rollback_requested", S.ROLLBACK_CLEAR_EVENT),
    }
    for flag, (created, removed) in expected.items():
        cells = [c for c in rows[flag].split("|") if c.strip()]
        assert f"`{created}`" in cells[2], f"{flag}: create edge names no {created}"
        assert f"`{removed}`" in cells[3], f"{flag}: remove edge names no {removed}"
    assert "rollback_requested" in Path(S.__file__).read_text(encoding="utf-8")


# ── #1963: the guard-by-path section points at the derivation, and agrees ────

def test_guard_coverage_names_the_matrix_script_and_states_what_it_prints():
    """The page links the derived set from the section a person reads, beside
    the one-gate citation — and the facts it quotes from the script's output are
    re-derived here, so the paragraph cannot outlive the tree."""
    from app.harness import guard_arm_matrix as G

    text = (ARCH / "guard-coverage.md").read_text(encoding="utf-8")
    start = text.index("## The guards, and where each one is wired")
    section = text[start:text.index("\n## ", start + 1)]
    assert "python scripts/maintenance/guard_arm_matrix.py" in section
    assert (ROOT / "scripts" / "maintenance" / "guard_arm_matrix.py").is_file()
    assert "def stale_gate_arm_points" in section, "the one-gate citation stays"
    assert section.index("def stale_gate_arm_points") < section.index(
        "scripts/maintenance/guard_arm_matrix.py")
    for guard in G.GUARDS:
        assert f"`{guard}`" in section, f"the section does not name {guard}"

    matrix = G.guard_arm_matrix(ROOT)
    assert all(matrix["app/routers/turn_options.py"].values())
    for rel in ("app/autonomy.py", "workers/sources/_common.py"):
        assert matrix[rel] == {"safety": False, "policy": True,
                               "outbound_content": True, "action_review": False}, (
            f"{rel} no longer arms what the page says it arms: {matrix[rel]}")
    assert matrix["agent_mcp/builtin_task.py"]["action_review"] is False


# --------------------------------------------------------------------------- #
# guard-coverage.md §4.5: the resolver-miss count is derived, not grep lines (#2022)
# --------------------------------------------------------------------------- #

def _guard_coverage_resolver_passage() -> tuple[str, list[str]]:
    """§4's fifth exclusion — prose, and the commands of the block that closes it."""
    section = _section(_text("guard-coverage.md"), "4. `_injection_probe`'s own exclusions")
    start = section.index("5. **Background sessions only.**")
    fence = section.index("```\n", start)
    end = section.index("```", fence + 4)
    cmds = [l for l in section[fence + 4:end].splitlines() if l.strip()]
    return " ".join(section[start:fence].split()), cmds


def _run(cmd: str) -> list[str]:
    out = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip().splitlines()


def test_the_resolver_miss_count_is_three_and_names_each_guard():
    """#2022 clause 1. "The other four" was the line count of one grep over
    `agent_mcp/main.py`: one of those lines is the probe the sentence contrasts
    with, one is the denial journal, and install provenance is not among them."""
    prose, _ = _guard_coverage_resolver_passage()
    assert "other four" not in _text("guard-coverage.md")
    assert "Three other guards" in prose
    for name in ("**desktop**", "**service control**", "**install provenance**",
                 "agent_mcp/main.py", "app/harness/service_control.py",
                 "app/harness/supply_chain.py"):
        assert name in prose, f"§4.5 does not name {name}"


def test_the_resolver_miss_block_derives_the_three_from_the_tree():
    """#2022 clause 2: the block names `app/harness/safety.py` and running it prints
    the fan-out to the two helpers; `classify_session(` is called from exactly the
    three guards' files. Every command in the block must hit."""
    _, cmds = _guard_coverage_resolver_passage()
    for cmd in cmds:
        assert cmd.startswith("git grep -n "), cmd
        assert _run(cmd), f"a proving command prints nothing: {cmd}"
    fan_out = next(c for c in cmds if "app/harness/safety.py" in c)
    lines = _run(fan_out)
    assert len(lines) == 2, lines
    assert "check_service_control(" in lines[0] and "check_install_provenance(" in lines[1]

    callers = next(c for c in cmds if "classify_session(" in c)
    calls = [l for l in _run(callers)
             if "def classify_session" not in l and "return classify_session(parent" not in l]
    assert sorted(l.split(":")[0] for l in calls) == [
        "agent_mcp/main.py", "app/harness/service_control.py",
        "app/harness/supply_chain.py"], calls


def test_the_passage_says_what_each_resolver_consumer_does_on_a_miss():
    """#2022 clauses 3 and 4: the probe keeps False, the journal is bookkeeping and
    never asks the resolver about a `task:` id, and the logging claim is bounded to
    the guards `classify_session` feeds — `_record_miss` has no other caller, so
    the probe's own miss is silent."""
    prose, cmds = _guard_coverage_resolver_passage()
    assert "keeps `False`" in prose and "one shadow row lost" in prose
    assert "bookkeeping" in prose and "never a guard" in prose
    assert "Every miss logs" not in prose
    assert "`classify_session`-fed guards" in prose and "**silent**" in prose

    order = _run(next(c for c in cmds if "denial_journal.py" in c))
    early = next(i for i, l in enumerate(order) if 'startswith("task:")' in l)
    asks = next(i for i, l in enumerate(order) if "is_background_session(" in l)
    assert early < asks, order

    misses = _run(next(c for c in cmds if "_record_miss(" in c))
    assert misses and all(l.startswith("app/harness/service_control.py:") for l in misses)
    src = (ROOT / "app/harness/service_control.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    owners = {fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
              for n in ast.walk(fn)
              if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_record_miss"}
    assert owners == {"classify_session"}, owners
    # The one caller outside `classify_session` is the read-only sandbox (#2025),
    # which logs its own miss; the probe still has none.
    elsewhere = subprocess.run(["git", "grep", "-l", "_record_miss(", "--", "agent_mcp"],
                               cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert elsewhere == ["agent_mcp/_tool_sandbox.py"], elsewhere


#: The commit that last edited `guard-coverage.md` before #2301's stamp existed
#: (#2022's landing). A stamped re-run sha must be this commit or a descendant of
#: it. The comparison is against a named sha and NOT against
#: `git log -1 -- architecture/guard-coverage.md`, because the commit that writes
#: a stamp is necessarily newer than any sha it is able to name: measuring "is
#: the stamp as new as the page" against the log's head is unsatisfiable from
#: inside the round that stamps. That is the exact defect #2301 exists to break,
#: and why the owed step after a landing is a reader re-stamping the new HEAD.
GUARD_COVERAGE_LAST_EDIT = "a0d3cbe4"


def _stamp_denominator(stamp: str) -> int:
    """The block denominator the stamp's own prose states. A re-run that counts
    nothing cannot age, so #2301 requires the number be written down."""
    m = re.search(r"\*\*(\d+) fenced blocks\*\*", " ".join(stamp.split()))
    assert m, "the stamp states no block denominator"
    return int(m.group(1))


def test_the_guard_coverage_stamp_names_a_real_commit_and_the_block_count():
    """#2022 clause 5, extended by #2301 clauses 1 and 2: the re-run line names
    `219e1314`, that sha is in this repo's history, and the page now states a
    block denominator that equals the fenced blocks actually on it — exactly 20,
    not the open-ended `>= 18` that let a page grow past its own stamp. The
    #2301 line is the first re-run stamped at a tree containing the page's own
    last edit; `a03a7300`, the newest sha the stamp carried before it, is an
    ancestor of that edit, which is what the descendant test can fail on."""
    text = _text("guard-coverage.md")
    how = _section(text, "How to read this page")
    assert "`219e1314` (2026-10-01)" in how and "all 18 blocks still hit" in how
    assert subprocess.run(["git", "cat-file", "-e", "219e1314^{commit}"], cwd=ROOT).returncode == 0
    blocks = re.findall(r"^```\n.*?^```$", text, re.S | re.M)
    assert len(blocks) == 20, f"the page carries {len(blocks)} fenced blocks"

    # Wrap-insensitive: a reflow of the paragraph must not break the extraction.
    m = re.search(r"Re-run in full on \d{4}-\d{2}-\d{2} at `([0-9a-f]{8})`",
                  " ".join(how.split()))
    assert m, "the stamp carries no #2301 full re-run line"
    stamped = m.group(1)
    assert subprocess.run(["git", "cat-file", "-e", f"{stamped}^{{commit}}"],
                          cwd=ROOT).returncode == 0, f"{stamped} is not a commit here"
    assert subprocess.run(["git", "merge-base", "--is-ancestor",
                           GUARD_COVERAGE_LAST_EDIT, stamped], cwd=ROOT).returncode == 0, (
        f"the stamp vouches for {stamped}, which predates the page's own last "
        f"edit {GUARD_COVERAGE_LAST_EDIT}")
    assert subprocess.run(["git", "merge-base", "--is-ancestor",
                           GUARD_COVERAGE_LAST_EDIT, "a03a7300"],
                          cwd=ROOT).returncode != 0, (
        "a03a7300 is now a descendant of #2022's landing, so it no longer shows "
        "what an un-refreshed stamp looks like — pick a genuinely older sha")
    assert _stamp_denominator(how) == len(blocks), (
        "the stamp counted a different number of blocks than the page carries")
    # The extractor reads the prose rather than restating a constant.
    assert _stamp_denominator("re-run at `abc12345`: **7 fenced blocks**") == 7


# --------------------------------------------------------------------------- #
# guard-coverage.md §4 opener: the family count has to print, and print the set
# the section names (#2301)
# --------------------------------------------------------------------------- #

def _guard_coverage_probe_limits() -> tuple[str, list[str]]:
    """§4's opener block's commands, plus the prose a reader brings to it: the
    sentence over the block, any comment riding a command, and exclusion 4 — the
    one whose number the block is read against. Item 4 sits *below* the block, so
    a helper that stopped at the fence would grade the command against prose that
    never mentions families at all."""
    section = _section(_text("guard-coverage.md"), "4. `_injection_probe`'s own exclusions")
    fence = section.index("```\n")
    end = section.index("```", fence + 4)
    cmds, comments = [], []
    for raw in section[fence + 4:end].splitlines():
        line = raw.strip()
        if not line:
            continue
        code, mark, tail = line.partition("  # ")
        assert mark or "#" not in code, f"an unquoted '#' in a command: {line}"
        cmds.append(code.strip())
        comments.append(tail.strip())
    assert cmds, "§4's opener block is empty — nothing was extracted"
    i4 = section.index("Eight regex families")
    i5 = section.index("Background sessions only")
    prose = " ".join(section[:fence].split() + comments + section[i4:i5].split())
    return prose, cmds


def test_the_family_count_command_prints_and_names_the_set_it_counts():
    """#2301 clause 3. `FAMILIES` moved to `agent_mcp/_injection_patterns.py` in
    #1959 and arrived dict-shaped, so the page's `'^    ("'` grep — the tuple
    shape the probe's own table used — matched nothing and `git grep -c` printed
    nothing and exited 1. A command this page presents as "the family count" was
    silent, and a stamp that only asks whether blocks *hit* cannot see a
    command/output mismatch. The command now counts the shared table's dict keys
    (13: the probe's 8 plus the gate's 6, `invisible_chars` read by both), and
    the prose says the eight belong to `PROBE_FAMILIES`, so the two numbers
    cannot be read as contradicting each other."""
    prose, cmds = _guard_coverage_probe_limits()
    family = next(c for c in cmds if "re.compile" in c)
    assert "agent_mcp/_injection_patterns.py" in family, family
    out = _run(family)
    assert out, f"the family-count command prints nothing: {family}"
    counts = {l.rsplit(":", 1)[0]: int(l.rsplit(":", 1)[1]) for l in out}
    assert counts == {"agent_mcp/_injection_patterns.py": 13}, counts

    ids = set(re.findall(r'"([a-z_]+)"',
                         " ".join(_run("git grep -A4 'PROBE_FAMILIES: tuple' "
                                      "-- agent_mcp/_injection_probe.py"))))
    assert len(ids) == 8, ids
    assert "Eight regex families" in prose, "§4 stopped stating the probe's count"
    assert "PROBE_FAMILIES" in prose and "thirteen" in prose, (
        "the prose the command sits under must name which set the 13 counts, or "
        "the command reads as contradicting 'Eight regex families'")
    for cmd in cmds:
        assert cmd.startswith("git grep "), cmd
        assert _run(cmd), f"a proving command prints nothing: {cmd}"


#: The three docs #2329 reworded, and the recovery-command shape they now name. A
#: fixed historical set, typed here rather than harvested from the docs: a node that
#: read the commands out of the prose it is grading would pass on docs that named
#: none, which is exactly the state the round started in.
RECOVERY_DOCS = ("autonomy-jobs.md", "arch-review.md", "index.md")
RECOVERY_CMD = re.compile(r"`git show ([0-9a-f]{7,40})\^:(architecture/[^`]*)`")


def _git_lines(*args: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), *args],
                         capture_output=True, text=True)
    assert out.returncode == 0, f"git {' '.join(args)} exited {out.returncode}"
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def test_every_archive_recovery_command_the_docs_name_resolves():
    """#2329 clause 5. Each of the three docs now tells a reader how to get a
    retired doc back, and a recovery command that does not resolve is a second dead
    pointer wearing the clothes of a fix — the failure this whole item exists to
    close, since the sentence it replaced pointed at a directory that is not in the
    tree.

    So every backticked `git show <rev>^:architecture/…` the three docs carry is
    run, not admired. A `<slug>` template is expanded over the roster git itself
    records for that retirement — the copies tracked at that commit's parent for
    the `.archive/` one, the files that commit deleted from `architecture/` for the
    other — so the check proves the command works for every doc it is offered for,
    not just for the one name someone happened to try.
    """
    found: dict[tuple[str, str], None] = {}
    for slug in RECOVERY_DOCS:
        found.update({m: None for m in RECOVERY_CMD.findall(_text(slug))})
    assert len(found) >= 3, (
        f"only {len(found)} recovery command(s) named across {RECOVERY_DOCS}; the "
        "docs are supposed to name a way back for each retirement, so a thin "
        "harvest means the pointers went away rather than getting fixed")
    archive_roster = [Path(p).stem for p in
                      _git_lines("ls-tree", "--name-only", "f80c9d00^",
                                 "architecture/.archive/")]
    # Stems, not paths: a doc offers the command as
    # `git show b94be171^:architecture/<slug>.md`, so the thing being substituted
    # into `<slug>` is the doc name with its directory and extension already taken
    # off — which is also how the Retired section lists the same eleven.
    deleted_roster = [Path(ln.split("\t")[1]).stem for ln in
                      _git_lines("show", "--name-status", "--format=", "b94be171")
                      if ln.startswith("D\tarchitecture/")]
    assert len(archive_roster) == 5, archive_roster
    assert len(deleted_roster) == 11, deleted_roster
    for sha, path in found:
        if "<slug>" not in path:
            targets = [path]
        elif ".archive/" in path:
            targets = [path.replace("<slug>", s) for s in archive_roster]
        else:
            targets = [path.replace("<slug>", s) for s in deleted_roster]
        for target in targets:
            probe = subprocess.run(
                ["git", "cat-file", "-e", f"{sha}^:{target}"],
                cwd=ROOT, capture_output=True)
            assert probe.returncode == 0, (
                f"a doc names `git show {sha}^:{target}` and it does not resolve: "
                f"{probe.stderr.strip() or 'no such path at that rev'}")


def test_the_reworded_docs_gain_no_unresolvable_path_citation():
    """#2329 clause 2's mechanical half. These three docs are in the corpus the
    citation check above grades, and its exemption list is `git_ignored()` — which
    is precisely why `architecture/.archive/` never went red there even while it was
    being asserted as current (`.gitignore` carries `/architecture/.archive/`). That
    exemption must not become a licence for the fix.

    Measured as a delta against `ad6ab9f4`, the commit this round started from: a
    reworded doc may cite fewer resolvable paths than it did (the dead ones are the
    point of the change) and may never cite more, so a sentence that sends a reader
    at a file which does not exist fails here whatever `.gitignore` says about the
    directory it mentions.
    """
    base = "ad6ab9f4"
    for slug in RECOVERY_DOCS:
        before = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:architecture/{slug}"],
                                capture_output=True, text=True)
        assert before.returncode == 0, f"{base}:architecture/{slug} does not exist"
        unres_now = {str(p) for p in unresolved_citations(cited_paths(_text(slug)))}
        unres_base = {str(p) for p in unresolved_citations(cited_paths(before.stdout))}
        assert len(unres_now) <= len(unres_base), (
            f"architecture/{slug} cites {len(unres_now)} unresolvable paths against "
            f"{len(unres_base)} at {base}: the reword added a pointer instead of "
            f"removing one — new: {sorted(unres_now - unres_base)}")


def test_arch_review_states_the_archive_directory_is_absent_here():
    """#2329 clause 2. The opening paragraph used to say 17 docs were retired, "of
    which **12 are still in the gitignored `.archive/`**" — present tense, and the
    box has no such directory (`ls -d architecture/.archive` is a `No such file`
    error), so the sentence sent a reader to a path that is not there while the
    number in it was already stale: only five retired copies were ever tracked, and
    the commit the same paragraph cites for deleting five is the one that untracked
    all five.

    Pinned against git's own record rather than against a second copy of the
    paragraph: the five `D` lines `f80c9d00` carries under `.archive/` are the
    denominator the sentence now states, so the prose cannot drift from the commit
    it names without this failing.
    """
    text = _text("arch-review.md")
    flat = _flat(text)
    assert "still in the gitignored" not in flat, (
        "the paragraph asserts retired docs are still sitting in .archive/, which "
        "is not in this checkout and holds nothing tracked")
    assert "are still in" not in flat, (
        "some sentence in arch-review.md again puts the retired docs in a place a "
        "reader can open")
    assert "does not exist here" in flat, (
        "the paragraph no longer says outright that architecture/.archive/ is "
        "absent, which is the fact a reader needs before the recovery commands")
    assert "`f80c9d00` is the commit that untracked all five" in flat, (
        "the paragraph must name which commit ended the tracked copies, or the "
        "five it counts have no owner and the count is folklore")
    deletions = [ln for ln in _git_lines("show", "--name-status", "--format=",
                                         "f80c9d00")
                 if ln.startswith("D\tarchitecture/.archive/")]
    assert len(deletions) == 5, (
        f"f80c9d00 carries {len(deletions)} deletions under architecture/.archive/, "
        "not the five the paragraph counts")
    for command in ("`git show f80c9d00^:architecture/.archive/<slug>.md`",
                    "`git show b94be171^:architecture/<slug>.md`"):
        assert command in text, (
            f"arch-review.md no longer offers {command}; saying the copies are gone "
            "without a way back leaves the reader exactly where the old sentence did")


#: #2362 clause 5: the rail's own page has to say how many surfaces it holds. The
#: heading and the bold sentence beneath it are the two places a reader looks, and
#: the lane census is the part that stops the next reader re-filing `memory_add`
#: as an open door: that tool's `file` argument is grammar-bound and cannot name a
#: task file, while Write/Edit and a Bash child genuinely can.
_AUTONOMY_SURFACE_HEADING = (
    "### The dispatch-affecting fields, and the three surfaces that write them")
_GRAMMAR_BOUND = "MEMORY.md|USER.md|topics/<slug>"


def test_the_dispatch_fields_page_names_every_surface_the_rail_holds():
    """source: architecture/autonomy.md § The dispatch-affecting fields.
    claim: the heading says "three surfaces", the sentence under it names
           `vault_write`, and the page records that `Write`/`Edit` and a Bash child
           stay open while `memory_add` is not a lane at all.
    verdict: #2362 closed the third door and rewrote both places; the census is
             what keeps "the rail holds them all" from meaning "nothing else can
             write a task file", which is not true and was never claimed.

    The count in the heading is load-bearing in a way a normal heading is not: a
    page that says "two surfaces" is a page that tells the next reader there are
    exactly two, and the two it named are the two that were already shut.
    """
    text = (ARCH / "autonomy.md").read_text(encoding="utf-8")
    assert _AUTONOMY_SURFACE_HEADING in text, (
        "the heading still counts two surfaces, so the page re-opens the question "
        "#2362 closed by answering it")
    assert "and the two surfaces that write them" not in text, (
        "a second heading or cross-reference still claims two surfaces")

    heading = text.index(_AUTONOMY_SURFACE_HEADING)
    paragraph = text[heading:heading + 1600]
    assert "`vault_write`" in paragraph, (
        "the sentence under the heading must name the surface #2362 added, not "
        "just raise the number in the heading")
    assert "autonomy_write_task" in paragraph and "vault-round landing route" in paragraph

    census = text[text.index("**What the rail does not reach"):]
    assert "`Write`/`Edit`" in census, census[:200]
    assert "agent_mcp/builtin_fs.py" in census, (
        "the open lane must be named by the module that carries no autonomy guard")
    assert "_path_sandbox.py" in census and "PROTECTED_WRITE_ROOTS" in census, (
        "the Bash child is open because its read-only bind list omits the autonomy "
        "dir; the census has to say which list, or the next reader re-derives it")
    assert _GRAMMAR_BOUND in census, (
        "naming memory_add as an open lane is the error this clause exists to "
        "stop; the census must show why it cannot reach a task file")
    assert "cannot name a task file" in census, census[:200]


def test_the_tools_page_points_at_the_renamed_heading():
    """source: architecture/tools.md § the dispatch guard cross-reference.
    claim: tools.md's pointer to the autonomy page says two more surfaces and
           quotes the heading as it now reads.
    verdict: #2362 renamed the heading, so a page that quotes its old wording
             points at a section that does not exist.

    A cross-reference is graded here rather than left to a human because it fails
    silently: nothing in the vault breaks, and the reader who follows it finds a
    heading three surfaces away from the one the pointer promised.
    """
    text = (ARCH / "tools.md").read_text(encoding="utf-8")
    start = text.index("`autonomy_write_task` call that moves")
    window = text[start:start + 900]
    assert "the three\n   surfaces that write them" in window, (
        "the pointer still quotes the two-surfaces heading that no longer exists")
    assert "vault_write" in window and "(#2362)" in window, (
        "the pointer still says one more surface than the tool, not two")


# --------------------------------------------------------------------------- #
# guard-coverage.md §3: the sentence under the probe's wiring grep names the
# files that grep prints (#2392)
# --------------------------------------------------------------------------- #

#: The fenced command §3 offers as the answer to "where is the probe wired?", and
#: the command the sentence under it cites for the one-call-site claim. Both are
#: typed here rather than read back out of the page: a node that extracted its own
#: needle would stay green on a page that carried neither, which is the vacuity
#: this file's header forbids. Verbatim, including the quoting — `git` reads a
#: differently-quoted pathspec as a different pattern.
PROBE_WIRING_GREP = "git grep -ln \"_injection_probe\" -- \"*.py\" ':!tests/*'"
PROBE_CALLSITE_GREP = 'git grep -n "_injection_probe.apply(" -- agent_mcp/main.py'

#: The three files `PROBE_WIRING_GREP` prints, with the role each mention plays.
#: A file set, not a line set: `agent_mcp/main.py` carries three mentions of one
#: wiring (the import, the `PROBED_TOOLS` gate and the `apply(` call), so a
#: line-count comparison would fail on a reflow of `main.py` and pass on a second
#: wiring.
PROBE_WIRING_FILES = {
    "agent_mcp/main.py": ("dispatch", "call site"),
    "agent_mcp/_injection_patterns.py": ("docstring", "cross-reference"),
    "app/harness/guard_arm_matrix.py": ("prose",),
}

_CODE_SPAN = re.compile(r"`([^`]*)`")
_SHELL_COMMAND = re.compile(
    r"\A\s*(?:git|grep|rg|sed|awk|ls|cat|find|python|pytest|bash|sh)\b")
_CITED_PY_PATH = re.compile(r"\b[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-/]*\.py\b")


def _section_3_wiring_paragraph() -> str:
    """§3's prose, from just under its wiring grep to the next blank line.

    Wrap-normalised for the same reason the stamp node normalises its stamp line:
    a reflow of the paragraph is not a change of claim, and an extractor that
    broke on one would be grading line lengths instead of the sentence.
    """
    text = _text("guard-coverage.md")
    assert PROBE_WIRING_GREP in text, (
        "§3 no longer carries the fenced `git grep -ln \"_injection_probe\"` block "
        "this group grades — the nodes below would be reading a paragraph that is "
        "not the one the page presents as its wiring answer")
    close = text.index("```", text.index(PROBE_WIRING_GREP))
    body = text[close + 3:].lstrip("\n")
    paragraph = body.split("\n\n", 1)[0]
    assert paragraph.strip(), "nothing follows the §3 wiring fence"
    return " ".join(paragraph.split())


def _files_named_by(paragraph: str) -> set[str]:
    """Repo-relative `.py` paths a paragraph cites, command spans excluded.

    A span opening with a shell verb is a *command*, not a filename, and is
    dropped before the scan. Without that rule the `agent_mcp/main.py` riding
    inside the cited `git grep -n "_injection_probe.apply(" -- agent_mcp/main.py`
    counts as a citation, and the set check would hold on prose that named no file
    at all — the exact page this item is fixing, plus a command.
    """
    prose = _CODE_SPAN.sub(
        lambda m: "" if _SHELL_COMMAND.match(m.group(1)) else m.group(1), paragraph)
    return set(_CITED_PY_PATH.findall(prose))


def _clause_about(paragraph: str, path: str) -> str:
    """The stretch of prose a reader attributes to one cited path: from that path
    up to the next path the paragraph cites."""
    i = paragraph.index(path)
    later = [paragraph.find(p, i + len(path)) for p in PROBE_WIRING_FILES]
    stops = [j for j in later if j > i]
    return paragraph[i + len(path):min(stops) if stops else len(paragraph)]


def test_guard_coverage_section_3_names_every_file_the_wiring_grep_prints():
    """source: architecture/guard-coverage.md §3, the paragraph under the
    `_injection_probe` wiring grep.
    claim: it names all three files that grep prints and says what each mention
           is — `agent_mcp/main.py` the dispatch wiring,
           `agent_mcp/_injection_patterns.py` a docstring cross-reference from the
           shared pattern table, `app/harness/guard_arm_matrix.py` a prose mention
           in the guard-arm matrix — and no longer says the answer names one file.
    verdict: #1959 moved `FAMILIES` into `agent_mcp/_injection_patterns.py` and
             #1963 added the guard-arm matrix, so "the only file" went false on the
             page while the suite stayed green; the fix is naming, because none of
             the three is a second wiring.

    The role words are graded with the paths because a bare list of three files is
    a different false claim: it reads as three places that screen content, and
    sends a reader looking for two more probes.
    """
    paragraph = _section_3_wiring_paragraph()
    assert "only file that answer names" not in _text("guard-coverage.md"), (
        "§3 still claims the wiring grep names one file, which has been false "
        "since #1959 and #1963 added the other two mentions")
    assert _files_named_by(paragraph) == set(PROBE_WIRING_FILES), paragraph

    assert len(PROBE_WIRING_FILES) == 3, (
        "the expected set stopped being three files, so this node is no longer "
        "grading the mismatch #2392 is about")
    for path, roles in PROBE_WIRING_FILES.items():
        clause = _clause_about(paragraph, path)
        assert clause.strip(), f"§3 cites {path} with no prose around it"
        for role in roles:
            assert role in clause, (
                f"§3 cites {path} without saying what the mention is ({role!r} "
                f"missing) — an unlabelled mention is what makes a docstring read "
                f"like a second wiring")


def test_guard_coverage_section_3_keeps_the_one_call_site_and_cites_its_command():
    """source: architecture/guard-coverage.md §3, same paragraph.
    claim: the one-tool-dispatch-call-site claim survives the correction, and it
           cites `git grep -n "_injection_probe.apply(" -- agent_mcp/main.py`
           rather than a hand-typed line number.
    verdict: the claim was never the problem — only the word "file" was. The
             command prints exactly one line at HEAD, which is the whole content of
             the claim, and a typed `:714` would rot on the next edit to `main.py`
             while this node stayed green.
    """
    paragraph = _section_3_wiring_paragraph()
    assert PROBE_CALLSITE_GREP in paragraph, (
        "§3 no longer cites the command that establishes the one-call-site claim; "
        "a line number typed in its place moves with every edit to main.py")
    assert "tool-dispatch call site" in paragraph, (
        "the paragraph stopped asserting the single call site, which is the claim "
        "#2392 exists to preserve while fixing the file count")
    assert not re.search(r"main\.py:\d+", paragraph), (
        "§3 pins a hand-typed line number for the dispatch site; cite the "
        "command instead")

    printed = _run(PROBE_CALLSITE_GREP)
    assert len(printed) == 1, (
        f"the call-site claim is false, not the wording: the command printed "
        f"{len(printed)} lines — {printed}")
    assert printed[0].startswith("agent_mcp/main.py:"), printed


def test_the_files_section_3_names_are_the_files_the_wiring_grep_prints():
    """source: architecture/guard-coverage.md §3, same paragraph.
    claim: the set of source files the paragraph names equals the set
           `git grep -ln "_injection_probe" -- "*.py" ':!tests/*'` prints.
    verdict: this is the rung that was missing. Three earlier rounds touched the
             page or the symbol (#1948, #1959, #1963) and none compared the
             sentence to its own command, so the page carried a stale count under a
             green suite; from here a fourth mention — a new guard module naming
             the probe — reddens the suite instead of quietly widening the set.

    Compared as file sets, never line sets: `main.py` carries three mentions of
    the one wiring, so lines would count reflows rather than guardrails.
    """
    printed = _run(PROBE_WIRING_GREP)
    assert printed, (
        f"{PROBE_WIRING_GREP} printed nothing, so there is no set to compare and "
        f"an equality against an empty set would prove nothing")
    assert _files_named_by(_section_3_wiring_paragraph()) == set(printed), (
        "the page's §3 paragraph and its own wiring grep disagree about which "
        "files mention the probe")

    # The check can fail, and fails for both directions of drift — on text this
    # file writes, not on the page.
    stale = "The only file that answer names is `agent_mcp/main.py`."
    assert _files_named_by(stale) == {"agent_mcp/main.py"}, (
        "the extractor found nothing in the stale sentence, so it cannot fail on "
        "the very wording this item replaced")
    grown = stale + " Also `app/harness/a_new_guard.py` now mentions it."
    assert "app/harness/a_new_guard.py" in _files_named_by(grown), (
        "a file added to the prose is invisible to the extractor")
    command_only = "Run `git grep -ln x -- agent_mcp/main.py` instead."
    assert _files_named_by(command_only) == set(), (
        "a filename inside a cited command is being read as a citation, which is "
        "how the set check would pass on prose that names no file")
