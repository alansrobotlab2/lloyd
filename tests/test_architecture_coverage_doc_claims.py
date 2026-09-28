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
  * paths resolve against the working tree, not `git ls-files` — the extension's
    `manifest.json` is untracked, and a tracked-files check reports a real file
    as missing. `$LLOYD_DATA/...` and `~/...` resolve under the data root and
    the home directory, because the scored store moved there on 2026-09-22
    (`architecture/data-home.md`) and a repo-rooted check would call it absent.
"""

from __future__ import annotations

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


@pytest.mark.parametrize("slug", NEW_DOCS)
def test_every_path_a_new_doc_cites_resolves_in_the_working_tree(slug):
    cited = cited_paths(_text(slug))
    assert cited, f"{slug} cites no path at all — the doc has stopped being checkable"
    missing = [str(p) for p, _ in cited if not p.exists()]
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
      fault_injection"` collect the same 23 nodes, so nothing in this file was
      moved out of the gate's reach;
    - the run reports **22 PASSED lines**, one per node collected less this one, so
      a skipped or xfailed node cannot hide inside a passing exit code;
    - the node #1763 files — `test_every_path_a_new_doc_cites_resolves_in_the_working_tree`
      `[authority-surfaces.md]` — is among those 22 and is reported PASSED **by
      name**, which is the verdict this item was filed for;
    - exit code 0, and no `skipped`/`xfailed` anywhere in the output.

    This node is the one exclusion, and it is a recursion guard, not a dodge: a node
    that runs its own file runs itself again — measured at 298 live pytest processes
    before it was bounded. The arithmetic carries that honestly (23 collected, 22
    reported), and this node's own verdict is what the gate's full-suite run records
    with no exclusion in it.
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
