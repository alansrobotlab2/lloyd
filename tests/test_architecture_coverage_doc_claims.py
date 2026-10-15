"""#1699 pins the architecture docs that made the unreviewed areas review
units to the tree they inventory.

`workers/sources/arch_review.py::doc_slugs` globs top-level
`architecture/*.md`, so a doc is what turns an area into something the
cadence reads at all — an undocumented area cannot be reached by `groups:`,
because `parse_groups` only accepts a `<doc-slug>:<section>` pair inside a doc
that already exists. The areas left without one were the eval/bench surface, the context
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
    missing; and a bare citation resolves under *every* root this box has — the
    checkout, the vault, the data root and the account home — because the docs
    inventory all four. `$LLOYD_DATA/...` and `~/...` name their own root and
    keep it: the scored store moved out of the tree on 2026-09-22
    (`architecture/data-home.md`), a repo-rooted check called it absent, and a
    check that let a `~/` spelling fall back into the repo would green a
    citation whose whole point is that it is not in the repo. The two roots the
    gate moves (`$HOME`, and `LLOYD_DATA` for the rungs that execute candidate
    code) are read through `app.data_root`'s passwd-derived names as well, so a
    real path is not reported missing because the gate redirected its root.

A *command* is a citation of the same kind, and it needed an extractor of its
own: `_CITATION` only fires on a span carrying a slash, so `run-fixtures-eval` —
the verb `architecture/measurement.md` once asserted for
`app/harness/supply_chain.py`, which never dispatched it — extracted as nothing,
and the check above stayed green while an operator following the doc got a usage
dump and exit 1 (#1792). The command half is at the bottom of this file and owes
the same three proofs.

**Three rulings about what is graded and where a citation lands, shipped here
because they are the reason the file is shaped as it is (#1890, #2582).**

*The command half grades every top-level `architecture/*.md`.* A wrong *verb*
fails the same way wherever it is written: `workers/sources/arch_review.py`
makes every doc a review unit, and a command claim in a doc nobody reads sends
an operator to the same usage dump and exit 1. Grading only the docs of #1699
was prevention that protected a couple of the corpus's command spans and left
the rest ungraded, and the one report the corpus-wide sweep produced
(`harness.md:1573`) was in an ungraded doc. So the command sweep reads the glob.

*So does the path half (#2582), and it always should have.* A doc's payload is
mostly path citations, and `test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree`
is parameterized over the same `ARCH.glob("*.md")` the command sweep and
`workers/sources/arch_review.py::doc_slugs` read — not over `NEW_DOCS`, which
was the "protected a named few of the corpus" shape the paragraph above was
written against, kept for paths. The cost is the same one and is the point: a
doc written tomorrow is graded the moment it lands. `ARCH_DOC_SURFACE` is the
vacuity guard for both halves — the glob's own count has to equal it, so a
sweep over "every doc" cannot mean a sweep over nothing — and
`test_the_path_surface_is_the_whole_corpus_the_vacuity_number_holds` pins that
for the path node while the command nodes pin it for theirs.

What a widened path check finds on a corpus written before it existed is
`PATH_CITATION_EXEMPTIONS`: per-doc citation spellings that are allowed not to
be on disk, each with the reason. The anti-rot in that list is mechanical, not a
reader — `test_every_path_citation_exemption_is_a_citation_the_doc_actually_makes`
refuses an entry no doc cites (a stale exemption, or one invented to swallow
future rot), so the only way to grow the list is to write the sentence that
needs it.

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
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architecture"
EVAL = ROOT / "eval"
HOME = Path.home()

#: The roots a bare architecture citation may legitimately land in, in the order
#: the check tries them. The docs inventory this machine, and this machine keeps
#: four trees: the checkout, the vault, the runtime data root and the account
#: home. Reading only the checkout is what made #2582's 31 ungraded docs safe to
#: write — and unsafe to read: a citation of `lloyd/SOUL.md`, `_pipeline/` or
#: `.config/systemd/user/` is true on this box and false under `ROOT`, so an
#: extractor that asked only the repo could never be widened without drowning in
#: reports that are the extractor's, not the doc's.
#:
#: The vault and data roots are `app.data_root`'s passwd-derived names, not
#: `Path.home()`'s. Inside an automod gate `HOME` is the round's symlink farm and
#: `LLOYD_DATA` is the round's own empty root (`scripts/automod/gate.py::_child_env`,
#: `scripts/automod/worktree.py::HOME_LINK_SKIP`), so `Path.home()/"lloyd-data"`
#: there is a directory holding nothing — where production's `_pipeline/` and
#: `sessions/` are real. A check that read the moved root off `$HOME` would go
#: red at the gate for a citation that is true, which is the failure mode
#: `app/paths.py`'s header exists to prevent.
def _passwd_root(name: str, fallback: Path) -> tuple[Path, str]:
    try:
        from app.data_root import ACCOUNT_HOME
    except Exception:                          # pragma: no cover - stdlib-only
        return fallback, "Path.home() (app.data_root unavailable)"
    return ACCOUNT_HOME / name, f"app.data_root.ACCOUNT_HOME/{name}"


ACCOUNT_HOME, DATA_ROOT, VAULT_ROOT = (
    _passwd_root("", HOME)[0], _passwd_root("lloyd-data", HOME / "lloyd-data")[0],
    _passwd_root("obsidian", HOME / "obsidian")[0])

CITATION_ROOTS: tuple[tuple[str, Path], ...] = (
    ("checkout", ROOT),
    ("vault", VAULT_ROOT),
    ("data", DATA_ROOT),
    ("home", ACCOUNT_HOME),
)

#: The data root the running process was pointed at, honoured beside the
#: production one so a `$LLOYD_DATA/...` citation resolves wherever `LLOYD_DATA`
#: says the data lives (a round's root, a canary's, a `tmp_path` in a unit test).
ENV_DATA_ROOT = Path(os.environ["LLOYD_DATA"]) if os.environ.get("LLOYD_DATA") else None

#: The docs this file pins as units, beyond the corpus-wide command sweep.
#: Named explicitly rather than glob-fresh: a new doc elsewhere must not
#: silently join the surface this file grades.
#: `threat-model.md` joined on #2563: a doc whose entire payload is a table of
#: path citations is the strongest case in the corpus for having those citations
#: read against the tree rather than re-read by a human, and it is graded here
#: rather than by a checker of its own so the rule has one owner.
NEW_DOCS = ("measurement.md", "context-window.md", "authority-surfaces.md",
            "browser-side-panel.md", "threat-model.md")

#: How many top-level architecture docs exist. The command sweep reads the glob,
#: not this number; the number is the vacuity guard, because a `glob("*.md")`
#: that found no corpus — a moved directory, a wrong `ROOT` — would otherwise let
#: a sweep over "every doc" pass by sweeping nothing. A doc landing here moves
#: the number, and that is the sweep admitting its new subject, which is the
#: #1890 ruling: every doc is graded, so every doc has to be in the surface.
ARCH_DOC_SURFACE = 36   # + threat-model.md (#2563, 2026-10-11)

#: The heading every doc in `NEW_DOCS` carries, in the words clause 3 asks for.
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
    """Where a citation with this prefix points, in the tree the check is running in.

    `~/` is `ACCOUNT_HOME`, the passwd-derived name, and not `Path.home()`. A
    doc writing `~/lloyd-data/workers.db` means the account's home — the one place
    that spelling has ever pointed on this box — and inside an automod gate
    `Path.home()` is `<round>/home`, a symlink farm that deliberately carries
    neither `lloyd` nor `lloyd-data` (`worktree.py::HOME_LINK_SKIP`). Expanding
    the tilde against *that* home turns a true citation into a missing file: the
    gate's `tests` rung went red on nine architecture docs for exactly this, and
    the red was reported against rounds that had touched none of them. It is the
    same rule the four roots above already follow, stated at
    `app/paths.py`'s header — a reader that wants live data reaches it through the
    passwd entry, never through `$HOME`.
    """
    if prefix == "$LLOYD_DATA/":
        return (ENV_DATA_ROOT or DATA_ROOT or ACCOUNT_HOME / "lloyd-data") / rel
    if prefix == "~/":
        return ACCOUNT_HOME / rel
    return ROOT / rel


def _rel_within(path: Path) -> Path | None:
    """`path` relative to whichever known root it sits under, or None.

    The inverse of `_resolve`, needed because `unresolved_citations` is handed
    resolved paths rather than the spans they came from: to ask the other roots
    the same question it has to recover the relative spelling first. The data
    root the process was pointed at is tried before the checkout, so a
    `$LLOYD_DATA/...` citation aimed at a round's or a fixture's root still
    yields its bare relative name.

    `HOME` comes before `ACCOUNT_HOME` because the two nest: inside a gate the
    round's home is `<ACCOUNT_HOME>/lloyd-work/<round>/home`, so a path under it
    is lexically under the account home as well, and whichever is tried first
    decides the answer. Tried last, the account home swallows it and the relative
    name comes back carrying the whole nesting — `lloyd-work/<round>/home/lloyd-data/workers.db`
    — which every root then fails to find, for a file that is sitting in plain
    sight in the data root. Most specific root first is the only order that can
    recover the spelling the doc actually wrote.
    """
    for base in (ENV_DATA_ROOT, ROOT, DATA_ROOT, VAULT_ROOT, HOME, ACCOUNT_HOME):
        if base is None:
            continue
        try:
            return Path(path).relative_to(base)
        except ValueError:
            continue
    return None


def citation_roots(path: Path) -> list[Path]:
    """Every path a citation that resolved to `path` could equally mean.

    A doc citing `lloyd/SOUL.md` means the vault's, and one citing `_pipeline/`
    means the data root's; neither spelling names its root, so the honest
    question is "does this relative path exist anywhere this machine keeps
    files", not "is it under the checkout". A path sitting under no known root
    comes back alone, so a citation to somebody else's tree is still reported
    against the one place it was read as.
    """
    rel = _rel_within(path)
    if rel is None:
        return [Path(path)]
    out = [base / rel for _, base in CITATION_ROOTS]
    if ENV_DATA_ROOT is not None:
        out.insert(0, ENV_DATA_ROOT / rel)
    out.append(Path(path))
    return list(dict.fromkeys(out))


def resolve_citation(path: Path) -> Path | None:
    """The first root that actually holds `path`, or None if none does."""
    for cand in citation_roots(path):
        if cand.exists():
            return cand
    return None


def cited_citations(text: str) -> list[tuple[str, Path, int | None]]:
    """`(span as written, resolved path, line anchor)` for every citation.

    The span is kept because the exemption list is keyed on what the doc wrote,
    not on the path the extractor built from it: an entry has to be checkable
    against the doc, and `lloyd/SOUL.md` is the citation a reader can find,
    whereas `/home/…/lloyd/SOUL.md` is an artefact of which root won.
    """
    out = []
    for span in _CODE_SPAN.findall(text):
        m = _CITATION.match(span.strip())
        if not m or "/" not in m.group("path"):
            continue
        out.append((span.strip(),
                    _resolve(m.group("pre"), m.group("path")),
                    int(m.group("line")) if m.group("line") else None))
    return out


def cited_paths(text: str) -> list[tuple[Path, int | None]]:
    """Every code-span citation in `text`, resolved. Repo-relative against the
    working tree — deliberately NOT `git ls-files`, because a citation has to
    resolve where the reader is standing and some real ones are untracked;
    `$LLOYD_DATA/...` against `~/lloyd-data`, the root the mover moved them to;
    `~/...` against home. Whether it is *there* is a separate question and it
    asks every root: see `citation_roots`."""
    return [(p, line) for _span, p, line in cited_citations(text)]


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
    """Cited paths that exist under no root the box has and are not git-ignored."""
    absent = [Path(p) for p, _ in cited if resolve_citation(p) is None]
    ignored = git_ignored(absent)
    return [p for p in absent if p not in ignored]


#: Every top-level architecture doc, read the way the cadence reads it:
#: `ARCH.glob("*.md")`, the same glob `workers/sources/arch_review.py::doc_slugs`
#: uses to decide what is a review unit at all. The path half is parameterized
#: over this and not over `NEW_DOCS` (#2582): a citation is a citation wherever
#: it is written, and the citations in the docs outside `NEW_DOCS` were read by
#: no node at all.
PATH_DOC_SLUGS = [p.name for p in sorted(ARCH.glob("*.md"))]

#: Citations allowed not to be on disk, keyed by the doc that writes them and by
#: the span exactly as written, with the reason the sentence is honest despite the
#: missing file. Four kinds earn a place here, and the reason has to name which:
#:
#:   * `generated:` runtime state the box writes under one of the roots — absent
#:     in a fresh worktree, before the job that makes it has run, or in an
#:     untracked production directory like `agent-services/cert/`;
#:   * `historical:` the sentence is dated or says itself that the path is gone,
#:     moved or absent ("no `archived/` directory of any kind", a `git show
#:     <sha>^:` recovery, a file-name *pattern*), so it is a record, not a pointer;
#:   * `other-tree:` the path is in a checkout the extractor cannot see — the
#:     `qmd/` fork, an upstream package's own module layout, a different box —
#:     including the sub-path case, where the sentence names a directory
#:     (`agent-services/`) and lists its internals relative to it;
#:   * `stale:` the path moved or was retired *in this repo* and the sentence
#:     still points at the old spelling. This is the one kind that is rot: it is
#:     here because clearing it is a doc edit with its own investigation, not a
#:     one-line change, which is what the owed-check on #2582 exists to watch.
#:
#: What this list must never become is a place to park rot: it is a report with a
#: date, and `test_every_path_citation_exemption_is_a_citation_the_doc_actually_makes`
#: refuses an entry no doc cites, so the only way to add one is to write the
#: sentence that needs it. The owed-check on #2582 rules on whether it shrinks.
PATH_CITATION_EXEMPTIONS: dict[str, dict[str, str]] = {
 'ambient-context-injection.md': {
  '~/lloyd-data/next-session-notes.json':
   ('generated: `app/next_session_notes.py` creates the store on its '
    'first note (`_save` writes it), and `_read` reads a missing file '
    'as an empty channel by design, so a box holding no pending note '
    'shows only the sibling `.lock` that `locked_file` takes for every '
    'read — the location the row names is the one the module writes'),
 },
 'arch-review.md': {
  '.archive/':
   ('historical: the page quotes the `.gitignore` rule'
    '`/architecture/.archive/` and the directory as it was; this is'
    'the subject of `test_index_md_sends_a_reader_to_history_rather'
    '_than_the_absent_archive`'),
  'agent_mcp/memory_ops.py:12':
   ('stale: the module was retired in ed78a3a9 with eleven other'
    'tools, and the worked example still cites its old shape'),
  'arch-review-conflicts/':
   ("generated: the gate's per-round conflict reports, written"
    'under the runtime root by a run and swept with the round'),
  'pending-research/':
   ('generated: the vault staging directory the research worker'
    'fills while a sweep is in flight and empties when the note'
    'lands'),
 },
 'automod.md': {
  '.venvs/':
   ('generated: interpreter venvs the install scripts build, never'
    'committed, so no worktree has them'),
  'app/x.py:164':
   ("historical: the invented path of the section's worked example,"
    "the same shape as this file's own `app/paths.py:999999`"
    'control'),
  'dist/':
   ("other-tree: the qmd fork's build output, named as the"
    'directory the daemon serves; the fork sits at `~/lloyd/qmd`,'
    'which is gitignored and in no worktree'),
  'graphify-out/':
   ("generated: graphify's own output directory, written by a hand"
    'run or a cadence job and left unanchored on purpose'
    '(automod.md:2088)'),
 },
 'autonomy-jobs.md': {
  '_archived/':
   ("historical: the vault task directory's archive subdirectory,"
    "which the page's point is that a non-recursive `glob` cannot"
    'see'),
  'agents/':
   ('generated: a scan target under the vault that exists only once'
    'a subagent profile has landed there'),
  'archived/':
   ("historical: the sentence's own claim is that no `archived/`"
    'directory exists under `~/obsidian/backlog/`, which is the'
    'finding being reported'),
  'memory/reflection/morning-briefing-latest.md':
   ('historical: the paragraph says plainly that this documented'
    'output exists nowhere and the job keeps no durable record'),
  'proposed/':
   ('generated: the staging directory the research-queue job'
    'creates when it has a candidate to propose'),
  'references/prefs.yaml':
   ('other-tree: a path inside the skill directory the sentence'
    'names, which the extractor roots only at the four box roots'),
  'scripts/groundskeeper/groundskeeper-survey.py':
   ('stale: retired in 201901b4 with its queue writer and both'
    'unwired consumers, which is #1012'),
  'scripts/memory/batch-process-orphans.py':
   ('stale: retired in the same 201901b4 sweep that took the survey'
    'and its queue writer (#1012)'),
  'scripts/memory/process-groundskeeper-queue.py':
   ('stale: retired in the same 201901b4 sweep that took the survey'
    'and its queue writer (#1012)'),
  'scripts/state.json':
   ('other-tree: a path inside the skill directory the sentence'
    "names; the job's own state lives under its runtime root"),
  '~/.local/state/lloyd-automod/landed_outcomes.jsonl':
   ('generated: the row #2572\'s daily record appends is created by '
    'its first `--record` run, and #97 had run 0 times when #2630 '
    'wrote the entry — `select count(*) from runs where task_id=97` '
    'was 0 on 2026-10-12, and the doc says in the same breath that '
    'the file does not exist yet. The one citation here whose '
    'absence is the fact being reported, not a pointer to a file'),
  '~/obsidian/autonomy/85-secondary-routing-eval.md':
   ('stale: a dated finding names the task file by a number the'
    'task no longer carries; the skill of that name is at'
    '`~/obsidian/skills/secondary-routing-eval`'),
  '~/obsidian/memory/facts/':
   ('stale: the fact-markdown directory the extraction used to'
    'write; the store now lives in `kg.sqlite`, so the layout is'
    'retired'),
 },
 'autonomy.md': {
  '_archived/':
   ("historical: the vault task directory's archive subdirectory,"
    "which the page's point is that a non-recursive `glob` cannot"
    'see'),
  '~/obsidian/autonomy/NN-slug.md':
   ('historical: the file-name pattern the page defines for a task'
    'file, with `NN` a placeholder rather than a path'),
 },
 'background-runs.md': {
  'sessions-background/':
   ('generated: background-run transcripts under the runtime root,'
    'created by the first concurrent run and bounded by the'
    'retention sweep'),
  'workers/handlers/ambient.py':
   ("stale: the ambient handler's old module path; the tool is"
    'dispatched from `agent_mcp/ambient.py`, which is what the row'
    'should say'),
 },
 'backlog.md': {
  '~/.openclaw/':
   ('historical: the sentence states that neither `~/.openclaw/`'
    'nor its database exists on this machine any more'),
  '~/.openclaw/data/backlog.db':
   ('historical: the predecessor store the page is describing as'
    'migrated away, not a live pointer'),
  '~/Development/clawdeck/':
   ('other-tree: a checkout on a different box that the page names'
    'as the predecessor project; it is not on this machine and'
    'never in a worktree'),
 },
 'data-home.md': {
  '.venvs/':
   ('generated: interpreter venvs the install scripts build, never'
    'committed, so no worktree has them'),
  'desktop/':
   ('generated: the desktop-lease directory under the runtime root,'
    'created the first time the desktop tool grants a lease'),
  'qmd/':
   ("other-tree: the qmd fork's checkout in the production tree,"
    'which `.gitignore:155` excludes, so no worktree contains it'),
 },
 'desktop.md': {
  'eval/desktop_grounding/':
   ("historical: the page's own finding is that the directory does"
    'not exist, which is why the section could not cite an outcome'),
  'tools/computer_use/':
   ('other-tree: a directory of the upstream hermes-agent project'
    'the row names, whose tree the extractor cannot see'),
  '~/lloyd-data/desktop/lease.json':
   ('generated: the lease file the desktop tool writes when it'
    'first grants one; the directory is not there before the'
    'feature runs'),
 },
 'djev.md': {
  'server/structured_server.py':
   ('other-tree: a file of the upstream djev-spark repository the'
    'same row names, not of this checkout'),
 },
 'editing-safeguards.md': {
  'app/inner_voice/heuristics.py':
   ('stale: retired in 4da219fa with the ensemble and grading'
    'machinery it belonged to; the sentence predates the thin-'
    'observer rewrite'),
 },
 'guard-coverage.md': {
  '.venvs/':
   ('generated: interpreter venvs the install scripts build, never'
    'committed, so no worktree has them'),
 },
 'harness.md': {
  'entrypoints/chat_utils.py:2000':
   ('other-tree: a module inside the upstream vLLM package whose'
    "line the page cites; that layout is not this repo's"),
 },
 'index.md': {
  '.archive/':
   ('historical: the page quotes the `.gitignore` rule'
    '`/architecture/.archive/` and the directory as it was; this is'
    'the subject of `test_index_md_sends_a_reader_to_history_rather'
    '_than_the_absent_archive`'),
  'architecture/.archive/':
   ("historical: the archive rule #2582 corrected in this page's"
    'own preamble: the name is a `.gitignore` pattern and the'
    'directory is absent'),
 },
 'infrastructure.md': {
  'agent-services/autonomy.service':
   ('stale: the unit is at `agent-'
    'services/systemd/autonomy.service`; the row drops the'
    'directory and so names a path the tree has never had'),
  'agent-services/cert/':
   ('generated: TLS material `scripts/gen-cert.sh` writes into the'
    "production tree; a worktree has no copy of the machine's"
    'secrets'),
  'bin/start-livekit-server.sh':
   ('other-tree: a sub-path of `agent-services/`, the directory the'
    'section is about, so the real file is `agent-services/bin/...`'),
  'bin/start-obsidian-sync.sh':
   ('other-tree: a sub-path of `agent-services/`, the directory the'
    'section is about, so the real file is `agent-services/bin/...`'),
  'bin/start-qwen3-tts.sh':
   ('other-tree: a sub-path of `agent-services/`, the directory the'
    'section is about, so the real file is `agent-services/bin/...`'),
  'bin/start-qwen3-tts-omni.sh':
   ('other-tree: a sub-path of `agent-services/`, the directory the'
    'section is about, so the real file is `agent-services/bin/...`'
    '; #2697 added the row for the omni TTS'),
  'bin/start-qwen38-flash-next.sh':
   ('other-tree: a sub-path of `agent-services/`, the directory the'
    'section is about, so the real file is `agent-services/bin/...`'),
  'bin/start-secondary.sh':
   ('other-tree: a sub-path of `agent-services/`, the directory the'
    'section is about, so the real file is `agent-services/bin/...`'),
  'conf.d/':
   ('other-tree: a sub-path of `agent-services/`, whose supervisor'
    'drop-in directory the sentence is describing'),
  'conf/livekit.yaml':
   ('other-tree: a sub-path of `agent-services/`, so the file the'
    'row means is `agent-services/conf/livekit.yaml`'),
  'scripts/qmd-watcher.sh':
   ('other-tree: a sub-path of `agent-services/`, so the file is'
    '`agent-services/scripts/qmd-watcher.sh`'),
  'starlette/middleware/cors.py:91':
   ('other-tree: a module of the starlette distribution installed in this'
    'interpreter\'s site-packages, which no citation root covers; the line'
    'is the middleware\'s inline preflight test, and'
    '`tests/test_api_client_gating.py` resolves the spelling this page and'
    'the gate comment cite'),
 },
 'knowledge-graph.md': {
  'facts-rebuild/':
   ('generated: the staging directory a fact-store rebuild writes'
    'before it swaps the new store in, and is removed after'),
  'vLLM/vLLM-state.md':
   ('historical: a vault note the rebuild collapsed into another'
    'entity; the collision count carried beside it is a dated'
    'figure'),
 },
 'memory.md': {
  'agent_mcp/memory_ops.py':
   ('stale: the module was retired in ed78a3a9 with eleven other'
    'duplicate or subsumed tools'),
 },
 'mission-control.md': {
  'agent-services/cert/':
   ('generated: TLS material `scripts/gen-cert.sh` writes into the'
    "production tree; a worktree has no copy of the machine's"
    'secrets'),
 },
 'qmd.md': {
  '$LLOYD_DATA/_pipeline/reflection/qmd-stray-keep-list.json':
   ('generated: the keep-list the qmd stray sweep writes under the'
    'runtime root when it next runs'),
  '$LLOYD_DATA/_pipeline/reflection/qmd-stray-release-list.json':
   ('generated: the optional hand-written release list beside the'
    'keep-list; `read_release_list` reads an absent file as no'
    'additional entries by design, and the seed that is always in'
    'force lives in the `RELEASED_STRAYS` constant, so the file'
    'exists only once a person writes a ruling into it'),
  'dist/':
   ("other-tree: the qmd fork's build output, named as the"
    'directory the daemon serves; the fork sits at `~/lloyd/qmd`,'
    'which is gitignored and in no worktree'),
  'dist/cli/qmd.js':
   ("other-tree: the qmd fork's built entry point, named in the"
    'same sentence as the fork at `~/lloyd/qmd`'),
  'src/':
   ("other-tree: the qmd fork's TypeScript sources, named in the"
    'same sentence as the fork at `~/lloyd/qmd`'),
  'src/llm.ts':
   ("other-tree: the qmd fork's model resolver, named in the same"
    'sentence as the fork at `~/lloyd/qmd`'),
  'src/mcp/server.ts':
   ("other-tree: the qmd fork's REST layer, named in the same"
    'sentence as the fork at `~/lloyd/qmd`'),
  'src/store.ts':
   ("other-tree: the qmd fork's indexing layer, named in the same"
    'sentence as the fork at `~/lloyd/qmd`'),
  'src/vecindex.ts':
   ("other-tree: the qmd fork's vector index, named in the same"
    'sentence as the fork at `~/lloyd/qmd`'),
 },
 'research-pipeline.md': {
  'pending-research/domain-research/':
   ('generated: the vault staging directory the research worker'
    'fills while a domain sweep is in flight, empty between runs'),
  '~/obsidian/lloyd/research-queue.md':
   ('stale: the queue lives at `~/obsidian/research-queue.md` (with'
    '`lloyd/research-queue-archive.md` beside it), so the'
    "sentence's directory segment is stale"),
 },
 'retrieval.md': {
  'architecture/recall-research-2026-09-24.md':
   ("historical: deleted in 4a6cdd54; index.md's Retired section"
    'reads it back with `git show`, which is what the citation'
    'means'),
 },
 'skills.md': {
  '.archived/':
   ('generated: the quarantine directory a skill is renamed into,'
    'present only while a skill is actually parked there'),
  'arxiv/scripts/':
   ('other-tree: a supporting path inside a skill directory under'
    'the vault, which the sentence names rather than spells out'),
  'autolink/autolink.py':
   ('other-tree: a supporting path inside a skill directory under'
    'the vault, which the sentence names rather than spells out'),
  'github-pr-workflow/references/':
   ('other-tree: a supporting path inside a skill directory under'
    'the vault, which the sentence names rather than spells out'),
  '~/.npm-global/lib/node_modules/openclaw/skills/':
   ("historical: the retired openclaw install's skill root, which"
    'the page lists as a location the migration left'),
  '~/.openclaw/skills/':
   ("historical: the retired openclaw install's skill root, listed"
    'as a location the migration away from it left behind'),
  '~/lloyd/skills/':
   ('historical: the pre-migration skills root; the library moved'
    'to `~/obsidian/skills/`, which is why the row lists it as a'
    'stale location'),
 },
 'subliminal.md': {
  'agent_mcp/subliminal.py':
   ('stale: the module was retired in f5789b71 by the'
    'prefetch/subliminal rework that moved the machinery elsewhere'),
  'architecture/recall-research-2026-09-24.md':
   ("historical: deleted in 4a6cdd54; index.md's Retired section"
    'reads it back with `git show`, which is what the citation'
    'means'),
  'sessions-background/':
   ('generated: background-run transcripts under the runtime root,'
    'created by the first concurrent run and bounded by the'
    'retention sweep'),
  'vault-derived/sessions-background/':
   ('generated: a directory of the nightly derived index under the'
    'vault, rebuilt by the consolidation job and absent before its'
    'first run'),
  'vault-derived/sessions/':
   ('generated: a directory of the nightly derived index under the'
    'vault, rebuilt by the consolidation job and absent before its'
    'first run'),
  '~/lloyd/agent_mcp/subliminal.py':
   ('stale: the same f5789b71 retirement, spelled from the account'
    'home by the sentence that names the checkout'),
  '~/lloyd/skills/':
   ('historical: the pre-migration skills root; the library moved'
    'to `~/obsidian/skills/`, which is why the row lists it as a'
    'stale location'),
 },
 'testing.md': {
  '.venvs/':
   ('generated: interpreter venvs the install scripts build, never'
    'committed, so no worktree has them'),
  'agent-services/cert/':
   ('generated: TLS material `scripts/gen-cert.sh` writes into the'
    "production tree; a worktree has no copy of the machine's"
    'secrets'),
  'qmd/':
   ("other-tree: the qmd fork's checkout in the production tree,"
    'which `.gitignore:155` excludes, so no worktree contains it'),
 },
 'tools.md': {
  'graphify-out/':
   ("generated: graphify's own output directory, written by a hand"
    'run or a cadence job and left unanchored on purpose'
    '(automod.md:2088)'),
 },
 'vllm.md': {
  '.archive/':
   ('historical: the page quotes the `.gitignore` rule'
    '`/architecture/.archive/` and the directory as it was; this is'
    'the subject of `test_index_md_sends_a_reader_to_history_rather'
    '_than_the_absent_archive`'),
  'v1/core/kv_cache_utils.py':
   ('other-tree: a module inside the upstream vLLM package the'
    'paragraph is describing, not a path in this repository'),
  '~/.config/systemd/user/nvidia-power-limit.service':
   ('generated: the installed copy of `agent-'
    'services/systemd/nvidia-power-limit.service`, which the'
    'install step links into the user unit directory when it runs'),
 },
 'voice.md': {
  'agent-services/services/tts/qwen3-tts/':
   ('generated: an untracked service checkout in the production'
    'tree, so a worktree has never contained it'),
  'api/backends/optimized_backend.py':
   ('other-tree: a module of the upstream vllm-omni project the'
    'paragraph names, whose tree the extractor cannot see'),
  'bin/':
   ('other-tree: a sub-path of `agent-services/`, the directory the'
    'section enumerates'),
  'listen/decide/':
   ('other-tree: a sub-path of the voice service directory the'
    'section enumerates, not of the repository root'),
  'models/':
   ('other-tree: a sub-path of the voice service directory the'
    'section enumerates, holding downloaded weights that are never'
    'committed'),
  'qwen3-tts/models/':
   ('other-tree: a sub-path of the untracked qwen3-tts service'
    'checkout, holding downloaded weights that are never committed'),
  'resemblyzer/audio.py':
   ('other-tree: a module of the upstream resemblyzer package the'
    'sentence names, whose tree the extractor cannot see'),
  'scripts/ww_replay.py':
   ('other-tree: a sub-path of `agent-services/`, so the file meant'
    'is `agent-services/scripts/ww_replay.py`'),
  'stop/':
   ('other-tree: a sub-path of the voice service directory the'
    'section enumerates'),
  'voice/addressee.py':
   ('other-tree: a sub-path of `agent-services/`, so the file meant'
    'is `agent-services/voice/addressee.py`'),
  'voice/thinking.py':
   ('other-tree: a sub-path of `agent-services/`, so the file meant'
    'is `agent-services/voice/thinking.py`'),
  'voice/timeline.py':
   ('other-tree: a sub-path of `agent-services/`, so the file meant'
    'is `agent-services/voice/timeline.py`'),
  'voice/wake.py':
   ('other-tree: a sub-path of `agent-services/`, so the file meant'
    'is `agent-services/voice/wake.py`'),
  'voice_library/profiles/dave_cullen/':
   ('generated: a cloned voice profile produced by a sample run, so'
    'it exists only on a box where that profile was built'),
  '~/Projects/lloyd-services/':
   ("other-tree: a checkout on the machine this page's history"
    'describes, not on this box and never in a worktree'),
 },
 'workers-jobs.md': {
  'distill/':
   ("historical: the sentence's finding is that the staging"
    'directory does not exist, from a layout the job no longer uses'),
  'pending-research/':
   ('generated: the vault staging directory the research worker'
    'fills while a sweep is in flight and empties when the note'
    'lands'),
  'pending-research/domain-research/':
   ('generated: the vault staging directory the research worker'
    'fills while a domain sweep is in flight, empty between runs'),
 },
 'workers.md': {
  'pending-research/':
   ('generated: the vault staging directory the research worker'
    'fills while a sweep is in flight and empties when the note'
    'lands'),
 },
}


def _resolved_or_none(path: Path) -> Path | None:
    return resolve_citation(path)


@pytest.mark.parametrize("slug", PATH_DOC_SLUGS)
def test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree(slug):
    """Clause 1 (#2582): the path check reads the corpus, not a named few.

    Existence is asked of every root the box has (`citation_roots`), because the
    corpus cites the vault, the data root and the account home as plainly as it
    cites the checkout — see the header. A citation still absent everywhere is
    legitimate only if `PATH_CITATION_EXEMPTIONS` carries the doc's reason for
    it, and a line anchor is checked against the file the citation actually
    resolved to, not against the checkout's guess at it.
    """
    cited = cited_citations(_text(slug))
    assert cited, f"{slug} cites no path at all — the doc has stopped being checkable"
    exempt = PATH_CITATION_EXEMPTIONS.get(slug, {})
    resolved = []
    for _span, path, line in cited:
        hit = _resolved_or_none(path)
        if hit is None:
            continue
        resolved.append(hit)
        if line is not None:
            n_lines = len(hit.read_text(encoding="utf-8", errors="replace").splitlines())
            assert n_lines >= line, (
                f"{slug} cites {hit}:{line}, but that file has {n_lines} lines — a "
                "line anchor past the end is a pointer to nothing")
    assert resolved, (
        f"{slug} cites {len(cited)} paths and not one of them exists under any root "
        "the box has — a doc whose every citation is exempt has stopped being "
        "checkable, which is the failure the exemption list exists to be unable to hide")
    absent = [pth for _s, pth, _l in cited if _resolved_or_none(pth) is None]
    ignored = git_ignored(absent)              # one batched call for the whole doc
    missing = sorted({span for span, path, _line in cited
                      if _resolved_or_none(path) is None
                      and path not in ignored and span not in exempt})
    assert not missing, (
        f"{slug} cites paths that exist under none of the "
        f"{len(CITATION_ROOTS)} roots, are not git-ignored, and carry no exemption "
        f"reason: {missing}")


def test_the_path_extractor_is_what_the_check_depends_on():
    """Negative control over deliberately wrong text: one real path, one
    invented one, one line number past the end. If the extractor could not tell
    these apart, `test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree`
    would pass on a doc full of fiction."""
    real = "app/paths.py"
    text = ("see `app/paths/no_such_module_really.py`, then `$LLOYD_DATA/eval/nope/`, "
            f"then `{real}:100` and `{real}:999999`, written over "
            "https://example.com/app/paths.py, with keep/raise/revert decisions and "
            "a bare `stats.py` and a rule over `web/src/**`")
    cited = cited_paths(text)
    paths = [p for p, _ in cited]
    assert ROOT / real in paths, "the extractor missed a path that is plainly there"
    # The data root a run is pointed at is not always `~/lloyd-data`: conftest
    # gives the suite a scratch one, and a gate round points `LLOYD_DATA` at the
    # round's. The assertion is that the prefix became a path under *that* root,
    # which is the behaviour the resolver owes, not which directory it names.
    data_span = [q for q in paths if q.name == "nope" and "eval" in q.parts]
    assert data_span, (
        "a $LLOYD_DATA citation was not extracted, so the data-root half of the "
        "resolution check has nothing to resolve")
    assert all(q.parent == (ENV_DATA_ROOT or DATA_ROOT) / "eval" for q in data_span), (
        f"the $LLOYD_DATA citation resolved outside the data root the run names "
        f"({ENV_DATA_ROOT or DATA_ROOT}): {data_span}")
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


# ── #2582: the path surface is the corpus, and a citation lands in any root ───

def test_the_path_surface_is_the_whole_corpus_the_vacuity_number_holds():
    """Clause 1's other half: the widened node's id-set is the corpus, and the
    corpus has a size.

    `test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree` is
    parameterized over `PATH_DOC_SLUGS`, which is the glob's own output — so if
    the glob ever finds nothing (a moved `architecture/`, a wrong `ROOT`, a
    worktree cut before the docs landed) pytest collects no id of that node and
    the widening passes by grading nothing. A parametrize over a glob needs a
    count beside it: this is the vacuity guard the #1890 command nodes have
    always had, now over the path half too.
    """
    on_disk = [p.name for p in sorted(ARCH.glob("*.md"))]
    assert on_disk == PATH_DOC_SLUGS, (
        "the node's parametrised id-set is no longer the glob it claims to read")
    assert len(PATH_DOC_SLUGS) == ARCH_DOC_SURFACE, (
        f"the glob found {len(PATH_DOC_SLUGS)} top-level architecture docs, want "
        f"{ARCH_DOC_SURFACE}: either a doc landed (the path node grades it now, so "
        f"move the constant up) or this file is reading the wrong tree")
    assert set(NEW_DOCS) <= set(PATH_DOC_SLUGS), (
        "a doc in NEW_DOCS is not in the corpus-wide glob, so the path node the "
        "#1699 nodes were written beside has quietly narrowed")


#: One real citation per root the path check is supposed to consult, drawn from
#: sentences the corpus already writes, plus one invented path that has to stay
#: unresolvable. The relative spellings are the point: every example but the
#: checkout row is a path `ROOT` does not have, so this node fails on a resolver
#: that went back to asking only the repo — the state #2582 was filed about, and
#: the state a `citation_roots` that stopped consulting the other roots silently
#: returns to.
ROOT_PROOFS: tuple[tuple[str, str], ...] = (
    ("checkout", "app/harness/safety.py"),      # in the tree under test
    ("vault", "lloyd/SOUL.md"),                 # the vault's identity file
    ("data", "_pipeline"),                      # the runtime store's own root
    ("home", ".config/systemd/user"),           # units outside every tree
)

#: Invented, and must stay missing under every root.
FICTIONAL_CITATION = "app/harness/no_such_root_proof_zzz.py"


def test_a_cited_path_resolves_under_every_root_the_box_has_not_just_the_repo():
    """Clause 2: the root map, pinned with one real citation per root and the
    fiction case beside it.

    `architecture/authority-surfaces.md` cites the vault's `lloyd/SOUL.md` and
    `architecture/data-home.md` cites `_pipeline/`; both were invisible to the old
    resolver, which asked only `ROOT` for a spelling that names no root. That is
    why the widening had to come with a root map rather than with the git-ignore
    exemption #2582 first proposed: `git check-ignore` answers rc=1 for `sessions`,
    `_pipeline` and `lloyd/MEMORY.md` — they are not ignored, they are in another
    tree, which is the one thing `git` was never asked about.

    Asserted per root, and asserted *against the checkout* for the three rows that
    are not it: a proof that resolved under `ROOT` as well would pass on a
    repo-only resolver and prove nothing, which is the trap
    `test_the_identity_file_is_still_cited_and_lands_outside_the_repo` names for
    the vault row.
    """
    named = dict(CITATION_ROOTS)
    assert [label for label, _ in CITATION_ROOTS] == ["checkout", "vault", "data", "home"]
    for label, rel in ROOT_PROOFS:
        assert (named[label] / rel).exists(), (
            f"the {label} root proof {named[label] / rel} is not on disk, so this "
            "node is grading a root it cannot see")
        hit = resolve_citation(ROOT / rel)
        assert hit is not None, (
            f"a citation of `{rel}` resolved under no root: the widened check has "
            f"lost the {label} root, which is the root that makes the corpus's "
            f"sentence about it true — `{rel}` is not under the checkout")
        if label != "checkout":
            assert not (ROOT / rel).exists(), (
                f"the {label} proof now resolves inside the checkout too, so this "
                "node can no longer tell a root-map resolver from a repo-only one "
                f"— move the proof to something only the {label} root has")
    assert resolve_citation(ROOT / FICTIONAL_CITATION) is None, (
        f"the invented `{FICTIONAL_CITATION}` resolved, so the widened check "
        "accepts fiction from some root the map is too generous about")


def test_the_root_map_covers_the_roots_and_prices_what_that_gives_up():
    """The shape of the map, so a later narrowing cannot hide inside a green.

    `citation_roots` is the one place the widened check decides where a citation
    may live, so its surface is pinned: the four roots of clause 2, the data root
    the process was pointed at when there is one, and the original path for a
    citation under no known root (which must come back unchanged rather than get
    re-rooted onto the box — a doc citing somebody else's tree is a finding, not
    a path to relocate).

    The priced cost, stated here because it is a real trade and not a bug: an
    unprefixed spelling that exists under two roots counts as resolved on the
    first of them, and a `~/`-spelled fiction whose relative name happens to exist
    in the checkout is not reported. It is the same case the git-ignore exemption
    already gives up (see `git_ignored`), and the fiction case stays covered by
    `test_the_path_extractor_is_what_the_check_depends_on`.
    """
    rel = "app/paths.py"
    roots = citation_roots(ROOT / rel)
    for label, base in CITATION_ROOTS:
        assert base / rel in roots, (
            f"citation_roots no longer offers the {label} root: {roots}")
    untouched = citation_roots(Path("/nowhere/on/this/box/app/x.py"))
    assert untouched == [Path("/nowhere/on/this/box/app/x.py")], (
        f"a path under no known root got re-rooted onto the box: {untouched}")
    assert resolve_citation(Path("/nowhere/on/this/box/app/x.py")) is None


def test_the_module_header_states_the_surface_it_grades_without_a_count():
    """Clause 5: the header says what the path half grades, and no hand-typed
    count of graded docs survives to rot.

    `architecture/threat-model.md`'s false parenthetical — "as for every other doc
    in this directory", over a five-name parametrize — was a *prose* claim about
    the surface, and it survived because the only account of the surface a reader
    gets is the header. A sentence that names a number is a second definition of
    the glob, and #2563's finding is that a second definition drifts the moment a
    doc lands without a test forcing the sentence to move. So the header names the
    glob and the guard constant by their real names, and a spelled-out or typed
    count in front of "docs" is what this node refuses.
    """
    header = __doc__
    assert "ARCH.glob(\"*.md\")" in header, (
        "the header no longer says the path half reads the corpus-wide glob")
    assert "ARCH_DOC_SURFACE" in header, (
        "the header no longer names the vacuity guard that keeps a sweep over "
        "\"every doc\" from meaning a sweep over nothing")
    counted = re.findall(
        r"(?i)\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirty|thirty-four|thirty-six)\s+(?:top-level\s+)?(?:architecture\s+)?docs?\b",
        header)
    assert counted == [], (
        f"the header types a count of graded docs ({counted}) instead of naming "
        "the glob — a number here is a second definition of the surface and the "
        "first thing to go stale when a doc lands")


def test_every_path_citation_exemption_is_a_citation_the_doc_actually_makes():
    """Clause 3's anti-growth rail. An exemption list read only by the check it
    exempts is how a report quietly becomes a permanent excuse: nothing in the
    widened node notices an entry whose sentence was deleted, so an unpoliced list
    outlives its reason and swallows the next author's rot.

    So each entry is checked against the doc it is filed under — the span has to
    be one that doc still cites, and the reason has to name one of the three kinds
    `PATH_CITATION_EXEMPTIONS`'s comment admits — and any doc that exempts
    anything has to have a citation that resolves, which is the same "a doc that
    cites nothing is an all-clear" rule the widened node enforces from the inside.
    The scale of the list is asserted too: exemptions are the corpus's debt, and a
    list reaching a quarter of every citation in it is not a report anybody will
    clear but a second, undocumented surface.
    """
    assert PATH_CITATION_EXEMPTIONS, (
        "the exemption list is empty while the path node reads the whole corpus. "
        "An empty list passing here and a corpus passing there are the same green "
        "for two different reasons — if the corpus really is clean, delete the "
        "list, this node and the exemption clause in the widened node together")
    for slug, entries in PATH_CITATION_EXEMPTIONS.items():
        assert slug in PATH_DOC_SLUGS, (
            f"{slug} is exempted but is not a top-level architecture doc: an entry "
            "that survives its doc is an exemption with nothing left to excuse")
        assert entries, f"{slug} maps to no exemption at all — drop the key"
        cited = {span for span, _p, _l in cited_citations(_text(slug))}
        for span, reason in entries.items():
            assert span in cited, (
                f"{slug} is exempted for `{span}` but does not cite it: the "
                "sentence went and the excuse stayed, which is how this list would "
                "come to cover a citation nobody wrote yet")
            kind = reason.split(":", 1)[0].strip()
            assert kind in ("generated", "historical", "other-tree", "stale"), (
                f"{slug}: the `{span}` exemption reasons with {reason!r}, which "
                "does not open with one of the four kinds the list admits "
                "(generated / historical / other-tree / stale) — a fifth kind is "
                "a decision nobody has reviewed")
            assert len(reason.split(":", 1)[-1].strip()) > 30, (
                f"{slug}: the `{span}` exemption reason {reason!r} names a kind "
                "but no mechanism — say what about this box makes the sentence "
                "true anyway")
        assert [p for _s, p, _l in cited_citations(_text(slug))
                if resolve_citation(p) is not None], (
            f"{slug} exempts every citation it makes, so nothing about it is "
            "checkable any more")
    total = sum(len(v) for v in PATH_CITATION_EXEMPTIONS.values())
    every_citation = sum(len(cited_citations(_text(slug))) for slug in PATH_DOC_SLUGS)
    assert total <= every_citation // 4, (
        f"{total} exemptions against {every_citation} citations in the corpus: the "
        "widening has stopped reporting rot and started documenting an absence, "
        "and the honest move is a doc fix or a filed item, not another entry")


# ── #2606: a collected count a doc states has to be re-measurable ────────────

#: The paragraph carrying the page's collected counts, found by its own marker and
#: not by a heading. A review log holds dated counts by the dozen; the one that
#: reads as current is the Status paragraph's, and `Re-measured` is written exactly
#: once in the page (asserted below), so the extractor cannot be grading some
#: older entry's numbers and calling the page current.
_REMEASURED_PARA = re.compile(r"^Re-measured .*?(?=\n\n)", re.S | re.M)

#: ``<code>tests/x.py</code> 93`` — a count is claimable only in that shape: a file
#: span immediately followed by its number. That is what makes the pair checkable
#: by a node rather than re-read by a human, and it is why the page's prose
#: ("collects 93 in `tests/test_prefetch.py`", the number behind the file) is
#: deliberately not a claim this reads.
_COLLECTED_PAIR = re.compile(r"`(?P<f>tests/[a-z0-9_]+\.py)`\s+(?P<n>\d+)")

#: The group total, in prose, parenthesised so it cannot be mistaken for a pair.
_COLLECTED_GROUP_TOTAL = re.compile(r"\((\d+) across those four\)")

#: The command the counts have to name. #2606's own page published "44 green … 73
#: in `test_prefetch.py`" with no command beside it, and no reader could say what
#: had been run that made 73 true. This is #2301/#2391's ruling for
#: guard-coverage's fenced blocks — a stated denominator, re-runnable — applied to
#: a test count instead of a block count.
COLLECT_COMMAND = "--collect-only"

#: The file whose ambient cases sit outside the four-file queue/resolver group.
PREFETCH_FILE = "tests/test_prefetch.py"


def _remeasured_paragraph(text: str) -> str:
    found = _REMEASURED_PARA.findall(text)
    assert len(found) == 1, (
        f"{len(found)} `Re-measured …` paragraphs on the page; the extractor reads "
        "one, so a second is a set of current counts nobody grades")
    return found[0]


def _collected_claims(para: str) -> dict[str, int]:
    claims = {m.group("f"): int(m.group("n"))
              for m in _COLLECTED_PAIR.finditer(para)}
    assert claims, (
        "no `file.py  N` pair came out of the Re-measured paragraph — that is this "
        "extractor failing, not the page having nothing to check")
    return claims


def collected_node_counts(files) -> dict[str, int]:
    """Nodes each file collects, from one batched `--collect-only -q` run.

    Batched because the per-file answer is the same either way — a node id names
    its own file before the first `::` — and one collection costs a fraction of
    what five cost. This is the command the page names, run as written; what it
    prints is the only number the page is allowed to state.
    """
    files = sorted(set(files))
    ran = _pytest(["--collect-only", "-q", *files])
    assert ran.returncode == 0, f"collecting {files} exited {ran.returncode}\n" \
                                + (ran.stdout + ran.stderr)[-900:]
    counts = dict.fromkeys(files, 0)
    for line in ran.stdout.splitlines():
        if "::" not in line:
            continue
        origin = line.strip().split("::", 1)[0]
        if origin in counts:
            counts[origin] += 1
    assert sum(counts.values()) > 0, f"no node ids attributed to {files}: {ran.stdout[:400]}"
    return counts


def _stale_collected_counts(claims: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Which claims the named command does not bear out, as (stated, actual)."""
    actual = collected_node_counts(claims)
    return {f: (n, actual[f]) for f, n in claims.items() if actual[f] != n}


def test_the_ambient_page_states_the_command_its_collected_counts_came_from():
    """#2606 clause 4: every collected count the ambient page states as current is
    what `--collect-only -q` collects for that file today, the command is named
    beside the numbers, the group total adds up, and the node that reads the
    page's own citations is named somewhere on the page.

    What was published instead was "17 / 5 / 16 / 6 = 44 green across the four
    files that pin this page, 73 in `test_prefetch.py`" — a four-file selection
    reported as though it were the pinning surface, over a number the same command
    contradicts (93), and it went out an hour before the file it had not run
    reported this page's citations red. A count with no command beside it can only
    be aged by the author who ran it, which is the failure #2606 names and this
    node prices in the only currency that ages by itself: re-running the command.

    The counts are *collected*, never passing — a passing count is a verdict with a
    date on it, and the whole item is a doc edit that reported the wrong verdict.
    """
    text = _text("ambient-context-injection.md")
    para = _remeasured_paragraph(text)
    assert COLLECT_COMMAND in para, (
        "the Re-measured paragraph states counts without the command that produces "
        "them — the shape #2606 was filed about, and the one no later reader can "
        "check")
    stale = _stale_collected_counts(_collected_claims(para))
    assert stale == {}, (
        f"the page states counts the command does not bear out (stated, actual): {stale}")
    claims = _collected_claims(para)
    group = {f: n for f, n in claims.items() if f != PREFETCH_FILE}
    assert len(group) == 4, f"the total's group is {sorted(group)}, not four files"
    total = _COLLECTED_GROUP_TOTAL.search(para)
    assert total, "the page states no group total, so its own arithmetic is unreadable"
    assert int(total.group(1)) == sum(group.values()), (
        f"the paragraph says {total.group(1)} across those four while the four it "
        f"names collect {sum(group.values())}: {group}")
    assert ("test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree"
            "[ambient-context-injection.md]") in text, (
        "the page no longer names the node that reads its own citations, so a "
        "reader cannot tell which selection a green claim covered")


def test_a_collected_count_the_command_contradicts_is_what_the_node_flags():
    """The same helper, aimed at the sentence #2606 was filed about: the published
    claim comes back flagged, with the stated number beside the measured one.

    Without this the node above could be green because `collected_node_counts`
    counts nothing at all — the failure every extractor in this file is written
    against. 73 is in the fixture because that is what the page published; the
    assertion is the mismatch, and the measured side of it is re-read here rather
    than quoted, so the node cannot go stale by outliving 93.
    """
    published = ("Re-measured 2026-10-12: the four files that pin this page are "
                 "green, and `tests/test_prefetch.py` 73.\n\n")
    claims = _collected_claims(published)
    assert claims == {PREFETCH_FILE: 73}, claims
    stale = _stale_collected_counts(claims)
    assert list(stale) == [PREFETCH_FILE], (
        f"the helper found nothing wrong with the claim that started #2606: {stale}")
    stated, actual = stale[PREFETCH_FILE]
    assert stated == 73 and actual != 73, stale


# ── #2583: a line anchor has to land inside a definition, not just a file ────

#: What the resolution node above asks of a `path:NNN` citation is that the FILE
#: is there and the number is smaller than its line count. `len(splitlines())`
#: is not an anchor: in a 2163-line file it points anywhere, so a citation whose
#: number lands on `def main()` in a module whose decider is elsewhere passed,
#: and passed in the two threat-model control cells that started #2583
#: (`guardian.py:2147` for a crash-loop verdict that `evaluate_liveness`
#: decides at `:689`). The node below asks the second question: is line N inside
#: something the file *defines*.
#:
#: The tolerance is the span, and only the span. The rule reports an anchor that
#: lands on prose, a comment, a blank line between two definitions, or a
#: definition other than the one the sentence is about; it does NOT ask that the
#: enclosing definition's *name* appear in the citing sentence, which is the
#: other half #2583 proposed and which is not shipped here: a threat-table cell
#: is prose that never names a symbol, and `architecture/threat-model.md`'s own
#: corrected cells would fail such a rule. Whether to add name proximity is an
#: open question on the item, settled only against this rule's report rate.
#:
#: What is graded: a citation whose file resolves under one of the roots and
#: parses as Python. Non-Python targets, a file that does not parse, and a
#: citation with no anchor stay with the resolution node above — this node adds
#: a question, it does not take one away.

#: A `.py:NNN` spelling seen without the extractor, used only to ask a doc that
#: has anchors whether the extractor graded any of them. Deliberately not
#: `_CITATION`: an assertion that reads the same regex as the extraction it
#: audits proves nothing about that extraction.
_ANCHORISH_PY = re.compile(r"[\w./$~#-]+\.py:\d+")

#: How many `.py:NNN` anchors the corpus-wide sweep has to grade to be a sweep.
#: Measured at shipping, over the corpus this diff re-pointed: 140 anchors into
#: parseable Python (the pre-fix scan counted 143; three of the 18 it reported
#: became name citations with no number left to grade). The floor is well below
#: the count and above nothing: a corpus whose anchors all went unresolvable — a
#: moved `ROOT`, a renamed package — would grade a handful, and a node grading a
#: handful of 140 is not reading the corpus.
ANCHOR_GRADED_FLOOR = 100


def definition_spans(py: Path) -> list[tuple[int, int, str]] | None:
    """Every span in a Python file a line anchor may land in, or None to skip.

    A *definition* is any `def`/`async def`/`class` at any nesting, with its
    decorator lines, because those lines are part of what the definition is
    (`@property` above a getter is the line a reader looks for). Beside them, the
    module's own bindings — `X = …` and `X: T = …` at column zero, which is what
    "module-level assignment" means in the clause. A multi-line `X = (…)` is one
    span, so an anchor anywhere inside a table written as a module constant is
    inside a binding.

    Anything else is not a span, and that is the whole rule: a comment, a
    docstring, the blank line between two statements, an `import`'s continuation
    line, a bare module-level call, a `try:` arm. Each of those is something the
    file *says* or something that happens once at import, not a thing the module
    defines — and every one of them is an anchor that moves on its own without
    the sentence about it changing. The corpus's 18 reported anchors were all in
    that set: `app/paths.py:100` and `app/harness/protected_paths.py:119` land on
    comment lines above the constant they name (`EVAL_BASELINES_DIR` is 21 lines
    below its own citation, `PROTECTED_WRITE_ROOTS` 32);
    `app/harness/microcompact.py:11` and `agent_mcp/egress.py:10` land on module
    docstrings, which state policy and decide none of it.

    What a doc does instead when the thing it means is a module-level call or a
    fallback arm is name it: "`eval/run_eval.py`'s `os.environ.setdefault
    ("LLOYD_DJEV_SHADOW", "0")`" outlives any insertion above it, which is the
    line number's only advantage, and it is what three of the re-points in this
    diff do.

    None means "ask me something else": the file does not parse, so its spans are
    not knowable. That skip is not free — `unparsable_py_citations` below counts
    exactly those citations and the sweep node pins the count to zero, so a file
    that stops parsing is a red node naming it, not a silent pass.
    """
    try:
        tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.end_lineno is not None):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            spans.append((start, node.end_lineno, node.name))
    for node in tree.body:                             # module level only
        target = (node.targets[0] if isinstance(node, ast.Assign)
                  else getattr(node, "target", None))
        if isinstance(target, ast.Name) and node.end_lineno is not None:
            spans.append((node.lineno, node.end_lineno, target.id))
    return spans


def py_anchor_citations(text: str) -> list[tuple[str, Path, int]]:
    """The citations the span rule owns: `path:NNN` spans whose file resolves and is Python.

    Resolution is `resolve_citation`, so a citation into the vault or the data
    root is read where it lives rather than where the checkout guesses. A `.py`
    that does not parse is still one of these — it lands in
    `unparsable_py_citations` instead of `definition_spans`, and the per-doc
    identity and the sweep's zero-count both have to account for it.
    """
    out: list[tuple[str, Path, int]] = []
    for span, path, line in cited_citations(text):
        if line is None:
            continue
        hit = resolve_citation(path)
        if hit is not None and hit.suffix == ".py":
            out.append((span, hit, line))
    return out


def unparsable_py_citations(text: str) -> list[tuple[str, Path, int]]:
    """The `.py` citations this rule cannot grade because their file will not parse.

    The one exit `definition_spans` has, spelled out so it can be counted: the
    sweep pins this list empty over the corpus, which is what stops "a file that
    does not parse is skipped" from being a way for a broken or renamed target to
    pass quietly. An anchor whose file no longer parses is not an anchor either.
    """
    return [(span, hit, line) for span, hit, line in py_anchor_citations(text)
            if definition_spans(hit) is None]


def anchor_span_report(text: str) -> tuple[list[tuple[str, Path, int]], list[str]]:
    """`(graded, mis_anchored)` for one doc: what the span rule read, and what it objects to.

    `mis_anchored` carries the span as the doc wrote it, which is the only
    spelling a reader can go and find.
    """
    graded: list[tuple[str, Path, int]] = []
    bad: list[str] = []
    for span, hit, line in py_anchor_citations(text):
        spans = definition_spans(hit)
        if spans is None:
            continue
        graded.append((span, hit, line))
        if not any(lo <= line <= hi for lo, hi, _name in spans):
            bad.append(span)
    return graded, bad


@pytest.mark.parametrize("slug", PATH_DOC_SLUGS)
def test_every_line_anchor_in_an_architecture_doc_lands_inside_a_definition(slug):
    """Clause 1 and clause 4 (#2583): a line anchor is graded on its span, over
    the whole corpus.

    Parameterized over `PATH_DOC_SLUGS` — the same `ARCH.glob("*.md")` the
    command sweep and the resolution node read, with `ARCH_DOC_SURFACE` as its
    vacuity guard — and not over `NEW_DOCS`: #2583's own two instances were in
    the threat model, but the defect is a citation written anywhere, and the
    sweep that found the corpus's other mis-anchors ran over the glob while the
    resolution node's own widening (#2582) had already made the glob the path
    surface. A doc written tomorrow is graded the moment it lands.
    """
    text = _text(slug)
    owned = py_anchor_citations(text)
    graded, bad = anchor_span_report(text)
    assert graded or not owned, (
        f"{slug} cites {owned} — `path:NNN` spans whose files are present and "
        "Python — and the span rule graded none of them. An extraction that "
        "matched nothing would pass this node for free, which is the failure "
        "every extractor-proof in this file exists to prevent")
    # The one skip the rule takes, bounded per doc. The first gate refused the
    # `graded + unparsable == owned` version of this: partitioning the same list
    # two ways cannot fail, so it graded nothing while reading like a check. This
    # one can: a cited file that stops parsing (moved, renamed, or turned into a
    # template) lands here rather than passing through `anchor_span_report`'s
    # `continue`.
    unparsable = unparsable_py_citations(text)
    assert not unparsable, (
        f"{slug} cites {[s for s, _p, _l in unparsable]}, which resolve to `.py` "
        "files that do not parse, so no span of theirs is knowable and the rule "
        "graded none of them")
    assert not bad, (
        f"{slug} anchors {len(graded)} citations into parseable Python and "
        f"{len(bad)} of them land outside every definition in the file: "
        f"{sorted(bad)}. A line number smaller than the file's line count is "
        "not an anchor — it is a pointer into whatever comment, docstring or "
        "blank line happens to be at that offset, and the next edit to the file "
        "moves the thing the sentence meant without moving the number.")


def test_the_anchor_span_sweep_grades_the_corpus_not_a_named_few():
    """Clause 4's own proof: the surface the sweep reads is the glob, the glob
    has a size, and the anchors it grades are a surface, not an accident.

    A parametrize over a glob collects no id at all when the glob finds nothing,
    so the sweep needs a count beside it — the same reason `ARCH_DOC_SURFACE`
    exists for the command half and `test_the_path_surface_is_the_whole_corpus_the_vacuity_number_holds`
    exists for the resolution half. This node re-reads the glob rather than
    trusting that the other node ran.
    """
    on_disk = [p.name for p in sorted(ARCH.glob("*.md"))]
    assert on_disk == PATH_DOC_SLUGS and len(on_disk) == ARCH_DOC_SURFACE, (
        f"the span sweep reads {PATH_DOC_SLUGS} while the glob has {on_disk} and "
        f"the pinned surface is {ARCH_DOC_SURFACE}")
    graded, per_doc = [], {}
    for slug in PATH_DOC_SLUGS:
        g, _bad = anchor_span_report(_text(slug))
        graded += g
        per_doc[slug] = len(g)
    assert len(graded) >= ANCHOR_GRADED_FLOOR, (
        f"the sweep graded only {len(graded)} line anchors "
        f"({per_doc}); the corpus measured 140 when this rule shipped, so this is "
        "anchors going unread — a path that stopped resolving, or a rename — not "
        "a clean corpus")
    read_by = {slug for slug, n in per_doc.items() if n}
    assert len(read_by) >= 15, (
        f"only {sorted(read_by)} contribute graded anchors: a sweep that reads "
        "one doc's anchors and skips the rest is the #1699 shape #2582 was filed "
        "against")
    # One-directional on purpose. `_CITATION` sees a subset of what a plain regex
    # does — it refuses a bare `entity_naming.py:871` for want of a directory and
    # a citation into a tree this box cannot read — so the surplus is the
    # extractor's known silence, recorded on the item, not this node's subject.
    # What the comparison does catch is the extractor inventing an anchor no doc
    # spelled, which would make the sweep grade a sentence nobody wrote.
    probe = sum(len(_ANCHORISH_PY.findall(_text(slug))) for slug in PATH_DOC_SLUGS)
    assert probe >= len(graded), (
        f"the independent probe counts {probe} `.py:NNN` spellings in the corpus "
        f"while the extractor graded {len(graded)} of them")
    # The skip `definition_spans` is allowed, bounded. The per-doc identity makes
    # an unparsable citation visible; this makes it a failure, because an anchor
    # into a file that no longer parses is an anchor into nothing — the same loss
    # #2582 was filed over when a whole root went unread. Zero was measured when
    # the rule shipped, over all 140 owned anchors.
    unparsable = [f"{slug}:{span}" for slug in PATH_DOC_SLUGS
                  for span, _p, _l in unparsable_py_citations(_text(slug))]
    assert not unparsable, (
        f"these `.py` citations resolve but do not parse, so nobody grades them: "
        f"{unparsable}. Either the target moved, was renamed, or became a "
        "template that never parses — re-point it or say so here in words.")


#: Three anchors the corpus really carried, each paired with the line this round
#: re-pointed its sentence to. These are the fixture for the two nodes below: a
#: pair invented for the occasion proves the rule fires on the invention, which
#: is not a claim about the corpus.
#:
#:   * `agent_mcp/facts.py:104` — the comment block above `_OPPOSING_PAIRS`,
#:     cited by `memory.md` for `_detect_contradictions_sync`, which is at `:430`.
#:   * `app/paths.py:100` — a comment line, cited by `measurement.md` for
#:     `EVAL_BASELINES_DIR`, which is at `:121`.
#:   * `tests/test_backlog_dedupe.py:58` — the blank line above the `EXISTING`
#:     fixture tuple, which the `dedupe`-row claim in `djev.md` is about.
REAL_WRONG_ANCHORS = (
    ("agent_mcp/facts.py", 104, 430),
    ("app/paths.py", 100, 121),
    ("tests/test_backlog_dedupe.py", 58, 59),
)


def test_the_span_rule_is_shown_able_to_fail_on_anchors_the_corpus_wrote():
    """Clause 2: the rule is shown able to fail, on anchors the corpus actually
    wrote, and each pair is first asserted on the AST.

    The order matters. A pair asserted only through the report could be green
    because the rule works or because the file moved under it and both numbers
    now sit where the rule likes them; so the wrong line is checked to be
    outside every span and the right line inside the span of the symbol the
    citing sentence names, and only then does the report get asked. And the
    #2583 repro is asserted here too, from the other side: the resolution node's
    own predicate (`n_lines >= line`) passes every one of these numbers, which is
    exactly why this second question had to be added.
    """
    for rel, wrong, right in REAL_WRONG_ANCHORS:
        spans = definition_spans(ROOT / rel)
        assert spans is not None, f"{rel} stopped parsing; the pair is ungradeable"
        inside = [(lo, hi, name) for lo, hi, name in spans if lo <= wrong <= hi]
        assert inside == [], f"{rel}:{wrong} is now inside {inside}, not a fixture"
        assert any(lo <= right <= hi for lo, hi, _ in spans), (
            f"{rel}:{right} is outside every definition, so the 'good' half of "
            "this pair proves nothing")
        n_lines = len((ROOT / rel).read_text(encoding="utf-8").splitlines())
        assert wrong < n_lines and right < n_lines, (
            f"{rel} has {n_lines} lines; the resolution node would report one of "
            f"{wrong}/{right} and the comparison below is not the one #2583 made")

        bad_text = f"the decision is at `{rel}:{wrong}`\n"
        graded, bad = anchor_span_report(bad_text)
        assert graded, f"`{rel}:{wrong}` was never graded, so it cannot be reported"
        assert bad == [f"{rel}:{wrong}"], (
            f"the corpus wrote `{rel}:{wrong}` and the span rule passed it: {bad}")

        good_text = f"the decision is at `{rel}:{right}`\n"
        graded_ok, bad_ok = anchor_span_report(good_text)
        assert graded_ok and bad_ok == [], (
            f"the re-pointed `{rel}:{right}` is reported ({bad_ok}), so the rule "
            "is not reading spans")


def test_the_span_rule_stops_short_of_a_wrong_anchor_that_lands_on_a_definition():
    """The boundary of this rule, pinned rather than left to be rediscovered.

    #2583's two founding instances are NOT anchors this rule can catch, and a
    green corpus must not be read as if it were: `guardian.py:2147` is the head
    of `def main()` and `:689` is the head of `evaluate_liveness`, so both
    numbers sit inside a definition and the span rule is silent on both. Same
    for `app/desktop_lease.py:89` (`grant`, the human's side) against `:123`
    (`agent_may_act`, the decision a turn crosses). Telling one `def` from the
    one a threat row means is the name-proximity half of the item, which triage
    ruled unsatisfiable as written — a threat-table cell is prose that never
    names a symbol, so that rule would refuse `architecture/threat-model.md`'s
    own corrected cells — and which the item carries as an owed ruling.

    So this node asserts the silence. It is the same four numbers #2583 was
    filed with, checked structurally rather than remembered, and it is what a
    later reader needs to know before trusting "every anchor is graded".
    """
    guardian, lease = ("agent-services/guardian/guardian.py", "app/desktop_lease.py")
    heads: dict[str, dict[str, int]] = {}
    for rel in (guardian, lease):
        spans = definition_spans(ROOT / rel)
        assert spans is not None, f"{rel} stopped parsing; the boundary is ungradeable"
        heads[rel] = {name: lo for lo, _hi, name in spans}
    assert heads[guardian]["main"] == 2147, (
        f"`main` moved off :2147 ({heads[guardian]['main']}) — if it is gone "
        "altogether, this boundary may no longer be the one worth pinning")
    assert heads[guardian]["evaluate_liveness"] == 689
    assert heads[lease]["grant"] == 89
    assert heads[lease]["agent_may_act"] == 123

    _g, bad = anchor_span_report(
        f"| T-26 | ships a crash loop | `{guardian}:2147` |\n"
        f"| T-35 | acted on with no lease | `{lease}:89` |\n")
    assert bad == [], (
        f"the span rule is now reporting anchors that land on a real `def` "
        f"({bad}) — either the rule grew a name window, which is the owed "
        "ruling and needs the item's sign-off, or these two lines moved")


def test_the_span_rule_grades_the_file_the_resolver_pointed_at(tmp_path, monkeypatch):
    """The one process boundary this rule crosses, put across on purpose.

    #2582 was filed because the resolution node read one root when the corpus
    cites several; a span rule bolted on beside it inherits the same seam, so a
    citation into the runtime data root has to be graded against the file the
    resolver actually returned — not against a same-named file in the checkout,
    and not against `ROOT` guessed a second time. The two copies disagree
    deliberately, and the span rule's verdict follows the copy under the data
    root, which is the only proof that `definition_spans` was handed the
    resolved path rather than a re-derived one.
    """
    rel = "eval/span_rule_probe.py"
    assert not (ROOT / rel).exists(), f"{rel} exists in the checkout; pick another"
    # The module's own data-root binding, not `app.paths`: `_resolve` reads this
    # name, so pointing it at `tmp_path` is what makes the citation resolve to a
    # file that exists in no checkout.
    monkeypatch.setattr(sys.modules[__name__], "ENV_DATA_ROOT", tmp_path)
    (tmp_path / "eval").mkdir(parents=True, exist_ok=True)
    probe = tmp_path / rel
    probe.write_text(
        "def data_root_copy():\n    return 2\n\n\n# a comment, no definition\n",
        encoding="utf-8")
    text = f"cites `$LLOYD_DATA/{rel}:4`, a comment in the data-root copy\n"
    graded = py_anchor_citations(text)
    assert len(graded) == 1, (
        f"the citation did not resolve to exactly one file: {graded}. It must "
        "exist ONLY under the data root — `resolve_citation` takes the first hit, "
        "so a checkout copy would be the honest answer and prove nothing")
    _span, target, line = graded[0]
    assert target == probe, f"resolved to {target}, not the file outside the tree"
    assert line == 4 and anchor_span_report(text)[1] == [f"$LLOYD_DATA/{rel}:4"], (
        "the anchor was not graded against the file the resolver pointed at: the "
        "copy outside the checkout has a bare comment at line 4 and has to be "
        "reported for that, which reading the checkout could not show")


def test_the_span_rule_sees_an_anchor_inside_a_table_cell():
    """Clause 3's extractor proof, on the form the corpus's anchors are written
    in. `architecture/threat-model.md` is a table of control citations, and so
    is every threat table that will follow it; an extractor whose `_CODE_SPAN`
    happened to need a line start would grade none of them and report a clean
    corpus.
    """
    row = "| T-99 | x | E | a threat that reads plausibly | `app/paths.py:100` |\n"
    graded, bad = anchor_span_report(row)
    assert graded, "an anchor inside a table cell was not extracted at all"
    assert bad == ["app/paths.py:100"], (
        f"app/paths.py:100 is a comment above `FAILURE_LEDGER_DB`, so the cell "
        f"has to be reported; got {bad}")
    clean = "| T-99 | x | E | a threat that reads plausibly | `app/paths.py:102` |\n"
    assert anchor_span_report(clean)[1] == [], (
        "the binding line itself is reported, so the rule has no tolerance for "
        "an anchor that is merely one line out")


ARCHIVE_DIR = ARCH / ".archive"

#: The sentence as it stood before #2582, kept as the fixture this node has to
#: catch. It told a reader that a doc describing something gone "is moved to
#: `architecture/.archive/` (gitignored)", present tense, in the page every new
#: author reads first.
ARCHIVE_CLAIM = "`architecture/.archive/` (gitignored)"

#: What the corrected sentence has to keep saying, and the commit the pre-fix
#: sentence is read from — this round's base, which stays an ancestor of `main`
#: however many docs land on top of it. `HEAD` is deliberately not the control:
#: the moment this fix is committed, `HEAD` is the page that passes.
ABSENT_CLAIM = "no `architecture/.archive/` directory is present"
PRE_FIX_INDEX = "828ce7a5"


def _live_preamble(text: str) -> str:
    """The part of `index.md` that speaks in the present tense: everything before
    the dated Review log. A log line is a record of what somebody believed on a
    date and stays as written; a preamble sentence is an instruction, and the two
    are not graded by the same rule — see the same split in
    `test_arch_review_states_the_archive_directory_is_absent_here`.
    """
    cut = text.find("## Review log")
    assert cut > 0, ("index.md lost its Review log section, so the split this "
                     "node grades by is no longer there")
    return text[:cut]


def test_index_md_sends_a_reader_to_history_rather_than_the_absent_archive():
    """Clause 4: the index's own archive sentence, and a node that reads it.

    Three facts, each checked rather than quoted:

    * `architecture/.archive/` is **not in this checkout** — `ls -d` on it is a
      `No such file`, which is why a present-tense "is moved to" there sent a
      reader to a path that is not there;
    * `.gitignore:192` carries `/architecture/.archive/`, and that pattern
      matches the *trailing-slash* spelling only — `git check-ignore
      architecture/.archive` returns rc=1. So "(gitignored)" was true of a rule
      and false of the box, and it was uncatchable by the git-ignore exemption in
      `unresolved_citations`, which asks the bare spelling;
    * retirements in this corpus are `git rm`'d and read back with `git show`,
      which is what `## Retired (in git history alone)` on the same page already
      does for one doc.

    So the fix is the sentence, and the check is this node rather than a reader:
    the preamble may not claim the archive is where a doc goes, must say the
    directory is absent, and must still name git history as where a retirement
    lives. The control runs the same predicate over the pre-fix page from git, so
    the node is shown to catch the sentence it exists for.
    """
    text = _text("index.md")
    pre = _live_preamble(text)
    assert not ARCHIVE_DIR.exists(), (
        f"{ARCHIVE_DIR} exists again, so this node's premise moved — check whether "
        "retired docs are being kept there and re-read the sentence it pins")
    assert ARCHIVE_CLAIM not in pre, (
        "the index again tells a reader a retired doc is moved to a directory this "
        "checkout does not have")
    assert ABSENT_CLAIM in pre, (
        "the corrected sentence no longer says out loud that the archive directory "
        "is absent here, which is the half a new author needs")
    assert "git history" in pre, (
        "the corrected sentence no longer says where a retirement actually goes")
    assert "## Retired (in git history alone)" in text, (
        "the section the preamble points at is gone, so the pointer is the new "
        "version of the old bug")

    # The git rule the sentence mistook for a fact.
    probe = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-v",
                            "architecture/.archive"], capture_output=True, text=True)
    assert probe.returncode == 1, (
        f"`git check-ignore architecture/.archive` now says {probe.stdout.strip()!r}; "
        "the bare spelling the exemption asks about is the one that was never "
        "ignored, and this node's reading of the incident has to be re-checked")
    slashed = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-v",
                             "architecture/.archive/"], capture_output=True, text=True)
    assert slashed.returncode == 0, (
        "the trailing-slash spelling stopped matching a .gitignore pattern, so the "
        "word \"gitignored\" is now false in a different way — see .gitignore:192")

    before = subprocess.run(["git", "-C", str(ROOT), "show",
                            f"{PRE_FIX_INDEX}:architecture/index.md"],
                            capture_output=True, text=True)
    assert before.returncode == 0, before.stderr
    assert ARCHIVE_CLAIM in _live_preamble(before.stdout), (
        f"the control lost its bite: {PRE_FIX_INDEX}'s index.md no longer carries "
        "the sentence this node was written to catch, so the node is green with "
        "nothing left for it to catch")


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
# The #1699 docs, as review units
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
        "elsewhere for — every doc in `NEW_DOCS` carries at least 4, so a section trimmed to "
        "one line is boilerplate wearing the heading, not a boundary")
    assert len(body) >= 400, (
        f"{slug} §{NOT_COVERED} has {len(body)} chars of prose, under the 400 "
        "floor — a heading with a stub under it reads as an all-clear, which is "
        "the failure this section exists to prevent")


@pytest.mark.parametrize("slug", NEW_DOCS)
def test_each_new_doc_is_a_review_unit_the_moment_it_lands(slug):
    """The whole reason for the #1699 docs: `doc_slugs` globs the directory, so
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
    These architecture docs spell vault paths with that prefix (`measurement.md`,
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
    # The account home, not `$HOME`: inside a gate the two differ, and the docs
    # spell `~/obsidian/…` meaning the account's — the farm there carries the vault
    # only as a symlink, so asserting against `$HOME` pins whichever spelling the
    # process happened to expand, which is not the claim this row makes.
    rooted = ACCOUNT_HOME / "obsidian" / "lloyd" / "SOUL.md"
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
NAMED_NODE = ("test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree"
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
    - the node #1763 files — `test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree`
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


#: The node #2645 was filed against, spelled with its parametrisation, for the
#: same reason `NAMED_NODE` above is: "the check is green" is equally true of a
#: check that no longer runs. This one is the red node the item names; the node
#: above is the one #1763 filed.
FILED_NODE = ("test_every_path_an_architecture_doc_cites_resolves_in_the_working_tree"
              "[voice.md]")


def test_a_home_prefixed_citation_resolves_under_the_gate_s_nested_home(tmp_path,
                                                                        monkeypatch):
    """Clause 1's mechanism, over two directories this node builds itself.

    The gate's `tests` rung runs candidate code with `HOME=<round>/home` and
    `LLOYD_DATA=<round>/home/lloyd-data` (`scripts/automod/gate.py::_child_env`).
    That home is a farm that by design carries neither `lloyd` nor `lloyd-data`
    (`scripts/automod/worktree.py::HOME_LINK_SKIP`), and it sits INSIDE the account
    home. Both facts are load-bearing here, so the fixture reproduces both: an
    account home holding the one real file, and a round home nested three levels
    below it holding nothing.

    Two defects compounded, and the node below fails on each of them separately.
    `_resolve` expanded `~/` against `Path.home()`, so a doc's `~/lloyd-data/…`
    pointed at the empty farm; and `_rel_within` tried `ACCOUNT_HOME` before
    `HOME`, which is true of anything under the account home — the farm included —
    so the recovered relative name carried the whole nesting and no root could
    re-find it. The result was a citation that is true of this box reported as a
    missing file, on nine architecture docs, in rounds that had changed none of
    them: `base_probe: probed 4 file(s) at base 5592e75c in a fresh data root: 16
    already failing` (`~/.local/state/lloyd-automod/rounds/SM_20261013_060835/gate.json`).
    """
    mod = sys.modules[__name__]
    account = tmp_path / "account"
    (account / "lloyd-data").mkdir(parents=True)
    real_file = account / "lloyd-data" / "workers.db"
    real_file.write_text("not really sqlite", encoding="utf-8")
    round_home = account / "lloyd-work" / "R_ONE" / "home"
    round_home.mkdir(parents=True)
    vault, data = account / "obsidian", account / "lloyd-data"
    for name, value in (("HOME", round_home), ("ACCOUNT_HOME", account),
                        ("DATA_ROOT", data), ("VAULT_ROOT", vault),
                        ("ENV_DATA_ROOT", tmp_path / "round-data")):
        monkeypatch.setattr(mod, name, value)
    monkeypatch.setattr(mod, "CITATION_ROOTS",
                        (("checkout", ROOT), ("vault", vault),
                         ("data", data), ("home", account)))

    m = _CITATION.match("~/lloyd-data/workers.db")
    assert m and m.group("pre") == "~/", (
        "the extractor no longer reads `~/lloyd-data/workers.db` as a home-prefixed "
        f"citation, so everything below answers a question nobody asked: {m}")

    resolved = _resolve(m.group("pre"), m.group("path"))
    assert resolved == real_file, (
        f"`~/` expanded against the moved home rather than the account home, which is "
        f"exactly how a true citation goes missing at the gate: {resolved}")
    assert resolve_citation(resolved) == real_file, (
        "a citation pointing at a file that exists came back unresolved")

    # The second defect, which the fix above would hide on its own: a path that
    # lands under the round home by some other route — a `$LLOYD_DATA/` citation,
    # or any caller handing `citation_roots` a path it built elsewhere — has to
    # give back the spelling the doc wrote, not the nesting it was found under.
    inside_farm = round_home / "lloyd-data" / "workers.db"
    assert _rel_within(inside_farm) == Path("lloyd-data/workers.db"), (
        "the account home swallowed a path under the round home, so the relative "
        f"name came back carrying the nesting: {_rel_within(inside_farm)}")
    assert resolve_citation(inside_farm) == real_file, (
        "the same file resolved through one spelling and not the other, which is the "
        "half that makes the guard depend on which root won the first match")


def test_the_filed_node_passes_with_home_and_the_data_root_redirected():
    """Clause 1 across the process boundary the gate actually crosses.

    `_pytest` above inherits this process's environment, and that is why the tree
    could read green locally for a day while every gate round carried seven red
    nodes: the failure needed the pair `Gate._child_env` sets, `HOME` and
    `LLOYD_DATA`, moved together. This node rebuilds that pair in a child
    interpreter and asks pytest — not this file — for the verdict on the node #2645
    names.

    The farm mirrors `worktree.py::ensure_round_home` instead of inventing a smaller
    one, because `~/.local/state/lloyd-automod/…` and `~/.config/…` citations in the
    corpus resolve *through* those symlinks: a home without them would fail here for
    a condition the gate never produces, which is the false alarm this suite has
    been paying for all round. `lloyd` is the checkout and `lloyd-data` is the
    round's own empty one, the two entries the farm withholds from the account.

    It is built UNDER the account home — `<ACCOUNT_HOME>/lloyd-work/<name>/home`,
    the layout `round_home` and `round_data_root` produce — and not under
    `tmp_path`, because the nesting IS the condition: a farm standing outside the
    account home can never be swallowed by it, so a `/tmp` layout would leave the
    `_rel_within` order unexercised and let a regression pass on a layout the gate
    does not build. Scope, said plainly: what this node pins across the child
    boundary is the `~` expansion; the ordering defect is pinned by the node above,
    where both homes are fixtures this file controls.
    """
    import shutil

    from scripts.automod.gate import TESTS_MARK_EXPR
    from scripts.automod.worktree import HOME_LINK_SKIP

    nesting = ACCOUNT_HOME / "lloyd-work" / f"check-2645-{os.getpid()}"
    home = nesting / "home"
    home.mkdir(parents=True)
    data = home / "lloyd-data"          # the round's own: real, and empty
    data.mkdir()
    try:
        for entry in sorted(Path.home().iterdir()):
            if entry.name in HOME_LINK_SKIP:
                continue
            try:
                (home / entry.name).symlink_to(entry)
            except OSError:
                pass
        (home / "lloyd").symlink_to(ROOT)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
             "no:cacheprovider", "-m", TESTS_MARK_EXPR, f"{THIS_FILE}::{FILED_NODE}"],
            cwd=ROOT, env=dict(os.environ, HOME=str(home), LLOYD_DATA=str(data)),
            capture_output=True, text=True, timeout=300)
    finally:
        # What lives under here is symlinks plus two empty directories, and
        # `rmtree` descends the link tree without following it, so nothing of the
        # account's own is in reach; the `finally` keeps a failure from leaving a
        # stray farm behind for the next round to trip over.
        shutil.rmtree(nesting, ignore_errors=True)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-2000:]
    assert "1 passed" in out, (
        "the child run recorded no passing verdict: an unmatched or deselected node id "
        f"also exits 0, and would read here as the fix working: {out[-600:]}")


# ── #2625 clause 2: those three nodes green because the page was re-measured ───
#
# The two reds #2625 filed were prose drift on `ambient-context-injection.md` — a
# line anchor `840bb0e8` left pointing at a `#:` comment when the function it names
# moved below it, and a collected count `14333737` left at 17 when it added the
# node that made it 18 — so the fix is a doc edit and the only enforcement that
# these three verdicts are green *by passing* is this: the ids still collected under
# the gate's own selection, and reported PASSED by a child pytest.

#: Two of the three nodes #2625 files, spelled as pytest spells them — the
#: parametrised one with its id, because an id that is no longer collected is
#: exactly the failure this guard exists for.
RED_NODES_2625 = (
    "test_the_ambient_page_states_the_command_its_collected_counts_came_from",
    "test_every_line_anchor_in_an_architecture_doc_lands_inside_a_definition"
    "[ambient-context-injection.md]",
)

#: The third node the item names. Its verdict cannot be read from a run this file
#: starts: it runs the file, so running it from here would run this node inside it.
SELF_RUNNER_2625 = "test_the_gate_selection_drops_nothing_and_the_named_node_passes"


def test_the_nodes_2625_files_are_green_with_nothing_hidden():
    """#2625 clause 2: the three nodes that item names are green because the page
    was re-measured, not because any of them came off the rung.

    The three ways a red node stops reading red with nothing fixed — moved out of
    the gate's mark expression, marked `skip`/`skipif`/`xfail`, or deleted — are the
    ones #1763 lists above, and they are graded here over the ids #2625 files rather
    than over the file in general. A child pytest asked with
    `scripts.automod.gate.TESTS_MARK_EXPR` still collects all three of them, the
    source carries no marker that could have taken one out, and the two that a node
    here may run report **PASSED by name** with no `skipped` or `xfailed` line in
    that run.

    The third is this file's own selection node, and the bargain it makes about
    itself applies unchanged to it: excluded by id, which is visible in the argv
    below, never by a mark, which is what the collection assert above would catch.
    Its verdict is the one the gate's full-suite run records with no exclusion in
    it, and its half here is collection plus the marker scan.
    """
    from scripts.automod.gate import TESTS_MARK_EXPR

    gated = [ln for ln in (_pytest(["--collect-only", "-q", "-m", TESTS_MARK_EXPR,
                                    THIS_FILE]).stdout.splitlines()) if "::" in ln]
    missing = [n for n in RED_NODES_2625 + (SELF_RUNNER_2625,)
               if not any(n in ln for ln in gated)]
    assert missing == [], (
        f"{missing} is no longer collected under the gate's selection "
        f"({TESTS_MARK_EXPR!r}): the node #2625 names was deleted or marked out of "
        f"the hard rung rather than fixed. {len(gated)} nodes are collected here")

    src = (ROOT / THIS_FILE).read_text(encoding="utf-8")
    hidden = _HIDE_RE.findall(src)
    assert hidden == [], (
        f"this file carries a marker that takes a node off the hard rung: {hidden}")

    ran = _pytest(["-v", "-m", TESTS_MARK_EXPR]
                  + [f"{THIS_FILE}::{n}" for n in RED_NODES_2625])
    out = ran.stdout + ran.stderr
    assert ran.returncode == 0, out[-1500:]
    low = out.lower()
    assert "skipped" not in low and "xfailed" not in low, out[-1500:]
    for node in RED_NODES_2625:
        named = [ln for ln in out.splitlines() if node in ln and " PASSED" in ln]
        assert len(named) == 1, (
            f"{node} has no single PASSED line, so a node #2625 was filed for has "
            f"no recorded verdict: {named}")


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
    # `sidecar` sat in this set from 2026-10-07 (#2349 built the arm and could
    # not run it: the figure it exists to produce needs the real primary engine)
    # until #2682 ran the `tool_clear,sidecar` pair on 2026-10-14. It is no
    # longer never-run, so the set is the seven presets that still have no row
    # in any tracked artifact — and `sidecar` being absent here is now the rail
    # that refuses a doc which puts the arm back in the never-run column while
    # `compaction-recall-sidecar-pair-pass*-2026-10-14.json` hold its twenty-four
    # rows each (the `compaction-recall-` prefix is what makes those two files
    # committable at all — see SIDECAR_PAIR_PASS1).
    # The rail above re-derives every `no` from the tracked artifacts, so this
    # literal is the only thing that can be stale here.
    assert {name for name, (mark, _k, _t) in rows.items() if mark == "no"} == {
        "self_record", "observation", "production_self_record",
        "production_observation", "rung4", "rung4_lossy", "rung4_self_record"}, (
        "the never-run set changed shape; the doc's not-yet-run column and the "
        "baselines have to be read together again")

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


#: The #2168 `tool_clear`/`sidecar` pair (#2682), run 2026-10-14 the way the
#: runner's own header requires: two passes over ONE scratch `--data-root`, the
#: first recording the reduction into `<sid>.microcompact-reduced.json`, the
#: second applying it at read time. Both are tracked because the write-up quotes
#: them, and the write-up is the only prose that reads them.
#: The pair's two artifacts. The `compaction-recall-` prefix is not decoration:
#: `.gitignore:141` ignores `eval/baselines/*` and un-ignores only a named set, of
#: which `!eval/baselines/compaction-recall-*.json` is the line that covers this
#: eval. An artifact named `compaction-sidecar-pair-pass1-*.json` — the shape this
#: round first reached for — is invisible to `git add`, so the pair would have
#: looked committed while being one `git clean` from nowhere. The assertion below
#: is that trap, not a style rule, and it asks `git ls-files` whether the three
#: paths are in the index instead of matching these constants' own names against
#: the carve-out: a name-based assert is green while the ignore rule or the index
#: moves underneath it, which is the only way the trap can actually spring.
SIDECAR_PAIR_PASS1 = EVAL / "baselines/compaction-recall-sidecar-pair-pass1-2026-10-14.json"
SIDECAR_PAIR_PASS2 = EVAL / "baselines/compaction-recall-sidecar-pair-pass2-2026-10-14.json"
SIDECAR_PAIR_WRITEUP = (
    EVAL / "measurements/compaction-recall-sidecar-pair-2026-10-14.md")

#: The sentence a pass two that applied nothing would leave behind: both arms
#: compacted the same prompt, so nothing about the flag was measured, and a reader
#: left with only two rates ranks them. #2682's first attempt ended in exactly
#: this state. It is now a refusal (`applied > 0`) rather than a disclosure, so the
#: only thing this constant still pins is the reverse error — the prose calling a
#: pair that did apply its reduction by that name.
SIDECAR_PAIR_NULL = "the pair is vacuous"


def _arm_rows(doc: dict, arm: str) -> list[dict]:
    return [r for r in doc["rows"] if r["arm"] == arm]


def _sidecar_applied(doc: dict) -> int:
    """Reductions the `sidecar` rows of one artifact applied at read time.

    This is the one predicate #2682 clause 1 turns on. A row is a single turn, and
    `load_and_compact_session` runs before that turn's relief can book anything, so
    pass one can only book into `<sid>.microcompact-reduced.json` and pass two is
    the first load that can apply: pass two's sidecar rows all reading 0 means both
    arms compacted the identical prompt and the pair compared nothing.
    """
    return sum(int(r.get("sidecar_applied") or 0) for r in _arm_rows(doc, "sidecar"))


def _tool_clear_applied(doc: dict) -> int:
    """The same sum over the arm the read-time reduction is not wired into.

    Non-zero here means the comparison was not one switch apart, so no delta
    between the two arms could be attributed to `microcompact_reduction_sidecar`.
    """
    return sum(int(r.get("sidecar_applied") or 0)
               for r in _arm_rows(doc, "tool_clear"))


def _require_pass_two_applied(p2: dict) -> int:
    """Refuse a pass two that applied its reduction to no row, and return the count.

    #2682's first attempt died in this state: a single run over a fresh root books
    nothing yet (a row is one turn, and `load_and_compact_session` runs before that
    turn's relief can record anything), so every `sidecar` row reads
    `sidecar_applied == 0` and the two arms are the same prompt graded twice. The
    pair's whole answer is the applied count, so the count is a gate here and not a
    figure to quote — `test_the_sidecar_pair_rail_refuses_a_vacuous_pass_two` runs
    an artifact in that state through this function.
    """
    applied = _sidecar_applied(p2)
    assert applied > 0, (
        f"pass two applied the reduction to 0 of {len(_arm_rows(p2, 'sidecar'))} "
        "sidecar rows, so the `sidecar` arm compacted the same prompt as "
        "`tool_clear` and its two rates are one prompt graded twice: the pair is "
        "vacuous and its numbers may not be recorded as a comparison of the flag")
    return applied


def _require_arms_one_switch_apart(doc: dict, pass_no: str) -> None:
    """Refuse an artifact whose two arms differ by more than the flag.

    `tool_clear` never opens the sidecar record, so it can neither book a reduction
    nor apply one; `sidecar` must consult it on every row. An artifact failing
    either half turns the delta between the two arms into something other than the
    effect of `microcompact_reduction_sidecar`.
    """
    for arm, want_flag in (("tool_clear", False), ("sidecar", True)):
        rows = _arm_rows(doc, arm)
        assert rows, f"{pass_no} has no {arm} rows"
        for r in rows:
            assert (r.get("turn_start") or {}).get(
                "microcompact_sidecar") is want_flag, (
                f"{pass_no}: a {arm} row read microcompact_sidecar="
                f"{(r.get('turn_start') or {}).get('microcompact_sidecar')!r}, so "
                "the two arms are no longer one switch apart")
    assert _tool_clear_applied(doc) == 0, (
        f"{pass_no}: a `tool_clear` row applied "
        f"{_tool_clear_applied(doc)} reduction(s), so the comparison is not one "
        "switch apart and no delta between the arms belongs to the flag")


def test_the_sidecar_pair_is_two_passes_over_one_root_said_so_in_the_doc():
    """Clauses 1, 2, 3 and 4 of #2682, every figure re-derived from the rows.

    What this node exists for is the one way the recorded pair could be worth
    nothing: a pass-two run that never applied its own reduction graded the same
    prompt twice, and the two rates off it read as a comparison of the flag. So
    nothing here is transcribed — including the applied count the write-up quotes,
    which means a future pair has to rewrite the prose, not just drop in new
    artifacts. The vacuous state is refused outright, and the node beside this one
    mutates a copy of the committed pass two to prove the rail fires on it:

      * all three files are in git's index, which is what makes "committed
        artifact" mean something under an ignored directory;
      * both artifacts hold rows for both presets, and pass two replayed pass
        one's own sessions — same keys AND same content hashes;
      * both were closed by the same full-grid invocation, so their `args` and
        their 12-entry `sessions` block name one grid and not the 3-session
        chunks that filled them, and both were graded against one loopback
        primary, which is the seam no test here crosses;
      * the one switch separating `sidecar` from `tool_clear` is visible on
        every row (`turn_start.microcompact_sidecar`), and no `tool_clear` row
        anywhere applied a reduction;
      * the recording pass applied nothing, which is what makes this two passes
        over one root rather than one run relabelled, while pass two applied at
        least one reduction — the pair's central predicate, as a refusal;
      * `architecture/measurement.md`'s `sidecar` row is `yes`, and the
        write-up's rates, kept counts, applied count and the branch those rates
        select are all re-derived from the pass-two artifact.
    """
    # Tracked before anything else — and asked of git, not of these constants' own
    # names. `eval/baselines/*` is ignored wholesale (.gitignore:141) and
    # `!eval/baselines/compaction-recall-*.json` (:144) is the only carve-out that
    # covers this eval, so an artifact under any other stem is invisible to
    # `git add`: the pair would sit untracked beside a green rail reading it off
    # disk, one `git clean` from nowhere. Pattern-matching the two paths this file
    # already chose can only restate that rule and cannot see it, or the index,
    # moving underneath the constant — `git ls-files` reads the index itself.
    want_tracked = [SIDECAR_PAIR_PASS1, SIDECAR_PAIR_PASS2, SIDECAR_PAIR_WRITEUP]
    rel_tracked = [str(p.relative_to(ROOT)) for p in want_tracked]
    listed = set(_git_lines("ls-files", "--", *rel_tracked))
    for rel in rel_tracked:
        assert rel in listed, (
            f"{rel} is on disk but `git ls-files` does not list it, so the pair "
            "the doc's `yes` cell and the write-up's figures rest on is not in "
            "the tree at all: .gitignore:141 ignores `eval/baselines/*` and line "
            "144 un-ignores only `eval/baselines/compaction-recall-*.json`")

    p1 = json.loads(SIDECAR_PAIR_PASS1.read_text(encoding="utf-8"))
    p2 = json.loads(SIDECAR_PAIR_PASS2.read_text(encoding="utf-8"))

    # One root, not two grids. `--data-root` is deliberately absent from `args`
    # (`_write` drops it), so the session content hashes are the only
    # in-artifact evidence that pass two replayed what pass one recorded.
    fingerprint = [(s["key"], s["sha256"]) for s in p1["sessions"]]
    assert fingerprint and fingerprint == [
        (s["key"], s["sha256"]) for s in p2["sessions"]], (
        "the two artifacts are not the same sessions, so they are not two passes "
        "over one --data-root and the pair measures nothing")
    assert p1["args"]["arms"] == p2["args"]["arms"] == "tool_clear,sidecar"
    assert p1["args"]["tools_snapshot"] == p2["args"]["tools_snapshot"], (
        "the two passes graded different tool schemas")
    # The one process boundary every graded row crossed: each driver POSTed to the
    # already-running primary over loopback HTTP (#2682 forbids booting a second
    # `agent-llm-primary`, and a pair graded against two engines is not one
    # measurement). No test can cross that seam itself — it needs the live 95 GB
    # engine — so what the rail does is read the endpoint each artifact recorded and
    # require one loopback host for both passes.
    assert (p1["args"]["base_url"] == p2["args"]["base_url"]
            and p1["args"]["base_url"].startswith("http://127.0.0.1:")), (
        f"the passes report {p1['args']['base_url']!r} and "
        f"{p2['args']['base_url']!r}: only one loopback primary may have graded "
        "the pair, or the two arms were measured on different engines")
    assert p1["args"]["out"] != p2["args"]["out"], (
        "both runs wrote the same --out: the second resumed the first's rows "
        "instead of re-running the grid over the recorded root")
    assert not p1["args"]["dry"] and not p2["args"]["dry"]

    # The grid each artifact was finalised by. `_write` records `args` and the
    # `sessions` block from the invocation that last ran, and neither pass could
    # be run as one 24-row process (a full-grid process was OOM-killed by the
    # 16 G background-Bash scope at 21 rows on 2026-10-13), so each was filled by
    # 3-session chunks and then closed by one full-grid invocation with every row
    # already recorded — which runs nothing and only reaches `_write` after the
    # loop. Without that closing call both artifacts would report the chunk's
    # `sessions: 3` and one size, and the pair would be two grids that happen to
    # share a root. These are the closing invocation's own numbers.
    for pass_no, doc in (("pass one", p1), ("pass two", p2)):
        a = doc["args"]
        assert (a["sessions"], a["sizes"], a["depths"], a["seed"]) == (
            6, "90000,130000", "0.1,0.5,0.85", 600), (
            f"{pass_no}'s artifact was closed by a {a['sessions']}-session "
            f"--sizes {a['sizes']} invocation, not the default 12-session grid")
        assert a["probe"] == "late" and a["shape"] == "auto" and not a["pointer"], (
            f"{pass_no}'s artifact is not the default shape/probe/pointer grid")
        assert len(doc["sessions"]) == 12, (
            f"{pass_no}: {len(doc['sessions'])} sessions recorded, not 12")
        assert len(doc["rows"]) == 24, (
            f"{pass_no}: {len(doc['rows'])} rows, not 12 sessions x 2 arms")

    for pass_no, doc in (("pass one", p1), ("pass two", p2)):
        _require_arms_one_switch_apart(doc, pass_no)

    # The recording pass. A row is one turn, and `load_and_compact_session`
    # runs before that turn's relief can book anything, so pass two is where a
    # reduction first lands — and nothing on pass one may have.
    assert _sidecar_applied(p1) == 0, (
        f"the pass-one artifact applied {_sidecar_applied(p1)} reduction(s), so it "
        "is not the recording pass and the pair is not two passes over one root")

    # Clause 1's central predicate, as a refusal and not a figure to quote. The
    # first revision of this rail summed the same field only to print it and then
    # tolerated a zero by asking the write-up for the word "vacuous", which left the
    # one state this pair exists to escape — both arms compacting the identical
    # prompt — able to be committed green. A vacuous pass two is refused by
    # `_require_pass_two_applied`, and
    # `test_the_sidecar_pair_rail_refuses_a_vacuous_pass_two` runs that state back
    # through the same function instead of trusting it never happens.
    applied = _require_pass_two_applied(p2)
    writeup = _flat(SIDECAR_PAIR_WRITEUP.read_text(encoding="utf-8"))
    # Both arms' two recall rates, at the precision the write-up quotes them,
    # read out of the artifact that produced them.
    for arm in ("tool_clear", "sidecar"):
        s = p2["summary"][arm]
        for measure in ("distinctive", "ambiguous"):
            assert f"{s[measure]['rate']:.2f}" in writeup, (
                f"the write-up never quotes {arm}'s {measure} rate "
                f"{s[measure]['rate']:.2f} from the pass-two artifact")
        assert str(s["kept"]) in writeup, (
            f"the write-up never names how many {arm} rows were kept")
    # The applied count is the figure that says whether the pair measured anything
    # at all, so it stands beside the rates and is re-derived, not transcribed. And
    # the prose may not hedge the other way either: calling a pair that did apply
    # its reduction "vacuous" talk a reader into re-running a measurement that
    # exists. Clause 4's ban on reporting a tie off an all-zero pass two is now
    # carried by the `applied > 0` refusal above — no all-zero pass two can be
    # committed at all, so no write-up can reach a tie to report.
    assert f"applied {applied} reductions" in writeup, (
        f"the write-up does not state the {applied} reductions pass two applied")
    assert SIDECAR_PAIR_NULL not in writeup, (
        f"the write-up says {SIDECAR_PAIR_NULL!r} while the pass-two artifact "
        f"beside it applied {applied} reductions: the prose has to describe the "
        "pair that was actually measured")
    # Which of the two branches the figures select, recomputed rather than copied:
    # the count of measures on which `sidecar` came out BELOW `tool_clear` is what
    # decides whether the ship-on round is owed or the negative is recorded, and a
    # write-up that states the wrong one sends the next reader down the wrong path.
    below = sum(1 for m in ("distinctive", "ambiguous")
                if p2["summary"]["sidecar"][m]["rate"]
                < p2["summary"]["tool_clear"][m]["rate"])
    assert f"{below} of the two measures came out below" in writeup, (
        f"the artifact puts {below} of the two measures below `tool_clear` and the "
        "write-up does not say so, so the branch it selects is not the one the "
        "committed rows support")

    # The doc's never-run column, retired in the same commit as the artifacts.
    _, body = recall_section(_text("measurement.md"))
    assert preset_rows(body)["sidecar"][0] == "yes", (
        "measurement.md still marks the sidecar preset never-run while the pair "
        "is tracked under eval/baselines/")


def test_every_test_node_the_sidecar_pair_writeup_cites_exists():
    """The re-key list a flip round will be handed has to name nodes that exist.

    The write-up's closing paragraph is the only place the flip round's work list
    lives — #2487 closed and cannot host one — so a node name that is not in the
    tree is not a typo there, it is an instruction to re-key a test that does not
    exist while the ones that do stay red. This round shipped exactly that defect:
    the prose cited `test_microcompact_defaults_are_conservative` and
    `test_microcompact_off_when_fraction_is_zero`, neither of which has ever been in
    `tests/test_compaction.py`, when the real pair is
    `test_the_shipped_default_does_not_consult_the_sidecar_at_all` (def :721) and
    `test_one_runs_options_value_arms_the_read_time_half_too` (def :1971). The
    gate's review rung then cited `app/harness/options.py` line 728 — the assert line
    of a node this prose had failed to name — against a 375-line file, and the round
    went ungraded twice.

    Only a whole backtick-wrapped `test_*` token counts as a citation, so a path
    like `tests/test_compaction.py` is not mistaken for a node. Defined names come
    out of the index with `git grep`, so the check follows a rename instead of
    restating a list of them.
    """
    body = SIDECAR_PAIR_WRITEUP.read_text(encoding="utf-8")
    cited = set(re.findall(r"`(test_[a-z0-9_]+)`", body))
    assert cited, (
        f"no `test_*` node citation at all in {SIDECAR_PAIR_WRITEUP.name}, so this "
        "node graded nothing — the prose stopped naming the nodes clause 5 relies on")
    defined = {line.removeprefix("def ")
               for line in _git_lines("grep", "-h", "-o", r"def test_[a-z0-9_]*",
                                      "--", "tests/")}
    assert defined, (
        "`git grep` found no test nodes in tests/ at all, so the corpus read "
        "failed rather than the citation resolving")
    missing = sorted(cited - defined)
    assert not missing, (
        f"{SIDECAR_PAIR_WRITEUP.name} cites node(s) no tracked test file defines: "
        f"{', '.join(missing)} — the flip round would be told to re-key tests that "
        "do not exist while the real ones stay red")


def test_the_sidecar_pair_rail_refuses_a_vacuous_pass_two():
    """Clause 1's `> 0` predicate, shown able to fail on a copy of the artifact.

    The review of this round's first attempt named the defect exactly: the node
    summed `sidecar_applied` purely to quote the number, and at zero asked the
    write-up for the word "vacuous" instead of failing — so the state the pair was
    run to escape was the one state the rail could not refuse. A `> 0` assert nobody
    has seen fire is the same kind of claim as the sentence it replaced, so the
    three mutations below go back through the guards themselves. The committed pass
    two holds twelve `sidecar` rows at `sidecar_applied: 1` and twelve `tool_clear`
    rows at 0, and each mutation changes one of those two facts:

      * every `sidecar` row emptied to 0 — the vacuous pair, which
        `_require_pass_two_applied` has to refuse;
      * every `sidecar` row switched to `microcompact_sidecar: False`, an arm that
        never opened the record being scored as though it had;
      * one `tool_clear` row applying a reduction, which
        `_require_arms_one_switch_apart` has to refuse.

    The unmutated copy is required to pass first: a guard that rejected everything
    would satisfy this node as well as one that can see the difference.
    """
    p2 = json.loads(SIDECAR_PAIR_PASS2.read_text(encoding="utf-8"))
    _require_arms_one_switch_apart(p2, "pass two")
    assert _require_pass_two_applied(p2) > 0, (
        "the guards refuse the committed artifact itself, so the refusals below "
        "would be about the mutation and nothing else")

    emptied = json.loads(json.dumps(p2))
    for r in _arm_rows(emptied, "sidecar"):
        r["sidecar_applied"] = 0
    assert _sidecar_applied(emptied) == 0, (
        "the mutation could not empty pass two's applied count, so it is not the "
        "vacuous state being tested")
    with pytest.raises(AssertionError, match="vacuous"):
        _require_pass_two_applied(emptied)

    blind = json.loads(json.dumps(p2))
    for r in _arm_rows(blind, "sidecar"):
        r["turn_start"]["microcompact_sidecar"] = False
    with pytest.raises(AssertionError, match="no longer one switch apart"):
        _require_arms_one_switch_apart(blind, "pass two")

    contaminated = json.loads(json.dumps(p2))
    _arm_rows(contaminated, "tool_clear")[0]["sidecar_applied"] = 1
    with pytest.raises(AssertionError, match="row applied"):
        _require_arms_one_switch_apart(contaminated, "pass two")


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
    # `:620` since #2583: the anchor used to be `:11`, a line of that module's
    # docstring, and a docstring is what a file says rather than what it does —
    # the same edit that made the new span rule green over this corpus. The site
    # is `microcompact()`, which is what the section prices as a front break.
    assert "app/harness/microcompact.py:620" in sec, flat[:400]
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
# property alone — see the finding appended to #1792 and the rulings in this
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
    """Clause 1 (#1890): the graded surface is the corpus, not the #1699 docs.

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
    # The widening itself: claims outside the #1699 docs are being graded, not
    # skipped. Anything that narrowed the sweep back — a per-doc exemption, a
    # re-introduced `for slug in NEW_DOCS` — empties this set.
    outside = sorted({g for g in graded if g[0] not in NEW_DOCS})
    assert outside, (
        "every graded command claim came from the #1699 docs, so the other "
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
    # `autonomy-jobs.md` joined the set with #2630, which is the set growing in the
    # right direction: §Watch the machinery cites the commands its seven jobs run —
    # `scripts/step_conformance.py replay`, `scripts/maintenance/referential_integrity.py
    # --record`, `python -m scripts.mitigation_drill`, `scripts/automod/landed_outcomes.py
    # --record` — and each of those modules has a dispatch chain, so the claims are
    # graded rather than excused. A doc that started making checkable command claims
    # is what this sweep exists to collect, not a defect; the entry above, whose
    # module has no chain, is the shape that would need reading.
    assert graded_by == {"automod.md", "autonomy-jobs.md", "harness.md",
                         "infrastructure.md", "measurement.md",
                         "research-pipeline.md", "voice.md"}, (
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


def _threat_model_row(row_id: str) -> str:
    """The `threat-model.md` table row whose first column is exactly `row_id` —
    located by id, not by line number, so the row cannot be renumbered out of
    this node's reach."""
    rows = [ln for ln in _text("threat-model.md").splitlines()
            if ln.startswith(f"| {row_id} |")]
    assert len(rows) == 1, f"{row_id} is not exactly one row of threat-model.md: {rows}"
    return rows[0]


def _egress_enforce() -> bool:
    """The shipped egress posture, read from the operand the docs describe: this
    tree's `config.yaml`, key `harness.egress_policy.enforce`.

    `agent_mcp/egress.py:224` arms the deny branch with
    `_env_flag("LLOYD_EGRESS_ENFORCE", _config_flag("enforce"))`. The three graded
    cells all state the *shipped* posture — what a fresh boot does with no
    process-specific override — so the config half is the operand, and the env
    half is deliberately not read here.
    """
    import yaml

    policy = ((yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
               or {}).get("harness") or {}).get("egress_policy") or {}
    enforce = policy.get("enforce")
    assert isinstance(enforce, bool), (
        f"harness.egress_policy.enforce is {enforce!r} in config.yaml, not a "
        "boolean: the graded cells state one of exactly two postures, and a value "
        "`_config_flag` refuses to arm on is neither of them")
    return enforce


#: How each graded cell has to read, per posture. The arm is looked up by the
#: parsed config value, never spelled out at the call site, so a landing that
#: armed the flag would have to reword all three cells and this file would tell it
#: which arm it was now grading (#2581: the §3 row said the key was one "no
#: `config.yaml` sets" a week after #2123 seeded `enforce: false` at
#: `config.yaml:712`, and every assertion in this node stayed green because none
#: of them opened `config.yaml`).
EGRESS_POSTURE_WORDING: dict[bool, dict[str, tuple[str, ...]]] = {
    False: {
        "guard-coverage egress row": ("is set `false`", "#2123"),
        "threat-model B8": ("records rather than refuses", "is set `false`"),
        "threat-model T-19": ("the measurement, not the gate", "false here"),
    },
    True: {
        "guard-coverage egress row": ("is set `true`", "#2123"),
        "threat-model B8": ("refuses rather than records", "is set `true`"),
        "threat-model T-19": ("the gate, not the measurement", "true here"),
    },
}

#: Wording no posture makes true, and both of these were in the corpus. The
#: first is the clause #2581 was filed to kill; the second is the shape a
#: well-meaning reword of it takes.
EGRESS_POSTURE_NEVER = ("no `config.yaml` sets", "not set in `config.yaml`")

#: Marks that are true of both arms, so they are never evidence *against* one.
#: #2123 seeded `harness.egress_policy` whichever way the flag is later set, so a
#: cell naming it stays correct after a landing arms the key — only the words
#: about the value discriminate, and treating a history citation as posture would
#: make the armed arm fail a cell that is merely more informative than it asks.
POSTURE_NEUTRAL_MARKS = {"#2123"}

#: Each cell's unarmed wording and its armed reading, paired. `_opposite_posture`
#: applies them in the direction the live value demands, so what the control
#: below checks is the row a landing that armed `enforce` would have to ship —
#: written here rather than smuggled into the doc.
_POSTURE_SWAPS = (
    ("is set `false`", "is set `true`"),
    ("records rather than refuses", "refuses rather than records"),
    ("the measurement, not the gate", "the gate, not the measurement"),
    ("false here", "true here"),
)


def _opposite_posture(text: str) -> str:
    """`text` rewritten to state the posture the live config does NOT set."""
    armed = _egress_enforce()
    for unarmed_wording, armed_wording in _POSTURE_SWAPS:
        text = text.replace(unarmed_wording if not armed else armed_wording,
                            armed_wording if not armed else unarmed_wording)
    return text


def _egress_posture_problems(cells: dict[str, str],
                            enforce: bool | None = None) -> list[str]:
    """Every way `cells` mis-states the posture. Empty means every cell reads for
    the arm under test and none carries the other arm's wording.

    `enforce` is the controls' only handle and is otherwise the live parsed
    value: passing it is how a control shows the verdict moves with the operand
    while the doc bytes do not.
    """
    arm = _egress_enforce() if enforce is None else enforce
    problems: list[str] = []
    for label, cell in cells.items():
        flat = " ".join(cell.split())
        for never in EGRESS_POSTURE_NEVER:
            if never in flat:
                problems.append(f"{label}: states {never!r}, true of neither posture")
        for mark in EGRESS_POSTURE_WORDING[arm][label]:
            if mark not in flat:
                problems.append(f"{label}: enforce={arm} requires {mark!r}")
        for mark in EGRESS_POSTURE_WORDING[not arm][label]:
            if mark in POSTURE_NEUTRAL_MARKS:
                continue
            if mark in flat:
                problems.append(f"{label}: carries {mark!r}, the wording of "
                                f"enforce={not arm}")
    return problems


def test_guard_coverage_cites_the_live_owner_of_the_egress_posture():
    """#1960 closed; a closed item is not an owner. The two ownership citations
    name #1965, the §3 row says where the posture is published, and the page's
    changelog sentence that records #1960 being *filed* is history and stays.

    #2581 added the edge this node never had. The posture the three graded cells
    state — `guard-coverage.md`'s §3 egress row, `threat-model.md`'s B8 and T-19 —
    is now graded against the parsed value of `harness.egress_policy.enforce` in
    `config.yaml`, the operand `enforce_on()` reads, rather than against a literal
    in this file. Before that the row could explain the unarmed deny branch as "a
    key **no `config.yaml` sets**" while `config.yaml:712` had said
    `enforce: false` since #2123 seeded it on 2026-10-04, and the whole node
    stayed green: a guard reading its own missing input."""
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

    cells = {
        "guard-coverage egress row": row[0],
        "threat-model B8": _threat_model_row("B8"),
        "threat-model T-19": _threat_model_row("T-19"),
    }
    enforce = _egress_enforce()
    problems = _egress_posture_problems(cells)
    assert problems == [], (
        f"harness.egress_policy.enforce parses to {enforce} in config.yaml, and the "
        f"graded cells do not state that posture: {problems}")

    # Clause 1's own scope, widened to the corpus the sweep grades: neither of the
    # two wordings that are false for every arm survives anywhere in
    # `architecture/`. The count beside it is the vacuity guard this file uses
    # everywhere — a glob that found no corpus would report a clean sweep of
    # nothing, which is the same blind spot #2581 is about.
    docs = sorted(p.name for p in ARCH.glob("*.md"))
    assert len(docs) == ARCH_DOC_SURFACE, (
        f"the sweep read {len(docs)} top-level architecture docs, want "
        f"{ARCH_DOC_SURFACE} — see the sweep node for why the count is the guard")
    offenders = [(name, never) for name in docs for never in EGRESS_POSTURE_NEVER
                 if never in " ".join((ARCH / name).read_text(encoding="utf-8").split())]
    assert offenders == [], (
        f"these docs state the posture in words no arm of config.yaml makes true: "
        f"{offenders}")

    # Edge onto the operand, in one direction: these three cells, byte for byte
    # unchanged, have to fail the moment the config value flips. An assertion
    # that only ever compared doc text to doc text would pass this control's
    # subject for either value, which is exactly how the false clause survived.
    flipped = _egress_posture_problems(cells, not enforce)
    assert flipped != [], (
        f"the same cells pass for enforce={not enforce} as well, so the assertion "
        "above grades wording and not the config value")

    # And in the other direction, in this file's existing counterfactual style:
    # text this file writes, worded for the opposite posture, fails the predicate
    # that the live value selects. Each cell has to actually change, or the swap
    # table stopped describing how these cells are worded.
    for label, cell in cells.items():
        opposite = _opposite_posture(cell)
        assert opposite != cell, (
            f"{label}: the swap table rewrites nothing here, so this control could "
            "not tell the two postures apart")
        assert _egress_posture_problems({label: opposite}) != [], (
            f"{label} worded for the opposite posture still passes, so the arm is "
            "not being selected by the config value")

    # The regression itself: put #2581's pre-fix clause back into the row and the
    # posture check has to refuse it, because the key is set, to `false`, at
    # `config.yaml:712` — a reviewed decision whose comment closes "Arming
    # `enforce` is a separate decision", not a default a reader could flip.
    prefix = row[0].replace("is set `false`", "a key **no `config.yaml` sets**", 1)
    assert prefix != row[0], (
        "the §3 row no longer says the key `is set `false``, so there is nothing "
        "here to put the pre-#2581 clause back into")
    assert _egress_posture_problems({"guard-coverage egress row": prefix}) != [], (
        "'a key **no `config.yaml` sets**' passes as a posture, which is the false "
        "claim #2581 was filed to kill")

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
    # Names, never a count. The block's grep matches `parent_of=parent_of`, so a
    # sixth session-scoped lane lands on it too (#2604 added one), and a literal
    # line count fails on a change that left the guards the doc describes exactly
    # as they were. What §4.5 claims is *which* guards `check_bash_command` feeds
    # the resolver, so the three names are what is pinned — and each has to appear
    # on a printed line, which is the one property a count was standing in for.
    for guard in ("check_service_control(", "check_install_provenance(",
                  "check_live_git_writes("):
        assert any(guard in l for l in lines), (guard, lines)

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


#: One dated full re-run, as the page writes them: `Re-run in full on <date> at
#: `<sha>``. The dated form is what separates a re-run that counted what it ran from
#: the earlier prose stamps, which named a sha mid-sentence and said "all blocks still
#: hit" with no denominator to age.
_STAMP = re.compile(r"Re-run in full on (\d{4}-\d{2}-\d{2}) at `([0-9a-f]{8})`")

#: One fenced block. One definition, shared by the count the stamp's denominator is
#: checked against and the pairs the content check compares, so the two cannot be
#: measuring different things and calling it agreement.
_FENCE = re.compile(r"^```\n.*?^```$", re.S | re.M)


def _fenced_blocks(text: str) -> list[str]:
    blocks = _FENCE.findall(text)
    assert blocks, "no fenced block extracted — the pattern matched nothing at all"
    return blocks


def _fenced_blocks_at(rev: str) -> list[str]:
    """The page's fenced blocks as of one revision, via `git show` rather than the
    working tree: the whole question a stamp answers is what the tree at the sha it
    names contained, and the working tree is the thing being vouched for."""
    out = subprocess.run(["git", "-C", str(ROOT), "show",
                          f"{rev}:architecture/guard-coverage.md"],
                         capture_output=True, text=True)
    assert out.returncode == 0, (
        f"`git show {rev}:architecture/guard-coverage.md` exited {out.returncode} — a "
        f"stamp has to name a commit this repo can resolve")
    return _fenced_blocks(out.stdout)


def _differing_blocks(a: list[str], b: list[str]) -> list[int]:
    """1-based indices of the blocks that are not byte-identical. A page that grew or
    lost a block differs here too, which is the same drift a stamp hides."""
    if a == b:
        return []
    return [i + 1 for i in range(max(len(a), len(b)))
            if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)]


def _newest_stamp(stamp: str) -> tuple[str, str, str]:
    """The LAST dated re-run in a stamp paragraph, as `(date, sha, text from there on)`.

    #2391 clause 3, and the reason is the page's own shape: the stamps are a log, older
    lines stay as history, so the FIRST match is always the stalest sha on the page and
    grading it is how a re-run from 2026-10-06 kept passing a page whose commands had
    been rewritten twice since. The paragraph is wrap-normalised because a reflow is not
    a change of claim, and the tail comes back too so the denominator is read from the
    stamp that is being graded rather than from whichever `**N fenced blocks**` appears
    first.
    """
    norm = " ".join(stamp.split())
    found = list(_STAMP.finditer(norm))
    assert found, "the stamp paragraph carries no `Re-run in full on <date> at <sha>` line"
    date, sha = found[-1].group(1), found[-1].group(2)
    return date, sha, norm[found[-1].start():]


def _stamp_denominator(stamp: str) -> int:
    """The block denominator a stamp's own prose states. A re-run that counts
    nothing cannot age, so #2301 requires the number be written down."""
    m = re.search(r"\*\*(\d+) fenced blocks\*\*", " ".join(stamp.split()))
    assert m, "the stamp states no block denominator"
    return int(m.group(1))


def test_the_stamp_line_names_an_existing_commit_and_what_the_old_one_vouched_for():
    """#2391 clause 1, holding #2022 clause 5 and #2301 clauses 1 and 2 inside it: the
    page's newest dated re-run line names a commit that exists in this repo, states the
    denominator it counted — `**20 fenced blocks**`, the blocks the page actually carries,
    not the open-ended `>= 18` that let a page grow past its own stamp — and says plainly
    that the `ed1cecfd` line above it vouched for two commands the page no longer prints,
    naming both.

    A date alone is not a finding. The reason #2301's stamp misleads is specific — the
    matrix block ran on the system `python`, which has no `httpx`, and §4's family count
    grepped the tuple shape `FAMILIES` had before #1959 — and both were repaired in the
    commit that wrote the stamp, so the sha it names describes commands nobody runs any
    more. A reader who is not told that repeats the mistake on the next re-run.
    """
    text = _text("guard-coverage.md")
    how = _section(text, "How to read this page")
    assert "`219e1314` (2026-10-01)" in how and "all 18 blocks still hit" in how
    assert subprocess.run(["git", "cat-file", "-e", "219e1314^{commit}"], cwd=ROOT).returncode == 0
    blocks = _fenced_blocks(text)
    assert len(blocks) == 20, f"the page carries {len(blocks)} fenced blocks"

    date, stamped, newest = _newest_stamp(how)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), f"{date} is not a dated re-run line"
    assert subprocess.run(["git", "cat-file", "-e", f"{stamped}^{{commit}}"],
                          cwd=ROOT).returncode == 0, f"{stamped} is not a commit here"
    assert _stamp_denominator(newest) == len(blocks) == 20, (
        "the stamp counted a different number of blocks than the page carries")

    assert "vouched for two commands this page no longer carries" in newest, (
        "the stamp does not say plainly that the `ed1cecfd` line above it vouched for "
        "commands the page does not print")
    assert "system `python`" in newest and "family count" in newest, (
        "the stamp names neither of the two commands it says were repaired under the old "
        "stamp: the matrix's interpreter line and §4's family count")


def test_the_stamp_extractor_grades_the_last_re_run_line_and_not_ed1cecfd():
    """#2391 clause 3: with the superseded stamp still on the page, the sha extracted is
    the newest one.

    The paragraph carries `ed1cecfd` first and this round's stamp second, deliberately:
    #2301's line is history worth keeping, and the defect being closed is that a
    `re.search` over that paragraph returns the FIRST match, which since #2391 is always
    the stalest sha on the page. Grading that stale sha is how a re-run dated 2026-10-06
    kept passing a page whose commands had been rewritten twice since.

    The date order is pinned beside the extractor because the two are one claim: "the
    last line" answers "the newest re-run" only while stamps are appended in date order,
    and an editor who pasted an older re-run underneath a newer one would move the graded
    sha back onto a tree whose blocks no longer match — the same defect, arriving by a
    route no node had re-measured. `sorted` over ISO dates is the comparison.
    """
    text = _text("guard-coverage.md")
    how = _section(text, "How to read this page")
    dated = _STAMP.findall(" ".join(how.split()))  # wrap-insensitive on purpose
    assert len(dated) >= 2, (
        f"{len(dated)} dated re-run line(s) on the page; with one, a last-match extractor "
        f"could not be told apart from a first-match one")
    assert ("2026-10-06", "ed1cecfd") in dated, (
        "the #2301 stamp line is gone from the page, so this node can no longer show the "
        "extractor passing the stale sha by")
    dates = [d for d, _ in dated]
    assert dates == sorted(dates), (
        f"the page's stamps run {dates}, so its LAST match is not its newest re-run and "
        f"the sha being graded is an older tree's")

    date, stamped, newest = _newest_stamp(how)
    assert stamped != "ed1cecfd", (
        "the graded sha is the 2026-10-06 stamp, so the extractor is reading the first "
        "match again and the newest re-run on the page goes ungraded")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), f"{date} is not a dated re-run line"

    # The denominator is read from the stamp being graded, not from whichever
    # `**N fenced blocks**` sits earliest on the page.
    two = ("Re-run in full on 2026-10-06 at `ed1cecfd` (#2301) older prose. "
           "Re-run in full on 2026-11-02 at `deadbeef` (#9999): **21 fenced blocks**.")
    assert _newest_stamp(two)[1] == "deadbeef", "the extractor is not reading the last line"
    assert _stamp_denominator(_newest_stamp(two)[2]) == 21, (
        "the denominator was read from the older stamp, so a growing page would keep "
        "its stale count green")
    assert _stamp_denominator("re-run at `abc12345`: **7 fenced blocks**") == 7


def test_the_stamped_tree_s_fenced_blocks_are_the_page_s_and_ed1cecfd_s_are_not():
    """#2391 clause 2: the stamped tree's fenced blocks are byte-identical to the page's,
    all 20 pairs, which is what lets the stamp be said to have vouched for the commands
    this page prints.

    This is also the check that keeps this item honest about its own diff: #2391 adds a
    prose paragraph and may not touch a block, because editing a block would make the sha
    it stamps a tree that does not contain that text. The 2026-10-06 stamp is the rejecting
    case and it is on the page to be read: two of `ed1cecfd`'s blocks are not the page's
    (the matrix's `.venvs/lloyd/bin/python` interpreter line, §4's dict-shaped family
    count), so at the sha that stamp vouches for, two of the commands presented here do
    not exist.
    """
    text = _text("guard-coverage.md")
    how = _section(text, "How to read this page")
    _, stamped, _ = _newest_stamp(how)
    page = _fenced_blocks(text)
    assert len(page) == 20, f"the page carries {len(page)} fenced blocks"

    at_stamp = _fenced_blocks_at(stamped)
    assert len(at_stamp) == len(page) == 20
    assert _differing_blocks(at_stamp, page) == [], (
        f"the stamp names {stamped}, but its fenced blocks differ from the page's at the "
        f"indices above: the re-run vouched for commands this page no longer prints")

    # Clause 2's invariant, read the way the page tells a reader to read everything.
    miss = subprocess.run(["git", "grep", "-n", "other four", "--",
                           "architecture/guard-coverage.md"],
                          cwd=ROOT, capture_output=True, text=True)
    assert miss.returncode != 0 and not miss.stdout, (
        f"`other four` is back on the page ({miss.stdout.strip()}): the hand-counted line "
        f"total #2022 replaced with the three guards it derives")


def test_the_content_check_rejects_ed1cecfd_and_the_hand_carried_ancestor_is_gone():
    """#2391 clause 4: the check that passes the new stamp fails `ed1cecfd`, and the
    hand-carried constant it replaced stays deleted.

    The ancestor node that used to sit here asked whether the stamped sha was a descendant
    of a hand-typed sha, which #2301's own landing showed is not the claim anyone wanted:
    that commit wrote the stamp and repaired two fenced blocks in the same commit, so the
    sha it named was a flawless descendant of text that did not exist yet. A descent
    assertion cannot see a fenced-block edit at the sha it grades; a content comparison can,
    and `ed1cecfd` is the measured example — blocks 4 and 13, nothing else.

    The deletions are part of the clause, so the evidence is this file's own text. Each
    needle is built from two pieces on purpose: written whole it would be sitting in the
    text being searched, and the assertion could then never come out green.
    """
    page = _fenced_blocks(_text("guard-coverage.md"))
    stale = _differing_blocks(_fenced_blocks_at("ed1cecfd"), page)
    assert stale == [4, 13], (
        f"ed1cecfd differs from the page at {stale}, not at blocks 4 and 13 — the matrix "
        f"interpreter line and §4's family count the page says were repaired under that "
        f"stamp. Either the check has stopped seeing a stale stamp or the page's account "
        f"of what #2301 repaired is wrong")

    here = Path(__file__).read_text(encoding="utf-8")
    hand_carried = "GUARD_COVERAGE_" + "LAST_EDIT"
    descent = "merge" + "-base"
    assert hand_carried not in here, (
        "the hand-carried stamp ancestor is back, and with it a freshness check that "
        "cannot see a fenced block edit at the sha it grades")
    assert descent not in here, (
        "a descent assertion on the stamped sha is back beside the content check")


#: The one command on this page that is supposed to print nothing: §3's absence grep
#: for the harness event the reviewer does not branch on. Named by pattern and not by
#: block index, so a reflow of §3 cannot quietly move the only silence a run may allow.
DOCUMENTED_ZERO = "text_delta"

#: …and the file its positive control reads, per the page's own convention that every
#: absence command rides a control that must hit.
ZERO_CONTROL = "app/harness/events.py"


def _block_commands(block: str) -> list[str]:
    """One fenced block's commands: backslash continuations joined into the command
    they are written as, and the trailing `# note` the page rides a line with split off
    so the runner gets a command and not a comment.

    An unquoted `#` anywhere else is refused rather than partitioned, because a
    truncated command that happens to print something is the one result worse than a
    failing one: it would look like the page verifying itself.
    """
    body = block.split("\n", 1)[1][:-4]
    assert not body.endswith("\\"), f"a block ends on a continuation: {block!r}"
    cmds, buf = [], ""
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        buf = f"{buf} {line}".strip() if buf else line
        if buf.endswith("\\"):
            buf = buf[:-1].rstrip()
            continue
        code, mark, _note = buf.partition("  # ")
        assert mark or "#" not in code, f"an unquoted '#' in a command: {buf}"
        cmds.append(code.strip())
        buf = ""
    assert not buf, f"a block ends mid-command: {block!r}"
    assert cmds, f"a fenced block carries no command: {block!r}"
    return cmds


def _run_block(block: str) -> list[tuple[str, int, list[str]]]:
    """Run one block's commands from the repository root — the way the page's own
    stamp paragraph tells a reader to run them — and return one
    `(command_as_run, exit status, stdout lines)` entry per command.

    `.venvs/lloyd/bin/python` becomes `sys.executable` because `.venvs/` is
    git-ignored and so absent from a round worktree: the page's interpreter line is
    the claim being run, not decoration, and left alone it dies at exit 127 with
    `No such file or directory` — which is how #2301's re-run first misread a working
    matrix script as a broken one. Every other command is run exactly as written.
    """
    runs = []
    for cmd in _block_commands(block):
        runnable = cmd.replace(".venvs/lloyd/bin/python", sys.executable)
        out = subprocess.run(runnable, shell=True, cwd=ROOT, capture_output=True,
                             text=True, timeout=600)
        runs.append((runnable, out.returncode, out.stdout.strip().splitlines()))
    return runs


def test_every_fenced_block_on_the_guard_coverage_page_prints_when_run():
    """#2391 clause 5. The page opens with "Run the command; do not trust the
    sentence", and until this node nothing in the suite did: the stamp node checked
    that a sha existed and the §4 nodes ran two commands, while eighteen blocks went
    unexecuted since whoever last typed them. Every fenced block is now extracted, its
    continuations joined, and run from the repository root, and each block has to put
    at least one line on stdout.

    What this can see that a stamp cannot: at `ed1cecfd` the matrix block died on
    `ModuleNotFoundError` under the system `python` and §4's family count printed
    nothing on a grep that had outlived the shape of `FAMILIES` — both under a stamp
    line saying every block hit. A stamp records that a re-run happened somewhere; this
    runs the commands in the tree the reader is standing in, on the interpreter the
    test is running under.

    One silent command is allowed, and only the documented one: §3's
    `git grep -c "text_delta" -- app/harness/action_review.py` is an absence claim, so
    the page's convention requires it ride a positive control, and this node accepts
    the zero only while `git grep -c text_delta -- app/harness/events.py` is printing.
    A zero whose control has gone quiet is a typo, which is the case the control exists
    to catch."""
    blocks = _fenced_blocks(_text("guard-coverage.md"))
    assert len(blocks) == 20, f"the page carries {len(blocks)} fenced blocks"

    runs_by_block: dict[int, list[tuple[str, int, list[str]]]] = {}
    for i, block in enumerate(blocks, 1):
        runs = runs_by_block[i] = _run_block(block)
        broke = [(c, rc) for c, rc, _ in runs if rc not in (0, 1)]
        assert not broke, (
            f"block {i} did not answer, it broke: {broke}. Only a `git grep` miss "
            f"exits 1 here; anything else is a command the page cannot run")
        printed = [ln for _, _, lines in runs for ln in lines]
        assert printed, f"block {i} printed nothing at all: {block!r}"
        silent = [(c, rc) for c, rc, lines in runs if rc == 1 and not lines]
        for cmd, _rc in silent:
            assert DOCUMENTED_ZERO in cmd, (
                f"block {i} carries an undocumented zero: {cmd!r} exits 1 and prints "
                f"nothing, and this page documents exactly one such command — §3's "
                f"`{DOCUMENTED_ZERO}` absence grep")
        if silent:
            assert len(silent) == 1, f"block {i} went quiet twice: {silent}"
            control = [(rc, lines) for c, rc, lines in runs if ZERO_CONTROL in c]
            assert control and control[0][1], (
                f"block {i}'s zero is unpaired: the page says an absence grep rides a "
                f"control that must hit, and nothing naming {ZERO_CONTROL} printed")

    # The matrix block on its own, because it is the one block whose interpreter is
    # git-ignored: it proves the substitution above ran rather than merely was written.
    matrix_at = [i for i, b in enumerate(blocks, 1) if "guard_arm_matrix.py" in b]
    assert len(matrix_at) == 1, (
        f"{len(matrix_at)} blocks name the guard-arm matrix, so the block graded below "
        f"is not the one the page presents as the arm table")
    assert ".venvs/lloyd/bin/python" in blocks[matrix_at[0] - 1], (
        "the matrix block stopped naming the git-ignored interpreter, so this node is "
        "no longer exercising the substitution it exists for")
    at_matrix = runs_by_block[matrix_at[0]]
    assert at_matrix[0][1] == 0, f"the matrix block exited {at_matrix[0][1]}"
    rows = [ln for _, _, lines in at_matrix for ln in lines if ln.startswith("| `")]
    assert rows, (
        f"the matrix printed no dispatch row (only {at_matrix[0][2][:2]!r}), so there is "
        f"no arm table on the page — the script's own failure mode is to print its header "
        f"over nothing")
    assert not any("Traceback" in ln or "No module named" in ln
                   for _, _, lines in at_matrix for ln in lines)

    # The runner can see a silent command at all, which is the whole worth of the
    # assertions above. Proved on a grep of a sentinel no source carries, so it costs
    # one subprocess rather than the page.
    quiet = _run_block("```\ngit grep -c 'lloyd-2391-no-such-sentinel' -- app\n```")
    assert quiet[0][1] == 1 and not quiet[0][2], (
        "the runner reports output for a command that printed nothing, so 'every block "
        "prints' would pass on a page of dead greps")
    joined = _block_commands("```\ngit grep -n foo \\\n-- app\n```")
    assert joined == ["git grep -n foo -- app"], (
        f"continuations are not being joined, so a wrapped command runs as two: {joined}")
    noted = _block_commands('```\ngit grep -c x -- app   # nothing, exit 1\n```')
    assert noted == ["git grep -c x -- app"], (
        f"a command still carries its trailing note, so the runner is executing a "
        f"comment: {noted}")


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


#: The docs #2329 reworded, and the recovery-command shape they now name. A
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
    """#2329 clause 5. Each of the #2329 docs now tells a reader how to get a
    retired doc back, and a recovery command that does not resolve is a second dead
    pointer wearing the clothes of a fix — the failure this whole item exists to
    close, since the sentence it replaced pointed at a directory that is not in the
    tree.

    So every backticked `git show <rev>^:architecture/…` those docs carry is
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
    """#2329 clause 2's mechanical half. These docs are in the corpus the
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


# --------------------------------------------------------------------------- #
# #2517 clause 4: the 06:38:16Z refusal is quoted against committed bytes.
#
# The page quotes two refusals out of the live denial journal and, for #2190's, names the
# witness copy under `backlog/data/` beside the live path. #2362's refusal — the
# `autonomy_write_task` denial that proves the `vault_write` door was open — was cited
# against the live file alone, and that file is bounded by nothing: it is not one of the
# groundskeeper sweep's seventeen stores, and it has grown from the 63 rows #2190 extracted
# to 203. A citation whose denominator moves has no history, so the paragraph now names
# `backlog/data/denials.2362.witness.jsonl` beside it, the way the #2190 sentence above
# already names `backlog/data/denials.jsonl`.
#
# Graded as a citation of the same kind as every other assertion in this file: the row/byte
# figures the paragraph prints are extracted from the prose and compared to the committed
# files, so the failure this catches is the doc drifting from the bytes, not a sentence
# being reworded.
# --------------------------------------------------------------------------- #

#: Where the paragraph says the refusal happened. The anchor is the timestamp because it is
#: the only string in the section that identifies which row is being quoted.
_DENIAL_QUOTE_ANCHOR = "was refused at 06:38:16Z"

#: The live store, which stays cited. The witness supplements it; it does not replace it.
_DENIAL_LIVE_PATH = "`~/lloyd-data/safety/denials.jsonl`"
_DENIAL_WITNESS_PATH = "backlog/data/denials.2362.witness.jsonl"
_FROZEN_DENIALS_PATH = "backlog/data/denials.jsonl"

#: "63 rows / 32,504 bytes", as the paragraph prints it, singular or plural.
_QUOTED_COUNT = re.compile(r"([\d,]+)\s+rows?\s*/\s*([\d,]+)\s+bytes")


def _denial_quote_window(text: str) -> str:
    """The flat prose from the 06:38:16Z refusal through its citation.

    Flat because the page is hand-wrapped at ~78 columns and the sentence whose counts
    matter spans four source lines — the same reason #2362 clause 5 uses `_flat`.
    """
    flat = _flat(text)
    start = flat.find(_DENIAL_QUOTE_ANCHOR)
    assert start >= 0, (
        "the paragraph quoting the 2026-10-05T06:38:16Z refusal no longer says it in the "
        "words this extractor anchors on, so the citation check below read nothing")
    return flat[start:start + 1200]


def _quoted_counts(window: str) -> list[tuple[int, int]]:
    """Every `(rows, bytes)` pair the prose prints, thousands separators stripped."""
    return [(int(a.replace(",", "")), int(b.replace(",", "")))
            for a, b in _QUOTED_COUNT.findall(window)]


def _measured_witness(name: str) -> tuple[int, int]:
    """`wc -l -c` over a committed `backlog/data/` file, in the vault under test."""
    from tests.board_presence import vault_root

    raw = (vault_root() / "backlog" / "data" / name).read_bytes()
    return len(raw.splitlines(keepends=True)), len(raw)


def test_the_refused_write_is_cited_against_committed_bytes():
    """source: architecture/autonomy.md § The dispatch-affecting fields, third surface.
    claim: the paragraph quoting the 2026-10-05T06:38:16Z refusal names
           `backlog/data/denials.2362.witness.jsonl` beside the live journal path, and every
           row/byte figure it prints for the two witnesses equals that file as committed.
    verdict: #2517 landed the witness; the counts are the half that stops the citation
             decaying into a path nobody can open.

    The live path is asserted present as well as the witness because the clause is "beside",
    not "instead of": the journal is where a reader checks the row is still the row, and the
    copy is what makes the quote outlive a store nothing bounds.
    """
    window = _denial_quote_window(_text("autonomy.md"))
    assert _DENIAL_LIVE_PATH in window, (
        "the paragraph no longer cites the live journal at all, which leaves the copy as the "
        "only source and drops the store the row was actually written to")
    assert _DENIAL_WITNESS_PATH in window, (
        "the 06:38:16Z refusal is quoted against `~/lloyd-data/safety/denials.jsonl` only, "
        "which is the state #2517 was filed to end — the live journal is bounded by nothing")
    assert _FROZEN_DENIALS_PATH in window, (
        "the paragraph justifies the separate witness by way of #2190's frozen extract, so it "
        "has to name that file for the distinction to be checkable")

    counts = _quoted_counts(window)
    assert counts, (
        "no `N rows / M bytes` figure was extracted from the refusal paragraph, so the "
        "citation was graded for a path and not for the numbers a reader re-derives")
    witness_measured = _measured_witness("denials.2362.witness.jsonl")
    frozen_measured = _measured_witness("denials.jsonl")
    assert witness_measured in counts, (
        f"the paragraph prints {counts} but the committed #2362 witness is "
        f"{witness_measured}: the quote and the bytes it cites disagree, which is the drift "
        "a bare path cannot show")
    assert frozen_measured in counts, (
        f"the paragraph prints {counts} but #2190's frozen extract is {frozen_measured}")


def test_the_committed_bytes_check_can_fail_on_a_stale_figure():
    """The check above fails on a moved number and on a dropped path, not on a rewording.

    Driven on text this file writes, with the same extractor, so a green citation check is
    never an extraction that quietly found nothing to compare.
    """
    window = _denial_quote_window(_text("autonomy.md"))
    counts = _quoted_counts(window)
    assert len(counts) >= 2, counts

    stale = window.replace("678 bytes", "6,780 bytes")
    assert stale != window, "the fixture did not change the figure, so nothing was tested"
    assert _measured_witness("denials.2362.witness.jsonl") not in _quoted_counts(stale), (
        "a witness figure with one digit moved is invisible to the extractor, so the "
        "comparison above would pass on any number the prose printed")

    dropped = window.replace(_DENIAL_WITNESS_PATH, "a witness copy")
    assert _DENIAL_WITNESS_PATH not in dropped, "the fixture did not drop the witness"
    assert _DENIAL_LIVE_PATH in dropped, (
        "the fixture lost the live path too, so the path assertion would have failed for the "
        "wrong reason and proved nothing about the witness")

    reworded = window.replace(_DENIAL_QUOTE_ANCHOR, "was denied that morning")
    assert _DENIAL_QUOTE_ANCHOR not in reworded, (
        "the anchor survives its own replacement, so the extractor's vacuity guard in "
        "_denial_quote_window would never fire")


# ── #2604 clause 5: rail 4 must name the check, not just assert the property ───

_RAIL4_ANCHOR = "the model never runs `git`"


def test_arch_review_rail_4_names_the_thing_that_enforces_it():
    """Rail 4 of `architecture/arch-review.md` asserts that the model never runs
    `git`, and for its first months the sentence was the whole enforcement: `Bash`
    is kept in the review's tool set by design, `tool_tier("Bash", "git -C <live>
    commit …")` is 1 — the tier of the `git log` the same prompt orders — and
    `DISALLOWED` never names the verb. #2604 made it a lane, and a rail that names
    its checker is the difference between a claim a reader can re-test and a claim
    a reader has to trust.

    So this node asks for the module and function by name, for the event key that
    says when the property broke anyway, and for the sentence to say which of the
    two halves rests on what. It does NOT re-test the behaviour — that is
    `tests/test_safety_gate_discrimination.py`, which is where a wrong answer costs
    anything.
    """
    text = (ARCH / "arch-review.md").read_text()
    assert _RAIL4_ANCHOR in text, (
        "rail 4's headline sentence was reworded away; the node and the rail need "
        "re-reading together")
    rail4 = text[text.index(_RAIL4_ANCHOR):text.index(_RAIL4_ANCHOR) + 1800]

    assert "check_live_git_writes" in rail4, (
        "rail 4 asserts the no-git property without naming the check that enforces "
        "it, which is the state #2604 was filed about")
    assert "app/harness/live_git_writes.py" in rail4, rail4[:200]
    assert "live_head_moved" in rail4, (
        "rail 4 does not say what the run event reports when the property broke "
        "anyway, so the report is undiscoverable from the doc that promises it")
    # Which half rests on what: the commit arrives through the blob check, the
    # no-git rule through the new lane. Both must be attributable, and the node
    # must not pass on a paragraph that merely contains both words.
    assert "commit_reviewed" in rail4, rail4[:200]
    assert "#2604" in rail4, (
        "the rail does not say when the sentence stopped being a prompt rule")

    # And the other half of clause 5: naming a rail is only honest if the same doc
    # says what the rail does *not* cover. `check_live_git_writes` refuses twelve
    # git verbs in one directory; a turn can still dirty production with `sed -i`
    # or a redirected `cp`, and a reader who was told "the model never runs git,
    # there is a lane" will not go looking for the hole the lane left. So the doc
    # has to carry that sentence, name at least one verb that survives the lane,
    # and name what still reports it — otherwise the paragraph is reassurance.
    anchor = "What still rests on the prompt"
    assert anchor in text, (
        "rail 4 gained a rail and the doc never said which part of the sentence "
        "still rests on the model's compliance")
    residual = text[text.index(anchor):text.index(anchor) + 1200]
    assert ("`sed -i`" in residual or "`cp`" in residual), residual[:300]
    for reporter in ("revert_live_ledger_writes", "live_strays"):
        assert reporter in residual, (reporter, residual[:300])


# ── #2640: autonomy.md documented the retry disagreement instead of a default ──
#
# `architecture/autonomy.md` was the artifact that MADE this bug survivable. Its
# failure section said, in print, that "`_DEFAULT_MAX_RETRIES` is 5", that 33 of
# the 39 task files carry `max_retries: 3`, and that on the five files carrying no
# key "the scheduler and the board disagree by definition" — a true sentence that
# described two numbers where one was due, and the reason the wrong one could
# stand for years: the doc, the board, the MCP row and Mission Control all agreed
# with each other on 3, and only the recorder disagreed. #2640 collapsed them onto
# one constant, so the section had to stop narrating a disagreement and state a
# default. What this node pins is that it states the default the CODE has, not a
# number the prose remembers: the expected value is read out of `app/autonomy.py`,
# so a later change to the constant that forgets the page goes red here rather
# than re-creating the drift with the doc now one step further from the truth.

def test_the_autonomy_page_states_one_retry_default_rather_than_a_disagreement():
    text = _text("autonomy.md")
    section = _section(text, "Failure is a state machine, not a log line")
    code = (ROOT / "app" / "autonomy.py").read_text(encoding="utf-8")

    code_m = re.search(r"^_DEFAULT_MAX_RETRIES = (\d+)$", code, re.MULTILINE)
    assert code_m, (
        "the retry default is no longer a plain module constant, so this node is "
        "no longer reading the number the scheduler uses — re-point it, do not "
        "delete it")
    default = int(code_m.group(1))

    doc_m = re.search(r"`_DEFAULT_MAX_RETRIES` is (\d+)", section)
    assert doc_m, (
        "the failure section no longer states the default beside its own constant "
        "name, which is the form that can be checked against the code at all")
    assert int(doc_m.group(1)) == default, (
        f"the page says {doc_m.group(1)} while `app/autonomy.py` says {default}: "
        "the doc has drifted from the scheduler, which is the #2640 defect in its "
        "newest form")

    # One computation, named, and the two projections that now call it. Naming
    # the function is what keeps a reader from re-typing a default again: the
    # sentence has to point at something that exists.
    assert "max_retries_budget" in section, (
        "the page does not name the one function that turns a task dict into a "
        "budget, so the next editor has nothing to call")
    assert "def max_retries_budget(" in code, (
        "the page names a function the scheduler does not have (#1520's defect)")
    for surface in ("app/routers/autonomy.py", "agent_mcp/autonomy.py"):
        assert surface in section, (
            f"the page does not say which projection reads the budget from "
            f"`{surface}`, so its claim of one number is unlocated")

    # The disagreement is gone from the present tense. `disagree` is banned
    # outright because every form of that sentence — "disagree by definition",
    # "the two surfaces disagree" — is the doc giving up rather than fixing;
    # `fm.get` is banned because quoting the reader's own default was how the old
    # paragraph told the story. Past tense about the retired bug is allowed: a
    # page that silently forgets why a constant is shared invites someone to
    # "simplify" it back into a literal.
    assert "disagree" not in section, (
        "the section is back to describing two numbers as a standing state")
    assert "fm.get" not in section, (
        "the section quotes a reader-local default again")
    assert not re.search(r"prints?\s+\*\*3\*\*|\bprint\w*\s+3\b", section), (
        "the section still tells a reader that a surface prints the old default")

    # And the five key-less files are described by what they now show. Before
    # #2640 the same five ids appeared as the exception list that proved the
    # disagreement; the ids staying put while the claim about them flips is the
    # whole of the fix, so the ids are pinned too — a rewrite that quietly drops
    # them would leave no record of which files the property is about.
    assert re.search(r"(?:declaring|carries?|carry) no(?: `max_retries:`)? key",
                     section) or "carry none" in section, (
        "the page no longer describes what happens to a file with no key")
    for task_id in ("#82", "#83", "#86", "#91", "#92"):
        assert task_id in section, (
            f"{task_id} has left the page: the key-less files must be named as "
            "showing the default, not remembered as an exception")
    assert re.search(r"no key is retired at\s+`_DEFAULT_MAX_RETRIES`", section), (
        "the page does not say a key-less file is retired AT the default it "
        "shows, which is the sentence the other four clauses exist to make true")
