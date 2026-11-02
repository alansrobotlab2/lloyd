"""#3544: the skill's claims about the mirror and about its own artifact.

`skills/supply-chain-inventory-regen/SKILL.md` is the procedure autonomy task #99 follows every
week, and a skill is loaded from the live vault rather than from this checkout, so its text is a
surface like any other doc-claim target — with one difference that decides the shape of these
nodes: a run follows it literally. Two claims in it were not just stale but *unusable*:

* the Reporting contract required an item to "name the advisories from the report's
  `sites[].advisories`", and the writer deletes that key before the report is written
  (`one.pop("advisories", None)` at `app/harness/supply_chain_inventory.py:475`,
  `merged.pop("advisories", None)` at `:505`) — so the instruction could not be obeyed;
* *Verifying a refresh ran* described the **frozen** 2026-09-30 mirror as the present ("the
  pre-state to expect until the owner runs the refresh"), after the refresh had run. A reader
  following it took four healthy probes for a broken mirror.

Read, never skipped when absent — `vault_root()` is env-overridable, and a skip on a box that
simply has the vault mounted at a different path would let exactly the wrong skill pass.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.board_presence import vault_root

SKILL = Path("skills") / "supply-chain-inventory-regen" / "SKILL.md"

# Measured off the live mirror 2026-10-30T02:43Z (`OsvScannerSpec().freshness()`, a zip listing
# and a scan of one observed lockfile). These are the numbers clause 4 says the section must
# carry, and they are what a future refresh pass has to re-point.
REFRESHED_STAMP = "2026-10-26"
REFRESHED_ENTRIES = "26,619"
REFRESHED_MAX_ID = "4325"

#: The frozen mirror's readings, which are history and must stay legible as history.
FROZEN_STAMP = "2026-09-30T14:27:46.692030Z"
FROZEN_ENTRIES = "25,781"
FROZEN_MAX_ID = "PYSEC-2026-4008"


def _skill() -> str:
    path = vault_root() / SKILL
    assert path.is_file(), f"{path} is not readable, so every claim below is ungraded"
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """One `###`/`##` section, up to the next heading of any level."""
    start = text.index(heading)
    rest = text[start + len(heading):]
    nxt = re.search(r"^#{2,4} ", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def _table_rows(section: str) -> list[list[str]]:
    """The reference table as cells, header and separator excluded."""
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


# ── clause 3: the contract asks only what a dated report can carry ──────────

def test_the_reporting_contract_no_longer_asks_for_ids_the_writer_deletes():
    """The imperative is gone; the retirement is stated; the live ask is the real one.

    Asserted on the *instruction*, not by a global absence of the key's name: this file now
    quotes the retired ask in order to explain it, and a test that merely grepped for
    `sites[].advisories` would have failed the fix for documenting the bug. What must not
    survive is a sentence that tells a run to name ids from it.
    """
    contract = _section(_skill(), "## Reporting contract")

    assert re.search(r"(?i)must name the advisories", contract) is None, (
        "the contract still commands a run to name advisory ids, which it cannot do")
    assert "name the advisories\n  from the report's" not in contract.replace(
        "  ", " "), "the unsatisfiable ask is present in imperative form"

    # Every surviving mention of the key has to be doing one of two honest jobs: explaining
    # that the writer removes it, or saying ids are not in the report. Anything else is the
    # old instruction wearing quotation marks.
    for line in contract.splitlines():
        if "sites[].advisories" in line:
            assert re.search(r"(?i)unsatisfiable|deleted|removes|not\b|never", line), (
                f"a live-looking instruction names the popped key as a source: {line!r}")

    assert "missed_packages" in contract and "distinct_missed_package_names" in contract, (
        "the contract does not ask for the two fields the report actually carries; the ask "
        "would still be answerable only by a key the writer pops")
    assert "`sites[].advisories`" in _skill(), (
        "the retirement is not explained, so the next pass re-files the same ask")


def test_the_contract_names_the_offline_rescan_as_the_route_when_ids_are_needed():
    """Saying "not in the report" without saying where they are is half a fix.

    The route is the synthesised lockfiles the scan already wrote — `observed/*/requirements.txt`
    — re-scanned offline. Pinned with the argv the shipped spec builds and the env var it
    actually reads, because the two plausible spellings of that variable differ, and the wrong
    one fails with `no offline version of the OSV database is available`: a message that reads
    like a missing mirror and is really a typo.
    """
    contract = _section(_skill(), "## Reporting contract")

    assert re.search(r"observed/\*?/?(\*/)?requirements\.txt", contract) or (
        "observed/" in contract and "requirements.txt" in contract), (
        "the route to the ids is not the synthesised lockfiles, so a future run will go "
        "looking for them somewhere that does not exist")
    assert "OSV_SCANNER_LOCAL_DB_CACHE_DIRECTORY" in contract, (
        "the re-scan block omits the env var the spec's own env() sets")
    # The wrong name may be *discussed* — the section warns about it, and that warning is what
    # saves the next reader an hour. What may not survive is it being *assigned*, which is the
    # form a run copies.
    assert re.search(r"(?<!\w)LLOYD_OSV_DB_CACHE_DIR\s*=", contract) is None, (
        "the block tells a reader to set a variable this codebase does not read; the scan "
        "then exits 127 with `no offline version of the OSV database is available`, which "
        "reads as a missing mirror when only the env var was wrong")
    assert "no offline version of the OSV database is available" in contract, (
        "the failure message that a wrong env var produces is not named, so the misreading "
        "of 'mirror missing' is still available to whoever hits it")
    assert "scan --offline" in contract, (
        "the documented argv is not the offline argv OsvScannerSpec builds, so the re-scan "
        "would reach the network — offline means offline here")


# ── clause 4: the mirror's present state, and the table's live row ──────────

def test_verifying_a_refresh_describes_the_mirror_that_is_on_the_box_now():
    """The section states the refreshed mirror, and the frozen readings only as history.

    The three readings clause 4 names are pinned together because they are one fact stated
    three ways: the stamp the refresh left, the zip's entry count, and the highest advisory id.
    Any one of them missing leaves a reader able to conclude the refresh did not happen — which
    is precisely the misreading the section caused while it described 2026-09-30 as the present.
    """
    text = _skill()
    sec = _section(text, "### Verifying a refresh ran")

    assert REFRESHED_STAMP in sec, (
        f"the section does not state the refreshed mirror's {REFRESHED_STAMP} stamp")
    assert REFRESHED_ENTRIES in sec, (
        f"the section does not carry the live zip's {REFRESHED_ENTRIES} entry count")
    assert re.search(rf"PYSEC-2026-{REFRESHED_MAX_ID}", sec), (
        f"the section does not state a highest id at or beyond {REFRESHED_MAX_ID}, which is "
        "the reading that distinguishes the refreshed mirror from the frozen one")

    assert re.search(r"(?i)refresh has run|#3238.*done|both .?done", sec), (
        "the section never says the refresh has run, so its probes have no expected direction")
    assert re.search(r"(?i)pre-state to expect until", sec) is None, (
        "the frozen mirror is still offered as what a reader should expect to find")

    # Internal consistency, so a swap of the two eras cannot pass by carrying all six numbers:
    # the refreshed database is strictly the larger and the later of the two.
    assert int(REFRESHED_ENTRIES.replace(",", "")) > int(FROZEN_ENTRIES.replace(",", "")), (
        "the constants themselves are inverted, so the ordering checks below are vacuous")
    assert REFRESHED_STAMP > FROZEN_STAMP[:10], "the constants themselves are inverted"
    # Framing is checked over a window of the paragraph rather than one physical line: the
    # prose wraps mid-sentence, so a line-granular check would fail on text that is framed
    # correctly three lines up — and would pass on a line that merely happens to contain the
    # word "frozen". Window ±400 chars: wide enough to reach the clause doing the framing,
    # narrow enough that "this is history" said 2000 chars away cannot vouch for a number.
    flat = " ".join(sec.split())
    for reading in (FROZEN_STAMP, FROZEN_ENTRIES, FROZEN_MAX_ID):
        at = flat.find(reading)
        assert at >= 0, f"the dated reading {reading} went missing with the stale claim"
        window = flat[max(0, at - 400):at + 400]
        assert re.search(r"(?i)history|before it|frozen|unchanged|blindness|2026-10-24", window), (
            f"the frozen-mirror reading {reading} has no dated framing near it, so a reader "
            "takes a 2026-09-30 database for the one on the box")

    # The `-2 days` bound is the one probe that legitimately reads empty on a healthy mirror
    # four days after a refresh, which is why the section has to say so.
    assert "-2 days" in sec and "-7 days" in sec, (
        "the find probe's bound is not stated beside its meaning, so an empty result on a "
        "healthy mirror invites a refresh that nobody needs")


def _artifact() -> dict:
    """The committed artifact's bytes, read beside this file and nowhere else.

    `supply_chain.baseline_path()` takes its root from `_repo_root()`, whose only fallback
    is `app.paths.LLOYD_HOME` — `app/paths.py`'s own parent — so the answer follows
    whichever `app` package the importing process loaded: this tree under the gate, the
    production checkout when a running backend imports it. That is the right behaviour for
    the shipped loader and the wrong shape for a witness, because a node comparing one
    tree's artifact against a vault page cannot say afterwards which tree it read; #2593's
    defect was exactly a baseline read against the wrong tree. So the path comes from
    `__file__`. (`_live_checkout()` is a separate helper, the write guard's root, and is
    not on this read path.)
    """
    import yaml

    path = (Path(__file__).resolve().parents[1]
            / "eval" / "supply-chain" / "baseline.yaml")
    assert path.is_file(), f"{path} is unreadable, so clause 3 of #3613 is ungraded"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_reference_tables_newest_row_is_the_number_that_is_true_today():
    """The table is what a run with no previous report compares against; 150 was a floor.

    Read as a row, not as a substring: `358` must be on the newest-dated-report row, and
    the stale `150` must be gone from it. A `assert "150" not in table` test would pass on
    the wrong table and fail on the right one, which is the opposite of helpful.

    This node used to pin the other row against the literal `150` as well, because that is
    what `eval/supply-chain/baseline.yaml` held. #3613 regenerates the artifact, so the
    artifact row is now derived from the artifact itself —
    `test_the_reference_tables_artifact_row_copies_the_committed_artifact` — and a literal
    in this file would be a second witness nobody re-measures.
    """
    text = _skill()
    rows = _table_rows(_section(text, "## Reporting contract"))
    assert len(rows) >= 3, f"expected a header plus two reference rows, got {len(rows)}"

    newest = [r for r in rows if "inventory-<date>.json" in " ".join(r)]
    assert len(newest) == 1, f"the newest-dated row must be unique, got {len(newest)}"
    missed = newest[-1][-1]
    assert re.search(r"\b358\b", missed), (
        f"the newest dated report's missed-advisory cell reads {missed!r}, not 358 — the "
        "figure 358 is the whole reason this item exists")
    assert "150" not in missed, (
        f"the stale 150 is still the newest-dated row's cell: {missed!r}")
    assert re.search(r"\b906\b", newest[-1][1]), (
        "the component_count cell moved off 906, or the row is not the row this node reads")

    assert "database_freshness" in text, (
        "the skill never names the key that lets two dated reports be compared, which is the "
        "other half of what #3544 added")


def test_the_reference_tables_artifact_row_copies_the_committed_artifact():
    """Clause 3 (#3613): the artifact row is a transcription, so it is graded as one.

    Up to this node's rewrite the assertion was `assert "150" in artifact[-1][-1]`, a
    literal that outlived the number it described. Nothing else in `tests/` reads the
    artifact's counts — `git grep -n "eval/supply-chain/baseline.yaml" -- tests` reaches
    four files and none of them asserts a count — so a regeneration that forgot the row
    landed green while the skill went on quoting a 20-day-old reading to every weekly
    run of autonomy task #99. Deriving all three facts the row states — both cells and the
    reading's timestamp — from the artifact's own bytes removes the copy nobody re-measured, and
    the same derivation catches the opposite error: an artifact row edited up to today's dated
    report, which would tell a reader with no previous report that the committed artifact already
    holds today.

    Read as a row against the repo's own bytes: `_artifact()` resolves beside this file
    rather than through `baseline_path()`, whose root follows whichever `app` package the
    importing process loaded — `_artifact()`'s docstring says why that is the wrong shape
    for a witness. The skill is the vault's, the artifact is this diff's, so the node
    crosses that surface pair on purpose.
    """
    text = _skill()
    rows = _table_rows(_section(text, "## Reporting contract"))
    artifact_rows = [r for r in rows if "baseline.yaml" in " ".join(r)]
    assert len(artifact_rows) == 1, (
        f"the reference table must carry exactly one artifact row to compare, got "
        f"{len(artifact_rows)}")
    row = artifact_rows[-1]

    doc = _artifact()
    obs = doc["observed"]
    components = obs["component_count"]
    missed = obs["advisories_in_observed_set_that_declared_lockfiles_missed"]
    assert (components, missed) != (698, 150), (
        "the artifact still holds the 2026-10-11 reading, so there is nothing to "
        "transcribe and this node is checking a number it was told to reject")

    assert row[1] == str(components), (
        f"the artifact row's component_count cell reads {row[1]!r} while the committed "
        f"artifact's observed.component_count is {components}. The row is a transcription "
        "of the file, not a remembered figure.")
    assert re.fullmatch(rf"{missed}\*?", row[-1].strip()), (
        f"the artifact row's missed-advisory cell reads {row[-1]!r} while the committed "
        f"artifact's observed.advisories_in_observed_set_that_declared_lockfiles_missed "
        f"is {missed}")
    # Compared to the artifact's stamp TRUNCATED TO THE SECOND, not to the day: the row
    # quotes a reading, and `[:10]` passed on a row saying only `2026-10-31` while the
    # file recorded `2026-10-31T22:22:49.743282Z` — the day right, the reading unnamed.
    # That is the half the review rung sent the first commit back for.
    stamp_full = str(doc["observed_generated_at"])
    stamp = stamp_full.split(".")[0]        # …T22:22:49.743282Z -> …T22:22:49
    assert stamp in row[0], (
        f"the artifact row names no reading a reader can check: its source cell is "
        f"{row[0]!r} while eval/supply-chain/baseline.yaml records observed_generated_at "
        f"{doc['observed_generated_at']!r}. {stamp!r} has to appear in the row for it to be "
        "about that regeneration rather than about a day — a date alone is the half of the "
        "row that went stale once already")
    assert "inventory-<date>.json" not in " ".join(row), (
        "the artifact row cites a dated report; the artifact is the other witness")


#: The readings the 7-day-rail section still carries from the era it supersedes: the line the
#: superseded artifact's `verdict` sat on, the mirror stamp that condemned it, and the verdict
#: word itself. They stay in the page on purpose — remove them and the rails read as machinery
#: that never fired — so what is pinned below is their *framing*, never their absence.
RETAINED_HISTORY_READINGS = ("eval/supply-chain/baseline.yaml:11", "2026-09-30", "verdict: stale")

#: Prose that places one of those readings in the past: either the tense that says so, or the
#: date after which it stopped being the present.
HISTORY_FRAMING = re.compile(
    r"(?i)\bbefore\b|\bwas\b|\bthen\b|\bmeant\b|\bstood\b|\bfrozen\b|history|superseded"
    r"|no longer|until 2026-10-31")

#: Prose that says the verdict in the same sentence is the state of the file, not of some
#: artifact it is comparing against.
PRESENT_FRAMING = re.compile(r"(?i)\bnow\b|\bregenerated\b|committed artifact")

#: The bounded window a reading's framing must sit inside: the sentence holding it. A
#: paragraph-level gesture is not enough — the failure this pins is a citation whose own
#: sentence stopped saying when it was true while a "this is history" survived two
#: paragraphs away, and the only window narrow enough to catch that is the sentence itself.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _sentence_containing(text: str, pos: int) -> str:
    """The sentence of `text` holding offset `pos`, split on the page's own sentence breaks."""
    start = max((m.end() for m in _SENTENCE_BREAK.finditer(text, 0, pos)), default=0)
    tail = _SENTENCE_BREAK.search(text, pos)
    return text[start:tail.end() if tail else len(text)]


def test_the_seven_day_rail_section_describes_the_artifact_that_is_committed():
    """Clause 4: the section that explains staleness has to describe the file that exists.

    *What one refresh buys against the 7-day rail* is the section a reader consults to
    decide whether an artifact's verdict is a finding or an aged stamp, and it was written
    while the tracked artifact still read `stale` on a frozen mirror. Two ways it goes wrong
    and both are invisible to a grep of the page: it keeps describing the superseded reading
    as the present — the misreading that cost #2593 twenty days — or it swaps the readings and
    states today's verdict while leaving the old citations framed as current fact. So the node
    reads the section as a section (`_section` stops at the next heading) and asks two
    questions of it, with the expected verdict derived from the artifact rather than repeated
    from the page:

    1. the artifact's own `verdict`, spelled as the artifact spells it, appears exactly once and
       its sentence marks it as the committed file's present state — which also means an artifact
       that went stale cannot have this section vouch for it, because the same section is
       required to frame `verdict: stale` as history;
    2. every occurrence of every reading in `RETAINED_HISTORY_READINGS` sits in a sentence that
       marks it as dated, so moving one citation out of its "what it meant before" sentence
       reddens the node rather than quietly re-dating it.
    """
    doc = _artifact()
    present = f"verdict: {doc['verdict']}"
    flat = " ".join(
        _section(_skill(), "### What one refresh buys against the 7-day rail").split())

    hits = [m.start() for m in re.finditer(re.escape(present), flat)]
    assert len(hits) == 1, (
        f"the section states {present!r} {len(hits)} times; the artifact carries exactly one "
        "verdict, so either the page is describing two states of it or the reading is quoted "
        "somewhere a reader cannot tell present from history")
    here = _sentence_containing(flat, hits[0])
    assert PRESENT_FRAMING.search(here), (
        f"the sentence stating {present!r} says nothing about it being the committed artifact "
        f"now — {here!r}")
    assert f"stale: {doc['stale']}" in here, (
        f"the sentence giving the verdict does not carry the artifact's own `stale: "
        f"{doc['stale']}`, and the verdict without the staleness list is the half a reader "
        "cannot check")

    for reading in RETAINED_HISTORY_READINGS:
        found = [m.start() for m in re.finditer(re.escape(reading), flat)]
        assert found, (
            f"the retained reading {reading!r} has left the section, so a reader cannot see "
            "what the staleness rails did before this artifact and takes the present reading "
            "for the only one the machinery ever produced")
        for at in found:
            here = _sentence_containing(flat, at)
            assert HISTORY_FRAMING.search(here), (
                f"the sentence holding {reading!r} carries nothing placing it in the past, so "
                f"the reading states the artifact's present state — {here!r}")

    # A framing rule that matches anything is a rule that passes anything, so the marker set is
    # checked against a sentence that states the same reading with no history marker in it.
    bare = ("`database.freshness` recorded `2026-09-30` 23.8 days past the rail, so the "
            "verdict here is decayed.")
    assert not HISTORY_FRAMING.search(bare) and not PRESENT_FRAMING.search(bare), (
        f"HISTORY_FRAMING matches {bare!r}, a sentence placing its reading nowhere in time, "
        "which means the framing checks above pass whatever the page says")


def test_the_popped_advisories_key_is_described_where_the_code_still_pops_it():
    """The skill's account of the writer has to stay in step with the writer.

    Cross-surface by construction: this node reads the module and the skill in the same breath.
    If someone stops popping the key — the other fix for the same item — the contract's
    "counts and names travel" sentence becomes the false one, and this is the node that says so
    rather than a reader discovering it while chasing missing ids.
    """
    source = (Path(__file__).resolve().parent.parent
              / "app" / "harness" / "supply_chain_inventory.py").read_text(encoding="utf-8")
    assert 'one.pop("advisories", None)' in source, (
        "the per-site pop is gone, so the skill's reason for not asking for ids is stale — "
        "re-point the contract's bullet and this node together")
    assert 'merged.pop("advisories", None)' in source, "the merged pop moved or went away"
    skill = _skill()
    assert "counts and" in skill and "rows do not" in skill, (
        "the skill no longer says what travels in a dated report and what does not")
