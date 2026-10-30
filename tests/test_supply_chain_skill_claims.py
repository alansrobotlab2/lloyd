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


def test_the_reference_tables_newest_row_is_the_number_that_is_true_today():
    """The table is what a run with no previous report compares against; 150 was a floor.

    Read as a row, not as a substring: `150` is still correct on the artifact row and stays
    there, and `358` must be on the newest-dated-report row. A `assert "150" not in table` test
    would pass on the wrong table and fail on the right one, which is the opposite of helpful.
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

    artifact = [r for r in rows if "baseline.yaml" in " ".join(r)]
    assert artifact and "150" in artifact[-1][-1], (
        "the committed artifact's row must still read 150: it is the number the file holds, "
        "and the fix is not to make the artifact row tell today's story")

    assert "database_freshness" in text, (
        "the skill never names the key that lets two dated reports be compared, which is the "
        "other half of what #3544 added")


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
