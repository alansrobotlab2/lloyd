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


# ── #3614: the accepted pillow/brotli floor ─────────────────────────────────

#: Lead-in of the floor bullet. A named marker rather than a search for the word "floor",
#: because this file already discusses "a floor" elsewhere (`150 was a floor` above), and a
#: prose-word probe would pass on that sentence and fail on the bullet.
FLOOR_MARKER = re.compile(r"^\s*-+\s*\*\*The accepted pillow/brotli floor")

#: The whole of `merged_scan.distinct_missed_package_names` in `inventory-2026-10-29.json`
#: and `inventory-2026-10-31.json` (measured 2026-11-02 by reading both files). All seven are
#: accepted: a bullet naming only pillow and brotli would be void the day it landed, because
#: the other five sit in that same field — `brotli` is the only name the 2026-10-29 reading
#: newly entered, since `pillow` was already in the 2026-10-22 set and what the mirror refresh
#: changed was which *sites* missed it, not which names were known.
ACCEPTED_MISSED_NAMES = ("accelerate", "brotli", "gradio", "pillow", "pip",
                         "transformers", "vllm")

#: The accepted versions and the sites they are accepted at, as one fact per row: the ruling
#: covers the model-serving venvs only, so a version is only the floor *at that site*.
FLOOR_VERSIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pillow", "12.1.1", ("qwen3-tts",)),
    ("pillow", "12.3.0", ("vllm-djev", "vllm-flash-next-main",
                          "vllm-flash-next-main-1012")),
    ("brotli", "1.2.0", ("qwen3-tts",)),
)


def _floor_bullet(contract: str) -> str:
    """The accepted-floor bullet, flattened: marker line to the next top-level bullet.

    Flattening is what makes the windowed assertions below meaningful: the file wraps
    mid-sentence, so a claim stated over four lines is one claim, and a physical-line check
    would fail on text that is correct.
    """
    lines = contract.splitlines()
    start = next((i for i, ln in enumerate(lines) if FLOOR_MARKER.match(ln)), None)
    assert start is not None, (
        "the Reporting contract carries no accepted-floor bullet, so a weekly run has no "
        "name set to compare against and re-files the same 208 every week")
    out = [lines[start]]
    for ln in lines[start + 1:]:
        if re.match(r"^- \S", ln) or re.match(r"^#{2,4} ", ln):
            break
        out.append(ln)
    return " ".join(" ".join(out).split())


def _near(bullet: str, needle: str, *must: str, window: int = 200) -> None:
    """Every occurrence of `needle` has all of `must` within `window` characters."""
    for m in re.finditer(re.escape(needle), bullet):
        span = bullet[max(0, m.start() - window):m.end() + window]
        for word in must:
            assert word in span, (
                f"{needle!r} appears with no {word!r} within {window} characters, so a "
                f"reader cannot tell which {needle!r} this is or what it was accepted at")


def _void_condition(bullet: str, number: int) -> str:
    """One numbered void condition: its label to the next sub-bullet, not the whole floor.

    A slice rather than a `in bullet` check, because a forward reference from the NAMES
    sub-bullet ("only a name outside the seven is news (void condition 2)") mentions the
    label without stating the condition: the label and its own field have to be in the same
    condition for a run to be able to act on either.
    """
    marker = f"**Void condition {number}"
    start = bullet.find(marker)
    assert start >= 0, (
        f"void condition {number} is not labelled in the floor bullet, so the bullet offers a "
        "floor with no named way back into an upgrade")
    nxt = bullet.find(" - **", start + len(marker))
    return bullet[start:nxt if nxt > 0 else len(bullet)]


def test_the_reporting_contract_records_the_accepted_pillow_brotli_floor():
    """#3614: the 208-row rise the mirror refresh made visible is a ruled floor, not a weekly item.

    Autonomy task #99 compares its headline missed-advisory count against the previous dated
    report and files an item on a delta. The 2026-10-29 reading rose `150 → 358` (+208) purely
    because the frozen OSV mirror was refreshed and could see advisories it had never seen, and
    the owed-check pass on #3544 ruled on 2026-10-31: "Accept as a documented floor for the
    model-serving venvs; do NOT upgrade pillow/brotli." Without that ruling recorded *in the
    procedure a run follows*, the next weekly run re-files the same 208 — which is what the item
    was filed for. So this node pins the five things a run has to be able to read off the page:
    the accepted versions per site, the seven names a run compares NAMES against, both void
    conditions, the attribution rule for a moving mirror, and the one number that must never be
    described as pillow/brotli rows.
    """
    bullet = _floor_bullet(_section(_skill(), "## Reporting contract"))

    # ── clause 1: what is accepted, where, on whose authority ────────────────
    for name, version, sites in FLOOR_VERSIONS:
        assert version in bullet, f"the accepted version {version} of {name} is not stated"
        _near(bullet, version, name, *sites)
    assert re.search(r"2026-10-31", bullet), (
        "the floor is undated, so a reader cannot tell it predates a new advisory and has no "
        "reason to re-open it")
    assert "#3544" in bullet and re.search(r"(?i)owed-check", bullet), (
        "the bullet does not name the ruling that authorised it, so the next pass treats the "
        "floor as this skill inventing an exemption rather than a decision someone took")
    assert re.search(r"(?i)do ?NOT upgrade|not upgrade", bullet), (
        "the bullet never says the ruling refused the upgrade, which is the half a future run "
        "is most likely to override")

    # The 208 is a row delta and nothing else. `packages_scanned` for site
    # `vllm-flash-next-main-1012` is also 208 and the reference-table prose quotes it as a
    # package count, so an unframed 208 in this section is ambiguous by construction — and no
    # per-package row count exists at all, because the writer pops the rows that hold them.
    assert re.search(r"208\s+rows?\s+(of|attributed)\s+(to\s+)?(pillow|brotli)", bullet, re.I) is None, (
        "the bullet attributes rows to pillow or brotli; a dated report cannot show that — the "
        "writer pops `advisories` per site and from the merged scan before the file lands")
    assert re.search(r"150\s*(→|->)\s*358", bullet), (
        "the 208 is not framed as the 150→358 row delta between the 2026-10-22 and "
        "2026-10-29 readings, so it reads as a package count")
    for m in re.finditer(r"\b208\b", bullet):
        span = bullet[max(0, m.start() - 220):m.end() + 220]
        assert "150" in span and "358" in span, (
            "a bare 208 in the floor bullet is the other 208 — `packages_scanned` at site "
            "vllm-flash-next-main-1012 — not the missed-advisory delta")

    # ── clause 2: the name set a weekly run compares against ─────────────────
    assert re.search(r"(?i)\bseven\b", bullet), (
        "the accepted set is not stated as a count of seven, so 'a name outside the set' has "
        "no size and a run cannot tell an addition from a re-format")
    for name in ACCEPTED_MISSED_NAMES:
        assert re.search(rf"`{name}`", bullet), (
            f"{name} is in merged_scan.distinct_missed_package_names today and is not on the "
            "floor, so the bullet would be void the day it landed")
    assert re.search(r"(?i)compare\b.{0,80}(package )?names?", bullet), (
        "the bullet does not tell a run to compare package NAMES, which is the only thing a "
        "dated report can carry — the rows are popped before the file exists")
    assert re.search(r"(?i)file (nothing|no item|no backlog item)", bullet), (
        "the bullet never says what to do with a name already on the floor, so the run falls "
        "back on its own delta rule and files the item again")

    # ── clause 3: both void conditions, each labelled beside its own field ───
    cond1 = _void_condition(bullet, 1)
    for word in ("missed_packages", "pillow", "brotli", "lloyd", "user-site"):
        assert word in cond1, (
            f"void condition 1 does not name {word!r}: it has to name the field "
            "(`sites[].missed_packages`), both packages, and the two sites whose exposure the "
            "ruling never accepted — a condition naming only one of them is not decidable")
    cond2 = _void_condition(bullet, 2)
    for word in ("distinct_missed_package_names", "outside"):
        assert word in cond2, (
            f"void condition 2 does not name {word!r}: it has to name the field "
            "(`merged_scan.distinct_missed_package_names`) and say the trigger is a name "
            "OUTSIDE the seven, not any movement in that list")
    for cond in (cond1, cond2):
        assert re.search(r"(?i)(is void|becomes? an upgrade|void and the package|upgrade)", cond), (
            "a void condition names its trigger but not its consequence: the floor stops being "
            "accepted and that package becomes an upgrade")

    # ── clause 4: a rising count, and when it is not an exposure ─────────────
    assert "database_freshness" in bullet, (
        "the attribution rule does not name the field that distinguishes a wider mirror from a "
        "worse tree")
    assert re.search(r"(?i)(differs|moved|changed).{0,160}visibility", bullet, re.S) or \
        re.search(r"(?i)visibility.{0,160}(not|never) (an )?exposure", bullet, re.S), (
        "the rule does not say that a differing stamp makes a rise visibility rather than "
        "exposure — the whole reason the 208 was not an incident")
    assert re.search(r"(?i)has (no|not) [`']?database_freshness[`']? key", bullet), (
        "the absent-stamp arm is missing: every dated report before inventory-2026-10-31.json "
        "has no such key, so a first comparison is None against a stamp, not a stamp change")
    assert re.search(r"(?i)unattributed", bullet), (
        "the absent-stamp arm does not name the third reading, so a run picks one of the two "
        "it was not given")
