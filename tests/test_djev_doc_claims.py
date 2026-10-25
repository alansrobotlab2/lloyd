"""What the djev prose is allowed to say about the seams it describes.

Three items live here, each because a description was read as a fact. The first
is `architecture/djev.md`'s record of the #1324 shadow-log leak: the behaviour
that closes the leak is pinned in `tests/test_djev_shadow_isolation.py`, and
this file pins the *other* half — what the doc may now say, and the numbers
inside that prose.

The second is #1372: `app/djev_shadow.py`, `agent_mcp/vault.py` and
`config.yaml` all described the `rerank` seam as a live one production passes
through, while since #1336 the dispatch has made it unreachable except under
the kill switch. Those assertions sit at the bottom of this file because they
are the same shape as the ones above: they read words out of files rather than
running code, so they need no daemon, no engine and no fixture, and every
needle is a string a later reader can `grep` to re-measure.

The third is #3326: the two comments that size the shadow queue — `config.yaml`'s
`shadow.queue_max` and `djev_shadow.DEFAULTS` — each justified the bound with a
rate counted over three seams, while `djev_shadow.SEAMS` has held four since
2026-09-25 and the fourth writes 99.5% of the log. Those nodes sit with the #1372
ones for the same reason and refuse the same two shapes: a seam count that is not
`len(SEAMS)`, and a calibration that asserts a relation the log refutes.

It matters because the doc is where the item came from. §11 carried "Test runs
write to production's shadow log" as an open gap and named #1324, so any
architecture review that read §11 after the fix would have re-filed the item
from the doc's own text — an open-gap list is a to-do list, and a stale entry in
one is worse than no entry, because it looks like a decision. The same section
now carries counts (36 of 44 `dedupe` rows, 27 of 32 `rerank`, 63 fixture rows
left in the log) that name a fixture by its title; if that title is renamed in
`tests/test_backlog_dedupe.py` the counts silently stop meaning anything, so
those titles are pinned against the module that defines them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_mcp import vault

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "djev.md"

#: The fixture title in `tests/test_backlog_dedupe.py:61` whose 32 rows the doc
#: counts, and the one in `tests/test_backlog_spawn_loop.py:792` that accounts
#: for 4 more. Both are asserted against those modules below, so a rename that
#: would make the doc's counts unfalsifiable fails a test.
FIXTURE_DEDUPE = "http_fetch error body says only the status code on a quarter of calls"
FIXTURE_SPAWN = "graph_refresh is advertised but never called by any tool"

CLOSED_HEADING = "### Closed 2026-09-21 — test runs wrote to production's shadow log (#1324)"


def _section(text: str, heading: str) -> str:
    """From `heading` to the next `## ` line — one section, sliced out."""
    i = text.index(heading)
    j = text.find("\n## ", i + len(heading))
    return text[i: j if j != -1 else len(text)]


def _closed_record(text: str) -> str:
    """The Closed subsection of §11. Asserts its own absence, so a doc that
    dropped the record fails here instead of at some `[1]` index."""
    body = _section(text, "## 11. Known gaps")
    assert CLOSED_HEADING in body, (
        "§11 has no closed record for the shadow-log leak, which means its open-gap "
        "list is still describing #1324 as live — the exact state that re-files it")
    return body.split(CLOSED_HEADING, 1)[1]


def test_the_open_gaps_no_longer_describe_the_shadow_log_leak():
    """Clause 3, first half: `architecture/djev.md` §11 stops listing "Test runs
    write to production's shadow log" as an open gap, and stops pointing at the
    item that closed it.
    """
    open_part = _section(DOC.read_text(), "## 11. Known gaps").split(CLOSED_HEADING, 1)[0]
    assert "Test runs write to production's shadow log" not in open_part, (
        "§11 still lists the leak as an open gap")
    assert "#1324" not in open_part, "the open gaps still point at the item that closed it"
    assert "fixture" not in open_part.lower(), (
        "the open gaps still carry the fixture-row complaint this item closed")


def test_the_open_gaps_no_longer_advertise_the_drift_message_flag_gap():
    """#3113 closed the half of this item that is prose: §11 used to carry
    "`SchemaDrift`'s message names `replay.py --calibrate`, which does not
    exist" as an OPEN gap, and a page that keeps advertising a gap the code no
    longer has is what makes the next arch review re-file the item.

    The changelog line that filed #3113 is history and stays, so the test pins
    both halves: `--calibrate` must survive on this page exactly once, on the
    changelog line that names #3113, and must appear nowhere among the open
    gaps. Without the count the assertion is only half a rail — deleting the
    open bullet AND the changelog line would otherwise pass it.
    """
    text = DOC.read_text()
    open_part = _section(text, "## 11. Known gaps").split(CLOSED_HEADING, 1)[0]
    assert "--calibrate" not in open_part, (
        "§11 still lists the invented `--calibrate` flag as an open gap (#3113)")
    assert "SchemaDrift" not in open_part, (
        "§11's open gaps still name the SchemaDrift message #3113 fixed")
    mentions = [ln for ln in text.splitlines() if "--calibrate" in ln]
    assert len(mentions) == 1, (
        f"`--calibrate` should remain on this page only as the changelog line "
        f"that filed #3113; found {len(mentions)} mentions")
    assert "#3113" in mentions[0], (
        "the surviving mention is not the changelog line that filed #3113, so "
        "the fix has no record of where the claim came from")


def test_the_closed_record_carries_the_mechanism_the_check_and_the_residue():
    """Clause 3, second half: the replacement has to be usable, not merely
    present. Each needle is something a later reader needs and cannot re-derive:
    which fixture the counts refer to, why the paths are redirected instead of
    muted (reverting to `LLOYD_DJEV_SHADOW=0` would make
    `test_djev_rerank_arm.py::test_the_eval_mutes_the_shadow_recorder` pass with
    the `eval/run_eval.py` line under it deleted), why the session fixture never
    restores (a restore at session end is the drain window), the command that
    shows the leak is gone, and the rows still in the file poisoning
    `eval/djev/replay.py --floors`.
    """
    closed = _closed_record(DOC.read_text())
    for needle in ("`tests/conftest.py::_isolate_djev_shadow`",   # the fix, by name
                   "does not restore them",                       # the mechanism that matters
                   FIXTURE_DEDUPE,                         # the fixture the counts name
                   "Paths are redirected rather than",            # the decision…
                   "`LLOYD_DJEV_SHADOW=0` set",                   # …and its reason
                   "test_the_eval_mutes_the_shadow_recorder",     # the test a mute would vacuate
                   "$FH/.local/state/lloyd-djev/shadow.jsonl",    # the check
                   "**does not exist**",                          # its expected result
                   "**What is left, and it is a person's call.**"):  # the residue, still owed
        assert needle in closed, f"the closed record no longer carries {needle!r}"


def test_the_closed_record_still_names_the_two_things_reading_the_log():
    """Closing the leak does not clean the corpus, and the doc must not read as
    if it had. Two writers survive this change: the fixture rows already in the
    file, and the automod gate's tests rung, which runs pytest with `HOME` set to
    the real home — `scripts/automod/gate.py::_child_env` redirects the automod
    and guardian state dirs and sets no djev path (`grep -n DJEV
    scripts/automod/gate.py` → 0 hits). Under the gate a repointed `SHADOW_LOG`
    proves nothing about `Path.home()`, because `Path.home()` is the real home
    there; the property that holds is the line count of the real file across the
    run. Either fact goes missing and the next reader of §9.3 step 5 gets a
    fixture-dominated floor or a false clean bill.
    """
    closed = _closed_record(DOC.read_text())
    assert "63" in closed, "the closed record no longer says how many fixture rows are still in the log"
    assert "replay.py --floors" in closed, "the closed record no longer names what the residue poisons"
    assert "_child_env" in closed, "the closed record no longer names the gate as the un-repointed runner"


def test_the_calibration_procedure_points_at_the_quarantine_first():
    """§9.3 step 5 is where a practitioner goes looking, not §11. It now says the
    floor comes *after* quarantining the pre-fix rows, so the calibration
    procedure cannot be followed in the order that produces a fixture-dominated
    floor.
    """
    doc = DOC.read_text()
    step = doc[doc.index("### 9.3 Adding a seam"):doc.index("### 9.4 Turning a schema")]
    assert step.count("#1324") == 1, "§9.3 step 5 no longer points at the quarantine the floor needs first"
    assert "fixture" in step, "§9.3 step 5 no longer says why quarantine precedes --floors"


def test_the_fixture_titles_the_doc_counts_still_exist():
    """The doc's counts are only checkable while the titles they name are still
    the titles the fixtures write. `tests/test_backlog_dedupe.py` and
    `tests/test_backlog_spawn_loop.py` write the first string into `meta.name`
    and the second into the backlog `name` of every row their dedupe seam
    shadows, which is how 36 of the log's 44 `dedupe` rows are attributable.
    Rename either and the doc's numbers become prose nobody can re-measure.
    """
    doc = DOC.read_text()
    closed = _closed_record(doc)
    assert FIXTURE_DEDUPE in closed, "the closed record no longer names the fixture it counts"
    assert FIXTURE_DEDUPE in (ROOT / "tests" / "test_backlog_dedupe.py").read_text(), (
        f"{FIXTURE_DEDUPE!r} is no longer the fixture title in test_backlog_dedupe.py, "
        "so the row counts in the closed record are no longer checkable")
    assert FIXTURE_SPAWN in (ROOT / "tests" / "test_backlog_spawn_loop.py").read_text(), (
        f"{FIXTURE_SPAWN!r} is no longer the fixture title in test_backlog_spawn_loop.py")


# ── #1372: the rerank seam's prose says when it can fire ────────────────────
#
# The item was filed from prose, not from code, and what follows is four
# PROBE-TIME readings (2026-09-23), not a present-state claim. At that probe
# `app/djev_shadow.py` opened with "Three seams call this — reranking in
# `agent_mcp/vault.py`", `config.yaml` called the hook "The lead seam: both
# `vault_recall` and `memory_ops.recall` pass through it", and
# `architecture/djev.md` said the seam runs "on every recall". Since #1336 none
# of those is true: djev IS the recall's ranker, so the hook sits under an `elif`
# the first arm never reaches. Two things have moved since, which is exactly why
# neither may be kept here as a fact: #1628 re-worded that `config.yaml` comment
# to name the firing condition instead of a rank (which is what lets the guard
# below cover the file at all), and the row stream is a log that grows — what the
# probe read was `rerank` newest at 2026-09-21T08:57:40Z with `dedupe` and
# `entity` still flowing, and the live figure is whatever `djev_status`'s
# `shadow.seam_log` reports per seam, which §6.1 now points to rather than
# restating. A description that presents a structurally dark seam as a live one
# is what makes its silence unreadable — so the descriptions are asserted, not
# just corrected once.

SHADOW_SRC = ROOT / "app" / "djev_shadow.py"
VAULT_SRC = ROOT / "agent_mcp" / "vault.py"
#: The live config. Its `shadow.seams` comments are the seam description no round
#: can repair: `config.yaml` sits in `scripts/automod/spec.py`'s `DENIED_GLOBS`,
#: so a self-modification round that read a wrong sentence in it would have to
#: hand the fix to a human, while the same sentence in `vault.py` is one edit
#: away. That asymmetry is the reason to guard the file, not a reason to skip it —
#: and #3084 is the second item filed from this prose after #1628, so the needle
#: is the reader.
CONFIG_SRC = ROOT / "config.yaml"


def _module_docstring(path: Path) -> str:
    """The module docstring of `path`, by splitting on the delimiters rather
    than importing: this test is about the words, and it must not need a live
    qmd daemon or djev engine to read them."""
    text = path.read_text()
    body = text.split('"""', 2)
    assert len(body) >= 3, f"{path.name} no longer opens with a module docstring"
    return body[1]


def test_the_shadow_docstring_names_when_the_rerank_seam_can_fire():
    """Clause 3. The replacement has to carry the condition, not just drop the
    claim: the kill switch pulled to `qmd` with the engine still answering, and
    structurally dark while djev is the ranker."""
    head = _module_docstring(SHADOW_SRC)
    assert "Three seams call this" not in head, (
        "the module still opens by naming three live callers, one of which "
        "cannot reach the recorder while djev is the recall's ranker")
    for needle in ("kill switch", "qmd", "dark"):
        assert needle in head, (
            f"the module docstring no longer says the rerank seam is dark "
            f"except under the kill switch to qmd — it lost {needle!r}")


#: Every file that describes the seams: the recorder, the caller, the doc the
#: #1324 record above is guarded in, and the live config. #1372 is a defect about
#: descriptions, so the scan covers every surface that carries one rather than
#: only the two the clause names — and the config is the surface a round cannot
#: edit, which is where a description is most likely to be found stale and least
#: likely to be fixed.
SEAM_DESCRIPTIONS = [SHADOW_SRC, VAULT_SRC, DOC, CONFIG_SRC]

#: The claims that make a structurally dark seam read as a live one: a rank
#: ("lead") and a frequency ("on every recall" / "the three shadow seams" — a
#: count of live callers). Both were in the prose #1372 was filed from, and both
#: describe volume where the truth is a condition.
LIVE_SEAM_NEEDLES = ["lead seam", "the three shadow seams"]


def _assert_no_live_seam_claims(text: str, *, where: str, frequency: bool = True) -> None:
    """Refuse any text that describes the rerank seam by a rank or a frequency.

    A helper rather than a node body because the check has two callers proving
    different things: the parametrised node runs it over every guarded file, and
    `test_the_needle_check_refuses_the_sentence_it_exists_to_refuse` feeds it the
    sentence it exists to refuse. A guard whose only green path is "the corpus
    happens to contain no match" cannot be told apart from one that cannot match
    anything, and the file that most needs this one is the file no round may edit.

    `frequency=False` is `vault.py`'s exemption, stated by the node that asks for
    it: it says "on every recall" about the #1335 latency of the graph arm, which
    is true of it.
    """
    lowered = text.lower()
    needles = list(LIVE_SEAM_NEEDLES) + (["on every recall"] if frequency else [])
    for needle in needles:
        assert needle not in lowered, (
            f"{where} still says {needle!r} about the rerank seam. While "
            "djev is the recall's ranker the dispatch never reaches it, so state "
            "the condition it fires under instead of a volume or a rank.")


@pytest.mark.parametrize("path", SEAM_DESCRIPTIONS,
                         ids=["djev_shadow", "vault", "djev.md", "config.yaml"])
def test_no_djev_description_calls_rerank_a_seam_production_reaches(path):
    """Clause 2, second half. "Lead seam" was the phrase in two of these files —
    `app/djev_shadow.py`'s eval-muting paragraph and
    `vault._djev_shadow_rerank`'s own docstring ("The lead seam of the three,
    because every `vault_recall` passes through it") — and "on every recall" was
    in the doc, two sections from the one that knew better (§8.1: "The shadow
    seam does not run when djev ranks"). Those are the claims that make a dark
    seam read as a quiet one, so the needles are asserted, not just corrected
    once: a rank or a frequency is where a stale seam description shows up.

    `config.yaml`'s case was the gap #3084 closes: its `shadow.seams` comment
    carried the lead-seam sentence until #1628, and nothing in the suite read the
    file, so the correction was invisible to every later round."""
    _assert_no_live_seam_claims(path.read_text(), where=path.name,
                                frequency=path != VAULT_SRC)
    if path is VAULT_SRC:
        import inspect
        hook = inspect.getsource(vault._djev_shadow_rerank).lower()
        for needle in ("on every recall", "passes through it", "unconditional"):
            assert needle not in hook, (
                f"`_djev_shadow_rerank` still claims it happens {needle!r}, "
                "which is the sentence that made the seam look live")


def test_the_needle_check_refuses_the_sentence_it_exists_to_refuse():
    """The guard's red side, which is the only thing distinguishing it from a
    no-op. `config.yaml` is in `DENIED_GLOBS`, so the way the lead-seam sentence
    gets back into it is a human edit no round ever sees, and the sole evidence
    the guard would leave is a green parametrisation — which is what #1372's
    correction had for company until now. So the helper is fed the sentence it was
    written for (the one the comment block above quotes, and what
    `config.yaml`'s `shadow.seams.rerank` comment replaced it with in #1628), and
    the assertion is on the needle that bit, not merely on raising.

    The paired green half is the same file's real text: without it, this node
    would only prove the helper is noisy."""
    lead_seam_comment = ("# Reranking, the lead seam: both `vault_recall` and "
                         "`memory_ops.recall` pass through it.")
    with pytest.raises(AssertionError) as raised:
        _assert_no_live_seam_claims(lead_seam_comment, where="config.yaml")
    assert "lead seam" in str(raised.value), (
        f"it raised, but not on the phrase — the needles have moved and the "
        f"parametrised node is now guarding something else: {raised.value}")
    _assert_no_live_seam_claims(CONFIG_SRC.read_text(), where="config.yaml")


#: Both places that size the shadow queue with a frequency sentence: the live
#: config and the recorder's own `DEFAULTS`. #3326 found the same stale claim on
#: each — `config.yaml`'s "a few hundred times a day" and `djev_shadow.py`'s
#: "140-300 times a day" — so one parametrised trio guards both, and a third such
#: comment inherits the three rather than escaping them.
QUEUE_MAX_COMMENTS = [CONFIG_SRC, SHADOW_SRC]

#: The probe #3326 landed on and the three readings it took from
#: `djev_status`'s `shadow.seam_log`: the aggregate over the seven full UTC days
#: 2026-10-18..2026-10-24 (9,241 rows, ~385 an hour), and `action_review`'s share
#: of the store (303,434 of 304,832 rows). Pinned as literals on purpose: these
#: sentences are readings, and a re-read that updates either comment has to move
#: the date and the figures here with it, which is the one thing that keeps a
#: dated rate from quietly becoming a present fact.
SEAM_PROBE_DATE = "2026-10-25"
SEAM_ROWS_PER_DAY = "9,241"
SEAM_ROWS_PER_HOUR = "385"
SEAM_DOMINANT_SHARE = "99.5%"

#: The rates the two comments shipped with. Named because both are refused over
#: the comment's whole text, and a third spelling of "we did not measure" would
#: otherwise slip through the pair.
STALE_QUEUE_MAX_RATES = ("few hundred", "140-300")

#: The calibration sentence #3326 refutes, by its own words: it asserted the
#: bound outlasts an hour of traffic, and a measured hour (~385 rows) has been
#: larger than `queue_max` since the fourth seam's first row, 2026-09-25T18:45Z.
OLD_HOUR_CLAIM = "deep enough to hold an hour"

_DAY_FIGURE = re.compile(r"\b([\d,]+) rows/day\b")
_HOUR_FIGURE = re.compile(r"\b([\d,]+) rows/hour\b")
_PROBE_DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")


def _queue_max_comment(path: Path) -> str:
    """The comment block sitting directly above a `queue_max` declaration.

    Located from the value and walked backwards over comment lines, so re-wrapping
    the prose or inserting a key above it moves the window along rather than
    leaving the check reading a neighbour's comment. Both surfaces put their
    justification in exactly this position; the recorder's `#:` and the config's
    `#` strip to the same prose, which is wrapped in places and so is flattened
    before any figure is read out of it.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    at = [i for i, ln in enumerate(lines)
          if ln.strip().startswith(("queue_max:", '"queue_max"'))]
    assert at, f"`queue_max` is gone from {path.name}, so this check has no subject"
    end = at[0]
    start = end
    while start > 0 and lines[start - 1].strip().startswith("#"):
        start -= 1
    prose = " ".join(ln.strip().lstrip("#").lstrip(":").strip()
                     for ln in lines[start:end])
    prose = " ".join(prose.split())
    assert prose, (
        f"nothing above `queue_max` in {path.name} justifies the bound any more. "
        "The comment that did was the subject of #3326")
    return prose


def _assert_a_dated_probe(flat: str, *, where: str) -> None:
    """Clause 1: a rate a reader can re-measure, not a rate to trust.

    Three things, no more: the unmeasured claim is gone, the reading carries the
    day it was taken, and it names `shadow.seam_log` as the place to re-take it.
    The date and the per-day figure are both pinned, because a sentence that keeps
    a date while swapping the number under it is how a dated reading becomes a
    present fact again — which is the failure this whole item is about.
    """
    for stale in STALE_QUEUE_MAX_RATES:
        assert stale not in flat, (
            f"{where} sizes `queue_max` by {stale!r} again. That figure was #3326: "
            "read `shadow.seam_log`, then write the measurement with its date.")
    dated = _PROBE_DATE.search(flat)
    assert dated, (f"{where}'s `queue_max` comment states a rate with no probe date, "
                   "so no reader can tell how stale the rate is")
    assert dated.group(0) == SEAM_PROBE_DATE, (
        f"{where}'s comment is dated {dated.group(0)} but still quotes the "
        f"{SEAM_PROBE_DATE} readings. Either re-measure or name which probe ran")
    assert "seam_log" in flat, (
        f"{where}'s comment quotes a rate without naming `shadow.seam_log` to "
        "re-measure it against, so the sentence cannot be checked")
    day = _DAY_FIGURE.search(flat)
    assert day, (
        f"{where}'s `queue_max` comment states no `N rows/day` figure, so there is "
        "no rate here to re-measure against `seam_log` — #3326's complaint was a "
        "frequency nobody could re-check, not a frequency nobody liked")
    assert day.group(1) == SEAM_ROWS_PER_DAY, (
        f"{where}'s comment quotes {day.group(1)} rows/day while the "
        f"{SEAM_PROBE_DATE} probe read {SEAM_ROWS_PER_DAY} over "
        "`shadow.seam_log`. The item's own ~28-40/day is refused here: it counted "
        "three of the four seams")


def _assert_the_seam_count_and_its_owner(flat: str, *, where: str) -> None:
    """Clause 2: the comment counts the seams the recorder declares.

    Read with the same `_SEAM_COUNT` that pins §6.1 of `architecture/djev.md` and
    compared against `len(djev_shadow.SEAMS)` — never against a literal four — so
    a fifth seam makes these comments wrong rather than making this check agree
    with them quietly. An absent count is a failure here even though the doc's own
    node lets a page name the set uncounted: for a queue bound, the count IS the
    argument, and a count of three sat unexamined in this comment for a month.
    """
    from app import djev_shadow

    seams = tuple(djev_shadow.SEAMS)
    found = _SEAM_COUNT.findall(flat)
    assert found, (
        f"{where}'s `queue_max` comment puts no count beside `shadow seams`, so it "
        f"cannot be checked against `SEAMS` = {seams}")
    counted = [int(digits) if digits else _NUMBER_WORDS[word.lower()]
               for digits, word in found]
    stale = [c for c in counted if c != len(seams)]
    assert not stale, (
        f"{where}'s comment counts {stale} shadow seams while `SEAMS` holds "
        f"{len(seams)}: {seams}. The comment that sizes the queue has to count the "
        "seams that fill it")
    assert "action_review" in flat, (
        f"{where}'s comment names no dominant seam. `action_review` writes "
        f"{SEAM_DOMINANT_SHARE} of the store, and a rate quoted without it is the "
        "error #3326 was filed on")
    assert SEAM_DOMINANT_SHARE in flat, (
        f"{where}'s comment names `action_review` without its {SEAM_DOMINANT_SHARE} "
        "share, which is the figure that makes it the dominant seam rather than a "
        "fourth name on a list")


def _assert_the_bound_is_a_reporting_bound(flat: str, bound: int,
                                          *, where: str) -> None:
    """Clause 3: no sentence may claim the bound outlasts an hour of traffic.

    Arithmetic against the live `queue_max`, not a phrase match, because the
    failure to catch is the one #3326 nearly shipped: an hour figure small enough
    to leave the old story standing. The day figure has to be a day OF the hour
    figure, so a three-seam rate and a four-seam rate cannot share a sentence, and
    what the bound is actually for — reporting overflow rather than absorbing it —
    has to survive as a reading on the probe's date.
    """
    assert OLD_HOUR_CLAIM not in flat, (
        f"{where}'s comment reasserts that `queue_max` outlasts an hour of traffic. "
        f"A measured hour ({SEAM_ROWS_PER_HOUR} rows, {SEAM_PROBE_DATE}) is larger "
        f"than {bound}, which is the whole finding")
    hour = _HOUR_FIGURE.search(flat)
    assert hour, (
        f"{where}'s comment no longer states an `N rows/hour` figure, so the bound's "
        "calibration is back to an unstated rule of thumb")
    assert hour.group(1) == SEAM_ROWS_PER_HOUR, (
        f"{where}'s comment quotes {hour.group(1)} rows/hour while the "
        f"{SEAM_PROBE_DATE} probe read {SEAM_ROWS_PER_HOUR}")
    hourly = int(hour.group(1).replace(",", ""))
    day = _DAY_FIGURE.search(flat)
    assert day, f"{where}'s comment states an hour with no day beside it"
    daily = int(day.group(1).replace(",", ""))
    assert daily >= 24 * hourly, (
        f"{where}'s comment puts {daily} rows/day beside {hourly} rows/hour, and "
        f"{daily} is not a day of that hour ({24 * hourly}): the two figures are "
        "from different readings, which is how a three-seam rate and a four-seam "
        "rate end up in one sentence")
    assert hourly > bound, (
        f"{where}'s comment quotes {hourly} rows/hour against a `queue_max` of "
        f"{bound} and so still leaves the bound looking deeper than an hour. If the "
        "log really does run under it again, rewrite this calibration rather than "
        "quietly shrink the figure")
    assert "overflow" in flat, (
        f"{where}'s comment dropped what the bound is for: shallow enough that a "
        "real backlog reports overflow instead of absorbing it")


@pytest.mark.parametrize("path", QUEUE_MAX_COMMENTS,
                         ids=["config.yaml", "djev_shadow"])
def test_the_queue_max_comment_states_a_dated_probe_not_a_frequency(path):
    """Clause 1, on both surfaces (and so clause 5 for the recorder's `DEFAULTS`
    comment, whose "140-300 times a day" is the same defect spelled twice).

    The comment exists to justify a number, and until #3326 it justified it with a
    frequency nobody had measured: "a few hundred times a day" was 5-7x high for
    the three seams it named and ~230x low for the four that actually run. What
    replaces an unmeasured claim has to be a measurement plus the way back to it,
    which is the date and the source."""
    _assert_a_dated_probe(_queue_max_comment(path), where=path.name)


@pytest.mark.parametrize("path", QUEUE_MAX_COMMENTS,
                         ids=["config.yaml", "djev_shadow"])
def test_the_queue_max_comment_counts_the_seams_the_recorder_declares(path):
    """Clause 2. Both comments counted three seams; `SEAMS` gained its fourth on
    2026-09-25 and has carried it for a month, during which the fourth wrote
    303,434 rows to the store's 304,832.

    The count is derived from `djev_shadow.SEAMS` through the regex §6.1's node
    already uses, so this node cannot be kept green by a comment that learned to
    say "four" today and means three after the next seam lands."""
    _assert_the_seam_count_and_its_owner(_queue_max_comment(path),
                                         where=path.name)


@pytest.mark.parametrize("path", QUEUE_MAX_COMMENTS,
                         ids=["config.yaml", "djev_shadow"])
def test_the_queue_max_comment_no_longer_claims_the_bound_outlasts_an_hour(path):
    """Clause 3, on both surfaces. The calibration's own words were the item's
    stronger half: "a queue deep enough to hold an hour of traffic would hide the
    overflow this bound exists to report" asserts that 256 exceeds an hour, and a
    measured hour is ~385 rows.

    The bound is read live — `DEFAULTS` for the recorder, the parsed document for
    the config — so this is also the check that refuses a re-tune done under cover
    of a comment: move `queue_max` and the relation this asserts moves with it."""
    from app import djev_shadow

    if path is SHADOW_SRC:
        bound = int(djev_shadow.DEFAULTS["queue_max"])
    else:
        import yaml
        bound = int(yaml.safe_load(path.read_text(encoding="utf-8"))
                    ["djev"]["shadow"]["queue_max"])
    _assert_the_bound_is_a_reporting_bound(_queue_max_comment(path), bound,
                                          where=path.name)


def test_the_probe_sits_in_comments_and_no_value_moved():
    """Clause 4: the entire edit is prose the YAML scanner never sees.

    Three witnesses, because what is being asserted is a landing route. Every line
    carrying one of the probe's figures is a whole-line comment. The bound the
    loader sees is still the recorder's default, so nothing was re-tuned on the way
    past the comment. And `comment_only_change` — the real one, the same call
    `COMMENT_ONLY_GLOBS`'s promotion route makes — has to call this file
    comment-only against its own comment-stripped body, which is where prose that
    had leaked into a quoted scalar or a block string would change the token
    stream and fail. Without that third witness the claim "this needed no human"
    rests on this round's diff alone and proves nothing about the shipped file.
    """
    import yaml

    from app import djev_shadow
    from scripts.automod.spec import COMMENT_ONLY_GLOBS, comment_only_change

    text = CONFIG_SRC.read_text(encoding="utf-8")
    assert CONFIG_SRC.name in COMMENT_ONLY_GLOBS, (
        "`config.yaml` left `COMMENT_ONLY_GLOBS`, so a comment here is a human "
        "edit again and this item's route is gone")
    for ln in text.splitlines():
        if any(needle in ln for needle in (SEAM_PROBE_DATE, "seam_log",
                                           SEAM_ROWS_PER_DAY, SEAM_ROWS_PER_HOUR,
                                           SEAM_DOMINANT_SHARE)):
            assert ln.strip().startswith("#"), (
                f"a probe figure reached outside a whole-line comment: {ln!r}")
    bound = int(yaml.safe_load(text)["djev"]["shadow"]["queue_max"])
    assert bound == int(djev_shadow.DEFAULTS["queue_max"]), (
        f"`config.yaml` now sets `queue_max` to {bound} while the recorder defaults "
        "to a different number. #3326 excludes re-tuning the bound")
    assert f"queue_max: {bound}" in text, (
        "`queue_max` no longer reads as a bare integer, so the value moved")
    stripped = "\n".join(ln for ln in text.splitlines()
                         if not ln.strip().startswith("#"))
    ok, why = comment_only_change(stripped, text)
    assert ok, (f"`config.yaml` does not survive its own comment-only test: {why}")


def test_the_queue_max_prose_check_refuses_the_rates_it_exists_to_refuse():
    """The red side of all three guards, and the only thing telling their greens
    apart from silence. Two seeds, both real text rather than straw.

    The first is the shipped comment as #3326 found it, quoted verbatim from
    `config.yaml` at base 414868f5: the guard must refuse it on the stale rate by
    name. The second is #3326's OWN clause 1 — the "~28-40/day" its triage
    refuted, which carries a date, carries `seam_log` and carries a per-day
    figure, and is still wrong by ~230x because it counts three of four seams.
    Only the count and the hour relation catch that one, which is exactly why
    those two checks are not folded into the first. The paired green half is the
    shipped comment itself, so a guard that had gone deaf would fail here too.
    """
    shipped = ("Small on purpose. The three seams fire a few hundred times a day "
               "between them and a decision is ~40 ms plus prefill, so a healthy "
               "backlog is single digits; a queue deep enough to hold an hour of "
               "traffic would hide the overflow this bound exists to report.")
    with pytest.raises(AssertionError) as stale_rate:
        _assert_a_dated_probe(shipped, where="config.yaml")
    assert "few hundred" in str(stale_rate.value), (
        f"it raised, but not on the stale rate — the needles have moved: "
        f"{stale_rate.value}")
    with pytest.raises(AssertionError) as hour_claim:
        _assert_the_bound_is_a_reporting_bound(shipped, 256, where="config.yaml")
    assert OLD_HOUR_CLAIM in str(hour_claim.value), (
        f"it raised, but not on the calibration it exists to refuse: "
        f"{hour_claim.value}")

    item_own_clause = ("Sized 2026-10-25 from `shadow.seam_log`: the three shadow "
                       "seams write 40 rows/day, so a queue deep enough to hold an "
                       "hour of traffic would hide the overflow this bound reports.")
    with pytest.raises(AssertionError) as wrong_rate:
        _assert_a_dated_probe(item_own_clause, where="config.yaml")
    assert SEAM_ROWS_PER_DAY in str(wrong_rate.value), (
        f"the item's own ~28-40/day got past the figure check: {wrong_rate.value}")
    with pytest.raises(AssertionError) as wrong_count:
        _assert_the_seam_count_and_its_owner(item_own_clause, where="config.yaml")
    assert "SEAMS" in str(wrong_count.value), (
        f"the three-of-four seam count got past the count check: {wrong_count.value}")
    with pytest.raises(AssertionError) as wrong_hour:
        _assert_the_bound_is_a_reporting_bound(item_own_clause, 256,
                                               where="config.yaml")
    assert OLD_HOUR_CLAIM in str(wrong_hour.value), (
        f"the hour-of-traffic calibration got past the relation check: "
        f"{wrong_hour.value}")

    for path in QUEUE_MAX_COMMENTS:
        _assert_a_dated_probe(_queue_max_comment(path), where=path.name)
        _assert_the_seam_count_and_its_owner(_queue_max_comment(path),
                                             where=path.name)


def test_the_seam_section_states_the_row_stream_it_left_behind():
    """§6.1 has to carry the SHAPE of the stream, not its readings, because
    "dark" is only checkable against something and a typed count is not: the
    numbers a reader needs are what `djev_status`'s `shadow.seam_log` reports per
    seam off `shadow.jsonl` itself, and that route can be re-measured while a
    figure in prose cannot.

    The node used to assert the literal `2026-09-21T08:57:40`, which inverted the
    guard: it passed on a doc asserting 34 rows and a newest row that had since
    moved (a `rerank` row landed 2026-10-11T17:44:09Z, so the rollback window
    reopened after the stated end), and it would have failed an honest correction.
    So what stays pinned is what cannot go stale — that the seam is unreachable
    rather than idle, what would lift it, and the pointer to the live source."""
    text = DOC.read_text()
    section = text[text.index("### 6.1"):text.index("### 6.2")]
    assert "structurally dark" in section, (
        "§6.1 no longer says the seam is unreachable rather than idle")
    assert "kill switch" in section, "§6.1 no longer names what would lift it"
    assert "shadow.seam_log" in section, (
        "§6.1 lost the pointer to the live row stream, and with it the only way a "
        "reader can check what §6.1 says about rows — every count typed here is a "
        "dated probe, and an undated one reads as a present state")
    assert "djev_status" in section, (
        "the pointer has to name the route that answers, not just the key inside it")
    probe = "probe 2026-09-23"
    assert probe in section.lower(), (
        f"{probe!r} no longer labels the figures in §6.1, so the 34/145/46 "
        "readings are back to being present-state claims a rerank row can falsify")


def test_the_shadow_hook_says_it_runs_only_under_the_other_ranker():
    """`vault._djev_shadow_rerank` is the one function a reader of the dispatch
    lands on, so its docstring carries the firing condition."""
    import inspect
    doc = inspect.getsource(vault._djev_shadow_rerank)
    assert "qmd" in doc, "the hook's docstring no longer names the ranker it needs"
    assert "djev is the recall's ranker" in doc or "djev ranks" in doc, (
        "the hook's docstring no longer says it is dark while djev ranks")


def test_the_stats_docstring_names_only_a_reader_that_exists():
    """`stats()` says the tool route is its only production reader, and used to
    name `/state` as a second one that never read it — the small version of this
    item's whole shape, a description carrying a consumer nobody wired up. So
    the sentence is pinned: add a reader and the test tells you to say so."""
    import inspect

    from app import djev_shadow
    doc = inspect.getsource(djev_shadow.stats)
    assert "djev_status" in doc, "stats() no longer names the route that reads it"
    assert "/state" not in doc, (
        "stats() advertises /state as a reader again. Grep for its call sites "
        "before adding a consumer back — the claim, not the consumer, is what "
        "has to be true here.")
    assert "djev_shadow.stats()" in (ROOT / "agent_mcp" / "djev.py").read_text(), (
        "the named reader stopped calling stats(), so the docstring is now the "
        "false half")
    readers = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "app").rglob("*.py")
                     if p.name != "djev_shadow.py" and "djev_shadow.stats()" in p.read_text())
    assert readers == [], (
        f"stats() gained a reader under app/ that its docstring does not name: {readers}")


def test_the_rerank_arm_says_it_applies_only_under_qmd_ranking():
    """Clause 2, first half, on the code side: the arm's own docstring and the
    note a caller gets must both name the ranker that lets it run."""
    import inspect
    assert "qmd" in inspect.getsource(vault._djev_rerank_pool), (
        "`_djev_rerank_pool` no longer says it is reached only when qmd ranks")
    assert "qmd" in vault.RECALL_ARM_UNUSED_NOTE, (
        "the unused-knob note no longer tells the caller which ranker would "
        "have applied its knob")


def test_the_recall_schema_does_not_advertise_the_arm_knobs():
    """Clause 2, schema half, as it stands after `d1b2f8e` took the seven eval
    knobs out of the `vault_recall` schema: an undeclared knob is now reported
    as stripped rather than silently obeyed, and the schema itself offers
    neither arm knob. If one is re-added it has to carry the condition in its
    own description — a parameter whose description promises an effect the
    dispatch cannot produce is how this item got filed."""
    import asyncio

    tool = next(t for t in asyncio.run(vault.list_tools()) if t.name == "vault_recall")
    props = tool.input_schema["properties"]
    for knob in ("djev_rerank", "djev_rerank_top"):
        assert knob not in props, (
            f"{knob} is advertised as a `vault_recall` parameter again, and its "
            "description cannot promise an effect the dispatch refuses whenever "
            "djev is the ranker — which is the sentence #1372 was filed from. "
            "It belongs in RECALL_EVAL_KNOBS, where a client that sends it gets "
            "told the knob was stripped.")
        assert knob in vault.RECALL_EVAL_KNOBS, (
            f"{knob} left RECALL_EVAL_KNOBS, so call_tool would let a client set it")


# ── #1944: the P10 ruling, in two files, one number ─────────────────────────
#
# The module docstring and `architecture/harness.md` P10 are read by different
# audiences and were both edited in the same round, so the failure mode worth
# pinning is not "the doc is missing a sentence" but "the two disagree about the
# rate" — the item itself carried a per-day figure 1.7x too low because someone
# divided by the file's span instead of the seam's, and the corrected number only
# survives if a reader is forced to notice when the two halves drift.
AR_PY = ROOT / "app" / "harness" / "action_review.py"
HARNESS_DOC = ROOT / "architecture" / "harness.md"

#: The seam's own first row. Anything that dates the corpus earlier than this is
#: quoting a row from a different seam (the file's head is a `rerank` row).
SEAM_FIRST = "2026-09-25T18:45"
#: Unrelated positives per day, from the 2026-10-01 snapshot the ruling rests on.
PER_DAY = "180.5"


def _p10(text: str) -> str:
    start = text.index("### P10 —")
    end = text.index("### D5 —", start)
    return text[start:end]


def test_the_p10_ruling_and_the_module_agree_on_the_rate_and_the_branch():
    """Clause 4: the tree says what the report says, in both places, with the
    same per-day figure and the same branch."""
    mod = AR_PY.read_text(encoding="utf-8")
    doc = _p10(HARNESS_DOC.read_text(encoding="utf-8"))
    assert PER_DAY in mod, (
        f"the module no longer quotes {PER_DAY}/day, so the P10 ruling and the "
        "code disagree about the volume that closed the question")
    assert f"**{PER_DAY} / day**" in doc, (
        f"`harness.md` P10 no longer quotes {PER_DAY}/day in its own table. It "
        "must be the same figure the module docstring quotes — the whole point of "
        "the ruling is that one measurement produced both.")
    assert "shadow permanently" in mod, "the module stopped naming the branch"
    assert "shadow permanently" in doc, "P10 stopped naming the branch"
    enum = re.search(r"harness\.action_review\.mode:\s*([\w|]+)", doc)
    assert enum and set(enum.group(1).split("|")) == {"off", "shadow"}, (
        f"P10 documents the reviewer's mode enum as {enum and enum.group(1)!r} "
        "while `MODES` in the code has no `warn` — the doc telling someone to set "
        "a mode the code will not accept is the drift this pins")


#: The write-up clause 3 asks for. It lives in the vault because every other
#: knowledge note does; the point of pinning it from here is that the prose and
#: the code are not allowed to drift apart.
MEASUREMENT_NOTE = (Path.home() / "obsidian" / "knowledge" / "ai" /
                    "action-review-threshold-measurement.md")

#: What clause 3 enumerates, in the exact words the note's own figure list uses.
#: A figure the note drops stops being a figure the ruling rests on, and the
#: sentence that fails says which one went.
NOTE_FIGURES = {
    "the branch": "Branch **(b)**",
    "positives over the corpus, per day, over the seam's own span":
        "**963** of **57,325** rows = **180.5/day**",
    "the span itself": "**5.335 days**",
    "the seam's first row, not the file's": "**2026-09-25T18:45:16Z**",
    "the would-fire rate": "**929** positives that ran = **174.1/day**",
    "the tier split": "**961** at tier 1",
    "the tier >= 2 rate": "**0.2/day**",
    "the injected count": "argmax in **0** of 57,325 rows",
    "the hook-denial overlap": "**35 `consistent` + 1 `unrelated`**",
    "the transcript window": "archived at **30 days**",
    "the corpus's own retention": "nothing rotates `shadow.jsonl`",
}


def test_the_measurement_note_states_every_figure_the_ruling_rests_on():
    """Clause 3: the measurement is written up WITH the counts that produced it.

    Six surfaces can state a ruling; only one of them is the write-up, and a
    grader that found the figures in `harness.md` and the module docstring
    correctly declined to credit the note for them. So this node opens the note
    and asks it for each figure by name — branch, per-day rate over the seam's
    own span, the injected count, the hook-denial overlap with its true counts,
    and the retention window the run sat inside — and then checks that the note
    and the tree quote the SAME per-day figure, since one measurement produced
    both. `tests/test_action_review_calibration.py::
    test_the_ruling_reproduces_from_the_committed_extract_with_no_transcripts`
    is what pins those same numbers against the bytes; this one pins them
    against the prose.
    """
    note = MEASUREMENT_NOTE.read_text(encoding="utf-8")
    assert "action_review" in note
    # A figure wrapped across two lines of markdown is the same figure, so the
    # note is matched with its wrapping collapsed and its bold markers intact.
    flat = " ".join(note.split())
    for what, needle in NOTE_FIGURES.items():
        assert needle in flat, (
            f"the write-up no longer states {what} as `{needle}` — clause 3 is "
            "the counts, not the conclusion")
    assert PER_DAY in note, (
        f"the note and `harness.md` P10 must quote the same per-day rate "
        f"({PER_DAY}); one measurement produced both")
    # The item's headline error, kept unrepeatable in prose too: 2026-09-21 is the
    # FILE's head row, not this seam's, and every legitimate mention of it says so.
    for line in note.splitlines():
        if "2026-09-21" in line:
            assert ("rerank" in line or "file" in line.lower()
                    or "missing" in line), (
                "the note dates a row of THIS seam to 2026-09-21, which is the "
                "file's head row and five days before the seam existed — the exact "
                "reading that produced the item's 92/day and the wrong span")


def test_no_claim_dates_the_seam_earlier_than_its_first_row():
    """The item's headline error, made unrepeatable: 92/day came from dividing by
    the file's span, whose first row is a `rerank` row five days before the seam
    existed. The corpus's earliest `action_review` row is 2026-09-25T18:45Z."""
    mod = AR_PY.read_text(encoding="utf-8")
    doc = _p10(HARNESS_DOC.read_text(encoding="utf-8"))
    for name, text in (("action_review.py docstring", mod), ("harness.md P10", doc)):
        assert SEAM_FIRST in text, (
            f"{name} no longer names the seam's own first row ({SEAM_FIRST}), "
            "which is the only anchor that keeps a per-day rate honest")
        early = [d for d in re.findall(r"2026-\d\d-\d\d", text) if d < SEAM_FIRST[:10]]
        assert not early, (
            f"{name} dates something in the seam's history to {early[0]}, before "
            "the first `action_review` row at 2026-09-25 — a row dated earlier "
            "than that belongs to another seam")


# ── #2699: the roster has one owner, and this page restates it twice ─────────

#: `SEAMS` in `app/djev_shadow.py` is the only place the roster is written — the
#: recorder iterates it, `stats()` reports per it, §9.4 counts its gated subset,
#: and §9.3 sends a contributor to it — but this page states the set twice more in
#: prose: a count in the frontmatter `summary:`, and the names in the opening
#: diagram. Neither was graded by anything. `grep -c SEAMS` on this file read 0
#: while five modules read `SEAMS` for behaviour, so the doc agreed with its owner
#: by luck. That is the state that reproduces the defect by itself:
#: `architecture/index.md` carried "three shadow seams" — the number from the
#: commit that first wired three, until a review deleted the integer,
#: and §9.3 step 2 never said the page also lists the set, so a contributor
#: following the procedure made it stale in two fresh places.
_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

#: A number counts the seams only when the noun stands beside it, and the number
#: may be a figure or a word — the corpus writes owned counts both ways. Tab and
#: space only, never a newline: a count and its noun are one phrase, not two lines
#: of prose that happen to end and begin near each other.
_SEAM_COUNT = re.compile(
    r"\b(?:(\d{1,2})|(" + "|".join(_NUMBER_WORDS) + r"))"
    r"[ \t]+shadow[ \t]+seams?\b", re.IGNORECASE)

#: A bare lower identifier — a diagram token that can be a seam name, as opposed
#: to the prose continuations that share the visual line (`(the`, `model's`).
_BARE_NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")


def _doc_text() -> str:
    """The doc, read through the module global so a control can aim it at a copy."""
    return DOC.read_text(encoding="utf-8")


def _fenced(text: str) -> str:
    """The concatenation of every ``` fenced block: the diagrams, not the prose.

    The roster is drawn in a diagram, and a node that scanned the whole file would
    meet the §6.1 heading and §9.3's step text as well, neither of which is the
    copy under test here.
    """
    return "\n".join(text.split("```")[1::2])


def _frontmatter_summary(text: str) -> str:
    """The frontmatter's `summary:` value, with its wrapped lines folded together.

    Folding is what makes the number findable: the shipped value breaks mid-list,
    so a line-scoped reader sees half a sentence and reports a clean sweep of a doc
    it never parsed.
    """
    front = text.split("---", 2)[1]
    started = re.search(r"^summary:[ \t]*", front, re.M)
    assert started, (
        "the doc's frontmatter has no `summary:` key, so the copy of the roster "
        "this node exists to grade has moved rather than been fixed")
    rest = front[started.end():]
    stop = re.search(r"^[A-Za-z_][\w-]*:", rest, re.M)
    return " ".join((rest[:stop.start()] if stop else rest).split())


def _summary_seam_counts(text: str) -> list[int]:
    """Every count the `summary:` attaches to the words `shadow seams`.

    An empty list is a passing answer, not a vacuous one: a page may name the set
    without counting it, which is what `architecture/index.md`'s Covers row does
    now, and a count that drifts can never be.
    """
    return [int(digits) if digits else _NUMBER_WORDS[word.lower()]
            for digits, word in _SEAM_COUNT.findall(_frontmatter_summary(text))]


def _roster_line(text: str, seams) -> str:
    """The fenced line that lists the seam names, found by the roster itself.

    Located through the owner rather than a literal so the line is still found after
    a seam lands or leaves: it is the line carrying the most seam names. A roster
    split across two lines then reports its missing half loudly instead of agreeing
    with a page that stopped listing.
    """
    names = set(seams)
    lines = [ln for ln in _fenced(text).splitlines() if ln.strip()]
    return max(lines, key=lambda ln: sum(1 for t in ln.split() if t in names),
               default="")


def _diagram_roster(text: str, seams) -> list[str]:
    """The seam names the diagram lists, in the order it lists them.

    Read as the run of bare identifiers from the first seam name onward, because
    the same visual line also carries prose from the columns to its left. An empty
    answer means the doc names no seam at all, which the caller reads as every seam
    missing — never as a comparison with nothing to compare.
    """
    names = set(seams)
    tokens = _roster_line(text, seams).split()
    first = next((i for i, t in enumerate(tokens) if t in names), None)
    assert first is not None, (
        "the diagram no longer names any seam, so the roster this page draws has "
        f"gone: none of {sorted(names)} appears on a fenced line. §9.3 sends a new "
        "seam to the diagram; a page listing none is not a page in agreement with "
        "`SEAMS`")
    run = []
    for token in tokens[first:]:
        if not _BARE_NAME.match(token):
            break
        run.append(token)
    return run


def test_the_doc_names_exactly_the_seams_the_shadow_recorder_declares():
    """Clauses 1 and 2: both copies on this page are the owner's set, not a memory.

    The roster is `djev_shadow.SEAMS`, so it is read and never re-typed — nothing
    below holds a seam name or a count literal. Two shapes of drift are refused. A
    count that is not `len(SEAMS)` is the defect #2699 was filed on, and a page may
    also name the set with no count at all, which is the fix `index.md` took, so an
    empty count list passes here on purpose. A name the diagram dropped, a name it
    invented, and a name it lists twice are each refused by name, because the
    diagram is the copy a reader copies the roster FROM.
    """
    from app import djev_shadow

    seams = tuple(djev_shadow.SEAMS)
    text = _doc_text()

    counts = _summary_seam_counts(text)
    stale = [c for c in counts if c != len(seams)]
    assert not stale, (
        f"the `summary:` counts the shadow seams {stale} while `SEAMS` holds "
        f"{len(seams)}: {seams}. Either agree or drop the number — the Covers row in "
        "`architecture/index.md` names the set with no integer and needs nobody to "
        "keep it true")

    roster = _diagram_roster(text, seams)
    missing = sorted(set(seams) - set(roster))
    extra = sorted(set(roster) - set(seams))
    twice = sorted({t for t in roster if roster.count(t) > 1})
    assert not (missing or extra or twice), (
        f"the diagram's seam roster {roster} is not the roster the recorder declares: "
        f"missing={missing} undocumented={extra} listed_twice={twice}. §9.3 step 2 "
        "sends a new seam to the diagram as well as to `SEAMS`")


def test_the_seam_roster_check_can_see_a_doc_that_lost_a_seam(tmp_path, monkeypatch):
    """Clause 3's control: the green above has to mean agreement, not an empty scan.

    Three seeds, each aimed at a half that could go vacuous. The summary counts one
    seam more than the owner has — built from `_NUMBER_WORDS` and the match itself,
    so the seed is derived and survives the shipped number changing. The diagram
    loses a seam, taken as `SEAMS[0]` rather than spelled, and the check has to name
    it. And the roster is widened by one name with the doc left exactly as shipped,
    which is what a real fifth seam looks like on the morning before anybody draws
    it: `index.md` sat in exactly that state for the fourth seam's whole life.
    """
    import sys

    from app import djev_shadow

    seams = tuple(djev_shadow.SEAMS)
    shipped = _doc_text()
    here = sys.modules[__name__]

    matched = _SEAM_COUNT.search(shipped)
    assert matched, (
        "the shipped summary names no seam count, so the count half of the check has "
        "nothing to seed and its green is unearned")
    wider_word = next(w for w, n in _NUMBER_WORDS.items() if n == len(seams) + 1)
    bumped = matched.group(0).replace(matched.group(1) or matched.group(2),
                                      wider_word)
    over = tmp_path / "over.md"
    over.write_text(shipped[:matched.start()] + bumped + shipped[matched.end():],
                    encoding="utf-8")
    monkeypatch.setattr(here, "DOC", over)
    seeded = _summary_seam_counts(_doc_text())
    assert seeded == [len(seams) + 1], (
        f"a summary seeded to count {wider_word} shadow seams read as {seeded!r}: "
        "the count in the frontmatter is not being read at all")

    dropped = seams[0]
    line = _roster_line(shipped, seams)
    thinned = tmp_path / "thinned.md"
    thinned.write_text(shipped.replace(line, line.replace(dropped, "", 1), 1),
                       encoding="utf-8")
    monkeypatch.setattr(here, "DOC", thinned)
    lost = set(seams) - set(_diagram_roster(_doc_text(), seams))
    assert lost == {dropped}, (
        f"a diagram seeded without `{dropped}` read as {lost!r}: the roster in the "
        "diagram is not being read")

    widened = seams + (f"seam_{len(seams) + 1}",)
    assert set(widened) - set(_diagram_roster(shipped, widened)) == {widened[-1]}, (
        "a roster widened by one name with the doc unchanged went unreported, so a "
        "real fifth seam would land green")
    assert _summary_seam_counts(shipped) != [len(widened)], (
        "the shipped count matched a roster the owner does not have")


#: What §9.3 step 2 must name, as the step words them: the two places on this page a
#: new seam has to be written into. Fragments rather than a whole sentence, so the
#: pin survives rewording and dies on losing the instruction.
STEP2_PLACES = ("summary:", "diagram")


def _procedure(text: str) -> str:
    """§9.3 "Adding a seam", up to §9.4 — the procedure, not the chapter.

    Bounded on both sides because `tests/conftest.py` cites §9.3 by step NUMBER, so a
    step inserted above the one it names silently re-points that citation at different
    instructions.
    """
    return _section(text, "### 9.3 Adding a seam").split("\n### 9.4")[0]


def _step(text: str, number: int) -> str:
    """One numbered step, cut at the next one — so a neighbour cannot vouch for it."""
    started = re.search(rf"^{number}[.] ", text, re.M)
    assert started, f"the procedure has no step {number}: {text[:80]!r}"
    rest = text[started.end():]
    nxt = re.search(rf"^{number + 1}[.] ", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def test_the_seam_procedure_sends_a_new_seam_to_the_page_as_well_as_the_code():
    """Clause 4: the procedure is why the defect recurs, so the procedure gets pinned.

    Step 2 always said to add the name to `djev_shadow.SEAMS`; it never said this page
    lists the set too, which is how a contributor who followed §9.3 exactly still left
    the count and the diagram behind — the shape #2699 records. The pin is that the two
    places are NAMED, plus the test that enforces them, so the instruction cannot be
    shortened back to its code-only half unnoticed. The step's LENGTH is not pinned and
    its neighbours are not renumbered: conftest cites step 5, and step 5 is still step
    5.
    """
    step = _step(_procedure(_doc_text()), 2)
    assert "`djev_shadow.SEAMS`" in step, (
        f"step 2 no longer sends a new seam to `SEAMS` at all: {step!r}")
    for place in STEP2_PLACES:
        assert place in step, (
            f"step 2 no longer names {place!r}, one of the two places on this page a "
            "new seam must be written into. Losing it is how #2699 happened: the step "
            f"read {step!r}")
    assert "test_djev_doc_claims.py" in step, (
        "step 2 no longer says which test enforces the page's two copies, so the next "
        f"contributor has no way to find the red they just caused: {step!r}")


def test_the_index_parity_check_still_never_reads_a_rosters_owner():
    """Clause 5's boundary, pinned from this file rather than by editing #2700's.

    `tests/test_architecture_index_parity.py` owns `_OWNED_COUNT` and #2700 is widening
    it to number words; two rounds editing that constant would collide, and nothing
    here touches it. What is asserted instead is the reason this node cannot live
    there: that check counts words in `index.md` and never reads a code owner, so it
    can never answer "does the page still match `SEAMS`". And `index.md`'s live rows
    must not regain a count of their own — the integer there was deleted by the
    `arch-review: doc:index — stale` commit, whose own Review-log quotation of "three
    shadow seams" is the record of doing it, and that file's verbatim log test keeps
    the quotation in place.
    """
    parity = (ROOT / "tests" / "test_architecture_index_parity.py").read_text(
        encoding="utf-8")
    assert "import djev_shadow" not in parity, (
        "the index parity check now reads the seam owner, which is this node's job; "
        "two places reading one owner is how this round and #2700 collide")

    index = (ROOT / "architecture" / "index.md").read_text(encoding="utf-8")
    live = index.split("## Review log")[0]
    hit = _SEAM_COUNT.search(live)
    assert hit is None, (
        f"index.md's live text counts the shadow seams again "
        f"({hit.group(0)!r}) after the doc:index review deleted it; an unowned count "
        "is what this item is about — name the set instead")
    assert "three shadow seams" in index, (
        "the Review-log entry recording that deletion has gone, and "
        "`test_architecture_index_parity.py` forbids rewriting the log")


# ── #3114: §6.2 may not say a dedupe row's null is unreadable ────────────────
#
# §6.2 is what #3114 was filed FROM: it recorded that a `dedupe` shadow row
# cannot tell "never eligible to merge" from "the merge was declined", called
# that unrecoverable, and named the two item numbers that owed the fix. The
# writer now passes `judged`, `spawn_write`, `forced` and the normalized `tags`
# into `meta`, so the paragraph is not free to keep the conclusion — an open gap
# left in a page's live text is a to-do list a later reader re-files, which is
# exactly how #934 came back as #3114. Same shape as `test_the_fixture_claim_
# matches_the_fixture`, one heading over: the prose must state what runs.

#: The keys §6.2 states as recorded, backticked exactly as the section writes
#: them. `name` and `candidates` belong to the set the section names, and the
#: whole set is what `tests/test_djev_shadow_isolation.py::
#: test_an_eligible_spawn_create_records_why_its_merged_into_is_null` asserts
#: against the row, so a key added or renamed in the writer fails here too.
DEDUPE_META_KEYS = ("`name`", "`candidates`", "`judged`", "`spawn_write`",
                    "`forced`", "`tags`")

#: The conclusion, and the measurement it was drawn from. Both are strings a
#: reader can grep to check this node, which is why they are needles and not
#: para-phrases. The wrap is the doc's own: the shipped paragraph broke "cannot
#: be" from "recovered" across a line.
STALE_DEDUPE_CLAIMS = ("the row does not record", "cannot be recovered",
                       "cannot be\nrecovered", "('candidates', 'name')")

#: An item number in a section's live text reads as an open gap. The Review log
#: at the foot of the page keeps both numbers as the record of when they were
#: filed and closed; §11's fixture entry above is the same failure, one heading
#: over, with the same remedy.
OPEN_ITEM_REFS = ("#934", "#3114")


def _dedupe_section(text: str) -> str:
    """§6.2 alone, bounded on BOTH sides.

    Sliced rather than searched whole-page: §6.3 is a different seam, so a
    "`judged`" satisfied by its prose would be a green earned from the wrong
    section — and §6.2 is the only paragraph this item was filed from.
    """
    started = text.find("### 6.2")
    assert started >= 0, (
        "architecture/djev.md has no §6.2 for the dedupe seam any more, so the "
        "prose this node exists to check is not where the item left it")
    end = text.find("### 6.3", started)
    assert end > started, "§6.2 no longer ends where §6.3 begins, so this is not one section"
    return text[started:end]


def _dedupe_problems(text: str) -> list[str]:
    """Every way §6.2 can be wrong about what a `dedupe` row records. Returned,
    not asserted, so the seed below can be run through the same code."""
    section = _dedupe_section(text)
    problems = [f"§6.2 no longer names {k} as recorded in `meta`"
                for k in DEDUPE_META_KEYS if k not in section]
    problems += [f"§6.2 still says {needle!r}" for needle in STALE_DEDUPE_CLAIMS
                 if needle in section]
    problems += [f"§6.2 still points at {ref} as though the gap were open"
                 for ref in OPEN_ITEM_REFS if ref in section]
    return problems


def test_section_6_2_states_the_eligibility_the_dedupe_row_records():
    """Clause 4: §6.2 names `judged`, `spawn_write`, `forced` and `tags` as
    recorded in `meta`, and has neither the "the row does not record which it
    was" sentence, the `('candidates', 'name')` measurement it came from, nor the
    sentence pointing at #934/#3114 as an open gap.

    The keys are then checked against the writer rather than against another
    paragraph, in this file's tradition of reading words out of files: `meta`
    must contain each key the prose names, and `_dedupe` must pass it in — a page
    that names a key the hook never receives is the claim that made this item.
    """
    problems = _dedupe_problems(_doc_text())
    assert not problems, "; ".join(problems)

    import inspect

    from agent_mcp import backlog

    hook = inspect.getsource(backlog._djev_shadow_dedupe)
    caller = inspect.getsource(backlog._dedupe)
    for key in ("judged", "spawn_write", "forced", "tags"):
        assert f'"{key}"' in hook, (
            f"§6.2 says the row records `{key}` and the hook never puts it in `meta`")
        assert f"{key}={key}" in caller, (
            f"`_dedupe` does not pass `{key}`, so the value in the row is a default "
            "or an accident — which is the defect, renamed")


def test_the_dedupe_prose_check_refuses_the_paragraph_it_exists_to_refuse():
    """Red side, for the same reason every other node here runs it: the check is
    not a word list copied off the shipped page. The seed below is §6.2 as it
    stood before this change, word for word, and it is refused on every count the
    clause names. The green half then rides the shipped file."""
    stale = (
        "### 6.2 Backlog dedupe — `agent_mcp/backlog.py::_dedupe`\n\n"
        "The hook sits at the `backlog_write_task` caller that holds `name` and\n"
        "`description`, **not** at `merge_target`. That function is handed rows of\n"
        "`{id, title, status, score, lexical, shared, created, source}` and no body text\n"
        "at all, so the one thing a language model needs to answer the question is the\n"
        "one thing it has never seen. It asks one `same_finding` `choice` per candidate\n"
        "(the top three) in a single canvas. The candidate heads are read by the worker.\n"
        "`actual` is what production did (`merged_into`, `rule`). The shadow fires for\n"
        "every create that has candidates, including human, `force`, `blocker` and\n"
        "`umbrella` writes that were never eligible to merge. For those, `merged_into:\n"
        "null` means \"not eligible\", not \"merge declined\", and the row does not record\n"
        "which it was — and unlike the first draft of this section, that bit cannot be\n"
        "recovered afterwards: `meta` carries `name` and `candidates` and nothing else\n"
        "(measured 2026-09-21 — every `dedupe` row in the live log has the meta key set\n"
        "`('candidates', 'name')`), and the tags and `force` flag that decided\n"
        "`mergeable` — the flag is `judged` at `backlog.py:613-616` today, and `forced`\n"
        "sits beside it at `:613` — are never passed to the hook, though all three are in\n"
        "scope at its call site (`:617`). Filed as #934, and still open there as of\n"
        "2026-10-21: `_djev_shadow_dedupe(args, similar, target, cfg)` takes none of them,\n"
        "so re-filed as #3114.\n"
        "\n### 6.3 Entity SAME/DIFFERENT\n")

    problems = _dedupe_problems(stale)
    assert problems, (
        "the check accepted the paragraph it was written to refuse, so it grades nothing")
    for fragment in ("the row does not record", "('candidates', 'name')", "#934", "#3114"):
        assert any(fragment in p for p in problems), (
            f"the seed carried {fragment!r} and the check did not report it")
    assert any("`spawn_write`" in p for p in problems), (
        "a §6.2 that never names the key is not the §6.2 the clause asks for")

    assert _dedupe_problems(_doc_text()) == [], "the shipped section fails the check it defines"


# ---------------------------------------------------------------------------
# #3115 — what this page may say about the cutoff `djev_rank` judged rows by
#
# Same shape as everything above: the doc is where the item came from (§11
# carried the fixed 0.5 as an open gap), so a §11 that still carries it after
# the fix is a to-do list that re-files a closed item, and a §5.1 that names a
# number without its authority is the sentence that made the annotation read as
# calibrated. The needles are therefore the key names and values the shipped
# tool emits, read out of `agent_mcp/djev.py` rather than typed in here: if the
# code renames a key, this fails and the doc has to move with it.
# ---------------------------------------------------------------------------

RANK_CLOSED_HEADING = ("### Closed 2026-10-21 — `djev_rank`'s fixed 0.5 "
                       "cutoff (#3115)")


def _rank_return_bullet(text: str) -> str:
    """§5.1's bullet describing `djev_rank`'s returned body, up to the sample
    call that follows it."""
    i = text.index("- It returns `ranked:")
    j = text.index("\n```", i)
    return text[i:j]


def test_section_5_1_names_the_cutoff_and_the_state_that_authorises_it():
    """Clause 5, first half. The bullet used to end "`low_label_mass_indexes`:
    the indexes under 0.5" — a number with no stated authority, which is
    exactly how a leftover default reads as a judgement.
    """
    from agent_mcp import djev as tools

    bullet = _rank_return_bullet(DOC.read_text())
    for needle in ("label_mass_floor", "low_label_mass_cutoff",
                   "low_label_mass_cutoff_source", "low_trust",
                   tools.CUTOFF_UNCALIBRATED_FALLBACK, tools.CUTOFF_FROM_SCHEMA):
        assert needle in bullet, (
            f"§5.1 no longer tells a reader about {needle!r}, which the shipped "
            "tool does emit — or the tool renamed it and this page drifted")
    assert "the indexes under 0.5" not in bullet, (
        "§5.1 is back to quoting the bare 0.5 as if it were the rule")
    # The two signals are different things and the page has to say so: with no
    # measured floor `low_trust` is false on every row while the fallback list
    # can still fire, and a reader who conflates them learns the row is fine.
    assert "must not be confused" in bullet


def test_the_open_gaps_no_longer_list_the_rank_cutoff_and_the_record_owns_it():
    """Clause 5, second half: §11 stops listing it as live, and the closed
    record carries what a later reader cannot re-derive."""
    text = DOC.read_text()
    body = _section(text, "## 11. Known gaps")
    assert RANK_CLOSED_HEADING in body, (
        "§11 has no closed record for #3115, so the fixed 0.5 has no owner here "
        "even once the code is fixed — and an open-gap list with no entry for a "
        "shipped decision is the state that re-files the item")
    open_part = body.split(CLOSED_HEADING, 1)[0]
    assert "low_label_mass_indexes" not in open_part, (
        "§11 still lists the rank tool's fixed 0.5 as an open gap")
    assert "#3115" not in open_part, (
        "the open gaps still point at the item that closed it")

    record = body.split(RANK_CLOSED_HEADING, 1)[1]
    for needle in ('schemas.get("rerank")',          # the lookup, by name
                   'floor_for("tool:rank")',          # and the one rejected…
                   "BY_SEAM",                         # …because this has no such key
                   "still `None`",                    # what is NOT fixed
                   "replay.py --floors",              # who does set it
                   "0.446",                           # the spread that forbids one
                   "advisory",                        # and it still refuses nothing
                   "app/djev.py:307",                 # the 0.5 deliberately left alone
                   "git grep"):                       # how to re-check it
        assert needle in record, (
            f"the #3115 closed record no longer carries {needle!r}")
