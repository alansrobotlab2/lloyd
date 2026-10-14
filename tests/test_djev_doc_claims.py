"""What the djev prose is allowed to say about the seams it describes.

Two items live here, both because a description was read as a fact. The first
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
# The item was filed from prose, not from code. `app/djev_shadow.py` opened with
# "Three seams call this — reranking in `agent_mcp/vault.py`", `config.yaml`
# still calls it "The lead seam: both `vault_recall` and `memory_ops.recall`
# pass through it", and `architecture/djev.md` says the seam runs "on every
# recall". Since #1336 none of those is true: djev IS the recall's ranker, so
# the hook sits under an `elif` the first arm never reaches, and `rerank`'s last
# production row is 2026-09-21T08:57:40Z while `dedupe` and `entity` keep
# flowing. A description that presents a structurally dark seam as a live one is
# what makes its silence unreadable — so the descriptions are asserted, not just
# corrected once.

SHADOW_SRC = ROOT / "app" / "djev_shadow.py"
VAULT_SRC = ROOT / "agent_mcp" / "vault.py"


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


#: Every file that describes the seams: the recorder, the caller, and the doc
#: the #1324 record above is guarded in. #1372 is a defect about descriptions,
#: so the scan covers all three rather than only the two the clause names.
SEAM_DESCRIPTIONS = [SHADOW_SRC, VAULT_SRC, DOC]


@pytest.mark.parametrize("path", SEAM_DESCRIPTIONS,
                         ids=["djev_shadow", "vault", "djev.md"])
def test_no_djev_description_calls_rerank_a_seam_production_reaches(path):
    """Clause 2, second half. "Lead seam" was the phrase in two of these files —
    `app/djev_shadow.py`'s eval-muting paragraph and
    `vault._djev_shadow_rerank`'s own docstring ("The lead seam of the three,
    because every `vault_recall` passes through it") — and "on every recall" was
    in the doc, two sections from the one that knew better (§8.1: "The shadow
    seam does not run when djev ranks"). Those are the claims that make a dark
    seam read as a quiet one, so the needles are asserted, not just corrected
    once: a rank or a frequency is where a stale seam description shows up."""
    text = path.read_text().lower()
    needles = ["lead seam", "the three shadow seams"]
    if path != VAULT_SRC:
        # `vault.py` is 2,000 lines and says "on every recall" about the #1335
        # latency of the graph arm, which is true of it; the frequency claim
        # that matters there is the hook's own, checked just below.
        needles.append("on every recall")
    for needle in needles:
        assert needle not in text, (
            f"{path.name} still says {needle!r} about the rerank seam. While "
            "djev is the recall's ranker the dispatch never reaches it, so state "
            "the condition it fires under instead of a volume or a rank.")
    if path is VAULT_SRC:
        import inspect
        hook = inspect.getsource(vault._djev_shadow_rerank).lower()
        for needle in ("on every recall", "passes through it", "unconditional"):
            assert needle not in hook, (
                f"`_djev_shadow_rerank` still claims it happens {needle!r}, "
                "which is the sentence that made the seam look live")


def test_the_seam_section_states_the_row_stream_it_left_behind():
    """The doc has to carry the measured end of the stream, because "dark" is
    only checkable against something: the newest `rerank` row is a fact a reader
    can re-measure in `shadow.jsonl`, and the day it stopped is the day #1336
    landed."""
    text = DOC.read_text()
    section = text[text.index("### 6.1"):text.index("### 6.2")]
    assert "structurally dark" in section, (
        "§6.1 no longer says the seam is unreachable rather than idle")
    assert "2026-09-21T08:57:40" in section, (
        "§6.1 lost the newest-rerank-row timestamp, which is the sentence that "
        "makes 'dark' a claim someone else can re-measure")
    assert "kill switch" in section, "§6.1 no longer names what would lift it"


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
